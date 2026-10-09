// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 decode routed experts: dual-NoC all-core sparse expert matmul (phase D, DESIGN-3 stage 1), compute.
// Per work unit (expert j, column col): DEST[mt] = sum_{k = 0..KT-1, in order} in0[mt, k] x W[k, col] with the fp32 DEST
// accumulating over the whole K (half 0 from cb_in0a / cb_wa, then half 1 from cb_in0b / cb_wb), packed once to cb_out.
// The stock 1D-mcast op accumulates the same products in the same order, spilling the fp32 DEST to a Float32 CB and
// reloading it with UnpackToDestFp32 between K blocks (lossless), so the results are expected to be bitwise equal.
// The same init (hw startup with SrcOrder::Reverse, matmul_block_init / matmul_block with rt_dim = ct_dim = 1).
//
// CT: 0 KT, 1 NT, 2 MT, 3 NC, 4 W0, 5 GX, 6 BATCH, 7 IN0_MODE, 8 cb_in0a, 9 cb_in0b, 10 cb_wa, 11 cb_wb, 12 cb_out,
//     13 cb_ctrl

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"


// Unit order of worker w (identical in the dataflow and compute kernels; host mirror: moe_sparse_mm.unit_order).
// MODE 0 (gate_up, in0 shared): round-robin u = w + i NC (consecutive cores read consecutive columns, so the
//   bank-affine column streams spread over all DRAM banks at every step).
// MODE 1 (down, in0 per expert): the contiguous range [w U / NC, (w + 1) U / NC) (<= 2 experts), each expert's
//   sub-range [s, s + n) rotated to start at s + (w - s) mod n (decorrelates the cores' banks; one in0 load per expert).
struct UnitOrder {
    uint32_t n = 0, n1 = 0, s1 = 0, r1 = 0, s2 = 0, n2 = 0, r2 = 0, w = 0, nc = 1, mode = 0;
    void init(uint32_t mode_, uint32_t w_, uint32_t nc_, uint32_t U, uint32_t NT) {
        mode = mode_;
        w = w_;
        nc = nc_;
        if (mode == 0) {
            n = U > w ? (U - w - 1) / nc + 1 : 0;
            return;
        }
        const uint32_t u0 = w * U / nc, u1 = (w + 1) * U / nc;
        n = u1 > u0 ? u1 - u0 : 0;
        if (n == 0) {
            return;
        }
        const uint32_t b = (u0 / NT + 1) * NT;  // the next expert boundary
        s1 = u0;
        n1 = (b < u1 ? b : u1) - u0;
        r1 = (w + n1 * NT - s1 % n1) % n1;
        s2 = b;
        n2 = n - n1;
        r2 = n2 ? (w + n2 * NT - s2 % n2) % n2 : 0;
    }
    uint32_t at(uint32_t i) const {
        if (mode == 0) {
            return w + i * nc;
        }
        if (i < n1) {
            return s1 + (r1 + i) % n1;
        }
        return s2 + (r2 + i - n1) % n2;
    }
};

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t MT = get_compile_time_arg_val(2);
    constexpr uint32_t NC = get_compile_time_arg_val(3);
    constexpr uint32_t W0 = get_compile_time_arg_val(4);
    constexpr uint32_t GX = get_compile_time_arg_val(5);
    constexpr uint32_t BATCH = get_compile_time_arg_val(6);
    constexpr uint32_t IN0_MODE = get_compile_time_arg_val(7);
    constexpr uint32_t cb_in0a = get_compile_time_arg_val(8);
    constexpr uint32_t cb_in0b = get_compile_time_arg_val(9);
    constexpr uint32_t cb_wa = get_compile_time_arg_val(10);
    constexpr uint32_t cb_wb = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out = get_compile_time_arg_val(12);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(13);
    constexpr uint32_t KH = KT / 2;
    constexpr uint32_t SEG = KH * MT;

    cb_wait_front(cb_ctrl, 1);
    const uint32_t k = read_tile_value(cb_ctrl, 0, 0);
    cb_pop_front(cb_ctrl, 1);

    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    if (c < W0 || c >= W0 + NC) {
        return;
    }
    const uint32_t U = k * NT;
    UnitOrder ord;
    ord.init(IN0_MODE, c - W0, NC, U, NT);
    if (ord.n == 0) {
        return;
    }

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0a, cb_wa, cb_out);
    matmul_block_init(cb_in0a, cb_wa, 0, 1, 1, BATCH);

    uint32_t cur_j = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < ord.n; ++i) {
        const uint32_t u = ord.at(i);
        const uint32_t j = u / NT;
        if (j != cur_j) {
            if constexpr (IN0_MODE == 1) {
                if (cur_j != 0xFFFFFFFFu) {
                    cb_pop_front(cb_in0a, SEG);
                    cb_pop_front(cb_in0b, SEG);
                }
                cb_wait_front(cb_in0a, SEG);
                cb_wait_front(cb_in0b, SEG);
            } else if (cur_j == 0xFFFFFFFFu) {
                cb_wait_front(cb_in0a, SEG);
                cb_wait_front(cb_in0b, SEG);
            }
            cur_j = j;
        }
        tile_regs_acquire();
        for (uint32_t b = 0; b < KH / BATCH; ++b) {
            cb_wait_front(cb_wa, BATCH);
            for (uint32_t i = 0; i < BATCH; ++i) {
                const uint32_t kk = b * BATCH + i;
                for (uint32_t mt = 0; mt < MT; ++mt) {
                    matmul_block(cb_in0a, cb_wa, kk * MT + mt, i, mt, 0, 1, 1, BATCH);
                }
            }
            cb_pop_front(cb_wa, BATCH);
        }
        for (uint32_t b = 0; b < KH / BATCH; ++b) {
            cb_wait_front(cb_wb, BATCH);
            for (uint32_t i = 0; i < BATCH; ++i) {
                const uint32_t kk = b * BATCH + i;
                for (uint32_t mt = 0; mt < MT; ++mt) {
                    matmul_block(cb_in0b, cb_wb, kk * MT + mt, i, mt, 0, 1, 1, BATCH);
                }
            }
            cb_pop_front(cb_wb, BATCH);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, MT);
        tile_regs_wait();
        for (uint32_t mt = 0; mt < MT; ++mt) {
            pack_tile(mt, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, MT);
    }
    cb_pop_front(cb_in0a, SEG);
    cb_pop_front(cb_in0b, SEG);
}
