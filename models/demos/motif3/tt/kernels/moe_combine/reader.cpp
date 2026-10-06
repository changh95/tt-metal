// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 compacted prefill MoE: gather combine (B2b), reader (RISCV_1).
//
//   y    [1, 1, ROWS, H] bf16 ROW_MAJOR   the expert outputs of the compacted rows (already weighted; page = one row)
//   keys uint32 ROW_MAJOR, page KP        the token of every row (rows sorted by (local expert, token)); >= M: pad
// -> part [1, 1, M, H] (compute + writer): part[t] = sum over t's rows j, in row order, of y[j].
//
// Work unit u = (tile row tr = u / G, column group g = u % G of CW = WT / G column tiles); worker q (row-major over
// GX columns, NW workers) takes the contiguous units [q NU / NW, (q + 1) NU / NW). For each tile row the reader scans
// the keys once and lists each of its 32 tokens' rows in row order (= local expert order: the dense path's summation
// order). L = max(1, longest list). Per unit it sends L (one page of cb_cnt), then for each level k < L one 32-row
// ROW_MAJOR block of CW tiles (cb_rm, CW pages): row r = row lst[r][k] of y, columns [32 c0, 32 (c0 + CW)) (one NoC
// read of CW x 64 B), or zeros when token r has fewer than k + 1 rows. The compute kernel tilizes each level and adds
// the levels into an fp32 dest exactly as fast_reduce_nc adds the 12 expert terms (absent terms are exact zeros either
// way). cb_z: one zero tile, the second operand of the adds.
//
// CT: 0 M, 1 ROWS, 2 WT, 3 G, 4 GX, 5 NW, 6 cb_rm, 7 cb_cnt, 8 cb_z, 9 cb_keys, 10 KP, 11 KPAGE (keys page bytes),
//     12 HB (y row bytes), 13.. TensorAccessorArgs(y), TensorAccessorArgs(keys)
// common RT: 0 y_addr, 1 keys_addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M = get_compile_time_arg_val(0);
    constexpr uint32_t ROWS = get_compile_time_arg_val(1);
    constexpr uint32_t WT = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr uint32_t GX = get_compile_time_arg_val(4);
    constexpr uint32_t NW = get_compile_time_arg_val(5);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(6);
    constexpr uint32_t cb_cnt = get_compile_time_arg_val(7);
    constexpr uint32_t cb_z = get_compile_time_arg_val(8);
    constexpr uint32_t cb_keys = get_compile_time_arg_val(9);
    constexpr uint32_t KP = get_compile_time_arg_val(10);
    constexpr uint32_t KPAGE = get_compile_time_arg_val(11);
    constexpr uint32_t HB = get_compile_time_arg_val(12);
    constexpr auto y_args = TensorAccessorArgs<13>();
    constexpr auto k_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr uint32_t CW = WT / G;
    constexpr uint32_t NU = (M / 32) * G;
    constexpr uint32_t MAXL = 16;  // list slots per token (the wrapper's cb_tl holds top-K <= MAXL levels)
    constexpr uint32_t SEG = CW * 64;  // bytes of one row segment (CW bf16 tiles wide)
    static_assert(WT % G == 0 && ROWS % 32 == 0, "combine shape");
    static_assert(SEG % 512 == 0, "a row segment must be a multiple of 512 B (zero fill)");

    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t u0 = q * NU / NW;
    const uint32_t u1 = (q + 1) * NU / NW;
    const auto ys = TensorAccessor(y_args, get_common_arg_val<uint32_t>(0), HB);
    const auto ks = TensorAccessor(k_args, get_common_arg_val<uint32_t>(1), KPAGE);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);

    // zero tile for the compute kernel's second operand
    cb_reserve_back(cb_z, 1);
    const uint32_t zl = get_write_ptr(cb_z);
    for (uint32_t i = 0; i < 4; ++i) {
        noc_async_read(zeros, zl + i * 512, 512);
    }
    // the keys of every row (one read)
    cb_reserve_back(cb_keys, 1);
    const uint32_t kl = (get_write_ptr(cb_keys) + 63) & ~63u;  // 64 B of slack in the CB
    noc_async_read(ks.get_noc_addr(KP, 0), kl, ROWS * 4);
    noc_async_read_barrier();
    invalidate_l1_cache();
    asm volatile("fence" ::: "memory");
    cb_push_back(cb_z, 1);
    const uint32_t* key = reinterpret_cast<const uint32_t*>(kl);
    uint32_t* lst = reinterpret_cast<uint32_t*>(kl + ROWS * 4);  // [32][MAXL]
    uint32_t n[32];
    uint32_t L = 1;
    uint32_t cur_tr = 0xFFFFFFFFu;

    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t tr = u / G;
        const uint32_t c0 = (u % G) * CW;
        if (tr != cur_tr) {
            for (uint32_t r = 0; r < 32; ++r) {
                n[r] = 0;
            }
            const uint32_t lo = tr * 32;
            L = 1;
            for (uint32_t j = 0; j < ROWS; ++j) {
                const uint32_t t = key[j] - lo;  // unsigned: other tile rows and pad keys (>= M) land >= 32
                if (t < 32) {
                    const uint32_t m = n[t];
                    if (m < MAXL) {
                        lst[t * MAXL + m] = j;
                    }
                    n[t] = m + 1;
                    L = m + 1 > L ? m + 1 : L;
                }
            }
            L = L > MAXL ? MAXL : L;
            cur_tr = tr;
        }
        cb_reserve_back(cb_cnt, 1);
        const uint32_t cp = get_write_ptr(cb_cnt);
        reinterpret_cast<volatile uint32_t*>(cp)[0] = L;
        asm volatile("" ::: "memory");
        cb_push_back(cb_cnt, 1);
        for (uint32_t k = 0; k < L; ++k) {
            cb_reserve_back(cb_rm, CW);
            const uint32_t dst = get_write_ptr(cb_rm);
            for (uint32_t r = 0; r < 32; ++r) {
                if (n[r] > k) {
                    noc_async_read(ys.get_noc_addr(lst[r * MAXL + k], c0 * 64), dst + r * SEG, SEG);
                } else {
                    for (uint32_t b = 0; b < SEG; b += 512) {
                        noc_async_read(zeros, dst + r * SEG + b, 512);
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_rm, CW);
        }
    }
}
