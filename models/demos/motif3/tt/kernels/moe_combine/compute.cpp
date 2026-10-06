// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 compacted prefill MoE: gather combine (B2b), compute. Per unit: tilize each of the L levels (CW tiles, exact
// bf16 copies), then per output tile dest = T_0 + ... + T_{L-1} with add_tiles(cb_tl, cb_z) accumulating into the fp32
// dest -- the ops (add_init with acc_to_dest, add_tiles against a zero tile, pack) of fast_reduce_nc, the dense path's
// expert sum -- and one pack into the output format.
//
// CT: 0 M, 1 WT, 2 G, 3 GX, 4 NW, 5 cb_rm, 6 cb_cnt, 7 cb_z, 8 cb_tl, 9 cb_o, 10 KMAX

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/tilize.h"

void kernel_main() {
    constexpr uint32_t M = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t GX = get_compile_time_arg_val(3);
    constexpr uint32_t NW = get_compile_time_arg_val(4);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(5);
    constexpr uint32_t cb_cnt = get_compile_time_arg_val(6);
    constexpr uint32_t cb_z = get_compile_time_arg_val(7);
    constexpr uint32_t cb_tl = get_compile_time_arg_val(8);
    constexpr uint32_t cb_o = get_compile_time_arg_val(9);
    constexpr uint32_t KMAX = get_compile_time_arg_val(10);  // levels cb_tl holds (top-K: a token has <= K rows)
    constexpr uint32_t CW = WT / G;
    constexpr uint32_t NU = (M / 32) * G;

    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t u0 = q * NU / NW;
    const uint32_t u1 = (q + 1) * NU / NW;
    compute_kernel_hw_startup(cb_rm, cb_z, cb_o);
    cb_wait_front(cb_z, 1);
    for (uint32_t u = u0; u < u1; ++u) {
        cb_wait_front(cb_cnt, 1);
        const uint32_t L = read_tile_value(cb_cnt, 0, 0);
        cb_pop_front(cb_cnt, 1);
        tilize_init(cb_rm, CW, cb_tl);
        for (uint32_t k = 0; k < L; ++k) {
            cb_wait_front(cb_rm, CW);
            cb_reserve_back(cb_tl, CW);
            tilize_block(cb_rm, CW, cb_tl);
            cb_push_back(cb_tl, CW);
            cb_pop_front(cb_rm, CW);
        }
        tilize_uninit(cb_rm, cb_tl);
        // cb_tl holds exactly KMAX levels: push the unused ones too, so every unit's tiles start at the buffer's base
        // (indexed reads of a CB do not wrap around its end)
        if (L < KMAX) {
            cb_reserve_back(cb_tl, (KMAX - L) * CW);
            cb_push_back(cb_tl, (KMAX - L) * CW);
        }
        cb_wait_front(cb_tl, KMAX * CW);
        for (uint32_t c = 0; c < CW; ++c) {
            add_init(cb_tl, cb_z, true);
            reconfig_data_format(cb_tl, cb_z);
            tile_regs_acquire();
            for (uint32_t k = 0; k < L; ++k) {
                add_tiles(cb_tl, cb_z, k * CW + c, 0, 0);
            }
            tile_regs_commit();
            cb_reserve_back(cb_o, 1);
            pack_reconfig_data_format(cb_o);
            tile_regs_wait();
            pack_tile(0, cb_o);
            tile_regs_release();
            cb_push_back(cb_o, 1);
        }
        cb_pop_front(cb_tl, KMAX * CW);
        pack_reconfig_data_format(cb_tl);
    }
    cb_pop_front(cb_z, 1);
}
