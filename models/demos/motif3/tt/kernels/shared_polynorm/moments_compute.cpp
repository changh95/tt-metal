// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused shared-expert PolyNorm (B5), moments program, compute.
//
// The release (tt/polynorm.py _moment_sums, decode: one tile row, concat_full) computes, per chip,
//   g2 = multiply(g, g, fp32); g4 = multiply(g2, g2); g6 = multiply(g4, g2)      (binary_ng SFPU mul, fp32 DEST)
//   s  = sum(concat([g2, g4, g6], dim 1), dim -1)                                (accurate fp32 reduce: per row of
//        tiles, acc = x_0, acc = add_binary_tile(acc, x_j) for j = 1..n-1, then sfpu_reduce REDUCE_ROW)
// This kernel issues exactly those LLK calls, in that order and with the same operand order, so the moment sums are
// bitwise the release's (no fused multiply-add can appear: every product and sum is its own SFPU tile op).
// DEST (fp32, full sync, 8 tiles): 0 / 1 / 2 = running sums of g^2 / g^4 / g^6, 3 = g_j, 4 / 5 / 6 = g_j^2 / ^4 / ^6.
// Output: 3 fp32 tiles (moment m in column 0 of tile m) -> cb_part.
//
// CT: 0 cb_g, 1 cb_part, 2 n

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_g = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);
    constexpr uint32_t n = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(cb_g, cb_part);
    cb_wait_front(cb_g, n);
    cb_reserve_back(cb_part, 3);
    tile_regs_acquire();
    for (uint32_t j = 0; j < n; ++j) {
        copy_init(cb_g);
        copy_tile(cb_g, j, 3);
        mul_binary_tile_init();
        if (j == 0) {
            mul_binary_tile(3, 3, 0);  // g^2 = g g
            mul_binary_tile(0, 0, 1);  // g^4 = g^2 g^2
            mul_binary_tile(1, 0, 2);  // g^6 = g^4 g^2
        } else {
            mul_binary_tile(3, 3, 4);
            mul_binary_tile(4, 4, 5);
            mul_binary_tile(5, 4, 6);
            add_binary_tile_init();
            add_binary_tile(0, 4, 0);  // acc + x (the reduce's fold operand order)
            add_binary_tile(1, 5, 1);
            add_binary_tile(2, 6, 2);
        }
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0, 1, 1);
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(1, 1, 1);
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(2, 1, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_part);
    pack_tile(1, cb_part);
    pack_tile(2, cb_part);
    tile_regs_release();
    cb_push_back(cb_part, 3);
    cb_pop_front(cb_g, n);
}
