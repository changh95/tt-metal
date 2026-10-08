// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 decode MoE combine (D4): R + fold(S), compute. One FPU add per tile, the LLK sequence of ttnn.add on two
// bf16 TILE tensors (binary_ng eltwise_binary_no_bcast: add_tiles, fp32 dest acc off, default unpack), so the sum is
// bitwise ttnn.add(R, fold(S)).
//
// CT: 0 cb_r, 1 cb_s, 2 cb_out, 3 NO, 4 PER, 5 GX

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_r = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t NO = get_compile_time_arg_val(3);
    constexpr uint32_t PER = get_compile_time_arg_val(4);
    constexpr uint32_t GX = get_compile_time_arg_val(5);

    const uint32_t w = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t o0 = w * PER;
    const uint32_t o1 = (o0 + PER < NO) ? o0 + PER : NO;
    if (o0 >= o1) {
        return;
    }
    compute_kernel_hw_startup(cb_r, cb_s, cb_out);
    add_tiles_init(cb_r, cb_s);
    for (uint32_t o = o0; o < o1; ++o) {
        cb_wait_front(cb_r, 1);
        cb_wait_front(cb_s, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        add_tiles(cb_r, cb_s, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_r, 1);
        cb_pop_front(cb_s, 1);
    }
}
