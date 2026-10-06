// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode router tail (B4), compute: biased = scores + expert_bias on the fp32 SFPU, the same
// add_binary_tile<NearestEven> that ttnn.add runs for two FLOAT32 operands (binary_ng), so the selection key of
// every expert is bitwise the release's. One page in (the worker's token row, contiguous), one page out.
//
// CT: 0 cb_s, 1 cb_b, 2 cb_o

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_s = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_o = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(cb_s, cb_o);
    cb_wait_front(cb_s, 1);
    cb_wait_front(cb_b, 1);
    cb_reserve_back(cb_o, 1);
    tile_regs_acquire();
    copy_init(cb_s);
    copy_tile(cb_s, 0, 0);
    copy_init(cb_b);
    copy_tile(cb_b, 0, 1);
    add_binary_tile_init();
    add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_o);
    tile_regs_release();
    cb_push_back(cb_o, 1);
    cb_pop_front(cb_s, 1);
    cb_pop_front(cb_b, 1);
}
