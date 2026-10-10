// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused shared-expert tail (Phase F, F3; tt/kernels/shared_tail.py), compute of the DOWN cores.
//
// One DOWN core owns NPER output tiles of y = h @ W_down ([1, 1, 32, KT 32] bf16 @ [KT 32, N] bfp8 -> bf16). It issues
// the stock 1D-mcast linear's LLK sequence (bmm_large_block_zm_fused_bias_activation, the shared down config
// in0_block_w = KT = the whole K in one block, out_subblock_w = per_core_N = NPER, out_subblock_h = 1): hw startup with
// SrcOrder::Reverse, matmul_block_init(in0, in1, 0, NPER, 1, KT), then for k = 0..KT-1 matmul_block(in0 k, in1 k NPER)
// accumulating in the fp32 DEST, packed once to bf16. One block: no spill / reload, so the output is the stock op's bit
// for bit (same fidelity / DEST mode: the "shared" compute role).
//
// CT: 0 cb_h, 1 cb_w, 2 cb_out, 3 KT, 4 NPER

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_h = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t KT = get_compile_time_arg_val(3);
    constexpr uint32_t NPER = get_compile_time_arg_val(4);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_h, cb_w, cb_out);
    matmul_block_init(cb_h, cb_w, 0, NPER, 1, KT);
    cb_wait_front(cb_w, KT * NPER);
    cb_wait_front(cb_h, KT);
    tile_regs_acquire();
    for (uint32_t k = 0; k < KT; ++k) {
        matmul_block(cb_h, cb_w, k, k * NPER, 0, 0, NPER, 1, KT);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, NPER);
    tile_regs_wait();
    for (uint32_t i = 0; i < NPER; ++i) {
        pack_tile(i, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, NPER);
    cb_pop_front(cb_w, KT * NPER);
    cb_pop_front(cb_h, KT);
}
