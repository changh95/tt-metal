// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode attention combine (D1, docs/OPTIMIZATION_PLAN.md §3.3 C3), compute.
//
// Per output tile, the LLK sequence of the three ttnn ops it replaces, in their order, with their configuration
// (bf16 operands, fp32_dest_acc_en = false, HiFi4, math approx off, half-sync DEST; default unpack for the addcmul
// operands, UnpackToDestFp32 for the binary_ng SFPU operands CB_D / CB_G / CB_ACT / CB_DG, as binary_ng sets them):
//   1. ttnn.addcmul(u_sig, v, u_noise, value=-1.0)   ternary_addc_ops_sfpu.cpp: copy a / b / c to DEST 0 / 1 / 2,
//      addcmul_tile<Float16_b>(0, 1, 2, 0, bits(-1.0f)) (one SFPU FMA, RNE to bf16), pack -> CB_D
//   2. ttnn.multiply(d, g)                            binary_ng eltwise_binary_sfpu_no_bcast.cpp (MUL is an SFPU op
//      unless fast_and_approximate_mode): copy d / g to DEST 0 / 1, mul_binary_tile(0, 1, 0), pack -> CB_DG
//   3. ttnn.where(active, dg, 0.0)                    binary_ng eltwise_where_no_bcast.cpp (WHERE_TTS): copy cond /
//      tensor to DEST 0 / 1, fill DEST 2 with the scalar bits, where_tile<Float16_b>(0, 1, 2, 0), pack -> CB_OUT
// The intermediates go through bf16 CBs exactly as they go through bf16 DRAM tensors in the op chain, so each stage
// sees bit-identical operands.
//
// CT: 0 cb_sig, 1 cb_v, 2 cb_noise, 3 cb_g, 4 cb_act, 5 cb_d, 6 cb_dg, 7 cb_out, 8 n, 9 PER, 10 GX, 11 value bits,
//     12 debug stage (0: production; 1: write d; 2: write dg -- diagnostics only)

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_sig = get_compile_time_arg_val(0);
    constexpr uint32_t cb_v = get_compile_time_arg_val(1);
    constexpr uint32_t cb_noise = get_compile_time_arg_val(2);
    constexpr uint32_t cb_g = get_compile_time_arg_val(3);
    constexpr uint32_t cb_act = get_compile_time_arg_val(4);
    constexpr uint32_t cb_d = get_compile_time_arg_val(5);
    constexpr uint32_t cb_dg = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t n = get_compile_time_arg_val(8);
    constexpr uint32_t PER = get_compile_time_arg_val(9);
    constexpr uint32_t GX = get_compile_time_arg_val(10);
    constexpr uint32_t VALUE = get_compile_time_arg_val(11);
    constexpr uint32_t DEBUG = get_compile_time_arg_val(12);
    constexpr uint32_t cb_d_out = DEBUG == 1 ? cb_out : cb_d;
    constexpr uint32_t cb_dg_out = DEBUG == 2 ? cb_out : cb_dg;

    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t j0 = q * PER;
    const uint32_t j1 = (j0 + PER < n) ? j0 + PER : n;
    if (j0 >= j1) {
        return;
    }

    compute_kernel_hw_startup(cb_sig, cb_v, cb_out);

    for (uint32_t j = j0; j < j1; ++j) {
        // ---- 1. d = u_sig + (-1) v u_noise (ternary addcmul, SFPU) ----------------------------------------------
        cb_wait_front(cb_sig, 1);
        cb_wait_front(cb_v, 1);
        cb_wait_front(cb_noise, 1);
        cb_reserve_back(cb_d_out, 1);
        tile_regs_acquire();
        copy_init(cb_sig);
        copy_tile(cb_sig, 0, 0);
        copy_init(cb_v);
        copy_tile(cb_v, 0, 1);
        copy_init(cb_noise);
        copy_tile(cb_noise, 0, 2);
        addcmul_tile_init();
        addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, VALUE);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_d_out);
        tile_regs_release();
        cb_push_back(cb_d_out, 1);
        cb_pop_front(cb_sig, 1);
        cb_pop_front(cb_v, 1);
        cb_pop_front(cb_noise, 1);
        if constexpr (DEBUG == 1) {
            cb_wait_front(cb_g, 1);
            cb_pop_front(cb_g, 1);
            cb_wait_front(cb_act, 1);
            cb_pop_front(cb_act, 1);
            continue;
        }

        // ---- 2. dg = d * g (binary_ng SFPU multiply) ------------------------------------------------------------
        cb_wait_front(cb_d, 1);
        cb_wait_front(cb_g, 1);
        cb_reserve_back(cb_dg_out, 1);
        tile_regs_acquire();
        copy_init(cb_d);
        copy_tile(cb_d, 0, 0);
        reconfig_data_format_srca(cb_d, cb_g);
        copy_init(cb_g);
        copy_tile(cb_g, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        reconfig_data_format_srca(cb_g, cb_d);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_dg_out);
        tile_regs_release();
        cb_push_back(cb_dg_out, 1);
        cb_pop_front(cb_d, 1);
        cb_pop_front(cb_g, 1);
        if constexpr (DEBUG == 2) {
            cb_wait_front(cb_act, 1);
            cb_pop_front(cb_act, 1);
            continue;
        }

        // ---- 3. out = where(active, dg, 0.0) (binary_ng WHERE_TTS, SFPU) ----------------------------------------
        cb_wait_front(cb_act, 1);
        cb_wait_front(cb_dg, 1);
        cb_reserve_back(cb_out, 1);
        tile_regs_acquire();
        copy_init(cb_act);
        copy_tile(cb_act, 0, 0);
        copy_init(cb_dg);
        copy_tile(cb_dg, 0, 1);
        fill_tile_init();
        fill_tile_bitcast(2, 0u);
        where_tile_init();
        where_tile<DataFormat::Float16_b>(0, 1, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_act, 1);
        cb_pop_front(cb_dg, 1);
    }
}
