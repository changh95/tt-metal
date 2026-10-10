// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode attention output chain (Phase F, F2; tt/kernels/attn_out.py), compute.
//
// Each stage issues the LLK sequence of the ttnn op it replaces, with that op's configuration and CB data formats,
// so every output equals the op chain bit for bit:
//   ROLE 0 (HiFi4, fp32 DEST, approx off, default unpack: the attn_heads role):
//     UV: one output tile of the per-head W_UV bmm (MatmulMultiCoreReuseProgramConfig in0_block_w 4: the reuse
//       factory's bmm_large_block_zm kernel): K = 16 in 4 blocks of 4; between blocks the fp32 partial is packed to
//       cb_part (fp32) and reloaded with copy_tile through SrcA (the factory leaves the partials CB UnpackToSrc, so
//       the reload rounds to TF32 exactly as the stock op does); the last block packs bf16.
//     WO (FUSED): the 2 output tiles of the 1D-mcast wo linear (auto config: in0_block_w 1; its fp32 partial reload is
//       UnpackToDestFp32, lossless): K = 32 in order in the fp32 DEST, ct_dim 2, packed bf16.
//   ROLE 1 (CMB: the D1 combine configuration: HiFi4, fp32 DEST off, approx off; UnpackToDestFp32 on cb_g / cb_act /
//     cb_d / cb_dg):
//     v = sigmoid(lam @ E_j): the matmul (E has one 1 per column: the value is lam exactly, as in the stock fp32-DEST
//       op), packed bf16, then the stock bf16 unary sigmoid (fp32 DEST off there too); then the D1 sequence
//       (attn_combine/compute.cpp): addcmul(u_sig, v, u_noise, -1) -> * g -> where(active, ., 0).
//
// CT: 0 ROLE, 1 FUSED, 2 DEBUG, 3 VALUE bits (addcmul -1.0)

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#include "common.h"

using namespace motif_aout;

#ifndef MOTIF_AOUT_EXP
#define MOTIF_AOUT_EXP 0  // experiments only: 16 = no wo matmuls (WO packs DEST as is)
#endif

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t FUSED = get_compile_time_arg_val(1);
    constexpr uint32_t DEBUG = get_compile_time_arg_val(2);
    constexpr uint32_t VALUE = get_compile_time_arg_val(3);

    if constexpr (ROLE == 1) {
        // ---- v = sigmoid(lam @ E_j) --------------------------------------------------------------------------------
        compute_kernel_hw_startup<SrcOrder::Reverse>(CB_LAM, CB_E, CB_VPRE);
        matmul_block_init(CB_LAM, CB_E, 0, 1, 1, 1);
        cb_wait_front(CB_LAM, NLAMT);
        cb_wait_front(CB_E, NLAMT);
        tile_regs_acquire();
        for (uint32_t k = 0; k < NLAMT; ++k) {
            matmul_block(CB_LAM, CB_E, k, k, 0, 0, 1, 1, 1);
        }
        tile_regs_commit();
        cb_reserve_back(CB_VPRE, 1);
        tile_regs_wait();
        pack_tile(0, CB_VPRE);
        tile_regs_release();
        cb_push_back(CB_VPRE, 1);
        cb_pop_front(CB_LAM, NLAMT);
        cb_pop_front(CB_E, NLAMT);

        reconfig_data_format_srca(CB_E, CB_VPRE);
        pack_reconfig_data_format(CB_V);
        copy_init(CB_VPRE);
        cb_wait_front(CB_VPRE, 1);
        cb_reserve_back(CB_V, 1);
        if constexpr (DEBUG) {
            cb_reserve_back(CB_VDBG, 1);
        }
        tile_regs_acquire();
        copy_tile(CB_VPRE, 0, 0);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::RC, false, false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_V);
        if constexpr (DEBUG) {
            pack_tile(0, CB_VDBG);
        }
        tile_regs_release();
        cb_push_back(CB_V, 1);
        if constexpr (DEBUG) {
            cb_push_back(CB_VDBG, 1);
        }
        cb_pop_front(CB_VPRE, 1);

        // ---- 1. d = u_sig + (-1) v u_noise (ternary addcmul, SFPU) ------------------------------------------------
        cb_wait_front(CB_SIG, 1);
        cb_wait_front(CB_V, 1);
        cb_wait_front(CB_NOISE, 1);
        cb_reserve_back(CB_D, 1);
        reconfig_data_format_srca(CB_VPRE, CB_SIG);
        pack_reconfig_data_format(CB_D);
        tile_regs_acquire();
        copy_init(CB_SIG);
        copy_tile(CB_SIG, 0, 0);
        copy_init(CB_V);
        copy_tile(CB_V, 0, 1);
        copy_init(CB_NOISE);
        copy_tile(CB_NOISE, 0, 2);
        addcmul_tile_init();
        addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, VALUE);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_D);
        tile_regs_release();
        cb_push_back(CB_D, 1);
        cb_pop_front(CB_SIG, 1);
        cb_pop_front(CB_V, 1);
        cb_pop_front(CB_NOISE, 1);

        // ---- 2. dg = d * g (binary_ng SFPU multiply) ----------------------------------------------------------------
        cb_wait_front(CB_D, 1);
        cb_wait_front(CB_G, 1);
        cb_reserve_back(CB_DG, 1);
        tile_regs_acquire();
        copy_init(CB_D);
        copy_tile(CB_D, 0, 0);
        reconfig_data_format_srca(CB_D, CB_G);
        copy_init(CB_G);
        copy_tile(CB_G, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        reconfig_data_format_srca(CB_G, CB_D);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_DG);
        tile_regs_release();
        cb_push_back(CB_DG, 1);
        cb_pop_front(CB_D, 1);
        cb_pop_front(CB_G, 1);

        // ---- 3. out = where(active, dg, 0.0) (binary_ng WHERE_TTS, SFPU) ------------------------------------------
        cb_wait_front(CB_ACT, 1);
        cb_wait_front(CB_DG, 1);
        cb_reserve_back(CB_OUTC, 1);
        tile_regs_acquire();
        copy_init(CB_ACT);
        copy_tile(CB_ACT, 0, 0);
        copy_init(CB_DG);
        copy_tile(CB_DG, 0, 1);
        fill_tile_init();
        fill_tile_bitcast(2, 0u);
        where_tile_init();
        where_tile<DataFormat::Float16_b>(0, 1, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUTC);
        tile_regs_release();
        cb_push_back(CB_OUTC, 1);
        cb_pop_front(CB_ACT, 1);
        cb_pop_front(CB_DG, 1);
        return;
    }

    // ---- ROLE 0 ------------------------------------------------------------------------------------------------------
    const uint32_t y = static_cast<uint32_t>(get_absolute_logical_y());
    const bool uv = y < NUV / UV_GX;
    bool started = false;

    if (uv) {  // W_UV: K = 16 in 4 blocks of 4 (the reuse bmm's spill / TF32 reload between blocks)
        compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IN0, CB_WA, CB_PART);
        started = true;
        matmul_block_init(CB_IN0, CB_WA, 0, 1, 1, UV_IBW);
        cb_wait_front(CB_IN0, KUV);
        constexpr uint32_t NB = KUV / UV_IBW;
        for (uint32_t b = 0; b < NB; ++b) {
            const uint32_t cw = b < NB / 2 ? CB_WA : CB_WB;
            const uint32_t o = (b % (NB / 2)) * UV_IBW;
            if (b == 0) {
                cb_wait_front(CB_WA, KH);
            } else if (b == NB / 2) {
                cb_wait_front(CB_WB, KH);
            }
            tile_regs_acquire();
            if (b > 0) {  // reload_from_dfb_to_dst
                reconfig_data_format_srca(cw, CB_PART);
                copy_init(CB_PART);
                cb_wait_front(CB_PART, 1);
                copy_tile(CB_PART, 0, 0);
                cb_pop_front(CB_PART, 1);
                reconfig_data_format_srca(CB_PART, cw);
                matmul_block_init(CB_IN0, cw, 0, 1, 1, UV_IBW);
            }
            for (uint32_t i = 0; i < UV_IBW; ++i) {
                matmul_block(CB_IN0, cw, b * UV_IBW + i, o + i, 0, 0, 1, 1, UV_IBW);
            }
            tile_regs_commit();
            if (b + 1 < NB) {
                cb_reserve_back(CB_PART, 1);
                tile_regs_wait();
                pack_tile(0, CB_PART);
                tile_regs_release();
                cb_push_back(CB_PART, 1);
            } else {
                cb_reserve_back(CB_U, 1);
                tile_regs_wait();
                pack_reconfig_data_format(CB_U);
                pack_tile(0, CB_U);
                tile_regs_release();
                cb_push_back(CB_U, 1);
            }
        }
        cb_pop_front(CB_WA, KH);
        cb_pop_front(CB_WB, KH);
        cb_pop_front(CB_IN0, KUV);
    }

    if constexpr (FUSED) {  // wo: 2 output tiles, K = 32 in order in the fp32 DEST
        if (!started) {
            compute_kernel_hw_startup<SrcOrder::Reverse>(CB_DGIN, CB_WOA, CB_WOUT);
        } else {
            reconfig_data_format(CB_WB, CB_WOA, CB_IN0, CB_DGIN);
            pack_reconfig_data_format(CB_WOUT);
        }
        matmul_block_init(CB_DGIN, CB_WOA, 0, NWO_PER, 1, 1);
        cb_wait_front(CB_DGIN, KWO);
        tile_regs_acquire();
        for (uint32_t c = 0; c < NWCH; ++c) {  // chunk c: k 4c .. 4c + 3 from cb_woa (0..3) / cb_wob (4, 5) / cb_woc (6, 7)
            const uint32_t cw = c < 4 ? CB_WOA : (c < 6 ? CB_WOB : CB_WOC);
            const uint32_t cl = c < 4 ? c : (c < 6 ? c - 4 : c - 6);  // chunk index within its cb
            cb_wait_front(cw, (cl + 1) * WO_CHUNK * NWO_PER);
            for (uint32_t kk = 0; kk < ((MOTIF_AOUT_EXP & 16) ? 0u : WO_CHUNK); ++kk) {
                matmul_block(CB_DGIN, cw, c * WO_CHUNK + kk, (cl * WO_CHUNK + kk) * NWO_PER, 0, 0, NWO_PER, 1, 1);
            }
        }
        tile_regs_commit();
        cb_reserve_back(CB_WOUT, NWO_PER);
        tile_regs_wait();
        for (uint32_t i = 0; i < NWO_PER; ++i) {
            pack_tile(i, CB_WOUT);
        }
        tile_regs_release();
        cb_push_back(CB_WOUT, NWO_PER);
        cb_pop_front(CB_WOA, 4 * WO_CHUNK * NWO_PER);
        cb_pop_front(CB_WOB, 2 * WO_CHUNK * NWO_PER);
        cb_pop_front(CB_WOC, 2 * WO_CHUNK * NWO_PER);
        cb_pop_front(CB_DGIN, KWO);
    }
}
