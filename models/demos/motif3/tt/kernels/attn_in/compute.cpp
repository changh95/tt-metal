// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode attention input chain (Phase F, F1; tt/kernels/attn_in.py), compute.
//
// Each stage issues the LLK sequence of the ttnn op it replaces, with that op's configuration (HiFi4, fp32 DEST, math
// approx off) and the same CB data formats, so every output equals the op chain bit for bit:
//   QN (ROLE 1): ttnn.rms_norm(cq fp32 [32, 1024], no gamma): the RMSNORM path of the stock layernorm.cpp (square,
//       row_wise_mean with the reduce scaler, + eps, rsqrt, x * col-bcast), interm fp32, block size 4 -> cb_cqn fp32.
//   KN (ROLE 2): ttnn.rms_norm(c_raw bf16 [32, 512]) -> bf16, then rotary_embedding_hf on the 2 k_pe tiles (the stock
//       rotary_embedding_hf.cpp sequence: -1 * rotated (first half), * sin, x * cos, add; bf16 interms) -> cb_kvrow.
//   main (ROLE 0):
//     QB unit: one output tile of the 1D-mcast linear q_b / gate: full K (32) accumulated in order in the fp32 DEST
//       (the stock op spills / reloads the fp32 DEST losslessly between in0 blocks), packed bf16.
//       gate: then the stock unary sigmoid (bf16 in / out: fp32 DEST off there, so sigmoid_tile<RC, approx off,
//       is_fp32_dest_acc_en = false>: the bf16-accurate exp / 1 Newton step and an RNE round to bf16 in the SFPU; the
//       value is bf16-exact in our fp32 DEST, so the pack is exact).
//       pe: rotary_embedding_hf of tile p (p = 0: rotated = -partner; p = 1: rotated = partner).
//     UK unit: one output tile of the per-head W_UK bmm (K = 4, one block), packed bf16.
//
// CT: 0 ROLE, 1 GX, 2 Wt (norm), 3 W (norm logical width), 4 LAT, 5 QB0, 6 UK0
//   LAT (main, c < 52): one output tile of the latent 1D-mcast linear (x @ Wq_lat -> fp32, x @ Wkv_lat -> bf16), K = 128
//       in order in the fp32 DEST.

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

#include "common.h"

namespace ckl = compute_kernel_lib;
namespace generic = norm::kernel_util::generic;
namespace numeric = norm::kernel_util::compute::numeric;
namespace policies = norm::kernel_util::compute::policies;

using namespace motif_ain;

#ifndef MOTIF_AIN_EXP
#define MOTIF_AIN_EXP 0  // experiments only: bit 2 = QN skips the norm (pushes cb_cqn unwritten)
#endif

struct FusedActivation : ckl::UnaryOp<FusedActivation, ckl::Dst::D0> {
    static ALWI void init() {}
    static ALWI void exec_impl([[maybe_unused]] uint32_t i) {}
};

// The RMSNORM (no pre-add, no gamma / beta) path of ttnn layernorm.cpp for one tile row: x in cb_in (Wt tiles,
// pushed by the reader in any multiple of the block), interm fp32 cb_xmm2 / cb_ex2 / cb_ex2pe, reduce scaler and
// eps from the reader, the normalized row pushed to cb_out per block of 4.
template <uint32_t cb_in, uint32_t cb_out, uint32_t Wt, uint32_t W>
ALWI void rms_norm_row() {
    constexpr uint32_t block_size = 4;  // fp32_dest_acc_en
    DataflowBuffer dfb_in(cb_in);
    DataflowBuffer& dfb_xmm = dfb_in;
    DataflowBuffer dfb_eps(CB_EPS);
    DataflowBuffer dfb_ex2pe(CB_EX2PE);
    DataflowBuffer dfb_scaler(CB_RSCAL);
    DataflowBuffer dfb_xmm2(CB_XMM2);
    DataflowBuffer dfb_ex2(CB_EX2);

    compute_kernel_hw_startup(cb_in, cb_in, CB_XMM2);
    dfb_eps.wait_front(1);

    const auto total_buffer_size = generic::blocks(Wt, block_size).total_with_remainder();
    constexpr auto row_shape = ckl::IterationShape::tiles(Wt).block_size(block_size, ckl::BlockTailSync::FullBlock);

    reconfig_data_format(cb_in, cb_in);
    pack_reconfig_data_format(CB_XMM2);

    ckl::square<
        ckl::input(
            cb_in,
            ckl::WaitPolicy::Cumulative,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Block,
            ckl::DataFormatReconfig::Disabled),
        ckl::output(
            CB_XMM2, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize, ckl::DataFormatReconfig::Disabled)>(
        row_shape);
    reconfig_data_format(cb_in, CB_XMM2, cb_in, CB_RSCAL);

    numeric::row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, true, policies::FullBlockWithPopPolicy>(
        dfb_xmm2, dfb_scaler, dfb_ex2, W, Wt, block_size, 32);

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Add,
            ckl::input(CB_EX2),
            ckl::input(CB_EPS, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
        ckl::Rsqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
        ckl::PackTile<ckl::output(CB_EX2PE)>{});

    for (auto block : generic::blocks(Wt, block_size)) {
        const auto block_shape =
            ckl::IterationShape::tiles(block.size()).block_size(block.full_block_size(), ckl::BlockTailSync::FullBlock);
        ckl::eltwise_chain(
            block_shape,
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(
                    cb_in,
                    ckl::WaitPolicy::Upfront,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Block,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileAddressing::Offset),
                ckl::input(CB_EX2PE, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None)>{
                block.start(), 0u},
            ckl::Optional<false, FusedActivation>{},
            ckl::PackTile<ckl::output(cb_out, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
    }
    dfb_ex2pe.pop_front(1);
    dfb_xmm.pop_front(total_buffer_size);
    dfb_scaler.pop_front(1);
}

// rotary_embedding_hf.cpp for one tile j of a 2-tile head: out = x_j * cos_j + rot_j * sin_j with rot_0 = -x_1
// (scalar -1 broadcast multiply into cb_rot) and rot_1 = x_0; bf16 interms. Every input is consumed (popped).
template <uint32_t cb_x, uint32_t cb_r, uint32_t cb_cos, uint32_t cb_sin, uint32_t cb_out>
ALWI void rope_tile(bool first_half) {
    if (first_half) {
        ckl::mul<
            ckl::input(cb_r),
            ckl::input(CB_SCAL, ckl::BroadcastDim::Scalar, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::output(CB_ROT)>(ckl::IterationShape::tiles(1));
        ckl::mul<ckl::input(CB_ROT), ckl::input(cb_sin), ckl::output(CB_SI)>(ckl::IterationShape::one_tile());
    } else {
        ckl::mul<ckl::input(cb_r), ckl::input(cb_sin), ckl::output(CB_SI)>(ckl::IterationShape::one_tile());
    }
    ckl::mul<ckl::input(cb_x), ckl::input(cb_cos), ckl::output(CB_CI)>(ckl::IterationShape::one_tile());
    ckl::add<ckl::input(CB_CI), ckl::input(CB_SI), ckl::output(cb_out)>(ckl::IterationShape::tiles(1));
}

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t GX = get_compile_time_arg_val(1);
    constexpr uint32_t NWt = get_compile_time_arg_val(2);
    constexpr uint32_t NW = get_compile_time_arg_val(3);

    if constexpr (ROLE == 1) {  // QN
        if constexpr ((MOTIF_AIN_EXP & 4) != 0) {
            for (uint32_t b = 0; b < KQ / 4; ++b) {
                cb_reserve_back(CB_CQN, 4);
                cb_push_back(CB_CQN, 4);
            }
            return;
        }
        rms_norm_row<CB_QX, CB_CQN, NWt, NW>();
        return;
    }
    if constexpr (ROLE == 2) {  // KN
        rms_norm_row<CB_KIN, CB_KVROW, NWt, NW>();
        cb_wait_front(CB_SCAL, 1);
        reconfig_data_format(CB_ROTIN, CB_SCAL);
        pack_reconfig_data_format(CB_ROT);
        for (uint32_t j = 0; j < 2; ++j) {
            rope_tile<CB_KPEX, CB_ROTIN, CB_COS, CB_SIN, CB_KVROW>(j == 0);
        }
        return;
    }

    // ---- main grid ------------------------------------------------------------------------------------------------
    constexpr uint32_t LAT = get_compile_time_arg_val(4);
    constexpr uint32_t QB0 = get_compile_time_arg_val(5);
    constexpr uint32_t UK0 = get_compile_time_arg_val(6);
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const bool lat = LAT && c < NLAT;
    const bool lat_q = lat && c < NLQ;
    const bool qb = c >= QB0 && c < QB0 + NQB;
    const uint32_t u = c - QB0;
    const bool is_gate = qb && u >= NQ;
    const uint32_t qk = u % HT;
    const bool is_nope = qb && !is_gate && qk < NOPE_T;
    const bool is_pe = qb && !is_gate && qk >= NOPE_T;
    const bool uk = c >= UK0 && c < UK0 + NUK;
    bool started = false;

    // ---- latent unit: one output tile of the 1D-mcast linear x @ Wq_lat (fp32 out) / x @ Wkv_lat (bf16 out), K = 128
    // in order in the fp32 DEST; weight batches alternate between the two rings (even b: cb_lw0, odd b: cb_lw1)
    if (lat) {
        const uint32_t cb_out = lat_q ? CB_LOQ : CB_LOK;
        compute_kernel_hw_startup<SrcOrder::Reverse>(CB_X, CB_LW0, cb_out);
        started = true;
        matmul_block_init(CB_X, CB_LW0, 0, 1, 1, LB);
        tile_regs_acquire();
        for (uint32_t b = 0; b < NLB; ++b) {
            const uint32_t cb_w = (b & 1u) ? CB_LW1 : CB_LW0;
            cb_wait_front(CB_X, (b + 1) * LB);
            cb_wait_front(cb_w, LB);
            for (uint32_t i = 0; i < LB; ++i) {
                matmul_block(CB_X, cb_w, b * LB + i, i, 0, 0, 1, 1, LB);
            }
            cb_pop_front(cb_w, LB);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(CB_X, KL);
    }

    // ---- QB: one output tile, K = 32 in order --------------------------------------------------------------------
    if (qb) {
        if (!started) {
            compute_kernel_hw_startup<SrcOrder::Reverse>(CB_CQN, CB_WA, CB_QOUT);
            started = true;
        } else {
            reconfig_data_format(CB_WA, CB_CQN);
            pack_reconfig_data_format(CB_QOUT);
        }
        matmul_block_init(CB_CQN, CB_WA, 0, 1, 1, KH);
        cb_wait_front(CB_WA, KH);
        cb_wait_front(CB_WB, KH);
        tile_regs_acquire();
        for (uint32_t k = 0; k < KQ; ++k) {
            if ((k & 3u) == 0) {
                cb_wait_front(CB_CQN, k + 4);
            }
            if (k < KH) {
                matmul_block(CB_CQN, CB_WA, k, k, 0, 0, 1, 1, KH);
            } else {
                matmul_block(CB_CQN, CB_WB, k, k - KH, 0, 0, 1, 1, KH);
            }
        }
        tile_regs_commit();
        if (is_nope) {
            cb_reserve_back(CB_QSEND, 1);
            tile_regs_wait();
            pack_tile(0, CB_QSEND);
            tile_regs_release();
            cb_push_back(CB_QSEND, 1);
        } else if (is_pe) {
            cb_reserve_back(CB_QOUT, 1);
            cb_reserve_back(CB_QSEND, 1);
            tile_regs_wait();
            pack_tile(0, CB_QOUT);
            pack_tile(0, CB_QSEND);
            tile_regs_release();
            cb_push_back(CB_QOUT, 1);
            cb_push_back(CB_QSEND, 1);
        } else {
            cb_reserve_back(CB_QOUT, 1);
            tile_regs_wait();
            pack_tile(0, CB_QOUT);
            tile_regs_release();
            cb_push_back(CB_QOUT, 1);
        }
        cb_pop_front(CB_WA, KH);
        cb_pop_front(CB_WB, KH);
        cb_pop_front(CB_CQN, KQ);

        if (is_gate) {  // the stock unary sigmoid on the bf16 linear output
            reconfig_data_format_srca(CB_QOUT);
            pack_reconfig_data_format(CB_GOUT);
            copy_init(CB_QOUT);
            cb_wait_front(CB_QOUT, 1);
            cb_reserve_back(CB_GOUT, 1);
            tile_regs_acquire();
            copy_tile(CB_QOUT, 0, 0);
            sigmoid_tile_init<false>();
            sigmoid_tile<VectorMode::RC, false, false>(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_GOUT);
            tile_regs_release();
            cb_pop_front(CB_QOUT, 1);
            cb_push_back(CB_GOUT, 1);
        } else if (is_pe) {  // rotary_embedding_hf of tile p = qk - 4 with the partner's tile
            cb_wait_front(CB_SCAL, 1);
            reconfig_data_format(CB_PEPART, CB_SCAL);
            pack_reconfig_data_format(CB_ROT);
            rope_tile<CB_QOUT, CB_PEPART, CB_COS, CB_SIN, CB_ROPE>(qk == NOPE_T);
        }
    }

    // ---- UK: per-head W_UK bmm, 2 output tiles --------------------------------------------------------------------
    if (uk) {
        if (!started) {
            compute_kernel_hw_startup<SrcOrder::Reverse>(CB_NOPE, CB_WUK, CB_UKOUT);
            started = true;
        } else {
            reconfig_data_format(CB_WUK, CB_NOPE);
            pack_reconfig_data_format(CB_UKOUT);
        }
        matmul_block_init(CB_NOPE, CB_WUK, 0, 1, 1, NOPE_T);
        cb_wait_front(CB_WUK, UK_PER * NOPE_T);
        cb_wait_front(CB_NOPE, NOPE_T);
        for (uint32_t t = 0; t < UK_PER; ++t) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < NOPE_T; ++k) {
                matmul_block(CB_NOPE, CB_WUK, k, t * NOPE_T + k, 0, 0, 1, 1, NOPE_T);
            }
            tile_regs_commit();
            cb_reserve_back(CB_UKOUT, 1);
            tile_regs_wait();
            pack_tile(0, CB_UKOUT);
            tile_regs_release();
            cb_push_back(CB_UKOUT, 1);
        }
        cb_pop_front(CB_NOPE, NOPE_T);
        cb_pop_front(CB_WUK, UK_PER * NOPE_T);
    }
}
