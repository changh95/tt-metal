// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Tile-level helpers shared by the gdn_decode_step compute kernels (plain and fused-conv variants).
//
// Every multi-tile helper processes up to kDstTiles fp32 tiles per tile_regs_acquire/commit/wait/release round (DST
// holds 8 fp32 tiles with fp32_dest_acc_en; SyncHalf hands one half = 4 tiles to each round). The per-tile op sequence
// is exactly the one-tile-per-round sequence, only the DST slot differs, so results are bit-identical to the unbatched
// helpers. Data-format reconfiguration (unpacker srcA/srcB, packer) is tracked in g_fmt and only issued when the
// operand actually changes; the (old, new) LLK variants additionally skip the (stalling) reprogram when the two
// operands share a format.
#pragma once
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace gdn_step {

// fp32 tiles per DST half (fp32_dest_acc_en + DstSync::SyncHalf); scatter_conv in the fused kernel relies on the same.
constexpr uint32_t kDstTiles = 4;
constexpr uint32_t kNoCb = 0xFFFFFFFFu;

// ---- tracked data-format state -------------------------------------------------------------------------------------
// The CBs the unpacker (srcA, srcB) and the packer are currently configured for. Each TRISC runs the whole control
// flow, so the tracker is consistent on all three threads. The tracker only needs a CB whose FORMAT matches the
// hardware state: the (old, new) reconfig LLKs compare formats (not ids), so a stale id of the same format is harmless.
// Kernel code that reconfigures formats outside these helpers (compute_kernel_hw_startup, compute_kernel_lib::reduce)
// must call note_formats() afterwards.
struct FmtState {
    uint32_t srca = kNoCb;
    uint32_t srcb = kNoCb;
    uint32_t pack = kNoCb;
};
static FmtState g_fmt;

inline void note_formats(uint32_t srca, uint32_t srcb, uint32_t pack) {
    g_fmt.srca = srca;
    g_fmt.srcb = srcb;
    g_fmt.pack = pack;
}

inline void pack_fmt(uint32_t out) {
    if (out == g_fmt.pack) {
        return;
    }
    if (g_fmt.pack == kNoCb) {
        pack_reconfig_data_format(out);
    } else {
        pack_reconfig_data_format(g_fmt.pack, out);
    }
    g_fmt.pack = out;
}

inline void srca_fmt(uint32_t a) {
    if (a == g_fmt.srca) {
        return;
    }
    if (g_fmt.srca == kNoCb) {
        reconfig_data_format_srca(a);
    } else {
        reconfig_data_format_srca(g_fmt.srca, a);
    }
    g_fmt.srca = a;
}

inline void srcb_fmt(uint32_t b) {
    if (b == g_fmt.srcb) {
        return;
    }
    if (g_fmt.srcb == kNoCb) {
        reconfig_data_format_srcb(b);
    } else {
        reconfig_data_format_srcb(g_fmt.srcb, b);
    }
    g_fmt.srcb = b;
}

// icb0 -> srcA, icb1 -> srcB (Regular) or swapped (Reverse, matmul: in1 -> srcA, in0 -> srcB)
template <SrcOrder order = SrcOrder::Regular>
inline void ab_fmt(uint32_t icb0, uint32_t icb1) {
    constexpr bool rev = (order == SrcOrder::Reverse);
    srca_fmt(rev ? icb1 : icb0);
    srcb_fmt(rev ? icb0 : icb1);
}

// ---- DST rounds -----------------------------------------------------------------------------------------------------
// n tiles in rounds of up to kDstTiles: op(i, d) computes output tile i into DST slot d; the slot is packed to out[i].
template <typename Op>
inline void dst_rounds(uint32_t n, uint32_t out, Op&& op) {
    for (uint32_t i0 = 0; i0 < n; i0 += kDstTiles) {
        const uint32_t m = (n - i0) < kDstTiles ? (n - i0) : kDstTiles;
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; ++d) {
            op(i0 + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < m; ++d) {
            pack_tile(d, out, i0 + d);
        }
        tile_regs_release();
    }
}

// ---- helpers --------------------------------------------------------------------------------------------------------

// out[i] = a[i] * b[i]  (plain eltwise; b_fixed -> b[0] for every i)
template <bool b_fixed>
inline void multiply_tiles(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_fmt(out);
    ab_fmt(a, b);
    mul_init(a, b);
    dst_rounds(n, out, [&](uint32_t i, uint32_t d) { mul_tiles(a, b, i, b_fixed ? 0 : i, d); });
    out_dfb.push_back(n);
}

// out[i] = a[i] - b[i]
inline void subtract_tiles(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_fmt(out);
    ab_fmt(a, b);
    sub_init(a, b);
    dst_rounds(n, out, [&](uint32_t i, uint32_t d) { sub_tiles(a, b, i, i, d); });
    out_dfb.push_back(n);
}

// out[i] = a[i] + b[i]
inline void add_tiles_n(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_fmt(out);
    ab_fmt(a, b);
    add_init(a, b);
    dst_rounds(n, out, [&](uint32_t i, uint32_t d) { add_tiles(a, b, i, i, d); });
    out_dfb.push_back(n);
}

// out[i] = a[i] scaled per row by column 0 of col_tile (tile 0 of b)
inline void scale_rows(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_fmt(out);
    ab_fmt(a, b);
    mul_bcast_cols_init(a, b);
    dst_rounds(n, out, [&](uint32_t i, uint32_t d) { mul_tiles_bcast_cols(a, b, i, 0, d); });
    out_dfb.push_back(n);
}

// out[i] = a[i] scaled per column by row 0 of b[i]
inline void scale_cols(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_fmt(out);
    ab_fmt(a, b);
    mul_bcast_rows_init(a, b);
    dst_rounds(n, out, [&](uint32_t i, uint32_t d) { mul_tiles_bcast_rows(a, b, i, i, d); });
    out_dfb.push_back(n);
}

// out[i] = a[i] * a[i]  (bf16 or fp32 in, fp32 out)
inline void square_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_fmt(out);
    ab_fmt(a, a);
    mul_init(a, a);
    dst_rounds(n, out, [&](uint32_t i, uint32_t d) { mul_tiles(a, a, i, i, d); });
    out_dfb.push_back(n);
}

// stats[0] = row sums of in[0..Wt) (column 0)  -- compute_kernel_lib::reduce with the formats set through the tracker
template <uint32_t in_cb, uint32_t scaler_cb, uint32_t out_cb>
inline void row_sum(uint32_t Wt) {
    ab_fmt(in_cb, scaler_cb);
    pack_fmt(out_cb);
    compute_kernel_lib::reduce<
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_ROW,
        in_cb,
        scaler_cb,
        out_cb,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(compute_kernel_lib::ReduceInputBlockShape::of(1, Wt));
    note_formats(in_cb, scaler_cb, out_cb);
}

// l2 variant: inv = (rsqrt(stats + eps) * post) * mask   (stats: row sums of squares in column 0)
inline void inverse_l2(
    uint32_t stats,
    uint32_t eps,
    uint32_t mask,
    uint32_t scratch,
    DataflowBuffer& scratch_dfb,
    uint32_t inv,
    DataflowBuffer& inv_dfb,
    uint32_t post_scale_bits) {
    scratch_dfb.reserve_back(1);
    pack_fmt(scratch);
    ab_fmt(stats, eps);
    add_init(stats, eps);
    tile_regs_acquire();
    add_tiles(stats, eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, post_scale_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch, 0);
    tile_regs_release();
    scratch_dfb.push_back(1);
    scratch_dfb.wait_front(1);
    inv_dfb.reserve_back(1);
    pack_fmt(inv);
    ab_fmt(scratch, mask);
    mul_init(scratch, mask);
    tile_regs_acquire();
    mul_tiles(scratch, mask, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, inv, 0);
    tile_regs_release();
    inv_dfb.push_back(1);
    scratch_dfb.pop_front(1);
}

// rms variant: inv = rsqrt(stats * inv_n + eps)
inline void inverse_rms(
    uint32_t stats,
    uint32_t eps,
    uint32_t scratch,
    DataflowBuffer& scratch_dfb,
    uint32_t inv,
    DataflowBuffer& inv_dfb,
    uint32_t inv_n_bits) {
    scratch_dfb.reserve_back(1);
    pack_fmt(scratch);
    srca_fmt(stats);
    copy_init(stats);
    tile_regs_acquire();
    copy_tile(stats, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_n_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch, 0);
    tile_regs_release();
    scratch_dfb.push_back(1);
    scratch_dfb.wait_front(1);
    inv_dfb.reserve_back(1);
    pack_fmt(inv);
    ab_fmt(scratch, eps);
    add_init(scratch, eps);
    tile_regs_acquire();
    add_tiles(scratch, eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, inv, 0);
    tile_regs_release();
    inv_dfb.push_back(1);
    scratch_dfb.pop_front(1);
}

// out[j] = sum_i a[i] @ b[i*Nt + j]   (a: 1 x Kt row of tiles, b: Kt x Nt tiles; each output accumulates in its slot)
inline void row_times_matrix(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t Kt, uint32_t Nt) {
    out_dfb.reserve_back(Nt);
    pack_fmt(out);
    ab_fmt<SrcOrder::Reverse>(a, b);
    matmul_init(a, b);
    dst_rounds(Nt, out, [&](uint32_t j, uint32_t d) {
        for (uint32_t i = 0; i < Kt; ++i) {
            matmul_tiles(a, b, i, i * Nt + j, d);
        }
    });
    out_dfb.push_back(Nt);
}

// out[i*Nt + j] = a[i] @ b[j]   (outer product of a column block and a row block, K = 1 tile)
inline void outer_product(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t Kt, uint32_t Nt) {
    out_dfb.reserve_back(Kt * Nt);
    pack_fmt(out);
    ab_fmt<SrcOrder::Reverse>(a, b);
    matmul_init(a, b);
    dst_rounds(Kt * Nt, out, [&](uint32_t t, uint32_t d) { matmul_tiles(a, b, t / Nt, t % Nt, d); });
    out_dfb.push_back(Kt * Nt);
}

// out[i] = transpose(a[i])  (32-bit in-DST transpose)
inline void transpose_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_fmt(out);
    srca_fmt(a);
    for (uint32_t i0 = 0; i0 < n; i0 += kDstTiles) {
        const uint32_t m = (n - i0) < kDstTiles ? (n - i0) : kDstTiles;
        // transpose_dest_init reprograms the math pipeline, so the datacopy must be re-initialised per round
        copy_init(a);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; ++d) {
            copy_tile(a, i0 + d, d);
        }
        transpose_dest_init<true>(a);
        for (uint32_t d = 0; d < m; ++d) {
            transpose_dest<true>(d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < m; ++d) {
            pack_tile(d, out, i0 + d);
        }
        tile_regs_release();
    }
    out_dfb.push_back(n);
}

// out[i] = a[i]
inline void copy_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_fmt(out);
    srca_fmt(a);
    copy_init(a);
    dst_rounds(n, out, [&](uint32_t i, uint32_t d) { copy_tile(a, i, d); });
    out_dfb.push_back(n);
}

// out = exp(a[0])
inline void exp_tile_copy(uint32_t a, uint32_t out, DataflowBuffer& out_dfb) {
    out_dfb.reserve_back(1);
    pack_fmt(out);
    srca_fmt(a);
    copy_init(a);
    tile_regs_acquire();
    copy_tile(a, 0, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out, 0);
    tile_regs_release();
    out_dfb.push_back(1);
}

}  // namespace gdn_step
