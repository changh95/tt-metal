// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Per-token building blocks shared by the fused-conv GDN decode compute kernels (one-token gdn_decode_step_conv.cpp and
// multi-token gdn_decode_step_conv_mt.cpp): the 0/1-selector scatter of the packed conv row, the beta / decay gates
// from the per-token scalar tiles, silu(z) and the gated-norm product. Pure moves from the one-token kernel: the
// instruction sequences are unchanged.
#pragma once
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/compute/gdn_step_helpers.hpp"

namespace gdn_step_conv {

using namespace gdn_step;

constexpr uint32_t kOneBits = 0x3F800000u;     // 1.0f
constexpr uint32_t kTwentyBits = 0x41A00000u;  // 20.0f (softplus threshold, as ttnn.softplus(1.0, 20.0))

// Scatter: tile c (row b = the user's row of conv_p, other rows exactly 0) = sel[c] @ conv_p, packed into qc | kc | vc.
template <uint32_t Kt, uint32_t Vt>
inline void scatter_conv(DataflowBuffer& qc, DataflowBuffer& kc, DataflowBuffer& vc) {
    constexpr uint32_t Ct = 2 * Kt + Vt;
    qc.reserve_back(Kt);
    kc.reserve_back(Kt);
    vc.reserve_back(Vt);
    pack_fmt(dfb::qc);  // qc | kc | vc share the fp32 format
    ab_fmt<SrcOrder::Reverse>(dfb::sel, dfb::conv_p);
    matmul_init(dfb::sel, dfb::conv_p);
    for (uint32_t c0 = 0; c0 < Ct; c0 += kDstTiles) {
        const uint32_t n = (Ct - c0) < kDstTiles ? (Ct - c0) : kDstTiles;
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            matmul_tiles(dfb::sel, dfb::conv_p, c0 + i, 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t c = c0 + i;
            if (c < Kt) {
                pack_tile(i, dfb::qc, c);
            } else if (c < 2 * Kt) {
                pack_tile(i, dfb::kc, c - Kt);
            } else {
                pack_tile(i, dfb::vc, c - 2 * Kt);
            }
        }
        tile_regs_release();
    }
    qc.push_back(Kt);
    kc.push_back(Kt);
    vc.push_back(Vt);
}

// beta_t = sigmoid(b) (all-equal scalar tile)
inline void gate_beta(DataflowBuffer& beta_t) {
    beta_t.reserve_back(1);
    pack_fmt(dfb::beta_t);
    srca_fmt(dfb::b_s);
    copy_init(dfb::b_s);
    tile_regs_acquire();
    copy_tile(dfb::b_s, 0, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::beta_t, 0);
    tile_regs_release();
    beta_t.push_back(1);
}

// dec = exp(neg_exp_A * softplus(a + dt_bias))  (all-equal scalar tile)
inline void gate_decay(DataflowBuffer& dec) {
    dec.reserve_back(1);
    pack_fmt(dfb::dec);
    srca_fmt(dfb::a_s);  // a_s / dtb_s / nea_s are all fp32 scalar tiles: one copy init
    copy_init(dfb::a_s);
    tile_regs_acquire();
    copy_tile(dfb::a_s, 0, 0);
    copy_tile(dfb::dtb_s, 0, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    softplus_tile_init();
    softplus_tile(0, kOneBits, kOneBits, kTwentyBits);
    copy_init(dfb::nea_s);  // FPU datacopy re-init after the SFPU ops
    copy_tile(dfb::nea_s, 0, 2);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
    exp_tile_init<false>();
    exp_tile<false>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::dec, 0);
    tile_regs_release();
    dec.push_back(1);
}

// zs[i] = silu(z[i])
inline void silu_tiles(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_fmt(out);
    srca_fmt(a);
    copy_init(a);
    silu_tile_init();
    dst_rounds(n, out, [&](uint32_t i, uint32_t d) {
        copy_tile(a, i, d);
        silu_tile(d);
    });
    out_dfb.push_back(n);
}

// gated = (on x w per column) * zs, with the final product on the SFPU (fp32); rows other than b stay exactly 0.
// Two tiles per DST round (each needs 2 of the 4 fp32 slots: product in d, zs in d + 2).
inline void gated_user(DataflowBuffer& tmp, uint32_t n) {
    constexpr uint32_t per_round = kDstTiles / 2;
    tmp.reserve_back(n);
    pack_fmt(dfb::tmp);
    ab_fmt(dfb::on, dfb::w_in);  // zs (fp32) shares on's srcA format: no reconfig between the two unpacks
    for (uint32_t i0 = 0; i0 < n; i0 += per_round) {
        const uint32_t m = (n - i0) < per_round ? (n - i0) : per_round;
        mul_bcast_rows_init(dfb::on, dfb::w_in);
        tile_regs_acquire();
        for (uint32_t d = 0; d < m; ++d) {
            mul_tiles_bcast_rows(dfb::on, dfb::w_in, i0 + d, i0 + d, d);
        }
        copy_init(dfb::zs);
        for (uint32_t d = 0; d < m; ++d) {
            copy_tile(dfb::zs, i0 + d, per_round + d);
        }
        mul_binary_tile_init();
        for (uint32_t d = 0; d < m; ++d) {
            mul_binary_tile(d, per_round + d, d);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t d = 0; d < m; ++d) {
            pack_tile(d, dfb::tmp, i0 + d);
        }
        tile_regs_release();
    }
    tmp.push_back(n);
}

}  // namespace gdn_step_conv
