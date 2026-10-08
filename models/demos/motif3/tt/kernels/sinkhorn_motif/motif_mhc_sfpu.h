// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 exact mHC coefficients: the SFPU routines of compute_sinkhorn_motif.cpp (logits, sigmoid / exp, the
// 20-iteration Sinkhorn in registers; see that file for the algorithm and the DEST layout). Shared with the fused
// decode coefficients kernel (../mhc_decode/coeffs_compute.cpp, Phase C D3), which runs the same routine on the same
// DEST layout. Moved here verbatim from compute_sinkhorn_motif.cpp (no code change).

#pragma once

#ifdef TRISC_MATH
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {
namespace motif_mhc {

using sfpi::dst_reg;
using sfpi::vFloat;

// dst_reg index (32 per tile; DST base = tile 0) of the SFPLOAD holding rows 4*strip .. 4*strip+3 of `tile`, token
// half `half` (columns 16*half .. 16*half+15) and column parity `par`. Faces: 0 (rows 0-15, cols 0-15), 1 (rows 0-15,
// cols 16-31), 2 (rows 16-31, cols 0-15), 3 (rows 16-31, cols 16-31); 8 dst_reg entries per face, 2 per 4-row strip.
constexpr int dreg(int tile, int strip, int half, int par) {
    return 32 * tile + 8 * (2 * (strip >> 2) + half) + 2 * (strip & 3) + par;
}

constexpr float f32(uint32_t bits) { return __builtin_bit_cast(float, bits); }

// 1/s for s in [floor, 4 * e^20] (positive normal): SFPARECIP + 2 Newton steps (sfpu_reciprocal_iter<2> without the
// 0/inf NaN guard, which cannot trigger here). Needs vConstFloatPrgm0 = 2.0 (motif_mhc_init).
sfpi_inline vFloat recip_pos(vFloat s) {
    vFloat y = sfpi::approx_recip(s);
    vFloat t = s * y - sfpi::vConstFloatPrgm0;
    y = y * -t - 0.0f;
    t = s * y - sfpi::vConstFloatPrgm0;
    y = y * -t - 0.0f;
    return y;
}

// m / s given r ~ 1/s: q = m r, then one residual correction q += r (m - q s) (fused MAD residual).
sfpi_inline vFloat div_refined(vFloat m, vFloat s, vFloat r) {
    vFloat q = m * r;
    vFloat e = m - q * s;
    return q + e * r;
}

template <uint32_t FLOOR_BITS, bool REFINE>
sfpi_inline void normalize4(vFloat& m0, vFloat& m1, vFloat& m2, vFloat& m3) {
    vFloat s = m0 + m1;
    s = s + m2;
    s = s + m3;
    s = sfpi::max(s, vFloat(f32(FLOOR_BITS)));
    vFloat r = recip_pos(s);
    if constexpr (REFINE) {
        m0 = div_refined(m0, s, r);
        m1 = div_refined(m1, s, r);
        m2 = div_refined(m2, s, r);
        m3 = div_refined(m3, s, r);
    } else {
        m0 = m0 * r;
        m1 = m1 * r;
        m2 = m2 * r;
        m3 = m3 * r;
    }
}

// L = clamp(a * p + b, -c, c) with the mixes p in DST0, alpha in DST1, bias in DST2 (same strip / group).
// torch rounds the product and the sum separately (alpha * p, then + bias). The SFPI compiler would fuse x * a + b
// into one SFPMAD (a single rounding, up to a few ulp away from torch under cancellation), so the rounded product
// takes a DST round trip (exact in the fp32 DST) before the add.
template <int S, int H, int P, uint32_t CLAMP_BITS>
sfpi_inline vFloat logit(void) {
    vFloat x = dst_reg[dreg(0, S, H, P)];
    vFloat a = dst_reg[dreg(1, S, H, P)];
    dst_reg[dreg(0, S, H, P)] = x * a;
    vFloat l = dst_reg[dreg(0, S, H, P)];
    vFloat b = dst_reg[dreg(2, S, H, P)];
    l = l + b;
    return sfpi::clamp(l, -f32(CLAMP_BITS), f32(CLAMP_BITS));
}

// Step 2a for one token group: pre / post sigmoids and the 16 res exponentials.
template <int H, int P, uint32_t CPP, uint32_t CRES, uint32_t HPOST, uint32_t MODE>
sfpi_inline void logits_group() {
    {  // h_pre (mix strip 0) -> DST1 strip 0 (its alpha was just read)
        vFloat l = logit<0, H, P, CPP>();
        if constexpr (MODE == 0) {
            l = _sfpu_sigmoid_<true>(l);
        }
        dst_reg[dreg(1, 0, H, P)] = l;
    }
    {  // h_post (mix strip 1) -> DST2 strip 0 (the pre bias there was consumed above)
        vFloat l = logit<1, H, P, CPP>();
        if constexpr (MODE == 0) {
            l = _sfpu_sigmoid_<true>(l);
            if constexpr (HPOST != 0x3f800000u) {
                l = l * f32(HPOST);
            }
        }
        dst_reg[dreg(2, 0, H, P)] = l;
    }
    // H rows i = 0..3 (mix strips 2..5) -> DST3 strips 0..3
    {
        vFloat l = logit<2, H, P, CRES>();
        if constexpr (MODE == 0) {
            l = _sfpu_exp_fp32_accurate_<false>(l);
        }
        dst_reg[dreg(3, 0, H, P)] = l;
    }
    {
        vFloat l = logit<3, H, P, CRES>();
        if constexpr (MODE == 0) {
            l = _sfpu_exp_fp32_accurate_<false>(l);
        }
        dst_reg[dreg(3, 1, H, P)] = l;
    }
    {
        vFloat l = logit<4, H, P, CRES>();
        if constexpr (MODE == 0) {
            l = _sfpu_exp_fp32_accurate_<false>(l);
        }
        dst_reg[dreg(3, 2, H, P)] = l;
    }
    {
        vFloat l = logit<5, H, P, CRES>();
        if constexpr (MODE == 0) {
            l = _sfpu_exp_fp32_accurate_<false>(l);
        }
        dst_reg[dreg(3, 3, H, P)] = l;
    }
}

// Step 2b for one token group: 20 x (row-normalize, column-normalize) in registers.
template <int H, int P, uint32_t ITERS, uint32_t FLOOR_BITS, bool REFINE>
sfpi_inline void sinkhorn_group() {
    vFloat m0 = dst_reg[dreg(3, 0, H, P)];
    vFloat m1 = dst_reg[dreg(3, 1, H, P)];
    vFloat m2 = dst_reg[dreg(3, 2, H, P)];
    vFloat m3 = dst_reg[dreg(3, 3, H, P)];
#pragma GCC unroll 0
    for (uint32_t it = 0; it < ITERS; ++it) {
        // m_i = row i of H (lane group j = column j). TRANSP -> m_j = column j (lane group i = row i): the row sums
        // are lanewise sums of the four registers.
        sfpi::subvec_transp(m0, m1, m2, m3);
        normalize4<FLOOR_BITS, REFINE>(m0, m1, m2, m3);
        sfpi::subvec_transp(m0, m1, m2, m3);
        normalize4<FLOOR_BITS, REFINE>(m0, m1, m2, m3);  // column sums
    }
    dst_reg[dreg(3, 0, H, P)] = m0;
    dst_reg[dreg(3, 1, H, P)] = m1;
    dst_reg[dreg(3, 2, H, P)] = m2;
    dst_reg[dreg(3, 3, H, P)] = m3;
}

template <int TILE, int S0, int S1, int H>
sfpi_inline void zero_strips() {  // compile-time recursion: every dst_reg index is an immediate
    if constexpr (S0 < S1) {
        dst_reg[dreg(TILE, S0, H, 0)] = 0.0f;
        dst_reg[dreg(TILE, S0, H, 1)] = 0.0f;
        zero_strips<TILE, S0 + 1, S1, H>();
    }
}

template <int H>
sfpi_inline void zero_unused_rows() {
    zero_strips<1, 1, 8, H>();  // h_pre^T: only rows 0-3 are outputs
    zero_strips<2, 1, 8, H>();  // h_post^T
    zero_strips<3, 4, 8, H>();  // H^T: rows 0-15
}

template <int H>
sfpi_inline void zero_half() {  // tokens of a skipped half -> 0 in every output
    zero_strips<1, 0, 8, H>();
    zero_strips<2, 0, 8, H>();
    zero_strips<3, 0, 8, H>();
}

// mode 1 (layout passthrough, a bit-exact data-movement check): outputs = raw mixes; DST1 rows 0-3 = mixes 0-3,
// DST2 rows 0-3 = mixes 4-7, DST3 rows 0-15 = mixes 8-23.
template <int H, int P>
sfpi_inline void passthrough_group() {
    dst_reg[dreg(1, 0, H, P)] = vFloat(dst_reg[dreg(0, 0, H, P)]);
    dst_reg[dreg(2, 0, H, P)] = vFloat(dst_reg[dreg(0, 1, H, P)]);
    dst_reg[dreg(3, 0, H, P)] = vFloat(dst_reg[dreg(0, 2, H, P)]);
    dst_reg[dreg(3, 1, H, P)] = vFloat(dst_reg[dreg(0, 3, H, P)]);
    dst_reg[dreg(3, 2, H, P)] = vFloat(dst_reg[dreg(0, 4, H, P)]);
    dst_reg[dreg(3, 3, H, P)] = vFloat(dst_reg[dreg(0, 5, H, P)]);
}

template <int H, uint32_t ITERS, uint32_t CPP, uint32_t CRES, uint32_t FLOOR, uint32_t HPOST, uint32_t MODE, bool REFINE>
sfpi_inline void half_tile() {
    if constexpr (MODE == 1) {
        passthrough_group<H, 0>();
        passthrough_group<H, 1>();
    } else {
        logits_group<H, 0, CPP, CRES, HPOST, MODE>();
        logits_group<H, 1, CPP, CRES, HPOST, MODE>();
        if constexpr (MODE == 0) {
            sinkhorn_group<H, 0, ITERS, FLOOR, REFINE>();
            sinkhorn_group<H, 1, ITERS, FLOOR, REFINE>();
        }
    }
    zero_unused_rows<H>();
}

template <uint32_t ITERS, uint32_t HALVES, uint32_t CPP, uint32_t CRES, uint32_t FLOOR, uint32_t HPOST, uint32_t MODE,
          bool REFINE>
inline void motif_mhc_tile() {
    half_tile<0, ITERS, CPP, CRES, FLOOR, HPOST, MODE, REFINE>();
    if constexpr (HALVES == 2) {
        half_tile<1, ITERS, CPP, CRES, FLOOR, HPOST, MODE, REFINE>();
    } else {
        zero_half<1>();
    }
}

inline void motif_mhc_init() {
    sfpi::vConstFloatPrgm0 = 2.0f;  // recip_pos / _sfpu_sigmoid_ (sfpu_reciprocal_iter) Newton constant
}

}  // namespace motif_mhc
}  // namespace ckernel::sfpu
#endif  // TRISC_MATH
