// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 exact mHC coefficients (Option B, WAVE_A_REVIEW MHC-6, design §2.3.3 step 3): compute.
//
// Started from ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_split_sinkhorn/device/kernels/compute/
// mhc_split_sinkhorn_compute.cpp. The stock kernel forms the logits with FPU matmuls (mixes @ SEL + base) and
// normalizes with FPU row/col-sum matmuls (m @ RB / m @ CB): every operand is TF32-truncated, it has no +-10 / +-20
// clamps, no 1e-8 floor, post = 2 sigma, and it chains ~130 dependent L1 tile round trips (52 us per call). Here the
// whole parametrization runs in fp32 SFPU registers on the Motif semantics (HF modeling_motif.py:226-249):
//
//   h_pre  = sigmoid(clamp(a_pre  * p_pre  + b_pre,  -10, 10))                 [T, 4]
//   h_post = 1.0 * sigmoid(clamp(a_post * p_post + b_post, -10, 10))           [T, 4]   (no x2)
//   H      = Sinkhorn_20(exp(clamp(a_res * p_res + B_res, -20, 20)))           [T, 16]  (H[i][j] at col 4i + j)
//            Sinkhorn: 20 x { m /= max(rowsum, 1e-8); m /= max(colsum, 1e-8) }, row first.
//
// Per 32-token tile (one tile = 32 tokens on rows, the 24 raw mixes p on columns 0..23):
//   1. transpose_tile(mixes) -> DST0 = mixes^T: row q = mix q, column = token. Unpack-to-dest fp32 + the 32-bit
//      in-dest transpose: bit exact (no TF32 truncation; the CBs are UnpackToDestFp32).
//      copy_tile(consts) -> DST1 = alpha tile (row q = alpha of mix q's group), DST2 = bias tile (row q = b_q).
//   2. SFPU, per group of 8 tokens (token half h, column parity p): an SFPLOAD of rows 4S..4S+3 holds 4 mixes x 8
//      tokens (lane = 8 * row_in_strip + token/2). Logits L = a * p + b (fp32 mul then add, like torch), clamp,
//      sigmoid (pre -> DST1 rows 0-3, post -> DST2 rows 0-3) or exp (res -> DST3 rows 0-15). The 4 res strips are
//      the 4 rows i of H (lane group = column j), so in registers:
//        column sums = lanewise SFPU adds of the 4 LREGs; row sums = the same after SFPTRANSP (LREG <-> lane group);
//      ((m0 + m1) + m2) + m3 in fp32, max(., 1e-8), reciprocal (approx + 2 Newton steps), multiply; 20 x (TRANSP,
//      row-norm, TRANSP, col-norm), all in LREGs (no DST / L1 round trip between iterations).
//   3. Unused rows of DST1..3 are zeroed, DST1..3 are packed to an fp32 scratch CB and transposed back with the same
//      exact fp32 transpose_tile, then packed to the three output CBs (h_pre, h_post, H; logical [T,4], [T,4], [T,16]
//      fp32 TILE, padding columns zero).
//
// CT args: [iters, num_halves (1: tokens 0..15 only, 2: all 32), clamp_pp_bits, clamp_res_bits, floor_bits,
//           hpost_coeff_bits, mode (0 full, 1 layout passthrough, 2 clamped logits), div_refine (0/1)]
// Common RT args: [num_tiles_total, num_cores, grid_y] (work split as in the reader)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

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

namespace {
constexpr uint32_t CB_MIXES = 0;   // c_0: mixes [32 tokens, 32] fp32, streamed
constexpr uint32_t CB_CONSTS = 1;  // c_1: alpha tile, bias tile (resident)
constexpr uint32_t CB_PRE = 2;     // c_2: h_pre out
constexpr uint32_t CB_POST = 3;    // c_3: h_post out
constexpr uint32_t CB_COMB = 4;    // c_4: H out
constexpr uint32_t CB_TMP = 24;    // c_24: transposed results (3 tiles), fp32 scratch
}  // namespace

void kernel_main() {
    constexpr uint32_t ITERS = get_compile_time_arg_val(0);
    constexpr uint32_t HALVES = get_compile_time_arg_val(1);
    constexpr uint32_t CPP = get_compile_time_arg_val(2);
    constexpr uint32_t CRES = get_compile_time_arg_val(3);
    constexpr uint32_t FLOOR = get_compile_time_arg_val(4);
    constexpr uint32_t HPOST = get_compile_time_arg_val(5);
    constexpr uint32_t MODE = get_compile_time_arg_val(6);
    constexpr bool REFINE = get_compile_time_arg_val(7) != 0;
    static_assert(HALVES == 1 || HALVES == 2, "num_halves must be 1 or 2");
    static_assert(MODE <= 2, "mode must be 0, 1 or 2");

    const uint32_t total_tiles = get_common_arg_val<uint32_t>(0);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(2);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t num_token_tiles = total_tiles / num_cores + (core_i < total_tiles % num_cores ? 1 : 0);

    compute_kernel_hw_startup(CB_MIXES, CB_CONSTS, CB_TMP);

    CircularBuffer cb_mixes(CB_MIXES), cb_consts(CB_CONSTS), cb_tmp(CB_TMP);
    CircularBuffer cb_pre(CB_PRE), cb_post(CB_POST), cb_comb(CB_COMB);
    cb_consts.wait_front(2);  // resident for the whole op

    for (uint32_t t = 0; t < num_token_tiles; ++t) {
        // ---- 1. mixes^T -> DST0, alpha -> DST1, bias -> DST2 (exact fp32 unpack-to-dest) ----
        cb_mixes.wait_front(1);
        tile_regs_acquire();
        transpose_init(CB_MIXES);
        transpose_tile(CB_MIXES, 0, 0);
        copy_init(CB_CONSTS);
        copy_tile(CB_CONSTS, 0, 1);
        copy_tile(CB_CONSTS, 1, 2);

        // ---- 2. logits, sigmoid / exp, Sinkhorn (SFPU, fp32) ----
        MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::motif_mhc::motif_mhc_init)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::motif_mhc::motif_mhc_tile<ITERS, HALVES, CPP, CRES, FLOOR, HPOST, MODE, REFINE>,
            0,
            VectorMode::RC_custom)));
        tile_regs_commit();

        cb_tmp.reserve_back(3);
        tile_regs_wait();
        pack_tile(1, CB_TMP);  // h_pre^T
        pack_tile(2, CB_TMP);  // h_post^T
        pack_tile(3, CB_TMP);  // H^T
        tile_regs_release();
        cb_tmp.push_back(3);
        cb_mixes.pop_front(1);

        // ---- 3. transpose back to token rows ----
        cb_tmp.wait_front(3);
        cb_pre.reserve_back(1);
        cb_post.reserve_back(1);
        cb_comb.reserve_back(1);
        tile_regs_acquire();
        transpose_init(CB_TMP);
        transpose_tile(CB_TMP, 0, 0);
        transpose_tile(CB_TMP, 1, 1);
        transpose_tile(CB_TMP, 2, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_PRE);
        pack_tile(1, CB_POST);
        pack_tile(2, CB_COMB);
        tile_regs_release();
        cb_pre.push_back(1);
        cb_post.push_back(1);
        cb_comb.push_back(1);
        cb_tmp.pop_front(3);
    }
}
