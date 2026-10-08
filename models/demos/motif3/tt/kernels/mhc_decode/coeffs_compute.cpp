// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused mHC decode coefficients (D3, docs/OPTIMIZATION_PLAN.md §3.3 C2): compute. One core, one 32-token tile.
//
// 1. finalize (tt/mhc.py finalize_mixes, the same LLK calls in the same order): DST0 = ((Y0 + Y1) + Y2) + ... (SFPU
//    fp32 adds of UnpackToDestFp32 copies), DST2 = ((S0 + S1) + ...) + eps, rsqrt, DST0 *= column 0 of DST2
//    (sfpu_mul_bcast_col). With HALF (T <= 16) the adds / eps / rsqrt run on the faces that reach the Sinkhorn only:
//    faces 0-1 of the projection sums (token rows 0..15, all 32 mix columns: VectorMode::R) and face 0 of the
//    statistics (rows 0..15, column 0: one face). The SFPU code per face is the release code, so those faces are
//    bitwise the release p; rows 16..31 of p are never read (the Sinkhorn processes token half 0 only, num_halves 1).
//    The temporary copies alternate between two DEST slots so the unpack of the next partial can overlap the add.
//    p is packed to CB_PM (fp32, exact).
// 2. Sinkhorn (sinkhorn_motif/compute_sinkhorn_motif.cpp, steps 1-2, through the shared motif_mhc_sfpu.h): p^T ->
//    DST0 (exact transpose), alpha / bias -> DST1 / DST2, motif_mhc_tile: h_pre^T in DST1 rows 0-3, h_post^T in DST2
//    rows 0-3, H^T in DST3 rows 0-15 (token columns; padding columns zero / unused rows zero as in the release).
// 3. Packed tile (instead of the release's transposes back + coefficient_layout): DST0 rows 0-3 = h_pre^T, 4-7 =
//    h_post^T, 8-23 = H^T, 24-31 = 0, every word rounded to nearest-even TF32 exactly as coefficient_layout's
//    rne_tf32 (Inf / NaN unchanged), packed to CB_OUT (fp32).
//
// CT args: 0 NB (partials), 1 CHUNK, 2 eps bits, 3 HALF (0/1), then the Sinkhorn's: 4 ITERS, 5 HALVES, 6 CPP bits,
//          7 CRES bits, 8 FLOOR bits, 9 HPOST bits, 10 REFINE.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/dataflow/circular_buffer.h"

#include "../sinkhorn_motif/motif_mhc_sfpu.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
namespace motif_mhc_pack {

using sfpi::dst_reg;
using sfpi::vFloat;
using sfpi::vInt;
using sfpi::vUInt;
using ckernel::sfpu::motif_mhc::dreg;

// fp32 bits -> nearest-even TF32 (coefficient_layout's rne_tf32): (b + 0xfff + bit13) & ~0x1fff unless Inf / NaN.
sfpi_inline vFloat rne_tf32(vFloat v) {
    vUInt b = sfpi::as<vUInt>(v);
    vInt d = sfpi::as<vInt>(b & 0x7f800000u) - 0x7f800000;  // < 0 <=> finite
    v_if(d < 0) {
        vUInt r = b + 0x0fffu;
        r = r + ((b >> 13) & 1u);
        b = r & 0xffffe000u;
    }
    v_endif;
    return sfpi::as<vFloat>(b);
}

template <int SRC_TILE, int SRC_STRIP, int DST_STRIP, int H, int P>
sfpi_inline void move_strip() {
    dst_reg[dreg(0, DST_STRIP, H, P)] = rne_tf32(dst_reg[dreg(SRC_TILE, SRC_STRIP, H, P)]);
}

template <int H, int P>
sfpi_inline void pack_group() {
    move_strip<1, 0, 0, H, P>();  // h_pre^T  -> rows 0..3
    move_strip<2, 0, 1, H, P>();  // h_post^T -> rows 4..7
    move_strip<3, 0, 2, H, P>();  // H^T rows 0..15 -> rows 8..23
    move_strip<3, 1, 3, H, P>();
    move_strip<3, 2, 4, H, P>();
    move_strip<3, 3, 5, H, P>();
    dst_reg[dreg(0, 6, H, P)] = 0.0f;
    dst_reg[dreg(0, 7, H, P)] = 0.0f;
}

inline void packed_tile() {
    pack_group<0, 0>();
    pack_group<0, 1>();
    pack_group<1, 0>();
    pack_group<1, 1>();
}

}  // namespace motif_mhc_pack
}  // namespace ckernel::sfpu
#endif  // TRISC_MATH

namespace {
constexpr uint32_t CB_Y = 0;     // projection partials (fp32, UnpackToDestFp32)
constexpr uint32_t CB_S = 1;     // sum(X^2) partials (fp32, UnpackToDestFp32)
constexpr uint32_t CB_C = 2;     // alpha / bias tiles (fp32, UnpackToDestFp32)
constexpr uint32_t CB_PM = 3;    // p (fp32, UnpackToDestFp32: exact transpose)
constexpr uint32_t CB_OUT = 16;  // packed coefficient tile
}  // namespace

void kernel_main() {
    constexpr uint32_t nb = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(2);
    constexpr uint32_t HALF = get_compile_time_arg_val(3);
    constexpr uint32_t ITERS = get_compile_time_arg_val(4);
    constexpr uint32_t HALVES = get_compile_time_arg_val(5);
    constexpr uint32_t CPP = get_compile_time_arg_val(6);
    constexpr uint32_t CRES = get_compile_time_arg_val(7);
    constexpr uint32_t FLOOR = get_compile_time_arg_val(8);
    constexpr uint32_t HPOST = get_compile_time_arg_val(9);
    constexpr bool REFINE = get_compile_time_arg_val(10) != 0;
    static_assert(HALF == 0 || HALVES == 1, "HALF needs the single-half Sinkhorn");

    compute_kernel_hw_startup(CB_Y, CB_PM);
    CircularBuffer cy(CB_Y), cs(CB_S), cc(CB_C), cpm(CB_PM), co(CB_OUT);

    // ---- 1. finalize -> p ----
    tile_regs_acquire();
    add_binary_tile_init();
    copy_init(CB_Y);
    for (uint32_t b = 0; b < nb; b += chunk) {  // DST0 = sum_b Y_b
        cy.wait_front(chunk);
        for (uint32_t i = 0; i < chunk; ++i) {
            const uint32_t k = b + i;
#ifdef MHC_CO_DEBUG_NO_Y
            if (k != 0) {
                continue;
            }
#endif
            if (k == 0) {
                copy_tile(CB_Y, i, 0);
            } else {
                const uint32_t tmp = (k & 1) ? 1 : 3;
                copy_tile(CB_Y, i, tmp);
#ifdef MHC_CO_DEBUG_NO_ADD
                if constexpr (false) {
#else
                if constexpr (HALF) {
#endif
                    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary,
                                           (APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE,
                                            ckernel::DstRoundingMode::Default),
                                           0, tmp, 0, VectorMode::R)));
                } else {
#ifndef MHC_CO_DEBUG_NO_ADD
                    add_binary_tile(0, tmp, 0);
#endif
                }
            }
        }
        cy.pop_front(chunk);
    }
    reconfig_data_format_srca(CB_Y, CB_S);
    copy_init(CB_S);
    for (uint32_t b = 0; b < nb; b += chunk) {  // DST2 = sum_b S_b (column 0)
        cs.wait_front(chunk);
        for (uint32_t i = 0; i < chunk; ++i) {
            const uint32_t k = b + i;
#ifdef MHC_CO_DEBUG_NO_S
            if (k != 0) {
                continue;
            }
#endif
            if (k == 0) {
                copy_tile(CB_S, i, 2);
            } else {
                const uint32_t tmp = (k & 1) ? 3 : 1;
                copy_tile(CB_S, i, tmp);
#ifdef MHC_CO_DEBUG_NO_ADD
                if constexpr (false) {
#else
                if constexpr (HALF) {
#endif
                    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary,
                                           (APPROX, ckernel::BinaryOp::ADD, 8, DST_ACCUM_MODE,
                                            ckernel::DstRoundingMode::Default),
                                           2, tmp, 2, VectorMode::RC_custom)));
                } else {
#ifndef MHC_CO_DEBUG_NO_ADD
                    add_binary_tile(2, tmp, 2);
#endif
                }
            }
        }
        cs.pop_front(chunk);
    }
    binop_with_scalar_tile_init();
    if constexpr (HALF) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
                             (APPROX, ckernel::ADD_UNARY, 8, DST_ACCUM_MODE), 2, VectorMode::RC_custom, eps_bits));
    } else {
        add_unary_tile(2, eps_bits);  // ss + 16384 * 1e-6
    }
    rsqrt_tile_init();
    if constexpr (HALF) {
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, 8, DST_ACCUM_MODE, false), 2,
                             VectorMode::RC_custom));
    } else {
        rsqrt_tile(2);
    }
    sfpu_mul_bcast_col_init();
    sfpu_mul_bcast_col(0, 2);  // p = p_un * r
    tile_regs_commit();
    cpm.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, CB_PM);
    tile_regs_release();
    cpm.push_back(1);

    // ---- 2. Sinkhorn (compute_sinkhorn_motif.cpp step 1-2) ----
    cc.wait_front(2);
    cpm.wait_front(1);
    tile_regs_acquire();
    reconfig_data_format_srca(CB_S, CB_PM);
    transpose_init(CB_PM);
    transpose_tile(CB_PM, 0, 0);
    copy_init(CB_C);
    copy_tile(CB_C, 0, 1);
    copy_tile(CB_C, 1, 2);
    MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::motif_mhc::motif_mhc_init)));
#ifndef MHC_CO_DEBUG_NO_SINKHORN
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::motif_mhc::motif_mhc_tile<ITERS, HALVES, CPP, CRES, FLOOR, HPOST, 0, REFINE>, 0,
        VectorMode::RC_custom)));
#endif
    // ---- 3. packed tile ----
    MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::motif_mhc_pack::packed_tile, 0, VectorMode::RC_custom)));
    tile_regs_commit();
    co.reserve_back(1);
    tile_regs_wait();
    pack_tile(0, CB_OUT);
    tile_regs_release();
    co.push_back(1);
    cpm.pop_front(1);
    cc.pop_front(2);
}
