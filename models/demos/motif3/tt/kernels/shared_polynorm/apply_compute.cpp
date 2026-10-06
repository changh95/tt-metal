// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused shared-expert PolyNorm (B5), apply program, compute.
//
// The release (tt/polynorm.py, decode, ar="ag_sum", horner="mac") after the moments all-gather runs
//   s  = sum(gat, dim -1)             accurate fp32 reduce: acc = tile k=0, acc = add_binary_tile(acc, tile k) k = 1..TP-1
//                                     (the zero-padded columns 1..31 add exact zeros, so column 0 is the sequential sum)
//   v  = mac(s, D, E)                 SFPMAD s D + E
//   a  = rsqrt(v)                     rsqrt_tile<Default>, approx off
//   t  = mac(g, a3, a2); t = mac(t, g, a1); t = mac(t, g, b)       SFPMAD, column / scalar broadcasts
//   h  = multiply(t, u) -> bf16       SFPU mul in fp32 DEST, then binary_ng's TYPECAST post-activation
//                                     (typecast_tile<Float32, Float16_b>: round to nearest even on the SFPU), packed
// This kernel issues the same operations with the same operands in the same order: the fold as SFPI adds
// (acc + x, k = 1..TP-1: the SFPADD add_binary_tile runs), s D + E as one SFPI multiply-add (the SFPMAD mac_tile runs),
// rsqrt_tile, then the Horner chain as SFPI multiply-adds in the release's operand order (g a3 + a2, t g + a1,
// t g + b, t u), then the same typecast_tile before the pack (a plain fp32 -> bf16 pack rounds exact ties away from zero,
// the typecast rounds them to even: without it 1 in ~1e6 values differed). The fold and s D + E only need column 0, so
// they run on faces 0 and 2 (the faces holding it). The per-row coefficients a_m and b are broadcast from column 0 to full tiles with an exact
// 0 + x column-broadcast add (non-finite values outside column 0 -- stale L1 in the constant tiles -- are zeroed first,
// as in B3: the broadcast helper does not tolerate them).
//
// CT: 0 cb_recv, 1 cb_k, 2 cb_g, 3 cb_u, 4 cb_coef, 5 cb_out, 6 n, 7 TP, 8 debug (0; 1 = write the 4 coefficient tiles
//     A1 A2 A3 B to cb_out (fp32) instead of h: diagnostics only)

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
namespace ckernel::sfpu::motif_spn {
using sfpi::dst_reg;
using sfpi::vFloat;
// DEST 0..TP-1 = the TP gathered partial tiles of one moment, 1 / 2 after the fold = D_m / E_m. Faces 0 and 2 only
// (vectors 0..7 and 16..23 of a tile: column 0 lives there). Sequential fold in gather order (acc + x), then s D + E.
template <int TP>
inline void fold() {
#pragma GCC unroll 1
    for (int i = 0; i < 8; ++i) {
#pragma GCC unroll 2
        for (int f = 0; f < 2; ++f) {
            vFloat acc = dst_reg[16 * f];
#pragma GCC unroll 8
            for (int k = 1; k < TP; ++k) {
                acc = acc + dst_reg[k * 32 + 16 * f];
            }
            dst_reg[16 * f] = acc;
        }
        dst_reg++;
    }
}
inline void mac_de() {
#pragma GCC unroll 1
    for (int i = 0; i < 8; ++i) {
#pragma GCC unroll 2
        for (int f = 0; f < 2; ++f) {
            vFloat s = dst_reg[16 * f];
            vFloat d = dst_reg[32 + 16 * f];
            vFloat e = dst_reg[64 + 16 * f];
            dst_reg[16 * f] = s * d + e;
        }
        dst_reg++;
    }
}
// DEST tile 0: every non-finite value -> 0 (column 0 is finite for finite inputs)
inline void finite_or_zero_t0() {
#pragma GCC unroll 1
    for (int i = 0; i < 32; ++i) {
        vFloat v = dst_reg[0];
        v_if(sfpi::exexp(v, sfpi::ExponentMode::Biased) == 255) { v = 0.0f; }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}
// DEST 0..3 = A1, A2, A3, B (full tiles); tile pair (4 + 2t, 5 + 2t) = (g, u) -> h into 4 + 2t, release order
template <int NT>
inline void horner() {
#pragma GCC unroll 1
    for (int i = 0; i < 32; ++i) {
        vFloat a1 = dst_reg[0], a2 = dst_reg[1 * 32], a3 = dst_reg[2 * 32], b = dst_reg[3 * 32];
#pragma GCC unroll 2
        for (int t = 0; t < NT; ++t) {
            vFloat g = dst_reg[(4 + 2 * t) * 32];
            vFloat r = g * a3 + a2;
            r = r * g + a1;
            r = r * g + b;
            dst_reg[(4 + 2 * t) * 32] = r * vFloat(dst_reg[(5 + 2 * t) * 32]);
        }
        dst_reg++;
    }
}
inline void noop_init() {}
}  // namespace ckernel::sfpu::motif_spn
#endif
#define MOTIF_SPN_SFPU(fn)                                                                                    \
    MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::motif_spn::noop_init))); \
    MATH((_llk_math_eltwise_unary_sfpu_params_(fn, 0, VectorMode::RC_custom)))

// DEST tile 0 (column 0 = the per-row value) -> full-tile column broadcast -> cb_coef
template <uint32_t cb_coef>
inline void bcast_pack() {
    MOTIF_SPN_SFPU((ckernel::sfpu::motif_spn::finite_or_zero_t0));
    fill_tile_init();
    fill_tile(1, 0.0f);
    sfpu_bcast_col_init();
    sfpu_add_bcast_col(1, 0);  // 0 + a: exact
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1, cb_coef);
    tile_regs_release();
    cb_push_back(cb_coef, 1);
}

void kernel_main() {
    constexpr uint32_t cb_recv = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr uint32_t cb_u = get_compile_time_arg_val(3);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t n = get_compile_time_arg_val(6);
    constexpr uint32_t TP = get_compile_time_arg_val(7);
    constexpr uint32_t DEBUG = get_compile_time_arg_val(8);

    compute_kernel_hw_startup(cb_recv, cb_coef);
    cb_wait_front(cb_k, 7);

    // ---- per-row coefficients a1, a2, a3 (moment order g^2, g^4, g^6), then b ----
    for (uint32_t m = 0; m < 3; ++m) {
        cb_reserve_back(cb_coef, 1);
        cb_wait_front(cb_recv, TP);
        tile_regs_acquire();
        copy_init(cb_recv);
        for (uint32_t k = 0; k < TP; ++k) {
            copy_tile(cb_recv, k, k);
        }
        cb_pop_front(cb_recv, TP);
        MOTIF_SPN_SFPU((ckernel::sfpu::motif_spn::fold<TP>));  // ((p0 + p1) + p2) ...
        copy_init(cb_k);
        copy_tile(cb_k, m, 1);      // D_m (column 0)
        copy_tile(cb_k, 3 + m, 2);  // E_m
        MOTIF_SPN_SFPU((ckernel::sfpu::motif_spn::mac_de));  // s D + E
        rsqrt_tile_init();
        rsqrt_tile(0);
        bcast_pack<cb_coef>();
    }
    cb_reserve_back(cb_coef, 1);
    tile_regs_acquire();
    copy_init(cb_k);
    copy_tile(cb_k, 6, 0);  // b
    bcast_pack<cb_coef>();
    cb_pop_front(cb_k, 7);

    // ---- h tiles, 2 per DEST round ----
    cb_wait_front(cb_coef, 4);
    cb_wait_front(cb_g, n);
    cb_wait_front(cb_u, n);
    if constexpr (DEBUG == 1) {
        for (uint32_t c = 0; c < 4; ++c) {
            cb_reserve_back(cb_out, 1);
            tile_regs_acquire();
            copy_init(cb_coef);
            copy_tile(cb_coef, c, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
        cb_pop_front(cb_coef, 4);
        cb_pop_front(cb_g, n);
        cb_pop_front(cb_u, n);
        return;
    }
    pack_reconfig_data_format(cb_out);
    for (uint32_t j0 = 0; j0 < n; j0 += 2) {
        const uint32_t nb = (n - j0) < 2 ? (n - j0) : 2;
        cb_reserve_back(cb_out, nb);
        tile_regs_acquire();
        copy_init(cb_coef);
        for (uint32_t c = 0; c < 4; ++c) {
            copy_tile(cb_coef, c, c);
        }
        copy_init(cb_g);
        for (uint32_t jj = 0; jj < nb; ++jj) {
            copy_tile(cb_g, j0 + jj, 4 + 2 * jj);
        }
        copy_init(cb_u);
        for (uint32_t jj = 0; jj < nb; ++jj) {
            copy_tile(cb_u, j0 + jj, 5 + 2 * jj);
        }
        if (nb == 2) {
            MOTIF_SPN_SFPU((ckernel::sfpu::motif_spn::horner<2>));
        } else {
            MOTIF_SPN_SFPU((ckernel::sfpu::motif_spn::horner<1>));
        }
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        for (uint32_t jj = 0; jj < nb; ++jj) {
            typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(4 + 2 * jj);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t jj = 0; jj < nb; ++jj) {
            pack_tile(4 + 2 * jj, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, nb);
    }
    cb_pop_front(cb_coef, 4);
    cb_pop_front(cb_g, n);
    cb_pop_front(cb_u, n);
}
