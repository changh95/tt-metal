// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused grouped MoE PolyNorm (B3; prototype logs/opt/phaseA/M10 "V2"), compute (custom SFPI moment and Horner
// loops; every operand is unpacked to DEST in exact fp32, every product / sum is an SFPU fp32 op).
//
// Per worker: expert e, rank k, n gate tiles per tile row, R tile rows.
// Phase 1 (per tile row): elementwise fp32 SFPU accumulation over the n gate tiles of g^2, g^4, g^6 (in tile order,
//   batches of 5 tiles), then the SFPU fp32 row reduce (the sum lands in column 0) -> 3 partial tiles -> CB_PART (the
//   writer sends them to the group).
// Phase 3 (per tile row): sum the G partials in rank order 0..G-1 (the same order on every member, so every member
//   gets bitwise the same moments), a_m = rsqrt(s_m D_m + E_m) (D_m = 1 / (I c_m^2), E_m = eps / c_m^2, i.e.
//   c_m rsqrt(mean + eps)); fold the routing weight w (column 0 of the w tile): A_m = w a_m, B = w b; broadcast the
//   four column vectors to full tiles (fill 0 + add_bcast_col) -> CB_COEF; then per gate tile
//   h = (((A3 g + A2) g + A1) g + B) u   -> bf16 -> CB_OUT.
// The x0.5 PolyNorm output scale is folded into W_down by the caller (as on the composite path); b is the clamped
// bias (+-0.5) of the routed experts.
//
// CT: 0 cb_g, 1 cb_u, 2 cb_w, 3 cb_part, 4 cb_recv, 5 cb_coef, 6 cb_out, 7 cb_k, 8 n, 9 R, 10 G
// CB_K: one fp32 tile, element 16 j = constant j (D1 D2 D3 E1 E2 E3 b; moment order g^2, g^4, g^6) as fp32 bits.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
namespace ckernel::sfpu::motif_pn {
using sfpi::dst_reg;
using sfpi::vFloat;
// phase 1: DEST tiles 0..NT-1 = gate tiles, 5 / 6 / 7 = running sums of g^2 / g^4 / g^6 (elementwise, fp32)
template <int NT, bool FIRST>
inline void moments() {
#pragma GCC unroll 1
    for (int i = 0; i < 32; ++i) {
        vFloat a2, a4, a6;
        if constexpr (FIRST) {
            a2 = 0.0f;
            a4 = 0.0f;
            a6 = 0.0f;
        } else {
            a2 = dst_reg[5 * 32];
            a4 = dst_reg[6 * 32];
            a6 = dst_reg[7 * 32];
        }
#pragma GCC unroll 5
        for (int t = 0; t < NT; ++t) {
            vFloat g = dst_reg[t * 32];
            vFloat g2 = g * g;
            vFloat g4 = g2 * g2;
            a2 = a2 + g2;
            a4 = a4 + g4;
            a6 = a6 + g4 * g2;
        }
        dst_reg[5 * 32] = a2;
        dst_reg[6 * 32] = a4;
        dst_reg[7 * 32] = a6;
        dst_reg++;
    }
}
// phase 3: DEST 0..3 = A1, A2, A3, B (full tiles); tile pair (4 + 2t, 5 + 2t) = (g, u) -> h into 4 + 2t
template <int NT>
inline void horner() {
#pragma GCC unroll 1
    for (int i = 0; i < 32; ++i) {
        vFloat a1 = dst_reg[0], a2 = dst_reg[1 * 32], a3 = dst_reg[2 * 32], b = dst_reg[3 * 32];
#pragma GCC unroll 2
        for (int t = 0; t < NT; ++t) {
            vFloat g = dst_reg[(4 + 2 * t) * 32];
            vFloat r = a3 * g + a2;
            r = r * g + a1;
            r = r * g + b;
            dst_reg[(4 + 2 * t) * 32] = r * vFloat(dst_reg[(5 + 2 * t) * 32]);
        }
        dst_reg++;
    }
}
inline void noop_init() {}
}  // namespace ckernel::sfpu::motif_pn
#endif
#define MOTIF_PN_SFPU(fn)                                                                                \
    MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::motif_pn::noop_init))); \
    MATH((_llk_math_eltwise_unary_sfpu_params_(fn, 0, VectorMode::RC_custom)))

template <bool FIRST>
inline void moments_batch(uint32_t nb) {
    if (nb == 5) {
        MOTIF_PN_SFPU((ckernel::sfpu::motif_pn::moments<5, FIRST>));
    } else if (nb == 4) {
        MOTIF_PN_SFPU((ckernel::sfpu::motif_pn::moments<4, FIRST>));
    } else if (nb == 3) {
        MOTIF_PN_SFPU((ckernel::sfpu::motif_pn::moments<3, FIRST>));
    } else if (nb == 2) {
        MOTIF_PN_SFPU((ckernel::sfpu::motif_pn::moments<2, FIRST>));
    } else {
        MOTIF_PN_SFPU((ckernel::sfpu::motif_pn::moments<1, FIRST>));
    }
}

void kernel_main() {
    constexpr uint32_t cb_g = get_compile_time_arg_val(0);
    constexpr uint32_t cb_u = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_part = get_compile_time_arg_val(3);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(4);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t cb_k = get_compile_time_arg_val(7);
    constexpr uint32_t n = get_compile_time_arg_val(8);
    constexpr uint32_t R = get_compile_time_arg_val(9);
    constexpr uint32_t G = get_compile_time_arg_val(10);

    compute_kernel_hw_startup(cb_g, cb_part);

    // ---------------- phase 1: local moment partials ----------------
    cb_wait_front(cb_g, n * R);
    cb_reserve_back(cb_part, 3 * R);
    for (uint32_t tr = 0; tr < R; ++tr) {
        tile_regs_acquire();
        for (uint32_t j0 = 0; j0 < n; j0 += 5) {
            const uint32_t nb = (n - j0) < 5 ? (n - j0) : 5;
            copy_init(cb_g);
            for (uint32_t jj = 0; jj < nb; ++jj) {
                copy_tile(cb_g, tr * n + j0 + jj, jj);
            }
            if (j0 == 0) {
                moments_batch<true>(nb);
            } else {
                moments_batch<false>(nb);
            }
        }
        sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(5);
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(6);
        sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(7);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(5, cb_part);
        pack_tile(6, cb_part);
        pack_tile(7, cb_part);
        tile_regs_release();
    }
    cb_push_back(cb_part, 3 * R);

    // ---------------- per-expert constants ----------------
    cb_wait_front(cb_k, 1);
    uint32_t Db[3], Eb[3];
    for (uint32_t m = 0; m < 3; ++m) {
        Db[m] = read_tile_value(cb_k, 0, 16 * m);
        Eb[m] = read_tile_value(cb_k, 0, 16 * (3 + m));
    }
    const uint32_t bb = read_tile_value(cb_k, 0, 16 * 6);

    // ---------------- phase 3: group moments -> coefficients -> h ----------------
    cb_wait_front(cb_u, n * R);
    cb_wait_front(cb_w, R);
    cb_wait_front(cb_recv, 3 * R * G);
    for (uint32_t tr = 0; tr < R; ++tr) {
        // coefficients of this tile row
        cb_reserve_back(cb_coef, 4);
        tile_regs_acquire();
        copy_init(cb_recv);
        for (uint32_t m = 0; m < 3; ++m) {
            copy_tile(cb_recv, 3 * tr + m, m);  // slot 0
        }
        for (uint32_t p = 1; p < G; ++p) {
            copy_init(cb_recv);
            for (uint32_t m = 0; m < 3; ++m) {
                copy_tile(cb_recv, p * 3 * R + 3 * tr + m, 3 + m);
            }
            add_binary_tile_init();
            for (uint32_t m = 0; m < 3; ++m) {
                add_binary_tile(m, 3 + m, m);
            }
        }
        binop_with_scalar_tile_init();
        for (uint32_t m = 0; m < 3; ++m) {
            mul_unary_tile(m, Db[m]);
            add_unary_tile(m, Eb[m]);
        }
        rsqrt_tile_init();
        for (uint32_t m = 0; m < 3; ++m) {
            rsqrt_tile(m);  // a_(m+1)
        }
        copy_init(cb_w);
        copy_tile(cb_w, tr, 3);
        copy_tile(cb_w, tr, 4);
        mul_binary_tile_init();
        for (uint32_t m = 0; m < 3; ++m) {
            mul_binary_tile(m, 3, m);  // A_m = w a_m
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(4, bb);  // B = w b
        fill_tile_init();
        fill_tile(3, 0.0f);
        fill_tile(5, 0.0f);
        fill_tile(6, 0.0f);
        fill_tile(7, 0.0f);
        sfpu_bcast_col_init();
        sfpu_add_bcast_col(3, 0);  // A1
        sfpu_add_bcast_col(5, 1);  // A2
        sfpu_add_bcast_col(6, 2);  // A3
        sfpu_add_bcast_col(7, 4);  // B
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(3, cb_coef);
        pack_tile(5, cb_coef);
        pack_tile(6, cb_coef);
        pack_tile(7, cb_coef);
        tile_regs_release();
        cb_push_back(cb_coef, 4);

        // h tiles, 2 per DEST round (custom SFPI Horner)
        cb_wait_front(cb_coef, 4);
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
                copy_tile(cb_g, tr * n + j0 + jj, 4 + 2 * jj);
            }
            copy_init(cb_u);
            for (uint32_t jj = 0; jj < nb; ++jj) {
                copy_tile(cb_u, tr * n + j0 + jj, 5 + 2 * jj);
            }
            if (nb == 2) {
                MOTIF_PN_SFPU((ckernel::sfpu::motif_pn::horner<2>));
            } else {
                MOTIF_PN_SFPU((ckernel::sfpu::motif_pn::horner<1>));
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            for (uint32_t jj = 0; jj < nb; ++jj) {
                pack_tile(4 + 2 * jj, cb_out);
            }
            pack_reconfig_data_format(cb_coef);
            tile_regs_release();
            cb_push_back(cb_out, nb);
        }
        cb_pop_front(cb_coef, 4);
    }
    cb_pop_front(cb_recv, 3 * R * G);
    cb_pop_front(cb_g, n * R);
    cb_pop_front(cb_u, n * R);
    cb_pop_front(cb_w, R);
    cb_pop_front(cb_k, 1);
}
