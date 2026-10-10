// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused shared-expert tail (Phase F, F3; tt/kernels/shared_tail.py), compute of the APPLY cores.
//
// The B5 apply program (shared_polynorm/apply_compute.cpp) split over the 5 APPLY cores: core j < 3 computes the per-row
// coefficient tile of moment j, core 3 the b tile, with exactly apply_compute.cpp's LLK sequence for that tile (the TP
// fold, s D + E, rsqrt, the finite guard and the exact 0 + x column broadcast, packed fp32); the writer sends it to slot
// j of every APPLY core's cb_coef. Then every core j evaluates h tile j from the 4 coefficient tiles with
// apply_compute.cpp's Horner chain (per element the same SFPI multiply-adds in the same order) and the same
// typecast_tile<Float32, Float16_b> before the bf16 pack. Each value is bitwise the B5 apply's.
//
// CT: 0 cb_recv, 1 cb_k, 2 cb_g, 3 cb_u, 4 cb_coef, 5 cb_out, 6 TP, 7 cb_co, 8 AY0

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
namespace ckernel::sfpu::motif_stail {
using sfpi::dst_reg;
using sfpi::vFloat;
// (shared_polynorm/apply_compute.cpp motif_spn::fold / mac_de / finite_or_zero_t0 / horner, unchanged)
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
}  // namespace ckernel::sfpu::motif_stail
#endif
#define MOTIF_STAIL_SFPU(fn)                                                                                    \
    MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::motif_stail::noop_init))); \
    MATH((_llk_math_eltwise_unary_sfpu_params_(fn, 0, VectorMode::RC_custom)))

template <uint32_t cb_co>
inline void bcast_pack() {
    MOTIF_STAIL_SFPU((ckernel::sfpu::motif_stail::finite_or_zero_t0));
    fill_tile_init();
    fill_tile(1, 0.0f);
    sfpu_bcast_col_init();
    sfpu_add_bcast_col(1, 0);  // 0 + a: exact
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1, cb_co);
    tile_regs_release();
    cb_push_back(cb_co, 1);
}

void kernel_main() {
    constexpr uint32_t cb_recv = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr uint32_t cb_u = get_compile_time_arg_val(3);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t TP = get_compile_time_arg_val(6);
    constexpr uint32_t cb_co = get_compile_time_arg_val(7);
    constexpr uint32_t AY0 = get_compile_time_arg_val(8);
    const uint32_t j = static_cast<uint32_t>(get_absolute_logical_y()) - AY0;

    compute_kernel_hw_startup(cb_recv, cb_co);
    cb_wait_front(cb_k, 7);
    if (j < 3) {  // the per-row coefficient of moment j (apply_compute.cpp's loop body, m = j)
        const uint32_t m = j;
        cb_reserve_back(cb_co, 1);
        cb_wait_front(cb_recv, TP);
        tile_regs_acquire();
        copy_init(cb_recv);
        for (uint32_t k = 0; k < TP; ++k) {
            copy_tile(cb_recv, k, k);
        }
        cb_pop_front(cb_recv, TP);
        MOTIF_STAIL_SFPU((ckernel::sfpu::motif_stail::fold<TP>));
        copy_init(cb_k);
        copy_tile(cb_k, m, 1);
        copy_tile(cb_k, 3 + m, 2);
        MOTIF_STAIL_SFPU((ckernel::sfpu::motif_stail::mac_de));
        rsqrt_tile_init();
        rsqrt_tile(0);
        bcast_pack<cb_co>();
    } else if (j == 3) {  // b
        cb_reserve_back(cb_co, 1);
        tile_regs_acquire();
        copy_init(cb_k);
        copy_tile(cb_k, 6, 0);
        bcast_pack<cb_co>();
    }
    cb_pop_front(cb_k, 7);

    // ---- h tile j ----
    cb_wait_front(cb_coef, 4);
    cb_wait_front(cb_g, 1);
    cb_wait_front(cb_u, 1);
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_init(cb_coef);
    for (uint32_t c = 0; c < 4; ++c) {
        copy_tile(cb_coef, c, c);
    }
    copy_init(cb_g);
    copy_tile(cb_g, 0, 4);
    copy_init(cb_u);
    copy_tile(cb_u, 0, 5);
    MOTIF_STAIL_SFPU((ckernel::sfpu::motif_stail::horner<1>));
    typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
    typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(4);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(4, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_coef, 4);
    cb_pop_front(cb_g, 1);
    cb_pop_front(cb_u, 1);
}
