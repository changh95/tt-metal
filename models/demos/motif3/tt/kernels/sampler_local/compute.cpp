// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused device-sampler per-chip stage (Phase F, F4; tt/kernels/sampler_local.py), compute.
//
// Every value is computed with the LLK calls of the ttnn op it replaces (tt/sampling.py MotifDeviceSampler, the
// _stage_norm / _stage_counters / _stage_gumbel / _stage_uniform chain), on the same operands, in the same operand
// order: binary_ng SFPU binaries (copy both operands to DEST with UnpackToDestFp32, BINARY_SFPU_OP(a, b, out), post
// activations), the unary SFPU ops (exp_tile<false>, log1p_tile<false>, the RHS activations negative_tile + log_tile),
// and the accurate fp32 reductions (SFPU fold over the tiles in vocab order: acc = t0, acc = add_binary_tile(acc, tk);
// then sfpu_reduce). The ops chain stores every intermediate exactly (fp32 / int32 DRAM tensors); here they stay in
// DEST or in fp32 CBs, also exactly. Column broadcasts of [32, 1] operands are the software broadcast binary_ng uses
// on Blackhole for fp32 / int32 (the dataflow kernel fills the rows). Max / min / where / lt only select values, so
// their cross-core split cannot change a bit.
//
// PROD (q; n local tiles), in the order H(0), A(0..n-1), H(1..n-1), S(0..n-1), max, C:
//   H  h1 = lowbias32(K ^ c1); y = (h1 ^ c2) * MIX; h2 = y ^ (y >> 16); A = f32(h1 >> 8); B = f32(h2 >> 9);
//      -W = (A + (B + 0.5) 2^-23) (-2^-24); F = log1p(-W); t = log(-F) (negative_tile, log_tile) -> CB_T
//   A  d = x - M (sub_binary_tile<NE>), a = d * (1 / T) (mul_binary_tile) -> CB_D, CB_A; exp(a), exp(d) -> CB_EA, CB_ED
//   S  s = a - t (sub_binary_tile<NE>) -> CB_S; the local max of s (fold + sfpu_reduce MAX ROW) -> CB_W
//   C  nm = s < m; li = min(where(nm, 2^24, id)); dw = max(where(nm, -BIG, d)) -> CB_W2
// ACC x: face x % 4 of acc = e_0, acc = add_binary_tile(acc, e_k) (k = 1 .. NT-1; the face-split fold below) -> CB_S2;
//   the row master (x % 4 == 0): the assembled fold tile -> sfpu_reduce SUM ROW -> CB_S2B; part = e0 * [s2 | 0]
// AGG: u = f32(lowbias32(c1) >> 8) * 2^-24 -> CB_U; m = max over the producers' maxima (REDUCE_COL) -> CB_GR;
//   li = min, dw = max -> CB_GR; q = e0 * [m | 0], e0 * [gid | 0], e0 * [dw | 0] -> CB_OUT
//
// CT: 0 ROLE, 1 NT, 2 TPP, 3 NP, 4 GX, 5 NPT, 6 bits of -BIG, 7 bits of the id sentinel 2^24

#include <cstdint>

#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/binary_shift.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "common.h"

using namespace motif_sl;

namespace {
constexpr uint32_t I32 = static_cast<uint32_t>(DataFormat::Int32);
constexpr uint32_t F32 = static_cast<uint32_t>(DataFormat::Float32);
constexpr uint32_t LOWBIAS_M1 = 0x21F0AAADu;
constexpr uint32_t LOWBIAS_M2 = 0x735A2D97u;
constexpr uint32_t GUMBEL_MIX = 0x846CA68Bu;
constexpr uint32_t HALF_F = 0x3F000000u;    // 0.5f
constexpr uint32_t P2M23_F = 0x34000000u;   // 2^-23
constexpr uint32_t NP2M24_F = 0xB3800000u;  // -2^-24
constexpr uint32_t P2M24_F = 0x33800000u;   // 2^-24 (U24)

inline void fill_i(uint32_t d, uint32_t v) {
    fill_tile_init();
    fill_tile_int<DataFormat::Int32>(d, v);
}
inline void fill_f(uint32_t d, uint32_t bits) {
    fill_tile_init();
    fill_tile_bitcast(d, bits);
}
inline void lrs(uint32_t a, uint32_t b, uint32_t o) {
    binary_shift_tile_init();
    binary_logical_right_shift_tile<DataFormat::Int32>(a, b, o);
}
inline void bxor(uint32_t a, uint32_t b, uint32_t o) {
    binary_bitwise_tile_init();
    bitwise_xor_binary_tile<DataFormat::Int32>(a, b, o);
}
inline void imul(uint32_t a, uint32_t b, uint32_t o) {
    mul_int_tile_init<DataFormat::Int32>();
    mul_int_tile<DataFormat::Int32>(a, b, o);
}
inline void i2f(uint32_t d) {
    typecast_tile_init<I32, F32>();
    typecast_tile<I32, F32>(d);
}

// tt/sampling.py MotifDeviceSampler._lowbias32 on DEST tile h (int32), scratch t1 / t2:
//   for (shift, mul) in ((16, M1), (15, M2), (15, None)): s = h >> shift; h = h ^ s; h = h * mul
inline void lowbias32(uint32_t h, uint32_t t1, uint32_t t2) {
    fill_i(t2, 16);
    lrs(h, t2, t1);
    bxor(h, t1, h);
    fill_i(t2, LOWBIAS_M1);
    imul(h, t2, h);
    fill_i(t2, 15);
    lrs(h, t2, t1);
    bxor(h, t1, h);
    fill_i(t2, LOWBIAS_M2);
    imul(h, t2, h);
    fill_i(t2, 15);
    lrs(h, t2, t1);
    bxor(h, t1, h);
}

// Face-split fold step (ACC cores): for j = j0 .. nsub-1, face 0 of DEST tile acc += face j of DEST tile src, each as
// calculate_sfpu_binary<ADD> does it per element (load in0, load in1, in0 + in1, store), so every element of face 0
// sees exactly the add_binary_tile chain of the accurate fp32 sum over the e tiles in k order.
}  // namespace

#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void sl_fold_faces(const uint32_t acc, const uint32_t src, const uint32_t, const uint32_t j0, const uint32_t nsub) {
    constexpr uint32_t TS = 32;  // sfpi rows per DEST tile (8 per face)
    for (uint32_t j = j0; j < nsub; ++j) {
#pragma GCC unroll 8
        for (uint32_t d = 0; d < 8; ++d) {
            sfpi::vFloat in0 = sfpi::dst_reg[acc * TS + d];
            sfpi::vFloat in1 = sfpi::dst_reg[src * TS + j * 8 + d];
            sfpi::dst_reg[acc * TS + d] = in0 + in1;
        }
    }
}
}  // namespace ckernel::sfpu
#endif

namespace {

inline void fold_faces(uint32_t acc, uint32_t src, uint32_t j0, uint32_t nsub) {
    MATH((_llk_math_eltwise_binary_sfpu_params_(ckernel::sfpu::sl_fold_faces, acc, src, acc, VectorMode::None, j0,
                                                 nsub)));
}

template <uint32_t from, uint32_t to>
inline void src(uint32_t) {
    reconfig_data_format_srca(from, to);
    copy_init(to);
}

// PROD, the Gumbel hash of vocab tile i up to the score's RHS: h1 = lowbias32(K ^ c1); y = (h1 ^ c2) * MIX;
// h2 = y ^ (y >> 16); A = f32(h1 >> 8); B = f32(h2 >> 9); -W = (A + (B + 0.5) 2^-23) (-2^-24); F = log1p(-W);
// t = log(-F) (the subtract's RHS activations NEG, LOG) -> CB_T. Expects the srcA format of CB_GK (int32).
inline void hash_tile(uint32_t i) {
    cb_wait_front(CB_GK, i + 1);
    cb_wait_front(CB_CI, 2);
    tile_regs_acquire();
    copy_init(CB_GK);
    copy_tile(CB_GK, i, 0);
    copy_init(CB_CI);
    copy_tile(CB_CI, 0, 1);
    bxor(0, 1, 0);        // K ^ c1
    lowbias32(0, 1, 2);   // h1
    copy_tile(CB_CI, 1, 1);
    bxor(0, 1, 1);        // h1 ^ c2
    fill_i(2, GUMBEL_MIX);
    imul(1, 2, 1);        // y
    fill_i(2, 16);
    lrs(1, 2, 3);
    bxor(1, 3, 1);        // h2 = y ^ (y >> 16)
    fill_i(2, 8);
    lrs(0, 2, 3);
    i2f(3);               // A = f32(h1 >> 8)
    fill_i(2, 9);
    lrs(1, 2, 4);
    i2f(4);               // B = f32(h2 >> 9)
    fill_f(2, HALF_F);
    add_binary_tile_init();
    add_binary_tile<ckernel::DstRoundingMode::NearestEven>(4, 2, 4);
    binop_with_scalar_tile_init();
    mul_unary_tile(4, P2M23_F);  // Bf = (B + 0.5) 2^-23
    add_binary_tile_init();
    add_binary_tile<ckernel::DstRoundingMode::NearestEven>(3, 4, 3);
    binop_with_scalar_tile_init();
    mul_unary_tile(3, NP2M24_F);  // -W = (A + Bf) (-2^-24)
    log1p_tile_init<false>();
    log1p_tile<false>(3);  // F = log1p(-W)
    negative_tile_init();
    negative_tile(3);
    log_tile_init();
    log_tile(3);  // log(-F) (the subtract's RHS activations)
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(3, CB_T);
    tile_regs_release();
    cb_push_back(CB_T, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t TPP = get_compile_time_arg_val(2);
    constexpr uint32_t NP = get_compile_time_arg_val(3);
    constexpr uint32_t GX = get_compile_time_arg_val(4);
    constexpr uint32_t NPT = get_compile_time_arg_val(5);
    constexpr uint32_t NEG_BIG = get_compile_time_arg_val(6);
    constexpr uint32_t ID_SENT = get_compile_time_arg_val(7);

    if constexpr (ROLE == ROLE_PROD) {
        const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
        const uint32_t k0 = q * TPP;
        const uint32_t n = (k0 + TPP <= NT) ? TPP : NT - k0;
        compute_kernel_hw_startup(CB_GK, CB_T);
        cb_reserve_back(CB_T, n);
        cb_reserve_back(CB_D, n);
        cb_reserve_back(CB_A, n);
        cb_reserve_back(CB_S, n);
        // order: H(0) (needs only the counters: overlaps the AGG's M broadcast), A(0..n-1) (feeds the ACC cores),
        // H(1..n-1), S(0..n-1)
        hash_tile(0);
        cb_wait_front(CB_X, n);
        cb_wait_front(CB_CF, 2);
        for (uint32_t i = 0; i < n; ++i) {
            // ---- A: d = x - M (sub_binary_tile<NE>), a = d * (1 / T); exp(a), exp(d) (the normaliser rows) ------
            tile_regs_acquire();
            src<CB_T, CB_X>(0);
            copy_tile(CB_X, i, 0);
            src<CB_X, CB_CF>(0);
            copy_tile(CB_CF, 0, 1);
            sub_binary_tile_init();
            sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);  // d = x - M
            copy_tile(CB_CF, 1, 2);
            mul_binary_tile_init();
            mul_binary_tile(0, 2, 1);  // a = d * (1 / T)
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_D);
            pack_tile(1, CB_A);
            tile_regs_release();
            cb_push_back(CB_D, 1);
            cb_push_back(CB_A, 1);
            cb_wait_front(CB_A, i + 1);
            cb_wait_front(CB_D, i + 1);
            cb_reserve_back(CB_EA, 1);
            cb_reserve_back(CB_ED, 1);
            tile_regs_acquire();
            copy_init(CB_A);
            copy_tile(CB_A, i, 0);
            exp_tile_init<false>();
            exp_tile<false>(0);
            copy_init(CB_D);
            copy_tile(CB_D, i, 1);
            exp_tile_init<false>();
            exp_tile<false>(1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_EA);
            pack_tile(1, CB_ED);
            tile_regs_release();
            cb_push_back(CB_EA, 1);
            cb_push_back(CB_ED, 1);
        }
        for (uint32_t i = 1; i < n; ++i) {
            src<CB_D, CB_GK>(0);
            hash_tile(i);
        }
        // ---- S: s = a - log(-F) (the subtract with the RHS activations) ---------------------------------------------
        cb_wait_front(CB_T, n);
        for (uint32_t i = 0; i < n; ++i) {
            tile_regs_acquire();
            src<CB_GK, CB_A>(0);
            copy_tile(CB_A, i, 0);
            copy_init(CB_T);
            copy_tile(CB_T, i, 1);
            sub_binary_tile_init();
            sub_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_S);
            tile_regs_release();
            cb_push_back(CB_S, 1);
        }
        // local max of s (the accurate fp32 max reduce: fold, then sfpu_reduce ROW)
        cb_wait_front(CB_S, n);
        cb_reserve_back(CB_W, 1);
        tile_regs_acquire();
        copy_init(CB_S);
        copy_tile(CB_S, 0, 0);
        binary_max_tile_init();
        for (uint32_t i = 1; i < n; ++i) {
            copy_tile(CB_S, i, 1);
            binary_max_tile(0, 1, 0);
        }
        sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
        sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_W);
        tile_regs_release();
        cb_push_back(CB_W, 1);
        // ---- C: the lowest id and the raw x - M at the global max ------------------------------------------------
        cb_wait_front(CB_MS, 1);
        cb_wait_front(CB_IO, n);
        cb_reserve_back(CB_W2, 2);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            copy_init(CB_S);
            copy_tile(CB_S, i, 0);
            copy_init(CB_MS);
            copy_tile(CB_MS, 0, 1);
            lt_binary_tile_init();
            lt_binary_tile(0, 1, 2);  // nm = s < m
            copy_init(CB_IO);
            copy_tile(CB_IO, i, 3);
            fill_f(4, ID_SENT);
            where_tile_init();
            where_tile<DataFormat::Float32>(2, 4, 3, i == 0 ? 6 : 5);
            if (i > 0) {
                binary_min_tile_init();
                binary_min_tile(6, 5, 6);
            }
            copy_init(CB_D);
            copy_tile(CB_D, i, 3);
            fill_f(4, NEG_BIG);
            where_tile_init();
            where_tile<DataFormat::Float32>(2, 4, 3, i == 0 ? 7 : 5);
            if (i > 0) {
                binary_max_tile_init();
                binary_max_tile(7, 5, 7);
            }
        }
        sfpu_reduce_init<PoolType::MIN, DataFormat::Float32>();
        sfpu_reduce<PoolType::MIN, DataFormat::Float32, ReduceDim::REDUCE_ROW>(6, 1, 1);
        sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
        sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_ROW>(7, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(6, CB_W2);
        pack_tile(7, CB_W2);
        tile_regs_release();
        cb_push_back(CB_W2, 2);
        cb_pop_front(CB_MS, 1);
        cb_pop_front(CB_X, n);
        cb_pop_front(CB_CF, 2);
        cb_pop_front(CB_GK, n);
        cb_pop_front(CB_CI, 2);
        cb_pop_front(CB_IO, n);
        cb_pop_front(CB_D, n);
        cb_pop_front(CB_A, n);
        cb_pop_front(CB_S, n);
        cb_pop_front(CB_T, n);
    } else if constexpr (ROLE == ROLE_ACC) {
        // face f of the fold acc = e_0, acc = add_binary_tile(acc, e_k), k = 1 .. NT-1: four e tiles' face f per
        // ring page (faces 0..3 of the page = k = 4p .. 4p+3), folded into face 0 of DEST tile 0 in k order with the
        // add_binary_tile per-element sequence (load acc, load e_k, add, store)
        constexpr uint32_t NSP = (NT + 3) / 4;
        const bool master = (get_absolute_logical_x() % 4) == 0;
        compute_kernel_hw_startup(CB_RING, CB_S2);
        cb_reserve_back(CB_S2, 1);
        copy_init(CB_RING);
        add_binary_tile_init();
        tile_regs_acquire();
        for (uint32_t p = 0; p < NSP; ++p) {
            const uint32_t nsub = (4 * p + 4 <= NT) ? 4 : NT - 4 * p;
            cb_wait_front(CB_RING, 1);
            if (p == 0) {
                copy_tile(CB_RING, 0, 0);  // face 0 of tile 0 = e_0 (the fold's first tile, unpacked as is)
                fold_faces(0, 0, 1, nsub);
            } else {
                copy_tile(CB_RING, 0, 1);
                fold_faces(0, 1, 0, nsub);
            }
            cb_pop_front(CB_RING, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_S2);
        tile_regs_release();
        cb_push_back(CB_S2, 1);
        if (master) {
            // the assembled fold tile -> sfpu_reduce SUM ROW; part = e0 * bcast(s2)
            cb_wait_front(CB_AF, 1);
            cb_reserve_back(CB_S2B, 1);
            tile_regs_acquire();
            copy_init(CB_AF);
            copy_tile(CB_AF, 0, 0);
            sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
            sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_S2B);
            tile_regs_release();
            cb_push_back(CB_S2B, 1);
            cb_pop_front(CB_AF, 1);
            cb_wait_front(CB_E0, 1);
            cb_wait_front(CB_SF, 1);
            cb_reserve_back(CB_OUT, 1);
            tile_regs_acquire();
            copy_init(CB_E0);
            copy_tile(CB_E0, 0, 0);
            copy_init(CB_SF);
            copy_tile(CB_SF, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_OUT);
            tile_regs_release();
            cb_push_back(CB_OUT, 1);
            cb_pop_front(CB_E0, 1);
            cb_pop_front(CB_SF, 1);
        }
    } else {  // ROLE_AGG
        compute_kernel_hw_startup(CB_RAWI, CB_U);
        // u = f32(lowbias32(c1) >> 8) * 2^-24 (column 0; _stage_counters + _stage_uniform)
        cb_wait_front(CB_RAWI, 1);
        cb_reserve_back(CB_U, 1);
        tile_regs_acquire();
        copy_init(CB_RAWI);
        copy_tile(CB_RAWI, 0, 0);
        lowbias32(0, 1, 2);
        fill_i(2, 8);
        lrs(0, 2, 1);
        i2f(1);
        fill_f(2, P2M24_F);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(1, CB_U);
        tile_regs_release();
        cb_push_back(CB_U, 1);
        cb_pop_front(CB_RAWI, 1);
        // m = max over the producers' row maxima (rows = producers, columns = lanes; exact)
        cb_wait_front(CB_PM, NPT);
        cb_reserve_back(CB_GR, 1);
        tile_regs_acquire();
        src<CB_RAWI, CB_PM>(0);
        copy_tile(CB_PM, 0, 0);
        binary_max_tile_init();
        for (uint32_t t = 1; t < NPT; ++t) {
            copy_tile(CB_PM, t, 1);
            binary_max_tile(0, 1, 0);
        }
        sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
        sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_COL>(0, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_GR);
        tile_regs_release();
        cb_push_back(CB_GR, 1);
        // li = min, dw = max over the producers
        cb_wait_front(CB_PL, NPT);
        cb_wait_front(CB_PD, NPT);
        cb_reserve_back(CB_GR, 2);
        tile_regs_acquire();
        copy_init(CB_PL);
        copy_tile(CB_PL, 0, 0);
        binary_min_tile_init();
        for (uint32_t t = 1; t < NPT; ++t) {
            copy_tile(CB_PL, t, 1);
            binary_min_tile(0, 1, 0);
        }
        sfpu_reduce_init<PoolType::MIN, DataFormat::Float32>();
        sfpu_reduce<PoolType::MIN, DataFormat::Float32, ReduceDim::REDUCE_COL>(0, 1, 1);
        copy_init(CB_PD);
        copy_tile(CB_PD, 0, 2);
        binary_max_tile_init();
        for (uint32_t t = 1; t < NPT; ++t) {
            copy_tile(CB_PD, t, 3);
            binary_max_tile(2, 3, 2);
        }
        sfpu_reduce_init<PoolType::MAX, DataFormat::Float32>();
        sfpu_reduce<PoolType::MAX, DataFormat::Float32, ReduceDim::REDUCE_COL>(2, 1, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_GR);
        pack_tile(2, CB_GR);
        tile_regs_release();
        cb_push_back(CB_GR, 2);
        // q = concat([e0 * m, e0 * gid, e0 * dw]) (the three column-broadcast multiplies; the dataflow kernel put m,
        // gid = li + chip_off and dw in column 0 of the operand tiles, with the sign 0 * v takes elsewhere)
        cb_wait_front(CB_SF, 3);
        cb_wait_front(CB_E0, 1);
        cb_reserve_back(CB_OUT, 3);
        tile_regs_acquire();
        copy_init(CB_E0);
        copy_tile(CB_E0, 0, 0);
        copy_tile(CB_E0, 0, 2);
        copy_tile(CB_E0, 0, 4);
        copy_init(CB_SF);
        copy_tile(CB_SF, 0, 1);
        copy_tile(CB_SF, 1, 3);
        copy_tile(CB_SF, 2, 5);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        mul_binary_tile(2, 3, 2);
        mul_binary_tile(4, 5, 4);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        pack_tile(2, CB_OUT);
        pack_tile(4, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 3);
        cb_pop_front(CB_SF, 3);
        cb_pop_front(CB_E0, 1);
        cb_pop_front(CB_PM, NPT);
        cb_pop_front(CB_PL, NPT);
        cb_pop_front(CB_PD, NPT);
    }
}
