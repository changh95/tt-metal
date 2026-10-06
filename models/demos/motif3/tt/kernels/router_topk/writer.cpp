// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode router tail (B4, docs/OPTIMIZATION_PLAN.md §3.3 "A5 and B4"), writer (RISCV_0).
//
// Worker q owns gathered token row q (tile row tr = q / 32, row rr = q % 32).
// 1. Top-K of the biased scores (CB_O, from the compute kernel; column j at float j) on order-preserving uint32 keys
//    (sign flip): an exact integer compare of the fp32 values. Ranked by (key descending, expert id ascending), so on an
//    exact tie the lower expert id is selected (deterministic; ttnn.topk's tie order is its sort network's).
// 2. The routing weights from the *unbiased* scores (CB_S base: the reader's copy, consumed by the compute kernel and
//    never overwritten in this program) in fp32 (soft float on this RISC, IEEE round to nearest even):
//    den = s_1 + ... + s_K (rank order), r = 1 / (den + 1e-20), w_k = s_k r (times SCALE unless it is 1.0).
// 3. Output w_loc [1, E_LOC, M, 1] fp32 TILE interleaved: tile (e, tr) = e R + tr, row rr: column 0 = the weight of
//    local expert e (global id base + e; 0 when not selected), columns 1..31 = 0 (both 64 B halves of the row are
//    written, so the whole tile is defined once all 32 row workers ran). base = element 0 of ids tile 0 (CB_ID).
//    WRITE_IDX: also the K selected ids (rank order) to idx [1, 1, M, K] uint32 ROW_MAJOR (page = row q).
//
// CT: 0 cb_s, 1 cb_o, 2 cb_id, 3 cb_st, 4 NE, 5 K, 6 E_LOC, 7 R, 8 GX, 9 scale bits (fp32), 10 WRITE_IDX,
//     11.. TensorAccessorArgs(w_loc), TensorAccessorArgs(idx) (the wrapper always passes idx and WRITE_IDX = 1)
// common RT: 0 w_loc_addr, 1 idx_addr (0 when not written)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
inline float as_float(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } c;
    c.u = u;
    return c.f;
}
inline uint32_t as_bits(float f) {
    union {
        float f;
        uint32_t u;
    } c;
    c.f = f;
    return c.u;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_s = get_compile_time_arg_val(0);
    constexpr uint32_t cb_o = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_st = get_compile_time_arg_val(3);
    constexpr uint32_t NE = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t E_LOC = get_compile_time_arg_val(6);
    constexpr uint32_t R = get_compile_time_arg_val(7);
    constexpr uint32_t GX = get_compile_time_arg_val(8);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(9);
    constexpr uint32_t WRITE_IDX = get_compile_time_arg_val(10);
    constexpr auto w_args = TensorAccessorArgs<11>();
    static_assert(K >= 1 && K <= 16, "router_topk: K must be in [1, 16]");

    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t tr = q >> 5;
    const uint32_t rr = q & 31;
    const uint32_t row_off = ((rr >> 4) * 2) * 1024 + (rr & 15) * 64;

    cb_wait_front(cb_o, 1);
    cb_wait_front(cb_id, 1);
    invalidate_l1_cache();
    asm volatile("fence" ::: "memory");  // and a compiler barrier: no L1 load is hoisted above the waits
    const uint32_t* bk = reinterpret_cast<const uint32_t*>(get_read_ptr(cb_o));
    const uint32_t* sv = reinterpret_cast<const uint32_t*>(get_read_ptr(cb_s));  // never advanced on this RISC

    // ---- 1. top-K on order-preserving keys; ties: lower id first ----
    uint32_t key[K];
    uint32_t ids[K];
    uint32_t n = 0;
    for (uint32_t i = 0; i < NE; ++i) {
        const uint32_t u = bk[i];
        const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        uint32_t j;
        if (n < K) {
            j = n++;
        } else if (k > key[K - 1]) {
            j = K - 1;
        } else {
            continue;
        }
        while (j > 0 && key[j - 1] < k) {
            key[j] = key[j - 1];
            ids[j] = ids[j - 1];
            --j;
        }
        key[j] = k;
        ids[j] = i;
    }

    // ---- 2. weights from the unbiased scores (fp32, rank order) ----
    float s[K];
    float den = 0.0f;
    for (uint32_t j = 0; j < K; ++j) {
        s[j] = as_float(sv[ids[j]]);
        den = den + s[j];
    }
    const float r = 1.0f / (den + 1e-20f);
    const uint32_t base = static_cast<uint32_t>(as_float(reinterpret_cast<const uint32_t*>(get_read_ptr(cb_id))[0]));

    // ---- 3. stage the E_LOC row halves (column 0 = weight) + a zero half, then write ----
    cb_reserve_back(cb_st, 1);
    const uint32_t st = get_write_ptr(cb_st);
    uint32_t* stw = reinterpret_cast<uint32_t*>(st);
    for (uint32_t i = 0; i < 16 * (E_LOC + 1) + 16; ++i) {
        stw[i] = 0;
    }
    for (uint32_t j = 0; j < K; ++j) {
        const uint32_t e = ids[j] - base;  // unsigned: ids below base wrap to >= E_LOC
        if (e < E_LOC) {
            float w = s[j] * r;
            if constexpr (SCALE_BITS != 0x3f800000u) {
                w = w * as_float(SCALE_BITS);
            }
            stw[16 * e] = as_bits(w);
        }
    }
    asm volatile("" ::: "memory");  // the staged stores land before the NoC reads them (write-through L1 cache)
    const uint32_t zero_half = st + 64 * E_LOC;
    const uint32_t idx_src = st + 64 * (E_LOC + 1);
    constexpr uint32_t tb = 4096;
    const auto ws = TensorAccessor(w_args, w_addr, tb);
    for (uint32_t e = 0; e < E_LOC; ++e) {
        const uint32_t page = e * R + tr;
        noc_async_write(st + 64 * e, ws.get_noc_addr(page, row_off), 64);
        noc_async_write(zero_half, ws.get_noc_addr(page, row_off + 1024), 64);
    }
    if constexpr (WRITE_IDX != 0) {
        constexpr auto i_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
        const uint32_t i_addr = get_common_arg_val<uint32_t>(1);
        const auto is = TensorAccessor(i_args, i_addr, K * 4);
        for (uint32_t j = 0; j < K; ++j) {
            stw[16 * (E_LOC + 1) + j] = ids[j];
        }
        asm volatile("" ::: "memory");
        noc_async_write(idx_src, is.get_noc_addr(q, 0), K * 4);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_o, 1);
    cb_pop_front(cb_id, 1);
}
