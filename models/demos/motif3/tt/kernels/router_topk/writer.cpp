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
// 4. DESIGN-2 replica mode (REPL = 1, tt/replicas.py; host mirror replicas.assign): w_loc has E_LOC + NREP slots, the
//    NREP replica slots holding the experts of the chip table (the reader stages it in CB_ID). Every worker sends its
//    K ids and its live flag (lane mask column 0 != 0; LANE = 0: live) to worker 0's CB_G (64 B record per row) and
//    bumps worker 0's semaphore G. Worker 0 waits for the M records, runs the deterministic greedy assignment (every
//    chip computes the same one from the same records), writes where[384] / load[32] to the assign tensor
//    [1, 1, 1, 416] uint32 RM, and multicasts this chip's 32-bit keep mask (slot s computes iff its expert is active
//    and assigned here) to every worker's CB_K with semaphore K. A worker writes the weight of slot s only when bit s
//    is kept and its row is live (else +0), so w_loc is already lane-masked. Inactive rows still send their record.
//    Every spin is bounded (a lost increment gives wrong data, never a hang).
//
// CT: 0 cb_s, 1 cb_o, 2 cb_id, 3 cb_st, 4 NE, 5 K, 6 E_LOC, 7 R, 8 GX, 9 scale bits (fp32), 10 WRITE_IDX,
//     11.. TensorAccessorArgs(w_loc), TensorAccessorArgs(idx) (the wrapper always passes idx and WRITE_IDX = 1),
//     then REPL, LANE, NREP, M, cb_g, cb_k, cb_sc, sem_g, sem_k, coordinator NOC x, y, rect A (x0, y0, x1, y1, n),
//     rect B (x0, y0, x1, y1, n), then TensorAccessorArgs(assign) (a copy of idx's when REPL = 0)
// common RT: 0 w_loc_addr, 1 idx_addr (0 when not written), 2 assign_addr (REPL)

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

constexpr uint32_t SPIN_LIMIT = 200000000u;
constexpr uint32_t NONE = 0xFFFFu;  // tt/replicas.py NONE
constexpr uint32_t PEND = 0xFFFEu;  // active, not yet assigned
constexpr uint32_t NE_REP = 384;    // tt/replicas.py N_EXPERTS
constexpr uint32_t N_CHIPS = 32;

inline uint64_t mcast_addr(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t l1) {
    return noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, l1) : get_noc_multicast_addr(x1, y1, x0, y0, l1);
}

inline bool wait_min(volatile tt_l1_ptr uint32_t* sem, uint32_t v) {
    uint32_t it = 0;
    do {
        invalidate_l1_cache();
        if (*sem >= v) {
            return true;
        }
    } while (++it < SPIN_LIMIT);
    return false;
}

// DESIGN-2 replica mode, step 4 of the header: returns this chip's keep mask (bit s = slot s computes).
template <uint32_t K, uint32_t M, uint32_t E_LOC, uint32_t NREP, uint32_t cb_g, uint32_t cb_k, uint32_t cb_sc,
          uint32_t sem_g_id, uint32_t sem_k_id, uint32_t CX, uint32_t CY, uint32_t AX0, uint32_t AY0, uint32_t AX1,
          uint32_t AY1, uint32_t AN, uint32_t BX0, uint32_t BY0, uint32_t BX1, uint32_t BY1, uint32_t BN, typename AArgs>
uint32_t replica_keep(uint32_t q, const uint32_t* ids, uint32_t live, uint32_t base,
                      const volatile tt_l1_ptr uint32_t* idw, uint32_t rec_l1, const AArgs& a_args, uint32_t a_addr) {
    // 1. this row's record -> worker 0's CB_G (64 B at q * 64): K ids, padding, word 15 = live
    volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rec_l1);
    for (uint32_t j = 0; j < 15; ++j) {
        rec[j] = j < K ? ids[j] : NONE;
    }
    rec[15] = live;
    asm volatile("" ::: "memory");
    const uint32_t gbuf = get_write_ptr(cb_g);
    noc_async_write(rec_l1, get_noc_addr(CX, CY, gbuf + q * 64), 64);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(CX, CY, get_semaphore(sem_g_id)), 1);
    const uint32_t kbuf_addr = get_write_ptr(cb_k);
    volatile tt_l1_ptr uint32_t* kbuf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kbuf_addr);
    const uint32_t semk_addr = get_semaphore(sem_k_id);
    volatile tt_l1_ptr uint32_t* semk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semk_addr);
    uint32_t keep = 0;
    if (q == 0) {
        // 2. worker 0: all M records, then the assignment (tt/replicas.py assign, bit for bit)
        volatile tt_l1_ptr uint32_t* semg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_g_id));
        wait_min(semg, M);
        *semg = 0;
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* g = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(gbuf);
        const uint32_t sc = get_write_ptr(cb_sc);
        uint32_t* where = reinterpret_cast<uint32_t*>(sc);
        uint32_t* load = where + NE_REP;
        const volatile tt_l1_ptr uint32_t* code = idw + 64;  // byte 256
        for (uint32_t e = 0; e < NE_REP; ++e) {
            where[e] = NONE;
        }
        for (uint32_t c = 0; c < N_CHIPS; ++c) {
            load[c] = 0;
        }
        for (uint32_t t = 0; t < M; ++t) {
            if (g[t * 16 + 15] != 0) {
                for (uint32_t j = 0; j < K; ++j) {
                    const uint32_t e = g[t * 16 + j];
                    if (e < NE_REP) {
                        where[e] = PEND;
                    }
                }
            }
        }
        for (uint32_t e = 0; e < NE_REP; ++e) {  // experts without a replica: their home
            if (where[e] == PEND && code[e] == NONE) {
                const uint32_t h = e / E_LOC;
                where[e] = h;
                load[h] += 1;
            }
        }
        for (uint32_t e = 0; e < NE_REP; ++e) {  // replicated: the less loaded holder, ties home
            if (where[e] == PEND) {
                const uint32_t h = e / E_LOC, r = code[e] >> 8;
                const uint32_t c = load[h] <= load[r] ? h : r;
                load[c] += 1;
                where[e] = c;
            }
        }
        for (uint32_t pass = 0; pass < 2; ++pass) {  // replicas.PASSES
            bool moved = false;
            for (uint32_t e = 0; e < NE_REP; ++e) {
                if (where[e] == NONE || code[e] == NONE) {
                    continue;
                }
                const uint32_t h = e / E_LOC, r = code[e] >> 8, c = where[e];
                const uint32_t o = c == h ? r : h;
                if (load[o] + 1 < load[c]) {
                    load[c] -= 1;
                    load[o] += 1;
                    where[e] = o;
                    moved = true;
                }
            }
            if (!moved) {
                break;
            }
        }
        const uint32_t me = base / E_LOC;
        for (uint32_t s = 0; s < E_LOC; ++s) {
            if (where[base + s] == me) {
                keep |= 1u << s;
            }
        }
        for (uint32_t rj = 0; rj < NREP; ++rj) {
            const uint32_t e = idw[448 + E_LOC + rj];
            if (e < NE_REP && where[e] == me) {
                keep |= 1u << (E_LOC + rj);
            }
        }
        asm volatile("" ::: "memory");
        const auto as = TensorAccessor(a_args, a_addr, (NE_REP + N_CHIPS) * 4);
        noc_async_write(sc, as.get_noc_addr(0, 0), (NE_REP + N_CHIPS) * 4);
        // 3. keep -> every other worker (data, then the linked flag), per rectangle of the worker grid
        kbuf[0] = keep;
        *semk = 1;
        asm volatile("" ::: "memory");
        if constexpr (AN > 0) {
            noc_async_write_multicast(kbuf_addr, mcast_addr(AX0, AY0, AX1, AY1, kbuf_addr), 64, AN, true);
            noc_semaphore_set_multicast(semk_addr, mcast_addr(AX0, AY0, AX1, AY1, semk_addr), AN, false);
        }
        if constexpr (BN > 0) {
            noc_async_write_multicast(kbuf_addr, mcast_addr(BX0, BY0, BX1, BY1, kbuf_addr), 64, BN, true);
            noc_semaphore_set_multicast(semk_addr, mcast_addr(BX0, BY0, BX1, BY1, semk_addr), BN, false);
        }
        noc_async_write_barrier();
        *semk = 0;
    } else {
        wait_min(semk, 1);
        *semk = 0;
        invalidate_l1_cache();
        keep = kbuf[0];
    }
    return keep;
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
    constexpr auto i_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t RB = i_args.next_compile_time_args_offset();
    constexpr uint32_t REPL = get_compile_time_arg_val(RB);
    constexpr uint32_t LANE = get_compile_time_arg_val(RB + 1);
    constexpr uint32_t NREP = get_compile_time_arg_val(RB + 2);
    constexpr uint32_t M = get_compile_time_arg_val(RB + 3);
    constexpr uint32_t cb_g = get_compile_time_arg_val(RB + 4);
    constexpr uint32_t cb_k = get_compile_time_arg_val(RB + 5);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(RB + 6);
    constexpr uint32_t sem_g_id = get_compile_time_arg_val(RB + 7);
    constexpr uint32_t sem_k_id = get_compile_time_arg_val(RB + 8);
    constexpr uint32_t CX = get_compile_time_arg_val(RB + 9);
    constexpr uint32_t CY = get_compile_time_arg_val(RB + 10);
    constexpr uint32_t AX0 = get_compile_time_arg_val(RB + 11);
    constexpr uint32_t AY0 = get_compile_time_arg_val(RB + 12);
    constexpr uint32_t AX1 = get_compile_time_arg_val(RB + 13);
    constexpr uint32_t AY1 = get_compile_time_arg_val(RB + 14);
    constexpr uint32_t AN = get_compile_time_arg_val(RB + 15);
    constexpr uint32_t BX0 = get_compile_time_arg_val(RB + 16);
    constexpr uint32_t BY0 = get_compile_time_arg_val(RB + 17);
    constexpr uint32_t BX1 = get_compile_time_arg_val(RB + 18);
    constexpr uint32_t BY1 = get_compile_time_arg_val(RB + 19);
    constexpr uint32_t BN = get_compile_time_arg_val(RB + 20);
    constexpr auto a_args = TensorAccessorArgs<RB + 21>();
    constexpr uint32_t E_TOT = E_LOC + (REPL ? NREP : 0);
    static_assert(K >= 1 && K <= 16, "router_topk: K must be in [1, 16]");
    static_assert(REPL == 0 || (E_TOT <= 32 && M <= 64 && K <= 15), "router_topk: replica mode limits");

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

    // ---- 3. stage the E_TOT row halves (column 0 = weight) + a zero half, then write ----
    cb_reserve_back(cb_st, 1);
    const uint32_t st = get_write_ptr(cb_st);
    uint32_t* stw = reinterpret_cast<uint32_t*>(st);
    for (uint32_t i = 0; i < 16 * (E_TOT + 1) + 16; ++i) {
        stw[i] = 0;
    }
    uint32_t keep = 0xFFFFFFFFu;  // REPL: the slots this chip computes; else every native slot
    uint32_t live = 1;
    const volatile tt_l1_ptr uint32_t* idw = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_id));
    if constexpr (REPL != 0) {
        if constexpr (LANE != 0) {
            live = (idw[16] & 0x7FFFFFFFu) != 0 ? 1u : 0u;  // byte 64: the lane mask row's column 0
        }
        keep = replica_keep<K, M, E_LOC, NREP, cb_g, cb_k, cb_sc, sem_g_id, sem_k_id, CX, CY, AX0, AY0, AX1, AY1, AN,
                            BX0, BY0, BX1, BY1, BN>(q, ids, live, base, idw, st + 64 * (E_TOT + 2), a_args,
                                                    get_common_arg_val<uint32_t>(2));
    }
    for (uint32_t j = 0; j < K; ++j) {
        uint32_t slot = ids[j] - base;  // unsigned: ids below base wrap to >= E_LOC
        if (slot >= E_LOC) {
            slot = 0xFFFFFFFFu;  // not a native expert of this chip (a replica slot below, or none)
        }
        if constexpr (REPL != 0) {
            if (slot == 0xFFFFFFFFu) {
                for (uint32_t rj = 0; rj < NREP; ++rj) {
                    if (idw[448 + E_LOC + rj] == ids[j]) {  // byte 1792: the slot experts
                        slot = E_LOC + rj;
                        break;
                    }
                }
            }
        }
        if (slot < E_TOT && live && ((keep >> slot) & 1u)) {
            float w = s[j] * r;
            if constexpr (SCALE_BITS != 0x3f800000u) {
                w = w * as_float(SCALE_BITS);
            }
            stw[16 * slot] = as_bits(w);
        }
    }
    asm volatile("" ::: "memory");  // the staged stores land before the NoC reads them (write-through L1 cache)
    const uint32_t zero_half = st + 64 * E_TOT;
    const uint32_t idx_src = st + 64 * (E_TOT + 1);
    constexpr uint32_t tb = 4096;
    const auto ws = TensorAccessor(w_args, w_addr, tb);
    for (uint32_t e = 0; e < E_TOT; ++e) {
        const uint32_t page = e * R + tr;
        noc_async_write(st + 64 * e, ws.get_noc_addr(page, row_off), 64);
        noc_async_write(zero_half, ws.get_noc_addr(page, row_off + 1024), 64);
    }
    if constexpr (WRITE_IDX != 0) {
        const uint32_t i_addr = get_common_arg_val<uint32_t>(1);
        const auto is = TensorAccessor(i_args, i_addr, K * 4);
        for (uint32_t j = 0; j < K; ++j) {
            stw[16 * (E_TOT + 1) + j] = ids[j];
        }
        asm volatile("" ::: "memory");
        noc_async_write(idx_src, is.get_noc_addr(q, 0), K * 4);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_o, 1);
    cb_pop_front(cb_id, 1);
}
