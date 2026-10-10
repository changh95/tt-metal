// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode attention input chain (Phase F, F1; tt/kernels/attn_in.py), data movement.
// One source for both RISCs (CT RISC 0 / 1, each on its own NoC) and the three roles (CT ROLE):
//
// ROLE 0 (main grid; core c = y GX + x):
//   QB  (c < 92): unit u = c: output column u of q_b (u < 60) or of the gate (u - 60). RISC r streams K half r of its
//       weight column (16 tiles) into cb_w[r] at once (overlaps the q norm), RISC 0 pushes the cq_n blocks as the QN
//       core's multicast lands (SEM_CQN = blocks). After compute:
//         nope tile (h, k < 4): RISC 1 writes it to slot k of cb_nope on the head's 8 UK cores, then bumps SEM_NOPE;
//         pe tile (h, 4 + p): RISC 1 writes it to the partner (h, 5 - p)'s cb_pepart and bumps its SEM_PE; RISC 0
//           pushes the partner's tile when its own SEM_PE is set; after RoPE RISC 1 writes the rows to the owner of
//           q_mla column 16 + p;
//         gate tile: RISC 1 writes the sigmoid tile to g.
//   UK  (c < 80): head h = c / 8, units j = 2 (c % 8) + {0, 1}: RISC 0 reads the W_UK tiles, pushes cb_nope when
//       SEM_NOPE = 4; RISC 1 writes the rows of each result tile to the owner of column j.
//   OWN (92 <= c < 110): column col = c - 92 of q_mla: RISC 0 zero-fills the rows no producer writes (heads 10..31 of
//       each lane tile), waits SEM_OWN = 10 (one per head) and writes the L lane tiles (l * 18 + col) to q_mla.
//   Row writes: lane l of a producer tile (row l: faces 0 / 1 for l < 16, else 2 / 3) -> row h (faces 0 / 1) of the
//   owner's lane tile l, as two 32 B writes.
// ROLE 1 (QN, one core): RISC 0 / 1 read cq tiles 0..15 / 16..31 (RISC 1 flags SEM_LOC), RISC 0 writes the reduce
//   scaler / eps tiles exactly as the stock rms_norm reader; compute normalizes; RISC 1 multicasts each 4-tile block
//   of cq_n to the same cb_cqn address on the QB rectangle and sets SEM_CQN there to the block count (linked writes).
// ROLE 2 (KN, one core): RISC 0 reads c_raw (cb_kin), the scaler / eps tiles, the RoPE streams of the stock
//   rotary_embedding_hf reader (per tile j: rotated input, sin_j, input, cos_j) and lam; compute writes
//   kv_row = [rms_norm(c_raw) | rope(kpe)] to cb_kvrow; RISC 1 writes lam, and kv_row either to DRAM (KVMODE 0) or row
//   l of every tile into lane l's shard of the height-sharded update input (KVMODE 1; RISC 0 lanes 0..3, RISC 1 4..7).
// Every spin is bounded (a bug never hangs the shared device; it only produces garbage).
//
// CT: 0 ROLE, 1 RISC, 2 GX, 3 L, 4 KVMODE, 5 mc_x0, 6 mc_y0, 7 mc_x1, 8 mc_y1, 9 mc_ndests, 10 eps bits, 11 DEBUG,
//     12 NCORES, 13.. TensorAccessorArgs of cq, kvl, cos, sin, wqb, wgate, wuk, qmla, g, lam, kvout, dbg
// common RT: 0..11 the 12 tensor addresses (same order), 12.. the NOC xy (x << 16 | y) of core 0 .. NCORES - 1,
//     then (KVMODE 1) the NOC xy of the 8 shard cores (lane order)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common.h"

using namespace motif_ain;

namespace {
constexpr uint32_t SPIN_LIMIT = 200000000u;

inline bool wait_sem_min(volatile tt_l1_ptr uint32_t* sem, uint32_t v) {
    uint32_t it = 0;
    do {
        invalidate_l1_cache();
        if (*sem >= v) {
            return true;
        }
    } while (++it < SPIN_LIMIT);
    return false;
}

inline volatile tt_l1_ptr uint32_t* sem_ptr(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

inline uint64_t core_noc(uint32_t xy, uint32_t addr) { return get_noc_addr(xy >> 16, xy & 0xFFFFu, addr); }

// stock rms_norm reader constants: reduce scaler (zeros, then bf16 1.0 in row 0 of every face) and eps (bf16 =
// upper 16 bits of the fp32 bits, column 0 of faces 0 and 2; the rest of the tile is never read)
inline void make_scaler_eps(uint32_t eps_bits) {
    cb_reserve_back(CB_RSCAL, 1);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_RSCAL));
    for (uint32_t i = 0; i < BF16_TILE / 4; ++i) {
        s[i] = 0;
    }
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t c = 0; c < 8; ++c) {
            s[f * 128 + c] = 0x3F803F80u;
        }
    }
    cb_push_back(CB_RSCAL, 1);
    cb_reserve_back(CB_EPS, 1);
    volatile tt_l1_ptr uint16_t* e = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CB_EPS));
    const uint16_t ev = static_cast<uint16_t>(eps_bits >> 16);
    for (uint32_t k = 0; k < 4; k += 2) {
        const uint32_t idx = k << 8;
        for (uint32_t j = 0; j < 256; j += 16) {
            e[idx + j] = ev;
        }
    }
    cb_push_back(CB_EPS, 1);
}

// stock rotary_embedding_hf reader: scalar tile with bf16 -1.0 at element 0
inline void make_rope_scalar() {
    cb_reserve_back(CB_SCAL, 1);
    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CB_SCAL));
    p[0] = 0xBF80u;
    cb_push_back(CB_SCAL, 1);
}

// rows 0..L-1 of the bf16 tile at src -> row h of lane tile l of the owner's q_mla assembly (two 32 B writes each)
template <uint32_t L>
inline void write_rows(uint32_t src, uint32_t owner_xy, uint32_t qmla_base, uint32_t h) {
    for (uint32_t l = 0; l < L; ++l) {
        const uint32_t so = src + (l < 16 ? 0u : 1024u) + (l & 15u) * 32u;
        const uint32_t d = qmla_base + l * BF16_TILE + h * 32u;
        noc_async_write(so, core_noc(owner_xy, d), 32);
        noc_async_write(so + 512u, core_noc(owner_xy, d + 512u), 32);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t RISC = get_compile_time_arg_val(1);
    constexpr uint32_t GX = get_compile_time_arg_val(2);
    constexpr uint32_t L = get_compile_time_arg_val(3);
    constexpr uint32_t KVMODE = get_compile_time_arg_val(4);
    constexpr uint32_t mc_x0 = get_compile_time_arg_val(5);
    constexpr uint32_t mc_y0 = get_compile_time_arg_val(6);
    constexpr uint32_t mc_x1 = get_compile_time_arg_val(7);
    constexpr uint32_t mc_y1 = get_compile_time_arg_val(8);
    constexpr uint32_t mc_ndests = get_compile_time_arg_val(9);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(10);
    constexpr uint32_t DEBUG = get_compile_time_arg_val(11);
    constexpr uint32_t NCORES = get_compile_time_arg_val(12);
    constexpr auto cq_args = TensorAccessorArgs<13>();
    constexpr auto kvl_args = TensorAccessorArgs<cq_args.next_compile_time_args_offset()>();
    constexpr auto cos_args = TensorAccessorArgs<kvl_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto wqb_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr auto wg_args = TensorAccessorArgs<wqb_args.next_compile_time_args_offset()>();
    constexpr auto wuk_args = TensorAccessorArgs<wg_args.next_compile_time_args_offset()>();
    constexpr auto qmla_args = TensorAccessorArgs<wuk_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<qmla_args.next_compile_time_args_offset()>();
    constexpr auto lam_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto kvo_args = TensorAccessorArgs<lam_args.next_compile_time_args_offset()>();
    constexpr auto dbg_args = TensorAccessorArgs<kvo_args.next_compile_time_args_offset()>();
    static_assert(L >= 1 && L <= 32, "one tile row of lanes");
    static_assert(KVMODE == 0 || L == 8, "the shard write serves the 8-lane draft-1 update");

    constexpr uint32_t XY0 = 12;  // first core xy common arg
    auto xy_of = [](uint32_t c) { return get_common_arg_val<uint32_t>(XY0 + c); };

    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();

    // =================================================================================================================
    if constexpr (ROLE == 1) {  // QN: q norm + cq_n multicast
        const auto cqs = TensorAccessor(cq_args, get_common_arg_val<uint32_t>(0), FP32_TILE);
        volatile tt_l1_ptr uint32_t* loc = sem_ptr(SEM_LOC);
        if constexpr (RISC == 0) {
            make_scaler_eps(EPS_BITS);
            cb_reserve_back(CB_QX, KQ);
            const uint32_t base = get_write_ptr(CB_QX);
            for (uint32_t t = 0; t < KH; ++t) {
                noc_async_read(cqs.get_noc_addr(t), base + t * FP32_TILE, FP32_TILE);
            }
            noc_async_read_barrier();
            cb_push_back(CB_QX, KH);
            wait_sem_min(loc, 1);
            *loc = 0;
            cb_push_back(CB_QX, KH);
        } else {
            const uint32_t base = get_write_ptr(CB_QX);
            for (uint32_t t = KH; t < KQ; ++t) {
                noc_async_read(cqs.get_noc_addr(t), base + t * FP32_TILE, FP32_TILE);
            }
            noc_async_read_barrier();
            *loc = 1;
            // multicast cq_n block by block as the norm produces it
            const uint32_t cbase = get_read_ptr(CB_CQN);
            const uint32_t sem_addr = get_semaphore(SEM_CQN);
            volatile tt_l1_ptr uint32_t* sem = sem_ptr(SEM_CQN);
            const uint64_t mc_sem = noc_index == 0 ? get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, sem_addr)
                                                   : get_noc_multicast_addr(mc_x1, mc_y1, mc_x0, mc_y0, sem_addr);
            for (uint32_t b = 0; b < QN_BLOCKS; ++b) {
                cb_wait_front(CB_CQN, 4 * (b + 1));
                const uint32_t src = cbase + b * 4 * FP32_TILE;
                const uint64_t dst = noc_index == 0 ? get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, src)
                                                    : get_noc_multicast_addr(mc_x1, mc_y1, mc_x0, mc_y0, src);
                noc_async_write_multicast(src, dst, 4 * FP32_TILE, mc_ndests, true);
                *sem = b + 1;
                noc_semaphore_set_multicast(sem_addr, mc_sem, mc_ndests, false);
                noc_async_writes_flushed();
            }
            if constexpr (DEBUG) {
                const auto ds = TensorAccessor(dbg_args, get_common_arg_val<uint32_t>(11), FP32_TILE);
                for (uint32_t t = 0; t < KQ; ++t) {
                    noc_async_write(cbase + t * FP32_TILE, ds.get_noc_addr(t), FP32_TILE);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(CB_CQN, KQ);
            *sem = 0;
        }
        return;
    }

    // =================================================================================================================
    if constexpr (ROLE == 2) {  // KN: kv norm + k_pe RoPE + lam + kv output
        const auto kvs = TensorAccessor(kvl_args, get_common_arg_val<uint32_t>(1), BF16_TILE);
        if constexpr (RISC == 0) {
            const auto coss = TensorAccessor(cos_args, get_common_arg_val<uint32_t>(2), BF16_TILE);
            const auto sins = TensorAccessor(sin_args, get_common_arg_val<uint32_t>(3), BF16_TILE);
            make_scaler_eps(EPS_BITS);
            cb_reserve_back(CB_KIN, 16);
            uint32_t d = get_write_ptr(CB_KIN);
            for (uint32_t t = 0; t < 16; ++t) {
                noc_async_read(kvs.get_noc_addr(t), d + t * BF16_TILE, BF16_TILE);
            }
            noc_async_read_barrier();
            cb_push_back(CB_KIN, 16);
            make_rope_scalar();
            for (uint32_t j = 0; j < 2; ++j) {  // the stock reader's stream order: rotated, sin, input, cos
                cb_reserve_back(CB_ROTIN, 1);
                noc_async_read(kvs.get_noc_addr(16 + (1 - j)), get_write_ptr(CB_ROTIN), BF16_TILE);
                noc_async_read_barrier();
                cb_push_back(CB_ROTIN, 1);
                cb_reserve_back(CB_SIN, 1);
                noc_async_read(sins.get_noc_addr(j), get_write_ptr(CB_SIN), BF16_TILE);
                noc_async_read_barrier();
                cb_push_back(CB_SIN, 1);
                cb_reserve_back(CB_KPEX, 1);
                noc_async_read(kvs.get_noc_addr(16 + j), get_write_ptr(CB_KPEX), BF16_TILE);
                noc_async_read_barrier();
                cb_push_back(CB_KPEX, 1);
                cb_reserve_back(CB_COS, 1);
                noc_async_read(coss.get_noc_addr(j), get_write_ptr(CB_COS), BF16_TILE);
                noc_async_read_barrier();
                cb_push_back(CB_COS, 1);
            }
            cb_reserve_back(CB_LAM, 2);
            d = get_write_ptr(CB_LAM);
            noc_async_read(kvs.get_noc_addr(18), d, BF16_TILE);
            noc_async_read(kvs.get_noc_addr(19), d + BF16_TILE, BF16_TILE);
            noc_async_read_barrier();
            cb_push_back(CB_LAM, 2);
        } else {
            const auto lams = TensorAccessor(lam_args, get_common_arg_val<uint32_t>(9), BF16_TILE);
            cb_wait_front(CB_LAM, 2);
            const uint32_t s = get_read_ptr(CB_LAM);
            noc_async_write(s, lams.get_noc_addr(0), BF16_TILE);
            noc_async_write(s + BF16_TILE, lams.get_noc_addr(1), BF16_TILE);
        }
        // kv_row out: DRAM tiles (RISC 1) or lane rows into the 8 update shards (RISC r: lanes 4r .. 4r + 3)
        if constexpr (KVMODE == 0) {
            if constexpr (RISC == 1) {
                const auto kvo = TensorAccessor(kvo_args, get_common_arg_val<uint32_t>(10), BF16_TILE);
                cb_wait_front(CB_KVROW, KVT);
                const uint32_t s = get_read_ptr(CB_KVROW);
                for (uint32_t t = 0; t < KVT; ++t) {
                    noc_async_write(s + t * BF16_TILE, kvo.get_noc_addr(t), BF16_TILE);
                }
            }
        } else {
            cb_wait_front(CB_KVROW, KVT);
            const uint32_t s = get_read_ptr(CB_KVROW);
            const uint32_t upd = get_common_arg_val<uint32_t>(10);
            for (uint32_t l = 4 * RISC; l < 4 * RISC + 4; ++l) {
                const uint32_t sxy = get_common_arg_val<uint32_t>(XY0 + NCORES + l);
                const uint32_t ro = (l < 16 ? 0u : 1024u) + (l & 15u) * 32u;
                for (uint32_t t = 0; t < KVT; ++t) {
                    const uint32_t so = s + t * BF16_TILE + ro;
                    noc_async_write(so, core_noc(sxy, upd + t * BF16_TILE), 32);
                    noc_async_write(so + 512u, core_noc(sxy, upd + t * BF16_TILE + 512u), 32);
                }
            }
        }
        noc_async_write_barrier();
        // cb_kvrow is waited on by both RISCs and never popped (single use per launch)
        return;
    }

    // =================================================================================================================
    // ROLE 0: main grid
    const bool qb = c < NQB;
    const bool is_gate = qb && c >= NQ;
    const uint32_t qh = c / HT, qk = c % HT;
    const bool is_nope = qb && !is_gate && qk < NOPE_T;
    const bool is_pe = qb && !is_gate && qk >= NOPE_T;
    const bool uk = c < NUK;
    const uint32_t uh = c / UK_CPH, ui = c % UK_CPH;
    const bool own = c >= OWN0 && c < OWN0 + NCOL;
    const uint32_t qmla_base = get_write_ptr(CB_QMLA);

    if constexpr (RISC == 0) {
        // ---- QB weights, K half 0 --------------------------------------------------------------------------------
        if (qb) {
            const uint32_t n = is_gate ? c - NQ : c;
            const uint32_t NT = is_gate ? NG : NQ;
            const uint32_t waddr = get_common_arg_val<uint32_t>(is_gate ? 5 : 4);
            cb_reserve_back(CB_WA, KH);
            const uint32_t d = get_write_ptr(CB_WA);
            if (is_gate) {
                const auto wgs = TensorAccessor(wg_args, waddr, BF16_TILE);
                for (uint32_t k = 0; k < KH; ++k) {
                    noc_async_read(wgs.get_noc_addr(k * NT + n), d + k * BF16_TILE, BF16_TILE);
                }
            } else {
                const auto wqs = TensorAccessor(wqb_args, waddr, BF16_TILE);
                for (uint32_t k = 0; k < KH; ++k) {
                    noc_async_read(wqs.get_noc_addr(k * NT + n), d + k * BF16_TILE, BF16_TILE);
                }
            }
            cb_reserve_back(CB_CQN, KQ);  // the multicast lands at the CB base
        }
        // ---- UK weights ------------------------------------------------------------------------------------------
        if (uk) {
            const auto wus = TensorAccessor(wuk_args, get_common_arg_val<uint32_t>(6), BF16_TILE);
            cb_reserve_back(CB_WUK, UK_PER * NOPE_T);
            const uint32_t d = get_write_ptr(CB_WUK);
            for (uint32_t t = 0; t < UK_PER; ++t) {
                const uint32_t j = UK_PER * ui + t;
                for (uint32_t k = 0; k < NOPE_T; ++k) {
                    noc_async_read(wus.get_noc_addr(uh * (NOPE_T * UKN) + k * UKN + j),
                                   d + (t * NOPE_T + k) * BF16_TILE, BF16_TILE);
                }
            }
            cb_reserve_back(CB_NOPE, NOPE_T);
        }
        // ---- pe: cos / sin tile p, the RoPE scalar ---------------------------------------------------------------
        if (is_pe) {
            const uint32_t p = qk - NOPE_T;
            const auto coss = TensorAccessor(cos_args, get_common_arg_val<uint32_t>(2), BF16_TILE);
            const auto sins = TensorAccessor(sin_args, get_common_arg_val<uint32_t>(3), BF16_TILE);
            cb_reserve_back(CB_COS, 1);
            noc_async_read(coss.get_noc_addr(p), get_write_ptr(CB_COS), BF16_TILE);
            cb_reserve_back(CB_SIN, 1);
            noc_async_read(sins.get_noc_addr(p), get_write_ptr(CB_SIN), BF16_TILE);
            make_rope_scalar();
            cb_reserve_back(CB_PEPART, 1);
        }
        noc_async_read_barrier();
        if (qb) {
            cb_push_back(CB_WA, KH);
        }
        if (uk) {
            cb_push_back(CB_WUK, UK_PER * NOPE_T);
        }
        if (is_pe) {
            cb_push_back(CB_COS, 1);
            cb_push_back(CB_SIN, 1);
        }
        // ---- owner: zero the rows no producer writes (heads 10..31 of every lane tile) -----------------------------
        if (own) {
            for (uint32_t l = 0; l < L; ++l) {
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(qmla_base + l * BF16_TILE);
                for (uint32_t i = H * 8; i < 128; ++i) {  // faces 0 / 1: rows 10..15 (8 words per row)
                    t[i] = 0;
                    t[128 + i] = 0;
                }
                for (uint32_t i = 256; i < 512; ++i) {  // faces 2 / 3
                    t[i] = 0;
                }
            }
        }
        // ---- QB: cq_n blocks as they land --------------------------------------------------------------------------
        if (qb) {
            volatile tt_l1_ptr uint32_t* sem = sem_ptr(SEM_CQN);
            for (uint32_t b = 0; b < QN_BLOCKS; ++b) {
                wait_sem_min(sem, b + 1);
                cb_push_back(CB_CQN, 4);
            }
        }
        // ---- pe: the partner's tile ---------------------------------------------------------------------------------
        if (is_pe) {
            volatile tt_l1_ptr uint32_t* sem = sem_ptr(SEM_PE);
            wait_sem_min(sem, 1);
            *sem = 0;
            cb_push_back(CB_PEPART, 1);
        }
        // ---- UK: the 4 nope tiles of the head -----------------------------------------------------------------------
        if (uk) {
            volatile tt_l1_ptr uint32_t* sem = sem_ptr(SEM_NOPE);
            wait_sem_min(sem, NOPE_T);
            *sem = 0;
            cb_push_back(CB_NOPE, NOPE_T);
        }
        // ---- owner: write the lane tiles ---------------------------------------------------------------------------
        if (own) {
            const uint32_t col = c - OWN0;
            volatile tt_l1_ptr uint32_t* sem = sem_ptr(SEM_OWN);
            wait_sem_min(sem, H);
            *sem = 0;
            const auto qs = TensorAccessor(qmla_args, get_common_arg_val<uint32_t>(7), BF16_TILE);
            for (uint32_t l = 0; l < L; ++l) {
                noc_async_write(qmla_base + l * BF16_TILE, qs.get_noc_addr(l * NCOL + col), BF16_TILE);
            }
            noc_async_write_barrier();
        }
        if (qb) {
            *sem_ptr(SEM_CQN) = 0;
        }
        return;
    }

    // ---- RISC 1 ---------------------------------------------------------------------------------------------------
    if (qb) {
        const uint32_t n = is_gate ? c - NQ : c;
        const uint32_t NT = is_gate ? NG : NQ;
        cb_reserve_back(CB_WB, KH);
        const uint32_t d = get_write_ptr(CB_WB);
        if (is_gate) {
            const auto wgs = TensorAccessor(wg_args, get_common_arg_val<uint32_t>(5), BF16_TILE);
            for (uint32_t k = 0; k < KH; ++k) {
                noc_async_read(wgs.get_noc_addr((KH + k) * NT + n), d + k * BF16_TILE, BF16_TILE);
            }
        } else {
            const auto wqs = TensorAccessor(wqb_args, get_common_arg_val<uint32_t>(4), BF16_TILE);
            for (uint32_t k = 0; k < KH; ++k) {
                noc_async_read(wqs.get_noc_addr((KH + k) * NT + n), d + k * BF16_TILE, BF16_TILE);
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_WB, KH);

        if (is_nope) {  // -> slot qk of cb_nope on the head's 8 UK cores
            cb_wait_front(CB_QSEND, 1);
            const uint32_t s = get_read_ptr(CB_QSEND);
            const uint32_t dst = get_write_ptr(CB_NOPE) + qk * BF16_TILE;
            for (uint32_t i = 0; i < UK_CPH; ++i) {
                noc_async_write(s, core_noc(xy_of(qh * UK_CPH + i), dst), BF16_TILE);
            }
            noc_async_write_barrier();
            const uint32_t sa = get_semaphore(SEM_NOPE);
            for (uint32_t i = 0; i < UK_CPH; ++i) {
                noc_semaphore_inc(core_noc(xy_of(qh * UK_CPH + i), sa), 1);
            }
            cb_pop_front(CB_QSEND, 1);
        } else if (is_pe) {  // -> the partner; then the RoPE rows -> the owner of column 16 + p
            const uint32_t p = qk - NOPE_T;
            const uint32_t partner = p == 0 ? c + 1 : c - 1;
            cb_wait_front(CB_QSEND, 1);
            noc_async_write(get_read_ptr(CB_QSEND), core_noc(xy_of(partner), get_write_ptr(CB_PEPART)), BF16_TILE);
            noc_async_write_barrier();
            noc_semaphore_inc(core_noc(xy_of(partner), get_semaphore(SEM_PE)), 1);
            cb_pop_front(CB_QSEND, 1);
            cb_wait_front(CB_ROPE, 1);
            const uint32_t oxy = xy_of(OWN0 + UKN + p);
            write_rows<L>(get_read_ptr(CB_ROPE), oxy, qmla_base, qh);
            noc_async_write_barrier();
            noc_semaphore_inc(core_noc(oxy, get_semaphore(SEM_OWN)), 1);
            cb_pop_front(CB_ROPE, 1);
        } else {  // gate -> g
            const auto gs = TensorAccessor(g_args, get_common_arg_val<uint32_t>(8), BF16_TILE);
            cb_wait_front(CB_GOUT, 1);
            noc_async_write(get_read_ptr(CB_GOUT), gs.get_noc_addr(c - NQ), BF16_TILE);
            noc_async_write_barrier();
            cb_pop_front(CB_GOUT, 1);
        }
    }
    if (uk) {  // result tiles -> rows of the owners of columns j
        cb_wait_front(CB_UKOUT, UK_PER);
        const uint32_t s = get_read_ptr(CB_UKOUT);
        for (uint32_t t = 0; t < UK_PER; ++t) {
            write_rows<L>(s + t * BF16_TILE, xy_of(OWN0 + UK_PER * ui + t), qmla_base, uh);
        }
        noc_async_write_barrier();
        for (uint32_t t = 0; t < UK_PER; ++t) {
            noc_semaphore_inc(core_noc(xy_of(OWN0 + UK_PER * ui + t), get_semaphore(SEM_OWN)), 1);
        }
        cb_pop_front(CB_UKOUT, UK_PER);
    }
    noc_async_atomic_barrier();
}
