// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode attention input chain (Phase F, F1; tt/kernels/attn_in.py), data movement.
// One source for both RISCs (CT RISC 0 / 1, each on its own NoC) and the three roles (CT ROLE):
//
// ROLE 0 (main grid; core c = y GX + x):
//   LAT (LAT = 1, c < 52): latent unit v = c: column v of Wq_lat (v < 32) or v - 32 of Wkv_lat, K = 128. RISC r streams
//       the weight batches b (8 tiles, b % 2 == r) through its ring cb_lw[r]; RISC 0 pushes the x chunks as the QN
//       core's multicast lands (polled between weight batches and while its ring is full: compute needs the x chunk
//       of a batch before it frees ring space, so RISC 0 must never block on the ring with x chunks pending). After compute RISC 1 writes the tile into the QN core's
//       cb_qx (q, fp32) or the KN core's cb_kin / cb_kpe2 / cb_lam (kv, bf16) and bumps that core's SEM_LAT.
//   QB  (QB0 <= c < QB0 + 92): unit u = c - QB0: output column u of q_b (u < 60) or of the gate (u - 60). RISC r reads
//       K half r of its weight column (16 tiles; after its latent batches) into cb_w[r]; RISC 0 pushes the cq_n blocks
//       as the QN core's multicast lands (SEM_CQN = blocks). After compute:
//         nope tile (h, k < 4): RISC 1 writes it to slot k of cb_nope on the head's 8 UK cores, then bumps SEM_NOPE;
//         pe tile (h, 4 + p): RISC 1 writes it to the partner (h, 5 - p)'s cb_pepart and bumps its SEM_PE; RISC 0
//           pushes the partner's tile when its own SEM_PE is set; after RoPE RISC 1 writes the rows to the owner of
//           q_mla column 16 + p;
//         gate tile: RISC 1 writes the sigmoid tile to g.
//   UK  (UK0 <= c < UK0 + 80): head h = (c - UK0) / 8, units j = 2 ((c - UK0) % 8) + {0, 1}: RISC 0 reads the W_UK
//       tiles, pushes cb_nope when SEM_NOPE = 4; RISC 1 writes the rows of each result tile to the owner of column j.
//   OWN (OWN0 <= c < OWN0 + 18): column col = c - OWN0 of q_mla: RISC 0 zero-fills the rows no producer writes (heads
//       10..31 of each lane tile) first, and last waits SEM_OWN = 10 (one per head) and writes the L lane tiles.
//   Row writes: lane l of a producer tile (row l: faces 0 / 1 for l < 16, else 2 / 3) -> row h (faces 0 / 1) of the
//   owner's lane tile l, as two 32 B writes.
// ROLE 1 (QN, one core): LAT: RISC r reads K half r of x in 8-tile chunks and multicasts each to the same cb_x address
//   on the latent rectangle, setting SEM_XA / SEM_XB there to the chunk count; then RISC 0 waits SEM_LAT = 32 (the cq
//   tiles in cb_qx). LAT = 0: RISC 0 / 1 read cq tiles 0..15 / 16..31 from DRAM (RISC 1 flags SEM_LOC). RISC 0 writes
//   the reduce scaler / eps tiles exactly as the stock rms_norm reader; compute normalizes; RISC 1 multicasts each 4-tile
//   block of cq_n to the same cb_cqn address on the QB rectangle and sets SEM_CQN there to the block count.
// ROLE 2 (KN, one core): RISC 0 gets c_raw (cb_kin; LAT: landed, else read), the scaler / eps tiles, the RoPE streams
//   of the stock rotary_embedding_hf reader (per tile j: rotated input, sin_j, input, cos_j) and lam; compute writes
//   kv_row = [rms_norm(c_raw) | rope(kpe)] to cb_kvrow; RISC 1 writes lam, and kv_row either to DRAM (KVMODE 0) or row
//   l of every tile into lane l's shard of the height-sharded update input (KVMODE 1; RISC 0 lanes 0..3, RISC 1 4..7).
// Every spin is bounded (a bug never hangs the shared device; it only produces garbage).
//
// CT: 0 ROLE, 1 RISC, 2 GX, 3 L, 4 KVMODE, 5..8 cq_n mcast rect (x0, y0, x1, y1), 9 its ndests, 10 eps bits, 11 DEBUG,
//     12 NCORES, 13 LAT, 14 QB0, 15 UK0, 16 OWN0, 17..20 x mcast rect, 21 its ndests,
//     22.. TensorAccessorArgs of cq, kvl, cos, sin, wqb, wgate, wuk, qmla, g, lam, kvout, dbg, x, wqlat, wkvlat
//     (LAT = 0: x / wqlat / wkvlat are copies of cq's; LAT = 1: cq / kvl are copies of x's)
// common RT: 0..14 the 15 tensor addresses (same order), 15.. the NOC xy (x << 16 | y) of core 0 .. NCORES - 1, then
//     (KVMODE 1) the NOC xy of the 8 shard cores (lane order)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common.h"

using namespace motif_ain;

#ifndef MOTIF_AIN_EXP
#define MOTIF_AIN_EXP 0  // experiments only: bit 0 skip QB weight reads, bit 1 skip UK weight reads
#endif

namespace {
constexpr uint32_t SPIN_LIMIT = 200000000u;
constexpr uint32_t NARGS = 15;  // tensor addresses before the core xy table

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

inline uint32_t sem_val(volatile tt_l1_ptr uint32_t* sem) {
    invalidate_l1_cache();
    return *sem;
}

inline volatile tt_l1_ptr uint32_t* sem_ptr(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

inline uint64_t core_noc(uint32_t xy, uint32_t addr) { return get_noc_addr(xy >> 16, xy & 0xFFFFu, addr); }

inline uint64_t mcast_addr(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t addr) {
    return noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, addr) : get_noc_multicast_addr(x1, y1, x0, y0, addr);
}

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

// one tile from this core's L1 (local NoC read) into a CB stream
inline void push_local(uint32_t cb, uint32_t src) {
    cb_reserve_back(cb, 1);
    noc_async_read(get_noc_addr(src), get_write_ptr(cb), BF16_TILE);
    noc_async_read_barrier();
    cb_push_back(cb, 1);
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
    constexpr uint32_t LAT = get_compile_time_arg_val(13);
    constexpr uint32_t QB0 = get_compile_time_arg_val(14);
    constexpr uint32_t UK0 = get_compile_time_arg_val(15);
    constexpr uint32_t OWN0 = get_compile_time_arg_val(16);
    constexpr uint32_t xm_x0 = get_compile_time_arg_val(17);
    constexpr uint32_t xm_y0 = get_compile_time_arg_val(18);
    constexpr uint32_t xm_x1 = get_compile_time_arg_val(19);
    constexpr uint32_t xm_y1 = get_compile_time_arg_val(20);
    constexpr uint32_t xm_ndests = get_compile_time_arg_val(21);
    constexpr auto cq_args = TensorAccessorArgs<22>();
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
    constexpr auto x_args = TensorAccessorArgs<dbg_args.next_compile_time_args_offset()>();
    constexpr auto wql_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto wkl_args = TensorAccessorArgs<wql_args.next_compile_time_args_offset()>();
    static_assert(L >= 1 && L <= 32, "one tile row of lanes");
    static_assert(KVMODE == 0 || L == 8, "the shard write serves the 8-lane draft-1 update");

    auto xy_of = [](uint32_t c) { return get_common_arg_val<uint32_t>(NARGS + c); };

    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();

    // =================================================================================================================
    if constexpr (ROLE == 1) {  // QN: (LAT) x multicast; q norm; cq_n multicast
        volatile tt_l1_ptr uint32_t* loc = sem_ptr(SEM_LOC);
        if constexpr (LAT) {  // x K half RISC: 8 chunks of 8 tiles, read and multicast as they land
            const auto xs = TensorAccessor(x_args, get_common_arg_val<uint32_t>(12), BF16_TILE);
            const uint32_t xbase = get_write_ptr(CB_X);
            const uint32_t sid = RISC == 0 ? SEM_XA : SEM_XB;
            const uint32_t sa = get_semaphore(sid);
            volatile tt_l1_ptr uint32_t* sem = sem_ptr(sid);
            const uint64_t mc_sem = mcast_addr(xm_x0, xm_y0, xm_x1, xm_y1, sa);
            constexpr uint32_t half = NXC / 2;
            for (uint32_t ch = 0; ch < half; ++ch) {
                const uint32_t t0 = (RISC * half + ch) * XC;
                const uint32_t dst = xbase + t0 * BF16_TILE;
                for (uint32_t i = 0; i < XC; ++i) {
                    noc_async_read(xs.get_noc_addr(t0 + i), dst + i * BF16_TILE, BF16_TILE);
                }
                noc_async_read_barrier();
                noc_async_write_multicast(dst, mcast_addr(xm_x0, xm_y0, xm_x1, xm_y1, dst), XC * BF16_TILE, xm_ndests,
                                          true);
                *sem = ch + 1;
                noc_semaphore_set_multicast(sa, mc_sem, xm_ndests, false);
                noc_async_writes_flushed();
            }
            noc_async_write_barrier();
            *sem = 0;
        }
        if constexpr (RISC == 0) {
            make_scaler_eps(EPS_BITS);
            cb_reserve_back(CB_QX, KQ);
            if constexpr (LAT) {
                volatile tt_l1_ptr uint32_t* lat = sem_ptr(SEM_LAT);
                wait_sem_min(lat, NLQ);
                *lat = 0;
                cb_push_back(CB_QX, KQ);
            } else {
                const auto cqs = TensorAccessor(cq_args, get_common_arg_val<uint32_t>(0), FP32_TILE);
                const uint32_t base = get_write_ptr(CB_QX);
                for (uint32_t t = 0; t < KH; ++t) {
                    noc_async_read(cqs.get_noc_addr(t), base + t * FP32_TILE, FP32_TILE);
                }
                noc_async_read_barrier();
                cb_push_back(CB_QX, KH);
                wait_sem_min(loc, 1);
                *loc = 0;
                cb_push_back(CB_QX, KH);
            }
        } else {
            if constexpr (!LAT) {
                const auto cqs = TensorAccessor(cq_args, get_common_arg_val<uint32_t>(0), FP32_TILE);
                const uint32_t base = get_write_ptr(CB_QX);
                for (uint32_t t = KH; t < KQ; ++t) {
                    noc_async_read(cqs.get_noc_addr(t), base + t * FP32_TILE, FP32_TILE);
                }
                noc_async_read_barrier();
                *loc = 1;
            }
            // multicast cq_n block by block as the norm produces it
            const uint32_t cbase = get_read_ptr(CB_CQN);
            const uint32_t sem_addr = get_semaphore(SEM_CQN);
            volatile tt_l1_ptr uint32_t* sem = sem_ptr(SEM_CQN);
            const uint64_t mc_sem = mcast_addr(mc_x0, mc_y0, mc_x1, mc_y1, sem_addr);
            for (uint32_t b = 0; b < QN_BLOCKS; ++b) {
                cb_wait_front(CB_CQN, 4 * (b + 1));
                const uint32_t src = cbase + b * 4 * FP32_TILE;
                noc_async_write_multicast(src, mcast_addr(mc_x0, mc_y0, mc_x1, mc_y1, src), 4 * FP32_TILE, mc_ndests,
                                          true);
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
        if constexpr (RISC == 0) {
            const auto kvs = TensorAccessor(kvl_args, get_common_arg_val<uint32_t>(1), BF16_TILE);
            const auto coss = TensorAccessor(cos_args, get_common_arg_val<uint32_t>(2), BF16_TILE);
            const auto sins = TensorAccessor(sin_args, get_common_arg_val<uint32_t>(3), BF16_TILE);
            make_scaler_eps(EPS_BITS);
            make_rope_scalar();
            // the 20 kvl tiles [c_raw 0..15 | kpe 16, 17 | lam 18, 19] at the cb_kin base (an alias of the cb_x region)
            cb_reserve_back(CB_KIN, 16);
            const uint32_t kin = get_write_ptr(CB_KIN);
            const uint32_t kpe = kin + 16 * BF16_TILE;
            if constexpr (LAT) {
                volatile tt_l1_ptr uint32_t* lat = sem_ptr(SEM_LAT);
                wait_sem_min(lat, NLK);
                *lat = 0;
            } else {
                for (uint32_t t = 0; t < KVIN; ++t) {
                    noc_async_read(kvs.get_noc_addr(t), kin + t * BF16_TILE, BF16_TILE);
                }
                noc_async_read_barrier();
            }
            cb_push_back(CB_KIN, 16);
            push_local(CB_LAM, kin + 18 * BF16_TILE);
            push_local(CB_LAM, kin + 19 * BF16_TILE);
            for (uint32_t j = 0; j < 2; ++j) {  // the stock reader's stream order: rotated, sin, input, cos
                push_local(CB_ROTIN, kpe + (1 - j) * BF16_TILE);
                cb_reserve_back(CB_SIN, 1);
                noc_async_read(sins.get_noc_addr(j), get_write_ptr(CB_SIN), BF16_TILE);
                noc_async_read_barrier();
                cb_push_back(CB_SIN, 1);
                push_local(CB_KPEX, kpe + j * BF16_TILE);
                cb_reserve_back(CB_COS, 1);
                noc_async_read(coss.get_noc_addr(j), get_write_ptr(CB_COS), BF16_TILE);
                noc_async_read_barrier();
                cb_push_back(CB_COS, 1);
            }
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
                const uint32_t sxy = get_common_arg_val<uint32_t>(NARGS + NCORES + l);
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
    const bool lat = LAT && c < NLAT;
    const bool lat_q = lat && c < NLQ;
    const bool qb = c >= QB0 && c < QB0 + NQB;
    const uint32_t u = c - QB0;
    const bool is_gate = qb && u >= NQ;
    const uint32_t qh = u / HT, qk = u % HT;
    const bool is_nope = qb && !is_gate && qk < NOPE_T;
    const bool is_pe = qb && !is_gate && qk >= NOPE_T;
    const bool uk = c >= UK0 && c < UK0 + NUK;
    const uint32_t uh = (c - UK0) / UK_CPH, ui = (c - UK0) % UK_CPH;
    const bool own = c >= OWN0 && c < OWN0 + NCOL;
    const uint32_t qmla_base = get_write_ptr(CB_QMLA);

    if constexpr (RISC == 0) {
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
        // ---- latent: even weight batches; push the x chunks as they land ---------------------------------------------
        if (lat) {
            const uint32_t NT = lat_q ? NLQ : NLK;
            const uint32_t n = lat_q ? c : c - NLQ;
            const auto wqs = TensorAccessor(wql_args, get_common_arg_val<uint32_t>(13), BF16_TILE);
            const auto wks = TensorAccessor(wkl_args, get_common_arg_val<uint32_t>(14), BF16_TILE);
            volatile tt_l1_ptr uint32_t* sxa = sem_ptr(SEM_XA);
            volatile tt_l1_ptr uint32_t* sxb = sem_ptr(SEM_XB);
            cb_reserve_back(CB_X, KL);
            uint32_t pushed = 0;  // x chunks pushed (in K order: 0..7 from SEM_XA, 8..15 from SEM_XB)
            auto poll_x = [&]() {
                while (pushed < NXC) {
                    const uint32_t have = pushed < NXC / 2 ? sem_val(sxa) : NXC / 2 + sem_val(sxb);
                    if (have <= pushed) {
                        break;
                    }
                    cb_push_back(CB_X, XC);
                    ++pushed;
                }
            };
            for (uint32_t b = 0; b < NLB; b += 2) {
                // compute consumes a weight batch only after its x chunk is pushed, and only this RISC pushes x: never
                // block on a full ring while x chunks are pending (poll them instead)
                uint32_t sp = 0;
                while (!cb_pages_reservable_at_back(CB_LW0, LB) && ++sp < SPIN_LIMIT) {
                    poll_x();
                }
                cb_reserve_back(CB_LW0, LB);
                const uint32_t d = get_write_ptr(CB_LW0);
                for (uint32_t i = 0; i < LB; ++i) {
                    const uint32_t k = b * LB + i;
                    if (lat_q) {
                        noc_async_read(wqs.get_noc_addr(k * NT + n), d + i * BF16_TILE, BF16_TILE);
                    } else {
                        noc_async_read(wks.get_noc_addr(k * NT + n), d + i * BF16_TILE, BF16_TILE);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(CB_LW0, LB);
                poll_x();
            }
            uint32_t it = 0;
            while (pushed < NXC && ++it < SPIN_LIMIT) {
                poll_x();
            }
            *sxa = 0;
            *sxb = 0;
        }
        // ---- QB weights, K half 0 --------------------------------------------------------------------------------
        if (qb) {
            const uint32_t n = is_gate ? u - NQ : u;
            const uint32_t NT = is_gate ? NG : NQ;
            const uint32_t waddr = get_common_arg_val<uint32_t>(is_gate ? 5 : 4);
            cb_reserve_back(CB_WA, KH);
            const uint32_t d = get_write_ptr(CB_WA);
            if (MOTIF_AIN_EXP & 1) {
            } else if (is_gate) {
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
            for (uint32_t t = 0; t < ((MOTIF_AIN_EXP & 2) ? 0u : UK_PER); ++t) {
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
    if (lat) {  // odd weight batches, then the output tile -> QN / KN
        const uint32_t NT = lat_q ? NLQ : NLK;
        const uint32_t n = lat_q ? c : c - NLQ;
        const auto wqs = TensorAccessor(wql_args, get_common_arg_val<uint32_t>(13), BF16_TILE);
        const auto wks = TensorAccessor(wkl_args, get_common_arg_val<uint32_t>(14), BF16_TILE);
        for (uint32_t b = 1; b < NLB; b += 2) {
            cb_reserve_back(CB_LW1, LB);
            const uint32_t d = get_write_ptr(CB_LW1);
            for (uint32_t i = 0; i < LB; ++i) {
                const uint32_t k = b * LB + i;
                if (lat_q) {
                    noc_async_read(wqs.get_noc_addr(k * NT + n), d + i * BF16_TILE, BF16_TILE);
                } else {
                    noc_async_read(wks.get_noc_addr(k * NT + n), d + i * BF16_TILE, BF16_TILE);
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_LW1, LB);
        }
        if (lat_q) {
            const uint32_t qxy = xy_of(NCORES - 1);  // the QN core (11, 9) is the last core
            cb_wait_front(CB_LOQ, 1);
            noc_async_write(get_read_ptr(CB_LOQ), core_noc(qxy, get_write_ptr(CB_QX) + n * FP32_TILE), FP32_TILE);
            noc_async_write_barrier();
            noc_semaphore_inc(core_noc(qxy, get_semaphore(SEM_LAT)), 1);
            cb_pop_front(CB_LOQ, 1);
        } else {
            const uint32_t kxy = xy_of(NCORES - 2);  // the KN core (10, 9)
            const uint32_t dst = get_write_ptr(CB_KIN) + n * BF16_TILE;  // KN's landing zone (cb_x alias)
            cb_wait_front(CB_LOK, 1);
            noc_async_write(get_read_ptr(CB_LOK), core_noc(kxy, dst), BF16_TILE);
            noc_async_write_barrier();
            noc_semaphore_inc(core_noc(kxy, get_semaphore(SEM_LAT)), 1);
            cb_pop_front(CB_LOK, 1);
        }
    }
    if (qb) {
        const uint32_t n = is_gate ? u - NQ : u;
        const uint32_t NT = is_gate ? NG : NQ;
        cb_reserve_back(CB_WB, KH);
        const uint32_t d = get_write_ptr(CB_WB);
        if (MOTIF_AIN_EXP & 1) {
        } else if (is_gate) {
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
                noc_async_write(s, core_noc(xy_of(UK0 + qh * UK_CPH + i), dst), BF16_TILE);
            }
            noc_async_write_barrier();
            const uint32_t sa = get_semaphore(SEM_NOPE);
            for (uint32_t i = 0; i < UK_CPH; ++i) {
                noc_semaphore_inc(core_noc(xy_of(UK0 + qh * UK_CPH + i), sa), 1);
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
            noc_async_write(get_read_ptr(CB_GOUT), gs.get_noc_addr(u - NQ), BF16_TILE);
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
