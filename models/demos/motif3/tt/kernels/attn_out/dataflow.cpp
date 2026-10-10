// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode attention output chain (Phase F, F2; tt/kernels/attn_out.py), data movement.
// One source for both RISCs (CT RISC 0 / 1, each on its own NoC) and the two roles (CT ROLE). The traffic is split by
// NoC: RISC 0 carries the W_UV reads and most of the wo weight stream (FUSED); RISC 1 carries the dependency chain
// (gather, row writes, quarter swap, u tiles, CMB inputs, dg aggregation) and its share of the stream only after its
// chain work: a weight stream on the same NoC delayed the chain's small messages by several us (measured).
//
// ROLE 0 (the 8-wide left block; core (x, y), x < 8):
//   UV (y < 5): unit u = 8 y + x = (h, t), h = u / 4, t = u % 4: one output tile of the per-head W_UV bmm
//       u[h] = o_heads[h] @ W_UV[h] (K = 16), o_heads[h] = rows h of o_lat [1, L, 10, 512] (the transpose is
//       addressing). RISC 1, level 1: core (h, t) reads the blocks b = h, h + 10, .. of quarter t (block b = (k, l,
//       face): rows 0..9 of one face of o_lat tile (l, k), 320 B, ONE DRAM read for all 10 heads) and writes row h' of
//       each block to row l of tile k of core (h', t)'s cb_in0 (stateful 32 B writes, grouped by destination), then
//       bumps SEM_L1 on the quarter's 10 cores; RISC 0 zero-fills rows L..31 of the core's quarter and bumps SEM_L1
//       here (11 = complete). Level 2 (RISC 1): the quarter (4 tiles) -> the head's other 3 cores, SEM_IN0 there
//       (3 = complete: cb_in0 pushed). RISC 0 reads the 16 W_UV tiles (k, t) into cb_wa / cb_wb first. After
//       compute (RISC 1): the u tile -> the CMB core of combine tile 4 h + t (signal heads, cb_sig) or the 4 CMB
//       cores of its group (noise heads, cb_noise), + their semaphore.
//   WO (FUSED, y < 8): w = 8 y + x: wo output tiles n0 = 2 w, n0 + 1, K = 32 in 8 chunks of 4 k rows (cb_woa: 0..3,
//       cb_wob: 4, 5, cb_woc: 6, 7). RISC 0 takes chunks 0 .. 7 - n1, RISC 1 the last n1 (CT N1_UV / N1_WO). Every
//       read of a chunk carries the transaction id 1 + c; a chunk is pushed once fully issued and flushed (in
//       order); issuing blocks under DRAM back-pressure, so each read is followed by a poll of the RISC's other
//       duties. Start: UV cores once their W_UV tiles landed (RISC 0) / their u tile left (RISC 1); WO-only cores at
//       WO_DELAY / WO_LATE wall-clock cycles. RISC 0 pushes cb_dgin when SEM_DG = 32 and writes the 2 output tiles.
// ROLE 1 (CMB, core (8 + x, y), x < 4, y < 8): combine tile j = 4 y + x: RISC 1 reads lam, E tiles (0, j) / (1, j),
//   g and active tile j; RISC 0 pushes the u tiles as they land (SEM_SIG / SEM_NOISE); RISC 1 then writes the output
//   tile: dg tile j to DRAM (stage "uv") or into slot j of the aggregator's (CMB core j = 0) cb_dgin; the aggregator
//   multicasts the 32 tiles to the WO rectangle in ONE write and sets SEM_DG = 32 there (FUSED; 32 concurrent
//   multicasts into one rectangle serialize: ~75 us measured).
// Every spin is bounded (a bug never hangs the shared device; it only produces garbage).
//
// CT: 0 ROLE, 1 RISC, 2 L, 3 FUSED, 4 DEBUG, 5..8 WO multicast rect (NOC x0, y0, x1, y1), 9 its ndests, 10 GX,
//     11 WO_DELAY, 12 WO_LATE (wall-clock cycles), 13 / 14 wo chunks on RISC 1 of UV / WO-only cores (0, 2, 4),
//     15.. TensorAccessorArgs of o_lat, w_uv, lam, E, g, active, w_o, out, dbg_u, dbg_v, dbg_dg,
//     tl
// common RT: 0..11 the 12 tensor addresses (same order), 12.. the NOC xy (x << 16 | y) of core c = y GX + x

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common.h"

using namespace motif_aout;

#ifndef MOTIF_AOUT_EXP
#define MOTIF_AOUT_EXP 0  // experiments only: 1 no wo weight reads, 2 no o_lat gather, 4 no W_UV weight reads,
                          // 8 no CMB input reads, 32 no cross-core waits (CMB / WO push at once),
                          // 256 timeline: wall-clock stamps of every RISC -> the tl tensor (row 2 c + RISC)
#endif
constexpr uint32_t EXP = MOTIF_AOUT_EXP;
constexpr bool TL = (EXP & 256) != 0;
#define STAMP(i)                                                  \
    do {                                                          \
        if constexpr (TL) {                                       \
            tl_buf[(i)] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L); \
        }                                                         \
    } while (0)

namespace {
constexpr uint32_t SPIN_LIMIT = 200000000u;
constexpr uint32_t NARGS = 12;  // tensor addresses before the core xy table

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

// byte offset of row r (0..31) of a bf16 tile (faces 0 / 1 for r < 16, else 2 / 3), left face
inline uint32_t row_off(uint32_t r) { return (r < 16 ? 0u : 1024u) + (r & 15u) * 32u; }

// zero rows L..31 of the bf16 tile at dst (local NoC reads of the hardware zero region, <= 512 B each)
template <uint32_t L>
inline void zero_rows(uint32_t dst) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    if constexpr (L < 16) {
        noc_async_read(z, dst + L * 32u, (16u - L) * 32u);
        noc_async_read(z, dst + 512u + L * 32u, (16u - L) * 32u);
        noc_async_read(z, dst + 1024u, 512u);
        noc_async_read(z, dst + 1536u, 512u);
    } else if constexpr (L < 32) {
        noc_async_read(z, dst + 1024u + (L - 16u) * 32u, (32u - L) * 32u);
        noc_async_read(z, dst + 1536u + (L - 16u) * 32u, (32u - L) * 32u);
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t RISC = get_compile_time_arg_val(1);
    constexpr uint32_t L = get_compile_time_arg_val(2);
    constexpr uint32_t FUSED = get_compile_time_arg_val(3);
    constexpr uint32_t DEBUG = get_compile_time_arg_val(4);
    constexpr uint32_t wm_x0 = get_compile_time_arg_val(5);
    constexpr uint32_t wm_y0 = get_compile_time_arg_val(6);
    constexpr uint32_t wm_x1 = get_compile_time_arg_val(7);
    constexpr uint32_t wm_y1 = get_compile_time_arg_val(8);
    constexpr uint32_t wm_ndests = get_compile_time_arg_val(9);
    constexpr uint32_t GX = get_compile_time_arg_val(10);
    constexpr uint32_t WO_DELAY = get_compile_time_arg_val(11);
    constexpr uint32_t WO_LATE = get_compile_time_arg_val(12);
    constexpr uint32_t N1_UV = get_compile_time_arg_val(13);
    constexpr uint32_t N1_WO = get_compile_time_arg_val(14);
    constexpr auto ol_args = TensorAccessorArgs<15>();
    constexpr auto wuv_args = TensorAccessorArgs<ol_args.next_compile_time_args_offset()>();
    constexpr auto lam_args = TensorAccessorArgs<wuv_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<lam_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr auto a_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto wo_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<wo_args.next_compile_time_args_offset()>();
    constexpr auto du_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto dv_args = TensorAccessorArgs<du_args.next_compile_time_args_offset()>();
    constexpr auto dd_args = TensorAccessorArgs<dv_args.next_compile_time_args_offset()>();
    constexpr auto tl_args = TensorAccessorArgs<dd_args.next_compile_time_args_offset()>();
    static_assert(L >= 1 && L <= 32, "one tile row of lanes");

    auto xy_of = [](uint32_t c) { return get_common_arg_val<uint32_t>(NARGS + c); };
    const uint32_t x = static_cast<uint32_t>(get_absolute_logical_x());
    const uint32_t y = static_cast<uint32_t>(get_absolute_logical_y());
    const uint32_t tl0 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile tt_l1_ptr uint32_t* tl_buf =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_TL) + RISC * 64u);
    if constexpr (TL) {
        for (uint32_t i = 0; i < 16; ++i) {
            tl_buf[i] = 0;
        }
        tl_buf[0] = tl0;
    }
    auto tl_flush = [&]() {
        if constexpr (TL) {
            const auto tls = TensorAccessor(tl_args, get_common_arg_val<uint32_t>(11), 64);
            noc_async_write(get_write_ptr(CB_TL) + RISC * 64u, tls.get_noc_addr(2 * (y * GX + x) + RISC), 64);
            noc_async_write_barrier();
        }
    };

    // =================================================================================================================
    if constexpr (ROLE == 1) {  // CMB: combine tile j
        const uint32_t j = y * TPH + (x - CMB_X0);
        if constexpr (RISC == 0) {  // the u tiles land at the cb bases (written by the UV cores)
            cb_reserve_back(CB_SIG, 1);
            cb_reserve_back(CB_NOISE, 1);
            volatile tt_l1_ptr uint32_t* ss = sem_ptr(SEM_SIG);
            volatile tt_l1_ptr uint32_t* sn = sem_ptr(SEM_NOISE);
            if constexpr ((EXP & 32) == 0) {
                wait_sem_min(ss, 1);
                *ss = 0;
            }
            STAMP(2);
            cb_push_back(CB_SIG, 1);
            if constexpr ((EXP & 32) == 0) {
                wait_sem_min(sn, 1);
                *sn = 0;
            }
            STAMP(3);
            cb_push_back(CB_NOISE, 1);
            tl_flush();
            return;
        }
        // RISC 1: inputs, then the output tile
        {
            const auto lams = TensorAccessor(lam_args, get_common_arg_val<uint32_t>(2), BF16_TILE);
            const auto es = TensorAccessor(e_args, get_common_arg_val<uint32_t>(3), BF16_TILE);
            const auto gs = TensorAccessor(g_args, get_common_arg_val<uint32_t>(4), BF16_TILE);
            const auto as = TensorAccessor(a_args, get_common_arg_val<uint32_t>(5), BF16_TILE);
            cb_reserve_back(CB_LAM, NLAMT);
            cb_reserve_back(CB_E, NLAMT);
            cb_reserve_back(CB_G, 1);
            cb_reserve_back(CB_ACT, 1);
            const uint32_t dl = get_write_ptr(CB_LAM);
            const uint32_t de = get_write_ptr(CB_E);
            if constexpr ((EXP & 8) == 0) {
                for (uint32_t k = 0; k < NLAMT; ++k) {
                    noc_async_read(lams.get_noc_addr(k), dl + k * BF16_TILE, BF16_TILE);
                    noc_async_read(es.get_noc_addr(k * NCMB + j), de + k * BF16_TILE, BF16_TILE);
                }
                noc_async_read(gs.get_noc_addr(j), get_write_ptr(CB_G), BF16_TILE);
                noc_async_read(as.get_noc_addr(j), get_write_ptr(CB_ACT), BF16_TILE);
            }
            noc_async_read_barrier();
            STAMP(1);
            cb_push_back(CB_LAM, NLAMT);
            cb_push_back(CB_E, NLAMT);
            cb_push_back(CB_G, 1);
            cb_push_back(CB_ACT, 1);
        }
        if constexpr (DEBUG) {
            const auto dvs = TensorAccessor(dv_args, get_common_arg_val<uint32_t>(9), BF16_TILE);
            cb_wait_front(CB_VDBG, 1);
            noc_async_write(get_read_ptr(CB_VDBG), dvs.get_noc_addr(j), BF16_TILE);
            noc_async_write_barrier();
            cb_pop_front(CB_VDBG, 1);
        }
        cb_wait_front(CB_OUTC, 1);
        STAMP(4);
        const uint32_t s = get_read_ptr(CB_OUTC);
        if constexpr (FUSED) {
            const uint32_t dgin = get_write_ptr(CB_DGIN);  // same address on every core
            const uint32_t agg_xy = xy_of(CMB_X0);         // core (8, 0)
            volatile tt_l1_ptr uint32_t* sdg = sem_ptr(SEM_DG);
            noc_async_write(s, core_noc(agg_xy, dgin + j * BF16_TILE), BF16_TILE);
            noc_async_write_barrier();
            STAMP(5);
            if (j != 0) {
                noc_semaphore_inc(core_noc(agg_xy, get_semaphore(SEM_DG)), 1);
            } else {
                if constexpr ((EXP & 32) == 0) {
                    wait_sem_min(sdg, NCMB - 1);
                }
                STAMP(6);
                const uint32_t sa = get_semaphore(SEM_DG);
                noc_async_write_multicast(dgin, mcast_addr(wm_x0, wm_y0, wm_x1, wm_y1, dgin), KWO * BF16_TILE,
                                          wm_ndests, true);
                *sdg = KWO;
                noc_semaphore_set_multicast(sa, mcast_addr(wm_x0, wm_y0, wm_x1, wm_y1, sa), wm_ndests, false);
                noc_async_write_barrier();
                STAMP(7);
                *sdg = 0;
            }
            if constexpr (DEBUG) {
                const auto dds = TensorAccessor(dd_args, get_common_arg_val<uint32_t>(10), BF16_TILE);
                noc_async_write(s, dds.get_noc_addr(j), BF16_TILE);
            }
        } else {
            const auto os = TensorAccessor(out_args, get_common_arg_val<uint32_t>(7), BF16_TILE);
            noc_async_write(s, os.get_noc_addr(j), BF16_TILE);
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        cb_pop_front(CB_OUTC, 1);
        STAMP(8);
        tl_flush();
        return;
    }

    // =================================================================================================================
    // ROLE 0
    const bool uv = y < NUV / UV_GX;
    const uint32_t u = y * UV_GX + x;
    const uint32_t h = u / TPH, t = u % TPH;
    const uint32_t w = y * WO_GX + x;
    const uint32_t n0 = NWO_PER * w;
    uint32_t spins = 0;

    // ---- the wo weight stream (FUSED): chunk c = k rows 4c .. 4c + 3 (8 tiles) -> cb_woa (c < 4) / cb_wob (4, 5) /
    // cb_woc (6, 7); RISC 0 issues chunks 0..5 (UV cores) or 0..3, RISC 1 the rest; one transaction id per chunk
    // (1 + c), pushed in order once fully issued and flushed. Issuing blocks under DRAM back-pressure, so every read is
    // followed by a poll of this RISC's other duties.
    constexpr uint32_t CH_T = WO_CHUNK * NWO_PER;  // tiles per chunk
    const uint32_t n1 = uv ? N1_UV : N1_WO;  // chunks on RISC 1 (0, 2 or 4: the last ones)
    const uint32_t c_lo = RISC == 0 ? 0u : NWCH - n1;
    const uint32_t c_hi = RISC == 0 ? NWCH - n1 : NWCH;
    uint32_t base_a = 0, base_b = 0, base_c = 0;
    uint32_t issued = c_lo, pushed = c_lo;
    auto cb_of = [](uint32_t c) { return c < 4 ? CB_WOA : (c < 6 ? CB_WOB : CB_WOC); };
    auto reserve_wo = [&]() {
        if constexpr (FUSED) {
            if (c_lo < 4) {
                cb_reserve_back(CB_WOA, 4 * CH_T);
                base_a = get_write_ptr(CB_WOA);
            }
            if (c_lo <= 4 && c_hi >= 6) {
                cb_reserve_back(CB_WOB, 2 * CH_T);
                base_b = get_write_ptr(CB_WOB);
            }
            if (c_lo <= 6 && c_hi == NWCH) {
                cb_reserve_back(CB_WOC, 2 * CH_T);
                base_c = get_write_ptr(CB_WOC);
            }
        }
    };
    auto poll_chunks = [&](bool force) {
        if (pushed < issued && (force || ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + pushed))) {
            cb_push_back(cb_of(pushed), CH_T);
            ++pushed;
            if (RISC == 0 && pushed == 4) {
                STAMP(3);
            }
        }
    };
    auto issue_chunks = [&](auto&& poll) {
        if constexpr (FUSED) {
            const auto wos = TensorAccessor(wo_args, get_common_arg_val<uint32_t>(6), BF16_TILE);
            for (uint32_t c = c_lo; c < c_hi; ++c) {
                const uint32_t d = (c < 4 ? base_a + c * CH_T * BF16_TILE
                                          : (c < 6 ? base_b + (c - 4) * CH_T * BF16_TILE
                                                   : base_c + (c - 6) * CH_T * BF16_TILE));
                for (uint32_t kk = 0; kk < ((EXP & 1) ? 0u : WO_CHUNK); ++kk) {
                    const uint32_t k = c * WO_CHUNK + kk;
                    for (uint32_t i = 0; i < NWO_PER; ++i) {
                        noc_async_read_set_trid(1 + c);
                        noc_async_read(wos.get_noc_addr(k * NWO_T + n0 + i), d + (kk * NWO_PER + i) * BF16_TILE,
                                       BF16_TILE);
                        noc_async_read_set_trid(0);
                        poll(false);
                    }
                }
                issued = c + 1;
            }
        }
    };

    if constexpr (RISC == 0) {
        // ---- UV prelude (off the chain's NoC): zero rows L..31 of this core's quarter, then SEM_L1 += 1 here; the 16
        // W_UV tiles (k, t) -> cb_wa / cb_wb (issued first; pushed when landed, polled in the stream's issue loop)
        constexpr uint32_t TU = 13, TZ = 14;
        bool w_pushed = !uv;
        if (uv) {
            const uint32_t in0 = get_write_ptr(CB_IN0);
            noc_async_read_set_trid(TZ);
            for (uint32_t kk = 0; kk < KQ4; ++kk) {
                zero_rows<L>(in0 + (KQ4 * t + kk) * BF16_TILE);
            }
            const auto wus = TensorAccessor(wuv_args, get_common_arg_val<uint32_t>(1), BF16_TILE);
            noc_async_read_set_trid(TU);
            cb_reserve_back(CB_WA, KH);
            cb_reserve_back(CB_WB, KH);
            const uint32_t wa = get_write_ptr(CB_WA), wb = get_write_ptr(CB_WB);
            for (uint32_t k = 0; k < ((EXP & 4) ? 0u : KUV); ++k) {
                noc_async_read(wus.get_noc_addr((h * KUV + k) * TPH + t), (k < KH ? wa : wb) + (k % KH) * BF16_TILE,
                               BF16_TILE);
            }
            noc_async_read_set_trid(0);
            noc_async_read_barrier_with_trid(TZ);
            noc_semaphore_inc(get_noc_addr(get_semaphore(SEM_L1)), 1);  // this core's zero fill is done
        }
        auto poll_w = [&](bool force) {
            if (!w_pushed && (force || ncrisc_noc_read_with_transaction_id_flushed(noc_index, TU))) {
                cb_push_back(CB_WA, KH);
                cb_push_back(CB_WB, KH);
                w_pushed = true;
                STAMP(8);
            }
        };
        // ---- the wo stream (chunks 0..5), cb_dgin, the output tiles ---------------------------------------------------
        while (!w_pushed || (FUSED && reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - tl0 < WO_DELAY)) {
            poll_w(++spins >= SPIN_LIMIT);
            if (!FUSED && spins >= 2 * SPIN_LIMIT) {
                break;
            }
            if (FUSED && !uv && reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - tl0 >= WO_DELAY) {
                break;  // WO-only core: the stream starts at WO_DELAY (UV cores: once their W_UV tiles landed)
            }
        }
        if constexpr (FUSED) {
            STAMP(1);
            reserve_wo();
            cb_reserve_back(CB_DGIN, KWO);  // the aggregator's multicast lands at the cb base
            volatile tt_l1_ptr uint32_t* sdg = sem_ptr(SEM_DG);
            bool dg_pushed = false;
            auto poll = [&](bool force) {
                poll_w(force);
                poll_chunks(force);
                if (!dg_pushed && ((EXP & 32) || force || sem_val(sdg) >= KWO)) {
                    *sdg = 0;
                    cb_push_back(CB_DGIN, KWO);
                    dg_pushed = true;
                    STAMP(5);
                }
            };
            issue_chunks(poll);
            STAMP(2);
            while (pushed < c_hi || !dg_pushed || !w_pushed) {
                poll(++spins >= SPIN_LIMIT);  // a bug produces garbage, never a hang
                if (spins >= 2 * SPIN_LIMIT) {
                    break;
                }
            }
            STAMP(4);
            const auto os = TensorAccessor(out_args, get_common_arg_val<uint32_t>(7), BF16_TILE);
            cb_wait_front(CB_WOUT, NWO_PER);
            STAMP(6);
            const uint32_t s = get_read_ptr(CB_WOUT);
            for (uint32_t i = 0; i < NWO_PER; ++i) {
                noc_async_write(s + i * BF16_TILE, os.get_noc_addr(n0 + i), BF16_TILE);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_WOUT, NWO_PER);
            STAMP(7);
        }
        noc_async_read_barrier();
        noc_async_atomic_barrier();
        tl_flush();
        return;
    }

    // ---- RISC 1 ----------------------------------------------------------------------------------------------------
    if (!uv) {  // WO-only core: its share of the stream (chunks 6, 7) from WO_LATE (UV cores: after their chain work)
        if constexpr (FUSED) {
            while (reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - tl0 < WO_LATE) {
            }
            reserve_wo();
            issue_chunks(poll_chunks);
            while (pushed < c_hi) {
                poll_chunks(++spins >= SPIN_LIMIT);
                if (spins >= 2 * SPIN_LIMIT) {
                    break;
                }
            }
            noc_async_read_barrier();
        }
        tl_flush();
        return;
    }
    // the UV chain
    constexpr uint32_t NB1 = KQ4 * L * 2;  // level-1 blocks per quarter
    auto uv_xy = [&](uint32_t uu) { return xy_of((uu / UV_GX) * GX + uu % UV_GX); };
    const uint32_t in0 = get_write_ptr(CB_IN0);
    const uint32_t stage = (get_write_ptr(CB_STAGE) + 63u) & ~63u;  // 64 B-aligned, 320 B per block
    constexpr uint32_t TG = 15;  // (1 .. 8: the wo chunks; 13, 14: RISC 0)
    {
        const auto ols = TensorAccessor(ol_args, get_common_arg_val<uint32_t>(0), BF16_TILE);
        noc_async_read_set_trid(TG);
        uint32_t i = 0;
        for (uint32_t b = h; b < NB1; b += H, ++i) {
            const uint32_t k = KQ4 * t + b / (2 * L), l = (b / 2) % L, f = b % 2;
            if constexpr ((EXP & 2) == 0) {
                noc_async_read(ols.get_noc_addr(l * KUV + k, f * 512u), stage + i * ROWB, ROWB);
            }
        }
        noc_async_read_set_trid(0);
        cb_reserve_back(CB_IN0, KUV);  // every quarter lands at the cb base (one address on every core)
        STAMP(1);
        // level 1: row h' of every block -> core (h', t)
        noc_async_read_barrier_with_trid(TG);
        STAMP(2);
        // by destination: one write state per core, then a 32 B write per block (the stateful one-packet API)
        for (uint32_t hh = 0; hh < H; ++hh) {
            noc_async_write_one_packet_set_state(core_noc(uv_xy(hh * TPH + t), in0), 32);
            i = 0;
            for (uint32_t b = h; b < NB1; b += H, ++i) {
                const uint32_t k = KQ4 * t + b / (2 * L), l = (b / 2) % L, f = b % 2;
                noc_async_write_one_packet_with_state(stage + i * ROWB + hh * 32u,
                                                      in0 + k * BF16_TILE + f * 512u + row_off(l));
            }
        }
        noc_async_write_barrier();
        for (uint32_t hh = 0; hh < H; ++hh) {
            noc_semaphore_inc(core_noc(uv_xy(hh * TPH + t), get_semaphore(SEM_L1)), 1);
        }
        STAMP(3);
    }
    bool g2_sent = false, in0_pushed = false, u_sent = false;
    volatile tt_l1_ptr uint32_t* sl1 = sem_ptr(SEM_L1);
    volatile tt_l1_ptr uint32_t* si0 = sem_ptr(SEM_IN0);
    const uint32_t qsrc = in0 + KQ4 * t * BF16_TILE;
    auto chain = [&](bool force) {
        poll_chunks(force);
        if (!g2_sent && (force || sem_val(sl1) >= H + 1)) {  // the 10 level-1 writers + this core's zero fill
            *sl1 = 0;
            STAMP(4);
            for (uint32_t i = 0; i < TPH; ++i) {  // level 2: the quarter -> the head's other 3 cores
                if (i != t) {
                    noc_async_write(qsrc, core_noc(uv_xy(h * TPH + i), qsrc), KQ4 * BF16_TILE);
                }
            }
            noc_async_write_barrier();
            for (uint32_t i = 0; i < TPH; ++i) {
                if (i != t) {
                    noc_semaphore_inc(core_noc(uv_xy(h * TPH + i), get_semaphore(SEM_IN0)), 1);
                }
            }
            g2_sent = true;
        }
        if (g2_sent && !in0_pushed && (force || sem_val(si0) >= TPH - 1)) {
            *si0 = 0;
            cb_push_back(CB_IN0, KUV);
            in0_pushed = true;
            STAMP(5);
        }
        if (in0_pushed && !u_sent && cb_pages_available_at_front(CB_U, 1)) {  // the u tile -> the CMB core(s)
            STAMP(6);
            const uint32_t s = get_read_ptr(CB_U);
            if (h < SG) {
                const uint32_t xy = xy_of(h * GX + CMB_X0 + t);  // CMB core (8 + t, h) holds combine tile 4 h + t
                noc_async_write(s, core_noc(xy, get_write_ptr(CB_SIG)), BF16_TILE);
                noc_async_write_barrier();
                noc_semaphore_inc(core_noc(xy, get_semaphore(SEM_SIG)), 1);
            } else {
                const uint32_t g0 = (h - SG) * HPG;
                for (uint32_t i = 0; i < HPG; ++i) {
                    noc_async_write(s, core_noc(xy_of((g0 + i) * GX + CMB_X0 + t), get_write_ptr(CB_NOISE)),
                                    BF16_TILE);
                }
                noc_async_write_barrier();
                for (uint32_t i = 0; i < HPG; ++i) {
                    noc_semaphore_inc(core_noc(xy_of((g0 + i) * GX + CMB_X0 + t), get_semaphore(SEM_NOISE)), 1);
                }
            }
            if constexpr (DEBUG) {
                const auto dus = TensorAccessor(du_args, get_common_arg_val<uint32_t>(8), BF16_TILE);
                noc_async_write(s, dus.get_noc_addr(h * TPH + t), BF16_TILE);
                noc_async_write_barrier();
            }
            cb_pop_front(CB_U, 1);
            u_sent = true;
            STAMP(7);
        }
    };
    while (!in0_pushed) {
        chain(++spins >= SPIN_LIMIT);  // a bug produces garbage, never a hang
        if (spins >= 2 * SPIN_LIMIT) {
            break;
        }
    }
    while (!u_sent) {  // the chain's NoC traffic first
        chain(++spins >= SPIN_LIMIT);
        if (spins >= 2 * SPIN_LIMIT) {
            break;
        }
    }
    reserve_wo();
    issue_chunks(chain);  // (FUSED) this RISC's share of the wo stream
    while (!u_sent || pushed < issued) {
        chain(++spins >= SPIN_LIMIT);
        if (spins >= 2 * SPIN_LIMIT) {
            break;
        }
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    tl_flush();
}
