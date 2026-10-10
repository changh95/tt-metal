// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 decode routed experts: dual-NoC all-core sparse expert matmul (phase D, DESIGN-3 stage 1; phase E stage 2),
// data movement. One source for both RISCs (CT RISC = 0 / 1, each on its own NoC). RISC r owns K half r of every unit:
//   in0 half r  (x for gate_up, h[e] for down)  -> cb_in0[r]
//   weights     W[e][k in half r][col]           -> cb_w[r]   (BATCH tiles per push)
// so both NoCs stream weights all the time (the stock 1D-mcast op reads in1 on one RISC only).
//
// Work: the active experts a[0..k) (sparsity != 0, in expert order) give U = k * NT units u = j * NT + col
// (expert-major); worker w (core c = W0 + w) owns the units of UnitOrder (at most 2 experts per core in mode 1).
// IN0_MODE 0 (gate_up, in0 shared by all experts): core 0 (also a worker) reads x half r in chunks of CH k and
//   multicasts each chunk to the same CB address of every other core, then sets their semaphore r to the chunk count;
//   the other workers push the chunks to compute as they land (polled between weight batches).
// IN0_MODE 1 (down, in0 = h[e] per expert): each worker reads half r of h[a[j]] at every expert change (cb_in0 holds
//   2 segments, so the reads of the next expert's h never wait for compute). With H_MC (stage 2, SUM_MODE 1) only
//   one worker of each expert reads h[e]: expert j's workers are the contiguous range [wa, wb]; s0, the first of them
//   whose first expert is j, reads its h half into its segment 0 and multicasts it to [s0 + 1, wb] (segment 0, up to 3
//   grid rectangles, data then a flag on the RISC's segment-0 semaphore) and unicasts it to wa's segment 1 when j is
//   wa's second expert. Receivers keep streaming weights and push h to compute when its flag is set.
//
// Sparsity:
//   SP_MODE 0 (stage 1): the sparsity stick (bf16 RM, sparsity != 0 = active) is read once per core.
//   SP_MODE 1 (stage 2, gate_up only): core 0 reads column 0 of the routing weights w_loc [1, E, M, 1] fp32 TILE
//     (faces 0 and 2 of each tile, into the unused cb_in0b of RISC 1 as scratch), sets expert e active when any row's
//     weight is nonzero (|w| bits != 0), writes the stick (bf16 1.0 / 0) to the sp_out tensor (the down call's
//     sparsity) and multicasts it with a ready flag (semaphore AUX) to every other core.
// Outputs:
//   SUM_MODE 0 (stage 1): compute packs MT tiles per unit into cb_out; RISC 1 writes them to y [1, E, M, N] after its
//     weight loop. Every core zero-fills a round-robin share of the inactive experts' output tiles (RISC 0, at the
//     end): the stock op's zeros_like output.
//   SUM_MODE 1 (stage 2, down only): the expert sum in the kernel. Column col is owned by core col % NCORES (slot
//     col / NCORES). RISC 1 writes each unit's MT bf16 tiles into the owner's cb_stage at slot (s E + e) MT (the same L1
//     address on every core), then (after one write barrier) increments the owner's semaphore AUX once per unit. An
//     owner waits for k increments per owned column, hands cb_stage to compute, which adds the E tiles of each output
//     tile in expert order into the fp32 DEST exactly as fast_reduce_nc does (add_tiles(y_e, zero) with acc_to_dest;
//     the stock y holds +0 for an inactive expert: an identity add, skipped) and packs cb_part; RISC 1 writes the
//     part [1, 1, M, N] tiles. No y tensor, no zero-fill.
// RISC 0 hands k and the active-expert bit mask to compute through cb_ctrl. Every spin is bounded (a bug can never
// hang the shared device).
//
// CT: 0 KT, 1 NT, 2 MT, 3 E, 4 NC, 5 W0, 6 GX, 7 BATCH, 8 RISC, 9 IN0_MODE, 10 CH, 11 cb_in0, 12 cb_w, 13 cb_out,
//     14 cb_sp, 15 cb_ctrl, 16 cb_zero, 17 sem_id, 18 mc_x0, 19 mc_y0, 20 mc_x1, 21 mc_y1, 22 mc_ndests,
//     23 sp_bytes, 24 SP_MODE, 25 SUM_MODE, 26 cb_stage, 27 cb_part, 28 sem_aux_id, 29 cb_scratch, 30 SLOTS,
//     31 H_MCAST (SUM_MODE 1: h multicast, H_MC), 32 sem_h_base (4 semaphores: RISC r segment s = base + 2 r + s),
//     33.. TensorAccessorArgs(w), (in0), (sp: the stick, or w_loc when SP_MODE 1), (out: y, or part when SUM_MODE 1),
//     (sp_out when SP_MODE 1, else a copy of sp's), (w_rep, or a copy of w's: the kernel always parses 6 accessors),
//     then E_NAT (DESIGN-2: experts [E_NAT, E) are replica slots read from w_rep; E_NAT = E without replicas)
// common RT: 0 w_addr, 1 in0_addr, 2 sp_addr, 3 out_addr, 4 sp_out_addr (0 unless SP_MODE 1),
//     5.. (SUM_MODE 1) the NOC coordinates (x << 16 | y) of core index 0 .. NCORES - 1, then w_rep_addr (0 if none)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"


// Unit order of worker w (identical in the dataflow and compute kernels; host mirror: moe_sparse_mm.unit_order).
// MODE 0 (gate_up, in0 shared): round-robin u = w + i NC (consecutive cores read consecutive columns, so the
//   bank-affine column streams spread over all DRAM banks at every step).
// MODE 1 (down, in0 per expert): the contiguous range [w U / NC, (w + 1) U / NC) (<= 2 experts), each expert's
//   sub-range [s, s + n) rotated to start at s + (w - s) mod n (decorrelates the cores' banks; one in0 load per expert).
struct UnitOrder {
    uint32_t n = 0, n1 = 0, s1 = 0, r1 = 0, s2 = 0, n2 = 0, r2 = 0, w = 0, nc = 1, mode = 0;
    void init(uint32_t mode_, uint32_t w_, uint32_t nc_, uint32_t U, uint32_t NT) {
        mode = mode_;
        w = w_;
        nc = nc_;
        if (mode == 0) {
            n = U > w ? (U - w - 1) / nc + 1 : 0;
            return;
        }
        const uint32_t u0 = w * U / nc, u1 = (w + 1) * U / nc;
        n = u1 > u0 ? u1 - u0 : 0;
        if (n == 0) {
            return;
        }
        const uint32_t b = (u0 / NT + 1) * NT;  // the next expert boundary
        s1 = u0;
        n1 = (b < u1 ? b : u1) - u0;
        r1 = (w + n1 * NT - s1 % n1) % n1;
        s2 = b;
        n2 = n - n1;
        r2 = n2 ? (w + n2 * NT - s2 % n2) % n2 : 0;
    }
    uint32_t at(uint32_t i) const {
        if (mode == 0) {
            return w + i * nc;
        }
        if (i < n1) {
            return s1 + (r1 + i) % n1;
        }
        return s2 + (r2 + i - n1) % n2;
    }
};

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
}  // namespace

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t MT = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t NC = get_compile_time_arg_val(4);
    constexpr uint32_t W0 = get_compile_time_arg_val(5);
    constexpr uint32_t GX = get_compile_time_arg_val(6);
    constexpr uint32_t BATCH = get_compile_time_arg_val(7);
    constexpr uint32_t RISC = get_compile_time_arg_val(8);
    constexpr uint32_t IN0_MODE = get_compile_time_arg_val(9);
    constexpr uint32_t CH = get_compile_time_arg_val(10);
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(11);
    constexpr uint32_t cb_w = get_compile_time_arg_val(12);
    constexpr uint32_t cb_out = get_compile_time_arg_val(13);
    constexpr uint32_t cb_sp = get_compile_time_arg_val(14);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(15);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(16);
    constexpr uint32_t sem_id = get_compile_time_arg_val(17);
    constexpr uint32_t mc_x0 = get_compile_time_arg_val(18);
    constexpr uint32_t mc_y0 = get_compile_time_arg_val(19);
    constexpr uint32_t mc_x1 = get_compile_time_arg_val(20);
    constexpr uint32_t mc_y1 = get_compile_time_arg_val(21);
    constexpr uint32_t mc_ndests = get_compile_time_arg_val(22);
    constexpr uint32_t sp_bytes = get_compile_time_arg_val(23);
    constexpr uint32_t SP_MODE = get_compile_time_arg_val(24);
    constexpr uint32_t SUM_MODE = get_compile_time_arg_val(25);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(26);
    constexpr uint32_t cb_part = get_compile_time_arg_val(27);
    constexpr uint32_t sem_aux_id = get_compile_time_arg_val(28);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(29);
    constexpr uint32_t SLOTS = get_compile_time_arg_val(30);
    constexpr uint32_t H_MCAST = get_compile_time_arg_val(31);
    constexpr uint32_t sem_h_base = get_compile_time_arg_val(32);
    constexpr auto w_args = TensorAccessorArgs<33>();
    constexpr auto in0_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto sp_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<sp_args.next_compile_time_args_offset()>();
    constexpr auto spo_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    // DESIGN-2 replica slots: experts e >= E_NAT live in a second weight tensor w_rep [1, E - E_NAT, K, N] (the 6th
    // accessor, a copy of w's when E_NAT == E); its address is the last common runtime arg.
    constexpr auto wr_args = TensorAccessorArgs<spo_args.next_compile_time_args_offset()>();
    constexpr uint32_t E_NAT = get_compile_time_arg_val(wr_args.next_compile_time_args_offset());

    constexpr uint32_t KH = KT / 2;
    constexpr uint32_t K0 = RISC * KH;
    constexpr uint32_t NCHUNK = KH / CH;
    constexpr uint32_t NCORES = mc_ndests + 1;
    static_assert(KT % 2 == 0 && KH % BATCH == 0 && KH % CH == 0, "K half must split into whole batches / chunks");
    static_assert(SP_MODE == 0 || IN0_MODE == 0, "the in-kernel sparsity is built by the gate_up call");
    static_assert(SUM_MODE == 0 || IN0_MODE == 1, "the in-kernel expert sum belongs to the down call");
    static_assert(E <= 16 && sp_bytes >= 2 * E, "one sparsity stick");
    static_assert(E_NAT >= 1 && E_NAT <= E, "native experts first, then the replica slots");

    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t in0_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(3);

    constexpr uint32_t wt = get_tile_size(cb_w);
    constexpr uint32_t it = get_tile_size(cb_in0);
    constexpr uint32_t ot = get_tile_size(cb_out);
    constexpr uint32_t pt = SUM_MODE ? get_tile_size(cb_part) : ot;  // the out tensor's page
    constexpr uint32_t spt = SP_MODE ? 4096u : sp_bytes;               // the sp tensor's page (fp32 tile / stick)
    const auto ws = TensorAccessor(w_args, w_addr, wt);
    const auto wrs = TensorAccessor(wr_args, get_common_arg_val<uint32_t>(5 + (SUM_MODE ? NCORES : 0)), wt);
    const auto xs = TensorAccessor(in0_args, in0_addr, it);
    const auto ss = TensorAccessor(sp_args, sp_addr, spt);
    const auto os = TensorAccessor(out_args, out_addr, pt);

    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    volatile tt_l1_ptr uint32_t* aux = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_aux_id));

    // ---- sparsity: one L1 copy per core ---------------------------------------------------------------------------
    uint32_t sp_l1;
    bool sp_sent = false;  // SP_MODE 1, core 0: writes to wait for before the end
    if constexpr (RISC == 0) {
        cb_reserve_back(cb_sp, 1);
        sp_l1 = get_write_ptr(cb_sp);
        if constexpr (SP_MODE == 0) {
            noc_async_read(ss.get_noc_addr(0), sp_l1, sp_bytes);
            noc_async_read_barrier();
        } else {
            if (c == 0) {
                const auto so = TensorAccessor(spo_args, get_common_arg_val<uint32_t>(4), sp_bytes);
                const uint32_t scr = get_write_ptr(cb_scratch);  // RISC 1's in0 CB: untouched until cb_sp is pushed
                for (uint32_t t = 0; t < E * MT; ++t) {          // column 0 lives in faces 0 (rows 0-15), 2 (16-31)
                    noc_async_read(ss.get_noc_addr(t, 0), scr + t * 2048, 1024);
                    noc_async_read(ss.get_noc_addr(t, 2048), scr + t * 2048 + 1024, 1024);
                }
                noc_async_read_barrier();
                invalidate_l1_cache();
                volatile tt_l1_ptr uint32_t* wv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
                volatile tt_l1_ptr uint16_t* st = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
                for (uint32_t i = 0; i < sp_bytes / 2; ++i) {
                    st[i] = 0;
                }
                for (uint32_t e = 0; e < E; ++e) {
                    uint32_t any = 0;
                    for (uint32_t mt = 0; mt < MT; ++mt) {
                        const uint32_t b = (e * MT + mt) * 512;  // row r, column 0: word r * 16 (faces 0 | 2)
                        for (uint32_t r = 0; r < 32; ++r) {
                            any |= wv[b + r * 16] & 0x7FFFFFFFu;
                        }
                    }
                    st[e] = any ? 0x3F80u : 0u;  // bf16 1.0
                }
                const uint32_t aux_addr = get_semaphore(sem_aux_id);
                const uint64_t mc_sp = noc_index == 0 ? get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, sp_l1)
                                                      : get_noc_multicast_addr(mc_x1, mc_y1, mc_x0, mc_y0, sp_l1);
                const uint64_t mc_aux = noc_index == 0
                                            ? get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, aux_addr)
                                            : get_noc_multicast_addr(mc_x1, mc_y1, mc_x0, mc_y0, aux_addr);
                *aux = 1;
                noc_async_write_multicast(sp_l1, mc_sp, sp_bytes, mc_ndests, true);
                noc_semaphore_set_multicast(aux_addr, mc_aux, mc_ndests, false);
                noc_async_write(sp_l1, so.get_noc_addr(0), sp_bytes);
                noc_async_writes_flushed();
                sp_sent = true;
            } else {
                wait_sem_min(aux, 1);
                *aux = 0;
            }
        }
        cb_push_back(cb_sp, 1);
    } else {
        cb_wait_front(cb_sp, 1);
        sp_l1 = get_read_ptr(cb_sp);
    }
    invalidate_l1_cache();
    uint32_t act[E];
    uint32_t inact[E];
    uint32_t k = 0, nz = 0, mask = 0;
    {
        volatile tt_l1_ptr uint16_t* sp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
        for (uint32_t e = 0; e < E; ++e) {
            if (sp[e] != 0) {
                act[k++] = e;
                mask |= 1u << e;
            } else {
                inact[nz++] = e;
            }
        }
    }
    if constexpr (RISC == 0) {
        cb_reserve_back(cb_ctrl, 1);
        volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctrl));
        ctl[0] = k;
        ctl[1] = mask;
        cb_push_back(cb_ctrl, 1);
    }

    const uint32_t U = k * NT;
    const bool worker = c >= W0 && c < W0 + NC;
    const uint32_t wi = worker ? c - W0 : 0;
    UnitOrder ord;
    if (worker) {
        ord.init(IN0_MODE, wi, NC, U, NT);
    }
    const uint32_t nu = ord.n;
    // SUM_MODE 1: the output columns this core owns (col = c + s NCORES)
    const uint32_t nslots = (SUM_MODE && c < NT) ? (NT - 1 - c) / NCORES + 1 : 0;

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    // ---- in0, mode 0: the multicast sender (core 0, also a worker) ------------------------------------------------
    uint32_t pushed = 0;  // mode 0: x chunks handed to compute
    if constexpr (IN0_MODE == 0) {
        if (c == 0 && k > 0) {
            const uint32_t base = get_write_ptr(cb_in0);  // the CB base: the same L1 address on every core
            const uint32_t sem_addr = get_semaphore(sem_id);
            const uint64_t mc_sem = noc_index == 0 ? get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, sem_addr)
                                                   : get_noc_multicast_addr(mc_x1, mc_y1, mc_x0, mc_y0, sem_addr);
            for (uint32_t ch = 0; ch < NCHUNK; ++ch) {
                const uint32_t cbase = base + ch * CH * MT * it;
                for (uint32_t kk = 0; kk < CH; ++kk) {
                    const uint32_t kg = K0 + ch * CH + kk;
                    for (uint32_t mt = 0; mt < MT; ++mt) {
                        noc_async_read(xs.get_noc_addr(mt * KT + kg), cbase + (kk * MT + mt) * it, it);
                    }
                }
                noc_async_read_barrier();
                const uint64_t mc_dst = noc_index == 0 ? get_noc_multicast_addr(mc_x0, mc_y0, mc_x1, mc_y1, cbase)
                                                       : get_noc_multicast_addr(mc_x1, mc_y1, mc_x0, mc_y0, cbase);
                *sem = ch + 1;
                noc_async_write_multicast(cbase, mc_dst, CH * MT * it, mc_ndests, true);
                noc_semaphore_set_multicast(sem_addr, mc_sem, mc_ndests, false);
                noc_async_writes_flushed();
            }
            if (worker && nu > 0) {
                cb_reserve_back(cb_in0, KH * MT);
                cb_push_back(cb_in0, KH * MT);
            }
            pushed = NCHUNK;
            noc_async_write_barrier();
            *sem = 0;
        }
    }

    // ---- H_MC (stage 2 down): h multicast helpers -------------------------------------------------------------------
    constexpr uint32_t H_MC = (SUM_MODE == 1 && H_MCAST) ? 1u : 0u;
    const uint32_t sem_h_id = sem_h_base + 2 * RISC;  // this RISC's segment 0 / 1 semaphores
    volatile tt_l1_ptr uint32_t* sem_h0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_h_id));
    volatile tt_l1_ptr uint32_t* sem_h1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_h_id + 1));
    bool sent = false;  // this core multicast an h (its local segment-0 semaphore is the source of the flag)
    // the worker owning unit u: the largest w with w U / NC <= u
    auto worker_of = [&](uint32_t u) -> uint32_t { return ((u + 1) * NC + U - 1) / U - 1; };
    // h (segment 0 at src) to the workers [a, b] (core c = W0 + w, rows of GX): up to 3 rectangles, data then flag
    auto mcast_rect = [&](uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t src, uint32_t bytes) {
        const uint32_t p0 = get_common_arg_val<uint32_t>(5 + y0 * GX + x0);
        const uint32_t p1 = get_common_arg_val<uint32_t>(5 + y1 * GX + x1);
        const uint32_t ax = p0 >> 16, ay = p0 & 0xFFFFu, bx = p1 >> 16, by = p1 & 0xFFFFu;
        const uint32_t nd = (x1 - x0 + 1) * (y1 - y0 + 1);
        const uint32_t sa = get_semaphore(sem_h_id);
        const uint64_t dd = noc_index == 0 ? get_noc_multicast_addr(ax, ay, bx, by, src)
                                           : get_noc_multicast_addr(bx, by, ax, ay, src);
        const uint64_t ds = noc_index == 0 ? get_noc_multicast_addr(ax, ay, bx, by, sa)
                                           : get_noc_multicast_addr(bx, by, ax, ay, sa);
        noc_async_write_multicast(src, dd, bytes, nd, true);
        noc_semaphore_set_multicast(sa, ds, nd, false);
    };
    auto mcast_range = [&](uint32_t a, uint32_t b, uint32_t src, uint32_t bytes) {
        *sem_h0 = 1;  // the flag value the receivers' segment-0 semaphore gets (this core never waits on it)
        const uint32_t ca = W0 + a, cbb = W0 + b;
        const uint32_t xa = ca % GX, ya = ca / GX, xb = cbb % GX, yb = cbb / GX;
        if (ya == yb) {
            mcast_rect(xa, ya, xb, yb, src, bytes);
            return;
        }
        mcast_rect(xa, ya, GX - 1, ya, src, bytes);
        if (yb > ya + 1) {
            mcast_rect(0, ya + 1, GX - 1, yb - 1, src, bytes);
        }
        mcast_rect(0, yb, xb, yb, src, bytes);
    };

    // ---- weights (and in0 for workers) ------------------------------------------------------------------------------
    auto poll_x = [&](bool block) {
        if constexpr (IN0_MODE == 0) {
            while (pushed < NCHUNK) {
                if (block) {
                    wait_sem_min(sem, pushed + 1);
                } else {
                    invalidate_l1_cache();
                    if (*sem < pushed + 1) {
                        return;
                    }
                }
                cb_push_back(cb_in0, CH * MT);
                ++pushed;
            }
        }
    };

    if (worker && nu > 0) {
        if constexpr (IN0_MODE == 0) {
            if (pushed < NCHUNK) {
                cb_reserve_back(cb_in0, KH * MT);
                poll_x(false);
            }
        }
        uint32_t cur_j = 0xFFFFFFFFu;
        uint32_t seg = 0;      // mode 1: the in0 segment of the current expert (0 = this core's first expert, 1 = second)
        bool h_wait = false;   // H_MC: the current expert's h is being multicast here; not pushed yet
        auto poll_h = [&](bool block) {
            if constexpr (IN0_MODE == 1 && H_MC) {
                if (!h_wait) {
                    return;
                }
                volatile tt_l1_ptr uint32_t* hs = seg ? sem_h1 : sem_h0;
                if (block) {
                    wait_sem_min(hs, 1);
                } else {
                    invalidate_l1_cache();
                    if (*hs < 1) {
                        return;
                    }
                }
                *hs = 0;
                cb_push_back(cb_in0, KH * MT);
                h_wait = false;
            }
        };
        for (uint32_t i = 0; i < nu; ++i) {
            const uint32_t u = ord.at(i);
            const uint32_t j = u / NT;
            const uint32_t col = u - j * NT;
            const uint32_t e = act[j];
            if constexpr (IN0_MODE == 1) {
                if (j != cur_j) {
                    poll_h(true);  // the previous expert's h first (the CB pushes stay in segment order)
                    seg = cur_j == 0xFFFFFFFFu ? 0 : 1;
                    cur_j = j;
                    cb_reserve_back(cb_in0, KH * MT);
                    const uint32_t dst0 = get_write_ptr(cb_in0);
                    // H_MC: expert j's workers are [wa, wb]; s0 (the first of them whose FIRST expert is j) reads h and
                    // multicasts it to [s0 + 1, wb] (their segment 0) and writes wa's segment 1 when j is wa's second.
                    uint32_t wa = 0, wb = 0, s0 = 0;
                    bool self_read = true, wa_first = true;
                    if constexpr (H_MC) {
                        wa = worker_of(j * NT);
                        wb = worker_of((j + 1) * NT - 1);
                        wa_first = wa * U / NC >= j * NT;
                        s0 = wa_first ? wa : wa + 1;
                        self_read = s0 > wb || wi == s0;  // no other core can send it / this core is the sender
                    }
                    if (self_read) {
                        uint32_t dst = dst0;
                        for (uint32_t kk = 0; kk < KH; ++kk) {
                            for (uint32_t mt = 0; mt < MT; ++mt) {
                                noc_async_read(xs.get_noc_addr((e * MT + mt) * KT + K0 + kk), dst, it);
                                dst += it;
                            }
                        }
                        noc_async_read_barrier();
                        if constexpr (H_MC) {
                            if (s0 <= wb) {  // the sender (its segment 0 = dst0)
                                if (wi < wb) {
                                    mcast_range(wi + 1, wb, dst0, KH * MT * it);
                                }
                                if (!wa_first) {  // wa holds j in its segment 1
                                    const uint32_t xy = get_common_arg_val<uint32_t>(5 + W0 + wa);
                                    noc_async_write(dst0, get_noc_addr(xy >> 16, xy & 0xFFFFu, dst0 + KH * MT * it),
                                                    KH * MT * it);
                                    noc_async_write_barrier();
                                    noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFFu, get_semaphore(sem_h_id + 1)), 1);
                                }
                                sent = true;
                            }
                        }
                        cb_push_back(cb_in0, KH * MT);
                    } else {
                        h_wait = true;  // lands in segment seg; pushed as soon as its semaphore is set
                        poll_h(false);
                    }
                }
            }
            const bool rep = (E_NAT < E) && e >= E_NAT;  // a replica slot: its weights are in w_rep
            uint32_t page = (rep ? e - E_NAT : e) * KT * NT + K0 * NT + col;
            for (uint32_t b = 0; b < KH / BATCH; ++b) {
                if constexpr (IN0_MODE == 0) {
                    // compute waits for all of x before it consumes weights: keep handing it chunks while cb_w is full
                    uint32_t spins = 0;
                    while (pushed < NCHUNK && !cb_pages_reservable_at_back(cb_w, BATCH)) {
                        poll_x(++spins >= SPIN_LIMIT);
                    }
                } else if constexpr (H_MC) {
                    // compute waits for h before it consumes this expert's weights: keep polling while cb_w is full
                    uint32_t spins = 0;
                    while (h_wait && !cb_pages_reservable_at_back(cb_w, BATCH)) {
                        poll_h(++spins >= SPIN_LIMIT);
                    }
                }
                cb_reserve_back(cb_w, BATCH);
                uint32_t dst = get_write_ptr(cb_w);
                for (uint32_t i = 0; i < BATCH; ++i) {
                    noc_async_read(rep ? wrs.get_noc_addr(page) : ws.get_noc_addr(page), dst, wt);
                    dst += wt;
                    page += NT;
                }
                noc_async_read_barrier();
                cb_push_back(cb_w, BATCH);
                if constexpr (IN0_MODE == 0) {
                    poll_x(false);
                } else {
                    poll_h(false);
                }
            }
        }
        poll_h(true);
        if constexpr (IN0_MODE == 0) {
            poll_x(true);
        }
    } else if (worker && IN0_MODE == 0 && k > 0 && c != 0) {
        // no units here, but the multicast still lands: wait for it so the semaphore can be re-armed
        wait_sem_min(sem, NCHUNK);
    }
    if constexpr (IN0_MODE == 0) {
        if (worker && k > 0 && c != 0) {
            *sem = 0;
        }
    }

    if constexpr (RISC == 1) {
        if constexpr (SUM_MODE == 1) {
            // ---- stage 2: this core's units -> the owners' cb_stage slots, then one increment per unit -------------
            const uint32_t stage_base = get_write_ptr(cb_stage);  // the CB base: the same L1 address on every core
            const uint32_t aux_addr = get_semaphore(sem_aux_id);
            for (uint32_t i = 0; i < nu; ++i) {
                const uint32_t u = ord.at(i);
                const uint32_t j = u / NT;
                const uint32_t col = u - j * NT;
                const uint32_t e = act[j];
                const uint32_t owner = col % NCORES;
                const uint32_t slot = col / NCORES;
                const uint32_t xy = get_common_arg_val<uint32_t>(5 + owner);
                cb_wait_front(cb_out, MT);
                noc_async_write(get_read_ptr(cb_out),
                                get_noc_addr(xy >> 16, xy & 0xFFFFu, stage_base + (slot * E + e) * MT * ot), MT * ot);
                noc_async_writes_flushed();
                cb_pop_front(cb_out, MT);
            }
            noc_async_write_barrier();  // the tiles have landed before the owners count them
            for (uint32_t i = 0; i < nu; ++i) {
                const uint32_t u = ord.at(i);
                const uint32_t col = u % NT;
                const uint32_t xy = get_common_arg_val<uint32_t>(5 + col % NCORES);
                noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFFu, aux_addr), 1);
            }
            noc_async_atomic_barrier();
            // ---- owner: wait for the k units of each owned column, sum (compute), write the part tiles -------------
            if (nslots > 0) {
                wait_sem_min(aux, k * nslots);
                *aux = 0;
                cb_reserve_back(cb_stage, SLOTS * E * MT);
                cb_push_back(cb_stage, SLOTS * E * MT);
                for (uint32_t s = 0; s < nslots; ++s) {
                    const uint32_t col = c + s * NCORES;
                    for (uint32_t mt = 0; mt < MT; ++mt) {
                        cb_wait_front(cb_part, 1);
                        noc_async_write(get_read_ptr(cb_part), os.get_noc_addr(mt * NT + col), pt);
                        noc_async_writes_flushed();
                        cb_pop_front(cb_part, 1);
                    }
                }
                noc_async_write_barrier();
            }
        } else {
            // ---- outputs of this core's units ---------------------------------------------------------------------
            for (uint32_t i = 0; i < nu; ++i) {
                const uint32_t u = ord.at(i);
                const uint32_t j = u / NT;
                const uint32_t col = u - j * NT;
                const uint32_t e = act[j];
                cb_wait_front(cb_out, MT);
                uint32_t src = get_read_ptr(cb_out);
                for (uint32_t mt = 0; mt < MT; ++mt) {
                    noc_async_write(src, os.get_noc_addr((e * MT + mt) * NT + col), ot);
                    src += ot;
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out, MT);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_sp, 1);
    } else {
        if constexpr (SUM_MODE == 1) {
            // ---- the zero tile the owner's compute adds for inactive experts (and pairs with every y tile) ----------
            if (nslots > 0) {
                cb_reserve_back(cb_zero, 1);
                volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_zero));
                for (uint32_t i = 0; i < get_tile_size(cb_zero) / 4; ++i) {
                    zp[i] = 0;
                }
                cb_push_back(cb_zero, 1);
            }
        } else {
            // ---- zero-fill: this core's round-robin share of the inactive experts' tiles --------------------------
            const uint32_t ztot = nz * MT * NT;
            if (c < ztot) {
                cb_reserve_back(cb_zero, 1);
                const uint32_t z = get_write_ptr(cb_zero);
                volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z);
                for (uint32_t i = 0; i < ot / 4; ++i) {
                    zp[i] = 0;
                }
                for (uint32_t t = c; t < ztot; t += NCORES) {
                    const uint32_t zi = t / (MT * NT);
                    const uint32_t r = t - zi * MT * NT;
                    noc_async_write(z, os.get_noc_addr(inact[zi] * MT * NT + r), ot);
                }
                noc_async_write_barrier();
            }
        }
        if constexpr (SP_MODE == 1) {
            if (sp_sent) {
                noc_async_write_barrier();
                *aux = 0;
            }
        }
    }
    if (sent) {  // H_MC sender: its flag source can be re-armed once the multicasts have landed
        noc_async_write_barrier();
        *sem_h0 = 0;
    }
}
