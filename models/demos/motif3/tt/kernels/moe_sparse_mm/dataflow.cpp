// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 decode routed experts: dual-NoC all-core sparse expert matmul (phase D, DESIGN-3 stage 1), data movement.
// One source for both RISCs (CT RISC = 0 / 1, each on its own NoC). RISC r owns K half r of every work unit:
//   in0 half r  (x for gate_up, h[e] for down)  -> cb_in0[r]
//   weights     W[e][k in half r][col]           -> cb_w[r]   (BATCH tiles per push)
// so both NoCs stream weights all the time (the stock 1D-mcast op reads in1 on one RISC only).
//
// Work: the active experts a[0..k) (sparsity != 0, in expert order) give U = k * NT units u = j * NT + col
// (expert-major); worker w (core c = W0 + w) owns the contiguous range [w U / NC, (w + 1) U / NC), at most 2 experts.
// IN0_MODE 0 (gate_up, in0 shared by all experts): core 0 (also a worker) reads x half r in chunks of CH k and
//   multicasts each chunk to the same CB address of every other core, then sets their semaphore r to the chunk count;
//   the other workers push the chunks to compute as they land (polled between weight batches).
// IN0_MODE 1 (down, in0 = h[e] per expert): each worker reads half r of h[a[j]] at every expert change (cb_in0 holds
//   2 segments, so the reads of the next expert's h never wait for compute).
// Outputs: compute packs MT tiles per unit into cb_out; RISC 1 writes them after its weight loop. Every core zero-fills
// a round-robin share of the inactive experts' output tiles (RISC 0, at the end): the stock op's zeros_like output.
// The sparsity stick is read once per core (RISC 0 -> cb_sp, RISC 1 reads the same L1 copy); RISC 0 hands the active
// count k to compute through cb_ctrl. Every spin is bounded (a bug can never hang the shared device).
//
// CT: 0 KT, 1 NT, 2 MT, 3 E, 4 NC, 5 W0, 6 GX, 7 BATCH, 8 RISC, 9 IN0_MODE, 10 CH, 11 cb_in0, 12 cb_w, 13 cb_out,
//     14 cb_sp, 15 cb_ctrl, 16 cb_zero, 17 sem_id, 18 mc_x0, 19 mc_y0, 20 mc_x1, 21 mc_y1, 22 mc_ndests,
//     23 sp_bytes, 24.. TensorAccessorArgs(w), (in0), (sp), (out)
// common RT: 0 w_addr, 1 in0_addr, 2 sp_addr, 3 out_addr

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
    constexpr auto w_args = TensorAccessorArgs<24>();
    constexpr auto in0_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto sp_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<sp_args.next_compile_time_args_offset()>();

    constexpr uint32_t KH = KT / 2;
    constexpr uint32_t K0 = RISC * KH;
    constexpr uint32_t NCHUNK = KH / CH;
    static_assert(KT % 2 == 0 && KH % BATCH == 0 && KH % CH == 0, "K half must split into whole batches / chunks");

    const uint32_t w_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t in0_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(3);

    constexpr uint32_t wt = get_tile_size(cb_w);
    constexpr uint32_t it = get_tile_size(cb_in0);
    constexpr uint32_t ot = get_tile_size(cb_out);
    const auto ws = TensorAccessor(w_args, w_addr, wt);
    const auto xs = TensorAccessor(in0_args, in0_addr, it);
    const auto ss = TensorAccessor(sp_args, sp_addr, sp_bytes);
    const auto os = TensorAccessor(out_args, out_addr, ot);

    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();

    // ---- sparsity: one L1 copy per core ---------------------------------------------------------------------------
    uint32_t sp_l1;
    if constexpr (RISC == 0) {
        cb_reserve_back(cb_sp, 1);
        sp_l1 = get_write_ptr(cb_sp);
        noc_async_read(ss.get_noc_addr(0), sp_l1, sp_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_sp, 1);
    } else {
        cb_wait_front(cb_sp, 1);
        sp_l1 = get_read_ptr(cb_sp);
    }
    invalidate_l1_cache();
    uint32_t act[E];
    uint32_t inact[E];
    uint32_t k = 0, nz = 0;
    {
        volatile tt_l1_ptr uint16_t* sp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
        for (uint32_t e = 0; e < E; ++e) {
            if (sp[e] != 0) {
                act[k++] = e;
            } else {
                inact[nz++] = e;
            }
        }
    }
    if constexpr (RISC == 0) {
        cb_reserve_back(cb_ctrl, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctrl))[0] = k;
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
        for (uint32_t i = 0; i < nu; ++i) {
            const uint32_t u = ord.at(i);
            const uint32_t j = u / NT;
            const uint32_t col = u - j * NT;
            const uint32_t e = act[j];
            if constexpr (IN0_MODE == 1) {
                if (j != cur_j) {
                    cur_j = j;
                    cb_reserve_back(cb_in0, KH * MT);
                    uint32_t dst = get_write_ptr(cb_in0);
                    for (uint32_t kk = 0; kk < KH; ++kk) {
                        for (uint32_t mt = 0; mt < MT; ++mt) {
                            noc_async_read(xs.get_noc_addr((e * MT + mt) * KT + K0 + kk), dst, it);
                            dst += it;
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_in0, KH * MT);
                }
            }
            uint32_t page = e * KT * NT + K0 * NT + col;
            for (uint32_t b = 0; b < KH / BATCH; ++b) {
                if constexpr (IN0_MODE == 0) {
                    // compute waits for all of x before it consumes weights: keep handing it chunks while cb_w is full
                    uint32_t spins = 0;
                    while (pushed < NCHUNK && !cb_pages_reservable_at_back(cb_w, BATCH)) {
                        poll_x(++spins >= SPIN_LIMIT);
                    }
                }
                cb_reserve_back(cb_w, BATCH);
                uint32_t dst = get_write_ptr(cb_w);
                for (uint32_t i = 0; i < BATCH; ++i) {
                    noc_async_read(ws.get_noc_addr(page), dst, wt);
                    dst += wt;
                    page += NT;
                }
                noc_async_read_barrier();
                cb_push_back(cb_w, BATCH);
                if constexpr (IN0_MODE == 0) {
                    poll_x(false);
                }
            }
        }
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
        // ---- outputs of this core's units -------------------------------------------------------------------------
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
        cb_pop_front(cb_sp, 1);
    } else {
        // ---- zero-fill: this core's round-robin share of the inactive experts' tiles ------------------------------
        constexpr uint32_t NCORES = mc_ndests + 1;
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
}
