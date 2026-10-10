// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused device-sampler per-chip stage (Phase F, F4; tt/kernels/sampler_local.py), data movement. One source for
// the three roles and both RISCs (CT ROLE, RISC); compute.cpp holds the math.
//
// PROD q (logical (x, y), y < GY - 1, q = y GX + x): vocab tiles k = q TPP .. min(q TPP + TPP, NT) - 1.
//   RISC 1: the logits, key and id tiles; once the AGG multicast the row broadcasts of M, 1 / T, c1, c2 into CB_CF /
//     CB_CI (SEM_RAW) hands them to compute; later the AGG's max-score row broadcast (SEM_M) -> CB_MS.
//   RISC 0: once compute packed exp(a) / exp(d), bumps the four ACC cores of that row (SEM_E); column 0 of the local
//     max-score tile -> row q % 32 of the AGG's CB_PM tile q / 32 (two 64 B writes), SEM_PM; later the local argmin
//     id / raw x - M -> CB_PL / CB_PD, SEM_PL.
// ACC j (logical (j, GY - 1), j < 8; row r = j / 4: exp(a) / exp(d); face f = j % 4): RISC 1 waits for every
//   producer (SEM_E = NP), then reads face f of the NT e tiles in vocab order, four per ring page. RISC 0 sends the
//   accumulated face to slot f of the row master's (f = 0) CB_AF (SEM_AF); the master hands the assembled tile to
//   compute, puts the reduced row sum in column 0 of a zero tile and writes part tile r.
// AGG (logical (8, GY - 1)): RISC 1 reads M, the 1 / T broadcast, the counters and their row broadcasts (host-written
//   rows of the counter tensor), builds rows 0..15 of the M broadcast (RISC 0 rows 16..31), multicasts CB_CF / CB_CI
//   to every producer (SEM_RAW); reads chip_off; fills the identity rows of CB_PM / PL / PD; waits for SEM_PM = NP;
//   builds rows 16..31 of the max-score broadcast; waits for SEM_PL = NP. RISC 0 builds e0, its halves of the
//   broadcasts, multicasts the max-score broadcast (SEM_M), writes u, puts m, gid = li + chip_off (exact integers) and
//   dw in column 0 of zero tiles (rows of a negative value: -0, the sign 0 * v takes) for the e0 multiplies, writes q.
// Every spin is bounded (a lost increment gives wrong data, never a hang).
//
// CT: 0 ROLE, 1 RISC, 2 NT, 3 TPP, 4 NP, 5 GX, 6 GY, 7 NPT, 8 RING, 9 EXP (experiments only: 256 = per-RISC
//     timeline -> the tl tensor, row 2 (y GX + x) + RISC), 10..13 producer rectangle NOC x0 y0 x1 y1, 14 / 15 AGG NOC
//     x / y, 16.. TensorAccessorArgs x, M, inv_t broadcast, ctr (3 pages: raw, c1 rows, c2 rows), gkey, iota,
//     chip_off, part, q, u, tl (a copy of u's without the timeline)
// common RT: 0..10 the eleven buffer addresses (same order), 11.. NOC x of logical column 0..GX-1, then NOC y of
//     logical row 0..GY-1

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common.h"

using namespace motif_sl;

namespace {
constexpr uint32_t SPIN_LIMIT = 200000000u;
volatile tt_l1_ptr uint32_t* tl_buf = nullptr;  // CB_TL page, 64 B per RISC (EXP & 256)
uint32_t tl_n = 0;
inline void stamp() {
    if (tl_buf != nullptr && tl_n < 16) {
        tl_buf[tl_n++] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    }
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

inline uint64_t mcast_addr(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t l1) {
    return noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, l1) : get_noc_multicast_addr(x1, y1, x0, y0, l1);
}

inline volatile tt_l1_ptr uint32_t* w32(uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); }

// dst[r][c] = v[r] for every c, rows r0 .. r1 - 1 (binary_ng's software column broadcast of a [32, 1] operand)
inline void bcast_rows(uint32_t dst, const uint32_t* v, uint32_t r0 = 0, uint32_t r1 = 32) {
    tt_l1_ptr uint32_t* d = reinterpret_cast<tt_l1_ptr uint32_t*>(dst);
    for (uint32_t r = r0; r < r1; ++r) {
        const uint32_t x = v[r];
        tt_l1_ptr uint32_t* p = d + tw(r, 0);
#pragma GCC unroll 16
        for (uint32_t c = 0; c < 16; ++c) {
            p[c] = x;
            p[256 + c] = x;
        }
    }
    asm volatile("" ::: "memory");
}

inline void col_of(uint32_t src, uint32_t col, uint32_t* v) {  // column `col` of a tile (lane r in row r)
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* s = w32(src);
    for (uint32_t r = 0; r < 32; ++r) {
        v[r] = s[tw(r, col)];
    }
}

inline void row0_of(uint32_t src, uint32_t* v) {  // row 0 of a tile (a REDUCE_COL result: lane r in column r)
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* s = w32(src);
    for (uint32_t r = 0; r < 32; ++r) {
        v[r] = s[tw(0, r)];
    }
}

// x[r][0] = v[r]; x[r][c > 0] = +0, or -0 when v[r] has its sign bit set: the operand of e0 * x whose products are
// exactly the column-broadcast multiply's (1 * v in column 0, 0 * v = sign(v) 0 elsewhere; v finite)
inline void zero_tiles(uint32_t dst, uint32_t n) {  // +0 from the hardware zeros (NoC reads; barrier included)
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t i = 0; i < n * FP32_TILE / MEM_ZEROS_SIZE; ++i) {
        noc_async_read(zeros, dst + i * MEM_ZEROS_SIZE, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
}
inline void build_col0(uint32_t dst, const uint32_t* v, uint32_t negpat) {  // dst: a zero tile (zero_tiles)
    bool neg = false;
    for (uint32_t r = 0; r < 32; ++r) {
        if (v[r] & 0x80000000u) {
            noc_async_write(negpat, get_noc_addr(dst + tw(r, 0) * 4), 64);
            noc_async_write(negpat, get_noc_addr(dst + tw(r, 16) * 4), 64);
            neg = true;
        }
    }
    if (neg) {
        noc_async_write_barrier();
    }
    volatile tt_l1_ptr uint32_t* d = w32(dst);
    for (uint32_t r = 0; r < 32; ++r) {
        d[tw(r, 0)] = v[r];
    }
}

// exact non-negative integers below 2^24 <-> their fp32 bit patterns (the fp32 add li + chip_off is exact)
inline uint32_t f2u(uint32_t b) {
    if ((b & 0x7FFFFFFFu) == 0) {
        return 0;
    }
    const int32_t e = static_cast<int32_t>((b >> 23) & 0xFF) - 127;
    const uint32_t m = (b & 0x7FFFFFu) | 0x800000u;
    return e >= 23 ? (m << (e - 23)) : (m >> (23 - e));
}
inline uint32_t u2f(uint32_t u) {
    if (u == 0) {
        return 0;
    }
    const uint32_t e = 31 - __builtin_clz(u);
    return ((e + 127) << 23) | ((u << (23 - e)) & 0x7FFFFFu);
}

// e0: 1.0 in column 0, 0 elsewhere (the sampler's _e0 / _e0_64 tiles)
inline void build_e0(uint32_t dst) {
    tt_l1_ptr uint32_t* d = reinterpret_cast<tt_l1_ptr uint32_t*>(dst);
#pragma GCC unroll 8
    for (uint32_t i = 0; i < 1024; ++i) {
        d[i] = 0;
    }
    for (uint32_t r = 0; r < 32; ++r) {
        d[tw(r, 0)] = ONE_F;
    }
    asm volatile("" ::: "memory");
}

// column 0 of a tile -> 32 contiguous words; then row q % 32 of the AGG tile q / 32 (two 64 B face rows)
inline void send_col0_row(uint32_t src_tile, uint32_t stg, uint32_t q, uint32_t agg_x, uint32_t agg_y, uint32_t cb_dst) {
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* s = w32(src_tile);
    volatile tt_l1_ptr uint32_t* g = w32(stg);
    for (uint32_t r = 0; r < 32; ++r) {
        g[r] = s[tw(r, 0)];
    }
    const uint32_t i = q & 31;
    const uint32_t base = get_write_ptr(cb_dst) + (q >> 5) * FP32_TILE;  // same L1 address on the AGG (CB spans all)
    noc_async_write(stg, get_noc_addr(agg_x, agg_y, base + tw(i, 0) * 4), 64);
    noc_async_write(stg + 64, get_noc_addr(agg_x, agg_y, base + tw(i, 16) * 4), 64);
}

inline void fill_rows(uint32_t tile, uint32_t i0, uint32_t i1, uint32_t v) {
    volatile tt_l1_ptr uint32_t* d = w32(tile);
    for (uint32_t i = i0; i < i1; ++i) {
        for (uint32_t c = 0; c < 32; ++c) {
            d[tw(i, c)] = v;
        }
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t RISC = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr uint32_t TPP = get_compile_time_arg_val(3);
    constexpr uint32_t NP = get_compile_time_arg_val(4);
    constexpr uint32_t GX = get_compile_time_arg_val(5);
    constexpr uint32_t GY = get_compile_time_arg_val(6);
    constexpr uint32_t NPT = get_compile_time_arg_val(7);
    constexpr uint32_t RING = get_compile_time_arg_val(8);
    constexpr uint32_t EXP = get_compile_time_arg_val(9);
    constexpr uint32_t PX0 = get_compile_time_arg_val(10);
    constexpr uint32_t PY0 = get_compile_time_arg_val(11);
    constexpr uint32_t PX1 = get_compile_time_arg_val(12);
    constexpr uint32_t PY1 = get_compile_time_arg_val(13);
    constexpr uint32_t GNX = get_compile_time_arg_val(14);
    constexpr uint32_t GNY = get_compile_time_arg_val(15);
    constexpr auto x_args = TensorAccessorArgs<16>();
    constexpr auto m_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto u_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto tl_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr bool TL = (EXP & 256) != 0;
    static_assert(TPP >= 1 && (NP - 1) * TPP < NT && NP * TPP >= NT, "sampler_local: every producer owns 1..TPP tiles");
    static_assert(NPT * 32 >= NP, "sampler_local: NPT partial tiles hold every producer row");
    static_assert(GX > AGG_X, "sampler_local: the last grid row holds 8 ACC cores and the AGG");

    constexpr uint32_t NARGS = 11;
    auto colx = [](uint32_t x) { return get_common_arg_val<uint32_t>(NARGS + x); };
    auto rowy = [](uint32_t y) { return get_common_arg_val<uint32_t>(NARGS + GX + y); };
    const uint32_t lx = static_cast<uint32_t>(get_absolute_logical_x());
    const uint32_t ly = static_cast<uint32_t>(get_absolute_logical_y());
    if constexpr (TL) {
        tl_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_TL) + RISC * 64u);
        for (uint32_t i = 0; i < 16; ++i) {
            tl_buf[i] = 0;
        }
    }
    stamp();

    if constexpr (ROLE == ROLE_PROD) {
        const uint32_t q = ly * GX + lx;
        const uint32_t k0 = q * TPP;
        const uint32_t n = (k0 + TPP <= NT) ? TPP : NT - k0;
        volatile tt_l1_ptr uint32_t* sraw = w32(get_semaphore(SEM_RAW));
        if constexpr (RISC == 1) {
            const auto xs = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), BF16_TILE);
            const auto ks = TensorAccessor(k_args, get_common_arg_val<uint32_t>(4), FP32_TILE);
            const auto is = TensorAccessor(i_args, get_common_arg_val<uint32_t>(5), FP32_TILE);
            cb_reserve_back(CB_GK, n);
            const uint32_t kb = get_write_ptr(CB_GK);
            for (uint32_t j = 0; j < n; ++j) {
                noc_async_read_page(k0 + j, ks, kb + j * FP32_TILE);
            }
            cb_reserve_back(CB_X, n);
            const uint32_t xb = get_write_ptr(CB_X);
            for (uint32_t j = 0; j < n; ++j) {
                noc_async_read_page(k0 + j, xs, xb + j * BF16_TILE);
            }
            cb_reserve_back(CB_IO, n);
            const uint32_t ib = get_write_ptr(CB_IO);
            for (uint32_t j = 0; j < n; ++j) {
                noc_async_read_page(k0 + j, is, ib + j * FP32_TILE);
            }
            // the row broadcasts of c1, c2, then M, 1 / T (multicast by the AGG into CB_CI / CB_CF)
            cb_reserve_back(CB_CI, 2);
            volatile tt_l1_ptr uint32_t* sci = w32(get_semaphore(SEM_CI));
            wait_min(sci, 1);
            *sci = 0;
            stamp();
            cb_push_back(CB_CI, 2);
            noc_async_read_barrier();
            cb_push_back(CB_GK, n);
            cb_push_back(CB_X, n);
            cb_push_back(CB_IO, n);
            cb_reserve_back(CB_CF, 2);
            wait_min(sraw, 1);
            *sraw = 0;
            stamp();
            cb_push_back(CB_CF, 2);
            // the AGG's max-score broadcast (written into CB_MS, same address on every core)
            cb_reserve_back(CB_MS, 1);
            volatile tt_l1_ptr uint32_t* sm = w32(get_semaphore(SEM_M));
            wait_min(sm, 1);
            stamp();
            *sm = 0;
            invalidate_l1_cache();
            cb_push_back(CB_MS, 1);
        } else {
            const uint32_t stg = get_write_ptr(CB_STG);
            const uint32_t sem_e = get_semaphore(SEM_E);
            cb_wait_front(CB_EA, n);
            stamp();
            for (uint32_t j = 0; j < NACC / 2; ++j) {
                noc_semaphore_inc(get_noc_addr(colx(j), rowy(GY - 1), sem_e), 1);
            }
            cb_wait_front(CB_ED, n);
            for (uint32_t j = NACC / 2; j < NACC; ++j) {
                noc_semaphore_inc(get_noc_addr(colx(j), rowy(GY - 1), sem_e), 1);
            }
            cb_wait_front(CB_W, 1);
            stamp();
            send_col0_row(get_read_ptr(CB_W), stg, q, GNX, GNY, CB_PM);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(GNX, GNY, get_semaphore(SEM_PM)), 1);
            cb_wait_front(CB_W2, 2);
            stamp();
            send_col0_row(get_read_ptr(CB_W2), stg + 128, q, GNX, GNY, CB_PL);
            send_col0_row(get_read_ptr(CB_W2) + FP32_TILE, stg + 256, q, GNX, GNY, CB_PD);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(GNX, GNY, get_semaphore(SEM_PL)), 1);
            noc_async_atomic_barrier();
            cb_pop_front(CB_W, 1);
            cb_pop_front(CB_W2, 2);
            cb_pop_front(CB_EA, n);
            cb_pop_front(CB_ED, n);
        }
    } else if constexpr (ROLE == ROLE_ACC) {
        const uint32_t r = lx / 4;  // 0: exp(a) (CB_EA), 1: exp(d) (CB_ED)
        const uint32_t f = lx % 4;  // the face this core folds
        const uint32_t mx = colx(4 * r), my = rowy(GY - 1);  // the row master (f = 0)
        if constexpr (RISC == 1) {
            const uint32_t src = get_read_ptr(r == 0 ? CB_EA : CB_ED) + f * FACE;  // same address on every producer
            volatile tt_l1_ptr uint32_t* se = w32(get_semaphore(SEM_E));
            wait_min(se, NP);
            stamp();
            *se = 0;
            invalidate_l1_cache();
            constexpr uint32_t NSP = (NT + 3) / 4;
            constexpr uint32_t B = RING / 3;
            for (uint32_t p = 0; p < NSP; p += B) {
                const uint32_t nb = (p + B <= NSP) ? B : NSP - p;
                cb_reserve_back(CB_RING, nb);
                uint32_t dst = get_write_ptr(CB_RING);
                for (uint32_t i = 0; i < 4 * nb; ++i) {
                    const uint32_t k = 4 * p + i;
                    if (k < NT) {
                        const uint32_t pq = k / TPP, j = k - pq * TPP;
                        noc_async_read(get_noc_addr(colx(pq % GX), rowy(pq / GX), src + j * FP32_TILE), dst, FACE);
                    }
                    dst += FACE;
                }
                noc_async_read_barrier();
                cb_push_back(CB_RING, nb);
                if (p == 0 || p + B >= NSP) {
                    stamp();
                }
            }
        } else {
            if (f == 0) {
                cb_reserve_back(CB_E0, 1);
                build_e0(get_write_ptr(CB_E0));
                cb_push_back(CB_E0, 1);
                cb_reserve_back(CB_AF, 1);
                cb_reserve_back(CB_SF, 1);
                zero_tiles(get_write_ptr(CB_SF), 1);
            }
            cb_wait_front(CB_S2, 1);
            stamp();
            const uint32_t af = get_write_ptr(CB_AF);  // same address on every ACC core
            noc_async_write(get_read_ptr(CB_S2), get_noc_addr(mx, my, af + f * FACE), FACE);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(mx, my, get_semaphore(SEM_AF)), 1);
            noc_async_atomic_barrier();
            cb_pop_front(CB_S2, 1);
            if (f == 0) {
                const auto ps = TensorAccessor(p_args, get_common_arg_val<uint32_t>(7), FP32_TILE);
                volatile tt_l1_ptr uint32_t* saf = w32(get_semaphore(SEM_AF));
                wait_min(saf, 4);
                stamp();
                *saf = 0;
                invalidate_l1_cache();
                cb_push_back(CB_AF, 1);
                cb_wait_front(CB_S2B, 1);
                uint32_t v[32];
                col_of(get_read_ptr(CB_S2B), 0, v);
                build_col0(get_write_ptr(CB_SF), v, 0);  // s >= +0: no negative rows
                cb_push_back(CB_SF, 1);
                cb_pop_front(CB_S2B, 1);
                cb_wait_front(CB_OUT, 1);
                stamp();
                noc_async_write(get_read_ptr(CB_OUT), ps.get_noc_addr(r), FP32_TILE);
                noc_async_write_barrier();
                cb_pop_front(CB_OUT, 1);
            }
        }
    } else {  // ROLE_AGG
        volatile tt_l1_ptr uint32_t* l0 = w32(get_semaphore(SEM_L0));
        volatile tt_l1_ptr uint32_t* l1s = w32(get_semaphore(SEM_L1));
        volatile tt_l1_ptr uint32_t* l2 = w32(get_semaphore(SEM_L2));
        volatile tt_l1_ptr uint32_t* l3 = w32(get_semaphore(SEM_L3));
        const uint32_t cf = get_write_ptr(CB_CF);
        const uint32_t ms = get_write_ptr(CB_MS);
        if constexpr (RISC == 1) {
            const auto mss = TensorAccessor(m_args, get_common_arg_val<uint32_t>(1), FP32_TILE);
            const auto ts = TensorAccessor(t_args, get_common_arg_val<uint32_t>(2), FP32_TILE);
            const auto cs = TensorAccessor(c_args, get_common_arg_val<uint32_t>(3), FP32_TILE);
            const auto os = TensorAccessor(o_args, get_common_arg_val<uint32_t>(6), FP32_TILE);
            cb_reserve_back(CB_RAW, 1);
            cb_reserve_back(CB_RAWI, 1);
            cb_reserve_back(CB_CF, 2);
            cb_reserve_back(CB_CI, 2);
            const uint32_t raw = get_write_ptr(CB_RAW);
            const uint32_t ci = get_write_ptr(CB_CI);
            noc_async_read_page(1, cs, ci);              // the host's c1 / c2 row broadcasts
            noc_async_read_page(2, cs, ci + FP32_TILE);
            noc_async_read_page(0, mss, raw);
            noc_async_read_page(0, ts, cf + FP32_TILE);  // the host's 1 / T row broadcast
            noc_async_read_page(0, cs, get_write_ptr(CB_RAWI));
            noc_async_read_barrier();
            stamp();
            *l0 = 1;
            // c1, c2 -> every producer (same addresses), then SEM_CI (linked: the semaphore lands after the data)
            const uint32_t sem_ci = get_semaphore(SEM_CI);
            volatile tt_l1_ptr uint32_t* sc = w32(sem_ci);
            *sc = 1;
            asm volatile("" ::: "memory");
            noc_async_write_multicast(ci, mcast_addr(PX0, PY0, PX1, PY1, ci), 2 * FP32_TILE, NP, true);
            noc_semaphore_set_multicast(sem_ci, mcast_addr(PX0, PY0, PX1, PY1, sem_ci), NP, false);
            uint32_t v[32];
            col_of(raw, 0, v);
            bcast_rows(cf, v, 0, 16);
            wait_min(l1s, 1);
            *l1s = 0;
            noc_async_write_barrier();
            *sc = 0;
            // M, 1 / T -> every producer, then SEM_RAW
            const uint32_t sem_raw = get_semaphore(SEM_RAW);
            volatile tt_l1_ptr uint32_t* sr = w32(sem_raw);
            *sr = 1;
            asm volatile("" ::: "memory");
            noc_async_write_multicast(cf, mcast_addr(PX0, PY0, PX1, PY1, cf), 2 * FP32_TILE, NP, true);
            noc_semaphore_set_multicast(sem_raw, mcast_addr(PX0, PY0, PX1, PY1, sem_raw), NP, false);
            noc_async_write_barrier();
            *sr = 0;
            stamp();
            cb_push_back(CB_RAWI, 1);  // the counters for u
            cb_reserve_back(CB_OFF, 1);
            noc_async_read_page(0, os, get_write_ptr(CB_OFF));
            // identity rows of the partial tiles (rows NP .. 32 NPT - 1; the producers write rows 0 .. NP - 1)
            cb_reserve_back(CB_PM, NPT);
            cb_reserve_back(CB_PL, NPT);
            cb_reserve_back(CB_PD, NPT);
            for (uint32_t t = NP / 32; t < NPT; ++t) {
                const uint32_t i0 = (t == NP / 32) ? (NP & 31) : 0;
                fill_rows(get_write_ptr(CB_PM) + t * FP32_TILE, i0, 32, NEG_INF);
                fill_rows(get_write_ptr(CB_PL) + t * FP32_TILE, i0, 32, POS_INF);
                fill_rows(get_write_ptr(CB_PD) + t * FP32_TILE, i0, 32, NEG_INF);
            }
            noc_async_read_barrier();
            volatile tt_l1_ptr uint32_t* spm = w32(get_semaphore(SEM_PM));
            wait_min(spm, NP);
            stamp();
            *spm = 0;
            invalidate_l1_cache();
            cb_push_back(CB_PM, NPT);
            // rows 16..31 of the max-score broadcast (RISC 0 builds 0..15 and multicasts)
            wait_min(l2, 1);
            *l2 = 0;
            row0_of(get_read_ptr(CB_GR), v);  // this RISC never pops CB_GR: its read pointer is page 0
            bcast_rows(ms, v, 16, 32);
            *l3 = 1;
            volatile tt_l1_ptr uint32_t* spl = w32(get_semaphore(SEM_PL));
            wait_min(spl, NP);
            stamp();
            *spl = 0;
            invalidate_l1_cache();
            cb_push_back(CB_PL, NPT);
            cb_push_back(CB_PD, NPT);
        } else {
            const auto qs = TensorAccessor(q_args, get_common_arg_val<uint32_t>(8), FP32_TILE);
            const auto us = TensorAccessor(u_args, get_common_arg_val<uint32_t>(9), FP32_TILE);
            const uint32_t negpat = get_write_ptr(CB_STG);
            for (uint32_t i = 0; i < 16; ++i) {
                w32(negpat)[i] = 0x80000000u;
            }
            cb_reserve_back(CB_E0, 1);
            build_e0(get_write_ptr(CB_E0));
            cb_push_back(CB_E0, 1);
            cb_reserve_back(CB_SF, 3);
            const uint32_t sf = get_write_ptr(CB_SF);
            // rows 16..31 of the M broadcast (RISC 1 builds 0..15 and multicasts)
            uint32_t v[32];
            wait_min(l0, 1);
            *l0 = 0;
            col_of(get_write_ptr(CB_RAW), 0, v);
            bcast_rows(cf, v, 16, 32);
            *l1s = 1;
            zero_tiles(sf, 3);  // the q operands' zero tiles
            // m: rows 0..15 of the broadcast (RISC 1: 16..31), multicast to every producer's CB_MS, then SEM_M
            cb_wait_front(CB_GR, 1);
            stamp();
            *l2 = 1;
            uint32_t mv[32];
            row0_of(get_read_ptr(CB_GR), mv);
            cb_reserve_back(CB_MS, 1);
            bcast_rows(ms, mv, 0, 16);
            wait_min(l3, 1);
            *l3 = 0;
            const uint32_t sem_m = get_semaphore(SEM_M);
            volatile tt_l1_ptr uint32_t* sm = w32(sem_m);
            *sm = 1;
            asm volatile("" ::: "memory");
            noc_async_write_multicast(ms, mcast_addr(PX0, PY0, PX1, PY1, ms), FP32_TILE, NP, true);
            noc_semaphore_set_multicast(sem_m, mcast_addr(PX0, PY0, PX1, PY1, sem_m), NP, false);
            noc_async_write_barrier();
            stamp();
            *sm = 0;
            cb_push_back(CB_MS, 1);
            cb_pop_front(CB_GR, 1);
            // the operands of q = e0 * [m; gid; dw]
            build_col0(sf, mv, negpat);
            cb_wait_front(CB_U, 1);
            noc_async_write(get_read_ptr(CB_U), us.get_noc_addr(0), FP32_TILE);
            cb_wait_front(CB_GR, 2);
            stamp();
            uint32_t lv[32], dv[32], ov[32];
            row0_of(get_read_ptr(CB_GR), lv);
            row0_of(get_read_ptr(CB_GR) + FP32_TILE, dv);
            col_of(get_write_ptr(CB_OFF), 0, ov);  // read by RISC 1 before its SEM_PM wait
            for (uint32_t r = 0; r < 32; ++r) {
                lv[r] = u2f(f2u(lv[r]) + f2u(ov[r]));  // gid = li + chip_off (exact: integers < 2^24)
            }
            build_col0(sf + FP32_TILE, lv, negpat);
            build_col0(sf + 2 * FP32_TILE, dv, negpat);
            cb_push_back(CB_SF, 3);
            cb_pop_front(CB_GR, 2);
            cb_wait_front(CB_OUT, 3);
            stamp();
            const uint32_t ob = get_read_ptr(CB_OUT);
            for (uint32_t t = 0; t < 3; ++t) {
                noc_async_write(ob + t * FP32_TILE, qs.get_noc_addr(t), FP32_TILE);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_OUT, 3);
            cb_pop_front(CB_U, 1);
        }
    }
    if constexpr (TL) {
        stamp();
        const auto tls = TensorAccessor(tl_args, get_common_arg_val<uint32_t>(10), 64);
        noc_async_write(get_write_ptr(CB_TL) + RISC * 64u, tls.get_noc_addr(2 * (ly * GX + lx) + RISC, 0), 64);
        noc_async_write_barrier();
    }
}
