// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused shared-expert tail (Phase F, F3; tt/kernels/shared_tail.py), data movement. One source for both roles and
// both RISCs (CT ROLE, RISC).
//
// ROLE 0, APPLY core j (j = 0..NA-1, logical (AX, AY0 + j)): the B5 apply program (shared_polynorm/apply_*.cpp) split
//   over the NA cores (compute_apply.cpp).
//   RISC 1 (reader): the 7 constants (64 B each, broadcast down column 0, as apply_reader.cpp), for j < 3 the TP
//     gathered tiles of moment j (faces 0 and 2), gate tile j and up tile NH + j; then it waits until the 4
//     coefficient tiles landed in cb_coef (semaphore COEF = 4) and hands them to compute.
//   RISC 0 (writer): core j < 4 sends its coefficient tile (cb_co, fp32) to slot j of every APPLY core's cb_coef and
//     bumps their COEF; then the h tile (bf16) goes to slot j of APPLY core 0's cb_h (the same L1 address on every core
//     of the program) and core j > 0 bumps core 0's AGG. Core 0 waits for NA - 1 bumps and multicasts the NA tiles (one
//     transfer: concurrent multicasts into one rectangle serialize, F2) to every DOWN core's cb_h, then sets their
//     semaphore H (linked).
// ROLE 1, DOWN core d (d = y DGX + x over the DGX x DGY rectangle at logical (0, 0)): output tiles NPER d .. NPER d +
//   NPER - 1 of y [1, 1, T, N].
//   RISC 1 (reader): the NH x NPER W_down tiles (k-major, as the stock in1 block), issued at launch (they overlap the
//     apply), then waits for H and hands cb_h (NH tiles) to compute.
//   RISC 0 (writer): fills a 1 KB scratch from the hardware zeros while it waits, then writes each output tile's rows
//     0..T-1 from cb_dout and rows T..31 from the scratch (padding = +0, no L1 copy), T in {8, 16, 32}.
// Every spin is bounded (a lost increment gives wrong data, never a hang).
//
// CT: 0 ROLE, 1 RISC, 2 T, 3 NA, 4 NH (h tiles = KT), 5 TP, 6 NPER, 7 DGX, 8 AX (logical x of the apply column),
//     9 AY0, 10 a0_noc_x, 11 a0_noc_y, 12..15 down rectangle NOC x0, y0, x1, y1, 16 ND (down cores), 17 NT_OUT,
//     18 cb_recv, 19 cb_k, 20 cb_ag, 21 cb_au, 22 cb_aout, 23 cb_h, 24 cb_w, 25 cb_dout, 26 sem_agg, 27 sem_h,
//     28 cb_coef, 29 cb_co, 30 sem_coef, 31 cb_z, 32 apply column NOC x, 33..37 the 5 apply cores' NOC y,
//     38.. TensorAccessorArgs(gat), (gu), (D), (E), (b), (w), (y), (tl: the timeline, or a copy of y's)
// common RT: 0 gat, 1 gu, 2 D, 3 E, 4 b, 5 w, 6 y, 7 tl (buffer addresses)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#ifndef MOTIF_STAIL_EXP
#define MOTIF_STAIL_EXP 0  // experiments only (timing; wrong results except 256): tt/kernels/shared_tail.py
#endif

namespace {
constexpr uint32_t SPIN_LIMIT = 200000000u;
constexpr uint32_t EXP = MOTIF_STAIL_EXP;
constexpr bool TL = (EXP & 256) != 0;  // timeline: wall-clock stamps of every RISC -> the tl tensor (row 2 c + RISC)
constexpr uint32_t CB_TL = 15;          // tt/kernels/shared_tail.py (experiments only)
volatile tt_l1_ptr uint32_t* tl_buf = nullptr;  // CB_TL page, 64 B per RISC
uint32_t tl_n = 0;
inline void stamp() {
    if constexpr (TL) {
        if (tl_n < 16) {
            tl_buf[tl_n++] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        }
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
}  // namespace

void kernel_main() {
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr uint32_t RISC = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr uint32_t NA = get_compile_time_arg_val(3);
    constexpr uint32_t NH = get_compile_time_arg_val(4);
    constexpr uint32_t TP = get_compile_time_arg_val(5);
    constexpr uint32_t NPER = get_compile_time_arg_val(6);
    constexpr uint32_t DGX = get_compile_time_arg_val(7);
    constexpr uint32_t AY0 = get_compile_time_arg_val(9);
    constexpr uint32_t A0X = get_compile_time_arg_val(10);
    constexpr uint32_t A0Y = get_compile_time_arg_val(11);
    constexpr uint32_t DX0 = get_compile_time_arg_val(12);
    constexpr uint32_t DY0 = get_compile_time_arg_val(13);
    constexpr uint32_t DX1 = get_compile_time_arg_val(14);
    constexpr uint32_t DY1 = get_compile_time_arg_val(15);
    constexpr uint32_t ND = get_compile_time_arg_val(16);
    constexpr uint32_t NT_OUT = get_compile_time_arg_val(17);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(18);
    constexpr uint32_t cb_k = get_compile_time_arg_val(19);
    constexpr uint32_t cb_ag = get_compile_time_arg_val(20);
    constexpr uint32_t cb_au = get_compile_time_arg_val(21);
    constexpr uint32_t cb_aout = get_compile_time_arg_val(22);
    constexpr uint32_t cb_h = get_compile_time_arg_val(23);
    constexpr uint32_t cb_w = get_compile_time_arg_val(24);
    constexpr uint32_t cb_dout = get_compile_time_arg_val(25);
    constexpr uint32_t sem_agg_id = get_compile_time_arg_val(26);
    constexpr uint32_t sem_h_id = get_compile_time_arg_val(27);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(28);
    constexpr uint32_t cb_co = get_compile_time_arg_val(29);
    constexpr uint32_t sem_coef_id = get_compile_time_arg_val(30);
    constexpr uint32_t cb_z = get_compile_time_arg_val(31);
    constexpr uint32_t ACX = get_compile_time_arg_val(32);
    constexpr auto gat_args = TensorAccessorArgs<38>();
    constexpr auto gu_args = TensorAccessorArgs<gat_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto tl_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    static_assert(T == 8 || T == 16 || T == 32, "shared tail: T rows in {8, 16, 32}");
    static_assert(NA == 5, "shared tail: 5 apply cores (3 moments + b + 1)");
    const uint32_t acy[5] = {get_compile_time_arg_val(33), get_compile_time_arg_val(34), get_compile_time_arg_val(35),
                             get_compile_time_arg_val(36), get_compile_time_arg_val(37)};

    constexpr uint32_t HB = 2048;  // bf16 tile bytes (h, y)
    if constexpr (TL) {
        tl_buf = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_TL) + RISC * 64u);
        for (uint32_t i = 0; i < 16; ++i) {
            tl_buf[i] = 0;
        }
    }
    stamp();
    uint32_t tl_row = 0;

    if constexpr (ROLE == 0) {
        const uint32_t j = static_cast<uint32_t>(get_absolute_logical_y()) - AY0;
        tl_row = 2 * (ND + j) + RISC;
        constexpr uint32_t tb = 4096;  // fp32 tile
        if constexpr (RISC == 1) {
            // ---- apply reader ----
            constexpr uint32_t kb = get_tile_size(cb_k);
            const auto gats = TensorAccessor(gat_args, get_common_arg_val<uint32_t>(0), tb);
            const auto gus = TensorAccessor(gu_args, get_common_arg_val<uint32_t>(1), get_tile_size(cb_ag));
            const auto ds = TensorAccessor(d_args, get_common_arg_val<uint32_t>(2), kb);
            const auto es = TensorAccessor(e_args, get_common_arg_val<uint32_t>(3), kb);
            const auto bs = TensorAccessor(b_args, get_common_arg_val<uint32_t>(4), kb);

            cb_reserve_back(cb_k, 7);
            const uint32_t kdst = get_write_ptr(cb_k);
            for (uint32_t m = 0; m < 3; ++m) {
                noc_async_read(ds.get_noc_addr(m), kdst + kb * m, 64);
                noc_async_read(es.get_noc_addr(m), kdst + kb * (3 + m), 64);
            }
            noc_async_read(bs.get_noc_addr(0), kdst + kb * 6, 64);
            if (j < 3) {
                cb_reserve_back(cb_recv, TP);
                uint32_t dst = get_write_ptr(cb_recv);
                for (uint32_t k = 0; k < TP; ++k) {
                    const uint64_t src = gats.get_noc_addr(j * TP + k);
                    noc_async_read(src, dst, tb / 4);
                    noc_async_read(src + tb / 2, dst + tb / 2, tb / 4);
                    dst += tb;
                }
            }
            cb_reserve_back(cb_ag, 1);
            cb_reserve_back(cb_au, 1);
            noc_async_read_page(j, gus, get_write_ptr(cb_ag));
            noc_async_read_page(NH + j, gus, get_write_ptr(cb_au));
            noc_async_read_barrier();
            for (uint32_t jj = 0; jj < 7; ++jj) {  // broadcast element 0 down column 0 (apply_reader.cpp)
                volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kdst + kb * jj);
                const uint32_t v = t[0];
                for (uint32_t r = 1; r < 16; ++r) {
                    t[16 * r] = v;
                }
                for (uint32_t r = 0; r < 16; ++r) {
                    t[512 + 16 * r] = v;
                }
            }
            if (j < 3) {
                cb_push_back(cb_recv, TP);
            }
            cb_push_back(cb_k, 7);
            cb_push_back(cb_ag, 1);
            cb_push_back(cb_au, 1);
            stamp();
            // the 4 coefficient tiles (written by APPLY cores 0..3 into cb_coef slots 0..3)
            cb_reserve_back(cb_coef, 4);
            volatile tt_l1_ptr uint32_t* sc =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_coef_id));
            if constexpr ((EXP & 16) == 0) {
                wait_min(sc, 4);
            }
            *sc = 0;
            invalidate_l1_cache();
            stamp();
            cb_push_back(cb_coef, 4);
        } else {
            // ---- apply writer ----
            if (j < 4) {  // this core's coefficient tile -> slot j of every APPLY core's cb_coef (same address)
                const uint32_t cdst = get_write_ptr(cb_coef) + j * tb;
                const uint32_t sem_coef = get_semaphore(sem_coef_id);
                cb_wait_front(cb_co, 1);
                stamp();
                const uint32_t src = get_read_ptr(cb_co);
                for (uint32_t i = 0; i < NA; ++i) {
                    noc_async_write(src, get_noc_addr(ACX, acy[i], cdst), tb);
                }
                noc_async_write_barrier();
                for (uint32_t i = 0; i < NA; ++i) {
                    noc_semaphore_inc(get_noc_addr(ACX, acy[i], sem_coef), 1);
                }
                noc_async_atomic_barrier();
                cb_pop_front(cb_co, 1);
            }
            const uint32_t hbase = get_write_ptr(cb_h);  // same L1 address on every core (cb_h spans the program)
            const uint32_t sem_agg = get_semaphore(sem_agg_id);
            const uint32_t sem_h = get_semaphore(sem_h_id);
            cb_wait_front(cb_aout, 1);
            stamp();
            const uint32_t src = get_read_ptr(cb_aout);
            noc_async_write(src, get_noc_addr(A0X, A0Y, hbase + j * HB), HB);  // (core 0: a local NoC copy)
            noc_async_write_barrier();
            if (j != 0) {
                if constexpr ((EXP & 2) == 0) {
                    noc_semaphore_inc(get_noc_addr(A0X, A0Y, sem_agg), 1);
                    noc_async_atomic_barrier();
                }
            } else {
                volatile tt_l1_ptr uint32_t* agg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_agg);
                if constexpr ((EXP & 2) == 0) {
                    wait_min(agg, NA - 1);
                }
                stamp();
                *agg = 0;
                invalidate_l1_cache();
                volatile tt_l1_ptr uint32_t* sh = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_h);
                *sh = 1;
                asm volatile("" ::: "memory");
                noc_async_write_multicast(hbase, mcast_addr(DX0, DY0, DX1, DY1, hbase), NA * HB, ND, true);
                noc_semaphore_set_multicast(sem_h, mcast_addr(DX0, DY0, DX1, DY1, sem_h), ND, false);
                noc_async_write_barrier();
                *sh = 0;
            }
            stamp();
            cb_pop_front(cb_aout, 1);
        }
    } else {
        const uint32_t d = static_cast<uint32_t>(get_absolute_logical_y()) * DGX + get_absolute_logical_x();
        const uint32_t n0 = d * NPER;
        tl_row = 2 * d + RISC;
        if constexpr (RISC == 1) {
            // ---- down reader: W_down tiles (k-major), then h ----
            constexpr uint32_t wb = get_tile_size(cb_w);
            const auto ws = TensorAccessor(w_args, get_common_arg_val<uint32_t>(5), wb);
            cb_reserve_back(cb_w, NH * NPER);
            uint32_t dst = get_write_ptr(cb_w);
            for (uint32_t k = 0; k < NH && (EXP & 4) == 0; ++k) {
                for (uint32_t c = 0; c < NPER; ++c) {
                    noc_async_read_page(k * NT_OUT + n0 + c, ws, dst);
                    dst += wb;
                }
            }
            stamp();
            cb_reserve_back(cb_h, NH);
            volatile tt_l1_ptr uint32_t* sh = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_h_id));
            if constexpr ((EXP & 1) == 0) {
                wait_min(sh, 1);
            }
            stamp();
            *sh = 0;
            invalidate_l1_cache();
            cb_push_back(cb_h, NH);
            noc_async_read_barrier();
            stamp();
            cb_push_back(cb_w, NH * NPER);
        } else {
            // ---- down writer: rows 0..T-1 from cb_dout, rows T..31 = +0 from a zero scratch ----
            const auto ys = TensorAccessor(y_args, get_common_arg_val<uint32_t>(6), HB);
            const uint32_t z = get_write_ptr(cb_z);
            if constexpr (T < 32) {
                const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
                noc_async_read(zeros, z, MEM_ZEROS_SIZE);
                noc_async_read(zeros, z + MEM_ZEROS_SIZE, MEM_ZEROS_SIZE);
                noc_async_read_barrier();
            }
            cb_wait_front(cb_dout, NPER);
            stamp();
            const uint32_t base = get_read_ptr(cb_dout);
            for (uint32_t i = 0; i < NPER && (EXP & 8) == 0; ++i) {
                const uint32_t s = base + i * HB;
                const uint64_t o = ys.get_noc_addr(n0 + i);
                if constexpr (T == 32) {
                    noc_async_write(s, o, HB);
                } else if constexpr (T == 16) {
                    noc_async_write(s, o, 1024);          // faces 0, 1 (rows 0..15)
                    noc_async_write(z, o + 1024, 1024);   // faces 2, 3 (rows 16..31): 0
                } else {                                  // T == 8
                    noc_async_write(s, o, 256);           // face 0 rows 0..7
                    noc_async_write(z, o + 256, 256);     // face 0 rows 8..15: 0
                    noc_async_write(s + 512, o + 512, 256);  // face 1 rows 0..7
                    noc_async_write(z, o + 768, 256);     // face 1 rows 8..15: 0
                    noc_async_write(z, o + 1024, 1024);   // faces 2, 3: 0
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_dout, NPER);
        }
    }
    if constexpr (TL) {
        stamp();
        const auto tls = TensorAccessor(tl_args, get_common_arg_val<uint32_t>(7), 64);
        noc_async_write(get_write_ptr(CB_TL) + RISC * 64u, tls.get_noc_addr(tl_row, 0), 64);
        noc_async_write_barrier();
    }
}
