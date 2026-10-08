// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 decode MoE combine (D4): R + fold(S), reader (RISCV_1).
//
//   R [1, 1, N L, F] bf16 TILE (the folded RS(dp) output: this DP row's L rows, F = H / N)
//   S [1, 1, L, H]   bf16 TILE (the TP add_partial, rows L .. 31 padding, never read)
//   -> out [1, 1, N L, F] = R + fold(S),  fold(S)[N j + q, c] = S[j, q F + c]   (the logical reshape, as fold.cpp)
// Output tile o = rt FT + ct: CB_R gets R's page o; CB_S gets the folded S tile, built from N 8-row bands of S (rows
// KB rt .. KB rt + 7 of column tile q FT + ct, two 256 B reads each) scattered row by row (32 B per face row).
// Worker w = GX y + x takes output tiles [w PER, min(NO, (w + 1) PER)).
//
// CT: 0 N, 1 L, 2 HT (= H / 32), 3 FT, 4 NO, 5 PER, 6 GX, 7 cb_r, 8 cb_s, 9 cb_scr, 10.. TensorAccessorArgs(R), (S)
// common RT: 0 R addr, 1 S addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
FORCE_INLINE void copy32(uint32_t dst, uint32_t src) {
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
    uint32_t* d = reinterpret_cast<uint32_t*>(dst);
    const uint32_t a0 = s[0], a1 = s[1], a2 = s[2], a3 = s[3], a4 = s[4], a5 = s[5], a6 = s[6], a7 = s[7];
    d[0] = a0;
    d[1] = a1;
    d[2] = a2;
    d[3] = a3;
    d[4] = a4;
    d[5] = a5;
    d[6] = a6;
    d[7] = a7;
}
FORCE_INLINE void fence_l1() { asm volatile("fence" ::: "memory"); }
}  // namespace

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t L = get_compile_time_arg_val(1);
    constexpr uint32_t HT = get_compile_time_arg_val(2);
    constexpr uint32_t FT = get_compile_time_arg_val(3);
    constexpr uint32_t NO = get_compile_time_arg_val(4);
    constexpr uint32_t PER = get_compile_time_arg_val(5);
    constexpr uint32_t GX = get_compile_time_arg_val(6);
    constexpr uint32_t cb_r = get_compile_time_arg_val(7);
    constexpr uint32_t cb_s = get_compile_time_arg_val(8);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(9);
    constexpr auto r_args = TensorAccessorArgs<10>();
    constexpr auto s_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 2048;
    constexpr uint32_t FACE = 512;
    constexpr uint32_t ROWB = 32;
    constexpr uint32_t KB = 32 / N;
    constexpr uint32_t BAND = KB * ROWB;
    static_assert(N * KB == 32 && KB == 8 && L % 8 == 0 && L <= 32 && HT == N * FT, "fold_add shape");

    const auto rs = TensorAccessor(r_args, get_common_arg_val<uint32_t>(0), TB);
    const auto ss = TensorAccessor(s_args, get_common_arg_val<uint32_t>(1), TB);

    const uint32_t w = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t o0 = w * PER;
    const uint32_t o1 = (o0 + PER < NO) ? o0 + PER : NO;
    const uint32_t bands = (get_write_ptr(cb_scr) + 63) & ~63u;  // N x 2 bands of BAND bytes
    for (uint32_t o = o0; o < o1; ++o) {
        const uint32_t rt = o / FT;
        const uint32_t ct = o % FT;
        const uint32_t j0 = KB * rt;  // S rows of every band (8-aligned, < L)
        const uint32_t band_off = ((j0 % 32) / 16) * 2 * FACE + (j0 % 16) * ROWB;
        cb_reserve_back(cb_r, 1);
        noc_async_read(rs.get_noc_addr(o, 0), get_write_ptr(cb_r), TB);
        for (uint32_t q = 0; q < N; ++q) {
            const uint32_t page = (j0 / 32) * HT + q * FT + ct;
            noc_async_read(ss.get_noc_addr(page, band_off), bands + (q * 2) * BAND, BAND);
            noc_async_read(ss.get_noc_addr(page, band_off + FACE), bands + (q * 2 + 1) * BAND, BAND);
        }
        cb_reserve_back(cb_s, 1);
        noc_async_read_barrier();
        fence_l1();
        const uint32_t ob = get_write_ptr(cb_s);
        for (uint32_t rr = 0; rr < 32; ++rr) {
            const uint32_t q = rr % N;
            const uint32_t k = rr / N;
            const uint32_t drow = ob + (rr / 16) * 2 * FACE + (rr % 16) * ROWB;
            copy32(drow, bands + (q * 2) * BAND + k * ROWB);
            copy32(drow + FACE, bands + (q * 2 + 1) * BAND + k * ROWB);
        }
        fence_l1();
        cb_push_back(cb_r, 1);
        cb_push_back(cb_s, 1);
    }
}
