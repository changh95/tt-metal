// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 decode MoE combine (D4): row unfold, one dataflow kernel on both data-movement RISCs. The inverse of fold.cpp:
//
//   R [1, 1, L N, F] bf16 TILE  ->  out [1, 1, L, N F] bf16 TILE:  out[j, q F + c] = R[N j + q, c]
// (the logical reshape [1, 1, L N, F] -> [1, 1, L, N F]; L <= 32, rows L .. 31 of the output tile row are zeros).
//
// Output tile o = q FT + ct: read the RT = L N / 32 tiles of column ct of R, then row j < L of the output tile is row
// N j + q of that column (32 B per face row); rows >= L stay zero. Slot s = 2 (GX y + x) + RISC takes output tiles
// [s PER, min(NO, (s + 1) PER)), with its own scratch half.
//
// CT: 0 N, 1 L, 2 FT, 3 PER, 4 GX, 5 cb_scr, 6 RISC, 7 SCR (scratch bytes per RISC), 8.. TensorAccessorArgs(R), (out)
// common RT: 0 R addr, 1 out addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
FORCE_INLINE void copy32(uint32_t dst, uint32_t src) {
#ifdef MOTIF_ROWFOLD_NOCOPY
    return;
#endif
    // 8 loads, then 8 stores (no volatile: the loads pipeline); the caller fences before the NoC reads the data.
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
    constexpr uint32_t FT = get_compile_time_arg_val(2);
    constexpr uint32_t PER = get_compile_time_arg_val(3);
    constexpr uint32_t GX = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(5);
    constexpr uint32_t RISC = get_compile_time_arg_val(6);
    constexpr uint32_t SCR = get_compile_time_arg_val(7);
    constexpr auto r_args = TensorAccessorArgs<8>();
    constexpr auto o_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 2048;
    constexpr uint32_t FACE = 512;
    constexpr uint32_t ROWB = 32;
    constexpr uint32_t RT = L * N / 32;
    constexpr uint32_t NO = N * FT;
    static_assert(L >= 1 && L <= 32 && (L * N) % 32 == 0, "row unfold shape");

    const auto rs = TensorAccessor(r_args, get_common_arg_val<uint32_t>(0), TB);
    const auto os = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1), TB);

    const uint32_t w = 2 * (static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x()) + RISC;
    const uint32_t o0 = w * PER;
    const uint32_t o1 = (o0 + PER < NO) ? o0 + PER : NO;
    if (o0 >= o1) {
        return;
    }
    const uint32_t base = ((get_write_ptr(cb_scr) + 63) & ~63u) + RISC * SCR;
    const uint32_t src = base;              // RT tiles of one R column
    const uint32_t outs = base + RT * TB;   // two output tiles (double buffer)

    // zero both output tiles once: rows >= L are never written below
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < 2 * TB; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, outs + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    fence_l1();

    uint32_t slot = 0;
    for (uint32_t o = o0; o < o1; ++o) {
        const uint32_t q = o / FT;
        const uint32_t ct = o % FT;
        for (uint32_t tr = 0; tr < RT; ++tr) {
            noc_async_read(rs.get_noc_addr(tr * FT + ct, 0), src + tr * TB, TB);
        }
        noc_async_read_barrier();
        fence_l1();
        const uint32_t ob = outs + slot * TB;
        if (o > o0 + 1) {
            noc_async_writes_flushed();  // the write issued from this slot two tiles ago has left L1
        }
        for (uint32_t j = 0; j < L; ++j) {
            const uint32_t s = N * j + q;
            const uint32_t sr = s % 32;
            const uint32_t srow = src + (s / 32) * TB + (sr / 16) * 2 * FACE + (sr % 16) * ROWB;
            const uint32_t drow = ob + (j / 16) * 2 * FACE + (j % 16) * ROWB;
            copy32(drow, srow);
            copy32(drow + FACE, srow + FACE);
        }
        fence_l1();
        noc_async_write(ob, os.get_noc_addr(o, 0), TB);
        slot ^= 1;
    }
    noc_async_write_barrier();
}
