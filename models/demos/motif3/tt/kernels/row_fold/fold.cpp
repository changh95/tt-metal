// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 decode MoE combine (D4, docs/OPTIMIZATION_PLAN.md §3.3 C4): row fold, one dataflow kernel (RISCV_0).
//
//   P [1, 1, N L, H] bf16 TILE (the routed partial of the N L gathered rows, row L b + j = DP row b's row j)
//   -> Q [1, N, L N, F] bf16 TILE, F = H / N:  Q[b, N j + q, c] = P[L b + j, q F + c]
// which is exactly the logical reshape [1, 1, N L, H] -> [1, N, L N, F] (pure data movement, bit for bit). A
// reduce-scatter of Q on dim 1 then leaves DP row b's L rows, summed over the N chips, folded into [1, 1, L N, F].
//
// Output tile o = (b RT + rt) FT + ct (RT = L N / 32 row tiles, FT = F / 32 column tiles). Its rows N k + q (k < KB,
// KB = 32 / N = 8) are P rows L b + KB rt + k, column tile q FT + ct of P: for each q one 8-row band of one P tile
// (rows 8-aligned, so inside one 16-row face band: two contiguous 256 B reads, one per face column). The bands land
// in scratch, are scattered row by row (32 B per face row) into the output tile, which is written as one page.
// Both data-movement RISCs of a core run this kernel (CT RISC = 0 / 1; the row copies are the cost, ~1.7 us per tile):
// slot s = 2 (GX y + x) + RISC takes output tiles [s PER, min(NO, (s + 1) PER)), with its own scratch half.
//
// CT: 0 N, 1 L, 2 HT (= H / 32), 3 FT, 4 NO (output tiles), 5 PER, 6 GX, 7 cb_scr, 8 RISC, 9 SCR (scratch bytes per
//     RISC), 10.. TensorAccessorArgs(P), (Q)
// common RT: 0 P addr, 1 Q addr

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
    constexpr uint32_t HT = get_compile_time_arg_val(2);
    constexpr uint32_t FT = get_compile_time_arg_val(3);
    constexpr uint32_t NO = get_compile_time_arg_val(4);
    constexpr uint32_t PER = get_compile_time_arg_val(5);
    constexpr uint32_t GX = get_compile_time_arg_val(6);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(7);
    constexpr uint32_t RISC = get_compile_time_arg_val(8);
    constexpr uint32_t SCR = get_compile_time_arg_val(9);
    constexpr auto p_args = TensorAccessorArgs<10>();
    constexpr auto q_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 2048;        // bf16 32 x 32 tile
    constexpr uint32_t FACE = 512;       // bf16 16 x 16 face
    constexpr uint32_t ROWB = 32;        // one face row (16 bf16)
    constexpr uint32_t KB = 32 / N;      // rows of one q band
    constexpr uint32_t BAND = KB * ROWB;  // bytes of one band half (one face column)
    constexpr uint32_t RT = L * N / 32;
    static_assert(N * KB == 32 && KB == 8, "row fold: N must be 4 (8-row bands)");
    static_assert(L % 8 == 0 && (L * N) % 32 == 0 && HT == N * FT, "row fold shape");

    const auto ps = TensorAccessor(p_args, get_common_arg_val<uint32_t>(0), TB);
    const auto qs = TensorAccessor(q_args, get_common_arg_val<uint32_t>(1), TB);

    const uint32_t w = 2 * (static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x()) + RISC;
    const uint32_t o0 = w * PER;
    const uint32_t o1 = (o0 + PER < NO) ? o0 + PER : NO;
    if (o0 >= o1) {
        return;
    }
    const uint32_t base = ((get_write_ptr(cb_scr) + 63) & ~63u) + RISC * SCR;
    const uint32_t bands = base;                 // N x 2 bands of BAND bytes
    const uint32_t outs = base + N * 2 * BAND;   // two output tiles (double buffer)

    uint32_t slot = 0;
    for (uint32_t o = o0; o < o1; ++o) {
        const uint32_t b = o / (RT * FT);
        const uint32_t rem = o % (RT * FT);
        const uint32_t rt = rem / FT;
        const uint32_t ct = rem % FT;
        const uint32_t srow = L * b + KB * rt;     // first P row of every band (8-aligned)
        const uint32_t stile_row = srow / 32;
        const uint32_t srr = srow % 32;
        const uint32_t band_off = (srr / 16) * 2 * FACE + (srr % 16) * ROWB;
        for (uint32_t q = 0; q < N; ++q) {
            const uint32_t page = stile_row * HT + q * FT + ct;
            noc_async_read(ps.get_noc_addr(page, band_off), bands + (q * 2) * BAND, BAND);
            noc_async_read(ps.get_noc_addr(page, band_off + FACE), bands + (q * 2 + 1) * BAND, BAND);
        }
        noc_async_read_barrier();
        fence_l1();
        const uint32_t ob = outs + slot * TB;
        if (o > o0 + 1) {
            noc_async_writes_flushed();  // the write issued from this slot two tiles ago has left L1
        }
        for (uint32_t rr = 0; rr < 32; ++rr) {
            const uint32_t q = rr % N;
            const uint32_t k = rr / N;
            const uint32_t drow = ob + (rr / 16) * 2 * FACE + (rr % 16) * ROWB;
            copy32(drow, bands + (q * 2) * BAND + k * ROWB);
            copy32(drow + FACE, bands + (q * 2 + 1) * BAND + k * ROWB);
        }
        fence_l1();
        noc_async_write(ob, qs.get_noc_addr(o, 0), TB);
        slot ^= 1;
    }
    noc_async_write_barrier();
}
