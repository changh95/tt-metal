// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 decode token gather (D4): ROW_MAJOR -> TILE as pure data movement (bitwise ttnn.to_layout / tilize for bf16).
//
//   in  [1, 1, M, W] bf16 ROW_MAJOR interleaved (page = one row, W * 2 bytes), M % 32 == 0, W % 32 == 0
//   out [1, 1, M, W] bf16 TILE interleaved
// Output tile o = tr WT + tc: row r of the tile is row 32 tr + r of the input, bytes [64 tc, 64 tc + 64): its first
// 32 B (columns 0..15) go to face (r / 16) 2 at face row r % 16, the next 32 B to face (r / 16) 2 + 1. The NoC reads
// land in place (64 reads of 32 B per tile), then the tile is written as one page. Both data-movement RISCs run this
// kernel: slot s = 2 (GX y + x) + RISC takes tiles [s PER, min(NO, (s + 1) PER)).
//
// CT: 0 WT, 1 NO, 2 PER, 3 GX, 4 cb_scr, 5 RISC, 6 SCR, 7 ROWB (input page bytes), 8.. TensorAccessorArgs(in), (out)
// common RT: 0 in addr, 1 out addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t WT = get_compile_time_arg_val(0);
    constexpr uint32_t NO = get_compile_time_arg_val(1);
    constexpr uint32_t PER = get_compile_time_arg_val(2);
    constexpr uint32_t GX = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(4);
    constexpr uint32_t RISC = get_compile_time_arg_val(5);
    constexpr uint32_t SCR = get_compile_time_arg_val(6);
    constexpr uint32_t ROWB = get_compile_time_arg_val(7);
    constexpr auto i_args = TensorAccessorArgs<8>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 2048;
    constexpr uint32_t FACE = 512;

    const auto is = TensorAccessor(i_args, get_common_arg_val<uint32_t>(0), ROWB);
    const auto os = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1), TB);
    const uint32_t w = 2 * (static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x()) + RISC;
    const uint32_t o0 = w * PER;
    const uint32_t o1 = (o0 + PER < NO) ? o0 + PER : NO;
    if (o0 >= o1) {
        return;
    }
    const uint32_t base = ((get_write_ptr(cb_scr) + 63) & ~63u) + RISC * SCR;  // two tiles (double buffer)
    uint32_t slot = 0;
    for (uint32_t o = o0; o < o1; ++o) {
        const uint32_t tr = o / WT;
        const uint32_t tc = o % WT;
        const uint32_t ob = base + slot * TB;
        if (o > o0 + 1) {
            noc_async_writes_flushed();  // the write issued from this slot two tiles ago has left L1
        }
        for (uint32_t r = 0; r < 32; ++r) {
            const uint64_t src = is.get_noc_addr(tr * 32 + r, tc * 64);
            const uint32_t dst = ob + (r / 16) * 2 * FACE + (r % 16) * 32;
            noc_async_read(src, dst, 32);
            noc_async_read(src + 32, dst + FACE, 32);
        }
        noc_async_read_barrier();
        noc_async_write(ob, os.get_noc_addr(o, 0), TB);
        slot ^= 1;
    }
    noc_async_write_barrier();
}
