// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 decode token gather (D4): TILE -> ROW_MAJOR for one tile row, pure data movement (bitwise untilize, bf16).
//
//   in  [1, 1, L, W] bf16 TILE interleaved (L <= 32: one tile row; rows L .. 31 padding, never read)
//   out [1, 1, L, W] bf16 ROW_MAJOR interleaved (page = one row, W * 2 bytes)
// Input tile t: read it whole, then row r < L goes to bytes [64 t, 64 t + 64) of output row r: 32 B from face
// (r / 16) 2 (face row r % 16) and 32 B from the next face (two NoC writes). Both data-movement RISCs run this
// kernel: slot s = 2 (GX y + x) + RISC takes tiles [s PER, min(WT, (s + 1) PER)).
//
// CT: 0 L, 1 WT, 2 PER, 3 GX, 4 cb_scr, 5 RISC, 6 SCR, 7 ROWB (output page bytes), 8.. TensorAccessorArgs(in), (out)
// common RT: 0 in addr, 1 out addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t L = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
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
    static_assert(L >= 1 && L <= 32, "untilize rows: one tile row");

    const auto is = TensorAccessor(i_args, get_common_arg_val<uint32_t>(0), TB);
    const auto os = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1), ROWB);
    const uint32_t w = 2 * (static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x()) + RISC;
    const uint32_t t0 = w * PER;
    const uint32_t t1 = (t0 + PER < WT) ? t0 + PER : WT;
    if (t0 >= t1) {
        return;
    }
    const uint32_t base = ((get_write_ptr(cb_scr) + 63) & ~63u) + RISC * SCR;  // two tiles (double buffer)
    uint32_t slot = 0;
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t ib = base + slot * TB;
        if (t > t0 + 1) {
            noc_async_writes_flushed();  // the writes issued from this slot two tiles ago have left L1
        }
        noc_async_read(is.get_noc_addr(t, 0), ib, TB);
        noc_async_read_barrier();
        for (uint32_t r = 0; r < L; ++r) {
            const uint32_t src = ib + (r / 16) * 2 * FACE + (r % 16) * 32;
            const uint64_t dst = os.get_noc_addr(r, t * 64);
            noc_async_write(src, dst, 32);
            noc_async_write(src + FACE, dst + 32, 32);
        }
        slot ^= 1;
    }
    noc_async_write_barrier();
}
