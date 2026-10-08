// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 decode MoE combine (D4): MotifCCL.partition of a TILE row slice as pure data movement (bitwise the release's
// untilize + mesh_partition + tilize, zero padding rows).
//
//   in  [1, 1, N L, W] bf16 TILE interleaved (e.g. the AR(dp) output: row L b + j = DP row b's row j)
//   idx uint32 ROW_MAJOR, one 64 B page per chip: word 0 = this chip's index b on the partition axis
//   out [1, 1, L, W]   bf16 TILE interleaved: out[j] = in[L b + j], rows L .. 31 zero; L = 8 or 16
// Rows L b .. L b + L - 1 are 8- (16-) aligned, so they are one contiguous band of L face rows in each face column of
// the source tile: two NoC reads of L x 32 B per output tile, landing in place (rows 0 .. L - 1 of faces 0 / 1).
// Both data-movement RISCs: slot s = 2 (GX y + x) + RISC takes output tiles [s PER, min(WT, (s + 1) PER)).
//
// CT: 0 L, 1 WT, 2 PER, 3 GX, 4 cb_scr, 5 RISC, 6 SCR, 7.. TensorAccessorArgs(in), (out), (idx)
// common RT: 0 in addr, 1 out addr, 2 idx addr

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
    constexpr auto i_args = TensorAccessorArgs<7>();
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto x_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 2048;
    constexpr uint32_t FACE = 512;
    static_assert(L == 8 || L == 16, "pick_rows: L = 8 or 16");

    const auto is = TensorAccessor(i_args, get_common_arg_val<uint32_t>(0), TB);
    const auto os = TensorAccessor(o_args, get_common_arg_val<uint32_t>(1), TB);
    const auto xs = TensorAccessor(x_args, get_common_arg_val<uint32_t>(2), 64);
    const uint32_t w = 2 * (static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x()) + RISC;
    const uint32_t t0 = w * PER;
    const uint32_t t1 = (t0 + PER < WT) ? t0 + PER : WT;
    if (t0 >= t1) {
        return;
    }
    const uint32_t base = ((get_write_ptr(cb_scr) + 63) & ~63u) + RISC * SCR;  // idx (64 B), then two output tiles
    const uint32_t outs = base + 64;
    noc_async_read(xs.get_noc_addr(0, 0), base, 64);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < 2 * TB; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, outs + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    asm volatile("fence" ::: "memory");
    const uint32_t b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base)[0];
    const uint32_t srow = L * b;
    const uint32_t tr = srow / 32;
    const uint32_t srr = srow % 32;
    const uint32_t band = (srr / 16) * 2 * FACE + (srr % 16) * 32;
    uint32_t slot = 0;
    for (uint32_t t = t0; t < t1; ++t) {
        const uint32_t ob = outs + slot * TB;
        if (t > t0 + 1) {
            noc_async_writes_flushed();  // the write issued from this slot two tiles ago has left L1
        }
        const uint64_t src = is.get_noc_addr(tr * WT + t, band);
        noc_async_read(src, ob, L * 32);
        noc_async_read(src + FACE, ob + FACE, L * 32);
        noc_async_read_barrier();
        noc_async_write(ob, os.get_noc_addr(t, 0), TB);
        slot ^= 1;
    }
    noc_async_write_barrier();
}
