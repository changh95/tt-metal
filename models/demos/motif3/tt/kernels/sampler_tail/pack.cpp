// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 device sampler, the output rows (Phase F, F4; tt/kernels/sampler_tail.py), one data movement kernel on one
// core. The op chain (tt/sampling.py _stage_outputs) on the eight [32, 1] fp32 result tiles (token, flag, logprob,
// n_kept, u, mass_kept, window mass, kth):
//   info   = untilize_with_unpadding(transpose(concat(rows, dim 3)))   -> [8, 32] fp32 ROW_MAJOR: info[i][l] = row i, lane l
//   tokens = untilize_with_unpadding(typecast(transpose(tok), uint32)) -> [1, 32] uint32 ROW_MAJOR
// Data movement only; the fp32 -> uint32 typecast of a token id (an exact non-negative integer below 2^24) is its
// integer value.
//
// CT: 0.. TensorAccessorArgs of the 8 row tiles, info, tokens
// common RT: 0..9 the buffer addresses (same order)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
inline uint32_t tw(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
inline uint32_t f2u(uint32_t b) {  // exact non-negative integer-valued fp32 -> uint32
    if ((b & 0x7FFFFFFFu) == 0 || (b & 0x80000000u)) {
        return 0;
    }
    const int32_t e = static_cast<int32_t>((b >> 23) & 0xFF) - 127;
    if (e < 0) {
        return 0;
    }
    const uint32_t m = (b & 0x7FFFFFu) | 0x800000u;
    return e >= 23 ? (m << (e - 23)) : (m >> (23 - e));
}
}  // namespace

void kernel_main() {
    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();
    constexpr auto ai = TensorAccessorArgs<a7.next_compile_time_args_offset()>();
    constexpr auto at = TensorAccessorArgs<ai.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 4096, RB = 128;
    const auto r0 = TensorAccessor(a0, get_common_arg_val<uint32_t>(0), TB);
    const auto r1 = TensorAccessor(a1, get_common_arg_val<uint32_t>(1), TB);
    const auto r2 = TensorAccessor(a2, get_common_arg_val<uint32_t>(2), TB);
    const auto r3 = TensorAccessor(a3, get_common_arg_val<uint32_t>(3), TB);
    const auto r4 = TensorAccessor(a4, get_common_arg_val<uint32_t>(4), TB);
    const auto r5 = TensorAccessor(a5, get_common_arg_val<uint32_t>(5), TB);
    const auto r6 = TensorAccessor(a6, get_common_arg_val<uint32_t>(6), TB);
    const auto r7 = TensorAccessor(a7, get_common_arg_val<uint32_t>(7), TB);
    const auto info = TensorAccessor(ai, get_common_arg_val<uint32_t>(8), RB);
    const auto toks = TensorAccessor(at, get_common_arg_val<uint32_t>(9), RB);

    const uint32_t buf = get_write_ptr(0);  // 8 x 32 face rows (64 B each) + 9 output rows (128 B)
    // column 0 of row l of tile i = word 0 of the 64 B face row (row l, columns 0..15)
    for (uint32_t l = 0; l < 32; ++l) {
        const uint32_t off = tw(l, 0) * 4;
        const uint32_t d = buf + l * 64;
        noc_async_read(r0.get_noc_addr(0, off), d + 0 * 32 * 64, 64);
        noc_async_read(r1.get_noc_addr(0, off), d + 1 * 32 * 64, 64);
        noc_async_read(r2.get_noc_addr(0, off), d + 2 * 32 * 64, 64);
        noc_async_read(r3.get_noc_addr(0, off), d + 3 * 32 * 64, 64);
        noc_async_read(r4.get_noc_addr(0, off), d + 4 * 32 * 64, 64);
        noc_async_read(r5.get_noc_addr(0, off), d + 5 * 32 * 64, 64);
        noc_async_read(r6.get_noc_addr(0, off), d + 6 * 32 * 64, 64);
        noc_async_read(r7.get_noc_addr(0, off), d + 7 * 32 * 64, 64);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    const uint32_t ob = buf + 8 * 32 * 64;
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ob);
    for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t l = 0; l < 32; ++l) {
            o[i * 32 + l] = v[(i * 32 + l) * 16];
        }
    }
    for (uint32_t l = 0; l < 32; ++l) {
        o[8 * 32 + l] = f2u(v[l * 16]);
    }
    for (uint32_t i = 0; i < 8; ++i) {
        noc_async_write(ob + i * RB, info.get_noc_addr(i), RB);
    }
    noc_async_write(ob + 8 * RB, toks.get_noc_addr(0), RB);
    noc_async_write_barrier();
}
