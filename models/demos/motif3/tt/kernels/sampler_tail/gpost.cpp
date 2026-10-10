// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 device sampler, the Gumbel race's global winner (Phase F, F4; tt/kernels/sampler_tail.py), one data movement
// kernel per core, core q = lane j (32 cores). The op chain (tt/sampling.py _gumbel_gather) on the gathered per-chip
// triples G [96, 1024] (rows 0..31 score, 32..63 global id, 64..95 raw x - M; chip k in column 32 k):
//   gmax  = max(where(sel_z, Gs, -BIG))             = max(-BIG, max_k Gs[j, 32 k])   (the other columns are -BIG)
//   win   = eq(Gs, gmax) * sel_z                    = [Gs[j, 32 k] == gmax]
//   g_tok = min(where(win, Gi, BIG))                = min(BIG, min over the winners of Gi[j, 32 k])
//   g_dx  = max(where(win, Gd, -BIG))               = max(-BIG, max over the winners of Gd[j, 32 k])
// Only selections of fp32 values: the same results bit for bit for every input without NaN or -0 (the scores, ids
// and raw differences of the sampler have neither: x - M is +0 or negative, s = a - log E). The outputs are the
// [32, 1] fp32 tiles of the three reductions: column 0 of row j = the value, the rest of row j +0.
//
// CT: 0 GXQ (cores per row of the 8 x 4 rectangle), 1.. TensorAccessorArgs G, g_score, g_tok, g_dx
// common RT: 0..3 the buffer addresses (same order); 4 bits of -BIG; 5 bits of BIG

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
inline uint32_t tw(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
// a total order on the non-NaN fp32 bit patterns that agrees with < (no -0 in the inputs)
inline uint32_t key(uint32_t b) { return (b & 0x80000000u) ? ~b : (b | 0x80000000u); }
}  // namespace

void kernel_main() {
    constexpr uint32_t GXQ = get_compile_time_arg_val(0);
    constexpr auto g_args = TensorAccessorArgs<1>();
    constexpr auto s_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 4096;
    const uint32_t j = static_cast<uint32_t>(get_absolute_logical_y()) * GXQ + get_absolute_logical_x();
    const auto gs = TensorAccessor(g_args, get_common_arg_val<uint32_t>(0), TB);
    const auto os = TensorAccessor(s_args, get_common_arg_val<uint32_t>(1), TB);
    const auto ot = TensorAccessor(t_args, get_common_arg_val<uint32_t>(2), TB);
    const auto od = TensorAccessor(d_args, get_common_arg_val<uint32_t>(3), TB);
    const uint32_t neg_big = get_common_arg_val<uint32_t>(4);
    const uint32_t big = get_common_arg_val<uint32_t>(5);

    const uint32_t buf = get_write_ptr(0);  // 96 face rows (64 B) + 3 output rows (128 B)
    const uint32_t off = tw(j, 0) * 4;      // row j, columns 0..15 of the tile (64 B aligned)
    for (uint32_t t = 0; t < 96; ++t) {     // tile (t / 32, t % 32) = rows 32 (t / 32) .., chip t % 32
        noc_async_read(gs.get_noc_addr(t, off), buf + t * 64, 64);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
    uint32_t gmax = neg_big;
    for (uint32_t k = 0; k < 32; ++k) {
        const uint32_t s = v[k * 16];
        if (key(s) > key(gmax)) {
            gmax = s;
        }
    }
    uint32_t tok = big, dx = neg_big;
    for (uint32_t k = 0; k < 32; ++k) {
        if (v[k * 16] == gmax) {
            const uint32_t i = v[(32 + k) * 16], d = v[(64 + k) * 16];
            if (key(i) < key(tok)) {
                tok = i;
            }
            if (key(d) > key(dx)) {
                dx = d;
            }
        }
    }
    const uint32_t ob = buf + 96 * 64;
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ob);
    const uint32_t vals[3] = {gmax, tok, dx};
    for (uint32_t q = 0; q < 3; ++q) {
        for (uint32_t c = 0; c < 32; ++c) {
            o[q * 32 + c] = (c == 0) ? vals[q] : 0u;
        }
    }
    const uint32_t off1 = tw(j, 16) * 4;
    noc_async_write(ob, os.get_noc_addr(0, off), 64);
    noc_async_write(ob + 64, os.get_noc_addr(0, off1), 64);
    noc_async_write(ob + 128, ot.get_noc_addr(0, off), 64);
    noc_async_write(ob + 192, ot.get_noc_addr(0, off1), 64);
    noc_async_write(ob + 256, od.get_noc_addr(0, off), 64);
    noc_async_write(ob + 320, od.get_noc_addr(0, off1), 64);
    noc_async_write_barrier();
}
