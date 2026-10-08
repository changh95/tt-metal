// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 mHC decode stream mixing from the packed coefficient tile (D3, docs/OPTIMIZATION_PLAN.md §3.3 C2): reader.
//
// The weighted reduce of attn_res_weighted_reduce_nc (pre: x_red = sum_c X_c w_pre[c]) and of tt/mhc.py post_mix
// (post: X'_r = sum_c X_c w[r, c] + out w[r, 4]) take their weights as [.., T, 1] fp32 TILE tensors: the scalar of
// token t in column 0 of row t (BroadcastType::COL reads column 0 of faces 0 and 2). The release path writes 4 (pre)
// / 20 (post) such tiles to L1 and every one of the ~120 workers reads the whole set over the NOC (16 / 80 KB per
// worker from 4 / 20 banks): the post mix spent 20 of its 23.4 us there (logs/opt/phaseC/D3/probe/probe1.json).
//
// Here every worker reads ONE packed tile P (fp32 32x32, row k = weight k over the 32 tokens of the tile row, the
// transposed coefficient layout; see tt/kernels/mhc_decode.py) from one of NCOPY identical copies (page
// row * NCOPY + core % NCOPY, so ~120 / NCOPY readers per bank) and expands it into the weight CB itself: the column-0
// words of faces 0 and 2 (word 16 t for t < 16, 512 + 16 (t - 16) for t >= 16) of every weight tile. The other words
// of the weight tiles are never read by the FPU (the release layout kernel leaves them stale too). Values are copied
// bit for bit, so the weights, and hence the outputs, are those of the release path.
//
//   weight (r, c) -> P row k:  pre (HAS_OUT = 0, NUM_R = 1): k = c (h_pre)
//                              post (HAS_OUT = 1, NUM_R = 4): k = 8 + 4 r + c (H[r][c]) for c < 4, k = 4 + r (h_post)
//
// The candidate reads (X pages i + c * inner, then out page i) are the release post_mix reader's.
//
// CT args: 0 inner (= Ht * Wt), 1 Wt, 2 NUM_X (4), 3 HAS_OUT, 4 NUM_R, 5 NCOPY,
//          6.. TensorAccessorArgs(X), (out) [HAS_OUT only], (P)
// Common RT args: 0 x_addr, 1 out_addr (ignored for pre), 2 p_addr, 3 total positions, 4 num_cores, 5 grid_y
// Core i = x * grid_y + y owns positions [start, start + n), the split of post_mix / sinkhorn_motif.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

#include "wr_expand.h"

void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t total = get_common_arg_val<uint32_t>(3);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(4);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(5);
    constexpr uint32_t inner = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_x = get_compile_time_arg_val(2);
    constexpr uint32_t has_out = get_compile_time_arg_val(3);
    constexpr uint32_t num_r = get_compile_time_arg_val(4);
    constexpr uint32_t ncopy = get_compile_time_arg_val(5);
    constexpr auto a_x = TensorAccessorArgs<6>();
    constexpr auto a_o = TensorAccessorArgs<a_x.next_compile_time_args_offset()>();
    constexpr auto a_p = TensorAccessorArgs<has_out ? a_o.next_compile_time_args_offset()
                                                    : a_x.next_compile_time_args_offset()>();
    constexpr uint32_t num_c = num_x + has_out;
    constexpr uint32_t cb_in0 = 0, cb_in1 = 1, cb_p = 2, cb_flag = 3;
    constexpr uint32_t in_bytes = get_tile_size(cb_in0);
    constexpr uint32_t p_bytes = get_tile_size(cb_p);
    const auto s_x = TensorAccessor(a_x, x_addr, in_bytes);
    const auto s_p = TensorAccessor(a_p, p_addr, p_bytes);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t q = total / num_cores, rem = total % num_cores;
    const uint32_t n = q + (core_i < rem ? 1 : 0);
    const uint32_t start = core_i * q + (core_i < rem ? core_i : rem);
    CircularBuffer c0(cb_in0), c1(cb_in1), cp(cb_p);
    cp.reserve_back(1);
    const uint32_t p_l1 = cp.get_write_ptr();
    for (uint32_t i = start; i < start + n; ++i) {
        const bool new_row = (i == start || i % Wt == 0);
        // candidates of position i (issued first: they overlap the packed-tile read and the expansion)
        c0.reserve_back(num_c);
        uint32_t l1 = c0.get_write_ptr();
        for (uint32_t c = 0; c < num_x; ++c) {
            noc_async_read(s_x.get_noc_addr(i + c * inner), l1, in_bytes);
            l1 += in_bytes;
        }
        if constexpr (has_out) {
            const auto s_o = TensorAccessor(a_o, o_addr, in_bytes);
            noc_async_read(s_o.get_noc_addr(i), l1, in_bytes);
        }
        if (new_row) {
            const uint32_t row = i / Wt;
            {
                const uint64_t pa = s_p.get_noc_addr(row * ncopy + core_i % ncopy);
#if MHC_P_READ_NONE
                (void)pa;
#elif MHC_P_READ_MIN
                // T <= 16 (tokens 16..31 have zero weights): only token columns 0..15 of the rows this mix uses
                if constexpr (has_out) {
                    noc_async_read(pa + 256, p_l1 + 256, 768);     // face 0 rows 4..15 (h_post, H rows 0..7)
                    noc_async_read(pa + 2048, p_l1 + 2048, 512);   // face 2 rows 0..7 (H rows 8..15)
                } else {
                    noc_async_read(pa, p_l1, 256);                 // face 0 rows 0..3 (h_pre)
                }
#else
                noc_async_read(pa, p_l1, p_bytes);
#endif
            }
            noc_async_read_barrier();
            c1.reserve_back(num_r * num_c);
#if MHC_EXPAND_SPLIT
            cp.push_back(1);  // the packed tile is in L1: the writer expands the second half of the weight set
#endif
            {
                uint32_t* w = reinterpret_cast<uint32_t*>(c1.get_write_ptr());
                const uint32_t* p = reinterpret_cast<const uint32_t*>(p_l1);
                constexpr uint32_t n_w = num_r * num_c;
                constexpr uint32_t w_end = MHC_EXPAND_SPLIT ? (n_w + 1) / 2 : n_w;
                for (uint32_t j = 0; j < (MHC_EXPAND_SKIP ? 0 : w_end); ++j) {
                    expand_weight<has_out, num_x, num_c>(w, p, j);
                }
                invalidate_l1_cache();  // fence
            }
#if MHC_EXPAND_SPLIT
            {
                CircularBuffer cf(cb_flag);
                cf.wait_front(1);
                cf.pop_front(1);
            }
#endif
            c1.push_back(num_r * num_c);
        } else {
            noc_async_read_barrier();
        }
        c0.push_back(num_c);
    }
}
