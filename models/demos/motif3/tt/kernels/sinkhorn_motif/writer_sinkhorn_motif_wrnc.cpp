// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 exact mHC coefficients (Option B, WAVE_A_REVIEW MHC-6): writer for the attn_res_weighted_reduce_nc layout.
//
// Instead of h_pre [T,4] / h_post [T,4] / H [T,16] it writes the two weight tensors that
// ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc consumes directly (MHC-4; design §2.3.3 step 6:
// no transpose / reshape / concat glue in tt/mhc.py):
//   w_pre  [1, 4, T, 1]: weight[0][c][t][0] = h_pre[t][c]                     -> x_red = sum_c w_pre[c] X_c
//   w_post [4, 5, T, 1]: weight[r][c][t][0] = H[t][r][c] (c < 4), h_post[t][r] (c = 4)
//                                                                              -> X'_r = sum_c w_post[r][c] [X | out]_c
// fp32 TILE; the scalar sits in column 0 of each tile row (BroadcastType::COL), the other columns are zero. Page id
// of tile (r, c, h) = (r * C + c) * Ht + h.
//
// The compute kernel is the same as for the default layout (token-row tiles in CB_PRE / CB_POST / CB_COMB); this
// writer copies column q of a token-row tile into column 0 of a zeroed scratch tile (32 words per tile, 768 per
// token tile) and writes the 24 scratch tiles. The scratch is zeroed once by the NOC, while compute runs.
//
// CT args: [cb_pre, cb_post, cb_comb, cb_scratch, TensorAccessorArgs(w_pre)..., TensorAccessorArgs(w_post)...]
// Common RT args: [w_pre_addr, w_post_addr, num_tiles_total, num_cores, grid_y] (work split as in the reader)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t TILE_WORDS = 1024;
constexpr uint32_t N_PRE = 4;       // w_pre tiles per token tile (c = 0..3)
constexpr uint32_t N_POST = 20;     // w_post tiles per token tile (r = 0..3, c = 0..4)
constexpr uint32_t N_SCRATCH = N_PRE + N_POST;

// dst column 0 <- src column q (q < 16: faces 0 / 2), 32 rows. Element (row, col) of an fp32 tile is at word
// ((row / 16) * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16. Plain (non-volatile) pointers: the source is stable
// after wait_front and the scratch is only read by the NOC after the copies, so the compiler may batch the loads
// (16 in flight) instead of serializing load-store pairs.
inline void copy_col_to_col0(const uint32_t* __restrict src, uint32_t* __restrict dst, uint32_t q) {
    uint32_t v[16];
#pragma GCC unroll 16
    for (uint32_t i = 0; i < 16; ++i) {
        v[i] = src[i * 16 + q];
    }
#pragma GCC unroll 16
    for (uint32_t i = 0; i < 16; ++i) {
        dst[i * 16] = v[i];
    }
#pragma GCC unroll 16
    for (uint32_t i = 0; i < 16; ++i) {
        v[i] = src[512 + i * 16 + q];
    }
#pragma GCC unroll 16
    for (uint32_t i = 0; i < 16; ++i) {
        dst[512 + i * 16] = v[i];
    }
}
}  // namespace

void kernel_main() {
    const uint32_t w_pre_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_post_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t total_tiles = get_common_arg_val<uint32_t>(2);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(4);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t q = total_tiles / num_cores, r = total_tiles % num_cores;
    const uint32_t num_token_tiles = q + (core_i < r ? 1 : 0);
    const uint32_t start_tile = core_i * q + (core_i < r ? core_i : r);

    constexpr uint32_t cb_pre = get_compile_time_arg_val(0);
    constexpr uint32_t cb_post = get_compile_time_arg_val(1);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(3);
    constexpr auto wpre_args = TensorAccessorArgs<4>();
    constexpr auto wpost_args = TensorAccessorArgs<wpre_args.next_compile_time_args_offset()>();

    const uint32_t page = get_local_cb_interface(cb_scratch).fifo_page_size;
    const auto s_wpre = TensorAccessor(wpre_args, w_pre_addr);
    const auto s_wpost = TensorAccessor(wpost_args, w_post_addr);

    Noc noc;
    CircularBuffer cbp(cb_pre), cbq(cb_post), cbc(cb_comb), cbs(cb_scratch);

    // Local scratch (never pushed): 24 tiles, zeroed once; only column 0 is rewritten afterwards.
    noc.async_write_zeros(cbs, N_SCRATCH * page);
    noc.write_zeros_l1_barrier();
    uint32_t* scratch = reinterpret_cast<uint32_t*>(cbs.get_write_ptr());

    for (uint32_t t = 0; t < num_token_tiles; ++t) {
        const uint32_t h = start_tile + t;
        cbp.wait_front(1);
        cbq.wait_front(1);
        cbc.wait_front(1);
        const uint32_t* pre = reinterpret_cast<const uint32_t*>(cbp.get_read_ptr());
        const uint32_t* post = reinterpret_cast<const uint32_t*>(cbq.get_read_ptr());
        const uint32_t* comb = reinterpret_cast<const uint32_t*>(cbc.get_read_ptr());

        for (uint32_t c = 0; c < 4; ++c) {  // w_pre[0][c] = h_pre[:, c]
            copy_col_to_col0(pre, scratch + c * TILE_WORDS, c);
        }
        for (uint32_t rr = 0; rr < 4; ++rr) {  // w_post[r][c] = H[:, r, c]; w_post[r][4] = h_post[:, r]
            uint32_t* base = scratch + (N_PRE + 5 * rr) * TILE_WORDS;
            for (uint32_t c = 0; c < 4; ++c) {
                copy_col_to_col0(comb, base + c * TILE_WORDS, 4 * rr + c);
            }
            copy_col_to_col0(post, base + 4 * TILE_WORDS, rr);
        }
        cbp.pop_front(1);
        cbq.pop_front(1);
        cbc.pop_front(1);

        for (uint32_t c = 0; c < N_PRE; ++c) {
            noc.async_write(cbs, s_wpre, page, {.offset_bytes = c * page}, {.page_id = c * total_tiles + h});
        }
        for (uint32_t k = 0; k < N_POST; ++k) {  // k = 5 r + c
            noc.async_write(
                cbs, s_wpost, page, {.offset_bytes = (N_PRE + k) * page}, {.page_id = k * total_tiles + h});
        }
        noc.async_write_barrier();  // the scratch is rewritten for the next token tile
    }
}
