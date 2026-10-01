// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 exact mHC coefficients (Option B, WAVE_A_REVIEW MHC-6): writer.
// Adapted from ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_split_sinkhorn/device/kernels/
// dataflow/writer_mhc_split_sinkhorn.cpp: writes the three per-token-tile outputs (h_pre [T,4], h_post [T,4],
// H [T,16], fp32 TILE) with the three writes of a token tile under one barrier.
//
// CT args: [cb_pre, cb_post, cb_comb, TensorAccessorArgs(pre)..., TensorAccessorArgs(post)..., TensorAccessorArgs(comb)...]
// Common RT args: [pre_addr, post_addr, comb_addr, num_tiles_total, num_cores, grid_y] (work split as in the reader)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t pre_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t post_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t comb_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t total_tiles = get_common_arg_val<uint32_t>(3);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(4);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(5);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t q = total_tiles / num_cores, r = total_tiles % num_cores;
    const uint32_t num_token_tiles = q + (core_i < r ? 1 : 0);
    const uint32_t start_tile = core_i * q + (core_i < r ? core_i : r);

    constexpr uint32_t cb_pre = get_compile_time_arg_val(0);
    constexpr uint32_t cb_post = get_compile_time_arg_val(1);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(2);
    constexpr auto pre_args = TensorAccessorArgs<3>();
    constexpr auto post_args = TensorAccessorArgs<pre_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();

    const uint32_t pre_page = get_local_cb_interface(cb_pre).fifo_page_size;
    const uint32_t post_page = get_local_cb_interface(cb_post).fifo_page_size;
    const uint32_t comb_page = get_local_cb_interface(cb_comb).fifo_page_size;

    const auto s_pre = TensorAccessor(pre_args, pre_addr);
    const auto s_post = TensorAccessor(post_args, post_addr);
    const auto s_comb = TensorAccessor(comb_args, comb_addr);

    Noc noc;
    CircularBuffer cbp(cb_pre), cbq(cb_post), cbc(cb_comb);

    for (uint32_t t = 0; t < num_token_tiles; ++t) {
        cbp.wait_front(1);
        cbq.wait_front(1);
        cbc.wait_front(1);
        noc.async_write(cbp, s_pre, pre_page, {}, {.page_id = start_tile + t});
        noc.async_write(cbq, s_post, post_page, {}, {.page_id = start_tile + t});
        noc.async_write(cbc, s_comb, comb_page, {}, {.page_id = start_tile + t});
        noc.async_write_barrier();
        cbp.pop_front(1);
        cbq.pop_front(1);
        cbc.pop_front(1);
    }
}
