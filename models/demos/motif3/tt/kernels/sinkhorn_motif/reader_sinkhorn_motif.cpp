// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 exact mHC coefficients (Option B, WAVE_A_REVIEW MHC-6): reader.
// Adapted from ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_split_sinkhorn/device/kernels/
// dataflow/reader_mhc_split_sinkhorn.cpp. Differences: 2 resident constant tiles (alpha tile, bias tile; see
// models/demos/motif3/tt/kernels/sinkhorn_motif.py) instead of 8, and the constant reads are issued together with
// the first mixes tile under one barrier (one DRAM round trip instead of three on the T <= 32 decode path).
//
// CT args: [cb_mixes, cb_consts, TensorAccessorArgs(mixes)..., TensorAccessorArgs(consts)...]
// Common RT args (one set for all cores, so host dispatch cost does not grow with the core count):
//   [mixes_addr, consts_addr, num_tiles_total, num_cores, grid_y]
// Core i = x * grid_y + y (column-major, = sinkhorn_motif._split_work) owns tiles [start, start + n) with
// n = q + (i < r), start = i q + min(i, r), (q, r) = divmod(num_tiles_total, num_cores).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t mixes_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t consts_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t total_tiles = get_common_arg_val<uint32_t>(2);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(4);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t q = total_tiles / num_cores, r = total_tiles % num_cores;
    const uint32_t num_token_tiles = q + (core_i < r ? 1 : 0);
    const uint32_t start_tile = core_i * q + (core_i < r ? core_i : r);

    constexpr uint32_t cb_mixes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_consts = get_compile_time_arg_val(1);
    constexpr auto mixes_args = TensorAccessorArgs<2>();
    constexpr auto consts_args = TensorAccessorArgs<mixes_args.next_compile_time_args_offset()>();
    constexpr uint32_t num_const_tiles = 2;

    const uint32_t mixes_page = get_local_cb_interface(cb_mixes).fifo_page_size;
    const uint32_t consts_page = get_local_cb_interface(cb_consts).fifo_page_size;

    const auto s_mixes = TensorAccessor(mixes_args, mixes_addr);
    const auto s_consts = TensorAccessor(consts_args, consts_addr);

    Noc noc;
    CircularBuffer cb_c(cb_consts);
    CircularBuffer cb_m(cb_mixes);

    // Constants (resident for the whole op; compute never pops them) + the first token tile, one barrier.
    cb_c.reserve_back(num_const_tiles);
    for (uint32_t i = 0; i < num_const_tiles; ++i) {
        noc.async_read(s_consts, cb_c, consts_page, {.page_id = i}, {.offset_bytes = i * consts_page});
    }
    uint32_t t = 0;
    if (num_token_tiles > 0) {
        cb_m.reserve_back(1);
        noc.async_read(s_mixes, cb_m, mixes_page, {.page_id = start_tile}, {.offset_bytes = 0});
    }
    noc.async_read_barrier();
    cb_c.push_back(num_const_tiles);
    if (num_token_tiles > 0) {
        cb_m.push_back(1);
        t = 1;
    }

    for (; t < num_token_tiles; ++t) {
        cb_m.reserve_back(1);
        noc.async_read(s_mixes, cb_m, mixes_page, {.page_id = start_tile + t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_m.push_back(1);
    }
}
