// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 exact-fp32 router logits (WAVE_A_REVIEW D1(b)), worker reader (RISCV_1).
//
// Worker q = 12 y + x (logical core (x, y), q = 32 g + h) owns experts [128 g, +128) x the k-tiles h, 32 + h, 64 + h,
// 96 + h (strided, so the 96 workers' reads of a chunk spread over all DRAM banks).
// Row 0: per chunk cc (one x k-tile) the 4 pre-permuted weight tiles (bf16, router_fp32.prepare_router_weight: tile
// 384 cc + 4 q + i = DEST tile 4 + i of that chunk) and x tile 32 cc + h of tile row 0, in the order the compute kernel
// consumes them. The weights are never popped before the end of the launch (CB_W holds all 16 tiles).
// Then per row r: prefetch the 4 x tiles of row r + 1 (tile ids 128 (r + 1) + 32 cc + h; the compute kernel needs
// them before the reduce-scatter of row r, see worker_compute.cpp), then wait until the 32 workers of the expert group
// have written their pieces of row r into receive buffer r % recv_bufs of this core's CB_RECV and incremented
// semaphore r % recv_bufs; re-arm that semaphore (no sender can reach row r + recv_bufs before this core has consumed
// row r) and publish the receive tiles to the compute kernel.
//
// CT args: 0 cb_x, 1 cb_w, 2 n_chunks, 3 w_tiles_per_chunk, 4 timing, 5 cb_time, 6 cb_recv, 7 sem_id0, 8 n_senders,
//          9 recv_tiles (per row), 10 x tile stride (32), 11 weight chunk stride (384), 12 x tiles per row (128),
//          13 recv_bufs, 14 grid_x (12), 15 k_groups (32), 16.. TensorAccessorArgs(x), TensorAccessorArgs(w)
// common RT args: 0 x_addr, 1 w_addr, 2 n_rows (no per-core RT args)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "timing.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t n_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t w_per_chunk = get_compile_time_arg_val(3);
    constexpr uint32_t timing = get_compile_time_arg_val(4);
    constexpr uint32_t cb_time = get_compile_time_arg_val(5);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(6);
    constexpr uint32_t sem_id0 = get_compile_time_arg_val(7);
    constexpr uint32_t n_senders = get_compile_time_arg_val(8);
    constexpr uint32_t recv_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t x_stride = get_compile_time_arg_val(10);
    constexpr uint32_t w_stride = get_compile_time_arg_val(11);
    constexpr uint32_t x_tiles_per_row = get_compile_time_arg_val(12);
    constexpr uint32_t recv_bufs = get_compile_time_arg_val(13);
    constexpr uint32_t grid_x = get_compile_time_arg_val(14);
    constexpr uint32_t k_groups = get_compile_time_arg_val(15);
    constexpr auto x_args = TensorAccessorArgs<16>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t n_rows = get_common_arg_val<uint32_t>(2);
    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * grid_x + get_absolute_logical_x();
    const uint32_t h = q % k_groups;
    const uint32_t w_tile0 = w_per_chunk * q;
    if constexpr (timing) {
        motif_router_stamp(get_write_ptr(cb_time), 12);
    }

    constexpr uint32_t x_bytes = get_tile_size(cb_x);
    constexpr uint32_t w_bytes = get_tile_size(cb_w);
    const auto xs = TensorAccessor(x_args, x_addr, x_bytes);
    const auto ws = TensorAccessor(w_args, w_addr, w_bytes);

    // ---- row 0: weights + x, chunk by chunk ----
    for (uint32_t cc = 0; cc < n_chunks; ++cc) {
        cb_reserve_back(cb_w, w_per_chunk);
        uint32_t dst = get_write_ptr(cb_w);
        for (uint32_t i = 0; i < w_per_chunk; ++i) {
            noc_async_read_page(w_tile0 + cc * w_stride + i, ws, dst);
            dst += w_bytes;
        }
        cb_reserve_back(cb_x, 1);
        noc_async_read_page(h + cc * x_stride, xs, get_write_ptr(cb_x));
        noc_async_read_barrier();
        cb_push_back(cb_w, w_per_chunk);
        cb_push_back(cb_x, 1);
    }
    if constexpr (timing) {
        motif_router_stamp(get_write_ptr(cb_time), 13);
    }

    for (uint32_t r = 0; r < n_rows; ++r) {
        if (r + 1 < n_rows) {  // prefetch the x tiles of the next row (CB_X holds two rows when n_rows > 1)
            const uint32_t x_tile0 = (r + 1) * x_tiles_per_row + h;
            for (uint32_t cc = 0; cc < n_chunks; ++cc) {
                cb_reserve_back(cb_x, 1);
                noc_async_read_page(x_tile0 + cc * x_stride, xs, get_write_ptr(cb_x));
                noc_async_read_barrier();
                cb_push_back(cb_x, 1);
            }
        }
        // ---- the 32 pieces of row r of this core's output half-face ----
        volatile tt_l1_ptr uint32_t* sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id0 + r % recv_bufs));
        cb_reserve_back(cb_recv, recv_tiles);
        noc_semaphore_wait(sem, n_senders);
        noc_semaphore_set(sem, 0);
        if constexpr (timing) {
            if (r == 0) {
                motif_router_stamp(get_write_ptr(cb_time), 14);
            }
        }
        cb_push_back(cb_recv, recv_tiles);
    }
}
