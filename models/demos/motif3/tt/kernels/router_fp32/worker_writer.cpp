// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 exact-fp32 router logits (WAVE_A_REVIEW D1(b)), worker writer (RISCV_0).
//
// Worker q = 12 y + x (logical core (x, y)) = 32 g + h. Per row r:
// Reduce-scatter send: the worker's 4 partial-logit tiles of the row (fp32, natural tile layout: 32 tokens x 32 experts
// of output tile 4 g + j) are split into 32 pieces of 512 B (piece p = tile p / 8, face (p % 8) / 2, half p % 2: 8
// token rows x 16 experts, contiguous in the tile). Piece p goes to slot h of receive buffer r % recv_bufs of worker
// (g, p)'s CB_RECV (same address on every worker), then that worker's semaphore r % recv_bufs is incremented.
// Optional debug dump (row 0 only): the 4 partial tiles also go to DRAM tensor pages 4 q + j.
// Final: the 512 B piece h of row r (summed by this core's compute, one row later when n_rows > 1: see
// worker_compute.cpp) is written to the logits tensor, page 12 r + 4 g + h / 8, byte offset 512 (h % 8).
// Optional timing dump (timing.h; row-0 stamps, the last final write).
//
// CT args: 0 cb_out, 1 cb_recv, 2 sem_id0, 3 n_vec, 4 debug (0/1), 5 timing (0/1), 6 cb_time, 7 cb_fin, 8 n_pieces,
//          9 recv_bufs, 10 grid_x (12), 11 k_groups (32), 12 out tiles per row (12), 13 recv_tiles (per row),
//          14 .. 14 + grid_x - 1: NoC x of logical columns 0 .. grid_x - 1, next 8: NoC y of logical rows 0..7,
//          then TensorAccessorArgs(debug tensor), TensorAccessorArgs(timing tensor), TensorAccessorArgs(out)
//          (fillers when off)
// common RT args: 0 debug addr, 1 timing addr, 2 out addr, 3 n_rows (no per-core RT args)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "timing.h"

namespace {
constexpr uint32_t GRID_X = get_compile_time_arg_val(10);
constexpr uint32_t GRID_ROWS = 8;
constexpr uint32_t NOC_XY0 = 14;
static_assert(GRID_X == 12, "router_fp32: 12 x 8 worker grid");
// NoC (virtual) coordinates of the logical worker columns / rows (host: worker_core_from_logical_core, checked there
// to be a separable grid). Indexed at run time (rodata).
constexpr uint32_t NOC_X[GRID_X] = {
    get_compile_time_arg_val(NOC_XY0 + 0), get_compile_time_arg_val(NOC_XY0 + 1),
    get_compile_time_arg_val(NOC_XY0 + 2), get_compile_time_arg_val(NOC_XY0 + 3),
    get_compile_time_arg_val(NOC_XY0 + 4), get_compile_time_arg_val(NOC_XY0 + 5),
    get_compile_time_arg_val(NOC_XY0 + 6), get_compile_time_arg_val(NOC_XY0 + 7),
    get_compile_time_arg_val(NOC_XY0 + 8), get_compile_time_arg_val(NOC_XY0 + 9),
    get_compile_time_arg_val(NOC_XY0 + 10), get_compile_time_arg_val(NOC_XY0 + 11)};
constexpr uint32_t NOC_Y[GRID_ROWS] = {
    get_compile_time_arg_val(NOC_XY0 + GRID_X + 0), get_compile_time_arg_val(NOC_XY0 + GRID_X + 1),
    get_compile_time_arg_val(NOC_XY0 + GRID_X + 2), get_compile_time_arg_val(NOC_XY0 + GRID_X + 3),
    get_compile_time_arg_val(NOC_XY0 + GRID_X + 4), get_compile_time_arg_val(NOC_XY0 + GRID_X + 5),
    get_compile_time_arg_val(NOC_XY0 + GRID_X + 6), get_compile_time_arg_val(NOC_XY0 + GRID_X + 7)};
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(1);
    constexpr uint32_t sem_id0 = get_compile_time_arg_val(2);
    constexpr uint32_t n_vec = get_compile_time_arg_val(3);
    constexpr uint32_t debug = get_compile_time_arg_val(4);
    constexpr uint32_t timing = get_compile_time_arg_val(5);
    constexpr uint32_t cb_time = get_compile_time_arg_val(6);
    constexpr uint32_t cb_fin = get_compile_time_arg_val(7);
    constexpr uint32_t n_pieces = get_compile_time_arg_val(8);
    constexpr uint32_t recv_bufs = get_compile_time_arg_val(9);
    constexpr uint32_t k_groups = get_compile_time_arg_val(11);
    constexpr uint32_t out_tiles_per_row = get_compile_time_arg_val(12);
    constexpr uint32_t recv_tiles = get_compile_time_arg_val(13);
    constexpr auto dbg_args = TensorAccessorArgs<NOC_XY0 + GRID_X + GRID_ROWS>();
    constexpr auto time_args = TensorAccessorArgs<dbg_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<time_args.next_compile_time_args_offset()>();
    static_assert(n_pieces == k_groups, "one piece per k group");

    const uint32_t dbg_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t time_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t n_rows = get_common_arg_val<uint32_t>(3);
    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GRID_X + get_absolute_logical_x();
    const uint32_t g = q / k_groups;
    const uint32_t slot = q % k_groups;  // h
    if constexpr (timing) {
        motif_router_stamp(get_write_ptr(cb_time), 8);
    }

    constexpr uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t piece_bytes = tile_bytes * n_vec / n_pieces;  // 512
    constexpr uint32_t recv_buf_bytes = tile_bytes * recv_tiles;     // one row of pieces (16 KB)
    const uint32_t recv_slot_addr = get_write_ptr(cb_recv) + slot * piece_bytes;  // CB_RECV base (never pushed here)
    const uint32_t out_page0 = n_vec * g + slot / 8;
    const uint32_t out_offset = piece_bytes * (slot % 8);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    // receivers of this core's pieces: the 32 workers (g, p), staggered (k group h starts with receiver h)
    uint8_t rx[n_pieces];  // NoC coordinates < 256 (64 B of stack)
    uint8_t ry[n_pieces];
    for (uint32_t pp = 0; pp < n_pieces; ++pp) {
        const uint32_t p = (pp + slot) % n_pieces;
        const uint32_t qr = k_groups * g + p;
        rx[pp] = NOC_X[qr % GRID_X];
        ry[pp] = NOC_Y[qr / GRID_X];
    }

    auto write_final = [&](uint32_t row) {
        cb_wait_front(cb_fin, 1);
        noc_async_write(
            get_read_ptr(cb_fin), out.get_noc_addr(out_tiles_per_row * row + out_page0, out_offset), piece_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_fin, 1);
    };

    for (uint32_t r = 0; r < n_rows; ++r) {
        cb_wait_front(cb_out, n_vec);
        if constexpr (timing) {
            if (r == 0) {
                motif_router_stamp(get_write_ptr(cb_time), 9);
            }
        }
        const uint32_t src = get_read_ptr(cb_out);
        if constexpr (debug) {
            if (r == 0) {
                const auto dbg = TensorAccessor(dbg_args, dbg_addr, tile_bytes);
                for (uint32_t j = 0; j < n_vec; ++j) {
                    noc_async_write_page(q * n_vec + j, dbg, src + j * tile_bytes);
                }
            }
        }
        const uint32_t buf = r % recv_bufs;
        const uint32_t recv_addr = recv_slot_addr + buf * recv_buf_bytes;
        const uint32_t sem_addr = get_semaphore(sem_id0 + buf);
        for (uint32_t pp = 0; pp < n_pieces; ++pp) {
            const uint32_t p = (pp + slot) % n_pieces;
            noc_async_write(src + p * piece_bytes, get_noc_addr(rx[pp], ry[pp], recv_addr), piece_bytes);
        }
        noc_async_write_barrier();
        if constexpr (timing) {
            if (r == 0) {
                motif_router_stamp(get_write_ptr(cb_time), 10);
            }
        }
        for (uint32_t pp = 0; pp < n_pieces; ++pp) {
            noc_semaphore_inc(get_noc_addr(rx[pp], ry[pp], sem_addr), 1);
        }
        noc_async_atomic_barrier();
        cb_pop_front(cb_out, n_vec);
        if constexpr (timing) {
            if (r == 0) {
                motif_router_stamp(get_write_ptr(cb_time), 11);
            }
        }
        if (r >= 1) {
            write_final(r - 1);
        }
    }
    write_final(n_rows - 1);
    if constexpr (timing) {
        motif_router_stamp(get_write_ptr(cb_time), 15);
        const auto tt = TensorAccessor(time_args, time_addr, 64);
        noc_async_write_page(q, tt, get_write_ptr(cb_time));
        noc_async_write_barrier();
    }
}
