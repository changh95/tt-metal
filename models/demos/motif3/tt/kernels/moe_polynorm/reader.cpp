// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused grouped MoE PolyNorm (B3, docs/OPTIMIZATION_PLAN.md §3.3; prototype logs/opt/phaseA/M10), reader
// (RISCV_1).
//
// Worker q = GX y + x (logical core (x, y), row-major over the first E G cores) -> expert e = q / G, rank k = q % G.
// The worker owns gate columns [k n, k n + n) (n = IT / G tiles) of expert e for each of the R tile rows.
//   gu  [1, E, 32 R, 2 I] fp32 TILE interleaved: tile (e, tr, c) = (e R + tr) 2 IT + c   (IT = I / 32 gate tiles)
//   w   [1, E, 32 R, 1]   fp32 TILE interleaved: tile (e, tr) = e R + tr                  (routing weight, column 0)
//   D   [3, E, 1, 1]      fp32 TILE: tile (m, e) = m E + e, value at element 0          (D_m = 1 / (I c_m^2))
//   Ec  [3, E, 1, 1]      fp32 TILE: same layout                                        (E_m = eps / c_m^2)
//   b   [1, E, 1, 1]      fp32 TILE: tile e                                             (bias, clamped to +-0.5)
// The 7 constants of expert e land in CB_K (one fp32 tile) at byte offsets 64 j (j = D1 D2 D3 E1 E2 E3 b): the
// compute kernel reads them with read_tile_value (element 16 j). They are per-chip device tensors (EP placement), so
// one SPMD program serves all 32 chips.
// After the gate / up / w reads: wait until all G group members have written their moment partials into this core's
// CB_RECV (semaphore == G; bounded spin so a bug can never hang the shared device), re-arm the semaphore and publish
// CB_RECV.
//
// CT: 0 cb_g, 1 cb_u, 2 cb_w, 3 cb_recv, 4 cb_k, 5 sem_id, 6 n, 7 R, 8 G, 9 recv_tiles, 10 IT, 11 E, 12 GX,
//     13.. TensorAccessorArgs(gu), (w), (D), (Ec), (b)
// common RT: 0 gu_addr, 1 w_addr, 2 D_addr, 3 Ec_addr, 4 b_addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_g = get_compile_time_arg_val(0);
    constexpr uint32_t cb_u = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w = get_compile_time_arg_val(2);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(3);
    constexpr uint32_t cb_k = get_compile_time_arg_val(4);
    constexpr uint32_t sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t n = get_compile_time_arg_val(6);
    constexpr uint32_t R = get_compile_time_arg_val(7);
    constexpr uint32_t G = get_compile_time_arg_val(8);
    constexpr uint32_t recv_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t IT = get_compile_time_arg_val(10);
    constexpr uint32_t E = get_compile_time_arg_val(11);
    constexpr uint32_t GX = get_compile_time_arg_val(12);
    constexpr auto gu_args = TensorAccessorArgs<13>();
    constexpr auto w_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();

    const uint32_t gu_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t d_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t e_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t e = q / G;
    const uint32_t k = q % G;

    constexpr uint32_t tb = get_tile_size(cb_g);
    constexpr uint32_t kb = get_tile_size(cb_k);
    const auto gus = TensorAccessor(gu_args, gu_addr, tb);
    const auto ws = TensorAccessor(w_args, w_addr, get_tile_size(cb_w));
    const auto ds = TensorAccessor(d_args, d_addr, kb);
    const auto es = TensorAccessor(e_args, e_addr, kb);
    const auto bs = TensorAccessor(b_args, b_addr, kb);

    // the 7 per-expert constants (64 B each: one aligned read of the tile's first bytes)
    cb_reserve_back(cb_k, 1);
    const uint32_t kdst = get_write_ptr(cb_k);
    for (uint32_t m = 0; m < 3; ++m) {
        noc_async_read(ds.get_noc_addr(m * E + e), kdst + 64 * m, 64);
        noc_async_read(es.get_noc_addr(m * E + e), kdst + 64 * (3 + m), 64);
    }
    noc_async_read(bs.get_noc_addr(e), kdst + 64 * 6, 64);

    // gate tiles (phase 1 needs them first)
    cb_reserve_back(cb_g, n * R);
    uint32_t dst = get_write_ptr(cb_g);
    for (uint32_t tr = 0; tr < R; ++tr) {
        const uint32_t base = (e * R + tr) * 2 * IT + k * n;
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_read_page(base + j, gus, dst);
            dst += tb;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_k, 1);
    cb_push_back(cb_g, n * R);

    // up tiles + routing weight tiles
    cb_reserve_back(cb_u, n * R);
    dst = get_write_ptr(cb_u);
    for (uint32_t tr = 0; tr < R; ++tr) {
        const uint32_t base = (e * R + tr) * 2 * IT + IT + k * n;
        for (uint32_t j = 0; j < n; ++j) {
            noc_async_read_page(base + j, gus, dst);
            dst += tb;
        }
    }
    cb_reserve_back(cb_w, R);
    uint32_t wdst = get_write_ptr(cb_w);
    for (uint32_t tr = 0; tr < R; ++tr) {
        noc_async_read_page(e * R + tr, ws, wdst);
        wdst += get_tile_size(cb_w);
    }
    noc_async_read_barrier();
    cb_push_back(cb_u, n * R);
    cb_push_back(cb_w, R);

    // moment partials of the G group members (written by their writers into this core's CB_RECV slots)
    cb_reserve_back(cb_recv, recv_tiles);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    uint32_t it = 0;
    do {
        invalidate_l1_cache();
        if (*sem >= G) {
            break;
        }
    } while (++it < 200000000u);
    noc_semaphore_set(sem, 0);
    cb_push_back(cb_recv, recv_tiles);
}
