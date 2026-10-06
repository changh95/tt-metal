// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused grouped MoE PolyNorm (B3; prototype logs/opt/phaseA/M10), writer (RISCV_0).
//
// Moment exchange: this core's 3 R partial tiles (fp32; the moment sums of its gate columns in column 0, rows =
// tokens) go to slot k of CB_RECV on every member of its expert group (the same L1 address on every core: CB_RECV is
// allocated identically on all workers), then each member's semaphore is incremented. Only faces 0 and 2 of a tile
// (the ones holding column 0) are sent. Then the n R output tiles (bf16) are written to h [1, E, 32 R, I]:
// tile (e, tr, c) = (e R + tr) IT + c.
//
// CT: 0 cb_part, 1 cb_out, 2 cb_recv, 3 sem_id, 4 n, 5 R, 6 G, 7 IT, 8 GX, 9..24 NoC x of logical columns 0..15,
//     25..40 NoC y of logical rows 0..15 (unused entries 0), 41.. TensorAccessorArgs(h)
// common RT: 0 h_addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t NX0 = 9;
constexpr uint32_t NY0 = 25;
constexpr uint32_t NOC_X[16] = {
    get_compile_time_arg_val(NX0 + 0),  get_compile_time_arg_val(NX0 + 1),  get_compile_time_arg_val(NX0 + 2),
    get_compile_time_arg_val(NX0 + 3),  get_compile_time_arg_val(NX0 + 4),  get_compile_time_arg_val(NX0 + 5),
    get_compile_time_arg_val(NX0 + 6),  get_compile_time_arg_val(NX0 + 7),  get_compile_time_arg_val(NX0 + 8),
    get_compile_time_arg_val(NX0 + 9),  get_compile_time_arg_val(NX0 + 10), get_compile_time_arg_val(NX0 + 11),
    get_compile_time_arg_val(NX0 + 12), get_compile_time_arg_val(NX0 + 13), get_compile_time_arg_val(NX0 + 14),
    get_compile_time_arg_val(NX0 + 15)};
constexpr uint32_t NOC_Y[16] = {
    get_compile_time_arg_val(NY0 + 0),  get_compile_time_arg_val(NY0 + 1),  get_compile_time_arg_val(NY0 + 2),
    get_compile_time_arg_val(NY0 + 3),  get_compile_time_arg_val(NY0 + 4),  get_compile_time_arg_val(NY0 + 5),
    get_compile_time_arg_val(NY0 + 6),  get_compile_time_arg_val(NY0 + 7),  get_compile_time_arg_val(NY0 + 8),
    get_compile_time_arg_val(NY0 + 9),  get_compile_time_arg_val(NY0 + 10), get_compile_time_arg_val(NY0 + 11),
    get_compile_time_arg_val(NY0 + 12), get_compile_time_arg_val(NY0 + 13), get_compile_time_arg_val(NY0 + 14),
    get_compile_time_arg_val(NY0 + 15)};
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(2);
    constexpr uint32_t sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t n = get_compile_time_arg_val(4);
    constexpr uint32_t R = get_compile_time_arg_val(5);
    constexpr uint32_t G = get_compile_time_arg_val(6);
    constexpr uint32_t IT = get_compile_time_arg_val(7);
    constexpr uint32_t GX = get_compile_time_arg_val(8);
    constexpr auto h_args = TensorAccessorArgs<41>();

    const uint32_t h_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t e = q / G;
    const uint32_t k = q % G;

    constexpr uint32_t ptb = get_tile_size(cb_part);  // fp32 tile
    constexpr uint32_t slot_bytes = 3 * R * ptb;
    const uint32_t recv_base = get_write_ptr(cb_recv);  // never advanced on this RISC: the CB base on every core

    cb_wait_front(cb_part, 3 * R);
    const uint32_t src = get_read_ptr(cb_part);
    const uint32_t sem_addr = get_semaphore(sem_id);
    for (uint32_t pp = 0; pp < G; ++pp) {
        const uint32_t p = (pp + k) % G;
        const uint32_t qp = e * G + p;
        const uint32_t nx = NOC_X[qp % GX];
        const uint32_t ny = NOC_Y[qp / GX];
        const uint32_t dst = recv_base + k * slot_bytes;
        for (uint32_t t = 0; t < 3 * R; ++t) {
            noc_async_write(src + t * ptb, get_noc_addr(nx, ny, dst + t * ptb), ptb / 4);  // face 0
            noc_async_write(src + t * ptb + ptb / 2, get_noc_addr(nx, ny, dst + t * ptb + ptb / 2), ptb / 4);  // face 2
        }
    }
    noc_async_write_barrier();
    for (uint32_t pp = 0; pp < G; ++pp) {
        const uint32_t p = (pp + k) % G;
        const uint32_t qp = e * G + p;
        noc_semaphore_inc(get_noc_addr(NOC_X[qp % GX], NOC_Y[qp / GX], sem_addr), 1);
    }
    noc_async_atomic_barrier();
    cb_pop_front(cb_part, 3 * R);

    constexpr uint32_t otb = get_tile_size(cb_out);
    const auto hs = TensorAccessor(h_args, h_addr, otb);
    for (uint32_t tr = 0; tr < R; ++tr) {
        const uint32_t base = (e * R + tr) * IT + k * n;
        for (uint32_t j = 0; j < n; ++j) {
            cb_wait_front(cb_out, 1);
            noc_async_write_page(base + j, hs, get_read_ptr(cb_out));
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
}
