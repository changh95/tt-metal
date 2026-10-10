// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode router tail (B4, docs/OPTIMIZATION_PLAN.md §3.3 "A5 and B4"), reader (RISCV_1).
//
// Worker q = GX y + x (logical core (x, y), row-major over the first M cores) owns gathered token row q
// (tile row tr = q / 32, row rr = q % 32 inside the tile).
//   scores [1, 1, M, NE] fp32 TILE interleaved: tile (tr, t) = tr NT + t, NT = NE / 32 (the sigmoid of the logits)
//   bias   [1, 1, 1, NE] fp32 TILE interleaved: tile t, row 0 (expert_bias)
//   ids    [1, E_LOC, 1, 1] fp32 TILE interleaved: tile 0, element 0 = the global id of this chip's first expert
// Row rr of a tile: face (rr / 16) * 2 (columns 0..15) at byte ((rr / 16) * 2) * 1024 + (rr % 16) * 64 and the next
// face (columns 16..31) 1024 bytes later; 64 B each. The NE values of the row land contiguously (column j at float
// j) in one fp32 page of CB_S, the bias row the same way in CB_B: the compute kernel adds the two pages elementwise
// (position-independent SFPU op), so the page's layout does not matter as long as both match.
//
// DESIGN-2 replica mode (REPL = 1, tt/replicas.py): the same CB_ID page also receives, at byte 64, the 64 B half-row of
// the lane mask lane [1, 1, M, 1] fp32 TILE holding this row's column 0 (LANE = 1), and the chip's table
// table [1, 1, 1, 400] uint32 ROW_MAJOR: worker 0 (the assignment coordinator) reads all of it to byte 256 (replica
// codes at 256, slot experts at 1792), the other workers only its slot experts (table bytes 1536..1599) to byte 1792.
//
// CT: 0 cb_s, 1 cb_b, 2 cb_id, 3 NT, 4 GX, 5.. TensorAccessorArgs(scores), (bias), (ids), then REPL, LANE,
//     TensorAccessorArgs(lane), (table) (copies of ids' when absent)
// common RT: 0 scores_addr, 1 bias_addr, 2 ids_addr, 3 lane_addr, 4 table_addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_s = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);
    constexpr uint32_t GX = get_compile_time_arg_val(4);
    constexpr auto s_args = TensorAccessorArgs<5>();
    constexpr auto b_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto i_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr uint32_t REPL = get_compile_time_arg_val(i_args.next_compile_time_args_offset());
    constexpr uint32_t LANE = get_compile_time_arg_val(i_args.next_compile_time_args_offset() + 1);
    constexpr auto l_args = TensorAccessorArgs<i_args.next_compile_time_args_offset() + 2>();
    constexpr auto t_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();

    const uint32_t s_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t i_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t tr = q >> 5;
    const uint32_t rr = q & 31;
    const uint32_t row_off = ((rr >> 4) * 2) * 1024 + (rr & 15) * 64;

    constexpr uint32_t tb = get_tile_size(cb_s);  // fp32 tile, 4096 B
    const auto ss = TensorAccessor(s_args, s_addr, tb);
    const auto bs = TensorAccessor(b_args, b_addr, tb);
    const auto is = TensorAccessor(i_args, i_addr, tb);

    cb_reserve_back(cb_s, 1);
    cb_reserve_back(cb_b, 1);
    cb_reserve_back(cb_id, 1);
    const uint32_t sdst = get_write_ptr(cb_s);
    const uint32_t bdst = get_write_ptr(cb_b);
    for (uint32_t t = 0; t < NT; ++t) {
        noc_async_read(ss.get_noc_addr(tr * NT + t, row_off), sdst + t * 128, 64);
        noc_async_read(ss.get_noc_addr(tr * NT + t, row_off + 1024), sdst + t * 128 + 64, 64);
        noc_async_read(bs.get_noc_addr(t, 0), bdst + t * 128, 64);
        noc_async_read(bs.get_noc_addr(t, 1024), bdst + t * 128 + 64, 64);
    }
    noc_async_read(is.get_noc_addr(0, 0), get_write_ptr(cb_id), 64);
    if constexpr (REPL != 0) {
        constexpr uint32_t TABLE_BYTES = 1600;  // 400 uint32: 384 replica codes + 16 slot experts
        const auto ts = TensorAccessor(t_args, get_common_arg_val<uint32_t>(4), TABLE_BYTES);
        if (q == 0) {
            noc_async_read(ts.get_noc_addr(0, 0), get_write_ptr(cb_id) + 256, TABLE_BYTES);
        } else {
            noc_async_read(ts.get_noc_addr(0, 1536), get_write_ptr(cb_id) + 1792, 64);
        }
        if constexpr (LANE != 0) {
            const auto ls = TensorAccessor(l_args, get_common_arg_val<uint32_t>(3), tb);
            noc_async_read(ls.get_noc_addr(tr, row_off), get_write_ptr(cb_id) + 64, 64);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_s, 1);
    cb_push_back(cb_b, 1);
    cb_push_back(cb_id, 1);
}
