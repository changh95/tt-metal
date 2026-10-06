// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused shared-expert PolyNorm (B5, docs/OPTIMIZATION_PLAN.md §3.3), apply program, reader (RISCV_1).
//
// One core. Inputs (per chip):
//   gat [1, 3, 32, 32 TP] fp32 TILE: the TP all-gather of every chip's moment tile; tile (m, k) = m TP + k holds chip
//                                    position k's sum of g^(2m+2) in column 0
//   gu  [1, 1, 32, 2 n 32] fp32 TILE: gate tiles 0..n-1, up tiles n..2n-1
//   D, E [1, 3, 1, 1] fp32 TILE: tile m, value at element 0 (D_m = 1 / (I c_m^2), E_m = eps / c_m^2;
//                                ScalarPolyNormConsts, replicated)
//   b   [1, 1, 1, 1] fp32 TILE: tile 0, element 0
// The 7 constants land in CB_K (tile j = D1 D2 D3 E1 E2 E3 b), each value broadcast down column 0 (rows 0..31) as in
// the B3 reader (kernels/moe_polynorm/reader.cpp); the compute kernel reads column 0 only. Of each gathered moment tile
// only faces 0 and 2 (the ones holding column 0) are read; the compute kernel's fold and s D + E touch only those, and
// the stale faces 1 / 3 are zeroed (non-finite) before the column broadcast. Each moment's TP tiles are pushed as soon
// as they land, so the compute kernel folds moment m while moment m + 1 is in flight.
//
// CT: 0 cb_recv, 1 cb_k, 2 cb_g, 3 cb_u, 4 n, 5 TP, 6.. TensorAccessorArgs(gat), (gu), (D), (E), (b)
// common RT: 0 gat_addr, 1 gu_addr, 2 D_addr, 3 E_addr, 4 b_addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_recv = get_compile_time_arg_val(0);
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);
    constexpr uint32_t cb_g = get_compile_time_arg_val(2);
    constexpr uint32_t cb_u = get_compile_time_arg_val(3);
    constexpr uint32_t n = get_compile_time_arg_val(4);
    constexpr uint32_t TP = get_compile_time_arg_val(5);
    constexpr auto gat_args = TensorAccessorArgs<6>();
    constexpr auto gu_args = TensorAccessorArgs<gat_args.next_compile_time_args_offset()>();
    constexpr auto d_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    constexpr auto e_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();

    const uint32_t gat_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gu_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t d_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t e_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(4);

    constexpr uint32_t tb = get_tile_size(cb_recv);
    constexpr uint32_t kb = get_tile_size(cb_k);
    const auto gats = TensorAccessor(gat_args, gat_addr, tb);
    const auto gus = TensorAccessor(gu_args, gu_addr, get_tile_size(cb_g));
    const auto ds = TensorAccessor(d_args, d_addr, kb);
    const auto es = TensorAccessor(e_args, e_addr, kb);
    const auto bs = TensorAccessor(b_args, b_addr, kb);

    // constants: one aligned 64 B read of each constant tile's first bytes into tile j of CB_K
    cb_reserve_back(cb_k, 7);
    const uint32_t kdst = get_write_ptr(cb_k);
    for (uint32_t m = 0; m < 3; ++m) {
        noc_async_read(ds.get_noc_addr(m), kdst + kb * m, 64);
        noc_async_read(es.get_noc_addr(m), kdst + kb * (3 + m), 64);
    }
    noc_async_read(bs.get_noc_addr(0), kdst + kb * 6, 64);

    // the first moment's gathered tiles (faces 0 and 2)
    uint32_t dst = 0;
    for (uint32_t m = 0; m < 1; ++m) {
        cb_reserve_back(cb_recv, TP);
        dst = get_write_ptr(cb_recv);
        for (uint32_t k = 0; k < TP; ++k) {
            const uint64_t src = gats.get_noc_addr(m * TP + k);
            noc_async_read(src, dst, tb / 4);
            noc_async_read(src + tb / 2, dst + tb / 2, tb / 4);
            dst += tb;
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_recv, TP);
    // broadcast each constant (element 0) down column 0 (rows 0..15: face 0, rows 16..31: face 2; 16 fp32 per face row)
    for (uint32_t j = 0; j < 7; ++j) {
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kdst + kb * j);
        const uint32_t v = t[0];
        for (uint32_t r = 1; r < 16; ++r) {
            t[16 * r] = v;
        }
        for (uint32_t r = 0; r < 16; ++r) {
            t[512 + 16 * r] = v;
        }
    }
    cb_push_back(cb_k, 7);
    for (uint32_t m = 1; m < 3; ++m) {
        cb_reserve_back(cb_recv, TP);
        dst = get_write_ptr(cb_recv);
        for (uint32_t k = 0; k < TP; ++k) {
            const uint64_t src = gats.get_noc_addr(m * TP + k);
            noc_async_read(src, dst, tb / 4);
            noc_async_read(src + tb / 2, dst + tb / 2, tb / 4);
            dst += tb;
        }
        noc_async_read_barrier();
        cb_push_back(cb_recv, TP);
    }

    // gate and up tiles
    constexpr uint32_t gb = get_tile_size(cb_g);
    cb_reserve_back(cb_g, n);
    dst = get_write_ptr(cb_g);
    for (uint32_t j = 0; j < n; ++j) {
        noc_async_read_page(j, gus, dst);
        dst += gb;
    }
    cb_reserve_back(cb_u, n);
    uint32_t udst = get_write_ptr(cb_u);
    for (uint32_t j = 0; j < n; ++j) {
        noc_async_read_page(n + j, gus, udst);
        udst += gb;
    }
    noc_async_read_barrier();
    cb_push_back(cb_g, n);
    cb_push_back(cb_u, n);
}
