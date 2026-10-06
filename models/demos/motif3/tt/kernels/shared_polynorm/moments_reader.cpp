// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused shared-expert PolyNorm (B5, docs/OPTIMIZATION_PLAN.md §3.3), moments program, reader (RISCV_1).
//
// One core. Reads this chip's n gate tiles of the decode gate_up output (one tile row):
//   gu [1, 1, 32, 2 n 32] fp32 TILE interleaved: gate tiles 0..n-1, up tiles n..2n-1.
//
// CT: 0 cb_g, 1 n, 2.. TensorAccessorArgs(gu)
// common RT: 0 gu_addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_g = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    constexpr auto gu_args = TensorAccessorArgs<2>();

    const uint32_t gu_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t tb = get_tile_size(cb_g);
    const auto gus = TensorAccessor(gu_args, gu_addr, tb);

    cb_reserve_back(cb_g, n);
    uint32_t dst = get_write_ptr(cb_g);
    for (uint32_t j = 0; j < n; ++j) {
        noc_async_read_page(j, gus, dst);
        dst += tb;
    }
    noc_async_read_barrier();
    cb_push_back(cb_g, n);
}
