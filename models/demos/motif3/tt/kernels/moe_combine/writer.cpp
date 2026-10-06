// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 compacted prefill MoE: gather combine (B2b), writer (RISCV_0): the output tiles of worker q's units (the
// contiguous range [q NU / NW, (q + 1) NU / NW)), in the
// compute kernel's order, to part [1, 1, M, H] TILE (tile (tr, c) = tr WT + c).
//
// CT: 0 M, 1 WT, 2 G, 3 GX, 4 NW, 5 cb_o, 6.. TensorAccessorArgs(part)
// common RT: 0 part_addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2);
    constexpr uint32_t GX = get_compile_time_arg_val(3);
    constexpr uint32_t NW = get_compile_time_arg_val(4);
    constexpr uint32_t cb_o = get_compile_time_arg_val(5);
    constexpr auto o_args = TensorAccessorArgs<6>();
    constexpr uint32_t CW = WT / G;
    constexpr uint32_t NU = (M / 32) * G;

    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t tb = get_tile_size(cb_o);
    const auto os = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), tb);
    const uint32_t u0 = q * NU / NW;
    const uint32_t u1 = (q + 1) * NU / NW;
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t tr = u / G;
        const uint32_t c0 = (u % G) * CW;
        for (uint32_t c = c0; c < c0 + CW; ++c) {
            cb_wait_front(cb_o, 1);
            noc_async_write(get_read_ptr(cb_o), os.get_noc_addr(tr * WT + c, 0), tb);
            noc_async_writes_flushed();
            cb_pop_front(cb_o, 1);
        }
    }
    noc_async_write_barrier();
}
