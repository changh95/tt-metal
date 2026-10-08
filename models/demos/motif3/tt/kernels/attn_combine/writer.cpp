// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode attention combine (D1), writer (RISCV_0): output tiles [q PER, min(n, (q + 1) PER)) of
// dg [1, 1, 32, Sg vdim] (interleaved), in order, as the compute kernel produces them.
//
// CT: 0 cb_out, 1 n, 2 PER, 3 GX, 4.. TensorAccessorArgs(out)
// common RT: 0 out_addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    constexpr uint32_t PER = get_compile_time_arg_val(2);
    constexpr uint32_t GX = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();

    constexpr uint32_t tb = get_tile_size(cb_out);
    const auto outs = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0), tb);
    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t j0 = q * PER;
    const uint32_t j1 = (j0 + PER < n) ? j0 + PER : n;
    for (uint32_t j = j0; j < j1; ++j) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(j, outs, get_read_ptr(cb_out));
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
