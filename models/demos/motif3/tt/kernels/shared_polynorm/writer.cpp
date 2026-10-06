// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused shared-expert PolyNorm (B5), writer (RISCV_0) of both programs: the first `count` tiles of the output
// tensor (interleaved), in order, one tile at a time as the compute kernel produces them.
//   moments: s [1, 3, 32, 32] fp32 (3 tiles);  apply: h [1, 1, 32, 32 n] bf16 (n tiles).
//
// CT: 0 cb_out, 1 count, 2.. TensorAccessorArgs(out)
// common RT: 0 out_addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t count = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t tb = get_tile_size(cb_out);
    const auto outs = TensorAccessor(out_args, out_addr, tb);
    for (uint32_t t = 0; t < count; ++t) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(t, outs, get_read_ptr(cb_out));
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
