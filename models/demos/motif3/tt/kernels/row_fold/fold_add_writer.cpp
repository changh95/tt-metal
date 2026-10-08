// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 decode MoE combine (D4): R + fold(S), writer (RISCV_0): output tiles [w PER, min(NO, (w + 1) PER)).
//
// CT: 0 cb_out, 1 NO, 2 PER, 3 GX, 4.. TensorAccessorArgs(out)
// common RT: 0 out addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t NO = get_compile_time_arg_val(1);
    constexpr uint32_t PER = get_compile_time_arg_val(2);
    constexpr uint32_t GX = get_compile_time_arg_val(3);
    constexpr auto o_args = TensorAccessorArgs<4>();
    constexpr uint32_t TB = 2048;
    const auto os = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), TB);
    const uint32_t w = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t o0 = w * PER;
    const uint32_t o1 = (o0 + PER < NO) ? o0 + PER : NO;
    for (uint32_t o = o0; o < o1; ++o) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), os.get_noc_addr(o, 0), TB);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
}
