// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused mHC decode coefficients (D3): writer. The packed coefficient tile -> NCOPY identical pages of P
// (pages 0 .. NCOPY - 1: one 32-token tile row; consumers read page core % NCOPY, spreading ~120 readers over NCOPY
// L1 banks).
//
// CT args: 0 NCOPY, 1.. TensorAccessorArgs(P). Common RT args: 0 p_addr

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t ncopy = get_compile_time_arg_val(0);
    constexpr auto a_p = TensorAccessorArgs<1>();
    constexpr uint32_t TB = 32 * 32 * 4;
    constexpr uint32_t cb_out = 16;
    const auto s_p = TensorAccessor(a_p, p_addr, TB);
    CircularBuffer co(cb_out);
    co.wait_front(1);
    const uint32_t l1 = co.get_read_ptr();
    for (uint32_t j = 0; j < ncopy; ++j) {
        noc_async_write(l1, s_p.get_noc_addr(j), TB);
    }
    noc_async_write_barrier();
    co.pop_front(1);
}
