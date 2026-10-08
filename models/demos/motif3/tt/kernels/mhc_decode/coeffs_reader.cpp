// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused mHC decode coefficients (D3): reader (one core, one 32-token tile).
// In the order compute consumes them: the NB projection partials (pages b of Y [1, NB, 32, 32] fp32), the NB sum(X^2)
// partials (pages b of S), then the site constants (2 tiles: alpha, bias). With HALF (T <= 16) only the bytes that
// reach the outputs are read: faces 0-1 of Y (token rows 0..15) and face 0 of S (rows 0..15, column 0).
//
// CT args: 0 NB, 1 CHUNK, 2 HALF, 3.. TensorAccessorArgs(Y), (S), (consts)
// Common RT args: 0 y_addr, 1 s_addr, 2 consts_addr

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(2);
    constexpr uint32_t nb = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t half = get_compile_time_arg_val(2);
    constexpr auto a_y = TensorAccessorArgs<3>();
    constexpr auto a_s = TensorAccessorArgs<a_y.next_compile_time_args_offset()>();
    constexpr auto a_c = TensorAccessorArgs<a_s.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 32 * 32 * 4;
    constexpr uint32_t y_bytes = half ? TB / 2 : TB;
    constexpr uint32_t s_bytes = half ? TB / 4 : TB;
    constexpr uint32_t cb_y = 0, cb_s = 1, cb_c = 2;
    const auto s_y = TensorAccessor(a_y, y_addr, TB);
    const auto s_s = TensorAccessor(a_s, s_addr, TB);
    const auto s_c = TensorAccessor(a_c, c_addr, TB);
    CircularBuffer cy(cb_y), cs(cb_s), cc(cb_c);
    for (uint32_t b = 0; b < nb; b += chunk) {
        cy.reserve_back(chunk);
        const uint32_t w = cy.get_write_ptr();
        for (uint32_t i = 0; i < chunk; ++i) {
            noc_async_read(s_y.get_noc_addr(b + i), w + i * TB, y_bytes);
        }
        noc_async_read_barrier();
        cy.push_back(chunk);
    }
    for (uint32_t b = 0; b < nb; b += chunk) {
        cs.reserve_back(chunk);
        const uint32_t w = cs.get_write_ptr();
        for (uint32_t i = 0; i < chunk; ++i) {
            noc_async_read(s_s.get_noc_addr(b + i), w + i * TB, s_bytes);
        }
        noc_async_read_barrier();
        cs.push_back(chunk);
    }
    cc.reserve_back(2);
    const uint32_t w = cc.get_write_ptr();
    noc_async_read(s_c.get_noc_addr(0), w, TB);
    noc_async_read(s_c.get_noc_addr(1), w + TB, TB);
    noc_async_read_barrier();
    cc.push_back(2);
}
