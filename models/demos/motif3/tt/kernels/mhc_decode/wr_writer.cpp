// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 mHC decode stream mixing (D3): writer. Verbatim tt/mhc.py post_mix's writer: output row r of position i ->
// page r * inner + i of the [1, NUM_R, T, D] output (NUM_R = 1: x_red [1, 1, T, D], page i).
//
// CT args: 0 inner, 1 NUM_R, 2.. TensorAccessorArgs(out). Common RT args: 0 out_addr, 1 total, 2 num_cores, 3 grid_y.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "wr_expand.h"

void kernel_main() {
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t total = get_common_arg_val<uint32_t>(1);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t inner = get_compile_time_arg_val(0);
    constexpr uint32_t num_r = get_compile_time_arg_val(1);
    constexpr auto a_y = TensorAccessorArgs<2>();
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t out_bytes = get_tile_size(cb_out);
    const auto s_y = TensorAccessor(a_y, y_addr, out_bytes);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t q = total / num_cores, rem = total % num_cores;
    const uint32_t n = q + (core_i < rem ? 1 : 0);
    const uint32_t start = core_i * q + (core_i < rem ? core_i : rem);
    CircularBuffer co(cb_out);
#if MHC_EXPAND_SPLIT
    // second half of the weight-set expansion (decode: one token tile row, one set): the reader has reserved the
    // weight CB (the writer never advances it, so its write pointer is the reader's) and published the packed tile
    if (n != 0) {
        CircularBuffer cp(2), c1(1), cf(3);
        cp.wait_front(1);
        invalidate_l1_cache();
        uint32_t* w = reinterpret_cast<uint32_t*>(c1.get_write_ptr());
        const uint32_t* p = reinterpret_cast<const uint32_t*>(cp.get_read_ptr());
        constexpr uint32_t n_w = num_r * MHC_WR_NUM_C;
        for (uint32_t j = (MHC_EXPAND_SKIP ? n_w : (n_w + 1) / 2); j < n_w; ++j) {
            expand_weight<MHC_WR_HAS_OUT, 4, MHC_WR_NUM_C>(w, p, j);
        }
        invalidate_l1_cache();  // fence: the stores land before the flag
        cf.reserve_back(1);
        cf.push_back(1);
    }
#endif
    for (uint32_t i = start; i < start + n; ++i) {
        co.wait_front(num_r);
        uint32_t l1 = co.get_read_ptr();
        for (uint32_t r = 0; r < num_r; ++r) {
            noc_async_write(l1, s_y.get_noc_addr(r * inner + i), out_bytes);
            l1 += out_bytes;
        }
        noc_async_write_barrier();
        co.pop_front(num_r);
    }
}
