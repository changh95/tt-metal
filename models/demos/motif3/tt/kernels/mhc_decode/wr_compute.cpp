// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 mHC decode stream mixing (D3): compute. Verbatim the compute of tt/mhc.py post_mix (_POST_COMPUTE_SRC),
// which is the stock attn_res_weighted_reduce_nc compute (num_groups = 1) with the work split taken from common
// runtime args: mul_tiles_bcast_cols with acc_to_dest = 1 (a MAC into DEST tile r), DEST zeroed by the packer on every
// tile_regs_release(). NUM_R = 1 / NUM_C = 4 is the pre reduce (the stock op with one site), NUM_R = 4 / NUM_C = 5 the
// post mix.
//
// CT args: 0 NUM_C, 1 Wt, 2 NUM_R. Common RT args: 0 total items, 1 num_cores, 2 grid_y (wr_expand.h wr_split).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"
#include "wr_expand.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t num_c = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_r = get_compile_time_arg_val(2);
    const uint32_t total = get_common_arg_val<uint32_t>(0);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(2);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const WrSplit sp = wr_split(total, num_cores, core_i);
    if (sp.n == 0) {
        return;
    }
    constexpr uint32_t ipp = items_per_position<MHC_WR_ITEMS, num_r>();
    const uint32_t p_first = sp.start / ipp;
    const uint32_t p_end = (sp.start + sp.n - 1) / ipp + 1;
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    CircularBuffer c0(cb_in0), c1(cb_in1), co(cb_out0);
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_in0, cb_in1);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        cb_in0, cb_in1, 1 /*acc_to_dest*/)));
    reconfig_data_format(cb_in0, cb_in1);
    constexpr uint32_t wset = num_r * num_c;
    uint32_t width_index = p_first % Wt;
    c1.wait_front(wset);
    for (uint32_t p = p_first; p < p_end; ++p) {
        if (p != p_first && width_index == 0) {
            c1.pop_front(wset);
            c1.wait_front(wset);
        }
        uint32_t r_lo, r_hi;
        row_range<MHC_WR_ITEMS, num_r>(sp, p, r_lo, r_hi);
        c0.wait_front(num_c);
        tile_regs_acquire();
        for (uint32_t c = 0; c < num_c; ++c) {
            for (uint32_t s = r_lo; s < r_hi; ++s) {  // output row s accumulates c = 0, 1, ... in order (DEST s - r_lo)
                mul_tiles_bcast_cols(cb_in0, cb_in1, c, s * num_c + c, s - r_lo);
            }
        }
        tile_regs_commit();
        c0.pop_front(num_c);
        co.reserve_back(r_hi - r_lo);
        pack_reconfig_data_format(cb_out0);
        tile_regs_wait();
        for (uint32_t s = 0; s < r_hi - r_lo; ++s) {
            pack_tile(s, cb_out0);
        }
        tile_regs_release();
        co.push_back(r_hi - r_lo);
        if (++width_index == Wt) {
            width_index = 0;
        }
    }
    c1.pop_front(wset);
}
