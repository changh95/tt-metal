// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

// #include "debug_print.h"
// SliceRange srr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };
// SliceRange srt = SliceRange{ .h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 4 };


// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
inline bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile uint16_t* ptr = reinterpret_cast<volatile uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++ i) {
        ptr[i] = val;
    }
    return true;
}

// inline void reset_cb(uint32_t cb_id, uint16_t val) {
//     uint32_t wr_ptr = cb_interface[cb_id].fifo_wr_ptr;
//     uint32_t sz_nbytes = cb_interface[cb_id].fifo_size >> 4;
//     uint32_t sz_nelems = sz_nbytes >> 1;
//     fill_with_val(wr_ptr, sz_nelems, val);
// }

// inline void print_cb_details(uint32_t cb_id) {
//     DPRINT << "in_cb_id: { "
//             << "size: " << cb_interface[cb_id].fifo_size << ", "
//             << "limit: " << cb_interface[cb_id].fifo_limit << ", "
//             << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
//             << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
//             << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
//             << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
//             << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }" << ENDL();
// }

/**
 * Max-pool 2D. Highly Unoptimized!!
 * TODO [AS]: reuse data moved to L1 instead of reading every time
 */
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t window_h = get_arg_val<uint32_t>(2);
    const uint32_t window_w = get_arg_val<uint32_t>(3);
    const uint32_t window_hw = get_arg_val<uint32_t>(4);
    const uint32_t window_hw_padded = get_arg_val<uint32_t>(5);
    const uint32_t stride_h = get_arg_val<uint32_t>(6);
    const uint32_t stride_w = get_arg_val<uint32_t>(7);
    const int32_t pad_h = get_arg_val<int32_t>(8);
    const int32_t pad_w = get_arg_val<int32_t>(9);
    const int32_t out_h = get_arg_val<int32_t>(10);
    const int32_t out_w = get_arg_val<int32_t>(11);
    const uint32_t in_nbytes_c = get_arg_val<uint32_t>(14);
    const int32_t in_h = get_arg_val<int32_t>(16);
    const int32_t in_w = get_arg_val<int32_t>(17);
    const int32_t in_c = get_arg_val<int32_t>(19);
    const int32_t in_cb_pagesize = get_arg_val<int32_t>(22);
    const int32_t in_cb_page_nelems_padded = get_arg_val<int32_t>(24);
    const int32_t out_w_loop_count = get_arg_val<int32_t>(25);
    const uint32_t in_log_base_2_of_page_size = get_arg_val<uint32_t>(26);

    constexpr bool is_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bf16_one_u32 = get_compile_time_arg_val(2);
    constexpr uint32_t out_nelems = get_compile_time_arg_val(3);
    constexpr bool use_pow2 = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t in_cb_id = tt::CB::c_in0;
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in1;

    constexpr uint32_t TILE_HW = 1024;

    // ROW_MAJOR input
    const InterleavedPow2AddrGenFast<is_in_dram> s_in = {
        .bank_base_address = in_addr,
        .log_base_2_of_page_size = in_log_base_2_of_page_size
    };
    // const InterleavedAddrGen<is_in_dram> s_in = {
    //     .bank_base_address = in_addr,
    //     .page_size = in_nbytes_c
    // };


    // Reduce scalar = 1
    cb_reserve_back(in_scalar_cb_id, 1);

    kernel_profiler::mark_time(7);

    uint16_t bf16_one_u16 = bf16_one_u32 >> 16;
    fill_with_val(get_write_ptr(in_scalar_cb_id), TILE_HW, bf16_one_u16);
    cb_push_back(in_scalar_cb_id, 1);

    uint32_t start_in_row_id = 0;
    uint32_t out_row_id = 0;

    // fill in_cb_id rows with -inf
    uint32_t in_l1_write_addr = get_write_ptr(in_cb_id);
    fill_with_val(in_l1_write_addr, in_cb_page_nelems_padded * out_nelems * 2, 0xff7f);

    kernel_profiler::mark_time(8);

    int32_t start_h = - pad_h;
    // for every output row (across all channels)
    for (int32_t out_h_i = 0; out_h_i < out_h; ++ out_h_i) {
        int32_t start_w = - pad_w;
        // for every output col
        for (int32_t out_w_i = 0; out_w_i < out_w_loop_count; ++ out_w_i) {
            // make sure cb is available to fill
            cb_reserve_back(in_cb_id, out_nelems);
            uint32_t in_l1_write_addr = get_write_ptr(in_cb_id);
            for (uint32_t out_elem_i = 0; out_elem_i < out_nelems; ++ out_elem_i) {
                // if (out_w_i * out_elem_i >= out_w) continue; // TODO: Need some guard for the out of bounds when out_w is not multiple of out_nelems

                // kernel_profiler::mark_time(9);

                // start = {start_h, curr_start_w}
                int32_t curr_start_w = start_w + stride_w * out_elem_i;
                int32_t end_h = start_h + window_h;
                int32_t end_w = curr_start_w + window_w;
                int32_t start_h_max = start_h < 0 ? 0 : start_h;
                int32_t start_w_max = curr_start_w < 0 ? 0 : curr_start_w;
                int32_t end_h_min = end_h < in_h ? end_h : in_h;
                int32_t end_w_min = end_w < in_w ? end_w : in_w;

                // read at most window_hw input rows into CB
                uint32_t read_rows = 0;
                uint32_t in_hw_row_id_base = in_w * start_h_max;  // TODO: get rid of *
                uint32_t curr_in_l1_write_addr = in_l1_write_addr;
                for (int32_t h = start_h_max; h < end_h_min; ++ h) {
                    for (int32_t w = start_w_max; w < end_w_min; ++ w) {
                        uint32_t in_hw_row_id = in_hw_row_id_base + w;
                        // uint64_t in_noc_addr = get_noc_addr(in_hw_row_id, s_in);
                        // noc_async_read(in_noc_addr, curr_in_l1_write_addr, in_nbytes_c);
                        s_in.noc_async_read_page(in_hw_row_id, curr_in_l1_write_addr);
                        curr_in_l1_write_addr += in_nbytes_c;
                        ++ read_rows;
                    }
                    in_hw_row_id_base += in_w;
                }
                if (read_rows != window_hw) {
                    // if needed, fill the remainining (window_hw - read_row_id) with -INF
                    fill_with_val(curr_in_l1_write_addr, (window_hw - read_rows) * in_c, 0xff7f);   // TODO: get rid of *
                }
                in_l1_write_addr += in_cb_pagesize;

                // kernel_profiler::mark_time(10);
            }
            noc_async_read_barrier();
            // input for current output index (out_h_i, out_w_i) are ready for this block to be consumed by triscs
            cb_push_back(in_cb_id, out_nelems);
            start_w += stride_w * out_nelems;
        }
        start_h += stride_h;
    }
} // kernel_main()
