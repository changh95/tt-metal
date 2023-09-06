// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {
void MAIN {

    constexpr uint32_t onetile = 1;

    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t transpose_hw = get_compile_time_arg_val(4);

    constexpr uint32_t cb_intermed0 = 24;
    constexpr uint32_t cb_intermed1 = 25;
    constexpr uint32_t cb_intermed2 = 26;
    constexpr uint32_t out_cb_id = 16;

    constexpr uint32_t num_rows_in_one_tile = 32;

    mm_init(tt::CB::c_in0, tt::CB::c_in1, out_cb_id, transpose_hw);

    for (uint32_t nb = 0; nb < batch; nb++)
    for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) // output tile of C
    for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) // output tile index of C
    {
        for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; tile_row_id++) {
            acquire_dst(tt::DstMode::Half);
            for (uint32_t kt = 0; kt < Kt; kt++) {
                if (tile_row_id == 0) {
                    cb_wait_front(tt::CB::c_in0, kt+1);
                }
                cb_wait_front(tt::CB::c_in1, onetile);

                matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, kt, 0, 0, transpose_hw);

                cb_pop_front(tt::CB::c_in1, onetile);
            }

            cb_reserve_back(cb_intermed0, onetile);
            pack_tile(0, cb_intermed0);
            release_dst(tt::DstMode::Half);
            cb_push_back(cb_intermed0, onetile);

            // untilize tile and write to CB::c_intermed1
            cb_wait_front(cb_intermed0, onetile);
            untilize_init_short(cb_intermed0);
            cb_reserve_back(cb_intermed1, 1);
            untilize_block(cb_intermed0, 1, cb_intermed1);
            cb_push_back(cb_intermed1, 1);

            cb_pop_front(cb_intermed0, 1);
            untilize_uninit(cb_intermed0);

            mm_init_short(transpose_hw);
        }
        cb_pop_front(tt::CB::c_in0, Kt);

        // cb_intermed2 comes from reader; untilized row-major tile
        cb_wait_front(cb_intermed2, 1);
        cb_reserve_back(tt::CB::c_out0, onetile);

        // tilize CB::intermed2 and write to CB::c_out0
        tilize_init_short(cb_intermed2, 1);
        tilize_block(cb_intermed2, 1, out_cb_id);
        cb_push_back(out_cb_id, 1);

        cb_pop_front(cb_intermed2, 1);
        tilize_uninit();

        mm_init_short(transpose_hw);
    }

}
} // NAMESPACE
