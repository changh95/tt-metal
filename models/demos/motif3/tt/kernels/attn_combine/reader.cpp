// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Motif-3 fused decode attention combine (D1, docs/OPTIMIZATION_PLAN.md §3.3 C3), reader (RISCV_1).
//
// Output tile j (0 .. n - 1) of the wo input [1, 1, 32, Sg vdim] is signal head h = j / TPH, column tile t = j % TPH.
// Per tile, in the order the compute kernel consumes them:
//   CB_SIG   u tile (h, t)                    = page h TPH + t of u [1, Sg + G, 32, vdim] (the W_UV output)
//   CB_V     v tile j                         sigmoid(lam @ E)
//   CB_NOISE u tile (Sg + h / HPG, t)          the noise head of h's group (what noise @ X copies into column tile j)
//   CB_G     g tile j                         sigmoid gate
//   CB_ACT   active tile j                    0 / 1 lane mask
// Worker q = GX y + x takes tiles [q PER, min(n, (q + 1) PER)).
//
// CT: 0 cb_sig, 1 cb_v, 2 cb_noise, 3 cb_g, 4 cb_act, 5 n, 6 TPH, 7 HPG, 8 Sg, 9 PER, 10 GX,
//     11.. TensorAccessorArgs(u), (v), (g), (act)
// common RT: 0 u_addr, 1 v_addr, 2 g_addr, 3 act_addr

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_sig = get_compile_time_arg_val(0);
    constexpr uint32_t cb_v = get_compile_time_arg_val(1);
    constexpr uint32_t cb_noise = get_compile_time_arg_val(2);
    constexpr uint32_t cb_g = get_compile_time_arg_val(3);
    constexpr uint32_t cb_act = get_compile_time_arg_val(4);
    constexpr uint32_t n = get_compile_time_arg_val(5);
    constexpr uint32_t TPH = get_compile_time_arg_val(6);
    constexpr uint32_t HPG = get_compile_time_arg_val(7);
    constexpr uint32_t Sg = get_compile_time_arg_val(8);
    constexpr uint32_t PER = get_compile_time_arg_val(9);
    constexpr uint32_t GX = get_compile_time_arg_val(10);
    constexpr auto u_args = TensorAccessorArgs<11>();
    constexpr auto v_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr auto g_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto a_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    constexpr uint32_t tb = get_tile_size(cb_sig);
    const auto us = TensorAccessor(u_args, get_common_arg_val<uint32_t>(0), tb);
    const auto vs = TensorAccessor(v_args, get_common_arg_val<uint32_t>(1), tb);
    const auto gs = TensorAccessor(g_args, get_common_arg_val<uint32_t>(2), tb);
    const auto as = TensorAccessor(a_args, get_common_arg_val<uint32_t>(3), tb);

    const uint32_t q = static_cast<uint32_t>(get_absolute_logical_y()) * GX + get_absolute_logical_x();
    const uint32_t j0 = q * PER;
    const uint32_t j1 = (j0 + PER < n) ? j0 + PER : n;
    for (uint32_t j = j0; j < j1; ++j) {
        const uint32_t h = j / TPH;
        const uint32_t t = j % TPH;
        cb_reserve_back(cb_sig, 1);
        cb_reserve_back(cb_v, 1);
        cb_reserve_back(cb_noise, 1);
        noc_async_read_page(h * TPH + t, us, get_write_ptr(cb_sig));
        noc_async_read_page(j, vs, get_write_ptr(cb_v));
        noc_async_read_page((Sg + h / HPG) * TPH + t, us, get_write_ptr(cb_noise));
        cb_reserve_back(cb_g, 1);
        cb_reserve_back(cb_act, 1);
        noc_async_read_page(j, gs, get_write_ptr(cb_g));
        noc_async_read_page(j, as, get_write_ptr(cb_act));
        noc_async_read_barrier();
        cb_push_back(cb_sig, 1);
        cb_push_back(cb_v, 1);
        cb_push_back(cb_noise, 1);
        cb_push_back(cb_g, 1);
        cb_push_back(cb_act, 1);
    }
}
