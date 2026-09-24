// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for the batched fused-conv variant: one core = value head `head`, users [u0, u0 + nu). Per core once: taps,
// norm weight, z tiles (rows = users), dt_bias / -exp(A) scalars, constants. Per user: 3 packed history slots, the new
// token packed from row b of the projection tile, a/b scalars from row b, per-user selector + mask tiles, the state.
//
// Two schedules in one body (`single` = nu == 1), both bit-identical (pure data movement):
// * nu == 1 (the width-1 decode bucket: 12 cores, one user each): the reads of a stage are issued before the single
//   read barrier of that stage and the CPU work (selector/mask bits, scalar-tile fills) overlaps the reads, so ~7
//   serialized DRAM round trips become 2 (op 48.6 -> ~40 us, served width-1 step -0.4 ms).
// * nu > 1 (widths 2..32: 24..96 cores): the original strictly sequential per-user sequence. The op is DRAM-bound
//   there (96 cores x 128 KB of state per user; a compute-free probe still takes 172 of 200 us); the merged /
//   overlapped schedule and a denser scalar fill both measured slower on the served width-32 step (+0.1..0.8 ms)
//   although faster in isolation. One shared body keeps the NCRISC binary small (a second inlined path was 14 KB
//   vs 6 KB and alone cost ~+0.2 ms on the 96-core launches through the per-launch kernel relay).
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

using namespace gdn_step_df;

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Nk,
    uint32_t Nv,
    uint32_t z_tile0,
    uint32_t ab_page,
    uint32_t dtb_fp32,
    uint32_t nea_fp32,
    uint32_t l2_eps_bits,
    uint32_t norm_eps_bits>
TT_KERNEL void reader(uint32_t head, uint32_t u0, uint32_t nu) {
    const auto qkv_acc = TensorAccessor(tensor::qkv);
    const auto dtb_acc = TensorAccessor(tensor::dtb);
    const auto nea_acc = TensorAccessor(tensor::nea);
    const auto state_acc = TensorAccessor(tensor::state);
    const auto w_acc = TensorAccessor(tensor::weight);
    const auto hist_acc = TensorAccessor(tensor::hist);
    const auto taps_acc = TensorAccessor(tensor::taps);
    DataflowBuffer hist(dfb::hist);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer cur(dfb::cur);
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer a_s(dfb::a_s);
    DataflowBuffer b_s(dfb::b_s);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
    DataflowBuffer mask(dfb::mask);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ct = 2 * Kt + Vt;
    constexpr uint32_t rf = Nv / Nk;
    constexpr bool dtb_is_fp32 = dtb_fp32 != 0;
    constexpr bool nea_is_fp32 = nea_fp32 != 0;
    const uint32_t h = head;
    const uint32_t hk = h / rf;
    const bool single = (nu == 1);
    // `single`: reads of a stage issued together, one barrier per stage; otherwise a barrier after every read
    auto sync = [&]() {
        if (!single) {
            noc.async_read_barrier();
        }
    };
    auto fill = [&](DataflowBuffer& dfb, uint32_t bits) {
        if (single) {
            fill_scalar_tile_fast(dfb, bits);
        } else {
            fill_scalar_tile(dfb, bits);
        }
    };

    // ---- per core: constants, norm weight, taps, z, the two head scalars
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(eps_l2, l2_eps_bits);
    generate_bcast_col_scalar(eps_norm, norm_eps_bits);
    w_in.reserve_back(Vt);
    read_tiles_at(w_acc, w_in, noc, 0, Vt, 0);
    sync();
    taps.reserve_back(4);
    read_tiles_at(taps_acc, taps, noc, h * 4, 4, 0);
    sync();
    z_in.reserve_back(Vt);
    read_tiles_at(qkv_acc, z_in, noc, z_tile0 + h * Vt, Vt, 0);  // all users' rows of this head's z
    sync();
    dtb_s.reserve_back(1);
    noc.async_read(dtb_acc, dtb_s, dtb_is_fp32 ? 4096 : 2048, {.page_id = 0}, {.offset_bytes = 0});
    sync();
    nea_s.reserve_back(1);
    noc.async_read(nea_acc, nea_s, nea_is_fp32 ? 4096 : 2048, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    w_in.push_back(Vt);
    taps.push_back(4);
    z_in.push_back(Vt);
    fill(dtb_s, tile_scalar_bits<dtb_is_fp32>(dtb_s, 0, h));
    dtb_s.push_back(1);
    fill(nea_s, tile_scalar_bits<nea_is_fp32>(nea_s, 0, h));
    nea_s.push_back(1);

    for (uint32_t ui = 0; ui < nu; ++ui) {
        const uint32_t b = u0 + ui;
        const uint32_t bh = b * Nv + h;
        // per-user selector + mask tiles (zero-filled, one 1.0 each); `single` also starts the history / a|b reads
        sel.reserve_back(Ct);
        mask.reserve_back(1);
        if (single) {
            hist.reserve_back(3);
            read_tiles_at(hist_acc, hist, noc, bh * 4 + 1, 3, 0);  // slots 1..3
            a_s.reserve_back(1);
            noc.async_read(qkv_acc, a_s, 2048, {.page_id = ab_page, .offset_bytes = 0}, {.offset_bytes = 0});
            cur.reserve_back(1);
            noc.async_write_zeros(cur, cur.get_entry_size());
        }
        noc.async_write_zeros(sel, Ct * sel.get_entry_size());
        noc.async_write_zeros(mask, mask.get_entry_size());
        noc.write_zeros_l1_barrier();
        set_user_selector_bits(sel, mask, b, Ct);
        if (!single) {
            sel.push_back(Ct);
            mask.push_back(1);
            read_tiles(hist_acc, hist, noc, bh * 4 + 1, 3);  // slots 1..3
            cur.reserve_back(1);
            zero_reserved(cur, noc, 1);
        }
        // the new token packed from row b of the projection tiles (24 face-row segments)
        pack_head_tile_user<Kt, Vt, Nk>(qkv_acc, cur, noc, hk, h, b, 0);
        if (!single) {
            noc.async_read_barrier();
            cur.push_back(1);
            // a[b,h], b[b,h]: row b of the a|b tile, columns h and Nv + h
            a_s.reserve_back(1);
            noc.async_read(qkv_acc, a_s, 2048, {.page_id = ab_page, .offset_bytes = 0}, {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        if (single) {
            hist.push_back(3);
            cur.push_back(1);
            sel.push_back(Ct);
            mask.push_back(1);
            // the 64 KB state read overlaps the scalar-tile fills
            state_in.reserve_back(KV);
            read_tiles_at(state_acc, state_in, noc, bh * KV, KV, 0);
        }
        const uint32_t a_bits = tile_scalar_bits<false>(a_s, b, h);
        const uint32_t b_bits = tile_scalar_bits<false>(a_s, b, Nv + h);
        fill(a_s, a_bits);
        a_s.push_back(1);
        b_s.reserve_back(1);
        fill(b_s, b_bits);
        b_s.push_back(1);
        if (single) {
            noc.async_read_barrier();
            state_in.push_back(KV);
        } else {
            read_tiles(state_acc, state_in, noc, bh * KV, KV);
        }
    }
}
