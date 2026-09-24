// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer of the MULTI-TOKEN (speculative-verify) mode of the batched fused-conv GDN decode step (num_tokens = T > 1);
// the one-token step keeps writer_gdn_decode_step_conv.cpp.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"

using namespace gdn_step_df;

namespace {
template <typename Accessor>
inline void write_tiles(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count) {
    dfb.wait_front(count);
    const uint32_t entry = dfb.get_entry_size();
    for (uint32_t t = 0; t < count; ++t) {
        noc.async_write(dfb, acc, entry, {.offset_bytes = t * entry}, {.page_id = first_page + t});
    }
    noc.async_write_barrier();
    dfb.pop_front(count);
}
}  // namespace

// ---- (see the op docstring for the contract). Per user: assemble the committed history = the last 4 rows of
// [slot1, slot2, slot3, prev rows 1..a_s, cur row 0] (packed at the user's parity s & 1, rows of the other parity
// through the staging tiles), then after the compute's state output (the state after the commit tokens; it implies
// this core's reader consumed the old slots) write the state and the 4 slots once. After the last user: the group's
// output rows [u0*T, (u0+nu)*T) of the head's tiles (one or two tile rows), and on the head's last group the zero
// padding rows [B*T, RP).
template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Nk,
    uint32_t Nv,
    uint32_t T,
    uint32_t WT,
    uint32_t OWT,
    uint32_t BT,
    uint32_t RP,
    uint32_t ACC_BYTES>
TT_KERNEL void writer(uint32_t head, uint32_t u0, uint32_t nu, uint32_t zero_pad) {
    const auto state_acc = TensorAccessor(tensor::state_out);
    const auto out_acc = TensorAccessor(tensor::out);
    const auto qkv_acc = TensorAccessor(tensor::qkv_w);
    const auto prev_acc = TensorAccessor(tensor::qkv_prev_w);
    static_assert(std::is_same_v<decltype(qkv_acc), decltype(prev_acc)>, "qkv / qkv_prev accessor types differ");
    const auto hist_acc = TensorAccessor(tensor::hist_w);
    const auto acc_acc = TensorAccessor(tensor::accept_w);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer out(dfb::out);
    DataflowBuffer wshift(dfb::wshift);
    DataflowBuffer wstage(dfb::wstage);
    DataflowBuffer wctl(dfb::wctl);
    DataflowBuffer zero_t(dfb::zero_t);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t rf = Nv / Nk;
    const uint32_t h = head;
    const uint32_t hk = h / rf;
    uint32_t accn[32];
    wctl.reserve_back(1);
    load_accept<ACC_BYTES, T>(acc_acc, wctl, noc, u0, nu, accn);
    wstage.reserve_back(2);
    zero_reserved(wstage, noc, 2);
    zero_t.reserve_back(1);  // zero source tile for the output padding rows (read ptr == write ptr: never pushed)
    zero_reserved(zero_t, noc, 1);
    for (uint32_t ui = 0; ui < nu; ++ui) {
        const uint32_t s = u0 + ui;
        const uint32_t par = s & 1u;
        const uint32_t a = accn[ui];
        const uint32_t bh = s * Nv + h;
        wshift.reserve_back(4);
        zero_reserved(wshift, noc, 4);
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t m = a + i;  // index into [slot1, slot2, slot3, prev 1..a, cur 0] (length 4 + a)
            if (m < 3) {
                read_tiles_at(hist_acc, wshift, noc, bh * 4 + m + 1, 1, i);
            } else {
                const bool prev = m < 3 + a;
                const uint32_t r = s * T + (prev ? m - 2 : 0);
                pack_head_tile_row_parity<Kt, Vt, Nk>(
                    prev ? prev_acc : qkv_acc, wshift, i, wstage, noc, hk, h, (r / 32) * WT, r % 32, par);
            }
        }
        noc.async_read_barrier();
        wshift.push_back(4);
        write_tiles(state_acc, hnew, noc, bh * KV, KV);  // waits for compute -> old history already consumed
        write_tiles(hist_acc, wshift, noc, bh * 4, 4);
    }
    // the group's output rows (L1 row r % 32 <-> tensor row r); an odd row count is rounded up by one zero row, which
    // only happens on the last group (a full group has an even number of users) where that row is padding
    out.wait_front(Vt);
    const uint32_t r_lo = u0 * T;
    const uint32_t r_hi = (u0 + nu) * T;
    auto piece = [&](uint32_t tr, uint32_t lo, uint32_t hi) {
        hi += hi & 1u;
        write_rows_span(out_acc, out, noc, tr * OWT + h * Vt, Vt, lo, hi);
    };
    if (r_lo / 32 == (r_hi - 1) / 32) {
        piece(r_lo / 32, r_lo % 32, r_hi - (r_lo / 32) * 32);
    } else {
        piece(r_lo / 32, r_lo % 32, 32);
        piece(r_hi / 32, 0, r_hi % 32);
    }
    if (zero_pad != 0) {
        const uint32_t entry = zero_t.get_entry_size();
        for (uint32_t r = BT + (BT & 1u); r < RP;) {
            const uint32_t tr = r / 32;
            const uint32_t lo = r % 32;
            if (lo == 0) {
                for (uint32_t t = 0; t < Vt; ++t) {
                    noc.async_write(zero_t, out_acc, entry, {.offset_bytes = 0}, {.page_id = tr * OWT + h * Vt + t});
                }
            } else {
                write_rows_span(out_acc, zero_t, noc, tr * OWT + h * Vt, Vt, lo, 32, /*same_src=*/true);
            }
            r = (tr + 1) * 32;
        }
    }
    noc.async_write_barrier();
    out.pop_front(Vt);
}
