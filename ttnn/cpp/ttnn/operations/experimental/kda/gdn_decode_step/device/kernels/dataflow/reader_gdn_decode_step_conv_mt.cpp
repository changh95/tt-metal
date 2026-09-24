// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader of the MULTI-TOKEN (speculative-verify) mode of the batched fused-conv GDN decode step (num_tokens = T > 1);
// the one-token step keeps reader_gdn_decode_step_conv.cpp. Same data movement primitives, one core = value head
// `head`, users [u0, u0 + nu).
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"

using namespace gdn_step_df;

// ---- (see the op docstring for the contract). Row grid r = s*T + j (tile r/32, row
// r%32) of qkv (this step) and qkv_prev (last step). Per core once: taps, norm weight, the group's z rows, dt_bias /
// -exp(A), the group's accept counts (-> ctl for the compute kernel), zeroed parity-staging tiles. Per user: the 3
// history slots, then per token (prev rows 1..a_s, cur rows 0..T-1) the packed token tile appended to the user's
// linear history window (parity s & 1, other-parity source rows moved through the staging tiles), the token's
// selector / mask tiles (row r%32) and a|b scalars; the state once after the first token. The window is padded to
// HCAP entries per user so it never wraps inside the 2*HCAP ring. All reads are per-token DRAM row segments.
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
    uint32_t norm_eps_bits,
    uint32_t T,
    uint32_t HCAP,
    uint32_t WT,
    uint32_t ACC_BYTES>
TT_KERNEL void reader(uint32_t head, uint32_t u0, uint32_t nu) {
    const auto qkv_acc = TensorAccessor(tensor::qkv);
    const auto prev_acc = TensorAccessor(tensor::qkv_prev);
    // qkv and qkv_prev have the same spec (validated): one accessor type, so the row-pack code exists once
    static_assert(std::is_same_v<decltype(qkv_acc), decltype(prev_acc)>, "qkv / qkv_prev accessor types differ");
    const auto acc_acc = TensorAccessor(tensor::accept);
    const auto dtb_acc = TensorAccessor(tensor::dtb);
    const auto nea_acc = TensorAccessor(tensor::nea);
    const auto state_acc = TensorAccessor(tensor::state);
    const auto w_acc = TensorAccessor(tensor::weight);
    const auto hist_acc = TensorAccessor(tensor::hist);
    const auto taps_acc = TensorAccessor(tensor::taps);
    DataflowBuffer hist(dfb::hist);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer a_s(dfb::a_s);
    DataflowBuffer b_s(dfb::b_s);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
    DataflowBuffer ctl(dfb::ctl);
    DataflowBuffer stage(dfb::stage);
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

    // ---- per core: constants, norm weight, taps, the two head scalars
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    generate_bcast_col_scalar(eps_l2, l2_eps_bits);
    generate_bcast_col_scalar(eps_norm, norm_eps_bits);
    w_in.reserve_back(Vt);
    read_tiles_at(w_acc, w_in, noc, 0, Vt, 0);
    taps.reserve_back(4);
    read_tiles_at(taps_acc, taps, noc, h * 4, 4, 0);
    dtb_s.reserve_back(1);
    noc.async_read(dtb_acc, dtb_s, dtb_is_fp32 ? 4096 : 2048, {.page_id = 0}, {.offset_bytes = 0});
    nea_s.reserve_back(1);
    noc.async_read(nea_acc, nea_s, nea_is_fp32 ? 4096 : 2048, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    w_in.push_back(Vt);
    taps.push_back(4);
    fill_scalar_tile_fast(dtb_s, tile_scalar_bits<dtb_is_fp32>(dtb_s, 0, h));
    dtb_s.push_back(1);
    fill_scalar_tile_fast(nea_s, tile_scalar_bits<nea_is_fp32>(nea_s, 0, h));
    nea_s.push_back(1);
    // the group's accept counts (clamped to T - 1): kept here and handed to the compute kernel through ctl
    uint32_t accn[32];
    ctl.reserve_back(1);
    load_accept<ACC_BYTES, T>(acc_acc, ctl, noc, u0, nu, accn);
    {
        auto lock = ctl.scoped_write_lock(1);
        auto p = lock.template get_ptr<volatile uint32_t>();
        for (uint32_t i = 0; i < nu; ++i) {
            p[i] = accn[i];
        }
    }
    ctl.push_back(1);
    // z rows of the group's output rows [u0*T, (u0+nu)*T): whole tiles when they lie in one tile row, else the two
    // row spans (the other rows are don't-care: they only meet exact zeros)
    const uint32_t r_lo = u0 * T;
    const uint32_t r_hi = (u0 + nu) * T;
    z_in.reserve_back(Vt);
    if (r_lo / 32 == (r_hi - 1) / 32) {
        read_tiles_at(qkv_acc, z_in, noc, (r_lo / 32) * WT + z_tile0 + h * Vt, Vt, 0);
    } else {
        zero_reserved(z_in, noc, Vt);
        read_rows_span(qkv_acc, z_in, noc, (r_lo / 32) * WT + z_tile0 + h * Vt, Vt, r_lo % 32, 32);
        uint32_t hi = r_hi % 32;
        hi += hi & 1u;
        read_rows_span(qkv_acc, z_in, noc, (r_hi / 32) * WT + z_tile0 + h * Vt, Vt, 0, hi);
    }
    noc.async_read_barrier();
    z_in.push_back(Vt);
    // staging tiles, zeroed once: entries 0 / 1 = parity moves (one per source parity; their other-parity rows are
    // never written), entry 2 = the token's a|b tile. The a_s / b_s scalar tiles are consumed through element (0, 0)
    // only (reduced-lane gates + mul_tiles_bcast_scalar in the compute kernel): both ring entries zeroed once, one word
    // per token.
    stage.reserve_back(3);
    zero_reserved(stage, noc, 3);
    a_s.reserve_back(2);
    zero_reserved(a_s, noc, 2);
    b_s.reserve_back(2);
    zero_reserved(b_s, noc, 2);

    for (uint32_t ui = 0; ui < nu; ++ui) {
        const uint32_t s = u0 + ui;
        const uint32_t par = s & 1u;  // the user's packed-history parity (T=1 convention)
        const uint32_t a = accn[ui];
        const uint32_t ntok = a + T;  // prev rows 1..a, then cur rows 0..T-1
        const uint32_t bh = s * Nv + h;
        // the window starts with history slots 1..3
        hist.reserve_back(3);
        read_tiles_at(hist_acc, hist, noc, bh * 4 + 1, 3, 0);
        noc.async_read_barrier();
        hist.push_back(3);
        for (uint32_t t = 0; t < ntok; ++t) {
            const bool prev = t < a;
            const uint32_t j = prev ? t + 1 : t - a;
            const uint32_t r = s * T + j;
            const uint32_t rr = r % 32;
            const uint32_t page_base = (r / 32) * WT;
            const auto& src = prev ? prev_acc : qkv_acc;
#if defined(GDN_MT_PROBE) && GDN_MT_PROBE == 2
            // probe: hand the compute kernel whatever is in L1 (no DRAM traffic, no fills)
            (void)src;
            (void)page_base;
            hist.reserve_back(1);
            hist.push_back(1);
            sel.reserve_back(Ct);
            sel.push_back(Ct);
            mask.reserve_back(1);
            mask.push_back(1);
            a_s.reserve_back(1);
            a_s.push_back(1);
            b_s.reserve_back(1);
            b_s.push_back(1);
#else
            // the token's packed tile, appended to the window
            hist.reserve_back(1);
            zero_reserved(hist, noc, 1);
            pack_head_tile_row_parity<Kt, Vt, Nk>(src, hist, 0, stage, noc, hk, h, page_base, rr, par);
            hist.push_back(1);
            // selector (packed row 2c + par -> row rr) and mask e_rr
            sel.reserve_back(Ct);
            mask.reserve_back(1);
            noc.async_write_zeros(sel, Ct * sel.get_entry_size());
            noc.async_write_zeros(mask, mask.get_entry_size());
            noc.write_zeros_l1_barrier();
            set_token_selector_bits(sel, mask, rr, par, Ct);
            sel.push_back(Ct);
            mask.push_back(1);
            // a[r, h], b[r, h]: row rr of the tile row's a|b tile, columns h and Nv + h -> element (0, 0) of a_s / b_s
            noc.async_read(
                src,
                stage,
                2048,
                {.page_id = page_base + ab_page, .offset_bytes = 0},
                {.offset_bytes = 2 * stage.get_entry_size()});
            noc.async_read_barrier();
            uint32_t a_bits, b_bits;
            {
                auto lock = stage.scoped_write_lock(3);
                auto p16 = lock.template get_ptr<volatile uint16_t>();
                a_bits = static_cast<uint32_t>(p16[2 * 1024 + tile_elem_index(rr, h)]) << 16;
                b_bits = static_cast<uint32_t>(p16[2 * 1024 + tile_elem_index(rr, Nv + h)]) << 16;
            }
            a_s.reserve_back(1);
            {
                auto lock = a_s.scoped_write_lock(1);
                lock.template get_ptr<volatile uint32_t>()[0] = a_bits;
            }
            a_s.push_back(1);
            b_s.reserve_back(1);
            {
                auto lock = b_s.scoped_write_lock(1);
                lock.template get_ptr<volatile uint32_t>()[0] = b_bits;
            }
            b_s.push_back(1);
#endif
            if (t == 0) {
                read_tiles(state_acc, state_in, noc, bh * KV, KV);  // once per (user, head)
            }
        }
        // pad the user's window to HCAP entries (the compute kernel pops exactly HCAP per user)
        const uint32_t pad = HCAP - 3 - ntok;
        if (pad > 0) {
            hist.reserve_back(pad);
            hist.push_back(pad);
        }
    }
}
