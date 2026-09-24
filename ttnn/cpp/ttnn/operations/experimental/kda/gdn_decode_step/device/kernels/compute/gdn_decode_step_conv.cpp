// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused-conv variant of the GDN decode step, batched: each core handles one value head for a contiguous group of users
// [u0, u0 + nu) (rows of the projection tile). Per user: 4-tap causal conv + SiLU on the packed history/tap tiles (row
// 2c + parity(b) of a packed tile = channel chunk c), scattered into the row-block layout (row b of tile c) with 0/1
// selector tiles, beta = sigmoid(b) and decay = exp(-exp(A) * softplus(a + dt_bias)) from per-user scalar tiles, the
// recurrence with the row mask e_b, and the gated RMSNorm output (row b valid, other rows exactly 0). The users'
// outputs are summed into one accumulator and written once as the group's row span.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/compute/gdn_step_conv_helpers.hpp"

// Phase B (opt-in, NOT bit-identical to the split passes): fold the decay into the key for the read,
// vread = (kn * dec) @ h, and produce each state row as h * dec + kt^T @ delta in ONE DST round (mul then matmul
// accumulating into the same slots), packed straight to hn and hnew -- no hd / outer / hn-copy passes.
#ifndef GDN_DECODE_FUSED_UPDATE
#define GDN_DECODE_FUSED_UPDATE 0
#endif

using namespace gdn_step;
using namespace gdn_step_conv;

namespace {

// conv_p = silu(hist[0]*taps[0] + hist[1]*taps[1] + hist[2]*taps[2] + cur*taps[3])  (one packed fp32 tile)
inline void causal_conv_silu_packed(DataflowBuffer& conv_p) {
    conv_p.reserve_back(1);
    pack_fmt(dfb::conv_p);
    ab_fmt(dfb::hist, dfb::taps);
    mul_init(dfb::hist, dfb::taps);
    tile_regs_acquire();
    mul_tiles(dfb::hist, dfb::taps, 0, 0, 0);
    mul_tiles(dfb::hist, dfb::taps, 1, 1, 1);
    mul_tiles(dfb::hist, dfb::taps, 2, 2, 2);
    mul_tiles(dfb::cur, dfb::taps, 0, 3, 3);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    add_binary_tile(0, 2, 0);
    add_binary_tile(0, 3, 0);
    silu_tile_init();
    silu_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::conv_p, 0);
    tile_regs_release();
    conv_p.push_back(1);
}

#if GDN_DECODE_FUSED_UPDATE
// hn[i*Vt + j] = hnew[i*Vt + j] = state_in[i*Vt + j] * dec + kt[i] @ delta[j], one DST round per state row i.
// The eltwise mul accumulates into the (packer-cleared) DST slot, the K=1 matmul then accumulates on top of it.
template <uint32_t Kt, uint32_t Vt>
inline void fused_state_update(DataflowBuffer& hn, DataflowBuffer& hnew) {
    static_assert(Vt <= kDstTiles, "one state row must fit a DST half");
    hn.reserve_back(Kt * Vt);
    hnew.reserve_back(Kt * Vt);
    for (uint32_t i = 0; i < Kt; ++i) {
        pack_fmt(dfb::hn);
        ab_fmt(dfb::state_in, dfb::dec);
        mul_init(dfb::state_in, dfb::dec);
        tile_regs_acquire();
        for (uint32_t j = 0; j < Vt; ++j) {
            mul_tiles(dfb::state_in, dfb::dec, i * Vt + j, 0, j);
        }
        ab_fmt<SrcOrder::Reverse>(dfb::kt, dfb::delta);  // all fp32: no format change, only the op init
        matmul_init(dfb::kt, dfb::delta);
        for (uint32_t j = 0; j < Vt; ++j) {
            matmul_tiles(dfb::kt, dfb::delta, i, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < Vt; ++j) {
            pack_tile(j, dfb::hn, i * Vt + j);
        }
        pack_fmt(dfb::hnew);
        for (uint32_t j = 0; j < Vt; ++j) {
            pack_tile(j, dfb::hnew, i * Vt + j);
        }
        tile_regs_release();
    }
    hn.push_back(Kt * Vt);
    hnew.push_back(Kt * Vt);
}
#endif

}  // namespace

template <uint32_t Kt, uint32_t Vt, uint32_t scale_bits, uint32_t inv_dv_bits>
TT_KERNEL void compute(uint32_t nu) {
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ct = 2 * Kt + Vt;
    constexpr uint32_t one_bits = kOneBits;
    DataflowBuffer hist(dfb::hist);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer cur(dfb::cur);
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer conv_p(dfb::conv_p);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer a_s(dfb::a_s);
    DataflowBuffer b_s(dfb::b_s);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
    DataflowBuffer mask(dfb::mask);
    DataflowBuffer qc(dfb::qc);
    DataflowBuffer kc(dfb::kc);
    DataflowBuffer vc(dfb::vc);
    DataflowBuffer beta_t(dfb::beta_t);
    DataflowBuffer zs(dfb::zs);
    DataflowBuffer tmp(dfb::tmp);
    DataflowBuffer stats(dfb::stats);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer inv(dfb::inv);
    DataflowBuffer qn(dfb::qn);
    DataflowBuffer kn(dfb::kn);
    DataflowBuffer vm(dfb::vm);
    DataflowBuffer dec(dfb::dec);
    DataflowBuffer hd(dfb::hd);
    DataflowBuffer vread(dfb::vread);
    DataflowBuffer delta(dfb::delta);
    DataflowBuffer kt(dfb::kt);
    DataflowBuffer outer(dfb::outer);
    DataflowBuffer hn(dfb::hn);
    DataflowBuffer hnew(dfb::hnew);
    DataflowBuffer o(dfb::o);
    DataflowBuffer on(dfb::on);
    DataflowBuffer out_acc(dfb::out_acc);
    DataflowBuffer acc2(dfb::acc2);
    DataflowBuffer out(dfb::out);

    compute_kernel_hw_startup(dfb::hist, dfb::state_in, dfb::out);
    note_formats(dfb::hist, dfb::state_in, dfb::out);
    scaler.wait_front(1);
    eps_l2.wait_front(1);
    eps_norm.wait_front(1);
    w_in.wait_front(Vt);
    taps.wait_front(4);
    dtb_s.wait_front(1);
    nea_s.wait_front(1);
    // zs = silu(z) for all users of this head (rows = users), once per core
    z_in.wait_front(Vt);
    silu_tiles(dfb::z_in, dfb::zs, zs, Vt);
    z_in.pop_front(Vt);
    zs.wait_front(Vt);

    for (uint32_t ui = 0; ui < nu; ++ui) {
        sel.wait_front(Ct);  // per-user selectors (row b <- packed row 2c + parity(b))
        mask.wait_front(1);  // per-user row mask e_b
        hist.wait_front(3);
        cur.wait_front(1);
        a_s.wait_front(1);
        b_s.wait_front(1);
        state_in.wait_front(KV);

        // conv + silu on the packed tile, scattered into qc, kc, vc ; gates
        causal_conv_silu_packed(conv_p);
        hist.pop_front(3);
        cur.pop_front(1);
        conv_p.wait_front(1);
        scatter_conv<Kt, Vt>(qc, kc, vc);
        conv_p.pop_front(1);
        sel.pop_front(Ct);
        gate_beta(beta_t);
        b_s.pop_front(1);
        gate_decay(dec);
        a_s.pop_front(1);
        qc.wait_front(Kt);
        kc.wait_front(Kt);
        vc.wait_front(Vt);

        // qn = l2norm(q) * scale, kn = l2norm(k)  (rows other than b -> 0 through the mask)
        square_tiles(dfb::qc, dfb::tmp, tmp, Kt);
        row_sum<dfb::tmp, dfb::scaler, dfb::stats>(Kt);
        stats.wait_front(1);
        inverse_l2(dfb::stats, dfb::eps_l2, dfb::mask, dfb::scratch, scratch, dfb::inv, inv, scale_bits);
        stats.pop_front(1);
        inv.wait_front(1);
        scale_rows(dfb::qc, dfb::inv, dfb::qn, qn, Kt);
        inv.pop_front(1);
        qc.pop_front(Kt);
        square_tiles(dfb::kc, dfb::tmp, tmp, Kt);
        row_sum<dfb::tmp, dfb::scaler, dfb::stats>(Kt);
        stats.wait_front(1);
        inverse_l2(dfb::stats, dfb::eps_l2, dfb::mask, dfb::scratch, scratch, dfb::inv, inv, one_bits);
        stats.pop_front(1);
        inv.wait_front(1);
        scale_rows(dfb::kc, dfb::inv, dfb::kn, kn, Kt);
        inv.pop_front(1);
        kc.pop_front(Kt);
        scale_rows(dfb::vc, dfb::mask, dfb::vm, vm, Vt);  // vm = v masked to row b
        vc.pop_front(Vt);

        dec.wait_front(1);
        kn.wait_front(Kt);
#if GDN_DECODE_FUSED_UPDATE
        // kd = kn * dec (staged in hd's slots) ; vread = kd @ h
        multiply_tiles<true>(dfb::kn, dfb::dec, dfb::hd, hd, Kt);
        hd.wait_front(Kt);
        row_times_matrix(dfb::hd, dfb::state_in, dfb::vread, vread, Kt, Vt);
        hd.pop_front(Kt);
#else
        // hd = h * decay ; vread = kn @ hd
        multiply_tiles<true>(dfb::state_in, dfb::dec, dfb::hd, hd, KV);
        dec.pop_front(1);
        state_in.pop_front(KV);
        hd.wait_front(KV);
        row_times_matrix(dfb::kn, dfb::hd, dfb::vread, vread, Kt, Vt);
#endif
        // delta = beta * (vm - vread)
        vread.wait_front(Vt);
        vm.wait_front(Vt);
        subtract_tiles(dfb::vm, dfb::vread, dfb::tmp, tmp, Vt);
        vread.pop_front(Vt);
        vm.pop_front(Vt);
        tmp.wait_front(Vt);
        beta_t.wait_front(1);
        multiply_tiles<true>(dfb::tmp, dfb::beta_t, dfb::delta, delta, Vt);
        tmp.pop_front(Vt);
        beta_t.pop_front(1);
        delta.wait_front(Vt);

        // hn = hd + kn^T @ delta ; hnew (writer copy)
        transpose_tiles(dfb::kn, dfb::kt, kt, Kt);
        kn.pop_front(Kt);
        kt.wait_front(Kt);
#if GDN_DECODE_FUSED_UPDATE
        fused_state_update<Kt, Vt>(hn, hnew);
        kt.pop_front(Kt);
        delta.pop_front(Vt);
        dec.pop_front(1);
        state_in.pop_front(KV);
        hn.wait_front(KV);
#else
        outer_product(dfb::kt, dfb::delta, dfb::outer, outer, Kt, Vt);
        kt.pop_front(Kt);
        delta.pop_front(Vt);
        outer.wait_front(KV);
        add_tiles_n(dfb::hd, dfb::outer, dfb::hn, hn, KV);
        hd.pop_front(KV);
        outer.pop_front(KV);
        hn.wait_front(KV);
        copy_tiles(dfb::hn, dfb::hnew, hnew, KV);
#endif

        // o = qn @ hn
        qn.wait_front(Kt);
        row_times_matrix(dfb::qn, dfb::hn, dfb::o, o, Kt, Vt);
        qn.pop_front(Kt);
        hn.pop_front(KV);
        o.wait_front(Vt);

        // gated = rmsnorm(o) * w * silu(z) for this user's row; accumulate over the group's users
        square_tiles(dfb::o, dfb::tmp, tmp, Vt);
        row_sum<dfb::tmp, dfb::scaler, dfb::stats>(Vt);
        stats.wait_front(1);
        inverse_rms(dfb::stats, dfb::eps_norm, dfb::scratch, scratch, dfb::inv, inv, inv_dv_bits);
        stats.pop_front(1);
        inv.wait_front(1);
        scale_rows(dfb::o, dfb::inv, dfb::on, on, Vt);
        inv.pop_front(1);
        o.pop_front(Vt);
        on.wait_front(Vt);
        gated_user(tmp, Vt);
        on.pop_front(Vt);
        mask.pop_front(1);
        tmp.wait_front(Vt);
        if (ui == 0) {
            copy_tiles(dfb::tmp, dfb::out_acc, out_acc, Vt);
        } else {
            out_acc.wait_front(Vt);
            add_tiles_n(dfb::out_acc, dfb::tmp, dfb::acc2, acc2, Vt);
            out_acc.pop_front(Vt);
            acc2.wait_front(Vt);
            copy_tiles(dfb::acc2, dfb::out_acc, out_acc, Vt);
            acc2.pop_front(Vt);
        }
        tmp.pop_front(Vt);
    }
    // the group's rows go out once
    out_acc.wait_front(Vt);
    copy_tiles(dfb::out_acc, dfb::out, out, Vt);
    out_acc.pop_front(Vt);
    zs.pop_front(Vt);
}
