// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MULTI-TOKEN (speculative-verify) mode of the batched fused-conv GDN decode step (num_tokens = T > 1); the one-token
// step keeps gdn_decode_step_conv.cpp. One core = one value head x a contiguous user group. Per user s with
// a = accept[s] (from ctl): tokens t = 0..a+T-1 = prev rows 1..a, cur row 0 (both COMMIT), cur rows 1..T-1 (DRAFT, L1
// only). Every token runs exactly the one-token kernel's per-token op sequence (conv window t, scatter, gates, norms,
// hd = state*dec, vread, delta, kt, outer, hn = hd + outer) with the state chained in L1 (state_in for t = 0, then hn);
// the writer's state copy (hnew) is produced once, after cur row 0; the gated-norm output row (row r%32 of the group's
// tile block) is produced for the cur rows only and accumulated into out_acc like the one-token users' rows. The
// split state update is kept (GDN_DECODE_FUSED_UPDATE does not apply here). Bit-identical per token to the T=1 kernel.
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/compute/gdn_step_conv_helpers.hpp"

using namespace gdn_step;
using namespace gdn_step_conv;

namespace {

// Code size: the one-token kernel inlines every tile helper at each call site (about 1 KB per site per TRISC) and just
// fits the 70656 B kernel config buffer together with its reader / writer; the multi-token kernel adds the
// accept-driven control flow, so the helpers used at several sites go through these NOINLINE wrappers (one copy each;
// the LLK immediates inside stay compile-time constants). Same instruction sequences per call.
#define GDN_NOINLINE __attribute__((noinline))
GDN_NOINLINE void square_tiles_1(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    square_tiles(a, out, out_dfb, n);
}
template <uint32_t in_cb, uint32_t scaler_cb, uint32_t out_cb>
GDN_NOINLINE void row_sum_1(uint32_t Wt) {
    row_sum<in_cb, scaler_cb, out_cb>(Wt);
}
GDN_NOINLINE void scale_rows_1(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    scale_rows(a, b, out, out_dfb, n);
}
GDN_NOINLINE void copy_tiles_1(uint32_t a, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    copy_tiles(a, out, out_dfb, n);
}
GDN_NOINLINE void add_tiles_1(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    add_tiles_n(a, b, out, out_dfb, n);
}
GDN_NOINLINE void row_times_matrix_1(
    uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t Kt, uint32_t Nt) {
    row_times_matrix(a, b, out, out_dfb, Kt, Nt);
}

// conv_p = silu(w[0]*taps[0] + w[1]*taps[1] + w[2]*taps[2] + w[3]*taps[3]) for window w = hist[t .. t+3] of the user's
// linear history [slot1, slot2, slot3, tok0, tok1, ...] (the one-token kernel's conv with a shifted tile index).
inline void causal_conv_silu_window(DataflowBuffer& conv_p, uint32_t t) {
    conv_p.reserve_back(1);
    pack_fmt(dfb::conv_p);
    ab_fmt(dfb::hist, dfb::taps);
    mul_init(dfb::hist, dfb::taps);
    tile_regs_acquire();
    mul_tiles(dfb::hist, dfb::taps, t, 0, 0);
    mul_tiles(dfb::hist, dfb::taps, t + 1, 1, 1);
    mul_tiles(dfb::hist, dfb::taps, t + 2, 2, 2);
    mul_tiles(dfb::hist, dfb::taps, t + 3, 3, 3);
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

// ---- reduced-lane SFPU variants -----------------------------------------------------------------------------------
// The SFPU computes every lane independently, so processing fewer lanes leaves the consumed lanes bit-identical to the
// full-tile ops of the one-token kernel. The gate tiles (a | b scalars, dt_bias, -exp(A)) are all-equal and are only
// consumed through element (0, 0) (mul_tiles_bcast_scalar), so sigmoid / softplus / exp / the SFPU add and mul run on
// one 32-lane group of faces 0 / 1 (rows 0..1) instead of 1024 lanes; the norm inverses are only consumed through
// column 0 (mul_tiles_bcast_cols), so rsqrt / mul_unary run on the column faces 0 / 2 (VectorMode::C). Unprocessed
// lanes keep their (finite) inputs; they are never read.
constexpr int kLaneIters = 1;  // 32 lanes = rows 0..1 of a face

inline void sigmoid_00(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sigmoid, (false, DST_ACCUM_MODE, kLaneIters), idst, VectorMode::R));
}
inline void softplus_00(uint32_t idst, uint32_t beta, uint32_t beta_reciprocal, uint32_t threshold) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_softplus,
        (APPROX, DST_ACCUM_MODE, kLaneIters),
        idst,
        VectorMode::R,
        beta,
        beta_reciprocal,
        threshold));
}
inline void exp_00(uint32_t idst) {  // exp_tile<false>: no approx, no scale, clamp negative
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_exponential,
        (false, DST_ACCUM_MODE, false, kLaneIters, true),
        idst,
        VectorMode::R,
        p_sfpu::kCONST_1_FP16B));
}
inline void add_binary_00(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::ADD, kLaneIters, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
        idst0,
        idst1,
        odst,
        VectorMode::R)));
}
inline void mul_binary_00(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (APPROX, ckernel::BinaryOp::MUL, kLaneIters, DST_ACCUM_MODE),
        idst0,
        idst1,
        odst,
        VectorMode::R)));
}
inline void rsqrt_col(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROX, 8, DST_ACCUM_MODE, false, false),
        idst,
        VectorMode::C));
}
inline void mul_unary_col(uint32_t idst, uint32_t param1) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, MUL_UNARY, 8, DST_ACCUM_MODE),
        idst,
        VectorMode::C,
        param1));
}

// beta_t(0,0) = sigmoid(b)
inline void gate_beta_00(DataflowBuffer& beta_t) {
    beta_t.reserve_back(1);
    pack_fmt(dfb::beta_t);
    srca_fmt(dfb::b_s);
    copy_init(dfb::b_s);
    tile_regs_acquire();
    copy_tile(dfb::b_s, 0, 0);
    sigmoid_tile_init();
    sigmoid_00(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::beta_t, 0);
    tile_regs_release();
    beta_t.push_back(1);
}

// dec(0,0) = exp(neg_exp_A * softplus(a + dt_bias))
inline void gate_decay_00(DataflowBuffer& dec) {
    dec.reserve_back(1);
    pack_fmt(dfb::dec);
    srca_fmt(dfb::a_s);
    copy_init(dfb::a_s);
    tile_regs_acquire();
    copy_tile(dfb::a_s, 0, 0);
    copy_tile(dfb::dtb_s, 0, 1);
    add_binary_tile_init();
    add_binary_00(0, 1, 0);
    softplus_tile_init();
    softplus_00(0, kOneBits, kOneBits, kTwentyBits);
    copy_init(dfb::nea_s);
    copy_tile(dfb::nea_s, 0, 2);
    mul_binary_tile_init();
    mul_binary_00(0, 2, 0);
    exp_tile_init<false>();
    exp_00(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::dec, 0);
    tile_regs_release();
    dec.push_back(1);
}

// out[i] = a[i] * b[0](0,0)  (the one-token kernel's multiply_tiles<true> with an all-equal b: same FPU multiply)
GDN_NOINLINE void multiply_scalar(uint32_t a, uint32_t b, uint32_t out, DataflowBuffer& out_dfb, uint32_t n) {
    out_dfb.reserve_back(n);
    pack_fmt(out);
    ab_fmt(a, b);
    mul_bcast_scalar_init(a, b);
    dst_rounds(n, out, [&](uint32_t i, uint32_t d) { mul_tiles_bcast_scalar(a, b, i, 0, d); });
    out_dfb.push_back(n);
}

// inverse_l2 with the SFPU on column 0 only: inv = (rsqrt(stats + eps) * post) * mask. NOTE the mask multiply is NOT a
// no-op numerically: an FPU multiply of fp32 operands (HiFi4, bf16-chunked) rounds its operand, and the one-token
// kernel's bits include that rounding -- the pass must stay (same for vm = v * mask below).
GDN_NOINLINE void inverse_l2_col(
    uint32_t stats,
    uint32_t eps,
    uint32_t mask,
    uint32_t scratch,
    DataflowBuffer& scratch_dfb,
    uint32_t inv,
    DataflowBuffer& inv_dfb,
    uint32_t post_scale_bits) {
    scratch_dfb.reserve_back(1);
    pack_fmt(scratch);
    ab_fmt(stats, eps);
    add_init(stats, eps);
    tile_regs_acquire();
    add_tiles(stats, eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_col(0);
    binop_with_scalar_tile_init();
    mul_unary_col(0, post_scale_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch, 0);
    tile_regs_release();
    scratch_dfb.push_back(1);
    scratch_dfb.wait_front(1);
    inv_dfb.reserve_back(1);
    pack_fmt(inv);
    ab_fmt(scratch, mask);
    mul_init(scratch, mask);
    tile_regs_acquire();
    mul_tiles(scratch, mask, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, inv, 0);
    tile_regs_release();
    inv_dfb.push_back(1);
    scratch_dfb.pop_front(1);
}

// inverse_rms with the SFPU on column 0 only: inv = rsqrt(stats * inv_n + eps)
inline void inverse_rms_col(
    uint32_t stats,
    uint32_t eps,
    uint32_t scratch,
    DataflowBuffer& scratch_dfb,
    uint32_t inv,
    DataflowBuffer& inv_dfb,
    uint32_t inv_n_bits) {
    scratch_dfb.reserve_back(1);
    pack_fmt(scratch);
    srca_fmt(stats);
    copy_init(stats);
    tile_regs_acquire();
    copy_tile(stats, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_col(0, inv_n_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, scratch, 0);
    tile_regs_release();
    scratch_dfb.push_back(1);
    scratch_dfb.wait_front(1);
    inv_dfb.reserve_back(1);
    pack_fmt(inv);
    ab_fmt(scratch, eps);
    add_init(scratch, eps);
    tile_regs_acquire();
    add_tiles(scratch, eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_col(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, inv, 0);
    tile_regs_release();
    inv_dfb.push_back(1);
    scratch_dfb.pop_front(1);
}

// The stages of the per-token sequence that run conditionally (cur rows only / the commit token / the first output
// row of the group). Each is a separate NOINLINE function: fully inlined into the kernel body, GCC merged their LLK
// init sequences with the surrounding unconditional code (code hoisting across the branch) and the address-mod /
// config immediates of the merged asm became variables ("impossible constraint in 'asm'" on the MATH thread for
// T >= 4). Inside each function every operand is a compile-time constant again; the op sequences are unchanged.

// qn = l2norm(q) * scale (rows other than the token's -> 0 through the mask); consumes stats / inv, leaves qc to the
// caller
template <uint32_t Kt, uint32_t scale_bits>
__attribute__((noinline)) void q_norm_stage(
    DataflowBuffer& tmp, DataflowBuffer& stats, DataflowBuffer& scratch, DataflowBuffer& inv, DataflowBuffer& qn) {
    square_tiles_1(dfb::qc, dfb::tmp, tmp, Kt);
    row_sum_1<dfb::tmp, dfb::scaler, dfb::stats>(Kt);
    stats.wait_front(1);
    inverse_l2_col(dfb::stats, dfb::eps_l2, dfb::mask, dfb::scratch, scratch, dfb::inv, inv, scale_bits);
    stats.pop_front(1);
    inv.wait_front(1);
    scale_rows_1(dfb::qc, dfb::inv, dfb::qn, qn, Kt);
    inv.pop_front(1);
}

// hnew = hn (the writer's copy of the committed state)
template <uint32_t KV>
__attribute__((noinline)) void commit_stage(DataflowBuffer& hnew) {
    copy_tiles_1(dfb::hn, dfb::hnew, hnew, KV);
}

// o = qn @ hn ; gated = rmsnorm(o) * w * silu(z) in the token's row -> tmp (Vt tiles, waited)
template <uint32_t Kt, uint32_t Vt, uint32_t inv_dv_bits>
__attribute__((noinline)) void output_stage(
    DataflowBuffer& qn,
    DataflowBuffer& o,
    DataflowBuffer& tmp,
    DataflowBuffer& stats,
    DataflowBuffer& scratch,
    DataflowBuffer& inv,
    DataflowBuffer& on) {
    qn.wait_front(Kt);
    row_times_matrix_1(dfb::qn, dfb::hn, dfb::o, o, Kt, Vt);
    qn.pop_front(Kt);
    o.wait_front(Vt);
    square_tiles_1(dfb::o, dfb::tmp, tmp, Vt);
    row_sum_1<dfb::tmp, dfb::scaler, dfb::stats>(Vt);
    stats.wait_front(1);
    inverse_rms_col(dfb::stats, dfb::eps_norm, dfb::scratch, scratch, dfb::inv, inv, inv_dv_bits);
    stats.pop_front(1);
    inv.wait_front(1);
    scale_rows_1(dfb::o, dfb::inv, dfb::on, on, Vt);
    inv.pop_front(1);
    o.pop_front(Vt);
    on.wait_front(Vt);
    gated_user(tmp, Vt);
    on.pop_front(Vt);
    tmp.wait_front(Vt);
}

// acc_cur = tmp (the group's first output row) ...
template <uint32_t Vt>
__attribute__((noinline)) void accumulate_first(uint32_t acc_cur, DataflowBuffer& acc_cur_dfb) {
    copy_tiles_1(dfb::tmp, acc_cur, acc_cur_dfb, Vt);
}

// ... acc_nxt = acc_cur + tmp (every further output row lands in its own, so far zero, row); the accumulator ping-pongs
// between out_acc and acc2 instead of being copied back (the one-token kernel's copy is exact, so the bits agree)
template <uint32_t Vt>
__attribute__((noinline)) void accumulate_next(
    uint32_t acc_cur, DataflowBuffer& acc_cur_dfb, uint32_t acc_nxt, DataflowBuffer& acc_nxt_dfb) {
    acc_cur_dfb.wait_front(Vt);
    add_tiles_1(acc_cur, dfb::tmp, acc_nxt, acc_nxt_dfb, Vt);
    acc_cur_dfb.pop_front(Vt);
}

#ifndef GDN_MT_PROBE
#define GDN_MT_PROBE 0
#endif
// timing-only section probes (bitmask; numerics garbage): 1 = no math at all, 4 = skip conv/scatter/gates/k-norm,
// 8 = skip the state update (hd, vread, delta, kt, outer, hn), 16 = skip the output stage
constexpr uint32_t kProbe = GDN_MT_PROBE;
inline void fake_produce(DataflowBuffer& d, uint32_t n) {
    d.reserve_back(n);
    d.push_back(n);
}

}  // namespace

// ---- (see the op docstring for the contract). Per user s with a = accept[s] (from
// ctl): tokens t = 0..a+T-1 = prev rows 1..a, cur row 0 (both COMMIT), cur rows 1..T-1 (DRAFT, L1 only). Every token
// runs exactly the T=1 per-token op sequence (conv window t, scatter, gates, norms, hd = state*dec, vread, delta, kt,
// outer, hn = hd + outer) with the state chained in L1 (state_in for t = 0, then hn); the writer's state copy (hnew) is
// produced once, after cur row 0; the gated-norm output row (its row r%32 in the group's tile block) is produced for
// the cur rows only and accumulated into out_acc like the T=1 users' rows. Bit-identical per token to the T=1 kernel.
template <uint32_t Kt, uint32_t Vt, uint32_t scale_bits, uint32_t inv_dv_bits, uint32_t T, uint32_t HCAP>
TT_KERNEL void compute(uint32_t nu) {
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ct = 2 * Kt + Vt;
    constexpr uint32_t one_bits = kOneBits;
    DataflowBuffer hist(dfb::hist);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer conv_p(dfb::conv_p);
    DataflowBuffer z_in(dfb::z_in);
    DataflowBuffer a_s(dfb::a_s);
    DataflowBuffer b_s(dfb::b_s);
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
    DataflowBuffer ctl(dfb::ctl);
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
    // zs = silu(z) for the group's output rows, once per core
    z_in.wait_front(Vt);
    silu_tiles(dfb::z_in, dfb::zs, zs, Vt);
    z_in.pop_front(Vt);
    zs.wait_front(Vt);
    ctl.wait_front(1);  // the group's accept counts (UNPACK reads L1, mailbox-broadcast to MATH / PACK)

    bool first_out = true;
    bool acc_in_out = true;  // which of out_acc / acc2 holds the group's accumulated output rows
    for (uint32_t ui = 0; ui < nu; ++ui) {
        uint32_t a = ctl.read_tile_value<uint32_t>(0, ui);
        if (a > T - 1) {
            a = T - 1;
        }
        const uint32_t ntok = a + T;
        for (uint32_t t = 0; t < ntok; ++t) {
            const bool emit = t >= a;    // cur rows produce an output row
            const bool commit = t == a;  // the state after cur row 0 goes to the writer
            hist.wait_front(4 + t);      // window t of the user's linear history
            sel.wait_front(Ct);
            mask.wait_front(1);
            a_s.wait_front(1);
            b_s.wait_front(1);
            if constexpr ((kProbe & 1) != 0) {  // probe: no math at all
                sel.pop_front(Ct);
                mask.pop_front(1);
                a_s.pop_front(1);
                b_s.pop_front(1);
                if (t == 0) {
                    state_in.wait_front(KV);
                    state_in.pop_front(KV);
                }
                if (commit) {
                    fake_produce(hnew, KV);
                }
                if (emit && first_out) {
                    fake_produce(out_acc, Vt);
                    first_out = false;
                }
                continue;
            }

            if constexpr ((kProbe & 4) != 0) {  // probe: skip conv / scatter / gates / norms
                sel.pop_front(Ct);
                a_s.pop_front(1);
                b_s.pop_front(1);
                mask.pop_front(1);
                fake_produce(beta_t, 1);
                fake_produce(dec, 1);
                fake_produce(kn, Kt);
                fake_produce(vm, Vt);
                if (emit) {
                    fake_produce(qn, Kt);
                }
            } else {
                // conv + silu on the packed window, scattered into qc, kc, vc ; gates
                causal_conv_silu_window(conv_p, t);
                conv_p.wait_front(1);
                scatter_conv<Kt, Vt>(qc, kc, vc);
                conv_p.pop_front(1);
                sel.pop_front(Ct);
                gate_beta_00(beta_t);
                b_s.pop_front(1);
                gate_decay_00(dec);
                a_s.pop_front(1);
                qc.wait_front(Kt);
                kc.wait_front(Kt);
                vc.wait_front(Vt);

                // qn = l2norm(q) * scale (cur rows only), kn = l2norm(k), vm = v masked to the token's row
                if (emit) {
                    q_norm_stage<Kt, scale_bits>(tmp, stats, scratch, inv, qn);
                }
                qc.pop_front(Kt);
                square_tiles_1(dfb::kc, dfb::tmp, tmp, Kt);
                row_sum_1<dfb::tmp, dfb::scaler, dfb::stats>(Kt);
                stats.wait_front(1);
                inverse_l2_col(dfb::stats, dfb::eps_l2, dfb::mask, dfb::scratch, scratch, dfb::inv, inv, one_bits);
                stats.pop_front(1);
                inv.wait_front(1);
                scale_rows_1(dfb::kc, dfb::inv, dfb::kn, kn, Kt);
                inv.pop_front(1);
                kc.pop_front(Kt);
                scale_rows_1(dfb::vc, dfb::mask, dfb::vm, vm, Vt);  // vm = v * e_d (the FPU multiply rounds: keep)
                vc.pop_front(Vt);
                mask.pop_front(1);
            }

            if constexpr ((kProbe & 8) != 0) {  // probe: skip the state update
                dec.wait_front(1);
                dec.pop_front(1);
                kn.wait_front(Kt);
                kn.pop_front(Kt);
                vm.wait_front(Vt);
                vm.pop_front(Vt);
                beta_t.wait_front(1);
                beta_t.pop_front(1);
                if (t == 0) {
                    state_in.wait_front(KV);
                    state_in.pop_front(KV);
                } else {
                    hn.pop_front(KV);
                }
                fake_produce(hn, KV);
                hn.wait_front(KV);
            } else {
                // hd = h * decay (h = the DRAM state for the first token, else the previous token's hn) ; vread = kn @
                // hd
                dec.wait_front(1);
                kn.wait_front(Kt);
                if (t == 0) {
                    state_in.wait_front(KV);
                }
                multiply_scalar(t == 0 ? dfb::state_in : dfb::hn, dfb::dec, dfb::hd, hd, KV);
                if (t == 0) {
                    state_in.pop_front(KV);
                } else {
                    hn.pop_front(KV);
                }
                dec.pop_front(1);
                hd.wait_front(KV);
                row_times_matrix_1(dfb::kn, dfb::hd, dfb::vread, vread, Kt, Vt);
                // delta = beta * (vm - vread)
                vread.wait_front(Vt);
                vm.wait_front(Vt);
                subtract_tiles(dfb::vm, dfb::vread, dfb::tmp, tmp, Vt);
                vread.pop_front(Vt);
                vm.pop_front(Vt);
                tmp.wait_front(Vt);
                beta_t.wait_front(1);
                multiply_scalar(dfb::tmp, dfb::beta_t, dfb::delta, delta, Vt);
                tmp.pop_front(Vt);
                beta_t.pop_front(1);
                delta.wait_front(Vt);
                // hn = hd + kn^T @ delta ; hnew (writer copy) after the commit token
                transpose_tiles(dfb::kn, dfb::kt, kt, Kt);
                kn.pop_front(Kt);
                kt.wait_front(Kt);
                outer_product(dfb::kt, dfb::delta, dfb::outer, outer, Kt, Vt);
                kt.pop_front(Kt);
                delta.pop_front(Vt);
                outer.wait_front(KV);
                add_tiles_1(dfb::hd, dfb::outer, dfb::hn, hn, KV);
                hd.pop_front(KV);
                outer.pop_front(KV);
                hn.wait_front(KV);
            }
            if (commit) {
                commit_stage<KV>(hnew);
            }
            if (!emit) {
                continue;
            }
            if constexpr ((kProbe & 16) != 0) {  // probe: skip the output stage
                qn.wait_front(Kt);
                qn.pop_front(Kt);
                if (first_out) {
                    fake_produce(out_acc, Vt);
                    first_out = false;
                }
                continue;
            }
            // o = qn @ hn ; gated = rmsnorm(o) * w * silu(z) in the token's row; accumulate over the group's rows
            output_stage<Kt, Vt, inv_dv_bits>(qn, o, tmp, stats, scratch, inv, on);
            if (first_out) {
                accumulate_first<Vt>(dfb::out_acc, out_acc);
                first_out = false;
            } else if (acc_in_out) {
                accumulate_next<Vt>(dfb::out_acc, out_acc, dfb::acc2, acc2);
                acc_in_out = false;
            } else {
                accumulate_next<Vt>(dfb::acc2, acc2, dfb::out_acc, out_acc);
                acc_in_out = true;
            }
            tmp.pop_front(Vt);
        }
        if constexpr ((kProbe & 1) == 0) {
            hn.pop_front(KV);  // the user's last (draft) state stays in L1 only
        }
        hist.wait_front(HCAP);  // the reader pads every user's window to HCAP entries
        hist.pop_front(HCAP);
    }
    // the group's rows go out once
    if (acc_in_out) {
        out_acc.wait_front(Vt);
        copy_tiles_1(dfb::out_acc, dfb::out, out, Vt);
        out_acc.pop_front(Vt);
    } else {
        acc2.wait_front(Vt);
        copy_tiles_1(dfb::acc2, dfb::out, out, Vt);
        acc2.pop_front(Vt);
    }
    zs.pop_front(Vt);
    ctl.pop_front(1);
}
