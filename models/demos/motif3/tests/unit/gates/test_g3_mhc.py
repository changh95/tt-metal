# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G3 — mHC parametrisation and stream mixing at Motif shapes.

Design §1.2 (rows 5-6), §2.3.3, §4.2 G3, §7.1 risk 3. Two ops:

1. ``ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(mixes [T,24] fp32, consts [8,32,32], n=4, iters=20, eps)``
   vs Motif's exact parametrisation (HF ``modeling_motif.py:226-249``):
     h_pre  = σ(clamp(α_pre·p_pre + b_pre, ±10))
     h_post = 1.0·σ(clamp(α_post·p_post + b_post, ±10))
     H      = Sinkhorn20(exp(clamp(α_res·p_res + B_res, ±20))) with row/col sums clamped ≥ 1e-8, fp32.
   The kernel computes pre = σ(·)+eps, post = 2σ(·), comb = Sinkhorn(exp(min(x, 80))) with eps added to
   the column-normalisation divisors and no 1e-8 floor. Two call modes are measured:
     * ``direct``     — α and biases folded into consts (SEL/base), raw projections as ``mixes``;
     * ``preclamped`` — logits computed and clamped (±10 / ±20) *before* the kernel, consts = identity
                        SEL, zero base. exp(min(x,80)) then equals Motif's exp(clamp(x,±20)) exactly.
   Logit regimes: realistic (std 0.5), moderate (3), wide (10), peaked (30, many clamped), degenerate
   rows (all four logits of a row ≤ -20, where Motif's 1e-8 floor binds), overflow (> 80).
   h_post is derived as ``post / 2`` (exact power-of-two scale); h_pre = ``pre`` when eps = 0.

2. ``ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc`` for the [1, 4, T, 4096] stream mixing:
   x_red = Σ_i h_pre_i·X_i (weights [1,4,T,1]), X' = H·X (weights [4,4,T,1]) and the fused post
   [X ‖ out] (C=5) with [H | h_post] (weights [4,5,T,1]); bf16 input, fp32 weights, golden = fp32 einsum
   + one bf16 rounding (HF ``modeling_motif.py:1207-1244``). Composite fallback multiply+sum(dim=1) too.

Pass (design §4.2): max|ΔH| ≤ 5e-3, max|Δh_pre|, max|Δh_post| ≤ 2e-3 on realistic-range logits;
weighted reduce PCC ≥ 0.9999 and ≤ 1 bf16 ulp from the fp32-einsum golden.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

N = 4
MIX = (2 + N) * N  # 24
ITERS = 20
HIDDEN = 4096
REC = gu.Recorder("G3")
FAB2D = gu.mesh_params()

# Real-checkpoint-like constants (01_motif_reference.md §6.4 / A.2): |α| ≤ 0.33, biases within ±0.55, B_res within ±0.48
ALPHA = (-0.22, 0.33, 0.18)


def make_bias(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    b_pre = (torch.rand(N, generator=g) - 0.5) * 1.0
    b_post = (torch.rand(N, generator=g) - 0.5) * 1.0
    b_res = (0.17 * torch.randn(N * N, generator=g)).clamp(-0.48, 0.48)
    return torch.cat([b_pre, b_post, b_res])


def make_projections(T: int, regime: str, seed: int) -> torch.Tensor:
    """Raw fp32 projections p [T, 24] whose logits α·p + b have the requested spread."""
    g = torch.Generator().manual_seed(seed)
    std = {"realistic": 0.5, "moderate": 3.0, "wide": 10.0, "peaked": 30.0, "degenerate": 0.5, "overflow": 0.5}[regime]
    alpha = torch.tensor([ALPHA[0]] * N + [ALPHA[1]] * N + [ALPHA[2]] * (N * N))
    p = torch.randn(T, MIX, generator=g) * std / alpha.abs()
    if regime == "degenerate":
        # every 4th token: row 1 of M has all four logits ≈ -25 (Motif clamps to -20; row sum 8.2e-9 < 1e-8 floor)
        idx = torch.arange(0, T, 4)
        p[idx[:, None], 2 * N + 4 + torch.arange(4)[None, :]] = -25.0 / ALPHA[2]
    if regime == "overflow":
        # every 4th token: one res logit at +90 (beyond the kernel's min(x, 80) cap and Motif's +20 clamp)
        idx = torch.arange(0, T, 4)
        p[idx, 2 * N + 5] = 90.0 / ALPHA[2]
    return p


def run_sinkhorn(mesh_device, mixes: torch.Tensor, consts: torch.Tensor, eps: float = 0.0):
    T = mixes.shape[0]
    m32 = torch.zeros(T, 32)
    m32[:, :MIX] = mixes
    mixes_tt = gu.to_mesh(m32[:, :MIX].contiguous(), mesh_device, ttnn.float32)
    consts_tt = gu.to_mesh(consts, mesh_device, ttnn.float32)
    pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(mixes_tt, consts_tt, N, ITERS, float(eps))
    return mixes_tt, consts_tt, (pre, post, comb)


def maxabs(a, b) -> float:
    d = (a.double() - b.double()).abs()
    if not bool(torch.isfinite(d).all()):
        return float("inf")
    return float(d.max())


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
@pytest.mark.parametrize("T", [8, 32, 4096])
def test_g3_sinkhorn(mesh_device, T):
    failures = []
    bias = make_bias(seed=11)
    consts_direct = gd.build_sinkhorn_consts(N, ALPHA, bias)
    consts_ident = gd.build_sinkhorn_consts(N, (1.0, 1.0, 1.0), torch.zeros(MIX))
    xcheck = gd.reference_crosscheck("sinkhorn", gd.motif_sinkhorn, torch.randn(T, N, N), ITERS)
    REC.add(f"T{T}/reference_crosscheck_sinkhorn", status="info", result=xcheck)

    for regime in ["realistic", "moderate", "wide", "peaked", "degenerate", "overflow"]:
        p = make_projections(T, regime, seed=gu.seed_of(T, regime))
        h_pre, h_post, H = gd.motif_mhc_maps(p, ALPHA, bias, ITERS)  # exact Motif fp32
        raw_logits = gd.mhc_logits(p, ALPHA, bias, clamp=False)
        H_plain = gd.plain_sinkhorn(raw_logits[:, 2 * N :].reshape(T, N, N), ITERS)  # kernel semantics, fp32
        for mode in ["direct", "preclamped"]:
            case = f"T{T}/{regime}/{mode}"
            if mode == "direct":
                mixes, consts = p, consts_direct
            else:
                mixes, consts = gd.mhc_logits(p, ALPHA, bias, clamp=True), consts_ident
            try:
                _, _, (pre_tt, post_tt, comb_tt) = run_sinkhorn(mesh_device, mixes, consts, eps=0.0)
                pre = gu.read_dev(pre_tt)[:T, :N]
                post = gu.read_dev(post_tt)[:T, :N]
                comb = gu.read_dev(comb_tt)[:T, : N * N].reshape(T, N, N)
                same = all(gu.replicas_identical(t)[0] for t in (pre_tt, post_tt, comb_tt))
            except Exception as e:
                REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
                failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
                continue
            e_H = maxabs(H, comb)
            e_pre = maxabs(h_pre, pre)
            e_post = maxabs(h_post, post / 2.0)
            e_H_kernel_sem = maxabs(
                H_plain if mode == "direct" else gd.plain_sinkhorn(mixes[:, 2 * N :].reshape(T, N, N), ITERS), comb
            )
            colsum_dev = float((comb.sum(dim=-2) - 1).abs().max()) if torch.isfinite(comb).all() else float("inf")
            rowsum_dev = float((comb.sum(dim=-1) - 1).abs().max()) if torch.isfinite(comb).all() else float("inf")
            row = dict(
                max_abs_H=e_H,
                max_abs_h_pre=e_pre,
                max_abs_h_post_from_post_div2=e_post,
                max_abs_H_vs_kernel_semantics=e_H_kernel_sem,
                H_colsum_dev=colsum_dev,
                H_rowsum_dev=rowsum_dev,
                post_over_sigma=float((post / (2 * h_post)).mean()) if regime == "realistic" else None,
                nonfinite=int(
                    (~torch.isfinite(comb)).sum() + (~torch.isfinite(pre)).sum() + (~torch.isfinite(post)).sum()
                ),
                replicas_identical=same,
            )
            if regime == "realistic":
                ok = e_H <= 5e-3 and e_pre <= 2e-3 and e_post <= 2e-3 and same
                REC.add(case, status="pass" if ok else "fail", **row)
                if not ok:
                    failures.append(f"{case}: |dH|={e_H:.2e} |dpre|={e_pre:.2e} |dpost|={e_post:.2e} replicas={same}")
            else:
                REC.add(case, status="measured", **row)

    # eps semantics (documentation): pre = σ + eps, column divisors + eps
    p = make_projections(T, "realistic", seed=5)
    h_pre, h_post, H = gd.motif_mhc_maps(p, ALPHA, bias, ITERS)
    try:
        _, _, (pre_tt, post_tt, comb_tt) = run_sinkhorn(mesh_device, p, consts_direct, eps=1e-6)
        pre = gu.read_dev(pre_tt)[:T, :N]
        comb = gu.read_dev(comb_tt)[:T, : N * N].reshape(T, N, N)
        REC.add(
            f"T{T}/realistic/direct_eps1e-6",
            status="info",
            mean_pre_minus_hpre=float((pre - h_pre).mean()),
            max_abs_H=maxabs(H, comb),
        )
    except Exception as e:
        REC.add(f"T{T}/realistic/direct_eps1e-6", status="error", error=f"{type(e).__name__}: {str(e)[:300]}")

    assert not failures, "G3 sinkhorn failures:\n" + "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
def test_g3_sinkhorn_perf(mesh_device):
    bias = make_bias(seed=11)
    consts = gd.build_sinkhorn_consts(N, ALPHA, bias)
    for T in (8, 32, 4096):
        p = make_projections(T, "realistic", seed=1)
        mixes_tt, consts_tt, _ = run_sinkhorn(mesh_device, p, consts)

        def fn():
            return ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(mixes_tt, consts_tt, N, ITERS, 0.0)

        eager = gu.time_eager(mesh_device, fn, iters=50)
        tr, tr_raw = gu.time_traced(mesh_device, fn, ops_per_trace=64, reps=9) if T <= 32 else (None, None)
        REC.add(f"perf/sinkhorn/T{T}", status="measured", eager_us=eager, traced_us=tr, traced_raw_us=tr_raw)


# ------------------------------------------------------------------------------------------------
# attn_res_weighted_reduce_nc
# ------------------------------------------------------------------------------------------------
def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    x = x.float().abs()
    return torch.where(x == 0, torch.full_like(x, 2.0**-133), 2.0 ** (torch.floor(torch.log2(x)) - 7))


def bf16_ulp_err(want_bf16: torch.Tensor, got: torch.Tensor) -> tuple[float, float]:
    """(max ulps, fraction exactly equal) of got vs a bf16 golden, in units of each golden element's own bf16 ulp
    (very strict near zero, where cancellation makes the golden's ulp tiny)."""
    w = want_bf16.float()
    d = (got.float() - w).abs() / bf16_ulp(w)
    return float(d.max()), float((got.float() == w).float().mean())


def scale_ulp_err(want: torch.Tensor, got: torch.Tensor, terms_absmax: torch.Tensor) -> float:
    """max |Δ| in units of the bf16 ulp of the largest summand |x_c·w_c| of each output element: the error a
    single bf16 rounding of the *inputs* to the sum would cause (cancellation-insensitive)."""
    d = (got.float() - want.float()).abs()
    return float((d / bf16_ulp(terms_absmax.clamp(min=1e-30))).max())


def weighted_reduce_cases(T: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    X = (2.0 * torch.randn(1, N, T, HIDDEN, generator=g)).bfloat16()
    out = (2.0 * torch.randn(1, 1, T, HIDDEN, generator=g)).bfloat16()
    p = make_projections(T, "realistic", seed=seed)
    h_pre, h_post, H = gd.motif_mhc_maps(p, ALPHA, make_bias(3), ITERS)
    w_pre = h_pre.T.reshape(1, N, T, 1).contiguous()  # [1, C=4, T, 1]
    w_res = H.permute(1, 2, 0).reshape(N, N, T, 1).contiguous()  # [R=4, C=4, T, 1]: w[r, c, t] = H_t[r, c]
    w_post = torch.cat([w_res, h_post.T.reshape(N, 1, T, 1)], dim=1).contiguous()  # [4, 5, T, 1]
    Xc = torch.cat([X, out], dim=1)  # [1, 5, T, 4096]
    return X, Xc, w_pre, w_res, w_post


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
@pytest.mark.parametrize("T", [8, 32, 4096])
def test_g3_weighted_reduce(mesh_device, T):
    failures = []
    X, Xc, w_pre, w_res, w_post = weighted_reduce_cases(T, seed=T)
    X_tt = gu.to_mesh(X, mesh_device, ttnn.bfloat16)
    Xc_tt = gu.to_mesh(Xc, mesh_device, ttnn.bfloat16)
    cases = [
        ("pre_x_red_R1C4", X, X_tt, w_pre),
        ("res_HX_R4C4", X, X_tt, w_res),
        ("post_fused_R4C5", Xc, Xc_tt, w_post),
    ]
    for name, xin, xin_tt, w in cases:
        prod = xin.float() * w.float()  # [R, C, T, 4096]
        want = prod.sum(dim=1, keepdim=True)  # [R, 1, T, 4096] fp32 einsum
        want_bf16 = want.bfloat16()
        terms = prod.abs().amax(dim=1, keepdim=True)
        for wdtype in (ttnn.float32, ttnn.bfloat16):
            case = f"T{T}/{name}/w_{'fp32' if wdtype == ttnn.float32 else 'bf16'}"
            w_tt = gu.to_mesh(w, mesh_device, wdtype)

            def fn():
                return ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(xin_tt, w_tt, dim=1)

            try:
                out_tt = fn()
                got = gu.read_dev(out_tt)[..., :T, :]
                same = gu.replicas_identical(out_tt)[0] if T <= 32 else None
                eager = gu.time_eager(mesh_device, fn, iters=20)
                tr = gu.time_traced(mesh_device, fn, ops_per_trace=64, reps=9)[0] if T <= 32 else None
            except Exception as e:
                REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
                failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
                continue
            s = gu.compare(want, got)
            max_ulp, frac_eq = bf16_ulp_err(want_bf16, got)
            sulp = scale_ulp_err(want, got, terms)
            # A single correct bf16 rounding of a C-term sum is <= 0.5*C ulps of the largest summand. Pass = PCC and
            # at most one extra rounding (<= C summand-ulps). The strict per-element ulp vs the golden (huge near
            # zero from cancellation) and the bit-exact fraction are reported, not gated.
            C = xin.shape[1]
            ok = s["pcc"] >= 0.9999 and sulp <= C and same in (None, True)
            REC.add(
                case,
                status=("pass" if ok else "fail") if wdtype == ttnn.float32 else "measured",
                pcc=s["pcc"],
                max_abs=s["max_abs"],
                max_bf16_ulp_strict=max_ulp,
                max_ulp_of_largest_summand=sulp,
                frac_bitexact_vs_fp32einsum_bf16=frac_eq,
                replicas_identical=same,
                eager_us=eager,
                traced_us=tr,
            )
            if not ok and wdtype == ttnn.float32:
                failures.append(f"{case}: {gu.fmt(s)} strict_ulp={max_ulp:.1f} summand_ulp={sulp:.2f}")

    # composite fallback for the pre reduction: multiply (bcast [1,4,T,1]) + sum(dim=1), fp32 weights
    w_tt = gu.to_mesh(w_pre, mesh_device, ttnn.float32)
    ckc = gu.hifi4(fp32_acc=True)

    def fallback():
        prod = ttnn.multiply(X_tt, w_tt, dtype=ttnn.float32)
        red = ttnn.sum(prod, dim=1, keepdim=True, compute_kernel_config=ckc)
        return ttnn.typecast(red, ttnn.bfloat16)

    try:
        got = gu.read_dev(fallback())[..., :T, :]
        prod = X.float() * w_pre.float()
        want = prod.sum(dim=1, keepdim=True)
        s = gu.compare(want, got)
        max_ulp, frac_eq = bf16_ulp_err(want.bfloat16(), got)
        sulp = scale_ulp_err(want, got, prod.abs().amax(dim=1, keepdim=True))
        eager = gu.time_eager(mesh_device, fallback, iters=20)
        tr = gu.time_traced(mesh_device, fallback, ops_per_trace=64, reps=9)[0] if T <= 32 else None
        REC.add(
            f"T{T}/pre_x_red_R1C4/composite_fallback",
            status="measured",
            pcc=s["pcc"],
            max_abs=s["max_abs"],
            max_bf16_ulp_strict=max_ulp,
            max_ulp_of_largest_summand=sulp,
            frac_bitexact_vs_fp32einsum_bf16=frac_eq,
            eager_us=eager,
            traced_us=tr,
        )
    except Exception as e:
        REC.add(f"T{T}/pre_x_red_R1C4/composite_fallback", status="error", error=f"{type(e).__name__}: {str(e)[:300]}")

    assert not failures, "G3 weighted-reduce failures:\n" + "\n".join(failures)
