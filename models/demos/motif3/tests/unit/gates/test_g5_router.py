# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G5 — fp32 router: linear -> sigmoid -> +bias -> topk(8) -> gather(unbiased) -> renorm -> x2.

Design §1.2 (row 7), §1.5, §1.6 (kill criterion), §2.3.7 router, §4.2 G5, §7.1 risk 6. HF semantics
(``modeling_motif.py:843-870``): scores = σ(x·Wᵀ) in fp32, idx = topk(scores + expert_bias, 8),
weights = scores[idx] / (Σ + 1e-20) · 2.0.

Per chip (decode): x [1,1,32,4096] (all 32 lanes, after the row all-gather) @ W_router [4096,384].
Variants: (a) bf16 x / bf16 W, HiFi4 + fp32 dest acc, fp32 output (draft-1 choice); (b) fp32 x / fp32 W
(the named escalation); (c) bf16 output (to quantify why fp32 is needed). topk on FLOAT32 at width 384
and padded to 512 (-inf). Agreement = fraction of tokens whose top-8 *set* equals the torch fp32 golden
computed from the same bf16-valued x and W. For statistics, 32 chips × 128 distinct tokens are routed per
call (row-independent math, identical kernels), with
  * real router weights / expert_bias / post_attention_layernorm γ of layers 2 and 20 when the local
    BF16 shards are present (``weights/Motif-3``), else synthetic (σ(W) matched, bias N(1.1, 0.03));
  * inputs f = γ ⊙ rmsnorm(h) with h Gaussian or heavy-tailed (Student-t(4) + outlier channels).
Pass (design §1.6): agreement ≥ 99.9 % for variant (a) or (b).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

E, H, K = 384, 4096, 8
TOK_PER_CHIP = 128
REC = gu.Recorder("G5")
FAB2D = gu.mesh_params()
WEIGHTS_DIR = Path(os.environ.get("MOTIF3_WEIGHTS", "/home/ttuser/hchang/experiments/motif-3/weights/Motif-3"))


def load_real_router(layer: int):
    """(W [384, 4096] bf16-valued fp32, expert_bias [384] fp32, γ_post [4096]) or None if not local."""
    try:
        import json

        from safetensors import safe_open

        idx = json.load(open(WEIGHTS_DIR / "model.safetensors.index.json"))["weight_map"]
        names = {
            "w": f"model.layers.{layer}.moe.router.gate.weight",
            "b": f"model.layers.{layer}.moe.expert_bias",
            "g": f"model.layers.{layer}.post_attention_layernorm.weight",
        }
        out = {}
        for key, name in names.items():
            shard = WEIGHTS_DIR / idx[name]
            if not shard.exists():
                return None
            with safe_open(str(shard), framework="pt") as f:
                out[key] = f.get_tensor(name).float()
        return out["w"], out["b"], out["g"]
    except Exception as e:  # weights still downloading / absent
        print(f"[G5] real router weights for layer {layer} unavailable: {e}")
        return None


def synthetic_router(seed: int):
    g = torch.Generator().manual_seed(seed)
    w = (0.02 * torch.randn(E, H, generator=g)).bfloat16().float()
    b = (1.1 + 0.03 * torch.randn(E, generator=g)).bfloat16().float()
    gamma = (0.014 + 0.33 * torch.rand(H, generator=g)).bfloat16().float()
    return w, b, gamma


def make_inputs(n_tok: int, gamma: torch.Tensor, dist: str, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if dist == "gauss":
        h = torch.randn(n_tok, H, generator=g)
    else:  # heavy-tailed + a few massive channels, as in real residual streams
        torch.manual_seed(seed)
        h = torch.distributions.StudentT(4.0).sample((n_tok, H))
        ch = torch.randint(0, H, (8,), generator=g)
        h[:, ch] *= 20.0
    f = h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * gamma
    return f.bfloat16().float()  # the router input is the bf16 output of post_attention_layernorm


class DeviceRouter:
    def __init__(self, mesh_device, w, b, variant: str, pad512: bool):
        self.mesh = mesh_device
        self.variant = variant
        self.pad = pad512
        self.width = 512 if pad512 else E
        wdtype = ttnn.float32 if variant == "fp32w" else ttnn.bfloat16
        wt = torch.zeros(H, self.width)
        wt[:, :E] = w.T
        self.w = gu.to_mesh(wt, mesh_device, wdtype)
        bias = torch.full((1, 1, 1, self.width), float("-inf"))
        bias[..., :E] = b
        self.b = gu.to_mesh(bias, mesh_device, ttnn.float32)
        self.ckc = gu.compute_cfg("HiFi4", fp32_acc=True, approx=False, packer_l1_acc=False)
        self.out_dtype = ttnn.bfloat16 if variant == "bf16out" else ttnn.float32

    def scores(self, x_tt):
        xin = ttnn.typecast(x_tt, ttnn.float32) if self.variant == "fp32w" else x_tt
        logits = ttnn.linear(xin, self.w, dtype=self.out_dtype, compute_kernel_config=self.ckc)
        if self.out_dtype != ttnn.float32:
            logits = ttnn.typecast(logits, ttnn.float32)
        return ttnn.sigmoid(logits)

    def __call__(self, x_tt):
        s = self.scores(x_tt)
        biased = ttnn.add(s, self.b)  # pad columns: σ(0)·… + (-inf) = -inf -> never selected
        _, idx = ttnn.topk(biased, k=K, dim=-1, largest=True, sorted=True)
        w = ttnn.gather(s, -1, idx)
        den = ttnn.add(ttnn.sum(w, dim=-1, keepdim=True), 1e-20)
        w = ttnn.multiply(ttnn.multiply(w, ttnn.reciprocal(den)), 2.0)
        return idx, w


def agreement(idx_dev: torch.Tensor, idx_ref: torch.Tensor, biased_ref: torch.Tensor):
    """(fraction of tokens with equal top-8 sets, list of golden 8th-9th margins for the mismatching tokens)."""
    a = torch.sort(idx_dev.long(), dim=-1).values
    b = torch.sort(idx_ref.long(), dim=-1).values
    eq = (a == b).all(dim=-1)
    top9 = torch.topk(biased_ref, k=K + 1, dim=-1).values
    margin = (top9[:, K - 1] - top9[:, K]).double()
    return float(eq.float().mean()), [float(m) for m in margin[~eq][:20]], eq, margin


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
def test_g5_router_agreement(mesh_device):
    n_tok = TOK_PER_CHIP * 32
    sources = []
    for layer in (2, 20):
        real = load_real_router(layer)
        if real is not None:
            sources.append((f"real_L{layer}", *real))
    sources.append(("synthetic", *synthetic_router(5)))
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=gu.MESH_SHAPE)
    failures = []
    best = {}
    for src, w, b, gamma in sources:
        REC.add(
            f"{src}/router_stats",
            status="info",
            w_std=float(w.std()),
            bias_mean=float(b.mean()),
            bias_std=float(b.std()),
            bias_min=float(b.min()),
            bias_max=float(b.max()),
        )
        for dist in ("gauss", "heavy"):
            x = make_inputs(n_tok, gamma, dist, seed=gu.seed_of(src, dist))
            idx_ref, w_ref, biased_ref = gd.router_golden(x, w, b, k=K, scale=2.0, dtype=torch.float64)
            x_tt = ttnn.from_torch(
                x.reshape(4, 8, TOK_PER_CHIP, H),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=mapper,
            )
            for variant in ("bf16w", "fp32w", "bf16out"):
                for pad512 in (False, True):
                    case = f"{src}/{dist}/{variant}/width{'512pad' if pad512 else '384'}"
                    try:
                        router = DeviceRouter(mesh_device, w, b, variant, pad512)
                        idx_tt, wt_tt = router(x_tt)
                        idx_dev = torch.cat(
                            [ttnn.to_torch(t).reshape(-1, K) for t in ttnn.get_device_tensors(idx_tt)], dim=0
                        )
                        w_dev = torch.cat(
                            [ttnn.to_torch(t).float().reshape(-1, K) for t in ttnn.get_device_tensors(wt_tt)], dim=0
                        )
                    except Exception as e:
                        REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
                        failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
                        continue
                    agr, margins, eq, margin = agreement(idx_dev, idx_ref, biased_ref)
                    # weight error on tokens whose sets agree (order-insensitive: compare sorted by index)
                    o_dev = torch.argsort(idx_dev.long(), dim=-1)
                    o_ref = torch.argsort(idx_ref.long(), dim=-1)
                    wd = w_dev.gather(-1, o_dev)[eq]
                    wr = w_ref.float().gather(-1, o_ref)[eq]
                    w_err = float((wd - wr).abs().max()) if wd.numel() else float("nan")
                    REC.add(
                        case,
                        status="measured",
                        tokens=n_tok,
                        agreement=agr,
                        n_mismatch=int((~eq).sum()),
                        mismatch_margins=margins,
                        golden_margin_p01=float(torch.quantile(margin.float(), 0.01)),
                        weight_max_abs_err=w_err,
                        weight_sum_dev=float((w_dev.sum(-1) - 2.0).abs().max()),
                    )
                    key = (src, dist)
                    if variant in ("bf16w", "fp32w"):
                        best[key] = max(best.get(key, 0.0), agr)
    for key, agr in best.items():
        ok = agr >= 0.999
        REC.add(f"{key[0]}/{key[1]}/best_of_bf16w_fp32w", status="pass" if ok else "fail", agreement=agr)
        if not ok:
            failures.append(f"{key}: best agreement {agr:.5f} < 0.999")
    assert not failures, "G5 failures:\n" + "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
def test_g5_router_perf(mesh_device):
    w, b, gamma = load_real_router(2) or synthetic_router(5)
    x = make_inputs(32, gamma, "gauss", seed=1)
    x_tt = gu.to_mesh(x.reshape(1, 1, 32, H), mesh_device, ttnn.bfloat16)
    for variant in ("bf16w", "fp32w"):
        for pad512 in (False, True):
            router = DeviceRouter(mesh_device, w, b, variant, pad512)
            case = f"perf/decode32/{variant}/width{'512pad' if pad512 else '384'}"
            try:
                eager = gu.time_eager(mesh_device, lambda: router(x_tt), iters=20)
                traced, raw = gu.time_traced(mesh_device, lambda: router(x_tt), ops_per_trace=64, reps=9)
                mm_traced, _ = gu.time_traced(mesh_device, lambda: router.scores(x_tt), ops_per_trace=64, reps=9)
                s_tt = router.scores(x_tt)
                biased = ttnn.add(s_tt, router.b)
                topk_traced, _ = gu.time_traced(
                    mesh_device,
                    lambda: ttnn.topk(biased, k=K, dim=-1, largest=True, sorted=True),
                    ops_per_trace=64,
                    reps=9,
                )
                idx_tt, wt_tt = router(x_tt)
                idx_dev = ttnn.to_torch(ttnn.get_device_tensors(idx_tt)[0]).reshape(-1, K)
                idx_ref, _, biased_ref = gd.router_golden(x, w, b, k=K, dtype=torch.float64)
                agr = agreement(idx_dev, idx_ref, biased_ref)[0]
                # routing-consistency invariant (design §2.3.7): identical inputs must route identically on all 32 chips
                same_idx = gu.replicas_identical(idx_tt)[0]
                same_w = gu.replicas_identical(wt_tt)[0]
                REC.add(
                    case,
                    status="measured",
                    replicas_identical_idx=same_idx,
                    replicas_identical_weights=same_w,
                    eager_us=eager,
                    traced_us=traced,
                    traced_raw_us=raw,
                    linear_sigmoid_traced_us=mm_traced,
                    topk_traced_us=topk_traced,
                    agreement_32tok=agr,
                )
            except Exception as e:
                REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")


# ------------------------------------------------------------------------------------------------
# diagnosis: which stage causes the near-tie flips, and which matmul variant removes them
# ------------------------------------------------------------------------------------------------
def split_bf16(w: torch.Tensor, parts: int):
    """w (bf16-valued) = sum of `parts` tensors whose mantissas have <= ceil(8/parts) significant bits each, so that a
    bf16-activation x part product needs <= 8 + ceil(8/parts) mantissa bits (exact in TF32's 11 when parts >= 3)."""
    out = []
    r = w.double()
    bits = -(-8 // parts)
    for i in range(parts):
        if i == parts - 1:
            out.append(r.float())
            break
        e = torch.floor(torch.log2(r.abs().clamp(min=1e-38)))
        q = torch.where(r == 0, torch.zeros_like(r), torch.round(r / 2.0 ** (e - bits + 1)) * 2.0 ** (e - bits + 1))
        out.append(q.float())
        r = r - q
    assert torch.equal(sum(p.double() for p in out).float(), w.float())
    return out


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
def test_g5_router_diagnose(mesh_device):
    real = load_real_router(2)
    w, b, gamma = real if real is not None else synthetic_router(5)
    src = "real_L2" if real is not None else "synthetic"
    n_tok = TOK_PER_CHIP * 32
    x = make_inputs(n_tok, gamma, "gauss", seed=gu.seed_of(src, "gauss"))
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=gu.MESH_SHAPE)
    x_tt = ttnn.from_torch(x.reshape(4, 8, TOK_PER_CHIP, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)
    logits_ref = x.double() @ w.double().T  # [n_tok, E]
    idx_ref, _, biased_ref = gd.router_golden(x, w, b, k=K, dtype=torch.float64)
    b_tt = gu.to_mesh(b.reshape(1, 1, 1, E), mesh_device, ttnn.float32)

    def gather_all(t):
        return torch.cat([ttnn.to_torch(s).double().reshape(-1, ttnn.to_torch(s).shape[-1]) for s in ttnn.get_device_tensors(t)], dim=0)

    def finish(logits_tt, tag):
        s_tt = ttnn.sigmoid(logits_tt)
        biased_tt = ttnn.add(s_tt, b_tt)
        _, idx_tt = ttnn.topk(biased_tt, k=K, dim=-1, largest=True, sorted=True)
        logits_dev = gather_all(logits_tt)
        s_dev = gather_all(s_tt)
        biased_dev = gather_all(biased_tt)
        idx_dev = gather_all(idx_tt).long()
        agr = agreement(idx_dev, idx_ref, biased_ref)[0]
        # stage errors
        l_err = (logits_dev - logits_ref).abs()
        s_err = (s_dev - torch.sigmoid(logits_dev)).abs()  # SFPU sigmoid vs exact sigmoid of the *device* logits
        _, idx_host_on_dev_scores = torch.topk(biased_dev, k=K, dim=-1)
        topk_exact = float((torch.sort(idx_host_on_dev_scores, -1).values == torch.sort(idx_dev, -1).values).all(-1).float().mean())
        idx_from_dev_logits = torch.topk(torch.sigmoid(logits_dev) + b.double(), k=K, dim=-1).indices
        agr_if_exact_sigmoid = float((torch.sort(idx_from_dev_logits, -1).values == torch.sort(idx_ref, -1).values).all(-1).float().mean())
        REC.add(
            f"diagnose/{src}/{tag}",
            status="measured",
            agreement=agr,
            logit_max_abs_err=float(l_err.max()),
            logit_rms_err=float(l_err.pow(2).mean().sqrt()),
            logit_rms=float(logits_ref.pow(2).mean().sqrt()),
            sigmoid_max_abs_err=float(s_err.max()),
            topk_matches_host_topk_of_device_scores=topk_exact,
            agreement_if_sigmoid_exact=agr_if_exact_sigmoid,
        )

    wt = w.T.contiguous().reshape(1, 1, H, E)
    w16 = gu.to_mesh(wt, mesh_device, ttnn.bfloat16)
    base = dict(math_approx_mode=False, fp32_dest_acc_en=True)
    variants = {
        "hifi4_fp32acc": ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, packer_l1_acc=False, **base),
        "hifi4_fp32acc_packerl1": ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, packer_l1_acc=True, **base),
        "hifi2_fp32acc": ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, packer_l1_acc=False, **base),
    }
    for tag, ckc in variants.items():
        try:
            finish(ttnn.linear(x_tt, w16, dtype=ttnn.float32, compute_kernel_config=ckc), tag)
        except Exception as e:
            REC.add(f"diagnose/{src}/{tag}", status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
    ckc = variants["hifi4_fp32acc"]
    # K split: 8 partial matmuls over K/8, summed with fp32 SFPU adds
    try:
        kk = H // 8
        parts = []
        for i in range(8):
            xs = ttnn.slice(x_tt, [0, 0, 0, i * kk], [1, 1, TOK_PER_CHIP, (i + 1) * kk])
            ws = gu.to_mesh(wt[:, :, i * kk : (i + 1) * kk, :].contiguous(), mesh_device, ttnn.bfloat16)
            parts.append(ttnn.linear(xs, ws, dtype=ttnn.float32, compute_kernel_config=ckc))
        acc = parts[0]
        for p_ in parts[1:]:
            acc = ttnn.add(acc, p_)
        finish(acc, "ksplit8_fp32add")
    except Exception as e:
        REC.add(f"diagnose/{src}/ksplit8_fp32add", status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
    # weight mantissa split: W = sum of parts with few mantissa bits each, so every partial product is exact
    for nparts in (2, 3):
        try:
            ws_parts = split_bf16(wt, nparts)
            acc = None
            for wp in ws_parts:
                wp_tt = gu.to_mesh(wp, mesh_device, ttnn.bfloat16)
                y = ttnn.linear(x_tt, wp_tt, dtype=ttnn.float32, compute_kernel_config=ckc)
                acc = y if acc is None else ttnn.add(acc, y)
            finish(acc, f"wsplit{nparts}_fp32add")
        except Exception as e:
            REC.add(f"diagnose/{src}/wsplit{nparts}_fp32add", status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
