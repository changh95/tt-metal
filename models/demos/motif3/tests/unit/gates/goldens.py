# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local torch goldens for the device op gates (pure torch, no ttnn, no device).

The CPU reference package ``models.demos.motif3.reference`` is being written in parallel and its API
may still change, so every gate computes its golden from these self-contained functions, which
transcribe the HF semantics directly (``hf_meta/modeling_motif.py``, cited per function). Where a
reference function exists, ``reference_crosscheck`` compares against it lazily and only records the
outcome; it never fails a gate.
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------------------------
# Motif constants (design §2.3.1; 01_motif_reference.md §3.11, A.1)
# ---------------------------------------------------------------------------------------------
SCALE_SWA = 192**-0.5  # 0.07216878
SCALE_GLOBAL = 192**-0.5 * (0.1 * math.log(64.0) + 1.0) ** 2  # 0.14467963
WINDOW = 129  # keys including the current one (HF sliding_window = 128 + 1)
ROPE_DIM = 64
ROPE_THETA = 1e4
YARN_FACTOR = 64.0
YARN_ORIG = 4096
YARN_BETA_FAST = 32.0
YARN_BETA_SLOW = 1.0


# ---------------------------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------------------------
def mla_decode_golden(q: torch.Tensor, kv: torch.Tensor, positions, scale: float, window, dv: int = 512):
    """Absorbed latent decode attention (nkv = 1, V = first ``dv`` columns of the latent).

    q [B, NH, Dqk], kv [B, S, Dqk] (already quantised the way the device cache stores it).
    User b at position p attends keys [lo, p], lo = 0 (global) or max(0, p - window + 1) (SWA),
    which is the HF/FA2 window_size=(window-1, 0) semantics (01 §3.12). Rows with p < 0 are zero.
    Returns [B, NH, dv] float64.
    """
    B, NH, _ = q.shape
    out = torch.zeros(B, NH, dv, dtype=torch.float64)
    for b, p in enumerate(positions):
        if p < 0:
            continue
        lo = 0 if window is None else max(0, p - window + 1)
        k = kv[b, lo : p + 1].double()
        s = (q[b].double() @ k.T) * scale
        a = torch.softmax(s, dim=-1)
        out[b] = a @ k[:, :dv]
    return out


def gqa_prefill_golden(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, window):
    """Causal (optionally sliding-window) GQA prefill attention.

    q [NQ, S, D], k [NKV, S, D], v [NKV, S, Dv]; q head h uses kv group h // (NQ // NKV)
    (Motif head grouping: q heads 5g..5g+4 belong to kv group g, HF modeling_motif.py:745-775).
    Returns [NQ, S, Dv] float32 (computed in chunks of 1024 query rows, fp32 softmax).
    """
    NQ, S, _ = q.shape
    NKV = k.shape[0]
    rep = NQ // NKV
    out = torch.empty(NQ, S, v.shape[-1], dtype=torch.float32)
    kpos = torch.arange(S)
    for g in range(NKV):
        kg = k[g].float()
        vg = v[g].float()
        for h in range(g * rep, (g + 1) * rep):
            qh = q[h].float()
            for q0 in range(0, S, 1024):
                q1 = min(S, q0 + 1024)
                qpos = torch.arange(q0, q1)[:, None]
                allow = kpos[None, :] <= qpos
                if window is not None:
                    allow &= kpos[None, :] >= qpos - (window - 1)
                s = (qh[q0:q1] @ kg.T) * scale
                s = s.masked_fill(~allow, float("-inf"))
                out[h, q0:q1] = torch.softmax(s, dim=-1) @ vg
    return out


# ---------------------------------------------------------------------------------------------
# mHC (HF modeling_motif.py:226-249)
# ---------------------------------------------------------------------------------------------
def motif_sinkhorn(logits: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """HF ``_sinkhorn_knopp_batch``: fp32 exp(clamp(L, ±20)), then ``iters`` x (row, col) with sums ≥ 1e-8.

    logits [..., 4, 4] with M[i][j] = p_res[4i+j] (row-major view of the 16 logits).
    """
    m = logits.float().clamp(-20.0, 20.0).exp()
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        m = m / m.sum(dim=-2, keepdim=True).clamp(min=1e-8)
    return m


def plain_sinkhorn(logits: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Kernel-semantics Sinkhorn (no clamps, no floor; exp(min(x, 80)) like the stock op, eps = 0)."""
    m = torch.minimum(logits.float(), torch.tensor(80.0)).exp()
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True)
        m = m / m.sum(dim=-2, keepdim=True)
    return m


def motif_mhc_maps(p: torch.Tensor, alpha, bias: torch.Tensor, iters: int = 20, n: int = 4):
    """h_pre, h_post, H_res from the 24 fp32 projections ``p`` [T, 24] = (pre 4 | post 4 | res 16).

    alpha = (a_pre, a_post, a_res) scalars; bias [24] = (b_pre | b_post | B_res.flatten()).
    h_pre = σ(clamp(a·p + b, ±10)); h_post = 1.0·σ(clamp(…, ±10)); H = Sinkhorn20(a_res·p_res + B_res).
    """
    a_pre, a_post, a_res = (float(alpha[0]), float(alpha[1]), float(alpha[2]))
    p = p.float()
    b = bias.float()
    lp = (a_pre * p[:, :n] + b[:n]).clamp(-10.0, 10.0)
    lq = (a_post * p[:, n : 2 * n] + b[n : 2 * n]).clamp(-10.0, 10.0)
    lr = a_res * p[:, 2 * n :] + b[2 * n :]
    h_pre = torch.sigmoid(lp)
    h_post = 1.0 * torch.sigmoid(lq)
    H = motif_sinkhorn(lr.reshape(-1, n, n), iters)
    return h_pre, h_post, H


def mhc_logits(p: torch.Tensor, alpha, bias: torch.Tensor, n: int = 4, clamp: bool = True) -> torch.Tensor:
    """The 24 Motif logits (optionally pre-clamped: ±10 for pre/post, ±20 for res), fp32 [T, 24]."""
    a_pre, a_post, a_res = (float(alpha[0]), float(alpha[1]), float(alpha[2]))
    p = p.float()
    b = bias.float()
    lp = a_pre * p[:, :n] + b[:n]
    lq = a_post * p[:, n : 2 * n] + b[n : 2 * n]
    lr = a_res * p[:, 2 * n :] + b[2 * n :]
    if clamp:
        lp, lq, lr = lp.clamp(-10, 10), lq.clamp(-10, 10), lr.clamp(-20, 20)
    return torch.cat([lp, lq, lr], dim=-1)


def build_sinkhorn_consts(n: int, scale, base: torch.Tensor) -> torch.Tensor:
    """[8, 32, 32] fp32 constants for ``mhc_split_sinkhorn`` (port of the DeepSeek-V4 d_p builder,
    ``models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py:55-117``; copied, not imported, per the import rule).

    Tile order: 0 SEL_pre 1 SEL_post 2 SEL_comb 3 base_pre 4 base_post 5 base_comb 6 RB 7 CB.
    """
    W = 32
    a_pre, a_post, a_res = (float(scale[0]), float(scale[1]), float(scale[2]))
    base = base.float()
    sel_pre = torch.zeros(W, W)
    sel_post = torch.zeros(W, W)
    sel_comb = torch.zeros(W, W)
    for p in range(n):
        sel_pre[p, p] = a_pre
        sel_post[n + p, p] = a_post
    for p in range(n * n):
        sel_comb[2 * n + p, p] = a_res
    base_pre = torch.zeros(W, W)
    base_post = torch.zeros(W, W)
    base_comb = torch.zeros(W, W)
    base_pre[:, :n] = base[0:n]
    base_post[:, :n] = base[n : 2 * n]
    base_comb[:, : n * n] = base[2 * n :]
    RB = torch.zeros(W, W)
    CB = torch.zeros(W, W)
    for p in range(n * n):
        pi, pj = divmod(p, n)
        for q in range(n * n):
            qi, qj = divmod(q, n)
            if qi == pi:
                RB[q, p] = 1.0
            if qj == pj:
                CB[q, p] = 1.0
    for p in range(n * n, W):
        RB[p, p] = 1.0
        CB[p, p] = 1.0
    return torch.stack([sel_pre, sel_post, sel_comb, base_pre, base_post, base_comb, RB, CB], dim=0)


# ---------------------------------------------------------------------------------------------
# router (HF modeling_motif.py:843-870)
# ---------------------------------------------------------------------------------------------
def router_golden(
    x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, k: int = 8, scale: float = 2.0, dtype=torch.float32
):
    """scores = σ(x·Wᵀ) (fp32), idx = topk(scores + bias), weights = renorm(scores[idx]) · scale.

    x [T, H], w [E, H], bias [E]. Returns (idx [T, k] sorted by biased score desc, weights [T, k],
    biased scores [T, E]).
    """
    logits = x.to(dtype) @ w.to(dtype).T
    s = torch.sigmoid(logits)
    biased = s + bias.to(dtype)
    _, idx = torch.topk(biased, k=k, dim=-1)
    top = s.gather(-1, idx)
    top = top / (top.sum(dim=-1, keepdim=True) + 1e-20) * scale
    return idx, top, biased


# ---------------------------------------------------------------------------------------------
# PolyNorm (HF modeling_motif.py:89-136, GroupedPolyNorm.forward_single)
# ---------------------------------------------------------------------------------------------
def poly_rms(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return z / torch.sqrt(z.pow(2).mean(-1, keepdim=True) + eps)


def grouped_polynorm_golden(g: torch.Tensor, u: torch.Tensor, c: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    """Per-expert PolyNorm·up in fp32, *without* the ×0.5 output scale (folded into W_down on TT).

    g, u [E, T, I]; c [E, 3] = σ(weight) (already sigmoided); b [E] = clamp(bias, ±0.5).
    h = (c0·N(g³) + c1·N(g²) + c2·N(g) + b) · u
    """
    gf = g.float()
    uf = u.float()
    c = c.float()[:, None, None, :]
    b = b.float()[:, None, None]
    poly = c[..., 0] * poly_rms(gf**3, eps) + c[..., 1] * poly_rms(gf**2, eps) + c[..., 2] * poly_rms(gf, eps) + b
    return poly * uf


# ---------------------------------------------------------------------------------------------
# RoPE (HF half-split / NeoX, modeling_motif.py:270-434; 01 §3.10)
# ---------------------------------------------------------------------------------------------
def plain_inv_freq(dim: int = ROPE_DIM, theta: float = ROPE_THETA) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def yarn_inv_freq(
    dim: int = ROPE_DIM,
    theta: float = ROPE_THETA,
    factor: float = YARN_FACTOR,
    orig: int = YARN_ORIG,
    beta_fast: float = YARN_BETA_FAST,
    beta_slow: float = YARN_BETA_SLOW,
) -> torch.Tensor:
    """Motif/DeepSeek YaRN inv_freq: f_i/factor·r_i + f_i·(1−r_i), ramp over the correction range [10, 23]."""

    def corr_dim(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(corr_dim(beta_fast)), 0)
    high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
    f = plain_inv_freq(dim, theta)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    return f / factor * ramp + f * (1 - ramp)


def cos_sin(inv_freq: torch.Tensor, positions: torch.Tensor, round_bf16: bool = True):
    """HF tables: cos/sin = cat(freqs, freqs) with freqs = pos·inv_freq in fp32, rounded to bf16 (01 §3.10)."""
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    c, s = emb.cos(), emb.sin()
    if round_bf16:
        c, s = c.bfloat16().float(), s.bfloat16().float()
    return c, s


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def rope_golden(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """y = x·cos + rotate_half(x)·sin in fp32 (HF math with bf16-rounded tables)."""
    return x.float() * cos + rotate_half(x.float()) * sin


# ---------------------------------------------------------------------------------------------
# optional cross-check against the in-progress CPU reference package (never fails a gate)
# ---------------------------------------------------------------------------------------------
def reference_crosscheck(name: str, fn_local, *args, **kwargs) -> str:
    """Compare a local golden against ``models.demos.motif3.reference`` if the matching function exists."""
    try:
        if name == "sinkhorn":
            from models.demos.motif3.reference.modules import sinkhorn as ref_fn  # lazy, CPU-only

            got = ref_fn(*args, **kwargs)
        elif name == "yarn_inv_freq":
            from models.demos.motif3.reference import rope as ref_rope  # lazy, CPU-only

            got = ref_rope.yarn_inv_freq(*args, **kwargs)
        else:
            return "not-wired"
        want = fn_local(*args, **kwargs)
        d = float((got.double() - want.double()).abs().max())
        return f"max|ref-local|={d:.3e}"
    except Exception as e:  # API in flux: record and move on
        return f"unavailable ({type(e).__name__}: {str(e)[:80]})"
