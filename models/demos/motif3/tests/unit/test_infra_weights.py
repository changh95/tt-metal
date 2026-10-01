# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for tt/weights.py: the torch-side transforms reproduce the reference math.

* GDLA: absorbed decode form built from the per-chip TP8 weights (split wkv_a, gamma_kv folded W_UK'/W_UV',
  split wq_b, local-first lambda, gate and wo shards), summed over the 8 chips, equals the HF expanded form
  (fp64, real head dims, global and SWA layers incl. window edges; also on real layer-0/1 weights).
* mHC: gamma-folded fused projection (and its per-stream device blocks) == rms_norm-then-projection.
* PolyNorm / MLP: TP-sharded gate|up, 3-moment statistics, x0.5 folded into down == HF MotifMLP.
* Router padding, per-chip shards reassemble, EP placement, lazy loader on the real checkpoint.

Run device-hidden with ``--noconftest`` (see test_infra_config.py docstring).
"""

import json
import struct

import pytest
import torch

from models.demos.motif3.tt import weights as W
from models.demos.motif3.tt.model_config import DEFAULT_HF_META_DIR, MeshAxes, MotifTTConfig
from models.demos.motif3.tt.rope import cos_sin_table, inv_freq_for_layer, rotate_half

CFG = MotifTTConfig.from_hf_config(str(DEFAULT_HF_META_DIR))
F64 = torch.float64


# ======================================================================================================
# helpers: HF-semantics GDLA in fp64 (modeling_motif.py:634-775), no dtype casts
# ======================================================================================================
def _rms(x, w=None, eps=1e-5):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def _rope64(x, cos, sin):
    return x * cos + rotate_half(x) * sin


def _allow(S, window):
    q = torch.arange(S)[:, None]
    k = torch.arange(S)[None, :]
    a = k <= q
    if window is not None:
        a = a & (k > q - window)
    return a


def _softmax(scores, allow):
    return scores.masked_fill(~allow, float("-inf")).softmax(-1)


def _tables(cfg, layer, S):
    cos, sin = cos_sin_table(inv_freq_for_layer(cfg, layer), torch.arange(S), torch.float32)
    return cos.to(F64), sin.to(F64)


def random_attention_weights(cfg, seed=0, dtype=F64):
    g = torch.Generator().manual_seed(seed)

    def lin(o, i):
        return torch.randn(o, i, generator=g, dtype=dtype) * i**-0.5

    def gam(n):
        return torch.rand(n, generator=g, dtype=dtype) + 0.5

    D, H = cfg.hidden_size, cfg.n_heads
    return {
        "wq_a": lin(cfg.q_lora_rank, D),
        "q_norm": gam(cfg.q_lora_rank),
        "wq_b": lin(H * cfg.head_dim, cfg.q_lora_rank),
        "wq_b_gate": lin(cfg.n_signal_heads * cfg.v_head_dim, cfg.q_lora_rank),
        "wkv_a": lin(cfg.kv_lora_rank + cfg.rope_dim, D),
        "kv_norm": gam(cfg.kv_lora_rank),
        "wkv_b": lin(cfg.n_kv_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank),
        "lambda_proj": lin(cfg.n_signal_heads, D),
        "wo": lin(D, cfg.n_signal_heads * cfg.v_head_dim),
    }


def gdla_expanded(a, P, cfg, layer):
    """HF ``MotifGDLAttention.forward`` (prefill over S tokens from position 0) in fp64."""
    S = a.shape[0]
    L = cfg.layer(layer)
    H, G, hd, nope, rd, v = (
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.qk_nope_head_dim,
        cfg.rope_dim,
        cfg.v_head_dim,
    )
    eps = cfg.rms_norm_eps
    cq = _rms(a @ P["wq_a"].T, P["q_norm"], eps)
    q = (cq @ P["wq_b"].T).view(S, H, hd)
    gate = (cq @ P["wq_b_gate"].T).view(S, cfg.n_signal_heads, v)
    q_nope, q_pe = q[..., :nope], q[..., nope:]
    kv = a @ P["wkv_a"].T
    c_raw, kpe = kv[:, : cfg.kv_lora_rank], kv[:, cfg.kv_lora_rank :]
    cos, sin = _tables(cfg, layer, S)
    q_pe = _rope64(q_pe, cos[:, None], sin[:, None])
    kpe = _rope64(kpe, cos, sin)
    c = _rms(c_raw, P["kv_norm"], eps)
    kvb = (c @ P["wkv_b"].T).view(S, G, nope + v)
    k_nope, vv = kvb[..., :nope], kvb[..., nope:]
    K = torch.cat([k_nope, kpe[:, None].expand(S, G, rd)], dim=-1)
    Q = torch.cat([q_nope, q_pe], dim=-1)
    r = cfg.heads_per_group
    scores = torch.einsum("shd,thd->hst", Q, K.repeat_interleave(r, dim=1)) * L.softmax_scale
    Pm = _softmax(scores, _allow(S, L.window))
    O = torch.einsum("hst,thd->shd", Pm, vv.repeat_interleave(r, dim=1))
    Og = O.view(S, G, r, v)
    sig = Og[:, :, : cfg.grouped_ratio].reshape(S, cfg.n_signal_heads, v)
    noi = Og[:, :, cfg.grouped_ratio].repeat_interleave(cfg.grouped_ratio, dim=1)
    lam = a @ P["lambda_proj"].T
    D = (sig - torch.sigmoid(lam)[..., None] * noi) * torch.sigmoid(gate)
    out = D.reshape(S, -1) @ P["wo"].T
    return out, {"scores": scores, "k_nope": k_nope, "v": vv, "c": c}


def gdla_absorbed_tp(a, P, cfg, layer, *, check=None):
    """Design §2.3.4 decode dataflow (absorbed, latent cache) per TP chip with tt/weights.py transforms,
    summed over the chips (= all_reduce(tp))."""
    S = a.shape[0]
    L = cfg.layer(layer)
    eps = cfg.rms_norm_eps
    cos, sin = _tables(cfg, layer, S)
    Hc, Gc, Sc = cfg.q_heads_per_chip, cfg.kv_groups_per_chip, cfg.signal_heads_per_chip
    nope, rd, r, v, R = cfg.qk_nope_head_dim, cfg.rope_dim, cfg.heads_per_group, cfg.v_head_dim, cfg.kv_lora_rank
    q0, c0, k0, l0 = 0, cfg.q_lora_rank, cfg.q_lora_rank + R, cfg.q_lora_rank + R + rd
    out = torch.zeros(S, cfg.hidden_size, dtype=a.dtype)
    allow = _allow(S, L.window)
    for tp in range(cfg.tp):
        lat = a @ W.latent_projection_for_chip(P["wq_a"], P["wkv_a"], P["lambda_proj"], cfg, tp)  # [S, 1664]
        cq = _rms(lat[:, q0:c0], P["q_norm"], eps)
        q = cq @ W.wq_b_for_chip(P["wq_b"], cfg, tp, layout="split")  # [S, 1920]
        q_nope = q[:, : Hc * nope].view(S, Hc, nope)
        q_pe = _rope64(q[:, Hc * nope :].view(S, Hc, rd), cos[:, None], sin[:, None])
        gate = torch.sigmoid(cq @ W.wq_b_gate_for_chip(P["wq_b_gate"], cfg, tp)).view(S, Sc, v)
        n = _rms(lat[:, c0:k0], None, eps)  # weightless kv_norm: the gamma-free cached latent
        kpe = _rope64(lat[:, k0:l0], cos, sin)
        lam = torch.sigmoid(lat[:, l0 : l0 + Sc])  # local signal heads, same slice on every chip
        w_uk = W.absorb_weights_for_chip(P["wkv_b"], P["kv_norm"], cfg, tp)  # [2, 128, 512]
        q_lat = torch.einsum("sgjn,gnr->sgjr", q_nope.view(S, Gc, r, nope), w_uk).reshape(S, Hc, R)
        Qa = torch.cat([q_lat, q_pe], dim=-1)  # [S, 10, 576]
        cache = torch.cat([n, kpe], dim=-1)  # [S, 576]
        scores = torch.einsum("shd,td->hst", Qa, cache) * L.softmax_scale
        o_lat = torch.einsum("hst,tr->shr", _softmax(scores, allow), n)  # V = first 512 cache columns
        ol = o_lat.view(S, Gc, r, R)
        d_lat = ol[:, :, : cfg.grouped_ratio] - lam.view(S, Gc, cfg.grouped_ratio, 1) * ol[:, :, cfg.grouped_ratio :]
        w_uv_t = W.unabsorb_weights_for_chip(P["wkv_b"], P["kv_norm"], cfg, tp)  # [2, 512, 128]
        Dh = torch.einsum("sgjr,grv->sgjv", d_lat, w_uv_t).reshape(S, Sc, v) * gate
        out = out + Dh.reshape(S, Sc * v) @ W.wo_for_chip(P["wo"], cfg, tp)
        if check is not None:
            check(tp=tp, cq=cq, q_nope=q_nope, q_lat=q_lat, scores=scores, n=n)
    return out


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


# ======================================================================================================
# GDLA
# ======================================================================================================
@pytest.mark.parametrize("layer", [0, 1], ids=["global_yarn", "swa_plain_w129"])
def test_absorbed_equals_expanded_random_real_dims(layer):
    P = random_attention_weights(CFG, seed=layer)
    S = 140  # covers SWA window edges at positions 127..139
    a = torch.randn(S, CFG.hidden_size, generator=torch.Generator().manual_seed(7), dtype=F64)
    ref, aux = gdla_expanded(a, P, CFG, layer)

    def check(tp, cq, q_nope, q_lat, scores, n):
        heads = CFG.chip_heads(tp).q_heads
        # absorbed logits == expanded logits for the chip's 10 heads (incl. masked entries)
        m = _allow(S, CFG.layer(layer).window)
        assert _rel(scores[:, m], aux["scores"][heads.start : heads.stop][:, m]) < 1e-11
        # per-head absorb and fully absorbed q projection give the same q_lat
        q2 = torch.einsum("shn,hnr->shr", q_nope, W.absorb_weights_per_head_for_chip(P["wkv_b"], P["kv_norm"], CFG, tp))
        assert _rel(q2, q_lat) < 1e-12
        if tp == 0:
            q3 = torch.einsum(
                "si,hir->shr",
                cq,
                W.absorbed_q_weights_per_head_for_chip(P["wq_b"], P["wkv_b"], P["kv_norm"], CFG, tp),
            )
            assert _rel(q3, q_lat) < 1e-11
        # prefill expansion from the latent reproduces expanded K_nope / V (V zero-padded to 192)
        blk = (n @ W.prefill_kv_expansion_for_chip(P["wkv_b"], P["kv_norm"], CFG, tp)).view(S, 2, 320)
        for gl, g in enumerate(CFG.chip_heads(tp).groups):
            assert _rel(blk[:, gl, :128], aux["k_nope"][:, g]) < 1e-12
            assert _rel(blk[:, gl, 128:256], aux["v"][:, g]) < 1e-12
            assert torch.count_nonzero(blk[:, gl, 256:]) == 0

    out = gdla_absorbed_tp(a, P, CFG, layer, check=check)
    assert _rel(out, ref) < 1e-10


@pytest.mark.parametrize("layer", [0, 1])
def test_absorbed_equals_expanded_real_weights(layer):
    loader = W.HFWeightLoader()
    if not loader.layer_available(layer):
        pytest.skip(f"layer {layer} not local")
    p = f"model.layers.{layer}.self_attn."
    P = {
        k: loader.get(p + k + ".weight", F64)
        for k in ("wq_a", "q_norm", "wq_b", "wq_b_gate", "wkv_a", "kv_norm", "wkv_b", "lambda_proj", "wo")
    }
    S = 136
    a = torch.randn(S, CFG.hidden_size, generator=torch.Generator().manual_seed(3), dtype=F64)
    ref, _ = gdla_expanded(a, P, CFG, layer)
    out = gdla_absorbed_tp(a, P, CFG, layer)
    assert _rel(out, ref) < 1e-10


def test_absorbed_matches_reference_module():
    """Our TP8 absorbed form vs the CPU golden GDLAttention (expanded mode). The reference keeps fp32 norm
    statistics even for fp64 inputs, hence the fp32-level tolerance."""
    ref_mod = pytest.importorskip("models.demos.motif3.reference.modules")
    ref_cfg = pytest.importorskip("models.demos.motif3.reference.config")
    args = ref_cfg.MotifArgs.from_hf_config(str(DEFAULT_HF_META_DIR)).replace(q_path_fp32=False)
    layer = 1
    m = ref_mod.GDLAttention(args, layer).to(F64)
    P = random_attention_weights(CFG, seed=11)
    sd = {k + ".weight": P[k] for k in P}
    sd = {("q_norm.weight" if k == "q_norm.weight" else k): t for k, t in sd.items()}
    m.load_state_dict({f"{k}": v for k, v in sd.items()}, strict=True)
    S = 40
    a = torch.randn(S, CFG.hidden_size, generator=torch.Generator().manual_seed(5), dtype=F64)
    with torch.no_grad():
        ref = m(a[None], torch.arange(S)[None], cache=None, mode="expanded")[0]
    out = gdla_absorbed_tp(a, P, CFG, layer)
    assert _rel(out, ref) < 1e-5


def test_attention_shards_reassemble():
    P = random_attention_weights(CFG, seed=2, dtype=torch.float32)
    # wq_b interleaved == HF rows; split == per-chip column permutation of it
    full = W.stack_tp(lambda tp: W.wq_b_for_chip(P["wq_b"], CFG, tp, layout="interleaved"), CFG, dim=1)
    assert torch.equal(full, P["wq_b"].T)
    for tp in (0, 5):
        inter = W.wq_b_for_chip(P["wq_b"], CFG, tp, layout="interleaved")
        split = W.wq_b_for_chip(P["wq_b"], CFG, tp, layout="split")
        perm = [h * 192 + d for h in range(10) for d in range(128)] + [
            h * 192 + 128 + d for h in range(10) for d in range(64)
        ]
        assert torch.equal(split, inter[:, perm])
    assert torch.equal(
        W.stack_tp(lambda tp: W.wq_b_gate_for_chip(P["wq_b_gate"], CFG, tp), CFG, dim=1), P["wq_b_gate"].T
    )
    assert torch.equal(W.stack_tp(lambda tp: W.wo_for_chip(P["wo"], CFG, tp), CFG, dim=0), P["wo"].T)
    # latent projection: identical cq / kv columns on every chip, local lambda first
    base = W.latent_projection(P["wq_a"], P["wkv_a"], P["lambda_proj"])
    assert base.shape == (4096, 1664)
    for tp in range(CFG.tp):
        lp = W.latent_projection_for_chip(P["wq_a"], P["wkv_a"], P["lambda_proj"], CFG, tp)
        assert torch.equal(lp[:, :1600], base[:, :1600])
        assert torch.equal(lp[:, 1600:1608], P["lambda_proj"][8 * tp : 8 * tp + 8].T)
        assert sorted(W.signal_order_for_chip(CFG, tp)) == list(range(64))
    dkv, kr = W.split_wkv_a(P["wkv_a"], CFG)
    assert torch.equal(torch.cat([dkv, kr]), P["wkv_a"]) and dkv.shape == (512, 4096) and kr.shape == (64, 4096)
    uq, qr = W.split_wq_b(P["wq_b"], CFG)
    assert uq.shape == (80, 128, 1024) and qr.shape == (80, 64, 1024)
    assert torch.equal(uq[37], P["wq_b"][37 * 192 : 37 * 192 + 128]) and torch.equal(
        qr[37], P["wq_b"][37 * 192 + 128 : 38 * 192]
    )
    uk, uv = W.split_wkv_b(P["wkv_b"], CFG)
    fk, fv = W.fold_kv_norm(uk, uv, P["kv_norm"])
    assert torch.equal(torch.cat([W.kv_up_for_chip(P["wkv_b"], P["kv_norm"], CFG, tp)[0] for tp in range(8)]), fk)
    assert torch.equal(torch.cat([W.kv_up_for_chip(P["wkv_b"], P["kv_norm"], CFG, tp)[1] for tp in range(8)]), fv)
    assert torch.equal(uk[3], P["wkv_b"][3 * 256 : 3 * 256 + 128]) and torch.equal(
        uv[3], P["wkv_b"][3 * 256 + 128 : 4 * 256]
    )


# ======================================================================================================
# mHC
# ======================================================================================================
def test_mhc_folded_projection():
    g = torch.Generator().manual_seed(0)
    T, E, D = 6, 4, 4096
    X = torch.randn(T, E, D, generator=g, dtype=F64).to(torch.bfloat16).to(F64)  # bf16 residual streams
    Wp = torch.randn(4, E * D, generator=g, dtype=F64) * (E * D) ** -0.5
    Wq = torch.randn(4, E * D, generator=g, dtype=F64) * (E * D) ** -0.5
    Wr = torch.randn(16, E * D, generator=g, dtype=F64) * (E * D) ** -0.5
    gamma = torch.randn(E * D, generator=g, dtype=F64) * 0.1 + 0.1  # real gammas: mean 0.06-0.14, some < 0
    Xf = X.reshape(T, E * D)
    inv_rms = torch.rsqrt(Xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = (gamma * (Xf * inv_rms)) @ torch.cat([Wp, Wq, Wr]).T  # rms_norm(x; gamma) @ W^T  (HF math, fp64)

    fn = W.mhc_fused_projection(Wp, Wq, Wr, gamma)
    assert fn.shape == (24, E * D) and fn.dtype == F64
    folded = (Xf @ fn.T) * inv_rms
    assert _rel(folded, ref) < 1e-12

    # device form: per-stream blocks [1,4,4096,32]; sum over the stream dim of X_i @ B_i; ss = sum_i sum_d X^2
    blocks = W.mhc_projection_blocks(fn)
    assert blocks.shape == (1, 4, D, 32)
    mixes_un = (X.permute(1, 0, 2).unsqueeze(0) @ blocks).sum(dim=1)[0]  # [T, 32]
    assert torch.count_nonzero(mixes_un[:, 24:]) == 0
    ss = (X * X).sum(-1).sum(-1, keepdim=True)
    dev = mixes_un[:, :24] * torch.rsqrt(ss / (E * D) + 1e-6)
    assert _rel(dev, ref) < 1e-12

    # one bf16 rounding of the folded weight vs HF's bf16 path: same order of error (informational bound)
    hf_bf16 = (
        (gamma.to(torch.bfloat16) * (Xf * inv_rms).to(torch.bfloat16)).float()
        @ torch.cat([Wp, Wq, Wr]).to(torch.bfloat16).float().T
    ).to(F64)
    tt_bf16 = (Xf.float() @ fn.to(torch.bfloat16).float().T).to(F64) * inv_rms
    assert _rel(tt_bf16, ref) < 1e-2 and _rel(hf_bf16, ref) < 1e-2

    # reference cross-check (fp32 folded weight)
    ref_mod = pytest.importorskip("models.demos.motif3.reference.modules")
    m = ref_mod.MHCLayer(4, D)
    with torch.no_grad():
        m.proj_merged.weight.copy_(torch.cat([Wp, Wq, Wr]).float())
        m.rms_norm.weight.copy_(gamma.float())
    assert torch.allclose(m.folded_mix_weight(), fn.float(), rtol=1e-6, atol=1e-9)


def test_mhc_source_and_scalars():
    g = torch.Generator().manual_seed(1)
    pre, post, res = (torch.randn(n, 16384, generator=g) for n in (4, 4, 16))
    p = "model.layers.3.mhc_attn"
    sd = {
        f"{p}.proj_pre.weight": pre,
        f"{p}.proj_post.weight": post,
        f"{p}.proj_res.weight": res,
        f"{p}.alpha_pre": torch.tensor([0.1]).to(torch.bfloat16),
        f"{p}.alpha_post": torch.tensor([0.2]),
        f"{p}.alpha_res": torch.tensor([0.3]),
        f"{p}.bias_pre": torch.arange(4.0),
        f"{p}.bias_post": torch.arange(4.0) + 4,
        f"{p}.bias_res": torch.arange(16.0).reshape(4, 4),
    }
    src = W.DictWeightSource(sd)
    assert torch.equal(W.mhc_projection_from_source(src, p), torch.cat([pre, post, res]))
    merged = W.DictWeightSource({f"{p}.proj_merged.weight": torch.cat([pre, post, res])})
    assert torch.equal(W.mhc_projection_from_source(merged, p), torch.cat([pre, post, res]))
    sc = W.mhc_scalars(src, p)
    assert sc["alpha_pre"].dtype == torch.float32 and sc["bias_res"].tolist() == list(range(16))  # row-major 4i+j


# ======================================================================================================
# PolyNorm / MLP / experts / router
# ======================================================================================================
def _poly_ref(gx, w, b, eps=1e-6):
    def N(z):
        return z / torch.sqrt(z.pow(2).mean(-1, keepdim=True) + eps)

    c = torch.sigmoid(w)
    return c[0] * N(gx**3) + c[1] * N(gx**2) + c[2] * N(gx) + b


@pytest.mark.parametrize("inter", [384, 1280], ids=["dense_like", "shared_1280"])
def test_tp_mlp_polynorm_fold(inter):
    g = torch.Generator().manual_seed(inter)
    T, D = 8, 256
    f = torch.randn(T, D, generator=g, dtype=F64)
    gate = torch.randn(inter, D, generator=g, dtype=F64) * D**-0.5
    up = torch.randn(inter, D, generator=g, dtype=F64) * D**-0.5
    down = torch.randn(D, inter, generator=g, dtype=F64) * inter**-0.5
    w = torch.randn(3, generator=g, dtype=F64)
    b = torch.tensor([0.8], dtype=F64)  # > 0.5: dense/shared bias must NOT be clamped
    # HF MotifMLP: h = bf16(poly(g) * up) * 0.5 ; out = h @ down^T   (fp64 here, no rounding)
    gx, ux = f @ gate.T, f @ up.T
    ref = (_poly_ref(gx, w, b) * ux * 0.5) @ down.T

    c, bb = W.polynorm_coefficients(w, b)  # no clamp for dense/shared
    assert torch.equal(bb, b)
    n = inter // CFG.tp
    m2 = torch.zeros(T, 1, dtype=F64)
    m4 = torch.zeros(T, 1, dtype=F64)
    m6 = torch.zeros(T, 1, dtype=F64)
    parts = []
    for tp in range(CFG.tp):  # local moments, then all_reduce(tp) of [T, 3]
        gu = f @ W.mlp_gate_up_for_chip(gate, up, CFG, tp)
        assert gu.shape == (T, 2 * n)
        gl, ul = gu[:, :n], gu[:, n:]
        m2, m4, m6 = (
            m2 + (gl**2).sum(-1, keepdim=True),
            m4 + (gl**4).sum(-1, keepdim=True),
            m6 + (gl**6).sum(-1, keepdim=True),
        )
        parts.append((gl, ul, tp))
    eps = CFG.polynorm_eps
    r2, r4, r6 = (torch.rsqrt(m / inter + eps) for m in (m2, m4, m6))
    out = torch.zeros(T, D, dtype=F64)
    for gl, ul, tp in parts:
        poly = c[0] * gl**3 * r6 + c[1] * gl**2 * r4 + c[2] * gl * r2 + bb
        out = out + (poly * ul) @ W.mlp_down_for_chip(down, CFG, tp)  # x0.5 folded
    assert _rel(out, ref) < 1e-12
    full_down = W.stack_tp(lambda tp: W.mlp_down_for_chip(down, CFG, tp), CFG, dim=0)
    assert torch.equal(full_down, down.T * 0.5)


def test_routed_experts_transforms_and_ep_layout():
    g = torch.Generator().manual_seed(4)
    n, I, D = 3, 16, 24  # shape-agnostic transforms on a small stand-in for [384, 2560, 4096]
    gate_up = torch.randn(n, 2 * I, D, generator=g).to(torch.bfloat16)
    down = torch.randn(n, D, I, generator=g).to(torch.bfloat16)
    x = torch.randn(5, D, generator=g).to(torch.bfloat16)
    gu_t = W.experts_gate_up(gate_up)
    dn_t = W.experts_down(down)
    assert gu_t.dtype == torch.bfloat16 and gu_t.shape == (n, D, 2 * I) and dn_t.shape == (n, I, D)
    for e in range(n):
        # gate = first half of the output columns (HF ``gate_up.chunk(2, -1)``), up = second half; exact transpose
        assert torch.equal(gu_t[e][:, :I], gate_up[e][:I].T) and torch.equal(gu_t[e][:, I:], gate_up[e][I:].T)
        ref_gate = x.float() @ gate_up[e].float().T
        assert torch.allclose((x.float() @ gu_t[e].float())[:, :I], ref_gate[:, :I], rtol=1e-6, atol=1e-5)
        assert torch.equal(dn_t[e].float(), down[e].float().T * 0.5)  # exact x0.5 fold

    # GroupedPolyNorm constants: sigmoid(w), bias clamped +-0.5 (routed only), EP layout [dp, 96, 1, 1]
    wE = torch.randn(384, 3, generator=g)
    bE = torch.linspace(-1, 1, 384).reshape(384, 1)
    t = W.expert_polynorm_tensors(wE, bE, CFG)
    assert t["c0"].shape == (4, 96, 1, 1)
    flat_b = t["b"].reshape(-1)
    assert flat_b.max().item() == 0.5 and flat_b.min().item() == -0.5
    assert torch.equal(t["c2"].reshape(-1), torch.sigmoid(wE)[:, 2])
    assert torch.equal(t["b"].reshape(-1), bE.reshape(-1).clamp(-0.5, 0.5))
    # EP layout: chip (dp, tp) slab [12k, 12k+12) with k = 8 dp + tp  -> flat order == expert order
    assert torch.equal(t["c0"].reshape(-1), torch.sigmoid(wE)[:, 0])


@pytest.mark.parametrize("mesh_shape", [(4, 8), (8, 4)])
def test_ep_placement_and_mapper_emulation(mesh_shape):
    cfg = MotifTTConfig.from_hf_config(str(DEFAULT_HF_META_DIR), mesh_shape=mesh_shape)
    ids = W.local_expert_ids(cfg)  # [4, 96, 1, 1]
    payload = W.ep_layout(torch.arange(384 * 2, dtype=torch.float32).reshape(384, 2), cfg)
    R, C = mesh_shape
    for row in range(R):
        for col in range(C):
            dp, tp = cfg.axes.roles(row, col)
            k = 8 * dp + tp
            loc = W.shard_for_device(ids, cfg.axes, row, col, dp_dim=0, tp_dim=1)
            assert loc.shape == (1, 12, 1, 1)
            assert loc.reshape(-1).tolist() == list(range(12 * k, 12 * k + 12)) == list(cfg.experts_of_chip(dp, tp))
            pl = W.shard_for_device(payload, cfg.axes, row, col, dp_dim=0, tp_dim=1)
            assert torch.equal(pl[0, :, 0], torch.arange(12 * k, 12 * k + 12, dtype=torch.float32) * 2)
    # TP-only / DP-only / replicate emulation
    t = torch.arange(4 * 16).reshape(4, 16)
    for row in range(R):
        for col in range(C):
            dp, tp = cfg.axes.roles(row, col)
            assert torch.equal(W.shard_for_device(t, cfg.axes, row, col, tp_dim=1), t[:, 2 * tp : 2 * tp + 2])
            assert torch.equal(W.shard_for_device(t, cfg.axes, row, col, dp_dim=0), t[dp : dp + 1])
            assert torch.equal(W.shard_for_device(t, cfg.axes, row, col), t)
    assert W.mapping_tag() == "rep" and W.mapping_tag(tp_dim=1) == "tp1" and W.mapping_tag(0, 1) == "dp0tp1"


def test_lm_head_and_norm_shards():
    cfg = MotifTTConfig(vocab_size=2048)
    lm = torch.randn(2048, 64)
    assert torch.equal(W.stack_tp(lambda tp: W.lm_head_for_chip(lm, cfg, tp), cfg, dim=1), lm.T)
    gamma = torch.randn(4096)
    nw = W.norm_weight(gamma)
    assert nw.shape == (1, 1, 128, 32) and torch.equal(nw.reshape(-1), gamma)


def test_router_padding_and_semantics():
    g = torch.Generator().manual_seed(9)
    x = torch.randn(32, 256, generator=g)
    Wr = torch.randn(384, 256, generator=g) * 256**-0.5
    bias = (1.1 + 0.03 * torch.randn(384, generator=g)).to(torch.bfloat16)  # stored bf16, used in fp32
    w_t, b = W.router_weights(Wr, bias)
    w_p, b_p = W.router_weights(Wr, bias, pad_to=512)
    assert w_t.shape == (256, 384) and w_p.shape == (256, 512) and b.dtype == torch.float32
    s = torch.sigmoid(x @ w_t)
    s_p = torch.sigmoid(x @ w_p)
    idx = torch.topk(s + b, 8, dim=-1).indices
    idx_p = torch.topk(s_p + b_p, 8, dim=-1).indices
    assert torch.equal(idx.sort(-1).values, idx_p.sort(-1).values) and int(idx_p.max()) < 384
    # HF TokenChoiceTopKRouter: unbiased scores of the selected experts, renormalized, x route_scale
    hf_scores = torch.sigmoid(torch.nn.functional.linear(x.float(), Wr.float()))
    hf_idx = torch.topk(hf_scores + bias.float(), 8, dim=1).indices
    hf_w = hf_scores.gather(1, hf_idx)
    hf_w = hf_w / (hf_w.sum(-1, keepdim=True) + 1e-20) * 2.0
    ours_w = s.gather(1, idx)
    ours_w = ours_w / (ours_w.sum(-1, keepdim=True) + 1e-20) * CFG.route_scale
    assert torch.equal(idx, hf_idx) and torch.allclose(ours_w, hf_w, atol=1e-6)


# ======================================================================================================
# loader
# ======================================================================================================
def _loader_or_skip():
    try:
        return W.HFWeightLoader()
    except FileNotFoundError:
        pytest.skip("checkpoint index not present")


def test_loader_index_and_layers():
    ld = _loader_or_skip()
    assert len(ld.keys()) == 2236
    assert 0 in ld.available_layers(3) or not ld.layer_available(0)
    if not ld.layer_available(0):
        pytest.skip("layer 0 not local")
    p = "model.layers.0.self_attn."
    shapes = {
        k: ld.shape(p + k + ".weight") for k in ("wq_a", "wq_b", "wq_b_gate", "wkv_a", "wkv_b", "lambda_proj", "wo")
    }
    assert shapes == {
        "wq_a": (1024, 4096),
        "wq_b": (15360, 1024),
        "wq_b_gate": (8192, 1024),
        "wkv_a": (576, 4096),
        "wkv_b": (4096, 512),
        "lambda_proj": (64, 4096),
        "wo": (4096, 8192),
    }
    assert ld.get(p + "kv_norm.weight").dtype == torch.bfloat16
    assert set(ld.download_state_layers()) >= {0, 1, 2} or not ld.download_state_layers()
    with pytest.raises(W.MissingWeightError):
        ld.get("model.layers.0.not_a_tensor")


def test_loader_expert_rows_match_raw_bytes():
    ld = _loader_or_skip()
    if not ld.layer_available(2):
        pytest.skip("layer 2 not local")
    name = "model.layers.2.moe.experts.down_proj"  # [384, 4096, 1280] bf16
    e0, e1 = 25, 27
    rows = ld.get_rows(name, e0, e1)
    assert rows.shape == (2, 4096, 1280) and rows.dtype == torch.bfloat16
    # independent read: safetensors header -> byte offsets of experts e0..e1
    path = ld.dir / ld.shard_of(name)
    with open(path, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
        start = 8 + hlen + hdr[name]["data_offsets"][0]
        per = 4096 * 1280 * 2
        fh.seek(start + e0 * per)
        raw = bytearray(fh.read((e1 - e0) * per))
    ref = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(2, 4096, 1280)
    assert torch.equal(rows, ref)
    small = "model.layers.2.moe.experts.act_fn.weight"
    assert torch.equal(ld.get_rows(small, 12, 24), ld.get(small)[12:24])


def test_dict_source_api():
    src = W.DictWeightSource({"a": torch.arange(6.0).reshape(3, 2)})
    assert src.has("a") and "a" in src and src.shape("a") == (3, 2)
    assert torch.equal(src.get_rows("a", 1, 3), torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
    assert src.get("a", torch.float64).dtype == torch.float64
    with pytest.raises(W.MissingWeightError):
        src.get("b")
