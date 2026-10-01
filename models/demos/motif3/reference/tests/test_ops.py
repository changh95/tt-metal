# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of the reference ops against the HF classes (bit-exact) and against their stated semantics."""

import pytest
import torch

from models.demos.motif3.reference import (
    MLP,
    GroupedPolyNorm,
    MHCLayer,
    MoE,
    PolyNorm,
    RMSNorm,
    Router,
    sinkhorn,
    tiny_random_args,
)

from .common import max_abs
from .hf_reference import load_hf_modules

DTYPES = [torch.float32, torch.bfloat16]
IDS = ["fp32", "bf16"]


@pytest.fixture(scope="module")
def mm():
    return load_hf_modules()[1]


def _g(seed):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_rmsnorm_matches_hf(mm, dtype):
    ref, hf = RMSNorm(64, 1e-5), mm.MotifRMSNorm(64, eps=1e-5)
    w = (torch.rand(64, generator=_g(0)) + 0.5).to(dtype)
    ref.weight.data, hf.weight.data = w.clone(), w.clone()
    x = (torch.randn(3, 7, 64, generator=_g(1)) * 3).to(dtype)
    assert torch.equal(ref(x), hf(x)) and ref(x).dtype == dtype
    # fp32 input with a bf16 gamma promotes to fp32 (the HF q path)
    ref.weight.data = w.to(torch.bfloat16)
    assert ref(x.float()).dtype == torch.float32


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_dense_polynorm_matches_hf_and_does_not_clamp_bias(mm, dtype):
    ref, hf = PolyNorm(), mm.PolyNormTorch()
    w, b = torch.tensor([-0.7, 0.2, 1.1]), torch.tensor([0.9])  # |b| > 0.5: must NOT be clamped on dense/shared
    ref.weight.data, ref.bias.data, hf.weight.data, hf.bias.data = w.clone(), b.clone(), w.clone(), b.clone()
    g = (torch.randn(5, 96, generator=_g(2)) * 4).to(dtype)
    u = torch.randn(5, 96, generator=_g(3)).to(dtype)
    out = ref.forward_mul(g, u)
    assert torch.equal(out, hf.forward_mul(g, u)) and out.dtype == dtype
    # explicit formula, fp32 math, single downcast
    gf = g.float()
    n = lambda z: z / torch.sqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)  # noqa: E731
    s = torch.sigmoid(w)
    expect = ((s[0] * n(gf**3) + s[1] * n(gf**2) + s[2] * n(gf) + b) * u.float()).to(dtype)
    assert max_abs(out, expect) <= 1e-6 * float(expect.abs().max().float())


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_grouped_polynorm_matches_hf_and_clamps_bias(mm, dtype):
    E = 4
    kw = dict(bias_clamp=0.5, output_scale=0.5, hidden_clamp=1e6)
    ref, hf = GroupedPolyNorm(E, **kw), mm.GroupedPolyNorm(E, **kw)
    w = torch.randn(E, 3, generator=_g(4))
    b = torch.tensor([[0.9], [-0.8], [0.2], [-0.1]])
    ref.weight.data, ref.bias.data, hf.weight.data, hf.bias.data = w.clone(), b.clone(), w.clone(), b.clone()
    g = (torch.randn(6, 80, generator=_g(5)) * 3).to(dtype)
    u = torch.randn(6, 80, generator=_g(6)).to(dtype)
    for e in range(E):
        assert torch.equal(ref.forward_single(g, u, e), hf.forward_single(g, u, e))
    with torch.no_grad():
        _, bias0 = ref.coefficients(0)
        _, bias1 = ref.coefficients(1)
    assert float(bias0) == 0.5 and float(bias1) == -0.5
    # clamped-bias expert == unclamped module with the bias pre-clamped
    plain = GroupedPolyNorm(E, bias_clamp=None, output_scale=0.5, hidden_clamp=1e6)
    plain.weight.data, plain.bias.data = w.clone(), b.clamp(-0.5, 0.5)
    assert torch.equal(ref.forward_single(g, u, 0), plain.forward_single(g, u, 0))


def test_sinkhorn_matches_hf_and_is_column_stochastic(mm):
    hf = mm.MHCLayer(4, 8)
    for std in (0.5, 3.0, 30.0):
        logits = torch.randn(64, 4, 4, generator=_g(7)) * std
        out = sinkhorn(logits, 20)
        assert torch.equal(out, hf._sinkhorn_knopp_batch(logits))
        assert torch.allclose(out.sum(-2), torch.ones(64, 4), atol=1e-6)  # columns exact (last step)
        assert (out >= 0).all()
    # with peaked logits 20 iterations leave rows only approximately normalized: iteration count is part of the spec
    peaked = sinkhorn(torch.randn(64, 4, 4, generator=_g(8)) * 10, 20)
    assert (peaked.sum(-1) - 1).abs().max() > 1e-3
    assert (sinkhorn(torch.randn(64, 4, 4, generator=_g(8)) * 10, 21) - peaked).abs().max() > 0


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_mhc_layer_matches_hf(mm, dtype):
    E, D = 4, 32
    ref = MHCLayer(E, D, sinkhorn_iters=20, h_post_coeff=1.0)
    hf = mm.MHCLayer(E, D, identity_init=False, sinkhorn_iters=20, h_post_coeff=1.0)
    g = _g(9)
    W = torch.randn(E * E + 2 * E, E * D, generator=g) / (E * D) ** 0.5
    params = dict(
        bias_pre=torch.randn(E, generator=g) * 0.3,
        bias_post=torch.randn(E, generator=g) * 0.3,
        bias_res=torch.randn(E, E, generator=g) * 0.3,
        alpha_pre=torch.tensor([0.4]),
        alpha_post=torch.tensor([0.5]),
        alpha_res=torch.tensor([0.6]),
    )
    gamma = torch.rand(E * D, generator=g) + 0.5
    ref.proj_merged.weight.data = W.clone()
    ref.rms_norm.weight.data = gamma.clone()
    hf.proj_pre.weight.data, hf.proj_post.weight.data, hf.proj_res.weight.data = W[:E], W[E : 2 * E], W[2 * E :]
    hf.rms_norm.weight.data = gamma.clone()
    for k, v in params.items():
        getattr(ref, k).data = v.clone()
        getattr(hf, k).data = v.clone()
    ref, hf = ref.to(dtype), hf.to(dtype)
    x = torch.randn(2, 5, E, D, generator=g).to(dtype)
    for a, b in zip(ref(x), hf(x)):
        assert torch.equal(a, b) and a.dtype == torch.float32
    h_pre, h_post, h_res = ref(x)
    assert torch.equal(MHCLayer.pre(x, h_pre), mm.MHCLayer.apply_h_pre(x, h_pre))
    out = torch.randn(2, 5, D, generator=g).to(dtype)
    expect = (torch.einsum("bsij,bsjd->bsid", h_res, x.float()) + h_post.unsqueeze(-1) * out.float().unsqueeze(2)).to(
        dtype
    )
    assert torch.equal(MHCLayer.post(x, out, h_post, h_res), expect)
    # fork (tilelang) precision: identical in fp32 up to rounding, close in bf16
    ref.mix_fp32 = True
    for a, b in zip(ref(x), hf(x)):
        assert max_abs(a, b) < (1e-5 if dtype == torch.float32 else 5e-2)


def test_router_semantics_and_hf(mm):
    N, D, E, k = 16, 32, 12, 4
    ref = Router(D, E, k, "sigmoid", True, 2.0)
    hf = mm.TokenChoiceTopKRouter(D, E, k, "sigmoid", True, 2.0)
    W = torch.randn(E, D, generator=_g(10)) / D**0.5
    ref.gate.weight.data, hf.gate.weight.data = W.to(torch.bfloat16), W.to(torch.bfloat16)
    bias = 1.0 + 0.3 * torch.randn(E, generator=_g(11))
    x = torch.randn(N, D, generator=_g(12)).to(torch.bfloat16)
    w, idx = ref(x, bias)
    hw, hidx, _ = hf(x, bias)
    assert torch.equal(w, hw) and torch.equal(idx, hidx) and w.dtype == torch.float32
    scores = torch.sigmoid(x.float() @ W.to(torch.bfloat16).float().T)
    assert torch.equal(idx, torch.topk(scores + bias, k, dim=1).indices)  # bias selects ...
    raw = scores.gather(1, idx)  # ... but weights are the unbiased scores, renormalized x 2.0
    torch.testing.assert_close(w, raw / (raw.sum(-1, keepdim=True) + 1e-20) * 2.0, rtol=1e-6, atol=1e-7)
    assert not torch.equal(idx, torch.topk(scores, k, dim=1).indices)  # the bias changes some selections here
    torch.testing.assert_close(w.sum(-1), torch.full((N,), 2.0))


@pytest.mark.parametrize("dtype", DTYPES, ids=IDS)
def test_moe_and_mlp_match_hf(mm, dtype):
    args = tiny_random_args(num_experts=8, experts_top_k=2, moe_intermediate_size=24, hidden_size=32)
    from models.demos.motif3.reference.tests.hf_reference import hf_config_from_args
    from models.demos.motif3.reference.weights import random_state_dict, round_state_dict

    cfg = hf_config_from_args(args)
    ref, hf = MoE(args, 2), mm.MoE(cfg)
    sd = random_state_dict(args, layer_ids=[2], seed=13)
    sd = {k[len("model.layers.2.moe.") :]: v for k, v in sd.items() if k.startswith("model.layers.2.moe.")}
    if dtype != torch.float32:
        sd = round_state_dict(sd, dtype)
    ref.load_state_dict(sd, strict=True)
    hf.load_state_dict(sd, strict=True)
    ref, hf = ref.to(dtype), hf.to(dtype)
    x = torch.randn(2, 9, 32, generator=_g(14)).to(dtype)
    with torch.no_grad():
        assert torch.equal(ref(x), hf(x)[0])
    mlp_ref = MLP(32, 48, hidden_clamp=1e6, output_scale=0.5).to(dtype)
    mlp_hf = mm.MotifMLP(cfg, intermediate_size=48).to(dtype)
    mlp_hf.load_state_dict(mlp_ref.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(mlp_ref(x), mlp_hf(x))
