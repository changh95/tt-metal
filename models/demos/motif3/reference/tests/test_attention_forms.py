# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""(b) MLA-absorbed latent attention == expanded GQA attention (fp32), including SWA far past the window."""

import pytest
import torch

from models.demos.motif3.reference import LatentKVCache, build_random_model, tiny_random_args
from models.demos.motif3.reference.golden import TensorRecorder
from models.demos.motif3.reference.rope import apply_rope, rope_cos_sin

from .common import max_abs, pcc, rand_ids

S_LONG = 300  # > 2x the 129-key window


def _attn_inputs(args, S, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, S, args.hidden_size, generator=g).to(dtype)


@pytest.fixture(scope="module")
def tiny_model():
    args = tiny_random_args()  # real window: 128 + 1 keys
    return build_random_model(args, seed=11)


@pytest.mark.parametrize("layer_idx", [0, 1, 4], ids=["global_L0", "swa_L1", "global_L4"])
def test_absorbed_equals_expanded_prefill(tiny_model, layer_idx):
    attn = tiny_model.model.layers[str(layer_idx)].self_attn
    x = _attn_inputs(tiny_model.args, S_LONG, seed=layer_idx)
    pos = torch.arange(S_LONG)[None]
    with torch.no_grad():
        out_e = attn(x, pos, mode="expanded")
        out_a = attn(x, pos, mode="absorbed")
    scale = out_e.abs().max().item()
    assert max_abs(out_a, out_e) <= 1e-5 * max(1.0, scale), f"max {max_abs(out_a, out_e)} (|out| {scale})"


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["global_L0", "swa_L1"])
def test_absorbed_decode_equals_expanded_full(tiny_model, layer_idx):
    """Decode with the latent cache (absorbed) reproduces the cache-free expanded prefill at every position."""
    args = tiny_model.args
    attn = tiny_model.model.layers[str(layer_idx)].self_attn
    x = _attn_inputs(args, S_LONG, seed=20 + layer_idx)
    pos = torch.arange(S_LONG)[None]
    n_pre = 100
    cache = LatentKVCache(1, S_LONG, args.kv_lora_rank, args.qk_rope_head_dim, x.dtype, attn.window)
    with torch.no_grad():
        ref = attn(x, pos, mode="expanded")
        outs = [attn(x[:, :n_pre], pos[:, :n_pre], cache, mode="absorbed")]
        for t in range(n_pre, S_LONG):
            outs.append(attn(x[:, t : t + 1], pos[:, t : t + 1], cache, mode="absorbed"))
    dec = torch.cat(outs, dim=1)
    assert max_abs(dec, ref) <= 1e-5 * max(1.0, ref.abs().max().item())
    # the window really matters on SWA layers: the same layer without a window differs past 129 tokens
    if attn.window is not None:
        w = attn.window
        attn.window = None
        try:
            with torch.no_grad():
                no_win = attn(x, pos, mode="expanded")
        finally:
            attn.window = w
        assert max_abs(no_win[:, :w], ref[:, :w]) < 1e-6
        assert max_abs(no_win[:, w:], ref[:, w:]) > 1e-3


def test_absorbed_equals_expanded_full_model(tiny_model):
    ids = rand_ids(tiny_model.args, 1, S_LONG, seed=1)
    with torch.no_grad():
        le = tiny_model(ids, attn_mode="expanded")
        la = tiny_model(ids, attn_mode="absorbed")
    assert max_abs(la, le) <= 1e-4, max_abs(la, le)


def test_absorbed_bf16_is_close():
    args = tiny_random_args()
    model = build_random_model(args, seed=11, dtype=torch.bfloat16)
    ids = rand_ids(args, 1, S_LONG, seed=1)
    with torch.no_grad():
        le, la = model(ids, attn_mode="expanded"), model(ids, attn_mode="absorbed")
    assert pcc(la, le) > 0.999


def test_latent_cache_contents(tiny_model):
    """The cache holds kv_norm(c_raw) WITH gamma (fork motif.py:694) and the roped shared k_pe; 576 values/token."""
    args = tiny_model.args
    attn = tiny_model.model.layers["1"].self_attn
    x = _attn_inputs(args, 50, seed=3)
    pos = torch.arange(50)[None]
    cache = LatentKVCache(1, 64, args.kv_lora_rank, args.qk_rope_head_dim, x.dtype, attn.window)
    rec = TensorRecorder()
    with torch.no_grad():
        attn(x, pos, cache, mode="absorbed", tap=rec)
        c_raw, kpe_raw = torch.split(attn.wkv_a(x), [args.kv_lora_rank, args.qk_rope_head_dim], dim=-1)
        c_expect = attn.kv_norm(c_raw)
        cos, sin = rope_cos_sin(attn.inv_freq(), pos, x.dtype)
        kpe_expect = apply_rope(kpe_raw.unsqueeze(2), cos, sin).squeeze(2)
    assert torch.equal(cache.c[:, :50], c_expect) and torch.equal(cache.c[:, :50], rec.tensors["c_kv"])
    assert torch.equal(cache.k_pe[:, :50], kpe_expect)
    assert int(cache.seq_lens[0]) == 50 and cache.c[:, 50:].abs().sum() == 0
    assert cache.c.shape[-1] + cache.k_pe.shape[-1] == args.kv_lora_rank + args.qk_rope_head_dim


def test_query_chunking_is_exact(tiny_model):
    """``sdpa_fp32`` processes queries in chunks for long prefill; chunking must not change the result."""
    from models.demos.motif3.reference.modules import attention_mask, sdpa_fp32

    g = torch.Generator().manual_seed(0)
    q, k, v = (
        torch.randn(1, 4, 50, 16, generator=g),
        torch.randn(1, 4, 50, 16, generator=g),
        torch.randn(1, 4, 50, 8, generator=g),
    )
    pos = torch.arange(50)[None]
    mask = attention_mask(pos, pos[0], 9)
    full = sdpa_fp32(q, k, v, mask, 0.25, torch.float32)
    torch.testing.assert_close(sdpa_fp32(q, k, v, mask, 0.25, torch.float32, q_chunk=7), full, atol=1e-6, rtol=1e-6)
