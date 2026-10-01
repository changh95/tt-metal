# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Config schedule / scales / YaRN / RoPE convention tests."""

import math

import pytest
import torch

from models.demos.motif3.reference import MotifArgs, tiny_random_args
from models.demos.motif3.reference.rope import (
    apply_rope,
    inv_freq_for_layer,
    plain_inv_freq,
    rope_cos_sin,
    rotate_half,
    yarn_correction_range,
    yarn_inv_freq,
)

from .hf_reference import HF_META_DIR, hf_config_from_args, hf_config_from_json, load_hf_modules


@pytest.fixture(scope="module")
def real_args():
    return MotifArgs.from_hf_config(HF_META_DIR)


def test_real_config_schedule(real_args):
    a = real_args
    assert a == MotifArgs.from_hf_config(HF_META_DIR / "config.json")
    assert (a.num_hidden_layers, a.hidden_size, a.vocab_size) == (53, 4096, 220160)
    assert (a.grouped_ratio, a.heads_per_group, a.n_signal_heads, a.kv_group_size) == (4, 5, 64, 5)
    assert (a.qk_nope_head_dim, a.qk_rope_head_dim, a.v_head_dim, a.kv_lora_rank) == (128, 64, 128, 512)
    glob = [i for i in range(53) if not a.is_swa_layer(i)]
    assert glob == list(range(0, 53, 4)) and len(glob) == 14
    assert sum(a.is_swa_layer(i) for i in range(53)) == 39
    assert [i for i in range(53) if not a.is_moe_layer(i)] == [0, 1]
    assert a.effective_sliding_window == 129 and a.attention_window(1) == 129 and a.attention_window(0) is None
    assert math.isclose(a.softmax_scale(1), 192**-0.5) and abs(a.softmax_scale(1) - 0.07216878) < 1e-8
    assert abs(a.softmax_scale(0) - 0.14467963) < 1e-8
    assert a.uses_yarn(0) and not a.uses_yarn(1) and a.mhc_h_post_coeff == 1.0
    assert a.eos_token_ids == (0, 3, 6) and a.bos_token_id == 1
    assert (a.polynorm_output_scale, a.polynorm_bias_clamp, a.hidden_clamp) == (0.5, 0.5, 1e6)
    assert a.mhc_rms_eps == 1e-6 and a.rms_norm_eps == 1e-5 and a.polynorm_eps == 1e-6
    assert a.shared_intermediate_size == 1280 and a.load_balance_coeff == 1e-4


def test_schedule_matches_hf_attention_modules(real_args):
    """Window/scale/rope per layer agree with HF ``MotifGDLAttention`` for all 53 layers (meta-device build)."""
    cfg_mod, mm = load_hf_modules()
    cfg = hf_config_from_json(HF_META_DIR, cfg_mod)
    with torch.device("meta"):
        for i in range(53):
            hf_attn = mm.MotifGDLAttention(cfg, layer_idx=i)
            assert hf_attn.sliding_window == real_args.attention_window(i), i
            assert hf_attn.scaling == real_args.softmax_scale(i), i
            assert (hf_attn.swa_rotary_emb is None) == real_args.uses_yarn(i), i


def test_yarn_inv_freq_values(real_args):
    p = real_args.yarn_params()
    assert yarn_correction_range(64, p["theta"], p["original_seq_len"], p["beta_fast"], p["beta_slow"]) == (10, 23)
    inv = inv_freq_for_layer(real_args, 0)
    plain = plain_inv_freq(64, 10000.0)
    ratio = plain / inv
    assert torch.allclose(ratio[:11], torch.ones(11)) and torch.allclose(ratio[23:], torch.full((9,), 64.0))
    assert abs(float(ratio[11]) - 1.082) < 1e-3 and abs(float(ratio[22]) - 10.947) < 1e-3  # study App. A.1
    assert torch.equal(inv_freq_for_layer(real_args, 1), plain)


@pytest.mark.parametrize("dim", [16, 64])
def test_yarn_matches_hf_function(dim):
    _, mm = load_hf_modules()
    args = (dim, 262144, 10000.0, 4096, 64.0, 32.0, 1.0)
    hf = mm._compute_yarn_inv_freq(*args)
    ours = yarn_inv_freq(dim, 10000.0, 262144, 4096, 64.0, 32.0, 1.0)
    assert torch.equal(hf, ours)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_rope_tables_match_hf_rotary_embedding(dtype):
    args = tiny_random_args(swa_rope_theta=5000.0)
    cfg_mod, mm = load_hf_modules()
    cfg = hf_config_from_args(args, cfg_mod)
    pos = torch.tensor([[0, 1, 5, 128, 129, 4095, 4096, 100000, 262143]])
    x = torch.zeros(1, pos.shape[1], 8, dtype=dtype)
    glob = mm.MotifRotaryEmbedding(cfg, rope_head_dim=args.qk_rope_head_dim)
    cos_h, sin_h = glob(x, pos)
    cos_r, sin_r = rope_cos_sin(inv_freq_for_layer(args, 0), pos, dtype)
    assert torch.equal(cos_h, cos_r) and torch.equal(sin_h, sin_r)
    swa_attn = mm.MotifGDLAttention(cfg, layer_idx=1)
    cos_h, sin_h = swa_attn.swa_rotary_emb(x, pos)
    cos_r, sin_r = rope_cos_sin(inv_freq_for_layer(args, 1), pos, dtype)
    assert torch.equal(cos_h, cos_r) and torch.equal(sin_h, sin_r)
    q = torch.randn(1, pos.shape[1], 3, args.qk_rope_head_dim).to(dtype)
    hf_rot = mm.apply_rotary_pos_emb_single(q.float(), cos_h.float(), sin_h.float()).to(dtype)
    assert torch.equal(apply_rope(q, cos_r, sin_r), hf_rot)


def test_half_split_equals_training_interleaved_rope_after_deinterleave():
    """Training applies complex (interleaved-pair) RoPE and then de-interleaves (``reorder_headdim_elements_rope``);
    that equals half-split RoPE on the de-interleaved vector, i.e. on the checkpoint rows as stored."""
    d = 16
    inv = plain_inv_freq(d, 10000.0)
    pos = torch.arange(7)
    x = torch.randn(7, d, dtype=torch.float64)
    ang = pos[:, None].double() * inv.double()[None]
    xc = torch.view_as_complex(x.reshape(7, d // 2, 2).contiguous())
    y_interleaved = torch.view_as_real(xc * torch.polar(torch.ones_like(ang), ang)).reshape(7, d)
    deint = lambda t: t.reshape(7, d // 2, 2).transpose(1, 2).reshape(7, d)  # noqa: E731  [even..., odd...]
    emb = torch.cat([ang, ang], -1)
    y_half = deint(x) * emb.cos() + rotate_half(deint(x)) * emb.sin()
    assert torch.allclose(deint(y_interleaved), y_half, atol=1e-12)


def test_tiny_args_structure():
    a = tiny_random_args()
    assert a.heads_per_group == 5 and a.grouped_ratio == 4 and a.kv_group_size == 5
    assert a.mhc_expansion_rate == 4 and a.mhc_sinkhorn_iters == 20 and a.experts_top_k == 8
    assert a.effective_sliding_window == 129
    kinds = [a.layer_kind(i) for i in range(a.num_hidden_layers)]
    assert kinds == ["global/dense", "swa/dense", "swa/moe", "swa/moe", "global/moe", "swa/moe"]
    p = a.yarn_params()
    assert yarn_correction_range(a.qk_rope_head_dim, p["theta"], p["original_seq_len"], 32.0, 1.0) == (2, 6)
    assert a.replace(sliding_window=8).effective_sliding_window == 9
    with pytest.raises(ValueError):
        tiny_random_args(num_attention_heads=21)
