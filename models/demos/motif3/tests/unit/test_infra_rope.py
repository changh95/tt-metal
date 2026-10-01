# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for tt/rope.py: inv_freq / tables / rotation equal the HF formulas and the CPU reference.

Run device-hidden with ``--noconftest`` (see test_infra_config.py docstring).
"""

import math

import pytest
import torch

from models.demos.motif3.tt.model_config import DEFAULT_HF_META_DIR, MotifTTConfig
from models.demos.motif3.tt.rope import (
    apply_rope_torch,
    cos_sin_table,
    inv_freq_for_kind,
    inv_freq_for_layer,
    lanes_to_rows,
    plain_inv_freq,
    positions_to_rot_idxs,
    rotate_half,
    rotate_half_matrix,
    yarn_correction_range,
    yarn_inv_freq,
)

CFG = MotifTTConfig.from_hf_config(str(DEFAULT_HF_META_DIR))


# ---- verbatim copies of the HF code paths (hf_meta/modeling_motif.py) -------------------------------------
def _hf_compute_yarn_inv_freq(dim, end, theta, original_seq_len, rope_factor, beta_fast=32.0, beta_slow=1.0):
    def find_correction_dim(num_rotations, d, base, max_seq_len):
        return d * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, d, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, d, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, d, base, max_seq_len))
        return max(low, 0), min(high, d - 1)

    def linear_ramp_factor(lo, hi, d):
        if lo == hi:
            hi += 0.001
        ramp = (torch.arange(d, dtype=torch.float32) - lo) / (hi - lo)
        return torch.clamp(ramp, 0, 1)

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if end > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, theta, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth
    return freqs


def _hf_plain_inv_freq(dim, theta):
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


def _hf_cos_sin(inv_freq, position_ids, dtype):
    """``MotifRotaryEmbedding.forward`` (attention_scaling = 1)."""
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _hf_rotate_half(x):
    half_size = x.shape[-1] // 2
    rotated_tensor = torch.roll(x, shifts=-half_size, dims=-1)
    rotated_tensor[..., :half_size] *= -1
    return rotated_tensor


def _hf_apply_rotary_pos_emb_single(x, cos, sin):
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return x * cos + _hf_rotate_half(x) * sin


# ---------------------------------------------------------------------------------------------------------
def test_yarn_inv_freq_bit_exact_vs_hf():
    ours = yarn_inv_freq(64, 1e4, 262144, 4096, 64.0, 32.0, 1.0)
    hf = _hf_compute_yarn_inv_freq(64, 262144, 1e4, 4096, 64.0, 32.0, 1.0)
    assert ours.dtype == torch.float32 and ours.shape == (32,)
    assert torch.equal(ours, hf), (ours - hf).abs().max()
    assert torch.equal(inv_freq_for_kind(CFG, "yarn"), hf)
    assert yarn_correction_range(64, 1e4, 4096, 32.0, 1.0) == (10, 23)
    plain = plain_inv_freq(64, 1e4)
    # study 01 Appendix A.1: dims 0..10 unchanged, 23..31 divided by 64, ramp in between
    assert torch.equal(ours[:11], plain[:11])
    assert torch.allclose(ours[23:], plain[23:] / 64, rtol=1e-6, atol=0)
    for i, v in {11: 3.897652e-02, 15: 8.286425e-03, 20: 7.677645e-04, 22: 1.624390e-04, 31: 2.083627e-06}.items():
        assert ours[i].item() == pytest.approx(v, rel=2e-6)


def test_plain_inv_freq_bit_exact_vs_hf():
    assert torch.equal(plain_inv_freq(64, 1e4), _hf_plain_inv_freq(64, 1e4))
    assert torch.equal(inv_freq_for_kind(CFG, "plain"), _hf_plain_inv_freq(64, 1e4))
    assert torch.equal(inv_freq_for_layer(CFG, 0), inv_freq_for_kind(CFG, "yarn"))
    assert torch.equal(inv_freq_for_layer(CFG, 1), inv_freq_for_kind(CFG, "plain"))


@pytest.mark.parametrize("kind", ["yarn", "plain"])
def test_cos_sin_tables_bit_exact_vs_hf(kind):
    inv = inv_freq_for_kind(CFG, kind)
    pos = torch.tensor([0, 1, 2, 127, 128, 129, 130, 1000, 4095, 4096, 5000, 32767])
    cos, sin = cos_sin_table(inv, pos, torch.bfloat16)
    hc, hs = _hf_cos_sin(inv, pos[None], torch.bfloat16)
    assert torch.equal(cos, hc[0]) and torch.equal(sin, hs[0])
    # half-split duplicated layout
    assert torch.equal(cos[:, :32], cos[:, 32:]) and torch.equal(sin[:, :32], sin[:, 32:])


@pytest.mark.parametrize("kind", ["yarn", "plain"])
def test_matches_reference_rope(kind):
    ref_rope = pytest.importorskip("models.demos.motif3.reference.rope")
    ref_cfg = pytest.importorskip("models.demos.motif3.reference.config")
    args = ref_cfg.MotifArgs.from_hf_config(str(DEFAULT_HF_META_DIR))
    layer = 0 if kind == "yarn" else 1
    ref_inv = ref_rope.inv_freq_for_layer(args, layer)
    assert torch.equal(inv_freq_for_layer(CFG, layer), ref_inv)
    pos = torch.arange(0, 32768, 37)
    c, s = cos_sin_table(ref_inv, pos, torch.bfloat16)
    rc, rs = ref_rope.rope_cos_sin(ref_inv, pos, torch.bfloat16)
    assert torch.equal(c, rc) and torch.equal(s, rs)
    x = torch.randn(len(pos), 3, 64, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    ours = apply_rope_torch(x, c[:, None], s[:, None])
    ref = ref_rope.apply_rope(x[None], c[None], s[None])[0]
    assert torch.equal(ours, ref)


def test_rotation_matrix_and_composite():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(5, 10, 64, generator=g, dtype=torch.float64)
    R = rotate_half_matrix(64, torch.float64)
    assert torch.equal(x @ R, rotate_half(x))
    assert torch.equal(rotate_half(x), _hf_rotate_half(x.clone()))
    # R is a signed permutation: R^T R = I, R^2 = -I
    assert torch.equal(R.t() @ R, torch.eye(64, dtype=torch.float64))
    assert torch.equal(R @ R, -torch.eye(64, dtype=torch.float64))

    inv = inv_freq_for_kind(CFG, "yarn")
    pos = torch.tensor([0, 7, 128, 129, 5000])
    cos, sin = cos_sin_table(inv, pos, torch.bfloat16)
    xb = x.to(torch.bfloat16)  # [S, H, 64]
    hf = _hf_apply_rotary_pos_emb_single(xb.float()[None], cos.float()[None], sin.float()[None])[0].to(torch.bfloat16)
    ours = apply_rope_torch(xb, cos[:, None], sin[:, None])
    assert torch.equal(ours, hf)
    # composite form used on device: x cos + (x @ R) sin  (fp32 math)
    comp = (xb.float() * cos[:, None].float() + (xb.float() @ R.float()) * sin[:, None].float()).to(torch.bfloat16)
    assert torch.equal(comp, hf)


def test_lane_layout():
    pos = torch.arange(32) * 100
    pos[5] = -1  # inactive lane
    rows = positions_to_rot_idxs(pos, CFG)
    assert rows.shape == (4, 32) and rows.dtype == torch.int32
    for r in range(4):
        exp = (torch.arange(8 * r, 8 * r + 8) * 100).clamp_min(0)
        if r == 0:
            exp[5] = 0
        assert torch.equal(rows[r, :8], exp.to(torch.int32))
        assert torch.equal(rows[r, 8:], torch.zeros(24, dtype=torch.int32))
    pt = torch.arange(32 * 512).reshape(32, 512)
    pr = lanes_to_rows(pt, CFG)
    assert pr.shape == (4, 8, 512) and torch.equal(pr[2, 3], pt[19])
    with pytest.raises(ValueError):
        positions_to_rot_idxs(torch.full((32,), 32768), CFG)
