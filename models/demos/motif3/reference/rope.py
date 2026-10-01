# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RoPE for Motif-3 GDLA (applied to the 64-dim rope slice of q and to the shared k_pe).

* Global (full-attention) layers: YaRN-interpolated ``inv_freq`` (DeepSeek-V3 formula; HF
  ``_compute_yarn_inv_freq`` == training ``precompute_freqs_cis_yarn`` == vLLM ``YaRNScalingRotaryEmbedding``).
  It is ALWAYS on because ``max_position_embeddings (262144) > original_seq_len (4096)``. cos/sin are never
  magnitude-scaled (``attention_factor = 1``); the YaRN mscale lives in the softmax scale instead.
* SWA layers: plain RoPE with ``swa_rope_theta``.
* Rotation is half-split / NeoX (``rotate_half``): pairs are ``(i, i + dim/2)``. The HF checkpoint stores the
  rope rows already de-interleaved, so no permutation is needed (study §3.10).
* Precision (HF modeling_motif.py:406-416, :669-672): angles ``pos * inv_freq`` in fp32, cos/sin rounded to
  the activation dtype, rotation math in fp32 on upcast inputs, result cast back.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


def plain_inv_freq(dim: int, theta: float) -> torch.Tensor:
    """``1 / theta^(2i/dim)``, fp32 ``[dim // 2]`` (HF modeling_motif.py:331-334)."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").float() / dim))


def yarn_correction_range(dim: int, theta: float, original_seq_len: int, beta_fast: float, beta_slow: float):
    """``(low, high)`` frequency-index range of the YaRN ramp ([10, 23] for Motif-3's 64-dim slice)."""

    def find_correction_dim(num_rotations):
        return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = math.floor(find_correction_dim(beta_fast))
    high = math.ceil(find_correction_dim(beta_slow))
    return max(low, 0), min(high, dim - 1)


def yarn_inv_freq(
    dim: int,
    theta: float,
    max_seq_len: int,
    original_seq_len: int,
    factor: float,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """YaRN ``inv_freq`` (fp32 ``[dim // 2]``), op-for-op equal to HF ``_compute_yarn_inv_freq`` (:270-306)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim))
    if max_seq_len > original_seq_len:
        low, high = yarn_correction_range(dim, theta, original_seq_len, beta_fast, beta_slow)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32, device="cpu") - low) / (high - low), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


@lru_cache(maxsize=64)
def _cached_inv_freq(
    kind: str, dim: int, theta: float, max_seq_len: int, orig: int, factor: float, bf: float, bs: float
):
    if kind == "yarn":
        return yarn_inv_freq(dim, theta, max_seq_len, orig, factor, bf, bs)
    return plain_inv_freq(dim, theta)


def inv_freq_for_layer(args, layer_idx: int, *, swa: bool | None = None) -> torch.Tensor:
    """The fp32 ``inv_freq`` used by attention layer ``layer_idx`` (``swa`` overrides the schedule, e.g. MTP)."""
    is_swa = args.is_swa_layer(layer_idx) if swa is None else swa
    dim = args.qk_rope_head_dim
    if args.uses_yarn(layer_idx, swa=is_swa):
        p = args.yarn_params()
        return _cached_inv_freq(
            "yarn",
            dim,
            p["theta"],
            p["max_seq_len"],
            p["original_seq_len"],
            p["factor"],
            p["beta_fast"],
            p["beta_slow"],
        )
    return _cached_inv_freq("plain", dim, args.rope_theta_for_layer(layer_idx, swa=is_swa), 0, 0, 1.0, 0.0, 0.0)


def rope_cos_sin(inv_freq: torch.Tensor, positions: torch.Tensor, dtype: torch.dtype):
    """cos/sin tables ``[..., S, dim]`` for integer ``positions [..., S]``, rounded to ``dtype``.

    ``freqs = pos * inv_freq`` in fp32 (bit-equal to HF's K=1 matmul), ``emb = cat(freqs, freqs)``.
    """
    freqs = positions.to(torch.float32)[..., None] * inv_freq.to(torch.float32)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """NeoX rotation: ``[x1, x2] -> [-x2, x1]`` (== HF's roll + negate, modeling_motif.py:419-434)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x [B, S, H, dim]`` with ``cos/sin [B, S, dim]``: fp32 math, result in ``x.dtype``."""
    out_dtype = x.dtype
    xf = x.to(torch.float32)
    c = cos.to(torch.float32).unsqueeze(-2)
    s = sin.to(torch.float32).unsqueeze(-2)
    return (xf * c + rotate_half(xf) * s).to(out_dtype)
