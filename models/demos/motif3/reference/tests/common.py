# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the reference tests (imported explicitly: the suite runs with ``--noconftest``)."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import pytest
import torch

from models.demos.motif3.reference import (
    MotifArgs,
    MotifCheckpoint,
    build_random_model,
    random_state_dict,
    reference_to_hf_state_dict,
)
from models.demos.motif3.reference.golden import pcc  # noqa: F401  (re-export for tests)
from models.demos.motif3.reference.weights import DEFAULT_WEIGHTS_DIR, round_state_dict

from .hf_reference import hf_config_from_args, hf_cpu_flash_attention, load_hf_modules

REAL_LAYERS = (0, 1, 2)


def real_checkpoint(layers: Iterable[int] = REAL_LAYERS) -> MotifCheckpoint:
    """The local checkpoint, or ``pytest.skip`` if the requested tensors are not on disk."""
    try:
        ckpt = MotifCheckpoint(DEFAULT_WEIGHTS_DIR)
    except FileNotFoundError as e:
        pytest.skip(f"real weights not available: {e}")
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for i in layers:
        names += ckpt.layer_names(i)
    missing = [n for n in names if not ckpt.is_local(n)]
    if missing:
        pytest.skip(f"real weights not local: {missing[:3]}...")
    return ckpt


def make_ref_and_hf(
    args: MotifArgs, seed: int = 0, dtype: torch.dtype = torch.float32, layer_ids: Optional[Iterable[int]] = None
) -> Tuple[torch.nn.Module, torch.nn.Module, dict]:
    """Reference model and HF ``MotifForCausalLM`` holding identical random weights (exact in ``dtype``).

    The HF model is constructed first (its ``post_init``/``_init_weights`` runs on random init) and the weights are
    loaded AFTER construction, so HF's init-time zeroing of |w| > 3*std cannot touch them. Call the HF model
    inside ``hf_cpu_flash_attention()``.
    """
    sd = random_state_dict(args, layer_ids, seed)
    if dtype != torch.float32:
        sd = round_state_dict(sd, dtype)
    ref = build_random_model(args, layer_ids, dtype=dtype, state_dict=sd)
    cfg_mod, mm = load_hf_modules()
    with hf_cpu_flash_attention():
        hf = mm.MotifForCausalLM(hf_config_from_args(args, cfg_mod)).eval()
    missing, unexpected = hf.load_state_dict(reference_to_hf_state_dict(sd), strict=True)
    assert not missing and not unexpected
    hf = hf.to(dtype)
    hf.requires_grad_(False)
    return ref, hf, sd


def rand_ids(args: MotifArgs, batch: int, seq: int, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, args.vocab_size, (batch, seq), generator=torch.Generator().manual_seed(seed))


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max())
