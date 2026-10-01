# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 golden CPU reference (pure PyTorch; never imports ttnn). See README.md."""

from .cache import LatentKVCache, MotifKVCache
from .config import MotifArgs, tiny_random_args
from .modules import (
    MLP,
    DecoderLayer,
    GDLAttention,
    GroupedPolyNorm,
    MHCLayer,
    MoE,
    MotifForCausalLM,
    MotifModel,
    MotifMTP,
    PolyNorm,
    RMSNorm,
    RoutedExperts,
    Router,
    attention_mask,
    sinkhorn,
)
from .rope import apply_rope, inv_freq_for_layer, plain_inv_freq, rope_cos_sin, rotate_half, yarn_inv_freq
from .weights import (
    MissingWeightsError,
    MotifCheckpoint,
    build_random_model,
    hf_to_reference_state_dict,
    load_mtp,
    load_reference_model,
    random_state_dict,
    reference_to_hf_state_dict,
)
