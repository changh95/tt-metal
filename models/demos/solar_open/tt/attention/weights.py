# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention weight loading for Solar-Open.

Solar-Open attention is bias-free (``attention_bias=False``) and has no attention-sink logits, so a layer's attention
weights are exactly two device tensors: the fused, column-parallel ``[Q | K | V]`` projection and the row-parallel
``o_proj``. The ``self_attn`` substate must already be in Meta (interleaved) RoPE format, i.e. produced by
``convert_hf_qkv_to_meta_format`` (``models/tt_transformers/tt/load_checkpoints.py``):

    q_proj.weight  [num_heads * head_dim, hidden_size]      Solar-Open-100B: [8192, 4096] (64 heads, Meta-permuted)
    k_proj.weight  [num_kv_heads * head_dim, hidden_size]   [1024, 4096] (8 heads, Meta-permuted)
    v_proj.weight  [num_kv_heads * head_dim, hidden_size]   [1024, 4096]
    o_proj.weight  [hidden_size, num_heads * head_dim]      [4096, 8192]

Cache stems (under ``<layer>/self_attn/``): ``wqkv`` and ``o_proj`` (``o_proj_padded`` when hidden_size / tp is not
tile aligned, which never happens for Solar-Open-100B at TP=8).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.solar_open.config import MeshConfig
from models.demos.solar_open.utils.general_utils import get_cache_file_name
from models.demos.solar_open.utils.substate import substate

from .config import AttentionConfig


@dataclass(frozen=True)
class AttentionWeights:
    """Container for attention weight tensors - immutable after creation.

    Attributes:
        wqkv: Fused QKV projection, column-parallel over the mesh TP axis. Host shape
            ``[1, 1, hidden_size, tp * (local_q + local_k + local_v)]``; per device ``[1, 1, 4096, 1280]`` for
            Solar-Open-100B at TP=8 (8 q heads + 1 k head + 1 v head of head_dim 128 = 40 tiles).
        o_proj: Output projection, row-parallel over the TP axis. Host shape
            ``[num_heads * head_dim, padded_hidden]``; per device ``[1024, 4096]`` at TP=8. Every device produces a
            partial sum over its own heads that the TP all-reduce combines.
    """

    wqkv: ttnn.Tensor
    o_proj: ttnn.Tensor


def load_attention_weights(
    mesh_device,
    config: AttentionConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
) -> AttentionWeights:
    """
    Load and shard attention weights.

    Args:
        mesh_device: TTNN mesh device
        config: Attention configuration
        state_dict: ``self_attn`` substate in Meta RoPE format (see module docstring). An empty dict selects
            cache-only mode: both tensors are read back from ``tensor_cache_path``.
        mesh_config: Mesh parallelization config
        weight_dtype: Data type for weights (default: bfloat8_b)
        tensor_cache_path: Optional path for weight caching

    Returns:
        AttentionWeights container with the fused QKV and o_proj device tensors
    """

    # Compute o_proj padding size based on config/mesh (independent of state_dict). The per-device output width
    # hidden_size / tp must be tile aligned for the CCL ops; for Solar-Open-100B (4096 / 8 = 512) the pad is 0.
    hidden_size = config.hidden_size
    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32  # Round up to tile boundary
    o_proj_pad_size = padded_local_hidden - local_hidden
    o_proj_cache_suffix = "_padded" if o_proj_pad_size > 0 and mesh_config.tp > 1 else ""

    if state_dict:
        # Solar-Open attention has neither biases nor attention-sink logits, and the code paths that consumed them do
        # not exist in this tree: the substate must hold exactly the four projection weights, so any extra tensor
        # (a *_proj.bias, a per-head sink logit, ...) fails here instead of being silently ignored.
        expected_keys = {f"{proj}.weight" for proj in ("q_proj", "k_proj", "v_proj", "o_proj")}
        unexpected_keys = sorted(set(state_dict) - expected_keys)
        assert (
            not unexpected_keys
        ), f"unsupported attention tensors {unexpected_keys}: Solar-Open attention has only the four projection weights"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert "bias" not in substate(state_dict, proj), f"{proj}.bias found: attention biases are not supported"

        # Extract projection weights from state dict
        q_proj_weight = substate(state_dict, "q_proj")["weight"]  # [num_heads * head_dim, hidden_size]
        k_proj_weight = substate(state_dict, "k_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]
        v_proj_weight = substate(state_dict, "v_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]

        o_proj = substate(state_dict, "o_proj")["weight"].transpose(-1, -2)  # [num_heads * head_dim, hidden_size]

        # Create fused QKV weight: split Q, K, V across devices, then concatenate per device. Device i holds q heads
        # [i * local_q_heads, (i + 1) * local_q_heads) and kv head(s) i - the GQA grouping that
        # nlp_create_qkv_heads_decode / nlp_create_qkv_heads assume (q head h attends with kv head h // group_size).
        qkv_list = []
        for i in range(mesh_config.tp):
            # Chunk weights across tensor parallel dimension
            wq_selected = torch.chunk(q_proj_weight, mesh_config.tp, dim=0)[i]
            wk_selected = torch.chunk(k_proj_weight, mesh_config.tp, dim=0)[i]
            wv_selected = torch.chunk(v_proj_weight, mesh_config.tp, dim=0)[i]

            # Transpose for matmul: [hidden_size, local_dim]
            wq = wq_selected.transpose(-2, -1)
            wk = wk_selected.transpose(-2, -1)
            wv = wv_selected.transpose(-2, -1)

            # Concatenate Q, K, V: [hidden_size, local_q_dim + local_k_dim + local_v_dim]
            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        # Concatenate across devices: [hidden_size, total_qkv_dim]
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size, total_qkv_dim]

        # Pad o_proj output dimension for tile alignment in CCL operations.
        # Without padding, local_hidden = hidden_size / TP may not be tile-aligned,
        # causing CCL to do expensive Untilize->Pad->Tilize cycles internally.
        if o_proj_pad_size > 0 and mesh_config.tp > 1:
            # Pad the output dimension of o_proj weight: [input_dim, hidden_size] -> [input_dim, padded_hidden]
            # Each TP device's output goes from local_hidden to padded_local_hidden
            padded_hidden = padded_local_hidden * mesh_config.tp
            o_proj = torch.nn.functional.pad(o_proj, (0, padded_hidden - hidden_size), "constant", value=0.0)
    else:
        # Cache-only mode: ttnn.as_tensor reads both tensors from their cache_file_name.
        qkv_cat = None
        o_proj = None

    # Clean mesh mapping using MeshConfig
    col_mesh_mapper = mesh_config.column_parallel(mesh_device)
    row_mesh_mapper = mesh_config.row_parallel(mesh_device)

    # Fused QKV weight: per device [1, 1, hidden_size, local_q_dim + local_k_dim + local_v_dim]
    wqkv = ttnn.as_tensor(
        qkv_cat,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=col_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Output projection: per device [local_heads * head_dim, padded_hidden]
    o_proj_tt = ttnn.as_tensor(
        o_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=row_mesh_mapper,
        cache_file_name=get_cache_file_name(tensor_cache_path, f"o_proj{o_proj_cache_suffix}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return AttentionWeights(wqkv=wqkv, o_proj=o_proj_tt)
