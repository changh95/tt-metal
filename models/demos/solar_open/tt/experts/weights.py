# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Expert weight loading and management.

Input layout (transformers >= 5 ``SolarOpenNaiveMoe`` after ``AutoModelForCausalLM.from_pretrained`` or
``SolarOpenDecoderLayer(config).state_dict()``):

* ``gate_up_proj [E, 2I, H]`` - nn.Linear orientation, rows ``0:I`` are gate_proj, rows ``I:2I`` are up_proj
  (NOT interleaved)
* ``down_proj [E, H, I]`` - nn.Linear orientation

Per-device layout produced here (TP sharded over the intermediate dimension, mesh column-parallel /
row-parallel mappers): ``gate_up_proj [1, E, H, 2 * Ip]`` = ``[gate_d | up_d]`` and ``down_proj [1, E, I/tp, H]``,
where ``Ip`` is the per-device intermediate width rounded up to a tile multiple (Solar-Open at TP=8:
``1280 / 8 = 160 = 5`` tiles, so no padding). There are no bias tensors.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.solar_open.config import MeshConfig, Mode
from models.demos.solar_open.utils.general_utils import get_cache_file_name

from .config import ExpertConfig

_TILE_SIZE = 32


@dataclass(frozen=True)  # ✅ Make immutable to prevent accidental modification
class ExpertWeights:
    """Container for expert weight tensors - immutable after creation.

    Gate and up projections are stored FUSED along the output dimension so the two projections run
    as one sparse_matmul in decode (the per-expert cost of that op is a fixed overhead, not bandwidth,
    so one call with N = 2 * intermediate is ~half the price of two calls) and as one batched dense matmul
    in EP=1 prefill. Per device the fused output is laid out as
    [gate (intermediate_padded_per_device) | up (intermediate_padded_per_device)], each half zero-padded
    from intermediate_size_per_device up to a tile multiple so that the halves can be split at a tile
    boundary and fed to the GLU / the down projection without any re-layout.

    The experts are bias-free. Per-expert weight slices needed by the dense prefill paths (per-expert
    loop, hot-expert group) are created on demand from these tensors and freed after use; nothing per-expert
    is kept alive between calls.
    """

    gate_up_proj: ttnn.Tensor  # [1, E, hidden, 2 * intermediate_padded_per_device] per device
    down_proj: ttnn.Tensor  # [1, E, intermediate_size_per_device, hidden] per device
    intermediate_size_per_device: int
    intermediate_padded_per_device: int
    # down_proj with K zero-padded from intermediate_size_per_device to intermediate_padded_per_device (dense matmul
    # requires matching logical K; the sparse_matmul path compares padded shapes). Set on first dense prefill; it
    # aliases down_proj when no padding is needed (Solar-Open: 160 = 5 tiles).
    down_proj_padded: ttnn.Tensor = None
    # {n: [n, n] bf16 identity} one-hot tables for the expert-sorted prefill path's scatter matmul (created on use).
    eye_tables: dict = None


def expert_shard_sizes(intermediate_size: int, tp: int) -> tuple[int, int]:
    """(per-device intermediate width, that width rounded up to a tile multiple) for TP sharding."""
    local = intermediate_size // tp
    padded = ((local + _TILE_SIZE - 1) // _TILE_SIZE) * _TILE_SIZE
    return local, padded


def _fuse_gate_up_per_device(gate, up, tp, local, padded):
    """Interleave per-device gate and up column blocks: [..., tp * 2 * padded] laid out as
    [gate_dev0 | up_dev0 | gate_dev1 | up_dev1 | ...] with each block zero-padded from `local` to `padded`
    columns, so that column-parallel sharding across `tp` devices gives every device [gate | up]."""
    out_shape = gate.shape[:-1] + (tp * 2 * padded,)
    fused = gate.new_zeros(out_shape)
    for d in range(tp):
        base = d * 2 * padded
        fused[..., base : base + local] = gate[..., d * local : (d + 1) * local]
        fused[..., base + padded : base + padded + local] = up[..., d * local : (d + 1) * local]
    return fused


def prepare_expert_weights_torch(state_dict, config: ExpertConfig, tp: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-side (torch only) adaptation of the HF fused expert layout to the per-device sharded layout.

    Args:
        state_dict: ``{"gate_up_proj": [E, 2I, H], "down_proj": [E, H, I]}`` (transformers >= 5 layout)
        config: expert configuration (E, I, H)
        tp: tensor-parallel degree over the intermediate dimension

    Returns:
        ``gate_up_proj [1, E, H, tp * 2 * Ip]`` whose column-parallel shard ``d`` is ``[gate_d | up_d]`` (each half
        zero-padded from ``I/tp`` to ``Ip`` columns), and ``down_proj [1, E, I, H]`` whose row-parallel shard ``d``
        holds the intermediate rows ``d*I/tp:(d+1)*I/tp``.

    Raises:
        ValueError: if either tensor does not have the expected shape.
    """
    E, I, H = config.num_experts, config.intermediate_size, config.hidden_size
    local, padded = expert_shard_sizes(I, tp)

    gate_up = state_dict["gate_up_proj"]
    if tuple(gate_up.shape) != (E, 2 * I, H):
        raise ValueError(
            f"experts.gate_up_proj must be [E, 2I, H] = {(E, 2 * I, H)}, got {tuple(gate_up.shape)} "
            "(transformers >= 5 fused layout expected: gate rows first, then up rows)"
        )
    gate = gate_up[:, :I, :].transpose(1, 2)  # [E, H, I]
    up = gate_up[:, I:, :].transpose(1, 2)  # [E, H, I]
    gate_up_proj = _fuse_gate_up_per_device(gate, up, tp, local, padded).reshape(1, E, H, tp * 2 * padded)

    down = state_dict["down_proj"]
    if tuple(down.shape) != (E, H, I):
        raise ValueError(f"experts.down_proj must be [E, H, I] = {(E, H, I)}, got {tuple(down.shape)}")
    down_proj = down.transpose(1, 2).reshape(1, E, I, H)
    return gate_up_proj, down_proj


def load_expert_weights(
    mesh_device,
    config: ExpertConfig,
    state_dict,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
) -> ExpertWeights:
    """
    Load and shard expert weights.

    Args:
        mesh_device: TTNN mesh device
        config: Expert configuration
        state_dict: ``{"gate_up_proj": [E, 2I, H], "down_proj": [E, H, I]}``; an empty dict loads from the cache
        mesh_config: Mesh parallelization configuration
        weight_dtype: Data type for weights (default bfloat8_b; bfloat4_b halves the footprint)
        tensor_cache_path: Optional path for weight caching (stems ``gate_up_proj_fused_tp{tp}`` and ``down_proj``)

    Returns:
        ExpertWeights with loaded and sharded tensors
    """
    tp = mesh_config.decode.tp
    intermediate_size_per_device = mesh_config.shard_size(config.intermediate_size, mode=Mode.DECODE)
    _, intermediate_padded_per_device = expert_shard_sizes(config.intermediate_size, tp)

    if state_dict:
        gate_up_proj, down_proj = prepare_expert_weights_torch(state_dict, config, tp)
    else:
        gate_up_proj = down_proj = None

    gate_up_proj_tt = ttnn.as_tensor(
        gate_up_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
        cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_up_proj_fused_tp{tp}"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down_proj_tt = ttnn.as_tensor(
        down_proj,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=weight_dtype,
        mesh_mapper=mesh_config.row_parallel(mesh_device),
        cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return ExpertWeights(
        gate_up_proj=gate_up_proj_tt,
        down_proj=down_proj_tt,
        intermediate_size_per_device=intermediate_size_per_device,
        intermediate_padded_per_device=intermediate_padded_per_device,
    )
