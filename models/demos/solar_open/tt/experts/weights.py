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

Always-on expert fusion (``MoEOptions.fuse_shared_expert``, design D2 follow-up): ``fuse_always_on_expert`` appends
the shared expert's per-device shards (``[1, 1, H, Ip]`` gate / up, ``[1, 1, Ip, H]`` down -- exactly one slot of the
layout above, because Solar's shared intermediate equals the routed one) as slot ``E`` of both tensors ON DEVICE with
``ttnn.concat``, so the routed and shared cache files stay the ones the unfused path reads and no new cache stem
exists. The router then emits ``[T, E + 1]`` routing tensors whose last column is the constant 1.0.
"""

from dataclasses import dataclass

import torch
from loguru import logger

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

    ``num_always_on_experts`` (0 unless ``fuse_always_on_expert`` built this container) counts the trailing slots of
    the E axis that hold always-on experts (Solar: the shared expert as slot 128 of 129). They are ordinary slots for
    every kernel; only the router (constant weight 1.0 in their columns), the decode path selection (T = 1 must take
    the batched union-of-experts path) and the expert-sorted prefill planner (an always-on slot is hot in every
    split and must not count against the hot-expert threshold) need to know about them.
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
    # Trailing always-on slots of the E axis (see the class docstring); 0 for the plain routed layout.
    num_always_on_experts: int = 0


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


_warned_dtype_mismatch = set()


def _warn_dtype_mismatch(shared_dtype, routed_dtype, what):
    key = (str(shared_dtype), str(routed_dtype))
    if key not in _warned_dtype_mismatch:
        _warned_dtype_mismatch.add(key)
        logger.warning(
            f"always-on expert fusion: the shared expert's {what} ({shared_dtype}) is quantised on device to the "
            f"routed experts' {routed_dtype}; SOLAR_OPEN_SHARED_EXPERT_DTYPE has no effect on the fused slot"
        )


def _cast_to(tensor, dtype):
    """``tensor`` in ``dtype`` (itself when it already is; otherwise an on-device typecast, the source freed)."""
    if tensor.dtype == dtype:
        return tensor
    cast = ttnn.typecast(tensor, dtype)
    tensor.deallocate(True)
    return cast


def fuse_always_on_expert(weights: ExpertWeights, w_gate, w_up, w_down) -> ExpertWeights:
    """Append one always-on expert (Solar: the shared expert) as the LAST slot of the per-device fused layout.

    Runs on device with ``ttnn.concat`` on the already-loaded (cached) shards, so no cache file changes and no host
    tensor is touched: ``[gate_d | up_d] = concat([w_gate, w_up], dim=3)`` gives the slot's ``[1, 1, H, 2 * Ip]``
    gate/up block, then both routed tensors grow by one slot along dim 1. The four input handles and the two routed
    tensors of ``weights`` are deallocated (the returned container replaces them). The width concat runs in bf16:
    ttnn judges a last-dim concat's alignment from ``padded_shape[-1] * element_size`` (1 for bfp8, so the 160-wide
    shards give 160 B, not 64-B aligned) and would otherwise fall back to transpose -> concat -> transpose, whose two
    bfp8 re-packs regroup the 16-element exponent blocks and move ~1 % of the mantissas by one ulp (measured
    2026-09-07). Widening bfp8 -> bf16 is exact, the bf16 concat (320 B rows) is a page copy, and the cast back to
    the routed dtype is exact for values already on its grid (same block grouping), so slot E of the result equals
    the unfused shared weights bit for bit when the dtypes agree; with bf16 shared shards or bfp4 experts the slot is
    quantised to the routed dtype here (warned once).

    Args:
        weights: the routed ``ExpertWeights`` (``gate_up_proj [1, E, H, 2 * Ip]``, ``down_proj [1, E, Ip, H]``),
            before any dense-prefill preparation (``down_proj_padded`` unset)
        w_gate, w_up: ``[1, 1, H, Ip]`` column-parallel shards (``SharedExpert.w_gate`` / ``w_up`` layout)
        w_down: ``[1, 1, Ip, H]`` row-parallel shard (``SharedExpert.w_down`` layout)

    Returns:
        ``ExpertWeights`` with ``E + 1`` slots and ``num_always_on_experts`` incremented.

    Raises:
        NotImplementedError: the per-device intermediate is zero-padded (``intermediate_size_per_device !=
            intermediate_padded_per_device``): the shared shards would need a host-side pad first (never the case
            for Solar-Open at TP=8: 160 == 160).
        ValueError: a shard does not have the shape of one slot, or ``weights`` already went through the dense
            prefill preparation.
    """
    gate_up, down = weights.gate_up_proj, weights.down_proj
    ip = weights.intermediate_padded_per_device
    if weights.intermediate_size_per_device != ip:
        raise NotImplementedError(
            f"always-on expert fusion needs a tile-aligned per-device intermediate (got {weights.intermediate_size_per_device} "
            f"padded to {ip}): the shared shards would have to be zero-padded on the host first"
        )
    if weights.down_proj_padded is not None or weights.eye_tables:
        raise ValueError("fuse_always_on_expert must run before the first dense prefill call (down_proj_padded is set)")
    hidden = gate_up.shape[2]
    if tuple(gate_up.shape)[3] != 2 * ip or tuple(down.shape)[2] != ip:
        raise ValueError(
            f"routed expert tensors {tuple(gate_up.shape)} / {tuple(down.shape)} do not match the [.., H, 2 * {ip}] / "
            f"[.., {ip}, H] slot layout"
        )
    expected = {"w_gate": (1, 1, hidden, ip), "w_up": (1, 1, hidden, ip), "w_down": (1, 1, ip, hidden)}
    for name, tensor in (("w_gate", w_gate), ("w_up", w_up), ("w_down", w_down)):
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(
                f"always-on expert {name} must have the per-device shape {expected[name]} of one expert slot, "
                f"got {tuple(tensor.shape)}"
            )

    if w_gate.dtype != gate_up.dtype or w_up.dtype != gate_up.dtype:
        _warn_dtype_mismatch(w_gate.dtype, gate_up.dtype, "gate/up shards")
    if w_down.dtype != down.dtype:
        _warn_dtype_mismatch(w_down.dtype, down.dtype, "down shard")
    # Slot layout [gate_d | up_d]: widen to bf16 (exact), concat along the last dim on the page-copy path (see the
    # docstring for why a bfp8 width concat is not exact), cast back to the routed dtype.
    gate_wide = _cast_to(w_gate, ttnn.bfloat16)
    up_wide = _cast_to(w_up, ttnn.bfloat16)
    shared_gate_up_wide = ttnn.concat([gate_wide, up_wide], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,H,2Ip]
    gate_wide.deallocate(True)
    up_wide.deallocate(True)
    shared_gate_up = _cast_to(shared_gate_up_wide, gate_up.dtype)
    shared_down = _cast_to(w_down, down.dtype)  # whole tile rows along dim 1: the page-copy path, exact

    fused_gate_up = ttnn.concat([gate_up, shared_gate_up], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gate_up.deallocate(True)
    shared_gate_up.deallocate(True)
    fused_down = ttnn.concat([down, shared_down], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    down.deallocate(True)
    shared_down.deallocate(True)

    return ExpertWeights(
        gate_up_proj=fused_gate_up,
        down_proj=fused_down,
        intermediate_size_per_device=weights.intermediate_size_per_device,
        intermediate_padded_per_device=ip,
        num_always_on_experts=weights.num_always_on_experts + 1,
    )
