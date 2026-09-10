# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This module defines the MeshConfig class which manages parallelization strategies
across a mesh of devices for the Solar-Open MoE model, and the MoEOptions flag bundle
(expert dtypes, router implementation) that is threaded from create_tt_model down to the MoE modules.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import ttnn


class Mode(Enum):
    """Execution mode for model forward pass"""

    DECODE = "decode"
    PREFILL = "prefill"


@dataclass
class ModeConfig:
    """Per-mode parallelization configuration"""

    tp: int  # Tensor parallel size
    ep: int = 1  # Expert parallel size
    sp: int = 1  # Sequence parallel size

    def __post_init__(self):
        if self.tp < 1 or self.ep < 1 or self.sp < 1:
            raise ValueError(f"Parallelism values must be >= 1: tp={self.tp}, ep={self.ep}, sp={self.sp}")


class MeshConfig:
    """Mode-aware mesh parallelization with dataclass-based mode configs"""

    def __init__(
        self,
        mesh_shape,
        decode: ModeConfig,
        prefill: ModeConfig = None,
        tp_axis: int = 1,
    ):
        """
        Args:
            mesh_shape: (rows, cols) - any mesh size
            decode: ModeConfig for decode mode
            prefill: ModeConfig for prefill mode (defaults to tp=decode.tp, sp=rows, ep=1)
            tp_axis: Which mesh axis is TP (0=rows, 1=cols, default: 1)

        Default behavior:
            - Decode: Typically ep=rows (expert-parallel), sp=1
            - Prefill: Automatically sp=rows (sequence-parallel), ep=1
        """
        self.mesh_shape = tuple(mesh_shape)
        self.tp_axis = tp_axis
        self.ep_axis = 0 if tp_axis == 1 else 1
        self.sp_axis = self.ep_axis

        self.total_devices = mesh_shape[0] * mesh_shape[1]

        # Store mode configs
        self.decode = decode
        # Default prefill: Same TP, use rows for SP (sequence parallel), EP=1
        self.prefill = prefill or ModeConfig(tp=decode.tp, sp=mesh_shape[0], ep=1)

        # Validate both configs
        self._validate_config(self.decode, Mode.DECODE)
        self._validate_config(self.prefill, Mode.PREFILL)

        # Legacy attributes point to decode config
        self.tp = self.decode.tp
        self.ep = self.decode.ep
        self.sp = self.decode.sp
        self.dp = self.total_devices // (self.decode.tp * self.decode.ep)

    def _validate_config(self, config: ModeConfig, mode: Mode):
        """Validate a mode config fits the mesh"""
        dp = self.total_devices // (config.tp * config.ep)
        if config.tp * dp * config.ep != self.total_devices:
            raise ValueError(
                f"{mode.value}: TP({config.tp}) × DP({dp}) × EP({config.ep}) != total_devices({self.total_devices})"
            )

        tp_dim_size = self.mesh_shape[self.tp_axis]
        if config.tp > tp_dim_size:
            raise ValueError(f"{mode.value}: TP({config.tp}) > mesh_{self.tp_axis}_size({tp_dim_size})")

        # EP>1 makes the expert dimension sharded across ep_axis, and ttnn.moe_routing_remap
        # requires expert_parallel_size to equal that axis extent exactly. Catch it here rather
        # than as a device-side TT_FATAL mid-forward. EP=1 is unconstrained: prefill runs EP=1
        # on multi-row meshes and never reaches the remap.
        ep_dim_size = self.mesh_shape[self.ep_axis]
        if config.ep > 1 and config.ep != ep_dim_size:
            raise ValueError(f"{mode.value}: EP({config.ep}) != mesh_{self.ep_axis}_size({ep_dim_size})")

    def get_config(self, mode: Mode) -> ModeConfig:
        """Type-safe mode config access"""
        return self.decode if mode == Mode.DECODE else self.prefill

    def shard_mapper(self, mesh_device, tensor_dim=None, mesh_dims=None, mode: Mode = Mode.DECODE):
        """Unified 2D sharding - replaces all individual mappers"""
        if mesh_dims is None:
            # Default: shard along TP axis only
            mesh_dims = (None, tensor_dim) if self.tp_axis == 1 else (tensor_dim, None)

        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=mesh_dims)

    # Clean semantic helpers (all use unified shard_mapper)
    def column_parallel(self, mesh_device):
        """Column-parallel weights (feature dimension sharding)"""
        return self.shard_mapper(mesh_device, tensor_dim=-1)

    def row_parallel(self, mesh_device):
        """Row-parallel weights (sequence/batch dimension sharding)"""
        return self.shard_mapper(mesh_device, tensor_dim=-2)

    def sequence_parallel(self, mesh_device):
        """Sequence sharding (for KV cache)"""
        return self.shard_mapper(mesh_device, tensor_dim=-3)

    def shard_size(self, total_size, mode: Mode = Mode.DECODE):
        """Size per device for tensor parallel sharding"""
        config = self.get_config(mode)
        return total_size // config.tp

    def allreduce(self, tensor, ccl_manager, memory_config=None, pad_size=None, axis=0):
        """
        General tensor parallel allreduce (reduce-scatter + all-gather)

        Note: Caller should check if communication is needed before calling
        """
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        # Optional performance padding (caller specifies, no magic numbers)
        padded = False
        if pad_size and tensor.shape[-2] >= 32:
            tensor_padded = ttnn.pad(tensor, [(0, 0), (0, 0), (0, 0), (0, pad_size)], 0)
            tensor.deallocate(True)
            tensor = tensor_padded
            padded = True

        # Reduce-scatter along TP axis
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            topology=ccl_manager.topology,
            cluster_axis=axis,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )
        # Free the full-size input (~94 MiB at ISL=16384) before the
        # all-gather allocates its full-size output. Without this, peak
        # live memory inside allreduce is tensor + scattered + gathered
        # (~200 MiB at ISL=16384) which fragments DRAM under
        # long-context prefill — see tt-shield run 26440169327 OOM.
        # Callers must NOT use `tensor` after this returns (they don't:
        # apply_allreduce assigns the return value and deallocates the
        # original handle, which becomes a no-op).
        tensor.deallocate(True)

        # All-gather back
        gathered = ttnn.experimental.all_gather_async(
            scattered,
            dim=3,
            cluster_axis=axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )
        scattered.deallocate(True)

        # Remove padding if applied
        if padded:
            gathered_sliced = gathered[:, :, :, :-pad_size]
            gathered.deallocate(True)
            gathered = gathered_sliced
        return gathered

    def allgather(self, tensor, ccl_manager, memory_config=None, axis=0, dim=3, linear=False):
        """
        All-gather operation for tensor parallel communication

        Note: Caller should check if communication is needed before calling
        """
        memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG

        return ttnn.experimental.all_gather_async(
            tensor,
            dim=dim,
            cluster_axis=axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ttnn.Topology.Linear if linear else ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=memory_config,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )

    def __repr__(self):
        decode_dp = self.total_devices // (self.decode.tp * self.decode.ep)
        prefill_dp = self.total_devices // (self.prefill.tp * self.prefill.ep)
        decode_str = f"decode[TP={self.decode.tp}, EP={self.decode.ep}, SP={self.decode.sp}, DP={decode_dp}]"
        prefill_str = f"prefill[TP={self.prefill.tp}, EP={self.prefill.ep}, SP={self.prefill.sp}, DP={prefill_dp}]"
        return f"MeshConfig({self.mesh_shape}, {decode_str}, {prefill_str})"


# Convenience factory functions for common configurations
def mesh_2x4():
    # decode: TP=4, EP=2, DP=1; prefill: TP=4, SP=2, EP=1, DP=1
    return MeshConfig((2, 4), decode=ModeConfig(tp=4, ep=2))


def mesh_4x8():
    # decode: TP=8, EP=4, DP=1; prefill: TP=8, SP=4, EP=1, DP=4
    return MeshConfig((4, 8), decode=ModeConfig(tp=8, ep=4))


def mesh_4x4():
    # decode: TP=4, EP=4, DP=1; prefill: TP=4, SP=4, EP=1, DP=1
    return MeshConfig((4, 4), decode=ModeConfig(tp=4, ep=4))


def mesh_1x8():
    # decode: TP=8, EP=1, DP=1; prefill: TP=8, SP=1, EP=1, DP=1 (no SP on single row)
    return MeshConfig((1, 8), decode=ModeConfig(tp=8, ep=1))


@dataclass(frozen=True)
class MoEOptions:
    """Solar-Open MoE knobs (design D10). The defaults are the bring-up configuration; ``from_env()`` applies the
    ``SOLAR_OPEN_*`` overrides. One instance is threaded ``create_tt_model -> Model -> DecoderLayer -> MLP ->
    {TopKRouter, Experts, SharedExpert}``; ``None`` at any of those sites means ``MoEOptions()``.

    Attributes:
        expert_dtype: routed expert weight dtype, ``SOLAR_OPEN_EXPERT_DTYPE`` bfp8 (default) | bfp4. Folded into
            the weight-cache directory name (``tensor_cache_<dtype>_exp<expert_dtype>_<mesh>``).
        shared_expert_dtype: shared expert weight dtype, ``SOLAR_OPEN_SHARED_EXPERT_DTYPE`` bfp8 (default) | bf16.
            Weights only: the shared partial is emitted in bf16 (bfloat8_b at decode with ``shared_down_bfp8``).
        router_impl: ``SOLAR_OPEN_ROUTER_IMPL`` "fused" (``moe_grouped_topk``, 3 launches, default) | "ops"
            (pure ttnn op chain, exact fp32 sigmoid, ~10 launches).
        router_fp32_logits: ``SOLAR_OPEN_ROUTER_FP32_LOGITS`` 1 (default) | 0. With 0 the router matmul emits
            bf16 logits (fp32-accumulated) and the fused op takes a bf16 bias; the selection math still runs in
            fp32 in both implementations.
        fuse_shared_expert: ``SOLAR_OPEN_FUSE_SHARED_EXPERT`` 0 (default) | 1. With 1 the shared expert runs as the
            always-on slot ``num_local_experts`` (the "129th expert") of the routed expert tensors with the constant
            routing weight 1.0 instead of as a separate ``SharedExpert`` module (phase 2, design D2 follow-up: -5
            launches per layer). The fused tensors are built ON DEVICE from the cached routed + shared shards, so the
            flag changes no cache file and is deliberately NOT part of ``marker_fields()``. Off until validated
            (fused-vs-unfused PCC, teacher-forced accuracy and the decode step time), then on.
        indexed_decode: ``SOLAR_OPEN_INDEXED_DECODE`` 1 (default) | 0. With 1 a single-user decode step (one token
            on the mesh row) runs the routed experts in the sparse_matmul INDEXED/GATHER mode (phase 2, profile lever
            1): the router hands the experts its top-k expert ids (uint16) and weights directly
            (``TopKRouter.route_indexed`` -> ``experts.IndexedRouting``) instead of the dense ``[1, E]`` routing
            tensor, both sparse_matmuls visit only the k selected experts and emit compact ``[1, k, 1, *]`` outputs
            (no 128-slot sparsity scan, no zero-filled full-E outputs, no bfp8 transpose glue). Batched steps (2..32
            users) always take the union-of-experts path. Requires the fused router (uint16 ids), EP=1 and the
            unfused shared expert; otherwise the flag is ignored (logged once per MLP). Cache-neutral (not in
            ``marker_fields()``). 0 restores the phase-1 single-user scan path for A/B runs.
        shared_down_bfp8: ``SOLAR_OPEN_SHARED_DOWN_BFP8`` 1 (default since phase 3e / A3) | 0. With 1 the unfused
            shared expert emits its
            DECODE down projection (the per-device partial the routed experts add in place to their bfloat8_b partial
            before the single TP all_reduce) in bfloat8_b instead of bf16, so that add is a same-dtype op (phase 3e /
            A3, design_decode_levers.md 2.6 (a): the mixed-dtype in-place add on the single-user [1, 1, 1, H] partial
            costs 12.6 us per layer, the bfp8 += bfp8 one ~2 us; measured traced real layer 0 b1 0.310 -> 0.299 ms,
            demo b1 14.67 -> 14.14 ms/step, b32 neutral). Not bit-identical: the shared partial is rounded to bfp8
            before instead of after the add (teacher-forced floors held, digits in the README). Prefill partials
            stay bf16 (the prefill path is unchanged byte for byte) and the fused shared expert
            (``fuse_shared_expert``) has no separate partial, so the flag is ignored there. Runtime-only (weights are
            untouched): cache-neutral, not in ``marker_fields()``. 0 = the phase-3d / A2 behaviour (bf16 partial).
    """

    expert_dtype: ttnn.DataType = ttnn.bfloat8_b
    shared_expert_dtype: ttnn.DataType = ttnn.bfloat8_b
    router_impl: str = "fused"
    router_fp32_logits: bool = True
    fuse_shared_expert: bool = False
    indexed_decode: bool = True
    shared_down_bfp8: bool = True

    ROUTER_IMPLS: ClassVar[tuple] = ("fused", "ops")
    EXPERT_DTYPES: ClassVar[tuple] = ("bfp8", "bfp4")
    SHARED_EXPERT_DTYPES: ClassVar[tuple] = ("bfp8", "bf16")
    _DTYPE_FROM_STR: ClassVar[dict] = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "bf16": ttnn.bfloat16}

    def __post_init__(self):
        if self.router_impl not in self.ROUTER_IMPLS:
            raise ValueError(f"router_impl must be one of {self.ROUTER_IMPLS}, got {self.router_impl!r}")
        if self.expert_dtype not in {self._DTYPE_FROM_STR[s] for s in self.EXPERT_DTYPES}:
            raise ValueError(f"expert_dtype must be one of {self.EXPERT_DTYPES}, got {self.expert_dtype}")
        if self.shared_expert_dtype not in {self._DTYPE_FROM_STR[s] for s in self.SHARED_EXPERT_DTYPES}:
            raise ValueError(
                f"shared_expert_dtype must be one of {self.SHARED_EXPERT_DTYPES}, got {self.shared_expert_dtype}"
            )

    @classmethod
    def _dtype_from_env(cls, var, default, allowed):
        value = os.getenv(var, default)
        if value not in allowed:
            raise ValueError(f"{var}={value!r} is not supported; choose one of {allowed}")
        return cls._DTYPE_FROM_STR[value]

    @classmethod
    def from_env(cls) -> "MoEOptions":
        """Build the options from the SOLAR_OPEN_* environment variables (unset -> Solar bring-up defaults)."""
        impl = os.getenv("SOLAR_OPEN_ROUTER_IMPL", "fused")
        if impl not in cls.ROUTER_IMPLS:
            raise ValueError(f"SOLAR_OPEN_ROUTER_IMPL={impl!r} is not supported; choose one of {cls.ROUTER_IMPLS}")
        return cls(
            expert_dtype=cls._dtype_from_env("SOLAR_OPEN_EXPERT_DTYPE", "bfp8", cls.EXPERT_DTYPES),
            shared_expert_dtype=cls._dtype_from_env("SOLAR_OPEN_SHARED_EXPERT_DTYPE", "bfp8", cls.SHARED_EXPERT_DTYPES),
            router_impl=impl,
            router_fp32_logits=os.getenv("SOLAR_OPEN_ROUTER_FP32_LOGITS", "1") == "1",
            fuse_shared_expert=os.getenv("SOLAR_OPEN_FUSE_SHARED_EXPERT", "0") == "1",
            indexed_decode=os.getenv("SOLAR_OPEN_INDEXED_DECODE", "1") == "1",
            shared_down_bfp8=os.getenv("SOLAR_OPEN_SHARED_DOWN_BFP8", "1") == "1",
        )

    @classmethod
    def dtype_str(cls, dtype) -> str:
        """Short name of a ttnn weight dtype: "bfp8" | "bfp4" | "bf16"."""
        return {v: k for k, v in cls._DTYPE_FROM_STR.items()}[dtype]

    @property
    def expert_dtype_str(self) -> str:
        """ "bfp8" | "bfp4" - used in the weight-cache directory name."""
        return self.dtype_str(self.expert_dtype)

    def marker_fields(self) -> dict:
        """JSON-serialisable record written into the weight cache's .weights_complete marker; a cache built with
        different options is rejected by ModelArgs.weight_cache_is_complete.

        ``fuse_shared_expert`` is deliberately absent: both modes read the SAME cache files (the fused expert tensors
        are concatenated on device from the cached routed and shared shards), so flipping it must not invalidate the
        marker or force a cold load. ``indexed_decode`` and ``shared_down_bfp8`` are absent for the same reason (pure
        runtime choices over the same weight tensors)."""
        return {
            "expert_dtype": self.expert_dtype_str,
            "shared_expert_dtype": self.dtype_str(self.shared_expert_dtype),
            "router_impl": self.router_impl,
            "router_fp32_logits": self.router_fp32_logits,
        }
