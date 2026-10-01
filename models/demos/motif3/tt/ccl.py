# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Axis-aware collectives for Motif-3 on the BH Galaxy (design §1.3, §2.3.4-2.3.8, §3.2-3.3).

Draft 1 uses only the generic, semaphore-free ``ttnn.all_gather`` / ``ttnn.reduce_scatter`` / ``ttnn.all_reduce``
(plus the per-device slice ``ttnn.mesh_partition``). That is the set GPT-OSS-120B validated on a 4x8 BH Galaxy
with mesh and torus fabrics, eager, cached and traced (commit ``55a2388c7ff``; pattern copied from
``models/demos/gpt_oss/tt/experts_throughput/gather_decode.py``, not imported). Consequences:

* No persistent semaphores or buffers are required: the ops create their internal semaphores at program-cache
  miss time, so the decode trace only needs every program compiled before capture (study 04 §9). An optional
  pre-allocated ``output_tensor`` can be passed to all_gather / reduce_scatter to pin output addresses.
* Topology and link count are resolved by the ops from the fabric config: ``FABRIC_2D_TORUS_XY`` (default)
  -> Torus -> Ring per axis (wrap-wired on both axes of the 8x4 torus); ``FABRIC_1D_RING`` (fallback) -> Ring.
  ``num_links`` defaults to the routing planes the fabric reports (2 on this Galaxy). For ``all_gather`` both
  arguments are deprecated no-ops and are never passed.
* Replica consistency (design §2.3.7 invariant): ``all_reduce`` is reduce-scatter + all-gather (or all-gather +
  local sum for small / unaligned payloads), so every chip of the reduced axis ends up with bitwise-identical
  data. Use :func:`replicas_identical` in tests to check it.
* Precision (measured, ``tests/unit/test_infra_device.py``, BH 4x8, 2026-10-01): bf16 reductions round like bf16
  (AR of 4-8 bf16 shards: PCC 0.999996-0.999997). fp32 inputs on the RS+AG path are reduced at **TF32 class**
  (``[1,1,32,4096]`` fp32 AR over DP: PCC 0.99999992, max-abs 8.9e-3 at |x| <= 10; an explicit HiFi4 / fp32-acc
  compute config on the reduce-scatter does not change it). Small fp32 payloads that take the all-gather +
  local-sum path are exact (``[1,1,8,32]`` AR over TP: max-abs 9.5e-7).
* Layout: an 8-row (sub-tile) all_gather in TILE takes the op's composite path (~4x the eager time of the same
  gather in ROW_MAJOR); see :meth:`MotifCCL.partition` for the TILE slice restriction.

Axis names. Every method takes ``axis`` as a role, never a raw cluster_axis:

=====================  ================================================  ==========================
name                   meaning                                           cluster_axis on (4,8)/(8,4)
=====================  ================================================  ==========================
``"tp"`` / ``"cols"``  across the 8 TP chips of one DP group (a row)       1 / 0
``"dp"`` / ``"rows"``  across the 4 DP groups (one TP index)               0 / 1
``0`` / ``1``          raw cluster_axis (escape hatch)                     as given
=====================  ================================================  ==========================

Motif payloads (per chip, decode): AR(tp) of ``[1,1,8,4096]`` after ``wo`` / MoE / dense MLP; AR(tp) of PolyNorm
moments ``[1,1,8,32]`` fp32; AG(dp) of ``[1,1,8,4096]`` -> ``[1,1,32,4096]`` (MoE token gather, lane order
``l = 8 dp + local``); AR(dp) of ``[1,1,32,4096]`` (MoE combine) followed by ``partition(dim=2, "dp")`` to keep this
row's 8 lanes. Prefill: RS(dp) ``[1,1,S,4096]`` -> ``[1,1,S/4,4096]``, AR(tp), AG(dp) back to ``[1,1,S,4096]``.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

import ttnn

from .model_config import MeshAxes, MotifTTConfig

Axis = Union[str, int]

_ROLE_ALIASES = {"tp": "tp", "cols": "tp", "col": "tp", "dp": "dp", "rows": "dp", "row": "dp", "ep": "dp"}


def resolve_axis(axes: MeshAxes, axis: Axis) -> int:
    """Role name (``"tp"``/``"cols"``, ``"dp"``/``"rows"``) or raw cluster_axis -> cluster_axis."""
    if isinstance(axis, str):
        role = _ROLE_ALIASES.get(axis.lower())
        if role is None:
            raise ValueError(f"unknown axis {axis!r}; use 'tp'/'cols', 'dp'/'rows' or 0/1")
        return axes.tp_axis if role == "tp" else axes.dp_axis
    axis = int(axis)
    if axis not in (0, 1):
        raise ValueError(f"cluster_axis must be 0 or 1, got {axis}")
    return axis


def topology_for_fabric(fabric_config) -> Optional["ttnn.Topology"]:
    """Explicit topology for reduce_scatter / all_reduce, or ``None`` to let the op derive it from the fabric
    (the default; the op maps Torus->Ring and Mesh->Linear per axis)."""
    if fabric_config in (ttnn.FabricConfig.FABRIC_1D_RING,):
        return ttnn.Topology.Ring
    if fabric_config in (ttnn.FabricConfig.FABRIC_1D,):
        return ttnn.Topology.Linear
    return None


class MotifCCL:
    """Thin, trace-safe wrappers around the generic ttnn collectives with Motif axis roles.

    Args:
        mesh_device: the opened mesh.
        cfg: a :class:`MotifTTConfig` (its ``axes``) -- or pass ``axes`` directly.
        num_links: links for reduce_scatter / all_reduce (``None`` = fabric routing planes, i.e. 2 here).
        topology: explicit ``ttnn.Topology`` for reduce_scatter / all_reduce (``None`` = derived from fabric).
        memory_config: default output memory config (DRAM interleaved).
    """

    def __init__(
        self,
        mesh_device,
        cfg: Optional[MotifTTConfig] = None,
        *,
        axes: Optional[MeshAxes] = None,
        num_links: Optional[int] = None,
        topology=None,
        memory_config=None,
    ):
        self.mesh_device = mesh_device
        if axes is None:
            axes = cfg.axes if cfg is not None else MeshAxes.detect(tuple(mesh_device.shape))
        if tuple(axes.mesh_shape) != tuple(mesh_device.shape):
            raise ValueError(f"axes are for mesh {axes.mesh_shape}, device mesh is {tuple(mesh_device.shape)}")
        self.axes = axes
        self.num_links = num_links
        self.topology = topology
        self.memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    # ---- axis helpers ------------------------------------------------------------------------------------
    def cluster_axis(self, axis: Axis) -> int:
        return resolve_axis(self.axes, axis)

    def axis_size(self, axis: Axis) -> int:
        return self.axes.mesh_shape[self.cluster_axis(axis)]

    def _kwargs_links_topology(self) -> dict:
        kw = {}
        if self.num_links is not None:
            kw["num_links"] = self.num_links
        if self.topology is not None:
            kw["topology"] = self.topology
        return kw

    # ---- collectives ---------------------------------------------------------------------------------------
    def all_gather(self, x, dim: int, axis: Axis, *, memory_config=None, output_tensor=None):
        """Concatenate the shards of ``axis`` along tensor ``dim`` (mesh order = DP/TP index order).

        A size-1 axis returns ``x`` unchanged. Inputs whose gather dim is tile-padded (e.g. 8 lanes in TILE layout)
        take the op's composite path automatically; results are identical either way.
        """
        ca = self.cluster_axis(axis)
        if self.axes.mesh_shape[ca] == 1:
            return x
        kw = {}
        if output_tensor is not None:
            kw["output_tensor"] = output_tensor
        return ttnn.all_gather(
            x,
            dim=dim,
            cluster_axis=ca,
            memory_config=memory_config if memory_config is not None else self.memory_config,
            **kw,
        )

    def reduce_scatter(
        self, x, dim: int, axis: Axis, *, memory_config=None, output_tensor=None, compute_kernel_config=None
    ):
        """Sum over ``axis`` and keep this chip's ``1/n`` slice of ``dim`` (slice index = chip index on ``axis``).

        Row-major inputs or non-tile-aligned output slices use the op's composite path. fp32 inputs accumulate in
        fp32 (the op enables fp32 dest acc for fp32 by default; pass ``compute_kernel_config`` to override).
        """
        ca = self.cluster_axis(axis)
        if self.axes.mesh_shape[ca] == 1:
            return x
        kw = self._kwargs_links_topology()
        if output_tensor is not None:
            kw["output_tensor"] = output_tensor
        if compute_kernel_config is not None:
            kw["compute_kernel_config"] = compute_kernel_config
        return ttnn.reduce_scatter(
            x,
            dim=dim,
            cluster_axis=ca,
            memory_config=memory_config if memory_config is not None else self.memory_config,
            **kw,
        )

    def all_reduce(self, x, axis: Axis, *, memory_config=None):
        """Sum over ``axis``; every chip of the axis gets the identical result (RS+AG or AG+local-sum).
        bf16 / bfp8 inputs reduce in bf16; fp32 inputs take the fp32 path."""
        ca = self.cluster_axis(axis)
        if self.axes.mesh_shape[ca] == 1:
            return x
        return ttnn.all_reduce(
            x,
            cluster_axis=ca,
            memory_config=memory_config if memory_config is not None else self.memory_config,
            **self._kwargs_links_topology(),
        )

    def partition(self, x, dim: int, axis: Axis, *, memory_config=None):
        """Per-device slice (no fabric traffic): chip ``i`` of ``axis`` keeps slice ``i`` of ``dim``.

        The inverse of ``all_gather``; e.g. after ``all_reduce(moe_partial [1,1,32,4096], "dp")`` keep this row's 8
        lanes with ``partition(x, 2, "dp")``. ``x.shape[dim]`` must divide by the axis size.

        ``ttnn.mesh_partition`` is a ``slice`` underneath, which rejects TILE slices of the last two dims that do not
        start / end on a tile boundary (measured on BH: 8 of 32 rows -> TT_FATAL "Can only slice tilized tensor with
        height begin index aligned to tiles"). Such slices are therefore done in ROW_MAJOR and re-tilized
        (2 extra layout ops). Avoid them on hot paths by keeping each group's lanes tile-aligned (e.g. pad each DP
        group to a full 32-row tile before the gather), so the slice is a whole number of tiles.
        """
        ca = self.cluster_axis(axis)
        n = self.axes.mesh_shape[ca]
        if n == 1:
            return x
        mc = memory_config if memory_config is not None else self.memory_config
        shape = list(x.shape)
        d = dim % len(shape)
        if shape[d] % n:
            raise ValueError(f"partition: dim {dim} of size {shape[d]} does not split over {n} chips")
        tile_unaligned = x.layout == ttnn.TILE_LAYOUT and d >= len(shape) - 2 and (shape[d] // n) % 32 != 0
        if not tile_unaligned:
            return ttnn.mesh_partition(x, dim, ca, memory_config=mc)
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        part = ttnn.mesh_partition(rm, dim, ca, memory_config=mc)
        ttnn.deallocate(rm)
        out = ttnn.to_layout(part, ttnn.TILE_LAYOUT, memory_config=mc)
        ttnn.deallocate(part)
        return out

    # ---- Motif role shortcuts ----------------------------------------------------------------------------
    def ar_tp(self, x, **kw):
        """All-reduce across the TP chips of a row (``wo``, MoE output, dense MLP, PolyNorm moments)."""
        return self.all_reduce(x, "tp", **kw)

    def ar_dp(self, x, **kw):
        """All-reduce across DP groups (decode MoE combine)."""
        return self.all_reduce(x, "dp", **kw)

    def ag_dp(self, x, dim: int = 2, **kw):
        """All-gather across DP groups (decode MoE token gather; prefill MoE output)."""
        return self.all_gather(x, dim, "dp", **kw)

    def rs_dp(self, x, dim: int = 2, **kw):
        """Reduce-scatter across DP groups (prefill MoE combine)."""
        return self.reduce_scatter(x, dim, "dp", **kw)

    def ag_tp(self, x, dim: int, **kw):
        return self.all_gather(x, dim, "tp", **kw)


# ------------------------------------------------------------------------------------------------------------
# Host-side readback helpers (tests / debugging; they synchronize and copy to host, never use inside a trace)
# ------------------------------------------------------------------------------------------------------------
def device_tensors_to_torch(t, mesh_device) -> torch.Tensor:
    """Every chip's local tensor, stacked: ``[R, C, *local_shape]`` with ``[r, c]`` = mesh coordinate (r, c).

    Works for local tensors of rank <= 4 (rank-4 local shape ``[a, b, h, w]`` is composed with a 2D concat on dims
    (0, 1) and un-interleaved on the host)."""
    R, C = (int(s) for s in tuple(mesh_device.shape))
    local = list(t.shape)
    rank = len(local)
    if rank > 4:
        raise ValueError(f"rank {rank} > 4 not supported")
    if rank < 2:
        raise ValueError("rank >= 2 required (unsqueeze first)")
    composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(R, C)))
    full = ttnn.to_torch(t, mesh_composer=composer)
    a, b = local[0], local[1]
    rest = list(full.shape[2:])
    full = full.reshape(R, a, C, b, *rest).permute(0, 2, 1, 3, *range(4, 4 + len(rest)))
    return full.contiguous()


def replicas_identical(t, mesh_device, axis: Axis, axes: Optional[MeshAxes] = None) -> bool:
    """True iff the chips along ``axis`` hold bitwise-identical tensors (for every index of the other axis)."""
    axes = axes or MeshAxes.detect(tuple(mesh_device.shape))
    ca = resolve_axis(axes, axis)
    shards = device_tensors_to_torch(t, mesh_device)  # [R, C, ...]
    ref = shards.select(ca, 0).unsqueeze(ca)
    return bool(torch.equal(shards, ref.expand_as(shards)))


__all__ = [
    "MotifCCL",
    "device_tensors_to_torch",
    "replicas_identical",
    "resolve_axis",
    "topology_for_fabric",
]
