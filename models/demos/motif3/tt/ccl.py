# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Axis-aware collectives for Motif-3 on the BH Galaxy (design §1.3, §2.3.4-2.3.8, §3.2-3.3).

Draft 1 uses only the generic ``ttnn.all_gather`` / ``ttnn.reduce_scatter`` / ``ttnn.all_reduce`` family (plus the
per-device slice ``ttnn.mesh_partition``) with op-internal semaphores. That is the set GPT-OSS-120B validated on a
4x8 BH Galaxy with mesh and torus fabrics, eager, cached and traced (commit ``55a2388c7ff``; pattern copied from
``models/demos/gpt_oss/tt/experts_throughput/gather_decode.py``, not imported). Consequences:

* No persistent semaphores or buffers have to be managed: the ops create their global semaphores at program-cache
  miss time and keep them with the cached program, so the decode trace only needs every program compiled before
  capture (study 04 §9). An optional pre-allocated ``output_tensor`` can be passed to all_gather / reduce_scatter.
* **Semaphores live in L1_SMALL, never in main L1** (WAVE_A_REVIEW attention P0; ``generator_api.L1_SMALL_SIZE``).
  A semaphore created in main L1 lands at whatever address is free when its program is first built (next to a live
  512 KB staging buffer or a module's L1 intermediates) and stays there for the life of the program cache; the first
  later program whose static circular buffers reach that address throws "Statically allocated circular buffers ...
  clash with L1 buffers" (global-layer prefill at S >= 1024 after a decode step, bf16-KV FlashMLA decode). When the
  mesh has an L1_SMALL region (``model_config.device_params()`` / the vLLM ``"tt"`` config set
  ``l1_small_size=32768``), every :class:`MotifCCL` collective puts its semaphores there (``l1_small_semaphores``):

  ======================================  ===========================================================================
  ttnn path (measured on this Galaxy)      MotifCCL with an L1_SMALL region
  ======================================  ===========================================================================
  all_gather, native (tile-aligned         ``ttnn.all_gather`` (the op picks L1_SMALL itself when the region exists;
  gather dim, ROW_MAJOR)                   its ``use_l1_small_for_semaphores`` argument is deprecated and ignored)
  all_gather, composite (gather dim        ``all_broadcast(use_l1_small_for_semaphores=True)`` + ``concat`` (the same
  padded, e.g. 8-row TILE): 2 sems in L1   ops ttnn's composite runs; results bitwise identical)
  reduce_scatter, direct (TILE, ring       ``ttnn.experimental.reduce_scatter_minimal_direct`` (the same prim; picks
  axis, <= 512 KiB per chip)               L1_SMALL itself; bitwise identical, 19.7 vs 19.5 us traced)
  reduce_scatter, ring / line / composite  ``ttnn.reduce_scatter(use_l1_small_for_semaphores=True)``
  (DP axis on TORUS_Y, prefill sizes,
  ROW_MAJOR): 4 sems in main L1
  all_reduce (no semaphore argument)       Python mirror of ``ttnn.all_reduce`` (``all_reduce_async``): the same scatter
                                           dim and RS + AG / AG + local-sum choice, built from the two rows above
                                           (results bitwise identical to ``ttnn.all_reduce``)
  mesh_partition                           no fabric, no semaphores
  ======================================  ===========================================================================

  ``tests/unit/test_infra_l1_small.py`` asserts, for every draft-1 payload, that main L1 is unchanged after a
  program-cache miss (the plain ops left 256 B = 4 semaphores per DP-axis / prefill all-reduce and reduce-scatter,
  128 B per composite gather) and that the results equal the plain ttnn ops bitwise; 4.7 KiB of the 32 KiB region
  hold every draft-1 CCL's semaphores. Cost (same device programs): traced identical (AR(tp) ``[8,4096]`` 33.5 vs
  33.5 us, AR(dp) ``[32,4096]`` 33.1 vs 33.3, AR(tp) ``[1024,4096]`` 251.8 vs 250.5); eager +23-67 us per all-reduce
  (two Python-dispatched ops instead of one C++ call; decode is traced, prefill pays it ~100x per pass). Dispatch
  decisions are cached per tensor spec. Without an L1_SMALL region (or with ``l1_small_semaphores=False``) the plain
  ttnn ops run unchanged (the legacy behaviour, with the hazard above). Upstream fix that would retire the mirror:
  give ``ttnn.all_reduce`` a ``use_l1_small_for_semaphores`` argument, or make the ring reduce-scatter and
  ``all_broadcast`` default to L1_SMALL like ``all_gather`` and the direct reduce-scatter do.
* Topology and link count are resolved by the ops from the fabric config: ``FABRIC_2D_TORUS_XY`` (default)
  -> Torus -> Ring per axis; ``FABRIC_1D_RING`` (fallback) -> Ring. ``num_links`` defaults to the routing planes the
  fabric reports (2 on this Galaxy). For ``all_gather`` both arguments are deprecated no-ops and are never passed.
* The topology mapper may commit less than was requested: since 2026-10-01 17:31 every open of this Galaxy with
  ``FABRIC_2D_TORUS_XY`` committed ``TORUS_Y`` (the 8-chip TP axis a ring, the 4-chip DP axis a **line**;
  WAVE_A_REVIEW §0.3; ``ttnn.get_usable_topology``: TP ``Torus``, DP ``Mesh``). Results stay correct; DP-axis CCL
  latencies differ, and DP-axis reduce-scatters take the line (ring-factory) path. :func:`fabric_report` /
  :func:`log_fabric` report the committed type -- every device test logs it (grep ``committed:``).
* Replica consistency (design §2.3.7 invariant): ``all_reduce`` is reduce-scatter + all-gather (or all-gather +
  local sum for small / unaligned payloads), so every chip of the reduced axis ends up with bitwise-identical
  data. Use :func:`replicas_identical` in tests to check it.
* Precision (measured, ``tests/unit/test_infra_device.py``, BH 4x8, 2026-10-01): bf16 reductions round like bf16
  (AR of 4-8 bf16 shards: PCC 0.999996-0.999997). fp32 inputs on the RS+AG path are reduced at **TF32 class**
  (``[1,1,32,4096]`` fp32 AR over DP: PCC 0.99999992, max-abs 8.9e-3 at |x| <= 10; an explicit HiFi4 / fp32-acc
  compute config on the reduce-scatter does not change it). Small fp32 payloads that take the all-gather +
  local-sum path are exact (``[1,1,8,32]`` AR over TP: max-abs 9.5e-7); :meth:`MotifCCL.ar_exact` is the explicit
  exact all-reduce for small payloads (any row count; 2-3 ops instead of the composite's ~7).
* Size-1 axes (small test meshes, e.g. (1, 8)): every collective returns a **new** tensor (a ``ttnn.clone`` in the
  requested memory config), never its input, so callers may free the input whatever the mesh (README §10 rule 3).
* Layout: an 8-row (sub-tile) all_gather in TILE takes the op's composite path (~4x the eager time of the same
  gather in ROW_MAJOR; gate G4 traced: 62 us TILE vs 12 us ROW_MAJOR). The MoE token gather therefore uses
  :meth:`MotifCCL.ag_dp_rows` (untilize -> ROW_MAJOR all_gather -> tilize, 26.5-28 us traced in all); see
  :meth:`MotifCCL.partition` for the TILE slice restriction.

Axis names. Every method takes ``axis`` as a role, never a raw cluster_axis:

=====================  ================================================  ==========================
name                   meaning                                           cluster_axis on (4,8)/(8,4)
=====================  ================================================  ==========================
``"tp"`` / ``"cols"``  across the 8 TP chips of one DP group (a row)       1 / 0
``"dp"`` / ``"rows"``  across the 4 DP groups (one TP index)               0 / 1
``0`` / ``1``          raw cluster_axis (escape hatch)                     as given
=====================  ================================================  ==========================

Motif payloads (per chip, decode): AR(tp) of ``[1,1,8,4096]`` after ``wo`` / MoE / dense MLP; exact AR(tp) of the
PolyNorm moments (``ar_exact``); AG(dp) of ``[1,1,8,4096]`` -> ``[1,1,32,4096]`` (MoE token gather, lane order
``l = 8 dp + local``, via ``ag_dp_rows``); AR(dp) of ``[1,1,32,4096]`` (MoE combine) followed by
``partition(dim=2, "dp")`` to keep this row's 8 lanes. Prefill: RS(dp) ``[1,1,S,4096]`` -> ``[1,1,S/4,4096]``, AR(tp),
AG(dp) back to ``[1,1,S,4096]``.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Union

import torch

import ttnn

from .model_config import COMPUTE_ROLES, MeshAxes, MotifTTConfig, active_fabric_name, make_compute_kernel_config
from .model_config import mesh_l1_small_bytes

Axis = Union[str, int]

_ROLE_ALIASES = {"tp": "tp", "cols": "tp", "col": "tp", "dp": "dp", "rows": "dp", "row": "dp", "ep": "dp"}

TILE = 32
# ttnn reduce_scatter.cpp ``k_direct_rs_max_input_bytes``: the direct (one-shot) reduce-scatter is used up to this many
# bytes of per-chip input; above it the ring factory wins (measured upstream on BH).
DIRECT_RS_MAX_INPUT_BYTES = 512 * 1024
# Bytes of one 32 x 32 tile per dtype (buffer page size of a TILE tensor).
_TILE_BYTES = {"BFLOAT16": 2048, "FLOAT32": 4096, "BFLOAT8_B": 1088, "BFLOAT4_B": 576, "UINT32": 4096, "INT32": 4096,
               "UINT16": 2048, "UINT8": 1024}  # fmt: skip
_ELEM_BYTES = {"BFLOAT16": 2, "FLOAT32": 4, "UINT32": 4, "INT32": 4, "UINT16": 2, "UINT8": 1}
# Conservative row alignment for ROW_MAJOR gathers on the last dim (DRAM 64 B on BH; L1 16 B): rows that are not a
# multiple take ttnn's composite all_gather, which MotifCCL replaces by its L1_SMALL composite.
_RM_ROW_ALIGN = 64

_WARNED_NO_L1_SMALL = False


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


# ------------------------------------------------------------------------------------------------------------
# Host-side mirrors of ttnn's CCL dispatch predicates (pure functions of shapes / layouts; unit-tested on the host)
# ------------------------------------------------------------------------------------------------------------
def _norm_dim(dim: int, rank: int) -> int:
    d = int(dim)
    d = d + rank if d < 0 else d
    if not 0 <= d < rank:
        raise ValueError(f"dim {dim} out of range for rank {rank}")
    return d


def _normalize_dim_4d(dim: int, rank: int) -> int:
    """``composite_common::normalize_dim_4d`` (the 4D index of ``dim``)."""
    if rank == 2:
        return 2 + dim
    diff = rank - 4
    return dim if dim < abs(diff) else dim - diff


def finding_scatter_dim(padded_shape, logical_rank: int, tiled: bool, num_devices: int) -> int:
    """``all_reduce_async.cpp:detail::finding_scatter_dim``: the last dim whose size (in tiles for the two innermost
    dims of a TILE tensor) divides by ``num_devices``; ``logical_rank`` if none does (-> the AG + local-sum path)."""
    vec = [int(d) for d in list(padded_shape)[-logical_rank:]]
    if tiled:
        for i in range(1, min(2, len(vec)) + 1):
            vec[-i] //= TILE
    for i in range(len(vec) - 1, -1, -1):
        if vec[i] % num_devices == 0:
            return i
    return logical_rank


def composite_ag_for_ar(logical_shape, tiled: bool, dim: int) -> bool:
    """``composite_common::use_composite_all_gather`` (all_reduce's AG predicate): ROW_MAJOR, or TILE padded on the
    gather dim."""
    shape = [int(d) for d in logical_shape]
    rank = len(shape)
    if not tiled:
        return True
    d = _norm_dim(dim, rank)
    return (d == rank - 2 and shape[-2] % TILE != 0) or (d == rank - 1 and shape[-1] % TILE != 0)


def composite_rs(logical_shape, tiled: bool, dim: int, num_devices: int) -> bool:
    """``composite_common::use_composite_reduce_scatter``: ROW_MAJOR, or a TILE output slice that is not tile aligned
    on the two innermost dims (only when the dim splits evenly)."""
    shape = [int(d) for d in logical_shape]
    rank = len(shape)
    d = _norm_dim(dim, rank)
    if shape[d] % num_devices:
        return False
    if not tiled:
        return True
    n4 = _normalize_dim_4d(d, rank)
    out = shape[d] // num_devices
    return (n4 == 3 and out % TILE != 0) or (n4 == 2 and out % TILE != 0)


def direct_rs_eligible(logical_shape, padded_shape, dtype_name: str, tiled: bool, sharded: bool, dim: int,
                       num_devices: int, ring_axis: bool) -> bool:  # fmt: skip
    """Whether ``ttnn.reduce_scatter`` would take its direct (one-shot) path (``reduce_scatter.cpp``
    ``use_direct_reduce_scatter`` + ``reduce_scatter_minimal_direct_is_applicable``) for an op with no explicit
    tuning arguments: TILE, a ring / torus axis, whole-page slices of the scatter dim (tiles for the two innermost
    dims, which must not be padded) and at most :data:`DIRECT_RS_MAX_INPUT_BYTES` of per-chip input. Sharded inputs
    are reported ineligible (conservative: they take the ring path with L1_SMALL semaphores)."""
    if not tiled or sharded or not ring_axis or num_devices < 2:
        return False
    shape = [int(d) for d in logical_shape]
    padded = [int(d) for d in padded_shape]
    rank = len(padded)
    if rank < 2 or len(shape) != rank:
        return False
    d = _norm_dim(dim, rank)
    pages = padded[d]
    if d >= rank - 2:
        if padded[d] != shape[d]:
            return False
        pages //= TILE
    if pages <= 0 or pages % num_devices:
        return False
    tile_bytes = _TILE_BYTES.get(dtype_name)
    if tile_bytes is None:
        return False
    tiles = math.prod(padded) // (TILE * TILE)
    return tiles * tile_bytes <= DIRECT_RS_MAX_INPUT_BYTES


def composite_ag(logical_shape, padded_shape, tiled: bool, dtype_name: str, dim: int) -> bool:
    """Whether ``ttnn.all_gather`` takes its composite (all_broadcast + concat) path for an interleaved tensor
    (``all_gather.cpp:use_composite_all_gather``): the gather dim is padded (e.g. 8 lanes of a TILE tensor), or a
    ROW_MAJOR gather on the last dim with rows that are not a multiple of the memory alignment (conservative)."""
    shape = [int(d) for d in logical_shape]
    padded = [int(d) for d in padded_shape]
    rank = len(shape)
    d = _norm_dim(dim, rank)
    if shape[d] != padded[d]:
        return True
    if not tiled and d == rank - 1:
        row = shape[-1] * _ELEM_BYTES.get(dtype_name, 1)
        return row % _RM_ROW_ALIGN != 0
    return False


def _dtype_name(t) -> str:
    dt = t.dtype
    return getattr(dt, "name", str(dt).split(".")[-1])


def _is_tiled(t) -> bool:
    return t.layout == ttnn.TILE_LAYOUT


def _is_sharded(t) -> bool:
    try:
        return bool(t.memory_config().is_sharded())
    except Exception:  # pragma: no cover - host tensors
        return False


# ------------------------------------------------------------------------------------------------------------
# The collectives
# ------------------------------------------------------------------------------------------------------------
class MotifCCL:
    """Thin, trace-safe wrappers around the generic ttnn collectives with Motif axis roles.

    Args:
        mesh_device: the opened mesh.
        cfg: a :class:`MotifTTConfig` (its ``axes``) -- or pass ``axes`` directly.
        num_links: links for reduce_scatter / all_reduce (``None`` = fabric routing planes, i.e. 2 here).
        topology: explicit ``ttnn.Topology`` for reduce_scatter / all_reduce (``None`` = derived from fabric).
        memory_config: default output memory config (DRAM interleaved).
        l1_small_semaphores: route every CCL's global semaphores to L1_SMALL (module docstring). ``None`` (default)
            = on iff the mesh has an L1_SMALL region; ``True`` requires one (ValueError otherwise); ``False`` = the
            plain ttnn ops (legacy behaviour, semaphores partly in main L1).
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
        l1_small_semaphores: Optional[bool] = None,
    ):
        global _WARNED_NO_L1_SMALL
        self.mesh_device = mesh_device
        if axes is None:
            axes = cfg.axes if cfg is not None else MeshAxes.detect(tuple(mesh_device.shape))
        if tuple(axes.mesh_shape) != tuple(mesh_device.shape):
            raise ValueError(f"axes are for mesh {axes.mesh_shape}, device mesh is {tuple(mesh_device.shape)}")
        self.axes = axes
        self.num_links = num_links
        self.topology = topology
        self.memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        self.l1_small_bytes = mesh_l1_small_bytes(mesh_device)
        has_region = bool(self.l1_small_bytes)
        if l1_small_semaphores is None:
            l1_small_semaphores = has_region
        elif l1_small_semaphores and not has_region:
            raise ValueError(
                f"MotifCCL(l1_small_semaphores=True) needs a mesh opened with an L1_SMALL region (l1_small_size); this "
                f"mesh has {self.l1_small_bytes} B per core"
            )
        self.l1_small_semaphores = bool(l1_small_semaphores)
        if self.l1_small_bytes == 0 and not _WARNED_NO_L1_SMALL:
            _WARNED_NO_L1_SMALL = True
            warnings.warn(
                "MotifCCL: the mesh has no L1_SMALL region, so CCL global semaphores are allocated in main L1 and can "
                "clash with later static circular buffers (README 'Running tests'; generator_api.L1_SMALL_SIZE). Open "
                "the mesh with model_config.device_params() / open_motif_mesh() or l1_small_size=32768.",
                RuntimeWarning,
                stacklevel=2,
            )
        role = COMPUTE_ROLES["ccl_reduce"]
        self._reduce_ckc = make_compute_kernel_config(
            role.fidelity, role.fp32_acc, approx=role.approx, packer_l1_acc=role.packer_l1_acc
        )
        self._ring_axis: Dict[int, bool] = {}  # cluster_axis -> usable topology is a ring / torus
        self._plans: Dict[tuple, Any] = {}  # (op, cluster_axis, tensor spec) -> dispatch decision (host-side cache)

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

    def _mc(self, memory_config):
        return memory_config if memory_config is not None else self.memory_config

    @staticmethod
    def _fresh(x, mc):
        """A new tensor equal to ``x`` in ``mc`` (size-1 axes: never hand the caller's input back)."""
        return ttnn.clone(x, memory_config=mc)

    def is_ring_axis(self, x, ca: int) -> bool:
        """``ttnn.get_usable_topology`` of axis ``ca`` (with and without the explicit topology) is a ring / torus.
        Cached per axis: it depends on the fabric and the mesh, not on the tensor's contents."""
        if ca not in self._ring_axis:
            rings = (ttnn.Topology.Ring, ttnn.Topology.Torus)
            t0 = ttnn.get_usable_topology(x, cluster_axis=ca)
            t1 = ttnn.get_usable_topology(x, topology=self.topology, cluster_axis=ca) if self.topology else t0
            self._ring_axis[ca] = t0 in rings and t1 in rings
        return self._ring_axis[ca]

    # ---- collectives ---------------------------------------------------------------------------------------
    def all_gather(self, x, dim: int, axis: Axis, *, memory_config=None, output_tensor=None):
        """Concatenate the shards of ``axis`` along tensor ``dim`` (mesh order = DP/TP index order).

        A size-1 axis returns a copy of ``x``. Inputs whose gather dim is tile-padded (e.g. 8 lanes in TILE layout)
        take the composite path (all_broadcast + concat; with L1_SMALL semaphores when the mesh has the region);
        results are identical either way.
        """
        ca = self.cluster_axis(axis)
        mc = self._mc(memory_config)
        if self.axes.mesh_shape[ca] == 1:
            return self._fresh(x, mc)
        if self.l1_small_semaphores:
            key = ("ag", dim, self._spec_key(x))
            comp = self._plans.get(key)
            if comp is None:
                comp = self._plans[key] = composite_ag(x.shape, x.padded_shape, _is_tiled(x), _dtype_name(x), dim)
            if comp:
                return self._all_gather_composite(x, dim, ca, mc)  # ttnn's composite ignores output_tensor as well
        kw = {}
        if output_tensor is not None:
            kw["output_tensor"] = output_tensor
        return ttnn.all_gather(x, dim=dim, cluster_axis=ca, memory_config=mc, **kw)

    def _all_gather_composite(self, x, dim: int, ca: int, mc):
        """ttnn's composite all_gather (``composite_common::composite_all_gather``: sharded -> interleaved, bfp8 with
        unaligned tiles -> bf16, ``all_broadcast``, ``concat``) with its semaphores in L1_SMALL."""
        src = x
        if _is_sharded(src):
            src = ttnn.to_memory_config(src, mc if not mc.is_sharded() else ttnn.DRAM_MEMORY_CONFIG)
        shape = [int(d) for d in src.shape]
        unaligned = _is_tiled(src) and ((len(shape) >= 2 and shape[-2] % TILE) or shape[-1] % TILE)
        to_bf16 = bool(unaligned) and src.dtype == ttnn.bfloat8_b
        if to_bf16:
            t = ttnn.typecast(src, ttnn.bfloat16)
            if src is not x:
                ttnn.deallocate(src)
            src = t
        parts = ttnn.all_broadcast(
            src, cluster_axis=ca, memory_config=src.memory_config(), use_l1_small_for_semaphores=True
        )
        cat_mc = mc if not mc.is_sharded() else src.memory_config()
        out = ttnn.concat(list(parts), _norm_dim(dim, len(shape)), memory_config=cat_mc)
        for p in parts:
            ttnn.deallocate(p)
        if src is not x:
            ttnn.deallocate(src)
        if to_bf16:
            t = ttnn.typecast(out, ttnn.bfloat8_b)
            ttnn.deallocate(out)
            out = t
        if mc.is_sharded():
            t = ttnn.to_memory_config(out, mc)
            ttnn.deallocate(out)
            out = t
        return out

    def reduce_scatter(
        self, x, dim: int, axis: Axis, *, memory_config=None, output_tensor=None, compute_kernel_config=None
    ):
        """Sum over ``axis`` and keep this chip's ``1/n`` slice of ``dim`` (slice index = chip index on ``axis``).

        Row-major inputs or non-tile-aligned output slices use the op's composite path. fp32 inputs accumulate in
        fp32 (the op enables fp32 dest acc for fp32 by default; pass ``compute_kernel_config`` to override). With
        L1_SMALL semaphores a small TILE payload on a ring axis takes the direct op explicitly (as ttnn would), every
        other payload ``use_l1_small_for_semaphores=True``.
        """
        ca = self.cluster_axis(axis)
        mc = self._mc(memory_config)
        n = self.axes.mesh_shape[ca]
        if n == 1:
            return self._fresh(x, mc)
        return self._reduce_scatter(x, dim, ca, n, mc, output_tensor=output_tensor, ckc=compute_kernel_config)

    @staticmethod
    def _spec_key(x) -> tuple:
        return (tuple(int(d) for d in x.shape), tuple(int(d) for d in x.padded_shape), _dtype_name(x), _is_tiled(x),
                _is_sharded(x))  # fmt: skip

    def _rs_direct(self, x, dim: int, ca: int, n: int) -> bool:
        key = ("rs_direct", ca, dim, self._spec_key(x))
        plan = self._plans.get(key)
        if plan is None:
            plan = _is_tiled(x) and not _is_sharded(x) and direct_rs_eligible(
                x.shape, x.padded_shape, _dtype_name(x), True, False, dim, n, self.is_ring_axis(x, ca)
            )
            self._plans[key] = plan
        return plan

    def _ar_plan(self, x, n: int):
        """``(scatter dim, composite?)`` of :meth:`all_reduce` for this tensor spec (cached)."""
        key = ("ar", n, self._spec_key(x))
        plan = self._plans.get(key)
        if plan is None:
            rank, tiled = len(x.shape), _is_tiled(x)
            dim = finding_scatter_dim(x.padded_shape, rank, tiled, n)
            comp_dim = 0 if dim == rank else dim
            composite = (
                dim != comp_dim or composite_ag_for_ar(x.shape, tiled, comp_dim) or composite_rs(x.shape, tiled, comp_dim, n)
            )
            plan = self._plans[key] = (dim, composite)
        return plan

    def _reduce_scatter(self, x, dim, ca, n, mc, *, output_tensor=None, ckc=None):
        kw = self._kwargs_links_topology()
        if output_tensor is not None:
            kw["output_tensor"] = output_tensor
        if ckc is not None:
            kw["compute_kernel_config"] = ckc
        if self.l1_small_semaphores:
            if output_tensor is None and ckc is None and self._rs_direct(x, dim, ca, n):
                return ttnn.experimental.reduce_scatter_minimal_direct(
                    x, dim, cluster_axis=ca, num_links=self.num_links, memory_config=mc
                )
            kw["use_l1_small_for_semaphores"] = True
        return ttnn.reduce_scatter(x, dim=dim, cluster_axis=ca, memory_config=mc, **kw)

    def all_reduce(self, x, axis: Axis, *, memory_config=None):
        """Sum over ``axis``; every chip of the axis gets the identical result (RS+AG or AG+local-sum).
        bf16 / bfp8 inputs reduce in bf16; fp32 inputs take the fp32 path. With L1_SMALL semaphores this is a
        Python mirror of ``ttnn.all_reduce`` (same scatter dim, same RS+AG / AG+local-sum choice, bitwise-identical
        results) whose collectives keep their semaphores in L1_SMALL (``ttnn.all_reduce`` has no such argument)."""
        ca = self.cluster_axis(axis)
        mc = self._mc(memory_config)
        n = self.axes.mesh_shape[ca]
        if n == 1:
            return self._fresh(x, mc)
        if not self.l1_small_semaphores:
            return ttnn.all_reduce(x, cluster_axis=ca, memory_config=mc, **self._kwargs_links_topology())
        rank = len(x.shape)
        if rank < 2:
            shape = [int(d) for d in x.shape]
            x2 = ttnn.reshape(x, [1] + shape)
            out = self.all_reduce(x2, ca, memory_config=mc)
            res = ttnn.reshape(out, shape)
            return res
        sharded = _is_sharded(x)
        work = x  # all_reduce_async: a sharded input is made interleaved first (its buffer type kept)
        if sharded:
            work = ttnn.sharded_to_interleaved(
                x, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, x.memory_config().buffer_type)
            )
        dim, composite = self._ar_plan(x, n)
        if composite:
            out = self._ar_gather_sum(work, ca, n, mc)
        else:  # RS + AG, both into the output memory config (as all_reduce_async does)
            rs = self._reduce_scatter(work, dim, ca, n, mc)
            out = ttnn.all_gather(rs, dim=dim, cluster_axis=ca, memory_config=mc)
            ttnn.deallocate(rs)
        if work is not x:
            ttnn.deallocate(work)
        if sharded and out.memory_config() != mc:
            t = ttnn.to_memory_config(out, mc)
            ttnn.deallocate(out)
            out = t
        return out

    def _ar_gather_sum(self, work, ca: int, n: int, out_mc):
        """``all_reduce_async``'s AG + local-sum branch: ``[1, *shape]`` -> composite all-gather on dim 0 (Linear
        all_broadcast, L1_SMALL semaphores) -> ``moreh_sum`` (bf16 / bfp8) or reshape-transpose-``sum`` (fp32) over
        the device dim -> ``shape``."""
        shape = [int(d) for d in work.shape]
        r = ttnn.reshape(work, [1] + shape)
        kw = {"topology": ttnn.Topology.Linear}
        if self.num_links is not None:
            kw["num_links"] = self.num_links
        unaligned = _is_tiled(r) and ((shape[-2] % TILE if len(shape) >= 2 else 0) or shape[-1] % TILE)
        to_bf16 = bool(unaligned) and r.dtype == ttnn.bfloat8_b
        src = ttnn.typecast(r, ttnn.bfloat16) if to_bf16 else r
        parts = ttnn.all_broadcast(src, cluster_axis=ca, memory_config=src.memory_config(), use_l1_small_for_semaphores=True, **kw)
        g = ttnn.concat(list(parts), 0)
        for p in parts:
            ttnn.deallocate(p)
        if src is not r:
            ttnn.deallocate(src)
        if to_bf16:
            t = ttnn.typecast(g, ttnn.bfloat8_b)
            ttnn.deallocate(g)
            g = t
        rm = g.layout == ttnn.ROW_MAJOR_LAYOUT
        gt = ttnn.to_layout(g, ttnn.TILE_LAYOUT) if rm else g
        if work.dtype == ttnn.float32:  # local_sum_float32: [n, 1, *shape] -> transpose(0, 1) -> sum(dim=1)
            gs = [int(d) for d in gt.shape]
            rr = ttnn.reshape(gt, [n, gs[0] // n] + gs[1:])
            tr = ttnn.transpose(rr, 0, 1)
            s = ttnn.sum(tr, 1, keepdim=False, memory_config=out_mc)
            ttnn.deallocate(tr)
        else:  # local_sum: moreh_sum over the device dim (bfp8 summed in bf16)
            src2 = ttnn.typecast(gt, ttnn.bfloat16) if gt.dtype == ttnn.bfloat8_b else gt
            s = ttnn.moreh_sum(src2, dim=0, keepdim=True, memory_config=out_mc)
            if src2 is not gt:
                ttnn.deallocate(src2)
            if gt.dtype == ttnn.bfloat8_b:
                t = ttnn.typecast(s, ttnn.bfloat8_b)
                ttnn.deallocate(s)
                s = t
        if gt is not g:
            ttnn.deallocate(gt)
        ttnn.deallocate(g)
        if rm:
            t = ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(s)
            s = t
        return ttnn.reshape(s, shape)

    def ar_exact(self, x, axis: Axis, *, memory_config=None, compute_kernel_config=None):
        """Exact small-payload all-reduce over ``axis`` (MLP request; ``tt/polynorm.py:_ar_ag_sum``): all-gather the
        shards, then one accurate fp32-accumulated ``ttnn.sum`` (role ``ccl_reduce``: HiFi4, fp32 dest acc), so fp32
        inputs are summed in fp32 for any row count (the RS+AG all-reduce adds fp32 at TF32 class above 32 rows) in
        2-3 ops instead of the composite all-reduce's ~7. Replicas are bitwise identical.

        * ``x [..., T, 1]`` (one value per row, e.g. PolyNorm moment sums): zero-pad the width to one tile ->
          native ``all_gather(dim=-1)`` -> ``sum(dim=-1, keepdim=True)`` -> ``[..., T, 1]`` (= ``_ar_ag_sum``).
        * any other width, rank >= 3 with a size-1 dim ``-3``: native ``all_gather(dim=-3)`` -> ``sum(dim=-3,
          keepdim=True)``.
        Intended for payloads of a few tiles (decode statistics); use :meth:`all_reduce` for activations."""
        ca = self.cluster_axis(axis)
        mc = self._mc(memory_config)
        n = self.axes.mesh_shape[ca]
        ckc = compute_kernel_config if compute_kernel_config is not None else self._reduce_ckc
        if n == 1:
            return self._fresh(x, mc)
        shape = [int(d) for d in x.shape]
        rank = len(shape)
        if x.layout != ttnn.TILE_LAYOUT:
            raise ValueError("ar_exact expects a TILE tensor")
        if shape[-1] == 1:
            pad = [(0, 0)] * (rank - 1) + [(0, TILE - 1)]
            # Within the tile padding ttnn.pad is an in-place zero fill of x's padding columns + a VIEW of x (README
            # gotchas): xp shares x's buffer, so it is never deallocated here (that would free the caller's tensor).
            xp = ttnn.pad(x, pad, 0.0, memory_config=mc)
            g = self.all_gather(xp, rank - 1, ca, memory_config=mc)
            if xp.buffer_address() != x.buffer_address():
                ttnn.deallocate(xp)
            out = ttnn.sum(g, dim=-1, keepdim=True, memory_config=mc, compute_kernel_config=ckc)
            ttnn.deallocate(g)
            return out
        if rank < 3 or shape[-3] != 1:
            raise ValueError(f"ar_exact needs [..., T, 1] or a size-1 dim -3, got {shape}")
        g = self.all_gather(x, rank - 3, ca, memory_config=mc)
        out = ttnn.sum(g, dim=rank - 3, keepdim=True, memory_config=mc, compute_kernel_config=ckc)
        ttnn.deallocate(g)
        return out

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
        mc = self._mc(memory_config)
        if n == 1:
            return self._fresh(x, mc)
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
        """All-reduce across the TP chips of a row (``wo``, MoE output, dense MLP)."""
        return self.all_reduce(x, "tp", **kw)

    def ar_dp(self, x, **kw):
        """All-reduce across DP groups (decode MoE combine)."""
        return self.all_reduce(x, "dp", **kw)

    def ag_dp(self, x, dim: int = 2, **kw):
        """All-gather across DP groups (prefill MoE output; the decode token gather uses :meth:`ag_dp_rows`)."""
        return self.all_gather(x, dim, "dp", **kw)

    def ag_dp_rows(self, x, *, memory_config=None, intermediate_memory_config=None, out_layout=ttnn.TILE_LAYOUT):
        """Decode MoE token gather (INFRA-5, MOE-1): this row's lanes ``[1, 1, L, 4096]`` (L = 8) ->
        ``[1, 1, 4 L, 4096]`` in lane order ``8 dp + l`` on every chip.

        ``to_layout(ROW_MAJOR)`` -> ``all_gather(dim=2, "dp")`` in ROW_MAJOR -> ``to_layout(out_layout)`` (TILE by
        default). Gate G4: the 8-row gather costs 12 us traced in ROW_MAJOR against 62 us in TILE (whose padded tiles
        force the composite path). Measured on this Galaxy (``test_infra_device.py``, TORUS_Y, traced, three runs):
        26.5-28 us in all with the default L1 intermediates (RM all_gather 11-12 + tilize L1->DRAM 11 + untilize 2.5),
        30-31 us with DRAM intermediates, vs 59-62 us for the TILE gather.

        The untilized input and the gathered ROW_MAJOR tensor live in ``intermediate_memory_config`` (default L1
        interleaved: 64 KB / 256 KB per chip) and are freed before returning, so no L1 buffer outlives the call. A
        ROW_MAJOR input skips the untilize; the input is not consumed. Output in ``memory_config`` (DRAM interleaved
        by default, the module-boundary convention). Replicas of the result are bitwise identical. A size-1 DP axis
        returns a new tensor (a copy, or the layout conversion)."""
        mc = self._mc(memory_config)
        imc = intermediate_memory_config if intermediate_memory_config is not None else ttnn.L1_MEMORY_CONFIG
        if self.axis_size("dp") == 1:
            return self._fresh(x, mc) if x.layout == out_layout else ttnn.to_layout(x, out_layout, memory_config=mc)
        rm = x if x.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=imc)
        g = self.all_gather(rm, 2, "dp", memory_config=imc)
        if rm is not x:
            ttnn.deallocate(rm)
        if out_layout == ttnn.ROW_MAJOR_LAYOUT:
            if imc == mc:
                return g
            out = ttnn.to_memory_config(g, mc)
            ttnn.deallocate(g)
            return out
        out = ttnn.to_layout(g, out_layout, memory_config=mc)
        ttnn.deallocate(g)
        return out

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


def l1_usage(mesh_device) -> Dict[str, int]:
    """Allocated bytes per bank of main L1 and L1_SMALL (``ttnn.get_memory_view``; allocator bookkeeping, no device
    sync). A CCL program-cache miss that leaves main L1 unchanged (with its outputs freed) placed no semaphore there:
    ``tests/unit/test_infra_l1_small.py`` uses it to assert the L1_SMALL routing."""
    out = {}
    for name, bt in (("l1", ttnn.BufferType.L1), ("l1_small", ttnn.BufferType.L1_SMALL)):
        try:
            out[name] = int(ttnn.get_memory_view(mesh_device, bt).total_bytes_allocated_per_bank)
        except Exception:  # pragma: no cover - no device
            out[name] = -1
    return out


# ------------------------------------------------------------------------------------------------------------
# Fabric topology report (INFRA-6; WAVE_A_REVIEW §0.3 / D3)
# ------------------------------------------------------------------------------------------------------------
# Ring (wrap) flags per dim of the active mesh graph's device topology (whose dim sizes get_physical_mesh_shapes()
# reports, (8, 4) on this Galaxy) for each fabric type get_all_mgd_fabric_types() can return: the MGD names dim 0 "Y"
# and dim 1 "X" (tt_metal/fabric/mesh_graph_descriptor.cpp infer_declared_fabric_type_from_dim_types). So TORUS_Y on
# the (8, 4) graph rings the size-8 (TP) axis and leaves the size-4 (DP) axis a line (WAVE_A_REVIEW §0.3).
_RING_DIMS = {"MESH": (False, False), "TORUS_Y": (True, False), "TORUS_X": (False, True), "TORUS_XY": (True, True)}


def fabric_report(mesh_device=None) -> Dict[str, Any]:
    """What fabric the opened mesh really runs on.

    * ``requested``: ``ttnn.get_fabric_config()`` (what the mesh was opened with, e.g. ``FABRIC_2D_TORUS_XY``);
    * ``committed``: the fabric type of the active mesh graph (``ttnn.get_all_mgd_fabric_types()``), i.e. what the
      topology mapper realized (``TORUS_XY`` on a healthy torus; ``TORUS_Y`` when the X wrap links are down);
    * ``physical_shape``: the physical mesh grid (``ttnn.get_physical_mesh_shapes()``);
    * ``tp_ring`` / ``dp_ring``: whether the 8-chip TP axis / 4-chip DP axis is a ring (else a line), from the
      committed type and the grid sizes;
    * ``degraded``: a 2D torus was requested but not fully committed (DP-axis CCLs then run on a line);
    * ``l1_small``: L1_SMALL bytes per core of the mesh (0 = none; the CCL semaphores then go to main L1).

    Only call it with a mesh open (``mesh_device`` given; the queries read the control plane). Missing APIs yield
    ``None`` fields instead of raising."""
    rep: Dict[str, Any] = {
        "requested": active_fabric_name(mesh_device) if mesh_device is not None else None,
        "committed": None,
        "physical_shape": None,
        "tp_ring": None,
        "dp_ring": None,
        "degraded": None,
        "mesh_shape": tuple(int(s) for s in mesh_device.shape) if mesh_device is not None else None,
        "l1_small": mesh_l1_small_bytes(mesh_device) if mesh_device is not None else None,
    }
    if mesh_device is None:
        return rep
    try:
        types = [getattr(t, "name", str(t).split(".")[-1]) for t in ttnn.get_all_mgd_fabric_types()]
        rep["committed"] = types[0] if len(types) == 1 else ",".join(types)
    except Exception as e:  # pragma: no cover - older ttnn
        rep["committed_error"] = f"{type(e).__name__}: {e}"
    try:
        shapes = ttnn.get_physical_mesh_shapes()
        if shapes:
            rep["physical_shape"] = tuple(int(s) for s in next(iter(shapes.values())))
    except Exception as e:  # pragma: no cover
        rep["physical_shape_error"] = f"{type(e).__name__}: {e}"
    ring = _RING_DIMS.get(rep["committed"])
    shape = rep["physical_shape"]
    if ring is not None and shape is not None and len(shape) == 2 and shape[0] != shape[1]:
        axes = MeshAxes.detect(rep["mesh_shape"])
        ring_sizes = {shape[i] for i in (0, 1) if ring[i]}
        rep["tp_ring"] = axes.tp_size in ring_sizes
        rep["dp_ring"] = axes.dp_size in ring_sizes
    req = rep["requested"] or ""
    if rep["committed"] is not None and "TORUS_XY" in req:
        rep["degraded"] = rep["committed"] != "TORUS_XY"
    return rep


def log_fabric(mesh_device, tag: str = "", printer=print) -> Dict[str, Any]:
    """Print (and return) :func:`fabric_report` as one greppable line, e.g.
    ``[motif3.fabric] committed: TORUS_Y requested: FABRIC_2D_TORUS_XY tp_ring=True dp_ring=False l1_small=32768
    DEGRADED ...``. Call it at the start of every device test (WAVE_A_REVIEW §5: device logs must show the committed
    topology)."""
    rep = fabric_report(mesh_device)
    note = ""
    if rep.get("degraded"):
        note = (
            " DEGRADED (2D torus requested, not committed: DP-axis CCLs run on a line; re-measure on a healthy torus)"
        )
    if rep.get("l1_small") == 0:
        note += " NO-L1_SMALL (CCL semaphores in main L1: open with l1_small_size, generator_api.L1_SMALL_SIZE)"
    printer(
        f"[motif3.fabric]{(' ' + tag) if tag else ''} committed: {rep['committed']} requested: {rep['requested']} "
        f"physical={rep['physical_shape']} mesh={rep['mesh_shape']} tp_ring={rep['tp_ring']} "
        f"dp_ring={rep['dp_ring']} l1_small={rep['l1_small']}{note}"
    )
    return rep


__all__ = [
    "DIRECT_RS_MAX_INPUT_BYTES",
    "MotifCCL",
    "composite_ag",
    "composite_ag_for_ar",
    "composite_rs",
    "device_tensors_to_torch",
    "direct_rs_eligible",
    "fabric_report",
    "finding_scatter_dim",
    "l1_usage",
    "log_fabric",
    "replicas_identical",
    "resolve_axis",
    "topology_for_fabric",
]
