# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``MotifTTConfig``: every static decision the Motif-3 TT modules share.

Built from the HF ``config.json`` (or a transformers config object, or ``GeneratorSettings``) plus an opened mesh
(or just a mesh shape for host-only use). It carries

* the mesh axis roles (design §3.1): logical mesh (4, 8) = 4 DP groups ("rows", 8 user lanes each) x TP8
  ("cols"). The TP axis is *detected* as the mesh dim of size 8, so an (8, 4) mesh (the plugin's ``BH-Galaxy``
  preset) works too. Modules never hard-code ``cluster_axis`` 0/1; they use ``cfg.axes.tp_axis`` /
  ``cfg.axes.dp_axis`` (or the role names in ``tt/ccl.py``);
* per-chip head counts (design §2.3.4): 10 q heads, 2 KV groups, 8 signal heads per TP chip;
* the dtype policy per weight class (design §1.5): bfp8 routed/shared/dense experts, bf16 attention, router,
  mHC, LM head and embedding, bfp8 latent KV cache, fp32 math knobs;
* the layer schedule (design §2.3.1; study 01 §2.1): dense 0-1 / MoE 2-52, global ``l % 4 == 0`` (14 layers,
  YaRN RoPE, scale 0.14467963) / SWA (39 layers, plain RoPE, scale 0.07216878, window 129 keys incl. current);
* the KV pool (design §1.5, §3.5, §5.1): block 64, 262,144 pool tokens. The block count is what
  ``allocate_kv_cache`` receives (``set_kv_geometry``); before that ``kv_num_blocks`` is the bridge / plugin
  formula (pool + 32-token null-block reserve + ``block * max_num_seqs``) = 4129 for the default flags;
* prefill buckets (powers of 2, 128 ... 32768; design §2.3.10), the compute-kernel roles and program configs the
  device gates validated (``tests/unit/gates/GATES_RESULTS.md``) plus the module builders measured in wave B1
  (attention / MLP / mHC / router / prefill experts / decode norm / LM head, each equal to the module's local config;
  ``compute_config_descriptor`` for ``ttnn.generic_op`` kernels), the fabric the mesh was opened with, the trace
  region size and the TT weight-cache layout ``<TT_CACHE_PATH>/<version-tag>/mesh<R>x<C>/{L<nn>|global}/<name>``
  (design §2.3.11);
* the device memory the model needs: an L1_SMALL region of :data:`DEFAULT_L1_SMALL_SIZE` (32768) bytes per core for
  the CCL semaphores (attention wave-B1 P0; ``tt/ccl.py``). :func:`device_params` (pytest), :func:`open_motif_mesh`
  (standalone) and the vLLM ``"tt"`` config (``generator_api.SERVING_TT_CONFIG``) all open the mesh with it;
  :func:`require_l1_small` checks an opened mesh;
* module defaults the decoder passes (``mhc_sinkhorn``, ``router_logits``) and the PolyNorm config semantics
  (``polynorm_sigmoid_weight``, ``polynorm_output_scale_per_layer``);
* the features of ``docs/features/FEATURES_DESIGN.md`` (chunked prefill, prefix caching, MTP speculation): the span
  cap and its buckets (``prefill_span_cap``, ``max_prefill_span``, ``prefill_span_buckets``), the resume alignment
  ``A`` (``prefill_resume_alignment``) and the sp1 program configs it comes from (``resumed_prefill_pc``,
  :data:`SP1_GLOBAL_CHUNKS`), the sp1 SDPA page-table width, the chunk cost table, ``plan_prefill_row`` with this
  config's geometry, the KV-R / decode KV-write mode (``kv_replicated_decode``, ``kv_write_mode``) and the MTP layer
  (``spec_tokens``, ``mtp_layer_spec()``, ``kv_pool_layers``);
* packed prefill (P5) and the T64 verify modes (docs/p5_t64/P5_T64_DESIGN.md §3, §4, §8.3): the packed segment sizes and
  pass cap (``pack_seg_buckets``, ``pack_sp1_seg_buckets``, ``pack_max_tokens``) and every packed shape the generator
  warms (``packed_prefill_shapes()``); ``spec_verify``, ``wide_rows_per_dp`` and ``wide_step_ratio``; the per-M
  (``m_tiles``) decode matmul configs of the 64-row step (``mcast1d_matmul_pc(per_core_m=)``, MoE router and experts,
  LM head); and the refusals ``validate`` enforces for it (F3N rule R1: ``ring_gather="safe"``; review edit R-E7).

Import rule (design §2.1): this module imports only the standard library, ``ttnn``, ``generator_api`` (stdlib +
torch; the KV-pool and weights-location helpers shared with the vLLM bridge) and ``prefill_plan`` (stdlib + torch +
``generator_api``: the prefill planner and its defaults). It never opens a device and never imports other
``models/demos/**`` packages.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import ttnn

from .generator_api import DEFAULT_KV_POOL_TOKENS as _API_DEFAULT_KV_POOL_TOKENS
from .generator_api import L1_SMALL_SIZE as _API_L1_SMALL_SIZE
from .generator_api import NUM_LANES, SUPPORTED_BLOCK_SIZES, cdiv, check_block_size, expected_num_blocks
from .generator_api import hf_cache_snapshot as _hf_cache_snapshot
from .generator_api import kv_cache_bytes_per_chip as _api_kv_cache_bytes_per_chip
from .generator_api import kv_pool_tokens_from_env, resolve_tt_cache_path, resolve_weights_location
from .generator_api import DEFAULT_PREFILL_SPAN_CAP, SUPPORTED_SPEC_TOKENS, check_prefill_span_cap, kv_write_mode
from .generator_api import prefill_span_cap_from_env
from .generator_api import DEFAULT_PACKED_PREFILL_MAX_SEG, DEFAULT_PACKED_PREFILL_MAX_TOKENS, PACK_BATCHES
from .generator_api import PACK_SEG_BUCKETS, PACK_SP1_SEG_BUCKETS, PACKED_PASS_KINDS, PK1_TAIL_VARIANTS
from .generator_api import SPEC_VERIFY_MODES, WIDE_SPEC_VERIFY_MODES, check_packed_prefill_max_seg
from .generator_api import check_packed_prefill_max_tokens
from .generator_api import packed_prefill_max_seg_from_env, packed_prefill_max_tokens_from_env
from .generator_api import packed_prefill_pk1_from_env
from .generator_api import HOST_STAGING_MODES, HOST_WAIT_MODES, check_capture_thread, check_prefill_trace
from . import prefill_plan as _plan

# ------------------------------------------------------------------------------------------------------------
# Paths and versioning
# ------------------------------------------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/ttuser/hchang/experiments/motif-3")
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights" / "Motif-3"
DEFAULT_HF_META_DIR = PROJECT_ROOT / "hf_meta"
DEFAULT_TT_CACHE_ROOT = PROJECT_ROOT / "tt_cache"
DEFAULT_WEIGHTS_REVISION = "2ed2ed5cfabffa10fdabb2fc0d0288f8e6de893a"  # pinned HF revision (design §5.3)

# Bump whenever a transform in tt/weights.py (or a mapper/layout convention) changes: it is part of the cache
# version tag, so stale .tensorbin files are never loaded silently.
CACHE_FORMAT_VERSION = 1

# Default device knobs (design §1.5, §5.1).
DEFAULT_TRACE_REGION_SIZE = 256 * 1024 * 1024  # 268435456; raise to 512 MiB on overflow
DEFAULT_KV_BLOCK_SIZE = 64
DEFAULT_KV_POOL_TOKENS = _API_DEFAULT_KV_POOL_TOKENS  # 262144 (= TIS max_tokens_all_users_override)
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_MAX_BATCH = NUM_LANES  # 32 decode lanes: the trace always runs 32, whatever --max-num-seqs is
DEFAULT_MIN_PREFILL_BUCKET = 128
DEFAULT_FABRIC = "FABRIC_2D_TORUS_XY"  # fallback: FABRIC_1D_RING (design §1.3, kill criterion G4)
# L1_SMALL bytes per core every Motif-3 mesh is opened with (generator_api.L1_SMALL_SIZE, attention P0): the CCL
# global semaphores live there (tt/ccl.py); without it they fragment main L1 and later static CBs clash with them.
DEFAULT_L1_SMALL_SIZE = _API_L1_SMALL_SIZE  # 32768; env MOTIF3_L1_SMALL_SIZE (experiments only)
DEFAULT_DISPATCH_CORE_AXIS = "col"  # BH: COL dispatch unless a fabric tensix config is set (root conftest, plugin)

TILE = 32

# KV cache dtype names used by the bridge (GeneratorSettings.kv_cache_dtype / MOTIF3_KV_CACHE_DTYPE).
KV_CACHE_DTYPE_BY_NAME = {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
MHC_SINKHORN_IMPLS = ("motif", "stock")  # MHCSite(sinkhorn=...)
ROUTER_LOGITS_IMPLS = ("composite", "exact_fp32")  # MotifMoE(router_logits=...)
# Decode routing-weight path (A5, docs/OPTIMIZATION_PLAN.md §3.3; MotifMoE(router_mask=...)): "gather" (the release:
# ttnn.gather of the top-8 scores + idx-based local mask) | "scatter" (MotifRouter.route_local: top-8 0/1 mask scattered
# from topk's idx, local one-hot extraction; same top-8 sets, weights within 1-4 fp32 ulp, -1.7 to -2.0 ms per step) |
# "fused" (B4: MotifRouter.route_fused, the router matmul + one generic_op kernels/router_topk.py for bias, top-8,
# normalize and local extraction; same sets except on exact fp32 ties at the 8th value, weights within fp32 rounding;
# the default since the Phase B eval passed, docs/OPT_PHASE_B_EVAL.md).
ROUTER_MASK_MODES = ("gather", "scatter", "fused")
# Decode routed experts (B1, docs/OPTIMIZATION_PLAN.md §3.3; MotifMoE(decode_experts=...)): "dense" (the release: every
# chip runs its 12 local experts on all M gathered rows) | "sparse" (ttnn.sparse_matmul skips the local experts no live
# row routes to; sparsity read on device, nnz=None; inactive lanes masked out of the routing weights). Live rows are
# bitwise equal to "dense" (logs/opt/phaseA/M6); decode only, prefill keeps the masked-dense path.
DECODE_EXPERTS_MODES = ("dense", "sparse")
# Decode routed-expert PolyNorm (B3, docs/OPTIMIZATION_PLAN.md §3.3; MotifMoE(moe_polynorm=...)): "composite" (the
# release: tt/polynorm.py's grouped Horner fp32, ~17 ops) | "fused" (tt/kernels/moe_polynorm.py: one generic_op over the
# gate_up output; fp32 moments + Horner + routing weight, one bf16 rounding; ~5 ms less per T32 step, not bitwise equal
# to the composite: 1-ulp bf16 differences in ~2e-5 of the values, logs/opt/phaseA/M10; the default since the Phase B
# eval passed, docs/OPT_PHASE_B_EVAL.md). Decode only (M = 32 / 64).
MOE_POLYNORM_MODES = ("composite", "fused")
# Decode shared-expert PolyNorm (B5, docs/OPTIMIZATION_PLAN.md §3.3; PolyNormMLP(shared_polynorm=...)): "composite" (the
# release: tt/polynorm.py polynorm_tp, 19 small programs incl. the moments all-gather) | "fused"
# (tt/kernels/shared_polynorm.py: a one-core moments kernel, the same TP all-gather, a one-core apply kernel; the same
# LLK operations in the same order, so bitwise equal to the composite by construction; the default since the B5 gates
# passed, logs/opt/phaseB/B5). Decode only (one 32-row tile).
SHARED_POLYNORM_MODES = ("composite", "fused")
# Prefill routed experts (B2a, docs/OPTIMIZATION_PLAN.md §3.3 B2; MotifMoE(prefill_moe=...)): "dense" (the release: every
# chip runs its 12 local experts on all rows of the chunk, masked by the routing weights) | "compact" (the default since
# the B2a gates passed; token-compacted:
# the host reads the chunk's routes from chip 0 and builds each chip's expert-sorted row lists; the chip gathers only
# its routed rows, runs them through ttnn.sparse_matmul in blocks of PREFILL_MOE_BLOCKS rows and combines them with a
# one-hot matmul; bitwise equal to "dense", logs/opt/phaseA/m7 and logs/opt/phaseB/B2a). Decode is not affected.
PREFILL_MOE_MODES = ("dense", "compact")
# Rows per expert block of the compacted prefill experts (MOTIF3_PREFILL_MOE_BLOCK): "auto" (moe.compact_block: 32 up
# to 2048-row chunks, 64 above) or a fixed 32 / 64 / 128.
PREFILL_MOE_BLOCKS = ("auto", "32", "64", "128")
# Chunks with fewer rows keep the dense prefill MoE (MOTIF3_PREFILL_MOE_MIN_ROWS; a multiple of 32, >= 32).
DEFAULT_PREFILL_MOE_MIN_ROWS = 1024
# Who builds the compacted rows (B2b, MOTIF3_PREFILL_MOE_DISPATCH): "host" (B2a: a blocking read of the routes, numpy
# row lists, one upload) | "device" (tt/kernels/moe_compact.py CompactDispatch: one generic_op builds the same lists on
# device from the routes; the host reads only the 32-byte block count). Bitwise the same rows either way.
PREFILL_MOE_DISPATCH_MODES = ("host", "device")
# How the compacted rows are summed back per token (B2b, MOTIF3_PREFILL_MOE_COMBINE): "matmul" (B2a: one-hot P^T @ y)
# | "gather" (tt/kernels/moe_compact.py GatherCombine: each token's rows gathered and added in an fp32 dest with
# fast_reduce_nc's ops). Bitwise equal to each other and to the dense path.
PREFILL_MOE_COMBINE_MODES = ("matmul", "gather")
# MotifCCL(ring_gather=...): "safe" (DEFAULT, lead decision 2026-10-03: the native single-page decode gathers still race
# ~1 event per 1e4 decode steps -- silent stale tiles, docs/determinism/FIX.md -- and +0.26-0.45 ms per decode step is
# cheap; T64 also requires it) reroutes every race-prone gather. "lean" routes every all-gather that ttnn would run on its multicast factory
# with even-ring alternate routes (a completion race: docs/determinism/INVESTIGATION.md, tt/ccl.py) AND that streams
# more than one circular-buffer page per link (where the race is frequent) through all_broadcast + concat, plus every
# such gather of a race_free=True call (the MoE prefill combine) or of a MotifCCL.race_free_scope(); the other
# single-page ones (every decode gather, the bucket-128 / 256 PolyNorm moments) stay native under "lean". "safe"
# reroutes every such gather (+0.26-0.45 ms per decode step); "native" = the plain ttnn.all_gather (the pre-fix
# behaviour, prefill not run-to-run reproducible). Validation and cost: docs/determinism/FIX.md.
RING_GATHER_MODES = ("safe", "lean", "native")
# T64 step / T32-spec step device-time ratio r (docs/p5_t64/P5_T64_DESIGN.md §4.9, option A'': 1.118 at 1K, 1.126 at
# 4K, 1.173 at 32K context): the input of the T64 drafting crossover c* (verify_plan.crossover_lanes). Gate G16
# re-measures it on the full model.
DEFAULT_WIDE_STEP_RATIO = 1.13
# Gathered decode row counts at which tt/moe.py's MotifRouter runs the exact-fp32 router kernel (RouterLogitsFP32) when
# router_logits="exact_fp32" (MotifRouter._use_logits_fn; moe.EXACT_ROUTER_DECODE_ROWS must list the same counts,
# test_infra_config checks it): 32 (T32 steps) and the 64 rows of a T64 step (review edit R-E7: WP-D D1, rows bitwise
# equal to M = 32 in test_moe_device_t64_rows[exact_fp32]; G-S5w passed under exact_fp32). validate() refuses
# spec_verify="auto" with router_logits="exact_fp32" at a T64 row count missing here: those rows would take the
# composite router, differ from the T32 rows, and "auto" (which switches between the two traces per step) would not be
# lossless.
ROUTER_EXACT_FP32_DECODE_ROWS: Tuple[int, ...] = (TILE, 2 * TILE)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else int(default)


def resolve_weights_dir() -> Path:
    """Checkpoint directory. One precedence order with the vLLM bridge (``generator_api.resolve_weights_location``):
    ``MOTIF3_WEIGHTS_DIR`` > ``HF_MODEL`` (a directory) > the HF-cache snapshot of a repo-id ``HF_MODEL`` at
    ``TT_MODEL_WEIGHTS_REVISION`` > the local snapshot ``DEFAULT_WEIGHTS_DIR``. (An uncached repo id falls back to the
    local snapshot here; the generator downloads it instead.) A set but missing ``MOTIF3_WEIGHTS_DIR`` raises."""
    loc = resolve_weights_location(None, os.environ)
    return Path(loc.path) if loc.is_local else DEFAULT_WEIGHTS_DIR


def resolve_hf_config_path(path: Optional[Union[str, os.PathLike]] = None) -> Path:
    """``config.json`` to read: ``path`` (file or dir), else the weights dir, else ``hf_meta``."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates += [resolve_weights_dir(), DEFAULT_HF_META_DIR]
    for c in candidates:
        f = c / "config.json" if c.is_dir() else c
        if f.is_file():
            return f
    raise FileNotFoundError(f"no Motif-3 config.json found in {candidates}")


def resolve_tt_cache_root() -> Path:
    """TT weight-cache root: ``MOTIF3_TT_CACHE_PATH`` > ``TT_CACHE_PATH`` (TIS / tt-model contract) >
    ``motif-3/tt_cache`` (``generator_api.resolve_tt_cache_path``, the bridge's order)."""
    v = resolve_tt_cache_path()
    return Path(v) if v else DEFAULT_TT_CACHE_ROOT


def fabric_config_from_name(name: Optional[str] = None):
    """``ttnn.FabricConfig`` by name (default ``MOTIF3_FABRIC`` env, else FABRIC_2D_TORUS_XY)."""
    name = name or os.environ.get("MOTIF3_FABRIC") or DEFAULT_FABRIC
    fc = ttnn.FabricConfig.__members__.get(name)
    if fc is None:
        raise ValueError(f"unknown fabric config {name!r}; expected one of {list(ttnn.FabricConfig.__members__)}")
    return fc


def active_fabric_name(mesh_device=None) -> Optional[str]:
    """Name of the fabric config the process runs with (``ttnn.get_fabric_config()``), e.g. "FABRIC_2D_TORUS_XY".

    ``ttnn.get_fabric_config`` reads the Metal context, so this only asks when ``mesh_device`` is a real, opened
    ``ttnn.MeshDevice`` (host tests pass fakes or nothing; asking then would bring up UMD). Returns None otherwise or
    on error. The fabric actually *committed* by the topology mapper (it may fall back from TORUS_XY to TORUS_Y)
    is reported by ``tt/ccl.py:fabric_report``."""
    mesh_cls = getattr(ttnn, "MeshDevice", None)
    if mesh_device is None or mesh_cls is None or not isinstance(mesh_device, mesh_cls):
        return None
    try:
        fc = ttnn.get_fabric_config()
    except Exception:  # pragma: no cover - older ttnn
        return None
    return getattr(fc, "name", None) or str(fc).split(".")[-1]


def device_params(
    fabric: Optional[str] = None,
    trace_region_size: Optional[int] = None,
    l1_small_size: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """``device_params`` for tt-metal's ``mesh_device`` pytest fixture (root ``conftest.py``)::

        @pytest.mark.parametrize("mesh_device, device_params", [((4, 8), device_params())], indirect=True)

    Defaults: ``FABRIC_2D_TORUS_XY`` (``MOTIF3_FABRIC``), ``trace_region_size`` 256 MiB
    (``MOTIF3_TRACE_REGION_SIZE``) and ``l1_small_size`` :data:`DEFAULT_L1_SMALL_SIZE` = 32768
    (``MOTIF3_L1_SMALL_SIZE``; required by the model: the CCL semaphores live in L1_SMALL, ``tt/ccl.py``).
    ``l1_small_size=0`` opens without the region (only to reproduce the hazard). Other keys understood by the
    fixture: ``worker_l1_size``, ``num_command_queues``, ``dispatch_core_axis`` (BH forces COL unless a fabric tensix
    config is also given), ``reliability_mode``, ``fabric_tensix_config``, ``fabric_router_config``.
    """
    p = {
        "fabric_config": fabric_config_from_name(fabric),
        "trace_region_size": int(
            trace_region_size
            if trace_region_size is not None
            else _env_int("MOTIF3_TRACE_REGION_SIZE", DEFAULT_TRACE_REGION_SIZE)
        ),
        "l1_small_size": int(
            l1_small_size if l1_small_size is not None else _env_int("MOTIF3_L1_SMALL_SIZE", DEFAULT_L1_SMALL_SIZE)
        ),
    }
    if p["l1_small_size"] < 0:
        raise ValueError(f"l1_small_size must be >= 0, got {p['l1_small_size']}")
    p.update(extra)
    return p


def mesh_l1_small_bytes(mesh_device) -> Optional[int]:
    """L1_SMALL bytes per core of an opened ``ttnn.MeshDevice`` (0 = opened without ``l1_small_size``), or ``None``
    for anything else (host fakes; the query needs a real mesh). Allocator bookkeeping only, no device traffic."""
    mesh_cls = getattr(ttnn, "MeshDevice", None)
    if mesh_device is None or mesh_cls is None or not isinstance(mesh_device, mesh_cls):
        return None
    try:
        return int(ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_per_bank)
    except Exception:  # pragma: no cover - older ttnn
        return None


def require_l1_small(mesh_device, minimum: int = DEFAULT_L1_SMALL_SIZE) -> Optional[int]:
    """Raise ``RuntimeError`` when a real mesh has less than ``minimum`` bytes of L1_SMALL per core (the generator's
    ``create`` check; the vLLM bridge does the same before ``create``). Returns :func:`mesh_l1_small_bytes`."""
    size = mesh_l1_small_bytes(mesh_device)
    if size is not None and size < int(minimum):
        raise RuntimeError(
            f"Motif-3 needs a mesh opened with l1_small_size >= {minimum} (the CCL semaphores live in L1_SMALL, "
            f"tt/ccl.py); this mesh has {size} B per core. pytest: device_params(); standalone: open_motif_mesh(); "
            f'vLLM: --additional-config \'{{"tt": {{"l1_small_size": {minimum}, ...}}}}\''
        )
    return size


def mesh_shape_from_env(default: Sequence[int] = (4, 8)) -> Tuple[int, int]:
    """``MESH_DEVICE`` (``"(4, 8)"`` / ``"4,8"``) as a tuple, else ``default`` (the plugin's preset names such as
    ``"BH-Galaxy"`` map to (8, 4))."""
    raw = (os.environ.get("MESH_DEVICE") or "").strip()
    if not raw:
        return tuple(int(d) for d in default)
    if raw.upper() in ("BH-GALAXY", "GALAXY", "TG"):
        return (8, 4)
    parts = [p for p in raw.strip("()[] ").replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"MESH_DEVICE={raw!r} is not a 2D mesh shape like '(4, 8)'")
    return int(parts[0]), int(parts[1])


def open_motif_mesh(
    mesh_shape: Optional[Sequence[int]] = None,
    *,
    fabric: Optional[str] = None,
    trace_region_size: Optional[int] = None,
    l1_small_size: Optional[int] = None,
    dispatch_core_axis: str = DEFAULT_DISPATCH_CORE_AXIS,
    reliability_mode=None,
    **open_kwargs: Any,
):
    """Open the Galaxy mesh the way the model needs it outside pytest and vLLM (standalone generator / demo runs):
    ``ttnn.set_fabric_config`` (``FABRIC_2D_TORUS_XY``, strict init) + ``ttnn.open_mesh_device(MeshShape,
    dispatch_core_config=COL, trace_region_size, l1_small_size)`` with :func:`device_params` defaults -- the same
    values the vLLM plugin uses with ``generator_api.SERVING_TT_CONFIG``. ``mesh_shape`` defaults to ``MESH_DEVICE`` or
    (4, 8). Close with :func:`close_motif_mesh`. Device code: never call at import time."""
    shape = tuple(int(d) for d in (mesh_shape if mesh_shape is not None else mesh_shape_from_env()))
    p = device_params(fabric, trace_region_size, l1_small_size)
    fc = p.pop("fabric_config")
    mode = reliability_mode if reliability_mode is not None else ttnn.FabricReliabilityMode.STRICT_INIT
    ttnn.set_fabric_config(fc, mode)
    axis = ttnn.DispatchCoreAxis.COL if str(dispatch_core_axis).lower() == "col" else ttnn.DispatchCoreAxis.ROW
    p.update(open_kwargs)
    try:
        return ttnn.open_mesh_device(
            ttnn.MeshShape(*shape), dispatch_core_config=ttnn.DispatchCoreConfig(axis=axis), **p
        )
    except Exception:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise


def close_motif_mesh(mesh_device) -> None:
    """Close a mesh opened by :func:`open_motif_mesh` (submeshes first) and reset the fabric config."""
    try:
        for sub in mesh_device.get_submeshes():
            ttnn.close_mesh_device(sub)
        ttnn.close_mesh_device(mesh_device)
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def kv_cache_dtype_from_name(name: str):
    """``"bfp8"`` -> ``ttnn.bfloat8_b``, ``"bf16"`` -> ``ttnn.bfloat16`` (``GeneratorSettings.kv_cache_dtype``)."""
    key = str(name).strip().lower()
    if key not in KV_CACHE_DTYPE_BY_NAME:
        raise ValueError(f"kv cache dtype must be one of {sorted(KV_CACHE_DTYPE_BY_NAME)}, got {name!r}")
    return KV_CACHE_DTYPE_BY_NAME[key]


def kv_cache_dtype_name(dtype) -> str:
    """``"bfp8"`` | ``"bf16"`` of a latent-cache dtype given as a name or a ttnn dtype (the inverse of
    :func:`kv_cache_dtype_from_name`; a cache tensor's ``.dtype``). Raises on any other dtype."""
    if isinstance(dtype, str) and dtype.strip().lower() in KV_CACHE_DTYPE_BY_NAME:
        return dtype.strip().lower()
    for name, dt in KV_CACHE_DTYPE_BY_NAME.items():
        if dtype == dt:
            return name
    known = sorted(KV_CACHE_DTYPE_BY_NAME)
    raise ValueError(f"KV cache dtype must be one of {known} (or their ttnn dtypes), got {dtype!r}")


# ------------------------------------------------------------------------------------------------------------
# HF config parsing helpers (INFRA-1)
# ------------------------------------------------------------------------------------------------------------
def rope_scaling_of(d: Mapping[str, Any]) -> Dict[str, Any]:
    """The YaRN parameters of a config dict: ``rope_scaling`` (``config.json``, transformers < 5) or
    ``rope_parameters`` (``PretrainedConfig.to_dict()`` in transformers 5.x, which drops ``rope_scaling``; this is
    what vLLM's ``hf_config`` gives). A per-layer-type ``rope_parameters`` (``{"full_attention": {...}, ...}``) yields
    its ``full_attention`` entry (Motif's global layers carry the YaRN; SWA layers use ``swa_rope_theta``)."""
    for key in ("rope_scaling", "rope_parameters"):
        rs = d.get(key)
        if isinstance(rs, Mapping) and rs:
            if "rope_type" not in rs and "type" not in rs and any(isinstance(v, Mapping) for v in rs.values()):
                nested = rs.get("full_attention")
                if not isinstance(nested, Mapping):
                    nested = next(v for v in rs.values() if isinstance(v, Mapping))
                return dict(nested)
            return dict(rs)
    return {}


def _generation_config_near(name_or_path: Any) -> Dict[str, Any]:
    """``generation_config.json`` next to a config's ``_name_or_path`` (a snapshot dir, or a repo id cached in the
    HF cache at ``TT_MODEL_WEIGHTS_REVISION``), else ``{}``."""
    if not name_or_path:
        return {}
    p = Path(str(name_or_path)).expanduser()
    if p.is_file():
        p = p.parent
    if not p.is_dir():
        snap = None
        try:
            snap = _hf_cache_snapshot(str(name_or_path), (os.environ.get("TT_MODEL_WEIGHTS_REVISION") or None))
        except Exception:  # pragma: no cover - malformed cache
            snap = None
        if snap is None:
            return {}
        p = snap
    g = p / "generation_config.json"
    if not g.is_file():
        return {}
    try:
        return json.loads(g.read_text())
    except Exception:  # pragma: no cover - malformed file
        return {}


def _per_layer_scales(v: Any) -> Dict[int, float]:
    """``polynorm_output_scale_per_layer`` -> ``{int layer: float scale}`` (JSON object keys are strings)."""
    if not v:
        return {}
    if not isinstance(v, Mapping):
        raise ValueError(f"polynorm_output_scale_per_layer must be a mapping {{layer: scale}}, got {v!r}")
    return {int(k): float(x) for k, x in v.items()}


# ------------------------------------------------------------------------------------------------------------
# Mesh axes
# ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MeshAxes:
    """Roles of the two mesh dims.

    * ``tp_axis`` (cluster_axis): the size-8 dim. Attention heads, dense/shared intermediates and the LM-head
      vocab are split over it; ``all_reduce(tp)`` closes each TP matmul. Called "cols" in the (4, 8) view.
    * ``dp_axis``: the other dim (size 4). DP groups of ``lanes_per_row`` decode lanes; the MoE EP gather /
      reduce runs over it. Called "rows" in the (4, 8) view.

    ``coord(dp, tp)`` is the mesh coordinate of a chip; ``chip_index(dp, tp) = dp * tp_size + tp`` is the
    linear chip order used for expert placement (EP32: chip k holds experts ``[12k, 12k + 12)``).
    """

    mesh_shape: Tuple[int, int]
    tp_axis: int
    dp_axis: int

    @staticmethod
    def detect(mesh_shape: Sequence[int], tp_size: int = 8) -> "MeshAxes":
        shape = tuple(int(s) for s in mesh_shape)
        if len(shape) != 2:
            raise ValueError(f"Motif-3 needs a 2D mesh, got shape {shape}")
        if shape[1] == tp_size:
            tp_axis = 1
        elif shape[0] == tp_size:
            tp_axis = 0
        else:
            # Small test meshes (1x1, 1x2, 1x4, 2x4, ...): TP is the larger dim (ties -> cols).
            tp_axis = 1 if shape[1] >= shape[0] else 0
        return MeshAxes(shape, tp_axis, 1 - tp_axis)

    @property
    def tp_size(self) -> int:
        return self.mesh_shape[self.tp_axis]

    @property
    def dp_size(self) -> int:
        return self.mesh_shape[self.dp_axis]

    @property
    def num_chips(self) -> int:
        return self.mesh_shape[0] * self.mesh_shape[1]

    @property
    def tag(self) -> str:
        return f"mesh{self.mesh_shape[0]}x{self.mesh_shape[1]}"

    def coord(self, dp: int, tp: int) -> Tuple[int, int]:
        """Mesh coordinate (row, col) of the chip with DP index ``dp`` and TP index ``tp``."""
        rc = [0, 0]
        rc[self.dp_axis] = dp
        rc[self.tp_axis] = tp
        return rc[0], rc[1]

    def roles(self, row: int, col: int) -> Tuple[int, int]:
        """(dp, tp) of mesh coordinate (row, col)."""
        rc = (row, col)
        return rc[self.dp_axis], rc[self.tp_axis]

    def chip_index(self, dp: int, tp: int) -> int:
        """Linear chip order for expert placement: ``k = dp * tp_size + tp`` (orientation independent)."""
        return dp * self.tp_size + tp

    def mesh_dims(
        self, dp_dim: Optional[int] = None, tp_dim: Optional[int] = None
    ) -> Tuple[Optional[int], Optional[int]]:
        """Tensor dims to shard over (mesh dim 0, mesh dim 1), given the tensor dims for the DP / TP roles."""
        dims = [None, None]
        dims[self.dp_axis] = dp_dim
        dims[self.tp_axis] = tp_dim
        return dims[0], dims[1]


# ------------------------------------------------------------------------------------------------------------
# Layer schedule
# ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LayerSpec:
    """Per-layer constants (each one is a per-layer compile-time constant, so a decode trace is fixed per layer)."""

    idx: int
    is_global: bool  # full causal attention (l % 4 == 0)
    is_moe: bool  # MoE FFN (l >= 2); else dense PolyNorm MLP
    window: Optional[int]  # keys attended incl. the current one: 129 on SWA layers, None on global layers
    softmax_scale: float
    rope_kind: str  # "yarn" (global) | "plain" (SWA)

    @property
    def is_swa(self) -> bool:
        return not self.is_global

    @property
    def is_dense(self) -> bool:
        return not self.is_moe

    @property
    def sliding_window_size(self) -> Optional[int]:
        """Value for ttnn SDPA / FlashMLA ``sliding_window_size`` (W keys including the current one)."""
        return self.window

    @property
    def attn_kind(self) -> str:
        """``"global"`` | ``"swa"``: the key of :func:`sdpa_prefill_pc`."""
        return "global" if self.is_global else "swa"

    @property
    def kind(self) -> str:
        return ("global" if self.is_global else "swa") + "/" + ("moe" if self.is_moe else "dense")


# ------------------------------------------------------------------------------------------------------------
# Dtype policy
# ------------------------------------------------------------------------------------------------------------
_DTYPE_BITS = {"BFLOAT4_B": "4", "BFLOAT8_B": "8", "BFLOAT16": "16", "FLOAT32": "32"}


@dataclass(frozen=True)
class DtypePolicy:
    """Device dtypes per weight / tensor class (design §1.5). Math knobs (fp32 accumulation, fp32 router logits,
    fp32 mHC mixes, fp32 PolyNorm statistics) live in the compute-kernel roles and in the modules."""

    routed_experts: Any = ttnn.bfloat8_b  # gate_up and down; v1: gate_up bfp4 (eval-gated)
    shared_expert: Any = ttnn.bfloat8_b
    dense_mlp: Any = ttnn.bfloat8_b
    attention: Any = ttnn.bfloat16  # W_lat, wq_b, wq_b_gate, W_UK', W_UV', wo
    router: Any = ttnn.bfloat16  # weights; logits / sigmoid / bias / top-k in fp32
    router_bias: Any = ttnn.float32
    mhc: Any = ttnn.bfloat16  # fused, gamma-folded projection blocks
    mhc_scalars: Any = ttnn.float32
    norms: Any = ttnn.bfloat16
    polynorm_coeffs: Any = ttnn.float32
    lm_head: Any = ttnn.bfloat16
    embedding: Any = ttnn.bfloat16
    kv_cache: Any = ttnn.bfloat8_b  # latent [unit-RMS n (512) | roped k_pe (64)]; bf16 via MOTIF3_KV_CACHE_DTYPE
    activations: Any = ttnn.bfloat16  # module boundaries, residual streams (never block-float, study 01 N3/N6)
    rope_tables: Any = ttnn.bfloat16  # host fp32 -> bf16 (HF rounds cos/sin to the activation dtype)

    @property
    def tag(self) -> str:
        """Compact cache-version component, e.g. ``e8s8d8a16r16m16l16v16`` (the KV dtype is not cached, not here)."""

        def b(dt):
            return _DTYPE_BITS.get(dt.name, dt.name.lower())

        return (
            f"e{b(self.routed_experts)}s{b(self.shared_expert)}d{b(self.dense_mlp)}a{b(self.attention)}"
            f"r{b(self.router)}m{b(self.mhc)}l{b(self.lm_head)}v{b(self.embedding)}"
        )

    @property
    def kv_cache_name(self) -> str:
        """``"bfp8"`` | ``"bf16"`` (the bridge's name for ``kv_cache``; ``"bfp4"`` is not a serving option)."""
        for name, dt in KV_CACHE_DTYPE_BY_NAME.items():
            if self.kv_cache == dt:
                return name
        return "bfp4" if self.kv_cache == ttnn.bfloat4_b else self.kv_cache.name.lower()


# ------------------------------------------------------------------------------------------------------------
# Compute kernel roles (INFRA-3) -- the configs the device gates validated
# ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ComputeRole:
    """One compute-kernel config: math fidelity, fp32 dest accumulation, SFPU approx mode, packer L1 accumulation.
    ``source`` cites the gate / design section that validated it."""

    fidelity: str
    fp32_acc: bool
    approx: bool = False
    packer_l1_acc: bool = False
    source: str = ""


def make_compute_kernel_config(fidelity: str, fp32_acc: bool, *, approx: bool = False, packer_l1_acc: bool = False):
    return ttnn.types.BlackholeComputeKernelConfig(  # alias of WormholeComputeKernelConfig in this ttnn
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


# role -> config. packer_l1_acc is False for every role (INFRA-3). The G6 expert matmuls and G8 rotary_embedding_hf
# ran with packer_l1_acc=True (tests/unit/gates/test_g6_experts.py:mm_cfg, test_g8_rope.py:CKC["hifi4_fp32acc"]), but
# the infra device test measured the G6 matmuls both ways with the shared builders: identical PCC (0.9999986) and max
# error, gate_up 435-440 us / down 227-230 us traced either way; rope PCC is identical too and G5 found no router
# effect. Flip a role only with a module test that shows a gain.
COMPUTE_ROLES: Dict[str, ComputeRole] = {
    "attn_latent": ComputeRole("HiFi4", True, source="design §1.5 (HF runs the q path in fp32): x @ W_lat [4096,1664]"),
    "attn_heads": ComputeRole("HiFi4", True, source="design §1.5: wq_b, wq_b_gate, absorb / un-absorb bmm, wo"),
    "sdpa_decode": ComputeRole(
        "HiFi4", True, source="G1: paged FlashMLA decode, fp32 acc free (PCC 0.99994 SWA / 0.99993 global)"
    ),
    "sdpa_prefill": ComputeRole(
        "HiFi4", False, source="G2: SDPA prefill validated with fp32 acc off (HiFi4 or the HiFi2-approx op default)"
    ),
    # Window-free SDPA prefill calls only (global layers; SWA at S <= 128): fp32 dest acc keeps the QK scores and the
    # softmax statistics in fp32 (op PCC 0.99975 -> 0.99999, attention module 0.99965 -> 0.99996-0.99999) but makes
    # ttnn use the legacy non-streaming kernel (+30-55 % per global call at 8K-32K, 1.38 MB static CBs). NEVER with a
    # sliding window: that kernel's window mask is wrong for S >= 256 (upstream bug; test_attention.py pins it).
    # Opt-in per layer (MotifAttention(sdpa_prefill_fp32_acc="auto")); the default stays "sdpa_prefill".
    "sdpa_prefill_fp32": ComputeRole(
        "HiFi4", True, source="attention wave B1: window-free SDPA prefill with fp32 scores (opt-in, no window)"
    ),
    "rope": ComputeRole("HiFi4", True, source="G8: fused rotary_embedding_hf, HiFi4 + fp32 acc (halves max err)"),
    "norm": ComputeRole("HiFi4", True, source="study 04 §6: rms_norm default config has no fp32 acc"),
    "mhc": ComputeRole("HiFi4", True, source="G3: fused projection -> fp32 mixes; stream mixing / reduce"),
    "router": ComputeRole("HiFi4", True, source="G5: bf16 x bf16 -> fp32 logits (packer_l1_acc: no effect)"),
    "polynorm": ComputeRole("HiFi4", True, source="G6: composite PolyNorm, rms_norm HiFi4 + fp32 acc"),
    "experts": ComputeRole(
        "HiFi4", True, source="G6: bfp8 expert matmuls with the 1D-mcast configs; HiFi4 free, 2.6x lower max err"
    ),
    "shared": ComputeRole("HiFi4", True, source="INFRA-3 (G6 analog: bandwidth-bound, HiFi4 free)"),
    "dense_mlp": ComputeRole("HiFi4", True, source="INFRA-3 (G6 analog: bandwidth-bound, HiFi4 free)"),
    "lm_head": ComputeRole("HiFi4", True, source="INFRA-3 (G6 analog: bandwidth-bound, HiFi4 free)"),
    "eltwise": ComputeRole("HiFi4", True, source="design §1.5"),
    "ccl_reduce": ComputeRole(
        "HiFi4", True, source="tt/ccl.py ar_exact: fp32-accumulated local sum after an all-gather (exact fp32 adds)"
    ),
}

# Roles whose fp32 dest accumulation is deliberately off (tests assert the set).
FP32_ACC_OFF_ROLES = frozenset({"sdpa_prefill"})

# Old role names: a lookup raises with the replacement instead of silently picking a config.
_RENAMED_ROLES = {"sdpa": "split into 'sdpa_decode' (FlashMLA decode, G1) and 'sdpa_prefill' (SDPA prefill, G2)"}


# ------------------------------------------------------------------------------------------------------------
# Program configs (INFRA-4) -- verbatim copies of the configs the device gates validated
# ------------------------------------------------------------------------------------------------------------
# G1: paged FlashMLA decode must always get an explicit k_chunk_size of 128. 0/None (dynamic chunking, the op default
# without a program config) drops the causal mask under a sliding window (upstream bug); 256 clashes with a
# height-sharded Q's L1 buffer; 512 overflows L1 (static CBs 1.70-1.75 MB > 1.5 MB).
FLASH_MLA_DECODE_K_CHUNK = 128
# max_cores_per_head_batch (mcph): the cores the op spreads one head batch's k-chunks over. Global layers keep G1's 16.
FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH = 16
# SWA layers (A2, docs/OPTIMIZATION_PLAN.md §3.3; probe logs/opt/phaseA/A2): 4 (cfg.flash_mla_swa_mcph,
# MOTIF3_FLASH_MLA_SWA_MCPH; 16 restores the release config). The 129-key window always spans exactly 2 k-chunks of 128,
# so every mcph >= 2 places them alike and the output is bitwise equal to mcph 16 (measured in situ, 53-layer T32
# trace, row / sampled paths, B32 and B1, P = 128-16384 incl. unaligned: logits sha1 and tokens equal), while the op
# uses fewer cores: 30.2 -> 23.3 us per SWA call, -0.18 to -0.26 ms per decode step (39 SWA calls). Lowering the
# GLOBAL value is not neutral (uniform 8: logits differ at P >= 4K) and uniform 4 is slower at 16K (+0.9 ms).
FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH_SWA = 4
# G2: SDPA prefill q/k chunks per layer kind (q512/k512 overflows L1 at S=4096).
SDPA_PREFILL_CHUNKS: Dict[str, Tuple[int, int]] = {"swa": (128, 128), "global": (256, 256)}
# G6: 1D-multicast expert matmuls (the auto config reaches only ~12 cores: 983 / 289 us instead of 426 / 222 us).
EXPERTS_GATE_UP_GRID = (10, 8)  # 80 cores x per_core_N 1 = the 80 tiles of N = 2 x 1280
EXPERTS_GATE_UP_IN0_BLOCK_W = 8
EXPERTS_DOWN_GRID = (8, 4)  # 32 cores x per_core_N 4 = the 128 tiles of N = 4096
EXPERTS_DOWN_IN0_BLOCK_W = 4
# T64 (docs/p5_t64/t64.md §2.5, §5.1; real layer-2 weights, L1 intermediates): at M = 64 (per_core_M 2) gate_up keeps
# 10 x 8 (weight bound, +11 us) and down moves to 8 x 8 cores (per_core_N 2): 1204.1 us per MoE layer against 1283.7 us
# with down on 8 x 4 (M = 32: 1038.6 us); the M = 64 rows are bitwise equal to the M = 32 module's on both down grids.
EXPERTS_DOWN_GRID_WIDE = (8, 8)


def _grid_xy(grid) -> Tuple[int, int]:
    """``(x, y)`` from a mesh device, a ``ttnn.CoreCoord`` or a pair."""
    if hasattr(grid, "compute_with_storage_grid_size"):
        grid = grid.compute_with_storage_grid_size()
    if hasattr(grid, "x") and hasattr(grid, "y"):
        return int(grid.x), int(grid.y)
    x, y = grid
    return int(x), int(y)


# The smallest SWA mcph accepted: the bitwise argument above (the 129-key window spans exactly 2 k-chunks, placed alike
# by every mcph >= 2) does not cover 1, and mcph 1 was never measured (OPT_PHASE_A_REVIEW I-5).
FLASH_MLA_DECODE_MIN_CORES_PER_HEAD_BATCH_SWA = 2


def check_flash_mla_mcph(n) -> int:
    """``n`` if it is an integer in ``[2, 16]`` (FlashMLA decode ``max_cores_per_head_batch`` on SWA layers; 1 is
    refused, see :data:`FLASH_MLA_DECODE_MIN_CORES_PER_HEAD_BATCH_SWA`), else ``ValueError``."""
    v = int(n)
    if float(n) != v:
        raise ValueError(f"FlashMLA max_cores_per_head_batch (MOTIF3_FLASH_MLA_SWA_MCPH) must be an integer, got {n!r}")
    lo, hi = FLASH_MLA_DECODE_MIN_CORES_PER_HEAD_BATCH_SWA, FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH
    if not lo <= v <= hi:
        raise ValueError(
            f"FlashMLA max_cores_per_head_batch (MOTIF3_FLASH_MLA_SWA_MCPH) must be in [{lo}, {hi}] (the bitwise "
            f"equality with 16 holds only for >= {lo}; 1 is untested), got {n!r}"
        )
    return v


def flash_mla_decode_pc(grid=(12, 10), kind="global", swa_mcph: Optional[int] = None):
    """G1 program config for ``ttnn.transformer.paged_flash_multi_latent_attention_decode`` (mandatory):
    ``SDPAProgramConfig(grid, q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False, max_cores_per_head_batch=m)``
    with ``m`` = 16 on global layers and, on SWA layers (``kind`` "swa" or a SWA ``LayerSpec``), ``swa_mcph``
    (default :data:`FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH_SWA` = 4; A2, bitwise equal to 16 there).
    ``grid``: the chip compute grid (mesh device, CoreCoord or ``(x, y)``; 12 x 10 here). Pair it with the
    ``sdpa_decode`` compute role."""
    x, y = _grid_xy(grid)
    if _attn_kind(kind) == "swa":
        m = check_flash_mla_mcph(FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH_SWA if swa_mcph is None else swa_mcph)
    else:
        m = FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
        q_chunk_size=0,
        k_chunk_size=FLASH_MLA_DECODE_K_CHUNK,
        exp_approx_mode=False,
        max_cores_per_head_batch=m,
    )


def _attn_kind(kind) -> str:
    if isinstance(kind, LayerSpec):
        return kind.attn_kind
    k = str(kind).lower()
    if k in ("swa", "sliding", "local"):
        return "swa"
    if k in ("global", "full"):
        return "global"
    raise ValueError(f"attention kind must be 'swa' or 'global' (or a LayerSpec), got {kind!r}")


def sdpa_prefill_chunks(kind, seq_len: Optional[int] = None) -> Tuple[int, int]:
    """``(q_chunk, k_chunk)`` of G2: 128/128 on SWA layers, 256/256 on global layers, clamped to ``seq_len`` for the
    128 bucket (G2 ran S = 128 with 128/128 on both kinds). ``seq_len`` must then be a multiple of both chunks."""
    qc, kc = SDPA_PREFILL_CHUNKS[_attn_kind(kind)]
    if seq_len is not None:
        s = int(seq_len)
        qc, kc = min(qc, s), min(kc, s)
        if s % TILE or s % qc or s % kc:
            raise ValueError(f"prefill length {s} is not a multiple of the SDPA chunks ({qc}, {kc})")
    return qc, kc


def sdpa_prefill_pc(kind, grid=(12, 10), seq_len: Optional[int] = None):
    """G2 program config for ``ttnn.transformer.scaled_dot_product_attention`` prefill (q ``[1,10,S,192]``, K/V
    ``[1,2,S,192]`` with V zero-padded to 192, causal, window 129 / None): ``SDPAProgramConfig(grid, q_chunk_size,
    k_chunk_size, exp_approx_mode=False)`` with :func:`sdpa_prefill_chunks`. Pair it with the ``sdpa_prefill`` role."""
    qc, kc = sdpa_prefill_chunks(kind, seq_len)
    x, y = _grid_xy(grid)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
        q_chunk_size=int(qc),
        k_chunk_size=int(kc),
        exp_approx_mode=False,
    )


# Resumed (sp1) prefill, global layers: ttnn.transformer.chunked_scaled_dot_product_attention(Q_abs [1,10,C,576],
# K = V = the paged latent cache, page_table [1, W'], chunk_start_idx_tensor [1]) (features design D2, §3.2.2).
# (max bucket, (q_chunk, k_chunk)) entries, ascending: a bucket C uses the first entry with C <= max bucket. Gate G9
# decided it per bucket (GATES_RESULTS §12.2; lead decision F5): 128/128 at C = 128 and C >= 2048, 64/64 at C =
# 256-1024, the fastest fp32-acc config of each bucket (vs 64/64 everywhere: 1.2x at C = 128, 1.2-1.46x at C >= 2048).
# A = cfg.prefill_resume_alignment = lcm(block, every q/k up to the span cap) = 128 (64/64 serves 128-aligned starts),
# so the vLLM budget = threshold = 8192 - 128 = 8064 (prefill_plan.recommended_budget) and the bridge's
# pre-generator default generator_api.DEFAULT_PREFILL_ALIGNMENT = 128. q/k <= 128 (256/128 overflows L1,
# tt_ops_chunked.md §3.3). The chunk start must be a multiple of q and k (the kernels divide it by q_chunk without a
# device check): prefill_plan aligns every chunk start to A, and attention asserts it before every call. The table
# itself lives in prefill_plan (pure host: the planner's tests check every sp1 start against it).
SP1_GLOBAL_CHUNKS: Tuple[Tuple[int, Tuple[int, int]], ...] = _plan.DEFAULT_SP1_GLOBAL_CHUNKS
# A bf16 latent cache keeps 64/64 at every bucket (A = 64): at 128/128 the op's static CBs run through the L1_SMALL
# region (CCL semaphores) without tt-metal noticing (measured; prefill_plan.SP1_GLOBAL_CHUNKS_BF16_KV).
# cfg.sp1_global_chunk_table picks the table by cache dtype.
SP1_GLOBAL_CHUNKS_BF16_KV: Tuple[Tuple[int, Tuple[int, int]], ...] = _plan.SP1_GLOBAL_CHUNKS_BF16_KV


def sp1_global_chunks(bucket: int, table: Sequence[Tuple[int, Tuple[int, int]]] = SP1_GLOBAL_CHUNKS) -> Tuple[int, int]:
    """``(q_chunk, k_chunk)`` of the sp1 global op for chunk bucket ``bucket`` (:data:`SP1_GLOBAL_CHUNKS`)."""
    C = int(bucket)
    for upto, (q, k) in table:
        if C <= int(upto):
            if C % TILE or q % TILE or k % TILE or C % q:
                raise ValueError(f"sp1 bucket {C} with q/k chunks {(q, k)}: all must be tile multiples, q | C")
            return int(q), int(k)
    raise ValueError(f"no sp1 global chunk entry for bucket {C} in {tuple(table)}")


def resumed_prefill_pc(
    kind,
    bucket: int,
    grid=(12, 10),
    swa_tail: int = _plan.DEFAULT_SWA_TAIL,
    table: Sequence[Tuple[int, Tuple[int, int]]] = SP1_GLOBAL_CHUNKS,
):
    """Program config of an sp1 (resumed) prefill chunk of ``bucket`` rows. The compute role differs per kind:

    * ``"global"``: the flexible chunked SDPA (:func:`sp1_global_chunks` of ``table``: :data:`SP1_GLOBAL_CHUNKS` for
      a bfp8 cache, :data:`SP1_GLOBAL_CHUNKS_BF16_KV` for a bf16 one), ``exp_approx_mode=False``. Pair
      it with the **``sdpa_prefill_fp32``** role (HiFi4, fp32 dest acc): gate G9 measured the bf16-dest
      ``sdpa_prefill`` role failing on the 576-wide latent heads in all 68 cases (PCC 0.9978-0.9991, worst row 0.988)
      against 0.99983-0.99999 with fp32 acc. The op is window-free, so fp32 acc is allowed (``attention.py`` uses
      ``cfg.compute_config("sdpa_prefill_fp32")``).
    * ``"swa"``: the square ``[tail ‖ chunk]`` causal + window-129 SDPA over ``swa_tail + bucket`` rows, i.e. the G2
      config :func:`sdpa_prefill_pc` ``("swa", seq_len=swa_tail + bucket)`` (features design §3.2.3). Pair it with
      the ``sdpa_prefill`` role (never fp32 acc with a sliding window: upstream window-mask bug).
    """
    k = _attn_kind(kind)
    if k == "swa":
        return sdpa_prefill_pc("swa", grid, seq_len=int(swa_tail) + int(bucket))
    qc, kc = sp1_global_chunks(bucket, table)
    x, y = _grid_xy(grid)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
        q_chunk_size=qc,
        k_chunk_size=kc,
        exp_approx_mode=False,
    )


def _subblock_w(per_core_n: int, fp32_acc: bool = True) -> int:
    """Widest out subblock width dividing ``per_core_n`` with ``h x w <= 4`` tiles (fp32 dest acc) or 8 (bf16 dest)."""
    cap = 4 if fp32_acc else 8
    return max(d for d in range(1, cap + 1) if per_core_n % d == 0)


def mcast1d_matmul_pc(
    grid: Tuple[int, int],
    n_tiles: int,
    in0_block_w: int,
    k_tiles: Optional[int] = None,
    *,
    per_core_n: Optional[int] = None,
    per_core_m: int = 1,
    fuse_batch: bool = False,
    fused_activation=None,
    fp32_acc: bool = True,
):
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` with in0 multicast over ``grid = (x, y)`` cores (the defaults
    are verbatim ``tests/unit/gates/test_g6_experts.py:mcast1d_cfg``).

    * M: ``per_core_m`` tile rows (in0 is multicast, so every core computes all M rows of its N block):
      ``per_core_M = out_block_h = per_core_m``. 1 (default) = one 32-row tile, every decode matmul of the 32-lane
      step; 2 = the 64 gathered rows of a T64 step (MoE router / experts, LM head; docs/p5_t64/P5_T64_DESIGN.md T4).
    * N split: without ``per_core_n`` N must split evenly over the cores (G6). With ``per_core_n`` the output has
      ``ceil(N / per_core_n)`` blocks, which must fit the grid; the last one may be partial (the 215 / 860 vocab tiles
      of the LM head: the 1D factory handles the tail, verified identical to the auto config; embed_head wave B1).
    * Out subblock: ``h = 1`` at every ``per_core_m`` (docs/p5_t64/t64.md §5.1: ``(2, 1)`` on the M = 64 gate_up ran
      slower than ``(1, 1)``) and ``w`` the widest of 1..4 dividing ``per_core_N`` (fp32 dest acc: ``h x w <= 4``;
      1..8 with ``fp32_acc=False``).
    * ``fuse_batch``: G6 / the router use False, the dense / shared MLP and the LM head True.
    * ``fused_activation``: e.g. ``ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)`` (router decode linear).
    """
    ncores = int(grid[0]) * int(grid[1])
    n_tiles = int(n_tiles)
    pcm = int(per_core_m)
    if pcm < 1:
        raise ValueError(f"per_core_m must be >= 1 tile row, got {per_core_m}")
    if per_core_n is None:
        if n_tiles % ncores:
            raise ValueError(f"N = {n_tiles} tiles does not split over {ncores} cores (pass per_core_n for a tail)")
        pcn = n_tiles // ncores
    else:
        pcn = int(per_core_n)
        if pcn < 1 or cdiv(n_tiles, pcn) > ncores:
            raise ValueError(f"{cdiv(n_tiles, max(pcn, 1))} output blocks of {pcn} tiles do not fit grid {tuple(grid)}")
    if k_tiles is not None and int(k_tiles) % int(in0_block_w):
        raise ValueError(f"in0_block_w {in0_block_w} does not divide K = {k_tiles} tiles")
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(int(grid[0]), int(grid[1])),
        in0_block_w=int(in0_block_w),
        out_subblock_h=1,
        out_subblock_w=_subblock_w(pcn, fp32_acc),
        out_block_h=pcm,
        out_block_w=pcn,
        per_core_M=pcm,
        per_core_N=pcn,
        fuse_batch=bool(fuse_batch),
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def reuse_matmul_pc(grid, in0_block_w: int, per_core_n: int, *, per_core_m: int = 1, out_subblock_w: Optional[int] = None):
    """``MatmulMultiCoreReuseProgramConfig`` (batched matmul, one batch entry per core): the per-head decode bmms of
    attention (``W_UK'`` / ``W_UV'``: one head per core, ``in0_block_w`` 4, subblock 2) and the mHC split-K
    projection (``in0_block_w`` = the whole K chunk: no partial-sum spill / TF32 reload; the batched-in1 factory needs
    ``per_core_N == N``). ``out_subblock_w`` defaults to ``per_core_n`` (h = 1)."""
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*_grid_xy(grid)),
        in0_block_w=int(in0_block_w),
        out_subblock_h=1,
        out_subblock_w=int(per_core_n if out_subblock_w is None else out_subblock_w),
        per_core_M=int(per_core_m),
        per_core_N=int(per_core_n),
    )


def experts_gate_up_pc(n_out: int = 2 * 1280, k_in: int = 4096, *, m_tiles: int = 1, grid=None):
    """G6 gate_up config: ``[1,12,32,4096] bf16 @ [1,12,4096,2560] bfp8`` -> 1D mcast on 10 x 8 = 80 cores,
    ``in0_block_w`` 8, ``per_core_M`` 1, ``per_core_N`` 1, ``out_subblock_w`` 1 (426 us HiFi2 / 433 us HiFi4 traced,
    314 GB/s). Use with the ``experts`` role. ``m_tiles=2``: the 64 rows of a T64 step, same grid with ``per_core_M``
    2 (docs/p5_t64/t64.md §5.1). ``grid`` overrides the core grid."""
    g = EXPERTS_GATE_UP_GRID if grid is None else grid
    return mcast1d_matmul_pc(g, n_out // TILE, EXPERTS_GATE_UP_IN0_BLOCK_W, k_in // TILE, per_core_m=m_tiles)


def experts_down_pc(n_out: int = 4096, k_in: int = 1280, *, m_tiles: int = 1, grid=None):
    """G6 down config: ``[1,12,32,1280] bf16 @ [1,12,1280,4096] bfp8`` -> 1D mcast on 8 x 4 = 32 cores,
    ``in0_block_w`` 4, ``per_core_N`` 4, ``out_subblock_w`` 4 (222 us HiFi2 / 226 us HiFi4 traced, 301 GB/s).
    ``m_tiles=2`` (a T64 step's 64 rows): ``per_core_M`` 2 on :data:`EXPERTS_DOWN_GRID_WIDE` = 8 x 8 cores, so
    ``per_core_N`` 2 (the measured best, docs/p5_t64/t64.md §5.1). ``grid`` overrides the core grid."""
    g = grid if grid is not None else (EXPERTS_DOWN_GRID if int(m_tiles) == 1 else EXPERTS_DOWN_GRID_WIDE)
    return mcast1d_matmul_pc(g, n_out // TILE, EXPERTS_DOWN_IN0_BLOCK_W, k_in // TILE, per_core_m=m_tiles)


# ------------------------------------------------------------------------------------------------------------
# Module program configs requested in wave B1 (README §5). Each equals the module's measured local helper exactly
# (tests/unit/test_infra_config.py::test_module_program_configs_match_the_modules compares the reprs), so a module can
# switch to the shared builder without a behaviour change. Grids that do not fit cfg.compute_grid give None (auto).
# ------------------------------------------------------------------------------------------------------------
def _largest_divisor(n: int, cap: int) -> int:
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def _fits(grid, compute_grid) -> bool:
    return int(grid[0]) <= int(compute_grid[0]) and int(grid[1]) <= int(compute_grid[1])


def decode_matmul_pc(k: int, n: int, grid, *, in0_block_w: Optional[int] = None, fp32_acc: bool = True):
    """Dense-MLP / shared-expert decode matmul ``[1, 1, 32, k] @ [k, n]`` (tt/mlp.py ``decode_matmul_pc``): 1D in0
    multicast on ``grid`` with ``per_core_N = ceil(N / cores)`` (an uneven split; every core must get >= 1 tile),
    ``fuse_batch=True``, ``in0_block_w`` = the largest divisor of ``k / 32`` that is <= 8 unless given."""
    gx, gy = (int(v) for v in grid)
    cores = gx * gy
    if k % TILE or n % TILE:
        raise ValueError(f"decode matmul dims must be tile multiples, got k={k} n={n}")
    k_tiles, n_tiles = k // TILE, n // TILE
    pcn = cdiv(n_tiles, cores)
    if (cores - 1) * pcn >= n_tiles:
        raise ValueError(f"grid {tuple(grid)} is too large for N = {n_tiles} tiles (per_core_N {pcn})")
    bw = int(in0_block_w) if in0_block_w is not None else _largest_divisor(k_tiles, 8)
    if k_tiles % bw:
        raise ValueError(f"in0_block_w {bw} does not divide K = {k_tiles} tiles")
    return mcast1d_matmul_pc((gx, gy), n_tiles, bw, k_tiles, per_core_n=pcn, fuse_batch=True, fp32_acc=fp32_acc)


# (kind, matmul) -> (grid, in0_block_w) of the dense-MLP / shared-expert decode matmuls (M = one tile), from the traced
# sweep tests/unit/test_mlp.py::test_mlp_device_variants (bfp8 weights, 2026-10-01; auto = ttnn's default config):
#   dense  gate_up [4096, 3072] fp32 out: (12, 8) bw 8  40.1 us | auto 93.5 us;  down [1536, 4096]: (8, 4) bw 8 24.7 us
#   shared gate_up [4096, 320]  fp32 out: (10, 1) bw 32 13.2 us | auto 51.3 us;  down [160, 4096]:  (8, 4) bw 5  5.2 us
#   stats="replicated_gate": gate_full [4096, 1280] (10, 4) bw 16 20.0 us (auto 64.5); up [4096, 160] (5, 1) bw 32.
MLP_DECODE_MATMUL_GRIDS: Dict[Tuple[str, str], Tuple[Tuple[int, int], int]] = {
    ("dense", "gate_up"): ((12, 8), 8),  # N = 96 tiles -> 1 per core
    ("dense", "down"): ((8, 4), 8),  # N = 128 tiles -> 4 per core
    ("shared", "gate_up"): ((10, 1), 32),  # N = 10 tiles -> 1 per core
    ("shared", "down"): ((8, 4), 5),  # N = 128 tiles -> 4 per core
    ("shared", "gate_full"): ((10, 4), 16),  # N = 40 tiles -> 1 per core
    ("shared", "up"): ((5, 1), 32),  # N = 5 tiles -> 1 per core
    ("dense", "gate_full"): ((12, 8), 8),  # [4096, 12288]: N = 384 tiles -> 4 per core (not a sensible choice)
    ("dense", "up"): ((12, 4), 8),  # [4096, 1536]: N = 48 tiles -> 1 per core
}


def mlp_decode_program_configs(kind: str, dims: Mapping[str, Tuple[int, int]], compute_grid, grids=None):
    """``({matmul: program config | None}, [fallbacks])`` for the decode matmuls ``dims = {name: (k, n)}`` of ``kind``
    ("dense" | "shared"; tt/mlp.py ``build_decode_program_configs``). A tuned grid that does not fit ``compute_grid``
    or the shape gives None (ttnn's auto config: correct, slower) and is listed in ``fallbacks``; matmuls without a
    tuned entry use the auto config (not a fallback). Host code only (config objects)."""
    grids = MLP_DECODE_MATMUL_GRIDS if grids is None else grids
    cx, cy = (int(v) for v in compute_grid)
    pcs, fallbacks = {}, []
    for mm, (k, n) in dims.items():
        entry = grids.get((kind, mm))
        if entry is None:
            pcs[mm] = None
            continue
        (gx, gy), bw = entry
        if gx > cx or gy > cy:
            pcs[mm] = None
            fallbacks.append(f"{mm}: grid {(gx, gy)} exceeds the compute grid {(cx, cy)}")
            continue
        try:
            pcs[mm] = decode_matmul_pc(k, n, (gx, gy), in0_block_w=bw)
        except ValueError as e:
            pcs[mm] = None
            fallbacks.append(f"{mm}: {e}")
    return pcs, fallbacks


# Attention decode matmuls ``[1, 1, 8, K] @ [K, N]`` (tt/attention.py ``decode_matmul_program_configs``, traced on this
# Galaxy): 1D multicast for Wq_lat (8x4, 62 -> 28 us), Wkv_lat (5x4, 59 -> 21 us), wq_b (12x5, 20 -> 12 us), wq_b_gate
# (8x4, 17 -> 8 us); wo keeps the auto config (25 us; every 1D config tried was slower); the per-head bmms one head
# per core (W_UV 65 -> 7 us, W_UK 20 -> 8 us).
ATTN_DECODE_MCAST: Dict[str, Tuple[Tuple[int, int], int]] = {
    "q_lat": ((8, 4), 8),
    "kv_lat": ((5, 4), 16),
    "wq_b": ((12, 5), 4),
    "gate": ((8, 4), 8),
}
ATTN_DECODE_BMM = {"w_uk": (4, 2), "w_uv": (4, 2)}  # (in0_block_w, out_subblock_w), per_core_N = all N tiles


def attn_decode_matmul_pcs(cfg: "MotifTTConfig") -> Dict[str, Any]:
    """``{name: program config | None}`` for the attention decode matmuls (names ``q_lat, kv_lat, wq_b, gate, wo,
    w_uk, w_uv``; None = ttnn's auto config). Equals ``attention.decode_matmul_program_configs(cfg)``."""
    gx, gy = cfg.compute_grid
    K, Kq, H = cfg.hidden_size // TILE, cfg.q_lora_rank // TILE, cfg.q_heads_per_chip
    cols = {
        "q_lat": (cfg.q_lora_rank // TILE, K),  # 32
        "kv_lat": ((cfg.kv_lora_rank + cfg.rope_dim + cfg.n_signal_heads) // TILE, K),  # 20
        "wq_b": (H * cfg.head_dim // TILE, Kq),  # 60
        "gate": (cfg.signal_heads_per_chip * cfg.v_head_dim // TILE, Kq),  # 32
    }
    out: Dict[str, Any] = {}
    for name, (grid, ibw) in ATTN_DECODE_MCAST.items():
        n_tiles, k_tiles = cols[name]
        out[name] = mcast1d_matmul_pc(grid, n_tiles, ibw, k_tiles) if _fits(grid, (gx, gy)) else None
    out["wo"] = None
    n_bmm = {"w_uk": cfg.kv_lora_rank // TILE, "w_uv": cfg.v_head_dim // TILE}
    for name, (ibw, sub_w) in ATTN_DECODE_BMM.items():
        out[name] = (
            reuse_matmul_pc((gx, gy), ibw, n_bmm[name], out_subblock_w=sub_w) if gx * gy >= H else None
        )  # one head (batch entry) per core
    return out


def mhc_decode_proj_pc(grid, k_tiles: int):
    """mHC decode split-K projection (tt/mhc.py ``_bmm_pc``): one output tile per core, the whole K chunk in one
    block (``in0_block_w = k_tiles``; 16 for 32 chunks of 16384): the auto config uses ``in0_block_w = 1`` (148 us vs
    8 us traced) and K-blocked configs reload partials through TF32 (-1.2e-3 bias)."""
    return reuse_matmul_pc(grid, int(k_tiles), 1, per_core_m=1)


ROUTER_DECODE_GRID = (12, 1)  # 12 cores x per_core_N 1 = the 12 output tiles of 384 experts
ROUTER_DECODE_IN0_BLOCK_W = 32


def router_decode_pc(n_tiles: int = 12, k_tiles: int = 128, *, sigmoid: bool = False, m_tiles: int = 1):
    """Router decode linear ``[1, 1, 32, 4096] @ [4096, 384]`` (tt/moe.py ``MotifRouter._decode_pc``): 1D multicast on
    12 x 1 cores, ``in0_block_w`` 32: 13.6 us instead of 52.5 us with the auto config, identical results. ``sigmoid``:
    the SFPU sigmoid as the config's ``fused_activation`` (bitwise-identical scores, one op fewer). ``m_tiles=2``: the
    64 gathered rows of a T64 step (``per_core_M`` 2, fused sigmoid included; docs/p5_t64/t64.md §2.5, §5.1)."""
    act = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID) if sigmoid else None
    return mcast1d_matmul_pc(
        ROUTER_DECODE_GRID,
        int(n_tiles),
        ROUTER_DECODE_IN0_BLOCK_W,
        int(k_tiles),
        per_core_m=m_tiles,
        fused_activation=act,
    )


EXPERTS_PREFILL_GRID = (8, 8)
EXPERTS_PREFILL_IN0_BLOCK_W = 8
EXPERTS_PREFILL_OUT_BLOCK_W = {"gate_up": 20, "down": 16}


def experts_prefill_pc(
    m_tiles: int, n_tiles: int, *, grid=EXPERTS_PREFILL_GRID, in0_block_w: int = EXPERTS_PREFILL_IN0_BLOCK_W,
    out_block_w: int = 16,
):  # fmt: skip
    """Batched prefill expert matmul ``[1, 12, M, K] @ [1, 12, K, N]`` (tt/moe.py ``prefill_experts_pc``): 1D in1
    **multicast** (``mcast_in0=False``, ``fuse_batch=False``), each of the 64 cores owns ``M / 64`` tile rows of every
    expert's output, so each weight block is read once (auto config: 35.6 / 18.3 ms for gate_up / down at M = 4096 vs
    17.8 / 10.1 ms, bitwise identical). ``out_block_w`` 20 for gate_up, 16 for down. None when M does not split over
    the grid or N over ``out_block_w`` (small chunks: auto)."""
    ncores = int(grid[0]) * int(grid[1])
    if m_tiles % ncores or n_tiles % out_block_w:
        return None
    pm = m_tiles // ncores
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(int(grid[0]), int(grid[1])),
        in0_block_w=int(in0_block_w),
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=pm,
        out_block_w=int(out_block_w),
        per_core_M=pm,
        per_core_N=int(n_tiles),
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=False,
    )


def decode_norm_configs(width: int, grid: Tuple[int, int] = (8, 4), rows: int = TILE):
    """Width-sharded decode RMSNorm of one 32-row tile row (tt/lm_head.py ``sharded_norm_configs``):
    ``(memory_config, LayerNormShardedMultiCoreProgramConfig)`` on ``grid`` cores, shard ``[rows, width / ncores]``.
    The interleaved ``ttnn.rms_norm`` parallelizes over tile rows only, so a ``[1, 1, 8 | 32, 4096]`` input runs on
    ONE core (65 us on this Galaxy); sharded over 8 x 4 cores it takes ~6 us (+3 us of reshards). Applies to every
    decoder norm (input_layernorm, post_attention_layernorm) and the final norm."""
    gx, gy = int(grid[0]), int(grid[1])
    n = gx * gy
    if width % (n * TILE):
        raise ValueError(f"width {width} does not split into {n} tile-aligned shards")
    block_w = width // n // TILE
    subblock_w = max(d for d in (4, 3, 2, 1) if block_w % d == 0)
    mc = ttnn.create_sharded_memory_config(
        shape=(int(rows), width // n),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        subblock_w=int(subblock_w),
        block_h=int(rows) // TILE,
        block_w=int(block_w),
        inplace=False,
    )
    return mc, pc


# (grid, per_core_N, in0_block_w) of the LM-head GEMM ``[.., 32, 4096] @ [4096, Vc]`` per vocab split (tt/lm_head.py
# DEFAULT_LM_HEAD_PC; device profiler, slowest chip): "mesh" 144 us = 391 GB/s (auto 143 us), "tp" 572-579 us (auto
# 570). Both at the measured DRAM stream ceiling: the head is bandwidth bound. The last block is partial.
LM_HEAD_PC: Dict[str, Tuple[Tuple[int, int], int, int]] = {
    "mesh": ((12, 9), 2, 16),  # 108 cores x 2 tiles = the 215 vocab tiles of 6880 (last block partial)
    "tp": ((12, 9), 8, 16),  # 108 cores x 8 tiles = the 860 vocab tiles of 27520 (last block partial)
}


def lm_head_pc(vocab_tiles: int, spec, compute_grid=(12, 10), *, m_tiles: int = 1):
    """LM-head program config for ``vocab_tiles`` output tiles per chip from ``spec = (grid, per_core_N,
    in0_block_w)`` (``LM_HEAD_PC[split]``); equals tt/lm_head.py ``lm_head_program_config``. ``m_tiles``: tile rows of
    the gathered input, 1 for the 32-lane step, 2 for the 64 rows of a T64 step (``per_core_M`` 2 on the same grid:
    149.2 vs 143.8 us, rows bitwise equal to M = 32; docs/p5_t64/t64.md §5.1)."""
    grid, pcn, ibw = spec
    if not _fits(grid, compute_grid):
        raise ValueError(f"grid {tuple(grid)} exceeds the chip compute grid {tuple(compute_grid)}")
    return mcast1d_matmul_pc(
        tuple(int(v) for v in grid),
        int(vocab_tiles),
        int(ibw),
        per_core_n=int(pcn),
        per_core_m=m_tiles,
        fuse_batch=True,
    )


# ------------------------------------------------------------------------------------------------------------
# generic_op kernels: ComputeConfigDescriptor from a role (sinkhorn kernel wave B1)
# ------------------------------------------------------------------------------------------------------------
NUM_CB_SLOTS = 64  # NUM_CIRCULAR_BUFFERS on Blackhole: length of ComputeConfigDescriptor.unpack_to_dest_mode


def compute_config_descriptor(
    role: str, *, fp32_unpack_cbs: Sequence[int] = (), dst_full_sync_en: bool = False, num_cb_slots: int = NUM_CB_SLOTS
):
    """``ttnn.ComputeConfigDescriptor`` for a ``ttnn.generic_op`` compute kernel from a compute role (README §4): the
    role's fidelity / fp32 dest acc / approx mode, plus ``UnpackToDestFp32`` on the circular buffers listed in
    ``fp32_unpack_cbs`` (exact fp32 unpack straight to DEST; the default unpack truncates fp32 to TF32 through SrcA /
    SrcB) and ``dst_full_sync_en``. A generic_op takes this descriptor, not the ``BlackholeComputeKernelConfig`` that
    :meth:`MotifTTConfig.compute_config` returns (that one cannot express per-CB unpack modes). The role's
    ``packer_l1_acc`` has no descriptor field (generic_op kernels pack explicitly). Pure host object."""
    if role in _RENAMED_ROLES:
        raise KeyError(f"compute role {role!r} was {_RENAMED_ROLES[role]}")
    if role not in COMPUTE_ROLES:
        raise KeyError(f"unknown compute role {role!r}; known: {sorted(COMPUTE_ROLES)}")
    r = COMPUTE_ROLES[role]
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = getattr(ttnn.MathFidelity, r.fidelity)
    cc.fp32_dest_acc_en = bool(r.fp32_acc)
    cc.math_approx_mode = bool(r.approx)
    cc.dst_full_sync_en = bool(dst_full_sync_en)
    modes = [ttnn.UnpackToDestMode.Default] * int(num_cb_slots)
    for cb in fp32_unpack_cbs:
        if not 0 <= int(cb) < int(num_cb_slots):
            raise ValueError(f"circular buffer index {cb} outside [0, {num_cb_slots})")
        modes[int(cb)] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = modes
    return cc


# ------------------------------------------------------------------------------------------------------------
# Per-chip head bookkeeping
# ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChipHeads:
    """Heads owned by TP index ``tp`` (design §2.3.4; HF head mapping ``modeling_motif.py:745-775``).

    q head ``h`` belongs to group ``g = h // 5``; heads ``5g..5g+3`` are signal (``s = 4g + j``), ``5g+4`` noise.
    TP chip ``tp`` owns groups ``[G*tp, G*tp + G)`` (G = 2), i.e. q heads ``[10 tp, 10 tp + 10)`` and signal heads
    ``[8 tp, 8 tp + 8)``; every slice is contiguous. Local indices: ``h_loc = 5 g_loc + j``, ``s_loc = 4 g_loc + j``.
    """

    tp: int
    groups: range
    q_heads: range
    signal_heads: range

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_q_heads(self) -> int:
        return len(self.q_heads)

    @property
    def n_signal(self) -> int:
        return len(self.signal_heads)


def _pack_bucket_fields(max_seg: int, pk1: bool) -> Dict[str, Tuple[int, ...]]:
    """``pack_seg_buckets`` / ``pack_sp1_seg_buckets`` for the settings knobs ``packed_prefill_max_seg`` /
    ``packed_prefill_pk1``: the segment sizes up to ``max_seg``; no pk1 sizes when pk1 is off."""
    s = check_packed_prefill_max_seg(max_seg)
    return {
        "pack_seg_buckets": tuple(b for b in PACK_SEG_BUCKETS if b <= s),
        "pack_sp1_seg_buckets": tuple(b for b in PACK_SP1_SEG_BUCKETS if b <= s) if pk1 else (),
    }


# ------------------------------------------------------------------------------------------------------------
# The config
# ------------------------------------------------------------------------------------------------------------
@dataclass
class MotifTTConfig:
    # ---- model (from config.json) ---------------------------------------------------------------------------
    hidden_size: int = 4096
    num_layers: int = 53  # may be truncated (MOTIF3_NUM_LAYERS) for bring-up runs
    num_hidden_layers: int = 53  # as in config.json
    vocab_size: int = 220160
    n_heads: int = 80
    n_kv_heads: int = 16  # = KV groups (one KV head per differential group)
    n_noise_heads: int = 16
    head_dim: int = 192
    rope_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    intermediate_size: int = 12288
    moe_intermediate_size: int = 1280
    num_experts: int = 384
    top_k: int = 8
    num_shared_experts: int = 1
    n_dense_layers: int = 2
    interleave_moe_layer_step: int = 1
    route_scale: float = 2.0
    route_norm: bool = True
    score_func: str = "sigmoid"
    rms_norm_eps: float = 1e-5
    mhc_rms_eps: float = 1e-6  # hard-coded in HF (modeling_motif.py:188)
    polynorm_eps: float = 1e-6  # hard-coded in HF (modeling_motif.py:56)
    polynorm_output_scale: float = 0.5  # (HF reads a missing key as 1.0; config.json 2ed2ed5c sets 0.5)
    # HF / fork semantics read by tt/polynorm.py (polynorm_output_scale, check_polynorm_semantics): sigmoid(w)
    # coefficients (False = raw coefficients: not supported, validate() raises) and per-layer output scales
    # {layer: scale} overriding polynorm_output_scale ({} for 2ed2ed5c).
    polynorm_sigmoid_weight: bool = True
    polynorm_output_scale_per_layer: Dict[int, float] = field(default_factory=dict)
    polynorm_bias_clamp: Optional[float] = 0.5  # routed experts only
    hidden_clamp: Optional[float] = 1e6
    n_streams: int = 4  # mHC expansion rate
    sinkhorn_iters: int = 20
    mhc_h_post_coeff: float = 1.0  # 1 + mhc_h_post_alpha_end (absent -> 0)
    use_sliding_window: bool = True
    sliding_window_config: Optional[int] = 128  # config value; effective window = +1 (= 129 keys incl. current)
    sliding_window_pattern: str = "interleave"
    sliding_window_period: int = 4
    rope_theta: float = 1e4
    swa_rope_theta: Optional[float] = 1e4
    max_position_embeddings: int = 262144
    original_seq_len: int = 4096  # top-level: softmax mscale
    rope_factor: float = 64.0  # top-level: softmax mscale
    mscale: float = 1.0
    yarn_factor: float = 64.0  # rope_scaling.factor
    yarn_original_max_pos: int = 4096  # rope_scaling.original_max_position_embeddings
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_theta: float = 1e4  # rope_scaling.rope_theta
    rope_type: str = "yarn"
    eos_token_ids: Tuple[int, ...] = (0, 3, 6)
    bos_token_id: int = 1
    pad_token_id: int = 0

    # ---- serving (draft 1, design §1.5) -------------------------------------------------------------------
    # max_batch is the decode trace's lane count and stays 32 whatever vLLM's --max-num-seqs is (WAVE_A_REVIEW M3):
    # lanes_per_row = max_batch // dp. max_num_seqs only sizes the plugin's per-sequence block reservation.
    max_batch: int = DEFAULT_MAX_BATCH
    max_num_seqs: int = DEFAULT_MAX_BATCH
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE
    kv_pool_tokens: int = DEFAULT_KV_POOL_TOKENS
    kv_num_blocks_actual: Optional[int] = None  # allocate_kv_cache's num_blocks once known (set_kv_geometry)
    min_prefill_bucket: int = DEFAULT_MIN_PREFILL_BUCKET
    moe_prefill_chunk: int = 4096  # rows per masked-dense MoE prefill chunk (design §2.3.7)
    prefill_row_chunk: int = 8192  # per-token sublayers run in row chunks for S > 8K (design §3.3)
    trace_region_size: int = DEFAULT_TRACE_REGION_SIZE
    fabric: str = DEFAULT_FABRIC  # with a mesh: ttnn.get_fabric_config() of the opened mesh (INFRA-6)
    l1_small_size: int = DEFAULT_L1_SMALL_SIZE  # what device_params() opens the mesh with (MOTIF3_L1_SMALL_SIZE)
    mesh_l1_small_size: Optional[int] = None  # with a real mesh: its L1_SMALL bytes per core (0 = none: CCL hazard)

    # ---- features (docs/features/FEATURES_DESIGN.md §2-§3; GeneratorSettings via from_settings) -------------------
    # Span cap (MOTIF3_PREFILL_MAX_BUCKET): the largest bucket a resumed-prefill generator compiles; longer spans are
    # split into sp0 + sp1 chunks (D8). Effective value: max_prefill_span = min(cap, max_model_len).
    prefill_span_cap: int = DEFAULT_PREFILL_SPAN_CAP
    # Eager prefill cost per bucket (s) for the chunk planner. sp1 chunks add their global attention over the cached
    # prefix: gate G9's per-bucket model (prefill_plan.DEFAULT_SP1_GLOBAL_COST) when prefill_sp1_s_per_row_key is None
    # (default), else that single pre-G9 constant per (chunk row, prefix key) (0.0 drops the prefix term).
    prefill_cost_table: Dict[int, float] = field(default_factory=lambda: dict(_plan.DEFAULT_PREFILL_COST_TABLE))
    prefill_sp1_s_per_row_key: Optional[float] = None
    kv_replicated_decode: bool = False  # KV-R: every decode KV write on all 32 chips (on with prefix caching)
    spec_tokens: int = 0  # MTP self-speculation drafts per step: 0 | 1
    num_nextn_predict_layers: int = 1  # config.json: MTP layers in the checkpoint (model.mtp_layers.0)
    # Packed prefill shapes (P5; docs/p5_t64/P5_T64_DESIGN.md §3): from GeneratorSettings.packed_prefill_max_seg /
    # _max_tokens / _pk1 (from_settings), else MOTIF3_PACKED_PREFILL_MAX_SEG / _MAX_TOKENS / _PK1. Packing itself is
    # switched by the settings (MOTIF3_PACKED_PREFILL); packed_prefill_shapes() lists every shape these allow.
    pack_seg_buckets: Tuple[int, ...] = PACK_SEG_BUCKETS  # pk0 segment rows S (ascending, from PACK_SEG_BUCKETS)
    pack_sp1_seg_buckets: Tuple[int, ...] = PACK_SP1_SEG_BUCKETS  # pk1 S (from PACK_SP1_SEG_BUCKETS); () = pk1 off
    pack_max_tokens: int = DEFAULT_PACKED_PREFILL_MAX_TOKENS  # the largest packed pass T = B * S (also <= span cap)
    # Speculative verify mode (T64; docs/p5_t64/P5_T64_DESIGN.md §2.2, §4): GeneratorSettings.spec_verify via
    # from_settings. "wide" / "auto" stage the 64-row trace when spec_tokens > 0 (wide_rows_per_dp); validate() then
    # requires ring_gather "safe" (F3N R1) and refuses "auto" with the exact-fp32 router at a T64 row count missing
    # from ROUTER_EXACT_FP32_DECODE_ROWS (R-E7).
    spec_verify: str = "packed"  # "packed" | "wide" | "auto" (generator_api.SPEC_VERIFY_MODES)
    # r = T64 step / T32-spec step: the input of c* (G16 updates it; MOTIF3_WIDE_STEP_RATIO via from_settings overrides)
    wide_step_ratio: float = DEFAULT_WIDE_STEP_RATIO

    # ---- module defaults the decoder passes (README §4, §10; wave-B1 decisions) ------------------------------------
    # mHC coefficients: "motif" = tt/kernels/sinkhorn_motif (Option B, exact fp32 SFPU, ~3 us/site); "stock" = the
    # pre-clamped mhc_split_sinkhorn fallback (MHC-3; misses the 5e-3 H bound on 1 of 56 real sites).
    mhc_sinkhorn: str = "motif"
    # Router decode logits (decision D1, decided on model-level metrics): "composite" (FPU fp32 composite, 99.81 % top-8
    # agreement on real tokens) | "exact_fp32" (tt/kernels/router_fp32, 99.997 %, +24 us per MoE layer).
    router_logits: str = "composite"  # MOTIF3_ROUTER_LOGITS
    # Ring all-gathers of the TP axis (tt/ccl.py MotifCCL, P1 determinism fix): "safe" (default) | "lean" | "native".
    # spec_verify "wide" / "auto" (T64) require "safe" (validate(); F3N rule R1, review edit R-E5).
    ring_gather: str = "safe"  # MOTIF3_RING_GATHER
    # FlashMLA decode max_cores_per_head_batch on SWA layers (A2, docs/OPTIMIZATION_PLAN.md §3.3): 4 (bitwise equal to
    # the release's 16 on SWA, -0.2 ms per step); 16 restores the release config. Global layers keep 16.
    flash_mla_swa_mcph: int = FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH_SWA  # MOTIF3_FLASH_MLA_SWA_MCPH
    # Decode routing weights (A5 / B4): "fused" (B4, default since the Phase B eval passed, docs/OPT_PHASE_B_EVAL.md) |
    # "gather" (the release) | "scatter" (A5, deprecated: B4 selects the same sets and is faster); neither is bitwise
    # equal to the release, ROUTER_MASK_MODES. Prefill always takes the gather path.
    router_mask: str = "fused"  # MOTIF3_ROUTER_MASK
    # Decode routed experts (B1): "dense" (default, the release) | "sparse" (skip the local experts no live row routes
    # to; live rows bitwise equal to "dense", DECODE_EXPERTS_MODES). Prefill is not affected.
    decode_experts: str = "dense"  # MOTIF3_DECODE_EXPERTS
    # Decode routed-expert PolyNorm (B3): "fused" (default since the Phase B eval passed, docs/OPT_PHASE_B_EVAL.md; one
    # kernel, not bitwise equal to the composite) | "composite" (the release; MOE_POLYNORM_MODES). Prefill is not
    # affected.
    moe_polynorm: str = "fused"  # MOTIF3_MOE_POLYNORM
    # Decode shared-expert PolyNorm (B5): "fused" (default: moments kernel + the release's moments all-gather + apply
    # kernel, bitwise equal to the release) | "composite" (the release; SHARED_POLYNORM_MODES). Prefill and the dense
    # MLPs are not affected.
    shared_polynorm: str = "fused"  # MOTIF3_SHARED_POLYNORM
    # Prefill routed experts (B2a): "compact" (default: token-compacted, bitwise equal to "dense",
    # logs/opt/phaseB/B2a) | "dense" (the release; PREFILL_MOE_MODES), its block rows ("auto" | "32" | "64" | "128")
    # and the smallest chunk it serves.
    prefill_moe: str = "compact"  # MOTIF3_PREFILL_MOE
    prefill_moe_block: str = "auto"  # MOTIF3_PREFILL_MOE_BLOCK
    prefill_moe_min_rows: int = DEFAULT_PREFILL_MOE_MIN_ROWS  # MOTIF3_PREFILL_MOE_MIN_ROWS
    # B2b: who builds the compacted rows ("host" | "device"; PREFILL_MOE_DISPATCH_MODES) and how they are combined
    # ("matmul" | "gather"; PREFILL_MOE_COMBINE_MODES). Defaults: B2a's.
    prefill_moe_dispatch: str = "host"  # MOTIF3_PREFILL_MOE_DISPATCH
    prefill_moe_combine: str = "matmul"  # MOTIF3_PREFILL_MOE_COMBINE
    # Per-decode-step host input staging (B6a): "fast" (default: the same device inputs with fewer host ops) |
    # "release" (the release code); generator_api.HOST_STAGING_MODES. Host only: device programs and inputs unchanged.
    host_staging: str = "fast"  # MOTIF3_HOST_STAGING
    # How a decode step waits for its trace replay (B6a): "spin" (default: poll until a few ms before the predicted end,
    # then the same blocking read) | "block" (the release); generator_api.HOST_WAIT_MODES. Host only.
    host_wait: str = "spin"  # MOTIF3_HOST_WAIT
    # Traced prefill (B7): an ascending comma list of buckets ("128" = the default since its gates passed, "128,256";
    # generator_api.check_prefill_trace, "on" = "128") whose solo sp0 / sp1 chunks replay a trace captured with the
    # decode traces (traced == eager bitwise) | "off" (the release: every prefill chunk eager).
    prefill_trace: str = "128"  # MOTIF3_PREFILL_TRACE
    # The host thread every trace capture runs on (E3 / B7): "main" (default, the release) | "worker" (a short-lived
    # thread with its own malloc arena: eager sp0 128 after a capture keeps its speed in process, but served 1K / 4K
    # TTFT with the B7 prefill traces is slower, logs/opt/phaseB/B7/report.md); generator_api.CAPTURE_THREAD_MODES
    capture_thread: str = "main"  # MOTIF3_CAPTURE_THREAD

    # ---- device / mesh ----------------------------------------------------------------------------------------
    mesh_shape: Tuple[int, int] = (4, 8)
    compute_grid: Tuple[int, int] = (12, 10)  # 1x-harvested BH: 12 x 10 = 120 cores (design §3.1)
    dtypes: DtypePolicy = field(default_factory=DtypePolicy)

    # ---- paths ------------------------------------------------------------------------------------------------
    weights_dir: Path = DEFAULT_WEIGHTS_DIR
    tt_cache_root: Path = DEFAULT_TT_CACHE_ROOT
    weights_revision: str = DEFAULT_WEIGHTS_REVISION

    def __post_init__(self):
        self.mesh_shape = tuple(int(s) for s in self.mesh_shape)
        self.compute_grid = tuple(int(s) for s in self.compute_grid)
        self.eos_token_ids = tuple(int(e) for e in self.eos_token_ids)
        self.polynorm_output_scale_per_layer = _per_layer_scales(self.polynorm_output_scale_per_layer)
        self.pack_seg_buckets = tuple(int(s) for s in self.pack_seg_buckets)
        self.pack_sp1_seg_buckets = tuple(int(s) for s in self.pack_sp1_seg_buckets)
        self.wide_step_ratio = float(self.wide_step_ratio)
        self.weights_dir = Path(self.weights_dir)
        self.tt_cache_root = Path(self.tt_cache_root)
        self.axes = MeshAxes.detect(self.mesh_shape)
        self._ckc: Dict[str, Any] = {}
        self.validate()
        self.layers: Tuple[LayerSpec, ...] = tuple(self._layer_spec(i) for i in range(self.num_layers))

    # ======================================================================================================
    # construction
    # ======================================================================================================
    @classmethod
    def from_hf_config(
        cls,
        hf_config: Union[None, str, os.PathLike, Mapping[str, Any], Any] = None,
        *,
        mesh_device=None,
        mesh_shape: Optional[Sequence[int]] = None,
        **overrides: Any,
    ) -> "MotifTTConfig":
        """Build from ``config.json`` (path to the file or its directory, a dict, a transformers
        ``PretrainedConfig`` such as vLLM's ``hf_config``, or ``None`` = weights dir / hf_meta) and an opened
        ``mesh_device`` (or a ``mesh_shape`` for host-only use; default (4, 8)).

        * YaRN comes from ``rope_scaling`` or, for transformers-5 objects / ``to_dict()`` dicts, ``rope_parameters``
          (INFRA-1: ``to_dict()`` drops ``rope_scaling``); all three inputs give identical fields and layers.
        * The EOS / BOS / PAD ids come from ``generation_config.json`` next to the file, or next to the object's /
          dict's ``_name_or_path`` (a snapshot dir or a cached repo id), else from the config itself.
        * With a real ``mesh_device``: ``mesh_shape``, ``compute_grid`` and ``fabric`` (``ttnn.get_fabric_config()``,
          INFRA-6) come from the device.
        * Environment overrides: ``MOTIF3_NUM_LAYERS``, ``MOTIF3_KV_POOL_TOKENS``, ``MOTIF3_MAX_MODEL_LEN``,
          ``MOTIF3_TRACE_REGION_SIZE``, ``MOTIF3_FABRIC`` (no mesh), ``MOTIF3_TT_CACHE_PATH`` / ``TT_CACHE_PATH``,
          ``MOTIF3_L1_SMALL_SIZE``, ``MOTIF3_ROUTER_LOGITS``, ``MOTIF3_RING_GATHER``, ``MOTIF3_FLASH_MLA_SWA_MCPH``, ``MOTIF3_ROUTER_MASK``,
          ``MOTIF3_DECODE_EXPERTS``, ``MOTIF3_MOE_POLYNORM``, ``MOTIF3_SHARED_POLYNORM``, ``MOTIF3_HOST_STAGING``, ``MOTIF3_HOST_WAIT``,
          ``MOTIF3_PREFILL_TRACE``, ``MOTIF3_CAPTURE_THREAD``, ``MOTIF3_PREFILL_MOE``, ``MOTIF3_PREFILL_MOE_BLOCK``,
          ``MOTIF3_PREFILL_MOE_MIN_ROWS``, ``MOTIF3_PREFILL_MOE_DISPATCH``, ``MOTIF3_PREFILL_MOE_COMBINE``,
          ``MOTIF3_PREFILL_MAX_BUCKET``,
          ``MOTIF3_PACKED_PREFILL_MAX_SEG`` / ``_MAX_TOKENS`` / ``_PK1``, ``MOTIF3_WEIGHTS_DIR`` /
          ``HF_MODEL``, ``TT_MODEL_WEIGHTS_REVISION``.
        * Explicit ``overrides`` win (any field, plus ``kv_cache_dtype="bfp8"|"bf16"`` mapped onto ``dtypes``).
        """
        gen: Dict[str, Any] = {}
        if hf_config is None or isinstance(hf_config, (str, os.PathLike)):
            path = resolve_hf_config_path(hf_config)
            d = json.loads(path.read_text())
            gen = _generation_config_near(path.parent)
        elif isinstance(hf_config, Mapping):
            d = dict(hf_config)
            gen = _generation_config_near(d.get("_name_or_path"))
        elif hasattr(hf_config, "to_dict"):
            d = hf_config.to_dict()
            name = (
                getattr(hf_config, "_name_or_path", None)
                or getattr(hf_config, "name_or_path", None)
                or d.get("_name_or_path")
            )
            gen = _generation_config_near(name)
        else:
            raise TypeError(f"unsupported hf_config {type(hf_config)}")

        rs = rope_scaling_of(d)
        rope_type = rs.get("rope_type", rs.get("type", "default")) if rs else "default"
        top_theta = d.get("rope_theta")
        rope_theta = float(top_theta if top_theta is not None else rs.get("rope_theta", 1e4))
        n_layers_cfg = int(d.get("num_hidden_layers", 53))
        eos = gen.get("eos_token_id", d.get("eos_token_id", (0, 3, 6)))
        is_yarn = rope_type == "yarn"
        kw: Dict[str, Any] = dict(
            hidden_size=int(d.get("hidden_size", 4096)),
            num_hidden_layers=n_layers_cfg,
            num_layers=_env_int("MOTIF3_NUM_LAYERS", n_layers_cfg),
            vocab_size=int(d.get("vocab_size", 220160)),
            n_heads=int(d.get("num_attention_heads", 80)),
            n_kv_heads=int(d.get("num_key_value_heads", 16)),
            n_noise_heads=int(d.get("num_noise_heads", 16)),
            head_dim=int(d.get("head_dim", 192)),
            rope_dim=int(d.get("qk_rope_head_dim", 64)),
            v_head_dim=int(d.get("v_head_dim", 128)),
            q_lora_rank=int(d.get("q_lora_rank", 1024)),
            kv_lora_rank=int(d.get("kv_lora_rank", 512)),
            intermediate_size=int(d.get("intermediate_size", 12288)),
            moe_intermediate_size=int(d.get("moe_intermediate_size", 1280)),
            num_experts=int(d.get("num_experts", 384)),
            top_k=int(d.get("experts_top_k", 8)),
            num_shared_experts=int(d.get("num_shared_experts", 1)),
            n_dense_layers=int(d.get("n_dense_first_layers", 2)),
            interleave_moe_layer_step=int(d.get("interleave_moe_layer_step", 1)),
            route_scale=float(d.get("route_scale", 2.0)),
            route_norm=bool(d.get("route_norm", True)),
            score_func=str(d.get("score_func", "sigmoid")),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-5)),
            polynorm_output_scale=float(d.get("polynorm_output_scale", 0.5)),
            polynorm_sigmoid_weight=bool(d.get("polynorm_sigmoid_weight", True)),
            polynorm_output_scale_per_layer=_per_layer_scales(d.get("polynorm_output_scale_per_layer")),
            polynorm_bias_clamp=d.get("polynorm_bias_clamp", 0.5),
            num_nextn_predict_layers=int(d.get("num_nextn_predict_layers", 1)),
            prefill_span_cap=prefill_span_cap_from_env() or DEFAULT_PREFILL_SPAN_CAP,
            **_pack_bucket_fields(packed_prefill_max_seg_from_env(), packed_prefill_pk1_from_env()),
            pack_max_tokens=packed_prefill_max_tokens_from_env(),
            hidden_clamp=d.get("hidden_clamp", 1e6),
            n_streams=int(d.get("mhc_expansion_rate", 4)),
            sinkhorn_iters=int(d.get("mhc_sinkhorn_iters", 20)),
            mhc_h_post_coeff=1.0 + float(d.get("mhc_h_post_alpha_end", 0.0) or 0.0),
            use_sliding_window=bool(d.get("use_sliding_window", True)),
            sliding_window_config=d.get("sliding_window", 128),
            sliding_window_pattern=str(d.get("sliding_window_pattern", "interleave")),
            sliding_window_period=int(d.get("sliding_window_period", 2)),
            rope_theta=rope_theta,
            swa_rope_theta=None if d.get("swa_rope_theta") is None else float(d["swa_rope_theta"]),
            max_position_embeddings=int(d.get("max_position_embeddings", 262144)),
            original_seq_len=int(d.get("original_seq_len", 32768)),  # HF default when absent
            rope_factor=float(d.get("rope_factor", 1.0)),
            mscale=float(d.get("mscale", 1.0)),
            # HF MotifRotaryEmbedding (modeling_motif.py:338-358) defaults, read from the same rope dict
            yarn_factor=float(rs.get("factor", d.get("rope_factor", 1.0))) if is_yarn else 1.0,
            yarn_original_max_pos=(
                int(
                    rs.get(
                        "original_max_position_embeddings", d.get("original_seq_len", d.get("max_position_embeddings"))
                    )
                )
                if is_yarn
                else int(d.get("max_position_embeddings", 262144))
            ),
            yarn_beta_fast=float(rs.get("beta_fast", 32)) if is_yarn else 32.0,
            yarn_beta_slow=float(rs.get("beta_slow", 1)) if is_yarn else 1.0,
            yarn_theta=float(rs.get("rope_theta", rope_theta)) if is_yarn else rope_theta,
            rope_type=str(rope_type),
            eos_token_ids=tuple(eos) if isinstance(eos, (list, tuple)) else (int(eos),),
            bos_token_id=int(gen.get("bos_token_id", d.get("bos_token_id", 1) or 1)),
            pad_token_id=int(gen.get("pad_token_id", d.get("pad_token_id", 0) or 0)),
            kv_pool_tokens=kv_pool_tokens_from_env(),
            max_model_len=_env_int("MOTIF3_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN),
            trace_region_size=_env_int("MOTIF3_TRACE_REGION_SIZE", DEFAULT_TRACE_REGION_SIZE),
            fabric=os.environ.get("MOTIF3_FABRIC") or DEFAULT_FABRIC,
            l1_small_size=_env_int("MOTIF3_L1_SMALL_SIZE", DEFAULT_L1_SMALL_SIZE),
            router_logits=(os.environ.get("MOTIF3_ROUTER_LOGITS") or "composite").strip(),
            ring_gather=(os.environ.get("MOTIF3_RING_GATHER") or "safe").strip(),
            flash_mla_swa_mcph=_env_int("MOTIF3_FLASH_MLA_SWA_MCPH", FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH_SWA),
            router_mask=(os.environ.get("MOTIF3_ROUTER_MASK") or "fused").strip().lower(),
            decode_experts=(os.environ.get("MOTIF3_DECODE_EXPERTS") or "dense").strip().lower(),
            moe_polynorm=(os.environ.get("MOTIF3_MOE_POLYNORM") or "fused").strip().lower(),
            shared_polynorm=(os.environ.get("MOTIF3_SHARED_POLYNORM") or "fused").strip().lower(),
            prefill_moe=(os.environ.get("MOTIF3_PREFILL_MOE") or "compact").strip().lower(),
            prefill_moe_block=(os.environ.get("MOTIF3_PREFILL_MOE_BLOCK") or "auto").strip().lower(),
            prefill_moe_min_rows=_env_int("MOTIF3_PREFILL_MOE_MIN_ROWS", DEFAULT_PREFILL_MOE_MIN_ROWS),
            prefill_moe_dispatch=(os.environ.get("MOTIF3_PREFILL_MOE_DISPATCH") or "host").strip().lower(),
            prefill_moe_combine=(os.environ.get("MOTIF3_PREFILL_MOE_COMBINE") or "matmul").strip().lower(),
            host_staging=(os.environ.get("MOTIF3_HOST_STAGING") or "fast").strip().lower(),
            host_wait=(os.environ.get("MOTIF3_HOST_WAIT") or "spin").strip().lower(),
            prefill_trace=os.environ.get("MOTIF3_PREFILL_TRACE") or "128",
            capture_thread=(os.environ.get("MOTIF3_CAPTURE_THREAD") or "main").strip().lower(),
            weights_dir=resolve_weights_dir(),
            tt_cache_root=resolve_tt_cache_root(),
            weights_revision=os.environ.get("TT_MODEL_WEIGHTS_REVISION") or DEFAULT_WEIGHTS_REVISION,
        )
        if mesh_device is not None:
            kw["mesh_shape"] = tuple(mesh_device.shape)
            try:
                g = mesh_device.compute_with_storage_grid_size()
                kw["compute_grid"] = (int(g.x), int(g.y))
            except Exception:  # pragma: no cover - older mesh objects / host fakes
                pass
            fab = active_fabric_name(mesh_device)
            if fab is not None:
                kw["fabric"] = fab
            kw["mesh_l1_small_size"] = mesh_l1_small_bytes(mesh_device)
        elif mesh_shape is not None:
            kw["mesh_shape"] = tuple(mesh_shape)
        kv_name = overrides.pop("kv_cache_dtype", None)
        kw.update(overrides)
        if kv_name is not None:
            kw["dtypes"] = dataclasses.replace(
                kw.get("dtypes") or DtypePolicy(), kv_cache=kv_cache_dtype_from_name(kv_name)
            )
        return cls(**kw)

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        mesh_device=None,
        mesh_shape: Optional[Sequence[int]] = None,
        hf_config: Any = None,
        **overrides: Any,
    ) -> "MotifTTConfig":
        """The config a ``MotifGenerator.create(hf_config=, mesh_device=, settings=)`` builds (GEN-1).

        ``settings`` is a ``generator_api.GeneratorSettings``. Source: ``<settings.weights_path>/config.json`` when the
        weights are local, else ``hf_config`` (vLLM's object; safe after INFRA-1), else the default resolution.
        Mapped fields: ``num_layers``, ``max_model_len = max_seq_len``, ``max_num_seqs = max_batch_size``,
        ``max_batch = NUM_LANES`` (the trace always runs 32 lanes), ``dtypes.kv_cache`` from ``kv_cache_dtype``,
        ``kv_block_size`` from ``block_size`` when known (``allocate_kv_cache`` stays authoritative: call
        ``set_kv_geometry`` there), ``weights_dir``, ``tt_cache_root`` (``cache_path``), ``weights_revision``; the
        fabric from the device. Features: ``kv_replicated_decode = settings.kv_replicated`` (KV-R resolved),
        ``spec_tokens``, ``prefill_span_cap`` when the settings carry one (else ``MOTIF3_PREFILL_MAX_BUCKET`` / 8192);
        P5 / T64 (docs/p5_t64/P5_T64_DESIGN.md §8.3): ``pack_seg_buckets`` / ``pack_sp1_seg_buckets`` from
        ``packed_prefill_max_seg`` and ``packed_prefill_pk1``, ``pack_max_tokens = packed_prefill_max_tokens``,
        ``spec_verify``, ``wide_step_ratio`` when the settings carry one (``MOTIF3_WIDE_STEP_RATIO``; else
        :data:`DEFAULT_WIDE_STEP_RATIO`) (settings objects without these fields keep the environment's / the
        defaults). ``overrides`` win."""
        weights = getattr(settings, "weights_path", None)
        local = bool(weights) and Path(weights).is_dir()
        src: Any = None
        if local and (Path(weights) / "config.json").is_file():
            src = Path(weights) / "config.json"
        elif hf_config is not None:
            src = hf_config
        kw: Dict[str, Any] = dict(
            num_layers=int(settings.num_layers),
            max_model_len=int(settings.max_seq_len),
            max_batch=NUM_LANES,
            max_num_seqs=int(settings.max_batch_size),
            kv_cache_dtype=str(settings.kv_cache_dtype),
        )
        if getattr(settings, "block_size", None) is not None:
            kw["kv_block_size"] = check_block_size(settings.block_size)
        if local:
            kw["weights_dir"] = Path(weights)
        if getattr(settings, "cache_path", None):
            kw["tt_cache_root"] = Path(settings.cache_path)
        if getattr(settings, "weights_revision", None):
            kw["weights_revision"] = str(settings.weights_revision)
        kw["kv_replicated_decode"] = bool(getattr(settings, "kv_replicated", False))
        kw["spec_tokens"] = int(getattr(settings, "spec_tokens", 0) or 0)
        if getattr(settings, "prefill_span_cap", None) is not None:
            kw["prefill_span_cap"] = int(settings.prefill_span_cap)
        if hasattr(settings, "packed_prefill_max_seg") or hasattr(settings, "packed_prefill_pk1"):
            kw.update(
                _pack_bucket_fields(
                    getattr(settings, "packed_prefill_max_seg", DEFAULT_PACKED_PREFILL_MAX_SEG),
                    bool(getattr(settings, "packed_prefill_pk1", True)),
                )
            )
        if getattr(settings, "packed_prefill_max_tokens", None) is not None:
            kw["pack_max_tokens"] = int(settings.packed_prefill_max_tokens)
        kw["spec_verify"] = str(getattr(settings, "spec_verify", None) or "packed")
        if getattr(settings, "wide_step_ratio", None) is not None:
            kw["wide_step_ratio"] = float(settings.wide_step_ratio)
        kw.update(overrides)
        return cls.from_hf_config(src, mesh_device=mesh_device, mesh_shape=mesh_shape, **kw)

    # ======================================================================================================
    # validation
    # ======================================================================================================
    def validate(self) -> None:
        a = self.axes
        if self.n_heads % self.n_noise_heads or (self.n_heads - self.n_noise_heads) % self.n_noise_heads:
            raise ValueError("heads must form groups of grouped_ratio signal + 1 noise head")
        if self.n_kv_heads != self.n_noise_heads:
            raise ValueError("Motif GDLA needs one KV head per differential group (n_kv == n_noise)")
        if self.n_kv_heads % a.tp_size:
            raise ValueError(f"TP {a.tp_size} must divide the {self.n_kv_heads} KV groups (whole groups per chip)")
        if self.qk_nope_head_dim % TILE or self.rope_dim % TILE or self.v_head_dim % TILE:
            raise ValueError("head dims must be tile aligned")
        if self.num_experts % a.num_chips:
            raise ValueError(f"{self.num_experts} experts do not split over {a.num_chips} chips")
        if self.max_batch % a.dp_size:
            raise ValueError(f"max_batch {self.max_batch} must split over {a.dp_size} DP groups")
        if not 1 <= self.max_num_seqs <= self.max_batch:
            raise ValueError(f"max_num_seqs {self.max_num_seqs} outside [1, max_batch {self.max_batch}]")
        # INFRA-7: W = cdiv(max_model_len, block) (as the bridge computes it), so max_model_len need not be a block
        # multiple; it must be tile aligned (the last prefill bucket is max_model_len). Serving additionally needs
        # a multiple of 256 (SDPA prefill chunk), enforced by the bridge (generator_api.check_max_model_len).
        if self.max_model_len < 1 or self.max_model_len % TILE:
            raise ValueError(f"max_model_len {self.max_model_len} must be a positive multiple of {TILE}")
        if self.kv_block_size < TILE or self.kv_block_size % TILE:
            raise ValueError(f"kv_block_size {self.kv_block_size} must be a multiple of the {TILE}-row tile")
        if self.kv_num_blocks_actual is not None and self.kv_num_blocks_actual < 2:
            raise ValueError(f"a KV pool of {self.kv_num_blocks_actual} blocks cannot hold the null block + a request")
        if self.vocab_size % (a.tp_size * TILE):
            raise ValueError(f"vocab {self.vocab_size} must split into tile-aligned TP blocks")
        if not 1 <= self.num_layers <= self.num_hidden_layers:
            raise ValueError(f"num_layers {self.num_layers} outside [1, {self.num_hidden_layers}]")
        if self.sliding_window_pattern not in ("interleave", "all"):
            raise ValueError(f"unknown sliding_window_pattern {self.sliding_window_pattern!r}")
        if self.score_func != "sigmoid" or not self.route_norm:
            raise NotImplementedError("Motif-3 routes with sigmoid scores and route_norm=True")
        if not self.polynorm_sigmoid_weight:
            raise NotImplementedError(
                "polynorm_sigmoid_weight=False (raw PolyNorm coefficients) is not supported: the TT PolyNorm applies "
                "sigmoid(w) and its Horner form needs c_k > 0 (tt/polynorm.py)"
            )
        for k in self.polynorm_output_scale_per_layer:
            if not 0 <= int(k) < self.num_hidden_layers:
                raise ValueError(f"polynorm_output_scale_per_layer has layer {k} outside [0, {self.num_hidden_layers})")
        if self.l1_small_size < 0:
            raise ValueError(f"l1_small_size must be >= 0, got {self.l1_small_size}")
        if self.mhc_sinkhorn not in MHC_SINKHORN_IMPLS:
            raise ValueError(f"mhc_sinkhorn must be one of {MHC_SINKHORN_IMPLS}, got {self.mhc_sinkhorn!r}")
        check_flash_mla_mcph(self.flash_mla_swa_mcph)
        if self.router_mask not in ROUTER_MASK_MODES:
            raise ValueError(
                f"router_mask (MOTIF3_ROUTER_MASK) must be one of {ROUTER_MASK_MODES}, got {self.router_mask!r}"
            )
        if self.decode_experts not in DECODE_EXPERTS_MODES:
            raise ValueError(
                f"decode_experts (MOTIF3_DECODE_EXPERTS) must be one of {DECODE_EXPERTS_MODES}, got "
                f"{self.decode_experts!r}"
            )
        if self.host_wait not in HOST_WAIT_MODES:
            raise ValueError(f"host_wait (MOTIF3_HOST_WAIT) must be one of {HOST_WAIT_MODES}, got {self.host_wait!r}")
        self.prefill_trace = check_prefill_trace(self.prefill_trace, name="prefill_trace (MOTIF3_PREFILL_TRACE)")
        self.capture_thread = check_capture_thread(self.capture_thread, name="capture_thread (MOTIF3_CAPTURE_THREAD)")
        if self.host_staging not in HOST_STAGING_MODES:
            raise ValueError(
                f"host_staging (MOTIF3_HOST_STAGING) must be one of {HOST_STAGING_MODES}, got {self.host_staging!r}"
            )
        if self.moe_polynorm not in MOE_POLYNORM_MODES:
            raise ValueError(
                f"moe_polynorm (MOTIF3_MOE_POLYNORM) must be one of {MOE_POLYNORM_MODES}, got {self.moe_polynorm!r}"
            )
        if self.shared_polynorm not in SHARED_POLYNORM_MODES:
            raise ValueError(
                f"shared_polynorm (MOTIF3_SHARED_POLYNORM) must be one of {SHARED_POLYNORM_MODES}, got "
                f"{self.shared_polynorm!r}"
            )
        if self.prefill_moe not in PREFILL_MOE_MODES:
            raise ValueError(
                f"prefill_moe (MOTIF3_PREFILL_MOE) must be one of {PREFILL_MOE_MODES}, got {self.prefill_moe!r}"
            )
        if str(self.prefill_moe_block) not in PREFILL_MOE_BLOCKS:
            raise ValueError(
                f"prefill_moe_block (MOTIF3_PREFILL_MOE_BLOCK) must be one of {PREFILL_MOE_BLOCKS}, got "
                f"{self.prefill_moe_block!r}"
            )
        if self.prefill_moe_dispatch not in PREFILL_MOE_DISPATCH_MODES:
            raise ValueError(
                f"prefill_moe_dispatch (MOTIF3_PREFILL_MOE_DISPATCH) must be one of {PREFILL_MOE_DISPATCH_MODES}, got "
                f"{self.prefill_moe_dispatch!r}"
            )
        if self.prefill_moe_combine not in PREFILL_MOE_COMBINE_MODES:
            raise ValueError(
                f"prefill_moe_combine (MOTIF3_PREFILL_MOE_COMBINE) must be one of {PREFILL_MOE_COMBINE_MODES}, got "
                f"{self.prefill_moe_combine!r}"
            )
        mr = self.prefill_moe_min_rows
        if isinstance(mr, bool) or not isinstance(mr, int) or mr < TILE or mr % TILE:
            raise ValueError(
                f"prefill_moe_min_rows (MOTIF3_PREFILL_MOE_MIN_ROWS) must be a multiple of {TILE}, >= {TILE}, got {mr!r}"
            )
        if self.router_logits not in ROUTER_LOGITS_IMPLS:
            raise ValueError(f"router_logits must be one of {ROUTER_LOGITS_IMPLS}, got {self.router_logits!r}")
        if self.ring_gather not in RING_GATHER_MODES:
            raise ValueError(f"ring_gather must be one of {RING_GATHER_MODES}, got {self.ring_gather!r}")
        # ---- features ----
        check_prefill_span_cap(self.prefill_span_cap, self.max_model_len)
        if self.max_prefill_span not in self.prefill_buckets:
            raise ValueError(f"span cap {self.max_prefill_span} is not a prefill bucket {self.prefill_buckets}")
        if not self.prefill_cost_table or any(int(k) < 1 or float(v) <= 0 for k, v in self.prefill_cost_table.items()):
            raise ValueError(f"prefill_cost_table must map buckets to positive seconds, got {self.prefill_cost_table}")
        if self.prefill_sp1_s_per_row_key is not None and self.prefill_sp1_s_per_row_key < 0:
            raise ValueError(f"prefill_sp1_s_per_row_key must be None or >= 0, got {self.prefill_sp1_s_per_row_key}")
        if self.kv_replicated_decode not in (True, False):
            raise TypeError(f"kv_replicated_decode must be a bool, got {self.kv_replicated_decode!r}")
        if self.spec_tokens not in SUPPORTED_SPEC_TOKENS:
            raise ValueError(f"spec_tokens must be one of {SUPPORTED_SPEC_TOKENS}, got {self.spec_tokens}")
        if self.spec_tokens and self.num_nextn_predict_layers < 1:
            raise ValueError("MTP speculation needs the checkpoint's MTP layer (num_nextn_predict_layers >= 1)")
        tail = self.prefill_swa_tail
        if tail % self.kv_block_size:
            raise ValueError(f"SWA tail {tail} is not a whole number of {self.kv_block_size}-token blocks")
        # ---- packed prefill (P5) ----
        for name, got, allowed in (
            ("pack_seg_buckets", self.pack_seg_buckets, PACK_SEG_BUCKETS),
            ("pack_sp1_seg_buckets", self.pack_sp1_seg_buckets, PACK_SP1_SEG_BUCKETS),
        ):
            if list(got) != sorted(set(got)) or not set(got) <= set(allowed):
                raise ValueError(f"{name} must hold ascending, distinct values of {allowed}, got {got}")
            unaligned = [s for s in got if s % self.kv_block_size]
            if unaligned:
                raise ValueError(
                    f"{name} {unaligned}: a packed segment must be whole {self.kv_block_size}-token blocks (the pass's "
                    f"fill table concatenates the segments' block entries)"
                )
        check_packed_prefill_max_tokens(self.pack_max_tokens)
        # ---- speculative verify mode (T64) and F3N rule R1 ----
        if self.spec_verify not in SPEC_VERIFY_MODES:
            raise ValueError(f"spec_verify must be one of {SPEC_VERIFY_MODES}, got {self.spec_verify!r}")
        if not (math.isfinite(self.wide_step_ratio) and self.wide_step_ratio >= 1.0):
            raise ValueError(
                f"wide_step_ratio (T64 step / T32 step) must be finite and >= 1, got {self.wide_step_ratio}"
            )
        rows = self.wide_rows_per_dp
        if rows:
            if rows > TILE:
                raise ValueError(
                    f"spec_verify={self.spec_verify!r} puts 2 x {self.lanes_per_row} rows on each DP row, more than "
                    f"one {TILE}-row tile row: the 64-row verify step needs >= 2 DP rows (mesh {self.mesh_shape})"
                )
            if self.ring_gather != "safe":
                raise ValueError(
                    f"spec_verify={self.spec_verify!r} needs ring_gather='safe', got {self.ring_gather!r} "
                    f"(MOTIF3_RING_GATHER): the 64-row trace and the second decode trace rely on every TP-ring "
                    f"all-gather being rerouted (F3N rule R1, docs/p5_t64/f3.md §6; P5_T64_DESIGN.md X3, R-E5: 'lean' "
                    f"keeps the decode-sized gathers native and needs its own G-X run first)"
                )
            wide_m = self.dp * rows
            if (
                self.spec_verify == "auto"
                and self.router_logits == "exact_fp32"
                and wide_m not in ROUTER_EXACT_FP32_DECODE_ROWS
            ):
                raise ValueError(
                    f"spec_verify='auto' with router_logits='exact_fp32' is refused: ROUTER_EXACT_FP32_DECODE_ROWS "
                    f"{ROUTER_EXACT_FP32_DECODE_ROWS} (the gathered decode row counts at which tt/moe.py runs the "
                    f"exact-fp32 router) lacks the {wide_m}-row T64 step, whose rows would then take the composite "
                    f"router and differ from the T32 rows (P5_T64_DESIGN.md R-E7: 'auto' would not be lossless). Use "
                    f"MOTIF3_ROUTER_LOGITS=composite, or MOTIF3_SPEC_VERIFY=packed / wide"
                )

    # ======================================================================================================
    # derived model quantities
    # ======================================================================================================
    @property
    def grouped_ratio(self) -> int:
        """Signal heads per group (4)."""
        return (self.n_heads - self.n_noise_heads) // self.n_noise_heads

    @property
    def heads_per_group(self) -> int:
        """q heads per group: 4 signal + 1 noise = 5."""
        return self.grouped_ratio + 1

    @property
    def n_signal_heads(self) -> int:
        return self.grouped_ratio * self.n_noise_heads  # 64

    @property
    def qk_nope_head_dim(self) -> int:
        return self.head_dim - self.rope_dim  # 128

    @property
    def kv_latent_dim(self) -> int:
        """Per-token cache width: unit-RMS latent (512) + roped k_pe (64) = 576."""
        return self.kv_lora_rank + self.rope_dim

    @property
    def latent_proj_dim(self) -> int:
        """Columns of the fused latent projection ``[wq_a | wkv_a | lambda_proj]`` = 1024 + 576 + 64 = 1664."""
        return self.q_lora_rank + self.kv_lora_rank + self.rope_dim + self.n_signal_heads

    @property
    def effective_sliding_window(self) -> Optional[int]:
        """129: ``sliding_window + 1`` keys including the current one (study 01 §3.12)."""
        if not self.use_sliding_window or self.sliding_window_config is None:
            return None
        return int(self.sliding_window_config) + 1

    @property
    def yarn_mscale(self) -> float:
        return 0.1 * self.mscale * math.log(self.rope_factor) + 1.0 if self.rope_factor > 0 else 1.0

    def is_global_layer(self, i: int) -> bool:
        if self.effective_sliding_window is None:
            return True
        if self.sliding_window_pattern == "all":
            return False
        return i % self.sliding_window_period == 0

    def is_moe_layer(self, i: int) -> bool:
        if self.interleave_moe_layer_step == 0 or self.num_experts == 0:
            return False
        return i >= self.n_dense_layers and (i + 1) % self.interleave_moe_layer_step == 0

    def softmax_scale(self, i: int) -> float:
        """``head_dim^-0.5``, times ``mscale^2`` on global layers (HF modeling_motif.py:579-585):
        SWA 0.07216878, global 0.14467963."""
        s = self.head_dim**-0.5
        if self.is_global_layer(i) and self.max_position_embeddings > self.original_seq_len:
            s *= self.yarn_mscale**2
        return s

    def rope_kind(self, i: int) -> str:
        """ "yarn" on global layers, "plain" (``swa_rope_theta``) on SWA layers (HF modeling_motif.py:626-632)."""
        if not self.is_global_layer(i) and self.swa_rope_theta is not None:
            return "plain"
        return "yarn" if self.rope_type == "yarn" else "plain"

    def polynorm_output_scale_for_layer(self, layer_idx: int) -> float:
        """PolyNorm output scale of layer ``layer_idx``: ``polynorm_output_scale_per_layer[l]`` if present, else
        ``polynorm_output_scale`` (0.5 for Motif-3; reference ``MotifArgs.polynorm_output_scale_for_layer``)."""
        v = self.polynorm_output_scale_per_layer.get(int(layer_idx))
        return float(self.polynorm_output_scale if v is None else v)

    def _layer_spec(self, i: int) -> LayerSpec:
        g = self.is_global_layer(i)
        return LayerSpec(
            idx=i,
            is_global=g,
            is_moe=self.is_moe_layer(i),
            window=None if g else self.effective_sliding_window,
            softmax_scale=self.softmax_scale(i),
            rope_kind=self.rope_kind(i),
        )

    def layer(self, i: int) -> LayerSpec:
        return self.layers[i]

    @property
    def global_layers(self) -> Tuple[int, ...]:
        return tuple(L.idx for L in self.layers if L.is_global)

    @property
    def swa_layers(self) -> Tuple[int, ...]:
        return tuple(L.idx for L in self.layers if L.is_swa)

    @property
    def moe_layers(self) -> Tuple[int, ...]:
        return tuple(L.idx for L in self.layers if L.is_moe)

    @property
    def dense_layers(self) -> Tuple[int, ...]:
        return tuple(L.idx for L in self.layers if L.is_dense)

    # ======================================================================================================
    # per-chip partitioning
    # ======================================================================================================
    @property
    def tp(self) -> int:
        return self.axes.tp_size

    @property
    def dp(self) -> int:
        return self.axes.dp_size

    @property
    def num_chips(self) -> int:
        return self.axes.num_chips

    @property
    def q_heads_per_chip(self) -> int:
        return self.n_heads // self.tp  # 10

    @property
    def kv_groups_per_chip(self) -> int:
        return self.n_kv_heads // self.tp  # 2

    @property
    def signal_heads_per_chip(self) -> int:
        return self.n_signal_heads // self.tp  # 8

    def chip_heads(self, tp_index: int) -> ChipHeads:
        G, H, Sg = self.kv_groups_per_chip, self.q_heads_per_chip, self.signal_heads_per_chip
        if not 0 <= tp_index < self.tp:
            raise ValueError(f"tp index {tp_index} outside [0, {self.tp})")
        return ChipHeads(
            tp=tp_index,
            groups=range(G * tp_index, G * tp_index + G),
            q_heads=range(H * tp_index, H * tp_index + H),
            signal_heads=range(Sg * tp_index, Sg * tp_index + Sg),
        )

    @property
    def experts_per_chip(self) -> int:
        return self.num_experts // self.num_chips  # 12

    def experts_of_chip(self, dp_index: int, tp_index: int) -> range:
        """EP32 placement (design §2.3.7): chip ``k = dp * tp_size + tp`` holds experts ``[12k, 12k + 12)``."""
        k = self.axes.chip_index(dp_index, tp_index)
        n = self.experts_per_chip
        return range(n * k, n * k + n)

    def chip_of_expert(self, e: int) -> Tuple[int, int]:
        """(dp, tp) of the chip that owns routed expert ``e``."""
        k = e // self.experts_per_chip
        return k // self.tp, k % self.tp

    @property
    def dense_intermediate_per_chip(self) -> int:
        return self.intermediate_size // self.tp  # 1536

    @property
    def shared_intermediate(self) -> int:
        return self.moe_intermediate_size * self.num_shared_experts  # 1280

    @property
    def shared_intermediate_per_chip(self) -> int:
        return self.shared_intermediate // self.tp  # 160

    @property
    def vocab_per_chip(self) -> int:
        return self.vocab_size // self.tp  # 27520

    # ---- lanes (design §2.3.10) ----------------------------------------------------------------------------
    @property
    def lanes_per_row(self) -> int:
        """Decode lanes per DP group (8): lane ``l`` lives on DP row ``l // 8`` (all TP chips of that row)."""
        return self.max_batch // self.dp

    def lane_row(self, lane: int) -> int:
        return lane // self.lanes_per_row

    def row_lanes(self, dp_index: int) -> range:
        n = self.lanes_per_row
        return range(n * dp_index, n * dp_index + n)

    # ======================================================================================================
    # KV pool (design §1.5, §3.5, §5.1; INFRA-2)
    # ======================================================================================================
    @property
    def kv_num_blocks_expected(self) -> int:
        """The block count the bridge + plugin allocate for this config: ``ceil((pool + 32 + block * max_num_seqs)
        / block)`` = 4129 for 262,144 / 64 / 32 (4105 with ``max_num_seqs=8``); ``generator_api.expected_num_blocks``.
        Planning / tests only."""
        return expected_num_blocks(self.kv_pool_tokens, self.kv_block_size, self.max_num_seqs)

    @property
    def kv_num_blocks(self) -> int:
        """Blocks per layer cache: ``allocate_kv_cache``'s ``num_blocks`` once :meth:`set_kv_geometry` recorded it,
        else :attr:`kv_num_blocks_expected`. A generator must use the hint's value, never this formula."""
        return self.kv_num_blocks_actual if self.kv_num_blocks_actual is not None else self.kv_num_blocks_expected

    def set_kv_geometry(self, num_blocks: int, block_size: int) -> None:
        """Record the pool geometry ``allocate_kv_cache(num_blocks=, block_size=)`` received (GEN-2). All ``kv_*``
        properties are computed, so they follow. ``block_size`` must be in ``SUPPORTED_BLOCK_SIZES`` (32, 64)."""
        bs = check_block_size(block_size)
        nb = int(num_blocks)
        old = (self.kv_block_size, self.kv_num_blocks_actual)
        self.kv_block_size, self.kv_num_blocks_actual = bs, nb
        try:
            self.validate()
        except Exception:
            self.kv_block_size, self.kv_num_blocks_actual = old
            raise

    @property
    def kv_pool_tokens_allocated(self) -> int:
        return self.kv_num_blocks * self.kv_block_size  # 264,256 for 4129 blocks of 64

    @property
    def kv_blocks_per_seq(self) -> int:
        """Page-table width W = ``min(cdiv(max_model_len, block), num_blocks)`` = 512 (the bridge's formula; the
        width ``warmup_decode`` receives is authoritative)."""
        return min(cdiv(self.max_model_len, self.kv_block_size), self.kv_num_blocks)

    def prefill_page_table_entries(self, bucket: int) -> int:
        """Page-table entries a prefill of bucket length ``bucket`` writes through: ``cdiv(bucket, block)``. Feed
        ``paged_fill_cache`` exactly the first that many entries, so every bucket has one fixed program shape
        whatever W is (WAVE_A_REVIEW M10, ATTN-6)."""
        n = cdiv(int(bucket), self.kv_block_size)
        if n > cdiv(self.max_model_len, self.kv_block_size):
            raise ValueError(f"bucket {bucket} exceeds max_model_len {self.max_model_len}")
        return n

    @property
    def kv_cache_shape(self) -> Tuple[int, int, int, int]:
        """Per-layer paged latent cache ``[num_blocks, 1, block, 576]`` (vLLM hint ``(N, 1, bs, 576)``)."""
        return (self.kv_num_blocks, 1, self.kv_block_size, self.kv_latent_dim)

    def kv_cache_bytes_per_chip(self) -> int:
        """All layers, replicated per chip, tile-exact (``generator_api.kv_cache_bytes_per_chip``): bfp8 1088 B and
        bf16 2048 B per 32 x 32 tile -> 8.57 GB for 53 layers x 4129 blocks of 64 in bfp8."""
        name = self.dtypes.kv_cache_name
        if name in KV_CACHE_DTYPE_BY_NAME:
            return _api_kv_cache_bytes_per_chip(self.kv_num_blocks, self.kv_block_size, self.num_layers, name)
        elems = self.num_layers * self.kv_num_blocks * self.kv_block_size * self.kv_latent_dim
        return elems * 576 // 1024  # bfp4_b

    # ======================================================================================================
    # prefill buckets (design §2.3.10)
    # ======================================================================================================
    @property
    def prefill_buckets(self) -> Tuple[int, ...]:
        """Powers of two from 128 to ``max_model_len`` (all warmed before decode trace capture)."""
        out, b = [], self.min_prefill_bucket
        while b < self.max_model_len:
            out.append(b)
            b *= 2
        out.append(self.max_model_len)
        return tuple(out)

    def prefill_bucket(self, seq_len: int) -> int:
        if seq_len < 1:
            raise ValueError("empty prompt")
        for b in self.prefill_buckets:
            if seq_len <= b:
                return b
        raise ValueError(f"prompt of {seq_len} tokens exceeds max_model_len {self.max_model_len}")

    # ======================================================================================================
    # resumed / chunked prefill (docs/features/FEATURES_DESIGN.md §3.1-§3.2; tt/prefill_plan.py)
    # ======================================================================================================
    @property
    def max_prefill_span(self) -> int:
        """Effective span cap ``min(prefill_span_cap, max_model_len)`` (8192): the largest bucket a resumed-prefill
        generator compiles and runs; longer spans become several chunks (``generator.max_prefill_span``)."""
        return min(int(self.prefill_span_cap), int(self.max_model_len))

    @property
    def prefill_span_buckets(self) -> Tuple[int, ...]:
        """The buckets ``<= max_prefill_span`` (128 ... 8192): what a resumed-prefill generator warms, for sp0 and sp1.
        ``prefill_buckets`` (all buckets to ``max_model_len``) stays the draft-1 list."""
        return tuple(b for b in self.prefill_buckets if b <= self.max_prefill_span)

    @property
    def prefill_swa_tail(self) -> int:
        """Earlier keys an SWA query sees: window 129 - 1 = 128 (0 without a sliding window). The sp1 SWA path reads
        that many cached rows before the chunk; ``c0`` is 0 below it."""
        w = self.effective_sliding_window
        return 0 if w is None else int(w) - 1

    def sp1_global_chunk_table(self, kv_dtype=None) -> Tuple[Tuple[int, Tuple[int, int]], ...]:
        """The sp1 global ``(max bucket, (q, k))`` table for a latent-cache dtype (a name or a ttnn dtype; default
        this config's ``dtypes.kv_cache``): :data:`SP1_GLOBAL_CHUNKS` for bfp8, :data:`SP1_GLOBAL_CHUNKS_BF16_KV` for
        bf16 (``prefill_plan.sp1_global_chunk_table``)."""
        name = self.dtypes.kv_cache_name if kv_dtype is None else kv_cache_dtype_name(kv_dtype)
        return _plan.sp1_global_chunk_table(name)

    def sp1_global_chunks(self, bucket: int, kv_dtype=None) -> Tuple[int, int]:
        """``(q_chunk, k_chunk)`` of the sp1 global op at ``bucket`` for a cache of ``kv_dtype`` (default this
        config's): :meth:`sp1_global_chunk_table`."""
        return sp1_global_chunks(bucket, self.sp1_global_chunk_table(kv_dtype))

    @property
    def prefill_resume_alignment(self) -> int:
        """``A = lcm(kv_block_size, q_chunk, k_chunk)`` over every sp1 global config up to the span cap
        (``prefill_plan.resume_alignment``) for this config's cache dtype: 128 for the defaults (bfp8, block 32 or 64;
        gate G9's per-bucket q/k 128/128 at C = 128 and C >= 2048, 64/64 at 256-1024), 64 with a bf16 cache (64/64).
        Every chunk start is a multiple of ``A`` (the chunked kernels divide the start by ``q_chunk`` with no device
        check); vLLM's chunk budget = threshold = ``max_prefill_span - A`` (8064; 8064 is also aligned for A = 64)."""
        a = int(self.kv_block_size)
        for b in self.prefill_span_buckets:
            q, k = self.sp1_global_chunks(b)
            a = math.lcm(a, q, k)
        return a

    @property
    def sp1_page_table_width(self) -> int:
        """``W'`` of the sp1 SDPA page table: ``round_up(cdiv(max_model_len + max_prefill_span, block), 8)`` = 640
        (32768 + 8192 at block 64). Fixed, so one program per bucket; padded with 0 (never -1)."""
        return _plan.sdpa_table_width(self.max_model_len, self.max_prefill_span, self.kv_block_size)

    def resumed_prefill_pc(self, kind, bucket: int, kv_dtype=None):
        """sp1 prefill program config (:func:`resumed_prefill_pc`) on this chip's grid; ``kind`` = "global" | "swa" |
        a ``LayerSpec`` | a layer index. ``kv_dtype``: the cache the global op reads (default this config's
        ``dtypes.kv_cache``); it selects the (q, k) table (:meth:`sp1_global_chunk_table`)."""
        if isinstance(kind, int) and not isinstance(kind, bool):
            kind = self.layer(kind)
        return resumed_prefill_pc(
            kind, bucket, self.compute_grid, self.prefill_swa_tail, table=self.sp1_global_chunk_table(kv_dtype)
        )

    def prefill_cost(self, bucket: int, start: int = 0) -> float:
        """Estimated seconds of one prefill chunk (:func:`prefill_plan.prefill_cost_model` with this config's table)."""
        return _plan.prefill_cost_model(self.prefill_cost_table, sp1_s_per_row_key=self.prefill_sp1_s_per_row_key)(
            int(bucket), int(start)
        )

    def plan_prefill_row(self, start: int, end: int) -> "_plan.RowPlan":
        """:func:`prefill_plan.plan_prefill_row` with this config's geometry: block size, ``A``, the span buckets, the
        span cap, the SWA tail and the cost table."""
        if int(end) > self.max_model_len:
            raise ValueError(f"prefill row end {end} exceeds max_model_len {self.max_model_len}")
        return _plan.plan_prefill_row(
            int(start),
            int(end),
            block_size=self.kv_block_size,
            align=self.prefill_resume_alignment,
            buckets=self.prefill_span_buckets,
            span_cap=self.max_prefill_span,
            swa_tail=self.prefill_swa_tail,
            cost=_plan.prefill_cost_model(self.prefill_cost_table, sp1_s_per_row_key=self.prefill_sp1_s_per_row_key),
        )

    # ======================================================================================================
    # packed prefill (P5; docs/p5_t64/P5_T64_DESIGN.md §3)
    # ======================================================================================================
    @property
    def pack_max_seg(self) -> int:
        """The largest packed segment S of either kind (0 when neither has one): the planner's ``max_seg``."""
        return max(self.pack_seg_buckets + self.pack_sp1_seg_buckets, default=0)

    @property
    def pack_pk1(self) -> bool:
        """pk1 passes (resumed chunks at one common start) are configured: ``pack_sp1_seg_buckets`` is not empty."""
        return bool(self.pack_sp1_seg_buckets)

    @property
    def pack_tokens_cap(self) -> int:
        """The largest packed pass T: ``min(pack_max_tokens, max_prefill_span)`` (8192). A pass of T rows runs the
        bucket-T programs the solo warmup already compiled, so T must be a span bucket."""
        return min(int(self.pack_max_tokens), self.max_prefill_span)

    def packed_prefill_shapes(self) -> Tuple[Tuple[Any, ...], ...]:
        """Every packed pass shape the generator warms before the decode capture (§3.5), in a fixed order:
        ``("pk0", T, S)`` for S in ``pack_seg_buckets``, then ``("pk1", T, S, tails)`` for S in
        ``pack_sp1_seg_buckets`` and ``tails`` in ``generator_api.PK1_TAIL_VARIANTS`` (review edit R-E2: the shared /
        distinct SWA tail gathers are different programs), with ``T = B * S`` for B in ``generator_api.PACK_BATCHES``,
        ``T <= pack_tokens_cap`` and T a span bucket. Each key is the ``PrefillPass.shape`` of a packed pass
        (``prefill_plan``; solo chunks keep ``(path, bucket)``): after the capture a pass whose key was not warmed runs
        as solo chunks. Defaults: 22 pk0 keys and 17 x 2 pk1 keys."""
        cap, buckets = self.pack_tokens_cap, set(self.prefill_span_buckets)

        def totals(S: int) -> List[int]:
            return [B * S for B in PACK_BATCHES if B * S <= cap and B * S in buckets]

        pk0, pk1 = PACKED_PASS_KINDS
        shapes: List[Tuple[Any, ...]] = [(pk0, T, S) for S in self.pack_seg_buckets for T in totals(S)]
        for S in self.pack_sp1_seg_buckets:
            shapes += [(pk1, T, S, tails) for T in totals(S) for tails in PK1_TAIL_VARIANTS]
        return tuple(shapes)

    # ======================================================================================================
    # decode KV writes (KV-R) and the MTP layer (docs/features/FEATURES_DESIGN.md §3.4-§3.6)
    # ======================================================================================================
    @property
    def kv_write_mode(self) -> str:
        """``row`` | ``row_split`` | ``all`` | ``all_split`` (``generator_api.kv_write_mode``): KV-R x speculation.
        Fixed per server (the decode trace is captured with it); one mode shared by all 53 layers + the MTP layer."""
        return kv_write_mode(self.kv_replicated_decode, self.spec_tokens > 0)

    @property
    def wide_rows_per_dp(self) -> int:
        """Rows per DP row of the T64 verify step (docs/p5_t64/P5_T64_DESIGN.md §4.1) when this launch stages it
        (``spec_tokens > 0`` and ``spec_verify`` "wide" / "auto"), else 0: ``2 * lanes_per_row`` = 16 (``[8 anchors |
        8 drafts]``, still one 32-row tile row; ``generator_api.WIDE_ROWS_PER_GROUP``). The gathered T64 step has
        ``dp * wide_rows_per_dp`` = 64 rows. Modules allocate their 64-row constants in their constructors when it is
        set (F3N rule R3: the LM head's 64-row argmax constants, the MoE's 64-row top-k pads), never after a capture."""
        if int(self.spec_tokens) > 0 and self.spec_verify in WIDE_SPEC_VERIFY_MODES:
            return 2 * self.lanes_per_row
        return 0

    @property
    def mtp_layer_idx(self) -> int:
        """The MTP layer's index: ``num_hidden_layers`` (53; the reference's ``layer_idx``, TT-cache part ``L53``),
        whatever ``num_layers`` a truncated run uses."""
        return int(self.num_hidden_layers)

    def mtp_layer_spec(self) -> LayerSpec:
        """The MTP layer (``model.mtp_layers.0``): SWA in "all" mode (window 129), plain RoPE (``swa_rope_theta``),
        softmax scale ``head_dim^-0.5`` = 0.07216878, dense MLP (reference ``MotifMTP``: ``GDLAttention(args, 53,
        swa=True)``). ``cfg.layer(53)`` does not exist and ``is_moe_layer(53)`` would say MoE: pass this spec
        explicitly (``MotifAttention(spec=...)``)."""
        if self.num_nextn_predict_layers < 1:
            raise ValueError("this checkpoint has no MTP layer (num_nextn_predict_layers = 0)")
        rope = "plain" if self.swa_rope_theta is not None else ("yarn" if self.rope_type == "yarn" else "plain")
        return LayerSpec(
            idx=self.mtp_layer_idx,
            is_global=False,
            is_moe=False,
            window=self.effective_sliding_window,
            softmax_scale=self.head_dim**-0.5,
            rope_kind=rope,
        )

    @property
    def mtp_kv_layers(self) -> int:
        """Latent caches beyond the ``num_layers`` main ones: 1 (the MTP layer) with speculation, else 0."""
        return 1 if self.spec_tokens > 0 else 0

    @property
    def kv_pool_layers(self) -> int:
        """Latent caches the generator allocates: ``num_layers + mtp_kv_layers`` (vLLM keeps accounting 53)."""
        return int(self.num_layers) + self.mtp_kv_layers

    def kv_pool_bytes_per_chip(self) -> int:
        """:meth:`kv_cache_bytes_per_chip` including the MTP cache (``kv_pool_layers`` layers): 8.73 GB for 53 + 1
        layers x 4129 blocks of 64 in bfp8."""
        name = self.dtypes.kv_cache_name
        if name not in KV_CACHE_DTYPE_BY_NAME:
            return self.kv_cache_bytes_per_chip() * self.kv_pool_layers // self.num_layers
        return _api_kv_cache_bytes_per_chip(self.kv_num_blocks, self.kv_block_size, self.kv_pool_layers, name)

    # ======================================================================================================
    # compute kernel configs, program configs and memory configs
    # ======================================================================================================
    def compute_role(self, role: str) -> ComputeRole:
        if role in _RENAMED_ROLES:
            raise KeyError(f"compute role {role!r} was {_RENAMED_ROLES[role]}")
        if role not in COMPUTE_ROLES:
            raise KeyError(f"unknown compute role {role!r}; known: {sorted(COMPUTE_ROLES)}")
        return COMPUTE_ROLES[role]

    def compute_config(self, role: str):
        """Compute-kernel config for an op class (see ``COMPUTE_ROLES``); math approx mode is off everywhere."""
        if role not in self._ckc:
            r = self.compute_role(role)
            self._ckc[role] = make_compute_kernel_config(
                r.fidelity, r.fp32_acc, approx=r.approx, packer_l1_acc=r.packer_l1_acc
            )
        return self._ckc[role]

    def flash_mla_decode_pc(self, kind="global"):
        """G1 FlashMLA decode program config on this chip's compute grid (:func:`flash_mla_decode_pc`); ``kind`` =
        "global" (default, mcph 16) | "swa" (mcph ``flash_mla_swa_mcph``, A2) | a ``LayerSpec`` | a layer index."""
        if isinstance(kind, int) and not isinstance(kind, bool):
            kind = self.layer(kind)
        return flash_mla_decode_pc(self.compute_grid, kind, swa_mcph=self.flash_mla_swa_mcph)

    def sdpa_prefill_pc(self, kind, seq_len: Optional[int] = None):
        """G2 SDPA prefill program config; ``kind`` = "swa" | "global" | a ``LayerSpec`` | a layer index."""
        if isinstance(kind, int) and not isinstance(kind, bool):
            kind = self.layer(kind)
        return sdpa_prefill_pc(kind, self.compute_grid, seq_len)

    def experts_gate_up_pc(self, m_tiles: int = 1, grid=None):
        """G6 routed-expert gate_up config (``[.,12,32,4096] @ [.,12,4096,2 * moe_intermediate]``); ``m_tiles=2`` for
        the 64 rows of a T64 step (:func:`experts_gate_up_pc`)."""
        return experts_gate_up_pc(2 * self.moe_intermediate_size, self.hidden_size, m_tiles=m_tiles, grid=grid)

    def experts_down_pc(self, m_tiles: int = 1, grid=None):
        """G6 routed-expert down config (``[.,12,32,moe_intermediate] @ [.,12,moe_intermediate,4096]``); ``m_tiles=2``
        for the 64 rows of a T64 step, on 8 x 8 cores unless ``grid`` says otherwise (:func:`experts_down_pc`)."""
        return experts_down_pc(self.hidden_size, self.moe_intermediate_size, m_tiles=m_tiles, grid=grid)

    # ---- wave-B1 module builders (README §5; each equals the module's measured local config) ----------------------
    def attn_decode_matmul_pcs(self) -> Dict[str, Any]:
        """Attention decode matmul configs ``{q_lat, kv_lat, wq_b, gate, wo, w_uk, w_uv}`` (:func:`attn_decode_matmul_pcs`)."""
        return attn_decode_matmul_pcs(self)

    def attn_decode_matmul_pc(self, name: str):
        """One entry of :meth:`attn_decode_matmul_pcs` (None = auto config)."""
        pcs = attn_decode_matmul_pcs(self)
        if name not in pcs:
            raise KeyError(f"unknown attention decode matmul {name!r}; known: {sorted(pcs)}")
        return pcs[name]

    def mlp_decode_dims(self, kind: str) -> Dict[str, Tuple[int, int]]:
        """``{matmul: (k, n)}`` per chip of the dense MLP / shared expert decode matmuls (gate | up fused)."""
        if kind == "dense":
            n = self.dense_intermediate_per_chip
            return {"gate_up": (self.hidden_size, 2 * n), "down": (n, self.hidden_size)}
        if kind == "shared":
            n = self.shared_intermediate_per_chip
            return {
                "gate_up": (self.hidden_size, 2 * n),
                "down": (n, self.hidden_size),
                "gate_full": (self.hidden_size, self.shared_intermediate),
                "up": (self.hidden_size, n),
            }
        raise ValueError(f"MLP kind must be 'dense' or 'shared', got {kind!r}")

    def mlp_decode_pcs(self, kind: str, dims: Optional[Mapping[str, Tuple[int, int]]] = None, grids=None):
        """``({matmul: pc | None}, [fallbacks])`` (:func:`mlp_decode_program_configs` on this chip's compute grid;
        ``dims`` default :meth:`mlp_decode_dims`)."""
        return mlp_decode_program_configs(kind, dims or self.mlp_decode_dims(kind), self.compute_grid, grids)

    def mlp_decode_pc(self, kind: str, matmul: str):
        """One dense-MLP / shared-expert decode config (None = auto: no tuned grid, or it does not fit)."""
        dims = self.mlp_decode_dims(kind)
        if matmul not in dims:
            raise KeyError(f"unknown {kind} matmul {matmul!r}; known: {sorted(dims)}")
        return self.mlp_decode_pcs(kind, {matmul: dims[matmul]})[0][matmul]

    def mhc_decode_proj_pc(self, decode_split: int = 32):
        """mHC decode split-K projection config on the chip grid: ``K = 4 x hidden / decode_split`` per block."""
        k = self.n_streams * self.hidden_size // int(decode_split)
        if k % TILE:
            raise ValueError(f"decode_split {decode_split} does not give whole-tile K chunks")
        return mhc_decode_proj_pc(self.compute_grid, k // TILE)

    def router_decode_pc(self, sigmoid: bool = False, m_tiles: int = 1):
        """Router decode linear config (:func:`router_decode_pc`; ``m_tiles=2`` for the 64 rows of a T64 step), or None
        when the grid has fewer than 12 columns."""
        if self.compute_grid[0] < ROUTER_DECODE_GRID[0]:
            return None
        return router_decode_pc(self.num_experts // TILE, self.hidden_size // TILE, sigmoid=sigmoid, m_tiles=m_tiles)

    def experts_prefill_pc(self, m_rows: int, n_out: int, out_block_w: int):
        """Batched prefill expert matmul config for ``m_rows`` rows and ``n_out`` output columns
        (:func:`experts_prefill_pc`; None = auto)."""
        if m_rows % TILE or n_out % TILE:
            raise ValueError(f"m_rows {m_rows} / n_out {n_out} must be tile multiples")
        return experts_prefill_pc(m_rows // TILE, n_out // TILE, out_block_w=out_block_w)

    def experts_prefill_gate_up_pc(self, m_rows: int):
        return self.experts_prefill_pc(m_rows, 2 * self.moe_intermediate_size, EXPERTS_PREFILL_OUT_BLOCK_W["gate_up"])

    def experts_prefill_down_pc(self, m_rows: int):
        return self.experts_prefill_pc(m_rows, self.hidden_size, EXPERTS_PREFILL_OUT_BLOCK_W["down"])

    def decode_norm_configs(self, grid: Tuple[int, int] = (8, 4), rows: int = TILE, width: Optional[int] = None):
        """Width-sharded decode RMSNorm ``(memory_config, program_config)`` (:func:`decode_norm_configs`)."""
        return decode_norm_configs(self.hidden_size if width is None else int(width), grid, rows)

    def vocab_per_shard(self, vocab_split: str = "mesh") -> int:
        """Vocab columns per chip: 6880 for "mesh" (all 32 chips, decision EMB-D1) or 27520 for "tp"."""
        n = {"mesh": self.num_chips, "tp": self.tp}.get(vocab_split)
        if n is None:
            raise ValueError(f"vocab_split must be 'mesh' or 'tp', got {vocab_split!r}")
        if self.vocab_size % (n * TILE):
            raise ValueError(f"vocab {self.vocab_size} does not split into {n} tile-aligned blocks")
        return self.vocab_size // n

    def lm_head_pc(self, vocab_split: str = "mesh", spec=None, m_tiles: int = 1):
        """LM-head GEMM config (:data:`LM_HEAD_PC`; ``spec`` overrides, ``"auto"`` = None); ``m_tiles=2`` for the 64
        rows of a T64 step (:func:`lm_head_pc`)."""
        if spec == "auto":
            return None
        spec = spec if spec is not None else LM_HEAD_PC[vocab_split]
        return lm_head_pc(self.vocab_per_shard(vocab_split) // TILE, spec, self.compute_grid, m_tiles=m_tiles)

    def compute_config_descriptor(self, role: str, **kw):
        """generic_op ``ComputeConfigDescriptor`` from a role (:func:`compute_config_descriptor`)."""
        return compute_config_descriptor(role, **kw)

    @property
    def hifi4_fp32(self):
        return self.compute_config("router")

    @property
    def hifi2_fp32(self):
        """A HiFi2 + fp32-acc config (no role uses HiFi2 any more; the experts are HiFi4 since G6)."""
        if "_hifi2_fp32" not in self._ckc:
            self._ckc["_hifi2_fp32"] = make_compute_kernel_config("HiFi2", True)
        return self._ckc["_hifi2_fp32"]

    @property
    def dram(self):
        return ttnn.DRAM_MEMORY_CONFIG

    @property
    def l1(self):
        return ttnn.L1_MEMORY_CONFIG

    @property
    def num_cores(self) -> int:
        return self.compute_grid[0] * self.compute_grid[1]

    @property
    def fabric_config(self):
        return fabric_config_from_name(self.fabric)

    def device_params(self, **extra: Any) -> Dict[str, Any]:
        """:func:`device_params` with this config's fabric, trace region and L1_SMALL size."""
        return device_params(self.fabric, self.trace_region_size, self.l1_small_size, **extra)

    # ======================================================================================================
    # TT weight cache (design §2.3.11)
    # ======================================================================================================
    @property
    def cache_version_tag(self) -> str:
        """e.g. ``motif3-2ed2ed5c-c1-e8s8d8a16r16m16l16v16``: checkpoint revision + transform format + dtypes."""
        return f"motif3-{self.weights_revision[:8]}-c{CACHE_FORMAT_VERSION}-{self.dtypes.tag}"

    @property
    def cache_dir(self) -> Path:
        """``<TT_CACHE_PATH>/<version-tag>/mesh<R>x<C>`` (shards differ between (4,8) and (8,4))."""
        return self.tt_cache_root / self.cache_version_tag / self.axes.tag

    def cache_file(self, name: str, layer: Optional[int] = None) -> Path:
        """Cache *prefix* for ``ttnn.as_tensor(cache_file_name=...)`` (ttnn appends
        ``_dtype_<D>_layout_<L>.tensorbin``). ``layer=None`` = model-global tensors (embedding, norm, LM head)."""
        sub = "global" if layer is None else f"L{int(layer):02d}"
        return self.cache_dir / sub / name

    # ======================================================================================================
    # misc
    # ======================================================================================================
    def describe(self) -> str:
        a = self.axes
        kv_src = "allocated" if self.kv_num_blocks_actual is not None else "expected"
        wide = ""
        if self.wide_rows_per_dp:
            wide = f" (T64 {self.wide_rows_per_dp} rows/DP row, r {self.wide_step_ratio:g})"

        def segs(sizes: Tuple[int, ...]) -> str:
            return "/".join(str(s) for s in sizes) or "off"

        return (
            f"MotifTTConfig(mesh={a.mesh_shape} tp_axis={a.tp_axis} dp_axis={a.dp_axis} tp={a.tp_size} dp={a.dp_size}; "
            f"fabric={self.fabric}; layers={self.num_layers}/{self.num_hidden_layers} global={len(self.global_layers)} "
            f"swa={len(self.swa_layers)} moe={len(self.moe_layers)}; "
            f"rope={self.rope_type} yarn_factor={self.yarn_factor}; "
            f"heads/chip q={self.q_heads_per_chip} kv={self.kv_groups_per_chip} sig={self.signal_heads_per_chip}; "
            f"experts/chip={self.experts_per_chip}; lanes/row={self.lanes_per_row} (max_num_seqs {self.max_num_seqs}); "
            f"kv blocks={self.kv_num_blocks}x{self.kv_block_size} ({kv_src}, {self.dtypes.kv_cache_name}) "
            f"W={self.kv_blocks_per_seq}; buckets={self.prefill_buckets[0]}..{self.prefill_buckets[-1]}; "
            f"trace={self.trace_region_size}; l1_small={self.l1_small_size} (mesh {self.mesh_l1_small_size}); "
            f"sinkhorn={self.mhc_sinkhorn} router={self.router_logits} router_mask={self.router_mask} "
            f"decode_experts={self.decode_experts} moe_polynorm={self.moe_polynorm} shared_polynorm={self.shared_polynorm} "
            f"prefill_moe={self.prefill_moe}/{self.prefill_moe_block}/{self.prefill_moe_min_rows}/{self.prefill_moe_dispatch}/{self.prefill_moe_combine} host_staging={self.host_staging} host_wait={self.host_wait} "
            f"prefill_trace={self.prefill_trace} capture_thread={self.capture_thread} "
            f"ring_gather={self.ring_gather} "
            f"mla_mcph swa={self.flash_mla_swa_mcph}/global={FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH}; "
            f"span cap={self.max_prefill_span} A={self.prefill_resume_alignment} kv_write={self.kv_write_mode} "
            f"spec={self.spec_tokens} spec_verify={self.spec_verify}{wide}; "
            f"pack S={segs(self.pack_seg_buckets)} pk1 S={segs(self.pack_sp1_seg_buckets)} T<={self.pack_tokens_cap}; "
            f"cache={self.cache_dir})"
        )


__all__ = [
    "ATTN_DECODE_BMM",
    "ATTN_DECODE_MCAST",
    "CACHE_FORMAT_VERSION",
    "COMPUTE_ROLES",
    "ChipHeads",
    "ComputeRole",
    "DEFAULT_FABRIC",
    "DEFAULT_KV_POOL_TOKENS",
    "DEFAULT_L1_SMALL_SIZE",
    "DEFAULT_WIDE_STEP_RATIO",
    "DtypePolicy",
    "EXPERTS_DOWN_GRID",
    "EXPERTS_DOWN_GRID_WIDE",
    "EXPERTS_GATE_UP_GRID",
    "EXPERTS_PREFILL_OUT_BLOCK_W",
    "FLASH_MLA_DECODE_K_CHUNK",
    "FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH",
    "FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH_SWA",
    "FLASH_MLA_DECODE_MIN_CORES_PER_HEAD_BATCH_SWA",
    "FP32_ACC_OFF_ROLES",
    "KV_CACHE_DTYPE_BY_NAME",
    "LM_HEAD_PC",
    "LayerSpec",
    "MHC_SINKHORN_IMPLS",
    "MLP_DECODE_MATMUL_GRIDS",
    "MeshAxes",
    "MotifTTConfig",
    "NUM_CB_SLOTS",
    "ROUTER_DECODE_GRID",
    "RING_GATHER_MODES",
    "ROUTER_EXACT_FP32_DECODE_ROWS",
    "ROUTER_LOGITS_IMPLS",
    "ROUTER_MASK_MODES",
    "DECODE_EXPERTS_MODES",
    "PREFILL_MOE_MODES",
    "PREFILL_MOE_BLOCKS",
    "PREFILL_MOE_COMBINE_MODES",
    "PREFILL_MOE_DISPATCH_MODES",
    "DEFAULT_PREFILL_MOE_MIN_ROWS",
    "MOE_POLYNORM_MODES",
    "SHARED_POLYNORM_MODES",
    "HOST_STAGING_MODES",
    "HOST_WAIT_MODES",
    "SDPA_PREFILL_CHUNKS",
    "SP1_GLOBAL_CHUNKS",
    "SP1_GLOBAL_CHUNKS_BF16_KV",
    "SUPPORTED_BLOCK_SIZES",
    "active_fabric_name",
    "attn_decode_matmul_pcs",
    "close_motif_mesh",
    "compute_config_descriptor",
    "decode_matmul_pc",
    "decode_norm_configs",
    "device_params",
    "experts_down_pc",
    "experts_gate_up_pc",
    "experts_prefill_pc",
    "fabric_config_from_name",
    "check_flash_mla_mcph",
    "flash_mla_decode_pc",
    "kv_cache_dtype_from_name",
    "kv_cache_dtype_name",
    "lm_head_pc",
    "make_compute_kernel_config",
    "mcast1d_matmul_pc",
    "mesh_l1_small_bytes",
    "mesh_shape_from_env",
    "mhc_decode_proj_pc",
    "mlp_decode_program_configs",
    "open_motif_mesh",
    "require_l1_small",
    "resolve_hf_config_path",
    "resolve_tt_cache_root",
    "resolve_weights_dir",
    "resumed_prefill_pc",
    "reuse_matmul_pc",
    "rope_scaling_of",
    "router_decode_pc",
    "sdpa_prefill_chunks",
    "sdpa_prefill_pc",
    "sp1_global_chunks",
]
