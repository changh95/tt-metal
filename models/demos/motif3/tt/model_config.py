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
  (``polynorm_sigmoid_weight``, ``polynorm_output_scale_per_layer``).

Import rule (design §2.1): this module imports only the standard library, ``ttnn`` and ``generator_api`` (stdlib +
torch; the KV-pool and weights-location helpers shared with the vLLM bridge). It never opens a device and never
imports other ``models/demos/**`` packages.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import ttnn

from .generator_api import DEFAULT_KV_POOL_TOKENS as _API_DEFAULT_KV_POOL_TOKENS
from .generator_api import L1_SMALL_SIZE as _API_L1_SMALL_SIZE
from .generator_api import NUM_LANES, SUPPORTED_BLOCK_SIZES, cdiv, check_block_size, expected_num_blocks
from .generator_api import hf_cache_snapshot as _hf_cache_snapshot
from .generator_api import kv_cache_bytes_per_chip as _api_kv_cache_bytes_per_chip
from .generator_api import kv_pool_tokens_from_env, resolve_tt_cache_path, resolve_weights_location

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
FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH = 16
# G2: SDPA prefill q/k chunks per layer kind (q512/k512 overflows L1 at S=4096).
SDPA_PREFILL_CHUNKS: Dict[str, Tuple[int, int]] = {"swa": (128, 128), "global": (256, 256)}
# G6: 1D-multicast expert matmuls (the auto config reaches only ~12 cores: 983 / 289 us instead of 426 / 222 us).
EXPERTS_GATE_UP_GRID = (10, 8)  # 80 cores x per_core_N 1 = the 80 tiles of N = 2 x 1280
EXPERTS_GATE_UP_IN0_BLOCK_W = 8
EXPERTS_DOWN_GRID = (8, 4)  # 32 cores x per_core_N 4 = the 128 tiles of N = 4096
EXPERTS_DOWN_IN0_BLOCK_W = 4


def _grid_xy(grid) -> Tuple[int, int]:
    """``(x, y)`` from a mesh device, a ``ttnn.CoreCoord`` or a pair."""
    if hasattr(grid, "compute_with_storage_grid_size"):
        grid = grid.compute_with_storage_grid_size()
    if hasattr(grid, "x") and hasattr(grid, "y"):
        return int(grid.x), int(grid.y)
    x, y = grid
    return int(x), int(y)


def flash_mla_decode_pc(grid=(12, 10)):
    """G1 program config for ``ttnn.transformer.paged_flash_multi_latent_attention_decode`` (mandatory):
    ``SDPAProgramConfig(grid, q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False, max_cores_per_head_batch=16)``.
    ``grid``: the chip compute grid (mesh device, CoreCoord or ``(x, y)``; 12 x 10 here). Pair it with the
    ``sdpa_decode`` compute role."""
    x, y = _grid_xy(grid)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
        q_chunk_size=0,
        k_chunk_size=FLASH_MLA_DECODE_K_CHUNK,
        exp_approx_mode=False,
        max_cores_per_head_batch=FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH,
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
    fuse_batch: bool = False,
    fused_activation=None,
    fp32_acc: bool = True,
):
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` with in0 multicast over ``grid = (x, y)`` cores and M = one
    tile row (the defaults are verbatim ``tests/unit/gates/test_g6_experts.py:mcast1d_cfg``).

    * N split: without ``per_core_n`` N must split evenly over the cores (G6). With ``per_core_n`` the output has
      ``ceil(N / per_core_n)`` blocks, which must fit the grid; the last one may be partial (the 215 / 860 vocab tiles
      of the LM head: the 1D factory handles the tail, verified identical to the auto config; embed_head wave B1).
    * Out subblock: the widest of 1..4 dividing ``per_core_N`` (fp32 dest acc; 1..8 with ``fp32_acc=False``).
    * ``fuse_batch``: G6 / the router use False, the dense / shared MLP and the LM head True.
    * ``fused_activation``: e.g. ``ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)`` (router decode linear).
    """
    ncores = int(grid[0]) * int(grid[1])
    n_tiles = int(n_tiles)
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
        out_block_h=1,
        out_block_w=pcn,
        per_core_M=1,
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


def experts_gate_up_pc(n_out: int = 2 * 1280, k_in: int = 4096):
    """G6 gate_up config: ``[1,12,32,4096] bf16 @ [1,12,4096,2560] bfp8`` -> 1D mcast on 10 x 8 = 80 cores,
    ``in0_block_w`` 8, ``per_core_M`` 1, ``per_core_N`` 1, ``out_subblock_w`` 1 (426 us HiFi2 / 433 us HiFi4 traced,
    314 GB/s). Use with the ``experts`` role."""
    return mcast1d_matmul_pc(EXPERTS_GATE_UP_GRID, n_out // TILE, EXPERTS_GATE_UP_IN0_BLOCK_W, k_in // TILE)


def experts_down_pc(n_out: int = 4096, k_in: int = 1280):
    """G6 down config: ``[1,12,32,1280] bf16 @ [1,12,1280,4096] bfp8`` -> 1D mcast on 8 x 4 = 32 cores,
    ``in0_block_w`` 4, ``per_core_N`` 4, ``out_subblock_w`` 4 (222 us HiFi2 / 226 us HiFi4 traced, 301 GB/s)."""
    return mcast1d_matmul_pc(EXPERTS_DOWN_GRID, n_out // TILE, EXPERTS_DOWN_IN0_BLOCK_W, k_in // TILE)


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


def router_decode_pc(n_tiles: int = 12, k_tiles: int = 128, *, sigmoid: bool = False):
    """Router decode linear ``[1, 1, 32, 4096] @ [4096, 384]`` (tt/moe.py ``MotifRouter._decode_pc``): 1D multicast on
    12 x 1 cores, ``in0_block_w`` 32: 13.6 us instead of 52.5 us with the auto config, identical results. ``sigmoid``:
    the SFPU sigmoid as the config's ``fused_activation`` (bitwise-identical scores, one op fewer)."""
    act = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID) if sigmoid else None
    return mcast1d_matmul_pc(
        ROUTER_DECODE_GRID, int(n_tiles), ROUTER_DECODE_IN0_BLOCK_W, int(k_tiles), fused_activation=act
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


def lm_head_pc(vocab_tiles: int, spec, compute_grid=(12, 10)):
    """LM-head program config for ``vocab_tiles`` output tiles per chip from ``spec = (grid, per_core_N,
    in0_block_w)`` (``LM_HEAD_PC[split]``); equals tt/lm_head.py ``lm_head_program_config``."""
    grid, pcn, ibw = spec
    if not _fits(grid, compute_grid):
        raise ValueError(f"grid {tuple(grid)} exceeds the chip compute grid {tuple(compute_grid)}")
    return mcast1d_matmul_pc(tuple(int(v) for v in grid), int(vocab_tiles), int(ibw), per_core_n=int(pcn), fuse_batch=True)


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

    # ---- module defaults the decoder passes (README §4, §10; wave-B1 decisions) ------------------------------------
    # mHC coefficients: "motif" = tt/kernels/sinkhorn_motif (Option B, exact fp32 SFPU, ~3 us/site); "stock" = the
    # pre-clamped mhc_split_sinkhorn fallback (MHC-3; misses the 5e-3 H bound on 1 of 56 real sites).
    mhc_sinkhorn: str = "motif"
    # Router decode logits (decision D1, decided on model-level metrics): "composite" (FPU fp32 composite, 99.81 % top-8
    # agreement on real tokens) | "exact_fp32" (tt/kernels/router_fp32, 99.997 %, +24 us per MoE layer).
    router_logits: str = "composite"  # MOTIF3_ROUTER_LOGITS

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
          ``MOTIF3_L1_SMALL_SIZE``, ``MOTIF3_ROUTER_LOGITS``, ``MOTIF3_WEIGHTS_DIR`` /
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
        fabric from the device. ``overrides`` win."""
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
        if self.router_logits not in ROUTER_LOGITS_IMPLS:
            raise ValueError(f"router_logits must be one of {ROUTER_LOGITS_IMPLS}, got {self.router_logits!r}")

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

    def flash_mla_decode_pc(self):
        """G1 FlashMLA decode program config on this chip's compute grid (:func:`flash_mla_decode_pc`)."""
        return flash_mla_decode_pc(self.compute_grid)

    def sdpa_prefill_pc(self, kind, seq_len: Optional[int] = None):
        """G2 SDPA prefill program config; ``kind`` = "swa" | "global" | a ``LayerSpec`` | a layer index."""
        if isinstance(kind, int) and not isinstance(kind, bool):
            kind = self.layer(kind)
        return sdpa_prefill_pc(kind, self.compute_grid, seq_len)

    def experts_gate_up_pc(self):
        """G6 routed-expert gate_up config (``[.,12,32,4096] @ [.,12,4096,2 * moe_intermediate]``)."""
        return experts_gate_up_pc(2 * self.moe_intermediate_size, self.hidden_size)

    def experts_down_pc(self):
        """G6 routed-expert down config (``[.,12,32,moe_intermediate] @ [.,12,moe_intermediate,4096]``)."""
        return experts_down_pc(self.hidden_size, self.moe_intermediate_size)

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

    def router_decode_pc(self, sigmoid: bool = False):
        """Router decode linear config (:func:`router_decode_pc`), or None when the grid has fewer than 12 columns."""
        if self.compute_grid[0] < ROUTER_DECODE_GRID[0]:
            return None
        return router_decode_pc(self.num_experts // TILE, self.hidden_size // TILE, sigmoid=sigmoid)

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

    def lm_head_pc(self, vocab_split: str = "mesh", spec=None):
        """LM-head GEMM config (:data:`LM_HEAD_PC`; ``spec`` overrides, ``"auto"`` = None)."""
        if spec == "auto":
            return None
        spec = spec if spec is not None else LM_HEAD_PC[vocab_split]
        return lm_head_pc(self.vocab_per_shard(vocab_split) // TILE, spec, self.compute_grid)

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
            f"sinkhorn={self.mhc_sinkhorn} router={self.router_logits}; cache={self.cache_dir})"
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
    "DtypePolicy",
    "EXPERTS_DOWN_GRID",
    "EXPERTS_GATE_UP_GRID",
    "EXPERTS_PREFILL_OUT_BLOCK_W",
    "FLASH_MLA_DECODE_K_CHUNK",
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
    "ROUTER_LOGITS_IMPLS",
    "SDPA_PREFILL_CHUNKS",
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
    "flash_mla_decode_pc",
    "kv_cache_dtype_from_name",
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
    "reuse_matmul_pc",
    "rope_scaling_of",
    "router_decode_pc",
    "sdpa_prefill_chunks",
    "sdpa_prefill_pc",
]
