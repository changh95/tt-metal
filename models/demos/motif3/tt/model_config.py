# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``MotifTTConfig``: every static decision the Motif-3 TT modules share.

Built from the HF ``config.json`` plus an opened mesh (or just a mesh shape for host-only use). It carries

* the mesh axis roles (design §3.1): logical mesh (4, 8) = 4 DP groups ("rows", 8 user lanes each) x TP8
  ("cols"). The TP axis is *detected* as the mesh dim of size 8, so an (8, 4) mesh (the plugin's ``BH-Galaxy``
  preset) works too. Modules never hard-code ``cluster_axis`` 0/1; they use ``cfg.axes.tp_axis`` /
  ``cfg.axes.dp_axis`` (or the role names in ``tt/ccl.py``);
* per-chip head counts (design §2.3.4): 10 q heads, 2 KV groups, 8 signal heads per TP chip;
* the dtype policy per weight class (design §1.5): bfp8 routed/shared/dense experts, bf16 attention, router,
  mHC, LM head and embedding, bfp8 latent KV cache, fp32 math knobs;
* the layer schedule (design §2.3.1; study 01 §2.1): dense 0-1 / MoE 2-52, global ``l % 4 == 0`` (14 layers,
  YaRN RoPE, scale 0.14467963) / SWA (39 layers, plain RoPE, scale 0.07216878, window 129 keys incl. current);
* the KV pool (design §1.5, §3.5, §5.1): block 64, 262,144 pool tokens + one block per lane of headroom
  (= 4128 blocks, the plugin's ``get_num_available_blocks_tt`` formula), ``max_model_len`` 32768;
* prefill buckets (powers of 2, 128 ... 32768; design §2.3.10), compute-kernel configs, trace region size and
  the TT weight-cache layout ``<TT_CACHE_PATH>/<version-tag>/mesh<R>x<C>/{L<nn>|global}/<name>`` (design §2.3.11).

Import rule (design §2.1): this module imports only the standard library, ``torch``-free code and ``ttnn``.
It never opens a device and never imports other ``models/demos/**`` packages.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import ttnn

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
DEFAULT_KV_POOL_TOKENS = 262144
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_MAX_BATCH = 32
DEFAULT_MIN_PREFILL_BUCKET = 128
DEFAULT_FABRIC = "FABRIC_2D_TORUS_XY"  # fallback: FABRIC_1D_RING (design §1.3, kill criterion G4)

TILE = 32


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else int(default)


def resolve_weights_dir() -> Path:
    """Checkpoint directory: ``MOTIF3_WEIGHTS_DIR``, else ``HF_MODEL`` (if it is a directory), else the local
    snapshot. (TIS sets ``HF_MODEL`` to a snapshot dir, design §2.3.11; a repo-id ``HF_MODEL`` is resolved by the
    vLLM bridge, not here.)"""
    for env in ("MOTIF3_WEIGHTS_DIR", "HF_MODEL"):
        v = os.environ.get(env)
        if v and Path(v).is_dir():
            return Path(v)
    return DEFAULT_WEIGHTS_DIR


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
    """TT weight-cache root: ``TT_CACHE_PATH`` (TIS / tt-model contract), else ``motif-3/tt_cache``."""
    v = os.environ.get("TT_CACHE_PATH")
    return Path(v) if v else DEFAULT_TT_CACHE_ROOT


def fabric_config_from_name(name: Optional[str] = None):
    """``ttnn.FabricConfig`` by name (default ``MOTIF3_FABRIC`` env, else FABRIC_2D_TORUS_XY)."""
    name = name or os.environ.get("MOTIF3_FABRIC") or DEFAULT_FABRIC
    fc = ttnn.FabricConfig.__members__.get(name)
    if fc is None:
        raise ValueError(f"unknown fabric config {name!r}; expected one of {list(ttnn.FabricConfig.__members__)}")
    return fc


def device_params(
    fabric: Optional[str] = None, trace_region_size: Optional[int] = None, **extra: Any
) -> Dict[str, Any]:
    """``device_params`` for tt-metal's ``mesh_device`` pytest fixture (root ``conftest.py``)::

        @pytest.mark.parametrize("mesh_device, device_params", [((4, 8), device_params())], indirect=True)

    Keys understood by the fixture: ``fabric_config``, ``trace_region_size``, ``l1_small_size``,
    ``worker_l1_size``, ``num_command_queues``, ``dispatch_core_axis`` (BH forces COL unless a fabric tensix
    config is also given), ``reliability_mode``, ``fabric_tensix_config``, ``fabric_router_config``.
    """
    p = {
        "fabric_config": fabric_config_from_name(fabric),
        "trace_region_size": int(
            trace_region_size
            if trace_region_size is not None
            else _env_int("MOTIF3_TRACE_REGION_SIZE", DEFAULT_TRACE_REGION_SIZE)
        ),
    }
    p.update(extra)
    return p


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
    def kind(self) -> str:
        return ("global" if self.is_global else "swa") + "/" + ("moe" if self.is_moe else "dense")


# ------------------------------------------------------------------------------------------------------------
# Dtype policy
# ------------------------------------------------------------------------------------------------------------
_DTYPE_BITS = {"BFLOAT4_B": "4", "BFLOAT8_B": "8", "BFLOAT16": "16", "FLOAT32": "32"}


@dataclass(frozen=True)
class DtypePolicy:
    """Device dtypes per weight / tensor class (design §1.5). Math knobs (fp32 accumulation, fp32 router logits,
    fp32 mHC mixes, fp32 PolyNorm statistics) live in the compute-kernel configs and in the modules."""

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
    kv_cache: Any = ttnn.bfloat8_b  # latent [unit-RMS n (512) | roped k_pe (64)]
    activations: Any = ttnn.bfloat16  # module boundaries, residual streams (never block-float, study 01 N3/N6)
    rope_tables: Any = ttnn.bfloat16  # host fp32 -> bf16 (HF rounds cos/sin to the activation dtype)

    @property
    def tag(self) -> str:
        """Compact cache-version component, e.g. ``e8s8d8a16r16m16l16v16``."""

        def b(dt):
            return _DTYPE_BITS.get(dt.name, dt.name.lower())

        return (
            f"e{b(self.routed_experts)}s{b(self.shared_expert)}d{b(self.dense_mlp)}a{b(self.attention)}"
            f"r{b(self.router)}m{b(self.mhc)}l{b(self.lm_head)}v{b(self.embedding)}"
        )


# ------------------------------------------------------------------------------------------------------------
# Compute kernel configs
# ------------------------------------------------------------------------------------------------------------
def make_compute_kernel_config(fidelity: str, fp32_acc: bool, *, approx: bool = False, packer_l1_acc: bool = True):
    return ttnn.types.BlackholeComputeKernelConfig(  # alias of WormholeComputeKernelConfig in this ttnn
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


# role -> (fidelity, fp32 dest acc). Design §1.5 / §2.3.x; gates G5/G6 may retune.
COMPUTE_ROLES: Dict[str, Tuple[str, bool]] = {
    "attn_latent": ("HiFi4", True),  # x @ W_lat [4096, 1664] (HF runs the q path in fp32)
    "attn_heads": ("HiFi4", True),  # wq_b, wq_b_gate, absorb / un-absorb bmm, wo
    "sdpa": ("HiFi4", True),  # FlashMLA decode, SDPA prefill
    "rope": ("HiFi4", True),
    "norm": ("HiFi4", True),  # rms_norm (default config has no fp32 acc, study 04 §6)
    "mhc": ("HiFi4", True),  # fused projection -> fp32 mixes; stream mixing
    "router": ("HiFi4", True),  # bf16 x bf16 -> fp32 logits
    "polynorm": ("HiFi4", True),
    "experts": ("HiFi2", True),  # bfp8 routed experts (try HiFi4 in G6)
    "shared": ("HiFi2", True),
    "dense_mlp": ("HiFi2", True),
    "lm_head": ("HiFi2", True),
    "eltwise": ("HiFi4", True),
}


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
    polynorm_output_scale: float = 0.5
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
    max_batch: int = DEFAULT_MAX_BATCH
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    kv_block_size: int = DEFAULT_KV_BLOCK_SIZE
    kv_pool_tokens: int = DEFAULT_KV_POOL_TOKENS
    min_prefill_bucket: int = DEFAULT_MIN_PREFILL_BUCKET
    moe_prefill_chunk: int = 4096  # rows per masked-dense MoE prefill chunk (design §2.3.7)
    prefill_row_chunk: int = 8192  # per-token sublayers run in row chunks for S > 8K (design §3.3)
    trace_region_size: int = DEFAULT_TRACE_REGION_SIZE
    fabric: str = DEFAULT_FABRIC

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
        ``PretrainedConfig``, or ``None`` = weights dir / hf_meta) and an opened ``mesh_device`` (or a
        ``mesh_shape`` for host-only use; default (4, 8)). ``generation_config.json`` next to the file, when
        present, supplies the EOS set. Environment overrides: ``MOTIF3_NUM_LAYERS``, ``MOTIF3_KV_POOL_TOKENS``,
        ``MOTIF3_MAX_MODEL_LEN``, ``MOTIF3_TRACE_REGION_SIZE``, ``MOTIF3_FABRIC``, ``TT_CACHE_PATH``,
        ``MOTIF3_WEIGHTS_DIR`` / ``HF_MODEL``, ``TT_MODEL_WEIGHTS_REVISION``. Explicit ``overrides`` win.
        """
        gen: Dict[str, Any] = {}
        if hf_config is None or isinstance(hf_config, (str, os.PathLike)):
            path = resolve_hf_config_path(hf_config)
            d = json.loads(path.read_text())
            gp = path.parent / "generation_config.json"
            if gp.is_file():
                gen = json.loads(gp.read_text())
        elif isinstance(hf_config, Mapping):
            d = dict(hf_config)
        elif hasattr(hf_config, "to_dict"):
            d = hf_config.to_dict()
        else:
            raise TypeError(f"unsupported hf_config {type(hf_config)}")

        rs = d.get("rope_scaling") or {}
        rope_type = rs.get("rope_type", rs.get("type", "default")) if isinstance(rs, dict) else "default"
        n_layers_cfg = int(d.get("num_hidden_layers", 53))
        eos = gen.get("eos_token_id", d.get("eos_token_id", (0, 3, 6)))
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
            polynorm_bias_clamp=d.get("polynorm_bias_clamp", 0.5),
            hidden_clamp=d.get("hidden_clamp", 1e6),
            n_streams=int(d.get("mhc_expansion_rate", 4)),
            sinkhorn_iters=int(d.get("mhc_sinkhorn_iters", 20)),
            mhc_h_post_coeff=1.0 + float(d.get("mhc_h_post_alpha_end", 0.0) or 0.0),
            use_sliding_window=bool(d.get("use_sliding_window", True)),
            sliding_window_config=d.get("sliding_window", 128),
            sliding_window_pattern=str(d.get("sliding_window_pattern", "interleave")),
            sliding_window_period=int(d.get("sliding_window_period", 2)),
            rope_theta=float(d.get("rope_theta", 1e4)),
            swa_rope_theta=None if d.get("swa_rope_theta") is None else float(d["swa_rope_theta"]),
            max_position_embeddings=int(d.get("max_position_embeddings", 262144)),
            original_seq_len=int(d.get("original_seq_len", 32768)),  # HF default when absent
            rope_factor=float(d.get("rope_factor", 1.0)),
            mscale=float(d.get("mscale", 1.0)),
            yarn_factor=float(rs.get("factor", d.get("rope_factor", 1.0))) if rope_type == "yarn" else 1.0,
            yarn_original_max_pos=(
                int(
                    rs.get(
                        "original_max_position_embeddings", d.get("original_seq_len", d.get("max_position_embeddings"))
                    )
                )
                if rope_type == "yarn"
                else int(d.get("max_position_embeddings", 262144))
            ),
            yarn_beta_fast=float(rs.get("beta_fast", 32)) if rope_type == "yarn" else 32.0,
            yarn_beta_slow=float(rs.get("beta_slow", 1)) if rope_type == "yarn" else 1.0,
            yarn_theta=(
                float(rs.get("rope_theta", d.get("rope_theta", 1e4)))
                if rope_type == "yarn"
                else float(d.get("rope_theta", 1e4))
            ),
            rope_type=str(rope_type),
            eos_token_ids=tuple(eos) if isinstance(eos, (list, tuple)) else (int(eos),),
            bos_token_id=int(gen.get("bos_token_id", d.get("bos_token_id", 1) or 1)),
            pad_token_id=int(gen.get("pad_token_id", d.get("pad_token_id", 0) or 0)),
            kv_pool_tokens=_env_int("MOTIF3_KV_POOL_TOKENS", DEFAULT_KV_POOL_TOKENS),
            max_model_len=_env_int("MOTIF3_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN),
            trace_region_size=_env_int("MOTIF3_TRACE_REGION_SIZE", DEFAULT_TRACE_REGION_SIZE),
            fabric=os.environ.get("MOTIF3_FABRIC") or DEFAULT_FABRIC,
            weights_dir=resolve_weights_dir(),
            tt_cache_root=resolve_tt_cache_root(),
            weights_revision=os.environ.get("TT_MODEL_WEIGHTS_REVISION") or DEFAULT_WEIGHTS_REVISION,
        )
        if mesh_device is not None:
            kw["mesh_shape"] = tuple(mesh_device.shape)
            try:
                g = mesh_device.compute_with_storage_grid_size()
                kw["compute_grid"] = (int(g.x), int(g.y))
            except Exception:  # pragma: no cover - older mesh objects
                pass
        elif mesh_shape is not None:
            kw["mesh_shape"] = tuple(mesh_shape)
        kw.update(overrides)
        return cls(**kw)

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
        if self.max_model_len % self.kv_block_size:
            raise ValueError("max_model_len must be a multiple of the KV block size")
        if self.vocab_size % (a.tp_size * TILE):
            raise ValueError(f"vocab {self.vocab_size} must split into tile-aligned TP blocks")
        if not 1 <= self.num_layers <= self.num_hidden_layers:
            raise ValueError(f"num_layers {self.num_layers} outside [1, {self.num_hidden_layers}]")
        if self.sliding_window_pattern not in ("interleave", "all"):
            raise ValueError(f"unknown sliding_window_pattern {self.sliding_window_pattern!r}")
        if self.score_func != "sigmoid" or not self.route_norm:
            raise NotImplementedError("Motif-3 routes with sigmoid scores and route_norm=True")

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
    # KV pool (design §1.5, §3.5, §5.1)
    # ======================================================================================================
    @property
    def kv_num_blocks(self) -> int:
        """Blocks the plugin allocates: ``ceil((pool + max(block, 1) * max_batch) / block)`` = 4128
        (``vllm_tt_plugin/worker.py:get_num_available_blocks_tt`` with one output token per step)."""
        return math.ceil((self.kv_pool_tokens + self.kv_block_size * self.max_batch) / self.kv_block_size)

    @property
    def kv_pool_tokens_allocated(self) -> int:
        return self.kv_num_blocks * self.kv_block_size  # 264,192

    @property
    def kv_blocks_per_seq(self) -> int:
        """Page-table width W = max_model_len / block = 512."""
        return self.max_model_len // self.kv_block_size

    @property
    def kv_cache_shape(self) -> Tuple[int, int, int, int]:
        """Per-layer paged latent cache ``[num_blocks, 1, block, 576]`` (vLLM hint ``(N, 1, bs, 576)``)."""
        return (self.kv_num_blocks, 1, self.kv_block_size, self.kv_latent_dim)

    def kv_cache_bytes_per_chip(self) -> int:
        """All layers, replicated per chip; bfp8 = 1088 B per 1024 elements."""
        elems = self.num_layers * self.kv_num_blocks * self.kv_block_size * self.kv_latent_dim
        if self.dtypes.kv_cache == ttnn.bfloat8_b:
            return elems * 1088 // 1024
        if self.dtypes.kv_cache == ttnn.bfloat4_b:
            return elems * 576 // 1024
        return elems * 2

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
    # compute kernel configs and memory configs
    # ======================================================================================================
    def compute_config(self, role: str):
        """Compute-kernel config for an op class (see ``COMPUTE_ROLES``); math approx mode is off everywhere."""
        if role not in COMPUTE_ROLES:
            raise KeyError(f"unknown compute role {role!r}; known: {sorted(COMPUTE_ROLES)}")
        if role not in self._ckc:
            fid, acc = COMPUTE_ROLES[role]
            self._ckc[role] = make_compute_kernel_config(fid, acc)
        return self._ckc[role]

    @property
    def hifi4_fp32(self):
        return self.compute_config("router")

    @property
    def hifi2_fp32(self):
        return self.compute_config("experts")

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
        return device_params(self.fabric, self.trace_region_size, **extra)

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
        return (
            f"MotifTTConfig(mesh={a.mesh_shape} tp_axis={a.tp_axis} dp_axis={a.dp_axis} tp={a.tp_size} dp={a.dp_size}; "
            f"layers={self.num_layers}/{self.num_hidden_layers} global={len(self.global_layers)} "
            f"swa={len(self.swa_layers)} moe={len(self.moe_layers)}; heads/chip q={self.q_heads_per_chip} "
            f"kv={self.kv_groups_per_chip} sig={self.signal_heads_per_chip}; experts/chip={self.experts_per_chip}; "
            f"lanes/row={self.lanes_per_row}; kv blocks={self.kv_num_blocks}x{self.kv_block_size} "
            f"W={self.kv_blocks_per_seq}; buckets={self.prefill_buckets[0]}..{self.prefill_buckets[-1]}; "
            f"trace={self.trace_region_size}; cache={self.cache_dir})"
        )


__all__ = [
    "CACHE_FORMAT_VERSION",
    "COMPUTE_ROLES",
    "ChipHeads",
    "DtypePolicy",
    "LayerSpec",
    "MeshAxes",
    "MotifTTConfig",
    "device_params",
    "fabric_config_from_name",
    "make_compute_kernel_config",
    "resolve_hf_config_path",
    "resolve_tt_cache_root",
    "resolve_weights_dir",
]
