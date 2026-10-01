# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM bridge for Motif-3 on a Blackhole Galaxy: ``MotifForCausalLM`` (vllm-tt-plugin plain-class contract).

This module is imported by vLLM's API server, by its registry-inspection subprocess and by EngineCore before the
mesh is open (study 05 §2). Its import is device-free: it pulls in only the standard library, numpy, torch, loguru
and ``generator_api``. vLLM, ttnn and the TT runtime are imported lazily, inside the methods that need them.

Registration (design 00 §5.1; study 05 §3)
------------------------------------------
vLLM 0.26 resolves the checkpoint's *bare* architecture ``MotifForCausalLM`` in ``ModelConfig.__post_init__``, before
the plugin rewrites it to ``TTMotifForCausalLM``, and that bare name is in ``_PREVIOUSLY_SUPPORTED_MODELS``
(``vllm/model_executor/models/registry.py:725``). Unregistered, it silently resolves to the Transformers backend
(``TransformersMoEForCausalLM``) or raises. ``EXTRA_MODELS_DIR`` bundles register only the ``TT``-prefixed name, so
the bare name must come from ``TT_MODEL_CLASS_OVERRIDES``, which registers both names unconditionally
(``vllm_tt_plugin/platform.py:1212-1227``)::

    export TT_MODEL_CLASS_OVERRIDES="MotifForCausalLM=models.demos.motif3.tt.generator_vllm:MotifForCausalLM"
    export EXTRA_MODELS_DIR=<dir containing motif-3-bh-galaxy/vllm_metadata.json>   # bundle contract (optional)
    export PYTHONPATH=$TT_METAL_HOME                                                 # every process imports models.*
    export MESH_DEVICE="(4, 8)"  VLLM_ENGINE_READY_TIMEOUT_S=14400  HF_MODEL=<snapshot dir>  TT_CACHE_PATH=<cache>

Launch flags (design 00 §5.1): ``--trust-remote-code --max-num-seqs 32 --block-size 64 --max-model-len 32768
--no-enable-prefix-caching --additional-config '{"tt": {"trace_mode": "decode_only", "trace_region_size": 268435456,
"fabric_config": "FABRIC_2D_TORUS_XY", "dispatch_core_axis": "col"}}'``. Never ``--tensor-parallel-size`` or
``--disable-sliding-window``. Optional parsers: see ``models/demos/motif3/vllm_plugins``.

What the plugin calls, and what this class does
------------------------------------------------
* ``model_capabilities`` (class level, read before any instance exists): draft 1 declares no device sampling, no
  prefix caching, no chunked prefill, no async decode, no speculative decoding, and ``supports_device_penalties:
  False`` explicitly (the plugin's default for that key is True).
* ``get_max_tokens_all_users``: the usable KV pool (``MOTIF3_KV_POOL_TOKENS``, default 262,144) plus a 32-token
  reserve that makes the plugin allocate exactly one extra block for vLLM's null block, which upstream
  ``get_num_available_blocks_tt`` does not budget (``vllm_tt_plugin/worker.py:553-675``; ``block_pool.py:190``).
* ``get_kv_cache_spec``: one ``MLAAttentionSpec(num_kv_heads=1, head_size=576)`` per decoder layer, so the plugin's
  allocation hint is ``(num_blocks, 1, block_size, 576)`` instead of the default ``FullAttentionSpec(16, 192, sw=128)``.
* ``allocate_kv_cache[_per_layer]``: validates that hint and asks the generator for the latent pool.
* ``prefill_forward`` / ``decode_forward``: translate vLLM rows and state slots into Motif lanes (``LaneMap``), zero
  the stale block ids vLLM leaves past each request's blocks in reused block-table rows (a bucket-padded prefill would
  otherwise overwrite another live request's KV), call the generator, and return host logits ``[B, 1, vocab]``
  (draft 1 samples on the host).
* ``warmup_model_prefill`` / ``warmup_model_decode`` / ``release_request`` / ``release_persistent_capture``.

Decode-reload contract v1, partial adapter (``vllm-tt-plugin/docs/DECODE_RELOAD_CONTRACT.md``): every decode must
carry ``reload_inputs=True`` (always the case without ``supports_async_decode``); ``reload_inputs=False``,
``reload_page_table=True`` and the legacy ``reset_batch`` keyword raise.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from .generator_api import (
    KV_LATENT_DIM,
    KV_LORA_RANK,
    LANES_PER_GROUP,
    MAX_CONTEXT,
    MESH_SHAPES,
    NUM_HIDDEN_LAYERS,
    NUM_LANES,
    QK_ROPE_HEAD_DIM,
    SUPPORTED_BLOCK_SIZES,
    DecodeBatch,
    GeneratorSettings,
    MotifGenerator,
    PrefillRequest,
    cdiv,
    check_logits,
    kv_cache_bytes_per_chip,
    kv_cache_dtype_from_env,
)

ARCHITECTURE = "MotifForCausalLM"
MAIN_CLASS = "models.demos.motif3.tt.generator_vllm:MotifForCausalLM"
# The bare-name registration vLLM 0.26 needs (see the module docstring). Use this exact string.
TT_MODEL_CLASS_OVERRIDES = f"{ARCHITECTURE}={MAIN_CLASS}"
DEFAULT_GENERATOR_CLASS = "models.demos.motif3.tt.generator:MotifGenerator"

DEFAULT_KV_POOL_TOKENS = 262144  # usable pool = TIS max_tokens_all_users_override (design 00 §5.2)
KV_POOL_ALIGNMENT = 128  # pool multiple of the largest supported block -> the reserve below adds exactly 1 block
NULL_BLOCK_RESERVE_TOKENS = 32  # <= one block for every supported block size (32/64/128)
MAX_KV_POOL_TOKENS = 4 * 1024 * 1024
DEFAULT_KV_MAX_GB_PER_CHIP = 16.0  # 34.2 GB DRAM - 14.2 GB weights - ~4 GB activations/trace/CCL (design 00 §1.1)
REQUIRED_NUM_DEVICES = 32

# Keyword arguments the plugin can send that this draft-1 adapter cannot honour.
_UNSUPPORTED_KWARGS = {
    "page_tables_per_layer": "multi-group (hybrid) KV cache configs; Motif-3 declares one uniform MLA spec",
    "num_valid_drafts": "speculative decoding",
    "accepted_counts": "speculative decoding",
    "spec_mode": "speculative decoding",
    "rope_deltas_all_users": "M-RoPE",
    "prompt_tokens": "device-side penalties (no device sampling in draft 1)",
    "output_tokens": "device-side penalties (no device sampling in draft 1)",
}


# ----------------------------------------------------------------------------------------------------------------
# Pool / spec helpers (pure; unit-tested on the host)
# ----------------------------------------------------------------------------------------------------------------
def kv_pool_tokens_from_env(environ=None) -> int:
    """Usable KV pool in tokens: ``MOTIF3_KV_POOL_TOKENS`` (default 262,144), a multiple of 128."""
    env = os.environ if environ is None else environ
    raw = env.get("MOTIF3_KV_POOL_TOKENS")
    if raw is None or raw.strip() == "":
        return DEFAULT_KV_POOL_TOKENS
    raw = raw.strip()
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"MOTIF3_KV_POOL_TOKENS must be a positive decimal token count, got {raw!r}")
    tokens = int(raw)
    if tokens % KV_POOL_ALIGNMENT or not KV_POOL_ALIGNMENT <= tokens <= MAX_KV_POOL_TOKENS:
        raise ValueError(
            f"MOTIF3_KV_POOL_TOKENS={tokens} must be a multiple of {KV_POOL_ALIGNMENT} in "
            f"[{KV_POOL_ALIGNMENT}, {MAX_KV_POOL_TOKENS}]"
        )
    return tokens


def kv_max_bytes_per_chip(environ=None) -> int:
    """Per-chip KV budget used for fail-fast checks: ``MOTIF3_KV_MAX_GB_PER_CHIP`` (default 16 GB)."""
    env = os.environ if environ is None else environ
    raw = env.get("MOTIF3_KV_MAX_GB_PER_CHIP")
    gb = DEFAULT_KV_MAX_GB_PER_CHIP if raw is None or raw.strip() == "" else float(raw)
    if not gb > 0:
        raise ValueError(f"MOTIF3_KV_MAX_GB_PER_CHIP must be positive, got {raw!r}")
    return int(gb * 1e9)


def plugin_num_blocks(max_tokens_all_users: int, block_size: int, max_num_seqs: int) -> int:
    """The block count vllm-tt-plugin allocates (``worker.get_num_available_blocks_tt``, AR model, no hybrid
    headroom): ``ceil((max_tokens_all_users + block_size * max_num_seqs) / block_size)``; vLLM then keeps block 0."""
    return cdiv(int(max_tokens_all_users) + int(block_size) * int(max_num_seqs), int(block_size))


def validate_block_size(block_size: int) -> int:
    block_size = int(block_size)
    if block_size not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            f"Motif-3 needs --block-size in {SUPPORTED_BLOCK_SIZES} (a multiple of the 32-row tile the paged latent "
            f"ops support; 64 is the tested default), got {block_size}. vLLM's default of 16 is not usable on TT."
        )
    return block_size


def _current_vllm_config():
    """The VllmConfig vLLM set for the running phase (EngineCore runs ``init_device`` inside
    ``set_current_vllm_config``), or None. Never imports vLLM itself: outside vLLM there is no config to find."""
    vllm_config_module = sys.modules.get("vllm.config")
    getter = getattr(vllm_config_module, "get_current_vllm_config_or_none", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:  # pragma: no cover - defensive: sizing must not fail on a lookup
        return None


def _validate_hf_config(hf_config: Any) -> None:
    """Duck-typed check that this is a Motif-3-shaped config (the dynamic MotifConfig class is never isinstance'd)."""
    kv_lora = int(getattr(hf_config, "kv_lora_rank", KV_LORA_RANK))
    rope = int(getattr(hf_config, "qk_rope_head_dim", QK_ROPE_HEAD_DIM))
    if kv_lora + rope != KV_LATENT_DIM:
        raise ValueError(
            f"Motif-3 latent KV is kv_lora_rank + qk_rope_head_dim = {KV_LATENT_DIM}; this config gives "
            f"{kv_lora} + {rope}"
        )
    model_type = getattr(hf_config, "model_type", None)
    if model_type not in (None, "Motif", "motif"):
        raise ValueError(f"MotifForCausalLM got a {model_type!r} config")
    if int(getattr(hf_config, "num_hidden_layers", NUM_HIDDEN_LAYERS)) < 1:
        raise ValueError("config has no decoder layers")


def _validate_mesh(mesh_device: Any) -> Tuple[int, int]:
    try:
        shape = tuple(int(s) for s in mesh_device.shape)
    except Exception as exc:  # pragma: no cover - not a mesh
        raise TypeError(f"initialize_vllm_model needs the plugin's ttnn.MeshDevice, got {type(mesh_device)}") from exc
    if shape not in MESH_SHAPES:
        raise ValueError(
            f"Motif-3 runs on the whole 32-chip BH Galaxy as a {MESH_SHAPES[0]} (or {MESH_SHAPES[1]}) mesh; the "
            f'plugin opened {shape}. Set MESH_DEVICE="(4, 8)".'
        )
    return shape


def _resolve_generator_class() -> type:
    path = os.environ.get("MOTIF3_GENERATOR_CLASS") or DEFAULT_GENERATOR_CLASS
    module_name, sep, class_name = path.partition(":")
    if not sep or not module_name or not class_name:
        raise ValueError(f"MOTIF3_GENERATOR_CLASS must be 'module.path:ClassName', got {path!r}")
    impl = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(impl, type) and issubclass(impl, MotifGenerator)):
        raise TypeError(f"{path} must be a subclass of models.demos.motif3.tt.generator_api.MotifGenerator")
    return impl


def _as_int_list(values: Any) -> List[int]:
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.reshape(-1).tolist()]
    return [int(v) for v in np.asarray(values).reshape(-1).tolist()]


# ----------------------------------------------------------------------------------------------------------------
# State slot -> lane bookkeeping
# ----------------------------------------------------------------------------------------------------------------
class LaneMap:
    """vLLM state slot -> Motif decode lane, kept in step with the plugin's ``slot_remap``.

    The plugin keeps each request's device state in a *state slot* (``vllm_tt_plugin/model_runner.py:1093-1238``):
    prefill row ``i`` initialises slot ``empty_slots[i]``; decode row ``i`` reads slot ``slot_remap[i]`` (identity
    when None); after the decode is accepted, slot ``i`` holds what slot ``slot_remap[i]`` held. Motif's per-request
    device state is its lane (decode writes a lane's KV only on its DP group's chips), so this keeps
    ``slot_to_lane``, an injective map onto lanes, and applies each accepted remap exactly once
    (``docs/DECODE_RELOAD_CONTRACT.md:43-93``). KV data never moves; lanes are stable for the life of a request.

    The initial map deals slots round-robin over the 4 DP groups (slot 0 -> lane 0, 1 -> 8, 2 -> 16, 3 -> 24,
    4 -> 1, ...): vLLM fills low slots first, so a lightly loaded server spreads its requests over the groups and
    the slowest group's attention work shrinks. With 32 slots the map is a permutation of all 32 lanes.
    """

    def __init__(self, num_slots: int, num_lanes: int = NUM_LANES, lanes_per_group: int = LANES_PER_GROUP):
        if not 1 <= num_slots <= num_lanes or num_lanes % lanes_per_group:
            raise ValueError(f"{num_slots} slots cannot map onto {num_lanes} lanes")
        groups = num_lanes // lanes_per_group
        self.num_slots = int(num_slots)
        self.num_lanes = int(num_lanes)
        self.lanes_per_group = int(lanes_per_group)
        self._slot_to_lane = [(s % groups) * lanes_per_group + s // groups for s in range(num_slots)]

    @property
    def slot_to_lane(self) -> Tuple[int, ...]:
        return tuple(self._slot_to_lane)

    def lane_of_slot(self, slot: int) -> int:
        slot = int(slot)
        if not 0 <= slot < self.num_slots:
            raise ValueError(f"state slot {slot} outside [0, {self.num_slots})")
        return self._slot_to_lane[slot]

    def group_of_slot(self, slot: int) -> int:
        return self.lane_of_slot(slot) // self.lanes_per_group

    def _parse_remap(self, slot_remap: Any) -> Optional[List[int]]:
        if slot_remap is None:
            return None
        remap = _as_int_list(slot_remap)
        if len(remap) != self.num_slots or sorted(remap) != list(range(self.num_slots)):
            raise ValueError(f"slot_remap must be a permutation of the {self.num_slots} state slots, got {remap}")
        return remap

    def decode_lanes(self, num_rows: int, slot_remap: Any = None) -> List[int]:
        """Lane read by each decode row: row ``i`` reads slot ``slot_remap[i]`` (does not commit)."""
        if not 1 <= num_rows <= self.num_slots:
            raise ValueError(f"decode sent {num_rows} rows for {self.num_slots} state slots")
        remap = self._parse_remap(slot_remap)
        slots = range(num_rows) if remap is None else remap[:num_rows]
        return [self._slot_to_lane[s] for s in slots]

    def commit(self, slot_remap: Any) -> None:
        """Apply an accepted decode's remap: slot ``i`` now holds the state slot ``slot_remap[i]`` held."""
        remap = self._parse_remap(slot_remap)
        if remap is not None:
            self._slot_to_lane = [self._slot_to_lane[remap[i]] for i in range(self.num_slots)]


# ----------------------------------------------------------------------------------------------------------------
# KV handle
# ----------------------------------------------------------------------------------------------------------------
@dataclass(eq=False)  # identity semantics (and hashable): the plugin hands back this exact object
class MotifKVCache:
    """What ``allocate_kv_cache`` returns to vLLM (opaque to the plugin, passed back as ``kv_cache=``)."""

    num_blocks: int
    block_size: int
    num_layers: int  # caches the generator allocated (== generator.num_layers)
    vllm_num_layers: int  # layers vLLM accounts for (53, or more than num_layers in a truncated run)
    kv_cache_dtype: str  # device dtype: "bfp8" | "bf16"
    vllm_dtype: Any  # torch dtype vLLM accounted with (bookkeeping only)
    page_table_width: int  # W of every page table sent to the generator
    bytes_per_chip: int
    device_cache: Any  # the generator's handle

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.num_blocks, 1, self.block_size, KV_LATENT_DIM)


# ----------------------------------------------------------------------------------------------------------------
# The vLLM model class
# ----------------------------------------------------------------------------------------------------------------
class MotifForCausalLM:
    """Motif-3 (314B MoE, GDLA latent attention, mHC) served from a 32-chip BH Galaxy through vllm-tt-plugin.

    Plain class (no ``nn.Module``): vLLM only inspects it and calls the TT-plugin hooks below. Do not add class
    attributes that vLLM reads with ``getattr`` (``is_hybrid``, ``has_inner_state``, ``is_attention_free``,
    ``supports_multimodal``, ``supports_pp``, ``is_pooling_model``, ``has_noops``, ``attn_type``, ...), nor
    ``_HYBRID_KV_CACHE_GROUPS_ENABLED``, ``tt_supported_decode_batch_sizes``, ``already_warmed_up_prefill`` or
    ``note_state_slots_moved`` (slot moves are applied from ``slot_remap`` inside ``decode_forward``; the hook would
    apply them a second time).
    """

    # Explicit commands (reload_inputs / reload_page_table / reload_sampling_params / reset_sampling_state).
    # Partial v1 adapter: async decode stays off, so the plugin sends reload_inputs=True on every decode.
    decode_input_update_contract = 1

    # Read by the plugin from the CLASS at config time (platform.py:1815-1822), before any instance exists.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,  # v1: True + "max_device_top_k": 32
        "supports_device_penalties": False,  # the plugin's default for an absent key is True
        "supports_spec_decode": False,  # MTP self-speculation is v1+
        "supports_async_spec_decode": False,
        "output_tokens_per_step": 1,
    }

    def __init__(
        self,
        generator: Optional[MotifGenerator] = None,
        settings: Optional[GeneratorSettings] = None,
        *,
        vllm_config: Any = None,
        prefix: str = "",
        **kwargs: Any,
    ):
        # ``vllm_config`` keeps vLLM's protocol check happy (interfaces_base._check_vllm_model_init); the TT plugin
        # never constructs the class that way.
        if generator is None:
            raise TypeError(
                "MotifForCausalLM is built by MotifForCausalLM.initialize_vllm_model() under vllm-tt-plugin "
                "(or MotifForCausalLM(generator, settings) in tests); it has no GPU/vLLM-native constructor"
            )
        if not isinstance(generator, MotifGenerator):
            raise TypeError(f"generator must implement generator_api.MotifGenerator, got {type(generator)}")
        self.generator = generator
        self.settings = settings if settings is not None else GeneratorSettings(num_layers=generator.num_layers)
        if generator.num_lanes != NUM_LANES:
            raise ValueError(f"generator runs {generator.num_lanes} lanes; the bridge needs {NUM_LANES}")
        if generator.num_layers != self.settings.num_layers:
            raise ValueError(
                f"generator runs {generator.num_layers} layers but settings.num_layers={self.settings.num_layers}"
            )
        self.vocab_size = int(generator.vocab_size)
        self._lanes = LaneMap(self.settings.max_batch_size)
        self._kv: Optional[MotifKVCache] = None
        self._prefill_warmed = False  # every bucket compiled (required before decode trace capture)
        self._warned: set = set()

    # ---- vLLM model-inspection protocol (registry._ModelInfo); never executed on TT ---------------------------
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("Motif-3 on TT runs through prefill_forward/decode_forward (vllm-tt-plugin)")

    def forward(self, input_ids, positions, **kwargs):
        raise NotImplementedError("Motif-3 on TT runs through prefill_forward/decode_forward (vllm-tt-plugin)")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("the TT generator owns the LM head; logits come back from prefill/decode_forward")

    def modules(self):
        """No torch submodules. vLLM's in-process ``LLMEngine`` finalizer (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) walks
        ``model.modules()`` to drop torch.compile hooks (``vllm/v1/engine/llm_engine.py:437-443``)."""
        return iter(())

    # ---- construction ---------------------------------------------------------------------------------------------
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        """Called once in EngineCore by ``vllm_tt_plugin/loader.py:38-45`` after the plugin opened the mesh.

        Opens nothing itself. ``hf_config`` is vLLM's (trust-remote-code) ``MotifConfig``; ``mesh_device`` the
        plugin's ``ttnn.MeshDevice``; ``max_batch_size`` = ``max_num_seqs``; ``max_seq_len`` = ``max_model_len``.
        Weights: ``HF_MODEL`` (dir or repo id + ``TT_MODEL_WEIGHTS_REVISION``), else ``hf_config._name_or_path``;
        TT cache: ``TT_CACHE_PATH``. The runtime class is ``MOTIF3_GENERATOR_CLASS`` (default
        ``models.demos.motif3.tt.generator:MotifGenerator``), imported here, never at module import time.
        """
        if int(tt_data_parallel) != 1:
            raise ValueError(
                f"Motif-3 runs as one vLLM engine (DP=1; MoE + standard DP is refused by the plugin), got "
                f"tt_data_parallel={tt_data_parallel}"
            )
        _validate_mesh(mesh_device)
        _validate_hf_config(hf_config)
        settings = GeneratorSettings.from_env(
            hf_config, max_batch_size=max_batch_size, max_seq_len=max_seq_len, optimizations=optimizations
        )
        impl = _resolve_generator_class()
        logger.info(
            "Motif-3 vLLM bridge: generator={}.{} layers={}/{} max_batch={} max_seq_len={} kv_dtype={} weights={}",
            impl.__module__,
            impl.__name__,
            settings.num_layers,
            getattr(hf_config, "num_hidden_layers", "?"),
            settings.max_batch_size,
            settings.max_seq_len,
            settings.kv_cache_dtype,
            settings.weights_path,
        )
        generator = impl.create(hf_config=hf_config, mesh_device=mesh_device, settings=settings)
        if int(generator.vocab_size) != int(getattr(hf_config, "vocab_size", generator.vocab_size)):
            raise ValueError(f"generator vocab {generator.vocab_size} != config vocab {hf_config.vocab_size}")
        return cls(generator, settings)

    # ---- KV pool sizing (runs in init_device, before the weights load) ------------------------------------------
    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = REQUIRED_NUM_DEVICES,
        tt_data_parallel: int = 1,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        **kwargs,
    ) -> int:
        """KV pool in tokens for ``worker.get_num_available_blocks_tt`` (called before the model loads).

        Returns ``MOTIF3_KV_POOL_TOKENS`` (default 262,144, which must equal TIS ``max_tokens_all_users_override``)
        plus ``NULL_BLOCK_RESERVE_TOKENS``. The plugin adds ``block_size * max_num_seqs`` and rounds up to whole
        blocks; the 32-token reserve makes that exactly one extra block for every supported block size, which vLLM's
        ``BlockPool`` then takes as its null block. Net: 32 users x (pool/32 tokens + one output block) fit exactly.
        Raises early (before an hour of weight loading) on configurations draft 1 cannot serve.
        """
        if int(tt_data_parallel) != 1:
            raise ValueError(f"Motif-3 needs tt_data_parallel=1 (one engine over the mesh), got {tt_data_parallel}")
        if int(num_devices) != REQUIRED_NUM_DEVICES:
            raise ValueError(
                f"Motif-3 draft 1 needs the whole {REQUIRED_NUM_DEVICES}-chip BH Galaxy, the plugin sees {num_devices}"
            )
        if max_num_seqs is not None and not 1 <= int(max_num_seqs) <= NUM_LANES:
            raise ValueError(f"--max-num-seqs must be in [1, {NUM_LANES}] for Motif-3, got {max_num_seqs}")
        pool = kv_pool_tokens_from_env()
        if max_model_len is not None:
            if int(max_model_len) > MAX_CONTEXT:
                raise ValueError(
                    f"max_model_len={max_model_len} exceeds the draft-1 context of {MAX_CONTEXT} (largest prefill "
                    f"bucket); pass --max-model-len {MAX_CONTEXT} or less"
                )
            if int(max_model_len) > pool:
                raise ValueError(f"max_model_len={max_model_len} does not fit the {pool}-token KV pool")

        # Fail-fast memory check with the real block size when vLLM exposes the config here (it does inside
        # EngineCore's init_device); otherwise use the largest supported block (an upper bound).
        block_size = max(SUPPORTED_BLOCK_SIZES)
        num_layers = NUM_HIDDEN_LAYERS
        vllm_config = _current_vllm_config()
        if vllm_config is not None and getattr(vllm_config, "cache_config", None) is not None:
            block_size = validate_block_size(vllm_config.cache_config.block_size)
            try:
                num_layers = int(vllm_config.model_config.hf_text_config.num_hidden_layers)
            except Exception:  # pragma: no cover - partial configs
                pass
        env_layers = os.environ.get("MOTIF3_NUM_LAYERS", "").strip()
        if env_layers.isdecimal() and int(env_layers) > 0:
            num_layers = min(num_layers, int(env_layers))
        kv_dtype = kv_cache_dtype_from_env()
        tokens = pool + NULL_BLOCK_RESERVE_TOKENS
        blocks = plugin_num_blocks(tokens, block_size, int(max_num_seqs or NUM_LANES))
        need = kv_cache_bytes_per_chip(blocks, block_size, num_layers, kv_dtype)
        cap = kv_max_bytes_per_chip()
        if need > cap:
            raise ValueError(
                f"a {pool}-token {kv_dtype} latent pool ({blocks} blocks of {block_size}, {num_layers} layers) needs "
                f"{need / 1e9:.2f} GB per chip, over the {cap / 1e9:.2f} GB KV budget (MOTIF3_KV_MAX_GB_PER_CHIP); "
                f"lower MOTIF3_KV_POOL_TOKENS"
            )
        return tokens

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Uniform MLA latent spec, one entry per decoder layer (``vllm_tt_plugin/worker.py:293-371``).

        ``{"model.layers.{i}.self_attn": MLAAttentionSpec(block_size, num_kv_heads=1, head_size=576, dtype)}`` for
        ``i < model_config.get_num_layers_by_block_type(parallel_config, "attention")`` (53). The plugin turns it into
        the per-layer allocation hint ``(num_blocks, 1, block_size, 576)``. vLLM's dtype is bookkeeping only (the
        block count is fixed by ``get_max_tokens_all_users``); the device dtype is ``MOTIF3_KV_CACHE_DTYPE``. The
        39 sliding-window layers are paged like the global ones (window 129 is applied in the kernels), as every TT
        hybrid model does today (study 05 §7.3 option A).
        """
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        block_size = validate_block_size(cache_config.block_size)
        hf = getattr(model_config, "hf_text_config", None) or model_config.hf_config
        _validate_hf_config(hf)
        num_layers = int(model_config.get_num_layers_by_block_type(vllm_config.parallel_config, "attention"))
        dtype = model_config.dtype
        cache_dtype = getattr(cache_config, "cache_dtype", "auto") or "auto"
        if cache_dtype != "auto":
            try:
                from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE

                dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
            except Exception:
                pass
            logger.warning(
                "Motif-3: --kv-cache-dtype {} only changes vLLM's bookkeeping; the device latent cache dtype is "
                "MOTIF3_KV_CACHE_DTYPE={}",
                cache_dtype,
                kv_cache_dtype_from_env(),
            )
        return {
            f"model.layers.{i}.self_attn": MLAAttentionSpec(
                block_size=block_size, num_kv_heads=1, head_size=KV_LATENT_DIM, dtype=dtype
            )
            for i in range(num_layers)
        }

    # ---- KV allocation ----------------------------------------------------------------------------------------
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Legacy uniform allocation hook (``model_runner.py:678-708``).

        Expects ``kv_cache_shape == (num_blocks, 1, block_size, 576)`` (the hint from ``get_kv_cache_spec``: heads
        ``1 // min(32, 1) = 1``) and ``num_layers`` = vLLM's attention-layer count (53). Allocates
        ``generator.num_layers`` caches (fewer than vLLM counts only in a ``MOTIF3_NUM_LAYERS`` truncated run) and
        returns a ``MotifKVCache``.
        """
        shape = tuple(int(s) for s in kv_cache_shape)
        if len(shape) != 4 or shape[1] != 1 or shape[3] != KV_LATENT_DIM:
            raise ValueError(
                f"Motif-3 expects the latent KV hint (num_blocks, 1, block_size, {KV_LATENT_DIM}) from its "
                f"get_kv_cache_spec hook, got {shape} (is the plugin using the default FullAttentionSpec?)"
            )
        num_blocks, _, block_size, _ = shape
        validate_block_size(block_size)
        if num_blocks < 2:
            raise ValueError(f"KV pool of {num_blocks} blocks cannot hold the null block plus one request")
        if self._kv is not None:
            raise RuntimeError("allocate_kv_cache called twice")
        vllm_layers = int(num_layers)
        layers = self.generator.num_layers
        if vllm_layers < layers:
            raise ValueError(f"vLLM accounts for {vllm_layers} attention layers but the generator runs {layers}")
        if vllm_layers > layers:
            logger.info("Motif-3 truncated run: allocating {} of vLLM's {} layer caches", layers, vllm_layers)
        kv_dtype = self.settings.kv_cache_dtype
        need = kv_cache_bytes_per_chip(num_blocks, block_size, layers, kv_dtype)
        cap = kv_max_bytes_per_chip()
        if need > cap:
            raise ValueError(
                f"KV pool {shape} x {layers} layers ({kv_dtype}) needs {need / 1e9:.2f} GB per chip, over the "
                f"{cap / 1e9:.2f} GB budget (MOTIF3_KV_MAX_GB_PER_CHIP)"
            )
        width = min(cdiv(self.settings.max_seq_len, block_size), num_blocks)
        handle = self.generator.allocate_kv_cache(num_blocks=num_blocks, block_size=block_size, num_layers=layers)
        self._kv = MotifKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=layers,
            vllm_num_layers=vllm_layers,
            kv_cache_dtype=kv_dtype,
            vllm_dtype=dtype,
            page_table_width=width,
            bytes_per_chip=need,
            device_cache=handle,
        )
        logger.info(
            "Motif-3 KV pool: {} blocks x {} tokens ({} usable after vLLM's null block) x {} layers, {} on device "
            "(vLLM accounts {}), {:.2f} GB per chip, page-table width {}",
            num_blocks,
            block_size,
            (num_blocks - 1) * block_size,
            layers,
            kv_dtype,
            dtype,
            need / 1e9,
            width,
        )
        return self._kv

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Per-layer hook (preferred by the plugin when present): ``[(shape, dtype, tensor_idx)]`` in layer order.

        Motif's spec is uniform, so every entry must carry the same shape and dtype and its own buffer
        (``tensor_idx == layer index``); then this is ``allocate_kv_cache(shape, dtype, len(per_layer_specs))``.
        """
        specs = list(per_layer_specs)
        if not specs:
            raise ValueError("no KV layer specs")
        shape0, dtype0, _ = specs[0]
        for i, (shape, dtype, tensor_idx) in enumerate(specs):
            if tuple(shape) != tuple(shape0) or dtype != dtype0:
                raise ValueError(f"Motif-3 KV specs must be uniform; layer {i} has {tuple(shape)} {dtype}")
            if int(tensor_idx) != i:
                raise ValueError(f"Motif-3 does not share KV buffers between layers (layer {i} -> {tensor_idx})")
        return self.allocate_kv_cache(tuple(shape0), dtype0, len(specs))

    def _check_kv(self, kv_cache) -> MotifKVCache:
        if self._kv is None:
            raise RuntimeError("the KV pool is not allocated (allocate_kv_cache has not run)")
        if kv_cache is not self._kv:
            raise ValueError("kv_cache must be the object allocate_kv_cache returned")
        return self._kv

    def _fit_page_table(self, page_table, kv: MotifKVCache, valid_blocks: torch.Tensor, where: str) -> torch.Tensor:
        """``torch.int32 [rows, W]`` (W = ``kv.page_table_width``): row ``r`` keeps its first ``valid_blocks[r]``
        vLLM block ids and is zero (null block) after them.

        The zeroing is load-bearing. vLLM's persistent block table is not cleared when a row is reused, so the
        entries past a request's own blocks can hold stale ids of blocks that now belong to OTHER live requests
        (observed on vLLM 0.26 + vllm-tt-plugin dd287f9 with 12 requests on 8 slots). A bucket-padded prefill
        writing positions ``S .. bucket-1`` through such an entry would corrupt another request's KV.
        """
        pt = torch.as_tensor(page_table)
        if pt.ndim != 2 or pt.shape[0] != valid_blocks.shape[0]:
            raise ValueError(f"{where}: page_table must be [{valid_blocks.shape[0]}, blocks], got {tuple(pt.shape)}")
        width = kv.page_table_width
        need = valid_blocks.to(torch.int64)
        if bool((need > min(width, pt.shape[1])).any()):
            raise ValueError(
                f"{where}: a row needs {int(need.max())} blocks; the block table has {pt.shape[1]} columns and the "
                f"context window {width}"
            )
        pt = pt[:, :width].to(torch.int32)
        if pt.shape[1] < width:
            pt = torch.nn.functional.pad(pt, (0, width - pt.shape[1]))
        valid = torch.arange(width)[None, :] < need[:, None]
        real = pt[valid]
        if bool((real < 1).any()):
            raise ValueError(f"{where}: a position this step needs is on the null block (block id 0)")
        if bool((real >= kv.num_blocks).any()):
            raise ValueError(f"{where}: page_table has block ids outside [1, {kv.num_blocks})")
        return torch.where(valid, pt, torch.zeros_like(pt)).contiguous()

    def _reject_unsupported(self, kwargs: dict, where: str) -> None:
        if "reset_batch" in kwargs:
            raise TypeError(
                "reset_batch is the legacy (v0) decode reload keyword; MotifForCausalLM declares "
                "decode_input_update_contract = 1 and needs a plugin that sends the explicit reload commands"
            )
        for key, value in kwargs.items():
            if key in _UNSUPPORTED_KWARGS:
                if value is not None:
                    raise NotImplementedError(f"{where}: {key} ({_UNSUPPORTED_KWARGS[key]}) is not supported")
            elif value is not None and key not in self._warned:
                self._warned.add(key)
                logger.warning("Motif-3 {}: ignoring unexpected keyword {!r}", where, key)

    # ---- prefill --------------------------------------------------------------------------------------------------
    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache,
        prompt_lens,
        start_pos=None,
        enable_trace=False,
        sampling_params=None,
        empty_slots=None,
        **kwargs,
    ):
        """Prefill a batch of new or resumed requests, one at a time (``model_runner.py:3059-3123``).

        Args (all keyword, as the plugin sends them):
            tokens: ``torch.int32 [B, max(prompt_lens)]``; row ``i`` is valid up to ``prompt_lens[i]`` (stale after).
            page_table: ``torch.int32 [B, W]`` vLLM block table rows. Entries past ``ceil(prompt_lens[i] / bs)`` can
                be stale ids of other requests' blocks (rows are reused uncleared); they are zeroed before the generator
                sees them, so bucket-padding writes go to null block 0.
            kv_cache: the ``MotifKVCache`` from ``allocate_kv_cache``.
            prompt_lens: numpy int64 ``[B]``: end of the chunk = full length (prompt + generated tokens for a
                resumed request); chunked prefill is disabled.
            start_pos: numpy int32 ``[B]``: tokens already computed; always 0 here (no prefix caching / chunking).
            enable_trace: plugin ``trace_mode == "all"``; passed through (draft-1 prefill is eager).
            sampling_params: only with device sampling (never declared) -> raises if given.
            empty_slots: ``list[int]`` destination state slot per row (always sent outside lane mode).

        Returns:
            Host logits ``[B, 1, vocab]`` (float32 or bfloat16) for each row's last token; the plugin's host sampler
            reads ``[rows, -1, :]``.
        """
        kv = self._check_kv(kv_cache)
        self._reject_unsupported(kwargs, "prefill_forward")
        if sampling_params is not None:
            raise NotImplementedError("Motif-3 draft 1 samples on the host; sampling_params implies device sampling")
        ends = np.asarray(prompt_lens, dtype=np.int64).reshape(-1)
        rows = int(ends.shape[0])
        if rows < 1:
            raise ValueError("prefill_forward got no rows")
        starts = np.zeros(rows, dtype=np.int64) if start_pos is None else np.asarray(start_pos).astype(np.int64)
        starts = starts.reshape(-1)
        tokens_t = torch.as_tensor(tokens)
        if tokens_t.ndim != 2 or tokens_t.shape[0] < rows or starts.shape[0] != rows:
            raise ValueError(
                f"prefill shapes disagree: tokens {tuple(tokens_t.shape)}, prompt_lens {rows}, start_pos {starts.shape}"
            )
        if empty_slots is None:
            if "empty_slots" not in self._warned:
                self._warned.add("empty_slots")
                logger.warning("Motif-3 prefill_forward: no empty_slots from the plugin; using rows as state slots")
            slots = list(range(rows))
        else:
            slots = [int(s) for s in empty_slots]
        if len(slots) != rows or len(set(slots)) != rows:
            raise ValueError(f"empty_slots {slots} must name {rows} distinct state slots")
        lanes = [self._lanes.lane_of_slot(s) for s in slots]
        max_len = min(self.settings.max_seq_len, int(self.generator.max_prefill_len))
        for i in range(rows):
            start, end = int(starts[i]), int(ends[i])
            if start != 0:
                raise NotImplementedError(
                    f"row {i}: start_pos={start}; Motif-3 draft 1 has no prefix caching or chunked prefill"
                )
            if not 1 <= end <= min(int(tokens_t.shape[1]), max_len):
                raise ValueError(f"row {i}: prompt length {end} outside [1, {min(int(tokens_t.shape[1]), max_len)}]")
        table = torch.as_tensor(page_table)
        if table.ndim != 2 or table.shape[0] < rows:
            raise ValueError(f"prefill page_table {tuple(table.shape)} has fewer rows than the {rows} prompts")
        need = torch.as_tensor([cdiv(int(e), kv.block_size) for e in ends], dtype=torch.int64)
        pt = self._fit_page_table(table[:rows], kv, need, "prefill_forward")
        requests = [
            PrefillRequest(
                lane=lanes[i],
                tokens=tokens_t[i, : int(ends[i])].to(torch.int32).contiguous(),
                page_table=pt[i].clone(),
            )
            for i in range(rows)
        ]
        outputs = []
        for req in requests:
            logits = self.generator.prefill_forward(req, kv_cache=kv.device_cache, enable_trace=bool(enable_trace))
            outputs.append(check_logits("MotifGenerator.prefill_forward", logits, (self.vocab_size,)))
        return torch.stack(outputs).unsqueeze(1)

    # ---- decode ---------------------------------------------------------------------------------------------------
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        slot_remap=None,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        **kwargs,
    ):
        """One decode step (``async_decode.py:1144-1292``).

        Args (keyword, as the plugin sends them):
            tokens: ``torch.int32 [B, 1]`` (B = ``max_num_seqs``, front-packed; padding rows token 0).
            start_pos: ``torch.int32 [B]`` position of each input token = KV write slot; padding rows ``-1``.
            page_table: ``torch.int32 [B, W]`` (padding rows 0; entries past ``start_pos // bs`` are zeroed, as above).
            kv_cache: the ``MotifKVCache``.
            enable_trace: plugin ``trace_mode in ("all", "decode_only")``.
            read_from_device: ignored; the result is always a host tensor (the plugin then skips its read hooks).
            sampling_params: device sampling only -> raises if given.
            slot_remap: ``torch.int32 [max_num_seqs]`` or None: row ``i`` reads state slot ``slot_remap[i]``. Applied
                to the lane map exactly once, after the generator accepted the step.
            reload_inputs / reload_page_table / reload_sampling_params / reset_sampling_state: contract-v1 commands.
                Without async decode the plugin always sends ``reload_inputs=True`` and False for the rest; there is
                no device sampler state to reload or reset.

        Returns:
            Host logits ``[B, 1, vocab]`` in row order (rows of padding are don't-care).
        """
        kv = self._check_kv(kv_cache)
        self._reject_unsupported(kwargs, "decode_forward")
        if sampling_params is not None:
            raise NotImplementedError("Motif-3 draft 1 samples on the host; sampling_params implies device sampling")
        if not reload_inputs:
            raise NotImplementedError(
                "MotifForCausalLM is a partial decode-reload v1 adapter: every decode must reload its inputs "
                "(supports_async_decode is False)"
            )
        if reload_page_table:
            raise ValueError("reload_page_table is only legal with reload_inputs=False (plugin contract)")
        tok = torch.as_tensor(tokens)
        if tok.ndim == 2:
            if tok.shape[1] != 1:
                raise NotImplementedError(f"decode tokens {tuple(tok.shape)}: one token per row (no speculation)")
            tok = tok[:, 0]
        if tok.ndim != 1:
            raise ValueError(f"decode tokens must be [B, 1], got {tuple(torch.as_tensor(tokens).shape)}")
        pos = torch.as_tensor(start_pos).reshape(-1).to(torch.int32)
        rows = int(tok.shape[0])
        if pos.shape[0] != rows:
            raise ValueError(f"decode shapes disagree: tokens {rows}, start_pos {pos.shape[0]}")
        active = pos >= 0
        if bool((pos < -1).any()) or bool((pos[active] >= self.settings.max_seq_len).any()):
            raise ValueError(f"decode positions must be -1 or in [0, {self.settings.max_seq_len})")
        need = torch.where(active, pos.to(torch.int64) // kv.block_size + 1, torch.zeros_like(pos, dtype=torch.int64))
        pt = self._fit_page_table(page_table, kv, need, "decode_forward")  # inactive rows come back all zero
        lanes = self._lanes.decode_lanes(rows, slot_remap)
        lane_idx = torch.tensor(lanes, dtype=torch.long)
        lane_tokens = torch.zeros(NUM_LANES, dtype=torch.int32)
        lane_pos = torch.full((NUM_LANES,), -1, dtype=torch.int32)
        lane_pt = torch.zeros((NUM_LANES, pt.shape[1]), dtype=torch.int32)
        lane_tokens[lane_idx] = torch.where(active, tok.to(torch.int32), torch.zeros_like(pos))
        lane_pos[lane_idx] = pos
        lane_pt[lane_idx] = pt
        batch = DecodeBatch(tokens=lane_tokens, positions=lane_pos, page_table=lane_pt)
        logits = self.generator.decode_forward(batch, kv_cache=kv.device_cache, enable_trace=bool(enable_trace))
        check_logits("MotifGenerator.decode_forward", logits, (NUM_LANES, self.vocab_size))
        # Accepted: commit the slot move exactly once (the plugin settles its own map right after we return).
        self._lanes.commit(slot_remap)
        if lanes == list(range(rows)):
            out = logits[:rows]
        else:
            out = logits.index_select(0, lane_idx)
        return out.unsqueeze(1)

    def read_decode_output(self, tt_out, async_read=False):
        """``decode_forward`` already returns host logits; nothing is outstanding on the device."""
        if not isinstance(tt_out, torch.Tensor):
            raise TypeError(f"expected the host logits decode_forward returned, got {type(tt_out)}")
        return (tt_out, []) if async_read else tt_out

    def process_decode_output_host(self, tt_out, is_tokens=False):
        if is_tokens:
            raise NotImplementedError("Motif-3 draft 1 has no device sampling; decode returns logits")
        if not isinstance(tt_out, torch.Tensor):
            raise TypeError(f"expected host logits, got {type(tt_out)}")
        return tt_out

    # ---- warmup ---------------------------------------------------------------------------------------------------
    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device=False, **kwargs):
        """Plugin phase 1 (eager) and, only with ``trace_mode="all"``, phase 2 (``model_runner.py:3727-3781``).

        Compiles every prefill bucket before any decode trace exists (a prefill shape compiled after capture can
        corrupt the trace).
        """
        kv = self._check_kv(kv_cache)
        if can_sample_on_device:
            raise ValueError("Motif-3 draft 1 does not sample on device (unset sample_on_device_mode)")
        self.generator.warmup_prefill(kv_cache=kv.device_cache, enable_trace=bool(enable_trace))
        self._prefill_warmed = True

    def warmup_model_decode(
        self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device=False, **kwargs
    ):
        """Eager decode warmup, then decode trace capture (``enable_trace=True``).

        ``num_blocks`` is the plugin's page-table width (``max_num_blocks_per_req``), fixed for the server's life;
        ``max_batch_size`` is ``max_num_seqs``.
        """
        kv = self._check_kv(kv_cache)
        if can_sample_on_device:
            raise ValueError("Motif-3 draft 1 does not sample on device (unset sample_on_device_mode)")
        if not 1 <= int(max_batch_size) <= self._lanes.num_slots:
            raise ValueError(f"decode warmup for {max_batch_size} rows, the bridge has {self._lanes.num_slots} slots")
        width = int(num_blocks)
        if width < 1 or width > kv.num_blocks:
            raise ValueError(f"page-table width {width} outside [1, {kv.num_blocks}]")
        if width != kv.page_table_width:
            logger.warning(
                "Motif-3: plugin page-table width {} differs from the bridge's {}; using the plugin's",
                width,
                kv.page_table_width,
            )
            kv.page_table_width = width
        if enable_trace and not self._prefill_warmed:
            raise RuntimeError(
                "decode trace capture before the prefill warmup: every prefill bucket must be compiled first"
            )
        self.generator.warmup_decode(kv_cache=kv.device_cache, enable_trace=bool(enable_trace), page_table_width=width)

    # ---- lifecycle ------------------------------------------------------------------------------------------------
    def release_request(self, slot):
        """A request finished or was preempted while owning state ``slot`` (``model_runner.py:851-868``)."""
        self.generator.release_lane(self._lanes.lane_of_slot(slot))

    def release_persistent_capture(self):
        """Shutdown, mesh still open (``model_runner.py:392-419``): free the decode trace."""
        self.generator.release_traces()

    def close(self):
        self.generator.close()


__all__ = [
    "ARCHITECTURE",
    "DEFAULT_GENERATOR_CLASS",
    "DEFAULT_KV_POOL_TOKENS",
    "LaneMap",
    "MAIN_CLASS",
    "MotifForCausalLM",
    "MotifKVCache",
    "NULL_BLOCK_RESERVE_TOKENS",
    "TT_MODEL_CLASS_OVERRIDES",
    "kv_pool_tokens_from_env",
    "plugin_num_blocks",
    "validate_block_size",
]
