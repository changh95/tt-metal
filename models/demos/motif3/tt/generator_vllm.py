# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM bridge for Motif-3 on a Blackhole Galaxy: ``MotifForCausalLM`` (vllm-tt-plugin plain-class contract).

This module is imported by vLLM's API server, by its registry-inspection subprocess and by EngineCore before the
mesh is open (study 05 §2). Its import is device-free: it pulls in only the standard library, numpy, torch, loguru,
``generator_api`` and ``prefill_plan`` (torch-only). vLLM, ttnn, vllm-tt-plugin and the TT runtime are imported
lazily, inside the methods that need them.

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

Launch flags
------------
Always (design 00 §5.1): ``--trust-remote-code --max-num-seqs 32 --block-size 64 --max-model-len 32768
--additional-config '{"tt": {"trace_mode": "decode_only", "trace_region_size": 268435456, "fabric_config":
"FABRIC_2D_TORUS_XY", "dispatch_core_axis": "col", "l1_small_size": 32768}}'`` (``generator_api.SERVING_TT_CONFIG`` /
``serving_additional_config()``). Never ``--tensor-parallel-size`` or ``--disable-sliding-window``. Optional parsers:
see ``models/demos/motif3/vllm_plugins``. ``--block-size`` must be 32 or 64 (the sizes gates G1/G7 validated) and
``--max-model-len`` a multiple of 256 (the last prefill bucket is ``max_model_len`` itself); both are refused in
``get_max_tokens_all_users``, before any weight is loaded.

Production launch (lead decision 1): chunked prefill + prefix caching + exact device sampling, no speculation
(``docs/features/FEATURES_DESIGN.md`` §1.1; :data:`FEATURE_VLLM_ARGS` without its ``--speculative-config`` pair), with
the feature switches unset (all on) or ``MOTIF3_PREFIX_CACHING=1 MOTIF3_CHUNKED_PREFILL=1`` exported in the API server
and EngineCore alike: ``--enable-chunked-prefill --max-num-batched-tokens 8064 --long-prefill-token-threshold 8064
--enable-prefix-caching --no-async-scheduling`` plus ``"tt": {..., "decode_interleave_prefill_steps": 1,
"decode_interleave_decode_steps": 1, "sample_on_device_mode": "decode_only"}`` (:data:`DEVICE_SAMPLING_TT_CONFIG`) and
``OMP_WAIT_POLICY=PASSIVE``. The budget and the threshold are ``prefill_plan.recommended_budget(span cap 8192, A)`` =
8064 (A = 128: gate G9's per-bucket sp1 q/k, lead decision F5); ``MOTIF3_CHUNK_BUDGET`` (OPTIMIZATION_PLAN.md A1a,
e.g. 4096) asks for a smaller one, aligned by ``recommended_budget`` (:func:`launch_chunk_budget`,
:func:`feature_vllm_args`; ``check_serving_config`` warns when vLLM's flags differ from it). MTP speculation is an OPT-IN launch for
greedy / agentic / low-concurrency serving (and the TIS greedy benchmark variant): add ``--speculative-config
'{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 1}'``
(``MOTIF3_SPEC_DECODE`` unset or 1); sampled rows of such a launch decode without speculation (the plugin's PS-1,
``SpecPlan.verify_requires_speculable_rows``), sampled on device as well. Honest benchmarks add
``--no-enable-prefix-caching`` (or a per-request ``cache_salt``); evals that need logprobs (N > 0) or structured output
are served on any launch without ``--speculative-config`` (those steps sample on the host).

Packed prefill and full-batch verify (``docs/p5_t64/P5_T64_DESIGN.md``; read by ``GeneratorSettings.from_env`` in
EngineCore, so export them in the API server and EngineCore alike):

* P5, ``MOTIF3_PACKED_PREFILL=1`` (off in the code; both TIS specs set it since gates CP-P / CP9-P / E2E-P passed;
  knobs ``MOTIF3_PACKED_PREFILL_MAX_SEG`` / ``_MAX_TOKENS`` / ``_PK1`` and ``MOTIF3_PACKED_WARMUP``): the generator
  runs the short chunks of one prefill step together. The bridge is unchanged: ONE ``prefill_forward_batch`` call per
  step. A row's tokens may then depend on which rows share its pass (near-ties only; lead sign-off 2026-10-04,
  ``docs/P5_T64_REVIEW.md`` §8 I-1).
* T64, ``MOTIF3_SPEC_VERIFY=auto`` on the MTP launch (``packed`` in the code; the MTP TIS spec sets ``auto`` since
  gates G-X / G-serve passed; ``wide`` is the one-trace fallback): a 64-row verify trace next to the 32-lane one. The
  bridge drafts every live lane once the live lanes reach the generator's ``c*`` (``MotifGenerator.drafts_all_lanes``:
  about 19 at the acceptance prior 0.85 and the T64 / T32 step ratio r = 1.13; ``MOTIF3_WIDE_MIN_LANES`` overrides
  ``c*``, ``MOTIF3_WIDE_STEP_RATIO`` overrides r), and keeps the idle-lane budget below it, so c = 32 greedy traffic
  speculates too.

The ``Motif-3 features:`` line, logged once the generator exists, shows ``spec_verify``, ``c*`` and ``packed_prefill``.

Rollback (FEATURES_REVIEW F2 / F6(d)): ``MOTIF3_*=0`` (or ``--no-enable-chunked-prefill --no-enable-prefix-caching``,
no ``--speculative-config``, no ``sample_on_device_mode``) turns the features off, but it is NOT draft 1: the span cap
8192 (design D8) still splits every prompt longer than 8192 tokens into sp0 + sp1 chunks inside the generator (other
numerics, other TTFT). Draft-1 behaviour needs the rollback PAIR ``MOTIF3_*=0`` + ``MOTIF3_PREFILL_MAX_BUCKET=32768``
(and ``MOTIF3_DEVICE_SAMPLING=0`` or no ``sample_on_device_mode`` for draft 1's host sampling).

L1_SMALL (attention P0, ``generator_api.L1_SMALL_SIZE``): the plugin opens the mesh with the ``"l1_small_size"`` of the
``"tt"`` config (``vllm_tt_plugin/worker.py`` ``device_params_from_tt_config``) and with no L1_SMALL region when the key
is absent; the model's CCL semaphores need one (>= 32768 B per core). ``get_max_tokens_all_users`` (inside
``init_device``, before the weights load) refuses a ``"tt"`` config without it, and ``initialize_vllm_model`` refuses a
mesh whose L1_SMALL region is smaller, so a misconfigured server fails at boot instead of at the first prefill after a
decode step.

Weights: ``MOTIF3_WEIGHTS_DIR`` > ``HF_MODEL`` (dir) > the HF-cache snapshot of a repo-id ``HF_MODEL`` at
``TT_MODEL_WEIGHTS_REVISION`` > ``hf_config._name_or_path`` (``generator_api.resolve_weights_location``, the same
order ``MotifTTConfig`` uses); the resolved path and the rule that matched are logged at model init. TT weight cache:
``MOTIF3_TT_CACHE_PATH`` > ``TT_CACHE_PATH``, used per ``MOTIF3_TT_CACHE_POLICY``
(``GeneratorSettings.tt_cache_policy``: ``auto`` never writes it, ``write`` converts and writes the parts that are not
complete; the generator's ``create:`` line logs the policy).

What the plugin calls, and what this class does
------------------------------------------------
* ``model_capabilities`` (class level, read before any instance exists, :func:`model_capabilities_from_env`):
  ``supports_async_decode`` only with ``MOTIF3_ASYNC_DECODE=on`` (B6b, below), ``supports_device_penalties: False`` explicitly (the plugin's default for that key is True);
  ``supports_sample_on_device`` follows ``MOTIF3_DEVICE_SAMPLING`` (default on; NO ``max_device_top_k``: the sampler
  is exact for every top-k / top-p, flagging what it cannot certify for the exact host fallback, and ``top_p = 1``
  lanes without top-k take its full-vocab Gumbel path); ``supports_prefix_caching`` / ``supports_chunked_prefill`` /
  ``supports_spec_decode`` follow the feature switches ``MOTIF3_PREFIX_CACHING`` / ``MOTIF3_CHUNKED_PREFILL`` /
  ``MOTIF3_SPEC_DECODE`` (they only *allow* a feature; vLLM's flags -- ``sample_on_device_mode`` for sampling --
  enable it). Speculation declares the model-owned drafter: ``spec_requirements`` ``(device_propose, hidden_feed)``
  with ``spec_hidden_handoff`` ``(on_device,)``.
* ``spec_plan`` (classmethod, config time, never raises): K = 1, ``lanes_per_request=2`` (a draft takes one more row:
  an idle partner lane in packed verify, its owner's draft row in the 64-row verify), the MTP latent cache as
  ``extra_bytes_per_token`` (612 B per chip in bfp8), ``supports_narrow_decode=True`` and, when the installed plugin
  has it, ``verify_requires_speculable_rows=True`` (PS-1). Refuses K < 1, ``max_num_seqs > 32``, any method but
  ``custom_class`` and a checkpoint without ``model.mtp_layers.0.*``.
* ``get_max_tokens_all_users``: the usable KV pool (``MOTIF3_KV_POOL_TOKENS``, default 262,144) plus a 32-token
  reserve that makes the plugin allocate exactly one extra block for vLLM's null block, which upstream
  ``get_num_available_blocks_tt`` does not budget (``vllm_tt_plugin/worker.py:553-675``; ``block_pool.py:190``). It
  also captures vLLM's scheduler config (:func:`serving_config_of`) for ``initialize_vllm_model`` and runs the
  fail-fast checks of features design §1.5 (:func:`check_serving_config`); the memory check counts the MTP layer.
* ``get_kv_cache_spec``: one ``MLAAttentionSpec(num_kv_heads=1, head_size=576)`` per decoder layer, so the plugin's
  allocation hint is ``(num_blocks, 1, block_size, 576)`` instead of the default ``FullAttentionSpec(16, 192, sw=128)``.
  The MTP layer's cache is model-owned (indexed by the same block ids) and is NOT a vLLM layer.
* ``allocate_kv_cache[_per_layer]``: validates that hint and asks the generator for the latent pool.
* ``prefill_forward``: translates vLLM rows and state slots into Motif lanes (``LaneMap``), zeroes the stale block ids
  vLLM leaves past each request's blocks in reused block-table rows (a bucket-padded prefill would otherwise overwrite
  another live request's KV), and makes ONE ``generator.prefill_forward_batch`` call with every row of the step (new,
  resumed after a prefix hit, or a chunk continuation: ``PrefillRequest.start`` = vLLM ``num_computed_tokens``).
* ``decode_forward``: a host-sampled step returns host logits ``[B, 1, vocab]``; a device-sampled step (the plugin
  sends ``sampling_params``: ``sample_on_device_mode`` ``"decode_only"`` and no host-only request in the step) returns
  the tokens ``int32 [B, 1]`` (plus the raw logprobs ``float32 [B]`` when a row asked for ``logprobs=0``) from
  ``generator.decode_forward_sampled`` (``docs/sampling/DEVICE_SAMPLER.md``); in a speculating launch every step runs
  ``generator.decode_forward_spec`` (ordinary device-sampled steps with ``sampling``) and a verify step (``[B, 2]``
  block + ``num_valid_drafts`` / ``accepted_counts`` / ``spec_mode="argmax_ids"``) returns ``VerifyOutput(argmax_ids
  [B, 2])`` whether or not it carries ``sampling_params`` (PS-1 keeps sampled rows out of verify steps).
* ``propose_draft_tokens``: host only; the drafts are the MTP predictions the last decode step already computed. How
  many: the idle-lane budget, or every live lane when the generator verifies them in one 64-row step
  (``drafts_all_lanes``, asked with the prior-smoothed running acceptance).
* ``warmup_model_prefill`` / ``warmup_model_decode`` / ``release_request`` / ``release_persistent_capture``.

Decode-reload contract v1 (``vllm-tt-plugin/docs/DECODE_RELOAD_CONTRACT.md``). ``MOTIF3_ASYNC_DECODE`` off (the
default, the release): a partial adapter, every decode must carry ``reload_inputs=True`` (always the case without
``supports_async_decode``); ``reload_inputs=False``, ``reload_page_table=True`` and the legacy ``reset_batch`` keyword
raise.

B6b, ``MOTIF3_ASYNC_DECODE=on`` (non-MTP launches; drop ``--no-async-scheduling`` so vLLM schedules asynchronously):
a full adapter with ``supports_async_decode``. A device-sampled plain step submitted with ``read_from_device=False``
returns a :class:`MotifPendingDecode` before its 1 KB read (``generator.submit_decode_sampled``). A steady step
(``reload_inputs=False``, the plugin's resident fast path) continues the previous one: same rows, lanes and sampling,
positions + 1, page table from the call (current), and its tokens = the previous step's sampled tokens, read on the host
after every other input of the step was written (``feed``). So vLLM's ``update_from_output`` / ``schedule`` and the
plugin's input build for step k + 1 run while step k replays; the read of step k, the token write and the replay of
k + 1 stay on the critical path. The token comes from the host read rather than from the device (the sampler's host
fallback may replace a flagged lane's token, so the device token is not final), which keeps every step bitwise the
synchronous one. Host-sampled steps (penalties, logprobs > 0, structured output, ...) and every step the plugin does
not deem steady reload as before; the plugin drains the pending step first. The MTP launch keeps it off (refused at
construction: the plugin refuses async scheduling with speculation).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger

from . import prefill_plan
from .generator_api import (  # noqa: F401  (pool constants re-exported: gv.NULL_BLOCK_RESERVE_TOKENS etc.)
    DEFAULT_BLOCK_SIZE,
    DEFAULT_KV_POOL_TOKENS,
    DEFAULT_PREFILL_ALIGNMENT,
    DEFAULT_PREFILL_SPAN_CAP,
    DEFAULT_SPEC_ALPHA_PRIOR,
    FEATURE_SWITCHES,
    KV_LATENT_DIM,
    KV_LORA_RANK,
    KV_POOL_ALIGNMENT,
    L1_SMALL_SIZE,
    LANES_PER_GROUP,
    MAX_CONTEXT,
    MAX_KV_POOL_TOKENS,
    MESH_SHAPES,
    MTP_LAYER_IDX,
    NULL_BLOCK_RESERVE_TOKENS,
    NUM_DP_GROUPS,
    NUM_HIDDEN_LAYERS,
    NUM_LANES,
    QK_ROPE_HEAD_DIM,
    SERVING_TT_CONFIG,
    SPEC_ALPHA_PRIOR_WEIGHT,
    SUPPORTED_BLOCK_SIZES,
    SUPPORTED_SPEC_TOKENS,
    WIDE_MIN_LANES_NEVER,
    WIDE_SPEC_VERIFY_MODES,
    DecodeBatch,
    GeneratorSettings,
    MotifGenerator,
    PrefillRequest,
    SpecDecodeBatch,
    SpecDecodeResult,
    cdiv,
    check_block_size,
    check_generator_features,
    check_logits,
    check_max_model_len,
    check_prefill_batch,
    check_spec_result,
    check_tt_config,
    chunk_budget_from_env,
    async_decode_from_env,
    feature_switch_from_env,
    host_staging_from_env,
    kv_cache_bytes_per_chip,
    kv_cache_dtype_from_env,
    kv_pool_tokens_from_env,
    kv_replicated_decode_from_env,
    plugin_num_blocks,
    prefill_span_cap_from_env,
    resolve_tt_cache_path,
    resolve_weights_location,
    serving_additional_config,
    smoothed_acceptance,
)

ARCHITECTURE = "MotifForCausalLM"
MAIN_CLASS = "models.demos.motif3.tt.generator_vllm:MotifForCausalLM"
# The bare-name registration vLLM 0.26 needs (see the module docstring). Use this exact string.
TT_MODEL_CLASS_OVERRIDES = f"{ARCHITECTURE}={MAIN_CLASS}"
DEFAULT_GENERATOR_CLASS = "models.demos.motif3.tt.generator:MotifGenerator"

DEFAULT_KV_MAX_GB_PER_CHIP = 16.0  # 34.2 GB DRAM - 14.2 GB weights - ~4 GB activations/trace/CCL (design 00 §1.1)
REQUIRED_NUM_DEVICES = 32

# ---- Features (docs/features/FEATURES_DESIGN.md §1.1-§1.3) -------------------------------------------------------
# Value of an UNSET feature switch: on (design §1.3, "1 once the code lands"). The default generator
# (tt/generator.py MotifGenerator) serves resumed prefill (chunked prefill, prefix caching) and MTP speculation, so the
# class allows all three and vLLM's own flags decide what is enabled. A launch that leaves vLLM's defaults alone (TIS:
# no --enable-chunked-prefill flag, a 32768 budget) therefore runs chunked prefill; --no-enable-prefix-caching and no
# --speculative-config keep the other two off. MOTIF3_*=0 restores the draft-1 capabilities exactly.
FEATURE_SWITCH_DEFAULT = True

# The model-owned drafter (vllm-tt-plugin docs/SPEC_DECODE_CONTRACT.md §4b): the only speculative method Motif serves.
SPEC_METHOD = "custom_class"
MODEL_OWNED_DRAFTER = "vllm_tt_plugin.model_owned_drafter"
SPEC_REQUIREMENTS = ("device_propose", "hidden_feed")  # the MTP layer reads the target's last hidden, on device
SPEC_HIDDEN_HANDOFF = ("on_device",)  # keeps supports_narrow_decode (SPEC_DECODE_CONTRACT.md §1a)
SPEC_ACCEPT_MODE = "argmax_ids"
# A drafted request takes two rows: its own lane + an idle partner lane (packed verify, features design §3.8.2), or its
# anchor row + its draft row on the owner's DP row (the 64-row verify, T64; docs/p5_t64/P5_T64_DESIGN.md §4.1).
SPEC_LANES_PER_REQUEST = 2
SPECULATIVE_CONFIG = {"method": SPEC_METHOD, "model": MODEL_OWNED_DRAFTER, "num_speculative_tokens": 1}

MTP_WEIGHT_PREFIX = "model.mtp_layers.0."
CHECKPOINT_INDEX = "model.safetensors.index.json"

# Exact device sampling (docs/sampling/DEVICE_SAMPLER.md; lead decision 1: on in the production launch). The class
# switch only allows it; the server enables it with "sample_on_device_mode": "decode_only" in --additional-config "tt"
# (prefill steps keep host sampling: one row, a few ms against a TTFT >= 0.7 s).
DEVICE_SAMPLING_SWITCH = "MOTIF3_DEVICE_SAMPLING"
SAMPLE_ON_DEVICE_MODE = "decode_only"
DEVICE_SAMPLING_TT_CONFIG = {"sample_on_device_mode": SAMPLE_ON_DEVICE_MODE}
# The bridge logs the device sampler's counters every this many decode steps of a device-sampling launch (device-sampled
# or host-routed) and at shutdown; MOTIF3_SAMPLING_LOG_EVERY overrides it (0 = at shutdown only).
SAMPLING_LOG_EVERY = 2000

# The vLLM flags of the opt-in MTP launch (features design §1.1 with the lead decision threshold = budget): chunked
# prefill, prefix caching and the model-owned MTP drafter. They are NOT the production default launch (lead decision
# 1), which is these flags WITHOUT the "--speculative-config" pair, plus DEVICE_SAMPLING_TT_CONFIG in
# --additional-config "tt" (module docstring).
def launch_chunk_budget(
    environ: Optional[Mapping[str, str]] = None,
    *,
    span_cap: int = DEFAULT_PREFILL_SPAN_CAP,
    align: int = DEFAULT_PREFILL_ALIGNMENT,
) -> int:
    """The chunk budget (= threshold) a launch passes vLLM (OPTIMIZATION_PLAN.md §3.3 A1a):
    ``prefill_plan.recommended_budget(span_cap, align, MOTIF3_CHUNK_BUDGET)``. Unset knob: 8064 (span cap 8192 - A,
    the code default); ``MOTIF3_CHUNK_BUDGET=4096``: 4096. ``environ`` defaults to ``os.environ``."""
    return prefill_plan.recommended_budget(span_cap, align, chunk_budget_from_env(environ))


def feature_vllm_args(environ: Optional[Mapping[str, str]] = None) -> Tuple[str, ...]:
    """The vLLM flags of the opt-in MTP launch with the chunk budget of :func:`launch_chunk_budget` (``environ``
    defaults to ``os.environ``); :data:`FEATURE_VLLM_ARGS` is this for an empty environment (budget 8064)."""
    b = str(launch_chunk_budget(environ))
    return (
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        b,
        "--long-prefill-token-threshold",
        b,
        "--enable-prefix-caching",
        "--speculative-config",
        json.dumps(SPECULATIVE_CONFIG),
        "--no-async-scheduling",
    )


FEATURE_VLLM_ARGS = feature_vllm_args({})

# The scheduler config of the VllmConfig this process serves, as seen by ``get_max_tokens_all_users`` (which the plugin
# calls in ``init_device`` inside ``set_current_vllm_config``). ``initialize_vllm_model`` runs later, in
# ``load_model``, where vLLM sets no current config, so this is how the block size (BRIDGE-4) and the enabled features
# reach GeneratorSettings. ``allocate_kv_cache``'s hint stays authoritative for the block size.
_SEEN_VLLM_BLOCK_SIZE: Optional[int] = None
_SEEN_VLLM_SERVING: Optional[Dict[str, Any]] = None
# vLLM's --seed (``model_config.seed``), seen there as well: it seeds the device sampler's host RNG behind unseeded
# lanes, so a server run repeats its unseeded draws for the same sequence of steps, as vLLM's host sampler does with its
# global generator. None = not seen (direct calls, tests): OS entropy.
_SEEN_VLLM_SEED: Optional[int] = None

# Keyword arguments the plugin can send that this adapter cannot honour. The speculative ones
# (num_valid_drafts / accepted_counts / spec_mode) are explicit parameters of decode_forward: honoured on a
# speculating launch, refused otherwise.
_UNSUPPORTED_KWARGS = {
    "page_tables_per_layer": "multi-group (hybrid) KV cache configs; Motif-3 declares one uniform MLA spec",
    "rope_deltas_all_users": "M-RoPE",
    "prompt_tokens": "device-side penalties (supports_device_penalties is False: penalized steps sample on the host)",
    "output_tokens": "device-side penalties (supports_device_penalties is False: penalized steps sample on the host)",
}


# ----------------------------------------------------------------------------------------------------------------
# Feature switches and class capabilities (features design §1.2)
# ----------------------------------------------------------------------------------------------------------------
def feature_switches(environ: Optional[Mapping[str, str]] = None) -> Dict[str, bool]:
    """``{switch: allowed}`` for :data:`generator_api.FEATURE_SWITCHES` (``1/true/yes/on`` or ``0/false/no/off``; unset
    = :data:`FEATURE_SWITCH_DEFAULT`; a typo raises instead of silently turning a feature off)."""
    env = os.environ if environ is None else environ
    return {name: feature_switch_from_env(name, env, default=FEATURE_SWITCH_DEFAULT) for name in FEATURE_SWITCHES}


_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")


def device_sampling_switch(environ: Optional[Mapping[str, str]] = None) -> bool:
    """``MOTIF3_DEVICE_SAMPLING`` (``1/true/yes/on`` or ``0/false/no/off``; unset = on, lead decision 1): whether the
    class declares ``supports_sample_on_device``. It only *allows* device sampling; the server enables it with
    ``"sample_on_device_mode": "decode_only"`` (a mode without the capability makes the plugin refuse to start). A typo
    raises instead of silently turning it off."""
    env = os.environ if environ is None else environ
    raw = env.get(DEVICE_SAMPLING_SWITCH)
    if raw is None or raw.strip() == "":
        return True
    v = raw.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    raise ValueError(f"{DEVICE_SAMPLING_SWITCH} must be one of {_TRUE_VALUES + _FALSE_VALUES}, got {raw!r}")


def model_capabilities_from_env(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The class-level ``model_capabilities`` for this environment (the API server and EngineCore must see the same
    switches: the plugin reads the dict from the class in both, before any instance exists, ``platform.py:1817-1822``).

    The switches only *allow* a feature; vLLM's own flags decide (vLLM 0.26 enables chunked prefill and prefix caching
    by default for a model that allows them; device sampling needs ``sample_on_device_mode``). With every switch off
    (``MOTIF3_*=0``, ``MOTIF3_DEVICE_SAMPLING=0``) this is exactly the draft-1 dict. Device sampling declares no
    ``max_device_top_k`` (the sampler is exact for every top-k / top-p; a declared bound would only send requests to
    the host) and keeps ``supports_device_penalties`` False (penalized steps sample on the host).
    ``supports_async_decode`` follows ``MOTIF3_ASYNC_DECODE`` (B6b; default ``off``, the release)."""
    sw = feature_switches(environ)
    caps: Dict[str, Any] = {
        "supports_prefix_caching": sw["MOTIF3_PREFIX_CACHING"],
        "supports_chunked_prefill": sw["MOTIF3_CHUNKED_PREFILL"],
        "supports_async_decode": async_decode_from_env(environ) == "on",  # B6b, MOTIF3_ASYNC_DECODE
        "supports_sample_on_device": device_sampling_switch(environ),
        "supports_device_penalties": False,  # the plugin's default for an absent key is True
        "supports_spec_decode": sw["MOTIF3_SPEC_DECODE"],
        "supports_async_spec_decode": False,
        "output_tokens_per_step": 1,  # > 1 would select the block-output rail and disable chunked prefill
    }
    if sw["MOTIF3_SPEC_DECODE"]:
        caps["spec_requirements"] = SPEC_REQUIREMENTS
        caps["spec_hidden_handoff"] = SPEC_HIDDEN_HANDOFF
    return caps


# ----------------------------------------------------------------------------------------------------------------
# Speculative plan helpers (config time: pure, never raise)
# ----------------------------------------------------------------------------------------------------------------
def mtp_extra_bytes_per_token(kv_cache_dtype: str) -> int:
    """``SpecPlan.extra_bytes_per_token``: device bytes per KV token the MTP layer's latent cache adds, PER CHIP.

    The unit, settled from the plugin code: the field is "device bytes per KV token beyond the target KV"
    (``vllm_tt_plugin/spec_decode.py`` SpecPlan) and is declared but not budgeted (no reader in the plugin; contract
    §3: "need a bytes-per-KV-token conversion the block-count function does not have"). The plugin sizes the target
    KV per logical copy (``worker._available_kv_cache_memory_bytes_for_num_blocks`` = ``page_size_bytes x
    num_blocks``, no device multiplier), and the tt-metal convention of issue #110 is per chip
    (``models/demos/gemma4/tt/generator_vllm.py`` spec_plan). Motif's latent cache is replicated on every chip, so per
    chip = per logical copy: one ``[*, 576]`` row of one more layer = 612 B in bfp8 (576 x 1088 / 1024), 1152 B in
    bf16. Device-wide it is 32 x that (one copy per chip)."""
    return kv_cache_bytes_per_chip(1, DEFAULT_BLOCK_SIZE, 1, kv_cache_dtype) // DEFAULT_BLOCK_SIZE


def spec_plan_supports_speculable_rows() -> bool:
    """The installed vllm-tt-plugin's ``SpecPlan`` has ``verify_requires_speculable_rows`` (PS-1: no verify step holds
    a sampled or penalized request). False when the plugin predates it or is not importable."""
    try:
        from vllm_tt_plugin.spec_decode import SpecPlan
    except Exception:  # pragma: no cover - no plugin: nothing to declare it to
        return False
    return any(f.name == "verify_requires_speculable_rows" for f in dataclasses.fields(SpecPlan))


def mtp_weights_status(vllm_config: Any = None, environ: Optional[Mapping[str, str]] = None) -> Tuple[bool, str]:
    """``(available, why)`` for the MTP layer's weights (``model.mtp_layers.0.*``, shard 104 of the checkpoint), cheap
    and device-free (``spec_plan`` runs at config time in the API server and in EngineCore). Never raises.

    Available when the checkpoint config does not declare ``num_nextn_predict_layers = 0`` and one of: a converted TT
    cache part ``L53`` (its ``.complete`` marker under ``MOTIF3_TT_CACHE_PATH`` / ``TT_CACHE_PATH``); a local checkpoint
    (``generator_api.resolve_weights_location``) whose ``model.safetensors.index.json`` maps ``model.mtp_layers.0.*``
    to shard files that are present; a checkpoint the generator downloads at load time (an uncached repo id, or no
    location at all)."""
    try:
        env = os.environ if environ is None else environ
        model_config = getattr(vllm_config, "model_config", None)
        hf = getattr(model_config, "hf_text_config", None) or getattr(model_config, "hf_config", None)
        n = getattr(hf, "num_nextn_predict_layers", None)
        if n is not None and int(n) < 1:
            return False, f"the checkpoint config declares num_nextn_predict_layers={n}: no MTP layer to draft with"
        reasons: List[str] = []
        root = resolve_tt_cache_path(env)
        if root:
            markers = sorted(Path(root).glob(f"motif3-*/mesh*/L{MTP_LAYER_IDX:02d}/.complete"))
            if markers:
                return True, f"converted TT cache part {markers[0].parent}"
            reasons.append(f"no converted L{MTP_LAYER_IDX:02d} part under the TT cache {root}")
        name = getattr(hf, "_name_or_path", None) or getattr(model_config, "model", None)
        loc = resolve_weights_location(SimpleNamespace(_name_or_path=name), env)
        if loc.path is None:
            return True, "no checkpoint location given (the generator's default snapshot); not verified here"
        if not loc.is_local:
            return True, f"{loc.source}: downloaded with the rest of the checkpoint at load time"
        d = Path(loc.path)
        index = d / CHECKPOINT_INDEX
        if index.is_file():
            try:
                weight_map = json.loads(index.read_text())["weight_map"]
            except Exception as exc:
                reasons.append(f"unreadable {index}: {exc!r}")
            else:
                files = sorted({str(v) for k, v in weight_map.items() if str(k).startswith(MTP_WEIGHT_PREFIX)})
                if not files:
                    reasons.append(f"{index} maps no {MTP_WEIGHT_PREFIX}* tensor")
                else:
                    missing = [f for f in files if not (d / f).is_file()]
                    if not missing:
                        return True, f"checkpoint {d} ({', '.join(files)})"
                    reasons.append(f"the checkpoint {d} lacks {missing}, which hold {MTP_WEIGHT_PREFIX}*")
        else:
            reasons.append(f"no {CHECKPOINT_INDEX} in the checkpoint {d}")
        return False, f"Motif-3 MTP weights ({MTP_WEIGHT_PREFIX}*) not found: " + "; ".join(reasons)
    except Exception as exc:  # spec_plan must not raise: report the failure as the reason
        return False, f"cannot locate the Motif-3 MTP weights ({MTP_WEIGHT_PREFIX}*): {exc!r}"


def _spec_plan(vllm_config: Any, max_num_seqs: int, requested_k: int):
    from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

    spec_cfg = getattr(vllm_config, "speculative_config", None)
    method = getattr(spec_cfg, "method", None)
    if method is not None and str(method) != SPEC_METHOD:
        return SpecReject(
            reason=(
                f"Motif-3 drafts with its own MTP layer (model.mtp_layers.0) only; got speculative method "
                f"{method!r}. Use --speculative-config '{json.dumps(SPECULATIVE_CONFIG)}'"
            ),
            supported_k=(1,),
        )
    if not 1 <= max_num_seqs <= NUM_LANES:
        return SpecReject(
            reason=f"Motif-3 speculates on its {NUM_LANES}-lane decode trace; max_num_seqs={max_num_seqs}",
            supported_k=(),
        )
    if requested_k < 1:
        return SpecReject(
            reason=f"num_speculative_tokens={requested_k}: Motif-3 has one MTP layer and drafts K = 1",
            supported_k=(1,),
        )
    ok, why = mtp_weights_status(vllm_config)
    if not ok:
        return SpecReject(reason=why, supported_k=())
    kwargs: Dict[str, Any] = dict(
        effective_k=1,  # one MTP layer; Motif's README: K = 1 is optimal. The platform publishes it back.
        lanes_per_request=SPEC_LANES_PER_REQUEST,
        extra_bytes_per_seq=0,  # no per-request device state (MTP weights and the trace are per launch)
        extra_bytes_per_token=mtp_extra_bytes_per_token(kv_cache_dtype_from_env()),
        accept_modes=(SPEC_ACCEPT_MODE,),
        drafter_state="internal",  # the MTP cache is model-owned, indexed by the target's block ids
        supports_narrow_decode=True,  # an ordinary step is the [B, 1] decode (the same spec trace, host logits)
    )
    if spec_plan_supports_speculable_rows():
        kwargs["verify_requires_speculable_rows"] = True  # PS-1: a verify step never holds a sampled request
    else:
        logger.warning(
            "Motif-3 spec_plan: the installed vllm-tt-plugin has no SpecPlan.verify_requires_speculable_rows (PS-1), "
            "so a sampled or penalized request that shares a verify step commits the target argmax. Serve sampled "
            "traffic without --speculative-config until a plugin with PS-1 is installed."
        )
    return SpecPlan(**kwargs)


# ----------------------------------------------------------------------------------------------------------------
# vLLM scheduler config capture and fail-fast checks (features design §1.5, §3.9 item 2)
# ----------------------------------------------------------------------------------------------------------------
def serving_config_of(vllm_config: Any) -> Optional[Dict[str, Any]]:
    """The scheduler facts of ``vllm_config`` that ``GeneratorSettings.from_env(serving=...)`` takes
    (``generator_api.SERVING_KEYS``), or None when it carries no cache config (a partial config in a direct call).
    Read after the platform's capability policy ran (``TTWorker.init_device`` -> ``check_and_update_config``), so a
    feature the class does not allow is already off here; ``spec_tokens`` is vLLM's ``num_speculative_tokens`` after
    the platform published ``SpecPlan.effective_k`` back into it."""
    cache = getattr(vllm_config, "cache_config", None)
    if cache is None or getattr(cache, "block_size", None) is None:
        return None
    sched = getattr(vllm_config, "scheduler_config", None)
    spec = getattr(vllm_config, "speculative_config", None)
    budget = getattr(sched, "max_num_batched_tokens", None)
    unit = getattr(cache, "prefix_match_unit", None)
    return {
        "block_size": int(cache.block_size),
        "enable_chunked_prefill": bool(getattr(sched, "enable_chunked_prefill", False)),
        "max_num_batched_tokens": None if budget is None else int(budget),
        "long_prefill_token_threshold": int(getattr(sched, "long_prefill_token_threshold", 0) or 0),
        "enable_prefix_caching": bool(getattr(cache, "enable_prefix_caching", False)),
        "prefix_match_unit": None if unit is None else int(unit),
        "spec_tokens": 0 if spec is None else int(getattr(spec, "num_speculative_tokens", 0) or 0),
    }


def check_serving_config(
    serving: Mapping[str, Any],
    *,
    max_model_len: Optional[int] = None,
    align: int = DEFAULT_PREFILL_ALIGNMENT,
    span_cap: Optional[int] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Features design §1.5 on a captured serving config: raises on configurations that would serve wrong outputs
    (prefix caching without KV-R, a ``--prefix-match-unit`` other than the block size, ``num_speculative_tokens`` not
    in :data:`generator_api.SUPPORTED_SPEC_TOKENS`), returns the performance warnings
    (``prefill_plan.check_scheduler_config``). ``align`` / ``span_cap`` default to the values a generator uses before
    it exists (``DEFAULT_PREFILL_ALIGNMENT``; ``MOTIF3_PREFILL_MAX_BUCKET`` or 8192, at most ``max_model_len``);
    ``initialize_vllm_model`` re-checks with the generator's own ``prefill_alignment`` / ``max_prefill_span``."""
    env = os.environ if environ is None else environ
    k = int(serving.get("spec_tokens") or 0)
    if k not in SUPPORTED_SPEC_TOKENS:
        raise ValueError(
            f"num_speculative_tokens={k}: Motif-3 drafts K = 1 with its MTP layer (spec_plan publishes effective_k=1)"
        )
    prefix = bool(serving.get("enable_prefix_caching"))
    forced = kv_replicated_decode_from_env(env)
    kv_replicated = prefix if forced is None else bool(forced)
    if span_cap is None:
        span_cap = prefill_span_cap_from_env(env) or DEFAULT_PREFILL_SPAN_CAP
        if max_model_len is not None:
            span_cap = min(int(span_cap), int(max_model_len))
    budget = serving.get("max_num_batched_tokens")
    chunked = bool(serving.get("enable_chunked_prefill"))
    threshold = int(serving.get("long_prefill_token_threshold") or 0)
    warnings = prefill_plan.check_scheduler_config(
        chunked=chunked,
        budget=None if budget is None else int(budget),
        threshold=threshold,
        align=int(align),
        span_cap=int(span_cap),
        prefix_caching=prefix,
        prefix_match_unit=serving.get("prefix_match_unit"),
        block_size=int(serving["block_size"]),
        kv_replicated=kv_replicated,
    )
    # A1a: MOTIF3_CHUNK_BUDGET names the budget this launch means to run; vLLM's flags decide, so say when they differ
    target = chunk_budget_from_env(env)
    if chunked and target is not None and int(span_cap) > int(align):
        want = prefill_plan.recommended_budget(int(span_cap), int(align), target)
        if budget is None or int(budget) != want or threshold != want:
            warnings.append(
                f"MOTIF3_CHUNK_BUDGET={target} asks for a chunk budget of {want} (alignment-aware), but vLLM runs "
                f"max_num_batched_tokens {budget} / long_prefill_token_threshold {threshold}: pass "
                f"--max-num-batched-tokens {want} --long-prefill-token-threshold {want}"
            )
    return warnings


def check_sample_on_device_mode(mode: Any) -> Optional[str]:
    """The ``"sample_on_device_mode"`` of the ``"tt"`` config: None (host sampling) or ``"decode_only"`` (exact device
    sampling of decode steps). ``"all"`` is refused (the bridge samples prefill rows on the host: their single row
    costs a few ms against a TTFT >= 0.7 s, and an intermediate chunk must not draw at all), before any weight loads."""
    if mode is None or mode == SAMPLE_ON_DEVICE_MODE:
        return mode
    raise ValueError(
        f"Motif-3 samples decode steps on device and prefill rows on the host: sample_on_device_mode must be "
        f"{SAMPLE_ON_DEVICE_MODE!r} (or unset for host sampling), got {mode!r}"
    )


def _inherits_default(impl: type, name: str) -> bool:
    """``impl`` does not override ``MotifGenerator.<name>`` (whose default says "unsupported")."""
    return inspect.getattr_static(impl, name, None) is MotifGenerator.__dict__[name]


def precheck_generator_class(impl: type, settings: GeneratorSettings) -> None:
    """Refuse, BEFORE ``create`` loads the weights (an hour on a cold cache), a generator class that keeps
    ``MotifGenerator``'s "unsupported" default for a feature vLLM enabled. A class that overrides the property is
    checked on the instance by ``generator_api.check_generator_features`` right after ``create``."""
    name = f"{impl.__module__}.{impl.__name__}"
    if settings.resumed_prefill and _inherits_default(impl, "supports_resumed_prefill"):
        enabled = " and ".join(
            f
            for f, on in (("chunked prefill", settings.chunked_prefill), ("prefix caching", settings.prefix_caching))
            if on
        )
        raise ValueError(
            f"vLLM enabled {enabled}, but the generator class {name} has no resumed prefill (supports_resumed_prefill "
            f"is MotifGenerator's default False); launch with --no-enable-chunked-prefill --no-enable-prefix-caching, "
            f"or set MOTIF3_CHUNKED_PREFILL=0 MOTIF3_PREFIX_CACHING=0 (API server and EngineCore)"
        )
    if settings.spec_decode and _inherits_default(impl, "supports_spec_decode"):
        raise ValueError(
            f"vLLM enabled speculative decoding (num_speculative_tokens={settings.spec_tokens}), but the generator "
            f"class {name} has no decode_forward_spec (supports_spec_decode is MotifGenerator's default False); drop "
            f"--speculative-config or set MOTIF3_SPEC_DECODE=0"
        )


# ----------------------------------------------------------------------------------------------------------------
# Pool / spec helpers (pure; unit-tested on the host). kv_pool_tokens_from_env / plugin_num_blocks live in
# generator_api (shared with MotifTTConfig) and are re-exported here.
# ----------------------------------------------------------------------------------------------------------------
def kv_max_bytes_per_chip(environ=None) -> int:
    """Per-chip KV budget used for fail-fast checks: ``MOTIF3_KV_MAX_GB_PER_CHIP`` (default 16 GB)."""
    env = os.environ if environ is None else environ
    raw = env.get("MOTIF3_KV_MAX_GB_PER_CHIP")
    gb = DEFAULT_KV_MAX_GB_PER_CHIP if raw is None or raw.strip() == "" else float(raw)
    if not gb > 0:
        raise ValueError(f"MOTIF3_KV_MAX_GB_PER_CHIP must be positive, got {raw!r}")
    return int(gb * 1e9)


def validate_block_size(block_size: int) -> int:
    """``block_size`` if it is in ``SUPPORTED_BLOCK_SIZES`` (32, 64: what gates G1/G7 validated), else ValueError."""
    return check_block_size(block_size)


def _vllm_block_size() -> Optional[int]:
    """vLLM's --block-size for this process: the current VllmConfig's, else the one ``get_max_tokens_all_users``
    saw in ``init_device`` (BRIDGE-4); None when neither is known (e.g. a direct test call)."""
    vllm_config = _current_vllm_config()
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None and getattr(cache_config, "block_size", None) is not None:
        return int(cache_config.block_size)
    return _SEEN_VLLM_BLOCK_SIZE


def _vllm_serving_config() -> Optional[Dict[str, Any]]:
    """vLLM's scheduler facts for this process (:func:`serving_config_of`): the current VllmConfig's, else the ones
    ``get_max_tokens_all_users`` captured in ``init_device``; None = unknown (draft 1: every feature off)."""
    serving = serving_config_of(_current_vllm_config())
    if serving is not None:
        return serving
    return None if _SEEN_VLLM_SERVING is None else dict(_SEEN_VLLM_SERVING)


def _seed_of(vllm_config: Any) -> Optional[int]:
    """``vllm_config.model_config.seed`` as an int, or None (no config, no seed)."""
    seed = getattr(getattr(vllm_config, "model_config", None), "seed", None)
    try:
        return None if seed is None else int(seed)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _vllm_seed() -> Optional[int]:
    """vLLM's ``--seed`` for this process: the current VllmConfig's, else the one ``get_max_tokens_all_users`` saw in
    ``init_device``; None = unknown (the device sampler then seeds its unseeded lanes from OS entropy)."""
    seed = _seed_of(_current_vllm_config())
    return seed if seed is not None else _SEEN_VLLM_SEED


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


def _tt_config_of(vllm_config) -> Optional[dict]:
    """The ``"tt"`` object of ``vllm_config.additional_config`` (what the plugin's ``get_tt_config`` reads), ``{}``
    when the server was started without one, or None when the object has no ``additional_config`` at all (a partial
    config in a direct call: nothing to check)."""
    if vllm_config is None or not hasattr(vllm_config, "additional_config"):
        return None
    additional = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional, dict):
        raise ValueError(f"--additional-config must be a JSON object, got {type(additional).__name__}")
    tt = additional.get("tt", {}) or {}
    if not isinstance(tt, dict):
        raise ValueError("--additional-config 'tt' must be a JSON object")
    return tt


def _mesh_l1_small_bytes(mesh_device: Any) -> Optional[int]:
    """L1_SMALL bytes per core of the plugin's mesh, or None when it is not a real ``ttnn.MeshDevice`` (host tests:
    the plugin imported ttnn to open a real mesh, so a mesh without ttnn loaded is a fake). Never imports ttnn."""
    ttnn = sys.modules.get("ttnn")
    mesh_cls = getattr(ttnn, "MeshDevice", None) if ttnn is not None else None
    if mesh_cls is None or not isinstance(mesh_device, mesh_cls):
        return None
    try:
        return int(ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_per_bank)
    except Exception:  # pragma: no cover - API drift: do not block serving on the query itself
        return None


def _validate_mesh_l1_small(mesh_device: Any) -> Optional[int]:
    size = _mesh_l1_small_bytes(mesh_device)
    if size is not None and size < L1_SMALL_SIZE:
        raise ValueError(
            f"the plugin opened the mesh with {size} B of L1_SMALL per core; Motif-3 needs >= {L1_SMALL_SIZE} for its "
            f"CCL semaphores (generator_api.L1_SMALL_SIZE). Launch with --additional-config "
            f'\'{{"tt": {{..., "l1_small_size": {L1_SMALL_SIZE}}}}}\' (TIS: override_tt_config l1_small_size)'
        )
    return size


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
    device state is its lane (decode writes a lane's KV only on its DP group's chips unless KV-R is on, and the
    speculation bookkeeping is keyed by lane), so this keeps ``slot_to_lane``, an injective map onto lanes, and applies
    each accepted remap exactly once (``docs/DECODE_RELOAD_CONTRACT.md:43-93``). KV data never moves; lanes are stable
    for the life of a request. A prefill may land a request on any lane (prefill writes every chip), so a chunked
    prompt may change lane between chunks.

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
    num_layers: int  # main-layer caches the generator allocated (== generator.num_layers)
    vllm_num_layers: int  # layers vLLM accounts for (53, or more than num_layers in a truncated run)
    kv_cache_dtype: str  # device dtype: "bfp8" | "bf16"
    vllm_dtype: Any  # torch dtype vLLM accounted with (bookkeeping only)
    page_table_width: int  # W of every page table sent to the generator
    bytes_per_chip: int  # main layers + the MTP layer
    device_cache: Any  # the generator's handle
    mtp_layers: int = 0  # 1 when the generator also holds the MTP layer's cache (speculation); vLLM never sees it

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.num_blocks, 1, self.block_size, KV_LATENT_DIM)

    @property
    def device_layers(self) -> int:
        """Latent caches on device: the main layers plus the MTP layer."""
        return self.num_layers + self.mtp_layers


# ----------------------------------------------------------------------------------------------------------------
# Speculation bookkeeping (features design §3.8.3, §3.9 item 4)
# ----------------------------------------------------------------------------------------------------------------
class SpecRetained(NamedTuple):
    """What one decode step computed for a lane, kept on the host until ``propose_draft_tokens`` reads it.

    ``pos0`` is the step's anchor position ``n``; ``argmax`` is ``(a0,)`` or, when the lane carried a draft, ``(a0,
    a1)``: the target's choice at ``n`` (the token at ``n + 1``) and at ``n + 1``; ``mtp`` is ``(m0,)`` / ``(m0, m1)``:
    the MTP layer's drafts for ``n + 2`` and ``n + 3``. The plugin commits ``argmax[:count]`` for a greedy row, so the
    next draft is ``mtp[count - 1]``."""

    pos0: int
    argmax: Tuple[int, ...]
    mtp: Tuple[int, ...]

    @property
    def drafted(self) -> bool:
        return len(self.argmax) > 1


@dataclass
class SpecStats:
    """Speculation counters of one bridge (logged at shutdown; read by tests)."""

    ordinary_steps: int = 0  # decode steps sent as the ordinary [B, 1] call
    verify_steps: int = 0  # decode steps sent as the [B, 2] verify block
    drafts_verified: int = 0  # verify rows that carried a draft
    proposals: int = 0  # propose_draft_tokens calls
    drafts_offered: int = 0  # rows offered a draft
    accepted: int = 0  # verified drafts the plugin accepted (count 2)
    rejected: int = 0  # verified drafts the plugin rejected (count 1)
    declined_stale: int = 0  # live rows with no retained step at the committed position
    declined_mismatch: int = 0  # committed tokens are not the retained argmax (a sampled row)
    declined_budget: int = 0  # the next step cannot verify the draft (no idle lane left, no 64-row step)
    # T64 (docs/p5_t64/P5_T64_DESIGN.md §4.7): proposals in which the generator let every live lane draft
    # (MotifGenerator.drafts_all_lanes), and the drafts they offered past the idle-lane budget (the 64-row step's own)
    all_lane_proposals: int = 0
    drafts_beyond_budget: int = 0

    @property
    def verdicts(self) -> int:
        """Verified drafts whose outcome the bridge saw (``accepted + rejected``)."""
        return self.accepted + self.rejected

    def acceptance(self, prior: float = DEFAULT_SPEC_ALPHA_PRIOR, weight: int = SPEC_ALPHA_PRIOR_WEIGHT) -> float:
        """The drafting policy's acceptance estimate (review edit R-E3, ``generator_api.smoothed_acceptance``):
        ``(accepted + weight * prior) / (verdicts + weight)``; ``prior`` before any verdict."""
        return smoothed_acceptance(self.accepted, self.verdicts, prior=prior, weight=weight)

    def as_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)


@dataclass
class SamplingStats:
    """Device-sampling counters of one bridge (logged every :data:`SAMPLING_LOG_EVERY` device steps and at shutdown,
    next to the generator's sampler counters; read by tests)."""

    device_steps: int = 0  # decode steps sampled on device (plain, or ordinary spec steps)
    device_rows: int = 0  # ... their active rows
    logprob_steps: int = 0  # ... that returned the raw logprobs (a row asked for logprobs=0)
    host_steps: int = 0  # decode steps the plugin sampled on the host (no sampling_params) on a device-sampling launch
    verify_steps_with_params: int = 0  # verify steps that carried sampling_params (the argmax path is kept)
    nongreedy_verify_rows: int = 0  # ... their non-greedy active rows (PS-1 keeps this at 0)
    # B6b (MOTIF3_ASYNC_DECODE=on): device-sampled steps returned before their read (read_from_device=False), and
    # steady steps that took their inputs from the previous step (reload_inputs=False)
    deferred_steps: int = 0
    resident_steps: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)


def _param_list(sampling_params: Any, name: str) -> List[Any]:
    v = getattr(sampling_params, name, None)
    if v is None:
        return []
    if isinstance(v, torch.Tensor):
        return v.reshape(-1).tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def lane_sampling_lists(
    sampling_params: Any, lanes: Sequence[int], *, num_lanes: int = NUM_LANES
) -> Tuple[List[float], List[float], List[int], List[Optional[int]]]:
    """The plugin's per-ROW ``TTSamplingParams`` (lists of length B: the rows of the step, padding rows included) and
    each row's lane -> lane-ordered ``(temperature, top_p, top_k, seeds)`` lists of length ``num_lanes`` for the
    generator's device sampler; lanes without a row get the plugin's padding defaults (greedy: temperature 0, top_p 1,
    top_k 1, seed None). The pure-Python twin of ``tt/sampling.lane_lists_from_rows`` (the bridge must not import
    ttnn at module import; ``test_generator_vllm_host`` checks they agree)."""
    T, P, K, S = (_param_list(sampling_params, n) for n in ("temperature", "top_p", "top_k", "seed"))
    rows = len(lanes)
    if not (len(T) == len(P) == len(K) == len(S) == rows):
        raise ValueError(
            f"sampling_params have ({len(T)}, {len(P)}, {len(K)}, {len(S)}) entries for {rows} decode rows"
        )
    t, p, k, sd = [0.0] * num_lanes, [1.0] * num_lanes, [1] * num_lanes, [None] * num_lanes
    for row, lane in enumerate(lanes):
        t[lane], p[lane], k[lane] = float(T[row]), float(P[row]), int(K[row])
        sd[lane] = None if S[row] is None else int(S[row])
    return t, p, k, sd


def wants_logprobs(sampling_params: Any) -> bool:
    """A row of the step asked for ``logprobs=0`` (the sampled token's raw logprob; N > 0 never reaches a device
    step): the plugin then expects ``(tokens, logprobs)``."""
    return any(bool(e) for e in _param_list(sampling_params, "enable_log_probs"))


# ----------------------------------------------------------------------------------------------------------------
# B6b: asynchronous decode (MOTIF3_ASYNC_DECODE=on)
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class _ResidentDecode:
    """The last device-sampled decode step of the bridge (row order): what a steady step (``reload_inputs=False``)
    continues. ``sub`` is the generator's ``SampledSubmission`` (its tokens feed the next step)."""

    rows: int
    lanes: List[int]
    active: torch.Tensor
    row_pos: torch.Tensor
    sub: Any


class MotifPendingDecode:
    """A device-sampled decode step returned before its read (``decode_forward(read_from_device=False)`` with
    ``MOTIF3_ASYNC_DECODE=on``). :meth:`output` reads it once (``generator.read_decode_sampled``, cached there; a later
    step that needs its tokens may have read it already) and returns what a synchronous ``decode_forward`` returns: the
    tokens ``int32 [B, 1]`` in row order, or ``(tokens, logprobs float32 [B])`` when a row asked for logprobs."""

    def __init__(self, bridge: "MotifForCausalLM", sub: Any, lane_idx: torch.Tensor, active: torch.Tensor,
                 sampling_params: Any):  # fmt: skip
        self._bridge, self.sub = bridge, sub
        self._lane_idx, self._active, self._params = lane_idx, active, sampling_params
        self._out: Any = None

    @property
    def resolved(self) -> bool:
        return self._out is not None

    def output(self):
        b = self._bridge
        with b._lock:
            if self._out is None:
                sample = b.generator.read_decode_sampled(self.sub)
                tokens = b._sampled_tokens(sample, self._lane_idx, self._active)
                self._out = b._sampled_output(sample, tokens, self._lane_idx, self._active, self._params)
            return self._out


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
    # MOTIF3_ASYNC_DECODE off (default): partial v1 adapter, async decode off, the plugin sends reload_inputs=True on
    # every decode. on (B6b): full v1 adapter with supports_async_decode (DECODE_RELOAD_CONTRACT.md "Additional
    # requirements"): reload_inputs=False / reload_page_table=True on steady device-sampled steps, read_from_device.
    decode_input_update_contract = 1

    # Read by the plugin from the CLASS at config time (platform.py:1815-1822), before any instance exists; the
    # feature switches are read when this module is imported (identical in the API server and EngineCore).
    model_capabilities = model_capabilities_from_env()

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
        # A feature vLLM enabled must be one the generator serves (features design §1.5, last row).
        check_generator_features(generator, self.settings)
        self.vocab_size = int(generator.vocab_size)
        self._lanes = LaneMap(self.settings.max_batch_size)
        self._kv: Optional[MotifKVCache] = None
        self._prefill_warmed = False  # every bucket compiled (required before decode trace capture)
        self._warned: set = set()
        # Speculation (settings.spec_tokens = 1): every decode step goes through generator.decode_forward_spec and the
        # bridge keeps, per OWNER lane, what the last step computed (lanes are stable for a request's life).
        self._spec = bool(self.settings.spec_decode)
        self._retained: Dict[int, SpecRetained] = {}
        self._propose_calls = 0
        self.spec_stats = SpecStats()
        # Device sampling (sample_on_device_mode "decode_only"): turned on in warmup_model_decode(can_sample_on_device)
        self._device_sampling = False
        self.sampling_stats = SamplingStats()
        raw = os.environ.get("MOTIF3_SAMPLING_LOG_EVERY", "").strip()
        self.sampling_log_every = int(raw) if raw.isdecimal() else SAMPLING_LOG_EVERY  # 0 = shutdown only
        # B6a (MOTIF3_HOST_STAGING): "fast" = the same decode inputs with fewer host ops (_fit_page_table on the used
        # columns only, cached lane index tensors)
        self.host_staging = host_staging_from_env()
        self._lane_idx_cache: Dict[Tuple[int, ...], torch.Tensor] = {}
        self._arange_cache: Optional[torch.Tensor] = None
        # B6b (MOTIF3_ASYNC_DECODE): what the class told the plugin (read at import, the same in every process)
        self._async = bool(self.model_capabilities.get("supports_async_decode", False))
        if self._async and self._spec:
            raise ValueError(
                "MOTIF3_ASYNC_DECODE=on serves the launches without speculation (the plugin refuses asynchronous "
                "scheduling with speculative decoding: supports_async_spec_decode is False); unset it on the MTP launch"
            )
        if self._async and not getattr(generator, "supports_async_decode", False):
            raise ValueError(
                f"MOTIF3_ASYNC_DECODE=on, but the generator {type(generator).__name__} has no split decode submission "
                f"(supports_async_decode / submit_decode_sampled / read_decode_sampled)"
            )
        self._resident: Optional[_ResidentDecode] = None  # the last device-sampled step (async decode)
        self._pending: Optional[MotifPendingDecode] = None  # ... while the plugin has not read it
        self._lock = threading.RLock()  # the plugin may resolve a deferred step from its output thread

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

    # ---- speculative feasibility (config time) -------------------------------------------------------------
    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """The plugin's speculative admission hook (``SPEC_DECODE_CONTRACT.md`` §2), called positionally at config time
        in the API server and again in EngineCore, before any instance exists. Never raises and reads no ``get_tt_*``
        helper: every refusal is a ``SpecReject`` whose reason the plugin quotes.

        Returns ``SpecPlan(effective_k=1, lanes_per_request=2, extra_bytes_per_seq=0,
        extra_bytes_per_token=mtp_extra_bytes_per_token(MOTIF3_KV_CACHE_DTYPE), accept_modes=("argmax_ids",),
        drafter_state="internal", supports_narrow_decode=True[, verify_requires_speculable_rows=True])``; the last
        field only when the installed plugin has it (:func:`spec_plan_supports_speculable_rows`). Refuses: a method
        other than ``custom_class``; ``max_num_seqs`` outside ``[1, 32]``; ``requested_k < 1``; missing MTP weights
        (:func:`mtp_weights_status`)."""
        try:
            return _spec_plan(vllm_config, int(max_num_seqs), int(requested_k))
        except Exception as exc:  # the contract forbids raising; the reason carries the failure
            from vllm_tt_plugin.spec_decode import SpecReject

            return SpecReject(reason=f"Motif-3 spec_plan failed: {exc!r}", supported_k=())

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
        plugin's ``ttnn.MeshDevice``; ``max_batch_size`` = ``max_num_seqs``; ``max_seq_len`` = ``max_model_len``
        (a multiple of 256). Weights (``generator_api.resolve_weights_location``): ``MOTIF3_WEIGHTS_DIR`` >
        ``HF_MODEL`` (dir) > HF-cache snapshot of a repo-id ``HF_MODEL`` at ``TT_MODEL_WEIGHTS_REVISION`` >
        ``hf_config._name_or_path``; TT cache: ``MOTIF3_TT_CACHE_PATH`` > ``TT_CACHE_PATH``, policy
        ``MOTIF3_TT_CACHE_POLICY`` (``GeneratorSettings.from_env``). ``settings`` carries vLLM's ``--block-size``
        (BRIDGE-4) and the features vLLM enabled (chunked prefill, prefix caching, ``num_speculative_tokens``),
        captured by ``get_max_tokens_all_users`` in ``init_device``. The runtime class is ``MOTIF3_GENERATOR_CLASS``
        (default ``models.demos.motif3.tt.generator:MotifGenerator``), imported here, never at module import time; a
        class that cannot serve an enabled feature is refused before ``create`` loads the weights
        (:func:`precheck_generator_class`), the instance right after (``check_generator_features``). The resolved
        launch is logged after ``create`` (:meth:`log_features`, the ``Motif-3 features:`` line).
        """
        if int(tt_data_parallel) != 1:
            raise ValueError(
                f"Motif-3 runs as one vLLM engine (DP=1; MoE + standard DP is refused by the plugin), got "
                f"tt_data_parallel={tt_data_parallel}"
            )
        _validate_mesh(mesh_device)
        l1_small = _validate_mesh_l1_small(mesh_device)
        _validate_hf_config(hf_config)
        serving = _vllm_serving_config()
        settings = GeneratorSettings.from_env(
            hf_config,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
            block_size=_vllm_block_size(),
            serving=serving,
        )
        impl = _resolve_generator_class()
        precheck_generator_class(impl, settings)
        logger.info(
            "Motif-3 vLLM bridge: generator={}.{} layers={}/{} max_batch={} max_seq_len={} block_size={} kv_dtype={} "
            "weights={} ({}, revision {}) l1_small={}",
            impl.__module__,
            impl.__name__,
            settings.num_layers,
            getattr(hf_config, "num_hidden_layers", "?"),
            settings.max_batch_size,
            settings.max_seq_len,
            settings.block_size,
            settings.kv_cache_dtype,
            settings.weights_path,
            settings.weights_source,
            settings.weights_revision,
            l1_small,
        )
        generator = impl.create(hf_config=hf_config, mesh_device=mesh_device, settings=settings)
        if int(generator.vocab_size) != int(getattr(hf_config, "vocab_size", generator.vocab_size)):
            raise ValueError(f"generator vocab {generator.vocab_size} != config vocab {hf_config.vocab_size}")
        bridge = cls(generator, settings)
        bridge._check_generator_geometry(serving)
        bridge.log_features()  # after create: c* is the generator's answer (drafts_all_lanes)
        return bridge

    def _check_generator_geometry(self, serving: Optional[Mapping[str, Any]]) -> None:
        """Re-run the scheduler checks of :func:`check_serving_config` with the generator's own resume alignment and
        span cap (``init_device`` used the defaults) and log the warnings that geometry adds."""
        if serving is None or not self.settings.resumed_prefill:
            return
        L = int(self.settings.max_seq_len)
        assumed = check_serving_config(serving, max_model_len=L)
        A, cap = int(self.generator.prefill_alignment), int(self.generator.max_prefill_span)
        for w in check_serving_config(serving, max_model_len=L, align=A, span_cap=cap):
            if w not in assumed:
                logger.warning("Motif-3 serving config (generator A={}, span cap {}): {}", A, cap, w)

    def log_features(self) -> None:
        """The ``Motif-3 features:`` line: the launch this bridge serves (``initialize_vllm_model`` logs it once the
        generator exists). It starts ``chunked_prefill=<b> (budget <n>, threshold <n>) prefix_caching=<b>
        kv_replicated=<b> spec_tokens=<k> kv_write=<mode> span_cap=<n|None>`` (the fields server checks grep), then
        ``spec_verify=<packed|wide|auto> c*=<...>`` (:meth:`drafting_threshold_text` at the acceptance prior: ``n/a``
        without speculation, ``never`` when the generator never lets every lane draft) and ``packed_prefill=<b>``
        (with its knobs when on).

        Also warns when ``spec_verify`` is ``"wide"`` / ``"auto"`` but the generator class keeps
        ``MotifGenerator.drafts_all_lanes``' default (always False): the 64-row verify would then never engage."""
        s = self.settings
        packed = "True (max_seg {}, max_tokens {}, pk1 {}, warmup {})".format(
            s.packed_prefill_max_seg, s.packed_prefill_max_tokens, s.packed_prefill_pk1, s.packed_warmup
        )
        logger.info(
            "Motif-3 features: chunked_prefill={} (budget {}, threshold {}) prefix_caching={} kv_replicated={} "
            "spec_tokens={} kv_write={} span_cap={} spec_verify={} {} packed_prefill={}",
            s.chunked_prefill,
            s.max_num_batched_tokens,
            s.long_prefill_token_threshold,
            s.prefix_caching,
            s.kv_replicated,
            s.spec_tokens,
            s.kv_write_mode,
            s.prefill_span_cap,
            s.spec_verify,
            self.drafting_threshold_text(),
            packed if s.packed_prefill else False,
        )
        never = _inherits_default(type(self.generator), "drafts_all_lanes")
        if self._spec and s.spec_verify in WIDE_SPEC_VERIFY_MODES and never:
            logger.warning(
                "Motif-3: spec_verify={!r}, but the generator class {} keeps MotifGenerator.drafts_all_lanes' default "
                "(False): the bridge keeps the idle-lane draft budget and the 64-row verify never engages",
                s.spec_verify,
                type(self.generator).__name__,
            )

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
        ``BlockPool`` then takes as its null block. Net: 32 users x (pool/32 tokens + one output block) fit exactly
        (4129 blocks of 64 for the defaults). Raises early (before an hour of weight loading) on configurations
        Motif-3 cannot serve, including a ``max_model_len`` that is not a multiple of 256 (BRIDGE-3) and, when vLLM's
        current config is visible, a ``"tt"`` additional config without ``l1_small_size >= L1_SMALL_SIZE``.

        With vLLM's current config visible (EngineCore's ``init_device``), it also captures the scheduler facts
        (:func:`serving_config_of`) for ``initialize_vllm_model`` and runs the features-design §1.5 checks
        (:func:`check_serving_config`: raises on prefix caching without KV-R or a foreign ``--prefix-match-unit``,
        logs performance warnings); the memory check then counts the MTP layer's cache when vLLM speculates.
        """
        global _SEEN_VLLM_BLOCK_SIZE, _SEEN_VLLM_SERVING, _SEEN_VLLM_SEED
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
            check_max_model_len(int(max_model_len))  # multiple of 256: bucket + SDPA-chunk alignment (BRIDGE-3)
            if int(max_model_len) > pool:
                raise ValueError(f"max_model_len={max_model_len} does not fit the {pool}-token KV pool")

        # Fail-fast memory check with the real block size when vLLM exposes the config here (it does inside
        # EngineCore's init_device); otherwise use the largest supported block (an upper bound).
        block_size = max(SUPPORTED_BLOCK_SIZES)
        num_layers = NUM_HIDDEN_LAYERS
        mtp_layers = 0
        vllm_config = _current_vllm_config()
        seed = _seed_of(vllm_config)
        if seed is not None:  # vLLM's --seed, for the device sampler's unseeded lanes (_enable_device_sampling)
            _SEEN_VLLM_SEED = seed
        tt_config = _tt_config_of(vllm_config)
        if tt_config is not None:  # the plugin opened (or will open) the mesh with exactly this l1_small_size
            check_tt_config(tt_config, where="vLLM --additional-config 'tt'")
            check_sample_on_device_mode(tt_config.get("sample_on_device_mode"))
        if vllm_config is not None and getattr(vllm_config, "cache_config", None) is not None:
            block_size = validate_block_size(vllm_config.cache_config.block_size)
            try:
                num_layers = int(vllm_config.model_config.hf_text_config.num_hidden_layers)
            except Exception:  # pragma: no cover - partial configs
                pass
            serving = serving_config_of(vllm_config)
            if serving is not None:
                for warning in check_serving_config(serving, max_model_len=max_model_len):
                    logger.warning("Motif-3 serving config: {}", warning)
                mtp_layers = 1 if serving["spec_tokens"] else 0
                logger.info(
                    "Motif-3 serving config: chunked_prefill={} (max_num_batched_tokens {}, "
                    "long_prefill_token_threshold {}) prefix_caching={} (prefix_match_unit {}) "
                    "num_speculative_tokens={}",
                    serving["enable_chunked_prefill"],
                    serving["max_num_batched_tokens"],
                    serving["long_prefill_token_threshold"],
                    serving["enable_prefix_caching"],
                    serving["prefix_match_unit"],
                    serving["spec_tokens"],
                )
            _SEEN_VLLM_BLOCK_SIZE = block_size  # for initialize_vllm_model, which runs without a current config
            _SEEN_VLLM_SERVING = serving
        env_layers = os.environ.get("MOTIF3_NUM_LAYERS", "").strip()
        if env_layers.isdecimal() and int(env_layers) > 0:
            num_layers = min(num_layers, int(env_layers))
        kv_dtype = kv_cache_dtype_from_env()
        tokens = pool + NULL_BLOCK_RESERVE_TOKENS
        blocks = plugin_num_blocks(tokens, block_size, int(max_num_seqs or NUM_LANES))
        need = kv_cache_bytes_per_chip(blocks, block_size, num_layers + mtp_layers, kv_dtype)
        cap = kv_max_bytes_per_chip()
        if need > cap:
            mtp = " + the MTP layer" if mtp_layers else ""
            raise ValueError(
                f"a {pool}-token {kv_dtype} latent pool ({blocks} blocks of {block_size}, {num_layers} layers{mtp}) "
                f"needs {need / 1e9:.2f} GB per chip, over the {cap / 1e9:.2f} GB KV budget "
                f"(MOTIF3_KV_MAX_GB_PER_CHIP); lower MOTIF3_KV_POOL_TOKENS"
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
        hybrid model does today (study 05 §7.3 option A). The MTP layer (speculation) is NOT listed: its cache is
        model-owned, indexed by the same block ids, so it travels with prefix hits and is freed with the request; the
        single-group allocation would silently drop a 54th entry (features design §3.4).
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
        ``generator.num_layers`` caches (fewer than vLLM counts only in a ``MOTIF3_NUM_LAYERS`` truncated run), plus
        the MTP layer's cache when ``settings.spec_tokens`` (the generator adds it itself), and returns a
        ``MotifKVCache``.
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
        if self.settings.block_size is not None and int(self.settings.block_size) != block_size:
            logger.warning(
                "Motif-3: the KV hint's block size {} differs from the --block-size {} seen at model init; the hint "
                "wins (the generator must take num_blocks / block_size from allocate_kv_cache)",
                block_size,
                self.settings.block_size,
            )
        vllm_layers = int(num_layers)
        layers = self.generator.num_layers
        if vllm_layers < layers:
            raise ValueError(f"vLLM accounts for {vllm_layers} attention layers but the generator runs {layers}")
        if vllm_layers > layers:
            logger.info("Motif-3 truncated run: allocating {} of vLLM's {} layer caches", layers, vllm_layers)
        kv_dtype = self.settings.kv_cache_dtype
        mtp = int(self.settings.mtp_kv_layers)
        need = kv_cache_bytes_per_chip(num_blocks, block_size, layers + mtp, kv_dtype)
        cap = kv_max_bytes_per_chip()
        if need > cap:
            raise ValueError(
                f"KV pool {shape} x {layers} layers{' + the MTP layer' if mtp else ''} ({kv_dtype}) needs "
                f"{need / 1e9:.2f} GB per chip, over the {cap / 1e9:.2f} GB budget (MOTIF3_KV_MAX_GB_PER_CHIP)"
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
            mtp_layers=mtp,
        )
        logger.info(
            "Motif-3 KV pool: {} blocks x {} tokens ({} usable after vLLM's null block) x {} layers{}, {} on device "
            "(vLLM accounts {}), {:.2f} GB per chip, page-table width {}",
            num_blocks,
            block_size,
            (num_blocks - 1) * block_size,
            layers,
            " + the MTP layer" if mtp else "",
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
        fitted = pt * valid  # the row's own block ids, then 0 (null block); int32
        if bool(((fitted < 1) & valid).any()):
            raise ValueError(f"{where}: a position this step needs is on the null block (block id 0)")
        if bool((fitted >= kv.num_blocks).any()):
            raise ValueError(f"{where}: page_table has block ids outside [1, {kv.num_blocks})")
        return fitted.contiguous()

    def _fit_page_table_fast(self, page_table, kv: MotifKVCache, valid_blocks: torch.Tensor, where: str) -> torch.Tensor:
        """:meth:`_fit_page_table` (``host_staging="fast"``): the same checks and errors, but only the first
        ``max(valid_blocks)`` columns of the result (``int32 [rows, top]``; every later column of the full result is
        0)."""
        pt = torch.as_tensor(page_table)
        if pt.ndim != 2 or pt.shape[0] != valid_blocks.shape[0]:
            raise ValueError(f"{where}: page_table must be [{valid_blocks.shape[0]}, blocks], got {tuple(pt.shape)}")
        width = kv.page_table_width
        need = valid_blocks.to(torch.int64)
        top = int(need.max()) if need.numel() else 0
        if top > min(width, pt.shape[1]):
            raise ValueError(
                f"{where}: a row needs {int(need.max())} blocks; the block table has {pt.shape[1]} columns and the "
                f"context window {width}"
            )
        ar = self._arange_cache
        if ar is None or ar.shape[0] < width:
            ar = self._arange_cache = torch.arange(width)
        valid = ar[None, :top] < need[:, None]
        fitted = pt[:, :top].to(torch.int32) * valid
        if bool(((fitted < 1) & valid).any()):
            raise ValueError(f"{where}: a position this step needs is on the null block (block id 0)")
        if bool((fitted >= kv.num_blocks).any()):
            raise ValueError(f"{where}: page_table has block ids outside [1, {kv.num_blocks})")
        return fitted

    def _lane_index(self, lanes: List[int]) -> torch.Tensor:
        """``torch.tensor(lanes, dtype=torch.long)``, cached per lane list (``host_staging="fast"``; callers never
        modify it)."""
        if self.host_staging != "fast":
            return torch.tensor(lanes, dtype=torch.long)
        key = tuple(lanes)
        t = self._lane_idx_cache.get(key)
        if t is None:
            if len(self._lane_idx_cache) >= 4096:
                self._lane_idx_cache.clear()
            t = self._lane_idx_cache[key] = torch.tensor(lanes, dtype=torch.long)
        return t

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
        """One prefill step (:meth:`_prefill_forward_unlocked` documents it) under the bridge's lock. It ends the
        resident decode chain (B6b: the plugin reloads the next decode)."""
        with self._lock:
            self._resident = None
            return self._prefill_forward_unlocked(
                tokens, page_table, kv_cache, prompt_lens, start_pos=start_pos, enable_trace=enable_trace,
                sampling_params=sampling_params, empty_slots=empty_slots, **kwargs,
            )  # fmt: skip

    def _prefill_forward_unlocked(
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
        """Prefill every row of one plugin step with ONE ``generator.prefill_forward_batch`` call
        (``model_runner.py:3155-3219``; features design §2.3).

        Args (all keyword, as the plugin sends them):
            tokens: ``torch.int32 [B, max(prompt_lens)]``; row ``i`` holds ALL its tokens from position 0 up to
                ``prompt_lens[i]`` (the cached prefix, earlier chunks, this chunk; prompt + generated tokens for a
                request resumed after preemption), stale after.
            page_table: ``torch.int32 [B, W]`` vLLM block table rows: shared cached blocks, then the request's own
                blocks (plus speculative lookahead blocks), then stale ids of other requests' blocks (rows are reused
                uncleared). Entries past ``ceil(prompt_lens[i] / bs)`` are zeroed before the generator sees them.
            kv_cache: the ``MotifKVCache`` from ``allocate_kv_cache``.
            prompt_lens: numpy int64 ``[B]``: END of the chunk vLLM scheduled (not the sequence length).
            start_pos: numpy int32 ``[B]``: ``num_computed_tokens`` = positions already in the cache (a prefix-cache
                hit: a block multiple; a chunk continuation: any integer). Must be 0 unless vLLM enabled chunked
                prefill or prefix caching for this launch (``settings.resumed_prefill``).
            enable_trace: plugin ``trace_mode == "all"``; passed through (prefill is eager).
            sampling_params: only with ``sample_on_device_mode`` "all" (refused: prefill rows sample on the host).
            empty_slots: ``list[int]`` destination state slot per row (always sent outside lane mode). A chunked
                request may get another slot (lane) for each chunk: no lane state crosses chunks.

        Returns:
            Host logits ``[B, 1, vocab]`` (float32 or bfloat16) of each row's position ``prompt_lens[i] - 1`` (also for
            an intermediate chunk, whose sample the plugin discards); the plugin's host sampler reads ``[rows, -1, :]``.
        """
        kv = self._check_kv(kv_cache)
        self._reject_unsupported(kwargs, "prefill_forward")
        if sampling_params is not None:
            raise NotImplementedError(
                "Motif-3 samples prefill rows on the host (sample_on_device_mode 'decode_only'); sampling_params on a "
                "prefill step means sample_on_device_mode 'all', which is not supported"
            )
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
            if not 1 <= end <= min(int(tokens_t.shape[1]), max_len):
                raise ValueError(f"row {i}: prompt length {end} outside [1, {min(int(tokens_t.shape[1]), max_len)}]")
            if start != 0 and not self.settings.resumed_prefill:
                raise NotImplementedError(
                    f"row {i}: start_pos={start}, but vLLM enabled neither prefix caching nor chunked prefill for "
                    f"this launch (GeneratorSettings), so every prefill row must start at 0"
                )
            if not 0 <= start < end:
                raise ValueError(f"row {i}: start_pos={start} outside [0, prompt_lens={end})")
        table = torch.as_tensor(page_table)
        if table.ndim != 2 or table.shape[0] < rows:
            raise ValueError(f"prefill page_table {tuple(table.shape)} has fewer rows than the {rows} prompts")
        need = torch.as_tensor([cdiv(int(e), kv.block_size) for e in ends], dtype=torch.int64)
        pt = self._fit_page_table(table[:rows], kv, need, "prefill_forward")
        requests = check_prefill_batch(
            [
                PrefillRequest(
                    lane=lanes[i],
                    tokens=tokens_t[i, : int(ends[i])].to(torch.int32).contiguous(),
                    page_table=pt[i].clone(),
                    start=int(starts[i]),
                )
                for i in range(rows)
            ]
        )
        logits = self.generator.prefill_forward_batch(
            requests, kv_cache=kv.device_cache, enable_trace=bool(enable_trace)
        )
        check_logits("MotifGenerator.prefill_forward_batch", logits, (rows, self.vocab_size))
        for lane in lanes:  # the lane now belongs to the prefilled request; the plugin never proposes after a prefill
            self._retained.pop(lane, None)
        return logits.unsqueeze(1)

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
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        **kwargs,
    ):
        """One decode step (the plugin's call; :meth:`_decode_forward_unlocked` documents it), under the bridge's lock:
        with ``MOTIF3_ASYNC_DECODE=on`` the plugin may resolve a deferred step (:class:`MotifPendingDecode`) from its
        output thread."""
        with self._lock:
            return self._decode_forward_unlocked(
                tokens, start_pos, page_table, kv_cache, enable_trace=enable_trace,
                read_from_device=read_from_device, sampling_params=sampling_params, slot_remap=slot_remap,
                reload_inputs=reload_inputs, reload_page_table=reload_page_table,
                reload_sampling_params=reload_sampling_params, reset_sampling_state=reset_sampling_state,
                num_valid_drafts=num_valid_drafts, accepted_counts=accepted_counts, spec_mode=spec_mode, **kwargs,
            )  # fmt: skip

    def _decode_forward_unlocked(
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
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        **kwargs,
    ):
        """One decode step (``async_decode.py:1144-1292``).

        Args (keyword, as the plugin sends them):
            tokens: ``torch.int32 [B, 1]`` (B = ``max_num_seqs``, front-packed; padding rows token 0); a verify step
                sends the ``[B, 2]`` candidate block: column 0 the row's last committed token, column 1 its draft
                (``PLACEHOLDER_TOKEN_ID`` = -1 where ``num_valid_drafts`` is 0).
            start_pos: ``torch.int32 [B]`` position of each input token = KV write slot; padding rows ``-1``. A verify
                step sends ``[B, 2]``: column 1 = column 0 + 1 on drafted rows, -1 elsewhere.
            page_table: ``torch.int32 [B, W]`` (padding rows 0; entries past the row's last position's block are
                zeroed, as above).
            kv_cache: the ``MotifKVCache``.
            enable_trace: plugin ``trace_mode in ("all", "decode_only")``.
            read_from_device: ``MOTIF3_ASYNC_DECODE`` off: ignored, the result is always a host tensor (the plugin then
                skips its read hooks). on (B6b): a device-sampled step with ``read_from_device=False`` returns a
                :class:`MotifPendingDecode` before its read (``read_decode_output`` / ``process_decode_output_host``
                resolve it); host-sampled and verify steps still return host tensors.
            sampling_params: ``TTSamplingParams`` of the step's rows (lists of length B, padding rows greedy) when the
                plugin samples this step on device (``sample_on_device_mode`` "decode_only" and no host-only request
                in the step): the step is sampled by the generator's exact device sampler. On a verify step they are
                ignored (argmax ids; PS-1 keeps sampled rows out of verify steps: a non-greedy row is counted and
                logged).
            slot_remap: ``torch.int32 [max_num_seqs]`` or None: row ``i`` reads state slot ``slot_remap[i]``. Applied
                to the lane map exactly once, after the generator accepted the step.
            reload_inputs / reload_page_table / reload_sampling_params / reset_sampling_state: contract-v1 commands.
                Without async decode the plugin always sends ``reload_inputs=True``. With ``MOTIF3_ASYNC_DECODE=on``
                a steady device-sampled step comes with ``reload_inputs=False`` (and ``reload_page_table=True`` when
                the blocks changed): its ``tokens`` / ``start_pos`` are stale and ignored; the step continues the
                previous one (:meth:`_decode_resident`). ``page_table`` is always current. The device sampler needs no
                reload or reset: it compares the lane parameters every step (a device write only on change) and its
                RNG has no hidden state (counters derived from (seed, position) every step).
            num_valid_drafts / accepted_counts: ``torch.int32 [B]``, a verify step only (``SPEC_DECODE_CONTRACT.md``
                §4a; refused on a launch without speculation). ``accepted_counts`` is informational: Motif's
                speculative state is the retained ids (``propose_draft_tokens`` receives the counts itself).
            spec_mode: ``"argmax_ids"`` on a verify step.

        Returns:
            Host logits ``[B, 1, vocab]`` in row order (rows of padding are don't-care); a device-sampled step returns
            the sampled tokens ``int32 [B, 1]``, or ``(tokens, logprobs float32 [B])`` when a row asked for
            ``logprobs=0`` (vLLM's raw logprob of the sampled token); a verify step returns
            ``VerifyOutput(spec_mode="argmax_ids", argmax_ids=int32 [B, 2], hidden=None)``: column 0 = the target's
            choice after the row's last committed token, column 1 = its choice after the draft (``-1`` on rows without
            one). In a speculating launch every step runs ``generator.decode_forward_spec`` (one decode trace; with
            ``MOTIF3_SPEC_VERIFY=auto`` the generator also holds the 64-row trace and picks one per step) and the
            bridge retains, per lane, the MTP drafts ``propose_draft_tokens`` hands out next.
        """
        kv = self._check_kv(kv_cache)
        self._reject_unsupported(kwargs, "decode_forward")
        spec_args = (num_valid_drafts, accepted_counts, spec_mode)
        is_verify = any(a is not None for a in spec_args)
        if is_verify and not self._spec:
            raise NotImplementedError(
                "decode_forward: num_valid_drafts / accepted_counts / spec_mode (speculative decoding) are not "
                "supported: vLLM enabled no speculation for this launch (GeneratorSettings.spec_tokens = 0)"
            )
        if is_verify and any(a is None for a in spec_args):
            missing = [n for n, a in zip(("num_valid_drafts", "accepted_counts", "spec_mode"), spec_args) if a is None]
            raise ValueError(
                f"decode_forward: a verify step carries num_valid_drafts, accepted_counts and spec_mode together; "
                f"missing {missing}"
            )
        if not reload_inputs:
            if not self._async:
                raise NotImplementedError(
                    "MotifForCausalLM is a partial decode-reload v1 adapter: every decode must reload its inputs "
                    "(supports_async_decode is False; MOTIF3_ASYNC_DECODE=on makes it a full one)"
                )
            if is_verify:
                raise ValueError("a verify step always reloads its inputs (plugin contract)")
            return self._decode_resident(
                kv, tokens, start_pos, page_table, slot_remap, sampling_params, bool(enable_trace),
                bool(read_from_device),
            )  # fmt: skip
        if reload_page_table:
            raise ValueError("reload_page_table is only legal with reload_inputs=False (plugin contract)")
        # every reloading step ends the resident chain (a device-sampled one starts a new chain below)
        self._resident = None
        if sampling_params is not None:
            self._require_device_sampling()
        if is_verify:
            return self._decode_verify(
                kv, tokens, start_pos, page_table, slot_remap, num_valid_drafts, accepted_counts, spec_mode,
                bool(enable_trace), sampling_params,
            )  # fmt: skip
        tok = torch.as_tensor(tokens)
        if tok.ndim == 2:
            if tok.shape[1] != 1:
                if self._spec:
                    raise ValueError(f"decode tokens {tuple(tok.shape)} without spec_mode: a verify block needs it")
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
        fast = self.host_staging == "fast"
        fit = self._fit_page_table_fast if fast else self._fit_page_table
        pt = fit(page_table, kv, need, "decode_forward")  # inactive rows come back all zero ("fast": used columns)
        lanes = self._lanes.decode_lanes(rows, slot_remap)
        lane_idx = self._lane_index(lanes)
        lane_tokens = torch.zeros(NUM_LANES, dtype=torch.int32)
        lane_pos = torch.full((NUM_LANES,), -1, dtype=torch.int32)
        lane_pt = torch.zeros((NUM_LANES, kv.page_table_width if fast else pt.shape[1]), dtype=torch.int32)
        lane_tokens[lane_idx] = torch.where(active, tok.to(torch.int32), torch.zeros_like(pos))
        lane_pos[lane_idx] = pos
        if fast:
            lane_pt[lane_idx, : pt.shape[1]] = pt
        else:
            lane_pt[lane_idx] = pt
        batch = DecodeBatch(tokens=lane_tokens, positions=lane_pos, page_table=lane_pt)
        retained: Dict[int, SpecRetained] = {}
        sampling = None if sampling_params is None else lane_sampling_lists(sampling_params, lanes)
        sample = logits = None
        if self._spec:
            spec_kw = {} if sampling is None else {"sampling": sampling}
            result = self.generator.decode_forward_spec(
                SpecDecodeBatch.from_decode_batch(batch),
                kv_cache=kv.device_cache,
                enable_trace=bool(enable_trace),
                want_logits=sampling is None,
                **spec_kw,
            )
            check_spec_result(
                "MotifGenerator.decode_forward_spec", result, want_logits=sampling is None, vocab_size=self.vocab_size
            )
            retained = self._spec_retained(result, lanes, pos.tolist(), active.tolist(), [False] * rows)
            if sampling is None:
                logits = result.logits
            else:
                sample = getattr(result, "sample", None)
        elif sampling is not None and self._async:  # B6b: submit, read later (or now with read_from_device)
            return self._submit_sampled(
                kv, batch, sampling, sampling_params, rows, lanes, lane_idx, active, pos, slot_remap,
                bool(enable_trace), bool(read_from_device), feed=None,
            )  # fmt: skip
        elif sampling is not None:
            sample = self.generator.decode_forward_sampled(
                batch, sampling, kv_cache=kv.device_cache, enable_trace=bool(enable_trace)
            )
        else:
            logits = self.generator.decode_forward(batch, kv_cache=kv.device_cache, enable_trace=bool(enable_trace))
            check_logits("MotifGenerator.decode_forward", logits, (NUM_LANES, self.vocab_size))
        if sampling is not None:
            tokens_out = self._sampled_tokens(sample, lane_idx, active)
        # Accepted: commit the slot move exactly once (the plugin settles its own map right after we return).
        self._lanes.commit(slot_remap)
        if self._spec:
            self._retained.update(retained)
            self.spec_stats.ordinary_steps += 1
        if sampling is not None:
            return self._sampled_output(sample, tokens_out, lane_idx, active, sampling_params)
        if self._device_sampling:
            self.sampling_stats.host_steps += 1
            self._maybe_log_sampling()
        if lanes == list(range(rows)):
            out = logits[:rows]
        else:
            out = logits.index_select(0, lane_idx)
        return out.unsqueeze(1)

    # ---- B6b: asynchronous decode (MOTIF3_ASYNC_DECODE=on) ------------------------------------------------------
    def _submit_sampled(
        self, kv, batch, sampling, sampling_params, rows, lanes, lane_idx, active, row_pos, slot_remap,
        enable_trace: bool, read_from_device: bool, *, feed,
    ):  # fmt: skip
        """Submit a device-sampled step (``generator.submit_decode_sampled``), commit the slot move, record the step
        as the resident chain's last and return its :class:`MotifPendingDecode` (resolved now with
        ``read_from_device``)."""
        gen = self.generator
        if feed is None and getattr(gen, "outstanding_decode", None) is not None:
            gen.settle_decode()  # a reloading step after a deferred one the plugin has not read (its result is kept)
        sub = gen.submit_decode_sampled(batch, sampling, kv_cache=kv.device_cache, enable_trace=enable_trace, feed=feed)
        # Accepted: commit the slot move exactly once (the plugin settles its own map right after we return).
        self._lanes.commit(slot_remap)
        self._resident = _ResidentDecode(rows=int(rows), lanes=list(lanes), active=active.clone(),
                                         row_pos=row_pos.clone(), sub=sub)  # fmt: skip
        handle = MotifPendingDecode(self, sub, lane_idx, active, sampling_params)
        if read_from_device:
            return handle.output()
        self.sampling_stats.deferred_steps += 1
        return handle

    def _decode_resident(
        self, kv, tokens, start_pos, page_table, slot_remap, sampling_params, enable_trace: bool,
        read_from_device: bool,
    ):  # fmt: skip
        """A steady device-sampled step (``reload_inputs=False``, decode-reload contract v1): the rows, lanes and
        sampling of the previous step, each active row one position further, its token the previous step's sampled
        token (``generator.submit_decode_sampled(feed=...)``: read on the host, after every other input of this step
        is written). ``tokens`` / ``start_pos`` are the plugin's stale host copies: only their shape and active rows
        are checked. ``page_table`` is current (a block crossing comes with ``reload_page_table=True``; it is fitted
        every step). Anything that is not a continuation of the previous step raises."""
        r = self._resident
        if r is None:
            raise RuntimeError(
                "decode_forward(reload_inputs=False) without a resident decode step: the plugin must reload after a "
                "prefill, a host-sampled or verify step, a layout change or a released request"
            )
        if sampling_params is None:
            raise ValueError("reload_inputs=False on a host-sampled decode step (host sampling reloads every step)")
        if slot_remap is not None:
            raise ValueError("reload_inputs=False with a slot_remap: a layout change must reload its inputs")
        tok = torch.as_tensor(tokens)
        stale = torch.as_tensor(start_pos)
        if tok.ndim != 2 or tok.shape[1] != 1 or int(tok.shape[0]) != r.rows or stale.reshape(-1).shape[0] != r.rows:
            raise ValueError(
                f"reload_inputs=False: decode tokens {tuple(tok.shape)} / start_pos {tuple(stale.shape)} do not "
                f"continue the previous step's {r.rows} rows"
            )
        if not torch.equal(stale.reshape(-1) >= 0, r.active):
            raise ValueError("reload_inputs=False, but the active rows changed: a layout change must reload its inputs")
        pos = torch.where(r.active, r.row_pos + 1, r.row_pos).to(torch.int32)
        if bool((pos[r.active] >= self.settings.max_seq_len).any()):
            raise ValueError(f"decode positions must be in [0, {self.settings.max_seq_len})")
        need = torch.where(r.active, pos.to(torch.int64) // kv.block_size + 1, torch.zeros_like(pos, dtype=torch.int64))
        fast = self.host_staging == "fast"
        fit = self._fit_page_table_fast if fast else self._fit_page_table
        pt = fit(page_table, kv, need, "decode_forward (resident)")
        lane_idx = self._lane_index(r.lanes)
        lane_pos = torch.full((NUM_LANES,), -1, dtype=torch.int32)
        lane_pt = torch.zeros((NUM_LANES, kv.page_table_width if fast else pt.shape[1]), dtype=torch.int32)
        lane_pos[lane_idx] = pos
        if fast:
            lane_pt[lane_idx, : pt.shape[1]] = pt
        else:
            lane_pt[lane_idx] = pt
        batch = DecodeBatch(tokens=torch.zeros(NUM_LANES, dtype=torch.int32), positions=lane_pos, page_table=lane_pt)
        sampling = lane_sampling_lists(sampling_params, r.lanes)
        self.sampling_stats.resident_steps += 1
        return self._submit_sampled(
            kv, batch, sampling, sampling_params, r.rows, r.lanes, lane_idx, r.active, pos, None, enable_trace,
            read_from_device, feed=r.sub,
        )  # fmt: skip

    # ---- device sampling ----------------------------------------------------------------------------------------
    def _require_device_sampling(self) -> None:
        """A step carries ``sampling_params``: the device sampler must be on (``warmup_model_decode`` turns it on
        with ``can_sample_on_device``). Without warmup (``enable_model_warmup`` false) it is turned on lazily here,
        before any decode trace exists."""
        if self._device_sampling:
            return
        if not self.model_capabilities.get("supports_sample_on_device"):
            raise NotImplementedError(
                f"decode_forward got sampling_params, but {DEVICE_SAMPLING_SWITCH}=0 (supports_sample_on_device is "
                f"False): unset sample_on_device_mode"
            )
        self._enable_device_sampling()

    def _enable_device_sampling(self) -> None:
        if self._device_sampling:
            return
        if not getattr(self.generator, "supports_device_sampling", False):
            raise NotImplementedError(
                f"sample_on_device_mode={SAMPLE_ON_DEVICE_MODE!r}, but the generator {type(self.generator).__name__} "
                f"has no device sampler (enable_device_sampling / decode_forward_sampled): unset sample_on_device_mode "
                f"or set {DEVICE_SAMPLING_SWITCH}=0"
            )
        seed = _vllm_seed()
        self.generator.enable_device_sampling(**({} if seed is None else {"rng_seed": seed}))
        self._device_sampling = True
        logger.info(
            "Motif-3 device sampling: on (sample_on_device_mode {}): exact top-k / top-p / temperature, full-vocab "
            "Gumbel-max for top_p = 1 without top-k, host fallback of uncertified lanes; unseeded lanes from {}; "
            "logged every {} device steps",
            SAMPLE_ON_DEVICE_MODE,
            "OS entropy" if seed is None else f"vLLM's --seed {seed}",
            self.sampling_log_every,
        )

    def _sampled_tokens(self, sample: Any, lane_idx: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Validate the generator's lane-order sample and return the row-order tokens ``int32 [B, 1]`` (0 on padding
        rows), before anything is committed."""
        toks = getattr(sample, "tokens", None)
        if not isinstance(toks, torch.Tensor) or toks.reshape(-1).shape[0] != NUM_LANES:
            raise TypeError(f"the generator's device sample must carry tokens [{NUM_LANES}], got {type(sample)}")
        rows_t = toks.reshape(-1).index_select(0, lane_idx).to(torch.int64)
        rows_t = torch.where(active, rows_t, torch.zeros_like(rows_t))
        if bool(((rows_t < 0) | (rows_t >= self.vocab_size))[active].any()):
            raise ValueError(f"device-sampled tokens outside [0, {self.vocab_size}): {rows_t[active].tolist()}")
        return rows_t.to(torch.int32).reshape(-1, 1)

    def _sampled_output(self, sample: Any, tokens: torch.Tensor, lane_idx, active, sampling_params):
        """What the plugin reads from a device-sampled step: tokens ``[B, 1]``, plus the raw logprobs ``float32 [B]``
        when a row asked for them; counters and the periodic log."""
        st = self.sampling_stats
        st.device_steps += 1
        st.device_rows += int(active.sum())
        out: Any = tokens
        if wants_logprobs(sampling_params):
            lp = sample.logprobs.reshape(-1).index_select(0, lane_idx).to(torch.float32)
            out = (tokens, torch.where(active, lp, torch.zeros_like(lp)))
            st.logprob_steps += 1
        self._maybe_log_sampling()
        return out

    def _maybe_log_sampling(self) -> None:
        """The periodic counter line: every ``sampling_log_every`` decode steps of a device-sampling launch, device-
        sampled or host-routed (so a phase of host-only traffic shows up too)."""
        st = self.sampling_stats
        if self.sampling_log_every and (st.device_steps + st.host_steps) % self.sampling_log_every == 0:
            self.log_sampling_stats()

    def log_sampling_stats(self) -> None:
        """One log line with the bridge's and the generator's device-sampling counters (host fallbacks, Gumbel lanes,
        the sampler's flags and parameter uploads)."""
        gen = getattr(self.generator, "sampling_stats", None)
        logger.info(
            "Motif-3 device sampling: {}",
            json.dumps({**self.sampling_stats.as_dict(), **(gen() if callable(gen) else {})}, sort_keys=True),
        )

    def _decode_verify(
        self,
        kv: MotifKVCache,
        tokens,
        start_pos,
        page_table,
        slot_remap,
        num_valid_drafts,
        accepted_counts,
        spec_mode,
        enable_trace: bool,
        sampling_params=None,
    ):
        """A verify step (``SPEC_DECODE_CONTRACT.md`` §4a, §6): ``[B, 1+K]`` candidate block -> ``generator.
        decode_forward_spec(want_logits=False)`` -> ``VerifyOutput(argmax_ids [B, 1+K])``. ``sampling_params`` (a
        device-sampling launch sends them on every step its routing samples on device, verify steps included) do not
        change the argmax path: PS-1 keeps sampled rows out of verify steps, so every active row is greedy; a
        non-greedy row is counted (``sampling_stats.nongreedy_verify_rows``) and logged once.

        The generator chooses how to verify (``spec_verify``): the 32-lane trace with the drafts on idle lanes, or one
        64-row step (T64). The call carries neither logits nor ``sampling`` on purpose: in ``"auto"`` such a step can
        always take the 64-row trace, so bridge traffic never needs the overflow pass (review edit R-E6)."""
        from vllm_tt_plugin.spec_decode import (
            ACCEPT_MODE_ARGMAX_IDS,
            PLACEHOLDER_TOKEN_ID,
            VerifyOutput,
            check_spec_side_tensors,
        )

        if spec_mode != ACCEPT_MODE_ARGMAX_IDS:
            raise NotImplementedError(
                f"Motif-3 verifies in spec_mode {ACCEPT_MODE_ARGMAX_IDS!r} only, got {spec_mode!r}"
            )
        K = int(self.settings.spec_tokens)
        tok = torch.as_tensor(tokens)
        pos = torch.as_tensor(start_pos)
        if tok.ndim != 2 or tok.shape[1] != 1 + K or tuple(pos.shape) != tuple(tok.shape):
            raise ValueError(
                f"verify block must be tokens [B, {1 + K}] and start_pos [B, {1 + K}] (K = {K}), got "
                f"{tuple(tok.shape)} / {tuple(pos.shape)}"
            )
        rows = int(tok.shape[0])
        check_spec_side_tensors(num_valid_drafts, accepted_counts, rows, K, call="Motif-3 verify")
        # One pass over host lists: the same checks as tensor ops, without ~30 tiny torch calls per step.
        tok_l, pos_l, nv_l = tok.tolist(), pos.tolist(), num_valid_drafts.tolist()
        L, V, bs = int(self.settings.max_seq_len), self.vocab_size, kv.block_size
        active, drafted, anchor, need = [], [], [], []
        for i in range(rows):
            (p0, p1), d = pos_l[i], nv_l[i] > 0
            if not -1 <= p0 < L:
                raise ValueError(f"verify anchor positions must be -1 or in [0, {L}), got {[p[0] for p in pos_l]}")
            if d and p0 < 0:
                raise ValueError("verify: a draft on a padding row (num_valid_drafts > 0 where start_pos is -1)")
            if d and p1 != p0 + 1:
                raise ValueError(f"verify: a draft must sit at the anchor position + 1, got start_pos {pos_l}")
            if not d and p1 != -1:
                raise ValueError(f"verify: a padded draft column must carry position -1, got start_pos {pos_l}")
            if d and p1 >= L:
                raise ValueError(f"verify: a draft position reaches max_model_len {L}")
            if d and not 0 <= tok_l[i][1] < V:
                raise ValueError(f"verify: a draft token outside [0, {V}): {tok_l[i][1]}")
            active.append(p0 >= 0)
            drafted.append(d)
            anchor.append(p0)
            need.append(((p1 if d else p0) // bs + 1) if p0 >= 0 else 0)
        if sampling_params is not None:
            temps = _param_list(sampling_params, "temperature")
            hot = sum(1 for i in range(min(rows, len(temps))) if active[i] and float(temps[i]) >= 1e-5)
            self.sampling_stats.verify_steps_with_params += 1
            if hot:
                self.sampling_stats.nongreedy_verify_rows += hot
                if "nongreedy_verify" not in self._warned:
                    self._warned.add("nongreedy_verify")
                    logger.warning(
                        "Motif-3 verify step with {} non-greedy row(s): a verify returns the target argmax ids, so "
                        "those rows commit the argmax (the plugin's PS-1 should keep them out of verify steps)",
                        hot,
                    )
        pt = self._fit_page_table(page_table, kv, torch.tensor(need, dtype=torch.int64), "decode_forward (verify)")
        lanes = self._lanes.decode_lanes(rows, slot_remap)
        lane_tokens, lane_pos, lane_draft = [0] * NUM_LANES, [-1] * NUM_LANES, [-1] * NUM_LANES
        for i, lane in enumerate(lanes):
            if active[i]:
                lane_tokens[lane], lane_pos[lane] = tok_l[i][0], anchor[i]
                if drafted[i]:
                    lane_draft[lane] = tok_l[i][1]
        lane_idx = torch.tensor(lanes, dtype=torch.long)
        lane_pt = torch.zeros((NUM_LANES, pt.shape[1]), dtype=torch.int32)
        lane_pt[lane_idx] = pt
        batch = SpecDecodeBatch(
            tokens=torch.tensor(lane_tokens, dtype=torch.int32),
            positions=torch.tensor(lane_pos, dtype=torch.int32),
            draft_tokens=torch.tensor(lane_draft, dtype=torch.int32),
            page_table=lane_pt,
        )
        result = self.generator.decode_forward_spec(
            batch, kv_cache=kv.device_cache, enable_trace=enable_trace, want_logits=False
        )
        check_spec_result("MotifGenerator.decode_forward_spec", result, want_logits=False, vocab_size=self.vocab_size)
        retained = self._spec_retained(result, lanes, anchor, active, drafted)
        # Accepted: commit the slot move exactly once, then the retained ids.
        self._lanes.commit(slot_remap)
        self._retained.update(retained)
        self.spec_stats.verify_steps += 1
        self.spec_stats.drafts_verified += sum(drafted)
        argmax = result.argmax.tolist()
        ids = [
            [argmax[lane][0] if on else 0, argmax[lane][1] if d else PLACEHOLDER_TOKEN_ID]
            for lane, on, d in zip(lanes, active, drafted)
        ]
        return VerifyOutput(
            spec_mode=ACCEPT_MODE_ARGMAX_IDS, argmax_ids=torch.tensor(ids, dtype=torch.int32), hidden=None
        )

    def _spec_retained(
        self,
        result: SpecDecodeResult,
        lanes: Sequence[int],
        anchor_pos: Sequence[int],
        active: Sequence[bool],
        drafted: Sequence[bool],
    ) -> Dict[int, SpecRetained]:
        """The per-lane entries a step leaves for ``propose_draft_tokens``, validated (ids in the vocabulary on every
        active owner lane; column 1 on drafted lanes) before anything is committed."""
        out: Dict[int, SpecRetained] = {}
        V = self.vocab_size
        argmax, mtp = result.argmax.tolist(), result.mtp_argmax.tolist()  # host lists: no per-element tensor indexing
        for lane, on, d, p in zip(lanes, active, drafted, anchor_pos):
            if not on:
                continue
            cols = 2 if d else 1
            a, m = tuple(argmax[lane][:cols]), tuple(mtp[lane][:cols])
            if not all(0 <= x < V for x in a + m):
                raise ValueError(
                    f"MotifGenerator.decode_forward_spec: lane {lane} returned argmax {a} / MTP argmax {m} outside "
                    f"[0, {V})"
                )
            out[int(lane)] = SpecRetained(int(p), a, m)
        return out

    # ---- drafting policy (T64: docs/p5_t64/P5_T64_DESIGN.md §4.5, §4.7; review edits R-E3, R-E9) --------------
    def spec_acceptance(self) -> float:
        """alpha_hat, the acceptance the drafting policy works with: the running acceptance of this bridge's verified
        drafts pulled toward the prior ``settings.spec_alpha_prior`` (0.85) with weight ``SPEC_ALPHA_PRIOR_WEIGHT``
        (64) (:meth:`SpecStats.acceptance`, review edit R-E3). It is the prior before any draft was verified, so a
        server whose first traffic is a 32-request burst still drafts: under the idle-lane budget alone it would never
        verify a draft, never measure alpha, and the 64-row verify would never engage."""
        return self.spec_stats.acceptance(self.settings.spec_alpha_prior)

    def all_lanes_threshold(self, acceptance: Optional[float] = None) -> int:
        """``c*``: the fewest live lanes from which the generator lets every live lane draft
        (``generator.drafts_all_lanes``), at ``acceptance`` (default :meth:`spec_acceptance`); ``WIDE_MIN_LANES_NEVER``
        (33) = never. Probed with the first ``c`` lanes of a fresh lane map, ``c`` = 1 .. 32 (host only, no device
        work). Not used to serve: every proposal asks the generator itself (:meth:`_draft_budget`)."""
        a = self.spec_acceptance() if acceptance is None else float(acceptance)
        lanes = LaneMap(NUM_LANES).slot_to_lane
        for c in range(1, NUM_LANES + 1):
            if bool(self.generator.drafts_all_lanes(lanes[:c], acceptance=a)):
                return c
        return WIDE_MIN_LANES_NEVER

    def drafting_threshold_text(self, acceptance: Optional[float] = None) -> str:
        """``c*=<n>`` for the log lines: ``c*=n/a`` without speculation, ``c*=never`` when no live-lane count makes the
        generator draft every lane, and in ``spec_verify="auto"`` what decides it: ``(MOTIF3_WIDE_MIN_LANES=<n>)`` or
        ``(alpha <a>)`` (:meth:`all_lanes_threshold` at ``acceptance``, default :meth:`spec_acceptance`)."""
        if not self._spec:
            return "c*=n/a"
        a = self.spec_acceptance() if acceptance is None else float(acceptance)
        c = self.all_lanes_threshold(a)
        text = f"c*={'never' if c > NUM_LANES else c}"
        s = self.settings
        if s.spec_verify == "auto":
            why = f"MOTIF3_WIDE_MIN_LANES={s.wide_min_lanes}" if s.wide_min_lanes is not None else f"alpha {a:.3f}"
            text += f" ({why})"
        return text

    def _idle_lane_budget(self, live_lanes: Sequence[int]) -> Callable[[int], bool]:
        """``take(owner_lane) -> bool``: whether the next step still has an idle lane for one more draft. A draft runs
        on a lane no request uses (packed verify, features design §3.8.2): with KV-R any idle lane (every chip holds
        every lane's KV), so ``32 - live`` drafts in all; without it an idle lane of the owner's DP row, so ``8 -
        active`` per DP row. A draft past the budget would cost the generator a second trace replay (its overflow
        pass), so it is declined instead."""
        if self.settings.kv_replicated:
            left = {None: NUM_LANES - len(set(live_lanes))}

            def key(lane: int):
                return None

        else:
            left = {g: LANES_PER_GROUP for g in range(NUM_DP_GROUPS)}
            for lane in set(live_lanes):
                left[lane // LANES_PER_GROUP] -= 1

            def key(lane: int):
                return int(lane) // LANES_PER_GROUP

        def take(lane: int) -> bool:
            k = key(lane)
            if left[k] <= 0:
                return False
            left[k] -= 1
            return True

        return take

    def _draft_budget(self, live_lanes: Sequence[int]) -> Callable[[int], bool]:
        """``take(owner_lane) -> bool``: whether the next step can verify one more draft.

        The generator answers first: ``generator.drafts_all_lanes(live_lanes, acceptance=alpha_hat)``, alpha_hat =
        :meth:`spec_acceptance` (review edits R-E3, R-E9). True: every live lane may draft; the generator verifies
        the drafts that do not fit idle lanes in one 64-row step (``spec_verify`` "wide", or "auto" from ``c*`` live
        lanes on). False (``MotifGenerator``'s default, ``spec_verify="packed"``, "auto" below ``c*``): the idle-lane
        budget, unchanged (:meth:`_idle_lane_budget`; without KV-R its per-DP-row budget, which a lane count could
        not express). Counted in ``spec_stats.all_lane_proposals`` / ``drafts_beyond_budget``."""
        live = tuple(dict.fromkeys(int(lane) for lane in live_lanes))
        budget = self._idle_lane_budget(live)
        if not live or not bool(self.generator.drafts_all_lanes(live, acceptance=self.spec_acceptance())):
            return budget
        stats = self.spec_stats
        stats.all_lane_proposals += 1

        def take_all(lane: int) -> bool:
            if not budget(lane):
                stats.drafts_beyond_budget += 1
            return True

        return take_all

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        """The model-owned drafter (``SPEC_DECODE_CONTRACT.md`` §4b; features design §3.8.3, §3.9 item 4): host only.

        Called by the plugin after every decode step of a speculating launch (never after a prefill), with that step's
        rows (padding included): ``committed_tokens [B, 1+K]`` (the committed prefix, ``-1`` after it),
        ``committed_positions [B, 1+K]`` (column 0 = the first committed token's position, -1 on padding rows) and
        ``accepted_counts [B]`` (how many tokens each row committed). The MTP layer already ran on every row of that
        step (``SpecRetained``), so the draft for the next step is ``mtp[count - 1]`` of the row's lane (the remap of
        the step is committed, so row ``i`` is state slot ``i``):

        * ordinary step or rejected draft (count 1, committed ``a0``): ``m0`` (for ``n + 2``);
        * accepted draft (count 2, committed ``(d = a0, a1)``): ``m1`` (for ``n + 3``).

        A row is declined (``num_valid`` 0, always legal) when it is padding, when the lane holds no step anchored at
        ``committed_positions[i, 0] - 1``, when the committed tokens are not the retained argmax (a sampled row: its
        next anchor is not what the MTP drafted after), or when the next step cannot verify its draft
        (:meth:`_draft_budget`: the idle-lane budget, unless the generator lets every live lane draft because it
        verifies them in one 64-row step; rows are visited from a rotating start so capped drafting is fair). The
        budget is asked after this step's verdicts are counted, so the acceptance it uses includes them. ``hidden`` is
        ignored (the MTP state stays on device: ``spec_hidden_handoff`` ``on_device``).

        Returns ``DraftOutput(draft_token_ids=int32 [B, K], num_valid=int32 [B])``."""
        from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, DraftOutput

        if not self._spec:
            raise RuntimeError("propose_draft_tokens on a launch without speculative decoding (spec_tokens = 0)")
        K = int(num_drafts)
        if K != int(self.settings.spec_tokens):
            raise ValueError(f"propose_draft_tokens: K = {K}, but this launch drafts {self.settings.spec_tokens}")
        committed = torch.as_tensor(committed_tokens)
        positions = torch.as_tensor(committed_positions)
        counts = torch.as_tensor(accepted_counts).reshape(-1)
        if committed.ndim != 2 or committed.shape[1] != 1 + K or tuple(positions.shape) != tuple(committed.shape):
            raise ValueError(
                f"propose_draft_tokens: committed tokens / positions must be [B, {1 + K}], got "
                f"{tuple(committed.shape)} / {tuple(positions.shape)}"
            )
        B = int(committed.shape[0])
        if not 1 <= B <= self._lanes.num_slots or counts.shape[0] != B:
            raise ValueError(
                f"propose_draft_tokens: {B} rows (counts {tuple(counts.shape)}) for {self._lanes.num_slots} slots"
            )
        first_pos = positions[:, 0].tolist()  # host lists: no per-element tensor indexing on the hot path
        count_l, committed_l = counts.tolist(), committed.tolist()
        live = [p >= 0 for p in first_pos]
        if any(on and not 1 <= c <= 1 + K for on, c in zip(live, count_l)):
            raise ValueError(f"propose_draft_tokens: accepted_counts outside [1, {1 + K}]: {count_l}")
        lanes = self._lanes.slot_to_lane[:B]
        draft_l = [PLACEHOLDER_TOKEN_ID] * B
        valid_l = [0] * B
        stats = self.spec_stats
        stats.proposals += 1
        first = self._propose_calls % B
        self._propose_calls += 1
        candidates: List[Tuple[int, int]] = []  # (row, draft) in the rotating visit order
        for k in range(B):
            i = (first + k) % B
            if not live[i]:
                continue
            st = self._retained.get(lanes[i])
            c = int(count_l[i])
            if st is None or first_pos[i] != st.pos0 + 1 or c > len(st.argmax):
                stats.declined_stale += 1
                continue
            if st.drafted:
                if c == 2:
                    stats.accepted += 1
                else:
                    stats.rejected += 1
            if tuple(committed_l[i][:c]) != st.argmax[:c]:
                stats.declined_mismatch += 1
                continue
            candidates.append((i, st.mtp[c - 1]))
        if candidates:  # after this step's verdicts: the acceptance the budget asks with includes them
            take = self._draft_budget([lane for lane, on in zip(lanes, live) if on])
            for i, draft in candidates:
                if not take(lanes[i]):
                    stats.declined_budget += 1
                    continue
                draft_l[i] = draft
                valid_l[i] = 1
                stats.drafts_offered += 1
        drafts = torch.tensor(draft_l, dtype=torch.int32).reshape(B, K)
        return DraftOutput(draft_token_ids=drafts, num_valid=torch.tensor(valid_l, dtype=torch.int32))

    @staticmethod
    def _is_host_output(tt_out) -> bool:
        if isinstance(tt_out, torch.Tensor):
            return True
        return isinstance(tt_out, tuple) and all(t is None or isinstance(t, torch.Tensor) for t in tt_out)

    def read_decode_output(self, tt_out, async_read=False):
        """``decode_forward`` returns host tensors (logits, device-sampled tokens, or ``(tokens, logprobs)``), or with
        ``MOTIF3_ASYNC_DECODE=on`` a :class:`MotifPendingDecode`: ``async_read`` hands it back with no events (the
        read happens in :meth:`process_decode_output_host`, or earlier when the next step needs its tokens); without
        ``async_read`` it is resolved now."""
        if isinstance(tt_out, MotifPendingDecode):
            return (tt_out, []) if async_read else tt_out.output()
        if not self._is_host_output(tt_out):
            raise TypeError(f"expected the host output decode_forward returned, got {type(tt_out)}")
        return (tt_out, []) if async_read else tt_out

    def process_decode_output_host(self, tt_out, is_tokens=False):
        """Host logits (``is_tokens=False``) or the device-sampled tokens / ``(tokens, logprobs)`` (``is_tokens``),
        returned unchanged (``decode_forward`` already read them from the device), or a :class:`MotifPendingDecode`
        resolved (its one read, if no later step has made it)."""
        if isinstance(tt_out, MotifPendingDecode):
            if not is_tokens:
                raise TypeError("a deferred Motif-3 decode step holds device-sampled tokens, not logits")
            return tt_out.output()
        if not self._is_host_output(tt_out):
            raise TypeError(f"expected host {'tokens' if is_tokens else 'logits'}, got {type(tt_out)}")
        return tt_out

    # ---- warmup ---------------------------------------------------------------------------------------------------
    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device=False, **kwargs):
        """Plugin phase 1 (eager) and, only with ``trace_mode="all"``, phase 2 (``model_runner.py:3727-3781``).

        Compiles every prefill shape (every ``(path, bucket)`` up to the span cap with resumed prefill) before any
        decode trace exists (a prefill shape compiled after capture can corrupt the trace). ``can_sample_on_device``
        (``sample_on_device_mode`` "all") is refused: prefill rows sample on the host (``"decode_only"``).
        """
        kv = self._check_kv(kv_cache)
        if can_sample_on_device:
            check_sample_on_device_mode("all")  # raises: prefill rows sample on the host
        self.generator.warmup_prefill(kv_cache=kv.device_cache, enable_trace=bool(enable_trace))
        self._prefill_warmed = True

    def warmup_model_decode(
        self, kv_cache, enable_trace, max_batch_size, num_blocks, can_sample_on_device=False, **kwargs
    ):
        """Eager decode warmup, then decode trace capture (``enable_trace=True``; the spec trace when speculating).

        ``num_blocks`` is the plugin's page-table width (``max_num_blocks_per_req``), fixed for the server's life;
        ``max_batch_size`` is ``max_num_seqs``. ``can_sample_on_device`` (``sample_on_device_mode`` "decode_only"):
        the generator builds its exact device sampler before the eager warmup, so the warmup compiles it and the
        capture puts it into the decode trace (plain or spec). With ``MOTIF3_SPEC_VERIFY=auto`` the generator stages,
        warms and captures its two decode traces here, once (the 32-lane spec trace with the sampler, then the 64-row
        verify trace; ``docs/p5_t64/P5_T64_DESIGN.md`` §2.3-§2.4); ``wide``: the 64-row trace alone.
        """
        kv = self._check_kv(kv_cache)
        if can_sample_on_device:
            self._enable_device_sampling()
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
        """A request finished or was preempted while owning state ``slot`` (``model_runner.py:851-868``): drop the
        lane's retained speculation entry and tell the generator."""
        lane = self._lanes.lane_of_slot(slot)
        self._retained.pop(lane, None)
        r = self._resident
        if r is not None and any(on and l == lane for l, on in zip(r.lanes, r.active.tolist())):
            # a request of the resident decode step left: the next decode reloads (a layout change), and a steady
            # step after this raises. A request that never decoded (e.g. max_tokens 1, finished at its prefill) is
            # released while the other rows keep their steady chain.
            self._resident = None
        self.generator.release_lane(lane)

    def release_persistent_capture(self):
        """Shutdown, mesh still open (``model_runner.py:392-419``): log the bridge's speculation counters (with
        alpha_hat and ``c*`` at it) and device-sampling counters and the generator's own counters
        (``MotifGenerator.stats``: prefill calls / rows / chunks / sp1 chunks / recomputed rows / MTP fills, decode and
        spec steps, drafts, packed drafts, cross-row partners, overflow passes, device-sampled steps, host fallbacks),
        then free the decode traces (and the sampler's trace outputs)."""
        if self._spec:
            alpha = self.spec_acceptance()
            try:
                drafting = self.drafting_threshold_text(alpha)
            except Exception as exc:  # the log must not keep release_traces from running
                drafting = f"c*=? ({exc!r})"
            logger.info("Motif-3 speculation: {} alpha_hat={:.4f} {}", self.spec_stats.as_dict(), alpha, drafting)
        if self._device_sampling:
            self.log_sampling_stats()
        stats = getattr(self.generator, "stats", None)
        if isinstance(stats, Mapping) and stats:
            logger.info("Motif-3 generator: {}", dict(stats))
        self.generator.release_traces()

    def close(self):
        self.generator.close()


__all__ = [
    "ARCHITECTURE",
    "CHECKPOINT_INDEX",
    "DEFAULT_GENERATOR_CLASS",
    "DEFAULT_KV_POOL_TOKENS",
    "FEATURE_SWITCH_DEFAULT",
    "FEATURE_VLLM_ARGS",
    "KV_POOL_ALIGNMENT",
    "L1_SMALL_SIZE",
    "LaneMap",
    "MAIN_CLASS",
    "MAX_KV_POOL_TOKENS",
    "MODEL_OWNED_DRAFTER",
    "MTP_WEIGHT_PREFIX",
    "MotifForCausalLM",
    "MotifKVCache",
    "MotifPendingDecode",
    "NULL_BLOCK_RESERVE_TOKENS",
    "SERVING_TT_CONFIG",
    "SPECULATIVE_CONFIG",
    "SPEC_HIDDEN_HANDOFF",
    "SPEC_LANES_PER_REQUEST",
    "SPEC_METHOD",
    "SPEC_REQUIREMENTS",
    "SpecRetained",
    "SpecStats",
    "TT_MODEL_CLASS_OVERRIDES",
    "DEVICE_SAMPLING_SWITCH",
    "DEVICE_SAMPLING_TT_CONFIG",
    "SAMPLE_ON_DEVICE_MODE",
    "SamplingStats",
    "check_sample_on_device_mode",
    "check_serving_config",
    "device_sampling_switch",
    "feature_switches",
    "feature_vllm_args",
    "lane_sampling_lists",
    "launch_chunk_budget",
    "wants_logprobs",
    "kv_max_bytes_per_chip",
    "kv_pool_tokens_from_env",
    "model_capabilities_from_env",
    "mtp_extra_bytes_per_token",
    "mtp_weights_status",
    "plugin_num_blocks",
    "precheck_generator_class",
    "serving_additional_config",
    "serving_config_of",
    "spec_plan_supports_speculable_rows",
    "validate_block_size",
]
