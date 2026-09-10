# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Solar-Open-100B helpers for the vLLM wrapper ``SolarOpenForCausalLM`` (``models/tt_transformers/tt/generator_vllm.py``).

That module imports vllm unconditionally at the top, so everything Solar-specific the wrapper needs lives here,
where it is importable and unit-testable without vllm (``tests/unit/test_vllm_wrapper_import.py``): the request
validation (1x8 mesh, TP=8, DP=1, batch <= 32, 131072 positions), the 48-layer full-attention KV-cache spec, the
all-user token capacity derived from the KV budget in ``tt/common.py``, the paged KV pool allocator (zeros written
straight to the device with ``ttnn.from_torch`` - no tensorbins - after the budget guard), the generation stop set,
the chat-template kwargs and the registration shim for Upstage's reasoning / tool parsers.

STATUS: written against the standalone TT plugin (tenstorrent/vllm-tt-plugin, ``vllm_tt_plugin``; upstream vLLM 0.25.1
built ``VLLM_TARGET_DEVICE=empty`` inside this tree's python_env since 2026-09-10) and the text-model wrappers of that
module. Host-verified with the real vllm import (``tests/unit/test_vllm_wrapper_import.py``: the KV pool arithmetic
the plugin applies, Upstage's parsers imported through the compat shims and instantiated on the Solar tokenizer);
NOT yet run against a live vLLM server on the device. The functions that need vllm (shims, parser registration)
import it lazily and fail with a clear message without it.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.solar_open.config import MoEOptions
from models.demos.solar_open.tt.common import KV_BUDGET_ENV, KV_BYTES_PER_ELEMENT, check_kv_budget, kv_budget_tokens
from models.demos.solar_open.tt.model_config import DEFAULT_HF_MODEL, MODEL_NAME
from models.demos.solar_open.utils.general_utils import get_layer_types
from models.tt_transformers.tt.common import PagedAttentionConfig

# ``hf_config.architectures[0]`` of the checkpoint. The TT plugin (tenstorrent/vllm-tt-plugin) registers it, prefixed,
# as ``TTSolarOpenForCausalLM`` from an EXTRA_MODELS_DIR bundle in every vLLM process (platform.py
# ``_register_models_from_extra_dir``): ``$EXTRA_MODELS_DIR/solar_open/vllm_metadata.json`` =
#   {"arch": "SolarOpenForCausalLM", "main_class": "models.tt_transformers.tt.generator_vllm:SolarOpenForCausalLM"}
# (tt-inference-server branch changh95/solar-open ships it under vllm-tt-metal/extra_models/solar_open/).
VLLM_ARCHITECTURE = "SolarOpenForCausalLM"
VLLM_WRAPPER_MODULE = "models.tt_transformers.tt.generator_vllm"
VLLM_WRAPPER_CLASS = "SolarOpenForCausalLM"

# The validated deployment (README): a logical 1x8 mesh, tensor parallel 8, no data parallelism, at most 32 decode
# users per mesh row (nlp_*_heads_decode / on-device sampling lanes), RoPE tables over 131072 positions.
MESH_SHAPE = (1, 8)
TENSOR_PARALLEL = 8
MAX_BATCH_SIZE = 32
MAX_SEQ_LEN = 131072
NUM_LAYERS = 48
NUM_KV_HEADS = 8
HEAD_DIM = 128
# Page block of the demo (create_tt_page_table, test_multi_user_consistency: 4096 x 64 for 32 x 8K); vLLM --block-size.
KV_BLOCK_SIZE = 64
# 48 layers x (K + V) x 1 KV head per device x 128 x 1.0625 B (bfp8) = 13,056 B per token per device.
KV_BYTES_PER_TOKEN_PER_DEVICE = NUM_LAYERS * 2 * (NUM_KV_HEADS // TENSOR_PARALLEL) * HEAD_DIM * KV_BYTES_PER_ELEMENT

# Class-level capability dict of the wrapper (the plugin snapshots it before any instance exists; a missing key
# means "not supported"). Mirrored literally in generator_vllm.py - the smoke test keeps the two in sync.
MODEL_CAPABILITIES = {
    "supports_prefix_caching": False,  # a nonzero start_pos (chunked-SDPA resume) is not validated for this model
    "supports_async_decode": True,  # Generator.decode_forward(read_from_device=False) + read_decode_output
    "supports_sample_on_device": True,  # TTSampling at TP=8 over the pow2-padded per-device vocab (32768)
    "max_device_top_k": 32,  # TTSampling.max_top_k default; the demo clamps larger top_k requests
}

# generation_config.json eos_token_id: <|endoftext|> 2, <|flush|> 24, <|calls|> 25 (ModelArgs.stop_token_ids reads
# the file; this constant is the documented fallback when a ModelArgs was built with dummy weights).
GENERATION_STOP_TOKEN_IDS = (2, 24, 25)

# chat_template.jinja parameters (l.2-5): the template defaults, overridable per request through
# ``chat_template_kwargs`` (e.g. {"reasoning_effort": "low"}). ``strftime_now`` (the dated provider system prompt)
# is a Jinja global vLLM's template environment provides.
CHAT_TEMPLATE_KWARG_DEFAULTS = {
    "reasoning_effort": "high",  # "low" / "minimal" prepend an empty <|think|><|end|> block
    "default_system_prompt": True,  # dated "Provider System Prompt" block
    "think_render_option": "lastthink",  # which earlier assistant think blocks are re-rendered ("all" | "lastthink")
}

# Upstage ships the vLLM parsers in the HF repo without a register decorator (its vLLM fork registers them
# internally under "solar_open"); register_vllm_parsers() below does the same on a stock vLLM + TT plugin build.
REASONING_PARSER_NAME = "solar_open"
TOOL_PARSER_NAME = "solar_open"
REASONING_PARSER_FILE = "solar_open_reasoning_parser.py"
TOOL_PARSER_FILE = "solar_open_tool_parser.py"


def vllm_available() -> bool:
    """True when the real vllm package is importable (a bare directory called ``vllm`` on sys.path is not)."""
    spec = importlib.util.find_spec("vllm")
    return spec is not None and spec.origin is not None


def plugin_headroom_tokens(max_num_seqs: int = MAX_BATCH_SIZE, block_size: int = KV_BLOCK_SIZE) -> int:
    """The plugin's own KV padding, ``block_size x max_num_seqs`` tokens (2,048 at 64 x 32).

    ``vllm_tt_plugin/worker.py::get_num_available_blocks_tt`` adds one extra block per user (vLLM's worst-case
    block-allocation heuristic) to whatever ``get_max_tokens_all_users`` returns before it derives the block count.
    """
    return int(block_size) * int(max_num_seqs)


def kv_budget_tokens_solar(moe_options: MoEOptions = None, block_size: int = KV_BLOCK_SIZE) -> int:
    """The KV budget in tokens with the Solar-Open shapes: what ``check_kv_budget`` enforces at allocation.

    ``SOLAR_OPEN_KV_BUDGET_GIB`` (or the per-expert-dtype default of ``tt/common.py``) over 13,056 B per token per
    device, in whole page blocks - 8 GiB -> 657,920 tokens, 13 GiB -> 1,069,120, 14 GiB -> 1,151,360.
    """
    moe_options = moe_options or MoEOptions.from_env()
    return kv_budget_tokens(
        moe_options,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        n_layers=NUM_LAYERS,
        tensor_parallel=TENSOR_PARALLEL,
        block_size=block_size,
    )


def max_tokens_all_users(
    moe_options: MoEOptions = None, block_size: int = KV_BLOCK_SIZE, max_num_seqs: int = MAX_BATCH_SIZE
) -> int:
    """All-user KV capacity (tokens) to report to the plugin: the KV budget MINUS the plugin's headroom.

    The plugin adds ``block_size x max_num_seqs`` to this figure and takes ``ceil(/block_size)`` blocks
    (``plugin_kv_pool_blocks``); ``allocate_paged_kv_cache`` then re-checks that pool against the same budget. So the
    figure handed out here is ``kv_budget_tokens_solar() - plugin_headroom_tokens()``: 8 GiB at 32 users -> 655,872
    tokens; the plugin sizes 657,920 / 64 = 10,280 blocks = 8.00 GiB, which passes the guard. (Returning the full
    budget made the pool 10,312 blocks = 8.025 GiB > 8.0 GiB -> ValueError before any weight was loaded.)

    ``max_num_seqs`` is the plugin's per-lane batch (vLLM ``--max-num-seqs``; the wrapper forwards the kwarg the
    plugin passes, defaulting to the 32-user cap). ``block_size`` must be the served ``--block-size`` (64, the only
    value the demo validates; the plugin does not pass it). ``max_num_seqs=0`` returns the raw budget figure.
    """
    budget = kv_budget_tokens_solar(moe_options, block_size=block_size)
    headroom = plugin_headroom_tokens(max_num_seqs, block_size)
    if headroom >= budget:
        raise ValueError(
            f"The plugin's KV headroom ({max_num_seqs} users x {block_size}-token blocks = {headroom} tokens) leaves "
            f"nothing of the {budget}-token KV budget; lower --max-num-seqs or raise {KV_BUDGET_ENV}"
        )
    return (budget - headroom) // block_size * block_size


def plugin_kv_pool_blocks(max_tokens: int, max_num_seqs: int = MAX_BATCH_SIZE, block_size: int = KV_BLOCK_SIZE) -> int:
    """Blocks the plugin allocates from a ``get_max_tokens_all_users`` figure: ``ceil((max_tokens +
    block_size x max_num_seqs) / block_size)`` (``worker.py::get_num_available_blocks_tt``; the sliding-window term
    does not apply to Solar's single full-attention group). The tests hold this pool against ``check_kv_budget``."""
    return -(-(int(max_tokens) + plugin_headroom_tokens(max_num_seqs, block_size)) // int(block_size))


def validate_vllm_model_request(
    mesh_shape,
    max_batch_size,
    max_seq_len,
    tt_data_parallel=1,
    hf_name_or_path="",
    optimizations=None,
):
    """Refuse configurations this tree does not support BEFORE ``create_tt_model`` starts the 393 GB host load.

    Raises ValueError. Batch sizes that do not map onto the per-user decode grid (11, 13, ...) are rejected later by
    ``Model.__init__`` (also before any weight is read).
    """
    if optimizations is not None:
        raise ValueError(
            f"Solar-Open has no optimization presets (ModelArgs ignores them); got optimizations={optimizations!r}"
        )
    shape = tuple(int(s) for s in mesh_shape)
    if shape != MESH_SHAPE:
        raise ValueError(
            f"Solar-Open is validated on the {MESH_SHAPE} mesh with TP={TENSOR_PARALLEL} only (MESH_DEVICE=P150x8); "
            f"got mesh shape {shape}"
        )
    if tt_data_parallel != 1:
        raise ValueError(
            f"Solar-Open runs TP={TENSOR_PARALLEL} over the whole 1x8 mesh; data parallelism is not supported "
            f"(tt_data_parallel={tt_data_parallel})"
        )
    if not 1 <= int(max_batch_size) <= MAX_BATCH_SIZE:
        raise ValueError(
            f"max_batch_size={max_batch_size}: a single mesh row decodes 1..{MAX_BATCH_SIZE} users "
            "(nlp_*_heads_decode and the on-device sampling lanes cap at 32); set vLLM --max-num-seqs <= 32"
        )
    if not 1 <= int(max_seq_len) <= MAX_SEQ_LEN:
        raise ValueError(
            f"max_seq_len={max_seq_len}: the RoPE tables cover 1..{MAX_SEQ_LEN} positions "
            "(max_position_embeddings); lower vLLM --max-model-len"
        )
    name = str(hf_name_or_path or "")
    if not name:
        logger.warning("hf_config._name_or_path is empty; cannot cross-check the vLLM model against HF_MODEL")
    elif MODEL_NAME.replace("-", "") not in name.replace("-", ""):
        raise ValueError(
            f"The model vLLM was started with ({name}) is not {MODEL_NAME}; the TT weights come from HF_MODEL "
            f"(basename {MODEL_NAME}), so both must name the same checkpoint"
        )


def full_attention_layer_types(hf_config) -> list:
    """``layer_types`` for the KV-cache spec: 48 x "full_attention" (SolarOpenConfig has no such field)."""
    layer_types = get_layer_types(hf_config)
    other = sorted({lt for lt in layer_types if lt != "full_attention"})
    if other:
        raise ValueError(f"Solar-Open layers are all full attention; unexpected layer_types {other} in {layer_types}")
    return layer_types


def kv_cache_spec_layer_names(num_layers: int) -> list:
    """vLLM attention-layer names (``model.layers.<i>.self_attn``) the plugin's ``_parse_layer_index`` maps back."""
    return [f"model.layers.{i}.self_attn" for i in range(int(num_layers))]


def build_full_attention_kv_cache_spec(num_layers: int, spec_cls, **common) -> dict:
    """One ``spec_cls(**common)`` (vLLM's ``FullAttentionSpec``) per layer, keyed by the vLLM layer name.

    ``spec_cls`` is passed in so this stays importable without vllm; ``common`` carries block_size, num_kv_heads,
    head_size and dtype exactly as ``HybridAttentionForCausalLM.get_kv_cache_spec`` builds them.
    """
    return {name: spec_cls(**common) for name in kv_cache_spec_layer_names(num_layers)}


def per_device_kv_heads(kv_heads: int, tensor_parallel: int = TENSOR_PARALLEL) -> int:
    """KV heads of one device's cache tensor from the head count vLLM puts into ``kv_cache_shape``.

    The TT worker divides the model's 8 KV heads by TP (1 at TP=8, the shape attention/kv_cache.py allocates); a
    plugin handing over the undivided 8 is accepted too (logged), anything else is a contract violation.
    """
    expected = max(1, NUM_KV_HEADS // tensor_parallel)
    kv_heads = int(kv_heads)
    if kv_heads == expected:
        return expected
    if kv_heads == NUM_KV_HEADS:
        logger.warning(
            f"kv_cache_shape carries the undivided {NUM_KV_HEADS} KV heads; allocating {expected} per device "
            f"(TP={tensor_parallel})"
        )
        return expected
    raise ValueError(
        f"kv_cache_shape has {kv_heads} KV heads; Solar-Open at TP={tensor_parallel} expects {expected} per device "
        f"(or the undivided {NUM_KV_HEADS})"
    )


def allocate_paged_kv_cache(
    models,
    kv_cache_shape,
    num_layers: int,
    moe_options: MoEOptions,
    vllm_dtype=None,
    cache_dtype=ttnn.bfloat8_b,
):
    """Paged K/V pools for vLLM: ``list[submesh][layer][k, v]`` of replicated bfp8 DRAM tensors.

    ``kv_cache_shape = (max_num_blocks, kv_heads, block_size, head_dim)`` as the TT plugin's ``initialize_kv_cache``
    passes it. The budget guard runs here because ``create_tt_model(create_kv_cache=False)`` skipped it and vLLM
    sizes the pool from ``max_tokens_all_users``. The tensors are zero-initialised straight on the device with
    ``ttnn.from_torch`` exactly like ``attention/kv_cache.py`` - NOT through ``generator_vllm.allocate_vllm_kv_cache``,
    whose ``ttnn.as_tensor(cache_file_name=...)`` would write ~27 GB of zero tensorbins per pool shape into
    TT_CACHE_PATH (the 2026-09-07 tech-lead decision that removed the KV cache stems). ``vllm_dtype`` (vLLM's torch
    dtype for the cache) is informational: the Solar KV cache is bfp8 on the device.
    """
    if len(kv_cache_shape) != 4:
        raise ValueError(
            f"kv_cache_shape must be (max_num_blocks, kv_heads, block_size, head_dim); got {kv_cache_shape}"
        )
    max_num_blocks, kv_heads, block_size, head_dim = (int(s) for s in kv_cache_shape)
    if head_dim != HEAD_DIM:
        raise ValueError(f"kv_cache_shape head_dim {head_dim} != {HEAD_DIM}")
    if block_size <= 0 or block_size % 32 != 0:
        raise ValueError(f"block_size {block_size} must be a positive multiple of 32 (paged ops work in tiles); use 64")
    if max_num_blocks <= 0 or num_layers <= 0:
        raise ValueError(f"max_num_blocks={max_num_blocks} and num_layers={num_layers} must be positive")
    tensor_parallel = int(models[0].mesh_device.shape[1])
    heads = per_device_kv_heads(kv_heads, tensor_parallel)
    check_kv_budget(
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        n_layers=int(num_layers),
        paged_attention_config=PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks),
        tensor_parallel=tensor_parallel,
        moe_options=moe_options,
    )
    if vllm_dtype is not None:
        logger.info(f"vLLM asked for a {vllm_dtype} KV cache; Solar-Open keeps its paged cache in {cache_dtype}")
    shape = [max_num_blocks, heads, block_size, head_dim]
    zeros = torch.zeros(shape)
    kv_cache = []
    for model in models:
        mesh_device = model.mesh_device
        layers = []
        for _ in range(int(num_layers)):
            layers.append(
                [
                    ttnn.from_torch(
                        zeros,
                        dtype=cache_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    )
                    for _ in ("k", "v")
                ]
            )
        kv_cache.append(layers)
    logger.info(
        f"Allocated the vLLM paged KV cache: {len(kv_cache)} submesh x {num_layers} layers x K/V of {shape} "
        f"{cache_dtype} (replicated)"
    )
    return kv_cache


def stop_token_ids_for_vllm(model_args) -> list:
    """Sorted generation stop set ({2, 24, 25}) from ``ModelArgs.stop_token_ids``; the documented fallback when empty.

    vLLM stops on ``generation_config.json``'s ``eos_token_id`` itself (default ``--generation-config auto``); the
    wrapper exposes this for the plugin / parity checks and never truncates generations on its own.
    """
    ids = sorted(int(t) for t in (getattr(model_args, "stop_token_ids", None) or ()))
    if not ids:
        logger.warning(f"ModelArgs has no stop token ids (dummy weights?); using {GENERATION_STOP_TOKEN_IDS}")
        return list(GENERATION_STOP_TOKEN_IDS)
    return ids


def chat_template_kwargs(**overrides) -> dict:
    """Template kwargs the demo applies (``ModelArgs.encode_prompt``): template defaults, then
    ``SOLAR_OPEN_REASONING_EFFORT`` / ``SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT``, then explicit overrides.

    For vLLM these go into the request's ``chat_template_kwargs`` (or the server-wide default-chat-template-kwargs
    option where the installed vLLM has it); the demo's ``low`` is a demo choice, not a wrapper default.
    """
    kwargs = dict(CHAT_TEMPLATE_KWARG_DEFAULTS)
    kwargs["reasoning_effort"] = os.getenv("SOLAR_OPEN_REASONING_EFFORT", kwargs["reasoning_effort"])
    kwargs["default_system_prompt"] = os.getenv("SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT", "1") == "1"
    kwargs.update(overrides)
    return kwargs


def _resolve_model_dir(model_dir=None) -> Path:
    path = Path(model_dir or os.getenv("HF_MODEL", DEFAULT_HF_MODEL))
    if not path.is_dir():
        raise FileNotFoundError(
            f"{path} is not a directory; point HF_MODEL (or model_dir) at the Solar-Open-100B snapshot directory "
            f"that holds {REASONING_PARSER_FILE} / {TOOL_PARSER_FILE} (HF hub snapshots are hashes: symlink one)"
        )
    return path


def _import_python_file(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; the Solar-Open HF repo ships it next to the safetensors shards")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_tool_parser_manager():
    # The manager moved between vLLM releases; try the current location first.
    try:
        from vllm.tool_parsers import ToolParserManager
    except ImportError:
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    return ToolParserManager


# Upstage's parser files were written against UpstageAI/vllm@v0.12.0-solar-open. vLLM >= 0.13 moved the modules they
# import: ``vllm.entrypoints.openai.protocol`` was split into per-endpoint protocol modules and
# ``vllm.entrypoints.openai.tool_parsers`` became ``vllm.tool_parsers``. install_vllm_compat_shims() re-creates the
# legacy module paths in sys.modules from their current homes (a no-op where the legacy modules still exist).
# tt-inference-server routes vLLM's records through its AsyncLogHandler (a queue drained by a background thread) and
# the API server force-kills a dead EngineCore the moment it reports ENGINE_CORE_DEAD, so the ``logger.exception`` of
# the fatal error in the EngineCore is routinely lost (three launches on P150x8, 2026-09-10, no traceback anywhere).
# ``install_vllm_sync_error_log_handler`` mirrors ERROR+ records of the ``vllm`` logger synchronously to stderr from
# inside the EngineCore (``initialize_vllm_model`` runs there). ``SOLAR_OPEN_VLLM_SYNC_ERROR_LOG=0`` disables it.
SYNC_ERROR_LOG_ENV = "SOLAR_OPEN_VLLM_SYNC_ERROR_LOG"
_SYNC_ERROR_LOG_HANDLER_NAME = "solar_open_sync_error_log"


def install_vllm_sync_error_log_handler(level=logging.ERROR, stream=None) -> bool:
    """Attach one synchronous ``StreamHandler`` (default stderr) for ERROR+ records to the ``vllm`` logger.

    Idempotent (keyed by handler name) and independent of vllm being importable: it configures the standard-library
    logger vLLM writes to. Returns True when a handler was added, False when disabled by the env or already present.
    """
    if os.getenv(SYNC_ERROR_LOG_ENV, "1") == "0":
        return False
    vllm_logger = logging.getLogger("vllm")
    if any(getattr(h, "name", None) == _SYNC_ERROR_LOG_HANDLER_NAME for h in vllm_logger.handlers):
        return False
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.name = _SYNC_ERROR_LOG_HANDLER_NAME
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(asctime)s [solar_open sync mirror] %(name)s: %(message)s"))
    vllm_logger.addHandler(handler)
    if vllm_logger.level == logging.NOTSET or vllm_logger.level > level:
        vllm_logger.setLevel(level if vllm_logger.level == logging.NOTSET else min(vllm_logger.level, level))
    logger.info(f"Installed a synchronous ERROR+ mirror on the 'vllm' logger ({SYNC_ERROR_LOG_ENV}=0 disables it)")
    return True


LEGACY_PROTOCOL_MODULE = "vllm.entrypoints.openai.protocol"
LEGACY_TOOL_PARSERS_PACKAGE = "vllm.entrypoints.openai.tool_parsers"
LEGACY_ABSTRACT_TOOL_PARSER_MODULE = LEGACY_TOOL_PARSERS_PACKAGE + ".abstract_tool_parser"
CURRENT_TOOL_PARSERS_PACKAGE = "vllm.tool_parsers"
# The eight names the two Upstage files import from the legacy protocol module -> their vLLM 0.25.1 homes.
LEGACY_PROTOCOL_NAMES = {
    "ChatCompletionRequest": "vllm.entrypoints.openai.chat_completion.protocol",
    "ResponsesRequest": "vllm.entrypoints.openai.responses.protocol",
    "DeltaMessage": "vllm.entrypoints.openai.engine.protocol",
    "DeltaFunctionCall": "vllm.entrypoints.openai.engine.protocol",
    "DeltaToolCall": "vllm.entrypoints.openai.engine.protocol",
    "ExtractedToolCallInformation": "vllm.entrypoints.openai.engine.protocol",
    "ToolCall": "vllm.entrypoints.openai.engine.protocol",
    "FunctionCall": "vllm.entrypoints.openai.engine.protocol",
}


def install_vllm_compat_shims() -> dict:
    """Make the vLLM-0.12 module paths Upstage's parser files import resolve on the installed vLLM.

    Returns ``{legacy_module: "native" | "shim"}``. "native": the module exists in this vLLM, nothing done. "shim": a
    module re-exporting the ``LEGACY_PROTOCOL_NAMES`` from their current homes (respectively an alias of
    ``vllm.tool_parsers`` / ``.abstract_tool_parser``) was put into ``sys.modules`` and onto ``vllm.entrypoints.openai``
    so both ``import`` forms resolve. Idempotent (a second call reports the existing shims). Raises ImportError naming
    the symbol if the installed vLLM moved a name again - then extend ``LEGACY_PROTOCOL_NAMES``.
    """
    import importlib
    import sys
    import types

    if not vllm_available():
        raise RuntimeError(
            "vllm is not importable in this environment; install_vllm_compat_shims is for the serving venv"
        )
    import vllm

    result = {}
    openai_pkg = importlib.import_module("vllm.entrypoints.openai")

    existing = sys.modules.get(LEGACY_PROTOCOL_MODULE)
    if existing is not None and getattr(existing, "__solar_open_shim__", False):
        result[LEGACY_PROTOCOL_MODULE] = "shim"
    else:
        try:
            importlib.import_module(LEGACY_PROTOCOL_MODULE)
            result[LEGACY_PROTOCOL_MODULE] = "native"
        except ImportError:
            shim = types.ModuleType(LEGACY_PROTOCOL_MODULE)
            shim.__doc__ = (
                "Compat shim (models/demos/solar_open/tt/vllm_support.py) re-exporting the vLLM 0.12 protocol names "
                "Upstage's Solar-Open parsers import from their current vLLM modules."
            )
            shim.__solar_open_shim__ = True
            for name, home in LEGACY_PROTOCOL_NAMES.items():
                try:
                    setattr(shim, name, getattr(importlib.import_module(home), name))
                except (ImportError, AttributeError) as exc:
                    raise ImportError(
                        f"{name} not found in {home} (vllm {vllm.__version__}); update LEGACY_PROTOCOL_NAMES in "
                        "models/demos/solar_open/tt/vllm_support.py"
                    ) from exc
            sys.modules[LEGACY_PROTOCOL_MODULE] = shim
            setattr(openai_pkg, "protocol", shim)
            result[LEGACY_PROTOCOL_MODULE] = "shim"

    existing = sys.modules.get(LEGACY_ABSTRACT_TOOL_PARSER_MODULE)
    if existing is not None and existing.__name__ != LEGACY_ABSTRACT_TOOL_PARSER_MODULE:
        result[LEGACY_ABSTRACT_TOOL_PARSER_MODULE] = "shim"
    else:
        try:
            importlib.import_module(LEGACY_ABSTRACT_TOOL_PARSER_MODULE)
            result[LEGACY_ABSTRACT_TOOL_PARSER_MODULE] = "native"
        except ImportError:
            package = importlib.import_module(CURRENT_TOOL_PARSERS_PACKAGE)
            abstract = importlib.import_module(CURRENT_TOOL_PARSERS_PACKAGE + ".abstract_tool_parser")
            sys.modules[LEGACY_TOOL_PARSERS_PACKAGE] = package
            sys.modules[LEGACY_ABSTRACT_TOOL_PARSER_MODULE] = abstract
            setattr(openai_pkg, "tool_parsers", package)
            result[LEGACY_ABSTRACT_TOOL_PARSER_MODULE] = "shim"
    logger.info(f"vLLM {vllm.__version__} compat shims for Upstage's Solar-Open parsers: {result}")
    return result


def register_vllm_parsers(model_dir=None, reasoning=True, tool=True) -> dict:
    """Register Upstage's ``SolarOpenReasoningParser`` / ``SolarOpenToolParser`` under "solar_open" in a vLLM process.

    The two files ship in the HF repo (``$HF_MODEL/solar_open_reasoning_parser.py`` / ``solar_open_tool_parser.py``)
    without a register decorator (Upstage's vLLM fork registers them internally), so a stock vLLM needs this shim,
    loaded through ``--reasoning-parser-plugin`` / ``--tool-parser-plugin`` (see ``vllm_plugins/solar_open_parsers.py``).
    ``install_vllm_compat_shims()`` runs first so the files' vLLM-0.12 imports resolve on vLLM 0.25.1; the tool parser
    also needs the ``pyjson5`` package in the serving venv. Returns ``{"reasoning": cls, "tool": cls}`` for what was
    registered. Importing the reasoning parser patches ``json._default_encoder`` (Upstage's file does that at import;
    this is why registration is opt-in and never runs on plain import of this module). Host-verified on vLLM 0.25.1
    (both parsers register, instantiate on the Solar tokenizer and parse a think/content sample) and live on 2026-09-10
    through tt-inference-server (reasoning / content split, streamed reasoning deltas, a tool call with a non-empty id).
    """
    if not vllm_available():
        raise RuntimeError("vllm is not importable in this environment; register_vllm_parsers is for the serving venv")
    install_vllm_compat_shims()
    model_dir = _resolve_model_dir(model_dir)
    registered = {}
    if reasoning:
        from vllm.reasoning import ReasoningParserManager

        module = _import_python_file(model_dir / REASONING_PARSER_FILE, "solar_open_reasoning_parser")
        parser_cls = module.SolarOpenReasoningParser
        ReasoningParserManager.register_module(REASONING_PARSER_NAME, force=True, module=parser_cls)
        registered["reasoning"] = parser_cls
        logger.info(f"Registered vLLM reasoning parser {REASONING_PARSER_NAME!r} -> {parser_cls.__name__}")
    if tool:
        manager = _import_tool_parser_manager()
        module = _import_python_file(model_dir / TOOL_PARSER_FILE, "solar_open_tool_parser")
        parser_cls = module.SolarOpenToolParser
        manager.register_module(TOOL_PARSER_NAME, force=True, module=parser_cls)
        registered["tool"] = parser_cls
        logger.info(f"Registered vLLM tool parser {TOOL_PARSER_NAME!r} -> {parser_cls.__name__}")
    return registered
