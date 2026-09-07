# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Solar-Open implementation of create_tt_model, compatible with the tt_transformers demo / Generator interface.

Besides building ModelArgs + Model it owns two bring-up guards: the warm-cache skip of the 205 GB host load and
the per-device KV DRAM budget check (design D6; paged pool or unpaged caches), which runs before any weight is read.
"""

import os

from loguru import logger

import ttnn
from models.demos.solar_open.config import MeshConfig, ModeConfig, MoEOptions
from models.demos.solar_open.tt.model_config import ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig

# The paged KV cache is bfloat8_b (attention/kv_cache.py): 1 mantissa byte + 1/16 byte of shared exponent per element.
KV_BYTES_PER_ELEMENT = 1.0625
# Default per-device KV budget (GiB) next to the fixed weights: ~14.3 GiB with bfp8 experts, ~8.7 GiB with bfp4
# (32 GiB per Blackhole device). Override with SOLAR_OPEN_KV_BUDGET_GIB.
KV_BUDGET_GIB_BFP8_EXPERTS = 16.0
KV_BUDGET_GIB_BFP4_EXPERTS = 22.0


def paged_kv_cache_gib(num_kv_heads, head_dim, n_layers, paged_attention_config: PagedAttentionConfig, tensor_parallel):
    """Per-device DRAM footprint (GiB) of the paged K+V cache over all layers.

    Mirrors attention/kv_cache.py: per layer ``[max_num_blocks, num_kv_heads // tp, block_size, head_dim]`` bfp8 for
    K and again for V. Solar-Open at TP=8: 48 x 2 x 1 x 128 x 1.0625 = 13,056 B per token per device
    (32 users x 8K = 3.19 GiB, 32 x 16K = 6.38, 32 x 32K = 12.75, 1 x 128K = 1.59).
    """
    kv_heads_per_device = max(1, num_kv_heads // tensor_parallel)
    tokens = paged_attention_config.max_num_blocks * paged_attention_config.block_size
    return tokens * 2 * kv_heads_per_device * head_dim * KV_BYTES_PER_ELEMENT * n_layers / 2**30


def unpaged_kv_cache_gib(num_kv_heads, head_dim, n_layers, max_local_batch_size, max_seq_len, tensor_parallel):
    """Per-device DRAM footprint (GiB) of the UNPAGED K+V cache over all layers.

    Mirrors attention/kv_cache.py: per layer ``[max_local_batch_size, num_kv_heads // tp, max_seq_len, head_dim]`` bfp8
    for K and again for V, i.e. the same 13,056 B per (user, position) per device as the paged pool at TP=8. A
    ``Model(max_seq_len=None)`` falls back to ``max_position_embeddings`` (131072): 32 users x 128K x 48 layers would be
    52 GiB per device, so the guard covers this shape too.
    """
    kv_heads_per_device = max(1, num_kv_heads // tensor_parallel)
    tokens = max_local_batch_size * max_seq_len
    return tokens * 2 * kv_heads_per_device * head_dim * KV_BYTES_PER_ELEMENT * n_layers / 2**30


def kv_budget_gib(moe_options: MoEOptions) -> float:
    """KV budget in GiB per device: SOLAR_OPEN_KV_BUDGET_GIB if set, else 16 (bfp8 experts) / 22 (bfp4 experts)."""
    default = KV_BUDGET_GIB_BFP8_EXPERTS if moe_options.expert_dtype == ttnn.bfloat8_b else KV_BUDGET_GIB_BFP4_EXPERTS
    return float(os.getenv("SOLAR_OPEN_KV_BUDGET_GIB", str(default)))


def check_kv_budget(
    num_kv_heads,
    head_dim,
    n_layers,
    paged_attention_config,
    tensor_parallel,
    moe_options,
    max_local_batch_size=None,
    max_seq_len=None,
):
    """Raise ValueError when the KV cache would exceed the per-device budget; returns the footprint in GiB.

    Paged (``paged_attention_config`` given): the pool of ``max_num_blocks x block_size`` tokens. Unpaged
    (``paged_attention_config=None``): ``max_local_batch_size x max_seq_len`` positions, the shape
    ``attention/kv_cache.py`` allocates per layer. A DRAM OOM after the 205 GB host load is the expensive failure this
    prevents (design X18): the check needs only the config, so it runs before the state dict is loaded.
    """
    budget = kv_budget_gib(moe_options)
    if paged_attention_config is not None:
        kv_gib = paged_kv_cache_gib(num_kv_heads, head_dim, n_layers, paged_attention_config, tensor_parallel)
        shape = (
            f"Paged KV cache: {paged_attention_config.max_num_blocks} blocks x {paged_attention_config.block_size} "
            "tokens"
        )
        remedy = "Reduce max_seq_len / batch (fewer page blocks)"
    else:
        assert max_local_batch_size and max_seq_len, "the unpaged KV check needs max_local_batch_size and max_seq_len"
        kv_gib = unpaged_kv_cache_gib(
            num_kv_heads, head_dim, n_layers, max_local_batch_size, max_seq_len, tensor_parallel
        )
        shape = f"Unpaged KV cache: {max_local_batch_size} users x {max_seq_len} positions"
        remedy = "Reduce max_seq_len / batch or use paged attention"
    logger.info(
        f"{shape}, {n_layers} layers, TP={tensor_parallel} -> {kv_gib:.2f} GiB per device (budget {budget:.1f} GiB, "
        f"experts {moe_options.expert_dtype_str})"
    )
    if kv_gib > budget:
        raise ValueError(
            f"{shape} needs {kv_gib:.2f} GiB per device, above the {budget:.1f} GiB budget for "
            f"{moe_options.expert_dtype_str} experts. {remedy}, use SOLAR_OPEN_EXPERT_DTYPE=bfp4, or raise "
            f"SOLAR_OPEN_KV_BUDGET_GIB once DRAM headroom is measured."
        )
    return kv_gib


def create_tt_model(
    mesh_device,
    max_batch_size,
    max_seq_len,
    optimizations=None,
    paged_attention_config: PagedAttentionConfig = None,
    dtype=ttnn.bfloat8_b,
    state_dict=None,
    num_layers=None,
    mesh_config=None,
    create_kv_cache=True,
    users_row_sharded=False,
    moe_options: MoEOptions = None,
):
    """Solar-Open version of create_tt_model that matches the tt_transformers interface.

    Returns ``(model_args, model, tt_kv_cache, state_dict)``. ``dtype`` is the attention / lm_head weight dtype;
    the expert dtypes and the router implementation come from ``moe_options`` (default: ``ModelArgs.moe_options``
    = ``MoEOptions.from_env()``). The effective options are recorded on ``model_args`` so the weight-cache
    directory and its completion marker always describe the tensors actually built.
    """
    from models.demos.solar_open.tt.model import Model

    # Use provided mesh_config or create the default MeshConfig for the mesh shape (TP over columns, EP over rows)
    if mesh_config is None:
        mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0]))

    model_args = ModelArgs(
        mesh_device,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
    )
    # Override num_layers if provided (useful for quick testing with fewer layers)
    if num_layers is not None:
        model_args.hf_config.num_hidden_layers = num_layers
        model_args.n_layers = num_layers

    moe_options = moe_options or model_args.moe_options
    model_args.moe_options = moe_options

    # KV budget guard (D6): fail before the expensive host load rather than with a device OOM after it. Covers the
    # paged pool and the unpaged [max_local_batch_size, 1, max_seq_len, 128] per-layer caches alike.
    if create_kv_cache:
        check_kv_budget(
            num_kv_heads=model_args.hf_config.num_key_value_heads,
            head_dim=model_args.head_dim,
            n_layers=model_args.n_layers,
            paged_attention_config=paged_attention_config,
            tensor_parallel=mesh_device.shape[1],
            moe_options=moe_options,
            max_local_batch_size=model_args.max_local_batch_size,
            max_seq_len=model_args.max_seq_len,
        )

    # Decide whether the HF weights are still needed on host. When the ttnn weight cache for
    # this (model, dtype, expert dtype, mesh shape) was already fully built on a previous run, ttnn.as_tensor
    # loads every weight from disk and the state_dict is never read -- so skip the expensive
    # from_pretrained host load entirely. This is what spares the e2e demo the prefill host-OOM
    # (#48509) on warm-cache runs, without relying on the manual --skip-model-load flag.
    #
    # state_dict is None  -> decide here (warm cache => {} skip, else cold load).
    # state_dict == {}     -> explicit skip (--skip-model-load) or a prior DP model already skipped.
    # state_dict populated -> reuse across DP models (avoid reloading for every submesh).
    loaded_real_weights = False
    if state_dict is None:
        if not model_args.dummy_weights and model_args.weight_cache_is_complete(dtype):
            logger.info("Warm ttnn weight cache detected -- skipping HF state_dict load.")
            state_dict = {}
        else:
            state_dict = model_args.load_state_dict(
                weights_path=model_args.model_path,
                dummy_weights=model_args.dummy_weights,
                convert_to_meta_format=True,
            )
            loaded_real_weights = bool(state_dict) and not model_args.dummy_weights

    model = Model.create_transformer_compatible(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        tensor_cache_path=str(model_args.weight_cache_path(dtype)),
        paged_attention_config=paged_attention_config,
        mesh_config=mesh_config,  # Pass explicit MeshConfig
        create_kv_cache=create_kv_cache,
        users_row_sharded=users_row_sharded,
        moe_options=moe_options,
    )

    # If this run populated the cache from a cold host load, record completion so future runs
    # can skip the load. Only for full-model builds (a num_layers override produces a partial
    # cache that must not satisfy the completeness check).
    if loaded_real_weights and num_layers is None:
        model_args.mark_weight_cache_complete(dtype)

    # Extract tt_kv_cache like tt_transformers does (layers expose the cache as self_attn.layer_past)
    tt_kv_cache = []
    if create_kv_cache:
        for layer in model.layers:
            tt_kv_cache.append(layer.self_attn.layer_past)

    return model_args, model, tt_kv_cache, state_dict
