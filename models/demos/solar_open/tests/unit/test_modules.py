# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Component and single-layer tests for the Solar-Open-100B TT implementation.

Every test builds a randomly initialised ``SolarOpenDecoderLayer`` (transformers >= 5.12, native ``solar_open``
model type) at the REAL model dimensions (H=4096, 64/8 heads x 128, 128 experts x 1280, shared expert 1280,
vocab 196608) so the per-device TP shapes are the production ones, converts its ``state_dict()`` with
``convert_hf_qkv_to_meta_format`` and feeds it to the TT ``DecoderLayer`` -- the same weight path the demo takes
after ``AutoModelForCausalLM.from_pretrained``.

Components (select with ``--test-modules=a,b``): ``rms_norm``, ``attention``, ``router``, ``experts``,
``shared_expert``, ``mlp``, ``decoder``; ``test_experts_shared_expert_hook`` proves that the experts add the shared
expert's partial BEFORE their single TP all_reduce; ``test_model`` runs a 1-layer ``SolarOpenForCausalLM`` end to end.
Every component receives bf16 activations (the production dtype: the residual stream is bf16) and the decode cases of
the attention / decoder components attend over a ``DECODE_CONTEXT_LEN``-token context per user written through the
TT prefill path, at RoPE positions ``pos_offset`` (0 or 70000, beyond YaRN's original context) onwards.

Environment: ``HF_MODEL`` must point at a directory named ``Solar-Open-100B`` holding a ``config.json``
(``models/demos/solar_open/configs/Solar-Open-100B`` when no checkpoint is present), ``MESH_DEVICE=P150x8``.
The ``SOLAR_OPEN_*`` MoE flags (``config.MoEOptions.from_env``) are honoured so the router/expert dtype
variants can be A/B tested with the same suite.
"""

import collections
import contextlib
import dataclasses
from unittest import mock

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import rope_scaling_model_factory
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format
from models.tt_transformers.tt.rope import compute_gather_cos_sin

from ...config import MoEOptions
from ...tt.attention.config import ProgramConfig as AttentionProgramConfig
from ...tt.experts import prefill as experts_prefill
from ...tt.layer import DecoderLayer
from ...tt.model import create_rope_setup
from ...utils.general_utils import resolve_rope_theta
from ..test_factory import TestFactory, compare_tensors, parametrize_batch_seq, parametrize_mesh_with_fabric

# Router agreement metric (probe round 2 in the design notes): the fused ``moe_grouped_topk`` kernel computes the
# sigmoid in the SFPU with ~1e-4..5e-4 absolute error, so when the 8th and 9th BIASED scores of a token are closer
# than this margin the kernel may legitimately pick the other expert. Such near-tie tokens are reported but
# excluded from the exact expert-set check; tokens with a decisive margin must agree 100% (fp32 logits).
ROUTER_DECISIVE_MARGIN = 1e-3
# Overall fraction of tokens allowed to route to a different expert set, keyed by ``router_fp32_logits``
# (measured ~2% with fp32 logits: ~10% of the tokens are near ties and ~20% of those flip; the bf16-logits flag adds
# bf16 rounding of the logits themselves).
ROUTER_MAX_SET_MISMATCH_FRACTION = {True: 0.05, False: 0.25}
# Small-batch floor of that cap (re-baselined 2026-09-07, design D13): at 32 tokens the expected 0.6 near-tie flips
# come as 2 in about one seed out of eight (measured 2/32 on P150x8, both with margins < 3.3e-4), so the decode cases
# allow 3 whatever the fraction says; the decisive-token check above still catches any real routing error.
ROUTER_MIN_MISMATCH_ALLOWANCE = 3
# The dense-routing PCC is asserted on the tokens whose expert set agrees at every token count (this population holds
# every decisive token) and on ALL tokens only from this many tokens on: a single near-tie flip moves ~1/8 of a row's
# weight to another expert, which costs ~0.004 whole-tensor PCC at 32 tokens (measured 0.9925 with 2 flips) but is
# noise at >= 128 tokens (<= ~2% flips, PCC >= 0.995). Same split as tests/unit/test_router.py.
ROUTER_ALL_TOKEN_PCC_MIN_TOKENS = 128
# Tolerance on the sum of the dense routing weights per token (must be routed_scaling_factor = 1.0).
ROUTER_ROW_SUM_TOLERANCE = 1e-2
# Decode cases attend over a real context: every user first gets this many prefix tokens written into the KV cache
# through the TT prefill path (RoPE at positions pos_offset .. pos_offset + P - 1), then decodes one token at position
# pos_offset + P; the HF reference sees prefix + token under a causal mask and its last row is compared. On an EMPTY
# cache at position 0 the decode softmax spans a single key, so the output is Wo * V whatever Q, K, the RoPE tables,
# the scale or the cache update do -- decode RoPE / SDPA / K-cache would be untested. 64 tokens = the small SDPA
# prefill chunk (q/k chunk 64), and the test cache (max(seq_len, 128) slots) still holds the decode token at slot 64.
# The cache SLOT (0..P) and the RoPE position (pos_offset + slot) are decoupled here so a 70000 offset fits the small
# test cache: attention depends on positions only through the rotated Q/K, so this is the same computation.
DECODE_CONTEXT_LEN = 64
# Production dtype of the tensors the components receive: the residual stream is bf16 (layer.py _residual_add), so
# the norm inputs, the attention / MoE inputs (norm outputs) and the layer input are bf16.
ACTIVATION_DTYPE = ttnn.bfloat16
# Realistic e_score_correction_bias values: +-0.01 in bf16-exact 2^-9 steps (the checkpoint has |b| <= 0.0096).
ROUTER_BIAS_STEP = 2.0**-9
# Shared-expert hook checks (contract C3 / design D2): the experts add the shared partial on every TP device BEFORE
# the single all_reduce, so a stub returning the constant c shifts the all-reduced output by exactly tp * c (c if it
# were added after the all_reduce, 0 if the hook were ignored), and a stub returning zeros leaves it unchanged.
SHARED_HOOK_CONSTANT = 1.0
SHARED_HOOK_ZERO_PCC = 0.999  # zeros hook vs no hook: identical up to the CCL reduction-order noise
SHARED_HOOK_REL_TOL = 0.1  # |shift - tp * c| <= 10% of tp * c (bfp8 re-quantisation of routed + c, then the all_reduce)
# Routed-experts memory contract (design X5): the prefill paths take per-expert weight slices on demand and free them
# within the split, so once a call's output is freed nothing of it may stay allocated. The FIRST call at a shape still
# grows the per-device DRAM allocation (logged, not asserted): the compiled programs it adds to the program cache own
# DRAM buffers holding their kernel binaries (measured on P150x8: +0.3-0.6 MiB for decode / the 128-token dense bmm,
# +2.7 MiB at 1024 and +8.9 MiB at 4096 tokens, all returned by clearing the program cache) and the expert-sorted path
# caches one [split, split] bf16 identity (2 MiB at the 1024-token split). A SECOND identical call must not move the
# allocation at all (measured: exactly 0 bytes); a single leaked expert slice would show as >= 0.66 MiB.
EXPERTS_DRAM_GROWTH_MAX_BYTES = 0
# CCL accounting (design D2, contract C6): the MoE block issues exactly ONE collective per call, the TP all_reduce of
# the routed + shared partial (the router's linear is replicated, the shared expert returns a per-device partial).
# Every ttnn / ttnn.experimental collective entry point is counted while the block runs.
CCL_OP_NAMES = {
    ttnn: (
        "all_reduce",
        "reduce_scatter",
        "all_gather",
        "all_broadcast",
        "all_to_all_combine",
        "all_to_all_dispatch",
        "point_to_point",
    ),
    ttnn.experimental: tuple(
        name
        for name in dir(ttnn.experimental)
        if any(key in name for key in ("all_reduce", "reduce_scatter", "all_gather", "all_to_all", "all_broadcast"))
    ),
}


@contextlib.contextmanager
def count_ccl_ops():
    """Count the collective ops launched inside the block: yields a ``Counter`` keyed by op name."""
    counts = collections.Counter()

    def counting(name, original):
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapper

    with contextlib.ExitStack() as stack:
        for module, names in CCL_OP_NAMES.items():
            for name in names:
                original = getattr(module, name, None)
                if callable(original):
                    stack.enter_context(mock.patch.object(module, name, counting(name, original)))
        yield counts


class DecodeContext:
    """The per-user KV context of a decode case (see ``DECODE_CONTEXT_LEN``).

    ``prefix`` ``[B, P, H]`` are the prefix hidden states (the attention INPUT of the prefix tokens; the decoder
    component applies ``input_layernorm`` before writing them), ``position_embeddings`` the HF (cos, sin) for the
    P + 1 positions ``pos_offset .. pos_offset + P`` (``[1, P + 1, head_dim]``, broadcast over users), ``mask`` the
    causal ``[1, 1, P + 1, P + 1]`` mask and ``prefix_rope_mats`` the TT prefill cos/sin of the P prefix positions.
    """

    def __init__(self, prefix, position_embeddings, mask, prefix_rope_mats):
        self.prefix = prefix
        self.position_embeddings = position_embeddings
        self.mask = mask
        self.prefix_rope_mats = prefix_rope_mats

    @property
    def context_len(self):
        return self.prefix.shape[1]

    def fill_kv_cache(self, mesh_device, decoder_layer, page_table, apply_input_norm):
        """Write every user's prefix K/V into the TT layer's cache through the prefill path (``user_id=u`` selects
        the user's cache rows / page-table row). ``apply_input_norm`` runs the layer's ``input_layernorm`` first, as
        the full layer does."""
        batch_size, context_len, hidden_size = self.prefix.shape
        for user in range(batch_size):
            tt_prefix = ttnn.from_torch(
                self.prefix[user].reshape(1, 1, context_len, hidden_size),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ACTIVATION_DTYPE,
            )
            if apply_input_norm:
                normed = decoder_layer.input_layernorm(tt_prefix)
                tt_prefix.deallocate(True)
                tt_prefix = normed
            out = decoder_layer.self_attn(
                tt_prefix,
                rope_mats=self.prefix_rope_mats,
                position_idx=None,
                page_table=page_table,
                kv_cache=None,
                is_decode=False,
                user_id=user,
            )
            out.deallocate(True)

    def reference_inputs(self, decode_hidden_states):
        """``(hidden_states [B, P + 1, H], position_embeddings, mask)`` for the HF reference of the decode step."""
        return torch.cat([self.prefix, decode_hidden_states], dim=1), self.position_embeddings, self.mask


def build_decode_context(
    setup, config, batch_size, pos_offset, local_batch_size, context_len=DECODE_CONTEXT_LEN, prefix=None
):
    """Prefix (random bf16-exact, or the given ``[B, P, H]`` tensor) + RoPE inputs for a decode case
    (``DecodeContext``); None on row-sharded meshes (the per-user cache fill assumes replicated caches)."""
    from transformers.models.solar_open.modeling_solar_open import SolarOpenRotaryEmbedding

    if setup["mesh_device"].shape[0] > 1 and local_batch_size > 1:
        return None
    if prefix is None:
        prefix = torch.randn(batch_size, context_len, config.hidden_size).to(torch.bfloat16).float()
    context_len = prefix.shape[1]
    position_ids = torch.arange(pos_offset, pos_offset + context_len + 1, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        position_embeddings = SolarOpenRotaryEmbedding(config)(prefix, position_ids)  # [1, P + 1, head_dim]
    mask = torch.triu(torch.full((1, 1, context_len + 1, context_len + 1), -float("inf")), diagonal=1)
    _, _, prefix_rope_mats, _ = build_rope_inputs(
        setup, config, prefix[:1], 1, context_len, pos_offset, local_batch_size, is_decode=False
    )
    return DecodeContext(prefix, position_embeddings, mask, prefix_rope_mats)


def assert_replicated(tt_tensor, label):
    """Every device holds the same replica of an all-reduced tensor (device 0 alone can look right while a CCL left a
    stale block on another device -- measured with all_gather_async on P150x8, see attention/operations.py)."""
    per_device = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tt_tensor)]
    for d, other in enumerate(per_device[1:], start=1):
        diff = (other.float() - per_device[0].float()).abs()
        assert torch.equal(other, per_device[0]), (
            f"{label}: device {d} differs from device 0 (max |diff| {diff.max().item():.4g}, "
            f"{int((diff.amax(dim=-1) > 0).sum())} rows) -- a collective left a stale replica"
        )


def run_attention_component(
    mesh_device,
    hidden_shape,
    mask,
    position_embeddings,
    rope_mats,
    tt_position_idx,
    reference_layer,
    decoder_layer,
    is_decode,
    is_row_sharded,
    pcc_threshold,
    page_table=None,
    decode_context=None,
):
    """Attention component (``SolarOpenAttention`` eager vs TT ``Attention``).

    When ``page_table`` is provided, attention runs through the paged kv-cache
    code path; the decoder_layer must have been constructed with a matching
    ``paged_attention_config``. In decode mode ``decode_context`` (``DecodeContext``) first writes every user's
    prefix into the KV cache, so the decode step attends over ``context_len + 1`` keys.
    """
    batch_size, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape).to(torch.bfloat16).float()

    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(1, 1, -1, hidden_states.shape[-1]),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ACTIVATION_DTYPE,
    )

    reference_input, reference_position_embeddings, reference_mask = hidden_states, position_embeddings, mask
    if is_decode and decode_context is not None:
        decode_context.fill_kv_cache(mesh_device, decoder_layer, page_table, apply_input_norm=False)
        reference_input, reference_position_embeddings, reference_mask = decode_context.reference_inputs(hidden_states)
    with torch.no_grad():
        reference_out, _ = reference_layer.self_attn(
            hidden_states=reference_input,
            position_embeddings=reference_position_embeddings,
            attention_mask=reference_mask,
        )
    if is_decode and decode_context is not None:
        reference_out = reference_out[:, -1:, :]  # the decode token's row of the [B, P + 1, H] output

    # TT attention: causal masking is handled inside the SDPA kernels.
    tt_out = decoder_layer.self_attn(
        tt_hidden_states,
        rope_mats=rope_mats,
        position_idx=tt_position_idx,
        page_table=page_table,
        kv_cache=None,
        is_decode=is_decode,
    )

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    tt_output_torch = ttnn.to_torch(tt_out, mesh_composer=mesh_composer)[..., : batch_size * seq_len, :hidden_size]
    if not is_row_sharded:
        assert_replicated(tt_out, "attention output")

    passing, output = compare_tensors(tt_output_torch, reference_out, mesh_device, pcc_threshold=pcc_threshold)
    assert passing, f"Attention test failed. Output: {output}"
    logger.info(f"Attention test passed. Output: {output}")


def run_rms_norm_component(
    mesh_device, hidden_shape, reference_layer, decoder_layer, is_decode, is_row_sharded, pcc_threshold
):
    """RMSNorm component (``input_layernorm``, T5-style, eps 1e-5)."""
    batch_size, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    with torch.no_grad():
        ref_output = reference_layer.input_layernorm(hidden_states)

    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        hidden_states,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ACTIVATION_DTYPE,
    )

    tt_output = decoder_layer.input_layernorm(tt_hidden_states)

    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape))
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[:batch_size, :seq_len, :]
    passing, output = compare_tensors(tt_output_torch, ref_output, mesh_device, pcc_threshold=pcc_threshold)
    assert passing, f"RMS Norm test failed. Output: {output}"
    logger.info(f"RMS Norm test passed. Output: {output}")


def reference_routing(reference_layer, hidden_2d):
    """Run the HF router on ``[T, H]`` inputs.

    Returns ``(indices [T, k] int64 unsorted, weights [T, k] fp32 normalised, dense [T, E] fp32, margin [T])`` where
    ``margin`` is the gap between the 8th and 9th BIASED sigmoid scores (the quantity the top-k decision depends on).
    """
    mlp = reference_layer.mlp
    with torch.no_grad():
        logits = mlp.gate(hidden_2d)  # fp32 [T, E]
        indices, weights = mlp.route_tokens_to_experts(logits)
        biased = torch.sigmoid(logits) + mlp.gate.e_score_correction_bias
        top_k = indices.shape[-1]
        ranked = biased.topk(top_k + 1, dim=-1).values
        margin = ranked[:, top_k - 1] - ranked[:, top_k]
    dense = torch.zeros_like(logits).scatter_(1, indices, weights)
    return indices, weights, dense, margin


def expert_set_agreement(tt_dense, ref_indices):
    """Per-token boolean: the experts with non-zero TT weight equal the reference top-k set."""
    top_k = ref_indices.shape[-1]
    tt_sets = tt_dense.topk(top_k, dim=-1).indices.sort(dim=-1).values
    ref_sets = ref_indices.long().sort(dim=-1).values
    return (tt_sets == ref_sets).all(dim=-1)


def run_topk_router_component(
    mesh_device,
    hidden_shape,
    reference_layer,
    decoder_layer,
    is_decode,
    is_row_sharded,
    pcc_threshold,
    fp32_logits=True,
):
    """Router component: HF ``gate`` + ``route_tokens_to_experts`` vs TT ``TopKRouter`` (contract C2).

    Checks (probe round 2 metric): (1) 100% expert-set agreement on tokens whose 8th/9th biased-score margin is
    decisive (>= ROUTER_DECISIVE_MARGIN) when the logits are fp32, (2) a bounded overall mismatch fraction,
    (3) PCC of the dense ``[T, E]`` routing tensor, (4) every row sums to ``routed_scaling_factor`` and is
    non-negative, (5) the informational ``indices`` output names the same experts as the dense tensor and (6) the
    dense tensor is identical on every device (it feeds replicated experts).
    """
    batch, seq_len, hidden_size = hidden_shape
    num_tokens = batch * seq_len
    # The device routes bf16 activations; the reference must route the SAME values (bf16 rounding of the input moves
    # the logits by up to ~8e-3 here, i.e. by more than the decisive margin, and would be charged to the router).
    hidden_states = torch.randn(hidden_shape).to(torch.bfloat16).float()
    hidden_2d = hidden_states.reshape(-1, hidden_size)

    ref_indices, ref_weights, ref_dense, margin = reference_routing(reference_layer, hidden_2d)
    top_k = ref_indices.shape[-1]
    num_experts = ref_dense.shape[-1]
    route_scale = float(getattr(reference_layer.mlp, "routed_scaling_factor", 1.0))

    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(1, 1, -1, hidden_size),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    tt_indices, tt_dense = decoder_layer.mlp.router(tt_hidden_states, is_decode=is_decode)

    # The dense tensor is replicated on every TP device: read device 0, and check the replication.
    dense_per_device = [ttnn.to_torch(t).float()[:num_tokens, :num_experts] for t in ttnn.get_device_tensors(tt_dense)]
    tt_dense_torch = dense_per_device[0]
    for d, other in enumerate(dense_per_device[1:], start=1):
        assert torch.equal(other, tt_dense_torch), f"router dense output differs between device 0 and device {d}"
    tt_indices_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_indices)[0]).long()[:num_tokens, :top_k]

    assert not torch.isnan(tt_dense_torch).any(), "router dense output contains NaN"
    assert (tt_dense_torch >= 0).all(), "router dense output must be non-negative (experts union mask uses sum > 0)"
    nonzero_per_row = (tt_dense_torch > 0).sum(dim=-1)
    assert (
        nonzero_per_row == top_k
    ).all(), f"expected exactly {top_k} non-zero routing weights per token, got {nonzero_per_row.tolist()}"
    row_sums = tt_dense_torch.sum(dim=-1)
    assert torch.allclose(
        row_sums, torch.full_like(row_sums, route_scale), atol=ROUTER_ROW_SUM_TOLERANCE
    ), f"dense routing rows must sum to {route_scale} +- {ROUTER_ROW_SUM_TOLERANCE}: {row_sums.tolist()}"
    # The informational indices must name the experts that carry the dense weights.
    dense_sets = tt_dense_torch.topk(top_k, dim=-1).indices.sort(dim=-1).values
    assert torch.equal(
        tt_indices_torch.sort(dim=-1).values, dense_sets
    ), "router indices disagree with the non-zero entries of the dense routing tensor"

    same_set = expert_set_agreement(tt_dense_torch, ref_indices)
    decisive = margin >= ROUTER_DECISIVE_MARGIN
    num_mismatch = int((~same_set).sum())
    num_decisive_mismatch = int((~same_set & decisive).sum())
    max_mismatch = max(
        ROUTER_MIN_MISMATCH_ALLOWANCE, int(ROUTER_MAX_SET_MISMATCH_FRACTION[bool(fp32_logits)] * num_tokens)
    )
    passing_all, output_all = compare_tensors(tt_dense_torch, ref_dense, mesh_device, pcc_threshold=pcc_threshold)
    passing_agree, output_agree, weights_output = True, None, None
    if same_set.any():
        passing_agree, output_agree = compare_tensors(
            tt_dense_torch[same_set], ref_dense[same_set], mesh_device, pcc_threshold=pcc_threshold
        )
        tt_weights = tt_dense_torch.gather(1, ref_indices.long())
        _, weights_output = compare_tensors(
            tt_weights[same_set], ref_weights[same_set], mesh_device, pcc_threshold=pcc_threshold
        )
    logger.info(
        f"TopK Router: {num_tokens} tokens, {int(decisive.sum())} decisive (margin >= {ROUTER_DECISIVE_MARGIN}), "
        f"{num_mismatch} with a different expert set ({num_decisive_mismatch} of them decisive), "
        f"dense PCC on agreeing tokens: {output_agree}, on all tokens: {output_all}, "
        f"weights PCC on agreeing tokens: {weights_output}"
    )
    problems = []
    if fp32_logits and num_decisive_mismatch:
        bad = (~same_set & decisive).nonzero().flatten().tolist()[:10]
        problems.append(f"{num_decisive_mismatch} decisive tokens routed to a different expert set (first: {bad})")
    if num_mismatch > max_mismatch:
        problems.append(f"{num_mismatch} tokens with a different expert set (allowed {max_mismatch})")
    if not passing_agree:
        problems.append(f"dense routing PCC on the agreeing tokens below threshold: {output_agree}")
    if num_tokens >= ROUTER_ALL_TOKEN_PCC_MIN_TOKENS and not passing_all:
        problems.append(f"dense routing PCC on all tokens below threshold: {output_all}")
    assert not problems, "TopK Router test failed:\n" + "\n".join(problems)
    logger.info(f"TopK Router test passed. Output: {output_all}")


def uniform_random_routing(num_tokens, num_experts, top_k):
    """Controlled routing for the experts tests: every token picks ``top_k`` distinct experts uniformly at random with
    normalised random weights.

    Returns ``(router_indices [T, k] int64, routing_dense [T, E] fp32 by expert id (the TT contract),
    routing_topk [T, k] fp32 by top-k position (the HF ``SolarOpenNaiveMoe`` convention))``.
    """
    router_indices = torch.zeros(num_tokens, top_k, dtype=torch.long)
    routing_dense = torch.zeros(num_tokens, num_experts)
    routing_topk = torch.zeros(num_tokens, top_k)
    for t in range(num_tokens):
        active_experts = torch.randperm(num_experts)[:top_k]
        weights = torch.rand(top_k)
        weights = weights / weights.sum()
        router_indices[t] = active_experts
        routing_dense[t, active_experts] = weights
        routing_topk[t] = weights
    return router_indices, routing_dense, routing_topk


def indexed_routing_from_torch(mesh_device, router_indices, routing_topk):
    """``IndexedRouting`` of ONE token from the controlled routing (``[1, k]`` ids / weights), replicated: the
    layout ``TopKRouter.route_indexed`` produces (uint16 ROW_MAJOR ids, bf16 TILE weights, both ``[1, 1, 1, k]``)."""
    from models.demos.solar_open.tt.experts import IndexedRouting

    k = router_indices.shape[-1]
    replicate = ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
    indices = ttnn.from_torch(
        router_indices.reshape(1, 1, 1, k).to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=replicate,
    )
    weights = ttnn.from_torch(
        routing_topk.reshape(1, 1, 1, k),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate,
    )
    return IndexedRouting(indices=indices, weights=weights, top_k=k)


def dram_allocated_bytes(mesh_device):
    """Bytes currently allocated in DRAM per device (the mesh allocator is in lock-step across the devices)."""
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return int(view.total_bytes_allocated_per_bank) * int(view.num_banks)


def run_experts_component(mesh_device, hidden_shape, config, reference_layer, decoder_layer, is_decode, pcc_threshold):
    """Routed experts only (``SolarOpenNaiveMoe`` vs TT ``Experts`` with ``shared_expert=None``) under controlled
    uniform random routing (contract C3).

    Also checks which prefill path ran (splits longer than ``dense_bmm_max_tokens`` must take the expert-sorted
    hot/cold path, the 128-token traced shape the dense bmm) and that the per-device DRAM allocation returns to its
    pre-call level (plus the sorted path's small cached identity) once the output is freed: no per-expert weight
    slice may outlive the call (design X5).
    """
    if decoder_layer.mlp.experts.weights.num_always_on_experts:
        pytest.skip(
            "shared expert fused into the routed experts (SOLAR_OPEN_FUSE_SHARED_EXPERT=1): the [T, 128] "
            "controlled-routing contract does not apply, see test_fused_shared_expert.py"
        )
    batch_size, seq_len, hidden_size = hidden_shape
    num_tokens = batch_size * seq_len
    hidden_states = torch.randn(hidden_shape)

    router_indices, routing_weights, routing_weights_topk = uniform_random_routing(
        num_tokens, config.num_local_experts, config.num_experts_per_tok
    )

    # SolarOpenNaiveMoe.forward(hidden_states [T, H], top_k_index [T, k], top_k_weights [T, k]) -- positional.
    reference_experts = reference_layer.mlp.experts.eval()
    with torch.no_grad():
        reference_output = reference_experts(
            hidden_states.reshape(-1, hidden_size), router_indices, routing_weights_topk
        )

    experts = decoder_layer.mlp.experts
    # Single-user decode with MoEOptions.indexed_decode: feed the token's top-k as the router would
    # (``MLP.route`` -> ``route_indexed``) so the indexed/gather expert path is what gets compared.
    use_indexed = is_decode and num_tokens == 1 and decoder_layer.mlp.indexed_decode
    program_config = experts.program_config
    split_len = min(seq_len, program_config.get_down_split_size(seq_len))
    expect_sorted = (
        not is_decode
        and split_len > program_config.dense_bmm_max_tokens
        and config.num_local_experts >= experts_prefill._SORTED_MOE_MIN_EXPERTS
    )
    experts_prefill.LAST_SORTED_MOE_PLAN.clear()
    replicate = ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))

    def call_experts():
        """Upload the inputs, run the experts, read the output back and free everything the call left behind."""
        tt_hidden_states = ttnn.from_torch(
            hidden_states.reshape(1, 1, -1, hidden_size),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )
        if use_indexed:
            indexed = indexed_routing_from_torch(mesh_device, router_indices, routing_weights_topk)
            tt_output = experts(hidden_states=tt_hidden_states, is_decode=True, indexed_routing=indexed)
            indexed.deallocate()
        else:
            tt_routing_weights = ttnn.from_torch(
                routing_weights,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            )
            tt_output = experts(
                hidden_states=tt_hidden_states,
                topk_expert_weights=tt_routing_weights,
                is_decode=is_decode,
            )
        out = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[..., :num_tokens, :hidden_size]
        tt_output.deallocate(True)
        assert not tt_hidden_states.is_allocated(), "the experts must consume their input"
        if not use_indexed and tt_routing_weights.is_allocated():
            tt_routing_weights.deallocate(True)  # decode leaves the caller's routing tensor alone; prefill consumes it
        return out

    dram_before = dram_allocated_bytes(mesh_device)
    tt_output_torch = call_experts()
    dram_first = dram_allocated_bytes(mesh_device) - dram_before
    plan = dict(experts_prefill.LAST_SORTED_MOE_PLAN)
    # Memory contract: a second identical call finds the program cache and the sorted path's identity warm, so it must
    # not move the per-device DRAM allocation at all (see EXPERTS_DRAM_GROWTH_MAX_BYTES).
    call_experts()
    dram_second = dram_allocated_bytes(mesh_device) - dram_before - dram_first
    logger.info(
        f"Experts ({'decode' if is_decode else 'prefill'} {num_tokens} tokens, "
        f"{'INDEXED/gather' if use_indexed else 'dense routing'} path): sorted-MoE plan {plan or None}; "
        f"per-device DRAM growth {dram_first / 2**20:.2f} MiB on the first call (kernel binaries of the newly cached "
        f"programs + the sorted path's identity), {dram_second / 2**20:.2f} MiB on the second"
    )
    if expect_sorted:
        assert plan.get("split") == split_len, (
            f"a {seq_len}-token prefill (splits of {split_len} > dense_bmm_max_tokens "
            f"{program_config.dense_bmm_max_tokens}) must take the expert-sorted hot/cold path; plan was {plan}"
        )
    else:
        assert (
            not plan
        ), f"the expert-sorted prefill path ran for a {num_tokens}-token {'decode' if is_decode else 'prefill'}: {plan}"
    assert dram_second <= EXPERTS_DRAM_GROWTH_MAX_BYTES, (
        f"per-device DRAM grew by {dram_second / 2**20:.2f} MiB across a repeated experts call (allowed "
        f"{EXPERTS_DRAM_GROWTH_MAX_BYTES} bytes): a per-expert weight slice or an activation outlived the call"
    )

    passing, output = compare_tensors(tt_output_torch, reference_output, mesh_device, pcc_threshold=pcc_threshold)
    assert passing, f"Experts test failed. Output: {output}"
    logger.info(f"Experts test passed. Output: {output}")


def run_shared_expert_component(mesh_device, hidden_shape, reference_layer, decoder_layer, is_decode, pcc_threshold):
    """Shared expert (``SolarOpenMLP`` of width 1280 vs TT ``SharedExpert``, contract C4).

    The TT module returns this device's PARTIAL sum over its 160 intermediate columns in bf16 (no CCL); the eight
    device partials are summed on host before the PCC comparison, and the partials themselves must differ.
    """
    batch_size, seq_len, hidden_size = hidden_shape
    num_tokens = batch_size * seq_len
    hidden_states = torch.randn(hidden_shape)

    with torch.no_grad():
        reference_output = reference_layer.mlp.shared_experts(hidden_states).reshape(num_tokens, hidden_size)

    shared_expert = decoder_layer.mlp.shared_expert
    if shared_expert is None and decoder_layer.mlp.experts.weights.num_always_on_experts:
        pytest.skip("shared expert fused into the routed experts; covered by test_fused_shared_expert.py")
    assert shared_expert is not None, "n_shared_experts > 0 in the config but the TT MLP has no shared expert"

    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(1, 1, -1, hidden_size),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
    )
    tt_partial = shared_expert(tt_hidden_states, is_decode=is_decode)

    assert (
        tt_partial.dtype == ttnn.bfloat16
    ), f"shared expert partial must be bf16 (contract C4), got {tt_partial.dtype}"
    assert (
        tt_partial.shape[-2] == num_tokens and tt_partial.shape[-1] == hidden_size
    ), f"shared expert partial shape {tt_partial.shape} != [1, 1, {num_tokens}, {hidden_size}]"

    partials = [ttnn.to_torch(t).float()[..., :num_tokens, :hidden_size] for t in ttnn.get_device_tensors(tt_partial)]
    if len(partials) > 1:
        assert not torch.allclose(
            partials[0], partials[1]
        ), "per-device shared-expert outputs are identical: expected TP partial sums over disjoint 160-column slices"
    tt_output = sum(partials).reshape(num_tokens, hidden_size)

    passing, output = compare_tensors(tt_output, reference_output, mesh_device, pcc_threshold=pcc_threshold)
    assert passing, f"Shared expert test failed. Output: {output}"
    logger.info(f"Shared expert test passed ({len(partials)} device partials summed). Output: {output}")


def run_full_mlp_pipeline(
    mesh_device, hidden_shape, reference_layer, decoder_layer, is_decode, is_row_sharded, pcc_threshold
):
    """Complete MoE block: router + routed experts + shared expert, one TP all-reduce (``SolarOpenMoE.forward``
    returns the single tensor ``experts(h) + shared_experts(h)``)."""
    batch, seq_len, hidden_size = hidden_shape
    hidden_states = torch.randn(hidden_shape)

    with torch.no_grad():
        reference_output = reference_layer.mlp(hidden_states)

    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_hidden_states = ttnn.from_torch(
        # Row-sharded: [batch, 1, seq, hidden] split across rows. Otherwise the decoder-layer
        # contract [1, 1, tokens, hidden] (tokens = batch * seq).
        hidden_states.unsqueeze(1) if is_row_sharded else hidden_states.reshape(1, 1, -1, hidden_size),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        dtype=ACTIVATION_DTYPE,
    )

    with count_ccl_ops() as ccl_counts:
        tt_output = decoder_layer.mlp(tt_hidden_states, is_decode=is_decode)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[..., : batch * seq_len, :hidden_size]
    if not is_row_sharded:
        assert_replicated(tt_output, "MoE output")

    # ONE collective per MoE call (the TP all_reduce; none at TP=1): routed + shared partials are summed on device first.
    expected_ccl = {"all_reduce": 1} if tuple(mesh_device.shape)[1] > 1 else {}
    assert (
        dict(ccl_counts) == expected_ccl
    ), f"the MoE block must issue exactly {expected_ccl}, it launched {dict(ccl_counts)}"
    passing, output = compare_tensors(tt_output_torch, reference_output, mesh_device, pcc_threshold=pcc_threshold)
    assert passing, f"MLP Pipeline test failed. Output: {output}"
    logger.info(f"MLP Pipeline test passed (CCL ops {dict(ccl_counts)}). Output: {output}")


def realistic_router_bias(num_experts):
    """e_score_correction_bias values with the checkpoint's statistics: few distinct values, |b| <= 0.01,
    bf16-exact multiples of 2^-9 (see the design notes on the layer 0-8 shard headers)."""
    return torch.randint(-5, 6, (num_experts,)).float() * ROUTER_BIAS_STEP


def setup_reference_layer(setup, layer_idx=0):
    """Random-init ``SolarOpenDecoderLayer`` at the real model dimensions (contract C11).

    A standalone layer holds ``torch.empty`` router/expert parameters (``_init_weights`` only runs inside
    ``PreTrainedModel.post_init``), so every MoE and attention weight is drawn from ``N(0, initializer_range)``
    (0.02, the model's own scale: router logits then have std ~1.3, the regime the bf16 activations resolve). The
    selection-only ``e_score_correction_bias`` buffer is set to realistic non-zero values so the biased top-k path
    is exercised. Both ``_attn_implementation`` and ``_experts_implementation`` must be "eager" for a standalone
    layer (transformers 5.x dispatch).
    """
    logger.info("Setting up reference layer...")
    from transformers.models.solar_open.modeling_solar_open import SolarOpenDecoderLayer

    config = setup["config"]
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    reference_layer = SolarOpenDecoderLayer(config, layer_idx=layer_idx)
    init_std = getattr(config, "initializer_range", 0.02)
    with torch.no_grad():
        for name, param in reference_layer.named_parameters():
            if name.startswith(("mlp.gate.", "mlp.experts.", "mlp.shared_experts.", "self_attn.")):
                param.normal_(0, init_std)
            # The checkpoint is bf16: keep every random weight bf16-exact (computed in fp32 like the HF model does),
            # so a weight the TT side stores in bf16 (the router gate) is the SAME operand on both sides and only the
            # device arithmetic is measured. (Weights stored in bfp8 on device are quantised further there anyway.)
            param.copy_(param.to(torch.bfloat16).float())
        reference_layer.mlp.gate.e_score_correction_bias.copy_(realistic_router_bias(config.num_local_experts))
    return reference_layer.eval()


def setup_decoder_layer(
    setup, reference_layer, local_batch_size, seq_len, layer_idx=0, paged_attention_config=None, moe_options=None
):
    """TT ``DecoderLayer`` fed with the reference layer's state dict (Meta-permuted q/k), production RoPE setup.

    ``moe_options`` defaults to ``MoEOptions.from_env()`` (the SOLAR_OPEN_* flags of the run)."""
    logger.info("Setting up TT decoder layer...")
    config = setup["config"]
    reference_state_swizzled = convert_hf_qkv_to_meta_format(reference_layer.state_dict(), config.head_dim)
    # Build the rope setup exactly like production (Model.__init__ -> create_rope_setup): the decode
    # transformation matrix is height-sharded one tile per user core and rotary_embedding_llama reads it
    # from the local core's L1, so its batch (and the row-sharding / mesh-dim handling on multi-row
    # meshes) must match the decode batch. A batch_size=1 setup leaves 31 of 32 cores reading
    # unallocated memory at local batch 32.
    users_row_sharded = setup["mesh_device"].shape[0] > 1 and local_batch_size > 1
    rope_setup = create_rope_setup(
        mesh_device=setup["mesh_device"],
        hf_config=config,
        max_local_batch_size=local_batch_size,
        users_row_sharded=users_row_sharded,
        datatype=ttnn.bfloat16,
        shard_batch_to_mesh_dim=0,
    )
    decoder_layer = DecoderLayer(
        setup["mesh_device"],
        config,
        reference_state_swizzled,
        layer_idx=layer_idx,
        ccl_manager=setup["ccl_manager"],
        dtype=setup["dtype"],
        mesh_config=setup["mesh_config"],
        transformation_mats=rope_setup.get_both_trans_mats(),
        max_seq_len=max(seq_len, 128),
        max_local_batch_size=local_batch_size,
        paged_attention_config=paged_attention_config,
        tokens_per_device=local_batch_size,
        moe_options=moe_options or MoEOptions.from_env(),
    )
    return decoder_layer


def make_paged_attention(mesh_device, local_batch_size, seq_len, block_size=64):
    """PagedAttentionConfig sized for ``local_batch_size`` users of ``max(seq_len, 128)`` tokens plus a sequential
    page table replicated on the mesh. Returns ``(paged_attention_config, page_table_tt)``."""
    from models.tt_transformers.tt.common import PagedAttentionConfig

    effective_seq_len = max(seq_len, 128)
    blocks_per_seq = max((effective_seq_len + block_size - 1) // block_size, 1)
    max_blocks = local_batch_size * blocks_per_seq
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_blocks)
    page_table_torch = torch.arange(max_blocks, dtype=torch.int32).reshape(local_batch_size, blocks_per_seq)
    page_table_tt = ttnn.from_torch(
        page_table_torch,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return paged_attention_config, page_table_tt


def build_rope_inputs(
    setup, config, hidden_states, batch_size, seq_len, pos_offset, local_batch_size, is_decode, cache_position=None
):
    """RoPE inputs for one test case, positions ``pos_offset .. pos_offset + seq_len - 1`` for every user.

    Returns ``(position_ids [B*S], (cos, sin) HF reference, [tt_cos, tt_sin] Meta-format device tensors,
    tt_position_idx)``. The TT tables come from ``compute_gather_cos_sin`` with the config's YaRN scaling (the
    same ``YarnRotaryEmbedding`` production's ``RotarySetup`` uses; attention factor 1.0693 on cos and sin), never
    from the unscaled ``precompute_freqs``. Decode cos/sin are height-sharded on the per-user grid Q/K/V use.
    ``cache_position`` (decode only) overrides the KV-cache slot the decode token is written to / attends up to
    (``tt_position_idx``); by default it equals the RoPE position.
    """
    from transformers.models.solar_open.modeling_solar_open import SolarOpenRotaryEmbedding

    mesh_device = setup["mesh_device"]
    is_row_sharded = mesh_device.shape[0] > 1 and local_batch_size > 1
    position_ids = torch.cat(
        [torch.arange(pos_offset, pos_offset + seq_len, dtype=torch.long) for _ in range(batch_size)]
    )

    # HF reference cos/sin (yarn-scaled, cast to the hidden dtype); [B, 1, hd] in decode, [1, S, hd] in prefill.
    rope_embeddings_ref = SolarOpenRotaryEmbedding(config)
    with torch.no_grad():
        position_embeddings_ref = rope_embeddings_ref(hidden_states, position_ids.unsqueeze(1 if is_decode else 0))

    # TT tables in Meta format: [1, 1, pos_offset + seq_len, head_dim] with the interleaved (cos, cos) pairs.
    cos_full, sin_full = compute_gather_cos_sin(
        dhead=config.head_dim,
        end=2 * (pos_offset + seq_len),
        theta=resolve_rope_theta(config),
        rope_scaling=rope_scaling_model_factory(config.rope_scaling),
    )
    cos_meta = cos_full[:, :, position_ids, :].reshape(1, batch_size, seq_len, config.head_dim)
    sin_meta = sin_full[:, :, position_ids, :].reshape(1, batch_size, seq_len, config.head_dim)

    mesh_mapper = (
        ttnn.ShardTensor2dMesh(dims=(-3, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    tt_cos = ttnn.from_torch(
        cos_meta, device=mesh_device, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    tt_sin = ttnn.from_torch(
        sin_meta, device=mesh_device, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    if is_decode:
        # rotary_embedding_llama reads cos/sin from the core that holds the user's Q shard, so the grids must agree
        # with RotarySetup / attention/decode.py: 8 wide for batches <= 8 or multiples of 32, the device compute
        # grid otherwise (13 wide for 16 users on Blackhole).
        batch_grid, _ = AttentionProgramConfig.get_decode_user_grid(mesh_device, local_batch_size)
        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, config.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        tt_cos = ttnn.interleaved_to_sharded(tt_cos, mem_config)
        tt_sin = ttnn.interleaved_to_sharded(tt_sin, mem_config)

    idx_mapper = (
        ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
        if is_row_sharded
        else None
    )
    cache_positions = position_ids if cache_position is None else torch.full_like(position_ids, cache_position)
    tt_position_idx = ttnn.from_torch(
        cache_positions.squeeze(),
        device=mesh_device,
        mesh_mapper=idx_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    return position_ids, position_embeddings_ref, [tt_cos, tt_sin], tt_position_idx


@parametrize_mesh_with_fabric([(1, 1), (1, 8), (4, 8)])
@parametrize_batch_seq(
    [
        (1, 1),  # decode, single user
        (128, 1),  # decode, row-sharded users (multi-row meshes only; skipped on 1xN)
        (32, 1),  # decode, 32 users on one mesh row (TP only, union-of-experts on the whole tile)
        (16, 1),  # decode, 16 users: exercises the device-grid (13-wide on Blackhole) per-user placement
        (1, 128),  # prefill (the traced prefill length; dense-bmm MoE path)
        (1, 1024),  # prefill 1k (expert-sorted hot/cold MoE path)
        (1, 4096),  # prefill 4k (one full sequence chunk)
    ],
    ids=[
        "decode_b1",
        "decode_b128",
        "decode_b32",
        "decode_b16",
        "prefill_128",
        "prefill_1024",
        "prefill_4096",
    ],
)
@pytest.mark.parametrize("layer_idx", [0], ids=["layer_0"])
@pytest.mark.parametrize(
    # Cover both the legacy non-paged kv-cache path and the paged path that vLLM
    # and the hybrid kv-cache-groups manager exercise. The paged path goes through
    # paged_fill_cache / paged_update_cache / paged_scaled_dot_product_attention_decode,
    # none of which the non-paged path touches.
    "paged",
    [False, True],
    ids=["unpaged", "paged"],
)
@pytest.mark.parametrize(
    # Position of the first token. 70000 lies beyond YaRN's original_max_position_embeddings (65536), so the
    # blended/interpolated frequency bands and the 1.0693 attention factor are exercised end to end. Prefill fills
    # the (max(seq_len, 128)-slot) test cache position-agnostically; decode cases place their DECODE_CONTEXT_LEN-token
    # context at slots 0..P with RoPE positions pos_offset .. pos_offset + P (see DECODE_CONTEXT_LEN).
    "pos_offset",
    [0, 70000],
    ids=["pos0", "pos70000"],
)
def test_decoder(
    mesh_device,
    device_params,
    batch_size,
    seq_len,
    layer_idx,
    paged,
    pos_offset,
    test_modules,
    test_thresholds,
    reset_seeds,
):
    """
    Test decoder layer components.

    Args:
        test_modules: Which modules to test (from --test-modules flag). Options:
            - "all": Test all components (default)
            - "attention": Test attention only
            - "rms_norm": Test RMS normalization only
            - "router": Test TopK router only
            - "experts": Test the routed experts only (controlled routing)
            - "shared_expert": Test the always-on shared expert only
            - "mlp": Test full MLP pipeline (router + experts + shared expert)
            - "decoder": Test full decoder layer only
            - Comma-separated: "attention,mlp" or "router,experts" etc.

    Usage:
        pytest test_modules.py  # runs all tests
        pytest test_modules.py --test-modules=attention
        pytest test_modules.py --test-modules=attention,mlp
    """
    # The paged/unpaged dimension only changes behavior for components that touch
    # the kv cache (attention + the full decoder layer). Skip non-kv-only paged
    # variants before constructing the reference model and device tensors.
    modules_to_test = set(test_modules.split(","))
    run_all = "all" in modules_to_test
    KV_USING_MODULES = {"attention", "decoder"}
    if paged and not run_all and not (modules_to_test & KV_USING_MODULES):
        pytest.skip(
            f"paged variant only exercises kv-cache-using components ({sorted(KV_USING_MODULES)}); "
            f"requested modules {sorted(modules_to_test)} don't touch the kv cache"
        )

    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] == 1 and batch_size > 32:
        pytest.skip(
            f"Skipping batch size {batch_size} for mesh shape {mesh_shape}. "
            "A single mesh row decodes at most 32 users; larger batches need row-sharding."
        )
    if mesh_shape == (1, 1) and batch_size > 1:
        pytest.skip(
            f"Skipping batch size {batch_size} on a single device (TP=1): multi-user decode is only "
            "validated with TP>1 (e.g. 1x8)."
        )
    if mesh_shape[0] > 1 and 1 < batch_size <= 32:
        pytest.skip(
            f"Skipping batch size {batch_size} for mesh shape {mesh_shape}: multi-row meshes batch users "
            "across rows (row-sharded, batch > 32); the single-row multi-user path is covered on 1xN."
        )

    assert batch_size == 1 or seq_len == 1, "Only single user prefill or single token decode is supported"
    is_decode = seq_len == 1
    mode = "decode" if is_decode else "prefill"

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    pcc_thresholds = test_thresholds[setup["model_args"].model_name][mode]
    config = setup["config"]
    moe_options = MoEOptions.from_env()

    if batch_size > 32:
        is_row_sharded = True
        assert batch_size % mesh_device.shape[0] == 0, "Batch size must be evenly divisible by mesh device shape"
        local_batch_size = batch_size // mesh_device.shape[0]
    else:
        is_row_sharded = False
        local_batch_size = batch_size

    # Paged attention: allocate a PagedAttentionConfig sized to fit the case's seq_len (one user per call
    # here) and a sequential page table. Both the decoder layer's kv cache allocation and the per-call
    # paged_fill_cache / paged_sdpa_decode invocations are gated by the same config + page_table, so the
    # test exercises the full paged path end-to-end (the `paged=False` path goes through ttnn.fill_cache +
    # non-paged SDPA).
    paged_attention_config, page_table_tt = (
        make_paged_attention(mesh_device, local_batch_size, seq_len) if paged else (None, None)
    )

    reference_layer = setup_reference_layer(setup, layer_idx=layer_idx)
    decoder_layer = setup_decoder_layer(
        setup,
        reference_layer,
        local_batch_size,
        seq_len,
        layer_idx=layer_idx,
        paged_attention_config=paged_attention_config,
    )

    # bf16-exact like every activation the device receives (the residual stream is bf16).
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size).to(torch.bfloat16).float()

    # Causal mask for the reference (every Solar-Open layer is full attention); none needed in decode.
    mask = None if is_decode else torch.triu(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=1)

    # Decode: the token is decoded at RoPE position pos_offset + P over a P-token context per user (cache slots 0..P);
    # its own rope mats / position index below are built for that step, the prefix's for the P earlier positions.
    decode_context = (
        build_decode_context(setup, config, batch_size, pos_offset, local_batch_size) if is_decode else None
    )
    context_len = decode_context.context_len if decode_context is not None else 0
    _, position_embeddings_ref, rope_mats, tt_position_idx = build_rope_inputs(
        setup,
        config,
        hidden_states,
        batch_size,
        seq_len,
        pos_offset + context_len,
        local_batch_size,
        is_decode,
        cache_position=context_len if decode_context is not None else None,
    )

    logger.info(
        f"Running tests: {test_modules} (paged={paged}, pos_offset={pos_offset}, "
        f"decode context {context_len} tokens per user)"
    )

    # Helper to check if a module should be tested. Non-kv components only run on
    # the unpaged variant (their behavior is independent of paged kv-cache state).
    def should_test(module_name):
        if paged and module_name not in KV_USING_MODULES:
            return False
        return run_all or module_name in modules_to_test

    if should_test("router"):
        logger.info("Testing TopK Router...")
        run_topk_router_component(
            setup["mesh_device"],
            hidden_states.shape,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            is_row_sharded=is_row_sharded,
            pcc_threshold=pcc_thresholds["router"],
            fp32_logits=moe_options.router_fp32_logits,
        )

    if should_test("experts"):
        logger.info(f"Testing Experts (routed only) for mesh shape {mesh_shape}...")
        run_experts_component(
            setup["mesh_device"],
            hidden_states.shape,
            config,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            pcc_threshold=pcc_thresholds["experts"],
        )

    if should_test("shared_expert"):
        logger.info("Testing Shared Expert...")
        run_shared_expert_component(
            setup["mesh_device"],
            hidden_states.shape,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            pcc_threshold=pcc_thresholds["shared_expert"],
        )

    if should_test("attention"):
        logger.info(f"Testing Attention (paged={paged})...")
        run_attention_component(
            setup["mesh_device"],
            hidden_states.shape,
            mask,
            position_embeddings_ref,
            rope_mats,
            tt_position_idx,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            is_row_sharded=is_row_sharded,
            pcc_threshold=pcc_thresholds["attention"],
            page_table=page_table_tt,
            decode_context=decode_context,
        )

    if should_test("rms_norm"):
        logger.info("Testing RMS Norm...")
        run_rms_norm_component(
            setup["mesh_device"],
            hidden_states.shape,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            is_row_sharded=is_row_sharded,
            pcc_threshold=pcc_thresholds["rms_norm"],
        )

    if should_test("mlp"):
        logger.info("Testing Full MLP Pipeline...")
        run_full_mlp_pipeline(
            setup["mesh_device"],
            hidden_states.shape,
            reference_layer,
            decoder_layer,
            is_decode=is_decode,
            is_row_sharded=is_row_sharded,
            pcc_threshold=pcc_thresholds["mlp"],
        )

    if should_test("decoder"):
        logger.info("Testing Full Decoder Layer...")
        mesh_mapper = (
            ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
            if is_row_sharded
            else None
        )
        tt_hidden_states = ttnn.from_torch(
            hidden_states.reshape(1, 1, batch_size * seq_len, -1),
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            dtype=ACTIVATION_DTYPE,
        )

        reference_input, reference_position_embeddings, reference_mask = hidden_states, position_embeddings_ref, mask
        if decode_context is not None:
            # The full layer writes input_layernorm(prefix) into the cache (that is what its attention consumes), then
            # decodes the token over the P-token context; the HF layer sees prefix + token and its last row is compared.
            decode_context.fill_kv_cache(mesh_device, decoder_layer, page_table_tt, apply_input_norm=True)
            reference_input, reference_position_embeddings, reference_mask = decode_context.reference_inputs(
                hidden_states
            )
        with torch.no_grad():
            reference_output = reference_layer(
                reference_input, attention_mask=reference_mask, position_embeddings=reference_position_embeddings
            )
        if decode_context is not None:
            reference_output = reference_output[:, -1:, :]

        with count_ccl_ops() as ccl_counts:
            tt_output = decoder_layer(
                tt_hidden_states,
                position_embeddings=rope_mats,
                position_idx=tt_position_idx,
                page_table=page_table_tt,
                is_decode=is_decode,
            )
        logger.info(f"Decoder layer CCL ops (attention + MoE): {dict(ccl_counts)}")

        mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_shape)
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer)[
            ..., : batch_size * seq_len, : config.hidden_size
        ]
        assert (
            tt_output.dtype == ttnn.bfloat16
        ), f"the residual stream must stay bf16, the layer emitted {tt_output.dtype}"
        if not is_row_sharded:
            assert_replicated(tt_output, "decoder layer output")
        passing, output = compare_tensors(
            tt_output_torch.squeeze(), reference_output.squeeze(), mesh_device, pcc_threshold=pcc_thresholds["decoder"]
        )
        assert passing, f"Decoder Layer test failed. Output: {output}"
        logger.info(f"Decoder Layer test passed. Output: {output}")

    logger.info(f"Tests completed successfully: {', '.join(sorted(modules_to_test))}")


def constant_shared_expert(value):
    """Stub with the SharedExpert calling convention (contract C4): ``[1, 1, rows, H]`` bf16 TILE interleaved with
    the (padded) shape of its input, every element ``value``. Accepts the ``is_decode`` kwarg the MLP's partial passes.
    """

    def hook(x, is_decode=None):
        return ttnn.full_like(x, value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=x.memory_config())

    return hook


@parametrize_mesh_with_fabric([(1, 8)])
@parametrize_batch_seq(
    [(1, 1), (32, 1), (1, 128), (1, 1024)],
    ids=["decode_b1", "decode_b32", "prefill_128", "prefill_1024"],
)
def test_experts_shared_expert_hook(mesh_device, device_params, batch_size, seq_len, reset_seeds):
    """The experts' ``shared_expert`` hook (contract C3): its partial is added on every TP device BEFORE the single
    all_reduce.

    Runs the routed experts three times on the same input and routing: without a hook, with a stub returning zeros
    (output must be unchanged) and with a stub returning the constant ``SHARED_HOOK_CONSTANT`` (the all-reduced output
    must shift by exactly ``tp * c`` on every element: ``c`` would mean the add happened after the all_reduce, ``0`` that
    the hook was ignored). Covers the single-user decode path (``[1, 1, 1, H]``), the batched decode path (32-row
    padded input) and the prefill chunk loop (dense bmm at 128 tokens, split/sorted paths at 1024).
    """
    mesh_shape = tuple(mesh_device.shape)
    tp = mesh_shape[1]
    is_decode = seq_len == 1
    num_tokens = batch_size * seq_len

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    hidden_size = config.hidden_size
    reference_layer = setup_reference_layer(setup)
    # The hook contract belongs to the unfused layout: pin it regardless of SOLAR_OPEN_FUSE_SHARED_EXPERT.
    decoder_layer = setup_decoder_layer(
        setup,
        reference_layer,
        batch_size,
        seq_len,
        moe_options=dataclasses.replace(MoEOptions.from_env(), fuse_shared_expert=False),
    )
    experts = decoder_layer.mlp.experts
    # The single-user case runs the indexed/gather path when MoEOptions.indexed_decode is on (as MLP.route would).
    use_indexed = is_decode and num_tokens == 1 and decoder_layer.mlp.indexed_decode

    hidden_states = torch.randn(1, 1, num_tokens, hidden_size)
    router_indices, routing_dense, routing_topk = uniform_random_routing(
        num_tokens, config.num_local_experts, config.num_experts_per_tok
    )
    replicate = ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_shape)

    def run(shared_expert):
        # The experts consume their input; the routing tensor is the caller's (a padded copy is made in decode).
        tt_hidden = ttnn.from_torch(
            hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )
        if use_indexed:
            indexed = indexed_routing_from_torch(mesh_device, router_indices, routing_topk)
            tt_out = experts(
                hidden_states=tt_hidden, is_decode=True, shared_expert=shared_expert, indexed_routing=indexed
            )
            indexed.deallocate()
        else:
            tt_routing = ttnn.from_torch(
                routing_dense, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
            )
            tt_out = experts(
                hidden_states=tt_hidden,
                topk_expert_weights=tt_routing,
                is_decode=is_decode,
                shared_expert=shared_expert,
            )
            tt_routing.deallocate(True)
        out = ttnn.to_torch(tt_out, mesh_composer=mesh_composer)[..., :num_tokens, :hidden_size]
        tt_out.deallocate(True)
        return out.reshape(num_tokens, hidden_size).float()

    baseline = run(None)
    with_zeros = run(constant_shared_expert(0.0))
    with_constant = run(constant_shared_expert(SHARED_HOOK_CONSTANT))

    zero_passing, zero_output = compare_tensors(with_zeros, baseline, mesh_device, pcc_threshold=SHARED_HOOK_ZERO_PCC)
    zero_max_diff = (with_zeros - baseline).abs().max().item()
    expected_shift = tp * SHARED_HOOK_CONSTANT
    shift = with_constant - baseline
    shift_err = (shift - expected_shift).abs()
    logger.info(
        f"shared-expert hook (batch {batch_size} x seq {seq_len}, TP={tp}, "
        f"{'indexed' if use_indexed else 'dense'} routing): zeros hook {zero_output} "
        f"(max |diff| {zero_max_diff:.4f}); constant hook shift mean {shift.mean():.4f} / expected {expected_shift:.1f}, "
        f"max |err| {shift_err.max():.4f}"
    )
    assert zero_passing, f"a shared expert returning zeros changed the routed output: {zero_output}"
    assert shift_err.max() <= SHARED_HOOK_REL_TOL * expected_shift, (
        f"a shared expert returning {SHARED_HOOK_CONSTANT} shifted the all-reduced output by "
        f"{shift.mean():.4f} (min {shift.min():.4f}, max {shift.max():.4f}) instead of tp * c = {expected_shift:.1f}: "
        "the shared partial is not summed on every device before the single TP all_reduce"
    )


def run_model_forward_test(
    mesh_device,
    config,
    state_dict_meta,
    reference_model,
    mesh_config,
    batch_size,
    seq_len,
    is_decode,
    pcc_threshold=0.88,
    tensor_cache_path=None,
):
    """
    Run a single forward pass test comparing TT model to reference model.

    Args:
        mesh_device: TTNN mesh device
        config: HuggingFace config (with num_hidden_layers already modified)
        state_dict_meta: Model weights in meta format for TT model
        reference_model: Already-instantiated HuggingFace reference model (eval mode)
        mesh_config: Mesh configuration
        batch_size: Batch size
        seq_len: Sequence length
        is_decode: True for decode mode (seq_len=1), False for prefill mode
        pcc_threshold: PCC threshold for comparison
    """
    from models.demos.solar_open.tt.ccl import CCLManager
    from models.demos.solar_open.tt.model import Model
    from models.demos.solar_open.utils.general_utils import get_default_num_links

    if batch_size > 32:
        is_row_sharded = True
        assert batch_size % mesh_device.shape[0] == 0, "Batch size must be divisible by mesh rows"
        local_batch_size = batch_size // mesh_device.shape[0]
    else:
        is_row_sharded = False
        local_batch_size = batch_size

    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))

    tt_model = Model(
        mesh_device=mesh_device,
        hf_config=config,
        state_dict=state_dict_meta,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=tensor_cache_path,
        paged_attention_config=None,
        mesh_config=mesh_config,
        create_kv_cache=True,
        max_local_batch_size=local_batch_size,
        users_row_sharded=is_row_sharded,
        moe_options=MoEOptions.from_env(),
        max_seq_len=max(seq_len, 128),  # sizes the unpaged KV cache of this test only
    )

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    with torch.no_grad():
        reference_output = reference_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,  # create_causal_mask builds the causal mask
            use_cache=False,
        )
        reference_logits = reference_output.logits  # [batch_size, seq_len, vocab_size]

    if is_decode:
        tokens_flat = input_ids.reshape(-1)  # [batch_size]
        current_pos = torch.zeros(batch_size, dtype=torch.long)
        tt_tokens, tt_current_pos, tt_rope_idxs, _ = tt_model.prepare_inputs_decode(
            tokens_flat, current_pos, page_table=None
        )
        tt_logits, _ = tt_model.ttnn_decode_forward(
            tokens=tt_tokens,
            current_pos=tt_current_pos,
            rot_mat_idxs=tt_rope_idxs,
            page_table=None,
            kv_cache=None,
        )
    else:
        tt_tokens = ttnn.from_torch(
            input_ids.unsqueeze(0).unsqueeze(0),  # [1, 1, batch, seq]
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        tt_embeds = ttnn.embedding(tt_tokens, tt_model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        if len(tt_embeds.shape) == 3:
            tt_embeds = ttnn.unsqueeze_to_4D(tt_embeds)
        tt_logits = tt_model.ttnn_prefill_forward(
            x=tt_embeds,
            user_id=0,
            rot_mats_global=None,  # the model computes RoPE itself
            page_table=None,
            kv_cache=None,
            get_last_token=-1,  # all tokens
        )

    mesh_composer_dims = (-2, -1) if is_row_sharded else (0, -1)
    tt_logits_torch = ttnn.to_torch(
        tt_logits,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=mesh_composer_dims, mesh_shape=tuple(mesh_device.shape)
        ),
    )
    tt_logits_torch = tt_logits_torch[0, 0].reshape(batch_size, seq_len, -1)

    # The lm_head weight is padded to padded_vocab_size (per-device pow2 shards for on-device sampling).
    vocab_size = config.vocab_size
    if tt_logits_torch.shape[-1] > vocab_size:
        tt_logits_torch = tt_logits_torch[:, :, :vocab_size]

    return compare_tensors(tt_logits_torch, reference_logits, mesh_device, pcc_threshold=pcc_threshold)


@parametrize_mesh_with_fabric([(1, 1), (1, 8), (4, 8)])
@pytest.mark.parametrize(
    "batch_size, seq_len, mode",
    [
        (1, 128, "prefill"),
        (128, 1, "decode"),
        (32, 1, "decode"),
        (1, 1, "decode"),  # single user: the indexed/gather expert path (MoEOptions.indexed_decode) end to end
    ],
    ids=[
        "prefill_b1_s128",
        "decode_b128_s1",
        "decode_b32_s1",
        "decode_b1_s1",
    ],
)
@pytest.mark.parametrize("num_layers", [1], ids=["1_layer"])
def test_model(mesh_device, device_params, batch_size, seq_len, mode, num_layers, reset_seeds):
    """
    Full model forward pass (embedding -> layers -> norm -> lm_head) vs a random-init ``SolarOpenForCausalLM``.

    1. Loads the model config and overrides num_hidden_layers
    2. Builds the TT model and the reference model from the same random weights (no checkpoint needed)
    3. Runs prefill (batch=1, seq=128) or decode (batch=32, seq=1) forward passes
    4. Compares the logits using PCC
    """
    from transformers.models.solar_open.modeling_solar_open import SolarOpenForCausalLM

    from models.demos.solar_open.config import MeshConfig, ModeConfig

    mesh_shape = tuple(mesh_device.shape)

    if mesh_shape[0] == 1 and batch_size > 32:
        pytest.skip(
            f"Skipping batch size {batch_size} for mesh shape {mesh_shape}. A single mesh row decodes at most 32 users."
        )
    if mesh_shape == (1, 1) and batch_size > 1:
        pytest.skip(f"Skipping batch size {batch_size} on a single device (TP=1): multi-user decode needs TP>1.")
    if mesh_shape[0] > 1 and 1 < batch_size <= 32:
        pytest.skip(
            f"Skipping batch size {batch_size} for mesh shape {mesh_shape}: multi-row meshes batch users across "
            "rows (row-sharded, batch > 32)."
        )

    is_decode = mode == "decode"

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]

    original_num_layers = config.num_hidden_layers
    config.num_hidden_layers = num_layers
    logger.info(f"Overriding num_hidden_layers from {original_num_layers} to {num_layers}")
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"

    mesh_config = MeshConfig(mesh_shape, decode=ModeConfig(tp=mesh_shape[1], ep=mesh_shape[0]))

    # Random-init reference at the real vocab (196608 x 4096 fp32 embedding + lm_head = 6.4 GB host). PreTrainedModel
    # initialises the router/expert/linear weights with initializer_range (0.02); only the zero
    # e_score_correction_bias buffers need realistic values so the biased selection path is exercised.
    tensor_cache_path = None
    reference_model_hf = SolarOpenForCausalLM(config)
    with torch.no_grad():
        for layer in reference_model_hf.model.layers:
            layer.mlp.gate.e_score_correction_bias.copy_(realistic_router_bias(config.num_local_experts))
    reference_model_hf.eval()
    state_dict_meta = convert_hf_qkv_to_meta_format(reference_model_hf.state_dict(), config.head_dim)

    logger.info(f"Running {mode} test with batch_size={batch_size}, seq_len={seq_len}, num_layers={num_layers}")

    passing, output = run_model_forward_test(
        mesh_device=mesh_device,
        config=config,
        state_dict_meta=state_dict_meta,
        reference_model=reference_model_hf,
        mesh_config=mesh_config,
        batch_size=batch_size,
        seq_len=seq_len,
        is_decode=is_decode,
        pcc_threshold=0.95 if num_layers == 1 else 0.85,
        tensor_cache_path=tensor_cache_path,
    )
    assert passing, f"Model {mode} test failed. PCC: {output}"
    logger.info(f"Model {mode} test passed. PCC: {output}")
