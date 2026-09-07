# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Layer 0 of Solar-Open-100B with REAL weights vs the transformers ``SolarOpenDecoderLayer``.

The random-weight matrix in ``tests/unit/test_modules.py`` cannot see the effects that only real weights have:
the routing statistics of the union-of-experts decode path (~112 of 128 experts active for 32 users), the
expert-sorted hot/cold prefill planner, the bf16-exact ``e_score_correction_bias`` values and the real magnitude
of the residual stream. This test reads ``model.layers.0.*`` (plus ``model.embed_tokens.weight`` so layer 0 sees
its real input: embeddings of real token ids) straight from the safetensors shards with ``safe_open`` -- it needs
only ``model-00001/00002-of-00042.safetensors`` (or whatever ``model.safetensors.index.json`` names for layer 0),
not the whole 205 GB checkpoint -- stacks the 128 per-expert tensors into the transformers >= 5 fused layout of
contract C1 (``gate_up_proj [128, 2560, 4096]`` gate rows first, ``down_proj [128, 4096, 1280]``), loads it into an
fp32 HF layer (``strict=True``; the router bias stays fp32) and into the TT ``DecoderLayer`` through
``convert_hf_qkv_to_meta_format`` -- the same weight path the demo takes.

Decode cases attend over a ``tm.DECODE_CONTEXT_LEN``-token context of real embeddings per user (written into the
TT KV cache through the prefill path), so decode RoPE, SDPA and the cache update are measured, not just Wo * V.

Per (batch, seq) case it checks the router (100 % expert-set agreement on the decisive tokens, a cap on the
near-tie flips, dense PCC on the agreeing tokens at every T and on all tokens from 128 tokens on -- the metric of
``test_modules.run_topk_router_component``; the reference routes the bf16 copy of the input the device gets), the
MoE block (routed + shared) and the full layer, and logs every measured value: these numbers are the real-weight
baseline the thresholds below are re-baselined from (design D13).

    HF_MODEL=/path/to/Solar-Open-100B MESH_DEVICE=P150x8 \
        pytest models/demos/solar_open/tests/test_layer0_real_weights.py -k 1x8
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn

from .test_factory import TestFactory, compare_tensors, parametrize_mesh_with_fabric
from .unit import test_modules as tm

LAYER_IDX = 0
# Shards that hold layer 0 when model.safetensors.index.json has not been downloaded yet (verified from the shard
# headers: shard 1 = embed_tokens + layer-0 attention + experts 0..101, shard 2 = experts 102..127 + gate/bias +
# shared_experts + norms). With the index present the weight_map is authoritative.
LAYER0_SHARDS_WITHOUT_INDEX = ("model-00001-of-00042.safetensors", "model-00002-of-00042.safetensors")
EMBED_KEY = "model.embed_tokens.weight"
# Prompts whose token ids feed layer 0 (real embeddings -> real routing statistics); random ids when no tokenizer.
PROMPTS_FILE = "models/demos/solar_open/demo/sample_prompts/input_data_questions_ko_en_prefill_128.json"

# Real-weight thresholds (design 5.5 / D13), re-baselined from the first device run (2026-09-07, fused fp32 router,
# bfp8 experts): router set agreement 1.0 (T=1, 32), 0.9766 (3/128) and 0.9912 (9/1024) -- every flipped token a
# near tie (8th-vs-9th biased margin <= 1.7e-4, 0 decisive). Real layer 0 makes near ties common: gate rms 0.041 and
# MoE input rms 0.16 give router logits of std ~0.5, so ~16 % of the tokens have a margin < 1e-3, and the device's
# fp32 linear (max err 1.4e-3, mean 1e-4 vs torch on the same bf16 operands) plus the fused kernel's internal sigmoid
# decide those either way. Correctness is guarded by the decisive-token check (100 % agreement on margin >= 1e-3);
# the raw bound only caps the near-tie noise (~0.02 below measured, with test_modules' floor of 3 mismatching tokens
# for the decode batches, where one flip would already be 3 % of 32 users). Measured mlp PCC 0.9990 (decode b1),
# 0.9993 (b32), 0.9997 (prefill 128 / 1024) and decoder PCC 0.9997 / 0.9995 / 0.9975 / 0.9940, paged == unpaged.
REAL_WEIGHT_THRESHOLDS = {
    "router_set_agreement": 0.95,  # fraction of tokens with the reference expert set (near ties included)
    "router_dense_pcc": 0.99,  # on the agreeing tokens at every T; on all tokens from ROUTER_ALL_TOKEN_PCC_MIN_TOKENS
    "mlp": 0.97,  # 0.95 -> 0.97 (D13, ~0.02 below the measured 0.9990 decode floor)
    "decoder": 0.97,  # 0.95 -> 0.97 (D13, ~0.02 below the measured 0.9940 prefill-1024 floor)
}


def _snapshot_dir():
    """The local checkpoint directory (``HF_MODEL``), or skip when it is not a directory."""
    model_path = os.getenv("HF_MODEL")
    if not model_path or not os.path.isdir(model_path):
        pytest.skip("HF_MODEL must point at a local Solar-Open-100B checkpoint directory (shards 1-2 are enough)")
    return Path(model_path)


def _shards_holding(snapshot, key_prefixes):
    """Shard files that hold any key starting with one of ``key_prefixes`` (index.json when present)."""
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = sorted({f for k, f in weight_map.items() if k.startswith(tuple(key_prefixes))})
    else:
        files = list(LAYER0_SHARDS_WITHOUT_INDEX)
    missing = [f for f in files if not (snapshot / f).is_file()]
    if missing:
        pytest.skip(f"checkpoint shards for layer {LAYER_IDX} not downloaded yet: {missing}")
    return [snapshot / f for f in files]


def load_layer_state_dict(snapshot, layer_idx, num_experts, with_embeddings=True):
    """Read ``model.layers.{layer_idx}.*`` (keys relative to the layer) in the fused transformers >= 5 layout plus,
    optionally, the embedding table. Returns ``(layer_state_dict, embed_weight | None)``."""
    prefix = f"model.layers.{layer_idx}."
    prefixes = [prefix] + ([EMBED_KEY] if with_embeddings else [])
    raw, embed = {}, None
    for shard in _shards_holding(snapshot, prefixes):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    raw[key[len(prefix) :]] = handle.get_tensor(key)
                elif with_embeddings and key == EMBED_KEY:
                    embed = handle.get_tensor(key)
    if not raw:
        pytest.skip(f"no tensors for layer {layer_idx} found in the downloaded shards")

    state_dict = {k: v for k, v in raw.items() if not k.startswith("mlp.experts.")}
    try:
        state_dict["mlp.experts.gate_up_proj"] = torch.stack(
            [
                torch.cat([raw[f"mlp.experts.{e}.gate_proj.weight"], raw[f"mlp.experts.{e}.up_proj.weight"]], dim=0)
                for e in range(num_experts)
            ]
        )  # [E, 2I, H], gate rows first (== the qwen2_moe conversion mapping of from_pretrained)
        state_dict["mlp.experts.down_proj"] = torch.stack(
            [raw[f"mlp.experts.{e}.down_proj.weight"] for e in range(num_experts)]
        )  # [E, H, I]
    except KeyError as exc:
        pytest.skip(f"layer {layer_idx} is only partially downloaded (missing {exc})")
    if with_embeddings and embed is None:
        pytest.skip(f"{EMBED_KEY} not found in the downloaded shards")
    return state_dict, embed


def _check_contract_c1(state_dict, config):
    """Shapes and dtypes of contract C1 for one layer (relative keys)."""
    H, E, I = config.hidden_size, config.num_local_experts, config.moe_intermediate_size
    Q, KV = config.num_attention_heads * config.head_dim, config.num_key_value_heads * config.head_dim
    expected = {
        "input_layernorm.weight": (H,),
        "post_attention_layernorm.weight": (H,),
        "self_attn.q_proj.weight": (Q, H),
        "self_attn.k_proj.weight": (KV, H),
        "self_attn.v_proj.weight": (KV, H),
        "self_attn.o_proj.weight": (H, Q),
        "mlp.gate.weight": (E, H),
        "mlp.gate.e_score_correction_bias": (E,),
        "mlp.experts.gate_up_proj": (E, 2 * I, H),
        "mlp.experts.down_proj": (E, H, I),
        "mlp.shared_experts.gate_proj.weight": (I * config.n_shared_experts, H),
        "mlp.shared_experts.up_proj.weight": (I * config.n_shared_experts, H),
        "mlp.shared_experts.down_proj.weight": (H, I * config.n_shared_experts),
    }
    assert set(state_dict) == set(expected), f"unexpected layer keys: {sorted(set(state_dict) ^ set(expected))}"
    for key, shape in expected.items():
        assert tuple(state_dict[key].shape) == shape, f"{key}: {tuple(state_dict[key].shape)} != {shape}"
        want = torch.float32 if key.endswith("e_score_correction_bias") else torch.bfloat16
        assert state_dict[key].dtype == want, f"{key}: dtype {state_dict[key].dtype} != {want}"
    bias = state_dict["mlp.gate.e_score_correction_bias"]
    logger.info(
        f"layer {LAYER_IDX} router bias: min {bias.min():.5f} max {bias.max():.5f}, "
        f"{bias.unique().numel()} distinct values; gate weight rms {state_dict['mlp.gate.weight'].float().pow(2).mean().sqrt():.4f}"
    )


def _token_ids(snapshot, num_tokens, vocab_size, model_args):
    """Real token ids for layer 0's input: chat-templated prompts when the tokenizer is available, else random."""
    ids = []
    tokenizer = getattr(model_args, "tokenizer", None)
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), trust_remote_code=True)
        except Exception as exc:  # tokenizer files may not be downloaded yet
            logger.warning(f"tokenizer unavailable ({type(exc).__name__}); using random token ids for layer 0's input")
            tokenizer = None
    if tokenizer is not None and os.path.isfile(PROMPTS_FILE):
        prompts = [p["prompt"] for p in json.loads(Path(PROMPTS_FILE).read_text())]
        for prompt in prompts:
            chat = [{"role": "user", "content": prompt}]
            encoded = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
            encoded = encoded["input_ids"] if isinstance(encoded, dict) or hasattr(encoded, "input_ids") else encoded
            ids.extend(int(t) for t in encoded)
            if len(ids) >= num_tokens:
                break
    if len(ids) < num_tokens:
        generator = torch.Generator().manual_seed(1234)
        ids.extend(torch.randint(0, vocab_size, (num_tokens - len(ids),), generator=generator).tolist())
    return torch.tensor(ids[:num_tokens], dtype=torch.long)


@pytest.fixture(scope="module")
def layer0_weights():
    """``(state_dict, embed_weight)`` of layer 0, read once per module (~4.2 GB bf16 + 1.6 GB embeddings)."""
    from transformers import AutoConfig

    snapshot = _snapshot_dir()
    config = AutoConfig.from_pretrained(str(snapshot), trust_remote_code=True)
    state_dict, embed = load_layer_state_dict(snapshot, LAYER_IDX, config.num_local_experts)
    _check_contract_c1(state_dict, config)
    return state_dict, embed


def build_reference_layer(config, state_dict):
    """fp32 ``SolarOpenDecoderLayer`` holding the real bf16 weights exactly (the reference computes in fp32; the
    checkpoint's bf16 values are representable, so this isolates the TT bfp8/bf16 error)."""
    from transformers.models.solar_open.modeling_solar_open import SolarOpenDecoderLayer

    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    layer = SolarOpenDecoderLayer(config, layer_idx=LAYER_IDX)
    layer.load_state_dict(state_dict, strict=True)
    assert layer.mlp.gate.e_score_correction_bias.dtype == torch.float32
    return layer.eval()


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [(1, 1), (32, 1), (1, 128), (1, 1024)],
    ids=["decode_b1", "decode_b32", "prefill_128", "prefill_1024"],
)
@pytest.mark.parametrize("paged", [False, True], ids=["unpaged", "paged"])
@parametrize_mesh_with_fabric([(1, 8)])
def test_layer0_real_weights(mesh_device, device_params, batch_size, seq_len, paged, layer0_weights, reset_seeds):
    """Router / MoE / full-layer PCC of layer 0 with real weights and real embeddings as input."""
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] != 1 or mesh_shape[1] != 8:
        pytest.skip(f"real-weight layer test is sized for the 1x8 TP=8 mesh, got {mesh_shape}")
    is_decode = seq_len == 1
    num_tokens = batch_size * seq_len

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)  # dummy ModelArgs: config only
    config = setup["config"]
    state_dict, embed = layer0_weights

    reference_layer = build_reference_layer(config, state_dict)
    paged_attention_config, page_table_tt = (
        tm.make_paged_attention(mesh_device, batch_size, seq_len) if paged else (None, None)
    )
    decoder_layer = tm.setup_decoder_layer(
        setup, reference_layer, batch_size, seq_len, layer_idx=LAYER_IDX, paged_attention_config=paged_attention_config
    )

    # Layer 0's real input: bf16 embeddings of real token ids (fp32 copies for the reference). Decode cases attend
    # over a DECODE_CONTEXT_LEN-token context of real embeddings per user (tm.DecodeContext), written into the TT KV
    # cache through the prefill path; the HF reference sees prefix + token and its last row is compared.
    context_len = tm.DECODE_CONTEXT_LEN if is_decode else 0
    token_ids = _token_ids(
        _snapshot_dir(), batch_size * (seq_len + context_len), config.vocab_size, setup["model_args"]
    )
    token_ids = token_ids.reshape(batch_size, seq_len + context_len)
    hidden_states = embed[token_ids[:, context_len:]].float().reshape(batch_size, seq_len, config.hidden_size)
    mask = None if is_decode else torch.triu(torch.full((1, 1, seq_len, seq_len), -float("inf")), diagonal=1)
    decode_context = (
        tm.build_decode_context(
            setup, config, batch_size, 0, batch_size, prefix=embed[token_ids[:, :context_len]].float()
        )
        if is_decode
        else None
    )
    _, position_embeddings_ref, rope_mats, tt_position_idx = tm.build_rope_inputs(
        setup,
        config,
        hidden_states,
        batch_size,
        seq_len,
        context_len,
        batch_size,
        is_decode,
        cache_position=context_len,
    )
    if decode_context is not None:
        ref_input, ref_position_embeddings, ref_mask = decode_context.reference_inputs(hidden_states)
    else:
        ref_input, ref_position_embeddings, ref_mask = hidden_states, position_embeddings_ref, mask
    replicate = ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_shape)
    results = {}

    # 1. Router on the post-attention-norm input the real MoE sees (same routing statistics as production).
    with torch.no_grad():
        attn_out, _ = reference_layer.self_attn(
            hidden_states=ref_input, position_embeddings=ref_position_embeddings, attention_mask=ref_mask
        )
        if decode_context is not None:
            attn_out = attn_out[:, -1:, :]
        moe_input = reference_layer.post_attention_layernorm(hidden_states + attn_out)
    # The device routes the bf16 copy of moe_input, so the reference must route the SAME values (as the router
    # component in test_modules does): fp32-vs-bf16 input rounding moves the real layer-0 logits by up to 4.5e-3,
    # more than the decisive margin, and changed 2/128 and 6/1024 reference decisions on its own (probe 2026-09-07).
    moe_input_bf16 = moe_input.to(torch.bfloat16).float()
    ref_indices, ref_weights, ref_dense, margin = tm.reference_routing(
        reference_layer, moe_input_bf16.reshape(-1, config.hidden_size)
    )
    tt_moe_input = ttnn.from_torch(
        moe_input_bf16.reshape(1, 1, -1, config.hidden_size),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )
    _, tt_dense = decoder_layer.mlp.router(tt_moe_input, is_decode=is_decode)
    tt_dense_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_dense)[0]).float()[:num_tokens, : ref_dense.shape[-1]]
    same_set = tm.expert_set_agreement(tt_dense_torch, ref_indices)
    decisive = margin >= tm.ROUTER_DECISIVE_MARGIN
    agreement = same_set.float().mean().item()
    num_mismatch = int((~same_set).sum())
    decisive_mismatch = int((~same_set & decisive).sum())
    max_mismatch = max(
        tm.ROUTER_MIN_MISMATCH_ALLOWANCE,
        int(round((1.0 - REAL_WEIGHT_THRESHOLDS["router_set_agreement"]) * num_tokens)),
    )
    mismatch_margin_max = margin[~same_set].max().item() if num_mismatch else 0.0
    dense_all_passing, dense_pcc = compare_tensors(
        tt_dense_torch, ref_dense, mesh_device, REAL_WEIGHT_THRESHOLDS["router_dense_pcc"]
    )
    dense_agree_passing, dense_pcc_agree = True, None
    if same_set.any():
        dense_agree_passing, dense_pcc_agree = compare_tensors(
            tt_dense_torch[same_set], ref_dense[same_set], mesh_device, REAL_WEIGHT_THRESHOLDS["router_dense_pcc"]
        )
    active_experts = int((tt_dense_torch.sum(0) > 0).sum())
    results["router"] = (
        f"set agreement {agreement:.4f} ({num_mismatch}/{num_tokens} differ, allowed {max_mismatch}; {decisive_mismatch} "
        f"decisive; near-tie fraction {1.0 - decisive.float().mean().item():.3f}, max margin of a flipped token "
        f"{mismatch_margin_max:.2e}), dense PCC {dense_pcc} (all tokens) / {dense_pcc_agree} (agreeing tokens), "
        f"union of experts {active_experts}/{ref_dense.shape[-1]}"
    )

    # 2. MoE block (router + routed experts + shared expert, one all-reduce).
    with torch.no_grad():
        ref_moe = reference_layer.mlp(moe_input)
    # The MoE block receives the post-attention-norm output, bf16 in production (bf16 residual stream).
    tt_moe_input_mlp = ttnn.from_torch(
        moe_input_bf16.reshape(1, 1, -1, config.hidden_size),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )
    tt_moe = decoder_layer.mlp(tt_moe_input_mlp, is_decode=is_decode)
    tm.assert_replicated(tt_moe, "MoE output")
    tt_moe_torch = ttnn.to_torch(tt_moe, mesh_composer=mesh_composer)[..., :num_tokens, : config.hidden_size]
    mlp_passing, mlp_pcc = compare_tensors(
        tt_moe_torch.reshape(num_tokens, -1),
        ref_moe.reshape(num_tokens, -1),
        mesh_device,
        REAL_WEIGHT_THRESHOLDS["mlp"],
    )
    results["mlp"] = f"PCC {mlp_pcc}"

    # 3. Full layer (decode: over the per-user context, written into the TT cache first).
    with torch.no_grad():
        ref_layer_out = reference_layer(ref_input, attention_mask=ref_mask, position_embeddings=ref_position_embeddings)
        if decode_context is not None:
            ref_layer_out = ref_layer_out[:, -1:, :]
    if decode_context is not None:
        decode_context.fill_kv_cache(mesh_device, decoder_layer, page_table_tt, apply_input_norm=True)
    # Layer input: the bf16 embeddings exactly as the model's ttnn.embedding(..., dtype=bf16) delivers them.
    tt_hidden = ttnn.from_torch(
        hidden_states.reshape(1, 1, num_tokens, -1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )
    tt_layer_out = decoder_layer(
        tt_hidden,
        position_embeddings=rope_mats,
        position_idx=tt_position_idx,
        page_table=page_table_tt,
        is_decode=is_decode,
    )
    assert tt_layer_out.dtype == ttnn.bfloat16, f"the residual stream must stay bf16, got {tt_layer_out.dtype}"
    tm.assert_replicated(tt_layer_out, "decoder layer output")
    tt_layer_torch = ttnn.to_torch(tt_layer_out, mesh_composer=mesh_composer)[..., :num_tokens, : config.hidden_size]
    decoder_passing, decoder_pcc = compare_tensors(
        tt_layer_torch.reshape(num_tokens, -1),
        ref_layer_out.reshape(num_tokens, -1),
        mesh_device,
        REAL_WEIGHT_THRESHOLDS["decoder"],
    )
    results["decoder"] = f"PCC {decoder_pcc}"

    case = f"layer {LAYER_IDX} real weights, batch {batch_size} x seq {seq_len}, paged={paged}"
    for name, value in results.items():
        logger.info(f"[{case}] {name}: {value}")

    problems = []
    if num_mismatch > max_mismatch:
        problems.append(
            f"router expert-set agreement {agreement:.4f} < {REAL_WEIGHT_THRESHOLDS['router_set_agreement']} "
            f"({num_mismatch}/{num_tokens} tokens differ, allowed {max_mismatch})"
        )
    if decisive_mismatch and tm.MoEOptions.from_env().router_fp32_logits:
        problems.append(
            f"{decisive_mismatch} decisive tokens (margin >= {tm.ROUTER_DECISIVE_MARGIN}) routed differently"
        )
    if not dense_agree_passing:
        problems.append(f"router dense PCC on the agreeing tokens {dense_pcc_agree}")
    if num_tokens >= tm.ROUTER_ALL_TOKEN_PCC_MIN_TOKENS and not dense_all_passing:
        problems.append(f"router dense PCC on all tokens {dense_pcc}")
    if not mlp_passing:
        problems.append(f"mlp {mlp_pcc}")
    if not decoder_passing:
        problems.append(f"decoder {decoder_pcc}")
    assert not problems, f"[{case}] " + "; ".join(problems)
