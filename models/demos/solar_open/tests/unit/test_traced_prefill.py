# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced prefill buckets (phase 3a(2), design_traced_prefill.md): the trace table, the MoE path selection and the
router helper persistence behind it, and a device test that replays every traced bucket and compares it with eager.

Host part (no device; ``SOLAR_OPEN_NUM_DEVICES=8`` keeps collection from enumerating the PCIe devices):

    SOLAR_OPEN_NUM_DEVICES=8 pytest models/demos/solar_open/tests/unit/test_traced_prefill.py -k Host

Device part (P150x8, real weights through the demo's model path; one bucket per case, a case whose bucket is not in
``ModelArgs.trace_prefill_supported_seq_lens`` skips -- today only ``prefill_128`` runs):

    pytest models/demos/solar_open/tests/unit/test_traced_prefill.py -k "1x8 and prefill_128" -x -p no:cacheprovider

Per bucket the device test prefills two different prompts of that bucket twice eagerly (the second run measures the
device's run-to-run noise: ttnn.all_reduce is not bit-reproducible), then twice with ``enable_trace=True`` (the first
call captures the trace on user 0 and replays it on user 1; the second call replays only), and asserts: the traced
logits match eager within the eager repeat's own noise (PCC floor, decisive top-1 agreement), the user's paged K/V
blocks written by the traced forward match the eager ones, the replays are stable, the trace is not stale (user 1's
prompt differs from the captured user 0's), every split of the traced forward ran a trace-safe MoE path
(``experts_prefill.LAST_PREFILL_MOE_PATH``), the router helpers of the bucket were persistent BEFORE the capture and no
persistent helper (router rows, expert identities) was allocated by the capture, and an eager prefill still matches
after the traces exist. TTFT eager vs replay and the trace-region usage are logged.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.solar_open.tt import model_config as mc
from models.demos.solar_open.tt.expert_configs import SolarOpenProgramConfig
from models.demos.solar_open.tt.experts import prefill as experts_prefill
from models.demos.solar_open.tt.experts.config import ProgramConfig
from models.demos.solar_open.tt.model_config import ModelArgs
from models.demos.solar_open.tt.topk import TopKRouter
from models.tt_transformers.tt.common import get_padded_prefill_len

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "Solar-Open-100B"
PROMPTS_128 = (
    Path(__file__).resolve().parents[2] / "demo" / "sample_prompts" / "input_data_questions_ko_en_prefill_128.json"
)
E = 128  # Solar-Open expert count

# Every bucket the Generator pads a prompt to (get_padded_prefill_len) up to the warmup cap; the device test skips the
# ones the trace table does not list, so a future table entry is covered without a test change.
TRACE_BUCKETS = [128, 1024, 2048, 4096]
BATCH = 2  # user 0 captures, user 1 replays on a different prompt of the same bucket
MAX_SEQ_LEN = 8 * 1024
BLOCK_SIZE = 64
# Phase-2 measurements of identical repeats: logit PCC min 0.99997 (tests/test_multi_user_consistency.py). A traced
# replay must stay within the eager repeat's noise; these floors keep >= 0.5 of margin to a real defect (PCC ~0).
LOGITS_PCC_FLOOR = 0.999
KV_PCC_FLOOR = 0.9999
DECISIVE_MARGIN = 0.5  # reference top-1 margin (logits) above which a top-1 flip is not a bf16 near tie
FILLER = (
    "The Blackhole mesh prefills the prompt in one forward pass; every layer routes its tokens to eight of the 128 "
    "experts and the shared expert adds its partial before the tensor-parallel all-reduce. "
)
SOLAR_ENV_VARS = (
    "SOLAR_OPEN_BATCHED_PREFILL",
    "SOLAR_OPEN_BATCHED_PREFILL_TOKENS",
    "SOLAR_OPEN_BATCHED_PREFILL_MAX_SEQ_LEN",
    "SOLAR_OPEN_PREFILL_EXPERT_MM",
    "SOLAR_OPEN_FUSE_SHARED_EXPERT",
)


# --------------------------------------------------------------------------------------------------------------
# Host: MoE path selection (experts/prefill.py)
# --------------------------------------------------------------------------------------------------------------


class TestHostPrefillPathSelection:
    def test_split_lens_mirror_the_chunking(self, expect_error):
        pc = SolarOpenProgramConfig()
        assert pc.sequence_chunk_size == 4096 and pc.base_down_split_size == 1024 and pc.dense_bmm_max_tokens == 256
        assert experts_prefill.prefill_split_lens(128, pc) == (128,)
        assert experts_prefill.prefill_split_lens(256, pc) == (256,)
        assert experts_prefill.prefill_split_lens(1024, pc) == (1024,)
        assert experts_prefill.prefill_split_lens(2048, pc) == (1024,)
        assert experts_prefill.prefill_split_lens(4096, pc) == (1024,)
        assert experts_prefill.prefill_split_lens(8192, pc) == (1024,)
        assert experts_prefill.prefill_split_lens(1152, pc) == (1024, 128)  # remainder split (non-Generator lengths)
        assert experts_prefill.prefill_split_lens(64 * 1024, pc) == (1024,)  # chunks stay <= 4096: no halved split
        with expect_error(ValueError, "seq_len must be positive"):
            experts_prefill.prefill_split_lens(0, pc)

    def test_paths(self):
        pc = SolarOpenProgramConfig()
        path = experts_prefill.moe_prefill_path
        assert path(128, pc, E) == experts_prefill.MOE_PATH_DENSE_BMM
        assert path(256, pc, E) == experts_prefill.MOE_PATH_DENSE_BMM
        assert path(288, pc, E) == experts_prefill.MOE_PATH_HOST_PLANNED
        assert path(1024, pc, E) == experts_prefill.MOE_PATH_HOST_PLANNED
        assert path(1024, pc, E + 1, always_on=1) == experts_prefill.MOE_PATH_HOST_PLANNED  # fused shared expert
        assert path(1024, pc, 32) == experts_prefill.MOE_PATH_STATIC_LOOP  # < _SORTED_MOE_MIN_EXPERTS
        assert path(1024, pc, E, dense_moe=False) == experts_prefill.MOE_PATH_SPARSE_EP
        knob = SolarOpenProgramConfig(trace_safe_split_lens=(1024,))
        assert path(1024, knob, E) == experts_prefill.MOE_PATH_STATIC_LOOP
        assert path(128, knob, E) == experts_prefill.MOE_PATH_DENSE_BMM
        assert experts_prefill.MOE_PATH_HOST_PLANNED not in experts_prefill.TRACE_SAFE_MOE_PATHS
        assert {
            experts_prefill.MOE_PATH_DENSE_BMM,
            experts_prefill.MOE_PATH_STATIC_LOOP,
            experts_prefill.MOE_PATH_SPARSE_EP,
        } <= experts_prefill.TRACE_SAFE_MOE_PATHS

    def test_trace_safe_lengths(self):
        pc = SolarOpenProgramConfig()
        safe = experts_prefill.is_trace_safe_prefill_len
        assert safe(128, pc, E) and safe(256, pc, E)
        assert not safe(1024, pc, E) and not safe(2048, pc, E) and not safe(4096, pc, E)
        assert not safe(1152, pc, E)  # its 1024-token split is host-planned even though the 128 remainder is not
        knob = SolarOpenProgramConfig(trace_safe_split_lens=(1024,))
        assert safe(1024, knob, E) and safe(2048, knob, E) and safe(4096, knob, E)
        assert safe(1024, pc, 32)  # few experts: the static loop everywhere
        assert safe(1024, pc, E, dense_moe=False)  # multi-row EP path

    def test_sorted_split_lens_to_prebuild(self):
        pc = SolarOpenProgramConfig()
        assert experts_prefill._sorted_split_lens(pc, E) == [1024]
        assert experts_prefill._sorted_split_lens(pc, E + 1, always_on=1) == [1024]
        assert experts_prefill._sorted_split_lens(SolarOpenProgramConfig(trace_safe_split_lens=(1024,)), E) == []
        assert experts_prefill._sorted_split_lens(pc, 32) == []
        # a chunk shorter than the split size caps the split (and the identity) at the chunk length
        assert experts_prefill._sorted_split_lens(SolarOpenProgramConfig(sequence_chunk_size=512), E) == [512]

    def test_program_config_knob(self, expect_error):
        assert SolarOpenProgramConfig().trace_safe_split_lens == ()
        assert ProgramConfig().trace_safe_split_lens == ()
        assert SolarOpenProgramConfig(trace_safe_split_lens=[2048, 1024, 1024]).trace_safe_split_lens == (1024, 2048)
        with expect_error(ValueError, "trace_safe_split_lens"):
            SolarOpenProgramConfig(trace_safe_split_lens=(100,))
        with expect_error(ValueError, "trace_safe_split_lens"):
            SolarOpenProgramConfig(trace_safe_split_lens=(0,))
        with expect_error(ValueError, "trace_safe_split_lens"):
            SolarOpenProgramConfig(trace_safe_split_lens=1024)


# --------------------------------------------------------------------------------------------------------------
# Host: the trace table and the router token counts (tt/model_config.py)
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture
def host_p150x8(monkeypatch, tmp_path):
    """ModelArgs against the in-tree config with a mock 1x8 mesh named P150x8 and a clean SOLAR_OPEN_* environment."""
    for var in SOLAR_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_MODEL", str(CONFIG_DIR))
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path / "tt_cache"))
    monkeypatch.setattr(mc, "determine_device_name", lambda mesh_device: "P150x8")
    mesh = MagicMock(name="mesh_device_1x8")
    mesh.shape = (1, 8)
    return mesh


class TestHostTraceTable:
    def test_shipped_table(self, host_p150x8):
        assert mc.TRACE_PREFILL_SEQ_LENS == {mc.MODEL_NAME: {"P150x8": [128]}}
        args = ModelArgs(mesh_device=host_p150x8, dummy_weights=True)
        assert args.trace_prefill_supported_seq_lens == [128]
        assert args.can_enable_trace(128) and not args.can_enable_trace(1024) and not args.can_enable_trace(128, 32)
        assert args.get_warmup_prefill_supported_seq_lens() == [128, 1024, 2048]
        assert args.batched_prefill_token_counts == []
        assert args.router_persistent_token_counts == [1, 32, 128]
        assert mc.TRACE_PREFILL_SEQ_LENS[mc.MODEL_NAME]["P150x8"] == [128]  # the table itself is never mutated

    def test_router_counts_follow_the_batch_and_the_packed_prefill(self, host_p150x8, monkeypatch):
        args = ModelArgs(mesh_device=host_p150x8, dummy_weights=True, max_batch_size=32)
        assert args.router_persistent_token_counts == [32, 128]
        # Packed multi-user prefill on: the row counts of its passes (B >= 2 users x 128 tokens within the budget)
        monkeypatch.setenv("SOLAR_OPEN_BATCHED_PREFILL", "1")
        args = ModelArgs(mesh_device=host_p150x8, dummy_weights=True, max_batch_size=32)
        assert args.batched_prefill_token_counts == [256, 512, 1024]
        assert args.router_persistent_token_counts == [32, 128, 256, 512, 1024]
        monkeypatch.setenv("SOLAR_OPEN_BATCHED_PREFILL_TOKENS", "4096")
        args = ModelArgs(mesh_device=host_p150x8, dummy_weights=True, max_batch_size=8)
        assert args.batched_prefill_token_counts == [256, 512, 1024, 2048, 4096]
        assert args.router_persistent_token_counts == [8, 32, 128, 256, 512, 1024, 2048, 4096]

    def test_unsafe_length_is_refused(self, host_p150x8, monkeypatch, expect_error):
        monkeypatch.setitem(mc.TRACE_PREFILL_SEQ_LENS[mc.MODEL_NAME], "P150x8", [128, 1024])
        with expect_error(ValueError, "not trace-safe"):
            ModelArgs(mesh_device=host_p150x8, dummy_weights=True)

    def test_length_admitted_with_a_trace_safe_split(self, host_p150x8, monkeypatch):
        monkeypatch.setitem(mc.TRACE_PREFILL_SEQ_LENS[mc.MODEL_NAME], "P150x8", [128, 1024, 2048])
        monkeypatch.setattr(mc, "SolarOpenProgramConfig", lambda: SolarOpenProgramConfig(trace_safe_split_lens=(1024,)))
        args = ModelArgs(mesh_device=host_p150x8, dummy_weights=True)
        assert args.trace_prefill_supported_seq_lens == [128, 1024, 2048]
        assert args.can_enable_trace(1024) and args.can_enable_trace(2048)
        assert args.get_warmup_prefill_supported_seq_lens() == [128, 1024, 2048]
        assert args.router_persistent_token_counts == [1, 32, 128, 1024, 2048]

    def test_cap_drops_4096_before_validation(self, host_p150x8, monkeypatch):
        monkeypatch.setitem(mc.TRACE_PREFILL_SEQ_LENS[mc.MODEL_NAME], "P150x8", [128, 4096])
        args = ModelArgs(mesh_device=host_p150x8, dummy_weights=True)  # 4096 > capped_warmup_seq_len: dropped
        assert args.trace_prefill_supported_seq_lens == [128] and not args.can_enable_trace(4096)

    def test_other_devices_have_no_traced_prefill(self, host_p150x8, monkeypatch):
        monkeypatch.setattr(mc, "determine_device_name", lambda mesh_device: "T3K")
        args = ModelArgs(mesh_device=host_p150x8, dummy_weights=True)
        assert args.trace_prefill_supported_seq_lens == [] and not args.can_enable_trace(128)
        assert args.router_persistent_token_counts == [1, 32]


# --------------------------------------------------------------------------------------------------------------
# Host: router helper persistence (tt/topk.py), ttnn mocked
# --------------------------------------------------------------------------------------------------------------


def _router_hf_config():
    return SimpleNamespace(num_experts_per_tok=8, num_local_experts=E, hidden_size=4096, routed_scaling_factor=1.0)


def _zeros_shapes(mock_ttnn):
    return sorted(call.args[0] for call in mock_ttnn.zeros.call_args_list)


class TestHostRouterPersistence:
    def test_default_counts(self):
        with patch("models.demos.solar_open.tt.topk.ttnn") as mock_ttnn:
            router = TopKRouter(MagicMock(name="mesh"), _router_hf_config(), {}, tokens_per_device=8)
            assert router.persistent_token_counts == frozenset({8, 32, 128})
            assert _zeros_shapes(mock_ttnn) == [[8, E], [32, E], [128, E]]
            assert set(router._row_tensors) == {8, 32, 128}
            _rows, transient = router._get_row_tensors(64)  # <= _KEEP_ROW_TENSORS_UP_TO: built lazily and kept
            assert not transient and 64 in router._row_tensors
            _rows, transient = router._get_row_tensors(1024)  # a long eager prefill: built and freed per call
            assert transient and 1024 not in router._row_tensors

    def test_explicit_counts(self):
        with patch("models.demos.solar_open.tt.topk.ttnn") as mock_ttnn:
            router = TopKRouter(
                MagicMock(name="mesh"),
                _router_hf_config(),
                {},
                tokens_per_device=32,
                persistent_token_counts=(128, 1024),
            )
            assert router.persistent_token_counts == frozenset({32, 128, 1024})  # the decode batch is always kept
            assert _zeros_shapes(mock_ttnn) == [[32, E], [128, E], [1024, E]]
            bias_repeats = sorted(call.args[1] for call in mock_ttnn.repeat.call_args_list)
            assert bias_repeats == [[32, 1], [128, 1], [1024, 1]]  # the fused op's [T, E] bias copies
            _rows, transient = router._get_row_tensors(1024)
            assert not transient  # the traced length's helpers are persistent: nothing is built under a capture
            assert mock_ttnn.zeros.call_count == 3 and mock_ttnn.repeat.call_count == 3
            _rows, transient = router._get_row_tensors(2048)
            assert transient and 2048 not in router._row_tensors

    def test_invalid_counts_are_rejected(self, expect_error):
        with patch("models.demos.solar_open.tt.topk.ttnn"):
            for bad in ((0,), (-1,), ("128",), (True,), (128.0,)):
                with expect_error(ValueError, "persistent_token_counts"):
                    TopKRouter(MagicMock(name="mesh"), _router_hf_config(), {}, persistent_token_counts=bad)


# --------------------------------------------------------------------------------------------------------------
# Device: traced vs eager per bucket (P150x8, real weights)
# --------------------------------------------------------------------------------------------------------------


def _clear_kv_caches(models):
    for m in models:
        for layer in m.layers:
            k_cache, v_cache = layer.self_attn.layer_past
            ttnn.mul(k_cache, 0, output_tensor=k_cache)
            ttnn.mul(v_cache, 0, output_tensor=v_cache)


def _prompt_ids_for_bucket(model_args, bucket, variant):
    """Chat-templated token ids of a prompt the Generator pads to ``bucket`` (variant 0 / 1 give different prompts):
    the demo's 128-token questions for the 128 bucket, a filler text sized by a word-count search otherwise."""
    if bucket == 128:
        with open(PROMPTS_128) as f:
            prompts = json.load(f)
        ids = model_args.encode_prompt(prompts[variant]["prompt"])
    else:
        lo, hi = bucket // 2 + 1, bucket - 8  # (previous bucket, bucket]: padded to ``bucket``
        words = (FILLER * (bucket // 8)).split()
        n = max(1, int(hi * 0.6))
        ids = model_args.encode_prompt(" ".join(words[:n]) + f" Variant {variant}.")
        for _ in range(12):
            if lo <= len(ids) <= hi:
                break
            n = max(1, int(n * (hi - 16) / max(1, len(ids))))
            ids = model_args.encode_prompt(" ".join(words[:n]) + f" Variant {variant}.")
    assert get_padded_prefill_len(len(ids)) == bucket, f"{len(ids)} tokens do not pad to {bucket}"
    return ids


def _prefill(generator, models, mesh_device, tt_kv_cache, page_table, prompt_ids, enable_trace):
    """Clear the KV cache and prefill every slot with its ids. Returns (logits [B, V] fp32, wall seconds)."""
    batch = len(prompt_ids)
    lens = [len(ids) for ids in prompt_ids]
    tokens = torch.zeros(batch, max(lens), dtype=torch.long)
    for b, ids in enumerate(prompt_ids):
        tokens[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    _clear_kv_caches(models)
    generator.prev_page_table = None
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    logits = generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=lens,
        enable_trace=enable_trace,
        warmup_prefill=False,  # the Solar path (text_demo / the sweep harness): no generic batch-1 warm-up sweep
    )
    ttnn.synchronize_device(mesh_device)
    wall = time.perf_counter() - t0
    vocab = models[0].vocab_size
    return logits.reshape(batch, -1).float()[:, :vocab], wall


def _kv_snapshot(tt_kv_cache, page_table, lens, layers, devices):
    """{(layer, user, device, "k"|"v"): the user's written K/V blocks [blocks, heads, block, head_dim] fp32}."""
    out = {}
    for layer in layers:
        k_cache, v_cache = tt_kv_cache[layer]
        for name, cache in (("k", k_cache), ("v", v_cache)):
            per_device = ttnn.get_device_tensors(cache)
            for d in devices:
                full = ttnn.to_torch(per_device[d]).float()
                for b, n in enumerate(lens):
                    blocks = page_table[b, : -(-n // BLOCK_SIZE)].long()
                    out[(layer, b, d, name)] = full[blocks].clone()
    return out


def _pcc(a, b):
    a = a.flatten().float() - a.float().mean()
    b = b.flatten().float() - b.float().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def _router_state(models):
    """Per layer: the persistent router row counts and the experts' identity tables (must not change under a capture)."""
    return [
        (tuple(sorted(layer.mlp.router._row_tensors)), tuple(sorted((layer.mlp.experts.weights.eye_tables or {}))))
        for layer in models[0].layers
    ]


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("bucket", TRACE_BUCKETS, ids=[f"prefill_{b}" for b in TRACE_BUCKETS])
@parametrize_mesh_with_fabric([(1, 8)])
def test_traced_prefill_matches_eager(mesh_device, device_params, bucket, state_dict, monkeypatch):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] != 1 or mesh_shape[1] < 8:
        pytest.skip(f"traced prefill is validated on 1x8 meshes (TP=8), got {mesh_shape}")
    for var in SOLAR_ENV_VARS:
        if var.startswith("SOLAR_OPEN_BATCHED_PREFILL"):
            monkeypatch.delenv(var, raising=False)  # the per-user (non-batched) prefill path of the trace table
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    if bucket not in setup["model_args"].trace_prefill_supported_seq_lens:
        pytest.skip(
            f"{bucket} is not a traced prefill bucket on this device "
            f"(trace table {setup['model_args'].trace_prefill_supported_seq_lens})"
        )
    from models.demos.solar_open.demo.text_demo import log_device_memory, prepare_solar_open_generator_args
    from models.tt_transformers.tt.generator import Generator

    page_params = {"page_block_size": BLOCK_SIZE, "page_max_num_blocks_per_dp": BATCH * (MAX_SEQ_LEN // BLOCK_SIZE)}
    model_args, models, page_table, tt_kv_cache, tokenizer, _processor, _paged_cfg = prepare_solar_open_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=BATCH,
        optimizations=None,
        max_seq_len=MAX_SEQ_LEN,
        page_params=page_params,
        paged_attention=True,
        mesh_config=setup["mesh_config"],
        state_dict=state_dict,
        users_row_sharded=False,
    )
    generator = Generator(models, model_args, mesh_device, processor=None, tokenizer=tokenizer)
    args0, model0, kv0 = model_args[0], models[0], tt_kv_cache[0]
    assert bucket in args0.trace_prefill_supported_seq_lens and args0.can_enable_trace(bucket)
    assert args0.disable_batched_prefill, "this test covers the per-user traced prefill path"
    prompt_ids = [_prompt_ids_for_bucket(args0, bucket, variant) for variant in range(BATCH)]
    lens = [len(ids) for ids in prompt_ids]
    assert prompt_ids[0] != prompt_ids[1], "the two users need different prompts (stale-trace check)"
    logger.info(f"bucket {bucket}: prompts of {lens} tokens (padded to {bucket}), {len(model0.layers)} layers")
    n_layers = len(model0.layers)
    kv_layers = sorted({0, n_layers // 2, n_layers - 1})
    kv_devices = sorted({0, mesh_shape[1] - 1})
    run = (generator, models, mesh_device, tt_kv_cache, page_table, prompt_ids)

    # 1. Eager reference (also the compile pass every flow runs before a capture) and its run-to-run noise floor.
    logits_a1, wall_a1 = _prefill(*run, enable_trace=False)
    kv_a1 = _kv_snapshot(kv0, page_table, lens, kv_layers, kv_devices)
    logits_a2, wall_a2 = _prefill(*run, enable_trace=False)
    kv_a2 = _kv_snapshot(kv0, page_table, lens, kv_layers, kv_devices)
    noise_pcc = min(_pcc(logits_a2[b], logits_a1[b]) for b in range(BATCH))
    noise_kv_pcc = min(_pcc(kv_a2[key], kv_a1[key]) for key in kv_a1)
    noise_top1 = [bool(logits_a2[b].argmax() == logits_a1[b].argmax()) for b in range(BATCH)]
    logger.info(
        f"eager repeat: {1000 * wall_a1:.0f} / {1000 * wall_a2:.0f} ms, logit PCC min {noise_pcc:.6f}, "
        f"KV PCC min {noise_kv_pcc:.6f}, top-1 stable {noise_top1}"
    )
    assert noise_pcc >= LOGITS_PCC_FLOOR, f"the eager repeat itself is below the floor ({noise_pcc:.5f})"

    # 2. The bucket's router helpers must be persistent BEFORE the capture (a transient helper would be built inside
    #    the trace), and nothing persistent may be allocated by the capture itself.
    for layer in model0.layers:
        router = layer.mlp.router
        assert bucket in router.persistent_token_counts, (
            f"layer {layer.layer_idx}: the traced length {bucket} is not in the router's persistent token counts "
            f"{sorted(router.persistent_token_counts)}; Model must pass ModelArgs.router_persistent_token_counts"
        )
        assert bucket in router._row_tensors, f"layer {layer.layer_idx}: no prebuilt router helpers for {bucket}"
        assert layer.mlp.experts.weights.down_proj_padded is not None, "the eager pass must have built the helpers"
    state_before = _router_state(models)

    # 3. Traced: user 0 captures (compile pass + capture + replay), user 1 replays on its own prompt and page table.
    experts_prefill.LAST_PREFILL_MOE_PATH.clear()
    logits_t1, wall_t1 = _prefill(*run, enable_trace=True)
    kv_t1 = _kv_snapshot(kv0, page_table, lens, kv_layers, kv_devices)
    moe_path = dict(experts_prefill.LAST_PREFILL_MOE_PATH)
    assert moe_path.get("path") in experts_prefill.TRACE_SAFE_MOE_PATHS and not moe_path.get(
        "sorted"
    ), f"a split of the traced forward ran the host-planned MoE path: {moe_path}"
    assert _router_state(models) == state_before, "the capture allocated a persistent router helper / identity"
    trace_keys = [k for k, v in generator.trace_id_prefill.items() if v is not None]
    assert trace_keys, "no prefill trace was captured"
    logger.info(f"prefill traces {trace_keys}; MoE path of the last split {moe_path}")
    log_device_memory(mesh_device, f"after the {bucket}-token prefill trace")
    logits_t2, wall_t2 = _prefill(*run, enable_trace=True)  # replays only
    kv_t2 = _kv_snapshot(kv0, page_table, lens, kv_layers, kv_devices)
    logits_a3, wall_a3 = _prefill(*run, enable_trace=False)  # eager still exact with the traces alive

    # 4. Compare against eager within the eager repeat's own noise.
    problems = []
    logit_floor = min(LOGITS_PCC_FLOOR, noise_pcc - 1e-4)
    for b in range(BATCH):
        ref = logits_a1[b]
        top2 = ref.topk(2).values
        margin = float(top2[0] - top2[1])
        for name, other in (("traced", logits_t1[b]), ("replay", logits_t2[b]), ("eager after traces", logits_a3[b])):
            pcc = _pcc(other, ref)
            same_top1 = bool(other.argmax() == ref.argmax())
            logger.info(
                f"user {b} ({lens[b]} tokens) {name}: logit PCC {pcc:.6f}, top-1 {'same' if same_top1 else 'DIFFERS'} "
                f"(eager margin {margin:.3f}), max |diff| {(other - ref).abs().max():.4f}"
            )
            if pcc < logit_floor:
                problems.append(f"user {b} {name}: logit PCC {pcc:.5f} < {logit_floor:.5f}")
            if not same_top1 and margin >= DECISIVE_MARGIN and noise_top1[b]:
                problems.append(f"user {b} {name}: top-1 differs on a decisive step (eager margin {margin:.3f})")
        pcc_replays = _pcc(logits_t2[b], logits_t1[b])
        if pcc_replays < logit_floor:
            problems.append(f"user {b}: two replays differ (PCC {pcc_replays:.5f})")
    kv_floor = min(KV_PCC_FLOOR, noise_kv_pcc - 1e-5)
    worst_kv, identical = 1.0, 0
    for key in kv_a1:
        for name, snap in (("traced", kv_t1), ("replay", kv_t2)):
            pcc = _pcc(snap[key], kv_a1[key])
            worst_kv = min(worst_kv, pcc)
            identical += int(torch.equal(snap[key], kv_a1[key]))
            if pcc < kv_floor:
                layer, b, d, name_kv = key
                problems.append(
                    f"{name} {name_kv} cache layer {layer} user {b} device {d}: PCC {pcc:.6f} < {kv_floor:.6f}"
                )
    logger.info(
        f"KV blocks (layers {kv_layers}, devices {kv_devices}): traced/replay vs eager PCC min {worst_kv:.6f}, "
        f"{identical} of {2 * len(kv_a1)} blocks bit-identical"
    )
    # 5. The trace is not stale: the two users' logits differ from each other like their eager runs do.
    cross_eager = _pcc(logits_a1[0], logits_a1[1])
    cross_traced = _pcc(logits_t1[0], logits_t1[1])
    logger.info(f"user 0 vs user 1: eager PCC {cross_eager:.4f}, traced {cross_traced:.4f}")
    if abs(cross_traced - cross_eager) > 0.05 or cross_traced > 0.9999:
        problems.append(
            f"the replay for user 1 looks stale (user0-vs-user1 PCC eager {cross_eager:.4f}, traced {cross_traced:.4f})"
        )
    logger.info(
        f"TTFT bucket {bucket} (both users): eager {1000 * wall_a2:.0f} ms, capture + replay {1000 * wall_t1:.0f} ms, "
        f"replays {1000 * wall_t2:.0f} ms, eager after traces {1000 * wall_a3:.0f} ms"
    )
    assert not problems, "traced prefill differs from eager:\n" + "\n".join(problems)
