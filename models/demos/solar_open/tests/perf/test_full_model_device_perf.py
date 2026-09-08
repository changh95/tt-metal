# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profiling helper, not a correctness test (SKIPS unless ``SOLAR_OPEN_PERF_PROFILE=1``): the FULL 48-layer model (warm
ttnn cache) driven like
the demo -- real prompts, traced prefill@128, traced decode with on-device greedy sampling -- with a signpost in front
of every decode step so the tracy ops CSV yields the per-op device time of complete steps (48 layers + embedding +
final norm + lm_head + sampling), i.e. the split between device-busy time and dispatch gaps of the measured step.

    SOLAR_OPEN_PERF_OUT=/path/fm.json python -m tracy -r -p -v --op-support-count 40000 -m pytest \
        models/demos/solar_open/tests/perf/test_full_model_device_perf.py -k b1 -x -p no:cacheprovider
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams
from models.demos.solar_open.demo.text_demo import prepare_solar_open_generator_args
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.tt_transformers.demo.simple_text_demo import load_inputs
from models.tt_transformers.tt.common import preprocess_inputs_prefill
from models.tt_transformers.tt.generator import Generator

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(header, message=None):
        logger.info(f"SIGNPOST {header}")


PERF_OUT = os.getenv("SOLAR_OPEN_PERF_OUT", "")
PROFILE = os.getenv("SOLAR_OPEN_PERF_PROFILE", "") == "1"  # opt-in: the helper is meant to run under the tracy profiler
PROMPTS_FILE = "models/demos/solar_open/demo/sample_prompts/input_data_questions_ko_en_prefill_128.json"
DECODE_STEPS = {
    1: int(os.getenv("SOLAR_OPEN_PERF_B1_STEPS", "12")),
    32: int(os.getenv("SOLAR_OPEN_PERF_B32_STEPS", "30")),
}
FLUSH_EVERY = 5


def _flush(mesh_device):
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
    except Exception as exc:
        logger.warning(f"ReadDeviceProfiler failed: {exc}")


def _dump(key, record):
    if not PERF_OUT:
        return
    data = {}
    if os.path.isfile(PERF_OUT):
        data = json.loads(open(PERF_OUT).read())
    data[key] = record
    with open(PERF_OUT, "w") as f:
        json.dump(data, f, indent=2)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("batch_size", [1, 32], ids=["b1", "b32"])
@parametrize_mesh_with_fabric([(1, 8)])
def test_full_model_device_perf(mesh_device, device_params, batch_size, reset_seeds):
    if not PROFILE:
        pytest.skip(
            "profiling helper: set SOLAR_OPEN_PERF_PROFILE=1 (and run under `python -m tracy -r -p -v -m pytest ...`)"
        )
    if tuple(mesh_device.shape) != (1, 8):
        pytest.skip("sized for the 1x8 mesh")
    os.environ.setdefault("SOLAR_OPEN_REASONING_EFFORT", "low")
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    max_seq_len = 4096 if batch_size == 1 else 8192
    page_params = {"page_block_size": 64, "page_max_num_blocks_per_dp": batch_size * max_seq_len // 64}

    t0 = time.perf_counter()
    model_args, model, page_table, tt_kv_cache, tokenizer, processor, _ = prepare_solar_open_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=setup["mesh_config"],
        state_dict=None,
    )
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)
    record = {"batch": batch_size, "model_ready_s": time.perf_counter() - t0}
    logger.info(f"model ready in {record['model_ready_s']:.1f} s")

    prompts, _ = load_inputs(PROMPTS_FILE, batch_size, instruct=False)
    tokens_list, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        prompts, tokenizer, model_args, instruct=False, max_generated_tokens=64, max_prefill_len=max_seq_len
    )
    tokens = torch.stack(tokens_list).view(batch_size, -1)
    sampling = SamplingParams(
        temperature=[0.0] * 32, top_k=[1] * 32, top_p=[1.0] * 32, enable_log_probs=[False] * 32, num_logprobs=[0] * 32
    )

    # Prefill warm-up / compile (user 0 with the FULL page table: this call also prepares the decode trace)
    _flush(mesh_device)
    t0 = time.perf_counter()
    generator.prefill_forward_text(
        tokens[:1],
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        enable_trace=True,
        warmup_prefill=False,
    )
    record["prefill_compile_s"] = time.perf_counter() - t0
    _flush(mesh_device)

    # Timed per-user traced prefills (flushing the profiler between users keeps its buffer small)
    prefilled = torch.zeros(batch_size, dtype=torch.long)
    prefill_walls = []
    for u in range(batch_size):
        signpost(f"fm_b{batch_size}_prefill_user{u}")
        t0 = time.perf_counter()
        logits = generator.prefill_forward_text(
            tokens[u : u + 1],
            page_table=page_table[u : u + 1],
            kv_cache=tt_kv_cache,
            prompt_lens=[decoding_pos[u]],
            empty_slots=[u],
            enable_trace=True,
            warmup_prefill=False,
        )
        prefill_walls.append((time.perf_counter() - t0) * 1e3)
        prefilled[u] = torch.argmax(logits.reshape(-1)).item()
        if (u + 1) % FLUSH_EVERY == 0:
            _flush(mesh_device)
    signpost(f"fm_b{batch_size}_prefill_done")
    record["prefill_ms_per_user"] = prefill_walls
    record["prompt_lens"] = [int(x) for x in decoding_pos]
    logger.info(f"prefill per user (ms): {[round(w, 1) for w in prefill_walls]}")
    _flush(mesh_device)

    # Decode: step 0 = compile + trace capture, steps >= 1 = trace replay (as in the demo)
    current_pos = torch.tensor([int(p) for p in decoding_pos])
    out_tok = prefilled
    all_tokens = [[int(prefilled[b])] for b in range(batch_size)]
    steps = DECODE_STEPS[batch_size]
    walls = []
    for it in range(steps):
        signpost(f"fm_b{batch_size}_decode_step{it:02d}")
        t0 = time.perf_counter()
        out_tok, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=True,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=sampling,
        )
        walls.append((time.perf_counter() - t0) * 1e3)
        current_pos += 1
        for b in range(batch_size):
            all_tokens[b].append(int(out_tok[b]))
        if it == 0 or (it + 1) % FLUSH_EVERY == 0:
            _flush(mesh_device)
    signpost(f"fm_b{batch_size}_decode_done")
    _flush(mesh_device)
    record["decode_steps"] = steps
    record["decode_ms_per_step"] = walls
    record["decode_ms_mean_steps_ge1"] = sum(walls[1:]) / max(1, len(walls) - 1)
    record["sample_output_user0"] = tokenizer.decode(all_tokens[0], skip_special_tokens=False)
    if batch_size > 1:
        record["sample_output_user22"] = tokenizer.decode(all_tokens[22], skip_special_tokens=False)
    logger.info(f"decode ms/step: {[round(w, 1) for w in walls]}")
    logger.info(f"user 0 output: {record['sample_output_user0']}")
    _dump(f"full_model_b{batch_size}", record)
