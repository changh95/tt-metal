# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device perf test of ONE real-weight decoder layer (phase 2): per-layer step times, optionally per-op profiles.

Not a correctness test. Without the profiler it reports the host wall time of the traced layer replay (decode: the
number that, times 48, reproduces the demo's decode step to within ~2 %) and of the eager prefill splits; run under
the tracy device profiler so every device op also lands in ops_perf_results_*.csv with signposts between the phases.
``SOLAR_OPEN_PERF_OUT`` names a JSON file that receives the host-side wall timings. Needs the real checkpoint shards
that hold layer 0 (see tests/test_layer0_real_weights.py) and the tokenizer.

    pytest models/demos/solar_open/tests/perf/test_layer0_device_perf.py -k 1x8 -x -p no:cacheprovider
    SOLAR_OPEN_PERF_OUT=/path/walls.json python -m tracy -r -p -v --op-support-count 20000 -m pytest \
        models/demos/solar_open/tests/perf/test_layer0_device_perf.py -k "1x8" -x -p no:cacheprovider

Recorded (P150x8, 2026-09-07, layer 0, traced replay per layer, blocking mean): phase 1 decode b1 1.152 ms / b32
(union 71) 1.911 ms, prefill_128 eager ~4.5 ms; phase 2 final tree (perf levers + perf-p1/p2 + perf-p0 merged) decode b1
0.376-0.379 ms (indexed path) / b32 1.30-1.33 ms, prefill_128 eager 3.94-3.97 ms, prefill_1024 eager 8.8-9.6 ms (eager
walls are host-load sensitive: the first run of a session measured 4.7 / 10.7). Per-lever history in the README's
"Recorded baselines" (perf rows) and scratchpad/phase2/perf_log.md.

Per case (decode_b1 / decode_b32 / prefill_128 / prefill_1024 / prefill_8192 -- the 8K case, 2 x 4096-token MoE chunks of
4 sorted splits each, is the eager long-prefill sample for the phase-3 profile and takes ~2 min to compile) it
  1. builds layer 0 with the REAL weights (same path as tests/test_layer0_real_weights.py) and real token embeddings,
  2. runs one LABELED eager step (a signpost between every sub-block: input norm, attention, residual, post norm,
     router, experts + shared expert + all_reduce, residual),
  3. runs N plain eager steps between "eager_<case>_start/stop" signposts,
  4. (decode) captures the layer in a trace and replays it M times between "trace_<case>_start/stop" signposts,
     recording the wall time per replay (non-blocking back-to-back and blocking).
"""

import functools
import json
import os
import time

import pytest
from loguru import logger

import ttnn
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.solar_open.tests.test_layer0_real_weights import (
    LAYER_IDX,
    _snapshot_dir,
    _token_ids,
    build_reference_layer,
    load_layer_state_dict,
)
from models.demos.solar_open.tests.unit import test_modules as tm

try:
    from tracy import signpost
except ModuleNotFoundError:  # plain pytest run without the profiler

    def signpost(header, message=None):
        logger.info(f"SIGNPOST {header}")


PERF_OUT = os.getenv("SOLAR_OPEN_PERF_OUT", "")
EAGER_STEPS = int(os.getenv("SOLAR_OPEN_PERF_EAGER_STEPS", "5"))
TRACE_REPLAYS = int(os.getenv("SOLAR_OPEN_PERF_TRACE_REPLAYS", "20"))


def _dump(case, record):
    if not PERF_OUT:
        return
    data = {}
    if os.path.isfile(PERF_OUT):
        data = json.loads(open(PERF_OUT).read())
    data[case] = record
    with open(PERF_OUT, "w") as f:
        json.dump(data, f, indent=2)


@pytest.fixture(scope="module")
def layer0_weights():
    from transformers import AutoConfig

    snapshot = _snapshot_dir()
    config = AutoConfig.from_pretrained(str(snapshot), trust_remote_code=True)
    return load_layer_state_dict(snapshot, LAYER_IDX, config.num_local_experts)


def _sync(mesh_device):
    ttnn.synchronize_device(mesh_device)


def _labeled_step(layer, tt_hidden, rope_mats, tt_position_idx, page_table_tt, is_decode, tag):
    """DecoderLayer.__call__ unrolled with a signpost in front of every sub-block."""
    residual = tt_hidden
    signpost(f"{tag}_input_norm")
    h = layer.input_layernorm(tt_hidden)
    signpost(f"{tag}_attention")
    a = layer.self_attn(
        h,
        rope_mats=rope_mats,
        position_idx=tt_position_idx,
        page_table=page_table_tt,
        kv_cache=None,
        is_decode=is_decode,
    )
    h.deallocate(True)
    signpost(f"{tag}_residual1")
    x = layer._residual_add(residual, a)
    residual = x
    signpost(f"{tag}_post_norm")
    h = layer.post_attention_layernorm(x)
    signpost(f"{tag}_router")
    # MLP.route decides the routing form exactly as the model does: the dense [T, E] tensor, or -- single user with
    # MoEOptions.indexed_decode -- the token's top-k as an IndexedRouting (indexed/gather expert path)
    dense, indexed = layer.mlp.route(h, is_decode=is_decode)
    union = None
    if is_decode and dense is not None:
        d0 = ttnn.to_torch(ttnn.get_device_tensors(dense)[0]).float()
        union = int((d0.sum(0) > 0).sum())
    elif is_decode:
        union = indexed.top_k
    signpost(f"{tag}_experts")
    shared = functools.partial(layer.mlp.shared_expert, is_decode=is_decode) if layer.mlp.shared_expert else None
    m = layer.mlp.experts(
        h, topk_expert_weights=dense, is_decode=is_decode, shared_expert=shared, indexed_routing=indexed
    )
    if indexed is not None:
        indexed.deallocate()
    h.deallocate(True)
    signpost(f"{tag}_residual2")
    x = layer._residual_add(residual, m)
    signpost(f"{tag}_end")
    _sync(layer.mesh_device)
    x.deallocate(True)
    return union


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [(1, 1), (32, 1), (1, 128), (1, 1024), (1, 8192)],
    ids=["decode_b1", "decode_b32", "prefill_128", "prefill_1024", "prefill_8192"],
)
@parametrize_mesh_with_fabric([(1, 8)])
def test_layer0_device_perf(mesh_device, device_params, batch_size, seq_len, layer0_weights, reset_seeds):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape != (1, 8):
        pytest.skip("sized for the 1x8 TP=8 mesh")
    is_decode = seq_len == 1
    case = f"decode_b{batch_size}" if is_decode else f"prefill_{seq_len}"
    num_tokens = batch_size * seq_len

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    state_dict, embed = layer0_weights
    reference_layer = build_reference_layer(config, state_dict)
    paged_attention_config, page_table_tt = tm.make_paged_attention(mesh_device, batch_size, seq_len)
    layer = tm.setup_decoder_layer(
        setup, reference_layer, batch_size, seq_len, layer_idx=LAYER_IDX, paged_attention_config=paged_attention_config
    )
    del reference_layer

    context_len = tm.DECODE_CONTEXT_LEN if is_decode else 0
    token_ids = _token_ids(
        _snapshot_dir(), batch_size * (seq_len + context_len), config.vocab_size, setup["model_args"]
    )
    token_ids = token_ids.reshape(batch_size, seq_len + context_len)
    hidden_states = embed[token_ids[:, context_len:]].float().reshape(batch_size, seq_len, config.hidden_size)
    if is_decode:
        decode_context = tm.build_decode_context(
            setup, config, batch_size, 0, batch_size, prefix=embed[token_ids[:, :context_len]].float()
        )
        decode_context.fill_kv_cache(mesh_device, layer, page_table_tt, apply_input_norm=True)
    _, _, rope_mats, tt_position_idx = tm.build_rope_inputs(
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
    replicate = ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)

    def make_input():
        return ttnn.from_torch(
            hidden_states.reshape(1, 1, num_tokens, -1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )

    def run_layer(x):
        return layer(
            x,
            position_embeddings=rope_mats,
            position_idx=tt_position_idx,
            page_table=page_table_tt,
            is_decode=is_decode,
        )

    record = {"case": case, "batch": batch_size, "seq_len": seq_len, "context_len": context_len}

    # compile pass (eager)
    _sync(mesh_device)
    t0 = time.perf_counter()
    out = run_layer(make_input())
    _sync(mesh_device)
    record["compile_step_s"] = time.perf_counter() - t0
    out.deallocate(True)
    # second eager step (compiled) for reference
    x = make_input()
    _sync(mesh_device)
    t0 = time.perf_counter()
    out = run_layer(x)
    _sync(mesh_device)
    record["eager_step_warm_s"] = time.perf_counter() - t0
    out.deallocate(True)

    # 1. labeled step
    union = _labeled_step(layer, make_input(), rope_mats, tt_position_idx, page_table_tt, is_decode, f"label_{case}")
    record["union_of_experts"] = union
    logger.info(f"[{case}] union of experts = {union}")

    # 2. plain eager steps
    inputs = [make_input() for _ in range(EAGER_STEPS)]
    _sync(mesh_device)
    signpost(f"eager_{case}_start")
    t0 = time.perf_counter()
    walls = []
    for x in inputs:
        t1 = time.perf_counter()
        out = run_layer(x)
        _sync(mesh_device)
        walls.append(time.perf_counter() - t1)
        out.deallocate(True)
    total = time.perf_counter() - t0
    signpost(f"eager_{case}_stop")
    record["eager_steps"] = EAGER_STEPS
    record["eager_ms_per_step"] = [w * 1e3 for w in walls]
    record["eager_ms_per_step_mean"] = total / EAGER_STEPS * 1e3
    logger.info(f"[{case}] eager {EAGER_STEPS} steps: {[round(w * 1e3, 2) for w in walls]} ms")

    # 3. trace (decode only: prefill 1024 takes the untraceable host-planned sorted path)
    if is_decode:
        tt_in = make_input()  # persistent trace input; the layer consumes a clone
        out = run_layer(ttnn.clone(tt_in))
        _sync(mesh_device)
        out.deallocate(True)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = run_layer(ttnn.clone(tt_in))
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        _sync(mesh_device)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        _sync(mesh_device)
        signpost(f"trace_{case}_start")
        t0 = time.perf_counter()
        for _ in range(TRACE_REPLAYS):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        _sync(mesh_device)
        nonblocking = (time.perf_counter() - t0) / TRACE_REPLAYS
        blocking = []
        for _ in range(TRACE_REPLAYS):
            t1 = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            blocking.append(time.perf_counter() - t1)
        signpost(f"trace_{case}_stop")
        record["trace_replays"] = TRACE_REPLAYS
        record["trace_ms_per_replay_nonblocking"] = nonblocking * 1e3
        record["trace_ms_per_replay_blocking_mean"] = sum(blocking) / len(blocking) * 1e3
        record["trace_ms_per_replay_blocking_min"] = min(blocking) * 1e3
        logger.info(
            f"[{case}] trace replay: {nonblocking * 1e3:.3f} ms non-blocking back-to-back, "
            f"{sum(blocking) / len(blocking) * 1e3:.3f} ms blocking (min {min(blocking) * 1e3:.3f})"
        )
        ttnn.release_trace(mesh_device, trace_id)
        out.deallocate(True)
        tt_in.deallocate(True)

    _dump(case, record)
    # drain the device profiler buffers before the mesh closes (harmless without the profiler)
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
    except Exception:
        pass
