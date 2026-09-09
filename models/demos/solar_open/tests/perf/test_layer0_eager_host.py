# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Eager (non-traced) HOST-time attribution of one real-weight decoder layer at decode (phase 3c, stage E1).

Companion of test_layer0_device_perf.py for its "eager ms per step" number. Eager decode is not a production path
(demos, teacher-forced and vLLM replay traces), but its b32 wall moved 3.21 -> 4.10 ms when EGP landed although the
device kernels fell 1255 -> 798 us per step, and later runs of the SAME arm spread 3.06-4.99 ms. Everything here runs
in ONE process so the arms share the host's CPU placement and frequency:

  1. arms (``SOLAR_OPEN_E1_ARMS``, default ``on,off,on,off`` = SOLAR_OPEN_DECODE_EGP presets) are swapped on the built
     layer (``layer.mlp.experts.program_config``) between blocks of ``SOLAR_OPEN_PERF_EAGER_STEPS`` eager steps; per step
     the wall splits into the host enqueue time (``run_layer`` returns) and the tail (``synchronize_device``);
  2. per block: the CPU the main thread ran on and its MHz, and the CPU time of every thread of the process over the
     block (utime + stime from /proc/self/task/*/stat), i.e. how much CPU the process burns per ms of wall;
  3. cProfile of one extra block per distinct arm: cumulative / total time per Python-visible callable (ttnn ops are
     nanobind builtins, so their C++ time lands on the op entry), dumped next to ``SOLAR_OPEN_PERF_OUT`` as
     ``<case>_<arm>.prof`` and summarized (top entries by tottime) in the JSON.

    SOLAR_OPEN_PERF_OUT=/path/e1.json pytest models/demos/solar_open/tests/perf/test_layer0_eager_host.py \
        -k "1x8 and decode_b32" -p no:cacheprovider

Not a correctness test; needs the real layer-0 shards and the tokenizer like test_layer0_device_perf.py.
"""

import cProfile
import io
import json
import os
import pstats
import threading
import time
from dataclasses import replace

import pytest
from loguru import logger

import ttnn
from models.demos.solar_open.tests.perf.test_layer0_device_perf import layer0_weights  # noqa: F401  (module fixture)
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.solar_open.tests.test_layer0_real_weights import (
    LAYER_IDX,
    _snapshot_dir,
    _token_ids,
    build_reference_layer,
)
from models.demos.solar_open.tests.unit import test_modules as tm
from models.demos.solar_open.tt.expert_configs import decode_egp_overrides, solar_open_program_config

PERF_OUT = os.getenv("SOLAR_OPEN_PERF_OUT", "")
EAGER_STEPS = int(os.getenv("SOLAR_OPEN_PERF_EAGER_STEPS", "20"))
ARMS = [a.strip() for a in os.getenv("SOLAR_OPEN_E1_ARMS", "on,off,on,off").split(",") if a.strip()]
PROFILE_TOP = 40


def _dump(case, record):
    if not PERF_OUT:
        return
    data = {}
    if os.path.isfile(PERF_OUT):
        data = json.loads(open(PERF_OUT).read())
    data[case] = record
    with open(PERF_OUT, "w") as f:
        json.dump(data, f, indent=2)


def _cpu_now():
    """CPU the calling thread last ran on (/proc/self/stat field 39; the comm field may contain spaces)."""
    with open("/proc/thread-self/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return int(fields[36])


def _cpu_mhz():
    mhz = {}
    cpu = None
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("processor"):
                cpu = int(line.split(":")[1])
            elif line.startswith("cpu MHz") and cpu is not None:
                mhz[cpu] = float(line.split(":")[1])
    return mhz


def _thread_cpu_times():
    """{tid: (comm, cpu_seconds)} for every thread of this process."""
    tck = os.sysconf("SC_CLK_TCK")
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/stat") as f:
                raw = f.read()
            comm = raw[raw.index("(") + 1 : raw.rindex(")")]
            fields = raw.rsplit(")", 1)[1].split()
            out[int(tid)] = (comm, (int(fields[11]) + int(fields[12])) / tck)
        except (FileNotFoundError, ProcessLookupError, ValueError):
            continue
    return out


def _block(run_layer, make_input, mesh_device, steps):
    """`steps` eager steps; returns per-step (enqueue_ms, sync_ms, wall_ms) plus the thread CPU-time deltas."""
    inputs = [make_input() for _ in range(steps)]
    ttnn.synchronize_device(mesh_device)
    cpu0, t_before = _cpu_now(), _thread_cpu_times()
    rows = []
    t_block = time.perf_counter()
    for x in inputs:
        t0 = time.perf_counter()
        out = run_layer(x)
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
        out.deallocate(True)
        rows.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t2 - t0) * 1e3))
    block_wall = time.perf_counter() - t_block
    t_after, cpu1 = _thread_cpu_times(), _cpu_now()
    mhz = _cpu_mhz()
    deltas = {}
    for tid, (comm, s1) in t_after.items():
        s0 = t_before.get(tid, (comm, 0.0))[1]
        if s1 - s0 > 0:
            deltas[f"{tid}:{comm}"] = round((s1 - s0) * 1e3, 1)  # ms of CPU over the block
    main_tid = threading.get_native_id()
    return {
        "steps": steps,
        "enqueue_ms": [round(r[0], 3) for r in rows],
        "sync_ms": [round(r[1], 3) for r in rows],
        "wall_ms": [round(r[2], 3) for r in rows],
        "wall_ms_mean": round(sum(r[2] for r in rows) / steps, 3),
        "wall_ms_min": round(min(r[2] for r in rows), 3),
        "enqueue_ms_mean": round(sum(r[0] for r in rows) / steps, 3),
        "sync_ms_mean": round(sum(r[1] for r in rows) / steps, 3),
        "block_wall_ms": round(block_wall * 1e3, 1),
        "cpu_before_after": [cpu0, cpu1],
        "cpu_mhz_main": mhz.get(cpu1),
        "cpu_mhz_min_max": [min(mhz.values()), max(mhz.values())] if mhz else None,
        "main_tid": main_tid,
        "thread_cpu_ms": dict(sorted(deltas.items(), key=lambda kv: -kv[1])),
        "process_cpu_ms_total": round(sum(deltas.values()), 1),
        "cpu_ms_per_wall_ms": round(sum(deltas.values()) / (block_wall * 1e3), 2),
    }


def _profile_block(run_layer, make_input, mesh_device, steps, path):
    inputs = [make_input() for _ in range(steps)]
    ttnn.synchronize_device(mesh_device)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for x in inputs:
        out = run_layer(x)
        ttnn.synchronize_device(mesh_device)
        out.deallocate(True)
    pr.disable()
    wall = time.perf_counter() - t0
    if path:
        pr.dump_stats(path)
    st = pstats.Stats(pr)
    total_tt = sum(v[2] for v in st.stats.values())
    rows = []
    for (fn, ln, name), (cc, nc, tt, ct, _callers) in st.stats.items():
        rows.append((tt, ct, nc, f"{os.path.basename(fn)}:{ln}:{name}" if fn != "~" else name))
    rows.sort(key=lambda r: -r[0])
    top_tt = [
        {
            "tottime_ms_per_step": round(tt * 1e3 / steps, 3),
            "cumtime_ms_per_step": round(ct * 1e3 / steps, 3),
            "calls_per_step": round(nc / steps, 1),
            "func": f,
        }
        for tt, ct, nc, f in rows[:PROFILE_TOP]
    ]
    rows.sort(key=lambda r: -r[1])
    top_ct = [
        {
            "cumtime_ms_per_step": round(ct * 1e3 / steps, 3),
            "tottime_ms_per_step": round(tt * 1e3 / steps, 3),
            "calls_per_step": round(nc / steps, 1),
            "func": f,
        }
        for tt, ct, nc, f in rows[:PROFILE_TOP]
    ]
    buf = io.StringIO()
    st.stream = buf
    return {
        "steps": steps,
        "wall_ms_per_step_under_cprofile": round(wall * 1e3 / steps, 3),
        "profiled_tottime_ms_per_step": round(total_tt * 1e3 / steps, 3),
        "top_by_tottime": top_tt,
        "top_by_cumtime": top_ct,
    }


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("batch_size", [32, 1], ids=["decode_b32", "decode_b1"])
@parametrize_mesh_with_fabric([(1, 8)])
def test_layer0_eager_host(mesh_device, device_params, batch_size, layer0_weights, reset_seeds):  # noqa: F811
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape != (1, 8):
        pytest.skip("sized for the 1x8 TP=8 mesh")
    seq_len = 1
    case = f"decode_b{batch_size}"

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    state_dict, embed = layer0_weights
    reference_layer = build_reference_layer(config, state_dict)
    paged_attention_config, page_table_tt = tm.make_paged_attention(mesh_device, batch_size, seq_len)
    layer = tm.setup_decoder_layer(
        setup, reference_layer, batch_size, seq_len, layer_idx=LAYER_IDX, paged_attention_config=paged_attention_config
    )
    del reference_layer

    context_len = tm.DECODE_CONTEXT_LEN
    token_ids = _token_ids(
        _snapshot_dir(), batch_size * (seq_len + context_len), config.vocab_size, setup["model_args"]
    )
    token_ids = token_ids.reshape(batch_size, seq_len + context_len)
    hidden_states = embed[token_ids[:, context_len:]].float().reshape(batch_size, seq_len, config.hidden_size)
    decode_context = tm.build_decode_context(
        setup, config, batch_size, 0, batch_size, prefix=embed[token_ids[:, :context_len]].float()
    )
    decode_context.fill_kv_cache(mesh_device, layer, page_table_tt, apply_input_norm=True)
    _, _, rope_mats, tt_position_idx = tm.build_rope_inputs(
        setup, config, hidden_states, batch_size, seq_len, context_len, batch_size, True, cache_position=context_len
    )
    replicate = ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)

    def make_input():
        return ttnn.from_torch(
            hidden_states.reshape(1, 1, batch_size, -1),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
        )

    def run_layer(x):
        return layer(
            x, position_embeddings=rope_mats, position_idx=tt_position_idx, page_table=page_table_tt, is_decode=True
        )

    base_pc = solar_open_program_config(mesh_device)
    record = {
        "case": case,
        "batch": batch_size,
        "arms": ARMS,
        "steps_per_block": EAGER_STEPS,
        "sched_affinity": sorted(os.sched_getaffinity(0)),
        "env": {
            k: os.environ.get(k)
            for k in ("SOLAR_OPEN_DECODE_EGP", "TT_METAL_NUMA_BASED_AFFINITY", "TT_SPARSE_MATMUL_EGP_ZERO_FILL")
        },
        "program_cache_entries": [],
        "blocks": [],
        "profiles": {},
    }
    compiled = set()
    for i, arm in enumerate(ARMS):
        layer.mlp.experts.program_config = replace(base_pc, **decode_egp_overrides(arm))
        n0 = mesh_device.num_program_cache_entries()
        # compile step of this arm (excluded); a second warm step so every arm block starts from the same state
        for _ in range(2 if arm not in compiled else 1):
            out = run_layer(make_input())
            ttnn.synchronize_device(mesh_device)
            out.deallocate(True)
        compiled.add(arm)
        n1 = mesh_device.num_program_cache_entries()
        blk = _block(run_layer, make_input, mesh_device, EAGER_STEPS)
        blk["arm"] = arm
        blk["block_index"] = i
        blk["program_cache_entries_before_after"] = [n0, n1]
        record["blocks"].append(blk)
        logger.info(
            f"[{case}] block {i} arm={arm}: wall {blk['wall_ms_mean']:.3f} ms/step (min {blk['wall_ms_min']:.3f}; "
            f"enqueue {blk['enqueue_ms_mean']:.3f} + sync {blk['sync_ms_mean']:.3f}), cpu {blk['cpu_before_after']} "
            f"@ {blk['cpu_mhz_main']} MHz, process CPU {blk['cpu_ms_per_wall_ms']} ms per wall ms, "
            f"cache {n0} -> {n1}"
        )
    # cProfile: one block per distinct arm, in the arm order given
    seen = []
    for arm in ARMS:
        if arm in seen:
            continue
        seen.append(arm)
        layer.mlp.experts.program_config = replace(base_pc, **decode_egp_overrides(arm))
        out = run_layer(make_input())
        ttnn.synchronize_device(mesh_device)
        out.deallocate(True)
        path = os.path.join(os.path.dirname(PERF_OUT), f"{case}_{arm}.prof") if PERF_OUT else ""
        prof = _profile_block(run_layer, make_input, mesh_device, EAGER_STEPS, path)
        record["profiles"][arm] = prof
        logger.info(
            f"[{case}] cProfile arm={arm}: {prof['wall_ms_per_step_under_cprofile']:.3f} ms/step under cProfile, "
            f"profiled tottime {prof['profiled_tottime_ms_per_step']:.3f} ms/step; top by tottime: "
            + "; ".join(f"{r['func']} {r['tottime_ms_per_step']:.3f}" for r in prof["top_by_tottime"][:8])
        )
    record["program_cache_entries"] = mesh_device.num_program_cache_entries()
    _dump(case, record)
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
    except Exception:
        pass
