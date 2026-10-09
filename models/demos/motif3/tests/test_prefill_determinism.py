# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill run-to-run determinism (FEATURES_REVIEW P1, FULL_MODEL_VALIDATION §5.1). Root cause and evidence:
``/home/ttuser/hchang/experiments/motif-3/docs/determinism/INVESTIGATION.md``; the fix and its validation:
``.../docs/determinism/FIX.md``.

Root cause: ``ttnn.all_gather``'s multicast factory (BH, small payloads) load-balances packets over two routes on an
even-sized ring (the 8-chip TP axis), but each worker's completion semaphore only travels its primary route, so the
chip N/2 hops away can retire the op -- and the next op read the output -- before the last alternate-route pages land.
In the model, the PolyNorm-moment gather of the shared expert (``polynorm._ar_ag_sum``: all_gather -> ``ttnn.sum``)
summed a stale tile on one chip in a few % of the calls; the reduce-scatter / all-gather combine then spread it to
every chip (all replicas identical, different between runs). Fix: ``MotifCCL(ring_gather="safe")`` (default since 2026-10-03; "lean" spares decode,
``cfg.ring_gather`` / ``MOTIF3_RING_GATHER``) runs every gather predicted on that path that streams more than one
circular-buffer page per link -- and, for ``race_free=True`` callers (the MoE prefill combine) or inside
``MotifCCL.race_free_scope()``, every predicted gather -- as all_broadcast + concat (bitwise the same data, no race);
decode's single-page gathers stay native (no decode cost). ``"safe"`` reroutes every predicted gather, decode's too.

* ``test_ring_gather_plan_host`` (host): the predicate (:func:`ccl.native_ag_multicast_alt_routes`) on the model's
  payloads, the knob plumbing (``MotifTTConfig.ring_gather`` / ``MOTIF3_RING_GATHER``, the validated
  ``MotifCCL.ring_gather``), the routing of ``all_gather`` / ``all_reduce`` per mode, ``race_free`` and
  ``race_free_scope`` (fake mesh, monkeypatched ops).
* ``test_fingerprint_compare_host`` (host): the cross-repeat / cross-process comparison of the device test.
* ``test_prefill_determinism`` (device, real weights from the TT cache; spawns its own worker processes): the
  acceptance test of the fix. ``MOTIF3_DET_PROCESSES`` (2) separate processes, one after the other; each opens the
  Galaxy as served (``open_motif_mesh``), builds the ``MOTIF3_DET_LAYERS`` (53) layer model and a ``MotifGenerator``,
  compiles every program (serving prefill buckets, the draft-1 single shots, the decode trace), then runs
  ``MOTIF3_DET_REPEATS`` (5) **cold** repeats of four prompts at buckets **128 / 1024 / 4096 / 8192** (the C2 stream
  cut at 100, 3100 and 8000 tokens, and the 893-token C2 prompt en_technical):

  1. model path (``MotifModel.prefill``, draft-1 single shot, real page table): the final residual streams of all
     rows (chip 0, sha256; first differing row vs repeat 0), the 32 replicas identical (repeat 0), and the last
     token's full-vocab logits, bitwise;
  2. serving path (``MotifGenerator.prefill_forward``, cold: start 0, fresh KV blocks every repeat) of the four
     prompts into lanes 0 / 8 / 16 / 24 (one per DP row): the last-token logits, bitwise; then
  3. ``MOTIF3_DET_GREEDY_STEPS`` (160) greedy tokens per lane with the traced decode: the token streams and the
     sha256 of every step's full-vocab logits of each lane.

  Everything must be identical across the repeats of a process AND across the processes (and no program may compile
  after the decode capture). The decode of step 3 runs with ``ring_gather=MOTIF3_DET_DECODE_RING_GATHER`` (default
  ``"safe"``: race-free decode, so the streams isolate the prefill; decode's own rare exposure under the default
  ``"lean"`` -- ~1 event per 10^4 steps, INVESTIGATION.md §4 -- is measured by setting it to ``"lean"``). Prefill
  runs with the configured default. ``MOTIF3_RING_GATHER=native`` is the pre-fix control (expected to FAIL).
  ``MOTIF3_DET_OUT=<dir>`` keeps every process's fingerprint JSON and log there (default: pytest's tmp_path).
  The workers are spawned by this pytest process, which must not have opened the Galaxy itself: UMD keeps the chips'
  ``CHIP_IN_USE`` lock until the process exits (a child would wait forever). It therefore runs before the fixture
  tests below and skips if the process already holds a device; run it alone by node id
  (``test_prefill_determinism.py::test_prefill_determinism``; ``-k`` also matches the module name).
  ``python models/demos/motif3/tests/test_prefill_determinism.py --worker --out X.json`` runs one worker by hand.
* ``test_ring_gather_safe_equals_native`` (device): for every race-prone payload class of the model (PolyNorm moments
  fp32 at prefill / decode sizes, the ar_tp gathers of small buckets and decode), the safe path is bitwise equal to the
  native ``ttnn.all_gather`` on all 32 chips, and ``all_reduce`` likewise; the predicate's routing is asserted.
* ``test_ring_gather_race_isolated`` (device, no weights): the race outside the model -- the MoE combine's
  collectives skew the chips, then a TP gather (or ``ar_tp``) and an immediate consumer, 300 iterations per payload
  (moments of buckets 4096 / 1024 / 512 / 128, the MoE combine of bucket 128, the decode AR): no payload the default
  mode reroutes may hand its consumer a stale tile; the ones it leaves native are reported
  (``MOTIF3_DET_NATIVE_CONTROL=1``: the native gather's counts too; up to 62 % of the calls raced).

Run (``-p no:cacheprovider --timeout=0``; ~11 min for the whole file: ~5 min per worker process, 1 min for the rest)::

    scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q \
        models/demos/motif3/tests/test_prefill_determinism.py -k host
    scripts/devrun.sh -t 3600 -n prefill_determinism -- python -m pytest \
        models/demos/motif3/tests/test_prefill_determinism.py -s -p no:cacheprovider --timeout=0
    # the acceptance test alone (e.g. with MOTIF3_DET_OUT=<dir>, MOTIF3_RING_GATHER=native as the control):
    scripts/devrun.sh -t 3600 -n prefill_determinism -- python -m pytest -s -p no:cacheprovider --timeout=0 \
        models/demos/motif3/tests/test_prefill_determinism.py::test_prefill_determinism
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pytest
import torch

import ttnn
from models.demos.motif3.tt.model_config import device_params

GOLDEN_DIR = Path(os.environ.get("MOTIF3_GOLDEN_STREAM_DIR", "/home/ttuser/hchang/experiments/motif-3/goldens/c2"))
GOLDEN_PROMPTS = GOLDEN_DIR / "prompts.json"
N_LAYERS = int(os.environ.get("MOTIF3_DET_LAYERS", "53"))
REPEATS = int(os.environ.get("MOTIF3_DET_REPEATS", "5"))
PROCESSES = int(os.environ.get("MOTIF3_DET_PROCESSES", "2"))
GREEDY_STEPS = int(os.environ.get("MOTIF3_DET_GREEDY_STEPS", "160"))
DECODE_RING_GATHER = (os.environ.get("MOTIF3_DET_DECODE_RING_GATHER") or "safe").strip()
WORKER_TIMEOUT_S = int(os.environ.get("MOTIF3_DET_WORKER_TIMEOUT", "3000"))
MAX_MODEL_LEN = 16384  # span cap min(8192, max_model_len) = 8192: the 8000-token prompt is one sp0 chunk of 8192
# (name, prompt spec, expected prefill bucket); lanes 0 / 8 / 16 / 24: one per DP row
PROMPTS: Tuple[Tuple[str, str, int], ...] = (
    ("short_100", "stream:100", 128),
    ("en_technical", "c2:en_technical", 1024),
    ("stream_3100", "stream:3100", 4096),
    ("stream_8000", "stream:8000", 8192),
)
LANES = (0, 8, 16, 24)
MESH = [pytest.param((4, 8), device_params(), id="4x8")]


# ============================================================================================================
# host
# ============================================================================================================
class _FakeMesh:
    def __init__(self, shape):
        self.shape = shape


def _fake_tensor(shape, dtype=ttnn.bfloat16, padded=None, layout=ttnn.TILE_LAYOUT):
    mc = SimpleNamespace(is_sharded=lambda: False)
    return SimpleNamespace(
        shape=list(shape), padded_shape=list(padded or shape), dtype=dtype, layout=layout, memory_config=lambda: mc
    )


def test_ring_gather_plan_host(monkeypatch):
    """The multicast-race predicate on the model's payloads, the knob, and the routing of all_gather / all_reduce."""
    from models.demos.motif3.tt import ccl as C
    from models.demos.motif3.tt.model_config import DEFAULT_HF_META_DIR, RING_GATHER_MODES, MotifTTConfig

    hf_meta = str(DEFAULT_HF_META_DIR)

    f = C.native_ag_multicast_alt_routes
    tp = dict(dim=3, num_devices=8, ring_axis=True)
    # PolyNorm moments [1, 3, R, 32] fp32 (fp32 tile 4096 B: multicast below 4 MB per link = R < 2731)
    for rows, mc in ((32, True), (256, True), (1024, True), (2048, True), (2731, False), (8192, False)):
        assert f([1, 3, rows, 32], [1, 3, rows, 32], "FLOAT32", True, **tp) is mc, rows
    # ar_tp gathers [1, 1, R, 512] bf16 (bf16 tile 2048 B: multicast below 1e6 B per link = R < 245)
    for rows, mc in ((8, True), (32, True), (128, True), (244, True), (256, False), (1024, False)):
        assert f([1, 1, rows, 512], [1, 1, max(rows, 32), 512], "BFLOAT16", True, **tp) is mc, rows
    # no alternate routes on a line (the 4-chip DP axis on this Galaxy) or an odd ring
    assert not f([1, 3, 1024, 32], [1, 3, 1024, 32], "FLOAT32", True, dim=3, num_devices=4, ring_axis=False)
    assert not f([1, 3, 1024, 32], [1, 3, 1024, 32], "FLOAT32", True, dim=3, num_devices=7, ring_axis=True)
    # unknown dtype: the safe side
    assert f([1, 1, 32, 32], [1, 1, 32, 32], "SOMETHING_NEW", True, **tp)
    # "lean": circular-buffer pages per link of the multicast factory (2 links, 4352 B packets, x4 for TILE)
    cb = C.native_ag_cb_pages_per_link
    assert cb([1, 1, 32, 512], "BFLOAT16", True) == 1  # decode ar_tp gather (8 lanes, tile-padded)
    assert cb([1, 3, 32, 32], "FLOAT32", True) == 1  # decode / bucket-128 PolyNorm moments
    assert cb([1, 1, 128, 512], "BFLOAT16", True) == 4  # bucket-128 attention ar_tp gather
    assert cb([1, 3, 1024, 32], "FLOAT32", True) == 12  # bucket-4096 moments (the observed race)
    # the isolated sweep (INVESTIGATION.md §3.5): every 2+ page payload raced; of the 1-page ones, bf16 [32, 512]
    assert [cb([1, 3, r, 32], "FLOAT32", True) for r in (64, 128, 256)] == [1, 2, 3]  # buckets 256 / 512 / 1024
    assert cb([1, 1, 64, 512], "BFLOAT16", True) == 2  # bucket-256 MoE ar_tp gather

    # knob: config default / env / validation
    monkeypatch.delenv("MOTIF3_RING_GATHER", raising=False)
    assert RING_GATHER_MODES == ("safe", "lean", "native")
    cfg = MotifTTConfig.from_hf_config(hf_meta, mesh_shape=(4, 8))
    assert cfg.ring_gather == "safe"  # default since 2026-10-03 (lead decision: decode gathers race ~1e-4/step under "lean")
    monkeypatch.setenv("MOTIF3_RING_GATHER", "native")
    assert MotifTTConfig.from_hf_config(hf_meta, mesh_shape=(4, 8)).ring_gather == "native"
    monkeypatch.setenv("MOTIF3_RING_GATHER", "bogus")
    with pytest.raises(ValueError):
        MotifTTConfig.from_hf_config(hf_meta, mesh_shape=(4, 8))
    monkeypatch.delenv("MOTIF3_RING_GATHER", raising=False)
    mesh = _FakeMesh((4, 8))
    with pytest.raises(ValueError):
        C.MotifCCL(mesh, cfg, ring_gather="bogus")
    assert C.MotifCCL(mesh, cfg).ring_gather == "safe"
    assert C.MotifCCL(mesh, None).ring_gather == "safe"
    assert C.MotifCCL(mesh, cfg, ring_gather="native").ring_gather == "native"
    switched = C.MotifCCL(mesh, cfg)
    switched.ring_gather = "lean"  # tests switch the shared MotifCCL between calls: validated like the constructor
    assert switched.ring_gather == "lean"
    with pytest.raises(ValueError):
        switched.ring_gather = "Safe"
    assert switched.ring_gather == "lean"

    # routing (fake ops): race-prone TP gathers -> composite; large / DP-line gathers -> native ttnn.all_gather
    calls = []
    monkeypatch.setattr(ttnn, "all_gather", lambda x, **kw: calls.append(("native", kw["cluster_axis"])) or "nat")
    monkeypatch.setattr(ttnn, "deallocate", lambda t, *a, **k: None)
    tp_ax, dp_ax = cfg.axes.tp_axis, cfg.axes.dp_axis
    for mode in ("safe", "lean", "native"):
        ccl = C.MotifCCL(mesh, cfg, ring_gather=mode)
        ccl.l1_small_semaphores = True  # as on a real mesh with an L1_SMALL region
        ccl._ring_axis = {tp_ax: True, dp_ax: False}  # TORUS_Y: TP ring, DP line
        monkeypatch.setattr(ccl, "_all_gather_safe", lambda x, dim, ca, mc: calls.append(("safe", ca)) or "cmp")
        calls.clear()
        ccl.all_gather(_fake_tensor([1, 3, 1024, 32], ttnn.float32), 3, "tp")  # moments: race-prone
        ccl.all_gather(_fake_tensor([1, 3, 8192, 32], ttnn.float32), 3, "tp")  # unicast regime
        ccl.all_gather(_fake_tensor([1, 1, 1024, 4096]), 2, "dp")  # DP line
        first = ("native", tp_ax) if mode == "native" else ("safe", tp_ax)  # 12 CB pages per link: lean routes it too
        assert calls == [first, ("native", tp_ax), ("native", dp_ax)], (mode, calls)
        calls.clear()
        ccl.all_gather(_fake_tensor([1, 3, 32, 32], ttnn.float32), 3, "tp")  # decode-sized: one CB page per link
        assert calls == [("safe" if mode == "safe" else "native", tp_ax)], (mode, calls)
        # race_free (latency-tolerant callers: the MoE prefill combine): every race-prone payload unless "native"
        rerouted = ("native" if mode == "native" else "safe", tp_ax)
        calls.clear()
        ccl.all_gather(_fake_tensor([1, 3, 32, 32], ttnn.float32), 3, "tp", race_free=True)
        assert calls == [rerouted], (mode, calls)
        # race_free_scope: the same for every call inside it, nestable, restored on exit (also on an exception);
        # never touches a gather that is not race-prone (unicast regime, DP line)
        calls.clear()
        with ccl.race_free_scope():
            assert ccl.in_race_free_scope
            ccl.all_gather(_fake_tensor([1, 3, 32, 32], ttnn.float32), 3, "tp")
            with ccl.race_free_scope():
                ccl.all_gather(_fake_tensor([1, 3, 64, 32], ttnn.float32), 3, "tp")
            ccl.all_gather(_fake_tensor([1, 3, 32, 32], ttnn.float32), 3, "tp")
            ccl.all_gather(_fake_tensor([1, 3, 8192, 32], ttnn.float32), 3, "tp")
            ccl.all_gather(_fake_tensor([1, 1, 1024, 4096]), 2, "dp")
        assert not ccl.in_race_free_scope
        assert calls == [rerouted, rerouted, rerouted, ("native", tp_ax), ("native", dp_ax)], (mode, calls)
        with pytest.raises(RuntimeError):
            with ccl.race_free_scope():
                raise RuntimeError("boom")
        assert not ccl.in_race_free_scope
        with ccl.race_free_scope(enabled=False):
            assert not ccl.in_race_free_scope
        calls.clear()  # after the scope: the decode-sized gather is back on its mode's default
        ccl.all_gather(_fake_tensor([1, 3, 32, 32], ttnn.float32), 3, "tp")
        assert calls == [("safe" if mode == "safe" else "native", tp_ax)], (mode, calls)
        # all_reduce: the AG half of RS + AG follows the same rule
        calls.clear()
        monkeypatch.setattr(ccl, "_reduce_scatter", lambda x, dim, ca, n, mc, **kw: _fake_tensor([1, 1, 128, 512]))
        ccl.all_reduce(_fake_tensor([1, 1, 128, 4096]), "tp")  # its [128, 512] gather: 4 CB pages per link
        assert calls == [first], (mode, calls)
        calls.clear()  # the MoE prefill combine at bucket 128: [32, 512] bf16, one CB page per link
        monkeypatch.setattr(ccl, "_reduce_scatter", lambda x, dim, ca, n, mc, **kw: _fake_tensor([1, 1, 32, 512]))
        ccl.ar_tp(_fake_tensor([1, 1, 32, 4096]))
        ccl.ar_tp(_fake_tensor([1, 1, 32, 4096]), race_free=True)
        with ccl.race_free_scope():
            ccl.ar_tp(_fake_tensor([1, 1, 32, 4096]))
        single = "safe" if mode == "safe" else "native"
        assert calls == [(single, tp_ax), rerouted, rerouted], (mode, calls)


def _fp(tok_shift=0, stream_shift=0, rep=3, logits_bad=None):
    """A synthetic fingerprint of one process (:func:`determinism_session` layout)."""
    names = [p[0] for p in PROMPTS]
    out = {"meta": {"repeats": rep}, "model": {}, "serving": {}, "greedy": {}}
    for i, n in enumerate(names):
        s = [f"s{i}"] * rep
        if stream_shift and i == 1:
            s[-1] = "s-bad"
        lg = [f"l{i}"] * rep
        if logits_bad == n:
            lg[1] = "l-bad"
        out["model"][n] = {"S": 10 + i, "bucket": PROMPTS[i][2], "streams": s, "logits": lg,
                           "replicas_identical": True, "first_diff_rows": [None] * (rep - 1)}
        out["serving"][n] = {"logits": [f"v{i}"] * rep}
        toks = [[1, 2, 3 + i]] * rep
        if tok_shift and i == 2:
            toks = toks[:-1] + [[1, 2, 99]]
        out["greedy"][n] = {"lane": LANES[i], "tokens": toks, "decode_logits": [f"d{i}"] * rep,
                            "margins": [[0.5, 0.25, 0.125]] * rep}
    return out


def test_fingerprint_compare_host():
    """compare_fingerprints: identical runs pass; a differing repeat / process / replica is reported by name."""
    ok, lines = compare_fingerprints([_fp(), _fp()])
    assert ok == [] and any("identical" in l for l in lines), (ok, lines)
    bad, _ = compare_fingerprints([_fp(stream_shift=1)])
    assert len(bad) == 1 and "en_technical" in bad[0] and "streams" in bad[0], bad
    bad, _ = compare_fingerprints([_fp(), _fp(tok_shift=1)])
    assert any("stream_3100" in b and "greedy" in b for b in bad), bad
    other = _fp()
    other["model"]["short_100"]["streams"] = ["s-other"] * 3
    bad, _ = compare_fingerprints([_fp(), other])
    assert len(bad) == 1 and "across processes" in bad[0] and "short_100" in bad[0], bad
    bad, _ = compare_fingerprints([_fp(logits_bad="stream_8000")])
    assert len(bad) == 1 and "stream_8000" in bad[0] and "logits" in bad[0], bad
    rep = _fp()
    rep["model"]["short_100"]["replicas_identical"] = False
    bad, _ = compare_fingerprints([rep])
    assert len(bad) == 1 and "replicas" in bad[0], bad


# ============================================================================================================
# the comparison (host, pure)
# ============================================================================================================
def _distinct(xs) -> int:
    return len({json.dumps(x) for x in xs})


def compare_fingerprints(fps: Sequence[dict]) -> Tuple[List[str], List[str]]:
    """``(failures, report lines)`` over the fingerprints of one or more processes (:func:`determinism_session`):
    every quantity must be identical across the repeats of each process and across the processes."""
    failures: List[str] = []
    lines: List[str] = []
    names = list(fps[0]["model"]) if fps else []
    checks = (  # (section, key, label)
        ("model", "streams", "model-path residual streams (all rows, chip 0)"),
        ("model", "logits", "model-path last-token logits"),
        ("serving", "logits", "serving prefill_forward last-token logits"),
        ("greedy", "tokens", "greedy token streams"),
        ("greedy", "decode_logits", "greedy per-step decode logits"),
    )
    for n in names:
        for sec, key, label in checks:
            per_proc = [fp[sec][n][key] for fp in fps]
            for p, vals in enumerate(per_proc):
                d = _distinct(vals)
                if d != 1:
                    extra = ""
                    if sec == "model" and key == "streams":
                        extra = f"; first differing rows vs repeat 0 {fps[p]['model'][n].get('first_diff_rows')}"
                    if sec == "greedy" and key == "tokens":
                        ref = vals[0]
                        div = [next((k for k, (a, b) in enumerate(zip(ref, v)) if a != b), None) for v in vals[1:]]
                        extra = f"; first divergence step per repeat vs repeat 0 {div}"
                    failures.append(f"{n}: {label} {sec}/{key}: {d} distinct of {len(vals)} repeats in process {p}"
                                    f"{extra}")
            flat = [v for vals in per_proc for v in vals]
            if len(fps) > 1 and all(_distinct(v) == 1 for v in per_proc) and _distinct(flat) != 1:
                failures.append(f"{n}: {label} {sec}/{key} differ across processes "
                                f"({[json.dumps(v[0])[:24] for v in per_proc]})")
        for p, fp in enumerate(fps):
            if not fp["model"][n].get("replicas_identical", False):
                failures.append(f"{n}: the 32 replicas of the model-path streams differ in process {p}")
        m0 = fps[0]["model"][n]
        nrep = sum(len(fp["model"][n]["streams"]) for fp in fps)
        lines.append(
            f"[determinism] {n} (S={m0['S']}, bucket {m0['bucket']}): {len(fps)} process(es) x "
            f"{len(m0['streams'])} repeats = {nrep} cold prefills; distinct: streams "
            f"{_distinct([v for fp in fps for v in fp['model'][n]['streams']])}, logits "
            f"{_distinct([v for fp in fps for v in fp['model'][n]['logits']])}, serving logits "
            f"{_distinct([v for fp in fps for v in fp['serving'][n]['logits']])}, greedy streams "
            f"{_distinct([v for fp in fps for v in fp['greedy'][n]['tokens']])} "
            f"({len(fps[0]['greedy'][n]['tokens'][0])} tokens), decode logits "
            f"{_distinct([v for fp in fps for v in fp['greedy'][n]['decode_logits']])}"
            + (" -> identical" if not any(f.startswith(f"{n}:") for f in failures) else " -> DIFFERENT")
        )
    return failures, lines


# ============================================================================================================
# device worker (one process): the session
# ============================================================================================================
def _sha(t: torch.Tensor) -> str:
    """sha256 (16 hex) of a tensor's bytes, any dtype (bf16 included)."""
    return hashlib.sha256(t.contiguous().flatten().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def _prompt_ids(spec: str) -> List[int]:
    """``"stream:N"`` = the 6 C2 golden prompts concatenated x10, first N ids; ``"c2:<name>"`` = one C2 prompt."""
    from models.demos.motif3.reference import golden_stream as gs

    prompts = {p.name: p for p in gs.load_prompt_set(GOLDEN_PROMPTS)}
    kind, _, arg = spec.partition(":")
    if kind == "stream":
        stream = [i for _ in range(10) for p in prompts.values() for i in p.ids]
        return stream[: int(arg)]
    if kind == "c2":
        return list(prompts[arg].ids)
    raise ValueError(f"unknown prompt spec {spec!r}")


def _cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _model_path(ttnn_, mesh, cfg, model, pool, ids, *, replicas: bool):
    """One draft-1 single-shot ``MotifModel.prefill`` (real page table, blocks 1..n): chip 0's final streams
    ``[1, 4, S, 4096]`` (host), whether all 32 chips hold the same streams (``replicas``; else None), and the last
    token's full-vocab logits (host)."""
    from models.demos.motif3.tt.generator import prefill_page_table_host

    S = len(ids)
    bucket = cfg.prefill_bucket(S)
    entries = cfg.prefill_page_table_entries(bucket)
    pt_host = prefill_page_table_host(torch.arange(1, entries + 1, dtype=torch.int32), entries, S, cfg.kv_block_size)
    tok = model.embed.prefill_tokens_device(torch.tensor(ids, dtype=torch.int32), bucket)
    pt = ttnn_.from_torch(pt_host, dtype=ttnn_.int32, layout=ttnn_.ROW_MAJOR_LAYOUT, device=mesh,
                          memory_config=ttnn_.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn_.ReplicateTensorToMesh(mesh))
    X = tile = None
    try:
        X = model.prefill(tok, page_table=pt, kv_caches=pool, return_streams=True)
        chips = ttnn_.get_device_tensors(X)
        x0 = ttnn_.to_torch(chips[0])[..., :S, :].contiguous()
        same = None
        if replicas:
            same = all(torch.equal(x0, ttnn_.to_torch(c)[..., :S, :]) for c in chips[1:])
        tile = model.head.forward_prefill(X, S - 1)
        logits = model.head.prefill_logits_to_host(tile, S - 1).clone()
    finally:
        for t in (tok, pt, tile, X):
            if t is not None:
                ttnn_.deallocate(t)
    return x0, same, logits


def _first_diff_row(a: torch.Tensor, b: torch.Tensor) -> Optional[int]:
    if torch.equal(a, b):
        return None
    rows = torch.nonzero((a != b).reshape(-1, a.shape[-2], a.shape[-1]).any(-1).any(0))[:, 0]
    return int(rows[0])


def determinism_session(mesh, *, layers: int = N_LAYERS, repeats: int = REPEATS, steps: int = GREEDY_STEPS,
                        decode_mode: str = DECODE_RING_GATHER, log: Callable[[str], None] = print) -> dict:
    """One process's run (module docstring): build, compile everything, then ``repeats`` cold repeats of the model
    path, the serving prefill and ``steps`` greedy tokens per prompt. Returns the fingerprint (JSON-able) or
    ``{"skip": reason}`` when the TT cache is incomplete."""
    from models.demos.motif3.tt import generator_api as api
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.model import MotifModel, layer_cache_complete
    from models.demos.motif3.tt.model_config import RING_GATHER_MODES, MotifTTConfig

    if decode_mode not in RING_GATHER_MODES:
        raise ValueError(f"MOTIF3_DET_DECODE_RING_GATHER must be one of {RING_GATHER_MODES}, got {decode_mode!r}")
    t_start = time.time()
    fab = log_fabric(mesh, "prefill_determinism", printer=log)
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh, max_model_len=MAX_MODEL_LEN, num_layers=layers)
    missing = [l for l in [None, *range(layers)] if not layer_cache_complete(cfg, l)]
    if missing:
        return {"skip": f"TT cache parts {missing} not converted (the test builds from the TT cache only)"}
    prompts = [(name, _prompt_ids(spec)) for name, spec, _ in PROMPTS]
    for (name, ids), (_, _, bucket) in zip(prompts, PROMPTS):
        if cfg.prefill_bucket(len(ids)) != bucket:
            got = cfg.prefill_bucket(len(ids))
            raise AssertionError(f"{name}: {len(ids)} tokens -> bucket {got}, expected {bucket}")
    model = MotifModel(mesh, cfg, layers=range(layers), cache="auto", mtp=False)
    if model.cache_misses:
        raise AssertionError(f"not a TT-cache-only build: {model.cache_misses}")
    gen = MotifGenerator(mesh, cfg, model, log=None)
    ccl = model.ccl  # shared by every module
    prefill_mode = ccl.ring_gather
    bs, W = cfg.kv_block_size, cfg.kv_blocks_per_seq
    model_blocks = max(cfg.prefill_page_table_entries(cfg.prefill_bucket(len(ids))) for _, ids in prompts)
    lane_blocks = [_cdiv(len(ids) + steps, bs) for _, ids in prompts]
    num_blocks = 1 + model_blocks + repeats * sum(lane_blocks) + 8  # block 0 = null; fresh lane blocks every repeat
    res: dict = {
        "meta": {
            "pid": os.getpid(), "layers": layers, "repeats": repeats, "steps": steps, "prefill_ring_gather":
            prefill_mode, "decode_ring_gather": decode_mode, "fabric": {k: str(v) for k, v in fab.items()},
            "prompts": {n: {"S": len(ids), "bucket": b, "lane": l} for (n, ids), (_, _, b), l in
                        zip(prompts, PROMPTS, LANES)},
            "num_blocks": num_blocks, "max_model_len": MAX_MODEL_LEN, "timings": {},
        },
        "model": {}, "serving": {}, "greedy": {},
    }
    tm = res["meta"]["timings"]
    try:
        pool = gen.allocate_kv_cache(num_blocks=num_blocks, block_size=bs, num_layers=layers)
        tm["boot_s"] = time.time() - t_start
        log(f"[determinism] pid {os.getpid()}: {layers} layers + pool {num_blocks}x{bs} in {tm['boot_s']:.1f} s; "
            f"prefill ring_gather={prefill_mode}, decode ring_gather={decode_mode}")
        # ---- compile every program before the decode capture (serving contract) -------------------------------
        t0 = time.time()
        gen.warmup_prefill(kv_cache=pool, enable_trace=False)  # the serving path, every bucket up to the span cap
        for name, ids in prompts:  # the draft-1 single shots
            _model_path(ttnn, mesh, cfg, model, pool, ids, replicas=False)
        tm["warmup_prefill_s"] = time.time() - t0
        t0 = time.time()
        ccl.ring_gather = decode_mode
        try:
            gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=W)
            gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
        finally:
            ccl.ring_gather = prefill_mode
        tm["warmup_decode_s"] = time.time() - t0
        pc0 = mesh.num_program_cache_entries()
        log(f"[determinism] programs compiled ({pc0}) and decode trace captured in "
            f"{tm['warmup_prefill_s'] + tm['warmup_decode_s']:.1f} s")
        ref_x0: Dict[str, torch.Tensor] = {}
        for name, ids in prompts:
            res["model"][name] = {"S": len(ids), "bucket": cfg.prefill_bucket(len(ids)), "streams": [], "logits": [],
                                  "argmax": [], "replicas_identical": None, "first_diff_rows": [], "time_s": []}
            res["serving"][name] = {"logits": [], "argmax": [], "ttft_s": []}
            res["greedy"][name] = {"lane": None, "tokens": [], "decode_logits": [], "margins": []}
        next_block = 1 + model_blocks
        for r in range(repeats):
            # ---- 1. model path ------------------------------------------------------------------------------
            for name, ids in prompts:
                t1 = time.time()
                x0, same, lg = _model_path(ttnn, mesh, cfg, model, pool, ids, replicas=(r == 0))
                m = res["model"][name]
                m["time_s"].append(time.time() - t1)
                m["streams"].append(_sha(x0))
                m["logits"].append(_sha(lg))
                m["argmax"].append(int(lg.float().argmax()))
                if r == 0:
                    m["replicas_identical"] = bool(same)
                    ref_x0[name] = x0
                else:
                    m["first_diff_rows"].append(_first_diff_row(ref_x0[name], x0))
                del x0
            # ---- 2. serving prefill, cold, fresh blocks -------------------------------------------------------
            blocks = {}
            nxt = torch.zeros(api.NUM_LANES, dtype=torch.int32)
            pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
            for (name, ids), lane, nb in zip(prompts, LANES, lane_blocks):
                blocks[lane] = torch.arange(next_block, next_block + nb, dtype=torch.int32)
                next_block += nb
                pt = torch.zeros(W, dtype=torch.int32)
                n_pref = _cdiv(len(ids), bs)
                pt[:n_pref] = blocks[lane][:n_pref]
                t1 = time.time()
                lg = gen.prefill_forward(api.PrefillRequest(lane=lane, tokens=torch.tensor(ids, dtype=torch.int32),
                                                            page_table=pt), kv_cache=pool)
                s = res["serving"][name]
                s["ttft_s"].append(time.time() - t1)
                s["logits"].append(_sha(lg))
                top2 = torch.topk(lg.float(), 2).values
                nxt[lane] = int(lg.float().argmax())
                s["argmax"].append(int(nxt[lane]))
                g = res["greedy"][name]
                g["lane"] = lane
                g["tokens"].append([int(nxt[lane])])
                g["margins"].append([float(top2[0] - top2[1])])
                pos[lane] = len(ids)
            # ---- 3. greedy decode (traced) --------------------------------------------------------------------
            hs = {lane: hashlib.sha256() for lane in LANES}
            t1 = time.time()
            ccl.ring_gather = decode_mode  # the trace replays what was captured; eager pieces follow the mode too
            try:
                for _ in range(steps - 1):
                    table = torch.zeros(api.NUM_LANES, W, dtype=torch.int32)
                    for lane in LANES:
                        n = int(pos[lane]) // bs + 1
                        table[lane, :n] = blocks[lane][:n]
                    lg = gen.decode_forward(api.DecodeBatch(tokens=nxt.clone(), positions=pos.clone(),
                                                            page_table=table), kv_cache=pool, enable_trace=True)
                    for (name, _), lane in zip(prompts, LANES):
                        row = lg[lane]
                        hs[lane].update(row.contiguous().view(torch.uint8).numpy().tobytes())
                        top2 = torch.topk(row.float(), 2).values
                        nxt[lane] = int(row.float().argmax())
                        g = res["greedy"][name]
                        g["tokens"][-1].append(int(nxt[lane]))
                        g["margins"][-1].append(float(top2[0] - top2[1]))
                    pos[list(LANES)] += 1
            finally:
                ccl.ring_gather = prefill_mode
            tm.setdefault("decode_s_per_step", []).append((time.time() - t1) / max(1, steps - 1))
            for (name, _), lane in zip(prompts, LANES):
                res["greedy"][name]["decode_logits"].append(hs[lane].hexdigest()[:16])
            log(f"[determinism] repeat {r}: " + "; ".join(
                f"{n} streams {res['model'][n]['streams'][-1]} logits {res['model'][n]['logits'][-1]} serving "
                f"{res['serving'][n]['logits'][-1]} greedy {res['greedy'][n]['decode_logits'][-1]}"
                for n, _ in prompts))
        pc1 = mesh.num_program_cache_entries()
        res["meta"]["program_cache"] = [pc0, pc1]
        if pc1 != pc0:
            raise AssertionError(f"{pc1 - pc0} programs compiled after the decode capture (the warmups must cover all)")
    finally:
        gen.close()  # the KV pool and the model weights
    tm["total_s"] = time.time() - t_start
    return res


# ============================================================================================================
# device: the acceptance test (worker processes)
# ============================================================================================================
def _holds_device() -> bool:
    """This process has the Tenstorrent chips open (UMD keeps them, and their CHIP_IN_USE locks, until exit)."""
    try:
        fds = os.listdir("/proc/self/fd")
    except OSError:  # pragma: no cover - not Linux
        return False
    for fd in fds:
        with contextlib.suppress(OSError):
            if os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent"):
                return True
    return False


def _run_worker(out: Path, log_path: Path, timeout_s: int) -> int:
    """One worker process (this file with ``--worker``), output streamed to ``log_path``; SIGTERM, then SIGKILL after
    a grace period, on timeout (killing a device job mid-op can wedge the next open)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out", str(out)]
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ))
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return 124


@pytest.mark.timeout(PROCESSES * (WORKER_TIMEOUT_S + 300) + 600)
@torch.no_grad()
def test_prefill_determinism(tmp_path):
    """The acceptance test of the fix (module docstring): ``PROCESSES`` worker processes x ``REPEATS`` cold repeats
    of four prompts (buckets 128 / 1024 / 4096 / 8192): model-path streams + logits, serving logits and 160-token
    greedy streams must be bitwise identical within and across processes."""
    if not Path("/dev/tenstorrent").exists() or not os.listdir("/dev/tenstorrent"):
        pytest.skip("no Tenstorrent device visible")
    if _holds_device():
        pytest.skip("this pytest process already opened the Galaxy (UMD keeps CHIP_IN_USE until exit, a worker would "
                    "wait forever): run it in its own process, test_prefill_determinism.py::test_prefill_determinism")
    out_dir = Path(os.environ.get("MOTIF3_DET_OUT") or tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = []
    for p in range(PROCESSES):
        out, log_path = out_dir / f"process_{p}.json", out_dir / f"process_{p}.log"
        out.unlink(missing_ok=True)
        t0 = time.time()
        rc = _run_worker(out, log_path, WORKER_TIMEOUT_S)
        tail = log_path.read_text(errors="replace").splitlines()
        for line in tail:
            if line.startswith("[determinism]") or line.startswith("[motif3.fabric]"):
                print(f"  p{p} {line}")
        print(f"[determinism] process {p}: exit {rc} after {time.time() - t0:.0f} s (log {log_path})")
        assert rc == 0 and out.exists(), f"worker {p} failed (exit {rc}); last lines:\n" + "\n".join(tail[-40:])
        fp = json.loads(out.read_text())
        if fp.get("skip"):
            pytest.skip(fp["skip"])
        fps.append(fp)
    failures, lines = compare_fingerprints(fps)
    meta = fps[0]["meta"]
    print(f"[determinism] prefill ring_gather={meta['prefill_ring_gather']}, decode ring_gather="
          f"{meta['decode_ring_gather']}, {meta['layers']} layers, {meta['steps']} greedy tokens per prompt")
    for line in lines:
        print(line)
    def median(xs):
        return sorted(xs)[len(xs) // 2]

    for fp in fps:
        t = fp["meta"]["timings"]
        ttft = ", ".join(f"{n} {median(fp['serving'][n]['ttft_s']) * 1e3:.0f} ms" for n in fp["serving"])
        print(f"[determinism] pid {fp['meta']['pid']}: serving TTFT (median over repeats) {ttft}; decode "
              f"{1e3 * median(t['decode_s_per_step']):.1f} ms/step (host loop); program cache "
              f"{fp['meta']['program_cache']}; total {t['total_s']:.0f} s")
    (out_dir / "summary.json").write_text(json.dumps({"failures": failures, "lines": lines}, indent=1))
    assert not failures, "\n".join(failures)


def _worker_main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="one prefill-determinism worker process (test_prefill_determinism)")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from models.demos.motif3.tt.model_config import close_motif_mesh, open_motif_mesh

    mesh = open_motif_mesh()
    try:
        with torch.no_grad():
            res = determinism_session(mesh, log=lambda m: print(m, flush=True))
    finally:
        close_motif_mesh(mesh)
    Path(a.out).write_text(json.dumps(res))
    return 0


# ============================================================================================================
# device: safe == native, bitwise
# ============================================================================================================
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_ring_gather_safe_equals_native(mesh_device, device_params):
    from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, log_fabric
    from models.demos.motif3.tt.model_config import MotifTTConfig

    rep = log_fabric(mesh_device, "ring_gather_safe_equals_native")
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)
    safe = MotifCCL(mesh_device, cfg, ring_gather="safe")
    native = MotifCCL(mesh_device, cfg, ring_gather="native")
    R, C = (int(s) for s in tuple(mesh_device.shape))
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C))
    g = torch.Generator().manual_seed(0)
    tp_ring = bool(rep.get("tp_ring"))
    cases = [  # (name, per-chip shape, dtype, gather dim, expected race-prone on a TP ring)
        ("moments_prefill_4096", [1, 3, 1024, 32], ttnn.float32, 3, True),
        ("moments_prefill_128", [1, 3, 32, 32], ttnn.float32, 3, True),
        ("ar_tp_ag_bucket_512", [1, 1, 128, 512], ttnn.bfloat16, 3, True),
        ("ar_tp_ag_decode", [1, 1, 8, 512], ttnn.bfloat16, 3, True),
        ("ar_tp_ag_bucket_4096", [1, 1, 1024, 512], ttnn.bfloat16, 3, False),
    ]
    failures = []
    for name, shape, dt, dim, prone in cases:
        host = torch.randn(R * shape[0], C * shape[1], *shape[2:], generator=g) * 10
        x = ttnn.from_torch(
            host,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        assert safe._ag_race_prone(x, dim, cfg.axes.tp_axis) is (prone and tp_ring), name
        assert not native._ag_race_prone(x, dim, cfg.axes.tp_axis)
        a = safe.all_gather(x, dim, "tp")
        b = native.all_gather(x, dim, "tp")
        ta, tb = device_tensors_to_torch(a, mesh_device), device_tensors_to_torch(b, mesh_device)
        ok = torch.equal(ta, tb)
        # the gathered value is the concatenation of the row's chips' shards
        r0 = torch.cat([host[0 : shape[0], c * shape[1] : (c + 1) * shape[1]] for c in range(C)], dim=dim)
        ok_ref = torch.equal(ta[0, 0].float(), r0.to(ta.dtype).float()) if dt == ttnn.float32 else True
        print(f"[ring_gather] {name}: race-prone {prone and tp_ring}; safe == native {ok}; == host concat {ok_ref}")
        if not (ok and ok_ref):
            failures.append(name)
        for t in (a, b):
            ttnn.deallocate(t)
        if dt == ttnn.bfloat16 and shape[-2] >= 32:  # all_reduce (RS + AG) of a TP partial, safe vs native
            y = ttnn.from_torch(
                torch.randn(R, C, shape[2], 4096, generator=g),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            a, b = safe.ar_tp(y), native.ar_tp(y)
            ok = torch.equal(device_tensors_to_torch(a, mesh_device), device_tensors_to_torch(b, mesh_device))
            print(f"[ring_gather] ar_tp [{shape[2]}, 4096]: safe == native {ok}")
            if not ok:
                failures.append(f"ar_tp {name}")
            for t in (a, b, y):
                ttnn.deallocate(t)
        ttnn.deallocate(x)
    assert not failures, failures


# ============================================================================================================
# device: the race itself, isolated (no weights)
# ============================================================================================================
# (name, op, per-chip input shape, dtype, CB pages per link of the gather, race_free, stale reads of the native gather
# measured in INVESTIGATION.md §3.5 under this skew). op "ag": MotifCCL.all_gather(dim 3, tp) of the input; "ar":
# MotifCCL.ar_tp of the input (its AG half gathers [1, 1, rows, 512]).
RACE_PAYLOADS = [
    ("moments_bucket_4096", "ag", [1, 3, 1024, 32], ttnn.float32, 12, False, "33 / 300"),
    ("moments_bucket_1024", "ag", [1, 3, 256, 32], ttnn.float32, 3, False, "1852 / 3000"),
    ("moments_bucket_512", "ag", [1, 3, 128, 32], ttnn.float32, 2, False, "72 / 3000"),
    ("moe_combine_bucket_128", "ar", [1, 1, 32, 4096], ttnn.bfloat16, 1, True, "27 / 3000 (the [32, 512] gather)"),
    ("moments_bucket_128_and_decode", "ag", [1, 3, 32, 32], ttnn.float32, 1, False, "0 / 3000"),
    ("ar_tp_decode", "ar", [1, 1, 8, 4096], ttnn.bfloat16, 1, False, "27 / 3000 (the padded [32, 512] gather)"),
    # Phase C D4 (MOTIF3_MOE_DECODE_CCL=rs): the new DP-axis collective, a direct reduce-scatter on dim 1 of the folded
    # decode partial [1, 4, 32, 1024] (no gather, so no alternate routes), and the whole D4 chain (fold, RS(dp),
    # unfold, AR(tp)) of a [1, 1, 32, 4096] partial. Always asserted (pages / race_free do not apply: "-").
    ("d4_rs_dp_decode", "rs_dp_d4", [1, 4, 32, 1024], ttnn.bfloat16, None, False, "- (new)"),
    ("d4_chain_decode", "d4_chain", [1, 1, 32, 4096], ttnn.bfloat16, None, False, "- (new)"),
    # Phase D DESIGN-3 (MOTIF3_DECODE_EXPERT_MM=dualnoc): the dual-NoC expert matmul pair (tt/kernels/moe_sparse_mm.py,
    # both NoCs saturated with DRAM reads on all 120 cores) with a different active-expert count per chip (k = 0..12:
    # the chips skew), then immediately the decode AR(tp) of [1, 1, 8, 4096]: the CCL's fabric writes land while the
    # slower chips still stream on NoC0 / NoC1. Always asserted.
    ("d3_smm_ar_tp_decode", "d3_smm", [1, 1, 8, 4096], ttnn.bfloat16, None, False, "- (new)"),
]


_D4_FOLD: dict = {}  # id(mesh) -> the D4 RowFold (tt/kernels/row_fold.py) of the race test's d4_chain payload
_D3_SMM: dict = {}  # id(mesh) -> (gate_up op, down op, x, h, per-chip sparsity, weights) of the d3_smm payload


def _d3_smm_setup(mesh_device):
    """The d3_smm race payload's dual-NoC expert matmul pair: random bfp8 weights in the production layouts, M = 32
    inputs, and a per-chip sparsity with k = (chip index) % 13 active experts (so the chips finish at different times)."""
    st = _D3_SMM.get(id(mesh_device))
    if st is None:
        from models.demos.motif3.tt.kernels.moe_sparse_mm import DualNocSparseMM

        R, C = (int(s) for s in tuple(mesh_device.shape))
        g = torch.Generator().manual_seed(33)
        rep_ = ttnn.ReplicateTensorToMesh(mesh_device)

        def up(t, dt, mc=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT, mapper=rep_):
            return ttnn.from_torch(t, dtype=dt, layout=layout, device=mesh_device, memory_config=mc, mesh_mapper=mapper)

        wg = up(torch.randn(1, 12, 4096, 2560, generator=g) * 0.02, ttnn.bfloat8_b)
        wd = up(torch.randn(1, 12, 1280, 4096, generator=g) * 0.02, ttnn.bfloat8_b)
        x = up(torch.randn(1, 1, 32, 4096, generator=g), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG)
        h = up(torch.randn(1, 12, 32, 1280, generator=g), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG)
        sp = torch.zeros(R, C, 1, 12)
        for r in range(R):
            for q in range(C):
                sp[r, q, 0, : (r * C + q) % 13] = 0.5
        spd = up(sp.to(torch.bfloat16), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                 mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C)))
        st = _D3_SMM[id(mesh_device)] = (DualNocSparseMM(mesh_device, wg, kind="gate_up", out_dtype=ttnn.float32),
                                         DualNocSparseMM(mesh_device, wd, kind="down", out_dtype=ttnn.bfloat16),
                                         x, h, spd, (wg, wd))
    return st


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_ring_gather_race_isolated(mesh_device, device_params):
    """The completion race outside the model (INVESTIGATION.md §3.5, E4c / E4d): the MoE combine's collectives
    (``rs_dp`` of a replicated ``[4096, 4096]`` bf16, ``ar_tp``, ``ag_dp``; no sync) skew the chips, then a TP gather
    (``MotifCCL.all_gather``, or the AG half of ``MotifCCL.ar_tp``) of each :data:`RACE_PAYLOADS` entry and,
    immediately, a consumer (``ttnn.clone``) on the same chips. After a sync the consumer's copy must equal the output
    bitwise (the input alternates between two tensors, so a stale page shows). Every payload the default
    ``ring_gather`` reroutes must show no stale read in ``MOTIF3_DET_RACE_ITERS`` (300) iterations; the payloads it
    leaves native (single-CB-page gathers without ``race_free``: decode, and the bucket-128 / 256 PolyNorm moments) are
    counted and reported, not asserted. ``MOTIF3_DET_NATIVE_CONTROL=1`` also counts the native gather of every
    payload (reported: the race is timing dependent). ``MOTIF3_DET_RACE_ONLY=name,...`` runs only those payloads.
    Phase C D4's payloads (``d4_*``: the direct RS(dp) on dim 1 and the fold / RS / unfold / AR(tp) chain) and Phase D's
    ``d3_smm_ar_tp_decode`` (the dual-NoC expert matmuls, skewed per chip, then the decode AR(tp)) are always
    asserted."""
    from collections import Counter

    from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, log_fabric, native_ag_cb_pages_per_link
    from models.demos.motif3.tt.model_config import MotifTTConfig

    rep = log_fabric(mesh_device, "ring_gather_race_isolated")
    if not rep.get("tp_ring"):
        pytest.skip("the TP axis is not a ring in this fabric: ttnn uses no alternate routes there")
    iters = int(os.environ.get("MOTIF3_DET_RACE_ITERS", "300"))
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)
    R, C = (int(s) for s in tuple(mesh_device.shape))
    DR = ttnn.DRAM_MEMORY_CONFIG
    g = torch.Generator().manual_seed(1024)
    big = ttnn.from_torch(
        torch.randn(1, 1, 4096, 4096, generator=g),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=DR,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C))
    modes = [cfg.ring_gather] + (["native"] if os.environ.get("MOTIF3_DET_NATIVE_CONTROL", "0") == "1" else [])
    ccls = {m: MotifCCL(mesh_device, cfg, ring_gather=m) for m in dict.fromkeys(modes)}
    failures = []
    try:
        only = [n_ for n_ in os.environ.get("MOTIF3_DET_RACE_ONLY", "").split(",") if n_]  # payload names (default all)
        for name, op, (_, m, rows, width), dt, pages, race_free, measured in RACE_PAYLOADS:
            if only and name not in only:
                continue
            xs = [ttnn.from_torch(torch.randn(R, m * C, rows, width, generator=g) * 100 + 1000 * (k + 1), dtype=dt,
                                  layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=DR, mesh_mapper=mapper)
                  for k in range(2)]  # fmt: skip
            gshape = [1, m, max(rows, 32), width if op == "ag" else width // C]  # the gathered per-chip payload
            dt_name = getattr(xs[0].dtype, "name", str(xs[0].dtype).split(".")[-1])
            d4 = op in ("rs_dp_d4", "d4_chain", "d3_smm")
            assert d4 or native_ag_cb_pages_per_link(gshape, dt_name, True) == pages, name
            for mode, ccl in ccls.items():
                rerouted = d4 or mode == "safe" or (mode == "lean" and (pages > 1 or race_free))

                def step(x):
                    rs = ccl.rs_dp(big, 2)  # the MoE combine: DP-line reduce-scatter, TP all-reduce, DP all-gather
                    ar = ccl.ar_tp(rs)
                    ttnn.deallocate(rs)
                    ttnn.deallocate(ccl.ag_dp(ar, 2))
                    ttnn.deallocate(ar)
                    if op == "ag":
                        o = ccl.all_gather(x, 3, "tp", race_free=race_free)
                    elif op == "rs_dp_d4":
                        o = ccl.reduce_scatter(x, 1, "dp", memory_config=ttnn.L1_MEMORY_CONFIG)
                    elif op == "d4_chain":
                        from models.demos.motif3.tt.kernels.row_fold import RowFold

                        rf = _D4_FOLD.setdefault(id(mesh_device), RowFold(mesh_device))
                        q = rf.fold(x, 8, memory_config=ttnn.L1_MEMORY_CONFIG)
                        r_ = ccl.reduce_scatter(q, 1, "dp", memory_config=ttnn.L1_MEMORY_CONFIG)
                        ttnn.deallocate(q)
                        u = rf.unfold(r_, 8, memory_config=ttnn.L1_MEMORY_CONFIG)
                        ttnn.deallocate(r_)
                        o = ccl.ar_tp(u)
                        ttnn.deallocate(u)
                    elif op == "d3_smm":
                        gu_op, dn_op, xg, hh, spd, _ = _d3_smm_setup(mesh_device)
                        ttnn.deallocate(gu_op(xg, spd, memory_config=ttnn.L1_MEMORY_CONFIG))
                        ttnn.deallocate(dn_op(hh, spd, memory_config=ttnn.L1_MEMORY_CONFIG))
                        o = ccl.ar_tp(x)
                    else:
                        o = ccl.ar_tp(x, race_free=race_free)
                    return o, ttnn.clone(o)  # the consumer, enqueued right behind the gather

                for t in step(xs[0]):  # compile
                    ttnn.deallocate(t)
                ttnn.synchronize_device(mesh_device)
                bad, chips = 0, Counter()
                for it in range(iters):
                    o, c = step(xs[it % 2])
                    ttnn.synchronize_device(mesh_device)
                    ne = ttnn.ne(c, o)
                    cnt_t = ttnn.sum(ne, dim=2, keepdim=True)
                    cnt = device_tensors_to_torch(cnt_t, mesh_device).float().reshape(R, C, -1).sum(-1)
                    if float(cnt.sum()) > 0:
                        bad += 1
                        chips.update(f"{r},{q}" for r in range(R) for q in range(C) if float(cnt[r, q]) > 0)
                    for t in (o, c, ne, cnt_t):
                        ttnn.deallocate(t)
                print(
                    f"[ring_gather] isolated race {name} ({pages} CB page(s) per link), ring_gather={mode} "
                    f"({'safe path' if rerouted else 'native op'}): {bad} of {iters} consumers read a stale tile; "
                    f"receiving chips {dict(chips)} (native measured: {measured})"
                )
                if rerouted and bad:
                    failures.append(f"{name} ({mode}): {bad} of {iters} stale reads on {dict(chips)}")
            for t in xs:
                ttnn.deallocate(t)
    finally:
        ttnn.deallocate(big)
    assert not failures, failures


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
