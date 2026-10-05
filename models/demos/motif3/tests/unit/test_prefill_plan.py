# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tests for ``tt/prefill_plan.py`` (docs/features/FEATURES_DESIGN.md §2.2, §3.1, §3.7.2, §5.1; packed prefill:
docs/p5_t64/P5_T64_DESIGN.md §3, §7.1).

* row planning over every ``s < e <= 1100`` (and samples to 32768) for block 32 / 64 and alignment 64 / 128: no write
  below ``w0``; every real row written exactly once per call; chunk starts multiples of ``A``; ``c0 = 0`` below the
  SWA tail; spans within the cap; padding only in the own last block; the design's worked examples (A = 64);
* the production geometry of gate G9's per-bucket sp1 global chunks (``DEFAULT_SP1_GLOBAL_CHUNKS``: 128/128 at C = 128
  and C >= 2048, 64/64 at 256-1024): A = 128 = ``generator_api.DEFAULT_PREFILL_ALIGNMENT``, budget = threshold = 8064;
  every planned sp1 chunk start is a multiple of its own bucket's q and k (A = 64 would violate it); worked examples
  at A = 128; the per-bucket sp1 cost model;
* the per-chunk tables: fill tables (``-1`` for shared and pure-padding blocks), SDPA tables (0-padded, never
  ``-1``), SWA tails (the ``tail`` positions before the chunk), RoPE rows (clamped);
* writer-first ordering (same-step prefix hits), cycles and write conflicts refused;
* ``check_scheduler_config`` errors / warnings;
* an emulated paged cache driven by a vLLM-like scheduler (prefix hits on admission, full blocks cached at
  allocation time, unaligned chunk ends, several rows per call, a new lane for every chunk): every cache read sees
  valid KV, every write lands in a block the row allocated, and no completed block is ever rewritten;
* packed prefill (P5): the segment sizes (every chunk size to the span cap); ``PackSegment`` / ``PrefillPass``;
  ``segment_dag`` against ``order_prefill_requests``; the design's scenarios (§3.3) and the odd-block same-step hit of
  review edit R-E1 / gate CP-P (vi); every burst length 1 .. 1100 (cold, behind a same-step prefix hit, mixed) and
  vLLM-like traffic to 32768 checked by an oracle independent of ``segment_dag`` (every chunk exactly once; every
  block an sp1 segment reads below ``max(a, w0)`` that the call writes is written in a strictly earlier pass), by the
  pass structure (B and T powers of two within the caps, pk1 at one start) and by the pass tables (fills never write a
  shared block, dummies write nothing, each segment's tables are its chunk's re-bucketed to S: R-E12; pk1 tails
  ``shared`` iff all equal, dummies copy segment 0: R-E2); the cost model and the cheapest cut of a group (brute
  force); the shape filter; the shapes against ``MotifTTConfig.packed_prefill_shapes()``; and the paged-cache
  emulator running every pass of a vLLM-like engine with decode steps, preemption re-prefills and block eviction.

Pure torch (no ttnn; one test imports ``model_config`` to compare the packed shapes and skips without it). Run
device-hidden::

    scripts/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib -q \
        models/demos/motif3/tests/unit/test_prefill_plan.py
"""

import itertools
import json
import os
import random
import subprocess
import sys
from collections import Counter, OrderedDict, deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.motif3.tt import generator_api as api
from models.demos.motif3.tt import prefill_plan as pp

TT_METAL = Path(__file__).resolve().parents[5]
BUCKETS = api.prefill_buckets(32768)  # 128 ... 32768
GEOMETRIES = [(32, 64), (64, 64), (32, 128), (64, 128)]  # (block size, alignment A)
TAIL = pp.DEFAULT_SWA_TAIL
SPAN_BUCKETS = pp.span_buckets(32768, 8192)  # 128 ... 8192: what the serving generator compiles
PROD_A = pp.resume_alignment(64, SPAN_BUCKETS)  # 128: gate G9's per-bucket q / k
PROD_BUDGET = pp.recommended_budget(8192, PROD_A)  # 8064 = --max-num-batched-tokens = --long-prefill-token-threshold


def _plan(s, e, *, bs=64, A=64, cap=8192, tail=TAIL, cost=None, buckets=BUCKETS):
    return pp.plan_prefill_row(s, e, block_size=bs, align=A, buckets=buckets, span_cap=cap, swa_tail=tail, cost=cost)


def _chunks(plan):
    return [(c.start, c.bucket, c.path) for c in plan.chunks]


def _cdiv(a, b):
    return -(-a // b)


def _page_table(n_real, width, *, seed=0, num_blocks=100000):
    """A random-permutation page table: ``n_real`` distinct real ids (>= 1), zeros after (what the bridge sends)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(num_blocks - 1, generator=g)[:n_real] + 1
    pt = torch.zeros(width, dtype=torch.int32)
    pt[:n_real] = ids.to(torch.int32)
    return pt


# ================================================================================================================
# import rule
# ================================================================================================================
PROBE = r"""
import importlib, json, sys
for m in ("models.demos.motif3.tt.prefill_plan", "models.demos.motif3.tt.generator_api"):
    importlib.import_module(m)
demos = sorted(m for m in sys.modules if m.startswith("models.demos.") and not m.startswith("models.demos.motif3"))
heavy = sorted(m for m in ("ttnn", "vllm", "transformers", "huggingface_hub", "safetensors") if m in sys.modules)
mods = sorted(m for m in sys.modules if m.startswith("models.demos.motif3"))
print(json.dumps({"demos": demos, "heavy": heavy, "motif3": mods}))
"""


def test_import_rule_torch_only():
    """prefill_plan (and the contract) import stdlib + torch only: the bridge (EngineCore, before any mesh), its host
    tests and model_config import it. Fresh interpreter, so other tests' imports cannot hide a leak."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TT_METAL) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=str(TT_METAL), env=env, capture_output=True, text=True, timeout=300
    )
    assert res.returncode == 0, res.stderr[-4000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["demos"] == [] and out["heavy"] == [], out
    assert set(out["motif3"]) <= {
        "models.demos.motif3",
        "models.demos.motif3.tt",
        "models.demos.motif3.tt.prefill_plan",
        "models.demos.motif3.tt.generator_api",
    }, out["motif3"]


# ================================================================================================================
# the design's worked examples (§3.1 table, §3.1 item 3, Appendix A)
# ================================================================================================================
@pytest.mark.parametrize(
    "s, e, w0, c0, chunks, written",
    [
        # cold 1000
        (0, 1000, 0, 0, [(0, 1024, "sp0")], range(0, 16)),
        # 1000-token prompt, 640 cached
        (640, 1000, 640, 640, [(640, 512, "sp1")], range(10, 16)),
        # hit of 64 (one block): c0 drops to 0 (< the 128-row SWA tail); block 0 is shared, never written
        (64, 900, 64, 0, [(0, 1024, "sp0")], range(1, 15)),
        # continuation after an unaligned chunk end 1348: block 21 is own and partial, rows 1344-1347 rewritten
        (1348, 3000, 1344, 1344, [(1344, 2048, "sp1")], range(21, 47)),
        # preempted: 1200 prompt + 300 generated, 1472 cached
        (1472, 1500, 1472, 1472, [(1472, 128, "sp1")], range(23, 24)),
        # 16,736 cold, chunking off: the generator splits internally (D8)
        (0, 16736, 0, 0, [(0, 8192, "sp0"), (8192, 8192, "sp1"), (16384, 512, "sp1")], range(0, 262)),
        # 2200 -> 2048 + 152 (bucket 256): 2.2 s instead of 2.91 s
        (0, 2200, 0, 0, [(0, 2048, "sp0"), (2048, 256, "sp1")], range(0, 35)),
        # Appendix A: 9000-token prompt, 6976 cached (109 full blocks)
        (6976, 9000, 6976, 6976, [(6976, 2048, "sp1")], range(109, 141)),
    ],
)
def test_design_worked_examples(s, e, w0, c0, chunks, written):
    p = _plan(s, e)
    assert (p.w0, p.c0) == (w0, c0)
    assert _chunks(p) == chunks
    assert list(p.write_blocks) == list(written)
    assert p.chunks[-1].last and not any(c.last for c in p.chunks[:-1])
    assert p.chunks[-1].end == e and p.chunks[-1].head_row == e - 1 - p.chunks[-1].start


def test_alignment_128_variant():
    """§5.2: a hit of 960 with A = 128 computes from c0 = 896 and never rewrites the shared block 14."""
    p = _plan(960, 4096, A=128)
    assert (p.w0, p.c0, p.recompute) == (960, 896, 64)
    assert _chunks(p) == [(896, 4096, "sp1")]
    pt = _page_table(64, 640, seed=3)
    fill = pp.fill_table(pt, p.chunks[0], p.w0, 64)
    assert int(fill[0]) == -1  # logical block 14 = positions 896-959: shared, recomputed but not written
    assert torch.equal(fill[1:50], pt[15:64])  # blocks 15-63 hold positions 960-4095: written
    assert bool((fill[50:] == -1).all())  # blocks 64-77 are pure bucket padding (positions >= 4096)


def test_appendix_a_tables():
    """Appendix A step 3: fill = blocks 109-140, SDPA = blocks 0-140 then 0 to 640 entries, tail = blocks 107-108."""
    p = _plan(6976, 9000)
    c = p.chunks[0]
    pt = _page_table(141, 512, seed=1)
    t = pp.chunk_tables(pt, p, c, sdpa_width=pp.sdpa_table_width(32768, 8192, 64), max_positions=32768)
    assert torch.equal(t.fill, pt[109:141])
    assert t.sdpa.shape == (640,) and torch.equal(t.sdpa[:141], pt[:141]) and bool((t.sdpa[141:] == 0).all())
    assert torch.equal(t.tail, pt[107:109])
    assert torch.equal(t.rope, torch.arange(6976, 9024, dtype=torch.int32))
    assert c.head_row == 2023


def test_generalized_head_split():
    """The planner applies the head split recursively when one split is not enough (cost table §3.1)."""
    assert _chunks(_plan(0, 6200)) == [(0, 4096, "sp0"), (4096, 2048, "sp1"), (6144, 128, "sp1")]
    assert _chunks(_plan(0, 7000)) == [(0, 4096, "sp0"), (4096, 2048, "sp1"), (6144, 1024, "sp1")]
    assert _chunks(_plan(0, 8192)) == [(0, 8192, "sp0")]  # exactly a bucket: one chunk
    assert _chunks(_plan(0, 8193)) == [(0, 8192, "sp0"), (8192, 128, "sp1")]
    assert _chunks(_plan(0, 4096)) == [(0, 4096, "sp0")]
    assert _chunks(_plan(0, 3500)) == [(0, 4096, "sp0")]  # 2048 + 1452 (2048) costs more than one 4096 chunk
    # span cap 32768: no forced split, a full 32K prompt runs single-shot (the draft-1 bucket); the §3.1 head split
    # still removes the 16K -> 32K bucket cliff (16384 + 512: ~12.3 s instead of 23.4 s)
    assert _chunks(_plan(0, 32768, cap=32768)) == [(0, 32768, "sp0")]
    assert _chunks(_plan(0, 32000, cap=32768)) == [(0, 32768, "sp0")]
    assert _chunks(_plan(0, 16736, cap=32768)) == [(0, 16384, "sp0"), (16384, 512, "sp1")]


# ================================================================================================================
# exhaustive row planning (pure integer checks)
# ================================================================================================================
def _expected_floors(s, bs, A, tail):
    w0 = s // bs * bs
    c0 = s // A * A
    return w0, (0 if c0 < tail else c0)


def _check_row(p, s, e, bs, A, cap, tail, usable, cost):
    """Every planning invariant of §3.1 for one row (no torch)."""
    w0, c0 = _expected_floors(s, bs, A, tail)
    assert (p.start, p.end, p.w0, p.c0, p.block_size, p.align) == (s, e, w0, c0, bs, A)
    assert c0 <= w0 <= s and (c0 == 0 or c0 >= tail)
    ch = p.chunks
    pos, expect_block = c0, w0 // bs
    for k, c in enumerate(ch):
        a, C = c.start, c.bucket
        assert a == pos and a % A == 0 and a % bs == 0, (s, e, k)
        assert C in usable and C <= cap
        assert c.end == min(e, a + C) and c.last == (k == len(ch) - 1)
        assert c.path == ("sp0" if a == 0 else "sp1") and (k == 0 or c.path == "sp1")
        if c.last:
            assert c.end == e
            assert C == min(b for b in usable if b >= e - a), "the last chunk pads to the smallest bucket"
            assert 0 <= c.head_row < C
        else:
            assert c.end == a + C, "only the last chunk is padded"
        if c.is_sp1:
            assert a >= tail, "an sp1 chunk needs the whole SWA tail before it"
            if A % PROD_A == 0:  # the sp1 global op of this bucket (G9 per-bucket q / k) accepts the start
                q, kc = pp.sp1_global_qk(C)
                assert a % q == 0 and a % kc == 0, (s, e, a, C, q, kc)
        # writes of this chunk (fill-table semantics): logical blocks overlapping [w0, c.end)
        lo, hi = max(a // bs, w0 // bs), min((a + C) // bs, _cdiv(c.end, bs))
        if hi > lo:
            assert lo == expect_block, "written blocks are contiguous and written exactly once per call"
            expect_block = hi
            real_lo, real_hi = max(a, lo * bs), min(c.end, hi * bs)
            assert (real_lo, real_hi) == (max(a, w0), c.end), "real rows [max(a, w0), end) written, none below w0"
            pad_hi = min(a + C, hi * bs)
            if pad_hi > c.end:  # padding rows written: only inside the own last block
                assert c.last and _cdiv(e, bs) - 1 == hi - 1 and c.end >= (hi - 1) * bs
        pos = c.end
    assert pos == e and expect_block == _cdiv(e, bs)
    if e - c0 <= cap:  # the plan is never worse than one padded chunk (cost model)
        single = cost(min(b for b in usable if b >= e - c0), c0)
        assert pp.plan_cost(p, cost) <= single + 1e-12


@pytest.mark.parametrize("bs, A", GEOMETRIES)
def test_plan_exhaustive_to_1100(bs, A):
    """Every 0 <= s < e <= 1100 at the serving span cap (8192)."""
    cost = pp.prefill_cost_model()
    usable = set(b for b in BUCKETS if b <= 8192)
    for s in range(0, 1100):
        for e in range(s + 1, 1101):
            p = _plan(s, e, bs=bs, A=A, cost=cost)
            _check_row(p, s, e, bs, A, 8192, TAIL, usable, cost)


@pytest.mark.parametrize("bs, A", GEOMETRIES)
def test_plan_exhaustive_small_cap(bs, A):
    """Every 0 <= s < e <= 700 with a 256-row span cap: rows become many chunks (internal splits, D8)."""
    cost = pp.prefill_cost_model()
    usable = {128, 256}
    many = 0
    for s in range(0, 700):
        for e in range(s + 1, 701):
            p = _plan(s, e, bs=bs, A=A, cap=256, cost=cost)
            _check_row(p, s, e, bs, A, 256, TAIL, usable, cost)
            many += len(p.chunks) > 2
    assert many > 0


@pytest.mark.parametrize("bs, A", GEOMETRIES)
@pytest.mark.parametrize("cap", [8192, 32768, 1024])
def test_plan_sampled_to_32768(bs, A, cap):
    """Random rows up to max_model_len 32768 and the edges (spans exactly a bucket / the cap, +- 1, the last
    position)."""
    rng = random.Random(1000 * bs + A + cap)
    cost = pp.prefill_cost_model()
    usable = set(b for b in BUCKETS if b <= cap)
    cases = [(0, 32768), (32767, 32768), (32704, 32768), (8191, 16384), (8192, 16385), (127, 8255), (128, 8256)]
    cases += [(0, cap), (0, cap + 1), (0, 2 * cap), (A, A + cap), (A + 1, A + cap + 1)]
    for _ in range(3000):
        e = rng.randint(1, 32768)
        cases.append((rng.randint(0, e - 1), e))
        s = rng.randrange(0, 32768 // bs) * bs  # prefix hits are block multiples
        if s < 32768:
            cases.append((s, rng.randint(s + 1, min(32768, s + 9000))))
    for s, e in cases:
        if not 0 <= s < e <= 32768:
            continue
        _check_row(_plan(s, e, bs=bs, A=A, cap=cap, cost=cost), s, e, bs, A, cap, TAIL, usable, cost)


def test_lone_request_chunks_need_no_recompute():
    """§1.1: with the budget 8192 - A, a lone long prompt's chunk ends are multiples of A, so every continuation
    starts aligned (no recompute) and fits one 8192 bucket; an unaligned (mixed-step) end recomputes < A rows and
    still fits. A continuation start below the 128-row tail recomputes from 0 instead (a rare tiny first chunk)."""
    for A in (64, 128):
        budget = pp.recommended_budget(8192, A)
        assert budget == 8192 - A
        for k in range(1, 4):
            s = k * budget
            p = _plan(s, min(s + budget, 32768), A=A)
            assert p.recompute == 0 and len(p.chunks) == 1 and p.chunks[0].bucket == 8192
        for s in range(128, 4000, 37):  # unaligned continuation
            p = _plan(s, s + budget, A=A)
            assert p.recompute < A and _chunks(p) == [(s // A * A, 8192, "sp1")]
    p = _plan(100, 100 + 8128)  # first chunk ended at 100 (< tail): recompute from 0, one extra chunk
    assert p.c0 == 0 and _chunks(p) == [(0, 8192, "sp0"), (8192, 128, "sp1")]
    p = _plan(100, 100 + PROD_BUDGET, A=PROD_A)  # the same at the production geometry: one 8192 chunk from 0
    assert p.c0 == 0 and _chunks(p) == [(0, 8192, "sp0")]


def test_chunk_budget_target_a1a():
    """OPTIMIZATION_PLAN.md §3.3 A1a: ``recommended_budget(cap, A, target)`` makes a requested budget alignment-aware
    (down to a multiple of A, at most cap - A; no target = cap - A = 8064, the code default). At the production geometry
    the A1a budget 4096 is accepted without warnings, and a lone prompt then runs one 4096 chunk per vLLM step with no
    recompute: sp0 4096, then sp1 4096 (and a smaller tail bucket), all shapes of the 8192-cap compile set."""
    rb = pp.recommended_budget
    assert rb(8192, 128) == rb(8192, 128, None) == 8064 == PROD_BUDGET
    assert rb(8192, 128, 4096) == 4096 and rb(8192, 128, 4000) == 3968 and rb(8192, 128, 8192) == 8064
    assert rb(8192, 128, 32768) == 8064 and rb(8192, 128, 128) == 128 and rb(8192, 64, 4100) == 4096
    assert rb(4096, 128, 4096) == 3968  # a smaller span cap still caps the budget at cap - A
    with pytest.raises(ValueError, match="below the resume alignment"):
        rb(8192, 128, 100)
    budget = rb(api.DEFAULT_PREFILL_SPAN_CAP, api.DEFAULT_PREFILL_ALIGNMENT, 4096)
    assert pp.check_scheduler_config(
        chunked=True, budget=budget, threshold=budget, align=PROD_A, span_cap=8192, prefix_caching=True,
        prefix_match_unit=None, block_size=64, kv_replicated=True,
    ) == []  # fmt: skip
    for L in (4096, 10000, 16384, 32768 - 64):
        for k in range(_cdiv(L, budget)):
            s, e = k * budget, min((k + 1) * budget, L)
            p = _plan(s, e, A=PROD_A)
            assert p.recompute == 0 and p.chunks[0].start == s, (L, k, _chunks(p))
            assert p.chunks[0].path == (pp.SP0 if s == 0 else pp.SP1)
            assert all(c.bucket in SPAN_BUCKETS for c in p.chunks)
            if e - s == 4096:
                assert _chunks(p) == [(s, 4096, pp.SP0 if s == 0 else pp.SP1)]


# ================================================================================================================
# production geometry: gate G9's per-bucket sp1 global chunks (lead decision F5)
# ================================================================================================================
def test_production_geometry_g9_per_bucket():
    """128/128 at C = 128 and C >= 2048, 64/64 at 256-1024 (GATES_RESULTS §12.2) -> A = 128 for block 32 and 64 ->
    budget = threshold = 8192 - 128 = 8064; the bridge's pre-generator default agrees (it sets FEATURE_VLLM_ARGS)."""
    want = {128: (128, 128), 256: (64, 64), 512: (64, 64), 1024: (64, 64), 2048: (128, 128), 4096: (128, 128)}
    want.update({8192: (128, 128), 16384: (128, 128), 32768: (128, 128), 6144: (128, 128)})
    assert {c: pp.sp1_global_qk(c) for c in want} == want
    assert PROD_A == 128 == api.DEFAULT_PREFILL_ALIGNMENT
    assert pp.resume_alignment(32, SPAN_BUCKETS) == 128 == pp.resume_alignment(64, pp.span_buckets(32768, 32768))
    assert pp.resume_alignment(64, pp.span_buckets(4096, 8192)) == 128  # any span cap: bucket 128 is always there
    assert pp.resume_alignment(64, SPAN_BUCKETS, ((32768, (64, 64)),)) == 64  # the pre-G9 all-64/64 table
    assert PROD_BUDGET == 8064 == pp.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, api.DEFAULT_PREFILL_ALIGNMENT)
    assert pp.check_scheduler_config(
        chunked=True, budget=PROD_BUDGET, threshold=PROD_BUDGET, align=PROD_A, span_cap=8192, prefix_caching=True,
        prefix_match_unit=None, block_size=64, kv_replicated=True,
    ) == []  # fmt: skip
    # a bf16 latent cache keeps 64/64 everywhere (A = 64): at 128/128 its static CBs overrun L1_SMALL (CCL semaphores)
    assert pp.sp1_global_chunk_table() == pp.sp1_global_chunk_table("bfp8") == pp.DEFAULT_SP1_GLOBAL_CHUNKS
    bf16 = pp.sp1_global_chunk_table("bf16")
    assert bf16 == pp.SP1_GLOBAL_CHUNKS_BF16_KV and all(pp.sp1_global_qk(c, bf16) == (64, 64) for c in want)
    assert pp.resume_alignment(64, SPAN_BUCKETS, bf16) == 64 == pp.resume_alignment(32, SPAN_BUCKETS, bf16)
    assert pp.check_scheduler_config(
        chunked=True, budget=PROD_BUDGET, threshold=PROD_BUDGET, align=64, span_cap=8192, prefix_caching=True,
        prefix_match_unit=None, block_size=64, kv_replicated=True,
    ) == []  # fmt: skip  # the production budget is clean for a bf16-cache server too
    with pytest.raises(ValueError, match="KV cache dtype"):
        pp.sp1_global_chunk_table("bfp4")
    for bad in (100, 0, -128):  # q = 128 does not divide 100; non-positive buckets
        with pytest.raises(ValueError):
            pp.sp1_global_qk(bad)
    with pytest.raises(ValueError, match="no sp1 global chunk entry"):
        pp.sp1_global_qk(65536)
    with pytest.raises(ValueError):
        pp.sp1_global_qk(256, ((1024, (0, 64)),))
    with pytest.raises(ValueError):
        pp.resume_alignment(0, SPAN_BUCKETS)


def test_per_bucket_op_needs_a_128():
    """Why A = 128: with A = 64 the planner emits sp1 chunks whose start is not a multiple of their own bucket's q / k
    (the 128 and >= 2048 buckets), which the op would floor silently (G9 negative control); with A = 128 none (the
    exhaustive tests check every row through ``_check_row``)."""
    bad = []
    for s in range(128, 4200, 64):
        for e in (s + 1, s + 100, s + 1500, s + 3000):
            for c in _plan(s, e, A=64).chunks:
                q, k = pp.sp1_global_qk(c.bucket)
                if c.is_sp1 and (c.start % q or c.start % k):
                    bad.append((s, e, c.start, c.bucket))
    assert bad and (192, 193, 192, 128) in bad, bad[:5]
    assert all(
        c.start % pp.sp1_global_qk(c.bucket)[0] == 0
        for s in range(128, 4200, 64)
        for e in (s + 1, s + 100, s + 1500, s + 3000)
        for c in _plan(s, e, A=PROD_A).chunks
    )


def test_plan_production_vllm_steps():
    """Every prompt length (stride 37) to 32768 as a lone request in vLLM steps of 8064, cold and behind a random
    block-aligned prefix hit: every step's span ``e - c0`` fits one 8192 bucket (no forced split), a cold request never
    recomputes a row (its chunk ends are multiples of A = 128), a hit at an odd multiple of 64 recomputes 64 rows per
    step, and every invariant of ``_check_row`` holds (including the per-bucket op alignment)."""
    cost = pp.prefill_cost_model()
    usable = set(SPAN_BUCKETS)
    rng = random.Random(8064)
    steps = recomputed = 0
    for L in range(1, 32769, 37):
        for hit in sorted({0, rng.randrange(0, L) // 64 * 64}):
            s = hit
            while s < L:
                e = min(s + PROD_BUDGET, L)
                p = _plan(s, e, A=PROD_A, cost=cost)
                _check_row(p, s, e, 64, PROD_A, 8192, TAIL, usable, cost)
                assert e - p.c0 <= 8192, (L, hit, s, e)
                assert p.recompute == s % PROD_A and (hit or p.recompute == 0), (L, hit, s)
                steps += 1
                recomputed += p.recompute
                s = e
    assert steps > 2000 and recomputed > 0


@pytest.mark.parametrize(
    "s, e, w0, c0, chunks",
    [
        # 30,832-token needle prompt, lone, in vLLM steps of 8064 (FEATURES_RESULTS §3.5 at the new budget): every
        # continuation starts aligned (no recompute) and fits one 8192 bucket; the 6640-row tail splits by cost
        (0, 8064, 0, 0, [(0, 8192, "sp0")]),
        (8064, 16128, 8064, 8064, [(8064, 8192, "sp1")]),
        (16128, 24192, 16128, 16128, [(16128, 8192, "sp1")]),
        (24192, 30832, 24192, 24192, [(24192, 4096, "sp1"), (28288, 2048, "sp1"), (30336, 512, "sp1")]),
        # the same prompt with chunking off (one call): the generator's internal split (D8)
        (0, 30832, 0, 0, [(0, 8192, "sp0"), (8192, 8192, "sp1"), (16384, 8192, "sp1"), (24576, 4096, "sp1"),
                          (28672, 2048, "sp1"), (30720, 128, "sp1")]),
        # FEATURES_RESULTS §3.3: a cold 6,196-token row, then the same prompt with 6,144 tokens hit
        (0, 6196, 0, 0, [(0, 4096, "sp0"), (4096, 2048, "sp1"), (6144, 128, "sp1")]),
        (6144, 6196, 6144, 6144, [(6144, 128, "sp1")]),
        # a 64-granular hit (35 blocks, §3.4 multi-turn): c0 = 2176 < w0 = 2240, 64 rows recomputed, block 34 skipped
        (2240, 2304, 2240, 2176, [(2176, 128, "sp1")]),
        # a continuation after an unaligned (mixed-step) vLLM chunk end: < 128 rows recomputed, still one bucket
        (8100, 16164, 8064, 8064, [(8064, 8192, "sp1")]),
    ],
)  # fmt: skip
def test_production_worked_examples(s, e, w0, c0, chunks):
    p = _plan(s, e, A=PROD_A)
    assert (p.w0, p.c0) == (w0, c0) and _chunks(p) == chunks
    for c in p.chunks:
        q, k = pp.sp1_global_qk(c.bucket)
        assert not c.is_sp1 or (c.start % q == 0 and c.start % k == 0)
    if (s, e) == (2240, 2304):
        pt = _page_table(36, 512, seed=11)
        fill = pp.fill_table(pt, p.chunks[0], p.w0, 64)
        assert int(fill[0]) == -1 and int(fill[1]) == int(pt[35])  # block 34 shared (read only), block 35 own


def test_custom_cost_models():
    calls = []

    def never_split(bucket, start):
        calls.append((bucket, start))
        return 1.0  # every chunk costs the same: a single chunk always wins

    assert _chunks(_plan(0, 6200, cost=never_split)) == [(0, 8192, "sp0")]
    assert (8192, 0) in calls and (4096, 0) in calls and any(st > 0 for _, st in calls)  # sp1 candidates priced
    linear = {b: b / 1000.0 for b in BUCKETS}  # no dispatch floor: splitting off padding always pays
    assert _chunks(_plan(0, 6200, cost=linear)) == [(0, 4096, "sp0"), (4096, 2048, "sp1"), (6144, 128, "sp1")]
    assert _chunks(_plan(0, 2200, cost=linear)) == [(0, 2048, "sp0"), (2048, 256, "sp1")]
    flat = {b: 0.64 for b in BUCKETS}
    assert _chunks(_plan(0, 2200, cost=flat)) == [(0, 4096, "sp0")]  # ties keep fewer chunks
    with pytest.raises(TypeError):
        _plan(0, 100, cost=3.0)


def test_cost_model_table_and_prefix_term():
    m = pp.prefill_cost_model()  # gate G9's per-bucket sp1 model (default)
    assert m(128, 0) == pytest.approx(0.642) and m(32768, 0) == pytest.approx(23.396)
    assert m(2048, 24576) == pytest.approx(1.55 + 14 * 0.92e-6 * 24576)
    assert m(128, 24448) == pytest.approx(0.642 + 14 * 0.43e-6 * 24448)
    # per (row, key) over 14 layers: K / V streaming makes short chunks behind long prefixes dear (G9 §12.2)
    per_row_key = {c: pp.sp1_prefix_cost(c, 10000) / (c * 10000) for c in pp.DEFAULT_SP1_GLOBAL_COST}
    assert per_row_key[128] == pytest.approx(4.70e-8, rel=0.01) and per_row_key[512] == pytest.approx(1.42e-8, rel=0.01)
    assert all(5e-9 < per_row_key[c] < 8e-9 for c in (1024, 2048, 4096, 8192))
    assert pp.sp1_prefix_cost(4096, 0) == 0.0 and pp.sp1_prefix_cost(4096, -64) == 0.0
    assert pp.sp1_prefix_cost(6144, 1000) == pytest.approx(14 * (1.88e-6 + 3.07e-6) / 2 * 1000)  # interpolated
    assert pp.sp1_prefix_cost(16384, 1000) == pytest.approx(14 * 3.07e-6 * 2 * 1000)  # per token above 8192
    assert pp.sp1_prefix_cost(2048, 1000, global_layers=1) == pytest.approx(0.92e-6 * 1000)
    # the pre-G9 single constant, explicitly
    old = pp.prefill_cost_model(sp1_s_per_row_key=pp.DEFAULT_SP1_ATTN_S_PER_ROW_KEY)
    assert old(2048, 24576) == pytest.approx(1.55 + pp.DEFAULT_SP1_ATTN_S_PER_ROW_KEY * 2048 * 24576)
    assert pp.prefill_cost_model(sp1_s_per_row_key=0.0)(2048, 24576) == pytest.approx(1.55)
    custom = pp.prefill_cost_model({128: 1.0}, sp1_global_cost={128: (0.0, 1e-3)}, global_layers=2)
    assert custom(128, 10) == pytest.approx(1.0 + 2 * 1e-3 * 10)
    for bad in (dict(sp1_s_per_row_key=-1.0), dict(sp1_global_cost={}), dict(sp1_global_cost={128: (0.0, -1.0)})):
        with pytest.raises(ValueError):
            pp.prefill_cost_model(**bad)
    with pytest.raises(ValueError):
        pp.prefill_cost_model(global_layers=-1)
    t = {128: 1.0, 512: 3.0}
    assert pp.table_cost(t, 256) == pytest.approx(5.0 / 3.0)  # interpolated
    assert pp.table_cost(t, 64) == 1.0 and pp.table_cost(t, 1024) == pytest.approx(6.0)  # clamp / per-token extrap
    assert pp.table_cost(pp.DEFAULT_PREFILL_COST_TABLE, 6144) == pytest.approx((2.912 + 5.8) / 2)
    with pytest.raises(ValueError):
        pp.table_cost({}, 128)


def test_plan_argument_validation():
    with pytest.raises(ValueError, match="start < end"):
        _plan(5, 5)
    with pytest.raises(ValueError, match="start < end"):
        _plan(-1, 5)
    with pytest.raises(ValueError, match="multiple of the block size"):
        _plan(0, 100, bs=64, A=96)
    with pytest.raises(ValueError, match="span_cap"):
        _plan(0, 100, cap=5000)
    with pytest.raises(ValueError, match="multiples of the alignment"):
        _plan(0, 100, A=256)  # bucket 128 is not a multiple of 256
    with pytest.raises(ValueError, match="strictly increasing"):
        _plan(0, 100, buckets=(256, 128))
    with pytest.raises(ValueError, match="swa_tail"):
        _plan(0, 100, tail=100)
    # no SWA (tail 0): c0 is the plain alignment floor
    p = _plan(64, 300, tail=0)
    assert p.c0 == 64 and _chunks(p) == [(64, 256, "sp1")]


# ================================================================================================================
# per-chunk tables
# ================================================================================================================
def _check_tables(p, pt, bs, cap, width, P=32768, tail=TAIL):
    for c in p.chunks:
        t = pp.chunk_tables(pt, p, c, sdpa_width=width, max_positions=P, swa_tail=tail)
        a, C = c.start, c.bucket
        # fill: -1 exactly for blocks wholly below w0 (shared) or wholly at / past the chunk end (padding)
        assert t.fill.dtype == torch.int32 and t.fill.shape == (C // bs,)
        for j in range(C // bs):
            L = a // bs + j
            lo, hi = L * bs, (L + 1) * bs
            if hi <= p.w0 or lo >= c.end:
                assert int(t.fill[j]) == -1
            else:
                assert int(t.fill[j]) == int(pt[L]) >= 1
        assert not bool((t.fill == 0).any()), "a fill table never targets the null block"
        assert torch.equal(t.rope, torch.arange(a, a + C, dtype=torch.int32).clamp(max=P - 1))
        assert bool((t.rope[: c.rows] == torch.arange(a, c.end, dtype=torch.int32)).all())
        if c.is_sp1:
            n = _cdiv(c.end, bs)
            assert t.sdpa.dtype == torch.int32 and t.sdpa.shape == (width,) and width % 8 == 0
            assert bool((t.sdpa >= 0).all()), "SDPA tables are never -1"
            assert torch.equal(t.sdpa[:n], pt[:n]) and bool((t.sdpa[n:] == 0).all())
            assert width * bs >= a + C, "padded query rows read up to a + C through valid (or null) entries"
            assert t.tail.shape == (tail // bs,)
            for q in range(a - tail, a):  # the tail = the cache rows of positions [a - tail, a)
                assert int(t.tail[(q - (a - tail)) // bs]) == int(pt[q // bs])
        else:
            assert t.sdpa is None and t.tail is None


@pytest.mark.parametrize("bs, A", GEOMETRIES)
def test_tables_sampled(bs, A):
    rng = random.Random(bs * 7 + A)
    width = pp.sdpa_table_width(32768, 8192, bs)
    assert width == {64: 640, 32: 1280}[bs]
    W = 32768 // bs
    cases = [(s, e) for s in range(0, 1100, 13) for e in (s + 1, s + 63, s + 64, s + 65, s + 200, s + 1000)]
    cases += [(rng.randint(0, 30000), 0) for _ in range(300)]
    for i, (s, e) in enumerate(cases):
        if e == 0:
            e = rng.randint(s + 1, min(32768, s + 17000))
        if e > 32768:
            continue
        p = _plan(s, e, bs=bs, A=A)
        pt = _page_table(_cdiv(e, bs), W, seed=i)
        _check_tables(p, pt, bs, 8192, width)


def test_table_rules_and_errors():
    p = _plan(1348, 3000)
    c = p.chunks[0]
    pt = _page_table(47, 512, seed=5)
    fill = pp.fill_table(pt, c, p.w0, 64)
    assert torch.equal(fill[:26], pt[21:47]) and bool((fill[26:] == -1).all())  # 26 real blocks, 6 of padding
    with pytest.raises(ValueError, match="null block"):
        bad = pt.clone()
        bad[30] = 0
        pp.fill_table(bad, c, p.w0, 64)
    with pytest.raises(ValueError, match="entries"):
        pp.fill_table(pt[:40], c, p.w0, 64)
    with pytest.raises(ValueError, match="multiple of 8"):
        pp.sdpa_table(pt, 3000, 64, 644)
    with pytest.raises(ValueError, match="width"):
        pp.sdpa_table(pt, 3000, 64, 40)
    with pytest.raises(ValueError, match="sp1"):
        pp.tail_blocks(pt, _plan(0, 100).chunks[0], 64)
    with pytest.raises(ValueError, match="RoPE"):
        pp.rope_positions(c, 2000)
    with pytest.raises(TypeError):
        pp.fill_table(pt.float(), c, p.w0, 64)
    with pytest.raises(ValueError, match="cover"):
        pp.chunk_tables(pt, p, c, sdpa_width=8, max_positions=32768)
    with pytest.raises(ValueError, match="belong"):
        pp.chunk_tables(pt, p, _plan(0, 100).chunks[0], sdpa_width=640, max_positions=32768)
    # RoPE rows of padding clamp at the table end; real rows never do
    last = _plan(32704, 32768).chunks[0]
    r = pp.rope_positions(last, 32768)
    assert int(r[63]) == 32767 and int(r[64]) == 32767 and int(r[-1]) == 32767
    assert pp.span_buckets(32768, 8192) == (128, 256, 512, 1024, 2048, 4096, 8192)
    assert pp.span_buckets(6144, 8192) == (128, 256, 512, 1024, 2048, 4096, 6144)
    assert pp.span_buckets(4096, 32768)[-1] == 4096


# ================================================================================================================
# writer-first ordering
# ================================================================================================================
def _req(start, end, blocks, *, width=512, lane=0):
    pt = torch.zeros(width, dtype=torch.int32)
    pt[: len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return api.PrefillRequest(lane=lane, tokens=torch.zeros(end, dtype=torch.int32), page_table=pt, start=start)


def _order(reqs, bs=64):
    plans = [_plan(r.start, r.seq_len, bs=bs) for r in reqs]
    return pp.order_prefill_requests(reqs, plans, bs), plans


def test_order_independent_rows_keep_input_order():
    reqs = [_req(0, 300, [1, 2, 3, 4, 5], lane=0), _req(640, 900, list(range(10, 25)), lane=1), _req(0, 64, [30])]
    assert _order(reqs)[0] == [0, 1, 2]


def test_order_same_step_hit():
    """vLLM admits A, caches A's full blocks at allocation, then admits B with a hit on them in the same step: B reads
    blocks A writes in this call. Whatever the row order, A runs first."""
    a_blocks = list(range(100, 116))  # A: 1000 new tokens, blocks 100-115 (15 full + 1 partial)
    a = _req(0, 1000, a_blocks, lane=3)
    b = _req(960, 1300, a_blocks[:15] + [200, 201, 202, 203, 204, 205], lane=9)  # B hits A's 15 full blocks
    order, plans = _order([b, a])
    assert plans[0].has_sp1 and order == [1, 0]
    assert _order([a, b])[0] == [0, 1]
    # a shared prompt-only prefix written in an EARLIER call creates no dependency
    c = _req(960, 1100, a_blocks[:15] + [300, 301, 302], lane=4)
    d = _req(0, 200, [400, 401, 402, 403], lane=5)
    assert _order([c, d])[0] == [0, 1]


def test_order_sp0_reader_has_no_dependency():
    """A hit shorter than the 128-row tail recomputes from 0 (sp0): it reads no cache, so it does not wait for the
    writer of its shared block (it never writes it either)."""
    w = _req(0, 700, list(range(1, 12)))
    r = _req(64, 300, [1, 50, 51, 52, 53], lane=1)  # hit of one block (block id 1, written by w in this call)
    order, plans = _order([r, w])
    assert not plans[0].has_sp1 and order == [0, 1]
    # ... but a long row with that hit gets sp1 continuations, which read block 1: it must wait
    r2 = _req(64, 9000, [1] + list(range(600, 740)), lane=2)
    order, plans = _order([r2, w])
    assert plans[0].paths == ("sp0", "sp1") and order == [1, 0]


def test_order_chain_and_stability():
    a = _req(0, 640, list(range(1, 11)), lane=0)  # writes 1-10
    b = _req(640, 1280, list(range(1, 11)) + list(range(11, 21)), lane=1)  # reads 1-10, writes 11-20
    c = _req(1280, 1500, list(range(1, 21)) + [21, 22, 23, 24], lane=2)  # reads 1-20 (a and b)
    x = _req(0, 100, [90, 91], lane=3)  # independent
    order, _ = _order([c, x, b, a])
    assert order == [1, 3, 2, 0]  # x first (smallest ready index), then a -> b -> c


def test_order_errors():
    a = _req(0, 640, list(range(1, 11)))
    b = _req(0, 640, list(range(5, 15)), lane=1)
    with pytest.raises(ValueError, match="both write"):
        _order([a, b])
    with pytest.raises(ValueError, match="twice"):
        _order([_req(0, 640, [1, 2, 3, 4, 5, 6, 7, 8, 9, 1])])
    with pytest.raises(ValueError, match="block id"):
        _order([_req(0, 640, [1, 2, 3, 0, 5, 6, 7, 8, 9, 10])])
    # cycle: each reads (as its cached prefix) what the other writes -- impossible under vLLM
    p = _req(640, 1280, list(range(11, 21)) + list(range(1, 11)))  # reads 11-20, writes 1-10
    q = _req(640, 1280, list(range(1, 11)) + list(range(11, 21)), lane=1)  # reads 1-10, writes 11-20
    with pytest.raises(ValueError, match="cycle"):
        _order([p, q])
    plans = [_plan(0, 640)]
    with pytest.raises(ValueError, match="does not match"):
        pp.order_prefill_requests([_req(64, 640, list(range(1, 11)))], plans, 64)
    with pytest.raises(ValueError, match="plans"):
        pp.order_prefill_requests([a, b], plans, 64)


def _brute_force_order(n, deps):
    """Lexicographically smallest permutation respecting deps (i before j) -- the 'stable topological' order."""
    for perm in itertools.permutations(range(n)):
        pos = {r: k for k, r in enumerate(perm)}
        if all(pos[i] < pos[j] for i, j in deps):
            return list(perm)
    return None


def test_order_random_dags_match_brute_force():
    """Random same-step hit structures (rows read full blocks of rows admitted before them), shuffled: the order
    respects every dependency and equals the lexicographically smallest valid order."""
    rng = random.Random(7)
    for trial in range(300):
        n = rng.randint(1, 6)
        next_id = [1]

        def fresh(k):
            ids = list(range(next_id[0], next_id[0] + k))
            next_id[0] += k
            return ids

        admitted = []  # (start, end, blocks) in admission order
        for i in range(n):
            pool = [b for (_, _, blks, s0) in admitted for b in blks[s0 // 64 : len(blks) - 1]]  # full written blocks
            k_hit = rng.randint(2, 4) if pool and rng.random() < 0.7 else 0
            hit = rng.sample(pool, min(k_hit, len(pool))) if k_hit else []
            if len(hit) < 2:
                hit = []
            start = 64 * len(hit)
            new = rng.randint(1, 400)
            blocks = hit + fresh(_cdiv(start + new, 64) - len(hit))
            admitted.append((start, start + new, blocks, start))
        perm = list(range(n))
        rng.shuffle(perm)
        reqs = [_req(admitted[k][0], admitted[k][1], admitted[k][2], lane=j) for j, k in enumerate(perm)]
        order, plans = _order(reqs)
        writer = {}
        for i, (r, p) in enumerate(zip(reqs, plans)):
            for b in r.page_table[p.read_only_blocks : _cdiv(p.end, 64)].tolist():
                writer[b] = i
        deps = set()
        for j, (r, p) in enumerate(zip(reqs, plans)):
            if p.has_sp1:
                for b in r.page_table[: p.read_only_blocks].tolist():
                    if b in writer and writer[b] != j:
                        deps.add((writer[b], j))
        assert order == _brute_force_order(n, deps), (trial, deps, order)


def test_plans_do_not_depend_on_lanes():
    """No per-lane / per-slot state: the same rows on other lanes (a slot change between chunks) plan identically."""
    blocks = list(range(1, 60))
    reqs1 = [_req(1348, 3000, blocks[:47], lane=0), _req(0, 500, blocks[47:55], lane=1)]
    reqs2 = [_req(1348, 3000, blocks[:47], lane=27), _req(0, 500, blocks[47:55], lane=12)]
    kw = dict(block_size=64, align=64, buckets=BUCKETS, span_cap=8192)
    p1, o1 = pp.plan_prefill_batch(reqs1, **kw)
    p2, o2 = pp.plan_prefill_batch(reqs2, **kw)
    assert p1 == p2 and o1 == o2
    for r1, r2, a, b in zip(reqs1, reqs2, p1, p2):
        for c1, c2 in zip(a.chunks, b.chunks):
            t1 = pp.chunk_tables(r1.page_table, a, c1, sdpa_width=640, max_positions=32768)
            t2 = pp.chunk_tables(r2.page_table, b, c2, sdpa_width=640, max_positions=32768)
            assert all(torch.equal(x, y) for x, y in zip((t1.fill, t1.rope), (t2.fill, t2.rope)))


# ================================================================================================================
# scheduler config (features design §1.5)
# ================================================================================================================
def _sched(**kw):
    base = dict(  # the production launch: A = 128 (G9 per-bucket chunks), budget = threshold = 8064
        chunked=True,
        budget=PROD_BUDGET,
        threshold=PROD_BUDGET,
        align=PROD_A,
        span_cap=8192,
        prefix_caching=True,
        prefix_match_unit=None,
        block_size=64,
        kv_replicated=True,
    )
    base.update(kw)
    return pp.check_scheduler_config(**base)


def test_scheduler_config_production_is_clean():
    assert _sched() == []
    assert _sched(align=64, budget=8128, threshold=8128) == []  # the pre-G9 all-64/64 geometry (A = 64)
    assert _sched(budget=3968, threshold=2048) == []  # the interactive variant of §1.1 at A = 128 (4096 - A)
    assert _sched(align=64, budget=4032, threshold=2048) == []  # ... and at A = 64
    assert _sched(threshold=0) == []
    assert _sched(prefix_match_unit=64) == []
    assert _sched(chunked=False, budget=None, threshold=0) == []  # chunking off: nothing to check
    assert _sched(prefix_caching=False, kv_replicated=False) == []


def test_scheduler_config_warnings():
    w = _sched(budget=8128, threshold=8128)  # the pre-G9 budget at A = 128: unaligned, and 8128 + 127 > 8192
    assert any("max_num_batched_tokens 8128" in m and "128" in m for m in w)
    assert any("long_prefill_token_threshold 8128" in m for m in w)
    assert any("span cap" in m and "<= 8064" in m for m in w) and len(w) == 3
    w = _sched(budget=8192, threshold=0)  # 8192 + 127 > 8192: rows split internally
    assert len(w) == 1 and "span cap" in w[0] and "8064" in w[0]
    w = _sched(align=64, budget=8192, threshold=0)  # ... 8192 + 63 at A = 64
    assert len(w) == 1 and "span cap" in w[0] and "8128" in w[0]
    w = _sched(budget=2048, threshold=0)  # vLLM's unpinned `vllm serve` default on TT
    assert len(w) == 1 and "2048" in w[0] and "eager" in w[0] and "pin 8064" in w[0]
    w = _sched(threshold=4100)
    assert len(w) == 1 and "4100" in w[0]


def test_scheduler_config_errors():
    with pytest.raises(ValueError, match="KV-R"):
        _sched(kv_replicated=False)
    with pytest.raises(ValueError, match="prefix-match-unit"):
        _sched(prefix_match_unit=128)
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        _sched(budget=None)
    with pytest.raises(ValueError, match="threshold"):
        _sched(threshold=-1)
    with pytest.raises(ValueError, match="block-size|block size|block_size"):
        _sched(block_size=16)
    with pytest.raises(ValueError, match="alignment"):
        _sched(align=96)
    with pytest.raises(ValueError, match="span cap"):
        _sched(span_cap=64)
    assert pp.recommended_budget(8192, 64) == 8128 and pp.recommended_budget(8192, 128) == 8064
    with pytest.raises(ValueError):
        pp.recommended_budget(64, 64)
    with pytest.raises(ValueError):
        pp.recommended_budget(128, 128)  # no budget fits a resumed chunk in one bucket; check_scheduler_config copes


def test_scheduler_config_span_cap_not_above_alignment():
    """MOTIF3_PREFILL_MAX_BUCKET=128 (allowed: a power of two in [128, 32768]) at A = 128 with vLLM's unpinned budget
    2048: the bridge's init_device check (check_serving_config: DEFAULT_PREFILL_ALIGNMENT and the env's span cap)
    must warn, not raise "span cap 128 must exceed the alignment 128" from recommended_budget (review of F5). No budget
    fits a resumed chunk in a 128-row span then, so the warning names the span cap, and the small-budget warning (its
    floor is min(4096, cap) - A) stays silent: the span cap, not the budget, keeps the chunks small."""
    cap = api.prefill_span_cap_from_env({"MOTIF3_PREFILL_MAX_BUCKET": "128"})
    assert cap == 128 and pp.resume_alignment(64, pp.span_buckets(32768, cap)) == 128 == api.DEFAULT_PREFILL_ALIGNMENT
    w = _sched(span_cap=cap, align=api.DEFAULT_PREFILL_ALIGNMENT, budget=2048, threshold=0)
    assert len(w) == 1 and "128-row span cap" in w[0] and "MOTIF3_PREFILL_MAX_BUCKET" in w[0], w
    assert "<=" not in w[0] and "pin" not in w[0], w
    w = _sched(span_cap=cap, budget=PROD_BUDGET, threshold=PROD_BUDGET)  # the TIS budget on that server
    assert len(w) == 1 and "MOTIF3_PREFILL_MAX_BUCKET" in w[0], w
    # The same span cap with a bf16 latent cache (A = 64): a budget fits, and the warning suggests it.
    w = _sched(span_cap=cap, align=64, budget=2048, threshold=0)
    assert len(w) == 1 and "span cap" in w[0] and "<= 64" in w[0], w
    # A span cap below 4096: the budget that fits it (cap - A) is the floor, not 4096 - A, so the pin clears both.
    w = _sched(span_cap=2048, budget=2048, threshold=0)
    assert len(w) == 1 and "span cap" in w[0] and "<= 1920" in w[0], w
    w = _sched(span_cap=2048, budget=1024, threshold=0)
    assert len(w) == 1 and "1024 < 1920" in w[0] and "pin 1920" in w[0], w
    assert _sched(span_cap=2048, budget=1920, threshold=1920) == []


def test_scheduler_config_every_span_cap_and_advice_clears():
    """Every span cap MOTIF3_PREFILL_MAX_BUCKET allows, at A = 64 (bf16 cache) and 128 (G9 per-bucket), block 32 and
    64, budgets from 1 to 32768 and thresholds 0 / 2048 / 8064: check_scheduler_config never raises, never suggests a
    budget below A, and following a suggestion clears the warning it came from ("pin X": all of them; "<= X": the
    span-cap one; "use X" for an unaligned value: the alignment one)."""
    caps = [b for b in BUCKETS if api.check_prefill_span_cap(b) == b]
    assert caps[0] == 128 and caps[-1] == 32768
    budgets = (1, 63, 64, 100, 128, 1920, 2048, 3968, 4032, 4096, 8064, 8128, 8192, 16384, 32640, 32768)
    seen_pin = seen_le = seen_cap = 0
    for cap, A, bs in itertools.product(caps, (64, 128), (32, 64)):
        for budget, thr in itertools.product(budgets, (0, 2048, 8064)):
            w = _sched(span_cap=cap, align=A, block_size=bs, budget=budget, threshold=thr)
            for m in w:
                if "pin " in m:
                    seen_pin += 1
                    p = int(m.rsplit("pin ", 1)[1].rstrip(")"))
                    assert p >= A and p % A == 0 and cap > A, (cap, A, budget, thr, m)
                    assert _sched(span_cap=cap, align=A, block_size=bs, budget=p, threshold=p) == [], (cap, A, m)
                if "span cap" in m:
                    if "<= " in m:
                        seen_le += 1
                        x = int(m.rsplit("<= ", 1)[1].rstrip(")"))
                        assert x >= A and cap > A, (cap, A, m)
                        again = _sched(span_cap=cap, align=A, block_size=bs, budget=x, threshold=x)
                        assert not any("span cap" in n for n in again), (cap, A, m, again)
                    else:
                        seen_cap += 1
                        assert cap <= A and "MOTIF3_PREFILL_MAX_BUCKET" in m, (cap, A, m)
                if "not a multiple of the resume alignment" in m:
                    u = int(m.rsplit("(use ", 1)[1].rstrip(")"))
                    assert u >= A and u % A == 0, m
            assert not any("small eager" in m for m in w) or budget < min(4096, cap) - A
    assert seen_pin and seen_le and seen_cap  # every kind of advice was exercised


# ================================================================================================================
# emulated paged cache + a vLLM-like scheduler
# ================================================================================================================
EMPTY, PAD = -1, -2


def _prefix_hashes(tokens):
    """KV at position p depends on tokens [0, p] only (G1): marker = a hash of that prefix."""
    h, out = 1469598103934665603, []
    for t in tokens:
        h = (h * 1099511628211 + int(t) + 1) & ((1 << 61) - 1)
        out.append(h)
    return out


class PagedKVEmu:
    """One layer's latent cache as every chip holds it after a prefill (prefill is replicated). Applies a row's chunk
    plan with exactly the tables the generator would build: SWA tail reads, the fill (global sp1: before the SDPA),
    global sp1 reads of [0, chunk.end) through the SDPA table. Asserts every read sees the KV of the reader's own
    token prefix, every write lands in a block the row allocated, and no completed block is ever rewritten."""

    def __init__(self, num_blocks, bs, *, max_len=32768, cap=8192, tail=TAIL):
        self.kv = [[EMPTY] * bs for _ in range(num_blocks)]
        self.bs, self.max_len, self.cap, self.tail = bs, max_len, cap, tail
        self.width = pp.sdpa_table_width(max_len, cap, bs)
        self.complete = set()
        self.writes = 0

    def _read(self, b, q, want, what):
        got = self.kv[b][q % self.bs]
        assert got == want, f"{what}: position {q} in block {b} holds {got}, expected the reader's KV {want}"

    def run_row(self, req, plan, hashes, allowed):
        bs, pt = self.bs, req.page_table
        for c in plan.chunks:
            t = pp.chunk_tables(pt, plan, c, sdpa_width=self.width, max_positions=self.max_len, swa_tail=self.tail)
            if c.is_sp1:  # SWA layers: the tail rows before the chunk, from the cache
                for q in range(c.start - self.tail, c.start):
                    self._read(int(t.tail[(q - c.start + self.tail) // bs]), q, hashes[q], "SWA tail")
            for r in range(c.bucket):  # fill (global sp1 fills before its SDPA)
                b = int(t.fill[r // bs])
                if b < 0:
                    continue
                p = c.start + r
                assert b in allowed, f"write of position {p} into block {b}, which the row did not allocate"
                assert b not in self.complete, f"block {b} was complete (maybe shared) and is written again"
                assert p >= plan.w0
                self.kv[b][p % bs] = hashes[p] if p < c.end else PAD
                self.writes += 1
            if c.is_sp1:  # global layers: keys [0, end) after the fill
                for q in range(c.end):
                    self._read(int(t.sdpa[q // bs]), q, hashes[q], "global sp1")

    def run_call(self, rows, *, order=None, align=64):
        """``rows`` = [(PrefillRequest, hashes, allocated block ids)]: one prefill_forward_batch call."""
        reqs = [r for r, _, _ in rows]
        plans, auto = pp.plan_prefill_batch(
            reqs, block_size=self.bs, align=align, buckets=BUCKETS, span_cap=self.cap, swa_tail=self.tail
        )
        for i in auto if order is None else order:
            self.run_row(reqs[i], plans[i], rows[i][1], rows[i][2])
        for req, hashes, _ in rows:  # after the call: the whole row is valid; its full blocks are complete
            for q in range(req.seq_len):
                self._read(int(req.page_table[q // self.bs]), q, hashes[q], "after the call")
            for k in range(req.seq_len // self.bs):
                self.complete.add(int(req.page_table[k]))
        return plans, auto


def _rows_of(tokens, start, end, blocks, *, width=512, lane=0):
    pt = torch.zeros(width, dtype=torch.int32)
    pt[: len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    tok = torch.tensor(tokens[:end], dtype=torch.int32)
    return api.PrefillRequest(lane=lane, tokens=tok, page_table=pt, start=start)


@pytest.mark.parametrize("cap", [8192, 512])
@pytest.mark.parametrize("bs, A", [(64, 64), (32, 128)])
def test_emu_unaligned_chunks_with_lane_changes(bs, A, cap):
    """A 5000-token prompt in vLLM chunks with unaligned ends, each chunk on another lane: every read valid."""
    rng = random.Random(bs + A + cap)
    tokens = [rng.randrange(1000) for _ in range(5000)]
    h = _prefix_hashes(tokens)
    emu = PagedKVEmu(400, bs, cap=cap)
    blocks = list(range(1, _cdiv(5000, bs) + 1))
    ends = [700, 1348, 1349, 2048, 3333, 5000]
    start = 0
    for k, end in enumerate(ends):
        req = _rows_of(tokens, start, end, blocks[: _cdiv(end, bs)], width=32768 // bs, lane=(7 * k) % 32)
        emu.run_call([(req, h, set(blocks))], align=A)
        start = end


def test_emu_prefix_hit_leaves_shared_blocks_untouched():
    bs = 64
    rng = random.Random(3)
    sys_prompt = [rng.randrange(1000) for _ in range(1280)]
    ta = sys_prompt + [rng.randrange(1000) for _ in range(720)]
    tb = sys_prompt + [rng.randrange(1000) for _ in range(500)]
    emu = PagedKVEmu(200, bs)
    a_blocks = list(range(1, 33))
    emu.run_call([(_rows_of(ta, 0, 2000, a_blocks), _prefix_hashes(ta), set(a_blocks))])
    snapshot = [list(emu.kv[b]) for b in a_blocks]
    b_blocks = a_blocks[:20] + list(range(100, 108))  # hit of 20 full blocks (1280 tokens) + 8 own blocks
    emu.run_call([(_rows_of(tb, 1280, 1780, b_blocks), _prefix_hashes(tb), set(b_blocks[20:]))])
    assert [list(emu.kv[b]) for b in a_blocks] == snapshot


def test_emu_same_step_hit_needs_writer_first():
    """B (row 0) hits full blocks that A (row 1) computes in the same call: the planned order runs A first; the
    input order would read unwritten KV (the emulator catches it)."""
    bs = 64
    rng = random.Random(4)
    ta = [rng.randrange(1000) for _ in range(3000)]
    tb = ta[:2560] + [rng.randrange(1000) for _ in range(300)]
    a_blocks = list(range(1, 48))
    b_blocks = a_blocks[:40] + list(range(100, 105))
    rows = [
        (_rows_of(tb, 2560, 2860, b_blocks, lane=8), _prefix_hashes(tb), set(b_blocks[40:])),
        (_rows_of(ta, 0, 3000, a_blocks, lane=0), _prefix_hashes(ta), set(a_blocks)),
    ]
    _, order = PagedKVEmu(200, bs).run_call(rows)
    assert order == [1, 0]
    with pytest.raises(AssertionError, match="global sp1|SWA tail"):
        PagedKVEmu(200, bs).run_call(rows, order=[0, 1])


class FakeVllmScheduler:
    """What shapes prefill rows in vLLM 0.26 (``v1/core/sched/scheduler.py``, ``kv_cache_manager.py``): prefix hits
    on admission only (full blocks, capped at ``num_tokens - 1``), partial prefills first, chunk = ``min(remaining,
    threshold, budget left)``, blocks allocated and full blocks cached at schedule time (so a later-admitted request
    can hit blocks an earlier row of the same step computes), at most 32 rows per step. Blocks are never freed."""

    def __init__(self, *, bs, num_blocks, budget, threshold, width):
        self.bs, self.budget, self.threshold, self.width = bs, budget, threshold, width
        self.free = deque(range(1, num_blocks))
        self.cached = {}
        self.waiting, self.running = deque(), []

    def add(self, tokens):
        h = _prefix_hashes(tokens)
        bh = [h[(k + 1) * self.bs - 1] for k in range(len(tokens) // self.bs)]
        self.waiting.append(SimpleNamespace(tokens=tokens, h=h, bh=bh, blocks=[], own=set(), computed=0))

    def _alloc(self, r, n):
        end = r.computed + n
        while len(r.blocks) < _cdiv(end, self.bs):
            b = self.free.popleft()
            r.blocks.append(b)
            r.own.add(b)
        for k in range(end // self.bs):
            self.cached.setdefault(r.bh[k], r.blocks[k])
        return r, end

    def schedule(self):
        budget, rows = self.budget, []
        cap = self.threshold if self.threshold > 0 else 1 << 30
        for r in self.running:
            if budget <= 0 or len(rows) >= 32:
                break
            n = min(len(r.tokens) - r.computed, cap, budget)
            rows.append(self._alloc(r, n))
            budget -= n
        while self.waiting and budget > 0 and len(rows) < 32:
            r = self.waiting.popleft()
            hit = 0
            while hit < (len(r.tokens) - 1) // self.bs and r.bh[hit] in self.cached:
                hit += 1
            r.blocks = [self.cached[r.bh[k]] for k in range(hit)]
            r.computed = hit * self.bs
            n = min(len(r.tokens) - r.computed, cap, budget)
            self.running.append(r)
            rows.append(self._alloc(r, n))
            budget -= n
        return rows

    def commit(self, rows):
        for r, end in rows:
            r.computed = end
        self.running = [r for r in self.running if r.computed < len(r.tokens)]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "bs, A, budget, threshold, cap",
    [
        (64, 128, 8064, 8064, 8192),  # production (G9 per-bucket q / k: A = 128; threshold = budget = 8064)
        (64, 64, 8128, 8128, 8192),  # the pre-G9 all-64/64 geometry (A = 64)
        (64, 64, 1000, 0, 8192),  # unaligned budget: every chunk end unaligned
        (64, 128, 777, 300, 512),  # A = 128, tiny cap: many internal chunks
        (32, 64, 2048, 1024, 1024),
    ],
)
def test_emu_vllm_like_schedule(bs, A, budget, threshold, cap, seed):
    """Shared system prompts (a fresh one per wave, so a later-admitted request hits blocks an earlier row computes
    in the SAME step), duplicate prompts (full-prompt hits), multi-turn extensions, unaligned chunk ends, rows of a
    call in shuffled order (the plugin keeps no admission order), a new lane for every row of every call."""
    rng = random.Random(1000 * seed + budget + threshold + cap + bs)
    width = 32768 // bs
    nb = 12000 if bs == 64 else 24000
    sched = FakeVllmScheduler(bs=bs, num_blocks=nb, budget=budget, threshold=threshold, width=width)
    emu = PagedKVEmu(nb, bs, cap=cap)
    systems = [[rng.randrange(5000) for _ in range(n)] for n in (64, 200, 1300, 2600)]
    history = []
    stats = dict(steps=0, rows=0, resumed=0, multi=0, sp1=0, same_step_hits=0, reordered=0)
    for wave in range(12):
        fresh = [rng.randrange(5000) for _ in range(rng.randint(130, 2600))]  # unseen: computed in this wave
        for _ in range(rng.randint(2, 3)):
            toks = fresh + [rng.randrange(5000) for _ in range(rng.randint(1, 700))]
            history.append(toks)
            sched.add(toks)
        for _ in range(rng.randint(0, 3)):
            kind = rng.random()
            if kind < 0.3 and history:
                toks = list(rng.choice(history))  # duplicate prompt: hit capped at num_tokens - 1
            elif kind < 0.6 and history:
                toks = list(rng.choice(history)) + [rng.randrange(5000) for _ in range(rng.randint(1, 900))]
            else:
                toks = rng.choice(systems) + [rng.randrange(5000) for _ in range(rng.randint(1, 1500))]
            history.append(toks)
            sched.add(toks)
        while sched.running or sched.waiting:
            rows = sched.schedule()
            perm = list(range(len(rows)))
            rng.shuffle(perm)
            lanes = rng.sample(range(32), len(rows))
            call, writer = [], {}
            for k, lane in zip(perm, lanes):
                r, end = rows[k]
                req = _rows_of(r.tokens, r.computed, end, r.blocks, width=width, lane=lane)
                call.append((req, r.h, r.own))
                stats["resumed"] += req.start > 0
            plans, order = emu.run_call(call, align=A)
            for i, ((req, _, _), p) in enumerate(zip(call, plans)):
                for b in req.page_table[p.read_only_blocks : _cdiv(p.end, bs)].tolist():
                    writer[b] = i
            for j, ((req, _, _), p) in enumerate(zip(call, plans)):
                if p.has_sp1 and any(writer.get(b, j) != j for b in req.page_table[: p.read_only_blocks].tolist()):
                    stats["same_step_hits"] += 1
            stats["reordered"] += order != sorted(order)
            stats["steps"] += 1
            stats["rows"] += len(rows)
            stats["multi"] += len(rows) > 1
            stats["sp1"] += sum(p.has_sp1 for p in plans)
            sched.commit(rows)
            assert stats["steps"] < 2000
    assert min(stats["resumed"], stats["multi"], stats["sp1"], stats["same_step_hits"], stats["reordered"]) > 0, stats
    print(
        f"[emu] bs={bs} A={A} budget={budget} threshold={threshold} cap={cap} seed={seed}: {stats}, "
        f"{emu.writes} row writes"
    )


# ================================================================================================================
# packed prefill (P5): docs/p5_t64/P5_T64_DESIGN.md §3, §7.1
# ================================================================================================================
# Production geometry: bs 64, A = 128, budget = threshold = 8064, span cap 8192; packing knobs at their defaults
# (S <= 1024, T <= 8192, B <= 32, pk1 on).
SDPA_W = pp.sdpa_table_width(32768, 8192, 64)  # 640
POS = 32768  # max_model_len: RoPE table rows
VARIANTS = api.PK1_TAIL_VARIANTS  # ("shared", "distinct")


class _Ids:
    """Fresh block ids >= 1 in a random order (vLLM's free queue)."""

    def __init__(self, seed=0, n=1 << 17):
        self.ids = (torch.randperm(n - 1, generator=torch.Generator().manual_seed(seed)) + 1).tolist()
        self.k = 0

    def take(self, n):
        out = self.ids[self.k : self.k + n]
        assert len(out) == n
        self.k += n
        return out


def _prow(start, end, blocks, lane, *, bs=64, tokens=None):
    """A PrefillRequest over ``blocks`` (position order; page-table width 32768 / bs). Tokens: ``tokens[:end]`` or a
    pattern that differs per lane."""
    pt = torch.zeros(32768 // bs, dtype=torch.int32)
    pt[: len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    tok = tokens[:end] if tokens is not None else [(7 * p + 13 * lane) % 5000 for p in range(end)]
    return api.PrefillRequest(lane=lane, tokens=torch.tensor(tok, dtype=torch.int32), page_table=pt, start=start)


def _plans(reqs, *, bs=64, A=PROD_A, cap=8192):
    return [_plan(r.start, r.seq_len, bs=bs, A=A, cap=cap) for r in reqs]


def _p5(reqs, plans=None, *, bs=64, A=PROD_A, cap=8192, **kw):
    """``(plan_prefill_passes(...), plans)`` at the production knobs unless overridden."""
    plans = _plans(reqs, bs=bs, A=A, cap=cap) if plans is None else plans
    kw.setdefault("max_tokens", min(8192, cap))
    return pp.plan_prefill_passes(reqs, plans, block_size=bs, **kw), plans


def _desc(passes):
    return [p.describe() for p in passes]


def _oracle(reqs, plans, passes, bs=64):
    """An oracle independent of ``segment_dag``. Each chunk ``(a, e)`` of row ``i`` writes the blocks of ``[max(a, w0),
    e)``. An sp1 chunk reads from the cache every position below ``max(a, w0)``: its global layers fill first and then
    read keys ``[0, a + i]``, and the fill skips ``[a, w0)`` when ``c0 < w0`` (review edit R-E1); its SWA tail ``[a -
    128, a)`` lies below too. Checks: every chunk is in exactly one pass; every block such a read hits that the call
    writes is written in a strictly earlier pass (no pass holds a dependency pair, chunks of a row run in order); no
    block is written twice. Returns ``{(row, chunk): pass index}``."""
    pass_of = {}
    for n, p in enumerate(passes):
        for g in p.segments:
            assert g.key not in pass_of, f"chunk {g.key} in passes {pass_of[g.key]} and {n}"
            pass_of[g.key] = n
    assert set(pass_of) == {(i, k) for i, pl in enumerate(plans) for k in range(len(pl.chunks))}, "chunk coverage"
    writer = {}
    for i, (r, pl) in enumerate(zip(reqs, plans)):
        pt = r.page_table.tolist()
        for k, c in enumerate(pl.chunks):
            for L in range(max(pl.w0, c.start) // bs, _cdiv(c.end, bs)):
                assert pt[L] not in writer, f"block {pt[L]} written by {writer.get(pt[L])} and {(i, k)}"
                writer[pt[L]] = (i, k)
    for j, (r, pl) in enumerate(zip(reqs, plans)):
        pt = r.page_table.tolist()
        for k, c in enumerate(pl.chunks):
            if c.path != "sp1":
                continue  # sp0 reads nothing from the cache
            for L in range(max(c.start, pl.w0) // bs):
                w = writer.get(pt[L])
                assert w is None or pass_of[w] < pass_of[(j, k)], (
                    f"row {j} chunk {k} (pass {pass_of[(j, k)]}) reads block {pt[L]} (logical {L}, w0 {pl.w0}, c0 "
                    f"{pl.c0}) that chunk {w} writes in pass {pass_of[w]}: {_desc(passes)}"
                )
    return pass_of


def _sizes(buckets, *, max_seg, max_tokens, max_batch):
    return tuple(S for S in buckets if S <= max_seg and 2 * S <= max_tokens and max_batch >= 2)


def _check_structure(
    passes, plans, *, max_seg=1024, max_tokens=8192, max_batch=32, pk1=True, seg_buckets=None, sp1_seg_buckets=None
):
    """Every pass's shape rules, recomputed from the knobs (not from the planner's helpers)."""
    lim = dict(max_seg=max_seg, max_tokens=max_tokens, max_batch=max_batch)
    seg = _sizes(api.PACK_SEG_BUCKETS if seg_buckets is None else seg_buckets, **lim)
    sp1 = _sizes(api.PACK_SP1_SEG_BUCKETS if sp1_seg_buckets is None else sp1_seg_buckets, **lim) if pk1 else ()
    for p in passes:
        for g in p.segments:
            c = plans[g.row].chunks[g.chunk_index]
            assert (g.start, g.end, g.path, g.last) == (c.start, c.end, c.path, c.last), (g, c)
            fit = [S for S in (sp1 if g.path == "sp1" else seg) if S >= g.real_rows]
            assert g.rows == (min(fit) if fit else None), (g, fit)
        if p.kind == "solo":
            g = p.segments[0]
            c = plans[g.row].chunks[g.chunk_index]
            assert (p.batch, p.seg_rows, p.start, p.tokens, p.tails) == (1, c.bucket, c.start, c.bucket, None)
            assert p.shape == (c.path, c.bucket) and p.dummies == 0
            assert p.fallback is None or p.fallback[0] in api.PACKED_PASS_KINDS
            continue
        S, B, T, n = p.seg_rows, p.batch, p.tokens, len(p.segments)
        assert p.fallback is None and B in api.PACK_BATCHES and 2 <= n <= B <= max_batch, p.describe()
        assert B == 1 << (n - 1).bit_length() and T == B * S <= max_tokens and T & (T - 1) == 0, p.describe()
        assert all(g.rows == S for g in p.segments) and p.dummies == B - n
        assert [g.key for g in p.segments] == sorted(g.key for g in p.segments)
        if p.kind == "pk0":
            assert p.start == 0 and p.tails is None and all(g.path == "sp0" for g in p.segments)
            assert p.shape == ("pk0", T, S)
        else:
            assert p.kind == "pk1" and pk1 and all(g.path == "sp1" and g.start == p.start for g in p.segments)
            assert p.tails in VARIANTS and p.shape == ("pk1", T, S, p.tails)


def _check_pass_tables(p, reqs, plans, *, bs=64, tail=TAIL, width=SDPA_W, P=POS):
    """The pass tables against first principles and the per-chunk builders: each segment's fill entry ``j`` (logical
    block ``a / bs + j``) is the block id iff the block overlaps ``[w0, end)`` (fills never write a shared block or a
    pure-padding block), and it equals the chunk's own fill table cut or -1-extended to ``S`` (review edit R-E12);
    RoPE rows ``min(a + i, P - 1)``; sp1 SDPA rows and tails are the per-chunk ones; dummies write nothing and copy
    segment 0's RoPE rows, end, SDPA row and tails (R-E2); a pk1 pass is ``shared`` iff all its tail rows are equal;
    the tokens are each segment's ``tokens[start:end]``, padded."""
    t = pp.pass_tables(p, reqs, plans, block_size=bs, sdpa_width=width, max_positions=P, swa_tail=tail)
    S, B, T, n = p.seg_rows, p.batch, p.tokens, p.seg_rows // bs
    assert t.fill.dtype == t.rope.dtype == t.ends.dtype == torch.int32
    assert t.fill.shape == (1, T // bs) and t.rope.shape == (T,) and t.ends.shape == (B,) and t.tails == p.tails
    assert not bool((t.fill == 0).any()), "a fill table never targets the null block"
    for k in range(B):
        g = p.segments[k] if k < len(p.segments) else p.segments[0]  # a dummy copies segment 0 (but writes nothing)
        pl, pt = plans[g.row], reqs[g.row].page_table
        f = t.fill[0, k * n : (k + 1) * n]
        if k >= len(p.segments):
            assert bool((f == -1).all()), f"dummy {k} of {p.describe()} writes {f.tolist()}"
        else:
            L = g.start // bs + torch.arange(n)
            on = (L * bs + bs > pl.w0) & (L * bs < g.end)
            want = torch.where(on, pt[L.clamp(max=pt.numel() - 1)], torch.tensor(-1, dtype=torch.int32))
            assert torch.equal(f, want), (p.describe(), k, f.tolist(), want.tolist())
            own = pp.fill_table(pt, pl.chunks[g.chunk_index], pl.w0, bs)  # the chunk at its own bucket
            m = min(n, own.numel())
            assert torch.equal(f[:m], own[:m]) and bool((own[m:] == -1).all()) and bool((f[m:] == -1).all())
        r = t.rope[k * S : (k + 1) * S]
        assert torch.equal(r, torch.arange(g.start, g.start + S, dtype=torch.int32).clamp(max=P - 1))
        assert int(t.ends[k]) == g.end
        if p.path == "sp1":
            assert torch.equal(t.sdpa[k], pp.sdpa_table(pt, g.end, bs, width))
            c = pl.chunks[g.chunk_index]
            assert torch.equal(t.tail[k], pp.tail_blocks(pt, c, bs, tail)) and int(t.start_idx[0]) == p.start
    if p.path == "sp1":
        assert t.sdpa.shape == (B, width) and bool((t.sdpa >= 0).all()) and t.tail.shape == (B, tail // bs)
        real = t.tail[: len(p.segments)]
        if p.kind == "pk1":
            assert p.tails == ("shared" if bool((real == real[:1]).all()) else "distinct"), p.describe()
            assert pp.tail_variant(t.tail) == p.tails
    else:
        assert t.sdpa is None and t.start_idx is None and t.tail is None
    tok = pp.pass_tokens(p, reqs, -7)
    for k, g in enumerate(p.segments):
        assert torch.equal(tok[k * S : k * S + g.real_rows], reqs[g.row].tokens[g.start : g.end])
        assert bool((tok[k * S + g.real_rows : (k + 1) * S] == -7).all())
    assert bool((tok[len(p.segments) * S :] == -7).all())


def _check_call(reqs, plans, passes, *, bs=64, tables=True, **lim):
    _check_structure(passes, plans, **lim)
    pass_of = _oracle(reqs, plans, passes, bs)
    if tables:
        for p in passes:
            _check_pass_tables(p, reqs, plans, bs=bs)
    return pass_of


def test_p5_constants_and_contract():
    assert (pp.SOLO, pp.PK0, pp.PK1) == ("solo", "pk0", "pk1") and pp.PASS_KINDS == ("solo",) + api.PACKED_PASS_KINDS
    assert (pp.SHARED_TAILS, pp.DISTINCT_TAILS) == VARIANTS == ("shared", "distinct")
    assert pp.DEFAULT_PACK_MAX_BATCH == max(api.PACK_BATCHES) == api.NUM_LANES == 32
    assert api.PACK_SEG_BUCKETS == (64, 128, 256, 512, 1024) and api.PACK_SP1_SEG_BUCKETS == (128, 256, 512, 1024)
    m = pp.DEFAULT_PACKED_COST
    # transposes + MTP fill: flat ~26 ms up to the 8192-row pass cap (G15a-rest; P5 review finding 6), growing above
    assert all(m.extras_s(T) == pytest.approx(0.0262) for T in (128, 1024, 2048, 4096, 8192))
    assert m.extras_s(16384) == pytest.approx(53 * 0.4e-3 * 2 + 5e-3)
    assert m.tails_s("shared", 32) == pytest.approx(39 * 0.65e-3)  # ~25 ms
    assert m.tails_s("distinct", 32) == pytest.approx(39 * (0.65e-3 + 64 * 0.06e-3))  # ~0.18 s
    assert all(m.tails_s("distinct", B) > m.tails_s("shared", B) for B in api.PACK_BATCHES)
    with pytest.raises(ValueError, match="tail variant"):
        m.tails_s("both", 2)
    with pytest.raises(ValueError):
        pp.PackedCostModel(head_s=-1.0)
    with pytest.raises(ValueError):
        pp.PackedCostModel(transpose_ref_tokens=0)


def test_p5_cost_model_calibration():
    """The packed-pass terms against their measurements (P5 review, finding 6; tt/prefill_plan.PackedCostModel):

    * transposes (G15a-rest, GATES_RESULTS §13.3): 4 per layer cost 0.32-0.42 ms of eager host dispatch at T 1024-8192
      (table (a)) and 115.4 / 397.3 us of device time at T 2048 / 8192 (table (d), traced). The term is the larger of
      the two (an eager pass overlaps them) within 0.1 ms per layer at every T of the pass cap;
    * distinct tails (G15a-rest table (c), eager per SWA layer): distinct minus shared = +3.92 / +0.74 / +0.04 ms at
      B = 32 / 8 / 2; the model's premium ``2 B * tail_op_s`` within 0.3 ms of each;
    * the whole pass: the review's every-packed-shape probe at 53 layers (logs/dev/20261003_193703_p5rev_all_shapes.log)
      measured pk0 T = 2048 S = 64 in 1.63 s and T = 8192 S = 256 / 512 / 1024 in 5.91 / 5.86 / 5.84 s: modelled within
      0.15 s (the table's eager bucket costs predate ring_gather="safe", +~0.07 s per pass at small buckets).
    """
    m = pp.DEFAULT_PACKED_COST
    host_ms = {1024: 0.33, 2048: 0.34, 4096: 0.34, 8192: 0.42}  # 4 eager transposes per layer (max over kinds, (a))
    dev_ms = {2048: 0.1154, 8192: 0.3973}  # traced, per layer, (d)
    for T in (1024, 2048, 4096, 8192):
        per_layer_ms = (m.extras_s(T) - m.mtp_fill_s) / m.layers * 1e3
        assert per_layer_ms == pytest.approx(max(host_ms[T], dev_ms.get(T, 0.0)), abs=0.1), T
    for B, premium_ms in ((32, 3.92), (8, 0.74), (2, 0.04)):
        model_ms = (m.tails_s("distinct", B) - m.tails_s("shared", B)) / m.swa_layers * 1e3
        assert model_ms == pytest.approx(premium_ms, abs=0.3), B
    c = pp.packed_pass_cost
    for (b, rows, S), sec in (((32, 40, 64), 1.63), ((32, 200, 256), 5.91), ((16, 400, 512), 5.86),
                              ((8, 900, 1024), 5.84)):  # fmt: skip
        assert c(_packed(b, rows, S)) == pytest.approx(sec, abs=0.15), (b, S)


def test_segment_rows_every_chunk_size():
    """S = the smallest segment size >= the chunk's real rows: sp0 64 ... 1024, sp1 128 ... 1024, None above (solo);
    narrower size lists (cfg.pack_seg_buckets without 64: G15's S_min = 128 fallback; pk1 off) are honoured. Review edit
    R-E4: the chunk's rows count, not the row's span (a 2108-token row's 60-row tail chunk packs at 128)."""
    for rows in range(1, 8193):
        sp0 = pp.ChunkPlan(start=0, bucket=8192, end=rows, path="sp0", last=True)
        sp1 = pp.ChunkPlan(start=2048, bucket=8192, end=2048 + rows, path="sp1", last=True)
        want0 = next((S for S in (64, 128, 256, 512, 1024) if S >= rows), None)
        want1 = next((S for S in (128, 256, 512, 1024) if S >= rows), None)
        assert pp.segment_rows(sp0, block_size=64) == want0 and pp.segment_rows(sp1, block_size=64) == want1
        assert pp.segment_rows(sp0, block_size=32, seg_buckets=(128, 256)) == next(
            (S for S in (128, 256) if S >= rows), None
        )
        assert pp.segment_rows(sp1, block_size=64, sp1_seg_buckets=()) is None
    p = _plan(0, 2108, A=PROD_A)
    assert _chunks(p) == [(0, 2048, "sp0"), (2048, 128, "sp1")]  # row span 2108, tail chunk of 60 rows
    assert [pp.segment_rows(c, block_size=64) for c in p.chunks] == [None, 128]
    with pytest.raises(ValueError, match="multiples of the block size"):
        pp.segment_rows(p.chunks[1], block_size=64, seg_buckets=(32, 64))
    with pytest.raises(ValueError, match="strictly increasing"):
        pp.segment_rows(p.chunks[1], block_size=64, sp1_seg_buckets=(256, 128))


def test_pass_dataclasses():
    sp0 = [pp.PackSegment(i, 0, "sp0", 0, 30 + i, 64, True) for i in range(3)]
    p = pp.PrefillPass("pk0", sp0, 64, 4, 0)
    assert (p.tokens, p.dummies, p.real_rows, p.padding_rows, p.path, p.is_packed) == (256, 1, 93, 163, "sp0", True)
    assert p.shape == ("pk0", 256, 64) and [p.offset(k) for k in range(4)] == [0, 64, 128, 192]
    assert p.head_rows() == [(0, 29), (1, 64 + 30), (2, 128 + 31)]
    assert p.describe() == "pk0 T=256 S=64 B=4 (3 real) a=0" and isinstance(p.segments, tuple)
    with pytest.raises(IndexError):
        p.offset(4)
    mid = pp.PackSegment(5, 0, "sp1", 2048, 2048 + 128, 128, False)  # not the row's last chunk: no head row
    last = pp.PackSegment(6, 1, "sp1", 2048, 2048 + 60, 128, True)
    q = pp.PrefillPass("pk1", (mid, last), 128, 2, 2048, tails="distinct")
    assert q.shape == ("pk1", 256, 128, "distinct") and q.head_rows() == [(1, 128 + 59)] and q.path == "sp1"
    assert q.describe().endswith("a=2048 tails=distinct") and q.head_rows()[0][1] == q.offset(1) + last.head_row_local
    s = pp.PrefillPass("solo", (pp.PackSegment(0, 0, "sp0", 0, 4000, None, True),), 4096, 1, 0)
    assert (s.shape, s.tokens, s.dummies, s.head_rows(), s.is_packed) == (("sp0", 4096), 4096, 0, [(0, 3999)], False)
    f = pp.PrefillPass("solo", (sp0[0],), 128, 1, 0, fallback=("pk0", 256, 64))
    assert f.shape == ("sp0", 128) and "fallback" in f.describe()
    assert sp0[1].key == (1, 0) and sp0[1].real_rows == 31 and sp0[1].head_row_local == 30 and not sp0[1].is_sp1
    bad_segments = [
        dict(row=0, chunk_index=0, path="sp2", start=0, end=5, rows=64, last=True),
        dict(row=-1, chunk_index=0, path="sp0", start=0, end=5, rows=64, last=True),
        dict(row=0, chunk_index=0, path="sp0", start=5, end=5, rows=64, last=True),
        dict(row=0, chunk_index=0, path="sp1", start=0, end=5, rows=64, last=True),  # sp1 at 0
        dict(row=0, chunk_index=0, path="sp0", start=128, end=200, rows=128, last=True),  # sp0 after 0
        dict(row=0, chunk_index=0, path="sp0", start=0, end=100, rows=64, last=True),  # rows < real rows
    ]
    for kw in bad_segments:
        with pytest.raises(ValueError):
            pp.PackSegment(**kw)
    bad_passes = [
        ("pk2", sp0, 64, 4, 0, None, None),
        ("pk0", (), 64, 4, 0, None, None),
        ("pk0", sp0 + [sp0[0]], 64, 4, 0, None, None),  # a segment twice
        ("pk0", sp0, 16, 4, 0, None, None),  # S below a segment's rows
        ("pk0", sp0, 64, 3, 0, None, None),  # B not a power of two
        ("pk0", sp0, 64, 64, 0, None, None),  # B above 32
        ("pk0", sp0, 64, 2, 0, None, None),  # more segments than B
        ("pk0", sp0, 64, 4, 64, None, None),  # pk0 starts at 0
        ("pk0", sp0, 64, 4, 0, "shared", None),  # no tails on pk0
        ("pk0", sp0, 64, 4, 0, None, ("pk0", 256, 64)),  # fallback on a packed pass
        ("pk1", sp0, 64, 4, 0, "shared", None),  # pk1 holds sp1 segments
        ("pk1", (mid, last), 128, 2, 2048, None, None),  # pk1 without its tail variant
        ("pk1", (mid, pp.PackSegment(7, 0, "sp1", 1024, 1100, 128, True)), 128, 2, 2048, "shared", None),  # starts
        ("solo", sp0[:1], 64, 2, 0, None, None),
        ("solo", sp0[:2], 64, 1, 0, None, None),
        ("solo", (mid,), 128, 1, 0, None, None),  # solo starts at its chunk
        ("solo", sp0[:1], 128, 1, 0, None, ("sp0", 128)),  # a fallback is a packed shape
    ]
    for kind, segs, S, B, a, tails, fb in bad_passes:
        with pytest.raises(ValueError):
            pp.PrefillPass(kind, segs, S, B, a, tails=tails, fallback=fb)


def test_segment_dag_deps_and_checks():
    """Chain dependencies, the R-E1 read set (the whole read-only prefix [0, w0), including [c0, w0)), sp0 segments
    without cache dependencies, and the row-level checks of order_prefill_requests with its messages."""
    ids = _Ids(1)
    X = ids.take(35)  # cold 2200: (0, 2048) writes blocks 0-31, (2048, 256) writes 32-34
    Y = X[:33] + ids.take(1)  # hits 33 blocks (2112): w0 2112, c0 2048
    Z = X[:1] + ids.take(4)  # hit of one block (64 < the 128-row tail): c0 0, sp0, reads nothing
    reqs = [_prow(2112, 2162, Y, 0), _prow(0, 2200, X, 1), _prow(64, 300, Z, 2)]
    plans = _plans(reqs)
    assert (plans[0].w0, plans[0].c0, _chunks(plans[0])) == (2112, 2048, [(2048, 128, "sp1")])
    segs, deps = pp.segment_dag(reqs, plans, 64)
    assert [g.key for g in segs] == [(0, 0), (1, 0), (1, 1), (2, 0)]
    assert [g.rows for g in segs] == [128, None, 256, 512]
    assert deps == [(1, 2), (), (1,), ()]  # Y needs X's chunk 1 (block 32 = [c0, w0)), not only chunk 0
    # the pre-R-E1 read set [0, a / bs) would have missed X's chunk 1: block 32 lies in [a, w0)
    assert X[32] in reqs[0].page_table[2048 // 64 : 2112 // 64].tolist()
    # row-level checks, same messages as order_prefill_requests
    a = _req(0, 640, list(range(1, 11)))
    with pytest.raises(ValueError, match="both write"):
        pp.segment_dag([a, _req(0, 640, list(range(5, 15)), lane=1)], [_plan(0, 640)] * 2, 64)
    with pytest.raises(ValueError, match="twice"):
        pp.segment_dag([_req(0, 640, [1, 2, 3, 4, 5, 6, 7, 8, 9, 1])], [_plan(0, 640)], 64)
    with pytest.raises(ValueError, match="block id"):
        pp.segment_dag([_req(0, 640, [1, 2, 3, 0, 5, 6, 7, 8, 9, 10])], [_plan(0, 640)], 64)
    p_ = _req(640, 1280, list(range(11, 21)) + list(range(1, 11)))
    q_ = _req(640, 1280, list(range(1, 11)) + list(range(11, 21)), lane=1)
    with pytest.raises(ValueError, match="cycle"):
        pp.segment_dag([p_, q_], [_plan(640, 1280)] * 2, 64)
    with pytest.raises(ValueError, match="does not match"):
        pp.segment_dag([_req(64, 640, list(range(1, 11)))], [_plan(0, 640)], 64)
    with pytest.raises(ValueError, match="plans"):
        pp.segment_dag([a], [], 64)
    with pytest.raises(ValueError, match="multiples of the block size"):
        pp.segment_dag([a], [_plan(0, 640)], 64, seg_buckets=(96,))


def test_segment_dag_rows_match_order_prefill_requests():
    """Random same-step hit structures (the generator of test_order_random_dags_match_brute_force, several chunks per
    row through a 256-row span cap): the row pairs that segment dependencies join are exactly the row-level
    dependencies of order_prefill_requests, every chain is complete, and sp0 segments depend on nothing else."""
    rng = random.Random(11)
    for trial in range(300):
        n = rng.randint(1, 6)
        nxt = [1]

        def fresh(k):
            out = list(range(nxt[0], nxt[0] + k))
            nxt[0] += k
            return out

        admitted = []
        for i in range(n):
            pool = [b for (_, _, blks, s0) in admitted for b in blks[s0 // 64 : len(blks) - 1]]
            k_hit = rng.randint(1, 5) if pool and rng.random() < 0.7 else 0
            hit = rng.sample(pool, min(k_hit, len(pool))) if k_hit else []
            start = 64 * len(hit)
            new = rng.randint(1, 700)
            admitted.append((start, start + new, hit + fresh(_cdiv(start + new, 64) - len(hit)), start))
        perm = list(range(n))
        rng.shuffle(perm)
        reqs = [_req(admitted[k][0], admitted[k][1], admitted[k][2], lane=j) for j, k in enumerate(perm)]
        plans = [_plan(r.start, r.seq_len, A=PROD_A, cap=256) for r in reqs]
        segs, deps = pp.segment_dag(reqs, plans, 64)
        writer = {}
        for i, (r, p) in enumerate(zip(reqs, plans)):
            for b in r.page_table[p.read_only_blocks : _cdiv(p.end, 64)].tolist():
                writer[b] = i
        rows = set()
        for j, (r, p) in enumerate(zip(reqs, plans)):
            if p.has_sp1:
                rows |= {(writer[b], j) for b in r.page_table[: p.read_only_blocks].tolist() if writer.get(b, j) != j}
        got = set()
        for s, (g, d) in enumerate(zip(segs, deps)):
            chain = {s - 1} if g.chunk_index else set()
            assert chain <= set(d) and (g.path == "sp1" or set(d) == chain), (trial, g, d)
            got |= {(segs[t].row, g.row) for t in d if segs[t].row != g.row}
        assert got == rows, (trial, got, rows)


class _BlockStore(dict):
    """Block id -> its ``bs`` slots, created EMPTY on first use (block ids up to 2^17 without a 2^17-entry list)."""

    def __init__(self, bs):
        super().__init__()
        self.bs = bs

    def __missing__(self, b):
        self[b] = v = [EMPTY] * self.bs
        return v


class PackedKVEmu(PagedKVEmu):
    """:class:`PagedKVEmu` running a call as the planner's passes, through the pass tables and the packed layout, one
    block at a time: the SWA tail reads of every sp1 segment before the pass writes anything, the fill of every segment
    (a dummy segment writes nothing; packed row ``k S + i`` must land in segment ``k``'s block of position ``a_k +
    i``), then the global reads ``[0, end)`` of every sp1 segment through its SDPA row. It knows nothing of the
    planner's dependencies: a pass that runs too early reads EMPTY or stale KV. Blocks can be evicted (freed and
    reused by the scheduler): they lose their content and their completeness."""

    def __init__(self, bs, *, max_len=32768, cap=8192, tail=TAIL):
        super().__init__(1, bs, max_len=max_len, cap=cap, tail=tail)
        self.kv = _BlockStore(bs)

    def evict(self, b):
        self.kv[b] = [EMPTY] * self.bs
        self.complete.discard(b)

    def _check_range(self, table, lo, hi, h, what):
        bs, q = self.bs, lo
        while q < hi:
            L = q // bs
            e = min(hi, (L + 1) * bs)
            b = int(table[L])
            got, want = self.kv[b][q - L * bs : e - L * bs], h[q:e]
            if got != want:
                x = next(i for i, (u, v) in enumerate(zip(got, want)) if u != v)
                raise AssertionError(
                    f"{what}: position {q + x} in block {b} holds {got[x]}, expected the reader's KV {want[x]}"
                )
            q = e

    def run_pass(self, p, reqs, plans, rows):
        bs, S, tail = self.bs, p.seg_rows, self.tail
        t = pp.pass_tables(
            p, reqs, plans, block_size=bs, sdpa_width=self.width, max_positions=self.max_len, swa_tail=tail
        )
        if t.tail is not None:  # SWA layers: the tail [a - tail, a) from the cache, before the pass writes
            for k, g in enumerate(p.segments):
                first = (g.start - tail) // bs
                table = {first + j: b for j, b in enumerate(t.tail[k].tolist())}
                self._check_range(table, g.start - tail, g.start, rows[g.row][1], f"SWA tail, {p.describe()}")
        fill, n = t.fill[0].tolist(), S // bs
        for k in range(p.batch):
            for j, b in enumerate(fill[k * n : (k + 1) * n]):
                if b < 0:
                    continue
                assert k < len(p.segments), f"{p.describe()}: dummy segment {k} writes block {b}"
                g = p.segments[k]
                req, h, own = rows[g.row]
                pos = g.start + j * bs
                assert int(req.page_table[pos // bs]) == b, f"{p.describe()}: packed rows land in the wrong block"
                assert pos >= plans[g.row].w0, f"{p.describe()}: a fill below w0"
                assert b in own, f"write of positions {pos}.. into block {b}, which row {g.row} did not allocate"
                assert b not in self.complete, f"block {b} was complete (maybe shared) and is written again"
                self.kv[b] = [h[x] if x < g.end else PAD for x in range(pos, pos + bs)]
                self.writes += bs
        if t.sdpa is not None:  # global layers: keys [0, end) through the SDPA row, after the fill
            for k, g in enumerate(p.segments):
                self._check_range(t.sdpa[k].tolist(), 0, g.end, rows[g.row][1], f"global, {p.describe()}")

    def run_passes(self, rows, plans, passes):
        reqs = [r for r, _, _ in rows]
        for p in passes:
            self.run_pass(p, reqs, plans, rows)
        for req, h, _ in rows:  # after the call: the whole row is valid; its full blocks are complete
            self._check_range(req.page_table.tolist(), 0, req.seq_len, h, "after the call")
            for k in range(req.seq_len // self.bs):
                self.complete.add(int(req.page_table[k]))

    def run_packed_call(self, rows, *, align=PROD_A, **kw):
        """Plans one call (rows as for ``run_call``) with packing and runs its passes; returns ``(plans, passes)``."""
        reqs = [r for r, _, _ in rows]
        plans, _ = pp.plan_prefill_batch(
            reqs, block_size=self.bs, align=align, buckets=BUCKETS, span_cap=self.cap, swa_tail=self.tail
        )
        kw.setdefault("max_tokens", min(8192, self.cap))
        passes = pp.plan_prefill_passes(reqs, plans, block_size=self.bs, swa_tail=self.tail, **kw)
        self.run_passes(rows, plans, passes)
        return plans, passes

    def decode_write(self, b, pos, value, own):
        assert b in own and b not in self.complete, f"decode write of position {pos} into block {b}"
        self.kv[b][pos % self.bs] = value


def _emu_rows(reqs, tokens_of, own_of):
    """Emulator rows ``(request, prefix hashes, writable blocks)``."""
    return [(r, _prefix_hashes(tokens_of[i]), set(own_of[i])) for i, r in enumerate(reqs)]


def test_p5_design_scenarios():
    """The planner's output for the scenarios of design §3.3 / P5N Appendix A (production geometry) and the cost model
    estimates of §10: 32 x 34 cold -> one pk0 pass S 64 T 2048 (~1.62 s, serial 20.59 s); 32 sharing a 2K prefix in
    one step -> solo sp0 2048, then pk1 S 128 T 4096 at a = 2048 with shared tails (~4.6 s); 32 behind a 64-token
    template (hit < 128, c0 = 0) -> one pk0 pass S 128 T 4096 (the hit rows skip the shared block and pack with their
    writer); the mixed step -> pk0 S 64 T 2048 (24 + 8 dummies), the six 300-500 rows as the cheapest cut (B 4 + B 2,
    2.62 s modelled, where one B 8 pass would cost 2.95 s), 2 solo 2048 chunks."""
    ids = _Ids(2)
    serial = lambda reqs, plans: sum(  # noqa: E731
        pp.packed_pass_cost(p) for p in pp.solo_prefill_passes(plans, pp.order_prefill_requests(reqs, plans, 64))
    )
    reqs = [_prow(0, 34, ids.take(1), i) for i in range(32)]
    passes, plans = _p5(reqs)
    assert _desc(passes) == ["pk0 T=2048 S=64 B=32 (32 real) a=0"]
    assert sum(map(pp.packed_pass_cost, passes)) == pytest.approx(1.55 + 32 * 1.4e-3 + 0.0262)
    assert serial(reqs, plans) == pytest.approx(32 * (0.642 + 1.4e-3))  # 20.59 s (20.56 s measured, FR §3.7a)
    _check_call(reqs, plans, passes)

    shared = ids.take(32)
    reqs = [_prow(0, 2108, shared + ids.take(1), 0)] + [
        _prow(2048, 2110, shared + ids.take(1), i) for i in range(1, 32)
    ]
    passes, plans = _p5(reqs)
    assert _desc(passes) == [
        "solo sp0 C=2048 a=0 (row 0 chunk 0)",
        "pk1 T=4096 S=128 B=32 (32 real) a=2048 tails=shared",
    ]
    assert sum(map(pp.packed_pass_cost, passes)) == pytest.approx(4.61, abs=0.01) and serial(reqs, plans) > 23
    _check_call(reqs, plans, passes)

    tmpl = ids.take(1)
    reqs = [_prow(0, 90, tmpl + ids.take(1), 0)] + [_prow(64, 95, tmpl + ids.take(1), i) for i in range(1, 32)]
    passes, plans = _p5(reqs)
    assert all(p.c0 == 0 and p.paths == ("sp0",) for p in plans) and plans[1].w0 == 64
    assert _desc(passes) == ["pk0 T=4096 S=128 B=32 (32 real) a=0"]
    _check_call(reqs, plans, passes)

    lens = [40] * 24 + [300, 350, 400, 420, 480, 500] + [1500, 1500]
    reqs = [_prow(0, n, ids.take(_cdiv(n, 64)), i) for i, n in enumerate(lens)]
    passes, plans = _p5(reqs)
    assert _desc(passes) == [
        "pk0 T=2048 S=64 B=32 (24 real) a=0",
        "pk0 T=1024 S=512 B=2 (2 real) a=0",
        "pk0 T=2048 S=512 B=4 (4 real) a=0",
        "solo sp0 C=2048 a=0 (row 30 chunk 0)",
        "solo sp0 C=2048 a=0 (row 31 chunk 0)",
    ]
    one_b8 = pp.PrefillPass("pk0", [g for p in passes[1:3] for g in p.segments], 512, 8, 0)
    assert pp.packed_pass_cost(passes[1]) + pp.packed_pass_cost(passes[2]) < pp.packed_pass_cost(one_b8) - 0.3
    assert sum(map(pp.packed_pass_cost, passes)) == pytest.approx(7.33, abs=0.01) and serial(reqs, plans) > 22
    _check_call(reqs, plans, passes)


def _odd_hit_call(ids, *, writer=2200, readers=(50, 50), unrelated=(2148,), reader_lanes=None):
    """The R-E1 case (design §0.4, CP-P (vi), docs/p5_t64/scripts/review_dag_odd_hit.py) in one legal vLLM step:
    unrelated cold rows (W: (0, 2048) + a short tail chunk at 2048), a cold writer X whose first chunk ends at 2048,
    readers Y that share X's first 2112 tokens (33 full blocks, cached at X's allocation: a same-step hit) plus a few
    own tokens: s = w0 = 2112, c0 = 2048 < w0, so Y's global layers read block 32 (rows 2048-2111) from the cache,
    which X's second chunk writes. Returns ``(reqs, tokens, owns)`` in admission order."""
    rng = random.Random(writer * 7 + len(readers))
    rows = []
    for n in unrelated:
        t = [rng.randrange(5000) for _ in range(n)]
        b = ids.take(_cdiv(n, 64))
        rows.append((0, n, b, t, set(b)))
    tx = [rng.randrange(5000) for _ in range(writer)]
    bx = ids.take(_cdiv(writer, 64))
    rows.append((0, writer, bx, tx, set(bx)))
    for extra in readers:
        t = tx[:2112] + [rng.randrange(5000) for _ in range(extra)]
        own = ids.take(_cdiv(2112 + extra, 64) - 33)
        rows.append((2112, 2112 + extra, bx[:33] + own, t, set(own)))
    reqs = [_prow(s, e, b, lane, tokens=t) for lane, (s, e, b, t, _) in enumerate(rows)]
    return reqs, [r[3] for r in rows], [r[4] for r in rows]


def test_p5_odd_block_same_step_hit_cp_p_vi():
    """Review edit R-E1 / gate CP-P (vi): the readers depend on the writer's (2048, 256) chunk (it writes block 32 =
    [c0, w0)), so they run one level after it and leave the pk1 group they share with the unrelated row's (2048, 128)
    chunk (5 passes instead of 4: review note D5). The paged-cache emulator confirms every read; a hand-made plan with
    the pre-R-E1 read set [0, a) (readers packed with the unrelated tail chunk, in the writer's level) fails both the
    independent oracle and the emulator."""
    reqs, toks, owns = _odd_hit_call(_Ids(3))
    passes, plans = _p5(reqs)
    W, X, Y1, Y2 = plans
    assert _chunks(W) == [(0, 2048, "sp0"), (2048, 128, "sp1")] and _chunks(X) == [(0, 2048, "sp0"), (2048, 256, "sp1")]
    assert (Y1.start, Y1.w0, Y1.c0, _chunks(Y1)) == (2112, 2112, 2048, [(2048, 128, "sp1")]) and Y1 == Y2
    assert _desc(passes) == [
        "solo sp0 C=2048 a=0 (row 0 chunk 0)",
        "solo sp0 C=2048 a=0 (row 1 chunk 0)",
        "solo sp1 C=128 a=2048 (row 0 chunk 1)",
        "solo sp1 C=256 a=2048 (row 1 chunk 1)",
        "pk1 T=256 S=128 B=2 (2 real) a=2048 tails=shared",
    ]
    pass_of = _check_call(reqs, plans, passes)
    assert pass_of[(2, 0)] == pass_of[(3, 0)] == pass_of[(1, 1)] + 1
    rows = _emu_rows(reqs, toks, owns)
    for order in (passes, passes[:2] + passes[3:4] + passes[2:3] + passes[4:]):  # any order within a level
        PackedKVEmu(64).run_passes(rows, plans, order)
    # the pre-R-E1 schedule: level 1 = {W1, X1, Y1, Y2}; pk1 (S 128, a 2048) = {W1, Y1, Y2}, solo X1
    segs = {g.key: g for p in passes for g in p.segments}
    old = passes[:2] + [
        pp.PrefillPass("pk1", (segs[(0, 1)], segs[(2, 0)], segs[(3, 0)]), 128, 4, 2048, tails="distinct"),
        passes[3],
    ]
    with pytest.raises(AssertionError, match="reads block"):
        _oracle(reqs, plans, old)
    with pytest.raises(AssertionError, match="global"):
        PackedKVEmu(64).run_passes(rows, plans, old)


@pytest.mark.parametrize("seed", range(4))
def test_p5_odd_block_hits_randomized(seed):
    """Randomized R-E1 cases: writers of 2113-3000 tokens (first chunk ends at c0 = 2048), 1-12 readers with 1-300 own
    tokens, 0-4 unrelated rows of 2049-2300 tokens (pk1 chunks at 2048 in the readers' (S, a) groups), shuffled row
    order: the oracle, the pass structure and tables, and the emulator."""
    rng = random.Random(seed)
    ids = _Ids(10 + seed)
    for trial in range(25):
        unrelated = tuple(rng.randint(2049, 2300) for _ in range(rng.randint(0, 4)))
        readers = tuple(rng.randint(1, 300) for _ in range(rng.randint(1, 12)))
        reqs, toks, owns = _odd_hit_call(ids, writer=rng.randint(2113, 3000), readers=readers, unrelated=unrelated)
        perm = list(range(len(reqs)))
        rng.shuffle(perm)
        reqs = [
            _prow(reqs[k].start, reqs[k].seq_len, reqs[k].page_table.tolist()[: _cdiv(reqs[k].seq_len, 64)], j,
                  tokens=toks[k])
            for j, k in enumerate(perm)
        ]  # fmt: skip
        toks, owns = [toks[k] for k in perm], [owns[k] for k in perm]
        passes, plans = _p5(reqs)
        _check_call(reqs, plans, passes)
        PackedKVEmu(64).run_passes(_emu_rows(reqs, toks, owns), plans, passes)


def _burst_calls(L, ids, rng):
    """Three calls for row length ``L`` (each ``(reqs, tokens, owns)``): a cold burst of ``1 + L % 32`` rows of ``L``,
    ``L - 1``, ``L - 2`` tokens; a same-step prefix burst (a cold writer of ``L + 1 .. L + 64`` tokens and ``1 + L %
    7`` readers that hit its ``L // 64`` full blocks, odd and even counts, plus 1-150 own tokens); a mixed call
    (``L``, ``L // 2 + 1``, ``L // 3 + 1``, ``L // 5 + 1`` and a ``2048 + L``-token row)."""
    calls = []
    lens = [max(1, L - i % 3) for i in range(1 + L % 32)]
    toks = [[rng.randrange(5000) for _ in range(n)] for n in lens]
    blks = [ids.take(_cdiv(n, 64)) for n in lens]
    calls.append(([_prow(0, n, b, i, tokens=t) for i, (n, b, t) in enumerate(zip(lens, blks, toks))], toks, blks))
    lw = L + rng.randint(1, 64)
    tw = [rng.randrange(5000) for _ in range(lw)]
    bw = ids.take(_cdiv(lw, 64))
    h = L // 64  # the writer's full blocks the readers share (its block h-1 ends at 64 h <= L < lw)
    reqs, toks, owns = [_prow(0, lw, bw, 0, tokens=tw)], [tw], [bw]
    for i in range(1 + L % 7):
        t = tw[:L] + [5000 + rng.randrange(5000) for _ in range(rng.randint(1, 150))]  # own tokens differ from tw[L]
        own = ids.take(_cdiv(len(t), 64) - h)
        reqs.append(_prow(64 * h, len(t), bw[:h] + own, i + 1, tokens=t))
        toks.append(t)
        owns.append(own)
    perm = list(range(len(reqs)))
    rng.shuffle(perm)
    calls.append(([reqs[k] for k in perm], [toks[k] for k in perm], [owns[k] for k in perm]))
    lens = [L, L // 2 + 1, L // 3 + 1, L // 5 + 1, 2048 + L]
    toks = [[rng.randrange(5000) for _ in range(n)] for n in lens]
    blks = [ids.take(_cdiv(n, 64)) for n in lens]
    calls.append(([_prow(0, n, b, i, tokens=t) for i, (n, b, t) in enumerate(zip(lens, blks, toks))], toks, blks))
    for reqs, toks, owns in calls:  # every request carries exactly its tokens
        assert all(r.seq_len == len(t) for r, t in zip(reqs, toks))
    return calls


@pytest.mark.parametrize("part", range(4))
def test_p5_bursts_every_length_to_1100(part):
    """Every row length L = 1 .. 1100 (L = part mod 4 here) in the three calls of ``_burst_calls`` at the production
    geometry: the independent oracle, the pass structure, every pass table, and the paged-cache emulator."""
    ids, rng, seen = _Ids(100 + part), random.Random(part), Counter()
    for L in range(1 + part, 1101, 4):
        for reqs, toks, owns in _burst_calls(L, ids, rng):
            passes, plans = _p5(reqs)
            _check_call(reqs, plans, passes)
            PackedKVEmu(64).run_passes(_emu_rows(reqs, toks, owns), plans, passes)
            seen.update(p.kind for p in passes)
            seen["dummies"] += sum(p.dummies for p in passes)
            seen["odd_hits"] += sum(p.c0 < p.w0 for p in plans)
            seen["shared"] += sum(p.tails == "shared" for p in passes)
    assert min(seen[k] for k in ("solo", "pk0", "pk1", "dummies", "odd_hits", "shared")) > 0, seen


def _segs(b, rows, S, *, path="sp0", start=0, last=True):
    return [pp.PackSegment(i, 0, path, start, start + rows, S, last) for i in range(b)]


def _packed(b, rows, S, *, kind="pk0", start=0, last=True, tails=None, B=None):
    path = "sp1" if kind == "pk1" else "sp0"
    segs = _segs(b, rows, S, path=path, start=start, last=last)
    tails = tails or ("shared" if kind == "pk1" else None)
    return pp.PrefillPass(kind, segs, S, B or 1 << (b - 1).bit_length(), start, tails=tails)


def _solo(rows, bucket, *, start=0, last=True):
    path = "sp1" if start else "sp0"
    return pp.PrefillPass("solo", _segs(1, rows, None, path=path, start=start, last=last), bucket, 1, start)


def test_p5_cost_model():
    """packed_pass_cost grows with T (B), with the heads and with the pk1 start; distinct tails cost more than shared
    ones; for S <= 512 a packed pass of b >= 2 segments always beats its b solo chunks (design §3.3 step 4), pk0 and
    pk1 at every start; at S = 1024 it does not always (5 segments at B = 8). Custom cost tables and terms steer it."""
    c = pp.packed_pass_cost
    rows_of = {64: 40, 128: 100, 256: 200, 512: 400, 1024: 1000}
    for S, rows in rows_of.items():
        top = min(32, 8192 // S)
        for b in range(2, top + 1):
            p = _packed(b, rows, S)
            assert c(_packed(b, rows, S, last=False)) < c(p)  # heads
            if p.batch < top:
                assert c(p) < c(_packed(b, rows, S, B=2 * p.batch))  # more T
            if b < top:
                assert c(p) <= c(_packed(b + 1, rows, S)) + 1e-12  # one more segment never costs less
            solo = b * c(_solo(rows, max(128, S)))
            if S <= 512:
                assert c(p) < solo, (S, b)
            if S <= 512 or S == 1024:
                for a in (128, 2048, 8064, 24576):
                    q = {v: c(_packed(b, rows, S, kind="pk1", start=a, tails=v)) for v in VARIANTS}
                    assert q["shared"] < q["distinct"]
                    assert c(_packed(b, rows, S, kind="pk1", start=a + 128)) > q["shared"]
                    if S <= 512:
                        assert max(q.values()) < b * c(_solo(rows, max(128, S), start=a)), (S, b, a)
    assert c(_packed(5, 1000, 1024)) > 5 * c(_solo(1000, 1024))  # B = 8 for 5 segments: solo is cheaper
    assert c(_packed(4, 1000, 1024)) + c(_solo(1000, 1024)) < 5 * c(_solo(1000, 1024))  # ... the cheapest cut packs 4
    p = _packed(32, 40, 64)
    assert c(p) == pytest.approx(1.55 + 32 * 1.4e-3 + 0.0262)
    assert c(p, {128: 0.6, 2048: 1.0}) == pytest.approx(1.0 + 32 * 1.4e-3 + 0.0262)  # a {bucket: s} table
    assert c(p, lambda bucket, start: 0.0) == pytest.approx(32 * 1.4e-3 + 0.0262)  # a callable
    assert c(p, model=pp.PackedCostModel(head_s=0.0, mtp_fill_s=0.0, transpose_s_per_layer=0.0)) == pytest.approx(1.55)
    assert c(_solo(4000, 4096, start=0)) == pytest.approx(2.912 + 1.4e-3)
    assert c(_solo(100, 128, start=2048)) == pytest.approx(0.642 + 14 * 0.43e-6 * 2048 + 1.4e-3 + 39 * 0.65e-3)
    assert c(_solo(100, 128, start=2048, last=False)) == pytest.approx(c(_solo(100, 128, start=2048)) - 1.4e-3)
    ids = _Ids(5)
    reqs = [_prow(0, 34, ids.take(1), i) for i in range(32)]
    huge = pp.PackedCostModel(mtp_fill_s=100.0)  # packing never pays: everything solo
    assert all(p.kind == "solo" for p in _p5(reqs, pack_cost=huge)[0])
    flat = {b: 0.64 for b in BUCKETS}  # no per-token cost: pack as much as possible
    assert _desc(_p5(reqs, cost=flat)[0]) == ["pk0 T=2048 S=64 B=32 (32 real) a=0"]


def _compositions(n, top):
    """Every split of n ordered items into runs of 1 .. top (2^(n-1) at most)."""
    if n == 0:
        yield ()
        return
    for b in range(1, min(n, top) + 1):
        for rest in _compositions(n - b, top):
            yield (b,) + rest


@pytest.mark.parametrize("S", [64, 128, 256, 512, 1024])
def test_p5_cheapest_cut_matches_brute_force(S):
    """A level of n identical cold rows of one segment size, n = 1 .. 32: the planner's passes cost exactly the minimum
    over every split into solo chunks and packed passes of 2 .. B_max segments (all compositions for n <= 12, a
    recurrence over part sizes above), never more than the design's simple cut (passes of B_max, the remainder
    packed only if cheaper than its solo chunks), and the cut is the expected one at S = 1024."""
    rows = {64: 40, 128: 100, 256: 200, 512: 400, 1024: 1000}[S]
    top, ids, c = min(32, 8192 // S), _Ids(S), pp.packed_pass_cost
    part = {b: (c(_solo(rows, max(128, S))) if b == 1 else c(_packed(b, rows, S))) for b in range(1, top + 1)}
    best = {0: 0.0}
    for n in range(1, 33):
        best[n] = min(best[n - b] + part[b] for b in range(1, min(n, top) + 1))
    for n in range(1, 33):
        reqs = [_prow(0, rows, ids.take(_cdiv(rows, 64)), i) for i in range(n)]
        passes, plans = _p5(reqs)
        got = sum(map(c, passes))
        if n <= 12:
            assert best[n] == pytest.approx(min(sum(part[b] for b in comp) for comp in _compositions(n, top)))
        assert got == pytest.approx(best[n], abs=1e-9), (S, n, _desc(passes))
        cut = [top] * (n // top) + ([n % top] if n % top else [])
        simple = sum(min(part[b], b * part[1]) if b > 1 else part[1] for b in cut)
        assert got <= simple + 1e-9
        _check_call(reqs, plans, passes, tables=n % 5 == 0)
        if S == 1024 and n in (3, 5, 6, 7, 8):
            sizes = sorted((len(p.segments) for p in passes), reverse=True)
            assert sizes == {3: [2, 1], 5: [4, 1], 6: [4, 2], 7: [4, 2, 1], 8: [8]}[n], (n, _desc(passes))


def test_p5_pk1_tail_variants():
    """Review edit R-E2: rows behind prefixes of the same length but different content (cached by an earlier call) form
    one pk1 group at that start. A pass is ``shared`` iff all its segments have the same tail blocks; members are
    ordered by tail blocks, so rows of one prefix share passes; dummies copy segment 0 (the tables check). The plan is
    never costlier than running every segment solo."""
    rng, ids = random.Random(7), _Ids(7)
    variants = Counter()
    for trial in range(60):
        prefixes = [ids.take(20) for _ in range(rng.randint(1, 4))]  # 1280 tokens each, cached by an earlier call
        n = rng.randint(2, 32)
        reqs = []
        for i in range(n):
            e = 1280 + rng.randint(1, 120 if trial % 3 else 1000)
            reqs.append(_prow(1280, e, rng.choice(prefixes) + ids.take(_cdiv(e, 64) - 20), i))
        passes, plans = _p5(reqs)
        _check_call(reqs, plans, passes)
        variants.update(p.tails for p in passes if p.kind == "pk1")
        solo = sum(
            pp.packed_pass_cost(pp.PrefillPass("solo", (g,), plans[g.row].chunks[0].bucket, 1, g.start))
            for p in passes
            for g in p.segments
        )
        assert sum(map(pp.packed_pass_cost, passes)) <= solo + 1e-9
        for p in passes:
            if p.kind == "pk1":
                tails = {tuple(reqs[g.row].page_table[18:20].tolist()) for g in p.segments}
                assert (p.tails == "shared") == (len(tails) == 1)
    assert variants["shared"] > 0 and variants["distinct"] > 0, variants


def test_p5_shape_filter():
    """After the decode capture only warmed shapes run packed: a packed pass whose shape is not in ``allowed`` becomes
    one solo pass per segment with ``fallback`` = that shape (packing never refuses a call). pk1 keys name the tail
    variant (R-E2): a warm ``shared`` key does not admit a ``distinct`` pass. Solo shapes are never filtered. Allowing
    every packed_pass_shapes() key changes nothing."""
    ids = _Ids(9)
    reqs = []
    for first in (0, 20):  # two 2K prefixes, each written by a cold row of this call and hit by others (same step)
        pre = ids.take(32)
        reqs.append(_prow(0, 2108, pre + ids.take(1), first))
        reqs += [_prow(2048, 2110, pre + ids.take(1), i) for i in range(first + 1, first + (8 if first == 0 else 3))]
        if first == 0:
            reqs += [_prow(0, 30, ids.take(1), i) for i in range(8, 20)]  # a pk0 burst beside it
    passes, plans = _p5(reqs)
    assert _desc(passes) == [
        "solo sp0 C=2048 a=0 (row 0 chunk 0)",
        "pk0 T=1024 S=64 B=16 (12 real) a=0",
        "solo sp0 C=2048 a=0 (row 20 chunk 0)",
        "pk1 T=2048 S=128 B=16 (11 real) a=2048 tails=distinct",  # one distinct pass beats two shared ones here
    ]
    every = set(pp.packed_pass_shapes(block_size=64))
    assert _p5(reqs, allowed=every)[0] == passes
    assert _p5(reqs, allowed=every | {("sp0", 2048)})[0] == passes
    for drop in (("pk0", 1024, 64), ("pk1", 2048, 128, "distinct")):
        got = _p5(reqs, allowed=every - {drop})[0]
        packed = next(p for p in passes if p.shape == drop)
        fb = [p for p in got if p.fallback is not None]
        assert [p.segments[0] for p in fb] == list(packed.segments) and all(p.fallback == drop for p in fb)
        for q in fb:
            g = q.segments[0]
            assert q.kind == "solo" and q.shape == (packed.path, plans[g.row].chunks[g.chunk_index].bucket)
        assert [p for p in got if p.fallback is None] == [p for p in passes if p.shape != drop]
        _check_call(reqs, plans, got)
    # the shared key alone does not admit the distinct pass
    only_shared = {k for k in every if not (k[0] == "pk1" and k[3] == "distinct")}
    assert sum(p.fallback == ("pk1", 2048, 128, "distinct") for p in _p5(reqs, allowed=only_shared)[0]) == 11
    nothing = _p5(reqs, allowed=set())[0]  # no packed shape warmed: every chunk solo, in level order
    assert all(p.kind == "solo" for p in nothing) and sum(p.fallback is not None for p in nothing) == 23
    assert next(p for p in nothing if p.segments[0].key == (0, 0)).fallback is None  # a solo shape is not filtered
    _check_call(reqs, plans, nothing)


def test_p5_knobs_and_validation():
    ids = _Ids(13)
    reqs = [_prow(0, 34, ids.take(1), i) for i in range(32)]
    assert all(p.kind == "solo" for p in _p5(reqs, max_seg=0)[0])
    assert all(p.kind == "solo" for p in _p5(reqs, max_batch=1)[0])
    assert all(p.kind == "solo" for p in _p5(reqs, max_tokens=127)[0])
    tiny, plans = _p5(reqs, max_tokens=128)  # S 64, B 2 only
    assert [p.shape for p in tiny] == [("pk0", 128, 64)] * 16
    _check_call(reqs, plans, tiny, max_tokens=128)
    p16, plans = _p5(reqs, max_batch=16)
    assert [p.shape for p in p16] == [("pk0", 1024, 64)] * 2
    s128, plans = _p5(reqs, seg_buckets=(128, 256))  # gate G15's S_min = 128 fallback
    assert [p.shape for p in s128] == [("pk0", 4096, 128)]
    _check_call(reqs, plans, s128, seg_buckets=(128, 256))
    shared = ids.take(32)
    hits = [_prow(2048, 2100, shared + ids.take(1), i) for i in range(4)]
    assert all(p.kind == "solo" for p in _p5(hits, pk1=False)[0])
    assert all(p.kind == "solo" for p in _p5(hits, sp1_seg_buckets=())[0])
    notail, plans = _p5(hits, swa_tail=0)  # no sliding window: nothing to gather, a pk1 pass counts as shared
    assert [p.shape for p in notail] == [("pk1", 512, 128, "shared")]
    t = pp.pass_tables(notail[0], hits, plans, block_size=64, sdpa_width=SDPA_W, max_positions=POS, swa_tail=0)
    assert t.tail is None and t.sdpa.shape == (4, SDPA_W)
    assert pp.plan_prefill_passes([], [], block_size=64) == []
    for kw, err in (
        (dict(block_size=0), ValueError),
        (dict(max_seg=-1), ValueError),
        (dict(max_tokens=0), ValueError),
        (dict(max_batch=0), ValueError),
        (dict(pk1=1), TypeError),
        (dict(seg_buckets=(96,)), ValueError),
        (dict(seg_buckets=(128, 64)), ValueError),
        (dict(swa_tail=100), ValueError),
        (dict(swa_tail=-64), ValueError),
        (dict(cost=3.0), TypeError),
    ):
        args = dict(block_size=64)
        args.update(kw)
        with pytest.raises(err):
            pp.plan_prefill_passes(reqs, _plans(reqs), **args)
    with pytest.raises(ValueError, match="plans"):
        pp.plan_prefill_passes(reqs, _plans(reqs)[:3], block_size=64)
    with pytest.raises(ValueError, match="does not match"):
        pp.plan_prefill_passes(reqs[:1], _plans(hits[:1]), block_size=64)


def test_p5_pass_helpers():
    """pass_rows / pass_tokens (the packed layout of per-row host inputs such as the MTP next tokens), pass_chunks
    (R-E12 re-bucketing), solo_prefill_passes (packing off: today's row order as passes), packed_pass_shapes, and the
    pass-table errors."""
    ids = _Ids(17)
    reqs = [_prow(0, 40, ids.take(1), 0), _prow(0, 64, ids.take(1), 1), _prow(0, 1000, ids.take(16), 2)]
    passes, plans = _p5(reqs)
    assert _desc(passes) == ["pk0 T=128 S=64 B=2 (2 real) a=0", "solo sp0 C=1024 a=0 (row 2 chunk 0)"]
    p = passes[0]
    assert [c.bucket for c in pp.pass_chunks(p, plans)] == [64, 64] and plans[0].chunks[0].bucket == 128
    assert pp.pass_chunks(passes[1], plans) == [plans[2].chunks[0]]  # a solo pass keeps its chunk
    v = pp.pass_rows(p, [list(range(40)), torch.arange(100, 164)], -1)
    assert v.dtype == torch.int32 and v.tolist() == list(range(40)) + [-1] * 24 + list(range(100, 164))
    with pytest.raises(ValueError, match="value rows"):
        pp.pass_rows(p, [list(range(40))], 0)
    with pytest.raises(ValueError, match="integer values"):
        pp.pass_rows(p, [list(range(39)), list(range(64))], 0)
    with pytest.raises(ValueError, match="integer values"):
        pp.pass_rows(p, [torch.zeros(40), list(range(64))], 0)
    assert torch.equal(
        pp.pass_tokens(passes[1], reqs, 0), torch.cat([reqs[2].tokens, torch.zeros(24, dtype=torch.int32)])
    )
    with pytest.raises(ValueError, match="SDPA page tables"):
        pp.pass_sdpa_tables(p, reqs, 64, SDPA_W)
    with pytest.raises(ValueError, match="SWA tails"):
        pp.pass_tail_blocks(p, reqs, 64)
    with pytest.raises(ValueError, match="plan"):
        pp.pass_chunks(p, plans[::-1])
    with pytest.raises(ValueError, match="multiple of the block size"):
        pp.pass_fill_table(pp.PrefillPass("pk0", p.segments, 96, 2, 0), reqs, plans, 64)
    with pytest.raises(ValueError, match="outside"):
        pp.pass_tokens(p, reqs[:1], 0)
    shared = ids.take(32)
    hits = [_prow(2048, 2100, shared + ids.take(1), 0), _prow(2048, 2100, ids.take(33), 1)]
    hp, hplans = _p5(hits)
    assert [q.shape for q in hp] == [("pk1", 256, 128, "distinct")]
    forged = pp.PrefillPass("pk1", hp[0].segments, 128, 2, 2048, tails="shared")
    with pytest.raises(ValueError, match="shared tails"):
        pp.pass_tables(forged, hits, hplans, block_size=64, sdpa_width=SDPA_W, max_positions=POS)
    with pytest.raises(ValueError, match="cover"):
        pp.pass_sdpa_tables(hp[0], hits, 64, 8)
    assert pp.tail_variant(torch.tensor([[3, 4], [3, 4]])) == "shared"
    assert pp.tail_variant(torch.tensor([[3, 4], [3, 5]])) == "distinct"
    with pytest.raises(ValueError):
        pp.tail_variant(torch.tensor([3, 4]))
    # packing off: one solo pass per chunk in today's writer-first row order
    long_ = [_prow(0, 9000, ids.take(141), 0), _prow(0, 50, ids.take(1), 1)]
    lplans = _plans(long_)
    solo = pp.solo_prefill_passes(lplans, [1, 0])
    assert [(q.segments[0].key, q.shape) for q in solo] == [((1, 0), ("sp0", 128)), ((0, 0), ("sp0", 8192)),
                                                          ((0, 1), ("sp1", 1024))]  # fmt: skip
    assert pp.solo_prefill_passes(lplans) == sorted(solo, key=lambda q: q.segments[0].key)
    _oracle(long_, lplans, solo)
    with pytest.raises(ValueError, match="permutation"):
        pp.solo_prefill_passes(lplans, [0, 0])
    shapes = pp.packed_pass_shapes(block_size=64)
    assert (
        len(shapes) == 22 + 2 * 17 and shapes[0] == ("pk0", 128, 64) and shapes[-1] == ("pk1", 8192, 1024, "distinct")
    )
    assert pp.packed_pass_shapes(block_size=64, max_seg=256, pk1=False, max_tokens=4096) == tuple(
        ("pk0", B * S, S) for S in (64, 128, 256) for B in (2, 4, 8, 16, 32) if B * S <= 4096
    )


class FakeVllmEngine:
    """:class:`FakeVllmScheduler`'s prefill rules (hits on admission only, capped at ``num_tokens - 1``; full blocks
    cached at allocation, so a later-admitted request hits blocks an earlier row computes in the same step; chunk =
    ``min(remaining, threshold, budget left)``; partial prefills first; <= 32 rows) plus what a long-running server adds
    (vLLM 0.26 v1 scheduler and KV cache manager):

    * a decode step after every prefill step: one token per decoding request; a block is cached once it is full;
    * finished requests free their blocks;
    * a fixed block pool with LRU reuse: a freed block keeps its hash (a later request may hit it) until it is
      reallocated, which evicts it (the emulator forgets its content);
    * preemption: a running request that needs blocks the pool does not have preempts the newest decoding request (an
      admission instead waits), and a random decoding request is preempted now and then. A preempted request frees its
      blocks and goes back to the front of the waiting queue with prompt + generated tokens; its re-prefill hits
      whatever of its blocks is still cached."""

    def __init__(self, emu, *, bs, num_blocks, budget, threshold, rng, preempt=0.15):
        self.emu, self.bs, self.budget, self.threshold, self.rng, self.preempt = (
            emu,
            bs,
            budget,
            threshold,
            rng,
            preempt,
        )
        self.free = OrderedDict(
            (b, None) for b in range(1, num_blocks)
        )  # refcount-0 blocks, least recently freed first
        self.ref = Counter()
        self.cached, self.hash_of = {}, {}  # block hash -> block, block -> its hash (the cached copy only)
        self.waiting, self.running = deque(), []
        self.stats = Counter()

    def add(self, tokens, max_new):
        h = _prefix_hashes(tokens)
        self.waiting.append(SimpleNamespace(tokens=list(tokens), h=h, blocks=[], own=set(), computed=0, state="wait",
                                            max_new=max_new, generated=0, preempted=False))  # fmt: skip

    @property
    def busy(self):
        return bool(self.waiting or self.running)

    def _bh(self, r, k):
        return r.h[(k + 1) * self.bs - 1]

    def _alloc(self, r):
        b, _ = self.free.popitem(last=False)
        h = self.hash_of.pop(b, None)
        if h is not None:
            del self.cached[h]
            self.stats["evicted"] += 1
        self.emu.evict(b)
        self.ref[b] = 1
        r.blocks.append(b)
        r.own.add(b)

    def _cache(self, r, upto):
        for k in range(upto // self.bs):
            h = self._bh(r, k)
            if h not in self.cached:
                self.cached[h] = r.blocks[k]
                self.hash_of[r.blocks[k]] = h

    def _release(self, r):
        for b in r.blocks:
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free[b] = None
        r.blocks, r.own = [], set()

    def _preempt(self, r):
        self.running.remove(r)
        self._release(r)
        r.computed, r.state, r.preempted = 0, "wait", True
        self.waiting.appendleft(r)
        self.stats["preempted"] += 1

    def _make_room(self, n, keep):
        """Free ``n`` blocks by preempting the newest decoding requests not in ``keep``; False if impossible."""
        while len(self.free) < n:
            victims = [r for r in self.running if r.state == "decode" and all(r is not k for k in keep)]
            if not victims:
                return False
            self._preempt(victims[-1])
            self.stats["preempted_for_blocks"] += 1
        return True

    def _extend(self, r, n):
        end = r.computed + n
        while len(r.blocks) < _cdiv(end, self.bs):
            self._alloc(r)
        self._cache(r, end)
        return r, end

    def schedule(self):
        """One prefill step: ``[(request, end)]`` (may be empty)."""
        victims = [r for r in self.running if r.state == "decode"]
        if victims and self.rng.random() < self.preempt:
            self._preempt(self.rng.choice(victims))
        budget, rows = self.budget, []
        cap = self.threshold if self.threshold > 0 else 1 << 30
        for r in list(self.running):
            if r.state != "prefill":
                continue
            if budget <= 0 or len(rows) >= 32:
                break
            n = min(len(r.tokens) - r.computed, cap, budget)
            if not self._make_room(_cdiv(r.computed + n, self.bs) - len(r.blocks), [q for q, _ in rows] + [r]):
                break
            rows.append(self._extend(r, n))
            budget -= n
        while self.waiting and budget > 0 and len(rows) < 32:
            r = self.waiting[0]
            hit = 0
            while hit < (len(r.tokens) - 1) // self.bs and self._bh(r, hit) in self.cached:
                hit += 1
            hits = [self.cached[self._bh(r, k)] for k in range(hit)]
            n = min(len(r.tokens) - hit * self.bs, cap, budget)
            need = _cdiv(hit * self.bs + n, self.bs) - hit + sum(self.ref[b] == 0 for b in hits)
            if len(self.free) < need:
                break  # an admission waits for blocks (vLLM preempts only to make room for running requests)
            self.waiting.popleft()
            for b in hits:
                if self.ref[b] == 0:
                    del self.free[b]
                self.ref[b] += 1
            r.blocks, r.own, r.computed, r.state = hits, set(), hit * self.bs, "prefill"
            self.stats["readmitted_with_hit"] += r.preempted and hit > 0
            self.running.append(r)
            rows.append(self._extend(r, n))
            budget -= n
        return rows

    def _next_token(self, r):
        """The request's next token (the host argmax of its last row's logits), or its end."""
        if r.generated >= r.max_new:
            self.running.remove(r)
            self._release(r)
            r.state = "done"
            self.stats["finished"] += 1
            return
        t = self.rng.randrange(5000)
        r.tokens.append(t)
        r.h.append((r.h[-1] * 1099511628211 + t + 1) & ((1 << 61) - 1))  # _prefix_hashes, one more token
        r.generated += 1
        r.state = "decode"

    def commit(self, rows):
        for r, end in rows:
            r.computed = end
            if end == len(r.tokens):
                self._next_token(r)

    def decode(self):
        for r in [r for r in self.running if r.state == "decode"]:
            if r.state != "decode":
                continue  # preempted for another request's block in this step
            n = r.computed  # the newest token's position: its KV is written now
            if n // self.bs >= len(r.blocks):
                if not self._make_room(1, [r]):
                    self._preempt(r)
                    continue
                self._alloc(r)
            b = r.blocks[n // self.bs]
            self.emu.decode_write(b, n, r.h[n], r.own)
            r.computed = n + 1
            if r.computed % self.bs == 0:
                self._cache(r, r.computed)
                self.emu.complete.add(b)
            self.stats["decode_tokens"] += 1
            self._next_token(r)


ENGINE_CASES = {
    # production: bs 64, A = 128, budget = threshold = 8064, span cap 8192, default packing knobs. The pools are small
    # enough that requests get preempted for blocks and freed blocks are evicted.
    "production": dict(bs=64, budget=8064, threshold=8064, cap=8192, blocks=420, plan={}),
    "pk1_off": dict(bs=64, budget=8064, threshold=8064, cap=8192, blocks=420, plan=dict(pk1=False)),
    "seg256_t4096": dict(bs=64, budget=8064, threshold=8064, cap=8192, blocks=420, plan=dict(max_seg=256,
                                                                                             max_tokens=4096)),
    "partial_warmup": dict(bs=64, budget=8064, threshold=8064, cap=8192, blocks=420, plan="half"),
    "small_cap": dict(bs=64, budget=777, threshold=300, cap=512, blocks=420, plan={}),
    "bs32": dict(bs=32, budget=2048, threshold=1024, cap=1024, blocks=840, plan={}),
    "unchunked_32k": dict(bs=64, budget=1 << 30, threshold=0, cap=8192, blocks=900, plan={}, long=True),
}  # fmt: skip


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("case", sorted(ENGINE_CASES))
def test_p5_emu_vllm_engine(case, seed):
    """vLLM-like serving traffic, every prefill call planned with packing and run pass by pass in the paged-cache
    emulator: fresh prompts shared by 2-4 requests of one wave (same-step hits; the first waves' shared prefixes have
    2112 = 33, 704 = 11, 2176, 1088 = 17 and 2240 = 35 blocks: odd counts make c0 < w0, review edit R-E1, and at 2112
    the writer's chunk boundary is c0), two system prompts of 1280 tokens but different content (pk1 groups with
    shared and distinct tails), multi-turn and duplicate prompts, short cold bursts (pk0), long prompts admitted
    behind others (unaligned chunk ends), decode steps, preemption re-prefills and evicted blocks. Every pass also meets
    the oracle, the structure rules and the table checks; the traffic must contain each feature the case is about."""
    cfg = ENGINE_CASES[case]
    bs, cap = cfg["bs"], cfg["cap"]
    rng = random.Random(1000 * seed + sum(map(ord, case)))
    emu = PackedKVEmu(bs, cap=cap)
    eng = FakeVllmEngine(
        emu, bs=bs, num_blocks=cfg["blocks"], budget=cfg["budget"], threshold=cfg["threshold"], rng=rng
    )
    plan_kw = dict(cfg["plan"]) if cfg["plan"] != "half" else {}
    lim = dict(max_tokens=plan_kw.get("max_tokens", min(8192, cap)))
    lim.update({k: v for k, v in plan_kw.items() if k in ("max_seg", "pk1")})
    if cfg["plan"] == "half":
        every = pp.packed_pass_shapes(block_size=bs, max_tokens=lim["max_tokens"])
        plan_kw["allowed"] = set(rng.sample(every, len(every) // 2))
    systems = [[rng.randrange(5000) for _ in range(n)] for n in (64, 200, 1280, 1280, 2112, 2600)]
    history, seen = [], Counter()

    def suffix(lo, hi):
        return [rng.randrange(5000) for _ in range(rng.randint(lo, hi))]

    for wave in range(10):
        fresh = suffix(*(2 * [(2112, 704, 2176, 1088, 2240)[wave]] if wave < 5 else (130, 2600)))
        for _ in range(rng.randint(2, 4)):
            history.append(fresh + suffix(1, 400))
            eng.add(history[-1], rng.randint(1, 24))
        if wave < 2:  # both 1280-token system prompts twice: same-step readers at a = 1280 with different tails
            for sys_ in (systems[2], systems[3], systems[2], systems[3]):
                eng.add(sys_ + suffix(1, 100), rng.randint(1, 24))
        for _ in range(rng.randint(0, 4)):
            eng.add(rng.choice(systems) + suffix(1, 300), rng.randint(1, 24))
        old = [h for h in history if len(h) < 6000]
        for _ in range(rng.randint(0, 2)):
            history.append(list(rng.choice(old)) + (suffix(1, 600) if rng.random() < 0.6 else []))
            eng.add(history[-1], rng.randint(1, 24))
        for _ in range(rng.randint(0, 6)):
            eng.add(suffix(1, 300), rng.randint(1, 24))
        if wave in (1, 3):  # admitted behind the others: its first chunk ends where the budget runs out
            eng.add(suffix(9000, 14000) if cfg["budget"] < 1 << 20 else suffix(9000, 9000), rng.randint(1, 8))
        if cfg.get("long") and wave % 2 == 0:
            eng.add(suffix(8200, 30000), rng.randint(1, 8))
        while eng.busy:
            rows = eng.schedule()
            if rows:
                lanes = rng.sample(range(32), len(rows))
                call = []
                for (r, end), lane in zip(rows, lanes):
                    req = _rows_of(r.tokens, r.computed, end, r.blocks, width=32768 // bs, lane=lane)
                    call.append((req, r.h[:end], set(r.own)))
                rng.shuffle(call)
                plans, passes = emu.run_packed_call(call, align=PROD_A, **plan_kw)
                reqs = [c[0] for c in call]
                _check_call(reqs, plans, passes, bs=bs, tables=False, **lim)
                for p in passes:
                    _check_pass_tables(p, reqs, plans, bs=bs, width=emu.width)
                writes = {}
                for i, (r, pl) in enumerate(zip(reqs, plans)):
                    for k, c in enumerate(pl.chunks):
                        for L in range(max(pl.w0, c.start) // bs, _cdiv(c.end, bs)):
                            writes[int(r.page_table[L])] = (i, c.start)
                for j, (r, pl) in enumerate(zip(reqs, plans)):
                    seen["rows"] += 1
                    seen["resumed"] += r.start > 0
                    seen["unaligned"] += r.start % bs != 0
                    seen["multi_chunk"] += len(pl.chunks) > 1
                    seen["long"] += r.seq_len > 8192
                    ro = [writes.get(int(b)) for b in r.page_table[: pl.read_only_blocks].tolist()]
                    seen["same_step_hit"] += pl.has_sp1 and any(w is not None and w[0] != j for w in ro)
                    odd = [writes.get(int(b)) for b in r.page_table[pl.c0 // bs : pl.w0 // bs].tolist()]
                    seen["odd_hit_same_step"] += pl.has_sp1 and any(w is not None and w[0] != j for w in odd)
                    seen["odd_hit_boundary_at_c0"] += pl.has_sp1 and any(
                        w is not None and w[0] != j and w[1] == pl.c0 for w in odd
                    )
                for p in passes:
                    seen[p.kind] += 1
                    seen["tails_" + str(p.tails)] += p.kind == "pk1"
                    seen["dummies"] += p.dummies
                    seen["fallbacks"] += p.fallback is not None
                seen["calls"] += 1
                seen["multi_row_calls"] += len(rows) > 1
                eng.commit(rows)
            eng.decode()
            seen["steps"] += 1
            assert seen["steps"] < 5000
    seen.update(eng.stats)
    need = ["pk0", "solo", "multi_row_calls", "same_step_hit", "odd_hit_same_step", "resumed", "preempted",
            "readmitted_with_hit", "evicted", "decode_tokens"]  # fmt: skip
    need += {  # an unbounded budget never ends a chunk inside a block: no unaligned starts there
        "production": ["pk1", "tails_shared", "tails_distinct", "odd_hit_boundary_at_c0", "dummies", "unaligned",
                       "multi_chunk"],
        "pk1_off": ["unaligned"],
        "seg256_t4096": ["pk1", "tails_distinct", "unaligned"],
        "partial_warmup": ["fallbacks", "unaligned"],
        "small_cap": ["pk1", "unaligned"],
        "bs32": ["pk1", "tails_distinct", "odd_hit_boundary_at_c0", "unaligned", "multi_chunk"],
        "unchunked_32k": ["multi_chunk", "long", "pk1", "tails_distinct", "odd_hit_boundary_at_c0"],
    }[case]  # fmt: skip
    assert all(seen[k] > 0 for k in need), f"{case}: no {[k for k in need if not seen[k]]} in {dict(seen)}"
    if case == "pk1_off":
        assert seen["pk1"] == 0
    if case != "partial_warmup":
        assert seen["fallbacks"] == 0
    print(f"[p5 emu] {case} seed={seed}: {dict(sorted(seen.items()))}, {emu.writes} row writes")


def test_p5_shapes_match_model_config(monkeypatch):
    """The planner can emit exactly the packed shapes the generator warms (``MotifTTConfig.packed_prefill_shapes()``,
    C1a) for the default config and the knob variants (max seg / tokens / pk1, the span cap), so after the capture no
    planned pass falls back for want of warmup: every pass of random traffic planned with the config's knobs has a
    warmed shape."""
    mc = pytest.importorskip("models.demos.motif3.tt.model_config")
    for v in ("MOTIF3_PACKED_PREFILL_MAX_SEG", "MOTIF3_PACKED_PREFILL_MAX_TOKENS", "MOTIF3_PACKED_PREFILL_PK1",
              "MOTIF3_MAX_MODEL_LEN", "MOTIF3_PREFILL_MAX_BUCKET", "MOTIF3_NUM_LAYERS"):  # fmt: skip
        monkeypatch.delenv(v, raising=False)
    raw = json.load(open(Path(mc.DEFAULT_HF_META_DIR) / "config.json"))
    cfgs = [
        mc.MotifTTConfig.from_hf_config(str(mc.DEFAULT_HF_META_DIR)),
        mc.MotifTTConfig.from_hf_config(str(mc.DEFAULT_HF_META_DIR), max_model_len=4096),
        mc.MotifTTConfig.from_hf_config(str(mc.DEFAULT_HF_META_DIR), max_model_len=6144),
    ]
    for kw in (dict(packed_prefill_max_seg=256, packed_prefill_pk1=False, packed_prefill_max_tokens=4096),
               dict(packed_prefill_max_seg=64), dict(packed_prefill_max_tokens=2048)):  # fmt: skip
        cfgs.append(mc.MotifTTConfig.from_settings(api.GeneratorSettings(**kw), mesh_shape=(4, 8), hf_config=raw))
    for cfg in cfgs:
        knobs = dict(
            max_seg=cfg.pack_max_seg,
            max_tokens=cfg.pack_tokens_cap,
            pk1=cfg.pack_pk1,
            seg_buckets=cfg.pack_seg_buckets,
            sp1_seg_buckets=cfg.pack_sp1_seg_buckets,
        )
        warm = cfg.packed_prefill_shapes()
        assert pp.packed_pass_shapes(block_size=cfg.kv_block_size, **knobs) == warm, cfg.describe()
        rng, ids = random.Random(cfg.pack_tokens_cap), _Ids(3)
        for trial in range(40):
            shared = ids.take(32)
            reqs = [_prow(0, rng.randint(1, 1100), [], i) for i in range(rng.randint(1, 24))]
            reqs = [_prow(0, r.seq_len, ids.take(_cdiv(r.seq_len, 64)), r.lane) for r in reqs]
            reqs += [
                _prow(2048, 2048 + rng.randint(1, 900), shared + ids.take(15), 24 + i) for i in range(rng.randint(0, 8))
            ]
            plans = [cfg.plan_prefill_row(r.start, r.seq_len) for r in reqs]
            passes = pp.plan_prefill_passes(reqs, plans, block_size=cfg.kv_block_size, allowed=set(warm), **knobs)
            assert all(p.fallback is None for p in passes) and {p.shape for p in passes if p.is_packed} <= set(warm)
