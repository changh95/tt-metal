# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tests for ``tt/prefill_plan.py`` (docs/features/FEATURES_DESIGN.md §2.2, §3.1, §3.7.2, §5.1).

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
  valid KV, every write lands in a block the row allocated, and no completed block is ever rewritten.

Pure torch (no ttnn). Run device-hidden::

    scripts/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib -q \
        models/demos/motif3/tests/unit/test_prefill_plan.py
"""

import itertools
import json
import os
import random
import subprocess
import sys
from collections import deque
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
