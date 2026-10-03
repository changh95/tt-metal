# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resumed / chunked prefill with prefix caching and the MTP KV-only fill, at generator level (features design §3.7,
D7-D9, §5.3 CP-H / CP-C / CP-L / CP-X / CP9; work package 4).

Host tests (``-k "cpu or host"``; no device, run through ``scripts/hostrun.sh``): the generator's prefill
orchestration against an emulated paged cache whose "KV" at position ``p`` is a hash of the token prefix ``[0, p]``
(G1). The fake model reads and writes that cache only through the chunk tables the generator built
(``attention.ChunkHostTables``: fill table, sp1 SDPA table, SWA tail blocks), so a wrong table, a wrong token slice, a
wrong row order (same-step hits), a write into a shared block or a wrong MTP next token shows up as a wrong logit or a
failed assertion. Schedules come from WP1's vLLM-like scheduler (``test_prefill_plan.FakeVllmScheduler``: prefix hits
on admission, full blocks cached at allocation, chunk budgets, shuffled rows, new lanes per call). Also: the warmed
``(path, bucket)`` set and its refusal rule after the decode capture, the sp1 bucket cap, the decoder / model
threading of ``chunk=`` / ``kv_write=``, the KV-R decode wiring, the MTP part's size estimate.

Device tests (the real 53-layer model from the TT cache, every weight from the cache; one generator boot shared by the
module: serving pool, chunked prefill + prefix caching -> KV-R decode, MTP on -> its cache filled during prefill; ONE
captured decode trace, the serving spec trace (``all_split``), as in serving: see ``MULTI_TRACE_NOTE`` for why a
second one is not captured)::

    scripts/devrun.sh -t 3600 -n wp4_cp -- python -m pytest models/demos/motif3/tests/test_resumed_prefill.py \
        -k "not cpu and not host" -s -p no:cacheprovider --timeout=0

* **CP-L (i)** (first: it compiles the draft-1 32K single-shot programs with no trace alive, then re-captures): a
  ~32K-token needle prompt chunked in vLLM steps of the recommended budget (``span cap - A``) and of 4096, vs the
  draft-1 single-shot prefill (bucket 32768); NLL of the last 2048 positions, top-1 agreement of the last 512, the
  needle answer and 64 greedy decode tokens per path (traced); TT's own floor on the same rows: the single shot
  repeated and the last 512 positions teacher-forced through the draft-1 decode numerics (the agreement bar is the
  weaker of the two), the budget schedule repeated (reported).
* **CP-L (ii)** (``MOTIF3_CP_LAYERS=4``; data from ``test_host_build_long_reference``, ``MOTIF3_BUILD_LONG_REF=1``):
  the truncated model (layers 0-3) on the needle prompt and its first 16384 tokens, single shot and chunked, vs the
  CPU reference prefix model: residual streams after layer 3 at sampled positions::

      MOTIF3_BUILD_LONG_REF=1 scripts/hostrun.sh -t 3600 -- python -m pytest -p no:cacheprovider -q -s \
          models/demos/motif3/tests/test_resumed_prefill.py -k build_long_reference          # ~7 min of CPU, once
      MOTIF3_CP_LAYERS=4 scripts/devrun.sh -t 1800 -n wp4_cpl2 -- python -m pytest \
          models/demos/motif3/tests/test_resumed_prefill.py -k cp_l_truncated -s -p no:cacheprovider --timeout=0

* **CP-H**: the 6 C2 prompts: (i) cold prefill (teacher-forced logits on every row) + 32 greedy decode tokens; (ii) the
  same prompt again with ``start = floor((S - 1) / 64) * 64`` on the first run's blocks, on another DP row, next to a
  cold repeat; (iii) ``P + X`` cold then ``P + Y`` (= the C2 prompt) with a hit on ``P``: teacher-forced suffix rows
  vs the fp32 golden.
* **CP-C**: the C2 prompts in forced vLLM chunk schedules (128-aligned, 200, 333, growing multiples of 64) through
  ``prefill_forward_batch`` with explicit starts; teacher-forced logits on every row vs the fp32 golden and cold TT.
* **CP-X**: request A (lane 0, DP row 0) prefills and decodes 200 tokens; request B = A's prompt + A's answer + a new
  turn on lane 8 (DP row 1) hits the decode-written blocks: as close to a cold B as a hit on prefill-written blocks
  (the sp1 floor) with KV-R, through the spec trace and through the plain ``all`` path (the non-speculating
  production decode; eager, no trace alive); and the negative control (§3.4): the same with A decoded in the draft-1
  ``row`` mode (eager, no trace alive) -> B's hit reads stale KV.
* **MTP fill**: the MTP cache an sp1 chunk fills (a vLLM chunk at 512, a prefix hit at 512) vs the cold sp0 fill.
* **CP9**: after the capture, a randomized mix of cold / hit / chunked / resumed / same-step rows interleaved with
  traced decode steps: the program cache does not grow and a decode replay equals the eager step bitwise.
* **TTFT** (report only): cold rows of 128 ... 32000 tokens and prefix hits behind 2K / 8K / 30K cached contexts.

Pass bars. At full depth single rows are chaotic: MoE top-8 near ties flip under any numerical perturbation (run-to-run
prefill nondeterminism, FULL_MODEL_VALIDATION §5.1; the draft-1 decode path; the sp1 path's bfp8 cache keys), so the
design's (§5.3) single-row bars are reported, and what is asserted is either pooled over many rows or relative to TT's
own floor measured in the same test. Every logits row a metric uses must be finite and below ``LOGIT_ABS_MAX``
(``assert_sane``): a NaN / Inf row, or one of finite device garbage, fails; it never drops out of a metric. Asserted
vs reported (2026-10-03; the relaxed bars are an acceptance decision for the lead):

=========== ================================================ =========================================================
test        design bar (§5.3), reported                      asserted instead
=========== ================================================ =========================================================
CP-L (i)    last-512 top-1 >= 0.98 (confident rows), NLL     top-1 >= the weaker draft-1 floor at 32K (single shot
            within +-1 %                                     repeated, teacher-forced decode) - 0.05; NLL <= single
                                                             shot + 1 % (one-sided); needle; greedy answer
CP-L (ii)   chunked vs single-shot TT streams >= 0.999       every path vs the CPU reference >= 0.99, chunked no
                                                             further from it than the single shot (0.001)
CP-H (ii)   last-token PCC >= 0.999                          same argmax unless margin < 0.5; greedy streams; plus the
                                                             pooled bars of CP-H (iii) and CP-C (asserted as designed)
CP-X        last-token PCC >= 0.999 (CP-H tolerances)        argmax rule; suffix median >= the sp1 floor - 0.003;
                                                             greedy streams; the negative control detected
=========== ================================================ =========================================================
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from models.demos.motif3.tt import generator_api as api
from models.demos.motif3.tt import prefill_plan as PP
from models.demos.motif3.tt.model_config import DEFAULT_HF_META_DIR, PROJECT_ROOT, MotifTTConfig

GOLD_BF16 = PROJECT_ROOT / "goldens" / "c2"
GOLD_FP32 = PROJECT_ROOT / "goldens" / "c2_fp32"
HF_META = str(DEFAULT_HF_META_DIR)
EMPTY, PAD = None, -2  # emulated cache: never written / bucket padding row


def log(msg: str) -> None:
    print(f"[resumed-prefill {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def host_cfg(**kw) -> MotifTTConfig:
    return MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8), **kw)


# ======================================================================================================================
# host emulation: a fake model on an emulated paged cache (KV at p = hash of the token prefix [0, p])
# ======================================================================================================================
def chain(h: int, tok: int) -> int:
    """The prefix hash of ``test_prefill_plan._prefix_hashes`` (KV at p depends on tokens [0, p] only, G1)."""
    return (h * 1099511628211 + int(tok) + 1) & ((1 << 61) - 1)


H0 = 1469598103934665603


def prefix_hashes(tokens: Sequence[int]) -> List[int]:
    h, out = H0, []
    for t in tokens:
        h = chain(h, t)
        out.append(h)
    return out


class FakeT:
    """A fake device tensor (the generator frees through ``is_allocated`` / ``ttnn.deallocate``)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.alive = True

    def is_allocated(self):
        return self.alive


class FakeChunk:
    """What ``model.chunk_inputs`` returns: the host tables, plus a free counter."""

    def __init__(self, host):
        self.host = host
        self.path, self.start, self.bucket, self.end = host.path, int(host.start), int(host.bucket), int(host.end)
        self.freed = 0

    @property
    def is_sp1(self):
        return self.path == PP.SP1

    @property
    def head_row(self):
        return self.end - 1 - self.start

    def free(self):
        self.freed += 1


class EmuPool:
    """The emulated KV pool: one main-layer cache and the MTP cache, ``{(block, row): value}``."""

    def __init__(self, num_blocks: int, block_size: int, mtp: bool):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.kv: Dict[Tuple[int, int], Any] = {}
        self.mtp_kv: Optional[Dict[Tuple[int, int], Any]] = {} if mtp else None
        self.mtp = self.mtp_kv  # the generator passes pool.mtp as the MTP layer's kv_cache
        self.complete: set = set()  # blocks whose 64 rows hold real KV: never written again (they may be shared)
        self.layers = [self.kv]

    def __len__(self):
        return 1

    @property
    def mtp_layers(self):
        return 0 if self.mtp_kv is None else 1


class FakeEmbed:
    def __init__(self, model):
        self.m = model

    def prefill_tokens_device(self, tokens, bucket):
        t = [int(x) for x in torch.as_tensor(tokens).reshape(-1).tolist()]
        assert 1 <= len(t) <= bucket, (len(t), bucket)
        self.m.ops.append(("embed", bucket))
        return FakeT(tokens=t + [PAD] * (bucket - len(t)), bucket=int(bucket), real=len(t))

    def rows_tokens_device(self, ids, rows):
        t = [int(x) for x in torch.as_tensor(ids).reshape(-1).tolist()]
        assert len(t) <= rows
        return FakeT(ids=t + [PAD] * (rows - len(t)), rows=int(rows))


class FakeHead:
    def __init__(self, model, vocab: int):
        self.m, self.V = model, int(vocab)

    def forward_prefill(self, X, row):
        self.m.ops.append(("head", X.bucket, row))
        return FakeT(h=X.h[row], row=int(row))

    def prefill_logits_to_host(self, tile, row):
        assert row == tile.row
        out = torch.zeros(self.V, dtype=torch.bfloat16)
        out[tile.h % self.V] = 1.0
        return out

    def stream_mean_norm(self, X):
        return FakeT(h=X.h, start=X.start, end=X.end, tokens=X.tokens, bucket=X.bucket)


class FakeMTP:
    """KV-only MTP fill: writes ``(p, t_{p+1})`` through the chunk's fill table and checks the next tokens: known
    tokens, except the row's last known position, which must take the argmax of the returned logits (``h % V``)."""

    def __init__(self, model):
        self.m = model

    def fill_kv_prefill(self, hn, nxt, *, kv_cache, chunk):
        m, host = self.m, chunk.host
        a, C, end, bs = int(host.start), int(host.bucket), int(host.end), int(host.block_size)
        n_known = m.known_tokens[id(hn.tokens)]  # the request's token count (set by prefill_chunk)
        toks = hn.tokens
        for r in range(end - a):
            p = a + r
            want = toks[r + 1] if r + 1 < end - a else (m.next_known[id(hn.tokens)] if end < n_known else None)
            if want is None:  # the last known position: the stand-in = the row's own argmax
                want = hn.h[r] % m.V
            assert nxt.ids[r] == want, f"MTP next token of position {p}: {nxt.ids[r]}, expected {want}"
        m.ops.append(("mtp", C))
        for j, blk in enumerate(host.fill[0].tolist()):
            if blk < 0:
                continue
            for q in range(bs):
                p = a + j * bs + q
                kv_cache[(blk, q)] = (p, nxt.ids[j * bs + q]) if p < end else (p, PAD)


class FakeModel:
    """The part of ``MotifModel`` the generator's prefill uses, on an :class:`EmuPool`. ``prefill_chunk`` emulates the
    attention's reads: an sp1 chunk reads positions ``[0, a)`` through its SDPA table (global layers; every entry must
    hold its position's prefix hash and chain correctly) and ``[a - 128, a)`` through its tail blocks (SWA layers; must
    equal the SDPA reads); the fill writes the chunk's rows through the fill table, never into a complete block."""

    def __init__(self, cfg: MotifTTConfig, *, mtp: bool = True, true_tokens=None):
        self.cfg = cfg
        self.layer_ids = (0,)
        self.num_layers = 1
        self.V = int(cfg.vocab_size)
        self.embed = FakeEmbed(self)
        self.head = FakeHead(self, cfg.vocab_size)
        self.mtp = FakeMTP(self) if mtp else None
        self.ccl = None
        self.ops: List[Tuple] = []
        self.known_tokens: Dict[int, int] = {}
        self.next_known: Dict[int, int] = {}
        self.chunks: List[FakeChunk] = []
        self.true_tokens = true_tokens  # {request id: tokens} for the next-token lookup (set by the harness)
        self.current = None  # the PrefillRequest being run (set by the harness's chunk_inputs hook)

    def allocate_kv_caches(self, num_blocks, block_size, dtype=None, *, mtp=None):
        return EmuPool(num_blocks, block_size, self.mtp is not None if mtp is None else mtp)

    def chunk_inputs(self, host):
        ch = FakeChunk(host)
        self.chunks.append(ch)
        self.ops.append(("inputs", host.path, int(host.bucket)))
        return ch

    def prefill_chunk(self, tok, *, chunk, kv_caches):
        host, pool = chunk.host, kv_caches
        a, C, end, bs = int(host.start), int(host.bucket), int(host.end), int(host.block_size)
        assert tok.bucket == C and tok.real == end - a, (tok.bucket, tok.real, C, end, a)
        self.ops.append(("layers", host.path, C, a))
        kv = pool.kv
        if host.path == PP.SP1:
            sd = host.sdpa[0].tolist()
            prev = None
            for q in range(a):
                got = kv.get((sd[q // bs], q % bs), EMPTY)
                assert got is not EMPTY and got[0] == q and got[1] != PAD, f"sp1 read of position {q}: {got}"
                if prev is not None:
                    assert got[2] == chain(prev[2], got[1]), f"position {q}: KV of another prefix (stale block)"
                prev = got
            tail = host.tail.tolist()
            T = len(tail) * bs
            for q in range(a - T, a):
                t = kv.get((tail[(q - a + T) // bs], q % bs), EMPTY)
                assert t == kv.get((sd[q // bs], q % bs)), f"SWA tail row {q} differs from the SDPA table's"
            h = prev[2]
        else:
            assert a == 0
            h = H0
        hs = []
        for r in range(C):
            if r < end - a:
                h = chain(h, tok.tokens[r])
                hs.append(h)
            else:
                hs.append(PAD)
        for j, blk in enumerate(host.fill[0].tolist()):  # sp0 / SWA fill after, global sp1 before its SDPA: same here
            if blk < 0:
                continue
            assert blk not in pool.complete, f"block {blk} is complete (maybe shared) and is written again"
            for q in range(bs):
                p = a + j * bs + q
                kv[(blk, q)] = (p, tok.tokens[j * bs + q], hs[j * bs + q]) if p < end else (p, PAD, PAD)
        for j, blk in enumerate(host.fill[0].tolist()):
            if blk >= 0 and all(kv.get((blk, q), (0, PAD))[1] != PAD for q in range(bs)):
                pool.complete.add(blk)
        req = self.current
        X = FakeT(h=hs, start=a, end=end, tokens=tok.tokens[: end - a], bucket=C)
        if req is not None:  # None: a warm-up chunk
            self.known_tokens[id(X.tokens)] = req.end
            if end < req.end:
                self.next_known[id(X.tokens)] = int(req.tokens[end])
        return X

    def deallocate(self):
        pass


def fake_generator(cfg: MotifTTConfig, monkeypatch, *, mtp: bool = True, num_blocks: int = 20000):
    """A ``MotifGenerator`` on a :class:`FakeModel` (no device): ``ttnn.deallocate`` recorded, the pool emulated."""
    from models.demos.motif3.tt import generator as G

    freed = []
    monkeypatch.setattr(G.ttnn, "deallocate", lambda t, *a, **k: freed.append(t), raising=False)
    model = FakeModel(cfg, mtp=mtp)
    gen = G.MotifGenerator(None, cfg, model, log=None)
    # the fake model needs the request of the chunk being run (for the next-token checks): wrap _run_row
    orig = gen._run_row

    def run_row(job, pool):
        model.current = job.request
        return orig(job, pool)

    gen._run_row = run_row
    pool = model.allocate_kv_caches(num_blocks, cfg.kv_block_size)
    gen._pool = pool
    return gen, model, pool, freed


def _rows_of(tokens, start, end, blocks, *, width, lane):
    pt = torch.zeros(width, dtype=torch.int32)
    pt[: len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return api.PrefillRequest(
        lane=lane, tokens=torch.tensor(tokens[:end], dtype=torch.int32), page_table=pt, start=start
    )


# ======================================================================================================================
# host tests
# ======================================================================================================================
def test_cpu_capabilities_and_prefill_shapes(monkeypatch):
    """Capabilities (the bridge's checks), the warmed shapes per span cap, and the sp1 bucket cap."""
    cfg = host_cfg()
    gen, *_ = fake_generator(cfg, monkeypatch)
    assert gen.supports_resumed_prefill and gen.supports_spec_decode  # the fake model has its MTP layer
    assert not gen.spec_launch and gen.serving_path == ("plain", "row")  # host_cfg: spec_tokens 0, no KV-R
    assert gen.prefill_alignment == cfg.prefill_resume_alignment == api.DEFAULT_PREFILL_ALIGNMENT == 128  # G9 q/k
    assert gen.max_prefill_span == 8192 and gen.max_prefill_len == 32768 and gen.max_sp1_bucket == 8192
    buckets = (128, 256, 512, 1024, 2048, 4096, 8192)
    assert gen.prefill_shapes() == [(p, b) for b in buckets for p in (PP.SP0, PP.SP1)]
    settings = api.GeneratorSettings(
        block_size=64, chunked_prefill=True, prefix_caching=True, max_num_batched_tokens=8128
    )
    api.check_generator_features(gen, settings)
    assert gen.decode_kv_mode == "row"  # host_cfg: kv_replicated_decode False
    assert MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8), kv_replicated_decode=True).kv_write_mode == "all"
    # max_model_len 8192: span cap 8192 = max_model_len, the SWA square [tail | chunk] caps sp1 at 4096
    c8 = host_cfg(max_model_len=8192)
    g8, *_ = fake_generator(c8, monkeypatch)
    assert g8.max_prefill_span == 8192 and g8.max_sp1_bucket == 4096
    shapes = g8.prefill_shapes()
    assert (PP.SP0, 8192) in shapes and (PP.SP1, 8192) not in shapes and (PP.SP1, 4096) in shapes
    # MOTIF3_PREFILL_MAX_BUCKET=32768: draft-1 single-shot buckets; sp1 up to 16384
    c32 = host_cfg(prefill_span_cap=32768)
    g32, *_ = fake_generator(c32, monkeypatch)
    assert g32.max_prefill_span == 32768 and g32.max_sp1_bucket == 16384
    assert [b for p, b in g32.prefill_shapes() if p == PP.SP0][-1] == 32768
    assert [b for p, b in g32.prefill_shapes() if p == PP.SP1][-1] == 16384


@pytest.mark.parametrize("max_model_len, cap", [(8192, None), (32768, 32768), (4096, None)])
def test_cpu_plan_row_caps_sp1_buckets(monkeypatch, max_model_len, cap):
    """When the span cap reaches max_model_len, an sp1 chunk must not use a bucket whose SWA square ``[tail ‖ chunk]``
    exceeds max_model_len: the generator re-plans such rows with ``max_sp1_bucket`` (prefill_plan invariants kept)."""
    kw = {"max_model_len": max_model_len} | ({"prefill_span_cap": cap} if cap else {})
    cfg = host_cfg(**kw)
    gen, *_ = fake_generator(cfg, monkeypatch)
    top, bs, A = gen.max_sp1_bucket, cfg.kv_block_size, gen.prefill_alignment
    replanned = 0
    rng = random.Random(max_model_len)
    cases = [(s, e) for s in (0, 1, 127, 128, 640, 1348) for e in (s + 1, max_model_len) if s < e <= max_model_len]
    cases += [(rng.randrange(0, max_model_len - 1), 0) for _ in range(300)]
    for s, e in cases:
        e = e or rng.randrange(s + 1, max_model_len + 1)
        plan = gen.plan_row(s, e)
        base = cfg.plan_prefill_row(s, e)
        replanned += plan != base
        if plan != base:
            assert any(c.is_sp1 and c.bucket > top for c in base.chunks)
        assert plan.w0 == s // bs * bs and plan.chunks[0].start == plan.c0
        pos = plan.c0
        for c in plan.chunks:
            assert c.start == pos and c.start % A == 0 and c.bucket <= gen.max_prefill_span
            assert not c.is_sp1 or c.bucket <= top, (s, e, plan.chunks)
            assert c.start + c.bucket <= max_model_len + gen.max_prefill_span
            pos = c.end
        assert pos == e and plan.chunks[-1].last
    assert replanned > 0 or top >= gen.max_prefill_span


@pytest.mark.parametrize(
    "bs, cap, budget, threshold, seed",
    [
        (64, 8192, 8064, 8064, 0),  # production (A = 128; lead decision 4: threshold = budget)
        (64, 512, 1000, 0, 1),  # small span cap: internal sp0 + sp1 chunks inside one call; unaligned vLLM chunk ends
        (64, 1024, 777, 300, 2),
        (32, 512, 2048, 1024, 3),  # block 32 (A = 128 > bs)
    ],
)
def test_cpu_prefill_batch_emulated_schedule(monkeypatch, bs, cap, budget, threshold, seed):
    """The generator's prefill_forward_batch under a vLLM-like schedule (shared system prompts, duplicate prompts,
    multi-turn extensions, same-step hits, unaligned chunk ends, shuffled rows, a new lane per row per call) on the
    emulated cache: every row's logits are those of its full token prefix, every sp1 read sees the reader's own prefix,
    no complete block is rewritten, the MTP cache gets ``t_{p+1}`` (or the argmax stand-in) for every row the call
    writes, every chunk's inputs are freed once, the LM head runs once per row (last chunk)."""
    from models.demos.motif3.tests.unit.test_prefill_plan import FakeVllmScheduler

    cfg = host_cfg(prefill_span_cap=cap)
    cfg.set_kv_geometry(24000 if bs == 32 else 12000, bs)
    gen, model, pool, freed = fake_generator(cfg, monkeypatch, num_blocks=cfg.kv_num_blocks)
    rng = random.Random(seed)
    width = 32768 // bs
    sched = FakeVllmScheduler(bs=bs, num_blocks=cfg.kv_num_blocks, budget=budget, threshold=threshold, width=width)
    systems = [[rng.randrange(5000) for _ in range(n)] for n in (64, 200, 1300, 2600)]
    history = []
    st = dict(calls=0, rows=0, resumed=0, internal_split=0, reordered=0, sp1=0, mtp_checked=0)
    for wave in range(10):
        fresh = [rng.randrange(5000) for _ in range(rng.randint(130, 2600))]
        for _ in range(rng.randint(2, 3)):
            toks = fresh + [rng.randrange(5000) for _ in range(rng.randint(1, 700))]
            history.append(toks)
            sched.add(toks)
        for _ in range(rng.randint(0, 3)):
            k = rng.random()
            if k < 0.3 and history:
                toks = list(rng.choice(history))
            elif k < 0.6 and history:
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
            reqs = []
            for k, lane in zip(perm, lanes):
                r, end = rows[k]
                reqs.append((_rows_of(r.tokens, r.computed, end, r.blocks, width=width, lane=lane), r))
            n_chunks = len(model.chunks)
            heads = sum(1 for o in model.ops if o[0] == "head")
            logits = gen.prefill_forward_batch([q for q, _ in reqs], kv_cache=pool)
            assert tuple(logits.shape) == (len(reqs), cfg.vocab_size)
            for i, (q, r) in enumerate(reqs):  # input order; the row's own full-prefix KV
                want = r.h[q.end - 1] % cfg.vocab_size
                assert int(logits[i].float().argmax()) == want, f"row {i}: logits of another prefix"
            new = model.chunks[n_chunks:]
            assert all(c.freed == 1 for c in new), "every chunk's inputs are freed exactly once"
            assert sum(1 for o in model.ops if o[0] == "head") - heads == len(reqs)
            batch = gen.last_prefill
            st["reordered"] += batch.order != sorted(batch.order)
            st["internal_split"] += sum(len(j.plan.chunks) > 1 for j in batch.jobs)
            st["sp1"] += sum(c.is_sp1 for j in batch.jobs for c in j.plan.chunks)
            st["resumed"] += sum(q.start > 0 for q, _ in reqs)
            st["calls"] += 1
            st["rows"] += len(reqs)
            # MTP: every position the call wrote holds (p, t_{p+1}) (the stand-in for each row's last position)
            for q, r in reqs:  # rows [w0, end) are the row's own blocks: written by this call
                plan = next(j.plan for j in batch.jobs if j.request is q)
                for p in range(plan.w0, q.end):
                    got = pool.mtp_kv.get((int(q.page_table[p // bs]), p % bs))
                    nxt = r.tokens[p + 1] if p + 1 < q.end else r.h[q.end - 1] % cfg.vocab_size
                    assert got == (p, nxt), f"MTP entry of position {p}: {got}, expected {(p, nxt)}"
                    st["mtp_checked"] += 1
            sched.commit(rows)
            assert st["calls"] < 3000
    assert st["resumed"] > 0 and st["reordered"] > 0 and st["sp1"] > 0, st
    if min(budget, threshold or budget) + gen.prefill_alignment - 1 > cap:  # some span exceeds the cap
        assert st["internal_split"] > 0, st
    assert not freed or all(isinstance(t, FakeT) for t in freed)
    log(f"emulated schedule bs={bs} cap={cap} budget={budget} threshold={threshold} seed={seed}: {st}")


def test_cpu_same_step_hit_and_refusals(monkeypatch):
    """Writer-first order for a same-step hit given in reader-first order; a refused call (unwarmed shape after the
    capture, bad token id, page table too short) runs no device op and leaves the pool unchanged."""
    cfg = host_cfg()
    gen, model, pool, _ = fake_generator(cfg, monkeypatch)
    W = 512
    rng = random.Random(7)
    toks = [rng.randrange(5000) for _ in range(1500)]
    h = prefix_hashes(toks)
    blk_w = list(range(1, 25))
    blk_r = blk_w[:20] + list(range(100, 104))  # hits the writer's first 20 blocks (1280 tokens) in the same call
    reader = _rows_of(toks, 1280, 1500, blk_r, width=W, lane=3)
    writer = _rows_of(toks, 0, 1500, blk_w, width=W, lane=11)
    out = gen.prefill_forward_batch([reader, writer], kv_cache=pool)
    assert gen.last_prefill.order == [1, 0]
    assert [int(x.float().argmax()) for x in out] == [h[1499] % cfg.vocab_size] * 2
    # refusals: nothing runs
    from models.demos.motif3.tt.generator import DecodePath

    gen._paths[gen.serving_path] = DecodePath(*gen.serving_path, width=W, trace_id=object())  # "captured"
    n_ops, snap = len(model.ops), dict(pool.kv)
    with pytest.raises(RuntimeError, match="not compiled before the decode trace capture"):
        gen.prefill_forward_batch([_rows_of(toks, 0, 100, [200, 201], width=W, lane=0)], kv_cache=pool)
    gen._warmed = {(PP.SP0, 128)}
    gen.prefill_forward_batch([_rows_of(toks, 0, 100, [200, 201], width=W, lane=0)], kv_cache=pool)  # warmed: runs
    n_ops, snap = len(model.ops), dict(pool.kv)
    bad = [
        _rows_of(toks, 0, 100, [202, 203], width=W, lane=0),
        _rows_of(toks, 1280, 1500, blk_r, width=W, lane=1),  # (sp1, 256) is not warmed
    ]
    with pytest.raises(RuntimeError, match=r"\('sp1', 256\)"):
        gen.prefill_forward_batch(bad, kv_cache=pool)
    gen._paths.clear()
    big = list(toks[:99]) + [cfg.vocab_size]
    with pytest.raises(ValueError, match="token ids"):
        gen.prefill_forward_batch([_rows_of(big, 0, 100, [202, 203], width=W, lane=0)], kv_cache=pool)
    short = api.PrefillRequest(
        lane=0,
        tokens=torch.tensor(toks[:300], dtype=torch.int32),
        page_table=torch.tensor([5, 6, 7, 8], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="page_table has 4 entries"):
        gen.prefill_forward_batch([short], kv_cache=pool)
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        long = api.PrefillRequest(
            lane=0, tokens=torch.zeros(32769, dtype=torch.int32), page_table=torch.arange(1, 514, dtype=torch.int32)
        )
        gen.prefill_forward_batch([long], kv_cache=pool)
    # page-table ids (review 2026-10-03): the fill kernel has no bounds check, so an id past the pool would write
    # outside the cache buffer; the sp1 SDPA / tail reads would read outside it
    nb = pool.num_blocks
    bad_tables = [
        (_rows_of(toks, 0, 100, [202, nb], width=W, lane=0), rf"entry 1 = {nb} .*not a block id in \[1, {nb}\)"),
        (_rows_of(toks, 0, 100, [202, 0], width=W, lane=0), r"entry 1 = 0 .*null block"),
        (_rows_of(toks, 0, 100, [202, 202], width=W, lane=0), r"block id 202 appears at page-table entries \[0, 1\]"),
        # an sp1 row whose READ-ONLY prefix (shared blocks, read through the SDPA table) holds an id past the pool
        (_rows_of(toks, 1280, 1500, blk_r[:3] + [nb + 7] + blk_r[4:], width=W, lane=0), rf"entry 3 = {nb + 7}"),
    ]
    for row, msg in bad_tables:
        with pytest.raises(ValueError, match=msg):
            gen.prefill_forward_batch([_rows_of(toks, 0, 100, [204, 205], width=W, lane=1), row], kv_cache=pool)
    assert len(model.ops) == n_ops and pool.kv == snap, "a refused call must not touch the device"


def test_cpu_check_prefill_page_table():
    """``generator.check_prefill_page_table``: only the entries a row uses (``cdiv(end, bs)``) are checked; each must be
    a distinct block id in ``[1, num_blocks)``; without a pool only the lower bound applies."""
    from models.demos.motif3.tt.generator import check_prefill_page_table as chk

    pt = torch.tensor([5, 6, 7, 0, 0, 999999], dtype=torch.int32)  # entries past cdiv(end, bs) are not looked at
    for end in (1, 64, 65, 192):
        chk(pt, end, block_size=64, num_blocks=8)
    with pytest.raises(ValueError, match="entry 2 = 7 .*not a block id in \\[1, 7\\)"):
        chk(pt, 192, block_size=64, num_blocks=7)
    with pytest.raises(ValueError, match="entry 3 = 0"):
        chk(pt, 193, block_size=64, num_blocks=8)
    with pytest.raises(ValueError, match="page_table has 6 entries, positions \\[0, 400\\) need 7"):
        chk(pt, 400, block_size=64)
    chk(torch.tensor([5, 10**6], dtype=torch.int32), 100, block_size=64)  # no pool: the upper bound is unknown
    with pytest.raises(ValueError, match="block id 5 appears"):
        chk(torch.tensor([5, 9, 5], dtype=torch.int32), 129, block_size=64, num_blocks=10)
    chk(torch.tensor([5, 9, 5], dtype=torch.int32), 128, block_size=64, num_blocks=10)  # the duplicate is unused
    chk(torch.tensor([3, 4], dtype=torch.int32), 33, block_size=32, num_blocks=5)


def test_cpu_metrics_nan_safety():
    """A NaN / Inf logits row, or one of finite device garbage (|x| ~ 1e18), fails every metric helper instead of
    dropping out of it (review 2026-10-03): ``assert_sane``; ``row_metrics`` raises on such logits; ``pooled``
    averages the NLL over the rows WITH a target and raises on a non-finite one; ``compare_streams`` fails a
    non-finite margin whichever stream carries it (``min(0.1, nan)`` is 0.1: a NaN lane was excused as a near tie)."""
    nan, inf = float("nan"), float("inf")
    assert_sane(torch.zeros(3, 10), "ok")
    lg = torch.zeros(3, 10)
    lg[1, 4] = nan
    with pytest.raises(AssertionError, match=r"positions \[101\]"):
        assert_sane(lg, "x", [100, 101, 102])
    lg[1, 4], lg[2, 0] = 0.0, -inf
    with pytest.raises(AssertionError, match=r"rows \[2\]"):
        assert_sane(lg, "x")
    assert_sane(torch.zeros(10, dtype=torch.bfloat16), "one row")
    lg[2, 0] = 0.0
    assert_sane(lg * 0 + LOGIT_ABS_MAX, "at the bound")
    lg[0, 7] = 7.9e18  # finite garbage (what the 2026-10-03 two-trace prefills returned)
    with pytest.raises(AssertionError, match=r"out-of-range .* rows \[0\] .*max \|logit\| 7.9e\+18"):
        assert_sane(lg, "x")
    # compare_streams: near ties still excused; a non-finite margin never
    assert compare_streams([1, 2, 3], [1.0, 1.0, 0.1], [1, 2, 4], [1.0, 1.0, 2.0])[0]
    assert not compare_streams([1, 2, 3], [1.0, 1.0, 1.0], [1, 2, 4], [1.0, 1.0, 2.0])[0]
    for ma, mb in (
        ([1.0, 0.1], [1.0, nan]),
        ([1.0, nan], [1.0, 0.1]),
        ([1.0, inf], [1.0, 0.1]),
        ([nan, 1.0], [1.0, 1.0]),
    ):
        ok, why = compare_streams([1, 5], ma, [1, 6], mb)
        assert not ok and "non-finite" in why, (ma, mb, why)
    ok, why = compare_streams([1, 5], [1.0, 1.0], [1, 5], [1.0, nan])  # equal tokens, NaN logits: still a failure
    assert not ok and "non-finite" in why
    # pooled: NaN means "no next token" only where has_target is False
    m = {"agree": torch.tensor([True, False, True]), "nll": torch.tensor([1.0, 3.0, nan], dtype=torch.float64),
         "has_target": torch.tensor([True, True, False]), "pcc": torch.tensor([0.99])}  # fmt: skip
    p = pooled([m])
    assert p["rows"] == 3 and abs(p["nll"] - 2.0) < 1e-12 and abs(p["agree"] - 2 / 3) < 1e-6  # agree is float32
    m["has_target"] = torch.tensor([True, True, True])
    with pytest.raises(AssertionError, match="non-finite NLL"):
        pooled([m])
    m["has_target"], m["pcc"] = torch.tensor([True, True, False]), torch.tensor([0.99, nan])
    with pytest.raises(AssertionError, match="non-finite logit PCC"):
        pooled([m])
    # row_metrics: a tiny golden (vocab 8, top-4); NaN logits raise, the last row has no target
    S, V = 3, 8
    g = GoldPrompt("tiny", [1, 2, 3], torch.tensor([[0, 1, 2, 3]] * S), torch.tensor([[4.0, 3.0, 2.0, 1.0]] * S),
                   torch.zeros(S, dtype=torch.int64), torch.tensor([2, 3, -1]), torch.tensor([0, 2]),
                   torch.randn(2, V))  # fmt: skip
    lg = torch.randn(S, V)
    out = row_metrics(g, torch.arange(S), lg)
    assert out["has_target"].tolist() == [True, True, False] and bool(torch.isnan(out["nll"][2]))
    assert abs(pooled([out])["nll"] - float(out["nll"][:2].mean())) < 1e-9
    lg[1, 3] = nan
    with pytest.raises(AssertionError, match=r"tiny logits: non-finite or out-of-range .* positions \[1\]"):
        row_metrics(g, torch.arange(S), lg)
    lg[1, 3] = -3e19
    with pytest.raises(AssertionError, match=r"tiny logits: non-finite or out-of-range .* positions \[1\]"):
        row_metrics(g, torch.arange(S), lg)


def test_cpu_emulation_negative_controls(monkeypatch):
    """The emulation catches what it is meant to catch: rows run in input order (no writer-first reordering) let a
    same-step hit read unwritten KV; an MTP stand-in other than the row's argmax is flagged; a fill table that writes a
    shared block is flagged; an SDPA table pointing one block off reads another prefix."""
    from models.demos.motif3.tt import attention as A
    from models.demos.motif3.tt import generator as G

    cfg = host_cfg()
    W, rng = 512, random.Random(11)
    toks = [rng.randrange(5000) for _ in range(1500)]
    blk_w = list(range(1, 25))
    reader = _rows_of(toks, 1280, 1500, blk_w[:20] + list(range(100, 104)), width=W, lane=3)
    writer = _rows_of(toks, 0, 1500, blk_w, width=W, lane=11)
    # (1) input order
    gen, model, pool, _ = fake_generator(cfg, monkeypatch)
    monkeypatch.setattr(G.PP, "order_prefill_requests", lambda reqs, plans, bs: list(range(len(reqs))))
    with pytest.raises(AssertionError, match="sp1 read of position 0"):
        gen.prefill_forward_batch([reader, writer], kv_cache=pool)
    monkeypatch.undo()
    # (2) a wrong MTP stand-in (the known-token rule broken for the row's last position)
    gen, model, pool, _ = fake_generator(cfg, monkeypatch)
    real = G.mtp_next_tokens
    monkeypatch.setattr(G, "mtp_next_tokens", lambda t, s, e, nxt: real(t, s, e, None if nxt is None else nxt + 1))
    with pytest.raises(AssertionError, match="MTP next token of position 1499"):
        gen.prefill_forward_batch([writer], kv_cache=pool)
    monkeypatch.undo()
    # (3) a fill table that also writes the shared blocks below w0
    gen, model, pool, _ = fake_generator(cfg, monkeypatch)
    gen.prefill_forward_batch([writer], kv_cache=pool)
    real_tables = A.chunk_host_tables

    def leaky(cfg_, plan, ch, pt):
        h = real_tables(cfg_, plan, ch, pt)
        if not h.is_sp1:
            return h
        fill = h.fill.clone()
        fill[0, 0] = int(h.sdpa[0, ch.start // cfg_.kv_block_size - 1])  # the block before the chunk: shared
        return _unchecked(h, fill=fill)

    monkeypatch.setattr(G, "chunk_host_tables", leaky)
    with pytest.raises(AssertionError, match="is complete .maybe shared. and is written again"):
        gen.prefill_forward_batch([reader], kv_cache=pool)
    monkeypatch.undo()
    # (4) an SDPA table shifted by one block: another prefix's KV
    gen, model, pool, _ = fake_generator(cfg, monkeypatch)
    gen.prefill_forward_batch([writer], kv_cache=pool)

    def shifted(cfg_, plan, ch, pt):
        h = real_tables(cfg_, plan, ch, pt)
        if not h.is_sp1:
            return h
        sd = h.sdpa.clone()
        sd[0, 1:] = h.sdpa[0, :-1]
        return _unchecked(h, sdpa=sd, tail=sd[0, ch.start // cfg_.kv_block_size - 2 : ch.start // cfg_.kv_block_size])

    monkeypatch.setattr(G, "chunk_host_tables", shifted)
    with pytest.raises(AssertionError, match="sp1 read of position"):
        gen.prefill_forward_batch([reader], kv_cache=pool)


def _unchecked(host, **changes):
    """``host`` with fields replaced, bypassing ``ChunkHostTables.__post_init__`` (negative controls only)."""
    obj = object.__new__(type(host))
    for f in dataclasses.fields(host):
        object.__setattr__(obj, f.name, changes.get(f.name, getattr(host, f.name)))
    return obj


def test_cpu_warmup_compiles_every_shape(monkeypatch):
    """warmup_prefill runs one warm-up chunk per (path, bucket) (writes nothing: all -1 fill tables), with the head
    and the MTP fill; the capture is refused until then; a second warmup is a no-op."""
    cfg = host_cfg(prefill_span_cap=1024)
    gen, model, pool, _ = fake_generator(cfg, monkeypatch)
    orig = model.prefill_chunk

    def prefill_chunk(tok, *, chunk, kv_caches):
        if chunk.host.path == PP.SP1 and int(chunk.host.fill.max()) < 0:  # warm-up: shapes only
            model.ops.append(("layers", chunk.host.path, chunk.host.bucket, chunk.host.start))
            C = chunk.host.bucket
            return FakeT(h=[0] * C, start=chunk.host.start, end=chunk.host.end, tokens=tok.tokens[:C], bucket=C)
        return orig(tok, chunk=chunk, kv_caches=kv_caches)

    model.prefill_chunk = prefill_chunk
    model.mtp.fill_kv_prefill = lambda hn, nxt, *, kv_cache, chunk: model.ops.append(("mtp", chunk.bucket))
    with pytest.raises(RuntimeError, match="before the prefill warmup"):
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=512)
    gen.warmup_prefill(kv_cache=pool, enable_trace=False)
    shapes = gen.prefill_shapes()
    assert gen.warmed_shapes == set(shapes) and len(shapes) == 8
    assert [o[1:3] for o in model.ops if o[0] == "layers"] == shapes
    assert sum(o[0] == "head" for o in model.ops) == len(shapes) and sum(o[0] == "mtp" for o in model.ops) == len(
        shapes
    )
    assert all(int(c.host.fill.max()) < 0 for c in model.chunks), "warm-up chunks write nothing"
    assert pool.kv == {} and not pool.mtp_kv
    n = len(model.ops)
    gen.warmup_prefill(kv_cache=pool, enable_trace=False)
    assert len(model.ops) == n
    gen.warmup_prefill(kv_cache=pool, enable_trace=True)  # trace_mode="all": prefill stays eager
    assert len(model.ops) == n


def test_cpu_mtp_part_estimate_and_pool():
    """Part bookkeeping of the MTP layer (review R8: cfg.layer(53) does not exist)."""
    from models.demos.motif3.tt import model as M

    cfg = host_cfg()
    assert M.is_mtp_part(cfg, 53) and not M.is_mtp_part(cfg, 52) and not M.is_mtp_part(cfg, None)
    assert M.estimate_part_bytes(cfg, 53) == M.EST_CORE_BYTES["mtp"] == 420_183_552
    assert M.estimate_part_bytes(cfg, 2) > M.estimate_part_bytes(cfg, 0) > 0
    pool = M.MotifKVPool([1, 2], (0, 1), 10, 64, "bfp8", mtp=3)
    assert len(pool) == 2 and pool.mtp == 3 and pool.mtp_layers == 1
    assert M.MotifKVPool([1], (0,), 10, 64, "bfp8").mtp_layers == 0


class _Rec:
    """Records calls; returns a fresh sentinel per call."""

    def __init__(self, name, log):
        self.name, self.log = name, log

    def __call__(self, *a, **k):
        out = SimpleNamespace(src=self.name, args=a, kw=k, shape=getattr(a[0], "shape", None) if a else None)
        self.log.append((self.name, a, k, out))
        return out


def test_cpu_decoder_and_model_thread_chunk_and_kv_write(monkeypatch):
    """``MotifDecoderLayer.forward_prefill(chunk=)`` passes the chunk (not a page table) to the attention, every other
    sub-module unchanged; ``forward_decode(kv_write=)`` passes the writer; ``MotifModel.prefill_chunk`` gives every
    layer the same chunk and its own cache; ``MotifModel.decode(kv_write=)`` checks the FlashMLA inputs, passes the
    writer to every layer and ends the step once."""
    from models.demos.motif3.tt import decoder as D
    from models.demos.motif3.tt import model as M

    calls = []
    fake = SimpleNamespace(
        rms_norm=_Rec("rms_norm", calls),
        to_memory_config=_Rec("to_mc", calls),
        deallocate=lambda *a, **k: None,
        DRAM_MEMORY_CONFIG="dram",
    )
    monkeypatch.setattr(D, "ttnn", fake)
    layer = object.__new__(D.MotifDecoderLayer)
    layer.mhc_attn = SimpleNamespace(pre=lambda X: ("xr", "c1"), post=lambda X, o, c: "X1")
    layer.mhc_ffn = SimpleNamespace(pre=lambda X: ("yr", "c2"), post=lambda X, u, c: "X2")
    seen = {}

    def attn_prefill(a, **kw):
        seen["prefill"] = kw
        return "o"

    def attn_decode(a, **kw):
        seen["decode"] = kw
        return "o"

    layer.attn = SimpleNamespace(forward_prefill=attn_prefill, forward_decode=attn_decode)
    layer.is_moe, layer.mlp = False, SimpleNamespace(forward_prefill=lambda f: "u", forward_decode=lambda f: "u")
    layer.input_norm = layer.post_attn_norm = "g"
    layer.eps, layer.ckc_norm, layer.norm_mc, layer.norm_pc = 1e-5, None, None, None
    chunk = SimpleNamespace(is_sp1=True, bucket=256)
    assert layer.forward_prefill("X", chunk=chunk, kv_cache="kv") == "X2"
    assert seen["prefill"] == {"chunk": chunk, "kv_cache": "kv"}
    layer.forward_prefill("X", page_table="pt", kv_cache="kv")
    assert seen["prefill"] == {"page_table": "pt", "kv_cache": "kv"}
    with pytest.raises(ValueError, match="not both"):
        layer.forward_prefill("X", chunk=chunk, page_table="pt", kv_cache="kv")
    w = object()
    layer.forward_decode("X", rot="r", cur_pos="c", page_table="p", kv_cache="kv", active="a", kv_write=w)
    assert seen["decode"]["kv_write"] is w
    layer.forward_decode("X", rot="r", cur_pos="c", page_table="p", kv_cache="kv", active="a")
    assert "kv_write" not in seen["decode"]  # draft 1: the attention's own update

    # ---- MotifModel ----
    model = object.__new__(M.MotifModel)
    got = []

    class L:
        def __init__(self, i):
            self.i = i

        def forward_prefill(self, X, **kw):
            got.append(("p", self.i, kw))
            return f"X{self.i}"

        def forward_decode(self, X, **kw):
            got.append(("d", self.i, kw))
            return f"X{self.i}"

    model.layers = [L(0), L(1), L(2)]
    model.embed = SimpleNamespace(forward_prefill=lambda t: "X", forward_decode=lambda t: "X")
    model.head = SimpleNamespace(forward_decode=lambda X, row_major: "logits")
    model.rope, model._rope_kinds = None, ("yarn",)
    model.cfg = SimpleNamespace(lanes_per_row=8)
    monkeypatch.setattr(M, "_free", lambda *a: None)
    monkeypatch.setattr(M.MotifAttention, "decode_rope_tables", staticmethod(lambda rope, idx, kinds: {}))
    monkeypatch.setattr(M.MotifAttention, "active_mask_from_cur_pos", staticmethod(lambda c, lanes: "act"))
    tok = SimpleNamespace(shape=(1, 256))
    out = model.prefill_chunk(tok, chunk=chunk, kv_caches=["k0", "k1", "k2"])
    assert out == "X2" and [g[2] for g in got] == [{"chunk": chunk, "kv_cache": f"k{i}"} for i in range(3)]
    with pytest.raises(ValueError, match="bucket is 256"):
        model.prefill_chunk(SimpleNamespace(shape=(1, 128)), chunk=chunk, kv_caches=["k0", "k1", "k2"])
    with pytest.raises(ValueError, match="needs kv_caches"):
        model.prefill_chunk(tok, chunk=chunk)
    got.clear()
    ends = []

    class KW:
        cur_pos, page_table = "cur", "pt"

        def check_flash_inputs(self, c, p):
            if (c, p) != (self.cur_pos, self.page_table):
                raise ValueError("FlashMLA must read kv_write.cur_pos / kv_write.page_table")

        def end_step(self):
            ends.append(1)

    kw = KW()
    assert (
        model.decode("t", rot_idxs="r", cur_pos="cur", page_table="pt", kv_caches=["k0", "k1", "k2"], kv_write=kw)
        == "logits"
    )
    assert [g[2]["kv_write"] for g in got] == [kw] * 3 and ends == [1]
    with pytest.raises(ValueError, match="kv_write.cur_pos"):
        model.decode("t", rot_idxs="r", cur_pos="other", page_table="pt", kv_caches=["k0", "k1", "k2"], kv_write=kw)
    got.clear()
    model.decode("t", rot_idxs="r", cur_pos="c", page_table="p", kv_caches=["k0", "k1", "k2"])
    assert all("kv_write" not in g[2] for g in got)


def test_cpu_generator_decode_kv_write_wiring(monkeypatch):
    """Plain decode wiring per mode: ``row`` (draft 1) keeps the path's own cur_pos / page_table and passes no writer;
    ``all`` (KV-R) builds one ``DecodeKVWrite`` with the path's inputs, writes every step as an ordinary
    ``KVWriteStep`` and passes its tensors and the writer to the model. (The spec path's wiring:
    ``tests/test_spec_decode_device.py``.)"""
    from models.demos.motif3.tt import generator as G

    class FakeKVW:
        built = []

        def __init__(self, mesh, cfg, *, ccl, page_table_width, mode):
            self.mode, self.width, self.steps = mode, page_table_width, []
            self.cur_pos, self.page_table = "kvw.cur", "kvw.pt"
            FakeKVW.built.append(self)

        def write_step(self, step, validate=True):
            self.steps.append(step)

        def deallocate(self):
            pass

    host_calls = []
    monkeypatch.setattr(G, "DecodeKVWrite", FakeKVW)
    monkeypatch.setattr(G, "shard_lanes", lambda rows, cfg, mesh, **kw: ("host", tuple(rows.shape)))
    monkeypatch.setattr(G.ttnn, "to_device", lambda v, mesh, memory_config=None: ("dev", v), raising=False)
    monkeypatch.setattr(G.ttnn, "copy_host_to_device_tensor", lambda h, d: host_calls.append((h, d)), raising=False)
    for kvr, mode in ((False, "row"), (True, "all")):
        cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8), kv_replicated_decode=kvr)
        model = FakeModel(cfg)
        model.embed.decode_tokens_host = lambda t: ("tok", tuple(t.shape))
        dec = {}
        model.decode = lambda tokens, **kw: dec.update(kw) or "out"
        gen = G.MotifGenerator(None, cfg, model, log=None)
        assert gen.decode_kv_mode == mode and gen.serving_path == ("plain", mode)
        p = gen._stage_path(gen.serving_path, 64)
        pos = torch.full((32,), -1, dtype=torch.int32)
        pos[3], pos[12] = 100, 7
        pt = torch.zeros(32, 64, dtype=torch.int32)
        pt[3, :2], pt[12, 0] = torch.tensor([5, 6]), 9
        gen._write_plain(p, api.DecodeBatch(tokens=torch.arange(32, dtype=torch.int32), positions=pos, page_table=pt))
        assert gen._plain_step(p, None) == "out"
        if mode == "row":
            assert set(p.inputs) == {"tokens", "rot", "cur", "pt"} and p.kv_write is None
            assert "kv_write" not in dec and dec["cur_pos"] == p.inputs["cur"]
        else:
            w = FakeKVW.built[-1]
            assert set(p.inputs) == {"tokens", "rot"} and p.kv_write is w and w.mode == "all" and w.width == 64
            assert dec["kv_write"] is w and dec["cur_pos"] == "kvw.cur" and dec["page_table"] == "kvw.pt"
            (step,) = w.steps
            assert torch.equal(step.positions, pos) and torch.equal(step.page_table, pt) and not bool(step.call_b.any())


# ======================================================================================================================
# device tests: the real model from the TT cache (design §5.3)
# ======================================================================================================================
CP_LAYERS = int(os.environ.get("MOTIF3_CP_LAYERS", "53"))  # < 53: a quick plumbing run (the bars assume 53 layers)
MAX_LEN = 32768
BS = 64
NUM_BLOCKS = api.expected_num_blocks()  # 4129: the serving pool (262,144 tokens, block 64, 32 seqs)
WIDTH = min(api.cdiv(MAX_LEN, BS), NUM_BLOCKS)  # 512: the plugin's page-table width
DECODE_STEPS = 32
NEAR_TIE = 0.5  # design §5.3: argmax may differ only where the reference margin (top-1 - top-2) is below this
PLAIN_KVR = ("plain", "all")  # the non-speculating production decode path (KV-R); CP-X runs it eagerly
MULTI_TRACE_NOTE = (
    "2026-10-03 (logs/dev/20261003_0112..0141_wp45_diag_*.log): a session that captured TWO decode traces (spec "
    "all_split + plain all) and then, with the traces released, compiled a new prefill bucket (a 12K or 32K single "
    "shot: 66-72 new programs) and re-captured, returned garbage 32-row tiles (finite, |logit| 1e18-1e20) in 1-4 of "
    "every 6 later bucket-1024 prefills while a trace was alive; clean with no trace alive, with one captured trace "
    "(0 of 30), without the compile (0 of 30) or with one small new program (0 of 30). The tracker "
    "(TT_METAL_TRACE_ALLOC_TRACKING=1) found no live unsafe buffer. Serving captures one trace and never re-captures."
)
SAMPLE_EVERY = 4  # full-vocab PCC rows (the fp32 reference logits are recomputed on the host for these)
CP_TIMEOUT = 3600
DEVICE_MARK = [pytest.mark.timeout(CP_TIMEOUT)]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-300))


def _row_pccs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.double(), b.double()
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-300)


def _margin(logits: torch.Tensor) -> float:
    v = torch.topk(logits.float(), 2).values
    return float(v[0] - v[1])


LOGIT_ABS_MAX = 1e4  # Motif-3 logits stay below ~60 in magnitude; device garbage reads as finite |x| ~ 1e18 .. 1e20


def assert_sane(logits: torch.Tensor, what: str, positions: Optional[Sequence[int]] = None) -> None:
    """Every row of ``logits [..., V]`` is finite and at most ``LOGIT_ABS_MAX`` in magnitude. A NaN / Inf row, or a
    row of device garbage (finite, |x| ~ 1e18: 2026-10-03, prefills with two captured decode traces after a program
    compile between captures), must fail loudly: it must never silently drop out of a pooled metric, pass an argmax
    comparison (``argmax`` of a NaN row is the NaN's index) or a margin rule."""
    lg = logits.float().reshape(-1, logits.shape[-1])
    bad = ~(torch.isfinite(lg).all(-1) & (lg.abs().amax(-1) <= LOGIT_ABS_MAX))
    if bool(bad.any()):
        idx = torch.nonzero(bad).flatten().tolist()
        pos = list(positions) if positions is not None else None
        where = [pos[i] for i in idx] if pos is not None else idx
        raise AssertionError(
            f"{what}: non-finite or out-of-range (|logit| > {LOGIT_ABS_MAX:g}) logits at "
            f"{'positions' if pos is not None else 'rows'} {where[:8]}{' ...' if len(where) > 8 else ''} "
            f"({len(where)} of {lg.shape[0]}; max |logit| {float(lg[bad].abs().nan_to_num(float('inf')).max()):.3g})"
        )


class RefuseReads:
    """Weight source that refuses every checkpoint tensor read: the model must build from the TT cache alone (the
    FULL_MODEL_VALIDATION method). Metadata calls go to the local checkpoint index."""

    def __init__(self, weights_dir):
        from models.demos.motif3.tt.weights import HFWeightLoader

        self._loader = HFWeightLoader(weights_dir)
        self.refused: List[str] = []

    def _refuse(self, name):
        self.refused.append(str(name))
        raise RuntimeError(f"checkpoint read of {name!r}: every weight must come from the TT cache")

    def get(self, name, dtype=None):
        self._refuse(name)

    def get_rows(self, name, start, stop, dtype=None):
        self._refuse(name)

    def shape(self, name):
        self._refuse(name)

    def __contains__(self, name):
        return name in self._loader

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        return getattr(self._loader, attr)


class Blocks:
    """KV block ids 1 .. NUM_BLOCKS - 1 (0 = vLLM's null block), never reused within the session."""

    def __init__(self, n: int):
        self.n, self.next = int(n), 1

    def take(self, k: int) -> List[int]:
        if self.next + k > self.n:
            raise RuntimeError(f"KV pool exhausted: {k} blocks at {self.next} of {self.n}")
        out = list(range(self.next, self.next + k))
        self.next += k
        return out


def page_row(blocks: Sequence[int], upto: int) -> torch.Tensor:
    """A request's page-table row for positions ``[0, upto)`` (the bridge's ``_fit_page_table``: zeros after)."""
    pt = torch.zeros(WIDTH, dtype=torch.int32)
    n = api.cdiv(upto, BS)
    assert n <= len(blocks), (n, len(blocks))
    pt[:n] = torch.tensor(list(blocks[:n]), dtype=torch.int32)
    return pt


def prefill_request(lane: int, ids: Sequence[int], end: int, blocks: Sequence[int], start: int = 0):
    return api.PrefillRequest(
        lane=lane,
        tokens=torch.tensor(list(ids[:end]), dtype=torch.int32),
        page_table=page_row(blocks, end),
        start=start,
    )


class RowCollector:
    """``gen.chunk_observer`` that collects teacher-forced logits: ``want = {row index in the call: (lo, hi)}`` ->
    ``rows[(row index, position)] = logits [V]`` for positions ``[lo, hi)`` computed by the call's chunks (the LM head
    on every 32-row tile through the tensor-args slice: no new program; FULL_MODEL_VALIDATION §2.1)."""

    def __init__(self, gen, want: Dict[int, Tuple[int, int]], streams_at: Optional[Dict[int, Sequence[int]]] = None):
        self.gen, self.want = gen, dict(want)
        self.rows: Dict[Tuple[int, int], torch.Tensor] = {}
        self.streams_at = {k: set(int(p) for p in v) for k, v in (streams_at or {}).items()}
        self.streams: Dict[Tuple[int, int], torch.Tensor] = {}  # (row index, position) -> X row [4, 4096] fp32

    def __call__(self, job, ch, X):
        import ttnn

        lo, hi = self.want.get(job.index, (0, 0))
        a, e = int(ch.start), int(ch.end)
        sp = sorted(p for p in self.streams_at.get(job.index, ()) if a <= p < e)
        if sp:  # residual streams of those positions (chip 0; prefill outputs are replicated)
            xt = ttnn.to_torch(ttnn.get_device_tensors(X)[0])  # [1, 4, C, 4096]
            for p in sp:
                self.streams[(job.index, p)] = xt[0, :, p - a].float().clone()
            del xt
        s0, s1 = max(lo, a), min(hi, e)
        if s0 >= s1:
            return
        head = self.gen.model.head
        for r0 in range((s0 - a) // 32 * 32, s1 - a, 32):
            li = min(r0 + 31, e - a - 1)
            tile = head.forward_prefill(X, li)
            try:
                reader = head._reader(tile)
                views = reader.read(tile)
                rows = head._assemble_decode(views, None, np_views=reader.np_views)  # [32, V]: the tile's rows
            finally:
                ttnn.deallocate(tile)
            for r in range(r0, min(r0 + 32, e - a)):
                if s0 <= a + r < s1:
                    self.rows[(job.index, a + r)] = rows[r - r0].clone()

    def take(self, index: int, lo: int, hi: int) -> torch.Tensor:
        missing = [p for p in range(lo, hi) if (index, p) not in self.rows]
        assert not missing, f"row {index}: no logits for positions {missing[:5]}..."
        out = torch.stack([self.rows[(index, p)] for p in range(lo, hi)])
        assert_sane(out, f"row {index} teacher-forced logits", range(lo, hi))
        return out


@dataclasses.dataclass
class GoldPrompt:
    name: str
    ids: List[int]
    topk_ids: torch.Tensor  # [S, 32]
    topk_vals: torch.Tensor  # [S, 32] fp32
    argmax: torch.Tensor  # [S]
    target: torch.Tensor  # [S] next token (-1 past the end)
    sampled: torch.Tensor  # positions with full-vocab reference rows
    ref_rows: torch.Tensor  # [n_sampled, V] fp32 reference logits (fp32 golden hidden @ lm_head fp32)

    @property
    def S(self) -> int:
        return len(self.ids)


def load_goldens() -> Dict[str, GoldPrompt]:
    """The C2 prompts and the fp32 golden (all 53 layers): top-32, argmax, targets, and the full-vocab fp32 reference
    logits on every ``SAMPLE_EVERY``-th position + the last (recomputed from the stored final hidden with the
    checkpoint's ``lm_head.weight``, the FULL_MODEL_VALIDATION analysis)."""
    from models.demos.motif3.reference import golden_stream as gs
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR
    from models.demos.motif3.tt.weights import HFWeightLoader

    prompts = gs.load_prompt_set(GOLD_BF16 / "prompts.json")
    lp, hp = gs.head_paths(GOLD_FP32, 52)
    t, meta = gs.load_tensors(lp)
    h, _ = gs.load_tensors(hp)
    assert meta["mode"]["dtype"] == "fp32" and not meta.get("early_exit", False), meta
    w = HFWeightLoader(DEFAULT_WEIGHTS_DIR).get("lm_head.weight").float()
    out = {}
    for p in prompts:
        S = len(p.ids)
        rows = torch.tensor(sorted(set(range(0, S, SAMPLE_EVERY)) | {S - 1}))
        ref = h[f"{p.name}.final_hidden"][0][rows].float() @ w.T
        out[p.name] = GoldPrompt(
            p.name,
            list(p.ids),
            t[f"{p.name}.topk_ids"],
            t[f"{p.name}.topk_logits"].float(),
            t[f"{p.name}.argmax"],
            t[f"{p.name}.target_ids"],
            rows,
            ref,
        )
    return out


def row_metrics(g: GoldPrompt, pos: torch.Tensor, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-row metrics of TT logits at positions ``pos`` vs the fp32 golden (``analyze_full_model.py`` definitions):
    tie-aware top-1 agreement, exact argmax, reference margin, NLL of the actual next token (NaN without one: use
    ``has_target``, never ``isfinite``, to select the rows that have one), and the full-vocab PCC on the sampled
    positions. Non-finite logits raise."""
    lg = logits.float()
    assert_sane(lg, f"{g.name} logits", pos.tolist())
    am = lg.argmax(-1)
    ids_k, vals_k, gam = g.topk_ids[pos], g.topk_vals[pos], g.argmax[pos]
    hit = ids_k == am[:, None]
    at = torch.where(hit, vals_k, torch.full_like(vals_k, float("-inf"))).max(-1).values
    agree = (am == gam) | (at == vals_k[:, 0])
    tgt = g.target[pos]
    lse = torch.logsumexp(lg.double(), -1)
    nll = lse - lg.double().gather(1, tgt.clamp_min(0)[:, None])[:, 0]
    nll = torch.where(tgt >= 0, nll, torch.full_like(nll, float("nan")))
    sidx = {int(q): i for i, q in enumerate(g.sampled.tolist())}
    sel = [i for i, q in enumerate(pos.tolist()) if q in sidx]
    pcc = _row_pccs(lg[sel], g.ref_rows[[sidx[int(pos[i])] for i in sel]]) if sel else torch.empty(0)
    return {
        "agree": agree,
        "exact": am == gam,
        "margin": vals_k[:, 0] - vals_k[:, 1],
        "nll": nll,
        "has_target": tgt >= 0,
        "pcc": pcc,
        "argmax": am,
        "pos": pos.clone(),
    }


def pooled(ms: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Pooled metrics of :func:`row_metrics` results. The NLL is averaged over the rows WITH a next token
    (``has_target``); a non-finite NLL or PCC on such a row raises instead of silently dropping out of the mean."""
    agree = torch.cat([m["agree"] for m in ms]).float()
    nll = torch.cat([m["nll"] for m in ms])
    has = torch.cat([m["has_target"] for m in ms])
    pcc = torch.cat([m["pcc"] for m in ms])
    nll = nll[has]
    assert bool(torch.isfinite(nll).all()), f"{int((~torch.isfinite(nll)).sum())} non-finite NLLs on rows with a target"
    assert bool(torch.isfinite(pcc).all()), f"{int((~torch.isfinite(pcc)).sum())} non-finite logit PCCs"
    return {
        "rows": int(agree.numel()),
        "agree": float(agree.mean()),
        "nll": float(nll.mean()),
        "pcc_median": float(pcc.median()) if pcc.numel() else float("nan"),
        "pcc_min": float(pcc.min()) if pcc.numel() else float("nan"),
    }


def compare_streams(a: Sequence[int], ma: Sequence[float], b: Sequence[int], mb: Sequence[float]) -> Tuple[bool, str]:
    """Greedy streams equal, or equal up to a first divergence where either path's logits were a near tie
    (margin < NEAR_TIE): after a flip the continuations differ by construction, so nothing later is compared. A
    non-finite margin up to the first divergence (NaN / Inf logits) fails: ``min`` would otherwise return the finite
    margin and excuse a NaN lane."""
    for i, (x, y) in enumerate(zip(a, b)):
        bad = [f"{tag} {m[i]}" for tag, m in (("a", ma), ("b", mb)) if i < len(m) and not math.isfinite(m[i])]
        if bad:
            return False, f"non-finite logits margin at step {i} ({', '.join(bad)})"
        if x != y:
            near = min(ma[i], mb[i]) < NEAR_TIE
            return (
                near,
                f"diverge at step {i} ({x} vs {y}; margins {ma[i]:.3f} / {mb[i]:.3f}{' near tie' if near else ''})",
            )
    return True, f"identical ({len(a)} tokens)"


@dataclasses.dataclass
class Lane:
    lane: int
    seq: List[int]  # every token so far: prompt + generated
    blocks: List[int]
    margins: List[float] = dataclasses.field(default_factory=list)


class Session:
    """The shared 53-layer generator of the module and the helpers its tests use."""

    def __init__(self, mesh, gen, pool, guard):
        self.mesh, self.gen, self.pool, self.guard = mesh, gen, pool, guard
        self.cfg = gen.cfg
        self.blocks = Blocks(NUM_BLOCKS)
        self._gold: Optional[Dict[str, GoldPrompt]] = None
        self.cold: Dict[str, Dict[str, Any]] = {}  # CP-H (i) results, reused by CP-C
        self._tok = None

    @property
    def gold(self) -> Dict[str, GoldPrompt]:
        if self._gold is None:
            t0 = time.time()
            self._gold = load_goldens()
            log(f"goldens + fp32 reference rows loaded in {time.time() - t0:.1f} s")
        return self._gold

    @property
    def tokenizer(self):
        if self._tok is None:
            from models.demos.motif3.reference.tokenizer import load_tokenizer

            self._tok = load_tokenizer()
        return self._tok

    def programs(self) -> int:
        return int(self.mesh.num_program_cache_entries())

    # ---- prefill ---------------------------------------------------------------------------------------------
    def prefill(self, rows: Sequence[api.PrefillRequest], want: Optional[Dict[int, Tuple[int, int]]] = None):
        """One ``prefill_forward_batch`` call; ``want`` = teacher-forced positions per row (RowCollector)."""
        col = RowCollector(self.gen, want or {}) if want else None
        self.gen.chunk_observer = col
        try:
            t0 = time.time()
            out = self.gen.prefill_forward_batch(list(rows), kv_cache=self.pool)
            dt = time.time() - t0
        finally:
            self.gen.chunk_observer = None
        assert_sane(out, "prefill logits")
        return out, col, dt

    def prefill_observed(self, rows: Sequence[api.PrefillRequest], collector: RowCollector):
        self.gen.chunk_observer = collector
        try:
            out = self.gen.prefill_forward_batch(list(rows), kv_cache=self.pool)
        finally:
            self.gen.chunk_observer = None
        assert_sane(out, "prefill logits")
        return out

    # ---- decode ----------------------------------------------------------------------------------------------
    def greedy(self, lanes: Sequence[Lane], steps: int, *, trace: bool = True, path=None) -> None:
        """``steps`` greedy decode steps for every lane together (the bridge's lane-ordered batch); appends the argmax
        tokens to ``lane.seq`` and the per-step margins to ``lane.margins``. ``path``: a decode path other than the
        serving one (``("plain", "all")``: the non-speculating KV-R trace the session also captures)."""
        for _ in range(steps):
            tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
            pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
            pt = torch.zeros(api.NUM_LANES, WIDTH, dtype=torch.int32)
            for ln in lanes:
                p = len(ln.seq) - 1
                tokens[ln.lane], pos[ln.lane], pt[ln.lane] = ln.seq[-1], p, page_row(ln.blocks, p + 1)
            out = self.gen.decode_forward(
                api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt), kv_cache=self.pool, enable_trace=trace,
                path=path,
            )  # fmt: skip
            assert_sane(out[[ln.lane for ln in lanes]], "decode logits", [ln.lane for ln in lanes])
            for ln in lanes:
                row = out[ln.lane].float()
                ln.seq.append(int(row.argmax()))
                ln.margins.append(_margin(row))

    def teacher_forced(
        self, ids: Sequence[int], lo: int, hi: int, blocks: Sequence[int], *, lane: int, path=None
    ) -> torch.Tensor:
        """Positions ``[lo, hi)`` of ``ids`` through traced decode steps on ``lane`` (token ``ids[p]`` at position
        ``p``; ``blocks`` must hold the KV of ``[0, lo)``): the logits row of every position, ``[hi - lo, V]``."""
        rows = []
        for p in range(int(lo), int(hi)):
            tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
            pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
            pt = torch.zeros(api.NUM_LANES, WIDTH, dtype=torch.int32)
            tokens[lane], pos[lane], pt[lane] = int(ids[p]), p, page_row(blocks, p + 1)
            out = self.gen.decode_forward(
                api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt), kv_cache=self.pool, enable_trace=True,
                path=path,
            )  # fmt: skip
            rows.append(out[lane].clone())
        lg = torch.stack(rows)
        assert_sane(lg, f"teacher-forced decode (lane {lane})", range(int(lo), int(hi)))
        return lg

    def row_mode_greedy(self, lanes: Sequence[Lane], steps: int) -> None:
        """Eager greedy decode in the draft-1 ``row`` KV-write mode (each lane's latent only on its own DP row: the
        no-KV-R negative control of §3.4). Compiles the row-mode update programs: only with no trace alive."""
        import ttnn

        from models.demos.motif3.tt.rope import shard_lanes

        assert not self.gen.trace_captured, "row-mode programs must not compile while a trace is alive"
        m, cfg, mesh = self.gen.model, self.cfg, self.mesh
        for _ in range(steps):
            tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
            pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
            pt = torch.zeros(api.NUM_LANES, WIDTH, dtype=torch.int32)
            for ln in lanes:
                p = len(ln.seq) - 1
                tokens[ln.lane], pos[ln.lane], pt[ln.lane] = ln.seq[-1], p, page_row(ln.blocks, p + 1)
            tok = m.embed.decode_tokens_device(tokens)
            rot = m.rope.rot_idxs_device(pos)
            cur = shard_lanes(pos, cfg, mesh, dtype=ttnn.int32, device=mesh)
            ptd = shard_lanes(pt, cfg, mesh, dtype=ttnn.int32, device=mesh)
            out = m.decode(tok, rot_idxs=rot, cur_pos=cur, page_table=ptd, kv_caches=self.pool)
            logits = m.head.logits_to_host(out)
            for t in (tok, rot, cur, ptd, out):
                ttnn.deallocate(t)
            assert_sane(logits[[ln.lane for ln in lanes]], "row-mode decode logits", [ln.lane for ln in lanes])
            for ln in lanes:
                row = logits[ln.lane].float()
                ln.seq.append(int(row.argmax()))
                ln.margins.append(_margin(row))

    def recapture(self) -> None:
        self.gen.warmup_decode(kv_cache=self.pool, enable_trace=True, page_table_width=WIDTH)


@pytest.fixture(scope="module")
def session():
    import ttnn

    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR, close_motif_mesh, open_motif_mesh

    if not (GOLD_FP32 / "final").is_dir():
        pytest.skip("C2 fp32 golden missing")
    mesh = open_motif_mesh()
    gen = None
    try:
        log_fabric(mesh, "resumed_prefill")
        A = api.DEFAULT_PREFILL_ALIGNMENT
        budget = PP.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, A)
        settings = api.GeneratorSettings(
            max_batch_size=api.NUM_LANES, max_seq_len=MAX_LEN, num_layers=CP_LAYERS, kv_cache_dtype="bfp8",
            weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=BS, weights_source="TT cache only (test)",
            chunked_prefill=True, prefix_caching=True, max_num_batched_tokens=budget,
            long_prefill_token_threshold=budget, spec_tokens=1,
        )  # fmt: skip
        guard = RefuseReads(DEFAULT_WEIGHTS_DIR)
        t0 = time.time()
        gen = MotifGenerator.create(hf_config=None, mesh_device=mesh, settings=settings, source=guard)
        log(
            f"generator: {gen.num_layers} layers + MTP {gen.mtp_enabled} in {time.time() - t0:.1f} s; cache misses "
            f"{gen.model.cache_misses}; refused reads {guard.refused[:3]}"
        )
        assert not gen.model.cache_misses and not guard.refused, "not a TT-cache-only boot"
        assert gen.decode_kv_mode == "all" and gen.mtp_enabled
        assert gen.serving_path == ("spec", "all_split")
        # ONE captured decode trace, as in serving: with a second one (the plain all trace) captured next to it, the
        # prefills after a release / new-program compile / re-capture (CP-L's 32K single shot) returned garbage tiles
        # (MULTI_TRACE_NOTE). CP-X runs the plain all decode eagerly, with no trace alive, instead.
        pool = gen.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BS, num_layers=CP_LAYERS)
        assert gen.prefill_alignment == A, "update the test's budget: A moved (gate G9 per-bucket chunks?)"
        t0 = time.time()
        gen.warmup_prefill(kv_cache=pool, enable_trace=False)
        t_wp = time.time() - t0
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=WIDTH)
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=WIDTH)
        log(
            f"warmup: prefill {t_wp:.1f} s ({len(gen.warmed_shapes)} shapes, MTP fill included), decode eager "
            f"{gen.timings.get('warmup_decode_eager_s', 0):.1f} s, capture "
            f"{gen.timings.get('capture_decode_s', 0):.1f} s;"
            f" program cache {mesh.num_program_cache_entries()}"
        )
        yield Session(mesh, gen, pool, guard)
    finally:
        if gen is not None:
            gen.close()
        close_motif_mesh(mesh)


def _needle_prompt(tok, target_tokens: int) -> Tuple[List[int], str]:
    """A ~``target_tokens`` chat prompt: tt-metal technical reports as the haystack, one needle sentence at 40 % depth,
    a question at the end (thinking off). Returns ``(ids, needle answer)``."""
    from models.demos.motif3.reference.tokenizer import encode_chat

    root = Path(__file__).resolve().parents[4]
    files = [
        "METALIUM_GUIDE.md",
        "tech_reports/LLMs/llms.md",
        "tech_reports/TT-Fabric/TT-Fabric-Architecture.md",
        "tech_reports/TT-Distributed/TT-Distributed-Architecture-1219.md",
        "tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md",
        "tech_reports/tensor_sharding/tensor_sharding.md",
        "tech_reports/MetalProfiler/metal-profiler.md",
        "tech_reports/GEMM_FLOPS/GEMM_FLOPS.md",
        "tech_reports/LLMs/vLLM_integration.md",
        "README.md",
    ]
    text = "\n\n".join((root / f).read_text(errors="ignore") for f in files if (root / f).is_file())
    while len(text) < 8 * target_tokens:
        text = text + "\n\n" + text
    answer = "7319-4426"
    needle = f"\n\nIMPORTANT NOTE: the secret passcode for the blue vault is {answer}. Remember it.\n\n"
    question = (
        "\n\nQuestion: what is the secret passcode for the blue vault mentioned in the documents above? "
        "Answer with the passcode only."
    )

    def build(n_chars: int) -> List[int]:
        body = text[:n_chars]
        cut = body.rfind("\n\n", 0, int(0.4 * n_chars))
        body = body[:cut] + needle + body[cut:]
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": body + question},
        ]
        return encode_chat(msgs, tok, enable_thinking=False)

    lo, hi = 1000, len(text)
    while hi - lo > 200:  # the longest prompt <= target_tokens
        mid = (lo + hi) // 2
        if len(build(mid)) <= target_tokens:
            lo = mid
        else:
            hi = mid
    return build(lo), answer


def _single_shot(
    s: "Session", ids: Sequence[int], blocks: Sequence[int], rows: Tuple[int, int], streams_at: Sequence[int] = ()
) -> RowCollector:
    """The draft-1 single-shot prefill of ``ids`` (one bucket, ``MotifModel.prefill``) into ``blocks``, with the LM head
    on the tiles of positions ``rows`` and the residual streams of ``streams_at`` collected. Compiles the bucket's
    programs when the span cap is below it: the caller must have released the decode trace."""
    import ttnn

    gen, cfg = s.gen, s.cfg
    assert not gen.trace_captured, "the single shot may compile programs: release the decode trace first"
    S = len(ids)
    bucket = cfg.prefill_bucket(S)
    model = gen.model
    tok = model.embed.prefill_tokens_device(torch.tensor(list(ids), dtype=torch.int32), bucket)
    pt = gen._prefill_page_table(page_row(blocks, S), bucket, S)
    X = model.prefill(tok, page_table=pt, kv_caches=s.pool, return_streams=True)
    col = RowCollector(gen, {0: rows}, {0: streams_at})
    col(SimpleNamespace(index=0), SimpleNamespace(start=0, end=S), X)
    for t in (X, tok, pt):
        ttnn.deallocate(t)
    return col


# ---- CP-L ------------------------------------------------------------------------------------------------------------
CPL_TAIL = 512  # the design's top-1 window: the last positions of the prompt
CPL_NLL_ROWS = 2048  # NLL over a longer window (the per-row NLL is noisy)
# Tolerance below the weaker draft-1 floor. The chunked schedules' agreement with the single shot over three sessions
# (2026-10-02/03): 0.905 / 0.914, 0.893 / 0.898, 0.876 / 0.891 (mean 0.896, SD 0.013), the decode floor 0.907 in both
# runs that measured it; the chunked path's own repeat 0.894 / 0.895 (prefill nondeterminism over 4 chunks). 0.05 puts
# the bar ~3 SD below the mean.
CPL_AGREE_TOL = 0.05


@pytest.mark.timeout(CP_TIMEOUT)
def test_cp_l_long_context(session):
    """CP-L (i): a ~32K-token needle prompt: vLLM-chunked with the recommended budget (``span cap - A``, threshold =
    budget, lead decision 4) and with 4096-token steps (the design's threshold variant) vs the draft-1 single-shot
    prefill (bucket 32768, ``MotifModel.prefill``); 64 greedy decode tokens per path (traced).

    TT's own floor at 32K, on the same rows (the design's single-row bars assume a floor TT does not have at full
    depth: prefill is not bitwise reproducible run to run, FULL_MODEL_VALIDATION §5.1, and MoE top-8 near ties flip
    under any numerical perturbation, CP-H):

    * ``single_rep``: the draft-1 single shot again, on other blocks: its run-to-run variability;
    * ``decode``: the last 512 positions teacher-forced through the traced draft-1 decode numerics (the serving decode
      trace) on a single-shot prefix of the first ``S - 512`` tokens. Every key is then a bfp8 cache row, as in an sp1
      chunk; FULL_MODEL_VALIDATION validated these numerics at top-1 0.9717 (prefill path 0.9724) vs the fp32 golden;
    * ``budget_rep`` (reported only): the budget schedule again, the chunked path's own run-to-run variability.

    Asserted, per chunked schedule: (1) top-1 agreement with the single shot over the last 512 positions, on the rows
    whose single-shot margin is >= 0.5, at least the weaker draft-1 floor (min of ``single_rep`` and ``decode`` on the
    same rows) minus ``CPL_AGREE_TOL``; (2) NLL of the actual next tokens over the last 2048 positions at most +1 %
    above the single shot's (one-sided: predicting the text better is not a failure); (3) the needle found whenever
    the single shot finds it, and a greedy decode equal to the single shot's except a near-tie divergence, or the same
    answer up to the end of turn. Reported: the design's bars (top-1 >= 0.98 on those rows, NLL within +-1 %) for the
    chunked paths and for the floors themselves, row PCCs, and the two chunk schedules against each other. The single
    shots compile programs after the warmup, so the traces are released first and re-captured after them (nothing
    compiles while a trace is alive). CP-L (ii) checks the long-context mechanics against the CPU reference on the
    truncated model."""
    s, gen, cfg = session, session.gen, session.cfg
    ids, answer = _needle_prompt(s.tokenizer, 32000)
    S = len(ids)
    tail, nll_rows = CPL_TAIL, CPL_NLL_ROWS
    budget = PP.recommended_budget(gen.max_prefill_span, gen.prefill_alignment)
    log(f"CP-L: needle prompt of {S} tokens; budget {budget}")
    fails = []
    # (a) single shot (draft-1 path, bucket 32768), no trace alive
    gen.release_traces()
    blk_a = s.blocks.take(api.cdiv(S + 64, BS))
    bucket = cfg.prefill_bucket(S)
    t0 = time.time()
    col = _single_shot(s, ids, blk_a, (S - nll_rows, S))
    t_single = time.time() - t0
    ref_all = col.take(0, S - nll_rows, S)
    ref = ref_all[-tail:]
    log(f"CP-L (a) single shot (bucket {bucket}): {t_single:.1f} s incl. its first compile")

    def chunked(blk: Sequence[int], step: int, lane0: int) -> Dict[str, Any]:
        """One vLLM chunk schedule through prefill_forward_batch (one call per step, a new lane per call)."""
        ends = list(range(step, S, step)) + [S]
        st, t1 = 0, time.time()
        for k, e in enumerate(ends):
            want = {0: (S - nll_rows, S)} if e == S else None
            out, c, _ = s.prefill([prefill_request((k * 9 + lane0) % 32, ids, e, blk, start=st)], want)
            st = e
        allrows = c.take(0, S - nll_rows, S)
        return dict(blocks=blk, logits=allrows[-tail:], all=allrows, last=out[0], seconds=time.time() - t1,
                    calls=len(ends))  # fmt: skip

    # (b) / (c): the vLLM chunk schedules
    paths: Dict[str, Dict[str, Any]] = {}
    for name, step in (("budget", budget), ("t4096", 4096)):
        paths[name] = chunked(s.blocks.take(api.cdiv(S + 64, BS)), step, 1)
        log(f"CP-L ({name}) {paths[name]['calls']} calls of {step}: {paths[name]['seconds']:.1f} s")
    # TT's floors on one block set, reused in turn (a run's KV is dead once its rows are taken)
    scratch = s.blocks.take(api.cdiv(S + 64, BS))
    floors: Dict[str, Dict[str, Any]] = {}
    t0 = time.time()
    rep_all = _single_shot(s, ids, scratch, (S - nll_rows, S)).take(0, S - nll_rows, S)
    floors["single_rep"] = dict(all=rep_all, logits=rep_all[-tail:])
    floors["budget_rep"] = chunked(scratch, budget, 5)
    _single_shot(s, ids[: S - tail], scratch, (0, 0))  # the decode floor's prefix: positions [0, S - 512)
    log(f"CP-L floors: single shot repeat, budget repeat, decode prefix in {time.time() - t0:.1f} s")
    s.recapture()
    pc0 = s.programs()
    t0 = time.time()
    floors["decode"] = dict(logits=s.teacher_forced(ids, S - tail, S, scratch, lane=5))
    log(f"CP-L decode floor: {tail} teacher-forced decode steps (traced) in {time.time() - t0:.1f} s")
    # metrics: top-1 agreement over the last 512 positions ("confident": the first argument's margin >= 0.5), NLL of
    # the actual next tokens (last 2048; last 512 where only those exist), row PCC
    tgt_all = torch.tensor(ids[S - nll_rows + 1 :] + [-1])

    def nll(lg: torch.Tensor) -> float:  # mean over the last lg.shape[0] positions
        tg = tgt_all[-lg.shape[0] :]
        lg, v = lg.double(), tg >= 0
        return float((torch.logsumexp(lg[v], -1) - lg[v].gather(1, tg[v][:, None])[:, 0]).mean())

    def agreement(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, int]:
        keep = torch.tensor([_margin(r) for r in a]) >= NEAR_TIE
        ag = a.float().argmax(-1) == b.float().argmax(-1)
        return float(ag.float().mean()), float(ag[keep].float().mean()), int(keep.sum())

    def compare(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
        a_all, a_keep, n_keep = agreement(a, b)
        pc = _row_pccs(b.float(), a.float())
        return dict(agree_all=a_all, agree_keep=a_keep, n_keep=n_keep, pcc_median=float(pc.median()),
                    pcc_min=float(pc.min()), nll_tail=nll(b))  # fmt: skip

    def line(m: Dict[str, float]) -> str:
        return (f"last {tail} top-1 {m['agree_all']:.4f} ({m['agree_keep']:.4f} on {m['n_keep']} rows with margin >= "
                f"{NEAR_TIE}; design bar 0.98 {'met' if m['agree_keep'] >= 0.98 else 'NOT met'}); row PCC median "
                f"{m['pcc_median']:.5f} min {m['pcc_min']:.5f}")  # fmt: skip

    n_ref, n_ref_tail = nll(ref_all), nll(ref)
    for name in ("single_rep", "decode"):
        f = floors[name]
        f.update(compare(ref, f["logits"]))
        if "all" in f:
            f["nll"] = nll(f["all"])
            f_nll = f"NLL (last {nll_rows}) {f['nll']:.4f} vs {n_ref:.4f} ({100 * (f['nll'] / n_ref - 1):+.2f} %)"
        else:
            f_nll = (f"NLL (last {tail}) {f['nll_tail']:.4f} vs {n_ref_tail:.4f} "
                     f"({100 * (f['nll_tail'] / n_ref_tail - 1):+.2f} %)")  # fmt: skip
        log(f"CP-L floor {name} vs single shot: {line(f)}; {f_nll}")
    floor_name = min(("single_rep", "decode"), key=lambda k: floors[k]["agree_keep"])
    floor_keep = floors[floor_name]["agree_keep"]
    br = compare(paths["budget"]["logits"], floors["budget_rep"]["logits"])
    n_br = nll(floors["budget_rep"]["all"])
    log(f"CP-L floor budget_rep vs budget (the chunked path's own run-to-run): {line(br)}; NLL {n_br:.4f}")
    design = []
    for name, p in paths.items():
        p.update(compare(ref, p["logits"]))
        p["nll"] = nll(p["all"])
        p["nll_rel"] = p["nll"] / n_ref - 1
        design.append(p["agree_keep"] >= 0.98 and abs(p["nll_rel"]) <= 0.01)
        bar = floor_keep - CPL_AGREE_TOL
        log(
            f"CP-L ({name}) vs single shot: {line(p)}; asserted bar >= the draft-1 floor {floor_keep:.4f} "
            f"({floor_name}) - {CPL_AGREE_TOL} = {bar:.4f}: {'met' if p['agree_keep'] >= bar else 'NOT met'}; NLL "
            f"(last {nll_rows}) {p['nll']:.4f} vs {n_ref:.4f} ({100 * p['nll_rel']:+.2f} %; design +-1 % "
            f"{'met' if abs(p['nll_rel']) <= 0.01 else 'NOT met'}); NLL (last {tail}) {p['nll_tail']:.4f} vs "
            f"{n_ref_tail:.4f}"
        )
        if not p["agree_keep"] >= bar:
            fails.append(
                f"CP-L {name}: top-1 {p['agree_keep']:.4f} on the confident rows < the draft-1 floor {floor_keep:.4f} "
                f"({floor_name}) - {CPL_AGREE_TOL}"
            )
        if not p["nll_rel"] <= 0.01:  # one-sided: a chunked path that predicts the text worse than the single shot
            fails.append(f"CP-L {name}: NLL {100 * p['nll_rel']:+.2f} % above the single shot (bar +1 %)")
    bt = compare(paths["t4096"]["logits"], paths["budget"]["logits"])
    n_b, n_t = paths["budget"]["nll"], paths["t4096"]["nll"]
    log(
        f"CP-L budget vs t4096 (both sp1): {line(bt)}; NLL {n_b:.4f} vs {n_t:.4f} ({100 * (n_b / n_t - 1):+.2f} %); "
        f"the design's CP-L bars met by {sum(design)} of {len(design)} chunked paths"
    )
    # 64 greedy decode tokens per path (traced), all three lanes in one batch
    lanes = {"single": Lane(2, list(ids) + [int(ref[-1].float().argmax())], blk_a, [_margin(ref[-1])])}
    for k, (name, p) in enumerate(paths.items()):
        lanes[name] = Lane(11 + 9 * k, list(ids) + [int(p["last"].float().argmax())], p["blocks"], [_margin(p["last"])])
    s.greedy(list(lanes.values()), 63)
    assert s.programs() == pc0, "a program compiled after the re-capture"
    eot = s.tokenizer.convert_tokens_to_ids("<|endofturn|>")

    def answer_ids(seq):  # the generated answer up to (incl.) the first end of turn
        g = seq[S:]
        return g[: g.index(eot) + 1] if eot in g else g

    texts = {k: s.tokenizer.decode(answer_ids(v.seq)) for k, v in lanes.items()}
    for name in paths:
        ok, why = compare_streams(
            lanes["single"].seq[S:], lanes["single"].margins, lanes[name].seq[S:], lanes[name].margins
        )
        same_answer = answer_ids(lanes[name].seq) == answer_ids(lanes["single"].seq)
        log(f"CP-L decode {name} vs single shot: {why}; answer up to the end of turn identical: {same_answer}")
        if not (ok or same_answer):
            fails.append(f"CP-L {name}: greedy decode {why} and a different answer")
    for k, t in texts.items():
        log(f"CP-L {k}: needle {'FOUND' if answer in t else 'not found'}; answer {t[:120]!r}")
    if answer in texts["single"]:
        fails += [f"CP-L {k}: the needle the single shot finds is lost" for k in paths if answer not in texts[k]]
    assert not fails, "\n".join(fails)


# ---- CP-L (ii): the truncated model (layers 0-3, BF16 on disk) vs the CPU reference at 16K / 32K positions ----------
LONG_REF_DIR = PROJECT_ROOT / "tt_cache" / "test" / "resumed_prefill"
LONG_REF_LAYERS = (0, 1, 2, 3)
LONG_TARGET = 32000
LONG_REF_CHUNK = 2048


def long_ref_positions(S: int) -> List[int]:
    """Sampled positions of the long reference: every 256th, both sides of 16384 and the last 32."""
    return sorted(set(range(0, S, 256)) | set(range(16384 - 8, min(S, 16384 + 8))) | set(range(S - 32, S)))


def long_ref_path(ids: Sequence[int]) -> Path:
    import hashlib

    h = hashlib.sha256(torch.tensor(list(ids), dtype=torch.int32).numpy().tobytes()).hexdigest()[:16]
    return LONG_REF_DIR / f"long_ref_L0-3_{len(ids)}_{h}.pt"


@pytest.mark.timeout(7200)
@pytest.mark.skipif(
    os.environ.get("MOTIF3_BUILD_LONG_REF") != "1",
    reason="minutes of CPU: set MOTIF3_BUILD_LONG_REF=1 (scripts/hostrun.sh) to build the CP-L (ii) data",
)
def test_host_build_long_reference():
    """CP-L (ii) data: the CPU reference prefix model (layers 0-3, bf16, real weights) on the 32K needle prompt in
    2048-token chunks through its latent cache (chunked = single-shot for the reference: one causal computation):
    the residual streams after layer 3 and the reference head's fp32 logits at :func:`long_ref_positions`."""
    from models.demos.motif3.reference.tokenizer import load_tokenizer
    from models.demos.motif3.reference.weights import load_reference_model

    torch.set_num_threads(min(32, os.cpu_count() or 8))
    ids, _ = _needle_prompt(load_tokenizer(), LONG_TARGET)
    path = long_ref_path(ids)
    if path.is_file():
        log(f"long reference exists: {path}")
        return
    S = len(ids)
    pos = long_ref_positions(S)
    ref = load_reference_model(layer_ids=LONG_REF_LAYERS, dtype=torch.bfloat16, lazy_experts=True)
    cache = ref.new_cache(1, S)
    streams, hidden = {}, {}
    t0 = time.time()
    with torch.no_grad():
        for a in range(0, S, LONG_REF_CHUNK):
            b = min(S, a + LONG_REF_CHUNK)
            h, x = ref.model(
                torch.tensor([list(ids[a:b])]), positions=torch.arange(a, b)[None], cache=cache, return_streams=True
            )
            for p in pos:
                if a <= p < b:
                    streams[p] = x[0, p - a].float().clone()  # [4, 4096]
                    hidden[p] = h[0, p - a].clone()
            log(f"long reference: positions [{a}, {b}) in {time.time() - t0:.0f} s")
        hs = torch.stack([hidden[p] for p in pos])
        logits = torch.nn.functional.linear(hs, ref.lm_head.weight).float()
    LONG_REF_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "ids": list(ids),
            "positions": pos,
            "streams": torch.stack([streams[p] for p in pos]),
            "logits": logits.to(torch.bfloat16),
            "layers": LONG_REF_LAYERS,
            "chunk": LONG_REF_CHUNK,
        },
        tmp,
    )
    os.replace(tmp, path)
    log(f"long reference ({S} tokens, {len(pos)} positions) written to {path} in {time.time() - t0:.0f} s")


@pytest.mark.timeout(CP_TIMEOUT)
def test_cp_l_truncated_vs_reference(session):
    """CP-L (ii) (``MOTIF3_CP_LAYERS=4`` only): the truncated TT model (layers 0-3) on the 32K needle prompt and on its
    first 16384 tokens, single shot (draft-1 buckets 32768 / 16384) and vLLM-chunked (recommended budget; 4096 steps),
    vs the CPU reference prefix model: residual-stream PCC after layer 3 >= 0.99 at every sampled position (the
    truncated-state bar, design §5.3), and every chunked path no further from the reference than the single shot (within
    0.001 at the worst position and in the median). Reported: the design's chunked vs single-shot TT streams >= 0.999
    (2026-10-02: min 0.9981-0.9987, median 0.99975-0.99995; the worst positions are the single shot's worst too)."""
    if CP_LAYERS != len(LONG_REF_LAYERS):
        pytest.skip(f"needs MOTIF3_CP_LAYERS={len(LONG_REF_LAYERS)} (the reference prefix model)")
    s, gen = session, session.gen
    ids, _ = _needle_prompt(s.tokenizer, LONG_TARGET)
    path = long_ref_path(ids)
    if not path.is_file():
        pytest.skip(
            f"no long reference {path}: MOTIF3_BUILD_LONG_REF=1 scripts/hostrun.sh -- python -m pytest "
            f"-p no:cacheprovider -q {Path(__file__).name} -k build_long_reference"
        )
    data = torch.load(path)
    assert data["ids"] == list(ids)
    pos_all = list(data["positions"])
    ref_st = {p: data["streams"][i] for i, p in enumerate(pos_all)}
    ref_lg = {p: data["logits"][i].float() for i, p in enumerate(pos_all)}
    budget = PP.recommended_budget(gen.max_prefill_span, gen.prefill_alignment)
    fails = []
    gen.release_traces()
    try:
        for S in (16384, len(ids)):
            sub = ids[:S]
            pos = [p for p in pos_all if p < S]
            cols = {"single": _single_shot(s, sub, s.blocks.take(api.cdiv(S, BS)), (S - 32, S), pos)}
            for name, step in (("budget", budget), ("t4096", 4096)):
                blk = s.blocks.take(api.cdiv(S, BS))
                st = 0
                col = RowCollector(gen, {0: (S - 32, S)}, {0: pos})
                for k, e in enumerate(list(range(step, S, step)) + [S]):
                    want = {0: (S - 32, S)} if e == S else None
                    sub_col = RowCollector(gen, want or {}, {0: [p for p in pos if st <= p < e]})
                    s.prefill_observed([prefill_request((7 * k + 3) % 32, sub, e, blk, start=st)], sub_col)
                    col.streams.update(sub_col.streams)
                    col.rows.update(sub_col.rows)
                    st = e
                cols[name] = col
            vs_ref = {}
            for name, col in cols.items():
                st_p = torch.tensor([_pcc(col.streams[(0, p)], ref_st[p]) for p in pos])
                lg_p = torch.tensor([_pcc(col.rows[(0, p)].float(), ref_lg[p]) for p in pos if (0, p) in col.rows])
                vs_ref[name] = st_p
                worst = pos[int(st_p.argmin())]
                log(
                    f"CP-L (ii) S={S} {name} vs CPU reference (layers 0-3): stream PCC min {float(st_p.min()):.5f} (at "
                    f"{worst}) median {float(st_p.median()):.5f} over {len(pos)} positions; last-32 logits PCC min "
                    f"{float(lg_p.min()):.5f}"
                )
                if not float(st_p.min()) >= 0.99:
                    fails.append(f"CP-L (ii) S={S} {name}: stream PCC {float(st_p.min()):.5f} < 0.99 at {worst}")
            one = vs_ref["single"]
            for name in ("budget", "t4096"):
                col, st_p = cols[name], vs_ref[name]
                tt = torch.tensor([_pcc(col.streams[(0, p)], cols["single"].streams[(0, p)]) for p in pos])
                # asserted: the chunked path no further from the reference than the single shot (within 0.001, at the
                # worst position and in the median); reported: the design's chunked vs single-shot TT >= 0.999
                closer = (
                    float(st_p.min()) >= float(one.min()) - 0.001
                    and float(st_p.median()) >= float(one.median()) - 0.001
                )
                log(
                    f"CP-L (ii) S={S} {name} vs single shot TT: stream PCC min {float(tt.min()):.6f} median "
                    f"{float(tt.median()):.6f} (design bar 0.999 {'met' if float(tt.min()) >= 0.999 else 'NOT met'}); "
                    f"vs the reference within 0.001 of the single shot: {closer}"
                )
                if not closer:
                    fails.append(
                        f"CP-L (ii) S={S} {name}: further from the reference than the single shot (min "
                        f"{float(st_p.min()):.5f} vs {float(one.min()):.5f}, median {float(st_p.median()):.5f} vs "
                        f"{float(one.median()):.5f})"
                    )
    finally:
        s.recapture()
    assert not fails, "\n".join(fails)


# ---- CP-H ------------------------------------------------------------------------------------------------------------
COLD_LANES = [0, 9, 18, 27, 4, 13]
HIT_LANES = [8, 17, 26, 3, 12, 21]  # another DP row than the cold lane of the same prompt
REPEAT_LANES = [16, 25, 2, 11, 20, 29]


def _kl(ref: torch.Tensor, got: torch.Tensor) -> float:
    """KL(softmax(ref) || softmax(got)) of one logits row (fp64)."""
    lr, lg = torch.log_softmax(ref.double(), -1), torch.log_softmax(got.double(), -1)
    return float((lr.exp() * (lr - lg)).sum())


def _cold_runs(s: Session) -> Dict[str, Dict[str, Any]]:
    """CP-H (i), shared with CP-C: each C2 prompt cold on its own blocks, teacher-forced logits on every row."""
    for i, (name, g) in enumerate(s.gold.items()):
        if name in s.cold:
            continue
        blk = s.blocks.take(api.cdiv(g.S + DECODE_STEPS + 1, BS))
        out, col, dt = s.prefill([prefill_request(COLD_LANES[i], g.ids, g.S, blk)], {0: (0, g.S)})
        rows = col.take(0, 0, g.S)
        m = row_metrics(g, torch.arange(g.S), rows)
        assert torch.equal(rows[-1], out[0]), "the serving logits are the teacher-forced last row"
        s.cold[name] = dict(blocks=blk, last=out[0], metrics=m, sampled_rows=rows[g.sampled].clone(), seconds=dt)
        log(
            f"cold {name} (S={g.S}): {dt:.2f} s; top-1 {float(m['agree'].float().mean()):.4f}, NLL "
            f"{float(m['nll'][m['has_target']].mean()):.4f}, PCC median {float(m['pcc'].median()):.5f}"
        )
    return s.cold


@pytest.mark.timeout(CP_TIMEOUT)
def test_cp_h_prefix_reuse(session):
    """CP-H: (i) cold prefill + 32 greedy tokens; (ii) the same prompt with ``start = floor((S - 1) / 64) * 64`` on the
    first run's blocks (another DP row): the same argmax unless margin < 0.5, greedy tokens identical except a
    near-tie divergence; (iii) ``P + X`` cold, then ``P + Y`` (= the C2 prompt, ``P`` its first ``floor(S / 2 / 64)``
    blocks) hitting ``P``: the teacher-forced suffix rows vs the fp32 golden within 0.3 pt top-1 and +-0.5 % NLL of the
    cold TT rows (pooled over the 6 prompts).

    Reported, not asserted: the design's last-token logits PCC >= 0.999 for (ii), next to a cold repeat (the identical
    prefill on other blocks: TT's run-to-run floor) and both rows' PCC / KL vs the fp32 reference. Single rows at full
    depth are chaotic: an sp1 row reads bfp8 cache keys where the single shot uses its bf16 in-flight latents, and that
    perturbation flips MoE top-8 selections at near ties (2026-10-02, en_technical position 892: 8th / 9th expert score
    gaps of 1e-6 .. 1e-3 in many layers; the hit at 832 flips 1-4 experts per layer from layer 3 on and ends at KL 0.83
    vs the fp32 reference, while the hits at 768 / 640 / 384 and the draft-1 decode path stay at the cold run's KL
    0.005; math_word_problem's last row: hits KL 0.11-0.19, decode path 0.23, cold 0.02; ko_passage: two identical
    cold prefills differ at PCC 0.931). Many-row statistics (iii), CP-C are the meaningful equality bars."""
    s = session
    pc0 = s.programs()
    cold = _cold_runs(s)
    fails = []
    lanes = []
    design_bar = []
    for i, (name, g) in enumerate(s.gold.items()):
        c = cold[name]
        k = (g.S - 1) // BS
        blk = c["blocks"][:k] + s.blocks.take(api.cdiv(g.S + DECODE_STEPS + 1, BS) - k)
        out, _, dt = s.prefill([prefill_request(HIT_LANES[i], g.ids, g.S, blk, start=k * BS)])
        plan = s.gen.last_prefill.jobs[0].plan
        # the floor: the identical cold prefill again (other lane, other blocks); prefill is not bitwise reproducible
        # run to run at full depth (FULL_MODEL_VALIDATION §5.1: last-token logits move by up to 4.8 between repeats)
        rep, _, _ = s.prefill([prefill_request(REPEAT_LANES[i], g.ids, g.S, s.blocks.take(api.cdiv(g.S, BS)))])
        ref = g.ref_rows[-1]  # the fp32 reference logits of position S - 1 (always a sampled row)
        lc, lh, lr = c["last"].float(), out[0].float(), rep[0].float()
        p, p_rep = _pcc(lc, lh), _pcc(lc, lr)
        r_c, r_h, r_r = _pcc(lc, ref), _pcc(lh, ref), _pcc(lr, ref)
        k_c, k_h, k_r = _kl(ref, lc), _kl(ref, lh), _kl(ref, lr)
        same = int(lh.argmax()) == int(lc.argmax())
        mg = _margin(c["last"])
        # hard bar: the design's argmax rule (+ the greedy streams below, + the pooled rows of (iii) and CP-C). The
        # design's single-row PCC >= 0.999 is reported: no TT path meets it reliably at full depth (module docstring,
        # "single rows"): MoE top-8 near ties flip under any perturbation (the cold repeat misses it too)
        ok = same or mg < NEAR_TIE
        design_bar.append(p >= 0.999)
        log(
            f"CP-H (ii) {name}: hit {k * BS} ({[(ch.start, ch.bucket, ch.path) for ch in plan.chunks]}, {dt:.2f} s): "
            f"last-token PCC hit/cold {p:.5f} (repeat/cold {p_rep:.5f}; design bar 0.999 "
            f"{'met' if p >= 0.999 else 'NOT met'}); vs fp32 PCC cold {r_c:.5f} hit {r_h:.5f} repeat {r_r:.5f}, KL "
            f"cold "
            f"{k_c:.4f} hit {k_h:.4f} repeat {k_r:.4f}; argmax {'same' if same else 'DIFF'} (margin {mg:.3f}) -> "
            f"{'ok' if ok else 'FAIL'}"
        )
        if not ok:
            fails.append(
                f"CP-H (ii) {name}: argmax differs at margin {mg:.3f} (PCC vs fp32 hit {r_h:.5f}, cold " f"{r_c:.5f})"
            )
        lanes.append(
            (
                name,
                Lane(COLD_LANES[i], list(g.ids) + [int(c["last"].float().argmax())], c["blocks"], [mg]),
                Lane(HIT_LANES[i], list(g.ids) + [int(out[0].float().argmax())], blk, [_margin(out[0])]),
            )
        )
    s.greedy([ln for _, a, b in lanes for ln in (a, b)], DECODE_STEPS - 1)
    for name, a, b in lanes:
        ok, why = compare_streams(a.seq[len(s.gold[name].ids) :], a.margins, b.seq[len(s.gold[name].ids) :], b.margins)
        log(f"CP-H (i)/(ii) {name}: {DECODE_STEPS} greedy tokens cold vs hit: {why}")
        if not ok:
            fails.append(f"CP-H {name}: greedy decode {why}")
    # (iii) P + X cold, then P + Y hit on P
    rng = random.Random(5)
    vocab_src = [t for g in s.gold.values() for t in g.ids]
    hit_m, cold_m = [], []
    for i, (name, g) in enumerate(s.gold.items()):
        k = max(1, g.S // 2 // BS)
        P = g.ids[: k * BS]
        X = [rng.choice(vocab_src) for _ in range(200)]
        blk_px = s.blocks.take(api.cdiv(len(P) + len(X), BS))
        s.prefill([prefill_request(30, P + X, len(P) + len(X), blk_px)])
        blk = blk_px[:k] + s.blocks.take(api.cdiv(g.S, BS) - k)
        _, col, _ = s.prefill([prefill_request(6, g.ids, g.S, blk, start=k * BS)], {0: (k * BS, g.S)})
        pos = torch.arange(k * BS, g.S)
        hm = row_metrics(g, pos, col.take(0, k * BS, g.S))
        cm = {kk: v[pos] for kk, v in cold[name]["metrics"].items() if kk != "pcc"}
        cm["pcc"] = torch.empty(0)  # (the cold rows' PCC is per sampled row: not comparable row by row here)
        hit_m.append(hm)
        cold_m.append(cm)
        log(
            f"CP-H (iii) {name}: hit {k * BS} of {g.S}: suffix top-1 {float(hm['agree'].float().mean()):.4f} (cold "
            f"{float(cm['agree'].float().mean()):.4f})"
        )
    ph, pc = pooled(hit_m), pooled(cold_m)
    d_agree, d_nll = ph["agree"] - pc["agree"], ph["nll"] / pc["nll"] - 1
    log(
        f"CP-H (iii) pooled suffix rows ({ph['rows']}): top-1 {ph['agree']:.4f} vs cold {pc['agree']:.4f} "
        f"({100 * d_agree:+.2f} pt); NLL {ph['nll']:.4f} vs {pc['nll']:.4f} ({100 * d_nll:+.2f} %); PCC median "
        f"{ph['pcc_median']:.5f}"
    )
    if not d_agree >= -0.003:
        fails.append(f"CP-H (iii): top-1 {100 * d_agree:+.2f} pt vs cold (bar -0.3)")
    if not abs(d_nll) <= 0.005:
        fails.append(f"CP-H (iii): NLL {100 * d_nll:+.2f} % vs cold (bar 0.5 %)")
    log(f"CP-H (ii): the design's last-token PCC >= 0.999 met on {sum(design_bar)} of {len(design_bar)} prompts")
    assert s.programs() == pc0, f"program cache grew after the capture: {pc0} -> {s.programs()}"
    assert not fails, "\n".join(fails)


# ---- CP-C ------------------------------------------------------------------------------------------------------------
def _schedule_ends(kind: str, S: int) -> List[int]:
    if kind == "a128":
        ends = list(range(128, S, 128))
    elif kind == "u200":
        ends = list(range(200, S, 200))
    elif kind == "u333":
        ends = list(range(333, S, 333))
    elif kind == "b64k":  # growing multiples of 64: 64, 192, 384, 640, ...
        ends, e, k = [], 0, 1
        while e + 64 * k < S:
            e += 64 * k
            ends.append(e)
            k += 1
    else:
        raise ValueError(kind)
    return ends + [S]


@pytest.mark.timeout(CP_TIMEOUT)
@pytest.mark.parametrize("kind", ["a128", "u200", "u333", "b64k"])
def test_cp_c_chunked_vs_golden(session, kind):
    """CP-C: the 6 C2 prompts prefilled in forced vLLM chunk schedules (one ``prefill_forward_batch`` call per chunk,
    explicit start, another lane per call); teacher-forced logits on every row of each call. Pooled over the prompts:
    top-1 vs the fp32 golden >= cold TT - 0.3 pt; NLL within +-0.5 % of cold TT; full-vocab row PCC vs the fp32
    reference median >= 0.997."""
    s = session
    pc0 = s.programs()
    cold = _cold_runs(s)
    ms, cs, vs_cold = [], [], []
    calls = 0
    t0 = time.time()
    for i, (name, g) in enumerate(s.gold.items()):
        blk = s.blocks.take(api.cdiv(g.S, BS))
        rows = []
        st = 0
        for j, e in enumerate(_schedule_ends(kind, g.S)):
            _, col, _ = s.prefill([prefill_request((5 * i + 9 * j) % 32, g.ids, e, blk, start=st)], {0: (st, e)})
            rows.append(col.take(0, st, e))
            st = e
            calls += 1
        lg = torch.cat(rows)
        m = row_metrics(g, torch.arange(g.S), lg)
        ms.append(m)
        cs.append(cold[name]["metrics"])
        vs_cold.append(_row_pccs(lg[g.sampled].float(), cold[name]["sampled_rows"].float()))
    p, pc = pooled(ms), pooled(cs)
    vc = torch.cat(vs_cold)
    d_agree, d_nll = p["agree"] - pc["agree"], p["nll"] / pc["nll"] - 1
    log(
        f"CP-C {kind}: {calls} calls in {time.time() - t0:.1f} s; top-1 {p['agree']:.4f} vs cold {pc['agree']:.4f} "
        f"({100 * d_agree:+.2f} pt); NLL {p['nll']:.4f} vs {pc['nll']:.4f} ({100 * d_nll:+.2f} %); PCC vs fp32 median "
        f"{p['pcc_median']:.5f} (cold {pc['pcc_median']:.5f}); PCC vs cold TT median {float(vc.median()):.5f} min "
        f"{float(vc.min()):.5f}"
    )
    fails = []
    if not d_agree >= -0.003:
        fails.append(f"CP-C {kind}: top-1 {100 * d_agree:+.2f} pt vs cold (bar -0.3)")
    if not abs(d_nll) <= 0.005:
        fails.append(f"CP-C {kind}: NLL {100 * d_nll:+.2f} % vs cold (bar 0.5 %)")
    if not p["pcc_median"] >= 0.997:
        fails.append(f"CP-C {kind}: PCC median {p['pcc_median']:.5f} < 0.997")
    assert s.programs() == pc0, f"program cache grew after the capture: {pc0} -> {s.programs()}"
    assert not fails, "\n".join(fails)


# ---- CP-X ------------------------------------------------------------------------------------------------------------
CPX_USER = (
    "Write a detailed, multi-paragraph explanation of how a household refrigerator works: the refrigeration "
    "cycle, the compressor, the condenser, the expansion valve, the evaporator, and why the back of the "
    "fridge feels warm."
)
CPX_TOL = 0.003  # KV-R hit vs the sp1 floor (suffix-row PCC median); the stale-KV negative control sits ~0.06 below


@pytest.mark.timeout(CP_TIMEOUT)
def test_cp_x_cross_row_hit(session):
    """CP-X: request A (lane 0, DP row 0) prefills and decodes 200 greedy tokens through the serving decode (the spec
    trace, KV-R ``all_split``); request B = A's prompt + A's answer + a new turn on lane 8 (DP row 1) hits the blocks
    A's decode wrote. The same for the non-speculating production decode (review 2026-10-03): A' (lane 2) decodes 100
    tokens through the plain ``all`` path (KV-R, no MTP; eager, with no trace alive: a second captured trace is
    avoided, ``MULTI_TRACE_NOTE``; eager == traced bitwise, G-S6), B' on lane 10 (DP row 1). Each B against:

    * a cold B on a third DP row, and a cold repeat (TT's run-to-run floor; reported);
    * the **sp1 floor**: B hitting blocks a PREFILL wrote (B's hit prefix prefilled cold into new blocks, then B from
      the same start on them): the same sp1 chunk, the same bfp8 cache keys, no decode-written block. A hit on blocks
      another DP row decode-wrote is as good as a hit can be iff it is as close to the cold B as this one.

    Asserted with KV-R (both decode paths): the argmax of the cold B unless its margin is below 0.5; the teacher-forced
    suffix rows' logit PCC median (vs the cold B) at least the sp1 floor's minus ``CPX_TOL``; 32 greedy tokens (same
    decode path) identical to the cold B's except a near-tie divergence. Reported: the design's last-token PCC >=
    0.999, the cold repeat. Negative control (design §3.4): A'' decoded in the draft-1 ``row`` mode (100 eager steps,
    no trace alive; its latents exist only on DP row 0): B'' (lane 9, DP row 1) reads stale KV on 3 of 4 DP rows, so
    its suffix rows must fall clearly below the sp1 floor (median lower by more than 0.01) or flip a confident
    argmax."""
    from models.demos.motif3.reference.tokenizer import encode_chat

    s = session
    tok = s.tokenizer
    prompt = encode_chat(
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": CPX_USER}],
        tok,
        enable_thinking=False,
    )
    turn2 = encode_chat(
        [{"role": "user", "content": "Thanks. Now summarize that in two sentences."}], tok, enable_thinking=False
    )
    L = len(prompt)
    fails = []
    pc0 = s.programs()

    def med(x: torch.Tensor) -> float:
        return float(x.median())

    def run_pair(A: Lane, D: int, hit_lane: int, cold_lane: int, tag: str) -> Dict[str, Any]:
        B = A.seq[: L + D] + turn2
        k = (L + D) // BS
        lo = k * BS
        want = {0: (lo, len(B))}
        blk_hit = A.blocks[:k] + s.blocks.take(api.cdiv(len(B) + DECODE_STEPS, BS) - k)
        out_h, col_h, _ = s.prefill([prefill_request(hit_lane, B, len(B), blk_hit, start=lo)], want)
        blk_cold = s.blocks.take(api.cdiv(len(B) + DECODE_STEPS, BS))
        out_c, col_c, _ = s.prefill([prefill_request(cold_lane, B, len(B), blk_cold)], want)
        out_r, col_r, _ = s.prefill(
            [prefill_request((cold_lane + 13) % 32, B, len(B), s.blocks.take(api.cdiv(len(B), BS)))], want
        )  # TT's run-to-run floor: the identical cold prefill again
        blk_p = s.blocks.take(api.cdiv(len(B), BS))  # the sp1 floor: B's hit prefix PREFILL-written, then the same hit
        s.prefill([prefill_request((hit_lane + 7) % 32, B, lo, blk_p)])
        out_p, col_p, _ = s.prefill([prefill_request((hit_lane + 16) % 32, B, len(B), blk_p, start=lo)], want)
        rows_c = col_c.take(0, lo, len(B)).float()
        rp = _row_pccs(col_h.take(0, lo, len(B)).float(), rows_c)
        fl = _row_pccs(col_r.take(0, lo, len(B)).float(), rows_c)
        sp = _row_pccs(col_p.take(0, lo, len(B)).float(), rows_c)
        lc = out_c[0].float()
        p, p_rep, p_sp = (_pcc(o[0].float(), lc) for o in (out_h, out_r, out_p))
        same = int(out_h[0].float().argmax()) == int(lc.argmax())
        mg = _margin(out_c[0])
        log(
            f"CP-X {tag}: B of {len(B)} tokens hits {k} blocks ({lo - L} decode-written positions) on lane {hit_lane}: "
            f"argmax {'same' if same else 'DIFF'} (cold margin {mg:.3f}); last-token PCC vs cold: hit {p:.5f}, sp1 "
            f"floor {p_sp:.5f}, repeat {p_rep:.5f}; suffix row PCC vs cold ({len(B) - lo} rows): hit median "
            f"{med(rp):.5f} min {float(rp.min()):.5f} | sp1 floor median {med(sp):.5f} min {float(sp.min()):.5f} | "
            f"repeat median {med(fl):.5f} min {float(fl.min()):.5f}"
        )
        return dict(B=B, pcc=p, pcc_rep=p_rep, pcc_sp1=p_sp, same=same, margin=mg, rows=rp, floor=fl, sp1=sp,
                    hit=(out_h[0], blk_hit), cold=(out_c[0], blk_cold))  # fmt: skip

    def kvr_case(A: Lane, D: int, hit_lane: int, cold_lane: int, tag: str, path, trace: bool = True) -> None:
        r = run_pair(A, D, hit_lane, cold_lane, tag)
        bar = med(r["sp1"]) - CPX_TOL
        ok = (r["same"] or r["margin"] < NEAR_TIE) and med(r["rows"]) >= bar
        log(
            f"CP-X {tag}: suffix median {med(r['rows']):.5f} vs the sp1 floor {med(r['sp1']):.5f} - {CPX_TOL} = "
            f"{bar:.5f}: {'met' if med(r['rows']) >= bar else 'NOT met'}; the design's last-token PCC >= 0.999 "
            f"{'met' if r['pcc'] >= 0.999 else 'NOT met'} ({r['pcc']:.5f}; sp1 floor {r['pcc_sp1']:.5f}, the "
            f"identical cold prefill repeated {r['pcc_rep']:.5f})"
        )
        lh = Lane(hit_lane, list(r["B"]) + [int(r["hit"][0].float().argmax())], r["hit"][1], [_margin(r["hit"][0])])
        lc = Lane(cold_lane, list(r["B"]) + [int(r["cold"][0].float().argmax())], r["cold"][1], [r["margin"]])
        s.greedy([lh, lc], DECODE_STEPS - 1, path=path, trace=trace)
        dec_ok, why = compare_streams(lc.seq[len(r["B"]) :], lc.margins, lh.seq[len(r["B"]) :], lh.margins)
        log(f"CP-X {tag}: {DECODE_STEPS} greedy tokens hit vs cold: {why}")
        if not (ok and dec_ok):
            fails.append(
                f"CP-X {tag}: suffix row PCC median {med(r['rows']):.5f} (sp1 floor {med(r['sp1']):.5f}, tol "
                f"{CPX_TOL}), argmax same {r['same']} margin {r['margin']:.3f}; decode {why}"
            )
        results[tag] = r

    results: Dict[str, Dict[str, Any]] = {}
    # ---- KV-R, the serving decode of this launch (the spec trace, all_split) ------------------------------------
    blk_a = s.blocks.take(api.cdiv(L + 200 + 1, BS))
    out, _, _ = s.prefill([prefill_request(0, prompt, L, blk_a)])
    A = Lane(0, list(prompt) + [int(out[0].float().argmax())], blk_a)
    s.greedy([A], 200)
    kvr_case(A, 200, 8, 16, "KV-R (spec all_split)", None)
    assert s.programs() == pc0, f"program cache grew after the capture: {pc0} -> {s.programs()}"
    # ---- with no trace alive: the plain all decode (eager) and the negative control (row mode, compiles programs) -
    s.gen.release_traces()
    try:
        # KV-R, the non-speculating production decode: the plain all path, staged now (no trace alive), freed after
        blk_ap = s.blocks.take(api.cdiv(L + 100 + 1, BS))
        out, _, _ = s.prefill([prefill_request(2, prompt, L, blk_ap)])
        Ap = Lane(2, list(prompt) + [int(out[0].float().argmax())], blk_ap)
        t0 = time.time()
        s.greedy([Ap], 100, trace=False, path=PLAIN_KVR)
        log(
            f"CP-X plain all: 100 eager decode steps in {time.time() - t0:.1f} s; A' tokens == A's (spec all_split): "
            f"{Ap.seq[: L + 100] == A.seq[: L + 100]}"
        )
        kvr_case(Ap, 100, 10, 18, "KV-R (plain all, eager)", PLAIN_KVR, trace=False)
        # negative control: A'' decoded in the draft-1 row mode
        blk_a2 = s.blocks.take(api.cdiv(L + 100 + 1, BS))
        out, _, _ = s.prefill([prefill_request(1, prompt, L, blk_a2)])
        A2 = Lane(1, list(prompt) + [int(out[0].float().argmax())], blk_a2)
        t0 = time.time()
        s.row_mode_greedy([A2], 100)
        log(
            f"CP-X negative control: 100 eager row-mode decode steps in {time.time() - t0:.1f} s; A'' tokens == A's: "
            f"{A2.seq[: L + 100] == A.seq[: L + 100]}"
        )
        n = run_pair(A2, 100, 9, 25, "no KV-R (negative control)")
    finally:
        if PLAIN_KVR in s.gen._paths:  # back to the serving path alone before the re-capture (one captured trace)
            s.gen._free_path(s.gen._paths[PLAIN_KVR])
        s.recapture()
    detected = med(n["rows"]) < med(n["sp1"]) - 0.01 or (not n["same"] and n["margin"] >= NEAR_TIE)
    with_kvr = ", ".join(f"{t} {med(r['rows']):.5f}" for t, r in results.items())
    log(
        f"CP-X negative control: stale cross-row KV {'DETECTED' if detected else 'NOT detected'} (suffix row PCC "
        f"median vs cold: {with_kvr}, without KV-R {med(n['rows']):.5f}; its sp1 floor {med(n['sp1']):.5f}, cold "
        f"repeat {med(n['floor']):.5f})"
    )
    if not detected:
        fails.append("CP-X negative control: a hit on blocks decode-written on another DP row without KV-R passed")
    assert not fails, "\n".join(fails)


# ---- MTP KV-only fill through sp1 chunks ---------------------------------------------------------------------------
MTP_FILL_PROMPT = "en_technical"  # 893 tokens: its first MTP_FILL_END are used
MTP_FILL_END = 768
MTP_FILL_SPLIT = 512


@pytest.mark.timeout(CP_TIMEOUT)
def test_mtp_fill_sp1_chunks(session):
    """The MTP layer's KV-only fill (design D9, §3.6.3) through sp1 chunks, at full depth (review 2026-10-03: no other
    device test reads an MTP cache an sp1 chunk filled; G-S4, G-S5 and the throughput runs prefill cold). The first
    768 tokens of a C2 prompt three ways: A cold (one sp0 chunk of 1024), B vLLM-chunked at 512 (sp0 512, then an sp1
    chunk at 512 of bucket 256), C a prefix hit at 512 on A's first 8 blocks (an sp1 chunk reading A's prefix). With
    every trace released, the cached latents of positions [512, 767) are read back (position 767's MTP entry takes the
    argmax stand-in: compared apart). Asserted:

    * layer 0's main cache rows [512, 768) are bitwise equal in A, B and C (its KV depends on the token embedding only:
      row-local whatever the path);
    * the MTP cache of B and of C vs A: the k_pe columns (roped at the row's position) row PCC min >= 0.99 and median
      >= 0.999, the n columns median >= 0.998. A wrong RoPE row could not pass: rotating A's own k_pe rows by one
      position on the host gives a median row PCC below the measured minimum (checked);
    * every cache read (L0, L1, L4, the last layer, MTP) is bitwise identical on all 32 chips (prefill writes all);
    * the stand-in rows of position 767 agree (PCC >= 0.99: the same argmax token, different hn), and the three
      last-token logits have the same argmax unless A's margin is below 0.5."""
    import ttnn

    from models.demos.motif3.tt.rope import cos_sin_table, inv_freq_for_kind, rotate_half

    s, gen, cfg = session, session.gen, session.cfg
    g = s.gold[MTP_FILL_PROMPT]
    E, SP = MTP_FILL_END, MTP_FILL_SPLIT
    assert g.S >= E, (g.S, E)
    ids = list(g.ids[:E])
    nb, k = api.cdiv(E, BS), SP // BS
    fails = []
    n_layers = gen.num_layers
    names = {f"L{i}": i for i in sorted({0, 1, min(4, n_layers - 1), n_layers - 1})}
    host: Dict[str, torch.Tensor] = {}
    bad_chips: Dict[str, List[int]] = {}

    def plan() -> List[Tuple[int, int, str]]:
        return [(c.start, c.bucket, c.path) for c in gen.last_prefill.jobs[0].plan.chunks]

    gen.release_traces()  # the read-back slices compile programs: no trace alive
    try:
        blk_a = s.blocks.take(nb)
        blk_b = s.blocks.take(nb)
        blk_c = blk_a[:k] + s.blocks.take(nb - k)
        la, _, _ = s.prefill([prefill_request(0, ids, E, blk_a)])
        plan_a = plan()
        s.prefill([prefill_request(9, ids, SP, blk_b)])
        lb, _, _ = s.prefill([prefill_request(17, ids, E, blk_b, start=SP)])
        plan_b = plan()
        lc, _, _ = s.prefill([prefill_request(25, ids, E, blk_c, start=SP)])
        plan_c = plan()
        log(f"MTP fill: plans A {plan_a}, B {plan_b} (after an sp0 {SP}), C {plan_c} (hit on A's first {k} blocks)")
        assert plan_a[0][2] == PP.SP0 and plan_b == plan_c and all(c[2] == PP.SP1 for c in plan_b), (plan_a, plan_b)
        used = blk_a + blk_b + blk_c
        b_lo, b_hi = min(used), max(used) + 1
        assert sorted(set(used)) == list(range(b_lo, b_hi)), "the test's blocks are not contiguous"
        caches = {name: s.pool[i] for name, i in names.items()}
        caches["MTP"] = s.pool.mtp
        for name, cache in caches.items():
            sl = ttnn.slice(cache, [b_lo, 0, 0, 0], [b_hi, 1, BS, api.KV_LATENT_DIM])
            try:
                shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(sl)]
            finally:
                ttnn.deallocate(sl)
            diff = [i for i, x in enumerate(shards) if not torch.equal(x, shards[0])]
            if diff:
                bad_chips[name] = diff
            host[name] = shards[0]
            del shards
    finally:
        s.recapture()

    def rows(t: torch.Tensor, blk: Sequence[int], lo: int, hi: int) -> torch.Tensor:
        return torch.stack([t[blk[p // BS] - b_lo, 0, p % BS] for p in range(lo, hi)])

    lo, hi = SP, E - 1  # position E - 1: the stand-in (compared apart)
    l0a = rows(host["L0"], blk_a, SP, E)
    for tag, blk in (("B", blk_b), ("C", blk_c)):
        same = torch.equal(rows(host["L0"], blk, SP, E), l0a)
        log(f"MTP fill: L0 rows [{SP}, {E}) A vs {tag} bitwise: {same}")
        if not same:
            fails.append(f"L0 rows [{SP}, {E}): A vs {tag} not bitwise equal (layer 0's KV is row-local)")
    for name in [n for n in names if n != "L0"]:  # deeper main layers: reported (sp1 numerics grow with depth)
        ra = rows(host[name], blk_a, lo, hi)
        for tag, blk in (("B", blk_b), ("C", blk_c)):
            rb = rows(host[name], blk, lo, hi)
            n_p, k_p = _row_pccs(rb[:, :512], ra[:, :512]), _row_pccs(rb[:, 512:], ra[:, 512:])
            log(
                f"MTP fill: {name} rows [{lo}, {hi}) A vs {tag}: n PCC median {float(n_p.median()):.6f} min "
                f"{float(n_p.min()):.6f}; k_pe PCC median {float(k_p.median()):.6f} min {float(k_p.min()):.6f}"
            )
    ma = rows(host["MTP"], blk_a, lo, hi)
    cos1, sin1 = cos_sin_table(inv_freq_for_kind(cfg, "plain"), torch.tensor([1]), dtype=torch.float32)
    kpe_a = ma[:, 512:]
    off1 = _row_pccs(kpe_a * cos1 + rotate_half(kpe_a) * sin1, kpe_a)  # each of A's rows roped one position too far
    k_min_all = []
    for tag, blk in (("B", blk_b), ("C", blk_c)):
        mb = rows(host["MTP"], blk, lo, hi)
        n_p, k_p = _row_pccs(mb[:, :512], ma[:, :512]), _row_pccs(mb[:, 512:], kpe_a)
        exact = int((mb == ma).all(dim=1).sum())
        k_min_all.append(float(k_p.min()))
        log(f"MTP fill: MTP rows [{lo}, {hi}) A vs {tag}: n PCC median {float(n_p.median()):.6f} min "
            f"{float(n_p.min()):.6f}; k_pe PCC median {float(k_p.median()):.6f} min {float(k_p.min()):.6f}; bitwise "
            f"rows {exact}/{hi - lo}")  # fmt: skip
        if not (float(k_p.min()) >= 0.99 and float(k_p.median()) >= 0.999 and float(n_p.median()) >= 0.998):
            fails.append(
                f"MTP cache A vs {tag}: k_pe PCC min {float(k_p.min()):.5f} (bar 0.99) median {float(k_p.median()):.5f}"
                f" (bar 0.999), n PCC median {float(n_p.median()):.5f} (bar 0.998)"
            )
    log(f"MTP fill: off-by-one RoPE reference (A's k_pe rows rotated one position): row PCC median "
        f"{float(off1.median()):.5f} max {float(off1.max()):.5f}")  # fmt: skip
    if not float(off1.median()) < min(k_min_all):
        fails.append(
            f"the k_pe metric cannot see a wrong RoPE row: off-by-one median {float(off1.median()):.5f} >= the "
            f"measured minimum {min(k_min_all):.5f}"
        )
    sa = rows(host["MTP"], blk_a, E - 1, E)[0]
    for tag, blk in (("B", blk_b), ("C", blk_c)):
        p_st = _pcc(rows(host["MTP"], blk, E - 1, E)[0], sa)
        log(f"MTP fill: MTP stand-in row {E - 1} A vs {tag} PCC {p_st:.5f}")
        if not p_st >= 0.99:
            fails.append(f"MTP stand-in row {E - 1}: A vs {tag} PCC {p_st:.5f} < 0.99")
    if bad_chips:
        fails.append(f"caches differ between chips (prefill writes every chip): {bad_chips}")
    log(f"MTP fill: every cache read identical on all 32 chips: {not bad_chips}")
    am = [int(x[0].float().argmax()) for x in (la, lb, lc)]
    mg = _margin(la[0])
    log(
        f"MTP fill: last-token argmax A/B/C {am} (A's margin {mg:.3f}); PCC A/B "
        f"{_pcc(la[0].float(), lb[0].float()):.5f} A/C {_pcc(la[0].float(), lc[0].float()):.5f}"
    )
    if len(set(am)) != 1 and mg >= NEAR_TIE:
        fails.append(f"last-token argmax differs A/B/C {am} at margin {mg:.3f}")
    assert not fails, "\n".join(fails)


# ---- CP9 -------------------------------------------------------------------------------------------------------------
@pytest.mark.timeout(CP_TIMEOUT)
def test_cp9_serving_mix_no_program_growth(session):
    """CP9 (short form): after the capture, 12 rounds of a randomized serving mix (cold rows, prefix hits, vLLM-chunked
    continuations with unaligned starts, same-step hits given hitter-first, preemption resumes = prompt + generated
    tokens, rows on every DP row) interleaved with traced decode steps: the program cache stays constant, every
    logits row is finite, and at the end a decode replay equals the eager step bitwise."""
    s = session
    rng = random.Random(9)
    src = [t for g in s.gold.values() for t in g.ids]

    def text(n):
        i = rng.randrange(0, len(src) - 1)
        return [src[(i + j) % len(src)] for j in range(n)]

    CAP = 64  # decode capacity of every request (positions past its prompt)
    KINDS = ("cold", "hit", "chunk", "same", "resume")
    next_kind = [0]
    pc0 = s.programs()
    live: Dict[int, Lane] = {}
    free_lanes = list(range(32))
    rng.shuffle(free_lanes)
    done: List[Lane] = []
    pending: List[Tuple[Lane, int, int]] = []  # (request, chunk end so far, round of that chunk): vLLM-chunked prompts
    kinds_seen = set()
    for rnd in range(12):
        rows, lanes = [], []
        n_rows = rng.randint(1, 3)
        while len(rows) < n_rows and free_lanes:
            kind = KINDS[next_kind[0] % len(KINDS)]  # every kind in turn (a missing precondition falls back to cold)
            next_kind[0] += 1
            lane = free_lanes.pop()
            ready = [x for x in pending if x[2] < rnd]  # a continuation only in a later call (one chunk per step)
            if kind == "chunk" and ready:
                pending.remove(ready[0])
                ln, st, _ = ready[0]
                e = min(len(ln.seq), st + rng.randint(100, 1500))
                rows.append(prefill_request(lane, ln.seq, e, ln.blocks, start=st))
                if e < len(ln.seq):
                    pending.append((ln, e, rnd))
                    free_lanes.insert(0, lane)
                    lanes.append(None)
                else:
                    lanes.append(Lane(lane, ln.seq[:], ln.blocks))
            elif kind == "hit" and [d for d in done if len(d.seq) - 1 >= BS]:
                base = rng.choice([d for d in done if len(d.seq) - 1 >= BS])
                k = rng.randint(1, max(1, (len(base.seq) - 1) // BS))
                seq = base.seq[: k * BS] + text(rng.randint(1, 600))
                blk = base.blocks[:k] + s.blocks.take(api.cdiv(len(seq) + CAP, BS) - k)
                rows.append(prefill_request(lane, seq, len(seq), blk, start=k * BS))
                lanes.append(Lane(lane, seq, blk))
            elif kind == "same" and len(rows) + 2 <= 4 and free_lanes:
                seq = text(rng.randint(300, 1500))
                blk = s.blocks.take(api.cdiv(len(seq) + CAP, BS))
                lane2 = free_lanes.pop()
                k = rng.randint(1, (len(seq) - 1) // BS)
                seq2 = seq[: k * BS] + text(rng.randint(1, 400))
                blk2 = blk[:k] + s.blocks.take(api.cdiv(len(seq2) + CAP, BS) - k)
                rows += [
                    prefill_request(lane2, seq2, len(seq2), blk2, start=k * BS),
                    prefill_request(lane, seq, len(seq), blk),
                ]  # hitter first: writer-first must reorder
                lanes += [Lane(lane2, seq2, blk2), Lane(lane, seq, blk)]
            elif kind == "resume" and done:
                base = rng.choice(done)  # preempted: prompt + generated tokens, re-prefilled on another lane
                k = rng.choice([0, max(0, (len(base.seq) - 1) // BS)])
                blk = base.blocks[:k] + s.blocks.take(api.cdiv(len(base.seq) + CAP, BS) - k)
                rows.append(prefill_request(lane, base.seq, len(base.seq), blk, start=k * BS))
                lanes.append(Lane(lane, base.seq[:], blk))
            else:
                kind = "cold"
                seq = text(rng.randint(1, 1500))
                blk = s.blocks.take(api.cdiv(len(seq) + CAP, BS))
                if rng.random() < 0.4 and len(seq) > 300:  # vLLM-chunked: the first chunk now, the rest later
                    e = rng.randint(100, len(seq) - 1)
                    rows.append(prefill_request(lane, seq, e, blk))
                    pending.append((Lane(lane, seq, blk), e, rnd))
                    free_lanes.insert(0, lane)
                    lanes.append(None)
                else:
                    rows.append(prefill_request(lane, seq, len(seq), blk))
                    lanes.append(Lane(lane, seq, blk))
            kinds_seen.add(kind)
        out, _, dt = s.prefill(rows)
        assert bool(torch.isfinite(out.float()).all()), "non-finite prefill logits"
        for ln, lg in zip(lanes, out):
            if ln is not None:
                ln.seq.append(int(lg.float().argmax()))
                live[ln.lane] = ln
        steps = rng.randint(1, 4)
        s.greedy(list(live.values()), steps)
        for lane in list(live):
            full = len(live[lane].seq) + 4 > len(live[lane].blocks) * BS
            if full or (rnd < 11 and rng.random() < 0.4):
                done.append(live.pop(lane))
                free_lanes.insert(0, lane)
        log(
            f"CP9 round {rnd}: {len(rows)} rows ({[r.start for r in rows]} starts, {dt:.1f} s), {steps} decode steps "
            f"for {len(live)} lanes; program cache {s.programs()}"
        )
    assert s.programs() == pc0, f"program cache grew after the capture: {pc0} -> {s.programs()}"
    # traced replay == eager step, bitwise (same inputs; the KV writes are idempotent)
    lanes = list(live.values())
    tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
    pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
    pt = torch.zeros(api.NUM_LANES, WIDTH, dtype=torch.int32)
    for ln in lanes:
        p = len(ln.seq) - 1
        tokens[ln.lane], pos[ln.lane], pt[ln.lane] = ln.seq[-1], p, page_row(ln.blocks, p + 1)
    batch = api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt)
    traced = s.gen.decode_forward(batch, kv_cache=s.pool, enable_trace=True).clone()
    eager = s.gen.decode_forward(batch, kv_cache=s.pool, enable_trace=False)
    act = [ln.lane for ln in lanes]
    assert act, "no live lane left for the replay check"
    assert torch.equal(traced[act], eager[act]), "decode trace replay != eager step"
    assert s.programs() == pc0
    log(f"CP9: kinds {sorted(kinds_seen)}; program cache constant at {pc0}; replay == eager on {len(act)} lanes")
    assert kinds_seen == set(KINDS), f"the mix missed row kinds: {sorted(set(KINDS) - kinds_seen)}"


# ---- TTFT (report only) ---------------------------------------------------------------------------------------------
@pytest.mark.timeout(CP_TIMEOUT)
def test_prefill_ttft_report(session):
    """Report-only (features design §8): ``prefill_forward_batch`` wall time from host tokens to host logits (MTP fill
    included, no test observer), cold rows of 128 ... 32000 tokens (2200 and 16,736 were the draft-1 bucket-cliff
    cases: 4K and 32K buckets) and prefix hits behind cached contexts (30K document + 100-token question, 8K history +
    300 new tokens, 2K shared system prompt + 200-token turn). Each case runs twice; the second (warm) time is reported.
    """
    s = session
    src = [t for g in s.gold.values() for t in g.ids]
    rng = random.Random(3)

    def text(n):
        i = rng.randrange(len(src))
        return [src[(i + j) % len(src)] for j in range(n)]

    pc0 = s.programs()
    rows = []
    scratch = s.blocks.take(api.cdiv(32000, BS))  # cold rows only write their own blocks: reuse one set
    for S in (128, 1024, 2200, 4096, 8192, 16736, 32000):
        ids = text(S)
        t = []
        for rep in range(2):
            blk = scratch
            t0 = time.time()
            s.gen.prefill_forward_batch([prefill_request(rep * 8, ids, S, blk)], kv_cache=s.pool)
            t.append(time.time() - t0)
        plan = s.gen.last_prefill.jobs[0].plan
        rows.append((f"cold {S}", t[-1], [(c.start, c.bucket, c.path) for c in plan.chunks]))
    own = s.blocks.take(8)
    for cached, new in ((30000, 100), (8192, 300), (2048, 200)):
        ids = text(cached + new)
        base = scratch  # the cached context (its own suffix blocks stay separate)
        s.gen.prefill_forward_batch([prefill_request(1, ids, cached, base)], kv_cache=s.pool)
        k = cached // BS
        t = []
        for rep in range(2):
            blk = base[:k] + own[: api.cdiv(cached + new, BS) - k]
            t0 = time.time()
            s.gen.prefill_forward_batch(
                [prefill_request(9 + rep * 8, ids, cached + new, blk, start=k * BS)], kv_cache=s.pool
            )
            t.append(time.time() - t0)
        plan = s.gen.last_prefill.jobs[0].plan
        rows.append((f"hit {cached} + {new}", t[-1], [(c.start, c.bucket, c.path) for c in plan.chunks]))
    for name, sec, chunks in rows:
        log(f"TTFT {name}: {sec:.2f} s  chunks {chunks}")
    assert s.programs() == pc0, f"program cache grew after the capture: {pc0} -> {s.programs()}"
