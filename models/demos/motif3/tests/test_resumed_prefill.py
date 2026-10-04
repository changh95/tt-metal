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
captured decode trace, the serving spec trace (``all_split``) of a ``packed`` launch; ``MULTI_TRACE_NOTE`` has the
history of the second trace this module does not capture, and why several traces are safe now)::

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
                                                             further from it than the single shot (0.001) in the
                                                             median and the mean error over the sampled positions
                                                             (lead decision 2026-10-03, F5; the worst position is
                                                             reported)
CP-H (ii)   last-token PCC >= 0.999                          the hit row's argmax is the fp32 golden's unless the
                                                             golden's margin < 0.5 (lead decision 2026-10-03, F5; the
                                                             pre-F5 rule against the cold row is reported); greedy
                                                             streams; plus the pooled bars of CP-H (iii) and CP-C
                                                             (asserted as designed)
CP-X        last-token PCC >= 0.999 (CP-H tolerances)        argmax rule; suffix median >= the sp1 floor - 0.003;
                                                             greedy streams; the negative control detected
CP-P        last-token PCC >= 0.999, cache rows >= 0.99999   within 2 x the bucket floor (per row at the packed
            (packed vs per-row)                              pass's T) in the median / pooled and per segment, the
                                                             head rows and L0-L2 rows exact; count bars + 1 (logged);
                                                             signed off by the lead 2026-10-04 (P5_T64_REVIEW I-1):
                                                             see ``test_cp_p_packed_vs_per_row``
=========== ================================================ =========================================================

* **CP-P** (P5 packed prefill, design §6.2): each case runs the same call per row (packing off) and packed (twice),
  on separate blocks, in the shared session (it warms the packed shapes before the capture, ``MOTIF3_CP_PACKED``):
  (i) 32 short prompts (C2 prefixes + text, 20-64 tokens: pk0 S 64, T 2048) and 32 of 65-128 tokens (S 128,
  T 4096); (ii) 32 rows sharing a 2K prefix in one call (solo sp0 2048 + pk1 S 128 B 32 a 2048, shared tails);
  (iii) the mixed step; (iv) 64-token template hits; (v) a same-step hit behind a writer with an internal split;
  (vi) the odd-block same-step hit of review edit R-E1; (vii) 8 sessions resumed at one start (pk1, distinct tails,
  R-E2). Checks: the expected pass kinds; per row the same argmax unless the per-row margin < 0.5; teacher-forced
  top-1 / NLL; 32 greedy tokens and the MTP draft acceptance; the repeat bitwise; cache rows (L0, L1, L2, the last
  layer, MTP) vs per-row and blocks outside the call's fill set untouched; per segment (P5 review finding 2): the
  head rows exact, each row's last-token logits and written cache rows (L0-L2 near-bitwise) against the floor; the
  call wall time. ``test_cp_p_burst_ttft_report``: burst TTFT and the decode stall of a packed burst between traced
  decode steps.
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
    """What ``model.chunk_inputs`` returns: the host tables (a chunk's ``ChunkHostTables`` or a packed pass's
    ``PackedHostTables``), plus a free counter."""

    def __init__(self, host):
        self.host = host
        self.path, self.start, self.bucket, self.end = host.path, int(host.start), int(host.bucket), int(host.end)
        self.is_packed = bool(host.is_packed)
        self.reads_cache = bool(host.reads_cache)
        self.segments = int(getattr(host, "segments", 1))
        self.seg_rows = getattr(host, "seg_rows", None)
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
        assert 0 <= row < X.bucket and X.h[row] != PAD, f"the head reads padding row {row}"
        return FakeT(h=X.h[row], row=int(row))

    def prefill_logits_to_host(self, tile, row):
        assert row == tile.row
        out = torch.zeros(self.V, dtype=torch.bfloat16)
        out[tile.h % self.V] = 1.0
        return out

    def stream_mean_norm(self, X):
        return FakeT(**{k: v for k, v in X.__dict__.items() if k != "alive"})


class FakeMTP:
    """KV-only MTP fill: writes ``(p, t_{p+1})`` through the chunk's fill table and checks the next tokens: known
    tokens, except the row's last known position, which must take the argmax of the returned logits (``h % V``). A
    packed pass: the same per real segment (its rows at ``k S``, its slice of the fill table); padding rows and dummy
    segments carry the pad token and write nothing."""

    def __init__(self, model):
        self.m = model

    def fill_kv_prefill(self, hn, nxt, *, kv_cache, chunk):
        if chunk.is_packed:
            return self._fill_packed(hn, nxt, kv_cache, chunk)
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

    def _fill_packed(self, hn, nxt, kv_cache, chunk):
        m, host = self.m, chunk.host
        p_, reqs = m.current_pass
        B, S, T, bs = int(host.segments), int(host.seg_rows), int(host.bucket), int(host.block_size)
        assert len(nxt.ids) == T and hn.bucket == T, (len(nxt.ids), hn.bucket, T)
        pad = int(m.cfg.pad_token_id)
        for k in range(B):
            sg = host.segment(k)
            rows = nxt.ids[k * S : (k + 1) * S]
            if k >= len(p_.segments):
                assert all(t == pad for t in rows), f"dummy segment {k}: MTP next tokens {set(rows)}, not padding"
                assert int(sg.fill.max()) < 0, f"dummy segment {k} writes the MTP cache"
                continue
            g = p_.segments[k]
            req = reqs[g.row]
            a, e = int(sg.start), int(sg.end)
            assert (a, e) == (g.start, g.end)
            for r in range(e - a):
                p = a + r
                want = int(req.tokens[p + 1]) if p + 1 < req.end else hn.h[k * S + r] % m.V  # known, else the argmax
                assert rows[r] == want, f"MTP next token of row {g.row} position {p}: {rows[r]}, expected {want}"
            assert all(t == pad for t in rows[e - a :]), f"segment {k}: MTP padding rows {set(rows[e - a:])}"
            for j, blk in enumerate(sg.fill[0].tolist()):
                if blk < 0:
                    continue
                for q in range(bs):
                    p = a + j * bs + q
                    kv_cache[(blk, q)] = (p, rows[j * bs + q]) if p < e else (p, PAD)
        m.ops.append(("mtp", T))


class FakeModel:
    """The part of ``MotifModel`` the generator's prefill uses, on an :class:`EmuPool`. ``prefill_chunk`` emulates the
    attention's reads: an sp1 chunk reads positions ``[0, a)`` through its SDPA table (global layers; every entry must
    hold its position's prefix hash and chain correctly) and ``[a - 128, a)`` through its tail blocks (SWA layers; must
    equal the SDPA reads), and its own rows in blocks the fill skips (shared blocks below ``w0`` when ``c0 < w0``: the
    global layers fill first and read rows ``[c0, w0)`` from the cache, review edit R-E1) must already hold exactly
    the KV the chunk computes there; the fill writes the chunk's rows through the fill table, never into a complete
    block. A packed pass (P5): every segment (``PackedHostTables.segment(k)``) the same, with ALL reads of the pass
    before any of its writes (no segment may depend on a write of its own pass) and dummy segments writing nothing."""

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
        self.current = None  # the PrefillRequest being run (set by the harness's _run_chunk hook)
        self.current_pass = None  # (PrefillPass, requests) of the packed pass being run (the _run_packed hook)
        self.warm_calls: List[Tuple] = []  # warm_attention calls: (shape, layers, mtp)

    def allocate_kv_caches(self, num_blocks, block_size, dtype=None, *, mtp=None):
        return EmuPool(num_blocks, block_size, self.mtp is not None if mtp is None else mtp)

    def chunk_inputs(self, host):
        ch = FakeChunk(host)
        self.chunks.append(ch)
        self.ops.append(("inputs", host.path, int(host.bucket)))
        return ch

    def warm_attention(self, chunk, kv_caches, *, layers=None, mtp=False):
        host = chunk.host
        assert host.is_packed and int(host.fill.max()) < 0, "a packed warm-up input writes nothing"
        assert not host.reads_cache or int(host.sdpa.max()) == 0, "a pk1 warm-up reads only the null block"
        self.warm_calls.append((host.shape, layers, bool(mtp)))
        self.ops.append(("warm_attention", host.shape, bool(mtp)))
        return (0, 1)

    @staticmethod
    def _read_prefix(host, kv) -> int:
        """The sp1 reads of one chunk / segment before its rows: ``[0, a)`` through the SDPA table, the SWA tail
        through the tail blocks. Returns the prefix hash at ``a - 1``."""
        a, bs = int(host.start), int(host.block_size)
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
        return prev[2]

    @staticmethod
    def _check_unfilled_rows(host, kv, tokens, hs) -> None:
        """R-E1: the global layers read the chunk's rows whose block the fill skips (shared, below ``w0``) from the
        cache: they must already hold this row's KV (written by an earlier pass / call)."""
        a, e, bs = int(host.start), int(host.end), int(host.block_size)
        fill, sd = host.fill[0].tolist(), host.sdpa[0].tolist()
        for r in range(e - a):
            if fill[r // bs] >= 0:
                continue
            q = a + r
            got = kv.get((sd[q // bs], q % bs), EMPTY)
            assert got == (q, tokens[r], hs[r]), f"global read of unfilled row {q} (c0 < w0): {got}, not this row's KV"

    @staticmethod
    def _write(host, pool, tokens, hs) -> None:
        a, e, bs, kv = int(host.start), int(host.end), int(host.block_size), pool.kv
        for j, blk in enumerate(host.fill[0].tolist()):  # sp0 / SWA fill after, global sp1 before its SDPA: same here
            if blk < 0:
                continue
            assert blk not in pool.complete, f"block {blk} is complete (maybe shared) and is written again"
            for q in range(bs):
                p = a + j * bs + q
                kv[(blk, q)] = (p, tokens[j * bs + q], hs[j * bs + q]) if p < e else (p, PAD, PAD)
        for j, blk in enumerate(host.fill[0].tolist()):
            if blk >= 0 and all(kv.get((blk, q), (0, PAD))[1] != PAD for q in range(bs)):
                pool.complete.add(blk)

    def prefill_chunk(self, tok, *, chunk, kv_caches):
        host, pool = chunk.host, kv_caches
        if host.is_packed:
            return self._prefill_packed(tok, host, pool)
        a, C, end, bs = int(host.start), int(host.bucket), int(host.end), int(host.block_size)
        assert tok.bucket == C and tok.real == end - a, (tok.bucket, tok.real, C, end, a)
        self.ops.append(("layers", host.path, C, a))
        kv = pool.kv
        if host.path == PP.SP1:
            h = self._read_prefix(host, kv)
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
        if host.path == PP.SP1 and int(host.fill.max()) >= 0:  # (a warm-up chunk writes and checks nothing)
            self._check_unfilled_rows(host, kv, tok.tokens, hs)
        self._write(host, pool, tok.tokens, hs)
        req = self.current
        X = FakeT(h=hs, start=a, end=end, tokens=tok.tokens[: end - a], bucket=C)
        if req is not None:  # None: a warm-up chunk
            self.known_tokens[id(X.tokens)] = req.end
            if end < req.end:
                self.next_known[id(X.tokens)] = int(req.tokens[end])
        return X

    def _prefill_packed(self, tok, host, pool):
        B, S, T = int(host.segments), int(host.seg_rows), int(host.bucket)
        assert tok.bucket == T and tok.real == T, (tok.bucket, tok.real, T)
        self.ops.append(("layers", host.path, T, int(host.start), S, B))
        p_, reqs = self.current_pass
        assert (p_.tokens, p_.seg_rows, p_.batch, p_.kind) == (T, S, B, host.path), (p_.describe(), host.shape)
        pad = int(self.cfg.pad_token_id)
        kv = pool.kv
        segs = [host.segment(k) for k in range(B)]
        heads = [self._read_prefix(sg, kv) if sg.is_sp1 else H0 for sg in segs]  # every read of the pass first
        hs: List[Any] = [PAD] * T
        for k, sg in enumerate(segs):
            rows = tok.tokens[k * S : (k + 1) * S]
            if k >= len(p_.segments):  # a dummy: pad tokens, writes nothing, outputs dropped
                assert all(t == pad for t in rows) and int(sg.fill.max()) < 0, f"dummy segment {k}"
                continue
            g = p_.segments[k]
            a, e = int(sg.start), int(sg.end)
            assert (a, e) == (g.start, g.end), (k, a, e, g)
            assert rows[: e - a] == [int(t) for t in reqs[g.row].tokens[a:e]], f"segment {k}: not row {g.row}'s tokens"
            assert all(t == pad for t in rows[e - a :]), f"segment {k}: padding rows are not the pad token"
            h = heads[k]
            for r in range(e - a):
                h = chain(h, rows[r])
                hs[k * S + r] = h
            if sg.is_sp1:
                self._check_unfilled_rows(sg, kv, rows, hs[k * S : (k + 1) * S])
        for k, sg in enumerate(segs[: len(p_.segments)]):  # then every write
            self._write(sg, pool, tok.tokens[k * S : (k + 1) * S], hs[k * S : (k + 1) * S])
        return FakeT(h=hs, start=int(host.start), end=int(host.end), tokens=list(tok.tokens), bucket=T, packed=True)

    def deallocate(self):
        pass


def fake_generator(
    cfg: MotifTTConfig, monkeypatch, *, mtp: bool = True, num_blocks: int = 20000, packed: bool = False, log=None
):
    """A ``MotifGenerator`` on a :class:`FakeModel` (no device): ``ttnn.deallocate`` recorded, the pool emulated.
    ``packed``: packed prefill on (``gen.packed_prefill``)."""
    from models.demos.motif3.tt import generator as G

    freed = []
    monkeypatch.setattr(G.ttnn, "deallocate", lambda t, *a, **k: freed.append(t), raising=False)
    model = FakeModel(cfg, mtp=mtp)
    gen = G.MotifGenerator(None, cfg, model, log=log)
    gen.packed_prefill = bool(packed)
    # the fake model needs the request(s) of the chunk / pass being run (for the token and next-token checks)
    orig_chunk, orig_packed = gen._run_chunk, gen._run_packed

    def run_chunk(job, ch, host, pool):
        model.current = job.request
        return orig_chunk(job, ch, host, pool)

    def run_packed(batch, p, host, pool):
        model.current_pass = (p, batch.requests)
        return orig_packed(batch, p, host, pool)

    gen._run_chunk, gen._run_packed = run_chunk, run_packed
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
    chunk = SimpleNamespace(is_sp1=True, reads_cache=True, path=PP.SP1, bucket=256)
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
    # review edit R-E11: the kv-cache guard keys on reads_cache (a pk1 pass is not is_sp1); pk0 reads no cache
    pk1 = SimpleNamespace(is_sp1=False, reads_cache=True, is_packed=True, path="pk1", bucket=256)
    with pytest.raises(ValueError, match="a pk1 pass reads the cached prefix: prefill_chunk needs kv_caches"):
        model.prefill_chunk(tok, chunk=pk1)
    got.clear()
    pk0 = SimpleNamespace(is_sp1=False, reads_cache=False, is_packed=True, path="pk0", bucket=256)
    assert model.prefill_chunk(tok, chunk=pk0) == "X2" and [g[2] for g in got] == [{"chunk": pk0, "kv_cache": None}] * 3
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


def test_cpu_model_warm_attention(monkeypatch):
    """``MotifModel.warm_attention`` (P5 warm-up, design §3.5): one zero input ``[1, 1, T, hidden]`` (allocated and
    freed), the attention of the first built global and the first built SWA layer with the warm-up chunk and each
    layer's own cache, the MTP layer's packed fill (pad next tokens, the pool's MTP cache) only with ``mtp=True``;
    explicit ``layers``; refusals: a layer that is not built, a pk1 input without caches, ``mtp=True`` without the MTP
    cache. Nothing else runs."""
    from models.demos.motif3.tt import model as M

    events = []
    fake = SimpleNamespace(
        bfloat16="bf16", TILE_LAYOUT="tile", DRAM_MEMORY_CONFIG="dram",
        empty=lambda shape, dt, lay, dev, mc: events.append(("empty", tuple(shape), dt, lay, mc)) or "E",
        fill=lambda e, v: events.append(("fill", e, v)) or "X0",
        deallocate=lambda t: events.append(("free", t)),
    )  # fmt: skip
    monkeypatch.setattr(M, "ttnn", fake)
    model = object.__new__(M.MotifModel)
    model.cfg = host_cfg()
    model.mesh_device = "mesh"
    model.layer_ids = (0, 1, 2, 3, 4, 5)  # L0 / L4 global, the rest SWA (l % 4 == 0)

    class Attn:
        def __init__(self, i):
            self.i = i

        def forward_prefill(self, x, *, chunk, kv_cache):
            events.append(("attn", self.i, x, chunk.bucket, kv_cache))
            return f"o{self.i}"

    model.layers = [SimpleNamespace(attn=Attn(i)) for i in model.layer_ids]
    model.embed = SimpleNamespace(rows_tokens_device=lambda t, rows: events.append(("nxt", t.tolist(), rows)) or "N")
    model.mtp = SimpleNamespace(
        fill_kv_prefill=lambda hn, nxt, *, kv_cache, chunk: events.append(("mtp", hn, nxt, kv_cache, chunk.bucket))
    )
    caches = M.MotifKVPool([f"kv{i}" for i in range(6)], model.layer_ids, 10, 64, "bfp8", mtp="kvm")
    pk0 = SimpleNamespace(bucket=256, reads_cache=False, path="pk0")
    pk1 = SimpleNamespace(bucket=512, reads_cache=True, path="pk1")
    assert model.attention_warm_layers() == (0, 1)
    assert model.warm_attention(pk0, caches) == (0, 1)
    assert events == [
        ("empty", (1, 1, 256, model.cfg.hidden_size), "bf16", "tile", "dram"), ("fill", "E", 0.0),
        ("attn", 0, "X0", 256, "kv0"), ("free", "o0"), ("attn", 1, "X0", 256, "kv1"), ("free", "o1"),
        ("free", "E"), ("free", "X0"),
    ]  # fmt: skip
    events.clear()
    assert model.warm_attention(pk1, caches, layers=(4, 5), mtp=True) == (4, 5)
    pad = int(model.cfg.pad_token_id)
    assert [e[0] for e in events] == ["empty", "fill", "attn", "free", "attn", "free", "nxt", "mtp", "free", "free",
                                      "free"]  # fmt: skip
    assert events[2][1] == 4 and events[4][1] == 5 and events[2][4] == "kv4" and events[4][4] == "kv5"
    assert events[6] == ("nxt", [pad] * 512, 512) and events[7] == ("mtp", "X0", "N", "kvm", 512)
    events.clear()
    with pytest.raises(ValueError, match="not built"):
        model.warm_attention(pk0, caches, layers=(7,))
    with pytest.raises(ValueError, match="pk1 input reads the paged cache"):
        model.warm_attention(pk1, None)
    with pytest.raises(ValueError, match="MTP cache"):
        model.warm_attention(pk0, [f"kv{i}" for i in range(6)], mtp=True)
    assert events == [], "a refused warm-up runs nothing"
    model.mtp = None
    model.warm_attention(pk0, caches, mtp=True)  # no MTP layer: mtp is ignored
    assert "mtp" not in [e[0] for e in events]


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
# host tests: packed prefill (P5, docs/p5_t64/P5_T64_DESIGN.md §3) at generator level, on the emulated cache
# ======================================================================================================================
P5_NUM_BLOCKS = 1 << 17  # test_prefill_plan._Ids hands out block ids below 2**17


def _p5_req(lane: int, tokens: Sequence[int], start: int, blocks: Sequence[int]) -> api.PrefillRequest:
    return _rows_of(list(tokens), start, len(tokens), list(blocks), width=512, lane=lane)


def _p5_scenarios(seed: int = 0) -> Dict[str, List[api.PrefillRequest]]:
    """The design §3.3 scenarios (P5N App. A) and CP-P (vi) with consistent tokens (a hit row's cached prefix is the
    writer's tokens): 32 x 34 cold; 32 sharing a 2K prefix in one step; 32 behind a 64-token template (hit < 128,
    c0 = 0); the mixed step; the R-E1 odd-block same-step hit (``test_prefill_plan._odd_hit_call``)."""
    from models.demos.motif3.tests.unit.test_prefill_plan import _Ids, _odd_hit_call

    rng = random.Random(seed)
    ids = _Ids(seed)

    def tok(n):
        return [rng.randrange(5000) for _ in range(n)]

    out = {"burst": [_p5_req(i, tok(34), 0, ids.take(1)) for i in range(32)]}
    P, shared = tok(2048), ids.take(32)
    out["shared_2k"] = [_p5_req(0, P + tok(60), 0, shared + ids.take(1))] + [
        _p5_req(i, P + tok(62), 2048, shared + ids.take(1)) for i in range(1, 32)
    ]
    Tm, tmpl = tok(64), ids.take(1)
    out["template"] = [_p5_req(0, Tm + tok(26), 0, tmpl + ids.take(1))] + [
        _p5_req(i, Tm + tok(31), 64, tmpl + ids.take(1)) for i in range(1, 32)
    ]
    lens = [40] * 24 + [300, 350, 400, 420, 480, 500] + [1500, 1500]
    out["mixed"] = [_p5_req(i, tok(n), 0, ids.take(api.cdiv(n, 64))) for i, n in enumerate(lens)]
    out["odd_hit"] = _odd_hit_call(ids)[0]
    return out


P5_EXPECTED_PASSES = {  # the planner's passes at the production geometry (test_prefill_plan.test_p5_design_scenarios)
    "burst": ["pk0 T=2048 S=64 B=32 (32 real) a=0"],
    "shared_2k": ["solo sp0 C=2048 a=0 (row 0 chunk 0)", "pk1 T=4096 S=128 B=32 (32 real) a=2048 tails=shared"],
    "template": ["pk0 T=4096 S=128 B=32 (32 real) a=0"],
    "mixed": [
        "pk0 T=2048 S=64 B=32 (24 real) a=0",
        "pk0 T=1024 S=512 B=2 (2 real) a=0",
        "pk0 T=2048 S=512 B=4 (4 real) a=0",
        "solo sp0 C=2048 a=0 (row 30 chunk 0)",
        "solo sp0 C=2048 a=0 (row 31 chunk 0)",
    ],
    "odd_hit": [
        "solo sp0 C=2048 a=0 (row 0 chunk 0)",
        "solo sp0 C=2048 a=0 (row 1 chunk 0)",
        "solo sp1 C=128 a=2048 (row 0 chunk 1)",
        "solo sp1 C=256 a=2048 (row 1 chunk 1)",
        "pk1 T=256 S=128 B=2 (2 real) a=2048 tails=shared",
    ],
}


def _p5_check_call(gen, pool, reqs: Sequence[api.PrefillRequest], logits: torch.Tensor) -> None:
    """Every row's logits are those of its full token prefix; the MTP cache holds ``(p, t_{p+1})`` at every position
    the call wrote (``[w0, end)``; the row's argmax stand-in at its last position)."""
    V, bs = int(gen.cfg.vocab_size), int(gen.cfg.kv_block_size)
    assert tuple(logits.shape) == (len(reqs), V)
    for i, r in enumerate(reqs):
        h = prefix_hashes(r.tokens.tolist())
        assert int(logits[i].float().argmax()) == h[r.end - 1] % V, f"row {i}: logits of another prefix"
        if pool.mtp_kv is None:
            continue
        for p in range(gen.last_prefill.jobs[i].plan.w0, r.end):
            got = pool.mtp_kv.get((int(r.page_table[p // bs]), p % bs))
            nxt = int(r.tokens[p + 1]) if p + 1 < r.end else h[r.end - 1] % V
            assert got == (p, nxt), f"row {i}: MTP entry of position {p}: {got}, expected {(p, nxt)}"


def _p5_pair(cfg, monkeypatch, **kw):
    """A packing-on and a packing-off fake generator on separate emulated pools."""
    return (fake_generator(cfg, monkeypatch, packed=True, num_blocks=P5_NUM_BLOCKS, **kw),
            fake_generator(cfg, monkeypatch, packed=False, num_blocks=P5_NUM_BLOCKS, **kw))  # fmt: skip


def test_cpu_packed_design_scenarios(monkeypatch):
    """P5 at generator level on the design §3.3 scenarios and CP-P (vi) (production geometry: A = 128, bs 64, span cap
    8192, the config's segment sizes): the passes are the planner's (``P5_EXPECTED_PASSES``); every row's logits and
    every cache entry (main and MTP) equal the packing-off call's on the same rows (the emulated cache is exact, so
    "packed == per-row" is equality here); every pass uploads its inputs once and frees them once, the LM head runs
    once per row and the MTP fill once per pass; the observers see each solo chunk / packed pass once with its rows;
    the counters; a packed call runs no more device passes than chunks."""
    cfg = host_cfg()
    for name, reqs in _p5_scenarios().items():
        (gp, mp, pool_p, _), (gs, ms, pool_s, _) = _p5_pair(cfg, monkeypatch)
        seen = []
        gp.pass_observer = lambda b, p, X: seen.append(("pass", p.shape, X.bucket))
        gp.chunk_observer = lambda job, ch, X: seen.append(("chunk", (ch.path, ch.bucket), X.bucket))
        lp = gp.prefill_forward_batch(reqs, kv_cache=pool_p)
        ls = gs.prefill_forward_batch(reqs, kv_cache=pool_s)
        batch = gp.last_prefill
        assert batch.packed and not gs.last_prefill.packed
        assert [p.describe() for p in batch.passes] == P5_EXPECTED_PASSES[name], (name, batch.describe())
        assert torch.equal(lp, ls), f"{name}: packed logits != per-row logits"
        assert pool_p.kv == pool_s.kv and pool_p.mtp_kv == pool_s.mtp_kv, f"{name}: packed cache != per-row cache"
        _p5_check_call(gp, pool_p, reqs, lp)
        n = len(batch.passes)
        assert len(mp.chunks) == n and all(c.freed == 1 for c in mp.chunks), "one upload and one free per pass"
        assert sum(o[0] == "head" for o in mp.ops) == len(reqs) and sum(o[0] == "mtp" for o in mp.ops) == n
        assert seen == [("pass" if p.is_packed else "chunk", p.shape, p.tokens) for p in batch.passes]
        assert batch.shapes == {p.shape for p in batch.passes} and n <= batch.chunks
        st, pk = gp.stats, batch.packed_passes
        assert (st["packed_calls"], st["packed_passes"], st["solo_passes"]) == (1, len(pk), n - len(pk))
        assert st["packed_pk1_passes"] == sum(p.kind == PP.PK1 for p in pk)
        assert st["packed_segments"] == sum(len(p.segments) for p in pk)
        assert st["packed_dummy_segments"] == sum(p.dummies for p in pk)
        assert st["packed_padding_rows"] == sum(p.padding_rows for p in pk) and st["packed_solo_fallbacks"] == 0
        assert st["packed_plan_errors"] == 0 and batch.plan_error is None
        assert st["prefill_chunks"] == gs.stats["prefill_chunks"] == batch.chunks == gs.stats["solo_passes"]
        assert st["mtp_fills"] == n and gs.stats["mtp_fills"] == batch.chunks and gs.stats["packed_passes"] == 0
        assert {"last_prefill_s", "last_prefill_plan_s"} <= set(gp.timings)
        log(f"P5 host scenario {name}: {batch.describe()}")


@pytest.mark.parametrize(
    "bs, cap, budget, threshold, seed",
    [
        (64, 8192, 8064, 8064, 0),  # production (A = 128, budget = threshold = 8064)
        (64, 8192, 8064, 8064, 1),
        (64, 1024, 3000, 700, 2),  # small span cap: internal chunks, packed tails of split rows, odd-block hits
        (32, 1024, 4000, 0, 3),  # block 32 (A = 128 > bs)
    ],
)
def test_cpu_packed_emulated_schedule(monkeypatch, bs, cap, budget, threshold, seed):
    """P5 under a vLLM-like schedule of bursts (``test_prefill_plan.FakeVllmScheduler``: prefix hits on admission, full
    blocks cached at allocation so later-admitted rows hit blocks an earlier row computes in the same step, chunk
    budgets, shuffled rows, new lanes per call): short prompts, shared system prompts (incl. 200 / 1100 / 2112 tokens:
    odd-block same-step hits whose writer's chunk boundary sits at the readers' c0, review edit R-E1), multi-turn
    extensions, duplicates and long prompts. Every call on a packing-on and a packing-off generator: the same logits,
    the same cache contents after every call, the emulation's read checks (prefix, SWA tail, the unfilled rows
    ``[c0, w0)`` the global layers read, no read of a write of the same pass) and the MTP entries."""
    from models.demos.motif3.tests.unit.test_prefill_plan import FakeVllmScheduler

    cfg = host_cfg(prefill_span_cap=cap)
    cfg.set_kv_geometry(24000 if bs == 32 else 12000, bs)
    (gp, mp, pool_p, _), (gs, ms, pool_s, _) = (
        fake_generator(cfg, monkeypatch, packed=True, num_blocks=cfg.kv_num_blocks),
        fake_generator(cfg, monkeypatch, packed=False, num_blocks=cfg.kv_num_blocks),
    )
    rng = random.Random(seed)
    width = 32768 // bs
    sched = FakeVllmScheduler(bs=bs, num_blocks=cfg.kv_num_blocks, budget=budget, threshold=threshold, width=width)
    systems = [[rng.randrange(5000) for _ in range(n)] for n in (64, 130, 200, 1100, 2112)]
    history: List[List[int]] = []
    st = dict(calls=0, rows=0, odd_hits=0, packed_odd_hits=0)
    for wave in range(16):
        for _ in range(rng.randint(4, 20)):
            k = rng.random()
            if k < 0.35:
                toks = [rng.randrange(5000) for _ in range(rng.randint(1, 300))]
            elif k < 0.65:
                toks = rng.choice(systems) + [rng.randrange(5000) for _ in range(rng.randint(1, 200))]
            elif k < 0.75 and history:
                toks = list(rng.choice(history)) + [rng.randrange(5000) for _ in range(rng.randint(1, 300))]
            elif k < 0.85 and history:
                toks = list(rng.choice(history))
            else:
                toks = [rng.randrange(5000) for _ in range(rng.randint(300, 2600))]
            history.append(toks)
            sched.add(toks)
        while sched.running or sched.waiting:
            rows = sched.schedule()
            perm = list(range(len(rows)))
            rng.shuffle(perm)
            lanes = rng.sample(range(32), len(rows))
            reqs = [_rows_of(rows[k][0].tokens, rows[k][0].computed, rows[k][1], rows[k][0].blocks, width=width,
                             lane=lane) for k, lane in zip(perm, lanes)]  # fmt: skip
            lp = gp.prefill_forward_batch(reqs, kv_cache=pool_p)
            ls = gs.prefill_forward_batch(reqs, kv_cache=pool_s)
            assert torch.equal(lp, ls), f"call {st['calls']}: packed logits != per-row"
            assert pool_p.kv == pool_s.kv and pool_p.mtp_kv == pool_s.mtp_kv, f"call {st['calls']}: caches differ"
            _p5_check_call(gp, pool_p, reqs, lp)
            batch = gp.last_prefill
            odd = {i for i, j in enumerate(batch.jobs) if j.plan.has_sp1 and j.plan.c0 < j.plan.w0}
            st["odd_hits"] += len(odd)
            st["packed_odd_hits"] += sum(g.row in odd for p in batch.packed_passes for g in p.segments)
            st["calls"] += 1
            st["rows"] += len(reqs)
            sched.commit(rows)
            assert st["calls"] < 3000
    s = gp.stats
    assert all(c.freed == 1 for c in mp.chunks) and len(mp.chunks) == s["solo_passes"] + s["packed_passes"]
    assert s["prefill_chunks"] == gs.stats["prefill_chunks"] and s["packed_passes"] > 0 and s["packed_pk1_passes"] > 0
    assert st["packed_odd_hits"] > 0, st  # R-E1: odd-block hit rows ran in packed passes (after their writers)
    assert s["solo_passes"] + s["packed_passes"] < s["prefill_chunks"], "packing saved no device pass"
    assert s["packed_plan_errors"] == 0, "a packed plan failed its own checks (the call ran per row)"
    log(f"P5 emulated schedule bs={bs} cap={cap} budget={budget} threshold={threshold} seed={seed}: {st}; "
        f"{ {k: v for k, v in s.items() if k.startswith(('packed', 'solo', 'prefill'))} }")  # fmt: skip


def test_cpu_packed_shape_filter_after_capture(monkeypatch):
    """After the decode capture (a fake captured path): a packed pass whose shape was not warmed runs as one solo pass
    per segment (``fallback`` = the packed shape, counter ``packed_solo_fallbacks``, logged once per shape), with the
    same results; warmed shapes pack; the pk1 key names the tail variant (review edit R-E2: ``shared`` warmed does not
    let a ``distinct`` pass run); an unwarmed solo shape still refuses the call (nothing runs, nothing is written);
    packing turned on after a capture without packed shapes never refuses a call."""
    from models.demos.motif3.tt.generator import DecodePath

    cfg = host_cfg()
    logs: List[str] = []
    gp, mp, pool, _ = fake_generator(cfg, monkeypatch, packed=True, num_blocks=P5_NUM_BLOCKS, log=logs.append)
    gp._paths[gp.serving_path] = DecodePath(*gp.serving_path, width=512, trace_id=object())  # "captured"
    gp._warmed = set(gp.prefill_shapes())  # solo shapes only: packing was off at the warm-up
    rng = random.Random(4)

    def burst(n, L, lane0=0):
        return [_p5_req(lane0 + i, [rng.randrange(5000) for _ in range(L)], 0, [1000 + 40 * lane0 + i])
                for i in range(n)]  # fmt: skip

    reqs = burst(32, 34)
    lg = gp.prefill_forward_batch(reqs, kv_cache=pool)
    _p5_check_call(gp, pool, reqs, lg)
    b = gp.last_prefill
    assert not b.packed_passes and len(b.fallbacks) == 32 and {p.fallback for p in b.fallbacks} == {("pk0", 2048, 64)}
    assert gp.stats["packed_solo_fallbacks"] == 32 and sum("('pk0', 2048, 64)" in m for m in logs) == 1
    gp.prefill_forward_batch(burst(32, 34), kv_cache=pool)
    assert gp.stats["packed_solo_fallbacks"] == 64 and sum("('pk0', 2048, 64)" in m for m in logs) == 1, "logged once"
    gp._warmed.add(("pk0", 2048, 64))
    reqs = burst(32, 30)
    lg = gp.prefill_forward_batch(reqs, kv_cache=pool)
    _p5_check_call(gp, pool, reqs, lg)
    assert [p.shape for p in gp.last_prefill.passes] == [("pk0", 2048, 64)] and not gp.last_prefill.fallbacks
    # pk1: 4 rows with their own 2048-token histories, resumed at 2048: distinct tails (R-E2)
    hist = []
    for i in range(4):
        t = [rng.randrange(5000) for _ in range(2048 + 50)]
        blk = list(range(3000 + 40 * i, 3000 + 40 * i + 33))
        hist.append((t, blk))
        gp.prefill_forward_batch([_rows_of(t, 0, 2048, blk, width=512, lane=i)], kv_cache=pool)  # solo sp0 2048
    resumed = [_rows_of(t, 2048, len(t), blk, width=512, lane=8 + i) for i, (t, blk) in enumerate(hist)]
    gp._warmed |= {("pk1", 512, 128, "shared")}
    n0 = gp.stats["packed_solo_fallbacks"]
    lg = gp.prefill_forward_batch(resumed, kv_cache=pool)
    _p5_check_call(gp, pool, resumed, lg)
    assert {p.fallback for p in gp.last_prefill.fallbacks} == {("pk1", 512, 128, "distinct")}
    assert gp.stats["packed_solo_fallbacks"] == n0 + 4
    # an unwarmed SOLO shape refuses the call before any device op
    gp._warmed.discard((PP.SP0, 128))
    n_ops, snap, mtp_snap = len(mp.ops), dict(pool.kv), dict(pool.mtp_kv)
    with pytest.raises(RuntimeError, match=r"\('sp0', 128\)"):
        gp.prefill_forward_batch([_p5_req(0, [1, 2, 3], 0, [9000])], kv_cache=pool)
    assert len(mp.ops) == n_ops and pool.kv == snap and pool.mtp_kv == mtp_snap, "a refused call ran something"


def test_cpu_packed_plan_error_runs_per_row(monkeypatch):
    """Packing never refuses a call (design §3.3 step 5; P5 review finding 6): when the packed plan fails its own checks
    -- the planner's invariants (``prefill_plan._check_passes``: ``AssertionError``) or a packed table check
    (``attention.packed_host_tables``: ``ValueError``) -- the call runs with the packing-off passes (writer-first rows):
    the same logits and caches as a packing-off generator, ``plan_error`` names the failure, the call counts
    ``packed_plan_errors`` and logs it; the next call packs again. Bad rows still refuse the call before any planning
    (two rows writing one block), and after the capture an unwarmed solo shape still refuses it (nothing runs)."""
    from models.demos.motif3.tt import generator as G
    from models.demos.motif3.tt.generator import DecodePath

    cfg = host_cfg()
    scen = _p5_scenarios(5)
    for what in ("planner", "tables"):
        logs: List[str] = []
        (gp, mp, pool_p, _), (gs, ms, pool_s, _) = _p5_pair(cfg, monkeypatch)
        gp.log = logs.append
        reqs = scen["mixed"]
        ls = gs.prefill_forward_batch(reqs, kv_cache=pool_s)
        with monkeypatch.context() as m:
            if what == "planner":
                m.setattr(PP, "_check_passes", lambda *a, **k: (_ for _ in ()).throw(AssertionError("injected bug")))
            else:
                m.setattr(G, "packed_host_tables", lambda *a, **k: (_ for _ in ()).throw(ValueError("injected bug")))
            lp = gp.prefill_forward_batch(reqs, kv_cache=pool_p)
        b = gp.last_prefill
        err = "AssertionError: injected bug" if what == "planner" else "ValueError: injected bug"
        assert b.packed and not b.packed_passes and b.plan_error == err, (what, b.plan_error, b.describe())
        assert [p.describe() for p in b.passes] == [p.describe() for p in gs.last_prefill.passes]
        assert torch.equal(lp, ls) and pool_p.kv == pool_s.kv and pool_p.mtp_kv == pool_s.mtp_kv
        _p5_check_call(gp, pool_p, reqs, lp)
        assert gp.stats["packed_plan_errors"] == 1 and gp.stats["packed_passes"] == 0
        assert sum("packed prefill plan" in x and err in x for x in logs) == 1, logs
        burst = scen["burst"]  # the next call packs again
        lb = gp.prefill_forward_batch(burst, kv_cache=pool_p)
        _p5_check_call(gp, pool_p, burst, lb)
        assert gp.last_prefill.plan_error is None and gp.last_prefill.packed_passes
        assert gp.stats["packed_plan_errors"] == 1
    # bad rows still refuse the call before any planning: two rows writing one block
    gp, mp, pool, _ = fake_generator(cfg, monkeypatch, packed=True, num_blocks=P5_NUM_BLOCKS)
    n_ops = len(mp.ops)
    with pytest.raises(ValueError, match="both write block 77"):
        gp.prefill_forward_batch([_p5_req(0, list(range(40)), 0, [77]), _p5_req(1, list(range(50)), 0, [77])],
                                 kv_cache=pool)  # fmt: skip
    assert len(mp.ops) == n_ops and gp.stats["packed_plan_errors"] == 0
    # after the capture, a failed packed plan whose solo shapes were not warmed refuses the call (nothing runs)
    gp._paths[gp.serving_path] = DecodePath(*gp.serving_path, width=512, trace_id=object())  # "captured"
    gp._warmed = set(gp.prefill_shapes()) - {(PP.SP0, 2048)}
    with monkeypatch.context() as m:
        m.setattr(PP, "_check_passes", lambda *a, **k: (_ for _ in ()).throw(AssertionError("injected bug")))
        with pytest.raises(RuntimeError, match=r"\('sp0', 2048\)"):
            gp.prefill_forward_batch(scen["mixed"], kv_cache=pool)
    assert len(mp.ops) == n_ops and pool.kv == {} and gp.stats["packed_plan_errors"] == 0


def test_cpu_packed_warmup(monkeypatch):
    """``warmup_prefill`` with packing on (span cap 1024: 22 packed shapes): the solo shapes first (unchanged), then one
    warm call per packed shape in ``cfg.packed_prefill_shapes()`` order, both pk1 tail variants (R-E2), the MTP
    layer's packed fill once per pass size T; nothing is written; the capture is refused until the packed shapes are
    warmed too; a second warm-up is a no-op; ``packed_warmup="full"`` runs one full warm-up pass per shape (head and
    MTP fill included); packing off warms no packed shape and the capture needs only the solo shapes."""
    cfg = host_cfg(prefill_span_cap=1024)

    def patched(gen, model):
        orig = model.prefill_chunk

        def prefill_chunk(tok, *, chunk, kv_caches):  # warm-up inputs (fill all -1): shapes only
            h = chunk.host
            if int(h.fill.max()) < 0 and (h.is_packed or h.path == PP.SP1):
                model.ops.append(("layers", h.path, int(h.bucket), int(h.start)))
                return FakeT(h=[0] * h.bucket, start=h.start, end=h.end, tokens=tok.tokens, bucket=h.bucket)
            return orig(tok, chunk=chunk, kv_caches=kv_caches)

        model.prefill_chunk = prefill_chunk
        model.mtp.fill_kv_prefill = lambda hn, nxt, *, kv_cache, chunk: model.ops.append(("mtp", chunk.bucket))

    gen, model, pool, _ = fake_generator(cfg, monkeypatch, packed=True)
    patched(gen, model)
    solo, packed = gen.prefill_shapes(), gen.packed_shapes()
    assert packed == list(cfg.packed_prefill_shapes()) and (len(solo), len(packed)) == (8, 22)  # pk0 10, pk1 6 x 2
    assert {s[3] for s in packed if s[0] == "pk1"} == set(api.PK1_TAIL_VARIANTS)
    with pytest.raises(RuntimeError, match="before the prefill warmup: 30 shapes"):
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=512)
    gen.warmup_prefill(kv_cache=pool, enable_trace=False)
    assert gen.warmed_shapes == set(solo) | set(packed)
    assert [o[1:3] for o in model.ops if o[0] == "layers"] == solo, "solo warm-ups unchanged"
    assert [c[0] for c in model.warm_calls] == packed
    first_t = {}
    for s in packed:
        first_t.setdefault(int(s[1]), s)
    assert [c[2] for c in model.warm_calls] == [first_t[int(s[1])] == s for s in packed], "the MTP fill once per T"
    assert pool.kv == {} and not pool.mtp_kv, "warm-ups write nothing"
    assert gen.required_prefill_shapes() == solo + packed
    n = len(model.ops)
    gen.warmup_prefill(kv_cache=pool, enable_trace=False)
    assert len(model.ops) == n
    # full warm-up passes
    g2, m2, p2, _ = fake_generator(cfg, monkeypatch, packed=True)
    patched(g2, m2)
    g2.packed_warmup = "full"
    g2.warmup_prefill(kv_cache=p2, enable_trace=False)
    full = [o for o in m2.ops if o[0] == "layers"][len(solo) :]
    assert [(o[1], o[2]) for o in full] == [(s[0], s[1]) for s in packed] and not m2.warm_calls
    assert sum(o[0] == "head" for o in m2.ops) == len(solo) + len(packed)
    assert sum(o[0] == "mtp" for o in m2.ops) == len(solo) + len(packed)
    # packing off: solo shapes only
    g3, m3, p3, _ = fake_generator(cfg, monkeypatch, packed=False)
    patched(g3, m3)
    g3.warmup_prefill(kv_cache=p3, enable_trace=False)
    assert g3.warmed_shapes == set(solo) and not m3.warm_calls and g3.required_prefill_shapes() == solo


def test_cpu_cp_p_harness(monkeypatch):
    """The device tests' harness on the host: ``Blocks`` hands out fresh ascending ids, then released ones; a scope
    (nested too) releases what it took on exit; ``_RebucketLast`` re-plans exactly the given rows with their last
    chunk at the pass's bucket (chunk starts / ends kept; tables still valid) and restores ``plan_row``; the bucket
    floor then runs per row (no packed pass) at those buckets on the emulated cache with the right logits."""
    b = Blocks(10)
    with b.scope():
        assert b.take(3) == [1, 2, 3]
        with b.scope():
            assert b.take(2) == [4, 5]
        assert b.free == [4, 5]
        assert b.take(4) == [6, 7, 8, 9]
        assert b.take(2) == [4, 5]  # fresh ids exhausted: released ones
        with pytest.raises(RuntimeError, match="KV pool exhausted"):
            b.take(1)
    assert sorted(b.free) == list(range(1, 10))
    cfg = host_cfg()
    gen, model, pool, _ = fake_generator(cfg, monkeypatch, num_blocks=P5_NUM_BLOCKS)
    sc = _p5_scenarios()
    m, sh = sc["mixed"], sc["shared_2k"]
    reqs = [dataclasses.replace(r, lane=i) for i, r in enumerate([m[0], m[24], m[25], m[30], sh[0], sh[1]])]
    keys = [(r.start, r.end) for r in reqs]
    assert len(set(keys)) == len(keys)
    base = [gen.plan_row(*k) for k in keys]
    pick = {keys[0]: 2048, keys[-2]: 4096, keys[-1]: 4096}  # a 40-token sp0 row, the writer's tail, a reader
    with _RebucketLast(gen, pick):
        got = [gen.plan_row(*k) for k in keys]
        logits = gen.prefill_forward_batch(reqs, kv_cache=pool)
        assert not gen.last_prefill.packed_passes
        # the 40-token row at 2048 (rebucketed), the 1500-token row and the writer's head chunk (2048 natively), and
        # the writer's tail and the reader at 4096 (rebucketed)
        assert (
            sorted(p.shape for p in gen.last_prefill.passes if p.shape[1] > 1024)
            == [("sp0", 2048)] * 3 + [("sp1", 4096)] * 2
        )
    assert [gen.plan_row(*k) for k in keys] == base, "plan_row restored"
    for k, g, p in zip(keys, got, base):
        assert g.chunks[:-1] == p.chunks[:-1] and (g.chunks[-1].start, g.chunks[-1].end) == (
            p.chunks[-1].start,
            p.chunks[-1].end,
        )
        assert g.chunks[-1].bucket == pick.get(k, p.chunks[-1].bucket), (k, g, p)
    _p5_check_call(gen, pool, reqs, logits)


class _SynTF:
    """A teacher-forced collector stand-in for :func:`_cpp_row_bars` on the host: ``rows[i][p]`` = logits of position
    ``p`` of request ``i``."""

    def __init__(self, rows: List[torch.Tensor]):
        self.rows = rows

    def take(self, index: int, lo: int, hi: int) -> torch.Tensor:
        return self.rows[index][lo:hi].clone()


def test_cpu_cp_p_row_bars():
    """The per-segment bars of CP-P (``_cpp_row_bars``, P5 review finding 2) on synthetic runs: 8 requests, per-row
    reference A, floor F and packed B on their own blocks, logits with a clear argmax per position, caches L0-L2
    (exact layers: B bitwise, F 1 % noise as the sp1 floor), L52 / MTP (floor-level noise). Clean runs pass. Each defect
    confined to ONE segment fails the new bars while the median / pooled / count bars of the test (as before the review)
    pass: (b) the head row of a segment off by one position, (c) a segment's last position computed wrong (its returned
    and teacher-forced logits agree), (d) a segment's L2 rows (layer 1's SWA attention) off by 3 %, (e) a 40-row
    segment's MTP rows on shifted next tokens in a call of 12,040 rows. An MTP stand-in row whose argmax token differs
    is excused (f)."""
    n, V, D = 8, 96, 32
    names = ("L0", "L1", "L2", "L52", "MTP")
    exact = ["L0", "L1", "L2"]

    def build(lens, *, seed=0):
        g = torch.Generator().manual_seed(seed)
        truth_lg = []
        for L in lens:  # a clear winner per position: noise at the floor's level never flips it
            lg = torch.randn(L, V, generator=g) * 3
            lg[torch.arange(L), torch.randint(V, (L,), generator=g)] += 8.0
            truth_lg.append(lg)
        truth_kv = {k: [torch.randn(L, D, generator=g) for L in lens] for k in names}
        caches = {k: torch.zeros(3 * sum(api.cdiv(L, BS) for L in lens) + 1, 1, BS, D) for k in names}
        runs = {}
        nxt = 1
        for tag, lg_noise, kv_noise in (("A", 0.0, {}), ("F", 0.15, {"L0": 0.01, "L1": 0.01, "L2": 0.01,
                                                                      "L52": 0.1, "MTP": 0.1}),
                                        ("B", 0.15, {"L52": 0.1, "MTP": 0.1})):  # fmt: skip
            blocks, rows = [], []
            for i, L in enumerate(lens):
                blocks.append(list(range(nxt, nxt + api.cdiv(L, BS))))
                nxt += api.cdiv(L, BS)
                rows.append(truth_lg[i] + lg_noise * torch.randn(L, V, generator=g))
                pos = torch.arange(L)
                blk = torch.tensor(blocks[-1])[pos // BS]
                for k in names:
                    caches[k][blk, 0, pos % BS] = truth_kv[k][i] + kv_noise.get(k, 0.0) * torch.randn(L, D, generator=g)
            reqs = [SimpleNamespace(start=0, end=L) for L in lens]
            runs[tag] = CppRun(reqs=reqs, blocks=blocks, logits=torch.stack([r[-1] for r in rows]), passes=[],
                               kinds=set(), fallbacks=0, seconds=0.0, tf=_SynTF(rows), writes=set())  # fmt: skip
        return runs, caches

    def old_bars(A, B, F, caches) -> List[str]:  # CP-P's median / pooled / count bars (unchanged by the review)
        out = []
        lb, lf = _cpp_last(A, B), _cpp_last(A, F)
        if 1 - lb["median"] > max(1 - CPP_LOGIT_PCC, CPP_FLOOR_RATIO * (1 - lf["median"])):
            out.append("median")
        if len(lb["flips"]) > len(lf["flips"]) + CPP_FLIP_SLACK:
            out.append("flips")
        for k, t in caches.items():
            ra, rb, rf = _cpp_rows(t, A), _cpp_rows(t, B), _cpp_rows(t, F)
            if 1 - _pcc(rb, ra) > max(1 - CPP_CACHE_PCC, CPP_FLOOR_RATIO * (1 - _pcc(rf, ra))):
                out.append(f"pooled {k}")
        return out

    lens = [40 + 10 * i for i in range(n)]
    # (a) clean
    runs, caches = build(lens)
    fails, notes = _cpp_row_bars(runs["A"], runs["B"], runs["F"], caches, exact)
    assert not fails and not old_bars(runs["A"], runs["B"], runs["F"], caches), (fails, notes)
    assert all(f"{k} rows per request: bitwise {n}/{n}" in "; ".join(notes) for k in exact), notes
    # (b) the head row of segment 3 off by one position (its teacher-forced rows are right)
    runs, caches = build(lens)
    B = runs["B"]
    B.logits[3] = B.tf.rows[3][-2]
    fails, _ = _cpp_row_bars(runs["A"], B, runs["F"], caches, exact)
    assert not old_bars(runs["A"], B, runs["F"], caches)
    assert any("packed run: the returned logits of rows [3]" in m for m in fails), fails
    # (c) segment 5's last position computed wrong (a RoPE / attention defect at that row: both reads agree)
    runs, caches = build(lens)
    B = runs["B"]
    B.tf.rows[5][-1] = B.tf.rows[5][-5].clone()
    B.logits[5] = B.tf.rows[5][-1]
    fails, _ = _cpp_row_bars(runs["A"], B, runs["F"], caches, exact)
    assert not old_bars(runs["A"], B, runs["F"], caches)
    assert len(fails) == 1 and "last-token logits of rows [5]" in fails[0], fails
    # (d) segment 2's L2 rows 3 % off (layer 1's SWA attention of one segment); the floor is 1 % off everywhere
    runs, caches = build(lens)
    B, g = runs["B"], torch.Generator().manual_seed(7)
    for p in range(lens[2]):
        blk = B.blocks[2][p // BS]
        caches["L2"][blk, 0, p % BS] += 0.03 * torch.randn(D, generator=g)
    fails, _ = _cpp_row_bars(runs["A"], B, runs["F"], caches, exact)
    assert not old_bars(runs["A"], B, runs["F"], caches)
    assert len(fails) == 1 and "L2 cache rows of requests [2]" in fails[0], fails
    # (e) a 40-row segment's MTP rows from shifted next tokens, beside a 12,000-row request (a 0.3 % share of the rows)
    big = [40, 12000]
    runs, caches = build(big, seed=1)
    B = runs["B"]
    rows = torch.stack([caches["MTP"][B.blocks[0][p // BS], 0, p % BS] for p in range(40)])
    for p in range(40):
        caches["MTP"][B.blocks[0][p // BS], 0, p % BS] = rows[(p + 1) % 40]
    fails, _ = _cpp_row_bars(runs["A"], B, runs["F"], caches, exact)
    assert not old_bars(runs["A"], B, runs["F"], caches)
    assert len(fails) == 1 and "MTP cache rows of requests [0]" in fails[0], fails
    # (f) segment 6's argmax flips (near tie: top-2 swapped in both reads): its MTP stand-in row is another token's
    runs, caches = build(lens)
    B = runs["B"]
    last = B.tf.rows[6][-1]
    top2 = torch.topk(last, 2).indices
    last[top2[0]], last[top2[1]] = last[top2[1]].clone(), last[top2[0]].clone()
    B.logits[6] = last.clone()
    p = lens[6] - 1
    caches["MTP"][B.blocks[6][p // BS], 0, p % BS] = torch.randn(D, generator=torch.Generator().manual_seed(3))
    fails, notes = _cpp_row_bars(runs["A"], B, runs["F"], caches, exact)
    assert not fails, (fails, notes)


def test_cpu_packed_negative_controls(monkeypatch):
    """The emulation catches what P5 must never do: (1) the pre-R-E1 schedule of CP-P (vi) (the odd-block readers in
    the writer's level, packed with an unrelated tail chunk: ``test_prefill_plan``'s hand-made plan) reads the
    unwritten block ``[c0, w0)``; (2) a dummy segment that writes; (3) segment tokens shifted by one row; (4) a wrong
    MTP stand-in in a packed pass; (5) the logits of another segment's row (head rows off by one segment)."""
    from models.demos.motif3.tt import generator as G
    from models.demos.motif3.tests.unit.test_prefill_plan import _Ids, _odd_hit_call

    cfg = host_cfg()
    # (1) R-E1
    reqs = _odd_hit_call(_Ids(3))[0]
    gen, model, pool, _ = fake_generator(cfg, monkeypatch, packed=True, num_blocks=P5_NUM_BLOCKS)
    good = gen.plan_prefill_batch(reqs).passes
    segs = {g.key: g for p in good for g in p.segments}
    old = good[:2] + [
        PP.PrefillPass("pk1", (segs[(0, 1)], segs[(2, 0)], segs[(3, 0)]), 128, 4, 2048, tails="distinct"), good[3]
    ]  # fmt: skip
    monkeypatch.setattr(G.PP, "plan_prefill_passes", lambda *a, **k: list(old))
    with pytest.raises(AssertionError, match="global read of unfilled row 2048"):
        gen.prefill_forward_batch(reqs, kv_cache=pool)
    monkeypatch.undo()
    scen = _p5_scenarios(1)
    # (2) a dummy segment that writes (the attention's table checks bypassed)
    gen, model, pool, _ = fake_generator(cfg, monkeypatch, packed=True, num_blocks=P5_NUM_BLOCKS)
    real_tables = G.packed_host_tables

    def leaky(cfg_, p, reqs_, plans):
        h = real_tables(cfg_, p, reqs_, plans)
        fill = h.fill.clone()
        fill[0, -1] = 4321
        return _unchecked(h, fill=fill)

    monkeypatch.setattr(G, "packed_host_tables", leaky)
    with pytest.raises(AssertionError, match="dummy segment 31"):
        gen.prefill_forward_batch(scen["mixed"][:24] + scen["mixed"][30:], kv_cache=pool)
    monkeypatch.undo()
    # (3) tokens shifted by one row
    gen, model, pool, _ = fake_generator(cfg, monkeypatch, packed=True, num_blocks=P5_NUM_BLOCKS)
    real_tok = G.PP.pass_tokens
    monkeypatch.setattr(G.PP, "pass_tokens", lambda p, r, pad: torch.roll(real_tok(p, r, pad), 1))
    with pytest.raises(AssertionError, match="segment 0: not row 0's tokens"):
        gen.prefill_forward_batch(scen["burst"], kv_cache=pool)
    monkeypatch.undo()
    # (4) a wrong MTP stand-in
    gen, model, pool, _ = fake_generator(cfg, monkeypatch, packed=True, num_blocks=P5_NUM_BLOCKS)
    real = G.mtp_next_tokens
    monkeypatch.setattr(G, "mtp_next_tokens", lambda t, s, e, nxt: real(t, s, e, None if nxt is None else nxt + 1))
    with pytest.raises(AssertionError, match="MTP next token of row 0 position 33"):
        gen.prefill_forward_batch(scen["burst"], kv_cache=pool)
    monkeypatch.undo()
    # (5) head rows of the next segment: row k gets row k + 1's logits (no MTP layer: its stand-in check would fire)
    gen, model, pool, _ = fake_generator(cfg, monkeypatch, packed=True, num_blocks=P5_NUM_BLOCKS, mtp=False)
    real_rows = PP.PrefillPass.head_rows
    monkeypatch.setattr(PP.PrefillPass, "head_rows", lambda self: [(k, r + self.seg_rows if k < len(self.segments) - 1
                                                                    else r) for k, r in real_rows(self)])  # fmt: skip
    lg = gen.prefill_forward_batch(scen["burst"], kv_cache=pool)
    with pytest.raises(AssertionError, match="row 0: logits of another prefix"):
        _p5_check_call(gen, pool, scen["burst"], lg)


# ======================================================================================================================
# device tests: the real model from the TT cache (design §5.3)
# ======================================================================================================================
CP_LAYERS = int(os.environ.get("MOTIF3_CP_LAYERS", "53"))  # < 53: a quick plumbing run (the bars assume 53 layers)
# P5: the session warms the packed shapes before the capture (default; MOTIF3_CP_PACKED=0 boots without them and
# skips the CP-P tests), then keeps packing off except inside the CP-P tests
CP_PACKED = os.environ.get("MOTIF3_CP_PACKED", "1").strip() not in ("0", "", "false", "off", "no")
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
    "every 6 later bucket-1024 prefills while a trace was alive; no garbage with no trace alive, with one captured "
    "trace (0 of 30), without the compile (0 of 30) or with one small new program (0 of 30). The tracker "
    "(TT_METAL_TRACE_ALLOC_TRACKING=1) found no live unsafe buffer. Root cause (docs/p5_t64/f3.md §0-§4, verified): "
    "not the traces but the TP-ring all-gather completion race (docs/determinism/INVESTIGATION.md), which corrupted "
    "prefills in every phase of those runs, with one trace, two traces or none; the trace history only moved which "
    "stale bytes the racing gather read, and so whether they looked like garbage. ring_gather='safe' (the default) "
    "closes it: on the same sequence 0 garbage prefills, one residual stream per prompt, and two traces replayed "
    "alternately bitwise equal to each path alone. Several decode traces are safe under F3N rules R1-R5 (f3.md §6; "
    "tt/generator.py): serving with MOTIF3_SPEC_VERIFY=auto captures TWO decode traces (T32-spec, then T64), each once "
    "at warmup, and never re-captures; packed, wide and non-speculating launches capture one. This module still "
    "captures only the serving spec trace of a packed launch and runs CP-X's plain all decode eagerly."
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
    """KV block ids 1 .. NUM_BLOCKS - 1 (0 = vLLM's null block). ``take`` hands out fresh ascending ids while there
    are any (a test may rely on contiguous ids), then ids released by earlier tests: every device test runs inside
    :meth:`scope` (the autouse fixture ``_recycled_blocks``), which returns the ids it took when the test ends (no
    later test reads a finished test's KV)."""

    def __init__(self, n: int):
        self.n, self.next = int(n), 1
        self.free: List[int] = []
        self._scope: Optional[List[int]] = None

    def take(self, k: int) -> List[int]:
        if self.next + k <= self.n:
            out = list(range(self.next, self.next + k))
            self.next += k
        elif len(self.free) >= k:
            out, self.free = self.free[:k], self.free[k:]
        else:
            raise RuntimeError(f"KV pool exhausted: {k} blocks at {self.next} of {self.n}, {len(self.free)} released")
        if self._scope is not None:
            self._scope.extend(out)
        return out

    def scope(self):
        """Context manager: the ids taken inside return to the free list on exit (also when nested: the caller is
        done with their KV)."""
        blocks = self

        class _Scope:
            def __enter__(self_):
                self_.outer, blocks._scope = blocks._scope, []
                return self_

            def __exit__(self_, *exc):
                taken, blocks._scope = blocks._scope, self_.outer
                blocks.free.extend(taken)

        return _Scope()


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
    """``gen.chunk_observer`` (and, for packed passes, ``gen.pass_observer = collector.on_pass``) that collects
    teacher-forced logits: ``want = {row index in the call: (lo, hi)}`` -> ``rows[(row index, position)] = logits [V]``
    for positions ``[lo, hi)`` computed by the call's chunks or packed segments (the LM head on every 32-row tile
    through the tensor-args slice: no new program; FULL_MODEL_VALIDATION §2.1)."""

    def __init__(self, gen, want: Dict[int, Tuple[int, int]], streams_at: Optional[Dict[int, Sequence[int]]] = None):
        self.gen, self.want = gen, dict(want)
        self.rows: Dict[Tuple[int, int], torch.Tensor] = {}
        self.streams_at = {k: set(int(p) for p in v) for k, v in (streams_at or {}).items()}
        self.streams: Dict[Tuple[int, int], torch.Tensor] = {}  # (row index, position) -> X row [4, 4096] fp32

    def __call__(self, job, ch, X):
        self._collect(job.index, int(ch.start), int(ch.end), X, 0)

    def on_pass(self, batch, p, X):
        """A packed pass: segment ``k`` holds positions ``[start, end)`` of row ``segments[k].row`` at packed rows
        ``p.offset(k) ...`` (an offset that is a multiple of the 32-row tile)."""
        for k, g in enumerate(p.segments):
            self._collect(int(g.row), int(g.start), int(g.end), X, p.offset(k))

    def _collect(self, index: int, a: int, e: int, X, off: int) -> None:
        import ttnn

        lo, hi = self.want.get(index, (0, 0))
        sp = sorted(p for p in self.streams_at.get(index, ()) if a <= p < e)
        if sp:  # residual streams of those positions (chip 0; prefill outputs are replicated)
            xt = ttnn.to_torch(ttnn.get_device_tensors(X)[0])  # [1, 4, C, 4096]
            for p in sp:
                self.streams[(index, p)] = xt[0, :, off + p - a].float().clone()
            del xt
        s0, s1 = max(lo, a), min(hi, e)
        if s0 >= s1:
            return
        assert off % 32 == 0, off
        head = self.gen.model.head
        for r0 in range((s0 - a) // 32 * 32, s1 - a, 32):
            li = min(r0 + 31, e - a - 1)
            tile = head.forward_prefill(X, off + li)
            try:
                reader = head._reader(tile)
                views = reader.read(tile)
                rows = head._assemble_decode(views, None, np_views=reader.np_views)  # [32, V]: the tile's rows
            finally:
                ttnn.deallocate(tile)
            for r in range(r0, min(r0 + 32, e - a)):
                if s0 <= a + r < s1:
                    self.rows[(index, a + r)] = rows[r - r0].clone()

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
    (margin <= NEAR_TIE): after a flip the continuations differ by construction, so nothing later is compared. The
    bound is inclusive since 2026-10-04 (lead decision, docs/FINAL_VALIDATION.md §6.1): bf16 logit margins come in
    steps of 1/8 here, and on the TORUS_XY fabric CP-H python_code parts at exactly 0.500 between two control tokens
    (step 11 at 0.250 / 0.125 on TORUS_Y). A
    non-finite margin up to the first divergence (NaN / Inf logits) fails: ``min`` would otherwise return the finite
    margin and excuse a NaN lane."""
    for i, (x, y) in enumerate(zip(a, b)):
        bad = [f"{tag} {m[i]}" for tag, m in (("a", ma), ("b", mb)) if i < len(m) and not math.isfinite(m[i])]
        if bad:
            return False, f"non-finite logits margin at step {i} ({', '.join(bad)})"
        if x != y:
            near = min(ma[i], mb[i]) <= NEAR_TIE
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
        self.gen.pass_observer = col.on_pass if col is not None else None
        try:
            t0 = time.time()
            out = self.gen.prefill_forward_batch(list(rows), kv_cache=self.pool)
            dt = time.time() - t0
        finally:
            self.gen.chunk_observer = self.gen.pass_observer = None
        assert_sane(out, "prefill logits")
        return out, col, dt

    def prefill_observed(self, rows: Sequence[api.PrefillRequest], collector: RowCollector):
        self.gen.chunk_observer, self.gen.pass_observer = collector, collector.on_pass
        try:
            out = self.gen.prefill_forward_batch(list(rows), kv_cache=self.pool)
        finally:
            self.gen.chunk_observer = self.gen.pass_observer = None
        assert_sane(out, "prefill logits")
        return out

    def packing(self, on: bool):
        """Context manager: the generator's packed-prefill switch (P5) for the block, restored after."""
        gen = self.gen

        class _Packing:
            def __enter__(self_):
                self_.old = gen.packed_prefill
                gen.packed_prefill = bool(on)

            def __exit__(self_, *exc):
                gen.packed_prefill = self_.old

        return _Packing()

    def cache_host(self, cache) -> torch.Tensor:
        """Chip 0's copy of a paged cache ``[num_blocks, 1, block, 576]`` on the host (a buffer read: no device
        program, safe while a trace is alive; prefill and KV-R write every chip alike)."""
        import ttnn

        return ttnn.to_torch(ttnn.get_device_tensors(cache)[0])

    def greedy_spec(self, lanes: Sequence[Lane], steps: int) -> Dict[int, List[bool]]:
        """:meth:`greedy` through ``decode_forward_spec`` (ordinary steps of the serving spec trace): also returns, per
        lane, whether the MTP layer's prediction of each step (``m0``: the token after the step's argmax) equals the
        next step's argmax (the draft acceptance a speculating server would see)."""
        acc: Dict[int, List[bool]] = {ln.lane: [] for ln in lanes}
        prev: Dict[int, int] = {}
        for _ in range(steps):
            tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
            pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
            pt = torch.zeros(api.NUM_LANES, WIDTH, dtype=torch.int32)
            for ln in lanes:
                p = len(ln.seq) - 1
                tokens[ln.lane], pos[ln.lane], pt[ln.lane] = ln.seq[-1], p, page_row(ln.blocks, p + 1)
            batch = api.SpecDecodeBatch.from_decode_batch(api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt))
            res = self.gen.decode_forward_spec(batch, kv_cache=self.pool, enable_trace=True, want_logits=True)
            assert_sane(res.logits[[ln.lane for ln in lanes]], "decode logits", [ln.lane for ln in lanes])
            for ln in lanes:
                row = res.logits[ln.lane].float()
                tok = int(row.argmax())
                if ln.lane in prev:
                    acc[ln.lane].append(prev[ln.lane] == tok)
                prev[ln.lane] = int(res.mtp_argmax[ln.lane, 0])
                ln.seq.append(tok)
                ln.margins.append(_margin(row))
        return acc

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
            long_prefill_token_threshold=budget, spec_tokens=1, packed_prefill=CP_PACKED,
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
        # ONE captured decode trace, as in a packed launch. With a second one (the plain all trace) captured next to
        # it, the prefills after a release / new-program compile / re-capture (CP-L's 32K single shot) once returned
        # garbage tiles: the TP-ring all-gather race, closed by ring_gather="safe" (MULTI_TRACE_NOTE). CP-X still runs
        # the plain all decode eagerly, with no trace alive.
        pool = gen.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BS, num_layers=CP_LAYERS)
        assert gen.prefill_alignment == A, "update the test's budget: A moved (gate G9 per-bucket chunks?)"
        t0 = time.time()
        gen.warmup_prefill(kv_cache=pool, enable_trace=False)  # with CP_PACKED: the packed shapes too (P5)
        t_wp = time.time() - t0
        assert gen.packed_prefill == CP_PACKED
        if CP_PACKED:
            assert set(gen.packed_shapes()) <= gen.warmed_shapes
        # the module's tests run the per-row path unless they turn packing on (CP-P: ``session.packing(True)``)
        gen.packed_prefill = False
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=WIDTH)
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=WIDTH)
        log(
            f"warmup: prefill {t_wp:.1f} s ({len(gen.warmed_shapes)} shapes, MTP fill included; packed "
            f"{gen.timings.get('warmup_packed_s', 0):.1f} s), decode eager "
            f"{gen.timings.get('warmup_decode_eager_s', 0):.1f} s, capture "
            f"{gen.timings.get('capture_decode_s', 0):.1f} s;"
            f" program cache {mesh.num_program_cache_entries()}"
        )
        yield Session(mesh, gen, pool, guard)
    finally:
        if gen is not None:
            gen.close()
        close_motif_mesh(mesh)


@pytest.fixture(autouse=True)
def _recycled_blocks(request):
    """Device tests (those that use ``session``): the KV blocks a test takes return to the session's free list when it
    ends, so the module's tests together never exhaust the 4129-block serving pool. Host tests: nothing (no boot)."""
    if "session" not in request.fixturenames:
        yield
        return
    with request.getfixturevalue("session").blocks.scope():
        yield


def _needle_prompt(tok, target_tokens: int) -> Tuple[List[int], str]:
    """A ~``target_tokens`` chat prompt: tt-metal technical reports as the haystack, one needle sentence at 40 % depth,
    a question at the end (thinking off). Returns ``(ids, needle answer)``."""
    from models.demos.motif3.reference.tokenizer import encode_chat

    # the tt-metal checkout's docs: TT_METAL_HOME (scripts/devrun.sh sets it), else this file's checkout (a run from a
    # copy of models/demos/motif3 alone would otherwise build a haystack of blank lines)
    root = Path(os.environ.get("TT_METAL_HOME") or Path(__file__).resolve().parents[4])
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
    found = [f for f in files if (root / f).is_file()]
    assert len(found) >= len(files) // 2, f"needle haystack: only {found} under {root}"
    text = "\n\n".join((root / f).read_text(errors="ignore") for f in found)
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
    truncated-state bar, design §5.3), and every chunked path no further from the reference than the single shot,
    within 0.001 in the median and in the mean error over the sampled positions (lead decision 2026-10-03, re-signed
    under F5: the worst single position is reported, F5's chunked paths sit 0.0015 under the single shot's worst at one
    position while their median and mean error improve). Reported: the design's chunked vs single-shot TT streams >=
    0.999 (2026-10-02: min 0.9981-0.9987, median 0.99975-0.99995; the worst positions are the single shot's worst too).
    """
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
                # asserted (lead decision 2026-10-03, re-signed under F5's per-bucket q / k): the chunked path no
                # further from the reference than the single shot, within 0.001, in the MEDIAN and in the MEAN ERROR
                # (1 - PCC averaged over the sampled positions); the worst single position is reported (F5: 0.99678
                # at 31961 vs the single shot's worst 0.99829, a chaotic single position; the >= 0.99 bar above still
                # holds at every position); reported: the design's chunked vs single-shot TT >= 0.999
                err, err_one = float((1 - st_p).mean()), float((1 - one).mean())
                closer = float(st_p.median()) >= float(one.median()) - 0.001 and err <= err_one + 0.001
                worst_ok = float(st_p.min()) >= float(one.min()) - 0.001
                log(
                    f"CP-L (ii) S={S} {name} vs single shot TT: stream PCC min {float(tt.min()):.6f} median "
                    f"{float(tt.median()):.6f} (design bar 0.999 {'met' if float(tt.min()) >= 0.999 else 'NOT met'}); "
                    f"vs the reference: median {float(st_p.median()):.5f} (single {float(one.median()):.5f}), mean "
                    f"error {err:.6f} (single {err_one:.6f}) -> within 0.001 of the single shot: {closer}; worst "
                    f"position within 0.001 of the single shot's: {worst_ok} (reported)"
                )
                if not closer:
                    fails.append(
                        f"CP-L (ii) S={S} {name}: further from the reference than the single shot (median "
                        f"{float(st_p.median()):.5f} vs {float(one.median()):.5f}, mean error {err:.6f} vs "
                        f"{err_one:.6f})"
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
    first run's blocks (another DP row): the hit row's last-token argmax is the fp32 golden's unless the golden's margin
    is below 0.5 (lead decision 2026-10-03: re-signed against the golden under F5's A = 128, whose hit at 192 resumes
    at 128; the pre-F5 rule against the cold TT row is reported), greedy tokens identical except a near-tie
    divergence; (iii) ``P + X`` cold, then ``P + Y`` (= the C2 prompt, ``P`` its first ``floor(S / 2 / 64)``
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
        # hard bar (lead decision 2026-10-03, re-signed under F5's A = 128): the hit row's argmax is the fp32 golden's
        # (tie-aware, row_metrics) unless the GOLDEN's margin is below 0.5 -- not the cold TT row's argmax and margin,
        # a noisy reference (multi_turn_chat: the hit picks the golden's token at golden margin 0.24 while the cold
        # row picks another at its own margin 0.531). Plus the greedy streams below and the pooled rows of (iii) and
        # CP-C. Reported: the pre-F5 rule (hit vs cold argmax unless the cold margin < 0.5) and the design's single-row
        # PCC >= 0.999, which no TT path meets reliably at full depth (module docstring, "single rows": MoE top-8 near
        # ties flip under any perturbation; the cold repeat misses it too)
        gm = row_metrics(g, torch.tensor([g.S - 1]), lh[None])
        g_agree, g_margin = bool(gm["agree"][0]), float(gm["margin"][0])
        ok = g_agree or g_margin < NEAR_TIE
        old_ok = same or mg < NEAR_TIE
        design_bar.append(p >= 0.999)
        log(
            f"CP-H (ii) {name}: hit {k * BS} ({[(ch.start, ch.bucket, ch.path) for ch in plan.chunks]}, {dt:.2f} s): "
            f"last-token PCC hit/cold {p:.5f} (repeat/cold {p_rep:.5f}; design bar 0.999 "
            f"{'met' if p >= 0.999 else 'NOT met'}); vs fp32 PCC cold {r_c:.5f} hit {r_h:.5f} repeat {r_r:.5f}, KL "
            f"cold {k_c:.4f} hit {k_h:.4f} repeat {k_r:.4f}; hit argmax {'=' if g_agree else '!='} fp32 golden's "
            f"(golden margin {g_margin:.3f}) -> {'ok' if ok else 'FAIL'}; pre-F5 rule (vs cold: argmax "
            f"{'same' if same else 'DIFF'}, cold margin {mg:.3f}) {'met' if old_ok else 'NOT met'} (reported)"
        )
        if not ok:
            fails.append(
                f"CP-H (ii) {name}: the hit row's argmax {int(lh.argmax())} is not the fp32 golden's "
                f"{int(g.argmax[g.S - 1])} at golden margin {g_margin:.3f} (PCC vs fp32 hit {r_h:.5f}, cold {r_c:.5f})"
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


# ---- CP-P: packed prefill (P5) vs per-row ---------------------------------------------------------------------------
CPP_CASES = (
    "i_s64",
    "i_s128",
    "ii_shared2k",
    "iii_mixed",
    "iv_template",
    "v_split_writer",
    "vi_odd_hit",
    "vii_resumed",
)
CPP_WALL_BARS = {"i_s64": 2.0, "i_s128": 3.3, "ii_shared2k": 5.0}  # design §6.2: the packed call's wall time (s)
# the packed passes each case must contain: (kind, tail variant) of its last call
CPP_KINDS = {
    "i_s64": {("pk0", None)},
    "i_s128": {("pk0", None)},
    "ii_shared2k": {("pk1", "shared")},
    "iii_mixed": {("pk0", None)},
    "iv_template": {("pk0", None)},
    "v_split_writer": {("pk1", "shared")},
    "vi_odd_hit": {("pk1", "shared")},
    "vii_resumed": {("pk1", "distinct")},
}
CPP_LOGIT_PCC = 0.999  # design §6.2: last-token logits, packed vs per-row (reported; the floor rule is asserted)
CPP_CACHE_PCC = 0.99999  # design §6.2: KV and MTP cache rows, packed vs per-row (reported; the floor rule is asserted)
CPP_TOP1_PT = 0.003  # teacher-forced top-1 within 0.3 pt of per-row
CPP_NLL_REL = 0.005  # ... NLL within +-0.5 %
CPP_ACCEPT = 0.02  # MTP draft acceptance over the greedy tokens within 2 points (design §7.3)
# The bucket floor (test_cp_p_packed_vs_per_row): packed's error vs per-row (1 - PCC: the median last-token logits
# row, the aggregate of the written cache rows) at most this multiple of the floor's own error, or within the design
# bar. 2026-10-03 at 53 layers the ratio measured 0.8-1.4 (one draw of a chaotic quantity: a few rows dominate); a wrong
# table, RoPE row or token gives errors orders of magnitude above the floor
CPP_FLOOR_RATIO = 2.0
# Count bars (argmax flips at a per-row margin >= 0.5, greedy divergences beyond a near tie): packed may exceed the
# floor's count by this much. The floor is ONE draw of a chaotic count, so an exact comparison fails at random; the
# slack is logged whenever it is used (2026-10-03: CP-P (iv) row 8, (v) row 7). A single bad segment cannot hide in it:
# the per-segment bars below are exact or floor-relative per row (P5 review, finding 2)
CPP_FLIP_SLACK = 1
# Per-segment (per-row) bars (P5 review, finding 2; :func:`_cpp_row_bars`). Every row's last-token logits, and every
# row's written cache rows at the last layer and the MTP layer, within ``1 - CPP_ROW_PCC`` (PCC 0.95) or
# ``CPP_FLOOR_RATIO`` x the floor's worst row; the cache rows of the layers whose inputs only layers 0-1 compute (dense
# MLPs, row-invariant: ``CPP_EXACT_LAYERS``) per row within the design bar or ``CPP_FLOOR_RATIO`` x the floor's error
# of the same row (expected bitwise: 2026-10-03 every L0 / L1 row was); the returned logits of every row equal the
# teacher-forced logits of its last position (the same pass output), bitwise.
CPP_ROW_PCC = 0.95
CPP_EXACT_LAYERS = (0, 1, 2)  # L0 global attention -> L1's cache; L1 SWA attention -> L2's cache
CppRow = Tuple[List[int], int, Optional[Tuple[int, int, int]]]  # (ids, start, (call, row, k): first k blocks shared)


def _cpp_calls(s: Session, case: str) -> List[List[CppRow]]:
    """The prefill calls of a CP-P case (module docstring), rows as ``(ids, start, share)``: ``share = (call, row,
    k)`` = the row's first ``k`` blocks are row ``row`` of call ``call``'s (a prefix hit; a same-step hit when it is
    the same call). Real tokens: C2 prompt prefixes and windows of the C2 token stream."""
    rng = random.Random(1000 + CPP_CASES.index(case))
    gold = list(s.gold.values())
    src = [t for g in gold for t in g.ids]

    def text(n: int) -> List[int]:
        i = rng.randrange(len(src))
        return [src[(i + j) % len(src)] for j in range(n)]

    if case in ("i_s64", "i_s128"):
        lo, hi = (20, 64) if case == "i_s64" else (65, 128)
        lens = [rng.randint(lo, hi) for _ in range(32)]
        return [[(list(gold[i].ids[: lens[i]]) if i < len(gold) else text(lens[i]), 0, None) for i in range(32)]]
    if case == "ii_shared2k":  # one call: the writer's sp0 2048 chunk, then its tail and 31 hits at 2048 (pk1)
        P = text(2048)
        return [[(P + text(60), 0, None)] + [(P + text(rng.randint(20, 100)), 2048, (0, 0, 32)) for _ in range(31)]]
    if case == "iii_mixed":
        return [[(text(n), 0, None) for n in [40] * 24 + [300, 350, 400, 420, 480, 500] + [1500, 1500]]]
    if case == "iv_template":  # hits of 64 tokens < the SWA tail: c0 = 0, they pack (pk0) with their writer
        Tm = list(gold[0].ids[:64])
        return [[(Tm + text(26), 0, None)] + [(Tm + text(rng.randint(20, 40)), 64, (0, 0, 1)) for _ in range(31)]]
    if case == "v_split_writer":  # the writer's (0, 2048) + (2048, 256) chunks; 8 hits on its first 32 blocks
        W = text(2300)
        return [[(W, 0, None)] + [(W[:2048] + text(rng.randint(30, 120)), 2048, (0, 0, 32)) for _ in range(8)]]
    if case == "vi_odd_hit":  # R-E1: readers hit 33 blocks (c0 2048 < w0 2112) of X, whose chunk boundary is 2048
        U, X = text(2148), text(2200)
        return [[(U, 0, None), (X, 0, None)] + [(X[:2112] + text(50), 2112, (0, 1, 33)) for _ in range(2)]]
    if case == "vii_resumed":  # 8 sessions with their own 1024-token histories, resumed at 1024 (pk1, distinct)
        H = [text(1024) for _ in range(8)]
        return [
            [(h, 0, None) for h in H],
            [(h + text(rng.randint(30, 120)), 1024, (0, i, 16)) for i, h in enumerate(H)],
        ]
    raise ValueError(case)


@dataclasses.dataclass
class CppRun:
    """One run of a CP-P case on its own blocks: the last call's rows."""

    reqs: List[api.PrefillRequest]
    blocks: List[List[int]]
    logits: torch.Tensor
    passes: List[str]
    kinds: set
    fallbacks: int
    seconds: float
    tf: Optional[RowCollector]
    writes: set  # every block id the run's calls write
    calls: List[List[api.PrefillRequest]] = dataclasses.field(default_factory=list)  # every call's requests


class _RebucketLast:
    """Context manager for the bucket floor: ``gen.plan_row`` re-plans every row ``(start, end)`` of ``buckets`` with
    its LAST chunk at bucket ``buckets[(start, end)]`` (the packed pass's ``T``: the row-local programs at the pass's
    rows, the per-row attention), the other chunks as planned."""

    def __init__(self, gen, buckets: Dict[Tuple[int, int], int]):
        self.gen, self.buckets = gen, dict(buckets)

    def __enter__(self):
        orig = self.orig = self.gen.plan_row

        def plan_row(start, end):
            plan = orig(start, end)
            T = self.buckets.get((int(start), int(end)))
            last = plan.chunks[-1]
            if T is None or T <= last.bucket:
                return plan
            return dataclasses.replace(plan, chunks=plan.chunks[:-1] + (dataclasses.replace(last, bucket=int(T)),))

        self.gen.plan_row = plan_row
        return self

    def __exit__(self, *exc):
        del self.gen.plan_row  # the instance override: back to the class method


def _cpp_run(
    s: Session, calls: List[List[CppRow]], *, packed: bool, observe: bool, rebucket: Optional[List[Dict]] = None
) -> CppRun:
    """Run ``calls`` on fresh blocks (capacity for the greedy tokens) with packing ``packed``; teacher-forced logits of
    every real row of the last call when ``observe``; ``rebucket[c]``: call ``c``'s rows at the bucket of the packed
    pass that would hold them (:class:`_RebucketLast`, per row: the bucket floor)."""
    blocks: List[List[List[int]]] = []
    writes: set = set()
    out = col = None
    all_reqs: List[List[api.PrefillRequest]] = []
    with s.packing(packed):
        for c, rows in enumerate(calls):
            reqs, blks = [], []
            for i, (ids, start, share) in enumerate(rows):
                need = api.cdiv(len(ids) + DECODE_STEPS + 1, BS)
                if share is None:
                    b = s.blocks.take(need)
                else:
                    sc, sr, k = share
                    src_blk = blocks[sc][sr] if sc < c else blks[sr]
                    b = list(src_blk[:k]) + s.blocks.take(need - k)
                blks.append(b)
                reqs.append(prefill_request(i, ids, len(ids), b, start=start))
                writes |= set(b[start // BS : api.cdiv(len(ids), BS)])
            blocks.append(blks)
            all_reqs.append(reqs)
            last = c == len(calls) - 1
            want = {i: (r.start, r.end) for i, r in enumerate(reqs)} if (observe and last) else None
            if rebucket and rebucket[c]:
                with _RebucketLast(s.gen, rebucket[c]):
                    out, col, _ = s.prefill(reqs, want)
            else:
                out, col, _ = s.prefill(reqs, want)
    b = s.gen.last_prefill
    return CppRun(
        reqs=reqs, blocks=blocks[-1], logits=out.clone(), passes=[p.describe() for p in b.passes],
        kinds={(p.kind, p.tails) for p in b.packed_passes}, fallbacks=len(b.fallbacks),
        seconds=s.gen.timings["last_prefill_s"] + s.gen.timings["last_prefill_plan_s"], tf=col, writes=writes,
        calls=all_reqs,
    )  # fmt: skip


def _cpp_tf(run: CppRun) -> Dict[str, float]:
    """Teacher-forced top-1 accuracy (argmax = the actual next token) and NLL over every real row of the last call
    that has a next token."""
    hit, nll = [], []
    for i, r in enumerate(run.reqs):
        lo, hi = r.start, r.end - 1  # rows with a known next token
        if hi <= lo:
            continue
        lg = run.tf.take(i, lo, hi).float()
        tgt = r.tokens[lo + 1 : hi + 1].to(torch.int64)
        hit.append(lg.argmax(-1) == tgt)
        nll.append(torch.logsumexp(lg.double(), -1) - lg.double().gather(1, tgt[:, None])[:, 0])
    return {"rows": int(sum(h.numel() for h in hit)), "top1": float(torch.cat(hit).float().mean()),
            "nll": float(torch.cat(nll).mean())}  # fmt: skip


def _cpp_rows(cache: torch.Tensor, run: CppRun) -> torch.Tensor:
    """The cache rows ``[n, 576]`` of every position a run's last call wrote (``[w0, end)`` of each row)."""
    out = []
    for r, blk in zip(run.reqs, run.blocks):
        for p in range(r.start // BS * BS, r.end):
            out.append(cache[blk[p // BS], 0, p % BS])
    return torch.stack(out).float()


def _cpp_last(A: CppRun, X: CppRun) -> Dict[str, Any]:
    """Last-token logits of ``X`` vs the per-row run ``A``: per-row PCC, bitwise rows, and the rows whose argmax
    differs from A's where A's margin is >= 0.5 (not a near tie)."""
    pcc = torch.tensor([_pcc(a.float(), x.float()) for a, x in zip(A.logits, X.logits)])
    flips = [i for i, (a, x) in enumerate(zip(A.logits, X.logits))
             if int(a.float().argmax()) != int(x.float().argmax()) and _margin(a) >= NEAR_TIE]  # fmt: skip
    return {"pcc": pcc, "median": float(pcc.median()), "min": float(pcc.min()),
            "below": int((pcc < CPP_LOGIT_PCC).sum()), "bitwise": sum(torch.equal(a, x) for a, x in zip(A.logits,
                                                                                                         X.logits)),
            "flips": flips}  # fmt: skip


def _cpp_req_rows(cache: torch.Tensor, run: CppRun) -> List[torch.Tensor]:
    """Per request of a run's last call: its written cache rows ``[w0, end)`` (``[n, 576]`` fp32, position order)."""
    out = []
    for r, blk in zip(run.reqs, run.blocks):
        rows = [cache[blk[p // BS], 0, p % BS] for p in range(r.start // BS * BS, r.end)]
        out.append(torch.stack(rows).float())
    return out


def _cpp_row_bars(
    A: CppRun, B: CppRun, F: CppRun, caches: Dict[str, torch.Tensor], exact: Sequence[str]
) -> Tuple[List[str], List[str]]:
    """The per-segment bars of CP-P (P5 review, finding 2): a defect confined to ONE packed segment (a wrong head row,
    RoPE row, token, fill, SDPA table, SWA tail or MTP next token of one row) must fail, although the median and
    pooled bars of :func:`test_cp_p_packed_vs_per_row` average it away and the count bars allow ``CPP_FLIP_SLACK``.
    ``A`` per row (the reference), ``B`` packed, ``F`` the bucket floor; ``caches``: host copies of the paged caches
    after the runs (``"L<i>"`` / ``"MTP"``), ``exact``: the names among them whose rows only layers 0-1 compute.
    Returns ``(fails, notes)``:

    * head rows (exact): every run's returned logits row equals the teacher-forced logits of the row's last position
      (the LM head on the same pass output, through the tile the generator read): a head row or logits-to-row mapping
      off by any row fails bitwise;
    * last-token logits: every packed row's error (1 - PCC vs per-row) within ``1 - CPP_ROW_PCC`` or
      ``CPP_FLOOR_RATIO`` x the floor's worst row;
    * the cache rows each request wrote, per request: ``exact`` layers within the design bar (1 - ``CPP_CACHE_PCC``) or
      ``CPP_FLOOR_RATIO`` x the floor's error on the same request (packed is bitwise there: every L0 / L1 row on
      2026-10-03); the other layers within ``1 - CPP_ROW_PCC`` or ``CPP_FLOOR_RATIO`` x the floor's worst request. The
      MTP row of a request's last position is left out where a run's last-token argmax (its MTP stand-in token)
      differs from per-row's: its input token differs by design."""
    fails: List[str] = []
    notes: List[str] = []
    for tag, run in (("per-row", A), ("floor", F), ("packed", B)):
        bad = []
        for i, r in enumerate(run.reqs):
            tf = run.tf.take(i, r.end - 1, r.end)[0]
            if not torch.equal(run.logits[i].float(), tf.float()):
                bad.append(i)
        if bad:
            fails.append(f"{tag} run: the returned logits of rows {bad} are not the teacher-forced logits of their "
                         f"last position (a head row or logits-to-row mapping is off)")  # fmt: skip
    pb = _row_pccs(B.logits.float(), A.logits.float())
    pf = _row_pccs(F.logits.float(), A.logits.float())
    eb, ef = 1 - pb, 1 - pf
    bar = max(1 - CPP_ROW_PCC, CPP_FLOOR_RATIO * float(ef.max()))
    worst = int(eb.argmax())
    notes.append(f"last-token per row: packed worst row {worst} PCC {float(pb[worst]):.5f}, floor worst "
                 f"{float(pf.min()):.5f}, bar {1 - bar:.5f}")  # fmt: skip
    if float(eb.max()) > bar:
        fails.append(f"last-token logits of rows {[i for i in range(len(eb)) if float(eb[i]) > bar]} vs per-row: PCC "
                     f"{[round(float(pb[i]), 5) for i in range(len(eb)) if float(eb[i]) > bar]} below {1 - bar:.5f} "
                     f"(the floor's worst row {float(pf.min()):.5f})")  # fmt: skip
    am = {tag: [int(lg.float().argmax()) for lg in run.logits] for tag, run in (("A", A), ("B", B), ("F", F))}
    for name, t in caches.items():
        ra, rb, rf = _cpp_req_rows(t, A), _cpp_req_rows(t, B), _cpp_req_rows(t, F)
        e_b, e_f = [], []
        for i in range(len(ra)):
            for tag, rx, acc in (("B", rb, e_b), ("F", rf, e_f)):
                keep = ra[i].shape[0]
                if name == "MTP" and am[tag][i] != am["A"][i]:
                    keep -= 1  # the stand-in row: another input token (the run's own argmax)
                same = keep <= 0 or torch.equal(rx[i][:keep], ra[i][:keep])
                acc.append(0.0 if same else 1 - _pcc(rx[i][:keep], ra[i][:keep]))
        if name in exact:
            bad = [i for i in range(len(ra)) if e_b[i] > max(1 - CPP_CACHE_PCC, CPP_FLOOR_RATIO * e_f[i])]
            nbit = sum(torch.equal(rb[i], ra[i]) for i in range(len(ra)))
            notes.append(f"{name} rows per request: bitwise {nbit}/{len(ra)}, worst error {max(e_b):.3g} (floor "
                         f"{max(e_f):.3g})")  # fmt: skip
            if bad:
                fails.append(f"{name} cache rows of requests {bad} vs per-row: errors "
                             f"{[float(f'{e_b[i]:.3g}') for i in bad]} above {1 - CPP_CACHE_PCC:g} and "
                             f"{CPP_FLOOR_RATIO} x the floor's on the same request "
                             f"{[float(f'{e_f[i]:.3g}') for i in bad]}")  # fmt: skip
        else:
            bar = max(1 - CPP_ROW_PCC, CPP_FLOOR_RATIO * max(e_f))
            bad = [i for i in range(len(ra)) if e_b[i] > bar]
            notes.append(f"{name} rows per request: worst PCC {1 - max(e_b):.5f} (request {e_b.index(max(e_b))}), "
                         f"floor worst {1 - max(e_f):.5f}, bar {1 - bar:.5f}")  # fmt: skip
            if bad:
                fails.append(f"{name} cache rows of requests {bad} vs per-row: PCC "
                             f"{[round(1 - e_b[i], 5) for i in bad]} below {1 - bar:.5f} (the floor's worst request "
                             f"{1 - max(e_f):.5f})")  # fmt: skip
    return fails, notes


@pytest.mark.timeout(CP_TIMEOUT)
@pytest.mark.parametrize("case", CPP_CASES)
def test_cp_p_packed_vs_per_row(session, case):
    """CP-P (design §6.2; P5): the case's prefill call(s) four ways, each on its own blocks, in the session that warmed
    the packed shapes before its capture: A per row (packing off: the reference); F the **bucket floor** -- per row
    with, in every call, each packed row's chunk at the bucket ``T`` of the pass that packs it (the row-local programs
    at the pass's rows: "the tolerance prefill already has between buckets", P5N §5.8); B packed; C packed again.

    At full depth single rows are chaotic (module docstring; CP-H: hit vs cold last-token PCC down to 0.85): the MoE
    layers at M = T and at M = the row's bucket are not bitwise (layers 0-1, dense, are), and MoE top-8 near ties
    amplify that. So the single-row design bars (last-token PCC >= 0.999 per row; cache rows >= 0.99999) are reported
    and asserted relative to F, as CP-X asserts against its sp1 floor. 2026-10-03 (53 layers) F missed them as much as
    B (i_s64: last-token median 0.9972 vs 0.9966, 30 vs 29 of 32 rows below 0.999; 32 greedy tokens identical to A on
    11 of 32 rows): greedy outputs depend on the rows a burst packs together through M = T, not through packing. The
    lead accepted this floor rule and the batch-dependence contract on 2026-10-04 (docs/P5_T64_REVIEW.md I-1: P5 on in
    both TIS specs; MOTIF3_PACKED_PREFILL=0 restores per-row prefill). Asserted:

    * the packed call runs the expected packed passes (``CPP_KINDS``; pk1 tail variants included), no solo fallback,
      and the program cache does not grow (nothing compiles after the capture: F3N rule R2);
    * last-token logits vs A: B's median error (1 - PCC) at most 2 x F's, or the design's 0.999; B's argmax differs
      from A's at an A margin >= 0.5 on no more rows than F's does, plus ``CPP_FLIP_SLACK`` (logged when used);
    * per segment (:func:`_cpp_row_bars`, P5 review finding 2; a defect in ONE segment must fail): every run's
      returned logits equal the teacher-forced logits of the row's last position, bitwise (head rows); every row's
      last-token error within 0.05 or 2 x F's worst row; per request, the written cache rows of L0, L1 and L2 (whose
      inputs only the dense layers 0-1 compute: packed is bitwise there) within the design's 0.99999 or 2 x F's error
      on the same request, and those of the last layer and the MTP layer within 0.05 or 2 x F's worst request (an MTP
      stand-in row whose argmax token differs is left out);
    * teacher-forced (the LM head on every tile of every segment): top-1 no more than 0.3 pt below A's and NLL within
      +-0.5 % of A's (the design's pooled bars, as CP-H (iii) and CP-C), or within 2 x F's own deviation from A;
    * the packed call repeated (C): logits and written cache rows bitwise identical to B;
    * the cache rows the call writes (layers 0, 1, 2, the last, and the MTP layer: chip 0) vs A: aggregate error (1 -
      PCC) at most 2 x F's, or the design's 0.99999; every block outside B's and C's fill sets bitwise untouched by
      them (layer 0 and the MTP cache);
    * 32 greedy tokens per row, B vs A: identical except a near-tie divergence (margin < 0.5 on either side), on all
      but as many rows as F's streams diverge from A's otherwise, plus ``CPP_FLIP_SLACK`` (logged when used); the MTP
      layer's draft acceptance over them within 2 points (rows whose streams agree);
    * the packed call's wall time within ``CPP_WALL_BARS`` (i: 2.0 s, i at <= 128 tokens: 3.3 s, ii: 5.0 s).

    Reported: the design bars as such, bitwise rows, F's identical greedy rows, the per-row and floor call times, the
    fp32-golden top-1 of the C2-prefix rows of (i)."""
    s = session
    if not CP_PACKED:
        pytest.skip("MOTIF3_CP_PACKED=0: the session did not warm the packed shapes")
    gen = s.gen
    pc0 = s.programs()
    calls = _cpp_calls(s, case)
    fails: List[str] = []
    A = _cpp_run(s, calls, packed=False, observe=True)
    rebucket: List[Dict[Tuple[int, int], int]] = []
    for reqs in A.calls:  # the packed plan of each call (host only): each packed row's pass size T
        with s.packing(True):
            plan = gen.plan_prefill_batch(reqs)
        m: Dict[Tuple[int, int], int] = {}
        for p in plan.packed_passes:
            for g in p.segments:
                if g.last:
                    key = (reqs[g.row].start, reqs[g.row].end)
                    m[key] = max(m.get(key, 0), p.tokens)
        rebucket.append(m)
    F = _cpp_run(s, calls, packed=False, observe=True, rebucket=rebucket)
    snap = {k: s.cache_host(c) for k, c in (("L0", s.pool[0]), ("MTP", s.pool.mtp))}
    B = _cpp_run(s, calls, packed=True, observe=True)
    C = _cpp_run(s, calls, packed=True, observe=False)
    log(
        f"CP-P {case}: per-row {A.seconds:.2f} s ({len(A.passes)} passes); floor {F.seconds:.2f} s "
        f"({'; '.join(F.passes)}); packed {B.seconds:.2f} s / repeat {C.seconds:.2f} s: {'; '.join(B.passes)}"
    )
    if not CPP_KINDS[case] <= B.kinds or B.fallbacks or C.passes != B.passes:
        fails.append(
            f"{case}: packed passes {B.passes} (kinds {B.kinds}, fallbacks {B.fallbacks}), want {CPP_KINDS[case]}"
        )
    # ---- last-token logits vs per-row: packed, and the bucket floor; the repeat ------------------------------------
    lb, lf = _cpp_last(A, B), _cpp_last(A, F)
    slack_used: List[str] = []
    if 1 - lb["median"] > max(1 - CPP_LOGIT_PCC, CPP_FLOOR_RATIO * (1 - lf["median"])):
        fails.append(f"{case}: last-token PCC median vs per-row {lb['median']:.6f}: error above {CPP_FLOOR_RATIO} x "
                     f"the floor's (median {lf['median']:.6f})")  # fmt: skip
    if len(lb["flips"]) > len(lf["flips"]) + CPP_FLIP_SLACK:
        fails.append(f"{case}: argmax differs from per-row at margin >= {NEAR_TIE} on rows {lb['flips']} (the floor: "
                     f"{lf['flips']})")  # fmt: skip
    elif len(lb["flips"]) > len(lf["flips"]):
        slack_used.append(f"argmax flips at margin >= {NEAR_TIE}: packed {lb['flips']}, floor {lf['flips']}")
    if not torch.equal(B.logits, C.logits):
        fails.append(f"{case}: the packed call repeated is not bitwise identical (logits)")
    # ---- teacher-forced rows ----------------------------------------------------------------------------------------
    ta, tb, tf = _cpp_tf(A), _cpp_tf(B), _cpp_tf(F)
    d_top1, d_nll = tb["top1"] - ta["top1"], tb["nll"] / ta["nll"] - 1
    f_top1, f_nll = tf["top1"] - ta["top1"], tf["nll"] / ta["nll"] - 1
    top1_bad = -d_top1 > max(CPP_TOP1_PT, CPP_FLOOR_RATIO * abs(f_top1))
    if top1_bad or abs(d_nll) > max(CPP_NLL_REL, CPP_FLOOR_RATIO * abs(f_nll)):
        fails.append(f"{case}: teacher-forced top-1 {100 * d_top1:+.2f} pt, NLL {100 * d_nll:+.3f} % vs per-row (the "
                     f"floor: {100 * f_top1:+.2f} pt, {100 * f_nll:+.3f} %)")  # fmt: skip
    gold_note = ""
    if case.startswith("i_"):  # C2-prefix rows: vs the fp32 golden
        gm: Dict[str, List[Dict[str, torch.Tensor]]] = {"A": [], "B": []}
        for i, g in enumerate(list(s.gold.values())[: len(A.reqs)]):
            e = A.reqs[i].end
            if e > g.S:
                continue
            for tag, run in (("A", A), ("B", B)):
                gm[tag].append(row_metrics(g, torch.arange(e), run.tf.take(i, 0, e)))
        if gm["A"]:
            ga, gb = pooled(gm["A"]), pooled(gm["B"])
            gold_note = f"; C2 rows vs fp32 golden top-1 per-row {ga['agree']:.4f} packed {gb['agree']:.4f}"
    # ---- caches: written rows vs per-row (and the floor's); the repeat; blocks outside the fill sets -----------------
    names = {f"L{i}": i for i in (*CPP_EXACT_LAYERS, gen.num_layers - 1) if i < gen.num_layers}
    after = {k: s.cache_host(s.pool[i]) for k, i in names.items()}
    after["MTP"] = s.cache_host(s.pool.mtp)
    exact = [f"L{i}" for i in CPP_EXACT_LAYERS if i < gen.num_layers]  # their inputs: layers 0-1 only (dense)
    row_fails, row_notes = _cpp_row_bars(A, B, F, after, exact)
    fails += [f"{case}: per segment: {m}" for m in row_fails]
    cache_note = []
    for k, t in after.items():
        ra, rb, rc, rf = _cpp_rows(t, A), _cpp_rows(t, B), _cpp_rows(t, C), _cpp_rows(t, F)
        p_b, p_f = _pcc(rb, ra), _pcc(rf, ra)
        n_bit = int((rb == ra).all(-1).sum())
        cache_note.append(f"{k} pcc {p_b:.7f} (row min {float(_row_pccs(rb, ra).min()):.6f}; floor {p_f:.7f}) "
                          f"bitwise {n_bit}/{ra.shape[0]}")  # fmt: skip
        if 1 - p_b > max(1 - CPP_CACHE_PCC, CPP_FLOOR_RATIO * (1 - p_f)):
            fails.append(f"{case}: {k} cache rows packed vs per-row PCC {p_b:.7f}: error above {CPP_FLOOR_RATIO} x the "
                         f"floor's ({p_f:.7f})")  # fmt: skip
        if not torch.equal(rb, rc):
            fails.append(f"{case}: {k} cache rows of the packed repeat not bitwise identical")
    allowed = B.writes | C.writes
    for k in ("L0", "MTP"):
        changed = set(torch.nonzero((snap[k] != after[k]).flatten(1).any(-1)).flatten().tolist())
        stray = sorted(changed - allowed)
        if stray:
            fails.append(f"{case}: {k} blocks outside the packed fill sets changed: {stray[:8]}")
    del snap, after
    # ---- greedy continuations + MTP acceptance -----------------------------------------------------------------------
    lanes, acc = {}, {}
    for tag, run in (("B", B), ("A", A), ("F", F)):
        ls = [Lane(i, list(r.tokens.tolist()) + [int(lg.float().argmax())], blk, [_margin(lg)])
              for i, (r, lg, blk) in enumerate(zip(run.reqs, run.logits, run.blocks))]  # fmt: skip
        acc[tag] = s.greedy_spec(ls, DECODE_STEPS - 1)
        lanes[tag] = ls
    same_rows, div, same_f = [], [], 0
    hard: Dict[str, List[str]] = {"B": [], "F": []}
    for i in range(len(A.reqs)):
        n0 = A.reqs[i].end
        la = lanes["A"][i]
        for tag in ("B", "F"):
            lx = lanes[tag][i]
            ok, why = compare_streams(la.seq[n0:], la.margins, lx.seq[n0:], lx.margins)
            if la.seq[n0:] == lx.seq[n0:]:
                if tag == "B":
                    same_rows.append(i)
                else:
                    same_f += 1
            elif not ok:
                hard[tag].append(f"row {i}: {why}")
            elif tag == "B":
                div.append(f"row {i}: {why}")
    if len(hard["B"]) > len(hard["F"]) + CPP_FLIP_SLACK:
        fails.append(f"{case}: greedy tokens diverge from per-row beyond a near tie on {hard['B']} (the floor: "
                     f"{hard['F']})")  # fmt: skip
    elif len(hard["B"]) > len(hard["F"]):
        slack_used.append(f"greedy divergences beyond a near tie: packed {hard['B']}, floor {hard['F']}")
    acc_a = [x for i in same_rows for x in acc["A"][i]]
    acc_b = [x for i in same_rows for x in acc["B"][i]]
    rate_a = sum(acc_a) / max(1, len(acc_a))
    rate_b = sum(acc_b) / max(1, len(acc_b))
    if acc_a and abs(rate_a - rate_b) > CPP_ACCEPT:
        fails.append(f"{case}: MTP acceptance packed {rate_b:.3f} vs per-row {rate_a:.3f} on the identical streams")
    bar = CPP_WALL_BARS.get(case)
    if bar is not None and C.seconds > bar:
        fails.append(f"{case}: packed call {C.seconds:.2f} s > {bar} s (per-row {A.seconds:.2f} s)")
    log(
        f"CP-P {case}: last-token PCC vs per-row: packed median {lb['median']:.6f} min {lb['min']:.6f} (< "
        f"{CPP_LOGIT_PCC}: {lb['below']}/{len(lb['pcc'])}), bitwise {lb['bitwise']}, argmax flips at margin >= 0.5 "
        f"{lb['flips']}; floor median {lf['median']:.6f} min {lf['min']:.6f} (< {CPP_LOGIT_PCC}: {lf['below']}), "
        f"bitwise {lf['bitwise']}, flips {lf['flips']}; teacher-forced ({tb['rows']} rows) top-1 {tb['top1']:.4f} vs "
        f"{ta['top1']:.4f} (floor {tf['top1']:.4f}), NLL {tb['nll']:.4f} vs {ta['nll']:.4f} (floor {tf['nll']:.4f})"
        f"{gold_note}; caches: {'; '.join(cache_note)}; greedy {len(same_rows)}/{len(A.reqs)} identical (floor "
        f"{same_f}/{len(A.reqs)}), near-tie divergences {div}, beyond a near tie {hard['B']} (floor {hard['F']}); MTP "
        f"acceptance packed {rate_b:.3f} per-row {rate_a:.3f} ({len(acc_a)} drafts); wall packed {C.seconds:.2f} s "
        f"(bar {bar}) vs per-row {A.seconds:.2f} s (x{A.seconds / max(C.seconds, 1e-9):.1f}); programs {pc0} -> "
        f"{s.programs()}"
    )
    log(f"CP-P {case} per segment: {'; '.join(row_notes)}; count slack ({CPP_FLIP_SLACK}) used: {slack_used or 'no'}")
    if s.programs() != pc0:
        fails.append(f"{case}: program cache grew after the capture: {pc0} -> {s.programs()}")
    assert not fails, "\n".join(fails)


@pytest.mark.timeout(CP_TIMEOUT)
def test_cp_p_burst_ttft_report(session):
    """P5 burst TTFT at generator level (design §1, §10; report, plus two bars): ``prefill_forward_batch`` wall time
    (host tokens to host logits, MTP fill included, no observer) per row and packed for bursts of 32 prompts of ~34,
    ~100, ~300 and ~600 tokens, of 32 rows behind a shared 2K prefix (one call), and of 16 short prompts; then the
    decode stall (design §1, §7.3): 16 lanes decode through the traced spec step and a packed 16-row burst of short
    prompts (the most a 32-lane server admits beside them) runs between two steps: the decode gap (one step + the
    call) within the design's 2.5 s stall bar of burst (a) (expected ~1.1 s + a step, design §10; per row ~10 s).
    Asserted: the 32 x ~34 packed call within 2.0 s, the decode gap within 2.5 s, the program cache constant."""
    s = session
    if not CP_PACKED:
        pytest.skip("MOTIF3_CP_PACKED=0: the session did not warm the packed shapes")
    rng = random.Random(77)
    src = [t for g in s.gold.values() for t in g.ids]
    pc0 = s.programs()

    def text(n):
        i = rng.randrange(len(src))
        return [src[(i + j) % len(src)] for j in range(n)]

    def burst(n_rows, lo, hi, shared=0):
        P = text(shared)
        rows, blk0 = [], s.blocks.take(api.cdiv(shared, BS)) if shared else []
        for i in range(n_rows):
            ids = P + text(rng.randint(lo, hi))
            b = blk0 + s.blocks.take(api.cdiv(len(ids) + DECODE_STEPS + 1, BS) - len(blk0))
            rows.append(prefill_request(i, ids, len(ids), b, start=shared if (shared and i) else 0))
        return rows

    report, fails = [], []
    per_row_16: Optional[float] = None
    for name, args in (("32 x ~34", (32, 30, 38)), ("32 x ~100", (32, 90, 110)), ("32 x ~300", (32, 280, 320)),
                       ("32 x ~600", (32, 560, 640)), ("16 x ~34", (16, 30, 38)),
                       ("32 behind a shared 2K (1 call)", (32, 30, 90, 2048))):  # fmt: skip
        t = {}
        for packed in (False, True):
            with s.packing(packed):
                for rep in range(2 if packed else 1):
                    with s.blocks.scope():  # the burst's KV is not needed afterwards
                        rows = burst(*args)
                        t0 = time.time()
                        s.gen.prefill_forward_batch(rows, kv_cache=s.pool)
                        t[(packed, rep)] = time.time() - t0
            if packed:
                desc = s.gen.last_prefill.describe()
        tp = min(t[(True, 0)], t[(True, 1)])
        report.append(f"{name}: per row {t[(False, 0)]:.2f} s, packed {t[(True, 0)]:.2f} / {t[(True, 1)]:.2f} s "
                      f"(x{t[(False, 0)] / tp:.1f}): {desc}")  # fmt: skip
        if name == "32 x ~34" and t[(True, 1)] > 2.0:
            fails.append(f"32 x ~34 packed call {t[(True, 1)]:.2f} s > 2.0 s")
        if name == "16 x ~34":
            per_row_16 = t[(False, 0)]
    # decode stall: 16 lanes decode; a packed 16-row burst between two traced steps
    lanes = []
    for i in range(16):
        ids = text(rng.randint(100, 300))
        b = s.blocks.take(api.cdiv(len(ids) + 64, BS))
        out, _, _ = s.prefill([prefill_request(16 + i, ids, len(ids), b)])
        lanes.append(Lane(16 + i, list(ids) + [int(out[0].float().argmax())], b))
    step_t = []
    for _ in range(4):
        t0 = time.time()
        s.greedy(lanes, 1)
        step_t.append(time.time() - t0)
    step = sorted(step_t)[len(step_t) // 2]
    with s.packing(True):
        rows = burst(16, 30, 38)
        t0 = time.time()
        s.greedy(lanes, 1)
        t1 = time.time()
        s.gen.prefill_forward_batch(rows, kv_cache=s.pool)
        t2 = time.time()
        s.greedy(lanes, 1)
        t3 = time.time()
    gap, call = t3 - t1, t2 - t1
    per_row = per_row_16 if per_row_16 is not None else float("nan")
    report.append(f"decode stall: 16 decoding lanes, a packed 16 x ~34 burst between two steps: step {step:.3f} s, "
                  f"burst call {call:.2f} s, decode gap {gap:.2f} s (bar 2.5 s; the same burst per row: a call of "
                  f"{per_row:.2f} s)")  # fmt: skip
    if gap > 2.5:
        fails.append(f"decode stall {gap:.2f} s > 2.5 s (burst call {call:.2f} s, one step {step:.3f} s)")
    for line in report:
        log(f"P5 burst TTFT: {line}")
    if s.programs() != pc0:
        fails.append(f"program cache grew after the capture: {pc0} -> {s.programs()}")
    assert not fails, "\n".join(fails)
