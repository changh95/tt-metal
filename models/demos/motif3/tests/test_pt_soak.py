# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving-order soaks of packed prefill (P5) and full-batch speculative verify (T64) on the real 53-layer model
(docs/p5_t64/P5_T64_DESIGN.md §2.3-§2.4, §6.2). This module holds **CP9-P** (work package I1, P5); the combined soak
**G-X** (P5 + T64 + the device sampler, work package I2) extends it with the 64-row verify trace.

Boot, in serving order (§2.4; the plugin's call order): weights from the TT cache (every checkpoint read refused) ->
the serving KV pool (4129 x 64, bfp8, KV-R) -> ``warmup_prefill`` with packed prefill on (every solo ``(path,
bucket)``, then every packed shape of ``cfg.packed_prefill_shapes()``: pk0 ``(T, S)``, pk1 ``(T, S)`` with both SWA tail
variants) -> ``enable_device_sampling`` -> ``warmup_decode`` (eager) -> the capture of the ONE T32-spec trace
(``("spec", "all_split")``, with the device sampler). F3N's rules hold by construction: ``ring_gather="safe"`` (B0),
everything compiled and allocated before the capture, every decode step ends in a blocking read, no re-capture.

**CP9-P** (``test_cp9p_packed_soak``; ``MOTIF3_SOAK_MINUTES``, default 10): after the capture, first a **sweep**
(:meth:`Soak.sweep`, P5 review finding 3): every warmed packed shape (56 at the production geometry) once, each as the
one packed pass of its own call, a decode round after each; then a randomized serving mix (:class:`Soak`). Prefill
calls of at most 32 rows and the vLLM budget of 8064 new tokens: bursts of cold
prompts of every segment-size class (pk0 at many ``(T, S)``), shared-prefix bursts (the prefix written in the same call
or cached earlier: pk1 ``shared``; prefixes of an odd number of blocks: the R-E1 same-step hits), resumed sessions at
one start (pk1 ``distinct``), long prompts in vLLM chunks (unaligned continuations; their tails pack), preemption
re-prefills (prompt + generated tokens) and mixed bursts. Between calls, traced decode steps of every kind the T32-spec
trace serves: ordinary steps (host logits), device-sampled steps (seeded and unseeded lanes), packed verify steps (MTP
drafts on idle lanes) and forced-overflow verify steps (more drafts than idle lanes: the second replay). Blocks are
reference-counted and recycled (the pool holds 4129). A fixed probe call (8 short prompts that pack + one 700-token
row) runs every 20 rounds on fresh blocks.

Asserted: the program cache is constant after the capture (nothing compiles: R2); every prefill and decode logits row
is finite and below 1e4; the probe's logits are bitwise identical every time; each sweep call ran exactly its shape and
every warmed packed shape ran; the mix ran pk0, pk1 shared and distinct passes, odd-block hits inside packed passes,
sampled, ordinary, verify and overflow steps; no solo fallback and no failed packed plan; at the end a decode replay
equals the eager step bitwise. Reported: the packed shapes reached, pass and step counts, prefill wall times.

``test_host_soak_driver_on_emulator`` runs the same driver on the host against the emulated paged cache of
``tests/test_resumed_prefill.py`` (packed prefill on; a fake decode that writes the emulated KV): every prefill read
of every call is checked there (prefix hashes, SWA tails, the unfilled rows of odd-block hits, no read of a write of
the same pass), so a driver bug (a hit on a freed or unwritten block) fails on the host, not on the device.

**G-X** (``test_gx_combined_soak``; work package I2; design §6.2; ``MOTIF3_GX_MINUTES``, default 30): the serving-order
boot with packed prefill on, ``MOTIF3_SPEC_VERIFY=auto`` and the device sampler: the T32-spec trace (with the sampler)
and the T64 trace (argmax only) captured once each. The same sweep and prefill mix as CP9-P (every warmed packed shape,
pk0 / pk1 shared and distinct, odd-block same-step hits, solo sp0 / sp1 chunks, MTP fills), and between the calls
(:class:`GXSoak`) every decode kind of the ``auto`` launch: T32 ordinary steps with host logits, T32 device-sampled
steps (seeded and unseeded lanes), T32 verify steps whose drafts fit idle lanes, T32 forced-overflow verify steps (a
verify that wants logits: the ``packed`` path check, two replays), T64 verify steps (every live lane drafting, more
drafts than idle lanes) and T64 steps with forced rejects (wrong drafts). Each step's routing is asserted (trace kind
and pass count). Asserted as CP9-P (sane logits everywhere, the probe call bitwise identical, the program cache
constant after the capture, every warmed packed shape run, no solo fallback) plus at the end: trace == eager bitwise on
both paths, and the same verify step on T64 equals it on T32 (packed + overflow pass) bitwise (option A'', F3N R1-R5:
two traces stay safe through the whole serving mix). A tracker variant (F3N R6) is the same test under
``TT_METAL_TRACE_ALLOC_TRACKING=1`` (the generator acknowledges the second trace's outputs after its capture).
``test_host_gx_driver_on_emulator`` runs :class:`GXSoak` on the emulator with ``auto``'s routing.

G-X measured (2026-10-03, 53 layers + MTP, ``logs/dev/20261003_231404_i2_gx.log``): boot 78 s to the two captures
(14 solo + 56 packed shapes warmed, program cache 1565); 33 minutes, 336 rounds, 410 prefill calls (4,697 rows): 652
solo, 339 pk0, 196 pk1 shared and 74 pk1 distinct passes, 580 odd-block hit rows inside packed passes, all 56 packed
shapes run, 0 solo fallbacks, 0 failed packed plans; decode: 311 T64 verify steps, 113 T64 steps with wrong drafts, 47
T32 verify steps, 96 T32 overflow verify steps (two replays), 121 T32 ordinary and 195 device-sampled steps (13 lanes
re-sampled on the host), each on its expected trace; the probe call bitwise identical 17 times; at the end both
traces replay == eager bitwise and one verify step on T64 == on T32 (packed + overflow pass) bitwise; the program cache
constant throughout. Tracker variant (``TT_METAL_TRACE_ALLOC_TRACKING=1 MOTIF3_GX_MINUTES=3``,
``logs/dev/20261003_234853_i2_gx_tracker.log``): the sweep and 34 rounds (51 T64 verify, 18 T64 reject, 15 T32 overflow
steps, ...) with no unsafe live buffer at any replay, the same end checks bitwise.

Run (device; one boot, ~4 min, + the soak)::

    scripts/devrun.sh -t 3600 -n cp9p -- python -m pytest models/demos/motif3/tests/test_pt_soak.py -k cp9p -s \
        -p no:cacheprovider --timeout=0
    scripts/devrun.sh -t 5400 -n gx -- env OMP_WAIT_POLICY=PASSIVE python -m pytest \
        models/demos/motif3/tests/test_pt_soak.py -k gx_combined -s -p no:cacheprovider --timeout=0
"""

from __future__ import annotations

import os
import random
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pytest
import torch

from models.demos.motif3.tt import generator_api as api
from models.demos.motif3.tt import prefill_plan as PP

SOAK_MINUTES = float(os.environ.get("MOTIF3_SOAK_MINUTES", "10"))
SOAK_LAYERS = int(os.environ.get("MOTIF3_SOAK_LAYERS", "53"))
MAX_LEN = 32768
BS = 64
NUM_BLOCKS = api.expected_num_blocks()  # 4129: the serving pool
WIDTH = min(api.cdiv(MAX_LEN, BS), NUM_BLOCKS)  # 512
DECODE_ROOM = 160  # positions a request may decode before it is retired
LOGIT_ABS_MAX = 1e4
PROBE_EVERY = 20
COVERAGE = ("pass_pk0", "pass_pk1_shared", "pass_pk1_distinct", "pass_solo", "odd_hit_rows_packed", "step_sampled",
            "step_ordinary", "step_verify", "step_overflow")  # fmt: skip
# G-X (I2): CP9-P's coverage with the decode kinds of a spec_verify="auto" launch
GX_MINUTES = float(os.environ.get("MOTIF3_GX_MINUTES", "30"))
GX_MIN_CALLS = 300  # prefill calls over a >= 30 min soak (design §6.2)
GX_COVERAGE = COVERAGE[:-2] + ("step_t32_verify", "step_t32_overflow", "step_t64_verify", "step_t64_rejects")
# The sweep (Soak.sweep): real rows per segment of S rows (S is the smallest segment size >= them), and the cached
# prefix the pk1 "shared" calls hit / the history each pk1 "distinct" row resumes after
SWEEP_ROWS = {64: 40, 128: 100, 256: 200, 512: 400, 1024: 900}
SWEEP_PREFIX = 2048
SWEEP_HISTORY = 128


def log(msg: str) -> None:
    print(f"[pt-soak {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def assert_sane(logits: torch.Tensor, what: str) -> None:
    """Finite and ``|logit| <= 1e4`` (device garbage reads as finite ~1e18; tests/test_resumed_prefill.py)."""
    lg = logits.float().reshape(-1, logits.shape[-1])
    bad = ~(torch.isfinite(lg).all(-1) & (lg.abs().amax(-1) <= LOGIT_ABS_MAX))
    assert not bool(bad.any()), f"{what}: non-finite / out-of-range logits in rows {torch.nonzero(bad).flatten()[:8]}"


class BlockPool:
    """vLLM-like block ids ``1 .. n - 1`` (0 = the null block), reference-counted: a request holds its blocks, a hit
    shares the cached prefix's; a block returns to the free queue when its last holder releases it (``on_free``)."""

    def __init__(self, n: int, seed: int, on_free: Optional[Callable[[int], None]] = None):
        ids = list(range(1, int(n)))
        random.Random(seed).shuffle(ids)
        self.free = ids
        self.refs: Counter = Counter()
        self.on_free = on_free

    def take(self, k: int) -> List[int]:
        if k > len(self.free):
            raise RuntimeError(f"block pool exhausted: {k} wanted, {len(self.free)} free")
        out, self.free = self.free[:k], self.free[k:]
        for b in out:
            self.refs[b] += 1
        return out

    def share(self, ids: List[int]) -> List[int]:
        for b in ids:
            assert self.refs[b] > 0, f"sharing freed block {b}"
            self.refs[b] += 1
        return list(ids)

    def release(self, ids: List[int]) -> None:
        for b in ids:
            self.refs[b] -= 1
            assert self.refs[b] >= 0
            if self.refs[b] == 0:
                del self.refs[b]
                self.free.append(b)
                if self.on_free is not None:
                    self.on_free(b)


@dataclass
class Req:
    """A request: every token so far, its blocks (capacity for ``len + DECODE_ROOM``), positions computed."""

    seq: List[int]
    blocks: List[int]
    computed: int = 0
    prompt_end: int = 0  # the prefill target (the prompt length when admitted)
    lane: int = -1
    sampling: Optional[Tuple[float, float, int, Optional[int]]] = None  # (temperature, top_p, top_k, seed)
    draft: Optional[int] = None  # the MTP draft for the next step (m0 / m1 of the previous spec step)

    @property
    def cap(self) -> int:
        return len(self.blocks) * BS


def page_row(blocks: List[int], upto: int) -> torch.Tensor:
    pt = torch.zeros(WIDTH, dtype=torch.int32)
    n = api.cdiv(upto, BS)
    assert n <= len(blocks), (n, len(blocks))
    pt[:n] = torch.tensor(blocks[:n], dtype=torch.int32)
    return pt


def request(lane: int, r: Req, end: int, start: int) -> api.PrefillRequest:
    return api.PrefillRequest(
        lane=lane, tokens=torch.tensor(r.seq[:end], dtype=torch.int32), page_table=page_row(r.blocks, end), start=start
    )


def soak_tokens() -> List[int]:
    """Real token ids: the C2 prompts (goldens/c2/prompts.json), concatenated."""
    from models.demos.motif3.reference import golden_stream as gs
    from models.demos.motif3.tests.test_resumed_prefill import GOLD_BF16

    return [t for p in gs.load_prompt_set(GOLD_BF16 / "prompts.json") for t in p.ids]


class Soak:
    """The serving-mix driver of CP9-P (module docstring): ``round_prefill`` (one prefill call), ``round_decode`` (one
    decode step), ``probe``. ``programs``: the program-cache size (``mesh.num_program_cache_entries``)."""

    COVERAGE = COVERAGE  # what :meth:`check` requires the mix to have run

    def __init__(self, gen, pool, *, seed: int, src: List[int], programs: Callable[[], int], on_free=None):
        self.gen, self.pool, self.programs = gen, pool, programs
        self.rng = random.Random(seed)
        self.blocks = BlockPool(int(pool.num_blocks), seed, on_free)
        self.src = list(src)
        self.live: Dict[int, Req] = {}
        self.free_lanes = list(range(api.NUM_LANES))
        self.done: "OrderedDict[int, Req]" = OrderedDict()  # finished requests (prefix-hit and resume sources), LRU
        self.pending: List[Req] = []  # vLLM-chunked prompts in progress
        self.systems: List[Req] = []  # cached shared prefixes (system prompts; one extra block reference each)
        self.st: Counter = Counter()
        self.shapes: Counter = Counter()
        self.walls: List[Tuple[str, float]] = []
        self.probe_ref: Optional[torch.Tensor] = None
        self.swept: Optional[List[Tuple]] = None  # the shapes :meth:`sweep` ran (None: no sweep)
        self._uid = 0

    # ---- helpers --------------------------------------------------------------------------------------------------
    def text(self, n: int) -> List[int]:
        i = self.rng.randrange(len(self.src))
        return [self.src[(i + j) % len(self.src)] for j in range(n)]

    def reserve(self, k: int) -> None:
        """Free blocks for ``k`` more: evict finished requests (LRU), then cached system prompts, then live lanes
        (a block returns once no other request holds it)."""
        while len(self.blocks.free) < k:
            if self.done:
                self.blocks.release(self.done.popitem(last=False)[1].blocks)
            elif self.systems:
                self.blocks.release(self.systems.pop(0).blocks)
            elif self.live:
                lane = next(iter(self.live))
                self.retire(self.live.pop(lane), keep=False)
                self.free_lanes.append(lane)
            else:
                raise RuntimeError(f"block pool exhausted: {k} wanted, {len(self.blocks.free)} free")

    def new_req(self, seq: List[int], prefix: Optional[Req] = None, k: int = 0) -> Req:
        """A request over ``seq``; with ``prefix``, its first ``k`` blocks are shared (a prefix hit: computed =
        ``k * BS``). The shared blocks are held before any eviction for the request's own blocks."""
        need = api.cdiv(len(seq) + DECODE_ROOM, BS)
        shared = self.blocks.share(prefix.blocks[:k]) if k else []
        self.reserve(need - k)
        own = self.blocks.take(need - k)
        return Req(seq=list(seq), blocks=shared + own, computed=k * BS, prompt_end=len(seq))

    def retire(self, r: Req, keep: bool = True) -> None:
        """A finished request: kept for later hits / resumes (LRU of 12), else its blocks are released."""
        if keep:
            self._uid += 1
            self.done[self._uid] = r
            while len(self.done) > 12:
                _, old = self.done.popitem(last=False)
                self.blocks.release(old.blocks)
        else:
            self.blocks.release(r.blocks)

    def ensure_lanes(self, n: int) -> None:
        while len(self.free_lanes) < n and self.live:
            lane = self.rng.choice(list(self.live))
            self.retire(self.live.pop(lane))
            self.free_lanes.append(lane)

    # ---- prefill calls --------------------------------------------------------------------------------------------
    def call(self, rows: List[Tuple[Req, int]], kind: str) -> torch.Tensor:
        """One ``prefill_forward_batch`` call: ``rows`` = (request, chunk end). Requests whose prompt completes become
        decoding lanes (enough lanes were freed by the caller); the others stay pending."""
        gen = self.gen
        reqs = [request(i, r, end, r.computed) for i, (r, end) in enumerate(rows)]
        t0 = time.time()
        out = gen.prefill_forward_batch(reqs, kv_cache=self.pool)
        self.walls.append((kind, time.time() - t0))
        assert_sane(out, f"prefill {kind}")
        b = gen.last_prefill
        for p in b.passes:
            self.shapes[p.shape] += 1
            self.st[f"pass_{p.kind}" + (f"_{p.tails}" if p.tails else "")] += 1
        odd = {i for i, j in enumerate(b.jobs) if j.plan.has_sp1 and j.plan.c0 < j.plan.w0}
        self.st["odd_hit_rows_packed"] += sum(g.row in odd for p in b.packed_passes for g in p.segments)
        self.st["fallbacks"] += len(b.fallbacks)
        self.st["calls"] += 1
        self.st["rows"] += len(rows)
        for (r, end), lg in zip(rows, out):
            r.computed = end
            if end < r.prompt_end:
                continue  # a vLLM chunk: the rest in a later call
            if r in self.pending:
                self.pending.remove(r)
            r.seq = r.seq[: r.prompt_end] + [int(lg.float().argmax())]
            lane = self.free_lanes.pop()
            r.lane, r.draft, r.sampling = lane, None, None
            if self.rng.random() < 0.3:  # a sampled request (seeded half the time)
                seed = self.rng.randrange(1 << 30) if self.rng.random() < 0.5 else None
                r.sampling = (self.rng.choice([0.6, 1.0]), self.rng.choice([0.9, 1.0]), self.rng.choice([0, 40]), seed)
            self.live[lane] = r
        return out

    @staticmethod
    def budget_rows(cands: List[Tuple[Req, int]], budget: int = 8064) -> List[Tuple[Req, int]]:
        """At most 32 rows and ``budget`` new tokens, vLLM-like: the row that crosses the budget is chunked."""
        rows, left = [], budget
        for r, end in cands:
            if left <= 0 or len(rows) == 32:
                break
            e = min(end, r.computed + left)
            rows.append((r, e))
            left -= e - r.computed
        return rows

    def round_prefill(self) -> None:
        rng = self.rng
        k = rng.random()
        cands: List[Tuple[Req, Optional[int]]] = []
        held: List[List[int]] = []  # prefixes pinned while this round's requests are built
        kind = "burst"
        conts: List[Tuple[Req, Optional[int]]] = [(r, None) for r in self.pending[:2]]  # vLLM-chunked continuations
        if k < 0.30:  # cold burst at one segment-size class
            lo, hi = rng.choice([(1, 64), (65, 128), (129, 256), (257, 512), (513, 1024), (20, 400)])
            for _ in range(rng.randint(2, 30)):
                cands.append((self.new_req(self.text(rng.randint(lo, hi))), None))
        elif k < 0.55:  # shared prefix: written in this call (same-step hits) or cached earlier
            kind = "shared"
            n_pre = rng.choice([128, 192, 1088, 2048, 2112, 2200, 3000])
            if not self.systems or rng.random() < 0.5:  # written by this call's first row: the others hit it (D7)
                base = self.new_req(self.text(n_pre + rng.randint(1, 60)))
                cands.append((base, None))
                self.systems.append(base)
                self.blocks.share(base.blocks)  # the prefix cache's own reference (the request keeps its own)
                if len(self.systems) > 4:
                    self.blocks.release(self.systems.pop(0).blocks)
            else:
                base = rng.choice(self.systems)
            hit = min(base.prompt_end - 1, n_pre) // BS  # full blocks of the prefix (vLLM: capped at len - 1)
            held.append(self.blocks.share(base.blocks[:hit]))
            for _ in range(rng.randint(2, 30)):
                seq = base.seq[: hit * BS] + self.text(rng.randint(1, 150))
                cands.append((self.new_req(seq, base, hit), None))
        elif k < 0.70 and len(self.done) >= 2:  # resumed sessions at one start (pk1 distinct) / preemption
            kind = "resume"
            olds = [r for r in self.done.values() if len(r.seq) > 2 * BS + 1]
            pick = rng.sample(olds, min(len(olds), rng.randint(2, 12)))
            if pick:
                kb = min((len(r.seq) - 1) // BS for r in pick) // 2 * 2  # an even block count: one common start
                held += [self.blocks.share(r.blocks[:kb]) for r in pick]
                for r in pick:
                    seq = r.seq + (self.text(rng.randint(1, 120)) if rng.random() < 0.7 else [])
                    cands.append((self.new_req(seq, r, kb), None))
        elif k < 0.82:  # a long prompt in vLLM chunks (+ short rows beside it)
            kind = "long"
            r = self.new_req(self.text(rng.randint(2500, 9000)))
            self.pending.append(r)
            cands.append((r, None))
            for _ in range(rng.randint(0, 6)):
                cands.append((self.new_req(self.text(rng.randint(1, 300))), None))
        else:  # a mixed burst
            kind = "mixed"
            for _ in range(rng.randint(3, 20)):
                n = rng.choice([rng.randint(1, 64), rng.randint(200, 900), rng.randint(1000, 1800)])
                cands.append((self.new_req(self.text(n)), None))
        for h in held:
            self.blocks.release(h)
        # continuations first (vLLM schedules running requests first), except behind a shared prefix written in this
        # call: its writer must not be cut by the budget (its readers hit the blocks it computes in this step)
        cands = cands + conts if kind in ("shared", "resume") else conts + cands
        cands = [(r, r.prompt_end if e is None else e) for r, e in cands]
        rows = self.budget_rows(cands)
        for r, _ in cands[len(rows) :]:  # cut by the budget and not running yet: never admitted
            if r not in self.pending:
                self.blocks.release(r.blocks)
        rows = [(r, e) for r, e in rows if e > r.computed]
        for r, e in rows:
            if e < r.prompt_end and r not in self.pending:
                self.pending.append(r)
        if rows:
            self.ensure_lanes(sum(e == r.prompt_end for r, e in rows))
            self.call(rows, kind)

    def probe(self) -> None:
        """The fixed probe call on fresh blocks: its logits must be bitwise identical every time."""
        rng = random.Random(4242)
        src = self.src
        seqs = [src[i * 97 : i * 97 + rng.randint(40, 60)] for i in range(8)] + [src[1000:1700]]
        reqs, held = [], []
        for i, s in enumerate(seqs):
            self.reserve(api.cdiv(len(s), BS))
            b = self.blocks.take(api.cdiv(len(s), BS))
            held.append(b)
            reqs.append(api.PrefillRequest(lane=i, tokens=torch.tensor(s, dtype=torch.int32),
                                           page_table=page_row(b, len(s)), start=0))  # fmt: skip
        out = self.gen.prefill_forward_batch(reqs, kv_cache=self.pool)
        assert_sane(out, "probe")
        assert self.gen.last_prefill.packed_passes, self.gen.last_prefill.describe()
        for b in held:
            self.blocks.release(b)
        if self.probe_ref is None:
            self.probe_ref = out.clone()
        else:
            same = torch.equal(out, self.probe_ref)
            self.st["probe_same" if same else "probe_diff"] += 1
            assert same, f"probe call {self.st['probes']}: logits differ from the first probe's"
        self.st["probes"] += 1

    # ---- the every-shape sweep (P5 review, finding 3) --------------------------------------------------------------
    def sweep_rows(self, shape: Tuple, prefix: Optional[Req]) -> Tuple[List[Tuple[Req, int]], List[Tuple[Req, int]]]:
        """``(prep rows, target rows)``: a call pair whose target call plans to exactly ONE packed pass of ``shape``
        (``B = T / S`` segments of ``SWEEP_ROWS[S]`` real rows). pk0: ``B`` cold prompts. pk1 ``shared``: ``B`` rows
        hitting the cached ``SWEEP_PREFIX``-token ``prefix`` (its first ``SWEEP_PREFIX / BS`` blocks), new tokens after
        it. pk1 ``distinct``: ``B`` rows with histories of their own, ``SWEEP_HISTORY`` tokens that the prep call
        computes (one pk0 pass), resumed there. At most 7,200 new tokens and 32 rows per call (the vLLM budget)."""
        kind, T, S = shape[0], int(shape[1]), int(shape[2])
        B, r = T // S, SWEEP_ROWS[S]
        if kind == PP.PK0:
            return [], [(self.new_req(self.text(r)), r) for _ in range(B)]
        if shape[3] == PP.SHARED_TAILS:
            k = SWEEP_PREFIX // BS
            rows = [self.new_req(prefix.seq[:SWEEP_PREFIX] + self.text(r), prefix, k) for _ in range(B)]
            return [], [(q, q.prompt_end) for q in rows]
        hist = [self.new_req(self.text(SWEEP_HISTORY + r)) for _ in range(B)]
        return [(h, SWEEP_HISTORY) for h in hist], [(h, h.prompt_end) for h in hist]

    def sweep(self) -> List[Tuple]:
        """Every packed shape the generator warmed (``gen.packed_shapes()``: pk0 ``(T, S)``, pk1 ``(T, S)`` x both tail
        variants), once, each as the one packed pass of a call (:meth:`sweep_rows`), each call followed by one decode
        round: the random mix alone reached 35 of the 56 in 10 minutes (P5 review finding 3). Asserted per call: its
        passes are exactly that shape (no solo fallback, no other cut). Returns the shapes run, in order."""
        shapes = [tuple(s) for s in self.gen.packed_shapes()]
        prefix, held = None, []
        if any(s[0] == PP.PK1 and s[3] == PP.SHARED_TAILS for s in shapes):  # the cached prefix of the shared calls
            prefix = self.new_req(self.text(SWEEP_PREFIX + 60))
            self.ensure_lanes(1)
            self.call([(prefix, prefix.prompt_end)], "sweep")
            held = self.blocks.share(prefix.blocks[: SWEEP_PREFIX // BS])  # held until the sweep ends
        done = []
        for shape in shapes:
            prep, rows = self.sweep_rows(shape, prefix)
            if prep:
                self.call(prep, "sweep")  # incomplete prompts: no lane
            self.ensure_lanes(len(rows))
            self.call(rows, "sweep")
            b = self.gen.last_prefill
            assert [p.shape for p in b.passes] == [shape], f"sweep {shape}: the call planned {b.describe()}"
            done.append(shape)
            self.st["sweep_calls"] += 1
            self.round_decode()
        self.blocks.release(held)
        self.swept = list(done)
        return done

    # ---- decode steps ---------------------------------------------------------------------------------------------
    def batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
        pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
        pt = torch.zeros(api.NUM_LANES, WIDTH, dtype=torch.int32)
        for lane, r in self.live.items():
            p = len(r.seq) - 1
            tokens[lane], pos[lane], pt[lane] = r.seq[-1], p, page_row(r.blocks, p + 2)
        return tokens, pos, pt

    def round_decode(self) -> None:
        gen, rng = self.gen, self.rng
        full = [l for l, r in self.live.items() if len(r.seq) + 3 > r.cap]
        if rng.random() < 0.15:  # requests finish (EOS): the occupancy varies, so drafts fit idle lanes sometimes
            full += rng.sample(list(self.live), min(len(self.live), rng.randint(1, 8)))
        for lane in dict.fromkeys(full):
            self.retire(self.live.pop(lane))
            self.free_lanes.append(lane)
        if not self.live:
            return
        tokens, pos, pt = self.batch()
        lanes = list(self.live)
        k = rng.random()
        if k < 0.3:  # device-sampled ordinary step (greedy lanes take temperature 0)
            temp, top_p, top_k, seeds = [0.0] * 32, [1.0] * 32, [0] * 32, [None] * 32
            for lane, r in self.live.items():
                if r.sampling is not None:
                    temp[lane], top_p[lane], top_k[lane], seeds[lane] = r.sampling
            b = api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt)
            res = gen.decode_forward_sampled(b, (temp, top_p, top_k, seeds), kv_cache=self.pool, enable_trace=True)
            for lane in lanes:
                r = self.live[lane]
                r.seq.append(int(res.tokens[lane]))
                r.draft = None
            self.st["step_sampled"] += 1
            return
        drafts = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
        verify = k < 0.75
        overflow = verify and k < 0.45 and len(lanes) > 16
        if verify:
            idle = api.NUM_LANES - len(lanes)
            if overflow:
                pick = lanes  # every live lane drafts: more drafts than idle lanes
            elif idle and rng.random() < 0.6:
                pick = rng.sample(lanes, min(len(lanes), idle))  # the bridge's idle-lane budget: every draft fits
            else:
                pick = [l for l in lanes if rng.random() < 0.6]
            for lane in pick:
                r = self.live[lane]
                drafts[lane] = r.draft if r.draft is not None else rng.randrange(1000)
        batch = api.SpecDecodeBatch(tokens=tokens, positions=pos, draft_tokens=drafts, page_table=pt)
        res = gen.decode_forward_spec(batch, kv_cache=self.pool, enable_trace=True, want_logits=not batch.is_verify)
        if res.logits is not None:
            assert_sane(res.logits[lanes], "decode logits")
        for lane in lanes:
            r = self.live[lane]
            a0, a1, d = int(res.argmax[lane, 0]), int(res.argmax[lane, 1]), int(drafts[lane])
            if d >= 0 and d == a0:  # the plugin's greedy accept walk: (d, a1)
                r.seq += [a0, a1]
                r.draft = int(res.mtp_argmax[lane, 1])
                self.st["accepted"] += 1
            else:
                r.seq.append(a0)
                r.draft = int(res.mtp_argmax[lane, 0])
        n_pass = len(gen.last_spec.passes)
        self.st["step_overflow" if n_pass > 1 else ("step_verify" if batch.is_verify else "step_ordinary")] += 1
        self.st["drafts"] += int((drafts >= 0).sum())

    # ---- the soak -------------------------------------------------------------------------------------------------
    def run(
        self, *, seconds: Optional[float] = None, rounds: Optional[int] = None, log_every: int = 10, sweep: bool = True
    ) -> int:
        """The probe, then (``sweep``) :meth:`sweep` over every warmed packed shape, then rounds of (one prefill call,
        1-4 decode steps), the probe every ``PROBE_EVERY`` rounds, until ``seconds`` (counted after the sweep) or
        ``rounds``; the program cache must stay constant throughout. Returns the number of rounds."""
        pc0 = self.programs()
        self.probe()
        if sweep:
            t0 = time.time()
            self.sweep()
            pcs = self.programs()
            log(f"sweep: {len(self.swept)} packed shapes, each one call, in {time.time() - t0:.1f} s; program cache "
                f"{pcs}")  # fmt: skip
            assert pcs == pc0, f"program cache grew in the sweep: {pc0} -> {pcs}"
        t_end = None if seconds is None else time.time() + seconds
        n = 0
        while (t_end is None or time.time() < t_end) and (rounds is None or n < rounds):
            self.round_prefill()
            for _ in range(self.rng.randint(1, 4)):
                self.round_decode()
            n += 1
            if n % PROBE_EVERY == 0:
                self.probe()
            if n % log_every == 0:
                pcs = self.programs()
                log(f"soak round {n}: {dict(self.st)}; live {len(self.live)}, free blocks {len(self.blocks.free)}; "
                    f"program cache {pcs}")  # fmt: skip
                assert pcs == pc0, f"program cache grew after the capture: {pc0} -> {pcs}"
        self.probe()
        assert self.programs() == pc0, f"program cache grew after the capture: {pc0} -> {self.programs()}"
        return n

    def check(self) -> None:
        """Coverage of the mix (with the sweep: every warmed packed shape), no fallback, no failed packed plan, the
        probes identical (module docstring)."""
        missing = [k for k in self.COVERAGE if self.st[k] == 0]
        assert not missing, f"the mix never ran {missing}: {dict(self.st)}"
        if self.swept is not None:
            unreached = [s for s in self.gen.packed_shapes() if s not in self.shapes]
            assert not unreached, f"warmed packed shapes never run after the capture: {unreached}"
        assert self.st["fallbacks"] == 0, f"{self.st['fallbacks']} chunks fell back to solo (an unwarmed shape)"
        assert self.gen.stats.get("packed_plan_errors", 0) == 0, "a packed plan failed its checks (ran per row)"
        assert self.st["probe_diff"] == 0 and self.st["probes"] >= 2, dict(self.st)

    def report(self, what: str) -> None:
        warmed = set(self.gen.packed_shapes())
        reached = sorted(k for k in self.shapes if k in warmed)
        walls: Dict[str, List[float]] = {}
        for kind, sec in self.walls:
            walls.setdefault(kind, []).append(sec)
        log(f"{what}: {dict(self.st)}; packed shapes reached {len(reached)} of {len(warmed)}: {reached}")
        parts = [f"{k} {sorted(v)[len(v) // 2]:.2f} / {max(v):.2f} ({len(v)})" for k, v in sorted(walls.items())]
        log(f"{what} prefill wall (median / max s, calls): " + ", ".join(parts))


class GXSoak(Soak):
    """G-X's driver (module docstring): CP9-P's prefill mix and checks, and every decode kind of a ``spec_verify=
    "auto"`` launch between the calls, with each step's routing asserted (``gen.last_verify_kind`` and the T32 pass
    count): ``sampled`` (T32 + the sampler), ``ordinary`` (T32, host logits), ``t32_verify`` (drafts that fit the idle
    lanes: one T32 replay), ``t32_overflow`` (more drafts than idle lanes in a verify that wants logits: T32 + the
    overflow pass, the ``packed`` path), ``t64_verify`` (more drafts than idle lanes: one T64 replay), ``t64_rejects``
    (the same with wrong drafts)."""

    COVERAGE = GX_COVERAGE
    KINDS = ("sampled", "ordinary", "t32_verify", "t32_overflow", "t64_verify", "t64_rejects")
    WEIGHTS = (22, 13, 15, 10, 28, 12)
    EXPECT = {  # kind -> (trace kind, T32 passes)
        "ordinary": ("spec", 1),
        "t32_verify": ("spec", 1),
        "t32_overflow": ("spec", 2),
        "t64_verify": ("wide", 1),
        "t64_rejects": ("wide", 1),
    }

    def round_decode(self) -> None:
        gen, rng = self.gen, self.rng
        full = [l for l, r in self.live.items() if len(r.seq) + 3 > r.cap]
        if rng.random() < 0.15:  # requests finish (EOS): the occupancy varies
            full += rng.sample(list(self.live), min(len(self.live), rng.randint(1, 8)))
        for lane in dict.fromkeys(full):
            self.retire(self.live.pop(lane))
            self.free_lanes.append(lane)
        if not self.live:
            return
        tokens, pos, pt = self.batch()
        lanes = list(self.live)
        idle = api.NUM_LANES - len(lanes)
        kind = rng.choices(self.KINDS, weights=self.WEIGHTS)[0]
        if kind in ("t32_overflow", "t64_verify", "t64_rejects") and len(lanes) <= idle:
            kind = "t32_verify"  # every draft fits an idle lane: T32 (auto never takes T64 then)
        if kind == "t32_verify" and idle == 0:
            kind = "t64_verify"
        if kind == "sampled":  # device-sampled ordinary step (greedy lanes take temperature 0): the T32 trace
            temp, top_p, top_k, seeds = [0.0] * 32, [1.0] * 32, [0] * 32, [None] * 32
            for lane, r in self.live.items():
                if r.sampling is not None:
                    temp[lane], top_p[lane], top_k[lane], seeds[lane] = r.sampling
            b = api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt)
            res = gen.decode_forward_sampled(b, (temp, top_p, top_k, seeds), kv_cache=self.pool, enable_trace=True)
            assert gen.last_verify_kind == "spec", "a sampled step ran on the T64 trace"
            for lane in lanes:
                r = self.live[lane]
                r.seq.append(int(res.tokens[lane]))
                r.draft = None
            self.st["step_sampled"] += 1
            return
        drafts = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
        if kind != "ordinary":
            pick = rng.sample(lanes, min(len(lanes), idle)) if kind == "t32_verify" else lanes
            V = int(gen.vocab_size)
            for lane in pick:
                r = self.live[lane]
                d = r.draft if r.draft is not None else rng.randrange(1000)
                if kind == "t64_rejects":  # a wrong draft (any token but the MTP's guess)
                    d = (d + 1 + rng.randrange(V - 1)) % V
                drafts[lane] = d
        batch = api.SpecDecodeBatch(tokens=tokens, positions=pos, draft_tokens=drafts, page_table=pt)
        want = kind in ("ordinary", "t32_overflow")
        res = gen.decode_forward_spec(batch, kv_cache=self.pool, enable_trace=True, want_logits=want)
        ran = gen.last_verify_kind
        n_pass = len(gen.last_spec.passes) if ran == "spec" else 1
        assert (ran, n_pass) == self.EXPECT[kind], f"{kind} ({len(lanes)} live): ran on {ran} in {n_pass} pass(es)"
        if res.logits is not None:
            assert_sane(res.logits[lanes], f"decode logits ({kind})")
        for lane in lanes:
            r = self.live[lane]
            a0, a1, d = int(res.argmax[lane, 0]), int(res.argmax[lane, 1]), int(drafts[lane])
            if d >= 0 and d == a0:  # the plugin's greedy accept walk: (d, a1)
                r.seq += [a0, a1]
                r.draft = int(res.mtp_argmax[lane, 1])
                self.st["accepted"] += 1
            else:
                r.seq.append(a0)
                r.draft = int(res.mtp_argmax[lane, 0])
                self.st["rejected"] += int(d >= 0)
        self.st[f"step_{kind}"] += 1
        self.st["drafts"] += int((drafts >= 0).sum())

    def fill_lanes(self, n: int) -> None:
        """Short cold prompts until ``n`` lanes decode (one prefill call each)."""
        while len(self.live) < n and self.free_lanes:
            r = self.new_req(self.text(self.rng.randint(20, 200)))
            self.call([(r, r.prompt_end)], "fill")

    def final_checks(self) -> Dict[str, bool]:
        """The end of G-X: at least 24 live lanes; an ordinary step traced == eager (T32, host logits); a verify step
        with every live lane drafting traced (auto: T64) == eager T64 == traced T32 (packed + overflow pass), a / m
        bitwise on every live lane."""
        gen, pool = self.gen, self.pool
        self.fill_lanes(24)
        tokens, pos, pt = self.batch()
        act = list(self.live)
        b = api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt)
        traced = gen.decode_forward(b, kv_cache=pool, enable_trace=True).clone()
        eager = gen.decode_forward(b, kv_cache=pool, enable_trace=False)
        out = {"ordinary trace == eager": bool(torch.equal(traced[act], eager[act]))}
        drafts = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
        for lane in act:
            r = self.live[lane]
            drafts[lane] = r.draft if r.draft is not None else 7
        v = api.SpecDecodeBatch(tokens=tokens, positions=pos, draft_tokens=drafts, page_table=pt)
        rt = gen.decode_forward_spec(v, kv_cache=pool, enable_trace=True, want_logits=False)
        assert gen.last_verify_kind == "wide", "a verify step with more drafts than idle lanes ran on T32"
        wide, spec = ("wide", "all_split"), ("spec", "all_split")
        re_ = gen.decode_forward_spec(v, kv_cache=pool, enable_trace=False, want_logits=False, path=wide)
        r32 = gen.decode_forward_spec(v, kv_cache=pool, enable_trace=True, want_logits=False, path=spec)
        assert len(gen.last_spec.passes) == 2, "the T32 run of the verify step should overflow its idle lanes"

        def same(x, y) -> bool:
            return bool(torch.equal(x.argmax[act], y.argmax[act])) and bool(torch.equal(x.mtp_argmax[act],
                                                                                      y.mtp_argmax[act]))  # fmt: skip

        out["T64 trace == eager"] = same(rt, re_)
        out["T64 == T32 (packed + overflow)"] = same(rt, r32)
        return out


# ======================================================================================================================
# device: CP9-P
# ======================================================================================================================
# One device session at a time in this module (CP9-P's packed launch, G-X's auto launch): a boot closes every other
# open session first (each fixture's own teardown is then a no-op).
_OPEN_SESSIONS: Dict[str, Callable[[], None]] = {}


def _boot(name: str, spec_verify: str):
    """The serving-order boot (module docstring) of a speculating launch with packed prefill and the device sampler,
    ``spec_verify`` "packed" (CP9-P: the one T32-spec trace) or "auto" (G-X: the T32-spec trace with the sampler, then
    the T64 trace); yields ``(mesh, gen, pool)``."""
    from models.demos.motif3.tests.test_resumed_prefill import RefuseReads
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR, close_motif_mesh, open_motif_mesh

    for other in [k for k in _OPEN_SESSIONS if k != name]:
        log(f"closing the {other!r} device session for the {name!r} one")
        _OPEN_SESSIONS.pop(other)()
    mesh = open_motif_mesh()
    holder: Dict[str, object] = {}

    def close() -> None:
        if holder.get("closed"):
            return
        holder["closed"] = True
        try:
            if holder.get("gen") is not None:
                holder["gen"].close()
        finally:
            close_motif_mesh(mesh)

    _OPEN_SESSIONS[name] = close
    try:
        log_fabric(mesh, f"pt_soak_{name}")
        A = api.DEFAULT_PREFILL_ALIGNMENT
        budget = PP.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, A)
        settings = api.GeneratorSettings(
            max_batch_size=api.NUM_LANES, max_seq_len=MAX_LEN, num_layers=SOAK_LAYERS, kv_cache_dtype="bfp8",
            weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=BS, weights_source="TT cache only (test)",
            chunked_prefill=True, prefix_caching=True, max_num_batched_tokens=budget,
            long_prefill_token_threshold=budget, spec_tokens=1, packed_prefill=True, spec_verify=spec_verify,
        )  # fmt: skip
        guard = RefuseReads(DEFAULT_WEIGHTS_DIR)
        t0 = time.time()
        gen = holder["gen"] = MotifGenerator.create(hf_config=None, mesh_device=mesh, settings=settings, source=guard)
        assert not gen.model.cache_misses and not guard.refused, "not a TT-cache-only boot"
        assert gen.packed_prefill and gen.serving_path == ("spec", "all_split") and gen.cfg.ring_gather == "safe"
        want_paths = [("spec", "all_split")] + ([("wide", "all_split")] if spec_verify == "auto" else [])
        assert gen.serving_paths == want_paths, gen.serving_paths
        pool = gen.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BS, num_layers=SOAK_LAYERS)
        assert gen.prefill_alignment == A
        t1 = time.time()
        gen.warmup_prefill(kv_cache=pool, enable_trace=False)
        t2 = time.time()
        gen.enable_device_sampling()
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=WIDTH)
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=WIDTH)
        assert set(gen.required_prefill_shapes()) <= gen.warmed_shapes
        assert gen._paths[("spec", "all_split")].so is not None, "the T32-spec trace holds the device sampler"
        if spec_verify == "auto":
            assert gen._paths[("wide", "all_split")].so is None, "auto's T64 trace holds no sampler"
        log(
            f"boot ({spec_verify}): weights {t1 - t0:.1f} s; warmup prefill {t2 - t1:.1f} s "
            f"({len(gen.prefill_shapes())} solo + {len(gen.packed_shapes())} packed shapes; packed "
            f"{gen.timings.get('warmup_packed_s', 0):.1f} s); "
            f"decode eager {gen.timings.get('warmup_decode_eager_s', 0):.1f} s, capture "
            f"{gen.timings.get('capture_decode_s', 0):.1f} s ({len(gen.serving_paths)} traces); program cache "
            f"{mesh.num_program_cache_entries()}"
        )
        yield mesh, gen, pool
    finally:
        _OPEN_SESSIONS.pop(name, None)
        close()


@pytest.fixture(scope="module")
def boot():
    yield from _boot("cp9p", "packed")


@pytest.fixture(scope="module")
def boot_gx():
    yield from _boot("gx", "auto")


@pytest.mark.timeout(int(SOAK_MINUTES * 60) + 1800)
def test_cp9p_packed_soak(boot):
    """CP9-P (module docstring): the randomized serving mix for ``MOTIF3_SOAK_MINUTES`` after the capture."""
    mesh, gen, pool = boot
    s = Soak(gen, pool, seed=int(os.environ.get("MOTIF3_SOAK_SEED", "9")), src=soak_tokens(),
             programs=lambda: int(mesh.num_program_cache_entries()))  # fmt: skip
    pc0 = s.programs()
    n = s.run(seconds=SOAK_MINUTES * 60)
    while len(s.live) < 4:
        s.round_prefill()
    tokens, pos, pt = s.batch()  # a decode replay equals the eager step bitwise (the KV writes are idempotent)
    b = api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt)
    traced = gen.decode_forward(b, kv_cache=pool, enable_trace=True).clone()
    eager = gen.decode_forward(b, kv_cache=pool, enable_trace=False)
    act = list(s.live)
    assert torch.equal(traced[act], eager[act]), "decode trace replay != eager step"
    assert s.programs() == pc0, f"program cache grew: {pc0} -> {s.programs()}"
    s.report(f"CP9-P ({n} rounds, {SOAK_MINUTES:g} min)")
    log(f"CP9-P generator stats: { {k: v for k, v in gen.stats.items() if v} }")
    s.check()


@pytest.mark.timeout(int(GX_MINUTES * 60) + 2400)
def test_gx_combined_soak(boot_gx):
    """G-X (module docstring): P5 + T64 + the device sampler in one serving-order session, ``MOTIF3_GX_MINUTES``."""
    mesh, gen, pool = boot_gx
    s = GXSoak(gen, pool, seed=int(os.environ.get("MOTIF3_SOAK_SEED", "11")), src=soak_tokens(),
               programs=lambda: int(mesh.num_program_cache_entries()))  # fmt: skip
    pc0 = s.programs()
    t0 = time.time()
    n = s.run(seconds=GX_MINUTES * 60)
    dt = (time.time() - t0) / 60
    ok = s.final_checks()
    log(f"G-X end: {ok}")
    assert all(ok.values()), ok
    assert s.programs() == pc0, f"program cache grew: {pc0} -> {s.programs()}"
    s.report(f"G-X ({n} rounds, {dt:.1f} min)")
    log(f"G-X generator stats: { {k: v for k, v in gen.stats.items() if v} }; sampler {gen.sampling_stats()}")
    s.check()
    if GX_MINUTES >= 30:
        assert s.st["calls"] >= GX_MIN_CALLS, f"{s.st['calls']} prefill calls in {dt:.1f} min (bar {GX_MIN_CALLS})"
    assert gen.stats["wide_steps"] > 0 and gen.stats["overflow_passes"] > 0 and gen.stats["sampled_steps"] > 0


# ======================================================================================================================
# host: the driver on the emulated paged cache
# ======================================================================================================================
class _FakeDecode:
    """Decode steps on the emulated cache of ``tests/test_resumed_prefill.py`` (KV at ``p`` = ``(p, token, hash of
    the prefix [0, p])``): writes each active lane's anchor (and draft) KV through its page-table row, answers the
    "model" ``argmax = hash % V`` (the emulator's prefill logits rule) and an MTP guess; checks the positions and
    page tables with the generator's real host plan (``plan_spec_step``)."""

    def __init__(self, gen, pool):
        from models.demos.motif3.tests.test_resumed_prefill import chain

        self.gen, self.pool, self.chain = gen, pool, chain
        self.V = int(gen.cfg.vocab_size)

    def _write(self, lane: int, p: int, tok: int, pt: torch.Tensor) -> int:
        kv = self.pool.kv
        prev = kv.get((int(pt[lane, (p - 1) // BS]), (p - 1) % BS))
        assert prev is not None and prev[0] == p - 1, f"lane {lane}: no KV at position {p - 1} before the step ({prev})"
        h = self.chain(prev[2], tok)
        kv[(int(pt[lane, p // BS]), p % BS)] = (p, int(tok), h)
        return h

    def spec(self, batch, *, kv_cache, enable_trace, want_logits, path=None, sampling=None):
        plan = self.gen.plan_spec_step(batch)
        self.gen.last_spec = plan
        am = torch.full((32, 2), -1, dtype=torch.int32)
        mm = torch.full((32, 2), -1, dtype=torch.int32)
        logits = torch.zeros(32, self.V, dtype=torch.bfloat16) if want_logits else None
        for lane in torch.nonzero(batch.positions >= 0).flatten().tolist():
            p, d = int(batch.positions[lane]), int(batch.draft_tokens[lane])
            h = self._write(lane, p, int(batch.tokens[lane]), batch.page_table)
            am[lane, 0], mm[lane, 0] = h % self.V, (h // 7) % self.V
            if logits is not None:
                logits[lane, h % self.V] = 1.0
            if d >= 0:
                h1 = self._write(lane, p + 1, d, batch.page_table)
                am[lane, 1], mm[lane, 1] = h1 % self.V, (h1 // 7) % self.V
        return api.SpecDecodeResult(logits=logits, argmax=am, mtp_argmax=mm)

    def sampled_tokens(self, batch, sampling, *, kv_cache, enable_trace, path=None):
        res = self.spec(api.SpecDecodeBatch.from_decode_batch(batch), kv_cache=kv_cache, enable_trace=enable_trace,
                        want_logits=False)  # fmt: skip
        tok = res.argmax[:, 0].to(torch.int64)
        temp = list(sampling[0])
        for lane in range(32):
            if temp[lane] > 0:  # a "sampled" lane takes another token of the vocabulary
                tok[lane] = (tok[lane] + 1 + lane) % self.V
        return type("SampleResult", (), {"tokens": tok})()


def test_host_soak_driver_on_emulator(monkeypatch):
    """The CP9-P driver on the host (module docstring): the sweep over the 56 packed shapes of the production geometry
    (each call planned as exactly its shape, every shape reached), then 400 rounds of the serving mix through
    ``prefill_forward_batch`` with packed prefill on against the emulated paged cache, a fake decode that writes the
    emulated KV, freed blocks erased from the emulation (so a stale hit cannot pass). Every prefill read of every call
    is checked by the emulator; the coverage the device soak asserts is reached; the probe is identical."""
    from models.demos.motif3.tests.test_resumed_prefill import fake_generator, host_cfg

    cfg = host_cfg(kv_replicated_decode=True)
    cfg.set_kv_geometry(NUM_BLOCKS, BS)
    gen, model, pool, _ = fake_generator(cfg, monkeypatch, packed=True, num_blocks=NUM_BLOCKS)
    fake = _FakeDecode(gen, pool)
    monkeypatch.setattr(gen, "decode_forward_spec", fake.spec)
    monkeypatch.setattr(gen, "decode_forward_sampled", fake.sampled_tokens)

    def on_free(b: int) -> None:  # a freed block's KV is stale: erase it from the emulation
        for q in range(BS):
            pool.kv.pop((b, q), None)
            pool.mtp_kv.pop((b, q), None)
        pool.complete.discard(b)

    rng = random.Random(1)
    src = [rng.randrange(5000) for _ in range(20000)]
    s = Soak(gen, pool, seed=3, src=src, programs=lambda: 0, on_free=on_free)
    n = s.run(rounds=400, log_every=100)
    s.report(f"host soak ({n} rounds)")
    s.check()
    assert gen.stats["packed_passes"] > 0 and gen.stats["prefill_calls"] == s.st["calls"] + s.st["probes"]
    assert s.swept == [tuple(x) for x in cfg.packed_prefill_shapes()] and len(s.swept) == 56, s.swept


def test_host_gx_driver_on_emulator(monkeypatch):
    """G-X's driver on the host (module docstring): :class:`GXSoak` through ``prefill_forward_batch`` with packed
    prefill on against the emulated paged cache, a fake decode that writes the emulated KV and routes every step like a
    ``spec_verify="auto"`` launch (``verify_plan.choose_verify_kind``; T64 steps planned by ``plan_wide_step``), 300
    rounds: every kind of step runs with the routing the driver asserts, the coverage of the device soak is reached,
    the probe is identical."""
    from models.demos.motif3.tests.test_resumed_prefill import fake_generator, host_cfg
    from models.demos.motif3.tt import verify_plan as VP

    cfg = host_cfg(kv_replicated_decode=True)
    cfg.set_kv_geometry(NUM_BLOCKS, BS)
    gen, model, pool, _ = fake_generator(cfg, monkeypatch, packed=True, num_blocks=NUM_BLOCKS)
    fake = _FakeDecode(gen, pool)

    def spec(batch, *, kv_cache, enable_trace, want_logits, path=None, sampling=None):
        kind = VP.choose_verify_kind(batch, "auto", want_logits, sampling, kv_mode="all_split") if path is None \
            else path[0]  # fmt: skip
        res = fake.spec(batch, kv_cache=kv_cache, enable_trace=enable_trace, want_logits=want_logits)
        if kind == "wide":
            gen.last_spec = VP.plan_wide_step(batch, cfg=gen.cfg, mode="all_split")
        gen.last_verify_kind = kind
        return res

    def sampled(batch, sampling, *, kv_cache, enable_trace, path=None):
        gen.last_verify_kind = "spec"
        return fake.sampled_tokens(batch, sampling, kv_cache=kv_cache, enable_trace=enable_trace)

    monkeypatch.setattr(gen, "decode_forward_spec", spec)
    monkeypatch.setattr(gen, "decode_forward_sampled", sampled)

    def on_free(b: int) -> None:
        for q in range(BS):
            pool.kv.pop((b, q), None)
            pool.mtp_kv.pop((b, q), None)
        pool.complete.discard(b)

    rng = random.Random(2)
    src = [rng.randrange(5000) for _ in range(20000)]
    s = GXSoak(gen, pool, seed=5, src=src, programs=lambda: 0, on_free=on_free)
    n = s.run(rounds=300, log_every=100, sweep=False)
    s.report(f"host G-X driver ({n} rounds)")
    s.check()
    assert all(s.st[f"step_{k}"] > 0 for k in GXSoak.KINDS), dict(s.st)
