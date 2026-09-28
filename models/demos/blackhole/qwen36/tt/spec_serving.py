# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host side of SERVED speculative decoding with the MTP drafter (the P/D decode engine): the T ladder over the live
decode slots, the per-slot lazy-prefix bookkeeping across plan changes, the admission-hold protocol with the vLLM
scheduler, and the per-step commit. Pure python / torch (no ttnn) so the state machine is unit-testable with a CPU
oracle (tests/test_spec_serving_cpu.py); the device half is tt/spec_decoder.py.

Grid = decode slots. The verify grid's user s IS decode slot s (the GDN kernel commits grid user s into state slot s,
tests/VERIFY_W32_AUDIT.md), so a (w, T) plan covers slots 0..w-1 and every slot below the highest live one is a
grid user: a live request, or a PADDING user (token 0, KV update / SDPA skipped at position -1, accept 0, its
residual rows kept finite by the plan's pad mask so the body's row-mixing 0/1 matmuls never see NaN).

Ladder (design (a)): bucket widths -> rows per user T = k + 1, chosen by w_grid = highest live slot + 1. One ladder
per drafter (``ladder_from_env``):
  * MTP head (``QWEN36_SPEC_DRAFTER=mtp``, k chained one-layer steps): w <= 8 : T = 4 (buckets 1, 2, 4, 8),
    w <= 10 : T = 3, w <= 16 : T = 2, w > 16 : plain decode. Every plan's R = w*T <= 32 runs the decode step's own
    ops, so a committed stream is bitwise the plain greedy decode's. ``QWEN36_SPEC_ALLOW_FRACTURED=1`` adds (32, 2):
    R = 64 on the fractured reduce-scatter path of verify_step.py whose numerics flip greedy near-ties.
  * DFlash2 block drafter (``QWEN36_SPEC_DRAFTER=dflash2``, ONE 8-row block step per verify step at any k, so the
    largest T the verify budget allows wins): w <= 8 : T = 8 (k = 7; buckets 1, 2, 4 are R <= 32 bitwise, bucket 8 is
    R = 64: the fractured path, near-tie-bounded -- tests/DFLASH2_RESULTS.md (8,7) x2.13 with 4 near-tie flips of
    gap <= 0.25 over 8 x 64 tokens), w <= 16 : T = 2 (R = 32, bitwise; ``QWEN36_SPEC_LADDER="1:8,2:8,4:8,8:8,16:4"``
    selects the R = 64 k = 3 band instead), w > 16 : plain decode. The bitwise / near-tie bound is the knob
    ``QWEN36_SPEC_ALLOW_FRACTURED``: unset = R <= 64 allowed (the default above); ``0`` = strictly bitwise, the ladder
    becomes ``1:8,2:8,4:8,8:4,16:2`` (R <= 32 everywhere, (8,4) x1.86); ``1`` = the R <= 64 ladder plus the (32, 2) tail.
  * HYBRID (``QWEN36_SPEC_DRAFTER=hybrid``): both drafters resident, chosen per plan BY WIDTH -- the served A/B
    (plugin docs/SPECULATIVE.md) has DFlash2 winning only up to 4 users (GSM8K TPOT 9.4 vs 12.1 ms at 1, 10.3 vs 13.0
    at 4) and MTP winning above (14.4 vs 15.3 at 8, 23 vs 31 at 16). Default ladder ``1:8,2:8,4:8,8:4,10:3,16:2``: the
    buckets w <= HYBRID_DFLASH2_MAX_W = 4 run T = 8 with the DFlash2 block drafter (R <= 32, bitwise), the MTP bands
    of the mtp ladder above (k = 3 to 8, 2 to 10, 1 to 16), plain decode above 16. ``Ladder.drafter_for(plan)`` names
    the drafter of a plan; a drafter change is a plan change and goes through the flush / migration protocol below
    (a band-up crossing 4 -> 5 users is held for a flush like any other, a band-down 5 -> 4 migrates), while the
    device half keeps BOTH drafters' per-slot state current on every step (tt/spec_decoder.py ``_HybridDrafter``).
    ``QWEN36_SPEC_ALLOW_FRACTURED=1`` adds the (32, 2) MTP tail as for mtp.
    CONTEXT RULE (``QWEN36_SPEC_DFLASH2_MAX_CTX``, default HYBRID_DFLASH2_MAX_CTX; 0 = off): the block drafter's
    acceptance falls with the context length (its 2048-token sliding window; tests/DFLASH2_RESULTS.md section 9)
    while the MTP head's does not, so DFlash2 drafts only while the LONGEST live context on the grid (prompt +
    generated tokens = the row's decode position) is <= the limit. Above it the grid runs the LONG ladder
    (``QWEN36_SPEC_LADDER_LONG``, default the mtp ladder ``1:4,2:4,4:4,8:4,10:3,16:2``): the MTP head at every
    width. The mode is a property of the state machine (``SpecServingState.long``), evaluated every step from the
    live rows' positions: a fixed set of users can only grow, so the mode flips at most ONCE per set (short -> long,
    when a user grows past the limit or a long-prompt user is admitted) and flips back only when the long users
    leave AND the longest remaining context is below the limit minus ``QWEN36_SPEC_DFLASH2_CTX_HYST`` (default
    HYBRID_DFLASH2_CTX_HYSTERESIS) -- no flapping every step. A mode change is a drafter change and a plan change
    (e.g. (4,8) -> (4,4)) taken through the flush / migration protocol below; when the pending rows do not fit the
    new plan (a_s up to 7 vs T-1 = 3) the state machine itself runs one FLUSH step at the current mode's plan for
    the grid width first (a same-band plan: the pending rows always fit it) and switches on the next step -- unlike a
    width crossing this needs no scheduler hold, because the current band's plan covers every live slot (a join
    above the band is a width crossing and held as such). ``QWEN36_SPEC_K=3`` clamps both ladders to the same (w,4)
    plans; then a mode change keeps the plan and only changes the drafter (both states are current every step).
``QWEN36_SPEC_LADDER="w:T,..."`` overrides any default (its R must respect the knob's bound; in hybrid mode the
drafter of an overridden plan is still chosen by its width).

Lazy GDN prefix across plan changes (design (c)): the kernel commits the PREVIOUS step's accepted rows 1..a_s from the
plan's ``qkv_prev`` buffers, whose row layout is s*T + j. A plan change carries the pending rows over with an exact
0/1 row-permutation matmul (verify_grid.migration_matrix) -- possible iff every pending a_s <= T_new - 1. Bucket
changes within a band (same T) and band-DOWN changes (fewer users, larger T) always fit; a band-UP change (a join
above the band's width, smaller T) may not, and a plain step never fits a pending prefix. Those transitions need a
FLUSH first: a verify step at the current plan with zero drafts (every scheduled draft rejected, one token per user,
pending rows committed). The flush cannot include a user the current plan does not cover and every scheduled user
must produce exactly 1..1+K_s tokens (vLLM's placeholder accounting), so the flush must run BEFORE the admission:
the runner publishes ``HoldInfo`` after every step and the TT scheduler (vllm_tt_plugin.scheduler) holds an
admission that would cross for one decode-only step flagged ``flush`` (docs/SPECULATIVE.md). A crossing the
scheduler did not hold raises ``SpecProtocolError`` (a bug signal, never silent state corruption).
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from models.demos.blackhole.qwen36.tt import verify_grid as vg

TILE = 32
FRACTURED_ROWS = 64  # the widest verify grid the fractured (R > 32) path is allowed to run (near-tie numerics)
DEFAULT_LADDER = "1:4,2:4,4:4,8:4,10:3,16:2"  # MTP head
DEFAULT_LADDER_MTP = DEFAULT_LADDER
DEFAULT_LADDER_DFLASH2 = (
    "1:8,2:8,4:8,8:8,16:2"  # DFlash2 block drafter, R <= 64 (bucket 8 at T=8 is the fractured plan)
)
DEFAULT_LADDER_DFLASH2_BITWISE = "1:8,2:8,4:8,8:4,16:2"  # DFlash2, R <= 32 everywhere (QWEN36_SPEC_ALLOW_FRACTURED=0)
DEFAULT_LADDER_HYBRID = "1:8,2:8,4:8,8:4,10:3,16:2"  # hybrid: DFlash2 T=8 up to 4 users, the MTP bands above
DEFAULT_LADDER_HYBRID_LONG = DEFAULT_LADDER_MTP  # hybrid past the context limit: the MTP head at every width
HYBRID_DFLASH2_MAX_W = 4  # hybrid: buckets up to this width draft with DFlash2, wider ones with the MTP head
HYBRID_DFLASH2_MAX_CTX = 12288  # hybrid: DFlash2 only while the longest live context (tokens) is <= this (0 = no rule)
HYBRID_DFLASH2_CTX_HYSTERESIS = 1024  # hybrid: back to DFlash2 only below MAX_CTX - this (after the long users left)
FRACTURED_LADDER_TAIL = "32:2"
DRAFTERS = ("mtp", "dflash2", "hybrid")


class SpecProtocolError(RuntimeError):
    """The batch changed in a way the pending lazy prefix cannot survive (the scheduler hold protocol was bypassed)."""


@dataclass(frozen=True)
class Plan:
    """A (w, T) verify grid over decode slots 0..w-1."""

    w: int
    T: int

    @property
    def k(self) -> int:
        return self.T - 1

    @property
    def R(self) -> int:
        return self.w * self.T

    def __str__(self):
        return f"({self.w},T={self.T})"


class Ladder:
    """Bucket widths -> T. ``entries`` ascending in w with T non-increasing; ``plan_for(w_grid)`` is the smallest
    bucket that covers the highest live slot, None = plain decode (w_grid above the last bucket)."""

    def __init__(
        self,
        entries: Sequence[tuple[int, int]],
        k_max: int,
        max_batch: int,
        max_rows: int = TILE,
        drafter: str = "mtp",
        long_entries: Optional[Sequence[tuple[int, int]]] = None,
        dflash2_max_ctx: Optional[int] = None,
        ctx_hysteresis: Optional[int] = None,
    ):
        """``long_entries`` / ``dflash2_max_ctx`` / ``ctx_hysteresis``: the hybrid context rule (module docstring) --
        the plans of the LONG mode and the context limit (None / 0 = no rule: the width rule alone); ignored for the
        single-drafter ladders."""
        k_max = int(k_max)
        max_rows = int(max_rows)
        assert drafter in DRAFTERS, f"unknown drafter {drafter!r} (expected one of {DRAFTERS})"
        assert k_max >= 1, "speculative decoding needs num_speculative_tokens >= 1"
        self.k_max = k_max
        self.max_batch = int(max_batch)
        self.max_rows = max_rows
        self.drafter = drafter
        self.plans = self._build(entries, "ladder")
        self.dflash2_plans = (
            frozenset(p for p in self.plans if p.w <= HYBRID_DFLASH2_MAX_W) if drafter == "hybrid" else frozenset()
        )
        limit = int(dflash2_max_ctx or 0)
        if drafter == "hybrid" and limit > 0:
            assert long_entries, "the hybrid context rule needs the long ladder's entries"
            self.long_plans = self._build(long_entries, "long ladder")
            self.dflash2_max_ctx = limit
            hyst = HYBRID_DFLASH2_CTX_HYSTERESIS if ctx_hysteresis is None else int(ctx_hysteresis)
            assert 0 <= hyst < limit, f"QWEN36_SPEC_DFLASH2_CTX_HYST={hyst}: expected 0 <= hyst < the limit {limit}"
            self.ctx_hysteresis = hyst
        else:
            self.long_plans = []
            self.dflash2_max_ctx = None
            self.ctx_hysteresis = 0
        self._long_set = frozenset(self.long_plans)
        self.all_plans = list(self.plans) + [p for p in self.long_plans if p not in set(self.plans)]

    def _build(self, entries, what) -> list:
        plans = []
        for w, T in entries:
            w, T = int(w), int(T)
            T = min(T, self.k_max + 1)
            if w > self.max_batch or T < 2:
                continue
            plans.append(Plan(w, T))
        # after clamping T the same (w, T) may repeat: keep the first
        seen, uniq = set(), []
        for p in plans:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        plans = uniq
        assert plans, f"empty {what} for k_max={self.k_max} max_batch={self.max_batch}: {list(entries)}"
        for a, b in zip(plans, plans[1:]):
            assert a.w < b.w, f"{what} widths must ascend: {a} {b}"
            assert a.T >= b.T, f"{what} T must not grow with the width: {a} {b}"
        for p in plans:
            assert p.R == vg.grid_rows(p.w, p.T), f"{p}: w*T must fill the tile-padded grid (no tile padding rows)"
            assert p.R <= self.max_rows, (
                f"{p}: R = {p.R} > {self.max_rows} (QWEN36_SPEC_ALLOW_FRACTURED unset/1 allows R <= {FRACTURED_ROWS} for "
                "the dflash2 drafter and adds the (32,2) tail at 1; 0 = bitwise R <= 32 only)"
            )
        return plans

    @property
    def has_ctx_rule(self) -> bool:
        return self.dflash2_max_ctx is not None

    def plans_of(self, long: bool = False) -> list:
        """The plans of a mode: the long ladder when the context rule is in its long mode, else the ladder."""
        return self.long_plans if (long and self.long_plans) else self.plans

    def drafter_for(self, plan: Optional["Plan"], long: bool = False) -> Optional[str]:
        """The drafter that drafts at ``plan``: the ladder's single drafter, or in hybrid mode "dflash2" for the
        ladder's buckets up to HYBRID_DFLASH2_MAX_W and "mtp" for every other plan (the wider buckets and the long
        ladder's); ``long`` = the context rule's long mode, which matters only when k_max clamps a long plan onto a
        DFlash2 plan (QWEN36_SPEC_K=3: (w,4) in both ladders). None for a plain step (plan None)."""
        if plan is None:
            return None
        if self.drafter != "hybrid":
            return self.drafter
        if plan not in self.dflash2_plans:
            return "mtp"
        if long and plan in self._long_set:
            return "mtp"
        return "dflash2"

    @property
    def drafters(self) -> tuple:
        """The drafter names this ladder needs resident (hybrid: both)."""
        return ("mtp", "dflash2") if self.drafter == "hybrid" else (self.drafter,)

    def widths_for(self, drafter: str) -> list:
        """The bucket widths ``drafter`` drafts at in some mode (its draft-step buffers / traces)."""
        return sorted({p.w for p in self.all_plans if drafter in (self.drafter_for(p), self.drafter_for(p, long=True))})

    @property
    def fractured_plans(self) -> list:
        """The plans on the fractured (R > 32) verify path: near-tie-bounded, not bitwise."""
        return [p for p in self.all_plans if p.R > TILE]

    @staticmethod
    def parse_spec(spec: str, allow_fractured: bool = False) -> list:
        """ "w:T,w:T,..." -> [(w, T), ...]; ``allow_fractured`` appends the (32, 2) tail."""
        entries = []
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            w, T = item.split(":")
            entries.append((int(w), int(T)))
        if allow_fractured:
            entries += [tuple(int(v) for v in FRACTURED_LADDER_TAIL.split(":"))]
        return entries

    @classmethod
    def from_spec(
        cls,
        spec: str,
        k_max: int,
        max_batch: int,
        allow_fractured: bool = False,
        max_rows=None,
        drafter: str = "mtp",
        long_spec: Optional[str] = None,
        dflash2_max_ctx: Optional[int] = None,
        ctx_hysteresis: Optional[int] = None,
    ):
        """``spec`` = "w:T,w:T,..."; ``allow_fractured`` appends the (32, 2) tail (to the long ladder too); ``max_rows``
        (default 64 with the tail, 32 without) bounds every plan's R; ``drafter`` names the ladder's drafter policy
        (DRAFTERS); ``long_spec`` / ``dflash2_max_ctx`` / ``ctx_hysteresis`` = the hybrid context rule."""
        entries = cls.parse_spec(spec, allow_fractured)
        long_entries = cls.parse_spec(long_spec, allow_fractured) if long_spec else None
        if max_rows is None:
            max_rows = FRACTURED_ROWS if allow_fractured else TILE
        return cls(
            entries,
            k_max,
            max_batch,
            max_rows=int(max_rows),
            drafter=drafter,
            long_entries=long_entries,
            dflash2_max_ctx=dflash2_max_ctx,
            ctx_hysteresis=ctx_hysteresis,
        )

    @classmethod
    def default(cls, k_max: int, max_batch: int, allow_fractured: bool = False, drafter: str = "mtp"):
        """The drafter's default ladder (module docstring): ``allow_fractured`` = the QWEN36_SPEC_ALLOW_FRACTURED=1
        reading (R <= 64 + the (32,2) tail); ``for_drafter`` has the three-valued knob."""
        return cls.for_drafter(drafter, k_max, max_batch, allow_fractured="1" if allow_fractured else None)

    @classmethod
    def for_drafter(
        cls,
        drafter: str,
        k_max: int,
        max_batch: int,
        spec=None,
        allow_fractured=None,
        long_spec=None,
        dflash2_max_ctx=None,
        ctx_hysteresis=None,
    ):
        """``drafter`` in DRAFTERS; ``spec`` = a QWEN36_SPEC_LADDER override or None; ``allow_fractured`` = the raw
        QWEN36_SPEC_ALLOW_FRACTURED value: None (unset), "0" or "1" (see the module docstring); hybrid only:
        ``long_spec`` = a QWEN36_SPEC_LADDER_LONG override, ``dflash2_max_ctx`` = the raw QWEN36_SPEC_DFLASH2_MAX_CTX
        (None / "" = the default HYBRID_DFLASH2_MAX_CTX, 0 = no context rule), ``ctx_hysteresis`` = the raw
        QWEN36_SPEC_DFLASH2_CTX_HYST (None = the default)."""
        assert drafter in DRAFTERS, f"unknown drafter {drafter!r} (expected one of {DRAFTERS})"
        af = None if allow_fractured in (None, "") else str(allow_fractured).strip()
        assert af in (None, "0", "1"), f"QWEN36_SPEC_ALLOW_FRACTURED={allow_fractured!r}: expected 0 or 1"
        tail = af == "1"
        long_default = None
        if drafter == "dflash2":
            bitwise_only = af == "0"
            default = DEFAULT_LADDER_DFLASH2_BITWISE if bitwise_only else DEFAULT_LADDER_DFLASH2
            max_rows = TILE if bitwise_only else FRACTURED_ROWS
        else:
            # mtp, and hybrid (bitwise by construction: every default plan has R <= 32; the knob's 1 adds the tail)
            default = DEFAULT_LADDER_HYBRID if drafter == "hybrid" else DEFAULT_LADDER_MTP
            max_rows = FRACTURED_ROWS if tail else TILE
            if drafter == "hybrid":
                long_default = DEFAULT_LADDER_HYBRID_LONG
        limit = HYBRID_DFLASH2_MAX_CTX if dflash2_max_ctx in (None, "") else int(str(dflash2_max_ctx).strip())
        assert limit >= 0, f"QWEN36_SPEC_DFLASH2_MAX_CTX={dflash2_max_ctx!r}: expected >= 0 (0 = no context rule)"
        hyst = None if ctx_hysteresis in (None, "") else int(str(ctx_hysteresis).strip())
        return cls.from_spec(
            spec or default,
            k_max,
            max_batch,
            allow_fractured=tail,
            max_rows=max_rows,
            drafter=drafter,
            long_spec=(long_spec or long_default) if drafter == "hybrid" else None,
            dflash2_max_ctx=limit if drafter == "hybrid" else None,
            ctx_hysteresis=hyst,
        )

    @classmethod
    def from_env(cls, drafter: str, k_max: int, max_batch: int, env=None):
        """The served ladder: QWEN36_SPEC_LADDER / QWEN36_SPEC_ALLOW_FRACTURED (+ the hybrid's QWEN36_SPEC_LADDER_LONG /
        QWEN36_SPEC_DFLASH2_MAX_CTX / QWEN36_SPEC_DFLASH2_CTX_HYST) of ``env`` (default os.environ)."""
        env = os.environ if env is None else env
        return cls.for_drafter(
            drafter,
            k_max,
            max_batch,
            spec=env.get("QWEN36_SPEC_LADDER"),
            allow_fractured=env.get("QWEN36_SPEC_ALLOW_FRACTURED"),
            long_spec=env.get("QWEN36_SPEC_LADDER_LONG"),
            dflash2_max_ctx=env.get("QWEN36_SPEC_DFLASH2_MAX_CTX"),
            ctx_hysteresis=env.get("QWEN36_SPEC_DFLASH2_CTX_HYST"),
        )

    @property
    def widths(self) -> list:
        """Every bucket width of every mode (the draft-step buffers are built per width)."""
        return sorted({p.w for p in self.all_plans})

    def plan_for(self, w_grid: int, long: bool = False) -> Optional[Plan]:
        for p in self.plans_of(long):
            if p.w >= w_grid:
                return p
        return None

    def band_max(self, T: int, long: bool = False) -> int:
        """The widest bucket running at T (a join landing at a slot >= band_max leaves the band)."""
        ws = [p.w for p in self.plans_of(long) if p.T == T]
        assert ws, f"no bucket at T={T}"
        return max(ws)

    def min_T_above(self, w: int, long: bool = False) -> int:
        """The smallest T a batch wider than w can run at: min over the buckets above, 1 when plain decode is
        reachable (the ladder stops below max_batch). Pending rows a_s fit every reachable band iff a_s <= that - 1."""
        plans = self.plans_of(long)
        above = [p.T for p in plans if p.w > w]
        if plans[-1].w < self.max_batch:
            above.append(1)
        return min(above) if above else plans[-1].T


@dataclass
class HoldInfo:
    """Runner -> scheduler sidecar after every decode step (vllm_tt_plugin.spec_mtp.set_tt_spec_hold).

    pending_any: some live user has an accepted prefix the next verify step must commit (a plain step or a step at a
    smaller T cannot). slots_before_crossing: how many admissions the current band absorbs without leaving it (free
    slots below the band's width) -- None when no reachable band can break the pending rows (or nothing pending).
    The scheduler holds an admission of n requests for one flush step iff pending_any and (a request forces the
    plain path or n > slots_before_crossing)."""

    pending_any: bool = False
    slots_before_crossing: Optional[int] = None
    T: int = 0
    band_max: int = 0


@dataclass
class StepPlan:
    """What the device half has to run for one decode step (SpecServingState.begin_step)."""

    mode: str  # "spec" | "plain"
    w_grid: int
    plan: Optional[Plan] = None
    prev_plan: Optional[Plan] = None
    migrate: bool = False  # carry qkv_prev rows prev_plan -> plan (pending rows exist)
    flush: bool = False
    live: list = field(default_factory=list)  # [plan.w] grid user is a live request
    n_drafts: list = field(default_factory=list)  # [plan.w] real drafts per grid user (0 = padding / flush)
    drafts: list = field(default_factory=list)  # [plan.w][k] draft tokens (padding 0)
    accept_prev: list = field(default_factory=list)  # [plan.w] the previous step's a_s (0 for pads / fresh)
    catchup: list = field(default_factory=list)  # fresh slots with drafter state (MTP: hidden row -> catch-up step)
    drafter: Optional[str] = None  # the drafter that drafts at this plan (Ladder.drafter_for; None for plain)
    long: bool = False  # the context rule's mode this step runs in (hybrid: the long ladder / the MTP head)
    ctx_max: int = 0  # the longest live context (decode position) the rule saw this step
    ctx_flush: bool = False  # this flush was the state machine's own (a mode change whose pending rows did not fit)


@dataclass
class SlotState:
    owner: Optional[str] = None
    accept_prev: int = 0
    seen: bool = False  # has run a step under this owner (a fresh slot may carry imported drafter state)


class SpecServingState:
    """Per-slot host bookkeeping + the ladder decision of the served speculative loop."""

    def __init__(self, ladder: Ladder, bmax: int):
        self.ladder = ladder
        self.bmax = int(bmax)
        self.slots = [SlotState() for _ in range(self.bmax)]
        self.plan: Optional[Plan] = None
        self.plan_long = False  # the context-rule mode the installed plan was chosen in
        self.drafter_last: Optional[str] = None  # the drafter of the last spec step (kept across plain steps)
        self.long = False  # the context rule's current mode (module docstring)
        self.ctx_max = 0
        self.stats = {
            "steps": 0,
            "spec_steps": 0,
            "plain_steps": 0,
            "flushes": 0,
            "migrations": 0,
            "plan_changes": 0,
            "drafter_switches": 0,  # hybrid: consecutive spec steps drafted by different drafters
            "ctx_switches": 0,  # hybrid context rule: mode changes (short <-> long)
            "ctx_flushes": 0,  # hybrid context rule: the state machine's own flush steps before a mode change
        }

    # ------------------------------------------------------------------------------------------ queries
    def live_mask(self, n: Optional[int] = None) -> list:
        n = self.bmax if n is None else n
        return [self.slots[s].owner is not None for s in range(n)]

    @property
    def pending_any(self) -> bool:
        return any(st.owner is not None and st.accept_prev > 0 for st in self.slots)

    def max_pending(self) -> int:
        return max((st.accept_prev for st in self.slots if st.owner is not None), default=0)

    def hold_info(self) -> HoldInfo:
        if not self.pending_any or self.plan is None:
            return HoldInfo(pending_any=False)
        band_max = self.ladder.band_max(self.plan.T, self.plan_long)
        fits = self.max_pending() <= self.ladder.min_T_above(band_max, self.plan_long) - 1
        free_below = sum(1 for s in range(min(band_max, self.bmax)) if self.slots[s].owner is None)
        return HoldInfo(
            pending_any=True,
            slots_before_crossing=None if fits else free_below,
            T=self.plan.T,
            band_max=band_max,
        )

    # ------------------------------------------------------------------------------------------ step planning
    def _sync_owners(self, row_req_ids: Sequence[Optional[str]]):
        """Track the request per slot: a new owner (admission, or a re-used row) starts with no pending prefix and
        the fresh flag; a vacated slot forgets its prefix."""
        assert len(row_req_ids) <= self.bmax, (len(row_req_ids), self.bmax)
        for s in range(self.bmax):
            rid = row_req_ids[s] if s < len(row_req_ids) else None
            st = self.slots[s]
            if rid != st.owner:
                st.owner = rid
                st.accept_prev = 0
                st.seen = False

    def _update_mode(self, live_all, ctx_lens) -> None:
        """The context rule (module docstring): long when the longest live context is above the limit; back to short
        only below the limit minus the hysteresis. Positions are the rows' decode positions (prompt + generated)."""
        lad = self.ladder
        if ctx_lens is None or not lad.has_ctx_rule:
            return
        n = min(len(ctx_lens), self.bmax)
        self.ctx_max = max((int(ctx_lens[s]) for s in range(n) if live_all[s]), default=0)
        if not self.long and self.ctx_max > lad.dflash2_max_ctx:
            self.long = True
            self.stats["ctx_switches"] += 1
        elif self.long and self.ctx_max <= lad.dflash2_max_ctx - lad.ctx_hysteresis:
            self.long = False
            self.stats["ctx_switches"] += 1

    def begin_step(
        self,
        row_req_ids: Sequence[Optional[str]],
        drafts: Sequence[Optional[Sequence[int]]],
        eligible: bool,
        flush: bool = False,
        has_state: Optional[Callable[[int], bool]] = None,
        has_hidden: Optional[Callable[[int], bool]] = None,
        ctx_lens: Optional[Sequence[int]] = None,
    ) -> StepPlan:
        """Decide this step. row_req_ids[s] = the request at slot s (None = pad); drafts[s] = its scheduled draft
        tokens (a -1 ends the list: unfilled scheduler placeholders); eligible = every live request may take the
        greedy verify path (else the whole step is plain decode); flush = the scheduler asked for a zero-draft step;
        has_state(s) = the drafter holds imported / prefilled state for the FRESH slot s (the MTP hidden row, the
        DFlash2 context K/V) -> the slot is listed in ``catchup`` on its first step (``has_hidden`` is the old name);
        ctx_lens[s] = the decode position of slot s (its context length: prompt + generated), read by the hybrid
        context rule (None = leave the mode alone).
        Raises SpecProtocolError when a pending prefix cannot survive the transition."""
        if has_state is None:
            has_state = has_hidden
        self._sync_owners(row_req_ids)
        self.stats["steps"] += 1
        live_all = self.live_mask()
        assert any(live_all), "a decode step needs at least one live slot"
        w_grid = max(s for s in range(self.bmax) if live_all[s]) + 1
        self._update_mode(live_all, ctx_lens)
        long = self.long
        plan = self.ladder.plan_for(w_grid, long) if eligible else None
        ctx_flush = False
        if flush and self.plan is not None and eligible and self.plan.w >= w_grid:
            plan, long = self.plan, self.plan_long  # the flush commits the pending rows at the plan that holds them
        elif (
            plan is not None
            and self.plan is not None
            and self.pending_any
            and self.drafter_last is not None
            and self.ladder.drafter_for(plan, long) != self.drafter_last
            and self.max_pending() > plan.T - 1
        ):
            # a drafter change whose pending rows do not fit the new plan. A width crossing is held by the scheduler
            # (hold_info) and must not get here; a context-rule mode change is the state machine's own: flush first at
            # the CURRENT mode's plan for this width (same band as the installed plan: the rows fit), switch next step
            fplan = self.ladder.plan_for(w_grid, self.plan_long)
            if fplan is None or self.max_pending() > fplan.T - 1:
                raise SpecProtocolError(
                    f"plan change {self.plan} -> {plan} ({self.drafter_last} -> {self.ladder.drafter_for(plan, long)}) "
                    f"with pending prefixes up to {self.max_pending()} > T-1 = {plan.T - 1}: the scheduler must hold "
                    "the admission for a flush"
                )
            plan, long, flush, ctx_flush = fplan, self.plan_long, True, True
        if plan is None:
            if self.pending_any:
                raise SpecProtocolError(
                    f"plain step with pending accepted prefixes {[(s, st.accept_prev) for s, st in enumerate(self.slots) if st.accept_prev]}"
                    f" (eligible={eligible}, w_grid={w_grid}): the scheduler must hold the admission for a flush"
                )
            self.stats["plain_steps"] += 1
            for s in range(self.bmax):
                if live_all[s]:
                    self.slots[s].seen = True
                    self.slots[s].accept_prev = 0
            return StepPlan(
                mode="plain", w_grid=w_grid, prev_plan=self.plan, flush=flush, long=long, ctx_max=self.ctx_max
            )
        prev = self.plan
        drafter = self.ladder.drafter_for(plan, long)
        migrate = False
        if prev != plan:
            if self.pending_any:
                bad = [
                    (s, st.accept_prev) for s, st in enumerate(self.slots) if st.owner and st.accept_prev > plan.T - 1
                ]
                if bad:
                    raise SpecProtocolError(
                        f"plan change {prev} -> {plan} with pending prefixes longer than T-1: {bad}; "
                        "the scheduler must hold the admission for a flush"
                    )
                migrate = True
                self.stats["migrations"] += 1
            self.stats["plan_changes"] += 1
        if self.drafter_last is not None and drafter != self.drafter_last:
            self.stats["drafter_switches"] += 1
        w, k = plan.w, plan.k
        live = live_all[:w]
        n_drafts, dr, acc, catchup = [], [], [], []
        for s in range(w):
            st = self.slots[s]
            d = []
            if live[s] and not flush and drafts[s] is not None:
                for t in drafts[s]:
                    if int(t) < 0:
                        break
                    d.append(int(t))
                d = d[:k]
            n_drafts.append(len(d))
            dr.append(d + [0] * (k - len(d)))
            acc.append(st.accept_prev if live[s] else 0)
            if live[s] and not st.seen and has_state is not None and has_state(s):
                catchup.append(s)
        self.stats["spec_steps"] += 1
        if flush:
            self.stats["flushes"] += 1
        if ctx_flush:
            self.stats["ctx_flushes"] += 1
        return StepPlan(
            mode="spec",
            w_grid=w_grid,
            plan=plan,
            prev_plan=prev,
            migrate=migrate,
            flush=flush,
            live=live,
            n_drafts=n_drafts,
            drafts=dr,
            accept_prev=acc,
            catchup=catchup,
            drafter=drafter,
            long=long,
            ctx_max=self.ctx_max,
            ctx_flush=ctx_flush,
        )

    def grid_tokens(self, sp: StepPlan, last: Sequence[int]) -> list:
        """tokens[s] = [t'_s, d_1..d_k] per grid user (pads: zeros)."""
        return [([int(last[s])] + list(sp.drafts[s])) if sp.live[s] else [0] * sp.plan.T for s in range(sp.plan.w)]

    def commit(self, sp: StepPlan, argmax_rows: torch.Tensor, tokens: Sequence[Sequence[int]]):
        """Accept / commit the verify step's argmax rows: returns (accepts [w], committed [w]) over grid users (pads:
        0, []). Records the accepted prefixes as the next step's lazy rows and installs the plan."""
        assert sp.mode == "spec"
        accepts, committed = vg.commit(argmax_rows, tokens, sp.plan.T, sp.n_drafts)
        for s in range(sp.plan.w):
            st = self.slots[s]
            if sp.live[s]:
                st.accept_prev = int(accepts[s])
                st.seen = True
            else:
                accepts[s] = 0
                committed[s] = []
        self.plan = sp.plan
        self.plan_long = sp.long
        self.drafter_last = sp.drafter
        return accepts, committed


# ============================================================================================== CPU oracle
class SlotOracle:
    """CPU stand-in of the device verify step over decode slots with the kernel's lazy-prefix semantics: at the start
    of a call, live slot s's prefix (tokens at positions < P_s) is completed from the PREVIOUS call's rows
    0..accept_prev[s] (row 0 + the accepted drafts), like the kernel commits prev rows 1..a_s + cur row 0 -- across
    plan changes (the rows are per slot, as the migrated qkv_prev) and through pad users (ignored). ``next_fn`` is a
    deterministic toy LM over the whole prefix."""

    def __init__(self, next_fn, bmax: int):
        self.next_fn = next_fn
        self.prefix = [None] * bmax
        self.prev = [None] * bmax
        self.calls = 0

    def set_prompt(self, slot: int, prompt: Sequence[int]):
        self.prefix[slot] = [int(t) for t in prompt]
        self.prev[slot] = None

    def clear(self, slot: int):
        self.prefix[slot] = None
        self.prev[slot] = None

    def run(self, tokens, positions, accept_prev, live):
        w = len(tokens)
        T = len(tokens[0])
        R = vg.grid_rows(w, T)
        out = torch.zeros(R, dtype=torch.int64)
        self.calls += 1
        for s in range(w):
            if not live[s]:
                continue
            assert self.prefix[s] is not None, f"slot {s} has no prompt"
            if self.prev[s] is not None:
                self.prefix[s].extend(int(t) for t in self.prev[s][: int(accept_prev[s]) + 1])
            assert len(self.prefix[s]) == positions[s], (s, len(self.prefix[s]), positions[s])
            for j in range(T):
                out[vg.row(s, j, T)] = int(self.next_fn(self.prefix[s] + [int(t) for t in tokens[s][: j + 1]]))
            self.prev[s] = [int(t) for t in tokens[s]]
        return out

    def plain_step(self, slot: int, token: int, position: int) -> int:
        """A plain decode step of one slot (commits the pending prefix first, like the fused decode's state)."""
        if self.prev[slot] is not None:
            # a plain step is only legal without a pending prefix (accept 0): the kernel commits row 0 only
            self.prefix[slot].extend(self.prev[slot][:1])
            self.prev[slot] = None
        assert len(self.prefix[slot]) == position, (slot, len(self.prefix[slot]), position)
        t = int(self.next_fn(self.prefix[slot] + [int(token)]))
        self.prefix[slot].append(int(token))
        return t
