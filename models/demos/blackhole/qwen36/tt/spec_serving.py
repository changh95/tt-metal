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

Ladder (design (a)): bucket widths -> rows per user T = k + 1, chosen by w_grid = highest live slot + 1:
    w <= 8 : T = 4 (buckets 1, 2, 4, 8)    w <= 10 : T = 3    w <= 16 : T = 2    w > 16 : plain decode
(``QWEN36_SPEC_ALLOW_FRACTURED=1`` adds (32, 2): R = 64, the fractured reduce-scatter numerics of verify_step.py that
flip greedy near-ties; ``QWEN36_SPEC_LADDER="1:4,2:4,4:4,8:4,10:3,16:2"`` overrides). Every plan's R = w*T <= 32 runs
the decode step's own ops, so a committed stream is bitwise the plain greedy decode's.

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

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from models.demos.blackhole.qwen36.tt import verify_grid as vg

TILE = 32
DEFAULT_LADDER = "1:4,2:4,4:4,8:4,10:3,16:2"
FRACTURED_LADDER_TAIL = "32:2"


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

    def __init__(self, entries: Sequence[tuple[int, int]], k_max: int, max_batch: int, max_rows: int = TILE):
        k_max = int(k_max)
        assert k_max >= 1, "speculative decoding needs num_speculative_tokens >= 1"
        plans = []
        for w, T in entries:
            w, T = int(w), int(T)
            T = min(T, k_max + 1)
            if w > max_batch or T < 2:
                continue
            plans.append(Plan(w, T))
        # after clamping T the same (w, T) may repeat: keep the first
        seen, uniq = set(), []
        for p in plans:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        plans = uniq
        assert plans, f"empty ladder for k_max={k_max} max_batch={max_batch}: {list(entries)}"
        for a, b in zip(plans, plans[1:]):
            assert a.w < b.w, f"ladder widths must ascend: {a} {b}"
            assert a.T >= b.T, f"ladder T must not grow with the width: {a} {b}"
        for p in plans:
            assert p.R == vg.grid_rows(p.w, p.T), f"{p}: w*T must fill the tile-padded grid (no tile padding rows)"
            assert p.R <= max_rows, f"{p}: R = {p.R} > {max_rows} (set QWEN36_SPEC_ALLOW_FRACTURED=1 for R > 32)"
        self.plans = plans
        self.k_max = k_max
        self.max_batch = int(max_batch)

    @classmethod
    def from_spec(cls, spec: str, k_max: int, max_batch: int, allow_fractured: bool = False):
        entries = []
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            w, T = item.split(":")
            entries.append((int(w), int(T)))
        if allow_fractured:
            entries += [tuple(int(v) for v in FRACTURED_LADDER_TAIL.split(":"))]
        return cls(entries, k_max, max_batch, max_rows=64 if allow_fractured else TILE)

    @classmethod
    def default(cls, k_max: int, max_batch: int, allow_fractured: bool = False):
        return cls.from_spec(DEFAULT_LADDER, k_max, max_batch, allow_fractured)

    @property
    def widths(self) -> list:
        return [p.w for p in self.plans]

    def plan_for(self, w_grid: int) -> Optional[Plan]:
        for p in self.plans:
            if p.w >= w_grid:
                return p
        return None

    def band_max(self, T: int) -> int:
        """The widest bucket running at T (a join landing at a slot >= band_max leaves the band)."""
        ws = [p.w for p in self.plans if p.T == T]
        assert ws, f"no bucket at T={T}"
        return max(ws)

    def min_T_above(self, w: int) -> int:
        """The smallest T a batch wider than w can run at: min over the buckets above, 1 when plain decode is
        reachable (the ladder stops below max_batch). Pending rows a_s fit every reachable band iff a_s <= that - 1."""
        above = [p.T for p in self.plans if p.w > w]
        if self.plans[-1].w < self.max_batch:
            above.append(1)
        return min(above) if above else self.plans[-1].T


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
    catchup: list = field(default_factory=list)  # slots whose first draft step must fill the head's KV first


@dataclass
class SlotState:
    owner: Optional[str] = None
    accept_prev: int = 0
    seen: bool = False  # has run a step under this owner (a fresh slot may carry an imported hidden row)


class SpecServingState:
    """Per-slot host bookkeeping + the ladder decision of the served speculative loop."""

    def __init__(self, ladder: Ladder, bmax: int):
        self.ladder = ladder
        self.bmax = int(bmax)
        self.slots = [SlotState() for _ in range(self.bmax)]
        self.plan: Optional[Plan] = None
        self.stats = {"steps": 0, "spec_steps": 0, "plain_steps": 0, "flushes": 0, "migrations": 0, "plan_changes": 0}

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
        band_max = self.ladder.band_max(self.plan.T)
        fits = self.max_pending() <= self.ladder.min_T_above(band_max) - 1
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

    def begin_step(
        self,
        row_req_ids: Sequence[Optional[str]],
        drafts: Sequence[Optional[Sequence[int]]],
        eligible: bool,
        flush: bool = False,
        has_hidden: Optional[Callable[[int], bool]] = None,
    ) -> StepPlan:
        """Decide this step. row_req_ids[s] = the request at slot s (None = pad); drafts[s] = its scheduled draft
        tokens (a -1 ends the list: unfilled scheduler placeholders); eligible = every live request may take the
        greedy verify path (else the whole step is plain decode); flush = the scheduler asked for a zero-draft step.
        Raises SpecProtocolError when a pending prefix cannot survive the transition."""
        self._sync_owners(row_req_ids)
        self.stats["steps"] += 1
        live_all = self.live_mask()
        assert any(live_all), "a decode step needs at least one live slot"
        w_grid = max(s for s in range(self.bmax) if live_all[s]) + 1
        plan = self.ladder.plan_for(w_grid) if eligible else None
        if flush and self.plan is not None and eligible and self.plan.w >= w_grid:
            plan = self.plan  # the flush commits the pending rows at the plan that holds them
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
            return StepPlan(mode="plain", w_grid=w_grid, prev_plan=self.plan, flush=flush)
        prev = self.plan
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
            if live[s] and not st.seen and has_hidden is not None and has_hidden(s):
                catchup.append(s)
        self.stats["spec_steps"] += 1
        if flush:
            self.stats["flushes"] += 1
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
