# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""No-device tests of the served speculative-decoding state machine (tt/spec_serving.py) and its verify_grid helpers.

A simulated engine (scheduler with the admission-hold protocol + runner) drives SpecServingState over the CPU
SlotOracle through joins / leaves / flushes / plain-forced steps with random, oracle and mixed drafts: every request's
committed stream must be the plain greedy stream of its prompt, exactly as the device test claims of the traced
steps. Run: pytest models/demos/blackhole/qwen36/tests/test_spec_serving_cpu.py -q
"""
import random

import pytest
import torch

from models.demos.blackhole.qwen36.tt import spec_serving as ss
from models.demos.blackhole.qwen36.tt import verify_grid as vg


def _toy_next(ctx):
    h = 17
    for t in ctx:
        h = (h * 1000003 + int(t) * 7919 + 1) % 1000003
    return h % 50


def _greedy_stream(prompt, n):
    ctx = list(prompt)
    out = []
    for _ in range(n):
        t = _toy_next(ctx)
        out.append(t)
        ctx.append(t)
    return out


# ------------------------------------------------------------------------------------------------ ladder
def test_ladder_default_and_plan_for():
    lad = ss.Ladder.default(k_max=3, max_batch=32)
    assert [(p.w, p.T) for p in lad.plans] == [(1, 4), (2, 4), (4, 4), (8, 4), (10, 3), (16, 2)]
    assert lad.plan_for(1) == ss.Plan(1, 4)
    assert lad.plan_for(3) == ss.Plan(4, 4)
    assert lad.plan_for(8) == ss.Plan(8, 4)
    assert lad.plan_for(9) == ss.Plan(10, 3)
    assert lad.plan_for(11) == ss.Plan(16, 2)
    assert lad.plan_for(17) is None  # plain decode
    assert lad.band_max(4) == 8 and lad.band_max(3) == 10 and lad.band_max(2) == 16
    assert lad.min_T_above(8) == 1  # plain is reachable (ladder stops at 16 < 32)
    assert lad.widths == [1, 2, 4, 8, 10, 16]


def test_ladder_clamps_k_and_fractured(expect_error):
    lad = ss.Ladder.default(k_max=1, max_batch=32)
    assert all(p.T == 2 for p in lad.plans) and lad.widths == [1, 2, 4, 8, 10, 16]
    lad = ss.Ladder.default(k_max=3, max_batch=8)
    assert lad.widths == [1, 2, 4, 8] and lad.min_T_above(8) == 4  # nothing above: stays in band
    lad = ss.Ladder.default(k_max=3, max_batch=32, allow_fractured=True)
    assert lad.plans[-1] == ss.Plan(32, 2) and lad.plan_for(17) == ss.Plan(32, 2)
    assert lad.min_T_above(8) == 2
    with expect_error(AssertionError, "R = 64"):
        ss.Ladder.from_spec("32:2", k_max=3, max_batch=32)
    with expect_error(AssertionError, "must not grow"):
        ss.Ladder.from_spec("4:2,8:4", k_max=3, max_batch=32)
    with expect_error(AssertionError, "tile-padded"):
        ss.Ladder.from_spec("12:3", k_max=3, max_batch=32)  # 36 rows -> 64


# ------------------------------------------------------------------------------------------------ helpers
def test_commit_with_real_draft_counts():
    T = 4
    toks = [[1, 5, 6, 7], [2, 5, 6, 7], [3, 0, 0, 0]]
    am = torch.tensor([5, 6, 7, 8, 5, 6, 9, 9, 0, 0, 0, 0])  # user 0: all 3 match, user 1: 2 match, user 2: pads
    accepts, committed = vg.commit(am, toks, T, n_drafts=[3, 1, 0])
    assert accepts == [3, 1, 0]
    assert committed == [[5, 6, 7, 8], [5, 6], [0]]
    accepts, committed = vg.commit(am, toks, T)  # default: every draft is real
    assert accepts == [3, 2, 3]  # the pad user's zero drafts match its zero argmaxes


def test_migration_matrix_moves_pending_rows():
    R_old, R_new = vg.grid_rows(4, 4), vg.grid_rows(8, 3)
    m = vg.migration_matrix(4, 4, R_old, 8, 3, R_new, keep=[True, False, True, True])
    x = torch.arange(R_old * 3, dtype=torch.float32).reshape(R_old, 3)  # row r = [3r, 3r+1, 3r+2]
    y = m @ x
    assert y.shape == (R_new, 3)
    for s in range(4):
        for j in range(3):
            src = x[vg.row(s, j, 4)]
            dst = y[vg.row(s, j, 3)]
            if s == 1:
                assert torch.equal(dst, torch.zeros(3))
            else:
                assert torch.equal(dst, src)
    assert torch.equal(y[4 * 3 :], torch.zeros(R_new - 12, 3))  # users 4..7: zero rows
    assert int(m.sum()) == 3 * 3


def test_chain_drafts_pads_keep_position_minus_one():
    seen = []

    def run_step(tok, pos):
        seen.append((list(tok), list(pos)))
        return [t + 1 for t in tok]

    d = vg.chain_drafts(run_step, 3, 2, last=[10, 20, 30], positions=[5, 9, 7], pad=[False, True, False])
    assert seen == [([10, 0, 30], [4, -1, 6]), ([11, 0, 31], [5, -1, 7])]
    assert d == [[11, 12], [0, 0], [31, 32]]


# ------------------------------------------------------------------------------------------------ simulated engine
class Sim:
    """A vLLM-shaped engine over the CPU oracle: slots, the scheduler's admission-hold protocol (holds an admission
    that would break the pending prefix for one flush step), the runner's per-step commit and draft proposal, the
    request streams. Drafts: "oracle" (true continuation), "random", "mixed"."""

    def __init__(self, ladder, bmax, policy, seed, respect_hold=True):
        self.state = ss.SpecServingState(ladder, bmax)
        self.oracle = ss.SlotOracle(_toy_next, bmax)
        self.bmax = bmax
        self.policy = policy
        self.rng = random.Random(seed)
        self.respect_hold = respect_hold
        self.owner = [None] * bmax  # slot -> request id
        self.req = {}  # id -> dict(prompt, ref, stream, last, pos, drafts, greedy, want)
        self.pending_join = []
        self.step_log = []

    def submit(self, rid, prompt, want, greedy=True):
        ref = _greedy_stream(prompt, want + 12)
        self.req[rid] = dict(
            prompt=list(prompt),
            ref=ref,
            stream=[ref[0]],
            last=ref[0],
            pos=len(prompt),
            drafts=[],
            greedy=greedy,
            want=want,
        )
        self.pending_join.append(rid)

    def _admit(self, rid):
        slot = self.owner.index(None)
        self.owner[slot] = rid
        self.oracle.set_prompt(slot, self.req[rid]["prompt"])

    def _release(self, slot):
        self.owner[slot] = None
        self.oracle.clear(slot)

    def live_count(self):
        return sum(o is not None for o in self.owner)

    def step(self):
        """One engine step: scheduler (admission / hold) then runner (spec or plain step). Returns the StepPlan."""
        hold = self.state.hold_info()
        flush = False
        ready = self.pending_join[: self.owner.count(None)]
        if ready and hold.pending_any and self.respect_hold:
            forces_plain = any(not self.req[r]["greedy"] for r in ready)
            crossing = hold.slots_before_crossing is not None and len(ready) > hold.slots_before_crossing
            if forces_plain or crossing:
                flush = True
                ready = []
        for rid in ready:
            self.pending_join.remove(rid)
            self._admit(rid)
        assert self.live_count() > 0
        eligible = all(self.req[o]["greedy"] for o in self.owner if o is not None)
        row_req_ids = list(self.owner)
        drafts = [self.req[o]["drafts"] if o is not None else None for o in self.owner]
        sp = self.state.begin_step(row_req_ids, drafts, eligible, flush=flush)
        committed_by_slot = {}
        if sp.mode == "spec":
            w, T = sp.plan.w, sp.plan.T
            last = [self.req[self.owner[s]]["last"] if sp.live[s] else 0 for s in range(w)]
            pos = [self.req[self.owner[s]]["pos"] if sp.live[s] else 0 for s in range(w)]
            tokens = self.state.grid_tokens(sp, last)
            am = self.oracle.run(tokens, pos, sp.accept_prev, sp.live)
            accepts, committed = self.state.commit(sp, am, tokens)
            for s in range(w):
                if sp.live[s]:
                    committed_by_slot[s] = committed[s]
                    # vLLM's accounting: 1..1+K_s tokens per scheduled request
                    assert 1 <= len(committed[s]) <= 1 + len(self.req[self.owner[s]]["drafts"]) or sp.flush
                    if sp.flush:
                        assert len(committed[s]) == 1
        else:
            for s in range(self.bmax):
                if self.owner[s] is not None:
                    r = self.req[self.owner[s]]
                    committed_by_slot[s] = [self.oracle.plain_step(s, r["last"], r["pos"])]
        # apply + propose drafts for the next step (drafts live on the host, so they survive plan changes)
        k_next = self.state.plan.k if (sp.mode == "spec" and self.state.plan) else 0
        for s, toks in committed_by_slot.items():
            r = self.req[self.owner[s]]
            r["stream"].extend(toks)
            r["pos"] += len(toks)
            r["last"] = toks[-1]
            n_done = len(r["stream"])
            true_next = r["ref"][n_done : n_done + k_next]
            if sp.mode != "spec" or not r["greedy"]:
                d = []
            elif self.policy == "oracle":
                d = list(true_next)
            elif self.policy == "random":
                d = [self.rng.randrange(50) for _ in range(k_next)]
            else:
                m = self.rng.randrange(k_next + 1)
                d = list(true_next[:m]) + [self.rng.randrange(50) for _ in range(k_next - m)]
            r["drafts"] = d
        # leaves: requests that reached their token budget
        for s in range(self.bmax):
            o = self.owner[s]
            if o is not None and len(self.req[o]["stream"]) - 1 >= self.req[o]["want"]:
                self._release(s)
        self.step_log.append((sp.mode, sp.plan, sp.flush, sp.migrate, sorted(committed_by_slot)))
        return sp

    def check_streams(self):
        for rid, r in self.req.items():
            n = min(len(r["stream"]), len(r["ref"]))
            assert r["stream"][:n] == r["ref"][:n], f"request {rid} diverged from the plain greedy stream"


@pytest.mark.parametrize("policy", ["random", "oracle", "mixed"])
@pytest.mark.parametrize("k_max", [3, 1])
def test_single_user_matches_greedy(policy, k_max):
    sim = Sim(ss.Ladder.default(k_max, 32), 32, policy, seed=k_max * 7 + len(policy))
    sim.submit("a", [1, 2, 3], want=60)
    while "a" in sim.pending_join or "a" in sim.owner:
        sim.step()
    sim.check_streams()
    assert len(sim.req["a"]["stream"]) - 1 >= 60
    if policy == "oracle":
        # every draft accepted: k+1 tokens per step
        assert sim.state.stats["spec_steps"] <= 60 // (k_max + 1) + 2


@pytest.mark.parametrize("policy", ["mixed", "oracle", "random"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_joins_leaves_cross_bands_with_hold_protocol(policy, seed):
    """Ramp 3 -> 9 -> 17 users (bucket changes inside the T=4 band, a band-up crossing 8 -> 10 that needs a flush, then
    plain decode above 16), then drain (band-down crossings by migration) -- every stream stays the greedy one."""
    rng = random.Random(seed)
    lad = ss.Ladder.default(3, 32)
    sim = Sim(lad, 32, policy, seed=seed)
    waves = [(0, 3), (3, 6), (7, 8), (40, 2)]  # (step, #requests)
    wants = {}
    n = 0
    for _, cnt in waves:
        for _ in range(cnt):
            wants[f"r{n}"] = rng.randrange(25, 45)
            n += 1
    submitted = {}
    i = 0
    for at, cnt in waves:
        for _ in range(cnt):
            submitted.setdefault(at, []).append(f"r{i}")
            i += 1
    step = 0
    while submitted or sim.pending_join or sim.live_count() > 0:
        for rid in submitted.pop(step, []):
            sim.submit(rid, [rng.randrange(50) for _ in range(3 + rng.randrange(4))], want=wants[rid])
        if sim.live_count() == 0 and not sim.pending_join:
            step += 1
            continue
        sim.step()
        step += 1
        assert step < 2000
    sim.check_streams()
    st = sim.state.stats
    plans = {p for m, p, *_ in sim.step_log if p is not None}
    assert len({p.T for p in plans}) >= 2, plans  # several bands visited
    assert st["plain_steps"] >= 1, st  # 17 users -> plain decode
    if policy == "oracle":  # every draft accepted -> pending prefixes at every transition
        assert st["migrations"] >= 2, st  # bucket changes inside a band / band-down changes carry the pending rows
        assert st["flushes"] >= 1, st  # the 8 -> 9 band-up crossing was held for a flush
    for rid, r in sim.req.items():
        assert len(r["stream"]) - 1 >= r["want"]


def test_plain_forcing_admission_needs_flush():
    sim = Sim(ss.Ladder.default(3, 32), 32, "oracle", seed=5)
    sim.submit("g", [1, 2, 3], want=30)
    sim.step()
    sim.step()  # pending prefixes exist now (oracle drafts -> accepts of 3)
    assert sim.state.pending_any
    sim.submit("s", [4, 5], want=10, greedy=False)
    sp = sim.step()
    assert sp.flush and sp.mode == "spec" and "s" in sim.pending_join  # held: flush, no admission
    assert not sim.state.pending_any
    sp = sim.step()
    assert sp.mode == "plain" and "s" in sim.owner  # admitted; whole step plain
    while sim.live_count():
        sim.step()
    sim.check_streams()


def test_crossing_without_hold_raises(expect_error):
    sim = Sim(ss.Ladder.default(3, 32), 32, "oracle", seed=9, respect_hold=False)
    for i in range(8):
        sim.submit(f"r{i}", [i, i + 1, i + 2], want=200)
    sim.step()  # 8 users at (8,4)
    sim.step()  # pending a=3 everywhere
    assert sim.state.plan == ss.Plan(8, 4) and sim.state.max_pending() == 3
    hold = sim.state.hold_info()
    assert hold.pending_any and hold.slots_before_crossing == 0 and hold.band_max == 8
    sim.submit("x", [7, 7, 7], want=10)  # the 9th user -> (10,3): a=3 does not fit T-1 = 2
    with expect_error(ss.SpecProtocolError, "plan change"):
        sim.step()


def test_plain_step_with_pending_raises(expect_error):
    st = ss.SpecServingState(ss.Ladder.default(3, 32), 4)
    sp = st.begin_step(["a", None, None, None], [[1, 2, 3], None, None, None], eligible=True)
    tokens = st.grid_tokens(sp, [9, 0, 0, 0])
    am = torch.tensor([1, 2, 3, 4])  # all drafts accepted
    accepts, committed = st.commit(sp, am, tokens)
    assert accepts == [3] and committed == [[1, 2, 3, 4]]
    with expect_error(ss.SpecProtocolError, "plain step"):
        st.begin_step(["a", "b", None, None], [[1], None, None, None], eligible=False)


def test_pad_users_and_fresh_slot_catchup():
    st = ss.SpecServingState(ss.Ladder.default(3, 32), 8)
    rows = [None, "a", None, "b", None, None, None, None]
    sp = st.begin_step(
        rows, [None, [1, -1, 5], None, [], None, None, None, None], eligible=True, has_hidden=lambda s: s == 3
    )
    assert sp.plan == ss.Plan(4, 4) and sp.live == [False, True, False, True]
    assert sp.n_drafts == [0, 1, 0, 0] and sp.drafts[1] == [1, 0, 0] and sp.catchup == [3]
    tokens = st.grid_tokens(sp, [0, 7, 0, 8])
    assert tokens[0] == [0, 0, 0, 0] and tokens[1] == [7, 1, 0, 0] and tokens[3] == [8, 0, 0, 0]
    am = torch.arange(16)
    am[vg.row(1, 0, 4)] = 1  # user a's draft 1 accepted
    accepts, committed = st.commit(sp, am, tokens)
    assert accepts == [0, 1, 0, 0] and committed[0] == [] and committed[1] == [1, int(am[vg.row(1, 1, 4)])]
    # next step: same owners -> no catch-up, pending a=1 for slot 1
    sp2 = st.begin_step(
        rows, [None, [3, 4, 5], None, [6, 7, 8], None, None, None, None], eligible=True, has_hidden=lambda s: True
    )
    assert sp2.catchup == [] and sp2.accept_prev == [0, 1, 0, 0] and not sp2.migrate
    # a new owner at slot 3 resets its pending and is fresh again
    sp3 = st.begin_step([None, "a", None, "c"], [None, [3], None, None], eligible=True, has_hidden=lambda s: True)
    assert sp3.catchup == [3] and sp3.accept_prev[3] == 0 and sp3.plan == ss.Plan(4, 4)


def test_hold_info_counts_free_slots_below_band():
    st = ss.SpecServingState(ss.Ladder.default(3, 32), 32)
    rows = [f"r{s}" if s in (0, 2, 5) else None for s in range(32)]
    sp = st.begin_step(rows, [[1, 2, 3] if r else None for r in rows], eligible=True)
    assert sp.plan == ss.Plan(8, 4)
    tokens = st.grid_tokens(sp, [1] * 8)
    am = torch.zeros(32, dtype=torch.int64)
    am[vg.row(2, 0, 4)] = 1
    am[vg.row(2, 1, 4)] = 2  # slot 2 accepts 2 drafts
    st.commit(sp, am, tokens)
    h = st.hold_info()
    assert h.pending_any and h.band_max == 8 and h.T == 4
    assert h.slots_before_crossing == 5  # free slots below 8: 1,3,4,6,7
    # with (32,2) reachable and a pending prefix of 2 > 1 -> still counts; a prefix of 1 would fit every band
    st2 = ss.SpecServingState(ss.Ladder.default(3, 32, allow_fractured=True), 32)
    sp = st2.begin_step(rows, [[1, 2, 3] if r else None for r in rows], eligible=True)
    am = torch.zeros(32, dtype=torch.int64)
    am[vg.row(2, 0, 4)] = 1  # slot 2 accepts 1 draft
    st2.commit(sp, am, st2.grid_tokens(sp, [1] * 8))
    assert st2.hold_info().slots_before_crossing is None  # a_s = 1 <= min T above - 1 = 1: migration always fits
