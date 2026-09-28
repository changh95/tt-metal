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


def test_ladder_dflash2_default_bitwise_and_fractured(expect_error):
    """The DFlash2 block drafter's ladder (spec_serving module docstring): T=8 up to 8 users (bucket 8 = the R=64
    fractured plan), T=2 up to 16, plain above; QWEN36_SPEC_ALLOW_FRACTURED unset / 0 / 1."""
    lad = ss.Ladder.for_drafter("dflash2", k_max=7, max_batch=32)
    assert [(p.w, p.T) for p in lad.plans] == [(1, 8), (2, 8), (4, 8), (8, 8), (16, 2)]
    assert lad.max_rows == 64 and [str(p) for p in lad.fractured_plans] == ["(8,T=8)"]
    assert lad.plan_for(1) == ss.Plan(1, 8) and lad.plan_for(3) == ss.Plan(4, 8) and lad.plan_for(4) == ss.Plan(4, 8)
    assert lad.plan_for(5) == ss.Plan(8, 8) and lad.plan_for(8) == ss.Plan(8, 8)
    assert lad.plan_for(9) == ss.Plan(16, 2) and lad.plan_for(16) == ss.Plan(16, 2)
    assert lad.plan_for(17) is None
    assert lad.band_max(8) == 8 and lad.band_max(2) == 16
    assert lad.min_T_above(8) == 1 and lad.min_T_above(16) == 1  # plain reachable above both bands
    assert lad.widths == [1, 2, 4, 8, 16]
    # 0 = strictly bitwise: (8,4) replaces (8,8), every R <= 32
    bit = ss.Ladder.for_drafter("dflash2", 7, 32, allow_fractured="0")
    assert [(p.w, p.T) for p in bit.plans] == [(1, 8), (2, 8), (4, 8), (8, 4), (16, 2)]
    assert bit.max_rows == 32 and bit.fractured_plans == [] and all(p.R <= 32 for p in bit.plans)
    # 1 = the R<=64 ladder plus the (32,2) tail
    fr = ss.Ladder.for_drafter("dflash2", 7, 32, allow_fractured="1")
    assert [(p.w, p.T) for p in fr.plans] == [(1, 8), (2, 8), (4, 8), (8, 8), (16, 2), (32, 2)]
    assert fr.plan_for(17) == ss.Plan(32, 2) and [str(p) for p in fr.fractured_plans] == ["(8,T=8)", "(32,T=2)"]
    # the T=4 band for 9..16 users is selectable (R = 64, near-tie-bounded)
    t4 = ss.Ladder.for_drafter("dflash2", 7, 32, spec="1:8,2:8,4:8,8:8,16:4")
    assert t4.plan_for(12) == ss.Plan(16, 4) and t4.plan_for(12).R == 64
    with expect_error(AssertionError, "R = 64"):
        ss.Ladder.for_drafter("dflash2", 7, 32, spec="1:8,2:8,4:8,8:8,16:4", allow_fractured="0")
    with expect_error(AssertionError, "R = 128"):
        ss.Ladder.for_drafter("dflash2", 7, 32, spec="1:8,16:8", allow_fractured="1")
    # k clamps T: QWEN36_SPEC_K=3 with the block drafter -> the first 3 path tokens, T=4 plans
    k3 = ss.Ladder.for_drafter("dflash2", 3, 32)
    assert [(p.w, p.T) for p in k3.plans] == [(1, 4), (2, 4), (4, 4), (8, 4), (16, 2)]
    # the MTP ladder through the same entry points is unchanged
    assert [(p.w, p.T) for p in ss.Ladder.for_drafter("mtp", 3, 32).plans] == [
        (1, 4),
        (2, 4),
        (4, 4),
        (8, 4),
        (10, 3),
        (16, 2),
    ]
    assert ss.Ladder.for_drafter("mtp", 3, 32).max_rows == 32
    assert ss.Ladder.for_drafter("mtp", 3, 32, allow_fractured="1").plans[-1] == ss.Plan(32, 2)
    assert ss.Ladder.for_drafter("mtp", 3, 32, allow_fractured="0").plans == ss.Ladder.for_drafter("mtp", 3, 32).plans
    env = {"QWEN36_SPEC_ALLOW_FRACTURED": "0", "QWEN36_SPEC_LADDER": "1:8,4:8"}
    assert [(p.w, p.T) for p in ss.Ladder.from_env("dflash2", 7, 32, env).plans] == [(1, 8), (4, 8)]
    assert [(p.w, p.T) for p in ss.Ladder.from_env("dflash2", 7, 32, {}).plans] == [(p.w, p.T) for p in lad.plans]
    with expect_error(AssertionError, "expected 0 or 1"):
        ss.Ladder.for_drafter("dflash2", 7, 32, allow_fractured="yes")
    with expect_error(AssertionError, "unknown drafter"):
        ss.Ladder.for_drafter("eagle", 7, 32)


def test_hold_info_dflash2_bands(expect_error):
    """T=8 band: a pending prefix of up to 7 rows never fits the T=2 band above (min T above 8 is 1: plain is
    reachable), so every admission that leaves the band is held; inside the band (free slots below 8) it is not."""
    st = ss.SpecServingState(ss.Ladder.for_drafter("dflash2", 7, 32), 32)
    rows = [f"r{s}" if s in (0, 1, 4) else None for s in range(32)]
    sp = st.begin_step(rows, [list(range(10, 17)) if r else None for r in rows], eligible=True)
    assert sp.plan == ss.Plan(8, 8) and sp.plan.R == 64 and sp.n_drafts[0] == 7
    tokens = st.grid_tokens(sp, [1] * 8)
    am = torch.zeros(64, dtype=torch.int64)
    for j in range(7):
        am[vg.row(0, j, 8)] = 10 + j  # user 0 accepts all 7 drafts
    accepts, committed = st.commit(sp, am, tokens)
    assert accepts[0] == 7 and len(committed[0]) == 8 and accepts[1] == 0 and committed[1] == [0]
    h = st.hold_info()
    assert h.pending_any and h.T == 8 and h.band_max == 8 and h.slots_before_crossing == 5  # free below 8: 2,3,5,6,7
    # five more users fit the band (w_grid 8): no flush needed, no migration within the (8,8) bucket
    rows2 = [f"r{s}" for s in range(8)] + [None] * 24
    sp2 = st.begin_step(rows2, [[1] if r else None for r in rows2], eligible=True)
    assert sp2.plan == ss.Plan(8, 8) and not sp2.migrate and sp2.accept_prev[0] == 7
    # a 9th user without a flush breaks the prefix (a_s = 7 > T_new - 1 = 1)
    st.commit(sp2, torch.zeros(64, dtype=torch.int64), st.grid_tokens(sp2, [1] * 8))
    sp3 = st.begin_step(rows2, [[1, 2, 3, 4, 5, 6, 7] if r else None for r in rows2], eligible=True)
    am = torch.zeros(64, dtype=torch.int64)
    am[vg.row(3, 0, 8)] = 1
    am[vg.row(3, 1, 8)] = 2  # user 3 accepts 2
    st.commit(sp3, am, st.grid_tokens(sp3, [1] * 8))
    rows3 = rows2[:8] + ["r8"] + [None] * 23
    with expect_error(ss.SpecProtocolError, "plan change"):
        st.begin_step(rows3, [[1] if r else None for r in rows3], eligible=True)


@pytest.mark.parametrize("policy", ["mixed", "oracle", "random"])
@pytest.mark.parametrize("knob", [None, "0", "1"])
def test_dflash2_ladder_joins_leaves_cross_bands(policy, knob):
    """The 3 -> 9 -> 17 ramp and drain on the DFlash2 ladders (k_max = 7): the T=8 -> T=2 band-up crossing at the 9th
    user needs a flush whenever a prefix is pending, the (32,2) tail (knob 1) keeps 17+ users speculative, the
    drain migrates back down -- every stream stays the greedy one."""
    seed = 11 + len(policy)
    rng = random.Random(seed)
    lad = ss.Ladder.for_drafter("dflash2", 7, 32, allow_fractured=knob)
    sim = Sim(lad, 32, policy, seed=seed)
    waves = [(0, 3), (3, 6), (7, 8), (40, 2)]
    n = 0
    wants, submitted = {}, {}
    for at, cnt in waves:
        for _ in range(cnt):
            wants[f"r{n}"] = rng.randrange(25, 60)
            submitted.setdefault(at, []).append(f"r{n}")
            n += 1
    step = 0
    while submitted or sim.pending_join or sim.live_count() > 0:
        for rid in submitted.pop(step, []):
            sim.submit(rid, [rng.randrange(50) for _ in range(3 + rng.randrange(4))], want=wants[rid])
        if sim.live_count() == 0 and not sim.pending_join:
            step += 1
            continue
        sp = sim.step()
        if sp.mode == "spec":
            assert sp.plan.R <= lad.max_rows
        step += 1
        assert step < 3000
    sim.check_streams()
    st = sim.state.stats
    plans = {p for m, p, *_ in sim.step_log if p is not None}
    assert {p.T for p in plans} >= {8, 2}, plans
    if knob == "1":
        assert ss.Plan(32, 2) in plans and st["plain_steps"] == 0, (plans, st)
    else:
        assert st["plain_steps"] >= 1, st  # 17 users -> plain decode
    if policy == "oracle":
        assert st["flushes"] >= 1 and st["migrations"] >= 1, st
        # k = 7 accepted every step in the T=8 band: 8 tokens per user per step
        t8 = [e for e in sim.step_log if e[0] == "spec" and e[1].T == 8 and not e[2]]
        assert t8, sim.step_log[:5]
    for rid, r in sim.req.items():
        assert len(r["stream"]) - 1 >= r["want"]


def test_ladder_hybrid_default_and_drafter_by_width(expect_error):
    """The hybrid ladder: DFlash2's T=8 buckets up to 4 users, the MTP bands above, plain past 16; every plan R <= 32
    (bitwise); the drafter of a plan is chosen by its width (also for a QWEN36_SPEC_LADDER override); the knob's 1
    adds the MTP (32,2) tail; k clamps the MTP bands like the mtp ladder."""
    lad = ss.Ladder.for_drafter("hybrid", 7, 32)
    assert lad.drafter == "hybrid" and lad.drafters == ("mtp", "dflash2")
    assert [(p.w, p.T, lad.drafter_for(p)) for p in lad.plans] == [
        (1, 8, "dflash2"),
        (2, 8, "dflash2"),
        (4, 8, "dflash2"),
        (8, 4, "mtp"),
        (10, 3, "mtp"),
        (16, 2, "mtp"),
    ]
    assert lad.max_rows == 32 and lad.fractured_plans == [] and all(p.R <= 32 for p in lad.plans)
    # the MTP head drafts at every width: the mtp bands, and the long ladder's (1,4) / (2,4) / (4,4) (context rule)
    assert lad.widths_for("dflash2") == [1, 2, 4] and lad.widths_for("mtp") == [1, 2, 4, 8, 10, 16]
    assert lad.widths == [1, 2, 4, 8, 10, 16]
    assert lad.drafter_for(lad.plan_for(3)) == "dflash2" and lad.drafter_for(lad.plan_for(4)) == "dflash2"
    assert lad.drafter_for(lad.plan_for(5)) == "mtp" and lad.drafter_for(lad.plan_for(16)) == "mtp"
    assert lad.plan_for(17) is None and lad.drafter_for(None) is None
    # band structure the hold protocol sees: T=8 band up to 4, then 8 / 10 / 16; plain reachable
    assert lad.band_max(8) == 4 and lad.band_max(4) == 8 and lad.band_max(3) == 10 and lad.band_max(2) == 16
    assert lad.min_T_above(4) == 1 and lad.min_T_above(16) == 1
    assert ss.Ladder.for_drafter("hybrid", 7, 32, allow_fractured="0").plans == lad.plans
    tail = ss.Ladder.for_drafter("hybrid", 7, 32, allow_fractured="1")
    assert tail.plans[-1] == ss.Plan(32, 2) and tail.drafter_for(tail.plans[-1]) == "mtp"
    # an override keeps the width rule: buckets <= 4 -> dflash2, wider -> mtp
    ov = ss.Ladder.for_drafter("hybrid", 7, 32, spec="1:8,4:8,8:2")
    assert [(p.w, p.T, ov.drafter_for(p)) for p in ov.plans] == [(1, 8, "dflash2"), (4, 8, "dflash2"), (8, 2, "mtp")]
    with expect_error(AssertionError, "R = 64"):
        ss.Ladder.for_drafter("hybrid", 7, 32, spec="1:8,4:8,8:8,16:2")
    # QWEN36_SPEC_K=3 shortens the DFlash2 band to the first 3 path tokens (T=4 plans), the MTP bands stay
    k3 = ss.Ladder.for_drafter("hybrid", 3, 32)
    assert [(p.w, p.T) for p in k3.plans] == [(1, 4), (2, 4), (4, 4), (8, 4), (10, 3), (16, 2)]
    assert [k3.drafter_for(p) for p in k3.plans] == ["dflash2"] * 3 + ["mtp"] * 3
    # the single-drafter ladders name their drafter for every plan; from_env reads the knob
    m = ss.Ladder.for_drafter("mtp", 3, 32)
    assert m.drafters == ("mtp",) and all(m.drafter_for(p) == "mtp" for p in m.plans) and m.widths_for("dflash2") == []
    d = ss.Ladder.for_drafter("dflash2", 7, 32)
    assert all(d.drafter_for(p) == "dflash2" for p in d.plans) and d.widths_for("dflash2") == d.widths
    env = {"QWEN36_SPEC_LADDER": "1:8,4:8,8:4"}
    assert [(p.w, p.T) for p in ss.Ladder.from_env("hybrid", 7, 32, env).plans] == [(1, 8), (4, 8), (8, 4)]


def _drafter_log(sim, lad):
    """(mode, drafter, plan, flush, migrate) per step of a Sim."""
    return [(m, d, p, fl, mg) for m, p, fl, mg, _, d in sim.step_log]


def test_ladder_hybrid_context_rule_plans(expect_error):
    """The hybrid context rule (QWEN36_SPEC_DFLASH2_MAX_CTX): the long ladder is the mtp ladder, its (1,4) / (2,4) /
    (4,4) plans are extra verify plans (all_plans), drafted by the MTP head; the wider plans are shared; the knobs
    parse (0 = off, custom limit / hysteresis / long ladder); QWEN36_SPEC_K=3 clamps both ladders onto the same
    plans and the mode alone picks the drafter."""
    lad = ss.Ladder.for_drafter("hybrid", 7, 32)
    assert lad.has_ctx_rule and lad.dflash2_max_ctx == ss.HYBRID_DFLASH2_MAX_CTX
    assert lad.ctx_hysteresis == ss.HYBRID_DFLASH2_CTX_HYSTERESIS
    assert [(p.w, p.T) for p in lad.long_plans] == [(1, 4), (2, 4), (4, 4), (8, 4), (10, 3), (16, 2)]
    assert [(p.w, p.T) for p in lad.all_plans] == [
        (1, 8),
        (2, 8),
        (4, 8),
        (8, 4),
        (10, 3),
        (16, 2),
        (1, 4),
        (2, 4),
        (4, 4),
    ]
    assert all(lad.drafter_for(p, long=True) == "mtp" for p in lad.long_plans)
    assert all(lad.drafter_for(p) == "mtp" for p in lad.long_plans)  # never a DFlash2 plan, whatever the mode
    assert lad.drafter_for(ss.Plan(4, 8)) == "dflash2" and lad.drafter_for(ss.Plan(4, 8), long=True) == "dflash2"
    assert lad.plan_for(3, long=True) == ss.Plan(4, 4) and lad.plan_for(3) == ss.Plan(4, 8)
    assert lad.plan_for(5, long=True) == ss.Plan(8, 4) and lad.plan_for(17, long=True) is None
    assert lad.band_max(4, long=True) == 8 and lad.band_max(3, long=True) == 10
    assert (
        lad.min_T_above(8, long=True) == 1
        and lad.min_T_above(4, long=True) == 1
        and lad.min_T_above(16, long=True) == 1
    )
    assert lad.fractured_plans == []
    # knobs
    off = ss.Ladder.from_env("hybrid", 7, 32, {"QWEN36_SPEC_DFLASH2_MAX_CTX": "0"})
    assert (
        not off.has_ctx_rule
        and off.long_plans == []
        and off.all_plans == off.plans
        and off.widths_for("mtp") == [8, 10, 16]
    )
    assert (
        off.plan_for(3, long=True) == ss.Plan(4, 8)
        and off.drafter_for(off.plan_for(3, long=True), long=True) == "dflash2"
    )
    env = {
        "QWEN36_SPEC_DFLASH2_MAX_CTX": "4096",
        "QWEN36_SPEC_DFLASH2_CTX_HYST": "512",
        "QWEN36_SPEC_LADDER_LONG": "1:4,4:4,8:2",
    }
    custom = ss.Ladder.from_env("hybrid", 7, 32, env)
    assert custom.dflash2_max_ctx == 4096 and custom.ctx_hysteresis == 512
    assert [(p.w, p.T) for p in custom.long_plans] == [(1, 4), (4, 4), (8, 2)]
    with expect_error(AssertionError, "CTX_HYST"):
        ss.Ladder.from_env(
            "hybrid", 7, 32, {"QWEN36_SPEC_DFLASH2_MAX_CTX": "1000", "QWEN36_SPEC_DFLASH2_CTX_HYST": "1000"}
        )
    tail = ss.Ladder.for_drafter("hybrid", 7, 32, allow_fractured="1")
    assert (
        tail.long_plans[-1] == ss.Plan(32, 2)
        and tail.plans[-1] == ss.Plan(32, 2)
        and tail.fractured_plans == [ss.Plan(32, 2)]
    )
    # single-drafter ladders ignore the rule
    for d in ("mtp", "dflash2"):
        single = ss.Ladder.from_env(d, 7, 32, {"QWEN36_SPEC_DFLASH2_MAX_CTX": "100"})
        assert not single.has_ctx_rule and single.long_plans == [] and single.all_plans == single.plans
    # k_max = 3: both ladders clamp onto (1,4) / (2,4) / (4,4); the plan is shared, the mode names the drafter
    k3 = ss.Ladder.for_drafter("hybrid", 3, 32)
    assert k3.all_plans == k3.plans and k3.long_plans == k3.plans
    assert k3.drafter_for(ss.Plan(4, 4)) == "dflash2" and k3.drafter_for(ss.Plan(4, 4), long=True) == "mtp"
    assert k3.drafter_for(ss.Plan(8, 4)) == "mtp" and k3.widths_for("dflash2") == [1, 2, 4]


def test_hybrid_user_grows_past_the_context_limit_switches_once():
    """One user at (1,8) DFlash2 with full prefixes pending grows past QWEN36_SPEC_DFLASH2_MAX_CTX mid-generation:
    the state machine runs ONE flush step of its own at (1,8) (the pending 7 rows do not fit the (1,4) plan), the next
    step is (1,4) with the MTP head, and the mode never changes again while the user lives (no flapping); the
    stream stays the greedy one."""
    lad = ss.Ladder.for_drafter("hybrid", 7, 32, dflash2_max_ctx=64, ctx_hysteresis=16)
    sim = Sim(lad, 32, "oracle", seed=5)
    sim.submit("r0", list(range(1, 21)), want=300)  # 20-token prompt: crosses 64 after ~44 generated tokens
    log = []
    while sim.live_count() or sim.pending_join:
        sp = sim.step()
        log.append((sp.plan, sp.drafter, sp.flush, sp.migrate, sp.long, sp.ctx_max, sim.state.long, sp.ctx_flush))
    sim.check_streams()
    st = sim.state.stats
    assert st["ctx_switches"] == 1 and st["ctx_flushes"] == 1 and st["drafter_switches"] == 1, st
    assert st["flushes"] == 1 and st["migrations"] == 0 and st["plain_steps"] == 0, st
    # before the crossing: (1,8) DFlash2 in short mode; the crossing step: the machine is in long mode already but
    # the flush runs at the short mode's (1,8) plan with the DFlash2 drafter (StepPlan.long = the plan's mode);
    # after it: (1,4) MTP for good
    i = next(i for i, e in enumerate(log) if e[7])
    assert all(e[0] == ss.Plan(1, 8) and e[1] == "dflash2" and not e[2] and not e[4] and not e[6] for e in log[:i])
    assert log[i][:5] == (ss.Plan(1, 8), "dflash2", True, False, False) and log[i][6] and log[i][5] > 64
    assert all(e[0] == ss.Plan(1, 4) and e[1] == "mtp" and not e[2] and e[4] and e[6] for e in log[i + 1 :])
    assert not log[i + 1][3], "nothing pending after the flush: no migration"
    assert log[i - 1][5] <= 64 < log[i][5]
    # with the pending rows fitting (random drafts accept little) the switch needs no flush: a plain migration
    lad2 = ss.Ladder.for_drafter("hybrid", 7, 32, dflash2_max_ctx=64, ctx_hysteresis=16)
    sim2 = Sim(lad2, 32, "random", seed=6)
    sim2.submit("r0", list(range(1, 21)), want=120)
    while sim2.live_count() or sim2.pending_join:
        sim2.step()
    sim2.check_streams()
    st2 = sim2.state.stats
    assert (
        st2["ctx_switches"] == 1 and st2["drafter_switches"] == 1 and st2["plan_changes"] == 2
    ), st2  # None -> (1,8) -> (1,4)
    assert st2["ctx_flushes"] <= 1, st2  # a flush only when rows > 3 were pending at the crossing


def test_hybrid_long_prompt_admission_into_the_dflash2_band_and_hysteresis_on_the_way_back():
    """Two short users at (2,8) DFlash2 with 7 pending rows; a LONG-prompt user is admitted at slot 2 (w_grid 3 ->
    the (4,8) bucket: not a width crossing, so the scheduler does not hold it): the state machine flushes at (4,8)
    with the new user on the grid, then runs (4,4) with the MTP head. When the long user leaves, the longest context
    is far below the limit minus the hysteresis: back to (2,8) DFlash2 by migration (the MTP rows <= 3 fit). A
    second long user whose context sits inside the hysteresis band keeps the MTP head after the first one leaves."""
    lad = ss.Ladder.for_drafter("hybrid", 7, 32, dflash2_max_ctx=200, ctx_hysteresis=50)
    sim = Sim(lad, 32, "oracle", seed=7)
    sim.submit("r0", [1, 2, 3], want=400)
    sim.submit("r1", [4, 5, 6], want=400)
    sim.step()
    sim.step()
    assert sim.state.plan == ss.Plan(2, 8) and sim.state.max_pending() == 7 and not sim.state.long
    h = sim.state.hold_info()
    assert h.pending_any and h.slots_before_crossing == 2  # 2 free slots below the T=8 band's width 4
    sim.submit("long", list(range(1, 301)), want=40)  # 300-token prompt > 200
    sp = sim.step()
    assert "long" in sim.owner and sp.plan == ss.Plan(4, 8) and sp.drafter == "dflash2" and sp.flush and sp.ctx_flush
    assert sp.migrate, "the (2,8) -> (4,8) flush carries the pending rows within the T=8 band"
    assert sim.state.long and not sp.long, "the machine is in long mode; the flush runs at the short mode's plan"
    assert sp.ctx_max == 300 and sp.live == [True, True, True, False] and sp.n_drafts == [0, 0, 0, 0]
    assert not sim.state.pending_any
    sp = sim.step()
    assert sp.plan == ss.Plan(4, 4) and sp.drafter == "mtp" and not sp.flush and not sp.migrate and sp.long
    assert sim.state.stats["ctx_switches"] == 1 and sim.state.stats["ctx_flushes"] == 1
    h = sim.state.hold_info()
    assert h.T == 4 and h.band_max == 8  # the long ladder's T=4 band
    while "long" in sim.owner:
        sp = sim.step()
        assert sp.plan == ss.Plan(4, 4) and sp.drafter == "mtp"
    sp = sim.step()  # 2 users again, longest context ~20 < 150: back to DFlash2 by migration
    assert sp.plan == ss.Plan(2, 8) and sp.drafter == "dflash2" and sp.migrate and not sp.flush and not sp.long
    assert sim.state.stats["ctx_switches"] == 2 and sim.state.stats["drafter_switches"] == 2
    # hysteresis: a 180-token user (inside (150, 200]) plus a 260-token user; when the 260 one leaves the 180 one
    # keeps the grid in long mode, and it flips back only when that one is gone too
    sim.submit("mid", list(range(1, 181)), want=120)
    sim.submit("big", list(range(1, 261)), want=30)
    for _ in range(2):
        sim.step()
    assert sim.state.long and sim.state.plan == ss.Plan(4, 4)
    while "big" in sim.owner:
        sim.step()
    assert sim.state.long, "the 180 + generated context sits above the limit minus the hysteresis"
    for _ in range(3):
        assert sim.step().drafter == "mtp"
    sim.req["mid"]["want"] = len(sim.req["mid"]["stream"]) - 1
    sim.step()
    sp = sim.step()
    assert not sim.state.long and sp.drafter == "dflash2" and sp.plan == ss.Plan(2, 8)
    for _ in range(4):
        sim.step()
    sim.check_streams()
    assert sim.state.stats["ctx_switches"] == 4, sim.state.stats


@pytest.mark.parametrize("policy", ["mixed", "oracle", "random"])
def test_hybrid_context_rule_never_flaps_between_ownership_changes(policy):
    """Random churn of short and long prompts through the 3 -> 5 -> 9 -> 17 ramp with a small context limit: the
    mode can change only when the set of live requests changes or a live user grows past the limit -- between two
    consecutive steps with the same owners the mode changes at most once, and never twice in a row -- every switch is
    a legal transition (flush before a shrinking T with pending rows, migration otherwise), every stream greedy."""
    rng = random.Random(11 + len(policy))
    lad = ss.Ladder.for_drafter("hybrid", 7, 32, dflash2_max_ctx=160, ctx_hysteresis=32)
    sim = Sim(lad, 32, policy, seed=3)
    # (step, [(prompt tokens, wanted tokens)]): short prompts never cross 160, the 100-token prompts cross it
    # mid-generation, 170 / 200 are past the limit on admission; the last wave outlives every long user
    waves = [
        (0, [(100, 90), (4, 200)]),  # two users in the DFlash2 band: the 100-token one crosses 160 by growth
        (100, [(200, 40), (170, 60), (8, 50), (4, 90)]),  # past the limit on admission (the first user left)
        (102, [(8, rng.randrange(40, 100)) for _ in range(8)]),
        (130, [(4, 100), (8, 100)]),  # outlive every long user: back to short mode at the end
    ]
    n, submitted = 0, {}
    for at, reqs in waves:
        for plen, want in reqs:
            submitted.setdefault(at, []).append((f"r{n}", plen, want))
            n += 1
    step, owners_prev, long_prev, growth_flips = 0, None, False, 0
    modes = []
    while submitted or sim.pending_join or sim.live_count() > 0:
        for rid, plen, want in submitted.pop(step, []):
            sim.submit(rid, [rng.randrange(50) for _ in range(plen)], want=want)
        if sim.live_count() == 0 and not sim.pending_join:
            step += 1
            continue
        sim.step()
        owners = tuple(sim.last_owners)  # the rows the rule saw this step (admissions in, departures not yet out)
        mode = sim.state.long
        if owners == owners_prev and mode != long_prev:
            growth_flips += 1
            assert mode, "a flip back to short mode needs a departure (positions only grow)"
        owners_prev, long_prev = owners, mode
        modes.append(mode)
        step += 1
        assert step < 3000
    sim.check_streams()
    st = sim.state.stats
    assert st["ctx_switches"] == 4 and set(modes) == {True, False}, (st, set(modes))
    assert growth_flips >= 1, "the 100-token user grew past the limit with the owners unchanged"
    assert st["drafter_switches"] >= 2 and st["plain_steps"] == 0, st
    # no two consecutive flips (a flip needs a departure or a growth crossing in between)
    flips = [i for i in range(1, len(modes)) if modes[i] != modes[i - 1]]
    assert flips and all(b - a > 1 for a, b in zip(flips, flips[1:])), flips
    assert st["ctx_switches"] == len(flips) + (1 if modes[0] else 0), (st, flips, modes[:3])
    if policy == "oracle":
        assert st["ctx_flushes"] >= 1, st


def test_hybrid_switch_4_to_5_is_held_for_a_flush_and_5_to_4_migrates():
    """4 users at (4,T=8) with pending DFlash2 prefixes: the 5th user crosses into the MTP (8,4) band -- a_s up to 7
    does not fit T-1 = 3, so the scheduler holds it for a flush (hold_info: slots_before_crossing = 0), the next step is
    the (8,4) MTP plan (a drafter switch), and when a user leaves the batch returns to (4,8) DFlash2 by migration."""
    lad = ss.Ladder.for_drafter("hybrid", 7, 32)
    sim = Sim(lad, 32, "oracle", seed=3)
    for i in range(4):
        sim.submit(f"r{i}", [i + 1, i + 2, i + 3], want=400)
    sim.step()
    sim.step()  # oracle drafts: everyone accepted 7 -> pending 7 rows
    assert sim.state.plan == ss.Plan(4, 8) and sim.state.max_pending() == 7
    h = sim.state.hold_info()
    assert h.pending_any and h.T == 8 and h.band_max == 4 and h.slots_before_crossing == 0
    sim.submit("r4", [9, 9, 9], want=400)
    sp = sim.step()
    assert sp.flush and sp.plan == ss.Plan(4, 8) and sp.drafter == "dflash2" and "r4" in sim.pending_join
    assert not sim.state.pending_any
    sp = sim.step()
    assert sp.plan == ss.Plan(8, 4) and sp.drafter == "mtp" and not sp.migrate and "r4" in sim.owner
    assert sim.state.stats["drafter_switches"] == 1 and sim.state.stats["flushes"] == 1
    sim.step()  # pending 3 rows everywhere under the MTP plan
    assert sim.state.max_pending() == 3
    # the newest user leaves: back to 4 users -> (4,8) DFlash2 with the pending MTP-band rows migrated (3 <= 7)
    sim.req["r4"]["want"] = len(sim.req["r4"]["stream"]) - 1
    sim.step()  # r4 reaches its budget at the end of this step and is released
    assert "r4" not in sim.owner
    sp = sim.step()
    assert sp.plan == ss.Plan(4, 8) and sp.drafter == "dflash2" and sp.migrate and not sp.flush
    assert sim.state.stats["drafter_switches"] == 2 and sim.state.stats["migrations"] == 1
    for _ in range(6):
        sim.step()
    sim.check_streams()


def test_hybrid_switch_16_to_17_flushes_to_plain_and_back():
    """16 users at (16,T=2) MTP with pending rows: the 17th crosses into plain decode (a_s = 1 never fits), held for a
    flush; the drain from 17 back to 16 re-enters the (16,2) MTP band (no pending after plain steps), and down to 4
    users lands in the DFlash2 band by migration."""
    lad = ss.Ladder.for_drafter("hybrid", 7, 32)
    sim = Sim(lad, 32, "oracle", seed=4)
    for i in range(16):
        sim.submit(f"r{i}", [i + 1, i + 2], want=60)
    sim.step()
    sim.step()
    assert sim.state.plan == ss.Plan(16, 2) and sim.state.max_pending() == 1
    assert sim.state.hold_info().slots_before_crossing == 0
    sim.submit("r16", [7, 7], want=8)
    sp = sim.step()
    assert sp.flush and sp.plan == ss.Plan(16, 2) and sp.drafter == "mtp"
    sp = sim.step()
    assert sp.mode == "plain" and sp.drafter is None and "r16" in sim.owner
    n_plain = 0
    while "r16" in sim.owner:
        assert sim.step().mode == "plain"
        n_plain += 1
    sp = sim.step()
    assert sp.mode == "spec" and sp.plan == ss.Plan(16, 2) and sp.drafter == "mtp" and not sp.migrate
    for rid in [f"r{i}" for i in range(4, 16)]:
        sim.req[rid]["want"] = len(sim.req[rid]["stream"]) - 1
    sim.step()  # they leave at the end of this step
    sp = sim.step()
    assert sp.plan == ss.Plan(4, 8) and sp.drafter == "dflash2" and sp.migrate
    while sim.live_count():
        sim.step()
    sim.check_streams()
    st = sim.state.stats
    assert st["plain_steps"] == n_plain + 1 and st["drafter_switches"] >= 1 and st["flushes"] >= 1


@pytest.mark.parametrize("policy", ["mixed", "oracle", "random"])
@pytest.mark.parametrize("seed", [1, 2])
def test_hybrid_ramp_3_5_9_17_and_back(policy, seed):
    """The hybrid scenario of the device harness: 3 -> 5 -> 9 -> 17 users and the drain -- the DFlash2 (4,8) band, the
    MTP (8,4) / (10,3) / (16,2) bands and plain decode are all visited, every drafter change is a plan change taken
    through the flush / migration protocol, every stream stays the greedy one."""
    rng = random.Random(seed)
    lad = ss.Ladder.for_drafter("hybrid", 7, 32)
    sim = Sim(lad, 32, policy, seed=seed)
    waves = [(0, 3), (2, 2), (4, 4), (6, 8), (60, 2)]  # tight enough for 17 live users at 8 tokens per oracle step
    n, wants, submitted = 0, {}, {}
    for at, cnt in waves:
        for _ in range(cnt):
            wants[f"r{n}"] = rng.randrange(60, 100)
            submitted.setdefault(at, []).append(f"r{n}")
            n += 1
    step = 0
    while submitted or sim.pending_join or sim.live_count() > 0:
        for rid in submitted.pop(step, []):
            sim.submit(rid, [rng.randrange(50) for _ in range(3 + rng.randrange(4))], want=wants[rid])
        if sim.live_count() == 0 and not sim.pending_join:
            step += 1
            continue
        sp = sim.step()
        if sp.mode == "spec":
            assert sp.plan.R <= 32 and sp.drafter == lad.drafter_for(sp.plan)
        step += 1
        assert step < 3000
    sim.check_streams()
    log = _drafter_log(sim, lad)
    drafters_seen = {d for m, d, *_ in log if m == "spec"}
    plans = {p for m, d, p, *_ in log if p is not None}
    assert drafters_seen == {"dflash2", "mtp"}, drafters_seen
    # the T=2 band is reached on the way down only when slot 16 frees before the others (stable slots: w_grid = the
    # highest live slot + 1); test_hybrid_switch_16_to_17_flushes_to_plain_and_back drives that case explicitly
    assert {p.T for p in plans} >= {8, 4}, plans
    st = sim.state.stats
    assert st["plain_steps"] >= 1 and st["drafter_switches"] >= 1, st
    # a drafter change between consecutive spec steps is a plan change: either the pending rows fit (migration) or
    # nothing was pending (a flush ran before, or nothing was accepted)
    prev = None
    for m, d, p, fl, mg in log:
        if m == "spec" and prev is not None and prev[0] == "spec" and prev[1] != d:
            assert p != prev[2]
        prev = (m, d, p)
    if policy == "oracle":
        assert st["flushes"] >= 2, st  # the 4 -> 5 and 8 -> 9 (and 16 -> 17) crossings with full prefixes pending
        assert st["migrations"] >= 1 and st["drafter_switches"] >= 2, st  # the drain's band-down changes


def test_fresh_slot_state_callback_alias():
    """``has_state`` is the drafter-neutral name of ``has_hidden`` (MTP hidden row / DFlash2 context K/V)."""
    st = ss.SpecServingState(ss.Ladder.for_drafter("dflash2", 7, 8), 8)
    rows = ["a", None, "b", None, None, None, None, None]
    sp = st.begin_step(
        rows, [[1, 2], None, [], None, None, None, None, None], eligible=True, has_state=lambda s: s == 2
    )
    assert (
        sp.plan == ss.Plan(4, 8) and sp.catchup == [2] and sp.n_drafts == [2, 0, 0, 0] and sp.drafts[0][:3] == [1, 2, 0]
    )
    st.commit(sp, torch.zeros(32, dtype=torch.int64), st.grid_tokens(sp, [5, 0, 6, 0]))
    sp2 = st.begin_step(rows, [[1], None, [], None, None, None, None, None], eligible=True, has_hidden=lambda s: True)
    assert sp2.catchup == []  # both slots have run a step


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
        self.last_owners = None  # the slot owners begin_step saw on the last step

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
        ctx = [self.req[o]["pos"] if o is not None else -1 for o in self.owner]  # the rows' decode positions
        self.last_owners = list(self.owner)
        sp = self.state.begin_step(row_req_ids, drafts, eligible, flush=flush, ctx_lens=ctx)
        if sp.mode == "spec":
            assert sp.drafter == self.state.ladder.drafter_for(sp.plan, sp.long)
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
        self.step_log.append((sp.mode, sp.plan, sp.flush, sp.migrate, sorted(committed_by_slot), sp.drafter))
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
