# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``tt/verify_plan.py``: T64's host side (``docs/p5_t64/P5_T64_DESIGN.md`` §2.2, §4.5, §4.7, §7.1; review edits R-E3,
R-E6, R-E9). Host only, no device.

* :func:`crossover_lanes`: the c* table of §7.1, including the R-E3 guard (``alpha <= r - 1`` gives 33, never), the
  clamp to [17, 33], and on a grid the defining inequality (T64 with every lane drafting beats the packed verify from
  c* on), monotonicity and the refusals.
* :func:`drafts_all_lanes`: ``packed`` / ``wide`` / ``auto`` around c*, the prior when nothing was verified yet
  (``None`` gives alpha_0; a fresh server's first 32-request burst drafts), the ``MOTIF3_WIDE_MIN_LANES`` override,
  and the smoothed running acceptance of ``generator_api.smoothed_acceptance``.
* :func:`plan_wide_step`: hand cases (idle lanes, positions at ``max_model_len - 2``, same-tile ``p`` / ``p + 1``, a
  draft across a block seam, both KV modes and cache dtypes) and refusals. Over random batches it accepts and refuses
  exactly what the T32 ``generator.plan_spec_step`` accepts and refuses, with the same messages.
* :meth:`WideStepPlan.result`: the split-order mapping by hand. Under a row-local fake model the T64 step returns the
  same ``SpecDecodeResult`` as the T32 packed verify (pass 1 + overflow pass) for every occupancy and both KV modes.
* :func:`choose_verify_kind`: ordinary -> spec, fits -> spec, overflow -> wide, logits / sampling -> spec, ``packed`` /
  ``wide`` fixed. In ``auto``, "spec" holds exactly when ``generator.plan_spec_step`` needs one T32 pass.
* the module is pure host: no ttnn import, nothing from the generator.

Run::

    S=/home/ttuser/hchang/experiments/motif-3/scripts
    $S/hostrun.sh -n verify_plan -- python -m pytest -p no:cacheprovider -q \\
        models/demos/motif3/tests/unit/test_verify_plan.py
"""

from __future__ import annotations

import ast
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.motif3.tt import kv_write as KW
from models.demos.motif3.tt import verify_plan as VP
from models.demos.motif3.tt.generator_api import (
    DEFAULT_SPEC_ALPHA_PRIOR,
    GeneratorSettings,
    SpecDecodeBatch,
    SpecDecodeResult,
    smoothed_acceptance,
)
from models.demos.motif3.tt.model_config import DEFAULT_HF_META_DIR, MotifTTConfig

HF_META = str(DEFAULT_HF_META_DIR)
N, LPR, DP, BS = 32, 8, 4, 64
R = 1.126  # the T64 / T32 step ratio of the design's c* table (4K context)


def spec_cfg(**kw) -> MotifTTConfig:
    """A speculating KV-R launch's config (``all_split``), host only; ``spec_verify="auto"`` stages T64."""
    kw = {"spec_tokens": 1, "spec_verify": "auto", "kv_replicated_decode": True, **kw}
    return MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8), **kw)


def plan_spec_step(*args, **kw):
    """``generator.plan_spec_step`` (the T32 plan), imported lazily: the generator imports the device modules."""
    from models.demos.motif3.tt.generator import plan_spec_step as fn

    return fn(*args, **kw)


def make_batch(seed: int, live: int, *, W: int = 16, p_draft: float = 0.9, spread: bool = True) -> SpecDecodeBatch:
    """A valid owner-lane batch with ``live`` active lanes (spread over the DP rows, or the first ``live`` lanes),
    drafts on a fraction of them, every lane on its own ``W`` distinct blocks (block 0 = null), positions with ``n + 1
    < 64 W``, anchors at the tile and block seams on odd lanes."""
    g = torch.Generator().manual_seed(seed)
    lanes = torch.randperm(N, generator=g)[:live].tolist() if spread else list(range(live))
    pt = torch.zeros(N, W, dtype=torch.int32)
    blocks = (torch.randperm(N * W, generator=g) + 1).to(torch.int32).reshape(N, W)
    pos = torch.full((N,), -1, dtype=torch.int32)
    dr = torch.full((N,), -1, dtype=torch.int32)
    tok = torch.zeros(N, dtype=torch.int32)
    for lane in lanes:
        pt[lane] = blocks[lane]
        if lane % 2:
            pos[lane] = BS * int(torch.randint(0, W - 1, (1,), generator=g)) + (0, 30, 31, 62, 63)[lane % 5]
        else:
            pos[lane] = int(torch.randint(0, BS * W - 1, (1,), generator=g))
        tok[lane] = int(torch.randint(0, 220160, (1,), generator=g))
        if float(torch.rand(1, generator=g)) < p_draft:
            dr[lane] = int(torch.randint(0, 220160, (1,), generator=g))
    return SpecDecodeBatch(tokens=tok, positions=pos, draft_tokens=dr, page_table=pt)


# ======================================================================================================================
# c* (R-E3)
# ======================================================================================================================
def unguarded(alpha: float, r: float) -> int:
    return math.ceil(32 * alpha * r / ((1 + alpha) - (1 - alpha) * r))


@pytest.mark.parametrize(
    "alpha, ratio, want",
    [
        (0.0, R, 33),  # the unguarded formula gives 0: the bridge would draft every lane with nothing accepted
        (0.05, R, 33),  # unguarded: -91 (denominator < 0)
        (0.061, R, 33),  # unguarded: ~ +600 (denominator ~ 0+)
        (R - 1 - 1e-6, R, 33),  # just below the break-even alpha = r - 1
        (0.2, R, 25),
        (0.88, R, 19),
        (1.0, R, 19),
        (1.0, 1.0, 17),  # clamped from 16
        (0.85, 1.13, 19),  # the defaults: prior alpha_0, MotifTTConfig.wide_step_ratio
    ],
)
def test_crossover_lanes_table(alpha, ratio, want):
    assert VP.crossover_lanes(alpha, ratio) == want


def test_crossover_lanes_guard_and_clamp():
    assert unguarded(0.0, R) == 0 and unguarded(0.05, R) == -91 and unguarded(0.061, R) > 500
    assert unguarded(1.0, 1.0) == 16 and VP.CROSSOVER_MIN_LANES == 17 and VP.CROSSOVER_NEVER == 33
    ratios = (1.0, 1.05, 1.1, 1.118, 1.126, 1.13, 1.173, 1.2, 1.3, 1.5, 2.0)
    alphas = [i / 100 for i in range(101)]
    for r in ratios:
        prev = None
        for a in alphas:
            c = VP.crossover_lanes(a, r)
            assert 17 <= c <= 33, (a, r, c)
            if a <= r - 1:
                assert c == 33, (a, r, c)
            if prev is not None:
                assert c <= prev, ("c* never grows with the acceptance", a, r)
            prev = c

            def gain(lanes):  # T64 tokens / time minus packed tokens / time (T32 step = 1, T64 step = r)
                return lanes * (1 + a) / r - (lanes + (32 - lanes) * a)

            for lanes in range(17, 33):
                if lanes >= c:
                    assert gain(lanes) >= -1e-9, ("T64 wins or ties from c* on", a, r, c, lanes)
                else:
                    assert gain(lanes) < 1e-9, ("the packed verify wins below c*", a, r, c, lanes)
    for a in (0.3, 0.6, 0.9, 1.0):
        cs = [VP.crossover_lanes(a, r) for r in ratios]
        assert cs == sorted(cs), ("c* never shrinks with the step ratio", a, cs)
    for bad in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            VP.crossover_lanes(bad, R)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            VP.crossover_lanes(0.9, bad)
    for a, r in ((True, R), (None, R), (0.9, True), (0.9, None)):
        with pytest.raises(TypeError):
            VP.crossover_lanes(a, r)


def test_drafts_all_lanes():
    prior, r = DEFAULT_SPEC_ALPHA_PRIOR, 1.13
    assert GeneratorSettings().spec_alpha_prior == prior and spec_cfg().wide_step_ratio == r
    c_star = VP.crossover_lanes(prior, r)
    assert c_star == 19
    for live in (range(0), range(1), range(16), range(32)):
        assert VP.drafts_all_lanes(live, spec_verify="packed", ratio=r) is False
        assert VP.drafts_all_lanes(live, spec_verify="wide", ratio=r) is True
    # auto, nothing verified yet: the prior alpha_0 decides (R-E3)
    assert not VP.drafts_all_lanes(range(c_star - 1), spec_verify="auto", ratio=r)
    assert VP.drafts_all_lanes(range(c_star), spec_verify="auto", ratio=r)
    # a fresh server whose first traffic is a 32-request burst drafts every lane from its first verify
    assert VP.drafts_all_lanes(range(32), spec_verify="auto", ratio=r, acceptance=None)
    a0 = smoothed_acceptance(0, 0)
    assert a0 == prior and VP.drafts_all_lanes(range(32), spec_verify="auto", ratio=r, acceptance=a0)
    # the running estimate: a well-accepted workload keeps drafting at c = 32; one below r - 1 stops (never at c <= 32)
    good = smoothed_acceptance(9000, 10000)
    assert VP.drafts_all_lanes(range(32), spec_verify="auto", ratio=r, acceptance=good)
    bad = smoothed_acceptance(100, 10000)
    assert bad < r - 1 and not VP.drafts_all_lanes(range(32), spec_verify="auto", ratio=r, acceptance=bad)
    mid = smoothed_acceptance(500, 1000)  # ~0.52: c* = 20
    c_mid = VP.crossover_lanes(mid, r)
    assert c_mid == 20
    assert not VP.drafts_all_lanes(range(c_mid - 1), spec_verify="auto", ratio=r, acceptance=mid)
    assert VP.drafts_all_lanes(range(c_mid), spec_verify="auto", ratio=r, acceptance=mid)
    # MOTIF3_WIDE_MIN_LANES overrides c* (33 = never)
    for m in (1, 17, 24, 32, 33):  # acceptance 0 alone would give 33 (never): the override wins
        for live in (m - 1, m, 32):
            if 0 <= live <= 32:
                got = VP.drafts_all_lanes(range(live), spec_verify="auto", ratio=r, acceptance=0.0, min_lanes=m)
                assert got is (live >= m), (m, live)
    # lanes are a set of lane ids: duplicates count once
    assert not VP.drafts_all_lanes([0] * 40 + list(range(17)), spec_verify="auto", ratio=r, min_lanes=18)
    assert VP.drafts_all_lanes(torch.arange(18).tolist() * 2, spec_verify="auto", ratio=r, min_lanes=18)
    for live, err in (([32], ValueError), ([-1], ValueError), ([1.5], ValueError), ([True], TypeError)):
        with pytest.raises(err):
            VP.drafts_all_lanes(live, spec_verify="auto", ratio=r)
    with pytest.raises(ValueError, match="spec_verify"):
        VP.drafts_all_lanes(range(4), spec_verify="t64", ratio=r)
    with pytest.raises(ValueError):
        VP.drafts_all_lanes(range(4), spec_verify="auto", ratio=r, prior=1.5)
    with pytest.raises(ValueError):
        VP.drafts_all_lanes(range(4), spec_verify="auto", ratio=r, min_lanes=34)


# ======================================================================================================================
# plan_wide_step
# ======================================================================================================================
def test_plan_wide_step_hand_cases():
    cfg = spec_cfg()
    assert cfg.kv_write_mode == "all_split" and cfg.wide_rows_per_dp == 16
    L, W = cfg.max_model_len, 512
    g = torch.Generator().manual_seed(5)
    pt = (torch.randperm(N * W, generator=g) + 1).to(torch.int32).reshape(N, W)
    pos = torch.full((N,), -1, dtype=torch.int32)
    tok = torch.zeros(N, dtype=torch.int32)
    dr = torch.full((N,), -1, dtype=torch.int32)
    cases = {  # lane: (anchor position n, anchor token, draft or -1)
        0: (L - 2, 11, 12),  # the last legal drafted position: n + 1 = max_model_len - 1
        2: (100, 21, -1),  # active, no draft
        3: (64 * 7 + 30, 31, 32),  # p, p + 1 in one tile (the G12 race the split avoids)
        9: (64 * 9 + 63, 91, 92),  # the draft crosses a block seam (page-table entry n // 64 + 1)
        13: (64 * 3 + 31, 131, 132),  # the draft crosses a tile seam
        31: (L - 1, 311, -1),  # the last legal undrafted position
    }
    for lane, (p, t, d) in cases.items():
        pos[lane], tok[lane], dr[lane] = p, t, d
    pt[[lane for lane in range(N) if lane not in cases]] = 0  # idle lanes: all-zero rows
    batch = SpecDecodeBatch(tokens=tok, positions=pos, draft_tokens=dr, page_table=pt)
    for mode, kv_name, gather, lpc in (
        ("all_split", "bfp8", "split", 32),
        ("all_split", "bf16", "split", 16),
        ("all_split", "bfp8", "natural", 32),
        ("row_split", "bfp8", "split", None),
    ):
        c = spec_cfg(kv_cache_dtype=kv_name, kv_replicated_decode=mode == "all_split")
        assert c.kv_write_mode == mode
        plan = VP.plan_wide_step(batch, cfg=c, width=W, num_blocks=N * W + 1, gather=gather)
        assert isinstance(plan, VP.WideStepPlan) and plan.mode == mode and plan.rows_per_dp == 16
        assert plan.gather == (gather if mode == "all_split" else "natural")
        assert plan.is_verify and plan.num_drafts == 4 and plan.drafted_lanes == (0, 3, 9, 13)
        st = plan.step
        assert st.lanes == 64 and st.width == W and torch.equal(plan.positions, st.positions)
        assert plan.tokens.dtype == torch.int32 and tuple(plan.tokens.shape) == (64,)
        for lane in range(N):
            ra, rd = KW.wide_rows(lane)
            p, t, d = cases.get(lane, (-1, 0, -1))
            assert int(st.positions[ra]) == p and int(plan.tokens[ra]) == t
            if lane in cases:
                assert torch.equal(st.page_table[ra], pt[lane])
            if d >= 0:
                assert int(st.positions[rd]) == p + 1 and int(plan.tokens[rd]) == d and bool(st.call_b[rd])
                assert int(st.owner[rd]) == ra and torch.equal(st.page_table[rd], pt[lane])
            else:
                assert int(st.positions[rd]) == -1 and int(plan.tokens[rd]) == 0 and not bool(st.call_b[rd])
        KW.check_wide_layout(st)
        KW.check_kv_write_step(st, mode, block_size=BS, max_seq_len=L, lanes_per_call=lpc, lanes_per_row=16,
                               gather=gather)  # fmt: skip
    # an ordinary step (no drafts) gives the wide mode's T64 step with idle draft rows
    plan = VP.plan_wide_step(SpecDecodeBatch.from_decode_batch(batch.anchors()), cfg=cfg)
    assert not plan.is_verify and plan.num_drafts == 0 and int(plan.step.positions[8:16].max()) == -1


def _refusal(batch, cfg, match, err=ValueError, **kw):
    with pytest.raises(err, match=match):
        VP.plan_wide_step(batch, cfg=cfg, **kw)


def test_plan_wide_step_refusals():
    cfg = spec_cfg()
    L, V = cfg.max_model_len, cfg.vocab_size
    b = make_batch(3, 24, p_draft=0.5)
    assert bool(b.has_draft.any()) and bool((b.active & ~b.has_draft).any())
    _refusal(b.anchors(), cfg, "SpecDecodeBatch", TypeError)
    for mode in ("row", "all"):
        _refusal(b, cfg, "split KV-write mode", mode=mode)
    _refusal(b, cfg, "gather order", gather="interleaved")
    _refusal(b, cfg, "width", width=b.page_table_width + 2)

    def edit(**ch):
        f = {k: getattr(b, k).clone() for k in ("tokens", "positions", "draft_tokens", "page_table")}
        for k, fn in ch.items():
            fn(f[k])
        return SpecDecodeBatch(**f)

    lane = int(torch.nonzero(b.has_draft)[0])
    und = int(torch.nonzero(b.active & ~b.has_draft)[0])
    big = torch.zeros(N, L // BS, dtype=torch.int32)
    big[:, : b.page_table_width] = b.page_table
    big[:, b.page_table_width :] = torch.arange(b.page_table_width, L // BS, dtype=torch.int32) + 10_000
    far = SpecDecodeBatch(b.tokens, b.positions, b.draft_tokens, big)
    _refusal(SpecDecodeBatch(far.tokens, far.positions.clone().index_fill_(0, torch.tensor([lane]), L - 1),
                             far.draft_tokens, far.page_table), cfg, "max_model_len")  # fmt: skip
    _refusal(SpecDecodeBatch(far.tokens, far.positions.clone().index_fill_(0, torch.tensor([und]), L),
                             far.draft_tokens, far.page_table), cfg, "max_model_len")  # fmt: skip
    _refusal(edit(tokens=lambda t: t.__setitem__(lane, V)), cfg, "token ids")
    _refusal(edit(draft_tokens=lambda t: t.__setitem__(lane, V)), cfg, "token ids")
    p = int(b.positions[lane])
    _refusal(edit(page_table=lambda t: t.__setitem__((lane, p // BS), 0)), cfg, "null block")
    _refusal(edit(page_table=lambda t: t.__setitem__((lane, 0), 0)), cfg, "null block")  # a read entry too
    _refusal(b, cfg, "block id", num_blocks=int(b.page_table.max()))
    _refusal(edit(positions=lambda t: t.__setitem__(lane, BS * b.page_table_width - 1)), cfg, "width")
    # two lanes writing one block (vLLM never shares a block being written): the KV-write race check refuses it
    other = next(x for x in range(N) if bool(b.active[x]) and x != lane)
    twin = edit(page_table=lambda t: t.__setitem__(other, t[lane].clone()), positions=lambda t: t.__setitem__(other, p))
    _refusal(twin, cfg, "both write|two rows")


def test_plan_wide_step_checks_match_plan_spec_step():
    """Over random batches (valid ones, and ones broken in one of the shared checks) ``plan_wide_step`` refuses exactly
    what the T32 ``generator.plan_spec_step`` refuses, with the same message."""
    cfg = spec_cfg()
    V = cfg.vocab_size
    g = torch.Generator().manual_seed(11)
    n_ok = n_bad = 0
    for trial in range(300):
        b = make_batch(trial, int(torch.randint(1, N + 1, (1,), generator=g)), W=8)
        f = {k: getattr(b, k).clone() for k in ("tokens", "positions", "draft_tokens", "page_table")}
        kind = trial % 6
        live = torch.nonzero(b.active).reshape(-1).tolist()
        lane = live[int(torch.randint(0, len(live), (1,), generator=g))]
        if kind == 1:
            f["tokens"][lane] = V + int(torch.randint(0, 5, (1,), generator=g))
        elif kind == 2:
            f["draft_tokens"][lane] = V
        elif kind == 3:
            f["page_table"][lane, int(torch.randint(0, int(f["positions"][lane]) // BS + 1, (1,), generator=g))] = 0
        elif kind == 4:
            f["positions"][lane] = BS * 8 - int(torch.randint(0, 2, (1,), generator=g))  # past the width (W = 8)
        bb = SpecDecodeBatch(**f)
        for mode in ("all_split", "row_split"):
            errs = []
            for fn in (lambda: VP.plan_wide_step(bb, cfg=cfg, mode=mode, num_blocks=N * 8 + 1),
                       lambda: plan_spec_step(bb, mode=mode, cfg=cfg, num_blocks=N * 8 + 1)):  # fmt: skip
                try:
                    fn()
                    errs.append(None)
                except ValueError as e:
                    errs.append(str(e))
            assert errs[0] == errs[1], (trial, kind, mode, errs)
            n_ok += errs[0] is None
            n_bad += errs[0] is not None
    assert n_ok > 150 and n_bad > 150, (n_ok, n_bad)


# ======================================================================================================================
# WideStepPlan.result
# ======================================================================================================================
def test_result_mapping_by_hand():
    cfg = spec_cfg()
    b = make_batch(7, 20)
    plan = VP.plan_wide_step(b, cfg=cfg)
    a = torch.arange(64, dtype=torch.int64) + 1000  # split order: [l] anchor row of lane l, [32 + l] its draft row
    m = (torch.arange(64, dtype=torch.int64) + 5000).reshape(1, 1, 1, 64)  # the device's [1, 1, 1, 64] shape
    res = plan.result(a, m)
    assert isinstance(res, SpecDecodeResult) and res.logits is None
    for lane in range(N):
        if not bool(b.active[lane]):
            assert res.argmax[lane].tolist() == [-1, -1] and res.mtp_argmax[lane].tolist() == [-1, -1]
        elif bool(b.has_draft[lane]):
            assert res.argmax[lane].tolist() == [1000 + lane, 1032 + lane]
            assert res.mtp_argmax[lane].tolist() == [5000 + lane, 5032 + lane]
        else:
            assert res.argmax[lane].tolist() == [1000 + lane, -1] and res.mtp_argmax[lane].tolist() == [5000 + lane, -1]
    lg = torch.randn(N, cfg.vocab_size)
    assert plan.result(a, m, logits=lg).logits is lg
    for bad, err in ((torch.arange(32), ValueError), (torch.arange(65), ValueError), (torch.zeros(64), TypeError)):
        with pytest.raises(err):
            plan.result(bad, m)
    with pytest.raises(ValueError):
        plan.result(a, m, logits=torch.randn(N - 1, cfg.vocab_size))


def _row_fn(salt: int):
    """A row-local fake model: each row's output depends only on its token, position and page-table row."""

    def f(tok: int, pos: int, pt_row: torch.Tensor) -> int:
        return (tok * 1_000_003 + pos * 7_919 + int(pt_row.to(torch.int64).sum()) * 31 + salt) % 220_160

    return f


def test_result_matches_the_t32_packed_verify():
    """Lossless contract mapping: when every row's outputs depend only on its own token, position and page-table row
    (the property G-S1w / G-S5w check on device: FlashMLA option A'', the KV write, row-local MoE and heads), the T64
    step returns exactly the ``SpecDecodeResult`` of the T32 packed verify (pass 1 and, when drafts overflow, pass 2),
    for every occupancy and both KV modes."""
    fa, fm = _row_fn(1), _row_fn(2)
    cfg = spec_cfg()
    n_overflow = 0
    for trial in range(120):
        live = (1, 8, 16, 17, 20, 24, 31, 32)[trial % 8]
        b = make_batch(200 + trial, live, W=8, spread=bool(trial % 3))
        for mode in ("all_split", "row_split"):
            t32 = plan_spec_step(b, mode=mode, cfg=cfg)
            outs = []
            for ps in t32.passes:
                pos_l, tok_l = ps.step.positions.tolist(), ps.tokens.tolist()
                a = [fa(tok_l[x], pos_l[x], ps.step.page_table[x]) if pos_l[x] >= 0 else 0 for x in range(N)]
                m = [fm(tok_l[x], pos_l[x], ps.step.page_table[x]) if pos_l[x] >= 0 else 0 for x in range(N)]
                outs.append((torch.tensor(a), torch.tensor(m)))
            n_overflow += len(t32.passes) > 1
            want = t32.result(outs)
            plan = VP.plan_wide_step(b, cfg=cfg, mode=mode)
            order = KW.split_order(64, 16)  # the device's split-order outputs
            pos64, tok64, pt64 = plan.step.positions.tolist(), plan.tokens.tolist(), plan.step.page_table
            a64 = [fa(tok64[r], pos64[r], pt64[r]) if pos64[r] >= 0 else 0 for r in order]
            m64 = [fm(tok64[r], pos64[r], pt64[r]) if pos64[r] >= 0 else 0 for r in order]
            got = plan.result(torch.tensor(a64), torch.tensor(m64))
            assert torch.equal(got.argmax, want.argmax) and torch.equal(got.mtp_argmax, want.mtp_argmax), (trial, mode)
    assert n_overflow > 30


# ======================================================================================================================
# choose_verify_kind
# ======================================================================================================================
def _batch_with(active, drafted, W: int = 4) -> SpecDecodeBatch:
    pos = torch.full((N,), -1, dtype=torch.int32)
    dr = torch.full((N,), -1, dtype=torch.int32)
    pt = torch.zeros(N, W, dtype=torch.int32)
    for lane in active:
        pos[lane] = 10 + lane
        pt[lane] = torch.arange(W, dtype=torch.int32) + 1 + W * lane
    for lane in drafted:
        dr[lane] = 7
    return SpecDecodeBatch(tokens=torch.zeros(N, dtype=torch.int32), positions=pos, draft_tokens=dr, page_table=pt)


def test_choose_verify_kind_cases():
    ordinary = _batch_with(range(32), [])
    half = _batch_with(range(16), range(16))  # 16 drafts, 16 idle lanes on DP rows 2-3 only
    c20 = _batch_with(range(20), range(20))  # 20 drafts, 12 idle lanes
    full = _batch_with(range(32), [5])
    spread = _batch_with([x for x in range(32) if x % 2 == 0], [x for x in range(32) if x % 2 == 0])  # 4 + 4 per row
    for b, kv_mode, want in (
        (ordinary, "all_split", "spec"),
        (half, "all_split", "spec"),  # KV-R: a draft may borrow an idle lane of any DP row
        (half, "row_split", "wide"),  # without KV-R only its own row's idle lanes: rows 0-1 have none
        (spread, "row_split", "spec"),
        (c20, "all_split", "wide"),
        (full, "all_split", "wide"),
        (full, "row_split", "wide"),
    ):
        assert VP.choose_verify_kind(b, "auto", kv_mode=kv_mode) == want, (kv_mode, want)
        fits = want == "spec"
        assert VP.drafts_fit_idle_lanes(b, kv_mode=kv_mode) == fits
        # logits / sampling keep the step on T32 (R-E6: test-made verifies; T32 + the overflow pass, as `packed`)
        assert VP.choose_verify_kind(b, "auto", True, kv_mode=kv_mode) == "spec"
        assert VP.choose_verify_kind(b, "auto", sampling=([1.0] * N, [1.0] * N, [0] * N, [None] * N),
                                     kv_mode=kv_mode) == "spec"  # fmt: skip
        assert VP.choose_verify_kind(b, "auto", sampling=False, kv_mode=kv_mode) == want
        assert VP.choose_verify_kind(b, "packed", kv_mode=kv_mode) == "spec"
        assert VP.choose_verify_kind(b, "wide", True, kv_mode=kv_mode) == "wide"
    assert not VP.drafts_fit_idle_lanes(full, kv_mode="all")  # no split mode: no draft is packed
    assert VP.drafts_fit_idle_lanes(ordinary, kv_mode="all")
    assert VP.VERIFY_STEP_KINDS == ("spec", "wide")
    with pytest.raises(ValueError, match="spec_verify"):
        VP.choose_verify_kind(ordinary, "t64", kv_mode="all_split")
    with pytest.raises(ValueError, match="mode"):
        VP.choose_verify_kind(ordinary, "auto", kv_mode="deferred")
    with pytest.raises(TypeError):
        VP.choose_verify_kind(ordinary.anchors(), "auto", kv_mode="all_split")


def test_wide_min_lanes_sweep_a4prime():
    """A4' (docs/OPTIMIZATION_PLAN.md §3.3, §6 M14): the c* sweep is a pure ``MOTIF3_WIDE_MIN_LANES`` change, and the
    knob does what the sweep assumes, end to end on the host: the env reaches ``GeneratorSettings.wide_min_lanes``; on
    an ``auto`` launch every live lane drafts from that many live lanes on (below the prior's c* = 19 too), never below
    it; and a verify step whose drafts do not fit idle lanes then runs on the T64 trace instead of dropping drafts. The
    8-user case of FINAL_BENCHMARKS anomaly 1: 8 users packed in one DP row of the honest launch (``row_split``, no KV-R)
    have no idle lane in their row, so under the idle-lane budget they cannot draft at all; with the knob at <= 8 they
    all draft and the step goes to T64. With KV-R (``all_split``) the same drafts fit idle lanes of other rows: T32."""
    hf = SimpleNamespace(num_hidden_layers=53)
    kw = dict(max_batch_size=32, max_seq_len=32768)
    c_star = VP.crossover_lanes(DEFAULT_SPEC_ALPHA_PRIOR, R)
    assert c_star == 19
    for raw, want in ((None, None), ("8", 8), ("16", 16), ("1", 1), ("33", 33)):
        env = {"MOTIF3_SPEC_VERIFY": "auto"} | ({} if raw is None else {"MOTIF3_WIDE_MIN_LANES": raw})
        s = GeneratorSettings.from_env(hf, environ=env, serving=dict(spec_tokens=1), **kw)
        assert s.spec_verify == "auto" and s.wide_min_lanes == want, raw
        for live in (1, 7, 8, 15, 16, 18, 19, 32):
            got = VP.drafts_all_lanes(range(live), spec_verify="auto", ratio=R, min_lanes=s.wide_min_lanes)
            assert got is (live >= (c_star if want is None else want)), (raw, live)
    one_row = _batch_with(range(8), range(8))  # 8 live lanes on DP row 0, all drafted: row 0 has no idle lane
    assert VP.choose_verify_kind(one_row, "auto", kv_mode="row_split") == "wide"
    assert VP.choose_verify_kind(one_row, "auto", kv_mode="all_split") == "spec"
    spread = _batch_with([0, 1, 8, 9, 16, 17, 24, 25], [0, 1, 8, 9, 16, 17, 24, 25])  # 2 per row: fits own row
    assert VP.choose_verify_kind(spread, "auto", kv_mode="row_split") == "spec"
    for bad in ("0", "34", "x"):
        with pytest.raises(ValueError, match="MOTIF3_WIDE_MIN_LANES"):
            GeneratorSettings.from_env(hf, environ={"MOTIF3_WIDE_MIN_LANES": bad}, **kw)


def test_choose_verify_kind_matches_the_t32_pass_count():
    """``auto`` picks the T32 trace exactly for the steps ``generator.plan_spec_step`` runs in one T32 pass, so bridge
    traffic never takes the overflow pass (R-E6)."""
    cfg = spec_cfg()
    n = {"spec": 0, "wide": 0}
    for trial in range(400):
        b = make_batch(900 + trial, int(torch.randint(1, N + 1, (1,), generator=torch.Generator().manual_seed(trial))),
                       W=4, p_draft=(0.0, 0.3, 0.7, 1.0)[trial % 4], spread=bool(trial % 2))  # fmt: skip
        for kv_mode in ("all_split", "row_split"):
            kind = VP.choose_verify_kind(b, "auto", kv_mode=kv_mode)
            passes = len(plan_spec_step(b, mode=kv_mode, cfg=cfg).passes)
            assert (kind == "spec") == (passes == 1), (trial, kv_mode, kind, passes)
            n[kind] += 1
    assert n["spec"] > 150 and n["wide"] > 150, n


# ======================================================================================================================
# module hygiene and host cost
# ======================================================================================================================
def test_verify_plan_is_pure_host():
    tree = ast.parse(Path(VP.__file__).read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(("." * node.level) + (node.module or ""))
    assert "ttnn" not in mods and not any(m.endswith("generator") for m in mods), mods
    assert {".generator_api", ".kv_write"} <= mods
    for name in VP.__all__:
        assert hasattr(VP, name), name


def test_plan_wide_step_host_cost():
    """Informational: the T64 plan runs on every T64 step (c = 32, W = 512: the serving width)."""
    cfg = spec_cfg()
    b = make_batch(42, 32, W=512, p_draft=1.0)
    for _ in range(20):
        VP.plan_wide_step(b, cfg=cfg)
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        VP.plan_wide_step(b, cfg=cfg)
        ts.append((time.perf_counter() - t0) * 1e6)
    t32 = []
    for _ in range(200):
        t0 = time.perf_counter()
        plan_spec_step(b, mode="all_split", cfg=cfg)
        t32.append((time.perf_counter() - t0) * 1e6)
    med = sorted(ts)[len(ts) // 2]
    print(f"[verify-plan] plan_wide_step c=32 W=512: median {med:.0f} us (T32 plan_spec_step, same batch with "
          f"overflow: {sorted(t32)[len(t32) // 2]:.0f} us)")  # fmt: skip
