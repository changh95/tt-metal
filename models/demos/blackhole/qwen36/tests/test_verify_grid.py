# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""No-device tests of the verify-step row grid, accept rule and commit (tt/verify_grid.py).

A VerifyController driven by the CPU lazy-prefix oracle must reproduce the plain greedy stream of every user for
ANY drafts (random, the true continuation, mixtures), which is the same claim the device test makes of the traced
step. Run: pytest models/demos/blackhole/qwen36/tests/test_verify_grid.py -q
"""
import random

import pytest
import torch

from models.demos.blackhole.qwen36.tt import verify_grid as vg


def _toy_next(ctx):
    """A deterministic toy LM: depends on the whole prefix (so a wrong prefix changes every later token)."""
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


def test_grid_rows():
    assert vg.grid_rows(1, 8) == 8
    assert vg.grid_rows(4, 8) == 32
    assert vg.grid_rows(8, 8) == 64
    assert vg.grid_rows(32, 4) == 128
    assert vg.grid_rows(32, 8) == 256
    assert vg.grid_rows(5, 8) == 64  # 40 rows -> tile padded
    assert vg.row(3, 2, 8) == 26


def test_accept_drafts_rule(expect_error):
    assert vg.accept_drafts([5, 6, 7, 8], [5, 6, 7]) == 3
    assert vg.accept_drafts([5, 6, 7, 8], [5, 6, 9]) == 2
    assert vg.accept_drafts([5, 6, 7, 8], [1, 6, 7]) == 0
    assert vg.accept_drafts([5, 6, 7, 8], [5, 9, 7]) == 1  # a later match does not count
    assert vg.accept_drafts([5], []) == 0
    with expect_error(ValueError, "verify argmaxes"):
        vg.accept_drafts([5, 6], [5, 6])


def test_row_columns_and_selects():
    T, w = 4, 3
    R = vg.grid_rows(w, T)
    toks = [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
    col = vg.row_tokens(toks, T, R)
    assert col.tolist() == [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]
    pos = vg.row_positions([100, 200, 300], T, R)
    assert pos.tolist() == [100, 101, 102, 103, 200, 201, 202, 203, 300, 301, 302, 303]
    assert vg.offset_positions([100, 200, 300], 2).tolist() == [102, 202, 302]
    sel, selT = vg.select_matrices(w, T, R)
    X = torch.arange(R * 5, dtype=torch.float32).reshape(1, 1, R, 5)
    for j in range(T):
        g = sel[j] @ X
        assert torch.equal(g[0, 0], torch.stack([X[0, 0, s * T + j] for s in range(w)]))
        back = selT[j] @ g
        ref = torch.zeros_like(X)
        for s in range(w):
            ref[0, 0, s * T + j] = X[0, 0, s * T + j]
        assert torch.equal(back, ref)
    # the scatters of all j tile the grid exactly once
    total = sum(selT[j] @ (sel[j] @ X) for j in range(T))
    assert torch.equal(total, X)


def test_accept_onehot_masks(expect_error):
    m = vg.accept_onehot_masks([0, 2, 3], users=3, T=4, bmax=5)
    assert m.shape == (5, 4, 1, 1)
    assert m[0, :, 0, 0].tolist() == [1, 0, 0, 0]
    assert m[1, :, 0, 0].tolist() == [0, 0, 1, 0]
    assert m[2, :, 0, 0].tolist() == [0, 0, 0, 1]
    assert m[3, :, 0, 0].tolist() == [1, 0, 0, 0]  # slots outside the grid keep their state
    assert m[4, :, 0, 0].tolist() == [1, 0, 0, 0]
    with expect_error(AssertionError, "outside"):
        vg.accept_onehot_masks([4], users=1, T=4, bmax=1)


def test_rope_matches_decode_host_packing():
    rd, theta = 64, 10_000_000.0
    pos = torch.tensor([0, 1, 5, 4097], dtype=torch.int32)
    cos, sin = vg.rope_cos_sin(pos, rd, theta)
    inv_freq = 1.0 / (theta ** (torch.arange(0, rd, 2).float() / rd))
    freqs = torch.outer(pos.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    assert torch.equal(cos, emb.cos().reshape(1, 1, 4, rd).to(torch.bfloat16))
    assert torch.equal(sin, emb.sin().reshape(1, 1, 4, rd).to(torch.bfloat16))


@pytest.mark.parametrize("w,T", [(1, 8), (8, 8), (32, 4), (32, 8), (3, 4)])
@pytest.mark.parametrize("policy", ["random", "oracle", "mixed"])
def test_controller_reproduces_greedy_for_any_drafts(w, T, policy):
    rng = random.Random(w * 100 + T * 10 + len(policy))
    prompts = [[rng.randrange(50) for _ in range(5 + (s % 3))] for s in range(w)]
    n_gen = 40
    refs = [_greedy_stream(p, n_gen * T + 2 * T) for p in prompts]  # every user commits <= T per step
    oracle = vg.CpuGreedyOracle(_toy_next, prompts, T)
    ctrl = vg.VerifyController(
        T=T, run=oracle, positions=[len(p) for p in prompts], last=[r[0] for r in refs]  # t_0 from 'prefill'
    )
    k = T - 1
    steps = 0
    while min(len(c) for c in ctrl.committed) < n_gen:
        drafts = []
        for s in range(w):
            n_done = len(ctrl.committed[s])  # committed so far: positions [len(prompt), len(prompt)+n_done)
            true_next = refs[s][n_done : n_done + k]  # the true continuation after the row-0 token
            assert len(true_next) == k
            if policy == "random":
                d = [rng.randrange(50) for _ in range(k)]
            elif policy == "oracle":
                d = list(true_next)
            else:
                m = rng.randrange(k + 1)
                d = list(true_next[:m]) + [rng.randrange(50) for _ in range(k - m)]
            drafts.append(d)
        accepts = ctrl.step(drafts)
        steps += 1
        for s in range(w):
            assert 0 <= accepts[s] <= k
        assert steps < 10 * n_gen
    for s in range(w):
        n = min(len(ctrl.committed[s]), len(refs[s]))
        assert ctrl.committed[s][:n] == refs[s][:n], f"user {s} diverged under {policy} drafts"
        assert len(ctrl.committed[s]) >= n_gen
        assert ctrl.positions[s] == len(prompts[s]) + len(ctrl.committed[s]) - 1
    if policy == "oracle":
        # every draft accepted: k+1 tokens per step
        assert all(a == k for acc in ctrl.accept_history for a in acc)
        assert steps == -(-(n_gen - 1) // T)
    if policy == "random" and T > 1:
        assert steps >= n_gen // 2  # random drafts over 50 ids are almost never accepted
