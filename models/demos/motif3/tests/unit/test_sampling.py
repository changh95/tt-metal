# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact on-device sampler (``tt/sampling.py``; ``docs/sampling/DEVICE_SAMPLER.md``).

CPU tests (no device; host wrapper with the root conftest, which the indirect ``mesh_device`` parametrization of the
device tests needs at collection -- the devices stay hidden)::

    scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q models/demos/motif3/tests/unit/test_sampling.py -k cpu

* ``test_cpu_rng_*``: the numpy ``lowbias32`` / uniform against a pure-Python reference (bit-exact), 24-bit grid,
  uniformity and serial independence of the per-lane counter stream, ``seed_key`` mixing.
* ``test_cpu_lane_params``: vLLM / plugin conventions (``T < 1e-5`` greedy, top-k off = vocab / <= 0, top-p bounds,
  padding slots), the device parameter columns.
* ``test_cpu_reference_vs_vllm``: the fp64 reference kept set (``exact_nucleus``) against a replica of vLLM 0.26's
  ``apply_temperature`` + ``apply_top_k_top_p_pytorch`` (and the real function when importable) on real rows and
  crafted ties: equal sizes, members equal up to the boundary tie group.
* ``test_cpu_emulation_*``: the fp32 emulation of the device pipeline: kept-set size = fp64 on every certified lane,
  the token = the fp64 inverse CDF fed the same ``u``, the certificate is sound (never certifies a nucleus that is not
  inside the candidates) and flags exactly the expected real rows; the host fallback equals the reference sampler.
* ``test_cpu_adversarial_rows``: the review rows (``_adversarial_rows``: a flat high-entropy row with a near-tie top-2
  whose T = 0.6 nucleus is ~2e5 tokens, the near-tie top-2 on a window-sized nucleus and alone, an exact top-2 tie)
  against vLLM's top-k/top-p, the host twin's certificate and the host fallback.
* ``test_cpu_import_rule``: ``tt/sampling.py`` imports no other ``models/demos`` package, vLLM, transformers or
  safetensors, and opens no device.

Device tests (lock wrapper; ~2 min; run with the trace-allocation tracker on)::

    scripts/devrun.sh -t 2400 -n sampling_tests -- env OMP_WAIT_POLICY=PASSIVE TT_METAL_TRACE_ALLOC_TRACKING=1 \
        python -m pytest models/demos/motif3/tests/unit/test_sampling.py -s -p no:cacheprovider \
        -k "not cpu and not probe"

(a) ``test_sampler_distribution`` -- 50,000 draws for each of 8 lane groups (real rows x parameters: top-p 0.95 with
    kept sets of 477 / 258 / 6 / 3 tokens, T = 1.5, top-k 20 with top-p off, top-k 20 with ties at the k-th value,
    top-k 100 + top-p 0.9 at T = 0.8) in one mixed 32-lane batch: chi-square and TV against the exact fp64
    distribution within multinomial noise, plus every single draw equal to the fp64 inverse CDF fed the same ``u``
    over the device's walk order (tokens with identical logits count as a tie-order difference; flagged draws must
    equal the host fallback).
(b) ``test_sampler_greedy_argmax`` -- greedy lanes bitwise ``torch.argmax`` of the full logits on real rows and crafted
    ties (across chips, inside a chip, more than K tied maxima -> flagged and resolved), mixed with sampled lanes.
(c) ``test_sampler_seeds_and_lanes`` -- determinism per (seed, position), lane relocation, independence from the other
    lanes' parameters, cross-lane independence statistics, unseeded lanes, the host uniform bit-exact.
(d) ``test_sampler_traced_equals_eager`` -- a captured trace replayed with new logits, parameters and positions equals
    eager bitwise; no program compiled after the capture.
(e) ``test_sampler_latency`` -- traced sampler latency at 32 lanes (budget ``MOTIF3_SAMPLER_BUDGET_MS``, 1.0 ms) and the
    per-step host costs (counter write, result read, a fallback step).
(f) ``test_sampler_rows_and_coverage`` -- the 32 real rows with their lane parameters (mixed greedy / T / top-k /
    top-p 1.0 lanes): kept-set sizes = fp64, tokens = fp64 inverse CDF with the same u over many positions, tokens and
    info identical on all 32 chips, flags exactly on the rows whose nucleus leaves the candidate set (and top-p 1.0 /
    top-k > W), the cumulative coverage counter, the host fallback; also on the (8, 4) mesh orientation.
(g) ``test_sampler_adversarial_rows`` -- the review rows at T = 0.6 (3,000 traced steps, ``MOTIF3_SAMPLER_ADV_STEPS``):
    the flat ~2e5-token nucleus flagged every step (fallback == ``exact_sample``), every certified draw of the near-tie
    / exact-tie rows == the fp64 inverse CDF over the lane's device order, chi-square / binomial tests on all draws.

Opt-in ``test_probe_sampler_variants`` (``MOTIF3_SAMPLER_PROBE=1``): per-stage traced latency and every option /
implementation variant (K, W, logprobs, rng, L1 intermediates, fused activations, the cut repair, the cost of the
exactness check).

Real rows: ``docs/sampling/data/sampler_rows.safetensors`` (32 decode logits rows of the model's own traces, bf16,
with the lane parameters of MEASURE_AND_CONTRACT §1.8; regenerate with docs/sampling/scripts/make_proto_rows.py).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROWS_FILE = Path(
    os.environ.get(
        "MOTIF3_SAMPLER_ROWS", "/home/ttuser/hchang/experiments/motif-3/docs/sampling/data/sampler_rows.safetensors"
    )
)
REPORT_DIR = Path("/home/ttuser/hchang/experiments/motif-3/docs/sampling/data")
VOCAB = 220160
NUM_CHIPS = 32
SLICE = VOCAB // NUM_CHIPS
LANES = 32
TT_METAL = Path(__file__).resolve().parents[5]


def _S():
    from models.demos.motif3.tt import sampling as S

    return S


# ============================================================================================================
# shared helpers (CPU)
# ============================================================================================================
def _real_rows():
    """The 32 real rows and their lane parameters (skip when the fixture is missing)."""
    if not ROWS_FILE.is_file():
        pytest.skip(f"real sampler rows missing: {ROWS_FILE} (docs/sampling/scripts/make_proto_rows.py)")
    from safetensors.torch import load_file

    d = load_file(str(ROWS_FILE))
    return {
        "logits": d["logits"].to(torch.bfloat16),
        "temperature": d["temperature"].float(),
        "top_p": d["top_p"].float(),
        "top_k": d["top_k"].long(),
        "seed": d["seed"].long(),
    }


def _lanes_of(rows):
    S = _S()
    return S.normalize_lanes(
        rows["temperature"].tolist(), rows["top_p"].tolist(), rows["top_k"].tolist(), rows["seed"].tolist(),
        vocab_size=VOCAB, num_lanes=int(rows["logits"].shape[0]),
    )  # fmt: skip


def _synthetic_rows(seed: int = 0) -> torch.Tensor:
    """Crafted logits rows ``[12, V]`` bf16: peaked, broad (huge nucleus), a ~200-token nucleus spread over chips, a
    nucleus concentrated in chip 0 (not inside 64 candidates), boundary ties, max ties across / inside chips."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(12, VOCAB, generator=g) * 1.5
    x = base.clone()
    x[[0, 2, 3, 4, 9, 10, 11]] -= 8.0  # a low background, so that the hot tokens below carry the nucleus
    x[0, 12345] = 30.0  # peaked
    x[1] = torch.randn(VOCAB, generator=g) * 0.7  # broad
    hot = torch.randperm(VOCAB, generator=g)[:600]
    x[2, hot[:200]] = 9.0 + torch.rand(200, generator=g)  # ~200-token nucleus over all chips
    x[3, :100] = 9.0 + torch.rand(100, generator=g) * 0.1  # 100 hot tokens all on chip 0
    x[4, hot[:40]] = 8.0  # boundary ties: 40 tokens with the same logit
    x[4, hot[40:45]] = 9.5
    x[5, [7, 6880 * 3 + 2, 6880 * 17 + 5]] = 25.0  # max tied across chips (lowest id 7)
    x[6, [6880 * 5 + 9, 6880 * 5 + 1, 6880 * 5 + 300]] = 25.0  # max tied inside one chip (lowest 6880*5+1)
    x[7, 6880 * 9 : 6880 * 9 + 70] = 25.0  # 70 tied maxima on one chip (> K = 64)
    x[8] = 0.0  # all equal
    x[9, hot[:3]] = torch.tensor([12.0, 11.0, 10.5])  # small nucleus
    x[10, hot[:600]] = 6.0 + torch.rand(600, generator=g)  # ~600 tokens: beyond a 512 window
    x[11, hot[:20]] = 10.0 + torch.rand(20, generator=g)
    return x.to(torch.bfloat16)


def _adversarial_rows():
    """Review rows (bf16 ``[V]`` each; sampled at T = 0.6, top-p 0.95 by the tests):

    * ``flat``: a flat high-entropy background (logits -2 .. -1.75) with a near-tie top-2 (2.0 / 1.9921875, one bf16
      ulp apart, on chips 7 and 21): the T = 0.6 nucleus is huge (~2.07e5 tokens) -> flagged every step, exact host
      fallback (a full sort);
    * ``window``: the same near-tie top-2 on top of 400 hot tokens spread over all chips: nucleus ~374 (certified);
    * ``pure``: a near-tie top-2 alone (4.0 / 3.96875 on one chip): p ~ 0.513 / 0.487;
    * ``tie``: an exact top-2 tie on chips 30 and 3 (ids 206500, 20647) + 60 tokens at ~1.5: nucleus 58, greedy = 20647.
    """
    g = torch.Generator().manual_seed(2026)
    flat = -2.0 + torch.rand(VOCAB, generator=g) * 0.25
    flat[SLICE * 7 + 11], flat[SLICE * 21 + 5] = 2.0, 1.9921875
    win = -8.0 + torch.randn(VOCAB, generator=g) * 0.5
    hot = torch.randperm(VOCAB, generator=g)[:400]
    win[hot] = 1.0 + torch.rand(400, generator=g) * 0.3
    win[int(hot[0])], win[int(hot[1])] = 2.0, 1.9921875
    tie = -8.0 + torch.randn(VOCAB, generator=g) * 0.5
    hot = torch.randperm(VOCAB, generator=g)[:60]
    tie[hot] = 1.4 + torch.rand(60, generator=g) * 0.2
    tie[SLICE * 30 + 100], tie[SLICE * 3 + 7] = 3.0, 3.0
    pure = -10.0 + torch.randn(VOCAB, generator=g) * 0.3
    pure[SLICE * 12 + 3000], pure[SLICE * 12 + 17] = 4.0, 3.96875
    return {k: v.to(torch.bfloat16) for k, v in (("flat", flat), ("window", win), ("pure", pure), ("tie", tie))}


def vllm_kept(x_bf16: torch.Tensor, temps, top_k, top_p) -> torch.Tensor:
    """[B, V] bool: the tokens vLLM 0.26's Sampler can draw (``apply_temperature`` + ``apply_top_k_top_p_pytorch``
    exactly as the plugin's host sampler runs them: fp32 logits / T, ascending unstable sort, top-k mask
    ``logits_sort < kth``, fp32 softmax + cumsum, mask ``cumsum <= 1 - p``, the last kept)."""
    logits = x_bf16.float()
    t = torch.as_tensor(temps, dtype=torch.float32)
    t = torch.where(t < 1e-5, torch.ones_like(t), t)
    logits = logits / t.unsqueeze(1)
    k = torch.as_tensor(top_k, dtype=torch.long)
    p = torch.as_tensor(top_p, dtype=torch.float32)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    top_k_mask = logits_sort.gather(1, (logits_sort.size(1) - k).unsqueeze(1))
    logits_sort.masked_fill_(logits_sort < top_k_mask, -float("inf"))
    probs_sort = logits_sort.softmax(dim=-1, dtype=torch.float32)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))
    out = logits.scatter(-1, logits_idx, logits_sort)
    return out > -float("inf")


def _expected_flag(row: torch.Tensor, lane, K: int, W: int):
    """Independent fp64 re-derivation of the device certificate: ``(flag, borderline)``; ``borderline`` = the window
    mass is within 1e-6 of ``p_eff + margin`` (the fp32 device sum may decide either way)."""
    S = _S()
    x = row.float()
    xc = x.view(NUM_CHIPS, SLICE)
    vals = torch.sort(xc, dim=1, descending=True, stable=True).values[:, :K]
    sv = torch.sort(vals.reshape(-1), descending=True).values[:W]
    lim = max(float(vals[:, K - 1].max()), float(sv[W - 1]))  # every chip's K-th candidate and the window's last
    if lane.greedy:
        return not (lim < float(x.max())), False
    if lane.full_support:
        return True, False
    if lane.top_k_active:
        if lane.top_k > W:
            return True, False
        return not (lim < float(sv[lane.top_k - 1])), False
    ids, _ = S.exact_nucleus(row, lane)
    v_last = float(x[ids[-1]])
    xd = row.double()
    logp = (xd - xd.max()) / lane.temperature
    mass_w = float(torch.exp(sv.double() / lane.temperature - xd.max() / lane.temperature
                             - torch.logsumexp(logp, 0)).sum())
    need = float(np.float32(lane.top_p)) + S.CERT_MARGIN
    return not (lim < v_last and mass_w >= need), abs(mass_w - need) < 1e-6


def _contained(row: torch.Tensor, ids: torch.Tensor, K: int) -> bool:
    """Every id of ``ids`` is among its chip's top-K (reference tie order: logit desc, id asc)."""
    x = row.float().view(NUM_CHIPS, SLICE)
    order = torch.sort(x, dim=1, descending=True, stable=True).indices[:, :K]
    cand = set((order + (torch.arange(NUM_CHIPS) * SLICE).view(-1, 1)).reshape(-1).tolist())
    return all(int(i) in cand for i in ids.tolist())


# ============================================================================================================
# CPU tests
# ============================================================================================================
def _lowbias32_py(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x21F0AAAD) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x735A2D97) & 0xFFFFFFFF
    x ^= x >> 15
    return x


def test_cpu_rng_reference():
    S = _S()
    rng = np.random.default_rng(1)
    xs = np.concatenate([rng.integers(0, 1 << 32, 4096, dtype=np.uint64), np.array([0, 1, 0xFFFFFFFF, 0x80000000])])
    got = S.lowbias32(xs)
    assert got.dtype == np.uint32
    assert [int(v) for v in got] == [_lowbias32_py(int(v)) for v in xs]
    u = S.uniform_from_counter(xs)
    assert u.dtype == np.float32
    assert float(u.min()) >= 0.0 and float(u.max()) < 1.0
    ref = np.array([(_lowbias32_py(int(v)) >> 8) * 2.0**-24 for v in xs], dtype=np.float32)
    assert np.array_equal(u, ref)
    assert np.array_equal(u * 2**24, np.round(u * 2**24))  # exact 24-bit grid
    # counter: key + pos * golden (mod 2^32), wraps
    assert S.lane_counter(0xFFFFFFFF, 1) == (0xFFFFFFFF + 0x9E3779B1) & 0xFFFFFFFF
    assert S.lane_counter(5, 0) == 5


def test_cpu_rng_uniformity():
    """The per-lane uniform stream (consecutive positions of one key) and the cross-lane values (consecutive seeds at
    one position) are uniform and serially uncorrelated (chi-square over 64 bins, lag-1 correlation)."""
    from scipy import stats

    S = _S()
    n = 200_000
    for key in (S.seed_key(0), S.seed_key(1), S.seed_key(12345), 0):
        ctr = (np.uint64(key) + np.arange(n, dtype=np.uint64) * np.uint64(S.GOLDEN32)) & np.uint64(S.MASK32)
        u = S.uniform_from_counter(ctr).astype(np.float64)
        h, _ = np.histogram(u, bins=64, range=(0, 1))
        p = stats.chisquare(h).pvalue
        r = np.corrcoef(u[:-1], u[1:])[0, 1]
        assert p > 1e-4, (key, p)
        assert abs(r) < 5 / math.sqrt(n), (key, r)
        # pairs (u_t, u_{t+1}) on an 8x8 grid: serial independence
        h2, _, _ = np.histogram2d(u[:-1:2], u[1::2], bins=8, range=[[0, 1], [0, 1]])
        assert stats.chisquare(h2.reshape(-1)).pvalue > 1e-4
    # across lanes: many seeds at the same position
    keys = np.array([S.seed_key(s) for s in range(20_000)], dtype=np.uint64)
    for pos in (0, 1, 1000):
        u = S.uniform_from_counter((keys + np.uint64(pos * S.GOLDEN32)) & np.uint64(S.MASK32)).astype(np.float64)
        h, _ = np.histogram(u, bins=32, range=(0, 1))
        assert stats.chisquare(h).pvalue > 1e-4
    # two lanes with neighbouring seeds over positions: uncorrelated
    a = S.uniform_from_counter((np.uint64(S.seed_key(7)) + np.arange(n, dtype=np.uint64) * np.uint64(S.GOLDEN32))
                               & np.uint64(S.MASK32)).astype(np.float64)
    b = S.uniform_from_counter((np.uint64(S.seed_key(8)) + np.arange(n, dtype=np.uint64) * np.uint64(S.GOLDEN32))
                               & np.uint64(S.MASK32)).astype(np.float64)
    assert abs(np.corrcoef(a, b)[0, 1]) < 5 / math.sqrt(n)


def test_cpu_seed_key():
    S = _S()
    ks = [S.seed_key(s) for s in range(1000)]
    assert len(set(ks)) == 1000 and all(0 <= k < 2**32 for k in ks)
    assert S.seed_key(42) == S.seed_key(42) and S.seed_key(-1) == S.seed_key(2**64 - 1)
    assert S.seed_key(2**40 + 3) != S.seed_key(3)  # high bits matter
    # bit balance of the keys of consecutive seeds
    bits = np.array([[(k >> b) & 1 for b in range(32)] for k in ks], dtype=np.float64)
    assert np.all(np.abs(bits.mean(0) - 0.5) < 0.07)


def test_cpu_lane_params():
    S = _S()
    n = S.normalize_lane
    assert n(0.0, 1.0, 1, None, vocab_size=VOCAB).greedy  # padding slot
    assert n(1e-6, 0.9, VOCAB, None, vocab_size=VOCAB).greedy
    assert not n(0.01, 0.9, VOCAB, None, vocab_size=VOCAB).greedy
    for off in (VOCAB, VOCAB + 5, 0, -1, None):
        assert n(1.0, 0.95, off, None, vocab_size=VOCAB).top_k == 0
    assert n(1.0, 0.95, 50, 3, vocab_size=VOCAB).top_k == 50
    assert n(None, None, None, None, vocab_size=VOCAB) == S.LaneParams(1.0, 1.0, 0, None)
    for bad in ((-0.1, 0.9), (float("nan"), 0.9), (1.0, 1.5), (1.0, -0.1)):
        with pytest.raises(ValueError):
            n(bad[0], bad[1], 0, None, vocab_size=VOCAB)
    assert n(1.0, 1.0, 0, None, vocab_size=VOCAB).full_support
    assert not n(1.0, 1.0, 40, None, vocab_size=VOCAB).full_support
    lanes = [S.LaneParams(0.0, 1.0, 1), S.LaneParams(0.5, 0.95, 0), S.LaneParams(1.0, 0.0, 1),
             S.LaneParams(1.0, 0.9, 600), S.LaneParams(2.0, 1.0, 512)]
    c = S.device_param_columns(lanes, 512)
    assert c["greedy"].tolist() == [1, 0, 0, 0, 0]
    assert c["inv_t"].tolist() == [1.0, 2.0, 1.0, 1.0, 0.5]
    assert c["k_active"].tolist() == [0, 0, 1, 1, 1]
    assert c["k_idx"].tolist() == [0, 0, 0, 511, 511]
    assert c["k_in_win"].tolist() == [0, 0, 1, 0, 1]
    assert float(c["top_p"][2]) == pytest.approx(S.MIN_TOP_P) and float(c["top_p"][1]) == pytest.approx(0.95)
    with pytest.raises(ValueError):
        S.normalize_lanes([1.0] * 3, None, None, None, vocab_size=VOCAB, num_lanes=4)
    # the top-k threshold mask: max(SV + k_ofs) = SV[k - 1] (top-k lanes) or ~-BIG (off)
    ofs = S.k_offset_rows(c, 512)
    sv = torch.linspace(10, -10, 512).expand(5, 512)
    kth = (sv + ofs).max(1).values
    assert float(kth[2]) == float(sv[2, 0])
    assert float(kth[3]) == float(sv[3, 511]) and float(kth[4]) == float(sv[4, 511])
    assert float(kth[0]) < -1e38 and float(kth[1]) < -1e38
    # the info tensor -> SampleResult
    info = torch.arange(8 * 32, dtype=torch.float32).reshape(1, 1, 8, 32)
    r = S.SampleResult.from_info(info)
    assert r.tokens.tolist() == list(range(32)) and r.flags.tolist() == [True] * 32 and float(r.kth[0]) == 7 * 32
    # the bridge helper: per-row plugin params -> lane-ordered lists (unused lanes = greedy padding)
    from types import SimpleNamespace

    sp = SimpleNamespace(temperature=[1.0, 0.6], top_p=[0.95, 0.9], top_k=[VOCAB, 20], seed=[None, 7])
    t, p, k, sd = S.lane_lists_from_rows(sp, [5, 17], num_lanes=32)
    assert (t[5], p[5], k[5], sd[5]) == (1.0, 0.95, VOCAB, None) and (t[17], p[17], k[17], sd[17]) == (0.6, 0.9, 20, 7)
    assert t[0] == 0.0 and k[0] == 1 and S.normalize_lane(t[0], p[0], k[0], sd[0], vocab_size=VOCAB).greedy


def _boundary_margin(row, lane) -> float:
    """|exclusive prefix mass - top_p * mass_k| at the first rank the fp64 reference excludes (or keeps last): how far
    the top-p cut is from a rounding tie (vLLM's fp32 cumsum resolves exact ties either way)."""
    x = row.double()
    o = torch.sort(row.float(), descending=True, stable=True).indices
    logp = (x - x.max()) / lane.temperature
    ps = torch.exp(logp[o] - torch.logsumexp(logp, 0))
    if lane.top_k_active:
        keep_k = x[o] >= x[o][lane.top_k - 1]
        ps = ps[keep_k]
        thr = lane.top_p * float(ps.sum())
    else:
        thr = lane.top_p
    excl = torch.cumsum(ps, 0) - ps
    return float((excl - thr).abs().min())


def _check_vs_vllm(x, lanes, kept_v, strict: bool = False):
    """Sizes equal (``strict``: always; else except a cut that sits on a rounding tie, e.g. an all-equal row);
    members equal except tokens tied with the boundary value."""
    S = _S()
    diffs = 0
    for i, lane in enumerate(lanes):
        if lane.greedy:
            continue
        ids, q = S.exact_nucleus(x[i], lane)
        mine = torch.zeros(VOCAB, dtype=torch.bool)
        mine[ids] = True
        n_mine, n_v = int(mine.sum()), int(kept_v[i].sum())
        v_last = x[i].float()[ids[-1]]
        if n_mine != n_v:
            # vLLM's ascending fp32 cumsum over ~2e5 terms errs by ~1e-6 .. 1e-5 on broad crafted rows
            assert not strict and _boundary_margin(x[i], lane) < 1e-4, (i, lane, n_mine, n_v)
            continue
        sym = torch.nonzero(mine ^ kept_v[i]).reshape(-1)
        if sym.numel():
            assert bool((x[i].float()[sym] == v_last).all()), (i, "non-tie member difference")
            diffs += 1
        assert abs(float(q.sum()) - 1.0) < 1e-12
    return diffs


def test_cpu_reference_vs_vllm():
    S = _S()
    # crafted rows at several parameter sets
    x = _synthetic_rows()
    for T, p, k in ((1.0, 0.95, 0), (0.6, 0.95, 0), (1.0, 0.9, 20), (0.7, 1.0, 5), (1.3, 0.5, 0), (1.0, 0.0, 1)):
        lanes = [S.LaneParams(T, p, k, None)] * x.shape[0]
        kept_v = vllm_kept(x, [T] * x.shape[0], [k if k else VOCAB] * x.shape[0], [p] * x.shape[0])
        _check_vs_vllm(x, lanes, kept_v)
    # real rows with their own lane parameters, and all rows at Motif's defaults / T = 0.6
    try:
        rows = _real_rows()
    except pytest.skip.Exception:
        return
    xr = rows["logits"]
    lanes = _lanes_of(rows)
    kept_v = vllm_kept(xr, rows["temperature"], rows["top_k"], rows["top_p"])
    diffs = _check_vs_vllm(xr, lanes, kept_v, strict=True)
    for T in (1.0, 0.6):
        kept_v = vllm_kept(xr, [T] * LANES, [VOCAB] * LANES, [0.95] * LANES)
        diffs += _check_vs_vllm(xr, [S.LaneParams(T, 0.95, 0)] * LANES, kept_v, strict=True)
    print(f"[sampling] reference vs vLLM replica: sizes equal, {diffs} boundary-tie member differences")
    # the real vLLM function, when importable without side effects
    try:
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
    except Exception as e:  # pragma: no cover - vLLM missing or its import refused here
        print(f"[sampling] real vLLM apply_top_k_top_p_pytorch not importable ({type(e).__name__}); replica only")
        return
    lg = xr.float() / torch.where(rows["temperature"] < 1e-5, torch.ones(LANES), rows["temperature"]).unsqueeze(1)
    out = apply_top_k_top_p_pytorch(lg.clone(), rows["top_k"], rows["top_p"])
    kept_real = out > -float("inf")
    rep = vllm_kept(xr, rows["temperature"], rows["top_k"], rows["top_p"])
    assert torch.equal(kept_real.sum(1), rep.sum(1))
    print("[sampling] the replica's kept-set sizes equal vLLM's own apply_top_k_top_p_pytorch on the real rows")


def test_cpu_emulation_real_rows():
    """The fp32 emulation of the device pipeline on the 32 real rows (their lane parameters) and on every row at
    T = 1.0 / 0.6, top-p 0.95: certified lanes keep exactly the fp64 nucleus and draw the fp64 inverse-CDF token for
    the same u; flags exactly on the rows whose nucleus leaves the candidates (+ top-p 1.0)."""
    S = _S()
    rows = _real_rows()
    x = rows["logits"]
    lanes = _lanes_of(rows)
    rng = np.random.default_rng(3)
    for K in (64, 32):
        for set_name, ls in (("own", lanes), ("T1", [S.LaneParams(1.0, 0.95, 0, 1)] * LANES),
                             ("T0.6", [S.LaneParams(0.6, 0.95, 0, 1)] * LANES)):
            for rep in range(3):
                ctr = rng.integers(0, 1 << 32, LANES, dtype=np.uint64)
                em = S.emulate_device(x, ls, ctr, local_k=K, window=512)
                u = S.uniform_from_counter(ctr)
                assert np.array_equal(em["u"].numpy(), u)
                for l, lane in enumerate(ls):
                    ids, q = S.exact_nucleus(x[l], lane)
                    contained = _contained(x[l], ids, K) and (lane.greedy or ids.numel() <= 512)
                    flag = bool(em["flag"][l] > 0.5)
                    want, borderline = _expected_flag(x[l], lane, K, 512)
                    if not borderline:
                        assert flag == want, (K, set_name, l, flag, want)
                    if not contained or lane.full_support:
                        assert flag, (K, set_name, l, "not contained but certified")
                    if flag:
                        continue
                    if lane.greedy:
                        assert int(em["token"][l]) == S.greedy_token(x[l])
                        continue
                    assert int(em["n_kept"][l]) == ids.numel(), (K, set_name, l)
                    # the token for the emulated walk order (tie order = the reference here)
                    ids_o, q_o = S.exact_nucleus(x[l], lane, order=em["order"][l])
                    want = int(ids_o[S.inverse_cdf(q_o, float(u[l]))])
                    assert int(em["token"][l]) == want, (K, set_name, l)
                    lp = S.raw_logprob(x[l], int(em["token"][l]))
                    assert abs(float(em["logprob"][l]) - lp) < 2e-5 * max(1.0, abs(lp))
    # the documented flags of the own-parameter batch at K = 64 (MEASURE_AND_CONTRACT §1.8: lanes 0, 1, 30)
    em = S.emulate_device(x, lanes, np.zeros(LANES, dtype=np.uint64), local_k=64, window=512)
    assert torch.nonzero(em["flag"] > 0.5).reshape(-1).tolist() == [0, 1, 30]


def test_cpu_emulation_synthetic():
    """Crafted rows: soundness of the certificate (never certifies a non-contained nucleus), greedy tie rules
    (lowest id among maxima; > K tied maxima flagged), boundary ties, nuclei beyond the window."""
    S = _S()
    x = _synthetic_rows(1)
    B = x.shape[0]
    rng = np.random.default_rng(4)
    params = [(1.0, 0.95, 0), (0.6, 0.95, 0), (1.0, 0.9, 20), (0.7, 1.0, 5), (1.0, 1.0, 0), (1.0, 0.95, 600),
              (0.0, 1.0, 1), (1.0, 0.0, 1), (2.0, 0.5, 0)]
    for T, p, k in params:
        lanes = [S.LaneParams(T, p, k, 9)] * B
        ctr = rng.integers(0, 1 << 32, B, dtype=np.uint64)
        em = S.emulate_device(x, lanes, ctr, local_k=64, window=512)
        u = S.uniform_from_counter(ctr)
        for l in range(B):
            lane = lanes[l]
            flag = bool(em["flag"][l] > 0.5)
            if lane.greedy:
                ties = int((x[l].float() == x[l].float().max()).view(NUM_CHIPS, SLICE).sum(1).max())
                assert flag == (ties >= 64), (l, ties)
                if not flag:
                    assert int(em["token"][l]) == S.greedy_token(x[l])
                continue
            ids, q = S.exact_nucleus(x[l], lane)
            want, borderline = _expected_flag(x[l], lane, 64, 512)
            if not borderline:
                assert flag == want, (T, p, k, l, flag, want)
            if lane.full_support or (lane.top_k_active and lane.top_k > 512):
                assert flag
                continue
            if not _contained(x[l], ids, 64) or ids.numel() > 512:
                assert flag, (T, p, k, l)
            if not flag:
                assert int(em["n_kept"][l]) == ids.numel()
                ids_o, q_o = S.exact_nucleus(x[l], lane, order=em["order"][l])
                assert int(em["token"][l]) == int(ids_o[S.inverse_cdf(q_o, float(u[l]))])
    # row 3 (100 hot tokens on chip 0) and row 10 (600-token nucleus) are flagged at T = 1, p = 0.95
    em = S.emulate_device(x, [S.LaneParams(1.0, 0.95, 0)] * B, np.zeros(B, dtype=np.uint64))
    assert bool(em["flag"][3] > 0.5) and bool(em["flag"][10] > 0.5) and not bool(em["flag"][9] > 0.5)


def test_cpu_fallback_sampler():
    """``fallback_sample`` (the host path of flagged lanes) == ``exact_sample`` (reference order) for every lane type,
    and its full-support path draws the full-vocab distribution."""
    S = _S()
    x = _synthetic_rows(2)
    rng = np.random.default_rng(5)
    for T, p, k in ((1.0, 0.95, 0), (0.6, 0.9, 0), (1.0, 0.95, 30), (1.0, 0.95, 3000), (1.5, 0.99, 0), (0.0, 1, 1),
                    (0.7, 1.0, 5)):
        lane = S.LaneParams(T, p, k, None)
        for l in (0, 2, 3, 4, 5, 9, 10, 11):
            ids, q = S.exact_nucleus(x[l], lane)
            for u in rng.random(4).astype(np.float32):
                want = int(ids[S.inverse_cdf(q, float(u))]) if not lane.greedy else int(ids[0])
                assert S.fallback_sample(x[l], lane, float(u)) == want, (T, p, k, l)
    # full support (top_p = 1, no top-k): token-id-order inverse CDF -> the full softmax distribution
    v = 50
    row = torch.zeros(VOCAB, dtype=torch.bfloat16) - 30.0
    row[:v] = torch.linspace(0, 3, v).to(torch.bfloat16)
    lane = S.LaneParams(1.0, 1.0, 0, None)
    us = rng.random(40_000)
    toks = np.array([S.fallback_sample(row, lane, float(u)) for u in us])
    p = torch.softmax(row.double(), 0).numpy()
    emp = np.bincount(toks, minlength=VOCAB) / us.size
    assert 0.5 * np.abs(emp - p).sum() < 0.02
    assert toks.max() < VOCAB


def test_cpu_adversarial_rows():
    """Review rows (``_adversarial_rows``) at T = 0.6, top-p 0.95: the fp64 reference against vLLM's top-k/top-p
    (replica, and the real function when importable: kept-set sizes equal up to fp32 rounding at the flat row's dense
    cut), the exact near-tie top-2 ratio, greedy on the exact tie, the host twin's certificate (the flat row's
    ~2e5-token nucleus is flagged; the other rows are certified with n_kept = fp64 and token = fp64 inverse CDF), and
    the flat row's host fallback == the reference sampler."""
    S = _S()
    rows = _adversarial_rows()
    names = ("flat", "window", "pure", "tie")
    x = torch.stack([rows[n] for n in names])
    T, P = float(np.float32(0.6)), float(np.float32(0.95))
    lane = S.LaneParams(T, P, 0, 7)
    lanes = [lane] * len(names)
    kept_v = vllm_kept(x, [T] * len(names), [VOCAB] * len(names), [P] * len(names))
    _check_vs_vllm(x, lanes, kept_v)
    try:
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
    except Exception:  # pragma: no cover - vLLM missing or its import refused here
        apply_top_k_top_p_pytorch = None
    if apply_top_k_top_p_pytorch is not None:  # the replica == vLLM's own function on these rows too
        out = apply_top_k_top_p_pytorch(x.float() / T, torch.full((len(names),), VOCAB), torch.full((len(names),), P))
        assert torch.equal((out > -float("inf")).sum(1), kept_v.sum(1))
    sizes ={n: int(S.exact_nucleus(x[i], lane)[0].numel()) for i, n in enumerate(names)}
    assert sizes["flat"] > 200_000 and 300 < sizes["window"] < 512 and sizes["pure"] == 2 and 50 < sizes["tie"] < 64
    ids, q = S.exact_nucleus(rows["pure"], lane)
    d = float(rows["pure"].float()[ids[0]] - rows["pure"].float()[ids[1]])
    assert abs(float(q[0] / (q[0] + q[1])) - 1.0 / (1.0 + math.exp(-d / T))) < 1e-9
    assert S.greedy_token(rows["tie"]) == SLICE * 3 + 7
    rng = np.random.default_rng(8)
    for _ in range(4):
        ctr = rng.integers(0, 1 << 32, len(names), dtype=np.uint64)
        em = S.emulate_device(x, lanes, ctr)
        u = S.uniform_from_counter(ctr)
        assert bool(em["flag"][0] > 0.5) and not bool((em["flag"][1:] > 0.5).any()), em["flag"]
        for i in range(1, len(names)):
            ids, q = S.exact_nucleus(x[i], lane)
            assert int(em["n_kept"][i]) == ids.numel(), (names[i], int(em["n_kept"][i]), ids.numel())
            ids_o, q_o = S.exact_nucleus(x[i], lane, order=em["order"][i])
            assert int(em["token"][i]) == int(ids_o[S.inverse_cdf(q_o, float(u[i]))]), names[i]
    for u in (0.0, 0.3, 0.5, 0.999):  # the 4096-token prefix cannot decide a 2e5-token nucleus: a full sort
        assert S.fallback_sample(rows["flat"], lane, u) == S.exact_sample(rows["flat"], lane, u)


def test_cpu_import_rule():
    probe = (
        "import sys, json; import models.demos.motif3.tt.sampling as m; "
        "bad=[k for k in sys.modules if k.startswith('models.demos.') and not k.startswith('models.demos.motif3')]; "
        "bad+=[k for k in ('vllm','transformers','safetensors','huggingface_hub') if k in sys.modules]; "
        "print(json.dumps(bad))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TT_METAL) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(TT_METAL), env=env,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) == []


# ============================================================================================================
# device helpers
# ============================================================================================================
def _mesh_params(shape=(4, 8)):
    from models.demos.motif3.tt.model_config import device_params

    return [pytest.param(tuple(shape), device_params(), id=f"{shape[0]}x{shape[1]}")]


class _Capture:
    """Exception-safe trace capture (a raise inside still ends and releases the capture; GATES_RESULTS §11.6)."""

    def __init__(self, mesh_device):
        self.mesh, self.tid = mesh_device, None

    def __enter__(self):
        import ttnn

        self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        return self

    def __exit__(self, exc_type, exc, tb):
        import ttnn

        try:
            ttnn.end_trace_capture(self.mesh, self.tid, cq_id=0)
        except Exception:
            if exc_type is None:
                raise
        if exc_type is not None:
            try:
                ttnn.release_trace(self.mesh, self.tid)
            except Exception:
                pass
        return False


def _setup(mesh_device, tag, **kw):
    from models.demos.motif3.tt.ccl import MotifCCL, log_fabric
    from models.demos.motif3.tt.model_config import MotifTTConfig
    from models.demos.motif3.tt.sampling import MotifDeviceSampler

    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)
    log_fabric(mesh_device, tag)
    ccl = MotifCCL(mesh_device, cfg)
    smp = MotifDeviceSampler(mesh_device, cfg, ccl=ccl, **kw)
    return cfg, ccl, smp


def _host_blocks(cfg, rows: torch.Tensor) -> torch.Tensor:
    """Lane-ordered ``[32, V]`` -> the LM head's per-chip "mesh" layout as one host tensor ``[R, 1, 32, C * 6880]``
    (chip (r, c) = vocab block r * C + c), for ``ShardTensor2dMesh(dims=(0, 3))``."""
    R, C = cfg.axes.mesh_shape
    return rows.reshape(LANES, R, C * SLICE).permute(1, 0, 2).reshape(R, 1, LANES, C * SLICE).contiguous()


def _logits_mapper(mesh_device, cfg):
    import ttnn

    return ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 3), mesh_shape=tuple(cfg.axes.mesh_shape))


def _upload_logits(mesh_device, cfg, rows: torch.Tensor):
    import ttnn

    return ttnn.from_torch(_host_blocks(cfg, rows.to(torch.bfloat16)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=_logits_mapper(mesh_device, cfg))


def _write_logits(mesh_device, cfg, dev, rows: torch.Tensor):
    import ttnn

    host = ttnn.from_torch(_host_blocks(cfg, rows.to(torch.bfloat16)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           mesh_mapper=_logits_mapper(mesh_device, cfg))
    ttnn.copy_host_to_device_tensor(host, dev)


def _free_out(out):
    import ttnn

    for t in out.tensors():
        ttnn.deallocate(t)


def _device_order(smp, out) -> torch.Tensor:
    """The device's window walk order ``[32, W]`` (global token ids) from a debug output."""
    import ttnn

    I = ttnn.to_torch(ttnn.get_device_tensors(out.debug["I"])[0]).float().reshape(LANES, -1)
    sidx = ttnn.to_torch(ttnn.get_device_tensors(out.debug["sidx"])[0]).float().reshape(LANES, -1)
    return I.gather(1, sidx.long()).long()


def _tokens_u32(out) -> torch.Tensor:
    import ttnn

    return ttnn.to_torch(ttnn.get_device_tensors(out.tokens)[0]).reshape(-1)[:LANES].to(torch.int64)


def _expect_tokens(S, rows, lanes, orders, u, res, *, tol=2e-6):
    """Compare every certified lane's device token with the fp64 inverse CDF over the device walk order fed the same
    ``u``. Returns (compared, mismatches, boundary) -- a mismatch within ``tol`` of a CDF step is a rounding
    boundary."""
    compared = mism = boundary = 0
    for l, lane in enumerate(lanes):
        if bool(res.flags[l]):
            continue
        compared += 1
        if lane.greedy:
            mism += int(int(res.tokens[l]) != S.greedy_token(rows[l]))
            continue
        ids, q = S.exact_nucleus(rows[l], lane, order=orders[l])
        j = S.inverse_cdf(q, float(u[l]))
        if int(res.tokens[l]) != int(ids[j]):
            C = torch.cumsum(q, 0)
            near = min(abs(float(C[j]) - float(u[l])), abs(float(C[j - 1]) - float(u[l])) if j > 0 else 1.0)
            if near < tol:
                boundary += 1
            else:
                mism += 1
    return compared, mism, boundary


# ============================================================================================================
# device tests
# ============================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params((4, 8)) + _mesh_params((8, 4)), indirect=True)
def test_sampler_rows_and_coverage(mesh_device):
    """(f) + exactness on the 32 real rows with their lane parameters (module docstring)."""
    import ttnn

    S = _S()
    rows = _real_rows()
    x = rows["logits"]
    lanes = _lanes_of(rows)
    cfg, ccl, smp = _setup(mesh_device, f"sampling_rows_{tuple(mesh_device.shape)}")
    dev = _upload_logits(mesh_device, cfg, x)
    smp.set_params(rows["temperature"], rows["top_p"], rows["top_k"], rows["seed"])
    # eager with the debug intermediates: the device walk order, n_kept, flags, logZ
    smp.set_positions(torch.full((LANES,), 100))
    out = smp.sample(dev, debug=True)
    res = smp.read(out, all_chips=True)
    orders = _device_order(smp, out)
    em = S.emulate_device(x, lanes, smp._last_ctr, local_k=smp.K, window=smp.W)
    assert torch.equal(res.flags, em["flag"] > 0.5), (res.flags, em["flag"])
    flagged = torch.nonzero(res.flags).reshape(-1).tolist()
    assert flagged == [0, 1, 30], flagged  # not contained at K = 64 (0, 1), top-p 1.0 (30)
    assert bool((res.per_chip == res.per_chip[0]).all()), "outputs differ between chips"
    assert torch.equal(_tokens_u32(out), res.tokens)
    M = ttnn.to_torch(ttnn.get_device_tensors(out.debug["M"])[0]).float().reshape(-1)[:LANES]
    Z = ttnn.to_torch(ttnn.get_device_tensors(out.debug["Z"])[0]).float().reshape(-1)[:LANES]
    worst_lz = 0.0
    for l, lane in enumerate(lanes):
        assert float(M[l]) == float(x[l].float().max())
        Tl = 1.0 if lane.greedy else lane.temperature
        lz_ref = float(torch.logsumexp(x[l].double() / Tl, 0))
        worst_lz = max(worst_lz, abs(math.log(float(Z[l])) + float(M[l]) / Tl - lz_ref))
        o = orders[l]
        assert bool((x[l].float()[o][1:] <= x[l].float()[o][:-1]).all()), "device window is not sorted"
        if l in flagged or lane.greedy:
            continue
        ids, q = S.exact_nucleus(x[l], lane)
        assert int(res.n_kept[l]) == ids.numel(), (l, int(res.n_kept[l]), ids.numel())
        ids_d, _ = S.exact_nucleus(x[l], lane, order=o)
        assert set(ids_d.tolist()) ^ set(ids.tolist()) <= {
            int(i) for i in torch.nonzero(x[l].float() == x[l].float()[ids[-1]]).reshape(-1)
        }  # the device kept set differs from the reference order's only inside the boundary tie group
    assert worst_lz < 2e-6, worst_lz
    _free_out(out)
    # a ROW_MAJOR input (the plain path's untilized logits) gives the same result as the TILE one
    rm = ttnn.to_layout(dev, ttnn.ROW_MAJOR_LAYOUT)
    o_t, o_r = smp.sample(dev), smp.sample(rm)
    r_t, r_r = smp.read(o_t), smp.read(o_r)
    for f in ("tokens", "flags", "logprobs", "n_kept", "u", "mass_kept", "window_mass", "kth"):
        assert torch.equal(getattr(r_t, f), getattr(r_r, f)), f
    _free_out(o_t), _free_out(o_r)
    ttnn.deallocate(rm)
    # fp32 logits are refused (a silent bf16 rounding would break greedy == torch.argmax of the host logits)
    f32 = ttnn.typecast(dev, ttnn.float32)
    with pytest.raises(ValueError):
        smp.sample(f32)
    ttnn.deallocate(f32)
    # many positions: tokens = fp64 inverse CDF with the host uniform over the device order; u bit-exact; logprobs
    smp.reset_coverage()
    out = smp.sample(dev)
    _free_out(out)
    smp.reset_coverage()
    steps, tot = 150, [0, 0, 0]
    counts0 = smp.coverage_counts()
    assert float(counts0.sum()) == 0
    worst_lp = 0.0
    seen_flags = torch.zeros(LANES)
    for step in range(steps):
        smp.set_positions(torch.arange(LANES) * 0 + 1000 + step)
        out = smp.sample(dev)
        res = smp.read(out)
        seen_flags += res.flags.float()
        assert all(bool(res.flags[l]) for l in flagged), "a not-certifiable lane was not flagged"
        u = smp.last_uniforms()
        assert np.array_equal(res.u.numpy().astype(np.float32), u), "device uniform != host twin"
        c, m, b = _expect_tokens(S, x, lanes, orders, u, res)
        tot = [tot[0] + c, tot[1] + m, tot[2] + b]
        for l in range(LANES):
            if not bool(res.flags[l]):
                worst_lp = max(worst_lp, abs(float(res.logprobs[l]) - S.raw_logprob(x[l], int(res.tokens[l]))))
        # the host fallback of the flagged lanes: exact (== the reference sampler with the same u)
        if step % 25 == 0:
            res = smp.resolve(res, x)
            for l in flagged:
                assert int(res.tokens[l]) == S.fallback_sample(x[l], lanes[l], float(u[l]))
                ids, _ = S.exact_nucleus(x[l], lanes[l])
                if not lanes[l].full_support:
                    assert int(res.tokens[l]) in set(ids.tolist())
        _free_out(out)
    assert tot[1] == 0, f"{tot[1]} non-boundary token mismatches of {tot[0]}"
    assert worst_lp < 1e-4, worst_lp
    counts = smp.coverage_counts()
    assert torch.equal(counts, seen_flags), (counts, seen_flags)
    extra = int(seen_flags.sum()) - steps * len(flagged)  # exactness-verification flags of certifiable lanes (rare)
    assert extra <= 3, f"{extra} verification flags in {steps} steps"
    print(f"[sampling] rows {tuple(mesh_device.shape)}: {tot[0]} draws compared, {tot[2]} rounding-boundary, "
          f"log Z err {worst_lz:.2e}, logprob err {worst_lp:.2e}, flagged {flagged} (+{extra} verification flags), "
          f"stats {smp.stats}")
    ttnn.deallocate(dev)
    smp.deallocate()


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_sampler_greedy_argmax(mesh_device):
    """(b) Greedy lanes = ``torch.argmax`` bitwise (real rows, crafted ties; mixed with sampled lanes)."""
    import ttnn

    S = _S()
    rows = _real_rows()
    syn = _synthetic_rows(7)
    x = torch.cat([rows["logits"][:20], syn], 0)[:LANES]
    cfg, ccl, smp = _setup(mesh_device, "sampling_greedy")
    dev = _upload_logits(mesh_device, cfg, x)
    for mode in ("all_greedy", "mixed"):
        T = [0.0] * LANES if mode == "all_greedy" else [0.0 if l % 2 == 0 else 1.0 for l in range(LANES)]
        smp.set_params(T, [0.95] * LANES, [VOCAB] * LANES, list(range(LANES)))
        smp.set_positions(torch.arange(LANES))
        out = smp.sample(dev)
        res = smp.read(out)
        lanes = smp.lane_params
        res = smp.resolve(res, x)
        for l in range(LANES):
            if lanes[l].greedy:
                assert int(res.tokens[l]) == int(torch.argmax(x[l].float())), (mode, l)
        # the crafted rows: 70 tied maxima on one chip (lane 27) and the all-equal row (lane 28) are flagged
        if mode == "all_greedy":
            assert torch.nonzero(res.flags).reshape(-1).tolist() == [27, 28], res.flags
            assert set(res.resolved) == {27, 28}
            assert int(res.tokens[25]) == 7 and int(res.tokens[26]) == 6880 * 5 + 1
        _free_out(out)
    ttnn.deallocate(dev)
    smp.deallocate()


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_sampler_seeds_and_lanes(mesh_device):
    """(c) Determinism per (seed, position), lane relocation, independence of the other lanes, cross-lane
    independence, unseeded lanes."""
    import ttnn
    from scipy import stats

    S = _S()
    rows = _real_rows()
    big, small = rows["logits"][2], rows["logits"][8]  # nucleus 477 / 3 at T = 1, p = 0.95
    cfg, ccl, smp = _setup(mesh_device, "sampling_seeds")
    x = torch.stack([big] * 16 + [small] * 16)
    dev = _upload_logits(mesh_device, cfg, x)
    seeds = [1000 + 7 * l for l in range(16)] * 2
    T, P, K = [1.0] * LANES, [0.95] * LANES, [VOCAB] * LANES

    def run(seed_list, positions, temps=T, dev_t=dev):
        smp.set_params(temps, P, K, seed_list)
        smp.set_positions(positions)
        out = smp.sample(dev_t)
        r = smp.read(out)
        _free_out(out)
        return r

    n_pos = 400
    A = [run(seeds, torch.full((LANES,), p)) for p in range(n_pos)]
    tok = torch.stack([r.tokens for r in A])  # [n_pos, 32]
    us = torch.stack([r.u for r in A]).double()
    # lanes l and l + 16 share seed and position but not the row: equal u; same token where the row is the same
    assert torch.equal(us[:, :16], us[:, 16:])
    # rerun: identical
    B = [run(seeds, torch.full((LANES,), p)) for p in range(0, n_pos, 37)]
    for i, p in enumerate(range(0, n_pos, 37)):
        assert torch.equal(B[i].tokens, tok[p])
    # lane relocation: permute lanes (rows and seeds together) -> tokens permute
    perm = torch.randperm(LANES, generator=torch.Generator().manual_seed(0))
    devp = _upload_logits(mesh_device, cfg, x[perm])
    for p in (3, 77, 250):
        r = run([seeds[i] for i in perm.tolist()], torch.full((LANES,), p), dev_t=devp)
        assert torch.equal(r.tokens, tok[p][perm])
    ttnn.deallocate(devp)
    # other lanes' parameters do not matter: lanes 16..31 greedy / T = 0.6 -> lanes 0..15 unchanged
    for p in (5, 123):
        r = run(seeds, torch.full((LANES,), p), temps=[1.0] * 16 + [0.0] * 8 + [0.6] * 8)
        assert torch.equal(r.tokens[:16], tok[p][:16])
    # the big row varies over positions; cross-lane independence of u and of the small row's tokens
    assert len(set(tok[:, 0].tolist())) > 50
    c = np.corrcoef(us[:, :16].T.numpy())
    off = c[~np.eye(16, dtype=bool)]
    assert np.abs(off).max() < 6 / math.sqrt(n_pos), np.abs(off).max()
    ids, q = S.exact_nucleus(small, S.LaneParams(1.0, 0.95, 0))
    cats = {int(t): i for i, t in enumerate(ids.tolist())}
    a = np.array([cats[int(t)] for t in tok[:, 16]])
    b = np.array([cats[int(t)] for t in tok[:, 17]])
    table = np.zeros((len(cats), len(cats)))
    np.add.at(table, (a, b), 1)
    table = table[table.sum(1) > 0][:, table.sum(0) > 0]
    if min(table.shape) > 1:
        assert stats.chi2_contingency(table)[1] > 1e-4
    # unseeded lanes: fresh randomness every step, uniform over positions
    r1 = run([None] * LANES, torch.full((LANES,), 9))
    r2 = run([None] * LANES, torch.full((LANES,), 9))
    assert not torch.equal(r1.u, r2.u)
    print(f"[sampling] seeds: {n_pos} positions x 32 lanes deterministic, relocation / independence OK, "
          f"max |corr(u)| {np.abs(off).max():.3f}")
    ttnn.deallocate(dev)
    smp.deallocate()


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_sampler_traced_equals_eager(mesh_device):
    """(d) Trace replay with new logits / parameters / positions == eager, bitwise; no program after capture."""
    import ttnn

    rows = _real_rows()
    x = rows["logits"]
    cfg, ccl, smp = _setup(mesh_device, "sampling_trace")
    dev = _upload_logits(mesh_device, cfg, x)
    g = torch.Generator().manual_seed(11)

    def params(i):
        T = [float(v) for v in (torch.rand(LANES, generator=g) * 1.4).tolist()]
        T = [0.0 if (i + l) % 5 == 0 else t for l, t in enumerate(T)]
        P = [0.95 if l % 3 else 0.8 for l in range(LANES)]
        K = [VOCAB if l % 4 else 40 for l in range(LANES)]
        return T, P, K, [i * 100 + l for l in range(LANES)]

    out = smp.sample(dev)  # warm every program
    _free_out(out)
    ttnn.synchronize_device(mesh_device)
    with _Capture(mesh_device) as cap:
        tout = smp.sample(dev)
    n0 = mesh_device.num_program_cache_entries()
    steps = 40
    for i in range(steps):
        perm = torch.randperm(LANES, generator=g)
        _write_logits(mesh_device, cfg, dev, x[perm])
        smp.set_params(*params(i))
        ctr = smp.set_positions(torch.arange(LANES) + 37 * i)
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
        rt = smp.read(tout)
        tok_t = _tokens_u32(tout)
        smp.set_counters(ctr)  # eager at the same counters
        eo = smp.sample(dev)
        re_ = smp.read(eo)
        assert torch.equal(rt.tokens, re_.tokens) and torch.equal(rt.flags, re_.flags), i
        for f in ("logprobs", "n_kept", "u", "mass_kept", "window_mass", "kth"):
            assert torch.equal(getattr(rt, f), getattr(re_, f)), (i, f)
        assert torch.equal(tok_t, _tokens_u32(eo))
        _free_out(eo)
    assert mesh_device.num_program_cache_entries() == n0, "a program was compiled after the capture"
    ttnn.release_trace(mesh_device, cap.tid)
    _free_out(tout)
    ttnn.deallocate(dev)
    smp.deallocate()
    print(f"[sampling] traced == eager on {steps} replays (new logits / params / positions each)")


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_sampler_distribution(mesh_device):
    """(a) 50,000 draws per group (8 groups of 4 lanes: real rows x lane parameters) vs the exact fp64 distribution
    (chi-square, TV vs the noise floor), and every draw = the fp64 inverse CDF with the same u."""
    import ttnn
    from scipy import stats

    S = _S()
    rows = _real_rows()
    lanes_src = _lanes_of(rows)
    def LP(t, p, k):  # fp32 values, as the plugin sends them
        return S.LaneParams(float(np.float32(t)), float(np.float32(p)), k, None)

    # (row, lane parameters): the rows' own top-p 0.95 lanes (kept 477, 258, 6, 3), T = 1.5 (kept 14), top-k 20 with
    # top-p off (kept 20), top-k 20 with ties at the k-th value (kept 23), top-k 100 + top-p 0.9 at T = 0.8 (kept 23)
    groups = [(2, lanes_src[2]), (7, lanes_src[7]), (15, lanes_src[15]), (8, lanes_src[8]),
              (15, LP(1.5, 0.95, 0)), (2, LP(1.0, 1.0, 20)), (28, LP(1.0, 1.0, 20)), (7, LP(0.8, 0.9, 100))]
    pick = [g[0] for g in groups]
    x = torch.stack([rows["logits"][pick[l // 4]] for l in range(LANES)])
    lanes = [groups[l // 4][1] for l in range(LANES)]
    draws = int(os.environ.get("MOTIF3_SAMPLER_DRAWS", "50000"))
    steps = draws // 4
    cfg, ccl, smp = _setup(mesh_device, "sampling_distribution")
    dev = _upload_logits(mesh_device, cfg, x)
    smp.set_params([l.temperature for l in lanes], [l.top_p for l in lanes], [l.top_k or VOCAB for l in lanes],
                   [500 + 13 * l for l in range(LANES)])
    smp.set_positions(torch.zeros(LANES))
    out = smp.sample(dev, debug=True)
    res = smp.read(out)
    assert not bool(res.flags.any())
    orders = _device_order(smp, out)
    _free_out(out)
    ref = []
    for r in range(len(pick)):
        ids, q = S.exact_nucleus(x[4 * r], lanes[4 * r], order=orders[4 * r])
        ref.append((ids, q, np.cumsum(q.numpy())))
    out = smp.sample(dev)
    _free_out(out)
    with _Capture(mesh_device) as cap:
        tout = smp.sample(dev)
    toks = np.zeros((steps, LANES), dtype=np.int64)
    mism = boundary = n_flag = tie_perm = 0
    bad_cases = []
    xf = x.float()
    t0 = time.perf_counter()
    for s in range(steps):
        ctr = smp.set_positions(torch.full((LANES,), s + 1))
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
        res = smp.read(tout)
        flg = res.flags.clone()
        u = smp.last_uniforms().astype(np.float64)
        if bool(flg.any()):  # rare exactness-verification flags: the production host fallback (exact, id tie order)
            n_flag += int(flg.sum())
            res = smp.resolve(res, x)
            for l in torch.nonzero(flg).reshape(-1).tolist():
                assert int(res.tokens[l]) == S.fallback_sample(x[l], lanes[l], float(u[l]))
        toks[s] = res.tokens.numpy()
        for r, (ids, q, C) in enumerate(ref):
            ls = slice(4 * r, 4 * r + 4)
            jj = np.minimum(np.searchsorted(C, u[ls], side="right"), len(C) - 1)
            want = ids.numpy()[jj]
            bad = np.nonzero(want != toks[s, ls])[0]
            for b in bad:
                l = 4 * r + int(b)
                got = int(toks[s, l])
                if float(xf[l, got]) == float(xf[l, int(want[b])]):  # tokens with identical logits: tie order
                    tie_perm += 1
                    continue
                near = np.abs(C - u[ls][b]).min()
                boundary += int(near < 2e-6)
                mism += int(near >= 2e-6)
                if near >= 2e-6 and len(bad_cases) < 12:
                    ids_l = ids.tolist()
                    bad_cases.append({"step": s, "lane": l, "group": r, "u": float(u[l]), "flagged": bool(flg[l]),
                                      "want_j": int(jj[b]), "got_j": ids_l.index(got) if got in ids_l else None,
                                      "ctr": ctr.copy(), "near": float(near)})
    wall = time.perf_counter() - t0
    ttnn.release_trace(mesh_device, cap.tid)
    _free_out(tout)
    for case in bad_cases:  # diagnose: the device walk of that lane at that counter (eager, debug)
        smp.set_counters(case.pop("ctr"))
        o = smp.sample(dev, debug=True)
        r_ = smp.read(o)
        l = case["lane"]
        EX = ttnn.to_torch(ttnn.get_device_tensors(o.debug["EX"])[0]).double().reshape(LANES, -1)[l]
        PR = ttnn.to_torch(ttnn.get_device_tensors(o.debug["PR"])[0]).double().reshape(LANES, -1)[l]
        mk = float(r_.mass_kept[l])
        ids, q, C = ref[case["group"]]
        j = case["want_j"]
        lo_, hi_ = max(0, j - 2), min(len(C), j + 3)
        case.update({"device_token": int(r_.tokens[l]), "mass_kept": mk, "n_kept": float(r_.n_kept[l]),
                     "u_dev": float(r_.u[l]), "C_exact": C[lo_:hi_].tolist(),
                     "EX_dev_norm": (EX[lo_ + 1: hi_ + 1] / mk).tolist(), "PR_dev": PR[lo_:hi_].tolist(),
                     "q_exact": q[lo_:hi_].tolist()})
        _free_out(o)
        print("[sampling] mismatch", json.dumps(case), flush=True)
    report = []
    for r, (ids, q, C) in enumerate(ref):
        got = toks[:, 4 * r : 4 * r + 4].reshape(-1)
        pos = {int(t): i for i, t in enumerate(ids.tolist())}
        assert all(int(t) in pos for t in np.unique(got)), f"group {groups[r]}: a token outside the exact kept set"
        cnt = np.bincount([pos[int(t)] for t in got], minlength=len(pos)).astype(np.float64)
        n = cnt.sum()
        exp_ = q.numpy() * n
        # chi-square with tail bins merged to expected >= 5
        o = np.argsort(-exp_)
        e_s, c_s = exp_[o], cnt[o]
        cut = int(np.searchsorted(-e_s, -5.0, side="right"))
        cut = max(1, min(cut, len(e_s)))
        e_b = np.append(e_s[:cut], e_s[cut:].sum()) if cut < len(e_s) else e_s
        c_b = np.append(c_s[:cut], c_s[cut:].sum()) if cut < len(e_s) else c_s
        keep = e_b > 0
        pval = stats.chisquare(c_b[keep], e_b[keep]).pvalue if keep.sum() > 1 else 1.0
        tv = 0.5 * np.abs(cnt / n - q.numpy()).sum()
        rng = np.random.default_rng(r)
        floor = [0.5 * np.abs(rng.multinomial(int(n), q.numpy()) / n - q.numpy()).sum() for _ in range(40)]
        fm, fs = float(np.mean(floor)), float(np.std(floor))
        lp_ = groups[r][1]
        report.append({"row": pick[r], "T": round(float(lp_.temperature), 4), "top_p": round(float(lp_.top_p), 4),
                       "top_k": int(lp_.top_k), "kept": len(pos), "draws": int(n), "chi2_p": float(pval),
                       "tv": float(tv), "tv_floor_mean": fm, "tv_floor_std": fs})
    print(f"[sampling] distribution: {steps} traced steps in {wall:.1f} s ({wall / steps * 1e3:.2f} ms per step incl. "
          f"host checks), {mism} mismatches, {boundary} rounding-boundary draws, {tie_perm} tie-order draws "
          f"(identical logits), {n_flag} verification-flagged draws (exact host fallback)")
    for line in report:
        print("  ", line)
    try:
        (REPORT_DIR / "device_distribution.json").write_text(json.dumps({"rows": report, "mismatches": mism,
                                                                         "boundary": boundary,
                                                                         "tie_order_draws": tie_perm,
                                                                         "flagged_draws": n_flag, "steps": steps},
                                                                        indent=1))
    except OSError:
        pass
    ttnn.deallocate(dev)
    smp.deallocate()
    for rr in report:
        assert rr["chi2_p"] > 1e-4, rr
        assert rr["tv"] <= rr["tv_floor_mean"] + 6 * rr["tv_floor_std"] + 1e-3, rr
    assert mism == 0, f"{mism} non-boundary token mismatches"
    assert n_flag <= 1e-3 * steps * LANES, f"{n_flag} flagged draws"


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_sampler_adversarial_rows(mesh_device):
    """Review batch (``_adversarial_rows``, T = 0.6, top-p 0.95), traced: the flat row's ~2e5-token nucleus is flagged
    on every step and its host fallback == ``exact_sample``; the near-tie top-2 rows (on a ~374-token nucleus and alone)
    and the exact top-2 tie are certified, every certified draw == the fp64 inverse CDF over THAT lane's device walk
    order fed the host-twin uniform, every resolved draw == ``fallback_sample``, and the empirical distributions (all
    draws: flagged ones resolved and included -- excluding them biases peaked rows, whose low-probability tail is what
    the draw check flags) pass chi-square / binomial tests; greedy on the exact tie = the lower id; top-k 1 / top-p 0
    (the plugin's top_k = 1 encoding) is one fixed member of the tie."""
    import ttnn
    from scipy import stats

    S = _S()
    ad = _adversarial_rows()
    f32 = lambda v: float(np.float32(v))  # noqa: E731
    lp = S.LaneParams(f32(0.6), f32(0.95), 0, 1)
    plan = [("flat", lp)] * 4 + [("window", lp)] * 8 + [("pure", lp)] * 8 + [("tie", lp)] * 8
    plan += [("tie", S.LaneParams(0.0, 1.0, 1, None))] * 2 + [("tie", S.LaneParams(f32(0.6), 0.0, 1, 1))]
    plan += [("flat", S.LaneParams(f32(0.6), f32(0.95), 0, None))]  # unseeded
    x = torch.stack([ad[n] for n, _ in plan])
    lanes = [l for _, l in plan]
    seeds = [None if l.seed is None else 7001 + 13 * i for i, l in enumerate(lanes)]
    steps = int(os.environ.get("MOTIF3_SAMPLER_ADV_STEPS", "3000"))
    cfg, ccl, smp = _setup(mesh_device, "sampling_adversarial")
    dev = _upload_logits(mesh_device, cfg, x)
    smp.set_params([l.temperature for l in lanes], [l.top_p for l in lanes], [l.top_k or VOCAB for l in lanes], seeds)
    lanes = smp.lane_params
    smp.set_positions(torch.full((LANES,), 3000))
    out = smp.sample(dev, debug=True)
    res = smp.read(out)
    orders = _device_order(smp, out)
    _free_out(out)
    flat_lanes = [i for i, (n, _) in enumerate(plan) if n == "flat"]
    flagged = torch.nonzero(res.flags).reshape(-1).tolist()
    assert set(flat_lanes) <= set(flagged) and len(flagged) <= len(flat_lanes) + 1, flagged  # + a rare draw-check flag
    ref = {}
    for l in range(LANES):
        if l in flat_lanes or lanes[l].greedy:
            continue
        ids, q = S.exact_nucleus(x[l], lanes[l], order=orders[l])
        assert ids.numel() == S.exact_nucleus(x[l], lanes[l])[0].numel(), l
        assert bool(res.flags[l]) or int(res.n_kept[l]) == ids.numel(), (l, float(res.n_kept[l]), ids.numel())
        ref[l] = (ids, np.cumsum(q.numpy()))
    out = smp.sample(dev)
    _free_out(out)
    with _Capture(mesh_device) as cap:
        tout = smp.sample(dev)
    toks = np.zeros((steps, LANES), dtype=np.int64)
    mism = n_res = fb_checked = 0
    for s in range(steps):
        smp.set_positions(torch.full((LANES,), 3001 + s))
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
        r = smp.read(tout)
        u = smp.last_uniforms()
        assert np.array_equal(r.u.numpy().astype(np.float32), u)
        assert all(bool(r.flags[l]) for l in flat_lanes)
        flg = r.flags.clone()
        flg[flat_lanes] = False
        if s % 300 == 0:  # the flat lanes' fallback (a full sort each: sparse)
            r = smp.resolve(r, x, lanes=flat_lanes)
            for l in flat_lanes:
                assert int(r.tokens[l]) == S.exact_sample(x[l], lanes[l], float(u[l]))
                fb_checked += 1
        if bool(flg.any()):  # draw-check flags of certifiable lanes: resolved and kept in the statistics
            r = smp.resolve(r, x, lanes=torch.nonzero(flg).reshape(-1).tolist())
        for l in range(LANES):
            got = int(r.tokens[l])
            if lanes[l].greedy:
                mism += int(got != SLICE * 3 + 7)
            elif l in ref:
                if bool(flg[l]):
                    n_res += 1
                    mism += int(got != S.fallback_sample(x[l], lanes[l], float(u[l])))
                else:
                    ids, C = ref[l]
                    mism += int(got != int(ids[min(int(np.searchsorted(C, float(u[l]), side="right")), len(C) - 1)]))
            toks[s, l] = got
    ttnn.release_trace(mesh_device, cap.tid)
    _free_out(tout)
    report = {"mismatches": mism, "resolved_draw_flags": n_res, "flat_fallbacks_checked": fb_checked, "groups": {}}
    for name, ls in (("window", range(4, 12)), ("pure", range(12, 20)), ("tie", range(20, 28))):
        ls = list(ls)
        l0 = ls[0]
        ids, q = S.exact_nucleus(x[l0], lanes[l0], order=orders[l0])  # the device walk order (its boundary tie pick)
        for l in ls[1:]:  # identical rows -> identical device walk orders (the tie order does not depend on the lane)
            assert torch.equal(orders[l][: ids.numel()], orders[l0][: ids.numel()]), (name, l)
        # categories: every kept token above the boundary value, plus the boundary tie group merged (vLLM keeps an
        # arbitrary subset of fixed size of it; a resolved draw picks its members in id order, the device in its own)
        xf = x[l0].float()
        v_last = float(xf[ids[-1]])
        above = [int(t) for t in ids.tolist() if float(xf[t]) != v_last]
        cat = {t: i for i, t in enumerate(above)}
        qa = q.numpy()[: len(above)]
        probs = np.append(qa, q.numpy()[len(above) :].sum())
        got = toks[:, ls].reshape(-1)
        idx = [cat[int(t)] if int(t) in cat else (len(above) if float(xf[int(t)]) == v_last else -1) for t in got]
        assert min(idx) >= 0, f"{name}: a token outside the exact kept set (boundary tie group included)"
        cnt = np.bincount(idx, minlength=len(probs)).astype(np.float64)
        n = cnt.sum()
        e = probs * n
        big = e >= 5
        c_b, e_b = np.append(cnt[big], cnt[~big].sum()), np.append(e[big], e[~big].sum())
        keep = e_b > 0
        chi_p = float(stats.chisquare(c_b[keep], e_b[keep]).pvalue)
        k1, k2 = int((got == int(ids[0])).sum()), int((got == int(ids[1])).sum())
        p12 = float(q[0] / (q[0] + q[1]))
        bin_p = float(stats.binomtest(k1, k1 + k2, p12).pvalue)
        report["groups"][name] = {"kept": int(ids.numel()), "boundary_tie_kept": int(ids.numel()) - len(above),
                                  "draws": int(n), "chi2_p": chi_p, "top2": [k1, k2], "p_first": p12,
                                  "binom_p": bin_p}
        assert chi_p > 1e-4 and bin_p > 1e-4, (name, report["groups"][name])
    report["topk1_tokens"] = sorted(set(toks[:, 30].tolist()))
    assert len(report["topk1_tokens"]) == 1 and report["topk1_tokens"][0] in (SLICE * 30 + 100, SLICE * 3 + 7)
    print(f"[sampling] adversarial rows: {json.dumps(report)}")
    ttnn.deallocate(dev)
    smp.deallocate()
    assert mism == 0, report
    assert n_res <= 1e-3 * steps * len(ref), report


def _traced_ms(mesh_device, fn, reps=200, n=1):
    import ttnn

    o = fn()
    _free_out(o)
    ttnn.synchronize_device(mesh_device)
    with _Capture(mesh_device) as cap:
        outs = [fn() for _ in range(n)]
    try:
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) * 1e3 / reps / n
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
        for o in outs:
            _free_out(o)


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_sampler_latency(mesh_device):
    """(e) Traced latency of the whole sampler at 32 lanes and the per-step host costs."""
    import ttnn

    rows = _real_rows()
    budget = float(os.environ.get("MOTIF3_SAMPLER_BUDGET_MS", "1.0"))
    cfg, ccl, smp = _setup(mesh_device, "sampling_latency")
    dev = _upload_logits(mesh_device, cfg, rows["logits"])
    smp.set_params(rows["temperature"], rows["top_p"], rows["top_k"], rows["seed"])
    rep = {"traced_ms": _traced_ms(mesh_device, lambda: smp.sample(dev))}
    # host side of one step: counter write, replay, result read; a fallback step adds the host logits row work
    out = smp.sample(dev)
    _free_out(out)
    with _Capture(mesh_device) as cap:
        tout = smp.sample(dev)
    tw, tr, tt = [], [], []
    for s in range(100):
        t0 = time.perf_counter()
        smp.set_positions(torch.full((LANES,), s))
        t1 = time.perf_counter()
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
        res = smp.read(tout)
        t2 = time.perf_counter()
        tw.append((t1 - t0) * 1e3)
        tr.append((t2 - t1) * 1e3)
        tt.append((t2 - t0) * 1e3)
    t0 = time.perf_counter()
    smp.resolve(res, rows["logits"], lanes=[0, 1, 30])
    rep["fallback_3_lanes_host_ms"] = (time.perf_counter() - t0) * 1e3
    ttnn.release_trace(mesh_device, cap.tid)
    _free_out(tout)
    rep.update({"set_positions_ms_median": float(np.median(tw)), "replay_plus_read_ms_median": float(np.median(tr)),
                "step_host_total_ms_median": float(np.median(tt))})
    t0 = time.perf_counter()
    smp.set_params(rows["temperature"], rows["top_p"], rows["top_k"], rows["seed"], force=True)
    rep["set_params_upload_ms"] = (time.perf_counter() - t0) * 1e3
    print(f"[sampling] latency: {json.dumps(rep)}")
    try:
        (REPORT_DIR / "device_latency.json").write_text(json.dumps(rep, indent=1))
    except OSError:
        pass
    ttnn.deallocate(dev)
    smp.deallocate()
    assert rep["traced_ms"] <= budget, f"traced sampler {rep['traced_ms']:.3f} ms > budget {budget} ms"


@pytest.mark.skipif(os.environ.get("MOTIF3_SAMPLER_PROBE") != "1", reason="opt-in probe (MOTIF3_SAMPLER_PROBE=1)")
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_probe_sampler_variants(mesh_device):
    """Opt-in: traced latency of every implementation variant and option (K, W, logprobs, rng, impl), with the
    tokens of each variant checked against the default's on the real rows."""
    import ttnn

    from models.demos.motif3.tt.sampling import MotifDeviceSampler

    rows = _real_rows()
    cfg, ccl, base = _setup(mesh_device, "sampling_probe")
    dev = _upload_logits(mesh_device, cfg, rows["logits"])
    args = (rows["temperature"], rows["top_p"], rows["top_k"], rows["seed"])
    base.set_params(*args)
    ctr = base.set_positions(torch.full((LANES,), 5))
    o = base.sample(dev)
    ref = base.read(o)
    _free_out(o)
    variants = [
        ("default", {}),
        ("no_fuse", {"impl": {"fuse": False}}),
        ("no_repair", {"impl": {"repair": False}}),
        ("no_verify_no_repair (cost of exactness; NOT exact)", {"impl": {"verify": False, "repair": False}}),
        ("rowsum=row", {"impl": {"rowsum": "row"}}),
        ("fused_exp", {"impl": {"fused_exp": True}}),
        ("prefix=1mm", {"impl": {"prefix": "1mm"}}),
        ("rng=host", {"rng": "host"}),
        ("no_logprobs", {"logprobs": False}),
        ("no_counter", {"coverage_counter": False}),
        ("K=32", {"local_k": 32}),
        ("K=128", {"local_k": 128}),
        ("W=256", {"window": 256}),
        ("W=1024", {"window": 1024}),
        ("l1_intermediates", {"l1_intermediates": True}),
        ("lean", {"rng": "host", "logprobs": False, "coverage_counter": False}),
        ("lean_l1", {"rng": "host", "logprobs": False, "coverage_counter": False, "l1_intermediates": True}),
    ]
    rep = {}
    # per-stage traced latency of the default sampler (each stage captured alone on the eager intermediates)
    tmp = []
    d = base.stages(dev, tmp)
    ttnn.synchronize_device(mesh_device)
    stage_fns = (
        ("local_topk_gather", lambda: base._stage_local(d["x"], [])),
        ("normalizer", lambda: base._stage_norm(d["x"], d["M"], [])),
        ("window_topk_probs", lambda: base._stage_window(d["Vb"], d["M"], d["Z"], [])),
        ("prefix_matmuls", lambda: base._stage_prefix(d["PR"], [])),
        ("select_repair_rng_cert_verify", lambda: base._stage_select(d, [])),
        ("outputs", lambda: base._stage_outputs(d, [])),
    )
    stages_ms = {}
    for name, fn in stage_fns:
        def run(fn=fn):
            o = fn()
            vals = o.values() if isinstance(o, dict) else o
            for t in vals:
                if t is not None:
                    ttnn.deallocate(t)
        run()
        ttnn.synchronize_device(mesh_device)
        with _Capture(mesh_device) as cap:
            run()
        try:
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(200):
                ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            stages_ms[name] = (time.perf_counter() - t0) * 1e3 / 200
        finally:
            ttnn.release_trace(mesh_device, cap.tid)
        print(f"[sampling] stage {name}: {stages_ms[name]:.3f} ms", flush=True)
    rep["stages_ms"] = stages_ms
    for t in {id(t): t for t in tmp}.values():
        ttnn.deallocate(t)
    for name, kw in variants:
        try:
            smp = MotifDeviceSampler(mesh_device, cfg, ccl=ccl, **kw)
            smp.set_params(*args)
            smp.set_counters(ctr)
            o = smp.sample(dev)
            r = smp.read(o)
            _free_out(o)
            same = bool(torch.equal(r.tokens[~r.flags], ref.tokens[~r.flags]))
            ms = _traced_ms(mesh_device, lambda: smp.sample(dev))
            rep[name] = {"traced_ms": ms, "tokens_equal_default": same, "flags": r.flags.nonzero().reshape(-1).tolist()}
            smp.deallocate()
        except Exception as e:  # report and continue
            rep[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"[sampling] probe {name}: {rep[name]}", flush=True)
    try:
        (REPORT_DIR / "device_variants.json").write_text(json.dumps(rep, indent=1))
    except OSError:
        pass
    ttnn.deallocate(dev)
    base.deallocate()
