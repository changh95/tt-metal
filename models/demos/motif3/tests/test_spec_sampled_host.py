# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase G (C10a, ``MOTIF3_SPEC_SAMPLED``): speculative sampling on the MTP launch, host only.

The MTP draft ``d`` is greedy (``q = onehot(d)``), so lossless rejection sampling is "draw ``x ~ p`` with the exact
sampler and noise independent of ``d``; accept iff ``x == d``; on a reject commit ``x``; the draft row's draw ``y`` is
the bonus" (``logs/opt/phaseG/DESIGN.md`` §3.1). These tests run that protocol on the device sampler's own host twins
(``tt/sampling.py``: the per-lane semantics temperature -> top-k -> top-p, the counter RNG ``(seed, position)``, the
53-bit uniform of the exact fallback, the Gumbel-max full-vocabulary path):

* seeded identity: the spec walk commits, seed for seed, exactly what plain sequential decoding commits (anchor
  counters at ``n``, draft-row counters at ``n + 1``); an off-by-one in the draft-row position is detected;
* distribution: with unseeded (fresh, independent) noise, the committed two-token joint law matches the exact target
  law (chi-square and TV inside multinomial noise) and the acceptance rate is ``p0(d)``;
* the host helpers of the generator and the bridge: the partner-lane parameter copy of a packed T32 verify, the draft
  sampler's RNG domain, the verify-trace routing, the owner-order ``sampled_ids`` / ``mtp_input`` mapping and its
  greedy cross-check counter, the knob, the bridge's per-class acceptance and the ``sampled_ids`` verify.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.demos.motif3.tt import generator_api as api
from models.demos.motif3.tt import sampling as S
from models.demos.motif3.tt import verify_plan as VP

V = 64
CASES = {
    "T0.6 gumbel": S.LaneParams(temperature=0.6, top_p=1.0, top_k=0),
    "T1.0 gumbel": S.LaneParams(temperature=1.0, top_p=1.0, top_k=0),
    "T1.0 top-p 0.95": S.LaneParams(temperature=1.0, top_p=0.95, top_k=0),
    "T0.6 top-p 0.95": S.LaneParams(temperature=0.6, top_p=0.95, top_k=0),
    "T1.0 top-k 20": S.LaneParams(temperature=1.0, top_p=1.0, top_k=20),
    "T0.8 top-k 20 top-p 0.8": S.LaneParams(temperature=0.8, top_p=0.8, top_k=20),
}


class ToyTarget:
    """Logits after ``(token, position)``: a fixed random row per key (bf16-representable), plus a "MTP" drafter that
    is the argmax of a perturbed row (a greedy, point-mass draft that is often but not always the target's mode)."""

    def __init__(self, seed: int = 0, noise: float = 1.0):
        g = torch.Generator().manual_seed(seed)
        self.table = (torch.randn(257, V, generator=g) * 2.5).to(torch.bfloat16).float()
        self.mtp_noise = torch.randn(257, V, generator=g) * noise

    def key(self, token: int, position: int) -> int:
        return (int(token) * 31 + int(position) * 7 + 11) % 257

    def logits(self, token: int, position: int) -> torch.Tensor:
        return self.table[self.key(token, position)]

    def draft(self, token: int, position: int) -> int:
        k = self.key(token, position)
        return int(torch.argmax(self.table[k] + self.mtp_noise[k]))


def twin_sample(row: torch.Tensor, lane: S.LaneParams, ctr) -> int:
    """The device sampler's draw for one lane (certified device token or exact host fallback: both are the exact
    sampler at the lane's counters): Gumbel-max for full-support lanes, else the exact inverse CDF at the 53-bit
    uniform of the counters."""
    ctr = np.asarray(ctr, dtype=np.uint32).reshape(2)
    if lane.greedy:
        return S.greedy_token(row)
    if lane.full_support:
        return S.gumbel_sample(row, lane, ctr)
    u = float(S.uniform64(ctr.reshape(1, 2))[0])
    return S.exact_sample(row, lane, u)


def seeded_ctr(seed: int, position: int):
    return S.lane_counters(seed, position)


def plain_decode(target: ToyTarget, lane, first_tok: int, pos0: int, steps: int, ctr_of):
    """Plain sequential decoding: the token after the one at ``n`` is drawn with counters at ``n``."""
    out, tok, n = [], first_tok, pos0
    for _ in range(steps):
        tok = twin_sample(target.logits(tok, n), lane, ctr_of(n))
        out.append(tok)
        n += 1
    return out


def spec_decode(target: ToyTarget, lane, first_tok: int, pos0: int, steps: int, ctr_a, ctr_b, *, draft_pos_shift=1):
    """The bridge / plugin protocol with K = 1: every step verifies ``[anchor, draft]``; the anchor row samples ``x``
    with the anchor counters at ``n``, the draft row samples ``y`` after ``d`` with the draft-row counters at ``n + 1``
    (``draft_pos_shift``); the id walk commits ``[d, y]`` iff ``x == d``, else ``[x]``; the next draft is the MTP's
    after the last committed token."""
    out, tok, n = [], first_tok, pos0
    accepted = verified = 0
    d = target.draft(tok, n)
    while len(out) < steps:
        x = twin_sample(target.logits(tok, n), lane, ctr_a(n))
        y = twin_sample(target.logits(d, n + 1), lane, ctr_b(n + draft_pos_shift))
        verified += 1
        if x == d:
            accepted += 1
            out += [d, y]
            tok, n = y, n + 2
        else:
            out.append(x)
            tok, n = x, n + 1
        d = target.draft(tok, n)
    return out[:steps], accepted, verified


# ------------------------------------------------------------------------------------------------------------------
# seeded identity
# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(CASES))
def test_seeded_spec_walk_equals_plain_decoding(case):
    lane = S.LaneParams(**{**CASES[case].__dict__, "seed": None})
    target = ToyTarget()
    same = acc = ver = 0
    seeds = range(300)
    for seed in seeds:
        lane_s = S.LaneParams(lane.temperature, lane.top_p, lane.top_k, seed)
        ctr = lambda n, s=seed: seeded_ctr(s, n)  # noqa: E731
        plain = plain_decode(target, lane_s, 3, 10, 12, ctr)
        spec, a, v = spec_decode(target, lane_s, 3, 10, 12, ctr, ctr)
        same += int(spec == plain)
        acc, ver = acc + a, ver + v
    assert same == len(seeds), f"{case}: {len(seeds) - same} seeds diverged from plain decoding"
    assert 0 < acc < ver, f"{case}: acceptance {acc}/{ver} does not exercise both branches"


def test_a_draft_row_keyed_at_the_anchor_position_breaks_identity():
    """The draft row's counters must be the ones plain decoding uses at ``n + 1``: keyed at ``n`` the accepted
    branch's bonus reuses the anchor's noise and the walk diverges from plain decoding (the gate is sensitive)."""
    lane = CASES["T1.0 top-p 0.95"]
    target = ToyTarget()
    diverged = 0
    for seed in range(200):
        lane_s = S.LaneParams(lane.temperature, lane.top_p, lane.top_k, seed)
        ctr = lambda n, s=seed: seeded_ctr(s, n)  # noqa: E731
        plain = plain_decode(target, lane_s, 3, 10, 12, ctr)
        spec, _, _ = spec_decode(target, lane_s, 3, 10, 12, ctr, ctr, draft_pos_shift=0)
        diverged += int(spec != plain)
    assert diverged > 0


# ------------------------------------------------------------------------------------------------------------------
# distribution (unseeded: independent fresh noise for the anchor and the draft rows)
# ------------------------------------------------------------------------------------------------------------------
def _exact_probs(row: torch.Tensor, lane: S.LaneParams) -> torch.Tensor:
    ids, q = S.exact_nucleus(row, lane)
    p = torch.zeros(V, dtype=torch.float64)
    p[ids.long()] = q.double() / q.double().sum()
    return p


def _chi2_tv(counts: np.ndarray, law: np.ndarray, n: int):
    exp = law * n
    big = exp >= 5
    chi = float((((counts[big] - exp[big]) ** 2) / exp[big]).sum())
    rest_o, rest_e = counts[~big].sum(), exp[~big].sum()
    dof = int(big.sum()) - 1
    if rest_e > 0:
        chi += float((rest_o - rest_e) ** 2 / rest_e)
        dof += 1
    tv = 0.5 * float(np.abs(counts / n - law).sum())
    rng = np.random.default_rng(3)
    null = [0.5 * np.abs(rng.multinomial(n, law) / n - law).sum() for _ in range(200)]
    return chi, max(dof, 1), tv, float(np.quantile(null, 0.999))


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("draft", ["mtp", "mode", "unlikely"])
def test_unseeded_spec_walk_commits_the_target_law(case, draft):
    lane = CASES[case]
    target = ToyTarget(seed=1)
    tok, n0 = 5, 20
    row0 = target.logits(tok, n0)
    p0 = _exact_probs(row0, lane)
    d = {"mtp": target.draft(tok, n0), "mode": int(torch.argmax(row0)), "unlikely": int(torch.argsort(row0)[V // 2])}[
        draft
    ]
    # exact joint law of the first two committed tokens (t1, t2)
    law = np.zeros((V, V))
    for t1 in range(V):
        if p0[t1] > 0:
            law[t1] = (p0[t1] * _exact_probs(target.logits(t1, n0 + 1), lane)).numpy()
    law = law.reshape(-1)
    law /= law.sum()
    N = 6000 if lane.full_support else 20000
    rng_a, rng_b = np.random.default_rng(11), np.random.default_rng(12)  # independent streams (sampler A / B)
    counts = np.zeros(V * V)
    acc = 0
    for _ in range(N):
        x = twin_sample(row0, lane, rng_a.integers(0, 1 << 32, size=2, dtype=np.uint64))
        y = twin_sample(target.logits(d, n0 + 1), lane, rng_b.integers(0, 1 << 32, size=2, dtype=np.uint64))
        if x == d:
            acc += 1
            t1, t2 = d, y
        else:  # reject: commit x; the next step's anchor draws position n0 + 2's token with fresh noise
            t1 = x
            t2 = twin_sample(target.logits(t1, n0 + 1), lane, rng_a.integers(0, 1 << 32, size=2, dtype=np.uint64))
        counts[t1 * V + t2] += 1
    chi, dof, tv, tv_q = _chi2_tv(counts, law, N)
    z = (chi - dof) / math.sqrt(2 * dof)
    assert z < 5.0, f"{case} / {draft}: chi2 {chi:.1f} on {dof} dof (z {z:.2f})"
    assert tv <= tv_q, f"{case} / {draft}: TV {tv:.4f} above the null's 99.9th percentile {tv_q:.4f}"
    pd = float(p0[d])
    assert abs(acc / N - pd) <= 5 * math.sqrt(max(pd * (1 - pd), 1e-12) / N) + 1e-9, (acc / N, pd)


# ------------------------------------------------------------------------------------------------------------------
# host helpers
# ------------------------------------------------------------------------------------------------------------------
def test_knob_parses_and_defaults_off():
    assert api.spec_sampled_from_env({}) is False
    assert api.spec_sampled_from_env({"MOTIF3_SPEC_SAMPLED": "on"}) is True
    assert api.spec_sampled_from_env({"MOTIF3_SPEC_SAMPLED": " OFF "}) is False
    with pytest.raises(ValueError, match="MOTIF3_SPEC_SAMPLED"):
        api.spec_sampled_from_env({"MOTIF3_SPEC_SAMPLED": "1"})
    assert api.GeneratorSettings().spec_sampled is False
    with pytest.raises(TypeError):
        api.GeneratorSettings(spec_sampled="on")


def _batch(positions, drafts):
    n = api.NUM_LANES
    pos = torch.full((n,), -1, dtype=torch.int32)
    dr = torch.full((n,), -1, dtype=torch.int32)
    for lane, p in positions.items():
        pos[lane] = p
    for lane, d in drafts.items():
        dr[lane] = d
    pt = torch.zeros(n, 4, dtype=torch.int32)
    pt[pos >= 0] = 1
    return api.SpecDecodeBatch(
        tokens=torch.zeros(n, dtype=torch.int32), positions=pos, draft_tokens=dr, page_table=pt
    )


def test_a_sampled_verify_routes_like_a_greedy_one_when_the_traces_sample():
    full = _batch({lane: 5 for lane in range(32)}, {lane: 7 for lane in range(32)})  # no idle lane: T64
    fits = _batch({0: 5}, {0: 7})  # one draft, 31 idle lanes: T32
    sampling = ([1.0] * 32, [1.0] * 32, [0] * 32, [None] * 32)
    kw = dict(kv_mode="all_split")
    assert VP.choose_verify_kind(full, "auto", False, sampling, **kw) == VP.SPEC_STEP  # Phase F rule
    assert VP.choose_verify_kind(full, "auto", False, sampling, sampled_verify=True, **kw) == VP.WIDE_STEP
    assert VP.choose_verify_kind(fits, "auto", False, sampling, sampled_verify=True, **kw) == VP.SPEC_STEP
    ordinary = _batch({0: 5}, {})
    assert VP.choose_verify_kind(ordinary, "auto", False, sampling, sampled_verify=True, **kw) == VP.SPEC_STEP


def test_partner_lane_takes_its_owner_parameters_and_seed():
    from models.demos.motif3.tt.generator import _greedy_lane_sampling, _partner_lane_sampling

    t, p, k, sd = _greedy_lane_sampling()
    t[3], p[3], k[3], sd[3] = 0.7, 0.9, 20, 1234
    t[4], p[4], k[4], sd[4] = 1.0, 1.0, 0, None
    pt, pp, pk, ps = _partner_lane_sampling((t, p, k, sd), {3: 17, 4: 30})
    assert (pt[17], pp[17], pk[17], ps[17]) == (0.7, 0.9, 20, 1234)
    assert (pt[30], pp[30], pk[30], ps[30]) == (1.0, 1.0, 0, None)
    assert (pt[3], ps[3]) == (0.7, 1234) and (pt[5], ps[5]) == (0.0, None)
    assert t[17] == 0.0, "the caller's lists are not modified"


def test_draft_sampler_rng_is_domain_separated_and_deterministic():
    from models.demos.motif3.tt.generator import sampler_b_rng_seed

    assert sampler_b_rng_seed(None) is None
    assert sampler_b_rng_seed(0) == sampler_b_rng_seed(0) != 0
    assert sampler_b_rng_seed(0) != sampler_b_rng_seed(1)
    assert 0 <= sampler_b_rng_seed(2**70) < 2**64


def _fake_gen():
    return SimpleNamespace(stats={k: 0 for k in ("greedy_s_ne_a", "sampled_verify_steps", "sampled_verify_rows",
                                                 "sampled_drafts")},
                           _warned_once=set(), log=lambda msg: None)  # fmt: skip


def _res(tokens):
    return SimpleNamespace(tokens=torch.tensor(tokens, dtype=torch.int64))


@pytest.mark.parametrize("wide", [False, True])
def test_sampled_ids_and_mtp_input_in_owner_order(wide):
    """Column 0 from the owner's anchor draw, column 1 from its partner lane (T32) or its draft row (T64); a greedy
    row carries its argmax; resolved tokens (host fallback) go to sampled_ids, the device draws to mtp_input."""
    from models.demos.motif3.tt.generator import MotifGenerator

    n = api.NUM_LANES
    batch = _batch({0: 5, 1: 9, 2: 4}, {0: 70, 1: 71})
    plan = SimpleNamespace(batch=batch, partner_of={} if wide else {0: 10, 1: 11})
    argmax = torch.full((n, 2), -1, dtype=torch.int32)
    argmax[0] = torch.tensor([100, 101])
    argmax[1] = torch.tensor([200, 201])
    argmax[2] = torch.tensor([300, -1])
    res = SimpleNamespace(argmax=argmax)
    temps = [0.0] * n
    temps[0] = 1.0  # lane 0 sampled; lanes 1, 2 greedy
    sampling = (temps, [1.0] * n, [0] * n, [None] * n)
    dev_a = list(range(1000, 1000 + n))
    dev_a[1], dev_a[2] = 200, 300  # greedy draws equal the argmax
    tok_a = list(dev_a)
    tok_a[0] = 555  # lane 0's anchor resolved on the host to another token
    if wide:
        dev_b = list(range(2000, 2000 + n))
        dev_b[1] = 201
        tok_b = list(dev_b)
        res_b, db = _res(tok_b), torch.tensor(dev_b)
    else:
        dev_a[11] = 201  # lane 1's partner lane draws its argmax
        tok_a[11] = 201
        res_b = db = None
    gen = _fake_gen()
    sid, mtp = MotifGenerator._spec_sampled_ids(
        gen, plan, wide, res, sampling, _res(tok_a), res_b, torch.tensor(dev_a), db, True
    )
    d0 = (2000 if wide else 1010)
    assert sid[0].tolist() == [555, d0] and mtp[0].tolist() == [1000, d0]
    assert sid[1].tolist() == [200, 201] and mtp[1].tolist() == [200, 201]
    assert sid[2].tolist() == [300, -1] and mtp[2].tolist() == [300, -1]
    assert sid[3].tolist() == [-1, -1]
    assert gen.stats["greedy_s_ne_a"] == 0
    assert gen.stats["sampled_verify_rows"] == 1 and gen.stats["sampled_drafts"] == 1
    # a greedy row whose device draw is not the argmax is counted, and still commits the argmax
    dev_a[2] = 301
    sid, _ = MotifGenerator._spec_sampled_ids(
        gen, plan, wide, res, sampling, _res(tok_a), res_b, torch.tensor(dev_a), db, False
    )
    assert sid[2].tolist() == [300, -1]
    assert gen.stats["greedy_s_ne_a"] == 1


def test_bridge_per_class_acceptance():
    from models.demos.motif3.tt.generator_vllm import SpecStats, _hot_lanes

    st = SpecStats(accepted=70, rejected=30, accepted_sampled=30, rejected_sampled=20)
    assert st.class_acceptance(False, 0.85, 64) == pytest.approx((40 + 64 * 0.85) / (50 + 64))
    assert st.class_acceptance(True, 0.85, 64) == pytest.approx((30 + 64 * 0.85) / (50 + 64))
    assert _hot_lanes(None) is None
    assert _hot_lanes(([0.0, 1e-6, 1e-5, 0.7], None, None, None)) == [False, False, True, True]


# ------------------------------------------------------------------------------------------------------------------
# the bridge: spec_plan and a sampled_ids verify
# ------------------------------------------------------------------------------------------------------------------
def test_spec_plan_offers_sampled_ids_only_with_the_knob(monkeypatch, tmp_path):
    from models.demos.motif3.tests import test_generator_vllm_host as T
    from models.demos.motif3.tt import generator_vllm as gv

    for var in ("MOTIF3_WEIGHTS_DIR", "HF_MODEL", "TT_CACHE_PATH", "MOTIF3_KV_CACHE_DTYPE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MOTIF3_TT_CACHE_PATH", str(T._mtp_cache_marker(tmp_path)))
    vc = T._spec_vllm_config(T._fake_checkpoint(tmp_path / "ckpt"))
    monkeypatch.delenv("MOTIF3_SPEC_SAMPLED", raising=False)
    assert gv.MotifForCausalLM.spec_plan(vc, 32, 1).accept_modes == ("argmax_ids",)
    monkeypatch.setenv("MOTIF3_SPEC_SAMPLED", "on")
    assert gv.MotifForCausalLM.spec_plan(vc, 32, 1).accept_modes == ("argmax_ids", "sampled_ids")


def _sampled_gen_cls():
    from models.demos.motif3.tests import test_generator_vllm_host as T
    from models.demos.motif3.tt.generator import SampledSpecDecodeResult

    class FakeSampledGen(T.FakeMotifGenerator):
        """A verify with sampling: column j of a hot lane is a deterministic "draw" ``(argmax + 1) % V`` (column 0)
        and the argmax (column 1); greedy lanes return the argmax. mtp_input = the draws."""

        def decode_forward_spec(self, batch, *, kv_cache, enable_trace, want_logits, sampling=None):
            if sampling is None or not batch.is_verify:
                return super().decode_forward_spec(
                    batch, kv_cache=kv_cache, enable_trace=enable_trace, want_logits=want_logits, sampling=sampling
                )
            res = super().decode_forward_spec(batch, kv_cache=kv_cache, enable_trace=enable_trace, want_logits=False)
            self.verify_sampling = sampling
            sid = res.argmax.clone()
            for lane, t in enumerate(sampling[0]):
                if t >= 1e-5 and int(batch.positions[lane]) >= 0:
                    sid[lane, 0] = (int(res.argmax[lane, 0]) + 1) % self._vocab
            return SampledSpecDecodeResult(
                logits=None, argmax=res.argmax, mtp_argmax=res.mtp_argmax, sample=None, sampled_ids=sid,
                mtp_input=sid.clone(),
            )  # fmt: skip

    return FakeSampledGen


@pytest.mark.parametrize("knob", [True, False])
def test_bridge_sampled_ids_verify(knob):
    from vllm_tt_plugin.async_decode import _verify_output_tensor

    from models.demos.motif3.tests import test_generator_vllm_host as T

    bridge, gen = T._bridge(num_slots=4, gen_cls=_sampled_gen_cls(), spec_tokens=1, spec_sampled=knob)
    kv = bridge.allocate_kv_cache((640, 1, 32, 576), torch.bfloat16, 3)
    d = T.PluginDriver(bridge, kv, num_slots=4, block_size=32, num_blocks=640, width=kv.page_table_width, vocab=4096)
    d.add("g", [101, 102, 103, 104, 105])
    d.add("h", [7, 8, 9])
    d.prefill(["g", "h"])
    gen.device_sampling = True
    kwargs = d._decode_common(["g", "h"])
    tokens = torch.zeros(4, 2, dtype=torch.int32)
    positions = torch.full((4, 2), -1, dtype=torch.int32)
    for i, r in enumerate(("g", "h")):
        n = len(d.seqs[r]) - 1
        tokens[i] = torch.tensor([d.seqs[r][-1], 77])
        positions[i] = torch.tensor([n, n + 1])
    sp = SimpleNamespace(temperature=[0.0, 1.0, 0.0, 0.0], top_p=[1.0] * 4, top_k=[1] * 4, seed=[None, 5, None, None])
    call = dict(tokens=tokens, start_pos=positions, num_valid_drafts=torch.tensor([1, 1, 0, 0], dtype=torch.int32),
                accepted_counts=torch.ones(4, dtype=torch.int32), spec_mode="sampled_ids", sampling_params=sp,
                **kwargs)  # fmt: skip
    if not knob:
        with pytest.raises(NotImplementedError, match="did not offer"):
            bridge.decode_forward(**call)
        return
    out = bridge.decode_forward(**call)
    ids = _verify_output_tensor(out, "MotifForCausalLM", "sampled_ids")
    a = {r: T.next_token(d.seqs[r], 4096) for r in ("g", "h")}
    assert int(ids[0, 0]) == a["g"], "a greedy row carries the argmax"
    assert int(ids[1, 0]) == (a["h"] + 1) % 4096, "a sampled row carries its draw"
    assert ids[2:].tolist() == [[0, -1]] * 2
    lane_h = bridge._lanes.slot_to_lane[1]
    st = bridge._retained[lane_h]
    assert st.sampled and st.argmax[0] == (a["h"] + 1) % 4096, "retained = the MTP input"
    assert not bridge._retained[bridge._lanes.slot_to_lane[0]].sampled
    assert bridge.spec_stats.sampled_verify_steps == 1
    assert gen.verify_sampling[3][lane_h] == 5


def test_bridge_refuses_a_sampled_verify_without_sampling_params():
    """A 'sampled_ids' verify routed to host sampling carries no sampling_params; the plugin holds such a step back
    (TTModelRunner._sampled_verify_needs_host_sampling). If one arrives anyway the bridge raises rather than commit
    argmax tokens on sampled rows."""
    from models.demos.motif3.tests import test_generator_vllm_host as T

    bridge, gen = T._bridge(num_slots=4, gen_cls=_sampled_gen_cls(), spec_tokens=1, spec_sampled=True)
    kv = bridge.allocate_kv_cache((640, 1, 32, 576), torch.bfloat16, 3)
    d = T.PluginDriver(bridge, kv, num_slots=4, block_size=32, num_blocks=640, width=kv.page_table_width, vocab=4096)
    d.add("g", [101, 102, 103, 104, 105])
    d.prefill(["g"])
    kwargs = d._decode_common(["g"])
    tokens = torch.zeros(4, 2, dtype=torch.int32)
    positions = torch.full((4, 2), -1, dtype=torch.int32)
    n = len(d.seqs["g"]) - 1
    tokens[0] = torch.tensor([d.seqs["g"][-1], 77])
    positions[0] = torch.tensor([n, n + 1])
    call = dict(tokens=tokens, start_pos=positions, num_valid_drafts=torch.tensor([1, 0, 0, 0], dtype=torch.int32),
                accepted_counts=torch.ones(4, dtype=torch.int32), spec_mode="sampled_ids", **kwargs)  # fmt: skip
    call.pop("sampling_params", None)
    with pytest.raises(RuntimeError, match="without sampling_params"):
        bridge.decode_forward(**call)
    assert bridge.spec_stats.sampled_verify_without_params == 1
