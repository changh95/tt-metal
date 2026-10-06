# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP self-speculative decoding at generator level: the T32-spec trace, packed verify, the overflow pass and KV-R
(features design §3.8, D10, §3.4-§3.5; gates G13b, G-S4, G-S5, G-S6, CP9 of §4 / §5.3; work package 5).

Host tests (``-k "cpu or host"``; no device, run through ``scripts/hostrun.sh``): the generator's speculative
orchestration on an emulated device whose KV caches are per-DP-row copies (main and MTP), written through the KV-write
calls of each step (``kv_write.kv_write_calls``: call A then call B; KV-R = every row) and read, like FlashMLA, through
each lane's page table from its OWN row's copy. The emulated main model is a deterministic function of the token
sequence it reads; the emulated MTP layer drafts the main model's next token on ~70 % of the sequences and only when
its own cache history (entry ``q`` = the token at ``q + 1``, design G8) is consistent. A wrong packing (a partner on
another DP row without KV-R, a partner reading the wrong page-table row, a missing anchor write) therefore shows up as
a wrong token. Covered: the step plan (packing, overflow, refusals, result assembly), greedy spec == greedy plain
token for token (every mode, cross-row partners, overflow after a batch change), the negative control, the decode-path
lifecycle (staging, capture, replay with new inputs, refusals after capture), ``MotifModel.decode_spec`` threading,
and the real vLLM bridge (``MotifForCausalLM`` + the plugin emulator of ``test_generator_vllm_host.py``) over the
real generator: ordinary -> propose -> verify -> accept walk, every token checked.

Device tests (53 layers + the MTP layer from the TT cache, every weight from the cache; one boot shared by the
module; production settings: chunked prefill + prefix caching + spec_tokens = 1 -> the spec trace in ``all_split``;
also captured in the same session: the plain ``all`` (KV-R, non-speculating production) and ``row`` (draft 1) traces
for comparisons)::

    scripts/devrun.sh -t 3600 -n wp5_spec -- env OMP_WAIT_POLICY=PASSIVE python -m pytest \
        models/demos/motif3/tests/test_spec_decode_device.py -k "not cpu and not host" -s -p no:cacheprovider \
        --timeout=0

* **G-S6** trace safety in serving order: prefill (sp0 cold, sp1 8K continuation) after the capture, verify steps
  with packed and cross-row partners, an overflow pass, ordinary steps; replay == eager bitwise at each phase; an
  earlier request's step is bitwise unchanged at the end (nothing corrupted); program cache constant.
* **R5** lane relocation at full depth: the same token at the same position on lanes of different DP rows, a packed
  partner (same row / other row) and an overflow row vs the ordinary step at ``n + 1``: logits, ``a``, ``m`` bitwise.
* **G-S5** lossless: greedy spec vs greedy plain (``all``) token for token, batch 1 (8 prompts: the 4 validation
  prompts x thinking off / on), 8, 16 (all rows speculating, partners forced onto other DP rows), 24 (capped drafting),
  32 (no idle lane: drafts declined) and a forced-overflow run (drafts injected after a batch change); plus plain
  ``row`` (draft 1) vs plain ``all``.
* **Throughput**: tokens/s of greedy spec vs plain at batch 1 / 8 / 16 / 32 (the bridge's drafting budget), with the
  per-step costs and the acceptance.
* **G-S4** acceptance on device: the 6 C2 prompts teacher-forced through the T32-spec path (ordinary steps), MTP on
  TT hiddens: on-policy acceptance vs the CPU bf16 reference MTP on the same rows (WP3's goldens; the CPU estimate
  over all rows is 0.835, spec_mtp.md §1.2).
* **G13b** KV-R at full depth: traced step costs of ``row`` / ``all`` / ``all_split`` (plain) and the spec trace at
  1K and 8K context, 32 lanes; the KV-R invariant (every chip's copy of the decode-written slots bitwise equal) after
  the run; trace == eager; the trace region use.
* **CP9-spec**: randomized cold / hit / chunked / resumed prefills interleaved with ordinary, verify and overflow spec
  steps: program cache constant, replay == eager bitwise at the end.

``OMP_WAIT_POLICY=PASSIVE`` is the serving setting (features design §1.1, §1.4): without it torch's spinning OpenMP
workers starve tt-metal's host threads after the step's host planning, and the next input copies stall (measured on
this host: 2.8 + 1.9 ms per verify step at 4 layers; ``generator.py`` logs a warning on a speculating launch).

Measured on this Galaxy (2026-10-02, 53 layers + MTP, serving pool 4129 x 64 bfp8, W = 512, fabric TORUS_Y committed;
``logs/dev/20261002_234045_wp5_dev_final3.log``: 7 passed in 13:37; two earlier identical runs, ``*wp5_dev53_final*``,
gave the same results within the ranges below):

* G-S5: token-exact in all 15 configurations (b1 x 8, b8, b16 with 1032 cross-row partners, b16, b24, b32, b32 with
  68 overflow passes, 16 + 12 admitted); plain ``row`` == plain ``all``. R5: every relocation / partner / overflow row
  bitwise. G-S6: replay == eager at every phase, the first step bitwise unchanged at the end, 995 programs throughout.
* Greedy tokens/s, spec vs plain ``all`` (KV-R) / ``row`` (draft 1), thinking-on prompts, the bridge's budget:
  b1 20.8 vs 11.3 / 11.6 (x1.83-1.84), b8 151.6-151.8 vs 84.8-85.0 (x1.78-1.79), b16 300.8-301.0 vs 159.0-159.6
  (x1.89), b32 327.9-328.4 vs 279.1-280.6 (x1.17:
  no idle lane until requests finish). Acceptance 0.85-0.90 on generated text. Step ms: plain ``all`` 87.9-88.6,
  spec verify 88.4-89.3, spec ordinary (with the logits read) 89.9-90.1.
* G-S4: on-policy acceptance 0.8539-0.8550 (~1566 rows) vs the CPU reference 0.8534 on the same rows; all rows
  0.655-0.656 (not §1.2's quantity).
* G13b: device ms per step at 1K / 8K: ``row`` 83.85 / 85.26, ``all`` 85.87 / 87.23, ``all_split`` 85.77 / 87.20 (KV-R
  write cost 1.87-1.94 ms over the runs, gate 2.0), spec 87.09 / 88.48; the KV-R invariant holds bitwise on all 32
  chips (L0, L1, L2, L52, MTP); 4 traces use 181.4 of 256 MiB.
* CP9-spec soak (``MOTIF3_CP9_ROUNDS=150``, ``logs/dev/20261002_231640_wp5_cp9_soak2.log``): 531 spec steps (394
  verify, 2695 drafts, 29 overflow passes) between cold / hit / chunked / resumed / burst prefills, 995 programs
  throughout, the final overflow verify replay == eager bitwise.

**T64** (``spec_verify`` "auto" / "wide"; docs/p5_t64/P5_T64_DESIGN.md §2.2-§2.4, §4, §6.2; work package I2).

Host tests: the generator's T64 path on the emulated device (``EmuModel.decode_wide``: the step's 64 physical rows
through its own update calls -- the split-order KV-R calls A / B, or ``row_split``'s 16-user calls -- each row reading
its own DP row's copy, the MTP history checked): the decode paths of each mode and the ``auto`` routing
(``test_cpu_wide_paths_and_routing``); greedy losslessness in ``auto`` / ``wide`` x ``all_split`` / ``row_split`` at 20
and 32 live lanes, traced and eager, never an overflow pass (``test_cpu_emulated_wide_lossless``); every precondition
refusal (F3N R1 / R-E5 / R-E7, settings vs config, argmax-only T64 in ``auto``; the R-E7 message names the list that
lacks the T64 row count; ``test_cpu_wide_refusals``); the launch warning of a ``packed`` / non-speculating launch whose
ring gather is not "safe" (design X3, ``test_cpu_ring_gather_launch_warning``); F3N R4
(``test_cpu_r4_replay_order``); ``drafts_all_lanes`` (R-E3 / R-E9); ``MotifModel.decode_wide``'s threading; the real
bridge + plugin emulator in ``packed`` and ``auto`` (``test_host_bridge_spec_roundtrip_t64``: in ``auto`` every live
lane drafts from ``c*`` live lanes on and one T64 step verifies them; every committed token checked).

Device (``t64_session``: production settings + packed prefill + ``spec_verify="auto"``; next to the two serving traces
the plain ``all`` trace and, with ``MOTIF3_T64_ROW_SPLIT`` (default), the ``row_split`` T32 / T64 traces)::

    scripts/devrun.sh -t 3600 -n t64 -- env OMP_WAIT_POLICY=PASSIVE python -m pytest \
        models/demos/motif3/tests/test_spec_decode_device.py -k "t64 and not cpu and not host" -s -p no:cacheprovider \
        --timeout=0

* **G16** (``test_t64_g16_step_cost``): T64 (32 lanes, a draft each) vs T32-spec (32 lanes, no drafts), traced, 1K / 8K
  / 32K, ``all_split`` and ``row_split``; the trace region per trace; eager prefills sp0 8192, sp1 8192 and a pk0 T =
  8192 pass right after a T64 replay; bar T64 / T32 <= 1.20 at 1K-8K (kill > 1.30), T64 trace <= 8 MiB per bank.
* **G-S5w** (``test_t64_gs5w_lossless``): 32 prompts x 256 tokens, two sets: (i) ``auto`` + the bridge rule == the
  non-speculative decode == the plain trace; (ii) every step on T64, idle draft rows / every lane drafting; (iii) the
  draft-row relocation probe (hidden state + logits bitwise); (iv) the rollback probe (wrong drafts on half the lanes,
  ``all_split`` and ``row_split``; caches compared by ``test_t64_release_readback``); (v) accept / reject sequences
  at c = 32 (T64) == at c = 8 (T32 packed).
* **G-S6w-lite** (``test_t64_gs6w_lite_two_traces``): ``MOTIF3_GS6W_PREFILLS`` (100) solo prefills between T32 and T64
  replays; the probe prompt bitwise; trace == eager on both paths; T64 == T32 (packed + overflow) for one verify step.
* ``test_t64_release_readback`` (last; releases the traces): the rollback probe's cache rows (main + MTP, every chip)
  and, with ``MOTIF3_G16_OPTION_A`` (default), the T64 step with option A (one B = 16 FlashMLA call on global layers).
* ``test_t64_wide_launch`` (its own session, ``spec_verify="wide"``, run with ``-k t64_wide``): the one-trace fallback
  with the anchors' logits and the device sampler on the T64 trace.

Measured on this Galaxy (2026-10-03, 53 layers + MTP; ``logs/dev/20261003_224935_i2_t64_l53.log``, quiet-host rerun
``logs/dev/20261003_230237_i2_t64_l53b.log``):

* G16, T64 / T32 device step (replay + sync, median of 20): ``all_split`` 1.119 / 1.134 / 1.173 (87.4 -> 97.8, 88.9 ->
  100.7, 93.1 -> 109.3 ms) and ``row_split`` 1.115 / 1.130 / 1.171 at 1K / 8K / 32K; end to end (host included)
  1.129 / 1.144 / 1.182: the T64 step's host work is ~1.0-1.6 ms more (plan 0.7-0.8 vs 0.35-0.4 ms, input writes
  1.9-3.0 vs 1.3-1.9 ms). The T64 trace 6.26 MiB per bank (T32-spec 5.96, plain 5.82; five traces 241 of 256 MiB).
  The prefills after a T64 replay sane; the program cache constant (1473). Option A (one B = 16 FlashMLA call on the
  global layers) is 0.49 / 0.81 / 0.64 ms per step cheaper than A'' (not bitwise; A'' ships).
* G-S5w: token-exact on all 32 lanes in every case; (i) thinking on: 116 of 150 verify steps on T64, acceptance 0.851,
  415 tok/s = x1.68 the non-speculative T32 steps at c = 32; thinking off: 52 T64 of 148 (requests end, c drops below
  c*), acceptance 0.862, x1.74; (ii) T64-only decode, idle draft rows and every lane drafting: exact; (iii) the draft
  row's final-norm hidden state (8 chips) and logits rows (32 chips) bitwise equal to the next T32 and T64 anchors;
  (iv) 469 wrong drafts rejected per KV mode, tokens exact, the main and MTP cache rows of 32 request pairs bitwise
  equal on every chip (``all_split``: 32 chips, ``row_split``: the DP row); (v) 8 of 8 accept / reject sequences
  identical.
* G-S5w + G-S6w-lite + the release readback under ``MOTIF3_ROUTER_LOGITS=exact_fp32`` (review edit R-E7; 2026-10-04 on
  an overlay whose only change is ``model_config.ROUTER_EXACT_FP32_DECODE_ROWS = (32, 64)``,
  ``logs/dev/20261004_011507_t64fix_exact_overlay.log``): (i)-(v) token-exact / bitwise as above (x1.73 thinking on,
  x1.88 thinking off), the rollback cache rows bitwise in both KV modes, G-S6w-lite passed (program cache constant at
  1474). The exact router makes the T64 step 100.1 / 103.0 / 111.6 ms at 1K / 8K / 32K (97.8 / 100.7 / 109.3 with the
  composite router).
* G-S6w-lite: 100 solo prefills (57 cold, 23 hits, 20 chunked) between 128 T32 and 80 T64 steps: the probe prompt
  bitwise identical, trace == eager on both paths, one verify step on T64 == on T32 (packed + overflow), program
  cache constant.
* ``wide``: ordinary T64 steps (host logits) and drafting T64 steps token-exact; 24 device-sampled T64 steps:
  temperature-0 lanes == greedy, seeded lanes reproducible bitwise.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import random
import statistics
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from models.demos.motif3.tt import generator_api as api
from models.demos.motif3.tt import kv_write as KW
from models.demos.motif3.tt import prefill_plan as PP
from models.demos.motif3.tt.model_config import DEFAULT_HF_META_DIR, PROJECT_ROOT, MotifTTConfig

HF_META = str(DEFAULT_HF_META_DIR)
NO_TOKEN = -7  # emulated device: argmax / MTP ids of lanes that ran nothing


def log(msg: str) -> None:
    print(f"[spec-decode {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def host_cfg(**kw) -> MotifTTConfig:
    return MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8), **kw)


def lane_of_slot(slot: int, num_slots: int = api.NUM_LANES) -> int:
    """The bridge's ``LaneMap``: state slots dealt round-robin over the 4 DP rows."""
    groups = api.NUM_DP_GROUPS
    return (slot % groups) * api.LANES_PER_GROUP + slot // groups


# ======================================================================================================================
# the greedy decode driver (host emulation and device): plugin + bridge semantics around one generator
# ======================================================================================================================
@dataclasses.dataclass
class Req:
    """One greedy request: its prompt (prefilled once), KV blocks, lane, and the generated tokens of the last run."""

    name: str
    prompt: List[int]
    blocks: List[int]
    lane: int
    first: int  # the prefill's argmax: the token at position len(prompt)
    max_new: int
    out: List[int] = dataclasses.field(default_factory=list)
    margins: List[float] = dataclasses.field(default_factory=list)
    done: bool = False
    draft: Optional[int] = None
    nxt: Optional[int] = None  # the MTP draft the last step left for this request (m0 or m1)
    verdicts: List[bool] = dataclasses.field(default_factory=list)  # accept (True) / reject per verified draft

    @property
    def S(self) -> int:
        return len(self.prompt)

    def anchor_pos(self) -> int:
        return self.S + len(self.out) - 1


@dataclasses.dataclass
class RunStats:
    steps: int = 0
    verify_steps: int = 0
    offered: int = 0
    accepted: int = 0
    tokens: int = 0
    seconds: float = 0.0
    step_ms: Dict[str, List[float]] = dataclasses.field(default_factory=dict)
    gen_stats: Dict[str, int] = dataclasses.field(default_factory=dict)

    @property
    def acceptance(self) -> float:
        return self.accepted / max(self.offered, 1)

    @property
    def tok_s(self) -> float:
        return self.tokens / max(self.seconds, 1e-9)

    def median_ms(self, kind: str) -> float:
        v = self.step_ms.get(kind, [])
        return statistics.median(v) if v else float("nan")


class GreedyDriver:
    """Greedy decode of a set of :class:`Req` through one generator, the way a speculating launch runs it:

    * ``run_plain(path)``: every step ``decode_forward`` (one token per request per step, host argmax).
    * ``run_spec(policy)``: every step ``decode_forward_spec`` (the plugin: ordinary step without drafts, verify step
      with them; ``accept_greedy_drafts``' rule: a draft equal to ``a0`` commits ``(d, a1)``, else ``a0``), then the
      bridge's ``propose_draft_tokens`` rule: the next draft is ``m[count - 1]``, only for requests whose committed
      tokens are the step's argmax (always, greedy), within the drafting budget (``policy="budget"``: one idle lane
      per draft, counted over the step's live lanes; ``"all"``: every request drafts, overflow passes included;
      ``"none"``: never; ``"bridge"``: the T64 bridge rule, every request drafts when ``generator.drafts_all_lanes(
      live lanes, acceptance=the run's prior-smoothed acceptance)`` says so, else the budget). A request's first step
      after its prefill never carries a draft (the plugin does not propose after a prefill). ``admit(step) -> [Req]``
      adds prefilled requests between steps (a batch change after the drafts were proposed: overflow). ``path`` forces
      a decode path (``("wide", mode)``: every step on the T64 trace), ``want_logits=False`` skips the logits read of
      ordinary steps (the argmax-only T64 trace of ``auto``), ``draft_fn(r) -> int`` replaces the proposed draft (the
      rollback probe's wrong drafts). Each request records its verdicts (``r.verdicts``).

    Each request's ``out`` starts with ``first`` and stops at an EOS token or ``max_new`` tokens."""

    def __init__(self, gen, pool, *, width: int, block_size: int, eos: Sequence[int] = (), trace: bool = True):
        self.gen, self.pool, self.W, self.bs = gen, pool, int(width), int(block_size)
        self.eos = set(int(e) for e in eos)
        self.trace = trace
        self._rot = 0  # the bridge's rotating propose start
        self.verdicts = [0, 0]  # accepted, verified since the last reset ("bridge" policy)

    def page_row(self, r: Req, upto: int) -> torch.Tensor:
        pt = torch.zeros(self.W, dtype=torch.int32)
        k = upto // self.bs + 1
        assert k <= len(r.blocks), f"{r.name}: position {upto} beyond its {len(r.blocks)} blocks"
        pt[:k] = torch.tensor(r.blocks[:k], dtype=torch.int32)
        return pt

    def _reset(self, reqs: Sequence[Req]) -> None:
        for r in reqs:
            r.out, r.margins, r.draft, r.nxt, r.verdicts = [r.first], [], None, None, []
            r.done = r.first in self.eos or r.max_new <= 1
        self.verdicts = [0, 0]  # accepted, verified: the "bridge" policy's acceptance estimate

    def _commit(self, r: Req, toks: Sequence[int]) -> None:
        for t in toks:
            r.out.append(int(t))
            if int(t) in self.eos or len(r.out) >= r.max_new:
                r.done = True
                return

    def run_plain(self, reqs: Sequence[Req], *, path=None) -> RunStats:
        self._reset(reqs)
        st = RunStats()
        t_all = time.perf_counter()
        while True:
            live = [r for r in reqs if not r.done]
            if not live:
                break
            tok = torch.zeros(api.NUM_LANES, dtype=torch.int32)
            pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
            pt = torch.zeros(api.NUM_LANES, self.W, dtype=torch.int32)
            for r in live:
                p = r.anchor_pos()
                tok[r.lane], pos[r.lane], pt[r.lane] = r.out[-1], p, self.page_row(r, p)
            t1 = time.perf_counter()
            logits = self.gen.decode_forward(
                api.DecodeBatch(tokens=tok, positions=pos, page_table=pt), kv_cache=self.pool, enable_trace=self.trace,
                path=path,
            )  # fmt: skip
            st.step_ms.setdefault("plain", []).append((time.perf_counter() - t1) * 1e3)
            st.steps += 1
            for r in live:
                row = logits[r.lane].float()
                top = torch.topk(row, 2).values
                r.margins.append(float(top[0] - top[1]))
                self._commit(r, [int(row.argmax())])
        st.seconds = time.perf_counter() - t_all
        st.tokens = sum(len(r.out) for r in reqs)
        return st

    def batch(self, live: Sequence[Req]) -> api.SpecDecodeBatch:
        """The step's ``SpecDecodeBatch`` (owner-lane order): each live request's last token at its position, its
        pending draft for ``n + 1`` (page-table row fitted to ``n + 1`` then)."""
        tok = torch.zeros(api.NUM_LANES, dtype=torch.int32)
        pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
        dr = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
        pt = torch.zeros(api.NUM_LANES, self.W, dtype=torch.int32)
        for r in live:
            p = r.anchor_pos()
            tok[r.lane], pos[r.lane] = r.out[-1], p
            if r.draft is not None:
                dr[r.lane] = r.draft
            pt[r.lane] = self.page_row(r, p + (1 if r.draft is not None else 0))
        return api.SpecDecodeBatch(tokens=tok, positions=pos, draft_tokens=dr, page_table=pt)

    def spec_step(
        self,
        live: Sequence[Req],
        *,
        policy: str = "budget",
        st: Optional[RunStats] = None,
        check_argmax: bool = True,
        trace: Optional[bool] = None,
        path=None,
        want_logits: Optional[bool] = None,
        draft_fn: Optional[Callable[[Req], int]] = None,
    ) -> api.SpecDecodeResult:
        """One speculative decode step of ``live`` (module docstring of the class): decode_forward_spec, the accept
        walk, then the proposals for the next step."""
        assert policy in ("budget", "all", "none", "bridge")
        batch = self.batch(live)
        verify = batch.is_verify
        want = (not verify) if want_logits is None else bool(want_logits) and not verify
        t1 = time.perf_counter()
        use = self.trace if trace is None else trace
        res = self.gen.decode_forward_spec(batch, kv_cache=self.pool, enable_trace=use, want_logits=want, path=path)
        dt = (time.perf_counter() - t1) * 1e3
        if st is not None:
            kind = getattr(self.gen, "last_verify_kind", None) or "spec"
            st.step_ms.setdefault(("verify" if verify else "ordinary") + ("_t64" if kind == "wide" else ""),
                                  []).append(dt)  # fmt: skip
            st.steps += 1
            st.verify_steps += int(verify)
        am, mm = res.argmax.tolist(), res.mtp_argmax.tolist()
        for r in live:
            a0, a1 = am[r.lane]
            m0, m1 = mm[r.lane]
            if check_argmax and want:
                row = res.logits[r.lane].float()
                assert a0 == int(row.argmax()), f"{r.name}: device a0 {a0} != host argmax {int(row.argmax())}"
            if r.draft is not None:
                ok = r.draft == a0
                r.verdicts.append(ok)
                self.verdicts[0] += int(ok)
                self.verdicts[1] += 1
                if st is not None:
                    st.offered += 1
                if ok:
                    if st is not None:
                        st.accepted += 1
                    commit, r.nxt = [a0, a1], m1
                else:
                    commit, r.nxt = [a0], m0
            else:
                commit, r.nxt = [a0], m0
            r.draft = None
            self._commit(r, commit)
        if policy == "bridge":  # the T64 bridge rule (R-E3, R-E9): every live lane drafts from c* live lanes on
            lanes = [r.lane for r in live if not r.done]
            acc = api.smoothed_acceptance(self.verdicts[0], self.verdicts[1])
            policy = "all" if lanes and self.gen.drafts_all_lanes(lanes, acceptance=acc) else "budget"
        # propose (the bridge's _draft_budget): one idle lane per draft, counted over this step's live lanes (any
        # row with KV-R, the owner's row without it); rotating start
        if policy != "none" and live:
            G8 = api.LANES_PER_GROUP
            cross = KW.allows_cross_row_partners(self.gen.spec_kv_mode)
            left = {None: api.NUM_LANES - len(live)} if cross else {g: G8 for g in range(api.NUM_DP_GROUPS)}
            if not cross:
                for r in live:
                    left[r.lane // G8] -= 1
            k = self._rot % len(live)
            self._rot += 1
            for r in list(live[k:]) + list(live[:k]):
                if r.done:
                    continue
                if policy == "budget":
                    key = None if cross else r.lane // G8
                    if left[key] <= 0:
                        continue
                    left[key] -= 1
                r.draft = r.nxt if draft_fn is None else int(draft_fn(r))
        return res

    def start(self, reqs: Sequence[Req]) -> None:
        """Put requests in their just-prefilled state (``out = [first]``, no draft)."""
        self._reset(reqs)

    def run_spec(
        self,
        reqs: Sequence[Req],
        *,
        policy: str = "budget",
        admit: Optional[Callable[[int], Sequence[Req]]] = None,
        check_argmax: bool = True,
        path=None,
        want_logits: Optional[bool] = None,
        draft_fn: Optional[Callable[[Req], int]] = None,
    ) -> RunStats:
        reqs = list(reqs)
        self._reset(reqs)
        st = RunStats()
        g0 = dict(self.gen.stats)
        t_all = time.perf_counter()
        while True:
            if admit is not None:
                new = list(admit(st.steps))
                self._reset(new)
                reqs.extend(new)
            live = [r for r in reqs if not r.done]
            if not live:
                break
            self.spec_step(live, policy=policy, st=st, check_argmax=check_argmax, path=path, want_logits=want_logits,
                           draft_fn=draft_fn)  # fmt: skip
        st.seconds = time.perf_counter() - t_all
        st.tokens = sum(len(r.out) for r in reqs)
        st.gen_stats = {k: int(self.gen.stats[k]) - int(g0.get(k, 0)) for k in self.gen.stats}
        return st


def first_divergence(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# ======================================================================================================================
# host emulation of the device (the decode paths' semantics)
# ======================================================================================================================
def lm_next(seq: Sequence[int], vocab: int) -> int:
    """The emulated main model: a deterministic, position-sensitive function of the whole token sequence (the same
    function as ``test_generator_vllm_host.next_token``); never below 100."""
    s = torch.as_tensor(list(seq), dtype=torch.int64)
    w = torch.arange(1, s.numel() + 1, dtype=torch.int64)
    return 100 + int(((s + 7) * w * w).sum().item() % (vocab - 100))


def lm_wrong(token: int, vocab: int) -> int:
    return 100 + ((int(token) - 100 + 1) % (vocab - 100))


def mtp_next(seq: Sequence[int], vocab: int) -> int:
    """The emulated MTP draft for ``seq`` = the tokens through ``n + 1`` (the anchor's sequence + the main argmax):
    the main model's own next token on ~70 % of the sequences (``test_generator_vllm_host.mtp_token``)."""
    truth = lm_next(seq, vocab)
    n = len(seq)
    return truth if (truth * 7 + n * 3) % 10 < 7 else lm_wrong(truth, vocab)


class _T:
    """A fake ttnn tensor (host or device): ``value``; freed through ``is_allocated`` / ``ttnn.deallocate``."""

    def __init__(self, value=None):
        self.value = value
        self.alive = True

    def is_allocated(self):
        return self.alive


class EmuDevice:
    """Per-DP-row copies of the main KV and the MTP KV (``[4, num_blocks, bs]`` token ids, -1 = never written)."""

    def __init__(self, cfg: MotifTTConfig, num_blocks: int, *, next_fn=lm_next, mtp_fn=mtp_next):
        self.cfg, self.V, self.bs, self.G = cfg, int(cfg.vocab_size), int(cfg.kv_block_size), int(cfg.dp)
        self.kv = torch.full((self.G, num_blocks, self.bs), -1, dtype=torch.int64)
        self.mtp_kv = torch.full((self.G, num_blocks, self.bs), -1, dtype=torch.int64)
        self.next_fn, self.mtp_fn = next_fn, mtp_fn
        self.stale_mtp = 0  # MTP history inconsistencies below the last entry (design G8: must stay 0)
        self.last_mismatch = 0  # last-entry mismatches (rejected drafts' rows: expected)
        self.runs: List[Tuple[str, KW.KVWriteStep]] = []

    def prefill(self, tokens: Sequence[int], page_table: torch.Tensor, first: int) -> None:
        """An emulated prefill: the prompt's KV on every row; the MTP KV of ``p`` = the token at ``p + 1`` (the row's
        last position takes ``first``, the stand-in argmax)."""
        t = torch.tensor(list(tokens), dtype=torch.int64)
        pos = torch.arange(t.numel())
        blk = page_table.long()[pos // self.bs]
        self.kv[:, blk, pos % self.bs] = t
        self.mtp_kv[:, blk, pos % self.bs] = torch.cat([t[1:], torch.tensor([int(first)])])

    def _write(
        self, cache, step: KW.KVWriteStep, mode: str, lanes_per_call, values, rows_per_dp: int, gather: str
    ) -> None:
        calls = KW.kv_write_calls(step, mode, lanes_per_call=lanes_per_call, lanes_per_row=rows_per_dp, gather=gather)
        for call in calls:
            for lane, p in zip(call.lanes, call.positions.tolist()):
                if p < 0:
                    continue
                b = int(step.page_table[lane, p // self.bs])
                for r in call.rows:
                    cache[r, b, p % self.bs] = int(values[lane])

    def _read(self, cache, lane: int, step: KW.KVWriteStep, lo: int, hi: int, rows_per_dp: int) -> torch.Tensor:
        idx = torch.arange(lo, hi)
        return cache[lane // rows_per_dp, step.page_table[lane].long()[idx // self.bs], idx % self.bs]

    def run(
        self,
        tokens: torch.Tensor,
        step: KW.KVWriteStep,
        mode: str,
        lanes_per_call,
        *,
        spec: bool,
        rows_per_dp: int = api.LANES_PER_GROUP,
        gather: str = "natural",
    ):
        """One decode step of every row (``step.lanes``: the 32 lanes, or the T64 step's 64 physical rows with
        ``rows_per_dp=16``): the writes of all 54 'layers' (here: one main cache) through the step's update calls
        (``kv_write_calls``, the T64 split-order gather included), then the reads, each row from its OWN DP row's copy;
        with ``spec`` the MTP layer's writes (value = the main argmax ``a``) and reads. Returns ``(a, m | None)`` per
        physical row."""
        self.runs.append(("spec" if spec else "plain", step))
        toks = [int(x) for x in tokens.tolist()]
        self._write(self.kv, step, mode, lanes_per_call, toks, rows_per_dp, gather)
        n = step.lanes
        a, seqs = [NO_TOKEN] * n, {}
        act = [l for l in range(n) if int(step.positions[l]) >= 0]
        for lane in act:
            p = int(step.positions[lane])
            seqs[lane] = self._read(self.kv, lane, step, 0, p + 1, rows_per_dp).tolist()
            a[lane] = self.next_fn(seqs[lane], self.V)
        if not spec:
            return a, None
        self._write(self.mtp_kv, step, mode, lanes_per_call, a, rows_per_dp, gather)
        m = [NO_TOKEN] * n
        for lane in act:
            p, seq = int(step.positions[lane]), seqs[lane]
            lo = max(0, p - 128)
            hist = self._read(self.mtp_kv, lane, step, lo, p, rows_per_dp).tolist()
            want = seq[lo + 1 : p + 1]
            # entries below p - 1 are committed history for every lane (G8); entry p - 1 holds the token at p only
            # for a committed token at p (a draft's last entry is the owner's a0: equal iff the draft is accepted)
            if hist[:-1] != want[:-1]:
                self.stale_mtp += 1
            if hist[-1:] != want[-1:]:
                self.last_mismatch += 1
            ok = hist == want
            m[lane] = (
                self.mtp_fn(seq + [a[lane]], self.V) if ok else lm_wrong(self.mtp_fn(seq + [a[lane]], self.V), self.V)
            )
        return a, m


class EmuReads:
    """Device -> host reads on the emulated device: a non-blocking read lands only at the next blocking read (a
    full-mesh finish), so reading a staging buffer too early returns stale data and fails the tests."""

    def __init__(self):
        self.pending: List[Tuple[Any, Any]] = []
        self.blocking = 0

    def copy(self, dev, host, blocking=True, cq_id=None):
        self.pending.append((dev, torch.as_tensor(dev.value).clone()))
        self.pending[-1] = (host, self.pending[-1][1])
        if blocking:
            self.flush()

    def flush(self):
        for host, val in self.pending:
            host.value = val
        self.pending = []
        self.blocking += 1


class EmuReader:
    """``lm_head.HostShardReader`` on the emulated device: persistent host staging ``host`` and its view."""

    def __init__(self, mesh, t):
        self.host = _T(torch.full((api.NUM_LANES,), NO_TOKEN))

    @staticmethod
    def spec_key(t):
        return ("emu",)

    @property
    def views(self):
        return [torch.as_tensor(self.host.value)]


class EmuHead:
    vocab_split = "mesh"

    def __init__(self, vocab: int, reads: Optional[EmuReads] = None):
        self.V = int(vocab)
        self.reads = reads

    def logits_to_host(self, rm) -> torch.Tensor:
        if self.reads is not None:  # a blocking read: every outstanding read lands
            self.reads.flush()
        a = rm.value
        out = torch.full((api.NUM_LANES, self.V), -1.0, dtype=torch.bfloat16)
        for lane, t in enumerate(a):
            if t >= 0:
                out[lane, t] = 1.0
        return out

    def tokens_to_host(self, t) -> torch.Tensor:
        if self.reads is not None:
            self.reads.flush()
        return torch.as_tensor(t.value, dtype=torch.int64).clone()


class EmuKVW:
    """``DecodeKVWrite`` on the emulated device: records the step (the emulated model applies its calls). ``rows=64``
    is the T64 writer (16 rows per DP row; ``gather`` "split" in the KV-R modes, as ``kv_write.DecodeKVWrite``)."""

    def __init__(self, mesh, cfg, *, ccl, page_table_width, mode, rows=None, gather=None):
        self.cfg, self.mode, self.width = cfg, mode, int(page_table_width)
        self.lanes = int(cfg.max_batch) if rows is None else int(rows)
        assert self.lanes in (32, 64), rows
        self.wide = self.lanes == 64
        self.lanes_per_row = int(cfg.lanes_per_row) * (2 if self.wide else 1)
        if gather is None:
            gather = "split" if (self.wide and KW.is_replicated(mode) and KW.is_split(mode)) else "natural"
        assert gather in KW.GATHER_ORDERS and (self.wide or gather == "natural"), gather
        self.gather = gather if KW.is_replicated(mode) else "natural"
        self.lanes_per_call = KW.lanes_per_call_for(
            mode, cfg.dtypes.kv_cache_name, lanes=self.lanes, lanes_per_row=self.lanes_per_row, gather=self.gather
        )
        self.cur_pos, self.page_table = _T("kvw.cur"), _T("kvw.pt")
        self.step = KW.KVWriteStep.inactive(self.width, self.lanes)
        self.writes = 0

    def write_step(self, step, *, validate=True, force=False):
        if validate:
            KW.check_kv_write_step(
                step, self.mode, block_size=self.cfg.kv_block_size, max_seq_len=self.cfg.max_model_len,
                lanes_per_call=self.lanes_per_call if KW.is_replicated(self.mode) else None,
                lanes_per_row=self.lanes_per_row, gather=self.gather,
            )  # fmt: skip
        assert step.width == self.width and step.lanes == self.lanes, (step.lanes, step.width, self.lanes, self.width)
        self.step, self.writes = step, self.writes + 1

    def deallocate(self):
        self.cur_pos = self.page_table = None


class EmuModel:
    """The part of ``MotifModel`` the generator's decode paths use, on an :class:`EmuDevice`."""

    def __init__(self, cfg: MotifTTConfig, dev: EmuDevice, *, mtp: bool = True):
        self.cfg, self.dev = cfg, dev
        self.layer_ids = (0,)
        self.layers = [None]
        self.mtp = SimpleNamespace(name="mtp") if mtp else None
        self.ccl = None
        self.embed = SimpleNamespace(decode_tokens_host=lambda t, rows_per_dp=None: _T(t.clone()))
        self.reads = EmuReads()
        self.head = EmuHead(cfg.vocab_size, self.reads)
        self.decode_calls = {"plain": 0, "spec": 0, "wide": 0}

    @property
    def num_layers(self):
        return 1

    def allocate_kv_caches(self, num_blocks, block_size, dtype=None, *, mtp=None):
        from models.demos.motif3.tt.model import MotifKVPool

        return MotifKVPool([_T("kv0")], (0,), num_blocks, block_size, dtype, mtp=_T("mtp") if self.mtp else None)

    def _check_rot(self, rot_idxs, positions, rows_per_dp=None):
        from models.demos.motif3.tt.rope import positions_to_rot_idxs

        assert torch.equal(
            rot_idxs.value, positions_to_rot_idxs(positions, self.cfg, rows_per_dp=rows_per_dp)
        ), "RoPE rows != the step's positions"

    def decode(self, tokens, *, rot_idxs, cur_pos, page_table, kv_caches, kv_write=None):
        self.decode_calls["plain"] += 1
        if kv_write is not None:
            assert cur_pos is kv_write.cur_pos and page_table is kv_write.page_table
            step, mode, lpc = kv_write.step, kv_write.mode, kv_write.lanes_per_call
        else:  # draft 1: the generator's own lane-ordered cur_pos / page_table
            step, mode, lpc = KW.KVWriteStep.ordinary(cur_pos.value, page_table.value), "row", None
        self._check_rot(rot_idxs, step.positions)
        a, _ = self.dev.run(tokens.value, step, mode, lpc, spec=False)
        return _T(a)

    def decode_spec(self, tokens, *, rot_idxs, kv_write, kv_caches, stop_after=None, keep_hidden=False):
        self.decode_calls["spec"] += 1
        assert kv_caches.mtp is not None, "the spec step writes the MTP cache"
        assert not getattr(kv_write, "wide", False), "the T32-spec step got the T64 writer"
        step = kv_write.step
        self._check_rot(rot_idxs, step.positions)
        a, m = self.dev.run(tokens.value, step, kv_write.mode, kv_write.lanes_per_call, spec=True)
        return _T(a), _T(torch.tensor(a)), _T(torch.tensor(m))

    def decode_wide(self, tokens, *, rot_idxs, kv_write, kv_caches, want_rm=False, stop_after=None, keep_hidden=False):
        """``MotifModel.decode_wide`` on the emulated device: the 64 physical rows of the T64 writer's step (anchors
        at ``n``, drafts at ``n + 1`` on their owners' DP rows), the split-order outputs ``a`` / ``m`` ``[64]``
        (``[l]`` = lane ``l``'s anchor row, ``[32 + l]`` its draft row) and, with ``want_rm``, the anchors' "logits"
        (the emulated head reads ``rm`` as the anchors' argmax)."""
        self.decode_calls["wide"] += 1
        assert kv_caches.mtp is not None, "the T64 step writes the MTP cache"
        assert kv_write.wide and kv_write.lanes_per_row == 16 and self.cfg.wide_rows_per_dp == 16
        step = kv_write.step
        assert step.lanes == 64 and int(tokens.value.numel()) == 64
        self._check_rot(rot_idxs, step.positions, rows_per_dp=16)
        a, m = self.dev.run(tokens.value, step, kv_write.mode, kv_write.lanes_per_call, spec=True, rows_per_dp=16,
                            gather=kv_write.gather)  # fmt: skip
        order = KW.split_order(64, 16)
        a_s = torch.tensor([a[order[u]] for u in range(64)])
        m_s = torch.tensor([m[order[u]] for u in range(64)])
        rm = _T(a_s[:32].tolist()) if want_rm else None
        return rm, _T(a_s), _T(m_s)

    def deallocate(self):
        pass


class EmuTrace:
    """``ttnn`` trace calls on the emulated device: a replay re-runs the captured path's step on its current inputs
    and writes the results into the captured output objects (what a trace replay with new inputs does)."""

    def __init__(self, gen):
        self.gen, self.n, self.replays, self.captures = gen, 0, 0, []
        self.kinds: List[str] = []  # the path kind of every replay

    def begin(self, mesh, cq_id=0):
        self.n += 1
        return ("trace", self.n)

    def end(self, mesh, tid, cq_id=0):
        self.captures.append(tid)

    def execute(self, mesh, tid, cq_id=0, blocking=False):
        p = next(p for p in self.gen._paths.values() if p.trace_id == tid)
        new = self.gen._device_step(p, p.pool)
        if isinstance(new, tuple):
            for o, x in zip(p.out, new):
                if o is not None:  # (the argmax-only T64 path has no logits output)
                    o.value = x.value
        else:
            p.out.value = new.value
        self.replays += 1
        self.kinds.append(p.kind)

    def release(self, mesh, tid):
        pass


def emu_generator(
    monkeypatch,
    *,
    kvr: bool = True,
    spec: bool = True,
    num_blocks: int = 3000,
    max_model_len: int = 8192,
    spec_verify: str = "packed",
    cfg_kw: Optional[dict] = None,
    **gen_kw,
):
    """A real ``MotifGenerator`` over :class:`EmuModel` (fake ttnn: host tensors, copies, traces). ``spec_verify``: the
    config's verify mode ("wide" / "auto" stage the T64 path, emulated by :meth:`EmuModel.decode_wide`)."""
    from models.demos.motif3.tt import generator as G

    cfg = host_cfg(kv_replicated_decode=kvr, spec_tokens=1 if spec else 0, max_model_len=max_model_len,
                   spec_verify=spec_verify, **dict(cfg_kw or {}))  # fmt: skip
    dev = EmuDevice(cfg, num_blocks)
    model = EmuModel(cfg, dev, mtp=True)
    monkeypatch.setattr(G, "DecodeKVWrite", EmuKVW)
    monkeypatch.setattr(G, "shard_lanes", lambda rows, cfg_, mesh, **kw: _T(rows.clone()))
    monkeypatch.setattr(G.ttnn, "to_device", lambda v, mesh, memory_config=None: _T(v.value.clone()), raising=False)

    def copy(h, d):
        d.value = h.value.clone()

    def dealloc(t, *a, **k):
        t.alive = False

    monkeypatch.setattr(G.ttnn, "copy_host_to_device_tensor", copy, raising=False)
    monkeypatch.setattr(G.ttnn, "deallocate", dealloc, raising=False)
    monkeypatch.setattr(G.ttnn, "synchronize_device", lambda mesh: model.reads.flush(), raising=False)
    monkeypatch.setattr(G.ttnn, "copy_device_to_host_tensor", model.reads.copy, raising=False)
    monkeypatch.setattr(G, "HostShardReader", EmuReader)
    # B6a host_staging="fast": the host mesh tensors come from ttnn.from_torch with a cached DP-row mapper; the emulated
    # embedding takes the lane-ordered tokens (as decode_tokens_host above)
    monkeypatch.setattr(G, "dp_row_mapper", lambda cfg_, mesh: object())
    monkeypatch.setattr(G, "decode_token_rows", lambda t, cfg_, n_streams=None, rows_per_dp=None: t.clone())
    monkeypatch.setattr(G.ttnn, "from_torch", lambda v, **kw: _T(v.clone()), raising=False)
    model.embed.n_streams = cfg.n_streams
    gen = G.MotifGenerator(None, cfg, model, log=None, **gen_kw)
    tr = EmuTrace(gen)
    monkeypatch.setattr(G.ttnn, "begin_trace_capture", tr.begin, raising=False)
    monkeypatch.setattr(G.ttnn, "end_trace_capture", tr.end, raising=False)
    monkeypatch.setattr(G.ttnn, "execute_trace", tr.execute, raising=False)
    monkeypatch.setattr(G.ttnn, "release_trace", tr.release, raising=False)
    pool = gen.allocate_kv_cache(num_blocks=num_blocks, block_size=64, num_layers=1)
    gen._warmed = set(gen.prefill_shapes())  # the prefill warmup is WP4's (tested there): mark it done
    return gen, model, dev, pool, tr


def emu_requests(
    dev: EmuDevice,
    n: int,
    *,
    seed: int,
    width: int,
    max_new: int,
    start_block: int = 1,
    lanes: Optional[Sequence[int]] = None,
    prompt_len=(40, 400),
    prefix: str = "r",
) -> List[Req]:
    """``n`` requests prefilled on the emulated device (random prompts, own blocks), named ``prefix + i``."""
    rng = random.Random(seed)
    out, nxt = [], start_block
    for i in range(n):
        S = rng.randrange(*prompt_len)
        prompt = [rng.randrange(100, dev.V) for _ in range(S)]
        nb = (S + max_new + 2) // dev.bs + 1
        blocks = list(range(nxt, nxt + nb))
        nxt += nb
        first = lm_next(prompt, dev.V)
        pt = torch.zeros(width, dtype=torch.int32)
        pt[:nb] = torch.tensor(blocks, dtype=torch.int32)
        dev.prefill(prompt, pt, first)
        lane = lanes[i] if lanes is not None else lane_of_slot(i)
        out.append(Req(f"{prefix}{i}", prompt, blocks, lane, first, max_new))
    return out


def reference_greedy(r: Req, vocab: int) -> List[int]:
    seq, out = list(r.prompt), []
    t = r.first
    while True:
        out.append(t)
        if len(out) >= r.max_new:
            return out
        seq.append(t)
        t = lm_next(seq, vocab)


# ======================================================================================================================
# host tests
# ======================================================================================================================
def _batch(pos: Dict[int, int], drafts: Dict[int, int], width: int = 64, block_of=None) -> api.SpecDecodeBatch:
    tok = torch.zeros(32, dtype=torch.int32)
    p = torch.full((32,), -1, dtype=torch.int32)
    d = torch.full((32,), -1, dtype=torch.int32)
    pt = torch.zeros(32, width, dtype=torch.int32)
    for lane, n in pos.items():
        tok[lane], p[lane] = 1000 + lane, n
        top = n + (1 if lane in drafts else 0)
        base = 1 + lane * 40 if block_of is None else block_of(lane)
        pt[lane, : top // 64 + 1] = torch.arange(base, base + top // 64 + 1, dtype=torch.int32)
        if lane in drafts:
            d[lane] = drafts[lane]
    return api.SpecDecodeBatch(tokens=tok, positions=p, draft_tokens=d, page_table=pt)


@pytest.mark.parametrize("mode", ["all_split", "row_split", "all", "row"])
def test_cpu_plan_spec_step_packing(mode):
    """``plan_spec_step`` over 300 random batches per mode: partners are idle lanes used once, on the owner's DP row
    unless the mode is ``all_split`` (KV-R), same-row first, the maximum number of drafts packed; pass 1 carries
    anchors at ``n`` (call A) and drafts at ``n + 1`` on the partners with the owner's page-table row (call B); pass 2
    exists iff some draft overflowed and runs exactly those drafts on their own lanes at ``n + 1``; non-split modes
    pack nothing; every pass passes ``check_kv_write_step``; ``result`` maps each pass's lane outputs back to owner
    columns."""
    from models.demos.motif3.tt import generator as G

    cfg = host_cfg(kv_replicated_decode=mode.startswith("all"), spec_tokens=1)
    cfg.set_kv_geometry(4000, 64)
    rng = random.Random(11)
    seen = {"cross": 0, "overflow": 0, "packed": 0}
    for it in range(300):
        nact = rng.randrange(1, 33)
        lanes = rng.sample(range(32), nact)
        pos = {l: rng.randrange(0, 2000) for l in lanes}
        drafts = {l: rng.randrange(100, 5000) for l in lanes if rng.random() < 0.8}
        b = _batch(pos, drafts)
        plan = G.plan_spec_step(b, mode=mode, cfg=cfg, num_blocks=4000)
        idle = [l for l in range(32) if l not in pos]
        partners = list(plan.partner_of.values())
        assert len(set(partners)) == len(partners) and set(partners) <= set(idle)
        assert set(plan.partner_of) | set(plan.overflow) == set(drafts) and not set(plan.partner_of) & set(
            plan.overflow
        )
        if not KW.is_split(mode):
            assert not plan.partner_of and len(plan.passes) == 1 + bool(drafts)
        else:
            row = lambda l: l // 8  # noqa: E731
            for o, d in plan.partner_of.items():
                assert mode == "all_split" or row(o) == row(d), "cross-row partner without KV-R"
            # maximum packing: same-row only (row_split) is a per-row matching; all_split packs min(drafts, idle)
            if mode == "all_split":
                assert len(plan.partner_of) == min(len(drafts), len(idle))
            else:
                for r in range(4):
                    dr = [o for o in drafts if row(o) == r]
                    il = [l for l in idle if row(l) == r]
                    assert sum(row(o) == r for o in plan.partner_of) == min(len(dr), len(il))
            seen["cross"] += plan.cross_row_partners
        seen["overflow"] += len(plan.overflow)
        seen["packed"] += len(plan.partner_of)
        assert len(plan.passes) == (2 if plan.overflow else 1)
        p1 = plan.passes[0]
        for lane, n in pos.items():
            assert int(p1.step.positions[lane]) == n and int(p1.tokens[lane]) == 1000 + lane
            assert not bool(p1.step.call_b[lane])
        for o, d in plan.partner_of.items():
            assert int(p1.step.positions[d]) == pos[o] + 1 and int(p1.tokens[d]) == drafts[o]
            assert bool(p1.step.call_b[d]) and int(p1.step.owner[d]) == o
            assert torch.equal(p1.step.page_table[d], b.page_table[o])
        for l in idle:
            if l not in partners:
                assert int(p1.step.positions[l]) == -1 and int(p1.tokens[l]) == 0
        if plan.overflow:
            p2 = plan.passes[1]
            act2 = {l for l in range(32) if int(p2.step.positions[l]) >= 0}
            assert act2 == set(plan.overflow)
            for o in plan.overflow:
                assert int(p2.step.positions[o]) == pos[o] + 1 and int(p2.tokens[o]) == drafts[o]
            assert not bool(p2.step.call_b.any())
        # result assembly: lane l of pass k returns 10 * l + k (a) and 10 * l + k + 5 (m)
        outs = [(torch.tensor([10 * l + k for l in range(32)]), torch.tensor([10 * l + k + 5 for l in range(32)]))
                for k in range(len(plan.passes))]  # fmt: skip
        res = plan.result(outs)
        for lane in range(32):
            if lane in pos:
                assert res.argmax[lane, 0] == 10 * lane and res.mtp_argmax[lane, 0] == 10 * lane + 5
                if lane in plan.partner_of:
                    d = plan.partner_of[lane]
                    assert res.argmax[lane, 1] == 10 * d and res.mtp_argmax[lane, 1] == 10 * d + 5
                elif lane in plan.overflow:
                    assert res.argmax[lane, 1] == 10 * lane + 1 and res.mtp_argmax[lane, 1] == 10 * lane + 6
                else:
                    assert res.argmax[lane, 1] == -1 and res.mtp_argmax[lane, 1] == -1
            else:
                assert res.argmax[lane].tolist() == [-1, -1]
    if KW.is_split(mode):
        assert seen["packed"] > 1000 and seen["overflow"] > 50
        assert (seen["cross"] > 100) == (mode == "all_split")
    log(f"plan {mode}: {seen}")


def test_cpu_plan_spec_step_refusals():
    """Every host check raises before any device op: trace width, positions at max_model_len (a draft at the last
    position), token ids, the null block / out-of-pool block ids / missing entries in the used part of a page table."""
    from models.demos.motif3.tt import generator as G

    cfg = host_cfg(kv_replicated_decode=True, spec_tokens=1, max_model_len=4096)
    cfg.set_kv_geometry(500, 64)
    ok = _batch({0: 100, 9: 300}, {9: 777})
    G.plan_spec_step(ok, mode="all_split", cfg=cfg, num_blocks=500)
    with pytest.raises(ValueError, match="width"):
        G.plan_spec_step(ok, mode="all_split", cfg=cfg, num_blocks=500, width=128)
    with pytest.raises(ValueError, match="max_model_len"):
        G.plan_spec_step(_batch({0: 4096}, {}, width=128), mode="all_split", cfg=cfg, num_blocks=500)
    with pytest.raises(ValueError, match="max_model_len"):  # the draft would sit at 4096
        G.plan_spec_step(_batch({0: 4095}, {0: 5}, width=128), mode="all_split", cfg=cfg, num_blocks=500)
    G.plan_spec_step(_batch({0: 4094}, {0: 5}, width=128), mode="all_split", cfg=cfg, num_blocks=500)
    G.plan_spec_step(_batch({0: 4095}, {}, width=128), mode="all_split", cfg=cfg, num_blocks=500)
    bad_tok = _batch({0: 10}, {0: cfg.vocab_size})
    with pytest.raises(ValueError, match="token ids"):
        G.plan_spec_step(bad_tok, mode="all_split", cfg=cfg, num_blocks=500)
    b = _batch({3: 200}, {})
    b.page_table[3, 1] = 0  # a null block inside the used entries
    with pytest.raises(ValueError, match="null block"):
        G.plan_spec_step(b, mode="all_split", cfg=cfg, num_blocks=500)
    b = _batch({3: 200}, {})
    b.page_table[3, 0] = 500  # out of the pool
    with pytest.raises(ValueError, match="not a block id"):
        G.plan_spec_step(b, mode="all_split", cfg=cfg, num_blocks=500)
    b = _batch({3: 127}, {3: 9})  # the draft at 128 needs entry 2, which the bridge must fit
    b.page_table[3, 2] = 0
    with pytest.raises(ValueError, match="entry 2"):
        G.plan_spec_step(b, mode="all_split", cfg=cfg, num_blocks=500)
    with pytest.raises(ValueError, match="kv-write mode"):
        G.plan_spec_step(ok, mode="bogus", cfg=cfg, num_blocks=500)


def test_cpu_plain_decode_refusals(monkeypatch):
    """Plain ``decode_forward`` (a non-speculating launch) refuses a bad step before any device op, as the spec path's
    plan does (review 2026-10-03): an active lane's token >= vocab (the embedding would read past its table) or < 0
    (it would be embedded as padding), a position at max_model_len, the null block or an out-of-pool id in the used
    page-table entries. Nothing is staged, copied or run; an inactive lane may carry any token (it is zeroed). The
    speculating launch refuses the same anchors."""
    from models.demos.motif3.tt import generator as G

    gen, model, dev, pool, tr = emu_generator(monkeypatch, spec=False, num_blocks=500, max_model_len=4096)
    assert gen.serving_path == ("plain", "all") and not gen._paths
    copies = []
    real_copy = G.ttnn.copy_host_to_device_tensor
    monkeypatch.setattr(G.ttnn, "copy_host_to_device_tensor", lambda h, d: copies.append(d) or real_copy(h, d),
                        raising=False)  # fmt: skip
    V, W = gen.vocab_size, 64

    def plain(tok0: int, *, pos0: Optional[int] = None, inactive_tok: int = 0, pt_patch=None) -> api.DecodeBatch:
        b = _batch({0: 100}, {}, width=W).anchors()  # lane 0 at position 100 on blocks 1, 2; lane 5 inactive
        b.tokens[0], b.tokens[5] = tok0, inactive_tok
        if pos0 is not None:
            b.positions[0] = pos0
        if pt_patch is not None:
            b.page_table[0, pt_patch[0]] = pt_patch[1]
        return b

    cases = [
        (plain(V), "token ids"),
        (plain(-3), "token ids"),
        (plain(7, pos0=4096), "max_model_len"),
        (plain(7, pt_patch=(1, 0)), "null block"),
        (plain(7, pt_patch=(0, 500)), r"not a block id in \[1, 500\)"),
    ]
    for b, msg in cases:
        with pytest.raises(ValueError, match=msg):
            gen.decode_forward(b, kv_cache=pool, enable_trace=False)
        assert not gen._paths and not copies and model.decode_calls["plain"] == 0, "a refused step touched the device"
    gen.decode_forward(plain(7, inactive_tok=V + 5), kv_cache=pool, enable_trace=False)
    assert model.decode_calls["plain"] == 1 and gen.serving_path in gen._paths and copies
    gen2, model2, _, pool2, _ = emu_generator(monkeypatch, num_blocks=500, max_model_len=4096)
    assert gen2.spec_launch
    for tok in (V, -3):
        with pytest.raises(ValueError, match="token ids"):
            gen2.decode_forward(plain(tok), kv_cache=pool2, enable_trace=False)
    assert not gen2._paths and model2.decode_calls == {"plain": 0, "spec": 0, "wide": 0}


@pytest.mark.parametrize("mode,kvr", [("all_split", True), ("row_split", False), ("all", True), ("row", False)])
def test_cpu_emulated_spec_lossless(monkeypatch, mode, kvr):
    """Greedy spec decode == greedy plain decode == the reference greedy chain, token for token, on the emulated
    device: 20 requests (the bridge's lanes), the bridge's drafting budget, then every request drafting (overflow
    passes), then 8 requests packed on rows 0-1 only (partners on other rows under KV-R) -- all on the same prefilled
    state (decode rewrites only positions >= the prompt). Traced (emulated replays) and eager. The MTP cache stays
    consistent on every anchor (design G8)."""
    gen, model, dev, pool, tr = emu_generator(monkeypatch, kvr=kvr, spec_kv_mode=mode,
                                              decode_kv_mode="all" if kvr else "row")  # fmt: skip
    W = 128
    gen.extra_decode_paths = [("plain", gen.decode_kv_mode)]
    gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=W)
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
    assert gen.serving_path == ("spec", mode) and len(tr.captures) == 2
    reqs = emu_requests(dev, 20, seed=3, width=W, max_new=60)
    drv = GreedyDriver(gen, pool, width=W, block_size=64)
    ref = {r.name: reference_greedy(r, dev.V) for r in reqs}
    sp = drv.run_plain(reqs, path=("plain", gen.decode_kv_mode))
    assert all(r.out == ref[r.name] for r in reqs), "plain decode != reference"
    runs = {}
    for policy in ("budget", "all", "none"):
        st = drv.run_spec(reqs, policy=policy)
        bad = [r.name for r in reqs if r.out != ref[r.name]]
        assert not bad, f"{mode} {policy}: spec decode differs from greedy for {bad}"
        runs[policy] = st
    assert runs["budget"].verify_steps > 10 and runs["budget"].accepted > 0
    if KW.is_split(mode):
        assert runs["budget"].gen_stats["packed_drafts"] > 0 and runs["all"].gen_stats["overflow_passes"] > 0
        assert runs["all"].steps < sp.steps  # speculation saves steps
    else:
        assert runs["budget"].gen_stats["packed_drafts"] == 0  # no split: every draft in pass 2
    # 12 requests: row 0 full (lanes 0-7) + 4 on row 1: row 0's drafts need partners on rows 1-3 (KV-R) or overflow
    lanes = list(range(12))
    rows01 = emu_requests(dev, 12, seed=5, width=W, max_new=40, start_block=2000, lanes=lanes)
    ref2 = {r.name: reference_greedy(r, dev.V) for r in rows01}
    st = drv.run_spec(rows01, policy="all")
    assert all(r.out == ref2[r.name] for r in rows01)
    if mode == "all_split":
        assert st.gen_stats["cross_row_partners"] > 0
    else:
        assert st.gen_stats["cross_row_partners"] == 0 and st.gen_stats["overflow_drafts"] > 0
    # eager (no trace) gives the same tokens
    drv.trace = False
    n_replays, n_spec = tr.replays, model.decode_calls["spec"]
    st = drv.run_spec(rows01, policy="budget")
    assert all(r.out == ref2[r.name] for r in rows01)
    assert tr.replays == n_replays and model.decode_calls["spec"] > n_spec  # eager steps, no replay
    assert dev.stale_mtp == 0, f"{dev.stale_mtp} anchors read an inconsistent MTP history (G8)"
    log(
        f"{mode}: plain {sp.steps} steps; spec budget {runs['budget'].steps} steps acc {runs['budget'].acceptance:.2f} "
        f"{runs['budget'].gen_stats}; all {runs['all'].steps} steps {runs['all'].gen_stats}"
    )


def test_cpu_emulated_overflow_after_batch_change(monkeypatch):
    """Drafts proposed under the budget of one step, then 12 more requests join (a batch change between propose and
    verify, as after a prefill step): the verify step has more drafts than idle lanes, so the generator runs the
    overflow pass, and every token is still the greedy one."""
    gen, model, dev, pool, tr = emu_generator(monkeypatch)
    W = 128
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
    reqs = emu_requests(dev, 16, seed=8, width=W, max_new=50)
    late = emu_requests(dev, 12, seed=9, width=W, max_new=30, start_block=1500, prefix="late",
                        lanes=[lane_of_slot(s) for s in range(16, 28)])  # fmt: skip
    ref = {r.name: reference_greedy(r, dev.V) for r in reqs + late}
    drv = GreedyDriver(gen, pool, width=W, block_size=64)
    st = drv.run_spec(reqs, policy="budget", admit=lambda step: late if step == 3 else [])
    bad = [r.name for r in reqs + late if r.out != ref[r.name]]
    assert not bad, f"spec decode differs after the batch change: {bad}"
    assert st.gen_stats["overflow_passes"] >= 1 and st.gen_stats["overflow_drafts"] > 0
    assert dev.stale_mtp == 0
    log(f"batch change: {st.gen_stats}")


def test_cpu_negative_control_cross_row_without_kvr(monkeypatch):
    """Without KV-R a partner on another DP row reads its own row's stale copy of the owner's history: the plan refuses
    it (``check_kv_write_step``); forced through (partner rule and check bypassed), the emulated device answers wrong
    tokens for those drafts, so a wrong packing cannot pass the lossless tests unnoticed."""
    from models.demos.motif3.tt import generator as G

    gen, model, dev, pool, tr = emu_generator(monkeypatch, kvr=False, spec_kv_mode="row_split", decode_kv_mode="row")
    W = 128
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
    reqs = emu_requests(dev, 8, seed=4, width=W, max_new=40, lanes=list(range(8)))  # row 0 full: no same-row idle
    drv = GreedyDriver(gen, pool, width=W, block_size=64)
    ref = {r.name: reference_greedy(r, dev.V) for r in reqs}
    st = drv.run_spec(reqs, policy="all")
    assert all(r.out == ref[r.name] for r in reqs) and st.gen_stats["packed_drafts"] == 0  # all overflow, correct
    b = _batch({0: 300}, {0: 5})
    with pytest.raises(ValueError, match="own DP row|owner's row|partner"):
        KW.check_kv_write_step(KW.KVWriteStep.packed_verify(b.positions, b.page_table, {0: 9}), "row_split",
                               block_size=64)  # fmt: skip
    # force cross-row partners without KV-R
    monkeypatch.setattr(G, "assign_partner_lanes", lambda pos, dr, cross_row, lanes_per_row=8: KW.assign_partner_lanes(
        pos, dr, cross_row=True, lanes_per_row=lanes_per_row))  # fmt: skip
    monkeypatch.setattr(G, "check_kv_write_step", lambda *a, **k: None)
    reqs = emu_requests(dev, 8, seed=4, width=W, max_new=40, lanes=list(range(8)), start_block=1500)
    ref = {r.name: reference_greedy(r, dev.V) for r in reqs}
    st = drv.run_spec(reqs, policy="all")
    wrong = [r.name for r in reqs if r.out != ref[r.name]]
    assert st.gen_stats["cross_row_partners"] > 0 and wrong, "the stale cross-row read went unnoticed"
    log(f"negative control: {len(wrong)}/8 requests wrong with forced cross-row partners and no KV-R")


@pytest.mark.parametrize("spec, kvr, wait", [(False, False, "block"), (False, True, "spin"), (True, True, "block"),
                                             (True, False, "spin")])  # fmt: skip
def test_cpu_host_staging_fast_lossless(monkeypatch, spec, kvr, wait):
    """B6a (``MOTIF3_HOST_STAGING=fast``, ``MOTIF3_HOST_WAIT``): on the emulated device the fast staging decodes
    exactly the release's tokens (plain path ``row`` / ``all``; spec ``row_split`` / ``all_split``, traced and eager),
    with fewer input copies (an input whose values did not change is not copied again: the page table between block
    crossings, idle lanes' rows), and every persistent input ends holding the values copied last. ``spin`` changes
    nothing the device sees."""
    results = {}
    for staging in ("release", "fast"):
        gen, model, dev, pool, tr = emu_generator(
            monkeypatch, kvr=kvr, spec=spec, cfg_kw=dict(host_staging=staging, host_wait=wait),
            decode_kv_mode="all" if kvr else "row",
        )  # fmt: skip
        assert gen.host_staging == staging and gen.host_wait == wait and (gen.waiter is not None) == (wait == "spin")
        W = 128
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=W)
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
        reqs = emu_requests(dev, 12, seed=8, width=W, max_new=70)
        ref = {r.name: reference_greedy(r, dev.V) for r in reqs}
        drv = GreedyDriver(gen, pool, width=W, block_size=64)
        if spec:
            drv.run_spec(reqs, policy="budget")
        else:
            drv.run_plain(reqs)
        assert all(r.out == ref[r.name] for r in reqs), f"{staging}: decode differs from the reference"
        drv.trace = False
        reqs2 = emu_requests(dev, 5, seed=9, width=W, max_new=20, start_block=2000)
        ref2 = {r.name: reference_greedy(r, dev.V) for r in reqs2}
        drv.run_spec(reqs2, policy="all") if spec else drv.run_plain(reqs2)
        assert all(r.out == ref2[r.name] for r in reqs2), f"{staging}: eager decode differs from the reference"
        for p in gen._paths.values():  # the device inputs hold the last values copied
            for k, v in p.host_last.items():
                assert torch.equal(p.inputs[k].value, v), (p.key, k)
        served = gen._paths[gen.serving_path]
        assert gen._inputs is served.inputs and not served.host_last  # a caller may write them: records dropped
        results[staging] = ({r.name: list(r.out) for r in reqs + reqs2}, dict(gen.stats))
    (out_r, st_r), (out_f, st_f) = results["release"], results["fast"]
    assert out_f == out_r
    assert st_f["decode_steps"] == st_r["decode_steps"] and st_r["input_copies_skipped"] == 0
    assert st_f["input_copies_skipped"] > 0 and st_f["input_copies"] < st_r["input_copies"]
    assert st_f["input_copies"] + st_f["input_copies_skipped"] == st_r["input_copies"]
    log(f"B6a spec={spec} kvr={kvr}: release {st_r['input_copies']} input copies; fast {st_f['input_copies']} "
        f"(+{st_f['input_copies_skipped']} skipped)")  # fmt: skip


def test_cpu_path_host_rows_match_release():
    """``_path_host_rows`` (fast) holds exactly the values the release's ``_path_host_inputs`` converts: the tokens
    rows of ``decode_token_rows`` (inactive lanes 0, negative ids -> pad), the RoPE rows, ``cur`` and the page table
    with idle lanes zeroed (int32 / int64 tables, T32 and T64 rows)."""
    from models.demos.motif3.tt import generator as G
    from models.demos.motif3.tt.embedding import decode_token_rows
    from models.demos.motif3.tt.rope import positions_to_rot_idxs

    cfg = host_cfg(kv_replicated_decode=False, spec_tokens=1, spec_verify="auto")
    stub = SimpleNamespace(cfg=cfg, model=SimpleNamespace(embed=SimpleNamespace(n_streams=4)), wide_rows_per_dp=16)
    rng = torch.Generator().manual_seed(3)
    for trial in range(60):
        kind = (G.PLAIN, G.SPEC, G.WIDE)[trial % 3]
        n = 64 if kind == G.WIDE else 32
        pos = torch.randint(-1, 4000, (n,), generator=rng, dtype=torch.int32)
        pos[torch.randint(0, n, (n // 3,), generator=rng)] = -1
        tok = torch.randint(-1, cfg.vocab_size, (n,), generator=rng, dtype=torch.int32)
        pt = torch.randint(0, 900, (n, 70), generator=rng, dtype=torch.int64 if trial % 2 else torch.int32)
        rows = G.MotifGenerator._path_host_rows(stub, kind, "row", tok, pos, pt)
        act = pos >= 0
        t0 = torch.where(act, tok, torch.zeros_like(tok))
        rpd = 16 if kind == G.WIDE else None
        assert torch.equal(rows["tokens"][0], decode_token_rows(t0, cfg, n_streams=4, rows_per_dp=rpd))
        assert torch.equal(rows["rot"][0], positions_to_rot_idxs(pos, cfg, rows_per_dp=rpd))
        assert rows["tokens"][1] == G.ttnn.uint32 and rows["rot"][1] == G.ttnn.uint32
        if kind == G.PLAIN:
            want = torch.where(act[:, None], pt.to(torch.int32), torch.zeros_like(pt)).to(torch.int32)
            assert rows["pt"][0].dtype == torch.int32 and torch.equal(rows["pt"][0], want)
            assert torch.equal(rows["cur"][0], pos) and rows["cur"][1] == rows["pt"][1] == G.ttnn.int32
            assert rows["cur"][0].data_ptr() != pos.data_ptr()  # kept in host_last: never the caller's tensor
        else:
            assert set(rows) == {"tokens", "rot"}


def test_cpu_replay_waiter():
    """``ReplayWaiter`` (``MOTIF3_HOST_WAIT=spin``) on a fake clock: no spin before a key's first timed replay; then
    it polls (calling ``idle`` each time) until ``residual_ms`` before the shortest of the last ``window`` replays;
    a replay that ended before the spin did (read waited < ``overshoot_ms``) shortens the prediction by
    ``backoff_ms`` (not recorded as a sample), until a real sample resets it; keys are independent."""
    from models.demos.motif3.tt.generator import ReplayWaiter

    t = [0.0]
    idles = []

    def idle():
        idles.append(t[0])
        t[0] += 0.0005

    w = ReplayWaiter(residual_ms=3.0, window=3, overshoot_ms=0.25, backoff_ms=2.0, clock=lambda: t[0], idle=idle)
    assert w.predicted_ms("a") is None
    assert w.spin("a", 0.0) == 0.0 and not idles  # nothing known yet
    w.done("a", 0.0, 0.0, 0.085)
    assert w.predicted_ms("a") == pytest.approx(85.0)
    t[0] = 1.0
    tr = w.spin("a", 1.0)  # spins to 1.0 + 82 ms
    assert tr == pytest.approx(1.082, abs=6e-4) and len(idles) > 100 and w.stats["spins"] == 1
    w.done("a", 1.0, tr, 1.0855)  # blocked 3.5 ms: a real sample
    assert w.predicted_ms("a") == pytest.approx(85.0)
    w.done("a", 2.0, 2.0, 2.084)
    assert w.predicted_ms("a") == pytest.approx(84.0)  # shortest of the window
    for i in range(3):
        w.done("a", 3.0 + i, 3.0 + i, 3.0 + i + 0.090)
    assert w.predicted_ms("a") == pytest.approx(90.0)  # the 84 / 85 ms samples left the window
    w.done("a", 9.0, 9.087, 9.0871)  # the read returned at once: the replay ended before the spin did
    assert w.stats["overshoots"] == 1 and w.predicted_ms("a") == pytest.approx(88.0)
    w.done("a", 10.0, 10.087, 10.0871)
    assert w.predicted_ms("a") == pytest.approx(86.0)
    w.done("a", 11.0, 11.080, 11.0855)  # a real sample: the backoff is dropped
    assert w.predicted_ms("a") == pytest.approx(85.5)
    t[0] = 20.0
    n = len(idles)
    assert w.spin("a", 19.0) == 20.0 and len(idles) == n  # already past the target: no polling
    assert w.predicted_ms("b") is None and w.spin("b", 20.0) == 20.0


def test_cpu_decode_paths_lifecycle(monkeypatch):
    """Decode paths: a speculating launch serves every step through the spec path (``decode_forward`` = an ordinary
    spec step, the MTP cache written too); ``warmup_decode`` stages the serving path and the extra paths, then captures
    each once; after a capture no new path / width is staged (refused) and every prefill shape is frozen; replays run
    the captured step on the new inputs; ``release_traces`` frees the outputs and a re-capture reuses the inputs; a
    non-speculating launch has no spec path in serving. The trace use is checked per path (width, pool)."""
    gen, model, dev, pool, tr = emu_generator(monkeypatch)
    W = 64
    assert gen.supports_spec_decode and gen.spec_launch and gen.serving_path == ("spec", "all_split")
    assert gen.decode_kv_mode == "all" and gen.spec_kv_mode == "all_split"
    api.check_generator_features(gen, api.GeneratorSettings(block_size=64, prefix_caching=True, spec_tokens=1))
    gen._warmed = set()
    with pytest.raises(RuntimeError, match="before the prefill warmup"):
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
    gen._warmed = set(gen.prefill_shapes())
    gen.extra_decode_paths = [("plain", "all"), ("plain", "row")]
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)  # eager warmup first, then 3 captures
    assert set(gen._paths) == {("spec", "all_split"), ("plain", "all"), ("plain", "row")}
    assert len(tr.captures) == 3 and gen.trace_captured
    assert gen._paths[("plain", "row")].kv_write is None and gen._paths[("plain", "all")].kv_write.mode == "all"
    with pytest.raises(RuntimeError, match="stage every path before the capture"):
        gen.decode_forward_spec(_batch({0: 10}, {}, width=W), kv_cache=pool, enable_trace=True, want_logits=False,
                                path=("spec", "row_split"))  # fmt: skip
    with pytest.raises(RuntimeError, match="stage every path before the capture"):
        gen.decode_forward(_batch({0: 10}, {}, width=W).anchors(), kv_cache=pool, enable_trace=True,
                           path=("plain", "all_split"))  # fmt: skip
    with pytest.raises(ValueError, match="width"):
        gen.decode_forward(_batch({0: 10}, {}, width=128).anchors(), kv_cache=pool, enable_trace=True)
    with pytest.raises(ValueError, match="width"):
        gen.decode_forward_spec(_batch({0: 10}, {}, width=128), kv_cache=pool, enable_trace=True, want_logits=False)
    gen._warmed.discard(("sp0", 256))  # a shape the warmup did not compile: refused once a trace exists
    with pytest.raises(RuntimeError, match="not compiled before the decode trace capture"):
        gen.plan_prefill_batch([api.PrefillRequest(lane=0, tokens=torch.arange(100, 300, dtype=torch.int32),
                                                   page_table=torch.arange(1, 9, dtype=torch.int32))])  # fmt: skip
    gen._warmed.add(("sp0", 256))
    # decode_forward on the speculating launch: the spec trace (an ordinary step), the MTP cache written
    reqs = emu_requests(dev, 3, seed=1, width=W, max_new=8)
    r = reqs[0]
    p = r.S
    b = _batch({r.lane: p}, {}, width=W, block_of=lambda lane: r.blocks[0])
    b.tokens[r.lane] = r.first
    n_spec = model.decode_calls["spec"]
    lg = gen.decode_forward(b.anchors(), kv_cache=pool, enable_trace=True)
    assert model.decode_calls["spec"] == n_spec + 1 and int(lg[r.lane].float().argmax()) == lm_next(
        r.prompt + [r.first], dev.V)  # fmt: skip
    blk, row = r.blocks[p // 64], p % 64
    assert int(dev.mtp_kv[0, blk, row]) == int(lg[r.lane].float().argmax()), "the ordinary step wrote no MTP entry"
    lg2 = gen.decode_forward(b.anchors(), kv_cache=pool, enable_trace=True, path=("plain", "all"))
    assert torch.equal(lg, lg2)
    # release: outputs freed, inputs kept, re-capture reuses them
    outs = [gen._paths[k].out for k in gen._paths]
    inputs = {k: dict(gen._paths[k].inputs) for k in gen._paths}
    gen.release_traces()
    assert not gen.trace_captured and all(
        not o.alive for out in outs for o in (out if isinstance(out, tuple) else (out,))
    )
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
    assert all(gen._paths[k].inputs == inputs[k] for k in gen._paths) and len(tr.captures) == 6
    gen.close()
    assert not gen._paths
    # a non-speculating launch: plain serving path, no spec path staged
    gen2, model2, dev2, pool2, tr2 = emu_generator(monkeypatch, spec=False)
    assert gen2.serving_path == ("plain", "all") and not gen2.spec_launch and gen2.supports_spec_decode
    gen2.warmup_decode(kv_cache=pool2, enable_trace=True, page_table_width=W)
    assert set(gen2._paths) == {("plain", "all")}


def test_cpu_legacy_single_path_aliases(monkeypatch):
    """The draft-1 internals other callers use (``tests/test_model_truncated.py``, ``analysis/full_model/``) map onto
    the serving path: ``_trace_id`` / ``_trace_out`` / ``_inputs`` (``tokens``, ``rot``, ``cur``, ``pt`` on the plain
    ``row`` path) / ``_width``, ``_host_inputs(batch)``, ``_write_inputs(batch)``, ``_decode_step(pool)`` and
    ``_capture(pool)`` (returns the trace id and its outputs)."""
    gen, model, dev, pool, tr = emu_generator(monkeypatch, kvr=False, spec=False)
    assert gen.serving_path == ("plain", "row") and gen._trace_id is None and gen._inputs is None
    W = 64
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
    p = gen._paths[gen.serving_path]
    assert gen._trace_id is p.trace_id is not None and gen._trace_out is p.out and gen._trace_pool is pool
    assert set(gen._inputs) == {"tokens", "rot", "cur", "pt"} and gen._width == W and gen._kv_write is None
    b = _batch({3: 100}, {}, width=W).anchors()
    h = gen._host_inputs(b)
    assert set(h) == {"tokens", "rot", "cur", "pt"} and int(h["cur"].value[3]) == 100
    gen._write_inputs(b)
    assert int(gen._inputs["cur"].value[3]) == 100 and int(gen._inputs["tokens"].value[3]) == 1003
    out = gen._decode_step(pool)
    assert int(model.head.logits_to_host(out).float()[3].argmax()) == out.value[3]
    tid, tout = gen._capture(pool)
    assert tid != p.trace_id and tout is not p.out
    gen2, *_ = emu_generator(monkeypatch)  # a speculating launch: the aliases name the spec path
    gen2.warmup_decode(kv_cache=gen2._pool, enable_trace=False, page_table_width=W)
    assert set(gen2._inputs) == {"tokens", "rot"} and gen2._kv_write.mode == "all_split"


def test_cpu_model_decode_spec_threading(monkeypatch):
    """``MotifModel.decode_spec``: every decoder layer gets the step's ``kv_write`` with ITS ``cur_pos`` /
    ``page_table`` and the active mask built from them; the head runs ``stream_mean_norm`` -> ``decode_logits`` (TILE)
    -> ``logits_rm`` + ``argmax_decode``; the MTP layer gets the same ``hn``, ``a``, RoPE tables, ``cur_pos`` /
    ``page_table`` / ``active`` / ``kv_write`` and the pool's MTP cache; ``end_step`` once after the MTP layer; the
    intermediates are freed; refusals without the MTP layer / kv_write / MTP cache."""
    from models.demos.motif3.tt import model as M

    model = object.__new__(M.MotifModel)
    got, freed = [], []

    class L:
        def __init__(self, i):
            self.i = i

        def forward_decode(self, X, **kw):
            got.append(("layer", self.i, X, kw))
            return f"X{self.i}"

    head_calls = []
    model.layers = [L(0), L(1)]
    model.embed = SimpleNamespace(forward_decode=lambda t: "X")
    model.head = SimpleNamespace(
        stream_mean_norm=lambda X: head_calls.append(("smn", X)) or "hn",
        decode_logits=lambda hn, **kw: head_calls.append(("logits", hn, kw)) or "lg",
        logits_rm=lambda lg: head_calls.append(("rm", lg)) or "rm",
        argmax_decode=lambda lg: head_calls.append(("argmax", lg)) or "a",
    )
    mtp_kw = {}

    def mtp_decode(hn, a, **kw):
        mtp_kw.update(hn=hn, a=a, **kw)
        return "m"

    model.mtp = SimpleNamespace(forward_decode=mtp_decode)
    model.rope, model._rope_kinds = None, ("yarn", "plain")
    model.cfg = SimpleNamespace(lanes_per_row=8)
    monkeypatch.setattr(M, "_free", lambda *a: freed.extend(a))
    rot = {"yarn": ("cy", "sy"), "plain": ("cp", "sp")}
    monkeypatch.setattr(M.MotifAttention, "decode_rope_tables", staticmethod(lambda rope, idx, kinds: rot))
    monkeypatch.setattr(M.MotifAttention, "active_mask_from_cur_pos", staticmethod(lambda c, lanes: ("act", c)))
    ends = []
    kvw = SimpleNamespace(cur_pos="cur", page_table="pt", end_step=lambda: ends.append(len(head_calls)))
    caches = M.MotifKVPool(["k0", "k1"], (0, 1), 10, 64, "bfp8", mtp="mtp-cache")
    out = model.decode_spec("tok", rot_idxs="ri", kv_write=kvw, kv_caches=caches)
    assert out == ("rm", "a", "m")
    for i, (_, li, X, kw) in enumerate(got):
        assert li == i and X == ("X" if i == 0 else f"X{i - 1}")
        assert kw == dict(
            rot=rot, cur_pos="cur", page_table="pt", kv_cache=f"k{i}", active=("act", "cur"), kv_write=kvw
        )
    assert head_calls == [("smn", "X1"), ("logits", "hn", {}), ("rm", "lg"), ("argmax", "lg")]
    assert mtp_kw == dict(hn="hn", a="a", rot=rot, cur_pos="cur", page_table="pt", kv_cache="mtp-cache",
                          active=("act", "cur"), kv_write=kvw)  # fmt: skip
    assert ends == [4], "end_step must run once, after the MTP layer"
    assert {"X", "X0", "X1", "lg", "hn", ("act", "cur"), "cy", "sy", "cp", "sp"} <= set(freed)
    assert "rm" not in freed and "a" not in freed and "m" not in freed
    with pytest.raises(ValueError, match="kv_write"):
        model.decode_spec("tok", rot_idxs="ri", kv_write=None, kv_caches=caches)
    with pytest.raises(ValueError, match="MTP cache"):
        model.decode_spec(
            "tok", rot_idxs="ri", kv_write=kvw, kv_caches=M.MotifKVPool(["k0", "k1"], (0, 1), 10, 64, "x")
        )
    model.mtp = None
    with pytest.raises(ValueError, match="MTP layer"):
        model.decode_spec("tok", rot_idxs="ri", kv_write=kvw, kv_caches=caches)


def test_host_bridge_spec_roundtrip(monkeypatch):
    """The real vLLM bridge (``MotifForCausalLM``) over the real generator on the emulated device, driven by the plugin
    emulator of the bridge suite (``test_generator_vllm_host.PluginDriver``: the plugin's own slot bookkeeping,
    ``_step_verifies``, ``_spec_candidate_block``, ``accept_greedy_drafts``, ``propose_draft_tokens``): 20 requests
    prefilled and decoded with speculation, then 10 more admitted mid-stream (a batch change: overflow passes), slot
    remaps included. Every committed token is checked against the emulated model by the driver."""
    H = importlib.import_module("models.demos.motif3.tests.test_generator_vllm_host")  # host only (skips with devices)
    gv = H.gv
    monkeypatch.setattr(gv, "_SEEN_VLLM_BLOCK_SIZE", None)
    monkeypatch.setattr(gv, "_SEEN_VLLM_SERVING", None)
    gen, model, dev, pool_unused, tr = emu_generator(monkeypatch, num_blocks=4000, max_model_len=2048)
    V = dev.V
    dev.next_fn, dev.mtp_fn = H.next_token, H.mtp_token
    settings = api.GeneratorSettings(
        max_batch_size=32, max_seq_len=2048, num_layers=1, block_size=64, chunked_prefill=True, prefix_caching=True,
        max_num_batched_tokens=PP.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, 64),
        long_prefill_token_threshold=PP.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, 64), spec_tokens=1,
    )  # fmt: skip
    gen._pool = None  # the bridge allocates through the generator

    def fake_prefill_batch(requests, *, kv_cache, enable_trace=False):
        out = []
        for r in api.check_prefill_batch(requests):
            toks = r.tokens.tolist()
            first = H.next_token(toks, V)
            dev.prefill(toks, r.page_table, first)
            out.append(H.one_hot(first, V, torch.bfloat16))
        return torch.stack(out)

    gen.prefill_forward_batch = fake_prefill_batch
    bridge = gv.MotifForCausalLM(gen, settings)
    kv = bridge.allocate_kv_cache((4000, 1, 64, api.KV_LATENT_DIM), torch.bfloat16, 53)
    bridge.warmup_model_decode(kv, enable_trace=False, max_batch_size=32, num_blocks=kv.page_table_width)
    drv = H.PluginDriver(bridge, kv, num_slots=32, block_size=64, num_blocks=4000, width=kv.page_table_width,
                         vocab=V, stale_tails=True)  # fmt: skip
    rng = random.Random(21)
    rids = [f"q{i}" for i in range(30)]
    for rid in rids:
        drv.add(rid, [rng.randrange(100, V) for _ in range(rng.randrange(30, 300))])
    drv.prefill(rids[:20])
    for step in range(40):
        if step == 6:
            drv.prefill(rids[20:])  # a prefill step between proposals and their verify: the batch grows by 10
        drv.spec_decode()
    s = bridge.spec_stats
    assert drv.stats["verify_steps"] > 10 and drv.stats["draft_count2"] > 0 and drv.stats["draft_count1"] > 0
    assert gen.stats["overflow_passes"] >= 1 and gen.stats["packed_drafts"] > 0 and dev.stale_mtp == 0
    log(f"bridge round trip: driver {dict(drv.stats)}; generator {gen.stats}; bridge {s}")


# ======================================================================================================================
# host tests: the T64 integration (spec_verify "wide" / "auto"; docs/p5_t64/P5_T64_DESIGN.md §2.2-§2.3, §4.4-§4.7)
# ======================================================================================================================
def test_cpu_wide_paths_and_routing(monkeypatch):
    """The decode paths of each verify mode and the ``auto`` routing: ``packed`` = the T32-spec path alone; ``auto`` =
    the T32-spec path, then the T64 path (``DecodeKVWrite(rows=64, gather="split")``), staged and run eagerly before
    the first capture, captured once each in that order (F3N R2 / R3; staging after the capture is refused); ``wide`` =
    the T64 path alone. ``verify_kind`` / the step routing in ``auto``: ordinary, logits-wanting, sampled and
    fitting verify steps on T32, overflowing verify steps on ONE T64 replay (no overflow pass, R-E6); the generator's
    counters and ``last_verify_kind`` / ``last_spec`` follow; the two-trace info line names the F3N rules."""
    from models.demos.motif3.tt import generator as G
    from models.demos.motif3.tt import verify_plan as VP

    W = 64
    gp, *_ = emu_generator(monkeypatch)
    assert gp.spec_verify == "packed" and gp.serving_paths == [("spec", "all_split")] and gp.wide_path is None
    assert not gp.drafts_all_lanes(range(32)) and "packed" in gp.describe_spec_verify()
    gw, mw, devw, poolw, trw = emu_generator(monkeypatch, spec_verify="wide")
    assert gw.serving_path == ("wide", "all_split") and gw.serving_paths == [("wide", "all_split")] and gw.spec_launch
    assert gw.wide_has_logits and gw.drafts_all_lanes([0]) and "T64 trace alone" in gw.describe_spec_verify()
    logs = []
    gen, model, dev, pool, tr = emu_generator(monkeypatch, spec_verify="auto")
    gen.log = logs.append
    assert gen.serving_path == ("spec", "all_split") and gen.wide_path == ("wide", "all_split")
    assert gen.serving_paths == [("spec", "all_split"), ("wide", "all_split")] == gen.decode_paths()
    assert not gen.wide_has_logits and "c*=19" in gen.describe_spec_verify()
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
    assert len(tr.captures) == 2 and tr.kinds == ["spec", "wide"], tr.kinds  # T32 captured (and replayed) first
    pw = gen._paths[("wide", "all_split")]
    assert pw.traced and pw.kv_write.wide and pw.kv_write.gather == "split" and pw.kv_write.lanes_per_call == 32
    assert pw.out[0] is None and pw.so is None  # argmax-only, no sampler in auto's T64 trace
    assert any("2 decode traces captured" in m and "R4" in m and not m.startswith("warning") for m in logs), logs
    with pytest.raises(RuntimeError, match="stage every path before the capture"):
        gen._stage_path(("wide", "row_split"), W)
    # routing table (host only)
    lanes20 = [lane_of_slot(i) for i in range(20)]
    pos20 = {l: 100 + l for l in lanes20}
    ordinary = _batch(pos20, {}, width=W)
    fits = _batch(pos20, {l: 7 for l in lanes20[:12]}, width=W)  # 12 drafts, 12 idle lanes (KV-R: any row)
    over = _batch(pos20, {l: 7 for l in lanes20}, width=W)  # 20 drafts, 12 idle lanes
    assert gen.verify_kind(ordinary) == "spec" and gen.verify_kind(fits) == "spec" and gen.verify_kind(over) == "wide"
    assert gen.verify_kind(over, want_logits=True) == "spec" and gen.verify_kind(over, sampling=("t",)) == "spec"
    assert gp.verify_kind(over) == "spec" and gw.verify_kind(ordinary) == "wide"
    for b in (ordinary, fits, over):
        assert gen.verify_kind(b) == VP.choose_verify_kind(b, "auto", kv_mode="all_split")
    reqs = emu_requests(dev, 20, seed=31, width=W, max_new=40, lanes=lanes20)
    drv = GreedyDriver(gen, pool, width=W, block_size=64)
    drv.start(reqs)
    st0 = dict(gen.stats)
    n0 = len(tr.kinds)
    drv.spec_step(reqs, policy="all")  # ordinary step (no drafts yet): T32
    assert gen.last_verify_kind == "spec" and tr.kinds[n0:] == ["spec"]
    drv.spec_step(reqs, policy="budget")  # 20 drafts > 12 idle lanes: one T64 replay
    assert gen.last_verify_kind == "wide" and tr.kinds[n0 + 1 :] == ["wide"]
    assert isinstance(gen.last_spec, VP.WideStepPlan)
    drv.spec_step(reqs, policy="budget")  # 12 drafts (the budget): every one fits an idle lane: T32, one replay
    assert gen.last_verify_kind == "spec" and tr.kinds[n0 + 2 :] == ["spec"] and len(gen.last_spec.passes) == 1
    d = {k: gen.stats[k] - st0[k] for k in st0}
    assert d["wide_steps"] == 1 and d["wide_verify_steps"] == 1 and d["wide_drafts"] == 20 and d["overflow_passes"] == 0
    assert d["auto_t32_verifies"] == 1 and d["packed_drafts"] == 12 and d["drafts"] == 32 and d["spec_steps"] == 3
    assert gen.spec_profile["wide"] == 1
    # decode_forward on the auto launch: an ordinary T32-spec step; on the wide launch: the T64 trace (with logits)
    k = len(tr.kinds)
    lg = gen.decode_forward(drv.batch(reqs[:1]).anchors(), kv_cache=pool, enable_trace=True)
    assert tr.kinds[k:] == ["spec"] and tuple(lg.shape) == (32, gen.vocab_size)
    assert G.WIDE == "wide" and G.SPEC_KINDS == ("spec", "wide")


@pytest.mark.parametrize("verify", ["auto", "wide"])
@pytest.mark.parametrize("mode,kvr", [("all_split", True), ("row_split", False)])
def test_cpu_emulated_wide_lossless(monkeypatch, verify, mode, kvr):
    """Greedy speculative decode through the T64 path == the reference greedy chain, token for token, on the emulated
    device (per-DP-row KV copies written through the T64 step's update calls: split-order KV-R calls A / B, or the
    16-user ``row_split`` calls; every row reads its own DP row's copy through its page-table row; the MTP layer's
    cache and history checked): 20 requests with the bridge's budget, every request drafting, no drafting; 32
    requests (no idle lane) every lane drafting; traced (emulated replays) and eager. ``auto``: ordinary and fitting
    verify steps on T32, the others on T64, never an overflow pass; ``wide``: every step on T64. The MTP history stays
    consistent on every anchor (G8)."""
    gen, model, dev, pool, tr = emu_generator(monkeypatch, kvr=kvr, spec_kv_mode=mode, spec_verify=verify,
                                              decode_kv_mode="all" if kvr else "row")  # fmt: skip
    W = 128
    gen.extra_decode_paths = [("plain", gen.decode_kv_mode)]
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
    want = (["spec"] if verify == "auto" else []) + ["wide", "plain"]
    assert tr.kinds[: len(want)] == want, tr.kinds
    reqs = emu_requests(dev, 20, seed=13, width=W, max_new=60)
    drv = GreedyDriver(gen, pool, width=W, block_size=64)
    ref = {r.name: reference_greedy(r, dev.V) for r in reqs}
    sp = drv.run_plain(reqs, path=("plain", gen.decode_kv_mode))
    assert all(r.out == ref[r.name] for r in reqs), "plain decode != reference"
    runs = {}
    for policy in ("budget", "all", "none"):
        st = drv.run_spec(reqs, policy=policy)
        bad = [r.name for r in reqs if r.out != ref[r.name]]
        assert not bad, f"{verify}/{mode} {policy}: spec decode differs from greedy for {bad}"
        runs[policy] = st
        g = st.gen_stats
        assert g["overflow_passes"] == 0, f"{verify} {policy}: an overflow pass ({g})"
        if verify == "wide":
            assert g["wide_steps"] == st.steps and g["packed_drafts"] == 0
    assert runs["all"].gen_stats["wide_steps"] > 0 and runs["all"].steps < sp.steps
    if verify == "auto":  # the budget keeps every draft on an idle lane: T32 only; "all" overflows: T64
        assert runs["budget"].gen_stats["wide_steps"] == 0 and runs["budget"].gen_stats["auto_t32_verifies"] > 0
        assert runs["all"].gen_stats["wide_verify_steps"] > runs["all"].verify_steps // 2
    # 32 requests, every lane drafting (c = 32: no idle lane at all)
    r32 = emu_requests(dev, 32, seed=17, width=W, max_new=40, start_block=1500, prefix="f")
    ref32 = {r.name: reference_greedy(r, dev.V) for r in r32}
    st = drv.run_spec(r32, policy="all")
    assert all(r.out == ref32[r.name] for r in r32), f"{verify}/{mode} c=32: spec decode differs from greedy"
    assert st.gen_stats["wide_drafts"] > 100 and st.accepted > 0 and st.gen_stats["overflow_passes"] == 0
    # eager (no trace) gives the same tokens
    drv.trace = False
    n_replays, n_wide = tr.replays, model.decode_calls["wide"]
    st = drv.run_spec(r32, policy="all")
    assert all(r.out == ref32[r.name] for r in r32)
    assert tr.replays == n_replays and model.decode_calls["wide"] > n_wide
    assert dev.stale_mtp == 0, f"{dev.stale_mtp} anchors read an inconsistent MTP history (G8)"
    log(f"T64 {verify}/{mode}: plain {sp.steps} steps; budget {runs['budget'].gen_stats['wide_steps']} T64 of "
        f"{runs['budget'].steps}; all {runs['all'].gen_stats['wide_steps']} T64 of {runs['all'].steps} steps, "
        f"acceptance {runs['all'].acceptance:.2f}")  # fmt: skip


def test_cpu_wide_refusals(monkeypatch):
    """Every T64 precondition refuses before anything is staged or run: settings that disagree with the config's
    ``spec_verify``; F3N R1 / R-E5 (``ring_gather`` "lean" / "native" on the config -- ``MotifTTConfig`` refuses --
    or on the model's ``MotifCCL``, at construction and again at the decode warmup); a non-split spec KV-write mode;
    R-E7 (``auto`` with the exact-fp32 router unless both the MoE and the config list the T64 row count); a T64 path
    on a ``packed`` config; ``want_logits`` / device sampling on the argmax-only T64 trace of ``auto``; a draft at
    ``max_model_len``."""
    from models.demos.motif3.tt import generator as G

    for ring in ("lean", "native"):
        with pytest.raises(ValueError, match="ring_gather"):
            host_cfg(spec_tokens=1, kv_replicated_decode=True, spec_verify="auto", ring_gather=ring)
    host_cfg(spec_tokens=1, kv_replicated_decode=True, spec_verify="packed", ring_gather="lean")  # packed: allowed
    gen, model, dev, pool, tr = emu_generator(monkeypatch, spec_verify="auto", max_model_len=4096)
    cfg = gen.cfg
    with pytest.raises(ValueError, match="settings.spec_verify"):
        G.MotifGenerator(None, cfg, model, settings=api.GeneratorSettings(spec_tokens=1, block_size=64), log=None)
    G.MotifGenerator(None, cfg, model, settings=api.GeneratorSettings(spec_tokens=1, block_size=64,
                                                                      spec_verify="auto"), log=None)  # fmt: skip
    model.ccl = SimpleNamespace(ring_gather="lean")
    with pytest.raises(ValueError, match="ring_gather='safe'"):
        G.MotifGenerator(None, cfg, model, log=None)
    model.ccl = None
    with pytest.raises(ValueError, match="split KV-write mode"):
        G.MotifGenerator(None, cfg, model, log=None, spec_kv_mode="all")
    # R-E7: the config refuses while ROUTER_EXACT_FP32_DECODE_ROWS lacks 64 and accepts once it lists it; the generator
    # also checks the MoE, and its message names the list that lacks the T64 row count
    from models.demos.motif3.tt import model_config as MC

    real = G._exact_router_refusal(64)  # the unpatched lists: D1 runs the exact router at M = 64 (tt/moe.py)
    if 64 not in MC.ROUTER_EXACT_FP32_DECODE_ROWS:
        assert real is not None and "lacks the 64-row T64 step" in real, real  # only the config's list lags
        with pytest.raises(ValueError, match="exact_fp32"):
            host_cfg(spec_tokens=1, kv_replicated_decode=True, spec_verify="auto", router_logits="exact_fp32")
    else:  # both list 64 (R-E7 closed): the config and the generator accept "auto" + exact_fp32
        assert real is None, real
        ce = host_cfg(spec_tokens=1, kv_replicated_decode=True, spec_verify="auto", router_logits="exact_fp32")
        G.MotifGenerator(None, ce, model, log=None)
    c2 = host_cfg(spec_tokens=1, kv_replicated_decode=True, spec_verify="auto")
    c2.router_logits = "exact_fp32"  # bypasses validate: the generator's own check
    monkeypatch.setattr(G, "ROUTER_EXACT_FP32_DECODE_ROWS", (32,))
    with pytest.raises(ValueError, match="R-E7") as e:
        G.MotifGenerator(None, c2, model, log=None)
    msg = str(e.value)  # the MoE runs it at 64: the refusal is the config's list, not a composite fallback
    assert "ROUTER_EXACT_FP32_DECODE_ROWS = (32,) lacks the 64-row T64 step" in msg, msg
    assert "would take the composite router" not in msg, msg
    monkeypatch.setattr(G, "ROUTER_EXACT_FP32_DECODE_ROWS", (32, 64))
    monkeypatch.setattr(G, "EXACT_ROUTER_DECODE_ROWS", (32,))
    with pytest.raises(ValueError, match="R-E7") as e:
        G.MotifGenerator(None, c2, model, log=None)
    msg = str(e.value)
    assert "runs the exact-fp32 router only at (32,)" in msg and "would take the composite router" in msg, msg
    monkeypatch.setattr(G, "EXACT_ROUTER_DECODE_ROWS", (32, 64))
    G.MotifGenerator(None, c2, model, log=None)  # both list 64: allowed
    # a T64 path on a packed config: refused before anything is staged
    gp, mp, dp_, poolp, trp = emu_generator(monkeypatch)
    gp.extra_decode_paths = [("wide", "all_split")]
    with pytest.raises(ValueError, match="T64 config"):
        gp.warmup_decode(kv_cache=poolp, enable_trace=False, page_table_width=64)
    assert not gp._paths
    # R1 again at the warmup (a test switched the model's ring gather after construction); a fresh emulator (each one
    # patches the fake ttnn trace calls for its own generator)
    gen, model, dev, pool, tr = emu_generator(monkeypatch, spec_verify="auto", max_model_len=4096)
    model.ccl = SimpleNamespace(ring_gather="lean")
    with pytest.raises(ValueError, match="ring_gather='safe'"):
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=64)
    assert not gen._paths
    model.ccl = None
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=64)
    over = _batch({l: 100 + l for l in range(20)}, {l: 7 for l in range(20)}, width=64)
    with pytest.raises(ValueError, match="argmax-only"):
        gen.decode_forward_spec(over, kv_cache=pool, enable_trace=True, want_logits=True, path=("wide", "all_split"))
    gen.sampler = SimpleNamespace()  # (any object: the T64 check comes before the sampler is used)
    with pytest.raises(ValueError, match="holds no sampler"):
        gen.decode_forward_spec(
            _batch({0: 10}, {}, width=64),
            kv_cache=pool,
            enable_trace=True,
            want_logits=False,
            path=("wide", "all_split"),
            sampling=([0.0] * 32, [1.0] * 32, [0] * 32, [None] * 32),
        )
    gen.sampler = None
    n, calls = tr.replays, dict(model.decode_calls)
    last = _batch({0: 4000}, {0: 5}, width=64)
    last.positions[0] = 4095  # a draft at n + 1 = max_model_len (4096): the T64 plan refuses
    last.page_table[0, :64] = torch.arange(1, 65, dtype=torch.int32)
    for path in (("wide", "all_split"), None):  # the T64 plan, and the routed step (T32: the draft fits)
        with pytest.raises(ValueError, match="max_model_len"):
            gen.decode_forward_spec(last, kv_cache=pool, enable_trace=True, want_logits=False, path=path)
    assert tr.replays == n and model.decode_calls == calls


def test_cpu_ring_gather_launch_warning(monkeypatch):
    """Design X3: a launch without a T64 path (``packed``, no speculation, or ``auto`` without the MTP draft) whose
    ring gather is not "safe" -- on the config or on the model's ``MotifCCL`` -- builds and logs ONE warning at
    construction that names the mode and its exposure ("native": prefill not reproducible, the F3 garbage; "lean": the
    native decode gathers); "safe" logs none. A T64 launch refuses those modes instead (``test_cpu_wide_refusals``) and
    so never gets here."""
    from models.demos.motif3.tt import generator as G

    gen, model, dev, pool, tr = emu_generator(monkeypatch)

    def build(cfg, ccl_ring=None):
        logs: List[str] = []
        model.ccl = None if ccl_ring is None else SimpleNamespace(ring_gather=ccl_ring)
        g = G.MotifGenerator(None, cfg, model, log=logs.append)
        assert g.wide_path is None
        return g, [m for m in logs if "ring_gather" in m]

    for kw in (dict(spec_tokens=1, spec_verify="packed"), dict(spec_tokens=0), dict(spec_tokens=0, spec_verify="auto")):
        for ring in ("safe", "lean", "native"):
            g, w = build(host_cfg(kv_replicated_decode=True, ring_gather=ring, **kw))
            assert g._ring_gather_warning() == (w[0] if w else None)
            if ring == "safe":
                assert not w, (kw, w)
                continue
            assert len(w) == 1 and w[0].startswith(f"warning: ring_gather '{ring}' on the config"), (kw, w)
            assert ("not run-to-run reproducible" in w[0]) == (ring == "native"), w
            assert ("10^4 decode steps" in w[0]) == (ring == "lean"), w
            assert "'safe'" in w[0] and "X3" in w[0], w
    # the model's MotifCCL counts too (a test may switch it): config "safe", MotifCCL "native" / "lean"
    for ring in ("native", "lean"):
        g, w = build(host_cfg(kv_replicated_decode=True, spec_tokens=1), ccl_ring=ring)
        assert len(w) == 1 and f"'safe' on the config, '{ring}' on the model's MotifCCL" in w[0], w
    g, w = build(host_cfg(kv_replicated_decode=True, spec_tokens=1), ccl_ring="safe")
    assert not w, w
    model.ccl = None


def test_cpu_r4_replay_order(monkeypatch):
    """F3N rule R4 (``auto``): steps alternate between the T32-spec and the T64 trace, each ending in a blocking read
    (tokens exact throughout); a replay of one trace while the other trace's outputs are still unread is refused before
    anything is enqueued; a step that raises after its replay synchronizes the device and clears the hold, so the next
    step may switch traces."""
    from models.demos.motif3.tt import generator as G

    gen, model, dev, pool, tr = emu_generator(monkeypatch, spec_verify="auto")
    W = 64
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
    reqs = emu_requests(dev, 24, seed=41, width=W, max_new=40)
    ref = {r.name: reference_greedy(r, dev.V) for r in reqs}
    drv = GreedyDriver(gen, pool, width=W, block_size=64)
    drv.start(reqs)
    kinds = []
    for i in range(8):  # "all" proposes every lane's draft (24 > 8 idle lanes: T64 next), "none" proposes nothing
        live = [r for r in reqs if not r.done]
        drv.spec_step(live, policy="all" if i % 2 == 0 else "none")
        kinds.append(gen.last_verify_kind)
        assert gen._unread is None, "a step returned with its outputs unread"
    assert kinds == ["spec", "wide"] * 4, kinds
    n = tr.replays
    gen._unread = ("spec", "all_split")  # a T32 replay whose outputs were never read
    live = [r for r in reqs if not r.done]
    for r in live:
        r.draft = r.nxt
    b = drv.batch(live)
    with pytest.raises(RuntimeError, match="R4"):
        gen.decode_forward_spec(b, kv_cache=pool, enable_trace=True, want_logits=False)
    assert tr.replays == n, "the refused step replayed"
    gen._unread = None
    syncs = []
    real_sync = G.ttnn.synchronize_device
    monkeypatch.setattr(G.ttnn, "synchronize_device", lambda mesh: syncs.append(1) or real_sync(mesh), raising=False)
    real_copy = G.ttnn.copy_device_to_host_tensor

    def boom(*a, **k):
        raise RuntimeError("injected read failure")

    monkeypatch.setattr(G.ttnn, "copy_device_to_host_tensor", boom, raising=False)
    with pytest.raises(RuntimeError, match="injected"):
        gen.decode_forward_spec(b, kv_cache=pool, enable_trace=True, want_logits=False)
    assert tr.replays == n + 1 and syncs and gen._unread is None, "the failed step did not synchronize / clear R4"
    monkeypatch.setattr(G.ttnn, "copy_device_to_host_tensor", real_copy, raising=False)
    for r in live:  # the failed step's KV writes are idempotent: the same step again, then the run to the end
        r.draft = r.nxt
    while any(not r.done for r in reqs):
        drv.spec_step([r for r in reqs if not r.done], policy="all")
    assert all(r.out == ref[r.name] for r in reqs), "tokens differ after the alternating traces"


def test_cpu_drafts_all_lanes(monkeypatch):
    """``MotifGenerator.drafts_all_lanes`` (R-E3, R-E9) = ``verify_plan.drafts_all_lanes`` with the launch's mode and
    settings: False without speculation or in ``packed``; True in ``wide``; in ``auto`` the guarded ``c*`` at the
    bridge's acceptance (None = the prior ``settings.spec_alpha_prior``), or ``settings.wide_min_lanes`` when set;
    duplicates count once; bad lanes / acceptances raise."""
    from models.demos.motif3.tt import verify_plan as VP

    rng = random.Random(3)
    gen_ns, *_ = emu_generator(monkeypatch, spec=False)
    assert not gen_ns.spec_launch and not gen_ns.drafts_all_lanes(range(32))
    for prior, fixed in ((0.85, None), (0.5, None), (0.1, None), (0.85, 24), (0.85, 33)):
        s = api.GeneratorSettings(spec_tokens=1, block_size=64, spec_verify="auto", spec_alpha_prior=prior,
                                  wide_min_lanes=fixed, prefix_caching=True)  # fmt: skip
        gen, *_ = emu_generator(monkeypatch, spec_verify="auto", settings=s)
        for _ in range(60):
            live = rng.sample(range(32), rng.randint(1, 32)) + [0, 0]
            acc = None if rng.random() < 0.3 else rng.random()
            want = VP.drafts_all_lanes(live, spec_verify="auto", ratio=gen.cfg.wide_step_ratio, acceptance=acc,
                                       min_lanes=fixed, prior=prior)  # fmt: skip
            assert gen.drafts_all_lanes(live, acceptance=acc) == want
        c = fixed if fixed is not None else VP.crossover_lanes(prior, gen.cfg.wide_step_ratio)
        assert gen.drafts_all_lanes(range(min(c, 32))) == (c <= 32)
        assert c > 32 or not gen.drafts_all_lanes(range(c - 1))
    gen, *_ = emu_generator(monkeypatch, spec_verify="auto")
    assert gen.drafts_all_lanes(range(19)) and not gen.drafts_all_lanes(range(18))  # c* = 19 at the prior 0.85
    assert not gen.drafts_all_lanes(range(32), acceptance=0.05)  # alpha <= r - 1: never (R-E3 guard)
    with pytest.raises(ValueError):
        gen.drafts_all_lanes([32])
    with pytest.raises(ValueError):
        gen.drafts_all_lanes([1], acceptance=1.5)


def test_cpu_model_decode_wide_threading(monkeypatch):
    """``MotifModel.decode_wide``: the 16-row tokens -> every decoder layer gets the T64 ``kv_write`` with ITS
    ``cur_pos`` / ``page_table`` and the 16-row active mask -> ``stream_mean_norm`` -> ``decode_logits(hn, halves=2)``
    -> (``logits_rm(lg, rows=32)`` only with ``want_rm``) -> ``argmax_decode`` -> the MTP layer with the same
    ``hn``, ``a``, RoPE tables, ``cur_pos`` / ``page_table`` / ``active`` / ``kv_write`` and the MTP cache -> one
    ``end_step``; intermediates freed; host refusals before any device op: no T64 config, a T32 writer, 8-row tokens,
    a "tp" head, no MTP cache / layer / kv_write."""
    from models.demos.motif3.tt import model as M

    model = object.__new__(M.MotifModel)
    got, freed, head_calls = [], [], []

    class L:
        def __init__(self, i):
            self.i = i

        def forward_decode(self, X, **kw):
            got.append(("layer", self.i, X, kw))
            return f"X{self.i}"

    model.layers = [L(0), L(1)]
    model.embed = SimpleNamespace(forward_decode=lambda t: head_calls.append(("embed", t.shape)) or "X")
    model.head = SimpleNamespace(
        vocab_split="mesh",
        stream_mean_norm=lambda X: head_calls.append(("smn", X)) or "hn",
        decode_logits=lambda hn, **kw: head_calls.append(("logits", hn, kw)) or "lg",
        logits_rm=lambda lg, **kw: head_calls.append(("rm", lg, kw)) or "rm",
        argmax_decode=lambda lg: head_calls.append(("argmax", lg)) or "a",
    )
    mtp_kw = {}
    model.mtp = SimpleNamespace(forward_decode=lambda hn, a, **kw: mtp_kw.update(hn=hn, a=a, **kw) or "m")
    model.rope, model._rope_kinds = None, ("yarn", "plain")
    model.cfg = SimpleNamespace(lanes_per_row=8, wide_rows_per_dp=16, max_batch=32)
    monkeypatch.setattr(M, "_free", lambda *a: freed.extend(a))
    rot = {"yarn": ("cy", "sy"), "plain": ("cp", "sp")}
    monkeypatch.setattr(M.MotifAttention, "decode_rope_tables", staticmethod(lambda rope, idx, kinds: rot))
    masks = []
    monkeypatch.setattr(M.MotifAttention, "active_mask_from_cur_pos",
                        staticmethod(lambda c, lanes: masks.append(lanes) or ("act", c)))  # fmt: skip
    ends = []
    kvw = SimpleNamespace(cur_pos="cur", page_table="pt", lanes_per_row=16, end_step=lambda: ends.append(1))
    caches = M.MotifKVPool(["k0", "k1"], (0, 1), 10, 64, "bfp8", mtp="mtp-cache")
    tok16 = SimpleNamespace(shape=(4, 16))
    for want_rm in (False, True):
        got.clear(), freed.clear(), head_calls.clear(), mtp_kw.clear(), ends.clear(), masks.clear()
        out = model.decode_wide(tok16, rot_idxs="ri", kv_write=kvw, kv_caches=caches, want_rm=want_rm)
        assert out == (("rm" if want_rm else None), "a", "m")
        for i, (_, li, X, kw) in enumerate(got):
            assert li == i and X == ("X" if i == 0 else f"X{i - 1}")
            assert kw == dict(rot=rot, cur_pos="cur", page_table="pt", kv_cache=f"k{i}", active=("act", "cur"),
                              kv_write=kvw)  # fmt: skip
        want_calls = [("embed", (4, 16)), ("smn", "X1"), ("logits", "hn", {"halves": 2})]
        want_calls += [("rm", "lg", {"rows": 32})] if want_rm else []
        assert head_calls == want_calls + [("argmax", "lg")]
        assert mtp_kw == dict(hn="hn", a="a", rot=rot, cur_pos="cur", page_table="pt", kv_cache="mtp-cache",
                              active=("act", "cur"), kv_write=kvw)  # fmt: skip
        assert masks == [16] and ends == [1]
        assert {"X", "X0", "X1", "lg", "hn", ("act", "cur"), "cy", "sy", "cp", "sp"} <= set(freed)
        assert not {"rm", "a", "m"} & set(freed)
    freed.clear()
    out = model.decode_wide(tok16, rot_idxs="ri", kv_write=kvw, kv_caches=caches, keep_hidden=True)
    assert out == (None, "a", "m", "hn") and "hn" not in freed
    got.clear()
    bad = [
        (dict(kv_write=None), "kv_write"),
        (dict(kv_write=SimpleNamespace(cur_pos="c", page_table="p", lanes_per_row=8)), "rows=64"),
        (dict(tokens=SimpleNamespace(shape=(4, 8))), "rows per DP row"),
        (dict(kv_caches=M.MotifKVPool(["k0", "k1"], (0, 1), 10, 64, "x")), "MTP cache"),
    ]
    for kw, msg in bad:
        args = dict(tokens=tok16, rot_idxs="ri", kv_write=kvw, kv_caches=caches) | kw
        tokens = args.pop("tokens")
        with pytest.raises(ValueError, match=msg):
            model.decode_wide(tokens, **args)
    model.head.vocab_split = "tp"
    with pytest.raises(ValueError, match="mesh"):
        model.decode_wide(tok16, rot_idxs="ri", kv_write=kvw, kv_caches=caches)
    model.head.vocab_split = "mesh"
    model.cfg.wide_rows_per_dp = 0
    with pytest.raises(ValueError, match="T64 config"):
        model.decode_wide(tok16, rot_idxs="ri", kv_write=kvw, kv_caches=caches)
    model.cfg.wide_rows_per_dp = 16
    model.mtp = None
    with pytest.raises(ValueError, match="MTP layer"):
        model.decode_wide(tok16, rot_idxs="ri", kv_write=kvw, kv_caches=caches)
    assert not got, "a refused decode_wide ran a layer"


@pytest.mark.parametrize("verify", ["packed", "auto"])
def test_host_bridge_spec_roundtrip_t64(monkeypatch, verify):
    """The real vLLM bridge over the real generator (emulated device) with ``spec_verify`` = ``verify``, driven by the
    plugin emulator at up to 30 live requests: the bridge asks ``generator.drafts_all_lanes`` with its prior-smoothed
    acceptance (R-E3: a fresh server whose traffic is a burst still drafts); in ``auto`` every live lane drafts from
    ``c*`` (19) live lanes on and the generator verifies the drafts that do not fit idle lanes in one T64 step, while
    ``packed`` keeps the idle-lane budget. Every committed token is checked by the driver (lossless)."""
    H = importlib.import_module("models.demos.motif3.tests.test_generator_vllm_host")  # host only (skips with devices)
    gv = H.gv
    monkeypatch.setattr(gv, "_SEEN_VLLM_BLOCK_SIZE", None)
    monkeypatch.setattr(gv, "_SEEN_VLLM_SERVING", None)
    settings = api.GeneratorSettings(
        max_batch_size=32, max_seq_len=2048, num_layers=1, block_size=64, chunked_prefill=True, prefix_caching=True,
        max_num_batched_tokens=PP.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, 64),
        long_prefill_token_threshold=PP.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, 64), spec_tokens=1,
        spec_verify=verify,
    )  # fmt: skip
    gen, model, dev, pool_unused, tr = emu_generator(monkeypatch, num_blocks=4000, max_model_len=2048,
                                                     spec_verify=verify, settings=settings)  # fmt: skip
    V = dev.V
    dev.next_fn, dev.mtp_fn = H.next_token, H.mtp_token
    gen._pool = None  # the bridge allocates through the generator

    def fake_prefill_batch(requests, *, kv_cache, enable_trace=False):
        out = []
        for r in api.check_prefill_batch(requests):
            toks = r.tokens.tolist()
            first = H.next_token(toks, V)
            dev.prefill(toks, r.page_table, first)
            out.append(H.one_hot(first, V, torch.bfloat16))
        return torch.stack(out)

    gen.prefill_forward_batch = fake_prefill_batch
    bridge = gv.MotifForCausalLM(gen, settings)
    kv = bridge.allocate_kv_cache((4000, 1, 64, api.KV_LATENT_DIM), torch.bfloat16, 53)
    gen._warmed = set(gen.prefill_shapes())
    bridge._prefill_warmed = True  # (the emulated model has no prefill programs to warm)
    bridge.warmup_model_decode(kv, enable_trace=True, max_batch_size=32, num_blocks=kv.page_table_width)
    assert len(tr.captures) == (2 if verify == "auto" else 1)
    drv = H.PluginDriver(bridge, kv, num_slots=32, block_size=64, num_blocks=4000, width=kv.page_table_width,
                         vocab=V, stale_tails=True)  # fmt: skip
    rng = random.Random(23)
    rids = [f"q{i}" for i in range(30)]
    for rid in rids:
        drv.add(rid, [rng.randrange(100, V) for _ in range(rng.randrange(30, 300))])
    drv.prefill(rids[:30])
    for _ in range(30):
        drv.spec_decode()
    s, g = bridge.spec_stats, gen.stats
    assert drv.stats["verify_steps"] > 10 and drv.stats["draft_count2"] > 0 and dev.stale_mtp == 0
    if verify == "auto":
        assert g["wide_steps"] > 5 and g["wide_drafts"] > 100 and g["overflow_passes"] == 0, g
        assert s.all_lane_proposals > 0 and s.drafts_beyond_budget > 0
    else:
        assert g["wide_steps"] == 0 and s.all_lane_proposals == 0
    log(f"bridge round trip ({verify}): driver {dict(drv.stats)}; generator "
        f"{ {k: v for k, v in g.items() if v} }; bridge {s}")  # fmt: skip


# ======================================================================================================================
# device tests
# ======================================================================================================================
SPEC_LAYERS = int(os.environ.get("MOTIF3_SPEC_LAYERS", "53"))  # < 53: a plumbing run (the bars assume 53 layers)
MAX_LEN = 32768
BS = 64
NUM_BLOCKS = api.expected_num_blocks()  # 4129: the serving pool (262,144 tokens, block 64, 32 seqs)
WIDTH = min(api.cdiv(MAX_LEN, BS), NUM_BLOCKS)  # 512: the plugin's page-table width
EXTRA_PATHS = (("plain", "all"), ("plain", "row"), ("plain", "all_split"))  # captured next to the spec trace
MAX_NEW = int(os.environ.get("MOTIF3_SPEC_MAX_NEW", "128"))
CP9_ROUNDS = int(os.environ.get("MOTIF3_CP9_ROUNDS", "10"))  # ~4 s per round at 53 layers (150: the design's 10 min)
SPEC_TIMEOUT = 3600
KVR_GATE_MS = 2.0  # G13b: the design's KV-R write gate (plain all_split - row, ms per step), at ring_gather "native"
KVR_GUARD_MS = 2.25  # G13b regression guard on plain all (KV-R without spec); the design's 2.0 gate is for all_split
# ring_gather="safe" (B0's default) reroutes every race-prone TP-ring gather: +0.26-0.45 ms on every decode path,
# unevenly, which moved the asserted difference by ~+0.1 ms to 1.98-2.05 ms (docs/determinism/INVESTIGATION.md §5.1,
# FIX.md §7.2: "give the gate some margin or more replays per median"): both bars get this margin under "safe", and the
# replay medians take G13B_REPLAYS replays (10 before)
SAFE_GATHER_MARGIN_MS = 0.45
G13B_REPLAYS = 20
GOLD_BF16 = PROJECT_ROOT / "goldens" / "c2"
GEN_SYSTEM = "You are a helpful assistant."
QUESTIONS = [  # the 4 validation prompts of FULL_MODEL_VALIDATION §3 first
    "Explain why the sky is blue during the day but often red or orange at sunset. Keep it to one short paragraph.",
    "대한민국의 수도는 어디이며, 그 도시가 역사적으로 중요한 이유를 두세 문장으로 설명해 주세요.",
    "A bakery sells muffins for $3 each and cookies for $2 each. On Monday it sold 45 muffins and twice as many "
    "cookies as muffins. On Tuesday it sold 30 muffins and 50 cookies. How much money did the bakery make in total "
    "over the two days? Show your work and give the final answer.",
    "Write a Python function `is_palindrome(s: str) -> bool` that returns True if the string is a palindrome, ignoring "
    "case and non-alphanumeric characters. Include a docstring and two example calls.",
    "What are the main differences between TCP and UDP? Answer in a few bullet points.",
    "Write a haiku about autumn leaves, then explain the imagery you used.",
    "한국의 전통 음식 세 가지를 소개하고 각각의 특징을 설명해 주세요.",
    "Solve for x: 3x + 7 = 2x - 5. Explain each step.",
    "Write a Python function that returns the n-th Fibonacci number using memoization, with a short docstring.",
    "Summarize the causes of the French Revolution in one paragraph.",
    "What is the difference between a list and a tuple in Python? Give an example of each.",
    "Explain how photosynthesis works to a ten-year-old.",
    "A train travels 180 km in 2.5 hours. What is its average speed in km/h and in m/s?",
    "Translate into English and explain the meaning of the Korean proverb '천 리 길도 한 걸음부터'.",
    "Write a SQL query that returns the top 5 customers by total order amount from tables customers(id, name) and "
    "orders(id, customer_id, amount).",
    "Why do we have seasons on Earth? Keep the answer short.",
    "Give three tips for writing clean, maintainable code.",
    "What is the time complexity of binary search, and why?",
    "서울에서 부산까지 KTX로 가는 데 걸리는 시간과 이용 팁을 알려 주세요.",
    "Explain the difference between supervised and unsupervised learning with one example each.",
    "Write a short story opening (three sentences) about a robot who learns to paint.",
    "If a rectangle has a perimeter of 30 cm and its length is twice its width, what are its dimensions?",
    "What does the HTTP status code 404 mean, and how is it different from 500?",
    "Write a bash one-liner that counts the number of lines in all .py files in a directory tree.",
    "Explain what a black hole is in two or three sentences.",
    "List the planets of the solar system in order from the Sun, with one fact about each.",
    "What is the Pythagorean theorem? Give a worked example.",
    "인공지능이 일상생활에 미치는 영향을 장점과 단점으로 나누어 설명해 주세요.",
    "Write a JavaScript function that debounces another function.",
    "Explain the concept of compound interest with a numeric example.",
    "What are the benefits of regular exercise? Answer in a short list.",
    "Describe the water cycle in four steps.",
]


class SpecSession:
    """The shared 53-layer speculating generator of the module (plus the plain traces) and its helpers."""

    def __init__(self, mesh, gen, pool, guard, *, cap: Optional[int] = None):
        from models.demos.motif3.tests.test_resumed_prefill import Blocks

        self.mesh, self.gen, self.pool, self.guard = mesh, gen, pool, guard
        self.cfg = gen.cfg
        self.blocks = Blocks(NUM_BLOCKS)
        self.eos = tuple(int(e) for e in self.cfg.eos_token_ids)
        self.drv = GreedyDriver(gen, pool, width=WIDTH, block_size=BS, eos=self.eos)
        self._tok = None
        self._chat: Dict[Tuple[Any, ...], Req] = {}
        self.report: Dict[str, Any] = {}
        self.cap = max(MAX_NEW, 64) if cap is None else int(cap)  # every test's max_new fits a prefilled request

    @property
    def tokenizer(self):
        if self._tok is None:
            from models.demos.motif3.reference.tokenizer import load_tokenizer

            self._tok = load_tokenizer()
        return self._tok

    def programs(self) -> int:
        return int(self.mesh.num_program_cache_entries())

    def prefill(self, r: Req, *, start: int = 0, end: Optional[int] = None) -> torch.Tensor:
        """One ``prefill_forward`` call of ``r.prompt[:end]`` from ``start`` (sets ``r.first`` at the prompt's end)."""
        e = r.S if end is None else int(end)
        pt = torch.zeros(WIDTH, dtype=torch.int32)
        n = api.cdiv(e, BS)
        pt[:n] = torch.tensor(r.blocks[:n], dtype=torch.int32)
        req = api.PrefillRequest(lane=r.lane, tokens=torch.tensor(r.prompt[:e], dtype=torch.int32), page_table=pt,
                                 start=int(start))  # fmt: skip
        lg = self.gen.prefill_forward(req, kv_cache=self.pool)
        # finite and in range (F3N rule R7): a stale tile of the TP-ring all-gather race reads as finite garbage,
        # |logit| 1e18-1e20 (F3, docs/p5_t64/f3.md §2; closed by ring_gather="safe", F3N R1), and must fail loudly;
        # these sessions hold several decode traces (two traces are safe under F3N R1-R5, f3.md §7)
        from models.demos.motif3.tests.test_resumed_prefill import assert_sane

        assert_sane(lg[None], f"{r.name} prefill logits [{start}, {e})")
        if e == r.S:
            r.first = int(lg.float().argmax())
        return lg

    def new_req(self, name: str, ids: Sequence[int], lane: int, max_new: int = MAX_NEW) -> Req:
        r = Req(name, list(ids), self.blocks.take(api.cdiv(len(ids) + max_new + 2, BS)), lane, -1, max_new)
        self.prefill(r)
        return r

    def chat_ids(self, qi: int, think: bool) -> List[int]:
        from models.demos.motif3.reference.tokenizer import encode_chat

        msgs = [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": QUESTIONS[qi]}]
        return encode_chat(msgs, self.tokenizer, enable_thinking=think)

    def chat_req(self, qi: int, think: bool, lane: int, max_new: int = MAX_NEW) -> Req:
        """The prefilled request of question ``qi`` (thinking on / off), prefilled once per session; later runs reuse
        its prompt KV on any lane (KV-R; every run rewrites its positions >= the prompt before reading them)."""
        key = (int(qi), bool(think))
        cap = self.cap  # every test's max_new fits the blocks
        if key not in self._chat:
            self._chat[key] = self.new_req(f"q{qi}{'T' if think else ''}", self.chat_ids(qi, think), lane, cap)
        r = self._chat[key]
        assert max_new <= cap
        r.lane, r.max_new = int(lane), int(max_new)
        return r


# One device session at a time in this module (the mesh opens once per session): a session fixture closes every
# other open session first (the T64 session runs after the packed one; each fixture's own teardown is then a no-op).
_OPEN_SESSIONS: Dict[str, Callable[[], None]] = {}


def _claim_device(name: str) -> None:
    for other in [k for k in _OPEN_SESSIONS if k != name]:
        log(f"closing the {other!r} device session for the {name!r} one")
        _OPEN_SESSIONS.pop(other)()


def _session_closer(name: str, mesh, holder: Dict[str, Any]) -> Callable[[], None]:
    from models.demos.motif3.tt.model_config import close_motif_mesh

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
    return close


@pytest.fixture(scope="module")
def spec_session():
    import ttnn

    from models.demos.motif3.tests.test_resumed_prefill import RefuseReads
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.model import device_bytes_per_chip
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR, open_motif_mesh

    if os.environ.get("OMP_WAIT_POLICY", "").upper() != "PASSIVE":
        log("WARNING: OMP_WAIT_POLICY is not PASSIVE (the serving setting): host timings will be pessimistic")
    _claim_device("spec")
    mesh = open_motif_mesh()
    holder: Dict[str, Any] = {}
    close = _session_closer("spec", mesh, holder)
    gen = None
    try:
        log_fabric(mesh, "spec_decode")
        A = api.DEFAULT_PREFILL_ALIGNMENT
        budget = PP.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, A)
        settings = api.GeneratorSettings(
            max_batch_size=api.NUM_LANES, max_seq_len=MAX_LEN, num_layers=SPEC_LAYERS, kv_cache_dtype="bfp8",
            weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=BS, weights_source="TT cache only (test)",
            chunked_prefill=True, prefix_caching=True, max_num_batched_tokens=budget,
            long_prefill_token_threshold=budget, spec_tokens=1,
        )  # fmt: skip
        guard = RefuseReads(DEFAULT_WEIGHTS_DIR)
        t0 = time.time()
        gen = holder["gen"] = MotifGenerator.create(hf_config=None, mesh_device=mesh, settings=settings, source=guard)
        log(f"generator: {gen.num_layers} layers + MTP in {time.time() - t0:.1f} s; cache misses "
            f"{gen.model.cache_misses}; refused reads {guard.refused[:3]}")  # fmt: skip
        assert not gen.model.cache_misses and not guard.refused, "not a TT-cache-only boot"
        assert gen.spec_launch and gen.serving_path == ("spec", "all_split") and gen.decode_kv_mode == "all"
        gen.extra_decode_paths = list(EXTRA_PATHS)
        pool = gen.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BS, num_layers=SPEC_LAYERS)
        t0 = time.time()
        gen.warmup_prefill(kv_cache=pool, enable_trace=False)
        t_wp = time.time() - t0
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=WIDTH)
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=WIDTH)
        trace = device_bytes_per_chip(mesh, ttnn.BufferType.TRACE) or {}
        dram = device_bytes_per_chip(mesh) or {}
        log(
            f"warmup: prefill {t_wp:.1f} s ({len(gen.warmed_shapes)} shapes), decode eager "
            f"{gen.timings.get('warmup_decode_eager_s', 0):.1f} s, capture "
            f"{gen.timings.get('capture_decode_s', 0):.1f} s ({len(gen.decode_paths())} traces: "
            f"{ {k: round(v, 2) for k, v in gen.timings.items() if k.startswith('capture_decode_')} }); program cache "
            f"{mesh.num_program_cache_entries()}; trace region {trace.get('allocated', 0) / 2**20:.1f} of "
            f"{trace.get('total', 0) / 2**20:.0f} MiB; DRAM {dram.get('allocated', 0) / 1e9:.2f} of "
            f"{dram.get('total', 0) / 1e9:.2f} GB per chip"
        )
        yield SpecSession(mesh, gen, pool, guard)
    finally:
        _OPEN_SESSIONS.pop("spec", None)
        close()


def _same_result(r1: api.SpecDecodeResult, r2: api.SpecDecodeResult, lanes: Sequence[int], drafted=()) -> bool:
    lanes = list(lanes)
    ok = bool(torch.equal(r1.argmax[lanes, 0], r2.argmax[lanes, 0]))
    ok = ok and bool(torch.equal(r1.mtp_argmax[lanes, 0], r2.mtp_argmax[lanes, 0]))
    d = list(drafted)
    if d:
        ok = ok and bool(torch.equal(r1.argmax[d, 1], r2.argmax[d, 1]))
        ok = ok and bool(torch.equal(r1.mtp_argmax[d, 1], r2.mtp_argmax[d, 1]))
    if r1.logits is not None and r2.logits is not None:
        ok = ok and bool(torch.equal(r1.logits[lanes], r2.logits[lanes]))
    return ok


def _replay_vs_eager(s: SpecSession, live: Sequence[Req], tag: str) -> None:
    """The same spec step traced and eager (idempotent KV writes): bitwise equal outputs."""
    b = s.drv.batch(live)
    want = not b.is_verify
    rt = s.gen.decode_forward_spec(b, kv_cache=s.pool, enable_trace=True, want_logits=want)
    re_ = s.gen.decode_forward_spec(b, kv_cache=s.pool, enable_trace=False, want_logits=want)
    lanes = [r.lane for r in live]
    drafted = [r.lane for r in live if r.draft is not None]
    ok = _same_result(rt, re_, lanes, drafted)
    log(f"G-S6 {tag}: replay == eager bitwise {ok} ({len(lanes)} lanes, {len(drafted)} drafts, "
        f"{len(s.gen.last_spec.passes)} pass(es))")  # fmt: skip
    assert ok, f"{tag}: trace replay != eager step"


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_gs6_trace_safety_serving_order(spec_session):
    """G-S6: in the session that captured every trace, the serving order prefill (sp0) -> ordinary spec step -> a
    vLLM-chunked 9000-token prompt (sp0 1024, then an sp1 8192 continuation) -> verify steps (row 0 full: cross-row
    partners) -> a batch change (overflow passes) -> a cold sp0 prefill -> ordinary steps. At each phase a spec step
    replayed == the same step eager, bitwise; at the end the very first step, re-run, is bitwise equal to its first
    run (nothing the later prefills / replays allocated or wrote corrupted its KV, the weights or the persistent
    inputs); the program cache never grows."""
    s = spec_session
    gen, drv = s.gen, s.drv
    pc0 = s.programs()
    A = s.chat_req(0, True, lane=0, max_new=48)
    drv.start([A])
    bA = drv.batch([A])
    ref = gen.decode_forward_spec(bA, kv_cache=s.pool, enable_trace=True, want_logits=True)
    _replay_vs_eager(s, [A], "phase 1 (ordinary, 1 lane)")
    # vLLM-chunked long prompt: [0, 1024) sp0, then [1024, 9000) as one sp1 chunk of bucket 8192
    src = []
    while len(src) < 9000:
        src += s.chat_ids(len(src) % len(QUESTIONS), True)[1:]
    P = Req("long", src[:9000], s.blocks.take(api.cdiv(9000 + 64, BS)), 9, -1, 24)
    s.prefill(P, end=1024)
    t0 = time.time()
    s.prefill(P, start=1024)
    plan = gen.last_prefill.jobs[0].plan
    log(
        f"G-S6 sp1 continuation: chunks {[(c.start, c.bucket, c.path) for c in plan.chunks]} in "
        f"{time.time() - t0:.1f} s"
    )
    assert all(c.path == "sp1" for c in plan.chunks) and max(c.bucket for c in plan.chunks) == 8192
    B = [s.chat_req(qi, False, lane=lane, max_new=40) for qi, lane in zip(range(1, 8), range(1, 8))]  # row 0 full
    drv.start(B + [P])
    live = [A] + B + [P]
    st = RunStats()
    g0 = dict(gen.stats)
    for i in range(5):
        live = [r for r in live if not r.done]
        if i == 2:
            _replay_vs_eager(s, live, "phase 2 (verify, cross-row partners)")
        drv.spec_step(live, policy="all", st=st)
    assert gen.stats["cross_row_partners"] > g0["cross_row_partners"], "no cross-row partner in the verify steps"
    C = [s.chat_req(qi, True, lane=lane, max_new=40) for qi, lane in zip(range(8, 28), range(10, 30))]
    drv.start(C)
    live = [r for r in live + C if not r.done]
    _replay_vs_eager(s, live, "phase 3 (batch change: overflow pass)")
    assert len(gen.last_spec.passes) == 2, "the batch change should have overflowed the idle lanes"
    for _ in range(3):
        live = [r for r in live if not r.done]
        drv.spec_step(live, policy="all", st=st)
    D = s.chat_req(28, False, lane=31, max_new=40)  # a cold sp0 prefill after verify and overflow steps
    drv.start([D])
    live = [r for r in live + [D] if not r.done]
    for _ in range(3):
        live = [r for r in live if not r.done]
        drv.spec_step(live, policy="none", st=st)
    live = [r for r in live if not r.done]
    _replay_vs_eager(s, live, "phase 4 (ordinary)")
    again = gen.decode_forward_spec(bA, kv_cache=s.pool, enable_trace=True, want_logits=True)
    same = _same_result(ref, again, [A.lane])
    log(
        f"G-S6: the first step re-run after {st.steps} spec steps and 4 prefills: bitwise equal {same}; overflow "
        f"passes {gen.stats['overflow_passes'] - g0['overflow_passes']}; program cache {pc0} -> {s.programs()}"
    )
    assert same, "the first step's outputs changed: something corrupted its KV, the weights or the inputs"
    assert s.programs() == pc0, "a program compiled after the capture"


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_r5_lane_relocation(spec_session):
    """Review R5 at full depth: the same token at the same position (the same KV history) computed on lanes of other
    DP rows / other slots of the same row gives bitwise equal logits, ``a`` and ``m``; a packed draft on a cross-row
    partner, on a same-row partner and in the overflow pass gives bitwise what the ordinary step on the owner's lane at
    ``n + 1`` gives (logits too: read through the generator's ``spec_observer``)."""
    s = spec_session
    gen, drv = s.gen, s.drv
    Q = s.chat_req(8, True, lane=5, max_new=64)
    drv.start([Q])
    for _ in range(3):  # a decode-written history
        drv.spec_step([Q], policy="none")
    b = drv.batch([Q])
    r1 = gen.decode_forward_spec(b, kv_cache=s.pool, enable_trace=True, want_logits=True)

    def moved(batch, frm, to):
        t, p, d, pt = (x.clone() for x in (batch.tokens, batch.positions, batch.draft_tokens, batch.page_table))
        for x in (t, p, d, pt):
            x[to] = x[frm].clone()
        t[frm], p[frm], d[frm], pt[frm] = 0, -1, -1, 0
        return api.SpecDecodeBatch(tokens=t, positions=p, draft_tokens=d, page_table=pt)

    for other in (6, 13, 29):
        r2 = gen.decode_forward_spec(moved(b, 5, other), kv_cache=s.pool, enable_trace=True, want_logits=True)
        ok = bool(torch.equal(r1.logits[5], r2.logits[other])) and r1.argmax[5, 0] == r2.argmax[other, 0]
        ok = ok and r1.mtp_argmax[5, 0] == r2.mtp_argmax[other, 0]
        log(f"R5: lane 5 (row 0) -> lane {other} (row {other // 8}): logits / a / m bitwise {bool(ok)}")
        assert ok, f"lane relocation 5 -> {other} is not bitwise"
    phys = {}
    gen.spec_observer = lambda plan, i, out: phys.__setitem__(i, (plan, {k: (None if v is None else v.clone())
                                                                         for k, v in out.items()}))  # fmt: skip
    gen.observe_logits = True
    try:
        d = int(r1.mtp_argmax[5, 0])
        fill_row0 = [
            s.chat_req(qi, False, lane=lane, max_new=8)
            for qi, lane in zip((10, 11, 12, 13, 14, 15, 16), (0, 1, 2, 3, 4, 6, 7))
        ]
        all_other = fill_row0 + [
            s.chat_req(qi, False, lane=lane, max_new=8)
            for qi, lane in zip(range(17, 32), [l for l in range(8, 32) if l != 5][:15])
        ]
        all_other += [s.chat_req(qi, True, lane=lane, max_new=8)
                      for qi, lane in zip(range(9, 18), [l for l in range(8, 32)][15:])]  # fmt: skip
        assert len({r.lane for r in all_other} | {5}) == 32
        cases = (("cross-row partner", fill_row0), ("same-row partner", []), ("overflow pass", all_other))
        for tag, others in cases:
            drv.start(others)
            Q.draft = d
            bv = drv.batch([Q] + others)
            phys.clear()
            rv = gen.decode_forward_spec(bv, kv_cache=s.pool, enable_trace=True, want_logits=False)
            plan = gen.last_spec
            if tag == "overflow pass":
                assert 5 in plan.overflow, plan.partner_of
                lane_out, k = 5, 1
            else:
                lane_out, k = plan.partner_of[5], 0
                assert (lane_out // 8 != 0) == (tag == "cross-row partner"), (tag, lane_out)
            part = phys[k][1]
            # the ordinary step with d at n + 1 on the owner's lane (Q alone)
            Q.out.append(d)
            ro = gen.decode_forward_spec(drv.batch([Q]), kv_cache=s.pool, enable_trace=True, want_logits=True)
            Q.out.pop()
            ok = bool(torch.equal(part["logits"][lane_out], ro.logits[5]))
            ok = ok and int(part["a"][lane_out]) == int(ro.argmax[5, 0]) == int(rv.argmax[5, 1])
            ok = ok and int(part["m"][lane_out]) == int(ro.mtp_argmax[5, 0]) == int(rv.mtp_argmax[5, 1])
            log(f"R5: draft on the {tag} (lane {lane_out}) == ordinary step at n + 1 on lane 5: bitwise {bool(ok)}")
            assert ok, f"{tag}: the draft row differs from the ordinary step at n + 1"
    finally:
        gen.spec_observer, gen.observe_logits = None, False


def _gs5_case(
    s: SpecSession,
    name: str,
    reqs: Sequence[Req],
    *,
    policy="budget",
    admit=None,
    late=(),
    plain_paths=(("plain", "all"),),
) -> Dict[str, Any]:
    """Greedy plain (``all``; optionally ``row``) vs greedy spec on the same prefilled requests: token-exact."""
    drv = s.drv
    base = {}
    out = {"name": name, "batch": len(reqs) + len(late)}
    for path in plain_paths:
        if late:  # the plain reference of a batch change: every request from the start (rows are independent)
            st = drv.run_plain(list(reqs) + list(late), path=path)
        else:
            st = drv.run_plain(reqs, path=path)
        base[path] = {r.name: list(r.out) for r in list(reqs) + list(late)}
        out[f"plain_{path[1]}"] = dict(steps=st.steps, tok_s=st.tok_s, step_ms=st.median_ms("plain"))
    st = drv.run_spec(reqs, policy=policy, admit=admit)
    spec = {r.name: list(r.out) for r in list(reqs) + list(late)}
    ref = base[plain_paths[0]]
    diverged = {n: first_divergence(ref[n], spec[n]) for n in ref if ref[n] != spec[n]}
    for path in plain_paths[1:]:
        for n in ref:
            if base[path][n] != ref[n]:
                diverged[f"{n}@{path[1]}"] = first_divergence(ref[n], base[path][n])
    out.update(
        spec=dict(steps=st.steps, verify_steps=st.verify_steps, offered=st.offered, accepted=st.accepted,
                  acceptance=st.acceptance, tok_s=st.tok_s, ordinary_ms=st.median_ms("ordinary"),
                  verify_ms=st.median_ms("verify"), gen=st.gen_stats),
        tokens=sum(len(v) for v in spec.values()), diverged=diverged,
    )  # fmt: skip
    g = st.gen_stats
    log(
        f"G-S5 {name}: {len(ref) - len(diverged)}/{len(ref)} token-exact ({out['tokens']} tokens); plain "
        f"{out[f'plain_{plain_paths[0][1]}']['steps']} steps vs spec {st.steps} ({st.verify_steps} verify); acceptance "
        f"{st.accepted}/{st.offered} = {st.acceptance:.3f}; "
        f"packed {g['packed_drafts']} (cross-row {g['cross_row_partners']}), overflow {g['overflow_drafts']} in "
        f"{g['overflow_passes']} passes; tok/s plain {out[f'plain_{plain_paths[0][1]}']['tok_s']:.1f} spec "
        f"{st.tok_s:.1f}" + (f"; divergences {diverged}" if diverged else "")
    )
    return out


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_gs5_lossless(spec_session):
    """G-S5: greedy decoding with speculation is token-identical to greedy decoding without it (the plain ``all``
    trace; plain ``row``, draft 1, too where noted), from the same prefilled state: batch 1 (each of the 4 validation
    prompts x thinking off / on), 8 (the same 8 together), 16 on rows 0-1 only (every row speculating, every partner
    on another DP row), 16 with the bridge's lanes, 24 (capped drafting: 8 idle lanes), 32 (no idle lane: every draft
    declined), 32 with every request drafting (every verify step overflows: two replays), and 16 + 12 admitted
    mid-stream (a batch change: overflow). Token-exact on every request is asserted (lane relocation and row
    independence are bitwise at full depth, R5)."""
    s = spec_session
    results = []
    p8 = [(qi, think) for think in (False, True) for qi in range(4)]
    for i, (qi, think) in enumerate(p8):
        r = s.chat_req(qi, think, lane=lane_of_slot(i))
        results.append(_gs5_case(s, f"b1 {r.name}", [r]))
    results.append(_gs5_case(s, "b8", [s.chat_req(qi, t, lane_of_slot(i)) for i, (qi, t) in enumerate(p8)],
                             plain_paths=(("plain", "all"), ("plain", "row"))))  # fmt: skip
    r16x = [s.chat_req(qi, True, lane) for qi, lane in zip(range(16), range(16))]
    results.append(_gs5_case(s, "b16 rows 0-1 (cross-row partners)", r16x))
    r16 = [s.chat_req(qi, True, lane_of_slot(i)) for i, qi in enumerate(range(16))]
    results.append(_gs5_case(s, "b16", r16))
    r24 = [s.chat_req(qi, True, lane_of_slot(i)) for i, qi in enumerate(range(24))]
    results.append(_gs5_case(s, "b24 (capped)", r24))
    r32 = [s.chat_req(qi, True, lane_of_slot(i)) for i, qi in enumerate(range(32))]
    results.append(_gs5_case(s, "b32 (no idle lane)", r32, plain_paths=(("plain", "all"), ("plain", "row"))))
    r32 = [s.chat_req(qi, True, lane_of_slot(i)) for i, qi in enumerate(range(32))]
    results.append(_gs5_case(s, "b32 forced overflow", r32, policy="all"))
    first16 = [s.chat_req(qi, False, lane_of_slot(i)) for i, qi in enumerate(range(16))]
    late12 = [s.chat_req(qi, False, lane_of_slot(16 + i)) for i, qi in enumerate(range(16, 28))]
    results.append(_gs5_case(s, "b16 + 12 admitted (batch change)", first16, late=late12,
                             admit=lambda step: late12 if step == 4 else []))  # fmt: skip
    s.report["gs5"] = results
    bad = {r["name"]: r["diverged"] for r in results if r["diverged"]}
    total = sum(len(r["diverged"]) for r in results)
    assert sum(r["spec"]["accepted"] for r in results) > 100, "speculation never accepted a draft"
    assert results[-1]["spec"]["gen"]["overflow_passes"] > 0 and results[-2]["spec"]["gen"]["overflow_passes"] > 0
    assert results[9]["spec"]["gen"]["cross_row_partners"] > 0
    assert not bad, f"{total} request streams differ (first divergence per request): {bad}"


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_spec_throughput(spec_session):
    """Throughput (report; the gain is asserted at batch 1 / 8 / 16): greedy tokens per second of a speculating
    launch vs the non-speculating production trace (plain ``all``, KV-R) and draft 1 (plain ``row``), batch 1 / 8 /
    16 / 32, thinking-on prompts (long generations), the bridge's drafting budget, ``MAX_NEW`` tokens per request.
    Every committed token is the plain path's (asserted)."""
    s = spec_session
    drv = s.drv
    rows = []
    s.gen.reset_spec_profile()
    for B in (1, 8, 16, 32):
        sets = [[s.chat_req(qi, True, lane_of_slot(i)) for i, qi in enumerate(range(B))]]
        if B == 1:
            sets = [[s.chat_req(qi, True, lane_of_slot(0))] for qi in range(4)]
        agg = {"plain_all": [0, 0.0], "plain_row": [0, 0.0], "spec": [0, 0.0]}
        acc = [0, 0]
        steps = {"plain": 0, "spec": 0, "verify": 0}
        ms = {"plain_all": [], "plain_row": [], "ordinary": [], "verify": []}
        for reqs in sets:
            for path in (("plain", "all"), ("plain", "row")):
                st = drv.run_plain(reqs, path=path)
                agg[f"plain_{path[1]}"][0] += st.tokens
                agg[f"plain_{path[1]}"][1] += st.seconds
                ms[f"plain_{path[1]}"] += st.step_ms.get("plain", [])
                ref = {r.name: list(r.out) for r in reqs}
                if path[1] == "all":
                    steps["plain"] += st.steps
                    ref_all = ref
            st = drv.run_spec(reqs, policy="budget")
            assert all(r.out == ref_all[r.name] for r in reqs), "spec tokens != plain tokens"
            agg["spec"][0] += st.tokens
            agg["spec"][1] += st.seconds
            acc[0] += st.accepted
            acc[1] += st.offered
            steps["spec"] += st.steps
            steps["verify"] += st.verify_steps
            ms["ordinary"] += st.step_ms.get("ordinary", [])
            ms["verify"] += st.step_ms.get("verify", [])
        prof = s.gen.reset_spec_profile()
        pn = max(prof["steps"], 1)
        tps = {k: v[0] / max(v[1], 1e-9) for k, v in agg.items()}
        med = {k: (statistics.median(v) if v else float("nan")) for k, v in ms.items()}
        row = dict(batch=B, tok_s=tps, speedup_vs_all=tps["spec"] / tps["plain_all"],
                   speedup_vs_row=tps["spec"] / tps["plain_row"], acceptance=acc[0] / max(acc[1], 1),
                   steps=steps, step_ms=med, tokens=agg["spec"][0])  # fmt: skip
        rows.append(row)
        log(
            f"throughput batch {B}: tok/s spec {tps['spec']:.1f} | plain all {tps['plain_all']:.1f} "
            f"(x{row['speedup_vs_all']:.2f}) | plain row {tps['plain_row']:.1f} (x{row['speedup_vs_row']:.2f}); "
            f"acceptance {row['acceptance']:.3f}; steps "
            f"plain {steps['plain']} spec {steps['spec']} ({steps['verify']} verify); step ms plain all "
            f"{med['plain_all']:.1f} row {med['plain_row']:.1f} spec ordinary {med['ordinary']:.1f} verify "
            f"{med['verify']:.1f}; "
            f"spec host ms per step: "
            + ", ".join(f"{k} {prof[k] / pn:.2f}" for k in ("plan", "write", "wait", "read", "result"))
        )
    s.report["throughput"] = rows
    for row in rows:
        if row["batch"] <= 16:
            assert row["speedup_vs_all"] > 1.2, f"batch {row['batch']}: speculation gives x{row['speedup_vs_all']:.2f}"


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_gs4_acceptance_teacher_forced(spec_session):
    """G-S4: the 6 C2 prompts teacher-forced through the T32-spec step on device (first 128 tokens prefilled, then one
    ordinary spec step per position with the prompt's own token as the anchor): the MTP layer runs on the TT main
    model's hidden states and argmax. Acceptance of position ``p`` = the draft ``m0[p]`` (for ``p + 2``) equals the
    main argmax ``a0[p + 1]``; on-policy = positions where ``a0[p]`` is the prompt's next token (what greedy decoding
    feeds back, the plugin's walk). Bar: on-policy acceptance within 2 points of the CPU bf16 reference MTP on the
    same rows (positions >= 128; WP3's goldens; 0.835 over all rows, spec_mtp.md §1.2, when they are missing); WP3's TT
    MTP on the golden hiddens gave 0.8340 over all rows. The all-positions rate is reported (it is not §1.2's quantity:
    off-policy rows feed the MTP layer the TT argmax, not the prompt token)."""
    from models.demos.motif3.reference import golden_stream as gs

    s = spec_session
    if not (GOLD_BF16 / "prompts.json").is_file():
        pytest.skip("C2 prompt set missing")
    prompts = gs.load_prompt_set(GOLD_BF16 / "prompts.json")
    K = 128
    reqs = []
    for i, p in enumerate(prompts):
        r = Req(p.name, list(p.ids[:K]), s.blocks.take(api.cdiv(len(p.ids) + 2, BS)), lane_of_slot(i), -1, 1)
        s.prefill(r)
        r.full = list(p.ids)
        reqs.append(r)
    a0 = {r.name: {} for r in reqs}
    m0 = {r.name: {} for r in reqs}
    t0 = time.time()
    steps = 0
    for p in range(K, max(len(r.full) for r in reqs)):
        live = [r for r in reqs if p < len(r.full)]
        tok = torch.zeros(32, dtype=torch.int32)
        pos = torch.full((32,), -1, dtype=torch.int32)
        pt = torch.zeros(32, WIDTH, dtype=torch.int32)
        for r in live:
            tok[r.lane], pos[r.lane] = r.full[p], p
            k = p // BS + 1
            pt[r.lane, :k] = torch.tensor(r.blocks[:k], dtype=torch.int32)
        b = api.SpecDecodeBatch(tokens=tok, positions=pos, draft_tokens=torch.full((32,), -1, dtype=torch.int32),
                                page_table=pt)  # fmt: skip
        res = s.gen.decode_forward_spec(b, kv_cache=s.pool, enable_trace=True, want_logits=False)
        steps += 1
        for r in live:
            a0[r.name][p] = int(res.argmax[r.lane, 0])
            m0[r.name][p] = int(res.mtp_argmax[r.lane, 0])
    dt = time.time() - t0
    tot = {"all": [0, 0], "on": [0, 0]}
    per = {}
    for r in reqs:
        c = {"all": [0, 0], "on": [0, 0]}
        for p in range(K, len(r.full) - 2):
            hit = int(m0[r.name][p] == a0[r.name][p + 1])
            c["all"][0] += hit
            c["all"][1] += 1
            if a0[r.name][p] == r.full[p + 1]:
                c["on"][0] += hit
                c["on"][1] += 1
        per[r.name] = {k: v[0] / max(v[1], 1) for k, v in c.items()} | {"n_on": c["on"][1]}
        for k in tot:
            tot[k][0] += c[k][0]
            tot[k][1] += c[k][1]
    alpha_all, alpha_on = tot["all"][0] / tot["all"][1], tot["on"][0] / tot["on"][1]
    ref = cpu_acceptance_same_rows(K)
    s.report["gs4"] = dict(all=alpha_all, on_policy=alpha_on, n_all=tot["all"][1], n_on=tot["on"][1], per=per, cpu=ref)
    log(f"G-S4: {steps} teacher-forced spec steps in {dt:.1f} s; acceptance all {alpha_all:.4f} "
        f"({tot['all'][1]} rows), "
        f"on-policy {alpha_on:.4f} ({tot['on'][1]} rows); CPU bf16 reference on the same rows (positions >= {K}): "
        f"{ref and round(ref['on'], 4)} on-policy / {ref and round(ref['all'], 4)} all; CPU over all rows 0.8365 / "
        f"0.7599 (spec_mtp.md §1.2: 0.835); WP3 TT on golden hiddens 0.8340; per prompt "
        f"{ {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in per.items()} }")  # fmt: skip
    bar = ref["on"] if ref is not None else 0.835
    assert abs(alpha_on - bar) <= 0.02, f"on-policy acceptance {alpha_on:.4f} is not within 2 points of {bar:.4f}"


MTP_GOLDENS = PROJECT_ROOT / "tt_cache" / "test" / "mtp" / "mtp_goldens_c2.pt"  # built by test_mtp.py's cpu tests (WP3)


def cpu_acceptance_same_rows(first_pos: int) -> Optional[Dict[str, float]]:
    """The CPU bf16 reference MTP's acceptance (spec_mtp.md §1.2 definition: ``m[p] == argmax_main[p + 1]``;
    on-policy = ``t_{p+1} == argmax_main[p]``) on the C2 rows with position ``>= first_pos``, from WP3's goldens
    (``m16``, ``target``, ``on_policy`` per prompt row range); ``None`` when the file is missing."""
    if not MTP_GOLDENS.is_file():
        return None
    g = torch.load(MTP_GOLDENS, weights_only=False)
    m, tgt, onp = g["m16"], g["target"], g["on_policy"].bool()
    c = {"all": [0, 0], "on": [0, 0]}
    for _, (a, n) in g["rows"].items():
        idx = torch.arange(a, a + n)
        sel = (idx - a) >= int(first_pos)
        hit = m[idx] == tgt[idx]
        for kind, mask in (("all", sel), ("on", sel & onp[idx])):
            c[kind][0] += int(hit[mask].sum())
            c[kind][1] += int(mask.sum())
    return {k: v[0] / max(v[1], 1) for k, v in c.items()} | {"n_on": c["on"][1], "n_all": c["all"][1]}


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_cp9_spec_serving_mix(spec_session):
    """CP9 with speculation: after the capture, ``MOTIF3_CP9_ROUNDS`` (10) rounds of a randomized serving mix -- cold
    rows, prefix hits, vLLM chunked continuations (unaligned starts), preemption resumes (prompt + generated tokens)
    -- each followed by 2-5 speculative steps with a random drafting policy (ordinary, budgeted verify, every request
    drafting: overflow passes). The program cache stays constant and, at the end, a verify step with more drafts than
    idle lanes (overflow) replays bitwise equal to the eager step."""
    from models.demos.motif3.reference import golden_stream as gs

    s = spec_session
    gen, drv = s.gen, s.drv
    rng = random.Random(17)
    src = [t for p in gs.load_prompt_set(GOLD_BF16 / "prompts.json") for t in p.ids]

    def text(n):
        i = rng.randrange(0, len(src) - 1)
        return [src[(i + j) % len(src)] for j in range(n)]

    pc0 = s.programs()
    live: List[Req] = []
    done: List[Req] = []
    pending: List[Tuple[Req, int]] = []
    free = list(range(32))
    rng.shuffle(free)
    kinds_seen = set()
    g0 = dict(gen.stats)

    def retire(rs):
        for r in rs:
            live.remove(r)
            done.append(r)
            free.insert(0, r.lane)

    for rnd in range(CP9_ROUNDS):
        burst = rnd % 15 == 7  # every 15th round: up to 20 short cold rows, then every row drafts (overflow passes)
        for _ in range(20 if burst else rng.randint(1, 3)):
            if not free:
                break
            lane = free.pop()
            kind = "burst" if burst else rng.choice(("cold", "hit", "chunk", "resume"))
            if kind == "chunk" and pending:
                r, st0 = pending.pop(0)
                r.lane = lane
                s.prefill(r, start=st0)
            elif kind == "hit" and done:
                base = rng.choice(done)
                k = rng.randint(1, max(1, (base.S - 1) // BS))
                ids = base.prompt[: k * BS] + text(rng.randint(1, 500))
                blk = base.blocks[:k] + s.blocks.take(api.cdiv(len(ids) + 40, BS) - k)
                r = Req(f"hit{rnd}", ids, blk, lane, -1, 24)
                s.prefill(r, start=k * BS)
            elif kind == "resume" and done:
                base = rng.choice(done)
                ids = (base.prompt + base.out)[:3000]  # bounded (a long soak's pool): still a hit on base's blocks
                k = rng.choice([0, max(0, (len(ids) - 1) // BS)])
                blk = base.blocks[:k] + s.blocks.take(api.cdiv(len(ids) + 40, BS) - k)
                r = Req(f"resume{rnd}", ids, blk, lane, -1, 24)
                s.prefill(r, start=k * BS)
            elif kind == "burst":
                ids = text(rng.randint(20, 200))
                r = Req(f"burst{rnd}", ids, s.blocks.take(api.cdiv(len(ids) + 40, BS)), lane, -1, 24)
                s.prefill(r)
            else:
                kind = "cold"
                ids = text(rng.randint(20, 1500))
                r = Req(f"cold{rnd}", ids, s.blocks.take(api.cdiv(len(ids) + 40, BS)), lane, -1, 24)
                if len(ids) > 400 and rng.random() < 0.5:  # vLLM-chunked: the first chunk now
                    e = rng.randint(130, len(ids) - 1)
                    s.prefill(r, end=e)
                    pending.append((r, e))
                    free.insert(0, lane)
                    kinds_seen.add(kind)
                    continue
                s.prefill(r)
            kinds_seen.add(kind)
            drv.start([r])
            live.append(r)
        for _ in range(rng.randint(2, 5)):
            retire([r for r in live if r.done])  # finished requests leave the batch and free their lanes
            if not live:
                break
            policy = "all" if burst else rng.choice(("budget", "budget", "all", "none"))
            drv.spec_step(live, policy=policy)
        retire([r for r in live if r.done or rng.random() < 0.25])
        log(f"CP9-spec round {rnd}: {len(live)} live, program cache {s.programs()}, gen "
            f"{gen.stats['verify_steps'] - g0['verify_steps']} "
            f"verify / {gen.stats['overflow_passes'] - g0['overflow_passes']} overflow passes so far")  # fmt: skip
    assert s.programs() == pc0, f"program cache grew after the capture: {pc0} -> {s.programs()}"
    live = [r for r in live if not r.done]
    while len(live) < 24 and free:  # a verify step with more drafts than idle lanes: the overflow pass
        ids = text(rng.randint(20, 300))
        r = Req(f"fill{len(live)}", ids, s.blocks.take(api.cdiv(len(ids) + 40, BS)), free.pop(), -1, 24)
        s.prefill(r)
        drv.start([r])
        live.append(r)
    for r in live:
        r.draft = r.nxt if r.nxt is not None else r.out[-1]
    _replay_vs_eager(s, live, "CP9-spec final verify step (overflow)")
    drv.spec_step(live, policy="budget")
    assert s.programs() == pc0
    log(
        f"CP9-spec: kinds {sorted(kinds_seen)}; program cache constant at {pc0}; gen stats delta "
        f"{ {k: gen.stats[k] - g0[k] for k in ('spec_steps', 'verify_steps', 'drafts', 'overflow_passes')} }"
    )
    assert gen.stats["verify_steps"] > g0["verify_steps"] and gen.stats["overflow_passes"] > g0["overflow_passes"]
    assert {"cold", "hit", "chunk", "resume"} <= kinds_seen, f"the mix missed row kinds: {sorted(kinds_seen)}"


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_g13b_kvr_full_model(spec_session):
    """G13b (KV-R at full depth; last: it releases the traces for the read-back): traced decode steps of 32 lanes at
    1K and 8K context through the plain ``row`` (draft 1), plain ``all`` (KV-R), plain ``all_split`` and the spec
    trace (``all_split`` + MTP; ordinary steps, and verify steps of 16 owners + 16 partners), 20 steps each with
    advancing positions: step costs (end to end and replay + sync; medians of ``G13B_REPLAYS`` replays), the KV-R write
    cost (design gate: plain ``all_split`` - ``row`` <= ``KVR_GATE_MS`` = 2.0 ms per step, + ``SAFE_GATHER_MARGIN_MS``
    (0.45) under ``ring_gather="safe"``, B0's default, whose rerouted decode gathers shift the difference by ~+0.1 ms
    to 1.98-2.05 ms; plain ``all`` - ``row``, the non-speculating production decode with prefix caching, is reported
    against the same bar and guarded at ``KVR_GUARD_MS`` (+ the same margin): it measured 1.96-2.03 ms at native
    gathers), trace == eager on the spec path at 8K, the trace region use; then, with
    every trace released, the KV-R invariant: every chip's copy of every slot the KV-R paths decode-wrote (layers 0, 1,
    2, the last and the MTP layer) is bitwise equal."""
    import ttnn

    from models.demos.motif3.tt.model import device_bytes_per_chip

    s = spec_session
    gen = s.gen
    STEPS = 20
    paths = [
        ("plain", "row"),
        ("plain", "all"),
        ("plain", "all_split"),
        ("spec", "all_split"),
        ("spec_verify", "all_split"),
    ]
    kvr_pool = s.blocks.take(32 * 4 * 2)  # the KV-R paths' own blocks, contiguous: [lo, hi) for the read-back
    kvr_blocks = list(kvr_pool)
    res = {}
    for ctx in (1024, 8192):
        hist = s.blocks.take(ctx // BS)
        for key in paths:
            if key[1] == "row":
                own = s.blocks.take(32)
            else:
                own, kvr_pool = kvr_pool[:32], kvr_pool[32:]
            lanes = list(range(32))
            owners = lanes if key[0] != "spec_verify" else [lane_of_slot(i) for i in range(16)]
            ms = []
            for i in range(STEPS + 3):
                tok = torch.zeros(32, dtype=torch.int32)
                pos = torch.full((32,), -1, dtype=torch.int32)
                dr = torch.full((32,), -1, dtype=torch.int32)
                pt = torch.zeros(32, WIDTH, dtype=torch.int32)
                for j, lane in enumerate(owners):
                    tok[lane], pos[lane] = 1000 + j + i, ctx + 2 * i
                    pt[lane, : ctx // BS] = torch.tensor(hist, dtype=torch.int32)
                    pt[lane, ctx // BS] = own[j]
                    if key[0] == "spec_verify":
                        dr[lane] = 2000 + j
                t1 = time.perf_counter()
                if key[0] == "plain":
                    gen.decode_forward(api.DecodeBatch(tokens=tok, positions=pos, page_table=pt), kv_cache=s.pool,
                                       enable_trace=True, path=key)  # fmt: skip
                else:
                    b = api.SpecDecodeBatch(tokens=tok, positions=pos, draft_tokens=dr, page_table=pt)
                    gen.decode_forward_spec(b, kv_cache=s.pool, enable_trace=True, want_logits=key[0] == "spec")
                if i >= 3:
                    ms.append((time.perf_counter() - t1) * 1e3)
            p = gen._paths[("spec", "all_split") if key[0].startswith("spec") else key]
            rs = []
            for _ in range(G13B_REPLAYS):
                t1 = time.perf_counter()
                ttnn.execute_trace(s.mesh, p.trace_id, cq_id=0, blocking=True)
                rs.append((time.perf_counter() - t1) * 1e3)
            res[(ctx, key)] = (statistics.median(ms), statistics.median(rs))
            if ctx == 8192 and key[0] == "spec_verify":  # trace == eager, a verify step with cross-row partners
                b = api.SpecDecodeBatch(tokens=tok, positions=pos, draft_tokens=dr, page_table=pt)
                rt = gen.decode_forward_spec(b, kv_cache=s.pool, enable_trace=True, want_logits=False)
                re_ = gen.decode_forward_spec(b, kv_cache=s.pool, enable_trace=False, want_logits=False)
                assert _same_result(rt, re_, owners, owners), "spec trace replay != eager at 8K"
        row_dev = res[(ctx, ("plain", "row"))][1]
        log(f"G13b ctx {ctx}: " + "; ".join(
            f"{k[0]}/{k[1]} {v[0]:.2f} ms end-to-end, {v[1]:.2f} ms replay+sync (+{v[1] - row_dev:.2f})"
            for (c, k), v in res.items() if c == ctx))  # fmt: skip
    trace = device_bytes_per_chip(s.mesh, ttnn.BufferType.TRACE) or {}
    log(f"G13b trace region: {trace.get('allocated', 0) / 2**20:.1f} of {trace.get('total', 0) / 2**20:.0f} MiB for "
        f"{len(gen.decode_paths())} traces")  # fmt: skip
    kvr_cost = {ctx: res[(ctx, ("plain", "all_split"))][1] - res[(ctx, ("plain", "row"))][1] for ctx in (1024, 8192)}
    kvr_all = {ctx: res[(ctx, ("plain", "all"))][1] - res[(ctx, ("plain", "row"))][1] for ctx in (1024, 8192)}
    s.report["g13b"] = dict(steps={f"{c}/{k[0]}/{k[1]}": v for (c, k), v in res.items()}, kvr_cost_ms=kvr_cost,
                            kvr_plain_all_ms=kvr_all)  # fmt: skip
    # ---- the KV-R invariant (every trace released first: the read-back slices compile programs) ----
    gen.release_traces()
    lo, hi = min(kvr_blocks), max(kvr_blocks) + 1
    assert sorted(kvr_blocks) == list(range(lo, hi)), "the KV-R blocks are not contiguous"
    caches = {"L0": s.pool[0], "L1": s.pool[1], "L2": s.pool[2], f"L{gen.num_layers - 1}": s.pool[gen.num_layers - 1],
              "MTP": s.pool.mtp}  # fmt: skip
    bad = {}
    for name, cache in caches.items():
        sl = ttnn.slice(cache, [lo, 0, 0, 0], [hi, 1, BS, api.KV_LATENT_DIM])
        shards = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(sl)]
        ttnn.deallocate(sl)
        diff = [i for i, x in enumerate(shards) if not torch.equal(x, shards[0])]
        written = int((shards[0].abs().sum(dim=(1, 3)) > 0).sum())
        log(f"G13b KV-R invariant {name}: {len(shards)} chip copies of blocks [{lo}, {hi}): {written} written rows, "
            f"{len(diff)} chips differ")  # fmt: skip
        if diff or written == 0:
            bad[name] = (diff, written)
    assert not bad, f"KV-R invariant broken: {bad}"
    margin = SAFE_GATHER_MARGIN_MS if gen._ring_gather() == "safe" else 0.0
    gate, guard = KVR_GATE_MS + margin, KVR_GUARD_MS + margin
    for ctx, c in kvr_cost.items():
        a = kvr_all[ctx]
        log(f"G13b KV-R write cost (device) at {ctx}: plain all_split - row {c:.2f} ms (gate {gate:.2f} ms asserted = "
            f"the design's {KVR_GATE_MS} + {margin} for ring_gather={gen._ring_gather()!r}; the design's bar "
            f"{'met' if c <= KVR_GATE_MS else 'NOT met'}); plain all - row {a:.2f} ms (the non-speculating production "
            f"decode with prefix caching; reported, regression guard {guard:.2f} ms)")  # fmt: skip
    assert all(c <= gate for c in kvr_cost.values()), f"KV-R costs {kvr_cost} ms per step (> {gate:.2f}: deferred KV-R)"
    assert all(a <= guard for a in kvr_all.values()), (
        f"plain all KV-R costs {kvr_all} ms per step (regression guard {guard:.2f} ms; 1.96-2.03 ms measured "
        "2026-10-02 at native gathers)"
    )


# ======================================================================================================================
# device: T64 -- spec_verify="auto": the T32-spec trace + the 64-row T64 trace (docs/p5_t64/P5_T64_DESIGN.md §2.2-§2.4,
# §4, §6.2 G16 / G-S5w / G-S6w-lite). One boot (production settings + packed prefill + spec_verify="auto"), the plain
# "all" trace for the non-speculative reference and, with MOTIF3_T64_ROW_SPLIT (default on), the row_split T32 / T64
# traces next to them (5 traces). Run:
#
#   scripts/devrun.sh -t 5400 -n i2_t64 -- env OMP_WAIT_POLICY=PASSIVE python -m pytest \
#       models/demos/motif3/tests/test_spec_decode_device.py -k t64 -s -p no:cacheprovider --timeout=0
# ======================================================================================================================
T64_ROW_SPLIT = os.environ.get("MOTIF3_T64_ROW_SPLIT", "1") != "0"
T64_MAX_NEW = int(os.environ.get("MOTIF3_T64_MAX_NEW", "256"))  # G-S5w: 256 tokens per request
G16_STEPS = int(os.environ.get("MOTIF3_G16_STEPS", "20"))  # timed steps / replays per (context, path), after 3 warm
G16_CONTEXTS = tuple(int(c) for c in os.environ.get("MOTIF3_G16_CONTEXTS", "1024,8192,32000").split(","))
G16_BAR, G16_KILL = 1.20, 1.30  # T64 / T32 device step at 1K-8K: pass <= 1.20, kill > 1.30 (design §6.2)
G16_OPTION_A = os.environ.get("MOTIF3_G16_OPTION_A", "1") != "0"  # the last test re-captures T64 with option A
T64_TRACE_BAR_MIB = 8.0  # G16: the T64 trace per DRAM bank of the trace region
GS6W_PREFILLS = int(os.environ.get("MOTIF3_GS6W_PREFILLS", "100"))
ROLLBACK_STEPS = 32  # G-S5w (iv): verify steps with wrong drafts on half the lanes
SPEC_ALL, WIDE_ALL = ("spec", "all_split"), ("wide", "all_split")
SPEC_ROW, WIDE_ROW = ("spec", "row_split"), ("wide", "row_split")


def _trace_bytes(mesh) -> int:
    import ttnn

    from models.demos.motif3.tt.model import device_bytes_per_chip

    return int((device_bytes_per_chip(mesh, ttnn.BufferType.TRACE) or {}).get("allocated", 0))


def _trace_banks(mesh) -> int:
    import ttnn

    try:
        return max(1, int(ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE).num_banks))
    except Exception:
        return 1


class T64Session(SpecSession):
    """The ``spec_verify="auto"`` session: :class:`SpecSession` plus the C2 prompts as generation prompts."""

    def c2_req(self, i: int, lane: int, max_new: int) -> Req:
        """C2 prompt ``i`` (goldens/c2/prompts.json; its first 600 ids, a generation prompt) prefilled once."""
        from models.demos.motif3.reference import golden_stream as gs

        key = ("c2", int(i))
        if key not in self._chat:
            p = gs.load_prompt_set(GOLD_BF16 / "prompts.json")[i]
            self._chat[key] = self.new_req(f"c2{p.name}", list(p.ids)[:600], lane, self.cap)
        r = self._chat[key]
        assert max_new <= self.cap
        r.lane, r.max_new = int(lane), int(max_new)
        return r

    def copy_req(self, r: Req, name: str) -> Req:
        """The same prompt prefilled again into fresh blocks (bitwise the same prompt KV: prefill is deterministic)."""
        c = Req(name, list(r.prompt), self.blocks.take(api.cdiv(len(r.prompt) + self.cap + 2, BS)), r.lane, -1,
                r.max_new)  # fmt: skip
        self.prefill(c)
        assert c.first == r.first, f"{name}: a re-prefill of {r.name}'s prompt gave another first token"
        return c

    def sets(self, max_new: int) -> Dict[str, List[Req]]:
        """G-S5w's two 32-request sets: the C2 prompts + chat prompts with thinking on, and 32 chat prompts with
        thinking off (the bridge's lanes)."""
        from models.demos.motif3.reference import golden_stream as gs

        n_c2 = min(len(gs.load_prompt_set(GOLD_BF16 / "prompts.json")), 8)
        think = [self.c2_req(i, lane_of_slot(i), max_new) for i in range(n_c2)]
        think += [self.chat_req(qi, True, lane_of_slot(n_c2 + j), max_new) for j, qi in enumerate(range(32 - n_c2))]
        nothink = [self.chat_req(qi, False, lane_of_slot(i), max_new) for i, qi in enumerate(range(32))]
        return {"c2+chat think": think, "chat nothink": nothink}


@pytest.fixture(scope="module")
def t64_session():
    import ttnn

    from models.demos.motif3.tests.test_resumed_prefill import RefuseReads
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.model import device_bytes_per_chip
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR, open_motif_mesh

    if os.environ.get("OMP_WAIT_POLICY", "").upper() != "PASSIVE":
        log("WARNING: OMP_WAIT_POLICY is not PASSIVE (the serving setting): host timings will be pessimistic")
    _claim_device("t64")
    mesh = open_motif_mesh()
    holder: Dict[str, Any] = {}
    close = _session_closer("t64", mesh, holder)
    try:
        log_fabric(mesh, "spec_decode_t64")
        A = api.DEFAULT_PREFILL_ALIGNMENT
        budget = PP.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, A)
        settings = api.GeneratorSettings(
            max_batch_size=api.NUM_LANES, max_seq_len=MAX_LEN, num_layers=SPEC_LAYERS, kv_cache_dtype="bfp8",
            weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=BS, weights_source="TT cache only (test)",
            chunked_prefill=True, prefix_caching=True, max_num_batched_tokens=budget,
            long_prefill_token_threshold=budget, spec_tokens=1, spec_verify="auto", packed_prefill=True,
        )  # fmt: skip
        guard = RefuseReads(DEFAULT_WEIGHTS_DIR)
        t0 = time.time()
        gen = holder["gen"] = MotifGenerator.create(hf_config=None, mesh_device=mesh, settings=settings, source=guard)
        log(f"T64 generator: {gen.num_layers} layers + MTP in {time.time() - t0:.1f} s; {gen.describe_spec_verify()}")
        assert not gen.model.cache_misses and not guard.refused, "not a TT-cache-only boot"
        assert gen.serving_paths == [SPEC_ALL, WIDE_ALL] and gen.cfg.ring_gather == "safe" and gen.packed_prefill
        gen.extra_decode_paths = [("plain", "all")] + ([SPEC_ROW, WIDE_ROW] if T64_ROW_SPLIT else [])
        pool = gen.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BS, num_layers=SPEC_LAYERS)
        t0 = time.time()
        gen.warmup_prefill(kv_cache=pool, enable_trace=False)
        t_wp = time.time() - t0
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=WIDTH)
        sizes: Dict[Tuple[str, str], int] = {}
        orig = gen._capture_path

        def capture(p, pool_):  # the trace region each capture takes
            before = _trace_bytes(mesh)
            out = orig(p, pool_)
            sizes[p.key] = _trace_bytes(mesh) - before
            return out

        gen._capture_path = capture
        try:
            gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=WIDTH)
        finally:
            del gen._capture_path
        trace = device_bytes_per_chip(mesh, ttnn.BufferType.TRACE) or {}
        banks = _trace_banks(mesh)
        warm = {k: round(v, 2) for k, v in gen.timings.items() if k.startswith("warmup_decode_")}
        per_trace = {k: round(v / banks / 2**20, 2) for k, v in sizes.items()}
        log(
            f"T64 warmup: prefill {t_wp:.1f} s ({len(gen.warmed_shapes)} shapes), decode eager {warm}, capture "
            f"{gen.timings.get('capture_decode_s', 0):.1f} s; trace region {trace.get('allocated', 0) / 2**20:.1f} of "
            f"{trace.get('total', 0) / 2**20:.0f} MiB ({banks} banks); per trace MiB per bank {per_trace}; program "
            f"cache {mesh.num_program_cache_entries()}"
        )
        s = T64Session(mesh, gen, pool, guard, cap=max(T64_MAX_NEW, MAX_NEW, 64) + 4)
        s.report["trace_bytes"] = dict(sizes)
        s.report["trace_banks"] = banks
        s.report["rollback"] = []
        yield s
    finally:
        _OPEN_SESSIONS.pop("t64", None)
        close()


def _t64_batch(n: int, hist: Sequence[int], own: Sequence[int], *, drafts: bool) -> api.SpecDecodeBatch:
    """32 active lanes at position ``n`` (``n % 64 < 62``), the shared history blocks ``hist`` for ``[0, len(hist) *
    64)`` and one own block per lane after them; a draft on every lane (``drafts``) or none."""
    tok = torch.zeros(32, dtype=torch.int32)
    pos = torch.full((32,), -1, dtype=torch.int32)
    dr = torch.full((32,), -1, dtype=torch.int32)
    pt = torch.zeros(32, WIDTH, dtype=torch.int32)
    k = len(hist)
    assert n // BS == k and (n + 1) // BS == k, (n, k)
    for lane in range(32):
        tok[lane], pos[lane] = 1000 + 37 * lane % 5000, n
        pt[lane, :k] = torch.tensor(list(hist), dtype=torch.int32)
        pt[lane, k] = int(own[lane])
        if drafts:
            dr[lane] = 2000 + 11 * lane % 3000
    return api.SpecDecodeBatch(tokens=tok, positions=pos, draft_tokens=dr, page_table=pt)


def _time_path(s: SpecSession, key, hist, own, n0: int, *, drafts: bool) -> Tuple[float, float]:
    """``(end-to-end ms, device ms)`` medians of ``key``'s traced step: ``G16_STEPS`` ``decode_forward_spec`` calls
    (positions advancing by 2) after 3 warm ones, then ``G16_STEPS`` blocking replays of the trace (replay + sync)."""
    import ttnn

    e2e = []
    for i in range(G16_STEPS + 3):
        b = _t64_batch(n0 + 2 * min(i, 20), hist, own, drafts=drafts)
        if i == 3:
            s.gen.reset_spec_profile()
        t1 = time.perf_counter()
        s.gen.decode_forward_spec(b, kv_cache=s.pool, enable_trace=True, want_logits=False, path=key)
        if i >= 3:
            e2e.append((time.perf_counter() - t1) * 1e3)
    assert s.gen.last_verify_kind == key[0]
    prof = s.gen.reset_spec_profile()
    k = max(prof["steps"], 1)
    stages = ("plan", "write", "enqueue", "wait", "read", "result")
    log(f"  {key} host ms per step: " + ", ".join(f"{q} {prof[q] / k:.2f}" for q in stages))
    p = s.gen._paths[key]
    rs = []
    for _ in range(G16_STEPS):
        t1 = time.perf_counter()
        ttnn.execute_trace(s.mesh, p.trace_id, cq_id=0, blocking=True)
        rs.append((time.perf_counter() - t1) * 1e3)
    return statistics.median(e2e), statistics.median(rs)


def _g16_blocks(s: SpecSession) -> Tuple[List[int], List[int]]:
    """G16's blocks, shared by every context and path (timing only: the history's content does not matter): the
    history of the longest context and one own block per lane."""
    if "g16_blocks" not in s.report:
        s.report["g16_blocks"] = (s.blocks.take(max(G16_CONTEXTS) // BS), s.blocks.take(32))
    return s.report["g16_blocks"]


def _g16_cost_table(s: SpecSession, modes: Sequence[str], tag: str) -> Dict[Tuple[int, str], Dict[str, float]]:
    """T32 (all 32 lanes, no drafts) vs T64 (all 32 lanes, a draft on every lane) per context and KV-write mode."""
    res = {}
    hist_all, own = _g16_blocks(s)
    for ctx in G16_CONTEXTS:
        hist = hist_all[: ctx // BS]
        for mode in modes:
            t32 = _time_path(s, ("spec", mode), hist, own, ctx, drafts=False)
            t64 = _time_path(s, ("wide", mode), hist, own, ctx, drafts=True)
            row = dict(t32_e2e=t32[0], t32_dev=t32[1], t64_e2e=t64[0], t64_dev=t64[1], ratio=t64[1] / t32[1],
                       ratio_e2e=t64[0] / t32[0])  # fmt: skip
            res[(ctx, mode)] = row
            log(f"G16 {tag} ctx {ctx} {mode}: T32 {t32[1]:.2f} ms device / {t32[0]:.2f} ms end-to-end, T64 "
                f"{t64[1]:.2f} / {t64[0]:.2f} ms: T64 / T32 = {row['ratio']:.3f} (device), {row['ratio_e2e']:.3f} "
                f"(end-to-end)")  # fmt: skip
    return res


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_t64_g16_step_cost(t64_session):
    """G16 (design §6.2): the traced T64 step (all 32 lanes, a draft on every lane; A'') vs the traced T32-spec step
    (all 32 lanes, no drafts) at 1K / 8K / 32K context, ``all_split`` (and ``row_split``): medians of ``G16_STEPS``
    replays after 3 warm steps. Pass: T64 / T32 <= 1.20 at 1K-8K (kill > 1.30); the T64 trace <= 8 MiB per bank of the
    trace region. Then, right after a T64 replay, eager prefills sp0 8192, sp1 8192 and a packed pk0 pass of T = 8192
    (16 segments of S = 512): every logits row sane (no stale tile, no static-CB clash), the program cache constant
    throughout. Option A (one B = 16 FlashMLA call on the global layers) is measured by the last test of the session."""
    import ttnn

    s = t64_session
    gen = s.gen
    pc0 = s.programs()
    modes = ["all_split"] + (["row_split"] if T64_ROW_SPLIT else [])
    res = _g16_cost_table(s, modes, "A''")
    s.report["g16"] = {f"{c}/{m}": v for (c, m), v in res.items()}
    banks = s.report["trace_banks"]
    t64_mib = {k: v / banks / 2**20 for k, v in s.report["trace_bytes"].items() if k[0] == "wide"}
    log(f"G16 trace region per bank: T64 {t64_mib} MiB (bar {T64_TRACE_BAR_MIB}); every trace "
        f"{ {k: round(v / banks / 2**20, 2) for k, v in s.report['trace_bytes'].items()} } MiB")  # fmt: skip
    # prefills right after a T64 replay (P5 in: a pk0 pass at T = 8192)
    ttnn.execute_trace(s.mesh, gen._paths[WIDE_ALL].trace_id, cq_id=0, blocking=True)
    src = []
    while len(src) < 9100:
        src += s.chat_ids(len(src) % len(QUESTIONS), True)[1:]
    a = Req("g16_sp0", src[:8000], s.blocks.take(api.cdiv(8000 + 8, BS)), 0, -1, 8)
    s.prefill(a)
    assert [(c.path, c.bucket) for c in gen.last_prefill.jobs[0].plan.chunks] == [("sp0", 8192)]
    b = Req("g16_sp1", src[:9000], s.blocks.take(api.cdiv(9000 + 8, BS)), 1, -1, 8)
    s.prefill(b, end=1024)
    s.prefill(b, start=1024)
    assert [(c.path, c.bucket) for c in gen.last_prefill.jobs[0].plan.chunks] == [("sp1", 8192)]
    rows = []
    for i in range(16):
        ids = src[200 + 450 * i : 200 + 450 * (i + 1)]
        blk = s.blocks.take(api.cdiv(len(ids), BS))
        pt = torch.zeros(WIDTH, dtype=torch.int32)
        pt[: len(blk)] = torch.tensor(blk, dtype=torch.int32)
        rows.append(api.PrefillRequest(lane=i, tokens=torch.tensor(ids, dtype=torch.int32), page_table=pt))
    from models.demos.motif3.tests.test_resumed_prefill import assert_sane

    lg = gen.prefill_forward_batch(rows, kv_cache=s.pool)
    assert_sane(lg, "G16 pk0 T = 8192 pass after a T64 replay")
    shapes = [p.shape for p in gen.last_prefill.passes]
    log(f"G16 prefills after a T64 replay: sp0 8192, sp1 8192, packed {shapes}: sane; program cache {pc0} -> "
        f"{s.programs()}")  # fmt: skip
    assert shapes == [("pk0", 8192, 512)], shapes
    assert s.programs() == pc0, f"a program compiled after the capture: {pc0} -> {s.programs()}"
    if SPEC_LAYERS < 53:
        log(f"G16: a {SPEC_LAYERS}-layer plumbing run: the bars (53 layers) are not asserted")
        return
    for (ctx, mode), row in res.items():
        if ctx <= 8192:
            assert row["ratio"] <= G16_KILL, f"G16 KILL: T64 / T32 = {row['ratio']:.3f} > {G16_KILL} at {ctx} {mode}"
            assert row["ratio"] <= G16_BAR, f"G16: T64 / T32 = {row['ratio']:.3f} > {G16_BAR} at {ctx} {mode}"
    for key, mib in t64_mib.items():
        assert 0 < mib <= T64_TRACE_BAR_MIB, f"the T64 trace {key}: {mib:.2f} MiB per bank (bar {T64_TRACE_BAR_MIB})"


def _gs5w_reference(s: SpecSession, reqs: Sequence[Req]) -> Dict[str, List[int]]:
    s.drv.run_plain(reqs, path=("plain", "all"))
    return {r.name: list(r.out) for r in reqs}


def _assert_same_tokens(tag: str, reqs: Sequence[Req], ref: Dict[str, List[int]]) -> None:
    bad = {r.name: first_divergence(ref[r.name], r.out) for r in reqs if r.out != ref[r.name]}
    assert not bad, f"{tag}: {len(bad)} of {len(reqs)} request streams differ (first divergence per request): {bad}"


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_t64_gs5w_lossless(t64_session):
    """G-S5w (design §6.2; ``router_logits`` of the launch: "composite", and with ``MOTIF3_ROUTER_LOGITS=exact_fp32``
    the R-E7 variant, which ``create`` accepts once ``model_config.ROUTER_EXACT_FP32_DECODE_ROWS`` lists the T64 step's
    64 rows): greedy decode of 32 concurrent prompts, 256 tokens each, two sets (the C2 prompts + chat prompts with
    thinking on; 32 chat prompts with thinking off):

    (i) ``auto`` + A'' with the bridge's drafting rule (every live lane drafts from ``c*`` live lanes on) == the
        non-speculative decode of the same session (ordinary T32-spec steps) == the plain-path trace, token for token;
    (ii) every step on the T64 trace: without drafts (draft rows idle: the non-speculative T64 decode) and with every
         lane drafting == the same tokens;
    (iii) draft-row relocation: a token at ``n + 1`` as a T64 draft row, as the next T32-spec step's anchor and as the
          next T64 step's anchor: the final-norm hidden state and the logits rows bitwise equal, ``a`` / ``m`` equal;
    (iv) rollback: the 32 requests re-prefilled into fresh blocks, every lane drafting for ``ROLLBACK_STEPS`` verify
         steps with WRONG drafts on half the lanes (all_split; row_split too with MOTIF3_T64_ROW_SPLIT): tokens equal
         the non-speculative run; the cache rows are compared by ``test_t64_release_readback`` (main and MTP caches,
         every chip);
    (v) per-request accept / reject sequence at c = 32 (T64) == the same requests at c = 8 (T32 packed verify, every
        step drafting)."""
    s = t64_session
    drv = s.drv
    pc0 = s.programs()
    sets = s.sets(T64_MAX_NEW)
    seqs32 = {}
    for name, reqs in sets.items():
        ref = _gs5w_reference(s, reqs)
        st0 = drv.run_spec(reqs, policy="none")  # (i) non-speculative: ordinary T32-spec steps
        _assert_same_tokens(f"G-S5w {name} non-spec T32-spec vs plain", reqs, ref)
        st = drv.run_spec(reqs, policy="bridge")  # (i) auto + A''
        _assert_same_tokens(f"G-S5w (i) {name} auto", reqs, ref)
        g = st.gen_stats
        log(f"G-S5w (i) {name}: token-exact; {st.steps} steps ({st.verify_steps} verify: {g['wide_steps']} T64, "
            f"{g['auto_t32_verifies']} T32), acceptance {st.accepted}/{st.offered} = {st.acceptance:.3f}, tok/s "
            f"{st.tok_s:.1f}; T64 verify step {st.median_ms('verify_t64'):.1f} ms, T32 ordinary "
            f"{st.median_ms('ordinary'):.1f} ms")  # fmt: skip
        assert g["wide_steps"] > 0 and g["overflow_passes"] == 0, g
        st_n = drv.run_spec(reqs, policy="none", path=WIDE_ALL, want_logits=False)  # (ii) T64, draft rows idle
        _assert_same_tokens(f"G-S5w (ii) {name} T64 non-spec", reqs, ref)
        st_a = drv.run_spec(reqs, policy="all", path=WIDE_ALL, want_logits=False)  # (ii) T64, every lane drafting
        _assert_same_tokens(f"G-S5w (ii) {name} T64 all drafting", reqs, ref)
        seqs32[name] = {r.name: list(r.verdicts) for r in reqs}
        log(f"G-S5w (ii) {name}: T64 non-spec {st_n.steps} steps, T64 drafting {st_a.steps} steps (acceptance "
            f"{st_a.acceptance:.3f}): token-exact")  # fmt: skip
        s.report.setdefault("gs5w", {})[name] = dict(auto_steps=st.steps, acceptance=st.acceptance, tok_s=st.tok_s,
                                                     t64_steps=g["wide_steps"], nonspec_tok_s=st0.tok_s)  # fmt: skip
        log(f"G-S5w {name}: tok/s auto {st.tok_s:.1f} vs non-speculative (T32-spec ordinary steps) {st0.tok_s:.1f}: "
            f"x{st.tok_s / max(st0.tok_s, 1e-9):.2f} at c = {len(reqs)}")  # fmt: skip
    # (v) accept / reject sequences: c = 8 (T32 packed, every draft fits an idle lane) vs the c = 32 T64 run
    sub = sets["c2+chat think"][:8]
    st8 = drv.run_spec(sub, policy="all")
    assert st8.gen_stats["wide_steps"] == 0 and st8.gen_stats["packed_drafts"] > 0, st8.gen_stats
    diff = {r.name: first_divergence(seqs32["c2+chat think"][r.name], r.verdicts) for r in sub
            if r.verdicts != seqs32["c2+chat think"][r.name]}  # fmt: skip
    log(f"G-S5w (v): accept / reject sequences of {len(sub)} requests at c = 8 (T32 packed) vs c = 32 (T64): "
        f"{len(sub) - len(diff)} identical")  # fmt: skip
    assert not diff, f"accept / reject sequences differ (first divergence): {diff}"
    _relocation_probe(s, sets["c2+chat think"][3])
    _rollback_runs(s, sets["c2+chat think"])
    assert s.programs() == pc0, f"a program compiled after the capture: {pc0} -> {s.programs()}"


def _rows_of(t, rows: Sequence[int]) -> List[torch.Tensor]:
    """Host copies of ``t``'s rows ``rows`` (dim -2) on every chip (TILE or ROW_MAJOR; a host-side read, no device
    program)."""
    import ttnn

    out = []
    for sh in ttnn.get_device_tensors(t):
        x = ttnn.to_torch(sh).float()
        out.append(x.reshape(-1, x.shape[-2], x.shape[-1])[0][list(rows)].clone())
    return out


def _relocation_probe(s: SpecSession, r: Req) -> None:
    """G-S5w (iii) at full depth, eager: request ``r`` (its committed history in the cache) with the anchor ``t`` at
    ``n`` and the true next token ``d`` drafted at ``n + 1`` in a T64 step (a), then ``d`` at ``n + 1`` as the
    anchor of a T32-spec ordinary step (b) and of a T64 step (c): the final-norm hidden state rows and the logits rows
    of ``d`` at ``n + 1`` bitwise equal on every chip, ``a`` / ``m`` equal."""
    gen, model = s.gen, s.gen.model
    k = 10
    n = r.S + k
    t, d = r.out[k], r.out[k + 1]
    lane, L = r.lane, api.LANES_PER_GROUP
    dp, j = divmod(lane, L)

    def batch(pos: int, tok: int, draft: Optional[int]) -> api.SpecDecodeBatch:
        tk = torch.zeros(32, dtype=torch.int32)
        ps = torch.full((32,), -1, dtype=torch.int32)
        dr = torch.full((32,), -1, dtype=torch.int32)
        pt = torch.zeros(32, WIDTH, dtype=torch.int32)
        tk[lane], ps[lane] = tok, pos
        if draft is not None:
            dr[lane] = draft
        top = pos + (1 if draft is not None else 0)
        pt[lane, : top // BS + 1] = torch.tensor(r.blocks[: top // BS + 1], dtype=torch.int32)
        return api.SpecDecodeBatch(tokens=tk, positions=ps, draft_tokens=dr, page_table=pt)

    def wide_step(b):
        plan = gen.plan_wide_step(b, path=WIDE_ALL)
        p = gen._paths[WIDE_ALL]
        gen._write_wide(p, plan)
        rm, a, m, hn = model.decode_wide(p.inputs["tokens"], rot_idxs=p.inputs["rot"], kv_write=p.kv_write,
                                         kv_caches=s.pool, keep_hidden=True)  # fmt: skip
        return (a, m, hn, model.head.decode_logits(hn, halves=2), rm)

    def spec_step(b):
        plan = gen.plan_spec_step(b, path=SPEC_ALL)
        p = gen._paths[SPEC_ALL]
        gen._write_spec(p, plan.passes[0])
        rm, a, m, hn = model.decode_spec(p.inputs["tokens"], rot_idxs=p.inputs["rot"], kv_write=p.kv_write,
                                         kv_caches=s.pool, keep_hidden=True)  # fmt: skip
        lg = model.head.decode_logits(hn)
        return (a, m, hn, lg, rm)

    def take(outs, hn_row, lg_row, id_idx):
        a, m, hn, lg = outs[:4]
        hn_rows = _rows_of(hn, [hn_row])
        res = dict(
            a=int(model.head.tokens_to_host(a)[id_idx]), m=int(model.head.tokens_to_host(m)[id_idx]),
            hn=[hn_rows[c] for c in _row_chips(s, dp)], lg=_rows_of(lg, [lg_row]),
        )  # fmt: skip
        from models.demos.motif3.tt.generator import _free

        _free(*[x for x in outs if x is not None])
        return res

    pc0 = s.programs()
    ra = take(wide_step(batch(n, t, d)), 8 + j, 32 + lane, 32 + lane)  # (a) the draft row of the T64 step
    rb = take(spec_step(batch(n + 1, d, None)), j, lane, lane)  # (b) the next T32-spec step's anchor
    rc = take(wide_step(batch(n + 1, d, None)), j, lane, lane)  # (c) the next T64 step's anchor
    ok = {}
    for tag, other in (("T32 anchor", rb), ("T64 anchor", rc)):
        ok[tag] = (
            ra["a"] == other["a"] and ra["m"] == other["m"]
            and all(torch.equal(x, y) for x, y in zip(ra["hn"], other["hn"]))
            and all(torch.equal(x, y) for x, y in zip(ra["lg"], other["lg"]))
        )  # fmt: skip
    log(f"G-S5w (iii) {r.name} lane {lane}: token {d} at {n + 1} as a T64 draft row vs {ok}: hidden state (8 chips) "
        f"and logits rows (32 chips) bitwise; a {ra['a']} (= ref {r.out[k + 2]}), m {ra['m']}")  # fmt: skip
    assert all(ok.values()), f"draft-row relocation not bitwise: {ok}"
    assert ra["a"] == r.out[k + 2], "the draft row's argmax is not the greedy token"
    assert s.programs() == pc0


def _rollback_runs(s: SpecSession, reqs: Sequence[Req]) -> None:
    """G-S5w (iv): every request re-prefilled into fresh blocks and decoded ``ROLLBACK_STEPS + 2`` tokens with every
    lane drafting and a WRONG draft (the greedy token + 1) on every other request; the originals decode the same tokens
    without drafts. Tokens must match; the cache rows are compared after the traces are released."""
    drv, gen = s.drv, s.gen
    T = ROLLBACK_STEPS + 2
    plans = [("all_split", SPEC_ALL, WIDE_ALL)] + ([("row_split", SPEC_ROW, WIDE_ROW)] if T64_ROW_SPLIT else [])
    for mode, spec_key, wide_key in plans:
        for r in reqs:
            r.max_new = T
        drv.run_spec(reqs, policy="none", path=spec_key, check_argmax=True)  # the non-speculative reference
        ref = {r.name: list(r.out) for r in reqs}
        copies = [s.copy_req(r, f"{r.name}~{mode}") for r in reqs]
        orig = {c.name: r.name for r, c in zip(reqs, copies)}
        wrong = {c.name for i, c in enumerate(copies) if i % 2 == 0}
        V = int(gen.vocab_size)

        def draft_fn(c: Req) -> int:
            truth = ref[orig[c.name]]
            i = len(c.out)  # the draft is for the token of index i (the step after this one commits it)
            if i >= len(truth):
                return int(c.nxt)
            return (truth[i] + 1) % V if c.name in wrong else int(c.nxt)

        st = drv.run_spec(copies, policy="all", path=wide_key, want_logits=False, draft_fn=draft_fn)
        bad = {c.name: first_divergence(ref[orig[c.name]], c.out) for c in copies if c.out != ref[orig[c.name]]}
        rej = sum(sum(not v for v in c.verdicts) for c in copies if c.name in wrong)
        acc_wrong = sum(sum(v for v in c.verdicts) for c in copies if c.name in wrong)
        short = {r.name: len(ref[r.name]) for r in reqs if len(ref[r.name]) < T}
        log(f"G-S5w (iv) {mode}: {st.steps} T64 steps, {rej} wrong drafts rejected ({acc_wrong} accepted), "
            f"{st.accepted}/{st.offered} accepted overall; tokens {'exact' if not bad else bad}; requests ended by EOS "
            f"before {T} tokens: {short}")  # fmt: skip
        assert not bad, f"rollback probe {mode}: tokens differ from the non-speculative run: {bad}"
        assert acc_wrong == 0 and rej > 100, (acc_wrong, rej)
        # every pair's committed tokens (EOS can end a request before T): positions S .. S + n - 2 were written by
        # committed tokens in both runs (as anchors, or as accepted drafts), the last token is never written
        pairs = [(r.blocks, c.blocks, r.S, len(ref[r.name]), r.lane) for r, c in zip(reqs, copies)]
        s.report["rollback"].append(dict(mode=mode, pairs=pairs))
        for r in reqs:
            r.max_new = T64_MAX_NEW


def _replay_vs_eager_path(s: SpecSession, b: api.SpecDecodeBatch, key, tag: str) -> api.SpecDecodeResult:
    want = not b.is_verify and key[0] == "spec"
    rt = s.gen.decode_forward_spec(b, kv_cache=s.pool, enable_trace=True, want_logits=want, path=key)
    re_ = s.gen.decode_forward_spec(b, kv_cache=s.pool, enable_trace=False, want_logits=want, path=key)
    lanes = torch.nonzero(b.positions >= 0).flatten().tolist()
    drafted = torch.nonzero(b.draft_tokens >= 0).flatten().tolist()
    ok = _same_result(rt, re_, lanes, drafted)
    log(f"{tag}: {key} replay == eager bitwise {ok} ({len(lanes)} lanes, {len(drafted)} drafts)")
    assert ok, f"{tag}: {key} trace replay != eager step"
    return rt


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_t64_gs6w_lite_two_traces(t64_session):
    """G-S6w-lite (design §6.2): ``auto`` with both traces captured once; ``GS6W_PREFILLS`` (>= 100) solo prefills
    (packing off; cold rows, prefix hits, vLLM-chunked continuations) interleaved with T32 ordinary steps, T32 verify
    steps (drafts on idle lanes) and T64 verify steps (every lane drafting, more drafts than idle lanes). As G-X: every
    prefill logits row sane; a fixed probe prompt re-prefilled every 20 prefills gives bitwise the same logits; at the
    end trace == eager bitwise on both paths and the same verify step on T64 equals it on T32 (packed + overflow
    pass) bitwise; the program cache constant."""
    from models.demos.motif3.reference import golden_stream as gs

    s = t64_session
    gen, drv = s.gen, s.drv
    pc0 = s.programs()
    gen.packed_prefill = False  # solo prefills only (P5's packed passes are G-X's)
    rng = random.Random(64)
    src = [t for p in gs.load_prompt_set(GOLD_BF16 / "prompts.json") for t in p.ids]

    def text(k):
        i = rng.randrange(0, len(src) - 1)
        return [src[(i + q) % len(src)] for q in range(k)]

    probe_ids = text(700)
    probe_ref = None
    live: List[Req] = []
    done: List[Req] = []
    free = list(range(32))
    kinds, steps = {}, {"spec": 0, "wide": 0}
    g0 = dict(gen.stats)

    def retire(q: Req) -> None:
        live.remove(q)
        done.append(q)
        free.append(q.lane)

    try:
        for i in range(GS6W_PREFILLS):
            if not free:  # every lane busy: a request finishes early
                retire(live[rng.randrange(len(live))])
            lane = free.pop(rng.randrange(len(free)))
            kind = rng.choice(("cold", "cold", "hit", "chunk")) if done else "cold"
            if kind == "hit":  # a prefix hit on a finished request's full blocks
                base = rng.choice(done)
                k = rng.randint(1, max(1, (base.S - 1) // BS))
                ids = base.prompt[: k * BS] + text(rng.randint(1, 300))
                r = Req(f"h{i}", ids, base.blocks[:k] + s.blocks.take(api.cdiv(len(ids) + 40, BS) - k), lane, -1, 24)
                s.prefill(r, start=k * BS)
            elif kind == "chunk":  # a vLLM-chunked prompt: two calls, the second at an unaligned start
                ids = text(rng.randint(500, 1000))
                r = Req(f"k{i}", ids, s.blocks.take(api.cdiv(len(ids) + 40, BS)), lane, -1, 24)
                e = rng.randint(130, len(ids) - 1)
                s.prefill(r, end=e)
                s.prefill(r, start=e)
            else:
                ids = text(rng.randint(20, 600))
                r = Req(f"c{i}", ids, s.blocks.take(api.cdiv(len(ids) + 40, BS)), lane, -1, 24)
                s.prefill(r)
            kinds[kind] = kinds.get(kind, 0) + 1
            drv.start([r])
            live.append(r)
            if i % 20 == 19:  # the probe on fresh blocks: one distinct output over the session
                pr = Req(f"probe{i}", probe_ids, s.blocks.take(api.cdiv(700, BS)), lane, -1, 1)
                lg = s.prefill(pr)
                if probe_ref is None:
                    probe_ref = lg.clone()
                assert torch.equal(lg, probe_ref), f"probe prefill {i}: logits differ from the first probe"
            for _ in range(rng.randint(1, 3)):
                for q in [q for q in live if q.done]:
                    retire(q)
                if not live:
                    break
                # the routed step (T32 unless the drafts overflow the idle lanes), or a forced T64 replay: few
                # requests live at once here, so the T64 trace is forced on ~40 % of the steps
                wide = rng.random() < 0.4
                policy = rng.choice(("none", "budget", "all", "all"))
                drv.spec_step(live, policy=policy, path=WIDE_ALL if wide else None, want_logits=False if wide else None)
                steps[gen.last_verify_kind] += 1
    finally:
        gen.packed_prefill = True
    assert s.programs() == pc0, f"program cache grew after the capture: {pc0} -> {s.programs()}"
    d = {k: gen.stats[k] - g0[k] for k in ("spec_steps", "wide_steps", "verify_steps", "auto_t32_verifies")}
    log(f"G-S6w-lite: {GS6W_PREFILLS} solo prefills ({kinds}) between {steps} decode steps (T32 / T64); {d}; the probe "
        f"bitwise identical; program cache constant at {pc0}")  # fmt: skip
    assert steps["wide"] > 5 and steps["spec"] > 5, steps
    # end: trace == eager on both paths; T64 == T32 (packed + overflow) bitwise for the same verify step
    while len(live) < 24 and free:
        ids = text(rng.randint(20, 300))
        r = Req(f"f{len(live)}", ids, s.blocks.take(api.cdiv(len(ids) + 40, BS)), free.pop(), -1, 24)
        s.prefill(r)
        drv.start([r])
        live.append(r)
    for _ in range(2):
        drv.spec_step(live, policy="none")
    for r in live:
        r.draft = r.nxt
    b = drv.batch(live)
    rt64 = _replay_vs_eager_path(s, b, WIDE_ALL, "G-S6w-lite end")
    rt32 = _replay_vs_eager_path(s, b, SPEC_ALL, "G-S6w-lite end")
    assert len(gen.last_spec.passes) == 2, "the T32 run of 24 drafts should overflow its 8 idle lanes"
    lanes = [r.lane for r in live]
    same = bool(torch.equal(rt64.argmax[lanes], rt32.argmax[lanes])) and bool(
        torch.equal(rt64.mtp_argmax[lanes], rt32.mtp_argmax[lanes]))  # fmt: skip
    log(f"G-S6w-lite: the same verify step (24 lanes drafting) on T64 vs T32 (packed + overflow): a / m bitwise {same}")
    assert same, "T64 rows != T32 rows for the same verify step"
    ordinary = drv.batch([r for r in live])
    no_drafts = torch.full((32,), -1, dtype=torch.int32)
    ordinary = api.SpecDecodeBatch(
        tokens=ordinary.tokens, positions=ordinary.positions, draft_tokens=no_drafts, page_table=ordinary.page_table
    )
    _replay_vs_eager_path(s, ordinary, SPEC_ALL, "G-S6w-lite end (ordinary)")
    assert s.programs() == pc0


def _contiguous_runs(ids: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for b in ids:
        if out and out[-1][1] == b:
            out[-1] = (out[-1][0], b + 1)
        else:
            out.append((b, b + 1))
    return out


def _cache_rows(cache, blocks: Sequence[int], lo: int, hi: int) -> List[torch.Tensor]:
    """Every chip's copy of positions ``[lo, hi)`` of a request whose page table is ``blocks`` (after the traces are
    released: the slices compile programs)."""
    import ttnn

    need = list(blocks[lo // BS : (hi - 1) // BS + 1])
    per_chip: Optional[List[List[torch.Tensor]]] = None
    for a, b in _contiguous_runs(need):
        sl = ttnn.slice(cache, [a, 0, 0, 0], [b, 1, BS, api.KV_LATENT_DIM])
        shards = [ttnn.to_torch(t).float().reshape(b - a, BS, api.KV_LATENT_DIM) for t in ttnn.get_device_tensors(sl)]
        ttnn.deallocate(sl)
        if per_chip is None:
            per_chip = [[] for _ in shards]
        for c, x in enumerate(shards):
            per_chip[c].append(x)
    off = (lo // BS) * BS
    return [torch.cat(xs).reshape(-1, api.KV_LATENT_DIM)[lo - off : hi - off] for xs in per_chip]


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_t64_release_readback(t64_session):
    """Last test of the T64 session (it releases the traces): (1) G-S5w (iv)'s cache check: for every request pair of
    the rollback probe, the decode rows ``[S, S + n - 1)`` (``n`` committed tokens, ``<= T`` when EOS ended the request:
    the positions both runs wrote) of the speculative copy (wrong drafts rejected on half the lanes) equal the
    non-speculative original's bitwise in layers 0, 1, the first global MoE layer, the last layer and
    the MTP cache, on every chip (``all_split``: all 32; ``row_split``: the request's DP row). (2) With
    ``MOTIF3_G16_OPTION_A``: the T64 trace re-captured with option A (one B = 16 FlashMLA call on the global layers)
    next to A'' (re-captured too), the step cost at G16's contexts (informational: A'' is the shipped option)."""
    s = t64_session
    gen = s.gen
    if not s.report.get("rollback"):
        pytest.skip("the rollback probe did not run (test_t64_gs5w_lossless)")
    gen.release_traces()
    n = gen.num_layers
    gl = next((l for l in range(2, n) if gen.cfg.layer(l).is_global and gen.cfg.layer(l).is_moe), None)
    caches = {f"L{l}": s.pool[l] for l in sorted({0, min(1, n - 1), n - 1} | ({gl} if gl is not None else set()))}
    caches["MTP"] = s.pool.mtp
    bad = {}
    for entry in s.report["rollback"]:
        mode = entry["mode"]
        for ref_blocks, cp_blocks, S, n_tok, lane in entry["pairs"]:
            lo, hi = S, S + n_tok - 1  # the positions both runs wrote with committed tokens
            for name, cache in caches.items():
                x, y = _cache_rows(cache, ref_blocks, lo, hi), _cache_rows(cache, cp_blocks, lo, hi)
                chips = range(len(x)) if mode == "all_split" else _row_chips(s, lane // api.LANES_PER_GROUP)
                diff = [c for c in chips if not torch.equal(x[c], y[c])]
                if diff:
                    bad[(mode, lane, name)] = diff[:4]
        log(f"G-S5w (iv) cache check {mode}: {len(entry['pairs'])} request pairs x {list(caches)}: "
            f"{'bitwise equal' if not any(k[0] == mode for k in bad) else 'DIFFER'}")  # fmt: skip
    assert not bad, f"rollback probe: cache rows differ after the rejected drafts were rewritten: {bad}"
    if not G16_OPTION_A:
        return
    # (2) option A vs A'': re-capture the T64 path both ways (the session's traces are released)
    import ttnn

    p = gen._paths[WIDE_ALL]
    pool = s.pool
    res = {}
    for option in ("A''", "A"):
        if option == "A":
            w = p.kv_write
            w.flash_groups = lambda w=w: [(slice(0, w.lanes_per_row), w.cur_pos, w.page_table)]
        gen._inactive_step(p, pool)  # option A's global B = 16 FlashMLA program compiles here (no trace alive)
        gen._write_inactive(p)
        ttnn.synchronize_device(s.mesh)
        p.trace_id, p.out = gen._capture_path(p, pool)
        p.pool = pool
        try:
            hist_all, own = _g16_blocks(s)
            for ctx in G16_CONTEXTS:
                res[(option, ctx)] = _time_path(s, WIDE_ALL, hist_all[: ctx // BS], own, ctx, drafts=True)
        finally:
            ttnn.release_trace(s.mesh, p.trace_id)
            from models.demos.motif3.tt.generator import _free

            _free(*[t for t in p.out if t is not None])
            p.trace_id = p.out = p.pool = None
            if option == "A":
                del p.kv_write.flash_groups
    for ctx in G16_CONTEXTS:
        a2, a1 = res[("A''", ctx)][1], res[("A", ctx)][1]
        log(f"G16 option A vs A'' at {ctx}: T64 device step A'' {a2:.2f} ms, A {a1:.2f} ms (A'' costs {a2 - a1:+.2f} "
            f"ms per step for bitwise T64 = T32 rows)")  # fmt: skip
    s.report["g16_option_a"] = {f"{o}/{c}": v for (o, c), v in res.items()}


def _row_chips(s: SpecSession, dp: int) -> List[int]:
    """Device-tensor indices (``ttnn.get_device_tensors`` order: mesh row-major) of DP row ``dp``'s chips."""
    a = s.cfg.axes
    C = int(a.mesh_shape[1])
    out = []
    for tp in range(s.cfg.tp):
        r, c = a.coord(dp, tp)
        out.append(r * C + c)
    return out


# ======================================================================================================================
# device: the "wide" fallback launch (the T64 trace alone, with the anchors' logits and the device sampler on them)
# ======================================================================================================================
@pytest.fixture(scope="module")
def wide_session():
    from models.demos.motif3.tests.test_resumed_prefill import RefuseReads
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR, open_motif_mesh

    _claim_device("wide")
    mesh = open_motif_mesh()
    holder: Dict[str, Any] = {}
    close = _session_closer("wide", mesh, holder)
    try:
        A = api.DEFAULT_PREFILL_ALIGNMENT
        budget = PP.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, A)
        settings = api.GeneratorSettings(
            max_batch_size=api.NUM_LANES, max_seq_len=MAX_LEN, num_layers=SPEC_LAYERS, kv_cache_dtype="bfp8",
            weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=BS, weights_source="TT cache only (test)",
            chunked_prefill=True, prefix_caching=True, max_num_batched_tokens=budget,
            long_prefill_token_threshold=budget, spec_tokens=1, spec_verify="wide",
        )  # fmt: skip
        guard = RefuseReads(DEFAULT_WEIGHTS_DIR)
        gen = holder["gen"] = MotifGenerator.create(hf_config=None, mesh_device=mesh, settings=settings, source=guard)
        assert gen.serving_paths == [WIDE_ALL] and gen.wide_has_logits
        pool = gen.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BS, num_layers=SPEC_LAYERS)
        gen.warmup_prefill(kv_cache=pool, enable_trace=False)
        gen.enable_device_sampling()
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=WIDTH)
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=WIDTH)
        p = gen._paths[WIDE_ALL]
        assert p.traced and p.so is not None and p.out[0] is not None, "the wide T64 trace holds the logits + sampler"
        yield T64Session(mesh, gen, pool, guard, cap=max(T64_MAX_NEW, MAX_NEW, 64) + 4)
    finally:
        _OPEN_SESSIONS.pop("wide", None)
        close()


@pytest.mark.timeout(SPEC_TIMEOUT)
def test_t64_wide_launch(wide_session):
    """``spec_verify="wide"`` (the one-trace fallback, design §2.2): every step on the T64 trace. (a) 32 greedy requests
    without drafts (ordinary T64 steps, the anchors' host logits: ``a0`` == the host argmax of every logits row) and
    with every lane drafting give the same tokens; (b) device-sampled ordinary steps on the T64 anchors: lanes at
    temperature 0 return the greedy token, the seeded lanes' tokens repeat bitwise when the same steps run again, no
    flagged lane is left unresolved; the program cache constant."""
    s = wide_session
    gen, drv = s.gen, s.drv
    pc0 = s.programs()
    max_new = min(T64_MAX_NEW, 96)
    reqs = [s.chat_req(qi, True, lane_of_slot(i), max_new) for i, qi in enumerate(range(32))]
    st0 = drv.run_spec(reqs, policy="none")  # ordinary T64 steps with host logits (check_argmax)
    ref = {r.name: list(r.out) for r in reqs}
    st = drv.run_spec(reqs, policy="all")
    _assert_same_tokens("wide launch: T64 drafting vs T64 ordinary", reqs, ref)
    assert st0.gen_stats["wide_steps"] == st0.steps and st.gen_stats["wide_steps"] == st.steps
    log(f"wide launch (a): {st0.steps} ordinary T64 steps vs {st.steps} drafting T64 steps (acceptance "
        f"{st.acceptance:.3f}): token-exact; step ms ordinary {st0.median_ms('ordinary_t64'):.1f}, verify "
        f"{st.median_ms('verify_t64'):.1f}")  # fmt: skip
    # (b) device-sampled steps on the T64 anchors: half the lanes greedy (temperature 0), half seeded at T = 0.8
    drv.start(reqs)
    temp = [0.0 if i % 2 == 0 else 0.8 for i in range(32)]  # by lane
    top_p = [1.0 if i % 2 == 0 else 0.95 for i in range(32)]
    top_k = [0] * 32
    seeds = [None if i % 2 == 0 else 1000 + i for i in range(32)]

    def sampled_run(n: int) -> List[List[int]]:
        out = []
        for r in reqs:
            r.out = [r.first]
        for _ in range(n):
            tok = torch.zeros(32, dtype=torch.int32)
            pos = torch.full((32,), -1, dtype=torch.int32)
            pt = torch.zeros(32, WIDTH, dtype=torch.int32)
            for r in reqs:
                q = r.anchor_pos()
                tok[r.lane], pos[r.lane], pt[r.lane] = r.out[-1], q, drv.page_row(r, q)
            b = api.DecodeBatch(tokens=tok, positions=pos, page_table=pt)
            res = gen.decode_forward_sampled(b, (temp, top_p, top_k, seeds), kv_cache=s.pool, enable_trace=True)
            assert gen.last_verify_kind == "wide"
            out.append([int(res.tokens[r.lane]) for r in reqs])
            for r in reqs:
                r.out.append(int(res.tokens[r.lane]))
        return out

    n = 24
    first = sampled_run(n)
    again = sampled_run(n)
    greedy_ok = all(first[k][i] == ref[r.name][k + 1] for k in range(n) for i, r in enumerate(reqs)
                    if temp[r.lane] == 0.0)  # fmt: skip
    log(f"wide launch (b): {n} device-sampled T64 steps x 32 lanes: temperature-0 lanes == greedy {greedy_ok}; seeded "
        f"repeat bitwise {first == again}; sampler {gen.sampling_stats()}")  # fmt: skip
    assert greedy_ok, "a temperature-0 lane sampled on the T64 anchors differs from the greedy token"
    assert first == again, "seeded device sampling on the T64 trace is not reproducible"
    assert s.programs() == pc0, f"a program compiled after the capture: {pc0} -> {s.programs()}"
