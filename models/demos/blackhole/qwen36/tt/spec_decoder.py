# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device side of SERVED speculative decoding with the MTP drafter (the P/D decode engine, TP=4).

One ``SpecDecoder`` per model owns the verify plans of the ladder (tt/spec_serving.py: one traced ``VerifyStep`` per
(bucket width, T), pad-safe, keep_hidden), the drafter's draft-step traces per bucket width (tt/mtp_head.py) and the
host state machine (``SpecServingState``). Per decode step the vLLM wrapper (tt/qwen36_vllm.py) hands it the padded
decode inputs of the plugin (tokens [B,1], positions [B] with -1 pad rows, block tables [B, blocks], the request per
row, the scheduled draft tokens per row, whether every request may take the greedy verify path, and the scheduler's
flush request) and gets back, per grid row, the committed tokens (1..1+K_s) and the drafts for the next step -- or
None when the step must run as a plain decode (above the ladder, or a request that forces host/sampled decoding).

Step = [migrate qkv_prev rows on a plan change] -> [catch-up draft step for freshly admitted users: writes the head's
KV at P_s-1 from the imported hidden row] -> verify (traced, R <= 32 = bitwise the decode step) -> commit -> select
the accepted rows' hidden states into the drafter -> k chained draft steps (traced). Every program runs compiled
before any trace is captured (``compile`` in the first warm-up phase, ``capture`` in the second; VERIFY_W32_AUDIT.md).
"""
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger

from models.demos.blackhole.qwen36.tt import spec_serving as ss
from models.demos.blackhole.qwen36.tt.verify_step import VerifyStep, attn_multi_token_update_available


@dataclass
class SpecStepResult:
    """One speculative decode step's outcome over the grid rows 0..w-1 (rows above the plan's width did not exist)."""

    w: int
    committed: list  # [w] tokens committed per grid row ([] for padding rows)
    next_drafts: list  # [w] draft tokens for the next step per grid row ([] for padding rows)
    accepts: list  # [w] accepted drafts per grid row
    plan: ss.Plan
    w_grid: int
    flush: bool
    migrated: bool
    n_catchup: int
    hold: ss.HoldInfo
    times_ms: dict = field(default_factory=dict)


class SpecDecoder:
    def __init__(self, model, head, ladder: ss.Ladder, bmax: int, num_blocks_pt: int, log_every: Optional[int] = None):
        """model: the Qwen36Model (KV caches allocated); head: its MTPHead (prefill buckets may already be compiled by
        the P-side installer; the draft-step buffers of the ladder's widths are built here); num_blocks_pt: the
        served block-table width (max_num_blocks_per_req, a multiple of 8). Allocates every persistent buffer
        (call before ANY trace capture)."""
        self.model = model
        self.head = head
        self.mesh = model.mesh_device
        self.ladder = ladder
        self.bmax = int(bmax)
        self.nb = int(num_blocks_pt)
        assert self.nb % 8 == 0, "block-table width must be a multiple of 8 blocks"
        assert attn_multi_token_update_available(), (
            "served speculative decoding needs the batched verify attention (paged_update_cache num_tokens=); "
            "the per-offset middle mixes padding rows through its 0/1 gathers"
        )
        self.state = ss.SpecServingState(ladder, self.bmax)
        zeros_pt = torch.zeros(self.bmax, self.nb, dtype=torch.int32)
        t0 = time.perf_counter()
        head.build_step_buffers(ladder.widths, zeros_pt)
        self.steps = {}
        for plan in ladder.plans:
            vs = VerifyStep(model, plan.w, plan.T, zeros_pt[: plan.w], keep_hidden=True, pad_safe=True)
            assert vs.plan.attn_mode == "batched", vs.plan.attn_mode
            head.bind_plan(vs.plan)
            self.steps[plan] = vs
        self.log_every = int(os.environ.get("QWEN36_SPEC_LOG_EVERY", "50")) if log_every is None else int(log_every)
        self._compiled = self._captured = False
        self.n_steps = 0
        self.acc = {"verify_ms": 0.0, "select_ms": 0.0, "draft_ms": 0.0, "catchup_ms": 0.0, "migrate_ms": 0.0}
        self.acc_tokens = 0
        self.acc_users = 0
        self.acc_accepted = 0
        logger.info(
            f"[spec] decoder: ladder {[str(p) for p in ladder.plans]} k_max={ladder.k_max} bmax={self.bmax} "
            f"page table {self.nb} blocks; {len(self.steps)} verify plans + {len(ladder.widths)} draft widths allocated "
            f"in {time.perf_counter() - t0:.1f}s"
        )

    # ------------------------------------------------------------------------------------------ warm-up
    def compile(self):
        """Phase 1 (no trace captured yet): every program the loop runs -- draft steps per width, verify bodies,
        selects, qkv_prev migrations between every plan pair (mutates the KV / GDN state of slots 0..w-1: warm-up)."""
        if self._compiled:
            return
        t0 = time.perf_counter()
        for w in self.ladder.widths:
            self.head.compile_step(w)
        for plan, vs in self.steps.items():
            t1 = time.perf_counter()
            vs.compile()
            self.head.compile_select(vs.plan)
            logger.info(f"[spec] verify {plan} R={vs.plan.R} compiled in {time.perf_counter() - t1:.1f}s")
        for a, va in self.steps.items():
            for b, vb in self.steps.items():
                if a != b:
                    vb.plan.compile_migration(va.plan)
        # the sparse hidden-row upload + one padded draft step (the catch-up) run the same programs as a draft step
        self._compiled = True
        logger.info(f"[spec] every speculative program compiled in {time.perf_counter() - t0:.1f}s")

    def capture(self):
        """Phase 2: capture the draft-step and verify traces (all programs compiled)."""
        assert self._compiled, "compile() first"
        if self._captured:
            return
        t0 = time.perf_counter()
        for w in self.ladder.widths:
            self.head.capture_step(w)
        for plan, vs in self.steps.items():
            vs.capture()
        self._captured = True
        logger.info(
            f"[spec] {len(self.steps)} verify + {len(self.ladder.widths)} draft traces captured in {time.perf_counter() - t0:.1f}s"
        )

    def release(self):
        for vs in self.steps.values():
            vs.release()
        self.head.release()

    # ------------------------------------------------------------------------------------------ per step
    def hold_info(self) -> ss.HoldInfo:
        return self.state.hold_info()

    def note_plain_step(self, row_req_ids, drafts=None):
        """A step that runs as plain decode: keeps the slot ownership in sync and checks nothing was pending."""
        rows = list(row_req_ids) + [None] * (self.bmax - len(row_req_ids))
        if not any(r is not None for r in rows):
            return
        self.state.begin_step(rows, drafts or [None] * self.bmax, eligible=False)

    def step(self, tokens, positions, page_table, row_req_ids, drafts, eligible=True, flush=False):
        """One decode step. tokens [B,1] / positions [B] (-1 = pad row) / page_table [B, blocks] as the plugin builds
        them (B >= highest live row + 1); row_req_ids[s] the request at row s (None = pad); drafts[s] its scheduled
        draft tokens (list, -1 = unfilled placeholder) or None. Returns a SpecStepResult, or None when the step must
        run as plain decode (the caller then runs the traced decode as usual)."""
        B = int(tokens.shape[0])
        pos_all = [int(p) for p in positions.reshape(-1).tolist()]
        rows = list(row_req_ids) + [None] * (self.bmax - len(row_req_ids))
        rows = [rows[s] if s < B and pos_all[s] >= 0 else None for s in range(self.bmax)]
        dr = list(drafts) + [None] * (self.bmax - len(drafts))
        sp = self.state.begin_step(rows, dr, eligible, flush=flush, has_hidden=lambda s: s in self.head.pending_rows)
        if sp.mode == "plain":
            return None
        plan, w, k = sp.plan, sp.plan.w, sp.plan.k
        vs = self.steps[plan]
        times = {}
        tok_all = tokens.reshape(-1).tolist()
        last = [int(tok_all[s]) if sp.live[s] else 0 for s in range(w)]
        pos = [pos_all[s] if sp.live[s] else -1 for s in range(w)]
        pt = page_table[:w].to(torch.int32).clone()
        for s in range(w):
            if not sp.live[s]:
                pt[s] = 0  # the null block: never written (update skipped at -1), finite zeros for the SDPA read
        pad = [not lv for lv in sp.live]

        if sp.migrate:
            t0 = time.perf_counter()
            vs.plan.migrate_qkv_prev_from(self.steps[sp.prev_plan].plan, live=sp.live)
            times["migrate_ms"] = 1e3 * (time.perf_counter() - t0)
        if sp.catchup:
            # the fresh users' first draft step: (t'_s, h_{P_s-1}) at P_s - 1 fills the head's KV hole left by the
            # prefill (positions ..P_s-2) before the post-verify chain attends over it; its draft is not used
            # (the scheduler gave these users no draft slots this step)
            t0 = time.perf_counter()
            rows_h = {s: self.head.pending_rows.pop(s) for s in sp.catchup}
            self.head.set_page_table(w, pt)
            self.head.upload_hidden_rows(w, rows_h)
            cu = set(sp.catchup)
            self.head.run_step(
                w, [last[s] if s in cu else 0 for s in range(w)], [pos[s] - 1 if s in cu else -1 for s in range(w)]
            )
            times["catchup_ms"] = 1e3 * (time.perf_counter() - t0)
        # a slot re-used by a new request whose hidden row never arrived: nothing to consume; a stale row of a
        # previous occupant is dropped when the owner changes (SpecServingState resets the slot -> not fresh here)
        for s in range(w):
            if sp.live[s] and s in self.head.pending_rows and s not in sp.catchup:
                self.head.pending_rows.pop(s, None)

        t0 = time.perf_counter()
        grid_tokens = self.state.grid_tokens(sp, last)
        argmax_rows = vs.run(grid_tokens, pos, sp.accept_prev, page_table=pt)
        accepts, committed = self.state.commit(sp, argmax_rows, grid_tokens)
        t1 = time.perf_counter()
        times["verify_ms"] = 1e3 * (t1 - t0)

        next_drafts = [[] for _ in range(w)]
        if k >= 1:
            self.head.select_hidden(vs.plan, accepts)
            t2 = time.perf_counter()
            times["select_ms"] = 1e3 * (t2 - t1)
            self.head.set_page_table(w, pt)
            new_last = [committed[s][-1] if sp.live[s] else 0 for s in range(w)]
            new_pos = [pos[s] + accepts[s] + 1 if sp.live[s] else 0 for s in range(w)]
            drafted = self.head.draft(w, k, new_last, new_pos, pad=pad)
            next_drafts = [list(drafted[s]) if sp.live[s] else [] for s in range(w)]
            times["draft_ms"] = 1e3 * (time.perf_counter() - t2)

        hold = self.state.hold_info()
        self._account(sp, committed, accepts, times)
        return SpecStepResult(
            w=w,
            committed=committed,
            next_drafts=next_drafts,
            accepts=accepts,
            plan=plan,
            w_grid=sp.w_grid,
            flush=sp.flush,
            migrated=sp.migrate,
            n_catchup=len(sp.catchup),
            hold=hold,
            times_ms=times,
        )

    # ------------------------------------------------------------------------------------------ metrics
    def _account(self, sp, committed, accepts, times):
        self.n_steps += 1
        n_live = sum(sp.live)
        self.acc_users += n_live
        self.acc_tokens += sum(len(c) for c in committed)
        self.acc_accepted += sum(accepts)
        for key, v in times.items():
            self.acc[key] = self.acc.get(key, 0.0) + v
        if self.log_every and self.n_steps % self.log_every == 0:
            st = self.state.stats
            n = self.log_every
            logger.info(
                f"[spec] step {self.n_steps}: w_grid={sp.w_grid} plan={sp.plan} live={n_live} flush={sp.flush} "
                f"| last {n} steps: {self.acc_tokens / max(1, self.acc_users):.2f} tok/user/step "
                f"({self.acc_accepted / max(1, self.acc_users):.2f} accepted), per step verify "
                f"{self.acc['verify_ms'] / n:.1f} + select {self.acc['select_ms'] / n:.1f} + draft "
                f"{self.acc['draft_ms'] / n:.1f} ms (catch-up {self.acc['catchup_ms'] / n:.2f}, migrate "
                f"{self.acc['migrate_ms'] / n:.2f}) | totals: spec {st['spec_steps']} plain {st['plain_steps']} "
                f"flushes {st['flushes']} plan changes {st['plan_changes']} migrations {st['migrations']}"
            )
            self.acc = {key: 0.0 for key in self.acc}
            self.acc_tokens = self.acc_users = self.acc_accepted = 0
