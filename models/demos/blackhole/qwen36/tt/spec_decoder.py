# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device side of SERVED speculative decoding (the P/D decode engine, TP=4) with either drafter.

One ``SpecDecoder`` per model owns the verify plans of the ladder (tt/spec_serving.py: one traced ``VerifyStep`` per
(bucket width, T), pad-safe), the drafter's per-width draft-step traces and the host state machine
(``SpecServingState``). Per decode step the vLLM wrapper (tt/qwen36_vllm.py) hands it the padded decode inputs of the
plugin (tokens [B,1], positions [B] with -1 pad rows, block tables [B, blocks], the request per row, the scheduled
draft tokens per row, whether every request may take the greedy verify path, and the scheduler's flush request) and
gets back, per grid row, the committed tokens (1..1+K_s) and the drafts for the next step -- or None when the step
must run as a plain decode (above the ladder, or a request that forces host/sampled decoding).

Drafters (``QWEN36_SPEC_DRAFTER``, tt/aux_hidden.py ``spec_drafter``), behind one adapter interface (``_MtpDrafter`` /
``_DFlash2Drafter``):
  * ``mtp`` -- the checkpoint's MTP head (tt/mtp_head.py). Verify plans keep the post-norm hidden rows
    (``keep_hidden``); step = [migrate qkv_prev rows on a plan change] -> [catch-up draft step for freshly admitted
    users: writes the head's KV at P_s-1 from the imported hidden row] -> verify -> commit -> select the accepted rows'
    hidden states into the head (exact 0/1 matmul) -> k chained draft steps (traced per width).
  * ``dflash2`` -- the DFlash2 block-diffusion drafter (tt/dflash2_head.py, z-lab/Qwen3.8-27B-DFlash2). Verify plans
    keep the target's aux hidden rows (``keep_aux_hidden`` -> ``plan.out_aux``); step = [migrate] -> verify -> commit
    -> ``drafter.commit(plan, positions)`` (traced per plan: the grid rows' aux -> the drafter's context K/V at
    P_s + j) -> ONE traced block draft step at the bucket width (7 drafts per user, the first k used). The prompt's
    context K/V arrive with the P/D payload (KV group "dflash2", imported into the drafter's caches by the connector
    before admission) or from a local prefill hook; the runner reports the import per slot (``note_context``) and a
    request without it decodes with no drafts (its rows are padding in the draft step). No hidden row and no
    catch-up step: the first verify step at P_s = N (row 0 = P's first token) commits the aux row of position N, so
    the first draft (anchor = the committed token at N + a_s, positions N + a_s + 1 ..) reads a gap-free context.
  * ``hybrid`` -- BOTH drafters resident, the ladder picks one per plan by width (tt/spec_serving.py
    ``Ladder.drafter_for``: DFlash2 at T = 8 up to 4 users, the MTP bands above; the served A/B behind the split is in
    the plugin's docs/SPECULATIVE.md) AND by context length (the context rule of spec_serving.py: the decoder hands
    the state machine the rows' decode positions every step; past QWEN36_SPEC_DFLASH2_MAX_CTX the grid runs the long
    ladder's MTP plans, so the decoder allocates / compiles / captures the verify plans of BOTH ladders and the
    migrations between all of them). Verify plans keep both the post-norm rows and the aux rows. Every spec step
    keeps BOTH drafters' per-slot state current so a drafter change (a plan change: flush / migration as usual) is
    exact from the new drafter's point of view without a re-prefill: the DFlash2 context commit runs on every step
    (also when the MTP head drafts), and the MTP head's K/V for the committed rows is written by ``MTPHead.keep_current``
    (the head's layer over the grid rows from the verify's argmax + post-norm rows) on every step the MTP chain did
    not just cover -- i.e. every DFlash2-drafted step and the first MTP-drafted step after one (or after a plain step);
    inside the MTP band the chain writes the head's K/V exactly as in mtp mode. The MTP catch-up step of a freshly
    admitted user (its imported hidden row) runs whichever drafter is active, the DFlash2 context report
    (``note_context``) is taken as in dflash2 mode; a user without DFlash2 context proposes nothing while DFlash2 is
    active and drafts normally once the MTP head is. P ships both states (payload v3: mtp.kv / mtp.hidden + KV group
    "dflash2"), D imports both. With ``QWEN36_SPEC_DRAFTER=mtp`` / ``dflash2`` nothing of this runs: those loops are
    the 2026-09-28 ones op for op.
Every program runs compiled before any trace is captured (``compile`` in the first warm-up phase, ``capture`` in the
second; tests/VERIFY_W32_AUDIT.md).
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
    drafter: Optional[str] = None  # the drafter that drafted at this plan (hybrid: by width and context)
    long: bool = False  # the hybrid context rule's mode of this step


# ================================================================================================ drafter adapters
class _MtpDrafter:
    """The MTP head behind the decoder's drafter interface (the op sequence of the 2026-09-25 served loop)."""

    name = "mtp"
    post_name = "select"  # the per-step op between the verify commit and the draft (metrics key)

    def __init__(self, head):
        self.head = head

    def verify_kwargs(self):
        return dict(keep_hidden=True)

    def build_step_buffers(self, widths, page_tables):
        self.head.build_step_buffers(widths, page_tables)

    def bind_plan(self, plan):
        self.head.bind_plan(plan)

    def compile_width(self, w):
        self.head.compile_step(w)

    def compile_plan(self, vs):
        self.head.compile_select(vs.plan)

    def capture_width(self, w):
        self.head.capture_step(w)

    def capture_plan(self, vs):
        pass

    def release(self):
        self.head.release()

    # --- per-slot state ---
    def has_state(self, slot) -> bool:
        return slot in self.head.pending_rows

    def drop_state(self, slot):
        self.head.pending_rows.pop(slot, None)

    def note_context(self, slot, req_id, meta):
        pass  # the hidden row arrives through MTPHead.set_hidden_in (pd_transfer.import_mtp_hidden)

    def drafts_enabled(self, slot, active=None) -> bool:
        return True

    def drop_stale(self, w, sp):
        """A slot re-used by a new request whose hidden row never arrived has nothing to consume; a stale row of a
        previous occupant is dropped when the owner changes (SpecServingState resets the slot -> not fresh here)."""
        for s in range(w):
            if sp.live[s] and self.has_state(s) and s not in sp.catchup:
                self.drop_state(s)

    def note_plain(self):
        pass

    # --- per step ---
    def catchup(self, w, sp, last, pos, pt):
        """The fresh users' first draft step: (t'_s, h_{P_s-1}) at P_s - 1 fills the head's KV hole left by the
        prefill (positions ..P_s-2) before the post-verify chain attends over it; its draft is not used (the scheduler
        gave these users no draft slots this step)."""
        rows_h = {s: self.head.pending_rows.pop(s) for s in sp.catchup}
        self.head.set_page_table(w, pt)
        self.head.upload_hidden_rows(w, rows_h)
        cu = set(sp.catchup)
        self.head.run_step(
            w, [last[s] if s in cu else 0 for s in range(w)], [pos[s] - 1 if s in cu else -1 for s in range(w)]
        )

    def after_verify(self, vs, sp, accepts, pos_before, argmax_rows=None, active=None):
        self.head.select_hidden(vs.plan, accepts)

    def draft(self, w, k, last, positions, pad, pt, active=None):
        self.head.set_page_table(w, pt)
        return self.head.draft(w, k, last, positions, pad=pad)


class _DFlash2Drafter:
    """The DFlash2 block drafter behind the decoder's drafter interface (module docstring)."""

    name = "dflash2"
    post_name = "commit"

    def __init__(self, head):
        self.head = head
        self.aux_layers = tuple(int(i) for i in head.cfg.target_layer_ids)
        self.context = {}  # slot -> (req_id, meta) of the imported / prefilled context K/V
        self._warned_window = False

    def verify_kwargs(self):
        return dict(keep_aux_hidden=True, aux_layers=self.aux_layers)

    def build_step_buffers(self, widths, page_tables):
        self.head.build_step_buffers(widths, page_tables)

    def bind_plan(self, plan):
        self.head.bind_plan(plan)

    def compile_width(self, w):
        self.head.compile_step(w)

    def compile_plan(self, vs):
        self.head.compile_commit(vs.plan)

    def capture_width(self, w):
        self.head.capture_step(w)

    def capture_plan(self, vs):
        self.head.capture_commit(vs.plan)  # reads the verify trace's out_aux: after vs.capture()

    def release(self):
        self.head.release()

    # --- per-slot state ---
    def note_context(self, slot, req_id, meta):
        """The runner (or a local prefill path) reports that decode ``slot``'s request ``req_id`` has its prompt's
        context K/V in the drafter caches (``meta`` = the payload group's header entry: first_pos, n_tokens, ...);
        ``meta`` None = the request arrived without them (it decodes with no drafts)."""
        slot = int(slot)
        if meta is None:
            self.context.pop(slot, None)
            return
        first = int(meta.get("first_pos", 0))
        if first != 0 and not self._warned_window:
            # a windowed transport (QWEN36_DFLASH2_CONTEXT_WINDOW on P) ships positions >= first_pos only; the device
            # drafter's first block step reads [total + 8 - window, total + 7] (cur_pos = P + 7): fine iff the device
            # window is applied and the shipped tail covers it, else the drafter reads stale blocks
            window = getattr(self.head, "window", None)
            total = first + int(meta.get("n_tokens", 0))
            if window is None or first > max(0, total - int(window)):
                self._warned_window = True
                logger.warning(
                    f"[spec] dflash2 context of slot {slot} starts at position {first} (prompt {total} tokens) but the "
                    f"device drafter attends {'the whole context (QWEN36_DFLASH2_DEVICE_WINDOW=0)' if window is None else f'the last {window} positions'}: "
                    "positions below hold stale blocks (drafts degrade; the committed stream is unaffected)"
                )
        self.context[slot] = (req_id, dict(meta))

    def has_state(self, slot) -> bool:
        return slot in self.context

    def drop_state(self, slot):
        self.context.pop(slot, None)

    def drafts_enabled(self, slot, active=None) -> bool:
        return slot in self.context

    def drop_stale(self, w, sp):
        pass  # the context of a slot is replaced by the next admission's report (note_context) / dropped with it

    def note_plain(self):
        pass

    # --- per step ---
    def catchup(self, w, sp, last, pos, pt):
        pass  # no hidden row: the context is complete once the first verify step's commit ran (module docstring)

    def after_verify(self, vs, sp, accepts, pos_before, argmax_rows=None, active=None):
        self.head.commit(vs.plan, pos_before)

    def draft(self, w, k, last, positions, pad, pt, active=None):
        self.head.set_page_table(w, pt)
        drafts7, _, _ = self.head.draft(w, last, positions, pad=pad)
        return [list(d[:k]) for d in drafts7]


class HybridHeads:
    """The two drafter modules of the hybrid policy (``make_drafter`` -> ``_HybridDrafter``)."""

    def __init__(self, mtp_head, dflash2_drafter):
        assert hasattr(mtp_head, "select_hidden") and hasattr(mtp_head, "keep_current"), type(mtp_head).__name__
        assert hasattr(dflash2_drafter, "commit") and hasattr(dflash2_drafter.cfg, "target_layer_ids")
        self.mtp = mtp_head
        self.dflash2 = dflash2_drafter


class _HybridDrafter:
    """Both drafters behind the decoder's interface; the ladder's ``drafter_for(plan)`` (passed as ``active``) picks
    who drafts, both states are kept current every step (module docstring). ``widths_of(name)`` = the ladder widths
    each drafter drafts at (set by the decoder before ``build_step_buffers``: the block step is built only for the
    DFlash2 buckets, the MTP step for every width because the catch-up runs at the current plan's width)."""

    name = "hybrid"
    post_name = "post"  # commit (+ keep-current) (+ select): the per-step ops between the verify commit and the draft

    def __init__(self, heads: HybridHeads, ladder):
        self.mtp = _MtpDrafter(heads.mtp)
        self.df2 = _DFlash2Drafter(heads.dflash2)
        self.ladder = ladder
        self.prev_active = None  # the drafter of the previous spec step (None after a plain step / at start)
        self.stats = {"commit_ms": 0.0, "keep_ms": 0.0, "select_ms": 0.0, "keep_steps": 0, "select_steps": 0}
        self.last_times = {}

    def verify_kwargs(self):
        kw = dict(self.df2.verify_kwargs())
        kw.update(self.mtp.verify_kwargs())
        return kw

    def build_step_buffers(self, widths, page_tables):
        self.mtp.build_step_buffers(widths, page_tables)  # every width: the catch-up step runs at the plan's width
        self.df2.build_step_buffers(self.ladder.widths_for("dflash2"), page_tables)

    def bind_plan(self, plan):
        self.mtp.bind_plan(plan)
        self.mtp.head.bind_keep_plan(plan)
        self.df2.bind_plan(plan)

    def compile_width(self, w):
        self.mtp.compile_width(w)
        if w in self.df2.head.sb:
            self.df2.compile_width(w)

    def compile_plan(self, vs):
        self.df2.compile_plan(vs)
        self.mtp.compile_plan(vs)
        self.mtp.head.compile_keep(vs.plan)

    def capture_width(self, w):
        self.mtp.capture_width(w)
        if w in self.df2.head.sb:
            self.df2.capture_width(w)

    def capture_plan(self, vs):
        self.df2.capture_plan(vs)  # reads out_aux
        self.mtp.head.capture_keep(vs.plan)  # reads out_hidden

    def release(self):
        self.mtp.release()
        self.df2.release()

    # --- per-slot state ---
    def has_state(self, slot) -> bool:
        return self.mtp.has_state(slot)  # the catch-up list: fresh slots whose MTP hidden row arrived

    def drop_state(self, slot):
        self.mtp.drop_state(slot)
        self.df2.drop_state(slot)

    def note_context(self, slot, req_id, meta):
        self.df2.note_context(slot, req_id, meta)

    def drafts_enabled(self, slot, active=None) -> bool:
        return self.df2.drafts_enabled(slot) if active == "dflash2" else True

    def drop_stale(self, w, sp):
        self.mtp.drop_stale(w, sp)

    def note_plain(self):
        self.prev_active = None  # a plain step wrote neither drafter's state: the next MTP step keeps current

    # --- per step ---
    def catchup(self, w, sp, last, pos, pt):
        self.mtp.catchup(w, sp, last, pos, pt)  # the head's KV at P_s - 1, whichever drafter is active

    def after_verify(self, vs, sp, accepts, pos_before, argmax_rows=None, active=None):
        """DFlash2 context commit (always); MTP keep-current unless the MTP chain of the previous step covered the
        committed rows (an MTP step right after an MTP step); the MTP hidden select when the MTP head drafts."""
        t0 = time.perf_counter()
        self.df2.after_verify(vs, sp, accepts, pos_before)
        t1 = time.perf_counter()
        times = {"commit_ms": 1e3 * (t1 - t0)}
        if active != "mtp" or self.prev_active != "mtp":
            self.mtp.head.keep_current(vs.plan, [int(t) for t in argmax_rows.tolist()])
            t2 = time.perf_counter()
            times["keep_ms"] = 1e3 * (t2 - t1)
            self.stats["keep_steps"] += 1
            t1 = t2
        if active == "mtp":
            self.mtp.after_verify(vs, sp, accepts, pos_before)
            times["select_ms"] = 1e3 * (time.perf_counter() - t1)
            self.stats["select_steps"] += 1
        for key, v in times.items():
            self.stats[key] += v
        self.last_times = times
        self.prev_active = active

    def draft(self, w, k, last, positions, pad, pt, active=None):
        if active == "dflash2":
            return self.df2.draft(w, k, last, positions, pad, pt)
        return self.mtp.draft(w, k, last, positions, pad, pt)


class RowChunkedProjector:
    """``DFlash2ContextProjector`` over row chunks (the P-side prefill hook's projector): the drafter's projection
    GEMMs use the small-M 1D matmul configs (every M tile on each core), whose static circular buffers exceed L1 at
    the 2048-row prefill chunk / masked buckets >= 512 (2.6 MB of CBs at 2048 rows; logs/spec_serving_df2a.log). The
    rows are independent (fc -> hidden_norm -> K/V GEMM -> k_norm -> RoPE per row), so the wrapper slices the
    replicated aux tensor into <= ``rows`` rows (tile aligned; ``QWEN36_DFLASH2_PROJECT_ROWS``, default 256, the
    largest size the drafter harness ran) and concatenates the per-layer host K/V it gets back."""

    def __init__(self, projector, rows=None):
        self.p = projector
        self.rows = int(rows if rows is not None else os.environ.get("QWEN36_DFLASH2_PROJECT_ROWS", "256"))
        if self.rows % 32 != 0 or self.rows <= 0:
            raise ValueError(f"QWEN36_DFLASH2_PROJECT_ROWS={self.rows}: expected a positive multiple of 32")
        self.head_dim = projector.head_dim
        self.n_kv_heads = projector.n_kv_heads

    @staticmethod
    def _cat(parts):
        if len(parts) == 1:
            return parts[0]
        n_layers = len(parts[0])
        return [(torch.cat([p[j][0] for p in parts]), torch.cat([p[j][1] for p in parts])) for j in range(n_layers)]

    def project(self, aux, positions):
        """Host rows [N, 5*dim] -> per layer (K, V) host bf16 [N, n_kv_heads, HD] (DFlash2ContextProjector.project)."""
        aux = torch.as_tensor(aux)
        pos = torch.as_tensor(positions).reshape(-1)
        N = int(aux.shape[0])
        return self._cat(
            [
                self.p.project(aux[a : min(N, a + self.rows)], pos[a : min(N, a + self.rows)])
                for a in range(0, N, self.rows)
            ]
        )

    def project_device(self, aux_rep, positions):
        """The REPLICATED device rows [1,1,N,5*dim] (left alone) -> per layer (K, V) host bf16 [N, n_kv_heads, HD]."""
        import ttnn

        N = int(aux_rep.shape[-2])
        pos = torch.as_tensor(positions).reshape(-1)[:N]
        if N <= self.rows:
            return self.p.project_device(aux_rep, pos)
        parts = []
        W = int(aux_rep.shape[-1])
        for a in range(0, N, self.rows):
            b = min(N, a + self.rows)
            sl = ttnn.slice(aux_rep, (0, 0, a, 0), (1, 1, b, W))
            parts.append(self.p.project_device(sl, pos[a:b]))
            ttnn.deallocate(sl)
        return self._cat(parts)


def make_drafter(head, ladder=None):
    """The adapter for a drafter module: an MTPHead or a DFlash2Drafter (duck-typed on their distinctive API), or the
    ``HybridHeads`` pair (needs the ladder for the per-plan choice)."""
    if isinstance(head, HybridHeads):
        assert ladder is not None and ladder.drafter == "hybrid", "HybridHeads need a hybrid ladder"
        return _HybridDrafter(head, ladder)
    if hasattr(head, "select_hidden") and hasattr(head, "pending_rows"):
        return _MtpDrafter(head)
    if hasattr(head, "commit") and hasattr(head, "cfg") and hasattr(head.cfg, "target_layer_ids"):
        return _DFlash2Drafter(head)
    raise TypeError(f"unknown drafter module {type(head).__name__}")


# ================================================================================================ the decoder
class SpecDecoder:
    def __init__(self, model, head, ladder: ss.Ladder, bmax: int, num_blocks_pt: int, log_every: Optional[int] = None):
        """model: the Qwen36Model (KV caches allocated); head: its drafter -- the MTPHead (prefill buckets may already
        be compiled by the P-side installer) or the DFlash2Drafter (weights + context caches built with the KV caches);
        the draft-step buffers of the ladder's widths are built here. num_blocks_pt: the served block-table width
        (max_num_blocks_per_req, a multiple of 8). Allocates every persistent buffer (call before ANY trace capture)."""
        self.model = model
        self.drafter = make_drafter(head, ladder)
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
        self.drafter.build_step_buffers(ladder.widths, zeros_pt)
        self.steps = {}
        for plan in ladder.all_plans:  # both modes' plans (the hybrid context rule's long ladder included)
            vs = VerifyStep(model, plan.w, plan.T, zeros_pt[: plan.w], pad_safe=True, **self.drafter.verify_kwargs())
            assert vs.plan.attn_mode == "batched", vs.plan.attn_mode
            self.drafter.bind_plan(vs.plan)
            self.steps[plan] = vs
        self.log_every = int(os.environ.get("QWEN36_SPEC_LOG_EVERY", "50")) if log_every is None else int(log_every)
        self._compiled = self._captured = False
        self.n_steps = 0
        self.post_key = f"{self.drafter.post_name}_ms"
        self.acc = {"verify_ms": 0.0, self.post_key: 0.0, "draft_ms": 0.0, "catchup_ms": 0.0, "migrate_ms": 0.0}
        self.acc_tokens = 0
        self.acc_users = 0
        self.acc_accepted = 0
        self.n_no_context = 0  # dflash2: user-steps drafted as padding (no context K/V for the request)
        self.n_steps_by_drafter = {}
        ctx_rule = (
            f"; context rule: DFlash2 while the longest live context <= {ladder.dflash2_max_ctx} (hysteresis "
            f"{ladder.ctx_hysteresis}), long ladder {[f'{p}:{ladder.drafter_for(p, long=True)}' for p in ladder.long_plans]}"
            if ladder.has_ctx_rule
            else ""
        )
        logger.info(
            f"[spec] decoder ({self.drafter.name}): ladder {[f'{p}:{ladder.drafter_for(p)}' for p in ladder.plans]} k_max={ladder.k_max} "
            f"bmax={self.bmax} page table {self.nb} blocks; fractured plans {[str(p) for p in ladder.fractured_plans]}; "
            f"{len(self.steps)} verify plans + {len(ladder.widths)} draft widths allocated in {time.perf_counter() - t0:.1f}s{ctx_rule}"
        )

    # ------------------------------------------------------------------------------------------ warm-up
    def compile(self):
        """Phase 1 (no trace captured yet): every program the loop runs -- draft steps per width, verify bodies, the
        drafter's per-plan op (MTP select / DFlash2 commit), qkv_prev migrations between every plan pair (mutates the
        KV / GDN state of slots 0..w-1: warm-up)."""
        if self._compiled:
            return
        t0 = time.perf_counter()
        for w in self.ladder.widths:
            self.drafter.compile_width(w)
        for plan, vs in self.steps.items():
            t1 = time.perf_counter()
            vs.compile()
            self.drafter.compile_plan(vs)
            logger.info(f"[spec] verify {plan} R={vs.plan.R} compiled in {time.perf_counter() - t1:.1f}s")
        for a, va in self.steps.items():
            for b, vb in self.steps.items():
                if a != b:
                    vb.plan.compile_migration(va.plan)
        # the sparse hidden-row upload + one padded draft step (the MTP catch-up) run the same programs as a draft step
        self._compiled = True
        logger.info(f"[spec] every speculative program compiled in {time.perf_counter() - t0:.1f}s")

    def capture(self):
        """Phase 2: capture the draft-step and verify traces (all programs compiled); the DFlash2 commit trace of a
        plan right after its verify trace (it reads the verify output buffer's fixed address)."""
        assert self._compiled, "compile() first"
        if self._captured:
            return
        t0 = time.perf_counter()
        for w in self.ladder.widths:
            self.drafter.capture_width(w)
        for plan, vs in self.steps.items():
            vs.capture()
            self.drafter.capture_plan(vs)
        self._captured = True
        logger.info(
            f"[spec] {len(self.steps)} verify + {len(self.ladder.widths)} draft traces captured in {time.perf_counter() - t0:.1f}s"
        )

    def release(self):
        for vs in self.steps.values():
            vs.release()
        self.drafter.release()

    # ------------------------------------------------------------------------------------------ per step
    def hold_info(self) -> ss.HoldInfo:
        return self.state.hold_info()

    def note_context(self, slot, req_id, meta):
        """Admission report (the runner, from the payload's ``kv_groups`` metadata; or a local prefill path): the
        drafter state of ``req_id`` in decode ``slot`` -- no-op for the MTP head."""
        self.drafter.note_context(int(slot), req_id, meta)

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
        sp = self.state.begin_step(
            rows, dr, eligible, flush=flush, has_state=self.drafter.has_state, ctx_lens=pos_all[: self.bmax]
        )
        if sp.mode == "plain":
            self.drafter.note_plain()
            return None
        plan, w, k = sp.plan, sp.plan.w, sp.plan.k
        active = sp.drafter  # the drafter of this plan (Ladder.drafter_for; the single drafter outside hybrid mode)
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
            t0 = time.perf_counter()
            self.drafter.catchup(w, sp, last, pos, pt)
            times["catchup_ms"] = 1e3 * (time.perf_counter() - t0)
        # a slot re-used by a new request whose drafter state never arrived: nothing to consume; a stale row of a
        # previous occupant is dropped when the owner changes (SpecServingState resets the slot -> not fresh here)
        self.drafter.drop_stale(w, sp)

        t0 = time.perf_counter()
        grid_tokens = self.state.grid_tokens(sp, last)
        argmax_rows = vs.run(grid_tokens, pos, sp.accept_prev, page_table=pt)
        accepts, committed = self.state.commit(sp, argmax_rows, grid_tokens)
        t1 = time.perf_counter()
        times["verify_ms"] = 1e3 * (t1 - t0)

        next_drafts = [[] for _ in range(w)]
        if k >= 1:
            self.drafter.after_verify(vs, sp, accepts, pos, argmax_rows=argmax_rows, active=active)
            t2 = time.perf_counter()
            times[self.post_key] = 1e3 * (t2 - t1)
            new_last = [committed[s][-1] if sp.live[s] else 0 for s in range(w)]
            new_pos = [pos[s] + accepts[s] + 1 if sp.live[s] else 0 for s in range(w)]
            # users the drafter has no state for (a request whose context K/V never arrived) are padding in the draft
            draft_pad = [pad[s] or not self.drafter.drafts_enabled(s, active) for s in range(w)]
            self.n_no_context += sum(1 for s in range(w) if sp.live[s] and draft_pad[s])
            drafted = self.drafter.draft(w, k, new_last, new_pos, draft_pad, pt, active=active)
            next_drafts = [list(drafted[s]) if not draft_pad[s] else [] for s in range(w)]
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
            drafter=active,
            long=sp.long,
        )

    # ------------------------------------------------------------------------------------------ metrics
    def _account(self, sp, committed, accepts, times):
        self.n_steps += 1
        self.n_steps_by_drafter[sp.drafter] = self.n_steps_by_drafter.get(sp.drafter, 0) + 1
        n_live = sum(sp.live)
        self.acc_users += n_live
        self.acc_tokens += sum(len(c) for c in committed)
        self.acc_accepted += sum(accepts)
        for key, v in times.items():
            self.acc[key] = self.acc.get(key, 0.0) + v
        if self.log_every and self.n_steps % self.log_every == 0:
            st = self.state.stats
            n = self.log_every
            hybrid = ""
            if self.drafter.name == "hybrid":
                hs = self.drafter.stats
                hybrid = (
                    f" [hybrid: commit {hs['commit_ms'] / n:.2f} keep {hs['keep_ms'] / n:.2f} ({hs['keep_steps']} steps) "
                    f"select {hs['select_ms'] / n:.2f} ({hs['select_steps']} steps) ms/step; steps by drafter "
                    f"{self.n_steps_by_drafter}; switches {st['drafter_switches']}"
                    + (
                        f"; ctx rule: {'long' if sp.long else 'short'} mode, max ctx {sp.ctx_max}, "
                        f"{st['ctx_switches']} mode changes, {st['ctx_flushes']} own flushes"
                        if self.ladder.has_ctx_rule
                        else ""
                    )
                    + "]"
                )
                for key in ("commit_ms", "keep_ms", "select_ms"):
                    hs[key] = 0.0
                hs["keep_steps"] = hs["select_steps"] = 0
            logger.info(
                f"[spec] step {self.n_steps}: w_grid={sp.w_grid} plan={sp.plan} drafter={sp.drafter} live={n_live} "
                f"flush={sp.flush} | last {n} steps: {self.acc_tokens / max(1, self.acc_users):.2f} tok/user/step "
                f"({self.acc_accepted / max(1, self.acc_users):.2f} accepted), per step verify "
                f"{self.acc['verify_ms'] / n:.1f} + {self.drafter.post_name} {self.acc[self.post_key] / n:.1f} + draft "
                f"{self.acc['draft_ms'] / n:.1f} ms (catch-up {self.acc['catchup_ms'] / n:.2f}, migrate "
                f"{self.acc['migrate_ms'] / n:.2f}) | totals: spec {st['spec_steps']} plain {st['plain_steps']} "
                f"flushes {st['flushes']} plan changes {st['plan_changes']} migrations {st['migrations']}"
                + (f" no-context user-steps {self.n_no_context}" if self.drafter.name in ("dflash2", "hybrid") else "")
                + hybrid
            )
            self.acc = {key: 0.0 for key in self.acc}
            self.acc_tokens = self.acc_users = self.acc_accepted = 0
