# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``MotifGenerator``: the Motif-3 TT runtime behind the vLLM bridge (``generator_api.MotifGenerator``; WAVE_A_REVIEW
GEN-1..7, design §2.3.10, §5.1). Default ``MOTIF3_GENERATOR_CLASS`` =
``models.demos.motif3.tt.generator:MotifGenerator``.

Lifecycle (vllm-tt-plugin call order, ``generator_api.MotifGenerator`` docstring):

1. ``create(hf_config=, mesh_device=, settings=)`` (GEN-1): ``model_config.require_l1_small(mesh)``, then
   ``MotifTTConfig.from_settings`` (``<weights>/config.json``, ``num_layers``, ``max_model_len``, ``max_batch = 32``, KV
   dtype, fabric from the device, the feature settings: span cap, KV-R, ``spec_tokens``), then the weights
   (:class:`~models.demos.motif3.tt.model.MotifModel`: TT cache where converted, else the HF checkpoint, lazily; with
   ``spec_tokens = 1`` also the MTP layer, part ``L53``; ``cache`` = ``settings.tt_cache_policy``,
   ``MOTIF3_TT_CACHE_POLICY``: ``"write"`` also writes and marks the parts it converts). No KV pool yet.
2. ``allocate_kv_cache(num_blocks=, block_size=, num_layers=)`` (GEN-2): ``cfg.set_kv_geometry`` with the plugin's
   values (4129 x 64 for the serving defaults), one ``ttnn.empty`` + ``ttnn.fill(0)`` cache per layer, plus the MTP
   layer's cache (``pool.mtp``) with speculation.
3. ``warmup_prefill`` (every ``(path, bucket)`` of :meth:`MotifGenerator.prefill_shapes`, eager, writes nothing;
   with packed prefill also every packed shape of :meth:`MotifGenerator.packed_shapes`) ->
   ``warmup_decode(enable_trace=False)`` (stages the persistent inputs of every decode path for width W, one eager
   all-inactive step each) -> ``warmup_decode(enable_trace=True)`` (captures each path's trace, exception-safe; with
   ``spec_verify="auto"`` the T32-spec trace, then the T64 trace).
   Capture refuses to run before every prefill shape was compiled, and after the capture a prefill chunk of a shape the
   warmup did not compile is refused (a program compiled after capture can corrupt the trace: plugin
   ``model_runner.py:3735-3745``; features design D12 / G7); a packed pass of an unwarmed shape runs as solo chunks
   instead (never refused, never compiled); a decode path is never staged after a capture (a buffer allocated after a
   capture and kept across replays may be overwritten by them).
4. ``prefill_forward_batch`` (features design §3.7; README §15) and ``decode_forward`` (GEN-3), below.
5. ``release_traces`` / ``close`` (GEN-6). ``release_lane`` is a no-op (no lane-owned device state: the paged cache is
   the only cross-call state).

Prefill (resumed / chunked; ``supports_resumed_prefill``). One call per plugin step with every row of the step:

* **Plan** (host, ``tt/prefill_plan.py`` through ``cfg.plan_prefill_row``): row ``[s, e)`` -> write floor ``w0 =
  floor(s / bs) bs``, compute floor ``c0 = floor(s / A) A`` (0 below the 128-row SWA tail), chunks with ``A``-aligned
  starts of power-of-two buckets ``<= cfg.max_prefill_span`` (8192; ``MOTIF3_PREFILL_MAX_BUCKET``): a chunk at 0 is
  **sp0** (the draft-1 path), every other chunk **sp1** (reads the paged cache). Spans longer than the cap are split
  inside the call in every mode (design D8: no 16K / 32K buckets, no bucket cliff). An sp1 chunk never uses a bucket
  above ``attention.max_sp1_bucket(cfg)`` (the SWA square ``[tail ‖ chunk]`` must fit ``max_model_len``): such a row is
  re-planned with that cap.
* **Order** (``prefill_plan.order_prefill_requests``, D7): writer-first, stable in input order. vLLM caches full blocks
  when it allocates them, so a row admitted later in the step can hit blocks another row of the same call writes.
* **Every host check runs before the first device op**: token ids, ``end <= max_model_len``, the page table
  (:func:`check_prefill_page_table`: every entry the row uses is a distinct block id in ``[1, num_blocks)``; the fill
  kernel does no bounds check), the plan, every chunk's tables (``attention.chunk_host_tables``, which cross-checks
  them), and after the decode capture that every chunk's ``(path, bucket)`` was warmed. A refused call therefore
  changes nothing.
* **Per chunk**: ``model.chunk_inputs`` (``PrefillChunkInputs``: fill table ``-1`` = skip for shared blocks below
  ``w0`` and pure-padding blocks; sp1 SDPA table, start, offset RoPE rows, SWA tail bounds; shared by every layer) ->
  ``model.prefill_chunk`` (embedding -> 53 layers) -> on the last chunk only the LM head at the chunk-local row
  ``e - 1 - a`` (tensor-args slice) and the host logits -> with the MTP layer, its **KV-only fill** (D9):
  ``head.stream_mean_norm(X)`` and the next tokens ``t_{p+1}`` (``mtp.mtp_next_tokens``: known tokens, except the
  row's last position, which takes the host argmax of the returned logits) -> ``mtp.fill_kv_prefill(chunk=)`` into
  ``pool.mtp`` through the same fill table. Every chunk tensor is freed before the next chunk.
* Returns ``[B, vocab]`` host logits (position ``end - 1`` of each row) in input order. ``prefill_forward(r)`` is
  ``prefill_forward_batch([r])[0]``.

Packed prefill (P5, ``settings.packed_prefill`` / ``MOTIF3_PACKED_PREFILL``; docs/p5_t64/P5_T64_DESIGN.md §3; the
switch :attr:`MotifGenerator.packed_prefill`). The call runs as **passes** (``PrefillBatchPlan.passes``, in order;
``prefill_plan.plan_prefill_passes``): a chunk of ``r`` rows is a segment of ``S`` rows (the smallest of
``cfg.pack_seg_buckets`` / ``cfg.pack_sp1_seg_buckets`` ``>= r``); the segments whose dependencies have run (a row's
previous chunk; for an sp1 segment every writer of its row's read-only prefix ``[0, w0)``, review edit R-E1) are
grouped as ``pk0`` (start 0, by ``S``) or ``pk1`` (one common start ``a``, by ``(S, a)``) and cut into packed passes of
``B`` segments (a power of two, dummies fill it), ``T = B * S`` rows (a compiled bucket), where the cost model prefers
it; the rest run **solo** (``_run_chunk``, bitwise the chunk path above). A packed pass (``_run_packed``): its host
tables (``attention.packed_host_tables``: every segment's slice is its chunk's tables re-bucketed to ``S``) ->
``model.chunk_inputs`` -> ``model.prefill_chunk`` at ``T`` (only the attention treats the segments apart) -> the LM
head once per segment that ends its row -> one MTP KV-only fill of the whole pass (each segment's next tokens, the
row's argmax stand-in where it ends). Every pass's tables are built before the first device op. After the decode
capture a packed pass whose shape (``("pk0", T, S)`` / ``("pk1", T, S, tails)``, review edit R-E2) was not warmed
runs as one solo pass per segment (``packed_solo_fallbacks``, logged once per shape). A packed plan that fails its own
checks (a planner bug) never refuses the call: the call runs with the packing-off passes (``packed_plan_errors``,
logged per call). Packing off: one solo pass per chunk in writer-first row order, the device-op sequence of the per-row
path.

Decode paths. A path is ``(kind, KV-write mode)`` with its own persistent inputs and trace (:class:`DecodePath`):

* **plain** (``decode_forward`` on a launch without speculation): embedding -> 53 layers -> LM head; the logits
  ``[32, 220160]`` are assembled on the host from the 32 vocab shards (``logits_to_host``, a fresh tensor every step).
  Mode ``row`` (draft 1, bitwise: the attention's own 8-lane update) or ``all`` (**KV-R**, on with prefix caching: one
  ``tt/kv_write.DecodeKVWrite`` shared by every layer, every decode KV write on all 32 chips, so a prefix hit on a
  block another DP row decode-wrote reads valid KV; README §16).
* **spec** (``decode_forward_spec``; every decode step of a speculating launch, ``decode_forward`` included): the
  **T32-spec** step ``MotifModel.decode_spec`` (features design §3.8.1, D10): the main layers with the split KV write
  (``row_split``, or ``all_split`` = KV-R + split, the production mode), the LM head, the main argmax ``a`` and the MTP
  layer on every lane (``m``, its own cache written at the same lanes and positions). Outputs: ROW_MAJOR logits (read
  only when wanted, ~2 ms), ``a`` and ``m`` (128 B each).
* **wide** (``decode_forward_spec`` with ``spec_verify`` "wide" / "auto"; docs/p5_t64/P5_T64_DESIGN.md §4): the
  **T64** step ``MotifModel.decode_wide``: per DP row 16 rows ``[8 anchors at n | the same lanes' 8 drafts at n +
  1]`` (each draft on its owner's DP row with the owner's page-table row), ``DecodeKVWrite(rows=64, gather="split")``,
  FlashMLA option A'' (bitwise the T32 rows), the M = 64 MoE / LM heads. Outputs ``a`` / ``m`` ``[64]`` (split order:
  ``[l]`` = lane ``l``'s anchor row, ``[32 + l]`` its draft row) and, in ``wide`` only, the anchors' ROW_MAJOR logits.

T64 verify (``spec_verify`` = ``cfg.spec_verify``, ``MOTIF3_SPEC_VERIFY``; design §2.2-§2.4, §4.5; X2-X4). The launch's
serving decode paths (:attr:`MotifGenerator.serving_paths`):

* ``packed`` (default): the T32-spec trace alone (packed verify below; drafts without an idle lane take an overflow
  pass);
* ``auto``: the T32-spec trace (with the device sampler) AND the T64 trace (argmax only), each staged, run eagerly once
  and captured once at warmup (T32 first). Each step is routed on the host (``verify_plan.choose_verify_kind``):
  ordinary steps, steps that want logits or sampling, and verify steps whose drafts all fit idle partner lanes run on
  T32 (one replay); every other verify step runs as ONE T64 replay (never an overflow pass for bridge traffic: its
  verify steps carry neither logits nor sampling, review edit R-E6). T64 rows equal T32 rows bitwise, so the switch is
  lossless;
* ``wide``: the T64 trace alone (ordinary steps too, with idle draft rows; it then also carries the anchors' logits and
  the device sampler on them).

:meth:`MotifGenerator.drafts_all_lanes` (review edits R-E3, R-E9) tells the bridge when every live lane may draft
(``verify_plan.drafts_all_lanes``: ``wide`` always, ``auto`` from ``c*`` live lanes, ``packed`` never). The F3N rules
(design §2.3) are invariants here: R1 -- ``create`` / the constructor / the capture refuse ``wide`` / ``auto`` unless
``ring_gather="safe"`` (``native`` and ``lean`` refused, R-E5) on the config and the model's ``MotifCCL`` (any other
launch with ``native`` / ``lean`` logs a warning, X3), and ``auto`` with ``router_logits="exact_fp32"`` unless the MoE
runs its exact router at the T64 row count and the config lists it (R-E7); R2 -- every
decode path is staged and run eagerly before the first capture (prefill shapes compiled first); R3 -- every persistent
input (the T64 path's ``DecodeKVWrite(rows=64)`` and its A'' group inputs included) is allocated by ``_stage_path``,
which refuses after a capture; R4 -- every decode step ends in a blocking read of the outputs it replayed, and a replay
of one trace while another trace's outputs are unread is refused; R5 -- serving never releases or re-captures.

Speculative decode (``decode_forward_spec``, features design §3.8.2; host plan :func:`plan_spec_step`, before any
device op). The ``SpecDecodeBatch`` is in owner-lane order (anchor ``t`` at ``n`` per active lane, at most one draft
``d`` for ``n + 1``); lanes with position ``-1`` are idle. Each drafted owner borrows an idle *partner* lane
(``kv_write.assign_partner_lanes``: its own DP row first, any row under KV-R): pass 1 runs the anchors on their lanes
(call A) and the drafts on the partners at ``n + 1`` with the owner's page-table row (call B, after A, both before
FlashMLA: two users of one tile in one update call race, gate G12) -> ``a0 = a[owner]``, ``m0 = m[owner]``, ``a1 =
a[partner]``, ``m1 = m[partner]``. Drafts without an idle lane (a batch that grew after the drafts were proposed) run in
**pass 2**, a second replay of the same trace with each such draft on its own lane at ``n + 1`` (pass 1 wrote the anchor
at ``n``) and every other lane inactive. A rejected draft's KV at ``n + 1`` (main and MTP) is overwritten by the next
step's anchor before anything reads it. Packed verify is lossless because a lane carries no device state besides the
paged KV and every op of the step is row-local: a draft on a partner lane computes exactly what an ordinary step on the
owner's lane at ``n + 1`` computes (gate G12: FlashMLA bitwise under lane relocation; the full-model probe is in
``tests/test_spec_decode_device.py``).

Host cost of a spec step (measured at 53 layers, ``OMP_WAIT_POLICY=PASSIVE``): the outputs of every pass are read
with ONE blocking read per step (``a`` / ``m`` into persistent host staging, non-blocking; the logits read, or the last
``m`` read, blocks): a blocking read ends in a full-mesh finish (an event round trip to all 32 chips), which cost
~0.7 ms each when ``a``, ``m`` and the logits were read one after another. An overflow pass is enqueued right behind
pass 1 (its inputs do not depend on pass 1's outputs; command-queue order keeps pass 1's reads ahead of pass 2's
replay). Verify step 89.4 ms vs 88.4 ms for the plain KV-R step; ordinary spec step (with the logits read) 90.1
ms. Without ``OMP_WAIT_POLICY=PASSIVE`` torch's spinning OpenMP workers stall the input copies after the step's host
planning (+4.7 ms per verify step measured): ``create`` logs a warning.

Device sampling (``docs/sampling/DEVICE_SAMPLER.md`` §6; the bridge turns it on for ``sample_on_device_mode:
"decode_only"`` through :meth:`MotifGenerator.enable_device_sampling`, before the decode warmup): the exact sampler
``tt/sampling.MotifDeviceSampler`` runs at the end of EVERY replay of the trace that serves sampled steps: the plain
trace, the T32-spec trace, or with ``spec_verify="wide"`` the T64 trace (on its anchor rows). In ``auto`` the T64 trace
holds no sampler: sampled steps are ordinary steps and run on T32 (PS-1 keeps them out of verify steps). Two decode
traces are safe under F3N rules R1-R5 (``docs/p5_t64/f3.md`` §6-§7: F3 was the TP-ring all-gather race, closed by
``ring_gather="safe"``, not a trace effect). The plain step samples the TILE logits (``model.decode(return_streams=
True)`` -> ``head.forward_decode`` -> ``logits_rm`` for the host + ``sampler.sample``), the spec step samples its
ROW_MAJOR logits ``rm`` (one tilize; ``MotifModel.decode_spec`` frees its TILE logits), the ``wide`` T64 step its
anchors' ROW_MAJOR logits (the same ``[1, 1, 32, 6880]``). The sampler's outputs (``tokens`` / ``info``) are
extra trace outputs (``DecodePath.so``), freed in :meth:`MotifGenerator.release_traces`. A device-sampled step
(:meth:`MotifGenerator.decode_forward_sampled`, or :meth:`MotifGenerator.decode_forward_spec` with ``sampling``)
writes the lane parameters (host compare, device write only on change) and the RNG counters of the lanes' positions,
replays, reads the 1 KB ``info`` and re-samples the flagged active lanes on the host from the logits (read only
then; ~1 % of 32-lane steps at T = 1.0 / top-p 0.95). Host-sampled steps (penalties, min_p, structured output,
top-N logprobs: the plugin's per-step routing) keep the logits contract of ``decode_forward`` from the same trace (the
sampler's 1.36 ms run is wasted there). Verify steps keep the argmax path (PS-1 keeps sampled rows out of them).

Lanes and inactive lanes (README §2): the bridge maps vLLM rows / state slots onto lanes (``LaneMap``, ``slot_remap``);
this class only sees lane-ordered tensors. Position ``-1`` marks an inactive lane: ``cur_pos = -1`` (no KV write,
FlashMLA skips it), rot index 0, an all-zero page-table row, token 0; its logits row is garbage (ignored).

Persistent decode inputs (per path, allocated before the first capture; per DP row): ``tokens [4, 8]`` uint32,
``rot_idxs [1, 32]`` uint32 and either ``cur_pos [8]`` int32 + ``page_table [8, W]`` int32 (the draft-1 plain ``row``
path) or the path's ``DecodeKVWrite`` inputs (its ``cur_pos`` / ``page_table`` for FlashMLA and the update-call inputs).
Every step rewrites them with ``ttnn.copy_host_to_device_tensor`` (the bridge sends ``reload_inputs=True``; the
``DecodeKVWrite`` skips inputs whose values did not change).

Import rule (design §2.1): stdlib, torch, ttnn and the motif3 ``tt/`` modules only (the bridge's host suite imports this
module device-free: ``test_real_generator_class_imports_device_free``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch

import ttnn

from . import generator_api as api
from . import prefill_plan as PP
from . import verify_plan as VP
from .attention import (
    ChunkHostTables,
    PackedHostTables,
    chunk_host_tables,
    max_sp1_bucket,
    packed_host_tables,
    warmup_chunk_host_tables,
    warmup_packed_host_tables,
)
from .embedding import check_token_ids, decode_token_rows
from .kv_write import (
    DecodeKVWrite,
    KVWriteStep,
    allows_cross_row_partners,
    assign_partner_lanes,
    check_kv_write_step,
    check_mode as _check_mode,
    is_replicated,
    is_split,
    lanes_per_call_for,
)
from .lm_head import HostShardReader
from .model import LazySource, MotifKVPool, MotifModel, normalize_cache_policy
from .model_config import ROUTER_EXACT_FP32_DECODE_ROWS, MotifTTConfig, require_l1_small
from .moe import EXACT_ROUTER_DECODE_ROWS
from .mtp import mtp_next_tokens
from .rope import dp_row_mapper, positions_to_rot_idxs, shard_lanes
from .sampling import MotifDeviceSampler, SampleResult, SamplerOutput

# A prefill program-set key: solo (path "sp0" | "sp1", bucket); packed ("pk0", T, S) | ("pk1", T, S, "shared" |
# "distinct") (prefill_plan.PrefillPass.shape; MotifTTConfig.packed_prefill_shapes()).
PrefillShape = Tuple[Any, ...]


def _log_default(msg: str) -> None:
    print(f"[motif3.generator {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prefill_page_table_host(page_table: torch.Tensor, entries: int, seq_len: int, block_size: int) -> torch.Tensor:
    """The draft-1 prefill fill's page table ``[1, entries]`` int32 (``entries =
    cfg.prefill_page_table_entries(bucket)``): the request's first ``cdiv(seq_len, block_size)`` block ids, **zero
    (the null block) after them** whatever the caller passed there. Kept for single-shot reference runs
    (``MotifModel.prefill``); serving prefill uses the chunk fill tables of ``prefill_plan.fill_table``, which skip
    (``-1``) pure-padding blocks instead."""
    own = min(-(-int(seq_len) // int(block_size)), int(entries), int(page_table.shape[0]))
    pt = torch.zeros(1, int(entries), dtype=torch.int32)
    pt[0, :own] = page_table[:own].to(torch.int32)
    return pt


def _free(*ts) -> None:
    for t in ts:
        if t is None:
            continue
        if isinstance(t, SamplerOutput):  # not a tensor: is_allocated() would raise and the tensors would leak
            _free(*t.tensors())
            continue
        try:
            if t.is_allocated():
                ttnn.deallocate(t)
        except Exception:
            pass


def _exact_router_refusal(wide_m: int) -> Optional[str]:
    """Why ``spec_verify="auto"`` cannot run with ``router_logits="exact_fp32"`` at the T64 step's ``wide_m`` gathered
    decode rows (review edit R-E7), or None when it can. Both lists must name ``wide_m``:
    ``moe.EXACT_ROUTER_DECODE_ROWS`` (where ``tt/moe.py`` runs the exact kernel; at any other row count the T64 rows
    would take the composite router and differ from the T32 rows, so ``auto`` would not be lossless) and
    ``model_config.ROUTER_EXACT_FP32_DECODE_ROWS`` (the config's validated list, which ``MotifTTConfig.validate``
    checks). Read at call time (tests patch them)."""
    moe_rows, cfg_rows = tuple(EXACT_ROUTER_DECODE_ROWS), tuple(ROUTER_EXACT_FP32_DECODE_ROWS)
    if wide_m not in moe_rows:
        return (
            f"tt/moe.py runs the exact-fp32 router only at {moe_rows} gathered decode rows "
            f"(moe.EXACT_ROUTER_DECODE_ROWS), so the {wide_m}-row T64 step would take the composite router and its "
            f"rows would differ from the T32 rows: 'auto' would not be lossless (P5_T64_DESIGN.md R-E7)"
        )
    if wide_m not in cfg_rows:
        return (
            f"tt/moe.py runs the exact-fp32 router at {moe_rows} gathered decode rows, but the config's validated list "
            f"model_config.ROUTER_EXACT_FP32_DECODE_ROWS = {cfg_rows} lacks the {wide_m}-row T64 step (the two lists "
            f"must agree; MotifTTConfig.validate refuses the launch until the config lists it, P5_T64_DESIGN.md R-E7)"
        )
    return None


@dataclass(frozen=True)
class SampledSpecDecodeResult(api.SpecDecodeResult):
    """``decode_forward_spec(..., sampling=...)``: the ordinary spec step's ids (``argmax`` / ``mtp_argmax``, for the
    bridge's speculation bookkeeping) plus the device-sampled tokens (``sample``, lane order, flagged active lanes
    already resolved on the host)."""

    sample: Optional[SampleResult] = None


LaneSampling = Tuple[Sequence[float], Sequence[float], Sequence[int], Sequence[Optional[int]]]


# ======================================================================================================================
# decode paths and the speculative step plan (features design §3.8, D10; host only)
# ======================================================================================================================
PLAIN, SPEC, WIDE = "plain", "spec", VP.WIDE_STEP  # "wide": the T64 step (verify_plan's kinds: "spec" | "wide")
DECODE_KINDS = (PLAIN, SPEC, WIDE)
SPEC_KINDS = (SPEC, WIDE)  # the decode paths of a speculating launch (decode_forward_spec)
DecodeKey = Tuple[str, str]  # (kind "plain" | "spec" | "wide", KV-write mode)
# decode_forward_spec's host profile (reset_spec_profile): "steps" calls, "passes" device runs, "wide" T64 steps
class ReplayWaiter:
    """``host_wait="spin"`` (B6a, ``MOTIF3_HOST_WAIT``): keeps the calling thread busy while a replayed decode trace
    runs, so the host code after the step's blocking read runs on a hot core.

    With the release's blocking read the thread sleeps for the whole replay (~85 ms on the 53-layer trace); under the
    host's ``schedutil`` governor the code that follows then runs ~2.5-3x slower than on a busy core
    (``logs/opt/phaseB/B6a``: the M1 step components match a cold-core microbenchmark, not a warm one). :meth:`spin`
    polls ``time.sleep(0)`` (a syscall that releases the GIL, so the process's other threads keep running) until
    ``residual_ms`` before the predicted end of the replay; the caller then makes its usual blocking read, which waits
    out the rest (a sleep of a few ms does not cool the core). The prediction, per key (path, pass), is the shortest of
    the last ``window`` replays that ended inside the blocking read (time from the enqueue to the read's return); a
    replay that had ended before the spin did (the read waited less than ``overshoot_ms``) shortens the prediction by
    ``backoff_ms`` instead, so a faster device is followed within a few steps and an overshoot costs at most one
    step's ``residual_ms``. Nothing is spun before the first replay of a key has been timed. The device sees exactly
    the same commands either way."""

    def __init__(self, residual_ms: float = 3.0, window: int = 8, overshoot_ms: float = 0.25, backoff_ms: float = 2.0,
                 clock: Callable[[], float] = time.perf_counter, idle: Callable[[], None] = lambda: time.sleep(0)):  # fmt: skip
        self.residual_ms = float(residual_ms)
        self.window = int(window)
        self.overshoot_ms = float(overshoot_ms)
        self.backoff_ms = float(backoff_ms)
        self.clock, self.idle = clock, idle
        self._hist: Dict[Any, List[float]] = {}
        self._cut: Dict[Any, float] = {}  # backoff applied to the prediction since the last real sample
        self.stats: Dict[str, float] = {"spins": 0, "spin_ms": 0.0, "blocked_ms": 0.0, "overshoots": 0}

    def predicted_ms(self, key) -> Optional[float]:
        h = self._hist.get(key)
        if not h:
            return None
        return max(0.0, min(h) - self._cut.get(key, 0.0))

    def spin(self, key, t_enqueue: float) -> float:
        """Poll until ``residual_ms`` before the predicted end of the replay enqueued at ``t_enqueue`` (clock
        seconds); returns the clock when the caller starts its blocking read."""
        pred = self.predicted_ms(key)
        now = self.clock()
        if pred is None:
            return now
        until = t_enqueue + (pred - self.residual_ms) / 1e3
        if now < until:
            self.stats["spins"] += 1
            t0 = now
            while now < until:
                self.idle()
                now = self.clock()
            self.stats["spin_ms"] += (now - t0) * 1e3
        return now

    def done(self, key, t_enqueue: float, t_read: float, t_done: float) -> None:
        """Record one replay: enqueued at ``t_enqueue``, blocking read started at ``t_read``, returned at
        ``t_done``."""
        blocked = (t_done - t_read) * 1e3
        self.stats["blocked_ms"] += blocked
        if blocked < self.overshoot_ms and self.predicted_ms(key) is not None:
            self.stats["overshoots"] += 1
            self._cut[key] = self._cut.get(key, 0.0) + self.backoff_ms
            return
        h = self._hist.setdefault(key, [])
        h.append((t_done - t_enqueue) * 1e3)
        del h[: -self.window]
        self._cut.pop(key, None)


SPEC_PROFILE_KEYS = ("steps", "passes", "wide", "plan", "write", "enqueue", "wait", "read", "result", "total")


@dataclass(frozen=True)
class SpecPass:
    """One run (trace replay or eager step) of the spec step: the physical-lane input tokens and the KV write.

    ``tokens [32]`` int32: per PHYSICAL lane, the anchor (owner lanes), the packed draft (partner lanes) or the
    overflow draft (pass 2), 0 on idle lanes. ``step``: the ``KVWriteStep`` (positions = KV write slots = FlashMLA's
    ``cur_pos`` and the RoPE positions; page tables; call-B partners)."""

    tokens: torch.Tensor
    step: KVWriteStep


@dataclass(frozen=True)
class SpecStepPlan:
    """The host plan of one ``decode_forward_spec`` step (:func:`plan_spec_step`): the packing of the drafts into idle
    lanes and the one or two passes that evaluate everything.

    ``partner_of``: drafted owner lane -> its idle partner lane in pass 1 (the draft runs there at ``n + 1`` with the
    owner's page-table row, written by call B). ``overflow``: drafted owners without a partner; pass 2 runs their
    drafts on their own lanes at ``n + 1`` (pass 1 wrote the anchors at ``n``), every other lane inactive."""

    mode: str
    batch: api.SpecDecodeBatch
    partner_of: Dict[int, int]
    overflow: Tuple[int, ...]
    passes: Tuple[SpecPass, ...]

    @property
    def is_verify(self) -> bool:
        return self.batch.is_verify

    @property
    def num_drafts(self) -> int:
        return len(self.partner_of) + len(self.overflow)

    @property
    def cross_row_partners(self) -> int:
        """Partners on another DP row than their owner (only with KV-R, ``all_split``)."""
        G = api.LANES_PER_GROUP
        return sum(int(o) // G != int(d) // G for o, d in self.partner_of.items())

    def result(
        self, outs: Sequence[Tuple[torch.Tensor, torch.Tensor]], logits: Optional[torch.Tensor] = None
    ) -> api.SpecDecodeResult:
        """``SpecDecodeResult`` (owner-lane order) from each pass's physical-lane ``(argmax [32], mtp_argmax [32])``:
        column 0 from pass 1's owner lanes, column 1 from the partner lane (pass 1) or the owner's own lane (pass 2).
        Entries the contract leaves unspecified (inactive lanes, column 1 of undrafted lanes) are -1."""
        if len(outs) != len(self.passes):
            raise ValueError(f"{len(outs)} pass outputs for a plan of {len(self.passes)} passes")
        n = api.NUM_LANES
        a1, m1 = (torch.as_tensor(t).reshape(-1).to(torch.int64).tolist() for t in outs[0])
        a2 = m2 = None
        if len(outs) > 1:
            a2, m2 = (torch.as_tensor(t).reshape(-1).to(torch.int64).tolist() for t in outs[1])
        am = [[-1, -1] for _ in range(n)]
        mm = [[-1, -1] for _ in range(n)]
        for lane, on in enumerate(self.batch.active.tolist()):
            if on:
                am[lane][0], mm[lane][0] = a1[lane], m1[lane]
        for o, d in self.partner_of.items():
            am[o][1], mm[o][1] = a1[d], m1[d]
        for o in self.overflow:
            am[o][1], mm[o][1] = a2[o], m2[o]
        return api.SpecDecodeResult(
            logits=logits,
            argmax=torch.tensor(am, dtype=torch.int32),
            mtp_argmax=torch.tensor(mm, dtype=torch.int32),
        )


def _ids_from_staging(reader, n: int = api.NUM_LANES) -> torch.Tensor:
    """``int64 [n]`` ids from a ``HostShardReader`` of a ``[1, 1, 1, n]`` uint32 output ("mesh" vocab split: identical
    on every chip, chip 0's copy): the T32 step's 32 lane-ordered ids, or the T64 step's 64 split-order ids; a fresh
    tensor (the staging is overwritten by the next read)."""
    return reader.views[0].reshape(-1)[: int(n)].to(torch.int64).clone()


def check_decode_page_tables(
    positions: torch.Tensor, page_table: torch.Tensor, *, block_size: int, num_blocks: Optional[int] = None
) -> None:
    """Every page-table entry an active lane uses (``0 .. positions // block_size``: FlashMLA reads them, the KV write
    lands in the last one) must be a real block id ``>= 1`` (0 is vLLM's null block) and ``< num_blocks`` (when
    given). ``positions [B]``: each lane's highest position this step (-1 = inactive). Raises ``ValueError``. Only the
    used columns are scanned (a few us per step at short contexts)."""
    pos = positions.to(torch.int64)
    W = int(page_table.shape[1])
    need = torch.where(pos >= 0, pos // int(block_size) + 1, torch.zeros_like(pos))
    top = int(need.max())
    if top > W:
        lane = int(torch.argmax(need))
        raise ValueError(
            f"lane {lane}: position {int(pos[lane])} needs page-table entry {int(need[lane]) - 1}, width is {W}"
        )
    if top == 0:
        return
    pt = page_table[:, :top]
    used = torch.arange(top)[None, :] < need[:, None]
    bad = used & (pt < 1)
    if num_blocks is not None:
        bad = bad | (used & (pt >= int(num_blocks)))
    if bool(bad.any()):
        lane, entry = (int(x) for x in torch.nonzero(bad)[0])
        raise ValueError(
            f"lane {lane}: page-table entry {entry} = {int(page_table[lane, entry])} for position {int(pos[lane])} is "
            f"not a block id in [1, {num_blocks if num_blocks is not None else 'num_blocks'}) (0 is vLLM's null block)"
        )


def check_prefill_page_table(
    page_table: torch.Tensor, end: int, *, block_size: int, num_blocks: Optional[int] = None, row: int = 0
) -> None:
    """The page-table entries one prefill row uses, ``0 .. cdiv(end, block_size) - 1`` (the fill writes the row's own
    blocks through them, an sp1 chunk reads its prefix and its SWA tail through them), must exist and be distinct real
    block ids in ``[1, num_blocks)`` (when given): 0 is vLLM's null block, and the fill kernel writes
    ``physical_block * block_stride`` with no bounds check, so an id past the pool would write outside the cache
    buffer (``PrefillRequest`` contract: the same id never twice in one row). The decode-side twin is
    :func:`check_decode_page_tables`. Raises ``ValueError``."""
    bs = int(block_size)
    need = api.cdiv(int(end), bs)
    n = int(page_table.shape[0])
    if n < need:
        raise ValueError(f"row {row}: page_table has {n} entries, positions [0, {end}) need {need}")
    ids = page_table[:need].to(torch.int64)
    bad = ids < 1
    if num_blocks is not None:
        bad = bad | (ids >= int(num_blocks))
    if bool(bad.any()):
        j = int(torch.nonzero(bad)[0])
        raise ValueError(
            f"row {row}: page-table entry {j} = {int(ids[j])} (positions [{j * bs}, {min((j + 1) * bs, int(end))})) "
            f"is not a block id in [1, {num_blocks if num_blocks is not None else 'num_blocks'}) (0 is vLLM's null "
            "block)"
        )
    if int(torch.unique(ids).numel()) != need:
        uniq, counts = torch.unique(ids, return_counts=True)
        dup = int(uniq[counts > 1][0])
        raise ValueError(
            f"row {row}: block id {dup} appears at page-table entries "
            f"{torch.nonzero(ids == dup).flatten().tolist()}: one physical block cannot hold two logical blocks' KV"
        )


def plan_spec_step(
    batch: api.SpecDecodeBatch,
    *,
    mode: str,
    cfg: MotifTTConfig,
    num_blocks: Optional[int] = None,
    width: Optional[int] = None,
    lanes_per_call: Optional[int] = None,
) -> SpecStepPlan:
    """Every host check and the packing of one speculative decode step (features design §3.8.2), before any device
    op. Raises ``ValueError`` on a bad batch.

    * Checks: page-table width (``width``: the trace's), positions (``< max_model_len``; ``n + 1`` too on drafted
      lanes), anchor / draft token ids (``< vocab``), every used page-table entry (:func:`check_decode_page_tables`,
      up to ``n + 1`` on drafted lanes), and each pass's ``KVWriteStep`` against ``mode``
      (``kv_write.check_kv_write_step``: partner rules, one draft per owner, no block written twice in one update
      call).
    * Packing (``kv_write.assign_partner_lanes``): in a split mode each drafted owner borrows the lowest idle lane of
      its own DP row, then (``all_split``: KV-R, every chip holds every lane's KV) of any row; the drafts left over run
      in pass 2 (overflow). Without a split mode (``row`` / ``all``) no draft is packed: every draft goes to pass 2
      (a correct but slower layout).

    The per-lane bookkeeping runs on Python lists (it runs every decode step; a torch op on a 32-lane vector costs a
    few us each and a torch-heavy plan measured 0.6-1.2 ms in the decode loop)."""
    if mode not in api.KV_WRITE_MODES:
        raise ValueError(f"kv-write mode must be one of {api.KV_WRITE_MODES}, got {mode!r}")
    if not isinstance(batch, api.SpecDecodeBatch):
        raise TypeError(f"decode_forward_spec needs a SpecDecodeBatch, got {type(batch).__name__}")
    W, bs = batch.page_table_width, int(cfg.kv_block_size)
    if width is not None and W != int(width):
        raise ValueError(f"page-table width {W} != the decode trace's width {width}")
    n = api.NUM_LANES
    pos_l, dr_l, tok_l = batch.positions.tolist(), batch.draft_tokens.tolist(), batch.tokens.tolist()
    L, V = int(cfg.max_model_len), int(cfg.vocab_size)
    top_l = [p + 1 if (d >= 0 and p >= 0) else p for p, d in zip(pos_l, dr_l)]
    for lane, t in enumerate(top_l):
        if t >= L:
            raise ValueError(f"lane {lane}: decode position {t} >= max_model_len {L}")
    tok1 = [t if p >= 0 else 0 for t, p in zip(tok_l, pos_l)]
    if any(not 0 <= t < V for t in tok1) or any(d >= V for d in dr_l):
        raise ValueError(f"decode token ids must be in [0, {V}): anchors {tok1}, drafts {dr_l}")
    check_decode_page_tables(torch.tensor(top_l), batch.page_table, block_size=bs, num_blocks=num_blocks)
    drafted = [lane for lane in range(n) if dr_l[lane] >= 0]
    if drafted and is_split(mode):
        partner_of, overflow = assign_partner_lanes(
            batch.positions, batch.has_draft, cross_row=allows_cross_row_partners(mode),
            lanes_per_row=int(cfg.lanes_per_row),
        )  # fmt: skip
    else:
        partner_of, overflow = {}, tuple(drafted)
    for o, d in partner_of.items():
        tok1[d] = dr_l[o]
    if partner_of:
        step1 = KVWriteStep.packed_verify(batch.positions, batch.page_table, partner_of)
    else:
        step1 = KVWriteStep.ordinary(batch.positions, batch.page_table)
    passes = [SpecPass(torch.tensor(tok1, dtype=torch.int32), step1)]
    if overflow:
        tok2 = [0] * n
        for o in overflow:
            tok2[o] = dr_l[o]
        step2 = KVWriteStep.overflow_pass(batch.positions, batch.page_table, overflow)
        passes.append(SpecPass(torch.tensor(tok2, dtype=torch.int32), step2))
    kw = dict(block_size=bs, max_seq_len=L, lanes_per_row=int(cfg.lanes_per_row))
    if is_replicated(mode):
        kw["lanes_per_call"] = (
            lanes_per_call
            if lanes_per_call is not None
            else lanes_per_call_for(
                mode, cfg.dtypes.kv_cache_name, lanes=int(cfg.max_batch), lanes_per_row=int(cfg.lanes_per_row)
            )
        )
    for ps in passes:
        check_kv_write_step(ps.step, mode, **kw)
    return SpecStepPlan(
        mode=mode, batch=batch, partner_of=dict(partner_of), overflow=tuple(overflow), passes=tuple(passes)
    )


@dataclass
class DecodePath:
    """One decode step kind with its persistent device inputs and, once captured, its trace.

    ``kind`` ``"plain"`` (embedding -> layers -> LM head logits; :meth:`MotifGenerator.decode_forward`), ``"spec"``
    (the T32-spec step ``MotifModel.decode_spec``: + main argmax + the MTP layer + MTP argmax;
    :meth:`MotifGenerator.decode_forward_spec`) or ``"wide"`` (the T64 step ``MotifModel.decode_wide``: 16 rows per
    DP row, anchors + drafts; :meth:`MotifGenerator.decode_forward_spec` with ``spec_verify`` "wide" / "auto").
    ``mode``: the KV-write mode (``row`` on the plain path = the draft-1 ops, no ``DecodeKVWrite``). ``inputs``:
    ``tokens`` / ``rot`` (``[4, 16]`` / 16 used rot rows per DP row on the wide path; and ``cur`` / ``pt`` for the
    draft-1 ``row`` path); ``kv_write``: the ``DecodeKVWrite`` (FlashMLA's ``cur_pos`` / ``page_table``, the A''
    groups and the update-call inputs; ``rows=64`` on the wide path) of every other path. ``out``: the captured
    outputs (plain: the ROW_MAJOR logits; spec / wide: ``(rm, a, m)``, ``rm`` None on an argmax-only wide path).
    ``so``: the captured device sampler's outputs (:class:`~models.demos.motif3.tt.sampling.SamplerOutput`) when the
    trace holds the sampler, else None."""

    kind: str
    mode: str
    width: int
    inputs: Dict[str, Any] = field(default_factory=dict)
    kv_write: Optional[DecodeKVWrite] = None
    trace_id: Any = None
    out: Any = None
    pool: Any = None
    warmed: bool = False
    so: Optional[SamplerOutput] = None
    # B6a ``host_staging="fast"``: the host values last copied into each of ``inputs`` (an input whose new values are
    # equal is not copied again; DecodeKVWrite.write_step does the same for its inputs)
    host_last: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def key(self) -> DecodeKey:
        return (self.kind, self.mode)

    @property
    def traced(self) -> bool:
        return self.trace_id is not None


@dataclass
class PrefillRowJob:
    """One row of a ``prefill_forward_batch`` call after the host checks: its plan and every chunk's host tables."""

    index: int  # input position of the row
    request: api.PrefillRequest
    plan: PP.RowPlan
    tables: List[ChunkHostTables]


@dataclass
class PrefillBatchPlan:
    """The host plan of one ``prefill_forward_batch`` call (:meth:`MotifGenerator.plan_prefill_batch`).

    ``jobs``: one per row, in input order (the row plan and every chunk's ``ChunkHostTables``). ``order``: the
    writer-first row order (``prefill_plan.order_prefill_requests``; the execution order with packing off).
    ``passes``: the execution order, one ``prefill_plan.PrefillPass`` per device run: ``solo`` (one chunk:
    ``tables[i]`` is its ``ChunkHostTables``) or packed ``pk0`` / ``pk1`` (``tables[i]`` its ``PackedHostTables``).
    With packing off, one solo pass per chunk in ``order``. ``shapes``: the program-set key of every pass (solo
    ``(path, bucket)``, packed ``PrefillPass.shape``). ``packed``: planned with packed prefill on. ``plan_error``: with
    packing on, the packed planner's (or a packed table check's) failure, ``"<exception type>: <message>"`` (a planner
    bug); the passes are then the packing-off ones (:meth:`MotifGenerator.plan_prefill_batch`). None otherwise."""

    jobs: List[PrefillRowJob]
    order: List[int]  # writer-first row order (indices into ``jobs`` = input order)
    shapes: Set[PrefillShape] = field(default_factory=set)
    passes: List[PP.PrefillPass] = field(default_factory=list)
    tables: List[Any] = field(default_factory=list)  # per pass: ChunkHostTables (solo) | PackedHostTables (packed)
    packed: bool = False
    plan_error: Optional[str] = None

    @property
    def chunks(self) -> int:
        return sum(len(j.plan.chunks) for j in self.jobs)

    @property
    def requests(self) -> List[api.PrefillRequest]:
        return [j.request for j in self.jobs]

    @property
    def packed_passes(self) -> List[PP.PrefillPass]:
        return [p for p in self.passes if p.is_packed]

    @property
    def fallbacks(self) -> List[PP.PrefillPass]:
        """Solo passes the post-capture shape filter made from a packed pass of an unwarmed shape."""
        return [p for p in self.passes if p.fallback is not None]

    def describe(self) -> str:
        """One line per call for logs: the passes in order."""
        return "; ".join(p.describe() for p in self.passes)


class MotifGenerator(api.MotifGenerator):
    """The Motif-3 runtime (see the module docstring). Build with :meth:`create` (vLLM) or directly from an existing
    :class:`MotifModel` (tests / demo): ``MotifGenerator(mesh_device, cfg, model)``.

    Test hooks: ``chunk_observer(job, chunk, streams)`` (default None), called after every solo prefill chunk's layers
    with the chunk's residual streams ``[1, 4, C, 4096]`` (eager; it must not free or keep them). Teacher-forced checks
    read the LM head on every tile of a chunk through it (``head.forward_prefill(streams, row)``: no new program).
    ``pass_observer(batch, pass_, streams)`` (default None): the same after every packed pass, with the pass's streams
    ``[1, 4, T, 4096]`` (segment ``k``'s rows start at ``pass_.offset(k)``; its row is ``batch.jobs[seg.row]``).
    ``spec_observer(plan, pass_index, out)`` (default None), called after every pass of a speculative step with the
    host ``out = {"a": [32], "m": [32], "logits": [32, V] or None}`` in PHYSICAL-lane order (``observe_logits=True``
    reads the logits of every pass, ~2 ms each); a T64 step calls it once with ``plan`` = its
    ``verify_plan.WideStepPlan``, ``pass_index`` 0 and ``out = {"a": [64], "m": [64], "logits": [32, V] or None}`` in
    split order. ``extra_decode_paths``: ``(kind, mode)`` decode paths that :meth:`warmup_decode` prepares and captures
    next to the serving paths (set before the warmup; e.g. the plain ``("plain", "all")`` trace next to the spec trace,
    for comparisons in one session). Tests only: serving captures :attr:`serving_paths` (one trace, two in
    ``spec_verify="auto"``). Several captured traces are safe under F3N rules R1-R5 (``docs/p5_t64/f3.md`` §6-§7: the
    garbage prefill tiles once blamed on a second trace, ``tests/test_resumed_prefill.py`` ``MULTI_TRACE_NOTE``, were
    the TP-ring all-gather race that ``ring_gather="safe"`` closes); :meth:`warmup_decode` logs the traces it holds and
    warns when ``ring_gather`` is not ``"safe"``. A launch whose ring gather is not ``"safe"`` is refused with a T64
    path and warned about otherwise, once, at construction (:meth:`_ring_gather_warning`, design X3).

    Speculative verify mode (module docstring "T64 verify"): :attr:`spec_verify` = ``cfg.spec_verify`` (``packed`` /
    ``wide`` / ``auto``; the modules' 64-row constants follow the config, so ``settings.spec_verify`` must agree).
    :meth:`verify_kind` names the trace a step runs on; :attr:`last_verify_kind` the one the last step ran on;
    :attr:`last_spec` is that step's plan (``SpecStepPlan`` on T32, ``verify_plan.WideStepPlan`` on T64).

    Args beyond the model: ``decode_kv_mode`` / ``spec_kv_mode`` override the KV-write mode of the plain / spec path
    (default ``generator_api.kv_write_mode(cfg.kv_replicated_decode, False / True)``: ``row`` / ``row_split``, or
    ``all`` / ``all_split`` with KV-R).

    Packed prefill (module docstring): :attr:`packed_prefill` (default ``settings.packed_prefill``, else off) switches
    it per call; the segment sizes and pass cap are the config's (``cfg.pack_seg_buckets``,
    ``cfg.pack_sp1_seg_buckets``, ``cfg.pack_tokens_cap``). :meth:`warmup_prefill` warms the packed shapes only while
    the switch is on; turned on after a capture without them, every packed pass runs as solo chunks.
    ``packed_warmup`` (default ``settings.packed_warmup``): ``"attention"`` (per shape the attention of one global and
    one SWA layer on zeros, ``model.warm_attention``) or ``"full"`` (one full warm-up pass per shape).

    Device sampling (module docstring): :meth:`enable_device_sampling` builds ``self.sampler`` before the decode
    warmup; ``sampling_observer(kind, positions, res, logits)`` (tests; default None) is called after every
    device-sampled step with the lane-order :class:`SampleResult` (flagged lanes resolved) and a callable returning
    the step's host logits (one 14 MB read, on demand)."""

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        model: MotifModel,
        *,
        settings: Optional[api.GeneratorSettings] = None,
        log: Optional[Callable[[str], None]] = _log_default,
        decode_kv_mode: Optional[str] = None,
        spec_kv_mode: Optional[str] = None,
    ):
        if tuple(model.layer_ids) != tuple(range(len(model.layer_ids))):
            raise ValueError(f"the generator runs a prefix model (layers 0..N-1), got layers {model.layer_ids}")
        if cfg.max_batch != api.NUM_LANES:
            raise ValueError(f"cfg.max_batch must be {api.NUM_LANES} (the decode trace runs all lanes)")
        # the verify mode is the config's: the modules' 64-row constants (MoE top-k pads, LM-head argmax) follow
        # cfg.wide_rows_per_dp, so settings that disagree would stage a T64 path the model cannot run
        self.spec_verify = str(getattr(cfg, "spec_verify", "packed") or "packed")
        if self.spec_verify not in api.SPEC_VERIFY_MODES:
            raise ValueError(f"cfg.spec_verify must be one of {api.SPEC_VERIFY_MODES}, got {self.spec_verify!r}")
        if settings is not None and settings.spec_verify != self.spec_verify:
            raise ValueError(
                f"settings.spec_verify={settings.spec_verify!r} but cfg.spec_verify={self.spec_verify!r}: build the "
                f"config from the settings (MotifTTConfig.from_settings), its 64-row module constants follow it"
            )
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.model = model
        self.settings = settings
        self.log = log or (lambda m: None)
        self._pool: Optional[MotifKVPool] = None
        self._paths: Dict[DecodeKey, DecodePath] = {}  # staged decode paths: persistent inputs (+ trace once captured)
        # prefill shapes compiled by warmup_prefill: solo (path, bucket) and packed ("pk0", T, S) / ("pk1", T, S, tails)
        self._warmed: Set[PrefillShape] = set()
        # packed prefill (P5): the per-call switch and the warm-up mode (module docstring)
        self.packed_prefill = bool(getattr(settings, "packed_prefill", False))
        warm = getattr(settings, "packed_warmup", None) or api.DEFAULT_PACKED_WARMUP
        if warm not in api.PACKED_WARMUP_MODES:
            raise ValueError(f"packed_warmup must be one of {api.PACKED_WARMUP_MODES}, got {warm!r}")
        self.packed_warmup = warm
        self._packed_mtp_warmed: Set[int] = set()  # pass rows T whose packed MTP fill (gathered RoPE at T) compiled
        self._fallback_logged: Set[PrefillShape] = set()  # unwarmed packed shapes already logged (once per shape)
        self._pack_cost = PP.prefill_cost_model(cfg.prefill_cost_table, sp1_s_per_row_key=cfg.prefill_sp1_s_per_row_key)
        kvr = bool(cfg.kv_replicated_decode)
        # plain decode steps: "row" (draft 1) or "all" (KV-R); speculative steps: "row_split" or "all_split" (KV-R)
        self.decode_kv_mode = _check_mode(decode_kv_mode or api.kv_write_mode(kvr, False))
        self.spec_kv_mode = _check_mode(spec_kv_mode or api.kv_write_mode(kvr, True))
        if kvr and not (is_replicated(self.decode_kv_mode) and is_replicated(self.spec_kv_mode)):
            self.log(
                f"warning: KV-R is on (prefix caching) but a decode path writes without it (plain "
                f"{self.decode_kv_mode!r}, spec {self.spec_kv_mode!r}): only for measurements, never for serving"
            )
        # A speculating launch (spec_tokens = 1 and the MTP layer) runs EVERY decode step through decode_forward_spec:
        # the T32-spec trace ("packed", "auto") or the T64 trace ("wide"); "auto" also captures the T64 trace for the
        # verify steps whose drafts do not fit idle lanes (serving_paths).
        spec = model.mtp is not None and int(cfg.spec_tokens) > 0
        if not spec:
            self.serving_path: DecodeKey = (PLAIN, self.decode_kv_mode)
        elif self.spec_verify == "wide":
            self.serving_path = (WIDE, self.spec_kv_mode)
        else:
            self.serving_path = (SPEC, self.spec_kv_mode)
        # the T64 path of this launch ("wide" / "auto" with speculation), else None
        self.wide_path: Optional[DecodeKey] = (
            (WIDE, self.spec_kv_mode) if spec and self.spec_verify in api.WIDE_SPEC_VERIFY_MODES else None
        )
        if self.wide_path is not None:
            self._check_wide_launch()  # refuses ring_gather "native" / "lean" (R1, R-E5)
        else:
            warn = self._ring_gather_warning()  # design X3: the other launches only warn
            if warn is not None:
                self.log(warn)
        self.extra_decode_paths: List[DecodeKey] = []
        self.chunk_observer: Optional[Callable[[PrefillRowJob, PP.ChunkPlan, Any], None]] = None
        self.pass_observer: Optional[Callable[[PrefillBatchPlan, PP.PrefillPass, Any], None]] = None
        self.spec_observer: Optional[Callable[[SpecStepPlan, int, Dict[str, Any]], None]] = None
        self.sampling_observer: Optional[Callable[..., None]] = None
        self.observe_logits = False
        self.last_prefill: Optional[PrefillBatchPlan] = None
        self.last_spec: Optional[Any] = None  # the last decode_forward_spec step's plan: SpecStepPlan | WideStepPlan
        self.last_verify_kind: Optional[str] = None  # the trace kind it ran on: "spec" (T32) | "wide" (T64)
        # F3N rule R4: the decode path whose replayed outputs are not read yet (cleared by the step's blocking read)
        self._unread: Optional[DecodeKey] = None
        # device sampling (enable_device_sampling): the sampler and the outputs of the step that ran last (eager runs
        # and captures hand them over through _take_so)
        self.sampler: Optional[MotifDeviceSampler] = None
        self._pending_so: Optional[SamplerOutput] = None
        self.stats: Dict[str, int] = {
            "prefill_calls": 0,
            "prefill_rows": 0,
            "prefill_chunks": 0,
            "sp1_chunks": 0,
            "recomputed_rows": 0,
            "mtp_fills": 0,
            # packed prefill (P5): device runs of a prefill call are solo chunks or packed passes
            "solo_passes": 0,  # solo chunks (packing off: every chunk)
            "packed_calls": 0,  # calls with at least one packed pass
            "packed_passes": 0,  # pk0 + pk1 passes
            "packed_pk1_passes": 0,  # ... of which pk1 (resumed segments at one start)
            "packed_segments": 0,  # real segments (chunks) run in packed passes
            "packed_dummy_segments": 0,  # dummy segments that filled B
            "packed_padding_rows": 0,  # rows of packed passes without a real token (segment padding + dummies)
            "packed_solo_fallbacks": 0,  # chunks run solo because their packed shape was not warmed (after capture)
            "packed_plan_errors": 0,  # packing-on calls whose packed plan failed (a planner bug): run per row instead
            "decode_steps": 0,  # device runs of a decode step (trace replays + eager steps), every path and pass
            "spec_steps": 0,  # decode_forward_spec calls
            "verify_steps": 0,  # ... with at least one draft
            "drafts": 0,
            "packed_drafts": 0,  # drafts evaluated on an idle partner lane (pass 1, call B)
            "cross_row_partners": 0,  # ... on another DP row than their owner (KV-R)
            "overflow_drafts": 0,  # drafts evaluated in pass 2 on their own lane
            "overflow_passes": 0,
            # T64 (spec_verify "wide" / "auto"): drafts = packed_drafts + overflow_drafts + wide_drafts
            "wide_steps": 0,  # decode_forward_spec steps run on the T64 trace (one device run each)
            "wide_verify_steps": 0,  # ... with at least one draft
            "wide_drafts": 0,  # drafts evaluated on their owners' draft rows (T64)
            "auto_t32_verifies": 0,  # verify steps of an "auto" launch that ran on T32 (every draft fit an idle lane)
            "sampled_steps": 0,  # device-sampled decode steps (plain or spec)
            "sampled_lanes": 0,  # ... active lanes they sampled
            "gumbel_lanes": 0,  # ... of which full-vocab (top_p = 1) Gumbel lanes
            "fallback_steps": 0,  # device-sampled steps that read the logits for the host fallback
            "fallback_lanes": 0,  # ... active lanes re-sampled on the host
        }
        self.timings: Dict[str, float] = {}
        self.spec_profile: Dict[str, float] = {k: 0.0 for k in SPEC_PROFILE_KEYS}
        # B6a (MOTIF3_HOST_STAGING): "fast" builds the same per-step device inputs with fewer host ops
        self.host_staging = api.check_host_staging(getattr(cfg, "host_staging", None), name="cfg.host_staging")
        self._dp_mapper = None  # dp_row_mapper, built on first use ("fast")
        self.stats["input_copies"] = 0  # host -> device copies of the paths' own inputs (both modes)
        self.stats["input_copies_skipped"] = 0  # ... skipped because the values did not change ("fast")
        self.host_wait = api.check_host_wait(getattr(cfg, "host_wait", None), name="cfg.host_wait")
        self.waiter: Optional[ReplayWaiter] = ReplayWaiter() if self.host_wait == "spin" else None
        self._readers: Dict[Tuple, HostShardReader] = {}  # host staging of the spec outputs' id reads (by role)

    # ==============================================================================================================
    # construction (GEN-1)
    # ==============================================================================================================
    @classmethod
    def create(
        cls, *, hf_config: Any, mesh_device: Any, settings: api.GeneratorSettings, **model_kwargs
    ) -> "MotifGenerator":
        """Build the runtime on the plugin's open mesh (``generator_api.MotifGenerator.create``). ``model_kwargs`` go
        to :class:`MotifModel` (``cache``, ``vocab_split``, ``layer_kwargs``, ``source``, ``mtp``). ``cache`` defaults
        to ``settings.tt_cache_policy`` (``MOTIF3_TT_CACHE_POLICY``: ``"auto"`` never writes the TT cache, ``"write"``
        converts and writes the parts that are not complete; ``tt/model.py``)."""
        log = model_kwargs.pop("log", _log_default)
        l1s = require_l1_small(mesh_device)
        if "cache" not in model_kwargs:  # an invalid policy is refused here, before anything loads
            model_kwargs["cache"] = api.check_tt_cache_policy(
                getattr(settings, "tt_cache_policy", api.DEFAULT_TT_CACHE_POLICY)
            )
        cache_policy = normalize_cache_policy(model_kwargs["cache"])
        # spec_verify "wide" / "auto": MotifTTConfig.validate refuses ring_gather != "safe" (F3N R1, R-E5) and "auto"
        # with the exact-fp32 router at an unsupported row count (R-E7) before any weight loads; the constructor
        # re-checks them against the built model (its MotifCCL, its MoE)
        cfg = MotifTTConfig.from_settings(settings, mesh_device=mesh_device, hf_config=hf_config)
        log(
            f"create: {cfg.describe()} (mesh L1_SMALL {l1s} B per core; weights {settings.weights_path} "
            f"[{settings.weights_source}]; TT cache policy {cache_policy!r})"
        )
        if settings.packed_prefill:
            log(
                f"create: packed prefill on (P5): pk0 S {'/'.join(map(str, cfg.pack_seg_buckets)) or 'off'}, pk1 S "
                f"{'/'.join(map(str, cfg.pack_sp1_seg_buckets)) or 'off'}, T <= {cfg.pack_tokens_cap}; "
                f"{len(cfg.packed_prefill_shapes())} packed shapes warmed before the decode capture "
                f"({settings.packed_warmup!r} warm-up); after it an unwarmed packed shape runs as solo chunks"
            )
        else:
            log("create: packed prefill off (MOTIF3_PACKED_PREFILL): the rows of a prefill call run one after another")
        if "source" not in model_kwargs:
            if settings.weights_are_local:
                model_kwargs["source"] = LazySource(settings.weights_path, log=log)
            elif settings.weights_path:  # an uncached repo id: only the shards of TT-cache misses, each disk-guarded
                model_kwargs["source"] = LazySource(
                    repo_id=settings.weights_path, revision=settings.weights_revision, log=log
                )
            else:
                model_kwargs["source"] = LazySource(cfg.weights_dir, log=log)
        t0 = time.time()
        model = MotifModel(mesh_device, cfg, layers=range(int(settings.num_layers)), log=log, **model_kwargs)
        log(
            f"create: {model.num_layers} layers{' + the MTP layer' if model.mtp is not None else ''} loaded in "
            f"{time.time() - t0:.1f} s"
        )
        gen = cls(mesh_device, cfg, model, settings=settings, log=log)
        if gen.spec_launch:
            log(f"create: {gen.describe_spec_verify()}")
        if gen.spec_launch and os.environ.get("OMP_WAIT_POLICY", "").strip().upper() != "PASSIVE":
            log(
                "warning: speculative decoding without OMP_WAIT_POLICY=PASSIVE (features design §1.1): torch's "
                "spinning OpenMP workers stall the input copies after the step's host planning (measured +4.7 ms per "
                "verify step)"
            )
        return gen

    # ==============================================================================================================
    # static facts
    # ==============================================================================================================
    @property
    def num_layers(self) -> int:
        return self.model.num_layers

    @property
    def vocab_size(self) -> int:
        return int(self.cfg.vocab_size)

    @property
    def max_prefill_len(self) -> int:
        """Longest row: ``max_model_len`` (rows above the span cap are split into chunks)."""
        return int(self.cfg.prefill_buckets[-1])

    @property
    def supports_resumed_prefill(self) -> bool:
        return True

    @property
    def prefill_alignment(self) -> int:
        """``A = cfg.prefill_resume_alignment`` = lcm(block, q_chunk, k_chunk) of the sp1 global op (every chunk start
        is a multiple of it)."""
        return int(self.cfg.prefill_resume_alignment)

    @property
    def max_prefill_span(self) -> int:
        """The span cap ``cfg.max_prefill_span`` (8192; ``MOTIF3_PREFILL_MAX_BUCKET``): the largest bucket compiled."""
        return int(self.cfg.max_prefill_span)

    @property
    def max_sp1_bucket(self) -> int:
        """The largest bucket an sp1 chunk uses (``attention.max_sp1_bucket``: the SWA square fits max_model_len)."""
        return int(max_sp1_bucket(self.cfg))

    @property
    def supports_spec_decode(self) -> bool:
        """``decode_forward_spec`` runs (the model has its MTP layer: the T32-spec step, packed verify)."""
        return self.model.mtp is not None

    @property
    def mtp_enabled(self) -> bool:
        """The model has its MTP layer: prefill also fills the MTP cache (KV-only, design D9)."""
        return self.model.mtp is not None

    @property
    def trace_captured(self) -> bool:
        """Some decode path holds a captured trace (prefill shapes and persistent inputs are then frozen)."""
        return any(p.traced for p in self._paths.values())

    @property
    def spec_launch(self) -> bool:
        """Every decode step runs through :meth:`decode_forward_spec` (``spec_tokens = 1`` with the MTP layer): the
        T32-spec trace, and / or the T64 trace (:attr:`spec_verify`)."""
        return self.serving_path[0] in SPEC_KINDS

    @property
    def serving_paths(self) -> List[DecodeKey]:
        """The decode paths a serving launch stages and captures, in capture order: :attr:`serving_path`, and with
        ``spec_verify="auto"`` the T64 path after it (``[("spec", m), ("wide", m)]``: the T32-spec trace is captured
        first); one path otherwise."""
        out = [self.serving_path]
        if self.wide_path is not None and self.wide_path not in out:
            out.append(self.wide_path)
        return out

    @property
    def wide_has_logits(self) -> bool:
        """The T64 path also untilizes the anchors' logits and, with device sampling, runs the sampler on them:
        ``spec_verify="wide"`` only (its ordinary steps run on T64). In ``auto`` the T64 trace is argmax-only (ordinary
        and sampled steps run on T32)."""
        return self.spec_verify == "wide"

    def _ring_gather(self) -> str:
        """The ring-gather mode the decode traces are built with: the model's ``MotifCCL`` (tests may switch it at run
        time), else the config's."""
        ccl = getattr(self.model, "ccl", None)
        return str(getattr(ccl, "ring_gather", None) or getattr(self.cfg, "ring_gather", "safe"))

    def _check_wide_launch(self, kv_mode: Optional[str] = None) -> None:
        """The preconditions of a launch that stages the T64 path (``spec_verify`` "wide" / "auto"; design X3, §2.3);
        raises ``ValueError``. Runs in the constructor (``create``), when a T64 path is staged (``kv_mode``: its
        KV-write mode, default :attr:`spec_kv_mode`) and before the decode capture.

        * F3N rule R1 / review edit R-E5: ``ring_gather="safe"`` on the config AND on the model's ``MotifCCL`` (every
          race-prone TP-ring all-gather rerouted). ``native`` races; ``lean`` keeps the decode-sized gathers native and
          needs its own G-X run first.
        * A T64 config and model: ``cfg.wide_rows_per_dp`` = 16 (the modules' 64-row constants exist), a split KV-write
          mode (call A anchors, call B drafts), the LM head's "mesh" vocab split (the split-order head).
        * Review edit R-E7: ``auto`` with ``router_logits="exact_fp32"`` only when the MoE runs its exact router at the
          T64 step's gathered row count (``moe.EXACT_ROUTER_DECODE_ROWS``) and the config lists it
          (``model_config.ROUTER_EXACT_FP32_DECODE_ROWS``); otherwise T64 rows would take the composite router, differ
          from the T32 rows, and ``auto`` would not be lossless. The message names the list that lacks the row count
          (:func:`_exact_router_refusal`).

        A launch without a T64 path is not refused for its ring gather; the constructor logs
        :meth:`_ring_gather_warning` instead (design X3)."""
        cfg, mode = self.cfg, self.spec_verify
        rows = 2 * int(cfg.lanes_per_row)
        if int(getattr(cfg, "wide_rows_per_dp", 0) or 0) != rows:
            raise ValueError(
                f"a T64 decode path needs a T64 config (cfg.wide_rows_per_dp = {rows}: spec_tokens > 0 and spec_verify "
                f"'wide' / 'auto', so the modules hold their 64-row constants); this one has spec_verify={mode!r}, "
                f"wide_rows_per_dp {getattr(cfg, 'wide_rows_per_dp', None)}"
            )
        ring = sorted({str(getattr(cfg, "ring_gather", "safe")), self._ring_gather()})
        if ring != ["safe"]:
            raise ValueError(
                f"spec_verify={mode!r} needs ring_gather='safe' on the config and on the model's MotifCCL, got {ring}: "
                f"the T64 trace and a second decode trace rely on every race-prone TP-ring all-gather being rerouted "
                f"(F3N rule R1, docs/p5_t64/f3.md §6; P5_T64_DESIGN.md X3, review edit R-E5: 'lean' keeps the decode "
                f"gathers native and needs its own G-X run first)"
            )
        kv = self.spec_kv_mode if kv_mode is None else _check_mode(kv_mode)
        if not is_split(kv):
            raise ValueError(
                f"spec_verify={mode!r}: the T64 step writes the drafts in call B, so it needs a split KV-write mode "
                f"(row_split / all_split), got {kv!r}"
            )
        vs = getattr(self.model.head, "vocab_split", "mesh")
        if vs != "mesh":
            raise ValueError(f"spec_verify={mode!r} needs the LM head's 'mesh' vocab split (the split-order head), got "
                             f"{vs!r}")  # fmt: skip
        if mode == "auto" and str(getattr(cfg, "router_logits", "composite")) == "exact_fp32":
            reason = _exact_router_refusal(int(cfg.dp) * rows)
            if reason is not None:
                raise ValueError(
                    f"spec_verify='auto' with router_logits='exact_fp32' is refused: {reason}. Use "
                    f"MOTIF3_ROUTER_LOGITS=composite, or MOTIF3_SPEC_VERIFY=packed / wide (one decode trace)"
                )

    def _ring_gather_warning(self) -> Optional[str]:
        """The launch warning (design X3) of a generator whose TP-ring all-gathers are not all rerouted: ``ring_gather``
        "native" or "lean" on the config or on the model's ``MotifCCL``; None under "safe". A launch with a T64 path
        refuses those modes instead (:meth:`_check_wide_launch`, F3N rule R1, review edit R-E5); every other launch
        (``packed``, no speculation) runs with them and the constructor logs this line once."""
        cfg_ring, ccl_ring = str(getattr(self.cfg, "ring_gather", "safe")), self._ring_gather()
        modes = sorted({cfg_ring, ccl_ring})
        if modes == ["safe"]:
            return None
        if "native" in modes:
            why = (
                "'native' leaves every race-prone TP-ring all-gather on ttnn's multicast factory, whose completion "
                "can precede its alternate-route pages: prefill is not run-to-run reproducible and a stale tile can "
                "reach the logits (the F3 garbage prefills, docs/p5_t64/f3.md §2; docs/determinism/INVESTIGATION.md)"
            )
        else:
            why = (
                "'lean' keeps the single-page decode gathers native, which race about once per 10^4 decode steps "
                "(silent stale tiles, docs/determinism/FIX.md)"
            )
        return (
            f"warning: ring_gather {cfg_ring!r} on the config, {ccl_ring!r} on the model's MotifCCL: {why}. Serve "
            f"with the default 'safe' (MOTIF3_RING_GATHER; P5_T64_DESIGN.md X3: spec_verify 'wide' / 'auto' refuse "
            f"anything else)"
        )

    def describe_spec_verify(self) -> str:
        """One line for the logs: the verify mode, its decode traces and routing, and when every live lane drafts."""
        if not self.spec_launch:
            return "no speculation: one plain decode trace"
        m, kv = self.spec_verify, self.spec_kv_mode
        if m == "packed":
            return (
                f"spec_verify='packed': one T32-spec decode trace (KV write {kv!r}); drafts run on idle lanes, drafts "
                f"without one in a second replay (overflow pass)"
            )
        s = self.settings
        if m == "wide":
            drafting = "every live lane may draft"
            traces = "the T64 trace alone serves every step (the anchors' logits and the device sampler on them)"
        else:
            prior = float(getattr(s, "spec_alpha_prior", api.DEFAULT_SPEC_ALPHA_PRIOR))
            fixed = getattr(s, "wide_min_lanes", None)
            if fixed is not None:
                drafting = f"every live lane drafts from {fixed} live lanes (MOTIF3_WIDE_MIN_LANES)"
            else:
                c = VP.crossover_lanes(prior, float(self.cfg.wide_step_ratio))
                drafting = (
                    f"every live lane drafts from c*={'never' if c >= api.WIDE_MIN_LANES_NEVER else c} live lanes at "
                    f"the acceptance prior {prior:g} (r {float(self.cfg.wide_step_ratio):g}; the bridge passes its "
                    f"running estimate)"
                )
            traces = (
                "the T32-spec trace (ordinary and sampled steps, verify steps whose drafts fit idle lanes) and the T64 "
                "trace (the other verify steps, argmax only), each captured once at warmup"
            )
        return (
            f"spec_verify={m!r}: {traces}; T64 = DecodeKVWrite(rows=64, KV write {kv!r}), FlashMLA A'' (rows bitwise "
            f"the T32 rows); ring_gather 'safe' (F3N R1); {drafting}"
        )

    def drafts_all_lanes(self, live_lanes: Sequence[int], acceptance: Optional[float] = None) -> bool:
        """``generator_api.MotifGenerator.drafts_all_lanes`` (design T7, §4.7; review edits R-E3, R-E9): may the bridge
        propose a draft for EVERY live lane of the next step? Host only, no state change:
        ``verify_plan.drafts_all_lanes`` with this launch's mode. False without speculation and in ``packed`` (the
        bridge keeps its idle-lane budget, so every draft fits the 32-lane trace); True in ``wide``; in ``auto`` True
        iff the distinct live lanes reach ``settings.wide_min_lanes`` when set, else ``c* =
        verify_plan.crossover_lanes(acceptance, cfg.wide_step_ratio)`` (17..33, 33 = never; ``acceptance=None``: the
        prior ``settings.spec_alpha_prior`` = 0.85, c* = 19 at r = 1.13). Raises on a live lane outside ``[0, 32)`` or
        an acceptance outside ``[0, 1]``."""
        s = self.settings
        mode = self.spec_verify if (self.spec_launch and self.wide_path is not None) else "packed"
        return VP.drafts_all_lanes(
            live_lanes,
            spec_verify=mode,
            ratio=float(self.cfg.wide_step_ratio),
            acceptance=acceptance,
            min_lanes=getattr(s, "wide_min_lanes", None),
            prior=float(getattr(s, "spec_alpha_prior", api.DEFAULT_SPEC_ALPHA_PRIOR)),
        )

    def verify_kind(self, batch: api.SpecDecodeBatch, *, want_logits: bool = False, sampling: Any = None) -> str:
        """The trace a :meth:`decode_forward_spec` step of ``batch`` runs on when no ``path`` is given: ``"spec"`` (the
        T32-spec trace) or ``"wide"`` (the T64 trace); ``verify_plan.choose_verify_kind`` with :attr:`spec_verify` and
        the T32 path's partner rule (host only). ``packed`` / no T64 path: always ``"spec"``; ``wide``: always
        ``"wide"``; ``auto``: ``"wide"`` only for a verify step whose drafts do not all fit idle lanes and that wants
        neither logits nor sampling."""
        mode = self.spec_verify if self.wide_path is not None else "packed"
        return VP.choose_verify_kind(
            batch,
            mode,
            bool(want_logits),
            sampling,
            kv_mode=self.spec_kv_mode,
            lanes_per_row=int(self.cfg.lanes_per_row),
        )

    # ==============================================================================================================
    # device sampling (docs/sampling/DEVICE_SAMPLER.md §6)
    # ==============================================================================================================
    @property
    def supports_device_sampling(self) -> bool:
        """:meth:`enable_device_sampling` / :meth:`decode_forward_sampled` are implemented (the bridge checks it)."""
        return True

    @property
    def device_sampling(self) -> bool:
        """The decode trace holds the exact device sampler (:meth:`enable_device_sampling` ran)."""
        return self.sampler is not None

    def enable_device_sampling(self, **sampler_kwargs) -> MotifDeviceSampler:
        """Build the device sampler (``MotifDeviceSampler(mesh, cfg, ccl=model.ccl, **sampler_kwargs)``; defaults:
        K = 64, W = 512, logprobs, device RNG, the Gumbel full-vocab path; the vLLM bridge passes ``rng_seed`` = vLLM's
        ``--seed`` for the unseeded lanes' host RNG) so that the decode warmup compiles it and the capture puts it into
        the decode trace. Idempotent. Refused once a decode trace exists (the trace would not hold it:
        ``release_traces()`` first); paths already warmed eagerly are marked unwarmed, so the capture runs the eager
        step (which compiles the sampler's programs) again first."""
        if self.sampler is not None:
            return self.sampler
        if self.trace_captured:
            raise RuntimeError("enable_device_sampling after the decode trace capture: release_traces() first")
        head = self.model.head
        if getattr(head, "vocab_split", "mesh") != "mesh":
            raise ValueError(f"the device sampler needs the LM head's 'mesh' vocab split, got {head.vocab_split!r}")
        t0 = time.time()
        self.sampler = MotifDeviceSampler(self.mesh_device, self.cfg, ccl=self.model.ccl, **sampler_kwargs)
        for p in self._paths.values():
            p.warmed = False
        s = self.sampler
        self.log(
            f"device sampling on: K {s.K}, W {s.W}, logprobs {s.logprobs}, rng {s.rng}, gumbel {s.gumbel} "
            f"({time.time() - t0:.1f} s); every decode step of the trace runs the sampler"
        )
        return s

    def _require_sampler(self) -> MotifDeviceSampler:
        if self.sampler is None:
            raise RuntimeError("device sampling is off (enable_device_sampling has not run before the decode warmup)")
        return self.sampler

    def _take_so(self) -> Optional[SamplerOutput]:
        so, self._pending_so = self._pending_so, None
        return so

    def _head_and_sample(self, X):
        """The plain step's head with the sampler: TILE logits -> ROW_MAJOR host logits ``rm`` + ``sampler.sample``
        (stashed for :meth:`_take_so`); the same head ops as ``MotifModel.decode`` (``forward_decode(row_major=True)``
        = ``forward_decode`` + ``logits_rm``), so ``rm`` is bitwise what the step without the sampler returns."""
        head = self.model.head
        try:
            lg = head.forward_decode(X)  # TILE [1, 1, 32, 6880]
        finally:
            _free(X)
        rm = None
        try:
            rm = head.logits_rm(lg)
            self._pending_so = self.sampler.sample(lg)
        except BaseException:
            _free(rm)
            raise
        finally:
            _free(lg)
        return rm

    def _finish_sampled(self, kind: str, positions: torch.Tensor, res: SampleResult, rm) -> SampleResult:
        """Counters, the host fallback of the flagged ACTIVE lanes (exact, the step's 53-bit uniform; the logits are
        read only then) and the test observer."""
        smp = self.sampler
        active = positions >= 0
        cache: List[torch.Tensor] = []

        def logits() -> torch.Tensor:
            if not cache:
                cache.append(self.model.head.logits_to_host(rm))
            return cache[0]

        st = self.stats
        st["sampled_steps"] += 1
        act = active.tolist()
        st["sampled_lanes"] += sum(act)
        if smp.gumbel:
            st["gumbel_lanes"] += sum(1 for on, lp in zip(act, smp.lane_params) if on and lp.full_support)
        flagged = res.flags & active
        if bool(flagged.any()):
            res = smp.resolve(res, logits, active=active)
            st["fallback_steps"] += 1
            st["fallback_lanes"] += int(flagged.sum())
        if self.sampling_observer is not None:
            self.sampling_observer(kind, positions, res, logits)
        return res

    def sampling_stats(self) -> Dict[str, Any]:
        """The device sampler's counters (``steps``, ``flagged_*``, ``resolved_*``, ``param_uploads``) next to the
        generator's sampled / Gumbel / fallback counters, or ``{}`` without device sampling."""
        if self.sampler is None:
            return {}
        st = self.stats
        out: Dict[str, Any] = {k: st[k] for k in ("sampled_steps", "sampled_lanes", "gumbel_lanes", "fallback_steps",
                                                  "fallback_lanes")}  # fmt: skip
        out.update({f"sampler_{k}": v for k, v in self.sampler.stats.items()})
        n = max(1, st["sampled_steps"])
        out["fallback_step_rate"] = round(st["fallback_steps"] / n, 6)
        out["fallback_lane_rate"] = round(st["fallback_lanes"] / max(1, st["sampled_lanes"]), 6)
        return out

    @property
    def warmed_shapes(self) -> Set[PrefillShape]:
        """The prefill shapes ``warmup_prefill`` compiled: solo ``(path, bucket)`` and packed shapes."""
        return set(self._warmed)

    def prefill_shapes(self) -> List[PrefillShape]:
        """Every solo ``(path, bucket)`` a prefill chunk can have (what ``warmup_prefill`` compiles first): sp0 for
        every bucket of ``cfg.prefill_span_buckets``, sp1 for those ``<= max_sp1_bucket``; ascending buckets, sp0
        first."""
        top = self.max_sp1_bucket
        out: List[PrefillShape] = []
        for b in self.cfg.prefill_span_buckets:
            out.append((PP.SP0, int(b)))
            if int(b) <= top:
                out.append((PP.SP1, int(b)))
        return out

    def packed_shapes(self) -> List[PrefillShape]:
        """Every packed pass shape (P5; ``cfg.packed_prefill_shapes()``: ``("pk0", T, S)``, then ``("pk1", T, S,
        tails)`` with both SWA tail variants, review edit R-E2) the planner can emit with the config's segment sizes
        and pass cap: what ``warmup_prefill`` compiles after the solo shapes while :attr:`packed_prefill` is on."""
        return [tuple(s) for s in self.cfg.packed_prefill_shapes()]

    def required_prefill_shapes(self) -> List[PrefillShape]:
        """The shapes the decode capture requires warmed: :meth:`prefill_shapes`, plus :meth:`packed_shapes` while
        :attr:`packed_prefill` is on."""
        return self.prefill_shapes() + (self.packed_shapes() if self.packed_prefill else [])

    # ==============================================================================================================
    # KV pool (GEN-2)
    # ==============================================================================================================
    def allocate_kv_cache(self, *, num_blocks: int, block_size: int, num_layers: int) -> MotifKVPool:
        if self._pool is not None:
            raise RuntimeError("allocate_kv_cache called twice")
        if int(num_layers) != self.num_layers:
            raise ValueError(f"allocate_kv_cache for {num_layers} layers, the generator runs {self.num_layers}")
        self.cfg.set_kv_geometry(int(num_blocks), int(block_size))  # validates the block size (32 / 64)
        t0 = time.time()
        self._pool = self.model.allocate_kv_caches(int(num_blocks), int(block_size), self.cfg.dtypes.kv_cache)
        self.timings["allocate_kv_cache_s"] = time.time() - t0
        mtp = " + the MTP layer" if self._pool.mtp is not None else ""
        per_chip = self.cfg.kv_cache_bytes_per_chip() * (self.num_layers + self._pool.mtp_layers) // self.num_layers
        self.log(
            f"KV pool: {num_layers}{mtp} x [{num_blocks}, 1, {block_size}, {self.cfg.kv_latent_dim}] "
            f"{self.cfg.dtypes.kv_cache_name} ({per_chip / 1e9:.2f} GB per chip) in "
            f"{self.timings['allocate_kv_cache_s']:.1f} s; prefill span cap {self.max_prefill_span}, A "
            f"{self.prefill_alignment}, decode path {self.serving_path}"
        )
        return self._pool

    def _check_pool(self, kv_cache) -> MotifKVPool:
        if self._pool is None:
            raise RuntimeError("the KV pool is not allocated (allocate_kv_cache has not run)")
        if kv_cache is not self._pool:
            raise ValueError("kv_cache must be the handle allocate_kv_cache returned")
        return self._pool

    # ==============================================================================================================
    # prefill: planning (host only)
    # ==============================================================================================================
    def plan_row(self, start: int, end: int) -> PP.RowPlan:
        """The chunks of row ``[start, end)``: ``cfg.plan_prefill_row``, re-planned with the span cap
        :attr:`max_sp1_bucket` when an sp1 chunk would exceed it (configs whose span cap is ``max_model_len``)."""
        cfg = self.cfg
        plan = cfg.plan_prefill_row(int(start), int(end))
        top = self.max_sp1_bucket
        if any(c.is_sp1 and c.bucket > top for c in plan.chunks):
            plan = PP.plan_prefill_row(
                int(start),
                int(end),
                block_size=cfg.kv_block_size,
                align=cfg.prefill_resume_alignment,
                buckets=tuple(b for b in cfg.prefill_span_buckets if b <= top),
                span_cap=top,
                swa_tail=cfg.prefill_swa_tail,
                cost=PP.prefill_cost_model(cfg.prefill_cost_table, sp1_s_per_row_key=cfg.prefill_sp1_s_per_row_key),
            )
        return plan

    def plan_prefill_batch(self, requests: Sequence[api.PrefillRequest]) -> PrefillBatchPlan:
        """Every host check and table of one ``prefill_forward_batch`` call, before any device op (module docstring):
        raises ``ValueError`` (bad rows) or ``RuntimeError`` (a solo shape the warmup did not compile, after the
        capture). The passes: with :attr:`packed_prefill`, ``prefill_plan.plan_prefill_passes`` with the config's
        segment sizes / pass cap and, once a decode trace is captured, the warmed shapes as ``allowed`` (a packed pass
        of another shape becomes solo chunks); without it, one solo pass per chunk in writer-first row order. Every
        pass's tables are built here (packed: ``attention.packed_host_tables``, which checks every segment's slice).

        Packing never refuses a call: the rows were already checked (``order_prefill_requests`` accepted them), so a
        packed planner or packed-table check that still fails (``AssertionError`` / ``ValueError``: a planner bug, e.g.
        ``prefill_plan._check_passes``) runs the call as with packing off, one solo pass per chunk in writer-first
        order; :attr:`PrefillBatchPlan.plan_error` names the failure and executing the call counts
        ``packed_plan_errors`` and logs it (P5 review, finding 6)."""
        reqs = api.check_prefill_batch(requests)
        cfg, bs = self.cfg, int(self.cfg.kv_block_size)
        num_blocks = None if self._pool is None else int(self._pool.num_blocks)
        captured = self.trace_captured
        jobs: List[PrefillRowJob] = []
        chunk_shapes: Set[PrefillShape] = set()
        for i, r in enumerate(reqs):
            e = r.end
            if e > self.max_prefill_len:
                raise ValueError(f"row {i}: prompt of {e} tokens exceeds max_model_len {self.max_prefill_len}")
            check_prefill_page_table(r.page_table, e, block_size=bs, num_blocks=num_blocks, row=i)
            check_token_ids(r.tokens, cfg)
            plan = self.plan_row(int(r.start), e)
            tables = [chunk_host_tables(cfg, plan, c, r.page_table) for c in plan.chunks]
            for c in plan.chunks:
                chunk_shapes.add((c.path, int(c.bucket)))
            jobs.append(PrefillRowJob(index=i, request=r, plan=plan, tables=tables))
        packed = bool(self.packed_prefill)
        if captured and not packed:
            self._refuse_unwarmed(chunk_shapes)
        plans = [j.plan for j in jobs]
        order = PP.order_prefill_requests(reqs, plans, bs)
        passes: Optional[List[PP.PrefillPass]] = None
        packed_tables: Dict[int, PackedHostTables] = {}
        plan_error: Optional[str] = None
        if packed:
            try:
                passes = PP.plan_prefill_passes(
                    reqs,
                    plans,
                    block_size=bs,
                    max_seg=cfg.pack_max_seg,
                    max_tokens=cfg.pack_tokens_cap,
                    pk1=cfg.pack_pk1,
                    allowed=self._warmed if captured else None,
                    cost=self._pack_cost,
                    seg_buckets=cfg.pack_seg_buckets,
                    sp1_seg_buckets=cfg.pack_sp1_seg_buckets,
                    swa_tail=cfg.prefill_swa_tail,
                )
                packed_tables = {
                    n: packed_host_tables(cfg, p, reqs, plans) for n, p in enumerate(passes) if p.is_packed
                }
            except (AssertionError, ValueError) as e:  # a packed-planner bug: the rows themselves passed every check
                plan_error = f"{type(e).__name__}: {e}"
                passes, packed_tables = None, {}
        if passes is None:  # packing off, or the packed plan failed: one solo pass per chunk, writer-first rows
            if captured and packed:
                self._refuse_unwarmed(chunk_shapes)
            passes = PP.solo_prefill_passes(plans, order)
        elif captured:  # solo passes (fallbacks included) still need their warmed (path, bucket)
            self._refuse_unwarmed({p.shape for p in passes if not p.is_packed})
        tables: List[Any] = []
        for n, p in enumerate(passes):
            if p.is_packed:
                tables.append(packed_tables[n])
            else:
                g = p.segments[0]
                tables.append(jobs[g.row].tables[g.chunk_index])
        return PrefillBatchPlan(
            jobs=jobs, order=order, shapes={p.shape for p in passes}, passes=passes, tables=tables, packed=packed,
            plan_error=plan_error,
        )  # fmt: skip

    def _refuse_unwarmed(self, shapes: Set[PrefillShape]) -> None:
        missing = sorted(s for s in shapes if s not in self._warmed)
        if missing:
            # compiling a new prefill program after the decode capture can corrupt the trace (plugin contract)
            raise RuntimeError(f"prefill shapes {missing} were not compiled before the decode trace capture")

    # ==============================================================================================================
    # prefill (GEN-4; features design §3.7)
    # ==============================================================================================================
    def prefill_forward(
        self, request: api.PrefillRequest, *, kv_cache: Any, enable_trace: bool = False
    ) -> torch.Tensor:
        """``prefill_forward_batch([request])[0]`` (the ``PrefillRequest`` contract; ``start`` may be > 0)."""
        return self.prefill_forward_batch([request], kv_cache=kv_cache, enable_trace=enable_trace)[0]

    def _prefill_page_table(self, page_table: torch.Tensor, bucket: int, seq_len: int):
        """The draft-1 single-shot fill table (:func:`prefill_page_table_host`) as a replicated ``[1, n]`` int32 device
        tensor, for reference runs through ``MotifModel.prefill`` (tests, validation scripts). Serving never uses it."""
        n = self.cfg.prefill_page_table_entries(bucket)
        pt = prefill_page_table_host(page_table, n, seq_len, self.cfg.kv_block_size)
        return ttnn.from_torch(
            pt,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def prefill_forward_batch(
        self, requests: Sequence[api.PrefillRequest], *, kv_cache: Any, enable_trace: bool = False
    ) -> torch.Tensor:
        """All rows of one plugin step (module docstring): ``[B, vocab]`` host logits of each row's ``end - 1``, in
        input order. ``enable_trace`` (plugin ``trace_mode="all"``) is ignored: prefill is eager. Timings:
        ``last_prefill_plan_s`` (host checks, plan and tables), ``last_prefill_s`` (the passes: device work and host
        reads, from the first upload to the last logits row)."""
        pool = self._check_pool(kv_cache)
        t0 = time.time()
        batch = self.plan_prefill_batch(requests)
        self.last_prefill = batch
        out: List[Optional[torch.Tensor]] = [None] * len(batch.jobs)
        t1 = time.time()
        for p, host in zip(batch.passes, batch.tables):
            if p.is_packed:
                for row, lg in self._run_packed(batch, p, host, pool).items():
                    out[row] = lg
                continue
            g = p.segments[0]
            job = batch.jobs[g.row]
            ch = job.plan.chunks[g.chunk_index]
            lg = self._run_chunk(job, ch, host, pool)
            if ch.last:
                out[g.row] = lg
        t2 = time.time()
        missing = [i for i, lg in enumerate(out) if lg is None]
        if missing:
            raise AssertionError(f"prefill rows {missing} got no logits (a plan without their last chunk)")
        self._count_prefill(batch)
        self.timings["last_prefill_plan_s"] = t1 - t0
        self.timings["last_prefill_s"] = t2 - t1
        return torch.stack(out)

    def _count_prefill(self, batch: PrefillBatchPlan) -> None:
        st = self.stats
        st["prefill_calls"] += 1
        st["prefill_rows"] += len(batch.jobs)
        st["prefill_chunks"] += batch.chunks
        st["sp1_chunks"] += sum(c.is_sp1 for j in batch.jobs for c in j.plan.chunks)
        st["recomputed_rows"] += sum(j.plan.recompute for j in batch.jobs)
        packed = batch.packed_passes
        st["solo_passes"] += len(batch.passes) - len(packed)
        st["packed_calls"] += int(bool(packed))
        st["packed_passes"] += len(packed)
        st["packed_pk1_passes"] += sum(p.kind == PP.PK1 for p in packed)
        st["packed_segments"] += sum(len(p.segments) for p in packed)
        st["packed_dummy_segments"] += sum(p.dummies for p in packed)
        st["packed_padding_rows"] += sum(p.padding_rows for p in packed)
        if batch.plan_error is not None:
            st["packed_plan_errors"] += 1
            self.log(
                f"error: the packed prefill plan of a {len(batch.jobs)}-row call failed ({batch.plan_error}); the call "
                f"ran per row in writer-first order instead (a packed-planner bug: please report; rows (start, end) "
                f"{[(int(j.request.start), int(j.request.end)) for j in batch.jobs][:8]}"
                f"{' ...' if len(batch.jobs) > 8 else ''})"
            )
        fallbacks = batch.fallbacks
        st["packed_solo_fallbacks"] += len(fallbacks)
        for shape in sorted({p.fallback for p in fallbacks} - self._fallback_logged):
            self._fallback_logged.add(shape)
            self.log(
                f"packed prefill: pass shape {shape} was not warmed before the decode capture (packing was off at the "
                "warm-up?): its segments run as solo chunks (logged once per shape)"
            )

    def _run_chunk(self, job: PrefillRowJob, ch: PP.ChunkPlan, host: ChunkHostTables, pool: MotifKVPool):
        """One chunk: inputs -> layers -> (last chunk) head + host logits -> (MTP) KV-only fill. Returns the logits of
        the chunk's last real row on the last chunk, else None. Every tensor of the chunk is freed on return."""
        model, req = self.model, job.request
        inp = tok = X = tile = hn = nxt = None
        logits = None
        try:
            inp = model.chunk_inputs(host)
            tok = model.embed.prefill_tokens_device(req.tokens[ch.start : ch.end], ch.bucket)
            X = model.prefill_chunk(tok, chunk=inp, kv_caches=pool)
            _free(tok)
            tok = None
            if self.chunk_observer is not None:
                self.chunk_observer(job, ch, X)
            if ch.last:
                tile = model.head.forward_prefill(X, ch.head_row)
                logits = model.head.prefill_logits_to_host(tile, ch.head_row)
                _free(tile)
                tile = None
            if model.mtp is not None:
                # t_{p+1} of every row; the row's last known position takes the host argmax (design §3.6.3, §3.7.1)
                stand_in = int(torch.argmax(logits)) if ch.end == req.end else None
                ids = mtp_next_tokens(req.tokens, ch.start, ch.end, stand_in)
                nxt = model.embed.rows_tokens_device(ids, ch.bucket)
                hn = model.head.stream_mean_norm(X)
                model.mtp.fill_kv_prefill(hn, nxt, kv_cache=pool.mtp, chunk=inp)
                self.stats["mtp_fills"] += 1
        finally:
            _free(tok, X, tile, hn, nxt)
            if inp is not None:
                inp.free()
        return logits

    def _run_packed(
        self, batch: PrefillBatchPlan, p: PP.PrefillPass, host: PackedHostTables, pool: MotifKVPool
    ) -> Dict[int, torch.Tensor]:
        """One packed pass (module docstring): inputs -> layers at ``T`` -> the LM head per segment that ends its row
        -> (MTP) one KV-only fill of the pass. Returns ``{row index: host logits of its last position}``. Every tensor
        of the pass is freed on return."""
        model, cfg = self.model, self.cfg
        reqs = batch.requests
        T = int(p.tokens)
        inp = tok = X = tile = hn = nxt = None
        logits: Dict[int, torch.Tensor] = {}
        try:
            inp = model.chunk_inputs(host)
            tok = model.embed.prefill_tokens_device(PP.pass_tokens(p, reqs, cfg.pad_token_id), T)
            X = model.prefill_chunk(tok, chunk=inp, kv_caches=pool)
            _free(tok)
            tok = None
            if self.pass_observer is not None:
                self.pass_observer(batch, p, X)
            for k, row in p.head_rows():  # packed row k S + end - 1 - start of each segment that ends its row
                tile = model.head.forward_prefill(X, row)
                logits[int(p.segments[k].row)] = model.head.prefill_logits_to_host(tile, row)
                _free(tile)
                tile = None
            if model.mtp is not None:
                # t_{p+1} of every segment row; a row's last known position takes the host argmax (design §3.4 "MTP")
                ids = []
                for g in p.segments:
                    req = reqs[g.row]
                    stand_in = int(torch.argmax(logits[g.row])) if g.end == req.end else None
                    ids.append(mtp_next_tokens(req.tokens, g.start, g.end, stand_in))
                nxt = model.embed.rows_tokens_device(PP.pass_rows(p, ids, cfg.pad_token_id), T)
                hn = model.head.stream_mean_norm(X)
                model.mtp.fill_kv_prefill(hn, nxt, kv_cache=pool.mtp, chunk=inp)
                self.stats["mtp_fills"] += 1
        finally:
            _free(tok, X, tile, hn, nxt)
            if inp is not None:
                inp.free()
        return logits

    # ==============================================================================================================
    # decode paths (GEN-3; features design §3.8, §3.11)
    # ==============================================================================================================
    def decode_paths(self) -> List[DecodeKey]:
        """The ``(kind, mode)`` decode paths :meth:`warmup_decode` prepares and captures, in capture order:
        :attr:`serving_paths` first (in ``auto`` the T32-spec path, then the T64 path), then :attr:`extra_decode_paths`
        (deduplicated)."""
        out = list(self.serving_paths)
        for kind, mode in self.extra_decode_paths:
            key = (str(kind), _check_mode(str(mode)))
            if key[0] not in DECODE_KINDS:
                raise ValueError(f"decode path kind must be one of {DECODE_KINDS}, got {key[0]!r}")
            if key[0] in SPEC_KINDS and self.model.mtp is None:
                raise ValueError(f"a {key[0]} decode path needs the MTP layer")
            if key not in out:
                out.append(key)
        return out

    def _resolve_key(self, kind: Any, path: Optional[DecodeKey]) -> DecodeKey:
        """The decode path of a call: ``path`` when given (its kind must be ``kind``: one kind or a tuple of kinds;
        None = any), else :attr:`serving_path` (``kind=None``) or the default path of the first kind."""
        kinds = DECODE_KINDS if kind is None else ((kind,) if isinstance(kind, str) else tuple(kind))
        if path is not None:
            key = (str(path[0]), _check_mode(str(path[1])))
            if key[0] not in DECODE_KINDS or key[0] not in kinds:
                raise ValueError(f"decode path {path!r} is not a {' / '.join(kinds)} path")
            return key
        if kind is None:
            return self.serving_path
        k = kinds[0]
        return (k, self.decode_kv_mode if k == PLAIN else self.spec_kv_mode)

    @property
    def wide_rows_per_dp(self) -> int:
        """Rows per DP row of the T64 step: ``2 * cfg.lanes_per_row`` = 16 (``[8 anchors | 8 drafts]``)."""
        return 2 * int(self.cfg.lanes_per_row)

    def _path_host_inputs(self, kind: str, mode: str, tokens: torch.Tensor, positions: torch.Tensor, page_table=None):
        """Host mesh tensors of a path's own persistent inputs: ``tokens`` (0 on inactive rows), the RoPE rows ``rot``
        of ``positions``, and for the draft-1 plain ``row`` path its ``cur`` / ``pt`` (every other path keeps FlashMLA's
        inputs in its ``DecodeKVWrite``). ``tokens`` / ``positions``: the 32 lanes, or on the wide path the T64 step's
        64 physical rows (``16 r + j``: 16 per DP row, ``verify_plan.WideStepPlan.tokens`` / ``.positions``)."""
        cfg, mesh = self.cfg, self.mesh_device
        pos = positions.to(torch.int32)
        active = pos >= 0
        tok = torch.where(active, tokens.to(torch.int32), torch.zeros_like(pos))
        if kind == WIDE:
            rpd = self.wide_rows_per_dp
            return {
                "tokens": self.model.embed.decode_tokens_host(tok, rows_per_dp=rpd),
                "rot": shard_lanes(positions_to_rot_idxs(pos, cfg, rows_per_dp=rpd), cfg, mesh, dtype=ttnn.uint32,
                                   device=None),  # fmt: skip
            }
        out = {
            "tokens": self.model.embed.decode_tokens_host(tok),
            "rot": shard_lanes(positions_to_rot_idxs(pos, cfg), cfg, mesh, dtype=ttnn.uint32, device=None),
        }
        if kind == PLAIN and mode == "row":
            pt = torch.where(active[:, None], page_table.to(torch.int32), torch.zeros_like(page_table))
            out["cur"] = shard_lanes(pos.contiguous(), cfg, mesh, dtype=ttnn.int32, device=None)  # [8] per DP row
            out["pt"] = shard_lanes(pt.contiguous(), cfg, mesh, dtype=ttnn.int32, device=None)
        return out

    def _path_host_rows(self, kind: str, mode: str, tokens: torch.Tensor, positions: torch.Tensor, page_table=None):
        """``host_staging="fast"``: the torch values and ttnn dtypes of :meth:`_path_host_inputs` (``{name: (rows,
        dtype)}``, every one ROW_MAJOR with the DP-row mapper), before any host mesh tensor is built. Identical
        values: the same helpers (:func:`decode_token_rows`, :func:`positions_to_rot_idxs`) on the same inputs."""
        cfg = self.cfg
        pos = positions.to(torch.int32)
        active = pos >= 0
        tok = torch.where(active, tokens.to(torch.int32), torch.zeros_like(pos))
        rpd = self.wide_rows_per_dp if kind == WIDE else None
        out = {
            "tokens": (decode_token_rows(tok, cfg, n_streams=self.model.embed.n_streams, rows_per_dp=rpd), ttnn.uint32),
            "rot": (positions_to_rot_idxs(pos, cfg, rows_per_dp=rpd), ttnn.uint32),
        }
        if kind == PLAIN and mode == "row":
            pt = page_table.to(torch.int32) * active[:, None]  # == torch.where(active, pt, 0) on int32
            out["cur"] = (pos.clone(), ttnn.int32)  # a copy: ``pos`` may be the caller's tensor (kept in host_last)
            out["pt"] = (pt.contiguous(), ttnn.int32)
        return out

    def _copy_path_inputs(self, p: DecodePath, rows) -> None:
        """Copy ``rows`` (:meth:`_path_host_rows`) into ``p.inputs``, skipping the inputs whose values equal the ones
        copied last (``host_staging="fast"``)."""
        if self._dp_mapper is None:
            self._dp_mapper = dp_row_mapper(self.cfg, self.mesh_device)
        last = p.host_last
        for k, (v, dtype) in rows.items():
            prev = last.get(k)
            if prev is not None and torch.equal(prev, v):
                self.stats["input_copies_skipped"] += 1
                continue
            host = ttnn.from_torch(v, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self._dp_mapper)
            last.pop(k, None)  # a failed copy leaves no stale record
            ttnn.copy_host_to_device_tensor(host, p.inputs[k])
            last[k] = v
            self.stats["input_copies"] += 1

    def _write_path_inputs(self, p: DecodePath, tokens, positions, page_table=None) -> None:
        """The path's own inputs (``tokens`` / ``rot``, and ``cur`` / ``pt`` on the plain ``row`` path) for one step,
        per :attr:`host_staging`."""
        if self.host_staging == "fast":
            self._copy_path_inputs(p, self._path_host_rows(p.kind, p.mode, tokens, positions, page_table))
            return
        h = self._path_host_inputs(p.kind, p.mode, tokens, positions, page_table)
        for k, v in h.items():
            ttnn.copy_host_to_device_tensor(v, p.inputs[k])
        self.stats["input_copies"] += len(h)

    def _stage_path(self, key: DecodeKey, width: int) -> DecodePath:
        """The path's persistent inputs for page-table width ``width``, allocated on first use. Refused once any trace
        is captured: a buffer allocated after a capture and kept across replays may be overwritten by them (the trace
        allocation tracker's warning), so every path is staged before the first capture (:meth:`warmup_decode`)."""
        kind, mode = key
        p = self._paths.get(key)
        if p is not None and p.width == int(width):
            return p
        if self.trace_captured:
            have = f"staged for width {p.width}" if p is not None else "not staged"
            raise RuntimeError(
                f"decode path {key} is {have} and a decode trace exists: stage every path before the capture "
                f"(extra_decode_paths + warmup_decode), or release_traces() first"
            )
        if kind in SPEC_KINDS and self.model.mtp is None:
            raise ValueError(f"the {kind} decode path needs the MTP layer")
        if kind == WIDE:  # before anything is freed or allocated: a T64 config, a split mode, R1 (ring_gather safe)
            self._check_wide_launch(mode)
        if p is not None:
            self._free_path(p)
        n = api.WIDE_ROWS if kind == WIDE else api.NUM_LANES
        h = self._path_host_inputs(
            kind, mode, torch.zeros(n, dtype=torch.int32), torch.full((n,), -1, dtype=torch.int32),
            torch.zeros(n, int(width), dtype=torch.int32),
        )  # fmt: skip
        p = DecodePath(kind=kind, mode=mode, width=int(width))
        p.inputs = {k: ttnn.to_device(v, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for k, v in h.items()}
        if kind == WIDE:  # 16 rows per DP row; the split-order KV-R gather (all_split); the A'' group inputs
            p.kv_write = DecodeKVWrite(
                self.mesh_device, self.cfg, ccl=self.model.ccl, page_table_width=int(width), mode=mode,
                rows=api.WIDE_ROWS, gather="split",
            )  # fmt: skip
        elif not (kind == PLAIN and mode == "row"):  # the draft-1 path keeps the attention's own 8-lane update
            p.kv_write = DecodeKVWrite(
                self.mesh_device, self.cfg, ccl=self.model.ccl, page_table_width=int(width), mode=mode
            )
        self._paths[key] = p
        return p

    def _free_path(self, p: DecodePath) -> None:
        if p.traced:
            raise RuntimeError(f"decode path {p.key} holds a trace: release_traces() first")
        _free(*p.inputs.values())
        if p.kv_write is not None:
            p.kv_write.deallocate()
        p.inputs, p.kv_write = {}, None
        self._paths.pop(p.key, None)

    def _write_plain(self, p: DecodePath, batch: api.DecodeBatch) -> None:
        self._write_path_inputs(p, batch.tokens, batch.positions, batch.page_table)
        if p.kv_write is not None:
            p.kv_write.write_step(KVWriteStep.ordinary(batch.positions, batch.page_table))

    def _write_spec(self, p: DecodePath, ps: SpecPass) -> None:
        """One spec pass's inputs (the plan's checks ran already: ``write_step`` skips its own validation)."""
        self._write_path_inputs(p, ps.tokens, ps.step.positions)
        p.kv_write.write_step(ps.step, validate=False)

    def _write_wide(self, p: DecodePath, plan: VP.WideStepPlan) -> None:
        """One T64 step's inputs (``verify_plan.plan_wide_step`` ran every check already): the 64 physical rows' tokens
        (``[4, 16]`` per DP row) and RoPE rows, then the ``DecodeKVWrite(rows=64)`` inputs (``cur_pos [16]`` /
        ``page_table [16, W]`` per DP row, the call inputs, the A'' group inputs; unchanged inputs are skipped)."""
        self._write_path_inputs(p, plan.tokens, plan.positions)
        p.kv_write.write_step(plan.step, validate=False)

    def _plain_step(self, p: DecodePath, pool: MotifKVPool):
        """One plain decode step on the device (eager or inside a capture): ROW_MAJOR logits ``[1, 1, 32, 6880]``.
        With device sampling the step also runs the sampler on the TILE logits; its outputs wait in
        :meth:`_take_so` (the caller owns them)."""
        d, w = p.inputs, p.kv_write
        kw = {"kv_caches": pool}
        if self.sampler is not None:
            kw["return_streams"] = True
        if w is None:
            out = self.model.decode(d["tokens"], rot_idxs=d["rot"], cur_pos=d["cur"], page_table=d["pt"], **kw)
        else:
            out = self.model.decode(
                d["tokens"], rot_idxs=d["rot"], cur_pos=w.cur_pos, page_table=w.page_table, kv_write=w, **kw
            )
        if self.sampler is None:
            return out
        return self._head_and_sample(out)

    def _spec_step(self, p: DecodePath, pool: MotifKVPool):
        """One T32-spec step on the device (eager or inside a capture): ``(rm, a, m)`` (``MotifModel.decode_spec``).
        With device sampling the step also samples ``rm`` (ROW_MAJOR: one tilize inside the sampler); its outputs
        wait in :meth:`_take_so`."""
        out = self.model.decode_spec(p.inputs["tokens"], rot_idxs=p.inputs["rot"], kv_write=p.kv_write, kv_caches=pool)
        if self.sampler is not None:
            try:
                self._pending_so = self.sampler.sample(out[0])
            except BaseException:
                _free(*out)
                raise
        return out

    def _wide_step(self, p: DecodePath, pool: MotifKVPool):
        """One T64 step on the device (eager or inside a capture): ``(rm, a, m)`` (``MotifModel.decode_wide``; ``rm``
        = the anchors' ROW_MAJOR logits only when :attr:`wide_has_logits`, else None). With ``spec_verify="wide"`` and
        device sampling the step also samples ``rm`` (the 32 anchors, exactly the ``[1, 1, 32, 6880]`` the T32 trace
        samples); its outputs wait in :meth:`_take_so`. In ``auto`` the T64 trace holds no sampler."""
        want_rm = self.wide_has_logits
        out = self.model.decode_wide(
            p.inputs["tokens"], rot_idxs=p.inputs["rot"], kv_write=p.kv_write, kv_caches=pool, want_rm=want_rm
        )
        if self.sampler is not None and want_rm:
            try:
                self._pending_so = self.sampler.sample(out[0])
            except BaseException:
                _free(*out)
                raise
        return out

    def _device_step(self, p: DecodePath, pool: MotifKVPool):
        if p.kind == SPEC:
            return self._spec_step(p, pool)
        if p.kind == WIDE:
            return self._wide_step(p, pool)
        return self._plain_step(p, pool)

    def _replay(self, p: DecodePath) -> None:
        """Enqueue one replay of ``p``'s trace (non-blocking). F3N rule R4 (design §2.3): with several traces, a trace's
        outputs must be read before another trace replays (a later capture may have put its outputs on an earlier
        trace's intermediates). Every decode step ends in a blocking read of the outputs it replayed
        (:meth:`_read_done`); a replay of another path while :attr:`_unread` is set raises before anything is
        enqueued."""
        if self._unread is not None and self._unread != p.key:
            raise RuntimeError(
                f"F3N rule R4: decode trace {p.key} replayed while the outputs of {self._unread} are not read yet "
                f"(every decode step must end in a blocking read of its outputs)"
            )
        ttnn.execute_trace(self.mesh_device, p.trace_id, cq_id=0, blocking=False)
        self._unread = p.key

    def _spin(self, key, t_enq: float) -> Optional[float]:
        """``host_wait="spin"``: :meth:`ReplayWaiter.spin` before the step's blocking read (else nothing)."""
        if self.waiter is None:
            return None
        return self.waiter.spin(key, t_enq)

    def _waited(self, key, t_enq: float, t_read: Optional[float]) -> None:
        """The step's blocking read returned: time the replay (``host_wait="spin"``)."""
        if self.waiter is not None and t_read is not None:
            self.waiter.done(key, t_enq, t_read, time.perf_counter())

    def _read_done(self) -> None:
        """The step's blocking read returned: every output it replayed is on the host (F3N rule R4)."""
        self._unread = None

    def _abort_step(self) -> None:
        """A decode step raised after a replay was enqueued: wait for the device before another trace may replay (R4);
        if even that fails, :attr:`_unread` stays set and a replay of another trace keeps refusing."""
        if self._unread is None:
            return
        try:
            ttnn.synchronize_device(self.mesh_device)
        except Exception:
            return
        self._unread = None

    def _check_trace_use(self, p: Optional[DecodePath], pool, width: int, enable_trace: bool) -> bool:
        use = bool(enable_trace) and p is not None and p.traced
        if use and width != p.width:
            raise ValueError(f"page-table width {width} != the traced width {p.width}")
        if use and p.pool is not pool:
            raise ValueError("the decode trace was captured with another KV pool")
        return use

    # ==============================================================================================================
    # decode (GEN-3)
    # ==============================================================================================================
    def decode_forward(
        self, batch: api.DecodeBatch, *, kv_cache: Any, enable_trace: bool, path: Optional[DecodeKey] = None
    ) -> torch.Tensor:
        """One decode step for every lane (``generator_api.MotifGenerator.decode_forward``): host logits ``[32,
        220160]`` (a fresh tensor). On a speculating launch the serving path is the spec trace (``spec_verify`` "wide":
        the T64 trace): the step is an ordinary spec step (``decode_forward_spec`` without drafts, the same logits; the
        MTP layer also writes its cache, design G8). ``path`` (tests): run a specific ``(kind, mode)`` path instead of
        the serving one."""
        pool = self._check_pool(kv_cache)
        key = self._resolve_key(None, path)
        if key[0] in SPEC_KINDS:
            res = self.decode_forward_spec(
                api.SpecDecodeBatch.from_decode_batch(batch), kv_cache=pool, enable_trace=enable_trace,
                want_logits=True, path=key,
            )  # fmt: skip
            return res.logits
        # every host check before the path is staged or written (the spec path's plan_spec_step does the same): an
        # active lane's token outside [0, vocab) would be embedded out of range (a negative one silently as padding)
        self._check_plain_batch(batch, pool)
        use_trace = self._check_trace_use(self._paths.get(key), pool, batch.page_table_width, enable_trace)
        p = self._stage_path(key, batch.page_table_width)
        self._write_plain(p, batch)
        self.stats["decode_steps"] += 1
        head = self.model.head
        if use_trace:
            self._replay(p)
            t_enq = time.perf_counter()
            try:
                t_read = self._spin(p.key, t_enq)
                lg = head.logits_to_host(p.out)  # blocking read of the trace output (fresh host tensor)
                self._waited(p.key, t_enq, t_read)
            except BaseException:
                self._abort_step()
                raise
            self._read_done()
            return lg
        out = self._plain_step(p, pool)
        so = self._take_so()  # a host-sampled step on a device-sampling launch: the sampler's outputs are not used
        try:
            return head.logits_to_host(out)
        finally:
            _free(out, so)

    def _check_plain_batch(self, batch: api.DecodeBatch, pool: MotifKVPool) -> None:
        """The plain path's host checks (before the path is staged or written)."""
        if int(batch.positions.max()) >= self.cfg.max_model_len:
            raise ValueError(f"decode position {int(batch.positions.max())} >= max_model_len {self.cfg.max_model_len}")
        check_token_ids(batch.tokens[batch.positions >= 0], self.cfg)
        check_decode_page_tables(
            batch.positions, batch.page_table, block_size=self.cfg.kv_block_size, num_blocks=pool.num_blocks
        )

    def decode_forward_sampled(
        self,
        batch: api.DecodeBatch,
        sampling: LaneSampling,
        *,
        kv_cache: Any,
        enable_trace: bool,
        path: Optional[DecodeKey] = None,
    ) -> SampleResult:
        """One device-sampled decode step (DEVICE_SAMPLER.md §6.2): the host checks and input writes of
        :meth:`decode_forward`, then ``sampler.set_params(*sampling)`` (``sampling`` = lane-ordered ``(temperature,
        top_p, top_k, seeds)`` in the plugin's conventions; a device write only when they changed) and
        ``set_positions(batch.positions)`` (the RNG counters), the replay (or an eager step), one 1 KB read and the
        exact host fallback of the flagged active lanes (logits read only then). Returns the lane-order
        :class:`SampleResult` (``tokens`` int64 ``[32]``, raw ``logprobs``; inactive lanes are don't-care). On a
        speculating launch this is the ordinary step of the spec trace (:meth:`decode_forward_spec` with
        ``sampling``; ``spec_verify="wide"``: of the T64 trace, sampling its anchor rows)."""
        smp = self._require_sampler()
        pool = self._check_pool(kv_cache)
        key = self._resolve_key(None, path)
        if key[0] in SPEC_KINDS:
            res = self.decode_forward_spec(
                api.SpecDecodeBatch.from_decode_batch(batch), kv_cache=pool, enable_trace=enable_trace,
                want_logits=False, path=key, sampling=sampling,
            )  # fmt: skip
            return res.sample
        self._check_plain_batch(batch, pool)
        use_trace = self._check_trace_use(self._paths.get(key), pool, batch.page_table_width, enable_trace)
        if use_trace and self._paths[key].so is None:
            raise RuntimeError("the decode trace was captured without the device sampler (release_traces() first)")
        p = self._stage_path(key, batch.page_table_width)
        t0 = time.perf_counter()
        self._write_plain(p, batch)
        smp.set_params(*sampling)
        smp.set_positions(batch.positions)
        self.stats["decode_steps"] += 1
        dev = so = None
        try:
            t_enq = t_read = None
            if use_trace:
                self._replay(p)
                t_enq = time.perf_counter()
                rm, so_read = p.out, p.so
                t_read = self._spin(p.key, t_enq)
            else:
                rm = dev = self._plain_step(p, pool)
                so = so_read = self._take_so()
            res = smp.read(so_read)  # blocking (chip 0, 1 KB): waits for the step
            if t_enq is not None:
                self._waited(p.key, t_enq, t_read)
            res = self._finish_sampled("plain", batch.positions, res, rm)
            self._read_done()
        except BaseException:
            self._abort_step()
            raise
        finally:
            _free(dev, so)
        self.timings["last_sampled_step_ms"] = (time.perf_counter() - t0) * 1e3
        return res

    def plan_spec_step(self, batch: api.SpecDecodeBatch, *, path: Optional[DecodeKey] = None) -> SpecStepPlan:
        """:func:`plan_spec_step` with this generator's config, pool and (once traced) trace width: every host check
        and the packing of one speculative step on the T32-spec trace, before any device op."""
        key = self._resolve_key(SPEC, path)
        p = self._paths.get(key)
        return plan_spec_step(
            batch,
            mode=key[1],
            cfg=self.cfg,
            num_blocks=None if self._pool is None else int(self._pool.num_blocks),
            width=p.width if (p is not None and p.traced) else None,
            lanes_per_call=None if (p is None or p.kv_write is None) else p.kv_write.lanes_per_call,
        )

    def plan_wide_step(self, batch: api.SpecDecodeBatch, *, path: Optional[DecodeKey] = None) -> VP.WideStepPlan:
        """``verify_plan.plan_wide_step`` with this generator's config and pool, and the T64 path's geometry (once
        staged: its ``DecodeKVWrite(rows=64)``'s users per call and gather order; once traced: its width): every host
        check and the 64-row layout of one T64 step, before any device op. Raises ``ValueError`` on a bad batch."""
        key = self._resolve_key(WIDE, path)
        p = self._paths.get(key)
        w = None if p is None else p.kv_write
        return VP.plan_wide_step(
            batch,
            cfg=self.cfg,
            width=p.width if (p is not None and p.traced) else None,
            mode=key[1],
            num_blocks=None if self._pool is None else int(self._pool.num_blocks),
            lanes_per_call=None if w is None else w.lanes_per_call,
            gather="split" if w is None else w.gather,
        )

    def decode_forward_spec(
        self,
        batch: api.SpecDecodeBatch,
        *,
        kv_cache: Any,
        enable_trace: bool,
        want_logits: bool,
        path: Optional[DecodeKey] = None,
        sampling: Optional[LaneSampling] = None,
    ) -> api.SpecDecodeResult:
        """One decode step of a speculating launch (``generator_api.MotifGenerator.decode_forward_spec``; features
        design §3.8; module docstrings "Speculative decode" and "T64 verify"). Returns ``SpecDecodeResult`` in
        owner-lane order: ``argmax = (a0, a1)``, ``mtp_argmax = (m0, m1)``, the anchors' logits when ``want_logits``.

        Routing (host, before any device op): ``path`` when given (tests: ``("spec", mode)`` or ``("wide", mode)``),
        else :meth:`verify_kind` (``verify_plan.choose_verify_kind`` with :attr:`spec_verify`).

        * **T32** (``"spec"``): the plan (host checks, packing), then one replay of the T32-spec trace (pass 1: anchors
          on their lanes at ``n``, packed drafts on idle partner lanes at ``n + 1``) and, only when some draft found no
          idle lane, a second replay (pass 2: those drafts on their own lanes at ``n + 1``). Logits: pass 1's owner
          lanes.
        * **T64** (``"wide"``): ``plan_wide_step`` (host checks, the 64-row layout), then ONE replay of the T64 trace:
          every lane's anchor at ``n`` and its draft at ``n + 1`` on its own DP row (no idle lane needed, no overflow
          pass). Logits (``spec_verify="wide"`` only; in ``auto`` the T64 trace is argmax-only and a step that wants
          logits or sampling runs on T32): the anchor rows.

        ``sampling`` (device sampling; an ORDINARY step only -- a verify step returns argmax ids and PS-1 keeps sampled
        rows out of it): the lane-ordered ``(temperature, top_p, top_k, seeds)`` of :meth:`decode_forward_sampled`;
        the step samples every lane on device (the anchors' positions key the RNG) and returns a
        :class:`SampledSpecDecodeResult` whose ``sample`` holds the tokens (flagged active lanes resolved on the
        host); ``a`` / ``m`` are still returned for the bridge's speculation bookkeeping.

        Every step ends in one blocking read of the outputs it replayed (F3N rule R4: a replay of the other trace while
        outputs are unread is refused, :meth:`_replay`)."""
        pool = self._check_pool(kv_cache)
        if self.model.mtp is None:
            raise NotImplementedError("decode_forward_spec needs the MTP layer (the generator was built without it)")
        t0 = time.perf_counter()
        if path is not None:
            key = self._resolve_key(SPEC_KINDS, path)
        else:
            key = (self.verify_kind(batch, want_logits=want_logits, sampling=sampling), self.spec_kv_mode)
        wide = key[0] == WIDE
        if wide:
            plan = self.plan_wide_step(batch, path=key)
            passes: Sequence[Any] = (plan,)
            if want_logits and not self.wide_has_logits:
                raise ValueError(
                    f"want_logits on the T64 trace of a spec_verify={self.spec_verify!r} launch: it is argmax-only "
                    f"(steps that want logits run on the T32-spec trace)"
                )
        else:
            plan = self.plan_spec_step(batch, path=key)
            passes = plan.passes
        smp = None
        if sampling is not None:
            smp = self._require_sampler()
            if plan.is_verify:
                raise ValueError(
                    "device sampling on a verify step: a verify returns the target argmax ids (PS-1 keeps sampled "
                    "rows out of verify steps); sample on ordinary steps only"
                )
            if wide and not self.wide_has_logits:
                raise ValueError(
                    f"device sampling on the T64 trace of a spec_verify={self.spec_verify!r} launch: it holds no "
                    f"sampler (sampled steps run on the T32-spec trace)"
                )
        use_trace = self._check_trace_use(self._paths.get(key), pool, batch.page_table_width, enable_trace)
        if smp is not None and use_trace and self._paths[key].so is None:
            raise RuntimeError("the decode trace was captured without the device sampler (release_traces() first)")
        p = self._stage_path(key, batch.page_table_width)
        head, outs, logits = self.model.head, [], None
        n_ids = api.WIDE_ROWS if wide else api.NUM_LANES  # a / m: 32 lane-ordered ids (T32), 64 split-order (T64)
        prof = self.spec_profile
        t_prev = time.perf_counter()
        prof["plan"] += (t_prev - t0) * 1e3
        if smp is not None:
            smp.set_params(*sampling)  # host compare; a device write only when the lanes' parameters changed
            smp.set_positions(batch.positions)  # the step's RNG counters (anchor positions n)
        # Fast path (traced, "mesh" vocab split, no observer): every pass is enqueued back to back -- its inputs, the
        # replay, then non-blocking reads of a / m into pass-indexed host staging -- and only the step's LAST read
        # blocks (the logits on an ordinary step, else the last m; on a device-sampled step the sampler's info read
        # of every chip). A blocking read ends in a full-mesh finish (an event round trip to all 32 chips), so one per
        # step instead of one per output; pass 2's inputs do not depend on pass 1's outputs, and in command-queue
        # order pass 1's reads complete before pass 2 overwrites the trace outputs.
        fast = use_trace and head.vocab_split == "mesh" and self.spec_observer is None
        staged = []
        n_pass = len(passes)
        res_s = None
        keep_dev = None  # an eager device-sampled step keeps its outputs until the host fallback ran
        rm_s = None
        try:
            for i, ps in enumerate(passes):
                if wide:
                    self._write_wide(p, ps)
                else:
                    self._write_spec(p, ps)
                dev = so = None
                t1 = time.perf_counter()
                t_enq = None
                if use_trace:
                    self._replay(p)
                    t_enq = time.perf_counter()
                    rm, a_t, m_t = p.out
                    so_read = p.so
                else:
                    rm, a_t, m_t = dev = self._device_step(p, pool)
                    so = so_read = self._take_so()
                read_logits = rm is not None and (
                    (bool(want_logits) and i == 0) or (self.spec_observer is not None and self.observe_logits)
                )
                t2 = time.perf_counter()
                lg = None
                sample_here = smp is not None and i == 0
                wkey = (p.key, n_pass, i)
                if fast:
                    ra, rmm = self._out_reader(f"a{i}", a_t), self._out_reader(f"m{i}", m_t)
                    ttnn.copy_device_to_host_tensor(a_t, ra.host, blocking=False)
                    block = i == n_pass - 1 and not read_logits and not sample_here
                    t_read = self._spin(wkey, t_enq) if (block or sample_here or read_logits) else None
                    ttnn.copy_device_to_host_tensor(m_t, rmm.host, blocking=block)
                    if sample_here:  # the step's one blocking read: every chip's info (lands a / m too)
                        res_s = smp.read(so_read, mesh_sync=True)
                    if block or sample_here:
                        self._waited(wkey, t_enq, t_read)
                    t3 = time.perf_counter()
                    if read_logits:
                        lg = head.logits_to_host(rm)  # blocking: every read enqueued before it has landed too
                        if not sample_here:
                            self._waited(wkey, t_enq, t_read)
                    staged.append((ra, rmm))
                    if sample_here:
                        rm_s = rm
                else:
                    try:
                        t_read = self._spin(wkey, t_enq) if t_enq is not None else None
                        a = head.tokens_to_host(a_t)  # blocking: waits for the step
                        if t_enq is not None:
                            self._waited(wkey, t_enq, t_read)
                        t3 = time.perf_counter()
                        m = head.tokens_to_host(m_t)
                        lg = head.logits_to_host(rm) if read_logits else None
                        if sample_here:
                            res_s = smp.read(so_read)
                            rm_s = rm
                    finally:
                        if dev is not None:
                            if sample_here:
                                keep_dev = (dev, so)  # freed after the host fallback (it may read rm)
                            else:
                                _free(*dev, so)
                    outs.append((a, m))
                t4 = time.perf_counter()
                prof["write"] += (t1 - t_prev) * 1e3
                prof["enqueue"] += (t2 - t1) * 1e3
                prof["wait"] += (t3 - t2) * 1e3
                prof["read"] += (t4 - t3) * 1e3
                prof["passes"] += 1
                t_prev = t4
                self.stats["decode_steps"] += 1
                if i == 0 and want_logits:
                    logits = lg
                if self.spec_observer is not None:
                    self.spec_observer(plan, i, {"a": outs[-1][0], "m": outs[-1][1], "logits": lg})
                    t_prev = time.perf_counter()
            for ra, rmm in staged:  # every staging buffer landed with the step's last (blocking) read
                outs.append((_ids_from_staging(ra, n_ids), _ids_from_staging(rmm, n_ids)))
            if wide:
                res = plan.result(outs[0][0], outs[0][1], logits=logits)
            else:
                res = plan.result(outs, logits=logits)
            if smp is not None:
                res_s = self._finish_sampled(key[0], batch.positions, res_s, rm_s)
                res = SampledSpecDecodeResult(
                    logits=res.logits, argmax=res.argmax, mtp_argmax=res.mtp_argmax, sample=res_s
                )
            self._read_done()
        except BaseException:
            self._abort_step()
            raise
        finally:
            if keep_dev is not None:
                _free(*keep_dev[0], keep_dev[1])
        st = self.stats
        st["spec_steps"] += 1
        st["verify_steps"] += int(plan.is_verify)
        st["drafts"] += plan.num_drafts
        if wide:
            st["wide_steps"] += 1
            st["wide_verify_steps"] += int(plan.is_verify)
            st["wide_drafts"] += plan.num_drafts
        else:
            st["packed_drafts"] += len(plan.partner_of)
            st["cross_row_partners"] += plan.cross_row_partners
            st["overflow_drafts"] += len(plan.overflow)
            st["overflow_passes"] += int(len(plan.passes) > 1)
            st["auto_t32_verifies"] += int(plan.is_verify and self.spec_verify == "auto")
        self.last_spec = plan
        self.last_verify_kind = key[0]
        t5 = time.perf_counter()
        prof["result"] += (t5 - t_prev) * 1e3
        prof["steps"] += 1
        prof["wide"] += int(wide)
        prof["total"] += (t5 - t0) * 1e3
        self.timings["last_spec_step_ms"] = (t5 - t0) * 1e3
        return res

    def _out_reader(self, role: str, t) -> HostShardReader:
        """Persistent host staging (``lm_head.HostShardReader``: every chip's copy, read concurrently) of one spec
        output per role (``a0`` / ``m0`` / ``a1`` / ``m1``: a and m share a spec, the passes share the tensors).
        Host memory only: allocating it after a capture is safe."""
        key = (role,) + HostShardReader.spec_key(t)
        r = self._readers.get(key)
        if r is None:
            r = self._readers[key] = HostShardReader(self.mesh_device, t)
        return r

    def reset_spec_profile(self) -> Dict[str, float]:
        """Return the accumulated host profile of ``decode_forward_spec`` (ms per stage, summed over ``steps`` calls
        / ``passes`` device runs) and start a new one. Stages: ``plan`` (host checks + packing), ``write`` (input host
        tensors + copies), ``enqueue`` (trace replay enqueue, or the whole eager step), ``wait`` (until the first
        output read returns: the device step; on the fast path the step's one blocking read), ``read`` (the remaining
        reads: the logits read on an ordinary step), ``result``."""
        out = dict(self.spec_profile)
        self.spec_profile = {k: 0.0 for k in SPEC_PROFILE_KEYS}
        return out

    # ==============================================================================================================
    # warmup (GEN-3 / GEN-4; features design §3.11, D12)
    # ==============================================================================================================
    def _warm_chunk(self, path: str, bucket: int, pool: MotifKVPool) -> None:
        """One warm-up chunk of ``(path, bucket)`` (``attention.warmup_chunk_host_tables``: the fill table is all
        ``-1``, an sp1 chunk sits at ``warmup_start`` with an all-zero SDPA table and null tail blocks), so nothing is
        written and only the null block is read; the LM head and, with the MTP layer, its KV-only fill run as in
        serving. Compiles every program of the shape."""
        self._warm_full(warmup_chunk_host_tables(self.cfg, path, bucket), pool)

    def _warm_full(self, host: Any, pool: MotifKVPool) -> None:
        """One full warm-up run of ``host`` (warm-up chunk or packed-pass tables: nothing written, only the null block
        read): embedding, every layer, the LM head on the first segment's last row, the MTP KV-only fill."""
        model, cfg = self.model, self.cfg
        rows = int(host.bucket)
        pad = torch.full((rows,), int(cfg.pad_token_id), dtype=torch.int32)
        inp = tok = X = tile = hn = nxt = None
        try:
            inp = model.chunk_inputs(host)
            tok = model.embed.prefill_tokens_device(pad, rows)
            X = model.prefill_chunk(tok, chunk=inp, kv_caches=pool)
            row = host.head_row(0) if host.is_packed else int(host.end) - 1 - int(host.start)
            tile = model.head.forward_prefill(X, row)
            model.head.prefill_logits_to_host(tile, row)
            if model.mtp is not None:
                nxt = model.embed.rows_tokens_device(pad, rows)
                hn = model.head.stream_mean_norm(X)
                model.mtp.fill_kv_prefill(hn, nxt, kv_cache=pool.mtp, chunk=inp)
        finally:
            _free(tok, X, tile, hn, nxt)
            if inp is not None:
                inp.free()

    def _warm_packed(self, shape: PrefillShape, pool: MotifKVPool) -> str:
        """Compile the programs of one packed pass shape (design §3.5; ``warmup_packed_host_tables``: nothing written,
        only the null block read). ``packed_warmup="attention"``: the attention of one global and one SWA layer
        (``model.warm_attention``; every other program of a pass of ``T`` rows is the solo bucket-``T`` chunk's), plus
        the MTP layer's packed fill once per ``T`` (gathered RoPE rows at ``T``; the sp1 bucket-``T`` warm-up compiles
        it too, unless ``T`` exceeds ``max_sp1_bucket``). ``"full"``: one full warm-up pass. Returns what ran."""
        model = self.model
        path, T, S = str(shape[0]), int(shape[1]), int(shape[2])
        tails = shape[3] if len(shape) > 3 else None
        host = warmup_packed_host_tables(self.cfg, path, T, S, tails)
        if self.packed_warmup == "full":
            self._warm_full(host, pool)
            self._packed_mtp_warmed.add(T)
            return "full pass"
        mtp = model.mtp is not None and T not in self._packed_mtp_warmed
        inp = model.chunk_inputs(host)
        try:
            layers = model.warm_attention(inp, pool, mtp=mtp)
        finally:
            inp.free()
        if mtp:
            self._packed_mtp_warmed.add(T)
        return f"attention L{'/L'.join(map(str, layers))}{' + MTP fill' if mtp else ''}"

    def warmup_prefill(self, *, kv_cache: Any, enable_trace: bool) -> None:
        """Compile every prefill shape before the decode capture (module docstring, step 3): one warm-up chunk per solo
        ``(path, bucket)`` (:meth:`prefill_shapes`), then, while :attr:`packed_prefill` is on, every packed shape
        (:meth:`packed_shapes`, :meth:`_warm_packed`). Writes nothing. Shapes already warmed are skipped."""
        pool = self._check_pool(kv_cache)
        if enable_trace:  # plugin trace_mode="all": prefill is eager
            return
        if self.trace_captured:
            raise RuntimeError("warmup_prefill after the decode trace capture (prefill shapes must compile before it)")
        t_all = time.time()
        for path, b in self.prefill_shapes():
            if (path, b) in self._warmed:
                continue
            t0 = time.time()
            self._warm_chunk(path, b, pool)
            self._warmed.add((path, b))
            key = f"warmup_prefill_{path}_{b}_s"
            self.timings[key] = time.time() - t0
            self.log(
                f"warmup prefill {path} bucket {b}{' (+ MTP fill)' if self.mtp_enabled else ''}: "
                f"{self.timings[key]:.1f} s"
            )
        if self.packed_prefill:
            t_pk = time.time()
            todo = [s for s in self.packed_shapes() if s not in self._warmed]
            per_t: Dict[int, List[str]] = {}
            for shape in todo:
                t0 = time.time()
                what = self._warm_packed(shape, pool)
                self._warmed.add(shape)
                tails = "/" + shape[3] if len(shape) > 3 else ""
                per_t.setdefault(int(shape[1]), []).append(
                    f"{shape[0]} S{shape[2]}{tails} {time.time() - t0:.2f} s ({what})"
                )
            for T, lines in sorted(per_t.items()):
                self.log(f"warmup packed prefill T={T}: {'; '.join(lines)}")
            self.timings["warmup_packed_s"] = time.time() - t_pk
            if todo:
                self.log(
                    f"warmup packed prefill: {len(todo)} shapes ({self.packed_warmup!r}) in "
                    f"{self.timings['warmup_packed_s']:.1f} s"
                )
        self.timings["warmup_prefill_s"] = time.time() - t_all

    def _inactive_step(self, p: DecodePath, pool: MotifKVPool) -> None:
        """One eager step of ``p`` with every lane inactive (compiles all of the path's programs, writes nothing; the
        host readers of its outputs are set up too)."""
        n = api.NUM_LANES
        tokens, pos = torch.zeros(n, dtype=torch.int32), torch.full((n,), -1, dtype=torch.int32)
        pt = torch.zeros(n, p.width, dtype=torch.int32)
        head = self.model.head
        if p.kind in SPEC_KINDS:
            if p.kind == WIDE:
                self._write_wide(p, self._inactive_wide_plan(p))
                rm, a, m = outs = self._wide_step(p, pool)
            else:
                self._write_spec(p, SpecPass(tokens, KVWriteStep.ordinary(pos, pt)))
                rm, a, m = outs = self._spec_step(p, pool)
            so = self._take_so()
            try:
                head.tokens_to_host(a)
                head.tokens_to_host(m)
                if rm is not None:
                    head.logits_to_host(rm)
                if so is not None:  # the sampler's reads (chip 0 and the staged every-chip read) before the capture
                    self.sampler.read(so, count=False)
                    self.sampler.read(so, mesh_sync=True, count=False)
            finally:
                _free(*outs, so)
        else:
            self._write_plain(p, api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt))
            out = self._plain_step(p, pool)
            so = self._take_so()
            try:
                head.logits_to_host(out)
                if so is not None:
                    self.sampler.read(so, count=False)
            finally:
                _free(out, so)
        self.stats["decode_steps"] += 1

    def _inactive_wide_plan(self, p: DecodePath) -> VP.WideStepPlan:
        """The T64 step with every row inactive (warm-up and capture steps: nothing written), planned with ``p``'s
        geometry."""
        n = api.NUM_LANES
        batch = api.SpecDecodeBatch(
            tokens=torch.zeros(n, dtype=torch.int32), positions=torch.full((n,), -1, dtype=torch.int32),
            draft_tokens=torch.full((n,), -1, dtype=torch.int32), page_table=torch.zeros(n, p.width, dtype=torch.int32),
        )  # fmt: skip
        w = p.kv_write
        return VP.plan_wide_step(batch, cfg=self.cfg, mode=p.mode, lanes_per_call=w.lanes_per_call, gather=w.gather)

    def _write_inactive(self, p: DecodePath) -> None:
        n = api.NUM_LANES
        tokens, pos = torch.zeros(n, dtype=torch.int32), torch.full((n,), -1, dtype=torch.int32)
        pt = torch.zeros(n, p.width, dtype=torch.int32)
        if p.kind == WIDE:
            self._write_wide(p, self._inactive_wide_plan(p))
        elif p.kind == SPEC:
            self._write_spec(p, SpecPass(tokens, KVWriteStep.ordinary(pos, pt)))
        else:
            self._write_plain(p, api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt))

    def warmup_decode(self, *, kv_cache: Any, enable_trace: bool, page_table_width: int) -> None:
        """``enable_trace=False``: stage every decode path (:meth:`decode_paths`: the serving paths, plus
        :attr:`extra_decode_paths`) for width ``W`` and run one eager all-inactive step of each (compiles every decode
        program and allocates every persistent input: F3N rules R2 / R3). ``enable_trace=True``: capture each path's
        trace once (after the prefill warmup and the eager decode warmup of EVERY path, which it runs first when
        needed), in :meth:`decode_paths` order; then one replay of each with every lane inactive. On a speculating
        launch the serving path is the T32-spec step (``("spec", "all_split")`` with prefix caching), the one trace
        that serves ordinary, verify and overflow steps; ``spec_verify="auto"`` adds the T64 trace (``("wide",
        "all_split")``, captured second); ``"wide"`` captures the T64 trace alone. A T64 path is staged and captured
        only under ``ring_gather="safe"`` (F3N R1, checked here again: tests may switch ``MotifCCL.ring_gather`` at run
        time)."""
        pool = self._check_pool(kv_cache)
        W = int(page_table_width)
        if W < 1:
            raise ValueError(f"page_table_width must be >= 1, got {W}")
        keys = self.decode_paths()
        for key in keys:
            if key[0] == WIDE:  # F3N R1 / R-E5 / R-E7 before any T64 staging or capture
                self._check_wide_launch(key[1])
        if not enable_trace:
            t0 = time.time()
            for key in keys:
                p = self._stage_path(key, W)  # refused for a new path / width while a trace exists
                t1 = time.time()
                self._inactive_step(p, pool)
                p.warmed = True
                self.timings[f"warmup_decode_{key[0]}_{key[1]}_s"] = time.time() - t1
            self.timings["warmup_decode_eager_s"] = time.time() - t0
            self.log(f"warmup decode (eager, W={W}): paths {keys} in {self.timings['warmup_decode_eager_s']:.1f} s")
            return
        missing = [s for s in self.required_prefill_shapes() if s not in self._warmed]
        if missing:
            raise RuntimeError(
                f"decode trace capture before the prefill warmup: {len(missing)} shapes not compiled: {missing[:6]}"
                f"{' ...' if len(missing) > 6 else ''}"
            )
        for key in keys:
            p = self._paths.get(key)
            if p is not None and p.traced and p.width != W:
                raise RuntimeError(f"a decode trace for width {p.width} exists; release_traces() first")
        # F3N R2: every path compiled (one eager step each) before the FIRST capture
        if any(key not in self._paths or not self._paths[key].warmed or self._paths[key].width != W for key in keys):
            self.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=W)
        t_all = time.time()
        for key in keys:
            p = self._paths[key]
            if p.traced:
                continue
            t0 = time.time()
            self._write_inactive(p)  # the capture's own replay below must write nothing
            ttnn.synchronize_device(self.mesh_device)
            p.trace_id, p.out = self._capture_path(p, pool)
            p.pool = pool
            self._acknowledge_trace_outputs(p)
            ttnn.execute_trace(self.mesh_device, p.trace_id, cq_id=0, blocking=True)  # one replay: all lanes inactive
            dt = time.time() - t0
            self.timings[f"capture_decode_{key[0]}_{key[1]}_s"] = dt
            what = {PLAIN: "plain", SPEC: "T32-spec", WIDE: "T64 (16 rows per DP row)"}[key[0]]
            self.log(
                f"decode trace captured: {what} path (W={W}, {self.num_layers} layers"
                f"{' + the MTP layer' if key[0] in SPEC_KINDS else ''}, KV write {key[1]!r}"
                f"{', device sampler' if p.so is not None else ''}) in {dt:.1f} s"
            )
        self.timings["capture_decode_s"] = time.time() - t_all
        traced = [k for k, q in self._paths.items() if q.traced]
        if len(traced) > 1:
            ring = self._ring_gather()
            who = (
                "the serving paths" if set(traced) <= set(self.serving_paths)
                else f"a test configuration: serving captures {self.serving_paths}"
            )  # fmt: skip
            msg = (
                f"{len(traced)} decode traces captured ({traced}; {who}). F3N rules (docs/p5_t64/f3.md §6): R1 "
                f"ring_gather={ring!r} (every race-prone TP-ring all-gather rerouted: {ring == 'safe'}), R2 every "
                f"decode path compiled before the first capture, R3 persistent inputs allocated before it, R4 every "
                f"step reads its outputs before another trace replays, R5 no re-capture while serving"
            )
            if ring != "safe":
                msg = f"warning: {msg}. With native TP-ring gathers a prefill can read stale tiles (F3, f3.md §2)"
            self.log(msg)

    def _acknowledge_trace_outputs(self, p: DecodePath) -> None:
        """F3N rule R6 (trace-allocation tracker runs only, ``TT_METAL_TRACE_ALLOC_TRACKING=1``; a no-op otherwise):
        the outputs of a trace captured while another trace exists live on memory that the other trace's replays may
        overwrite, which ``ttnn.execute_trace`` would report on every replay. They are read before the other trace
        replays (R4), so they are acknowledged as corruptible right after the capture."""
        try:
            from ttnn.tools import trace_allocation_tracker as tat
        except Exception:
            return
        if not getattr(tat, "TRACE_ALLOC_TRACKING", False):
            return
        if not any(q.traced for q in self._paths.values() if q is not p):
            return
        outs = list(p.out if isinstance(p.out, tuple) else (p.out,))
        if p.so is not None:
            outs += list(p.so.tensors())
        for t in outs:
            if t is not None:
                tat.acknowledge_corruptible(t)

    def _capture_path(self, p: DecodePath, pool: MotifKVPool):
        """Exception-safe capture of one decode step of ``p`` (a dangling capture hung close_mesh_device once,
        GATES_RESULTS §11.6): on any error the capture is ended and released before the error propagates."""
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        try:
            out = self._device_step(p, pool)
        except BaseException:
            _free(self._take_so())
            try:
                ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            except Exception:
                pass
            try:
                ttnn.release_trace(self.mesh_device, tid)
            except Exception:
                pass
            raise
        so = self._take_so()  # the device sampler's outputs: extra trace outputs of the path
        try:
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        except BaseException:
            try:
                ttnn.release_trace(self.mesh_device, tid)
            except Exception:
                pass
            _free(*(out if isinstance(out, tuple) else (out,)), so)
            raise
        p.so = so
        return tid, out

    # ==============================================================================================================
    # legacy single-path internals (draft-1 callers: tests/test_model_truncated.py, analysis/full_model/*): the
    # serving path's inputs and trace under their draft-1 names. Not an API: new code uses the decode paths.
    # ==============================================================================================================
    @property
    def _serving(self) -> Optional[DecodePath]:
        return self._paths.get(self.serving_path)

    @property
    def _trace_id(self):
        p = self._serving
        return None if p is None else p.trace_id

    @property
    def _trace_out(self):
        p = self._serving
        return None if p is None else p.out

    @property
    def _trace_pool(self):
        p = self._serving
        return None if p is None else p.pool

    @property
    def _inputs(self) -> Optional[Dict[str, Any]]:
        p = self._serving
        return None if p is None else p.inputs

    @property
    def _kv_write(self) -> Optional[DecodeKVWrite]:
        p = self._serving
        return None if p is None else p.kv_write

    @property
    def _width(self) -> Optional[int]:
        p = self._serving
        return None if p is None else p.width

    def _host_inputs(self, batch: api.DecodeBatch) -> Dict[str, Any]:
        """Draft-1 form: the serving path's host input tensors for ``batch`` (``tokens`` / ``rot`` / ``cur`` /
        ``pt`` on the plain ``row`` path)."""
        kind, mode = self.serving_path
        return self._path_host_inputs(kind, mode, batch.tokens, batch.positions, batch.page_table)

    def _write_inputs(self, batch: api.DecodeBatch) -> None:
        """Draft-1 form: write ``batch`` (an ordinary step) into the serving path's persistent inputs."""
        p = self._stage_path(self.serving_path, batch.page_table_width)
        if p.kind == SPEC:
            self._write_spec(p, SpecPass(batch.tokens, KVWriteStep.ordinary(batch.positions, batch.page_table)))
        else:
            self._write_plain(p, batch)

    def _decode_step(self, pool: MotifKVPool):
        """Draft-1 form: one device step of the serving path on ``pool`` (eager or inside a capture)."""
        return self._device_step(self._serving, pool)

    def _capture(self, pool: MotifKVPool):
        """Draft-1 form: capture one step of the serving path on ``pool``; returns ``(trace id, outputs)`` (the caller
        owns them)."""
        return self._capture_path(self._serving, pool)

    # ==============================================================================================================
    # lifecycle (GEN-6)
    # ==============================================================================================================
    def release_traces(self) -> None:
        """Release every captured decode trace and its outputs (the persistent inputs stay: a re-capture reuses
        them)."""
        waiter = getattr(self, "waiter", None)
        if waiter is not None and waiter.stats["spins"]:
            w = waiter.stats
            self.log(
                f"host wait (spin): {int(w['spins'])} replays, {w['spin_ms'] / max(1, w['spins']):.2f} ms polled and "
                f"{w['blocked_ms'] / max(1, w['spins']):.2f} ms blocked per replay, {int(w['overshoots'])} overshoots"
            )
        err = None
        for p in self._paths.values():
            if p.trace_id is None:
                continue
            try:
                ttnn.release_trace(self.mesh_device, p.trace_id)
            except Exception as e:  # keep releasing the others
                err = err or e
            finally:
                p.trace_id = None
                _free(*(p.out if isinstance(p.out, tuple) else (p.out,)))
                _free(p.so)  # the sampler's trace outputs (tokens, info)
                p.out = None
                p.so = None
                p.pool = None
        if err is not None:
            raise err

    def close(self) -> None:
        """Release the traces, the persistent inputs, the device sampler, the KV pool and the model weights
        (standalone runs)."""
        self.release_traces()
        for p in list(self._paths.values()):
            self._free_path(p)
        if self.sampler is not None:
            self.sampler.deallocate()
            self.sampler = None
        if self._pool is not None:
            self._pool.deallocate()
            self._pool = None
        self.model.deallocate()


__all__ = [
    "DECODE_KINDS",
    "DecodePath",
    "MotifGenerator",
    "PLAIN",
    "LaneSampling",
    "PrefillBatchPlan",
    "PrefillRowJob",
    "ReplayWaiter",
    "SPEC",
    "SPEC_KINDS",
    "SampledSpecDecodeResult",
    "SpecPass",
    "SpecStepPlan",
    "check_decode_page_tables",
    "check_prefill_page_table",
    "plan_spec_step",
    "prefill_page_table_host",
    "WIDE",
]
