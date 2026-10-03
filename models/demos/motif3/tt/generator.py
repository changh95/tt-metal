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
   ``spec_tokens = 1`` also the MTP layer, part ``L53``). No KV pool yet.
2. ``allocate_kv_cache(num_blocks=, block_size=, num_layers=)`` (GEN-2): ``cfg.set_kv_geometry`` with the plugin's
   values (4129 x 64 for the serving defaults), one ``ttnn.empty`` + ``ttnn.fill(0)`` cache per layer, plus the MTP
   layer's cache (``pool.mtp``) with speculation.
3. ``warmup_prefill`` (every ``(path, bucket)`` of :meth:`MotifGenerator.prefill_shapes`, eager, writes nothing) ->
   ``warmup_decode(enable_trace=False)`` (stages the persistent inputs of every decode path for width W, one eager
   all-inactive step each) -> ``warmup_decode(enable_trace=True)`` (captures each path's trace, exception-safe).
   Capture refuses to run before every prefill shape was compiled, and after the capture a prefill chunk of a shape the
   warmup did not compile is refused (a program compiled after capture can corrupt the trace: plugin
   ``model_runner.py:3735-3745``; features design D12 / G7); a decode path is never staged after a capture (a buffer
   allocated after a capture and kept across replays may be overwritten by them).
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
replay).
Verify step 89.4 ms vs 88.4 ms for the plain KV-R step; ordinary spec step (with the logits read) 90.1 ms. Without
``OMP_WAIT_POLICY=PASSIVE`` torch's spinning OpenMP workers stall the input copies after the step's host planning
(+4.7 ms per verify step measured): ``create`` logs a warning.

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
from .attention import ChunkHostTables, chunk_host_tables, max_sp1_bucket, warmup_chunk_host_tables
from .embedding import check_token_ids
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
from .model import LazySource, MotifKVPool, MotifModel
from .model_config import MotifTTConfig, require_l1_small
from .mtp import mtp_next_tokens
from .rope import positions_to_rot_idxs, shard_lanes

PrefillShape = Tuple[str, int]  # (path "sp0" | "sp1", bucket)


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
        if t is not None:
            try:
                if t.is_allocated():
                    ttnn.deallocate(t)
            except Exception:
                pass


# ======================================================================================================================
# decode paths and the speculative step plan (features design §3.8, D10; host only)
# ======================================================================================================================
PLAIN, SPEC = "plain", "spec"
DECODE_KINDS = (PLAIN, SPEC)
DecodeKey = Tuple[str, str]  # (kind "plain" | "spec", KV-write mode)
SPEC_PROFILE_KEYS = ("steps", "passes", "plan", "write", "enqueue", "wait", "read", "result", "total")


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


def _ids_from_staging(reader) -> torch.Tensor:
    """Lane-ordered ``int64 [32]`` ids from a ``HostShardReader`` of a ``[1, 1, 1, 32]`` uint32 output ("mesh" vocab
    split: identical on every chip, chip 0's copy); a fresh tensor (the staging is overwritten by the next read)."""
    return reader.views[0].reshape(-1)[: api.NUM_LANES].to(torch.int64).clone()


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

    ``kind`` ``"plain"`` (embedding -> layers -> LM head logits; :meth:`MotifGenerator.decode_forward`) or ``"spec"``
    (the T32-spec step ``MotifModel.decode_spec``: + main argmax + the MTP layer + MTP argmax;
    :meth:`MotifGenerator.decode_forward_spec`). ``mode``: the KV-write mode (``row`` on the plain path = the
    draft-1 ops, no ``DecodeKVWrite``). ``inputs``: ``tokens`` / ``rot`` (and ``cur`` / ``pt`` for the draft-1
    ``row`` path); ``kv_write``: the ``DecodeKVWrite`` (FlashMLA's ``cur_pos`` / ``page_table`` and the update-call
    inputs) of every other path. ``out``: the captured outputs (plain: the ROW_MAJOR logits; spec: ``(rm, a, m)``)."""

    kind: str
    mode: str
    width: int
    inputs: Dict[str, Any] = field(default_factory=dict)
    kv_write: Optional[DecodeKVWrite] = None
    trace_id: Any = None
    out: Any = None
    pool: Any = None
    warmed: bool = False

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
    """The host plan of one ``prefill_forward_batch`` call (:meth:`MotifGenerator.plan_prefill_batch`)."""

    jobs: List[PrefillRowJob]
    order: List[int]  # writer-first execution order (indices into ``jobs`` = input order)
    shapes: Set[PrefillShape] = field(default_factory=set)

    @property
    def chunks(self) -> int:
        return sum(len(j.plan.chunks) for j in self.jobs)


class MotifGenerator(api.MotifGenerator):
    """The Motif-3 runtime (see the module docstring). Build with :meth:`create` (vLLM) or directly from an existing
    :class:`MotifModel` (tests / demo): ``MotifGenerator(mesh_device, cfg, model)``.

    Test hooks: ``chunk_observer(job, chunk, streams)`` (default None), called after every prefill chunk's layers with
    the chunk's residual streams ``[1, 4, C, 4096]`` (eager; it must not free or keep them). Teacher-forced checks read
    the LM head on every tile of a chunk through it (``head.forward_prefill(streams, row)``: no new program).
    ``spec_observer(plan, pass_index, out)`` (default None), called after every pass of a speculative step with the
    host ``out = {"a": [32], "m": [32], "logits": [32, V] or None}`` in PHYSICAL-lane order (``observe_logits=True``
    reads the logits of every pass, ~2 ms each). ``extra_decode_paths``: ``(kind, mode)`` decode paths that
    :meth:`warmup_decode` prepares and captures next to the serving path (set before the warmup; e.g. the plain
    ``("plain", "all")`` trace next to the spec trace, for comparisons in one session). Tests only: serving captures
    the serving path alone. With two captured traces, prefills after a ``release_traces`` / new-program compile /
    re-capture cycle returned garbage tiles (2026-10-03, ``tests/test_resumed_prefill.py`` ``MULTI_TRACE_NOTE``);
    :meth:`warmup_decode` logs a warning when it captures more than one trace.

    Args beyond the model: ``decode_kv_mode`` / ``spec_kv_mode`` override the KV-write mode of the plain / spec path
    (default ``generator_api.kv_write_mode(cfg.kv_replicated_decode, False / True)``: ``row`` / ``row_split``, or
    ``all`` / ``all_split`` with KV-R)."""

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
        if settings is not None and settings.spec_verify != "packed":
            raise ValueError(f"spec_verify={settings.spec_verify!r} (the 64-row verify trace, S3) is not implemented")
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.model = model
        self.settings = settings
        self.log = log or (lambda m: None)
        self._pool: Optional[MotifKVPool] = None
        self._paths: Dict[DecodeKey, DecodePath] = {}  # staged decode paths: persistent inputs (+ trace once captured)
        self._warmed: Set[PrefillShape] = set()  # (path, bucket) compiled by warmup_prefill
        kvr = bool(cfg.kv_replicated_decode)
        # plain decode steps: "row" (draft 1) or "all" (KV-R); speculative steps: "row_split" or "all_split" (KV-R)
        self.decode_kv_mode = _check_mode(decode_kv_mode or api.kv_write_mode(kvr, False))
        self.spec_kv_mode = _check_mode(spec_kv_mode or api.kv_write_mode(kvr, True))
        if kvr and not (is_replicated(self.decode_kv_mode) and is_replicated(self.spec_kv_mode)):
            self.log(
                f"warning: KV-R is on (prefix caching) but a decode path writes without it (plain "
                f"{self.decode_kv_mode!r}, spec {self.spec_kv_mode!r}): only for measurements, never for serving"
            )
        # A speculating launch (spec_tokens = 1 and the MTP layer) runs EVERY decode step through the spec trace.
        self.serving_path: DecodeKey = (
            (SPEC, self.spec_kv_mode)
            if (model.mtp is not None and int(cfg.spec_tokens) > 0)
            else (PLAIN, self.decode_kv_mode)
        )
        self.extra_decode_paths: List[DecodeKey] = []
        self.chunk_observer: Optional[Callable[[PrefillRowJob, PP.ChunkPlan, Any], None]] = None
        self.spec_observer: Optional[Callable[[SpecStepPlan, int, Dict[str, Any]], None]] = None
        self.observe_logits = False
        self.last_prefill: Optional[PrefillBatchPlan] = None
        self.last_spec: Optional[SpecStepPlan] = None
        self.stats: Dict[str, int] = {
            "prefill_calls": 0,
            "prefill_rows": 0,
            "prefill_chunks": 0,
            "sp1_chunks": 0,
            "recomputed_rows": 0,
            "mtp_fills": 0,
            "decode_steps": 0,  # device runs of a decode step (trace replays + eager steps), every path and pass
            "spec_steps": 0,  # decode_forward_spec calls
            "verify_steps": 0,  # ... with at least one draft
            "drafts": 0,
            "packed_drafts": 0,  # drafts evaluated on an idle partner lane (pass 1, call B)
            "cross_row_partners": 0,  # ... on another DP row than their owner (KV-R)
            "overflow_drafts": 0,  # drafts evaluated in pass 2 on their own lane
            "overflow_passes": 0,
        }
        self.timings: Dict[str, float] = {}
        self.spec_profile: Dict[str, float] = {k: 0.0 for k in SPEC_PROFILE_KEYS}
        self._readers: Dict[Tuple, HostShardReader] = {}  # host staging of the spec outputs' id reads (by role)

    # ==============================================================================================================
    # construction (GEN-1)
    # ==============================================================================================================
    @classmethod
    def create(
        cls, *, hf_config: Any, mesh_device: Any, settings: api.GeneratorSettings, **model_kwargs
    ) -> "MotifGenerator":
        """Build the runtime on the plugin's open mesh (``generator_api.MotifGenerator.create``). ``model_kwargs`` go
        to :class:`MotifModel` (``cache``, ``vocab_split``, ``layer_kwargs``, ``source``, ``mtp``)."""
        log = model_kwargs.pop("log", _log_default)
        if settings.spec_verify != "packed":
            raise ValueError(f"spec_verify={settings.spec_verify!r} (the 64-row verify trace, S3) is not implemented")
        l1s = require_l1_small(mesh_device)
        cfg = MotifTTConfig.from_settings(settings, mesh_device=mesh_device, hf_config=hf_config)
        log(
            f"create: {cfg.describe()} (mesh L1_SMALL {l1s} B per core; weights {settings.weights_path} "
            f"[{settings.weights_source}])"
        )
        if settings.packed_prefill:
            log(
                "create: MOTIF3_PACKED_PREFILL is set, but packed multi-row prefill (design §3.12.1, gate G15) is not "
                "implemented: rows run one after another"
            )
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
        """Every decode step runs through the spec trace (``spec_tokens = 1`` with the MTP layer)."""
        return self.serving_path[0] == SPEC

    @property
    def warmed_shapes(self) -> Set[PrefillShape]:
        """The ``(path, bucket)`` prefill shapes ``warmup_prefill`` compiled."""
        return set(self._warmed)

    def prefill_shapes(self) -> List[PrefillShape]:
        """Every ``(path, bucket)`` a prefill chunk can have (what ``warmup_prefill`` compiles): sp0 for every bucket of
        ``cfg.prefill_span_buckets``, sp1 for those ``<= max_sp1_bucket``; ascending buckets, sp0 first."""
        top = self.max_sp1_bucket
        out: List[PrefillShape] = []
        for b in self.cfg.prefill_span_buckets:
            out.append((PP.SP0, int(b)))
            if int(b) <= top:
                out.append((PP.SP1, int(b)))
        return out

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
        raises ``ValueError`` (bad rows) or ``RuntimeError`` (a shape the warmup did not compile, after the capture)."""
        reqs = api.check_prefill_batch(requests)
        cfg, bs = self.cfg, int(self.cfg.kv_block_size)
        num_blocks = None if self._pool is None else int(self._pool.num_blocks)
        jobs: List[PrefillRowJob] = []
        shapes: Set[PrefillShape] = set()
        for i, r in enumerate(reqs):
            e = r.end
            if e > self.max_prefill_len:
                raise ValueError(f"row {i}: prompt of {e} tokens exceeds max_model_len {self.max_prefill_len}")
            check_prefill_page_table(r.page_table, e, block_size=bs, num_blocks=num_blocks, row=i)
            check_token_ids(r.tokens, cfg)
            plan = self.plan_row(int(r.start), e)
            tables = [chunk_host_tables(cfg, plan, c, r.page_table) for c in plan.chunks]
            for c in plan.chunks:
                shapes.add((c.path, int(c.bucket)))
            jobs.append(PrefillRowJob(index=i, request=r, plan=plan, tables=tables))
        if self.trace_captured:
            missing = sorted(s for s in shapes if s not in self._warmed)
            if missing:
                # compiling a new prefill program after the decode capture can corrupt the trace (plugin contract)
                raise RuntimeError(f"prefill shapes {missing} were not compiled before the decode trace capture")
        order = PP.order_prefill_requests(reqs, [j.plan for j in jobs], bs)
        return PrefillBatchPlan(jobs=jobs, order=order, shapes=shapes)

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
        input order. ``enable_trace`` (plugin ``trace_mode="all"``) is ignored: prefill is eager."""
        pool = self._check_pool(kv_cache)
        batch = self.plan_prefill_batch(requests)
        self.last_prefill = batch
        out: List[Optional[torch.Tensor]] = [None] * len(batch.jobs)
        t0 = time.time()
        for i in batch.order:
            out[i] = self._run_row(batch.jobs[i], pool)
        self.stats["prefill_calls"] += 1
        self.stats["prefill_rows"] += len(batch.jobs)
        self.timings["last_prefill_s"] = time.time() - t0
        return torch.stack(out)

    def _run_row(self, job: PrefillRowJob, pool: MotifKVPool) -> torch.Tensor:
        """Every chunk of one row, in order; returns the host logits of the row's last position."""
        logits = None
        for ch, host in zip(job.plan.chunks, job.tables):
            lg = self._run_chunk(job, ch, host, pool)
            if ch.last:
                logits = lg
        self.stats["prefill_chunks"] += len(job.plan.chunks)
        self.stats["sp1_chunks"] += sum(c.is_sp1 for c in job.plan.chunks)
        self.stats["recomputed_rows"] += job.plan.recompute
        if logits is None:
            raise AssertionError("a row plan without a last chunk")
        return logits

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

    # ==============================================================================================================
    # decode paths (GEN-3; features design §3.8, §3.11)
    # ==============================================================================================================
    def decode_paths(self) -> List[DecodeKey]:
        """The ``(kind, mode)`` decode paths :meth:`warmup_decode` prepares and captures: the serving path first, then
        :attr:`extra_decode_paths` (deduplicated)."""
        out = [self.serving_path]
        for kind, mode in self.extra_decode_paths:
            key = (str(kind), _check_mode(str(mode)))
            if key[0] not in DECODE_KINDS:
                raise ValueError(f"decode path kind must be one of {DECODE_KINDS}, got {key[0]!r}")
            if key[0] == SPEC and self.model.mtp is None:
                raise ValueError("a spec decode path needs the MTP layer")
            if key not in out:
                out.append(key)
        return out

    def _resolve_key(self, kind: Optional[str], path: Optional[DecodeKey]) -> DecodeKey:
        if path is not None:
            key = (str(path[0]), _check_mode(str(path[1])))
            if key[0] not in DECODE_KINDS or (kind is not None and key[0] != kind):
                raise ValueError(f"decode path {path!r} is not a {kind or 'plain / spec'} path")
            return key
        if kind is None:
            return self.serving_path
        return (kind, self.decode_kv_mode if kind == PLAIN else self.spec_kv_mode)

    def _path_host_inputs(self, kind: str, mode: str, tokens: torch.Tensor, positions: torch.Tensor, page_table=None):
        """Host mesh tensors of a path's own persistent inputs: ``tokens`` (lane order, 0 on inactive lanes), the RoPE
        rows ``rot`` of ``positions``, and for the draft-1 plain ``row`` path its ``cur`` / ``pt`` (every other path
        keeps FlashMLA's inputs in its ``DecodeKVWrite``)."""
        cfg, mesh = self.cfg, self.mesh_device
        pos = positions.to(torch.int32)
        active = pos >= 0
        tok = torch.where(active, tokens.to(torch.int32), torch.zeros_like(pos))
        out = {
            "tokens": self.model.embed.decode_tokens_host(tok),
            "rot": shard_lanes(positions_to_rot_idxs(pos, cfg), cfg, mesh, dtype=ttnn.uint32, device=None),
        }
        if kind == PLAIN and mode == "row":
            pt = torch.where(active[:, None], page_table.to(torch.int32), torch.zeros_like(page_table))
            out["cur"] = shard_lanes(pos.contiguous(), cfg, mesh, dtype=ttnn.int32, device=None)  # [8] per DP row
            out["pt"] = shard_lanes(pt.contiguous(), cfg, mesh, dtype=ttnn.int32, device=None)
        return out

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
        if p is not None:
            self._free_path(p)
        if kind == SPEC and self.model.mtp is None:
            raise ValueError("the spec decode path needs the MTP layer")
        n = api.NUM_LANES
        h = self._path_host_inputs(
            kind, mode, torch.zeros(n, dtype=torch.int32), torch.full((n,), -1, dtype=torch.int32),
            torch.zeros(n, int(width), dtype=torch.int32),
        )  # fmt: skip
        p = DecodePath(kind=kind, mode=mode, width=int(width))
        p.inputs = {k: ttnn.to_device(v, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for k, v in h.items()}
        if not (kind == PLAIN and mode == "row"):  # the draft-1 path keeps the attention's own 8-lane update
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
        h = self._path_host_inputs(p.kind, p.mode, batch.tokens, batch.positions, batch.page_table)
        for k, v in h.items():
            ttnn.copy_host_to_device_tensor(v, p.inputs[k])
        if p.kv_write is not None:
            p.kv_write.write_step(KVWriteStep.ordinary(batch.positions, batch.page_table))

    def _write_spec(self, p: DecodePath, ps: SpecPass) -> None:
        """One spec pass's inputs (the plan's checks ran already: ``write_step`` skips its own validation)."""
        h = self._path_host_inputs(p.kind, p.mode, ps.tokens, ps.step.positions)
        for k, v in h.items():
            ttnn.copy_host_to_device_tensor(v, p.inputs[k])
        p.kv_write.write_step(ps.step, validate=False)

    def _plain_step(self, p: DecodePath, pool: MotifKVPool):
        """One plain decode step on the device (eager or inside a capture): ROW_MAJOR logits ``[1, 1, 32, 6880]``."""
        d, w = p.inputs, p.kv_write
        if w is None:
            return self.model.decode(
                d["tokens"], rot_idxs=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_caches=pool
            )
        return self.model.decode(
            d["tokens"], rot_idxs=d["rot"], cur_pos=w.cur_pos, page_table=w.page_table, kv_caches=pool, kv_write=w
        )

    def _spec_step(self, p: DecodePath, pool: MotifKVPool):
        """One T32-spec step on the device (eager or inside a capture): ``(rm, a, m)`` (``MotifModel.decode_spec``)."""
        return self.model.decode_spec(p.inputs["tokens"], rot_idxs=p.inputs["rot"], kv_write=p.kv_write, kv_caches=pool)

    def _device_step(self, p: DecodePath, pool: MotifKVPool):
        return self._spec_step(p, pool) if p.kind == SPEC else self._plain_step(p, pool)

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
        220160]`` (a fresh tensor). On a speculating launch the serving path is the spec trace: the step is an
        ordinary spec step (``decode_forward_spec`` without drafts, the same logits; the MTP layer also writes its
        cache, design G8). ``path`` (tests): run a specific ``(kind, mode)`` path instead of the serving one."""
        pool = self._check_pool(kv_cache)
        key = self._resolve_key(None, path)
        if key[0] == SPEC:
            res = self.decode_forward_spec(
                api.SpecDecodeBatch.from_decode_batch(batch), kv_cache=pool, enable_trace=enable_trace,
                want_logits=True, path=key,
            )  # fmt: skip
            return res.logits
        if int(batch.positions.max()) >= self.cfg.max_model_len:
            raise ValueError(f"decode position {int(batch.positions.max())} >= max_model_len {self.cfg.max_model_len}")
        # every host check before the path is staged or written (the spec path's plan_spec_step does the same): an
        # active lane's token outside [0, vocab) would be embedded out of range (a negative one silently as padding)
        check_token_ids(batch.tokens[batch.positions >= 0], self.cfg)
        check_decode_page_tables(
            batch.positions, batch.page_table, block_size=self.cfg.kv_block_size, num_blocks=pool.num_blocks
        )
        use_trace = self._check_trace_use(self._paths.get(key), pool, batch.page_table_width, enable_trace)
        p = self._stage_path(key, batch.page_table_width)
        self._write_plain(p, batch)
        self.stats["decode_steps"] += 1
        head = self.model.head
        if use_trace:
            ttnn.execute_trace(self.mesh_device, p.trace_id, cq_id=0, blocking=False)
            return head.logits_to_host(p.out)  # blocking read of the trace output (fresh host tensor)
        out = self._plain_step(p, pool)
        try:
            return head.logits_to_host(out)
        finally:
            _free(out)

    def plan_spec_step(self, batch: api.SpecDecodeBatch, *, path: Optional[DecodeKey] = None) -> SpecStepPlan:
        """:func:`plan_spec_step` with this generator's config, pool and (once traced) trace width: every host check
        and the packing of one speculative step, before any device op."""
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

    def decode_forward_spec(
        self,
        batch: api.SpecDecodeBatch,
        *,
        kv_cache: Any,
        enable_trace: bool,
        want_logits: bool,
        path: Optional[DecodeKey] = None,
    ) -> api.SpecDecodeResult:
        """One decode step of a speculating launch (``generator_api.MotifGenerator.decode_forward_spec``; features
        design §3.8; module docstring "Speculative decode"): the plan (host checks, packing), then one replay of the
        T32-spec trace (pass 1: anchors on their lanes at ``n``, packed drafts on idle partner lanes at ``n + 1``) and,
        only when some draft found no idle lane, a second replay (pass 2: those drafts on their own lanes at ``n +
        1``). Returns ``SpecDecodeResult`` in owner-lane order: ``argmax = (a0, a1)``, ``mtp_argmax = (m0, m1)``,
        logits of the anchor rows when ``want_logits`` (pass 1's owner lanes). ``path`` (tests): a non-default spec
        path ``("spec", mode)``."""
        pool = self._check_pool(kv_cache)
        if self.model.mtp is None:
            raise NotImplementedError("decode_forward_spec needs the MTP layer (the generator was built without it)")
        t0 = time.perf_counter()
        key = self._resolve_key(SPEC, path)
        plan = self.plan_spec_step(batch, path=key)
        use_trace = self._check_trace_use(self._paths.get(key), pool, batch.page_table_width, enable_trace)
        p = self._stage_path(key, batch.page_table_width)
        head, outs, logits = self.model.head, [], None
        prof = self.spec_profile
        t_prev = time.perf_counter()
        prof["plan"] += (t_prev - t0) * 1e3
        # Fast path (traced, "mesh" vocab split, no observer): every pass is enqueued back to back -- its inputs, the
        # replay, then non-blocking reads of a / m into pass-indexed host staging -- and only the step's LAST read
        # blocks (the logits on an ordinary step, else the last m). A blocking read ends in a full-mesh finish (an
        # event round trip to all 32 chips), so one per step instead of one per output; pass 2's inputs do not depend
        # on pass 1's outputs, and in command-queue order pass 1's reads complete before pass 2 overwrites the
        # trace outputs.
        fast = use_trace and head.vocab_split == "mesh" and self.spec_observer is None
        staged = []
        n_pass = len(plan.passes)
        for i, ps in enumerate(plan.passes):
            self._write_spec(p, ps)
            read_logits = (bool(want_logits) and i == 0) or (self.spec_observer is not None and self.observe_logits)
            dev = None
            t1 = time.perf_counter()
            if use_trace:
                ttnn.execute_trace(self.mesh_device, p.trace_id, cq_id=0, blocking=False)
                rm, a_t, m_t = p.out
            else:
                rm, a_t, m_t = dev = self._spec_step(p, pool)
            t2 = time.perf_counter()
            lg = None
            if fast:
                ra, rmm = self._out_reader(f"a{i}", a_t), self._out_reader(f"m{i}", m_t)
                ttnn.copy_device_to_host_tensor(a_t, ra.host, blocking=False)
                last = i == n_pass - 1
                ttnn.copy_device_to_host_tensor(m_t, rmm.host, blocking=last and not read_logits)
                t3 = time.perf_counter()
                if read_logits:
                    lg = head.logits_to_host(rm)  # blocking: every read enqueued before it has landed too
                staged.append((ra, rmm))
            else:
                try:
                    a = head.tokens_to_host(a_t)  # blocking: waits for the step
                    t3 = time.perf_counter()
                    m = head.tokens_to_host(m_t)
                    lg = head.logits_to_host(rm) if read_logits else None
                finally:
                    if dev is not None:
                        _free(*dev)
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
            outs.append((_ids_from_staging(ra), _ids_from_staging(rmm)))
        res = plan.result(outs, logits=logits)
        st = self.stats
        st["spec_steps"] += 1
        st["verify_steps"] += int(plan.is_verify)
        st["drafts"] += plan.num_drafts
        st["packed_drafts"] += len(plan.partner_of)
        st["cross_row_partners"] += plan.cross_row_partners
        st["overflow_drafts"] += len(plan.overflow)
        st["overflow_passes"] += int(len(plan.passes) > 1)
        self.last_spec = plan
        t5 = time.perf_counter()
        prof["result"] += (t5 - t_prev) * 1e3
        prof["steps"] += 1
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
        model, cfg = self.model, self.cfg
        host = warmup_chunk_host_tables(cfg, path, bucket)
        pad = torch.full((int(bucket),), int(cfg.pad_token_id), dtype=torch.int32)
        inp = tok = X = tile = hn = nxt = None
        try:
            inp = model.chunk_inputs(host)
            tok = model.embed.prefill_tokens_device(pad, bucket)
            X = model.prefill_chunk(tok, chunk=inp, kv_caches=pool)
            row = int(host.end) - 1 - int(host.start)
            tile = model.head.forward_prefill(X, row)
            model.head.prefill_logits_to_host(tile, row)
            if model.mtp is not None:
                nxt = model.embed.rows_tokens_device(pad, bucket)
                hn = model.head.stream_mean_norm(X)
                model.mtp.fill_kv_prefill(hn, nxt, kv_cache=pool.mtp, chunk=inp)
        finally:
            _free(tok, X, tile, hn, nxt)
            if inp is not None:
                inp.free()

    def warmup_prefill(self, *, kv_cache: Any, enable_trace: bool) -> None:
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
        self.timings["warmup_prefill_s"] = time.time() - t_all

    def _inactive_step(self, p: DecodePath, pool: MotifKVPool) -> None:
        """One eager step of ``p`` with every lane inactive (compiles all of the path's programs, writes nothing; the
        host readers of its outputs are set up too)."""
        n = api.NUM_LANES
        tokens, pos = torch.zeros(n, dtype=torch.int32), torch.full((n,), -1, dtype=torch.int32)
        pt = torch.zeros(n, p.width, dtype=torch.int32)
        head = self.model.head
        if p.kind == SPEC:
            self._write_spec(p, SpecPass(tokens, KVWriteStep.ordinary(pos, pt)))
            rm, a, m = outs = self._spec_step(p, pool)
            try:
                head.tokens_to_host(a)
                head.tokens_to_host(m)
                head.logits_to_host(rm)
            finally:
                _free(*outs)
        else:
            self._write_plain(p, api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt))
            out = self._plain_step(p, pool)
            try:
                head.logits_to_host(out)
            finally:
                _free(out)
        self.stats["decode_steps"] += 1

    def _write_inactive(self, p: DecodePath) -> None:
        n = api.NUM_LANES
        tokens, pos = torch.zeros(n, dtype=torch.int32), torch.full((n,), -1, dtype=torch.int32)
        pt = torch.zeros(n, p.width, dtype=torch.int32)
        if p.kind == SPEC:
            self._write_spec(p, SpecPass(tokens, KVWriteStep.ordinary(pos, pt)))
        else:
            self._write_plain(p, api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt))

    def warmup_decode(self, *, kv_cache: Any, enable_trace: bool, page_table_width: int) -> None:
        """``enable_trace=False``: stage every decode path (:meth:`decode_paths`: the serving path, plus
        :attr:`extra_decode_paths`) for width ``W`` and run one eager all-inactive step of each (compiles every decode
        program). ``enable_trace=True``: capture each path's trace (after the prefill warmup and the eager decode
        warmup, which it runs first when needed); then one replay with every lane inactive. On a speculating launch the
        serving path is the T32-spec step (``("spec", "all_split")`` with prefix caching), the one trace that serves
        ordinary, verify and overflow steps."""
        pool = self._check_pool(kv_cache)
        W = int(page_table_width)
        if W < 1:
            raise ValueError(f"page_table_width must be >= 1, got {W}")
        keys = self.decode_paths()
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
        missing = [s for s in self.prefill_shapes() if s not in self._warmed]
        if missing:
            raise RuntimeError(f"decode trace capture before the prefill warmup: shapes {missing} not compiled")
        for key in keys:
            p = self._paths.get(key)
            if p is not None and p.traced and p.width != W:
                raise RuntimeError(f"a decode trace for width {p.width} exists; release_traces() first")
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
            ttnn.execute_trace(self.mesh_device, p.trace_id, cq_id=0, blocking=True)  # one replay: all lanes inactive
            dt = time.time() - t0
            self.timings[f"capture_decode_{key[0]}_{key[1]}_s"] = dt
            self.log(
                f"decode trace captured: {key[0]} path (W={W}, {self.num_layers} layers"
                f"{' + the MTP layer' if key[0] == SPEC else ''}, KV write {key[1]!r}) in {dt:.1f} s"
            )
        self.timings["capture_decode_s"] = time.time() - t_all
        traced = [k for k, q in self._paths.items() if q.traced]
        if len(traced) > 1:
            self.log(
                f"warning: {len(traced)} decode traces captured ({traced}): a test configuration (serving captures "
                "one). With two traces, prefills after a release_traces / new-program compile / re-capture cycle "
                "returned garbage tiles (2026-10-03; tests/test_resumed_prefill.py MULTI_TRACE_NOTE)"
            )

    def _capture_path(self, p: DecodePath, pool: MotifKVPool):
        """Exception-safe capture of one decode step of ``p`` (a dangling capture hung close_mesh_device once,
        GATES_RESULTS §11.6): on any error the capture is ended and released before the error propagates."""
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        try:
            out = self._device_step(p, pool)
        except BaseException:
            try:
                ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            except Exception:
                pass
            try:
                ttnn.release_trace(self.mesh_device, tid)
            except Exception:
                pass
            raise
        try:
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        except BaseException:
            try:
                ttnn.release_trace(self.mesh_device, tid)
            except Exception:
                pass
            _free(*(out if isinstance(out, tuple) else (out,)))
            raise
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
                p.out = None
                p.pool = None
        if err is not None:
            raise err

    def close(self) -> None:
        """Release the traces, the persistent inputs, the KV pool and the model weights (standalone runs)."""
        self.release_traces()
        for p in list(self._paths.values()):
            self._free_path(p)
        if self._pool is not None:
            self._pool.deallocate()
            self._pool = None
        self.model.deallocate()


__all__ = [
    "DECODE_KINDS",
    "DecodePath",
    "MotifGenerator",
    "PLAIN",
    "PrefillBatchPlan",
    "PrefillRowJob",
    "SPEC",
    "SpecPass",
    "SpecStepPlan",
    "check_decode_page_tables",
    "check_prefill_page_table",
    "plan_spec_step",
    "prefill_page_table_host",
]
