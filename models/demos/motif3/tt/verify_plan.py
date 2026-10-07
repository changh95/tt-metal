# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""T64's host side: the plan of one 64-row verify step, the ``auto`` choice between the two decode traces, the
drafting crossover ``c*`` and the result mapping (``docs/p5_t64/P5_T64_DESIGN.md`` §2.2, §4.5, §4.7, decisions X2 /
T5 / T7; review edits R-E3, R-E6, R-E9; ``docs/p5_t64/t64.md`` §3).

Pure host: it uses torch and the motif3 host model (``generator_api``, ``tt/kv_write.py``'s step, checks and partner
packing) and runs no device op. Every check runs before the step's first device op, so a refused step changes
nothing. ``tt/generator.py`` imports this module, and this module imports nothing from the generator.

**The T64 step** (design §4.1). Each DP row ``r`` has 16 physical rows, still one 32-row tile row:

* rows ``16 r + j`` (``j < 8``): the anchor of lane ``8 r + j``, the last committed token at ``n`` (idle: ``-1``);
* rows ``16 r + 8 + j``: the same lane's draft at ``n + 1``, with the lane's own page-table row (no draft: ``-1``).

:func:`plan_wide_step` turns the unchanged owner-lane :class:`~models.demos.motif3.tt.generator_api.SpecDecodeBatch`
into those 64 rows: tokens, positions (which give the RoPE rows, FlashMLA's ``cur_pos [16]`` and the A'' halves), page
tables and the ``KVWriteStep.wide_verify`` KV write. It runs the same host checks as ``generator.plan_spec_step``
first. The device returns the main and MTP argmax ``a`` / ``m`` ``[64]`` in split order: users 0..31 are the anchors
in lane order, users 32..63 the drafts (``kv_write.split_order``, the order of the split-order LM-head gather).
:meth:`WideStepPlan.result` maps them to the :class:`~models.demos.motif3.tt.generator_api.SpecDecodeResult` that a
T32 step returns: ``argmax[l] = (a[l], a[32 + l])`` and ``mtp_argmax[l] = (m[l], m[32 + l])``. The plugin / bridge
contract therefore does not change.

**Choosing the trace** (``MOTIF3_SPEC_VERIFY``, design §2.2), :func:`choose_verify_kind`:

* ``packed``: always the T32 spec trace (``"spec"``). Drafts without an idle lane take the overflow pass.
* ``wide``: always the T64 trace (``"wide"``).
* ``auto``: ``"spec"`` for an ordinary step, for a step that wants logits or sampling, and for a verify step whose
  drafts all get idle partner lanes (:func:`drafts_fit_idle_lanes`: ``kv_write.assign_partner_lanes`` leaves no
  overflow, so the T32 step needs one replay). ``"wide"`` otherwise: one T64 replay (~1.12x a T32 step) instead of two
  T32 replays. Bridge verify steps never want logits or sampling (R-E6), so ``auto`` never runs the overflow pass for
  them.

**Drafting all lanes** (design T7, §4.7; R-E3, R-E9). :func:`crossover_lanes` is the guarded ``c* = ceil(32 a r / ((1 +
a) - (1 - a) r))``. It returns the live-lane count from which T64 with every lane drafting beats the T32 packed
verify, in ``[17, 33]``; 33 means never. Here ``a`` is the acceptance and ``r`` the T64 / T32 step ratio
(``MotifTTConfig.wide_step_ratio``). The guard returns 33 whenever ``a <= r - 1``, because T64 then never wins at
``c <= 32``. :func:`drafts_all_lanes` is the generator's ``MotifGenerator.drafts_all_lanes`` answer as a pure function:
``packed`` gives False and ``wide`` gives True. ``auto`` gives True iff the live lanes reach
``GeneratorSettings.wide_min_lanes`` when it is set, else ``c*``. The acceptance is the bridge's prior-smoothed
estimate (``generator_api.smoothed_acceptance``); None (nothing verified yet) means the prior ``alpha_0`` (0.85), so a
server whose first traffic is a 32-request burst still drafts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple

import torch

from .generator_api import (
    DEFAULT_SPEC_ALPHA_PRIOR,
    LANES_PER_GROUP,
    NUM_LANES,
    SPEC_VERIFY_MODES,
    WIDE_MIN_LANES_NEVER,
    WIDE_ROWS,
    SpecDecodeBatch,
    SpecDecodeResult,
    check_wide_min_lanes,
)
from .kv_write import (
    KVWriteStep,
    allows_cross_row_partners,
    assign_partner_lanes,
    check_kv_write_step,
    check_mode,
    check_wide_layout,
    gathers_split,
    is_replicated,
    is_split,
    lanes_per_call_for,
    split_order,
)

if TYPE_CHECKING:  # pragma: no cover
    from .model_config import MotifTTConfig

SPEC_STEP = "spec"  # the T32 spec trace (generator decode path kind "spec")
WIDE_STEP = "wide"  # the T64 trace (generator decode path kind "wide")
VERIFY_STEP_KINDS = (SPEC_STEP, WIDE_STEP)
# c* bounds (R-E3): with c <= 16 live lanes every draft fits one of the >= 16 idle lanes (KV-R), so the clamp starts at
# 17; WIDE_MIN_LANES_NEVER (33 = NUM_LANES + 1) means "never draft every lane".
CROSSOVER_MIN_LANES = NUM_LANES // 2 + 1  # 17
CROSSOVER_NEVER = WIDE_MIN_LANES_NEVER  # 33


# ======================================================================================================================
# the drafting crossover (design T7, §4.5; R-E3)
# ======================================================================================================================
def _check_rate(name: str, value: Any) -> float:
    if isinstance(value, bool) or value is None:
        raise TypeError(f"{name} must be a number in [0, 1], got {value!r}")
    v = float(value)
    if not 0.0 <= v <= 1.0:  # NaN fails too
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return v


def crossover_lanes(alpha: float, ratio: float) -> int:
    """The live-lane count ``c*`` from which T64 with every lane drafting beats the T32 packed verify (design §4.5,
    review edit R-E3), clamped to ``[17, 33]``; 33 (:data:`CROSSOVER_NEVER`) means never.

    Per step, T64 commits ``c (1 + a)`` tokens in time ``r``. The packed verify commits ``c + (32 - c) a`` tokens in
    time 1, because only the ``32 - c`` idle lanes can take drafts. T64 wins iff ``c ((1 + a) - (1 - a) r) > 32 a r``.

    * ``a <= r - 1`` (``r`` 1.126: ``a <= 0.126``): T64 never wins at ``c <= 32``, so the function returns 33. Without
      this guard the formula fails at low ``a``: ``a = 0`` gives 0, and ``a <= (r - 1) / (r + 1)`` makes the denominator
      ``<= 0``, which would draft every lane exactly when T64 loses.
    * Otherwise it returns ``min(33, max(17, ceil(32 a r / ((1 + a) - (1 - a) r))))``. Below 17 live lanes every draft
      fits an idle lane under KV-R anyway.

    Args:
        alpha: the acceptance ``a`` in ``[0, 1]`` (the bridge's prior-smoothed estimate).
        ratio: ``r`` = T64 step time / T32 spec step time (``MotifTTConfig.wide_step_ratio``, 1.21 by default; G16
            measures it), finite and ``> 0``.
    """
    a = _check_rate("the acceptance alpha", alpha)
    if isinstance(ratio, bool) or ratio is None:
        raise TypeError(f"the T64 / T32 step ratio must be a positive number, got {ratio!r}")
    r = float(ratio)
    if not (math.isfinite(r) and r > 0.0):
        raise ValueError(f"the T64 / T32 step ratio must be finite and > 0, got {ratio!r}")
    if a <= r - 1.0:
        return CROSSOVER_NEVER
    den = (1.0 + a) - (1.0 - a) * r  # > r (r - 1) >= 0 here
    c = math.ceil(NUM_LANES * a * r / den)
    return min(CROSSOVER_NEVER, max(CROSSOVER_MIN_LANES, c))


def _live_count(live_lanes: Iterable[int]) -> int:
    lanes = set()
    for x in live_lanes:
        if isinstance(x, bool):
            raise TypeError(f"live lanes must be lane ids, got {x!r}")
        lane = int(x)
        if lane != x or not 0 <= lane < NUM_LANES:
            raise ValueError(f"live lane {x!r} is not a lane id in [0, {NUM_LANES})")
        lanes.add(lane)
    return len(lanes)


def drafts_all_lanes(
    live_lanes: Iterable[int],
    *,
    spec_verify: str,
    ratio: float,
    acceptance: Optional[float] = None,
    min_lanes: Optional[int] = None,
    prior: float = DEFAULT_SPEC_ALPHA_PRIOR,
) -> bool:
    """``MotifGenerator.drafts_all_lanes`` as a pure function (design §4.7, R-E9). It says whether the bridge may
    propose a draft for every live lane of the next step (True), or keeps its idle-lane budget (False).

    * ``spec_verify="packed"``: False. Every draft must fit an idle lane of the T32 trace.
    * ``"wide"``: True. Every verify step runs in the T64 trace.
    * ``"auto"``: True iff the distinct live lanes reach ``min_lanes`` (``GeneratorSettings.wide_min_lanes``,
      ``MOTIF3_WIDE_MIN_LANES``; 33 = never) when it is set, else ``crossover_lanes(acceptance, ratio)``.

    Args:
        live_lanes: the lanes that carry a request in the next step (ids in ``[0, 32)``; duplicates count once).
        spec_verify: the launch's ``spec_verify`` (:data:`generator_api.SPEC_VERIFY_MODES`).
        ratio: ``MotifTTConfig.wide_step_ratio``.
        acceptance: the bridge's acceptance estimate (``generator_api.smoothed_acceptance``), or None before any draft
            was verified. None then means ``prior`` (R-E3: without it, a server whose first traffic is a 32-request
            burst would never draft and never measure an acceptance).
        min_lanes: the explicit threshold, or None for ``c*``.
        prior: ``GeneratorSettings.spec_alpha_prior`` (``alpha_0``).
    """
    if spec_verify not in SPEC_VERIFY_MODES:
        raise ValueError(f"spec_verify must be one of {SPEC_VERIFY_MODES}, got {spec_verify!r}")
    live = _live_count(live_lanes)
    if spec_verify == "packed":
        return False
    if spec_verify == "wide":
        return True
    if min_lanes is not None:
        return live >= check_wide_min_lanes(min_lanes)
    alpha = _check_rate("the acceptance prior", prior) if acceptance is None else acceptance
    return live >= crossover_lanes(alpha, ratio)


# ======================================================================================================================
# routing a step of a speculating launch (design §2.2, §4.5; R-E6)
# ======================================================================================================================
def drafts_fit_idle_lanes(batch: SpecDecodeBatch, *, kv_mode: str, lanes_per_row: int = LANES_PER_GROUP) -> bool:
    """True iff the T32 packed verify runs ``batch`` in ONE replay, i.e. every draft gets an idle partner lane:
    ``kv_write.assign_partner_lanes`` under ``kv_mode``'s row rule (any DP row with ``all_split``, the owner's row with
    ``row_split``) leaves no overflow. That is exactly the case where ``generator.plan_spec_step`` makes one pass. A
    step without drafts fits. Without a split mode no draft is packed, so a step with drafts does not fit."""
    if not isinstance(batch, SpecDecodeBatch):
        raise TypeError(f"need a SpecDecodeBatch, got {type(batch).__name__}")
    check_mode(kv_mode)
    if not batch.is_verify:
        return True
    if not is_split(kv_mode):
        return False
    _, overflow = assign_partner_lanes(
        batch.positions, batch.has_draft, cross_row=allows_cross_row_partners(kv_mode), lanes_per_row=lanes_per_row
    )
    return not overflow


def choose_verify_kind(
    batch: SpecDecodeBatch,
    mode: str,
    want_logits: bool = False,
    sampling: Any = None,
    *,
    kv_mode: str,
    lanes_per_row: int = LANES_PER_GROUP,
) -> str:
    """The decode trace that runs ``batch`` (design §2.2, §4.5): :data:`SPEC_STEP` (the T32 spec trace) or
    :data:`WIDE_STEP` (the T64 trace).

    Args:
        batch: the step (owner-lane order).
        mode: the launch's ``spec_verify``: ``"packed"`` gives always ``"spec"``, ``"wide"`` always ``"wide"``.
            ``"auto"`` gives ``"spec"`` for an ordinary step, for a step that wants logits or sampling, and for a
            verify step whose drafts all fit idle lanes (:func:`drafts_fit_idle_lanes`); ``"wide"`` otherwise.
        want_logits: the step returns the anchors' logits (``decode_forward_spec(want_logits=True)``). Bridge verify
            steps never do (R-E6), only test-made ones, which then run like ``packed`` (T32, with the overflow pass
            when the drafts do not fit).
        sampling: the step's device-sampling parameters (None or False = none). The device sampler lives in the T32
            trace only in ``auto``.
        kv_mode: the T32 spec path's KV-write mode (its partner rule decides what fits).
        lanes_per_row: owner lanes per DP row (8).
    """
    if mode not in SPEC_VERIFY_MODES:
        raise ValueError(f"spec_verify must be one of {SPEC_VERIFY_MODES}, got {mode!r}")
    if not isinstance(batch, SpecDecodeBatch):
        raise TypeError(f"need a SpecDecodeBatch, got {type(batch).__name__}")
    check_mode(kv_mode)
    if mode == "packed":
        return SPEC_STEP
    if mode == "wide":
        return WIDE_STEP
    if not batch.is_verify or want_logits or (sampling is not None and sampling is not False):
        return SPEC_STEP
    return SPEC_STEP if drafts_fit_idle_lanes(batch, kv_mode=kv_mode, lanes_per_row=lanes_per_row) else WIDE_STEP


# ======================================================================================================================
# the plan of one T64 step (design §4.5; T64N §3.1)
# ======================================================================================================================
def check_page_table_entries(
    positions: Sequence[int], page_table: torch.Tensor, *, block_size: int, num_blocks: Optional[int] = None
) -> None:
    """Every page-table entry an active owner lane uses must be a real block id ``>= 1`` (0 is vLLM's null block) and
    ``< num_blocks`` (when given). The used entries are ``0 .. top // block_size``: FlashMLA reads them, and the KV
    writes land in the last ones. ``positions``: each lane's highest position this step (``n + 1`` on drafted lanes,
    ``-1`` = inactive). Raises ``ValueError``. This is ``generator.check_decode_page_tables``, kept here so that this
    module does not import the generator. Only the used columns are scanned."""
    if isinstance(positions, torch.Tensor):
        pos = positions.reshape(-1).to(torch.int64)
    else:
        pos = torch.tensor(list(positions), dtype=torch.int64)
    if int(pos.shape[0]) != int(page_table.shape[0]):
        raise ValueError(f"{int(pos.shape[0])} positions for a page table of {int(page_table.shape[0])} rows")
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


def _ids(name: str, t: Any, n: int) -> List[int]:
    ids = torch.as_tensor(t).reshape(-1)
    if ids.dtype.is_floating_point or ids.dtype == torch.bool:
        raise TypeError(f"{name} must hold integer token ids, got {ids.dtype}")
    if int(ids.numel()) != n:
        raise ValueError(f"{name} must hold the {n} split-order rows of the T64 step, got {int(ids.numel())}")
    return ids.to(torch.int64).tolist()


@dataclass(frozen=True)
class WideStepPlan:
    """The host plan of one T64 step (:func:`plan_wide_step`). All tensors are host torch tensors.

    Attributes:
        mode: the KV-write mode of the step (``row_split`` / ``all_split``).
        gather: the KV-R gather order the step was checked for (``kv_write.GATHER_ORDERS``).
        batch: the owner-lane :class:`~models.demos.motif3.tt.generator_api.SpecDecodeBatch`.
        tokens: ``int32 [64]`` per physical row: the anchor token in each lane's anchor row, the draft in its draft
            row, and 0 on idle rows (``embedding.decode_tokens_host(..., rows_per_dp=16)``).
        step: the 64-row ``KVWriteStep.wide_verify`` (its ``positions [64]`` are ``n`` / ``n + 1`` / ``-1``: the
            RoPE rows, FlashMLA's ``cur_pos`` and the A'' halves; its page tables carry each draft's owner row).
        lanes_per_row: owner lanes per DP row (8); the step has twice as many rows per DP row.
    """

    mode: str
    gather: str
    batch: SpecDecodeBatch
    tokens: torch.Tensor
    step: KVWriteStep
    lanes_per_row: int = LANES_PER_GROUP

    @property
    def positions(self) -> torch.Tensor:
        """``int32 [64]`` physical-row positions (``n`` / ``n + 1`` / ``-1``)."""
        return self.step.positions

    @property
    def rows_per_dp(self) -> int:
        return 2 * int(self.lanes_per_row)

    @property
    def is_verify(self) -> bool:
        return self.batch.is_verify

    @property
    def num_drafts(self) -> int:
        return self.step.num_partners

    @property
    def drafted_lanes(self) -> Tuple[int, ...]:
        return tuple(int(x) for x in torch.nonzero(self.batch.has_draft).reshape(-1).tolist())

    def result(self, a: Any, m: Any, logits: Optional[torch.Tensor] = None) -> SpecDecodeResult:
        """The step's :class:`~models.demos.motif3.tt.generator_api.SpecDecodeResult` in owner-lane order, from the
        device's main / MTP argmax ``a`` / ``m`` (64 ids each, split order: ``[l]`` = lane ``l``'s anchor row, ``[32 +
        l]`` = its draft row; ``[1, 1, 1, 64]`` or flat): ``argmax[l] = (a[l], a[32 + l])``, ``mtp_argmax[l] = (m[l],
        m[32 + l])``. Entries the contract leaves unspecified (inactive lanes, column 1 of undrafted lanes) are -1, as
        in ``generator.SpecStepPlan.result``. ``logits``: the anchors' host logits ``[32, vocab]`` (only the ``wide``
        mode's host-sampled ordinary steps read them), else None."""
        n = NUM_LANES
        a_l, m_l = _ids("a", a, 2 * n), _ids("m", m, 2 * n)
        am = [[-1, -1] for _ in range(n)]
        mm = [[-1, -1] for _ in range(n)]
        for lane, (p, d) in enumerate(zip(self.batch.positions.tolist(), self.batch.draft_tokens.tolist())):
            if p < 0:
                continue
            am[lane][0], mm[lane][0] = a_l[lane], m_l[lane]
            if d >= 0:
                am[lane][1], mm[lane][1] = a_l[n + lane], m_l[n + lane]
        return SpecDecodeResult(
            logits=logits, argmax=torch.tensor(am, dtype=torch.int32), mtp_argmax=torch.tensor(mm, dtype=torch.int32)
        )


def plan_wide_step(
    batch: SpecDecodeBatch,
    *,
    cfg: "MotifTTConfig",
    width: Optional[int] = None,
    mode: Optional[str] = None,
    num_blocks: Optional[int] = None,
    lanes_per_call: Optional[int] = None,
    gather: str = "split",
) -> WideStepPlan:
    """Every host check and the 64-row layout of one T64 step (design §4.5), before any device op. Raises
    ``ValueError`` (``TypeError`` for a non-batch) on a bad batch.

    * Checks, as ``generator.plan_spec_step``: the page-table width (``width``: the trace's), positions ``<
      max_model_len`` (also ``n + 1`` on drafted lanes), anchor and draft token ids ``< vocab``, every used page-table
      entry (:func:`check_page_table_entries`, up to ``n + 1`` on drafted lanes).
    * Layout: ``KVWriteStep.wide_verify(positions, page_table, has_draft)`` and the tokens per physical row
      (``kv_write.wide_rows`` / ``kv_write.split_order``). Then ``kv_write.check_kv_write_step`` checks the step at 16
      rows per DP row (partner rules, one draft per owner, no block written twice in one update call) together with
      the T64 layout (``kv_write.check_wide_layout``).

    Args:
        batch: the step in owner-lane order (an ordinary step gives a T64 step with idle draft rows: the ``wide``
            mode's ordinary steps).
        cfg: the launch's ``MotifTTConfig`` (block size, ``max_model_len``, vocab, lanes per DP row, KV dtype,
            ``kv_write_mode``).
        width: the T64 trace's page-table width, once captured (None = not checked).
        mode: the T64 path's KV-write mode (default ``cfg.kv_write_mode``); a split mode (every T64 step can carry
            drafts in call B).
        num_blocks: the KV pool's block count (None = ids not bounded above).
        lanes_per_call: the T64 ``DecodeKVWrite``'s users per KV-R update call (default ``kv_write.lanes_per_call_for``
            for the gather: 32 for bfp8, 16 for bf16).
        gather: the T64 ``DecodeKVWrite``'s KV-R gather order (``"split"``, the production order, or ``"natural"``;
            the row modes have no gather).
    """
    if not isinstance(batch, SpecDecodeBatch):
        raise TypeError(f"plan_wide_step needs a SpecDecodeBatch, got {type(batch).__name__}")
    mode = check_mode(cfg.kv_write_mode if mode is None else mode)
    if not is_split(mode):
        raise ValueError(
            f"the T64 step needs a split KV-write mode (row_split / all_split: the drafts are written in call B), got "
            f"{mode!r}"
        )
    gathers_split(mode, gather)  # validates the order
    W, bs = batch.page_table_width, int(cfg.kv_block_size)
    if width is not None and W != int(width):
        raise ValueError(f"page-table width {W} != the decode trace's width {width}")
    n, lpr = NUM_LANES, int(cfg.lanes_per_row)
    if int(cfg.max_batch) != n:
        raise ValueError(f"the T64 step needs a {n}-lane config, got max_batch {cfg.max_batch}")
    pos_l, dr_l, tok_l = batch.positions.tolist(), batch.draft_tokens.tolist(), batch.tokens.tolist()
    L, V = int(cfg.max_model_len), int(cfg.vocab_size)
    top_l = [p + 1 if (d >= 0 and p >= 0) else p for p, d in zip(pos_l, dr_l)]
    for lane, t in enumerate(top_l):
        if t >= L:
            raise ValueError(f"lane {lane}: decode position {t} >= max_model_len {L}")
    tok1 = [t if p >= 0 else 0 for t, p in zip(tok_l, pos_l)]
    if any(not 0 <= t < V for t in tok1) or any(d >= V for d in dr_l):
        raise ValueError(f"decode token ids must be in [0, {V}): anchors {tok1}, drafts {dr_l}")
    check_page_table_entries(top_l, batch.page_table, block_size=bs, num_blocks=num_blocks)
    step = KVWriteStep.wide_verify(batch.positions, batch.page_table, batch.has_draft, lanes_per_row=lpr)
    rows_per_dp = 2 * lpr
    order = split_order(WIDE_ROWS, rows_per_dp)  # [lane] = its anchor row, [n + lane] = its draft row (wide_rows)
    tokens = [0] * WIDE_ROWS
    for lane in range(n):
        tokens[order[lane]] = tok1[lane]
        if dr_l[lane] >= 0:
            tokens[order[n + lane]] = dr_l[lane]
    kw = dict(block_size=bs, max_seq_len=L, lanes_per_row=rows_per_dp, gather=gather)
    if is_replicated(mode):
        kw["lanes_per_call"] = (
            lanes_per_call
            if lanes_per_call is not None
            else lanes_per_call_for(
                mode, cfg.dtypes.kv_cache_name, lanes=WIDE_ROWS, lanes_per_row=rows_per_dp, gather=gather
            )
        )
    check_kv_write_step(step, mode, **kw)
    if not gathers_split(mode, gather):  # the split gather's check covers the layout
        check_wide_layout(step, lanes_per_row=rows_per_dp)
    return WideStepPlan(
        mode=mode,
        gather=gather if is_replicated(mode) else "natural",
        batch=batch,
        tokens=torch.tensor(tokens, dtype=torch.int32),
        step=step,
        lanes_per_row=lpr,
    )


__all__ = [
    "CROSSOVER_MIN_LANES",
    "CROSSOVER_NEVER",
    "SPEC_STEP",
    "VERIFY_STEP_KINDS",
    "WIDE_STEP",
    "WideStepPlan",
    "check_page_table_entries",
    "choose_verify_kind",
    "crossover_lanes",
    "drafts_all_lanes",
    "drafts_fit_idle_lanes",
    "plan_wide_step",
]
