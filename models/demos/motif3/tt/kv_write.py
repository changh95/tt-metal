# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode KV writes of every decode layer: KV-R and the speculative two-call split (features design §3.4-§3.5, D6,
D10, §3.8.2; README CONVENTIONS §16; gates G12 / G13a in ``tests/unit/gates/GATES_RESULTS.md`` §12.6-§12.7).

One :class:`DecodeKVWrite` per server serves all 54 decode layers (the 53 decoder layers and the MTP layer, no
per-layer variant: design risk R8). It owns the per-step lane tensors of the decode KV write, and the per-row
``cur_pos`` / ``page_table`` that FlashMLA reads, so the write and the read always describe the same lanes.
:meth:`DecodeKVWrite.write` (the ``tt.attention.DecodeKVWriter`` protocol) writes one layer's new latent rows
``kv_row [1, 1, 8, 576]`` (the attention's ``concat(n, rope(k_pe))``) into that layer's paged cache. The mode is fixed
when the decode trace is captured (``cfg.kv_write_mode`` = ``generator_api.kv_write_mode(kv_replicated, spec)``):

=============  ==========================================  ======================================================
mode           when                                        ops per layer
=============  ==========================================  ======================================================
``row``        draft 1 (no prefix caching, no MTP)         ``transpose`` -> ``[1, 8, 1, 576]`` on 8 cores, one
                                                           8-lane ``paged_update_cache`` (each DP row's chips
                                                           write their own 8 lanes: exactly the draft-1 ops)
``row_split``  speculation without prefix caching          the same input, call A then call B (8 lanes each)
``all``        prefix caching (KV-R)                       ``ccl.ag_dp_rows`` -> ``[1, 1, 32, 576]`` (lane order
                                                           ``8 dp + l``) -> ``transpose`` -> ``[1, 32, 1, 576]`` on
                                                           32 cores, one 32-lane update on every chip
``all_split``  prefix caching + speculation (production)   the gathered latent, call A then call B (32 lanes each)
=============  ==========================================  ======================================================

**KV-R** (design D6, §3.4). One vLLM block pool and one prefix-cache hash table serve all 32 lanes, and vLLM caches
full blocks that decode filled. Without KV-R a lane's decode latent exists only on its own DP row, so a prefix hit on
such a block from another row reads stale KV on 3 of 4 rows, and the replicated prefill plus the MoE reduce-scatter
spread the error to every row. KV-R writes every decode row on all 32 chips: AG(dp) of the row's ``[8, 576]`` latent
and a 32-lane update with the replicated ``[32, W]`` page table on every chip.

**Two calls** (D10, gate G12). ``paged_update_cache`` runs one user per core and read-modify-writes the user's whole
32-row tile. Packed verify puts a draft at ``n + 1`` on an idle *partner* lane whose page-table row is the owner's
(the owner writes its anchor at ``n``). In one call the two users race on the shared tile and an update is lost:
G12 lost 2-3 (8-lane) / 10-13 (32-lane) updates in every trial. So the split modes write call A (owners and plain
lanes) and then call B (partners). On ordinary steps call B is all ``-1`` (skipped by the kernel, ~0.2 us). FlashMLA
runs after both calls.

**A step** (:class:`KVWriteStep`, host torch, physical-lane order): ``positions [32]`` = the KV write slot of each
lane (``-1`` = no write), which is also FlashMLA's ``cur_pos``; ``page_table [32, W]``; ``call_b [32]`` marks the
partner lanes and ``owner [32]`` names each partner's owner. Build it with :meth:`KVWriteStep.ordinary`,
:meth:`KVWriteStep.packed_verify` (owners at ``n``, drafts at ``n + 1`` on the partner lanes chosen by
:func:`assign_partner_lanes`) or :meth:`KVWriteStep.overflow_pass` (the second replay for drafts that found no idle
lane: each draft at ``n + 1`` on its own lane, all other lanes inactive; design §3.8.2 item 4).

**Partners.** A partner reads its owner's history and the anchor at ``n`` from its own chips' cache copy. With
``all_split`` every chip holds every lane's KV (KV-R, anchor written in call A before FlashMLA), so a partner may sit
on any DP row. With ``row_split`` it must sit on the owner's DP row (G13a negative control: a cross-row partner
without KV-R read a stale anchor, PCC down to 0.40). :func:`check_kv_write_step` enforces this, the partner rules
(``position = owner + 1``, the owner's page-table row, one draft per owner, owner in call A) and the race rule: within
one update call no block id is written twice (two rows of one request never share a call; stricter than "no shared
tile", which is the actual kernel hazard). :meth:`DecodeKVWrite.write_step` runs it on every step by default.

**Trace safety.** Every device input is a persistent tensor allocated once for the trace's page-table width ``W``
(:meth:`DecodeKVWrite.write_step` rewrites it with ``copy_host_to_device_tensor`` before a replay, and skips inputs
whose values did not change, e.g. page tables between block boundaries). :meth:`DecodeKVWrite.write` issues a fixed,
mode-dependent op sequence with fixed shapes: no host round trip, no data-dependent control flow, no program that a
warmup step does not compile (one eager step, all lanes inactive, compiles everything; one program per op for all 54
layers because every layer's cache has the same shape).

**bf16 KV caches** (``MOTIF3_KV_CACHE_DTYPE=bf16``). The update op sizes its output circular buffer as ``B x Wt``
tiles per core (``B`` = users of the call, ``Wt`` = 18): 32 x 18 x 2048 B = 1.18 MB, which clashes with the sharded
input in L1 (G12). The KV-R modes therefore split the 32 gathered lanes into calls of :data:`MAX_LANES_PER_CALL`
users: ``transpose`` to DRAM ``[1, 32, 1, 576]``, then per chunk ``slice`` (dim 1) -> ``to_memory_config`` (16
cores) -> the chunk's call(s). ``lanes_per_call=8`` is the fallback if a 32-lane bfp8 update ever clashes with an L1
buffer of the full decode (the 32-lane bfp8 update's static CBs are ~0.8 MB per core; G13a request 3).

**Costs** (G13a, traced, 54-layer decode-sized trace incl. FlashMLA and the ``wo`` AR, delta per step vs ``row``):
``row_split`` +0.19 ms (3.5 us per layer), ``all`` +1.56 ms (28.8-29.1 us), ``all_split`` +1.87-1.89 ms
(34.6-34.9 us) at 1K-8K context: ~2.2 % of the 83.8 ms decode step, inside the 2.0 ms gate. :meth:`DecodeKVWrite.write`
alone, traced (``tests/unit/test_kv_write.py``, 2026-10-02): ``row`` 8.1, ``row_split`` 12.9, ``all`` 37.0,
``all_split`` 41.9 us per layer with bfp8 (+1.83 ms per 54-layer step for ``all_split`` vs ``row``); bf16 ``all`` 53.2
/ ``all_split`` 63.0 us (2 x 16-lane calls). Alternatives measured and rejected: the sharded all-gather (G13a "sag")
costs the same; gathering in ROW_MAJOR and tilizing straight into the sharded layout costs 53-93 us. Host side,
:meth:`DecodeKVWrite.write_step` validates in 26-53 us; each changed input then costs one ``copy_host_to_device_tensor``
(~0.2-0.4 ms; a steady ordinary step changes ``cur_pos`` and the call-A positions). The deferred KV-R variant (design
§3.12.3: own-row writes per layer, one batched remote write after the last layer, +1.35 ms) is a later optimization;
it would live in :meth:`DecodeKVWrite.write` / :meth:`DecodeKVWrite.end_step` and requires same-row partners.

**T64: the 64-row verify step** (``docs/p5_t64/P5_T64_DESIGN.md`` §4.1, §4.3, T2 / T3; ``docs/p5_t64/t64.md`` §2.4).
``DecodeKVWrite(rows=64)`` serves the T64 step: each DP row carries 16 rows ``[8 anchors | 8 drafts]``, still one
32-row tile row. Lane ``8 r + j`` has its anchor in row ``16 r + j`` (the last committed token at ``n``) and its draft
in row ``16 r + 8 + j`` (at ``n + 1``, with the lane's own page-table row) (:func:`wide_rows`). Build the step with
:meth:`KVWriteStep.wide_verify`, which is :meth:`KVWriteStep.packed_verify` with ``partner_of = {16 r + j: 16 r + 8 +
j}``. :func:`check_wide_layout` holds the layout that the split gather and the FlashMLA groups rely on. The ops per
layer:

=====================  ==================================================================  =========================
mode (gather)          ops per layer                                                       update calls x users
=====================  ==================================================================  =========================
``row_split``          per DP row ``transpose`` -> ``[1, 16, 1, 576]`` on 16 cores, call   2 x 16 (each DP row)
                       A (anchors at ``n``), then call B (drafts at ``n + 1``)
``all_split``          ``ccl.ag_dp_rows(kv_row, halves=2)`` -> ``[1, 1, 64, 576]``: rows   2 x 32 (bfp8),
(``split``, default)   0..31 = the anchors in T32 lane order ``8 dp + l``, rows 32..63 =   4 x 16 (bf16, R-E8)
                       the drafts. Per half one tile-row ``slice`` + ``transpose`` to 32
                       cores, then call A on the anchors and call B on the drafts. Both use
                       ONE replicated ``pt [32, W]``: a draft carries its owner's row.
``all_split``          ``ccl.ag_dp_rows(kv_row)`` -> rows ``16 dp + j``, then per 32-row     4 x 32 (bfp8),
(``natural``)          chunk (2 DP rows): call A, then call B                              8 x 16 (bf16)
=====================  ==================================================================  =========================

The split order is the production layout. Traced per layer with this writer (``tests/unit/test_kv_write.py``
``test_kv_write_device_wide_cost``, 2026-10-03), it costs 58.4 us against 66.2 us for natural (T32 ``all_split``:
41.8), and ``row_split`` costs 16.8 us (T32: 11.6). A bf16 split write (4 x 16) costs 90.3 us (T32 bf16: 62.6).
G16-lite measured 59.0 / 66.3 / 16.9 us with an injected writer of the same op sequence (T64N §5.1). ``row`` /
``all`` have no call B, so :func:`check_kv_write_step` refuses their T64 steps with drafts, as at 32 rows. The row
modes have no gather and ignore ``gather``.

**FlashMLA groups** (option A'', design §4.3 / T3). :meth:`DecodeKVWrite.flash_groups` returns the FlashMLA inputs
per group of rows of this chip's DP row as ``[(row slice, cur_pos, page_table)]``. At 8 rows per DP row there is one
group, ``(0:8, cur_pos, page_table)``, which is the unchanged T32 call. At 16 rows there are two B = 8 groups:
``(0:8, cur_pos_a, pt_a)`` for the anchors at ``n`` and ``(8:16, cur_pos_d, pt_a)`` for the drafts at ``n + 1``.
``pt_a`` is the anchors' ``[8, W]`` rows, and the drafts share it: a draft carries its owner's row, and an idle draft
row is skipped (``cur_pos = -1``). The global layers run one FlashMLA call per group. The SWA layers keep the single
B = 16 call on :attr:`DecodeKVWrite.cur_pos` / :attr:`DecodeKVWrite.page_table` (``[16]`` / ``[16, W]`` per DP row).
At B = 16 the global layers' per-user core split differs from B = 8, while the SWA layers' does not. With the groups,
every T64 row therefore equals the T32 row bit for bit. The group inputs are persistent and :meth:`write_step` writes
them (F3N rule R3).

**Integration** (callers; this module edits nothing else):

* attention (WP2b, landed): ``MotifAttention.forward_decode(..., kv_write=w)`` calls ``w.write(kv_row, kv_cache,
  cur_pos=cur_pos, page_table=page_table)`` once per layer in place of the draft-1 ``transpose`` +
  ``paged_update_cache``, before FlashMLA, and frees ``kv_row`` itself. :meth:`DecodeKVWrite.write` requires
  ``cur_pos`` / ``page_table`` to be its own :attr:`DecodeKVWrite.cur_pos` / :attr:`DecodeKVWrite.page_table`. In
  ``row`` mode the device ops are then exactly draft 1's (only ``kv_row`` is freed after the update instead of before
  it; ``tests/unit/test_attention_kvr.py`` checks both on the host and bitwise on device).
* generator / model (WP5): build one ``DecodeKVWrite(mesh, cfg, ccl=..., page_table_width=W)`` with the decode inputs
  (before ``warmup_decode``); per step ``kv_write.write_step(step)``; pass ``cur_pos=kv_write.cur_pos,
  page_table=kv_write.page_table, kv_write=kv_write`` to every layer and to the MTP layer, and build the ``active``
  mask from ``kv_write.cur_pos``; call ``kv_write.end_step()`` after the last layer (a no-op today).
* T64 (WP-I I2 / WP-A A2): ``DecodeKVWrite(mesh, cfg, ccl=..., page_table_width=W, rows=64, gather="split")`` staged
  with the T64 path, before any capture. The per-step :class:`KVWriteStep` comes from ``verify_plan.plan_wide_step``.
  The global layers' FlashMLA runs once per entry of :meth:`DecodeKVWrite.flash_groups`, on ``q[:, rows]``, and
  concatenates the outputs on dim 1.

**Host model.** :func:`kv_write_calls` lists the update calls of a step (which chips, which users, which positions),
:func:`apply_kv_writes_host` applies them to per-DP-row torch cache copies (what every chip of that row must hold
afterwards), and :func:`kv_write_inputs_host` gives the values of every persistent device input. The device tests
compare every chip's cache with that model bit-exactly (``tests/unit/test_kv_write.py``).

Import rule: ``torch``, ``ttnn`` and the motif3 shared infra (``generator_api``, ``model_config``, ``rope``). The host
planning functions use torch only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

import ttnn

from .generator_api import KV_LATENT_DIM, KV_WRITE_MODES, LANES_PER_GROUP, NUM_LANES, WIDE_ROWS_PER_GROUP
from .model_config import TILE, MotifTTConfig
from .rope import shard_lanes

SPLIT_MODES = ("row_split", "all_split")  # call A (owners + plain lanes), then call B (packed-verify partners)
REPLICATED_MODES = ("all", "all_split")  # KV-R: every decode KV write lands on all chips
# Users per paged_update_cache call in the KV-R modes, per KV-cache dtype (gate G12, module docstring "bf16 KV caches").
MAX_LANES_PER_CALL = {"bfp8": 32, "bf16": 16}
CALL_KINDS = ("A", "B")  # split modes; the non-split modes have one call kind ""
# Row order of a T64 step's KV-R gather (module docstring "T64"): "natural" = rows 16 dp + j (ccl.ag_dp_rows),
# "split" = the anchors of every DP row, then the drafts (ccl.ag_dp_rows(x, halves=2)). Only the KV-R modes gather.
GATHER_ORDERS = ("natural", "split")
# "row": dim 0 sharded over the DP rows (lanes_per_row rows each: 8, or 16 at T64), replicated over TP; "rep":
# replicated on every chip
Placement = str


# ======================================================================================================================
# modes
# ======================================================================================================================
def check_mode(mode: str) -> str:
    """``mode`` if it is one of :data:`generator_api.KV_WRITE_MODES`, else ``ValueError``."""
    if mode not in KV_WRITE_MODES:
        raise ValueError(f"kv_write mode must be one of {KV_WRITE_MODES}, got {mode!r}")
    return mode


def is_split(mode: str) -> bool:
    """The mode writes call A and call B (speculation: packed-verify partners in call B)."""
    return check_mode(mode) in SPLIT_MODES


def is_replicated(mode: str) -> bool:
    """KV-R: every decode KV write lands on all chips."""
    return check_mode(mode) in REPLICATED_MODES


def allows_cross_row_partners(mode: str) -> bool:
    """A packed-verify partner may sit on another DP row than its owner. Only ``all_split``: every chip holds every
    lane's history (KV-R) and the anchor at ``n`` is written on every chip in call A, before FlashMLA reads it."""
    return check_mode(mode) == "all_split"


def check_gather(gather: str) -> str:
    """``gather`` if it is one of :data:`GATHER_ORDERS`, else ``ValueError``."""
    if gather not in GATHER_ORDERS:
        raise ValueError(f"the KV-R gather order must be one of {GATHER_ORDERS}, got {gather!r}")
    return gather


def gathers_split(mode: str, gather: str) -> bool:
    """The step's rows reach the update calls in split order: ``gather == "split"`` in a KV-R mode. The row modes have
    no gather (each DP row writes its own rows), so they ignore ``gather``. ``"all"`` with ``"split"`` raises
    ``ValueError``: the split order writes the anchors (call A) and the drafts (call B) separately, and ``all`` has no
    call B."""
    check_gather(gather)
    if gather != "split" or not is_replicated(mode):
        return False
    if not is_split(mode):
        raise ValueError(
            f"the split-order gather writes anchors in call A and drafts in call B: mode {mode!r} has no call B"
        )
    return True


def lanes_per_call_for(
    mode: str,
    kv_dtype_name: str = "bfp8",
    *,
    lanes: int = NUM_LANES,
    lanes_per_row: int = LANES_PER_GROUP,
    requested: Optional[int] = None,
    gather: str = "natural",
) -> int:
    """Users per ``paged_update_cache`` call. Row modes: ``lanes_per_row`` (each DP row's chips write their own lanes;
    ``requested`` must be None or equal). KV-R modes: ``requested`` (a divisor of ``lanes`` up to the dtype maximum),
    default ``min(lanes, MAX_LANES_PER_CALL[kv_dtype_name])``: 32 for bfp8, 16 for bf16. With the split-order gather
    (:func:`gathers_split`; a T64 step) the calls take the anchors and the drafts separately: ``requested`` divides
    ``lanes / 2``, default ``min(lanes / 2, MAX_LANES_PER_CALL)`` (T64: 32 for bfp8, 16 for bf16)."""
    if kv_dtype_name not in MAX_LANES_PER_CALL:
        raise ValueError(f"KV cache dtype must be one of {tuple(MAX_LANES_PER_CALL)}, got {kv_dtype_name!r}")
    if not is_replicated(mode):
        return _users_per_call(mode, lanes, lanes_per_row, requested)
    if lanes % lanes_per_row:
        raise ValueError(f"{lanes} lanes do not split into DP rows of {lanes_per_row}")
    split = gathers_split(mode, gather)
    if split and lanes_per_row % 2:
        raise ValueError(f"the split-order gather needs an even number of rows per DP row, got {lanes_per_row}")
    users = lanes // 2 if split else lanes
    cap = min(users, MAX_LANES_PER_CALL[kv_dtype_name])
    n = cap if requested is None else int(requested)
    if n < 1 or users % n or n > cap:
        what = f"the {users} anchors (and drafts) of a split-order step" if split else f"{lanes}"
        raise ValueError(
            f"lanes_per_call must divide {what} and be <= {cap} for a {kv_dtype_name} cache (the update op's output "
            f"CB is users x 18 tiles per core), got {requested}"
        )
    return n


# ======================================================================================================================
# T64 rows (host)
# ======================================================================================================================
def wide_rows(lane: int, *, lanes_per_row: int = LANES_PER_GROUP) -> Tuple[int, int]:
    """``(anchor row, draft row)`` of owner ``lane`` in a T64 step (design §4.1). DP row ``r = lane // lanes_per_row``
    holds ``2 * lanes_per_row`` rows, ``[anchors | drafts]``: lane ``8 r + j`` -> rows ``16 r + j`` and ``16 r + 8 +
    j``. ``lanes_per_row``: owner lanes per DP row (8)."""
    n = int(lanes_per_row)
    r, j = divmod(int(lane), n)
    a = 2 * n * r + j
    return a, a + n


@lru_cache(maxsize=None)
def split_order(rows: int, rows_per_dp: int = WIDE_ROWS_PER_GROUP) -> Tuple[int, ...]:
    """The split-order gather of a T64 step (design §4.1): the physical row of each gathered user ``u``. Users ``[0,
    rows / 2)`` are the first halves of the DP rows in DP order (the anchors, in T32 lane order ``8 dp + l``); users
    ``[rows / 2, rows)`` are the second halves in the same order (the drafts). ``ccl.ag_dp_rows(x, halves=2)`` (view
    ``[1, 2, L / 2, W]``, gather dim 2 over DP) produces this order: user ``u`` of the gathered ``[1, 1, rows, W]`` is
    physical row ``split_order(rows, L)[u]``."""
    rows, per = int(rows), int(rows_per_dp)
    if per < 2 or per % 2 or rows % per:
        raise ValueError(f"the split order needs DP rows of an even number of rows dividing {rows}, got {per}")
    half, dp = per // 2, rows // per
    return tuple(per * r + h * half + j for h in range(2) for r in range(dp) for j in range(half))


@lru_cache(maxsize=None)
def _split_user_of(rows: int, rows_per_dp: int) -> Tuple[int, ...]:
    """Inverse of :func:`split_order`: the gathered user index of each physical row."""
    inv = [0] * int(rows)
    for u, row in enumerate(split_order(rows, rows_per_dp)):
        inv[row] = u
    return tuple(inv)


@lru_cache(maxsize=None)
def _anchor_rows(lanes: int, lanes_per_row: int) -> Tuple[int, ...]:
    return tuple(wide_rows(lane, lanes_per_row=lanes_per_row)[0] for lane in range(int(lanes)))


# ======================================================================================================================
# one step (host)
# ======================================================================================================================
def _int32(name: str, t: torch.Tensor, ndim: int) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    if t.dtype != torch.int32 or t.ndim != ndim:
        raise TypeError(f"{name} must be a {ndim}-D torch.int32 tensor, got {t.dtype} {tuple(t.shape)}")
    return t


@dataclass(frozen=True)
class KVWriteStep:
    """The KV writes of one decode step, in physical-lane order (host torch tensors).

    Attributes:
        positions: ``int32 [B]``: the KV write slot of each lane, ``-1`` = no write. Also FlashMLA's ``cur_pos``: an
            active lane attends to ``[0, position]`` (owners and plain lanes at their token ``n``, partners at ``n +
            1``).
        page_table: ``int32 [B, W]``: each lane's block ids (a partner carries its owner's row). Rows of inactive
            lanes are ignored (uploaded as zeros).
        call_b: ``bool [B]``: lanes written by call B (packed-verify partners). Everything else is call A.
        owner: ``int32 [B]``: the owner lane of each call-B lane, ``-1`` on every other lane.

    Construct with :meth:`inactive`, :meth:`ordinary`, :meth:`packed_verify` or :meth:`overflow_pass`; a T64 step
    (``2 x`` the lanes, module docstring "T64") with :meth:`wide_verify`. :func:`check_kv_write_step` validates a step
    against a mode (and :func:`check_wide_layout` a T64 step's layout).
    """

    positions: torch.Tensor
    page_table: torch.Tensor
    call_b: torch.Tensor
    owner: torch.Tensor

    def __post_init__(self):
        _int32("positions", self.positions, 1)
        _int32("page_table", self.page_table, 2)
        _int32("owner", self.owner, 1)
        if not isinstance(self.call_b, torch.Tensor) or self.call_b.dtype != torch.bool or self.call_b.ndim != 1:
            raise TypeError("call_b must be a 1-D torch.bool tensor")
        B = int(self.positions.shape[0])
        for name in ("call_b", "owner"):
            if int(getattr(self, name).shape[0]) != B:
                raise ValueError(f"{name} has {int(getattr(self, name).shape[0])} lanes, positions {B}")
        if int(self.page_table.shape[0]) != B:
            raise ValueError(f"page_table has {int(self.page_table.shape[0])} rows, positions {B} lanes")
        if int(self.page_table.shape[1]) < 1:
            raise ValueError("page_table needs at least one column")
        if bool((self.positions < -1).any()):
            raise ValueError("positions must be -1 (no write) or >= 0")

    # ---- constructors ------------------------------------------------------------------------------------------
    @classmethod
    def inactive(cls, width: int, lanes: int = NUM_LANES) -> "KVWriteStep":
        """Every lane inactive (warmup / capture steps: nothing is written)."""
        return cls.ordinary(
            torch.full((lanes,), -1, dtype=torch.int32), torch.zeros(lanes, int(width), dtype=torch.int32)
        )

    @classmethod
    def ordinary(cls, positions: torch.Tensor, page_table: torch.Tensor) -> "KVWriteStep":
        """An ordinary decode step: every active lane writes its own token at ``positions[l]`` (call A only)."""
        pos = torch.as_tensor(positions).to(torch.int32).reshape(-1)
        B = int(pos.shape[0])
        return cls(
            positions=pos.clone(),
            page_table=torch.as_tensor(page_table).to(torch.int32).clone(),
            call_b=torch.zeros(B, dtype=torch.bool),
            owner=torch.full((B,), -1, dtype=torch.int32),
        )

    @classmethod
    def packed_verify(
        cls, positions: torch.Tensor, page_table: torch.Tensor, partner_of: Mapping[int, int]
    ) -> "KVWriteStep":
        """A packed-verify step (design §3.8.2) from an OWNER-lane-order batch (``SpecDecodeBatch.positions`` /
        ``page_table``: anchors at ``n``, ``-1`` = idle) and ``partner_of = {owner lane: idle partner lane}`` (e.g.
        :func:`assign_partner_lanes`): each owner keeps its lane and anchor (call A); its draft is written at ``n + 1``
        on the partner lane, with the owner's page-table row (call B). Owners without a partner are not in this pass
        (:meth:`overflow_pass`)."""
        pos = torch.as_tensor(positions).to(torch.int32).reshape(-1).clone()
        pt = torch.as_tensor(page_table).to(torch.int32).clone()
        B = int(pos.shape[0])
        call_b = torch.zeros(B, dtype=torch.bool)
        owner = torch.full((B,), -1, dtype=torch.int32)
        pairs = sorted((int(o), int(d)) for o, d in partner_of.items())
        if pairs:
            anchors = pos.tolist()
            used = set()
            for o, d in pairs:
                if not (0 <= o < B and 0 <= d < B):
                    raise ValueError(f"partner_of entry {o} -> {d} outside [0, {B})")
                if anchors[o] < 0:
                    raise ValueError(f"owner lane {o} is inactive (position -1)")
                if anchors[d] >= 0 or d in used:
                    raise ValueError(f"partner lane {d} of owner {o} is not idle")
                used.add(d)
            o_t = torch.tensor([o for o, _ in pairs], dtype=torch.long)
            d_t = torch.tensor([d for _, d in pairs], dtype=torch.long)
            pos[d_t] = pos[o_t] + 1
            pt[d_t] = pt[o_t]
            call_b[d_t] = True
            owner[d_t] = o_t.to(torch.int32)
        return cls(positions=pos, page_table=pt, call_b=call_b, owner=owner)

    @classmethod
    def overflow_pass(cls, positions: torch.Tensor, page_table: torch.Tensor, lanes: Sequence[int]) -> "KVWriteStep":
        """The second replay of a verify step (design §3.8.2 item 4): each overflow draft on its OWN lane at ``n + 1``
        (the first pass wrote the anchor at ``n``), every other lane inactive. ``positions`` / ``page_table`` are the
        owner-lane-order batch (anchors at ``n``)."""
        anchors = torch.as_tensor(positions).to(torch.int32).reshape(-1)
        pos = torch.full_like(anchors, -1)
        for l in lanes:
            l = int(l)
            if int(anchors[l]) < 0:
                raise ValueError(f"overflow lane {l} is inactive")
            pos[l] = anchors[l] + 1
        return cls.ordinary(pos, page_table)

    @classmethod
    def wide_verify(
        cls,
        positions: torch.Tensor,
        page_table: torch.Tensor,
        has_draft: Optional[torch.Tensor] = None,
        *,
        lanes_per_row: int = LANES_PER_GROUP,
    ) -> "KVWriteStep":
        """A T64 step (design §4.1, §4.5; module docstring "T64") from an OWNER-lane-order batch: ``positions [B]``
        (anchors at ``n``, ``-1`` = idle), ``page_table [B, W]`` and ``has_draft [B]`` (bool; None = no drafts, the
        ``wide`` mode's ordinary step). Returns ``2 B`` physical rows, per DP row ``[anchors | drafts]``
        (:func:`wide_rows`). Each lane's anchor sits in its anchor row (call A). A drafted lane's draft is written at
        ``n + 1`` in its draft row, with the lane's page-table row (call B). Draft rows of idle or undrafted lanes are
        idle. This is :meth:`packed_verify` on the physical rows with ``partner_of = {16 r + j: 16 r + 8 + j}``.
        ``lanes_per_row``: owner lanes per DP row (8). Inputs are not modified."""
        pos = torch.as_tensor(positions).to(torch.int32).reshape(-1)
        pt = torch.as_tensor(page_table).to(torch.int32)
        B, n = int(pos.shape[0]), int(lanes_per_row)
        if n < 1 or B % n:
            raise ValueError(f"{B} lanes do not split into DP rows of {lanes_per_row}")
        if pt.ndim != 2 or int(pt.shape[0]) != B:
            raise ValueError(f"page_table must be [{B}, W], got {tuple(pt.shape)}")
        if has_draft is None:
            draft_l = [False] * B
        else:
            draft = torch.as_tensor(has_draft).reshape(-1).to(torch.bool)
            if int(draft.shape[0]) != B:
                raise ValueError(f"has_draft has {int(draft.shape[0])} lanes, positions {B}")
            draft_l = draft.tolist()
        pos_l = pos.tolist()
        bad = [lane for lane in range(B) if draft_l[lane] and pos_l[lane] < 0]
        if bad:
            raise ValueError(f"a draft on an inactive lane (has_draft where positions == -1): lanes {bad}")
        # = packed_verify(pos2, pt2, {anchor row: draft row}) on the physical rows, built from lists and one concat (it
        # runs every T64 step; tests/unit/test_kv_write.py checks the equality)
        anchor_l = _anchor_rows(B, n)
        pos2, cb2, ow2 = [-1] * (2 * B), [False] * (2 * B), [-1] * (2 * B)
        for lane in range(B):
            ra = anchor_l[lane]
            pos2[ra] = pos_l[lane]
            if draft_l[lane]:
                pos2[ra + n], cb2[ra + n], ow2[ra + n] = pos_l[lane] + 1, True, ra
        Wd = int(pt.shape[1])
        pt_rows = pt.reshape(B // n, n, Wd)  # per DP row: the anchors' rows, then the drafts' (zero without a draft)
        keep = torch.tensor(draft_l, dtype=torch.int32).reshape(B // n, n, 1)
        pt2 = torch.cat([pt_rows, pt_rows * keep], dim=1).reshape(2 * B, Wd)
        return cls(
            positions=torch.tensor(pos2, dtype=torch.int32),
            page_table=pt2,
            call_b=torch.tensor(cb2, dtype=torch.bool),
            owner=torch.tensor(ow2, dtype=torch.int32),
        )

    # ---- views -------------------------------------------------------------------------------------------------
    @property
    def lanes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def width(self) -> int:
        return int(self.page_table.shape[1])

    @property
    def active(self) -> torch.Tensor:
        """``bool [B]``: lanes that write (and are read by FlashMLA) this step."""
        return self.positions >= 0

    @property
    def num_partners(self) -> int:
        return int(self.call_b.sum())

    def call_positions(self, kind: str) -> torch.Tensor:
        """``int32 [B]`` update indices of call ``kind``: ``"A"`` (non-partners), ``"B"`` (partners), ``""`` (all)."""
        if kind == "":
            return self.positions.clone()
        if kind == "A":
            return torch.where(self.call_b, torch.full_like(self.positions, -1), self.positions)
        if kind == "B":
            return torch.where(self.call_b, self.positions, torch.full_like(self.positions, -1))
        raise ValueError(f"call kind must be 'A', 'B' or '', got {kind!r}")

    def write_slots(self, block_size: int) -> List[Tuple[int, int, int]]:
        """``[(lane, block id, row in block)]`` of every active lane (call the check first: ids must exist)."""
        out = []
        for l in torch.nonzero(self.active).reshape(-1).tolist():
            p = int(self.positions[l])
            out.append((l, int(self.page_table[l, p // block_size]), p % block_size))
        return out


def assign_partner_lanes(
    positions: torch.Tensor,
    has_draft: torch.Tensor,
    *,
    cross_row: bool,
    lanes_per_row: int = LANES_PER_GROUP,
) -> Tuple[Dict[int, int], Tuple[int, ...]]:
    """Reference packing of drafts into idle lanes (design §3.8.2): ``positions [B]`` in owner-lane order (``-1`` =
    idle), ``has_draft [B]`` (bool). Each drafted owner (ascending) gets the lowest idle lane on its own DP row; with
    ``cross_row`` (``allows_cross_row_partners(mode)``: KV-R) the owners left over then get the lowest idle lane of
    any row. Returns ``(partner_of {owner: partner}, overflow owners)``; the overflow drafts run in
    :meth:`KVWriteStep.overflow_pass`. Every idle lane is used at most once. This is the maximum matching under the
    row constraint (rows are independent without ``cross_row``)."""
    pos = torch.as_tensor(positions).reshape(-1)
    draft = torch.as_tensor(has_draft).reshape(-1).to(torch.bool)
    B = int(pos.shape[0])
    if int(draft.shape[0]) != B:
        raise ValueError(f"has_draft has {int(draft.shape[0])} lanes, positions {B}")
    if bool((draft & (pos < 0)).any()):
        raise ValueError("a draft on an inactive lane (has_draft where positions == -1)")
    pos_l, draft_l = pos.tolist(), draft.tolist()
    free = [l for l in range(B) if pos_l[l] < 0]
    owners = [l for l in range(B) if draft_l[l]]
    partner_of: Dict[int, int] = {}
    for o in owners:
        same = [l for l in free if l // lanes_per_row == o // lanes_per_row]
        if same:
            partner_of[o] = same[0]
            free.remove(same[0])
    if cross_row:
        for o in owners:
            if o not in partner_of and free:
                partner_of[o] = free.pop(0)
    overflow = tuple(o for o in owners if o not in partner_of)
    return partner_of, overflow


# ======================================================================================================================
# checks, calls and the host model
# ======================================================================================================================
@dataclass(frozen=True)
class KVWriteCall:
    """One ``paged_update_cache`` call of a step.

    Attributes:
        kind: ``"A"`` / ``"B"`` (split modes) or ``""`` (``row``, ``all``).
        rows: DP rows whose chips run the call, i.e. whose cache copies it writes (row modes: one row each; KV-R: all).
        lanes: the call's users in user (core) order, as global lane ids.
        positions: ``int32 [len(lanes)]``: update index per user, ``-1`` = skipped.
    """

    kind: str
    rows: Tuple[int, ...]
    lanes: Tuple[int, ...]
    positions: torch.Tensor


def _users_per_call(
    mode: str, lanes: int, lanes_per_row: int, lanes_per_call: Optional[int], split: bool = False
) -> int:
    """Users per update call of the host plan: ``lanes_per_row`` in the row modes, ``lanes_per_call`` (default all
    lanes; any divisor) in the KV-R modes; with the split-order gather (``split``) a divisor of ``lanes / 2`` (default
    ``lanes / 2``: the anchors and the drafts never share a call). The dtype cap is a device limit
    (:func:`lanes_per_call_for`)."""
    check_mode(mode)
    if lanes % lanes_per_row:
        raise ValueError(f"{lanes} lanes do not split into DP rows of {lanes_per_row}")
    if not is_replicated(mode):
        if lanes_per_call is not None and int(lanes_per_call) != lanes_per_row:
            raise ValueError(
                f"mode {mode!r} writes {lanes_per_row} lanes per call, got lanes_per_call={lanes_per_call}"
            )
        return lanes_per_row
    if split and lanes_per_row % 2:
        raise ValueError(f"the split-order gather needs an even number of rows per DP row, got {lanes_per_row}")
    users = lanes // 2 if split else lanes
    n = users if lanes_per_call is None else int(lanes_per_call)
    if n < 1 or users % n:
        raise ValueError(f"lanes_per_call must divide {users}, got {lanes_per_call}")
    return n


def check_wide_layout(step: KVWriteStep, *, lanes_per_row: int = WIDE_ROWS_PER_GROUP) -> None:
    """Raise ``ValueError`` unless ``step`` has the T64 row layout of :meth:`KVWriteStep.wide_verify` (module docstring
    "T64"). Each DP row of ``lanes_per_row`` rows (16) holds its anchors in the first half; they are never call-B
    rows. Each row of the second half (slot ``half + j``) is idle, or the call-B draft of the anchor in slot ``j`` of
    the same DP row.

    The split-order gather (anchor calls A, draft calls B, one shared page table) and the FlashMLA groups of
    :meth:`DecodeKVWrite.flash_groups` (the drafts read the anchors' ``[8, W]`` table) rely on this layout.
    :func:`check_kv_write_step` checks the rest: positions ``n + 1``, the owner's page-table row, races."""
    B, per = step.lanes, int(lanes_per_row)
    if per < 2 or per % 2 or B % per:
        raise ValueError(f"a T64 step needs DP rows of an even number of rows dividing {B}, got {lanes_per_row}")
    half = per // 2
    pos_l, cb_l, ow_l = step.positions.tolist(), step.call_b.tolist(), step.owner.tolist()
    for row in range(B):
        r, j = divmod(row, per)
        if j < half:
            if cb_l[row]:
                raise ValueError(
                    f"row {row} (DP row {r}, anchor slot {j}) is a call-B (draft) row: T64 drafts sit in slots "
                    f"{half}..{per - 1} of their DP row"
                )
        elif pos_l[row] >= 0 and not (cb_l[row] and ow_l[row] == row - half):
            raise ValueError(
                f"row {row} (DP row {r}, draft slot {j}) must be idle or the call-B draft of anchor row {row - half} "
                f"(the T64 layout of KVWriteStep.wide_verify); got position {pos_l[row]}, call B {bool(cb_l[row])}, "
                f"owner {ow_l[row]}"
            )


def kv_write_calls(
    step: KVWriteStep,
    mode: str,
    *,
    lanes_per_call: Optional[int] = None,
    lanes_per_row: int = LANES_PER_GROUP,
    gather: str = "natural",
) -> List[KVWriteCall]:
    """The update calls of ``step`` under ``mode``, in device order. Row modes: per DP row, call A (then B); its chips
    write that row's lanes. KV-R modes: per chunk of ``lanes_per_call`` lanes (default all), call A (then B), on every
    chip. KV-R with the split-order gather (``gather="split"``, a T64 step; :func:`gathers_split`): the anchor users
    (:func:`split_order`), per chunk of ``lanes_per_call`` (default all of them), each in one call A; then the draft
    users the same way in calls B. ``lanes`` of each call are physical row ids, in user (core) order."""
    B = step.lanes
    split_g = gathers_split(mode, gather)
    n = _users_per_call(mode, B, lanes_per_row, lanes_per_call, split=split_g)
    kinds = CALL_KINDS if is_split(mode) else ("",)
    pos = {k: step.call_positions(k) for k in kinds}
    dp = B // lanes_per_row
    calls = []
    if split_g:
        order, half = split_order(B, lanes_per_row), B // 2
        for kind, base in (("A", 0), ("B", half)):
            for c in range(half // n):
                lanes = order[base + c * n : base + (c + 1) * n]
                calls.append(KVWriteCall(kind, tuple(range(dp)), lanes, pos[kind][list(lanes)].clone()))
    elif is_replicated(mode):
        for c in range(B // n):
            lanes = tuple(range(c * n, (c + 1) * n))
            for k in kinds:
                calls.append(KVWriteCall(k, tuple(range(dp)), lanes, pos[k][c * n : (c + 1) * n].clone()))
    else:
        for r in range(dp):
            lanes = tuple(range(r * lanes_per_row, (r + 1) * lanes_per_row))
            for k in kinds:
                calls.append(KVWriteCall(k, (r,), lanes, pos[k][lanes[0] : lanes[-1] + 1].clone()))
    return calls


def check_kv_write_step(
    step: KVWriteStep,
    mode: str,
    *,
    block_size: int,
    max_seq_len: Optional[int] = None,
    lanes_per_call: Optional[int] = None,
    lanes_per_row: int = LANES_PER_GROUP,
    gather: str = "natural",
) -> None:
    """Raise ``ValueError`` unless ``step`` is a safe step for ``mode``:

    * every active position is ``< max_seq_len`` (when given) and its page-table entry ``p // block_size`` exists and
      holds a real block id (``>= 1``: block 0 is vLLM's null block);
    * call-B lanes only in split modes; each is active, has an active call-A owner, ``position = owner's + 1`` and the
      owner's page-table row; one partner per owner; ``owner`` is ``-1`` everywhere else; without
      :func:`allows_cross_row_partners` the partner sits on the owner's DP row;
    * no two lanes write the same ``(block, row)`` slot in the step;
    * within one update call (:func:`kv_write_calls`) no block id is written twice. Two rows of one request therefore
      never share a call, and no two users of a call read-modify-write the same tile (the race of gate G12).

    With the split-order gather (``gather="split"`` in a KV-R mode, a T64 step) the step must also have the T64 layout
    (:func:`check_wide_layout`): the anchor halves get calls A only and the draft halves calls B only.
    """
    check_mode(mode)
    bs = int(block_size)
    pos, pt, call_b, owner = step.positions, step.page_table, step.call_b, step.owner
    B, Wd = step.lanes, step.width
    if B % lanes_per_row:
        raise ValueError(f"{B} lanes do not split into DP rows of {lanes_per_row}")
    split_g = gathers_split(mode, gather)
    n = _users_per_call(mode, B, lanes_per_row, lanes_per_call, split=split_g)
    if split_g:
        check_wide_layout(step, lanes_per_row=lanes_per_row)
    user_l = _split_user_of(B, lanes_per_row) if split_g else None
    # 32-lane vectors as Python lists: a few us, where a torch op on them costs 3-5 us each (this runs every step)
    pos_l, cb_l, ow_l = pos.tolist(), call_b.tolist(), owner.tolist()
    act_l = [p >= 0 for p in pos_l]
    if max_seq_len is not None:
        for l, p in enumerate(pos_l):
            if p >= int(max_seq_len):
                raise ValueError(f"lane {l}: position {p} >= max_seq_len {max_seq_len}")
    entry_l = [p // bs if a else 0 for p, a in zip(pos_l, act_l)]
    for l, e in enumerate(entry_l):
        if e >= Wd:
            raise ValueError(f"lane {l}: position {pos_l[l]} needs page-table entry {e}, width is {Wd}")
    blocks_l = pt.gather(1, torch.tensor(entry_l, dtype=torch.long)[:, None])[:, 0].tolist()
    for l in range(B):
        if act_l[l] and blocks_l[l] < 1:
            raise ValueError(
                f"lane {l}: position {pos_l[l]} maps to block {blocks_l[l]} (page-table entry {entry_l[l]}); "
                f"decode writes need a real block id >= 1 (0 is vLLM's null block)"
            )
    # ---- partners ----------------------------------------------------------------------------------------------
    for l in range(B):
        if ow_l[l] != -1 and not cb_l[l]:
            raise ValueError(f"lane {l} names owner {ow_l[l]} but is not a call-B (partner) lane")
    partners = [d for d in range(B) if cb_l[d]]
    if partners:
        if not is_split(mode):
            raise ValueError(
                f"mode {mode!r} has no call B: packed-verify partners need row_split / all_split (one call would race "
                f"owner and partner on the shared tile, gate G12)"
            )
        seen = set()
        cross = allows_cross_row_partners(mode)
        for d in partners:
            o = ow_l[d]
            ok = act_l[d] and 0 <= o < B and o != d and not cb_l[o] and act_l[o] and o not in seen
            ok = ok and pos_l[d] == pos_l[o] + 1 and (cross or d // lanes_per_row == o // lanes_per_row)
            if not ok:
                _raise_partner_error(step, mode, act_l, lanes_per_row)
            seen.add(o)
        # every partner carries its owner's page-table row: one comparison for all of them (a T64 step has up to 32)
        if not torch.equal(pt[partners], pt[[ow_l[d] for d in partners]]):
            _raise_partner_error(step, mode, act_l, lanes_per_row)
    # ---- slots and calls ---------------------------------------------------------------------------------------
    split = is_split(mode)
    slots, calls = {}, {}
    for l in range(B):
        if not act_l[l]:
            continue
        s_key = (blocks_l[l], pos_l[l] % bs)
        if s_key in slots:
            raise ValueError(f"lanes {slots[s_key]} and {l} both write block {s_key[0]} row {s_key[1]} in one step")
        slots[s_key] = l
        # the lane's update call (DP row in the row modes, lane chunk in the KV-R modes; call A / B; with the split
        # gather the chunk of its gathered user, whose half names the kind): one block once
        if user_l is not None:
            c_key = (user_l[l] // n, user_l[l] >= B // 2, blocks_l[l])
        else:
            c_key = (l // n, bool(cb_l[l]) and split, blocks_l[l])
        if c_key in calls:
            kind = ("B" if c_key[1] else "A") if split else "(single)"
            rows = (l // lanes_per_row,) if not is_replicated(mode) else tuple(range(B // lanes_per_row))
            raise ValueError(
                f"update call {kind} on rows {rows} writes block {blocks_l[l]} for lanes {[calls[c_key], l]}: one call "
                f"must not carry two rows of one block (tile read-modify-write race, gate G12)"
            )
        calls[c_key] = l


def _raise_partner_error(step: KVWriteStep, mode: str, act: Sequence[bool], lanes_per_row: int) -> None:
    """The per-lane partner checks of :func:`check_kv_write_step`: raise ``ValueError`` naming the first violation."""
    pos, pt, call_b, owner = step.positions, step.page_table, step.call_b, step.owner
    B = step.lanes
    seen = set()
    for d in torch.nonzero(call_b).reshape(-1).tolist():
        o = int(owner[d])
        if not bool(act[d]):
            raise ValueError(f"partner lane {d} is inactive")
        if not 0 <= o < B or o == d:
            raise ValueError(f"partner lane {d} has owner {o}")
        if bool(call_b[o]) or not bool(act[o]):
            raise ValueError(f"partner lane {d}: owner {o} must be an active call-A lane")
        if o in seen:
            raise ValueError(f"owner {o} has two partners (one draft per request, K = 1)")
        seen.add(o)
        if int(pos[d]) != int(pos[o]) + 1:
            raise ValueError(f"partner lane {d} writes {int(pos[d])}, its owner {o} {int(pos[o])}: must be + 1")
        if not torch.equal(pt[d], pt[o]):
            raise ValueError(f"partner lane {d} must carry its owner {o}'s page-table row")
        if not allows_cross_row_partners(mode) and d // lanes_per_row != o // lanes_per_row:
            raise ValueError(
                f"partner lane {d} (DP row {d // lanes_per_row}) of owner {o} (row {o // lanes_per_row}): mode "
                f"{mode!r} keeps a lane's KV on its own DP row, so a partner must sit on its owner's row (only "
                f"all_split, KV-R, allows cross-row partners)"
            )
    raise AssertionError("kv_write: the partner check failed but no lane violates a rule")  # pragma: no cover


def apply_kv_writes_host(
    caches: Union[torch.Tensor, Sequence[torch.Tensor]],
    step: KVWriteStep,
    mode: str,
    rows: torch.Tensor,
    *,
    block_size: int,
    lanes_per_call: Optional[int] = None,
    lanes_per_row: int = LANES_PER_GROUP,
    gather: str = "natural",
) -> List[torch.Tensor]:
    """Host model of one layer's write: ``caches`` = the cache copy ``[N, 1, bs, D]`` of each DP row (one tensor =
    every row starts from it), ``rows [B, D]`` = each lane's latent row (``kv_row``, already representable in the
    cache dtype for an exact comparison; a T64 step: per physical row, ``[anchors | drafts]`` per DP row). Returns the
    new copy of every DP row (inputs untouched): what every chip of that row holds after :meth:`DecodeKVWrite.write`.
    ``gather``: the KV-R gather order of the calls (:func:`kv_write_calls`)."""
    B = step.lanes
    dp = B // lanes_per_row
    if isinstance(caches, torch.Tensor):
        out = [caches.clone() for _ in range(dp)]
    else:
        if len(caches) != dp:
            raise ValueError(f"need one cache copy per DP row ({dp}), got {len(caches)}")
        out = [c.clone() for c in caches]
    if tuple(rows.shape[:1]) != (B,):
        raise ValueError(f"rows must be [{B}, D], got {tuple(rows.shape)}")
    bs = int(block_size)
    calls = kv_write_calls(step, mode, lanes_per_call=lanes_per_call, lanes_per_row=lanes_per_row, gather=gather)
    for call in calls:
        for l, p in zip(call.lanes, call.positions.tolist()):
            if p < 0:
                continue
            b = int(step.page_table[l, p // bs])
            for r in call.rows:
                out[r][b, 0, p % bs] = rows[l].to(out[r].dtype)
    return out


def kv_write_inputs_host(
    step: KVWriteStep,
    mode: str,
    *,
    lanes_per_call: Optional[int] = None,
    lanes_per_row: int = LANES_PER_GROUP,
    gather: str = "natural",
    wide: bool = False,
) -> Dict[str, Tuple[Placement, torch.Tensor]]:
    """Values of every persistent device input of :class:`DecodeKVWrite` for ``step``: ``{name: (placement,
    int32 tensor)}``. ``"row"`` tensors are lane ordered (``[B]`` / ``[B, W]``; DP row ``r`` receives lanes ``8 r ..
    8 r + 7``, or rows ``16 r .. 16 r + 15`` of a T64 step), ``"rep"`` tensors are replicated on every chip.

    * always ``cur_pos`` (``positions``) and ``page_table`` (rows of inactive lanes zeroed), per row: FlashMLA and the
      ``row`` update;
    * ``row_split``: ``cur_a`` / ``cur_b`` (per row);
    * KV-R, per chunk ``c`` of ``lanes_per_call`` lanes: ``cur{c}`` (``all``) or ``cur_a{c}`` / ``cur_b{c}``
      (``all_split``), and ``pt{c}`` (``[n, W]``), replicated;
    * KV-R with the split-order gather (a T64 step that passed :func:`check_kv_write_step` with ``gather="split"``),
      per chunk ``c`` of ``lanes_per_call`` lanes (owner lane order): ``cur_a{c}`` = the anchors' call-A positions,
      ``cur_b{c}`` = the drafts' call-B positions, and ONE ``pt{c}`` = the anchors' rows, shared by both calls (a draft
      carries its owner's row), replicated;
    * ``wide`` (a T64 step, ``lanes_per_row`` = 16): the FlashMLA groups of option A'' per row, in owner lane order:
      ``flash_cur_a`` (the anchors' positions), ``flash_cur_d`` (the drafts' positions, ``-1`` = none) and
      ``flash_pt`` (the anchors' page-table rows, shared by both groups).
    """
    B = step.lanes
    split_g = gathers_split(mode, gather)
    n = _users_per_call(mode, B, lanes_per_row, lanes_per_call, split=split_g)
    act = step.active
    pos = step.positions.to(torch.int32)
    pt = torch.where(act[:, None], step.page_table, torch.zeros_like(step.page_table)).to(torch.int32)
    out: Dict[str, Tuple[Placement, torch.Tensor]] = {"cur_pos": ("row", pos.clone()), "page_table": ("row", pt)}
    if mode == "row_split":
        out["cur_a"] = ("row", step.call_positions("A"))
        out["cur_b"] = ("row", step.call_positions("B"))
    elif split_g:
        order, half = split_order(B, lanes_per_row), B // 2
        ca, cb = step.call_positions("A"), step.call_positions("B")
        for c in range(half // n):
            ua = list(order[c * n : (c + 1) * n])
            ub = list(order[half + c * n : half + (c + 1) * n])
            out[f"cur_a{c}"] = ("rep", ca[ua].clone())
            out[f"cur_b{c}"] = ("rep", cb[ub].clone())
            out[f"pt{c}"] = ("rep", pt[ua].clone())
    elif is_replicated(mode):
        for c in range(B // n):
            sl = slice(c * n, (c + 1) * n)
            if mode == "all":
                out[f"cur{c}"] = ("rep", pos[sl].clone())
            else:
                out[f"cur_a{c}"] = ("rep", step.call_positions("A")[sl].clone())
                out[f"cur_b{c}"] = ("rep", step.call_positions("B")[sl].clone())
            out[f"pt{c}"] = ("rep", pt[sl].clone())
    if wide:
        if lanes_per_row % 2:
            raise ValueError(f"a T64 step needs an even number of rows per DP row, got {lanes_per_row}")
        half = lanes_per_row // 2
        a_rows = [r * lanes_per_row + j for r in range(B // lanes_per_row) for j in range(half)]
        d_rows = [x + half for x in a_rows]
        out["flash_cur_a"] = ("row", pos[a_rows].clone())
        out["flash_cur_d"] = ("row", pos[d_rows].clone())
        out["flash_pt"] = ("row", pt[a_rows].clone())
    return out


# ======================================================================================================================
# device
# ======================================================================================================================
def update_input_memory_config(users: int, grid) -> "ttnn.MemoryConfig":
    """HEIGHT_SHARDED L1 config of the update input ``[1, users, 1(32), 576]``: one user's padded tile row per core on
    the first ``users`` cores (row-wise). For ``users = 8`` it is exactly ``MotifAttention.update_mc``."""
    cores = ttnn.num_cores_to_corerangeset(int(users), grid, row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(TILE, KV_LATENT_DIM),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class DecodeKVWrite:
    """The decode KV write shared by every decode layer of one server (module docstring).

    Args:
        mesh_device: the opened mesh.
        cfg: ``MotifTTConfig`` of the mesh (lanes, DP rows, block size, KV dtype, ``kv_write_mode``).
        ccl: the model's ``MotifCCL`` (``ag_dp_rows`` for KV-R, ``ag_dp_rows(x, halves=2)`` for the T64 split order;
            semaphores in L1_SMALL).
        page_table_width: ``W`` of the decode trace (fixed for its life, like the generator's decode inputs).
        mode: one of :data:`generator_api.KV_WRITE_MODES`; default ``cfg.kv_write_mode``.
        lanes_per_call: KV-R users per update call (default :func:`lanes_per_call_for`: 32 for bfp8, 16 for bf16).
        rows: decode rows of the step: ``cfg.max_batch`` (32, default: the T32 step, 8 rows per DP row) or ``2 *
            cfg.max_batch`` (64: the T64 verify step, 16 rows per DP row ``[8 anchors | 8 drafts]``; module docstring
            "T64").
        gather: the KV-R gather order of a T64 step (:data:`GATHER_ORDERS`): ``"split"`` (default with ``rows=64``
            in ``all_split``) or ``"natural"``. The row modes have no gather (each DP row writes its own rows) and
            ignore it; the T32 step has only ``"natural"``.

    Persistent device inputs (int32 ROW_MAJOR DRAM; rewritten by :meth:`write_step`): ``cur_pos [L]`` and
    ``page_table [L, W]`` per DP row (``L`` = :attr:`lanes_per_row`: 8, or 16 at T64; FlashMLA's inputs, also the
    ``row`` update's), the mode's call inputs (:func:`kv_write_inputs_host`), and at T64 the FlashMLA group inputs
    ``flash_cur_a [8]`` / ``flash_cur_d [8]`` / ``flash_pt [8, W]`` per DP row (:meth:`flash_groups`). Never write
    them directly: :meth:`write_step` skips inputs whose values did not change since its last write.
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        *,
        ccl,
        page_table_width: int,
        mode: Optional[str] = None,
        lanes_per_call: Optional[int] = None,
        rows: Optional[int] = None,
        gather: Optional[str] = None,
    ):
        self.mode = check_mode(cfg.kv_write_mode if mode is None else mode)
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.ccl = ccl
        lanes = int(cfg.max_batch)
        self.lanes = lanes if rows is None else int(rows)
        if self.lanes not in (lanes, 2 * lanes):
            raise ValueError(
                f"rows must be {lanes} (the T32 step) or {2 * lanes} (the T64 step: per DP row 8 anchors + 8 drafts), "
                f"got {rows}"
            )
        self.wide = self.lanes == 2 * lanes
        self.lanes_per_row = int(cfg.lanes_per_row) * (2 if self.wide else 1)
        self.block_size = int(cfg.kv_block_size)
        self.width = int(page_table_width)
        if self.width < 1:
            raise ValueError(f"page_table_width must be >= 1, got {page_table_width}")
        if gather is None:
            gather = "split" if (self.wide and self.replicated and self.split) else "natural"
        check_gather(gather)
        if gather == "split" and not self.wide:
            raise ValueError("the split-order gather orders a T64 step's [anchors | drafts] halves: it needs rows=64")
        self.gather = gather if self.replicated else "natural"  # the row modes have no gather
        gathers_split(self.mode, self.gather)  # "all" + "split" raises
        self.kv_dtype_name = cfg.dtypes.kv_cache_name
        self.lanes_per_call = lanes_per_call_for(
            self.mode,
            self.kv_dtype_name,
            lanes=self.lanes,
            lanes_per_row=self.lanes_per_row,
            requested=lanes_per_call,
            gather=self.gather,
        )
        grid = mesh_device.compute_with_storage_grid_size()
        self.row_mc = update_input_memory_config(self.lanes_per_row, grid)
        self.call_mc = update_input_memory_config(self.lanes_per_call, grid) if self.replicated else None
        self._kinds = CALL_KINDS if self.split else ("",)
        self._dev: Dict[str, Any] = {}
        self._last: Dict[str, torch.Tensor] = {}
        step = KVWriteStep.inactive(self.width, self.lanes)
        for name, (place, val) in self._values(step).items():
            self._dev[name] = ttnn.to_device(self._host(place, val), mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self._last[name] = val
        self.step = step
        self._bind_inputs()

    def _bind_inputs(self) -> None:
        """Name the persistent inputs: FlashMLA's ``cur_pos`` / ``page_table`` and its groups (:meth:`flash_groups`),
        and the update-call plan: per chunk of gathered users ``(lo, hi, [(update index tensor, page table tensor),
        ...])`` in device order (call A, then call B; the split order has the anchor chunks' calls A, then the draft
        chunks' calls B). The row modes have one entry (this DP row's rows)."""
        d = self._dev
        self.cur_pos = d["cur_pos"]
        self.page_table = d["page_table"]
        L, n = self.lanes_per_row, self.lanes_per_call
        if self.mode == "row":
            plan = [(0, L, [(self.cur_pos, self.page_table)])]
        elif self.mode == "row_split":
            plan = [(0, L, [(d["cur_a"], self.page_table), (d["cur_b"], self.page_table)])]
        elif self.gather == "split":  # the anchor chunks (calls A), then the draft chunks (calls B), one pt{c} each
            half = self.lanes // 2
            plan = [
                (base + c * n, base + (c + 1) * n, [(d[f"{nm}{c}"], d[f"pt{c}"])])
                for base, nm in ((0, "cur_a"), (half, "cur_b"))
                for c in range(self.num_chunks)
            ]
        else:
            names = ("cur",) if self.mode == "all" else ("cur_a", "cur_b")
            plan = [
                (c * n, (c + 1) * n, [(d[f"{nm}{c}"], d[f"pt{c}"]) for nm in names]) for c in range(self.num_chunks)
            ]
        self._plan = plan
        if self.wide:  # option A'': anchors at n, drafts at n + 1, both on the anchors' [8, W] rows
            h = L // 2
            self._flash = [
                (slice(0, h), d["flash_cur_a"], d["flash_pt"]),
                (slice(h, L), d["flash_cur_d"], d["flash_pt"]),
            ]
        else:
            self._flash = [(slice(0, L), self.cur_pos, self.page_table)]

    # ---- properties ------------------------------------------------------------------------------------------
    @property
    def split(self) -> bool:
        return is_split(self.mode)

    @property
    def replicated(self) -> bool:
        return is_replicated(self.mode)

    @property
    def cross_row_partners(self) -> bool:
        """Packed-verify partners may sit on another DP row than their owner (:func:`allows_cross_row_partners`)."""
        return allows_cross_row_partners(self.mode)

    @property
    def num_chunks(self) -> int:
        """KV-R lane chunks (one per ``lanes_per_call`` lanes, each with its own ``pt{c}``); with the split-order
        gather the chunks of the anchors (the drafts reuse their ``pt{c}``); 1 in the row modes (per-row tensors)."""
        if not self.replicated:
            return 1
        return (self.lanes // 2 if self.gather == "split" else self.lanes) // self.lanes_per_call

    @property
    def calls_per_layer(self) -> int:
        """``paged_update_cache`` calls per layer (per chip)."""
        if self.gather == "split":
            return 2 * self.num_chunks
        return self.num_chunks * len(self._kinds)

    @property
    def device_inputs(self) -> Dict[str, Any]:
        """The persistent device inputs by name (read-only use: tests, diagnostics)."""
        return dict(self._dev)

    def flash_groups(self) -> List[Tuple[slice, Any, Any]]:
        """FlashMLA's inputs per group of rows of this chip's DP row, ``[(row slice, cur_pos, page_table)]`` (module
        docstring "FlashMLA groups", design §4.3 option A''): one group ``(0:8, cur_pos, page_table)`` at 8 rows per
        DP row (the T32 call, unchanged); two B = 8 groups at 16 rows: the anchors ``(0:8, flash_cur_a, flash_pt)`` at
        ``n`` and the drafts ``(8:16, flash_cur_d, flash_pt)`` at ``n + 1`` (they share the anchors' rows). The global
        layers run one FlashMLA call per group on ``q[:, rows]`` and concatenate the outputs on dim 1, so every T64 row
        equals its T32 row bit for bit; the SWA layers keep one call on :attr:`cur_pos` / :attr:`page_table`. The
        tensors are persistent and rewritten by :meth:`write_step` (trace-safe)."""
        return list(self._flash)

    # ---- per step (host) ---------------------------------------------------------------------------------------
    def _values(self, step: KVWriteStep) -> Dict[str, Tuple[Placement, torch.Tensor]]:
        return kv_write_inputs_host(
            step,
            self.mode,
            lanes_per_call=self.lanes_per_call if self.replicated else None,
            lanes_per_row=self.lanes_per_row,
            gather=self.gather,
            wide=self.wide,
        )

    def _host(self, place: Placement, val: torch.Tensor):
        if place == "row":
            return shard_lanes(val.contiguous(), self.cfg, self.mesh_device, dtype=ttnn.int32, device=None)
        return ttnn.from_torch(
            val.contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def check(self, step: KVWriteStep) -> None:
        """:func:`check_kv_write_step` with this object's mode and geometry, and at T64 :func:`check_wide_layout`
        (raises ``ValueError``)."""
        if step.lanes != self.lanes or step.width != self.width:
            raise ValueError(
                f"step has {step.lanes} lanes x width {step.width}; this kv_write was built for {self.lanes} x "
                f"{self.width} (the decode trace's page-table width)"
            )
        if self.wide and self.gather != "split":  # the split gather's check includes it
            check_wide_layout(step, lanes_per_row=self.lanes_per_row)
        check_kv_write_step(
            step,
            self.mode,
            block_size=self.block_size,
            max_seq_len=self.cfg.max_model_len,
            lanes_per_call=self.lanes_per_call if self.replicated else None,
            lanes_per_row=self.lanes_per_row,
            gather=self.gather,
        )

    def host_inputs(self, step: KVWriteStep) -> Dict[str, Any]:
        """``{name: host mesh tensor}`` for ``ttnn.copy_host_to_device_tensor`` into :attr:`device_inputs`."""
        return {name: self._host(place, val) for name, (place, val) in self._values(step).items()}

    def write_step(self, step: KVWriteStep, *, validate: bool = True, force: bool = False) -> int:
        """Write ``step`` into the persistent device inputs (before the eager step or the trace replay that uses it).
        ``validate`` runs :meth:`check` first. Inputs whose values equal the last written ones are skipped unless
        ``force``. Returns the number of inputs copied."""
        if validate:
            self.check(step)
        elif step.lanes != self.lanes or step.width != self.width:
            raise ValueError(f"step shape {step.lanes} x {step.width} != {self.lanes} x {self.width}")
        n = 0
        for name, (place, val) in self._values(step).items():
            if not force and torch.equal(self._last[name], val):
                continue
            ttnn.copy_host_to_device_tensor(self._host(place, val), self._dev[name])
            self._last[name] = val
            n += 1
        self.step = step
        return n

    def check_flash_inputs(self, cur_pos, page_table) -> None:
        """Raise ``ValueError`` unless FlashMLA's ``cur_pos`` / ``page_table`` are this object's tensors (the write and
        the read must describe the same lanes; Python identity check, trace-safe)."""
        if cur_pos is not self.cur_pos or page_table is not self.page_table:
            raise ValueError(
                "with kv_write, FlashMLA must read kv_write.cur_pos / kv_write.page_table (pass them as cur_pos= / "
                "page_table=): the KV write and the attention read must describe the same lanes"
            )

    # ---- per layer (device, trace-safe) --------------------------------------------------------------------------
    def write(self, kv_row, kv_cache, *, cur_pos, page_table) -> None:
        """Write this step's latent rows of one layer into ``kv_cache``: the ``tt.attention.DecodeKVWriter`` hook,
        called by ``MotifAttention.forward_decode(..., kv_write=self)`` once per layer, before FlashMLA (trace-safe;
        module docstring).

        Args:
            kv_row: ``[1, 1, L, 576]`` bf16 TILE DRAM (this DP row's rows, replicated in the row; ``L`` =
                :attr:`lanes_per_row`, 8, or 16 at T64 = ``[8 anchors | 8 drafts]``): the attention's ``concat(n,
                rope(k_pe))``. Not consumed (the caller frees it).
            kv_cache: the layer's paged latent cache ``[N, 1, block, 576]`` (TILE, DRAM; any decode layer, the MTP
                layer included). Updated in place.
            cur_pos / page_table: FlashMLA's per-row inputs of this layer. They must be :attr:`cur_pos` /
                :attr:`page_table` (:meth:`check_flash_inputs`): the write and the read describe the same lanes.
        """
        self.check_flash_inputs(cur_pos, page_table)
        if not self.replicated:
            u = ttnn.transpose(kv_row, 1, 2, memory_config=self.row_mc)  # [1, L, 1, 576], one row per core
            for cur, pt in self._plan[0][2]:
                ttnn.experimental.paged_update_cache(kv_cache, u, update_idxs_tensor=cur, page_table=pt)
            ttnn.deallocate(u)
            return
        if self.gather == "split":  # [1, 1, 64, 576]: rows 0..31 the anchors, 32..63 the drafts (lane order)
            g = self.ccl.ag_dp_rows(kv_row, halves=2)
        else:  # [1, 1, rows, 576] TILE DRAM, row order L dp + l, identical on every chip
            g = self.ccl.ag_dp_rows(kv_row)
        if len(self._plan) == 1:  # T32, 32 users: one transpose straight into the sharded layout
            u = ttnn.transpose(g, 1, 2, memory_config=self.call_mc)  # [1, 32, 1, 576], one lane per core
            ttnn.deallocate(g)
            for cur, pt in self._plan[0][2]:
                ttnn.experimental.paged_update_cache(kv_cache, u, update_idxs_tensor=cur, page_table=pt)
            ttnn.deallocate(u)
            return
        if self.lanes_per_call == TILE:  # T64, 32 users per call: each chunk is one tile row of g
            for lo, hi, calls in self._plan:
                s = ttnn.slice(g, [0, 0, lo, 0], [1, 1, hi, KV_LATENT_DIM])
                u = ttnn.transpose(s, 1, 2, memory_config=self.call_mc)  # [1, 32, 1, 576], one user per core
                ttnn.deallocate(s)
                for cur, pt in calls:
                    ttnn.experimental.paged_update_cache(kv_cache, u, update_idxs_tensor=cur, page_table=pt)
                ttnn.deallocate(u)
            ttnn.deallocate(g)
            return
        t = ttnn.transpose(g, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, rows, 1, 576]
        ttnn.deallocate(g)
        for lo, hi, calls in self._plan:
            s = ttnn.slice(t, [0, lo, 0, 0], [1, hi, 1, KV_LATENT_DIM])
            u = ttnn.to_memory_config(s, self.call_mc)  # [1, n, 1, 576], one lane per core
            ttnn.deallocate(s)
            for cur, pt in calls:
                ttnn.experimental.paged_update_cache(kv_cache, u, update_idxs_tensor=cur, page_table=pt)
            ttnn.deallocate(u)
        ttnn.deallocate(t)

    def end_step(self) -> None:
        """Call once per decode step after the last layer's :meth:`write` (inside the traced step). A no-op for the
        current modes; the deferred KV-R variant (design §3.12.3) would issue its batched remote writes here."""
        return None

    def deallocate(self) -> None:
        """Free every persistent device input (never while a trace that reads them is alive)."""
        for t in self._dev.values():
            ttnn.deallocate(t)
        self._dev = {}
        self._plan = []
        self._flash = []
        self.cur_pos = self.page_table = None


# ======================================================================================================================
# readback helpers (tests / diagnostics only: they synchronize and copy to host; never inside a trace)
# ======================================================================================================================
def cache_copies(kv_cache, mesh_device, cfg: MotifTTConfig) -> Dict[Tuple[int, int], torch.Tensor]:
    """``{(dp, tp): host copy of kv_cache on that chip}`` (float32)."""
    shards = ttnn.get_device_tensors(kv_cache)
    C = int(cfg.axes.mesh_shape[1])
    out = {}
    for dp in range(cfg.dp):
        for tp in range(cfg.tp):
            r, c = cfg.axes.coord(dp, tp)
            out[(dp, tp)] = ttnn.to_torch(shards[r * C + c]).float()
    return out


def cache_mismatches(
    kv_cache, mesh_device, cfg: MotifTTConfig, expected: Sequence[torch.Tensor]
) -> Dict[Tuple[int, int], int]:
    """``{(dp, tp): number of differing elements}`` of every chip whose cache copy differs from ``expected[dp]``
    (:func:`apply_kv_writes_host`); empty = every chip holds exactly its DP row's expected copy."""
    bad = {}
    for (dp, tp), got in cache_copies(kv_cache, mesh_device, cfg).items():
        want = expected[dp]
        if not torch.equal(got, want.float()):
            bad[(dp, tp)] = int((got != want.float()).sum())
    return bad


__all__ = [
    "CALL_KINDS",
    "DecodeKVWrite",
    "GATHER_ORDERS",
    "KVWriteCall",
    "KVWriteStep",
    "MAX_LANES_PER_CALL",
    "REPLICATED_MODES",
    "SPLIT_MODES",
    "allows_cross_row_partners",
    "apply_kv_writes_host",
    "assign_partner_lanes",
    "cache_copies",
    "cache_mismatches",
    "check_gather",
    "check_kv_write_step",
    "check_mode",
    "check_wide_layout",
    "gathers_split",
    "is_replicated",
    "is_split",
    "kv_write_calls",
    "kv_write_inputs_host",
    "lanes_per_call_for",
    "split_order",
    "update_input_memory_config",
    "wide_rows",
]
