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

**Host model.** :func:`kv_write_calls` lists the update calls of a step (which chips, which users, which positions),
:func:`apply_kv_writes_host` applies them to per-DP-row torch cache copies (what every chip of that row must hold
afterwards), and :func:`kv_write_inputs_host` gives the values of every persistent device input. The device tests
compare every chip's cache with that model bit-exactly (``tests/unit/test_kv_write.py``).

Import rule: ``torch``, ``ttnn`` and the motif3 shared infra (``generator_api``, ``model_config``, ``rope``). The host
planning functions use torch only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

import ttnn

from .generator_api import KV_LATENT_DIM, KV_WRITE_MODES, LANES_PER_GROUP, NUM_LANES
from .model_config import TILE, MotifTTConfig
from .rope import shard_lanes

SPLIT_MODES = ("row_split", "all_split")  # call A (owners + plain lanes), then call B (packed-verify partners)
REPLICATED_MODES = ("all", "all_split")  # KV-R: every decode KV write lands on all chips
# Users per paged_update_cache call in the KV-R modes, per KV-cache dtype (gate G12, module docstring "bf16 KV caches").
MAX_LANES_PER_CALL = {"bfp8": 32, "bf16": 16}
CALL_KINDS = ("A", "B")  # split modes; the non-split modes have one call kind ""
Placement = str  # "row": dim 0 sharded over the DP rows (8 lanes each), replicated over TP; "rep": replicated


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


def lanes_per_call_for(
    mode: str,
    kv_dtype_name: str = "bfp8",
    *,
    lanes: int = NUM_LANES,
    lanes_per_row: int = LANES_PER_GROUP,
    requested: Optional[int] = None,
) -> int:
    """Users per ``paged_update_cache`` call. Row modes: ``lanes_per_row`` (each DP row's chips write their own lanes;
    ``requested`` must be None or equal). KV-R modes: ``requested`` (a divisor of ``lanes`` up to the dtype maximum),
    default ``min(lanes, MAX_LANES_PER_CALL[kv_dtype_name])``: 32 for bfp8, 16 for bf16."""
    if kv_dtype_name not in MAX_LANES_PER_CALL:
        raise ValueError(f"KV cache dtype must be one of {tuple(MAX_LANES_PER_CALL)}, got {kv_dtype_name!r}")
    if not is_replicated(mode):
        return _users_per_call(mode, lanes, lanes_per_row, requested)
    if lanes % lanes_per_row:
        raise ValueError(f"{lanes} lanes do not split into DP rows of {lanes_per_row}")
    cap = min(lanes, MAX_LANES_PER_CALL[kv_dtype_name])
    n = cap if requested is None else int(requested)
    if n < 1 or lanes % n or n > cap:
        raise ValueError(
            f"lanes_per_call must divide {lanes} and be <= {cap} for a {kv_dtype_name} cache (the update op's output "
            f"CB is users x 18 tiles per core), got {requested}"
        )
    return n


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

    Construct with :meth:`inactive`, :meth:`ordinary`, :meth:`packed_verify` or :meth:`overflow_pass`.
    :func:`check_kv_write_step` validates a step against a mode.
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


def _users_per_call(mode: str, lanes: int, lanes_per_row: int, lanes_per_call: Optional[int]) -> int:
    """Users per update call of the host plan: ``lanes_per_row`` in the row modes, ``lanes_per_call`` (default all
    lanes; any divisor) in the KV-R modes. The dtype cap is a device limit (:func:`lanes_per_call_for`)."""
    check_mode(mode)
    if lanes % lanes_per_row:
        raise ValueError(f"{lanes} lanes do not split into DP rows of {lanes_per_row}")
    if not is_replicated(mode):
        if lanes_per_call is not None and int(lanes_per_call) != lanes_per_row:
            raise ValueError(
                f"mode {mode!r} writes {lanes_per_row} lanes per call, got lanes_per_call={lanes_per_call}"
            )
        return lanes_per_row
    n = lanes if lanes_per_call is None else int(lanes_per_call)
    if n < 1 or lanes % n:
        raise ValueError(f"lanes_per_call must divide {lanes}, got {lanes_per_call}")
    return n


def kv_write_calls(
    step: KVWriteStep,
    mode: str,
    *,
    lanes_per_call: Optional[int] = None,
    lanes_per_row: int = LANES_PER_GROUP,
) -> List[KVWriteCall]:
    """The update calls of ``step`` under ``mode``, in device order. Row modes: per DP row, call A (then B); its chips
    write that row's lanes. KV-R modes: per chunk of ``lanes_per_call`` lanes (default all), call A (then B), on every
    chip."""
    B = step.lanes
    n = _users_per_call(mode, B, lanes_per_row, lanes_per_call)
    kinds = CALL_KINDS if is_split(mode) else ("",)
    pos = {k: step.call_positions(k) for k in kinds}
    dp = B // lanes_per_row
    calls = []
    if is_replicated(mode):
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
    """
    check_mode(mode)
    bs = int(block_size)
    pos, pt, call_b, owner = step.positions, step.page_table, step.call_b, step.owner
    B, Wd = step.lanes, step.width
    if B % lanes_per_row:
        raise ValueError(f"{B} lanes do not split into DP rows of {lanes_per_row}")
    n = _users_per_call(mode, B, lanes_per_row, lanes_per_call)
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
            if not ok or not torch.equal(pt[d], pt[o]):
                _raise_partner_error(step, mode, act_l, lanes_per_row)
            seen.add(o)
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
        # the lane's update call (DP row in the row modes, lane chunk in the KV-R modes; call A / B): one block once
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
) -> List[torch.Tensor]:
    """Host model of one layer's write: ``caches`` = the cache copy ``[N, 1, bs, D]`` of each DP row (one tensor =
    every row starts from it), ``rows [B, D]`` = each lane's latent row (``kv_row``, already representable in the
    cache dtype for an exact comparison). Returns the new copy of every DP row (inputs untouched): what every chip of
    that row holds after :meth:`DecodeKVWrite.write`."""
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
    for call in kv_write_calls(step, mode, lanes_per_call=lanes_per_call, lanes_per_row=lanes_per_row):
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
) -> Dict[str, Tuple[Placement, torch.Tensor]]:
    """Values of every persistent device input of :class:`DecodeKVWrite` for ``step``: ``{name: (placement,
    int32 tensor)}``. ``"row"`` tensors are lane ordered (``[B]`` / ``[B, W]``; DP row ``r`` receives lanes ``8 r ..
    8 r + 7``), ``"rep"`` tensors are replicated on every chip.

    * always ``cur_pos`` (``positions``) and ``page_table`` (rows of inactive lanes zeroed), per row: FlashMLA and the
      ``row`` update;
    * ``row_split``: ``cur_a`` / ``cur_b`` (per row);
    * KV-R, per chunk ``c`` of ``lanes_per_call`` lanes: ``cur{c}`` (``all``) or ``cur_a{c}`` / ``cur_b{c}``
      (``all_split``), and ``pt{c}`` (``[n, W]``), replicated.
    """
    B = step.lanes
    n = _users_per_call(mode, B, lanes_per_row, lanes_per_call)
    act = step.active
    pos = step.positions.to(torch.int32)
    pt = torch.where(act[:, None], step.page_table, torch.zeros_like(step.page_table)).to(torch.int32)
    out: Dict[str, Tuple[Placement, torch.Tensor]] = {"cur_pos": ("row", pos.clone()), "page_table": ("row", pt)}
    if mode == "row_split":
        out["cur_a"] = ("row", step.call_positions("A"))
        out["cur_b"] = ("row", step.call_positions("B"))
    elif is_replicated(mode):
        for c in range(B // n):
            sl = slice(c * n, (c + 1) * n)
            if mode == "all":
                out[f"cur{c}"] = ("rep", pos[sl].clone())
            else:
                out[f"cur_a{c}"] = ("rep", step.call_positions("A")[sl].clone())
                out[f"cur_b{c}"] = ("rep", step.call_positions("B")[sl].clone())
            out[f"pt{c}"] = ("rep", pt[sl].clone())
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
        ccl: the model's ``MotifCCL`` (``ag_dp_rows`` for KV-R; semaphores in L1_SMALL).
        page_table_width: ``W`` of the decode trace (fixed for its life, like the generator's decode inputs).
        mode: one of :data:`generator_api.KV_WRITE_MODES`; default ``cfg.kv_write_mode``.
        lanes_per_call: KV-R users per update call (default :func:`lanes_per_call_for`: 32 for bfp8, 16 for bf16).

    Persistent device inputs (int32 ROW_MAJOR DRAM; rewritten by :meth:`write_step`): ``cur_pos [8]`` and
    ``page_table [8, W]`` per DP row (FlashMLA's inputs; also the ``row`` update's), plus the mode's call inputs
    (:func:`kv_write_inputs_host`). Never write them directly: :meth:`write_step` skips inputs whose values did not
    change since its last write.
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
    ):
        self.mode = check_mode(cfg.kv_write_mode if mode is None else mode)
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.ccl = ccl
        self.lanes = int(cfg.max_batch)
        self.lanes_per_row = int(cfg.lanes_per_row)
        self.block_size = int(cfg.kv_block_size)
        self.width = int(page_table_width)
        if self.width < 1:
            raise ValueError(f"page_table_width must be >= 1, got {page_table_width}")
        self.kv_dtype_name = cfg.dtypes.kv_cache_name
        self.lanes_per_call = lanes_per_call_for(
            self.mode, self.kv_dtype_name, lanes=self.lanes, lanes_per_row=self.lanes_per_row, requested=lanes_per_call
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
        """Name the persistent inputs: FlashMLA's ``cur_pos`` / ``page_table`` and, per lane chunk, the update calls
        ``[(update index tensor, page table tensor), ...]`` in device order (call A, then call B)."""
        self.cur_pos = self._dev["cur_pos"]
        self.page_table = self._dev["page_table"]
        if self.mode == "row":
            calls = [[(self.cur_pos, self.page_table)]]
        elif self.mode == "row_split":
            calls = [[(self._dev["cur_a"], self.page_table), (self._dev["cur_b"], self.page_table)]]
        else:
            names = ("cur",) if self.mode == "all" else ("cur_a", "cur_b")
            calls = [[(self._dev[f"{nm}{c}"], self._dev[f"pt{c}"]) for nm in names] for c in range(self.num_chunks)]
        self._calls = calls

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
        """KV-R lane chunks (one per ``lanes_per_call`` lanes); 1 in the row modes (per-row tensors)."""
        return self.lanes // self.lanes_per_call if self.replicated else 1

    @property
    def calls_per_layer(self) -> int:
        """``paged_update_cache`` calls per layer (per chip)."""
        return self.num_chunks * len(self._kinds)

    @property
    def device_inputs(self) -> Dict[str, Any]:
        """The persistent device inputs by name (read-only use: tests, diagnostics)."""
        return dict(self._dev)

    # ---- per step (host) ---------------------------------------------------------------------------------------
    def _values(self, step: KVWriteStep) -> Dict[str, Tuple[Placement, torch.Tensor]]:
        return kv_write_inputs_host(
            step,
            self.mode,
            lanes_per_call=self.lanes_per_call if self.replicated else None,
            lanes_per_row=self.lanes_per_row,
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
        """:func:`check_kv_write_step` with this object's mode and geometry (raises ``ValueError``)."""
        if step.lanes != self.lanes or step.width != self.width:
            raise ValueError(
                f"step has {step.lanes} lanes x width {step.width}; this kv_write was built for {self.lanes} x "
                f"{self.width} (the decode trace's page-table width)"
            )
        check_kv_write_step(
            step,
            self.mode,
            block_size=self.block_size,
            max_seq_len=self.cfg.max_model_len,
            lanes_per_call=self.lanes_per_call if self.replicated else None,
            lanes_per_row=self.lanes_per_row,
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
            kv_row: ``[1, 1, 8, 576]`` bf16 TILE DRAM (this DP row's lanes, replicated in the row): the attention's
                ``concat(n, rope(k_pe))``. Not consumed (the caller frees it).
            kv_cache: the layer's paged latent cache ``[N, 1, block, 576]`` (TILE, DRAM; any decode layer, the MTP
                layer included). Updated in place.
            cur_pos / page_table: FlashMLA's per-row inputs of this layer. They must be :attr:`cur_pos` /
                :attr:`page_table` (:meth:`check_flash_inputs`): the write and the read describe the same lanes.
        """
        self.check_flash_inputs(cur_pos, page_table)
        if not self.replicated:
            u = ttnn.transpose(kv_row, 1, 2, memory_config=self.row_mc)  # [1, 8, 1, 576], one lane per core
            for cur, pt in self._calls[0]:
                ttnn.experimental.paged_update_cache(kv_cache, u, update_idxs_tensor=cur, page_table=pt)
            ttnn.deallocate(u)
            return
        g = self.ccl.ag_dp_rows(kv_row)  # [1, 1, 32, 576] TILE DRAM, lane order 8 dp + l, identical on every chip
        if self.num_chunks == 1:
            u = ttnn.transpose(g, 1, 2, memory_config=self.call_mc)  # [1, 32, 1, 576], one lane per core
            ttnn.deallocate(g)
            for cur, pt in self._calls[0]:
                ttnn.experimental.paged_update_cache(kv_cache, u, update_idxs_tensor=cur, page_table=pt)
            ttnn.deallocate(u)
            return
        t = ttnn.transpose(g, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 32, 1, 576]
        ttnn.deallocate(g)
        n = self.lanes_per_call
        for c, calls in enumerate(self._calls):
            s = ttnn.slice(t, [0, c * n, 0, 0], [1, (c + 1) * n, 1, KV_LATENT_DIM])
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
        self._calls = []
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
    "check_kv_write_step",
    "check_mode",
    "is_replicated",
    "is_split",
    "kv_write_calls",
    "kv_write_inputs_host",
    "lanes_per_call_for",
    "update_input_memory_config",
]
