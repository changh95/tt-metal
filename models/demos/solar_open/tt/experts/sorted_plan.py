# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side planner of the expert-sorted prefill MoE (``experts/prefill.py``): pure functions, NO ttnn import.

The device half (``prefill._sorted_moe_plan`` per split, ``prefill._sorted_moe_chunk_plan`` per chunk) reads the
per-expert routed-token counts back to the host and hands them to the cost model here; everything in this module can
be exercised without a device (``tests/unit/test_sorted_moe_chunk_plan.py``).

Phase 3c (design_traced_prefill.md G1, measurements.md "Mechanism of the model-level slot dependence"): the sorted MoE
partitions the experts of a split into HOT (count > cap: dense over the whole split) and COLD (count <= cap: gathered
``cap`` rows each). Planned per 1024-token split, a 32 x 128-token packed pass -- ONE 4096-token chunk cut into four
splits of 8 users -- gives a token's expert the hot path in one split and the cold path in another, so identical users
in different slots get different bfp8-level numerics. ``plan_chunk_from_counts`` plans ONCE per chunk from the
per-split counts of all its splits and the same (cap, hot set, cold mask) then serves every split of the chunk:

* the HOT SET is decided on the chunk AVERAGE per-split count of every expert (total over the chunk / n_splits),
  which depends only on the SET of users in the pass, not on how they are partitioned into splits -- a hot set
  taken from the per-split maxima (or the union of per-split plans) would still move with the slot layout, and a
  hot <-> cold flip of one expert changes every token routed to it (the hot group and the gathered path are two
  different bfp8 accumulations);
* the CAP (gathered rows per cold expert) is raised from the cost model's choice to the largest per-split count of
  any cold expert, so every cold expert fits ``cap`` rows in EVERY split (topk keeps only ``cap`` rows: a smaller cap
  would drop tokens). The cap is numerics-neutral within SORTED_MOE_CAPS (every cold-path op works per row with the
  same blocking for 32..256 rows; only the fp32 order of a token's <= 9 scatter terms moves, ~1e-5 in PCC), so a
  layout-dependent cap does not make the pass layout dependent. A cold expert with more than the largest cap in
  some split (it would be hot in that split's per-split plan) is promoted to hot -- the one layout-dependent case,
  rare and diagnosed (``n_promoted``).

A pass that fits one chunk is therefore slot independent; its numerics still depend on the SET of co-batched users
(inherent to hot/cold).
"""

import dataclasses
import os

import torch

_SORTED_MOE_MAX_HOT = 16  # more hot experts than this -> dense per-expert loop for the split
# Cost model (ms per 1024-token split, P150) used to pick the hot/cold threshold on the host. Re-measured for the
# Solar-Open shapes (H=4096 / Ip=160, bfp8) from the tracy device profile of the real-weight layer 0 at 1024 tokens
# (2026-09-07; the shipped values were the H=2880 / Ip=384 numbers of tt-metal PR #55589: 2.5, 0.27, 1.0, 0.25 and
# 0.125): fixed sorted cost ~0.5 ms (planner ops, topk, index/slot glue, untilizes), 0.23 ms per 1024 gathered rows
# (embedding gather 46 + gate/up bmm 49 + down bmm 72 + one-hot scatter matmul 49 + slices/GLU/mul 9 us), hot group
# fixed ~0.3 ms (concats, reduce, adds) + 0.25 ms per hot expert (repeat + bmm + slices), dense per-expert loop
# 0.15 ms (gate/up linear 77 + down 33 + 2 weight slices 18 + GLU/mul/concat share ~20 us). The constants only
# steer the hot/cold split and the sorted-vs-dense choice, never correctness. perf-p1 (2026-09-07): with the hot
# group as per-expert linears (HOT_EXPERTS_PER_EXPERT_LINEAR) its wall time at 1024 tokens fits 0.39 + 0.136 ms x
# n_hot (measured 0.93 / 1.48 / 2.42 ms at 4 / 8 / 15 hot experts vs 1.25 / 2.26 / 4.06 for repeat + bmm, i.e.
# 0.23 + 0.255 x n_hot) -> _HOT_FIXED_MS, _HOT_PER_EXPERT_MS = 0.4, 0.135. Phase 3 (the minimal_matmul gate|up and
# the K-concatenated hot down of SolarOpenProgramConfig) makes the hot group cheaper (~-35 us per expert, ~-0.3 ms at
# 15 hot); the constants are conservative until re-derived from a layer-0 tracy (pinned by tests/unit/test_p1_layout).
_SORTED_FIXED_MS, _SORTED_PER_KROW_MS, _HOT_FIXED_MS, _HOT_PER_EXPERT_MS = 0.5, 0.23, 0.4, 0.135
_DENSE_PER_EXPERT_MS = 0.15  # per-expert cost of the dense loop over a 1024-token split (gate/up + down + slices)
# Measured on P150x8 (PR #55589): the sorted path halves E=128 prefill at ISL >= 1024 but is slower than the dense
# loop for E=32 (~128 routed tokens per expert per 1024, so the gathered rows are not much fewer and the fixed
# cost + host round-trip dominate).
_SORTED_MOE_MIN_EXPERTS = 64
# Candidate caps (gathered rows per cold expert), tried in this order; a cap must stay below the split length.
SORTED_MOE_CAPS = (32, 64, 96, 128, 160, 192, 256)

# ---------------------------------------------------------------------------------------------------------------------
# Plan granularity (phase 3c), env SOLAR_OPEN_SORTED_MOE_PLAN:
#   auto   (default) plan once per chunk for a PACKED multi-user pass (Model.ttnn_prefill_forward with batch_size > 1
#          marks the pass, prefill.packed_prefill_pass) and per split otherwise -- the single-user prefill path keeps
#          the phase-3b ops and numbers byte for byte, a packed pass becomes slot independent
#   chunk  plan once per chunk for EVERY prefill (also the single-user 2K+ prefills: one count readback per chunk
#          instead of one per split -- design G1 -- with the chunk-average hot set; outputs of 1025+-token single-user
#          prompts move at the bfp8 floor, TTFT effect to be measured)
#   split  the phase-3b per-split plan for every prefill (the A/B arm; packed passes are then slot dependent again)
# A chunk with a single host-planned split is always planned by the per-split code (identical result, same ops).
SORTED_MOE_PLAN_MODES = ("auto", "chunk", "split")
SORTED_MOE_PLAN_DEFAULT = "auto"
# Hot-set rule of the per-chunk plan (plan_chunk_from_counts ``hot_rule``), env SOLAR_OPEN_SORTED_MOE_CHUNK_HOT:
#   average (default) hot = chunk-average per-split count > the cost model's cap on the average; the cap is raised
#           to fit every cold expert in every split -- layout invariant hot set (slot independence), possibly a
#           larger gather cap than the per-split plans would use (perf to be measured)
#   max     hot = per-expert MAX over the splits > the cost model's cap on the maxima (= the union of the per-split
#           plans' hot sets at that cap; the cap fits by construction): the cheaper caps of the per-split plans, but
#           the hot set moves with the slot layout (users re-partitioned into splits change the maxima) -- the A/B arm
SORTED_MOE_CHUNK_HOT_RULES = ("average", "max")
SORTED_MOE_CHUNK_HOT_DEFAULT = "average"


def sorted_moe_plan_mode_from_env(default=SORTED_MOE_PLAN_DEFAULT):
    """``SOLAR_OPEN_SORTED_MOE_PLAN`` validated against SORTED_MOE_PLAN_MODES (unset / empty -> ``default``)."""
    raw = os.getenv("SOLAR_OPEN_SORTED_MOE_PLAN")
    mode = default if raw is None or raw.strip() == "" else raw.strip().lower()
    if mode not in SORTED_MOE_PLAN_MODES:
        raise ValueError(f"SOLAR_OPEN_SORTED_MOE_PLAN={raw!r} is not one of {SORTED_MOE_PLAN_MODES}")
    return mode


def sorted_moe_chunk_hot_rule_from_env(default=SORTED_MOE_CHUNK_HOT_DEFAULT):
    """``SOLAR_OPEN_SORTED_MOE_CHUNK_HOT`` validated against SORTED_MOE_CHUNK_HOT_RULES (unset / empty -> ``default``)."""
    raw = os.getenv("SOLAR_OPEN_SORTED_MOE_CHUNK_HOT")
    rule = default if raw is None or raw.strip() == "" else raw.strip().lower()
    if rule not in SORTED_MOE_CHUNK_HOT_RULES:
        raise ValueError(f"SOLAR_OPEN_SORTED_MOE_CHUNK_HOT={raw!r} is not one of {SORTED_MOE_CHUNK_HOT_RULES}")
    return rule


def plan_per_chunk(mode, packed_pass, n_planned_splits):
    """Whether the ``n_planned_splits`` host-planned splits of one chunk share ONE plan (True) or are planned one by
    one (False) under plan ``mode``; ``packed_pass`` is the Model's "this forward is a packed multi-user pass" mark.
    A chunk with fewer than two planned splits is always planned per split (one split: the two plans coincide)."""
    if mode not in SORTED_MOE_PLAN_MODES:
        raise ValueError(f"unknown sorted-MoE plan mode {mode!r}; choose one of {SORTED_MOE_PLAN_MODES}")
    if n_planned_splits < 2:
        return False
    if mode == "chunk":
        return True
    if mode == "split":
        return False
    return bool(packed_pass)


def split_pieces(length, size):
    """Piece lengths ``ttnn.split(x, size)`` cuts a ``length``-long axis into: full pieces, then the remainder."""
    return [size] * (length // size) + ([length % size] if length % size else [])


def _plan_from_counts(routed_counts, split_len, num_experts, always_on=0):
    """Cost-model core of the sorted planners on the host-side per-expert routed-token counts (torch int64 [E_routed]).

    Returns ``(cost_ms, cap, n_hot)`` for the cheapest cap of SORTED_MOE_CAPS below ``split_len`` with at most
    _SORTED_MOE_MAX_HOT routed hot experts (count > cap), or None when every cap is over the hot limit or none beats
    the dense per-expert loop (``_DENSE_PER_EXPERT_MS`` x E x split / 1024). Pure function (unit-tested on synthetic
    count distributions)."""
    best = None
    for cap in SORTED_MOE_CAPS:
        if cap >= split_len:
            # strict: an always-on slot has count == split_len and must stay OUT of the cold mask le(counts, cap)
            break
        hot = int((routed_counts > cap).sum().item())
        if hot > _SORTED_MOE_MAX_HOT:
            continue
        n_hot_group = hot + always_on
        cost = (
            _SORTED_FIXED_MS
            + _SORTED_PER_KROW_MS * (num_experts * cap / 1024)
            + (_HOT_FIXED_MS + _HOT_PER_EXPERT_MS * n_hot_group if n_hot_group else 0.0)
        )
        if best is None or cost < best[0]:
            best = (cost, cap, hot)
    # The sorted path only pays off when the routed rows are few relative to E x split (E=128, top-8: ~64 routed
    # tokens per expert per 1024); when it is not cheaper than the dense per-expert loop (which has no host
    # round-trip) the loop is kept.
    dense_cost = _DENSE_PER_EXPERT_MS * num_experts * split_len / 1024
    if best is None or best[0] >= dense_cost:
        return None
    return best


@dataclasses.dataclass(frozen=True)
class SortedMoeChunkPlan:
    """One plan for every host-planned split of a chunk (``plan_chunk_from_counts``).

    ``hot_ids`` are the routed experts whose chunk-average per-split count exceeds ``cap_cost`` (plus the promoted
    ones, ``n_promoted``) followed by the always-on slots; ``n_hot`` counts the routed ones. ``cap`` is the number of
    rows gathered per cold expert in every split (>= ``cap_cost``, the cost model's cap on the average split, raised
    to fit the largest per-split count of any cold expert). ``cold_mask`` is the ``[E]`` float mask (1.0 = cold, on
    the sorted path in every split; 0.0 = hot or always-on) the device multiplies into the cold slot weights --
    identical for all splits. ``per_split_hot`` is diagnostic: how many routed experts exceed ``cap`` in each split
    on its own (what the per-split plans would have made hot at this cap)."""

    split_lens: tuple
    cap: int
    cap_cost: int
    hot_ids: tuple
    n_hot: int
    n_promoted: int
    cost_ms: float
    per_split_hot: tuple
    cold_mask: torch.Tensor
    hot_rule: str = SORTED_MOE_CHUNK_HOT_DEFAULT


def plan_chunk_from_counts(counts, split_lens, num_experts, always_on=0, hot_rule=SORTED_MOE_CHUNK_HOT_DEFAULT):
    """Per-chunk plan from ``counts`` (torch int64 ``[E, n_splits]``: routed tokens per expert slot in each
    host-planned split of the chunk, always-on slots last) -> ``SortedMoeChunkPlan`` or None (every planned split
    takes the dense per-expert loop). ``hot_rule`` "average" (default, layout invariant) is described below; "max"
    runs ``_plan_from_counts`` on the per-expert maxima over the splits instead (hot = max > cap, the cap then fits
    by construction, nothing is promoted): the union of the per-split plans' hot sets, the A/B arm.

    1. ``_plan_from_counts`` on the chunk AVERAGE per-split counts (total / n_splits; the shortest planned split is the
       cap bound so ``cap < split_len`` holds in every split) gives ``cap_cost`` and the sorted-vs-dense decision; the
       hot set is ``average > cap_cost`` -- a function of the SET of users in the pass only.
    2. Cold experts whose count in some split exceeds the largest cap below the split length cannot be gathered there
       and are promoted to hot (layout dependent, rare; ``n_promoted``). More than _SORTED_MOE_MAX_HOT hot experts ->
       None.
    3. ``cap`` = the smallest SORTED_MOE_CAPS entry >= max(cap_cost, largest per-split count of any cold expert): every
       cold expert fits in every split.
    With one split this is exactly the per-split plan of ``_plan_from_counts``."""
    split_lens = tuple(int(n) for n in split_lens)
    n_splits = len(split_lens)
    if n_splits == 0:
        raise ValueError("a chunk plan needs at least one host-planned split")
    E_routed = num_experts - always_on
    if E_routed < _SORTED_MOE_MIN_EXPERTS:
        return None
    if tuple(counts.shape) != (num_experts, n_splits):
        raise ValueError(f"counts must be [E={num_experts}, n_splits={n_splits}], got {tuple(counts.shape)}")
    routed = counts[:E_routed].to(torch.int64)
    cap_split_len = min(split_lens)
    caps_below = [c for c in SORTED_MOE_CAPS if c < cap_split_len]
    if not caps_below:
        return None
    if hot_rule not in SORTED_MOE_CHUNK_HOT_RULES:
        raise ValueError(f"unknown hot_rule {hot_rule!r}; choose one of {SORTED_MOE_CHUNK_HOT_RULES}")
    max_split = routed.max(dim=1).values
    if hot_rule == "max":
        best = _plan_from_counts(max_split, cap_split_len, num_experts, always_on)
        if best is None:
            return None
        cost, cap_cost, _ = best
        hot = max_split > cap_cost
        promoted = torch.zeros_like(hot)
    else:
        average = routed.sum(dim=1).to(torch.float64) / n_splits  # partition invariant
        best = _plan_from_counts(average, cap_split_len, num_experts, always_on)
        if best is None:
            return None
        cost, cap_cost, _ = best
        hot = average > cap_cost
        promoted = (~hot) & (max_split > caps_below[-1])
        hot = hot | promoted
    n_hot = int(hot.sum().item())
    if n_hot > _SORTED_MOE_MAX_HOT:
        return None
    need = cap_cost
    if bool((~hot).any()):
        need = max(need, int(max_split[~hot].max().item()))
    cap = next(c for c in caps_below if c >= need)  # exists: need <= caps_below[-1] after the promotion
    assert cap < cap_split_len, (cap, cap_split_len)  # the cold mask relies on count(always-on) == split_len > cap
    hot_ids = tuple(int(e) for e in torch.nonzero(hot).reshape(-1).tolist()) + tuple(range(E_routed, num_experts))
    cold_mask = torch.zeros(num_experts, dtype=torch.float32)
    cold_mask[:E_routed] = (~hot).to(torch.float32)  # always-on slots: 0 (hot group), never double counted
    per_split_hot = tuple(int((routed[:, s] > cap).sum().item()) for s in range(n_splits))
    return SortedMoeChunkPlan(
        split_lens=split_lens,
        cap=cap,
        cap_cost=cap_cost,
        hot_ids=hot_ids,
        n_hot=n_hot,
        n_promoted=int(promoted.sum().item()),
        cost_ms=cost,
        per_split_hot=per_split_hot,
        cold_mask=cold_mask,
        hot_rule=hot_rule,
    )
