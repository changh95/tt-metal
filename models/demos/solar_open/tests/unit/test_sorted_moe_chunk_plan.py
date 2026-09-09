# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the phase-3c per-chunk sorted-MoE planner and the packed-prefill gather head (no device).

``tt/experts/sorted_plan.py`` and ``tt/packed_prefill.py`` import no ttnn; they can be loaded by file path with
importlib for a check on a box whose devices are busy (that is how the B1 stage exercised them). Under pytest the
package imports below are fine (importing ttnn opens no device).

  * ``plan_chunk_from_counts``: the per-chunk plan equals the per-split plan for a single split; for several splits
    the hot set comes from the chunk-AVERAGE per-split counts (a function of the set of users only), the cap is
    raised so that every cold expert fits ``cap`` rows in EVERY split, the cold mask is the complement (always-on
    slots 0), identical for all splits; promotion of a cold expert above the largest cap; ``None`` over the hot
    limit; the slot-independence property on the measured mechanism (any permutation of the users over the slots
    leaves hot set, cold mask and cost-model cap unchanged, whereas the per-split plans move);
  * ``plan_per_chunk``: the auto / chunk / split decision table;
  * ``gather_head_rows``: the last-token row indices of a packed pass and their validation.

    SOLAR_OPEN_NUM_DEVICES=8 pytest models/demos/solar_open/tests/unit/test_sorted_moe_chunk_plan.py
"""

import torch

from models.demos.solar_open.tt.experts import sorted_plan as sp
from models.demos.solar_open.tt.packed_prefill import GATHER_HEAD_ROWS, gather_head_rows

E, TOP_K = 128, 8


def _routing(split_len, num_splits, seed=0, hot=None):
    """[E, num_splits] routed-token counts of top-8 routing; ``hot`` = {expert: probability} forced per token."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.zeros(E, num_splits, dtype=torch.int64)
    for s in range(num_splits):
        for _ in range(split_len):
            forced = [e for e, p in (hot or {}).items() if torch.rand(1, generator=g).item() < p]
            others = [e for e in torch.randperm(E, generator=g).tolist() if e not in forced]
            counts[torch.tensor((forced + others)[:TOP_K]), s] += 1
    return counts


class TestPlanChunkFromCounts:
    def test_single_split_equals_the_per_split_plan(self):
        counts = _routing(1024, 1, hot={3: 0.6, 7: 0.25})
        plan = sp.plan_chunk_from_counts(counts, (1024,), E)
        best = sp._plan_from_counts(counts[:, 0], 1024, E)
        assert plan is not None and best is not None
        assert (plan.cost_ms, plan.cap, plan.n_hot) == best and plan.cap_cost == plan.cap and plan.n_promoted == 0
        assert plan.hot_ids == tuple(torch.nonzero(counts[:, 0] > plan.cap).reshape(-1).tolist())
        assert torch.equal(plan.cold_mask, (counts[:, 0] <= plan.cap).float())
        assert plan.per_split_hot == (plan.n_hot,)

    def test_shared_plan_hot_set_from_the_average_and_cap_fits_every_split(self):
        # expert 3 heavy in split 0 only, expert 7 heavy in split 1 only (+ the uniform background)
        counts = torch.cat(
            [_routing(1024, 1, seed=1, hot={3: 0.5}), _routing(1024, 1, seed=2, hot={7: 0.5})], dim=1
        )  # [E, 2]
        plan = sp.plan_chunk_from_counts(counts, (1024, 1024), E)
        assert plan is not None
        average = counts.sum(dim=1).double() / 2
        expected_hot = tuple(torch.nonzero(average > plan.cap_cost).reshape(-1).tolist())
        assert plan.hot_ids == expected_hot and plan.n_hot == len(expected_hot) and plan.n_promoted == 0
        assert torch.equal(plan.cold_mask, (average <= plan.cap_cost).float())
        # every cold expert fits cap rows in EVERY split (no dropped tokens); the cap is a listed cap below the split
        cold = plan.cold_mask > 0
        assert (counts[cold] <= plan.cap).all()
        assert plan.cap >= plan.cap_cost and plan.cap in sp.SORTED_MOE_CAPS and plan.cap < 1024
        assert plan.cap == next(c for c in sp.SORTED_MOE_CAPS if c >= max(plan.cap_cost, int(counts[cold].max())))
        # the cost model's cap and the sorted-vs-dense decision are those of the average split
        assert sp._plan_from_counts(average, 1024, E)[1] == plan.cap_cost
        assert len(plan.per_split_hot) == 2

    def test_cold_expert_above_the_largest_cap_in_one_split_is_promoted(self):
        counts = _routing(1024, 4, seed=8)  # ~64 per expert per split
        counts[5] = torch.tensor([300, 10, 10, 10])  # average 82.5 (cold at cap >= 96), but 300 > 256 in split 0
        plan = sp.plan_chunk_from_counts(counts, (1024,) * 4, E)
        assert plan is not None
        assert 5 in plan.hot_ids and plan.n_promoted == 1 and plan.cold_mask[5] == 0.0
        assert (counts[plan.cold_mask > 0] <= plan.cap).all()

    def test_slot_independence_of_the_measured_mechanism(self):
        """The 32 x 128 packed pass: four 1024-token splits of 8 users. Any permutation of the users over the slots
        must leave the hot set, the cold mask and the cost-model cap unchanged (the plan sees the multiset of all
        users' routings); only the gather cap may follow the per-split maxima. The phase-3b per-split plans of split 0
        change with the layout for the same routing."""
        users = [
            _routing(128, 1, seed=100 + u, hot={u: 0.6}) for u in range(32)
        ]  # [E, 1] each: user u favours expert u

        def counts_of(layout):
            return torch.cat([sum(users[layout[8 * s + i]] for i in range(8)) for s in range(4)], dim=1)  # [E, 4]

        layout_a = list(range(32))
        plan_a = sp.plan_chunk_from_counts(counts_of(layout_a), (1024,) * 4, E)
        assert plan_a is not None
        g = torch.Generator().manual_seed(7)
        split0_hot_sets = set()
        for _ in range(12):
            layout_b = torch.randperm(32, generator=g).tolist()
            counts_b = counts_of(layout_b)
            plan_b = sp.plan_chunk_from_counts(counts_b, (1024,) * 4, E)
            assert plan_b is not None
            assert (plan_a.cap_cost, plan_a.hot_ids, plan_a.n_promoted) == (plan_b.cap_cost, plan_b.hot_ids, 0)
            assert torch.equal(plan_a.cold_mask, plan_b.cold_mask)
            assert (counts_b[plan_b.cold_mask > 0] <= plan_b.cap).all()
            split0 = sp._plan_from_counts(counts_b[:, 0], 1024, E)
            assert split0 is not None
            split0_hot_sets.add(frozenset(torch.nonzero(counts_b[:, 0] > split0[1]).reshape(-1).tolist()))
        assert len(split0_hot_sets) > 1, "the synthetic routing must expose the per-split dependence"

    def test_always_on_slot_rides_in_the_hot_group_and_is_never_cold(self):
        counts = _routing(1024, 4, seed=3)
        fused = torch.cat([counts, torch.full((1, 4), 1024, dtype=torch.int64)])  # slot E = always-on
        plan = sp.plan_chunk_from_counts(fused, (1024,) * 4, E + 1, always_on=1)
        plain = sp.plan_chunk_from_counts(counts, (1024,) * 4, E)
        assert plan is not None and plain is not None
        assert plan.hot_ids[-1] == E and plan.n_hot == plain.n_hot  # the slot rides in the hot group, not the count
        assert plan.cold_mask[E] == 0.0 and plan.cold_mask.shape == (E + 1,)
        assert plan.cost_ms > plain.cost_ms

    def test_none_when_the_union_exceeds_the_hot_limit_or_a_cap_is_impossible(self):
        counts = torch.full((E, 4), 1024 // 4, dtype=torch.int64)
        for s in range(
            4
        ):  # 5 distinct heavy experts per split -> 20 in the union, above _SORTED_MOE_MAX_HOT at every cap
            counts[5 * s : 5 * s + 5, s] = 1000
        assert sp.plan_chunk_from_counts(counts, (1024,) * 4, E) is None
        assert sp.plan_chunk_from_counts(_routing(32, 2), (32, 32), E) is None  # every cap >= the split
        assert sp.plan_chunk_from_counts(_routing(1024, 2)[:32], (1024, 1024), 32) is None  # too few experts

    def test_max_rule_is_the_union_of_the_per_split_plans_and_may_move_with_the_layout(self, expect_error):
        """hot_rule "max" (the A/B arm): hot = per-expert max over the splits > the cost model's cap on the maxima,
        the cap fits by construction; the same 32-user routing as the slot-independence test shows its hot set
        changes with the layout (that is why "average" is the default)."""
        users = [_routing(128, 1, seed=100 + u, hot={u: 0.6}) for u in range(32)]

        def counts_of(layout):
            return torch.cat([sum(users[layout[8 * s + i]] for i in range(8)) for s in range(4)], dim=1)

        counts = counts_of(list(range(32)))
        plan = sp.plan_chunk_from_counts(counts, (1024,) * 4, E, hot_rule="max")
        assert plan is not None and plan.hot_rule == "max" and plan.n_promoted == 0 and plan.cap == plan.cap_cost
        max_split = counts.max(dim=1).values
        assert sp._plan_from_counts(max_split, 1024, E)[1] == plan.cap_cost
        assert plan.hot_ids == tuple(torch.nonzero(max_split > plan.cap).reshape(-1).tolist())
        assert (counts[plan.cold_mask > 0] <= plan.cap).all()
        g = torch.Generator().manual_seed(11)
        hot_sets = {plan.hot_ids}
        for _ in range(12):
            other = sp.plan_chunk_from_counts(
                counts_of(torch.randperm(32, generator=g).tolist()), (1024,) * 4, E, hot_rule="max"
            )
            hot_sets.add(other.hot_ids if other is not None else None)
        assert len(hot_sets) > 1, "the max rule is expected to depend on the layout for this routing"
        with expect_error(ValueError, "unknown hot_rule"):
            sp.plan_chunk_from_counts(counts, (1024,) * 4, E, hot_rule="union")

    def test_cap_bound_is_the_shortest_planned_split(self):
        counts = torch.cat([_routing(1024, 1, seed=4), _routing(512, 1, seed=5)], dim=1)
        plan = sp.plan_chunk_from_counts(counts, (1024, 512), E)
        assert plan is not None and plan.cap < 512 and plan.split_lens == (1024, 512)

    def test_shape_validation(self, expect_error):
        with expect_error(ValueError, "counts must be"):
            sp.plan_chunk_from_counts(_routing(1024, 2), (1024,), E)
        with expect_error(ValueError, "at least one"):
            sp.plan_chunk_from_counts(_routing(1024, 1), (), E)


class TestPlanPerChunk:
    def test_decision_table(self, expect_error):
        assert sp.plan_per_chunk("auto", packed_pass=True, n_planned_splits=4) is True
        assert sp.plan_per_chunk("auto", packed_pass=False, n_planned_splits=4) is False  # single-user 4K: phase-3b ops
        assert sp.plan_per_chunk("chunk", packed_pass=False, n_planned_splits=2) is True
        assert sp.plan_per_chunk("split", packed_pass=True, n_planned_splits=4) is False
        for mode in sp.SORTED_MOE_PLAN_MODES:  # one planned split: always the per-split code (identical plan)
            assert sp.plan_per_chunk(mode, packed_pass=True, n_planned_splits=1) is False
            assert sp.plan_per_chunk(mode, packed_pass=True, n_planned_splits=0) is False
        with expect_error(ValueError, "unknown sorted-MoE plan mode"):
            sp.plan_per_chunk("per-user", True, 4)

    def test_mode_from_env(self, monkeypatch, expect_error):
        monkeypatch.delenv("SOLAR_OPEN_SORTED_MOE_PLAN", raising=False)
        assert sp.sorted_moe_plan_mode_from_env() == "auto"
        monkeypatch.setenv("SOLAR_OPEN_SORTED_MOE_PLAN", " Chunk ")
        assert sp.sorted_moe_plan_mode_from_env() == "chunk"
        monkeypatch.setenv("SOLAR_OPEN_SORTED_MOE_PLAN", "")
        assert sp.sorted_moe_plan_mode_from_env() == "auto"
        monkeypatch.setenv("SOLAR_OPEN_SORTED_MOE_PLAN", "user")
        with expect_error(ValueError, "SOLAR_OPEN_SORTED_MOE_PLAN"):
            sp.sorted_moe_plan_mode_from_env()

    def test_hot_rule_from_env(self, monkeypatch, expect_error):
        monkeypatch.delenv("SOLAR_OPEN_SORTED_MOE_CHUNK_HOT", raising=False)
        assert sp.sorted_moe_chunk_hot_rule_from_env() == "average"
        monkeypatch.setenv("SOLAR_OPEN_SORTED_MOE_CHUNK_HOT", "MAX")
        assert sp.sorted_moe_chunk_hot_rule_from_env() == "max"
        monkeypatch.setenv("SOLAR_OPEN_SORTED_MOE_CHUNK_HOT", "union")
        with expect_error(ValueError, "SOLAR_OPEN_SORTED_MOE_CHUNK_HOT"):
            sp.sorted_moe_chunk_hot_rule_from_env()

    def test_split_pieces(self):
        assert sp.split_pieces(4096, 1024) == [1024] * 4
        assert sp.split_pieces(1152, 1024) == [1024, 128]
        assert sp.split_pieces(512, 1024) == [512]


class TestGatherHeadRows:
    def test_rows_and_padding(self):
        rows = gather_head_rows([80, 128, 1], 128)
        assert len(rows) == GATHER_HEAD_ROWS == 32
        assert rows[:3] == [79, 128 + 127, 2 * 128 + 0] and rows[3:] == [0] * 29

    def test_thirty_two_users(self):
        rows = gather_head_rows([128] * 32, 128)
        assert rows == [u * 128 + 127 for u in range(32)] and max(rows) == 32 * 128 - 1

    def test_validation(self, expect_error):
        with expect_error(ValueError, "at least one"):
            gather_head_rows([], 128)
        with expect_error(ValueError, "exceed"):
            gather_head_rows([8] * 33, 128)
        with expect_error(ValueError, "must be in 1..128"):
            gather_head_rows([129], 128)
        with expect_error(ValueError, "must be in 1..128"):
            gather_head_rows([0], 128)
