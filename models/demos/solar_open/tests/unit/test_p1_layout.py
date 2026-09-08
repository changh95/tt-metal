# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the perf-p1 layout levers of the expert paths (no device).

  * the sorted-MoE cost model (``prefill._plan_from_counts``) with the shipped constants: uniform top-8 and the skewed
    unit-test distribution at 1024 tokens take the sorted path, an always-on slot rides in the hot group, > 16 hot
    experts fall back to the dense loop, tiny splits (every cap >= split) return None;
  * the hot group's K-concatenated down config (``prefill._hot_down_kconcat_config``): exact grid fills at the real
    shapes, None when the shapes do not divide;
  * the module switches default ON and the batched decode path multiplies the routing weights into the down INPUT
    (source-level contract of ``decode.ROUTING_WEIGHTS_ON_DOWN_INPUT``).

    SOLAR_OPEN_NUM_DEVICES=8 pytest models/demos/solar_open/tests/unit/test_p1_layout.py
"""

import torch

from models.demos.solar_open.tt.experts import decode as experts_decode
from models.demos.solar_open.tt.experts import prefill as experts_prefill

E, H, IP = 128, 4096, 160


def _uniform_counts(split_len, top_k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    counts = torch.zeros(E, dtype=torch.int64)
    for _ in range(split_len):
        counts[torch.randperm(E, generator=g)[:top_k]] += 1
    return counts


def _skewed_counts(split_len, hot={3: 0.60, 7: 0.25, 11: 0.12}, top_k=8, seed=1):
    """The distribution of tests/test_experts_skewed_routing.py."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.zeros(E, dtype=torch.int64)
    for _ in range(split_len):
        forced = [e for e, p in hot.items() if torch.rand(1, generator=g).item() < p]
        others = [e for e in torch.randperm(E, generator=g).tolist() if e not in forced]
        counts[torch.tensor((forced + others)[:top_k])] += 1
    return counts


class TestPlanFromCounts:
    def test_uniform_1024_takes_the_sorted_path(self):
        best = experts_prefill._plan_from_counts(_uniform_counts(1024), 1024, E)
        assert best is not None
        cost, cap, hot = best
        assert cap in (96, 128, 160, 192) and hot <= 2, best  # ~64 routed tokens per expert
        assert cost < experts_prefill._DENSE_PER_EXPERT_MS * E

    def test_skewed_1024_has_three_hot_experts(self):
        best = experts_prefill._plan_from_counts(_skewed_counts(1024), 1024, E)
        assert best is not None
        _, cap, hot = best
        assert hot == 3 and cap < 1024, best  # experts 3 / 7 / 11 exceed every cap

    def test_always_on_slot_rides_in_the_hot_group_but_not_the_hot_count(self):
        counts = _uniform_counts(1024)
        plain = experts_prefill._plan_from_counts(counts, 1024, E)
        fused = experts_prefill._plan_from_counts(counts, 1024, E + 1, always_on=1)
        assert plain is not None and fused is not None
        assert fused[2] == plain[2]  # routed hot count unchanged
        assert fused[0] > plain[0]  # ... but the always-on expert costs one hot-expert slot (plus E+1 cold rows)

    def test_too_many_hot_experts_fall_back_to_the_dense_loop(self):
        counts = torch.full((E,), 1024 // 4, dtype=torch.int64)  # every expert takes a quarter of the split
        counts[:20] = 1000  # 20 experts above every cap
        assert experts_prefill._plan_from_counts(counts, 1024, E) is None

    def test_short_split_has_no_cap(self):
        assert experts_prefill._plan_from_counts(_uniform_counts(32), 32, E) is None

    def test_hot_constants_re_derived(self):
        assert (experts_prefill._HOT_FIXED_MS, experts_prefill._HOT_PER_EXPERT_MS) == (0.4, 0.135)

    def test_sorted_beats_dense_by_a_margin_at_1024(self):
        """With the re-derived hot constants the 1024-token sorted plan must stay well below the dense loop."""
        cost = experts_prefill._plan_from_counts(_skewed_counts(1024), 1024, E)[0]
        dense = experts_prefill._DENSE_PER_EXPERT_MS * E
        assert cost < 0.6 * dense, (cost, dense)


class TestHotDownKConcatConfig:
    def test_real_shapes_fill_the_grid_exactly(self):
        for n_hot in (2, 4, 8, 15, 16):
            cfg = experts_prefill._hot_down_kconcat_config((8, 8), 1024, n_hot * IP, H)
            assert cfg is not None, n_hot
            assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (8, 8)
            assert cfg.per_core_M == 4 and cfg.per_core_N == 16  # 32 x 128 output tiles over 8 x 8 cores
            assert cfg.in0_block_w == 5  # one expert's Ip per K block
            assert cfg.out_block_h == 4 and cfg.out_block_w == 16
            assert cfg.out_subblock_h == 1 and cfg.out_subblock_w == 8
            assert cfg.transpose_mcast is False

    def test_512_token_split(self):
        cfg = experts_prefill._hot_down_kconcat_config((8, 8), 512, 4 * IP, H)
        assert cfg is not None and cfg.per_core_M == 2

    def test_non_dividing_shapes_return_none(self):
        assert experts_prefill._hot_down_kconcat_config((8, 8), 1024 + 32, 4 * IP, H) is None  # Mt 33 % 8
        assert experts_prefill._hot_down_kconcat_config((7, 8), 1024, 4 * IP, H) is None  # Nt 128 % 7
        assert experts_prefill._hot_down_kconcat_config(None, 1024, 4 * IP, H) is None
        assert experts_prefill._hot_down_kconcat_config((0, 0), 1024, 4 * IP, H) is None  # the auto A/B variant

    def test_shipped_cores(self):
        assert experts_prefill.HOT_DOWN_KCONCAT_CORES == (8, 8)


class TestSwitches:
    def test_defaults_on(self):
        assert experts_prefill.HOT_EXPERTS_PER_EXPERT_LINEAR is True
        assert (
            experts_prefill.HOT_DOWN_KCONCAT is False
        )  # measured slower than bmm + fast_reduce_nc (kept as the alternative)
        assert experts_prefill.ELIDE_ROUTING_COPIES is True
        assert experts_decode.ROUTING_WEIGHTS_ON_DOWN_INPUT is True

    def test_batched_decode_multiplies_the_down_input(self):
        """Source-level contract: in the batched path the routing mul precedes the down sparse_matmul."""
        import inspect

        src = inspect.getsource(experts_decode._decode_forward_batched)
        mul_in = src.index("down_input = ttnn.mul(down_input, token_expert_weights")
        down = src.index("down = ttnn.sparse_matmul(")
        assert mul_in < down

    def test_concat_expert_slices_dim_default(self):
        import inspect

        sig = inspect.signature(experts_prefill._concat_expert_slices)
        assert sig.parameters["dim"].default == 1
