# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of the phase-2 (perf-p0) program-config builders and their ON defaults in the merged main tree: the
explicit 1D matmul configs of the shared expert and the router, the width-sharded decode norm (and the padded-row
gate that keeps it off the shapes it cannot shard), the attention decode qkv config with its HiFi2 (bf16-dst) compute
config and the dense prefill down config -- plus the phase-3 prefill expert matmul knobs of SolarOpenProgramConfig
(the [Mt x 2] dense down subblock, the minimal_matmul blockings of the per-expert gate|up and the K-concatenated hot
down, the bfp8 activation broadcast switch, the grid-aware factory and its A/B presets). No device: the builders only
produce ttnn config objects. Their device numerics / timings are covered by tests/perf/test_config_candidates.py and
tests/perf/test_prefill_matmul_candidates.py (torch fp32 reference) and the component tests.

    pytest models/demos/solar_open/tests/unit/test_p0_program_configs.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ttnn
from models.demos.solar_open.tt.attention_configs import SolarOpenAttentionProgramConfig
from models.demos.solar_open.tt.expert_configs import (
    DECODE_EGP_ENV,
    DECODE_EGP_LEGACY,
    DECODE_EGP_PRESETS,
    DENSE_EXPERT_GATE_UP_MINIMAL,
    DENSE_EXPERT_GATE_UP_MINIMAL_ALT,
    HOT_DOWN_KCONCAT_MINIMAL,
    PREFILL_EXPERT_MM_ENV,
    PREFILL_EXPERT_MM_PRESETS,
    SolarOpenProgramConfig,
    decode_egp_overrides,
    prefill_expert_mm_overrides,
    solar_open_program_config,
)
from models.demos.solar_open.tt.experts import decode as experts_decode
from models.demos.solar_open.tt.experts.config import MinimalMatmulBlocking, ProgramConfig, SparseMatmulConfig
from models.demos.solar_open.tt.experts.prefill import (
    HOT_DOWN_KCONCAT,
    _dense_down_program_config,
    _dense_expert_gate_up_config,
    _hot_down_kconcat_minimal_config,
)
from models.demos.solar_open.tt.linear_configs import grid_fits, mcast_1d_linear_config
from models.demos.solar_open.tt.rms_norm import DECODE_NORM_GRID, decode_norm_applies, decode_norm_sharded_configs
from models.demos.solar_open.tt.shared_expert import (
    SHARED_EXPERT_CONFIG_MAX_ROWS,
    SHARED_EXPERT_DOWN_CORES,
    SHARED_EXPERT_GATE_UP_CORES,
    shared_expert_program_configs,
)
from models.demos.solar_open.tt.topk import _LINEAR_CONFIG_MAX_TOKENS, router_linear_program_config

H, IP, E, QKV_N = 4096, 160, 128, 1280
BH_GRID = ttnn.CoreCoord(11, 10)  # P150 compute grid on the 8-card box
WH_GRID = ttnn.CoreCoord(8, 8)
SMALL_GRID = ttnn.CoreCoord(4, 4)


class TestMcast1DLinearConfig:
    def test_gate_up_shape(self):
        cfg = mcast_1d_linear_config((5, 1), 1, IP, H, 32)
        assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (5, 1)
        assert cfg.per_core_N == 1 and cfg.per_core_M == 1 and cfg.in0_block_w == 32
        assert cfg.out_subblock_h == 1 and cfg.out_subblock_w == 1
        assert cfg.out_block_h == 1 and cfg.out_block_w == 1
        assert cfg.mcast_in0 and not cfg.fuse_batch

    @pytest.mark.parametrize("rows, mt", [(1, 1), (8, 1), (32, 1), (33, 2), (128, 4)])
    def test_rows_to_tiles(self, rows, mt):
        cfg = mcast_1d_linear_config((5, 1), rows, IP, H, 32)
        assert cfg.per_core_M == mt and cfg.out_block_h == mt

    def test_down_shape(self):
        cfg = mcast_1d_linear_config((8, 8), 32, H, IP, 5, out_subblock_w=2)
        assert cfg.per_core_N == 2 and cfg.out_subblock_w == 2 and cfg.out_block_w == 2 and cfg.in0_block_w == 5

    def test_in0_block_w_snaps_to_a_divisor_of_kt(self):
        assert mcast_1d_linear_config((8, 8), 32, H, IP, 32).in0_block_w == 5  # Kt = 5 is prime
        assert (
            mcast_1d_linear_config((8, 8), 32, H, 1280, 32).in0_block_w == 20
        )  # Kt = 40: 20 is the largest divisor <= 32

    def test_out_subblock_w_snaps_and_respects_fp32_budget(self):
        assert mcast_1d_linear_config((4, 4), 32, H, IP, 5, out_subblock_w=6).out_subblock_w == 4  # per_core_N 8 -> 4
        assert mcast_1d_linear_config((4, 4), 32, H, IP, 5, out_subblock_w=8).out_subblock_w == 8
        assert mcast_1d_linear_config((4, 4), 32, H, IP, 5, out_subblock_w=8, fp32_dest_acc=True).out_subblock_w == 4

    def test_rejects_indivisible_n(self, expect_error):
        with expect_error(ValueError, "divisible"):
            mcast_1d_linear_config((4, 1), 32, IP, H, 32)  # 5 tiles over 4 cores

    def test_grid_fits(self):
        assert grid_fits((8, 8), BH_GRID) and grid_fits((8, 8), WH_GRID) and not grid_fits((8, 8), SMALL_GRID)
        assert not grid_fits((5, 1), None)


class TestSharedExpertConfigs:
    @pytest.mark.parametrize("rows", [1, 32, 128])
    def test_decode_and_prefill_128_rows_get_configs(self, rows):
        gate_up, down = shared_expert_program_configs(rows, H, IP, BH_GRID)
        assert gate_up is not None and down is not None
        assert (gate_up.compute_with_storage_grid_size.x, gate_up.compute_with_storage_grid_size.y) == (5, 1)
        assert gate_up.per_core_N == 1 and gate_up.in0_block_w == 32 and gate_up.per_core_M == max(1, rows // 32)
        assert (down.compute_with_storage_grid_size.x, down.compute_with_storage_grid_size.y) == (8, 8)
        assert down.per_core_N == 2 and down.in0_block_w == 5 and down.out_subblock_w == 2

    def test_long_prefill_rows_stay_auto(self):
        assert shared_expert_program_configs(SHARED_EXPERT_CONFIG_MAX_ROWS + 1, H, IP, BH_GRID) == (None, None)
        assert shared_expert_program_configs(1024, H, IP, BH_GRID) == (None, None)
        assert shared_expert_program_configs(32, H, IP, BH_GRID, max_rows=0) == (None, None)  # A/B switch

    def test_small_grid_falls_back_per_linear(self):
        gate_up, down = shared_expert_program_configs(32, H, IP, SMALL_GRID)
        assert gate_up is None  # (5, 1) does not fit a 4-wide grid
        assert down is None  # (8, 8) does not fit
        assert (5, 1) == SHARED_EXPERT_GATE_UP_CORES and (8, 8) == SHARED_EXPERT_DOWN_CORES

    def test_tp1_shapes_still_build(self):
        # TP=1: I = 1280 (40 tiles over 5 cores = 8 per core), down K = 1280 (Kt 40: in0_block_w 5 divides it)
        gate_up, down = shared_expert_program_configs(32, H, 1280, WH_GRID)
        assert gate_up.per_core_N == 8 and down.in0_block_w == 5


class TestRouterLinearConfig:
    @pytest.mark.parametrize("tokens", [1, 8, 32, 128])
    def test_short_token_counts_get_the_4x1_config(self, tokens):
        cfg = router_linear_program_config(tokens, E, H, BH_GRID)
        assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (4, 1)
        assert cfg.per_core_N == 1 and cfg.in0_block_w == 32 and cfg.out_subblock_w == 1
        assert cfg.per_core_M == max(1, tokens // 32)

    def test_long_prefills_stay_auto(self):
        assert router_linear_program_config(_LINEAR_CONFIG_MAX_TOKENS + 1, E, H, BH_GRID) is None
        assert router_linear_program_config(1024, E, H, BH_GRID) is None
        assert router_linear_program_config(32, E, H, BH_GRID, max_tokens=0) is None  # A/B switch

    def test_grid_too_narrow_stays_auto(self):
        assert router_linear_program_config(32, E, H, ttnn.CoreCoord(3, 10)) is None


class TestDecodeNormConfigs:
    def test_8x4_shards_of_128(self):
        assert DECODE_NORM_GRID == (8, 4)
        mem, pc = decode_norm_sharded_configs(H, BH_GRID)
        assert mem is not None and pc is not None
        assert tuple(mem.shard_spec.shape) == (32, 128)
        assert mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and mem.buffer_type == ttnn.BufferType.L1
        assert pc.block_h == 1 and pc.block_w == 4 and pc.subblock_w == 4 and not pc.inplace
        assert (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y) == (8, 4)

    def test_fits_wormhole_grid(self):
        mem, pc = decode_norm_sharded_configs(H, WH_GRID)
        assert mem is not None and pc is not None

    @staticmethod
    def _tensor(shape, padded, dtype=ttnn.bfloat16, sharded=False):
        return SimpleNamespace(
            shape=tuple(shape),
            padded_shape=tuple(padded),
            dtype=dtype,
            memory_config=lambda: SimpleNamespace(is_sharded=lambda: sharded),
        )

    @pytest.mark.parametrize(
        "shape, padded, applies",
        [
            ((1, 1, 1, H), (1, 1, 32, H), True),  # decode b1
            ((1, 1, 32, H), (1, 1, 32, H), True),  # decode b32
            ((1, 1, 16, H), (1, 1, 32, H), True),  # decode b16
            ((1, 1, 128, H), (1, 1, 128, H), False),  # prefill: default kernel
            ((1, 1, 33, H), (1, 1, 64, H), False),
            ((32, 1, H), (32, 32, H), False),  # the rms_norm component test's HF-shaped decode_b32 input: 32 tile
            #                                     rows (physical height 1024) -- the worktree ladder's TT_FATAL shape
            ((1, 32, 1, H), (1, 32, 32, H), False),
            ((1, 1, 1, H + 32), (1, 1, 32, H + 32), False),  # not this norm's width
        ],
    )
    def test_decode_norm_applies_only_to_one_padded_tile_row(self, shape, padded, applies):
        assert decode_norm_applies(self._tensor(shape, padded), H) is applies

    def test_decode_norm_needs_interleaved_bf16(self):
        assert not decode_norm_applies(self._tensor((1, 1, 32, H), (1, 1, 32, H), dtype=ttnn.bfloat8_b), H)
        assert not decode_norm_applies(self._tensor((1, 1, 32, H), (1, 1, 32, H), sharded=True), H)

    def test_disabled_or_too_small(self):
        assert decode_norm_sharded_configs(H, SMALL_GRID) == (None, None)
        import models.demos.solar_open.tt.rms_norm as rms_norm_module

        saved = rms_norm_module.DECODE_NORM_GRID
        rms_norm_module.DECODE_NORM_GRID = None  # module A/B switch
        try:
            assert decode_norm_sharded_configs(H, BH_GRID) == (None, None)
        finally:
            rms_norm_module.DECODE_NORM_GRID = saved
        assert decode_norm_sharded_configs(H, None) == (None, None)
        assert decode_norm_sharded_configs(4096 + 32, BH_GRID) == (None, None)  # not 32 x 32-aligned


class TestAttentionQkvConfig:
    def test_default_is_wired(self):
        apc = SolarOpenAttentionProgramConfig()
        assert apc.decode_qkv_cores == (8, 5) and apc.decode_qkv_in0_block_w in (8, 16)
        cfg = apc.get_decode_qkv_config(32, QKV_N, H)
        assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (8, 5)
        assert cfg.per_core_N == 1 and cfg.per_core_M == 1 and cfg.in0_block_w == apc.decode_qkv_in0_block_w
        assert cfg.fuse_batch and cfg.mcast_in0
        assert apc.decode_out_cores == (
            8,
            8,
        )  # phase 3e / P1: the (8, 8) o_proj from the interleaved in0 (bit-identical, faster)
        assert SolarOpenAttentionProgramConfig(tp=1).decode_out_cores is None  # TP = 1 keeps the auto o_proj

    def test_compute_config_restates_hifi2_with_bf16_dst(self):
        # bf16 destination by default: the fp32-dst variant is closer to the fp32 reference on the qkv op alone
        # (0.999994 vs 0.999862) but measured slightly worse on the whole model (teacher-forced b1 top-1 0.9219 vs
        # 0.9258, KL 0.0333 vs 0.0300) and interacts with the sharded decode norm in the random 1-layer test_model
        # (decode_b1_s1 logits PCC 0.99769 vs 0.99935), at identical traced time -> True is the A/B switch.
        apc = SolarOpenAttentionProgramConfig()
        assert not apc.decode_qkv_fp32_dest_acc
        cc = apc.get_decode_qkv_compute_config(ttnn.device.Arch.BLACKHOLE)
        assert cc.math_fidelity == ttnn.MathFidelity.HiFi2 and not cc.math_approx_mode
        assert not cc.fp32_dest_acc_en and cc.packer_l1_acc
        cc32 = SolarOpenAttentionProgramConfig(decode_qkv_fp32_dest_acc=True).get_decode_qkv_compute_config(
            ttnn.device.Arch.BLACKHOLE
        )
        assert cc32.math_fidelity == ttnn.MathFidelity.HiFi2 and cc32.fp32_dest_acc_en and cc32.packer_l1_acc

    def test_none_switch(self):
        assert SolarOpenAttentionProgramConfig(decode_qkv_cores=None).get_decode_qkv_config(32, QKV_N, H) is None


class TestDenseDownConfig:
    def test_default_and_gating(self):
        pc = SolarOpenProgramConfig()
        assert pc.dense_down_cores == (8, 8)
        for rows in (32, 128, 160, 192, 256):
            cfg = _dense_down_program_config(pc, rows, H, IP)
            assert cfg is not None and cfg.per_core_M == rows // 32 and cfg.in0_block_w == 5 and cfg.per_core_N == 2
        assert _dense_down_program_config(pc, 512, H, IP) is None  # above dense_bmm_max_tokens: auto
        assert _dense_down_program_config(pc, 100, H, IP) is None  # not a tile multiple
        assert _dense_down_program_config(SolarOpenProgramConfig(dense_down_cores=None), 128, H, IP) is None
        assert _dense_down_program_config(None, 128, H, IP) is None

    @pytest.mark.parametrize(
        "rows, sub_h", [(32, 1), (64, 2), (96, 3), (128, 4), (160, 1), (192, 3), (224, 1), (256, 4)]
    )
    def test_phase3_tall_subblock(self, rows, sub_h):
        """Phase 3: out_subblock_h = the tallest divisor of Mt <= 4 with h x 2 <= 8 dst tiles (bit-identical, -8 %
        at 96-128 rows); dense_down_max_subblock_h 1 restores the phase-2 one-row subblock, everything else equal."""
        pc = SolarOpenProgramConfig()
        assert pc.dense_down_max_subblock_h == 4
        cfg = pc.get_dense_down_config(rows, H, IP)
        assert cfg.out_subblock_h == sub_h and cfg.out_subblock_w == 2
        assert cfg.out_block_h == rows // 32 and cfg.out_block_w == 2
        p2 = SolarOpenProgramConfig(dense_down_max_subblock_h=1).get_dense_down_config(rows, H, IP)
        assert p2.out_subblock_h == 1
        for attr in ("in0_block_w", "out_subblock_w", "out_block_h", "out_block_w", "per_core_M", "per_core_N"):
            assert getattr(p2, attr) == getattr(cfg, attr), attr

    def test_subblock_budget_with_wider_subblock_w(self, expect_error):
        # per_core_N 4 (8x4 cores): out_subblock_w 4 -> h <= 2 at Mt 4, h 1 at Mt 3
        pc = SolarOpenProgramConfig(dense_down_cores=(8, 4))
        assert (
            pc.get_dense_down_config(128, H, IP).out_subblock_h,
            pc.get_dense_down_config(128, H, IP).out_subblock_w,
        ) == (2, 4)
        assert pc.get_dense_down_config(96, H, IP).out_subblock_h == 1
        with expect_error(ValueError, "dense_down_max_subblock_h"):
            SolarOpenProgramConfig(dense_down_max_subblock_h=0)


class TestDecodeExpertGroups:
    """Phase 3b: the expert-group (EGP) decode sparse_matmul configs. ``_build_matmul_config(expert_groups=G)`` fills the
    grid with G x ceil(Nt / per_core_N) cores (exact rectangle, >= 2 cores); the getters return the program config and
    the group count together (SparseMatmulConfig) because neither is legal without the other at the op."""

    @staticmethod
    def _key(cfg):
        pc = cfg.program_config
        g = pc.compute_with_storage_grid_size
        return (g.x, g.y), pc.per_core_N, pc.out_subblock_w, pc.out_block_w, pc.in0_block_w, cfg.expert_groups

    def test_shipped_defaults(self):
        pc = SolarOpenProgramConfig()
        assert (pc.decode_gate_up_cores, pc.decode_gate_up_expert_groups) == ((11, 10), 11)
        assert (pc.decode_down_cores_batched, pc.decode_down_batched_subblock_w) == ((11, 8), 8)
        assert (pc.decode_down_batched_expert_groups, pc.decode_down_batched_min_tokens) == (11, 2)
        assert (pc.decode_down_cores, pc.decode_down_subblock_w, pc.decode_down_expert_groups) == ((8, 4), 4, None)
        assert (pc.decode_down_indexed_cores, pc.decode_down_indexed_subblock_w) == ((8, 8), 2)
        assert pc.decode_down_indexed_expert_groups is None
        # gate|up: 10 output blocks of 1 tile x 11 groups = the whole 11x10 grid, the whole K as one block
        for rows in (1, 32):
            assert self._key(pc.get_decode_gate_up_config(rows, 2 * IP, k=H)) == ((11, 10), 1, 1, 1, 128, 11)
        # batched down: 8 blocks of 16 tiles x 11 groups = 11x8, out_subblock_w 8 (the widest legal), Kt 5 as one block
        for users in (2, 8, 16, 32):
            assert self._key(pc.get_decode_down_config(users, H, k=IP)) == ((11, 8), 16, 8, 8, 5, 11)
        # single-user scan-path down: the legacy 8x4 x 4 tiles, no groups
        assert self._key(pc.get_decode_down_config(1, H, k=IP)) == ((8, 4), 4, 4, 4, 5, None)
        # the indexed compact-A down (the shipped b1 path, phase 3c): the legacy 8x8 x 2 tiles, out_subblock_w 2, no groups
        assert self._key(pc.get_decode_down_config(1, H, k=IP, indexed=True)) == ((8, 8), 2, 2, 2, 5, None)
        assert isinstance(pc.get_decode_gate_up_config(32, 2 * IP, k=H), SparseMatmulConfig)

    def test_indexed_down_grid(self, expect_error):
        """Phase 3c: ``get_decode_down_config(indexed=True)`` takes the decode_down_indexed_* values when set, else the
        single-user ones (never the batched grid); it serves one token only."""
        pc = SolarOpenProgramConfig()
        with expect_error(ValueError, "exactly one token"):
            pc.get_decode_down_config(2, H, k=IP, indexed=True)
        # None = the single-user (scan-path) values, whatever the batched threshold says
        pc_off = SolarOpenProgramConfig(decode_down_indexed_cores=None, decode_down_batched_min_tokens=1)
        assert self._key(pc_off.get_decode_down_config(1, H, k=IP, indexed=True)) == ((8, 4), 4, 4, 4, 5, None)
        assert self._key(pc_off.get_decode_down_config(1, H, k=IP)) == ((11, 8), 16, 8, 8, 5, 11)
        # the indexed grid resolves under the same exact-fill rule (identity for the shipped values, shrink otherwise)
        pc_egp = SolarOpenProgramConfig(
            decode_down_indexed_cores=(11, 8), decode_down_indexed_subblock_w=8, decode_down_indexed_expert_groups=11
        )
        assert self._key(pc_egp.get_decode_down_config(1, H, k=IP, indexed=True)) == ((11, 8), 16, 8, 8, 5, 11)
        with expect_error(ValueError, "decode_down_indexed_cores"):
            SolarOpenProgramConfig(decode_down_indexed_cores=(0, 8))

    @pytest.mark.parametrize(
        "cores, n, groups, expected",
        [
            ((11, 10), 2 * IP, 11, ((11, 10), 1)),  # 11 x 10 blocks
            ((10, 10), 2 * IP, 5, ((10, 5), 1)),  # 5 x 10 blocks: the tallest fill of a 10-wide grid is 10x5
            ((11, 10), 2 * IP, 2, ((10, 2), 1)),  # 20 cores: 10x2 (the wider shape wins the tie with 5x4 / 4x5 / 2x10)
            ((11, 10), 2 * IP, 1, ((10, 1), 1)),  # G 1 keeps the exact-fill rule of the legacy search
            ((11, 10), 2 * IP, None, ((10, 1), 1)),
            ((11, 8), H, 11, ((11, 8), 16)),  # 11 x 8 blocks of 16 tiles
            ((8, 10), H, 5, ((8, 10), 8)),  # 5 x 16 blocks of 8 tiles
            ((8, 8), H, 4, ((8, 8), 8)),  # 4 x 16 blocks
            ((8, 8), H, 2, ((8, 8), 4)),  # 2 x 32 blocks of 4 tiles
            ((11, 10), H, 11, ((11, 10), 13)),  # 11 x 10 blocks of 13 tiles (the last block holds 11): legal, osw 1
        ],
    )
    def test_exact_fill_with_groups(self, cores, n, groups, expected):
        cfg = ProgramConfig()._build_matmul_config(
            cores, 32, n, in0_block_w=5, out_subblock_w=8, k=IP, expert_groups=groups
        )
        g = cfg.compute_with_storage_grid_size
        assert ((g.x, g.y), cfg.per_core_N) == expected
        # the factory's rectangle rule: G x blocks cores row-major from (0, 0) fill whole rows of the config grid
        blocks = -(-(-(-n // 32)) // cfg.per_core_N)
        assert (groups or 1) * blocks == g.x * g.y and g.x * g.y >= 2
        assert cfg.per_core_N % cfg.out_subblock_w == 0 and cfg.out_block_w == cfg.out_subblock_w

    def test_no_fill_raises(self, expect_error):
        pc = ProgramConfig()
        with expect_error(ValueError, "expert_groups=11"):
            pc._build_matmul_config((8, 8), 32, 2 * IP, k=H, expert_groups=11)  # 11 does not divide any w*h <= 8x8
        with expect_error(ValueError, "expert_groups=13"):
            pc._build_matmul_config((11, 10), 32, 2 * IP, k=H, expert_groups=13)  # no multiple of 13 fits 11x10
        # a group count that does not fill the requested grid shrinks it to a sub-rectangle it does fill: 7 x 10 blocks
        # = 70 cores as 7 full rows of 10 (the wider shape wins the tie with 7x10)
        cfg = pc._build_matmul_config((11, 10), 32, 2 * IP, k=H, expert_groups=7)
        assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y, cfg.per_core_N) == (
            10,
            7,
            1,
        )
        with expect_error(ValueError, "single-core"):
            pc._build_matmul_config((1, 1), 32, 32, k=H, expert_groups=1)

    def test_field_validation(self, expect_error):
        for name in (
            "decode_gate_up_expert_groups",
            "decode_down_expert_groups",
            "decode_down_indexed_expert_groups",
            "decode_down_batched_expert_groups",
        ):
            for bad in (0, -1, 2.0, True, "11"):
                with expect_error(ValueError, name):
                    SolarOpenProgramConfig(**{name: bad})
        assert SolarOpenProgramConfig(decode_gate_up_expert_groups=None).decode_gate_up_expert_groups is None

    def test_legacy_preset_reproduces_phase2(self, monkeypatch, expect_error):
        assert set(DECODE_EGP_PRESETS) == {"on", "p3b", "off"} and DECODE_EGP_PRESETS["off"] is DECODE_EGP_LEGACY
        assert decode_egp_overrides("on") == {} and decode_egp_overrides("") == {}
        monkeypatch.setenv(DECODE_EGP_ENV, "off")
        assert decode_egp_overrides() == DECODE_EGP_LEGACY
        pc = SolarOpenProgramConfig(**decode_egp_overrides())
        assert self._key(pc.get_decode_gate_up_config(32, 2 * IP, k=H)) == ((5, 2), 1, 1, 1, 128, None)
        assert self._key(pc.get_decode_down_config(8, H, k=IP)) == ((8, 4), 4, 4, 4, 5, None)
        assert self._key(pc.get_decode_down_config(16, H, k=IP)) == ((8, 8), 2, 2, 2, 5, None)
        assert self._key(pc.get_decode_down_config(32, H, k=IP)) == ((8, 8), 2, 2, 2, 5, None)
        # phase 2 ran the indexed down on the single-user 8x4 grid
        assert self._key(pc.get_decode_down_config(1, H, k=IP, indexed=True)) == ((8, 4), 4, 4, 4, 5, None)
        # p3b = phase 3b: EGP on, only the indexed down back on the 8x4 grid (the phase-3c A/B arm)
        monkeypatch.setenv(DECODE_EGP_ENV, "p3b")
        assert decode_egp_overrides() == {"decode_down_indexed_cores": None}
        pc = SolarOpenProgramConfig(**decode_egp_overrides())
        assert self._key(pc.get_decode_gate_up_config(1, 2 * IP, k=H)) == ((11, 10), 1, 1, 1, 128, 11)
        assert self._key(pc.get_decode_down_config(32, H, k=IP)) == ((11, 8), 16, 8, 8, 5, 11)
        assert self._key(pc.get_decode_down_config(1, H, k=IP, indexed=True)) == ((8, 4), 4, 4, 4, 5, None)
        monkeypatch.setenv(DECODE_EGP_ENV, "nonsense")
        with expect_error(ValueError, "preset"):
            decode_egp_overrides()

    @pytest.mark.parametrize("grid_xy, egp", [((11, 10), True), ((13, 10), True), ((10, 10), False), ((11, 9), False)])
    def test_factory_grid_gate(self, grid_xy, egp, monkeypatch):
        monkeypatch.delenv(DECODE_EGP_ENV, raising=False)
        mesh_device = MagicMock()
        mesh_device.compute_with_storage_grid_size.return_value = SimpleNamespace(x=grid_xy[0], y=grid_xy[1])
        pc = solar_open_program_config(mesh_device)
        expected = SolarOpenProgramConfig() if egp else SolarOpenProgramConfig(**DECODE_EGP_LEGACY)
        for name in DECODE_EGP_LEGACY:
            assert getattr(pc, name) == getattr(expected, name), name
        # the down / gate|up configs of both arms resolve on their own grids (identity, no shrink)
        gx, gy = pc.get_decode_gate_up_config(32, 2 * IP, k=H).program_config.compute_with_storage_grid_size.x, None
        assert gx == (11 if egp else 5)

    def test_decode_passes_program_config_and_groups_together(self):
        """Source contract: every decode sparse_matmul takes program_config AND expert_groups from one
        SparseMatmulConfig (an expert-group grid is illegal without its group count and vice versa)."""
        import inspect
        import re

        for fn in (
            experts_decode.decode_forward,
            experts_decode._decode_forward_indexed,
            experts_decode._decode_forward_batched,
        ):
            src = inspect.getsource(fn)
            calls = len(re.findall(r"ttnn\.sparse_matmul\(", src))
            assert calls == 2, fn.__name__
            assert (
                len(re.findall(r"program_config=(\w+_cfg)\.program_config,\n\s+expert_groups=\1\.expert_groups,", src))
                == 2
            ), fn.__name__


class TestMinimalMatmulBlocking:
    def test_gate_up_default_reproduces_the_measured_config(self):
        b = SolarOpenProgramConfig().dense_expert_gate_up_minimal
        assert b == DENSE_EXPERT_GATE_UP_MINIMAL == MinimalMatmulBlocking(cores=(11, 5), k_block=16, subblock=(3, 2))
        # [1024, 4096] x [4096, 320]: M > N -> M over the 11 columns (33 padded tiles: 3 per core), N over 5 rows
        assert b.blocks(1024, H, 2 * IP) == (3, 16, 2, 3, 2)
        cfg = b.config(1024, H, 2 * IP)
        assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size, cfg.subblock_h, cfg.subblock_w) == (
            3,
            16,
            2,
            3,
            2,
        )
        assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (11, 5)
        # the 512-token splits above 32K context: 16 M tiles over 11 columns -> 2 per core, subblock snaps to 2x2
        assert b.blocks(512, H, 2 * IP) == (2, 16, 2, 2, 2)
        # the better-numerics alternative: K blocks of 8 on the full grid, N_block 1
        assert DENSE_EXPERT_GATE_UP_MINIMAL_ALT.blocks(1024, H, 2 * IP) == (3, 8, 1, 3, 1)

    @pytest.mark.parametrize("n_hot", [1, 2, 4, 8, 15, 16])
    def test_hot_down_kconcat_default(self, n_hot):
        b = SolarOpenProgramConfig().hot_down_kconcat_minimal
        assert b == HOT_DOWN_KCONCAT_MINIMAL == MinimalMatmulBlocking(cores=(11, 10), k_block=5, subblock=(4, 2))
        # [1024, n_hot * 160] x [n_hot * 160, 4096]: M < N -> M over the 10 rows (4 per core), N over 11 columns (12)
        assert b.blocks(1024, n_hot * IP, H) == (4, 5, 12, 4, 2)
        assert b.blocks(512, n_hot * IP, H) == (2, 5, 12, 2, 2)

    def test_snapping_rules(self, expect_error):
        # K block snaps to a divisor of Kt; the subblock to divisors of the blocks within the dst budget
        b = MinimalMatmulBlocking(cores=(11, 10), k_block=16, subblock=(4, 8))
        assert b.blocks(1024, 5 * 32, H)[1] == 5  # Kt = 5 is prime
        assert b.blocks(1024, 96, H)[1] == 3  # Kt = 3
        m_block, _, n_block, sub_h, sub_w = b.blocks(1024, H, H)  # M == N: no transpose, M over 10 rows, N over 11
        assert (m_block, n_block) == (4, 12)
        assert (sub_h, sub_w) == (4, 2)  # 8 does not divide 12 -> 6; 4 x 6 > 8 dst tiles -> narrow to 4, 3, 2
        assert MinimalMatmulBlocking(cores=(11, 10), k_block=5, subblock=(4, 2), dst_tiles=4).blocks(1024, IP, H)[
            3:
        ] == (4, 1)
        with expect_error(ValueError, "cores"):
            MinimalMatmulBlocking(cores=(1, 10), k_block=5)  # the kernel needs >= 2x2
        with expect_error(ValueError, "k_block"):
            MinimalMatmulBlocking(cores=(11, 10), k_block=0)
        with expect_error(ValueError, "hot_down_kconcat_minimal"):
            SolarOpenProgramConfig(hot_down_kconcat_minimal=(11, 10))

    def test_fits(self):
        assert HOT_DOWN_KCONCAT_MINIMAL.fits(BH_GRID) and HOT_DOWN_KCONCAT_MINIMAL.fits((13, 10))
        assert not HOT_DOWN_KCONCAT_MINIMAL.fits(WH_GRID) and not DENSE_EXPERT_GATE_UP_MINIMAL.fits((8, 8))
        assert DENSE_EXPERT_GATE_UP_MINIMAL.fits((11, 5)) and not DENSE_EXPERT_GATE_UP_MINIMAL.fits((11, 4))
        assert not DENSE_EXPERT_GATE_UP_MINIMAL.fits(None)

    def test_getters_and_prefill_helpers(self):
        pc = SolarOpenProgramConfig()
        dense_grid = SimpleNamespace(x=11, y=10)  # ttnn.CoreGrid of _dense_core_grid on P150
        cfg = _dense_expert_gate_up_config(pc, dense_grid, 1024, H, 2 * IP)
        assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size) == (3, 16, 2)
        cfg = _hot_down_kconcat_minimal_config(pc, dense_grid, 1024, 15 * IP, H)
        assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size, cfg.subblock_h, cfg.subblock_w) == (
            4,
            5,
            12,
            4,
            2,
        )
        # the auto forms: no program config, a grid the blocking does not fit, or the field unset
        assert _dense_expert_gate_up_config(None, dense_grid, 1024, H, 2 * IP) is None
        assert _dense_expert_gate_up_config(pc, SimpleNamespace(x=8, y=8), 1024, H, 2 * IP) is None
        assert _hot_down_kconcat_minimal_config(pc, SimpleNamespace(x=8, y=8), 1024, IP, H) is None
        off = SolarOpenProgramConfig(dense_expert_gate_up_minimal=None, hot_down_kconcat_minimal=None)
        assert off.get_dense_expert_gate_up_config(1024, H, 2 * IP) is None
        assert off.get_hot_down_kconcat_config(1024, IP, H) is None
        assert pc.get_dense_expert_gate_up_config(1024, H, 2 * IP, grid=None) is not None  # no grid check
        # the base config keeps the current (auto) behaviour everywhere
        base = ProgramConfig()
        assert base.dense_expert_gate_up_minimal is None and base.hot_down_kconcat_minimal is None
        assert base.dense_down_max_subblock_h == 1 and base.dense_activation_bfp8 is False

    def test_numerics_switch_default_off_and_legacy_switch_untouched(self):
        assert SolarOpenProgramConfig().dense_activation_bfp8 is False
        assert HOT_DOWN_KCONCAT is False  # the perf-p1 2D K-concat stays the measured alternative


class TestSolarOpenFactory:
    @staticmethod
    def _mesh(x, y):
        mesh_device = MagicMock()
        mesh_device.compute_with_storage_grid_size.return_value = SimpleNamespace(x=x, y=y)
        return mesh_device

    @pytest.mark.parametrize("grid_xy", [(11, 10), (13, 10)], ids=["bh_11x10", "bh_13x10"])
    def test_blackhole_grids_keep_the_minimal_blockings(self, grid_xy, monkeypatch):
        monkeypatch.delenv(PREFILL_EXPERT_MM_ENV, raising=False)
        pc = solar_open_program_config(self._mesh(*grid_xy))
        assert pc.dense_expert_gate_up_minimal == DENSE_EXPERT_GATE_UP_MINIMAL
        assert pc.hot_down_kconcat_minimal == HOT_DOWN_KCONCAT_MINIMAL
        assert pc.dense_down_max_subblock_h == 4 and pc.dense_down_cores == (8, 8)

    @pytest.mark.parametrize("grid_xy", [(8, 8), (10, 10), (7, 8)], ids=["wh_8x8", "10x10", "narrow_7x8"])
    def test_narrow_grids_drop_to_the_auto_forms(self, grid_xy, monkeypatch):
        monkeypatch.delenv(PREFILL_EXPERT_MM_ENV, raising=False)
        pc = solar_open_program_config(self._mesh(*grid_xy))
        assert pc.dense_expert_gate_up_minimal is None and pc.hot_down_kconcat_minimal is None
        assert pc.dense_down_max_subblock_h == 4  # harmless without dense_down_cores; bit-identical with it

    def test_presets(self, monkeypatch, expect_error):
        assert set(PREFILL_EXPERT_MM_PRESETS) == {"tuned", "phase2", "gate_up_alt", "bfp8_act"}
        assert prefill_expert_mm_overrides("tuned") == {} and prefill_expert_mm_overrides("") == {}
        monkeypatch.setenv(PREFILL_EXPERT_MM_ENV, "phase2")
        pc = solar_open_program_config(self._mesh(11, 10))
        assert pc.dense_expert_gate_up_minimal is None and pc.hot_down_kconcat_minimal is None
        assert pc.dense_down_max_subblock_h == 1 and pc.dense_down_cores == (8, 8)
        monkeypatch.setenv(PREFILL_EXPERT_MM_ENV, "gate_up_alt")
        assert (
            solar_open_program_config(self._mesh(11, 10)).dense_expert_gate_up_minimal
            == DENSE_EXPERT_GATE_UP_MINIMAL_ALT
        )
        monkeypatch.setenv(PREFILL_EXPERT_MM_ENV, "bfp8_act")
        pc = solar_open_program_config(self._mesh(11, 10))
        assert pc.dense_activation_bfp8 is True and pc.hot_down_kconcat_minimal == HOT_DOWN_KCONCAT_MINIMAL
        monkeypatch.setenv(PREFILL_EXPERT_MM_ENV, "nonsense")
        with expect_error(ValueError, "preset"):
            solar_open_program_config(self._mesh(11, 10))
