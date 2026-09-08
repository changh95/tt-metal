# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of the phase-2 (perf-p0) program-config builders and their ON defaults in the merged main tree: the
explicit 1D matmul configs of the shared expert and the router, the width-sharded decode norm (and the padded-row
gate that keeps it off the shapes it cannot shard), the attention decode qkv config with its HiFi2 (bf16-dst) compute
config and the dense prefill down config. No device: the builders only produce ttnn config objects. Their device
numerics / timings are covered by tests/perf/test_config_candidates.py (torch fp32 reference) and the component tests.

    pytest models/demos/solar_open/tests/unit/test_p0_program_configs.py
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.solar_open.tt.attention_configs import SolarOpenAttentionProgramConfig
from models.demos.solar_open.tt.expert_configs import SolarOpenProgramConfig
from models.demos.solar_open.tt.experts.prefill import _dense_down_program_config
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
        assert apc.decode_out_cores is None  # o_proj stays auto (no gain on the width-sharded input)

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
