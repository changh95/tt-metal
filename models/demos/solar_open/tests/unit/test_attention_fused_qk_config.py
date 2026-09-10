# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks of the phase-3e (lane B1) decode attention levers -- no device is opened (``ttnn`` only for the core
coordinate types; the fake mesh device only answers ``compute_with_storage_grid_size``).

  * ``fused_qk``: the knob resolution (``SOLAR_OPEN_ATTENTION_FUSED_QK``), the doubled-batch per-user grid rule of
    ``ProgramConfig.get_decode_user_grid(fused_qk=True)`` and ``get_decode_qk_fused_grids`` -- the 2B-core grid handed
    to ``nlp_create_qkv_heads_decode(overlap_qk_coregrid=False)`` must tile into Q = cores [0, B) and K = cores [B, 2B)
    in row-major order (the op's ``compute_output_specs`` derivation, mirrored here), the Q cores must be the SDPA
    reducer cores ``(b % grid.x, b // grid.x)`` and the whole grid must be ``RotarySetup(use_qk_fused=True).batch_grid``
    (rope.py ``get_batch_grid`` on the doubled batch: 8x8 when 2B % 32 == 0, the device grid otherwise).
  * o_proj: ``SOLAR_OPEN_ATTENTION_OUT_GRID`` parsing, the (8, 8) 1D config values (per_core_N 2, in0_block_w 4,
    out_subblock_w 2, fuse_batch, mcast_in0) and the HiFi2 compute config restated for it.

    pytest models/demos/solar_open/tests/unit/test_attention_fused_qk_config.py
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.solar_open.tt.attention.config import ProgramConfig, _cores_row_major
from models.demos.solar_open.tt.attention_configs import (
    ATTENTION_FUSED_QK_DEFAULT,
    ATTENTION_FUSED_QK_ENV,
    ATTENTION_OUT_GRID_DEFAULT,
    ATTENTION_OUT_GRID_ENV,
    AUTO,
    FROM_ENV,
    SolarOpenAttentionProgramConfig,
    attention_fused_qk_from_env,
    attention_out_grid_from_env,
    resolve_fused_qk,
    resolve_out_grid,
)

H, O_K, N_HEADS = 4096, 1024, 8  # per device: o_proj K = 8 heads x 128, N = hidden


def fake_device(x, y):
    return SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(x, y))


BH_11x10 = fake_device(11, 10)  # the P150x8 box
BH_13x10 = fake_device(13, 10)  # P100 / the grid the docstrings quote
WH_8x8 = fake_device(8, 8)


def grid_cores(w, h):
    return [(x, y) for y in range(h) for x in range(w)]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ATTENTION_FUSED_QK_ENV, raising=False)
    monkeypatch.delenv(ATTENTION_OUT_GRID_ENV, raising=False)


class TestKnobs:
    def test_defaults(self):
        # phase 3e / P1: both levers ON for TP > 1 (bit-identical to the phase-3d chain, faster) and OFF at TP = 1 (no
        # width-sharded qkv there; the (8, 8) o_proj was measured at the TP = 8 shapes only)
        assert ATTENTION_FUSED_QK_DEFAULT is True and ATTENTION_OUT_GRID_DEFAULT == (8, 8)
        apc = SolarOpenAttentionProgramConfig()  # tp unknown = the production layout
        assert apc.fused_qk is True and apc.decode_out_cores == (8, 8)
        tp8 = SolarOpenAttentionProgramConfig(tp=8)
        assert tp8.fused_qk is True and tp8.decode_out_cores == (8, 8)
        tp1 = SolarOpenAttentionProgramConfig(tp=1)
        assert tp1.fused_qk is False and tp1.decode_out_cores is None
        assert tp1.get_decode_out_config(32, H, O_K) is None
        assert (
            SolarOpenAttentionProgramConfig(decode_out_cores=None).decode_out_cores is None
        )  # the auto o_proj, explicitly
        assert SolarOpenAttentionProgramConfig(decode_out_cores=FROM_ENV).decode_out_cores == (8, 8)
        base = ProgramConfig()
        assert base.fused_qk is False and base.decode_out_interleave_in0 is False and base.decode_out_cores is None

    def test_rejects_unknown_out_cores_sentinel(self, expect_error):
        with expect_error(ValueError, "decode_out_cores"):
            SolarOpenAttentionProgramConfig(decode_out_cores="8x8")

    @pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", None), ("auto", None), ("AUTO", None)])
    def test_fused_qk_env(self, monkeypatch, value, expected):
        monkeypatch.setenv(ATTENTION_FUSED_QK_ENV, value)
        assert attention_fused_qk_from_env() is expected
        effective = ATTENTION_FUSED_QK_DEFAULT if expected is None else expected
        assert resolve_fused_qk(8) is effective and resolve_fused_qk(None) is effective
        assert SolarOpenAttentionProgramConfig().fused_qk is effective
        assert SolarOpenAttentionProgramConfig(tp=8).fused_qk is effective
        # the default rule turns off at TP = 1; an explicit 1 stays on there (the decode chain then raises)
        assert resolve_fused_qk(1) is (expected is True)
        assert SolarOpenAttentionProgramConfig(tp=1).fused_qk is (expected is True)
        # an explicit field value wins over the environment
        assert SolarOpenAttentionProgramConfig(fused_qk=not effective).fused_qk is (not effective)

    @pytest.mark.parametrize("value", ["true", "yes", "2"])
    def test_fused_qk_env_rejects_garbage(self, monkeypatch, value, expect_error):
        monkeypatch.setenv(ATTENTION_FUSED_QK_ENV, value)
        with expect_error(ValueError, ATTENTION_FUSED_QK_ENV):
            attention_fused_qk_from_env()

    @pytest.mark.parametrize(
        "value, expected", [("8x8", (8, 8)), ("8X4", (8, 4)), ("", None), ("auto", AUTO), ("0", AUTO), ("none", AUTO)]
    )
    def test_out_grid_env(self, monkeypatch, value, expected):
        monkeypatch.setenv(ATTENTION_OUT_GRID_ENV, value)
        assert attention_out_grid_from_env() == expected
        effective = ATTENTION_OUT_GRID_DEFAULT if expected is None else (None if expected == AUTO else expected)
        assert resolve_out_grid(8) == effective and resolve_out_grid(None) == effective
        assert SolarOpenAttentionProgramConfig().decode_out_cores == effective
        assert SolarOpenAttentionProgramConfig(tp=8).decode_out_cores == effective
        # the default rule keeps the auto linear at TP = 1; an explicit grid applies there too
        assert resolve_out_grid(1) == (None if expected in (None, AUTO) else expected)
        assert SolarOpenAttentionProgramConfig(tp=1).decode_out_cores == (
            None if expected in (None, AUTO) else expected
        )

    @pytest.mark.parametrize("value", ["8", "8x", "axb", "0x8", "-8x8"])
    def test_out_grid_env_rejects_garbage(self, monkeypatch, value, expect_error):
        monkeypatch.setenv(ATTENTION_OUT_GRID_ENV, value)
        with expect_error(ValueError, ATTENTION_OUT_GRID_ENV):
            attention_out_grid_from_env()

    def test_explicit_out_cores_win(self, monkeypatch):
        monkeypatch.setenv(ATTENTION_OUT_GRID_ENV, "8x4")
        assert SolarOpenAttentionProgramConfig(decode_out_cores=(8, 8)).decode_out_cores == (8, 8)


class TestOutProjConfig:
    def test_8x8_interleaved_config(self):
        apc = SolarOpenAttentionProgramConfig(decode_out_cores=(8, 8))
        assert apc.decode_out_interleave_in0 is True and apc.decode_out_fp32_dest_acc is False
        for m in (1, 32):  # b1 (logical M 1, one tile) and b32
            cfg = apc.get_decode_out_config(m, H, O_K)
            assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (8, 8)
            assert cfg.per_core_N == 2 and cfg.per_core_M == 1  # 128 N tiles over 64 cores; M = one tile row
            assert cfg.in0_block_w == 4 and cfg.out_subblock_w == 2 and cfg.out_subblock_h == 1
            assert cfg.out_block_h == 1 and cfg.out_block_w == 2
            assert cfg.fuse_batch and cfg.mcast_in0 and cfg.fused_activation is None

    def test_compute_config_restates_hifi2(self):
        apc = SolarOpenAttentionProgramConfig(decode_out_cores=(8, 8))
        cc = apc.get_decode_out_compute_config(ttnn.device.Arch.BLACKHOLE)
        assert cc.math_fidelity == ttnn.MathFidelity.HiFi2 and not cc.math_approx_mode
        assert not cc.fp32_dest_acc_en and cc.packer_l1_acc
        cc32 = SolarOpenAttentionProgramConfig(
            decode_out_cores=(8, 8), decode_out_fp32_dest_acc=True
        ).get_decode_out_compute_config(ttnn.device.Arch.BLACKHOLE)
        assert cc32.fp32_dest_acc_en and cc32.math_fidelity == ttnn.MathFidelity.HiFi2
        # the qkv compute config is untouched by the o_proj knobs
        qkv = apc.get_decode_qkv_compute_config(ttnn.device.Arch.BLACKHOLE)
        assert qkv.math_fidelity == ttnn.MathFidelity.HiFi2 and not qkv.fp32_dest_acc_en

    def test_grid_must_divide_n_tiles(self, expect_error):
        with expect_error(ValueError, "must be a tile multiple divisible by"):
            SolarOpenAttentionProgramConfig(decode_out_cores=(7, 7)).get_decode_out_config(32, H, O_K)


class TestFusedQkGrids:
    @pytest.mark.parametrize("device", [BH_11x10, BH_13x10, WH_8x8], ids=["bh11x10", "bh13x10", "wh8x8"])
    @pytest.mark.parametrize("batch", list(range(1, 33)))
    def test_placement_contract(self, device, batch):
        grid = device.compute_with_storage_grid_size()
        heads, q, k = ProgramConfig.get_decode_qk_fused_grids(device, batch)
        heads_l, q_l, k_l = _cores_row_major(heads), _cores_row_major(q), _cores_row_major(k)
        assert len(heads_l) == 2 * batch and len(q_l) == batch and len(k_l) == batch
        assert q_l + k_l == heads_l and not (set(q_l) & set(k_l))
        # RotarySetup(use_qk_fused=True).batch_grid on the doubled batch: 8x8 when 2B % 32 == 0, else the device grid,
        # row-major -> cos/sin row i sits on core i: Q of user b next to row b, K of user b next to row B + b
        base = (8, 8) if (2 * batch) % 32 == 0 else (grid.x, grid.y)
        assert heads_l == grid_cores(*base)[: 2 * batch]
        # nlp_create_qkv_heads_decode's derivation: K starts at the last core of the (B + 1)-core prefix of the grid
        assert k_l[0] == heads_l[batch]
        # the SDPA reducer core of user b is (b % grid.x, b // grid.x) of the SDPA grid and must hold Q of user b
        user_cores, sdpa_grid = ProgramConfig.get_decode_user_grid(device, batch, fused_qk=True)
        assert _cores_row_major(user_cores) == q_l
        assert q_l == [(b % sdpa_grid.x, b // sdpa_grid.x) for b in range(batch)]

    @pytest.mark.parametrize("device", [BH_11x10, BH_13x10], ids=["bh11x10", "bh13x10"])
    def test_production_batches_keep_their_grids(self, device):
        # B = 1 and B = 32 (the sweep's batches): same Q cores and SDPA grid as the legacy chain
        for batch in (1, 32):
            legacy_cores, legacy_grid = ProgramConfig.get_decode_user_grid(device, batch)
            fused_cores, fused_grid = ProgramConfig.get_decode_user_grid(device, batch, fused_qk=True)
            assert _cores_row_major(legacy_cores) == _cores_row_major(fused_cores)
            assert (legacy_grid.x, legacy_grid.y) == (fused_grid.x, fused_grid.y)
        _, _, k32 = ProgramConfig.get_decode_qk_fused_grids(device, 32)
        assert _cores_row_major(k32) == grid_cores(8, 8)[32:]  # K on rows 4-7 of the 8x8 grid
        _, q1, k1 = ProgramConfig.get_decode_qk_fused_grids(device, 1)
        assert _cores_row_major(q1) == [(0, 0)] and _cores_row_major(k1) == [(1, 0)]

    def test_b16_moves_to_8x8(self):
        # the documented exception: 2B = 32 puts RotarySetup on the 8x8 grid, so the SDPA grid follows (device grid
        # -> 8x8: fewer cores per user, a different reduction split beyond one k chunk)
        _, legacy_grid = ProgramConfig.get_decode_user_grid(BH_11x10, 16)
        _, fused_grid = ProgramConfig.get_decode_user_grid(BH_11x10, 16, fused_qk=True)
        assert (legacy_grid.x, legacy_grid.y) == (11, 10) and (fused_grid.x, fused_grid.y) == (8, 8)
        heads, q, k = ProgramConfig.get_decode_qk_fused_grids(BH_11x10, 16)
        assert _cores_row_major(q) == grid_cores(8, 8)[:16] and _cores_row_major(k) == grid_cores(8, 8)[16:32]

    def test_legacy_rule_unchanged(self):
        # the default argument reproduces the phase-1 rule: 8x8 for B % 32 == 0 or B <= 8, the device grid otherwise
        for batch, expect in [(1, (8, 8)), (8, (8, 8)), (16, (11, 10)), (24, (11, 10)), (32, (8, 8))]:
            _, grid = ProgramConfig.get_decode_user_grid(BH_11x10, batch)
            assert (grid.x, grid.y) == expect

    @pytest.mark.parametrize("batch", [0, 33])
    def test_batch_bounds(self, batch, expect_error):
        with expect_error(ValueError, "is not in 1..32"):
            ProgramConfig.get_decode_qk_fused_grids(BH_11x10, batch)

    def test_sdpa_config_follows_the_knob(self):
        cfg_legacy = SolarOpenAttentionProgramConfig(fused_qk=False).get_decode_sdpa_config(BH_11x10, 16)
        cfg_fused = SolarOpenAttentionProgramConfig(fused_qk=True).get_decode_sdpa_config(BH_11x10, 16)
        assert (cfg_legacy.compute_with_storage_grid_size.x, cfg_legacy.compute_with_storage_grid_size.y) == (11, 10)
        assert (cfg_fused.compute_with_storage_grid_size.x, cfg_fused.compute_with_storage_grid_size.y) == (8, 8)
        for batch in (1, 32):
            a = SolarOpenAttentionProgramConfig(fused_qk=False).get_decode_sdpa_config(BH_11x10, batch)
            b = SolarOpenAttentionProgramConfig(fused_qk=True).get_decode_sdpa_config(BH_11x10, batch)
            assert (a.compute_with_storage_grid_size.x, a.compute_with_storage_grid_size.y) == (
                b.compute_with_storage_grid_size.x,
                b.compute_with_storage_grid_size.y,
            )


class TestDecodeStageGuards:
    def test_tp1_refuses_fused_qk(self, expect_error):
        from models.demos.solar_open.tt.attention.decode import fused_qk_enabled

        assert fused_qk_enabled(SolarOpenAttentionProgramConfig(fused_qk=False), SimpleNamespace(tp=1)) is False
        assert fused_qk_enabled(SolarOpenAttentionProgramConfig(fused_qk=True), SimpleNamespace(tp=8)) is True
        with expect_error(ValueError, "needs the width-sharded qkv layout"):
            fused_qk_enabled(SolarOpenAttentionProgramConfig(fused_qk=True), SimpleNamespace(tp=1))
        assert fused_qk_enabled(ProgramConfig(), SimpleNamespace(tp=1)) is False  # base config: no field set
        # the default rule: on for TP > 1, the legacy chain at TP = 1 without raising
        assert fused_qk_enabled(SolarOpenAttentionProgramConfig(tp=8), SimpleNamespace(tp=8)) is True
        assert fused_qk_enabled(SolarOpenAttentionProgramConfig(tp=1), SimpleNamespace(tp=1)) is False
