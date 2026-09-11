# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the phase 3g / D2 "sequential numerics" knob (``SOLAR_OPEN_PACKED_PREFILL_SEQ_NUMERICS``): the level
switch, ttnn's auto ``in0_block_w`` rule against the configs D1 identified on device by bit-identity, the row-piece
arithmetic, the pass marker's per-user row count and the level-2 dense-split rule. No device.

    pytest models/demos/solar_open/tests/unit/test_packed_numerics.py
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.solar_open.tt import packed_numerics as pn
from models.demos.solar_open.tt import packed_prefill as pp
from models.demos.solar_open.tt.experts import prefill as experts_prefill

H = 4096


# D1 r7 / r8 (README "Phase 3g rows"): the auto configs identified by bit-identity against explicit candidates.
@pytest.mark.parametrize(
    "m, n, k, expected",
    [
        (128, 1280, H, 2),  # qkv M=128: 1D k=2
        (256, 1280, H, 1),  # qkv M=256: 2D k=1
        (1024, 1280, H, 1),
        (4096, 1280, H, 1),
        (32, H, 1024, 2),  # o_proj M<=256: 1D k=2
        (128, H, 1024, 2),
        (256, H, 1024, 2),
        (1024, H, 1024, 1),  # o_proj M>=1024: 2D k=1
        (4096, H, 1024, 1),
        (32, 32768, H, 2),  # lm_head M=32..1024: 1D k=2
        (128, 32768, H, 2),
        (1024, 32768, H, 2),
        (4096, 32768, H, 1),  # lm_head M=4096: 2D k=1
        (256, 160, H, 1),  # shared gate M=256: 2D k=1
        (32, 160, H, 2),  # (auto at 32 rows would be the narrow form; production runs the explicit config there)
    ],
)
def test_auto_in0_block_w_matches_d1(m, n, k, expected):
    assert pp.auto_in0_block_w(m, n, k) == expected


def test_seq_numerics_pins_only_where_the_auto_choice_moves():
    assert pp.seq_numerics_in0_block_w(256, 128, 1280, H) == 2  # qkv: pin the 128-row k=2 at T=256
    assert pp.seq_numerics_in0_block_w(4096, 128, 1280, H) == 2
    assert pp.seq_numerics_in0_block_w(4096, 1024, 1280, H) is None  # S=1024 users: T and S both 2D k=1
    assert pp.seq_numerics_in0_block_w(256, 128, H, 1024) is None  # o_proj at 256: 1D k=2 like 128
    assert pp.seq_numerics_in0_block_w(512, 128, H, 1024) == 2
    assert pp.seq_numerics_in0_block_w(4096, 128, H, 1024) == 2
    assert pp.seq_numerics_in0_block_w(1024, 32, 32768, H) is None  # lm_head pieces of 1024 rows = the 32-row config
    assert pp.seq_numerics_in0_block_w(4096, 32, 32768, H) == 2


def test_lm_head_pieces_and_row_pieces(expect_error):
    assert pp.lm_head_piece_rows(64, 32768, H) == 64
    assert pp.lm_head_piece_rows(256, 32768, H) == 256
    assert pp.lm_head_piece_rows(1024, 32768, H) == 1024
    assert pp.lm_head_piece_rows(4096, 32768, H) == 1024  # capped at LM_HEAD_SEQ_ROWS
    assert (
        pp.lm_head_piece_rows(4096, 32768, H, max_rows=4096) == 2048
    )  # halves until the auto k matches the 32-row one
    assert pp.row_pieces(4096, 1024) == [(0, 1024), (1024, 1024), (2048, 1024), (3072, 1024)]
    assert pp.row_pieces(160, 64) == [(0, 64), (64, 64), (128, 32)]
    with expect_error(ValueError, "positive sizes"):
        pp.row_pieces(0, 32)


def test_level_parsing_and_context(monkeypatch, expect_error):
    assert pp.read_packed_seq_numerics("") == pp.PACKED_SEQ_NUMERICS_DEFAULT == 2  # D2: exact packed passes by default
    assert pp.read_packed_seq_numerics("1") == 1 and pp.read_packed_seq_numerics("2") == 2
    assert pp.read_packed_seq_numerics("on") == 1 and pp.read_packed_seq_numerics("off") == 0
    with expect_error(ValueError, "is not one of"):
        pp.read_packed_seq_numerics("3")
    monkeypatch.setenv(pp.PACKED_SEQ_NUMERICS_ENV, "2")
    assert pp.read_packed_seq_numerics() == 2
    before = pp.packed_seq_numerics_level()
    with pp.packed_seq_numerics(1):
        assert pp.packed_seq_numerics_level() == 1
        with pp.packed_seq_numerics(2):
            assert pp.packed_seq_numerics_level() == 2
        assert pp.packed_seq_numerics_level() == 1
    assert pp.packed_seq_numerics_level() == before
    with expect_error(ValueError, "is not one of"):
        with pp.packed_seq_numerics(7):
            pass


def test_marker_carries_the_per_user_rows_and_gates_the_knob():
    # outside a packed pass the knob is never in effect, whatever its level: the single-user path is untouched
    with pp.packed_seq_numerics(2):
        assert experts_prefill.packed_seq_numerics_level() == 0
        assert pn.seq_numerics_level() == 0 and not pn.seq_numerics_active() and pn.seq_numerics_rows() is None
        with experts_prefill.packed_prefill_pass(False, seq_len=128):
            assert experts_prefill.packed_prefill_seq_len() is None and pn.seq_numerics_level() == 0
        with experts_prefill.packed_prefill_pass(True, seq_len=128):
            assert experts_prefill.packed_prefill_pass_active() and experts_prefill.packed_prefill_seq_len() == 128
            assert pn.seq_numerics_level() == 2 and pn.seq_numerics_rows() == 128
            assert pn.shared_expert_piece_rows(4096) == 128 and pn.shared_expert_piece_rows(128) is None
            assert pn.shared_expert_piece_rows(192) is None  # S does not divide the rows: run as is
            # level 2: dense-bmm splits of dense_bmm_max_tokens rows when S runs the dense bmm itself
            assert experts_prefill.seq_numerics_dense_split_size(1024, 256) == 256
            assert experts_prefill.seq_numerics_dense_split_size(128, 256) == 128
        with experts_prefill.packed_prefill_pass(True, seq_len=1024):
            assert experts_prefill.seq_numerics_dense_split_size(1024, 256) == 1024  # S=1024 is sorted itself
            assert pn.shared_expert_piece_rows(4096) == 1024
    with pp.packed_seq_numerics(1), experts_prefill.packed_prefill_pass(True, seq_len=128):
        assert experts_prefill.seq_numerics_dense_split_size(1024, 256) == 1024  # level 1 leaves the MoE alone
    assert not experts_prefill.packed_prefill_pass_active() and experts_prefill.packed_prefill_seq_len() is None


def test_matmul_config_2d_uses_the_auto_blocking_with_the_pinned_k():
    grid = SimpleNamespace(x=11, y=10)
    cfg = pn.matmul_config_2d(4096, 1280, H, 2, grid)  # qkv at T=4096
    assert (cfg.in0_block_w, cfg.per_core_M, cfg.per_core_N, cfg.out_subblock_h, cfg.out_subblock_w) == (2, 13, 4, 1, 4)
    assert (cfg.out_block_h, cfg.out_block_w) == (13, 4)
    cfg = pn.matmul_config_2d(256, 1280, H, 2, grid)  # the r7-verified candidate: 2D k=2 pcM=1 pcN=4
    assert (cfg.in0_block_w, cfg.per_core_M, cfg.per_core_N) == (2, 1, 4)
    cfg = pn.matmul_config_2d(4096, H, 1024, 2, grid)  # o_proj at T=4096: the r7-verified 2D k=2 pcM=13 pcN=12
    assert (cfg.in0_block_w, cfg.per_core_M, cfg.per_core_N, cfg.out_subblock_w) == (2, 13, 12, 4)
    cfg = pn.matmul_config_2d(512, H, 1024, 2, grid)
    assert (cfg.per_core_M, cfg.per_core_N) == (2, 12)
    assert pn.matmul_config_2d(256, 160, H, 33, grid).in0_block_w == 32  # snaps to a divisor of Kt = 128


def test_attention_configs_only_inside_a_packed_pass():
    grid = SimpleNamespace(x=11, y=10)
    mesh_device = SimpleNamespace(compute_with_storage_grid_size=lambda: grid, arch=lambda: ttnn.device.Arch.BLACKHOLE)
    weights = SimpleNamespace(
        wqkv=SimpleNamespace(shape=(1, 1, H, 1280)), o_proj=SimpleNamespace(shape=(1, 1, 1024, H))
    )
    none = (None, None)
    with pp.packed_seq_numerics(1):
        # single-user call (batch 1) and no pass marker: auto everywhere
        assert pn.attention_seq_numerics_configs(4096, 4096, 1, weights, mesh_device, None, False) == (none, none)
        with experts_prefill.packed_prefill_pass(True, seq_len=128):
            assert pn.attention_seq_numerics_configs(4096, 4096, 1, weights, mesh_device, None, False) == (none, none)
            (qkv_pc, qkv_ckc), (out_pc, out_ckc) = pn.attention_seq_numerics_configs(
                256, 128, 2, weights, mesh_device, None, False
            )
            assert qkv_pc.in0_block_w == 2 and qkv_ckc is not None  # T=256: qkv pinned
            assert out_pc is None and out_ckc is None  # o_proj auto at 256 is the 128-row config already
            (qkv_pc, _), (out_pc, out_ckc) = pn.attention_seq_numerics_configs(
                4096, 128, 32, weights, mesh_device, None, False
            )
            assert qkv_pc.in0_block_w == 2 and out_pc.in0_block_w == 2 and out_ckc is not None
    with pp.packed_seq_numerics(0), experts_prefill.packed_prefill_pass(True, seq_len=128):
        assert pn.attention_seq_numerics_configs(4096, 128, 32, weights, mesh_device, None, False) == (none, none)
