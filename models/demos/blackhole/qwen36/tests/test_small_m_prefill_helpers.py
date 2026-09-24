# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the small-M prefill matmul helpers (tp_common "Small-M prefill matmuls", item J): the env
gate, the per-role row thresholds and the 1D progcfg shapes for the real Qwen3.8-27B TP=4 per-device matmuls.
No device: runs in the plain python_env.

  python_env/bin/python -m pytest models/demos/blackhole/qwen36/tests/test_small_m_prefill_helpers.py -q
"""

import math

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

# (m, k, n): the per-device shapes the path serves at TP=4 (bucket 128 / 256 rows).
SHAPES = {
    "gdn_qkvzab": (5120, 4128),
    "attn_qkv": (5120, 3584),
    "mlp_w1_w3": (5120, 4352),
    "mlp_w2": (4352, 5120),
    "attn_wo": (1536, 5120),
}


def test_small_m_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("QWEN36_PREFILL_SMALLM_MAX", raising=False)
    assert tpc.small_m_prefill_max() == 0
    for s in (64, 128, 256):
        assert not tpc.small_m_rows(s)
        assert not tpc.small_m_in_proj(s)
    monkeypatch.setenv("QWEN36_PREFILL_SMALLM_MAX", "junk")
    assert tpc.small_m_prefill_max() == 0


def test_small_m_thresholds(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_SMALLM_MAX", "256")
    # decode rows (<= one tile) never take the path; in-projections stop at 128 rows, the rest at the env max
    assert not tpc.small_m_rows(32) and not tpc.small_m_in_proj(32)
    assert tpc.small_m_rows(64) and tpc.small_m_in_proj(64)
    assert tpc.small_m_rows(128) and tpc.small_m_in_proj(128)
    assert tpc.small_m_rows(256) and not tpc.small_m_in_proj(256)
    assert not tpc.small_m_rows(512) and not tpc.small_m_in_proj(512)
    monkeypatch.setenv("QWEN36_PREFILL_SMALLM_MAX", "128")
    assert tpc.small_m_rows(128) and not tpc.small_m_rows(256)


@pytest.mark.parametrize("m", [128, 256])
@pytest.mark.parametrize("name", sorted(SHAPES))
@pytest.mark.parametrize("grid_w", [11, 8])
def test_small_m_progcfg_shapes(name, m, grid_w):
    k, n = SHAPES[name]
    act = ttnn.UnaryOpType.SILU if name == "mlp_w1_w3" else None
    pc = tpc.small_m_progcfg(m, k, n, fused_activation=act, grid_w=grid_w)
    assert isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig)
    assert pc.mcast_in0 and pc.fuse_batch
    m_tiles, k_tiles, n_tiles = m // 32, k // 32, math.ceil(n / 32)
    # mcast_in0: every core streams the full K -> the block must divide it; one M block per core
    assert pc.per_core_M == m_tiles
    assert k_tiles % pc.in0_block_w == 0
    cols, rows = pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y
    assert cols <= grid_w and rows <= 10, (cols, rows)  # fits the 11x10 BH P150 / 8x8 WH worker grid
    assert pc.per_core_N * cols * rows >= n_tiles
    if grid_w == 11:  # the measured BH P150 grid: <= 2 N tiles per core (weight-bandwidth-bound regime)
        assert pc.per_core_N <= 2
    # fp32 dest accumulation caps the subblock at 4 tiles
    assert pc.out_subblock_h * pc.out_subblock_w <= 4
    assert m_tiles % pc.out_subblock_h == 0 and pc.per_core_N % pc.out_subblock_w == 0
    assert (pc.fused_activation is not None) == (act is not None)
    # cached: same object for the same key
    assert tpc.small_m_progcfg(m, k, n, fused_activation=act, grid_w=grid_w) is pc
