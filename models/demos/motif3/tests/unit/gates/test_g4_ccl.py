# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G4 — generic CCLs with the exact Motif payloads on the logical (4, 8) mesh.

Design §1.3, §2.3.7 (decode dataflow steps 2/9/10/11), §3.2, §3.3, §4.2 G4, §7.2 Q4. Mesh axes: rows =
cluster_axis 0 (4 chips, DP groups), cols = cluster_axis 1 (8 chips, TP8). Call patterns follow
``models/demos/gpt_oss/tt/experts_throughput/gather_decode.py`` (``ttnn.all_gather(x, dim=2,
cluster_axis=...)``, ``ttnn.reduce_scatter(x, dim=2, cluster_axis=...)``, ``ttnn.all_reduce(x,
cluster_axis=...)``; default topology / links).

Payloads per chip:
  decode  AR(cols) [1,1,8,4096] bf16             (attention wo output, MoE output)        x2 per MoE layer
          AG(rows) [1,1,8,4096] -> [1,1,32,4096]  (MoE token gather; TILE and ROW_MAJOR; dim-1 variant)
          AR(rows) [1,1,32,4096] bf16 / fp32      (MoE combine)
          AR(cols) [1,1,8,3] fp32                 (shared-expert PolyNorm TP statistics)
  prefill RS(rows) [1,1,S,4096] -> [1,1,S/4,4096], AG(rows) [1,1,S/4,4096] -> [1,1,S,4096], AR(cols) [1,1,S,4096]

Each chip holds distinct data; the golden is the torch fp32 sum / concat per reduce group. Checks:
correctness (PCC / max-abs), *bitwise identical* results across every chip of a reduce group (the
routing-consistency invariant of §2.3.7), eager latency and traced latency (two-length slope).
Run on FABRIC_2D_TORUS_XY (draft-1 fabric) and FABRIC_1D_RING (named fallback).
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu

ROWS, COLS = gu.MESH_SHAPE
REC = gu.Recorder("G4")

FABRICS = [
    pytest.param(*gu.mesh_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY), id="torus2d"),
    pytest.param(*gu.mesh_params(ttnn.FabricConfig.FABRIC_1D_RING), id="ring1d"),
]

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG

# name, op, per-chip shape, dtype, layout, cluster_axis, dim, memory config
DECODE_CASES = [
    ("AR_cols_8x4096_bf16_dram", "all_reduce", [1, 1, 8, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, 1, None, DRAM),
    ("AR_cols_8x4096_bf16_l1", "all_reduce", [1, 1, 8, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, 1, None, L1),
    ("AG_rows_8x4096_bf16_tile_dram", "all_gather", [1, 1, 8, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, 0, 2, DRAM),
    ("AG_rows_8x4096_bf16_tile_l1", "all_gather", [1, 1, 8, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, 0, 2, L1),
    ("AG_rows_8x4096_bf16_rm_dram", "all_gather", [1, 1, 8, 4096], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, 0, 2, DRAM),
    ("AG_rows_8x4096_bf16_tile_dim1", "all_gather", [1, 1, 8, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, 0, 1, DRAM),
    ("AR_rows_32x4096_bf16_dram", "all_reduce", [1, 1, 32, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, 0, None, DRAM),
    ("AR_rows_32x4096_fp32_dram", "all_reduce", [1, 1, 32, 4096], ttnn.float32, ttnn.TILE_LAYOUT, 0, None, DRAM),
    ("AR_cols_stats_8x3_fp32_dram", "all_reduce", [1, 1, 8, 3], ttnn.float32, ttnn.TILE_LAYOUT, 1, None, DRAM),
    ("AR_cols_stats_8x3_fp32_l1", "all_reduce", [1, 1, 8, 3], ttnn.float32, ttnn.TILE_LAYOUT, 1, None, L1),
]

PREFILL_CASES = [
    ("RS_rows_128x4096_bf16", "reduce_scatter", [1, 1, 128, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, 0, 2, DRAM),
    ("RS_rows_1024x4096_bf16", "reduce_scatter", [1, 1, 1024, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, 0, 2, DRAM),
    ("AG_rows_32x4096_bf16_prefill128", "all_gather", [1, 1, 32, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, 0, 2, DRAM),
    ("AG_rows_256x4096_bf16_prefill1024", "all_gather", [1, 1, 256, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, 0, 2, DRAM),
    ("AR_cols_128x4096_bf16_prefill", "all_reduce", [1, 1, 128, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, 1, None, DRAM),
    (
        "AR_cols_1024x4096_bf16_prefill",
        "all_reduce",
        [1, 1, 1024, 4096],
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        1,
        None,
        DRAM,
    ),
]


def chip_index(r: int, c: int) -> int:
    return r * COLS + c


def make_input(shape, seed: int, dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(ROWS, COLS, *shape[2:], generator=g)
    if dtype == ttnn.bfloat16:
        x = x.bfloat16().float()
    return x


def golden(op: str, x: torch.Tensor, axis: int, dim):
    """Per-chip expected output, indexed [r][c] -> tensor of the per-chip output shape (fp64)."""
    xd = x.double()
    out = [[None] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            if op == "all_reduce":
                s = xd[r].sum(dim=0) if axis == 1 else xd[:, c].sum(dim=0)
                out[r][c] = s[None, None]
            elif op == "all_gather":
                parts = (
                    [xd[rr, c][None, None] for rr in range(ROWS)]
                    if axis == 0
                    else [xd[r, cc][None, None] for cc in range(COLS)]
                )
                out[r][c] = torch.cat(parts, dim=dim)
            elif op == "reduce_scatter":
                s = xd[:, c].sum(dim=0) if axis == 0 else xd[r].sum(dim=0)
                n = ROWS if axis == 0 else COLS
                k = r if axis == 0 else c
                h = s.shape[0] // n
                out[r][c] = s[k * h : (k + 1) * h][None, None]
    return out


def groups(axis: int):
    """Lists of chip indices that must hold bitwise-identical results after an all-reduce / all-gather."""
    if axis == 1:
        return [[chip_index(r, c) for c in range(COLS)] for r in range(ROWS)]
    return [[chip_index(r, c) for r in range(ROWS)] for c in range(COLS)]


def call(op: str, t, axis: int, dim, mem):
    if op == "all_reduce":
        return ttnn.all_reduce(t, cluster_axis=axis, memory_config=mem)
    if op == "all_gather":
        return ttnn.all_gather(t, dim=dim, cluster_axis=axis, memory_config=mem)
    if op == "reduce_scatter":
        return ttnn.reduce_scatter(t, dim=dim, cluster_axis=axis, memory_config=mem)
    raise ValueError(op)


def run_case(mesh_device, fabric_id: str, case, seed: int):
    name, op, shape, dtype, layout, axis, dim, mem = case
    tag = f"{fabric_id}/{name}"
    x = make_input(shape, seed, dtype)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=gu.MESH_SHAPE)
    t = ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh_device, memory_config=mem, mesh_mapper=mapper)

    def fn():
        return call(op, t, axis, dim, mem)

    try:
        out = fn()
        ttnn.synchronize_device(mesh_device)
        got = gu.read_all(out)
    except Exception as e:
        REC.add(tag, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
        return False, f"{tag}: {type(e).__name__}: {str(e)[:200]}"

    want = golden(op, x, axis, dim)
    worst_pcc, worst_abs, shape_ok = 1.0, 0.0, True
    for r in range(ROWS):
        for c in range(COLS):
            g = got[chip_index(r, c)]
            w = want[r][c]
            if list(g.shape) != list(w.shape):
                shape_ok = False
                g = g.reshape(-1)[: w.numel()].reshape(w.shape) if g.numel() >= w.numel() else g
            if g.shape == w.shape:
                s = gu.compare(w, g)
                worst_pcc = min(worst_pcc, s["pcc"])
                worst_abs = max(worst_abs, s["max_abs"])
    replica_ok = True
    if op in ("all_reduce", "all_gather"):
        for grp in groups(axis):
            ref = got[grp[0]]
            for i in grp[1:]:
                if not torch.equal(gu._bits(got[i].contiguous()), gu._bits(ref.contiguous())):
                    replica_ok = False
    out_shape = list(got[0].shape)
    del out

    try:
        eager = gu.time_eager(mesh_device, fn, iters=20)
    except Exception as e:
        eager = f"error {type(e).__name__}"
    try:
        traced, traced_raw = gu.time_traced(mesh_device, fn, ops_per_trace=64, reps=9)
    except Exception as e:
        traced, traced_raw = f"error {type(e).__name__}: {str(e)[:200]}", None

    pcc_min = 0.9999 if dtype == ttnn.float32 else 0.999
    ok = shape_ok and worst_pcc >= pcc_min and replica_ok
    bytes_per_chip = 1
    for d in shape:
        bytes_per_chip *= d
    bytes_per_chip *= 4 if dtype == ttnn.float32 else 2
    REC.add(
        tag,
        status="pass" if ok else "fail",
        op=op,
        cluster_axis=axis,
        dim=dim,
        in_shape=shape,
        out_shape=out_shape,
        dtype=str(dtype),
        layout=str(layout),
        mem="L1" if mem == L1 else "DRAM",
        payload_bytes_per_chip=bytes_per_chip,
        worst_pcc=worst_pcc,
        worst_max_abs=worst_abs,
        replicas_bitwise_identical=replica_ok,
        eager_us=eager,
        traced_us=traced,
        traced_raw_us=traced_raw,
    )
    if not ok:
        return False, f"{tag}: shape_ok={shape_ok} pcc={worst_pcc:.6f} replicas={replica_ok} out_shape={out_shape}"
    return True, ""


@pytest.mark.parametrize("mesh_device, device_params", FABRICS, indirect=True)
def test_g4_ccl_decode(mesh_device, device_params):
    fabric_id = gu.fabric_name(device_params["fabric_config"])
    floor = gu.time_trace_replay_floor(mesh_device)
    REC.add(f"{fabric_id}/trace_replay_floor", status="info", us=floor)
    failures = []
    for i, case in enumerate(DECODE_CASES):
        ok, msg = run_case(mesh_device, fabric_id, case, seed=100 + i)
        if not ok:
            failures.append(msg)
    # the composite AR(rows)+slice alternative to an 8-row reduce-scatter (design §2.3.7 step 9) is the
    # AR_rows_32x4096 case above; a direct 8-row-shard RS is recorded for completeness
    ok, msg = run_case(
        mesh_device,
        fabric_id,
        (
            "RS_rows_32x4096_bf16_8row_shards",
            "reduce_scatter",
            [1, 1, 32, 4096],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            0,
            2,
            DRAM,
        ),
        seed=99,
    )
    if not ok:
        REC.add(f"{fabric_id}/RS_rows_32x4096_bf16_8row_shards/note", status="info", note="informational only: " + msg)
    assert not failures, "G4 decode CCL failures:\n" + "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", FABRICS, indirect=True)
def test_g4_ccl_prefill(mesh_device, device_params):
    fabric_id = gu.fabric_name(device_params["fabric_config"])
    failures = []
    for i, case in enumerate(PREFILL_CASES):
        ok, msg = run_case(mesh_device, fabric_id, case, seed=200 + i)
        if not ok:
            failures.append(msg)
    assert not failures, "G4 prefill CCL failures:\n" + "\n".join(failures)
