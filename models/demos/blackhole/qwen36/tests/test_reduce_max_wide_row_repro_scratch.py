# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproducer for upstream: ttnn.max(x, dim=-1) over a WIDE bf16 TILE row returns wrong maxima (P150x4 mesh).

Observed 2026-09-24 in the Qwen3.8-27B speculative-verify step (models/demos/blackhole/qwen36, tests/VERIFY_W32_AUDIT.md):
on a [1, 1, 128, 62080] bf16 TILE tensor (the vocab-sharded LM-head logits at R = 128 rows, V/TP = 62080 = 1940 tiles),
`ttnn.max(logits, dim=-1)` returned a wrong value for every one of the 128 rows on every device of the (1,4) mesh
(logs/verify_324_head.log: 512/512 (device, row) maxima != torch max of the same logits read back to host), while
`ttnn.argmax(ttnn.to_layout(logits, ROW_MAJOR), dim=-1)` on the same tensor was right on every row, and the two-stage
form `ttnn.max(reshape(pad(x), [1, R, 1940->1952, 32]), -1)` then `ttnn.max(..., -1)` (text_demo `_maxval_dev_b`) was right.
The same single reduce returned correct maxima for the same shape in a smaller process (tests/test_verify_probe_scratch.py:
'max_tile ... match=256/256' at R=256) and in tests/test_verify_step1_scratch.py; the failing process had ~64 layers of a
27B model resident (~13 GB DRAM per chip), several traces captured (prefill chunk + 5 masked buckets + decode + verify),
and the reduce ran EAGERLY right after a trace replay. What differs is therefore the allocator / program-cache state of
the process, not the input values -- the reduce_w program for a 1940-tile-wide row (the multi-core W reduction over
many tile columns) is the suspect; a smaller per-core width (32 tiles -> the two-stage form) is exact in every state.

This standalone test exercises the op at that shape in three states: fresh process, after allocating a large DRAM
footprint, and after capturing + replaying an unrelated trace; it PASSES when the bug does not reproduce (the bug may
need the exact allocator state of the model process -- see the run recipe below), and prints per-state mismatch counts.

  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 pytest .../test_reduce_max_wide_row_repro_scratch.py -s
Model-process repro (deterministic, ~4 min): scripts/verify_step_chain.sh
  "repro|VERIFY_CONFIGS=32,4 VERIFY_EXACT=1 VERIFY_TIMING=0 VERIFY_DECODE_WIDTHS=32 VERIFY_MIN_TOKENS=2 VERIFY_POLICIES=random
   VERIFY_DEBUG_EAGER=1 VERIFY_DEBUG_ROWDIAG=1 QWEN36_VERIFY_MAX=plain"
  -> logs/repro.log "[verify-head] ... device max != host max at ... (512 total)"; with QWEN36_VERIFY_MAX=tile: 0 total.
"""
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp

R, V = 128, 62080


def _rep(mesh, t, dtype, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, memory_config=mem, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _check(mesh, tag, x_host):
    xt = _rep(mesh, x_host, ttnn.bfloat16)
    ref = x_host.float().reshape(R, V).max(-1).values
    plain = ttnn.to_torch(ttnn.get_device_tensors(ttnn.max(xt, dim=-1))[0]).float().reshape(-1)[:R]
    # two-stage form
    C = 32
    n_rows = -(-V // C)
    n_rows_t = -(-n_rows // 32) * 32
    padded = ttnn.pad(xt, [(0, 0), (0, 0), (0, 0), (0, n_rows_t * C - V)], value=-1e30)
    grid = ttnn.reshape(padded, (1, R, n_rows_t, C))
    part = ttnn.max(grid, dim=-1)
    two = ttnn.to_torch(ttnn.get_device_tensors(ttnn.max(ttnn.reshape(part, (1, 1, R, n_rows_t)), dim=-1))[0])
    two = two.float().reshape(-1)[:R]
    rm = ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT)
    am = (
        ttnn.to_torch(ttnn.get_device_tensors(ttnn.argmax(rm, dim=-1, keepdim=False))[0])
        .reshape(-1)[:R]
        .to(torch.int64)
    )
    am_ref = x_host.float().reshape(R, V).argmax(-1)
    res = (int((plain != ref).sum()), int((two != ref).sum()), int((am != am_ref).sum()))
    logger.info(
        f"REPRO {tag}: plain-max wrong rows {res[0]}/{R}, two-stage-max wrong rows {res[1]}/{R}, argmax wrong rows {res[2]}/{R}"
    )
    for t in (xt, padded, grid, part, rm):
        ttnn.deallocate(t)
    return res


@parametrize_mesh_tp()
def test_reduce_max_wide_row(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(1, 1, R, V, generator=g) * 4).to(torch.bfloat16)
    results = {"fresh": _check(mesh, "fresh process", x)}
    # state 2: a large resident DRAM footprint (~12 GB per chip) + many small allocations
    hold = [_rep(mesh, torch.randn(1, 1, 4096, 8192).to(torch.bfloat16), ttnn.bfloat16) for _ in range(180)]
    small = [_rep(mesh, torch.randn(1, 1, 32, 1024).to(torch.bfloat16), ttnn.bfloat16) for _ in range(200)]
    results["big_footprint"] = _check(mesh, "after ~12 GB resident DRAM", x)
    # state 3: an unrelated trace captured and replayed just before the eager reduce
    a = _rep(mesh, torch.randn(1, 1, 256, 5120).to(torch.bfloat16), ttnn.bfloat16)
    b = _rep(mesh, torch.randn(1, 1, 5120, 4352).to(torch.bfloat16), ttnn.bfloat8_b)
    y = ttnn.matmul(a, b)
    ttnn.deallocate(y)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    y = ttnn.matmul(a, b)
    z = ttnn.max(y, dim=-1)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    for _ in range(5):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    results["after_trace_replays"] = _check(mesh, "after trace replays", x)
    ttnn.release_trace(mesh, tid)
    for t in hold + small:
        ttnn.deallocate(t)
    print("REPRO_RESULT " + " | ".join(f"{k}: plain={v[0]} two_stage={v[1]} argmax={v[2]}" for k, v in results.items()))
    # the two-stage form and argmax must be exact in every state; the plain max is reported (the bug is state dependent)
    assert all(v[1] == 0 and v[2] == 0 for v in results.values())
