# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase F F5 (``MOTIF3_CCL_FUSE=ar``, ``tt/kernels/ccl_ar.py``): the fused all-reduce program equals the Phase E
reduce-scatter (direct) + all-gather (safe: all_broadcast + concat) chain bit for bit.

Device (lock wrapper)::

    S=/home/ttuser/hchang/experiments/motif-3/scripts
    $S/devrun.sh -t 900 -n ccl_fuse -- python -m pytest models/demos/motif3/tests/unit/test_ccl_fuse.py -s \
        -p no:cacheprovider

``F5_CASES`` (comma list of case names, default all) runs a subset; ``F5_REPEAT`` the repeated-call count.

* ``test_ccl_fuse_ar_bitwise``: every decode all-reduce payload (AR(tp) of [1, 1, L, 4096] at L = 8 / 16 / 32 into DRAM
  and L1, AR(dp) of [1, 1, 32 / 64, 4096]) with distinct random data per chip: fused == ops on all 32 chips (padding
  tiles included: the full padded tiles are compared through a TILE->host copy of the padded shape), the fused path
  really ran, replicas identical, F5_REPEAT repeated fused calls identical; a traced fused call replayed with new inputs
  equals the eager ops path.
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, log_fabric, replicas_identical
from models.demos.motif3.tt.model_config import MotifTTConfig, device_params

TRACE = 32 * 1024 * 1024
MESH = [pytest.param((4, 8), device_params("FABRIC_2D_TORUS_XY", TRACE), id="4x8-torus2d")]

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG
# name -> (axis, local shape, input mc, output mc)
CASES = {
    "tp32_dram": ("tp", (1, 1, 32, 4096), DRAM, DRAM),
    "tp8_dram": ("tp", (1, 1, 8, 4096), DRAM, DRAM),
    "tp8_l1": ("tp", (1, 1, 8, 4096), L1, L1),
    "tp16_dram": ("tp", (1, 1, 16, 4096), DRAM, DRAM),
    "dp32_dram": ("dp", (1, 1, 32, 4096), DRAM, DRAM),
    "dp64_dram": ("dp", (1, 1, 64, 4096), DRAM, DRAM),
    "tp32_l1in_dram": ("tp", (1, 1, 32, 4096), L1, DRAM),
}


def _words(t, mesh_device):
    """Every chip's tensor as int16 words of the PADDED tile data ([R, C, ...])."""
    return device_tensors_to_torch(t, mesh_device).contiguous().view(torch.int16)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_ccl_fuse_ar_bitwise(mesh_device, device_params):
    rep = log_fabric(mesh_device, "ccl_fuse_ar")
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)
    ops = MotifCCL(mesh_device, cfg, fuse="off")
    fus = MotifCCL(mesh_device, cfg, fuse="ar")
    R, C = (int(s) for s in tuple(mesh_device.shape))
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C))
    names = [c for c in (os.environ.get("F5_CASES") or ",".join(CASES)).split(",") if c]
    reps = int(os.environ.get("F5_REPEAT", "20"))
    g = torch.Generator().manual_seed(55)
    failures = []
    for name in names:
        axis, shape, imc, omc = CASES[name]
        xh = (torch.randn(R, C, *shape[2:], generator=g) * 4.0).bfloat16().float()
        x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh_device, mesh_mapper=mapper, memory_config=imc)
        t0 = time.time()
        b = ops.all_reduce(x, axis, memory_config=omc)
        ttnn.synchronize_device(mesh_device)
        t1 = time.time()
        c0 = fus._fused_ar.calls if fus._fused_ar is not None else 0
        a = fus.all_reduce(x, axis, memory_config=omc)
        ttnn.synchronize_device(mesh_device)
        t2 = time.time()
        used = (fus._fused_ar.calls if fus._fused_ar is not None else 0) - c0
        wa, wb = device_tensors_to_torch(a, mesh_device), device_tensors_to_torch(b, mesh_device)
        same = bool(torch.equal(wa.view(torch.int16), wb.view(torch.int16)))
        ndiff = int((wa.view(torch.int16) != wb.view(torch.int16)).sum())
        repl = replicas_identical(a, mesh_device, axis)
        ref = xh.double().sum(dim=(0 if axis == "dp" else 1), keepdim=True)
        err = float((wa.double().reshape(R, C, *shape[2:]) - ref).abs().max())
        print(f"[f5] {name}: fused used {used}, bitwise {same} (ndiff {ndiff}), replicas {repl}, max|fused - fp64| "
              f"{err:.3e}, ops {1e3 * (t1 - t0):.1f} ms (eager, first), fused {1e3 * (t2 - t1):.1f} ms (eager, first), "
              f"out {a.memory_config().buffer_type} shape {list(a.shape)} padded {list(a.padded_shape)}", flush=True)
        if not (same and used == 1 and repl):
            failures.append(f"{name}: used {used} same {same} ndiff {ndiff} repl {repl}")
        rep_ok = True
        for i in range(reps):
            a2 = fus.all_reduce(x, axis, memory_config=omc)
            w2 = device_tensors_to_torch(a2, mesh_device)
            ttnn.deallocate(a2)
            if not torch.equal(w2.view(torch.int16), wa.view(torch.int16)):
                rep_ok = False
                failures.append(f"{name}: repeat {i} differs")
                break
        print(f"[f5] {name}: {reps} repeated fused calls identical: {rep_ok}", flush=True)
        for t in (a, b, x):
            ttnn.deallocate(t)
    if os.environ.get("F5_TRACE", "1") == "1" and "tp8_l1" in names:
        axis, shape, imc, omc = CASES["tp8_l1"]
        x = ttnn.from_torch(torch.randn(R, C, *shape[2:], generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh_device, mesh_mapper=mapper, memory_config=imc)
        ttnn.deallocate(fus.all_reduce(x, axis, memory_config=omc))
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = fus.all_reduce(x, axis, memory_config=omc)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        try:
            for it in range(3):
                new = ttnn.from_torch(torch.randn(R, C, *shape[2:], generator=g), dtype=ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
                ttnn.copy_host_to_device_tensor(new, x)
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                ref = ops.all_reduce(x, axis, memory_config=omc)
                ok = bool(torch.equal(_words(out, mesh_device), _words(ref, mesh_device)))
                ttnn.deallocate(ref)
                print(f"[f5] traced fused replay {it} == eager ops bitwise: {ok}", flush=True)
                if not ok:
                    failures.append(f"trace replay {it}")
        finally:
            ttnn.release_trace(mesh_device, tid)
            ttnn.deallocate(out)
            ttnn.deallocate(x)
    print(f"[f5] fabric committed {rep.get('committed')}; failures {failures}", flush=True)
    assert not failures, failures


def _traced_us(mesh_device, fn, n: int = 32, reps: int = 10) -> float:
    """Median replay time of a trace of ``n`` back-to-back calls of ``fn`` (each output freed), per call, in us."""
    import statistics

    outs = [fn()]  # compile
    for o in outs:
        ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(n):
        ttnn.deallocate(fn())
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ts = []
    try:
        for _ in range(reps):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            ts.append((time.perf_counter() - t0) * 1e6 / n)
    finally:
        ttnn.release_trace(mesh_device, tid)
    return statistics.median(ts)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_ccl_fuse_ar_timing(mesh_device, device_params):
    """Traced cost per all-reduce, ops chain vs fused, for every case (``F5_CASES``)."""
    log_fabric(mesh_device, "ccl_fuse_ar_timing")
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)
    ccls = {"off": MotifCCL(mesh_device, cfg, fuse="off"), "ar": MotifCCL(mesh_device, cfg, fuse="ar")}
    R, C = (int(s) for s in tuple(mesh_device.shape))
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C))
    names = [c for c in (os.environ.get("F5_CASES") or ",".join(CASES)).split(",") if c]
    g = torch.Generator().manual_seed(56)
    for name in names:
        axis, shape, imc, omc = CASES[name]
        x = ttnn.from_torch(torch.randn(R, C, *shape[2:], generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=mesh_device, mesh_mapper=mapper, memory_config=imc)
        res = {}
        for mode in ("off", "ar", "off", "ar"):
            us = _traced_us(mesh_device, lambda: ccls[mode].all_reduce(x, axis, memory_config=omc))
            res.setdefault(mode, []).append(us)
        print(f"[f5-time] {name}: ops {min(res['off']):.1f} us, fused {min(res['ar']):.1f} us per all-reduce "
              f"(traced, 32 back-to-back, median of 10 replays; runs {res})", flush=True)
        ttnn.deallocate(x)
