# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G0 — wall-clock timing calibration for the gate harness on the 32-chip (4, 8) mesh.

Not a design gate: it measures how ``execute_trace`` + ``synchronize_device`` and eager dispatch scale with
the number of ops, so the per-op latencies reported by G1-G8 can be interpreted (replay floor, linearity of
t(n) in n, eager host-dispatch cost per op on 32 chips).
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu

REC = gu.Recorder("G0")


@pytest.mark.parametrize("mesh_device, device_params", [gu.mesh_params()], indirect=True, ids=["4x8_torus2d"])
def test_g0_timing_calibration(mesh_device):
    small = gu.to_mesh(torch.zeros(1, 1, 32, 32), mesh_device, ttnn.bfloat16)
    mid = gu.to_mesh(torch.randn(1, 1, 32, 4096), mesh_device, ttnn.bfloat16)
    big = gu.to_mesh(torch.randn(1, 1, 1024, 4096), mesh_device, ttnn.bfloat16)
    ops = {
        "add_1tile": lambda: ttnn.add(small, 1.0),
        "add_32x4096": lambda: ttnn.add(mid, 1.0),
        "add_1024x4096": lambda: ttnn.add(big, 1.0),
    }
    for name, fn in ops.items():
        o = fn()
        ttnn.synchronize_device(mesh_device)
        del o
        for n in (1, 2, 4, 8, 16, 32, 64):
            times = gu._trace_total_us(mesh_device, fn, n, reps=7)
            REC.add(f"trace/{name}/n{n}", status="info", n=n, total_us_sorted=[round(t, 1) for t in times], median_us=times[3], per_op_median_us=times[3] / n)
        for n in (1, 8, 32):
            ts = []
            for _ in range(5):
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                outs = [fn() for _ in range(n)]
                ttnn.synchronize_device(mesh_device)
                ts.append((time.perf_counter() - t0) * 1e6)
                del outs
            ts.sort()
            REC.add(f"eager/{name}/n{n}", status="info", n=n, total_us_sorted=[round(t, 1) for t in ts], per_op_median_us=ts[2] / n)
    # synchronize-only cost
    ts = []
    for _ in range(9):
        t0 = time.perf_counter()
        ttnn.synchronize_device(mesh_device)
        ts.append((time.perf_counter() - t0) * 1e6)
    REC.add("synchronize_only", status="info", us_sorted=[round(t, 1) for t in sorted(ts)])
