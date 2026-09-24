# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: per-op time of the fused-conv ttnn.experimental.kda.gdn_decode_step alone (single device, trace replay).

Captures a trace of REPLAYS (50) back-to-back ops on a Bmax=32 state for B in QWEN36_GDN_TIMING_BS (default 1,32) and
reports wall time per op (best of 3 trace executions): device-only time including per-op dispatch, no host overhead.

  pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step_timing_scratch.py -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_gdn_decode_step_scratch import KD, VD, Dk, Dv, Nk, Nv, _pack_rows_user

BS = [int(b) for b in os.environ.get("QWEN36_GDN_TIMING_BS", "1,32").split(",")]
REPLAYS = int(os.environ.get("QWEN36_GDN_TIMING_REPLAYS", "50"))
# Probe knob: math fidelity of the op (LoFi|HiFi2|HiFi3|HiFi4; default = the op's HiFi4). Numerics change: timing only.
FIDELITY = os.environ.get("QWEN36_GDN_TIMING_FIDELITY", "")
BMAX = 32


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
def test_gdn_decode_step_conv_timing(device):
    torch.manual_seed(3)
    scale = Dk**-0.5
    C = 2 * KD + VD
    W = C + VD + 32
    az = C + VD
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev((0.05 * torch.randn(BMAX, Nv, Dk, Dv)).float(), ttnn.float32)
    w_dev = to_dev((1.0 + 0.1 * torch.randn(Dv)).bfloat16(), ttnn.bfloat16)
    taps_dev = to_dev(
        _pack_rows_user([(0.3 * torch.randn(C)).bfloat16() for _ in range(4)], 0, both=True), ttnn.bfloat16
    )
    hist_dev = to_dev(
        torch.stack([_pack_rows_user([(0.5 * torch.randn(C)).bfloat16() for _ in range(4)], b) for b in range(BMAX)]),
        ttnn.bfloat16,
    )
    dtb_dev = to_dev((0.1 * torch.randn(Nv)).bfloat16(), ttnn.bfloat16)
    nea_dev = to_dev((-torch.exp(0.2 * torch.randn(Nv))).bfloat16(), ttnn.bfloat16)
    ckc = None
    if FIDELITY:
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, FIDELITY),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        logger.info(f"GDN_OP_TIMING fidelity probe: {FIDELITY}")
    results = {}
    for B in BS:
        rows = (0.5 * torch.randn(B, W)).bfloat16()
        rows[:, az + 2 * Nv :] = 0
        qkv = to_dev(rows.reshape(1, B, W), ttnn.bfloat16)

        def op():
            return ttnn.experimental.kda.gdn_decode_step(
                qkv,
                dtb_dev,
                nea_dev,
                state,
                w_dev,
                Nv,
                Nk,
                Dk,
                Dv,
                scale=scale,
                output_dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                conv_hist=hist_dev,
                conv_taps=taps_dev,
                qkvz_dim=az,
                compute_kernel_config=ckc,
            )

        ttnn.deallocate(op())  # compile + program cache
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        outs = [op() for _ in range(REPLAYS)]
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            best = min(best, time.perf_counter() - t0)
        ttnn.release_trace(device, tid)
        for o in outs:
            ttnn.deallocate(o)
        ttnn.deallocate(qkv)
        results[B] = best / REPLAYS * 1e6
        logger.info(f"GDN_OP_TIMING B={B}: {results[B]:.1f} us per op ({REPLAYS} traced replays, best of 3)")
    print(
        f"GDN_OP_TIMING{'[' + FIDELITY + ']' if FIDELITY else ''} "
        + " ".join(f"B{b}={us:.1f}us" for b, us in results.items()),
        flush=True,
    )
