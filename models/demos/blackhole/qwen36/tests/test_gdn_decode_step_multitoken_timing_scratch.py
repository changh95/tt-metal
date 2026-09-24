# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: per-op time of the MULTI-TOKEN (verify) mode of ttnn.experimental.kda.gdn_decode_step (trace replay).

Same method as test_gdn_decode_step_timing_scratch.py: REPLAYS (50) back-to-back ops captured in one trace on a
Bmax=32 state, wall time per op (best of 3 executions) = device time incl. per-op dispatch. Cases (B, T) from
QWEN36_GDN_MT_TIMING (default "1x8,8x8,32x4,32x8,1x1,32x1"; T = 1 runs the one-token op as the reference) with the
accept pattern QWEN36_GDN_MT_ACCEPT in {rand (uniform 0..k, seeded), zero, full (k)}; rand and full by default.

  pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step_multitoken_timing_scratch.py -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_gdn_decode_step_scratch import KD, VD, Dk, Dv, Nk, Nv, _pack_rows_user

CASES = [
    tuple(int(x) for x in c.split("x"))
    for c in os.environ.get("QWEN36_GDN_MT_TIMING", "1x8,8x8,32x4,32x8,1x1,32x1").split(",")
]
REPLAYS = int(os.environ.get("QWEN36_GDN_TIMING_REPLAYS", "50"))
ACCEPTS = os.environ.get("QWEN36_GDN_MT_ACCEPT", "rand,full").split(",")
BMAX = 32
C = 2 * KD + VD
W = C + VD + 32
AZ = C + VD


@pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
def test_gdn_decode_step_multitoken_timing(device):
    torch.manual_seed(3)
    scale = Dk**-0.5
    to_dev = lambda t, dt, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=layout, device=device)
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
    results = []
    for B, T in CASES:
        k = T - 1
        R = ((B * T + 31) // 32) * 32
        if T == 1:
            rows = (0.5 * torch.randn(B, W)).bfloat16()
            rows[:, AZ + 2 * Nv :] = 0
            qkv = to_dev(rows.reshape(1, B, W), ttnn.bfloat16)
            variants = [("t1", None, None, 0.0)]
        else:
            rows = (0.5 * torch.randn(R, W)).bfloat16()
            rows[:, AZ + 2 * Nv :] = 0
            qkv = to_dev(rows.reshape(1, 1, R, W), ttnn.bfloat16)
            prev = to_dev(rows.reshape(1, 1, R, W), ttnn.bfloat16)
            variants = []
            for pat in ACCEPTS:
                a = {"rand": torch.randint(0, k + 1, (B,)), "zero": torch.zeros(B), "full": torch.full((B,), k)}[pat]
                a = a.to(torch.int32)
                variants.append((pat, prev, to_dev(a, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), a.float().mean().item()))
        for pat, prev, acc, mean_a in variants:

            def op():
                kw = dict(num_tokens=T, qkv_prev=prev, accept=acc) if T > 1 else {}
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
                    qkvz_dim=AZ,
                    **kw,
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
            us = best / REPLAYS * 1e6
            results.append((B, T, pat, mean_a, us))
            logger.info(
                f"GDN_MT_TIMING B={B} T={T} accept={pat} (mean a={mean_a:.2f}): {us:.1f} us per op "
                f"({REPLAYS} traced replays, best of 3)"
            )
            if acc is not None:
                ttnn.deallocate(acc)
        ttnn.deallocate(qkv)
        if T > 1:
            ttnn.deallocate(variants[0][1])
    print("GDN_MT_TIMING " + " ".join(f"B{b}T{t}[{p},a={a:.1f}]={us:.1f}us" for b, t, p, a, us in results), flush=True)
