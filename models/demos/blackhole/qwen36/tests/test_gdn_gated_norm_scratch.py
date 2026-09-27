# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: op-level A/B of the GDN prefill chunk-body gated norm (round-4 item 14, zero-C++ stepping stone).

Reference = the current chain in gdn/tp.py forward_prefill (rms_norm -> reshape -> nlp_concat_heads -> reshape ->
multiply(., silu(z), dtype=bf16)). Candidates = ttnn.experimental.kda.sigmoid_gated_rms_norm(o, gate=z, w) followed by
multiply(., z, dtype=bf16) with an fp32 ("1") or bf16 ("bf16") intermediate -- exactly what
TPGatedDeltaNet._fused_gated_norm_prefill runs. Shapes: the TP=4 chunk (o [12, 2048, 128] bf16 head-major,
z [1, 2048, 1536] bf16, norm_w [128]). Reports PCC / max|d| / mismatch count vs the chain AND vs a float64 torch
reference (which candidate is closer to the truth), a NaN/inf check with |z| up to 40, and 50 traced replays per chain.

  pytest models/demos/blackhole/qwen36/tests/test_gdn_gated_norm_scratch.py -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2, "fabric_config": ttnn.FabricConfig.FABRIC_1D}]
NV, DV, T = 12, 128, int(os.environ.get("QWEN36_GN_T", "2048"))
REPLAYS = int(os.environ.get("QWEN36_GN_REPLAYS", "50"))
EPS = 1e-6


def _pcc(a, b):
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def _report(label, ref, out):
    d = (ref.double() - out.double()).abs()
    rel = d.max().item() / (ref.double().abs().max().item() + 1e-30)
    logger.info(
        f"[gn] {label:38s} pcc={_pcc(ref, out):.8f} max|d|={d.max().item():.4e} (rel {rel:.2e}) "
        f"mismatch={int((d != 0).sum())}/{d.numel()} finite={bool(torch.isfinite(out.double()).all())}"
    )
    return _pcc(ref, out)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_gdn_gated_norm_ab(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    rep = ttnn.ReplicateTensorToMesh(mesh)
    torch.manual_seed(0)
    # o ~ the scan output (bf16, O(1) with a per-head scale); z ~ the output gate with a wide tail (incl. |z| ~ 40).
    o = (torch.randn(NV, T, DV) * (0.5 + torch.rand(NV, 1, 1))).to(torch.bfloat16)
    z = (torch.randn(1, T, NV * DV) * 3.0).to(torch.bfloat16)
    z[0, :64, :256] = 40.0
    z[0, 64:128, :256] = -40.0
    w = (1.0 + 0.1 * torch.randn(DV)).to(torch.bfloat16)
    _dram, _L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG

    def up(t, dtype=ttnn.bfloat16, mc=_dram):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=mc)

    def first(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])

    # o as the fused chunk op hands it over: [Nv, NC, C, Dv] head-major (chunk 32).
    to = up(o.reshape(NV, T // 32, 32, DV))
    tz = up(z)
    w4 = up(w.reshape(1, 1, 1, DV))  # tw["norm_w"] is 4-D
    w1 = up(w)  # the fused op wants [Dv]
    exact = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    def chain():
        n = ttnn.rms_norm(to, weight=w4, epsilon=EPS, memory_config=_L1)
        n = ttnn.reshape(n, (1, NV, T, DV))
        n = ttnn.experimental.nlp_concat_heads(n, memory_config=_L1)
        out_f = ttnn.reshape(n, (1, T, NV * DV))
        s = ttnn.silu(tz, memory_config=_dram)
        gated = ttnn.multiply(out_f, s, dtype=ttnn.bfloat16, memory_config=_dram)
        ttnn.deallocate(out_f)
        ttnn.deallocate(s)
        return gated

    def fused(mid):
        o3 = ttnn.reshape(to, (NV, T, DV))
        gs = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            o3, tz, w1, NV, epsilon=EPS, memory_config=_dram, compute_kernel_config=exact, output_dtype=mid
        )
        gated = ttnn.multiply(gs, tz, dtype=ttnn.bfloat16, memory_config=_dram)
        ttnn.deallocate(gs)
        return gated

    variants = {
        "chain (current tree)": chain,
        "fused fp32 mid + mul": lambda: fused(ttnn.float32),
        "fused bf16 mid + mul": lambda: fused(ttnn.bfloat16),
    }
    # float64 truth: rmsnorm per (head, token) row over Dv, * w, * silu(z), head-concat -> [T, Nv*Dv]
    o64 = o.double()
    n64 = o64 * torch.rsqrt(o64.pow(2).mean(-1, keepdim=True) + EPS) * w.double()
    truth = (n64.permute(1, 0, 2).reshape(1, T, NV * DV) * torch.nn.functional.silu(z.double())).float()

    outs, times = {}, {}
    for label, fn in variants.items():
        g = fn()
        ttnn.synchronize_device(mesh)
        outs[label] = first(g).float()
        assert outs[label].shape == (1, T, NV * DV), (label, outs[label].shape)
        ttnn.deallocate(g)
        # 50 traced replays
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        g = fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        assert torch.equal(first(g).float(), outs[label]), f"{label}: traced replay differs from eager"
        t0 = time.perf_counter()
        for _ in range(REPLAYS):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        times[label] = 1e6 * (time.perf_counter() - t0) / REPLAYS
        ttnn.release_trace(mesh, tid)
        ttnn.deallocate(g)

    ref = outs["chain (current tree)"]
    assert bool(torch.isfinite(ref).all()), "reference chain produced non-finite values"
    logger.info(f"[gn] chain vs float64 truth: pcc={_pcc(truth, ref):.8f}")
    pccs = {}
    for label, out in outs.items():
        assert bool(torch.isfinite(out).all()), f"{label}: non-finite output (|z| up to 40)"
        if label != "chain (current tree)":
            pccs[label] = _report(f"{label} vs chain", ref, out)
        _report(f"{label} vs float64 truth", truth, out)
    print(
        "GATED_NORM_RESULT "
        + " ".join(f"[{k}]={v:.1f}us" for k, v in times.items())
        + " "
        + " ".join(f"pcc[{k}]={v:.6f}" for k, v in pccs.items())
    )
