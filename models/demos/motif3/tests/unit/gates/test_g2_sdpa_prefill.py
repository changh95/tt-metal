# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G2 — ``ttnn.transformer.scaled_dot_product_attention`` prefill at Motif per-chip shapes.

Design §1.2 (row 2), §2.3.4 prefill step 3, §3.3, §4.2 G2. Per chip (TP8 over heads, one user):
  q [1, 10, S, 192]   (10 q heads = KV groups 2c, 2c+1; GQA 10/2)
  k [1,  2, S, 192]   k = [k_nope 128 ‖ k_pe 64]
  v [1,  2, S, 192]   V zero-padded 128 -> 192 (the non-MLA op requires dv == dqk; confirmed below)
  is_causal=True, sliding_window_size=129 (SWA, scale 0.07216878) or None (global, scale 0.14467963)
  S ∈ {128, 1024, 4096}

Golden: fp32 masked attention with the HF grouping (q head h -> kv group h // 5) and the FA2 window
(keys [p-128, p]). Pass: PCC ≥ 0.999 on the sliced [..., :128] output. Latency: eager wall clock
(prefill is eager in draft 1). Also checks: V with head dim 128 is rejected by the plain op, and the
``flash_mla_prefill(q, k, v)`` separate-V overload as a padding-free alternative for global layers.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

NQ, NKV, DQK, DV = 10, 2, 192, 128
PCC_MIN = 0.999
REC = gu.Recorder("G2")
FAB2D = gu.mesh_params()

LAYERS = [("swa", gd.WINDOW, gd.SCALE_SWA), ("global", None, gd.SCALE_GLOBAL)]


def make_qkv(S: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, NQ, S, DQK, generator=g).bfloat16().float()
    k = torch.randn(1, NKV, S, DQK, generator=g).bfloat16().float()
    v = torch.randn(1, NKV, S, DV, generator=g).bfloat16().float()
    v_pad = torch.zeros(1, NKV, S, DQK)
    v_pad[..., :DV] = v
    return q, k, v, v_pad


def sdpa_pc(mesh_device, qc: int, kc: int, exp_approx: bool = False):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=gu.grid_size(mesh_device),
        q_chunk_size=qc,
        k_chunk_size=kc,
        exp_approx_mode=exp_approx,
    )


CKC = {
    "hifi2": dict(fidelity="HiFi2", fp32_acc=False, approx=True),  # the op default (sdpa.cpp:51-52)
    "hifi4": dict(fidelity="HiFi4", fp32_acc=False, approx=False),
}


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
@pytest.mark.parametrize("S", [128, 1024, 4096])
def test_g2_sdpa_prefill(mesh_device, S):
    failures = []
    q, k, v, v_pad = make_qkv(S, seed=S)
    q_tt = gu.to_mesh(q, mesh_device, ttnn.bfloat16)
    k_tt = gu.to_mesh(k, mesh_device, ttnn.bfloat16)
    vpad_tt = gu.to_mesh(v_pad, mesh_device, ttnn.bfloat16)
    chunk_opts = [(128, 128)] if S == 128 else [(128, 128), (256, 256), (128, 256)]
    if S >= 4096:
        chunk_opts.append((512, 512))

    replica_checked = False
    for name, window, scale in LAYERS:
        want = gd.gqa_prefill_golden(q[0], k[0], v[0], scale, window)  # [NQ, S, 128]
        for ck_name, ck in CKC.items():
            for qc, kc in chunk_opts:
                case = f"S{S}/{name}/{ck_name}/q{qc}_k{kc}"
                pc = sdpa_pc(mesh_device, qc, kc, exp_approx=ck["approx"])
                ckc = gu.compute_cfg(ck["fidelity"], fp32_acc=ck["fp32_acc"], approx=ck["approx"])

                def fn():
                    return ttnn.transformer.scaled_dot_product_attention(
                        q_tt,
                        k_tt,
                        vpad_tt,
                        is_causal=True,
                        scale=scale,
                        sliding_window_size=window,
                        program_config=pc,
                        compute_kernel_config=ckc,
                    )

                try:
                    out_tt = fn()
                    full = gu.read_dev(out_tt, 0)[0, :, :S, :]
                    got, pad_cols = full[..., :DV], full[..., DV:]
                    same = None
                    if not replica_checked:  # one 32-replica bitwise check per S (readback is S·NQ·192·2 B per chip)
                        same, _ = gu.replicas_identical(out_tt)
                        replica_checked = True
                    ttnn.deallocate(out_tt)
                    eager = gu.time_eager(mesh_device, fn, iters=5 if S >= 4096 else 10, warmup=1)
                except Exception as e:
                    msg = str(e)
                    l1_overflow = "beyond max L1 size" in msg or "clash with L1 buffers" in msg
                    REC.add(case, status="unsupported_L1" if l1_overflow else "error", error=f"{type(e).__name__}: {msg[:400]}")
                    if not l1_overflow:  # chunk configs whose CBs do not fit L1 are a tuning limit, not a gate failure
                        failures.append(f"{case}: {type(e).__name__}: {msg[:200]}")
                    continue
                s = gu.compare(want, got)
                ok = s["pcc"] >= PCC_MIN and s["nonfinite"] == 0 and (same is None or same)
                REC.add(
                    case,
                    status="pass" if ok else "fail",
                    pcc=s["pcc"],
                    max_abs=s["max_abs"],
                    mean_abs=s["mean_abs"],
                    pad_cols_absmax=float(pad_cols.abs().max()),
                    replicas_identical=same,
                    eager_us=eager,
                    program_config=f"q_chunk={qc} k_chunk={kc} exp_approx={ck['approx']}",
                    compute=ck,
                    dtype="bf16 q/k/v, DRAM interleaved",
                )
                if not ok:
                    failures.append(f"{case}: {gu.fmt(s)}")

    # bfp8 K/V (bandwidth lever) at the largest S only
    if S == 4096:
        k8 = gu.to_mesh(k, mesh_device, ttnn.bfloat8_b)
        v8 = gu.to_mesh(v_pad, mesh_device, ttnn.bfloat8_b)
        kq = gu.host_roundtrip(k, ttnn.bfloat8_b)
        vq = gu.host_roundtrip(v_pad, ttnn.bfloat8_b)[..., :DV]
        for name, window, scale in LAYERS:
            pc = sdpa_pc(mesh_device, 256, 256)
            ckc = gu.compute_cfg("HiFi2", fp32_acc=False, approx=True)

            def fn8():
                return ttnn.transformer.scaled_dot_product_attention(
                    q_tt,
                    k8,
                    v8,
                    is_causal=True,
                    scale=scale,
                    sliding_window_size=window,
                    program_config=pc,
                    compute_kernel_config=ckc,
                )

            case = f"S{S}/{name}/hifi2/q256_k256/kv_bfp8"
            try:
                got = gu.read_dev(fn8(), 0)[0, :, :S, :DV]
                eager = gu.time_eager(mesh_device, fn8, iters=5, warmup=1)
                s = gu.compare(gd.gqa_prefill_golden(q[0], kq[0], vq[0], scale, window), got)
                REC.add(case, status="measured", pcc=s["pcc"], max_abs=s["max_abs"], eager_us=eager)
            except Exception as e:
                REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:300]}")

    assert not failures, "G2 failures:\n" + "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
def test_g2_v_padding_and_mla_alternative(mesh_device):
    """Confirms V must be zero-padded to 192 for the plain op; probes ``flash_mla_prefill(q, k, v)`` (no window)."""
    S = 1024
    q, k, v, v_pad = make_qkv(S, seed=7)
    q_tt = gu.to_mesh(q, mesh_device, ttnn.bfloat16)
    k_tt = gu.to_mesh(k, mesh_device, ttnn.bfloat16)
    v_tt = gu.to_mesh(v, mesh_device, ttnn.bfloat16)
    pc = sdpa_pc(mesh_device, 256, 256)
    ckc = gu.compute_cfg("HiFi2", fp32_acc=False, approx=True)

    # (a) unpadded V into the plain op: expected to be rejected (sdpa_device_operation.cpp K/V dim check)
    try:
        out = ttnn.transformer.scaled_dot_product_attention(
            q_tt,
            k_tt,
            v_tt,
            is_causal=True,
            scale=gd.SCALE_SWA,
            sliding_window_size=gd.WINDOW,
            program_config=pc,
            compute_kernel_config=ckc,
        )
        got = gu.read_dev(out, 0)[0, :, :S, :DV]
        s = gu.compare(gd.gqa_prefill_golden(q[0], k[0], v[0], gd.SCALE_SWA, gd.WINDOW), got)
        REC.add("v128_into_plain_sdpa", status="accepted", pcc=s["pcc"], note="op accepted dv=128 != dqk=192")
    except Exception as e:
        REC.add("v128_into_plain_sdpa", status="rejected", error=f"{type(e).__name__}: {str(e)[:300]}")

    # (b) MLA prefill with a separate, narrower V (global layers only: the MLA op has no sliding window)
    def fn():
        return ttnn.transformer.flash_mla_prefill(
            q_tt, k_tt, v_tt, is_causal=True, scale=gd.SCALE_GLOBAL, program_config=pc, compute_kernel_config=ckc
        )

    try:
        got = gu.read_dev(fn(), 0)[0, :, :S, :DV]
        eager = gu.time_eager(mesh_device, fn, iters=10, warmup=1)
        s = gu.compare(gd.gqa_prefill_golden(q[0], k[0], v[0], gd.SCALE_GLOBAL, None), got)
        REC.add(
            "flash_mla_prefill_separate_v128_global",
            status="pass" if s["pcc"] >= PCC_MIN else "fail",
            pcc=s["pcc"],
            max_abs=s["max_abs"],
            eager_us=eager,
        )
    except Exception as e:
        REC.add("flash_mla_prefill_separate_v128_global", status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
