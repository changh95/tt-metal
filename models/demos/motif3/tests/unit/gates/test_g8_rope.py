# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G8 — ``ttnn.experimental.rotary_embedding_hf`` (HF half-split RoPE, head dim 64) vs torch.

Design §1.2 (row 9), §2.3.4 decode step 5 / prefill step 1, §4.2 G8. Motif applies half-split (NeoX) RoPE
to the 64-dim q_pe of every head and to the single k_pe (01_motif_reference.md §3.10); tables are fp32
angles rounded to bf16 (HF), YaRN inv_freq on global layers and θ=1e4 on SWA layers.

* decode : q_pe [1, B=8, 10, 64] and k_pe [1, 8, 1, 64], height-sharded one user per core (shard [32, 64]);
           cos/sin [1, 8, 1, 64] gathered per user position, sharded on the same 8 cores.
* prefill: q_pe [1, 10, S, 64], k_pe [1, 1, S, 64], cos/sin [1, 1, S, 64], S ∈ {128, 1024, 4096}.
Golden: x·cos + rotate_half(x)·sin in fp32 with the bf16 tables. Pass: PCC ≥ 0.9999.
Composite fallback measured too: x·cos + (x @ R)·sin with the exact ±1 rotate-half matrix R [64, 64].
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

B, NH, DR = 8, 10, 64
DEC_POS = [0, 1, 127, 128, 129, 1000, 5000, 32767]
REC = gu.Recorder("G8")
FAB2D = gu.mesh_params()

CKC = {
    "default": None,  # op default: HiFi4, approx, no fp32 acc
    "hifi4_fp32acc": dict(fidelity="HiFi4", fp32_acc=True, approx=False, packer_l1_acc=True),
}


def tables(kind: str, positions: torch.Tensor):
    inv = gd.yarn_inv_freq() if kind == "global_yarn" else gd.plain_inv_freq()
    return gd.cos_sin(inv, positions, round_bf16=True)  # [n, 64] each, bf16-valued


def rot_matrix() -> torch.Tensor:
    """x @ R == rotate_half(x) = cat(-x[32:], x[:32])."""
    R = torch.zeros(DR, DR)
    h = DR // 2
    for i in range(h):
        R[i + h, i] = -1.0  # out[i] = -x[i+32]
        R[i, i + h] = 1.0  # out[i+32] = x[i]
    return R


def ckc_of(name):
    c = CKC[name]
    return None if c is None else gu.compute_cfg(c["fidelity"], fp32_acc=c["fp32_acc"], approx=c["approx"], packer_l1_acc=c["packer_l1_acc"])


def sharded_mc(mesh_device, rows: int):
    grid = ttnn.num_cores_to_corerangeset(B, gu.grid_size(mesh_device), row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(rows, DR), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True
    )


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
def test_g8_rope_decode(mesh_device):
    failures = []
    pos = torch.tensor(DEC_POS)
    R_tt = gu.to_mesh(rot_matrix().reshape(1, 1, DR, DR), mesh_device, ttnn.bfloat16)
    for kind in ("global_yarn", "swa_plain"):
        cos, sin = tables(kind, pos)  # [8, 64]
        cos_b = cos.reshape(1, B, 1, DR)
        sin_b = sin.reshape(1, B, 1, DR)
        cs_mc = sharded_mc(mesh_device, 32)
        # cos/sin [1, B, 1(->32), 64] sharded [32, 64] per user (same layout as the decode inputs)
        cos_tt = gu.to_mesh(cos_b, mesh_device, ttnn.bfloat16, memory_config=cs_mc)
        sin_tt = gu.to_mesh(sin_b, mesh_device, ttnn.bfloat16, memory_config=cs_mc)
        for tname, nheads in (("q_pe", NH), ("k_pe", 1)):
            x = torch.randn(1, B, nheads, DR).bfloat16().float()
            want = gd.rope_golden(x, cos_b, sin_b)
            x_tt = gu.to_mesh(x, mesh_device, ttnn.bfloat16, memory_config=sharded_mc(mesh_device, 32))
            for ck in CKC:
                case = f"decode/{kind}/{tname}/{ck}"

                def fn():
                    return ttnn.experimental.rotary_embedding_hf(x_tt, cos_tt, sin_tt, is_decode_mode=True, compute_kernel_config=ckc_of(ck))

                try:
                    got = gu.read_dev(fn())[:, :, :nheads, :]
                    eager = gu.time_eager(mesh_device, fn, iters=20)
                    traced, raw = gu.time_traced(mesh_device, fn, ops_per_trace=64, reps=9)
                except Exception as e:
                    REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
                    failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
                    continue
                s = gu.compare(want, got)
                ok = s["pcc"] >= 0.9999
                REC.add(case, status="pass" if ok else "fail", pcc=s["pcc"], max_abs=s["max_abs"], eager_us=eager, traced_us=traced, traced_raw_us=raw, positions=DEC_POS)
                if not ok:
                    failures.append(f"{case}: {gu.fmt(s)}")
            # composite fallback (interleaved L1): x*cos + (x@R)*sin, cos/sin broadcast over heads
            xi = gu.to_mesh(x, mesh_device, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ci = gu.to_mesh(cos_b, mesh_device, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            si = gu.to_mesh(sin_b, mesh_device, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ckc = gu.hifi4(fp32_acc=True)

            def comp():
                rot = ttnn.matmul(xi, R_tt, compute_kernel_config=ckc)
                return ttnn.add(ttnn.multiply(xi, ci), ttnn.multiply(rot, si))

            case = f"decode/{kind}/{tname}/composite_x@R"
            try:
                got = gu.read_dev(comp())[:, :, :nheads, :]
                eager = gu.time_eager(mesh_device, comp, iters=20)
                traced, raw = gu.time_traced(mesh_device, comp, ops_per_trace=64, reps=9)
                s = gu.compare(want, got)
                REC.add(case, status="pass" if s["pcc"] >= 0.9999 else "fail", pcc=s["pcc"], max_abs=s["max_abs"], eager_us=eager, traced_us=traced, traced_raw_us=raw)
            except Exception as e:
                REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
    assert not failures, "G8 decode failures:\n" + "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
@pytest.mark.parametrize("S", [128, 1024, 4096])
def test_g8_rope_prefill(mesh_device, S):
    failures = []
    R_tt = gu.to_mesh(rot_matrix().reshape(1, 1, DR, DR), mesh_device, ttnn.bfloat16)
    for kind in ("global_yarn", "swa_plain"):
        cos, sin = tables(kind, torch.arange(S))
        cos_t = cos.reshape(1, 1, S, DR)
        sin_t = sin.reshape(1, 1, S, DR)
        cos_tt = gu.to_mesh(cos_t, mesh_device, ttnn.bfloat16)
        sin_tt = gu.to_mesh(sin_t, mesh_device, ttnn.bfloat16)
        for tname, nheads in (("q_pe", NH), ("k_pe", 1)):
            x = torch.randn(1, nheads, S, DR).bfloat16().float()
            want = gd.rope_golden(x, cos_t, sin_t)
            x_tt = gu.to_mesh(x, mesh_device, ttnn.bfloat16)
            for ck in CKC:
                case = f"prefill/S{S}/{kind}/{tname}/{ck}"

                def fn():
                    return ttnn.experimental.rotary_embedding_hf(x_tt, cos_tt, sin_tt, is_decode_mode=False, compute_kernel_config=ckc_of(ck))

                try:
                    got = gu.read_dev(fn())
                    eager = gu.time_eager(mesh_device, fn, iters=10)
                except Exception as e:
                    REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
                    failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
                    continue
                s = gu.compare(want, got)
                ok = s["pcc"] >= 0.9999
                REC.add(case, status="pass" if ok else "fail", pcc=s["pcc"], max_abs=s["max_abs"], eager_us=eager)
                if not ok:
                    failures.append(f"{case}: {gu.fmt(s)}")

            def comp():
                rot = ttnn.matmul(x_tt, R_tt, compute_kernel_config=gu.hifi4(fp32_acc=True))
                return ttnn.add(ttnn.multiply(x_tt, cos_tt), ttnn.multiply(rot, sin_tt))

            case = f"prefill/S{S}/{kind}/{tname}/composite_x@R"
            try:
                got = gu.read_dev(comp())
                s = gu.compare(want, got)
                REC.add(case, status="pass" if s["pcc"] >= 0.9999 else "fail", pcc=s["pcc"], max_abs=s["max_abs"], eager_us=gu.time_eager(mesh_device, comp, iters=10))
            except Exception as e:
                REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
    assert not failures, "G8 prefill failures:\n" + "\n".join(failures)
