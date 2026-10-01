# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G6 — dense-local expert matmuls (bfp8 weights in DRAM) and the composite PolyNorm, per chip.

Design §1.2 (row 8), §2.3.5, §2.3.7 decode steps 5-7, §3.2, §3.6, §4.2 G6, §7.2 Q3 (the bandwidth that sets
draft-1 TPOT within ±40 %). Per chip (EP32: 12 experts) for one decode step (32 gathered tokens):
  gate_up : [1,12,32,4096] bf16 @ [1,12,4096,2560] bfp8 -> [1,12,32,2560]   (gate = [:1280], up = [1280:])
  down    : [1,12,32,1280] bf16 @ [1,12,1280,4096] bfp8 -> [1,12,32,4096]
  PolyNorm: h = (c0·N(g³) + c1·N(g²) + c2·N(g) + b)·u, N = weightless rms_norm over 1280 (eps 1e-6),
            c_k = σ(w_e,k), b = clamp(b_e, ±0.5) as [1,12,1,1] broadcast tensors (×0.5 folded into W_down).
Data: chip-0's real experts 0..11 of layer 2 (BF16 shards in weights/Motif-3, sliced per expert) when present,
else synthetic; inputs f = γ_post ⊙ rmsnorm(h). Effective GB/s = bfp8 weight bytes / traced time.
Also measures the flattened alternatives the dense-local formulation allows (all 12 experts see the same
32 tokens): gate_up as one [32,4096]@[4096,30720] and down as one [32,15360]@[15360,4096] (routing weights
folded into h before the K-concat), both with DRAM width-sharded weights.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

E_LOCAL, T, HID, INTER = 12, 32, 4096, 1280
REC = gu.Recorder("G6")
FAB2D = gu.mesh_params()
WEIGHTS_DIR = Path(os.environ.get("MOTIF3_WEIGHTS", "/home/ttuser/hchang/experiments/motif-3/weights/Motif-3"))
BFP8_BYTES_PER_ELEM = 1088 / 1024


def load_real_experts(layer: int = 2, first: int = 0):
    try:
        from safetensors import safe_open

        idx = json.load(open(WEIGHTS_DIR / "model.safetensors.index.json"))["weight_map"]
        pre = f"model.layers.{layer}"
        tensors = {}
        for key, name in {
            "gu": f"{pre}.moe.experts.gate_up_proj",
            "dn": f"{pre}.moe.experts.down_proj",
            "w": f"{pre}.moe.experts.act_fn.weight",
            "b": f"{pre}.moe.experts.act_fn.bias",
            "gamma": f"{pre}.post_attention_layernorm.weight",
        }.items():
            with safe_open(str(WEIGHTS_DIR / idx[name]), framework="pt") as f:
                if key in ("gu", "dn", "w", "b"):
                    tensors[key] = f.get_slice(name)[first : first + E_LOCAL].float()
                else:
                    tensors[key] = f.get_tensor(name).float()
        w_gu = tensors["gu"].transpose(1, 2).contiguous()  # [12, 4096, 2560]: x @ W  (rows 0..1279 = gate, 1280.. = up)
        w_dn = tensors["dn"].transpose(1, 2).contiguous()  # [12, 1280, 4096]
        c = torch.sigmoid(tensors["w"])  # [12, 3]
        b = tensors["b"].reshape(E_LOCAL).clamp(-0.5, 0.5)
        return w_gu, w_dn, c, b, tensors["gamma"], f"real_L{layer}_e{first}-{first + E_LOCAL - 1}"
    except Exception as e:
        print(f"[G6] real experts unavailable ({type(e).__name__}: {e}); using synthetic")
        return None


def synthetic_experts(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w_gu = (0.02 * torch.randn(E_LOCAL, HID, 2 * INTER, generator=g)).bfloat16().float()
    w_dn = (0.02 * torch.randn(E_LOCAL, INTER, HID, generator=g)).bfloat16().float()
    c = 0.12 + 0.85 * torch.rand(E_LOCAL, 3, generator=g)
    b = (torch.rand(E_LOCAL, generator=g) - 0.8) * 0.6
    gamma = 0.05 + 0.2 * torch.rand(HID, generator=g)
    return w_gu, w_dn, c, b, gamma, "synthetic"


def router_input(gamma: torch.Tensor, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(T, HID, generator=g)
    return (h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * gamma).bfloat16().float()


def mm_cfg(fidelity: str):
    return gu.compute_cfg(fidelity, fp32_acc=True, approx=False, packer_l1_acc=True)


def reuse_cfg(per_core_n: int, in0_block_w: int, grid=(12, 10)):
    # fp32 dest accumulation halves DEST to 4 tiles, so out_subblock_h * out_subblock_w <= 4
    sub_w = max(d for d in (1, 2, 4) if per_core_n % d == 0)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        per_core_M=1,
        per_core_N=per_core_n,
    )


def mcast1d_cfg(grid: tuple, n_tiles: int, in0_block_w: int):
    """MatmulMultiCoreReuseMultiCast1DProgramConfig over grid=(x, y) cores, N split evenly (fuse_batch=False)."""
    ncores = grid[0] * grid[1]
    assert n_tiles % ncores == 0, (n_tiles, ncores)
    pcn = n_tiles // ncores
    sub_w = max(d for d in (1, 2, 4) if pcn % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        out_block_h=1,
        out_block_w=pcn,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


class Weights:
    def __init__(self, mesh_device, w_gu, w_dn, c, b):
        self.gu = gu.to_mesh(w_gu.reshape(1, E_LOCAL, HID, 2 * INTER), mesh_device, ttnn.bfloat8_b)
        self.dn = gu.to_mesh(w_dn.reshape(1, E_LOCAL, INTER, HID), mesh_device, ttnn.bfloat8_b)
        self.gu_q = gu.host_roundtrip(w_gu.reshape(1, E_LOCAL, HID, 2 * INTER), ttnn.bfloat8_b)[0]
        self.dn_q = gu.host_roundtrip(w_dn.reshape(1, E_LOCAL, INTER, HID), ttnn.bfloat8_b)[0]
        coef = lambda v: gu.to_mesh(v.reshape(1, E_LOCAL, 1, 1).float(), mesh_device, ttnn.float32)
        self.c0, self.c1, self.c2 = coef(c[:, 0]), coef(c[:, 1]), coef(c[:, 2])
        self.b = coef(b)
        coef16 = lambda v: gu.to_mesh(v.reshape(1, E_LOCAL, 1, 1).float(), mesh_device, ttnn.bfloat16)
        self.c0h, self.c1h, self.c2h, self.bh = coef16(c[:, 0]), coef16(c[:, 1]), coef16(c[:, 2]), coef16(b)


def polynorm_tt(gu_tt, W: Weights, mode: str):
    """Composite grouped PolyNorm·up on [1,12,32,2560] (gate | up). mode: 'fp32' or 'bf16' intermediates."""
    g = ttnn.slice(gu_tt, [0, 0, 0, 0], [1, E_LOCAL, T, INTER])
    u = ttnn.slice(gu_tt, [0, 0, 0, INTER], [1, E_LOCAL, T, 2 * INTER])
    ckc = gu.hifi4(fp32_acc=True)
    if mode == "fp32":
        g = ttnn.typecast(g, ttnn.float32)
        u = ttnn.typecast(u, ttnn.float32)
        c0, c1, c2, b = W.c0, W.c1, W.c2, W.b
    else:
        c0, c1, c2, b = W.c0h, W.c1h, W.c2h, W.bh
    g2 = ttnn.multiply(g, g)
    g3 = ttnn.multiply(g2, g)
    n1 = ttnn.rms_norm(g, epsilon=1e-6, compute_kernel_config=ckc)
    n2 = ttnn.rms_norm(g2, epsilon=1e-6, compute_kernel_config=ckc)
    n3 = ttnn.rms_norm(g3, epsilon=1e-6, compute_kernel_config=ckc)
    t = ttnn.multiply(n3, c0)
    t = ttnn.add(t, ttnn.multiply(n2, c1))
    t = ttnn.add(t, ttnn.multiply(n1, c2))
    t = ttnn.add(t, b)
    return ttnn.multiply(t, u, dtype=ttnn.bfloat16)


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
def test_g6_expert_matmuls_and_polynorm(mesh_device):
    src = load_real_experts() or synthetic_experts()
    w_gu, w_dn, c, b, gamma, tag = src
    f = router_input(gamma)
    W = Weights(mesh_device, w_gu, w_dn, c, b)
    x = f.reshape(1, 1, T, HID).expand(1, E_LOCAL, T, HID).contiguous()
    x_tt = gu.to_mesh(x, mesh_device, ttnn.bfloat16)
    gu_bytes = E_LOCAL * HID * 2 * INTER * BFP8_BYTES_PER_ELEM
    dn_bytes = E_LOCAL * INTER * HID * BFP8_BYTES_PER_ELEM
    REC.add(
        f"{tag}/data",
        status="info",
        gu_bytes=gu_bytes,
        dn_bytes=dn_bytes,
        c_range=[float(c.min()), float(c.max())],
        b_range=[float(b.min()), float(b.max())],
    )
    failures = []

    # ---------------- gate_up ----------------
    gate_cfgs = [
        ("default", None),
        ("mcast1d_80c_k8", mcast1d_cfg((10, 8), 80, 8)),
        ("mcast1d_40c_k8", mcast1d_cfg((8, 5), 80, 8)),
        ("mcast1d_40c_k16", mcast1d_cfg((8, 5), 80, 16)),
        ("mcast1d_20c_k8", mcast1d_cfg((5, 4), 80, 8)),
    ]
    gu_out = None
    want_gu = torch.matmul(x[0].double(), W.gu_q.double()).float()  # [12, 32, 2560] (golden on the quantised weights)
    want_gu_e2e = torch.matmul(x[0].double(), w_gu.double()).float()
    for fid in ("HiFi2", "HiFi4"):
        for name, pc in gate_cfgs:
            case = f"{tag}/gate_up/{name}/{fid}"

            def fn():
                return ttnn.matmul(
                    x_tt, W.gu, program_config=pc, compute_kernel_config=mm_cfg(fid), dtype=ttnn.bfloat16
                )

            try:
                out = fn()
                got = gu.read_dev(out)[0]
                eager = gu.time_eager(mesh_device, fn, iters=10)
                traced, raw = gu.time_traced(mesh_device, fn, ops_per_trace=24, reps=5)
            except Exception as e:
                REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
                continue
            s = gu.compare(want_gu, got)
            s2 = gu.compare(want_gu_e2e, got)
            REC.add(
                case,
                status="pass" if s["pcc"] >= 0.999 else "fail",
                pcc=s["pcc"],
                pcc_vs_bf16_weights=s2["pcc"],
                max_abs=s["max_abs"],
                eager_us=eager,
                traced_us=traced,
                traced_raw_us=raw,
                eff_GBps=gu_bytes / (max(traced, 1e-3) * 1e-6) / 1e9,
            )
            if name == "default" and fid == "HiFi2":
                gu_out = out
                if s["pcc"] < 0.999:
                    failures.append(f"{case}: {gu.fmt(s)}")

    # ---------------- PolyNorm (on the device gate_up output, so the golden isolates PolyNorm) ----------------
    gu_dev = gu.read_dev(gu_out)[0]  # [12, 32, 2560] bf16 values
    g_dev, u_dev = gu_dev[..., :INTER], gu_dev[..., INTER:]
    want_h = gd.grouped_polynorm_golden(g_dev, u_dev, c, b)
    h_out = {}
    for mode in ("fp32", "bf16"):
        case = f"{tag}/polynorm/{mode}_intermediates"
        try:
            h_tt = polynorm_tt(gu_out, W, mode)
            got = gu.read_dev(h_tt)[0]
            eager = gu.time_eager(mesh_device, lambda: polynorm_tt(gu_out, W, mode), iters=10)
            traced, raw = gu.time_traced(mesh_device, lambda: polynorm_tt(gu_out, W, mode), ops_per_trace=32, reps=5)
        except Exception as e:
            REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
            failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
            continue
        s = gu.compare(want_h, got)
        ok = s["pcc"] >= 0.9995
        REC.add(
            case,
            status="pass" if ok else "fail",
            pcc=s["pcc"],
            max_abs=s["max_abs"],
            ref_absmax=s["ref_absmax"],
            rel_fro=s["rel_fro"],
            eager_us=eager,
            traced_us=traced,
            traced_raw_us=raw,
            ops=13,
        )
        h_out[mode] = h_tt
        if not ok and mode == "fp32":
            failures.append(f"{case}: {gu.fmt(s)}")

    # ---------------- down ----------------
    h_tt = h_out.get("fp32")
    if h_tt is not None:
        h_dev = gu.read_dev(h_tt)[0]
        want_dn = torch.matmul(h_dev.double(), W.dn_q.double()).float()
        dn_cfgs = [
            ("default", None),
            ("mcast1d_64c_k8", mcast1d_cfg((8, 8), 128, 8)),
            ("mcast1d_32c_k8", mcast1d_cfg((8, 4), 128, 8)),
            ("mcast1d_32c_k4", mcast1d_cfg((8, 4), 128, 4)),
            ("mcast1d_16c_k8", mcast1d_cfg((8, 2), 128, 8)),
        ]
        for fid in ("HiFi2", "HiFi4"):
            for name, pc in dn_cfgs:
                case = f"{tag}/down/{name}/{fid}"

                def fn():
                    return ttnn.matmul(
                        h_tt, W.dn, program_config=pc, compute_kernel_config=mm_cfg(fid), dtype=ttnn.bfloat16
                    )

                try:
                    got = gu.read_dev(fn())[0]
                    eager = gu.time_eager(mesh_device, fn, iters=10)
                    traced, raw = gu.time_traced(mesh_device, fn, ops_per_trace=24, reps=5)
                except Exception as e:
                    REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
                    continue
                s = gu.compare(want_dn, got)
                REC.add(
                    case,
                    status="pass" if s["pcc"] >= 0.999 else "fail",
                    pcc=s["pcc"],
                    max_abs=s["max_abs"],
                    eager_us=eager,
                    traced_us=traced,
                    traced_raw_us=raw,
                    eff_GBps=dn_bytes / (max(traced, 1e-3) * 1e-6) / 1e9,
                )
                if name == "default" and fid == "HiFi2" and s["pcc"] < 0.999:
                    failures.append(f"{case}: {gu.fmt(s)}")
    assert not failures, "G6 failures:\n" + "\n".join(failures)


# ------------------------------------------------------------------------------------------------
# flattened, DRAM-sharded alternatives (bandwidth ceiling for the dense-local formulation)
# ------------------------------------------------------------------------------------------------
def dram_sharded_weight(mesh_device, w: torch.Tensor, dtype=ttnn.bfloat8_b):
    K, N = w.shape
    banks = mesh_device.dram_grid_size().x
    n_pad = math.ceil(N / (32 * banks)) * 32 * banks
    wp = torch.zeros(K, n_pad)
    wp[:, :N] = w
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    spec = ttnn.ShardSpec(grid, [K, n_pad // banks], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)
    return gu.to_mesh(wp.reshape(1, 1, K, n_pad), mesh_device, dtype, memory_config=mc), n_pad


def dram_sharded_matmul(mesh_device, x: torch.Tensor, w: torch.Tensor, cores: tuple, fid: str):
    """x [32, K] bf16 @ w [K, N] bfp8 DRAM-sharded; in0 width-sharded on cores (x, y); returns (fn, golden)."""
    K, N = w.shape
    w_tt, n_pad = dram_sharded_weight(mesh_device, w)
    ncores = cores[0] * cores[1]
    assert K % (32 * ncores) == 0 and n_pad % (32 * ncores) == 0, (K, n_pad, ncores)
    x_tt = gu.to_mesh(x.reshape(1, 1, T, K), mesh_device, ttnn.bfloat16)
    x_sh = ttnn.interleaved_to_sharded(
        x_tt,
        ttnn.CoreCoord(cores[0], cores[1]),
        [T, K // ncores],
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    k_tiles_per_core = K // ncores // 32
    in0_bw = max(d for d in range(1, min(k_tiles_per_core, 8) + 1) if k_tiles_per_core % d == 0)
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_bw, per_core_M=1, per_core_N=n_pad // ncores // 32, fused_activation=None
    )
    out_mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
    ckc = mm_cfg(fid)

    def fn():
        return ttnn.matmul(
            x_sh, w_tt, program_config=pc, memory_config=out_mc, dtype=ttnn.bfloat16, compute_kernel_config=ckc
        )

    wq = gu.host_roundtrip(w, ttnn.bfloat8_b)
    return (
        fn,
        (x.double() @ wq.double()).float(),
        n_pad,
        f"in0_block_w={in0_bw} per_core_N={n_pad // ncores // 32} cores={cores}",
    )


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
def test_g6_flattened_dram_sharded(mesh_device):
    src = load_real_experts() or synthetic_experts()
    w_gu, w_dn, c, b, gamma, tag = src
    f = router_input(gamma)
    w_gu_flat = w_gu.permute(1, 0, 2).reshape(HID, E_LOCAL * 2 * INTER)  # [4096, 30720]: expert-major columns
    w_dn_flat = w_dn.reshape(E_LOCAL * INTER, HID)  # [15360, 4096]: expert-major rows
    h = torch.randn(T, E_LOCAL * INTER).bfloat16().float()
    for name, x, w, core_opts in [
        ("gate_up_flat_4096x30720", f, w_gu_flat, [(8, 4), (8, 8), (8, 2)]),
        ("down_flat_15360x4096", h, w_dn_flat, [(8, 4), (8, 2), (4, 4)]),
    ]:
        nbytes = w.numel() * BFP8_BYTES_PER_ELEM
        for cores in core_opts:
            for fid in ("HiFi2",):
                case = f"{tag}/{name}/dram_sharded/{cores[0]}x{cores[1]}/{fid}"
                try:
                    fn, want, n_pad, cfg = dram_sharded_matmul(mesh_device, x, w, cores, fid)
                    got = gu.read_dev(ttnn.sharded_to_interleaved(fn(), ttnn.DRAM_MEMORY_CONFIG))[
                        0, 0, :T, : w.shape[1]
                    ]
                    eager = gu.time_eager(mesh_device, fn, iters=10)
                    traced, raw = gu.time_traced(mesh_device, fn, ops_per_trace=24, reps=5)
                except Exception as e:
                    REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
                    continue
                s = gu.compare(want, got)
                REC.add(
                    case,
                    status="measured",
                    pcc=s["pcc"],
                    max_abs=s["max_abs"],
                    program_config=cfg,
                    eager_us=eager,
                    traced_us=traced,
                    traced_raw_us=raw,
                    eff_GBps=nbytes / (max(traced, 1e-3) * 1e-6) / 1e9,
                )
