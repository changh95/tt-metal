# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: is the decode reduce-scatter (tt_all_reduce -> reduce_scatter_minimal_async) row-position dependent at
> 1 tile row, and does an fp32 reduce reproduce the fused decode all_reduce_async bitwise? (no model weights)"""
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import tt_all_reduce


def _t0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


@parametrize_mesh_tp()
def test_rs_probes(mesh_device):
    mesh = mesh_device
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

    args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=32, max_seq_len=4096)
    ccl = tpc.Qwen36CCL(mesh, args)
    dim, nd = args.dim, mesh.get_num_devices()
    res = {}
    g = torch.Generator().manual_seed(0)
    # per-device partials [nd, 1,1,64,dim] with tile-row 1 == tile-row 0 (and rows 8..15 == 40..47 by construction)
    parts = torch.randn(nd, 1, 1, 64, dim, generator=g) * 3
    parts[:, :, :, 32:] = parts[:, :, :, :32]
    parts = parts.to(torch.bfloat16)
    ref_fp32 = parts.float().sum(0).to(torch.bfloat16)  # exact sum rounded once

    def shard(t, dtype):
        return ttnn.from_torch(
            t.reshape(nd, 1, -1, dim),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )

    for trial in range(2):
        x = ttnn.reshape(shard(parts, ttnn.bfloat16), (1, 1, 64, dim))
        out = tt_all_reduce(
            x, mesh, ccl, cluster_axis=0, dim=3, topology=args.ccl_topology(), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        o = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).reshape(64, dim)
        res[f"rs_bf16_tilerows_equal_t{trial}"] = torch.equal(o[:32], o[32:])
        res[f"rs_bf16_vs_exactsum_rows0_31_t{trial}"] = int((o[:32] != ref_fp32.reshape(64, dim)[:32]).sum())
        res[f"rs_bf16_vs_exactsum_rows32_63_t{trial}"] = int((o[32:] != ref_fp32.reshape(64, dim)[32:]).sum())
    # fp32 reduce-scatter of the same partials
    x32 = ttnn.reshape(shard(parts.float(), ttnn.float32), (1, 1, 64, dim))
    out32 = tt_all_reduce(
        x32,
        mesh,
        ccl,
        cluster_axis=0,
        dim=3,
        topology=args.ccl_topology(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.float32,
    )
    o32 = ttnn.to_torch(out32, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).reshape(64, dim)
    res["rs_fp32_out_dtype"] = str(out32.dtype)
    o32b = o32.to(torch.bfloat16)
    res["rs_fp32_tilerows_equal"] = torch.equal(o32b[:32], o32b[32:])
    res["rs_fp32_vs_exactsum_mismatch"] = int((o32b != ref_fp32.reshape(64, dim)).sum())
    # the fused decode all_reduce_async on the first 32 rows (bf16 partials): equals the exact sum?
    ar = ccl.decode_all_reduce
    if ar is not None:
        ar.begin_step()
        x1 = ttnn.reshape(shard(parts[:, :, :, :32].contiguous(), ttnn.bfloat16), (1, 1, 32, dim))
        x1 = ttnn.to_memory_config(x1, ttnn.L1_MEMORY_CONFIG)
        y = ar(x1)
        yy = _t0(ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)).reshape(32, dim)
        res["fusedAR_vs_exactsum_mismatch"] = int((yy != ref_fp32.reshape(64, dim)[:32]).sum())
        res["fusedAR_vs_rs_fp32_rows0_31_mismatch"] = int((yy != o32b[:32]).sum())
        res["fusedAR_vs_rs_bf16_rows0_31_mismatch"] = int((yy != o[:32]).sum())
    # fp32 reduce-scatter at the verify row counts: exactness vs the fp32 sum of the bf16 partials
    for R in (64, 128, 256):
        parts = (torch.randn(nd, 1, 1, R, dim, generator=g) * 3).to(torch.bfloat16)
        ref = parts.float().sum(0)
        x32 = ttnn.reshape(shard(parts.float(), ttnn.float32), (1, 1, R, dim))
        out32 = tt_all_reduce(
            x32,
            mesh,
            ccl,
            cluster_axis=0,
            dim=3,
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.float32,
        )
        o32 = ttnn.to_torch(out32, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).reshape(R, dim)
        d = (o32 - ref.reshape(R, dim)).abs()
        res[f"rs_fp32_R{R}_maxabs_vs_exact"] = float(d.max())
        res[f"rs_fp32_R{R}_rows_with_err"] = int((d.max(-1).values > 1e-3).sum())
        xb = ttnn.reshape(shard(parts, ttnn.bfloat16), (1, 1, R, dim))
        ob = tt_all_reduce(
            xb, mesh, ccl, cluster_axis=0, dim=3, topology=args.ccl_topology(), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        obt = ttnn.to_torch(ob, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float().reshape(R, dim)
        db = (obt - ref.reshape(R, dim)).abs()
        res[f"rs_bf16_R{R}_maxabs_vs_exact"] = float(db.max())
        res[f"rs_bf16_R{R}_rows_with_err"] = int((db.max(-1).values > 0.5).sum())
    for k, v in res.items():
        logger.info(f"PROBE {k}: {v}")
    print("PROBE_RESULTS " + " | ".join(f"{k}={v}" for k, v in res.items()))
