# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH round 2 for the fused decode all-reduce: residual-add variants on the 32-core width shard, out-proj matmul
grid variants, bfp8 partials, and the whole old (mm->RS->add->AG->LN->S2I) vs new (mm->AR->add->LN->S2I) sub-layer
chains, trace-timed on the served (1,4) FABRIC_1D half."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.tt_transformers.tt.ccl import TT_CCL

DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 64 * 1024 * 1024,
    }
]
K = int(os.environ.get("CCL_BENCH_K", "32"))
H, TP, KIN = 5120, 4, 1536


def _trace_time(mesh, fn, k=K, reps=7):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [fn() for _ in range(k)]
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = min(best, time.perf_counter() - t0)
    ttnn.release_trace(mesh, tid)
    for o in outs:
        try:
            ttnn.deallocate(o)
        except Exception:
            pass
    return best / k * 1e6


def _d0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _t(mesh, name, fn):
    try:
        us = _trace_time(mesh, fn)
        print(f"AR2 {name} us={us:.1f}", flush=True)
        return us
    except Exception as e:
        print(f"AR2 {name} FAILED {str(e).splitlines()[0][:200]}", flush=True)
        ttnn.synchronize_device(mesh)
        return None


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_fused_all_reduce_scratch2(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    tt_ccl = TT_CCL(mesh)
    links = tt_ccl.get_num_links(1)
    act_cfg = tpc.create_activation_shard_config(H)
    grid = act_cfg.shard_spec.grid
    buf_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, (H // 32) * TP], ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(mesh)

    def frm(t, dtype=ttnn.bfloat16, mc=ttnn.L1_MEMORY_CONFIG, mapper=rep):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mc, mesh_mapper=mapper
        )

    x_sh = frm(torch.randn(1, 1, 32, H, dtype=torch.bfloat16), mc=act_cfg)
    y_sh = frm(torch.randn(1, 1, 32, H, dtype=torch.bfloat16), mc=act_cfg)
    x_il = frm(torch.randn(1, 1, 32, H, dtype=torch.bfloat16))
    y_il = frm(torch.randn(1, 1, 32, H, dtype=torch.bfloat16))
    x_fr = frm(torch.randn(1, 1, 32, H // TP, dtype=torch.bfloat16), mc=ttnn.DRAM_MEMORY_CONFIG)
    y_fr = frm(torch.randn(1, 1, 32, H // TP, dtype=torch.bfloat16), mc=ttnn.DRAM_MEMORY_CONFIG)

    # ---- residual add variants ----
    _t(mesh, "add sharded default", lambda: ttnn.add(x_sh, y_sh))
    _t(mesh, "add sharded memcfg=act", lambda: ttnn.add(x_sh, y_sh, memory_config=act_cfg))
    _t(mesh, "add L1-interleaved [32,5120]", lambda: ttnn.add(x_il, y_il))
    _t(mesh, "add DRAM-interleaved [32,1280] (today)", lambda: ttnn.add(x_fr, y_fr))
    # ---- matmul grids ----
    a_tt = frm(torch.randn(1, 1, 32, KIN, dtype=torch.bfloat16))
    w = torch.randn(KIN, H, dtype=torch.bfloat16) * 0.02
    w_tt = frm(w, dtype=ttnn.bfloat8_b, mc=ttnn.DRAM_MEMORY_CONFIG)
    gw = mesh.compute_with_storage_grid_size().x

    def mm(pc, mc, dtype=None):
        return ttnn.linear(
            a_tt, w_tt, compute_kernel_config=tpc.COMPUTE_HIFI2, program_config=pc, memory_config=mc, dtype=dtype
        )

    pc33 = tpc.create_matmul_1d_decode_progcfg(32, KIN, H, num_cores=33, grid_w=gw)
    pc32w = tpc.create_matmul_1d_decode_progcfg(32, KIN, H, num_cores=32, grid_w=8)  # 8x4
    pc32t = tpc.create_matmul_1d_decode_progcfg(32, KIN, H, num_cores=32, grid_w=4)  # 4x8
    for _ in range(2):
        _t(mesh, "mm 11x3 DRAM-out", lambda: mm(pc33, ttnn.DRAM_MEMORY_CONFIG))
        _t(mesh, "mm 11x3 L1-out", lambda: mm(pc33, ttnn.L1_MEMORY_CONFIG))
        _t(mesh, "mm 8x4 L1-out", lambda: mm(pc32w, ttnn.L1_MEMORY_CONFIG))
        _t(mesh, "mm 8x4 sharded-out", lambda: mm(pc32w, act_cfg))
        _t(mesh, "mm 4x8 sharded-out", lambda: mm(pc32t, act_cfg))
        _t(mesh, "mm 8x4 sharded-out bfp8", lambda: mm(pc32w, act_cfg, ttnn.bfloat8_b))
    _t(mesh, "mm 11x3 L1-out + I2S", lambda: ttnn.to_memory_config(mm(pc33, ttnn.L1_MEMORY_CONFIG), act_cfg))

    # ---- all_reduce variants ----
    bufs = [frm(torch.zeros(1, 1, 32, H * TP, dtype=torch.bfloat16), mc=buf_cfg) for _ in range(2)]
    sems = [ttnn.create_global_semaphore(mesh, tt_ccl.sub_device_crs, 0) for _ in range(2)]
    call = {"i": 0}

    def ar(x, topo=ttnn.Topology.Ring, nl=links, noc1=False, dtype=None, b=bufs):
        i = call["i"] % 2
        call["i"] += 1
        return ttnn.experimental.all_reduce_async(
            x,
            b[i],
            cluster_axis=1,
            mesh_device=mesh,
            multi_device_global_semaphore=sems[i],
            memory_config=act_cfg,
            topology=topo,
            num_links=nl,
            fp32_dest_acc=True,
            use_noc1_only=noc1,
            dtype=dtype,
        )

    # ---- chains ----
    wn = frm(torch.ones(1, 1, 1, H, dtype=torch.bfloat16), mc=ttnn.DRAM_MEMORY_CONFIG)
    ln_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4], subblock_w=1, block_h=1, block_w=5, inplace=False
    )

    def old_chain():
        p = mm(pc33, ttnn.DRAM_MEMORY_CONFIG)
        r = ttnn.experimental.reduce_scatter_minimal_async(
            p,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        ttnn.deallocate(p)
        h = ttnn.add(x_fr, r)
        ttnn.deallocate(r)
        g = ttnn.experimental.all_gather_async(
            h,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=links,
            topology=ttnn.Topology.Ring,
            memory_config=act_cfg,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        n = ttnn.rms_norm(g, epsilon=1e-6, weight=wn, program_config=ln_pc, memory_config=act_cfg)
        ttnn.deallocate(g)
        o = ttnn.to_memory_config(n, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(n)
        ttnn.deallocate(h)
        return o

    def new_chain(i2s=False, add_fn=ttnn.add):
        if i2s:
            p0 = mm(pc33, ttnn.L1_MEMORY_CONFIG)
            p = ttnn.to_memory_config(p0, act_cfg)
            ttnn.deallocate(p0)
        else:
            p = mm(pc32w, act_cfg)
        r = ar(p)
        ttnn.deallocate(p)
        h = add_fn(x_sh, r)
        ttnn.deallocate(r)
        n = ttnn.rms_norm(h, epsilon=1e-6, weight=wn, program_config=ln_pc, memory_config=act_cfg)
        o = ttnn.to_memory_config(n, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(n)
        ttnn.deallocate(h)
        return o

    for _ in range(3):
        _t(mesh, "CHAIN old mm11x3->RS->add->AG->LN->S2I", old_chain)
        _t(mesh, "CHAIN new mm8x4sh->AR->add->LN->S2I", new_chain)
        _t(mesh, "CHAIN new mm11x3->I2S->AR->add->LN->S2I", lambda: new_chain(i2s=True))
    _t(mesh, "AR ring 2L", lambda: ar(x_sh))
    _t(mesh, "AR ring 2L noc1", lambda: ar(x_sh, noc1=True))
    _t(mesh, "AR linear 2L", lambda: ar(x_sh, topo=ttnn.Topology.Linear))
    print("AR2 DONE", flush=True)
