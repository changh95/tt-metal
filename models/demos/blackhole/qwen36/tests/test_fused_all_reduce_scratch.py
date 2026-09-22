# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: fused ttnn.experimental.all_reduce_async (buffer_tensor overload) for the qwen36 decode sub-layer
all-reduce on the served (1,4) FABRIC_1D half. Checks legality of the 32-core [1,1,32,5120] bf16 width-sharded
output, torch-sum correctness, bit-identity of the 4 replicas, trace-replay stability (alternating 2 buffer/semaphore
pairs) and per-op device time vs today's reduce_scatter + all_gather chain; also the 1D out-proj matmul with a
width-sharded output (must be bit-identical to the interleaved output) and the embedding all-gather into the shard.

  TT_VISIBLE_DEVICES=2,3,4,5 MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_fused_all_reduce_scratch.py -s
"""
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


def _trace_time(mesh, fn, k=K, reps=5):
    fn()  # compile / warm-up
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


def _dev_tensors(t):
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_fused_all_reduce_scratch(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    tt_ccl = TT_CCL(mesh)
    max_links = tt_ccl.get_num_links(1)
    print(f"AR_SCRATCH max_links(axis1)={max_links} mesh={tuple(mesh.shape)}", flush=True)

    act_cfg = tpc.create_activation_shard_config(H)  # 32 cores (8x4), shard [32,160]
    grid = act_cfg.shard_spec.grid
    print(f"AR_SCRATCH act_cfg grid={grid} shard={act_cfg.shard_spec.shape}", flush=True)
    buf_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, (H // 32) * TP], ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch.manual_seed(0)
    # per-device partials [4, 1, 32, H] -> shard dim 0 across the 4 devices
    partials = torch.randn(TP, 1, 32, H, dtype=torch.bfloat16)
    golden = partials.float().sum(0, keepdim=True)
    x_in = ttnn.from_torch(
        partials,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=act_cfg,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    bufs = [
        ttnn.from_torch(
            torch.zeros(1, 1, 32, H * TP, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=buf_cfg,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for _ in range(2)
    ]
    sems = [ttnn.create_global_semaphore(mesh, tt_ccl.sub_device_crs, 0) for _ in range(2)]
    call = {"i": 0}

    def ar(topo, links, fp32=True, x=x_in):
        i = call["i"] % 2
        call["i"] += 1
        return ttnn.experimental.all_reduce_async(
            x,
            bufs[i],
            cluster_axis=1,
            mesh_device=mesh,
            multi_device_global_semaphore=sems[i],
            memory_config=act_cfg,
            topology=topo,
            num_links=links,
            fp32_dest_acc=fp32,
        )

    topos = {"ring": ttnn.Topology.Ring, "linear": ttnn.Topology.Linear}
    ok_cfgs = []
    for tname, topo in topos.items():
        for links in sorted({1, max_links}):
            t0 = time.perf_counter()
            try:
                out = ar(topo, links)
                ttnn.synchronize_device(mesh)
                devs = _dev_tensors(out)
                pcc_ok = all(torch.allclose(d, golden[0], atol=0.05, rtol=0.02) for d in devs)
                maxd = max((d - golden[0]).abs().max().item() for d in devs)
                same = all(torch.equal(d, devs[0]) for d in devs[1:])
                print(
                    f"AR_SCRATCH eager topo={tname} links={links} ok={pcc_ok} max|d|={maxd:.4f} replicas_identical={same} "
                    f"({time.perf_counter() - t0:.1f}s)",
                    flush=True,
                )
                ttnn.deallocate(out)
                if pcc_ok and same:
                    ok_cfgs.append((tname, topo, links))
            except Exception as e:
                print(f"AR_SCRATCH eager topo={tname} links={links} FAILED {str(e).splitlines()[0][:200]}", flush=True)
                ttnn.synchronize_device(mesh)
    assert ok_cfgs, "no all_reduce_async config works"

    # trace timing (K calls per replay, alternating pairs) + replay stability check
    for tname, topo, links in ok_cfgs:
        us = _trace_time(mesh, lambda: ar(topo, links))
        print(f"AR_SCRATCH traced all_reduce_async topo={tname} links={links} us={us:.1f}", flush=True)
        # after the traced replays the op must still produce the right sum (semaphore reset / buffer reuse ok)
        out = ar(topo, links)
        devs = _dev_tensors(out)
        assert all(torch.allclose(d, golden[0], atol=0.05, rtol=0.02) for d in devs)
        assert all(torch.equal(d, devs[0]) for d in devs[1:])
        ttnn.deallocate(out)
    # fp32_dest_acc off, for reference
    tname, topo, links = ok_cfgs[0]
    us = _trace_time(mesh, lambda: ar(topo, links, fp32=False))
    print(f"AR_SCRATCH traced all_reduce_async topo={tname} links={links} fp32_acc=False us={us:.1f}", flush=True)

    # --- baseline: today's RS (DRAM out) + AG into the shard ---
    x_il = ttnn.to_memory_config(x_in, ttnn.L1_MEMORY_CONFIG)

    def rs_ag(topo, links):
        r = ttnn.experimental.reduce_scatter_minimal_async(
            x_il,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topo,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        g = ttnn.experimental.all_gather_async(
            r,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=links,
            topology=topo,
            memory_config=act_cfg,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        ttnn.deallocate(r)
        return g

    for tname, topo in topos.items():
        for links in sorted({1, max_links}):
            try:
                us = _trace_time(mesh, lambda: rs_ag(topo, links))
                print(f"AR_SCRATCH traced RS+AG topo={tname} links={links} us={us:.1f}", flush=True)
            except Exception as e:
                print(f"AR_SCRATCH RS+AG topo={tname} links={links} FAILED {str(e).splitlines()[0][:120]}", flush=True)
                ttnn.synchronize_device(mesh)

    # --- 1D out-proj matmul with width-sharded output vs interleaved output (bit-identity + time) ---
    a = torch.randn(1, 1, 32, KIN, dtype=torch.bfloat16)
    w = torch.randn(KIN, H, dtype=torch.bfloat16) * 0.02
    a_tt = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    w_tt = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    gw = mesh.compute_with_storage_grid_size().x
    pc33 = tpc.create_matmul_1d_decode_progcfg(32, KIN, H, num_cores=33, grid_w=gw)
    pc32 = tpc.create_matmul_1d_decode_progcfg(32, KIN, H, num_cores=32, grid_w=8)
    print(
        f"AR_SCRATCH pc33 grid={pc33.compute_with_storage_grid_size} per_core_N={pc33.per_core_N}; pc32 grid={pc32.compute_with_storage_grid_size} per_core_N={pc32.per_core_N}",
        flush=True,
    )

    def mm(pc, mc):
        return ttnn.linear(a_tt, w_tt, compute_kernel_config=tpc.COMPUTE_HIFI2, program_config=pc, memory_config=mc)

    ref = _dev_tensors(mm(pc33, ttnn.DRAM_MEMORY_CONFIG))[0]
    try:
        o32 = mm(pc32, act_cfg)
        got = _dev_tensors(o32)[0]
        print(
            f"AR_SCRATCH matmul 8x4 sharded-out bit_identical_to_33core_dram={torch.equal(got, ref)} max|d|={(got - ref).abs().max().item():.4g}",
            flush=True,
        )
        ttnn.deallocate(o32)
        us = _trace_time(mesh, lambda: mm(pc32, act_cfg))
        print(f"AR_SCRATCH traced matmul 8x4 sharded-out us={us:.1f}", flush=True)
        us = _trace_time(mesh, lambda: mm(pc32, ttnn.L1_MEMORY_CONFIG))
        print(f"AR_SCRATCH traced matmul 8x4 L1-interleaved us={us:.1f}", flush=True)
    except Exception as e:
        print(f"AR_SCRATCH matmul sharded-out FAILED {str(e).splitlines()[0][:200]}", flush=True)
        ttnn.synchronize_device(mesh)
    us = _trace_time(mesh, lambda: mm(pc33, ttnn.DRAM_MEMORY_CONFIG))
    print(f"AR_SCRATCH traced matmul 11x3 DRAM-out us={us:.1f}", flush=True)

    # --- fused chain: matmul(sharded) -> all_reduce ; vs matmul(dram) -> RS -> AG ---
    tname, topo, links = ok_cfgs[0]
    try:
        us = _trace_time(mesh, lambda: ar(topo, links, x=mm(pc32, act_cfg)))
        print(f"AR_SCRATCH traced CHAIN matmul+all_reduce topo={tname} links={links} us={us:.1f}", flush=True)
    except Exception as e:
        print(f"AR_SCRATCH CHAIN FAILED {str(e).splitlines()[0][:200]}", flush=True)
        ttnn.synchronize_device(mesh)

    # --- residual add + sharded rms_norm on the replicated [32,5120] shard; AG of the embedding [32,1280] ---
    wn = ttnn.from_torch(
        torch.ones(1, 1, 1, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    y_in = ttnn.from_torch(
        partials[:1],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=act_cfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    us = _trace_time(mesh, lambda: ttnn.add(x_in, y_in))
    print(f"AR_SCRATCH traced sharded add [32,5120] us={us:.1f}", flush=True)
    ln_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4], subblock_w=1, block_h=1, block_w=5, inplace=False
    )
    us = _trace_time(
        mesh,
        lambda: ttnn.rms_norm(x_in, epsilon=1e-6, weight=wn, program_config=ln_pc, memory_config=act_cfg),
    )
    print(f"AR_SCRATCH traced sharded rms_norm [32,5120] 8x4 us={us:.1f}", flush=True)
    emb = ttnn.from_torch(
        torch.randn(1, 1, 32, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
    )

    def emb_ag():
        return ttnn.experimental.all_gather_async(
            emb,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=max_links,
            topology=ttnn.Topology.Ring,
            memory_config=act_cfg,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    us = _trace_time(mesh, emb_ag)
    print(f"AR_SCRATCH traced embedding AG dram[32,1280]->shard[32,5120] us={us:.1f}", flush=True)
    print("AR_SCRATCH DONE", flush=True)
