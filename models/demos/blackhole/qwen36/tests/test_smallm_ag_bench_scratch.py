# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH microbench (item J): tune the plain all_gather_async that replaces the fused AGMM gather at small M.

Gathers the K-sharded prefill activation [1,1,M,K/4] -> [1,1,M,K] bf16 on the (1,4) ring for M in {128, 256} and
K in {5120 (norm output), 6144 (GDN gated activation)}, TRACED, over all_gather_async knobs: num_links, workers per
link, chunks_per_sync, buffers per channel, output placement (L1/DRAM) and the in-kernel barrier.

  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_smallm_ag_bench_scratch.py -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL

DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 64 * 1024 * 1024,
    }
]
MS = [int(m) for m in os.environ.get("SMALLM_MS", "128,256").split(",")]
KS = [int(k) for k in os.environ.get("SMALLM_KS", "5120,6144").split(",")]
REPS = int(os.environ.get("SMALLM_REPS", "10"))
# name -> kwargs
VARIANTS = {
    "l2_bar": dict(num_links=2, barrier=True),
    "l1_bar": dict(num_links=1, barrier=True),
    "l2_nobar": dict(num_links=2, barrier=False),
    "l2_w2_c10_b2_bar": dict(
        num_links=2, barrier=True, num_workers_per_link=2, chunks_per_sync=10, num_buffers_per_channel=2
    ),
    "l2_w4_c10_b2_bar": dict(
        num_links=2, barrier=True, num_workers_per_link=4, chunks_per_sync=10, num_buffers_per_channel=2
    ),
    "l2_w2_c10_b8_bar": dict(
        num_links=2, barrier=True, num_workers_per_link=2, chunks_per_sync=10, num_buffers_per_channel=8
    ),
    "l2_w4_c20_b8_bar": dict(
        num_links=2, barrier=True, num_workers_per_link=4, chunks_per_sync=20, num_buffers_per_channel=8
    ),
    "l2_w1_c10_b2_bar": dict(
        num_links=2, barrier=True, num_workers_per_link=1, chunks_per_sync=10, num_buffers_per_channel=2
    ),
    "l1_w4_c10_b8_bar": dict(
        num_links=1, barrier=True, num_workers_per_link=4, chunks_per_sync=10, num_buffers_per_channel=8
    ),
    "l2_w2_c10_b2_bar_dram": dict(
        num_links=2, barrier=True, num_workers_per_link=2, chunks_per_sync=10, num_buffers_per_channel=2, dram=True
    ),
    "l2_bar_dram": dict(num_links=2, barrier=True, dram=True),
    "l2_w2_c10_b2_nobar": dict(
        num_links=2, barrier=False, num_workers_per_link=2, chunks_per_sync=10, num_buffers_per_channel=2
    ),
}


def _traced_us(mesh, fn, reps=REPS, replays=5):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    best = None
    for _ in range(replays):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        dt = (time.perf_counter() - t0) / reps * 1e6
        best = dt if best is None else min(best, dt)
    ttnn.release_trace(mesh, tid)
    return best


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_smallm_ag_bench(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    tt_ccl = TT_CCL(mesh)
    topo = ttnn.Topology.Ring
    torch.manual_seed(0)
    for K in KS:
        for M in MS:
            x_t = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
            x = ttnn.from_torch(
                x_t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            )
            for vname, kw in VARIANTS.items():
                kw = dict(kw)
                bar = kw.pop("barrier")
                dram = kw.pop("dram", False)

                def fn(kw=kw, bar=bar, dram=dram):
                    extra = {"barrier_semaphore": tt_ccl.get_and_cycle_barrier_semaphore_handle(1)} if bar else {}
                    out = ttnn.experimental.all_gather_async(
                        x,
                        dim=3,
                        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
                        topology=topo,
                        cluster_axis=1,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG if dram else ttnn.L1_MEMORY_CONFIG,
                        **extra,
                        **kw,
                    )
                    return out

                try:
                    out = fn()
                    ttnn.synchronize_device(mesh)
                    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
                    ok = torch.equal(got, x_t.float())
                    ttnn.deallocate(out)

                    def body():
                        ttnn.deallocate(fn())

                    us = _traced_us(mesh, body)
                    logger.info(f"SMALLM_AG K={K} M={M:4d} {vname:24s} {us:7.1f} us exact={ok}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"SMALLM_AG K={K} M={M:4d} {vname:24s} ERR {str(e).splitlines()[0][:120]}")
            ttnn.deallocate(x)
    print("SMALLM_AG_DONE")
