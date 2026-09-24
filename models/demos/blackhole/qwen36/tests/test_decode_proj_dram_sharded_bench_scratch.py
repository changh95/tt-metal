# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn-level microbench: qwen36 27B TP=4 decode projection matmuls (M = 1 tile) on the current 1D mcast_in0 config
vs MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig with num_workers_per_dram_bank = 1 / 2 / 3.

Per matrix (per-device shapes) prints one line per config:
    PROJ_BENCH <mat> <cfg> us=<traced us/op> pcc_torch=<..> maxdiff_vs_1d=<..>

Env: BENCH_MATS (comma list of gate,up,down,qkvzab,out,attn_qkv; default all), BENCH_K (trace iterations, 20),
BENCH_WORKERS ("1,2,3").

RESULT 2026-09-21 (P150x4 TP=4 mesh, compute grid 11x10, logs/itemA_bench1.log): every workers>1 config fails at
program creation with "TT_FATAL tt_metal/impl/device/experimental/device.cpp:20: mesh->num_devices() == 1" -- the
multi-reader core assignment (matmul_utilities.cpp get_dram_bank_reader_assignments) calls
experimental::Device::get_worker_noc_hop_distance(IDevice*), which only accepts a unit MeshDevice, so on a
multi-device mesh only num_workers_per_dram_bank=1 is reachable (a tt-metal fix: delegate to the mesh's first local
device, as the MeshCoordinate overload already does). 1-worker traced us/op vs the tuned 1D path:
gate 71.4/50.9, up 54.4/46.0, down 54.6/66.2 (in0 17 cores x 8 tiles; 8 cores x 1-tile blocks: 144), qkvzab
86.8/62.8, out 34.8/27.5, attn_qkv 73.8/53.4 -> only the down projection wins; maxdiff vs 1D <= 0.016 (bf16 lsb),
pcc_vs_1d >= 0.99999.

RESULT 2026-09-24 (logs/itemA_bench2_w123.log, after the device.cpp fix that measures the hop distance on the
mesh's first device): workers 2/3 run on the (1,4) mesh. Best traced us/op per matrix (1D = tuned default):
gate 1D 50.7 | w1 71.1 | w2 50.3 | w3 43.5 (in0c32b5 pcn4); up 45.7 | 54.0 | 39.5 | 35.7 (in0c32b5 pcn4);
down 66.1 | 54.2 | 82.4 | 60.1; qkvzab 62.6 | 86.6 | 65.2 | 60.1; out 27.2 | 34.7 | 32.6 | 27.5;
attn_qkv 53.4 | 73.4 | 51.9 | 50.2. per_core_N values that give fewer output cores than 8*workers readers fail
loudly ("Worker x-y has no storage area assigned"). Enabled in the model for gate/up (QWEN36_DECODE_DRAM_SHARDED=gateup).

Run on half A:
  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 ARCH_NAME=blackhole python_env/bin/python -m pytest \
  models/demos/blackhole/qwen36/tests/test_decode_proj_dram_sharded_bench_scratch.py -x -s
"""
import math
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt import tp_common as tpc

K_ITERS = int(os.environ.get("BENCH_K", "20"))
MATS = [m for m in os.environ.get("BENCH_MATS", "gate,up,down,qkvzab,out,attn_qkv").split(",") if m]
WORKERS = [int(w) for w in os.environ.get("BENCH_WORKERS", "1,2,3").split(",") if w]

LOFI_L1ACC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
)

# name: (K, N, weight dtype, compute cfg, fused act, 1D num_cores, 1D grid_w, in0 grids [(cores, in0_block_w)], per_core_N list)
# 1D configs mirror model_config.py (gate/up 44 @ grid_w 11, down/out 33 @ 11, attn_qkv 64 @ 8).
MATRICES = {
    "gate": (
        5120,
        4352,
        ttnn.bfloat4_b,
        LOFI_L1ACC,
        ttnn.UnaryOpType.SILU,
        44,
        11,
        [(32, 5), (40, 4), (16, 10)],
        [8, 4, 17],
    ),
    "up": (5120, 4352, ttnn.bfloat4_b, LOFI_L1ACC, None, 44, 11, [(32, 5), (40, 4)], [8, 4]),
    "down": (4352, 5120, ttnn.bfloat8_b, LOFI_L1ACC, None, 33, 11, [(17, 8), (34, 4), (8, 1)], [5, 10]),
    "qkvzab": (5120, 4128, ttnn.bfloat8_b, tpc.COMPUTE_HIFI2, None, 44, 11, [(32, 5), (40, 4)], [3, 8]),
    "out": (1536, 5120, ttnn.bfloat8_b, tpc.COMPUTE_HIFI2, None, 33, 11, [(24, 2), (16, 3), (12, 4)], [5, 10]),
    "attn_qkv": (5120, 3584, ttnn.bfloat8_b, tpc.COMPUTE_HIFI2, None, 64, 8, [(32, 5), (40, 4)], [4, 7, 8]),
}


def _trace_time(mesh, fn, k=K_ITERS, reps=5):
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


def _pcc(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def _dev0(mesh, t):
    if t.memory_config().is_sharded():
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _in0_cfg(mesh, K, cores):
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.num_cores_to_corerangeset(cores, grid, True)
    return ttnn.create_sharded_memory_config(
        shape=(tpc.TILE_SIZE, K // cores),
        core_grid=crs,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_decode_proj_bench(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    grid = mesh.compute_with_storage_grid_size()
    print(f"PROJ_BENCH grid={grid.x}x{grid.y} dram_banks={mesh.dram_grid_size().x} K_ITERS={K_ITERS}", flush=True)
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(mesh)

    for name in MATS:
        K, N, wdt, ckc, act, cores1d, gw, in0_grids, pcns = MATRICES[name]
        x = (torch.randn(1, 1, 32, K) * 0.5).to(torch.bfloat16)
        w = (torch.randn(K, N) / math.sqrt(K)).to(torch.bfloat16)
        ref = x.float() @ w.float()
        if act is not None:
            ref = torch.nn.functional.silu(ref)
        x_il = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=rep,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        w_il = ttnn.as_tensor(
            w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        pc1d = tpc.create_matmul_1d_decode_progcfg(1, K, N, num_cores=cores1d, fused_activation=act, grid_w=gw)
        out1d = ttnn.linear(
            x_il, w_il, compute_kernel_config=ckc, program_config=pc1d, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        o1d = _dev0(mesh, out1d)[..., :N]
        ttnn.deallocate(out1d)
        us = _trace_time(
            mesh,
            lambda: ttnn.linear(
                x_il, w_il, compute_kernel_config=ckc, program_config=pc1d, memory_config=ttnn.L1_MEMORY_CONFIG
            ),
        )
        print(f"PROJ_BENCH {name} 1d_c{cores1d} us={us:.1f} pcc_torch={_pcc(o1d, ref):.6f} maxdiff_vs_1d=0", flush=True)
        ttnn.deallocate(w_il)

        for workers in WORKERS:
            mc = (
                tpc.create_dram_sharded_mem_config(K, N, workers)
                if "workers" in tpc.create_dram_sharded_mem_config.__code__.co_varnames
                else None
            )
            if mc is None:
                # pre-edit tp_common: build the padded memcfg inline (N padded to 8*workers*32 * ceil)
                n_tiles = math.ceil(N / 32)
                per_reader = math.ceil(n_tiles / (8 * workers))
                shard_w = per_reader * workers * 32
                mc = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(tpc.DRAM_GRID, (K, shard_w), ttnn.ShardOrientation.ROW_MAJOR),
                )
            try:
                w_ds = ttnn.as_tensor(
                    w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep, memory_config=mc
                )
            except Exception as e:
                print(f"PROJ_BENCH {name} ds_w{workers} WEIGHT_FAILED {str(e).splitlines()[0][:160]}", flush=True)
                continue
            for cores, ibw in in0_grids:
                try:
                    x_sh = ttnn.to_memory_config(x_il, _in0_cfg(mesh, K, cores))
                except Exception as e:
                    print(
                        f"PROJ_BENCH {name} ds_w{workers}_in0c{cores} IN0_FAILED {str(e).splitlines()[0][:160]}",
                        flush=True,
                    )
                    continue
                for pcn in pcns:
                    tag = f"ds_w{workers}_in0c{cores}b{ibw}_pcn{pcn}"
                    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=ibw,
                        per_core_M=1,
                        per_core_N=pcn,
                        fused_activation=act,
                        num_workers_per_dram_bank=workers,
                    )

                    def run():
                        return ttnn.linear(
                            x_sh,
                            w_ds,
                            compute_kernel_config=ckc,
                            program_config=pc,
                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        )

                    try:
                        o = run()
                        ods = _dev0(mesh, o)[..., :N]
                        ttnn.deallocate(o)
                        us = _trace_time(mesh, run)
                        print(
                            f"PROJ_BENCH {name} {tag} us={us:.1f} pcc_torch={_pcc(ods, ref):.6f} "
                            f"maxdiff_vs_1d={(ods - o1d).abs().max().item():.5f} pcc_vs_1d={_pcc(ods, o1d):.7f}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"PROJ_BENCH {name} {tag} FAILED {str(e).splitlines()[0][:200]}", flush=True)
                        ttnn.synchronize_device(mesh)
                ttnn.deallocate(x_sh)
            ttnn.deallocate(w_ds)
        ttnn.deallocate(x_il)
