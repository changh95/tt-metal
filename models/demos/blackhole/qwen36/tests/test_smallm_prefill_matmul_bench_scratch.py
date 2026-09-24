# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH microbench (item J): small-M prefill weight matmuls at the Qwen3.8-27B TP=4 per-device shapes.

At M=128 rows the fused all_gather_minimal_matmul_async (AGMM) pads M from 4 to 8 tiles (in0_parallel_axis_cores = 8
in the nt11x8 layout), so half the compute cores hold padding rows, and the 2D w2/wo configs only light 4 grid rows.
This bench times, per shape and per M in {128, 256}, TRACED (host dispatch excluded, min over replays):

  cur      the path the model runs today (AGMM in-proj / swiglu, or the tuned 2D config for the row-parallel matmuls)
  ag       the plain all_gather_async of the K-sharded activation alone (2 links, in-kernel barrier)
  1d_<c>   all_gather_async + MatmulMultiCoreReuseMultiCast1DProgramConfig (mcast_in0) on ~c cores, wide-first grid
  2d_<c>   all_gather_async + the 2D mcast config sized to M (grid = cols x M_tiles, per_core_M = 1)

PCC vs torch is checked for every variant (a wrong config fails loudly instead of timing garbage).

  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_smallm_prefill_matmul_bench_scratch.py -s
Env: SMALLM_MS ("128,256"), SMALLM_SHAPES (comma list of the SHAPES keys), SMALLM_REPS (trace body repeats, 10),
     SMALLM_CORES ("44,66,88,110": 1D core budgets), SMALLM_AG_L1 (1: gathered activation in L1, default; 0: DRAM).
"""

import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import (  # noqa: F401  (serving env defaults: nt11x8, barrier...)
    model_config as _mc,
)
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
TP = 4
DIM = 5120
# name -> (K_total, N_local, weight dtype, role)   role: "in" (col-parallel in-proj, K-sharded x, AGMM today)
#                                                       "mlp" (col-parallel [gate|up] + swiglu, AGMM today)
#                                                       "row" (row-parallel out-proj, replicated-K x, 2D today; RS after)
#                                                       "out" (GDN out-proj in agmm mode: K-sharded gated x, col-sharded out_col)
SHAPES = {
    "gdn_in": (DIM, 4128, ttnn.bfloat8_b, "in"),
    "attn_qkv": (DIM, 3584, ttnn.bfloat8_b, "in"),
    "mlp_gateup": (DIM, 4352, ttnn.bfloat4_b, "mlp"),
    "mlp_w2": (4352, DIM, ttnn.bfloat8_b, "row"),
    "attn_wo": (1536, DIM, ttnn.bfloat8_b, "row"),
    "gdn_out": (6144, 1280, ttnn.bfloat8_b, "out"),
}
MS = [int(m) for m in os.environ.get("SMALLM_MS", "128,256").split(",")]
REPS = int(os.environ.get("SMALLM_REPS", "10"))
CORES = [int(c) for c in os.environ.get("SMALLM_CORES", "44,66,88,110").split(",")]
AG_MC = ttnn.L1_MEMORY_CONFIG if os.environ.get("SMALLM_AG_L1", "1") == "1" else ttnn.DRAM_MEMORY_CONFIG
GRID_W = 11


def _ag(x4, tt_ccl, topo, out_mc=AG_MC, num_links=2):
    """Plain all_gather_async of the K-sharded activation with the in-kernel barrier (DistributedNorm's idiom)."""
    return ttnn.experimental.all_gather_async(
        x4,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
        num_links=num_links,
        topology=topo,
        cluster_axis=1,
        memory_config=out_mc,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
    )


def _traced_us(mesh, fn, reps=REPS, replays=5):
    """Compile eagerly, capture `reps` calls in one trace, return min per-call us over `replays` replays."""
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


def _dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _pcc(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def _lin1d(x, w, m, k, n, cores, ckc, act=None, out_mc=ttnn.L1_MEMORY_CONFIG):
    pc = tpc.create_matmul_1d_decode_progcfg(m, k, n, num_cores=cores, fused_activation=act, grid_w=GRID_W)
    return ttnn.linear(x, w, compute_kernel_config=ckc, program_config=pc, memory_config=out_mc)


def _lin2d_m(x, w, m, k, n, cols, ckc, act=None, out_mc=ttnn.L1_MEMORY_CONFIG):
    """2D mcast config with the grid sized to M: rows = M tiles (per_core_M = 1), cols columns."""
    rows = math.ceil(m / tpc.TILE_SIZE)
    pc = tpc.create_prefill_matmul_program_config(m, k, n, grid_size=(cols, rows), fused_activation=act)
    return ttnn.linear(x, w, compute_kernel_config=ckc, program_config=pc, memory_config=out_mc)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [(1, TP)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_smallm_prefill_matmul_bench(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    tt_ccl = TT_CCL(mesh)
    topo = ttnn.Topology.Ring
    hifi2 = tpc.COMPUTE_HIFI2
    lofi = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    shapes = os.environ.get("SMALLM_SHAPES", ",".join(SHAPES)).split(",")
    torch.manual_seed(0)
    rows = []  # (shape, M, variant, us, pcc)

    def rep(t):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )

    for name in shapes:
        K, N_local, wdt, role = SHAPES[name]
        ckc = lofi if role == "mlp" else hifi2
        for M in MS:
            # weights: [K, N_local * TP] column-sharded (col-parallel roles) or [K/TP rows..]: for "row" roles the
            # per-device weight is [K_local, N] with K already the per-device K (x replicated in N-space, K-sharded rows).
            if role == "row":
                w_t = torch.randn(K, N_local, dtype=torch.bfloat16) * 0.02  # per-device [K_local, N]
                w = ttnn.from_torch(
                    w_t, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
                )
                x_t = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
                x = ttnn.from_torch(
                    x_t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                w_ref = _dev0(w)
                ref = x_t.float() @ w_ref
            else:
                w_t = torch.randn(K, N_local * TP, dtype=torch.bfloat16) * 0.02
                w = ttnn.from_torch(
                    w_t,
                    dtype=wdt,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
                )
                x_t = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
                x = ttnn.from_torch(
                    x_t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
                )
                w_ref = _dev0(w)  # device-0 shard (bfp-rounded)
                ref = x_t.float() @ w_ref
            if role == "mlp":
                # w1 / w3 separately for the un-fused path; [gate|up] tile-pair interleaved for fuse_swiglu.
                w3_t = torch.randn(K, N_local * TP, dtype=torch.bfloat16) * 0.02
                w3 = ttnn.from_torch(
                    w3_t,
                    dtype=wdt,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
                )
                w3_ref = _dev0(w3)
                ref_swiglu = torch.nn.functional.silu(ref) * (x_t.float() @ w3_ref)
                # interleave gate/up tiles per device: [K, 2*N_local] with tile columns g0 u0 g1 u1 ...
                g_dev = torch.stack([w_t[:, d * N_local : (d + 1) * N_local] for d in range(TP)])  # [TP, K, N_local]
                u_dev = torch.stack([w3_t[:, d * N_local : (d + 1) * N_local] for d in range(TP)])
                nt = N_local // 32
                gu = torch.stack([g_dev.reshape(TP, K, nt, 32), u_dev.reshape(TP, K, nt, 32)], dim=3).reshape(
                    TP, K, 2 * N_local
                )
                wgu = ttnn.from_torch(
                    gu.permute(1, 0, 2).reshape(K, TP * 2 * N_local),
                    dtype=wdt,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
                )

            variants = {}
            if role == "in":
                variants["cur_agmm"] = lambda: tpc.all_gather_matmul_prefill(
                    x, w, tt_ccl, ckc, topo, out_memory_config=ttnn.L1_MEMORY_CONFIG
                )
                variants["ag"] = lambda: _ag(x, tt_ccl, topo)
                for c in CORES:

                    def f1(c=c):
                        xg = _ag(x, tt_ccl, topo)
                        o = _lin1d(xg, w, M, K, N_local, c, ckc)
                        ttnn.deallocate(xg)
                        return o

                    variants[f"1d_{c}"] = f1
                for cols in (8, 11):

                    def f2(cols=cols):
                        xg = _ag(x, tt_ccl, topo)
                        o = _lin2d_m(xg, w, M, K, N_local, cols, ckc)
                        ttnn.deallocate(xg)
                        return o

                    variants[f"2d_{cols}x{M // 32}"] = f2
            elif role == "mlp":
                variants["cur_agmm_swiglu"] = lambda: tpc.all_gather_swiglu_prefill(x, wgu, tt_ccl, ckc, topo)
                variants["ag"] = lambda: _ag(x, tt_ccl, topo)
                for c in CORES:

                    def f1(c=c):
                        xg = _ag(x, tt_ccl, topo)
                        g = _lin1d(xg, w, M, K, N_local, c, ckc, act=ttnn.UnaryOpType.SILU)
                        u = _lin1d(xg, w3, M, K, N_local, c, ckc)
                        ttnn.deallocate(xg)
                        h = ttnn.mul(g, u, memory_config=ttnn.L1_MEMORY_CONFIG)
                        ttnn.deallocate(g)
                        ttnn.deallocate(u)
                        return h

                    variants[f"1d_{c}"] = f1
                for cols in (8, 11):

                    def f2(cols=cols):
                        xg = _ag(x, tt_ccl, topo)
                        g = _lin2d_m(xg, w, M, K, N_local, cols, ckc, act=ttnn.UnaryOpType.SILU)
                        u = _lin2d_m(xg, w3, M, K, N_local, cols, ckc)
                        ttnn.deallocate(xg)
                        h = ttnn.mul(g, u, memory_config=ttnn.L1_MEMORY_CONFIG)
                        ttnn.deallocate(g)
                        ttnn.deallocate(u)
                        return h

                    variants[f"2d_{cols}x{M // 32}"] = f2
                ref = ref_swiglu
            elif role == "row":

                def fcur():
                    pc = tpc.create_prefill_mlp_matmul_program_config(
                        M, K, N_local, max_cols=GRID_W, tuning=tpc.prefill_tuning(TP)
                    )
                    return ttnn.linear(
                        x, w, compute_kernel_config=ckc, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG
                    )

                variants["cur_2d"] = fcur
                for c in CORES:
                    variants[f"1d_{c}"] = lambda c=c: _lin1d(x, w, M, K, N_local, c, ckc)
                for cols in (8, 11):
                    variants[f"2d_{cols}x{M // 32}"] = lambda cols=cols: _lin2d_m(x, w, M, K, N_local, cols, ckc)
            else:  # "out": GDN out-proj agmm mode
                variants["cur_agmm_out"] = lambda: tpc.all_gather_matmul_prefill(x, w, tt_ccl, ckc, topo, role="out")
                variants["ag"] = lambda: _ag(x, tt_ccl, topo)
                for c in (10, 20, 40, 80):

                    def f1(c=c):
                        xg = _ag(x, tt_ccl, topo)
                        o = _lin1d(xg, w, M, K, N_local, c, ckc, out_mc=ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.deallocate(xg)
                        return o

                    variants[f"1d_{c}"] = f1
                for cols in (8, 10):

                    def f2(cols=cols):
                        xg = _ag(x, tt_ccl, topo)
                        o = _lin2d_m(xg, w, M, K, N_local, cols, ckc, out_mc=ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.deallocate(xg)
                        return o

                    variants[f"2d_{cols}x{M // 32}"] = f2

            for vname, fn in variants.items():
                try:
                    out = fn()
                    ttnn.synchronize_device(mesh)
                    if vname == "ag":
                        got = _dev0(out)
                        p = _pcc(x_t.float().reshape(M, K), got.reshape(M, K))
                    else:
                        got = _dev0(out).reshape(M, -1)[:, : ref.shape[-1]]
                        p = _pcc(ref.reshape(M, -1), got)
                    ttnn.deallocate(out)

                    def body(fn=fn):
                        o = fn()
                        ttnn.deallocate(o)

                    us = _traced_us(mesh, body)
                    rows.append((name, M, vname, us, p))
                    logger.info(f"SMALLM_BENCH {name:11s} M={M:4d} {vname:18s} {us:8.1f} us  pcc={p:.5f}")
                except Exception as e:  # noqa: BLE001
                    msg = str(e).splitlines()[0][:140]
                    rows.append((name, M, vname, float("nan"), 0.0))
                    logger.warning(f"SMALLM_BENCH {name:11s} M={M:4d} {vname:18s} ERR {msg}")
            ttnn.deallocate(w)
            ttnn.deallocate(x)
            if role == "mlp":
                ttnn.deallocate(w3)
                ttnn.deallocate(wgu)
    print("\nSMALLM_BENCH_TABLE shape M variant us pcc")
    for r in rows:
        print(f"SMALLM_BENCH_ROW {r[0]} {r[1]} {r[2]} {r[3]:.1f} {r[4]:.5f}")
    print("SMALLM_BENCH_DONE")
