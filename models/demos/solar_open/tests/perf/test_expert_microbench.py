# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profiling helper, not a correctness test (SKIPS unless ``SOLAR_OPEN_PERF_PROFILE=1``): micro-benchmarks of the
Solar-Open expert / attention /
norm / router shapes on ONE device (random bfp8 weights, real shapes: H=4096, Ip=160, E=128, fused gate|up N=320,
down N=4096).

Every configuration runs REPS times between "mb_<name>_start" / "mb_<name>_stop" signposts so the tracy ops CSV
gives its device kernel duration; SOLAR_OPEN_PERF_OUT receives the host wall times and any validation failures.
Single-core sparse_matmul grids are excluded: a 1x1 mcast_in0 grid (no receivers) hung the device (2026-09-07).

    SOLAR_OPEN_PERF_OUT=/path/mb.json python -m tracy -r -p -v -m pytest \
        models/demos/solar_open/tests/perf/test_expert_microbench.py -k 1x1 -x -p no:cacheprovider
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.solar_open.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.solar_open.tt.experts.prefill import _DENSE_COMPUTE_KERNEL_CONFIG

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(header, message=None):
        logger.info(f"SIGNPOST {header}")


PERF_OUT = os.getenv("SOLAR_OPEN_PERF_OUT", "")
PROFILE = os.getenv("SOLAR_OPEN_PERF_PROFILE", "") == "1"  # opt-in: the helper is meant to run under the tracy profiler
REPS = int(os.getenv("SOLAR_OPEN_PERF_REPS", "8"))
H, IP, E, N_GU = 4096, 160, 128, 320
TILE = ttnn.Tile([32, 32])
SILU = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)]


def _pc(cores, in0_block_w, per_core_M, per_core_N, out_subblock_w=1, mcast_in0=True, fuse_batch=False):
    """Mirror of experts/config.py ProgramConfig._build_matmul_config (out_block_w == out_subblock_w)."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*cores),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=fuse_batch,
        fused_activation=None,
        mcast_in0=mcast_in0,
    )


def _bmm_pc(grid, mt, kt, nt_per_core, in0_block_w=None):
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w or next(d for d in (6, 5, 4, 3, 2, 1) if kt % d == 0),
        out_subblock_h=1,
        out_subblock_w=next(d for d in (8, 6, 4, 3, 2, 1) if nt_per_core % d == 0),
        per_core_M=mt,
        per_core_N=nt_per_core,
    )


def _sparsity(device, nnz, generator):
    mask = torch.zeros(1, 1, 1, E)
    idx = torch.randperm(E, generator=generator)[:nnz]
    mask[..., idx] = 0.125
    return ttnn.from_torch(mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), idx.sort().values


class Bench:
    def __init__(self, device):
        self.device = device
        self.results = {}
        self.failures = {}

    def run(self, name, fn, reps=REPS):
        """fn() -> output tensor (deallocated here). Compile run, then `reps` signposted runs."""
        try:
            out = fn()
            ttnn.synchronize_device(self.device)
            if out is not None:
                out.deallocate(True)
        except Exception as exc:  # validation failure of the program config
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
            logger.warning(f"[mb {name}] FAILED: {msg}")
            self.failures[name] = msg
            return False
        walls = []
        signpost(f"mb_{name}_start")
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(self.device)
            walls.append((time.perf_counter() - t0) * 1e3)
            if out is not None:
                out.deallocate(True)
        signpost(f"mb_{name}_stop")
        self.results[name] = {"wall_ms_min": min(walls), "wall_ms_mean": sum(walls) / len(walls), "reps": reps}
        logger.info(f"[mb {name}] wall min {min(walls):.3f} ms mean {sum(walls) / len(walls):.3f} ms")
        return True

    def note(self, key, value):
        self.results[key] = value
        logger.info(f"[mb note] {key} = {value}")

    def dump(self):
        if not PERF_OUT:
            return
        data = {}
        if os.path.isfile(PERF_OUT):
            data = json.loads(open(PERF_OUT).read())
        data["microbench"] = {"results": self.results, "failures": self.failures}
        with open(PERF_OUT, "w") as f:
            json.dump(data, f, indent=2)


@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(1, 1)])
def test_expert_microbench(mesh_device, device_params, reset_seeds):
    if not PROFILE:
        pytest.skip(
            "profiling helper: set SOLAR_OPEN_PERF_PROFILE=1 (and run under `python -m tracy -r -p -v -m pytest ...`)"
        )
    device = mesh_device
    g = torch.Generator().manual_seed(7)
    bench = Bench(device)
    compute = _DENSE_COMPUTE_KERNEL_CONFIG
    grid_x_max = device.compute_with_storage_grid_size().x
    grid_y = device.compute_with_storage_grid_size().y
    logger.info(f"compute grid {grid_x_max}x{grid_y}")

    def up(t, dtype=ttnn.bfloat8_b, mem=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=mem)

    logger.info("uploading random expert weights (bfp8) ...")
    w_gu_t = torch.randn(1, E, H, N_GU, generator=g) * 0.02
    w_down_t = torch.randn(1, E, IP, H, generator=g) * 0.02
    w_gu = up(w_gu_t)  # [1, E, H, 2Ip]
    w_down = up(w_down_t)  # [1, E, Ip, H]
    x1_t = torch.randn(1, 1, 1, H, generator=g)
    x32_t = torch.randn(1, 1, 32, H, generator=g)
    x1 = up(x1_t, dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)  # decode b1 input
    x32 = up(x32_t, dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)  # decode b32 input
    x32_bfp8 = up(x32_t, mem=ttnn.L1_MEMORY_CONFIG)
    act_down = up(torch.randn(1, E, 32, IP, generator=g))  # GLU output (bfp8) for the down projection
    sparsities = {}
    sparse_idx = {}
    for nnz in (8, 9, 64, 112, 113, 128):
        sparsities[nnz], sparse_idx[nnz] = _sparsity(device, nnz, g)

    # ---------------- A. fused gate|up decode sparse_matmul (Nt = 10) ----------------
    def gate_up(x, nnz, cores, in0_block_w, pcn, osw=1):
        return lambda: ttnn.sparse_matmul(
            x,
            w_gu,
            sparsity=sparsities[nnz],
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=TILE,
            program_config=_pc(cores, in0_block_w, 1, pcn, osw),
            dtype=ttnn.bfloat8_b,
        )

    for nnz, x in ((8, x1), (9, x1), (112, x32), (113, x32), (128, x32), (64, x32), (8, x32)):
        tag = f"gu_m{x.shape[2]}_nnz{nnz}"
        bench.run(f"{tag}_5x2_bw32", gate_up(x, nnz, (5, 2), 32, 1))  # production
        if nnz in (8, 112) and x is not x32 or nnz == 112:
            for bw in (16, 64, 128):
                bench.run(f"{tag}_5x2_bw{bw}", gate_up(x, nnz, (5, 2), bw, 1))
            bench.run(f"{tag}_2x5_bw32", gate_up(x, nnz, (2, 5), 32, 1))
            bench.run(f"{tag}_10x1_bw32", gate_up(x, nnz, (10, 1), 32, 1))
            bench.run(f"{tag}_1x10_bw32", gate_up(x, nnz, (1, 10), 32, 1))
            bench.run(f"{tag}_5x1_pcn2_bw32", gate_up(x, nnz, (5, 1), 32, 2, 1))
            bench.run(f"{tag}_5x1_pcn2_osw2_bw32", gate_up(x, nnz, (5, 1), 32, 2, 2))
            bench.run(f"{tag}_2x1_pcn5_osw5_bw32", gate_up(x, nnz, (2, 1), 32, 5, 5))
    # in0 in bfp8 (halves the per-expert in0 multicast bytes) -- numerics change, measured for the record
    bench.run("gu_m32_nnz112_5x2_bw32_in0bfp8", gate_up(x32_bfp8, 112, (5, 2), 32, 1))
    bench.run("gu_m32_nnz112_5x2_bw128_in0bfp8", gate_up(x32_bfp8, 112, (5, 2), 128, 1))

    # ---------------- A-EGP. expert-group parallelism (phase 3b): ttnn.sparse_matmul(expert_groups=G) ----------------
    # Grid = G x output blocks (gate|up: 10 blocks -> (10, G) or (11, 10) at G 11). Bit-identical to the legacy kernel
    # (checked here as a torch.equal note per configuration; the op-level matrix is tests/ttnn/unit_tests/operations/
    # matmul/test_sparse_matmul_expert_groups.py). Shipped: G 11 on 11x10 (gate|up), pcn16 osw8 G 11 on 11x8 (down).
    def gate_up_egp(x, nnz, G, bw=128, cores=None):
        cores = cores or ((11, 10) if G == 11 else (10, G))
        return lambda: ttnn.sparse_matmul(
            x,
            w_gu,
            sparsity=sparsities[nnz],
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=TILE,
            program_config=_pc(cores, bw, 1, 1, 1),
            dtype=ttnn.bfloat8_b,
            expert_groups=G,
        )

    for nnz, x in ((8, x1), (8, x32), (64, x32), (112, x32), (128, x32)):
        tag = f"gu_m{x.shape[2]}_nnz{nnz}"
        for G in (1, 5, 11):
            if bench.run(f"{tag}_egp_G{G}_bw128", gate_up_egp(x, nnz, G)) and nnz in (8, 112):
                legacy = gate_up(x, nnz, (5, 2), 128, 1)()
                egp = gate_up_egp(x, nnz, G)()
                bench.note(
                    f"{tag}_egp_G{G}_bw128_equal_legacy", bool(torch.equal(ttnn.to_torch(legacy), ttnn.to_torch(egp)))
                )
                legacy.deallocate(True)
                egp.deallocate(True)
    bench.run("gu_m32_nnz112_egp_G11_bw32", gate_up_egp(x32, 112, 11, bw=32))  # K in 4 blocks: a different sum order

    # ---------------- A'. indexed / gather mode (top-k ids instead of the 128-slot scan) ----------------
    idx8 = up(sparse_idx[8].reshape(1, 1, 1, 8).to(torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
    idx9 = up(sparse_idx[9].reshape(1, 1, 1, 9).to(torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def gate_up_idx(x, idx, nnz, cores=(5, 2), bw=32):
        return lambda: ttnn.sparse_matmul(
            x,
            w_gu,
            sparsity=sparsities[nnz],
            indices=idx,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=TILE,
            program_config=_pc(cores, bw, 1, 1),
            dtype=ttnn.bfloat8_b,
        )

    def gate_up_idx_egp(x, idx, nnz, G=11, cores=(11, 10), bw=128):
        return lambda: ttnn.sparse_matmul(
            x,
            w_gu,
            sparsity=sparsities[nnz],
            indices=idx,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=TILE,
            program_config=_pc(cores, bw, 1, 1),
            dtype=ttnn.bfloat8_b,
            expert_groups=G,
        )

    ok_gu_idx = bench.run("gu_m1_idx8_5x2_bw32", gate_up_idx(x1, idx8, 8))
    bench.run("gu_m1_idx9_5x2_bw32", gate_up_idx(x1, idx9, 9))
    bench.run("gu_m1_idx8_5x2_bw128", gate_up_idx(x1, idx8, 8, bw=128))
    if bench.run("gu_m1_idx8_egp_G11_bw128", gate_up_idx_egp(x1, idx8, 8)):  # shipped b1 indexed gate|up (phase 3b)
        legacy = gate_up_idx(x1, idx8, 8, bw=128)()
        egp = gate_up_idx_egp(x1, idx8, 8)()
        bench.note(
            "gu_m1_idx8_egp_G11_bw128_equal_legacy", bool(torch.equal(ttnn.to_torch(legacy), ttnn.to_torch(egp)))
        )
        legacy.deallocate(True)
        egp.deallocate(True)
    bench.run("gu_m1_idx8_egp_G5_bw128", gate_up_idx_egp(x1, idx8, 8, G=5, cores=(10, 5)))
    if ok_gu_idx:
        # correctness of the compact output vs torch, and the compact down projection (A compact [1, 8, 32, Ip])
        gu_c = gate_up_idx(x1, idx8, 8)()
        logger.info(f"indexed gate_up output shape {gu_c.shape}")
        ref_gu = torch.einsum("k,ekn->en", x1_t.reshape(H).float(), w_gu_t[0, sparse_idx[8]].float())  # [8, 320]
        got = ttnn.to_torch(gu_c).float().reshape(-1, N_GU)
        bench.note("gu_idx8_pcc_vs_torch", comp_pcc(ref_gu, got, 0.98)[1])
        gu_c = ttnn.reshape(gu_c, (1, 8, 1, N_GU))
        gate_c = ttnn.slice(gu_c, [0, 0, 0, 0], [1, 8, 1, IP])
        up_c = ttnn.slice(gu_c, [0, 0, 0, IP], [1, 8, 1, N_GU])
        act_c = ttnn.mul(up_c, gate_c, input_tensor_b_activations=SILU)  # [1, 8, 1, Ip] compact

        def down_idx(cores, pcn, osw=1, a_sparse=True):
            return lambda: ttnn.sparse_matmul(
                act_c,
                w_down,
                sparsity=sparsities[8],
                indices=idx8,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=TILE,
                is_input_a_sparse=a_sparse,
                is_input_b_sparse=True,
                program_config=_pc(cores, 5, 1, pcn, osw),
                dtype=ttnn.bfloat8_b,
            )

        def down_idx_egp(cores, pcn, osw, G):
            return lambda: ttnn.sparse_matmul(
                act_c,
                w_down,
                sparsity=sparsities[8],
                indices=idx8,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=TILE,
                is_input_a_sparse=True,
                is_input_b_sparse=True,
                program_config=_pc(cores, 5, 1, pcn, osw),
                dtype=ttnn.bfloat8_b,
                expert_groups=G,
            )

        ok_down_idx = bench.run("down_idx8_8x4_pcn4", down_idx((8, 4), 4))
        if not ok_down_idx:
            ok_down_idx = bench.run("down_idx8_8x4_pcn4_a_not_sparse", down_idx((8, 4), 4, a_sparse=False))
        if ok_down_idx:
            bench.run("down_idx8_8x8_pcn2", down_idx((8, 8), 2))
            bench.run("down_idx8_8x4_pcn4_osw4", down_idx((8, 4), 4, 4))
            bench.run("down_idx8_8x8_pcn2_osw2", down_idx((8, 8), 2, 2))  # the measured best b1 indexed down (legacy)
            bench.run("down_idx8_egp_pcn16_osw8_G11_11x8", down_idx_egp((11, 8), 16, 8, 11))  # no EGP gain at k = 8
            bench.run("down_idx8_egp_pcn8_osw8_G5_8x10", down_idx_egp((8, 10), 8, 8, 5))
            d_c = down_idx((8, 4), 4)()
            logger.info(f"indexed down output shape {d_c.shape}")
            act_ref = ttnn.to_torch(act_c).float().reshape(8, IP)
            ref_d = torch.einsum("ek,ekn->en", act_ref, w_down_t[0, sparse_idx[8]].float())
            got_d = ttnn.to_torch(d_c).float().reshape(-1, H)
            bench.note("down_idx8_pcc_vs_torch", comp_pcc(ref_d, got_d, 0.98)[1])
            d_c.deallocate(True)
        for t in (gu_c, gate_c, up_c, act_c):
            t.deallocate(True)

    # ---------------- B. down decode sparse_matmul (Nt = 128, Kt = 5) ----------------
    def down(nnz, cores, pcn, osw=1, bw=5):
        return lambda: ttnn.sparse_matmul(
            act_down,
            w_down,
            sparsity=sparsities[nnz],
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=TILE,
            is_input_a_sparse=True,
            program_config=_pc(cores, bw, 1, pcn, osw),
            dtype=ttnn.bfloat8_b,
        )

    def down_egp(nnz, cores, pcn, osw, G, bw=5):
        return lambda: ttnn.sparse_matmul(
            act_down,
            w_down,
            sparsity=sparsities[nnz],
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=TILE,
            is_input_a_sparse=True,
            program_config=_pc(cores, bw, 1, pcn, osw),
            dtype=ttnn.bfloat8_b,
            expert_groups=G,
        )

    for nnz in (8, 9, 64, 112, 113, 128):
        tag = f"down_nnz{nnz}"
        bench.run(f"{tag}_8x4_pcn4", down(nnz, (8, 4), 4))  # production b1 (legacy)
        bench.run(f"{tag}_8x8_pcn2", down(nnz, (8, 8), 2))  # phase-2 production b32 (legacy)
        # phase 3b (expert groups): the shipped batched down pcn16 osw8 G 11 on 11x8 and the pcn8 osw8 G 4 / 5 fallbacks
        if bench.run(f"{tag}_egp_pcn16_osw8_G11_11x8", down_egp(nnz, (11, 8), 16, 8, 11)) and nnz in (8, 112):
            legacy = down(nnz, (8, 8), 2, 2)()
            egp = down_egp(nnz, (11, 8), 16, 8, 11)()
            bench.note(
                f"{tag}_egp_pcn16_osw8_G11_11x8_equal_legacy",
                bool(torch.equal(ttnn.to_torch(legacy), ttnn.to_torch(egp))),
            )
            legacy.deallocate(True)
            egp.deallocate(True)
        if nnz in (8, 112):
            bench.run(f"{tag}_egp_pcn8_osw8_G4_8x8", down_egp(nnz, (8, 8), 8, 8, 4))
            bench.run(f"{tag}_egp_pcn8_osw8_G5_8x10", down_egp(nnz, (8, 10), 8, 8, 5))
        if nnz in (8, 112):
            bench.run(f"{tag}_8x4_pcn4_osw2", down(nnz, (8, 4), 4, 2))
            bench.run(f"{tag}_8x4_pcn4_osw4", down(nnz, (8, 4), 4, 4))
            bench.run(f"{tag}_8x8_pcn2_osw2", down(nnz, (8, 8), 2, 2))
            bench.run(f"{tag}_4x8_pcn4", down(nnz, (4, 8), 4))
            bench.run(f"{tag}_4x8_pcn4_osw4", down(nnz, (4, 8), 4, 4))
            bench.run(f"{tag}_8x2_pcn8", down(nnz, (8, 2), 8))
            bench.run(f"{tag}_8x2_pcn8_osw8", down(nnz, (8, 2), 8, 8))
            bench.run(f"{tag}_8x2_pcn8_osw4", down(nnz, (8, 2), 8, 4))
            bench.run(f"{tag}_4x4_pcn8_osw4", down(nnz, (4, 4), 8, 4))
            bench.run(f"{tag}_8x1_pcn16_osw8", down(nnz, (8, 1), 16, 8))
            bench.run(f"{tag}_8x4_pcn4_bw1", down(nnz, (8, 4), 4, 1, 1))

    # ---------------- C. dense (all-expert) batched matmuls: decode-b32 alternative and prefill splits -------------
    def bmm_gate_up(hidden_rep, s, grid, nt_per_core=N_GU // 32):
        cfg = _bmm_pc(grid, s // 32, H // 32, nt_per_core)
        return lambda: ttnn.matmul(
            hidden_rep,
            w_gu,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=cfg,
            compute_kernel_config=compute,
        )

    def bmm_down_auto(act, grid):
        return lambda: ttnn.matmul(
            act,
            w_down,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=grid,
            compute_kernel_config=compute,
        )

    def bmm_down_pc(act, s, grid, nt_per_core):
        cfg = _bmm_pc(grid, s // 32, IP // 32, nt_per_core)
        return lambda: ttnn.matmul(
            act,
            w_down,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=cfg,
            compute_kernel_config=compute,
        )

    def bmm_down_1d(act, s, cores, pcn, bw):
        cfg = _pc(cores, bw, s // 32, pcn, 1)
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*cores),
            in0_block_w=bw,
            out_subblock_h=1,
            out_subblock_w=2 if pcn % 2 == 0 else 1,
            out_block_h=s // 32,
            out_block_w=pcn,
            per_core_M=s // 32,
            per_core_N=pcn,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        return lambda: ttnn.matmul(
            act,
            w_down,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=cfg,
            compute_kernel_config=compute,
        )

    grids = {
        "w10": ttnn.CoreGrid(y=grid_y, x=10),
        f"w{min(11, grid_x_max)}": ttnn.CoreGrid(y=grid_y, x=min(11, grid_x_max)),
        "8x8": ttnn.CoreGrid(y=8, x=8),
    }
    for s in (32, 128, 256, 512):
        hidden = up(torch.randn(1, 1, s, H, generator=g), dtype=ttnn.bfloat16)
        bench.run(f"repeat_s{s}", lambda: ttnn.repeat(hidden, ttnn.Shape((1, E, 1, 1))))
        bench.run(
            f"repeat_s{s}_bfp8out", lambda: ttnn.repeat(ttnn.typecast(hidden, ttnn.bfloat8_b), ttnn.Shape((1, E, 1, 1)))
        )
        hidden_rep = ttnn.repeat(hidden, ttnn.Shape((1, E, 1, 1)))
        hidden_rep8 = ttnn.typecast(hidden_rep, ttnn.bfloat8_b)
        act = up(torch.randn(1, E, s, IP, generator=g))
        for gname, grid in grids.items():
            bench.run(f"bmm_gu_s{s}_{gname}", bmm_gate_up(hidden_rep, s, grid))
            bench.run(f"bmm_gu_s{s}_{gname}_in0bfp8", bmm_gate_up(hidden_rep8, s, grid))
            bench.run(f"bmm_down_s{s}_{gname}_auto", bmm_down_auto(act, grid))
            for nt in (128, 64, 32, 16):
                bench.run(f"bmm_down_s{s}_{gname}_pcn{nt}", bmm_down_pc(act, s, grid, nt))
        bench.run(f"bmm_down_s{s}_1d_8x8_pcn2_bw5", bmm_down_1d(act, s, (8, 8), 2, 5))
        bench.run(f"bmm_down_s{s}_1d_8x4_pcn4_bw5", bmm_down_1d(act, s, (8, 4), 4, 5))
        gu = up(torch.randn(1, E, s, N_GU, generator=g))
        gsl = ttnn.slice(gu, [0, 0, 0, 0], [1, E, s, IP])
        usl = ttnn.slice(gu, [0, 0, 0, IP], [1, E, s, N_GU])
        bench.run(f"slice_gate_s{s}", lambda: ttnn.slice(gu, [0, 0, 0, 0], [1, E, s, IP]))
        bench.run(f"glu_s{s}", lambda: ttnn.mul(usl, gsl, input_tensor_b_activations=SILU))
        rw = up(torch.rand(1, E, s, 1, generator=g), dtype=ttnn.bfloat16)
        bench.run(f"routing_mul_down_input_s{s}", lambda: ttnn.mul(act, rw))
        dn = up(torch.randn(1, E, s, H, generator=g))
        bench.run(f"routing_mul_down_output_s{s}", lambda: ttnn.mul(dn, rw))
        bench.run(f"fast_reduce_nc_s{s}", lambda: ttnn.experimental.fast_reduce_nc(dn, dims=[1]))
        for t in (hidden, hidden_rep, hidden_rep8, act, gu, gsl, usl, rw, dn):
            t.deallocate(True)
    # one expert of the dense per-expert prefill loop at 1024 tokens
    h1024 = up(torch.randn(1, 1, 1024, H, generator=g), dtype=ttnn.bfloat16)
    w_e = ttnn.slice(w_gu, [0, 0, 0, 0], [1, 1, H, N_GU])
    wd_e = ttnn.slice(w_down, [0, 0, 0, 0], [1, 1, IP, H])
    bench.run("expert_slice_gu", lambda: ttnn.slice(w_gu, [0, 3, 0, 0], [1, 4, H, N_GU]))
    bench.run(
        "dense_expert_gu_s1024",
        lambda: ttnn.linear(
            h1024,
            w_e,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=grids["w10"],
            compute_kernel_config=compute,
        ),
    )
    a1024 = up(torch.randn(1, 1, 1024, IP, generator=g))
    bench.run(
        "dense_expert_down_s1024",
        lambda: ttnn.linear(
            a1024,
            wd_e,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=grids["w10"],
            compute_kernel_config=compute,
        ),
    )
    for t in (h1024, w_e, wd_e, a1024):
        t.deallocate(True)

    # ---------------- D. shared expert (4 launches) at M = 1 and 32: auto vs explicit 1D configs ----------------
    w_sg = up(torch.randn(1, 1, H, IP, generator=g))
    w_su = up(torch.randn(1, 1, H, IP, generator=g))
    w_sd = up(torch.randn(1, 1, IP, H, generator=g))
    w_sgu = up(torch.randn(1, 1, H, N_GU, generator=g))  # fused [gate|up] alternative
    shared_compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def lin(x, w, pc=None):
        return lambda: ttnn.linear(
            x,
            w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=shared_compute,
            program_config=pc,
        )

    for x in (x1, x32):
        m = x.shape[2]
        bench.run(f"shared_gate_m{m}_auto", lin(x, w_sg))
        for bw in (8, 16, 32, 64, 128):
            bench.run(f"shared_gate_m{m}_5x1_bw{bw}", lin(x, w_sg, _pc((5, 1), bw, 1, 1)))
        bench.run(f"shared_gate_m{m}_1x5_bw32", lin(x, w_sg, _pc((1, 5), 32, 1, 1)))
        bench.run(f"shared_gateup_fused_m{m}_5x2_bw32", lin(x, w_sgu, _pc((5, 2), 32, 1, 1)))
        bench.run(f"shared_gateup_fused_m{m}_10x1_bw32", lin(x, w_sgu, _pc((10, 1), 32, 1, 1)))
        bench.run(f"shared_gateup_fused_m{m}_5x2_bw128", lin(x, w_sgu, _pc((5, 2), 128, 1, 1)))
        gsh = lin(x, w_sg)()
        ush = lin(x, w_su)()
        bench.run(f"shared_glu_m{m}", lambda: ttnn.mul(ush, gsh, input_tensor_b_activations=SILU))
        a_sh = ttnn.mul(ush, gsh, input_tensor_b_activations=SILU)
        bench.run(f"shared_down_m{m}_auto", lin(a_sh, w_sd))
        bench.run(f"shared_down_m{m}_8x8_pcn2_bw5", lin(a_sh, w_sd, _pc((8, 8), 5, 1, 2, 1)))
        bench.run(f"shared_down_m{m}_8x8_pcn2_osw2_bw5", lin(a_sh, w_sd, _pc((8, 8), 5, 1, 2, 2)))
        bench.run(f"shared_down_m{m}_8x4_pcn4_osw4_bw5", lin(a_sh, w_sd, _pc((8, 4), 5, 1, 4, 4)))
        for t in (gsh, ush, a_sh):
            t.deallocate(True)

    # ---------------- E. router linear (bf16 x bf16 -> fp32, HiFi4, fp32 acc) ----------------
    w_router = up(torch.randn(1, 1, H, 128, generator=g) * 0.04, dtype=ttnn.bfloat16)
    router_compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def router(x, pc=None, dtype=ttnn.float32):
        return lambda: ttnn.linear(
            x,
            w_router,
            dtype=dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=router_compute,
            program_config=pc,
        )

    for x in (x1, x32):
        m = x.shape[2]
        bench.run(f"router_m{m}_auto_fp32", router(x))
        bench.run(f"router_m{m}_auto_bf16out", router(x, dtype=ttnn.bfloat16))
        for bw in (8, 16, 32, 64, 128):
            bench.run(f"router_m{m}_4x1_bw{bw}_fp32", router(x, _pc((4, 1), bw, 1, 1)))
        bench.run(f"router_m{m}_2x1_pcn2_bw32_fp32", router(x, _pc((2, 1), 32, 1, 2, 2)))
        bench.run(f"router_m{m}_2x2_bw32_fp32", router(x, _pc((2, 2), 32, 1, 1)))

    # ---------------- F. RMSNorm: default (1 core) vs width-sharded program config ----------------
    w_norm = up(torch.randn(1, 1, H // 32, 32, generator=g), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    for x in (x1, x32):
        m = x.shape[2]
        bench.run(f"rmsnorm_m{m}_default", lambda: ttnn.rms_norm(x, weight=w_norm, epsilon=1e-5))
        for gx, gy in ((8, 4), (8, 8), (4, 4), (8, 2)):
            ncores = gx * gy
            shard_w = H // ncores
            if shard_w % 32:
                continue
            block_w = shard_w // 32
            subblock_w = next(d for d in (4, 3, 2, 1) if block_w % d == 0)
            mem = ttnn.create_sharded_memory_config(
                shape=(32, shard_w),
                core_grid=ttnn.CoreGrid(y=gy, x=gx),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[gx, gy],
                subblock_w=subblock_w,
                block_h=1,
                block_w=block_w,
                inplace=False,
            )
            try:
                xs = ttnn.interleaved_to_sharded(x, mem)
            except Exception as exc:
                bench.failures[f"rmsnorm_m{m}_shard_{gx}x{gy}"] = str(exc).splitlines()[0][:200]
                continue
            bench.run(f"i2s_m{m}_{gx}x{gy}", lambda: ttnn.interleaved_to_sharded(x, mem))
            bench.run(
                f"rmsnorm_m{m}_sharded_{gx}x{gy}",
                lambda: ttnn.rms_norm(xs, weight=w_norm, epsilon=1e-5, program_config=pc, memory_config=mem),
            )
            ys = ttnn.rms_norm(xs, weight=w_norm, epsilon=1e-5, program_config=pc, memory_config=mem)
            bench.run(f"s2i_m{m}_{gx}x{gy}", lambda: ttnn.sharded_to_interleaved(ys, ttnn.L1_MEMORY_CONFIG))
            if gx == 8 and gy == 4:
                ref = ttnn.to_torch(ttnn.rms_norm(x, weight=w_norm, epsilon=1e-5)).float()
                got = ttnn.to_torch(ttnn.sharded_to_interleaved(ys, ttnn.L1_MEMORY_CONFIG)).float()
                bench.note(f"rmsnorm_m{m}_sharded_8x4_pcc_vs_default", comp_pcc(ref, got, 0.99)[1])
            xs.deallocate(True)
            ys.deallocate(True)

    # ---------------- G. attention decode projections and lm_head ----------------
    w_qkv = up(torch.randn(1, 1, H, 1280, generator=g))
    w_o = up(torch.randn(1, 1, 1024, H, generator=g))
    w_lm = up(torch.randn(1, 1, H, 24576, generator=g))
    sdpa_out = up(torch.randn(1, 1, 32, 1024, generator=g), dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
    attn_compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    for x in (x1, x32):
        m = x.shape[2]
        bench.run(
            f"qkv_m{m}_auto",
            lambda: ttnn.matmul(x, w_qkv, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG),
        )
        for bw in (4, 8, 16, 32, 64, 128):
            bench.run(
                f"qkv_m{m}_8x5_bw{bw}",
                lambda bw=bw: ttnn.matmul(
                    x,
                    w_qkv,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    program_config=_pc((8, 5), bw, 1, 1, 1),
                    compute_kernel_config=attn_compute,
                ),
            )
        bench.run(
            f"qkv_m{m}_auto_l1interleaved",
            lambda: ttnn.matmul(x, w_qkv, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG),
        )
        bench.run(f"lm_head_m{m}_auto", lambda: ttnn.matmul(x, w_lm, dtype=ttnn.bfloat16))
        for bw in (4, 8, 16):
            bench.run(
                f"lm_head_m{m}_8x8_pcn12_bw{bw}",
                lambda bw=bw: ttnn.matmul(
                    x,
                    w_lm,
                    dtype=ttnn.bfloat16,
                    program_config=_pc((8, 8), bw, 1, 12, 4),
                    compute_kernel_config=attn_compute,
                ),
            )
            bench.run(
                f"lm_head_m{m}_11x10_pcn7_bw{bw}",
                lambda bw=bw: ttnn.matmul(
                    x,
                    w_lm,
                    dtype=ttnn.bfloat16,
                    program_config=_pc((min(11, grid_x_max), 10), bw, 1, 7, 7),
                    compute_kernel_config=attn_compute,
                ),
            )
    bench.run(
        "oproj_m32_auto",
        lambda: ttnn.linear(sdpa_out, w_o, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG),
    )
    for bw in (4, 8, 16, 32):
        for osw in (1, 2):
            bench.run(
                f"oproj_m32_8x8_bw{bw}_osw{osw}",
                lambda bw=bw, osw=osw: ttnn.linear(
                    sdpa_out,
                    w_o,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    program_config=_pc((8, 8), bw, 1, 2, osw),
                    compute_kernel_config=attn_compute,
                ),
            )
    bench.run(
        "oproj_m32_auto_l1interleaved",
        lambda: ttnn.linear(sdpa_out, w_o, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG),
    )
    bench.run("typecast_bf16_to_bfp8_m32", lambda: ttnn.typecast(x32, ttnn.bfloat8_b))

    bench.dump()
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass
    logger.info(f"micro-benchmark failures: {json.dumps(bench.failures, indent=1)}")
