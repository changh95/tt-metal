# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One-device A/B of the phase-3 PREFILL expert matmul candidates against a torch fp32 reference
(scratchpad/phase3/design_prefill_matmuls.md).

Runs on a 1x1 mesh with random bfloat8_b weights at the real Solar-Open TP=8 per-device shapes (H = 4096, Ip = 160,
E = 128, fused gate|up N = 320, down N = 4096). Every candidate is scored on numerics (PCC / max abs error vs the fp32
matmul of the device-rounded operands, plus PCC vs the production form on identical inputs) and on time: the
synchronised host wall of ONE launch (min of REPS, the test_config_candidates.py measure) and the per-op time of a BURST
of back-to-back launches followed by a single synchronize -- the launches pipeline on the device, so ``burst / BURST``
approaches the device kernel time for ops longer than the ~10-20 us host launch cost -- from which the TFLOP/s of the
op are derived. A candidate whose program config is rejected (TT_FATAL at validation / program creation) is recorded
under ``failures`` and skipped; the test only asserts that the production forms reproduce the fp32 reference (PCC >=
MIN_PCC) so the design numbers are always collected.

Sections (``SOLAR_OPEN_PREFILL_MM_SECTIONS=<comma list>`` selects a subset; default all):

  * ``hot_gu``: the per-expert fused gate|up linear ``[1, 1, 1024, 4096] bf16 x [1, 1, 4096, 320] bfp8 -> bfp8`` of the
    sorted path's hot group / the dense per-expert loop (``experts/prefill.py``, ``core_grid=`` auto = a 2D mcast
    config with in0_block_w 4): legacy 2D ``MatmulMultiCoreReuseMultiCastProgramConfig`` grids with in0_block_w 8-32
    and 4x1 / 4x2 subblocks, the 1D mcast-in1 "in0 reuse" form (in0 [1, 1, M, K] broadcast over a batched in1: ONE
    launch for n hot experts without ``ttnn.repeat``), ``ttnn.experimental.minimal_matmul`` (default and explicit
    blockings), two ``[4096, 160]`` linears vs the fused one, and the N-concat of n hot experts
    ``[1024, 4096] x [4096, n * 320]``.
  * ``hot_down``: the per-expert down ``[1, 1, 1024, 160] x [1, 1, 160, 4096]`` (auto vs minimal_matmul vs 2D), the
    batched hot down ``[1, n, 1024, 160] x [1, n, 160, 4096]`` (bmm auto = production vs the 2D config with the batch
    loop) + ``fast_reduce_nc``, and the K-concat form ``[1024, n * 160] x [n * 160, 4096]`` (minimal_matmul, legacy 2D).
  * ``bmm_gu``: the all-expert gate|up bmm ``[1, E, S, 4096] x [1, E, 4096, 320]`` at S = 32 / 64 / 96 / 128 (the dense
    128-token path and the sorted path's cold experts at ``cap`` rows): production ``_bmm_config`` (in0_block_w 4,
    subblock 1x2) vs in0_block_w 8 / 16 / 32 with Mt x 2 / 1 x 5 subblocks, 11x10 vs 8x8 grids, bfp8 in0.
  * ``repeat``: the activation broadcast of the dense path at S = 128: ``ttnn.repeat`` bf16 (production), typecast +
    repeat bfp8, ``ttnn.multiply`` broadcast (bf16 / bfp8 output).
  * ``bmm_down``: the all-expert down ``[1, E, S, 160] x [1, E, 160, 4096]`` at S = 32 / 96 / 128: production 1D (8,8)
    bw5 pcn2 osw2 (``get_dense_down_config``) vs out_subblock_h = Mt, the 2D (8,4) pcM1 pcN16 batch-loop form and the
    bmm with per_core_N = 128 / in0_block_w 1 (L1 probe).
  * ``dense_kcat``: the dense 128-token path's down + reduce as ONE K-concatenated matmul ``[128, E * 160] x [E * 160,
    4096]`` (the down weights viewed as ``[E * Ip, H]`` for free; the GLU output re-laid out token-major with
    permute + reshape) vs the production bmm + fast_reduce_nc; also the 1D in0-reuse gate|up form at 128 rows (4 cores).
  * ``scatter``: the sorted path's scatter matmul ``one-hot^T [1024, 16384] @ [16384, 4096]``: auto with transpose_a
    (production) vs 2D configs vs minimal_matmul on a pre-transposed one-hot (+ the transpose cost).

The ``*_wired`` candidates are the phase-3 production forms exactly as ``experts/prefill.py`` launches them from the
``SolarOpenProgramConfig`` defaults (``dense_expert_gate_up_minimal``, ``hot_down_kconcat_minimal`` over bf16 act
pieces, ``dense_down_max_subblock_h``) at the 1024-token split and the 512-token split of > 32K contexts; they are
asserted against the fp32 reference (PCC >= MIN_PCC, and within WIRED_PCC_MARGIN of the phase-2 form) and, for the
dense down, bit-identical to the phase-2 config.

    SOLAR_OPEN_PERF_OUT=/path/prefill_mm.json pytest models/demos/solar_open/tests/perf/test_prefill_matmul_candidates.py \
        -k 1x1 -x -p no:cacheprovider
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
from models.demos.solar_open.tt.expert_configs import SolarOpenProgramConfig
from models.demos.solar_open.tt.experts.prefill import (
    _DENSE_COMPUTE_KERNEL_CONFIG,
    _bmm_config,
    _dense_core_grid,
    _glu,
    _hot_down_kconcat_config,
)

PERF_OUT = os.getenv("SOLAR_OPEN_PERF_OUT", "")
REPS = int(os.getenv("SOLAR_OPEN_PERF_REPS", "5"))
BURST = int(os.getenv("SOLAR_OPEN_PERF_BURST", "8"))
SECTIONS = set(
    filter(
        None,
        os.getenv("SOLAR_OPEN_PREFILL_MM_SECTIONS", "hot_gu,hot_down,bmm_gu,repeat,bmm_down,dense_kcat,scatter").split(
            ","
        ),
    )
)
H, IP, E, N_GU = 4096, 160, 128, 320
SPLIT = 1024
SPLIT_LONG_CONTEXT = 512  # get_down_split_size above 32K tokens
MIN_PCC = 0.999  # production forms vs the fp32 reference (bfp8 output floor is ~0.9999)
WIRED_PCC_MARGIN = 2e-4  # a wired phase-3 form may not trail the phase-2 form's PCC vs fp32 by more than this
HOT_COUNTS = (4, 8, 15)
# The pooled operands hold one slot more than the largest hot count: a slice over the FULL leading range aliases its
# input (a ttnn.slice view), so ``w_hot.deallocate(True)`` at n = n_pool would free the pool itself (the 2026-09-08
# run lost w_gu_pool that way and its per-expert chain probes failed with "input_tensor.is_allocated()").
N_POOL = max(HOT_COUNTS) + 1

# HiFi2, bf16 destination (8 dst tiles) = the production compute config of every dense expert matmul.
COMPUTE = _DENSE_COMPUTE_KERNEL_CONFIG
COMPUTE_FP32 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


class Bench:
    """Timing + numerics ledger. ``run`` returns the candidate's output tensor (caller frees it) or None on failure."""

    def __init__(self, device):
        self.device = device
        self.results = {}
        self.failures = {}

    def time(self, name, fn, flops=None, reps=REPS, burst=BURST):
        try:
            out = fn()
            ttnn.synchronize_device(self.device)
        except Exception as exc:  # program config rejected (validation / program creation)
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:400]}"
            logger.warning(f"[pmm {name}] FAILED: {msg}")
            self.failures[name] = msg
            return None
        walls = []
        for _ in range(reps):
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(self.device)
            walls.append((time.perf_counter() - t0) * 1e3)
            o.deallocate(True)
        outs = []
        t0 = time.perf_counter()
        for _ in range(burst):
            outs.append(fn())
        ttnn.synchronize_device(self.device)
        burst_ms = (time.perf_counter() - t0) * 1e3 / burst
        for o in outs:
            o.deallocate(True)
        rec = {"wall_ms_min": min(walls), "wall_ms_mean": sum(walls) / len(walls), "burst_ms_per_op": burst_ms}
        if flops:
            rec["tflops_burst"] = flops / (burst_ms * 1e-3) / 1e12
            rec["tflops_wall_min"] = flops / (min(walls) * 1e-3) / 1e12
        self.results.setdefault(name, {}).update(rec)
        tf = f" {rec['tflops_burst']:.1f} TFLOP/s" if flops else ""
        logger.info(f"[pmm {name}] wall min {min(walls):.3f} ms, burst {burst_ms:.3f} ms/op{tf}")
        return out

    def numerics(self, name, out, reference, prod=None, shape=None):
        """PCC / max abs err of ``out`` vs the torch fp32 ``reference`` (and vs the production output ``prod``)."""
        got = ttnn.to_torch(out).float()
        ref = reference.reshape(got.shape) if shape is None else reference.reshape(shape)
        if shape is not None:
            got = got.reshape(shape)
        _, pcc = comp_pcc(ref, got, 0.0)
        rec = {"pcc_vs_ref": pcc, "max_abs_err": (ref - got).abs().max().item()}
        if prod is not None:
            p = prod.reshape(got.shape)
            _, rec["pcc_vs_prod"] = comp_pcc(p, got, 0.0)
            rec["identical_to_prod"] = bool(torch.equal(p, got))
        self.results.setdefault(name, {}).update(rec)
        logger.info(f"[pmm {name}] {json.dumps(rec)}")
        return got

    def note(self, key, value):
        self.results[key] = value
        logger.info(f"[pmm note] {key} = {value}")

    def dump(self):
        if not PERF_OUT:
            return
        data = json.loads(open(PERF_OUT).read()) if os.path.isfile(PERF_OUT) else {}
        data["prefill_matmul_candidates"] = {"results": self.results, "failures": self.failures}
        with open(PERF_OUT, "w") as f:
            json.dump(data, f, indent=2)


def _mm2d(cores, bw, pcm, pcn, sub_h, sub_w, transpose_mcast=False, fuse_batch=True):
    """Legacy 2D mcast config: one [pcm x pcn] output block per core, K in blocks of ``bw`` tiles."""
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*cores),
        in0_block_w=bw,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=pcm,
        out_block_w=pcn,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=transpose_mcast,
        fused_activation=None,
        fuse_batch=fuse_batch,
    )


def _mm1d(cores, bw, pcm, pcn, sub_h, sub_w, mcast_in0, fuse_batch=False):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*cores),
        in0_block_w=bw,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=pcm,
        out_block_w=pcn,
        per_core_M=pcm,
        per_core_N=pcn,
        fuse_batch=fuse_batch,
        fused_activation=None,
        mcast_in0=mcast_in0,
    )


def _bmm(cores, bw, pcm, pcn, sub_h, sub_w):
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=cores,
        in0_block_w=bw,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=pcm,
        per_core_N=pcn,
    )


def _mmc(grid, mb, kb, nb, sh, sw):
    return ttnn.MinimalMatmulConfig(
        M_block_size=mb,
        K_block_size=kb,
        N_block_size=nb,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
    )


def _matmul(a, b, pc=None, core_grid=None, compute=COMPUTE, dtype=ttnn.bfloat8_b, **kw):
    placement = {}
    if pc is not None:
        placement["program_config"] = pc
    elif core_grid is not None:
        placement["core_grid"] = core_grid
    return lambda: ttnn.matmul(
        a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dtype, compute_kernel_config=compute, **placement, **kw
    )


def _minimal(a, b, cfg=None, compute=COMPUTE, dtype=ttnn.bfloat8_b):
    return lambda: ttnn.experimental.minimal_matmul(
        a, b, config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=dtype, compute_kernel_config=compute
    )


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(1, 1)])
def test_prefill_matmul_candidates(mesh_device, device_params, reset_seeds):
    device = mesh_device
    g = torch.Generator().manual_seed(31)
    bench = Bench(device)
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    dense_grid = _dense_core_grid(device, SolarOpenProgramConfig().dense_grid_max_width)
    bench.note("compute_grid", f"{gx}x{gy}")
    bench.note("dense_grid", f"{dense_grid.x}x{dense_grid.y}")
    logger.info(f"compute grid {gx}x{gy}, dense grid {dense_grid}, sections {sorted(SECTIONS)}")

    def up(t, dtype=ttnn.bfloat8_b, mem=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=mem)

    # ------------------------------------------------------------------------------------------------------------
    # 1. hot_gu: per-expert fused gate|up linear at 1024 tokens
    # ------------------------------------------------------------------------------------------------------------
    if "hot_gu" in SECTIONS or "hot_down" in SECTIONS:
        n_pool = N_POOL
        hidden = up(torch.randn(1, 1, SPLIT, H, generator=g), dtype=ttnn.bfloat16)
        w_gu_pool = up(torch.randn(1, n_pool, H, N_GU, generator=g) * 0.02)  # n_pool expert slots
        hid_t = ttnn.to_torch(hidden).float()  # [1, 1, S, H] bf16-rounded
        wgu_t = ttnn.to_torch(w_gu_pool).float()  # [1, n, H, 2Ip] bfp8-rounded

    if "hot_gu" in SECTIONS:
        w_e = ttnn.slice(w_gu_pool, [0, 0, 0, 0], [1, 1, H, N_GU])
        ref_e = hid_t[0, 0] @ wgu_t[0, 0]  # [S, 2Ip]
        flops_e = 2.0 * SPLIT * H * N_GU
        prod = bench.time("hot_gu_auto", _matmul(hidden, w_e, core_grid=dense_grid), flops_e)
        prod_t = bench.numerics("hot_gu_auto", prod, ref_e)
        prod.deallocate(True)
        assert bench.results["hot_gu_auto"]["pcc_vs_ref"] >= MIN_PCC, bench.results["hot_gu_auto"]
        # legacy 2D configs (Mt = 32, Nt = 10, Kt = 128)
        cands = {
            "hot_gu_2d_10x8_bw8_sub4x1": _mm2d((10, 8), 8, 4, 1, 4, 1),
            "hot_gu_2d_10x8_bw16_sub4x1": _mm2d((10, 8), 16, 4, 1, 4, 1),
            "hot_gu_2d_10x8_bw32_sub4x1": _mm2d((10, 8), 32, 4, 1, 4, 1),
            "hot_gu_2d_10x8_bw16_sub1x1": _mm2d((10, 8), 16, 4, 1, 1, 1),
            "hot_gu_2d_10x8_bw16_sub2x1": _mm2d((10, 8), 16, 4, 1, 2, 1),
            "hot_gu_2d_5x8_bw16_sub4x2": _mm2d((5, 8), 16, 4, 2, 4, 2),
            "hot_gu_2d_5x8_bw32_sub4x2": _mm2d((5, 8), 32, 4, 2, 4, 2),
            "hot_gu_2d_10x4_bw16_sub4x1": _mm2d((10, 4), 16, 8, 1, 4, 1),
            "hot_gu_2d_10x4_bw16_sub8x1": _mm2d((10, 4), 16, 8, 1, 8, 1),
            "hot_gu_2d_8x10_bw16_sub4x1_tmcast": _mm2d((8, 10), 16, 4, 1, 4, 1, transpose_mcast=True),
            # 1D mcast-in1 (in0 reuse): 32 cores own one M tile row each, the [K, 10] weight is multicast
            "hot_gu_1d_in1_8x4_bw16_sub1x5": _mm1d((8, 4), 16, 1, 10, 1, 5, mcast_in0=False),
            "hot_gu_1d_in1_8x4_bw32_sub1x5": _mm1d((8, 4), 32, 1, 10, 1, 5, mcast_in0=False),
            "hot_gu_1d_in1_8x4_bw128_sub1x5": _mm1d((8, 4), 128, 1, 10, 1, 5, mcast_in0=False),
            "hot_gu_1d_in1_8x4_bw32_sub1x2": _mm1d((8, 4), 32, 1, 10, 1, 2, mcast_in0=False),
        }
        for name, pc in cands.items():
            out = bench.time(name, _matmul(hidden, w_e, pc=pc), flops_e)
            if out is not None:
                bench.numerics(name, out, ref_e, prod=prod_t)
                out.deallocate(True)
        # minimal_matmul (M > N -> the op transposes its grid: M over x, N over y)
        mcands = {
            "hot_gu_mmin_default": None,
            f"hot_gu_mmin_{gx}x{gy}_m3k8n1_s3x1": _mmc((gx, gy), 3, 8, 1, 3, 1),
            f"hot_gu_mmin_{gx}x{gy}_m3k16n1_s3x1": _mmc((gx, gy), 3, 16, 1, 3, 1),
            f"hot_gu_mmin_{gx}x{gy}_m3k32n1_s3x1": _mmc((gx, gy), 3, 32, 1, 3, 1),
            "hot_gu_mmin_8x10_m4k16n1_s4x1": _mmc((8, 10), 4, 16, 1, 4, 1),
            "hot_gu_mmin_8x10_m4k32n1_s4x1": _mmc((8, 10), 4, 32, 1, 4, 1),
            "hot_gu_mmin_8x5_m4k16n2_s4x2": _mmc((8, 5), 4, 16, 2, 4, 2),
            "hot_gu_mmin_8x5_m4k32n2_s4x2": _mmc((8, 5), 4, 32, 2, 4, 2),
            f"hot_gu_mmin_{gx}x5_m3k16n2_s3x2": _mmc((gx, 5), 3, 16, 2, 3, 2),
        }
        for name, cfg in mcands.items():
            out = bench.time(name, _minimal(hidden, w_e, cfg), flops_e)
            if out is not None:
                bench.numerics(name, out, ref_e, prod=prod_t)
                out.deallocate(True)
        # the WIRED phase-3 form: SolarOpenProgramConfig.dense_expert_gate_up_minimal exactly as _expert_linear runs it
        pc_wired = SolarOpenProgramConfig()
        gu_wired = pc_wired.get_dense_expert_gate_up_config(SPLIT, H, N_GU, grid=dense_grid)
        bench.note("hot_gu_wired_config", str(gu_wired))
        if gu_wired is not None:
            out = bench.time("hot_gu_wired", _minimal(hidden, w_e, gu_wired), flops_e)
            assert out is not None, bench.failures.get("hot_gu_wired")
            bench.numerics("hot_gu_wired", out, ref_e, prod=prod_t)
            out.deallocate(True)
            assert bench.results["hot_gu_wired"]["pcc_vs_ref"] >= MIN_PCC
            assert (
                bench.results["hot_gu_wired"]["pcc_vs_ref"]
                >= bench.results["hot_gu_auto"]["pcc_vs_ref"] - WIRED_PCC_MARGIN
            ), (bench.results["hot_gu_wired"], bench.results["hot_gu_auto"])
            # the 512-token split of > 32K contexts (16 M tiles over 11 columns: M_block 2, subblock 2x2)
            hidden_512 = ttnn.slice(hidden, [0, 0, 0, 0], [1, 1, SPLIT_LONG_CONTEXT, H])
            gu_wired_512 = pc_wired.get_dense_expert_gate_up_config(SPLIT_LONG_CONTEXT, H, N_GU, grid=dense_grid)
            bench.note("hot_gu_wired_m512_config", str(gu_wired_512))
            out = bench.time("hot_gu_wired_m512", _minimal(hidden_512, w_e, gu_wired_512), flops_e / 2)
            assert out is not None, bench.failures.get("hot_gu_wired_m512")
            bench.numerics("hot_gu_wired_m512", out, ref_e[:SPLIT_LONG_CONTEXT])
            out.deallocate(True)
            assert bench.results["hot_gu_wired_m512"]["pcc_vs_ref"] >= MIN_PCC
            out = bench.time("hot_gu_auto_m512", _matmul(hidden_512, w_e, core_grid=dense_grid), flops_e / 2)
            if out is not None:
                bench.numerics("hot_gu_auto_m512", out, ref_e[:SPLIT_LONG_CONTEXT])
                out.deallocate(True)
            hidden_512.deallocate(True)
        # fp32 destination variants (numerics: dst 4 tiles -> subblock <= 4)
        out = bench.time(
            "hot_gu_mmin_8x10_m4k16n1_s4x1_fp32dst",
            _minimal(hidden, w_e, _mmc((8, 10), 4, 16, 1, 4, 1), COMPUTE_FP32),
            flops_e,
        )
        if out is not None:
            bench.numerics("hot_gu_mmin_8x10_m4k16n1_s4x1_fp32dst", out, ref_e, prod=prod_t)
            out.deallocate(True)
        out = bench.time(
            "hot_gu_2d_10x8_bw16_sub4x1_fp32dst",
            _matmul(hidden, w_e, pc=_mm2d((10, 8), 16, 4, 1, 4, 1), compute=COMPUTE_FP32),
            flops_e,
        )
        if out is not None:
            bench.numerics("hot_gu_2d_10x8_bw16_sub4x1_fp32dst", out, ref_e, prod=prod_t)
            out.deallocate(True)
        # bfp8 in0 (numerics change: the MoE input rounded to bfp8) with the auto config
        hidden8 = ttnn.typecast(hidden, ttnn.bfloat8_b)
        out = bench.time("hot_gu_auto_in0bfp8", _matmul(hidden8, w_e, core_grid=dense_grid), flops_e)
        if out is not None:
            bench.numerics("hot_gu_auto_in0bfp8", out, ref_e, prod=prod_t)
            out.deallocate(True)
        bench.time("typecast_hidden_1024_bf16_to_bfp8", lambda: ttnn.typecast(hidden, ttnn.bfloat8_b))
        hidden8.deallocate(True)
        # two [4096, 160] linears (gate, up) vs the fused one
        w_gate = ttnn.slice(w_gu_pool, [0, 0, 0, 0], [1, 1, H, IP])
        w_up = ttnn.slice(w_gu_pool, [0, 0, 0, IP], [1, 1, H, N_GU])
        out = bench.time("hot_gate_only_auto", _matmul(hidden, w_gate, core_grid=dense_grid), flops_e / 2)
        if out is not None:
            bench.numerics("hot_gate_only_auto", out, ref_e[:, :IP])
            out.deallocate(True)
        out = bench.time(
            "hot_gate_only_mmin_8x5_m4k16n1_s4x1", _minimal(hidden, w_gate, _mmc((8, 5), 4, 16, 1, 4, 1)), flops_e / 2
        )
        if out is not None:
            bench.numerics("hot_gate_only_mmin_8x5_m4k16n1_s4x1", out, ref_e[:, :IP])
            out.deallocate(True)
        w_gate.deallocate(True)
        w_up.deallocate(True)
        w_e.deallocate(True)

        # batched hot experts in ONE launch: (a) 1D in0-reuse (no repeat), (b) N-concat of the weights
        for n in HOT_COUNTS:
            w_hot = ttnn.slice(w_gu_pool, [0, 0, 0, 0], [1, n, H, N_GU])  # [1, n, H, 2Ip] (device copy)
            ref_hot = torch.matmul(hid_t, wgu_t[:, :n])  # [1, n, S, 2Ip]
            flops_n = flops_e * n
            for bw, sw in ((16, 5), (32, 5), (128, 5)):
                name = f"hot_gu_batched{n}_1d_in1_8x4_bw{bw}_sub1x{sw}"
                out = bench.time(
                    name, _matmul(hidden, w_hot, pc=_mm1d((8, 4), bw, 1, 10, 1, sw, mcast_in0=False)), flops_n
                )
                if out is not None:
                    bench.numerics(name, out, ref_hot)
                    out.deallocate(True)
            # N-concat: [1, 1, H, n * 2Ip] weight (320-wide bfp8 pieces = 320 B rows: page-copy concat, exact)
            pieces = [ttnn.slice(w_gu_pool, [0, e, 0, 0], [1, e + 1, H, N_GU]) for e in range(n)]
            w_cat = bench.time(f"hot_gu_ncat{n}_weight_concat", lambda: ttnn.concat(pieces, dim=3))
            for p in pieces:
                p.deallocate(True)
            if w_cat is not None:
                ref_cat = ref_hot[0].permute(1, 0, 2).reshape(SPLIT, n * N_GU)  # [S, n * 2Ip]
                nt = n * N_GU // 32
                out = bench.time(f"hot_gu_ncat{n}_auto", _matmul(hidden, w_cat, core_grid=dense_grid), flops_n)
                if out is not None:
                    bench.numerics(f"hot_gu_ncat{n}_auto", out, ref_cat)
                    out.deallocate(True)
                out = bench.time(f"hot_gu_ncat{n}_mmin_default", _minimal(hidden, w_cat), flops_n)
                if out is not None:
                    bench.numerics(f"hot_gu_ncat{n}_mmin_default", out, ref_cat)
                    out.deallocate(True)
                # explicit: M over y (10 rows -> 4 tiles/core, padded 40), N over x
                nb = max(1, -(-nt // gx))
                nb_blk = next(d for d in (8, 6, 5, 4, 3, 2, 1) if nb % d == 0)
                sw = next(d for d in (2, 1) if nb_blk % d == 0)
                cfg = _mmc((gx, gy), 4, 16, nb_blk, 4, sw)
                name = f"hot_gu_ncat{n}_mmin_{gx}x{gy}_m4k16n{nb_blk}_s4x{sw}"
                out = bench.time(name, _minimal(hidden, w_cat, cfg), flops_n)
                if out is not None:
                    bench.numerics(name, out, ref_cat)
                    out.deallocate(True)
                if nt % 8 == 0:
                    pcn = nt // 8
                    sw = next(d for d in (2, 1) if pcn % d == 0)
                    name = f"hot_gu_ncat{n}_2d_8x8_bw16_pcn{pcn}_sub4x{sw}"
                    out = bench.time(name, _matmul(hidden, w_cat, pc=_mm2d((8, 8), 16, 4, pcn, 4, sw)), flops_n)
                    if out is not None:
                        bench.numerics(name, out, ref_cat)
                        out.deallocate(True)
                # the per-expert split of the N-concat output: n slices [S, 2Ip] + concat along dim 1
                try:
                    out_cat = _minimal(hidden, w_cat)()
                except Exception:  # recorded above as a failure
                    out_cat = _matmul(hidden, w_cat, core_grid=dense_grid)()

                def split_out(out_cat=out_cat, n=n):
                    parts = [ttnn.slice(out_cat, [0, 0, 0, e * N_GU], [1, 1, SPLIT, (e + 1) * N_GU]) for e in range(n)]
                    stacked = ttnn.concat(parts, dim=1)
                    for p in parts:
                        p.deallocate(True)
                    return stacked

                bench.time(f"hot_gu_ncat{n}_output_split", split_out)
                out_cat.deallocate(True)
                w_cat.deallocate(True)
            w_hot.deallocate(True)
        # per-expert reference cost: n launches of the auto linear (weight slice + linear), as production does
        for n in HOT_COUNTS:

            def per_expert(n=n):
                outs = []
                for e in range(n):
                    w = ttnn.slice(w_gu_pool, [0, e, 0, 0], [1, e + 1, H, N_GU])
                    outs.append(_matmul(hidden, w, core_grid=dense_grid)())
                    w.deallocate(True)
                cat = ttnn.concat(outs, dim=1)
                for o in outs:
                    o.deallocate(True)
                return cat

            bench.time(f"hot_gu_per_expert{n}_prod_chain", per_expert, flops_e * n, burst=2)

            def per_expert_mmin(n=n):
                outs = []
                cfg = SolarOpenProgramConfig().get_dense_expert_gate_up_config(SPLIT, H, N_GU, grid=dense_grid)
                for e in range(n):
                    w = ttnn.slice(w_gu_pool, [0, e, 0, 0], [1, e + 1, H, N_GU])
                    outs.append(
                        _minimal(hidden, w, cfg)() if cfg is not None else _matmul(hidden, w, core_grid=dense_grid)()
                    )
                    w.deallocate(True)
                cat = ttnn.concat(outs, dim=1)
                for o in outs:
                    o.deallocate(True)
                return cat

            bench.time(f"hot_gu_per_expert{n}_wired_chain", per_expert_mmin, flops_e * n, burst=2)

    # ------------------------------------------------------------------------------------------------------------
    # 2. hot_down: per-expert / batched / K-concat down at 1024 tokens
    # ------------------------------------------------------------------------------------------------------------
    if "hot_down" in SECTIONS:
        n_pool = N_POOL
        w_dn_pool = up(torch.randn(1, n_pool, IP, H, generator=g) * 0.02)
        act_pool = up(torch.randn(1, n_pool, SPLIT, IP, generator=g))  # GLU outputs (bfp8) of n_pool hot experts
        wdn_t = ttnn.to_torch(w_dn_pool).float()
        act_t = ttnn.to_torch(act_pool).float()
        flops_d = 2.0 * SPLIT * IP * H
        pc_wired = SolarOpenProgramConfig()
        # the production GLU in bf16 (_glu with dtype) on bfp8 gate / up pieces: the wired K-concat's act source
        gate_probe = ttnn.slice(act_pool, [0, 0, 0, 0], [1, 2, SPLIT, IP])
        up_probe = ttnn.slice(act_pool, [0, 2, 0, 0], [1, 4, SPLIT, IP])
        glu16 = bench.time("hot_glu_bf16_n2", lambda: _glu(gate_probe, up_probe, "silu", dtype=ttnn.bfloat16))
        glu8 = bench.time("hot_glu_bfp8_n2", lambda: _glu(gate_probe, up_probe, "silu"))
        if glu16 is not None and glu8 is not None:
            g_t, u_t = ttnn.to_torch(gate_probe).float(), ttnn.to_torch(up_probe).float()
            ref_glu = u_t * torch.nn.functional.silu(g_t)
            bench.numerics("hot_glu_bf16_n2", glu16, ref_glu)
            bench.numerics("hot_glu_bfp8_n2", glu8, ref_glu)
            assert glu16.dtype == ttnn.bfloat16 and glu8.dtype == ttnn.bfloat8_b
            glu16.deallocate(True)
            glu8.deallocate(True)
        gate_probe.deallocate(True)
        up_probe.deallocate(True)
        a_e = ttnn.slice(act_pool, [0, 0, 0, 0], [1, 1, SPLIT, IP])
        wd_e = ttnn.slice(w_dn_pool, [0, 0, 0, 0], [1, 1, IP, H])
        ref_d = act_t[0, 0] @ wdn_t[0, 0]
        prod = bench.time("hot_down_auto", _matmul(a_e, wd_e, core_grid=dense_grid), flops_d)
        prod_t = bench.numerics("hot_down_auto", prod, ref_d)
        prod.deallocate(True)
        assert bench.results["hot_down_auto"]["pcc_vs_ref"] >= MIN_PCC
        dcands = {
            "hot_down_2d_8x8_bw5_pcn16_sub4x2": _matmul(a_e, wd_e, pc=_mm2d((8, 8), 5, 4, 16, 4, 2)),
            "hot_down_2d_8x8_bw5_pcn16_sub2x4": _matmul(a_e, wd_e, pc=_mm2d((8, 8), 5, 4, 16, 2, 4)),
            "hot_down_2d_8x8_bw5_pcn16_sub1x8": _matmul(a_e, wd_e, pc=_mm2d((8, 8), 5, 4, 16, 1, 8)),
            "hot_down_mmin_default": _minimal(a_e, wd_e),
            f"hot_down_mmin_{gx}x{gy}_m4k5n4_s4x2": _minimal(a_e, wd_e, _mmc((gx, gy), 4, 5, 4, 4, 2)),
            f"hot_down_mmin_{gx}x{gy}_m4k5n12_s4x2": _minimal(a_e, wd_e, _mmc((gx, gy), 4, 5, 12, 4, 2)),
            f"hot_down_mmin_{gx}x{gy}_m4k5n6_s2x3": _minimal(a_e, wd_e, _mmc((gx, gy), 4, 5, 6, 2, 3)),
            "hot_down_mmin_8x8_m4k5n16_s4x2": _minimal(a_e, wd_e, _mmc((8, 8), 4, 5, 16, 4, 2)),
            "hot_down_mmin_8x8_m4k5n8_s4x2": _minimal(a_e, wd_e, _mmc((8, 8), 4, 5, 8, 4, 2)),
        }
        for name, fn in dcands.items():
            out = bench.time(name, fn, flops_d)
            if out is not None:
                bench.numerics(name, out, ref_d, prod=prod_t)
                out.deallocate(True)
        a_e.deallocate(True)
        wd_e.deallocate(True)
        for n in HOT_COUNTS:
            a_n = ttnn.slice(act_pool, [0, 0, 0, 0], [1, n, SPLIT, IP])
            w_n = ttnn.slice(w_dn_pool, [0, 0, 0, 0], [1, n, IP, H])
            ref_n = torch.matmul(act_t[:, :n], wdn_t[:, :n])  # [1, n, S, H]
            ref_sum = ref_n.sum(dim=1)  # [1, S, H]
            prod = bench.time(f"hot_down_batched{n}_bmm_auto", _matmul(a_n, w_n, core_grid=dense_grid), flops_d * n)
            prod_t = bench.numerics(f"hot_down_batched{n}_bmm_auto", prod, ref_n)
            bench.time(
                f"hot_down_batched{n}_fast_reduce_nc", lambda p=prod: ttnn.experimental.fast_reduce_nc(p, dims=[1])
            )
            red = ttnn.experimental.fast_reduce_nc(prod, dims=[1])
            bench.numerics(f"hot_down_batched{n}_bmm_auto_reduced", red, ref_sum)
            red.deallocate(True)
            prod.deallocate(True)
            for tag, pc in (
                ("2d_8x8_bw5_pcn16_sub4x2", _mm2d((8, 8), 5, 4, 16, 4, 2, fuse_batch=False)),
                ("2d_8x8_bw5_pcn16_sub1x8", _mm2d((8, 8), 5, 4, 16, 1, 8, fuse_batch=False)),
                ("bmm_bw5_sub4x2", _bmm(ttnn.CoreCoord(dense_grid.x, dense_grid.y), 5, 32, 128, 4, 2)),
            ):
                name = f"hot_down_batched{n}_{tag}"
                out = bench.time(name, _matmul(a_n, w_n, pc=pc), flops_d * n)
                if out is not None:
                    bench.numerics(name, out, ref_n, prod=prod_t)
                    out.deallocate(True)
            # K-concat: act [1, 1, S, n * Ip] (bf16 pieces: 320 B rows -> page copy) x [1, 1, n * Ip, H]
            parts = [ttnn.slice(a_n, [0, i, 0, 0], [1, i + 1, SPLIT, IP]) for i in range(n)]
            act_cat8 = bench.time(f"hot_down_kcat{n}_act_concat_bfp8", lambda parts=parts: ttnn.concat(parts, dim=3))
            parts16 = [ttnn.typecast(p, ttnn.bfloat16) for p in parts]
            act_cat16 = bench.time(
                f"hot_down_kcat{n}_act_concat_bf16", lambda parts16=parts16: ttnn.concat(parts16, dim=3)
            )
            for p in parts + parts16:
                p.deallocate(True)
            wparts = [ttnn.slice(w_dn_pool, [0, i, 0, 0], [1, i + 1, IP, H]) for i in range(n)]
            w_cat = ttnn.concat(wparts, dim=2)  # [1, 1, n * Ip, H] (whole tile rows: page copy)
            for p in wparts:
                p.deallocate(True)
            kt = n * IP // 32
            # the WIRED phase-3 form: bf16 act pieces + SolarOpenProgramConfig.hot_down_kconcat_minimal (M4 K5 N12 s4x2)
            kc_wired = pc_wired.get_hot_down_kconcat_config(SPLIT, n * IP, H, grid=dense_grid)
            bench.note(f"hot_down_kcat{n}_wired_config", str(kc_wired))
            if kc_wired is not None and act_cat16 is not None:
                name = f"hot_down_kcat{n}_wired"
                out = bench.time(name, _minimal(act_cat16, w_cat, kc_wired), flops_d * n)
                assert out is not None, bench.failures.get(name)
                bench.numerics(name, out, ref_sum)
                out.deallocate(True)
                assert bench.results[name]["pcc_vs_ref"] >= MIN_PCC
                assert (
                    bench.results[name]["pcc_vs_ref"]
                    >= bench.results[f"hot_down_batched{n}_bmm_auto_reduced"]["pcc_vs_ref"] - WIRED_PCC_MARGIN
                ), (bench.results[name], bench.results[f"hot_down_batched{n}_bmm_auto_reduced"])
                # the 512-token split of > 32K contexts (M_block 2, subblock 2x2)
                act_512 = ttnn.slice(act_cat16, [0, 0, 0, 0], [1, 1, SPLIT_LONG_CONTEXT, n * IP])
                kc_512 = pc_wired.get_hot_down_kconcat_config(SPLIT_LONG_CONTEXT, n * IP, H, grid=dense_grid)
                name = f"hot_down_kcat{n}_wired_m512"
                out = bench.time(name, _minimal(act_512, w_cat, kc_512), flops_d * n / 2)
                assert out is not None, bench.failures.get(name)
                bench.numerics(name, out, ref_sum[:, :SPLIT_LONG_CONTEXT])
                out.deallocate(True)
                assert bench.results[name]["pcc_vs_ref"] >= MIN_PCC
                act_512.deallocate(True)
            for act_cat, atag in ((act_cat8, "in0bfp8"), (act_cat16, "in0bf16")):
                if act_cat is None:
                    continue
                for tag, fn in (
                    (
                        f"kcat{n}_{atag}_2d_prod",
                        _matmul(act_cat, w_cat, pc=_hot_down_kconcat_config((8, 8), SPLIT, n * IP, H)),
                    ),
                    (f"kcat{n}_{atag}_2d_8x8_bw5_sub4x2", _matmul(act_cat, w_cat, pc=_mm2d((8, 8), 5, 4, 16, 4, 2))),
                    (f"kcat{n}_{atag}_mmin_default", _minimal(act_cat, w_cat)),
                    (f"kcat{n}_{atag}_mmin_8x8_m4k5n16_s4x2", _minimal(act_cat, w_cat, _mmc((8, 8), 4, 5, 16, 4, 2))),
                    (
                        f"kcat{n}_{atag}_mmin_{gx}x{gy}_m4k5n12_s4x2",
                        _minimal(act_cat, w_cat, _mmc((gx, gy), 4, 5, 12, 4, 2)),
                    ),
                    (
                        f"kcat{n}_{atag}_mmin_{gx}x{gy}_m4k{5 if kt % 15 else 15}n12_s4x2",
                        _minimal(act_cat, w_cat, _mmc((gx, gy), 4, 5 if kt % 15 else 15, 12, 4, 2)),
                    ),
                ):
                    name = f"hot_down_{tag}"
                    out = bench.time(name, fn, flops_d * n)
                    if out is not None:
                        bench.numerics(name, out, ref_sum)
                        out.deallocate(True)
                act_cat.deallocate(True)
            w_cat.deallocate(True)
            a_n.deallocate(True)
            w_n.deallocate(True)
        act_pool.deallocate(True)
        w_dn_pool.deallocate(True)

    if "hot_gu" in SECTIONS or "hot_down" in SECTIONS:
        hidden.deallocate(True)
        w_gu_pool.deallocate(True)

    # ------------------------------------------------------------------------------------------------------------
    # 3. bmm_gu / repeat: the all-expert gate|up bmm at S rows (dense 128-token path, cold experts at cap rows)
    # ------------------------------------------------------------------------------------------------------------
    if "bmm_gu" in SECTIONS or "repeat" in SECTIONS:
        w_gu = up(torch.randn(1, E, H, N_GU, generator=g) * 0.02)  # [1, E, H, 2Ip]: 178 MB
        wgu_all_t = ttnn.to_torch(w_gu).float()
    if "bmm_gu" in SECTIONS:
        grid_full = ttnn.CoreCoord(dense_grid.x, dense_grid.y)
        grid_8x8 = ttnn.CoreCoord(8, 8)
        for s in (32, 64, 96, 128):
            mt = s // 32
            x = up(torch.randn(1, 1, s, H, generator=g), dtype=ttnn.bfloat16)
            x_rep = ttnn.repeat(x, ttnn.Shape((1, E, 1, 1)))
            x_t = ttnn.to_torch(x).float()
            ref = torch.matmul(x_t, wgu_all_t)  # [1, E, s, 2Ip]
            flops = 2.0 * E * s * H * N_GU
            prod_cfg = _bmm_config(dense_grid, mt, H // 32, N_GU // 32)
            bench.note(f"bmm_gu_s{s}_prod_config", str(prod_cfg))
            prod = bench.time(f"bmm_gu_s{s}_prod", _matmul(x_rep, w_gu, pc=prod_cfg), flops)
            prod_t = bench.numerics(f"bmm_gu_s{s}_prod", prod, ref)
            prod.deallocate(True)
            assert bench.results[f"bmm_gu_s{s}_prod"]["pcc_vs_ref"] >= MIN_PCC
            variants = [
                (grid_full, 8, 1, 2),
                (grid_full, 16, 1, 2),
                (grid_full, 16, mt, 2),
                (grid_full, 16, 1, 5),
                (grid_full, 32, mt, 2),
                (grid_full, 32, 1, 5),
                (grid_full, 64, mt, 2),
                (grid_8x8, 16, mt, 2),
                (grid_8x8, 32, mt, 2),
            ]
            if mt * 2 > 8:
                variants = [v for v in variants if v[2] * v[3] <= 8]
            for gcore, bw, sh, sw in variants:
                name = f"bmm_gu_s{s}_{gcore.x}x{gcore.y}_bw{bw}_sub{sh}x{sw}"
                out = bench.time(name, _matmul(x_rep, w_gu, pc=_bmm(gcore, bw, mt, N_GU // 32, sh, sw)), flops)
                if out is not None:
                    bench.numerics(name, out, ref, prod=prod_t)
                    out.deallocate(True)
            # bfp8 in0 with the production and one tuned config (numerics: bfp8-rounded activations)
            x_rep8 = ttnn.typecast(x_rep, ttnn.bfloat8_b)
            for tag, pc in (
                ("prod", prod_cfg),
                ("bw16_sub{mt}x2".format(mt=mt), _bmm(grid_full, 16, mt, N_GU // 32, mt, 2)),
            ):
                name = f"bmm_gu_s{s}_{tag}_in0bfp8"
                out = bench.time(name, _matmul(x_rep8, w_gu, pc=pc), flops)
                if out is not None:
                    bench.numerics(name, out, ref, prod=prod_t)
                    out.deallocate(True)
            x_rep8.deallocate(True)
            x_rep.deallocate(True)
            x.deallocate(True)

    if "repeat" in SECTIONS:
        s = 128
        x = up(torch.randn(1, 1, s, H, generator=g), dtype=ttnn.bfloat16)
        x_t = ttnn.to_torch(x).float()
        ref_rep = x_t.expand(1, E, s, H)
        prod = bench.time("repeat_s128_bf16_prod", lambda: ttnn.repeat(x, ttnn.Shape((1, E, 1, 1))))
        bench.numerics("repeat_s128_bf16_prod", prod, ref_rep)
        prod.deallocate(True)
        out = bench.time(
            "repeat_s128_typecast_bfp8_then_repeat",
            lambda: ttnn.repeat(ttnn.typecast(x, ttnn.bfloat8_b), ttnn.Shape((1, E, 1, 1))),
        )
        if out is not None:
            bench.numerics("repeat_s128_typecast_bfp8_then_repeat", out, ref_rep)
            out.deallocate(True)
        ones_e = up(torch.ones(1, E, 1, 1), dtype=ttnn.bfloat16)
        out = bench.time(
            "repeat_s128_multiply_bcast_bf16", lambda: ttnn.multiply(x, ones_e, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        if out is not None:
            bench.numerics("repeat_s128_multiply_bcast_bf16", out, ref_rep)
            out.deallocate(True)
        out = bench.time(
            "repeat_s128_multiply_bcast_bfp8out",
            lambda: ttnn.multiply(x, ones_e, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b),
        )
        if out is not None:
            bench.numerics("repeat_s128_multiply_bcast_bfp8out", out, ref_rep)
            out.deallocate(True)
        x8 = ttnn.typecast(x, ttnn.bfloat8_b)
        ones8 = up(torch.ones(1, E, 1, 1))
        out = bench.time(
            "repeat_s128_multiply_bcast_in_bfp8",
            lambda: ttnn.multiply(x8, ones8, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        )
        if out is not None:
            bench.numerics("repeat_s128_multiply_bcast_in_bfp8", out, ref_rep)
            out.deallocate(True)
        # the matmul that consumes the broadcast: bf16 vs bfp8 in0 at the best bmm config (numerics of the pair)
        if "bmm_gu" in SECTIONS:
            ref = torch.matmul(x_t, wgu_all_t)
            flops = 2.0 * E * s * H * N_GU
            rep16 = ttnn.repeat(x, ttnn.Shape((1, E, 1, 1)))
            rep8 = ttnn.repeat(x8, ttnn.Shape((1, E, 1, 1)))
            cfg = _bmm(ttnn.CoreCoord(dense_grid.x, dense_grid.y), 16, 4, N_GU // 32, 4, 2)
            for tag, src in (("bf16", rep16), ("bfp8", rep8)):
                name = f"bmm_gu_s128_bw16_sub4x2_after_repeat_{tag}"
                out = bench.time(name, _matmul(src, w_gu, pc=cfg), flops)
                if out is not None:
                    bench.numerics(name, out, ref)
                    out.deallocate(True)
            rep16.deallocate(True)
            rep8.deallocate(True)
        for t in (x, x8, ones_e, ones8):
            t.deallocate(True)
    if "bmm_gu" in SECTIONS or "repeat" in SECTIONS:
        w_gu.deallocate(True)

    # ------------------------------------------------------------------------------------------------------------
    # 4. bmm_down: the all-expert down at S rows
    # ------------------------------------------------------------------------------------------------------------
    if "bmm_down" in SECTIONS:
        # the phase-2 form (one tile row per compute pass) is the reference; the wired phase-3 form is the [Mt x 2]
        # subblock of SolarOpenProgramConfig's default dense_down_max_subblock_h (bit-identical, -8 %)
        pc_prod = SolarOpenProgramConfig(dense_down_cores=(8, 8), dense_down_max_subblock_h=1)
        pc_wired = SolarOpenProgramConfig(dense_down_cores=(8, 8))
        w_down = up(torch.randn(1, E, IP, H, generator=g) * 0.02)  # 89 MB
        wd_t = ttnn.to_torch(w_down).float()
        for s in (32, 96, 128, 256):
            mt = s // 32
            act = up(torch.randn(1, E, s, IP, generator=g))
            ref = torch.matmul(ttnn.to_torch(act).float(), wd_t)  # [1, E, s, H]
            flops = 2.0 * E * s * IP * H
            cfg = pc_prod.get_dense_down_config(s, H, IP)
            bench.note(f"bmm_down_s{s}_prod_config", str(cfg))
            prod = bench.time(f"bmm_down_s{s}_prod_1d_8x8_bw5_pcn2_osw2", _matmul(act, w_down, pc=cfg), flops)
            prod_t = bench.numerics(f"bmm_down_s{s}_prod_1d_8x8_bw5_pcn2_osw2", prod, ref)
            prod.deallocate(True)
            assert bench.results[f"bmm_down_s{s}_prod_1d_8x8_bw5_pcn2_osw2"]["pcc_vs_ref"] >= MIN_PCC
            cfg_wired = pc_wired.get_dense_down_config(s, H, IP)
            bench.note(f"bmm_down_s{s}_wired_config", str(cfg_wired))
            out = bench.time(f"bmm_down_s{s}_wired", _matmul(act, w_down, pc=cfg_wired), flops)
            assert out is not None, bench.failures.get(f"bmm_down_s{s}_wired")
            bench.numerics(f"bmm_down_s{s}_wired", out, ref, prod=prod_t)
            out.deallocate(True)
            assert bench.results[f"bmm_down_s{s}_wired"]["identical_to_prod"], bench.results[f"bmm_down_s{s}_wired"]
            if s == 256:  # only the wired-vs-phase-2 pair at the dense_bmm_max_tokens edge (sub 4x2 at Mt 8)
                act.deallocate(True)
                continue
            cands = {
                f"bmm_down_s{s}_auto": _matmul(act, w_down, core_grid=dense_grid),
                f"bmm_down_s{s}_1d_8x8_bw5_pcn2_sub{mt}x2": _matmul(
                    act, w_down, pc=_mm1d((8, 8), 5, mt, 2, mt, 2, mcast_in0=True)
                ),
                f"bmm_down_s{s}_1d_8x8_bw1_pcn2_sub{mt}x2": _matmul(
                    act, w_down, pc=_mm1d((8, 8), 1, mt, 2, mt, 2, mcast_in0=True)
                ),
                f"bmm_down_s{s}_1d_4x8_bw5_pcn4_sub{mt}x2": _matmul(
                    act, w_down, pc=_mm1d((4, 8), 5, mt, 4, mt, 2, mcast_in0=True)
                ),
            }
            if mt * 4 <= 8:
                cands[f"bmm_down_s{s}_1d_4x8_bw5_pcn4_sub{mt}x4"] = _matmul(
                    act, w_down, pc=_mm1d((4, 8), 5, mt, 4, mt, 4, mcast_in0=True)
                )
            if mt >= 2:  # 2D batch-loop form: M split over rows (pcM 1), N over 8 columns
                cands[f"bmm_down_s{s}_2d_8x{mt}_bw5_pcm1_pcn16_sub1x8"] = _matmul(
                    act, w_down, pc=_mm2d((8, mt), 5, 1, 16, 1, 8, fuse_batch=False)
                )
                cands[f"bmm_down_s{s}_2d_8x{mt}_bw5_pcm1_pcn16_sub1x4"] = _matmul(
                    act, w_down, pc=_mm2d((8, mt), 5, 1, 16, 1, 4, fuse_batch=False)
                )
            if mt == 4:
                cands[f"bmm_down_s{s}_2d_8x8_bw5_pcm4_pcn16_sub4x2_batchloop"] = _matmul(
                    act, w_down, pc=_mm2d((8, 8), 5, 4, 16, 4, 2, fuse_batch=False)
                )
            if mt == 1:  # bmm with the whole N per core (L1 probe)
                cands[f"bmm_down_s{s}_bmm_bw1_pcn128_sub1x8"] = _matmul(
                    act, w_down, pc=_bmm(ttnn.CoreCoord(dense_grid.x, dense_grid.y), 1, 1, 128, 1, 8)
                )
                cands[f"bmm_down_s{s}_bmm_bw5_pcn128_sub1x8"] = _matmul(
                    act, w_down, pc=_bmm(ttnn.CoreCoord(dense_grid.x, dense_grid.y), 5, 1, 128, 1, 8)
                )
            for name, fn in cands.items():
                out = bench.time(name, fn, flops)
                if out is not None:
                    bench.numerics(name, out, ref, prod=prod_t)
                    out.deallocate(True)
            act.deallocate(True)
        w_down.deallocate(True)

    # ------------------------------------------------------------------------------------------------------------
    # 4b. dense_kcat: the 128-token dense path's down + reduce as one K-concatenated matmul
    # ------------------------------------------------------------------------------------------------------------
    if "dense_kcat" in SECTIONS:
        s = 128
        w_down = up(torch.randn(1, E, IP, H, generator=g) * 0.02)
        wd_t = ttnn.to_torch(w_down).float()
        act = up(torch.randn(1, E, s, IP, generator=g))  # GLU output x routing weights, [1, E, s, Ip] bfp8
        act_t = ttnn.to_torch(act).float()
        ref_sum = torch.matmul(act_t, wd_t).sum(dim=1)  # [1, s, H] = sum_e act_e @ W_e
        flops = 2.0 * E * s * IP * H
        # production: bmm (1D 8x8 bw5 pcn2 osw2) + fast_reduce_nc
        cfg = SolarOpenProgramConfig(dense_down_cores=(8, 8)).get_dense_down_config(s, H, IP)

        def prod_chain():
            d = _matmul(act, w_down, pc=cfg)()
            r = ttnn.experimental.fast_reduce_nc(d, dims=[1])
            d.deallocate(True)
            return r

        prod = bench.time("dense_kcat_s128_prod_bmm_plus_reduce", prod_chain, flops)
        prod_t = bench.numerics("dense_kcat_s128_prod_bmm_plus_reduce", prod, ref_sum)
        prod.deallocate(True)

        # the token-major re-layout of the activation: [1, E, s, Ip] -> [1, s, E, Ip] -> [1, 1, s, E * Ip]
        def relayout(src=act):
            p = ttnn.permute(src, (0, 2, 1, 3))
            r = ttnn.reshape(p, (1, 1, s, E * IP))
            if r.buffer_address() != p.buffer_address():
                p.deallocate(True)
            return r

        act_cat = bench.time("dense_kcat_s128_act_permute_reshape_bfp8", relayout)
        act16 = ttnn.typecast(act, ttnn.bfloat16)
        act_cat16 = bench.time("dense_kcat_s128_act_permute_reshape_bf16", lambda: relayout(act16))
        w_flat = ttnn.reshape(w_down, (1, 1, E * IP, H))  # free view (whole tile rows)
        bench.note("dense_kcat_w_flat_is_view", bool(w_flat.buffer_address() == w_down.buffer_address()))
        if act_cat is not None:
            got = ttnn.to_torch(act_cat).float().reshape(s, E * IP)
            exp = act_t[0].permute(1, 0, 2).reshape(s, E * IP)
            bench.note("dense_kcat_relayout_bit_exact", bool(torch.equal(got, exp)))
        for act_c, atag in ((act_cat, "in0bfp8"), (act_cat16, "in0bf16")):
            if act_c is None:
                continue
            for tag, fn in (
                (f"{atag}_auto", _matmul(act_c, w_flat, core_grid=dense_grid)),
                (f"{atag}_mmin_default", _minimal(act_c, w_flat)),
                (f"{atag}_mmin_{gx}x{gy}_m1k16n12_s1x4", _minimal(act_c, w_flat, _mmc((gx, gy), 1, 16, 12, 1, 4))),
                (f"{atag}_mmin_8x4_m1k32n16_s1x8", _minimal(act_c, w_flat, _mmc((8, 4), 1, 32, 16, 1, 8))),
                (
                    f"{atag}_1d_8x8_bw16_pcm4_pcn2_sub4x2",
                    _matmul(act_c, w_flat, pc=_mm1d((8, 8), 16, 4, 2, 4, 2, mcast_in0=True)),
                ),
                (
                    f"{atag}_1d_8x8_bw32_pcm4_pcn2_sub4x2",
                    _matmul(act_c, w_flat, pc=_mm1d((8, 8), 32, 4, 2, 4, 2, mcast_in0=True)),
                ),
                (
                    f"{atag}_1d_8x8_bw64_pcm4_pcn2_sub4x2",
                    _matmul(act_c, w_flat, pc=_mm1d((8, 8), 64, 4, 2, 4, 2, mcast_in0=True)),
                ),
                (f"{atag}_2d_8x4_bw16_pcm1_pcn16_sub1x8", _matmul(act_c, w_flat, pc=_mm2d((8, 4), 16, 1, 16, 1, 8))),
                (f"{atag}_2d_8x4_bw32_pcm1_pcn16_sub1x8", _matmul(act_c, w_flat, pc=_mm2d((8, 4), 32, 1, 16, 1, 8))),
            ):
                name = f"dense_kcat_s128_{tag}"
                out = bench.time(name, fn, flops)
                if out is not None:
                    bench.numerics(name, out, ref_sum, prod=prod_t)
                    out.deallocate(True)
            act_c.deallocate(True)
        act16.deallocate(True)
        act.deallocate(True)
        # the 1D in0-reuse gate|up at 128 rows: in0 [1, 1, 128, 4096] broadcast over all E experts (Mt = 4 -> 4 cores)
        w_gu2 = up(torch.randn(1, E, H, N_GU, generator=g) * 0.02)
        x = up(torch.randn(1, 1, s, H, generator=g), dtype=ttnn.bfloat16)
        ref_gu = torch.matmul(ttnn.to_torch(x).float(), ttnn.to_torch(w_gu2).float())
        name = "bmm_gu_s128_1d_in0reuse_4x1_bw32_sub1x5"
        out = bench.time(
            name,
            _matmul(x, w_gu2, pc=_mm1d((4, 1), 32, 1, 10, 1, 5, mcast_in0=False)),
            2.0 * E * s * H * N_GU,
            reps=2,
            burst=2,
        )
        if out is not None:
            bench.numerics(name, out, ref_gu)
            out.deallocate(True)
        x.deallocate(True)
        w_gu2.deallocate(True)
        w_down.deallocate(True)

    # ------------------------------------------------------------------------------------------------------------
    # 5. scatter: one-hot^T [1024, 16384] @ [16384, 4096]
    # ------------------------------------------------------------------------------------------------------------
    if "scatter" in SECTIONS:
        cap = 128
        rows = E * cap
        idx = torch.randint(0, SPLIT, (rows,), generator=g)
        onehot_t = torch.zeros(rows, SPLIT)
        onehot_t[torch.arange(rows), idx] = 1.0
        onehot = up(onehot_t.reshape(1, 1, rows, SPLIT), dtype=ttnn.bfloat16)
        down_flat = up(torch.randn(1, 1, rows, H, generator=g) * 0.05)
        df_t = ttnn.to_torch(down_flat).float()
        ref = onehot_t.t() @ df_t.reshape(rows, H)
        flops = 2.0 * SPLIT * rows * H
        prod = bench.time(
            "scatter_auto_transpose_a", _matmul(onehot, down_flat, core_grid=dense_grid, transpose_a=True), flops
        )
        prod_t = bench.numerics("scatter_auto_transpose_a", prod, ref)
        prod.deallocate(True)
        assert bench.results["scatter_auto_transpose_a"]["pcc_vs_ref"] >= MIN_PCC
        for name, pc in (
            ("scatter_2d_8x8_bw8_pcm4_pcn16_sub4x2_transpose_a", _mm2d((8, 8), 8, 4, 16, 4, 2)),
            ("scatter_2d_8x8_bw16_pcm4_pcn16_sub4x2_transpose_a", _mm2d((8, 8), 16, 4, 16, 4, 2)),
            ("scatter_2d_8x8_bw16_pcm4_pcn16_sub2x4_transpose_a", _mm2d((8, 8), 16, 4, 16, 2, 4)),
        ):
            out = bench.time(name, _matmul(onehot, down_flat, pc=pc, transpose_a=True), flops)
            if out is not None:
                bench.numerics(name, out, ref, prod=prod_t)
                out.deallocate(True)
        onehot_T = bench.time("scatter_onehot_transpose", lambda: ttnn.transpose(onehot, 2, 3))
        if onehot_T is not None:
            for name, fn in (
                ("scatter_pretransposed_auto", _matmul(onehot_T, down_flat, core_grid=dense_grid)),
                ("scatter_pretransposed_mmin_default", _minimal(onehot_T, down_flat)),
                (
                    "scatter_pretransposed_mmin_8x8_m4k16n16_s4x2",
                    _minimal(onehot_T, down_flat, _mmc((8, 8), 4, 16, 16, 4, 2)),
                ),
                (
                    f"scatter_pretransposed_mmin_{gx}x{gy}_m4k16n12_s4x2",
                    _minimal(onehot_T, down_flat, _mmc((gx, gy), 4, 16, 12, 4, 2)),
                ),
                (
                    f"scatter_pretransposed_mmin_{gx}x{gy}_m4k32n6_s2x3",
                    _minimal(onehot_T, down_flat, _mmc((gx, gy), 4, 32, 6, 2, 3)),
                ),
            ):
                out = bench.time(name, fn, flops)
                if out is not None:
                    bench.numerics(name, out, ref, prod=prod_t)
                    out.deallocate(True)
            onehot_T.deallocate(True)
        onehot.deallocate(True)
        down_flat.deallocate(True)

    bench.dump()
    logger.info(f"prefill matmul candidate failures: {json.dumps(bench.failures, indent=1)}")
    logger.info(json.dumps(bench.results, indent=1))
