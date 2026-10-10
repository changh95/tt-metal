# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused per-chip stage of the exact device sampler (Phase F, F4; ``MOTIF3_SAMPLER_LOCAL=fused``,
docs/OPTIMIZATION_PLAN.md "Phase F"; results ``logs/opt/phaseF/F4``).

Between the candidate all-gather (which gives the global max ``M``) and the normaliser / Gumbel all-gathers, the
Phase E sampler (``tt/sampling.py`` ``MotifDeviceSampler``) runs ~40 programs per step on every chip's 6880-wide
logits slice: the exact full-vocab normaliser partials (``_stage_norm``: typecast, x - M, / T, concat, exp, the
accurate fp32 row sums on 2 cores, e0 x), the counter slices, the whole Gumbel race (``_stage_gumbel``: the 47-bit hash
of every token, log1p, the scores, three single-core reductions over 215 tiles, e0 x, concat) and the device uniform
(``_stage_uniform``, 11 programs on one tile). This module computes the same three results in ONE ``generic_op``:

* ``part [1, 1, 64, 32]`` fp32: ``e0 * [s; s1]`` (the normaliser partials ``sum exp((x - M) / T)``, ``sum exp(x - M)``);
* ``q [1, 1, 96, 32]`` fp32: ``e0 * [m; gid; dw]`` (the local Gumbel winner's score, global id and raw ``x - M``);
* ``u [1, 1, 32, 1]`` fp32: the 24-bit uniform of the counters' ``c1``.

Cores (logical, on the GX x GY compute grid; 12 x 10 on Blackhole):

* PROD = rows 0 .. GY - 2 (108 cores), TPP = 2 of the 215 vocab tiles each (the last one 1): the Gumbel hash of the
  tile up to ``t = log(-log1p(-W))`` (needs only the counters), ``d = x - M``, ``a = d / T``, ``exp(a)``, ``exp(d)``
  (kept in L1 for the ACC cores), ``s = a - t``, the local max of ``s``; after the AGG's broadcast of the per-row max
  ``m``, the lowest local id and the raw ``x - M`` among the elements equal to ``m``.
* ACC = (0..7, GY - 1): the accurate fp32 row sums of ``exp(a)`` (x = 0..3) and ``exp(d)`` (x = 4..7) -- the reduce's
  exact operation order (``acc = e_0``, ``acc = add_binary_tile(acc, e_k)`` for k = 1 .. 214, per element, then
  ``sfpu_reduce``) split by FACE: core x folds face x % 4 of the 215 tiles (pulled from the producers' L1, four
  consecutive tiles' faces per ring page) with the add_binary_tile element sequence (load, load, add, store); the
  row's first core assembles the four folded faces and runs the ``sfpu_reduce`` of the op.
* AGG = (8, GY - 1): reads ``M`` and the host-written row broadcasts of ``1 / T``, ``c1``, ``c2`` and multicasts them
  (with the row broadcast of ``M`` it builds) to the producers; the uniform ``u``; the max over the producers' maxima
  (max / min only select: exact in any order), its row broadcast multicast back; min / max over the producers' id /
  raw candidates; ``q``.

Column broadcasts of ``[32, 1]`` operands are binary_ng's software broadcast on Blackhole (fp32 / int32): built by the
data movement RISCs, or written by the host where the values come from the host (``1 / T`` in ``set_params``; ``c1``,
``c2`` as rows 32..95 of the counter tensor in ``set_counters``, one copy as before). The ``e0 *`` products of ``part``
and ``q`` take operand tiles with the value in column 0 and a zero of the value's sign elsewhere: the SFPU multiply
then gives exactly the products of the column-broadcast multiply. Every arithmetic operation is the LLK call of the op
it replaces with the same operands in the same order (``sampler_local/compute.cpp``), so ``part``, ``q`` and ``u`` are
bitwise the ops chain's (``tests/unit/test_sampling.py::test_sampler_local_fused_bitwise``).

Trace safety: fresh output allocation per call; the descriptor is cached per buffer layout with a memoized program
hash; a traced call only patches the common runtime args (addresses). Semaphores are re-armed by their owners; every
spin is bounded (a lost increment gives wrong data, never a hang).
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "sampler_local"
DATAFLOW_SRC = KERNEL_DIR / "dataflow.cpp"
COMPUTE_SRC = KERNEL_DIR / "compute.cpp"
COMMON_H = KERNEL_DIR / "common.h"

TILE = 32
VC = 6880  # vocab per chip ("mesh" split)
NT = VC // TILE  # 215 vocab tiles per chip
RING = 12  # ACC ring pages (reads in batches of RING / 3)
FP32, BF16 = 4096, 2048
# CB / semaphore ids: sampler_local/common.h
CB_EA, CB_ED, CB_PM, CB_PL, CB_PD, CB_MS, CB_CF, CB_CI = range(0, 8)
CB_X, CB_GK, CB_IO, CB_D, CB_A, CB_S, CB_W, CB_W2, CB_STG = range(8, 17)
CB_RING, CB_S2, CB_AF, CB_S2B, CB_SF, CB_E0, CB_OUT = range(17, 24)
CB_GR, CB_U, CB_OFF, CB_RAW, CB_RAWI = range(24, 29)
CB_T, CB_TL = 29, 30
SEMS = tuple(range(11))  # SEM_E, SEM_PM, SEM_PL, SEM_M, SEM_RAW, SEM_AF, SEM_L0..L3, SEM_CI
ROLE_PROD, ROLE_ACC, ROLE_AGG = 0, 1, 2
NACC, AGG_X = 8, 8  # last grid row: ACC cores x = 0..7 (row sum r = x // 4, face x % 4), the AGG at x = 8
BIG = 3.0e38  # tt/sampling.py BIG
ID_SENTINEL = float(1 << 24)


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _sources_tag() -> str:
    h = hashlib.sha1()
    for p in (DATAFLOW_SRC, COMPUTE_SRC, COMMON_H):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _is_interleaved(mc) -> bool:
    return mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def plan(grid: Tuple[int, int]) -> Dict[str, int]:
    """The core plan on a ``(GX, GY)`` compute grid: producers on rows 0 .. GY - 2, TPP tiles each (every producer
    owns 1 .. TPP tiles), NPT partial tiles at the aggregator."""
    gx, gy = int(grid[0]), int(grid[1])
    np_ = gx * (gy - 1)
    if gx <= AGG_X or np_ < 1:
        raise ValueError(f"sampler_local: grid {grid} too small")
    tpp = -(-NT // np_)
    if (np_ - 1) * tpp >= NT:  # an idle producer: shrink to the producers that own a tile
        raise ValueError(f"sampler_local: {np_} producers x {tpp} tiles leaves an idle producer (grid {grid})")
    return {"GX": gx, "GY": gy, "NP": np_, "TPP": tpp, "NPT": -(-np_ // TILE)}


class FusedSamplerLocal:
    """The fused per-chip sampler stage of one mesh (module docstring). Stateless apart from the cached descriptors."""

    def __init__(self, mesh_device, *, grid=None):
        g = mesh_device.compute_with_storage_grid_size()
        gx, gy = (int(g.x), int(g.y)) if grid is None else (int(grid[0]), int(grid[1]))
        self.mesh_device = mesh_device
        self.p = plan((gx, gy))
        if self.p["TPP"] > 2:
            raise ValueError(f"sampler_local: {self.p['TPP']} tiles per producer (the plan assumes <= 2)")
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _sources_tag()
        import os

        # experiments only (dataflow.cpp EXP; 256 = per-RISC timeline in self.last_timeline)
        self.exp = int(os.environ.get("MOTIF3_SL_EXP", "0") or 0)
        self.last_timeline = None
        # logical -> NOC: separable on the worker grid (checked)
        self.colx = [self._noc(x, 0)[0] for x in range(gx)]
        self.rowy = [self._noc(0, y)[1] for y in range(gy)]
        for y in range(gy):
            for x in range(gx):
                if self._noc(x, y) != (self.colx[x], self.rowy[y]):
                    raise RuntimeError("sampler_local: the logical -> NOC map is not separable")

    def _noc(self, x, y):
        c = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return int(c.x), int(c.y)

    # ---- checks -----------------------------------------------------------------------------------------------------
    @staticmethod
    def check(x, M, inv_t, ctr, gkey, iota, chip_off) -> None:
        """``inv_t``: the ``[1, 1, 32, 32]`` row broadcast of 1 / T; ``ctr``: ``[1, 1, 96, 32]`` int32 (rows 0..31 the
        counter tile, 32..63 / 64..95 the row broadcasts of c1 / c2) -- host-written (MotifDeviceSampler)."""
        def need(ok, msg):
            if not ok:
                raise ValueError(f"sampler_local: {msg}")

        def shape(t):
            return tuple(int(v) for v in t.shape)

        need(shape(x) == (1, 1, TILE, VC) and x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT,
             f"logits must be bf16 TILE [1, 1, 32, {VC}], got {shape(x)} {x.dtype} {x.layout}")
        for name, t, dt in (("M", M, ttnn.float32), ("chip_off", chip_off, ttnn.float32)):
            need(shape(t) == (1, 1, TILE, 1) and t.dtype == dt and t.layout == ttnn.TILE_LAYOUT,
                 f"{name} must be fp32 TILE [1, 1, 32, 1], got {shape(t)} {t.dtype}")
        need(shape(inv_t) == (1, 1, TILE, TILE) and inv_t.dtype == ttnn.float32 and inv_t.layout == ttnn.TILE_LAYOUT,
             "the 1 / T broadcast must be fp32 TILE [1, 1, 32, 32]")
        need(shape(ctr) == (1, 1, 3 * TILE, TILE) and ctr.dtype == ttnn.int32 and ctr.layout == ttnn.TILE_LAYOUT,
             "the counters must be int32 TILE [1, 1, 96, 32] (raw, c1 rows, c2 rows)")
        need(shape(gkey) == (1, 1, TILE, VC) and gkey.dtype == ttnn.int32 and gkey.layout == ttnn.TILE_LAYOUT,
             "the Gumbel keys must be int32 TILE [1, 1, 32, 6880]")
        need(shape(iota) == (1, 1, TILE, VC) and iota.dtype == ttnn.float32 and iota.layout == ttnn.TILE_LAYOUT,
             "the local ids must be fp32 TILE [1, 1, 32, 6880]")
        for t in (x, M, inv_t, ctr, gkey, iota, chip_off):
            need(_is_interleaved(t.memory_config()), "operands must be interleaved")

    def supports(self, *a) -> bool:
        try:
            self.check(*a)
            return True
        except ValueError:
            return False

    # ---- program ----------------------------------------------------------------------------------------------------
    def _program(self, ts):
        key = tuple(tuple(_accessor(t)) for t in ts)
        desc = self._desc.get(key)
        if desc is None:
            p = self.p
            GX, GY, NP, TPP, NPT = p["GX"], p["GY"], p["NP"], p["TPP"], p["NPT"]
            CR, CC = ttnn.CoreRange, ttnn.CoreCoord
            r_prod = CR(CC(0, 0), CC(GX - 1, GY - 2))
            r_acc = CR(CC(0, GY - 1), CC(NACC - 1, GY - 1))
            r_agg = CR(CC(AGG_X, GY - 1), CC(AGG_X, GY - 1))
            prod, acc, agg = ttnn.CoreRangeSet({r_prod}), ttnn.CoreRangeSet({r_acc}), ttnn.CoreRangeSet({r_agg})
            union = ttnn.CoreRangeSet({r_prod, r_acc, r_agg})
            acc_agg = ttnn.CoreRangeSet({r_acc, r_agg})
            prod_agg = ttnn.CoreRangeSet({r_prod, r_agg})

            def cb(i, n, page, dt, cores):
                return ttnn.CBDescriptor(total_size=n * page, core_ranges=cores, format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])

            f32, bf, i32 = ttnn.float32, ttnn.bfloat16, ttnn.int32
            # first: the CBs written / read remotely (one L1 address on every core)
            cbs = [cb(CB_EA, TPP, FP32, f32, union), cb(CB_ED, TPP, FP32, f32, union),
                   cb(CB_PM, NPT, FP32, f32, union), cb(CB_PL, NPT, FP32, f32, union),
                   cb(CB_PD, NPT, FP32, f32, union), cb(CB_MS, 1, FP32, f32, union),
                   cb(CB_CF, 2, FP32, f32, union), cb(CB_CI, 2, FP32, i32, union)]
            cbs += [cb(CB_X, TPP, BF16, bf, prod), cb(CB_GK, TPP, FP32, i32, prod), cb(CB_IO, TPP, FP32, f32, prod),
                    cb(CB_D, TPP, FP32, f32, prod), cb(CB_A, TPP, FP32, f32, prod), cb(CB_S, TPP, FP32, f32, prod),
                    cb(CB_W, 1, FP32, f32, prod), cb(CB_W2, 2, FP32, f32, prod), cb(CB_STG, 1, FP32, f32, prod_agg),
                    cb(CB_T, TPP, FP32, f32, prod)]
            # identical lists on the 8 ACC cores (CB_AF: one address on all of them)
            cbs += [cb(CB_RING, RING, FP32, f32, acc), cb(CB_S2, 1, FP32, f32, acc), cb(CB_AF, 1, FP32, f32, acc),
                    cb(CB_S2B, 1, FP32, f32, acc), cb(CB_SF, 3, FP32, f32, acc_agg),
                    cb(CB_E0, 1, FP32, f32, acc_agg), cb(CB_OUT, 3, FP32, f32, acc_agg)]
            cbs += [cb(CB_GR, 3, FP32, f32, agg), cb(CB_U, 1, FP32, f32, agg), cb(CB_OFF, 1, FP32, f32, agg),
                    cb(CB_RAW, 1, FP32, f32, agg), cb(CB_RAWI, 1, FP32, i32, agg)]
            if self.exp & 256:
                cbs.append(cb(CB_TL, 1, 128, ttnn.uint32, union))
            sems = [ttnn.SemaphoreDescriptor(id=s_, core_ranges=union, initial_value=0) for s_ in SEMS]
            p0, p1 = self._noc(0, 0), self._noc(GX - 1, GY - 2)
            gnx, gny = self._noc(AGG_X, GY - 1)
            accs = []
            for t in ts:
                accs += _accessor(t)
            defines = [("MOTIF_SL_SRC", self._tag)]
            from ..model_config import compute_config_descriptor  # lazy

            unpack32 = [CB_EA, CB_ED, CB_PM, CB_PL, CB_PD, CB_MS, CB_CF, CB_CI, CB_GK, CB_IO, CB_D, CB_A, CB_S,
                        CB_RING, CB_AF, CB_SF, CB_E0, CB_RAWI, CB_T]
            cc = compute_config_descriptor("eltwise", fp32_unpack_cbs=unpack32, dst_full_sync_en=True)
            crt = [0] * len(ts) + self.colx + self.rowy
            kernels = []
            for role, cores in ((ROLE_PROD, prod), (ROLE_ACC, acc), (ROLE_AGG, agg)):
                for risc in (0, 1):
                    ct = [role, risc, NT, TPP, NP, GX, GY, NPT, RING, self.exp, *p0, *p1, gnx, gny] + accs
                    cfg = ttnn.WriterConfigDescriptor() if risc == 0 else ttnn.ReaderConfigDescriptor()
                    kernels.append(ttnn.KernelDescriptor(
                        kernel_source=str(DATAFLOW_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=cores, compile_time_args=ct, defines=defines, runtime_args=[],
                        common_runtime_args=list(crt), config=cfg))
                kernels.append(ttnn.KernelDescriptor(
                    kernel_source=str(COMPUTE_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores,
                    compile_time_args=[role, NT, TPP, NP, GX, NPT, f32_bits(-BIG), f32_bits(ID_SENTINEL)],
                    defines=defines, runtime_args=[], common_runtime_args=[], config=cc))
            desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hk = (tuple(accs), tuple(self.colx), tuple(self.rowy), GX, GY, self._tag, self.exp)
                hv = self._hash.get(hk)
                if hv is None:
                    hv = self._hash[hk] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        args = [t.buffer_address() for t in ts]
        for k in desc.kernels:
            if k.common_runtime_args:
                k.common_runtime_args = args + self.colx + self.rowy
        return desc

    # ---- call -------------------------------------------------------------------------------------------------------
    def __call__(self, x, M, inv_t, ctr, gkey, iota, chip_off, *, memory_config=None) -> Tuple[object, object, object]:
        """``x [1, 1, 32, 6880]`` bf16 (the logits), ``M`` / ``inv_t`` / ``chip_off`` fp32 ``[1, 1, 32, 1]``, the
        counter tile ``ctr`` int32 ``[1, 1, 32, 32]``, the Gumbel keys ``gkey`` int32 and the local ids ``iota`` fp32
        ``[1, 1, 32, 6880]`` -> ``(part [1, 1, 64, 32], q [1, 1, 96, 32], u [1, 1, 32, 1])`` fp32 TILE
        (``memory_config``, default DRAM interleaved). Consumes nothing."""
        self.check(x, M, inv_t, ctr, gkey, iota, chip_off)
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        if not _is_interleaved(mc):
            raise ValueError("sampler_local: the outputs must be interleaved")

        def alloc(shape):
            return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh_device,
                                                  mc)

        part, q, u = alloc([1, 1, 2 * TILE, TILE]), alloc([1, 1, 3 * TILE, TILE]), alloc([1, 1, TILE, 1])
        tl = None
        if self.exp & 256:  # timeline experiment: [1, 1, 2 GX GY, 16] uint32, row 2 (y GX + x) + RISC
            tl = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 2 * self.p["GX"] * self.p["GY"], 16]), ttnn.uint32,
                                                ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG)
            self.last_timeline = tl
        ts = (x, M, inv_t, ctr, gkey, iota, chip_off, part, q, u, tl if tl is not None else u)
        ttnn.generic_op(list(ts[:10]) + ([tl] if tl is not None else []), self._program(ts))
        return part, q, u

    def deallocate(self) -> None:
        self._desc.clear()


__all__ = ["FusedSamplerLocal", "NT", "plan"]
