# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decode tail of the MoE shared expert (Phase F, F3; ``MOTIF3_MOE_LOCAL=shared``, docs/OPTIMIZATION_PLAN.md
"Phase F"; results ``logs/opt/phaseF/F3``).

Phase E runs the shared expert of every MoE layer (``tt/mlp.py`` ``PolyNormMLP._rows``, B5 fused PolyNorm) as::

    xp  = pad(x, 8 -> 32 rows)          in place (fill the tile padding), 120 cores
    gu  = linear(xp, W_gu)              [1, 1, 32, 320] fp32, 10 cores
    s   = moments(gu)                   B5 kernel, 1 core
    gat = all_gather(s, "tp")           CCL
    h   = apply(gat, gu)                B5 kernel, 1 core: 5 h tiles (16 us per layer, F0)
    y   = linear(h, W_dn)               [1, 1, 32, 4096] bf16 L1, 32 cores
    out = slice(y, rows 0..T-1)         -> [1, 1, T, 4096] DRAM

This module replaces ``apply``, the down linear and the slice by ONE ``generic_op`` (and the caller drops the pad: the
gate_up linear runs on the T-row input, whose tile-padding rows only reach padding rows, every op being row-local):

* APPLY cores (5, logical column x = 8, y = 0..4; ``shared_tail/compute_apply.cpp``): the B5 apply program split
  over them. Core j < 3 computes the per-row coefficient tile of moment j, core 3 the b tile, each with the B5
  kernel's LLK sequence for that tile (TP fold, ``s D + E``, rsqrt, finite guard, exact column broadcast, fp32 pack);
  each is sent to every APPLY core. Then core j evaluates h tile j (B5's Horner chain per element and its RNE
  typecast to bf16). So every h value is bitwise the B5 kernel's; the 3 + 1 coefficient tiles and the 5 Horner tiles
  run in parallel instead of in sequence on one core. The h tiles are gathered on APPLY core 0 and sent to the DOWN
  cores in ONE multicast (concurrent multicasts into one rectangle serialize, F2).
* DOWN cores (8 x 4 at logical (0, 0); ``shared_tail/compute_down.cpp``): the stock shared down linear
  (``mlp.DECODE_MATMUL_GRIDS`` ("shared", "down"): (8, 4) cores, in0_block_w 5 = the whole K, per_core_N 4) issued
  as the bmm kernel issues it (one block, fp32 DEST, packed bf16 once). Their W_down reads start at launch; the output
  is written as T rows (T = 8 / 16 / 32; padding rows +0, written from the hardware zeros).

Measured (layer 2, real inputs, traced, TORUS_XY; ``logs/opt/phaseF/F3``) [M]: the fused program 11.3 us against
apply 16.5 + down 5.1 + slice 5.7 + pad 5.7 us run alone; the shared expert's ``forward_decode`` partial 61.9 -> 44.8 us
(8 rows) / 61.9 -> 44.7 us (16 rows).

Bitwise equal to the Phase E path on every logical row (``tests/unit/test_mlp.py::
test_mlp_device_moe_local_shared_tail``: all 51 shared experts, 8 and 16 rows, trace replays, 20 repeated calls).
Trace safety: fresh output allocation per call; the descriptor is cached per (T, buffer layouts) with a memoized
program hash; a traced call only patches the common runtime args (addresses). Semaphores are re-armed by the kernels;
every spin is bounded. Experiments only: ``MOTIF3_STAIL_EXP`` (``dataflow.cpp`` bits; 256 = per-RISC timeline).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "shared_tail"
DATAFLOW_SRC = KERNEL_DIR / "dataflow.cpp"
COMPUTE_DOWN_SRC = KERNEL_DIR / "compute_down.cpp"
APPLY_COMPUTE_SRC = KERNEL_DIR / "compute_apply.cpp"

TILE = 32
NA = 5  # h tiles = APPLY cores (160 / 32)
DGX, DGY = 8, 4  # DOWN rectangle (the stock config's grid)
AX, AY0 = 8, 0  # APPLY column
NT_OUT = 128  # 4096 / 32
NPER = NT_OUT // (DGX * DGY)
# CB indices: APPLY (B5 apply_compute.cpp's six + the coefficient out), the union cb_h, DOWN, the timeline
CB_RECV, CB_K, CB_AG, CB_AU, CB_COEF, CB_AOUT, CB_H, CB_W, CB_DOUT, CB_CO, CB_Z = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
CB_TL = 15  # dataflow.cpp (experiments only)
SEM_AGG, SEM_H, SEM_COEF = 0, 1, 2
ROWS = (8, 16, 32)  # T: the output's padding rows are written as whole 256 B face-row runs
FP32, BF16, BFP8 = 4096, 2048, 1088


def _sources_tag() -> str:
    h = hashlib.sha1()
    for p in (DATAFLOW_SRC, COMPUTE_DOWN_SRC, APPLY_COMPUTE_SRC):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _is_interleaved(mc) -> bool:
    return mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


class FusedSharedTail:
    """apply + down + row slice of the shared expert of every MoE layer of one mesh (module docstring). Stateless apart
    from the cached descriptors (the layer's tensors are runtime args).

    Args:
        mesh_device: the mesh (needs a compute grid of at least 9 x 5).
        tp: chips in the TP ring (the gathered moments hold ``tp`` tiles per moment).
    """

    def __init__(self, mesh_device, *, tp: int, exp: int = None):
        g = mesh_device.compute_with_storage_grid_size()
        if int(g.x) <= AX or int(g.y) < max(DGY, AY0 + NA):
            raise ValueError(f"fused shared tail: needs a 9 x 5 compute grid, got {(int(g.x), int(g.y))}")
        if not 1 <= int(tp) <= 8:
            raise ValueError(f"fused shared tail: TP {tp} outside [1, 8]")
        self.mesh_device = mesh_device
        self.tp = int(tp)
        import os

        # experiments only (dataflow.cpp MOTIF_STAIL_EXP; timing, wrong results except 256 = timeline)
        self.exp = int(os.environ.get("MOTIF3_STAIL_EXP", "0") or 0) if exp is None else int(exp)
        self.last_timeline = None
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _sources_tag()

    def _noc(self, x, y):
        c = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        return int(c.x), int(c.y)

    # ---- checks -----------------------------------------------------------------------------------------------------
    def check(self, gat, gu, w_down, consts, rows: int) -> None:
        def need(ok, msg):
            if not ok:
                raise ValueError(f"fused shared tail: {msg}")

        T = int(rows)
        need(T in ROWS, f"rows {T} not in {ROWS}")
        need(tuple(int(v) for v in gu.shape) == (1, 1, T, 2 * NA * TILE), f"gu must be [1, 1, {T}, 320], got {tuple(gu.shape)}")
        need(gu.dtype == ttnn.float32 and gu.layout == ttnn.TILE_LAYOUT, "gu must be fp32 TILE")
        need(tuple(int(v) for v in gat.shape) == (1, 3, TILE, TILE * self.tp), f"gat must be [1, 3, 32, {32 * self.tp}]")
        need(gat.dtype == ttnn.float32 and gat.layout == ttnn.TILE_LAYOUT, "gat must be fp32 TILE")
        need(tuple(int(v) for v in w_down.shape)[-2:] == (NA * TILE, NT_OUT * TILE), "W_down must be [160, 4096]")
        need(w_down.dtype in (ttnn.bfloat8_b,) and w_down.layout == ttnn.TILE_LAYOUT, "W_down must be bfp8 TILE")
        for name, t, shape in (("D", consts.D, (1, 3, 1, 1)), ("E", consts.E, (1, 3, 1, 1)),
                               ("b", consts.b["fp32"], (1, 1, 1, 1))):
            need(t.dtype == ttnn.float32 and t.layout == ttnn.TILE_LAYOUT and tuple(int(v) for v in t.shape) == shape,
                 f"constant {name} must be fp32 TILE {shape}")
        for t in (gat, gu, w_down, consts.D, consts.E, consts.b["fp32"]):
            need(_is_interleaved(t.memory_config()), "operands must be interleaved")

    def supports(self, *a, **k) -> bool:
        try:
            self.check(*a, **k)
            return True
        except ValueError:
            return False

    # ---- program ----------------------------------------------------------------------------------------------------
    def _program(self, T, ts):
        key = (T,) + tuple(tuple(_accessor(t)) for t in ts)
        desc = self._desc.get(key)
        if desc is None:
            ra = ttnn.CoreRange(ttnn.CoreCoord(AX, AY0), ttnn.CoreCoord(AX, AY0 + NA - 1))
            rd = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DGX - 1, DGY - 1))
            apply_cores, down_cores, union = ttnn.CoreRangeSet({ra}), ttnn.CoreRangeSet({rd}), ttnn.CoreRangeSet({ra, rd})

            def cb(i, n, page, dt, cores):
                return ttnn.CBDescriptor(total_size=n * page, core_ranges=cores, format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])

            f32, bf = ttnn.float32, ttnn.bfloat16
            cbs = [cb(CB_H, NA, BF16, bf, union)]  # first: the same L1 address on every core (multicast target)
            cbs += [cb(CB_COEF, 4, FP32, f32, apply_cores),  # remotely written: same address on every APPLY core
                    cb(CB_RECV, self.tp, FP32, f32, apply_cores), cb(CB_K, 7, FP32, f32, apply_cores),
                    cb(CB_AG, 1, FP32, f32, apply_cores), cb(CB_AU, 1, FP32, f32, apply_cores),
                    cb(CB_CO, 1, FP32, f32, apply_cores), cb(CB_AOUT, 2, BF16, bf, apply_cores)]
            cbs += [cb(CB_W, NA * NPER, BFP8, ttnn.bfloat8_b, down_cores), cb(CB_DOUT, NPER, BF16, bf, down_cores),
                    cb(CB_Z, 1, BF16, bf, down_cores)]
            if self.exp & 256:
                cbs.append(cb(CB_TL, 1, BF16, bf, union))
            sems = [ttnn.SemaphoreDescriptor(id=s_, core_ranges=union, initial_value=0)
                    for s_ in (SEM_AGG, SEM_H, SEM_COEF)]
            acx = self._noc(AX, AY0)[0]
            acy = [self._noc(AX, AY0 + j)[1] for j in range(NA)]
            if any(self._noc(AX, AY0 + j)[0] != acx for j in range(NA)):
                raise RuntimeError("fused shared tail: the APPLY column is not one NOC column")
            a0 = self._noc(AX, AY0)
            d0, d1 = self._noc(0, 0), self._noc(DGX - 1, DGY - 1)
            acc = []
            for t in ts:
                acc += _accessor(t)
            defines = [("MOTIF_STAIL_SRC", self._tag)]
            if self.exp:
                defines.append(("MOTIF_STAIL_EXP", str(self.exp)))
            from ..model_config import compute_config_descriptor  # lazy

            apply_cc = compute_config_descriptor("polynorm", fp32_unpack_cbs=[CB_RECV, CB_K, CB_AG, CB_AU, CB_COEF],
                                                 dst_full_sync_en=True)
            down_cc = compute_config_descriptor("shared")
            kernels = []
            for role, cores in ((0, apply_cores), (1, down_cores)):
                for risc in (0, 1):
                    ct = [role, risc, T, NA, NA, self.tp, NPER, DGX, AX, AY0, *a0, *d0, *d1, DGX * DGY, NT_OUT,
                          CB_RECV, CB_K, CB_AG, CB_AU, CB_AOUT, CB_H, CB_W, CB_DOUT, SEM_AGG, SEM_H,
                          CB_COEF, CB_CO, SEM_COEF, CB_Z, acx, *acy] + acc
                    cfg = ttnn.WriterConfigDescriptor() if risc == 0 else ttnn.ReaderConfigDescriptor()
                    kernels.append(ttnn.KernelDescriptor(
                        kernel_source=str(DATAFLOW_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=cores, compile_time_args=ct, defines=defines, runtime_args=[],
                        common_runtime_args=[0] * len(ts), config=cfg))
            kernels.append(ttnn.KernelDescriptor(
                kernel_source=str(APPLY_COMPUTE_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=apply_cores,
                compile_time_args=[CB_RECV, CB_K, CB_AG, CB_AU, CB_COEF, CB_AOUT, self.tp, CB_CO, AY0],
                defines=defines, runtime_args=[], common_runtime_args=[], config=apply_cc))
            kernels.append(ttnn.KernelDescriptor(
                kernel_source=str(COMPUTE_DOWN_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=down_cores, compile_time_args=[CB_H, CB_W, CB_DOUT, NA, NPER], defines=defines,
                runtime_args=[], common_runtime_args=[], config=down_cc))
            desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hk = (T, tuple(acc), a0, d0, d1, self.tp, self._tag, self.exp)
                hv = self._hash.get(hk)
                if hv is None:
                    hv = self._hash[hk] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        args = [t.buffer_address() for t in ts]
        for k in desc.kernels:
            if k.common_runtime_args:
                k.common_runtime_args = args
        return desc

    # ---- call -------------------------------------------------------------------------------------------------------
    def __call__(self, gat, gu, w_down, consts, *, rows: int, memory_config=None):
        """``gat [1, 3, 32, 32 TP]`` (the gathered B5 moments), ``gu [1, 1, T, 320]`` fp32 (gate | up of the T-row
        input), ``W_down [1, 1, 160, 4096]`` bfp8, the layer's ``ScalarPolyNormConsts`` -> ``y [1, 1, T, 4096]`` bf16
        (``memory_config``, default DRAM interleaved): the shared expert's TP partial. Consumes nothing."""
        T = int(rows)
        self.check(gat, gu, w_down, consts, T)
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        if not _is_interleaved(mc):
            raise ValueError("fused shared tail: the output must be interleaved")
        y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, T, NT_OUT * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                           self.mesh_device, mc)
        tl = None
        if self.exp & 256:  # timeline experiment: [1, 1, 2 (ND + NA), 16] uint32, row 2 c + RISC
            tl = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 2 * (DGX * DGY + NA), 16]), ttnn.uint32,
                                                ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG)
            self.last_timeline = tl
        ts = (gat, gu, consts.D, consts.E, consts.b["fp32"], w_down, y, tl if tl is not None else y)
        ttnn.generic_op(list(ts[:7]) + ([tl] if tl is not None else []), self._program(T, ts))
        return y

    def deallocate(self) -> None:
        self._desc.clear()


__all__ = ["FusedSharedTail", "NA", "NPER"]
