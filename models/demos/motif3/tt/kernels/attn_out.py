# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decode attention output chain (Phase F, F2; ``MOTIF3_ATTN_OUT``; docs/OPTIMIZATION_PLAN.md "Phase F").

After FlashMLA, :meth:`MotifAttention.forward_decode` ran (logs/opt/phaseF/F0, per layer at 32 lanes)::

    o_heads = transpose(o_lat, 1, 2)                 [1, L, 10, 512] -> [1, 10, L, 512]       4.2 us, 120 cores
    u       = o_heads @ W_UV (per head)              reuse bmm, 10 cores, K 16 (in0_block_w 4) 7.9 us
    v       = sigmoid(lam @ E)                       1D linear + a separate unary sigmoid    2 x 1.5 us
    dg      = attn_combine(u, v, g, active)          D1 generic_op                            4.0 us
    part    = dg @ wo                                1D linear (auto config), 64 cores        23.4 us
    out     = all_reduce(tp)(part)                   (F5: kept)

plus about 0.55 us of dispatch gap per program. This module runs them as ONE ``ttnn.generic_op`` over the 12 x 10 grid (``attn_out/{dataflow,compute}.cpp``):

* UV cores (x < 8, y < 5; unit u = 8 y + x = (head u / 4, column tile u % 4)): one W_UV output tile each. o_heads[h]
  (rows h of the L lane tiles of o_lat; the transpose becomes addressing) is gathered in two levels: each core reads
  a share of the 320 B blocks (rows 0..9 of one face of an o_lat tile: one DRAM read serves the 10 heads) and writes
  each head's 32 B row to that head's core of the same quarter; then the 4 cores of a head swap their quarters. The
  tile is computed with the stock reuse bmm's block structure (K 16 in blocks of 4, the fp32 partial reloaded through
  SrcA as TF32 between blocks, as the stock op does).
* CMB cores (x 8..11, y < 8; combine tile j = 4 y + x - 8): v = sigmoid(lam @ E_j) and the D1 combine (addcmul,
  multiply, where) with D1's configuration (fp32 DEST off), the u tiles written into their L1 by the UV cores.
* Stage "uv" writes dg ``[1, 1, L, 1024]`` (the stock wo linear follows). Stage "fused" also runs wo: WO cores (x < 8,
  y < 8; w = 8 y + x) stream their 2 weight columns (128 KB), receive dg in ONE 64 KB multicast from an aggregator
  CMB core (32 concurrent multicasts into one rectangle serialize: ~75 us measured), and accumulate K = 32 in order in
  the fp32 DEST (the stock 1D op's fp32 partial reload is lossless) -> ``part [1, 1, L, 4096]``.
* NoC split: NoC 0 (RISC 0) carries the W_UV reads and most of the wo stream; NoC 1 (RISC 1) carries the dependency
  chain (gather, row writes, quarter swap, u tiles, CMB inputs, the dg aggregation) and 2 of the 8 wo chunks, started
  late (``WO_DELAY_NS`` / ``WO_LATE_NS`` / ``WO_N1``): a weight stream on the same NoC delays the chain's small
  messages by several us, and one issued at launch delays the W_UV reads at DRAM.

The mHC post-mix runs after the AR(tp) on the reduced output, so it stays a separate program (fusing it needs F5).

Every arithmetic stage issues the LLK sequence of the op it replaces with that op's configuration and CB formats, so
the outputs are expected to be bitwise equal to the op chain (checked by ``tests/unit/test_attn_out.py``). Trace
safety: fresh outputs per call; one descriptor per (stage, rows, buffer layouts) with a memoized program hash; a traced
call only patches the common runtime args (addresses).
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "attn_out"
SOURCES = {name: KERNEL_DIR / f"{name}.cpp" for name in ("dataflow", "compute")}
HEADER = KERNEL_DIR / "common.h"

TILE = 32
BF16, FP32 = 2048, 4096
GRID = (12, 10)
# CB ids (attn_out/common.h)
(CB_SIG, CB_NOISE, CB_DGIN, CB_IN0, CB_STAGE, CB_WA, CB_WB, CB_PART, CB_U, CB_LAM, CB_E, CB_VPRE, CB_V, CB_G, CB_ACT,
 CB_D, CB_DG, CB_OUTC, CB_VDBG, CB_WOA, CB_WOB, CB_WOUT, CB_TL, CB_WOC) = range(24)
N_SEMS = 5  # attn_out/common.h SEM_SIG .. SEM_L1
H, SG, TPH, KUV, NLAMT, KWO, NWO_T, NWO_PER = 10, 8, 4, 16, 2, 32, 128, 2
NCMB = SG * TPH  # 32 combine tiles = 1024 / 32
UV_ROWS, WO_ROWS, LEFT_X, CMB_X0 = 5, 8, 8, 8
STAGES = ("uv", "fused")
# wo weight stream schedule (timing only; the values never depend on it). Measured (logs/opt/phaseF/F2, bench*.json):
# a stream issued at launch starves the chain's W_UV reads at DRAM and its L1 messages on the NoC, so the WO-only
# cores start at WO_DELAY_NS (NoC 0) / WO_LATE_NS (NoC 1), the UV cores once their W_UV tiles landed (NoC 0) / after
# their u tile left (NoC 1); WO_N1 = chunks (of 8) on NoC 1 of the UV / WO-only cores.
WO_DELAY_NS = 3000
WO_LATE_NS = 3000
WO_N1 = (2, 2)
VALUE_BITS = struct.unpack("<I", struct.pack("<f", -1.0))[0]  # addcmul value=-1.0 as the ternary op packs it


def _sources_tag() -> str:
    h = hashlib.sha1()
    for p in list(SOURCES.values()) + [HEADER]:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _is_interleaved(mc) -> bool:
    return mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def mm_compute_config():
    """The attn_heads role (W_UV bmm, wo linear): HiFi4, fp32 DEST, approx off, half-sync DEST, default unpack (the
    reuse bmm leaves its partials CB UnpackToSrc: the TF32 reload is part of its arithmetic)."""
    from ..model_config import NUM_CB_SLOTS  # lazy: keeps this module's import light

    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = True
    cc.math_approx_mode = False
    cc.dst_full_sync_en = False
    cc.unpack_to_dest_mode = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    return cc


def combine_compute_config():
    """D1's combine configuration (``attn_combine._compute_config``): HiFi4, fp32 DEST off, approx off, half-sync DEST;
    ``UnpackToDestFp32`` on the binary_ng SFPU operands (d, g, active, dg)."""
    from ..model_config import NUM_CB_SLOTS

    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = False
    cc.math_approx_mode = False
    cc.dst_full_sync_en = False
    modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    for cb in (CB_D, CB_G, CB_ACT, CB_DG):
        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = modes
    return cc


class FusedAttnOut:
    """The fused decode attention output chain of every attention layer of one mesh (module docstring). Stateless
    apart from the cached program descriptors (the layer's weights / activations are runtime args).

    Args:
        mesh_device: the mesh (12 x 10 compute grid).
        debug: 1 = also write u ``[1, 10, L, 128]``, v ``[1, 1, L, 1024]`` and (stage "fused") dg to debug tensors.
    """

    def __init__(self, mesh_device, *, debug: int = 0, exp: Optional[int] = None, wo_delay_ns: Optional[int] = None,
                 wo_late_ns: Optional[int] = None, wo_n1: Optional[Tuple[int, int]] = None):
        g = mesh_device.compute_with_storage_grid_size()
        if (int(g.x), int(g.y)) != GRID:
            raise ValueError(f"fused attention output: needs the {GRID} compute grid, got {(int(g.x), int(g.y))}")
        self.mesh_device = mesh_device
        self.debug = int(debug)
        import os

        # experiments only (kernel-cost breakdown; wrong results): attn_out/dataflow.cpp MOTIF_AOUT_EXP bits
        self.exp = int(os.environ.get("MOTIF3_AOUT_EXP", "0") or 0) if exp is None else int(exp)
        # the wo weight stream's start delay (stage "fused"): the chain's reads reach DRAM first (timing only)
        self.wo_delay_ns = int(os.environ.get("MOTIF3_AOUT_WO_DELAY_NS", "") or WO_DELAY_NS) if wo_delay_ns is None \
            else int(wo_delay_ns)
        self.wo_late_ns = int(os.environ.get("MOTIF3_AOUT_WO_LATE_NS", "") or WO_LATE_NS) if wo_late_ns is None \
            else int(wo_late_ns)
        n1 = os.environ.get("MOTIF3_AOUT_WO_N1", "")
        self.wo_n1 = tuple(int(v) for v in n1.split(",")) if n1 and wo_n1 is None else tuple(wo_n1 or WO_N1)
        if len(self.wo_n1) != 2 or any(v not in (0, 2, 4) for v in self.wo_n1):
            raise ValueError(f"fused attention output: wo_n1 must be two of (0, 2, 4), got {self.wo_n1}")
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _sources_tag()
        self._xy = None

    # ---- geometry ---------------------------------------------------------------------------------------------------
    def _core_xy(self):
        if self._xy is None:
            gx, gy = GRID
            xy = []
            for c in range(gx * gy):
                v = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(c % gx, c // gx))
                xy.append((int(v.x) << 16) | int(v.y))
            self._xy = xy
        return self._xy

    def _wo_rect_noc(self):
        a = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        b = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(LEFT_X - 1, WO_ROWS - 1))
        return int(a.x), int(a.y), int(b.x), int(b.y)

    # ---- checks -----------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_t(name, t, shape):
        if tuple(int(v) for v in t.shape) != tuple(shape):
            raise ValueError(f"fused attention output: {name} must be {tuple(shape)}, got {tuple(t.shape)}")
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or not _is_interleaved(t.memory_config()):
            raise ValueError(f"fused attention output: {name} must be bf16 TILE interleaved")

    def check(self, o_lat, g, lam, active, w_uv, lam_expand, w_o=None) -> int:
        """Raises ``ValueError`` unless the operands have the call contract; returns the rows ``L``."""
        shp = tuple(int(v) for v in o_lat.shape)
        if len(shp) != 4 or shp[0] != 1 or shp[2:] != (H, 16 * TILE) or not 1 <= shp[1] <= TILE:
            raise ValueError(f"fused attention output: o_lat must be [1, L <= 32, {H}, 512], got {shp}")
        L = shp[1]
        self._check_t("o_lat", o_lat, shp)
        self._check_t("g", g, (1, 1, L, NCMB * TILE))
        self._check_t("lam", lam, (1, 1, L, NLAMT * TILE))
        self._check_t("active", active, (1, 1, L, NCMB * TILE))
        self._check_t("w_uv", w_uv, (1, H, KUV * TILE, TPH * TILE))
        if tuple(int(v) for v in lam_expand.shape)[-2:] != (NLAMT * TILE, NCMB * TILE):
            raise ValueError(f"fused attention output: lam_expand must be [64, 1024], got {tuple(lam_expand.shape)}")
        self._check_t("lam_expand", lam_expand, tuple(int(v) for v in lam_expand.shape))
        if w_o is not None:
            if tuple(int(v) for v in w_o.shape)[-2:] != (KWO * TILE, NWO_T * TILE):
                raise ValueError(f"fused attention output: w_o must be [1024, 4096], got {tuple(w_o.shape)}")
            self._check_t("w_o", w_o, tuple(int(v) for v in w_o.shape))
        return L

    def supports(self, *args, **kw) -> bool:
        try:
            self.check(*args, **kw)
            return True
        except ValueError:
            return False

    # ---- program ----------------------------------------------------------------------------------------------------
    def _program(self, stage, L, ts):
        """``ts`` = (o_lat, w_uv, lam, E, g, active, w_o, out, dbg_u, dbg_v, dbg_dg, tl)."""
        fused = 1 if stage == "fused" else 0
        key = (stage, L) + tuple(tuple(_accessor(t)) for t in ts)
        desc = self._desc.get(key)
        gx, gy = GRID
        if desc is None:
            left_rows = WO_ROWS if fused else UV_ROWS
            union = _rect(0, 0, gx - 1, WO_ROWS - 1)  # left block + CMB block: the remotely written CBs live here
            left = _rect(0, 0, LEFT_X - 1, left_rows - 1)
            cmb = _rect(CMB_X0, 0, gx - 1, WO_ROWS - 1)

            def cb(i, n, page, dt, cores):
                return ttnn.CBDescriptor(total_size=n * page, core_ranges=cores, format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)])

            bf, f32 = ttnn.bfloat16, ttnn.float32
            cbs = [cb(CB_SIG, 1, BF16, bf, union), cb(CB_NOISE, 1, BF16, bf, union),
                   cb(CB_DGIN, KWO if fused else 1, BF16, bf, union), cb(CB_IN0, KUV, BF16, bf, union),
                   cb(CB_TL, 1, BF16, bf, union)]
            cbs += [cb(CB_STAGE, 9, BF16, bf, left), cb(CB_WA, KUV // 2, BF16, bf, left),
                    cb(CB_WB, KUV // 2, BF16, bf, left), cb(CB_PART, 1, FP32, f32, left), cb(CB_U, 1, BF16, bf, left)]
            if fused:
                cbs += [cb(CB_WOA, KWO, BF16, bf, left), cb(CB_WOB, KWO // 2, BF16, bf, left),
                        cb(CB_WOC, KWO // 2, BF16, bf, left), cb(CB_WOUT, NWO_PER, BF16, bf, left)]
            cbs += [cb(CB_LAM, NLAMT, BF16, bf, cmb), cb(CB_E, NLAMT, BF16, bf, cmb), cb(CB_VPRE, 1, BF16, bf, cmb),
                    cb(CB_V, 1, BF16, bf, cmb), cb(CB_G, 1, BF16, bf, cmb), cb(CB_ACT, 1, BF16, bf, cmb),
                    cb(CB_D, 1, BF16, bf, cmb), cb(CB_DG, 1, BF16, bf, cmb), cb(CB_OUTC, 1, BF16, bf, cmb),
                    cb(CB_VDBG, 1, BF16, bf, cmb)]
            sems = [ttnn.SemaphoreDescriptor(id=s_, core_ranges=union, initial_value=0) for s_ in range(N_SEMS)]
            wm = self._wo_rect_noc()
            acc = []
            for t in ts:
                acc += _accessor(t)
            defines = [("MOTIF_AOUT_SRC", self._tag)]
            if self.exp:
                defines.append(("MOTIF_AOUT_EXP", str(self.exp)))
            n_common = len(ts) + gx * gy
            kernels = []
            for role, cores, ccfg in ((0, left, mm_compute_config()), (1, cmb, combine_compute_config())):
                for risc in (0, 1):
                    ct = [role, risc, L, fused, self.debug, *wm, LEFT_X * WO_ROWS, gx,
                          (self.wo_delay_ns * 27) // 20, (self.wo_late_ns * 27) // 20, *self.wo_n1] + acc
                    cfg = ttnn.ReaderConfigDescriptor() if risc == 0 else ttnn.WriterConfigDescriptor()
                    kernels.append(ttnn.KernelDescriptor(
                        kernel_source=str(SOURCES["dataflow"]), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=cores, compile_time_args=ct, defines=defines, runtime_args=[],
                        common_runtime_args=[0] * n_common, config=cfg))
                kernels.append(ttnn.KernelDescriptor(
                    kernel_source=str(SOURCES["compute"]), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores, compile_time_args=[role, fused, self.debug, VALUE_BITS], defines=defines,
                    runtime_args=[], common_runtime_args=[], config=ccfg))
            desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hk = (stage, L, tuple(acc), self.debug, wm, self.exp, self.wo_delay_ns, self.wo_late_ns, self.wo_n1, self._tag)
                hv = self._hash.get(hk)
                if hv is None:
                    hv = self._hash[hk] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        args = [t.buffer_address() for t in ts] + self._core_xy()
        for k in desc.kernels:
            if k.common_runtime_args:
                k.common_runtime_args = args
        return desc

    # ---- call -------------------------------------------------------------------------------------------------------
    def _alloc(self, shape):
        return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh_device,
                                              ttnn.DRAM_MEMORY_CONFIG)

    def __call__(self, stage, o_lat, g, lam, active, w_uv, lam_expand, w_o=None):
        """Stage "uv": the wo input ``dg = where(active, (u_sig - sigmoid(lam @ E) u_noise) g, 0)`` ``[1, 1, L,
        1024]`` from ``o_lat [1, L, 10, 512]`` (FlashMLA's output); stage "fused": ``dg @ wo`` ``[1, 1, L, 4096]`` (the
        AR(tp) input). bf16, DRAM interleaved. Consumes nothing. With ``debug`` the result is a tuple ``(out, u, v,
        dg)`` (``dg`` is ``out`` in stage "uv")."""
        if stage not in STAGES:
            raise ValueError(f"fused attention output: stage {stage!r} not in {STAGES}")
        fused = stage == "fused"
        if fused and w_o is None:
            raise ValueError("fused attention output: stage 'fused' needs w_o")
        L = self.check(o_lat, g, lam, active, w_uv, lam_expand, w_o if fused else None)
        out = self._alloc([1, 1, L, (NWO_T if fused else NCMB) * TILE])
        dbg = []
        if self.debug:
            dbg = [self._alloc([1, H, L, TPH * TILE]), self._alloc([1, 1, L, NCMB * TILE])]
            dbg.append(self._alloc([1, 1, L, NCMB * TILE]) if fused else out)
        du, dv, dd = (dbg if dbg else (out, out, out))
        wo = w_o if fused else w_uv
        tl = None
        if self.exp & 256:  # timeline experiment: [1, 1, 240, 16] uint32, row 2 c + RISC
            tl = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 2 * GRID[0] * GRID[1], 16]), ttnn.uint32,
                                                ttnn.ROW_MAJOR_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG)
            self.last_timeline = tl
        ts = (o_lat, w_uv, lam, lam_expand, g, active, wo, out, du, dv, dd, tl if tl is not None else out)
        desc = self._program(stage, L, ts)
        ins = [o_lat, w_uv, lam, lam_expand, g, active] + ([w_o] if fused else [])
        outs = [out] + [t for t in dbg if t is not out] + ([tl] if tl is not None else [])
        ttnn.generic_op(ins + outs, desc)
        if self.debug:
            return (out, du, dv, dd)
        return out

    def deallocate(self) -> None:
        self._desc.clear()


__all__ = ["FusedAttnOut", "STAGES", "mm_compute_config", "combine_compute_config"]
