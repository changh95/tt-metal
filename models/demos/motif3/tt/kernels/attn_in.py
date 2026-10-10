# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decode attention input chain (Phase F, F1; ``MOTIF3_ATTN_IN=fused``; docs/OPTIMIZATION_PLAN.md "Phase F").

After the two latent projections, :meth:`MotifAttention.forward_decode` ran 15 programs per layer on one tile row of
lanes (logs/opt/phaseF/F0)::

    cq_n = rms_norm(cq)                                   fp32 [T, 1024], 1 core
    q    = cq_n @ Wq_b          g = sigmoid(cq_n @ Wg)    1D-mcast linears (+ a separate unary sigmoid)
    c_raw, kpe, lam = split(kvl);  n = rms_norm(c_raw)    2 x nlp_create_q_heads_split, 1-core rms_norm
    q_nope, q_pe = split(q)                               nlp_create_q_heads_split
    q_lat = q_nope @ W_UK (per head)                      reuse bmm
    q_pe_r = rope(q_pe);  k_pe = rope(kpe)                2 x rotary_embedding_hf
    q_mla = transpose(concat(q_lat, q_pe_r))              concat + transpose -> [1, T, 10, 576]
    kv    = concat(n, k_pe) -> transpose to the update shards (draft-1 row write) | kv_row (writers)

This module runs them as ONE ``ttnn.generic_op`` over the 12 x 10 grid (``attn_in/{dataflow,compute}.cpp``):

* QN core (11, 9): rms_norm(cq) (the stock RMSNORM LLK sequence), multicast block by block (4 fp32 tiles) to the QB
  cores while it is produced.
* QB cores 0..91 (row-major): one output tile each of q_b (60) or the gate (32). Each streams its 64 KB weight column
  on both RISCs / NoCs (K half per RISC) while the norm runs, then accumulates K = 32 in order in the fp32 DEST (the
  stock op's lossless fp32 spill / reload between in0 blocks gives the same sums). Gate tiles get the stock bf16
  sigmoid; pe tiles exchange their pre-RoPE tile with the head's other pe core and apply rotary_embedding_hf; nope
  tiles go to the head's 8 UK cores.
* UK cores 0..79: 2 W_UK output tiles of head ``c // 8`` each.
* Owner cores 92..109: column ``col`` of q_mla; producers write their rows (lane l of head h -> row h of lane tile l)
  straight into the owner's L1, the owner writes the L lane tiles to DRAM. The transpose / concat become addressing.
* KN core (10, 9): rms_norm(c_raw), rope(kpe), lam; kv_row to DRAM, or (draft-1 row write) row l of every kv tile
  into lane l's shard of the height-sharded ``paged_update_cache`` input.

Every arithmetic stage issues the LLK sequence of the op it replaces with that op's configuration and CB formats
(``attn_in/compute.cpp``), so the outputs are expected to be bitwise equal to the op chain (checked by
``tests/unit/test_attn_in.py``). Trace safety: fresh outputs per call; one descriptor per (rows, kv mode, buffer
layouts) with a memoized program hash; a traced call only patches the common runtime args (addresses).
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "attn_in"
SOURCES = {name: KERNEL_DIR / f"{name}.cpp" for name in ("dataflow", "compute")}
HEADER = KERNEL_DIR / "common.h"

TILE = 32
BF16, FP32 = 2048, 4096
GRID = (12, 10)
QN_CORE, KN_CORE = (11, 9), (10, 9)
# CB ids (attn_in/common.h)
(CB_CQN, CB_WA, CB_WB, CB_QOUT, CB_QSEND, CB_GOUT, CB_NOPE, CB_WUK, CB_UKOUT, CB_PEPART, CB_COS, CB_SIN, CB_SCAL,
 CB_ROT, CB_CI, CB_SI, CB_ROPE, CB_QMLA, CB_KVROW, CB_QX, CB_XMM2, CB_EX2, CB_EX2PE, CB_RSCAL, CB_EPS, CB_KIN,
 CB_ROTIN, CB_KPEX, CB_LAM) = range(29)
N_SEMS = 5
H, HT, NQ, NG, KQ, UKN, NCOL, KVT = 10, 6, 60, 32, 32, 16, 18, 18
NQB, NUK, OWN0 = NQ + NG, 80, 92
KV_MODES = ("dram", "shard")


def _sources_tag() -> str:
    h = hashlib.sha1()
    for p in list(SOURCES.values()) + [HEADER]:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _is_interleaved(mc) -> bool:
    return mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


def compute_config():
    """HiFi4, fp32 DEST, approx off, half-sync DEST, default unpack (the norm / linear / rope roles of the chain)."""
    from ..model_config import NUM_CB_SLOTS  # lazy: keeps this module's import light

    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = True
    cc.math_approx_mode = False
    cc.dst_full_sync_en = False
    cc.unpack_to_dest_mode = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    return cc


class FusedAttnIn:
    """The fused decode attention input chain of every attention layer of one mesh (module docstring). Stateless apart
    from the cached program descriptors (the layer's weights / activations are runtime args).

    Args:
        mesh_device: the mesh.
        eps: the rms_norm epsilon (``cfg.rms_norm_eps``).
        debug: 1 = also write cq_n to a debug tensor (diagnostics).
    """

    def __init__(self, mesh_device, *, eps: float, debug: int = 0):
        g = mesh_device.compute_with_storage_grid_size()
        if (int(g.x), int(g.y)) != GRID:
            raise ValueError(f"fused attention input: needs the {GRID} compute grid, got {(int(g.x), int(g.y))}")
        self.mesh_device = mesh_device
        self.eps = float(eps)
        self.debug = int(debug)
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _sources_tag()
        self._xy = None
        self._mc = None

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

    def _mcast_rect(self):
        """The QB rectangle (rows 0..7: cores 0..95) in NOC coordinates."""
        if self._mc is None:
            a = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
            b = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(GRID[0] - 1, 7))
            self._mc = (int(a.x), int(a.y), int(b.x), int(b.y))
        return self._mc

    def _shard_xy(self, upd):
        spec = upd.memory_config().shard_spec
        cores = ttnn.corerange_to_cores(spec.grid, row_wise=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR)
        out = []
        for cc in cores:
            v = self.mesh_device.worker_core_from_logical_core(cc)
            out.append((int(v.x) << 16) | int(v.y))
        return out

    # ---- checks -----------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_t(name, t, shape, dtype):
        if tuple(int(v) for v in t.shape) != tuple(shape):
            raise ValueError(f"fused attention input: {name} must be {tuple(shape)}, got {tuple(t.shape)}")
        if t.dtype != dtype or t.layout != ttnn.TILE_LAYOUT or not _is_interleaved(t.memory_config()):
            raise ValueError(f"fused attention input: {name} must be {dtype} TILE interleaved")

    def check(self, cq, kvl, cos, sin, w_q_b, w_gate, w_uk) -> int:
        """Raises ``ValueError`` unless the operands have the call contract; returns the rows ``L``."""
        shp = tuple(int(v) for v in cq.shape)
        if len(shp) != 4 or shp[:2] != (1, 1) or shp[3] != KQ * TILE or not 1 <= shp[2] <= TILE:
            raise ValueError(f"fused attention input: cq must be [1, 1, L <= 32, 1024], got {shp}")
        L = shp[2]
        self._check_t("cq", cq, (1, 1, L, KQ * TILE), ttnn.float32)
        self._check_t("kvl", kvl, (1, 1, L, 640), ttnn.bfloat16)
        for name, t in (("cos", cos), ("sin", sin)):
            if tuple(int(v) for v in t.shape)[-1] != 64 or int(t.padded_shape[-2]) != TILE or t.dtype != ttnn.bfloat16:
                raise ValueError(f"fused attention input: {name} must be [1, 1, 32, 64] bf16, got {tuple(t.shape)}")
            if t.layout != ttnn.TILE_LAYOUT or not _is_interleaved(t.memory_config()):
                raise ValueError(f"fused attention input: {name} must be TILE interleaved")
        for name, t, shape in (("w_q_b", w_q_b, (KQ * TILE, NQ * TILE)), ("w_gate", w_gate, (KQ * TILE, NG * TILE))):
            if tuple(int(v) for v in t.shape)[-2:] != shape or t.dtype != ttnn.bfloat16:
                raise ValueError(f"fused attention input: {name} must be {shape} bf16, got {tuple(t.shape)}")
        if tuple(int(v) for v in w_uk.shape) != (1, H, 128, 512) or w_uk.dtype != ttnn.bfloat16:
            raise ValueError(f"fused attention input: w_uk must be [1, 10, 128, 512] bf16, got {tuple(w_uk.shape)}")
        return L

    # ---- program ----------------------------------------------------------------------------------------------------
    def _program(self, L, kvmode, ins, outs):
        cq, kvl, cos, sin, wqb, wg, wuk = ins
        qmla, g, lam, kvo, dbg = outs
        ts = (cq, kvl, cos, sin, wqb, wg, wuk, qmla, g, lam, kvo, dbg)
        key = (L, kvmode) + tuple(tuple(_accessor(t)) for t in ts)
        desc = self._desc.get(key)
        gx, gy = GRID
        ncores = gx * gy
        if desc is None:
            all_cores = _rect(0, 0, gx - 1, gy - 1)
            main = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 2)),
                                      ttnn.CoreRange(ttnn.CoreCoord(0, gy - 1), ttnn.CoreCoord(gx - 3, gy - 1))})
            qn = _rect(*QN_CORE, *QN_CORE)
            kn = _rect(*KN_CORE, *KN_CORE)
            norm_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*KN_CORE), ttnn.CoreCoord(*QN_CORE))})
            # uniform CBs (every core; the remotely written ones need one address everywhere) first, then per role
            spec = [
                (CB_CQN, KQ, FP32, ttnn.float32, all_cores),
                (CB_KVROW, KVT, BF16, ttnn.bfloat16, all_cores),
                (CB_QMLA, max(L, 1), BF16, ttnn.bfloat16, all_cores),
                (CB_NOPE, 4, BF16, ttnn.bfloat16, all_cores),
                (CB_PEPART, 1, BF16, ttnn.bfloat16, all_cores),
                (CB_COS, 2, BF16, ttnn.bfloat16, all_cores),
                (CB_SIN, 2, BF16, ttnn.bfloat16, all_cores),
                (CB_SCAL, 1, BF16, ttnn.bfloat16, all_cores),
                (CB_ROT, 1, BF16, ttnn.bfloat16, all_cores),
                (CB_CI, 1, BF16, ttnn.bfloat16, all_cores),
                (CB_SI, 1, BF16, ttnn.bfloat16, all_cores),
                (CB_WA, 16, BF16, ttnn.bfloat16, main),
                (CB_WB, 16, BF16, ttnn.bfloat16, main),
                (CB_QOUT, 1, BF16, ttnn.bfloat16, main),
                (CB_QSEND, 1, BF16, ttnn.bfloat16, main),
                (CB_GOUT, 1, BF16, ttnn.bfloat16, main),
                (CB_WUK, 8, BF16, ttnn.bfloat16, main),
                (CB_UKOUT, 2, BF16, ttnn.bfloat16, main),
                (CB_ROPE, 1, BF16, ttnn.bfloat16, main),
                (CB_QX, KQ, FP32, ttnn.float32, qn),
                (CB_XMM2, KQ, FP32, ttnn.float32, norm_cores),
                (CB_EX2, 1, FP32, ttnn.float32, norm_cores),
                (CB_EX2PE, 1, FP32, ttnn.float32, norm_cores),
                (CB_RSCAL, 1, BF16, ttnn.bfloat16, norm_cores),
                (CB_EPS, 1, BF16, ttnn.bfloat16, norm_cores),
                (CB_KIN, 16, BF16, ttnn.bfloat16, kn),
                (CB_ROTIN, 2, BF16, ttnn.bfloat16, kn),
                (CB_KPEX, 2, BF16, ttnn.bfloat16, kn),
                (CB_LAM, 2, BF16, ttnn.bfloat16, kn),
            ]
            cbs = [ttnn.CBDescriptor(total_size=n * page, core_ranges=cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=page)]) for i, n, page, dt, cores in spec]
            sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=all_cores, initial_value=0) for s in range(N_SEMS)]
            x0, y0, x1, y1 = self._mcast_rect()
            acc = []
            for t in ts:
                acc += _accessor(t)
            defines = [("MOTIF_AIN_SRC", self._tag)]
            n_common = 12 + ncores + (8 if kvmode == "shard" else 0)
            kvm = 1 if kvmode == "shard" else 0
            from ..model_config import NUM_CB_SLOTS  # noqa: F401  (import check)

            kernels = []
            for role, cores, wt, w in ((0, main, 0, 0), (1, qn, KQ, KQ * TILE), (2, kn, 16, 512)):
                for risc in (0, 1):
                    ct = [role, risc, gx, L, kvm, x0, y0, x1, y1, 96, _f32_bits(self.eps), self.debug, ncores] + acc
                    cfg = ttnn.ReaderConfigDescriptor() if risc == 0 else ttnn.WriterConfigDescriptor()
                    kernels.append(ttnn.KernelDescriptor(
                        kernel_source=str(SOURCES["dataflow"]), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=cores, compile_time_args=ct, defines=defines, runtime_args=[],
                        common_runtime_args=[0] * n_common, config=cfg))
                kernels.append(ttnn.KernelDescriptor(
                    kernel_source=str(SOURCES["compute"]), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores, compile_time_args=[role, gx, wt, w], defines=defines, runtime_args=[],
                    common_runtime_args=[], config=compute_config()))
            desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hk = (L, kvmode, tuple(acc), self.debug, self._mc, _f32_bits(self.eps))
                hv = self._hash.get(hk)
                if hv is None:
                    hv = self._hash[hk] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        args = [t.buffer_address() for t in ts] + self._core_xy()
        if kvmode == "shard":
            args += self._shard_xy(kvo)
        for k in desc.kernels:
            if k.common_runtime_args:
                k.common_runtime_args = args
        return desc

    # ---- call -------------------------------------------------------------------------------------------------------
    def __call__(self, cq, kvl, cos, sin, w_q_b, w_gate, w_uk, *, update_mc=None):
        """``(q_mla [1, L, 10, 576], g [1, 1, L, 1024], lam [1, 1, L, 64], kv)`` bf16 from ``cq [1, 1, L, 1024]``
        fp32 and ``kvl [1, 1, L, 640]`` bf16 (the two latent projections), the step's RoPE tables (``[1, 1, 32, 64]``)
        and the layer's weights. ``kv``: with ``update_mc`` (the draft-1 height-sharded update layout, L = 8) the
        ``paged_update_cache`` input ``[1, L, 1, 576]`` in that layout (only row 0 of each lane tile is written: the
        rows the update reads); otherwise ``kv_row [1, 1, L, 576]`` DRAM. Consumes nothing. With ``debug`` the
        returned tuple has a 5th entry, cq_n ``[1, 1, L, 1024]`` fp32."""
        L = self.check(cq, kvl, cos, sin, w_q_b, w_gate, w_uk)
        dram = ttnn.DRAM_MEMORY_CONFIG
        dev = self.mesh_device

        def alloc(shape, dtype=ttnn.bfloat16, mc=dram):
            return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, dev, mc)

        qmla = alloc([1, L, H, NCOL * TILE])
        g = alloc([1, 1, L, NG * TILE])
        lam = alloc([1, 1, L, 64])
        if update_mc is not None:
            if L != 8:
                raise ValueError("fused attention input: the shard write serves the 8-lane draft-1 update")
            kvmode = "shard"
            kvo = alloc([1, L, 1, KVT * TILE], mc=update_mc)
        else:
            kvmode = "dram"
            kvo = alloc([1, 1, L, KVT * TILE])
        dbg = alloc([1, 1, L, KQ * TILE], ttnn.float32) if self.debug else cq
        ins = (cq, kvl, cos, sin, w_q_b, w_gate, w_uk)
        outs = (qmla, g, lam, kvo, dbg)
        desc = self._program(L, kvmode, ins, outs)
        io = [cq, kvl, cos, sin, w_q_b, w_gate, w_uk, qmla, g, lam, kvo] + ([dbg] if self.debug else [])
        ttnn.generic_op(io, desc)
        if self.debug:
            return qmla, g, lam, kvo, dbg
        return qmla, g, lam, kvo

    def deallocate(self) -> None:
        self._desc.clear()


__all__ = ["FusedAttnIn", "KV_MODES", "compute_config"]
