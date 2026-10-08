# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Row fold / unfold of the decode MoE combine (Phase C D4 / plan item C4, ``MOTIF3_MOE_DECODE_CCL=rs``).

The decode MoE ends (release) with ``AR(dp)`` of the routed partial ``P [1, 1, N L, H]`` (N = 4 DP rows of L = 8
lanes, or 16 rows in the T64 step; H = 4096), ``partition(2, "dp")`` to keep this row's L rows (untilize +
mesh_partition + tilize), ``+ add_partial`` and ``AR(tp)`` of the tile-padded ``[1, 1, L, H]``. A row reduce-scatter
cannot express "8 of the 32 rows of a tile". Folding each DP row's ``L x H`` block into ``L N x F`` (F = H / N) makes
it whole tiles::

    fold     : P [1, 1, N L, H] -> Q [1, N, L N, F]   Q[b, N j + q, c] = P[L b + j, q F + c]
    RS(dp) on dim 1             -> R [1, 1, L N, F]   this chip's DP row b, summed over the N chips of its column
    fold_add : R + fold(S)                            S = add_partial [1, 1, L, H]; fold(S)[N j + q, c] = S[j, q F + c]
    AR(tp)   on [1, 1, L N, F]                        whole tiles: 4x fewer than the padded [1, 1, L, H] (L = 8)
    unfold   : [1, 1, L N, F]   -> [1, 1, L, H]       out[j, q F + c] = X[N j + q, c]

Every map is exactly a logical reshape (``[1, 1, N L, H] <-> [1, N, L N, F]``, ``[1, 1, L N, F] <-> [1, 1, L, H]``):
the kernels are pure data movement on both data-movement RISCs (``fold`` 3.7 us, ``unfold`` 2.2 us traced at L = 8
with L1 operands, against 12 / 5.7 us for ``ttnn.reshape``), and ``fold_add`` adds with ttnn.add's LLK sequence
(bitwise ``ttnn.add(R, ttnn.reshape(S, ...))``, 3.5 us). The unfolded output's padding rows (L .. 31) are zeros.
Chain at L = 8, traced: 80.4 us (release) -> 53.1 us (logs/opt/phaseC/D4/probe/probe5.json).

Shapes: bf16 TILE interleaved (L1 or DRAM); N = 4; L a multiple of 8 and <= 32. Trace safety: fresh output
allocations; one program per buffer layout, hash memoized; addresses are common runtime args.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "row_fold"
SOURCES = {name: KERNEL_DIR / f"{name}.cpp"
           for name in ("fold", "unfold", "fold_add_reader", "fold_add_compute", "fold_add_writer")}
TILE = 32
TB = 2048
CB_SCR = 0
CB_R, CB_S, CB_FA_SCR, CB_OUT = 0, 1, 2, 16  # fold_add
FOLD_WORKERS = 64
UNFOLD_WORKERS = 64


def _sources_tag() -> str:
    h = hashlib.sha1()
    for p in SOURCES.values():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cores(n: int, gx: int):
    full, rem = divmod(int(n), int(gx))
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges)


def _scratch(cores, nbytes: int):
    size = ((int(nbytes) + 64 + 63) // 64) * 64
    return ttnn.CBDescriptor(total_size=size, core_ranges=cores,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_SCR, data_format=ttnn.bfloat16,
                                                                         page_size=size)])


def _interleaved(t) -> bool:
    return t.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


class RowFold:
    """Fold / unfold programs of one mesh (module docstring). Stateless apart from cached program descriptors."""

    def __init__(self, mesh_device, *, n: int = 4, fold_workers: int = FOLD_WORKERS,
                 unfold_workers: int = UNFOLD_WORKERS):
        if int(n) != 4:
            raise ValueError(f"row fold: n must be 4, got {n}")
        self.mesh_device = mesh_device
        self.n = int(n)
        self.fold_workers, self.unfold_workers = int(fold_workers), int(unfold_workers)
        g = mesh_device.compute_with_storage_grid_size()
        self.grid = (int(g.x), int(g.y))
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _sources_tag()
        self.debug_defines: tuple = ()  # probes only: ("MOTIF_ROWFOLD_NOCOPY", "1") etc. (wrong output)

    @staticmethod
    def supports_shape(rows: int, width: int, n: int = 4) -> bool:
        """Whether ``L = rows`` per DP row, ``H = width`` and ``n`` DP rows fit the kernels."""
        L, H = int(rows), int(width)
        return int(n) == 4 and L % 8 == 0 and 8 <= L <= TILE and H % (TILE * int(n)) == 0

    def supports(self, rows: int, width: int) -> bool:
        """Whether ``L = rows`` per DP row and ``H = width`` fit the kernels."""
        return self.supports_shape(rows, width, self.n)

    def _check(self, t, shape, what):
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT or not _interleaved(t):
            raise ValueError(f"row fold: {what} must be bf16 TILE interleaved, got {t.dtype} {t.layout}")
        if tuple(int(d) for d in t.shape) != tuple(shape):
            raise ValueError(f"row fold: {what} must be {list(shape)}, got {list(t.shape)}")

    def _run(self, kind, src, out, ct_head, n_units, scr_bytes):
        """``n_units`` output tiles over ``2 x workers`` slots (both data-movement RISCs of each core), at most
        ``self.<kind>_workers`` cores; ``scr_bytes`` of scratch per RISC."""
        cap = self.fold_workers if kind == "fold" else self.unfold_workers
        per = math.ceil(n_units / (2 * cap))
        slots = math.ceil(n_units / per)
        workers = math.ceil(slots / 2)
        gx = min(self.grid[0], 8 if workers <= 64 else self.grid[0])
        if math.ceil(workers / gx) > self.grid[1]:
            raise ValueError(f"row fold: {workers} workers do not fit the {self.grid} grid")
        scr = ((int(scr_bytes) + 63) // 64) * 64
        head = list(ct_head) + [per, gx, CB_SCR]
        tail = [scr] + _accessor(src) + _accessor(out)
        key = (kind, tuple(head + tail), tuple(self.debug_defines))
        desc = self._desc.get(key)
        if desc is None:
            cores = _cores(workers, gx)
            defines = [("MOTIF_ROWFOLD_SRC", self._tag)] + [tuple(d) for d in self.debug_defines]

            def kern(risc, config):
                return ttnn.KernelDescriptor(
                    kernel_source=str(SOURCES[kind]), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores, compile_time_args=head + [risc] + tail, defines=defines, runtime_args=[],
                    common_runtime_args=[0, 0], config=config)

            kernels = [kern(0, ttnn.WriterConfigDescriptor()), kern(1, ttnn.ReaderConfigDescriptor())]
            desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[_scratch(cores, 2 * scr)])
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hv = self._hash.get(key)
                if hv is None:
                    hv = self._hash[key] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        addrs = [src.buffer_address(), out.buffer_address()]
        for k in desc.kernels:
            k.common_runtime_args = addrs
        ttnn.generic_op([src, out], desc)
        return out

    def fold(self, p, rows: int, *, memory_config=None):
        """``P [1, 1, N L, H]`` -> ``Q [1, N, L N, H / N]`` (L = ``rows``). Consumes nothing."""
        N, L = self.n, int(rows)
        H = int(p.shape[-1])
        if not self.supports(L, H):
            raise ValueError(f"row fold: L = {L}, H = {H} not supported")
        self._check(p, (1, 1, N * L, H), "P")
        F = H // N
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        q = ttnn.allocate_tensor_on_device(ttnn.Shape([1, N, L * N, F]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                           self.mesh_device, mc)
        n_out = N * (L * N // TILE) * (F // TILE)
        ct = [N, L, H // TILE, F // TILE, n_out]
        return self._run("fold", p, q, ct, n_out, N * 2 * 8 * 32 + 2 * TB)

    def unfold(self, r, rows: int, *, memory_config=None):
        """``R [1, 1, L N, F]`` -> ``[1, 1, L, N F]`` (padding rows zero). Consumes nothing."""
        N, L = self.n, int(rows)
        F = int(r.shape[-1])
        if not self.supports(L, F * N):
            raise ValueError(f"row unfold: L = {L}, H = {F * N} not supported")
        self._check(r, (1, 1, L * N, F), "R")
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, L, N * F]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.mesh_device, mc)
        FT = F // TILE
        ct = [N, L, FT]
        return self._run("unfold", r, out, ct, N * FT, (L * N // TILE) * TB + 2 * TB)

    def fold_add(self, r, s, rows: int, *, memory_config=None):
        """``R [1, 1, N L, F] + fold(S)`` with ``S [1, 1, L, N F]`` (the TP add_partial; ``fold(S)[N j + q, c] =
        S[j, q F + c]``, the logical reshape): ``[1, 1, N L, F]``, one bf16 FPU add per tile (bitwise ``ttnn.add(R,
        ttnn.reshape(S, [1, 1, N L, F]))``). Consumes nothing."""
        N, L = self.n, int(rows)
        F = int(r.shape[-1])
        H = F * N
        if not self.supports(L, H):
            raise ValueError(f"row fold_add: L = {L}, H = {H} not supported")
        self._check(r, (1, 1, L * N, F), "R")
        self._check(s, (1, 1, L, H), "S")
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, L * N, F]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.mesh_device, mc)
        FT = F // TILE
        n_out = (L * N // TILE) * FT
        per = math.ceil(n_out / self.fold_workers)
        workers = math.ceil(n_out / per)
        gx = min(self.grid[0], 8)
        if math.ceil(workers / gx) > self.grid[1]:
            raise ValueError(f"row fold_add: {workers} workers do not fit the {self.grid} grid")
        r_ct = [N, L, H // TILE, FT, n_out, per, gx, CB_R, CB_S, CB_FA_SCR] + _accessor(r) + _accessor(s)
        c_ct = [CB_R, CB_S, CB_OUT, n_out, per, gx]
        w_ct = [CB_OUT, n_out, per, gx] + _accessor(out)
        key = ("fold_add", tuple(r_ct), tuple(w_ct))
        desc = self._desc.get(key)
        if desc is None:
            cores = _cores(workers, gx)
            defines = [("MOTIF_ROWFOLD_SRC", self._tag)]

            def kern(name, ct, config):
                return ttnn.KernelDescriptor(
                    kernel_source=str(SOURCES[name]), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores, compile_time_args=ct, defines=defines, runtime_args=[],
                    common_runtime_args=[0] * (2 if name == "fold_add_reader" else 1 if name == "fold_add_writer"
                                               else 0), config=config)

            cc = ttnn.ComputeConfigDescriptor()
            cc.math_fidelity = ttnn.MathFidelity.HiFi4
            cc.fp32_dest_acc_en = False
            cc.math_approx_mode = False
            cc.dst_full_sync_en = False

            def cb(index, pages):
                return ttnn.CBDescriptor(total_size=pages * TB, core_ranges=cores, format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=TB)])

            cbs = [cb(CB_R, 2), cb(CB_S, 2), cb(CB_OUT, 2),
                   ttnn.CBDescriptor(total_size=N * 2 * 8 * 32 + 64, core_ranges=cores, format_descriptors=[
                       ttnn.CBFormatDescriptor(buffer_index=CB_FA_SCR, data_format=ttnn.bfloat16,
                                               page_size=N * 2 * 8 * 32 + 64)])]
            kernels = [kern("fold_add_reader", r_ct, ttnn.ReaderConfigDescriptor()),
                       kern("fold_add_writer", w_ct, ttnn.WriterConfigDescriptor()),
                       kern("fold_add_compute", c_ct, cc)]
            desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hv = self._hash.get(key)
                if hv is None:
                    hv = self._hash[key] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        desc.kernels[0].common_runtime_args = [r.buffer_address(), s.buffer_address()]
        desc.kernels[1].common_runtime_args = [out.buffer_address()]
        ttnn.generic_op([r, s, out], desc)
        return out

    def deallocate(self) -> None:
        self._desc.clear()


__all__ = ["RowFold"]
