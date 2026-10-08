# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Row fold / unfold of the decode MoE combine (Phase C D4 / plan item C4, ``MOTIF3_MOE_DECODE_CCL=rs``).

The decode MoE ends (release) with ``AR(dp)`` of the routed partial ``P [1, 1, N L, H]`` (N = 4 DP rows of L = 8
lanes, or 16 rows in the T64 step; H = 4096), then ``partition(2, "dp")`` to keep this row's L rows. The partition of
an 8-row slice of a TILE tensor is untilize + mesh_partition + tilize, and the all-reduce gathers 4x the rows a chip
keeps. A row reduce-scatter cannot express "8 of the 32 rows of a tile". Folding each DP row's ``L x H`` block into
``L N x F`` (F = H / N) makes it whole tiles::

    fold   : P [1, 1, N L, H]  -> Q [1, N, L N, F]   Q[b, N j + q, c] = P[L b + j, q F + c]
    RS(dp) on dim 1            -> R [1, 1, L N, F]   this chip's DP row b, summed over the N chips of its column
    unfold : R [1, 1, L N, F]  -> [1, 1, L, H]       out[j, q F + c] = R[N j + q, c]

Both maps are exactly the logical reshapes ``[1, 1, N L, H] <-> [1, N, L N, F]`` / ``[1, 1, L N, F] <-> [1, 1, L,
H]`` (``ttnn.reshape`` computes the same values bit for bit at 12.9 / 5.7 us traced; these kernels are pure data
movement on 64 / 32 cores). The unfolded output's padding rows (L .. 31) are zeros.

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
SOURCES = {name: KERNEL_DIR / f"{name}.cpp" for name in ("fold", "unfold")}
TILE = 32
TB = 2048
CB_SCR = 0
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

    def deallocate(self) -> None:
        self._desc.clear()


__all__ = ["RowFold"]
