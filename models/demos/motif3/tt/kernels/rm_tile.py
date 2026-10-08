# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Row-major <-> tile layout of the decode token gather as pure data movement (Phase C D4 / plan item C4 "fewer
to_layouts", ``MOTIF3_AG_ROWS_LAYOUT=kernel``).

``MotifCCL.ag_dp_rows`` (the MoE token gather of every MoE layer, and the LM head's) is ``untilize`` of this row's
``[1, 1, L, W]`` -> ``all_gather`` over DP in ROW_MAJOR -> ``tilize`` of ``[1, 1, 4 L, W]``. At L = 8, W = 4096 the
two layout ops cost 4.6 + 11 us traced (``logs/opt/phaseC/D4/probe/probe2.json``; ttnn's tilize puts one 32-row block
on one core). For bf16 both are pure byte moves (a tile face row is 16 consecutive elements of one row):

* ``untilize``: TILE ``[1, 1, L, W]`` (L <= 32, one tile row) -> ROW_MAJOR, each input tile read once, two 32 B NoC
  writes per row (``rm_tile/untilize.cpp``);
* ``tilize``: ROW_MAJOR ``[1, 1, M, W]`` (M % 32 == 0) -> TILE, 64 NoC reads of 32 B per tile landing in place
  (``rm_tile/tilize.cpp``).

Both run on both data-movement RISCs of up to 64 cores and are bitwise equal to ``ttnn.to_layout`` (logical rows; the
tilized output has no padding rows: M is whole tiles). Operands: bf16 interleaved; the ROW_MAJOR side in L1 (the
32 B NoC transfers are L1-aligned; DRAM wants 64 B alignment on Blackhole). Trace safety as ``row_fold``: fresh
outputs, one program per buffer layout (hash memoized), addresses as common runtime args.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "rm_tile"
SOURCES = {name: KERNEL_DIR / f"{name}.cpp" for name in ("tilize", "untilize")}
TILE = 32
TB = 2048
CB_SCR = 0
WORKERS = 64


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


def _interleaved(t) -> bool:
    return t.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


def _is_l1(mc) -> bool:
    return mc.buffer_type == ttnn.BufferType.L1


class RowLayout:
    """Tilize / untilize programs of one mesh (module docstring). Stateless apart from cached program descriptors."""

    def __init__(self, mesh_device, *, workers: int = WORKERS):
        self.mesh_device = mesh_device
        g = mesh_device.compute_with_storage_grid_size()
        self.grid = (int(g.x), int(g.y))
        self.workers = int(workers)
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _sources_tag()

    # ---- support predicates (host-side, cheap) ------------------------------------------------------------------
    @staticmethod
    def untilize_supported(x, out_memory_config) -> bool:
        s = [int(d) for d in x.shape]
        return (x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT and _interleaved(x) and len(s) == 4
                and s[0] == 1 and s[1] == 1 and 1 <= s[2] <= TILE and s[3] % TILE == 0 and _is_l1(out_memory_config)
                and out_memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)

    @staticmethod
    def tilize_supported(x, out_memory_config) -> bool:
        s = [int(d) for d in x.shape]
        return (x.dtype == ttnn.bfloat16 and x.layout == ttnn.ROW_MAJOR_LAYOUT and _interleaved(x) and len(s) == 4
                and s[0] == 1 and s[1] == 1 and s[2] % TILE == 0 and s[3] % TILE == 0
                and _is_l1(x.memory_config())
                and out_memory_config.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED)

    # ---- programs -----------------------------------------------------------------------------------------------
    def _run(self, kind, src, out, head, n_units, scr_bytes, rowb):
        per = math.ceil(n_units / (2 * self.workers))
        slots = math.ceil(n_units / per)
        workers = math.ceil(slots / 2)
        gx = min(self.grid[0], 8)
        if math.ceil(workers / gx) > self.grid[1]:
            raise ValueError(f"rm_tile {kind}: {workers} workers do not fit the {self.grid} grid")
        scr = ((int(scr_bytes) + 63) // 64) * 64
        h = list(head) + [per, gx, CB_SCR]
        tail = [scr, int(rowb)] + _accessor(src) + _accessor(out)
        key = (kind, tuple(h + tail))
        desc = self._desc.get(key)
        if desc is None:
            cores = _cores(workers, gx)
            defines = [("MOTIF_RMTILE_SRC", self._tag)]

            def kern(risc, config):
                return ttnn.KernelDescriptor(
                    kernel_source=str(SOURCES[kind]), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores, compile_time_args=h + [risc] + tail, defines=defines, runtime_args=[],
                    common_runtime_args=[0, 0], config=config)

            size = 2 * scr + 64
            cb = ttnn.CBDescriptor(total_size=size, core_ranges=cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCR, data_format=ttnn.bfloat16, page_size=size)])
            desc = ttnn.ProgramDescriptor(
                kernels=[kern(0, ttnn.WriterConfigDescriptor()), kern(1, ttnn.ReaderConfigDescriptor())],
                semaphores=[], cbs=[cb])
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

    def untilize(self, x, *, memory_config=None):
        """TILE ``[1, 1, L, W]`` (L <= 32) -> ROW_MAJOR ``[1, 1, L, W]`` in ``memory_config`` (L1 interleaved, the
        default). Consumes nothing."""
        mc = memory_config or ttnn.L1_MEMORY_CONFIG
        if not self.untilize_supported(x, mc):
            raise ValueError(f"rm_tile untilize: unsupported {x.dtype} {x.layout} {list(x.shape)} -> {mc}")
        L, W = int(x.shape[2]), int(x.shape[3])
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, L, W]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
                                             self.mesh_device, mc)
        WT = W // TILE
        return self._run("untilize", x, out, [L, WT], WT, 2 * TB, W * 2)

    def tilize(self, x, *, memory_config=None):
        """ROW_MAJOR ``[1, 1, M, W]`` (M % 32 == 0, L1) -> TILE ``[1, 1, M, W]`` in ``memory_config`` (DRAM
        interleaved by default). Consumes nothing."""
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        if not self.tilize_supported(x, mc):
            raise ValueError(f"rm_tile tilize: unsupported {x.dtype} {x.layout} {list(x.shape)} -> {mc}")
        M, W = int(x.shape[2]), int(x.shape[3])
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.mesh_device, mc)
        WT = W // TILE
        NO = (M // TILE) * WT
        return self._run("tilize", x, out, [WT, NO], NO, 2 * TB, W * 2)

    def deallocate(self) -> None:
        self._desc.clear()


__all__ = ["RowLayout"]
