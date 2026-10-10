# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device sampler's two selection / packing tails as data movement programs (Phase F, F4;
``MOTIF3_SAMPLER_LOCAL=fused``; with ``tt/kernels/sampler_local.py``; results ``logs/opt/phaseF/F4``).

* :meth:`FusedSamplerTail.gumbel_winner` -- the end of ``MotifDeviceSampler._gumbel_gather``: from the gathered per-chip
  Gumbel triples ``G [1, 1, 96, 1024]`` (score / global id / raw ``x - M``, chip k in column 32 k) the global max score,
  the lowest id attaining it and its raw ``x - M`` (``g_score``, ``g_tok``, ``g_dx``, ``[1, 1, 32, 1]`` fp32). The op
  chain is 3 slices + 8 programs with three single-core reductions over 1024 columns (~56 us per step, F0); here 32
  cores (one per lane) read the 96 values of their lane and select (``sampler_tail/gpost.cpp``).
* :meth:`FusedSamplerTail.pack` -- ``MotifDeviceSampler._stage_outputs`` without the coverage counter: the eight
  ``[32, 1]`` result columns -> ``info [1, 1, 8, 32]`` fp32 and ``tokens [1, 1, 1, 32]`` uint32 ROW_MAJOR (the op chain:
  a one-core concat, 2 transposes, a typecast, 2 untilizes, ~45 us); here one core (``sampler_tail/pack.cpp``).

Both only move and select values (max / min / equality of fp32 numbers without NaN or -0, the integer value of a token
id): bitwise the op chain's results (``tests/unit/test_sampling.py::test_sampler_local_fused_bitwise``). Trace safe:
fresh outputs per call, descriptors cached per buffer layout, a traced call patches the addresses only.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Dict, Sequence, Tuple

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "sampler_tail"
GPOST_SRC = KERNEL_DIR / "gpost.cpp"
PACK_SRC = KERNEL_DIR / "pack.cpp"
TILE = 32
GXQ, GYQ = 8, 4  # the 32 gpost cores (lane j = y GXQ + x)
BIG = 3.0e38  # tt/sampling.py BIG


def _bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _tag() -> str:
    h = hashlib.sha1()
    for p in (GPOST_SRC, PACK_SRC):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


class FusedSamplerTail:
    """The two tails of one mesh (module docstring). Stateless apart from the cached descriptors."""

    def __init__(self, mesh_device):
        g = mesh_device.compute_with_storage_grid_size()
        if int(g.x) < GXQ or int(g.y) < GYQ:
            raise ValueError(f"sampler_tail: needs an {GXQ} x {GYQ} compute grid, got {(int(g.x), int(g.y))}")
        self.mesh_device = mesh_device
        self._desc: Dict[tuple, object] = {}
        self._tag = _tag()

    @staticmethod
    def _interleaved(*ts) -> bool:
        return all(t.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED for t in ts)

    def supports_gumbel(self, G) -> bool:
        return (tuple(int(v) for v in G.shape) == (1, 1, 3 * TILE, TILE * TILE) and G.dtype == ttnn.float32
                and G.layout == ttnn.TILE_LAYOUT and self._interleaved(G))

    def supports_pack(self, rows: Sequence) -> bool:
        return len(rows) == 8 and all(
            tuple(int(v) for v in r.shape) == (1, 1, TILE, 1) and r.dtype == ttnn.float32 and r.layout == ttnn.TILE_LAYOUT
            for r in rows) and self._interleaved(*rows)

    def _program(self, kind, ts, n_args):
        key = (kind,) + tuple(tuple(_accessor(t)) for t in ts)
        desc = self._desc.get(key)
        if desc is None:
            acc = []
            for t in ts:
                acc += _accessor(t)
            if kind == "gpost":
                cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GXQ - 1, GYQ - 1))})
                src, ct, size = GPOST_SRC, [GXQ] + acc, 96 * 64 + 3 * 128
            else:
                cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
                src, ct, size = PACK_SRC, acc, 8 * 32 * 64 + 9 * 128
            cbs = [ttnn.CBDescriptor(total_size=size, core_ranges=cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=size)])]
            kernels = [ttnn.KernelDescriptor(
                kernel_source=str(src), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=cores,
                compile_time_args=ct, defines=[("MOTIF_ST_SRC", self._tag)], runtime_args=[],
                common_runtime_args=[0] * n_args, config=ttnn.ReaderConfigDescriptor())]
            desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                desc.custom_program_hash = ttnn.compute_program_descriptor_hash(desc)
            self._desc[key] = desc
        return desc

    def gumbel_winner(self, G, *, memory_config=None) -> Tuple[object, object, object]:
        """``G [1, 1, 96, 1024]`` fp32 TILE -> ``(g_score, g_tok, g_dx)`` ``[1, 1, 32, 1]`` fp32 TILE."""
        if not self.supports_gumbel(G):
            raise ValueError("sampler_tail: G must be fp32 TILE [1, 1, 96, 1024] interleaved")
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE, 1]), ttnn.float32, ttnn.TILE_LAYOUT,
                                               self.mesh_device, mc) for _ in range(3)]
        ts = (G, *outs)
        desc = self._program("gpost", ts, 6)
        args = [t.buffer_address() for t in ts] + [_bits(-BIG), _bits(BIG)]
        for k in desc.kernels:
            k.common_runtime_args = args
        ttnn.generic_op(list(ts), desc)
        return outs[0], outs[1], outs[2]

    def pack(self, rows: Sequence, *, memory_config=None) -> Tuple[object, object]:
        """The eight ``[1, 1, 32, 1]`` fp32 result columns (token, flag, logprob, n_kept, u, mass_kept, window mass,
        kth) -> ``(tokens [1, 1, 1, 32] uint32, info [1, 1, 8, 32] fp32)`` ROW_MAJOR."""
        if not self.supports_pack(rows):
            raise ValueError("sampler_tail: the result columns must be eight fp32 TILE [1, 1, 32, 1]")
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        info = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 8, TILE]), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT,
                                              self.mesh_device, mc)
        tokens = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, TILE]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT,
                                                self.mesh_device, mc)
        ts = (*rows, info, tokens)
        desc = self._program("pack", ts, 10)
        args = [t.buffer_address() for t in ts]
        for k in desc.kernels:
            k.common_runtime_args = args
        ttnn.generic_op(list(ts), desc)
        return tokens, info

    def deallocate(self) -> None:
        self._desc.clear()


__all__ = ["FusedSamplerTail"]
