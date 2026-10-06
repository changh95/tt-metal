# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compacted prefill MoE kernels (B2b, docs/OPTIMIZATION_PLAN.md §3.3 B2; review docs/OPT_PHASE_A_REVIEW.md §7.2 step 6:
"build the metadata / local-dispatch kernel first"). Two ``generic_op`` programs that replace the parts of B2a's
compacted layer (``MotifMoE._compact_partial``) the B2b probe measured as the cost (``logs/opt/phaseB/B2b``):

* :class:`CompactDispatch` (``moe_dispatch/dispatch.cpp``): the row lists B2a builds on the host
  (:func:`tt.moe.compact_upload_fast`) built on device from the router's ``idx`` and this chip's ``w_loc``, so the
  layer no longer reads the routes, runs numpy and uploads (the blocking read drained the device: 9-10 ms of idle device
  per layer at 4K, 5-7 ms at 1K [M, probe1]). The host reads only ``need`` (32 B, chip 0) to pick the block count
  ``NB`` of the program shapes that follow (a ladder entry, exactly B2a's), and the routing weights of the rows are
  copied straight from ``w_loc`` (B2a's on-device transpose + gather: 0.9-3.4 ms traced per layer).
  Outputs (``CAP`` = the ladder's last entry, so they never depend on the routes): ``rows [1, 1, 2, CAP MB]`` uint32
  ROW_MAJOR (page 0 the token of every row, page 1 the combine key, pad rows M), ``wcol [1, CAP, MB, 1]`` fp32 TILE,
  ``sp [1, CAP, 1, 32]`` bf16 ROW_MAJOR (block one-hot), ``blk [1, CAP, 1, 32]`` bf16 TILE (B2a's block word: one-hot +
  the PolyNorm constants at columns 16..19) and ``need [1, 1, 1, 8]`` uint32 (``[need, NB or 0, used]``). Only blocks
  ``[0, NB)`` are written: the caller slices that prefix.
* :class:`GatherCombine` (``moe_combine/{reader,compute,writer}.cpp``): ``part[t] = sum of y[j] over the rows j of
  token t`` in row order, added into an fp32 dest with the ops of ``fast_reduce_nc`` (the dense path's expert sum) and
  packed once -- in place of B2a's one-hot ``P^T @ y`` matmul (2.5-3.8 ms traced per layer at 4K [M, probe1]: it
  multiplies by M x R mostly-zero entries).

Both are exact: the dispatch outputs equal the host lists bit for bit, and the combine adds the same bf16 terms in the
same order into the same fp32 dest as the dense path (absent terms are exact zeros on both). Deterministic (no
races: every output byte has one writer; no atomics). Programs are built on the first eager call per shape and their
hashes memoized; nothing here allocates persistent device memory except the two small constant tables of
:class:`CompactDispatch` (slot table, per-chip meta).

Host helpers (pure): :func:`dispatch_reference` (the kernel's outputs from host routes, via the B2a packer),
:func:`chip_meta`, :func:`slot_table`, :func:`combine_groups`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import torch

import ttnn

DISPATCH_DIR = Path(__file__).resolve().parent / "moe_dispatch"
COMBINE_DIR = Path(__file__).resolve().parent / "moe_combine"
DISPATCH_SRC = DISPATCH_DIR / "dispatch.cpp"
COMBINE_READER = COMBINE_DIR / "reader.cpp"
COMBINE_COMPUTE = COMBINE_DIR / "compute.cpp"
COMBINE_WRITER = COMBINE_DIR / "writer.cpp"

TILE = 32
MAX_LADDER = 16
META_WORDS = 64
BF16_ONE = 0x3F80


def _tag(paths) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _core_range_set(n: int, gx: int):
    full, rem = divmod(int(n), int(gx))
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges)


def _scratch(index: int, cores, nbytes: int, fmt=ttnn.float32):
    size = ((int(nbytes) + 64 + 63) // 64) * 64  # + 64 B: the kernels align their base up to 64 B
    return ttnn.CBDescriptor(total_size=size, core_ranges=cores,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt,
                                                                         page_size=size)])


def _cb(index: int, cores, page: int, n_pages: int, fmt):
    return ttnn.CBDescriptor(total_size=int(page) * int(n_pages), core_ranges=cores,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt,
                                                                         page_size=int(page))])


def _hash(desc, cache: dict, key):
    if not hasattr(ttnn, "compute_program_descriptor_hash"):
        return
    hv = cache.get(key)
    if hv is None:
        hv = cache[key] = ttnn.compute_program_descriptor_hash(desc)
    desc.custom_program_hash = hv


# =====================================================================================================================
# host side (pure)
# =====================================================================================================================
def slot_table(local_ids: torch.Tensor) -> torch.Tensor:
    """``local_ids [P, E]`` (chip ``p``'s local expert ``e``: global id) -> ``[NE]`` int64: global id -> ``p E + e``
    (every id must be held by exactly one chip)."""
    P, E = (int(v) for v in local_ids.shape)
    flat = local_ids.reshape(-1).long()
    ne = int(flat.max()) + 1
    out = torch.full((ne,), -1, dtype=torch.long)
    out[flat] = torch.arange(P * E, dtype=torch.long)
    if bool((out < 0).any()) or ne != P * E:
        raise ValueError("compact dispatch: every global expert id must be one chip's local expert exactly once")
    return out


def chip_meta(local_ids: torch.Tensor, pn_bits) -> torch.Tensor:
    """Per-chip constants of the dispatch kernel ``[P, 64]`` int64: ``[0]`` the chip index ``p`` (row-major mesh
    coordinate), ``[1 .. E]`` its global ids, ``[16 + 4 e + i]`` the bf16 bit pattern of expert ``e``'s PolyNorm
    constant ``i`` (c0, c1, c2, b; ``pn_bits [P, E, 4]``)."""
    P, E = (int(v) for v in local_ids.shape)
    if E > 12:
        raise ValueError(f"compact dispatch: {E} local experts > 12")
    pb = torch.as_tensor(pn_bits).to(torch.int64) & 0xFFFF
    if tuple(pb.shape) != (P, E, 4):
        raise ValueError(f"compact dispatch: PolyNorm bits of shape {tuple(pb.shape)}, want {(P, E, 4)}")
    m = torch.zeros(P, META_WORDS, dtype=torch.int64)
    m[:, 0] = torch.arange(P)
    m[:, 1:1 + E] = local_ids.long()
    m[:, 16:16 + 4 * E] = pb.reshape(P, 4 * E)
    return m


def dispatch_reference(u, mb: int, nb: int, M: int):
    """What the dispatch kernel writes for blocks ``[0, NB)``, from B2a's upload ``u [P, 4, NB MB]``
    (:func:`tt.moe.compact_upload_fast`): ``(tix [P, R], keys [P, R], sp [P, NB, 32] int16 bits, blk_row [P, NB, 32]
    int bits)``; ``wcol`` is checked against the gather of ``w_loc`` by the caller."""
    u = torch.as_tensor(u).to(torch.int64)
    P, _, R = (int(v) for v in u.shape)
    if R != int(nb) * int(mb):
        raise ValueError("dispatch_reference: upload rows != NB x MB")
    blk = u[:, 3, : 32 * int(nb)].reshape(P, int(nb), 32)
    sp = blk.clone()
    sp[:, :, 12:] = 0
    return u[:, 0], u[:, 1], sp, blk


def combine_groups(M: int, n_workers: int, wt: int) -> int:
    """Column groups ``G`` of the combine: the smallest power of two dividing ``wt`` with ``(M / 32) G >= n_workers``
    (each unit is one tile row x ``wt / G`` column tiles), at most 16."""
    n_tr = int(M) // TILE
    g = 1
    while n_tr * g < int(n_workers) and g < 16 and int(wt) % (2 * g) == 0:
        g *= 2
    return g


# =====================================================================================================================
# device
# =====================================================================================================================
class CompactRows(NamedTuple):
    rows: object  # [1, 1, 2, CAP MB] uint32 ROW_MAJOR (token, key)
    wcol: object  # [1, CAP, MB, 1] fp32 TILE
    sp: object  # [1, CAP, 1, 32] bf16 ROW_MAJOR
    blk: object  # [1, CAP, 1, 32] bf16 TILE
    need: object  # [1, 1, 1, 8] uint32 ROW_MAJOR

    def free(self) -> None:
        for t in self:
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)


class CompactDispatch:
    """The on-device local dispatch of one model's compacted prefill MoE layers (module docstring). Owns two small
    constant tensors: the slot table ``[1, 1, 1, NE]`` uint32 (replicated) and the per-chip meta ``[1, 1, 1, 64]``
    uint32 (:func:`chip_meta`; per-chip shards through ``mapper``). ``pn_bits`` differ per layer: a layer passes its
    own meta tensor (:meth:`make_meta`)."""

    def __init__(self, mesh_device, local_ids: torch.Tensor, *, mapper, top_k: int, memory_config=None):
        self.mesh_device = mesh_device
        self.mapper = mapper
        self.local_ids = local_ids.long()
        self.P, self.E = (int(v) for v in local_ids.shape)
        self.top_k = int(top_k)
        self.dram = memory_config or ttnn.DRAM_MEMORY_CONFIG
        st = slot_table(self.local_ids)
        self.n_experts = int(st.numel())
        base = self.local_ids[:, :1]
        self.contiguous = bool(torch.equal(self.local_ids, base + torch.arange(self.E).unsqueeze(0)))
        self.slot = ttnn.from_torch(st.reshape(1, 1, 1, -1).to(torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=self.dram,
                                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
        g = mesh_device.compute_with_storage_grid_size()
        self.grid = (int(g.x), int(g.y))
        if self.E + 1 > self.grid[0] * self.grid[1]:
            raise ValueError("compact dispatch: grid too small")
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _tag([DISPATCH_SRC])

    def make_meta(self, pn_bits):
        """This layer's per-chip meta tensor (:func:`chip_meta`) on device; the caller owns it."""
        R, C = (int(v) for v in tuple(self.mesh_device.shape))
        m = chip_meta(self.local_ids, pn_bits).to(torch.int32).reshape(R, C, 1, META_WORDS)
        return ttnn.from_torch(m, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device,
                               memory_config=self.dram, mesh_mapper=self.mapper)

    def _check(self, idx, wsrc, M: int, w_is_loc: bool):
        ishape = tuple(int(v) for v in idx.shape)
        if ishape != (1, 1, M, self.top_k):
            raise ValueError(f"compact dispatch: idx must be [1, 1, {M}, {self.top_k}], got {ishape}")
        if idx.layout != ttnn.TILE_LAYOUT or idx.dtype != ttnn.uint32:
            raise ValueError(f"compact dispatch: idx must be uint32 TILE, got {idx.dtype} {idx.layout}")
        want = (1, self.E, M, 1) if w_is_loc else (1, 1, M, self.top_k)
        wshape = tuple(int(v) for v in wsrc.shape)
        if wshape != want or wsrc.dtype != ttnn.float32 or wsrc.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"compact dispatch: weights must be {list(want)} fp32 TILE, got {wshape} {wsrc.dtype} "
                             f"{wsrc.layout}")
        for t in (idx, wsrc):
            if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                raise ValueError("compact dispatch: inputs must be interleaved")

    def __call__(self, idx, wsrc, meta, *, M: int, mb: int, ladder: Sequence[int], w_is_loc: bool = True) -> CompactRows:
        """``idx [1, 1, M, K]`` uint32 TILE, ``wsrc``: this chip's ``w_loc [1, E, M, 1]`` (``w_is_loc``) or the router's
        ``w [1, 1, M, K]`` (fp32 TILE), ``meta`` (:meth:`make_meta`) -> :class:`CompactRows` (fresh DRAM tensors)."""
        M, mb = int(M), int(mb)
        ladder = tuple(int(v) for v in ladder)
        if not 1 <= len(ladder) <= MAX_LADDER or list(ladder) != sorted(set(ladder)):
            raise ValueError(f"compact dispatch: ladder {ladder} must be 1..{MAX_LADDER} increasing entries")
        if M % TILE or mb % TILE:
            raise ValueError(f"compact dispatch: M = {M}, MB = {mb} must be multiples of {TILE}")
        self._check(idx, wsrc, M, bool(w_is_loc))
        cap = ladder[-1]
        dev, dram = self.mesh_device, self.dram
        rows = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 2, cap * mb]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT,
                                              dev, dram)
        wcol = ttnn.allocate_tensor_on_device(ttnn.Shape([1, cap, mb, 1]), ttnn.float32, ttnn.TILE_LAYOUT, dev, dram)
        sp = ttnn.allocate_tensor_on_device(ttnn.Shape([1, cap, 1, 32]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, dev,
                                            dram)
        blk = ttnn.allocate_tensor_on_device(ttnn.Shape([1, cap, 1, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, dram)
        need = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, 8]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dev, dram)
        out = CompactRows(rows, wcol, sp, blk, need)
        ins = (idx, wsrc, self.slot, meta)
        wflag = 0 if w_is_loc else 1
        key = (M, mb, ladder, wflag) + tuple(tuple(_accessor(t)) for t in ins + tuple(out))
        desc = self._desc.get(key)
        if desc is None:
            cores = _core_range_set(self.E + 1, self.grid[0])
            mt = M // TILE
            cbs = [
                _scratch(0, cores, mt * 2048),
                _scratch(1, cores, mt * 2048),
                _scratch(2, cores, 16384 + cap * mb * 4),  # tables + one expert's token list (<= CAP MB rows)
                _scratch(3, cores, 2 * cap * mb * 4),
                _scratch(4, cores, 6 * 1024),
            ]
            lad = list(ladder) + [0] * (MAX_LADDER - len(ladder))
            ct = ([M, self.top_k, self.E, self.P, self.n_experts, mb, cap, len(ladder)] + lad
                  + [self.grid[0], wflag, int(self.contiguous), 0, 1, 2, 3, 4])
            for t in ins + tuple(out):
                ct += _accessor(t)
            k = ttnn.KernelDescriptor(
                kernel_source=str(DISPATCH_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores, compile_time_args=ct, defines=[("MOTIF_DISPATCH_SRC", self._tag)],
                runtime_args=[], common_runtime_args=[0] * 9, config=ttnn.WriterConfigDescriptor())
            desc = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs)
            _hash(desc, self._hash, tuple(ct))
            self._desc[key] = desc
        desc.kernels[0].common_runtime_args = [t.buffer_address() for t in ins + tuple(out)]
        ttnn.generic_op(list(ins) + list(out), desc)
        return out

    def deallocate(self) -> None:
        if self.slot is not None and self.slot.is_allocated():
            ttnn.deallocate(self.slot)
        self.slot = None
        self._desc.clear()


class GatherCombine:
    """The gather combine (module docstring). No device memory of its own. ``cw``: column tiles per work unit (one
    NoC read of ``cw x 64`` bytes per token row and level)."""

    def __init__(self, mesh_device, *, compute_role: str = "eltwise", cw: int = 8, top_k: int = 8):
        self.mesh_device = mesh_device
        g = mesh_device.compute_with_storage_grid_size()
        self.grid = (int(g.x), int(g.y))
        self.role = compute_role
        self.cw = int(cw)
        self.top_k = int(top_k)
        if not 1 <= self.top_k <= 16:
            raise ValueError(f"gather combine: top_k {top_k} outside [1, 16]")
        if self.cw not in (8, 16, 32):
            raise ValueError(f"gather combine: cw must be 8, 16 or 32, got {cw}")
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _tag([COMBINE_READER, COMBINE_COMPUTE, COMBINE_WRITER])

    def __call__(self, y_rm, keys, *, M: int, key_page: int, out_dtype=ttnn.bfloat16, memory_config=None):
        """``y_rm [1, 1, R, H]`` bf16 ROW_MAJOR (R = NB MB), ``keys``: a uint32 ROW_MAJOR tensor whose page
        ``key_page`` holds at least R keys (row j's token, >= M for pad rows) -> ``part [1, 1, M, H]`` ``out_dtype``
        TILE. A token may have at most ``top_k`` rows (one per routed expert)."""
        M = int(M)
        ys = tuple(int(v) for v in y_rm.shape)
        if len(ys) != 4 or ys[:2] != (1, 1) or ys[2] % TILE or ys[3] % (TILE * self.cw):
            raise ValueError(f"gather combine: y must be [1, 1, R, H] (R a tile multiple, H of {self.cw} tiles), "
                             f"got {ys}")
        if y_rm.dtype != ttnn.bfloat16 or y_rm.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"gather combine: y must be bf16 ROW_MAJOR, got {y_rm.dtype} {y_rm.layout}")
        if keys.dtype != ttnn.uint32 or keys.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"gather combine: keys must be uint32 ROW_MAJOR, got {keys.dtype} {keys.layout}")
        ks = tuple(int(v) for v in keys.shape)
        R, H = ys[2], ys[3]
        kwidth = ks[-1]
        if kwidth < R or not 0 <= int(key_page) < (ks[0] * ks[1] * ks[2]):
            raise ValueError(f"gather combine: keys {ks} page {key_page} cannot hold {R} keys")
        if M % TILE:
            raise ValueError(f"gather combine: M = {M} must be a multiple of {TILE}")
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        part = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, H]), out_dtype, ttnn.TILE_LAYOUT,
                                              self.mesh_device, mc)
        WT = H // TILE
        G = WT // self.cw
        NW = min(self.grid[0] * self.grid[1], (M // TILE) * G)
        key = (M, R, H, int(key_page), kwidth, str(out_dtype), self.cw) + tuple(
            tuple(_accessor(t)) for t in (y_rm, keys, part))
        desc = self._desc.get(key)
        if desc is None:
            from ..model_config import compute_config_descriptor  # lazy: keeps this module's import light

            cores = _core_range_set(NW, self.grid[0])
            out_tile = 4096 if out_dtype == ttnn.float32 else 2048
            cbs = [
                _cb(0, cores, 2048, 2 * self.cw, ttnn.bfloat16),  # cb_rm: 2 levels of cw ROW_MAJOR "tiles"
                _cb(1, cores, 64, 2, ttnn.uint32),  # cb_cnt
                _cb(2, cores, 2048, 1, ttnn.bfloat16),  # cb_z
                _scratch(3, cores, R * 4 + 32 * 16 * 4),  # keys + per-token row lists
                _cb(4, cores, 2048, self.top_k * self.cw, ttnn.bfloat16),  # cb_tl: every level of one unit
                _cb(16, cores, out_tile, 2, out_dtype),  # cb_o
            ]
            rct = ([M, R, WT, G, self.grid[0], NW, 0, 1, 2, 3, int(key_page), kwidth * 4, H * 2, self.top_k]
                   + _accessor(y_rm) + _accessor(keys))
            cct = [M, WT, G, self.grid[0], NW, 0, 1, 2, 4, 16, self.top_k]
            wct = [M, WT, G, self.grid[0], NW, 16] + _accessor(part)
            defines = [("MOTIF_COMBINE_SRC", self._tag)]
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(COMBINE_READER), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores, compile_time_args=rct, defines=defines, runtime_args=[],
                    common_runtime_args=[0, 0], config=ttnn.ReaderConfigDescriptor()),
                ttnn.KernelDescriptor(
                    kernel_source=str(COMBINE_WRITER), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores, compile_time_args=wct, defines=defines, runtime_args=[],
                    common_runtime_args=[0], config=ttnn.WriterConfigDescriptor()),
                ttnn.KernelDescriptor(
                    kernel_source=str(COMBINE_COMPUTE), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=cores, compile_time_args=cct, defines=defines, runtime_args=[],
                    common_runtime_args=[], config=compute_config_descriptor(self.role)),
            ]
            desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
            _hash(desc, self._hash, (tuple(rct), tuple(cct), tuple(wct), NW, str(out_dtype), self.role, self.top_k))
            self._desc[key] = desc
        desc.kernels[0].common_runtime_args = [y_rm.buffer_address(), keys.buffer_address()]
        desc.kernels[1].common_runtime_args = [part.buffer_address()]
        ttnn.generic_op([y_rm, keys, part], desc)
        return part

    def deallocate(self) -> None:
        self._desc.clear()


__all__ = ["CompactDispatch", "CompactRows", "GatherCombine", "chip_meta", "combine_groups", "dispatch_reference",
           "slot_table"]
