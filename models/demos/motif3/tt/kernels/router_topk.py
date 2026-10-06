# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decode router tail: bias + top-8 + normalize + local extraction in one ``generic_op`` (B4,
docs/OPTIMIZATION_PLAN.md §3.3 "A5 and B4"; review docs/OPT_PHASE_A_REVIEW.md §7.2 step 5: B4 builds on A5's
gather-free router).

The decode router of ``tt/moe.py`` is ``scores = sigmoid(f @ W^T)`` (one matmul with the sigmoid fused into its program
config; unchanged here) followed by a tail of small ops: the release (``router_mask="gather"``) runs add, concat (-inf
pad), topk, gather, sum, multiply and the local mask (typecast, eq, multiply, sum); A5 (``"scatter"``) replaces gather and
the local mask with scatter, to_layout and multiply / sum ops. Traced at M = 32 the head up to topk is 63 us and the
whole router + mask 140 us (release) / 104 us (A5) per layer (``logs/opt/phaseA/A5/summary.json``). This kernel
replaces everything after the matmul::

    scores [1, 1, M, NE] fp32 TILE interleaved     sigmoid(logits) (the matmul's output)
    bias   [1, 1, 1, NE] fp32 TILE interleaved     expert_bias (selection only)
    ids    [1, E_LOC, 1, 1] fp32 TILE interleaved  this chip's global expert ids (MotifMoE.local_ids; contiguous)
    ->
    w_loc  [1, E_LOC, M, 1] fp32 TILE interleaved  w_loc[e, t] = scale s[t, g] / (sum_k s[t, idx_k] + 1e-20) if this
                                                   chip's expert g = base + e is in token t's top-K, else exactly 0
    (idx   [1, 1, M, K] uint32 ROW_MAJOR, L1       the top-K ids in rank order; optional, for taps / tests)

Decomposition: one worker core per gathered token row (M = 32 or 64 cores, row-major over the compute grid). The
reader fetches the row's NE scores and the bias row (64 B halves of the tile rows) into one fp32 page each; the
compute kernel adds them on the fp32 SFPU with ``add_binary_tile<NearestEven>`` -- the op ``ttnn.add`` runs for two
FLOAT32 operands -- so every selection key is bitwise the release's ``biased``; the writer selects the top-K by exact
integer compares of order-preserving keys (rank = biased descending, then expert id ascending), sums the K unbiased
scores in rank order, takes ``r = 1 / (den + 1e-20)`` and ``w = s r`` (x ``scale`` unless it is 1.0) in IEEE fp32
(soft float on the data-movement RISC, round to nearest even) and writes this chip's E_LOC weights into column 0 of
its row of the w_loc tiles (columns 1..31 = 0).

Numerics: the top-K set equals ``ttnn.topk``'s on every row without an exact fp32 tie at the K-th value (the keys are
the same fp32 values, compared exactly); on an exact tie the lower expert id wins (``ttnn.topk``'s choice there follows
its sort network: the A5 probe saw it pick the lower id on 14 of 16 synthetic 8th / 9th ties). The weights differ
from the gather path's by fp32 rounding (sum order, a correctly rounded division instead of the SFPU reciprocal);
not bitwise equal to the release. Deterministic and identical on all chips; a row's outputs depend only on its own
scores (one worker per row, same code at every M), so the T64 step's rows equal the T32 rows bitwise.

Trace safety: no host round trip; outputs are fresh device allocations per call; the program (one per buffer-type
combination, M, scale and idx flag) is built on the first eager call and its hash memoized, so a traced call only
patches the common runtime args (buffer addresses). L1 per worker: 5 pages of 4 KB (CBs only).

Host helpers (pure torch): :func:`plan`, :func:`emulate_fp32` (the kernel's algorithm bit for bit, except that the
SFPU add is modelled as IEEE fp32 addition), :func:`order_keys`. Kernel sources: ``router_topk/{reader,writer,
compute}.cpp``.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Dict, Tuple

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "router_topk"
READER_SRC = KERNEL_DIR / "reader.cpp"
WRITER_SRC = KERNEL_DIR / "writer.cpp"
COMPUTE_SRC = KERNEL_DIR / "compute.cpp"

TILE = 32
MAX_K = 16
EPS = 1e-20
CB_S, CB_B, CB_O, CB_ID, CB_ST = 0, 1, 2, 3, 4
FP32_TILE_BYTES = 4096


def _sources_tag() -> str:
    """Content hash of the kernel sources: a compile define, so an edited kernel never hits a stale JIT build."""
    h = hashlib.sha1()
    for p in (READER_SRC, WRITER_SRC, COMPUTE_SRC):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


# =====================================================================================================================
# host side (pure)
# =====================================================================================================================
def plan(M: int, n_experts: int, top_k: int, e_loc: int, grid: Tuple[int, int] = (12, 10)) -> Dict[str, object]:
    """Layout of one call (one worker per token row). Raises ``ValueError`` for an unsupported shape."""
    M, NE, K, E_loc = int(M), int(n_experts), int(top_k), int(e_loc)
    gx, gy = int(grid[0]), int(grid[1])
    if M < TILE or M % TILE:
        raise ValueError(f"fused router: M = {M} rows must be a positive multiple of {TILE}")
    if M > gx * gy:
        raise ValueError(f"fused router: M = {M} rows need {M} workers, more than the {gx} x {gy} grid")
    if NE % TILE or NE < TILE:
        raise ValueError(f"fused router: {NE} experts is not a multiple of {TILE}")
    if not 1 <= K <= min(MAX_K, NE):
        raise ValueError(f"fused router: top_k {K} outside [1, {min(MAX_K, NE)}]")
    if E_loc < 1 or 64 * (E_loc + 2) > FP32_TILE_BYTES or E_loc > NE:
        raise ValueError(f"fused router: {E_loc} local experts outside [1, {FP32_TILE_BYTES // 64 - 2}]")
    return dict(M=M, NE=NE, K=K, E_loc=E_loc, R=M // TILE, NT=NE // TILE, n_workers=M, grid=(gx, gy))


def order_keys(x: torch.Tensor) -> torch.Tensor:
    """fp32 values -> int64 keys with the kernel's order (sign-flip map of the bit patterns: exact, total; -0 < +0)."""
    u = x.contiguous().to(torch.float32).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    neg = (u & 0x80000000) != 0
    return torch.where(neg, (~u) & 0xFFFFFFFF, u | 0x80000000)


def emulate_fp32(scores: torch.Tensor, bias: torch.Tensor, *, top_k: int, base: int, e_loc: int,
                 scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """The kernel's algorithm in fp32 torch: ``scores [..., M, NE]`` fp32, ``bias [NE]`` -> ``(w_loc [..., E_loc, M,
    1] fp32, idx [..., M, K] int64 in rank order)``. Rank = (scores + bias) descending (exact fp32 keys), then id
    ascending; ``den`` summed in rank order, ``r = 1 / (den + 1e-20)``, ``w = s r`` (x ``scale`` unless 1.0), every op
    rounded to fp32 (the SFPU add is modelled as IEEE fp32 addition)."""
    s = scores.to(torch.float32)
    lead, M, NE = s.shape[:-2], s.shape[-2], s.shape[-1]
    s2 = s.reshape(-1, NE)
    biased = s2 + bias.to(torch.float32).reshape(1, NE)
    keys = order_keys(biased)
    # descending key, ascending id: sort ids ascending first, then a stable descending sort on the key
    order = torch.sort(keys, dim=-1, descending=True, stable=True).indices
    idx = order[:, :top_k]
    sel = torch.gather(s2, 1, idx)
    den = torch.zeros(s2.shape[0], dtype=torch.float32)
    for j in range(top_k):
        den = den + sel[:, j]
    r = torch.tensor(1.0, dtype=torch.float32) / (den + torch.tensor(EPS, dtype=torch.float32))
    w = sel * r.unsqueeze(1)
    if float(scale) != 1.0:
        w = w * torch.tensor(float(scale), dtype=torch.float32)
    w_loc = torch.zeros(s2.shape[0], e_loc, dtype=torch.float32)
    loc = idx - int(base)
    hit = (loc >= 0) & (loc < e_loc)
    rows = torch.arange(s2.shape[0]).unsqueeze(1).expand_as(idx)
    w_loc[rows[hit], loc[hit]] = w[hit]
    w_loc = w_loc.reshape(*lead, M, e_loc).movedim(-1, -2).unsqueeze(-1)  # [..., E_loc, M, 1]
    return w_loc.contiguous(), idx.reshape(*lead, M, top_k)


def worker_cores(n_workers: int, gx: int):
    """The first ``n_workers`` logical cores in row-major order over ``gx`` columns, as ``(x, y)`` tuples."""
    return [(q % gx, q // gx) for q in range(int(n_workers))]


def _core_range_set(n_workers: int, gx: int):
    full, rem = divmod(int(n_workers), int(gx))
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges)


# =====================================================================================================================
# device
# =====================================================================================================================
def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cb(index: int, cores, page: int = FP32_TILE_BYTES, n_pages: int = 1):
    return ttnn.CBDescriptor(
        total_size=n_pages * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=page)],
    )


def _is_interleaved(mc) -> bool:
    return mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


class FusedRouterTopK:
    """The fused router tail of one MoE layer (module docstring). Holds references to the layer's ``expert_bias``
    (``MotifRouter.bias``) and ``local_ids`` (``MotifMoE.local_ids``; not owned) and the cached program descriptors; no
    device memory of its own.

    Args:
        mesh_device: the mesh (every chip runs the same program; ``ids`` differ per chip).
        bias: ``[1, 1, 1, NE]`` fp32 TILE interleaved.
        local_ids: ``[1, E_LOC, 1, 1]`` fp32 TILE interleaved; element 0 of tile 0 = the chip's first global id (the
            chip's experts are ``[base, base + E_LOC)``, ``weights.ep_layout``).
        top_k: experts per token (8).
    """

    def __init__(self, mesh_device, bias, local_ids, *, top_k: int):
        self.mesh_device = mesh_device
        self.bias, self.ids = bias, local_ids
        bshape = tuple(int(v) for v in bias.shape)
        ishape = tuple(int(v) for v in local_ids.shape)
        if len(bshape) != 4 or bshape[:3] != (1, 1, 1):
            raise ValueError(f"fused router: bias must be [1, 1, 1, NE], got {bshape}")
        if len(ishape) != 4 or ishape[0] != 1 or ishape[2:] != (1, 1):
            raise ValueError(f"fused router: local ids must be [1, E_LOC, 1, 1], got {ishape}")
        for name, t in (("bias", bias), ("local ids", local_ids)):
            if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT or not _is_interleaved(t.memory_config()):
                raise ValueError(f"fused router: {name} must be fp32 TILE interleaved, got {t.dtype} {t.layout}")
        self.n_experts, self.e_loc, self.top_k = bshape[3], ishape[1], int(top_k)
        g = mesh_device.compute_with_storage_grid_size()
        self.grid = (int(g.x), int(g.y))
        plan(TILE, self.n_experts, self.top_k, self.e_loc, self.grid)  # validate the shape up front (no device op)
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _sources_tag()

    def supports(self, scores) -> bool:
        """True iff ``scores`` has the call contract (fp32 TILE interleaved ``[1, 1, M, NE]``, M placeable)."""
        try:
            self._check(scores)
            return True
        except ValueError:
            return False

    def _check(self, scores) -> int:
        shp = tuple(int(v) for v in scores.shape)
        if len(shp) != 4 or shp[:2] != (1, 1) or shp[3] != self.n_experts:
            raise ValueError(f"fused router: scores must be [1, 1, M, {self.n_experts}], got {shp}")
        if scores.dtype != ttnn.float32 or scores.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"fused router: scores must be fp32 TILE, got {scores.dtype} {scores.layout}")
        if not _is_interleaved(scores.memory_config()):
            raise ValueError("fused router: scores must be interleaved")
        plan(shp[2], self.n_experts, self.top_k, self.e_loc, self.grid)
        return shp[2]

    def _build(self, p, scale_bits: int, ios):
        scores, w_loc, idx = ios
        cores = _core_range_set(p["n_workers"], self.grid[0])
        cbs = [_cb(CB_S, cores), _cb(CB_B, cores), _cb(CB_O, cores), _cb(CB_ID, cores), _cb(CB_ST, cores)]
        from ..model_config import compute_config_descriptor  # lazy: keeps this module's import light

        cc = compute_config_descriptor("router", fp32_unpack_cbs=[CB_S, CB_B])
        defines = [("MOTIF_RT_SRC", self._tag)]
        gx = self.grid[0]
        reader_ct = [CB_S, CB_B, CB_ID, p["NT"], gx] + _accessor(scores) + _accessor(self.bias) + _accessor(self.ids)
        writer_ct = ([CB_S, CB_O, CB_ID, CB_ST, p["NE"], p["K"], p["E_loc"], p["R"], gx, int(scale_bits),
                      int(idx is not None)] + _accessor(w_loc) + _accessor(idx if idx is not None else w_loc))
        # (without idx the w_loc accessor is repeated as a placeholder: the writer's idx accessor is parsed either way)
        compute_ct = [CB_S, CB_B, CB_O]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(READER_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores, compile_time_args=reader_ct, defines=defines, runtime_args=[],
                common_runtime_args=[0, 0, 0], config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(WRITER_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores, compile_time_args=writer_ct, defines=defines, runtime_args=[],
                common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(COMPUTE_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores, compile_time_args=compute_ct, defines=defines, runtime_args=[],
                common_runtime_args=[], config=cc),
        ]
        desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return desc, (tuple(reader_ct), tuple(writer_ct), tuple(compute_ct), p["n_workers"])

    def _program(self, M, scale_bits, scores, w_loc, idx):
        """The cached descriptor for these tensors' buffer types, M, scale and idx flag, patched with this call's
        addresses (common runtime args only: a cache hit rebuilds nothing)."""
        ts = (scores, w_loc, self.bias, self.ids) + ((idx,) if idx is not None else ())
        key = (M, scale_bits, idx is not None) + tuple(tuple(_accessor(t)) for t in ts)
        desc = self._desc.get(key)
        if desc is None:
            p = plan(M, self.n_experts, self.top_k, self.e_loc, self.grid)
            desc, prog_key = self._build(p, scale_bits, (scores, w_loc, idx))
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hv = self._hash.get(prog_key)
                if hv is None:
                    hv = self._hash[prog_key] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        desc.kernels[0].common_runtime_args = [scores.buffer_address(), self.bias.buffer_address(),
                                               self.ids.buffer_address()]
        desc.kernels[1].common_runtime_args = [w_loc.buffer_address(), idx.buffer_address() if idx is not None else 0]
        return desc

    def __call__(self, scores, *, scale: float = 1.0, memory_config=None, want_idx: bool = False):
        """``scores [1, 1, M, NE]`` fp32 -> ``w_loc [1, E_LOC, M, 1]`` fp32 TILE in ``memory_config`` (interleaved;
        default L1), or ``(w_loc, idx)`` with ``want_idx`` (``idx [1, 1, M, K]`` uint32 ROW_MAJOR in L1). Consumes
        nothing."""
        M = self._check(scores)
        mc = memory_config or ttnn.L1_MEMORY_CONFIG
        if not _is_interleaved(mc):
            raise ValueError("fused router: the output memory_config must be interleaved")
        w_loc = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.e_loc, M, 1]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh_device, mc
        )
        idx = None
        if want_idx:
            idx = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, M, self.top_k]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self.mesh_device,
                ttnn.L1_MEMORY_CONFIG,
            )
        desc = self._program(M, _f32_bits(scale), scores, w_loc, idx)
        ios = [scores, self.bias, self.ids] + ([idx] if idx is not None else []) + [w_loc]
        ttnn.generic_op(ios, desc)
        return (w_loc, idx) if want_idx else w_loc

    def deallocate(self) -> None:
        """Drops the cached descriptors and the constant references (the constants belong to the MoE module)."""
        self._desc.clear()
        self.bias = self.ids = None


__all__ = ["FusedRouterTopK", "emulate_fp32", "order_keys", "plan", "worker_cores"]
