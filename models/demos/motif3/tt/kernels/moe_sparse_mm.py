# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dual-NoC all-core sparse expert matmul for the decode routed experts (phase D, DESIGN-3 stage 1;
``logs/opt/phaseD/DESIGN-3``, judge plan ``logs/opt/phaseD/JUDGE/judge.json``).

Replaces ``ttnn.sparse_matmul`` (1D in0 multicast, in1 read on ONE RISC / NoC per core; ~34.8 + 17.8 us per active
expert, read-pattern bound [M, DESIGN-3]) in :meth:`MotifMoE.sparse_experts` by one ``ttnn.generic_op`` per matmul::

    gate_up  x  [1, 1, M, K]  bf16 TILE  @ W [1, E, K, N] bfp8 TILE, sparsity [1, 1, 1, E] bf16 RM  -> [1, E, M, N]
    down     h  [1, E, M, K]  bf16 TILE  @ W [1, E, K, N] bfp8 TILE (in0 per expert)              -> [1, E, M, N]

with the experts whose sparsity entry is 0 skipped and their output slices exactly 0 (the stock op zero-fills its
output with a separate ``zeros_like``; here the inactive tiles are written by the kernel). Work units are
``(active expert j, output column)`` pairs in expert-major order; worker ``w`` owns a contiguous range of them (at most
2 experts per core), every core of the 12 x 10 grid streams weights on BOTH RISCs / NoCs (RISC r owns K half r: its
in0 half and its weight half), 120 workers. gate_up: core 0 also multicasts ``x`` (each RISC its K half, in chunks)
to every other core; down: each worker reads ``h`` of its <= 2 experts.

Numerics: per output tile the fp32 DEST accumulates the products over the whole K in order (HiFi4, the ``experts``
role), packed once. The stock op accumulates the same products in the same order and spills the fp32 DEST to a Float32
CB reloaded with ``UnpackToDestFp32`` between K blocks (lossless), so the outputs are expected to be bitwise equal
(the device unit test ``tests/unit/test_moe_sparse_mm.py`` checks ``torch.equal`` against ``ttnn.sparse_matmul``).

Trace safety: fresh output allocation per call, descriptors cached per (M, buffer layouts) with a memoized program
hash; a traced call only patches the common runtime args. Semaphores are re-armed by the kernels.
Kernel sources: ``moe_sparse_mm/{dataflow,compute}.cpp``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "moe_sparse_mm"
DATAFLOW_SRC = KERNEL_DIR / "dataflow.cpp"
COMPUTE_SRC = KERNEL_DIR / "compute.cpp"

TILE = 32
KINDS = ("gate_up", "down")
CB_IN0A, CB_IN0B, CB_WA, CB_WB, CB_OUT, CB_SP, CB_CTRL, CB_ZERO = 0, 1, 2, 3, 4, 5, 6, 7
SEM_A, SEM_B = 0, 1
BFP8_TILE_BYTES, BF16_TILE_BYTES, FP32_TILE_BYTES = 1088, 2048, 4096
SP_BYTES = 32  # one sparsity stick (E <= 16 bf16 values), read at L1 / DRAM-safe alignment
CTRL_BYTES = 64
W_BUFS = 4  # weight batches in flight per RISC
MAX_ROW_TILES = 2  # M = 32 (T32) or 64 (T64)


def _sources_tag() -> str:
    h = hashlib.sha1()
    for p in (DATAFLOW_SRC, COMPUTE_SRC):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


# =====================================================================================================================
# host side (pure)
# =====================================================================================================================
def plan(kind: str, E: int, K: int, N: int, M: int, out_bytes: int, grid: Tuple[int, int] = (12, 10)) -> Dict:
    """Layout of one call: tile counts, workers, batch / chunk sizes, CB tile counts and the L1 bytes of CBs per core.
    Raises ``ValueError`` for an unsupported shape."""
    if kind not in KINDS:
        raise ValueError(f"moe_sparse_mm: kind {kind!r} not in {KINDS}")
    E, K, N, M = int(E), int(K), int(N), int(M)
    gx, gy = int(grid[0]), int(grid[1])
    if M % TILE or not 1 <= M // TILE <= MAX_ROW_TILES:
        raise ValueError(f"moe_sparse_mm: M = {M} rows must be 32 or 64")
    if K % (2 * TILE) or N % TILE:
        raise ValueError(f"moe_sparse_mm: K = {K} must be a multiple of 64 and N = {N} of 32")
    if not 1 <= E <= 16:
        raise ValueError(f"moe_sparse_mm: E = {E} experts outside [1, 16]")
    KT, NT, MT = K // TILE, N // TILE, M // TILE
    KH = KT // 2
    ncores = gx * gy
    if ncores < 2:
        raise ValueError("moe_sparse_mm: needs at least 2 cores")
    batch = next(b for b in (8, 10, 5, 4, 2, 1) if KH % b == 0)
    chunk = next(c for c in (8, 4, 2, 1) if KH % c == 0)
    mode = 0 if kind == "gate_up" else 1
    w0 = 0
    nc = ncores - w0
    max_units = -(-E * NT // nc)
    if mode == 1 and max_units > NT:
        raise ValueError(f"moe_sparse_mm: down needs <= 2 experts per core ({max_units} units > NT {NT})")
    seg = KH * MT
    in0_tiles = seg if mode == 0 else 2 * seg
    cb_tiles = {
        CB_IN0A: (in0_tiles, BF16_TILE_BYTES),
        CB_IN0B: (in0_tiles, BF16_TILE_BYTES),
        CB_WA: (W_BUFS * batch, BFP8_TILE_BYTES),
        CB_WB: (W_BUFS * batch, BFP8_TILE_BYTES),
        CB_OUT: (max_units * MT, out_bytes),
        CB_SP: (1, SP_BYTES),
        CB_CTRL: (1, CTRL_BYTES),
        CB_ZERO: (1, out_bytes),
    }
    return dict(kind=kind, mode=mode, E=E, KT=KT, NT=NT, MT=MT, KH=KH, batch=batch, chunk=chunk, w0=w0, nc=nc,
                ncores=ncores, grid=(gx, gy), max_units=max_units, cb_tiles=cb_tiles,
                l1_bytes=sum(t * b for t, b in cb_tiles.values()))


def unit_order(mode: int, w: int, nc: int, k: int, NT: int):
    """Host mirror of the kernels' ``UnitOrder``: the units (``u = j NT + col``, j = index into the active experts)
    worker ``w`` processes, in order, for ``k`` active experts. Mode 0 (gate_up): round-robin ``w + i nc``. Mode 1
    (down): the contiguous range ``[w U / nc, (w + 1) U / nc)``, each expert's sub-range ``[s, s + n)`` rotated to
    start at ``s + (w - s) mod n``."""
    U = int(k) * int(NT)
    if mode == 0:
        return list(range(w, U, nc))
    u0, u1 = w * U // nc, (w + 1) * U // nc
    if u1 <= u0:
        return []
    b = (u0 // NT + 1) * NT
    out = []
    for s, e in ((u0, min(b, u1)), (b, u1)):
        n = e - s
        if n <= 0:
            continue
        r = (w - s) % n
        out += [s + (r + i) % n for i in range(n)]
    return out


def golden(in0, w, sparsity, kind: str):
    """torch reference (fp32 math, not bitwise): ``in0 [1, 1|E, M, K]``, ``w [1, E, K, N]``, ``sparsity [E]`` ->
    ``[1, E, M, N]`` fp32 with the inactive slices 0."""
    import torch

    E = w.shape[1]
    out = torch.zeros(1, E, in0.shape[-2], w.shape[-1])
    for e in range(E):
        if float(sparsity.reshape(-1)[e]) != 0.0:
            a = in0[0, 0] if kind == "gate_up" else in0[0, e]
            out[0, e] = a.float() @ w[0, e].float()
    return out


# =====================================================================================================================
# device
# =====================================================================================================================
def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _is_interleaved(mc) -> bool:
    return mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


_DT = {ttnn.bfloat16: BF16_TILE_BYTES, ttnn.float32: FP32_TILE_BYTES}


class DualNocSparseMM:
    """One expert matmul (``kind`` gate_up or down) of a decode MoE layer. Holds the weight reference (not owned) and
    the cached descriptors; no device memory of its own.

    Args:
        mesh_device: the mesh (every chip runs the same program on its own weights).
        w: ``[1, E, K, N]`` bfp8 TILE interleaved weights.
        kind: ``"gate_up"`` (in0 ``[1, 1, M, K]`` shared) or ``"down"`` (in0 ``[1, E, M, K]`` per expert).
        out_dtype: ``ttnn.float32`` or ``ttnn.bfloat16``.
        compute_role: the compute role (default ``experts``: HiFi4 + fp32 dest acc, as the stock op).
    """

    def __init__(self, mesh_device, w, *, kind: str, out_dtype, compute_role: str = "experts"):
        if kind not in KINDS:
            raise ValueError(f"moe_sparse_mm: kind {kind!r} not in {KINDS}")
        if out_dtype not in _DT:
            raise ValueError(f"moe_sparse_mm: out_dtype {out_dtype} must be float32 or bfloat16")
        shp = tuple(int(v) for v in w.shape)
        if len(shp) != 4 or shp[0] != 1:
            raise ValueError(f"moe_sparse_mm: weights must be [1, E, K, N], got {shp}")
        if w.dtype != ttnn.bfloat8_b or w.layout != ttnn.TILE_LAYOUT or not _is_interleaved(w.memory_config()):
            raise ValueError(f"moe_sparse_mm: weights must be bfp8 TILE interleaved, got {w.dtype} {w.layout}")
        self.mesh_device = mesh_device
        self.w = w
        self.kind = kind
        self.E, self.K, self.N = shp[1], shp[2], shp[3]
        self.out_dtype = out_dtype
        self.compute_role = compute_role
        g = mesh_device.compute_with_storage_grid_size()
        self.grid = (int(g.x), int(g.y))
        for M in (TILE, 2 * TILE):
            plan(kind, self.E, self.K, self.N, M, _DT[out_dtype], self.grid)
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _sources_tag()
        self._mc = None

    # ---- checks ---------------------------------------------------------------------------------------------------
    def supports(self, in0, sparsity) -> bool:
        try:
            self._check(in0, sparsity)
            return True
        except ValueError:
            return False

    def _check(self, in0, sparsity) -> int:
        shp = tuple(int(v) for v in in0.shape)
        e_in = 1 if self.kind == "gate_up" else self.E
        if len(shp) != 4 or shp[0] != 1 or shp[1] != e_in or shp[3] != self.K:
            raise ValueError(f"moe_sparse_mm {self.kind}: in0 must be [1, {e_in}, M, {self.K}], got {shp}")
        M = shp[2]
        if M % TILE or not 1 <= M // TILE <= MAX_ROW_TILES:
            raise ValueError(f"moe_sparse_mm {self.kind}: M = {M} rows must be 32 or 64")
        if in0.dtype != ttnn.bfloat16 or in0.layout != ttnn.TILE_LAYOUT or not _is_interleaved(in0.memory_config()):
            raise ValueError(f"moe_sparse_mm {self.kind}: in0 must be bf16 TILE interleaved")
        sshp = tuple(int(v) for v in sparsity.shape)
        if sshp[-1] != self.E or int(sparsity.logical_volume()) != self.E:
            raise ValueError(f"moe_sparse_mm {self.kind}: sparsity must hold {self.E} values, got {sshp}")
        if sparsity.dtype != ttnn.bfloat16 or sparsity.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(f"moe_sparse_mm {self.kind}: sparsity must be bf16 ROW_MAJOR")
        if not _is_interleaved(sparsity.memory_config()):
            raise ValueError(f"moe_sparse_mm {self.kind}: sparsity must be interleaved")
        return M

    # ---- program --------------------------------------------------------------------------------------------------
    def _mcast_rect(self):
        if self._mc is None:
            gx, gy = self.grid
            a = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
            b = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(gx - 1, gy - 1))
            self._mc = (int(a.x), int(a.y), int(b.x), int(b.y))
        return self._mc

    def _build(self, p, in0, sp, out):
        gx, gy = p["grid"]
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        fmt = {CB_IN0A: ttnn.bfloat16, CB_IN0B: ttnn.bfloat16, CB_WA: ttnn.bfloat8_b, CB_WB: ttnn.bfloat8_b,
               CB_OUT: self.out_dtype, CB_SP: ttnn.bfloat16, CB_CTRL: ttnn.uint32, CB_ZERO: self.out_dtype}
        cbs = [
            ttnn.CBDescriptor(
                total_size=n * page, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt[i], page_size=page)])
            for i, (n, page) in sorted(p["cb_tiles"].items())
        ]
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cores, initial_value=0) for s in (SEM_A, SEM_B)]
        x0, y0, x1, y1 = self._mcast_rect()
        acc = _accessor(self.w) + _accessor(in0) + _accessor(sp) + _accessor(out)
        defines = [("MOTIF_SMM_SRC", self._tag)]

        def dm(risc):
            ct = [p["KT"], p["NT"], p["MT"], p["E"], p["nc"], p["w0"], gx, p["batch"], risc, p["mode"], p["chunk"],
                  CB_IN0A if risc == 0 else CB_IN0B, CB_WA if risc == 0 else CB_WB, CB_OUT, CB_SP, CB_CTRL, CB_ZERO,
                  SEM_A if risc == 0 else SEM_B, x0, y0, x1, y1, p["ncores"] - 1, SP_BYTES] + acc
            cfg = ttnn.ReaderConfigDescriptor() if risc == 0 else ttnn.WriterConfigDescriptor()
            return ttnn.KernelDescriptor(
                kernel_source=str(DATAFLOW_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores, compile_time_args=ct, defines=defines, runtime_args=[],
                common_runtime_args=[0, 0, 0, 0], config=cfg), ct

        from ..model_config import compute_config_descriptor  # lazy: keeps this module's import light

        cc = compute_config_descriptor(self.compute_role)
        compute_ct = [p["KT"], p["NT"], p["MT"], p["nc"], p["w0"], gx, p["batch"], p["mode"], CB_IN0A, CB_IN0B, CB_WA,
                      CB_WB, CB_OUT, CB_CTRL]
        k0, ct0 = dm(0)
        k1, ct1 = dm(1)
        kc = ttnn.KernelDescriptor(
            kernel_source=str(COMPUTE_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=compute_ct, defines=defines, runtime_args=[],
            common_runtime_args=[], config=cc)
        desc = ttnn.ProgramDescriptor(kernels=[k0, k1, kc], semaphores=sems, cbs=cbs)
        return desc, (tuple(ct0), tuple(ct1), tuple(compute_ct), str(self.out_dtype), self.compute_role)

    def _program(self, M, in0, sp, out):
        key = (M,) + tuple(tuple(_accessor(t)) for t in (self.w, in0, sp, out))
        desc = self._desc.get(key)
        if desc is None:
            p = plan(self.kind, self.E, self.K, self.N, M, _DT[self.out_dtype], self.grid)
            desc, prog_key = self._build(p, in0, sp, out)
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hv = self._hash.get(prog_key)
                if hv is None:
                    hv = self._hash[prog_key] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        args = [self.w.buffer_address(), in0.buffer_address(), sp.buffer_address(), out.buffer_address()]
        desc.kernels[0].common_runtime_args = args
        desc.kernels[1].common_runtime_args = args
        return desc

    def __call__(self, in0, sparsity, *, memory_config=None):
        """``in0`` (``[1, 1, M, K]`` gate_up / ``[1, E, M, K]`` down, bf16), ``sparsity`` ``[1, 1, 1, E]`` bf16 RM ->
        ``[1, E, M, N]`` in ``out_dtype`` (inactive slices 0), TILE interleaved in ``memory_config`` (default DRAM).
        Consumes nothing."""
        M = self._check(in0, sparsity)
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        if not _is_interleaved(mc):
            raise ValueError("moe_sparse_mm: the output memory_config must be interleaved")
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.E, M, self.N]), self.out_dtype, ttnn.TILE_LAYOUT, self.mesh_device, mc
        )
        desc = self._program(M, in0, sparsity, out)
        ttnn.generic_op([in0, self.w, sparsity, out], desc)
        return out

    def deallocate(self) -> None:
        self._desc.clear()
        self.w = None


__all__ = ["DualNocSparseMM", "KINDS", "golden", "plan", "unit_order"]
