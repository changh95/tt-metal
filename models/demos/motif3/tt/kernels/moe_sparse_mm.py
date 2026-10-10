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

Stage 2 (phase E, ``MOTIF3_DECODE_EXPERT_MM=fused``; judge plan ``logs/opt/phaseD/JUDGE/judge.json``): the same two
kernels with two compile-time modes that remove the ops around them in :meth:`MotifMoE.sparse_experts`::

    gate_up_routed  x [1, 1, M, K], w_loc [1, E, M, 1] fp32 TILE  -> gu [1, E, M, N], sp [1, 1, 1, 16] bf16 RM
    down_sum        h [1, E, M, K], sp                             -> part [1, 1, M, N] (= sum_e y[e])

* ``SP_MODE`` (gate_up): core 0 reads column 0 of the (lane-masked) routing weights, an expert is active when any row's
  weight is nonzero, and multicasts the stick to every core (also written to ``sp`` for the down call). Replaces
  :meth:`MotifMoE.decode_sparsity` (max + typecast + to_layout + reshape, 4 ops). Same active set as the stock path
  except for a weight in (0, ~1e-38] (bf16 rounds it to 0 there; routing weights are normalized top-8 scores).
* ``SUM_MODE`` (down): the expert sum inside the kernel. Each output column is owned by one core; producers write their
  bf16 ``y`` tiles into the owner's L1 staging CB and bump its semaphore; the owner adds the E tiles in expert order
  into the fp32 DEST with the same ``add_tiles(y_e, zero)`` / ``acc_to_dest`` sequence as ``fast_reduce_nc`` (an inactive
  expert adds a zero tile, as the stock ``y`` holds zeros), packed once in the part dtype. Replaces the ``y`` tensor,
  the zero-fill and :meth:`MotifMoE.reduce_experts`.
* ``H_MC`` (down, with ``SUM_MODE``; the 18.8 us intercept, D-3): every worker of stage 1 read the ``h`` of its <= 2
  experts from L1 (k = 1: 120 cores x 80 KB from the 40 banks of one expert, the larger part of a down call). Now the
  first worker whose first expert is j reads ``h[j]`` once and multicasts it to the other workers of j (a contiguous
  core range: up to 3 rectangles of the 12 x 10 grid, data then a flag semaphore per RISC and segment), plus a unicast
  to the range's first worker when j is its second expert. Receivers keep streaming weights while ``h`` is in flight.
  Data movement only (bitwise the same).

Trace safety: fresh output allocation per call, descriptors cached per (M, mode, buffer layouts) with a memoized program
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
CB_IN0A, CB_IN0B, CB_WA, CB_WB, CB_OUT, CB_SP, CB_CTRL, CB_ZERO, CB_STAGE, CB_PART = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
SEM_A, SEM_B, SEM_AUX, SEM_H = 0, 1, 2, 3  # SEM_H .. SEM_H + 3: the down h multicast flags (RISC r, segment s)
SP_STICK = 16  # the stage-2 sparsity stick: [1, 1, 1, 16] bf16 RM (32 B, one aligned page)
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
def plan(kind: str, E: int, K: int, N: int, M: int, out_bytes: int, grid: Tuple[int, int] = (12, 10), *,
         sp_mode: int = 0, sum_mode: int = 0, part_bytes: int = BF16_TILE_BYTES) -> Dict:
    """Layout of one call: tile counts, workers, batch / chunk sizes, CB tile counts and the L1 bytes of CBs per core.
    ``sp_mode`` (gate_up) / ``sum_mode`` (down, bf16 ``y`` tiles; ``part_bytes`` = the part tile) select stage 2.
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
    if sp_mode and mode != 0:
        raise ValueError("moe_sparse_mm: the in-kernel sparsity (sp_mode) belongs to gate_up")
    if sum_mode and (mode != 1 or out_bytes != BF16_TILE_BYTES):
        raise ValueError("moe_sparse_mm: the in-kernel expert sum (sum_mode) belongs to down with bf16 y tiles")
    seg = KH * MT
    in0_tiles = seg if mode == 0 else 2 * seg
    if sp_mode and in0_tiles * BF16_TILE_BYTES < E * MT * 2048:
        raise ValueError("moe_sparse_mm: cb_in0b is too small to hold the w_loc faces")
    slots = -(-NT // ncores) if sum_mode else 0
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
    if sum_mode:
        cb_tiles[CB_STAGE] = (slots * E * MT, out_bytes)
        cb_tiles[CB_PART] = (2, int(part_bytes))
    return dict(kind=kind, mode=mode, E=E, KT=KT, NT=NT, MT=MT, KH=KH, batch=batch, chunk=chunk, w0=w0, nc=nc,
                ncores=ncores, grid=(gx, gy), max_units=max_units, cb_tiles=cb_tiles, sp_mode=int(bool(sp_mode)),
                sum_mode=int(bool(sum_mode)), slots=slots,
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


def h_mcast_plan(k: int, NT: int, nc: int) -> Dict[int, Dict]:
    """Host mirror of the down kernel's H_MC for ``k`` active experts: per expert j ``{"workers": (wa, wb), "sender": s0
    or None, "mcast": (s0 + 1, wb) or None (their segment 0), "uni": wa or None (its segment 1)}``; with no sender the
    single worker wa reads h itself (``self_read``)."""
    U = int(k) * int(NT)
    out = {}
    if U == 0:
        return out

    def worker_of(u):
        return ((u + 1) * nc + U - 1) // U - 1

    for j in range(int(k)):
        wa, wb = worker_of(j * NT), worker_of((j + 1) * NT - 1)
        wa_first = wa * U // nc >= j * NT
        s0 = wa if wa_first else wa + 1
        if s0 > wb:
            out[j] = dict(workers=(wa, wb), sender=None, mcast=None, uni=None, self_read=wa)
        else:
            out[j] = dict(workers=(wa, wb), sender=s0, mcast=(s0 + 1, wb) if s0 < wb else None,
                          uni=None if wa_first else wa, self_read=None)
    return out


def owner_of(col: int, ncores: int) -> Tuple[int, int]:
    """Stage 2 (sum_mode): the core index that sums output column ``col`` and its slot there (host mirror)."""
    return int(col) % int(ncores), int(col) // int(ncores)


def sparsity_of(w_loc) -> list:
    """Host mirror of the stage-2 in-kernel sparsity: ``w_loc [1, E, M, 1]`` (torch fp32) -> active expert ids (any row
    with a nonzero weight bit pattern apart from the sign)."""
    import torch

    b = w_loc.float().contiguous().view(torch.int32) & 0x7FFFFFFF
    return [e for e in range(w_loc.shape[1]) if bool((b[0, e] != 0).any())]


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
        w_rep: DESIGN-2 replica slots (``MOTIF3_MOE_REPLICAS``, tt/replicas.py): ``[1, R, K, N]`` bfp8 TILE interleaved
            weights of R more local experts; the op then serves ``E = E_nat + R`` experts, ``[0, E_nat)`` from ``w`` and
            ``[E_nat, E)`` from ``w_rep`` (only the weight reads differ: per output tile the same full-K fp32 DEST
            accumulation, so a native expert's tiles are bitwise the same with or without ``w_rep``).
    """

    def __init__(self, mesh_device, w, *, kind: str, out_dtype, compute_role: str = "experts", w_rep=None):
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
        self.E_nat = self.E
        self.w_rep = w_rep
        if w_rep is not None:
            rshp = tuple(int(v) for v in w_rep.shape)
            if len(rshp) != 4 or rshp[0] != 1 or rshp[2:] != shp[2:]:
                raise ValueError(f"moe_sparse_mm: replica weights must be [1, R, {shp[2]}, {shp[3]}], got {rshp}")
            if w_rep.dtype != ttnn.bfloat8_b or w_rep.layout != ttnn.TILE_LAYOUT \
                    or not _is_interleaved(w_rep.memory_config()):
                raise ValueError("moe_sparse_mm: replica weights must be bfp8 TILE interleaved")
            self.E = self.E_nat + rshp[1]
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
        self._xy = None
        self.h_mcast = True  # stage 2 down: h read once per expert and multicast to its workers (False: per-core reads)

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

    def supports_routed(self, x, w_loc) -> bool:
        """Stage 2 gate_up: ``x`` as :meth:`__call__` and ``w_loc`` ``[1, E, M, 1]`` fp32 TILE interleaved."""
        if self.kind != "gate_up":
            return False
        try:
            M = self._check_in0(x)
            self._check_wloc(w_loc, M)
            return True
        except ValueError:
            return False

    def _check_in0(self, in0) -> int:
        shp = tuple(int(v) for v in in0.shape)
        e_in = 1 if self.kind == "gate_up" else self.E
        if len(shp) != 4 or shp[0] != 1 or shp[1] != e_in or shp[3] != self.K:
            raise ValueError(f"moe_sparse_mm {self.kind}: in0 must be [1, {e_in}, M, {self.K}], got {shp}")
        M = shp[2]
        if M % TILE or not 1 <= M // TILE <= MAX_ROW_TILES:
            raise ValueError(f"moe_sparse_mm {self.kind}: M = {M} rows must be 32 or 64")
        if in0.dtype != ttnn.bfloat16 or in0.layout != ttnn.TILE_LAYOUT or not _is_interleaved(in0.memory_config()):
            raise ValueError(f"moe_sparse_mm {self.kind}: in0 must be bf16 TILE interleaved")
        return M

    def _check_wloc(self, w_loc, M: int) -> None:
        shp = tuple(int(v) for v in w_loc.shape)
        if shp != (1, self.E, M, 1):
            raise ValueError(f"moe_sparse_mm: w_loc must be [1, {self.E}, {M}, 1], got {shp}")
        if w_loc.dtype != ttnn.float32 or w_loc.layout != ttnn.TILE_LAYOUT or not _is_interleaved(w_loc.memory_config()):
            raise ValueError("moe_sparse_mm: w_loc must be fp32 TILE interleaved")

    def _check_stick(self, sp) -> None:
        if tuple(int(v) for v in sp.shape) != (1, 1, 1, SP_STICK) or sp.dtype != ttnn.bfloat16 \
                or sp.layout != ttnn.ROW_MAJOR_LAYOUT or not _is_interleaved(sp.memory_config()):
            raise ValueError(f"moe_sparse_mm: the stage-2 sparsity must be [1, 1, 1, {SP_STICK}] bf16 RM interleaved")

    # ---- program --------------------------------------------------------------------------------------------------
    def _mcast_rect(self):
        if self._mc is None:
            gx, gy = self.grid
            a = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
            b = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(gx - 1, gy - 1))
            self._mc = (int(a.x), int(a.y), int(b.x), int(b.y))
        return self._mc

    def _core_xy(self):
        """Stage 2: the NOC coordinates ``x << 16 | y`` of logical core index ``c = y GX + x`` (owner table)."""
        if self._xy is None:
            gx, gy = self.grid
            xy = []
            for c in range(gx * gy):
                v = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(c % gx, c // gx))
                xy.append((int(v.x) << 16) | int(v.y))
            self._xy = xy
        return self._xy

    def _build(self, p, in0, sp, out, sp_out=None):
        gx, gy = p["grid"]
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        fmt = {CB_IN0A: ttnn.bfloat16, CB_IN0B: ttnn.bfloat16, CB_WA: ttnn.bfloat8_b, CB_WB: ttnn.bfloat8_b,
               CB_OUT: self.out_dtype, CB_SP: ttnn.bfloat16, CB_CTRL: ttnn.uint32, CB_ZERO: self.out_dtype}
        stage2 = p["sp_mode"] or p["sum_mode"]
        if p["sum_mode"]:
            fmt[CB_ZERO] = ttnn.bfloat16
            fmt[CB_STAGE] = ttnn.bfloat16
            fmt[CB_PART] = out.dtype
        cbs = [
            ttnn.CBDescriptor(
                total_size=n * page, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt[i], page_size=page)])
            for i, (n, page) in sorted(p["cb_tiles"].items())
        ]
        sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=cores, initial_value=0)
                for s in ((SEM_A, SEM_B, SEM_AUX) + ((SEM_H, SEM_H + 1, SEM_H + 2, SEM_H + 3) if p["sum_mode"] else ())
                          if stage2 else (SEM_A, SEM_B))]
        x0, y0, x1, y1 = self._mcast_rect()
        acc = _accessor(self.w) + _accessor(in0) + _accessor(sp) + _accessor(out)
        acc += _accessor(sp_out if p["sp_mode"] else sp)  # the kernel always parses 6 accessors (dummy: sp)
        acc += _accessor(self.w_rep if self.w_rep is not None else self.w)  # (dummy: w)
        acc += [self.E_nat]
        defines = [("MOTIF_SMM_SRC", self._tag)]

        def dm(risc):
            ct = [p["KT"], p["NT"], p["MT"], p["E"], p["nc"], p["w0"], gx, p["batch"], risc, p["mode"], p["chunk"],
                  CB_IN0A if risc == 0 else CB_IN0B, CB_WA if risc == 0 else CB_WB, CB_OUT, CB_SP, CB_CTRL, CB_ZERO,
                  SEM_A if risc == 0 else SEM_B, x0, y0, x1, y1, p["ncores"] - 1, SP_BYTES, p["sp_mode"],
                  p["sum_mode"], CB_STAGE, CB_PART, SEM_AUX, CB_IN0B, p["slots"], int(bool(self.h_mcast)), SEM_H] + acc
            cfg = ttnn.ReaderConfigDescriptor() if risc == 0 else ttnn.WriterConfigDescriptor()
            return ttnn.KernelDescriptor(
                kernel_source=str(DATAFLOW_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores, compile_time_args=ct, defines=defines, runtime_args=[],
                common_runtime_args=[0] * self._n_common(p), config=cfg), ct

        from ..model_config import compute_config_descriptor  # lazy: keeps this module's import light

        cc = compute_config_descriptor(self.compute_role)
        compute_ct = [p["KT"], p["NT"], p["MT"], p["nc"], p["w0"], gx, p["batch"], p["mode"], CB_IN0A, CB_IN0B, CB_WA,
                      CB_WB, CB_OUT, CB_CTRL, p["E"], p["sum_mode"], CB_STAGE, CB_ZERO, CB_PART, p["slots"], p["ncores"]]
        k0, ct0 = dm(0)
        k1, ct1 = dm(1)
        kc = ttnn.KernelDescriptor(
            kernel_source=str(COMPUTE_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cores, compile_time_args=compute_ct, defines=defines, runtime_args=[],
            common_runtime_args=[], config=cc)
        desc = ttnn.ProgramDescriptor(kernels=[k0, k1, kc], semaphores=sems, cbs=cbs)
        return desc, (tuple(ct0), tuple(ct1), tuple(compute_ct), str(self.out_dtype), str(out.dtype), self.compute_role,
                      self.E_nat)

    @staticmethod
    def _n_common(p) -> int:
        return 6 + (p["ncores"] if p["sum_mode"] else 0)

    def _program(self, M, in0, sp, out, *, sp_out=None, sp_mode=0, sum_mode=0):
        ts = (self.w, in0, sp, out) + ((sp_out,) if sp_mode else ()) + ((self.w_rep,) if self.w_rep is not None else ())
        key = (M, sp_mode, sum_mode, str(out.dtype), bool(self.h_mcast), self.E_nat) + tuple(
            tuple(_accessor(t)) for t in ts)
        desc = self._desc.get(key)
        if desc is None:
            p = plan(self.kind, self.E, self.K, self.N, M, _DT[self.out_dtype], self.grid, sp_mode=sp_mode,
                     sum_mode=sum_mode, part_bytes=_DT.get(out.dtype, BF16_TILE_BYTES))
            desc, prog_key = self._build(p, in0, sp, out, sp_out)
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hv = self._hash.get(prog_key)
                if hv is None:
                    hv = self._hash[prog_key] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        args = [self.w.buffer_address(), in0.buffer_address(), sp.buffer_address(), out.buffer_address(),
                sp_out.buffer_address() if sp_mode else 0]
        if sum_mode:
            args += self._core_xy()
        args += [self.w_rep.buffer_address() if self.w_rep is not None else 0]
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
        ttnn.generic_op([in0, self.w, sparsity] + self._rep_io() + [out], desc)
        return out

    def gate_up_routed(self, x, w_loc, *, memory_config=None):
        """Stage 2 gate_up: ``x [1, 1, M, K]`` bf16, ``w_loc [1, E, M, 1]`` fp32 TILE (lane-masked routing weights) ->
        ``(gu [1, E, M, N] out_dtype with the inactive slices 0, sp [1, 1, 1, 16] bf16 RM)``; ``sp`` (1.0 = active) is
        :meth:`down_sum`'s sparsity. Consumes nothing."""
        if self.kind != "gate_up":
            raise ValueError("moe_sparse_mm: gate_up_routed needs a gate_up op")
        M = self._check_in0(x)
        self._check_wloc(w_loc, M)
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        if not _is_interleaved(mc):
            raise ValueError("moe_sparse_mm: the output memory_config must be interleaved")
        sp = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, SP_STICK]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT,
                                            self.mesh_device, mc)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.E, M, self.N]), self.out_dtype, ttnn.TILE_LAYOUT, self.mesh_device, mc
        )
        desc = self._program(M, x, w_loc, out, sp_out=sp, sp_mode=1)
        ttnn.generic_op([x, self.w, w_loc, sp] + self._rep_io() + [out], desc)
        return out, sp

    def down_sum(self, h, sp, *, part_dtype=None, memory_config=None):
        """Stage 2 down: ``h [1, E, M, K]`` bf16, ``sp`` (:meth:`gate_up_routed`'s stick) -> ``part [1, 1, M, N]`` =
        ``sum_e y[e]`` in ``part_dtype`` (default bf16), bitwise ``fast_reduce_nc(down(h), dims=[1])``. Consumes
        nothing."""
        if self.kind != "down" or self.out_dtype != ttnn.bfloat16:
            raise ValueError("moe_sparse_mm: down_sum needs a down op with bf16 y tiles")
        pd = part_dtype or ttnn.bfloat16
        if pd not in _DT:
            raise ValueError(f"moe_sparse_mm: part dtype {pd} must be float32 or bfloat16")
        M = self._check_in0(h)
        self._check_stick(sp)
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        if not _is_interleaved(mc):
            raise ValueError("moe_sparse_mm: the output memory_config must be interleaved")
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, M, self.N]), pd, ttnn.TILE_LAYOUT, self.mesh_device, mc)
        desc = self._program(M, h, sp, out, sum_mode=1)
        ttnn.generic_op([h, self.w, sp] + self._rep_io() + [out], desc)
        return out

    def _rep_io(self) -> list:
        return [self.w_rep] if self.w_rep is not None else []

    def deallocate(self) -> None:
        self._desc.clear()
        self.w = None
        self.w_rep = None


__all__ = ["DualNocSparseMM", "KINDS", "SP_STICK", "golden", "h_mcast_plan", "owner_of", "plan", "sparsity_of",
           "unit_order"]
