# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused grouped MoE PolyNorm for the decode routed experts (B3, docs/OPTIMIZATION_PLAN.md §3.3; prototype
``logs/opt/phaseA/M10``, review docs/OPT_PHASE_A_REVIEW.md "M10 / B3").

One ``ttnn.generic_op`` replaces the composite grouped PolyNorm of the decode MoE (``tt/moe.py``
``MotifMoE.polynorm`` -> ``tt/polynorm.py`` Horner fp32: 2 slices + 1 multiply + 3 multiplies, concat, sum, mac,
rsqrt, 3 slices, 3 mac, multiply; ~127 us at M = 32 and ~193 us at M = 64 traced [M10]) by a single kernel that
reads the gate_up output once and writes ``h`` once (~25 / ~57 us [M10, microbenchmark])::

    gu   [1, E, M, 2 I] fp32 TILE interleaved     the gate_up matmul output (gate | up), E = 12 local experts
    w    [1, E, M, 1]   fp32 TILE interleaved     the routing weights w_loc (0 where a row does not route to e)
    ->
    h    [1, E, M, I]   bf16 TILE interleaved     h = (c0 N(g^3) + c1 N(g^2) + c2 N(g) + b) * (w u)

with ``N(z) = z / sqrt(mean_I(z^2) + eps)``, ``c_k = sigmoid(act_fn.weight)`` and ``b = clamp(act_fn.bias, +-0.5)``
(routed experts only) taken from the layer's :class:`~models.demos.motif3.tt.polynorm.GroupedPolyNormConsts`
(``D, E`` Horner scale constants and the fp32 ``b``: per-chip device tensors, read by the kernel, so one SPMD program
serves all 32 chips). The x0.5 PolyNorm output scale is folded into ``W_down`` exactly as on the composite path, so
the down matmul consumes ``h`` unchanged.

Decomposition (per chip): expert ``e`` is split over ``G`` cores (default 4; worker ``q = GX y + x``, row-major over
the first ``E G`` = 48 cores, ``e = q / G``, rank ``k = q % G``), each owning ``n = I / 32 / G`` = 10 gate and 10 up
tiles of every tile row. Phase 1: fp32 SFPU moment partials ``sum g^2, g^4, g^6`` of its columns (elementwise over its
tiles in order, then the SFPU row reduce). Exchange: every worker writes its 3 R partial tiles (faces 0 and 2, the
ones holding column 0) into slot ``k`` of each group member's receive buffer and increments their semaphores. Phase 3:
every member sums the G partials in rank order 0..G-1 (so all members hold bitwise the same moments),
``a_m = rsqrt(s_m D_m + E_m)``, folds the routing weight (``A_m = w a_m``, ``B = w b``) and evaluates
``h = (((A3 g + A2) g + A1) g + B) u`` in fp32 for its tiles, packing bf16 once.

Numerics: fp32 throughout (moments at the fp32 SFPU, never TF32), one bf16 rounding of ``h``. **Not bitwise equal**
to the composite (different moment summation order and the routing weight folded into the coefficients instead of
``u``): M10 measured 9-14 of 491,520 values differing by one bf16 ulp (max 4.9e-4 at M = 32), PCC vs the composite
0.9999999999997, the same error vs an fp64 golden (PCC 0.99999859). Deterministic: bitwise run to run and identical on
all chips; every output row depends only on its own input row (per-row moments), with the same ``G`` at M = 32 and 64,
so the T64 step's rows equal the T32 rows bitwise (M10 ``t64_rows0_31_bitwise_eq_t32``).

Trace safety: no host round trip; the output is a fresh device allocation per call; the program (one per buffer-type
combination and M) is built on the first eager call and its hash is memoized, so a traced call only patches the
common runtime args (buffer addresses). No persistent L1: receive buffers are per-program CBs, the semaphore is
re-armed by the reader. L1 per worker: ~196 KB of CBs at M = 32, ~340 KB at M = 64 (:func:`plan`).

Host helpers (pure torch / no device): :func:`plan` (layout and CB budget), :func:`golden_fp64` (the HF semantics),
:func:`emulate_fp32` (the kernel's algorithm in fp32 torch: grouping, rank-order sums, coefficient fold, Horner).
Kernel sources: ``moe_polynorm/{reader,writer,compute}.cpp``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "moe_polynorm"
READER_SRC = KERNEL_DIR / "reader.cpp"
WRITER_SRC = KERNEL_DIR / "writer.cpp"
COMPUTE_SRC = KERNEL_DIR / "compute.cpp"

TILE = 32
DEFAULT_GROUP = 4  # cores per expert: the M10 best at M = 32 (25.3 us); M = 64 keeps it (T64 rows == T32 rows)
MAX_ROW_TILES = 2  # M <= 64 (the T32 and T64 decode steps); larger M would need a smaller CB footprint
NOC_TABLE = 16  # NoC coordinate table entries (logical columns / rows) passed to the writer
CB_G, CB_U, CB_W, CB_PART, CB_RECV, CB_COEF, CB_OUT, CB_K = 0, 1, 2, 3, 4, 5, 6, 7
SEM_ID = 0
OUT_TILES = 4  # bf16 output tiles in flight (the compute kernel packs 2 per DEST round)
FP32_TILE_BYTES, BF16_TILE_BYTES = 4096, 2048


def _sources_tag() -> str:
    """Content hash of the kernel sources: a compile define, so an edited kernel never hits a stale JIT build."""
    h = hashlib.sha1()
    for p in (READER_SRC, WRITER_SRC, COMPUTE_SRC):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


# =====================================================================================================================
# host side (pure)
# =====================================================================================================================
def plan(E: int, inter: int, M: int, G: int = DEFAULT_GROUP, grid: Tuple[int, int] = (12, 10)) -> Dict[str, object]:
    """Layout of one call: ``n`` gate tiles per worker and tile row, ``R`` tile rows, the worker count and its grid
    rows, the CB tile counts and the L1 bytes of CBs per worker. Raises ``ValueError`` for an unsupported shape."""
    E, inter, M, G = int(E), int(inter), int(M), int(G)
    gx, gy = int(grid[0]), int(grid[1])
    if M % TILE or not 1 <= M // TILE <= MAX_ROW_TILES:
        raise ValueError(f"fused MoE PolyNorm: M = {M} rows must be 32 or 64")
    if inter % TILE:
        raise ValueError(f"fused MoE PolyNorm: intermediate size {inter} is not a multiple of {TILE}")
    IT = inter // TILE
    if G < 1 or IT % G:
        raise ValueError(f"fused MoE PolyNorm: group size {G} does not divide the {IT} gate tiles")
    n_workers = E * G
    if gx > NOC_TABLE or gx < 1:
        raise ValueError(f"fused MoE PolyNorm: grid width {gx} outside [1, {NOC_TABLE}]")
    rows = -(-n_workers // gx)
    if rows > min(gy, NOC_TABLE):
        raise ValueError(f"fused MoE PolyNorm: {E} experts x G {G} = {n_workers} workers do not fit the {gx} x {gy} grid")
    R = M // TILE
    n = IT // G
    cb_tiles = {
        CB_G: (n * R, FP32_TILE_BYTES),
        CB_U: (n * R, FP32_TILE_BYTES),
        CB_W: (R, FP32_TILE_BYTES),
        CB_PART: (3 * R, FP32_TILE_BYTES),
        CB_RECV: (3 * R * G, FP32_TILE_BYTES),
        CB_COEF: (4, FP32_TILE_BYTES),
        CB_OUT: (OUT_TILES, BF16_TILE_BYTES),
        CB_K: (7, FP32_TILE_BYTES),
    }
    return dict(E=E, inter=inter, M=M, G=G, IT=IT, n=n, R=R, n_workers=n_workers, grid=(gx, gy), rows=rows,
                cb_tiles=cb_tiles, l1_bytes=sum(t * b for t, b in cb_tiles.values()))


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


def golden_fp64(gu: torch.Tensor, w: torch.Tensor, c: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-6):
    """HF semantics in fp64: ``gu [.., E, M, 2 I]``, ``w [.., E, M, 1]``, ``c [E, 3]`` (``c0, c1, c2`` of ``N(g^3),
    N(g^2), N(g)``, sigmoid already applied), ``b [E]`` (already clamped) -> ``(c0 N(g^3) + c1 N(g^2) + c2 N(g) + b)
    * (w u)`` (no x0.5: that scale lives in ``W_down``)."""
    I = gu.shape[-1] // 2
    g, u = gu[..., :I].double(), gu[..., I:].double()
    E = g.shape[-3]
    cc = c.double().reshape(E, 1, 3)
    bb = b.double().reshape(E, 1, 1)

    def N(z):
        return z / torch.sqrt(z.pow(2).mean(-1, keepdim=True) + eps)

    return (cc[..., 0:1] * N(g**3) + cc[..., 1:2] * N(g**2) + cc[..., 2:3] * N(g) + bb) * (w.double() * u)


def _f32(x):
    return x.to(torch.float32)


def emulate_fp32(gu: torch.Tensor, w: torch.Tensor, D: torch.Tensor, Ec: torch.Tensor, b: torch.Tensor, *,
                 G: int = DEFAULT_GROUP) -> torch.Tensor:
    """The kernel's algorithm in fp32 torch (every op rounded to fp32; the SFPU's fused multiply-adds, row-reduce
    order and rsqrt are not modelled bit for bit): ``gu [1, E, M, 2 I]`` fp32, ``w [1, E, M, 1]``, ``D, Ec [E, 3]``
    (moment order ``g^2, g^4, g^6``), ``b [E]`` -> ``h [1, E, M, I]`` bf16.

    Per expert the gate columns split into G contiguous groups of ``n`` tiles; group partials accumulate tile by tile
    (elementwise per (row, column-in-tile) position), are row-reduced, and summed in rank order 0..G-1;
    ``a = rsqrt(s D + E)``, ``A = w a``, ``B = w b``, ``h = (((A3 g + A2) g + A1) g + B) u``."""
    gu = _f32(gu)
    _, E, M, N2 = gu.shape
    I = N2 // 2
    IT = I // TILE
    n = IT // G
    g = gu[0, :, :, :I].reshape(E, M, IT, TILE)
    u = gu[0, :, :, I:]
    s = []
    for k in range(G):
        a2 = torch.zeros(E, M, TILE)
        a4 = torch.zeros(E, M, TILE)
        a6 = torch.zeros(E, M, TILE)
        for j in range(k * n, (k + 1) * n):
            gj = g[:, :, j]
            g2 = gj * gj
            g4 = g2 * g2
            a2 = a2 + g2
            a4 = a4 + g4
            a6 = a6 + g4 * g2
        s.append(torch.stack([a2.sum(-1), a4.sum(-1), a6.sum(-1)], dim=-1))  # [E, M, 3]
    tot = s[0]
    for k in range(1, G):
        tot = tot + s[k]
    a = torch.rsqrt(tot * _f32(D).reshape(E, 1, 3) + _f32(Ec).reshape(E, 1, 3))  # [E, M, 3]
    wv = _f32(w)[0, :, :, 0:1]  # [E, M, 1]
    A = wv * a
    B = wv * _f32(b).reshape(E, 1, 1)
    gg = gu[0, :, :, :I]
    r = A[..., 2:3] * gg + A[..., 1:2]
    r = r * gg + A[..., 0:1]
    r = r * gg + B
    return (r * u).to(torch.bfloat16).unsqueeze(0)


# =====================================================================================================================
# device
# =====================================================================================================================
def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cb(index: int, n_tiles: int, dtype, cores):
    page = {ttnn.bfloat16: BF16_TILE_BYTES, ttnn.float32: FP32_TILE_BYTES}[dtype]
    return ttnn.CBDescriptor(
        total_size=n_tiles * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _is_interleaved(mc) -> bool:
    return mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


class FusedGroupedPolyNorm:
    """The fused grouped PolyNorm of one MoE layer (module docstring). Holds references to the layer's constant
    tensors (``consts.D``, ``consts.E``, ``consts.c["fp32"]["b"]``; not owned) and the cached program descriptors; no
    device memory of its own.

    Args:
        mesh_device: the mesh (every chip runs the same program; the constants differ per chip).
        consts: the layer's :class:`~models.demos.motif3.tt.polynorm.GroupedPolyNormConsts` (fp32 ``D``, ``E``,
            ``b``, TILE).
        e_loc / inter: local experts per chip (12) and the intermediate size (1280).
        group: cores per expert (default :data:`DEFAULT_GROUP`; must divide ``inter / 32``).
        debug: 0 (production) or a diagnostic stage 1..4 of ``compute.cpp``'s ``coef_block`` (h tiles 0..3 of every
            worker's tile rows hold that stage's tiles instead of h).
    """

    def __init__(self, mesh_device, consts, *, e_loc: int, inter: int, group: int = DEFAULT_GROUP, debug: int = 0):
        self.mesh_device = mesh_device
        if debug not in (0, 1, 2, 3, 4):
            raise ValueError(f"fused MoE PolyNorm: debug stage {debug} not in 0..4")
        self.debug = int(debug)  # 0 = production; 1..4 = compute.cpp's coef_block debug stages (diagnostics only)
        self.D = consts.D
        self.Ec = consts.E
        self.b = consts.c["fp32"]["b"]
        for name, t, shape in (("D", self.D, (3, e_loc, 1, 1)), ("E", self.Ec, (3, e_loc, 1, 1)),
                               ("b", self.b, (1, e_loc, 1, 1))):
            if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT or tuple(int(v) for v in t.shape) != shape:
                raise ValueError(f"fused MoE PolyNorm: constant {name} must be fp32 TILE {shape}, got {t.dtype} "
                                 f"{t.layout} {tuple(t.shape)}")
            if not _is_interleaved(t.memory_config()):
                raise ValueError(f"fused MoE PolyNorm: constant {name} must be interleaved")
        self.e_loc, self.inter, self.group = int(e_loc), int(inter), int(group)
        g = mesh_device.compute_with_storage_grid_size()
        self.grid = (int(g.x), int(g.y))
        for M in (TILE, 2 * TILE):  # validate every supported shape up front (no device op)
            plan(self.e_loc, self.inter, M, self.group, self.grid)
        self._noc = None
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _sources_tag()

    # ---- checks ---------------------------------------------------------------------------------------------------
    def supports(self, gu, w) -> bool:
        """True iff ``(gu, w)`` have the call contract (fp32 TILE interleaved ``[1, E, M, 2 I]`` / ``[1, E, M, 1]``,
        M = 32 or 64)."""
        try:
            self._check(gu, w)
            return True
        except ValueError:
            return False

    def _check(self, gu, w) -> int:
        shp = tuple(int(v) for v in gu.shape)
        if len(shp) != 4 or shp[0] != 1 or shp[1] != self.e_loc or shp[3] != 2 * self.inter:
            raise ValueError(f"fused MoE PolyNorm: gu must be [1, {self.e_loc}, M, {2 * self.inter}], got {shp}")
        M = shp[2]
        if M not in (TILE, 2 * TILE):
            raise ValueError(f"fused MoE PolyNorm: M = {M} rows must be 32 or 64")
        if tuple(int(v) for v in w.shape) != (1, self.e_loc, M, 1):
            raise ValueError(f"fused MoE PolyNorm: w must be [1, {self.e_loc}, {M}, 1], got {tuple(w.shape)}")
        for name, t in (("gu", gu), ("w", w)):
            if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT:
                raise ValueError(f"fused MoE PolyNorm: {name} must be fp32 TILE, got {t.dtype} {t.layout}")
            if not _is_interleaved(t.memory_config()):
                raise ValueError(f"fused MoE PolyNorm: {name} must be interleaved")
        return M

    # ---- program --------------------------------------------------------------------------------------------------
    def _noc_tables(self):
        if self._noc is None:
            gx, gy = self.grid
            xs, ys = [0] * NOC_TABLE, [0] * NOC_TABLE
            seen_x, seen_y = [None] * gx, [None] * min(gy, NOC_TABLE)
            for y in range(min(gy, NOC_TABLE)):
                for x in range(gx):
                    c = self.mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                    cx, cy = int(c.x), int(c.y)
                    if seen_x[x] is None:
                        seen_x[x] = cx
                    if seen_y[y] is None:
                        seen_y[y] = cy
                    if (seen_x[x], seen_y[y]) != (cx, cy):
                        raise RuntimeError(f"fused MoE PolyNorm: NoC coordinates not separable at {(x, y)}")
            xs[:gx] = seen_x
            ys[: len(seen_y)] = seen_y
            self._noc = (xs, ys)
        return self._noc

    def _build(self, p, gu, w, h):
        cores = _core_range_set(p["n_workers"], self.grid[0])
        dts = {CB_OUT: ttnn.bfloat16}
        cbs = [_cb(i, t, dts.get(i, ttnn.float32), cores) for i, (t, _) in sorted(p["cb_tiles"].items())]
        sems = [ttnn.SemaphoreDescriptor(id=SEM_ID, core_ranges=cores, initial_value=0)]
        from ..model_config import compute_config_descriptor  # lazy: keeps this module's import light

        cc = compute_config_descriptor("polynorm", fp32_unpack_cbs=[CB_G, CB_U, CB_W, CB_RECV, CB_COEF, CB_K],
                                       dst_full_sync_en=True)
        nx, ny = self._noc_tables()
        defines = [("MOTIF_PN_SRC", self._tag)]
        n, R, G, IT = p["n"], p["R"], p["G"], p["IT"]
        reader_ct = ([CB_G, CB_U, CB_W, CB_RECV, CB_K, SEM_ID, n, R, G, 3 * R * G, IT, self.e_loc, self.grid[0]]
                     + _accessor(gu) + _accessor(w) + _accessor(self.D) + _accessor(self.Ec) + _accessor(self.b))
        writer_ct = [CB_PART, CB_OUT, CB_RECV, SEM_ID, n, R, G, IT, self.grid[0]] + list(nx) + list(ny) + _accessor(h)
        compute_ct = [CB_G, CB_U, CB_W, CB_PART, CB_RECV, CB_COEF, CB_OUT, CB_K, n, R, G, self.debug]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(READER_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores, compile_time_args=reader_ct, defines=defines, runtime_args=[],
                common_runtime_args=[0, 0, 0, 0, 0], config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(WRITER_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores, compile_time_args=writer_ct, defines=defines, runtime_args=[],
                common_runtime_args=[0], config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(
                kernel_source=str(COMPUTE_SRC), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores, compile_time_args=compute_ct, defines=defines, runtime_args=[],
                common_runtime_args=[], config=cc),
        ]
        desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)
        return desc, (tuple(reader_ct), tuple(writer_ct), tuple(compute_ct), p["n_workers"])

    def _program(self, M, gu, w, h):
        """The cached descriptor for these tensors' buffer types and M, patched with this call's addresses (common
        runtime args only: a cache hit rebuilds nothing)."""
        key = (M,) + tuple(tuple(_accessor(t)) for t in (gu, w, h, self.D, self.Ec, self.b))
        desc = self._desc.get(key)
        if desc is None:
            p = plan(self.e_loc, self.inter, M, self.group, self.grid)
            desc, prog_key = self._build(p, gu, w, h)
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hv = self._hash.get(prog_key)
                if hv is None:
                    hv = self._hash[prog_key] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        desc.kernels[0].common_runtime_args = [gu.buffer_address(), w.buffer_address(), self.D.buffer_address(),
                                               self.Ec.buffer_address(), self.b.buffer_address()]
        desc.kernels[1].common_runtime_args = [h.buffer_address()]
        return desc

    def __call__(self, gu, w, *, memory_config=None):
        """``gu [1, E, M, 2 I]`` fp32, ``w [1, E, M, 1]`` fp32 -> ``h [1, E, M, I]`` bf16 TILE in ``memory_config``
        (interleaved; default L1). Consumes nothing."""
        M = self._check(gu, w)
        mc = memory_config or ttnn.L1_MEMORY_CONFIG
        if not _is_interleaved(mc):
            raise ValueError("fused MoE PolyNorm: the output memory_config must be interleaved")
        h = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, self.e_loc, M, self.inter]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh_device, mc
        )
        desc = self._program(M, gu, w, h)
        ttnn.generic_op([gu, w, self.D, self.Ec, self.b, h], desc)
        return h

    def deallocate(self) -> None:
        """Drops the cached descriptors and the constant references (the constants belong to the MoE module)."""
        self._desc.clear()
        self.D = self.Ec = self.b = None


__all__ = [
    "DEFAULT_GROUP",
    "FusedGroupedPolyNorm",
    "emulate_fp32",
    "golden_fp64",
    "plan",
    "worker_cores",
]
