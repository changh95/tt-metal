# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused decode PolyNorm of the MoE shared expert (B5, docs/OPTIMIZATION_PLAN.md §3.3; review
docs/OPT_PHASE_A_REVIEW.md §7; reuses B3's ``kernels/moe_polynorm`` constant-tile, non-finite-guard and Horner patterns).

The shared expert's intermediate (I = 1280) is TP8-sharded: each chip owns ``n = I / 8 / 32 = 5`` gate and 5 up tiles
of one 32-row tile row (decode pads its 8 or 16 lanes to 32 rows). The release (``tt/polynorm.py`` ``polynorm_tp``,
decode defaults ``moments="sum", ar="ag_sum", horner="mac"``) runs 19 small programs between the gate_up and down
matmuls: 2 slices, 3 multiplies, concat, sum, pad, the moments all-gather, sum, mac, rsqrt, 3 slices, 3 mac and the
up multiply (73.8 us traced per layer, ``tt/polynorm.py`` docstring). B5 replaces them with three::

    s   = moments(gu)            one core: [sum g^2, g^4, g^6] of this chip's gate columns -> [1, 3, 32, 32] fp32
    gat = ccl.all_gather(s, 3, "tp")                       the release's moments gather, unchanged ([1, 3, 32, 256])
    h   = apply(gat, gu)         one core: sum the 8 partials, a_m = rsqrt(s_m D_m + E_m), Horner, * up -> bf16

**Bitwise equal to the release by construction.** Both kernels issue the LLK calls the release's ttnn ops run, in the
same order and with the same operands: the fp32 SFPU multiplies of ``_moment_sums``; the accurate fp32 ``ttnn.sum``
(sequential ``add_binary_tile`` fold over the tiles of a row, then ``sfpu_reduce`` REDUCE_ROW); the gathered sum (the
same sequential SFPADD fold, in SFPI on the two faces holding column 0; the release's zero padding adds exact zeros);
one SFPMAD for ``s D + E`` (what ``mac_tile`` runs); ``rsqrt_tile``
(default mode, approx off, fp32 DEST); the Horner chain ``g a3 + a2``, ``t g + a1``, ``t g + b`` as SFPMADs and
``t u`` as an SFPU multiply followed by the ``typecast_tile<Float32, Float16_b>`` post-activation the release's
``multiply(.., dtype=bf16)`` runs (round to nearest even; a plain fp32 -> bf16 pack rounds exact ties away from zero),
packed to bf16 once. Every input is unpacked to DEST in exact fp32, as in the release
(binary_ng / ternary / unary / reduce all set ``UnpackToDestFp32`` for fp32 operands). Gate:
``tests/unit/test_mlp.py::test_mlp_device_fused_shared_polynorm`` (fused == composite bitwise on real weights).

Shapes (per chip; TILE, interleaved; nothing is consumed)::

    gu  [1, 1, 32, 2 n 32] fp32     the decode gate_up output (gate | up), ``PolyNormMLP._rows``
    s   [1, 3, 32, 32]     fp32     moment m in column 0 of tile m (columns 1..31: don't care)
    gat [1, 3, 32, 32 TP]  fp32     ``ccl.all_gather(s, 3, "tp")``
    h   [1, 1, 32, n 32]   bf16     the down matmul's input

Constants: the layer's :class:`~models.demos.motif3.tt.polynorm.ScalarPolyNormConsts` (``D``, ``E`` ``[1, 3, 1, 1]``
fp32, ``b["fp32"]`` ``[1, 1, 1, 1]``; replicated), read by the apply reader from the device tensors (their addresses
are common runtime args), so one program serves all 51 shared experts.

Measured (TORUS_XY, real layer-2 weights, traced; ``test_mlp_device_fused_shared_polynorm``, logs/opt/phaseB/B5):
PolyNorm 80.2 -> 37.0 us per layer (moments 6.8 + the gather 13.9 + apply ~15); ``forward_decode`` partial
104.8 -> 61.9 us; bitwise == the composite on all 51 shared experts.

Trace safety: no host round trip; outputs are fresh device allocations; each program (per buffer-type combination)
is built on the first eager call and its hash memoized, so traced calls only patch common runtime args (addresses).
L1: one worker core, 32 KB of CBs (moments) and 184 KB (apply) at n = 5, TP = 8 (:func:`plan`).

Host helpers (pure torch): :func:`plan`, :func:`golden_fp64` (HF semantics), :func:`emulate_fp32` (the release /
kernel algorithm in fp32 torch; the SFPU row reduce's internal order is not modelled bit for bit).
Kernel sources: ``shared_polynorm/{moments_reader,moments_compute,apply_reader,apply_compute,writer}.cpp``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Tuple

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "shared_polynorm"
SOURCES = {
    name: KERNEL_DIR / f"{name}.cpp"
    for name in ("moments_reader", "moments_compute", "apply_reader", "apply_compute", "writer")
}

TILE = 32
MAX_TILES = 8  # gate tiles per chip the single-core kernels hold (shared expert: 5; the dense MLP's 48 do not fit)
MAX_TP = 8  # the apply kernel folds one moment's TP partial tiles in DEST (8 fp32 tiles)
WORKER = (0, 0)  # the one worker core (logical) of both programs
# moments program CBs
CB_G, CB_PART = 0, 1
# apply program CBs
CB_RECV, CB_K, CB_AG, CB_AU, CB_COEF, CB_OUT = 0, 1, 2, 3, 4, 5
FP32_TILE_BYTES, BF16_TILE_BYTES = 4096, 2048
OUT_TILES = 2  # bf16 output tiles in flight (2 per DEST round)


def _sources_tag() -> str:
    """Content hash of the kernel sources: a compile define, so an edited kernel never hits a stale JIT build."""
    h = hashlib.sha1()
    for p in SOURCES.values():
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


# =====================================================================================================================
# host side (pure)
# =====================================================================================================================
def plan(n: int, tp: int) -> Dict[str, object]:
    """Layout of one call: ``n`` gate tiles per chip, ``tp`` chips in the TP ring; CB tile counts and L1 bytes of each
    program. Raises ``ValueError`` for an unsupported shape."""
    n, tp = int(n), int(tp)
    if not 1 <= n <= MAX_TILES:
        raise ValueError(f"fused shared PolyNorm: {n} gate tiles per chip outside [1, {MAX_TILES}]")
    if not 1 <= tp <= MAX_TP:
        raise ValueError(f"fused shared PolyNorm: TP {tp} outside [1, {MAX_TP}]")
    moments = {CB_G: (n, FP32_TILE_BYTES), CB_PART: (3, FP32_TILE_BYTES)}
    apply = {
        CB_RECV: (3 * tp, FP32_TILE_BYTES),
        CB_K: (7, FP32_TILE_BYTES),
        CB_AG: (n, FP32_TILE_BYTES),
        CB_AU: (n, FP32_TILE_BYTES),
        CB_COEF: (4, FP32_TILE_BYTES),
        CB_OUT: (OUT_TILES, BF16_TILE_BYTES),
    }
    return dict(n=n, tp=tp, moments_cbs=moments, apply_cbs=apply,
                moments_l1=sum(t * b for t, b in moments.values()), apply_l1=sum(t * b for t, b in apply.values()))


def golden_fp64(g: torch.Tensor, u: torch.Tensor, c_by_power, b: float, *, eps: float = 1e-6) -> torch.Tensor:
    """HF scalar PolyNorm in fp64 on the FULL intermediate: ``g, u [..., T, I]``, ``c_by_power = (c of N(g), N(g^2),
    N(g^3))`` (sigmoid applied), ``b`` (no clamp) -> ``(c2 N(g) + c1 N(g^2) + c0 N(g^3) + b) u`` (x0.5 lives in W_down)."""
    g, u = g.double(), u.double()

    def N(z):
        return z / torch.sqrt(z.pow(2).mean(-1, keepdim=True) + eps)

    k1, k2, k3 = (float(c) for c in c_by_power)
    return (k1 * N(g) + k2 * N(g**2) + k3 * N(g**3) + float(b)) * u


def emulate_fp32(gu_chips, D: torch.Tensor, Ec: torch.Tensor, b: float):
    """The release / kernel algorithm in fp32 torch: ``gu_chips`` = the TP chips' ``[T, 2 n 32]`` gate_up outputs in
    gather order, ``D, Ec [3]`` (moment order g^2, g^4, g^6), ``b`` -> each chip's ``h [T, n 32]`` bf16. Per chip:
    tile-sequential fold of g^2 / g^4 (= g^2 g^2) / g^6 (= g^4 g^2) over the n tiles (elementwise), row sum; then the
    chip partials summed in gather order; ``a = rsqrt(s D + E)``; ``((g a3 + a2) g + a1) g + b``; ``* u`` -> bf16.
    (The SFPU's single-rounding multiply-add and its row-reduce order are not modelled bit for bit.)"""
    f = torch.float32
    parts = []
    for gu in gu_chips:
        gu = gu.to(f)
        T, W2 = gu.shape
        n = W2 // 2 // TILE
        g = gu[:, : n * TILE].reshape(T, n, TILE)
        acc = None
        for j in range(n):
            gj = g[:, j]
            g2 = gj * gj
            g4 = g2 * g2
            g6 = g4 * g2
            x = torch.stack([g2, g4, g6])
            acc = x if acc is None else acc + x
        parts.append(acc.sum(-1))  # [3, T]
    s = parts[0]
    for p in parts[1:]:
        s = s + p
    a = torch.rsqrt(s * D.to(f).reshape(3, 1) + Ec.to(f).reshape(3, 1))  # [3, T]
    outs = []
    for gu in gu_chips:
        gu = gu.to(f)
        nn = gu.shape[-1] // 2
        gg, uu = gu[:, :nn], gu[:, nn:]
        t = gg * a[2].reshape(-1, 1) + a[1].reshape(-1, 1)
        t = t * gg + a[0].reshape(-1, 1)
        t = t * gg + float(b)
        outs.append((t * uu).to(torch.bfloat16))
    return outs


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


def _kernel(name, cores, ct, defines, n_common, config):
    return ttnn.KernelDescriptor(
        kernel_source=str(SOURCES[name]), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=cores,
        compile_time_args=ct, defines=defines, runtime_args=[], common_runtime_args=[0] * n_common, config=config)


class FusedSharedPolyNorm:
    """The fused decode PolyNorm of one shared expert (module docstring). Holds references to the layer's constant
    tensors (``consts.D``, ``consts.E``, ``consts.b["fp32"]``; not owned), the CCL and the cached program descriptors;
    no device memory of its own.

    Args:
        mesh_device: the mesh (every chip runs the same programs).
        consts: the shared expert's :class:`~models.demos.motif3.tt.polynorm.ScalarPolyNormConsts`.
        ccl: the :class:`~models.demos.motif3.tt.ccl.MotifCCL` whose ``all_gather(.., "tp")`` the release's moments
            all-reduce uses.
        n_local: this chip's intermediate width (160 for the shared expert).
        debug: 0 (production) or 1 (diagnostics: :meth:`apply` returns the four coefficient tiles ``[A1 | A2 | A3 | B]``
            as ``[1, 1, 32, 128]`` fp32, column-broadcast, instead of ``h``).
    """

    def __init__(self, mesh_device, consts, ccl, *, n_local: int, debug: int = 0):
        self.mesh_device = mesh_device
        if debug not in (0, 1):
            raise ValueError(f"fused shared PolyNorm: debug {debug} not in (0, 1)")
        self.debug = int(debug)  # 0 = production; 1 = apply writes the A1 A2 A3 B coefficient tiles (fp32, diagnostics)
        self.ccl = ccl
        self.tp = int(ccl.axis_size("tp"))
        if int(n_local) % TILE:
            raise ValueError(f"fused shared PolyNorm: n_local {n_local} is not a multiple of {TILE}")
        self.n_local = int(n_local)
        self.n = self.n_local // TILE
        self.plan = plan(self.n, self.tp)
        self.D, self.Ec, self.b = consts.D, consts.E, consts.b["fp32"]
        for name, t, shape in (("D", self.D, (1, 3, 1, 1)), ("E", self.Ec, (1, 3, 1, 1)), ("b", self.b, (1, 1, 1, 1))):
            if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT or tuple(int(v) for v in t.shape) != shape:
                raise ValueError(f"fused shared PolyNorm: constant {name} must be fp32 TILE {shape}, got {t.dtype} "
                                 f"{t.layout} {tuple(t.shape)}")
            if not _is_interleaved(t.memory_config()):
                raise ValueError(f"fused shared PolyNorm: constant {name} must be interleaved")
        g = mesh_device.compute_with_storage_grid_size()
        if int(g.x) <= WORKER[0] or int(g.y) <= WORKER[1]:
            raise ValueError(f"fused shared PolyNorm: worker core {WORKER} outside the {g.x} x {g.y} grid")
        self._desc: Dict[tuple, object] = {}
        self._hash: Dict[tuple, int] = {}
        self._tag = _sources_tag()

    # ---- checks ---------------------------------------------------------------------------------------------------
    def supports(self, gu) -> bool:
        """True iff ``gu`` has the call contract (fp32 TILE interleaved ``[1, 1, 32, 2 n_local]``)."""
        try:
            self._check(gu)
            return True
        except ValueError:
            return False

    def _check(self, gu, *, any_rows: bool = False) -> None:
        shp = tuple(int(v) for v in gu.shape)
        rows_ok = (1 <= shp[2] <= TILE) if (any_rows and len(shp) == 4) else (len(shp) == 4 and shp[2] == TILE)
        if len(shp) != 4 or shp[:2] != (1, 1) or shp[3] != 2 * self.n_local or not rows_ok:
            raise ValueError(f"fused shared PolyNorm: gu must be [1, 1, {TILE}, {2 * self.n_local}], got {shp}")
        if gu.dtype != ttnn.float32 or gu.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"fused shared PolyNorm: gu must be fp32 TILE, got {gu.dtype} {gu.layout}")
        if not _is_interleaved(gu.memory_config()):
            raise ValueError("fused shared PolyNorm: gu must be interleaved")

    # ---- programs -------------------------------------------------------------------------------------------------
    def _cached(self, key, build):
        desc = self._desc.get(key)
        if desc is None:
            desc, prog_key = build()
            if hasattr(ttnn, "compute_program_descriptor_hash"):
                hv = self._hash.get(prog_key)
                if hv is None:
                    hv = self._hash[prog_key] = ttnn.compute_program_descriptor_hash(desc)
                desc.custom_program_hash = hv
            self._desc[key] = desc
        return desc

    def _cores(self):
        x, y = WORKER
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))])

    def _compute_config(self, fp32_cbs):
        from ..model_config import compute_config_descriptor  # lazy: keeps this module's import light

        return compute_config_descriptor("polynorm", fp32_unpack_cbs=fp32_cbs, dst_full_sync_en=True)

    def _moments_program(self, gu, s):
        key = ("moments",) + tuple(tuple(_accessor(t)) for t in (gu, s))

        def build():
            cores = self._cores()
            cbs = [_cb(i, t, ttnn.float32, cores) for i, (t, _) in sorted(self.plan["moments_cbs"].items())]
            defines = [("MOTIF_SPN_SRC", self._tag)]
            r_ct = [CB_G, self.n] + _accessor(gu)
            c_ct = [CB_G, CB_PART, self.n]
            w_ct = [CB_PART, 3] + _accessor(s)
            kernels = [
                _kernel("moments_reader", cores, r_ct, defines, 1, ttnn.ReaderConfigDescriptor()),
                _kernel("writer", cores, w_ct, defines, 1, ttnn.WriterConfigDescriptor()),
                _kernel("moments_compute", cores, c_ct, defines, 0, self._compute_config([CB_G])),
            ]
            return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs), ("m", tuple(r_ct), tuple(w_ct))

        desc = self._cached(key, build)
        desc.kernels[0].common_runtime_args = [gu.buffer_address()]
        desc.kernels[1].common_runtime_args = [s.buffer_address()]
        return desc

    def _apply_program(self, gat, gu, h):
        ts = (gat, gu, self.D, self.Ec, self.b, h)
        key = ("apply",) + tuple(tuple(_accessor(t)) for t in ts)

        def build():
            cores = self._cores()
            dts = {CB_OUT: ttnn.bfloat16 if not self.debug else ttnn.float32}
            cbs = [_cb(i, t, dts.get(i, ttnn.float32), cores) for i, (t, _) in sorted(self.plan["apply_cbs"].items())]
            defines = [("MOTIF_SPN_SRC", self._tag)]
            r_ct = ([CB_RECV, CB_K, CB_AG, CB_AU, self.n, self.tp] + _accessor(gat) + _accessor(gu) + _accessor(self.D)
                    + _accessor(self.Ec) + _accessor(self.b))
            c_ct = [CB_RECV, CB_K, CB_AG, CB_AU, CB_COEF, CB_OUT, self.n, self.tp, self.debug]
            w_ct = [CB_OUT, self.n if not self.debug else 4] + _accessor(h)
            kernels = [
                _kernel("apply_reader", cores, r_ct, defines, 5, ttnn.ReaderConfigDescriptor()),
                _kernel("writer", cores, w_ct, defines, 1, ttnn.WriterConfigDescriptor()),
                _kernel("apply_compute", cores, c_ct, defines, 0,
                        self._compute_config([CB_RECV, CB_K, CB_AG, CB_AU, CB_COEF])),
            ]
            return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs), ("a", tuple(r_ct), tuple(w_ct))

        desc = self._cached(key, build)
        desc.kernels[0].common_runtime_args = [gat.buffer_address(), gu.buffer_address(), self.D.buffer_address(),
                                               self.Ec.buffer_address(), self.b.buffer_address()]
        desc.kernels[1].common_runtime_args = [h.buffer_address()]
        return desc

    # ---- calls ----------------------------------------------------------------------------------------------------
    def moments(self, gu, *, memory_config=None, any_rows: bool = False):
        """``gu [1, 1, 32, 2 n_local]`` fp32 -> ``s [1, 3, 32, 32]`` fp32 (this chip's moment sums in column 0).
        ``any_rows`` (Phase F F3, ``kernels/shared_tail.py``): ``gu`` may hold T <= 32 logical rows (one padded tile
        row; the kernel reads whole tiles and every row's moments depend on that row only)."""
        self._check(gu, any_rows=any_rows)
        mc = memory_config or ttnn.L1_MEMORY_CONFIG
        s = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 3, TILE, TILE]), ttnn.float32, ttnn.TILE_LAYOUT,
                                           self.mesh_device, mc)
        ttnn.generic_op([gu, s], self._moments_program(gu, s))
        return s

    def apply(self, gat, gu, *, memory_config=None):
        """``gat [1, 3, 32, 32 TP]`` (the gathered moments), ``gu`` -> ``h [1, 1, 32, n_local]`` bf16."""
        self._check(gu)
        shp = tuple(int(v) for v in gat.shape)
        if shp != (1, 3, TILE, TILE * self.tp) or gat.dtype != ttnn.float32 or gat.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"fused shared PolyNorm: gathered moments must be fp32 TILE [1, 3, {TILE}, "
                             f"{TILE * self.tp}], got {gat.dtype} {gat.layout} {shp}")
        mc = memory_config or ttnn.L1_MEMORY_CONFIG
        if not (_is_interleaved(gat.memory_config()) and _is_interleaved(mc)):
            raise ValueError("fused shared PolyNorm: gathered moments and output must be interleaved")
        width, dt = (self.n_local, ttnn.bfloat16) if not self.debug else (4 * TILE, ttnn.float32)
        h = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, TILE, width]), dt, ttnn.TILE_LAYOUT, self.mesh_device, mc)
        ttnn.generic_op([gat, gu, self.D, self.Ec, self.b, h], self._apply_program(gat, gu, h))
        return h

    def __call__(self, gu, *, memory_config=None):
        """``gu [1, 1, 32, 2 n_local]`` fp32 -> ``h [1, 1, 32, n_local]`` bf16 (``memory_config``, default L1):
        moments -> the release's TP all-gather -> apply. Consumes nothing; frees its intermediates."""
        mc = memory_config or ttnn.L1_MEMORY_CONFIG
        s = self.moments(gu, memory_config=mc)
        gat = self.ccl.all_gather(s, 3, "tp", memory_config=mc)
        ttnn.deallocate(s)
        h = self.apply(gat, gu, memory_config=mc)
        ttnn.deallocate(gat)
        return h

    def deallocate(self) -> None:
        """Drops the cached descriptors and the constant references (the constants belong to the MLP module)."""
        self._desc.clear()
        self.D = self.Ec = self.b = None


__all__ = [
    "FusedSharedPolyNorm",
    "emulate_fp32",
    "golden_fp64",
    "plan",
]
