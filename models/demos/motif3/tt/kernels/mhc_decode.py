# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused mHC decode site (Phase C D3 / plan item C2, ``MOTIF3_MHC_DECODE=fused``).

The release decode site (``tt/mhc.py`` ``MHCSite``, T <= 32 tokens on one tile row) runs seven programs::

    y = X_b @ proj_b^T           split-K partials [1, 32, 32, 32] fp32       (matmul, 32 cores)
    s = rms_norm_pre_all_gather  sum(X^2) partials [1, 32, 32, 32] fp32       (32 cores)
    p = finalize_mixes(y, s)     (sum_b y_b) * rsqrt(sum_b s_b + eps)         (1 core)
    h_pre, h_post, H = motif_sinkhorn(p)                                      (1 core)
    w_pre, w_post = coefficient_layout(...)   TF32-rounded [.., T, 1] weight tiles in L1   (4 cores)
    x_red = attn_res_weighted_reduce_nc(X, w_pre)                             (120 cores)
    X' = post_mix(X, out, w_post)                                             (120 cores, after the sublayer)

Measured per site (8 lanes, traced, ``logs/opt/phaseC/D3/probe/probe1.json``): finalize 16.9 us, Sinkhorn 2.7, layout
3.6, pre 9.7, post 23.4. Two costs dominate and neither is arithmetic:

* the stream mixes read their weights over the NOC: every one of ~120 workers reads the whole weight set (post: 20
  fp32 tiles = 80 KB per worker from 20 L1 banks); without those reads the post mix takes 3.6 us;
* ``finalize_mixes`` sums 32 full fp32 tiles twice on one core, though only the faces holding the real rows and the
  24 mix columns (and column 0 of the statistics) are used downstream.

This module replaces finalize + Sinkhorn + layout with ONE single-core program (:func:`coefficients_packed`) that
writes a **packed coefficient tile** ``P``, and the two weighted reduces with programs (:func:`mix_packed`) whose
readers expand ``P`` into their weight CBs locally.

Packed tile ``P`` (fp32, one 32x32 tile per 32-token tile row, ``NCOPY`` identical copies on separate L1 banks)::

    P[k, t] = weight k of token t, TF32-rounded (nearest even) exactly as coefficient_layout rounds it:
      k = 0..3   h_pre[t, k]
      k = 4..7   h_post[t, k - 4]
      k = 8..23  H[t, i, j] with k = 8 + 4 i + j
      k = 24..31 zero

i.e. the transposed coefficient layout the Sinkhorn kernel already holds in DEST (``h_pre^T``, ``h_post^T``,
``H^T``) before its final transposes. Expansion copies bit patterns, so the weights the FPU sees are exactly the
release weights and both mixes are bitwise unchanged.

Fused coefficients program (one core): the finalize sums run the release LLK sequence (UnpackToDestFp32 copies, SFPU
fp32 adds ``((Y0 + Y1) + Y2) + ...``, ``add eps``, ``rsqrt``, column-broadcast multiply) restricted to the faces that
reach the Sinkhorn: for ``T <= 16`` the Sinkhorn reads token rows 0..15 only (``num_halves = 1``), so the projection
sums run on faces 0-1 and the statistics on face 0 (``VectorMode::R`` / one face). The per-face SFPU code is the same,
so every value that reaches an output is bitwise the release value. The Sinkhorn is the ``sinkhorn_motif`` kernel's
SFPU routine (``sinkhorn_motif/motif_mhc_sfpu.h``, shared with ``compute_sinkhorn_motif.cpp``), and the writer packs
``P`` from the transposed DEST tiles with the TF32 rounding of ``coefficient_layout``.

Contracts: decode only (``X`` ``[1, 4, T, D]`` with one 32-row tile row); trace-safe (fixed programs, buffer
addresses as common runtime args, memoized program hashes); one eager call per shape before a trace capture (kernel
binaries load at the first enqueue). Import rule: module level imports only stdlib / ttnn / torch.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Dict, Optional

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "mhc_decode"
SINKHORN_DIR = Path(__file__).resolve().parent / "sinkhorn_motif"
WR_SOURCES = {n: KERNEL_DIR / f"wr_{n}.cpp" for n in ("reader", "compute", "writer")}
WR_HEADERS = (KERNEL_DIR / "wr_expand.h",)
CO_SOURCES = {n: KERNEL_DIR / f"coeffs_{n}.cpp" for n in ("reader", "compute", "writer")}
SHARED_HEADERS = (SINKHORN_DIR / "motif_mhc_sfpu.h",)

TILE = 32
N_STREAMS = 4
F32_TILE = TILE * TILE * 4
BF16_TILE = TILE * TILE * 2
DEFAULT_NCOPY = 16
NUM_CB_SLOTS = 64

# wr program CBs
WR_CB_IN, WR_CB_W, WR_CB_P, WR_CB_FLAG, WR_CB_OUT = 0, 1, 2, 3, 16
# How the mixes' readers expand P into their weight CBs (wr_reader.cpp / wr_expand.h):
#   reader             the reader RISC expands the whole set (tokens 0..31)
#   split              the reader and the writer RISC expand half of the set each
#   zero_hi            tokens 16..31 are written as +0.0 without reading them (T <= 16: P is exactly zero there)
#   *_min              read only the P rows / token columns the mix uses (T <= 16)
#   skip*              no expansion (wrong weights): timing floors for probes only
#   auto (default)     pre: zero_hi_min, post: split_zero_hi_min when T <= 16; else reader / split
# Every non-skip mode gives the same weights bit for bit.
EXPAND_MODES = ("reader", "split", "split_zero_hi", "zero_hi", "split_zero_hi_min", "zero_hi_min", "skip",
                "skip_min", "skip_none")
DEFAULT_EXPAND = "auto"
# Work split of the mixes over the cores (wr_expand.h): "positions" (the release split: a core does every output row
# of its tile positions; post: 128 positions on 120 cores, so 8 cores do two) | "rows" ((position, output row) items:
# post 512 items, at most 5 per core) | "auto" (default: positions; the rows split measured no faster at 8 and 16
# lanes, logs/opt/phaseC/D3/probe/probe8.json: the post mix is not MAC-bound). Every output keeps its MAC order, so the
# split does not change a bit.
SPLITS = ("positions", "rows", "auto")
DEFAULT_SPLIT = "auto"

_TAGS: Dict[str, str] = {}


def _tag(kind: str) -> str:
    t = _TAGS.get(kind)
    if t is None:
        h = hashlib.sha1()
        srcs = list((WR_SOURCES if kind == "wr" else CO_SOURCES).values())
        srcs += list(WR_HEADERS if kind == "wr" else SHARED_HEADERS)
        for p in srcs:
            h.update(p.read_bytes())
        t = _TAGS[kind] = h.hexdigest()[:16]
    return t


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _first_cores(grid, n: int) -> ttnn.CoreRangeSet:
    """The first ``n`` cores in column-major order (core i = (i // gy, i % gy)), as in tt/mhc.py."""
    gy = int(grid.y)
    ranges = []
    full_cols, rest = divmod(n, gy)
    if full_cols:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_cols - 1, gy - 1)))
    if rest:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(full_cols, 0), ttnn.CoreCoord(full_cols, rest - 1)))
    return ttnn.CoreRangeSet(ranges)


def _cb(index: int, n_tiles: int, cores, dtype, tile_bytes: int) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=n_tiles * tile_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)],
    )


def _compute_descriptor(fidelity: str, approx: bool, fp32_unpack_cbs=()) -> ttnn.ComputeConfigDescriptor:
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = getattr(ttnn.MathFidelity, fidelity)
    cc.fp32_dest_acc_en = True
    cc.math_approx_mode = bool(approx)
    cc.dst_full_sync_en = False
    modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    for i in fp32_unpack_cbs:
        modes[int(i)] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = modes
    return cc


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def packed_shape(T: int, ncopy: int = DEFAULT_NCOPY):
    """Logical shape of the packed coefficient tensor: ``[1, 1, 32 * ncopy * ceil(T / 32), 32]`` fp32 (page
    ``row * ncopy + j`` = copy ``j`` of tile row ``row``)."""
    return [1, 1, TILE * int(ncopy) * (-(-int(T) // TILE)), TILE]


def allocate_packed(device, T: int, *, ncopy: int = DEFAULT_NCOPY, memory_config=None) -> ttnn.Tensor:
    mc = memory_config if memory_config is not None else ttnn.L1_MEMORY_CONFIG
    return ttnn.allocate_tensor_on_device(ttnn.Shape(packed_shape(T, ncopy)), ttnn.float32, ttnn.TILE_LAYOUT, device,
                                          mc)


def packed_torch(h_pre, h_post, H, *, ncopy: int = DEFAULT_NCOPY):
    """Host golden of ``P`` (one tile row, T <= 32): ``[32 * ncopy, 32]`` fp32 from ``h_pre [T,4]``, ``h_post [T,4]``,
    ``H [T,16]`` (TF32 nearest-even rounding of ``tt/mhc.py`` ``tf32_rne``; padding tokens and rows 24..31 zero)."""
    import torch

    from ..mhc import tf32_rne

    T = int(h_pre.shape[0])
    if T > TILE:
        raise ValueError("packed_torch: one tile row (T <= 32) only")
    P = torch.zeros(TILE, TILE, dtype=torch.float32)
    P[0:4, :T] = h_pre.float().t()
    P[4:8, :T] = h_post.float().t()
    P[8:24, :T] = H.float().t()
    P[:24] = tf32_rne(P[:24])
    return P.repeat(int(ncopy), 1)


_WR_HASH: Dict[tuple, int] = {}


def mix_packed(X: ttnn.Tensor, P: ttnn.Tensor, out: Optional[ttnn.Tensor] = None, *, ncopy: int = DEFAULT_NCOPY,
               memory_config=None, fidelity: str = "HiFi4", approx: bool = False, expand: str = DEFAULT_EXPAND,
               split: str = DEFAULT_SPLIT) -> ttnn.Tensor:
    """Decode stream mixing with the packed coefficients ``P``:

    * ``out is None`` (pre): ``x_red [1, 1, T, D]`` = ``attn_res_weighted_reduce_nc(X, w_pre, dim=1)`` bitwise;
    * ``out`` given (post): ``X' [1, 4, T, D]`` = ``tt/mhc.py`` ``post_mix(X, out, w_post)`` bitwise.

    ``X [1, 4, T, D]`` / ``out [1, 1, T, D]`` bf16 TILE interleaved, one tile row (T <= 32); ``P`` from
    :func:`coefficients_packed` (or :func:`allocate_packed` + host data) with ``ncopy`` copies."""
    xs = [int(d) for d in X.shape]
    if len(xs) != 4 or xs[0] != 1 or xs[1] != N_STREAMS:
        raise ValueError(f"mix_packed needs X [1, 4, T, D], got {xs}")
    Tp = int(X.padded_shape[2])
    if Tp != TILE:
        raise ValueError(f"mix_packed is the decode path (one tile row), got padded T {Tp}")
    if X.dtype != ttnn.bfloat16 or P.dtype != ttnn.float32:
        raise ValueError(f"mix_packed needs bf16 X and fp32 P, got {X.dtype} {P.dtype}")
    if int(P.buffer_num_pages()) < int(ncopy):
        raise ValueError(f"P has {P.buffer_num_pages()} pages < ncopy {ncopy}")
    post = out is not None
    if post:
        os_ = [int(d) for d in out.shape]
        if os_ != [1, 1, xs[2], xs[3]] or out.dtype != ttnn.bfloat16 or int(out.padded_shape[2]) != Tp:
            raise ValueError(f"mix_packed post needs out [1, 1, T, D] bf16, got {os_} {out.dtype}")
    if expand == "auto":
        small = int(X.shape[2]) <= 16
        expand = ("split_zero_hi_min" if small else "split") if post else ("zero_hi_min" if small else "reader")
    if split == "auto":
        split = "positions"
    if split not in SPLITS[:2]:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if expand not in EXPAND_MODES:
        raise ValueError(f"expand must be one of {EXPAND_MODES}, got {expand!r}")
    if ("zero_hi" in expand or expand.endswith("_min")) and int(X.shape[2]) > 16:
        raise ValueError("expand zero_hi needs T <= 16 (the coefficients of tokens 16..31 are zero then)")
    D = xs[3]
    Ht, Wt = Tp // TILE, D // TILE
    inner = Ht * Wt
    num_r = N_STREAMS if post else 1
    num_c = N_STREAMS + (1 if post else 0)
    dev = X.device()
    mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, num_r, xs[2], D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mc)
    grid = dev.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    total = inner * (num_r if split == "rows" else 1)
    n_cores = min(total, gx * gy)
    cores = _first_cores(grid, n_cores)
    rd_ct = [inner, Wt, N_STREAMS, 1 if post else 0, num_r, int(ncopy)] + _accessor(X)
    if post:
        rd_ct += _accessor(out)
    rd_ct += _accessor(P)
    cp_ct = [num_c, Wt, num_r]
    wr_ct = [inner, num_r] + _accessor(y)
    work = [total, n_cores, gy]
    split_x = expand.startswith("split")
    defines = [
        ("MOTIF_MHC_DECODE_SRC", _tag("wr")),
        ("MHC_EXPAND_SPLIT", "1" if split_x else "0"),
        ("MHC_EXPAND_ZERO_HI", "1" if "zero_hi" in expand else "0"),
        ("MHC_EXPAND_SKIP", "1" if expand.startswith("skip") else "0"),
        ("MHC_P_READ_MIN", "1" if expand.endswith("_min") else "0"),
        ("MHC_P_READ_NONE", "1" if expand.endswith("_none") else "0"),
        ("MHC_WR_NUM_C", str(num_c)),
        ("MHC_WR_HAS_OUT", "1" if post else "0"),
        ("MHC_WR_ITEMS", "1" if split == "rows" else "0"),
    ]
    fp = ttnn.KernelDescriptor.SourceType.FILE_PATH
    kernels = [
        ttnn.KernelDescriptor(kernel_source=str(WR_SOURCES["reader"]), source_type=fp, core_ranges=cores,
                              compile_time_args=rd_ct, defines=defines,
                              common_runtime_args=[X.buffer_address(), out.buffer_address() if post else 0,
                                                   P.buffer_address()] + work,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(WR_SOURCES["writer"]), source_type=fp, core_ranges=cores,
                              compile_time_args=wr_ct, defines=defines,
                              common_runtime_args=[y.buffer_address()] + work,
                              config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(WR_SOURCES["compute"]), source_type=fp, core_ranges=cores,
                              compile_time_args=cp_ct, defines=defines, common_runtime_args=work,
                              config=_compute_descriptor(fidelity, approx)),
    ]
    cbs = [  # post_mix's CBs (candidates double-buffered, one weight set, outputs double-buffered) + the packed tile
        _cb(WR_CB_IN, 2 * num_c, cores, ttnn.bfloat16, BF16_TILE),
        _cb(WR_CB_W, num_r * num_c, cores, ttnn.float32, F32_TILE),
        _cb(WR_CB_P, 1, cores, ttnn.float32, F32_TILE),
        _cb(WR_CB_FLAG, 1, cores, ttnn.float32, 64),
        _cb(WR_CB_OUT, 2 * num_r, cores, ttnn.bfloat16, BF16_TILE),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    key = (tuple(rd_ct), tuple(wr_ct), tuple(cp_ct), gx, gy, n_cores, fidelity, bool(approx), _tag("wr"), expand, split)
    h = _WR_HASH.get(key)
    if h is None:
        h = _WR_HASH[key] = ttnn.compute_program_descriptor_hash(desc)
    desc.custom_program_hash = h
    ttnn.generic_op([X, out, P, y] if post else [X, P, y], desc)
    return y


# coefficients program CBs (must match mhc_decode/coeffs_compute.cpp)
CO_CB_Y, CO_CB_S, CO_CB_C, CO_CB_PM, CO_CB_OUT = 0, 1, 2, 3, 16
_CO_HASH: Dict[tuple, int] = {}


def coefficients_packed(y: ttnn.Tensor, s: ttnn.Tensor, consts: ttnn.Tensor, *, eps: float, T: int,
                        P: Optional[ttnn.Tensor] = None, ncopy: int = DEFAULT_NCOPY, iters: int = 20,
                        half: Optional[bool] = None, chunk: int = 8, memory_config=None, fidelity: str = "HiFi4",
                        approx: bool = False) -> ttnn.Tensor:
    """The packed coefficient tile ``P`` (module docstring) of one decode site from the split-K partials ``y`` /
    ``s`` (``[1, NB, 32, 32]`` fp32, ``MHCSite._partials_decode``) and the site's Motif constants (``[64, 32]`` fp32,
    ``sinkhorn_motif.build_consts``), in one single-core program: release ``finalize_mixes`` + ``motif_sinkhorn`` +
    ``coefficient_layout`` (TF32 rounding) values, bit for bit. ``half`` (default: ``T <= 16``) restricts the finalize
    to the faces the single-half Sinkhorn reads. ``P`` (``allocate_packed``) is allocated when not given."""
    from . import sinkhorn_motif as km

    T = int(T)
    if not 1 <= T <= TILE:
        raise ValueError(f"coefficients_packed is the decode path (1 <= T <= 32), got T = {T}")
    nb = int(y.shape[1])
    if [int(d) for d in y.padded_shape] != [1, nb, TILE, TILE] or [int(d) for d in s.padded_shape] != [1, nb, TILE, TILE]:
        raise ValueError(f"y / s must be [1, NB, 32, 32], got {list(y.padded_shape)} / {list(s.padded_shape)}")
    if y.dtype != ttnn.float32 or s.dtype != ttnn.float32 or consts.dtype != ttnn.float32:
        raise ValueError("y, s and consts must be float32")
    chunk = min(int(chunk), nb)
    if nb % chunk:
        raise ValueError(f"NB {nb} must be a multiple of chunk {chunk}")
    half = (T <= 16) if half is None else bool(half)
    if half and T > 16:
        raise ValueError("half needs T <= 16")
    halves = 1 if T <= 16 else 2
    dev = y.device()
    if P is None:
        P = allocate_packed(dev, T, ncopy=ncopy, memory_config=memory_config)
    if int(P.buffer_num_pages()) < int(ncopy):
        raise ValueError(f"P has {P.buffer_num_pages()} pages < ncopy {ncopy}")
    grid = dev.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    rd_ct = [nb, chunk, 1 if half else 0] + _accessor(y) + _accessor(s) + _accessor(consts)
    wr_ct = [int(ncopy)] + _accessor(P)
    cp_ct = [nb, chunk, f32_bits(eps), 1 if half else 0, int(iters), halves, f32_bits(km.PRE_POST_CLAMP),
             f32_bits(km.RES_CLAMP), f32_bits(km.SUM_FLOOR), f32_bits(km.H_POST_COEFF), 0]
    defines = [("MOTIF_MHC_DECODE_SRC", _tag("co"))]
    fp = ttnn.KernelDescriptor.SourceType.FILE_PATH
    kernels = [
        ttnn.KernelDescriptor(kernel_source=str(CO_SOURCES["reader"]), source_type=fp, core_ranges=cores,
                              compile_time_args=rd_ct, defines=defines,
                              common_runtime_args=[y.buffer_address(), s.buffer_address(), consts.buffer_address()],
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(CO_SOURCES["writer"]), source_type=fp, core_ranges=cores,
                              compile_time_args=wr_ct, defines=defines, common_runtime_args=[P.buffer_address()],
                              config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=str(CO_SOURCES["compute"]), source_type=fp, core_ranges=cores,
                              compile_time_args=cp_ct, defines=defines, common_runtime_args=[],
                              config=_compute_descriptor(fidelity, approx, (CO_CB_Y, CO_CB_S, CO_CB_C, CO_CB_PM))),
    ]
    cbs = [
        _cb(CO_CB_Y, 2 * chunk, cores, ttnn.float32, F32_TILE),
        _cb(CO_CB_S, 2 * chunk, cores, ttnn.float32, F32_TILE),
        _cb(CO_CB_C, 2, cores, ttnn.float32, F32_TILE),
        _cb(CO_CB_PM, 1, cores, ttnn.float32, F32_TILE),
        _cb(CO_CB_OUT, 1, cores, ttnn.float32, F32_TILE),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    key = (tuple(rd_ct), tuple(wr_ct), tuple(cp_ct), int(grid.x), int(grid.y), fidelity, bool(approx), _tag("co"))
    h = _CO_HASH.get(key)
    if h is None:
        h = _CO_HASH[key] = ttnn.compute_program_descriptor_hash(desc)
    desc.custom_program_hash = h
    ttnn.generic_op([y, s, consts, P], desc)
    return P


__all__ = ["DEFAULT_NCOPY", "allocate_packed", "coefficients_packed", "mix_packed", "packed_shape", "packed_torch"]
