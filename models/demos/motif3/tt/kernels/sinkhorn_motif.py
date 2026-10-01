# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact Motif-3 mHC coefficients on device: ``h_pre``, ``h_post`` and the 20-iteration Sinkhorn ``H``.

WAVE_A_REVIEW MHC-6 ("Option B", design §2.3.3 steps 3 and 6). One ``ttnn.generic_op`` launch with out-of-tree kernels
(``sinkhorn_motif/{reader,compute,writer,writer_..._wrnc}_sinkhorn_motif.cpp``, JIT-compiled at first use, no
tt-metal rebuild), adapted from ``ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn``. It replaces the draft-1
"pre-clamped stock op" path (MHC-3: 3 tiny pre-clamp ops + the 52 us stock kernel + ``0.5 * post``).

Semantics (Motif ``MHCLayer.forward``, HF ``modeling_motif.py:226-249``, ``reference.modules.MHCLayer``), per token,
from the 24 raw projections ``p = [p_pre(4) | p_post(4) | p_res(16, row-major 4i+j)]``:

    h_pre  = sigmoid(clamp(alpha_pre  * p_pre  + bias_pre,  -10, 10))            -> h_pre  [T, 4]
    h_post = 1.0 * sigmoid(clamp(alpha_post * p_post + bias_post, -10, 10))      -> h_post [T, 4]   (no x2, no 0.5)
    H      = Sinkhorn20(alpha_res * p_res + bias_res)                            -> H      [T, 16]  (H[i][j] at col 4i+j)
    Sinkhorn: m = exp(clamp(L, -20, 20)); 20 x { m /= max(rowsum, 1e-8); m /= max(colsum, 1e-8) }  (rows first)

Everything is fp32 SFPU math (no FPU, so no TF32 truncation): logits as an fp32 multiply then an fp32 add (bitwise equal
to torch's ``alpha * p + b``), the accurate fp32 exp / sigmoid of the LLK library, the 4-wide row / column sums as SFPU
adds ``((m0 + m1) + m2) + m3`` in registers (row sums after SFPTRANSP), ``max(sum, fp32(1e-8))``, reciprocal =
SFPARECIP + 2 Newton steps, all 40 normalizations in LREGs. The transposes into / out of the register layout are bit
exact (fp32 unpack-to-dest + 32-bit in-dest transpose; the CBs are ``UnpackToDestFp32``).

Measured on this Galaxy (``tests/unit/test_sinkhorn_motif.py``, 4x8 mesh, committed fabric TORUS_Y; replicated inputs
give bitwise-identical outputs on all 32 chips, row-varying decode inputs bitwise-identical outputs on the 8 TP chips of
each DP row):

* Accuracy vs the reference's exact fp32 maps: real mixes of layers 0-5 and 28-35 (28 sites, 4096 real tokens,
  |L_res| up to 67) max|dH| <= 3.0e-7, max|dh_pre|, max|dh_post| <= 1.2e-7; synthetic G3 regimes (incl. the 1e-8 floor
  and +90 logits) <= 4.5e-7 / 1.8e-7; token counts 1 .. 32768 (all prefill buckets) <= 3.6e-7; row-varying real decode
  lanes (every chip vs its own row) <= 1.8e-7; logits bitwise equal to torch. The test bounds (1e-5 on H, 1e-6 on
  h_pre / h_post) are self-imposed fp32-exactness targets: WAVE_A_REVIEW MHC-6 sets no number, and the MHC-5 / README
  §12 acceptance of the mHC module is max|dH| <= 5e-3, h_pre / h_post <= 2e-3.
* Traced device time per call, as the raw per-call time ``t(n) / n`` of a trace of ``n`` back-to-back calls (an upper
  bound: it includes ``sync / n``) and the slope ``(t(n) - t(n/2)) / (n/2)``; with n = 256 at T <= 32 the two agree
  within 0.25 us. ``layout="stock"``: 5.3 us at T = 8 and 7.3 us at T = 32 (stock ``mhc_split_sinkhorn`` 52.5 us:
  7.2-9.9x), 23.8 us at T = 4096 (stock 114.9 us), and at the 8192 / 16384 / 32768 prefill buckets 33 / 52 / 94 us
  raw (slope 29 / 45 / 86 us). ``layout="wrnc"``: 10.6 / 12.5 us at T = 8 / 32 (the stock layout + a 5-op ttnn glue to
  the same weights: 29.6 / 33.6 us). **But it writes 8x the bytes of the stock layout**: 24 instead of 3 fp32 tiles
  per 32 tokens, because ``attn_res_weighted_reduce_nc`` takes ``[.., T, 1]`` TILE weights (32x column padding; any
  producer of those weights writes the same bytes). In prefill it is DRAM-write bound (~205 GB/s): 62 / 124 / 241 /
  471 us raw at T = 4096 / 8192 / 16384 / 32768, i.e. 49.9 ms per 106-site 32K prefill vs 9.9 ms for the stock layout.
  Use it in decode; in prefill weigh it against the consumer path (``test_prefill_latency`` prints the figures).
* Host: an eager call with preallocated outputs costs ~91-104 us (``generic_op`` dispatch 84-95 us, descriptor-cache
  hit 4.7-4.9 us); a cache miss (full validation 27 us + descriptor build) costs 52-115 us once per new combination of
  tensor specs and buffer addresses; three fresh output allocations add ~105 us.

API
---
Host (pure torch, device-free):

* ``build_consts(alpha_pre, alpha_post, alpha_res, bias_pre, bias_post, bias_res) -> torch.float32 [64, 32]``:
  the per-site constants (2 fp32 tiles: row q of tile 0 = alpha of mix q's group, row q of tile 1 = bias of mix q,
  both broadcast over the 32 columns; rows 24..31 zero). ``consts_from_scalars(weights.mhc_scalars(source, prefix))``
  builds it from the shared weight transform (``bias_res`` row-major ``4i + j``); ``scalars_from_consts`` inverts it.
* ``motif_mhc_maps_torch(p [T, 24], consts | scalars, iters=20, dtype=fp32) -> (h_pre [T,4], h_post [T,4], H [T,16])``:
  the exact golden (bitwise equal to ``reference.modules.MHCLayer`` / ``reference.modules.sinkhorn``);
  ``motif_logits_torch``; ``wrnc_weights_torch(h_pre, h_post, H) -> (w_pre, w_post)`` (the wrnc layout below).

Device:

* ``consts_to_device(consts, mesh_device) -> ttnn.Tensor``: ``[64, 32]`` fp32 TILE, DRAM, replicated on every chip
  (bring-up / tests; the model uploads the same tensor through ``weights.as_tensor``, see ``tt/mhc.py``).
* ``motif_sinkhorn(mixes, consts, *, iters=20, memory_config=None, outputs=None, layout="stock")``:

  - ``mixes``: the raw projections, fp32 TILE (32x32 tiles), interleaved (DRAM or L1), logical ``[..., T, W]`` with all
    leading dims 1 and ``24 <= W <= 32`` (columns 0..23 used; e.g. the MHC-2 projection output ``[1, 1, T, 32]``), any
    T >= 1. Only the ``ceil(T / 32)`` tiles holding logical rows are read (one 32-token tile per core, tiles split over
    the compute grid), so a view whose padded height is larger (``ttnn.reshape(t, [1,1,8,32], [1,1,64,32])``) is fine;
    padding rows and columns 24..31 may hold anything (NaN / Inf included); T <= 16 processes only token rows 0..15.
  - ``consts``: from ``consts_to_device`` (one per mHC site; 106 per model).
  - ``layout="stock"`` (default) returns ``(h_pre [T, 4], h_post [T, 4], H [T, 16])`` with the stock op's output
    specs (rank 2, fp32 TILE, DRAM interleaved unless ``memory_config``); padding columns are zero; padding token rows
    (``T <= t < 32 ceil(T/32)``) are unspecified (computed from the input's padding rows, NaN if those hold NaN) except
    that rows 16..31 are zero when T <= 16, so consumers must not reduce across token rows of the padding (row-wise ops
    like ``attn_res_weighted_reduce_nc`` are fine); ``ttnn.reshape(x, [1, 1, T, n])`` is a view. Drop-in for MHC-3's
    ``mhc_split_sinkhorn(clamp(alpha p + b), consts_identity, 4, 20, 0.0)`` **with final values**: pass the raw
    mixes (no pre-clamp ops), ``h_post`` is already ``sigmoid`` (do NOT halve it), ``H`` is read as ``[T, 4, 4]``
    row-major (no transpose), exactly like the stock ``comb``.
  - ``layout="wrnc"`` returns ``(w_pre [1, 4, T, 1], w_post [4, 5, T, 1])`` fp32 TILE: the weights of
    ``ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc`` (MHC-4) with no glue ops:
    ``w_pre[0, c, t, 0] = h_pre[t, c]`` (``x_red = wrnc(X [1,4,T,D], w_pre, dim=1)``) and
    ``w_post[r, c, t, 0] = H[t, r, c]`` for c < 4, ``h_post[t, r]`` for c = 4
    (``X' = wrnc(concat([X, out], dim=1) [1,5,T,D], w_post, dim=1)`` -> ``[4, 1, T, D]``, a view of ``[1, 4, T, D]``).
    Scalar in column 0 of each tile row (BroadcastType::COL), other columns zero. 8x the output bytes of "stock"
    (see the latency note above).
  - ``outputs``: preallocated output tensors to skip the allocation (~35 us of host time per tensor on the 32-chip
    mesh). They must have exactly the specs of ``allocate_outputs(mixes, memory_config, layout)``: the
    ``output_shapes(T, layout)`` logical shapes with standard tile padding, fp32 TILE (32x32), interleaved, on the
    mixes' mesh (checked; a mismatch raises ``ValueError``).
  - Validation and caching: the descriptor (incl. its runtime args) is cached per (options, mesh, buffer addresses)
    and an entry is reused only if the ``TensorSpec`` of every tensor equals the one it was validated with; any new
    combination re-runs the full input / output validation. The mesh is identified by ``MeshDevice.id()``, a
    process-wide counter, so a closed and reopened mesh (possibly with another compute grid, e.g. another
    ``dispatch_core_axis``) never reuses a descriptor built for the old one, unlike Python ``id()``.
  - Trace-safe: no host round trip, one fixed program per (shape, layout); buffer addresses are *common* runtime args
    (patched on program-cache hits; host launch cost ~100 us eager, independent of T and core count); replicated inputs
    give bitwise-identical replicas. **Warm-up before a trace capture must be one real (eager) call**:
    ``ttnn.experimental.prepare_generic_op`` compiles and caches the program, but the kernel binaries reach DRAM only
    at the first enqueue ("Cannot load new binaries during trace capture" otherwise).
  - ``mode`` (``"full"`` / ``"passthrough"`` (layout check) / ``"logits"``), ``div_refine``, ``num_halves`` and
    ``max_cores`` are bring-up / tuning switches; ``program_descriptor(...)`` exposes the ``ttnn.ProgramDescriptor``.

* ``MotifSinkhorn(mesh_device, scalars | consts, iters=None, memory_config=None, reuse_outputs=False, cfg=None)``: one
  mHC site (holds the device consts). **A bring-up / test helper, not the production path**: ``tt/mhc.py`` (the mHC
  module, MHC-2..4) owns the per-site constants (uploaded through ``weights.as_tensor``, h_post coefficient checked
  against the config) and calls ``motif_sinkhorn`` directly. With ``cfg`` (a ``MotifTTConfig``) it follows the same
  conventions: ``cfg.mhc_h_post_coeff`` must be 1.0 (the kernel's fixed coefficient), ``iters`` defaults to
  ``cfg.sinkhorn_iters`` and the consts are uploaded with ``weights.as_tensor`` (no cache); without it, ``iters``
  defaults to 20 and ``consts_to_device`` is used. ``MotifSinkhorn.from_source(mesh_device, source, layer_idx, site,
  cfg=None)`` reads ``model.layers.{l}.{site}.{alpha_*, bias_*}`` through ``weights.mhc_scalars``;
  ``site(mixes, layout=...) -> outputs`` as ``motif_sinkhorn``; ``reuse_outputs=True`` overwrites one output set per
  shape (consume before the next call of the same site); ``site.golden(p)``; ``site.release()``.
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "sinkhorn_motif"
READER_SRC = KERNEL_DIR / "reader_sinkhorn_motif.cpp"
WRITER_SRC = KERNEL_DIR / "writer_sinkhorn_motif.cpp"
WRITER_WRNC_SRC = KERNEL_DIR / "writer_sinkhorn_motif_wrnc.cpp"
COMPUTE_SRC = KERNEL_DIR / "compute_sinkhorn_motif.cpp"

N_STREAMS = 4
N_MIXES = (2 + N_STREAMS) * N_STREAMS  # 24
TILE = 32
TILE_BYTES_FP32 = TILE * TILE * 4
DEFAULT_ITERS = 20
PRE_POST_CLAMP = 10.0  # HF modeling_motif.py:236-240
RES_CLAMP = 20.0  # :226-233
SUM_FLOOR = 1e-8  # torch clamp(min=1e-8) on fp32 -> fp32(1e-8)
# Motif-3's h_post coefficient (1 + mhc_h_post_alpha_end, absent -> 1.0), a compile-time constant of the kernel.
# Callers with a config check it: tt/mhc.py (cfg.mhc_h_post_coeff) and MotifSinkhorn(cfg=...).
H_POST_COEFF = 1.0

# CB indices (must match compute_sinkhorn_motif.cpp)
CB_MIXES, CB_CONSTS, CB_PRE, CB_POST, CB_COMB, CB_TMP, CB_WSCRATCH = 0, 1, 2, 3, 4, 24, 25
N_WSCRATCH = 24  # wrnc layout: 4 w_pre + 20 w_post tiles per token tile
LAYOUTS = ("stock", "wrnc")
NUM_CB_SLOTS = 64  # NUM_CIRCULAR_BUFFERS on Blackhole (size of the unpack_to_dest_mode vector)
# compute config values; they equal model_config.COMPUTE_ROLES["mhc"] (see _compute_config)
COMPUTE_FIDELITY, COMPUTE_FP32_ACC, COMPUTE_APPROX = "HiFi4", True, False

_MODES = {"full": 0, "passthrough": 1, "logits": 2}


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _sources_tag() -> str:
    """Content hash of the kernel sources: a compile define, so an edited kernel never hits a stale JIT build."""
    h = hashlib.sha1()
    for p in (READER_SRC, WRITER_SRC, WRITER_WRNC_SRC, COMPUTE_SRC):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


# =====================================================================================================================
# host side
# =====================================================================================================================
def build_consts(
    alpha_pre,
    alpha_post,
    alpha_res,
    bias_pre: Sequence[float] | torch.Tensor,
    bias_post: Sequence[float] | torch.Tensor,
    bias_res: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Per-site constants ``[64, 32]`` fp32: tile 0 row q = alpha of mix q's group, tile 1 row q = bias of mix q.

    Mix order ``q``: 0..3 pre, 4..7 post, 8..23 res with ``q = 8 + 4i + j`` <-> ``H[i][j]`` (``bias_res`` given as
    ``[4, 4]`` or row-major ``[16]``). Values are rounded to fp32 once (the checkpoint stores them as bf16/fp32)."""

    def vec(x, n):
        t = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
        if t.numel() != n:
            raise ValueError(f"expected {n} values, got {t.numel()}")
        return t

    def scalar(x):
        t = torch.as_tensor(x, dtype=torch.float32).reshape(-1)
        if t.numel() != 1:
            raise ValueError(f"alpha must be a scalar, got {t.numel()} values")
        return float(t[0])

    a = torch.zeros(TILE, TILE, dtype=torch.float32)
    b = torch.zeros(TILE, TILE, dtype=torch.float32)
    a[0:4, :] = scalar(alpha_pre)
    a[4:8, :] = scalar(alpha_post)
    a[8:24, :] = scalar(alpha_res)
    b[0:4, :] = vec(bias_pre, 4)[:, None]
    b[4:8, :] = vec(bias_post, 4)[:, None]
    b[8:24, :] = vec(bias_res, 16)[:, None]
    return torch.cat([a, b], dim=0).contiguous()


def consts_from_scalars(scalars: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """``build_consts`` from ``weights.mhc_scalars(source, prefix)`` (keys alpha_pre/post/res, bias_pre/post/res)."""
    return build_consts(
        scalars["alpha_pre"],
        scalars["alpha_post"],
        scalars["alpha_res"],
        scalars["bias_pre"],
        scalars["bias_post"],
        scalars["bias_res"],
    )


def scalars_from_consts(consts: torch.Tensor) -> dict:
    """Inverse of :func:`build_consts` (fp32)."""
    c = torch.as_tensor(consts, dtype=torch.float32).reshape(2 * TILE, TILE)
    a, b = c[:TILE, 0], c[TILE:, 0]
    return dict(
        alpha_pre=a[0:1].clone(),
        alpha_post=a[4:5].clone(),
        alpha_res=a[8:9].clone(),
        bias_pre=b[0:4].clone(),
        bias_post=b[4:8].clone(),
        bias_res=b[8:24].clone(),
    )


def sinkhorn_torch(logits: torch.Tensor, iters: int = DEFAULT_ITERS) -> torch.Tensor:
    """Motif Sinkhorn on ``[..., 4, 4]`` logits in fp32 (verbatim ``reference.modules.sinkhorn``)."""
    m = logits.float().clamp(-RES_CLAMP, RES_CLAMP).exp()
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True).clamp(min=SUM_FLOOR)
        m = m / m.sum(dim=-2, keepdim=True).clamp(min=SUM_FLOOR)
    return m


def motif_logits_torch(
    p: torch.Tensor, consts_or_scalars: Union[torch.Tensor, Mapping[str, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unclamped fp32 logits ``(L_pre [T,4], L_post [T,4], L_res [T,4,4])`` = ``alpha * p + bias`` (mul, then add)."""
    s = scalars_from_consts(consts_or_scalars) if isinstance(consts_or_scalars, torch.Tensor) else consts_or_scalars
    p = p.float().reshape(-1, p.shape[-1])
    f = lambda k: torch.as_tensor(s[k], dtype=torch.float32)  # noqa: E731
    l_pre = f("alpha_pre").reshape(-1)[:1] * p[:, 0:4] + f("bias_pre").reshape(4)
    l_post = f("alpha_post").reshape(-1)[:1] * p[:, 4:8] + f("bias_post").reshape(4)
    l_res = f("alpha_res").reshape(-1)[:1] * p[:, 8:24].reshape(-1, 4, 4) + f("bias_res").reshape(4, 4)
    return l_pre, l_post, l_res


def motif_mhc_maps_torch(
    p: torch.Tensor,
    consts_or_scalars: Union[torch.Tensor, Mapping[str, torch.Tensor]],
    iters: int = DEFAULT_ITERS,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Golden ``(h_pre [T,4], h_post [T,4], H [T,16])`` from raw projections ``p [T, >=24]`` (Motif semantics; fp32 by
    default, ``dtype=torch.float64`` for an exact-arithmetic reference)."""
    l_pre, l_post, l_res = (x.to(dtype) for x in motif_logits_torch(p, consts_or_scalars))
    h_pre = torch.sigmoid(l_pre.clamp(-PRE_POST_CLAMP, PRE_POST_CLAMP))
    h_post = H_POST_COEFF * torch.sigmoid(l_post.clamp(-PRE_POST_CLAMP, PRE_POST_CLAMP))
    m = l_res.clamp(-RES_CLAMP, RES_CLAMP).exp()
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True).clamp(min=SUM_FLOOR)
        m = m / m.sum(dim=-2, keepdim=True).clamp(min=SUM_FLOOR)
    return h_pre, h_post, m.reshape(-1, N_STREAMS * N_STREAMS)


def wrnc_weights_torch(h_pre: torch.Tensor, h_post: torch.Tensor, H: torch.Tensor):
    """``(w_pre [1,4,T,1], w_post [4,5,T,1])`` from ``(h_pre [T,4], h_post [T,4], H [T,16])`` (the wrnc layout):
    ``w_pre[0,c,t,0] = h_pre[t,c]``; ``w_post[r,c,t,0] = H[t,r,c]`` (c < 4), ``h_post[t,r]`` (c = 4)."""
    T = h_pre.shape[0]
    w_pre = h_pre.t().reshape(1, N_STREAMS, T, 1)
    w = torch.cat([H.reshape(T, N_STREAMS, N_STREAMS), h_post.reshape(T, N_STREAMS, 1)], dim=-1)  # [T, r, c]
    w_post = w.permute(1, 2, 0).reshape(N_STREAMS, N_STREAMS + 1, T, 1)
    return w_pre.contiguous(), w_post.contiguous()


def output_shapes(T: int, layout: str = "stock"):
    """Logical output shapes: stock ``([T,4], [T,4], [T,16])``; wrnc ``(w_pre [1,4,T,1], w_post [4,5,T,1])``."""
    if layout == "stock":
        return ([T, N_STREAMS], [T, N_STREAMS], [T, N_STREAMS * N_STREAMS])
    if layout == "wrnc":
        return ([1, N_STREAMS, T, 1], [N_STREAMS, N_STREAMS + 1, T, 1])
    raise ValueError(f"layout must be one of {LAYOUTS}")


def _tile_padded(shape) -> list:
    """Standard 32x32 TILE padding of a logical shape (last two dims rounded up to 32)."""
    s = [int(d) for d in shape]
    return s[:-2] + [-(-d // TILE) * TILE for d in s[-2:]]


def _pages_written(n_tiles: int, layout: str) -> Tuple[int, ...]:
    """Highest page id + 1 the writer addresses per output (stock: page h; wrnc: page c * n_tiles + h)."""
    return (n_tiles,) * 3 if layout == "stock" else (N_STREAMS * n_tiles, N_STREAMS * (N_STREAMS + 1) * n_tiles)


# =====================================================================================================================
# device side
# =====================================================================================================================
def consts_to_device(consts: torch.Tensor, mesh_device, memory_config=None) -> ttnn.Tensor:
    """Upload ``build_consts`` output (``[64, 32]`` fp32) as a replicated fp32 TILE tensor (bring-up / tests; the model
    path uploads the same host tensor through ``weights.as_tensor``)."""
    c = torch.as_tensor(consts, dtype=torch.float32).reshape(2 * TILE, TILE).contiguous()
    kw = {}
    if isinstance(mesh_device, ttnn.MeshDevice):
        kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(
        c,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        **kw,
    )


def _device_uid(dev):
    """Process-unique identity of a (mesh) device: ``MeshDevice.id()`` comes from a process-wide counter
    (``generate_unique_mesh_id``), so a reopened mesh never aliases a closed one (Python ``id()`` can)."""
    f = getattr(dev, "id", None)
    if callable(f):
        return int(f())
    g = dev.compute_with_storage_grid_size()  # fallback: Python id() can be reused, so also key on the grid
    return ("py", id(dev), int(g.x), int(g.y))


def _check_tensor(name: str, t, *, dev_uid=None) -> None:
    if not isinstance(t, ttnn.Tensor):
        raise TypeError(f"{name} must be a ttnn.Tensor, got {type(t).__name__}")
    if t.storage_type() != ttnn.StorageType.DEVICE:
        raise ValueError(f"{name} must be a device tensor")
    if not t.is_allocated():
        raise ValueError(f"{name} is deallocated")
    if t.dtype != ttnn.float32 or t.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"{name} must be float32 TILE, got {t.dtype} {t.layout}")
    if int(t.buffer_page_size()) != TILE_BYTES_FP32:  # 32x32 fp32 tiles (the CB / NOC page size of the kernels)
        raise ValueError(f"{name} must use 32x32 fp32 tiles ({TILE_BYTES_FP32}-byte pages), got {t.buffer_page_size()}")
    if t.memory_config().is_sharded():
        raise ValueError(f"{name} must be interleaved (DRAM or L1); sharded tensors are not supported")
    if dev_uid is not None and _device_uid(t.device()) != dev_uid:
        raise ValueError(f"{name} is on another mesh device than mixes")


def _validate(mixes: ttnn.Tensor, consts: ttnn.Tensor, outs: Sequence[ttnn.Tensor], layout: str) -> Tuple[int, int]:
    """Full check of one call's tensors (run on every descriptor-cache miss); returns ``(T, n_tiles)``.

    ``n_tiles = ceil(T / 32)`` comes from the *logical* T, like the outputs (``output_shapes``): a mixes view whose
    padded height is larger is read only in the tiles that hold logical rows, so the writers never address pages
    beyond the outputs (the wrnc writer's page ids ``c * n_tiles + h`` need exactly ``ceil(T / 32)`` tile rows per
    output)."""
    _check_tensor("mixes", mixes)
    dev_uid = _device_uid(mixes.device())
    _check_tensor("consts", consts, dev_uid=dev_uid)
    shape, padded = [int(d) for d in mixes.shape], [int(d) for d in mixes.padded_shape]
    if len(shape) < 2 or any(d != 1 for d in shape[:-2]):
        raise ValueError(f"mixes must be [..., T, W] with leading dims 1, got {shape}")
    if not (N_MIXES <= shape[-1] <= TILE) or padded[-1] != TILE:
        raise ValueError(f"mixes last dim must be in [{N_MIXES}, {TILE}] (one tile wide), got {shape} / {padded}")
    T = shape[-2]
    if T < 1:
        raise ValueError(f"mixes must hold at least one token row, got {shape}")
    n_tiles = -(-T // TILE)
    if int(mixes.buffer_num_pages()) < n_tiles:
        raise ValueError(f"mixes buffer has {mixes.buffer_num_pages()} pages, < ceil(T/32) = {n_tiles}")
    cp = [int(d) for d in consts.padded_shape]
    if math.prod(cp) != 2 * TILE * TILE or cp[-1] != TILE:
        raise ValueError(f"consts must be the [64, 32] tensor of build_consts, got padded shape {cp}")
    expected = output_shapes(T, layout)
    if len(outs) != len(expected):
        raise ValueError(f"layout {layout!r} takes {len(expected)} output tensors, got {len(outs)}")
    for k, (o, shp, pages) in enumerate(zip(outs, expected, _pages_written(n_tiles, layout))):
        _check_tensor(f"outputs[{k}]", o, dev_uid=dev_uid)
        got, got_p = [int(d) for d in o.shape], [int(d) for d in o.padded_shape]
        if got != shp or got_p != _tile_padded(shp):
            raise ValueError(
                f"outputs[{k}] must have logical shape {shp} (padded {_tile_padded(shp)}) for T={T}, layout "
                f"{layout!r} (allocate_outputs), got {got} (padded {got_p})"
            )
        if int(o.buffer_num_pages()) < pages:  # belt and braces: never write past the buffer
            raise ValueError(f"outputs[{k}] buffer has {o.buffer_num_pages()} pages, the writer addresses {pages}")
    return T, n_tiles


def _split_work(grid, n_tiles: int, max_cores: Optional[int] = None):
    """[(core, n, start)] in the stock op's order (column-major over the grid: core i = (i // gy, i % gy))."""
    gx, gy = int(grid.x), int(grid.y)
    if max_cores is not None and int(max_cores) < 1:
        raise ValueError(f"max_cores must be >= 1, got {max_cores}")
    n_cores = min(n_tiles, gx * gy, int(max_cores) if max_cores is not None else gx * gy)
    q, r = divmod(n_tiles, n_cores)
    out, start = [], 0
    for i in range(n_cores):
        n = q + (1 if i < r else 0)
        out.append((ttnn.CoreCoord(i // gy, i % gy), n, start))
        start += n
    return out


def _core_range_set(cores) -> ttnn.CoreRangeSet:
    by_x = {}
    for c in cores:
        by_x.setdefault(int(c.x), []).append(int(c.y))
    ranges = []
    for x, ys in sorted(by_x.items()):
        ys = sorted(ys)
        y0 = prev = ys[0]
        for y in ys[1:] + [None]:
            if y is None or y != prev + 1:
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y0), ttnn.CoreCoord(x, prev)))
                if y is not None:
                    y0 = y
            if y is not None:
                prev = y
    return ttnn.CoreRangeSet(ranges)


def _cb(index: int, n_tiles: int, cores: ttnn.CoreRangeSet) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=n_tiles * TILE_BYTES_FP32,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=TILE_BYTES_FP32)
        ],
    )


def _compute_config() -> ttnn.ComputeConfigDescriptor:
    """The compute kernel's config. Not built from ``cfg.compute_config("mhc")`` (README §4) on purpose:
    ``ttnn.generic_op`` takes a ``ComputeConfigDescriptor``, and the kernel needs a per-CB ``UnpackToDestFp32`` on the
    mixes / consts / scratch CBs (exact fp32 unpack-to-dest for the transposes; the default unpack truncates to TF32),
    which the role's ``WormholeComputeKernelConfig`` cannot express. The other values are those of the ``mhc`` role
    (HiFi4, fp32 dest acc, approx off; ``test_host_consts_and_golden`` asserts they agree) and are requirements of the
    kernel rather than tuning knobs: fp32 dest acc keeps DST 32-bit (the SFPU math and the in-dest transposes are fp32);
    fidelity and approx mode do not affect it (no FPU math; the SFPU functions are called with explicit templates)."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = getattr(ttnn.MathFidelity, COMPUTE_FIDELITY)
    cfg.fp32_dest_acc_en = COMPUTE_FP32_ACC
    cfg.math_approx_mode = COMPUTE_APPROX
    cfg.dst_full_sync_en = False
    modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    for cb in (CB_MIXES, CB_CONSTS, CB_TMP):
        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32  # exact fp32 unpack (else TF32 truncation)
    cfg.unpack_to_dest_mode = modes
    return cfg


_SRC_TAG: Optional[str] = None
# (options, mesh uid, buffer addresses) -> (TensorSpecs it was validated with, ProgramDescriptor); bounded, see below
_DESC_CACHE: "dict" = {}
_DESC_CACHE_MAX = 4096
# program key (everything that defines the compiled program: CT args incl. accessor page sizes, grid, core count) ->
# program hash (runtime args excluded, as ttnn.compute_program_descriptor_hash)
_HASH_CACHE: "dict" = {}


def _options_key(iters, mode, div_refine, num_halves, max_cores, layout):
    return (layout, int(iters), mode, bool(div_refine), num_halves, max_cores)


def program_descriptor(
    mixes: ttnn.Tensor,
    consts: ttnn.Tensor,
    *outs: ttnn.Tensor,
    iters: int = DEFAULT_ITERS,
    mode: str = "full",
    div_refine: bool = False,
    num_halves: Optional[int] = None,
    max_cores: Optional[int] = None,
    layout: str = "stock",
) -> ttnn.ProgramDescriptor:
    """The ``ttnn.ProgramDescriptor`` of one call (exposed for ``ttnn.experimental.prepare_generic_op`` / tests); always
    validates the inputs and outputs (:func:`_validate`).

    Buffer addresses are runtime args (the program hash excludes them), so one compiled program serves every site
    and call of a given shape; ``ttnn.generic_op`` patches the addresses on a program-cache hit."""
    return _descriptor_and_program_key(
        mixes,
        consts,
        outs,
        iters=iters,
        mode=mode,
        div_refine=div_refine,
        num_halves=num_halves,
        max_cores=max_cores,
        layout=layout,
    )[0]


def _descriptor_and_program_key(mixes, consts, outs, *, iters, mode, div_refine, num_halves, max_cores, layout):
    """``(descriptor, program key)``; the key is everything the compiled program depends on (see the end)."""
    global _SRC_TAG
    if _SRC_TAG is None:
        _SRC_TAG = _sources_tag()
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if int(iters) < 0:
        raise ValueError(f"iters must be >= 0, got {iters}")
    T, n_tiles = _validate(mixes, consts, outs, layout)
    if num_halves is None:
        num_halves = 1 if (n_tiles == 1 and T <= 16) else 2
    if num_halves not in (1, 2) or (num_halves == 1 and not (n_tiles == 1 and T <= 16)):
        raise ValueError(f"num_halves must be 2, or 1 for a single tile with T <= 16 (T={T}, tiles={n_tiles})")
    grid = mixes.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    work = _split_work(grid, n_tiles, max_cores)
    cores = _core_range_set([c for c, _, _ in work])

    reader_ct = [CB_MIXES, CB_CONSTS]
    reader_ct += list(ttnn.TensorAccessorArgs(mixes).get_compile_time_args())
    reader_ct += list(ttnn.TensorAccessorArgs(consts).get_compile_time_args())
    writer_ct = [CB_PRE, CB_POST, CB_COMB] + ([] if layout == "stock" else [CB_WSCRATCH])
    for t in outs:
        writer_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    compute_ct = [
        int(iters),
        int(num_halves),
        _f32_bits(PRE_POST_CLAMP),
        _f32_bits(RES_CLAMP),
        _f32_bits(SUM_FLOOR),
        _f32_bits(H_POST_COEFF),
        _MODES[mode],
        1 if div_refine else 0,
    ]

    # Common runtime args only (no per-core args): every core derives its tile range from its logical coordinate,
    # so the host cost of a launch is independent of the core count (per-core args cost ~6 us per core on 32 chips).
    n_cores = len(work)
    reader_common = [mixes.buffer_address(), consts.buffer_address(), n_tiles, n_cores, gy]
    writer_common = [o.buffer_address() for o in outs] + [n_tiles, n_cores, gy]
    compute_common = [n_tiles, n_cores, gy]

    defines = [("MOTIF_SINKHORN_SRC", _SRC_TAG)]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(READER_SRC),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=reader_ct,
        defines=defines,
        common_runtime_args=reader_common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(WRITER_SRC if layout == "stock" else WRITER_WRNC_SRC),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=writer_ct,
        defines=defines,
        common_runtime_args=writer_common,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(COMPUTE_SRC),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=compute_ct,
        defines=defines,
        common_runtime_args=compute_common,
        config=_compute_config(),
    )
    cbs = [
        _cb(CB_MIXES, 2, cores),
        _cb(CB_CONSTS, 2, cores),
        _cb(CB_PRE, 2, cores),
        _cb(CB_POST, 2, cores),
        _cb(CB_COMB, 2, cores),
        _cb(CB_TMP, 3, cores),
    ]
    if layout == "wrnc":
        cbs.append(_cb(CB_WSCRATCH, N_WSCRATCH, cores))
    desc = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    # Program key: everything the compiled program depends on (CT args incl. the accessors' page sizes / DRAM-vs-L1
    # flags, compute CT args, grid and core count = core ranges; sources, defines, CB sizes and the compute config are
    # fixed per layout in this process). Runtime args (addresses, n_tiles) are excluded, as in the program hash.
    prog_key = (layout, tuple(reader_ct), tuple(writer_ct), tuple(compute_ct), gx, gy, n_cores)
    return desc, prog_key


_DEFAULT_OPTIONS = dict(
    iters=DEFAULT_ITERS, mode="full", div_refine=False, num_halves=None, max_cores=None, layout="stock"
)


def _lookup_key(mixes, consts, outs, kw):
    """``(dict key, tensor specs)`` of one call: the hashable parts (options, ``MeshDevice.id()`` and buffer address of
    each tensor) and the ``TensorSpec``s (logical + padded shape, dtype, layout, tile, memory config), which are only
    ``==``-comparable and are checked on a hit."""
    tensors = (mixes, consts, *outs)
    key = (
        _options_key(kw["iters"], kw["mode"], kw["div_refine"], kw["num_halves"], kw["max_cores"], kw["layout"]),
        tuple(_device_uid(t.device()) for t in tensors),
        tuple(t.buffer_address() for t in tensors),
    )
    return key, tuple(t.spec for t in tensors)


def _cached_descriptor(mixes, consts, outs, **kw) -> ttnn.ProgramDescriptor:
    """Host-overhead cut for eager calls: the descriptor (incl. its runtime args) is a pure function of the options, the
    mesh, the tensor specs and the buffer addresses, which the allocator reuses in steady state. An entry is reused
    only if every ``TensorSpec`` equals the one it was validated with, so a different tensor at a recycled address (or
    on a reopened mesh) is re-validated. The program hash is memoized per program key (``custom_program_hash``) so
    ``generic_op`` skips the descriptor walk."""
    for k, v in _DEFAULT_OPTIONS.items():
        kw.setdefault(k, v)
    try:
        key, specs = _lookup_key(mixes, consts, outs, kw)
    except Exception:  # a host / non-ttnn tensor: report it like the validation does
        _validate(mixes, consts, outs, kw["layout"])
        raise
    ent = _DESC_CACHE.get(key)
    if ent is not None and ent[0] == specs:
        return ent[1]
    desc, prog_key = _descriptor_and_program_key(mixes, consts, outs, **kw)
    h = _HASH_CACHE.get(prog_key)
    if h is None:
        h = _HASH_CACHE[prog_key] = ttnn.compute_program_descriptor_hash(desc)
    desc.custom_program_hash = h
    if len(_DESC_CACHE) >= _DESC_CACHE_MAX:
        _DESC_CACHE.clear()
    _DESC_CACHE[key] = (specs, desc)
    return desc


def allocate_outputs(mixes: ttnn.Tensor, memory_config=None, layout: str = "stock"):
    """Fresh fp32 TILE output tensors for ``layout`` (stock: the ``mhc_split_sinkhorn`` output specs)."""
    T = int(mixes.shape[-2])
    mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    try:
        dev = mixes.device()
    except Exception as e:
        raise ValueError("mixes must be a device tensor") from e
    if dev is None:
        raise ValueError("mixes must be a device tensor")
    return tuple(
        ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
        for shape in output_shapes(T, layout)
    )


def motif_sinkhorn(
    mixes: ttnn.Tensor,
    consts: ttnn.Tensor,
    *,
    iters: int = DEFAULT_ITERS,
    memory_config=None,
    outputs: Optional[Sequence[ttnn.Tensor]] = None,
    mode: str = "full",
    div_refine: bool = False,
    num_halves: Optional[int] = None,
    max_cores: Optional[int] = None,
    layout: str = "stock",
) -> Tuple[ttnn.Tensor, ...]:
    """Coefficients from raw mixes ``[..., T, 24..32]`` fp32 and per-site ``consts``.

    ``layout="stock"``: ``(h_pre [T,4], h_post [T,4], H [T,16])``; ``layout="wrnc"``: ``(w_pre [1,4,T,1],
    w_post [4,5,T,1])``, the ``attn_res_weighted_reduce_nc`` weights (see the module docstring). ``outputs`` lets a
    caller reuse preallocated output tensors with exactly the specs of :func:`allocate_outputs` (validated; a mismatch
    raises ``ValueError`` before anything is launched)."""
    outs = tuple(outputs) if outputs is not None else allocate_outputs(mixes, memory_config, layout)
    desc = _cached_descriptor(
        mixes,
        consts,
        outs,
        iters=iters,
        mode=mode,
        div_refine=div_refine,
        num_halves=num_halves,
        max_cores=max_cores,
        layout=layout,
    )
    ttnn.generic_op([mixes, consts, *outs], desc)
    return outs


class MotifSinkhorn:
    """One mHC site's coefficient op (bring-up / test helper; the production path is ``tt/mhc.py``, which owns the
    per-site constants and calls :func:`motif_sinkhorn`): holds the device constants; ``site(mixes) -> outputs``.

    ``cfg`` (a ``MotifTTConfig``, optional): checks ``cfg.mhc_h_post_coeff`` against the kernel's fixed 1.0, takes
    ``iters`` from ``cfg.sinkhorn_iters`` (unless given) and uploads the consts with ``weights.as_tensor`` (README §10
    rule 1; no cache). Without it: ``iters`` 20 and :func:`consts_to_device`.

    ``reuse_outputs=True`` keeps one set of output tensors per input shape and overwrites it on every call (saves the
    ~105 us of host time of three output allocations per eager call on the 32-chip mesh); the caller must consume
    (or copy) the coefficients before calling the same site object again. Default: fresh outputs per call."""

    def __init__(
        self,
        mesh_device,
        scalars_or_consts: Union[torch.Tensor, Mapping[str, torch.Tensor]],
        *,
        iters: Optional[int] = None,
        memory_config=None,
        reuse_outputs: bool = False,
        cfg=None,
    ):
        if cfg is not None:
            coeff = float(cfg.mhc_h_post_coeff)
            if coeff != H_POST_COEFF:
                raise ValueError(f"h_post coefficient {coeff} != {H_POST_COEFF} (the kernel's fixed coefficient)")
        if iters is None:
            iters = int(cfg.sinkhorn_iters) if cfg is not None else DEFAULT_ITERS
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.iters = int(iters)
        self.memory_config = memory_config
        self.reuse_outputs = bool(reuse_outputs)
        self._outputs: dict = {}
        self.consts_host = (
            torch.as_tensor(scalars_or_consts, dtype=torch.float32).reshape(2 * TILE, TILE)
            if isinstance(scalars_or_consts, torch.Tensor)
            else consts_from_scalars(scalars_or_consts)
        )
        if cfg is not None:
            from models.demos.motif3.tt import weights as W

            self.consts = W.as_tensor(
                self.consts_host, mesh_device=mesh_device, cfg=cfg, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
            )
        else:
            self.consts = consts_to_device(self.consts_host, mesh_device)

    @classmethod
    def from_source(cls, mesh_device, source, layer_idx: int, site: str, **kw) -> "MotifSinkhorn":
        """``site`` in {"mhc_attn", "mhc_ffn"}; scalars via ``weights.mhc_scalars`` (HF names); ``kw`` as ``__init__``
        (e.g. ``cfg=``)."""
        from models.demos.motif3.tt import weights as W

        if site not in ("mhc_attn", "mhc_ffn"):
            raise ValueError(f"site must be mhc_attn or mhc_ffn, got {site!r}")
        return cls(mesh_device, W.mhc_scalars(source, W.hf_name(layer_idx, site)), **kw)

    def golden(self, p: torch.Tensor, dtype: torch.dtype = torch.float32):
        return motif_mhc_maps_torch(p, self.consts_host, self.iters, dtype=dtype)

    def __call__(self, mixes: ttnn.Tensor, **kw):
        kw.setdefault("iters", self.iters)
        kw.setdefault("memory_config", self.memory_config)
        if self.reuse_outputs and kw.get("outputs") is None:
            layout = kw.get("layout", "stock")
            key = (tuple(int(d) for d in mixes.shape), str(kw["memory_config"]), layout)
            outs = self._outputs.get(key)
            if outs is None:
                outs = self._outputs[key] = allocate_outputs(mixes, kw["memory_config"], layout)
            kw["outputs"] = outs
        return motif_sinkhorn(mixes, self.consts, **kw)

    def release(self):
        """Deallocate the device constants and any reused outputs."""
        for outs in self._outputs.values():
            for o in outs:
                ttnn.deallocate(o)
        self._outputs.clear()
        ttnn.deallocate(self.consts)
