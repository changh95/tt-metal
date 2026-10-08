# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 mHC site on the BH Galaxy: ``MHCSite`` (design §2.3.3, FEASIBILITY_REPORT H1, WAVE_A_REVIEW §5.3 MHC-1..6).

One instance per (layer, site in {"mhc_attn", "mhc_ffn"}); 106 per model. Math (HF ``modeling_motif.py:226-249,
1205-1244``; ``reference.modules.MHCLayer``), per token, on the 4-stream residual ``X`` (stream ``i`` = ``X_i``):

    p      = RMSNorm_16384(X_flat; gamma, eps 1e-6) @ (W_pre || W_post || W_res)^T          [24] fp32 ("mixes")
    h_pre  = sigmoid(clamp(alpha_pre  * p_pre  + b_pre,  -10, 10))                          [4]
    h_post = 1.0 * sigmoid(clamp(alpha_post * p_post + b_post, -10, 10))                    [4]
    H      = Sinkhorn20(exp(clamp(alpha_res * p_res + B_res, -20, 20)))  (rows, then cols, sums >= 1e-8)   [4, 4]
    x_red  = sum_i h_pre_i X_i                    (fp32 accumulate, one bf16 cast)   -> sublayer input
    X'     = H @ X + h_post (x) out               (fp32 accumulate, one bf16 cast)   -> next residual

Device formulation
------------------
1. **Mixes p (MHC-2).** Host: ``fn = (W_pre||W_post||W_res) * gamma * 128`` folded in fp32 and rounded once to bf16
   (x128 = sqrt(16384) is exact and moves the 1/16384 of the mean into the weights), so that
   ``p = (X @ fn^T) * rsqrt(sum(X^2) + 16384 * 1e-6)`` (the rsqrt commutes with the linear map). **One** weight tensor
   per site, ``proj [1, 1, 128, 4096]`` bf16 in "stream rows" (row ``32 s + c`` = mix ``c`` of stream ``s``'s 4096
   columns of ``fn``; rows ``32 s + 24 .. 32 s + 31`` zero), serves both modes through ``transpose_b`` (measured bitwise
   equal to separate per-stream blocks ``[1,4,4096,32]`` / a prefill ``[4096, 128]`` wall, at the same latency):

   * decode (T <= 32, one tile row): **split-K**. ``X [1,4,32,4096]`` (the 8 logical rows re-viewed as 32) is viewed
     as ``[1,NB,32,16384/NB]`` and ``proj`` as ``[1,NB,32,16384/NB]`` (zero copy: with one tile row both keep their
     TILE page order); one batched matmul ``X_b @ proj_b^T`` with an explicit ``MatmulMultiCoreReuseProgramConfig``
     (NB = 32 cores, the whole 16-tile K chunk in one block) gives NB fp32 partials (8 us). The auto config for a
     batched in1 uses ``in0_block_w = 1`` (148 us); K-blocked explicit configs reload partials through TF32 (biased up
     to -1.2e-3). ``sum(X^2)`` partials: ``ttnn.rms_norm_pre_all_gather`` on the same view (fp32, column 0; 10 us).
   * prefill (T = S > 32): ``X [1,4,S,4096]`` viewed as ``[1,1,4S,4096]`` times ``proj^T [4096, 128]`` (one
     non-batched matmul; the mcast factories reload K-blocked partials through an UnpackToDestFp32 alias, so they are
     exact); its diagonal blocks ``X_s @ fn_s^T`` are the 4 per-stream partials. ``sum(X^2)`` per stream:
     ``rms_norm_pre_all_gather``. (The batched-in1 reuse factory needs the whole K per block and mis-computes with > 1
     output block per core.)
   * ``finalize_mixes`` (one ``ttnn.generic_op``; decode: 1 core; prefill: 32-token tiles split over the grid) sums the
     NB (decode) / 4 diagonal (prefill) partials exactly (UnpackToDestFp32 + SFPU adds), applies ``rsqrt`` and the
     column-broadcast multiply. ``finalize="ttnn"`` is the op form (decode: 2 accurate ``ttnn.sum(dim=1)`` + add/rsqrt +
     multiply, the same values to 1.2e-7; prefill: diagonal mask + 3 sums + the same; the ``ttnn.sum(dim=1)`` over the
     streams alone takes 6.4 ms at S = 32768).
   * Precision (``test_mhc_probe_*``): the FPU dot products have a ~3.3e-4 relative RMS floor on ``p`` (the intra-tile
     adder; not the products); everything else is exact fp32 to ~1e-7 (``fast_reduce_nc`` is NOT: -3.5e-4 biased, not
     used). Real weights / streams: ``p`` within 0.8-2.4e-4 of the fp64 golden.

2. **Coefficients (MHC-3 / MHC-6)**, ``sinkhorn=``:

   * ``"motif"`` (the default and the production path): ``models.demos.motif3.tt.kernels.sinkhorn_motif.motif_sinkhorn``
     (the kernel module, owned separately) -- Motif's exact semantics (+-10 / +-20 clamps, 1e-8 floors, eps 0, h_post =
     1.0 sigmoid) in fp32 SFPU math from the raw mixes ``p`` and per-site alpha / bias constants: ~3 us. **No silent
     fallback**: if the kernel module cannot be imported, the constructor raises (with the ImportError chained); any
     other error inside the kernel module propagates.
   * ``"stock"`` (MHC-3, the draft-1 pre-clamped mode; explicit opt-in for comparison / debugging only): ``L = clamp(p *
     alpha_vec + b_vec, lo_vec, hi_vec)`` (exact fp32 ops, +-10 on pre/post, +-20 on res), then
     ``ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(L, consts, 4, 20, 0.0)`` with identity SEL and zero base
     (``preclamped_consts()``); ``h_pre = pre``, ``h_post = post / 2`` (the op's post is 2 sigma), ``H = comb``
     row-major. No 1e-8 floor and TF32 arithmetic inside: max|dH| 5.1e-3 on real logits (it misses the 5e-3 MHC-5
     bound), ~57 us more per decode site. **Never** the direct mode (alpha / bias in SEL / base: max|dH| 0.82).

   ``h_pre [T,4]``, ``h_post [T,4]``, ``H [T,16]`` (``H[t, 4i+j]`` = ``H_t[i][j]``) are laid out for the mixing as
   ``w_pre [1,4,T,1]`` and ``w_post [4,5,T,1]`` = ``[H | h_post]`` (``w_post[r, c, t]`` = ``H_t[r][c]`` for c < 4,
   ``h_post_t[r]`` for c = 4) by ``coefficient_layout`` (a data-movement generic_op: fp32 bit moves, the stock op's /2
   as an exponent decrement and, for ``mix="wr"``, the round-to-nearest-even to TF32 of step 3; ~6.5 us) or
   ``glue="ttnn"`` (3 exact permutes + 1 concat, 17 us; it does not round, so with ``mix="wr"`` the FPU truncation of
   step 3 applies: a layout-debug path only).

3. **Stream mixing (MHC-4)**, ``mix="wr"`` (default): the FPU weighted reduce of
   ``ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc`` (HiFi4, fp32 dest accumulation). Pre: the stock
   op, ``x_red = WR(X, w_pre)`` -> ``[1,1,T,4096]``. Post: ``post_mix``, one generic_op with the stock op's compute
   kernel and a two-input reader (``X'[r] = sum_c X_c w[r,c] + out w[r,4]``; ``X`` and ``out`` are read in place, no
   ``[X | out]`` concat), written as ``[1,4,T,4096]`` directly; bitwise equal to the stock op on the concat
   (``post_concat=True``). **Weight precision**: the FPU unpacks the fp32 weights to TF32 by *truncating* their 13 low
   mantissa bits, which biased every output low by ~-3.3e-4 relative (synthetic and real data alike). The layout kernel
   therefore rounds the weights to TF32 (nearest-even) first: unbiased, <= 2^-11 relative error per weight. The bf16
   ``X`` / ``out`` are exact in TF32, the products exact in fp32, the 4 / 5-term sum is fp32 and rounded once to bf16:
   HF's single-rounding fp32 einsum, with TF32-rounded coefficients. Measured on real decode sites: the outputs equal
   the correctly rounded exact result of the device weights on >= 99.98% of the elements; the op's packer rounds
   fp32 -> bf16 ties *away from zero* (``ttnn.typecast`` rounds them to even), which touches 1e-5 (X') to 3e-4 (x_red)
   of the real elements (bias <= 2e-7). Against the correctly rounded result of the *unrounded* coefficients the
   mixing bias is +4e-6 (x_red) / +7e-6 (X') over the 56 real decode sites (TF32-truncated weights: -3.6e-4); weights
   just below 1 (a near-identity H at deep attention sites, sigmoid(10)) round to 1.0, so the rounded rows of H sum to
   1 + eps (eps < 2^-12): up to +1.5e-5 on such a site before the bf16 rounding of X'. In decode the weights live in L1
   (``mix_l1``; the op re-reads the whole weight set on every core). ``mix="composite"`` (fp32 multiply +
   ``sum(dim=1)`` on the unrounded weights, with the concat) is the exact debug path.

Measured on this Galaxy (``tests/unit/test_mhc.py``, ``logs/dev/20261001_234900_mhc_fix_final.log``; TORUS_Y fabric,
no CCL here): decode site (pre + post, 8 lanes) **76 us traced** (projection 8 + statistics 10 + finalize 17 +
Sinkhorn 3.4 + layout 7 + pre 9.6 + post 24.7), 1.07 ms eager; with the stock op on the ``[X | out]`` concat
(``post_concat=True``) 84 us; the stock backend 137 us. Real weights and streams (layers 0-5 and 28-35, both sites, 2
prompts, 56 sites): max|dH| 5.1e-4, max|dh_pre| 6.1e-4, max|dh_post| 3.4e-4 vs the fp64 reference (HF's own bf16
path is off by up to 9.8e-3); x_red / X' PCC >= 0.9999984; the maps from the device's own p within 2.4e-7 of fp32;
the mixing within 2.2 / 4.2 bf16 ulps of the largest summand, bias +4.3e-6 / +7.2e-6 over all sites. Prefill, eager
per site: S = 4096 2.9 ms; S = 32768 21.5 ms (mixes 10.5 = matmul 7.3 + statistics 2.9 + finalize; post 7.9; the op
forms take 20.3 ms for the mixes and 14.1 ms for the post on the concat).

Memory: per site 1 MB of bf16 projection weights (``proj``; the 24 mixes padded to 32 rows per stream) + 8 KB of fp32
constants: 106 MB per chip for the 106 sites. Prefill at S = 32768 allocates, besides the caller's ``X`` (1.07 GB) and
``out`` (0.27 GB): ``X'`` 1.07 GB, ``x_red`` 0.27 GB, ``w_post`` 84 MB and ``w_pre`` 17 MB (fp32 ``[.., T, 1]``
tiles, 32x column padding), and transient fp32 intermediates (the projection output 67 MB, the ``sum(X^2)`` partials
17 MB, ``p`` and the maps 4 MB each). No ``[X | out]`` concat (1.34 GB) unless ``post_concat=True``. The module has no
row chunking of its own: design §3.3 chunks the per-token sublayers at the decoder level, and every op here is
row-local, so a caller may pass row chunks of any multiple of 32 rows.

Tensor contract (README CONVENTIONS §3): ``X`` ``[1, 4, T, 4096]`` bf16 TILE DRAM (decode T = 8 lanes of the chip's DP
row, replicated over the 8 TP chips of the row; prefill T = the bucket S, replicated on all chips); ``out`` (sublayer
output after its all-reduce) ``[1, 1, T, 4096]`` bf16; ``x_red`` ``[1, 1, T, 4096]`` bf16 DRAM; ``X'`` ``[1, 4, T, 4096]``
bf16 DRAM. Inputs are never deallocated. Every op runs identically on every chip (no CCL), so replicas are bitwise
identical. The rows of the 32-row tile beyond the T lanes are computed too (row-local, ignored; NaN / Inf there do not
reach the real rows).

Trace safety: decode has no host round trip and fixed shapes; views are metadata only. The generic_ops of this module
(``finalize_mixes``, ``coefficient_layout``, ``post_mix``) and the kernel module's Sinkhorn take their buffer addresses
as common runtime args, which ``generic_op`` re-applies on every program-cache hit, but they need **one eager call per
shape before a trace capture**: their kernel binaries reach the device at the first enqueue ("Cannot load new binaries
during trace capture" otherwise). A decoder warm-up that runs every bucket / the decode step once eagerly covers it.

Zero-copy views (README §10 rule 2): ``_view_logical`` relies on the legacy volume-changing view of ``ttnn.reshape``
(``reshape.cpp:725``, kept upstream as a workaround for issues 15137 / 15558) to turn the 8 decode rows into the 32
padded rows; the split-K views use ``ttnn.experimental.view`` with a changed last dimension, which the op documents as
unsupported (``view_nanobind.cpp:26``; it is exact for TILE tensors whose page order is unchanged, i.e. one tile row).
Both are metadata-only today: the constructor checks the weight view and ``tests/unit/test_mhc.py::test_mhc_kernels``
the input views (same buffer address); an upstream change would fail loudly.

TT weight cache (README §7): ``{site}.proj_rows_x128`` (the x128-scaled, gamma-folded stream-rows tensor above -- NOT
the README's unscaled ``mhc_projection_blocks`` layout, hence the distinct name; the old ``{site}.blocks`` /
``{site}.wall`` names are unused), ``{site}.motif_consts`` ``[64, 32]`` fp32 (motif) or ``{site}.{alpha,bias,lo,hi}_row``
``[1, 1, 1, 32]`` fp32 (stock), per layer. A converter should build ``MHCSite(..., cache=True)`` instead of re-deriving
the transform.

Interface::

    site = MHCSite(mesh_device, cfg, layer_idx, "mhc_attn", source=src, cache=True)
    x_red, coeffs = site.pre(X)            # coefficients + pre mix
    ...                                    # out = sublayer(norm(x_red))
    X_next = site.post(X, out, coeffs)     # post mix; deallocates coeffs (release=False keeps them)

Import rule: module-level imports are stdlib, torch, ttnn and the shared ``tt/`` infra only; the Motif Sinkhorn kernel
module is imported lazily (``motif_kernel_module()``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

import ttnn

from . import weights as W
from .model_config import COMPUTE_ROLES, MHC_DECODE_MODES, ComputeRole, MotifTTConfig

N_STREAMS = 4
N_MIXES = (2 + N_STREAMS) * N_STREAMS  # 24
TILE = 32
PRE_POST_CLAMP = 10.0  # HF modeling_motif.py:236-240
RES_CLAMP = 20.0  # :226-233
DECODE_SPLIT_K = 32  # K-chunks of the decode projection (16384 / 32 = 512 = 16 tiles per core)
SITES = ("mhc_attn", "mhc_ffn")
SINKHORN_IMPLS = ("motif", "stock")
# TT-cache name of the projection weights: x128-scaled stream rows (see the module docstring). Deliberately not the
# README §7 example name "<site>.blocks", which documents the unscaled mhc_projection_blocks layout.
PROJ_CACHE_NAME = "proj_rows_x128"


# =====================================================================================================================
# host-side constants (pure torch)
# =====================================================================================================================
def build_sinkhorn_consts(n: int, scale, base: torch.Tensor) -> torch.Tensor:
    """``[8, 32, 32]`` fp32 constants of ``ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn`` (MHC-1; verbatim copy
    of ``tests/unit/gates/goldens.build_sinkhorn_consts``, itself a port of the DeepSeek-V4 d_p builder
    ``models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py:55-117``; copied, not imported, per the import rule).

    Tile order: 0 SEL_pre 1 SEL_post 2 SEL_comb 3 base_pre 4 base_post 5 base_comb 6 RB 7 CB. ``scale`` = (a_pre, a_post,
    a_res) folded into SEL; ``base`` [24] = (b_pre | b_post | B_res row-major). Motif's pre-clamped mode uses
    ``scale = (1, 1, 1)`` and ``base = 0`` (the logits are formed and clamped on device before the call)."""
    W_ = TILE
    a_pre, a_post, a_res = (float(scale[0]), float(scale[1]), float(scale[2]))
    base = torch.as_tensor(base, dtype=torch.float32).reshape(-1)
    sel_pre = torch.zeros(W_, W_)
    sel_post = torch.zeros(W_, W_)
    sel_comb = torch.zeros(W_, W_)
    for p in range(n):
        sel_pre[p, p] = a_pre
        sel_post[n + p, p] = a_post
    for p in range(n * n):
        sel_comb[2 * n + p, p] = a_res
    base_pre = torch.zeros(W_, W_)
    base_post = torch.zeros(W_, W_)
    base_comb = torch.zeros(W_, W_)
    base_pre[:, :n] = base[0:n]
    base_post[:, :n] = base[n : 2 * n]
    base_comb[:, : n * n] = base[2 * n : 2 * n + n * n]
    RB = torch.zeros(W_, W_)
    CB = torch.zeros(W_, W_)
    for p in range(n * n):
        pi, pj = divmod(p, n)
        for q in range(n * n):
            qi, qj = divmod(q, n)
            if qi == pi:
                RB[q, p] = 1.0
            if qj == pj:
                CB[q, p] = 1.0
    for p in range(n * n, W_):
        RB[p, p] = 1.0
        CB[p, p] = 1.0
    return torch.stack([sel_pre, sel_post, sel_comb, base_pre, base_post, base_comb, RB, CB], dim=0)


def preclamped_consts(n: int = N_STREAMS) -> torch.Tensor:
    """Identity SEL, zero base: the stock op then reads the already-formed (and clamped) logits (MHC-1 / MHC-3)."""
    return build_sinkhorn_consts(n, (1.0, 1.0, 1.0), torch.zeros((2 + n) * n))


def logit_vectors(scalars: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Per-column fp32 rows ``[1, 1, 1, 32]`` for ``L = clamp(p * alpha + b, lo, hi)`` (stock path, MHC-3):
    ``alpha = [a_pre x4 | a_post x4 | a_res x16 | 0 x8]``, ``b = [b_pre | b_post | B_res row-major | 0]``,
    ``lo / hi = -/+[10 x8 | 20 x16 | 0 x8]`` (Motif's clamps; the pad columns stay 0)."""
    f = lambda k: torch.as_tensor(scalars[k], dtype=torch.float32).reshape(-1)  # noqa: E731
    alpha = torch.zeros(TILE)
    bias = torch.zeros(TILE)
    bound = torch.zeros(TILE)
    alpha[0:4] = f("alpha_pre")[0]
    alpha[4:8] = f("alpha_post")[0]
    alpha[8:24] = f("alpha_res")[0]
    bias[0:4] = f("bias_pre")
    bias[4:8] = f("bias_post")
    bias[8:24] = f("bias_res")
    bound[0:8] = PRE_POST_CLAMP
    bound[8:24] = RES_CLAMP
    row = lambda v: v.reshape(1, 1, 1, TILE).contiguous()  # noqa: E731
    return {"alpha": row(alpha), "bias": row(bias), "lo": row(-bound), "hi": row(bound)}


def folded_projection(source, prefix: str, *, hidden: int = 4096, n_streams: int = N_STREAMS) -> torch.Tensor:
    """``fn = 128 * (W_pre||W_post||W_res) * gamma`` ``[24, 16384]`` fp32 (the x128 = sqrt(16384) is exact)."""
    proj = W.mhc_projection_from_source(source, prefix)  # [24, 16384]
    gamma = source.get(f"{prefix}.rms_norm.weight")
    return W.mhc_fused_projection(proj[:4], proj[4:8], proj[8:], gamma) * float(math.sqrt(hidden * n_streams))


def projection_blocks(source, prefix: str, *, hidden: int = 4096, n_streams: int = N_STREAMS) -> torch.Tensor:
    """Per-stream blocks ``[1, 4, 4096, 32]`` of ``fn`` (fp32; ``weights.mhc_projection_blocks`` of the x128 fold):
    block ``i`` = stream ``i``'s 4096 rows of ``fn^T`` with the 24 mixes in columns 0..23 (pre 0-3, post 4-7, res 8-23,
    ``res[4i + j]`` = ``H[i][j]``) and 8 zero columns. (One bf16 rounding of the gamma fold costs 5-7e-5 relative on
    ``p`` on real weights, well below the 3.3e-4 FPU dot-product floor; a hi/lo bf16 split was measured and dropped.)"""
    fn = folded_projection(source, prefix, hidden=hidden, n_streams=n_streams)
    return W.mhc_projection_blocks(fn, n_streams=n_streams, pad_to=TILE)


def projection_rows(blocks: torch.Tensor) -> torch.Tensor:
    """The device weights ``proj [1, 1, 4 * 32, 4096]`` ("stream rows") from the blocks ``[1, 4, 4096, 32]``:
    ``proj[0, 0, 32 s + c, k] = blocks[0, s, k, c] = fn[c, 4096 s + k]``. Used with ``transpose_b``: decode views it as
    ``[1, NB, 32, 16384 / NB]`` (page order = the flattened 16384 index), prefill multiplies ``X[4S, 4096] @ proj^T``."""
    b = blocks.reshape(N_STREAMS, -1, TILE)  # [4, 4096, 32]
    return b.permute(0, 2, 1).reshape(1, 1, N_STREAMS * TILE, b.shape[1]).contiguous()


def projection_tensor(source, prefix: str, *, hidden: int = 4096, n_streams: int = N_STREAMS) -> torch.Tensor:
    """``projection_rows(projection_blocks(...))``: the host tensor behind ``MHCSite.proj`` (fp32, bf16 at upload)."""
    return projection_rows(projection_blocks(source, prefix, hidden=hidden, n_streams=n_streams))


def stream_diag_mask() -> torch.Tensor:
    """``[1, 4, 1, 128]`` fp32: 1 on stream ``s``'s own 32 columns of ``X_s @ proj^T``, 0 elsewhere (``finalize="ttnn"``
    prefill path)."""
    m = torch.zeros(1, N_STREAMS, 1, N_STREAMS * TILE)
    for s in range(N_STREAMS):
        m[0, s, 0, TILE * s : TILE * (s + 1)] = 1.0
    return m


def tf32_rne(t: torch.Tensor) -> torch.Tensor:
    """fp32 -> nearest-even TF32 (10 explicit mantissa bits), Inf / NaN unchanged: the host model of the rounding in
    ``coefficient_layout(round_tf32=True)`` (bit-exact)."""
    b = t.float().contiguous().view(torch.int32)
    finite = (b & 0x7F800000) != 0x7F800000
    r = (b + 0x0FFF + ((b >> 13) & 1)) & ~0x1FFF
    return torch.where(finite, r, b).view(torch.float32)


# =====================================================================================================================
# device helpers
# =====================================================================================================================
def _view_logical(t: ttnn.Tensor, logical) -> ttnn.Tensor:
    """Zero-copy change of the logical shape of a TILE tensor inside its padded shape (e.g. ``[1,4,8,4096]`` <->
    ``[1,4,32,4096]``): ``ttnn.reshape(t, logical, padded)`` takes the metadata-only view path when the padded shape
    is unchanged (the legacy volume-changing view, see the module docstring). The padding rows become (or stop being)
    logical data; ops on them are row-local."""
    logical = [int(d) for d in logical]
    if [int(d) for d in t.shape] == logical:
        return t
    padded = [int(d) for d in t.padded_shape]
    return ttnn.reshape(t, ttnn.Shape(logical), ttnn.Shape(padded))


def _bmm_pc(grid: Tuple[int, int], k_tiles: int, per_core_m: int = 1, per_core_n: int = 1):
    """``MatmulMultiCoreReuseProgramConfig`` for the batched decode split-K projection: one output tile per core and the
    whole K in one block (``in0_block_w = k_tiles``: no partial-sum spill / TF32 reload). The auto config for a batched
    in1 uses ``in0_block_w = 1`` (148 us traced, -1.2e-3 biased). Local: model_config has no builder for it (requested
    shared change ``cfg.mhc_decode_proj_pc()``)."""
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(int(grid[0]), int(grid[1])),
        in0_block_w=int(k_tiles),
        out_subblock_h=1,
        out_subblock_w=int(per_core_n),
        per_core_M=int(per_core_m),
        per_core_N=int(per_core_n),
    )


def _free(*ts) -> None:
    for t in ts:
        if isinstance(t, ttnn.Tensor):
            try:
                ttnn.deallocate(t)
            except Exception:
                pass


def _replicated(t, mesh_device, dtype, *, cfg: MotifTTConfig, cache_name: Optional[str], layer):
    return W.as_tensor(
        t, mesh_device=mesh_device, cfg=cfg, dtype=dtype, layout=ttnn.TILE_LAYOUT, cache_name=cache_name, layer=layer
    )


def _mesh_uid(mesh_device):
    """Process-unique identity of a mesh: ``MeshDevice.id()`` comes from a process-wide counter, so a closed and reopened
    mesh never aliases the old one (Python ``id()`` can be reused). Fallback for objects without ``id()``: Python id +
    grid."""
    f = getattr(mesh_device, "id", None)
    if callable(f):
        return ("mesh", int(f()))
    g = mesh_device.compute_with_storage_grid_size()
    return ("py", id(mesh_device), int(g.x), int(g.y))


_SHARED: Dict[tuple, ttnn.Tensor] = {}  # (mesh uid, name) -> constant shared by all sites of that mesh


def _shared_tensor(mesh_device, key: str, make, dtype) -> ttnn.Tensor:
    k = (_mesh_uid(mesh_device), key)
    t = _SHARED.get(k)
    if t is None or not t.is_allocated():
        t = _SHARED[k] = ttnn.from_torch(
            make(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    return t


def _shared_preclamped_consts(mesh_device) -> ttnn.Tensor:
    return _shared_tensor(mesh_device, "preclamped_consts", preclamped_consts, ttnn.float32)


def release_shared(mesh_device=None) -> None:
    """Drop the shared per-mesh constants (``diag_mask``, the stock pre-clamped consts): with an (open) ``mesh_device``
    its entries are deallocated (sites recreate them on their next eager call; do not release while a captured trace
    still uses them); with ``None`` every entry is forgotten (not deallocated: their meshes may be closed)."""
    uid = None if mesh_device is None else _mesh_uid(mesh_device)
    for k in [k for k in _SHARED if uid is None or k[0] == uid]:
        t = _SHARED.pop(k)
        if uid is not None:
            _free(t)


_NUM_CB_SLOTS = 64  # NUM_CIRCULAR_BUFFERS on Blackhole (length of unpack_to_dest_mode)


def _compute_descriptor(role: Optional[ComputeRole] = None, fp32_unpack_cbs=()) -> ttnn.ComputeConfigDescriptor:
    """``ttnn.ComputeConfigDescriptor`` for this module's generic_op compute kernels, from the shared compute role
    (``mhc`` by default: math fidelity and approx mode as the role says; the kernels need fp32 dest accumulation),
    half-sync DEST, ``UnpackToDestFp32`` on ``fp32_unpack_cbs``. Local helper: model_config has no descriptor builder
    (requested shared change)."""
    role = role if role is not None else COMPUTE_ROLES["mhc"]
    if not role.fp32_acc:
        raise ValueError(f"the mHC generic_op kernels need fp32 dest accumulation, role is {role}")
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = getattr(ttnn.MathFidelity, role.fidelity)
    cc.fp32_dest_acc_en = True
    cc.math_approx_mode = bool(role.approx)
    cc.dst_full_sync_en = False
    modes = [ttnn.UnpackToDestMode.Default] * _NUM_CB_SLOTS
    for i in fp32_unpack_cbs:
        modes[int(i)] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = modes
    return cc


def _first_cores(grid, n: int) -> ttnn.CoreRangeSet:
    """The first ``n`` cores in column-major order (core i = (i // gy, i % gy)), as a CoreRangeSet. The kernels split
    their work items by ``core_i = x * grid_y + y`` (common runtime args only: the host cost does not grow with the core
    count)."""
    gy = int(grid.y)
    ranges = []
    full_cols, rest = divmod(n, gy)
    if full_cols:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_cols - 1, gy - 1)))
    if rest:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(full_cols, 0), ttnn.CoreCoord(full_cols, rest - 1)))
    return ttnn.CoreRangeSet(ranges)


def _set_program_hash(cache: Dict[tuple, int], key: tuple, desc) -> None:
    """Memoize the program hash per program key (everything the compiled program depends on; runtime args excluded)."""
    h = cache.get(key)
    if h is None:
        h = cache[key] = ttnn.compute_program_descriptor_hash(desc)
    desc.custom_program_hash = h


def _cb(index: int, n_tiles: int, cores, dtype, tile_bytes: int) -> ttnn.CBDescriptor:
    return ttnn.CBDescriptor(
        total_size=n_tiles * tile_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)],
    )


_F32_TILE = TILE * TILE * 4
_BF16_TILE = TILE * TILE * 2


# ---------------------------------------------------------------------------------------------------------------------
# coefficient layout kernel (ttnn.generic_op, one data-movement kernel; replaces 3 permutes + 1 concat (+ 0.5 scale))
# ---------------------------------------------------------------------------------------------------------------------
# h_pre [T,4], h_post [T,4], H [T,16] (fp32 TILE, one row per token) -> w_pre [1,4,T,1] (tile c: column 0 = h_pre[:, c])
# and w_post [4,5,T,1] (tile (r, c): column 0 = H[:, 4r + c] for c < 4, h_post[:, r] for c = 4). Only column 0 of each
# weight tile is written (attn_res_weighted_reduce_nc broadcasts column 0). fp32 bit moves; POST_HALVE subtracts 1 from
# the fp32 exponent (exact x0.5 for the stock op's 2*sigma); ROUND_TF32 rounds every weight to nearest-even TF32 (the
# FPU of the weighted reduce truncates fp32 operands to TF32, which biases the mix by -3.3e-4; pre-rounded weights pass
# through that truncation unchanged). Work: (token tile row, mixing row) items split over cores (decode: 4 cores).
_LAYOUT_SRC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t TILE_BYTES = 32 * 32 * 4;
constexpr uint32_t FACE_BYTES = 16 * 16 * 4;
// fp32 tile element (r, c) -> word index (4 faces of 16 x 16, row-major in each face)
inline uint32_t widx(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
// fp32 bits -> nearest-even TF32 (10 explicit mantissa bits); Inf / NaN unchanged
inline uint32_t rne_tf32(uint32_t b) {
    return (b & 0x7f800000u) == 0x7f800000u ? b : ((b + 0x0fffu + ((b >> 13) & 1u)) & ~0x1fffu);
}
}  // namespace

// Work item w = (token tile row t, mixing row g in 0..3): w_pre tile g and w_post tiles (g, 0..4). Only faces 0 and 2 of
// each weight tile (they hold column 0) are written.
void kernel_main() {
    const uint32_t pre_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t post_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t comb_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t wpre_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t wpost_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t ht = get_common_arg_val<uint32_t>(5);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(6);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(7);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t post_halve = get_compile_time_arg_val(2);
    constexpr uint32_t round_tf32 = get_compile_time_arg_val(3);
    constexpr auto a_pre = TensorAccessorArgs<4>();
    constexpr auto a_post = TensorAccessorArgs<a_pre.next_compile_time_args_offset()>();
    constexpr auto a_comb = TensorAccessorArgs<a_post.next_compile_time_args_offset()>();
    constexpr auto a_wpre = TensorAccessorArgs<a_comb.next_compile_time_args_offset()>();
    constexpr auto a_wpost = TensorAccessorArgs<a_wpre.next_compile_time_args_offset()>();

    const auto s_pre = TensorAccessor(a_pre, pre_addr, TILE_BYTES);
    const auto s_post = TensorAccessor(a_post, post_addr, TILE_BYTES);
    const auto s_comb = TensorAccessor(a_comb, comb_addr, TILE_BYTES);
    const auto s_wpre = TensorAccessor(a_wpre, wpre_addr, TILE_BYTES);
    const auto s_wpost = TensorAccessor(a_wpost, wpost_addr, TILE_BYTES);

    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t total = ht * 4;
    const uint32_t q = total / num_cores, rem = total % num_cores;
    const uint32_t n = q + (core_i < rem ? 1 : 0);
    const uint32_t start = core_i * q + (core_i < rem ? core_i : rem);

    CircularBuffer cbi(cb_in), cbo(cb_out);
    cbi.reserve_back(3);
    cbo.reserve_back(6);
    const uint32_t l1_in = cbi.get_write_ptr();
    const uint32_t l1_out = cbo.get_write_ptr();
    const uint32_t* in_pre = reinterpret_cast<const uint32_t*>(l1_in);
    const uint32_t* in_post = reinterpret_cast<const uint32_t*>(l1_in + TILE_BYTES);
    const uint32_t* in_comb = reinterpret_cast<const uint32_t*>(l1_in + 2 * TILE_BYTES);
    uint32_t* out = reinterpret_cast<uint32_t*>(l1_out);

    uint32_t loaded = 0xffffffffu;
    for (uint32_t w = start; w < start + n; ++w) {
        const uint32_t t = w >> 2, g = w & 3;
        if (t != loaded) {
            noc_async_write_barrier();  // pending writes still read l1_out (and the input tiles are reloaded)
            noc_async_read(s_pre.get_noc_addr(t), l1_in, TILE_BYTES);
            noc_async_read(s_post.get_noc_addr(t), l1_in + TILE_BYTES, TILE_BYTES);
            noc_async_read(s_comb.get_noc_addr(t), l1_in + 2 * TILE_BYTES, TILE_BYTES);
            noc_async_read_barrier();
            loaded = t;
        } else {
            noc_async_write_barrier();
        }
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t o = widx(r, 0);
            uint32_t v = in_pre[widx(r, g)];  // w_pre tile g
            if constexpr (round_tf32) {
                v = rne_tf32(v);
            }
            out[o] = v;
            for (uint32_t j = 0; j < 4; ++j) {  // w_post tiles (g, 0..3) = H[:, 4g + j]
                uint32_t h = in_comb[widx(r, 4 * g + j)];
                if constexpr (round_tf32) {
                    h = rne_tf32(h);
                }
                out[(1 + j) * 1024 + o] = h;
            }
            uint32_t b = in_post[widx(r, g)];  // w_post tile (g, 4) = h_post[:, g]
            if constexpr (post_halve) {
                const uint32_t e = b & 0x7f800000u;
                if (e != 0 && e != 0x7f800000u) {
                    b -= 0x00800000u;
                }
            }
            if constexpr (round_tf32) {
                b = rne_tf32(b);
            }
            out[5 * 1024 + o] = b;
        }
        for (uint32_t k = 0; k < 6; ++k) {
            const uint64_t dst = k == 0 ? s_wpre.get_noc_addr(g * ht + t) : s_wpost.get_noc_addr((5 * g + k - 1) * ht + t);
            noc_async_write(l1_out + k * TILE_BYTES, dst, FACE_BYTES);  // face 0 (rows 0-15)
            noc_async_write(l1_out + k * TILE_BYTES + 2 * FACE_BYTES, dst + 2 * FACE_BYTES, FACE_BYTES);  // face 2
        }
    }
    noc_async_write_barrier();
}
"""

_LAYOUT_CB_IN, _LAYOUT_CB_OUT = 0, 1
_LAYOUT_HASH: Dict[tuple, int] = {}


def coefficient_layout(h_pre, h_post, H, *, post_halve: bool = False, round_tf32: bool = True, memory_config=None):
    """``(w_pre [1,4,T,1], w_post [4,5,T,1])`` fp32 from ``h_pre [T,4]``, ``h_post [T,4]``, ``H [T,16]`` (one generic_op;
    ``post_halve`` scales h_post by 0.5 exactly; ``round_tf32`` rounds every weight to nearest-even TF32, bit-exactly
    ``tf32_rne``). Without rounding the values are those of the ttnn permute / concat glue (bitwise, column 0)."""
    T = int(h_pre.shape[-2])
    ht = int(h_pre.padded_shape[-2]) // TILE
    dev = h_pre.device()
    mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    w_pre = ttnn.allocate_tensor_on_device(ttnn.Shape([1, N_STREAMS, T, 1]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    w_post = ttnn.allocate_tensor_on_device(
        ttnn.Shape([N_STREAMS, N_STREAMS + 1, T, 1]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc
    )
    grid = dev.compute_with_storage_grid_size()
    n_cores = min(4 * ht, int(grid.x) * int(grid.y))  # work items = (token tile row, mixing row)
    cores = _first_cores(grid, n_cores)
    ct = [_LAYOUT_CB_IN, _LAYOUT_CB_OUT, 1 if post_halve else 0, 1 if round_tf32 else 0]
    for t in (h_pre, h_post, H, w_pre, w_post):
        ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    common = [
        h_pre.buffer_address(), h_post.buffer_address(), H.buffer_address(), w_pre.buffer_address(),
        w_post.buffer_address(), ht, n_cores, int(grid.y),
    ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=_LAYOUT_SRC,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=ct,
        common_runtime_args=common,
        config=ttnn.ReaderConfigDescriptor(),
    )
    cbs = [
        _cb(_LAYOUT_CB_IN, 3, cores, ttnn.float32, _F32_TILE),
        _cb(_LAYOUT_CB_OUT, 6, cores, ttnn.float32, _F32_TILE),
    ]
    desc = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
    _set_program_hash(_LAYOUT_HASH, (n_cores, tuple(ct), int(grid.x), int(grid.y)), desc)
    ttnn.generic_op([h_pre, h_post, H, w_pre, w_post], desc)
    return w_pre, w_post


# ---------------------------------------------------------------------------------------------------------------------
# mixes finalize kernel (ttnn.generic_op): p = (sum_b Y_b) * rsqrt(sum_b S_b[:, 0] + eps) per 32-token tile
# ---------------------------------------------------------------------------------------------------------------------
# Replaces (decode) ttnn.sum(Y, dim=1) + ttnn.sum(S, dim=1) + add/rsqrt + column-broadcast multiply (4 ops, ~28 us
# traced) for the split-K partials of one 32-token tile, and (prefill) the diagonal mask + 3 sums + the same 2 ops
# (~11 ms at S = 32768) over all token tiles. Exact fp32: UnpackToDestFp32 copies, SFPU adds ((Y0 + Y1) + Y2) + ...,
# SFPU rsqrt (approx off), SFPU column-broadcast multiply. Partial b of token tile t: Y page b * ysb + t * ysr, S page
# b * ssb + t * ssr (decode: Y / S [1, NB, 32, 32], pages b; prefill: Y [1, 1, 4S, 128] diagonal tiles
# (b * St + t) * 4 + b, S [1, 4, S, 32] pages b * St + t). Token tiles split over cores by core index.
_FIN_READER_SRC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t total = get_common_arg_val<uint32_t>(2);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(4);
    constexpr uint32_t cb_y = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t nb = get_compile_time_arg_val(2);
    constexpr uint32_t chunk = get_compile_time_arg_val(3);
    constexpr uint32_t ysb = get_compile_time_arg_val(4);
    constexpr uint32_t ysr = get_compile_time_arg_val(5);
    constexpr uint32_t ssb = get_compile_time_arg_val(6);
    constexpr uint32_t ssr = get_compile_time_arg_val(7);
    constexpr auto a_y = TensorAccessorArgs<8>();
    constexpr auto a_s = TensorAccessorArgs<a_y.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 32 * 32 * 4;
    const auto s_y = TensorAccessor(a_y, y_addr, TB);
    const auto s_s = TensorAccessor(a_s, s_addr, TB);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t q = total / num_cores, rem = total % num_cores;
    const uint32_t n = q + (core_i < rem ? 1 : 0);
    const uint32_t start = core_i * q + (core_i < rem ? core_i : rem);
    CircularBuffer cy(cb_y), cs(cb_s);
    // per token tile: all Y chunks, then all S chunks -- the order compute consumes them (interleaving the two streams
    // deadlocks once the S buffer is full while compute still waits for Y)
    for (uint32_t t = start; t < start + n; ++t) {
        for (uint32_t b = 0; b < nb; b += chunk) {
            cy.reserve_back(chunk);
            const uint32_t wy = cy.get_write_ptr();
            for (uint32_t i = 0; i < chunk; ++i) {
                noc_async_read(s_y.get_noc_addr((b + i) * ysb + t * ysr), wy + i * TB, TB);
            }
            noc_async_read_barrier();
            cy.push_back(chunk);
        }
        for (uint32_t b = 0; b < nb; b += chunk) {
            cs.reserve_back(chunk);
            const uint32_t ws = cs.get_write_ptr();
            for (uint32_t i = 0; i < chunk; ++i) {
                noc_async_read(s_s.get_noc_addr((b + i) * ssb + t * ssr), ws + i * TB, TB);
            }
            noc_async_read_barrier();
            cs.push_back(chunk);
        }
    }
}
"""

_FIN_COMPUTE_SRC = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    constexpr uint32_t cb_y = get_compile_time_arg_val(0);
    constexpr uint32_t cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t nb = get_compile_time_arg_val(3);
    constexpr uint32_t chunk = get_compile_time_arg_val(4);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(5);
    const uint32_t total = get_common_arg_val<uint32_t>(0);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(2);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t n = total / num_cores + (core_i < total % num_cores ? 1 : 0);
    compute_kernel_hw_startup(cb_y, cb_out);
    CircularBuffer cy(cb_y), cs(cb_s), co(cb_out);
    for (uint32_t t = 0; t < n; ++t) {
        tile_regs_acquire();
        if (t != 0) {
            reconfig_data_format_srca(cb_s, cb_y);
        }
        add_binary_tile_init();
        copy_init(cb_y);
        for (uint32_t b = 0; b < nb; b += chunk) {  // DST0 = sum_b Y_b
            cy.wait_front(chunk);
            for (uint32_t i = 0; i < chunk; ++i) {
                if (b + i == 0) {
                    copy_tile(cb_y, i, 0);
                } else {
                    copy_tile(cb_y, i, 1);
                    add_binary_tile(0, 1, 0);
                }
            }
            cy.pop_front(chunk);
        }
        reconfig_data_format_srca(cb_y, cb_s);
        copy_init(cb_s);
        for (uint32_t b = 0; b < nb; b += chunk) {  // DST2 = sum_b S_b (column 0)
            cs.wait_front(chunk);
            for (uint32_t i = 0; i < chunk; ++i) {
                if (b + i == 0) {
                    copy_tile(cb_s, i, 2);
                } else {
                    copy_tile(cb_s, i, 3);
                    add_binary_tile(2, 3, 2);
                }
            }
            cs.pop_front(chunk);
        }
        binop_with_scalar_tile_init();
        add_unary_tile(2, eps_bits);  // ss + 16384 * 1e-6
        rsqrt_tile_init();
        rsqrt_tile(2);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 2);  // p = p_un * r (column 0 of DST2 broadcast over the 32 columns)
        tile_regs_commit();
        co.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        co.push_back(1);
    }
}
"""

_FIN_WRITER_SRC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t total = get_common_arg_val<uint32_t>(1);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr auto a_o = TensorAccessorArgs<1>();
    constexpr uint32_t TB = 32 * 32 * 4;
    const auto s_o = TensorAccessor(a_o, out_addr, TB);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t q = total / num_cores, rem = total % num_cores;
    const uint32_t n = q + (core_i < rem ? 1 : 0);
    const uint32_t start = core_i * q + (core_i < rem ? core_i : rem);
    CircularBuffer co(cb_out);
    for (uint32_t t = start; t < start + n; ++t) {
        co.wait_front(1);
        noc_async_write(co.get_read_ptr(), s_o.get_noc_addr(t), TB);
        noc_async_write_barrier();
        co.pop_front(1);
    }
}
"""

_FIN_CB_Y, _FIN_CB_S, _FIN_CB_OUT = 0, 1, 16
_FIN_HASH: Dict[tuple, int] = {}


def _f32_bits(x: float) -> int:
    return int(torch.tensor([float(x)], dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF


def finalize_mixes(
    y: ttnn.Tensor,
    s: ttnn.Tensor,
    *,
    eps: float,
    T: int,
    nb: Optional[int] = None,
    n_tiles: int = 1,
    y_strides: Tuple[int, int] = (1, 0),
    s_strides: Tuple[int, int] = (1, 0),
    chunk: Optional[int] = None,
    memory_config=None,
    compute_role: Optional[ComputeRole] = None,
):
    """``p [1,1,T,32]`` fp32 = ``(sum_b y_b) * rsqrt(sum_b s_b + eps)`` per 32-token tile, in one generic_op.

    Partial ``b`` of token tile ``t`` is page ``b * y_strides[0] + t * y_strides[1]`` of ``y`` (projection partials)
    and page ``b * s_strides[0] + t * s_strides[1]`` of ``s`` (``sum(X^2)`` partials in column 0). Defaults = the decode
    split-K layout (``y``, ``s`` ``[1, NB, 32, 32]``, one token tile, ``nb = NB``); ``MHCSite.mixes`` passes the prefill
    layout (4 diagonal partials per token tile, ``n_tiles = S / 32`` split over the compute grid)."""
    nb = int(y.shape[1]) if nb is None else int(nb)
    chunk = min(8, nb) if chunk is None else int(chunk)
    n_tiles = int(n_tiles)
    if nb < 1 or nb % chunk:
        raise ValueError(f"nb {nb} must be a positive multiple of chunk {chunk}")
    if -(-int(T) // TILE) != n_tiles:
        raise ValueError(f"T = {T} does not fill n_tiles = {n_tiles} token tiles")
    dev = y.device()
    mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, int(T), TILE]), ttnn.float32, ttnn.TILE_LAYOUT, dev, mc)
    grid = dev.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    n_cores = min(n_tiles, gx * gy)
    cores = _first_cores(grid, n_cores)
    rd_ct = [_FIN_CB_Y, _FIN_CB_S, nb, chunk, int(y_strides[0]), int(y_strides[1]), int(s_strides[0]), int(s_strides[1])]
    rd_ct += list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
    rd_ct += list(ttnn.TensorAccessorArgs(s).get_compile_time_args())
    wr_ct = [_FIN_CB_OUT] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    cp_ct = [_FIN_CB_Y, _FIN_CB_S, _FIN_CB_OUT, nb, chunk, _f32_bits(eps)]
    split = [n_tiles, n_cores, gy]
    src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    kernels = [
        ttnn.KernelDescriptor(kernel_source=_FIN_READER_SRC, source_type=src, core_ranges=cores,
                              compile_time_args=rd_ct, common_runtime_args=[y.buffer_address(), s.buffer_address()] + split,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_FIN_WRITER_SRC, source_type=src, core_ranges=cores,
                              compile_time_args=wr_ct, common_runtime_args=[out.buffer_address()] + split,
                              config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_FIN_COMPUTE_SRC, source_type=src, core_ranges=cores,
                              compile_time_args=cp_ct, common_runtime_args=split,
                              config=_compute_descriptor(compute_role, (_FIN_CB_Y, _FIN_CB_S))),
    ]
    cbs = [
        _cb(_FIN_CB_Y, 2 * chunk, cores, ttnn.float32, _F32_TILE),
        _cb(_FIN_CB_S, 2 * chunk, cores, ttnn.float32, _F32_TILE),
        _cb(_FIN_CB_OUT, 2, cores, ttnn.float32, _F32_TILE),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    role = compute_role if compute_role is not None else COMPUTE_ROLES["mhc"]
    _set_program_hash(_FIN_HASH, (tuple(rd_ct), tuple(wr_ct), tuple(cp_ct), gx, gy, n_cores, role.fidelity), desc)
    ttnn.generic_op([y, s, out], desc)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# fused post mix (ttnn.generic_op): X' = attn_res_weighted_reduce_nc(concat([X, out], dim=1), w_post) without the concat
# ---------------------------------------------------------------------------------------------------------------------
# The compute kernel is the stock op's (ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_weighted_reduce_nc/
# device/kernels/weighted_reduce_nc.cpp: mul_tiles_bcast_cols with acc_to_dest, one DEST tile per output row r), with
# the work split taken from common runtime args (core index) instead of per-core args; num_groups = 1 (4 rows <= the 4
# fp32 DEST tiles of half-sync). The reader takes candidates 0..3 from X (pages i + c * inner) and candidate 4 from out
# (page i) instead of one concatenated tensor; the writer writes row r of position i to page r * inner + i of
# X' [1, 4, T, D] (the page layout of the op's [4, 1, T, D] output).
_POST_READER_SRC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t total = get_common_arg_val<uint32_t>(3);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(4);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(5);
    constexpr uint32_t inner = get_compile_time_arg_val(0);    // tile positions per stream (Ht * Wt)
    constexpr uint32_t Wt = get_compile_time_arg_val(1);       // tile columns per row
    constexpr uint32_t w_inner = get_compile_time_arg_val(2);  // weight tiles per (row, candidate) = Ht
    constexpr uint32_t num_x = get_compile_time_arg_val(3);    // stream candidates (4); candidate num_x = out
    constexpr uint32_t num_r = get_compile_time_arg_val(4);    // output rows (4)
    constexpr auto a_x = TensorAccessorArgs<5>();
    constexpr auto a_o = TensorAccessorArgs<a_x.next_compile_time_args_offset()>();
    constexpr auto a_w = TensorAccessorArgs<a_o.next_compile_time_args_offset()>();
    constexpr uint32_t num_c = num_x + 1;
    constexpr uint32_t cb_in0 = 0, cb_in1 = 1;
    constexpr uint32_t in_bytes = get_tile_size(cb_in0);
    constexpr uint32_t w_bytes = get_tile_size(cb_in1);
    const auto s_x = TensorAccessor(a_x, x_addr, in_bytes);
    const auto s_o = TensorAccessor(a_o, o_addr, in_bytes);
    const auto s_w = TensorAccessor(a_w, w_addr, w_bytes);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t q = total / num_cores, rem = total % num_cores;
    const uint32_t n = q + (core_i < rem ? 1 : 0);
    const uint32_t start = core_i * q + (core_i < rem ? core_i : rem);
    CircularBuffer c0(cb_in0), c1(cb_in1);
    for (uint32_t i = start; i < start + n; ++i) {
        // a weight set (num_r x num_c tiles, row-major = the compute's s * num_c + c) per token tile row, fetched on the
        // first position and at every row boundary (the compute turns it over on the same test)
        if (i == start || i % Wt == 0) {
            const uint32_t row = i / Wt;
            c1.reserve_back(num_r * num_c);
            uint32_t l1 = c1.get_write_ptr();
            for (uint32_t r = 0; r < num_r; ++r) {
                for (uint32_t c = 0; c < num_c; ++c) {
                    noc_async_read(s_w.get_noc_addr((r * num_c + c) * w_inner + row), l1, w_bytes);
                    l1 += w_bytes;
                }
            }
            noc_async_read_barrier();
            c1.push_back(num_r * num_c);
        }
        c0.reserve_back(num_c);
        uint32_t l1 = c0.get_write_ptr();
        for (uint32_t c = 0; c < num_x; ++c) {
            noc_async_read(s_x.get_noc_addr(i + c * inner), l1, in_bytes);
            l1 += in_bytes;
        }
        noc_async_read(s_o.get_noc_addr(i), l1, in_bytes);
        noc_async_read_barrier();
        c0.push_back(num_c);
    }
}
"""

_POST_COMPUTE_SRC = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

using namespace ckernel;

// The stock attn_res_weighted_reduce_nc compute (num_groups = 1): the MATH init with acc_to_dest = 1 turns every
// mul_tiles_bcast_cols into a MAC against its DEST tile; DEST is zeroed before kernel_main and by the packer on every
// tile_regs_release(), so the accumulators start at zero.
void kernel_main() {
    constexpr uint32_t num_c = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t num_r = get_compile_time_arg_val(2);
    const uint32_t total = get_common_arg_val<uint32_t>(0);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(1);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(2);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t q = total / num_cores, rem = total % num_cores;
    const uint32_t n = q + (core_i < rem ? 1 : 0);
    const uint32_t start = core_i * q + (core_i < rem ? core_i : rem);
    if (n == 0) {
        return;
    }
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    CircularBuffer c0(cb_in0), c1(cb_in1), co(cb_out0);
    compute_kernel_hw_startup(cb_in0, cb_in1, cb_out0);
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(cb_in0, cb_in1);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        cb_in0, cb_in1, 1 /*acc_to_dest*/)));
    reconfig_data_format(cb_in0, cb_in1);
    constexpr uint32_t wset = num_r * num_c;
    uint32_t width_index = start % Wt;
    c1.wait_front(wset);
    for (uint32_t i = 0; i < n; ++i) {
        if (i != 0 && width_index == 0) {
            c1.pop_front(wset);
            c1.wait_front(wset);
        }
        c0.wait_front(num_c);
        tile_regs_acquire();
        for (uint32_t c = 0; c < num_c; ++c) {
            for (uint32_t s = 0; s < num_r; ++s) {
                mul_tiles_bcast_cols(cb_in0, cb_in1, c, s * num_c + c, s);
            }
        }
        tile_regs_commit();
        c0.pop_front(num_c);
        co.reserve_back(num_r);
        pack_reconfig_data_format(cb_out0);
        tile_regs_wait();
        for (uint32_t s = 0; s < num_r; ++s) {
            pack_tile(s, cb_out0);
        }
        tile_regs_release();
        co.push_back(num_r);
        if (++width_index == Wt) {
            width_index = 0;
        }
    }
    c1.pop_front(wset);
}
"""

_POST_WRITER_SRC = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t total = get_common_arg_val<uint32_t>(1);
    const uint32_t num_cores = get_common_arg_val<uint32_t>(2);
    const uint32_t grid_y = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t inner = get_compile_time_arg_val(0);
    constexpr uint32_t num_r = get_compile_time_arg_val(1);
    constexpr auto a_y = TensorAccessorArgs<2>();
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t out_bytes = get_tile_size(cb_out);
    const auto s_y = TensorAccessor(a_y, y_addr, out_bytes);
    const uint32_t core_i = static_cast<uint32_t>(get_absolute_logical_x()) * grid_y + get_absolute_logical_y();
    const uint32_t q = total / num_cores, rem = total % num_cores;
    const uint32_t n = q + (core_i < rem ? 1 : 0);
    const uint32_t start = core_i * q + (core_i < rem ? core_i : rem);
    CircularBuffer co(cb_out);
    for (uint32_t i = start; i < start + n; ++i) {
        co.wait_front(num_r);
        uint32_t l1 = co.get_read_ptr();
        for (uint32_t r = 0; r < num_r; ++r) {
            noc_async_write(l1, s_y.get_noc_addr(r * inner + i), out_bytes);
            l1 += out_bytes;
        }
        noc_async_write_barrier();
        co.pop_front(num_r);
    }
}
"""

_POST_CB_IN, _POST_CB_W, _POST_CB_OUT = 0, 1, 16
_POST_HASH: Dict[tuple, int] = {}


def post_mix(X: ttnn.Tensor, out: ttnn.Tensor, w_post: ttnn.Tensor, *, memory_config=None,
             compute_role: Optional[ComputeRole] = None) -> ttnn.Tensor:
    """``X' [1, 4, T, D]`` bf16 = ``attn_res_weighted_reduce_nc(concat([X, out], dim=1), w_post, dim=1)`` (bitwise),
    without materializing the concat: ``X [1,4,T,D]`` and ``out [1,1,T,D]`` bf16 TILE interleaved are read in place;
    ``w_post [4,5,T,1]`` fp32 TILE (column 0 of each tile; see ``coefficient_layout``)."""
    xs, os_, ws = ([int(d) for d in t.shape] for t in (X, out, w_post))
    if len(xs) != 4 or xs[0] != 1 or xs[1] != N_STREAMS or os_ != [1, 1, xs[2], xs[3]]:
        raise ValueError(f"post_mix needs X [1, 4, T, D] and out [1, 1, T, D], got {xs} and {os_}")
    if ws != [N_STREAMS, N_STREAMS + 1, xs[2], 1]:
        raise ValueError(f"post_mix needs w_post [4, 5, T, 1], got {ws}")
    if X.dtype != ttnn.bfloat16 or out.dtype != ttnn.bfloat16 or w_post.dtype != ttnn.float32:
        raise ValueError(f"post_mix needs bf16 X / out and fp32 w_post, got {X.dtype} {out.dtype} {w_post.dtype}")
    Tp = int(X.padded_shape[2])
    if int(out.padded_shape[2]) != Tp or int(w_post.padded_shape[2]) != Tp:
        raise ValueError("X, out and w_post must have the same padded token count")
    D = xs[3]
    Ht, Wt = Tp // TILE, D // TILE
    inner = Ht * Wt
    dev = X.device()
    mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, N_STREAMS, xs[2], D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mc)
    grid = dev.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    n_cores = min(inner, gx * gy)
    cores = _first_cores(grid, n_cores)
    rd_ct = [inner, Wt, Ht, N_STREAMS, N_STREAMS]
    for t in (X, out, w_post):
        rd_ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    cp_ct = [N_STREAMS + 1, Wt, N_STREAMS]
    wr_ct = [inner, N_STREAMS] + list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
    split = [inner, n_cores, gy]
    src = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    kernels = [
        ttnn.KernelDescriptor(kernel_source=_POST_READER_SRC, source_type=src, core_ranges=cores,
                              compile_time_args=rd_ct,
                              common_runtime_args=[X.buffer_address(), out.buffer_address(), w_post.buffer_address()]
                              + split,
                              config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_POST_WRITER_SRC, source_type=src, core_ranges=cores,
                              compile_time_args=wr_ct, common_runtime_args=[y.buffer_address()] + split,
                              config=ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=_POST_COMPUTE_SRC, source_type=src, core_ranges=cores,
                              compile_time_args=cp_ct, common_runtime_args=split,
                              config=_compute_descriptor(compute_role)),
    ]
    cbs = [  # the stock factory's CBs: candidates double-buffered, one weight set, 4 outputs double-buffered
        _cb(_POST_CB_IN, 2 * (N_STREAMS + 1), cores, ttnn.bfloat16, _BF16_TILE),
        _cb(_POST_CB_W, N_STREAMS * (N_STREAMS + 1), cores, ttnn.float32, _F32_TILE),
        _cb(_POST_CB_OUT, 2 * N_STREAMS, cores, ttnn.bfloat16, _BF16_TILE),
    ]
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    role = compute_role if compute_role is not None else COMPUTE_ROLES["mhc"]
    _set_program_hash(_POST_HASH, (tuple(rd_ct), tuple(wr_ct), tuple(cp_ct), gx, gy, n_cores, role.fidelity), desc)
    ttnn.generic_op([X, out, w_post, y], desc)
    return y


# ---------------------------------------------------------------------------------------------------------------------
# Sinkhorn kernel module (owned separately)
# ---------------------------------------------------------------------------------------------------------------------
_KM_IMPORT_ERROR: Optional[ImportError] = None


def motif_kernel_module():
    """The Motif Sinkhorn kernel module (``tt/kernels/sinkhorn_motif.py``), or ``None`` if it cannot be imported (the
    ImportError is kept in ``_KM_IMPORT_ERROR``). Only ImportError is caught: a broken module (SyntaxError, a ttnn API
    error at import, ...) raises here."""
    global _KM_IMPORT_ERROR
    try:
        from models.demos.motif3.tt.kernels import sinkhorn_motif  # lazy: owned by the kernel module

        return sinkhorn_motif
    except ImportError as e:
        _KM_IMPORT_ERROR = e
        return None


@dataclass
class MHCCoeffs:
    """Per-call mHC coefficients of one site. ``w_pre [1,4,T,1]`` and ``w_post [4,5,T,1]`` fp32 (the stream-mix
    weights; TF32-rounded for ``mix="wr"``); ``p`` (raw mixes ``[1,1,T,32]`` fp32, columns 0..23), ``h_pre [T,4]``,
    ``h_post [T,4]``, ``H [T,16]`` (unrounded fp32) are kept only with ``keep=True`` (tests / debugging)."""

    w_pre: Optional[ttnn.Tensor]
    w_post: Optional[ttnn.Tensor]
    T: int
    p: Optional[ttnn.Tensor] = None
    h_pre: Optional[ttnn.Tensor] = None
    h_post: Optional[ttnn.Tensor] = None
    H: Optional[ttnn.Tensor] = None
    # fused decode (MOTIF3_MHC_DECODE=fused): the packed coefficient tile (tt/kernels/mhc_decode.py) instead of
    # w_pre / w_post
    packed: Optional[ttnn.Tensor] = None

    def deallocate(self) -> None:
        _free(self.w_pre, self.w_post, self.p, self.h_pre, self.h_post, self.H, self.packed)


# =====================================================================================================================
# the site
# =====================================================================================================================
class MHCSite:
    """One mHC site (layer ``layer_idx``, ``site`` in {"mhc_attn", "mhc_ffn"}); see the module docstring.

    Args:
        source: ``HFWeightLoader`` / ``DictWeightSource`` (HF names ``model.layers.{l}.{site}.{proj_pre,proj_post,
            proj_res}.weight``, ``rms_norm.weight``, ``alpha_{pre,post,res}``, ``bias_{pre,post,res}``; the reference's
            ``proj_merged.weight`` is accepted too). Kept (lazily read) for ``scalars()``.
        cache: TT weight cache (``False`` for random weights). Names ``{site}.proj_rows_x128`` ``[1,1,128,4096]`` bf16
            and ``{site}.motif_consts`` ``[64,32]`` fp32 (or, stock: ``{site}.{alpha,bias,lo,hi}_row`` ``[1,1,1,32]``
            fp32), per layer.
        sinkhorn: ``"motif"`` (default; ``None`` means the same) | ``"stock"`` (explicit; misses MHC-5 on real data).
            There is no fallback: "motif" raises if the kernel module cannot be imported.
        mix: ``"wr"`` (FPU weighted reduce with TF32-rounded weights, default) | ``"composite"`` (fp32 multiply + sum on
            the unrounded weights; exact debug path).
        post_concat: ``mix="wr"`` post on the ``[X | out]`` concat with the stock op (verification; default: the
            concat-free ``post_mix`` generic_op, bitwise equal).
        decode_split: K-chunks of the decode projection (must divide 512; 32 = 16 tiles per core; 16 measured equal,
            8 and fewer slower).
        glue: ``"kernel"`` (``coefficient_layout``, default) | ``"ttnn"`` (permutes + concat; unrounded weights).
        mix_l1: decode stream-mix weights (and the ``post_concat`` concat) in L1 (default; DRAM in prefill always).
        l1_intermediates: decode projection / statistics / coefficient intermediates in L1 (default).
        finalize: ``"kernel"`` (``finalize_mixes``, default) | ``"ttnn"`` (accurate ttnn sums + rsqrt + multiply).
        decode: ``"ops"`` | ``"fused"`` (``None`` = ``cfg.mhc_decode``, ``MOTIF3_MHC_DECODE``; Phase C D3): the decode
            site (one 32-row tile) as 5 programs -- projection, statistics, ONE fused coefficients program writing a
            packed coefficient tile, and the two stream mixes expanding it locally (``tt/kernels/mhc_decode.py``),
            bitwise equal to "ops". Applies with the production options only (``sinkhorn="motif"``, ``mix="wr"``,
            ``glue="kernel"``, ``finalize="kernel"``, no ``post_concat``) and not to ``coefficients(keep=True)``; any
            other combination runs the op path.

    Per-site device memory: 1 MB of bf16 projection weights + 8 KB constants.
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        layer_idx: int,
        site: str,
        *,
        source,
        cache: bool = True,
        sinkhorn: Optional[str] = None,
        mix: str = "wr",
        post_concat: bool = False,
        decode_split: int = DECODE_SPLIT_K,
        glue: str = "kernel",
        mix_l1: bool = True,
        l1_intermediates: bool = True,
        finalize: str = "kernel",
        decode: Optional[str] = None,
    ):
        if site not in SITES:
            raise ValueError(f"site must be one of {SITES}, got {site!r}")
        if cfg.n_streams != N_STREAMS or cfg.sinkhorn_iters != 20:
            raise ValueError(f"MHCSite supports 4 streams / 20 Sinkhorn iterations, got {cfg.n_streams}/{cfg.sinkhorn_iters}")
        if mix not in ("wr", "composite"):
            raise ValueError(f"mix must be 'wr' or 'composite', got {mix!r}")
        if glue not in ("kernel", "ttnn"):
            raise ValueError(f"glue must be 'kernel' or 'ttnn', got {glue!r}")
        if finalize not in ("kernel", "ttnn"):
            raise ValueError(f"finalize must be 'kernel' or 'ttnn', got {finalize!r}")
        decode = (getattr(cfg, "mhc_decode", "ops") if decode is None else decode).strip().lower()
        if decode not in MHC_DECODE_MODES:
            raise ValueError(f"decode must be one of {MHC_DECODE_MODES}, got {decode!r}")
        if abs(float(cfg.mhc_h_post_coeff) - 1.0) > 0:
            raise ValueError(f"h_post coefficient {cfg.mhc_h_post_coeff} != 1.0 is not supported")
        if cfg.dtypes.mhc_scalars != ttnn.float32:
            raise ValueError(f"the mHC constants must be fp32 (cfg.dtypes.mhc_scalars = {cfg.dtypes.mhc_scalars})")
        # ---- Sinkhorn backend (no silent fallback; checked before any device allocation) ---------------------------
        sinkhorn = "motif" if sinkhorn is None else sinkhorn
        if sinkhorn not in SINKHORN_IMPLS:
            raise ValueError(f"sinkhorn must be one of {SINKHORN_IMPLS}, got {sinkhorn!r}")
        km = None
        if sinkhorn == "motif":
            km = motif_kernel_module()
            if km is None:
                raise RuntimeError(
                    "sinkhorn='motif' (the default) needs models.demos.motif3.tt.kernels.sinkhorn_motif, which failed to "
                    f"import ({_KM_IMPORT_ERROR!r}); sinkhorn='stock' runs the draft-1 path explicitly (it misses the "
                    "MHC-5 accuracy bound on real data)"
                ) from _KM_IMPORT_ERROR
        self.sinkhorn = sinkhorn
        self._km = km
        self.decode = decode
        # the fused decode site needs the production options (module docstring of tt/kernels/mhc_decode.py)
        self.decode_fused = (decode == "fused" and sinkhorn == "motif" and mix == "wr" and glue == "kernel"
                             and finalize == "kernel" and not post_concat)
        self._md = None
        self.glue = glue
        self.finalize = finalize
        self.mix = mix
        self.post_concat = bool(post_concat)
        # stream-mix weights (and the post_concat [X | out]) in L1 interleaved: the weighted reduce re-reads the whole
        # weight set on every core
        self.mix_mc = ttnn.L1_MEMORY_CONFIG if mix_l1 else ttnn.DRAM_MEMORY_CONFIG
        # decode: the small projection / statistics / coefficient intermediates (<= 128 KB each) in L1 interleaved
        self.dec_mc = ttnn.L1_MEMORY_CONFIG if l1_intermediates else ttnn.DRAM_MEMORY_CONFIG
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.site = site
        self.prefix = W.hf_name(layer_idx, site)
        self.hidden = int(cfg.hidden_size)
        self.width = N_STREAMS * self.hidden  # 16384
        self.iters = int(cfg.sinkhorn_iters)
        self.ss_eps = float(cfg.mhc_rms_eps) * self.width  # 16384 * 1e-6 (the 1/16384 is folded into the weights)
        if decode_split < 1 or (self.width // TILE) % decode_split:
            raise ValueError(f"decode_split {decode_split} must divide the {self.width // TILE} K tiles")
        self.decode_split = int(decode_split)
        self.role = cfg.compute_role("mhc")
        self.ckc = cfg.compute_config("mhc")
        grid = mesh_device.compute_with_storage_grid_size()
        self.grid = (int(grid.x), int(grid.y))

        # ---- weights ----------------------------------------------------------------------------------------------
        name = (lambda n: f"{site}.{n}") if cache else (lambda n: None)
        self._scalars = None
        self._source = source
        self.proj = _replicated(
            lambda: projection_tensor(source, self.prefix, hidden=self.hidden), mesh_device, cfg.dtypes.mhc, cfg=cfg,
            cache_name=name(PROJ_CACHE_NAME), layer=layer_idx,
        )  # [1, 1, 128, 4096] bf16, x128-scaled stream rows
        kp = self.width // self.decode_split
        self.proj_split = ttnn.experimental.view(self.proj, [1, self.decode_split, TILE, kp])  # zero copy
        if self.proj_split.buffer_address() != self.proj.buffer_address():
            raise RuntimeError("ttnn.experimental.view of the mHC weights is no longer zero-copy (see tt/mhc.py)")
        self.pc_decode = _bmm_pc(self.grid, kp // TILE)  # NB = 32 blocks on 32 cores, the whole K = 16 tiles each

        # ---- Sinkhorn constants -----------------------------------------------------------------------------------
        sdt = cfg.dtypes.mhc_scalars
        if sinkhorn == "motif":
            self.motif_consts = _replicated(
                lambda: km.consts_from_scalars(self.scalars()), mesh_device, sdt, cfg=cfg,
                cache_name=name("motif_consts"), layer=layer_idx,
            )  # [64, 32] fp32
        else:
            vec = None

            def v(key):
                nonlocal vec
                if vec is None:
                    vec = logit_vectors(self.scalars())
                return vec[key]

            self.alpha_row = _replicated(lambda: v("alpha"), mesh_device, sdt, cfg=cfg, cache_name=name("alpha_row"),
                                         layer=layer_idx)
            self.bias_row = _replicated(lambda: v("bias"), mesh_device, sdt, cfg=cfg, cache_name=name("bias_row"),
                                        layer=layer_idx)
            self.lo_row = _replicated(lambda: v("lo"), mesh_device, sdt, cfg=cfg, cache_name=name("lo_row"),
                                      layer=layer_idx)
            self.hi_row = _replicated(lambda: v("hi"), mesh_device, sdt, cfg=cfg, cache_name=name("hi_row"),
                                      layer=layer_idx)
            _shared_preclamped_consts(mesh_device)  # create the per-mesh constant now (not during a trace capture)

    @property
    def stock_consts(self) -> ttnn.Tensor:
        """The shared identity-SEL / zero-base consts of the stock op (looked up per call: ``release_shared`` may
        replace them)."""
        return _shared_preclamped_consts(self.mesh_device)

    # ------------------------------------------------------------------------------------------------------------
    def scalars(self) -> Dict[str, torch.Tensor]:
        """fp32 alphas / biases of the site (``weights.mhc_scalars``), read once."""
        if self._scalars is None:
            self._scalars = W.mhc_scalars(self._source, self.prefix)
        return self._scalars

    # ------------------------------------------------------------------------------------------------------------
    # step 1: raw mixes p [1, 1, T, 32] fp32
    # ------------------------------------------------------------------------------------------------------------
    def mixes(self, X: ttnn.Tensor) -> ttnn.Tensor:
        """``p`` = RMSNorm(X_flat) @ (W_pre||W_post||W_res)^T as ``[1, 1, T, 32]`` fp32 (columns 0..23; 24..31 zero)."""
        shape = [int(d) for d in X.shape]
        if len(shape) != 4 or shape[0] != 1 or shape[1] != N_STREAMS or shape[3] != self.hidden:
            raise ValueError(f"X must be [1, {N_STREAMS}, T, {self.hidden}], got {shape}")
        T = shape[2]
        Tp = int(X.padded_shape[2])
        if Tp == TILE:  # decode: split-K partials [1, NB, 32, 32]
            mc = self.dec_mc
            y, s = self._partials_decode(X)
            if self.finalize == "kernel":
                p = finalize_mixes(y, s, eps=self.ss_eps, T=T, memory_config=mc, compute_role=self.role)
                _free(y, s)
                return p
            p_un = ttnn.sum(y, dim=1, keepdim=True, memory_config=mc, compute_kernel_config=self.ckc)  # exact fp32
            ss = ttnn.sum(s, dim=1, keepdim=True, memory_config=mc, compute_kernel_config=self.ckc)
            _free(y, s)
        else:  # prefill: projection [1, 1, 4S, 128] (diagonal blocks = per-stream partials), sum(X^2) [1, 4, S, 32]
            mc = ttnn.DRAM_MEMORY_CONFIG
            St = Tp // TILE
            y, s = self._partials_prefill(X)
            if self.finalize == "kernel":
                p = finalize_mixes(
                    y, s, eps=self.ss_eps, T=T, nb=N_STREAMS, n_tiles=St, y_strides=(N_STREAMS * St + 1, N_STREAMS),
                    s_strides=(St, 1), chunk=N_STREAMS, memory_config=mc, compute_role=self.role,
                )
                _free(y, s)
                return p
            # op form: diagonal blocks by an exact 0/1 mask -> exact sums over the streams and the 4 column tiles
            ym = ttnn.multiply(ttnn.reshape(y, [1, N_STREAMS, Tp, N_STREAMS * TILE]), self._diag_mask())
            _free(y)
            y1 = ttnn.sum(ym, dim=1, keepdim=True, compute_kernel_config=self.ckc)  # [1, 1, S, 128]
            _free(ym)
            p4 = ttnn.sum(ttnn.experimental.view(y1, [St, N_STREAMS, TILE, TILE]), dim=1, keepdim=True,
                          compute_kernel_config=self.ckc)  # [S/32, 1, 32, 32]
            _free(y1)
            p_un = ttnn.reshape(p4, [1, 1, Tp, TILE])
            ss = ttnn.sum(s, dim=1, keepdim=True, compute_kernel_config=self.ckc)
            _free(s)
        # 128 * rsqrt(ms + eps) with the x128 in the weights; rsqrt fused into the add (one op)
        r = ttnn.add(ss, self.ss_eps, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT)], memory_config=mc)
        _free(ss)
        r_col = _view_logical(r, [1, 1, Tp, 1])  # column 0 = the per-token scale
        p = ttnn.multiply(p_un, r_col, memory_config=mc)
        _free(p_un, r)
        return _view_logical(p, [1, 1, T, TILE]) if T != Tp else p

    def _partials_decode(self, X):
        """Split-K partials ``[1, NB, 32, 32]`` of the projection and of sum(X^2) (column 0)."""
        X32 = _view_logical(X, [1, N_STREAMS, TILE, self.hidden])
        xv = ttnn.experimental.view(X32, [1, self.decode_split, TILE, self.width // self.decode_split])
        y = ttnn.matmul(
            xv, self.proj_split, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.ckc,
            program_config=self.pc_decode, memory_config=self.dec_mc,
        )
        s = ttnn.rms_norm_pre_all_gather(xv, dtype=ttnn.float32, compute_kernel_config=self.ckc,
                                         memory_config=self.dec_mc)
        return y, s

    def _partials_prefill(self, X):
        """``(y [1, 1, 4S, 128], s [1, 4, S, 32])`` for T > 32: ``X [1,4,S,4096]`` viewed as ``[1,1,4S,4096]`` times
        ``proj^T [4096, 128]`` (one non-batched matmul; the mcast configs reload K-blocked partials exactly), whose
        diagonal blocks (stream s rows x stream s columns) are the per-stream partials, and the per-stream sum(X^2)
        (column 0). 4x the projection FLOPs, at prefill only."""
        S = int(X.padded_shape[2])
        Xp = _view_logical(X, [1, N_STREAMS, S, self.hidden])  # logical T -> the padded rows (metadata only)
        y = ttnn.matmul(
            ttnn.reshape(Xp, [1, 1, N_STREAMS * S, self.hidden]), self.proj, transpose_b=True, dtype=ttnn.float32,
            compute_kernel_config=self.ckc,
        )
        s = ttnn.rms_norm_pre_all_gather(Xp, dtype=ttnn.float32, compute_kernel_config=self.ckc)
        return y, s

    def _diag_mask(self) -> ttnn.Tensor:
        return _shared_tensor(self.mesh_device, "diag_mask", stream_diag_mask, ttnn.float32)

    # ------------------------------------------------------------------------------------------------------------
    # step 2: coefficients
    # ------------------------------------------------------------------------------------------------------------
    def maps(self, p: ttnn.Tensor, *, halve_post: bool = True, memory_config=None):
        """``(h_pre [T,4], h_post [T,4], H [T,16])`` fp32 from raw mixes ``p [1,1,T,32]`` (Motif semantics). For the
        stock op with ``halve_post=False`` the returned post is the op's ``2 sigma`` (the layout kernel halves it)."""
        if self.sinkhorn == "motif":
            return self._km.motif_sinkhorn(p, self.motif_consts, iters=self.iters, memory_config=memory_config)
        L = ttnn.multiply(p, self.alpha_row)  # exact fp32 (SFPU), multiply then add like torch
        L2 = ttnn.add(L, self.bias_row)
        _free(L)
        L3 = ttnn.maximum(L2, self.lo_row)  # per-column clamps: -10 / -20, then +10 / +20
        _free(L2)
        L4 = ttnn.minimum(L3, self.hi_row)
        _free(L3)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(L4, self.stock_consts, N_STREAMS, self.iters, 0.0)
        _free(L4)
        if not halve_post:
            return pre, post, comb
        h_post = ttnn.multiply(post, 0.5)  # the stock op's post is 2 sigma (exact power-of-two scale)
        _free(post)
        return pre, h_post, comb

    def _mhc_decode(self):
        if self._md is None:
            from .kernels import mhc_decode  # lazy: the kernel module

            self._md = mhc_decode
        return self._md

    def coefficients(self, X: ttnn.Tensor, *, keep: bool = False) -> MHCCoeffs:
        """All per-call coefficients of the site for streams ``X [1,4,T,4096]``."""
        T = int(X.shape[2])
        if self.decode_fused and not keep and int(X.padded_shape[2]) == TILE:
            shape = [int(d) for d in X.shape]
            if len(shape) != 4 or shape[0] != 1 or shape[1] != N_STREAMS or shape[3] != self.hidden:
                raise ValueError(f"X must be [1, {N_STREAMS}, T, {self.hidden}], got {shape}")
            y, s = self._partials_decode(X)
            md = self._mhc_decode()
            P = md.coefficients_packed(y, s, self.motif_consts, eps=self.ss_eps, T=T, iters=self.iters,
                                       memory_config=self.mix_mc, fidelity=self.role.fidelity,
                                       approx=self.role.approx)
            _free(y, s)
            return MHCCoeffs(w_pre=None, w_post=None, T=T, packed=P)
        p = self.mixes(X)
        dec = int(X.padded_shape[2]) == TILE
        mmc = self._mix_mc(X)
        h_pre, h_post, H = self.maps(
            p, halve_post=self.glue != "kernel", memory_config=self.dec_mc if dec else ttnn.DRAM_MEMORY_CONFIG
        )
        if self.glue == "kernel":
            w_pre, w_post = coefficient_layout(
                h_pre, h_post, H, post_halve=self.sinkhorn == "stock", round_tf32=self.mix == "wr", memory_config=mmc
            )
        else:  # ttnn ops: 3 fp32-exact permutes + 1 concat (unrounded)
            w_pre = ttnn.permute(ttnn.reshape(h_pre, [1, 1, T, N_STREAMS]), (0, 3, 2, 1), memory_config=mmc)
            w_res = ttnn.reshape(
                ttnn.permute(ttnn.reshape(H, [1, 1, T, N_STREAMS * N_STREAMS]), (0, 3, 2, 1)),
                [N_STREAMS, N_STREAMS, T, 1],
            )  # [4, 4, T, 1]: w_res[r, c, t] = H_t[r][c]
            w_hp = ttnn.reshape(
                ttnn.permute(ttnn.reshape(h_post, [1, 1, T, N_STREAMS]), (0, 3, 2, 1)), [N_STREAMS, 1, T, 1]
            )
            w_post = ttnn.concat([w_res, w_hp], dim=1, memory_config=mmc)  # [4, 5, T, 1]
            _free(w_res, w_hp)
        if keep:
            if self.sinkhorn == "stock" and self.glue == "kernel":  # keep the final h_post (the op's post is 2 sigma)
                hq = ttnn.multiply(h_post, 0.5)
                _free(h_post)
                h_post = hq
            return MHCCoeffs(w_pre=w_pre, w_post=w_post, T=T, p=p, h_pre=h_pre, h_post=h_post, H=H)
        _free(p, h_pre, h_post, H)
        return MHCCoeffs(w_pre=w_pre, w_post=w_post, T=T)

    # ------------------------------------------------------------------------------------------------------------
    # step 3: stream mixing
    # ------------------------------------------------------------------------------------------------------------
    def apply_pre(self, X: ttnn.Tensor, coeffs: MHCCoeffs) -> ttnn.Tensor:
        """``x_red = sum_i h_pre_i X_i`` -> ``[1, 1, T, 4096]`` bf16."""
        if coeffs.packed is not None:
            return self._mhc_decode().mix_packed(X, coeffs.packed, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                 fidelity=self.role.fidelity, approx=self.role.approx)
        if self.mix == "composite":
            prod = ttnn.multiply(X, coeffs.w_pre, dtype=ttnn.float32)
            red = ttnn.sum(prod, dim=1, keepdim=True, compute_kernel_config=self.ckc)
            _free(prod)
            out = ttnn.typecast(red, ttnn.bfloat16)
            _free(red)
            return out
        return ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(
            X, coeffs.w_pre, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.ckc
        )

    def apply_post(self, X: ttnn.Tensor, out: ttnn.Tensor, coeffs: MHCCoeffs) -> ttnn.Tensor:
        """``X' = H @ X + h_post (x) out`` -> ``[1, 4, T, 4096]`` bf16 (one rounding)."""
        T = int(X.shape[2])
        if coeffs.packed is not None:
            return self._mhc_decode().mix_packed(X, coeffs.packed, out, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                 fidelity=self.role.fidelity, approx=self.role.approx)
        if self.mix == "wr" and not self.post_concat:
            return post_mix(X, out, coeffs.w_post, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_role=self.role)
        xc = ttnn.concat([X, out], dim=1, memory_config=self._mix_mc(X))  # [1, 5, T, 4096]
        if self.mix == "composite":
            parts = []
            for r in range(N_STREAMS):
                wr = ttnn.slice(coeffs.w_post, [r, 0, 0, 0], [r + 1, N_STREAMS + 1, T, 1])  # [1, 5, T, 1]
                prod = ttnn.multiply(xc, wr, dtype=ttnn.float32)
                parts.append(ttnn.sum(prod, dim=1, keepdim=True, compute_kernel_config=self.ckc))
                _free(wr, prod)
            red = ttnn.concat(parts, dim=1)
            _free(*parts)
            y = ttnn.typecast(red, ttnn.bfloat16)
            _free(red, xc)
            return y
        y = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(
            xc, coeffs.w_post, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.ckc
        )  # [4, 1, T, 4096]
        _free(xc)
        return ttnn.reshape(y, [1, N_STREAMS, T, self.hidden])

    # ------------------------------------------------------------------------------------------------------------
    # decoder-facing entry points
    # ------------------------------------------------------------------------------------------------------------
    def pre(self, X: ttnn.Tensor, *, keep: bool = False) -> Tuple[ttnn.Tensor, MHCCoeffs]:
        """``(x_red [1,1,T,4096] bf16, coeffs)``: the sublayer input and the coefficients ``post`` needs."""
        coeffs = self.coefficients(X, keep=keep)
        return self.apply_pre(X, coeffs), coeffs

    def post(self, X: ttnn.Tensor, out: ttnn.Tensor, coeffs: MHCCoeffs, *, release: bool = True) -> ttnn.Tensor:
        """``X' [1,4,T,4096]`` bf16 from the site input ``X`` and the sublayer output ``out [1,1,T,4096]``. With
        ``release`` (default) the coefficients are deallocated afterwards."""
        y = self.apply_post(X, out, coeffs)
        if release:
            coeffs.deallocate()
        return y

    def _mix_mc(self, X):
        """Memory config of the stream-mix weights (and of the ``post_concat`` [X | out]): L1 for decode (when enabled),
        DRAM for prefill (the concat is 10-1340 MB there)."""
        return self.mix_mc if int(X.padded_shape[2]) == TILE else ttnn.DRAM_MEMORY_CONFIG

    def release(self) -> None:
        """Deallocate the site's device weights / constants (the shared per-mesh constants stay; ``release_shared``)."""
        _free(self.proj, getattr(self, "motif_consts", None), getattr(self, "alpha_row", None),
              getattr(self, "bias_row", None), getattr(self, "lo_row", None), getattr(self, "hi_row", None))


__all__ = [
    "DECODE_SPLIT_K",
    "MHCCoeffs",
    "MHCSite",
    "N_MIXES",
    "N_STREAMS",
    "PROJ_CACHE_NAME",
    "build_sinkhorn_consts",
    "coefficient_layout",
    "finalize_mixes",
    "folded_projection",
    "logit_vectors",
    "motif_kernel_module",
    "post_mix",
    "preclamped_consts",
    "projection_blocks",
    "projection_rows",
    "projection_tensor",
    "release_shared",
    "stream_diag_mask",
    "tf32_rne",
]
