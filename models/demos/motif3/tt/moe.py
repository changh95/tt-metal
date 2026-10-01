# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 routed MoE (layers 2-52) on the BH Galaxy: fp32 composite router + EP32 dense-local experts.

Implements design §2.3.7 (MoE), §3.2 (decode dataflow per MoE layer), §3.3 (prefill) and WAVE_A_REVIEW §5.6
MOE-1..7 with the op configs the device gates validated (``tests/unit/gates/GATES_RESULTS.md`` G4 / G5 / G6). The
shared expert is **not** here: the decoder (or the shared-expert module) either adds its output to this module's output
or hands its TP partial in through ``add_partial`` so that one ``all_reduce(tp)`` closes both (design §2.3.7 decode
steps 10-11; README §8 CCL schedule).

Semantics (HF ``modeling_motif.py:843-870, 929-1008``; reference ``Router`` / ``RoutedExperts``)::

    s   = sigmoid(x @ W_router^T)                    fp32 logits, fp32 sigmoid
    idx = topk(s + expert_bias, 8)                   bias for selection only
    w   = s[idx] / (sum(s[idx]) + 1e-20) * 2.0       unbiased scores, renormalized, route_scale 2.0
    out = sum_k w_k E_idx_k(x),   E_e(x) = (0.5 PolyNorm_e(g) u) @ W_down,e,   [g | u] = x @ W_gate_up,e

Router (:class:`MotifRouter`; MOE-2, G5): bf16 router weights (never bfp8), **fp32 logits** (``router`` role: HiFi4,
fp32 dest acc; decode shape on a 12-core 1D-multicast program config whose ``fused_activation`` is the sigmoid --
other shapes run the sigmoid as a separate op), ``+ expert_bias`` (fp32 ``[1,1,1,384]``), ``ttnn.topk(k=8)`` on FLOAT32
(decode: the biased scores padded with -inf to width 1024 so topk takes its multi-core path; prefill: width 384),
``ttnn.gather`` of the unbiased scores, ``* 1 / (sum + 1e-20)`` with the sum on the accurate fp32 SFPU reduce.
``test_moe_device_router_variants`` asserts that each of these fusions, and the module default, gives the same top-8 sets
and weights (|dw| <= 1e-5; measured 0) as the plain G5 composite on all 2971 real layer-2 tokens, both at prefill shape
(one 2976-row call) and in 93 decode-shape calls (where the padded top-k and the decode program config apply). Every
chip routes identical inputs with identical programs, so routes are bitwise identical on all 32 chips (design §2.3.7
invariant; asserted in the tests).
Precision (G5, WAVE_A_REVIEW D1): the FPU forms the logit dot products with TF32-class partial sums; on real router
inputs of 11 layers (32,681 token-layers) the top-8 set agrees with an fp64 router on **99.81 %** of tokens, every flip a
near-tie (8th/9th gap <= 1.9e-4). ``router_logits="exact_fp32"`` swaps in the exact SFPU kernel
(``kernels/router_fp32.py``, another agent's) for decode-shape calls.

Decode (:meth:`MotifMoE.forward_decode`; EP32 gather path, GPT-OSS BH pattern; per chip, the 8 lanes of its DP row):

1. ``f_all = ccl.ag_dp_rows(x)``: ``[1,1,8,4096]`` -> ``[1,1,32,4096]`` TILE, lane order ``8 dp + l`` (MOE-1, INFRA-5),
   identical on all 32 chips.
2. Router on all 32 tokens -> ``idx [1,1,32,8]`` uint32, ``w [1,1,32,8]`` fp32 (identical on every chip).
3. ``w_loc [1,12,32,1]`` fp32 = ``sum_k w[t,k] (idx[t,k] == e)`` for this chip's 12 experts (``local_expert_ids`` in
   fp32; exactly 0 for experts a token did not select; MOE-3).
4. Experts (MOE-4, G6): ``repeat(f_all) [1,12,32,4096] @ W_gate_up [1,12,4096,2560]`` (bfp8, ``experts_gate_up_pc``,
   ``experts`` role HiFi4 + fp32 acc, fp32 output) -> grouped PolyNorm from ``tt/polynorm.py`` (exact fp32 moments +
   Horner ``mac``, fp32 intermediates) with the routing weights folded into ``up`` (``h = poly(g) * (w_loc * u)``,
   rounded to bf16 once; never block float) -> ``@ W_down [1,12,1280,4096]`` (bfp8, ``experts_down_pc``; the PolyNorm
   x0.5 and route_scale x2.0 folded into ``W_down``: net x1.0, exact) -> ``y [1,12,32,4096]`` = weighted expert outputs.
5. Combine (MOE-5): ``fast_reduce_nc(y, dims=[1])`` (bf16 terms summed in an fp32 dest, packed in ``combine_dtype``:
   bf16 by default; ``combine_dtype=fp32`` packs the fp32 dest into a preallocated fp32 output, so the per-chip partial
   is a true fp32 sum) -> ``[1,1,32,4096]`` -> ``ccl.ar_dp`` (sum over the 4 chips of the column) ->
   ``ccl.partition(2, "dp")`` (this row's 8 lanes) -> ``+ add_partial`` -> ``ccl.ar_tp`` (sum over the 8 columns) ->
   ``[1,1,8,4096]`` bf16 DRAM: the full routed MoE output, replicated in the row.
   All decode intermediates live in L1 (freed inside the call; ~220 us faster than DRAM); CCL payloads and the output
   are DRAM.

Measured on this Galaxy (TORUS_Y fabric, 4x8, real layer-2/4 weights; ``tests/unit/test_moe.py``): decode **1036 /
1041 us traced** per call at L2 / L4 (eager 4.6 ms, dispatch bound). Per stage at L2 (each traced alone; small stages
measured with enough calls to clear the ~150 us sync floor): gather 26, router 118, local mask 19, repeat 6, gate_up 429,
PolyNorm 126, down 220, expert sum 7, AR(dp) 32, partition 13, AR(tp) 34 us (sum 1030 us); 200.5 MB of expert weights
per chip at 309 GB/s. Variants: PolyNorm bf16 1018-1023 us, fp32 partials 1072 us, DRAM intermediates 1260 us.
Accuracy vs the fp32 reference (reference modules, bf16-valued weights): PCC 0.99996 (L2 / L4, bfp8 experts,
rel 0.9 %), every token >= 0.9999 against the reference experts on the device's routes; bf16 experts: 0.999994 (real L2,
``fold_route_scale=False``), 0.999993 (random); traced output == eager bitwise.

Prefill (:meth:`MotifMoE.forward_prefill`; MOE-6, design §3.3, "masked dense"): tokens replicated on all 32 chips; every
chip routes all S tokens (composite router, DRAM) and runs its 12 experts on all of them in chunks of ``prefill_chunk``
rows (unselected experts get weight 0; bf16 PolyNorm intermediates, ``rms`` impl), then ``ccl.rs_dp`` ->
``[1,1,S/4,4096]`` -> ``+ add_partial`` -> ``ccl.ar_tp`` -> ``ccl.ag_dp`` -> ``[1,1,S,4096]`` replicated on all chips. For
chunks of >= 2048 rows the batched expert matmuls use an in1-multicast program config (:func:`prefill_experts_pc`, 2x
the op's auto config; the prefill output is bitwise equal to the auto-config one at S = 2048 / 4096). Eager, per layer:
4-8 ms at S = 128 (dispatch bound), 24 ms at 2048, 44 ms at 4096, 354 ms at 32768. PCC 0.9999 vs the reference at every
S, every position >= 0.99986 against the reference experts on the device's routes; the positions below 0.999 against the
reference's own routes are all near-tie route flips (5 / 7 / 66 of 2048 / 4096 / 32768). 48x the useful expert FLOPs;
the v1 replacement is a compacting dispatch / sparse experts.

Weights (``tt/weights.py``): ``router_weights`` (``[4096, 384]`` bf16 + fp32 bias), ``experts_gate_up`` /
``experts_down`` + ``ep_layout`` with ``as_tensor(dp_dim=0, tp_dim=1)`` (chip ``k = 8 dp + tp`` holds experts
``[12k, 12k+12)``), ``local_expert_ids``; PolyNorm constants ``polynorm.GroupedPolyNormConsts``. Cache names
``moe.router.weight``, ``moe.router.expert_bias``, ``moe.experts.gate_up``, ``moe.experts.down_x1`` (W_down with the
x0.5 x 2.0 = x1.0 fold; ``moe.experts.down`` when ``fold_route_scale=False``), ``moe.experts.polynorm.{c0,c1,c2,b,D,E}``,
``moe.local_expert_ids``.

Trace safety: :meth:`MotifMoE.forward_decode` has fixed shapes, per-layer-constant slices only, no host round trips and
no data-dependent control flow; it allocates and frees its intermediates in a fixed order (all constants -- router
weights, the topk pad, PolyNorm constants, local expert ids -- are created in the constructor; the only in-call
allocation besides op outputs is the host-side ``allocate_tensor_on_device`` of the fp32 partial when
``combine_dtype=fp32``). Rows (tokens) are independent: an inactive lane's input never changes another lane's output
(tested bitwise).

Ownership: neither forward ever frees the caller's ``x`` / ``add_partial`` (or test taps). On a mesh with a size-1 axis
(e.g. (1, 8), which ``MeshAxes.detect`` / ``validate`` accept) that axis' collectives hand their input back, so every
free goes through :class:`_Keep` (free ``old`` only if the op returned a new tensor); tested with stand-in tensors on
the host and on a (1, 8) submesh (``test_moe_device_submesh_1x8``).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import ttnn

from . import polynorm as _pn
from . import weights as W
from .ccl import MotifCCL
from .model_config import TILE, MotifTTConfig, mcast1d_matmul_pc

POLYNORM_MODES = ("fp32", "bf16")
POLYNORM_IMPLS = ("horner", "rms", "local")  # tt/polynorm.py impls + this file's G6 copy
COMBINE_MODES = ("fold", "multiply_sum")


def _free(*ts) -> None:
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


class _Keep:
    """Tensors a forward must never free: the caller's inputs (``x``, ``add_partial``) and test taps.

    ``drop(old, new)`` frees ``old`` after an op produced ``new`` from it, unless ``old`` is protected or the op handed
    it back unchanged: on a size-1 mesh axis every :class:`MotifCCL` collective (and ``ag_dp_rows``) returns its input
    (``tt/ccl.py``), so an unconditional free would free the caller's input or the result (meshes with a size-1 axis,
    e.g. (1, 8), which ``MeshAxes.detect`` and ``validate`` accept)."""

    def __init__(self, *ts):
        self.ids = {id(t) for t in ts if t is not None}

    def drop(self, old, new=None) -> None:
        if old is None or old is new or id(old) in self.ids:
            return
        ttnn.deallocate(old)


# Local program-config helper (README §10 rule 1 wants ``cfg.*_pc()`` builders): requested shared change -- move it
# into model_config as ``experts_prefill_pc(m_rows, n_out)`` (like tt/mlp.py's ``decode_matmul_pc``).
def prefill_experts_pc(m_tiles: int, n_tiles: int, *, grid=(8, 8), in0_block_w: int = 8, out_block_w: int = 16):
    """Batched prefill expert matmul ``[1, 12, M, K] @ [1, 12, K, N]`` with in1 **multicast** (1D, ``mcast_in0=False``,
    ``fuse_batch=False``): each of the ``grid`` cores owns ``per_core_M = M / ncores`` tile rows of every expert's
    output, so each weight block is read from DRAM once and multicast (the op's auto config re-reads the weights per
    core: 35.6 / 18.3 ms for gate_up / down at M = 4096 vs 17.8 / 10.1 ms with this, bitwise identical results;
    ``test_moe_device_prefill_matmul_variants`` asserts the identity at the module's configs, and
    ``test_moe_device_prefill`` asserts it end to end). None when M does not split over the grid (small chunks:
    auto)."""
    ncores = int(grid[0]) * int(grid[1])
    if m_tiles % ncores or n_tiles % out_block_w:
        return None
    pm = m_tiles // ncores
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(int(grid[0]), int(grid[1])),
        in0_block_w=int(in0_block_w),
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=pm,
        out_block_w=int(out_block_w),
        per_core_M=pm,
        per_core_N=int(n_tiles),
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=False,
    )


def _dtype_of(name_or_dtype, default=ttnn.bfloat16):
    """``"fp32"``/``"float32"`` | ``"bf16"``/``"bfloat16"`` | a ttnn dtype -> ttnn dtype; ``None`` -> ``default`` (the
    argument's documented default, never a silent fp32)."""
    if name_or_dtype is None:
        return default
    if isinstance(name_or_dtype, str):
        key = name_or_dtype.lower()
        if key in ("fp32", "float32"):
            return ttnn.float32
        if key in ("bf16", "bfloat16"):
            return ttnn.bfloat16
        raise ValueError(f"unknown dtype name {name_or_dtype!r} (use 'fp32' / 'bf16' or a ttnn dtype)")
    return name_or_dtype


def grouped_polynorm(gu, consts: Dict[str, object], *, inter: int, mode: str = "fp32", eps: float = 1e-6,
                     compute_kernel_config=None, memory_config=None, out_dtype=ttnn.bfloat16, row_scale=None):
    """Composite grouped PolyNorm * up on ``gu [1, E, M, 2 * inter]`` (gate = ``[..., :inter]``, up = ``[inter:]``)
    -> ``[1, E, M, inter]`` in ``out_dtype`` (bf16; never block float, study 01 N3).

    ``h = (c0 N(g^3) + c1 N(g^2) + c2 N(g) + b) * u`` with ``N(z) = z / sqrt(mean(z^2) + eps)`` over the intermediate
    dim (weightless ``ttnn.rms_norm``); ``consts`` holds ``c0, c1, c2, b`` as ``[1, E, 1, 1]`` tensors in the math dtype
    of ``mode`` ("fp32": fp32 intermediates, the design's decode default, G6 PCC 0.9999990; "bf16": bf16 intermediates
    with fp32 accumulation inside the norms, G6 PCC 0.999991). The x0.5 output scale is folded into ``W_down``.
    ``row_scale`` (``[1, E, M, 1]`` fp32, e.g. the routing weights ``w_loc``) multiplies every row in fp32 before the
    single rounding to ``out_dtype`` (``h * w`` = the routed combine folded into the down projection's input).
    Local copy of the G6 composite (tt/polynorm.py did not exist yet); consumes nothing."""
    if mode not in POLYNORM_MODES:
        raise ValueError(f"polynorm mode must be one of {POLYNORM_MODES}, got {mode!r}")
    mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    B, E, M, N = (int(s) for s in gu.shape)
    if N != 2 * inter:
        raise ValueError(f"gate_up width {N} != 2 x {inter}")
    g = ttnn.slice(gu, [0, 0, 0, 0], [B, E, M, inter], memory_config=mc)
    u = ttnn.slice(gu, [0, 0, 0, inter], [B, E, M, 2 * inter], memory_config=mc)
    math_dtype = ttnn.float32 if mode == "fp32" else ttnn.bfloat16
    if g.dtype != math_dtype:
        g32 = ttnn.typecast(g, math_dtype, memory_config=mc)
        u32 = ttnn.typecast(u, math_dtype, memory_config=mc)
        _free(g, u)
        g, u = g32, u32
    g2 = ttnn.multiply(g, g, memory_config=mc)
    g3 = ttnn.multiply(g2, g, memory_config=mc)
    n1 = ttnn.rms_norm(g, epsilon=eps, compute_kernel_config=compute_kernel_config, memory_config=mc)
    _free(g)
    n2 = ttnn.rms_norm(g2, epsilon=eps, compute_kernel_config=compute_kernel_config, memory_config=mc)
    _free(g2)
    n3 = ttnn.rms_norm(g3, epsilon=eps, compute_kernel_config=compute_kernel_config, memory_config=mc)
    _free(g3)
    t = ttnn.multiply(n3, consts["c0"], memory_config=mc)
    _free(n3)
    a = ttnn.multiply(n2, consts["c1"], memory_config=mc)
    _free(n2)
    t2 = ttnn.add(t, a, memory_config=mc)
    _free(t, a)
    a = ttnn.multiply(n1, consts["c2"], memory_config=mc)
    _free(n1)
    t = ttnn.add(t2, a, memory_config=mc)
    _free(t2, a)
    t2 = ttnn.add(t, consts["b"], memory_config=mc)
    _free(t)
    if row_scale is not None:
        us = ttnn.multiply(row_scale, u, memory_config=mc)  # A = row_scale (fp32): fp32 product, broadcast over cols
        _free(u)
        u = us
    h = ttnn.multiply(t2, u, dtype=out_dtype, memory_config=mc)
    _free(t2, u)
    return h


class MotifRouter:
    """fp32 composite router of one MoE layer (MOE-2, G5), replicated on every chip.

    ``route_logits(f)`` -> fp32 ``[1, 1, M, 384]``; ``__call__(f)`` -> ``(idx [1, 1, M, 8] uint32, w [1, 1, M, 8] fp32)``.
    Weights: ``weights.router_weights`` -> ``W^T [4096, 384]`` in ``cfg.dtypes.router`` (bf16; bfp8 would flip 3.7 %
    of the sets, verify_numerics (e)) and ``expert_bias [1, 1, 1, 384]`` fp32 (selection only).

    Op-fusion knobs (``test_moe_device_router_variants`` compares each one, and the module default, with the plain G5
    composite -- linear, separate ``ttnn.sigmoid``, add, topk on width 384, gather, sum, add, reciprocal, multiply -- on
    all 2971 real layer-2 tokens, both at prefill shape (one 2976-row call) and at decode shape (93 calls of 32 rows);
    the module default must give identical top-8 sets and weights within 1e-5):

    * ``sigmoid_in_pc`` (on): at decode shape the sigmoid runs in the router linear's program config
      (``fused_activation``: SFPU sigmoid on the fp32 dest before the pack; one op fewer, bitwise-identical scores;
      the traced gain is within the slope method's noise: router 126.4 vs ~128 us, MoE call +-8 us);
    * ``fuse_sigmoid`` (on): other shapes use ``ttnn.linear(activation="sigmoid")``. ttnn dispatches ``activation=``
      as a separate ``unary_chain`` op whenever no ``core_grid`` is given (``ttnn/.../matmul/matmul.cpp``), so this is
      the plain composite under another name (same op count, identical results);
    * ``fuse_normalize`` (on): ``w * 1 / (sum + 1e-20)`` is one multiply with B activations ``[+1e-20, recip]``;
    * ``topk_pad_to`` (1024): at decode shape (32 rows) the biased scores are padded with -inf to a power-of-two width,
      so ``ttnn.topk`` takes its multi-core path (concat + topk ~46 us instead of ~64 us); prefill keeps width 384 (the
      multi-core path would need >= 8192 there);
    * ``router_decode_pc`` (on): decode-shape linear on a 12-core 1D-multicast program config
      (``model_config.mcast1d_matmul_pc((12, 1), 12, 32)``): ~14 us instead of ~52 us with the auto config. Built here
      from the shared G6 builder; requested shared change: a ``cfg.router_decode_pc()`` builder in model_config;
    * ``topk_sorted``: unsorted top-8 (no gain measured; the downstream math is order-free).

    Measured (traced, decode shape, DRAM intermediates, ``test_moe_device_router_variants``): plain composite 189 us,
    module default 126 us (118 us inside the MoE decode with L1 intermediates).
    ``logits_fn``: optional decode-shape logits replacement (``kernels.router_fp32.RouterLogitsFP32``, exact fp32).
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        layer_idx: int,
        *,
        source,
        cache: bool = True,
        topk_sorted: bool = True,
        fuse_sigmoid: bool = True,
        fuse_normalize: bool = True,
        topk_pad_to: Optional[int] = 1024,
        router_decode_pc: bool = True,
        sigmoid_in_pc: bool = True,
        logits_fn=None,
    ):
        self.mesh_device = mesh_device
        # Optional replacement of the decode-shape (M = 32) logits, e.g. the exact-fp32 SFPU kernel
        # ``kernels.router_fp32.RouterLogitsFP32`` (same contract as :meth:`route_logits`); prefill shapes keep the
        # composite linear (the kernel handles exactly 32 tokens).
        self.logits_fn = logits_fn
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.top_k = cfg.top_k
        self.n_experts = cfg.num_experts
        self.route_scale = float(cfg.route_scale)
        self.topk_sorted = bool(topk_sorted)
        self.fuse_sigmoid = bool(fuse_sigmoid)
        self.sigmoid_in_pc = bool(sigmoid_in_pc)
        self.fuse_normalize = bool(fuse_normalize)
        self.topk_pad_to = topk_pad_to
        # Decode-shape (M = 32) router linear: 1D multicast over 12 cores (per_core_N 1 = the 12 output tiles),
        # in0_block_w 32 -> ~14 us traced instead of ~52 us with the auto config. Built with the shared G6 builder
        # model_config.mcast1d_matmul_pc (requested shared change: cfg.router_decode_pc()); None = the op's auto
        # config. decode_pc_sigmoid: the same config with the sigmoid as its fused_activation (sigmoid_in_pc).
        self.decode_pc = None
        self.decode_pc_sigmoid = None
        gx, gy = cfg.compute_grid
        if router_decode_pc and gx >= 12:
            self.decode_pc = self._decode_pc()
            self.decode_pc_sigmoid = self._decode_pc(sigmoid=True)
        self._decode_pc_base = self.decode_pc  # decode_pc_sigmoid is this config + the sigmoid
        l = self.layer_idx

        def router():
            return W.router_weights(
                source.get(W.hf_name(l, "moe.router.gate.weight")), source.get(W.hf_name(l, "moe.expert_bias"))
            )

        def as_t(src, dtype, name):
            return W.as_tensor(
                src, mesh_device=mesh_device, cfg=cfg, dtype=dtype, cache_name=name if cache else None, layer=l
            )

        self.weight = as_t(lambda: router()[0], cfg.dtypes.router, "moe.router.weight")
        self.bias = as_t(lambda: router()[1].reshape(1, 1, 1, -1), cfg.dtypes.router_bias, "moe.router.expert_bias")
        self.ckc = cfg.compute_config("router")
        self.ckc_eltwise = cfg.compute_config("eltwise")
        self.dram = ttnn.DRAM_MEMORY_CONFIG
        self._pads: Dict[int, object] = {}
        if topk_pad_to:
            if topk_pad_to & (topk_pad_to - 1) or topk_pad_to <= self.n_experts:
                raise ValueError(f"topk_pad_to must be a power of two > {self.n_experts}, got {topk_pad_to}")
            self._pads[TILE] = self._make_pad(TILE)  # decode: one tile row of tokens

    def _make_pad(self, rows: int):
        import torch

        return ttnn.from_torch(
            torch.full((1, 1, rows, self.topk_pad_to - self.n_experts), float("-inf")),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.dram,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _decode_pc(self, sigmoid: bool = False):
        pc = mcast1d_matmul_pc((12, 1), self.n_experts // TILE, 32, self.cfg.hidden_size // TILE)
        if sigmoid:  # SFPU sigmoid (accurate: sigmoid_tile<RC, false>) on the fp32 dest, before the pack
            pc.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
        return pc

    def _pc(self, f):
        return self.decode_pc if int(f.shape[-2]) == TILE else None

    def _use_logits_fn(self, f) -> bool:
        return self.logits_fn is not None and int(f.shape[-2]) == TILE

    def route_logits(self, f, *, memory_config=None):
        """``f [1, 1, M, 4096]`` bf16 -> fp32 logits ``[1, 1, M, 384]`` (bf16 x bf16, HiFi4, fp32 dest acc, fp32 out),
        or ``logits_fn(f)`` (the exact-fp32 kernel) for decode-shape inputs when one is set."""
        if self._use_logits_fn(f):
            return self.logits_fn(f, memory_config=memory_config or self.dram)
        return ttnn.linear(
            f, self.weight, dtype=ttnn.float32, compute_kernel_config=self.ckc, program_config=self._pc(f),
            memory_config=memory_config or self.dram,
        )

    def __call__(
        self, f, *, scale: Optional[float] = None, taps: Optional[dict] = None, memory_config=None
    ) -> Tuple[object, object]:
        """Router on ``f [1, 1, M, 4096]`` bf16 (M a multiple of 32) -> ``(idx [1, 1, M, 8] uint32, w [1, 1, M, 8]
        fp32)`` with ``w = s[idx] / (sum + 1e-20) * scale`` (``scale`` defaults to ``route_scale`` = 2.0). Intermediates
        and outputs in ``memory_config`` (DRAM default; decode passes L1). ``taps`` (tests, eager) receives ``scores``
        and ``biased``."""
        mc = memory_config or self.dram
        scale = self.route_scale if scale is None else float(scale)
        M = int(f.shape[-2])
        exact = self._use_logits_fn(f)
        pc = self._pc(f)
        if not exact and self.sigmoid_in_pc and pc is not None and pc is self._decode_pc_base:
            # decode shape: sigmoid fused into the linear's program config (no separate op); a replaced decode_pc
            # (diagnostics) falls through to the separate sigmoid with that config
            scores = ttnn.linear(
                f, self.weight, dtype=ttnn.float32, compute_kernel_config=self.ckc,
                program_config=self.decode_pc_sigmoid, memory_config=mc,
            )
        elif self.fuse_sigmoid and not exact:
            # ttnn runs ``activation=`` as a separate unary_chain op here (no core_grid): same as the plain composite
            scores = ttnn.linear(
                f, self.weight, dtype=ttnn.float32, compute_kernel_config=self.ckc, activation="sigmoid",
                program_config=pc, memory_config=mc,
            )
        else:
            logits = self.route_logits(f, memory_config=mc)
            scores = ttnn.sigmoid(logits, memory_config=mc)
            _free(logits)
        biased = ttnn.add(scores, self.bias, memory_config=mc)
        pad = self._pads.get(M)
        if pad is not None:  # power-of-two width -> multi-core topk (pads are -inf: never selected)
            wide = ttnn.concat([biased, pad], dim=-1, memory_config=mc)
            vals, idx = ttnn.topk(wide, k=self.top_k, dim=-1, largest=True, sorted=self.topk_sorted, memory_config=mc)
            _free(wide)
        else:
            vals, idx = ttnn.topk(biased, k=self.top_k, dim=-1, largest=True, sorted=self.topk_sorted, memory_config=mc)
        _free(vals)
        if taps is not None:
            taps["scores"], taps["biased"] = scores, biased
        else:
            _free(biased)
        w = ttnn.gather(scores, -1, idx, memory_config=mc)
        if taps is None:
            _free(scores)
        den = ttnn.sum(w, dim=-1, keepdim=True, compute_kernel_config=self.ckc_eltwise, memory_config=mc)
        if self.fuse_normalize:
            wn = ttnn.multiply(
                w,
                den,
                input_tensor_b_activations=[
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, 1e-20),
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.RECIP),
                ],
                memory_config=mc,
            )
            _free(w, den)
        else:
            den2 = ttnn.add(den, 1e-20, memory_config=mc)
            _free(den)
            inv = ttnn.reciprocal(den2, memory_config=mc)
            _free(den2)
            wn = ttnn.multiply(w, inv, memory_config=mc)
            _free(w, inv)
        if scale != 1.0:
            ws = ttnn.multiply(wn, scale, memory_config=mc)
            _free(wn)
            wn = ws
        return idx, wn

    def deallocate(self) -> None:
        _free(self.weight, self.bias, *self._pads.values())
        self._pads = {}
        if self.logits_fn is not None and hasattr(self.logits_fn, "deallocate"):
            self.logits_fn.deallocate()
        self.logits_fn = None


class MotifMoE:
    """Routed MoE of one layer (layers 2-52): fp32 composite router + EP32 dense-local experts + combine.

    Args:
        mesh_device: the opened (4, 8) (or (8, 4)) mesh; meshes with a size-1 axis work too (``(1, 8)``: 48 experts
            per chip, tested on a submesh).
        cfg: :class:`MotifTTConfig` (axes, dtypes, compute roles, program configs, experts per chip).
        layer_idx: decoder layer (must be a MoE layer, ``cfg.layer(l).is_moe``).
        source: ``weights.HFWeightLoader`` or ``weights.DictWeightSource`` (HF names).
        ccl: :class:`MotifCCL` of the mesh.
        cache: use the TT weight cache (False for random weights: no files written).
        experts_dtype: routed-expert weight dtype (default ``cfg.dtypes.routed_experts`` = bfp8; ``ttnn.bfloat16`` gives
            the fp32-faithful accuracy check of the tests).
        decode_polynorm / prefill_polynorm: "fp32" | "bf16" PolyNorm intermediates (design §1.5: fp32 in decode,
            bf16 in prefill; G6: both pass).
        combine_dtype: dtype of this chip's routed partial and of the cross-chip partial sums (``ar_dp`` / ``rs_dp`` /
            ``ar_tp`` payloads): bf16 (default, G4: 33 us per AR) or fp32. With fp32 the expert sum packs its fp32
            dest straight into an fp32 tensor (``fast_reduce_nc`` into a preallocated output -- its Python binding has
            no ``output_dtype``, and the preallocated output's dtype sets the pack format), so the per-chip partial is
            never rounded to bf16; the CCL additions are then TF32-class (tt/ccl.py) and the module output is cast to
            bf16 once at the end. ``None`` = bf16. Measured (real L2 / L4 decode): PCC 0.999963 / 0.999960 vs 0.999961
            / 0.999957 with bf16, 1072 vs 1036-1041 us traced. The gain is small because a chip's partial row usually
            holds no or one expert term (a token picks 8 of 384 experts; ~2.6 % of (chip, token) rows have >= 2 local
            experts), so it is exact in bf16 already except for those rows and the cross-chip sums.
        gate_up_dtype: dtype of the gate_up matmul output (default: fp32 when the PolyNorm mode is fp32, so no
            typecast and no bf16 rounding of g / u; bf16 otherwise).
        down_dtype: dtype of the weighted expert outputs ``y [1, 12, M, 4096]`` (the down matmul output; bf16 default,
            ``None`` = bf16); the expert sum accumulates them in an fp32 dest either way.
        combine_mode: "fold" (default): the routing weights ``w_loc`` multiply the PolyNorm output in fp32 before its
            one bf16 rounding, the down matmul then yields the weighted expert outputs and
            ``ttnn.experimental.fast_reduce_nc(dims=[1])`` (fp32 dest accumulation) sums the 12 experts
            (the fork's routing-weight-into-GEMM2 order); "multiply_sum": ``ttnn.multiply(w_loc, y)`` in fp32 +
            ``ttnn.sum(dim=1)`` on the accurate fp32 path (the literal HF order; ~200 us slower per call).
        fold_route_scale: fold ``route_scale`` (2.0) into ``W_down`` together with the PolyNorm x0.5 (exact powers of
            two, net x1.0; saves one op per call); the experts path then uses the unscaled normalized weights.
        decode_l1: decode intermediates in L1 interleaved (default; DRAM otherwise). Prefill always uses DRAM.
        polynorm_impl / prefill_polynorm_impl: "horner" (decode default) / "rms" (prefill default):
            ``tt/polynorm.py``'s grouped PolyNorm (the MLP module's: exact fp32 moments + ``ttnn.mac`` Horner, or the G6
            rms_norm composite with fused ``mac``; decode fp32 L1 127 / 120 us, prefill bf16 at 4096 rows 14.6 / 11.2
            ms); "local": this file's G6 copy (:func:`grouped_polynorm`, the fallback). The routing weights are folded
            into ``up`` either way.
        router_logits: "composite" (default: ``ttnn.linear`` HiFi4 fp32-out, TF32-class partial sums, ~99.8 % top-8
            set agreement on real inputs) | "exact_fp32" (decode-shape logits from ``kernels.router_fp32`` -- true fp32
            accumulation on the SFPU, ~45-50 us instead of ~14 us; prefill keeps the composite).
        prefill_pc: in1-multicast program configs for the prefill expert matmuls when the chunk has >= 2048 rows
            (:func:`prefill_experts_pc`; 2x faster than the auto config, bitwise identical); False = auto config.
        prefill_chunk: rows per masked-dense prefill chunk (default ``cfg.moe_prefill_chunk`` = 4096).
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        layer_idx: int,
        *,
        source,
        ccl: MotifCCL,
        cache: bool = True,
        experts_dtype=None,
        decode_polynorm: str = "fp32",
        prefill_polynorm: str = "bf16",
        combine_dtype=ttnn.bfloat16,
        gate_up_dtype=None,
        down_dtype=ttnn.bfloat16,
        combine_mode: str = "fold",
        fold_route_scale: bool = True,
        decode_l1: bool = True,
        polynorm_impl: str = "horner",
        prefill_polynorm_impl: str = "rms",
        router_logits: str = "composite",
        prefill_pc: bool = True,
        prefill_chunk: Optional[int] = None,
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.spec = cfg.layer(self.layer_idx)
        if not self.spec.is_moe:
            raise ValueError(f"layer {layer_idx} is not a MoE layer ({self.spec.kind})")
        self.ccl = ccl
        for m in (decode_polynorm, prefill_polynorm):
            if m not in POLYNORM_MODES:
                raise ValueError(f"polynorm mode must be one of {POLYNORM_MODES}, got {m!r}")
        self.decode_polynorm = decode_polynorm
        self.prefill_polynorm = prefill_polynorm
        self.combine_dtype = _dtype_of(combine_dtype, ttnn.bfloat16)
        self.gate_up_dtype = None if gate_up_dtype is None else _dtype_of(gate_up_dtype)  # None: follow the PolyNorm mode
        self.down_dtype = _dtype_of(down_dtype, ttnn.bfloat16)
        self.prefill_chunk = int(prefill_chunk or cfg.moe_prefill_chunk)
        if self.prefill_chunk % TILE:
            raise ValueError(f"prefill_chunk {self.prefill_chunk} must be a multiple of {TILE}")
        if combine_mode not in COMBINE_MODES:
            raise ValueError(f"combine_mode must be one of {COMBINE_MODES}, got {combine_mode!r}")
        self.combine_mode = combine_mode
        for impl in (polynorm_impl, prefill_polynorm_impl):
            if impl not in POLYNORM_IMPLS:
                raise ValueError(f"polynorm impl must be one of {POLYNORM_IMPLS}, got {impl!r}")
        self.polynorm_impl = polynorm_impl
        self.prefill_polynorm_impl = prefill_polynorm_impl
        self.prefill_pc = bool(prefill_pc)
        # decode intermediates (<= 4 MB per tensor, all freed inside the call) live in L1: the composite PolyNorm
        # drops from 286 to 135 us traced (fp32); the CCL payloads and the module output stay in DRAM
        self.decode_mc = ttnn.L1_MEMORY_CONFIG if decode_l1 else ttnn.DRAM_MEMORY_CONFIG

        self.hidden = cfg.hidden_size
        self.inter = cfg.moe_intermediate_size
        self.n_experts = cfg.num_experts
        self.e_loc = cfg.experts_per_chip
        self.top_k = cfg.top_k
        self.route_scale = float(cfg.route_scale)
        self.experts_dtype = experts_dtype if experts_dtype is not None else cfg.dtypes.routed_experts
        # Scale of the routing weights the experts path multiplies with; route_scale is folded into W_down when
        # possible ((0.5 h) (2 w) = h w exactly).
        self.fold_route_scale = bool(fold_route_scale)
        self.internal_route_scale = 1.0 if self.fold_route_scale else self.route_scale
        down_scale = cfg.polynorm_output_scale * (self.route_scale if self.fold_route_scale else 1.0)

        l = self.layer_idx
        hf = lambda suffix: W.hf_name(l, suffix)  # noqa: E731
        cname = (lambda n: n) if cache else (lambda n: None)

        def as_t(src, dtype, name, **kw):
            return W.as_tensor(src, mesh_device=mesh_device, cfg=cfg, dtype=dtype, cache_name=cname(name), layer=l, **kw)

        # ---- router (replicated): W^T [4096, 384] bf16, expert_bias [1, 1, 1, 384] fp32 ----------------------
        if router_logits not in ("composite", "exact_fp32"):
            raise ValueError(f"router_logits must be 'composite' or 'exact_fp32', got {router_logits!r}")
        logits_fn = None
        if router_logits == "exact_fp32":
            from .kernels.router_fp32 import RouterLogitsFP32  # lazy: generic_op kernel, decode shape only

            logits_fn = RouterLogitsFP32.from_source(mesh_device, cfg, l, source=source, cache=cache)
        self.router_logits = router_logits
        self.router = MotifRouter(mesh_device, cfg, l, source=source, cache=cache, logits_fn=logits_fn)

        # ---- routed experts (EP32: chip (dp, tp) holds [1, 12, ...] = experts [12k, 12k + 12), k = 8 dp + tp) --------
        ep = dict(dp_dim=0, tp_dim=1)
        self.w_gate_up = as_t(
            lambda: W.ep_layout(W.experts_gate_up(source.get(hf("moe.experts.gate_up_proj"))), cfg),
            self.experts_dtype,
            "moe.experts.gate_up",
            **ep,
        )
        # x0.5 PolyNorm output scale (and route_scale when folded) folded into W_down; the cache name records the scale
        self.w_down = as_t(
            lambda: W.ep_layout(W.experts_down(source.get(hf("moe.experts.down_proj")), down_scale), cfg),
            self.experts_dtype,
            "moe.experts.down" if down_scale == cfg.polynorm_output_scale else f"moe.experts.down_x{down_scale:g}",
            **ep,
        )

        # grouped PolyNorm constants (tt/polynorm.py; cache names moe.experts.polynorm.{c0,c1,c2,b,D,E})
        self.pn_consts = _pn.GroupedPolyNormConsts.from_source(mesh_device, cfg, l, source=source, cache=cache)
        self.pn_fp32 = self.pn_consts.as_dict("fp32")  # {c0, c1, c2, b} [1, 12, 1, 1] (local impl, diagnostics)
        self.pn_bf16 = self.pn_consts.as_dict("bf16")
        self.local_ids = as_t(lambda: W.local_expert_ids(cfg), ttnn.float32, "moe.local_expert_ids", **ep)

        # ---- configs (never ad hoc: roles and gate program configs from model_config) ------------------------------
        self.ckc_experts = cfg.compute_config("experts")
        self.ckc_polynorm = cfg.compute_config("polynorm")
        self.ckc_eltwise = cfg.compute_config("eltwise")
        self.pc_gate_up = cfg.experts_gate_up_pc()  # decode, M = 32 (one tile row)
        self.pc_down = cfg.experts_down_pc()
        self.dram = ttnn.DRAM_MEMORY_CONFIG

    # ==========================================================================================================
    # router (MOE-2) and local routing weights (MOE-3)
    # ==========================================================================================================
    def route(self, f, *, scale: Optional[float] = None, taps: Optional[dict] = None) -> Tuple[object, object]:
        """See :meth:`MotifRouter.__call__`."""
        return self.router(f, scale=scale, taps=taps)

    def local_weights(self, idx, w, *, memory_config=None):
        """``(idx [1,1,M,8] uint32, w [1,1,M,8] fp32)`` -> this chip's ``w_loc [1, 12, M, 1]`` fp32 (MOE-3):
        ``w_loc[e, t] = sum_k w[t, k] * (idx[t, k] == local_id[e])`` (exactly 0 for unselected experts; the sum has one
        nonzero term and runs on the accurate fp32 path)."""
        mc = memory_config or self.dram
        idx_f = ttnn.typecast(idx, ttnn.float32, memory_config=mc)
        hit = ttnn.eq(idx_f, self.local_ids, memory_config=mc)  # [1, 12, M, 8] (1.0 / 0.0, fp32)
        _free(idx_f)
        sel = ttnn.multiply(hit, w, memory_config=mc)
        _free(hit)
        w_loc = ttnn.sum(sel, dim=-1, keepdim=True, compute_kernel_config=self.ckc_eltwise, memory_config=mc)
        _free(sel)
        return w_loc

    # ==========================================================================================================
    # experts (MOE-4) and combine (MOE-5)
    # ==========================================================================================================
    def experts(self, f, *, polynorm: str, decode: bool, row_scale=None, memory_config=None):
        """``f [1, 1, M, 4096]`` bf16 (identical on all chips) -> ``y [1, 12, M, 4096]`` = this chip's 12 experts
        applied to every token (``row_scale [1, 12, M, 1]``: routing weights folded into the PolyNorm output, so ``y``
        is already weighted). Decode (M = 32) uses the G6 1D-multicast program configs; prefill (M = chunk) the
        in1-multicast :func:`prefill_experts_pc` when M splits over its 64 cores (>= 2048 rows), else the op's auto
        config. Intermediates and ``y`` in ``memory_config`` (decode: L1; prefill: DRAM)."""
        mc = memory_config or self.dram
        M = int(f.shape[-2])
        x12 = ttnn.repeat(f, ttnn.Shape([1, self.e_loc, 1, 1]), memory_config=mc)
        gu_dtype = self.gate_up_dtype
        if gu_dtype is None:
            gu_dtype = ttnn.float32 if polynorm == "fp32" else ttnn.bfloat16
        if decode and M == TILE:
            pc_gu, pc_dn = self.pc_gate_up, self.pc_down
        elif not decode and self.prefill_pc:
            pc_gu = prefill_experts_pc(M // TILE, 2 * self.inter // TILE, out_block_w=20)
            pc_dn = prefill_experts_pc(M // TILE, self.hidden // TILE, out_block_w=16)
        else:
            pc_gu = pc_dn = None
        gu = ttnn.matmul(
            x12, self.w_gate_up, program_config=pc_gu, compute_kernel_config=self.ckc_experts, dtype=gu_dtype,
            memory_config=mc,
        )
        if x12 is not f:  # (a 1-expert-per-chip repeat could hand f back)
            _free(x12)
        h = self.polynorm(gu, mode=polynorm, row_scale=row_scale, memory_config=mc,
                          impl=self.polynorm_impl if decode else self.prefill_polynorm_impl)
        _free(gu)
        y = ttnn.matmul(
            h, self.w_down, program_config=pc_dn, compute_kernel_config=self.ckc_experts, dtype=self.down_dtype,
            memory_config=mc,
        )
        _free(h)
        return y

    def polynorm(self, gu, *, mode: str, row_scale=None, memory_config=None, impl: Optional[str] = None):
        """Grouped PolyNorm * up on ``gu [1, 12, M, 2560]`` -> ``h [1, 12, M, 1280]`` bf16 in ``memory_config``
        (intermediates too); ``row_scale [1, 12, M, 1]`` fp32 (routing weights) multiplies ``up`` in fp32 first, so
        ``h = poly(g) * (w * u)`` is rounded to bf16 once. ``impl`` defaults to ``self.polynorm_impl``."""
        mc = memory_config or self.dram
        impl = impl or self.polynorm_impl
        if impl == "local":
            consts = self.pn_fp32 if mode == "fp32" else self.pn_bf16
            return grouped_polynorm(gu, consts, inter=self.inter, mode=mode, eps=self.cfg.polynorm_eps,
                                    compute_kernel_config=self.ckc_polynorm, memory_config=mc, row_scale=row_scale)
        B, E, M, _ = (int(v) for v in gu.shape)
        g = ttnn.slice(gu, [0, 0, 0, 0], [B, E, M, self.inter], memory_config=mc)
        u = ttnn.slice(gu, [0, 0, 0, self.inter], [B, E, M, 2 * self.inter], memory_config=mc)
        if row_scale is not None:
            us = ttnn.multiply(row_scale, u, memory_config=mc)  # A = w (fp32): fp32 product, broadcast over cols
            _free(u)
            u = us
        h = _pn.grouped_polynorm(g, self.pn_consts, inter=self.inter, mode=mode, eps=self.cfg.polynorm_eps,
                                 compute_kernel_config=self.ckc_polynorm, memory_config=mc, up=u, impl=impl,
                                 intermediate_memory_config=mc)
        _free(g, u)
        return h

    def reduce_experts(self, y, *, memory_config=None):
        """``sum_e y[e]``: ``[1, 12, M, 4096]`` (already weighted) -> ``[1, 1, M, 4096]`` in ``combine_dtype``
        (``fast_reduce_nc``: FPU adds of the ``y`` terms into an fp32 dest, ``eltwise`` role, keepdim).

        The op packs its dest in its *output* dtype, which defaults to the input's; its Python binding exposes no
        ``output_dtype`` (``fast_reduce_nc_nanobind.cpp``), but a preallocated ``output`` sets the pack format
        (``compute_output_specs`` / the program factory's output CB format). So when ``combine_dtype`` differs from
        ``y``'s dtype the sum is packed straight into a ``combine_dtype`` tensor: an fp32 partial is never rounded to
        bf16, and no typecast op is needed. The allocation is host-side only (trace-safe)."""
        mc = memory_config or self.dram
        out = None
        if y.dtype != self.combine_dtype:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, int(y.shape[-2]), int(y.shape[-1])]), self.combine_dtype, ttnn.TILE_LAYOUT,
                self.mesh_device, mc,
            )
        return ttnn.experimental.fast_reduce_nc(
            y, dims=[1], output=out, memory_config=mc, compute_kernel_config=self.ckc_eltwise
        )

    def combine(self, y, w_loc, *, memory_config=None):
        """``combine_mode="multiply_sum"``: ``sum_e w_loc[e] * y[e]`` with ``y [1, 12, M, 4096]``, ``w_loc [1, 12, M, 1]``
        fp32 -> ``[1, 1, M, 4096]`` in ``combine_dtype`` (fp32 multiply, accurate fp32 reduction over the experts)."""
        mc = memory_config or self.dram
        weighted = ttnn.multiply(w_loc, y, memory_config=mc)  # follows A: fp32
        part = ttnn.sum(weighted, dim=1, keepdim=True, compute_kernel_config=self.ckc_eltwise, memory_config=mc)
        _free(weighted)
        if part.dtype != self.combine_dtype:
            p2 = ttnn.typecast(part, self.combine_dtype, memory_config=mc)
            _free(part)
            part = p2
        return part

    def local_partial(self, f, *, polynorm: str, decode: bool, taps: Optional[dict] = None, memory_config=None):
        """This chip's routed partial for the tokens ``f [1, 1, M, 4096]`` (identical on all chips): route, mask,
        experts, combine -> ``[1, 1, M, 4096]`` in ``combine_dtype``, DRAM (still to be summed over all 32 chips).
        ``memory_config``: intermediates (decode: L1). ``taps`` (tests) receives ``idx``, ``w`` (unscaled when
        ``route_scale`` is folded: multiply by ``route_scale / internal_route_scale`` to compare), ``w_loc``."""
        mc = memory_config or self.dram
        idx, w = self.router(f, scale=self.internal_route_scale, memory_config=mc)
        w_loc = self.local_weights(idx, w, memory_config=mc)
        if taps is not None:
            taps["idx"], taps["w"], taps["w_loc"] = idx, w, w_loc
        else:
            _free(idx, w)
        if self.combine_mode == "fold":
            y = self.experts(f, polynorm=polynorm, decode=decode, row_scale=w_loc, memory_config=mc)
            part = self.reduce_experts(y, memory_config=self.dram)
        else:
            y = self.experts(f, polynorm=polynorm, decode=decode, memory_config=mc)
            part = self.combine(y, w_loc, memory_config=self.dram)
        _free(y)
        if taps is None:
            _free(w_loc)
        return part

    # ==========================================================================================================
    # decode (MOE-1, MOE-5)
    # ==========================================================================================================
    def forward_decode(self, x, *, add_partial=None, reduce_tp: bool = True, taps: Optional[dict] = None):
        """Decode MoE for this DP row's 8 lanes.

        Args:
            x: ``[1, 1, 8, 4096]`` bf16 TILE (post_attention_layernorm output; replicated in the row, rows differ).
                Not consumed.
            add_partial: optional ``[1, 1, 8, 4096]`` TP partial (e.g. the shared expert's down-projection partial
                before its all_reduce) added before the final ``all_reduce(tp)``; dtype bf16 or ``combine_dtype``.
            reduce_tp: False returns the column partial ``[1, 1, 8, 4096]`` (``combine_dtype``) without the final
                ``all_reduce(tp)`` (the caller closes it).
            taps: tests only (eager): receives ``f_all``, ``idx``, ``w``, ``w_loc``, ``part`` (not freed).

        Returns ``[1, 1, 8, 4096]`` bf16 TILE DRAM: the routed MoE output (+ ``add_partial``), replicated in the row.
        On a mesh with a size-1 axis the collectives of that axis hand their input back; nothing the caller owns (``x``,
        ``add_partial``, ``taps``) is ever freed (:class:`_Keep`).
        """
        f_all = self.ccl.ag_dp_rows(x, memory_config=self.decode_mc)  # [1, 1, 32, 4096], lane order 8 dp + l
        part = self.local_partial(f_all, polynorm=self.decode_polynorm, decode=True, taps=taps,
                                  memory_config=self.decode_mc)
        if taps is not None:
            taps["f_all"] = f_all
            taps["part"] = part
        keep = _Keep(x, add_partial, *(taps.values() if taps is not None else ()))
        keep.drop(f_all)  # (is x itself when the DP axis has size 1)
        red = self.ccl.ar_dp(part)  # sum over the 4 chips of this column (all 32 tokens)
        keep.drop(part, red)
        mine = self.ccl.partition(red, 2, "dp")  # this row's 8 lanes
        keep.drop(red, mine)
        return self._close_tp(mine, add_partial, reduce_tp, keep)

    def _close_tp(self, part, add_partial, reduce_tp, keep: _Keep):
        if add_partial is not None:
            other = add_partial
            if other.dtype != part.dtype:
                other = ttnn.typecast(add_partial, part.dtype, memory_config=self.dram)
            s = ttnn.add(part, other, memory_config=self.dram)
            if other is not add_partial:
                _free(other)
            keep.drop(part, s)
            part = s
        if not reduce_tp:
            return part
        out = self.ccl.ar_tp(part)
        keep.drop(part, out)
        if out.dtype != ttnn.bfloat16:
            o2 = ttnn.typecast(out, ttnn.bfloat16, memory_config=self.dram)
            keep.drop(out, o2)
            out = o2
        return out

    # ==========================================================================================================
    # prefill (MOE-6)
    # ==========================================================================================================
    def dp_slice(self, t):
        """``[1, 1, S, ...]`` (replicated) -> this DP row's ``[1, 1, S/4, ...]`` rows ``[dp S/4, (dp+1) S/4)``: the rows
        the prefill ``add_partial`` must cover (e.g. compute the shared expert on ``dp_slice(f)``)."""
        return self.ccl.partition(t, 2, "dp")

    def forward_prefill(self, x, *, add_partial=None, reduce_tp: bool = True):
        """Prefill MoE for one user.

        Args:
            x: ``[1, 1, S, 4096]`` bf16 TILE, replicated on all 32 chips (S = bucket: a multiple of 128). Not consumed.
            add_partial: optional ``[1, 1, S/4, 4096]`` TP partial for this DP row's rows (:meth:`dp_slice`), added
                before ``all_reduce(tp)``.
            reduce_tp: False returns this row's reduced-over-DP slice ``[1, 1, S/4, 4096]`` before ``ar_tp`` / ``ag_dp``.

        Returns ``[1, 1, S, 4096]`` bf16, the routed MoE output, replicated on all chips. Never frees ``x`` or
        ``add_partial`` (also on meshes with a size-1 axis, where that axis' collectives return their input).
        """
        S = int(x.shape[-2])
        if S % (TILE * self.cfg.dp):
            raise ValueError(f"prefill length {S} must be a multiple of {TILE * self.cfg.dp}")
        keep = _Keep(x, add_partial)
        C = min(S, self.prefill_chunk)
        parts = []
        for c0 in range(0, S, C):
            c1 = min(S, c0 + C)
            xc = x if (c0 == 0 and c1 == S) else ttnn.slice(x, [0, 0, c0, 0], [1, 1, c1, self.hidden], memory_config=self.dram)
            parts.append(self.local_partial(xc, polynorm=self.prefill_polynorm, decode=False))
            keep.drop(xc)
        if len(parts) == 1:
            part = parts[0]
        else:
            part = ttnn.concat(parts, dim=2, memory_config=self.dram)
            _free(*parts)
        rs = self.ccl.rs_dp(part, 2)  # [1, 1, S/4, 4096]: this row's rows, summed over the column's 4 chips
        keep.drop(part, rs)
        if add_partial is not None:
            other = add_partial if add_partial.dtype == rs.dtype else ttnn.typecast(add_partial, rs.dtype)
            s = ttnn.add(rs, other, memory_config=self.dram)
            if other is not add_partial:
                _free(other)
            keep.drop(rs, s)
            rs = s
        if not reduce_tp:
            return rs
        ar = self.ccl.ar_tp(rs)
        keep.drop(rs, ar)
        out = self.ccl.ag_dp(ar, 2)
        keep.drop(ar, out)
        if out.dtype != ttnn.bfloat16:
            o2 = ttnn.typecast(out, ttnn.bfloat16, memory_config=self.dram)
            keep.drop(out, o2)
            out = o2
        return out

    # ==========================================================================================================
    # misc
    # ==========================================================================================================
    @property
    def expert_weight_bytes_per_chip(self) -> int:
        """Bytes of this chip's routed-expert weights (gate_up + down) on device (bfp8: 1088 B per 1024 values)."""
        n = self.e_loc * self.hidden * 3 * self.inter
        if self.experts_dtype == ttnn.bfloat8_b:
            return n * 1088 // 1024
        if self.experts_dtype == ttnn.bfloat4_b:
            return n * 576 // 1024
        return n * 2

    def deallocate(self) -> None:
        self.router.deallocate()
        self.pn_consts.deallocate()
        _free(self.w_gate_up, self.w_down, self.local_ids)


__all__ = ["COMBINE_MODES", "MotifMoE", "MotifRouter", "POLYNORM_IMPLS", "POLYNORM_MODES", "grouped_polynorm",
           "prefill_experts_pc"]
