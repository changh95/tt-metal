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
   A5 (``router_mask="scatter"``, ``MOTIF3_ROUTER_MASK``; off by default, docs/OPTIMIZATION_PLAN.md §3.3): steps 2-3
   without ``ttnn.gather`` -- :meth:`MotifRouter.route_local` scatters topk's idx into a 0/1 mask, normalizes over
   the 384 columns and extracts this chip's 12 weights with a constant one-hot (same top-8 sets incl. exact ties;
   weights within 1-4 fp32 ulp of the gather path; router + local mask 140 -> 104 us traced at M = 32,
   logs/opt/phaseA/A5).
   B4 (``router_mask="fused"``; off by default, docs/OPTIMIZATION_PLAN.md §3.3, results logs/opt/phaseB/B4): steps 2-3
   as the router matmul + one ``generic_op`` (:meth:`MotifRouter.route_fused`,
   :class:`~models.demos.motif3.tt.kernels.router_topk.FusedRouterTopK`, one core per gathered row): bias add on the
   fp32 SFPU, top-8 by exact fp32 compares (exact ties: lower id), fp32 normalization, this chip's 12 weights.
   B1 (``decode_experts="sparse"``, ``MOTIF3_DECODE_EXPERTS``; off by default, docs/OPTIMIZATION_PLAN.md §3.3, probe
   logs/opt/phaseA/M6): ``w_loc *= lane_mask`` (the step's gathered ``[1, 1, M, 1]`` fp32 0/1 mask of the live rows,
   :meth:`MotifMoE.decode_lane_mask`, built once per step by the model) so inactive lanes route nothing, then the
   sparsity ``s = max_M(w_loc)`` -> bf16 -> ROW_MAJOR ``[1, 1, 1, 12]`` (nonzero = some live row routes to the expert).
4. Experts (MOE-4, G6): ``repeat(f_all) [1,12,32,4096] @ W_gate_up [1,12,4096,2560]`` (bfp8, ``experts_gate_up_pc``,
   ``experts`` role HiFi4 + fp32 acc, fp32 output) -> grouped PolyNorm from ``tt/polynorm.py`` (exact fp32 moments +
   Horner ``mac``, fp32 intermediates) with the routing weights folded into ``up`` (``h = poly(g) * (w_loc * u)``,
   rounded to bf16 once; never block float) -> ``@ W_down [1,12,1280,4096]`` (bfp8, ``experts_down_pc``; the PolyNorm
   x0.5 and route_scale x2.0 folded into ``W_down``: net x1.0, exact) -> ``y [1,12,32,4096]`` = weighted expert outputs.
   B3 (``moe_polynorm="fused"``, ``MOTIF3_MOE_POLYNORM``; off by default, docs/OPTIMIZATION_PLAN.md §3.3, prototype
   logs/opt/phaseA/M10): the grouped PolyNorm and the routing-weight multiply are one ``generic_op``
   (:class:`~models.demos.motif3.tt.kernels.moe_polynorm.FusedGroupedPolyNorm`: fp32 moments + Horner on 48 cores,
   ``h`` rounded to bf16 once; not bitwise equal to the composite) at the decode row counts 32 / 64, dense and B1 sparse.
5. Combine (MOE-5): ``fast_reduce_nc(y, dims=[1])`` (bf16 terms summed in an fp32 dest, packed in ``combine_dtype``:
   bf16 by default; ``combine_dtype=fp32`` packs the fp32 dest into a preallocated fp32 output, so the per-chip partial
   is a true fp32 sum) -> ``[1,1,32,4096]`` -> ``ccl.ar_dp`` (sum over the 4 chips of the column) ->
   ``ccl.partition(2, "dp")`` (this row's 8 lanes) -> ``+ add_partial`` -> ``ccl.ar_tp`` (sum over the 8 columns) ->
   ``[1,1,8,4096]`` bf16 DRAM: the full routed MoE output, replicated in the row.
   All decode intermediates live in L1 (freed inside the call; ~220 us faster than DRAM); CCL payloads and the output
   are DRAM.

   B1 sparse experts (``decode_experts="sparse"``): step 4 runs ``ttnn.sparse_matmul(f_all, W_gate_up, s)`` (no
   repeat; ``is_input_b_sparse``: the experts with ``s == 0`` are skipped and their output slices zero-filled) ->
   the same PolyNorm (zero rows give exactly 0: ``row_scale`` is 0 there) -> ``sparse_matmul(h, W_down, s)``
   (``is_input_a_sparse`` and ``is_input_b_sparse``) with the same program configs, compute config and dtypes as
   the dense matmuls. ``nnz`` is always ``None`` (counted on device at run time): a static ``nnz`` that differs from
   the real count deadlocks the op on BH (tt-metal #45943 / #45052), and the count is data dependent. The live rows
   are bitwise equal to the dense path (an expert no live row routes to adds exactly 0 there: its weight is 0); the
   rows of inactive lanes come out as 0 + the shared expert instead of garbage (nothing reads them). Measured
   (M6, real L2 routes, traced, TORUS_XY): 1035 -> 524 / 631 / 904 us per layer at 1 / 8 / 32 live lanes, T64
   1211 -> 687 / 911 / 1183 us at 2 / 16 / 64 live rows; sparsity 20 us per layer, lane mask 37 us per step.

Measured on this Galaxy (TORUS_Y fabric, 4x8, real layer-2/4 weights; ``tests/unit/test_moe.py``): decode **1036 /
1041 us traced** per call at L2 / L4 (eager 4.6 ms, dispatch bound). Per stage at L2 (each traced alone; small stages
measured with enough calls to clear the ~150 us sync floor): gather 26, router 118, local mask 19, repeat 6, gate_up 429,
PolyNorm 126, down 220, expert sum 7, AR(dp) 32, partition 13, AR(tp) 34 us (sum 1030 us); 200.5 MB of expert weights
per chip at 309 GB/s. Variants: PolyNorm bf16 1018-1023 us, fp32 partials 1072 us, DRAM intermediates 1260 us.
Accuracy vs the fp32 reference (reference modules, bf16-valued weights): PCC 0.99996 (L2 / L4, bfp8 experts,
rel 0.9 %), every token >= 0.9999 against the reference experts on the device's routes; bf16 experts: 0.999994 (real L2,
``fold_route_scale=False``), 0.999993 (random); traced output == eager bitwise.

T64 verify step (docs/p5_t64/P5_T64_DESIGN.md §4.2, T4; ``cfg.wide_rows_per_dp`` = 16): ``forward_decode`` also takes
16 rows per DP row (``[8 anchors | 8 drafts]``, still one tile row), so the gather yields M = 64 tokens (natural order
``16 dp + j``) and ``partition`` returns the row's 16. The decode program configs are per M (:data:`DECODE_ROWS`): the
router linear (12 x 1 cores, sigmoid fused) and gate_up (10 x 8) with ``per_core_M`` 2, the down projection with
``per_core_M`` 2 on 8 x 8 cores (``model_config.experts_*_pc(m_tiles=2)`` / ``router_decode_pc(m_tiles=2)``, exactly the
G16-lite probe's configs), L1 intermediates as at M = 32, and a 64-row -inf top-k pad allocated in the constructor when
the config stages the T64 step (never after a trace capture; F3N rule R3). With ``router_logits="exact_fp32"`` the
router runs the exact kernel at M = 64 too (:data:`EXACT_ROUTER_DECODE_ROWS`; the kernel takes M = 32 n, review edit
R-E7).
Every row of the M = 64 module equals the M = 32 module's row of the same token bitwise (both routers; G16-lite measured
it with the composite router: 1204.1 us per layer at M = 64 vs 1038.6 us, +16 %; ``test_moe_device_t64_rows``).
``forward_decode`` refuses a gathered row count the module was not built for (no silent fallback to auto configs).

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

B2a token-compacted prefill (``prefill_moe="compact"``, ``MOTIF3_PREFILL_MOE``; the default, docs/OPTIMIZATION_PLAN.md
§3.3 B2, prototype logs/opt/phaseA/m7, results logs/opt/phaseB/B2a), per chunk of at least ``prefill_moe_min_rows``
rows (:meth:`MotifMoE._compact_partial`): the router as above; the chunk's ``idx`` read from chip 0 (one blocking read:
routes are identical on every chip); on the host (:func:`compact_upload_fast`) every chip's (local expert, token) rows
sorted by expert then token, each expert padded to whole blocks of ``mb`` rows (:func:`compact_block`), the busiest
chip's block count rounded up the ladder of the chunk size (:func:`compact_ladder`); one uint32 upload per chunk (row
tokens, combine keys, routing-weight gather indices, per-block sparsity and PolyNorm constants as bf16 bit patterns).
On every chip (:meth:`MotifMoE._compact_device`): ``ttnn.embedding`` gathers the routed rows -> ``[1, nb, mb, 4096]``;
``ttnn.sparse_matmul`` (one expert per block, ``nnz = nb`` exactly) for gate_up and down at the decode experts' program
configs; the grouped PolyNorm with per-block constants and the routing weights (gathered from the dense path's
``w_loc``) folded into ``up``; the combine is ``P^T [M, R] @ y [R, 4096]`` with ``P^T`` the one-hot of each row's token,
so a token's expert terms add in the dense order. The partial is bitwise equal to the dense path's on all 32 chips;
the RS(dp) / AR(tp) / AG(dp) that follow are unchanged. Every (chunk size, block count) of the ladder compiles in
:meth:`MotifMoE.warm_compact` (the generator's ``warmup_prefill``); a chunk beyond the ladder's cap, or (after the decode
capture, :class:`CompactPrefillState`) a shape not warmed, runs the dense path on the same routes.

B2b kernels (logs/opt/phaseB/B2b; ``prefill_moe_dispatch`` / ``prefill_moe_combine``, ``MOTIF3_PREFILL_MOE_DISPATCH`` /
``MOTIF3_PREFILL_MOE_COMBINE``): "device" builds the same rows with one ``generic_op`` from the router's ``idx`` and
``w`` into capacity buffers (:class:`~models.demos.motif3.tt.kernels.moe_compact.CompactDispatch`); the host reads only
the 32-byte block count and slices the first NB blocks (:meth:`MotifMoE._compact_partial_device`), so the routes are
never read and no upload is made. "gather" sums each token's rows with
:class:`~models.demos.motif3.tt.kernels.moe_compact.GatherCombine` (fast_reduce_nc's fp32-dest adds) instead of the
one-hot matmul. Every combination is bitwise equal to the dense path.

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

import math
from typing import Dict, Iterable, List, Optional, Tuple

import ttnn

from . import polynorm as _pn
from . import weights as W
from .ccl import MotifCCL, device_tensors_to_torch
from .model_config import (DECODE_EXPERTS_MODES, MOE_POLYNORM_MODES, PREFILL_MOE_COMBINE_MODES,
                           PREFILL_MOE_DISPATCH_MODES, PREFILL_MOE_MODES, ROUTER_MASK_MODES, TILE, MotifTTConfig,
                           mcast1d_matmul_pc)

POLYNORM_MODES = ("fp32", "bf16")
POLYNORM_IMPLS = ("horner", "rms", "local")  # tt/polynorm.py impls + this file's G6 copy
COMBINE_MODES = ("fold", "multiply_sum")
# Gathered decode token counts M with measured decode program configs: the 32 lanes of the 32-lane step (one tile row)
# and the 64 rows of the T64 verify step (16 per DP row; docs/p5_t64/P5_T64_DESIGN.md §4.2). A module serves M = 64 at
# decode only when built with a T64 config (cfg.wide_rows_per_dp > 0: its 64-row constants exist).
DECODE_ROWS = (TILE, 2 * TILE)
# Gathered decode token counts at which MotifRouter runs the exact-fp32 logits kernel when it has one
# (router_logits="exact_fp32"; kernels/router_fp32.py takes M = 32 n, the two-row launch pipelines its tile rows with
# the same arithmetic per row). Review edit R-E7: at both counts the T64 rows then equal the T32 rows bitwise.
# model_config.ROUTER_EXACT_FP32_DECODE_ROWS must list the same counts (it gates spec_verify="auto" + "exact_fp32").
EXACT_ROUTER_DECODE_ROWS = (TILE, 2 * TILE)


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


def _same_buffer(a, b) -> bool:
    try:
        return a.buffer_address() == b.buffer_address()
    except Exception:  # host tensors / stand-ins: no buffer
        return False


def _reshape(t, shape):
    """``ttnn.reshape`` that keeps exactly one live handle: frees the input when the op copied (a view shares the
    buffer, and then the input handle must not be freed)."""
    r = ttnn.reshape(t, ttnn.Shape(list(shape)))
    if r is not t and not _same_buffer(r, t):
        _free(t)
    return r


def resolve_decode_experts(decode_experts: Optional[str], cfg, combine_mode: str) -> str:
    """B1: the decode-experts mode a :class:`MotifMoE` runs. ``decode_experts`` (explicit) or ``cfg.decode_experts``
    (``MOTIF3_DECODE_EXPERTS``; ``None`` = "dense") must be in :data:`DECODE_EXPERTS_MODES`. "sparse" needs
    ``combine_mode="fold"`` (the routing weights folded into the PolyNorm output, so a skipped expert's zero slice is
    its exact contribution): an explicit request raises, the config default falls back to "dense" for the HF-order
    ``multiply_sum`` diagnostic module."""
    explicit = decode_experts is not None
    mode = str(decode_experts if explicit else (getattr(cfg, "decode_experts", None) or "dense"))
    if mode not in DECODE_EXPERTS_MODES:
        raise ValueError(f"decode_experts must be one of {DECODE_EXPERTS_MODES}, got {mode!r}")
    if mode == "sparse" and combine_mode != "fold":
        if explicit:
            raise ValueError(
                f"decode_experts='sparse' needs combine_mode='fold' (the routing weights folded into the PolyNorm "
                f"output, so a skipped expert's zero slice is its exact contribution), got {combine_mode!r}"
            )
        return "dense"
    return mode


def resolve_moe_polynorm(moe_polynorm: Optional[str], cfg, *, decode_polynorm: str, combine_mode: str,
                         gate_up_dtype=None) -> str:
    """B3: the decode routed-expert PolyNorm a :class:`MotifMoE` runs. ``moe_polynorm`` (explicit) or
    ``cfg.moe_polynorm`` (``MOTIF3_MOE_POLYNORM``; ``None`` = "composite") must be in :data:`MOE_POLYNORM_MODES`.
    "fused" needs the fp32 decode PolyNorm with an fp32 gate_up output (the kernel reads fp32 ``gu``) and
    ``combine_mode="fold"`` (it folds the routing weights): an explicit request raises otherwise, the config default
    falls back to "composite" (diagnostic modules)."""
    explicit = moe_polynorm is not None
    mode = str(moe_polynorm if explicit else (getattr(cfg, "moe_polynorm", None) or "composite"))
    if mode not in MOE_POLYNORM_MODES:
        raise ValueError(f"moe_polynorm must be one of {MOE_POLYNORM_MODES}, got {mode!r}")
    if mode == "fused":
        why = None
        if decode_polynorm != "fp32":
            why = f"decode_polynorm='fp32' (got {decode_polynorm!r})"
        elif gate_up_dtype is not None and gate_up_dtype != ttnn.float32:
            why = f"an fp32 gate_up output (got {gate_up_dtype})"
        elif combine_mode != "fold":
            why = f"combine_mode='fold' (got {combine_mode!r})"
        if why is not None:
            if explicit:
                raise ValueError(f"moe_polynorm='fused' needs {why}")
            return "composite"
    return mode


def resolve_prefill_moe(prefill_moe: Optional[str], cfg, *, combine_mode: str, prefill_polynorm: str,
                        prefill_polynorm_impl: str) -> str:
    """B2a: the prefill routed-experts mode a :class:`MotifMoE` runs. ``prefill_moe`` (explicit) or ``cfg.prefill_moe``
    (``MOTIF3_PREFILL_MOE``, default "compact"; a config without the field = "dense") must be in
    :data:`PREFILL_MOE_MODES`. "compact" needs
    ``combine_mode="fold"`` (routing weights folded into ``up``), the bf16 prefill PolyNorm (its per-block constants
    travel as bf16) and the ``rms`` impl (constants as a ``{c0, c1, c2, b}`` mapping): an explicit request raises
    otherwise, the config default falls back to "dense" (diagnostic modules)."""
    explicit = prefill_moe is not None
    mode = str(prefill_moe if explicit else (getattr(cfg, "prefill_moe", None) or "dense"))
    if mode not in PREFILL_MOE_MODES:
        raise ValueError(f"prefill_moe must be one of {PREFILL_MOE_MODES}, got {mode!r}")
    if mode == "compact":
        why = None
        if combine_mode != "fold":
            why = f"combine_mode='fold' (got {combine_mode!r})"
        elif prefill_polynorm != "bf16":
            why = f"prefill_polynorm='bf16' (got {prefill_polynorm!r})"
        elif prefill_polynorm_impl != "rms":
            why = f"prefill_polynorm_impl='rms' (got {prefill_polynorm_impl!r})"
        if why is not None:
            if explicit:
                raise ValueError(f"prefill_moe='compact' needs {why}")
            return "dense"
    return mode


def resolve_prefill_moe_kernels(dispatch: Optional[str], combine: Optional[str], cfg) -> Tuple[str, str]:
    """B2b: ``(prefill_moe_dispatch, prefill_moe_combine)`` of a compacted :class:`MotifMoE`: explicit values or
    ``cfg.prefill_moe_dispatch`` / ``cfg.prefill_moe_combine`` (``MOTIF3_PREFILL_MOE_DISPATCH`` / ``_COMBINE``; a config
    without the fields = B2a's "host" / "matmul"); each must be in :data:`PREFILL_MOE_DISPATCH_MODES` /
    :data:`PREFILL_MOE_COMBINE_MODES`."""
    d = str(dispatch if dispatch is not None else (getattr(cfg, "prefill_moe_dispatch", None) or "host"))
    c = str(combine if combine is not None else (getattr(cfg, "prefill_moe_combine", None) or "matmul"))
    if d not in PREFILL_MOE_DISPATCH_MODES:
        raise ValueError(f"prefill_moe_dispatch must be one of {PREFILL_MOE_DISPATCH_MODES}, got {d!r}")
    if c not in PREFILL_MOE_COMBINE_MODES:
        raise ValueError(f"prefill_moe_combine must be one of {PREFILL_MOE_COMBINE_MODES}, got {c!r}")
    return d, c


# B2b device dispatch: the largest chunk the kernel serves (its L1 holds the chunk's routes and weights: ~0.6 MB per core
# at 4096 rows); a larger compacted chunk builds its rows on the host (B2a).
PREFILL_MOE_DISPATCH_MAX_ROWS = 4096


# ==============================================================================================================
# B2a: token-compacted prefill experts (host-built row lists; docs/OPTIMIZATION_PLAN.md §3.3 B2, logs/opt/phaseB/B2a)
# ==============================================================================================================
# The NB ladder (blocks per chip) of one chunk size: geometric from the floor (max(e_loc, M / 4 rows)) by this ratio up
# to PREFILL_MOE_CAP x M rows; a chunk whose busiest chip needs more blocks than the cap runs the dense path (exact too).
PREFILL_MOE_LADDER_RATIO = 1.25
PREFILL_MOE_CAP = 2.0


def compact_block(rows: int, block) -> int:
    """Rows per expert block of the compacted prefill experts for a chunk of ``rows`` rows: ``block`` ("auto" | "32" |
    "64" | "128", ``cfg.prefill_moe_block``). "auto": 32 up to 2048-row chunks, 64 above (M7: the hot experts of the
    deep layers favour bigger blocks at 4K and 8K, the shallow layers 32 at every size)."""
    b = str(block).strip().lower()
    if b == "auto":
        return TILE if int(rows) <= 2048 else 2 * TILE
    v = int(b)
    if v not in (32, 64, 128):
        raise ValueError(f"prefill MoE block must be 'auto', 32, 64 or 128, got {block!r}")
    return v


def compact_ladder(rows: int, mb: int, e_loc: int) -> Tuple[int, ...]:
    """The block counts NB the compacted path of a ``rows``-row chunk runs at (each one compiled program set): from
    ``max(e_loc, ceil(rows / 4 / mb))`` up by :data:`PREFILL_MOE_LADDER_RATIO` (at least +1) to
    ``ceil(PREFILL_MOE_CAP * rows / mb)``, which is always the last entry."""
    lo = max(int(e_loc), -(-int(rows) // (4 * int(mb))))
    cap = max(lo, -(-int(PREFILL_MOE_CAP * int(rows)) // int(mb)))
    out, b = [], lo
    while b < cap:
        out.append(b)
        b = max(b + 1, math.ceil(b * PREFILL_MOE_LADDER_RATIO))
    out.append(cap)
    return tuple(out)


def compact_bucket(need: int, ladder: Tuple[int, ...]) -> Optional[int]:
    """Smallest ladder entry >= ``need`` blocks, or None (beyond the cap: the dense path)."""
    for b in ladder:
        if b >= int(need):
            return int(b)
    return None


def compact_need_blocks(idx, local_ids, mb: int) -> int:
    """Blocks the busiest chip needs: ``idx [S, K]`` (global expert ids, torch int), ``local_ids [P, E]`` (chip ``p``'s
    local expert ``e``, global id) -> ``max_p sum_e ceil(count(p, e) / mb)``."""
    import torch

    P, E = (int(v) for v in local_ids.shape)
    pe = _slots(idx, local_ids)
    cnt = torch.bincount(pe, minlength=P * E).reshape(P, E)
    return int(((cnt + mb - 1) // mb).sum(1).max())


def _slots(idx, local_ids):
    """Chip-expert slot ``p * E + e`` of every routed id of ``idx`` (flattened); an id no chip holds raises."""
    g2s = _global_to_slot(local_ids)
    flat = idx.reshape(-1).long()
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= int(g2s.numel())):
        raise ValueError("a routed expert id is not any chip's local expert")
    pe = g2s[flat]
    if bool((pe < 0).any()):
        raise ValueError("a routed expert id is not any chip's local expert")
    return pe


def _global_to_slot(local_ids):
    import torch

    P, E = (int(v) for v in local_ids.shape)
    flat = local_ids.reshape(-1).long()
    g2s = torch.full((int(flat.max()) + 1,), -1, dtype=torch.long)
    g2s[flat] = torch.arange(P * E, dtype=torch.long)
    if int((g2s >= 0).sum()) != P * E:
        raise ValueError("local expert ids must be distinct")
    return g2s


def compact_prefill_meta(idx, w, local_ids, mb: int, nb: int, pn=None) -> Dict[str, object]:
    """Host row lists of the compacted prefill experts (pure torch; every chip at once).

    Args:
        idx: ``[S, K]`` global expert ids of each token (the router's top-8, identical on every chip).
        w: ``[S, K]`` fp32 routing weights (same order as ``idx``; copied bit-exactly).
        local_ids: ``[P, E]`` chip ``p``'s local expert ``e`` (global id), ``p`` = row-major mesh coordinate.
        mb: rows per block; nb: blocks per chip (``>=`` :func:`compact_need_blocks`).
        pn: optional ``[P, E, C]`` per-expert constants (the PolyNorm ``c0, c1, c2, b``); gathered per block.

    Returns per chip (rows ``R = nb * mb``): ``tok [P, R]`` int32 (token of each row; pad rows 0), ``tokv [P, R]`` fp32
    (token, pad rows -1: they never match a token in the combine), ``w [P, R]`` fp32 (pad rows 0), ``eblk [P, nb]``
    (local expert of each block; the unused trailing blocks repeat the last expert: their rows are pad rows),
    ``sparsity [P, nb, E]`` (one-hot of ``eblk``: exactly ``nb`` nonzeros per chip), ``cblk [P, nb, C]`` (with ``pn``),
    ``need`` (blocks the busiest chip needs). Rows are sorted by (local expert, token): expert ``e``'s rows start at a
    block boundary, its tokens ascending -- the dense path's expert order, so the combine adds a token's expert terms
    in the same order."""
    import torch

    S, K = (int(v) for v in idx.shape)
    P, E = (int(v) for v in local_ids.shape)
    mb, nb = int(mb), int(nb)
    R = nb * mb
    pe = _slots(idx, local_ids)  # slot p * E + e of every (token, k)
    t = torch.arange(S, dtype=torch.long).repeat_interleave(K)
    cnt = torch.bincount(pe, minlength=P * E)  # [P * E]
    nblk = (cnt + mb - 1) // mb
    need = int(nblk.reshape(P, E).sum(1).max())
    if need > nb:
        raise ValueError(f"{nb} blocks of {mb} rows hold fewer rows than the busiest chip needs ({need} blocks)")
    blk_end = nblk.reshape(P, E).cumsum(1)  # [P, E]
    blk_off = (blk_end - nblk.reshape(P, E)).reshape(-1)  # [P * E] first block of each slot
    order = torch.argsort(pe * S + t)  # unique keys: (slot, token) ascending
    pe_s, t_s = pe[order], t[order]
    w_s = w.reshape(-1).to(torch.float32)[order]
    start = cnt.cumsum(0) - cnt  # first sorted index of each slot
    rank = torch.arange(S * K, dtype=torch.long) - start[pe_s]
    p_s = pe_s // E
    row = blk_off[pe_s] * mb + rank
    tok = torch.zeros(P, R, dtype=torch.int32)
    tokv = torch.full((P, R), -1.0, dtype=torch.float32)
    wv = torch.zeros(P, R, dtype=torch.float32)
    tok[p_s, row] = t_s.to(torch.int32)
    tokv[p_s, row] = t_s.to(torch.float32)
    wv[p_s, row] = w_s
    b = torch.arange(nb, dtype=torch.long)
    eblk = (blk_end.unsqueeze(-1) <= b.view(1, 1, nb)).sum(1).clamp(max=E - 1)  # [P, nb]
    sp = torch.nn.functional.one_hot(eblk, E).to(torch.float32)  # [P, nb, E]
    out = dict(tok=tok, tokv=tokv, w=wv, eblk=eblk, sparsity=sp, need=need)
    if pn is not None:
        out["cblk"] = torch.gather(pn, 1, eblk.unsqueeze(-1).expand(P, nb, int(pn.shape[-1])))
    return out


def compact_upload_fast(idx, g2s, P: int, E: int, mb: int, ladder: Tuple[int, ...], pn_bits, rows_m: int):
    """The serving path's host metadata in one numpy pass (B2a): ``(need, nb, u)`` with ``u`` the ``[P, 4, nb * mb]``
    int32 upload of :meth:`MotifMoE._compact_device` -- word for word what ``compact_upload_rows(compact_prefill_meta(
    ...))`` builds (tested) -- or ``(need, None, None)`` when the busiest chip needs more blocks than the ladder's cap.

    ``idx [S, K]`` int (global expert ids), ``g2s`` the global id -> slot ``p * E + e`` table (:func:`_global_to_slot`,
    numpy), ``pn_bits [P, E, 4]`` int32 bf16 bit patterns of the per-expert constants, ``rows_m`` the chunk size M (the
    pad rows' combine key, and the stride of the routing-weight gather index ``e M + t``)."""
    import numpy as np

    K = int(np.asarray(idx).shape[-1])
    idx = np.asarray(idx).reshape(-1).astype(np.int64, copy=False)
    S_K = idx.size
    pad_key = int(rows_m)
    if S_K and (int(idx.min()) < 0 or int(idx.max()) >= g2s.size):
        raise ValueError("a routed expert id is not any chip's local expert")
    pe = g2s[idx]
    if S_K and int(pe.min()) < 0:
        raise ValueError("a routed expert id is not any chip's local expert")
    cnt = np.bincount(pe, minlength=P * E)
    nblk = ((cnt + (mb - 1)) // mb).reshape(P, E)
    need = int(nblk.sum(1).max()) if S_K else 0
    nb = compact_bucket(need, ladder)
    if nb is None:
        return need, None, None
    rows = int(nb) * int(mb)
    order = np.argsort(pe, kind="stable")  # flattened order is token-major: tokens ascending within each slot
    pe_s = pe[order]
    start = np.cumsum(cnt) - cnt
    blk_end = np.cumsum(nblk, axis=1)
    blk_off = (blk_end - nblk).reshape(-1)
    row = blk_off[pe_s] * mb + (np.arange(S_K) - start[pe_s])
    flat = (pe_s // E) * (4 * rows) + row
    t_s = (order // K).astype(np.int32)
    u = np.zeros((P, 4, rows), dtype=np.int32)
    u[:, 1, :] = int(pad_key)
    uf = u.reshape(-1)
    uf[flat] = t_s
    uf[flat + rows] = t_s
    uf[flat + 2 * rows] = (pe_s % E).astype(np.int32) * int(pad_key) + t_s
    eblk = np.minimum((blk_end[:, :, None] <= np.arange(nb)[None, None, :]).sum(1), E - 1)  # [P, nb]
    blk = np.zeros((P, int(nb), 32), dtype=np.int32)
    np.put_along_axis(blk, eblk[:, :, None], 0x3F80, axis=2)  # bf16 1.0 at the block's expert
    blk[:, :, 16:20] = np.take_along_axis(pn_bits, eblk[:, :, None], axis=1)
    u[:, 3, : 32 * int(nb)] = blk.reshape(P, -1)
    return need, int(nb), u


class CompactPrefillState:
    """State the compacted prefill MoE layers of one model share (B2a): the per-chip upload mapper, the host copy of the
    chips' local expert ids, the combine's token-index columns per chunk size, the (rows, blocks) shapes compiled so far
    and counters.

    ``frozen`` (a callable; the generator sets it to "a decode trace is captured"): while it returns True only the
    shapes :meth:`MotifMoE.warm_compact` compiled run compacted; any other chunk falls back to the dense path (compiled
    by the same warm-up), so nothing compiles after the decode trace capture (F3N rule R2). Before a capture (tests,
    scripts, the warm-up itself) every shape compiles on first use. ``owner``: the object that frees :attr:`iota`."""

    def __init__(self, owner=None):
        self.owner = owner
        self.mapper = None
        self.local_ids = None  # torch [P, E] long
        self.iota: Dict[int, object] = {}  # rows -> [1, 1, rows, 1] fp32 TILE (0 .. rows-1), replicated
        self.warmed = set()  # (rows, mb, nb)
        self.frozen = lambda: False
        self.stats: Dict[str, int] = {"compact": 0, "dense_cap": 0, "dense_unwarmed": 0}
        self.blocks: Dict[Tuple[int, int, int], int] = {}  # (rows, mb, nb) -> calls
        self.host_bufs: Dict[tuple, object] = {}  # (shape, dtype, layout) -> host staging tensor of the routes read
        self.g2s = None  # numpy global expert id -> slot p * E + e
        self.dispatch = None  # B2b: kernels.moe_compact.CompactDispatch (device dispatch; owns the slot table)
        self.combine = None  # B2b: kernels.moe_compact.GatherCombine (gather combine)
        self.stats["device_dispatch"] = 0

    def allows(self, rows: int, mb: int, nb: int) -> bool:
        return (int(rows), int(mb), int(nb)) in self.warmed or not self.frozen()

    def deallocate(self) -> None:
        for t in self.iota.values():
            _free(t)
        self.iota = {}
        self.host_bufs = {}
        if self.dispatch is not None:
            self.dispatch.deallocate()
            self.dispatch = None
        if self.combine is not None:
            self.combine.deallocate()
            self.combine = None


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


def wide_decode_rows(cfg) -> int:
    """Gathered decode token count of the T64 verify step ``cfg`` stages: ``dp * cfg.wide_rows_per_dp`` (64: 16 rows
    per DP row), or 0 when it stages none (``spec_verify="packed"`` or no speculation). The modules allocate their
    64-row constants when it is set (F3N rule R3). Raises for a T64 row count without decode configs here."""
    m = int(cfg.dp) * int(getattr(cfg, "wide_rows_per_dp", 0) or 0)
    if m and m not in DECODE_ROWS[1:]:
        raise ValueError(f"T64 decode of {m} gathered rows has no MoE decode configs (have {DECODE_ROWS[1:]})")
    return m


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
    ``logits_fn``: optional decode-shape logits replacement (``kernels.router_fp32.RouterLogitsFP32``, exact fp32), used
    at the gathered decode row counts :data:`EXACT_ROUTER_DECODE_ROWS` (32 and the T64 step's 64).

    Decode shapes are per gathered row count M (:data:`DECODE_ROWS`; T64, docs/p5_t64/P5_T64_DESIGN.md §4.2): M = 32
    uses ``decode_pc`` / ``decode_pc_sigmoid`` (diagnostics may replace ``decode_pc``; the fused twin is then not used),
    M = 64 ``decode_pcs_wide[64]`` (the same 12 x 1 configs with ``per_core_M`` 2, ``model_config.router_decode_pc(
    m_tiles=2)``). The -inf top-k pads ``_pads[M]``: M = 32 always (with ``topk_pad_to``), M = 64 when
    ``cfg.wide_rows_per_dp`` is set -- allocated here, before any trace capture (F3N rule R3); without it a 64-row call
    takes the unpadded single-core top-k.
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
        # Optional replacement of the decode-shape logits (M in EXACT_ROUTER_DECODE_ROWS: 32, and the T64 step's 64),
        # e.g. the exact-fp32 SFPU kernel ``kernels.router_fp32.RouterLogitsFP32`` (same contract as
        # :meth:`route_logits`); prefill shapes keep the composite linear (the kernel's prefill cost, ~30 us per 32
        # tokens, would dominate there).
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
        # T64: the 64-row twins (per_core_M 2), {M: (config, config + sigmoid)}; configs only (no device memory)
        self.decode_pcs_wide: Dict[int, Tuple[object, object]] = {}
        gx, gy = cfg.compute_grid
        if router_decode_pc and gx >= 12:
            self.decode_pc = self._decode_pc()
            self.decode_pc_sigmoid = self._decode_pc(sigmoid=True)
            for m in DECODE_ROWS[1:]:
                self.decode_pcs_wide[m] = (
                    self._decode_pc(m_tiles=m // TILE),
                    self._decode_pc(sigmoid=True, m_tiles=m // TILE),
                )
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
            wide = wide_decode_rows(cfg)
            if wide:  # the T64 step's 64 gathered rows: allocated before any trace capture (F3N rule R3)
                self._pads[wide] = self._make_pad(wide)

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

    def _decode_pc(self, sigmoid: bool = False, m_tiles: int = 1):
        """Decode router linear config on ``m_tiles`` tile rows of gathered tokens (``per_core_M``; 2 = the T64 step's
        64 rows); equals ``model_config.router_decode_pc(sigmoid=, m_tiles=)``."""
        pc = mcast1d_matmul_pc(
            (12, 1), self.n_experts // TILE, 32, self.cfg.hidden_size // TILE, per_core_m=int(m_tiles)
        )
        if sigmoid:  # SFPU sigmoid (accurate: sigmoid_tile<RC, false>) on the fp32 dest, before the pack
            pc.fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
        return pc

    def _decode_pcs(self, M: int) -> Tuple[object, object]:
        """``(linear config, its fused-sigmoid twin or None)`` of the router linear on ``M`` gathered decode rows;
        ``(None, None)`` for other M (prefill shapes: ttnn's auto config). M = 32 reads ``decode_pc`` /
        ``decode_pc_sigmoid``: a replaced ``decode_pc`` (diagnostics) runs without the fused twin."""
        if M == TILE:
            pc = self.decode_pc
            return pc, (self.decode_pc_sigmoid if pc is not None and pc is self._decode_pc_base else None)
        return self.decode_pcs_wide.get(M, (None, None))

    def _pc(self, f):
        return self._decode_pcs(int(f.shape[-2]))[0]

    def _use_logits_fn(self, f) -> bool:
        """The decode-shape logits replacement (``logits_fn``) applies: one is set and ``f`` holds a gathered decode
        row count of :data:`EXACT_ROUTER_DECODE_ROWS` (32, or the T64 step's 64; review edit R-E7)."""
        return self.logits_fn is not None and int(f.shape[-2]) in EXACT_ROUTER_DECODE_ROWS

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
        scores, biased, idx = self._scores_topk(f, mc)
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

    def _scores(self, f, mc):
        """``f [1, 1, M, 4096]`` -> ``scores [1, 1, M, 384]`` fp32 ``= sigmoid(logits)`` in ``mc``: the matmul part of
        the router head, shared by :meth:`_scores_topk` and :meth:`route_fused`."""
        M = int(f.shape[-2])
        exact = self._use_logits_fn(f)
        pc, pc_sigmoid = self._decode_pcs(M)
        if not exact and self.sigmoid_in_pc and pc_sigmoid is not None:
            # decode shape (M = 32 or 64): sigmoid fused into the linear's program config (no separate op); a replaced
            # decode_pc (diagnostics) falls through to the separate sigmoid with that config
            scores = ttnn.linear(
                f, self.weight, dtype=ttnn.float32, compute_kernel_config=self.ckc,
                program_config=pc_sigmoid, memory_config=mc,
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
        return scores

    def _scores_topk(self, f, mc) -> Tuple[object, object, object]:
        """The router head shared by :meth:`__call__` and :meth:`route_local`: ``f [1, 1, M, 4096]`` ->
        ``(scores [1, 1, M, 384] fp32 = sigmoid(logits), biased = scores + expert_bias, idx [1, 1, M, 8])``, all in
        ``mc`` (the top-k values are freed)."""
        M = int(f.shape[-2])
        scores = self._scores(f, mc)
        biased = ttnn.add(scores, self.bias, memory_config=mc)
        pad = self._pads.get(M)
        if pad is not None:  # power-of-two width -> multi-core topk (pads are -inf: never selected)
            wide = ttnn.concat([biased, pad], dim=-1, memory_config=mc)
            vals, idx = ttnn.topk(wide, k=self.top_k, dim=-1, largest=True, sorted=self.topk_sorted, memory_config=mc)
            _free(wide)
        else:
            vals, idx = ttnn.topk(biased, k=self.top_k, dim=-1, largest=True, sorted=self.topk_sorted, memory_config=mc)
        _free(vals)
        return scores, biased, idx

    def route_fused(self, f, kernel, *, scale: Optional[float] = None, taps: Optional[dict] = None,
                    memory_config=None):
        """B4 fused router tail (docs/OPTIMIZATION_PLAN.md §3.3 "A5 and B4"; ``router_mask="fused"``): this chip's
        routing weights ``w_loc [1, E_loc, M, 1]`` fp32 from the router matmul (:meth:`_scores`, unchanged) and one
        ``generic_op`` (``kernel``, :class:`~models.demos.motif3.tt.kernels.router_topk.FusedRouterTopK`) that adds
        the bias on the fp32 SFPU (the op ``ttnn.add`` runs), selects the top-8 by exact fp32 compares (exact ties: the
        lower expert id), normalizes over the 8 unbiased scores in fp32 and writes this chip's weights -- in place of
        add, concat, topk and the gather / scatter mask ops. Decode row counts only. Same top-8 sets as
        :meth:`__call__` except on exact fp32 ties at the 8th value; weights within fp32 rounding of the gather path
        (not bitwise equal). ``taps`` (tests, eager) receives ``idx`` (``[1, 1, M, 8]`` uint32 ROW_MAJOR, rank order)
        and ``scores`` (not freed)."""
        mc = memory_config or self.dram
        scale = self.route_scale if scale is None else float(scale)
        scores = self._scores(f, mc)
        if taps is not None:
            w_loc, idx = kernel(scores, scale=scale, memory_config=mc, want_idx=True)
            taps["idx"], taps["scores"] = idx, scores
        else:
            w_loc = kernel(scores, scale=scale, memory_config=mc)
            _free(scores)
        return w_loc

    def route_local(
        self,
        f,
        local_mask,
        consts: Tuple[object, object],
        *,
        scale: Optional[float] = None,
        taps: Optional[dict] = None,
        memory_config=None,
    ):
        """A5 "scatter" router mask (docs/OPTIMIZATION_PLAN.md §3.3 A5; probe logs/opt/phaseA/A5): this chip's routing
        weights ``w_loc [1, 12, M, 1]`` fp32 straight from the router, without ``ttnn.gather`` and the idx-based local
        mask (:meth:`MotifMoE.local_weights`). Decode shapes only (``consts`` exist per gathered row count).

        The head (linear + sigmoid, bias, padded top-k) is :meth:`__call__`'s. Then
        ``sel = to_layout(scatter(zeros [1,1,M,384], -1, idx, ones [1,1,M,8]))`` is today's top-8 set as a 0/1 mask
        (built from topk's own indices, so exact fp32 ties select what topk selected: a plain ``biased >= 8th value``
        threshold selects 9 experts on such a tie, 1 in 32,681 real token-layers), ``ws = scores * sel`` (fp32),
        ``den = sum(ws)`` over the 384 columns, ``w_loc = sum(ws * local_mask) * 1 / (den + 1e-20)`` with
        ``local_mask [1, 12, 1, 384]`` the one-hot of this chip's 12 experts (``weights.local_expert_mask``), times
        ``scale`` unless it is 1.0. Same top-8 sets as :meth:`__call__` + ``local_weights``; the weights differ from
        them by 1-4 fp32 ulp on ~27 % of rows (the normalizing sum runs over 384 columns, not the 8 gathered ones), so
        decode outputs are not bitwise equal to the gather path (MoE output <= 1 bf16 ulp, PCC 0.9999999998 on real
        L2 tokens). Rows stay independent: T64 rows equal T32 rows bitwise, reruns and trace replays are bitwise.
        Measured (traced, L1): router + local mask 140.1 -> 103.9 us at M = 32, 166.9 -> 131.9 us at M = 64; -1.7 to
        -2.0 ms per 53-layer decode step.

        ``taps`` (tests, eager) receives ``idx`` and ``sel`` (not freed)."""
        mc = memory_config or self.dram
        scale = self.route_scale if scale is None else float(scale)
        zeros, ones = consts
        scores, biased, idx = self._scores_topk(f, mc)
        _free(biased)
        m_rm = ttnn.scatter(zeros, -1, idx, ones, memory_config=mc)  # bf16 ROW_MAJOR (scatter refuses fp32 TILE)
        if taps is not None:
            taps["idx"] = idx
        else:
            _free(idx)
        sel = ttnn.to_layout(m_rm, ttnn.TILE_LAYOUT, memory_config=mc)
        _free(m_rm)
        ws = ttnn.multiply(scores, sel, dtype=ttnn.float32, memory_config=mc)  # A = fp32 scores
        if taps is not None:
            taps["sel"] = sel
        else:
            _free(sel)
        _free(scores)
        den = ttnn.sum(ws, dim=-1, keepdim=True, compute_kernel_config=self.ckc_eltwise, memory_config=mc)
        wl = ttnn.multiply(ws, local_mask, memory_config=mc)  # [1, 12, M, 384]
        _free(ws)
        wu = ttnn.sum(wl, dim=-1, keepdim=True, compute_kernel_config=self.ckc_eltwise, memory_config=mc)
        _free(wl)
        w_loc = ttnn.multiply(
            wu,
            den,
            input_tensor_b_activations=[
                ttnn.UnaryWithParam(ttnn.UnaryOpType.ADD_UNARY_SFPU, 1e-20),
                ttnn.UnaryWithParam(ttnn.UnaryOpType.RECIP),
            ],
            memory_config=mc,
        )
        _free(wu, den)
        if scale != 1.0:
            t = ttnn.multiply(w_loc, scale, memory_config=mc)
            _free(w_loc)
            w_loc = t
        return w_loc

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
            accumulation on the SFPU, ~45-50 us instead of ~14 us; at the 32-lane step's 32 and the T64 step's 64
            gathered rows, :data:`EXACT_ROUTER_DECODE_ROWS`; prefill keeps the composite).
        prefill_pc: in1-multicast program configs for the prefill expert matmuls when the chunk has >= 2048 rows
            (:func:`prefill_experts_pc`; 2x faster than the auto config, bitwise identical); False = auto config.
        prefill_chunk: rows per masked-dense prefill chunk (default ``cfg.moe_prefill_chunk`` = 4096).
        router_mask: decode routing weights (A5; ``None`` = ``cfg.router_mask``, ``MOTIF3_ROUTER_MASK``): "gather"
            (the release: router ``(idx, w)`` + :meth:`local_weights`) | "scatter" (:meth:`MotifRouter.route_local`:
            no ``ttnn.gather``, same top-8 sets, weights within 1-4 fp32 ulp; about -35 us per layer). Prefill always
            takes the gather path. "scatter" builds its constants here (the local one-hot ``[1, 12, 1, 384]`` fp32 and,
            per decode row count, the scatter's zeros / ones), before any trace capture (F3N rule R3). "fused" (B4):
            :meth:`MotifRouter.route_fused`, the router matmul + one ``generic_op``
            (:class:`~models.demos.motif3.tt.kernels.router_topk.FusedRouterTopK`: bias, top-8, normalize, local
            extraction) at the decode row counts; same top-8 sets except on exact fp32 ties at the 8th value (lower id
            wins), weights within fp32 rounding of the gather path; no device constants.
        moe_polynorm: decode routed-expert PolyNorm (B3; ``None`` = ``cfg.moe_polynorm``, ``MOTIF3_MOE_POLYNORM``):
            "composite" (the release, ``polynorm_impl``) | "fused" (one ``generic_op``,
            :class:`~models.demos.motif3.tt.kernels.moe_polynorm.FusedGroupedPolyNorm`, at the decode row counts 32 /
            64 with the routing weights folded in; not bitwise equal to the composite, ~-100 us per layer at M = 32;
            :func:`resolve_moe_polynorm`; a chip layout the kernel cannot place, e.g. 48 experts per chip on a (1, 8)
            submesh, raises when explicit and falls back to "composite" from the config). Prefill and other row counts
            keep the composite. No device constants beyond
            the layer's PolyNorm constants; the program compiles on the first eager decode call (before any capture).
        prefill_moe_dispatch / prefill_moe_combine: the compacted prefill's row builder and combine (B2b; ``None`` =
            ``cfg.prefill_moe_dispatch`` / ``cfg.prefill_moe_combine``): "host" (B2a: blocking read of the routes, numpy
            lists, one upload) | "device" (one ``generic_op`` builds the same rows from the routes; the host reads only
            the 32-byte block count); "matmul" (B2a: one-hot ``P^T @ y``) | "gather" (each token's rows added in an
            fp32 dest, :class:`~models.demos.motif3.tt.kernels.moe_compact.GatherCombine`). Every combination is
            bitwise equal to the dense path.
        decode_experts: decode routed experts (B1; ``None`` = ``cfg.decode_experts``, ``MOTIF3_DECODE_EXPERTS``):
            "dense" (the release) | "sparse" (``ttnn.sparse_matmul`` skips the local experts no live row routes to;
            live rows bitwise equal to "dense"; needs ``combine_mode="fold"``). Prefill always runs masked dense. No
            constants: the sparsity tensor is built per call, the lane mask per step (:meth:`decode_lane_mask`).
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
        router_mask: Optional[str] = None,
        decode_experts: Optional[str] = None,
        moe_polynorm: Optional[str] = None,
        prefill_moe: Optional[str] = None,
        prefill_moe_dispatch: Optional[str] = None,
        prefill_moe_combine: Optional[str] = None,
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
        # T64 decode (M = 64, 16 rows per DP row): per_core_M 2, gate_up on the same 10 x 8 cores, down on 8 x 8
        # (model_config.experts_*_pc(m_tiles=2) = the G16-lite configs); configs only, no device memory
        self.pc_wide = {
            m: (cfg.experts_gate_up_pc(m_tiles=m // TILE), cfg.experts_down_pc(m_tiles=m // TILE))
            for m in DECODE_ROWS[1:]
        }
        # gathered decode row counts forward_decode serves: 32, and 64 when cfg stages the T64 step (the router then
        # holds its 64-row top-k pad, allocated above)
        wide = wide_decode_rows(cfg)
        self.decode_rows: Tuple[int, ...] = (TILE,) + ((wide,) if wide else ())
        self.dram = ttnn.DRAM_MEMORY_CONFIG

        # ---- A5 router mask (decode only): constants before any trace capture (F3N rule R3) -------------------------
        self.router_mask = str(router_mask if router_mask is not None else getattr(cfg, "router_mask", "gather"))
        if self.router_mask not in ROUTER_MASK_MODES:
            raise ValueError(f"router_mask must be one of {ROUTER_MASK_MODES}, got {self.router_mask!r}")
        self.local_mask = None  # [1, 12, 1, 384] fp32 one-hot of this chip's experts
        self.scatter_consts: Dict[int, Tuple[object, object]] = {}  # M -> (zeros [1,1,M,384], ones [1,1,M,8]) bf16 RM
        if self.router_mask == "scatter":
            import torch

            self.local_mask = as_t(lambda: W.local_expert_mask(cfg), ttnn.float32, None, **ep)

            def rm(t):
                return ttnn.from_torch(
                    t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )  # fmt: skip

            for m in self.decode_rows:
                self.scatter_consts[m] = (rm(torch.zeros(1, 1, m, self.n_experts)), rm(torch.ones(1, 1, m, self.top_k)))

        # ---- B4 fused router tail (decode only): program descriptors only, no device memory (reads router.bias and
        # local_ids); the program compiles on the first eager decode call, before any capture
        self.router_fused = None
        if self.router_mask == "fused":
            from .kernels.router_topk import FusedRouterTopK  # lazy: generic_op kernel, decode shapes only

            self.router_fused = FusedRouterTopK(mesh_device, self.router.bias, self.local_ids, top_k=self.top_k)

        # ---- B1 decode experts: "dense" | "sparse" (no device constants) ----------------------------------------------
        self.decode_experts = resolve_decode_experts(decode_experts, cfg, self.combine_mode)

        # ---- B3 fused decode PolyNorm: "composite" | "fused" (program descriptors only; no device memory) ---------------
        self.moe_polynorm = resolve_moe_polynorm(moe_polynorm, cfg, decode_polynorm=self.decode_polynorm,
                                                 combine_mode=self.combine_mode, gate_up_dtype=self.gate_up_dtype)
        self.pn_fused = None
        if self.moe_polynorm == "fused":
            from .kernels.moe_polynorm import FusedGroupedPolyNorm  # lazy: generic_op kernel, decode shapes only

            try:
                self.pn_fused = FusedGroupedPolyNorm(mesh_device, self.pn_consts, e_loc=self.e_loc, inter=self.inter)
            except ValueError:
                # a layout the kernel cannot serve (e.g. 48 experts per chip on a (1, 8) submesh: 192 workers > the
                # grid): an explicit request raises, the config default falls back to the composite
                if moe_polynorm is not None:
                    raise
                self.moe_polynorm = "composite"

        # ---- B2a compacted prefill experts: "dense" | "compact" (host copies read lazily; no device constants) ----------
        self.prefill_moe = resolve_prefill_moe(prefill_moe, cfg, combine_mode=self.combine_mode,
                                               prefill_polynorm=self.prefill_polynorm,
                                               prefill_polynorm_impl=self.prefill_polynorm_impl)
        self.prefill_moe_block = str(getattr(cfg, "prefill_moe_block", "auto"))
        self.prefill_moe_min_rows = int(getattr(cfg, "prefill_moe_min_rows", 1024))
        self.prefill_moe_dispatch, self.prefill_moe_combine = resolve_prefill_moe_kernels(
            prefill_moe_dispatch, prefill_moe_combine, cfg)
        self._disp_meta = None  # B2b: this layer's per-chip dispatch constants (ids + PolyNorm bits), device
        self.compact_state = CompactPrefillState(owner=self)  # MotifModel hands every layer one shared state
        self._pn_host = None  # [P, 12, 4] fp32 (bf16 values): this layer's c0, c1, c2, b per chip
        self._pn_bits = None  # the same as int32 bf16 bit patterns (numpy), for compact_upload_fast

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
        is already weighted). Decode (M = 32) uses the G6 1D-multicast program configs, decode M = 64 (the T64 step)
        their ``per_core_M`` 2 twins (``pc_wide``); prefill (M = chunk) the in1-multicast :func:`prefill_experts_pc`
        when M splits over its 64 cores (>= 2048 rows), else the op's auto config. Intermediates and ``y`` in
        ``memory_config`` (decode: L1; prefill: DRAM)."""
        mc = memory_config or self.dram
        M = int(f.shape[-2])
        x12 = ttnn.repeat(f, ttnn.Shape([1, self.e_loc, 1, 1]), memory_config=mc)
        gu_dtype = self.gate_up_dtype
        if gu_dtype is None:
            gu_dtype = ttnn.float32 if polynorm == "fp32" else ttnn.bfloat16
        if decode and M == TILE:
            pc_gu, pc_dn = self.pc_gate_up, self.pc_down
        elif decode and M in self.pc_wide:
            pc_gu, pc_dn = self.pc_wide[M]
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
        h = self._decode_fused(gu, row_scale, mc) if decode else None
        if h is None:
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

    def _decode_fused(self, gu, row_scale, mc):
        """B3: ``h`` from the fused kernel when this module runs it (``moe_polynorm="fused"``), the call is at a decode
        row count with routing weights and ``gu`` / ``row_scale`` meet its contract (fp32 TILE interleaved); else
        ``None`` (the caller runs the composite)."""
        fused = getattr(self, "pn_fused", None)
        if fused is None or row_scale is None or int(gu.shape[-2]) not in self.decode_rows:
            return None
        if not fused.supports(gu, row_scale):
            return None
        return fused(gu, row_scale, memory_config=mc)

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

    def local_partial(self, f, *, polynorm: str, decode: bool, taps: Optional[dict] = None, memory_config=None,
                      lane_mask=None):
        """This chip's routed partial for the tokens ``f [1, 1, M, 4096]`` (identical on all chips): route, mask,
        experts, combine -> ``[1, 1, M, 4096]`` in ``combine_dtype``, DRAM (still to be summed over all 32 chips).
        ``memory_config``: intermediates (decode: L1). ``taps`` (tests) receives ``idx``, ``w`` (unscaled when
        ``route_scale`` is folded: multiply by ``route_scale / internal_route_scale`` to compare), ``w_loc``; with
        ``router_mask="scatter"`` at a decode row count ``idx``, ``sel`` (the top-8 0/1 mask) and ``w_loc`` instead; with
        ``router_mask="fused"`` (B4) at a decode row count ``idx`` (ROW_MAJOR, rank order), ``scores`` and ``w_loc``.

        B1 (``decode_experts="sparse"``, decode row counts only): ``lane_mask`` (``[1, 1, M, 1]`` fp32 0/1, the
        step's live rows in gathered order; :meth:`decode_lane_mask`) zeroes the routing weights of inactive rows
        first (``None``: no masking -- still exact on every row, only fewer experts are skipped), then
        :meth:`sparse_experts` skips the local experts with no routed row. ``taps`` then also get ``sparsity`` and
        ``w_loc`` is the masked one. ``lane_mask`` is ignored on the dense path (it is the caller's; never freed)."""
        mc = memory_config or self.dram
        M = int(f.shape[-2])
        consts = getattr(self, "scatter_consts", {}).get(M) if decode else None
        fused = getattr(self, "router_fused", None) if decode and M in getattr(self, "decode_rows", ()) else None
        if fused is not None:  # B4 fused router tail at a decode row count (taps get idx, scores, w_loc; no "w")
            w_loc = self.router.route_fused(f, fused, scale=self.internal_route_scale, taps=taps, memory_config=mc)
        elif consts is not None:  # A5 "scatter" router mask at a decode row count (taps get idx, sel, w_loc; no "w")
            w_loc = self.router.route_local(
                f, self.local_mask, consts, scale=self.internal_route_scale, taps=taps, memory_config=mc
            )
        else:
            idx, w = self.router(f, scale=self.internal_route_scale, memory_config=mc)
            if not decode and taps is None and self.compact_applies(M):
                part = self._compact_partial(f, idx, w)  # None: the dense path below (beyond the cap / not warmed)
                if part is not None:
                    _free(idx, w)
                    return part
            w_loc = self.local_weights(idx, w, memory_config=mc)
            if taps is not None:
                taps["idx"], taps["w"] = idx, w
            else:
                _free(idx, w)
        sparse = decode and getattr(self, "decode_experts", "dense") == "sparse" and M in self.decode_rows
        if sparse and lane_mask is not None:
            wm = ttnn.multiply(w_loc, lane_mask, memory_config=mc)  # [1, 12, M, 1] * [1, 1, M, 1]: exact (x 0 / x 1)
            _free(w_loc)
            w_loc = wm
        if taps is not None:
            taps["w_loc"] = w_loc
        if sparse:
            s = self.decode_sparsity(w_loc, memory_config=mc)
            if taps is not None:
                taps["sparsity"] = s
            y = self.sparse_experts(f, s, polynorm=polynorm, row_scale=w_loc, memory_config=mc)
            if taps is None:
                _free(s)
            part = self.reduce_experts(y, memory_config=self.dram)
        elif self.combine_mode == "fold":
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
    # B2a: token-compacted prefill experts
    # ==========================================================================================================
    def compact_applies(self, rows: int) -> bool:
        """This module runs a prefill chunk of ``rows`` rows compacted (``prefill_moe="compact"`` and ``rows >=
        prefill_moe_min_rows``)."""
        return getattr(self, "prefill_moe", "dense") == "compact" and int(rows) >= int(self.prefill_moe_min_rows)

    def _mesh_rc(self) -> Tuple[int, int]:
        R, C = (int(v) for v in tuple(self.mesh_device.shape))
        return R, C

    def prepare_compact(self) -> None:
        """Host copies the compacted path needs, read once (eager; the generator's warm-up reads every layer's before
        the decode capture): the chips' local expert ids (shared by all layers) and this layer's bf16 PolyNorm
        constants ``[P, 12, 4]``; plus the per-chip upload mapper."""
        import torch

        st = self.compact_state
        R, C = self._mesh_rc()
        if st.mapper is None:
            st.mapper = ttnn.create_mesh_mapper(
                self.mesh_device,
                ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], ttnn.MeshShape(R, C)),
            )
        if st.local_ids is None:
            ids = device_tensors_to_torch(self.local_ids, self.mesh_device).float().reshape(R * C, self.e_loc)
            st.local_ids = ids.round().long()
        if st.g2s is None:
            st.g2s = _global_to_slot(st.local_ids).numpy()
        if self._pn_host is None:
            c = self.pn_consts.c["bf16"]
            self._pn_host = torch.stack(
                [device_tensors_to_torch(c[k], self.mesh_device).float().reshape(R * C, self.e_loc)
                 for k in ("c0", "c1", "c2", "b")], dim=-1)  # fmt: skip
        if getattr(self, "_pn_bits", None) is None:
            self._pn_bits = (self._pn_host.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF).numpy()
        if getattr(self, "prefill_moe_dispatch", "host") == "device":
            from .kernels.moe_compact import CompactDispatch

            if st.dispatch is None:
                st.dispatch = CompactDispatch(self.mesh_device, st.local_ids, mapper=st.mapper, top_k=self.top_k,
                                              memory_config=self.dram)
            if self._disp_meta is None:
                self._disp_meta = st.dispatch.make_meta(self._pn_bits)
        if getattr(self, "prefill_moe_combine", "matmul") == "gather" and st.combine is None:
            from .kernels.moe_compact import GatherCombine

            st.combine = GatherCombine(self.mesh_device, top_k=self.top_k)

    def _iota(self, rows: int):
        """The combine's token-index column ``[1, 1, rows, 1]`` fp32 (0 .. rows-1), one per chunk size (shared)."""
        import torch

        st = self.compact_state
        t = st.iota.get(int(rows))
        if t is None:
            t = ttnn.from_torch(
                torch.arange(int(rows), dtype=torch.float32).reshape(1, 1, int(rows), 1), dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT, device=self.mesh_device, memory_config=self.dram,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )  # fmt: skip
            st.iota[int(rows)] = t
        return t

    def _read_routes(self, idx, rows: int):
        """``idx`` of a prefill chunk -> host ``[rows, 8]`` int64 numpy of chip 0 (the routes are identical on every
        chip): one blocking copy into a host staging tensor allocated once per spec. The routing weights stay on device
        (the compacted rows gather them from ``w_loc``)."""
        import torch

        st = self.compact_state
        key = (tuple(idx.shape), idx.dtype, idx.layout)
        h = st.host_bufs.get(key)
        if h is None:
            h = st.host_bufs[key] = ttnn.allocate_tensor_on_host(idx.spec, self.mesh_device)
        ttnn.copy_device_to_host_tensor(idx, h, blocking=True)
        hi = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).reshape(-1, self.top_k)[:rows]
        return hi.to(torch.int64).numpy()

    def _compact_partial(self, f, idx, w):
        """``local_partial`` of a prefill chunk, compacted (B2a): read the routes from chip 0 (they are identical on
        every chip), build each chip's row lists on the host (:func:`compact_prefill_meta`), run
        :meth:`_compact_device`. Returns None when the busiest chip needs more blocks than the ladder's cap, or when the
        shape was not compiled by :meth:`warm_compact` and the state requires it: the caller runs the dense path on the
        same routes (bitwise the same result)."""
        st = self.compact_state
        M = int(f.shape[-2])
        mb = compact_block(M, self.prefill_moe_block)
        self.prepare_compact()
        if getattr(self, "prefill_moe_dispatch", "host") == "device" and M <= PREFILL_MOE_DISPATCH_MAX_ROWS:
            return self._compact_partial_device(f, idx, w, M, mb)
        hi = self._read_routes(idx, M)
        R, C = self._mesh_rc()
        need, nb, u = compact_upload_fast(hi, st.g2s, R * C, self.e_loc, mb, compact_ladder(M, mb, self.e_loc),
                                          self._pn_bits, M)
        if nb is None:
            st.stats["dense_cap"] += 1
            return None
        if not st.allows(M, mb, nb) or (st.frozen() and M not in st.iota):
            st.stats["dense_unwarmed"] += 1
            return None
        part = self._compact_device(f, idx, w, u, mb, nb)
        st.stats["compact"] += 1
        st.blocks[(M, mb, nb)] = st.blocks.get((M, mb, nb), 0) + 1
        return part

    def _read_need(self, need) -> Tuple[int, int]:
        """The dispatch kernel's ``need [1, 1, 1, 8]`` -> ``(need, NB)`` of chip 0 (every chip computes the same):
        one blocking copy into a host staging tensor allocated once."""
        st = self.compact_state
        key = (tuple(need.shape), need.dtype, need.layout)
        h = st.host_bufs.get(key)
        if h is None:
            h = st.host_bufs[key] = ttnn.allocate_tensor_on_host(need.spec, self.mesh_device)
        ttnn.copy_device_to_host_tensor(need, h, blocking=True)
        v = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).reshape(-1)
        return int(v[0]), int(v[1])

    def _compact_partial_device(self, f, idx, w, M: int, mb: int):
        """B2b: ``_compact_partial`` with the rows built on device (:class:`CompactDispatch` from ``idx`` and the
        router's ``w``, whose routed values are bitwise this chip's ``w_loc`` (``local_weights`` adds exact zeros to
        one product by 1.0; checked on device, logs/opt/phaseB/B2b), then the host reads only the block count NB.
        None (the dense path on the same routes) beyond the ladder's cap or for an unwarmed shape, as B2a."""
        st = self.compact_state
        ladder = compact_ladder(M, mb, self.e_loc)
        if st.frozen() and not any(k[0] == M and k[1] == mb for k in st.warmed):
            # a chunk size the warm-up did not compile: its dispatch program would compile after the decode capture
            # (F3N rule R2) -- the dense path, before any device work (B2a decides the same after its host pass)
            st.stats["dense_unwarmed"] += 1
            return None
        rows = st.dispatch(idx, w, self._disp_meta, M=M, mb=mb, ladder=ladder, w_is_loc=False)
        need, nb = self._read_need(rows.need)
        if nb == 0:
            rows.free()
            st.stats["dense_cap"] += 1
            return None
        if not st.allows(M, mb, nb) or (st.frozen() and getattr(self, "prefill_moe_combine", "matmul") == "matmul"
                                            and M not in st.iota):
            rows.free()
            st.stats["dense_unwarmed"] += 1
            return None
        part = self._compact_rows_device(f, rows, mb, nb)
        rows.free()
        st.stats["compact"] += 1
        st.stats["device_dispatch"] += 1
        st.blocks[(M, mb, nb)] = st.blocks.get((M, mb, nb), 0) + 1
        return part

    def _compact_rows_device(self, f, rows, mb: int, nb: int):
        """Device half of the compacted prefill experts from the dispatch kernel's capacity buffers ``rows``
        (:class:`CompactRows`, not consumed): slice the first ``nb`` blocks, then :meth:`_compact_experts` -- the same
        ops on the same values as :meth:`_compact_device` (B2a's upload) -- bitwise equal to the dense path."""
        M = int(f.shape[-2])
        E = self.e_loc
        R = int(nb) * int(mb)
        dram = self.dram
        tix = ttnn.slice(rows.rows, [0, 0, 0, 0], [1, 1, 1, R], memory_config=dram)  # uint32 ROW_MAJOR
        x_rm = ttnn.to_layout(f, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        X = ttnn.embedding(tix, x_rm, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        _free(x_rm, tix)
        X = _reshape(X, (1, nb, mb, self.hidden))
        sp = ttnn.slice(rows.sp, [0, 0, 0, 0], [1, nb, 1, E], memory_config=dram)  # [1, nb, 1, 12] bf16 ROW_MAJOR
        cblk = {k: ttnn.slice(rows.blk, [0, 0, 0, 16 + i], [1, nb, 1, 17 + i], memory_config=dram)
                for i, k in enumerate(("c0", "c1", "c2", "b"))}  # fmt: skip
        wcol = ttnn.slice(rows.wcol, [0, 0, 0, 0], [1, nb, mb, 1], memory_config=dram)
        y = self._compact_experts(X, sp, cblk, wcol, mb, nb)  # consumes X, sp, cblk, wcol
        return self._compact_combine(y, M, keys=rows.rows, key_page=1)

    def _compact_experts(self, X, sp, cblk, wcol, mb: int, nb: int):
        """The compacted experts on ``X [1, nb, mb, 4096]`` (consumes ``X``, ``sp``, ``cblk``, ``wcol``) -> ``y [1, 1,
        nb mb, 4096]``: gate_up with ``ttnn.sparse_matmul`` (one expert per block, ``nnz = nb``) at the decode experts'
        program configs for ``mb`` rows; the grouped PolyNorm with per-block constants and the routing weights folded into
        ``up``; down with ``sparse_matmul``."""
        H, I = self.hidden, self.inter
        dram = self.dram
        m_tiles = int(mb) // TILE
        pc_gu = self.cfg.experts_gate_up_pc(m_tiles=m_tiles)
        pc_dn = self.cfg.experts_down_pc(m_tiles=m_tiles)
        gu_dtype = self.gate_up_dtype or ttnn.bfloat16
        gu = ttnn.allocate_tensor_on_device(ttnn.Shape([1, nb, mb, 2 * I]), gu_dtype, ttnn.TILE_LAYOUT,
                                            self.mesh_device, dram)
        gu = ttnn.sparse_matmul(
            X, self.w_gate_up, sparsity=sp, nnz=int(nb), is_input_a_sparse=False, is_input_b_sparse=True,
            program_config=pc_gu, compute_kernel_config=self.ckc_experts, dtype=gu_dtype, memory_config=dram,
            optional_output_tensor=gu,
        )  # fmt: skip
        _free(X)
        g = ttnn.slice(gu, [0, 0, 0, 0], [1, nb, mb, I], memory_config=dram)
        up = ttnn.slice(gu, [0, 0, 0, I], [1, nb, mb, 2 * I], memory_config=dram)
        _free(gu)
        us = ttnn.multiply(wcol, up, memory_config=dram)  # A = w (fp32): fp32 product, broadcast over cols
        _free(up, wcol)
        h = _pn.grouped_polynorm(g, cblk, inter=I, mode="bf16", eps=self.cfg.polynorm_eps,
                                 compute_kernel_config=self.ckc_polynorm, memory_config=dram, up=us, impl="rms",
                                 intermediate_memory_config=dram)
        _free(g, us, *cblk.values())
        y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, nb, mb, H]), self.down_dtype, ttnn.TILE_LAYOUT,
                                           self.mesh_device, dram)
        y = ttnn.sparse_matmul(
            h, self.w_down, sparsity=sp, nnz=int(nb), is_input_a_sparse=False, is_input_b_sparse=True,
            program_config=pc_dn, compute_kernel_config=self.ckc_experts, dtype=self.down_dtype, memory_config=dram,
            optional_output_tensor=y,
        )  # fmt: skip
        _free(h, sp)
        return _reshape(y, (1, 1, int(nb) * int(mb), H))

    def _compact_combine(self, y, M: int, *, keys, key_page: int, keys_f32=None):
        """``y [1, 1, R, 4096]`` (consumed) -> this chip's partial ``[1, 1, M, 4096]`` in ``combine_dtype``. ``keys``:
        a uint32 ROW_MAJOR tensor whose page ``key_page`` holds every row's token (pad rows M; not consumed).
        "gather" (B2b): :class:`GatherCombine` on ``y`` untilized. "matmul" (B2a): ``P^T [M, R] @ y`` with ``P^T[t, j] =
        (key[j] == t)`` (``keys_f32 [1, 1, 1, R]`` fp32 TILE when the caller has it, consumed; else built from
        ``keys``)."""
        dram = self.dram
        R = int(y.shape[-2])
        if getattr(self, "prefill_moe_combine", "matmul") == "gather":
            y_rm = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
            _free(y)
            part = self.compact_state.combine(y_rm, keys, M=M, key_page=key_page, out_dtype=self.combine_dtype,
                                              memory_config=dram)
            _free(y_rm, keys_f32)
            return part
        if keys_f32 is None:
            k_rm = ttnn.slice(keys, [0, 0, key_page, 0], [1, 1, key_page + 1, R], memory_config=dram)
            k_t = ttnn.to_layout(k_rm, ttnn.TILE_LAYOUT, memory_config=dram)
            _free(k_rm)
            keys_f32 = ttnn.typecast(k_t, ttnn.float32, memory_config=dram)  # exact: integers <= M
            _free(k_t)
        PT = ttnn.eq(keys_f32, self._iota(M), dtype=ttnn.bfloat16, memory_config=dram)
        _free(keys_f32)
        part = ttnn.matmul(PT, y, compute_kernel_config=self.ckc_experts, dtype=self.combine_dtype,
                           memory_config=dram)
        _free(PT, y)
        return part

    @staticmethod
    def compact_upload_rows(meta: Dict[str, object], rows_m: int, mb: int, nb: int):
        """The single per-chip upload of :meth:`_compact_device` from :func:`compact_prefill_meta`'s lists (the
        reference packer; serving uses :func:`compact_upload_fast`): ``[P, 4, rows]`` int64 holding uint32 words below
        2^31: row 0 the row tokens (embedding indices; pad rows 0), row 1 the combine keys as integers (the token, or
        ``rows_m`` -- the chunk size M, which no token equals -- for pad rows), row 2 the routing-weight gather index
        ``e M + t`` into this chip's ``w_loc`` (pad rows 0), row 3 per block ``[sparsity (12) | 0 (4) | c0 c1 c2 b |
        0 (12)]`` as bf16 bit patterns (< 2^16; 32 words per block, ``nb`` blocks; the rest 0). Only integers and bf16
        bit patterns travel: ``ttnn.bitcast`` to fp32 is not exact on device (its unpack truncates the mantissa), the
        uint32 -> uint16 typecast and the 16-bit bitcast are."""
        import torch

        tok = meta["tok"]
        P, rows = (int(v) for v in tok.shape)
        E = int(meta["sparsity"].shape[-1])
        if rows != int(nb) * int(mb) or E > 16 or 32 * int(nb) > rows:
            raise ValueError(f"compact upload: {rows} rows for {nb} x {mb}, {E} experts")
        valid = meta["tokv"] >= 0
        erow = meta["eblk"].repeat_interleave(int(mb), dim=1)  # [P, rows] local expert of every row
        blk = torch.zeros(P, int(nb), 32, dtype=torch.float32)
        blk[:, :, :E] = meta["sparsity"]
        blk[:, :, 16:20] = meta["cblk"]
        b16 = blk.to(torch.bfloat16)
        if not torch.equal(b16.float(), blk):
            raise ValueError("compact upload: block constants must be bf16 values")
        u = torch.zeros(P, 4, rows, dtype=torch.int64)
        u[:, 0] = tok.to(torch.int64)
        u[:, 1] = torch.where(valid, meta["tokv"].to(torch.int64), torch.full_like(u[:, 1], int(rows_m)))
        u[:, 2] = torch.where(valid, erow * int(rows_m) + tok.to(torch.int64), torch.zeros_like(u[:, 2]))
        u[:, 3, : 32 * int(nb)] = b16.reshape(P, -1).view(torch.int16).to(torch.int64) & 0xFFFF
        return u

    def _compact_device(self, f, idx, w, u, mb: int, nb: int):
        """Device half of the compacted prefill experts for the chunk ``f [1, 1, M, 4096]`` (identical on all chips),
        its routes ``(idx, w)`` (the router's; not consumed) and the upload words ``u [P, 4, nb * mb]``
        (:func:`compact_upload_fast`, or :meth:`compact_upload_rows` of :func:`compact_prefill_meta`'s lists) -> this
        chip's partial ``[1, 1, M, 4096]`` in ``combine_dtype`` (DRAM), bitwise equal to the dense path's.

        One upload (per-chip shards, uint32 ROW_MAJOR ``[1, 1, 4, R]``; integer uploads are ~3x cheaper than fp32 ones),
        unpacked with exact ops only (uint32 tilize, slices, the integer keys typecast to fp32, the block row's bf16 bit
        patterns typecast to uint16 and ``bitcast`` to bf16). The routing weights of the rows are gathered from this
        chip's ``w_loc`` (the dense path's :meth:`local_weights`, transposed: an exact fp32 copy) at ``e M + t``.
        Then: gather the rows (``ttnn.embedding`` of the untilized chunk: an exact copy) -> ``[1, nb, mb, 4096]``; gate_up
        with ``ttnn.sparse_matmul`` (``b`` sparse, one expert per block: ``nnz = nb`` exactly, compact output) at the
        decode experts' program configs for ``mb`` rows; the grouped PolyNorm on the blocks with per-block constants and
        the routing weights folded into ``up`` (the dense path's ops on the same values); down with ``sparse_matmul``;
        combine ``P^T [M, R] @ y [R, 4096]`` with ``P^T[t, j] = (key[j] == t)`` (rows in expert order, so a token's
        expert terms add in the dense order; pad rows have key ``M``, which matches no token), or with
        ``prefill_moe_combine="gather"`` :class:`GatherCombine` on the same keys (B2b)."""
        import numpy as np
        import torch

        st = self.compact_state
        M = int(f.shape[-2])
        H, I, E = self.hidden, self.inter, self.e_loc
        R, C = self._mesh_rc()
        rows = int(nb) * int(mb)
        dram = self.dram

        if isinstance(u, np.ndarray):
            u = torch.from_numpy(u)
        if tuple(u.shape) != (R * C, 4, rows):
            raise ValueError(f"compact upload of shape {tuple(u.shape)}, want {(R * C, 4, rows)}")
        U = ttnn.from_torch(u.reshape(R, C, 4, rows).to(torch.int32), dtype=ttnn.uint32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device, memory_config=dram,
                            mesh_mapper=st.mapper)  # [1, 1, 4, rows] per chip
        # gather the routed rows
        tix = ttnn.slice(U, [0, 0, 0, 0], [1, 1, 1, rows], memory_config=dram)  # uint32 ROW_MAJOR
        x_rm = ttnn.to_layout(f, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        X = ttnn.embedding(tix, x_rm, layout=ttnn.TILE_LAYOUT, memory_config=dram)
        _free(x_rm, tix)
        X = _reshape(X, (1, nb, mb, H))
        # block sparsity / PolyNorm constants (bf16 bit patterns)
        blk = ttnn.slice(U, [0, 0, 3, 0], [1, 1, 4, 32 * nb], memory_config=dram)
        blk = _reshape(blk, (1, nb, 1, 32))
        bt = ttnn.to_layout(blk, ttnn.TILE_LAYOUT, memory_config=dram)
        _free(blk)
        b16 = ttnn.typecast(bt, ttnn.uint16, memory_config=dram)  # exact: every word < 2^16
        _free(bt)
        bb = ttnn.bitcast(b16, ttnn.bfloat16, memory_config=dram)
        _free(b16)
        cblk = {k: ttnn.slice(bb, [0, 0, 0, 16 + i], [1, nb, 1, 17 + i], memory_config=dram)
                for i, k in enumerate(("c0", "c1", "c2", "b"))}  # fmt: skip
        brm = ttnn.to_layout(bb, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        _free(bb)
        sp = ttnn.slice(brm, [0, 0, 0, 0], [1, nb, 1, E], memory_config=dram)  # [1, nb, 1, 12] bf16 ROW_MAJOR
        _free(brm)
        # combine keys (integers -> fp32; the matmul combine only) and the rows' routing weights (gathered from w_loc)
        gather = getattr(self, "prefill_moe_combine", "matmul") == "gather"
        Ut = ttnn.to_layout(U, ttnn.TILE_LAYOUT, memory_config=dram)
        keys = None
        if not gather:
            k32 = ttnn.slice(Ut, [0, 0, 1, 0], [1, 1, 2, rows], memory_config=dram)
            keys = ttnn.typecast(k32, ttnn.float32, memory_config=dram)  # exact: integers <= M
            _free(k32)
        gi = ttnn.slice(Ut, [0, 0, 2, 0], [1, 1, 3, rows], memory_config=dram)
        _free(Ut)
        w_loc = self.local_weights(idx, w, memory_config=dram)  # [1, 12, M, 1] fp32, as the dense path
        wT = ttnn.transpose(w_loc, -2, -1, memory_config=dram)  # [1, 12, 1, M] (exact)
        _free(w_loc)
        wT = _reshape(wT, (1, 1, 1, E * M))
        wrow = ttnn.gather(wT, -1, gi, memory_config=dram)  # [1, 1, 1, rows] fp32 (exact copy)
        _free(wT, gi)
        wcol = ttnn.transpose(wrow, -2, -1, memory_config=dram)  # [1, 1, rows, 1]
        _free(wrow)
        wcol = _reshape(wcol, (1, nb, mb, 1))

        y = self._compact_experts(X, sp, cblk, wcol, mb, nb)
        part = self._compact_combine(y, M, keys=U, key_page=1, keys_f32=keys)
        _free(U)
        return part

    def warm_compact(self, rows: int) -> Tuple[int, ...]:
        """Warm-up (before the decode capture): compile the prefill local partial of a ``rows``-row chunk on the dense
        path (the fallback) and on the compacted path at every block count of its ladder (synthetic row lists: pad rows
        only, so nothing is combined), and create the chunk size's combine column. Marks the shapes warmed in the shared
        state (every MoE layer runs the same programs). Returns the ladder."""
        import torch

        st = self.compact_state
        M = int(rows)
        mb = compact_block(M, self.prefill_moe_block)
        ladder = compact_ladder(M, mb, self.e_loc)
        self.prepare_compact()
        self._iota(M)
        R, C = self._mesh_rc()
        P = R * C
        x = ttnn.from_torch(
            torch.zeros(1, 1, M, self.hidden), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
            memory_config=self.dram, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )  # fmt: skip
        try:
            idx, w = self.router(x, scale=self.internal_route_scale, memory_config=self.dram)
            w_loc = self.local_weights(idx, w, memory_config=self.dram)
            y = self.experts(x, polynorm=self.prefill_polynorm, decode=False, row_scale=w_loc, memory_config=self.dram)
            _free(w_loc)
            _free(self.reduce_experts(y, memory_config=self.dram), y)
            if getattr(self, "prefill_moe_dispatch", "host") == "device" and M <= PREFILL_MOE_DISPATCH_MAX_ROWS:
                # the dispatch program (one per chunk size) on the warm routes, then the post-dispatch programs of
                # every ladder entry on synthetic capacity buffers (pad rows only: nothing is combined)
                rows_d = st.dispatch(idx, w, self._disp_meta, M=M, mb=mb, ladder=ladder, w_is_loc=False)
                self._read_need(rows_d.need)
                rows_d.free()
                for nb in ladder:
                    rows_s = self._synthetic_rows(M, mb, ladder[-1])
                    _free(self._compact_rows_device(x, rows_s, mb, int(nb)))
                    rows_s.free()
                    st.warmed.add((M, mb, int(nb)))
                _free(idx, w)
                return ladder
            for nb in ladder:
                eblk = (torch.arange(nb) % self.e_loc).unsqueeze(0).expand(P, nb)
                meta = dict(
                    tok=torch.zeros(P, nb * mb, dtype=torch.int32), tokv=torch.full((P, nb * mb), -1.0),
                    w=torch.zeros(P, nb * mb), eblk=eblk,
                    sparsity=torch.nn.functional.one_hot(eblk, self.e_loc).to(torch.float32),
                    cblk=torch.gather(self._pn_host, 1, eblk.unsqueeze(-1).expand(P, nb, 4)),
                )  # fmt: skip
                _free(self._compact_device(x, idx, w, self.compact_upload_rows(meta, M, mb, nb), mb, nb))
                st.warmed.add((M, mb, int(nb)))
            _free(idx, w)
        finally:
            _free(x)
        return ladder

    def _synthetic_rows(self, M: int, mb: int, cap: int):
        """Warm-up only: :class:`CompactRows` capacity buffers with the dispatch kernel's specs holding pad rows only
        (token 0, key M, weight 0; block b one-hot at expert ``b % 12`` with its PolyNorm constants of chip 0)."""
        import torch

        from .kernels.moe_compact import CompactRows

        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        dev, dram, E = self.mesh_device, self.dram, self.e_loc

        def up(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev, memory_config=dram, mesh_mapper=rep)

        rr = torch.zeros(1, 1, 2, cap * mb, dtype=torch.int32)
        rr[0, 0, 1] = int(M)
        e = torch.arange(cap) % E
        sp = torch.zeros(1, cap, 1, 32)
        sp[0, torch.arange(cap), 0, e] = 1.0
        blk = sp.clone()
        blk[0, :, 0, 16:20] = self._pn_host[0][e].float()
        return CompactRows(up(rr, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), up(torch.zeros(1, cap, mb, 1), ttnn.float32,
                           ttnn.TILE_LAYOUT), up(sp, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                           up(blk, ttnn.bfloat16, ttnn.TILE_LAYOUT), None)

    # ==========================================================================================================
    # B1: sparse decode experts
    # ==========================================================================================================
    def decode_sparsity(self, w_loc, *, memory_config=None):
        """``w_loc [1, 12, M, 1]`` fp32 TILE (lane-masked) -> the ``sparse_matmul`` sparsity ``[1, 1, 1, 12]`` bf16
        ROW_MAJOR (one stick): ``max`` over the rows (weights are >= 0, exactly 0 where no row routes), bf16, untilize,
        view. 4 ops, ~20 us traced per layer (M6). A weight too small for bf16 (below ~1e-38) would read as 0 and
        skip the expert; real routing weights are normalized top-8 sigmoid scores, orders of magnitude above that."""
        mc = memory_config or self.dram
        m = ttnn.max(w_loc, dim=2, keepdim=True, memory_config=mc)  # [1, 12, 1, 1] fp32
        b = ttnn.typecast(m, ttnn.bfloat16, memory_config=mc)
        _free(m)
        r = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        _free(b)
        return _reshape(r, (1, 1, 1, self.e_loc))

    def sparse_experts(self, f, sparsity, *, polynorm: str, row_scale, memory_config=None):
        """:meth:`experts` at a decode row count with the local experts whose ``sparsity`` entry is 0 skipped:
        ``f [1, 1, M, 4096]`` -> ``y [1, 12, M, 4096]`` (skipped experts' slices exactly 0). The same program configs,
        compute config and dtypes as the dense decode matmuls (bitwise equal on the computed slices; M6, and
        ``sparsity`` all ones == dense on every row). ``nnz=None`` always (a static count deadlocks on BH: tt-metal
        #45943 / #45052)."""
        mc = memory_config or self.dram
        M = int(f.shape[-2])
        gu_dtype = self.gate_up_dtype
        if gu_dtype is None:
            gu_dtype = ttnn.float32 if polynorm == "fp32" else ttnn.bfloat16
        pc_gu, pc_dn = (self.pc_gate_up, self.pc_down) if M == TILE else self.pc_wide[M]
        gu = ttnn.sparse_matmul(
            f, self.w_gate_up, sparsity=sparsity, program_config=pc_gu, nnz=None, is_input_a_sparse=False,
            is_input_b_sparse=True, memory_config=mc, compute_kernel_config=self.ckc_experts, dtype=gu_dtype,
        )  # [1, 1, 1, 12, M, 2560]
        gu = _reshape(gu, (1, self.e_loc, M, 2 * self.inter))
        h = self._decode_fused(gu, row_scale, mc)
        if h is None:
            h = self.polynorm(gu, mode=polynorm, row_scale=row_scale, memory_config=mc, impl=self.polynorm_impl)
        _free(gu)
        y = ttnn.sparse_matmul(
            h, self.w_down, sparsity=sparsity, program_config=pc_dn, nnz=None, is_input_a_sparse=True,
            is_input_b_sparse=True, memory_config=mc, compute_kernel_config=self.ckc_experts, dtype=self.down_dtype,
        )
        _free(h)
        if len(y.shape) != 4:
            y = _reshape(y, (1, self.e_loc, M, self.hidden))
        return y

    @staticmethod
    def decode_lane_mask(ccl: MotifCCL, active, rows_per_dp: int):
        """The step's MoE lane mask for B1: the live rows of all DP rows in the MoE's gathered order (row ``L dp + j``,
        L = ``rows_per_dp``: 8 lanes, or 16 rows in the T64 step), ``[1, 1, 4 L, 1]`` fp32 0/1 DRAM, identical on every
        chip. From the step's ``active`` mask (``MotifAttention.active_mask_from_cur_pos(cur_pos, L)``,
        ``[1, 1, L, width]`` bf16 TILE, width >= 32; not consumed): its first 32 columns gathered over DP. Build it
        once per step (trace-safe: fixed shapes, ~37 us) and pass it to every MoE layer's :meth:`forward_decode`;
        the caller frees it."""
        L = int(rows_per_dp)
        a32 = ttnn.slice(active, [0, 0, 0, 0], [1, 1, L, TILE], memory_config=ttnn.L1_MEMORY_CONFIG)
        g = ccl.ag_dp_rows(a32, memory_config=ttnn.L1_MEMORY_CONFIG)  # [1, 1, 4 L, 32] bf16 TILE
        if g is not a32:
            _free(a32)
        M = int(g.shape[-2])
        c = ttnn.slice(g, [0, 0, 0, 0], [1, 1, M, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        _free(g)
        m = ttnn.typecast(c, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _free(c)
        return m

    # ==========================================================================================================
    # decode (MOE-1, MOE-5)
    # ==========================================================================================================
    def forward_decode(self, x, *, add_partial=None, reduce_tp: bool = True, taps: Optional[dict] = None,
                       lane_mask=None):
        """Decode MoE for this DP row's rows: its 8 lanes, or the T64 verify step's 16 rows (``[8 anchors | 8
        drafts]``, still one tile row; docs/p5_t64/P5_T64_DESIGN.md §4.2).

        Args:
            x: ``[1, 1, L, 4096]`` bf16 TILE, L = 8 (or 16 when the module was built with a T64 config:
                ``self.decode_rows`` holds 64 = 4 x 16 gathered rows), post_attention_layernorm output; replicated in
                the row, rows differ. Not consumed.
            add_partial: optional ``[1, 1, L, 4096]`` TP partial (e.g. the shared expert's down-projection partial
                before its all_reduce) added before the final ``all_reduce(tp)``; dtype bf16 or ``combine_dtype``.
            reduce_tp: False returns the column partial ``[1, 1, L, 4096]`` (``combine_dtype``) without the final
                ``all_reduce(tp)`` (the caller closes it).
            taps: tests only (eager): receives ``f_all``, ``idx``, ``w``, ``w_loc``, ``part`` (not freed).
            lane_mask: B1 (``decode_experts="sparse"``): the step's ``[1, 1, 4 L, 1]`` fp32 live-row mask
                (:meth:`decode_lane_mask`; not consumed). ``None`` = no masking (exact, fewer experts skipped);
                ignored by the dense path.

        Returns ``[1, 1, L, 4096]`` bf16 TILE DRAM: the routed MoE output (+ ``add_partial``), replicated in the row.
        Row ``j`` depends on input row ``j`` only, bitwise the same at L = 8 and 16 (the per-M configs keep every row's
        arithmetic). On a mesh with a size-1 axis the collectives of that axis hand their input back; nothing the caller
        owns (``x``, ``add_partial``, ``taps``) is ever freed (:class:`_Keep`). A row count whose gathered M is not in
        ``self.decode_rows`` raises before any device op (no silent fallback to auto configs).
        """
        rows = int(x.shape[-2])
        if rows * int(self.cfg.dp) not in self.decode_rows:
            raise ValueError(
                f"MotifMoE layer {getattr(self, 'layer_idx', '?')}: decode input of {rows} rows per DP row gathers "
                f"{rows * int(self.cfg.dp)} tokens; this module serves {self.decode_rows} (the T64 step's 64 need a "
                f"config that stages it, cfg.wide_rows_per_dp > 0: spec_tokens > 0 and spec_verify 'wide' / 'auto', so "
                f"its 64-row constants exist before any trace capture, F3N rule R3)"
            )
        f_all = self.ccl.ag_dp_rows(x, memory_config=self.decode_mc)  # [1, 1, 4 L, 4096], natural order L dp + j
        part = self.local_partial(f_all, polynorm=self.decode_polynorm, decode=True, taps=taps,
                                  memory_config=self.decode_mc, lane_mask=lane_mask)
        if taps is not None:
            taps["f_all"] = f_all
            taps["part"] = part
        keep = _Keep(x, add_partial, lane_mask, *(taps.values() if taps is not None else ()))
        keep.drop(f_all)  # (is x itself when the DP axis has size 1)
        red = self.ccl.ar_dp(part)  # sum over the 4 chips of this column (all 4 L tokens)
        keep.drop(part, red)
        mine = self.ccl.partition(red, 2, "dp")  # this row's L rows
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
        # race_free: the AG half of this AR runs on ttnn's racy multicast gather for S/4 < 245 rows; prefill takes the
        # safe path even for its single-CB-page size (bucket 128: [32, 512] bf16, which raced in isolation; tt/ccl.py
        # module docstring, docs/determinism/INVESTIGATION.md). Decode's AR is not affected (forward_decode).
        ar = self.ccl.ar_tp(rs, race_free=True)
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
        _free(self.w_gate_up, self.w_down, self.local_ids, getattr(self, "local_mask", None))
        for z, o in getattr(self, "scatter_consts", {}).values():
            _free(z, o)
        self.local_mask, self.scatter_consts = None, {}
        if getattr(self, "pn_fused", None) is not None:
            self.pn_fused.deallocate()
            self.pn_fused = None
        if getattr(self, "router_fused", None) is not None:
            self.router_fused.deallocate()
            self.router_fused = None
        _free(getattr(self, "_disp_meta", None))
        self._disp_meta = None
        st = getattr(self, "compact_state", None)
        if st is not None and getattr(st, "owner", None) is self:
            st.deallocate()


__all__ = ["COMBINE_MODES", "CompactPrefillState", "DECODE_EXPERTS_MODES", "DECODE_ROWS", "EXACT_ROUTER_DECODE_ROWS",
           "MOE_POLYNORM_MODES", "MotifMoE", "MotifRouter", "POLYNORM_IMPLS", "POLYNORM_MODES", "compact_block",
           "compact_bucket", "compact_ladder", "compact_need_blocks", "compact_prefill_meta", "compact_upload_fast",
           "PREFILL_MOE_DISPATCH_MAX_ROWS", "resolve_prefill_moe_kernels",
           "grouped_polynorm",
           "prefill_experts_pc", "resolve_decode_experts", "resolve_moe_polynorm", "resolve_prefill_moe",
           "wide_decode_rows"]
