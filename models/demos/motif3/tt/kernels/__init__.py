# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree device kernels for Motif-3, launched from Python with ``ttnn.generic_op`` (no tt-metal rebuild).

* ``sinkhorn_motif`` -- exact Motif mHC coefficients (h_pre, h_post, Sinkhorn H; optionally straight into the
  ``attn_res_weighted_reduce_nc`` weight layout) in fp32 SFPU math (WAVE_A_REVIEW MHC-6 "Option B"); kernel sources
  in ``sinkhorn_motif/`` (the SFPU routine in ``motif_mhc_sfpu.h``, shared with ``mhc_decode``).
* ``mhc_decode`` -- the fused decode mHC site (D3, ``MOTIF3_MHC_DECODE=fused``): finalize + Sinkhorn + layout in one
  program writing a packed coefficient tile, and the pre / post stream mixes expanding it locally (bitwise equal to the
  op path); kernel sources in ``mhc_decode/``.
* ``attn_combine`` -- the fused decode attention epilogue (D1, ``MOTIF3_ATTN_EPILOGUE=fused``); ``attn_combine/``.
* ``router_fp32`` -- exact-fp32 router logits (WAVE_A_REVIEW D1(b)); kernel sources in ``router_fp32/``.
* ``moe_polynorm`` -- fused grouped PolyNorm of the decode routed experts (B3, ``MOTIF3_MOE_POLYNORM=fused``); kernel
  sources in ``moe_polynorm/``.
* ``shared_polynorm`` -- fused decode PolyNorm of the MoE shared expert (B5, ``MOTIF3_SHARED_POLYNORM=fused``; a
  moments kernel + the release's TP all-gather + an apply kernel, bitwise equal to the composite); kernel sources in
  ``shared_polynorm/``.
* ``row_fold`` -- the decode MoE combine's row fold / unfold / fold-add (D4, ``MOTIF3_MOE_DECODE_CCL=rs``; ``row_fold/``).
* ``rm_tile`` -- bf16 untilize / tilize of ``MotifCCL.ag_dp_rows`` as data movement (D4, ``MOTIF3_AG_ROWS_LAYOUT``;
  ``rm_tile/``; bitwise ``ttnn.to_layout``).
* ``moe_compact`` -- the compacted prefill MoE's on-device row dispatch (B2b, ``MOTIF3_PREFILL_MOE_DISPATCH=device``;
  ``moe_dispatch/``) and gather combine (``MOTIF3_PREFILL_MOE_COMBINE=gather``; ``moe_combine/``), both exact.

Kernel sources are compiled from their absolute paths at first use (JIT cache ``~/.cache/tt-metal-cache``;
``sinkhorn_motif`` also passes a content hash of its sources as a define, so an edited kernel always rebuilds). This
``__init__`` imports nothing (device-free, cheap); import the submodules explicitly. Import rule (README §13): this
package and ``sinkhorn_motif`` import only ttnn / torch at module level
(``tests/unit/test_sinkhorn_motif.py::test_import_is_self_contained``, the ``test_infra_import`` probe).
"""
