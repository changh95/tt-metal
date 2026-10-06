# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree device kernels for Motif-3, launched from Python with ``ttnn.generic_op`` (no tt-metal rebuild).

* ``sinkhorn_motif`` -- exact Motif mHC coefficients (h_pre, h_post, Sinkhorn H; optionally straight into the
  ``attn_res_weighted_reduce_nc`` weight layout) in fp32 SFPU math (WAVE_A_REVIEW MHC-6 "Option B"); kernel sources
  in ``sinkhorn_motif/``.
* ``router_fp32`` -- exact-fp32 router logits (WAVE_A_REVIEW D1(b)); kernel sources in ``router_fp32/``.
* ``moe_polynorm`` -- fused grouped PolyNorm of the decode routed experts (B3, ``MOTIF3_MOE_POLYNORM=fused``); kernel
  sources in ``moe_polynorm/``.

Kernel sources are compiled from their absolute paths at first use (JIT cache ``~/.cache/tt-metal-cache``;
``sinkhorn_motif`` also passes a content hash of its sources as a define, so an edited kernel always rebuilds). This
``__init__`` imports nothing (device-free, cheap); import the submodules explicitly. Import rule (README §13): this
package and ``sinkhorn_motif`` import only ttnn / torch at module level
(``tests/unit/test_sinkhorn_motif.py::test_import_is_self_contained``, the ``test_infra_import`` probe).
"""
