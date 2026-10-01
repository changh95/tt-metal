# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn implementation of Motif-3 for the Blackhole Galaxy (design: docs/study/00_feasibility_and_design.md).

Shared infrastructure (import explicitly; this package ``__init__`` imports nothing so that importing
``models.demos.motif3.tt.<module>`` stays cheap and device-free):

* ``model_config`` -- ``MotifTTConfig`` (mesh axis roles, per-chip heads/experts, dtypes, layer schedule, KV pool,
  prefill buckets, compute configs, cache paths) and ``device_params()`` for the pytest mesh fixture.
* ``ccl``          -- ``MotifCCL``: axis-role all_gather / reduce_scatter / all_reduce / partition; replica checks.
* ``weights``      -- ``HFWeightLoader`` / ``DictWeightSource``, torch-side transforms, ``as_tensor`` with role-based
  mesh mappers and the TT weight cache.
* ``rope``         -- YaRN / plain tables, ``MotifRope`` (decode per-lane gather, prefill slices, composite / HF apply).
"""
