# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hang localisation for eager prefill (Phase C P1diag, logs/opt/phaseC/P1diag): ``MOTIF3_DEBUG_SYNC``.

"off" (default): nothing (no device call, no output). "layer": :func:`debug_sync` blocks on the whole mesh
(``ttnn.synchronize_device``) at each call site and prints one ``[motif3.dbgsync]`` line after it returns, so a device
hang leaves the last completed site in the log: before a chunk's first layer, after every decoder layer of an eager
``prefill_chunk`` (release and P4 split paths), and before B2b's on-device dispatch of a compacted MoE chunk. Only
passes of more than :data:`MAX_TRACED_ROWS` rows sync (traced prefill buckets are at most 512 rows, so no sync is ever
issued inside a trace capture). Debug only: it serialises host and device (slower prefill) and changes no device
program and no value."""

from __future__ import annotations

import os
import time

DEBUG_SYNC_MODES = ("off", "layer")
# the largest traced prefill bucket (MOTIF3_PREFILL_TRACE <= 512): passes this size or smaller never sync
MAX_TRACED_ROWS = 512


def debug_sync_mode(environ=None) -> str:
    """``MOTIF3_DEBUG_SYNC`` (:data:`DEBUG_SYNC_MODES`; case and blanks ignored; unset or empty = "off")."""
    env = os.environ if environ is None else environ
    mode = (env.get("MOTIF3_DEBUG_SYNC") or "off").strip().lower() or "off"
    if mode not in DEBUG_SYNC_MODES:
        raise ValueError(f"MOTIF3_DEBUG_SYNC must be one of {DEBUG_SYNC_MODES}, got {mode!r}")
    return mode


def debug_sync_on(rows: int) -> bool:
    """Whether a pass of ``rows`` rows syncs: ``MOTIF3_DEBUG_SYNC=layer`` and ``rows`` > :data:`MAX_TRACED_ROWS`. Call
    sites check this first, so with "off" they build no tag and touch no device."""
    return int(rows) > MAX_TRACED_ROWS and debug_sync_mode() != "off"


def debug_sync(mesh_device, tag: str) -> None:
    """Wait for every queued op of ``mesh_device``, then print ``tag`` with the wait time (call only when
    :func:`debug_sync_on`)."""
    import ttnn

    t0 = time.perf_counter()
    ttnn.synchronize_device(mesh_device)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"[motif3.dbgsync {time.strftime('%H:%M:%S')}] {tag} ok ({dt:.1f} ms)", flush=True)


__all__ = ["DEBUG_SYNC_MODES", "MAX_TRACED_ROWS", "debug_sync", "debug_sync_mode", "debug_sync_on"]
