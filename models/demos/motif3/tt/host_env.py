# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Process environment the Motif-3 package sets before the first device open (P2, logs/opt/phaseC/P2). No ttnn import:
``tt/__init__.py`` runs :func:`apply_host_env` before any module of the package imports ttnn."""

from __future__ import annotations

import os

# tt-metal's per-process shared-memory allocation tracking (P2, MOTIF3_SHM_TRACKING): "off" (default) sets
# TT_METAL_SHM_TRACKING_DISABLED=1 before the first device open unless the caller set it already | "on" leaves tt-metal's
# default (tracking on). The tracker feeds tt-smi's per-process memory view only, and costs two getpid() syscalls per
# chip per buffer allocation and free: ~0.2 s of host time in a 53-layer 1K eager prefill (~8000 buffers x 32 chips,
# logs/opt/phaseC/P2). Host bookkeeping only: device programs and data are unchanged.
SHM_TRACKING_MODES = ("off", "on")


def apply_host_env(environ=None) -> str:
    """P2 (``MOTIF3_SHM_TRACKING``, :data:`SHM_TRACKING_MODES`): with "off" (default) set
    ``TT_METAL_SHM_TRACKING_DISABLED=1`` unless the caller set that variable. tt-metal reads it once, when its runtime
    options are first built, so this must run before the first device open; ``tt/__init__.py`` calls it when the package is
    first imported (the vLLM plugin imports the model class before it opens the mesh) and
    ``model_config.open_motif_mesh`` again.
    Returns the mode."""
    env = os.environ if environ is None else environ
    mode = (env.get("MOTIF3_SHM_TRACKING") or "off").strip().lower()
    if mode not in SHM_TRACKING_MODES:
        raise ValueError(f"MOTIF3_SHM_TRACKING must be one of {SHM_TRACKING_MODES}, got {mode!r}")
    if mode == "off":
        env.setdefault("TT_METAL_SHM_TRACKING_DISABLED", "1")
    return mode


__all__ = ["SHM_TRACKING_MODES", "apply_host_env"]
