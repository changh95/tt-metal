# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Process environment the Motif-3 package sets before the first device open (P2, logs/opt/phaseC/P2). No ttnn import:
``tt/__init__.py`` runs :func:`apply_host_env` before any module of the package imports ttnn."""

from __future__ import annotations

import os
from pathlib import Path

# tt-metal's per-process shared-memory allocation tracking (P2, MOTIF3_SHM_TRACKING): "off" (default) sets
# TT_METAL_SHM_TRACKING_DISABLED=1 before the first device open unless the caller set it already | "on" leaves tt-metal's
# default (tracking on). The tracker feeds tt-smi's per-process memory view only, and costs two getpid() syscalls per
# chip per buffer allocation and free: ~0.2 s of host time in a 53-layer 1K eager prefill (~8000 buffers x 32 chips,
# logs/opt/phaseC/P2). Host bookkeeping only: device programs and data are unchanged.
SHM_TRACKING_MODES = ("off", "on")


# Physical grouping descriptor (PGD, MOTIF3_PGD): tt-metal's fabric init picks the 4x8 mesh placement from a PGD textproto
# it looks up under $TT_METAL_HOME/tests/tt_metal/tt_fabric/physical_groupings/ (or TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH;
# tt_metal/fabric/physical_grouping_descriptor_core.cpp find_and_load). A dev tree has the files and commits
# "4x8_Mesh_flat_torus_xy (TORUSXY)"; the ttnn wheel ships none (TT_METAL_HOME = site-packages/ttnn), so the bundle fell back
# to "MGD placement fallback M0 (MESH)". "auto" (default): when the variable is unset and the PGD tt-metal would choose for
# this board is missing under TT_METAL_HOME, point the variable at the byte-identical copy shipped in tt/pgd/ (tt-metal
# 3f050a42be2 tests/tt_metal/tt_fabric/physical_groupings/). The choice mirrors tt-metal's: a Blackhole Galaxy whose board
# id bits [35:32] are >= 3 is rev C (wh_bh_rev_c_galaxy_*), else rev A/B (bh_galaxy_rev_ab_*); the board id is the kmd's
# tt_serial. Any other card, unreadable sysfs or boards that disagree: leave the variable unset (tt-metal's own lookup).
# "off": never set it.
PGD_MODES = ("auto", "off")
PGD_ENV = "TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH"
PGD_DIR = Path(__file__).resolve().parent / "pgd"
PGD_REV_C = "wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto"
PGD_REV_AB = "bh_galaxy_rev_ab_physical_grouping_descriptor.textproto"
TT_SYSFS = Path("/sys/class/tenstorrent")


def bh_galaxy_pgd_name(sysfs=TT_SYSFS):
    """The PGD filename tt-metal selects for this host's Blackhole Galaxy, or None (not a BH Galaxy / unreadable /
    boards disagree)."""
    names = set()
    try:
        devs = sorted(Path(sysfs).iterdir())
    except OSError:
        return None
    for d in devs:
        try:
            card = (d / "tt_card_type").read_text().strip()
            serial = int((d / "tt_serial").read_text().strip(), 16)
        except (OSError, ValueError):
            return None
        if card != "galaxy-blackhole":
            return None
        names.add(PGD_REV_C if ((serial >> 32) & 0xF) >= 3 else PGD_REV_AB)
    return names.pop() if len(names) == 1 else None


def apply_pgd_env(environ=None, sysfs=TT_SYSFS):
    """MOTIF3_PGD (:data:`PGD_MODES`). Returns the path set, or None when the variable was left alone."""
    env = os.environ if environ is None else environ
    mode = (env.get("MOTIF3_PGD") or "auto").strip().lower()
    if mode not in PGD_MODES:
        raise ValueError(f"MOTIF3_PGD must be one of {PGD_MODES}, got {mode!r}")
    if mode == "off" or env.get(PGD_ENV):
        return None
    name = bh_galaxy_pgd_name(sysfs)
    if name is None:
        return None
    home = Path(env.get("TT_METAL_HOME") or ".")
    if (home / "tests" / "tt_metal" / "tt_fabric" / "physical_groupings" / name).is_file():
        return None  # dev tree: tt-metal finds it itself
    shipped = PGD_DIR / name
    if not shipped.is_file():
        return None
    env[PGD_ENV] = str(shipped)
    return env[PGD_ENV]


def apply_host_env(environ=None) -> str:
    """P2 (``MOTIF3_SHM_TRACKING``, :data:`SHM_TRACKING_MODES`): with "off" (default) set
    ``TT_METAL_SHM_TRACKING_DISABLED=1`` unless the caller set that variable. tt-metal reads it once, when its runtime
    options are first built, so this must run before the first device open; ``tt/__init__.py`` calls it when the package is
    first imported (the vLLM plugin imports the model class before it opens the mesh) and
    ``model_config.open_motif_mesh`` again. Also applies :func:`apply_pgd_env` (MOTIF3_PGD).
    Returns the mode."""
    env = os.environ if environ is None else environ
    mode = (env.get("MOTIF3_SHM_TRACKING") or "off").strip().lower()
    if mode not in SHM_TRACKING_MODES:
        raise ValueError(f"MOTIF3_SHM_TRACKING must be one of {SHM_TRACKING_MODES}, got {mode!r}")
    if mode == "off":
        env.setdefault("TT_METAL_SHM_TRACKING_DISABLED", "1")
    apply_pgd_env(env)
    return mode


__all__ = ["PGD_MODES", "SHM_TRACKING_MODES", "apply_host_env", "apply_pgd_env", "bh_galaxy_pgd_name"]
