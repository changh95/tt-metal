# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Phase C P4 (docs/OPTIMIZATION_PLAN.md §3.3 C1): DP-row sequence split of the non-MoE prefill
(``MOTIF3_PREFILL_SP``).

The release prefill runs every row of a pass on all 4 DP rows: mHC, the norms, the attention projections, epilogue
and AR(tp), and the dense MLPs are computed 4 times. Only the MoE is split (each chip routes every row; the routed
partial is reduce-scattered over DP, so a DP row owns S/4 rows from the RS until the final all-gather).

With ``prefill_sp="dp"`` an sp0 pass of ``S >= prefill_sp_min_rows`` rows keeps the residual streams split from the
embedding to the last layer: DP row ``d`` owns rows ``[d R, (d + 1) R)`` with ``R = S / 4``.

* mHC, norms, projections, epilogue (+ AR(tp)), dense MLP, shared expert: row-local, on the row's R rows.
* Attention: the row's latent ``[n | rope(k_pe)]`` ``[1, 1, R, 576]`` is all-gathered over DP (``[1, 1, S, 576]``, the
  release's cache-fill tensor bit for bit, filled as before). Queries are the row's R rows (RoPE rows ``[d R, d R + R)``
  of the release tables, ``ccl.partition``).
  - Global layers: K / V expanded from the gathered latent (all S rows), SDPA with ``is_causal=False`` and an explicit
    per-DP-row causal mask ``[1, 1, R, S]`` (``0`` / ``-inf``; a mesh tensor whose rows differ per DP row).
  - SWA layers (window 129 = 128 rows back): K / V expanded from ``[tail_1 | tail_2 | tail_3 | own R rows]``, where
    ``tail_k`` = gathered rows ``[k R - 128, k R)`` (the same slices on every chip) and a per-row mask ``[1, 1, R,
    384 + R]`` that keeps the tail just before the row's rows and the 129-key window.
  - The SDPA call keeps the release's program config (q / k chunks) and compute config. Fully masked k-chunks then add
    exact zeros (exp(-inf) = 0, the running max unchanged), and every live chunk sees the same keys in the same order
    as the causal / windowed kernel, so the output is bitwise the release's (measured: logs/opt/phaseC/P4).
* MoE: the row's R FFN inputs are all-gathered over DP (the release's replicated ``f``) and the MoE runs unchanged up to
  RS(dp) + the shared partial + AR(tp); the release's final AG(dp) is skipped (the row keeps its R rows).
* After the last layer the streams are all-gathered over DP once (``[1, 4, S, 4096]``: the LM head and the MTP fill are
  unchanged).

Every new collective is a DP all-gather (a copy): ``race_free=True`` routes race-prone payloads to the safe gather.
The pass is bitwise equal to the release (logs/opt/phaseC/P4). It serves only sp0, non-packed passes (``chunk.path``
"sp0" or the draft-1 ``page_table`` call) whose rows split into 4 whole SDPA chunks of at least 128 rows; sp1, packed
and traced passes keep the release path.

Measured (logs/opt/phaseC/P4): 5-layer wrapper (synced stack wall) 4096 rows 127.7 -> 85.8 ms, 2048 rows 71.4 -> 63.1
ms, 1024 rows 56.1 -> 58.1 ms (host-bound), hence the 2048-row default floor; 53 layers, solo prefill 2K / 4K / 8K
0.958 / 1.696 / 3.409 -> 0.796 / 1.261 / 2.545 s with bitwise logits and KV pages.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

import ttnn

from .model_config import DEFAULT_PREFILL_SP, DEFAULT_PREFILL_SP_MIN_ROWS, PREFILL_SP_MODES, MotifTTConfig

SP_TAIL = 128  # SWA look-back rows: the window is 129 keys including the current one


def sp_rows_ok(cfg: MotifTTConfig, rows: int) -> bool:
    """Shape rule: ``rows`` splits over the DP rows into ``R`` rows that are whole SDPA chunks of both kinds (multiples
    of 256 and of the block size) and hold the SWA look-back (``R >= 128``)."""
    dp = int(cfg.dp)
    rows = int(rows)
    if dp < 2 or rows % dp:
        return False
    R = rows // dp
    return R >= SP_TAIL and R % 256 == 0 and R % int(cfg.kv_block_size) == 0


def sp_applies(cfg: MotifTTConfig, rows: int, path: Optional[str] = "sp0") -> bool:
    """``prefill_sp == "dp"``, an sp0 non-packed pass (``path`` "sp0"; ``None`` = the draft-1 page-table call) of
    ``rows >= prefill_sp_min_rows`` rows, :func:`sp_rows_ok`, and every SWA window at most ``SP_TAIL + 1`` keys."""
    if getattr(cfg, "prefill_sp", "off") != "dp":
        return False
    if path not in (None, "sp0"):
        return False
    if int(rows) < int(getattr(cfg, "prefill_sp_min_rows", DEFAULT_PREFILL_SP_MIN_ROWS)):
        return False
    if any(L.window is not None and int(L.window) > SP_TAIL + 1 for L in cfg.layers):
        return False
    return sp_rows_ok(cfg, rows)


def _masks_host(dp: int, S: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(global [dp, 1, R, S], swa [dp, 1, R, (dp - 1) 128 + R])`` bf16-exact 0 / -inf masks of the DP rows
    (module docstring)."""
    R = S // dp
    neg = float("-inf")
    q = torch.arange(R)[:, None]
    g = torch.full((dp, 1, R, S), neg)
    k = torch.arange(S)[None, :]
    for d in range(dp):
        g[d, 0][k <= d * R + q] = 0.0
    nt = dp - 1
    Kw = nt * SP_TAIL + R
    w = torch.full((dp, 1, R, Kw), neg)
    for d in range(dp):
        p = d * R + q
        pos = torch.empty(Kw, dtype=torch.long)
        for t in range(1, dp):
            pos[(t - 1) * SP_TAIL : t * SP_TAIL] = t * R - SP_TAIL + torch.arange(SP_TAIL)
        pos[nt * SP_TAIL :] = d * R + torch.arange(R)
        live = torch.zeros(Kw, dtype=torch.bool)
        live[nt * SP_TAIL :] = True
        if d > 0:
            live[(d - 1) * SP_TAIL : d * SP_TAIL] = True
        ok = live[None, :] & (pos[None, :] <= p) & (pos[None, :] >= p - SP_TAIL)
        w[d, 0][ok] = 0.0
    return g, w


class PrefillSP:
    """Per-model state of the split: the per-DP-row masks and the partitioned RoPE rows, per pass length (built on
    first use; eager only, never inside a trace)."""

    def __init__(self, mesh_device, cfg: MotifTTConfig, ccl, rope):
        self.mesh_device, self.cfg, self.ccl, self.rope = mesh_device, cfg, ccl, rope
        self.dp = int(cfg.dp)
        self._masks: Dict[int, Dict[str, object]] = {}
        self._rot: Dict[Tuple[str, int], Tuple[object, object]] = {}
        self._mapper = None

    def applies(self, rows: int, path: Optional[str] = "sp0") -> bool:
        return sp_applies(self.cfg, rows, path)

    def rows(self, S: int) -> int:
        return int(S) // self.dp

    def masks(self, S: int) -> Dict[str, object]:
        S = int(S)
        m = self._masks.get(S)
        if m is None:
            if self._mapper is None:
                from .rope import dp_row_mapper

                self._mapper = dp_row_mapper(self.cfg, self.mesh_device)
            g, w = _masks_host(self.dp, S)
            m = {
                name: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=self._mapper)
                for name, t in (("global", g), ("swa", w))
            }
            self._masks[S] = m
        return m

    def cos_sin(self, kind: str, S: int, rot_tables=None):
        """This DP row's RoPE rows ``(cos, sin)`` ``[1, 1, R, 64]``: the release's prefill tables (``rot_tables`` =
        ``None``, cached) or the given ``(cos, sin)`` (partitioned per call; the caller frees the result)."""
        if rot_tables is not None:
            cos, sin = rot_tables
            return self.ccl.partition(cos, 2, "dp"), self.ccl.partition(sin, 2, "dp")
        key = (kind, int(S))
        cs = self._rot.get(key)
        if cs is None:
            cos, sin = self.rope.prefill_cos_sin(kind, S)
            cs = self._rot[key] = (self.ccl.partition(cos, 2, "dp"), self.ccl.partition(sin, 2, "dp"))
        return cs

    def is_cached_rot(self, cs) -> bool:
        return any(cs is v for v in self._rot.values())

    def split(self, X):
        """``[1, 4, S, 4096]`` replicated -> this DP row's ``[1, 4, R, 4096]``."""
        return self.ccl.partition(X, 2, "dp")

    def gather(self, Xs):
        """This DP row's ``[1, 4, R, 4096]`` -> ``[1, 4, S, 4096]`` on every chip."""
        return self.ccl.ag_dp(Xs, 2, race_free=True)

    def release(self) -> None:
        for m in self._masks.values():
            for t in m.values():
                ttnn.deallocate(t)
        self._masks.clear()
        for c, s in self._rot.values():
            ttnn.deallocate(c)
            ttnn.deallocate(s)
        self._rot.clear()


__all__ = [
    "DEFAULT_PREFILL_SP",
    "DEFAULT_PREFILL_SP_MIN_ROWS",
    "PREFILL_SP_MODES",
    "PrefillSP",
    "SP_TAIL",
    "sp_applies",
    "sp_rows_ok",
]
