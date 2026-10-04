# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 final head on TT: stream mean -> final RMSNorm -> vocab-sharded LM head -> logits (design §2.3.8-2.3.9;
WAVE_A_REVIEW EMB-2, EMB-3).

Math (HF ``modeling_motif.py:1435-1439, 1667-1668``; reference ``MotifModel.reduce_streams`` / ``norm`` /
``MotifForCausalLM.lm_head``)::

    h = mean(X, streams)              # [T, 4096]  (bf16 in the reference: fp32 sum, one rounding)
    h = rms_norm(h; gamma_final, 1e-5)
    logits = h @ lm_head^T            # [T, 220160], untied lm_head, bf16 GEMM, returned as fp32 by HF

Device dataflow (decode, per chip; X ``[1, 4, 8, 4096]`` = the DP row's 8 lanes, README §3)::

    h  = sum_streams(X)               [1, 1, 8, 4096]  ttnn.sum(dim=1), fp32 acc; x 1/4 is folded into the norm eps
    hn = rms_norm(h, gamma, 16 eps)   [1, 1, 8, 4096]  width-sharded on 8 x 4 cores (interleaved = 1 core, 65 us)
    vocab_split="mesh" (default, decision EMB-D1 below):
      g      = ag_dp_rows(hn)          [1, 1, 32, 4096]  all 32 lanes on every chip (lane order 8 dp + l, ~27 us)
      logits = g @ W_b                 [1, 1, 32, 6880]  chip (r, c) owns vocab block b = r * C + c (mesh linear index)
    vocab_split="tp" (design §2.3.8 / EMB-2, README §3 / §6 as written):
      logits = hn @ W_tp               [1, 1, 8, 27520]  chip (dp, tp) owns vocab block tp, replicated over DP

Stream-mean numerics: ``rms_norm(s; 16 eps) == rms_norm(s / 4; eps)`` is an exact identity (1/4 is a power of two:
bit-exact in the reference's bf16 numerics, ``test_cpu_stream_sum_eps_identity``), so folding the 1/4 into the epsilon
costs nothing. The device sum itself is not single-rounding: ``ttnn.sum(dim=1)`` (fp32 dest acc) equals
``bf16(fp32 sum)`` on 94-98 % of the elements of real streams (after layers 1 / 8 / 16 / 35; 99.8 % on random streams of
4 different scales) and is 1 bf16 ulp off on the rest (``test_lm_head`` / ``test_lm_head_late_layers`` report it). The
norm output is therefore not bit-identical to the reference's bf16 mean -> RMSNorm (69-72 % of the elements are): PCC
0.999995 vs the bf16 reference and 0.999997 vs the fp32 ideal on real streams, max-abs 0.031 = 1 ulp at |h| 4-8
(README §12 RMSNorm threshold 0.9999).

Prefill (one user, X ``[1, 4, S, 4096]`` replicated, S = bucket): the 32-row tile holding the last real token
``last_index = S_real - 1`` is cut out with the **tensor-args** ``ttnn.slice`` (start/end live in two persistent
``[4]`` int32 device tensors, rewritten per call by :meth:`MotifLMHead.set_prefill_position`), run through the head
and untilized; the device returns that tile row ``[1, 1, 32, Vc]`` ROW_MAJOR per chip and
:meth:`MotifLMHead.prefill_logits_to_host` picks row ``last_index % 32`` on the host (EMB-3). Every program is keyed on
the bucket only (the slice program hash includes literal starts / ends, the tensor-args variant's does not): one call
per bucket before decode capture compiles everything, and no later prompt length adds a program (a program compiled
after capture allocates a kernel-binary DRAM buffer that the trace-allocation tracker reports as corruptible; P1 of the
embed_head review, ``test_prefill_head_trace_safety``). The device ops are traceable (``forward_prefill(x)`` without
``last_index`` reads the position last written by :meth:`set_prefill_position`).

Host logits (``logits_to_host`` / ``prefill_logits_to_host``): a **fresh** CPU tensor ``[32, 220160]`` (lane order)
/ ``[220160]``, bf16 by default (vLLM's sampler upcasts; HF's logits are bf16 GEMM outputs too). The decode logits are
untilized on device (inside the decode trace: ``forward_decode(row_major=True)``, +4 us), read from all 32 chips at
once into persistent host staging (:class:`HostShardReader`, ~0.3-0.5 ms: the 32 PCIe links work in parallel, 440 KB
each) and copied into the fresh output with numpy block copies on ``host_copy_threads`` (4) Python threads (~0.7 ms).
Measured end to end 1.7-2.1 ms median (host load 9-14). Gathering the shards onto the four PCIe x8 chips first cannot
beat this: an x8 chip reads ~8-9.7 GB/s vs ~2.8 GB/s for x1 on large reads, but at 440 KB per chip every read is
latency bound (~0.22 ms alone), and 14 MB from one x8 chip alone is >= 1.5 ms before any on-device gather. Avoided
host paths (measured): the ttnn mesh composer (~110 ms of host concatenation), per-shard ``ttnn.to_torch`` (~7.5 ms),
and a parallel ``torch.cat`` (its spinning OpenMP workers slow the next device read from 0.5 to 2.4-3.2 ms).

Serving pattern (trace-safe): capture ``forward_decode(x, row_major=True)`` (and, for device greedy, ``argmax_decode``
on ``forward_decode(x)``) in the decode trace; outside it read with ``logits_to_host`` / ``tokens_to_host`` (no device
op on a ROW_MAJOR input). A TILE input to ``logits_to_host`` / ``prefill_logits_to_host`` runs an eager untilize, so
warm that path before capture too. Prefill: ``forward_prefill(X, last_index)`` once per bucket during warmup.
Speculative decode (features design §3.8.1; README §17): ``forward_decode`` = :meth:`MotifLMHead.stream_mean_norm` +
:meth:`MotifLMHead.decode_logits`; the spec step keeps ``hn = stream_mean_norm(X)`` (``decode_logits(hn)`` does not
consume it by default) and hands it to the MTP layer (``tt/mtp.py``), whose ``final_layernorm`` output goes through
``decode_logits`` + :meth:`MotifLMHead.argmax_decode` as well (the shared LM head).

Decision EMB-D1 (2026-10-01; module owner; awaiting the README §3 / §6 and design §2.3.8-2.3.9 update by the shared-
infra owner): the default vocab split is "mesh" (6880 vocab per chip over all 32 chips) instead of the "tp" split the
README and the design describe (27520 per TP chip, replicated over DP).

* Why: the head is DRAM-bandwidth bound. "mesh" streams 56 MB of weight per chip per step instead of 225 MB: device
  time 605 -> 218 us per decode step (traced 606 -> 197 us), and 169 MB less DRAM per chip, for one extra ROW_MAJOR
  token gather (``ag_dp_rows``, ~27-45 us). The logits volume the host reads is 14 MB either way.
* Contract: decode logits are ``[1, 1, 32, 6880]`` per chip (all 32 lanes x vocab block ``r * C + c``) instead of
  ``[1, 1, 8, 27520]``; prefill logits are the last token's tile row ``[1, 1, 32, Vc]`` ROW_MAJOR.
* Rule for consumers (generator, sampler): never index the device logits directly; use only :meth:`logits_to_host`,
  :meth:`prefill_logits_to_host`, :meth:`argmax_decode` and :meth:`tokens_to_host`, which handle both splits (the
  bridge only ever sees host ``[32, V]`` / ``[V]`` tensors).
* "tp" stays available and tested (``vocab_split="tp"``) for a later device top-k sampler with one sampling group per
  DP row (design §2.3.9 v1); with "mesh" such a sampler is one group over all 32 chips (local top-k, then the same two
  all-gathers ``argmax_decode`` uses). Reverting is a constructor argument; nothing else depends on the split.

Measured device time per call (device profiler, slowest of 32 chips; ``test_profile_head``), decode, bf16:
"mesh" head 218 us = stream sum 15 + sharded norm 9 + ag_dp_rows ~45 (DP axis a line on TORUS_Y) + GEMM 142-144
(391 GB/s); traced 197 us/call. "tp" head 605 us (GEMM 570-580 us); traced 606 us. Argmax ("mesh", vector local
stage) traced 217 us; ("tp", rm local stage) 307 us. Prefill head (host-timed, ``test_lm_head`` /
``test_prefill_head_trace_safety``): eager 0.94-1.0 ms per call including the position upload (host-dispatch bound),
traced replay + sync 234-239 us (S = 256); host read of the tile row 0.39 ms ("mesh", latency bound like one row) /
0.93 ms ("tp", 1.76 MB per chip); a new bucket compiles 1 program (its slice), a new position none.

Greedy (``argmax_decode``): per chip the row max and the lowest index attaining it (vectorized: ``max``, ``lt``,
int32 ``min``), then the per-chip (value, index) pairs are gathered across the vocab shards and the token is the
**lowest global index among the maxima** (int32 min with a sentinel; the tie rule of ``torch.argmax`` over the full
row). Returns token ids ``[1, 1, 1, 32]`` uint32 ROW_MAJOR in lane order on every chip ("mesh"), or ``[1, 1, 1, 8]``
per DP row ("tp").

T64 verify step (docs/p5_t64/P5_T64_DESIGN.md §4.2, §4.4; "mesh" split): ``hn [1, 1, 16, 4096]`` per DP row (``[8
anchors | 8 drafts]``) -> ``decode_logits(hn, halves=2)``: the split-order gather ``ccl.ag_dp_rows(hn, halves=2)`` ->
``[1, 1, 64, 4096]`` (rows 0..31 = the anchors in lane order, i.e. the 32-lane layout; rows 32..63 = the drafts) -> the
GEMM with ``per_core_M`` 2 (``lm_head_program_config(m_tiles=2)``, the G16-lite config: 149.2 vs 143.8 us, weight
bound) -> ``[1, 1, 64, 6880]``. ``argmax_decode`` takes the row count L from the logits (32 or 64; its constants for
64 rows -- the 1.76 MB int32 iota among them -- are allocated in the constructor when ``cfg.wide_rows_per_dp`` is set,
never after a trace capture, F3N rule R3): ``[1, 1, 1, 64]`` = ``(a0 of lanes 0..31, a1 of lanes 0..31)``.
``logits_rm(logits, rows=32)`` untilizes the anchor rows only (a tile-aligned slice; the ``wide`` mode's host-sampled
steps). Every row equals the 32-lane head's row of the same hidden state bitwise (gather, GEMM and argmax are
row-local; G16-lite measured GEMM rows bitwise; ``test_embed_head.py::test_embed_head_t64``).

Compute roles (README §4): ``norm`` (HiFi4, fp32 acc) for the reduction / norm, ``lm_head`` (HiFi4, fp32 acc) for the
GEMM; explicit 1D-multicast program configs (the head is bandwidth bound).

Import rule: ttnn, torch and the shared motif3 infra only; no other ``models/demos/**`` package.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

import ttnn

from . import weights as W
from .ccl import MotifCCL
from .model_config import TILE, MotifTTConfig

LM_HEAD_WEIGHT = "lm_head.weight"
FINAL_NORM_WEIGHT = "model.norm.weight"
VOCAB_SPLITS = ("mesh", "tp")
# Larger than any global vocab index (220160 < 2^18) and exact in bf16 / int32: marks "not the maximum" in the
# cross-shard argmax (same constant as the BH Qwen3 local-first argmax, tt_sampling.TIEBREAK_INDEX_SENTINEL).
TIEBREAK_SENTINEL = 1 << 24


# ============================================================================================================
# host-side layout helpers (pure torch; CPU-tested)
# ============================================================================================================
def vocab_block_of_coord(cfg: MotifTTConfig, row: int, col: int, vocab_split: str = "mesh") -> int:
    """Vocab block held by mesh coordinate (row, col): ``row * C + col`` ("mesh", 32 blocks of 6880) or the chip's TP
    index ("tp", 8 blocks of 27520)."""
    if vocab_split == "mesh":
        return int(row) * cfg.axes.mesh_shape[1] + int(col)
    if vocab_split == "tp":
        return cfg.axes.roles(row, col)[1]
    raise ValueError(f"vocab_split must be one of {VOCAB_SPLITS}, got {vocab_split!r}")


def vocab_per_shard(cfg: MotifTTConfig, vocab_split: str = "mesh") -> int:
    n = cfg.num_chips if vocab_split == "mesh" else cfg.tp
    if cfg.vocab_size % (n * TILE):
        raise ValueError(f"vocab {cfg.vocab_size} does not split into {n} tile-aligned blocks")
    return cfg.vocab_size // n


def lm_head_device_layout(lm_head: torch.Tensor, cfg: MotifTTConfig, vocab_split: str = "mesh") -> torch.Tensor:
    """HF ``lm_head.weight [V, 4096]`` -> host tensor for the mesh upload (pure transposes: dtype kept, exact).

    * "mesh": ``[R, 1, 4096, C * Vc]`` (Vc = 6880); mesh dim 0 shards tensor dim 0 and mesh dim 1 tensor dim 3, so
      chip (r, c) gets ``lm_head[Vc b : Vc (b + 1)]^T`` with ``b = r * C + c``.
    * "tp": ``[1, 1, 4096, V]``; TP shard on dim 3 = ``weights.lm_head_for_chip(lm_head, cfg, tp)`` per chip."""
    V, D = lm_head.shape
    if V != cfg.vocab_size or D != cfg.hidden_size:
        raise ValueError(f"{LM_HEAD_WEIGHT} has shape {tuple(lm_head.shape)}, expected {(cfg.vocab_size, cfg.hidden_size)}")
    wt = lm_head.t()  # [D, V] view
    if vocab_split == "tp":
        return wt.contiguous().reshape(1, 1, D, V)
    R, C = cfg.axes.mesh_shape
    return wt.reshape(D, R, C * (V // (R * C))).permute(1, 0, 2).contiguous().reshape(R, 1, D, V // R)


def lm_head_mesh_dims(cfg: MotifTTConfig, vocab_split: str = "mesh") -> Tuple[Optional[int], Optional[int]]:
    """``(dp_dim, tp_dim)`` for ``weights.as_tensor`` that realize :func:`lm_head_device_layout`."""
    if vocab_split == "tp":
        return None, 3
    a = cfg.axes  # mesh dim 0 -> tensor dim 0, mesh dim 1 -> tensor dim 3, whatever the role of each mesh dim
    return (0 if a.dp_axis == 0 else 3), (0 if a.tp_axis == 0 else 3)


def assemble_decode_logits(shards: torch.Tensor, cfg: MotifTTConfig, vocab_split: str = "mesh") -> torch.Tensor:
    """Per-chip decode logits ``[R, C, rows, Vc]`` (``[r, c]`` = mesh coordinate) -> lane-ordered ``[32, V]``.

    "mesh": every chip holds all 32 lanes of vocab block ``b(r, c)``. "tp": chip (dp, tp) holds lanes
    ``8 dp .. 8 dp + 7`` of vocab block tp (rows beyond the 8 lanes are ignored)."""
    R, C = cfg.axes.mesh_shape
    Vc = shards.shape[-1]
    out = torch.empty(cfg.max_batch, cfg.vocab_size, dtype=shards.dtype)
    for r in range(R):
        for c in range(C):
            b = vocab_block_of_coord(cfg, r, c, vocab_split)
            if vocab_split == "mesh":
                out[:, b * Vc : (b + 1) * Vc] = shards[r, c, : cfg.max_batch]
            else:
                dp = cfg.axes.roles(r, c)[0]
                L = cfg.lanes_per_row
                out[dp * L : (dp + 1) * L, b * Vc : (b + 1) * Vc] = shards[r, c, :L]
    return out


def argmax_offsets(cfg: MotifTTConfig, vocab_split: str = "mesh", lanes: Optional[int] = None) -> torch.Tensor:
    """``[1, 1, 32 n, lanes]`` int32 table of the cross-shard argmax: row ``32 j`` holds the global vocab offset of
    the ``j``-th gathered shard (gather order: TP first, then DP), every other row ``TIEBREAK_SENTINEL`` (those rows
    hold the tile padding of the gathered index tiles and can never win the min)."""
    a = cfg.axes
    vc = vocab_per_shard(cfg, vocab_split)
    if vocab_split == "mesh":
        order = [(dp, tp) for dp in range(a.dp_size) for tp in range(a.tp_size)]  # j = dp * tp_size + tp
    else:
        order = [(0, tp) for tp in range(a.tp_size)]
    lanes = (cfg.max_batch if vocab_split == "mesh" else cfg.lanes_per_row) if lanes is None else int(lanes)
    t = torch.full((1, 1, TILE * len(order), lanes), TIEBREAK_SENTINEL, dtype=torch.int32)
    for j, (dp, tp) in enumerate(order):
        r, c = a.coord(dp, tp)
        t[:, :, TILE * j, :] = vocab_block_of_coord(cfg, r, c, vocab_split) * vc
    return t


LOGITS_DTYPES = (ttnn.bfloat16, ttnn.float32)


def resolve_logits_dtype(cfg: MotifTTConfig, logits_dtype=None):
    """Device / host logits dtype: the activation dtype (bf16) unless given -- deliberately **not** the LM-head weight
    dtype ``cfg.dtypes.lm_head``, so a bfp8 head weight (memory lever) still yields bf16 logits. Only bf16 / fp32:
    block-float logits have no ROW_MAJOR layout (the device untilize and the zero-copy host read need one)."""
    dt = cfg.dtypes.activations if logits_dtype is None else logits_dtype
    if dt not in LOGITS_DTYPES:
        raise ValueError(f"logits dtype must be one of {LOGITS_DTYPES} (no block-float: the host read is ROW_MAJOR), got {dt}")
    return dt


def mcast1d_pc(
    grid: Tuple[int, int],
    n_tiles: int,
    per_core_n: int,
    in0_block_w: int,
    fuse_batch: bool = True,
    *,
    per_core_m: int = 1,
):
    """``MatmulMultiCoreReuseMultiCast1DProgramConfig`` (in0 multicast) with ``ceil(N / per_core_n)`` output blocks on
    ``grid``; unlike ``model_config.mcast1d_matmul_pc`` the last block may be partial (215 vocab tiles do not split
    evenly; the 1D factory handles the tail, verified bit-identical to the auto config). M = ``per_core_m`` tile rows
    (in0 is multicast: every core computes all of them): 1 for the 32-lane step, 2 for the T64 step's 64 rows (out
    subblock height 1 either way)."""
    n_blocks = -(-int(n_tiles) // int(per_core_n))
    if n_blocks > int(grid[0]) * int(grid[1]):
        raise ValueError(f"{n_blocks} output blocks do not fit grid {grid}")
    pcm = int(per_core_m)
    if pcm < 1:
        raise ValueError(f"per_core_m must be >= 1 tile row, got {per_core_m}")
    sub_w = max(d for d in (1, 2, 4) if per_core_n % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(int(grid[0]), int(grid[1])),
        in0_block_w=int(in0_block_w),
        out_subblock_h=1,
        out_subblock_w=sub_w,
        out_block_h=pcm,
        out_block_w=int(per_core_n),
        per_core_M=pcm,
        per_core_N=int(per_core_n),
        fuse_batch=bool(fuse_batch),
        fused_activation=None,
        mcast_in0=True,
    )


# (grid, per_core_N, in0_block_w) per vocab split for the M = 32 decode / prefill-row GEMM. Device-profiler kernel time
# on this Galaxy (tests/unit/test_embed_head.py::test_profile_head, slowest chip): "mesh" 144 us = 391 GB/s of bf16
# weights (ttnn's auto config 143 us; fewer cores are slower: 54 cores 164 us); "tp" 572-579 us (auto 570 us). Both are
# at the measured DRAM stream ceiling (~393 GB/s), so the head is purely bandwidth bound.
DEFAULT_LM_HEAD_PC = {
    "mesh": ((12, 9), 2, 16),  # 108 cores x 2 tiles = the 215 vocab tiles of 6880 (last block partial)
    "tp": ((12, 9), 8, 16),  # 108 cores x 8 tiles = the 860 vocab tiles of 27520 (last block partial)
}


def lm_head_program_config(cfg: MotifTTConfig, vocab_split: str = "mesh", spec=None, *, m_tiles: int = 1):
    """Program config of the LM-head GEMM ``[.., 32 m_tiles, 4096] @ [4096, Vc]`` (``m_tiles=2``: the T64 step's 64
    rows, ``per_core_M`` 2 on the same grid; equals ``model_config.lm_head_pc(m_tiles=)``). ``spec`` = ``(grid,
    per_core_N, in0_block_w)`` overrides the measured default; ``spec="auto"`` returns None (ttnn's auto config)."""
    if spec == "auto":
        return None
    grid, pcn, ibw = spec if spec is not None else DEFAULT_LM_HEAD_PC[vocab_split]
    gx, gy = int(grid[0]), int(grid[1])
    if gx > cfg.compute_grid[0] or gy > cfg.compute_grid[1]:
        raise ValueError(f"grid {grid} exceeds the chip compute grid {cfg.compute_grid}")
    return mcast1d_pc((gx, gy), vocab_per_shard(cfg, vocab_split) // TILE, pcn, ibw, per_core_m=m_tiles)


def _np_view(t: torch.Tensor):
    """Zero-copy numpy view of a torch tensor (bf16 viewed as int16: numpy has no bf16)."""
    return (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy()


def _concat_views(views, lanes: int, vc: int, vocab_split: str, axes, C: int, pool=None, np_views=None) -> torch.Tensor:
    """Assemble per-chip ``[.., lanes, vc]`` views (mesh row-major order) into a fresh lane-major ``[32, V]`` with
    numpy copies (no torch intra-op threads; see :meth:`MotifLMHead._assemble_decode`). ``pool``: a
    ``ThreadPoolExecutor`` splitting the copy over its workers (numpy releases the GIL in the block copies);
    ``np_views``: the matching numpy views, if already made (the staging reader caches them)."""
    import numpy as np

    dt = views[0].dtype
    src = [(_np_view(v) if np_views is None else np_views[i]).reshape(lanes, vc) for i, v in enumerate(views)]
    if vocab_split == "mesh":  # chip i holds vocab block i for all lanes
        out = np.empty((lanes, len(src) * vc), dtype=src[0].dtype)
        blocks = out.reshape(lanes, len(src), vc)

        def copy(idx):
            for i in idx:
                blocks[:, i] = src[i]

    else:  # chip (dp, tp) holds lanes 8 dp .. of vocab block tp
        out = np.empty((axes.dp_size * lanes, axes.tp_size * vc), dtype=src[0].dtype)
        blocks = out.reshape(axes.dp_size, lanes, axes.tp_size, vc)
        dst = [axes.roles(*divmod(i, C)) for i in range(len(src))]

        def copy(idx):
            for i in idx:
                dp, tp = dst[i]
                blocks[dp, :, tp] = src[i]

    if pool is None:
        copy(range(len(src)))
    else:
        n = int(getattr(pool, "_max_workers", 4))
        list(pool.map(copy, [range(k, len(src), n) for k in range(n)]))
    t = torch.from_numpy(out)
    return t.view(torch.bfloat16) if dt == torch.bfloat16 else t


def _prefill_row_from_views(views, row: int, vc: int, vocab_split: str, axes, np_views=None) -> torch.Tensor:
    """Per-chip prefill tile-row views ``[.., 32, vc]`` (mesh row-major order) -> **fresh** ``[V]`` = tile row ``row``
    of every vocab block in vocab order ("mesh": chip ``i`` holds block ``i``; "tp": the chips of DP row 0, by tp)."""
    import numpy as np

    C = axes.mesh_shape[1]
    src = [(_np_view(v) if np_views is None else np_views[i]) for i, v in enumerate(views)]
    if vocab_split == "mesh":
        parts = src
    else:
        by_tp = {}
        for i, v in enumerate(src):
            dp, tp = axes.roles(*divmod(i, C))
            if dp == 0:
                by_tp[tp] = v
        parts = [by_tp[tp] for tp in range(axes.tp_size)]
    out = torch.from_numpy(np.concatenate([v.reshape(TILE, vc)[int(row)] for v in parts]))  # one copy
    return out.view(torch.bfloat16) if views[0].dtype == torch.bfloat16 else out


class HostShardReader:
    """Persistent host staging for repeated reads of one device tensor spec (ROW_MAJOR): a host mesh tensor
    allocated once (``ttnn.allocate_tensor_on_host``) and, per chip, a **zero-copy** torch view of its buffer
    (``to_torch_with_padded_shape`` wraps a ROW_MAJOR host buffer without copying). :meth:`read` is one
    ``ttnn.copy_device_to_host_tensor`` (the mesh command queue reads every chip's shard concurrently) and returns
    the views in mesh row-major order. The views are overwritten by the next read: callers copy out of them (the
    host logits are assembled into a fresh tensor). Measured on this Galaxy (``test_probe_transfer``, 14.1 MB of decode
    logits): the ttnn mesh composer spends ~107 ms of host time concatenating 32 x 440 KB and per-shard
    ``ttnn.to_torch`` ~7.5 ms; this read is ~0.3-0.5 ms, plus the caller's copy into the fresh output (decode: numpy
    block copies on ``host_copy_threads`` threads, ~0.7 ms with 4; prefill: one row per chip)."""

    def __init__(self, mesh_device, device_tensor):
        if device_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError("HostShardReader reads ROW_MAJOR tensors (untilize on device first)")
        self.mesh_device = mesh_device
        self.key = self.spec_key(device_tensor)
        self.host = ttnn.allocate_tensor_on_host(device_tensor.spec, mesh_device)
        self.views = [s.to_torch_with_padded_shape() for s in ttnn.get_device_tensors(self.host)]
        self.np_views = [_np_view(v) for v in self.views]  # zero-copy numpy views of the same staging memory

    @staticmethod
    def spec_key(t) -> tuple:
        return (tuple(t.shape), t.dtype, t.layout)

    def read(self, device_tensor):
        if self.spec_key(device_tensor) != self.key:
            raise ValueError(f"tensor {self.spec_key(device_tensor)} does not match the staging spec {self.key}")
        ttnn.copy_device_to_host_tensor(device_tensor, self.host, blocking=True)
        return self.views


def sharded_norm_configs(cfg: MotifTTConfig, grid: Tuple[int, int] = (8, 4), rows: int = TILE):
    """Width-sharded decode RMSNorm (one 32-row tile row): ``(memory_config, LayerNormShardedMultiCoreProgramConfig)``
    on ``grid`` cores, shard ``[32, hidden / ncores]``. The interleaved ``ttnn.rms_norm`` parallelizes over tile rows
    only, so a ``[1, 1, 8 | 32, 4096]`` input runs on ONE core (65 us device time on this Galaxy, device profiler);
    sharded over 32 cores it is a few us (``test_profile_head``)."""
    gx, gy = int(grid[0]), int(grid[1])
    n = gx * gy
    if cfg.hidden_size % (n * TILE):
        raise ValueError(f"hidden {cfg.hidden_size} does not split into {n} tile-aligned shards")
    block_w = cfg.hidden_size // n // TILE
    subblock_w = max(d for d in (4, 3, 2, 1) if block_w % d == 0)
    mc = ttnn.create_sharded_memory_config(
        shape=(int(rows), cfg.hidden_size // n),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        subblock_w=int(subblock_w),
        block_h=int(rows) // TILE,
        block_w=int(block_w),
        inplace=False,
    )
    return mc, pc


# ============================================================================================================
# the module
# ============================================================================================================
class MotifLMHead:
    """Final stream mean + RMSNorm + vocab-sharded LM head, host logits and on-device greedy argmax.

    Args:
        mesh_device / cfg / source / ccl / cache: README §10 module pattern (``source`` holds ``model.norm.weight``
            and ``lm_head.weight``; ``cache=False`` for random weights).
        vocab_split: "mesh" (default, decision EMB-D1; 6880 vocab per chip over all 32 chips) or "tp" (27520 per TP
            chip, replicated over DP; design §2.3.8).
        logits_dtype: device / host logits dtype: ``cfg.dtypes.activations`` (bf16) by default, or ``ttnn.float32``
            (doubles the host transfer). Independent of the weight dtype ``cfg.dtypes.lm_head`` (a bfp8 head weight
            still produces bf16 logits); block-float logits are rejected (no ROW_MAJOR layout for the host read).
        stream_reduce: "sum" (``ttnn.sum(dim=1)`` + 16 eps, default) or "wreduce"
            (``attn_res_weighted_reduce_nc`` with 0.25 fp32 weights; design §2.3.8 alternative).
        program_config: ``(grid, per_core_N, in0_block_w)`` | "auto" | None (measured default).
        norm_grid: cores of the width-sharded final RMSNorm (default 8 x 4); None = interleaved ``rms_norm``
            (single core for one tile row, 65 us).
        argmax_local: local stage of :meth:`argmax_decode`: "vector" (``max`` / ``lt`` / int32 ``min``, the default
            for "mesh") or "rm" (untilize + ``ttnn.argmax``, the default for "tp": its per-row scan is cheaper on 8
            rows than the single-core W reductions over 860 tiles).
        host_copy_threads: Python threads of the host logits assembly copy (1 = inline).
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        *,
        source,
        ccl: Optional[MotifCCL] = None,
        cache: bool = True,
        vocab_split: str = "mesh",
        logits_dtype=None,
        stream_reduce: str = "sum",
        program_config=None,
        norm_grid: Optional[Tuple[int, int]] = (8, 4),
        argmax_local: Optional[str] = None,
        host_copy_threads: int = 4,
        memory_config=None,
    ):
        if vocab_split not in VOCAB_SPLITS:
            raise ValueError(f"vocab_split must be one of {VOCAB_SPLITS}, got {vocab_split!r}")
        if stream_reduce not in ("sum", "wreduce"):
            raise ValueError(f"stream_reduce must be 'sum' or 'wreduce', got {stream_reduce!r}")
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.ccl = ccl if ccl is not None else MotifCCL(mesh_device, cfg)
        self.vocab_split = vocab_split
        self.stream_reduce = stream_reduce
        self.n_streams = int(cfg.n_streams)
        self.hidden = int(cfg.hidden_size)
        self.lanes = int(cfg.lanes_per_row)
        self.vc = vocab_per_shard(cfg, vocab_split)
        self.logits_dtype = resolve_logits_dtype(cfg, logits_dtype)
        self.memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        self.eps = float(cfg.rms_norm_eps)
        # rms_norm(s; n^2 eps) == rms_norm(s / n; eps): the 1/n of the stream mean is folded into the epsilon
        self.norm_eps = self.eps * self.n_streams**2 if stream_reduce == "sum" else self.eps
        self.ckc_norm = cfg.compute_config("norm")
        self.ckc_lm = cfg.compute_config("lm_head")
        self.pc = lm_head_program_config(cfg, vocab_split, program_config)
        self.argmax_local = argmax_local or ("vector" if vocab_split == "mesh" else "rm")
        if self.argmax_local not in ("vector", "rm"):
            raise ValueError(f"argmax_local must be 'vector' or 'rm', got {argmax_local!r}")
        self.norm_grid = tuple(norm_grid) if norm_grid is not None else None
        self._norm_sharded = sharded_norm_configs(cfg, self.norm_grid) if self.norm_grid is not None else None
        self._readers = {}  # HostShardReader per device-tensor spec (host staging, allocated on first read)
        self.host_copy_threads = max(1, int(host_copy_threads))
        self._copy_pool = None
        rep = self._rep = ttnn.ReplicateTensorToMesh(mesh_device)

        self.gamma = W.as_tensor(
            lambda: W.norm_weight(source.get(W.hf_name(None, FINAL_NORM_WEIGHT))),
            mesh_device=mesh_device,
            cfg=cfg,
            dtype=cfg.dtypes.norms,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_name=("final_norm.weight" if cache else None),
            layer=None,
        )
        dp_dim, tp_dim = lm_head_mesh_dims(cfg, vocab_split)
        self.weight = W.as_tensor(
            lambda: lm_head_device_layout(source.get(W.hf_name(None, LM_HEAD_WEIGHT)), cfg, vocab_split),
            mesh_device=mesh_device,
            cfg=cfg,
            dtype=cfg.dtypes.lm_head,
            layout=ttnn.TILE_LAYOUT,
            dp_dim=dp_dim,
            tp_dim=tp_dim,
            cache_name=(f"lm_head.weight_{vocab_split}" if cache else None),
            layer=None,
        )
        # constants (replicated, allocated once: trace-safe)
        self._w_mean = {}
        if stream_reduce == "wreduce":
            for T in (self.lanes, TILE):
                self._w_mean[T] = ttnn.from_torch(
                    torch.full((1, self.n_streams, T, 1), 1.0 / self.n_streams),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rep,
                )
        self.argmax_lanes = cfg.max_batch if vocab_split == "mesh" else self.lanes
        self._am_zeros, self._am_zeros_i32, self._am_iota, self._am_offsets = self._argmax_constants(self.argmax_lanes)
        # T64 verify step ("mesh"; docs/p5_t64/P5_T64_DESIGN.md §4.2): the GEMM config of its 64 gathered rows
        # (per_core_M 2; a config only) and, when cfg stages that step, the 64-row argmax constants -- allocated here,
        # never after a trace capture (F3N rule R3; ~2 MB per chip, the int32 iota [64, 6880] among them).
        self.pc_wide: Dict[int, object] = {}
        self._am_wide: Dict[int, Tuple[object, object, object, object]] = {}
        if vocab_split == "mesh":
            wide = 2 * self.argmax_lanes
            self.pc_wide[wide] = lm_head_program_config(cfg, vocab_split, program_config, m_tiles=wide // TILE)
            if int(getattr(cfg, "wide_rows_per_dp", 0) or 0):
                self._am_wide[wide] = self._argmax_constants(wide)
        # Prefill tile-row position: the start / end of the tensor-args slice live on the device (persistent [4]
        # int32 ROW_MAJOR, allocated here, before any trace exists, and rewritten in place per call by
        # set_prefill_position). The slice program is then keyed on the bucket shape only, never on the position.
        self._pf_start, self._pf_end = (
            ttnn.from_torch(
                torch.zeros(4, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            for _ in range(2)
        )

    def _argmax_constants(self, rows: int):
        """``(zeros [1,1,L,32] logits dtype, zeros [1,1,L,32] int32, iota [1,1,L,Vc] int32 or None, offsets)`` of
        :meth:`argmax_decode` for ``L = rows`` logits rows (replicated, DRAM; created once, before any trace)."""
        rep, dram = self._rep, ttnn.DRAM_MEMORY_CONFIG

        def up(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                                   memory_config=dram, mesh_mapper=rep)  # fmt: skip

        L = int(rows)
        # same dtype as the logits, so the per-shard maxima are compared across shards without a rounding step
        zeros = up(torch.zeros(1, 1, L, TILE, dtype=torch.float32), self.logits_dtype)
        zeros_i32 = up(torch.zeros(1, 1, L, TILE, dtype=torch.int32), ttnn.int32)
        iota = None
        if self.argmax_local == "vector":
            iota = up(torch.arange(self.vc, dtype=torch.int32).reshape(1, 1, 1, -1).expand(1, 1, L, -1).contiguous(),
                      ttnn.int32)  # fmt: skip
        offsets = up(argmax_offsets(self.cfg, self.vocab_split, L), ttnn.int32)
        return zeros, zeros_i32, iota, offsets

    # ---- pieces ------------------------------------------------------------------------------------------
    def stream_mean_norm(self, x) -> ttnn.Tensor:
        """``X [1, 4, T, 4096]`` -> ``rms_norm(mean_streams(X); gamma_final)`` ``[1, 1, T, 4096]`` bf16 (T <= 32: decode
        8 lanes, the T64 step's 16 rows, or the prefill tile row), DRAM interleaved. The norm runs width-sharded on
        ``norm_grid``."""
        T = int(x.shape[2])
        if self.stream_reduce == "wreduce":
            w = self._w_mean.get(T)
            if w is None:
                raise ValueError(f"no stream-mean weights for T={T} (have {sorted(self._w_mean)})")
            h = ttnn.experimental.deepseek_prefill.attn_res_weighted_reduce_nc(x, w, dim=1)
        else:
            h = ttnn.sum(x, dim=1, keepdim=True, compute_kernel_config=self.ckc_norm, memory_config=self.memory_config)
        if self._norm_sharded is None or T > TILE:
            hn = ttnn.rms_norm(
                h, epsilon=self.norm_eps, weight=self.gamma, compute_kernel_config=self.ckc_norm,
                memory_config=self.memory_config,
            )
            ttnn.deallocate(h)
            return hn
        smc, spc = self._norm_sharded
        hs = ttnn.to_memory_config(h, smc)
        ttnn.deallocate(h)
        hns = ttnn.rms_norm(
            hs, epsilon=self.norm_eps, weight=self.gamma, program_config=spc, memory_config=smc,
            compute_kernel_config=self.ckc_norm,
        )
        ttnn.deallocate(hs)
        hn = ttnn.to_memory_config(hns, self.memory_config)
        ttnn.deallocate(hns)
        return hn

    def project(self, hn) -> ttnn.Tensor:
        """``[1, 1, M, 4096] @ W`` -> ``[1, 1, M, Vc]`` logits (``lm_head`` role, 1D-mcast config): M <= 32 (one tile
        row) the measured config, M = 64 (the T64 step, "mesh") its ``per_core_M`` 2 twin ``pc_wide[64]``."""
        return ttnn.linear(
            hn,
            self.weight,
            program_config=self._pc_for_rows(int(hn.shape[-2])),
            compute_kernel_config=self.ckc_lm,
            dtype=self.logits_dtype,
            memory_config=self.memory_config,
        )

    def _pc_for_rows(self, rows: int):
        if rows <= TILE:
            return self.pc
        pcs = getattr(self, "pc_wide", {})
        if rows not in pcs:
            raise ValueError(
                f"no LM-head GEMM config for {rows} rows: one tile row, or {sorted(pcs)} (the T64 step's gathered "
                f"rows, vocab split 'mesh')"
            )
        return pcs[rows]

    # ---- forwards ----------------------------------------------------------------------------------------
    def forward_decode(self, x, *, row_major: bool = False) -> ttnn.Tensor:
        """Decode head. ``x``: ``X [1, 4, 8, 4096]`` (this DP row's lanes; not consumed).

        Returns device logits, ``logits_dtype``: "mesh" ``[1, 1, 32, 6880]`` per chip (rows = all 32 lanes in lane
        order 8 dp + l; chip (r, c) = vocab block r * C + c); "tp" ``[1, 1, 8, 27520]`` (the row's lanes x vocab
        block tp). TILE by default (the input of :meth:`argmax_decode`); ``row_major=True`` untilizes on device (the
        layout :meth:`logits_to_host` reads without a host-side untilize). Trace-safe.

        = :meth:`stream_mean_norm` + :meth:`decode_logits` (``consume=True``), the same ops in the same order."""
        return self.decode_logits(self.stream_mean_norm(x), row_major=row_major, consume=True)

    def decode_logits(self, hn, *, row_major: bool = False, consume: bool = False, halves: int = 1) -> ttnn.Tensor:
        """The decode logits of a normalized hidden ``hn [1, 1, 8, 4096]`` bf16 (this DP row's lanes): "mesh"
        ``ccl.ag_dp_rows`` -> ``[1, 1, 32, 4096]`` -> :meth:`project`; "tp" :meth:`project`. Output as
        :meth:`forward_decode` (TILE, or ROW_MAJOR with ``row_major``). Trace-safe.

        ``hn`` is the post-final-norm hidden (:meth:`stream_mean_norm` of the residual streams) or the MTP layer's
        ``final_layernorm`` output (``tt/mtp.py``; README §17). ``consume=True`` frees ``hn`` (what
        :meth:`forward_decode` does); by default it is kept, so the speculative decode step can feed the same ``hn`` to
        the MTP layer after the main head (features design §3.8.1).

        ``halves=2`` (T64 verify step, "mesh" only; docs/p5_t64/P5_T64_DESIGN.md §4.4): ``hn [1, 1, 16, 4096]`` per DP
        row = ``[8 anchors | 8 drafts]``; the split-order gather ``ccl.ag_dp_rows(hn, halves=2)`` gives ``[1, 1, 64,
        4096]`` and the logits ``[1, 1, 64, 6880]``: rows 0..31 = the anchors in lane order (the 32-lane layout), rows
        32..63 = the drafts, each row bitwise the 32-lane head's row of the same hidden state."""
        h = int(halves)
        if h != 1 and self.vocab_split != "mesh":
            raise ValueError(f"decode_logits(halves={halves}) needs the 'mesh' vocab split (got {self.vocab_split!r})")
        x = hn
        if self.vocab_split == "mesh":
            if h == 1:
                x = self.ccl.ag_dp_rows(hn, memory_config=self.memory_config)  # [1, 1, 32, 4096]
            else:  # [1, 1, 64, 4096]: rows 0..31 anchors (lane order), 32..63 drafts
                x = self.ccl.ag_dp_rows(hn, memory_config=self.memory_config, halves=h)
            if consume:
                ttnn.deallocate(hn)
        logits = self.project(x)
        if x is not hn or consume:
            ttnn.deallocate(x)
        if row_major:
            rm = self.logits_rm(logits)
            ttnn.deallocate(logits)
            return rm
        return logits

    def set_prefill_position(self, last_index: int, seq_len: Optional[int] = None) -> int:
        """Write the tile row holding ``last_index`` (start ``[0, 0, 32 * (last_index // 32), 0]``, end ``start +
        [1, 4, 32, 4096]``) into the persistent device index tensors that :meth:`forward_prefill`'s tensor-args slice
        reads. Two tiny host -> device copies on CQ0 (a buffer write waits for the programs enqueued before it, so a
        prefill still in flight keeps its own position). Not traceable: call it before ``forward_prefill(x)`` or before
        replaying a trace that holds it. Returns the in-tile row ``last_index % 32`` (what
        :meth:`prefill_logits_to_host` picks)."""
        li = int(last_index)
        if li < 0 or (seq_len is not None and li >= int(seq_len)):
            raise ValueError(f"last_index {li} outside [0, {seq_len})")
        r0 = (li // TILE) * TILE
        for dev, vals in ((self._pf_start, (0, 0, r0, 0)), (self._pf_end, (1, self.n_streams, r0 + TILE, self.hidden))):
            host = ttnn.from_torch(torch.tensor(vals, dtype=torch.int32), dtype=ttnn.int32, mesh_mapper=self._rep)
            ttnn.copy_host_to_device_tensor(host, dev)
        return li - r0

    def forward_prefill(self, x, last_index: Optional[int] = None) -> ttnn.Tensor:
        """Prefill head. ``x``: ``X [1, 4, S, 4096]`` TILE, replicated (S = bucket, a multiple of 32; not consumed);
        ``last_index`` = S_real - 1 (written with :meth:`set_prefill_position` first; ``None`` = use the position
        written last, e.g. inside a trace capture).

        The 32-row tile holding ``last_index`` is cut out with the tensor-args slice (position on the device), reduced,
        normed and projected; returns that tile row's logits ``[1, 1, 32, Vc]`` ROW_MAJOR per chip ("mesh": vocab block
        r * C + c; "tp": block tp, all DP rows equal); row ``last_index % 32`` is the token's
        (:meth:`prefill_logits_to_host` picks it). The programs depend on S only: one call per bucket compiles them
        all, and the device ops are trace-safe (no position-dependent program, constants preallocated)."""
        S = int(x.shape[2])
        if S < TILE or S % TILE:
            raise ValueError(f"prefill length {S} must be a positive multiple of {TILE} (a bucket)")
        if tuple(int(d) for d in x.shape) != (1, self.n_streams, S, self.hidden):
            raise ValueError(f"expected X [1, {self.n_streams}, S, {self.hidden}], got {list(x.shape)}")
        if last_index is not None:
            self.set_prefill_position(last_index, S)
        xs = ttnn.slice(
            x, self._pf_start, self._pf_end, slice_dim=2, num_devices=S // TILE, memory_config=self.memory_config
        )  # [1, 4, 32, 4096]: tile row start[2] // 32 (output geometry = S / num_devices rows)
        hn = self.stream_mean_norm(xs)
        ttnn.deallocate(xs)
        logits = self.project(hn)  # [1, 1, 32, Vc]
        ttnn.deallocate(hn)
        rm = ttnn.untilize(logits, memory_config=self.memory_config, use_multicore=True)
        ttnn.deallocate(logits)
        return rm

    # ---- host logits (EMB-3) --------------------------------------------------------------------------------
    def logits_rm(self, logits, *, check_lanes: bool = True, rows: Optional[int] = None) -> ttnn.Tensor:
        """Device untilize of the decode logits (the host then reads the 8 / 32 real rows only, without a host-side
        untilize). Not consumed. ``check_lanes``: require the decode row count ("tp": the row's 8 lanes).

        ``rows`` (T64, "mesh"): untilize only the first ``rows`` rows (a multiple of 32, a tile-aligned slice; one
        program, its bounds are fixed) -- ``rows=32`` on the T64 step's ``[1, 1, 64, 6880]`` logits gives the anchors'
        ``[1, 1, 32, 6880]``, the 32-lane layout :meth:`logits_to_host` reads (the ``wide`` mode's host-sampled steps).
        ``None`` (or all rows) = the whole tensor, as before."""
        n = int(logits.shape[2])
        if rows is not None and int(rows) != n:
            r = int(rows)
            if self.vocab_split != "mesh" or r <= 0 or r % TILE or r > n:
                raise ValueError(
                    f"logits_rm(rows={rows}): a positive multiple of {TILE} up to the {n} logits rows, 'mesh' split "
                    f"only"
                )
            part = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, r, int(logits.shape[3])], memory_config=self.memory_config)
            rm = ttnn.untilize(part, memory_config=self.memory_config, use_multicore=True)
            ttnn.deallocate(part)
            return rm
        if check_lanes and self.vocab_split == "tp" and n != self.lanes:
            raise ValueError(f"expected {self.lanes} lanes per chip, got {list(logits.shape)}")
        return ttnn.untilize(logits, memory_config=self.memory_config, use_multicore=True)

    def _reader(self, t) -> HostShardReader:
        key = HostShardReader.spec_key(t)
        r = self._readers.get(key)
        if r is None:
            r = self._readers[key] = HostShardReader(self.mesh_device, t)
        return r

    def logits_to_host(self, logits, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Decode device logits (:meth:`forward_decode` output; ROW_MAJOR from ``row_major=True`` -- a TILE input is
        untilized on device here first, an eager op) -> **fresh** host tensor ``[32, 220160]`` in lane order (``dtype``
        default: the device dtype, bf16). One concurrent read of every chip into persistent staging
        (:class:`HostShardReader`), then numpy block copies on ``host_copy_threads`` Python threads into the fresh
        output (:meth:`_assemble_decode`). Rows of inactive lanes are garbage (the bridge ignores them). The T64 step's
        64-row logits are refused: read ``logits_rm(logits, rows=32)`` (its anchors, the 32-lane layout)."""
        if self.vocab_split == "mesh" and int(logits.shape[2]) != self.cfg.max_batch:
            raise ValueError(
                f"logits_to_host reads the {self.cfg.max_batch}-lane decode logits, got {list(logits.shape)} (the T64 "
                f"step's 64 rows: pass logits_rm(logits, rows={self.cfg.max_batch}), its anchors)"
            )
        t = self.logits_rm(logits) if logits.layout == ttnn.TILE_LAYOUT else logits
        try:
            reader = self._reader(t)
            views = reader.read(t)
        finally:
            if t is not logits:
                ttnn.deallocate(t)
        return self._assemble_decode(views, dtype, np_views=reader.np_views)

    def _assemble_decode(self, views, dtype, np_views=None) -> torch.Tensor:
        """Per-chip zero-copy views (mesh row-major order) -> fresh ``[32, V]``: numpy block copies (each byte copied
        once) split over ``host_copy_threads`` Python threads (numpy releases the GIL in the copies).

        A parallel ``torch.cat`` leaves torch's OpenMP workers spinning, which starves tt-metal's completion-queue
        reader threads on the next device read (measured: read 0.5 ms -> 2.4-3.2 ms with 32 torch threads), so the
        copy deliberately avoids torch's intra-op pool (4 threads: read 0.37 + copy 0.7 ms; 1 thread: copy ~1.0 ms)."""
        a = self.cfg.axes
        C = a.mesh_shape[1]
        L = self.cfg.max_batch if self.vocab_split == "mesh" else self.lanes
        if self._copy_pool is None and self.host_copy_threads > 1:
            from concurrent.futures import ThreadPoolExecutor

            self._copy_pool = ThreadPoolExecutor(self.host_copy_threads, thread_name_prefix="motif3-logits")
        out = _concat_views(views, L, self.vc, self.vocab_split, a, C, pool=self._copy_pool, np_views=np_views)
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype)
        return out

    def prefill_logits_to_host(self, tile, last_index: int, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """:meth:`forward_prefill` output (the tile row ``[1, 1, 32, Vc]`` per chip; ROW_MAJOR, or TILE -> an eager
        device untilize first) and the same ``last_index`` -> **fresh** host ``[220160]`` = row ``last_index % 32``.

        One concurrent read of every chip into persistent staging (:class:`HostShardReader`; 440 KB per chip for
        "mesh", the same latency-bound ~0.3-0.4 ms as one row), then one row per chip is copied out in vocab order
        ("tp": the chips of DP row 0, one per vocab block)."""
        li = int(last_index)
        if li < 0:
            raise ValueError(f"last_index must be >= 0, got {li}")
        if int(tile.shape[2]) != TILE or int(tile.shape[3]) != self.vc:
            raise ValueError(f"expected the prefill tile row [1, 1, {TILE}, {self.vc}], got {list(tile.shape)}")
        t = self.logits_rm(tile, check_lanes=False) if tile.layout == ttnn.TILE_LAYOUT else tile
        try:
            reader = self._reader(t)
            reader.read(t)
        finally:
            if t is not tile:
                ttnn.deallocate(t)
        out = _prefill_row_from_views(reader.views, li % TILE, self.vc, self.vocab_split, self.cfg.axes,
                                      np_views=reader.np_views)
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype)
        return out

    # ---- on-device greedy ------------------------------------------------------------------------------------
    def argmax_decode(self, logits) -> ttnn.Tensor:
        """Greedy token ids from the TILE decode logits (not consumed); trace-safe, all constants preallocated.

        Returns ``[1, 1, 1, 32]`` uint32 ROW_MAJOR in lane order, identical on every chip ("mesh"), or
        ``[1, 1, 1, 8]`` per DP row = that row's lanes ("tp"). Semantics: lowest vocab index among the maxima of the
        device logits (== ``torch.argmax`` of :meth:`logits_to_host`'s rows).

        The row count L comes from the logits: the 32 lanes ("mesh"; 8 per DP row with "tp"), or the T64 step's 64
        split-order rows ("mesh", :meth:`decode_logits` ``halves=2``; its constants exist when the head was built with
        a T64 config, ``cfg.wide_rows_per_dp``): ``[1, 1, 1, 64]`` = ``(a0 of lanes 0..31, a1 of lanes 0..31)``, each
        id the one the 32-row call computes for the same logits row. Any other L raises.

        Local stage (vectorized, multi-core; ``ttnn.argmax`` on ROW_MAJOR scans each row on one RISC-V: 242 us):
        ``m = max_W(logits)``, ``c = iota + 2^24 * (logits < m)`` in int32, ``idx = min_W(c)``. Cross-shard stage: the
        per-shard (max, idx) pairs, lanes on columns (W-broadcast + transpose: whole tiles for the H all-gathers),
        are gathered over the vocab shards; the token is ``min(idx + offset_shard + 2^24 * (max_shard < max))``."""
        L = int(logits.shape[2])
        if L == self.argmax_lanes:
            am_zeros, am_zeros_i32, am_iota, am_offsets = (
                self._am_zeros, self._am_zeros_i32, self._am_iota, self._am_offsets
            )  # fmt: skip
        elif L in getattr(self, "_am_wide", {}):
            am_zeros, am_zeros_i32, am_iota, am_offsets = self._am_wide[L]
        else:
            raise ValueError(
                f"argmax_decode: no constants for {L} logits rows (have {self.argmax_lanes} and "
                f"{sorted(getattr(self, '_am_wide', {}))}; the T64 step's 64 rows need a head built with a T64 config, "
                f"cfg.wide_rows_per_dp > 0, vocab split 'mesh')"
            )
        mc = self.memory_config
        S = float(TIEBREAK_SENTINEL)
        m = ttnn.max(logits, dim=3, keepdim=True, memory_config=mc)  # [1,1,L,1] bf16 (exact)
        if self.argmax_local == "vector":
            nm = ttnn.lt(logits, m, memory_config=mc)  # 1.0 strictly below the row max
            sm = ttnn.multiply(nm, S, memory_config=mc)  # 0 / 2^24, exact in bf16
            ttnn.deallocate(nm)
            si = ttnn.typecast(sm, ttnn.int32, memory_config=mc)
            ttnn.deallocate(sm)
            c = ttnn.add(si, am_iota, memory_config=mc)  # int32: local index (+ 2^24 off the max)
            ttnn.deallocate(si)
            li = ttnn.min(c, dim=3, keepdim=True, memory_config=mc)  # [1,1,L,1] int32: lowest index of the max
            ttnn.deallocate(c)
        else:  # untilize + ttnn.argmax (strict ">" scan: the lowest index of the max)
            u = ttnn.untilize(logits, memory_config=mc, use_multicore=True)
            a = ttnn.argmax(u, dim=-1, keepdim=True, memory_config=mc)  # [1,1,L,1] uint32 RM
            ttnn.deallocate(u)
            at = ttnn.to_layout(a, ttnn.TILE_LAYOUT, memory_config=mc)
            ttnn.deallocate(a)
            li = ttnn.typecast(at, ttnn.int32, memory_config=mc)
            ttnn.deallocate(at)
        # lanes on columns (rows of each 32-row block all equal; the offsets table masks all but row 32 j)
        vb = ttnn.add(am_zeros, m, memory_config=mc)  # [1,1,L,32]
        ttnn.deallocate(m)
        vg = ttnn.transpose(vb, 2, 3, memory_config=mc)  # [1,1,32,L]
        ttnn.deallocate(vb)
        ib = ttnn.add(am_zeros_i32, li, memory_config=mc)
        ttnn.deallocate(li)
        ig = ttnn.transpose(ib, 2, 3, memory_config=mc)  # [1,1,32,L] int32
        ttnn.deallocate(ib)
        for axis in (("tp", "dp") if self.vocab_split == "mesh" else ("tp",)):
            nv = self.ccl.all_gather(vg, 2, axis, memory_config=mc)
            ni = self.ccl.all_gather(ig, 2, axis, memory_config=mc)
            ttnn.deallocate(vg)
            ttnn.deallocate(ig)
            vg, ig = nv, ni  # [1,1,32 n,L]
        gmax = ttnn.max(vg, dim=2, keepdim=True, memory_config=mc)  # [1,1,1,L]
        not_max = ttnn.lt(vg, gmax, memory_config=mc)
        ttnn.deallocate(vg)
        ttnn.deallocate(gmax)
        sent = ttnn.multiply(not_max, S, memory_config=mc)
        ttnn.deallocate(not_max)
        sent_i = ttnn.typecast(sent, ttnn.int32, memory_config=mc)
        ttnn.deallocate(sent)
        glob = ttnn.add(ig, am_offsets, memory_config=mc)
        ttnn.deallocate(ig)
        masked = ttnn.add(glob, sent_i, memory_config=mc)
        ttnn.deallocate(glob)
        ttnn.deallocate(sent_i)
        tok = ttnn.min(masked, dim=2, keepdim=True, memory_config=mc)  # int32 min: exact (SFPU reduce)
        ttnn.deallocate(masked)
        tok_u = ttnn.typecast(tok, ttnn.uint32, memory_config=mc)
        ttnn.deallocate(tok)
        out = ttnn.untilize_with_unpadding(tok_u, [0, 0, 0, L - 1], memory_config=mc, use_multicore=True)
        ttnn.deallocate(tok_u)
        return out  # [1,1,1,L] uint32 RM

    def close(self) -> None:
        """Release the host-side resources (the copy pool and the staging buffers); device tensors stay owned by the
        caller / are freed with the module."""
        if self._copy_pool is not None:
            self._copy_pool.shutdown(wait=True)
            self._copy_pool = None
        self._readers.clear()

    def tokens_to_host(self, token_ids) -> torch.Tensor:
        """:meth:`argmax_decode` output -> host ``int64 [32]`` in lane order ("mesh": one chip's 128 B; "tp": every
        chip's 32 B through the staging reader, one chip per DP row used). The T64 step's ``[1, 1, 1, 64]`` ("mesh")
        gives ``int64 [64]``: ``[a0 of lanes 0..31 | a1 of lanes 0..31]``."""
        if self.vocab_split == "mesh":
            return ttnn.to_torch(ttnn.get_device_tensors(token_ids)[0]).reshape(-1).to(torch.int64)
        a = self.cfg.axes
        C = a.mesh_shape[1]
        views = self._reader(token_ids).read(token_ids)
        rows = {}
        for i, v in enumerate(views):
            dp, tp = a.roles(*divmod(i, C))
            if tp == 0:
                rows[dp] = v.reshape(-1)
        return torch.cat([rows[dp] for dp in range(a.dp_size)]).to(torch.int64)


__all__ = [
    "DEFAULT_LM_HEAD_PC",
    "FINAL_NORM_WEIGHT",
    "HostShardReader",
    "LM_HEAD_WEIGHT",
    "LOGITS_DTYPES",
    "MotifLMHead",
    "TIEBREAK_SENTINEL",
    "VOCAB_SPLITS",
    "argmax_offsets",
    "assemble_decode_logits",
    "lm_head_device_layout",
    "lm_head_mesh_dims",
    "lm_head_program_config",
    "mcast1d_pc",
    "resolve_logits_dtype",
    "sharded_norm_configs",
    "vocab_block_of_coord",
    "vocab_per_shard",
]
