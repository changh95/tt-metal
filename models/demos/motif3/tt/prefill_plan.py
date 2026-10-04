# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resumed / chunked / packed prefill planning: pure host, torch only (features design §2.2, §3.1, §3.7.2;
packed prefill: docs/p5_t64/P5_T64_DESIGN.md §3).

A prefill *row* is one :class:`~models.demos.motif3.tt.generator_api.PrefillRequest`: tokens at positions
``0 .. e-1``, of which ``[0, s)`` are already in the paged latent cache (``s = request.start`` = vLLM
``num_computed_tokens``, ``e = request.end`` = the end of the chunk vLLM scheduled). This module turns ``(s, e)``
into the chunks the generator runs and builds every host-side table those chunks need. Every alignment and
page-table rule of the features design lives here, so ``tests/unit/test_prefill_plan.py`` checks them exhaustively.

Notation: ``bs`` = KV block size (64), ``A`` = resume alignment = ``lcm(bs, every q_chunk / k_chunk)`` of the sp1
global op over the buckets ``<= cap`` (:func:`resume_alignment`; ``cfg.prefill_resume_alignment``). Gate G9's
per-bucket chunks (:data:`DEFAULT_SP1_GLOBAL_CHUNKS`: 128/128 at C = 128 and C >= 2048, 64/64 at 256-1024) give a
uniform ``A = 128``, so the vLLM budget is ``cap - A = 8064`` (:func:`recommended_budget`). ``tail`` = the SWA tail
(128 = window 129 minus the current key; ``cfg.prefill_swa_tail``), ``cap`` = the span cap (8192: the largest bucket
the generator compiles; ``cfg.max_prefill_span``), ``buckets`` = the prefill buckets ``<= cap``
(``cfg.prefill_span_buckets``).

Rules (features design §3.1):

1. **Write floor** ``w0 = floor(s / bs) * bs``. Full blocks below ``w0`` may be cached and shared by other requests:
   they are never written. The block holding ``s`` (when ``s`` is unaligned) is the request's own partial block
   (vLLM never caches a partial block), so rewriting its rows ``[w0, s)`` is safe.
2. **Compute floor** ``c0 = floor(s / A) * A``, and ``c0 = 0`` when that is below ``tail`` (the SWA tail needs
   ``tail`` real positions before the first sp1 chunk). Rows ``[c0, s)`` are recomputed; their KV is rewritten only
   where ``>= w0``. Always ``c0 <= w0 <= s`` (``A`` is a multiple of ``bs``).
3. **Chunks.** ``[c0, e)`` is cut into consecutive chunks with ``A``-aligned starts. While more than ``cap`` rows
   remain, a full-``cap`` chunk. The remainder ``r`` is one chunk of bucket ``b(r)`` (the smallest bucket ``>= r``),
   or, when the cost model says it is cheaper, a head chunk of the largest bucket ``< r`` followed by the best plan
   of the rest (the same rule, recursively). This generalizes the design's single head split (2200 -> 2048 + 256;
   16,736 -> 8192 + 8192 + 512 are unchanged) to the cases where one split is not enough (6200 -> 4096 + 2048 + 128).
   An sp1 chunk's cost adds gate G9's per-bucket price of its global attention over the cached prefix
   (:func:`prefill_cost_model`). Only the last chunk is padded and only the last chunk runs the LM head (local row
   ``e - 1 - a``). A chunk at start 0 is **sp0** (the draft-1 path: square causal SDPA over the chunk's own rows; no
   cache reads); every other chunk is **sp1** (global layers attend over the paged cache from the chunk start; SWA
   layers read a ``tail``-row tail from the cache).
4. **Tables per chunk** (start ``a``, bucket ``C``):
   * fill table ``[C / bs]``: ``-1`` (the fill kernel skips ``0xFFFFFFFF``) for blocks entirely below ``w0``
     (shared) and for blocks entirely at or past the chunk's end (pure padding), else the block id. Bucket padding
     rows inside the request's own last block are written; decode overwrites them before they are read.
   * SDPA table ``[W']``: the real ids of blocks ``[0, cdiv(chunk.end, bs))``, then **0** (the null block) -- never
     ``-1``: the SDPA reader maps every entry as a block id. ``W' = sdpa_table_width(max_model_len, cap, bs)``
     (640 for 32768 / 8192 / 64), a multiple of 8, fixed: one program per bucket.
   * SWA tail block ids ``[tail / bs]``: the blocks of positions ``[a - tail, a)``.
   * RoPE positions ``min(a + i, max_positions - 1)`` for ``i < C`` (padded rows clamp; their outputs are dropped).
5. **Row order** (§3.7.2). Inside one ``prefill_forward_batch`` call, a row with an sp1 chunk reads its read-only
   prefix (blocks ``[0, w0 / bs)``), which another row of the same call may be writing: vLLM caches full blocks when
   it allocates them, so a later-admitted request can hit blocks an earlier row computes in the same step.
   :func:`order_prefill_requests` runs writers first (stable topological order; raises on a cycle or on two rows
   writing one block).
6. **Packed passes** (P5, design §3.1-§3.3; :func:`plan_prefill_passes`, ``MOTIF3_PACKED_PREFILL``). Every chunk of a
   call is a *segment* (:class:`PackSegment`). A chunk of ``r = end - start`` real rows packs at ``S`` = the smallest
   segment size ``>= r`` (sp0: 64 ... 1024; sp1: 128 ... 1024; review edit R-E4: the chunk's rows, not the row's
   span), else it runs solo. A :class:`PrefillPass` is ``solo`` (one chunk at its own bucket: the paths above,
   bitwise unchanged), ``pk0`` (``B`` sp0 segments) or ``pk1`` (``B`` sp1 segments at one common start ``a``):
   ``B`` a power of two (dummy segments fill it), ``T = B * S`` rows, segment ``k`` at packed rows ``[k S, k S + S)``.
   A segment depends on its row's previous chunk and, when sp1, on every segment of the call that writes a block of
   its row's read-only prefix ``[0, w0)`` (review edit R-E1: when ``c0 < w0`` the global layers read rows
   ``[c0, w0)`` from the cache; the range of rule 5). Passes run level by level, so a pass never holds a segment
   together with one of its dependencies. A segment's tables are the rule-4 tables of its chunk re-bucketed to ``S``
   (R-E12); a pk1 pass gathers its SWA tails once when every segment has the same tail blocks (``shared``), else per
   segment (``distinct``), and the variant is part of the pass shape (R-E2).

No per-lane or per-slot state appears anywhere: a request may change lane between chunks, the paged cache is the
only cross-chunk state.

Import rule: stdlib, torch and ``generator_api`` only (never ttnn): the bridge, its host tests and
``model_config`` import this module.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .generator_api import (
    DEFAULT_PACKED_PREFILL_MAX_SEG,
    DEFAULT_PACKED_PREFILL_MAX_TOKENS,
    MIN_PREFILL_BUCKET,
    PACK_BATCHES,
    PACK_SEG_BUCKETS,
    PACK_SP1_SEG_BUCKETS,
    PACKED_PASS_KINDS,
    PK1_TAIL_VARIANTS,
    cdiv,
    check_block_size,
)

SP0 = "sp0"  # chunk at start 0: draft-1 prefill path (no cache reads)
SP1 = "sp1"  # chunk at start > 0: reads the paged cache (resumed prefill)
PATHS = (SP0, SP1)
DEFAULT_SWA_TAIL = 128  # SWA window 129 keys incl. the current one -> 128 earlier keys
SDPA_TABLE_WIDTH_MULTIPLE = 8  # flexible chunked SDPA: page-table stick W * 4 B must be a multiple of 32 B

# Eager single-row prefill TTFT per bucket on this Galaxy, seconds (features design §3.1). Measured [V
# FULL_MODEL_VALIDATION.md §4.3]: 128, 1024, 4096, 16384, 32768; interpolated [I]: 256, 512, 2048, 8192. Re-measured
# by G9 / CP-L; the generator passes cfg.prefill_cost_table.
DEFAULT_PREFILL_COST_TABLE: Mapping[int, float] = {
    128: 0.642,
    256: 0.66,
    512: 0.75,
    1024: 1.008,
    2048: 1.55,
    4096: 2.912,
    8192: 5.8,
    16384: 11.547,
    32768: 23.396,
}
# sp1 global op (ttnn.transformer.chunked_scaled_dot_product_attention over the paged latent cache): (q_chunk, k_chunk)
# per chunk bucket, as (max bucket, (q, k)) entries, ascending; a bucket C uses the first entry with C <= max bucket.
# Gate G9 (tests/unit/gates/GATES_RESULTS.md §12.2, fp32 dest acc, the fastest config per bucket; lead decision F5):
# 128/128 at C = 128 and C >= 2048, 64/64 at C = 256-1024. A chunk start must be a multiple of its q and k (the kernels
# floor it silently, G9 negative control), so the planner aligns every start to the uniform A = lcm(bs, every q / k up
# to the span cap) = 128 (64/64 serves 128-aligned starts). model_config.SP1_GLOBAL_CHUNKS is this table; the bridge's
# pre-generator default generator_api.DEFAULT_PREFILL_ALIGNMENT must equal resume_alignment(64, the 8192-cap buckets).
DEFAULT_SP1_GLOBAL_CHUNKS: Tuple[Tuple[int, Tuple[int, int]], ...] = (
    (128, (128, 128)),
    (1024, (64, 64)),
    (32768, (128, 128)),
)
# A bf16 latent cache (MOTIF3_KV_CACHE_DTYPE=bf16; bfp8 is the serving default) keeps 64/64 at every bucket (A = 64).
# Its K / V tiles are 2048 B instead of 1088 B, and at 128/128 (fp32 dest acc) the op's static CB region ends at
# 1,572,480 B (measured at C = 2048): through the whole 32 KB L1_SMALL region at the top of the 1.5 MB L1, where the
# CCL global semaphores live. tt-metal checks a CB region only against the full L1 size and the lowest MAIN-L1 buffer,
# so the overlap is silent: test_attention_resumed's bf16-KV schedules returned garbage, lost replica consistency and
# hung in a later CCL (2026-10-03), while the op alone was exact. bfp8 at 128/128 (C = 128 and 2048) and bf16 at 64/64
# measured below an L1 page held at 1,533,824 B, i.e. below L1_SMALL (bf16 at 128/128 also fit at C = 128 only).
SP1_GLOBAL_CHUNKS_BF16_KV: Tuple[Tuple[int, Tuple[int, int]], ...] = ((32768, (64, 64)),)
SP1_GLOBAL_CHUNKS_BY_KV_DTYPE: Mapping[str, Tuple[Tuple[int, Tuple[int, int]], ...]] = {
    "bfp8": DEFAULT_SP1_GLOBAL_CHUNKS,
    "bf16": SP1_GLOBAL_CHUNKS_BF16_KV,
}

# Gate G9's sp1 cost model (GATES_RESULTS §12.2; fp32 acc, eager, the q / k above): one global layer's sp1 attention at
# bucket C behind a cached prefix of s keys takes t0(C) + slope(C) * s. {bucket: (t0 s, slope s per prefix key)}. The
# op streams the whole prefix per (head, q chunk), so short chunks behind long prefixes cost far more per (row, key)
# than the FLOP estimate (C = 128: 4.7e-8 s per row-key over 14 layers, C = 512: 1.4e-8, C >= 1024: 5.2e-9 - 7.4e-9).
DEFAULT_SP1_GLOBAL_COST: Mapping[int, Tuple[float, float]] = {
    128: (0.30e-3, 0.43e-6),
    256: (0.60e-3, 0.50e-6),
    512: (0.38e-3, 0.52e-6),
    1024: (0.50e-3, 0.54e-6),
    2048: (1.22e-3, 0.92e-6),
    4096: (4.21e-3, 1.88e-6),
    8192: (12.28e-3, 3.07e-6),
}
SP1_GLOBAL_LAYERS = 14  # Motif-3: layers l % 4 == 0 of 53

# The pre-G9 single constant [I] (prefill_cost_model(sp1_s_per_row_key=...)): 14 global layers x 10 heads x 2 x (576 +
# 576) FLOP per (row, key) at ~50 TFLOP/s per chip (features design §3.2.2, review R10) = 6.5e-9 s per (chunk row,
# prefix key). G9 measured 6.1-7.7e-9 at C >= 1024 but 1.5e-8 - 4.8e-8 below (DEFAULT_SP1_GLOBAL_COST).
DEFAULT_SP1_ATTN_S_PER_ROW_KEY = 6.5e-9

CostFn = Callable[[int, int], float]  # (bucket, chunk start) -> estimated seconds


# ----------------------------------------------------------------------------------------------------------------
# sp1 global op geometry (gate G9)
# ----------------------------------------------------------------------------------------------------------------
def sp1_global_qk(
    bucket: int, table: Sequence[Tuple[int, Tuple[int, int]]] = DEFAULT_SP1_GLOBAL_CHUNKS
) -> Tuple[int, int]:
    """``(q_chunk, k_chunk)`` of the sp1 global op at chunk bucket ``bucket``: the first ``(max bucket, (q, k))``
    entry of ``table`` with ``bucket <= max bucket``. Both must be positive and ``q`` must divide the bucket.
    ``model_config.sp1_global_chunks`` is the same lookup on the same table, with the device's tile checks."""
    C = int(bucket)
    for upto, (q, k) in table:
        if C <= int(upto):
            q, k = int(q), int(k)
            if C < 1 or q < 1 or k < 1 or C % q:
                raise ValueError(f"sp1 bucket {C} with q/k chunks {(q, k)}: chunks must be positive and q | C")
            return q, k
    raise ValueError(f"no sp1 global chunk entry for bucket {C} in {tuple(table)}")


def sp1_global_chunk_table(kv_cache_dtype: str = "bfp8") -> Tuple[Tuple[int, Tuple[int, int]], ...]:
    """The sp1 global ``(max bucket, (q, k))`` table for a latent-cache dtype name: gate G9's per-bucket table for
    ``"bfp8"`` (the serving default), 64/64 everywhere for ``"bf16"`` (:data:`SP1_GLOBAL_CHUNKS_BF16_KV`)."""
    name = str(kv_cache_dtype)
    if name not in SP1_GLOBAL_CHUNKS_BY_KV_DTYPE:
        known = sorted(SP1_GLOBAL_CHUNKS_BY_KV_DTYPE)
        raise ValueError(f"no sp1 global chunk table for KV cache dtype {name!r}; known {known}")
    return SP1_GLOBAL_CHUNKS_BY_KV_DTYPE[name]


def resume_alignment(
    block_size: int, buckets: Sequence[int], table: Sequence[Tuple[int, Tuple[int, int]]] = DEFAULT_SP1_GLOBAL_CHUNKS
) -> int:
    """``A = lcm(block_size, q, k of every bucket in buckets)`` (``cfg.prefill_resume_alignment`` with ``buckets`` =
    the span buckets): every chunk start a multiple of ``A`` is a valid start for every bucket's sp1 global op. 128 for
    the default table at block 32 or 64 (64 for an all-64/64 table at block 64)."""
    a = int(block_size)
    if a < 1:
        raise ValueError(f"block_size must be positive, got {a}")
    for b in buckets:
        q, k = sp1_global_qk(int(b), table)
        a = math.lcm(a, q, k)
    return a


# ----------------------------------------------------------------------------------------------------------------
# Plans
# ----------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkPlan:
    """One prefill chunk of a row: real rows ``[start, end)``, padded to ``bucket`` rows.

    Attributes:
        start: ``a_k``, absolute position of the chunk's first row (a multiple of ``A``; 0 for sp0).
        bucket: ``C_k``, padded rows (a prefill bucket ``<= cap``).
        end: ``min(e, a_k + C_k)``, one past the last real row (``a_k + C_k`` for every chunk but the last).
        path: ``"sp0"`` if ``start == 0`` else ``"sp1"``.
        last: final chunk of the row (the only one that runs the LM head).
    """

    start: int
    bucket: int
    end: int
    path: str
    last: bool

    @property
    def rows(self) -> int:
        """Real rows of the chunk."""
        return self.end - self.start

    @property
    def padding(self) -> int:
        """Bucket padding rows (0 for every chunk but the last)."""
        return self.bucket - self.rows

    @property
    def head_row(self) -> int:
        """Chunk-local row of the last real token (the LM head's tensor-args row on the last chunk)."""
        return self.end - 1 - self.start

    @property
    def is_sp1(self) -> bool:
        return self.path == SP1


@dataclass(frozen=True)
class RowPlan:
    """The chunks of one prefill row (features design §3.1). ``start``/``end`` are the request's ``s``/``e``."""

    start: int
    end: int
    w0: int  # write floor: floor(s / bs) * bs
    c0: int  # compute floor: floor(s / A) * A, or 0 below the SWA tail
    chunks: Tuple[ChunkPlan, ...]
    block_size: int
    align: int

    @property
    def has_sp1(self) -> bool:
        """The row reads the paged cache (some chunk is sp1): it depends on the writers of its read-only prefix."""
        return any(c.is_sp1 for c in self.chunks)

    @property
    def recompute(self) -> int:
        """Rows ``[c0, s)`` computed again although they are in the cache."""
        return self.start - self.c0

    @property
    def read_only_blocks(self) -> int:
        """Leading logical blocks ``[0, w0 / bs)`` the row never writes (full, maybe shared)."""
        return self.w0 // self.block_size

    @property
    def write_blocks(self) -> range:
        """Logical blocks the row writes: ``[w0 / bs, cdiv(e, bs))`` (the last one may be the own partial block)."""
        return range(self.w0 // self.block_size, cdiv(self.end, self.block_size))

    @property
    def buckets(self) -> Tuple[int, ...]:
        return tuple(c.bucket for c in self.chunks)

    @property
    def paths(self) -> Tuple[str, ...]:
        return tuple(c.path for c in self.chunks)

    @property
    def last_chunk(self) -> ChunkPlan:
        return self.chunks[-1]


# ----------------------------------------------------------------------------------------------------------------
# Cost model
# ----------------------------------------------------------------------------------------------------------------
def table_cost(table: Mapping[int, float], bucket: int) -> float:
    """``table[bucket]``, else piecewise-linear interpolation between the neighbouring buckets (a non-power-of-two
    last bucket such as ``max_model_len = 6144``), linear extrapolation per token above the largest entry and the
    smallest entry's value below it."""
    if not table:
        raise ValueError("empty prefill cost table")
    b = int(bucket)
    if b in table:
        return float(table[b])
    keys = sorted(int(k) for k in table)
    if b < keys[0]:
        return float(table[keys[0]])
    if b > keys[-1]:
        return float(table[keys[-1]]) * b / keys[-1]
    hi = next(k for k in keys if k > b)
    lo = max(k for k in keys if k < b)
    t = (b - lo) / (hi - lo)
    return float(table[lo]) * (1.0 - t) + float(table[hi]) * t


def sp1_prefix_cost(
    bucket: int,
    start: int,
    *,
    sp1_global_cost: Optional[Mapping[int, Tuple[float, float]]] = None,
    global_layers: int = SP1_GLOBAL_LAYERS,
) -> float:
    """Seconds an sp1 chunk of ``bucket`` rows at ``start`` spends attending over its cached prefix in the global
    layers: ``global_layers * slope(C) * start`` with gate G9's per-bucket slope (:data:`DEFAULT_SP1_GLOBAL_COST`;
    piecewise-linear between measured buckets, per token above the largest). 0 at ``start = 0`` (sp0). The fixed part
    ``t0(C)`` is not added: the bucket's table cost already holds the chunk's own causal attention (the absorbed op
    costs ``t0 - sp0`` more, about 0.12 s per 8192-row chunk over 14 layers and <= 0.04 s below 8192)."""
    if int(start) <= 0:
        return 0.0
    slopes = _sp1_slopes(DEFAULT_SP1_GLOBAL_COST if sp1_global_cost is None else sp1_global_cost)
    return float(global_layers) * table_cost(slopes, int(bucket)) * int(start)


def _sp1_slopes(model: Mapping[int, Tuple[float, float]]) -> Dict[int, float]:
    """``{bucket: slope}`` of an sp1 cost model ``{bucket: (t0, slope)}`` (validated)."""
    if not model or any(int(c) < 1 or float(t0) < 0 or float(sl) < 0 for c, (t0, sl) in model.items()):
        raise ValueError(f"sp1_global_cost must map buckets to non-negative (t0, slope) pairs, got {dict(model)}")
    return {int(c): float(sl) for c, (_, sl) in model.items()}


def prefill_cost_model(
    table: Optional[Mapping[int, float]] = None,
    *,
    sp1_s_per_row_key: Optional[float] = None,
    sp1_global_cost: Optional[Mapping[int, Tuple[float, float]]] = None,
    global_layers: int = SP1_GLOBAL_LAYERS,
) -> CostFn:
    """The default chunk cost: ``table_cost(table, C)`` (eager single-row TTFT per bucket) plus, for an sp1 chunk at
    ``a > 0``, its global attention over the cached prefix: gate G9's per-bucket model :func:`sp1_prefix_cost`
    (default), or, when ``sp1_s_per_row_key`` is given, the pre-G9 single constant ``sp1_s_per_row_key * C * a``
    (``0.0`` drops the prefix term)."""
    t = dict(DEFAULT_PREFILL_COST_TABLE if table is None else table)
    if sp1_s_per_row_key is not None:
        k = float(sp1_s_per_row_key)
        if k < 0:
            raise ValueError(f"sp1_s_per_row_key must be >= 0, got {k}")

        def cost(bucket: int, start: int) -> float:
            return table_cost(t, bucket) + (k * bucket * start if start > 0 else 0.0)

        return cost
    slopes = _sp1_slopes(DEFAULT_SP1_GLOBAL_COST if sp1_global_cost is None else sp1_global_cost)
    layers = int(global_layers)
    if layers < 0:
        raise ValueError(f"global_layers must be >= 0, got {layers}")

    def cost(bucket: int, start: int) -> float:
        prefix = layers * table_cost(slopes, bucket) * start if start > 0 else 0.0
        return table_cost(t, bucket) + prefix

    return cost


def _cost_fn(cost: Union[None, Mapping[int, float], CostFn]) -> CostFn:
    if cost is None:
        return prefill_cost_model()
    if isinstance(cost, Mapping):
        return prefill_cost_model(cost)
    if callable(cost):
        return cost
    raise TypeError(f"cost must be None, a {{bucket: seconds}} mapping or a callable (bucket, start), got {cost!r}")


# ----------------------------------------------------------------------------------------------------------------
# Row planning
# ----------------------------------------------------------------------------------------------------------------
def span_buckets(max_seq_len: int, span_cap: int, min_bucket: int = MIN_PREFILL_BUCKET) -> Tuple[int, ...]:
    """The prefill buckets a span-capped generator compiles: powers of two from ``min_bucket`` up to
    ``min(span_cap, max_seq_len)``; the last is ``max_seq_len`` itself when that is the smaller one."""
    cap = min(int(span_cap), int(max_seq_len))
    out, b = [], int(min_bucket)
    while b < cap:
        out.append(b)
        b *= 2
    out.append(cap)
    return tuple(out)


def sdpa_table_width(max_model_len: int, span_cap: int, block_size: int) -> int:
    """``W'`` of the sp1 SDPA page table: ``round_up(cdiv(max_model_len + span_cap, bs), 8)`` (640 for 32768 / 8192 /
    64, 1280 for block 32). Covers ``a + C`` for every chunk (``a < max_model_len``, ``C <= span_cap``)."""
    n = cdiv(int(max_model_len) + int(span_cap), int(block_size))
    m = SDPA_TABLE_WIDTH_MULTIPLE
    return -(-n // m) * m


def _check_geometry(block_size: int, align: int, buckets: Sequence[int], span_cap: int, swa_tail: int):
    bs, A, cap, tail = int(block_size), int(align), int(span_cap), int(swa_tail)
    if bs < 1:
        raise ValueError(f"block_size must be positive, got {bs}")
    if A < 1 or A % bs:
        raise ValueError(f"align {A} must be a positive multiple of the block size {bs} (A = lcm(bs, q, k))")
    if tail < 0 or tail % bs:
        raise ValueError(f"swa_tail {tail} must be a non-negative multiple of the block size {bs} (whole tail blocks)")
    bl = tuple(int(b) for b in buckets)
    if not bl or any(b2 <= b1 for b1, b2 in zip(bl, bl[1:])) or bl[0] < 1:
        raise ValueError(f"buckets must be strictly increasing positive sizes, got {bl}")
    if cap not in bl:
        raise ValueError(f"span_cap {cap} must be one of the buckets {bl}")
    usable = tuple(b for b in bl if b <= cap)
    bad = [b for b in usable if b % A]
    if bad:
        raise ValueError(f"buckets {bad} are not multiples of the alignment {A}: chunk starts would lose alignment")
    return bs, A, cap, tail, usable


def _smallest_fit(usable: Tuple[int, ...], r: int) -> int:
    for b in usable:
        if b >= r:
            return b
    raise AssertionError(f"no bucket >= {r} in {usable}")  # r <= cap is guaranteed by the caller


def _plan_remainder(a: int, r: int, usable: Tuple[int, ...], cost: CostFn) -> Tuple[float, List[Tuple[int, int]]]:
    """Best ``[(start, bucket), ...]`` covering ``[a, a + r)``, ``0 < r <= cap``: one padded chunk, or the largest
    bucket ``< r`` as a full head chunk plus the best plan of the rest. Ties keep the single chunk (fewer chunks)."""
    fit = _smallest_fit(usable, r)
    best = (float(cost(fit, a)), [(a, fit)])
    heads = [b for b in usable if b < r]
    if heads:
        head = heads[-1]
        rest_cost, rest = _plan_remainder(a + head, r - head, usable, cost)
        split = float(cost(head, a)) + rest_cost
        if split < best[0]:
            best = (split, [(a, head)] + rest)
    return best


def plan_prefill_row(
    start: int,
    end: int,
    *,
    block_size: int,
    align: int,
    buckets: Sequence[int],
    span_cap: int,
    swa_tail: int = DEFAULT_SWA_TAIL,
    cost: Union[None, Mapping[int, float], CostFn] = None,
) -> RowPlan:
    """Plan one prefill row ``[start, end)`` (module docstring rules 1-3).

    Args:
        start: ``s``, positions already in the cache (``0 <= s < e``).
        end: ``e``, one past the last position to prefill (the logits are for ``e - 1``).
        block_size: KV block size ``bs``.
        align: ``A``, a multiple of ``bs``; every chunk start is a multiple of it.
        buckets: the prefill buckets (strictly increasing, each a multiple of ``A`` up to ``span_cap``); buckets above
            ``span_cap`` are ignored.
        span_cap: the largest bucket a chunk may use (one of ``buckets``).
        swa_tail: SWA tail rows (128); ``c0`` is 0 below it.
        cost: ``None`` (the default model), a ``{bucket: seconds}`` table, or ``cost(bucket, start) -> seconds``.
    """
    s, e = int(start), int(end)
    if not 0 <= s < e:
        raise ValueError(f"prefill row needs 0 <= start < end, got start={s} end={e}")
    bs, A, cap, tail, usable = _check_geometry(block_size, align, buckets, span_cap, swa_tail)
    cost_fn = _cost_fn(cost)
    w0 = s // bs * bs
    c0 = s // A * A
    if c0 < tail:
        c0 = 0
    spans: List[Tuple[int, int]] = []
    a = c0
    while e - a > cap:
        spans.append((a, cap))
        a += cap
    spans += _plan_remainder(a, e - a, usable, cost_fn)[1]
    chunks = tuple(
        ChunkPlan(start=a_k, bucket=b_k, end=min(e, a_k + b_k), path=SP0 if a_k == 0 else SP1, last=k == len(spans) - 1)
        for k, (a_k, b_k) in enumerate(spans)
    )
    return RowPlan(start=s, end=e, w0=w0, c0=c0, chunks=chunks, block_size=bs, align=A)


def plan_cost(plan: RowPlan, cost: Union[None, Mapping[int, float], CostFn] = None) -> float:
    """Estimated seconds of a plan under ``cost`` (the planner's objective)."""
    fn = _cost_fn(cost)
    return float(sum(fn(c.bucket, c.start) for c in plan.chunks))


# ----------------------------------------------------------------------------------------------------------------
# Per-chunk tables
# ----------------------------------------------------------------------------------------------------------------
def _row(page_table_row: Any, name: str = "page_table_row") -> torch.Tensor:
    pt = torch.as_tensor(page_table_row)
    if pt.ndim != 1 or pt.dtype.is_floating_point or pt.dtype == torch.bool:
        raise TypeError(f"{name} must be a 1-D integer tensor of block ids, got {pt.dtype} {tuple(pt.shape)}")
    return pt


def _real_ids(pt: torch.Tensor, lo: int, hi: int, what: str) -> torch.Tensor:
    if hi > pt.shape[0]:
        raise ValueError(f"{what} needs page-table entries [{lo}, {hi}), the row has {pt.shape[0]}")
    ids = pt[lo:hi].to(torch.int32)
    if ids.numel() and int(ids.min()) < 1:
        raise ValueError(f"{what}: a real position maps to block id {int(ids.min())} (the null block or padding)")
    return ids


def fill_table(page_table_row: Any, chunk: ChunkPlan, w0: int, block_size: int) -> torch.Tensor:
    """``paged_fill_cache`` table of one chunk: ``int32 [C / bs]``. Entry ``j`` (logical block ``a / bs + j``) is the
    block id when that block overlaps ``[w0, chunk.end)``, else ``-1`` (skip): blocks entirely below ``w0`` are
    shared / read-only, blocks entirely at or past ``chunk.end`` hold only bucket padding."""
    pt, bs = _row(page_table_row), int(block_size)
    a, C, end, w0 = int(chunk.start), int(chunk.bucket), int(chunk.end), int(w0)
    if a % bs or C % bs or w0 % bs:
        raise ValueError(f"chunk start {a}, bucket {C} and w0 {w0} must be multiples of the block size {bs}")
    first, n = a // bs, C // bs
    lo = max(first, w0 // bs)  # first written logical block
    hi = min(first + n, cdiv(end, bs))  # one past the last written logical block
    out = torch.full((n,), -1, dtype=torch.int32)
    if hi > lo:
        out[lo - first : hi - first] = _real_ids(pt, lo, hi, "fill table")
    return out


def sdpa_table(page_table_row: Any, end: int, block_size: int, width: int) -> torch.Tensor:
    """sp1 SDPA page table ``int32 [width]``: the real ids of blocks ``[0, cdiv(end, bs))``, then 0 (null block).
    Never ``-1``: the SDPA reader maps every entry as a block id (padded query rows read up to ``a + C``, so
    ``width * bs`` must cover ``a + C``: use :func:`sdpa_table_width`). ``width`` must be a multiple of 8."""
    pt, bs, W = _row(page_table_row), int(block_size), int(width)
    n = cdiv(int(end), bs)
    if W % SDPA_TABLE_WIDTH_MULTIPLE:
        raise ValueError(f"SDPA page-table width {W} must be a multiple of {SDPA_TABLE_WIDTH_MULTIPLE}")
    if W < n:
        raise ValueError(f"SDPA page-table width {W} < the {n} blocks of positions [0, {end})")
    out = torch.zeros(W, dtype=torch.int32)
    out[:n] = _real_ids(pt, 0, n, "SDPA table")
    return out


def tail_blocks(page_table_row: Any, chunk: ChunkPlan, block_size: int, tail: int = DEFAULT_SWA_TAIL) -> torch.Tensor:
    """sp1 SWA tail: ``int32 [tail / bs]`` block ids of positions ``[a - tail, a)`` in position order (for bs 64:
    ``page_table[a/bs - 2], page_table[a/bs - 1]``). All SWA layers share them."""
    pt, bs, a, t = _row(page_table_row), int(block_size), int(chunk.start), int(tail)
    if chunk.path != SP1:
        raise ValueError("tail blocks exist for sp1 chunks only")
    if t <= 0 or t % bs or a % bs:
        raise ValueError(f"tail {t} and chunk start {a} must be positive multiples of the block size {bs}")
    if a < t:
        raise ValueError(f"sp1 chunk at {a} has fewer than {t} positions before it (c0 must be >= the tail)")
    return _real_ids(pt, (a - t) // bs, a // bs, "SWA tail").clone()


def rope_positions(chunk: ChunkPlan, max_positions: int) -> torch.Tensor:
    """RoPE table rows of the chunk: ``int32 [C]`` = ``min(a + i, max_positions - 1)`` (padded rows clamp; their
    outputs are dropped). Real rows never clamp (``chunk.end <= max_positions`` is checked)."""
    a, C, P = int(chunk.start), int(chunk.bucket), int(max_positions)
    if chunk.end > P:
        raise ValueError(f"chunk rows up to {chunk.end} exceed the {P}-row RoPE tables")
    return torch.arange(a, a + C, dtype=torch.int32).clamp_(max=P - 1)


@dataclass(frozen=True)
class ChunkTables:
    """Every host table one chunk needs (built by :func:`chunk_tables`). ``sdpa`` and ``tail`` are None for sp0."""

    fill: torch.Tensor  # int32 [C / bs], -1 = skip
    rope: torch.Tensor  # int32 [C]
    sdpa: Optional[torch.Tensor]  # int32 [W'], 0-padded
    tail: Optional[torch.Tensor]  # int32 [tail / bs]


def chunk_tables(
    page_table_row: Any,
    plan: RowPlan,
    chunk: ChunkPlan,
    *,
    sdpa_width: int,
    max_positions: int,
    swa_tail: int = DEFAULT_SWA_TAIL,
) -> ChunkTables:
    """:func:`fill_table`, :func:`rope_positions` and, for sp1, :func:`sdpa_table` (through ``chunk.end``) and
    :func:`tail_blocks` of one chunk of ``plan``."""
    if chunk not in plan.chunks:
        raise ValueError("chunk does not belong to plan")
    bs = plan.block_size
    sdpa = tail = None
    if chunk.is_sp1:
        if int(sdpa_width) * bs < chunk.start + chunk.bucket:
            raise ValueError(f"SDPA width {sdpa_width} x {bs} does not cover the chunk's {chunk.start + chunk.bucket}")
        sdpa = sdpa_table(page_table_row, chunk.end, bs, sdpa_width)
        tail = tail_blocks(page_table_row, chunk, bs, swa_tail)
    return ChunkTables(
        fill=fill_table(page_table_row, chunk, plan.w0, bs),
        rope=rope_positions(chunk, max_positions),
        sdpa=sdpa,
        tail=tail,
    )


# ----------------------------------------------------------------------------------------------------------------
# Row order inside one call (same-step prefix hits)
# ----------------------------------------------------------------------------------------------------------------
def order_prefill_requests(requests: Sequence[Any], plans: Sequence[RowPlan], block_size: int) -> List[int]:
    """Writer-first execution order of one call's rows (features design §3.7.2): row ``j`` runs after row ``i`` when
    ``j`` has an sp1 chunk and a block of its read-only prefix ``page_table_j[: w0_j / bs]`` is one that ``i`` writes
    (``page_table_i[w0_i / bs : cdiv(e_i, bs)]``). sp0-only rows read nothing from the cache. Topological order,
    stable in input order (the smallest ready index first).

    ``requests`` are ``PrefillRequest``-like (``.start``, ``.seq_len``, ``.page_table``), ``plans[i]`` their row plans.

    Raises ``ValueError`` on: a plan that does not match its request; a real position on block id < 1; a block id
    twice in one row; two rows writing the same block (no legal vLLM schedule does that); a dependency cycle (a row
    can only hit blocks cached before its own admission, so a cycle means a broken caller)."""
    bs = int(block_size)
    if bs < 1:
        raise ValueError(f"block_size must be positive, got {bs}")
    n = len(requests)
    if len(plans) != n:
        raise ValueError(f"{n} requests but {len(plans)} plans")
    tables: List[List[int]] = []
    writer: Dict[int, int] = {}
    for i, (req, plan) in enumerate(zip(requests, plans)):
        if (int(req.start), int(req.seq_len)) != (plan.start, plan.end) or plan.block_size != bs:
            raise ValueError(
                f"row {i}: plan (start {plan.start}, end {plan.end}, bs {plan.block_size}) does not match the request "
                f"(start {int(req.start)}, end {int(req.seq_len)}, bs {bs})"
            )
        ids = _real_ids(_row(req.page_table), 0, cdiv(plan.end, bs), f"row {i}").tolist()
        if len(set(ids)) != len(ids):
            raise ValueError(f"row {i}: a block id appears twice in positions [0, {plan.end})")
        tables.append(ids)
        for b in ids[plan.read_only_blocks :]:
            if b in writer:
                raise ValueError(f"rows {writer[b]} and {i} both write block {b} in one call")
            writer[b] = i
    children: List[List[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for j, plan in enumerate(plans):
        if not plan.has_sp1:
            continue
        deps = {writer[b] for b in tables[j][: plan.read_only_blocks] if b in writer}
        deps.discard(j)
        for i in deps:
            children[i].append(j)
            indeg[j] += 1
    ready = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(ready)
    order: List[int] = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for j in children[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(ready, j)
    if len(order) != n:
        stuck = sorted(set(range(n)) - set(order))
        raise ValueError(f"prefill rows {stuck} read each other's blocks (dependency cycle)")
    return order


def plan_prefill_batch(
    requests: Sequence[Any],
    *,
    block_size: int,
    align: int,
    buckets: Sequence[int],
    span_cap: int,
    swa_tail: int = DEFAULT_SWA_TAIL,
    cost: Union[None, Mapping[int, float], CostFn] = None,
) -> Tuple[List[RowPlan], List[int]]:
    """``([plan_prefill_row(r.start, r.seq_len, ...) for r in requests], order_prefill_requests(...))``."""
    plans = [
        plan_prefill_row(
            int(r.start),
            int(r.seq_len),
            block_size=block_size,
            align=align,
            buckets=buckets,
            span_cap=span_cap,
            swa_tail=swa_tail,
            cost=cost,
        )
        for r in requests
    ]
    return plans, order_prefill_requests(requests, plans, block_size)


# ----------------------------------------------------------------------------------------------------------------
# Packed multi-row prefill (P5; docs/p5_t64/P5_T64_DESIGN.md §3.1-§3.5; module docstring rule 6)
# ----------------------------------------------------------------------------------------------------------------
SOLO = "solo"  # one chunk alone at its own bucket: the single-chunk paths, bitwise unchanged
PK0, PK1 = PACKED_PASS_KINDS  # "pk0": B sp0 segments; "pk1": B sp1 segments at one common start
PASS_KINDS = (SOLO, PK0, PK1)
SHARED_TAILS, DISTINCT_TAILS = PK1_TAIL_VARIANTS  # pk1 SWA tails gathered once / per segment (review edit R-E2)
# A level holds at most one segment per row (a row's chunks form a chain) and a call at most NUM_LANES = 32 rows.
DEFAULT_PACK_MAX_BATCH = max(PACK_BATCHES)

PassShape = Tuple[Any, ...]  # solo ("sp0" | "sp1", bucket); ("pk0", T, S); ("pk1", T, S, "shared" | "distinct")


@dataclass(frozen=True)
class PackSegment:
    """One chunk of one row of a call, as the packed planner sees it (design §3.3 step 1; :func:`segment_dag`).

    Attributes:
        row: the row's index in the call (``requests[row]``, ``plans[row]``).
        chunk_index: ``k``: the chunk is ``plans[row].chunks[k]``.
        path: the chunk's path, ``"sp0"`` (start 0) or ``"sp1"``.
        start: ``a``, the chunk's first position.
        end: one past the chunk's last real row.
        rows: ``S``, the segment rows the chunk packs at (:func:`segment_rows`), or None when it cannot pack (it then
            runs solo at its own bucket).
        last: the row's last chunk; its segment runs the LM head.
    """

    row: int
    chunk_index: int
    path: str
    start: int
    end: int
    rows: Optional[int]
    last: bool

    def __post_init__(self):
        if self.path not in PATHS:
            raise ValueError(f"segment path must be one of {PATHS}, got {self.path!r}")
        if int(self.row) < 0 or int(self.chunk_index) < 0:
            raise ValueError(f"segment row {self.row} and chunk index {self.chunk_index} must be >= 0")
        if not 0 <= int(self.start) < int(self.end):
            raise ValueError(f"a segment needs 0 <= start < end, got start={self.start} end={self.end}")
        if (int(self.start) == 0) != (self.path == SP0):
            raise ValueError(f"an sp0 segment starts at 0 and an sp1 segment later, got {self.path} at {self.start}")
        if self.rows is not None and int(self.rows) < self.real_rows:
            raise ValueError(f"segment rows {self.rows} < the chunk's {self.real_rows} real rows")

    @property
    def key(self) -> Tuple[int, int]:
        """``(row, chunk_index)``, unique within a call."""
        return (int(self.row), int(self.chunk_index))

    @property
    def real_rows(self) -> int:
        """The chunk's real rows ``end - start``."""
        return int(self.end) - int(self.start)

    @property
    def head_row_local(self) -> int:
        """Segment-local row of the last real token (the LM head reads it on the row's last chunk)."""
        return int(self.end) - 1 - int(self.start)

    @property
    def is_sp1(self) -> bool:
        return self.path == SP1


@dataclass(frozen=True)
class PrefillPass:
    """One eager run of the prefill layers inside a ``prefill_forward_batch`` call (design §3.2;
    :func:`plan_prefill_passes`).

    Attributes:
        kind: ``"solo"`` (one chunk at its own bucket), ``"pk0"`` (sp0 segments) or ``"pk1"`` (sp1 segments at one
            common start).
        segments: the real segments in packed order: segment ``k`` occupies packed rows ``[k S, k S + S)``, its real
            rows ``[k S, k S + end - start)``.
        seg_rows: ``S`` (packed); the chunk's bucket (solo).
        batch: ``B``, one of ``generator_api.PACK_BATCHES`` (packed; dummy segments fill ``[len(segments), B)``); 1
            (solo).
        start: the pk1 common start ``a``; 0 (pk0); the chunk's start (solo).
        tails: pk1 only: the SWA tail variant (review edit R-E2), ``"shared"`` (every segment has the same tail blocks:
            one gather + repeat) or ``"distinct"`` (a gather per segment). The variant selects programs, so it is part
            of :attr:`shape`.
        fallback: on a solo pass the shape filter made from a packed pass whose shape was not warmed: that packed
            shape (the generator counts it as ``packed_solo_fallbacks`` and logs it once per shape); else None.

    A dummy segment writes nothing (fill entries -1) and its outputs are dropped. It holds pad tokens and copies
    segment 0's RoPE rows and, in a pk1 pass, segment 0's SDPA row and tail blocks (R-E2), so it reads only what
    segment 0 reads and keeps a pass whose real segments share their tails on the ``shared`` variant.
    """

    kind: str
    segments: Tuple[PackSegment, ...]
    seg_rows: int
    batch: int
    start: int
    tails: Optional[str] = None
    fallback: Optional[PassShape] = None

    def __post_init__(self):
        object.__setattr__(self, "segments", tuple(self.segments))
        segs, S, B, a = self.segments, int(self.seg_rows), int(self.batch), int(self.start)
        if self.kind not in PASS_KINDS:
            raise ValueError(f"pass kind must be one of {PASS_KINDS}, got {self.kind!r}")
        if not segs or not all(isinstance(g, PackSegment) for g in segs):
            raise ValueError("a pass holds one or more PackSegment")
        if len({g.key for g in segs}) != len(segs):
            raise ValueError(f"a pass holds a segment twice: {[g.key for g in segs]}")
        if S < 1 or any(g.real_rows > S for g in segs):
            raise ValueError(f"segment rows {S} must hold every segment's real rows {[g.real_rows for g in segs]}")
        if self.kind == SOLO:
            if B != 1 or len(segs) != 1 or a != segs[0].start or self.tails is not None:
                raise ValueError("a solo pass holds one segment at batch 1, starts at its chunk, has no tail variant")
            if self.fallback is not None and (len(self.fallback) < 3 or self.fallback[0] not in PACKED_PASS_KINDS):
                raise ValueError(f"a fallback is the packed shape the solo pass came from, got {self.fallback!r}")
            return
        if self.fallback is not None:
            raise ValueError("only a solo pass has a fallback shape")
        if B not in PACK_BATCHES or len(segs) > B:
            raise ValueError(f"a packed pass has a batch in {PACK_BATCHES} >= its {len(segs)} segments, got {B}")
        path = SP0 if self.kind == PK0 else SP1
        if any(g.path != path for g in segs):
            raise ValueError(f"a {self.kind} pass holds {path} segments only, got {[g.path for g in segs]}")
        if self.kind == PK0:
            if a != 0 or self.tails is not None:
                raise ValueError("a pk0 pass starts at 0 and has no tail variant")
            return
        if any(g.start != a for g in segs):  # the batched chunked SDPA takes ONE start (design R-P4)
            raise ValueError(f"a pk1 pass shares one start {a}, got segment starts {[g.start for g in segs]}")
        if self.tails not in PK1_TAIL_VARIANTS:
            raise ValueError(f"a pk1 pass needs its tail variant, one of {PK1_TAIL_VARIANTS}, got {self.tails!r}")

    @property
    def tokens(self) -> int:
        """``T = B * S``: the pass's rows (the bucket its row-local programs run at)."""
        return int(self.batch) * int(self.seg_rows)

    @property
    def path(self) -> str:
        """The segments' path: ``"sp0"`` (pk0 / solo sp0) or ``"sp1"`` (pk1 / solo sp1)."""
        return self.segments[0].path

    @property
    def is_packed(self) -> bool:
        return self.kind != SOLO

    @property
    def shape(self) -> PassShape:
        """The key that selects the pass's programs: solo ``(path, bucket)`` (the generator's prefill shape), pk0
        ``("pk0", T, S)``, pk1 ``("pk1", T, S, tails)`` (R-E2). Packed keys are those of
        ``MotifTTConfig.packed_prefill_shapes()`` (:func:`packed_pass_shapes`)."""
        if self.kind == SOLO:
            return (self.path, int(self.seg_rows))
        if self.kind == PK0:
            return (PK0, self.tokens, int(self.seg_rows))
        return (PK1, self.tokens, int(self.seg_rows), self.tails)

    @property
    def dummies(self) -> int:
        """Dummy segments ``B - len(segments)``."""
        return int(self.batch) - len(self.segments)

    @property
    def real_rows(self) -> int:
        return sum(g.real_rows for g in self.segments)

    @property
    def padding_rows(self) -> int:
        """Rows of the pass that hold no real token: segment padding and dummy segments."""
        return self.tokens - self.real_rows

    def offset(self, k: int) -> int:
        """First packed row of segment ``k`` (``0 <= k < B``, dummies included): ``k * S``."""
        k = int(k)
        if not 0 <= k < int(self.batch):
            raise IndexError(f"segment {k} outside the pass's {self.batch} segments")
        return k * int(self.seg_rows)

    def head_rows(self) -> List[Tuple[int, int]]:
        """``[(k, packed row)]``, ascending ``k``, for every segment that is its row's last chunk: the LM head reads
        packed row ``k S + end - 1 - start``, and the logits are those of row ``segments[k].row``."""
        S = int(self.seg_rows)
        return [(k, k * S + g.head_row_local) for k, g in enumerate(self.segments) if g.last]

    def describe(self) -> str:
        """A one-line summary for logs."""
        if self.kind == SOLO:
            g = self.segments[0]
            why = f", fallback from {self.fallback}" if self.fallback is not None else ""
            return f"solo {g.path} C={self.seg_rows} a={g.start} (row {g.row} chunk {g.chunk_index}{why})"
        tails = f" tails={self.tails}" if self.kind == PK1 else ""
        return (
            f"{self.kind} T={self.tokens} S={self.seg_rows} B={self.batch} ({len(self.segments)} real) "
            f"a={self.start}{tails}"
        )


@dataclass(frozen=True)
class PackedCostModel:
    """What :func:`packed_pass_cost` adds to the row cost model's ``cost(bucket, start)`` (design §3.3 step 4;
    calibrated on GATES_RESULTS §13.3 (gate G15a-rest), P5N §14, GATES_RESULTS §12.4 (G10) and the P5 review's
    every-packed-shape probe, ``logs/dev/20261003_193703_p5rev_all_shapes.log``). A pass of ``T`` rows at start ``a`` (0
    for pk0) costs ``cost(T, a)``: the bucket-T chunk, whose sp1 prefix term prices pk1's batched global attention
    (51 ms measured at B = 32, S = 128, a = 2048 against 54 ms modelled), plus

    * ``head_s`` per segment that is its row's last chunk (one LM-head call and host read; solo chunks pay it too);
    * :meth:`extras_s`: the CN transposes around the batched SDPA, ``layers * transpose_s_per_layer * max(1, T /
      transpose_ref_tokens)``, and one MTP KV-only fill ``mtp_fill_s``. The 4 transposes of a layer cost ~0.4 ms of
      eager host dispatch (G15a-rest (a): 0.32-0.42 ms at T 1024-8192) and 115 / 397 us of device time at T 2048 /
      8192 (G15a-rest (d), traced). An eager pass overlaps the two, so it pays about the larger: ~0.4 ms per layer at
      ``1024 <= T <= 8192`` (0.5-0.7 ms of host dispatch at T <= 256: at most ~18 ms more per pass, which decides
      nothing). ``transpose_ref_tokens = 8192`` keeps the term flat at ~21 ms up to the 8192-row pass cap (P5 review,
      finding 6: the earlier reference of 2048 priced 42 / 85 ms at T 4096 / 8192 against 21 ms; over 3093 host-planned
      calls the recalibration changed no pass shape, only equal-cost tie-breaks);
    * pk1, :meth:`tails_s`: the SWA tail gathers. ``shared``: ``swa_layers * tail_gather_s`` (one 2-block gather per
      layer: 4 dispatches, 0.61-0.72 ms measured). ``distinct``: ``swa_layers * (tail_gather_s + 2 B * tail_op_s)``, a
      gather's work plus the ``2 B`` tensor-args slices (and the concat that replaces the ``repeat``): G15a-rest (c),
      eager per SWA layer, distinct minus shared = +3.92 / +0.74 / +0.04 ms at B = 32 / 8 / 2, ~0.06 ms per slice, so
      ~0.18 s per pass at B = 32 (the earlier ``(2 B + 2) * 0.15 ms`` priced 0.39 s). A solo sp1 chunk pays one
      shared-size gather. In the full model that premium is host dispatch, which a device-bound pass (``T >= 1024``)
      hides: the review probe ran every pk1 shape at the same wall time with either variant (within 0.1 s). So the term
      is an upper bound; it keeps ``shared`` cheaper than ``distinct`` and decides no pack-vs-solo choice.

    The default reproduces today's serial burst: 32 rows of 34 tokens cost 32 x (0.642 + 0.0014) = 20.59 s (20.56 s
    measured); packed, one pk0 pass of T = 2048 costs ~1.62 s (1.63 s measured in the review probe)."""

    head_s: float = 1.4e-3
    layers: int = 53
    transpose_s_per_layer: float = 0.4e-3
    transpose_ref_tokens: int = 8192
    mtp_fill_s: float = 5.0e-3
    swa_layers: int = 39
    tail_gather_s: float = 0.65e-3
    tail_op_s: float = 0.06e-3

    def __post_init__(self):
        terms = (self.head_s, self.layers, self.transpose_s_per_layer, self.mtp_fill_s, self.swa_layers)
        if min(terms + (self.tail_gather_s, self.tail_op_s)) < 0 or int(self.transpose_ref_tokens) < 1:
            raise ValueError(f"packed-pass cost terms must be >= 0 (transpose_ref_tokens >= 1), got {self}")

    def extras_s(self, tokens: int) -> float:
        """Transposes and the MTP fill of a packed pass of ``tokens`` rows."""
        scale = max(1.0, int(tokens) / int(self.transpose_ref_tokens))
        return self.layers * self.transpose_s_per_layer * scale + self.mtp_fill_s

    def tails_s(self, tails: str, batch: int) -> float:
        """SWA tail gathers of a pass of ``batch`` segments with tail variant ``tails`` (``distinct`` always costs more
        than ``shared``)."""
        if tails == SHARED_TAILS:
            return self.swa_layers * self.tail_gather_s
        if tails == DISTINCT_TAILS:
            return self.swa_layers * (self.tail_gather_s + 2 * int(batch) * self.tail_op_s)
        raise ValueError(f"tail variant must be one of {PK1_TAIL_VARIANTS}, got {tails!r}")


DEFAULT_PACKED_COST = PackedCostModel()


def _pass_cost(
    fn: CostFn, model: PackedCostModel, kind: str, seg_rows: int, batch: int, start: int, heads: int, sp1: bool, tails
) -> float:
    if kind == SOLO:
        return float(fn(seg_rows, start)) + model.head_s * heads + (model.tails_s(SHARED_TAILS, 1) if sp1 else 0.0)
    T = int(batch) * int(seg_rows)
    c = float(fn(T, start if kind == PK1 else 0)) + model.head_s * heads + model.extras_s(T)
    return c + (model.tails_s(tails, batch) if kind == PK1 else 0.0)


def packed_pass_cost(
    p: PrefillPass,
    cost: Union[None, Mapping[int, float], CostFn] = None,
    *,
    model: Optional[PackedCostModel] = None,
) -> float:
    """Estimated seconds of pass ``p`` (any kind) under the row cost model ``cost`` (as for :func:`plan_prefill_row`)
    and the packed terms ``model`` (default :data:`DEFAULT_PACKED_COST`). A solo pass costs ``cost(bucket, start)``
    plus its head (and an sp1 chunk's tail gather); a packed pass costs ``cost(T, a)`` plus the
    :class:`PackedCostModel` terms. :func:`plan_prefill_passes` minimizes the sum over a call's passes."""
    fn = _cost_fn(cost)
    m = DEFAULT_PACKED_COST if model is None else model
    heads = sum(1 for g in p.segments if g.last)
    return _pass_cost(fn, m, p.kind, p.seg_rows, p.batch, p.start, heads, p.path == SP1, p.tails)


def _check_seg_buckets(buckets: Sequence[int], block_size: int, name: str) -> Tuple[int, ...]:
    bl, bs = tuple(int(b) for b in buckets), int(block_size)
    if any(b2 <= b1 for b1, b2 in zip(bl, bl[1:])) or (bl and bl[0] < 1):
        raise ValueError(f"{name} must be strictly increasing positive segment sizes, got {bl}")
    bad = [b for b in bl if b % bs]
    if bad:
        raise ValueError(
            f"{name} {bad} are not multiples of the block size {bs}: a packed segment is whole blocks (the fill "
            "kernel maps packed row i to table entry i // bs)"
        )
    return bl


def _max_pass_batch(seg_rows: int, max_tokens: int, max_batch: int) -> int:
    """The largest ``B`` of ``PACK_BATCHES`` with ``B <= max_batch`` and ``B * S <= max_tokens`` (0: none)."""
    return max((B for B in PACK_BATCHES if B <= max_batch and B * int(seg_rows) <= max_tokens), default=0)


def _pass_batch(segments: int) -> int:
    """``B`` of a pass of ``segments >= 2`` real segments: the smallest power of two ``>= segments``."""
    return 1 << (int(segments) - 1).bit_length()


def _pack_buckets(
    block_size: int,
    max_seg: int,
    max_tokens: int,
    max_batch: int,
    pk1: bool,
    seg_buckets: Optional[Sequence[int]],
    sp1_seg_buckets: Optional[Sequence[int]],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """The sp0 / sp1 segment sizes a pass can use: the given sizes (default ``PACK_SEG_BUCKETS`` /
    ``PACK_SP1_SEG_BUCKETS``; no sp1 sizes when ``pk1`` is off) up to ``max_seg`` for which a pass of two segments
    fits ``max_tokens`` and ``max_batch``."""
    bs = int(block_size)
    if bs < 1:
        raise ValueError(f"block_size must be positive, got {bs}")
    ms, mt, mb = int(max_seg), int(max_tokens), int(max_batch)
    if ms < 0 or mt < 1 or mb < 1:
        raise ValueError(f"need max_seg >= 0, max_tokens >= 1 and max_batch >= 1, got {ms}, {mt}, {mb}")
    if not isinstance(pk1, bool):
        raise TypeError(f"pk1 must be a bool, got {pk1!r}")
    seg = _check_seg_buckets(PACK_SEG_BUCKETS if seg_buckets is None else seg_buckets, bs, "seg_buckets")
    sp1 = PACK_SP1_SEG_BUCKETS if sp1_seg_buckets is None else sp1_seg_buckets
    sp1 = _check_seg_buckets(sp1, bs, "sp1_seg_buckets") if pk1 else ()

    def fits(S: int) -> bool:
        return S <= ms and _max_pass_batch(S, mt, mb) >= 2

    return tuple(S for S in seg if fits(S)), tuple(S for S in sp1 if fits(S))


def _seg_rows(chunk: Any, seg_buckets: Tuple[int, ...], sp1_seg_buckets: Tuple[int, ...]) -> Optional[int]:
    buckets = sp1_seg_buckets if chunk.path == SP1 else seg_buckets
    r = int(chunk.end) - int(chunk.start)
    return next((S for S in buckets if S >= r), None)


def segment_rows(
    chunk: ChunkPlan,
    *,
    block_size: int,
    seg_buckets: Sequence[int] = PACK_SEG_BUCKETS,
    sp1_seg_buckets: Sequence[int] = PACK_SP1_SEG_BUCKETS,
) -> Optional[int]:
    """``S`` of a chunk: the smallest of ``seg_buckets`` (sp0 chunk) or ``sp1_seg_buckets`` (sp1 chunk) that is
    ``>=`` the chunk's real rows ``end - start`` (review edit R-E4: a 2108-token row's 60-row tail chunk packs at 128),
    or None when none is (the chunk runs solo). Sizes must be increasing multiples of ``block_size``."""
    bs = int(block_size)
    return _seg_rows(
        chunk,
        _check_seg_buckets(seg_buckets, bs, "seg_buckets"),
        _check_seg_buckets(sp1_seg_buckets, bs, "sp1_seg_buckets"),
    )


def segment_dag(
    requests: Sequence[Any],
    plans: Sequence[RowPlan],
    block_size: int,
    *,
    seg_buckets: Sequence[int] = PACK_SEG_BUCKETS,
    sp1_seg_buckets: Sequence[int] = PACK_SP1_SEG_BUCKETS,
) -> Tuple[List[PackSegment], List[Tuple[int, ...]]]:
    """The segments of one call and their dependencies (design §3.3 steps 1-2): writer-first at segment granularity.

    Returns ``(segments, deps)``: one :class:`PackSegment` per chunk of every row, in (row, chunk) order, with
    ``rows`` from :func:`segment_rows`; ``deps[s]`` = the indices (ascending) of the segments segment ``s`` must run
    after:

    * the row's previous chunk (chunk ``k`` reads rows ``[w0, a)`` its row's earlier chunks write);
    * for an sp1 segment, every segment of another row that writes a block of its row's read-only prefix
      ``page_table[: w0 / bs]`` (review edit R-E1). Global layers fill first and then read keys ``[0, a + i]`` from
      the cache, and the fill skips the blocks below ``w0``: with ``c0 < w0`` (a hit of an odd number of 64-row blocks
      at A = 128) the chunk reads rows ``[c0, w0)`` from the cache, so ``[0, a)`` is not enough. The SWA tail
      ``[a - tail, a)`` lies inside ``[0, w0)`` or the row's own earlier chunks.

    A segment writes the blocks of ``[max(w0, a), end)`` (:func:`fill_table`). Every check of
    :func:`order_prefill_requests` runs first, with its messages (plan / request mismatch, a real position on a block
    id < 1, a block id twice in a row, two rows writing one block, a dependency cycle between rows); an acyclic row
    order makes the segment DAG acyclic, since a segment cycle would project onto a row cycle."""
    bs = int(block_size)
    order_prefill_requests(requests, plans, bs)
    seg_b = _check_seg_buckets(seg_buckets, bs, "seg_buckets")
    sp1_b = _check_seg_buckets(sp1_seg_buckets, bs, "sp1_seg_buckets")
    segments = [
        PackSegment(i, k, c.path, c.start, c.end, _seg_rows(c, seg_b, sp1_b), c.last)
        for i, plan in enumerate(plans)
        for k, c in enumerate(plan.chunks)
    ]
    tables = [_row(r.page_table).tolist() for r in requests]
    writer: Dict[int, int] = {}
    for s, g in enumerate(segments):
        for b in tables[g.row][max(plans[g.row].w0, g.start) // bs : cdiv(g.end, bs)]:
            writer[b] = s
    deps: List[Tuple[int, ...]] = []
    for s, g in enumerate(segments):
        d = {s - 1} if g.chunk_index > 0 else set()  # segments are in (row, chunk) order
        if g.is_sp1:
            for b in tables[g.row][: plans[g.row].read_only_blocks]:
                w = writer.get(b)
                if w is not None and segments[w].row != g.row:
                    d.add(w)
        deps.append(tuple(sorted(d)))
    return segments, deps


def _segment_chunk(g: PackSegment, bucket: int) -> ChunkPlan:
    return ChunkPlan(start=int(g.start), bucket=int(bucket), end=int(g.end), path=g.path, last=bool(g.last))


def _page_table(requests: Sequence[Any], row: int) -> torch.Tensor:
    if not 0 <= int(row) < len(requests):
        raise ValueError(f"segment row {row} outside the call's {len(requests)} rows")
    return _row(requests[int(row)].page_table)


def _solo_pass(g: PackSegment, plans: Sequence[RowPlan], fallback: Optional[PassShape] = None) -> PrefillPass:
    bucket = plans[g.row].chunks[g.chunk_index].bucket
    return PrefillPass(SOLO, (g,), int(bucket), 1, int(g.start), fallback=fallback)


def _cheapest_partition(
    n: int, max_b: int, solo: Callable[[int], float], packed: Callable[[int, int], float]
) -> List[Tuple[int, int]]:
    """The cheapest cut of ``n`` ordered items into consecutive runs ``[(i, j), ...]``: ``j - i == 1`` runs solo,
    ``2 <= j - i <= max_b`` runs as one packed pass. Dynamic programming over prefixes; ties keep the longer run."""
    best = [0.0] + [math.inf] * n
    run = [0] * (n + 1)
    for j in range(1, n + 1):
        for b in range(min(j, max_b), 0, -1):
            c = best[j - b] + (solo(j - 1) if b == 1 else packed(j - b, j))
            if c < best[j]:
                best[j], run[j] = c, b
    out: List[Tuple[int, int]] = []
    j = n
    while j:
        out.append((j - run[j], j))
        j -= run[j]
    return out[::-1]


def _plan_group(
    kind: str,
    S: int,
    a: int,
    members: List[int],
    segments: List[PackSegment],
    plans: Sequence[RowPlan],
    tails_of: Mapping[int, Tuple[int, ...]],
    *,
    max_b: int,
    fn: CostFn,
    model: PackedCostModel,
) -> List[PrefillPass]:
    """The passes of one level's group of packable segments (all ``kind`` at ``S``, pk1 also at start ``a``)."""
    if kind == PK1:  # segments that share a prefix (equal tail blocks) become neighbours: shared-tail passes
        members = sorted(members, key=lambda s: (tails_of[s], segments[s].key))
    heads = [0]
    for s in members:
        heads.append(heads[-1] + int(segments[s].last))

    def variant(i: int, j: int) -> Optional[str]:
        if kind == PK0:
            return None
        return SHARED_TAILS if tails_of[members[i]] == tails_of[members[j - 1]] else DISTINCT_TAILS

    def solo(i: int) -> float:
        g = segments[members[i]]
        bucket = plans[g.row].chunks[g.chunk_index].bucket
        return _pass_cost(fn, model, SOLO, bucket, 1, g.start, int(g.last), g.is_sp1, None)

    def packed(i: int, j: int) -> float:
        return _pass_cost(fn, model, kind, S, _pass_batch(j - i), a, heads[j] - heads[i], kind == PK1, variant(i, j))

    out: List[PrefillPass] = []
    for i, j in _cheapest_partition(len(members), max_b, solo, packed):
        if j - i == 1:
            out.append(_solo_pass(segments[members[i]], plans))
        else:
            part = tuple(sorted((segments[s] for s in members[i:j]), key=lambda g: g.key))
            out.append(PrefillPass(kind, part, S, _pass_batch(j - i), a, variant(i, j)))
    return out


def _check_passes(
    passes: Sequence[PrefillPass],
    segments: Sequence[PackSegment],
    deps: Sequence[Tuple[int, ...]],
    *,
    max_tokens: int,
    max_batch: int,
) -> None:
    """The planner's own invariants (design R-P3, R-P4); a failure is a planner bug: every segment in exactly one
    pass, every dependency in a strictly earlier pass, every packed pass of >= 2 segments at ``B`` = the next power of
    two within the caps. Raises ``AssertionError`` (explicitly, so ``python -O`` keeps the check) before any device
    work; the generator then runs the call per row in writer-first order (``MotifGenerator.plan_prefill_batch``,
    counter ``packed_plan_errors``), so packing never refuses a call."""
    index = {g.key: s for s, g in enumerate(segments)}
    pass_of = [-1] * len(segments)
    for n, p in enumerate(passes):
        for g in p.segments:
            s = index.get(g.key)
            if s is None or segments[s] != g or pass_of[s] >= 0:
                raise AssertionError(f"pass {n} ({p.describe()}): segment {g.key} is unknown or planned twice")
            pass_of[s] = n
        if p.is_packed and not (
            len(p.segments) >= 2
            and p.batch == _pass_batch(len(p.segments))
            and p.batch <= max_batch
            and p.tokens <= max_tokens
        ):
            raise AssertionError(f"pass {n} ({p.describe()}) breaks the batch / token caps")
    if any(n < 0 for n in pass_of):
        raise AssertionError(f"segments {[segments[s].key for s, n in enumerate(pass_of) if n < 0]} are in no pass")
    for s, d in enumerate(deps):
        for t in d:
            if pass_of[t] >= pass_of[s]:
                raise AssertionError(
                    f"segment {segments[s].key} (pass {pass_of[s]}) depends on segment {segments[t].key}, planned in "
                    f"pass {pass_of[t]}: not strictly earlier"
                )


def plan_prefill_passes(
    requests: Sequence[Any],
    plans: Sequence[RowPlan],
    *,
    block_size: int,
    max_seg: int = DEFAULT_PACKED_PREFILL_MAX_SEG,
    max_tokens: int = DEFAULT_PACKED_PREFILL_MAX_TOKENS,
    max_batch: int = DEFAULT_PACK_MAX_BATCH,
    pk1: bool = True,
    allowed: Optional[Collection[PassShape]] = None,
    cost: Union[None, Mapping[int, float], CostFn] = None,
    seg_buckets: Optional[Sequence[int]] = None,
    sp1_seg_buckets: Optional[Sequence[int]] = None,
    swa_tail: int = DEFAULT_SWA_TAIL,
    pack_cost: Optional[PackedCostModel] = None,
) -> List[PrefillPass]:
    """The passes of one ``prefill_forward_batch`` call with packed prefill on, in execution order (design §3.3;
    module docstring rule 6). Pure host; raises ``ValueError`` (bad rows: :func:`segment_dag`) before any device work.

    1. Segments and dependencies: :func:`segment_dag` (it runs every check of :func:`order_prefill_requests`).
    2. Level scheduling: the segments whose dependencies have all run form the next level. Its packable segments are
       grouped by ``("pk0", S)`` and ``("pk1", S, a)``; the others run solo. Each group is cut into packed passes of
       ``b >= 2`` segments (``B`` = the next power of two, ``B <= max_batch``, ``B * S <= max_tokens``) and solo
       chunks, choosing the cut of least :func:`packed_pass_cost` (a lone segment always runs solo). pk1 groups are
       ordered by tail blocks first, so segments that share a prefix share a pass (the ``shared`` variant). A level's
       passes run in the order of their first segment ``(row, chunk)``. No pass holds a segment together with one of
       its dependencies.
    3. Shape filter: with ``allowed`` (the generator's warmed shapes once the decode trace is captured), a packed pass
       whose :attr:`PrefillPass.shape` is not in it runs as one solo pass per segment, each with ``fallback`` = that
       shape: packing never refuses a call and never compiles a program. Solo shapes are not filtered here (the
       generator refuses an unwarmed solo shape, as it does without packing).

    Args:
        requests / plans: the call's rows (``PrefillRequest``-like: ``start``, ``seq_len``, ``page_table``) and their
            row plans (``plan_prefill_row``), as for :func:`order_prefill_requests`.
        block_size: KV block size.
        max_seg: the largest segment ``S`` (``cfg.pack_max_seg``; 0 = nothing packs).
        max_tokens: the largest pass ``T`` (``cfg.pack_tokens_cap``; at most the span cap, so every ``T = B * S``, a
            power of two, is a compiled bucket).
        max_batch: the largest ``B``.
        pk1: pack sp1 chunks at a common start (``cfg.pack_pk1``); False: they run solo.
        allowed: the shapes a packed pass may have; None = any (before the decode capture).
        cost: the row cost model, as for :func:`plan_prefill_row` (the generator's ``cfg.prefill_cost_table`` model).
        seg_buckets / sp1_seg_buckets: the sp0 / sp1 segment sizes (``cfg.pack_seg_buckets`` /
            ``cfg.pack_sp1_seg_buckets``; default ``generator_api.PACK_SEG_BUCKETS`` / ``PACK_SP1_SEG_BUCKETS``),
            narrowed to ``max_seg`` and to the sizes for which a pass of two segments fits the caps.
        swa_tail: SWA tail rows (``cfg.prefill_swa_tail``): a pk1 pass is ``shared`` iff all its segments have the same
            blocks at ``[a - tail, a)``.
        pack_cost: the packed-pass cost terms (default :data:`DEFAULT_PACKED_COST`).
    """
    bs = int(block_size)
    seg_b, sp1_b = _pack_buckets(bs, max_seg, max_tokens, max_batch, pk1, seg_buckets, sp1_seg_buckets)
    tail = int(swa_tail)
    if tail < 0 or tail % bs:
        raise ValueError(f"swa_tail {tail} must be a non-negative multiple of the block size {bs}")
    fn = _cost_fn(cost)
    model = DEFAULT_PACKED_COST if pack_cost is None else pack_cost
    segments, deps = segment_dag(requests, plans, bs, seg_buckets=seg_b, sp1_seg_buckets=sp1_b)
    tails_of: Dict[int, Tuple[int, ...]] = {}
    for s, g in enumerate(segments):
        if g.is_sp1 and g.rows is not None:
            ids = tail_blocks(_page_table(requests, g.row), _segment_chunk(g, g.rows), bs, tail) if tail else None
            tails_of[s] = () if ids is None else tuple(ids.tolist())
    passes: List[PrefillPass] = []
    done = [False] * len(segments)
    pending = list(range(len(segments)))
    while pending:
        ready = [s for s in pending if all(done[t] for t in deps[s])]
        if not ready:  # unreachable: segment_dag refuses row cycles, and an acyclic row order makes the DAG acyclic
            raise ValueError(f"prefill segments {[segments[s].key for s in pending]} depend on each other (cycle)")
        level: List[PrefillPass] = []
        groups: Dict[Tuple[str, int, int], List[int]] = {}
        for s in ready:
            g = segments[s]
            if g.rows is None:
                level.append(_solo_pass(g, plans))
            else:
                groups.setdefault((PK1, g.rows, g.start) if g.is_sp1 else (PK0, g.rows, 0), []).append(s)
        for (kind, S, a), members in groups.items():
            max_b = _max_pass_batch(S, int(max_tokens), int(max_batch))
            level += _plan_group(kind, S, a, members, segments, plans, tails_of, max_b=max_b, fn=fn, model=model)
        level.sort(key=lambda p: min(g.key for g in p.segments))
        for p in level:
            if p.is_packed and allowed is not None and p.shape not in allowed:
                passes += [_solo_pass(g, plans, fallback=p.shape) for g in p.segments]
            else:
                passes.append(p)
        for s in ready:
            done[s] = True
        pending = [s for s in pending if not done[s]]
    _check_passes(passes, segments, deps, max_tokens=int(max_tokens), max_batch=int(max_batch))
    return passes


def solo_prefill_passes(plans: Sequence[RowPlan], order: Optional[Sequence[int]] = None) -> List[PrefillPass]:
    """One solo pass per chunk: the rows in ``order`` (default input order; with packing off the generator passes
    :func:`order_prefill_requests`), each row's chunks in sequence. That is the row-by-row execution of a call
    without packing, as passes."""
    idx = list(range(len(plans))) if order is None else [int(i) for i in order]
    if sorted(idx) != list(range(len(plans))):
        raise ValueError(f"order {idx} is not a permutation of the {len(plans)} rows")
    return [
        PrefillPass(SOLO, (PackSegment(i, k, c.path, c.start, c.end, None, c.last),), c.bucket, 1, c.start)
        for i in idx
        for k, c in enumerate(plans[i].chunks)
    ]


def packed_pass_shapes(
    *,
    block_size: int,
    max_seg: int = DEFAULT_PACKED_PREFILL_MAX_SEG,
    max_tokens: int = DEFAULT_PACKED_PREFILL_MAX_TOKENS,
    max_batch: int = DEFAULT_PACK_MAX_BATCH,
    pk1: bool = True,
    seg_buckets: Optional[Sequence[int]] = None,
    sp1_seg_buckets: Optional[Sequence[int]] = None,
) -> Tuple[PassShape, ...]:
    """Every shape a packed pass of :func:`plan_prefill_passes` can have with these knobs, in the order of
    ``MotifTTConfig.packed_prefill_shapes()`` (the list the generator warms; a host test checks they agree): pk0
    ``("pk0", T, S)`` per sp0 segment size ``S`` and ``T = B * S`` ascending, then pk1 ``("pk1", T, S, tails)`` for
    both tail variants. Defaults: 22 pk0 and 17 x 2 pk1 shapes."""
    seg_b, sp1_b = _pack_buckets(block_size, max_seg, max_tokens, max_batch, pk1, seg_buckets, sp1_seg_buckets)

    def batches(S: int) -> List[int]:
        top = _max_pass_batch(S, int(max_tokens), int(max_batch))
        return [B for B in PACK_BATCHES if B <= top]

    out: List[PassShape] = [(PK0, B * S, S) for S in seg_b for B in batches(S)]
    out += [(PK1, B * S, S, v) for S in sp1_b for B in batches(S) for v in PK1_TAIL_VARIANTS]
    return tuple(out)


# ---- per-pass host tables (each segment's slice is the per-chunk table of its chunk re-bucketed to S) ------------
def pass_chunks(p: PrefillPass, plans: Sequence[RowPlan]) -> List[ChunkPlan]:
    """The chunk of every real segment of ``p``, re-bucketed to the pass's segment rows (review edit R-E12:
    ``dataclasses.replace(chunk, bucket=S)`` with ``S >=`` the chunk's rows, so the per-chunk table builders give
    ``S / bs`` fill entries and ``S`` RoPE rows; a solo pass keeps its chunk). ``ValueError`` when a segment does not
    match its row plan."""
    out = []
    for g in p.segments:
        if not 0 <= g.row < len(plans) or not 0 <= g.chunk_index < len(plans[g.row].chunks):
            raise ValueError(f"segment {g.key} has no chunk in the {len(plans)} row plans")
        c = plans[g.row].chunks[g.chunk_index]
        if (c.start, c.end, c.path, c.last) != (g.start, g.end, g.path, g.last):
            raise ValueError(f"segment {g} does not match chunk {c} of row {g.row}'s plan")
        out.append(c if c.bucket == p.seg_rows else replace(c, bucket=int(p.seg_rows)))
    return out


def pass_fill_table(p: PrefillPass, requests: Sequence[Any], plans: Sequence[RowPlan], block_size: int) -> torch.Tensor:
    """``paged_fill_cache`` table of pass ``p``: ``int32 [1, T / bs]``. Entries ``[k S / bs, (k + 1) S / bs)`` are
    segment ``k``'s :func:`fill_table` (its chunk re-bucketed to ``S``: blocks below the row's ``w0`` and pure-padding
    blocks -1); a dummy segment's entries are -1 (it writes nothing). The fill kernel maps packed row ``i`` to entry
    ``i // bs``, so ``S`` must be a multiple of ``bs``."""
    bs, S = int(block_size), int(p.seg_rows)
    if bs < 1 or S % bs:
        raise ValueError(f"segment rows {S} must be a positive multiple of the block size {bs}")
    n = S // bs
    out = torch.full((1, int(p.batch) * n), -1, dtype=torch.int32)
    for k, (g, c) in enumerate(zip(p.segments, pass_chunks(p, plans))):
        plan = plans[g.row]
        if int(plan.block_size) != bs:
            raise ValueError(f"row {g.row}'s plan has block size {plan.block_size}, the pass {bs}")
        out[0, k * n : (k + 1) * n] = fill_table(_page_table(requests, g.row), c, plan.w0, bs)
    return out


def pass_rope_positions(p: PrefillPass, max_positions: int) -> torch.Tensor:
    """RoPE table rows of pass ``p``: ``int32 [T]``, segment ``k``'s :func:`rope_positions` at ``[k S, k S + S)``
    (``min(a_k + i, max_positions - 1)``; real rows never clamp); dummy segments copy segment 0's rows."""
    rows = [rope_positions(_segment_chunk(g, p.seg_rows), max_positions) for g in p.segments]
    return torch.cat(rows + [rows[0]] * p.dummies)


def pass_sdpa_tables(p: PrefillPass, requests: Sequence[Any], block_size: int, width: int) -> torch.Tensor:
    """sp1 SDPA page tables of a pk1 (or solo sp1) pass: ``int32 [B, width]``, row ``k`` = segment ``k``'s
    :func:`sdpa_table` (the real ids of blocks ``[0, cdiv(end, bs))``, then 0; never -1); dummy segments copy segment
    0's row. ``width * bs`` must cover ``a + S`` (padded query rows read up to there)."""
    if p.path != SP1:
        raise ValueError(f"SDPA page tables exist for sp1 passes only, got a {p.kind} {p.path} pass")
    bs, W = int(block_size), int(width)
    if W * bs < int(p.start) + int(p.seg_rows):
        raise ValueError(f"SDPA width {W} x {bs} does not cover the pass's {int(p.start) + int(p.seg_rows)} positions")
    rows = [sdpa_table(_page_table(requests, g.row), g.end, bs, W) for g in p.segments]
    return torch.stack(rows + [rows[0]] * p.dummies)


def pass_tail_blocks(
    p: PrefillPass, requests: Sequence[Any], block_size: int, tail: int = DEFAULT_SWA_TAIL
) -> torch.Tensor:
    """SWA tail block ids of a pk1 (or solo sp1) pass: ``int32 [B, tail / bs]``, row ``k`` = segment ``k``'s
    :func:`tail_blocks` (positions ``[a - tail, a)``); dummy segments copy segment 0's row."""
    if p.path != SP1:
        raise ValueError(f"SWA tails exist for sp1 passes only, got a {p.kind} {p.path} pass")
    rows = [
        tail_blocks(_page_table(requests, g.row), _segment_chunk(g, p.seg_rows), block_size, tail) for g in p.segments
    ]
    return torch.stack(rows + [rows[0]] * p.dummies)


def tail_variant(tail: torch.Tensor) -> str:
    """The pk1 tail variant of a ``[B, nt]`` tail table (review edit R-E2): ``"shared"`` when every row equals row 0,
    else ``"distinct"`` (no partial dedupe: the program set must not depend on the data)."""
    t = torch.as_tensor(tail)
    if t.ndim != 2 or t.shape[0] < 1:
        raise ValueError(f"tail table must be [B, nt] with B >= 1, got {tuple(t.shape)}")
    return SHARED_TAILS if bool((t == t[:1]).all()) else DISTINCT_TAILS


def pass_rows(p: PrefillPass, values: Sequence[Any], fill: int) -> torch.Tensor:
    """A per-row host input in the pass's packed layout: ``int32 [T]`` holding ``values[k]`` (segment ``k``'s
    ``end - start`` real-row values) at ``[k S, k S + end - start)`` and ``fill`` elsewhere (segment padding, dummy
    segments). E.g. the MTP layer's next tokens (``mtp.mtp_next_tokens`` per segment)."""
    if len(values) != len(p.segments):
        raise ValueError(f"{len(values)} value rows for the pass's {len(p.segments)} segments")
    S = int(p.seg_rows)
    out = torch.full((p.tokens,), int(fill), dtype=torch.int32)
    for k, (g, v) in enumerate(zip(p.segments, values)):
        v = torch.as_tensor(v).reshape(-1)
        if v.numel() != g.real_rows or v.dtype.is_floating_point:
            raise ValueError(f"segment {k} takes {g.real_rows} integer values, got {v.dtype} [{v.numel()}]")
        out[k * S : k * S + g.real_rows] = v.to(torch.int32)
    return out


def pass_tokens(p: PrefillPass, requests: Sequence[Any], pad_id: int) -> torch.Tensor:
    """Token ids of pass ``p``: ``int32 [T]``, segment ``k``'s ``tokens[start:end]`` at ``[k S, ...)``, ``pad_id``
    in padding rows and dummy segments."""
    rows = []
    for g in p.segments:
        if not 0 <= g.row < len(requests):
            raise ValueError(f"segment row {g.row} outside the call's {len(requests)} rows")
        rows.append(torch.as_tensor(requests[g.row].tokens)[g.start : g.end])
    return pass_rows(p, rows, pad_id)


@dataclass(frozen=True)
class PassTables:
    """Every host table of one pass except its tokens (:func:`pass_tables`); the attention owner's
    ``PackedHostTables`` holds the same data for packed passes.

    Attributes:
        fill: ``int32 [1, T / bs]`` (:func:`pass_fill_table`).
        rope: ``int32 [T]`` (:func:`pass_rope_positions`; packed passes always gather their RoPE rows, review edit
            R-E11).
        ends: ``int32 [B]``: one past each segment's last real row (absolute position); dummies copy segment 0's.
        sdpa: sp1: ``int32 [B, W']`` (:func:`pass_sdpa_tables`); else None.
        start_idx: sp1: ``int32 [1]`` = ``[a]``; else None.
        tail: sp1 with an SWA tail: ``int32 [B, tail / bs]`` (:func:`pass_tail_blocks`); else None.
        tails: the pass's tail variant (pk1), else None.
    """

    fill: torch.Tensor
    rope: torch.Tensor
    ends: torch.Tensor
    sdpa: Optional[torch.Tensor] = None
    start_idx: Optional[torch.Tensor] = None
    tail: Optional[torch.Tensor] = None
    tails: Optional[str] = None


def pass_tables(
    p: PrefillPass,
    requests: Sequence[Any],
    plans: Sequence[RowPlan],
    *,
    block_size: int,
    sdpa_width: int,
    max_positions: int,
    swa_tail: int = DEFAULT_SWA_TAIL,
) -> PassTables:
    """Every host table of pass ``p`` (any kind; a solo pass gets its chunk's tables of rule 4, with ``[1, ...]``
    leading dims). Raises ``ValueError`` when a ``shared`` pk1 pass has segments whose tail blocks differ: the
    shared program gathers segment 0's tail only."""
    ends = [int(g.end) for g in p.segments]
    out = dict(
        fill=pass_fill_table(p, requests, plans, block_size),
        rope=pass_rope_positions(p, max_positions),
        ends=torch.tensor(ends + ends[:1] * p.dummies, dtype=torch.int32),
        tails=p.tails,
    )
    if p.path == SP1:
        out["sdpa"] = pass_sdpa_tables(p, requests, block_size, sdpa_width)
        out["start_idx"] = torch.tensor([int(p.start)], dtype=torch.int32)
        if int(swa_tail) > 0:
            out["tail"] = pass_tail_blocks(p, requests, block_size, swa_tail)
            if p.tails == SHARED_TAILS and tail_variant(out["tail"]) != SHARED_TAILS:
                raise ValueError(
                    f"{p.describe()}: planned with shared tails, but its segments' tail blocks differ "
                    f"{out['tail'].tolist()}"
                )
    return PassTables(**out)


# ----------------------------------------------------------------------------------------------------------------
# vLLM scheduler configuration (bridge fail-fast checks, features design §1.5)
# ----------------------------------------------------------------------------------------------------------------
SMALL_BUDGET_TOKENS = 4096


def check_scheduler_config(
    *,
    chunked: bool,
    budget: Optional[int],
    threshold: int,
    align: int,
    span_cap: int,
    prefix_caching: bool,
    prefix_match_unit: Optional[int],
    block_size: int,
    kv_replicated: bool,
) -> List[str]:
    """Check vLLM's scheduler settings against the generator's prefill geometry (features design §1.5).

    Args:
        chunked: vLLM ``enable_chunked_prefill`` (after the platform's policy).
        budget: ``max_num_batched_tokens``; required when ``chunked``.
        threshold: ``long_prefill_token_threshold`` (0 = no per-request cap).
        align: the generator's resume alignment ``A`` (``generator.prefill_alignment``; before the generator exists
            ``generator_api.DEFAULT_PREFILL_ALIGNMENT``).
        span_cap: the generator's span cap (``generator.max_prefill_span``).
        prefix_caching: vLLM ``enable_prefix_caching``.
        prefix_match_unit: vLLM's prefix-match unit (None = the block size).
        block_size: vLLM ``--block-size``.
        kv_replicated: KV-R resolved (``GeneratorSettings.kv_replicated``).

    Raises ``ValueError`` on configurations that would serve wrong outputs; returns warnings (performance only), each
    naming the change that clears it (for the bucket geometry: ``A`` a power of two that divides the span cap). Every
    span cap ``MOTIF3_PREFILL_MAX_BUCKET`` allows is accepted, including one that does not exceed ``A`` (128 at
    A = 128): no budget then fits a resumed chunk in one bucket, so the warning names the span cap instead of a
    budget, and the small-budget warning (whose floor is ``min(4096, span cap) - A``) stays silent."""
    bs = check_block_size(block_size)
    A, cap, thr = int(align), int(span_cap), int(threshold)
    if A < 1 or A % bs:
        raise ValueError(f"resume alignment {A} must be a positive multiple of the block size {bs}")
    if cap < MIN_PREFILL_BUCKET:
        raise ValueError(f"span cap {cap} is below the smallest prefill bucket {MIN_PREFILL_BUCKET}")
    if thr < 0:
        raise ValueError(f"long_prefill_token_threshold must be >= 0, got {thr}")
    if prefix_caching and not kv_replicated:
        raise ValueError(
            "prefix caching needs KV-R (replicated decode KV writes): a hit on a block decode-wrote on another DP row "
            "reads stale KV and the MoE reduce-scatter spreads it to every row (features design §3.4); unset "
            "MOTIF3_KV_REPLICATED_DECODE=0 or pass --no-enable-prefix-caching"
        )
    if prefix_caching and prefix_match_unit is not None and int(prefix_match_unit) != bs:
        raise ValueError(
            f"--prefix-match-unit {prefix_match_unit} must be unset or the block size {bs} (one KV group: vLLM's "
            f"unitary coordinator asserts hash_block_size == block_size; Motif's hits must be whole blocks)"
        )
    warnings: List[str] = []
    if not chunked:
        return warnings
    if budget is None or int(budget) < 1:
        raise ValueError(f"chunked prefill needs a positive max_num_batched_tokens, got {budget!r}")
    budget = int(budget)
    if budget % A:
        warnings.append(
            f"max_num_batched_tokens {budget} is not a multiple of the resume alignment {A}: a lone long prompt's "
            f"chunk ends are unaligned, so every continuation recomputes up to {A - 1} tokens (use "
            f"{max(A, budget // A * A)})"
        )
    if thr > 0 and thr % A:
        warnings.append(
            f"long_prefill_token_threshold {thr} is not a multiple of the resume alignment {A}: continuations "
            f"recompute up to {A - 1} tokens (use {max(A, thr // A * A)})"
        )
    per_row = min(budget, thr) if thr > 0 else budget
    if per_row + A - 1 > cap:
        # A span cap <= A (MOTIF3_PREFILL_MAX_BUCKET=128 at A = 128) leaves no budget that fits a resumed chunk plus its
        # recompute in one bucket: point at the cap instead of suggesting a budget <= 0.
        fix = (
            f"use max_num_batched_tokens <= {cap - A}"
            if cap > A
            else f"no budget avoids that while the span cap does not exceed the resume alignment {A}: raise "
            f"MOTIF3_PREFILL_MAX_BUCKET"
        )
        warnings.append(
            f"a {per_row}-token chunk plus up to {A - 1} recomputed tokens exceeds the {cap}-row span cap: such rows "
            f"are split internally, so a prefill step can take longer than the budget suggests ({fix})"
        )
    # Below the aligned 4K budget (the §1.1 interactive variant is 4096 - A), or below the span cap's own budget when
    # that is smaller: under a small span cap the chunks are small whatever the budget (the warning above says so), and
    # the suggested pin (cap - A) must not trip this warning again. Never fires when cap <= A (small_floor <= 0).
    small_floor = min(SMALL_BUDGET_TOKENS, cap) - A
    if budget < small_floor:
        warnings.append(
            f"max_num_batched_tokens {budget} < {small_floor}: long prompts run as many small eager "
            f"chunks, each paying up to the ~0.64 s dispatch floor (vLLM's unpinned default for `vllm serve` on TT is "
            f"2048; pin {recommended_budget(cap, A)})"
        )
    return warnings


def recommended_budget(span_cap: int, align: int) -> int:
    """The chunk budget that keeps every lone-request span inside one ``span_cap`` bucket: ``span_cap - A``
    (8064 for the production 8192 / 128 of gate G9's per-bucket chunks; 8128 for an all-64/64 table, A = 64). Also the
    recommended ``--long-prefill-token-threshold`` (lead decision: threshold = budget)."""
    cap, A = int(span_cap), int(align)
    if A < 1 or cap <= A:
        raise ValueError(f"span cap {cap} must exceed the alignment {A}")
    return (cap - A) // A * A


__all__ = [
    "ChunkPlan",
    "ChunkTables",
    "CostFn",
    "DEFAULT_PACKED_COST",
    "DEFAULT_PACK_MAX_BATCH",
    "DEFAULT_PREFILL_COST_TABLE",
    "DEFAULT_SP1_ATTN_S_PER_ROW_KEY",
    "DEFAULT_SP1_GLOBAL_CHUNKS",
    "DEFAULT_SP1_GLOBAL_COST",
    "DEFAULT_SWA_TAIL",
    "DISTINCT_TAILS",
    "PASS_KINDS",
    "PATHS",
    "PK0",
    "PK1",
    "PackSegment",
    "PackedCostModel",
    "PassShape",
    "PassTables",
    "PrefillPass",
    "RowPlan",
    "SDPA_TABLE_WIDTH_MULTIPLE",
    "SHARED_TAILS",
    "SMALL_BUDGET_TOKENS",
    "SOLO",
    "SP0",
    "SP1",
    "SP1_GLOBAL_CHUNKS_BF16_KV",
    "SP1_GLOBAL_CHUNKS_BY_KV_DTYPE",
    "SP1_GLOBAL_LAYERS",
    "check_scheduler_config",
    "chunk_tables",
    "fill_table",
    "order_prefill_requests",
    "packed_pass_cost",
    "packed_pass_shapes",
    "pass_chunks",
    "pass_fill_table",
    "pass_rope_positions",
    "pass_rows",
    "pass_sdpa_tables",
    "pass_tables",
    "pass_tail_blocks",
    "pass_tokens",
    "plan_cost",
    "plan_prefill_batch",
    "plan_prefill_passes",
    "plan_prefill_row",
    "prefill_cost_model",
    "recommended_budget",
    "resume_alignment",
    "rope_positions",
    "sdpa_table",
    "sdpa_table_width",
    "segment_dag",
    "segment_rows",
    "solo_prefill_passes",
    "sp1_global_chunk_table",
    "sp1_global_qk",
    "sp1_prefix_cost",
    "span_buckets",
    "table_cost",
    "tail_blocks",
    "tail_variant",
]
