# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resumed / chunked prefill planning: pure host, torch only (features design §2.2, §3.1, §3.7.2).

A prefill *row* is one :class:`~models.demos.motif3.tt.generator_api.PrefillRequest`: tokens at positions
``0 .. e-1``, of which ``[0, s)`` are already in the paged latent cache (``s = request.start`` = vLLM
``num_computed_tokens``, ``e = request.end`` = the end of the chunk vLLM scheduled). This module turns ``(s, e)``
into the chunks the generator runs and builds every host-side table those chunks need. Every alignment and
page-table rule of the features design lives here, so ``tests/unit/test_prefill_plan.py`` checks them exhaustively.

Notation: ``bs`` = KV block size (64), ``A`` = resume alignment = ``lcm(bs, q_chunk, k_chunk)`` of the sp1 global
op (64 by default; ``cfg.prefill_resume_alignment``), ``tail`` = the SWA tail (128 = window 129 minus the current
key; ``cfg.prefill_swa_tail``), ``cap`` = the span cap (8192: the largest bucket the generator compiles;
``cfg.max_prefill_span``), ``buckets`` = the prefill buckets ``<= cap`` (``cfg.prefill_span_buckets``).

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
   Only the last chunk is padded and only the last chunk runs the LM head (local row ``e - 1 - a``). A chunk at
   start 0 is **sp0** (the draft-1 path: square causal SDPA over the chunk's own rows; no cache reads); every other
   chunk is **sp1** (global layers attend over the paged cache from the chunk start; SWA layers read a ``tail``-row
   tail from the cache).
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

No per-lane or per-slot state appears anywhere: a request may change lane between chunks, the paged cache is the
only cross-chunk state.

Import rule: stdlib, torch and ``generator_api`` only (never ttnn): the bridge, its host tests and
``model_config`` import this module.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .generator_api import MIN_PREFILL_BUCKET, cdiv, check_block_size

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
# sp1 chunks additionally run the absorbed global attention over the cached prefix [I]: 14 global layers x 10 heads x
# 2 x (576 + 576) FLOP per (row, key) at ~50 TFLOP/s per chip (features design §3.2.2, review R10) = 6.5e-9 s per
# (chunk row, prefix key). The chunk's own causal triangle is already in the bucket's table cost.
DEFAULT_SP1_ATTN_S_PER_ROW_KEY = 6.5e-9

CostFn = Callable[[int, int], float]  # (bucket, chunk start) -> estimated seconds


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


def prefill_cost_model(
    table: Optional[Mapping[int, float]] = None, *, sp1_s_per_row_key: float = DEFAULT_SP1_ATTN_S_PER_ROW_KEY
) -> CostFn:
    """The default chunk cost [I]: ``table_cost(table, C)`` plus, for an sp1 chunk at ``a > 0``, the absorbed global
    attention over the cached prefix ``sp1_s_per_row_key * C * a``."""
    t = dict(DEFAULT_PREFILL_COST_TABLE if table is None else table)
    k = float(sp1_s_per_row_key)

    def cost(bucket: int, start: int) -> float:
        return table_cost(t, bucket) + (k * bucket * start if start > 0 else 0.0)

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

    Raises ``ValueError`` on configurations that would serve wrong outputs; returns warnings (performance only)."""
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
        warnings.append(
            f"a {per_row}-token chunk plus up to {A - 1} recomputed tokens exceeds the {cap}-row span cap: such rows "
            f"are split internally, so a prefill step can take longer than the budget suggests (use "
            f"max_num_batched_tokens <= {cap - A})"
        )
    if budget < SMALL_BUDGET_TOKENS - A:  # below the aligned 4K budget (the §1.1 interactive variant is 4096 - A)
        warnings.append(
            f"max_num_batched_tokens {budget} < {SMALL_BUDGET_TOKENS - A}: long prompts run as many small eager "
            f"chunks, each paying up to the ~0.64 s dispatch floor (vLLM's unpinned default for `vllm serve` on TT is "
            f"2048; pin {recommended_budget(cap, A)})"
        )
    return warnings


def recommended_budget(span_cap: int, align: int) -> int:
    """The chunk budget that keeps every lone-request span inside one ``span_cap`` bucket: ``span_cap - A``
    (8128 for 8192 / 64, 8064 for 8192 / 128). Also the recommended ``--long-prefill-token-threshold``."""
    cap, A = int(span_cap), int(align)
    if A < 1 or cap <= A:
        raise ValueError(f"span cap {cap} must exceed the alignment {A}")
    return (cap - A) // A * A


__all__ = [
    "ChunkPlan",
    "ChunkTables",
    "CostFn",
    "DEFAULT_PREFILL_COST_TABLE",
    "DEFAULT_SP1_ATTN_S_PER_ROW_KEY",
    "DEFAULT_SWA_TAIL",
    "PATHS",
    "RowPlan",
    "SDPA_TABLE_WIDTH_MULTIPLE",
    "SMALL_BUDGET_TOKENS",
    "SP0",
    "SP1",
    "check_scheduler_config",
    "chunk_tables",
    "fill_table",
    "order_prefill_requests",
    "plan_cost",
    "plan_prefill_batch",
    "plan_prefill_row",
    "prefill_cost_model",
    "recommended_budget",
    "rope_positions",
    "sdpa_table",
    "sdpa_table_width",
    "span_buckets",
    "table_cost",
    "tail_blocks",
]
