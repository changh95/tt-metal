# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure helpers of the packed multi-user prefill's GATHER head (phase 3c, design_packed_prefill.md 2.6 item 1). No
ttnn import: ``tests/unit/test_sorted_moe_chunk_plan.py`` exercises the arithmetic without a device.

A packed pass of ``B_pad`` users x ``S`` tokens leaves the residual stream as ``[1, 1, T = B_pad * S, H]``. The v1
head (the tt_transformers batched path) runs norm + lm_head on all ``T`` rows and reads one 32-row tile per user back;
the gather head picks the ``B`` last-token rows ``u * S + len_u - 1`` into ONE ``[1, 1, 32, H]`` tile row (padded with
row 0), so norm + lm_head run on 32 rows -- the shapes of the single-user head -- and the host reads 32 x vocab once.

Phase 3g / D2 (backlog item 8, ``SOLAR_OPEN_PACKED_PREFILL_SEQ_NUMERICS``): the pure half of the "sequential numerics"
knob -- the level switch, the in0_block_w rule of ttnn's auto matmul configs and the row-piece arithmetic -- lives here
too; ``tt/packed_numerics.py`` holds the ttnn side. D1 (``tests/test_packed_bias_bisect.py``, README "Phase 3g rows")
found that a packed pass of ``T = B x S`` rows leaves the sequential ``S``-row pass at layer 0, op ``qkv``, because ttnn
picks another matmul program config for ``T`` rows than for ``S`` rows (1D systolic ``in0_block_w = 2`` at M = 128 vs 2D
mcast ``in0_block_w = 1`` from M = 256 on), and that the shared expert (explicit ``in0_block_w = 32`` configs up to 128
rows, auto above), the head (the 32-row tile takes the width-sharded decode norm kernel; the lm_head takes a 2D config at
M = 4096) and o_proj (1D k = 2 up to 256 rows, 2D k = 1 from 1024) change with the row count as well. A bf16-destination
matmul's result depends on the K-block length ``in0_block_w`` (the partial sums are spilled through a bf16 intermediate
per K block) and on nothing else in the program config: the same rows give the SAME BITS under any M / N blocking with
the same ``in0_block_w`` and compute config (D1 r7 / r8: qkv 2D k=2 at 256 rows == 1D k=2 at 128; o_proj 2D k=2 at 4096
== 1D k=2 at 128; lm_head 1D k=2 at 32 == 128 == 1024 rows; shared gate 1D k=32 at 256 == 128).
"""

import contextlib
import os

GATHER_HEAD_ROWS = 32  # one tile row: the head's shapes are those of the single-user prefill head

# ---------------------------------------------------------------------------------------------------------------------
# Phase 3g / D2: the "sequential numerics" knob of a packed pass (SOLAR_OPEN_PACKED_PREFILL_SEQ_NUMERICS).
#   0  off (phase 3a-3f behaviour: ttnn auto configs at T rows, the T-row head): the packed pass is the WORSE arm at
#      the first token (A0: 22 / 32 users top-1 = HF vs 32 / 32 sequential, KL 0.3517 vs 0.0698)
#   1  the row-wise matmuls of a packed pass reproduce the S-row (per-user) sequential pass bit for bit: qkv / o_proj
#      with an explicit program config carrying the S-row auto in0_block_w (attention/prefill.py), the shared expert in
#      S-row pieces (shared_expert.py), the head as 32-row tiles through the sequential head's norm kernel and the
#      lm_head in <= LM_HEAD_SEQ_ROWS-row pieces (model.py). A pass of T <= dense_bmm_max_tokens rows is then
#      bit-identical to the sequential prefills (D2 r1 / r2: 48 / 48 layers, 32 / 32 users at T = 256); the
#      expert-sorted MoE of longer splits keeps its own numerics (an algorithm, not a config) and leaves a residual
#      first-token shift (D2 r2, one 32 x 128 pass: 26 / 32 top-1 = HF, KL 0.1865, gap -0.93 vs the sequential arm).
#   2  (DEFAULT since D2) level 1 plus the routed experts as dense-bmm splits of dense_bmm_max_tokens rows whenever the
#      per-user S itself runs the dense bmm (S <= dense_bmm_max_tokens; experts/prefill.py): every op of the pass then
#      equals the S-row sequential pass bit for bit at any T (D2 r3: one 32 x 128 pass 32 / 32 users exact, KL(HF)
#      0.0698 = sequential), at the dense path's cost (README "Phase 3g rows" for the pass times).
# Module state so a test can toggle it inside one process (``packed_seq_numerics``); read through the accessor.
PACKED_SEQ_NUMERICS_ENV = "SOLAR_OPEN_PACKED_PREFILL_SEQ_NUMERICS"
PACKED_SEQ_NUMERICS_LEVELS = (0, 1, 2)
PACKED_SEQ_NUMERICS_DEFAULT = 2
# lm_head pieces of a level >= 1 head: ttnn's auto config for [M, 4096] x [4096, 32768] is the narrow 1D in0_block_w 2
# form up to M = 2048 (a width / height ratio > 8) and a 2D in0_block_w 1 form at M = 4096; pieces of at most this many
# rows were verified bit-identical to the 32-row head's matmul (D1 r8: M = 32 == 128 == 256 == 1024).
LM_HEAD_SEQ_ROWS = 1024
NARROW_SHAPE_RATIO_THRESHOLD = 8  # matmul_program_config.cpp::is_narrow_shape
TILE = 32


def read_packed_seq_numerics(raw=None):
    """``SOLAR_OPEN_PACKED_PREFILL_SEQ_NUMERICS`` (or ``raw``) as a level in PACKED_SEQ_NUMERICS_LEVELS; unset /
    empty = PACKED_SEQ_NUMERICS_DEFAULT; ``on`` / ``true`` = 1, ``off`` / ``false`` = 0."""
    raw = os.getenv(PACKED_SEQ_NUMERICS_ENV) if raw is None else raw
    text = (raw or "").strip().lower()
    if text == "":
        return PACKED_SEQ_NUMERICS_DEFAULT
    aliases = {"on": 1, "true": 1, "yes": 1, "off": 0, "false": 0, "no": 0}
    if text in aliases:
        return aliases[text]
    try:
        level = int(text)
    except ValueError:
        level = None
    if level not in PACKED_SEQ_NUMERICS_LEVELS:
        raise ValueError(f"{PACKED_SEQ_NUMERICS_ENV}={raw!r} is not one of {PACKED_SEQ_NUMERICS_LEVELS}")
    return level


PACKED_SEQ_NUMERICS = read_packed_seq_numerics()


def packed_seq_numerics_level():
    """The knob's current level (0 / 1 / 2) -- the module state, not the environment."""
    return PACKED_SEQ_NUMERICS


@contextlib.contextmanager
def packed_seq_numerics(level):
    """Set the knob to ``level`` for the enclosed block and restore it afterwards (tests: several arms per process)."""
    global PACKED_SEQ_NUMERICS
    if level not in PACKED_SEQ_NUMERICS_LEVELS:
        raise ValueError(f"packed_seq_numerics level {level!r} is not one of {PACKED_SEQ_NUMERICS_LEVELS}")
    previous = PACKED_SEQ_NUMERICS
    PACKED_SEQ_NUMERICS = int(level)
    try:
        yield
    finally:
        PACKED_SEQ_NUMERICS = previous


def is_narrow_shape(m, n, all_dram=True):
    """``matmul_program_config.cpp::is_narrow_shape``: the 1D systolic config is chosen for a height / width ratio above
    NARROW_SHAPE_RATIO_THRESHOLD, or (all-DRAM operands) for a dimension within one tile."""
    ratio = m // n if m > n else n // m
    if ratio > NARROW_SHAPE_RATIO_THRESHOLD:
        return True
    return bool(all_dram) and (m <= TILE or n <= TILE)


def auto_in0_block_w(m, n, k, grid_x=11):
    """``in0_block_w`` of the program config ttnn picks for a DRAM-interleaved ``[m, k] x [k, n]`` matmul without a
    user config (``matmul_program_config.cpp::create_simple_matmul_program_config``): the 1D systolic form for narrow
    shapes (``Kt % 2 == 0 ? 2 : 1``), the 2D mcast form otherwise (``Kt % grid_x == 0 ? Kt / grid_x : 1``; 1 for every
    K of Solar-Open on the 11-wide grid). Verified by bit-identity on device (D1 r7 / r8) for qkv (128 -> 2, 256 /
    1024 / 4096 -> 1), o_proj (<= 256 -> 2, >= 1024 -> 1), lm_head (32..1024 -> 2, 4096 -> 1) and the shared gate
    (256 -> 1)."""
    kt = -(-k // TILE)
    if is_narrow_shape(m, n):
        return 2 if kt % 2 == 0 else 1
    return kt // grid_x if kt % grid_x == 0 else 1


def seq_numerics_in0_block_w(m, rows, n, k, grid_x=11):
    """The ``in0_block_w`` a ``[m, k] x [k, n]`` matmul of a packed pass must be pinned to so that its rows come out as
    in the ``rows``-row (per-user) pass, or None when ttnn's auto choice at ``m`` rows already is the same."""
    want = auto_in0_block_w(rows, n, k, grid_x)
    return None if want == auto_in0_block_w(m, n, k, grid_x) else want


def row_pieces(total, piece):
    """``(start, length)`` of the pieces of ``piece`` rows (the last one shorter) covering ``total`` rows."""
    if total <= 0 or piece <= 0:
        raise ValueError(f"row_pieces needs positive sizes, got total={total}, piece={piece}")
    return [(start, min(piece, total - start)) for start in range(0, total, piece)]


def lm_head_piece_rows(total, n, k, ref_rows=TILE, max_rows=LM_HEAD_SEQ_ROWS, grid_x=11):
    """Rows per lm_head piece of a level >= 1 head: the largest tile-multiple <= ``max_rows`` (halving from there) whose
    auto in0_block_w equals the ``ref_rows``-row (sequential head) one; ``total`` when the whole matmul already
    matches. Pure."""
    if total <= max_rows and seq_numerics_in0_block_w(total, ref_rows, n, k, grid_x) is None:
        return total
    piece = min(max_rows, total)
    piece -= piece % TILE
    while piece > TILE and seq_numerics_in0_block_w(piece, ref_rows, n, k, grid_x) is not None:
        piece //= 2
        piece -= piece % TILE
    return max(piece, TILE)


def gather_head_rows(prompt_lens, seq_len, rows=GATHER_HEAD_ROWS):
    """Row indices ``[rows]`` into the ``[T, H]`` residual of a packed pass: user ``u``'s last prompt token
    ``u * seq_len + len_u - 1`` for the ``B = len(prompt_lens)`` users re-slotted to ``0..B-1``, then row 0 as the
    filler of the unused entries (a valid row: user 0's first token; its logits are never read)."""
    lens = [int(n) for n in prompt_lens]
    seq_len = int(seq_len)
    if not lens:
        raise ValueError("gather_head_rows needs at least one user")
    if len(lens) > rows:
        raise ValueError(f"{len(lens)} users exceed the {rows}-row gather head")
    for user, length in enumerate(lens):
        if not 0 < length <= seq_len:
            raise ValueError(
                f"user {user}: prompt length {length} must be in 1..{seq_len} (the pass's per-user length)"
            )
    idx = [user * seq_len + length - 1 for user, length in enumerate(lens)]
    return idx + [0] * (rows - len(idx))
