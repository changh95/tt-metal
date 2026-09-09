# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure helpers of the packed multi-user prefill's GATHER head (phase 3c, design_packed_prefill.md 2.6 item 1). No
ttnn import: ``tests/unit/test_sorted_moe_chunk_plan.py`` exercises the arithmetic without a device.

A packed pass of ``B_pad`` users x ``S`` tokens leaves the residual stream as ``[1, 1, T = B_pad * S, H]``. The v1
head (the tt_transformers batched path) runs norm + lm_head on all ``T`` rows and reads one 32-row tile per user back;
the gather head picks the ``B`` last-token rows ``u * S + len_u - 1`` into ONE ``[1, 1, 32, H]`` tile row (padded with
row 0), so norm + lm_head run on 32 rows -- the shapes of the single-user head -- and the host reads 32 x vocab once.
"""

GATHER_HEAD_ROWS = 32  # one tile row: the head's shapes are those of the single-user prefill head


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
