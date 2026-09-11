# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Paged-cache fill bounds of a single-user eager prefill (phase 3g / L1). Pure python (no ttnn): the rule, host-testable.

The tt_transformers Generator pads a prompt to the model's prefill length (Solar: 128, 1024, 2048, ...) and hands the
model a page table of exactly ``padded_len / block_size`` columns (``Generator._get_prefill_user_page_table``): the
request's real blocks followed by whatever the serving stack pads the row with -- under vLLM 0.25 + vllm-tt-plugin the
block table is zero-padded, and block 0 is vLLM's null block. ``attention/prefill.py`` fills ``min(T, columns x
block_size)`` rows through that table, so every padded block beyond the last real one is written (into the null block,
several times, under vLLM). Live finding (L1, P150x8, 2026-09-10): those writes corrupt the request's OWN decode
intermittently -- with the instruction of the prompt in its tail, prompts padded to 1024 with 0 / 1 / 5 / 13 pad
entries answered 4/4 / 2/4 / 2/4 / 0/4 times; the traced 128-token prefill (2 real blocks, no pad entry used) and the
demo (a real page table of 64 blocks) never fail. The model therefore bounds the fill to the blocks that hold real
tokens, the way Gemma4 treats ``get_last_token + 1`` as the real fill length (``Generator._chunk_prefill_get_last_token``).

Scope: the single-user EAGER Generator path only, recognised by its exact-fit table. The traced path pads its table to
the whole pool (never exact-fit), a packed pass (``batch_size > 1``) builds its own tables, and a chunked prefill is left
alone: its intermediate chunks inherit the LAST chunk's ``get_last_token`` (the Generator's legacy default), so bounding
them would under-fill their K / V. The last chunk of a > 32K-token prompt keeps its pad-entry writes (open item).
"""

TILE = 32


def fill_blocks_for_prompt(get_last_token, block_size):
    """Number of blocks that hold the prompt's real tokens, from the model's ``get_last_token`` (the first row of the
    32-row tile of the last real token) and the paged cache's ``block_size`` (a multiple of 32): the last tile that
    must be written ends at ``get_last_token + 32``, and ceil((get_last_token + 32) / block_size) equals
    ceil((last_token_idx + 1) / block_size) for every last_token_idx in that tile. None when ``get_last_token`` is -1
    (no last-token slicing requested) or unknown."""
    if get_last_token is None or get_last_token < 0 or block_size <= 0:
        return None
    return -(-(get_last_token + TILE) // block_size)


def fill_table_columns(table_columns, block_size, seq_len, batch_size, get_last_token, chunked):
    """Columns of the fill page table to keep for one prefill call, or None to leave the table as it is.

    ``table_columns``: width of the page table the call received; ``seq_len``: the (padded) row count of the pass;
    ``chunked``: a chunk_start_idx or chunk_page_table was given. Returns a count only when the table is the eager
    Generator's exact-fit table (``table_columns * block_size == seq_len``) of a single user with a last-token index and
    the real blocks are fewer than the columns."""
    if batch_size != 1 or chunked or table_columns is None or table_columns <= 0:
        return None
    if table_columns * block_size != seq_len:
        return None
    blocks = fill_blocks_for_prompt(get_last_token, block_size)
    if blocks is None or blocks >= table_columns:
        return None
    return blocks
