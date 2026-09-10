# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure helpers of the chunked single-user prefill (phase 3d). No ttnn import: ``tests/unit/test_chunked_prefill.py``
exercises the arithmetic without a device, and the demo reads the chunk knob before any model is built.

Mechanism (tt_transformers ``Generator.prefill_forward_single_user_text``, unchanged): a single-user prompt whose
PADDED length exceeds ``ModelArgs.max_prefill_chunk_size`` is cut into equal chunks of ``chunk_size`` tokens (the
largest multiple of 2048 that divides the padded length and is <= ``max_prefill_chunk_size``; padded lengths are
powers of two, so this is ``max_prefill_chunk_size`` itself). Chunk ``i`` runs ``Model.ttnn_prefill_forward`` on
tokens ``[i * chunk_size, (i + 1) * chunk_size)`` with

* ``chunk_start_idx = i * chunk_size`` (a python int): RoPE cos / sin rows ``[start, start + chunk_size)`` of the
  131072-row YaRN tables (``Model.prepare_inputs_prefill``), and for ``start > 0`` the attention runs
  ``ttnn.transformer.chunked_scaled_dot_product_attention`` over the paged KV cache -- causal over the chunk, full
  over the ``start`` cached positions before it (``attention/prefill.py``);
* ``chunk_page_table = page_table[:, start // block_size : end // block_size]``: the chunk's K / V are written into
  its own blocks with ``paged_fill_cache`` before the SDPA reads the cache.

Chunk 0 has no prefix: it takes the legacy path (fill through its page-table slice, plain causal SDPA over its own
K / V) and is therefore bit-identical to an unchunked prefill of ``chunk_size`` tokens. The Generator stops after the
chunk that holds the last prompt token (trailing padding-only chunks never run). Knob: ``SOLAR_OPEN_PREFILL_CHUNK_TOKENS``
(default 32768; a power of two in 2048..131072; 131072 = never chunk = the phase-3c behaviour).
"""

import os
from dataclasses import dataclass

PREFILL_CHUNK_TOKENS_ENV = "SOLAR_OPEN_PREFILL_CHUNK_TOKENS"
# 32K: the longest prefill whose per-layer numerics the phase-3c tree records (bf16 attention output, 1024-token MoE
# splits, no ttnn.move of the residual); a 64K single pass switches the attention output to bfloat8_b above 32K
# (attention/prefill.py) and holds ~1.5-2.5 GiB of transients (README "Memory per device"), a 128K single pass was
# never run on the 1x8 mesh (DRAM). 32K chunks keep every recorded metric <= 32K bit-identical and bound the
# transient peak at the 32K prefill's whatever the prompt length.
PREFILL_CHUNK_TOKENS_DEFAULT = 32 * 1024
# The Generator's MIN_CHUNK_SIZE (models/tt_transformers/tt/common.py get_max_prefill_chunk_size).
PREFILL_CHUNK_TOKENS_MIN = 2048
# max_position_embeddings of Solar-Open-100B: one chunk of this size == the unchunked prefill.
MAX_POSITIONS = 128 * 1024
PREFILL_CHUNK_TOKENS_MAX = MAX_POSITIONS
# Longest UNCHUNKED single-user prefill validated on the 1x8 mesh (demo prefill_64k, TTFT ~29-30 s); above it the
# demo admits a prompt only when the chunk knob makes the Generator chunk it.
SINGLE_ROW_UNCHUNKED_PREFILL_CAP = 64 * 1024
# The chunked SDPA needs chunk_start_idx to be a multiple of its q and k chunk sizes: SolarOpenAttentionProgramConfig
# uses 256 x 256 from 2048 tokens on (prefill_threshold), 64 x 64 below -- every 2048-multiple satisfies both.
SDPA_CHUNK_ALIGN = 256
LAST_TOKEN_TILE = 32  # the Generator slices the 32-row tile that holds the last prompt token before norm + lm_head


def validate_prefill_chunk_tokens(chunk_tokens):
    """``chunk_tokens`` as an int, or ValueError: a power of two in ``PREFILL_CHUNK_TOKENS_MIN..PREFILL_CHUNK_TOKENS_MAX``
    (the Generator chunks in multiples of 2048; padded prefill lengths are powers of two, so a power-of-two chunk
    divides every padded length above it exactly)."""
    try:
        n = int(chunk_tokens)
    except (TypeError, ValueError):
        raise ValueError(f"{PREFILL_CHUNK_TOKENS_ENV} must be an integer, got {chunk_tokens!r}") from None
    if n < PREFILL_CHUNK_TOKENS_MIN or n > PREFILL_CHUNK_TOKENS_MAX or n & (n - 1):
        raise ValueError(
            f"{PREFILL_CHUNK_TOKENS_ENV}={n}: the prefill chunk must be a power of two between "
            f"{PREFILL_CHUNK_TOKENS_MIN} and {PREFILL_CHUNK_TOKENS_MAX} tokens ({PREFILL_CHUNK_TOKENS_MAX} = never chunk)"
        )
    return n


def prefill_chunk_tokens_from_env(default=PREFILL_CHUNK_TOKENS_DEFAULT):
    """``SOLAR_OPEN_PREFILL_CHUNK_TOKENS`` validated, or ``default`` when unset / empty."""
    raw = os.getenv(PREFILL_CHUNK_TOKENS_ENV)
    if raw is None or not raw.strip():
        return validate_prefill_chunk_tokens(default)
    return validate_prefill_chunk_tokens(raw.strip())


def single_row_prefill_cap(chunk_tokens, max_positions=MAX_POSITIONS):
    """Longest single-user prompt (padded length) the demo admits on a single-row mesh: every position of the RoPE
    tables once the Generator chunks at or below the validated unchunked length, else that unchunked cap."""
    chunk_tokens = validate_prefill_chunk_tokens(chunk_tokens)
    return int(max_positions) if chunk_tokens <= SINGLE_ROW_UNCHUNKED_PREFILL_CAP else SINGLE_ROW_UNCHUNKED_PREFILL_CAP


def rope_slice_bounds(start_pos, seq_len, table_len):
    """``(start, end)`` rows of the prefill cos / sin tables for ``seq_len`` tokens starting at absolute position
    ``start_pos``; ValueError when the chunk runs past the ``table_len`` positions the tables were built for."""
    start, seq_len, table_len = int(start_pos), int(seq_len), int(table_len)
    if start < 0 or seq_len <= 0:
        raise ValueError(f"rope slice needs start_pos >= 0 and seq_len > 0, got {start} / {seq_len}")
    end = start + seq_len
    if end > table_len:
        raise ValueError(
            f"prefill chunk at positions [{start}, {end}) exceeds the {table_len} positions of the RoPE tables "
            "(max_position_embeddings)"
        )
    return start, end


def generator_chunk_size(seq_len, max_chunk_tokens):
    """Chunk size the Generator uses for a ``seq_len``-token (padded) prefill above ``max_chunk_tokens``: the largest
    multiple of 2048 that divides ``seq_len`` and is <= ``max_chunk_tokens`` (mirrors ``get_max_prefill_chunk_size``).
    """
    seq_len, max_chunk_tokens = int(seq_len), int(max_chunk_tokens)
    if seq_len <= 0 or max_chunk_tokens <= 0:
        raise ValueError("seq_len and max_chunk_tokens must be positive")
    if seq_len % PREFILL_CHUNK_TOKENS_MIN or max_chunk_tokens % PREFILL_CHUNK_TOKENS_MIN:
        raise ValueError(f"seq_len ({seq_len}) and max_chunk_tokens ({max_chunk_tokens}) must be multiples of 2048")
    for chunk in range(min(seq_len, max_chunk_tokens), 0, -PREFILL_CHUNK_TOKENS_MIN):
        if seq_len % chunk == 0:
            return chunk
    raise ValueError(f"no multiple of 2048 <= {max_chunk_tokens} divides {seq_len}")  # unreachable: 2048 divides


@dataclass(frozen=True)
class PrefillChunk:
    """One Generator-level chunk: absolute token range ``[start, end)``, its page-table block range
    ``[block_start, block_end)``, the ``get_last_token`` row offset handed to the model and whether the Generator
    returns this chunk's logits (the chunk holding the last prompt token; later chunks never run)."""

    start: int
    end: int
    block_start: int
    block_end: int
    get_last_token: int
    is_last: bool

    @property
    def chunk_start_idx(self):
        """The ``chunk_start_idx`` the layers receive (``None`` = the legacy path of chunk 0)."""
        return self.start if self.start > 0 else None


def chunk_schedule(seq_len, max_chunk_tokens, block_size, last_token_idx, num_cached_tokens=0):
    """The chunks ``Generator.prefill_forward_single_user_text`` runs for a padded ``seq_len``-token prefill
    (``num_cached_tokens`` already in the cache, ``last_token_idx`` absolute) under ``max_chunk_tokens``.

    Mirrors the Generator's arithmetic exactly, including the early return after the chunk that holds
    ``last_token_idx``. A prefill that fits one chunk (and resumes nothing) is a single legacy pass.
    """
    seq_len, block_size, last_token_idx = int(seq_len), int(block_size), int(last_token_idx)
    num_cached_tokens = int(num_cached_tokens)
    max_chunk_tokens = int(max_chunk_tokens)
    if seq_len <= 0 or block_size <= 0 or num_cached_tokens < 0:
        raise ValueError("seq_len and block_size must be positive, num_cached_tokens >= 0")
    if not 0 <= last_token_idx < seq_len + num_cached_tokens:
        raise ValueError(f"last_token_idx {last_token_idx} outside [0, {seq_len + num_cached_tokens})")
    if seq_len <= max_chunk_tokens and num_cached_tokens == 0:
        blocks = -(-seq_len // block_size)
        return (PrefillChunk(0, seq_len, 0, blocks, (last_token_idx // LAST_TOKEN_TILE) * LAST_TOKEN_TILE, True),)
    chunk_size = generator_chunk_size(seq_len, max_chunk_tokens) if seq_len > max_chunk_tokens else seq_len
    if chunk_size % block_size:
        raise ValueError(f"chunk size {chunk_size} is not a multiple of the page block {block_size}")
    if chunk_size % SDPA_CHUNK_ALIGN:
        raise ValueError(f"chunk size {chunk_size} is not a multiple of the SDPA chunk {SDPA_CHUNK_ALIGN}")
    last_in_seq = last_token_idx - num_cached_tokens
    last_in_chunk = last_in_seq % chunk_size
    last_chunk_start = (last_in_seq // chunk_size) * chunk_size
    get_last_token = (last_in_chunk // LAST_TOKEN_TILE) * LAST_TOKEN_TILE
    chunks = []
    for start in range(num_cached_tokens, num_cached_tokens + seq_len, chunk_size):
        end = start + chunk_size
        is_last = (start - num_cached_tokens) == last_chunk_start
        chunks.append(PrefillChunk(start, end, start // block_size, end // block_size, get_last_token, is_last))
        if is_last:
            break
    return tuple(chunks)
