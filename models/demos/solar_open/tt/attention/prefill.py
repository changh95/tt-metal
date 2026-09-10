# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from .config import AttentionConfig, ProgramConfig
from .operations import (
    apply_allgather_and_slice,
    apply_allreduce,
    apply_output_projection,
    apply_output_projection_fused_rs,
    apply_qkv_projection,
    apply_rope,
    attention_bf16_output,
    concat_heads,
    is_shape_fused_mm_rs_supported,
    split_qkv_heads_prefill,
)
from .weights import AttentionWeights

# Per-user tokens of one prefill pass above which the attention branch output (o_proj output, all-reduce input) is bfp8
# instead of bf16 (phase 1; a 32K chunk of a chunked prefill is below it, a 64K single pass above -- the one attention-side
# difference between the two arms of the 64K parity pair in tests/test_chunked_prefill.py).
ATTENTION_BFP8_OUTPUT_ABOVE_TOKENS = 32 * 1024


def prefill_forward(
    hidden_states,
    rope_mats,
    weights: AttentionWeights,
    kv_cache,
    config: AttentionConfig,
    mesh_config,
    mesh_device,
    program_config: ProgramConfig,
    transformation_mat,
    position_idx,
    page_table,
    ccl_manager,
    user_id=0,
    batch_size=1,
    chunk_page_table=None,
    chunk_start_idx=None,
):
    """
    Prefill forward pass - optimized for sequence processing (seq_len>1).

    Bias-free, sink-free GQA (Solar-Open): fused QKV linear -> head split -> RoPE -> KV cache fill -> causal SDPA
    -> concat heads -> o_proj partial -> TP all-reduce.

    Chunked single-user prefill (phase 3d, ``tt/chunked_prefill.py``): the Generator cuts a prompt longer than
    ``ModelArgs.max_prefill_chunk_size`` into chunks and calls this once per chunk with ``rope_mats`` already offset
    to the chunk's absolute positions (``Model.prepare_inputs_prefill``), ``chunk_page_table`` = the page-table slice
    of the chunk's own blocks (the K / V fill target) and ``chunk_start_idx`` = the chunk's absolute start (a python
    int). With ``chunk_start_idx > 0`` the SDPA is ``ttnn.transformer.chunked_scaled_dot_product_attention`` over the
    paged cache: causal inside the chunk, full over the ``chunk_start_idx`` positions written by the earlier chunks
    (the chunk's own K / V go into the cache first, then the op reads the whole prefix back). Chunk 0 (``None`` or 0)
    takes the legacy path -- fill through its page-table slice, plain causal SDPA over its own K / V -- and is
    bit-identical to an unchunked prefill of the same length. Chunking needs the paged cache and ``batch_size == 1``.

    Args:
        hidden_states: Input tensor [1, 1, batch * seq_len, hidden_size]
        rope_mats: Tuple of (cos, sin) matrices for RoPE
        weights: Attention weights
        kv_cache: KV cache [k_cache, v_cache]
        config: Attention configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        program_config: Model-specific program configs
        transformation_mat: Transformation matrix for RoPE
        position_idx: Position indices (unused in prefill)
        page_table: Page table for paged attention (optional); for a chunked prefill the user's WHOLE table (every
            block of the prefix and this chunk), which the chunked SDPA reads the cache through
        ccl_manager: Communication manager
        chunk_page_table: page-table slice of this chunk's blocks (paged_fill_cache target); None = ``page_table``
        chunk_start_idx: absolute position of this chunk's first token (int); None / 0 = no cached prefix

    Returns:
        Attention output [1, 1, batch * seq_len, hidden_size], all-reduced over the TP axis (bf16 up to 32K tokens per
        user, bfloat8_b above). Phase 1 rounds the o_proj input to bfloat8_b; the SOLAR_OPEN_ATTENTION_BF16_OUTPUT
        option keeps it bf16 (operations.attention_bf16_output).
    """
    activation_dtype = ttnn.bfloat16
    total_seq_len = hidden_states.shape[-2]
    hidden_size = hidden_states.shape[-1]
    seq_len = total_seq_len // batch_size  # Per-user sequence length
    # Above this many tokens per user (per PASS: a 32K chunk of a longer prompt stays bf16) the o_proj output -- the
    # attention branch entering the residual stream -- is bfp8 (phase 1, DRAM at 64K: the bf16 [S, 4096] output is
    # 512 MiB). tests/test_chunked_prefill.py pins the constant to isolate the chunking mechanism from this rule.
    if seq_len > ATTENTION_BFP8_OUTPUT_ABOVE_TOKENS:
        activation_dtype = ttnn.bfloat8_b
    else:
        activation_dtype = ttnn.bfloat16

    # Validate prefill mode
    if seq_len <= 1:
        raise ValueError(f"Prefill mode requires seq_len>1, got {seq_len}. Use decode mode for single tokens.")

    # Chunked-prefill contract (the Model resolves the Generator's argument to a python int or None)
    if isinstance(chunk_start_idx, ttnn.Tensor):
        raise TypeError("chunk_start_idx must be a python int (Model.ttnn_prefill_forward resolves the device tensor)")
    chunk_start_idx = int(chunk_start_idx) if chunk_start_idx is not None else 0
    if chunk_start_idx < 0:
        raise ValueError(f"chunk_start_idx must be >= 0, got {chunk_start_idx}")
    chunked = chunk_start_idx > 0
    if (chunked or chunk_page_table is not None) and batch_size > 1:
        raise NotImplementedError("chunked prefill (chunk_page_table / chunk_start_idx) is single-user only")
    if (chunked or chunk_page_table is not None) and page_table is None:
        raise ValueError("chunked prefill needs the paged KV cache (page_table is None)")

    # QKV projection
    xqkv_fused = apply_qkv_projection(hidden_states, weights)
    hidden_states.deallocate(True)  # Free input activations after projection

    # Reshape for batch: [1, 1, B*S, QKV] -> [B, 1, S, QKV]
    if batch_size > 1:
        xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len, -1])

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)

    tt_q, tt_k, tt_v = split_qkv_heads_prefill(xqkv_fused, num_local_heads, num_local_kv_heads)
    xqkv_fused.deallocate(True)

    # Apply RoPE (use per-user seq_len positions). A packed multi-user pass (batch_size > 1) needs the S-row cos/sin:
    # Model.prepare_inputs_prefill already hands those over (cached per S), so the slice below only runs for callers
    # that still pass the T = B*S-row (or longer) matrices -- it is a device op that allocates per layer per call.
    if batch_size > 1 and rope_mats[0].shape[2] != seq_len:
        rope_mats_sliced = [rope_mats[0][:, :, :seq_len, :], rope_mats[1][:, :, :seq_len, :]]
    else:
        rope_mats_sliced = rope_mats
    tt_q_orig = tt_q
    tt_k_orig = tt_k
    tt_q = apply_rope(tt_q, rope_mats_sliced, transformation_mat, is_decode_mode=False)
    tt_k = apply_rope(tt_k, rope_mats_sliced, transformation_mat, is_decode_mode=False)
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Fill KV cache
    k_cache, v_cache = kv_cache
    tt_k_pre_cast = tt_k
    tt_v_pre_cast = tt_v
    tt_k = ttnn.typecast(tt_k, k_cache.dtype)
    tt_v = ttnn.typecast(tt_v, v_cache.dtype)
    tt_k_pre_cast.deallocate(True)
    tt_v_pre_cast.deallocate(True)

    if page_table is not None:
        block_size = k_cache.shape[2]
        # A chunked prefill writes the chunk's K / V into ITS blocks (the Generator's slice of the user's table); the
        # legacy call writes S tokens into the first S / block_size blocks of the whole table (the same blocks for chunk 0).
        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        page_len = fill_page_table.shape[-1] * block_size
        if batch_size > 1:
            # Per-user paged cache fill. The flattened approach (reshape batch into seq
            # + flattened page_table) produces wrong cache for users beyond the first —
            # paged_fill_cache doesn't correctly handle positions beyond the original
            # page_table's block count. Use per-user calls with batch_idx=0 instead.
            for b in range(batch_size):
                k_b = tt_k[b : b + 1, :, :, :]
                v_b = tt_v[b : b + 1, :, :, :]
                pt_b = page_table[b : b + 1, :]
                k_b_fill = k_b[:, :, :page_len, :] if page_len < k_b.shape[2] else k_b
                v_b_fill = v_b[:, :, :page_len, :] if page_len < v_b.shape[2] else v_b
                ttnn.experimental.paged_fill_cache(k_cache, k_b_fill, pt_b, batch_idx=0)
                ttnn.experimental.paged_fill_cache(v_cache, v_b_fill, pt_b, batch_idx=0)
        else:
            tt_k_sliced = tt_k[:, :, :page_len, :] if page_len < tt_k.shape[2] else tt_k
            tt_v_sliced = tt_v[:, :, :page_len, :] if page_len < tt_v.shape[2] else tt_v
            ttnn.experimental.paged_fill_cache(k_cache, tt_k_sliced, fill_page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(v_cache, tt_v_sliced, fill_page_table, batch_idx=user_id)
            if page_len < tt_k.shape[2]:
                tt_k_sliced.deallocate(True)
            if page_len < tt_v.shape[2]:
                tt_v_sliced.deallocate(True)

    else:
        # Non-paged attention
        if batch_size > 1:
            for b in range(batch_size):
                k_b = ttnn.slice(tt_k, (b, 0, 0, 0), (b + 1, tt_k.shape[1], tt_k.shape[2], tt_k.shape[3]))
                v_b = ttnn.slice(tt_v, (b, 0, 0, 0), (b + 1, tt_v.shape[1], tt_v.shape[2], tt_v.shape[3]))
                ttnn.fill_cache(k_cache, k_b, batch_idx=b)
                ttnn.fill_cache(v_cache, v_b, batch_idx=b)
                k_b.deallocate(True)
                v_b.deallocate(True)
        else:
            ttnn.fill_cache(k_cache, tt_k, batch_idx=user_id)
            ttnn.fill_cache(v_cache, tt_v, batch_idx=user_id)

    # Scaled dot-product attention (no attention-sink term; sliding_window is None for every Solar-Open layer)
    sdpa_config = program_config.get_prefill_sdpa_config(mesh_device, seq_len)
    if chunked:
        # Chunk i > 0 of a chunked prefill: attend over the paged cache -- the chunk_start_idx positions the earlier
        # chunks wrote plus this chunk's own K / V (filled above), causal inside the chunk. The op requires the start
        # to be a multiple of both SDPA chunk sizes (256 x 256 from 2048 tokens on; every 2048-multiple qualifies).
        if config.sliding_window is not None:
            raise NotImplementedError("sliding-window attention has no chunked prefill path")
        for name, size in (("q_chunk_size", sdpa_config.q_chunk_size), ("k_chunk_size", sdpa_config.k_chunk_size)):
            if chunk_start_idx % size:
                raise ValueError(f"chunk_start_idx {chunk_start_idx} is not a multiple of the SDPA {name} {size}")
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        tt_sdpa_out = ttnn.transformer.chunked_scaled_dot_product_attention(
            tt_q,
            k_cache,
            v_cache,
            page_table,
            chunk_start_idx,
            program_config=sdpa_config,
            compute_kernel_config=program_config.get_compute_kernel_config(),
        )
        tt_q.deallocate(True)
    else:
        tt_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            sliding_window_size=config.sliding_window,
            program_config=sdpa_config,
            compute_kernel_config=program_config.get_compute_kernel_config(),
        )
        tt_q.deallocate(True)
        tt_k.deallocate(True)
        tt_v.deallocate(True)

    # Concat heads and apply output projection
    tt_sdpa_out_pre_concat = tt_sdpa_out
    tt_sdpa_out = concat_heads(tt_sdpa_out, is_decode_mode=False)
    tt_sdpa_out_pre_concat.deallocate(True)

    # Flatten back for output projection: [B, 1, S, H] -> [1, 1, B*S, H]
    if batch_size > 1:
        tt_sdpa_out = ttnn.reshape(tt_sdpa_out, [1, 1, total_seq_len, -1])

    # Output projection + tensor-parallel allreduce.
    # When TP > 1 we use the fused matmul + reduce-scatter op; the trailing
    # all-gather + padding slice stay as separate ops. See
    # apply_output_projection_fused_rs for the per-shape tuned configs.
    if mesh_config.tp > 1 and is_shape_fused_mm_rs_supported(tt_sdpa_out):
        rs_out = apply_output_projection_fused_rs(tt_sdpa_out, weights, mesh_config, ccl_manager)
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size)
    else:
        tt_out = apply_output_projection(
            tt_sdpa_out, weights, activation_dtype, keep_bf16=attention_bf16_output(program_config)
        )
        tt_sdpa_out.deallocate(True)
        tt_out_result = apply_allreduce(tt_out, mesh_config, ccl_manager, hidden_size)
    return tt_out_result
