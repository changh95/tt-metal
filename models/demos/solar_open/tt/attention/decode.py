# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from .config import AttentionConfig, ProgramConfig
from .operations import apply_rope
from .weights import AttentionWeights


def decode_forward(
    hidden_states,
    rope_mats,
    weights: AttentionWeights,
    kv_cache,
    config: AttentionConfig,
    mesh_config,
    mesh_device,
    program_config: ProgramConfig,
    transformation_mat,
    kv_mem_cfg,
    position_idx,
    page_table,
    ccl_manager,
):
    """
    Decode forward pass - optimized for single token (seq_len=1).

    Bias-free, sink-free GQA (Solar-Open): fused QKV matmul -> per-user head split -> RoPE -> KV cache update ->
    SDPA decode -> concat heads -> o_proj partial -> TP all-reduce.

    Args:
        hidden_states: Input tensor [1, 1, batch, hidden_size] (tile layout, interleaved)
        rope_mats: Tuple of (cos, sin) matrices for RoPE, one user per core (RotarySetup.get_rot_mats)
        weights: Attention weights
        kv_cache: KV cache [k_cache, v_cache]
        config: Attention configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        program_config: Model-specific program configs
        transformation_mat: Transformation matrix for RoPE
        kv_mem_cfg: Memory config for KV tensors
        position_idx: Current position index per user
        page_table: Page table for paged attention (optional)
        ccl_manager: Communication manager

    Returns:
        Attention output [1, 1, batch, hidden_size] bfloat8_b, all-reduced over the TP axis
    """
    _, seq_len, batch_size, hidden_size = hidden_states.shape

    # Validate decode mode
    if seq_len != 1:
        raise ValueError(f"Decode mode requires seq_len=1, got {seq_len}")

    # QKV projection. With TP>1 the per-device QKV is small enough to fit in
    # an L1 width-sharded layout that nlp_create_qkv_heads_decode consumes
    # directly. With TP=1 (e.g. single Blackhole card) the per-device QKV is
    # TP× larger and overflows the per-core CB if width-sharded, so we use
    # DRAM interleaved instead. The kernel-side aligned-read fix in
    # nlp_create_qkv_heads_decode (PR #43292) handles the BH NOC alignment
    # constraint for DRAM-interleaved inputs; before that fix this path
    # produced silent corruption (every odd Q/K/V head returned the previous
    # user's row).
    qkv_memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mesh_config.tp > 1 else ttnn.DRAM_MEMORY_CONFIG
    xqkv_fused = ttnn.matmul(hidden_states, weights.wqkv, dtype=ttnn.bfloat16, memory_config=qkv_memory_config)

    # Split into Q, K, V heads
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    head_dim = config.head_dim

    # One user per core on the grid RoPE and SDPA decode expect (see ProgramConfig.get_decode_user_grid).
    # This placement is load-bearing: with a bare L1_HEIGHT_SHARDED_MEMORY_CONFIG the op falls back to
    # the *device* compute grid, which is 8 wide on Wormhole but 13 wide on Blackhole, so for a batch
    # that is a multiple of 32 user b would land on (b % 13, b // 13) while RotarySetup's cos/sin and
    # the paged SDPA reducer live at (b % 8, b // 8): every downstream op silently reads another user's
    # Q/K/V (no TT_FATAL). Batch 1 only worked because core (0, 0) coincides.
    batch_grid, _ = program_config.get_decode_user_grid(mesh_device, batch_size)
    qkv_heads_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_q, tt_k, tt_v = ttnn.experimental.nlp_create_qkv_heads_decode(
        xqkv_fused,
        num_heads=num_local_heads,
        num_kv_heads=num_local_kv_heads,
        memory_config=qkv_heads_mem_config,
    )

    xqkv_fused.deallocate(True)

    # Apply RoPE
    tt_q_orig = tt_q
    tt_k_orig = tt_k
    tt_q = apply_rope(tt_q, rope_mats, transformation_mat, is_decode_mode=True)
    tt_k = apply_rope(tt_k, rope_mats, transformation_mat, is_decode_mode=True)
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Update KV cache
    k_cache, v_cache = kv_cache
    tt_k = ttnn.to_memory_config(tt_k, kv_mem_cfg)
    tt_v = ttnn.to_memory_config(tt_v, kv_mem_cfg)

    ttnn.experimental.paged_update_cache(
        k_cache,
        tt_k,
        update_idxs_tensor=position_idx,
        page_table=page_table,
    )
    ttnn.experimental.paged_update_cache(
        v_cache,
        tt_v,
        update_idxs_tensor=position_idx,
        page_table=page_table,
    )

    tt_k.deallocate(True)
    tt_v.deallocate(True)

    # Calculate padded heads (must be tile-aligned, e.g., 32)
    # Use local heads per device, not global heads
    padded_heads = ((num_local_heads + 31) // 32) * 32

    # SDPA writes to DRAM; reshard onto a rectangular one-core-per-user grid for nlp_concat_heads_decode
    # (it needs a single CoreRange as input grid; the RoPE/SDPA user grid is not one for 8 < B < 32 on
    # Blackhole's 13-wide compute grid).
    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(padded_heads, head_dim),  # Shape per shard (tile-aligned)
        core_grid=program_config.get_decode_concat_grid(batch_size),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    # Scaled dot-product attention (no attention-sink term; sliding_window is None for every Solar-Open layer)
    if page_table is not None:
        tt_sdpa_tensor = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=position_idx,
            sliding_window_size=config.sliding_window,
            page_table_tensor=page_table,
            scale=config.scaling,
            program_config=program_config.get_decode_sdpa_config(mesh_device, batch_size),
            compute_kernel_config=program_config.get_compute_kernel_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_sdpa_tensor = ttnn.to_memory_config(tt_sdpa_tensor, height_sharded_mem_config)
    else:
        # GQA (num_kv_heads > 1) rejects sharded output in the SDPA decode
        # device op — match the paged path: write to DRAM, then to_memory_config
        # into the height-sharded layout that downstream concat_heads needs.
        tt_sdpa_tensor = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            k_cache,
            v_cache,
            cur_pos_tensor=position_idx,
            sliding_window_size=config.sliding_window,
            scale=config.scaling,
            program_config=program_config.get_decode_sdpa_config(mesh_device, batch_size),
            compute_kernel_config=program_config.get_compute_kernel_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_sdpa_tensor = ttnn.to_memory_config(tt_sdpa_tensor, height_sharded_mem_config)
    tt_q.deallocate(True)

    # Concat heads and apply output projection
    tt_sdpa_out = ttnn.experimental.nlp_concat_heads_decode(tt_sdpa_tensor, num_heads=num_local_heads)
    tt_sdpa_tensor.deallocate(True)

    tt_out = ttnn.linear(
        tt_sdpa_out, weights.o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    )

    tt_sdpa_out.deallocate(True)
    # Bias-free: go straight from the width-sharded matmul output to the L1-interleaved bfloat8_b tensor the reshape
    # and all_reduce below consume (the folded o_proj bias add used to perform this reshard as a side effect).
    tt_out = ttnn.to_memory_config(tt_out, ttnn.L1_MEMORY_CONFIG)
    tt_out = ttnn.typecast(tt_out, ttnn.bfloat8_b)

    # Calculate padded hidden size for tile-aligned CCL operations.
    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    padded_hidden = padded_local_hidden * mesh_config.tp if mesh_config.tp > 1 else hidden_size

    tt_out = ttnn.reshape(
        tt_out,
        (1, 1, batch_size, padded_hidden),
        (1, 1, 32, padded_hidden),
    )

    # Drop the tile-alignment padding columns: [1, 1, B, padded_hidden] -> [1, 1, B, hidden_size]. No-op for
    # Solar-Open-100B: 4096 / 8 = 512 is tile aligned, so padded_hidden == hidden_size.
    if padded_hidden != hidden_size and mesh_config.tp > 1:
        tt_out = ttnn.slice(
            tt_out,
            starts=[0, 0, 0, 0],
            ends=[1, 1, batch_size, hidden_size],
            steps=[1, 1, 1, 1],
        )
        # Workaround: ttnn.slice on bf8 TILE produces a buffer with non-standard
        # page layout under L1 fragmentation that CCL all_reduce misreads.
        # DRAM round-trip normalizes the buffer. See #41640 for repro and root cause analysis.
        tt_out = ttnn.to_memory_config(tt_out, ttnn.DRAM_MEMORY_CONFIG)
        tt_out = ttnn.to_memory_config(tt_out, ttnn.L1_MEMORY_CONFIG)

    # Tensor parallel all-reduce (AllBroadcast, ~80μs vs RS+AG ~138μs).
    if mesh_config.tp > 1:
        tt_out = ttnn.all_reduce(
            tt_out,
            num_links=ccl_manager.num_links,
            topology=ttnn.Topology.Ring,
            cluster_axis=mesh_config.tp_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    return tt_out
