# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from ..ccl import FUSED_SITE_ATTENTION
from .config import AttentionConfig, ProgramConfig
from .operations import apply_rope, apply_rope_fused_qk, attention_bf16_output, update_kv_cache_fused
from .weights import AttentionWeights


def fused_qk_enabled(program_config, mesh_config) -> bool:
    """True when the decode chain runs the phase-3e fused Q/K RoPE + fused K/V update (``ProgramConfig.fused_qk``).

    Refused at TP = 1: the create-heads input is DRAM interleaved there and ``nlp_create_qkv_heads_decode`` forces
    ``overlap_qk_coregrid=True`` for a non-sharded input, so the 2B-core placement the fused ops need cannot be built.
    """
    fused = bool(getattr(program_config, "fused_qk", False))
    if fused and mesh_config.tp == 1:
        raise ValueError(
            "attention decode: fused_qk (SOLAR_OPEN_ATTENTION_FUSED_QK=1) needs the width-sharded qkv layout of "
            "TP > 1; unset it for a single-device run"
        )
    return fused


def decode_qkv_heads(
    xqkv_fused,
    rope_mats,
    kv_cache,
    config: AttentionConfig,
    mesh_config,
    mesh_device,
    program_config: ProgramConfig,
    transformation_mat,
    kv_mem_cfg,
    position_idx,
    page_table,
    batch_size: int,
):
    """Head split -> RoPE -> KV cache update of one decode step; returns the rotated Q ``[1, B, heads, head_dim]``
    height-sharded one user per core (the SDPA input). Frees ``xqkv_fused``.

    Legacy chain (``fused_qk`` False, phases 1-3d): ``nlp_create_qkv_heads_decode`` onto the B-core per-user grid,
    ``rotary_embedding_llama`` on Q and on K, ``to_memory_config(kv_mem_cfg)`` (no-op by construction) and two
    ``paged_update_cache``. Fused chain (phase 3e B1 slice 1): ``nlp_create_qkv_heads_decode(overlap_qk_coregrid=
    False)`` onto the 2B-core grid of ``get_decode_qk_fused_grids`` (Q / V on cores [0, B), K on [B, 2B)),
    ``rotary_embedding_llama_fused_qk`` and ``paged_fused_update_cache`` -- two launches fewer per layer; the two
    ``to_memory_config`` calls are dropped (K's grid differs from ``kv_mem_cfg``'s, and the fused update takes the
    create-heads grids as they are). ``rope_mats`` must then carry 2B rows (``RotarySetup(use_qk_fused=True)`` +
    doubled position ids, see ``tt/model.py::get_tt_pos_idx``) on exactly that grid.
    """
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    head_dim = config.head_dim
    fused_qk = fused_qk_enabled(program_config, mesh_config)

    # One user per core on the grid RoPE and SDPA decode expect (see ProgramConfig.get_decode_user_grid).
    # This placement is load-bearing: with a bare L1_HEIGHT_SHARDED_MEMORY_CONFIG the op falls back to
    # the *device* compute grid, which is 8 wide on Wormhole but 13 wide on Blackhole, so for a batch
    # that is a multiple of 32 user b would land on (b % 13, b // 13) while RotarySetup's cos/sin and
    # the paged SDPA reducer live at (b % 8, b // 8): every downstream op silently reads another user's
    # Q/K/V (no TT_FATAL). Batch 1 only worked because core (0, 0) coincides.
    create_heads_kwargs = {}
    if fused_qk:
        # 2B cores: Q (and V) of user b on core b, K of user b on core B + b -- the cores RotarySetup(use_qk_fused)
        # put cos/sin rows b and B + b on (get_decode_qk_fused_grids asserts the derivation).
        batch_grid, _, _ = program_config.get_decode_qk_fused_grids(mesh_device, batch_size)
        create_heads_kwargs["overlap_qk_coregrid"] = False
        cos_rows = rope_mats[0].shape[1]
        if cos_rows != 2 * batch_size:
            raise ValueError(
                f"fused_qk decode needs cos/sin with 2 x {batch_size} rows (Q users, then the same positions for K), "
                f"got {cos_rows}: build the RotarySetup with use_qk_fused=True and double the position ids "
                "(tt/model.py::create_rope_setup / get_tt_pos_idx)"
            )
    else:
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
        **create_heads_kwargs,
    )

    xqkv_fused.deallocate(True)

    k_cache, v_cache = kv_cache
    if fused_qk:
        # Apply RoPE to Q and K in one launch, then write K and V in one launch.
        tt_q_orig = tt_q
        tt_k_orig = tt_k
        tt_q, tt_k = apply_rope_fused_qk(tt_q, tt_k, rope_mats, transformation_mat)
        tt_q_orig.deallocate(True)
        tt_k_orig.deallocate(True)
        update_kv_cache_fused(k_cache, tt_k, v_cache, tt_v, position_idx, page_table)
        tt_k.deallocate(True)
        tt_v.deallocate(True)
        return tt_q

    # Apply RoPE
    tt_q_orig = tt_q
    tt_k_orig = tt_k
    tt_q = apply_rope(tt_q, rope_mats, transformation_mat, is_decode_mode=True)
    tt_k = apply_rope(tt_k, rope_mats, transformation_mat, is_decode_mode=True)
    tt_q_orig.deallocate(True)
    tt_k_orig.deallocate(True)

    # Update KV cache
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
    return tt_q


def decode_output_projection(
    tt_sdpa_out, weights: AttentionWeights, program_config: ProgramConfig, mesh_device, batch_size
):
    """o_proj of one decode step on the width-sharded ``nlp_concat_heads_decode`` output ``[1, 1, B, heads x head_dim]``
    (one head per core); returns the bf16 per-device partial ``[1, 1, B, hidden]`` in the matmul's natural layout:
    L1 WIDTH_SHARDED for the auto arm and the sharded-in0 arm, L1 INTERLEAVED for the interleaved-in0 arm. The
    all-reduce site in ``decode_forward`` (lane A) takes either (the fused kernel casts the sharded partial in place,
    the interleaved one is resharded onto the fused output grid; the composite path reshards to interleaved). Frees
    the input.

    ``decode_out_cores`` None (default, phases 1-3d): the auto linear writing L1 width-sharded. With
    ``decode_out_cores`` set (phase 3e B1 slice 2, SOLAR_OPEN_ATTENTION_OUT_GRID=8x8) and ``decode_out_interleave_in0``:
    the reshard the composite all-reduce needs anyway moves in FRONT of the matmul (``sharded_to_interleaved``, ~1 us
    either way) and the explicit 1D mcast config (8x8 cores x 2 output tiles, in0_block_w 4) runs from the interleaved
    in0 straight into an L1-interleaved output -- 23 -> ~12 us measured in the micro-benchmark; the HiFi2 compute
    config is restated explicitly (an explicit program config without one drops the bf16 x bfp8 matmul to LoFi).
    ``decode_out_interleave_in0=False`` keeps the phase-2 sharded-in0 arm (measured no gain: in0_block_w is pinned to
    the 4-tile shard width) for A/B.
    """
    k = tt_sdpa_out.shape[-1]
    n = weights.o_proj.shape[-1]
    out_program_config = program_config.get_decode_out_config(batch_size, n, k)
    if out_program_config is None:
        tt_out = ttnn.linear(
            tt_sdpa_out, weights.o_proj, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        )
        tt_sdpa_out.deallocate(True)
        return tt_out

    out_compute_config = program_config.get_decode_out_compute_config(mesh_device.arch())
    if getattr(program_config, "decode_out_interleave_in0", False):
        tt_in0 = ttnn.sharded_to_interleaved(tt_sdpa_out, ttnn.L1_MEMORY_CONFIG)
        tt_sdpa_out.deallocate(True)
        tt_out = ttnn.linear(
            tt_in0,
            weights.o_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=out_program_config,
            compute_kernel_config=out_compute_config,
        )
        tt_in0.deallocate(True)
        return tt_out

    tt_out = ttnn.linear(
        tt_sdpa_out,
        weights.o_proj,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=out_program_config,
        compute_kernel_config=out_compute_config,
    )
    tt_sdpa_out.deallocate(True)
    return tt_out


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
    SDPA decode -> concat heads -> o_proj partial -> TP all-reduce. Phase 3e (lane B1) levers, off by default:
    ``program_config.fused_qk`` (fused Q/K RoPE + fused K/V update, ``decode_qkv_heads``) and
    ``program_config.decode_out_cores`` + ``decode_out_interleave_in0`` (explicit o_proj from an interleaved in0,
    ``decode_output_projection``). Lane A owns the all-reduce call at the end of this function.

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
        Attention output [1, 1, batch, hidden_size], all-reduced over the TP axis; bfloat8_b by default, bfloat16 with
        the SOLAR_OPEN_ATTENTION_BF16_OUTPUT option (operations.attention_bf16_output). The all-reduce is the composite
        ``ttnn.all_reduce`` or, with SOLAR_OPEN_DECODE_CCL=fused, the fused ``all_reduce_async`` (tt/ccl.py).
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
    num_local_heads = mesh_config.shard_size(config.num_heads)
    num_local_kv_heads = mesh_config.shard_size(config.num_kv_heads)
    head_dim = config.head_dim
    # Explicit 1D program config for the width-sharded TP>1 layout (SolarOpenAttentionProgramConfig: (8, 5) cores
    # x 1 output tile, 60 -> 20 us per layer); None = auto. ttnn drops a bf16 x bfp8 matmul from HiFi2 to LoFi
    # when a program config is given, so the compute config restates the auto fidelity.
    qkv_program_config = None
    qkv_compute_config = None
    if mesh_config.tp > 1:
        qkv_n = (num_local_heads + 2 * num_local_kv_heads) * head_dim  # per-device fused N (1280 at TP=8)
        qkv_program_config = program_config.get_decode_qkv_config(batch_size, qkv_n, hidden_size)
        if qkv_program_config is not None:
            qkv_compute_config = program_config.get_decode_qkv_compute_config(mesh_device.arch())
    xqkv_fused = ttnn.matmul(
        hidden_states,
        weights.wqkv,
        dtype=ttnn.bfloat16,
        memory_config=qkv_memory_config,
        program_config=qkv_program_config,
        compute_kernel_config=qkv_compute_config,
    )

    # Split into Q, K, V heads, apply RoPE, write this step's K / V (legacy or fused chain: decode_qkv_heads)
    tt_q = decode_qkv_heads(
        xqkv_fused,
        rope_mats,
        kv_cache,
        config,
        mesh_config,
        mesh_device,
        program_config,
        transformation_mat,
        kv_mem_cfg,
        position_idx,
        page_table,
        batch_size,
    )
    k_cache, v_cache = kv_cache

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

    # Concat heads and apply output projection (auto linear, or the explicit config: decode_output_projection); the
    # result is the bf16 per-device partial, L1 width-sharded (auto / sharded-in0 arms) or L1 interleaved (the
    # interleaved-in0 arm of SOLAR_OPEN_ATTENTION_OUT_GRID).
    tt_sdpa_out = ttnn.experimental.nlp_concat_heads_decode(tt_sdpa_tensor, num_heads=num_local_heads)
    tt_sdpa_tensor.deallocate(True)

    tt_out = decode_output_projection(tt_sdpa_out, weights, program_config, mesh_device, batch_size)

    # Phase 3e / A2 (SOLAR_OPEN_DECODE_CCL=fused): ONE fused kernel (ttnn.experimental.all_reduce_async on the
    # CCLManager's persistent pair of this site) instead of the composite RS + AG below. The bf16 o_proj partial is
    # cast to bfloat8_b in its own layout (the composite path casts the interleaved copy), reduced into the 8x4
    # width-sharded output and brought back to the L1-interleaved [1, 1, B, hidden] tensor the residual add reads --
    # the composite path's output contract, with the same dtype (bf16 stays bf16 with the bf16-output option). A
    # width-sharded partial feeds the kernel directly; the interleaved partial of the interleaved-in0 o_proj arm is
    # resharded onto the fused output grid first (fused_decode_all_reduce_interleaved: i2s -> kernel -> s2i).
    # Not bit-identical to the composite (a different, deterministic reduction order; A1: closer to the fp32 sum).
    if mesh_config.tp > 1 and ccl_manager.fused_decode_applies(tt_out, hidden_size):
        if not attention_bf16_output(program_config):
            tt_cast = ttnn.typecast(tt_out, ttnn.bfloat8_b)
            tt_out.deallocate(True)
            tt_out = tt_cast
        if tt_out.memory_config().is_sharded():
            tt_reduced = ccl_manager.fused_decode_all_reduce(
                tt_out, FUSED_SITE_ATTENTION, cluster_axis=mesh_config.tp_axis
            )
            tt_out.deallocate(True)
            tt_out = ttnn.to_memory_config(tt_reduced, ttnn.L1_MEMORY_CONFIG)
            tt_reduced.deallocate(True)
        else:
            tt_out = ccl_manager.fused_decode_all_reduce_interleaved(
                tt_out, FUSED_SITE_ATTENTION, cluster_axis=mesh_config.tp_axis
            )
        return ttnn.reshape(tt_out, (1, 1, batch_size, hidden_size), (1, 1, ttnn.TILE_SIZE, hidden_size))

    # Bias-free: go straight from the width-sharded matmul output to the L1-interleaved tensor the reshape and
    # all_reduce below consume (the folded o_proj bias add used to perform this reshard as a side effect; a no-op for
    # the already-interleaved partial of the interleaved-in0 o_proj arm). Phase 1 rounds the per-device partial to
    # bfloat8_b before the 8-way sum; the bf16-output option keeps it bf16 (one launch fewer per layer, 256 KiB
    # instead of 136 KiB per device on the all_reduce wire).
    tt_out = ttnn.to_memory_config(tt_out, ttnn.L1_MEMORY_CONFIG)
    if not attention_bf16_output(program_config):
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
