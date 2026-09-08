# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import os

import ttnn

from .weights import AttentionWeights

# Phase 2 (design_misc.md (a)): keep the attention branch output in bf16 through o_proj and the TP all_reduce instead
# of rounding it to bfp8 first. Default off = the phase-1 numerics recorded in README "Numerics". Weights are untouched,
# so the option needs no cache rebuild and is deliberately NOT part of MoEOptions / the weight-cache marker.
ATTENTION_BF16_OUTPUT_ENV = "SOLAR_OPEN_ATTENTION_BF16_OUTPUT"


def attention_bf16_output(program_config=None) -> bool:
    """True when the attention output stays bf16 through o_proj and the all_reduce (off by default).

    Enabled by ``SOLAR_OPEN_ATTENTION_BF16_OUTPUT=1`` or by a ``bf16_output=True`` attribute on the attention program
    config (either source suffices; the base ProgramConfig carries no such field). Prefill: the o_proj input is not cast
    to bfp8, so the bf16 x bfp8 matmul runs HiFi2 instead of the LoFi ttnn picks for two bfp8 operands
    (matmul_device_operation.cpp: HiFi2 unless both inputs are bfp8/bfp4) and the ``[S, 1024]`` bfp8 copy disappears.
    Decode: the per-device o_proj partial enters the 8-way all_reduce as bf16 (``[1, 1, 32, 4096]``: 256 KiB instead of
    136 KiB per layer per device) and 48 typecast launches per step disappear. Numerics/perf are measured only through
    the teacher-forced test and the demo step times (device lane); the default stays off until that says otherwise.
    """
    if getattr(program_config, "bf16_output", False):
        return True
    return os.getenv(ATTENTION_BF16_OUTPUT_ENV, "0") == "1"


def apply_qkv_projection(hidden_states, weights: AttentionWeights):
    """
    Apply the fused, bias-free QKV projection.

    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        weights: Attention weights container

    Returns:
        Fused QKV tensor [batch, seq_len, total_qkv_dim]
    """
    xqkv_fused = ttnn.linear(hidden_states, weights.wqkv, dtype=ttnn.bfloat16)
    return xqkv_fused


def split_qkv_heads_prefill(xqkv_fused, num_heads: int, num_kv_heads: int):
    """
    Split QKV into separate head tensors for prefill mode.

    Args:
        xqkv_fused: Fused QKV tensor
        num_heads: Number of Q heads
        num_kv_heads: Number of K/V heads

    Returns:
        Tuple (Q, K, V) with shapes [1, num_heads, seq_len, head_dim]
    """
    return ttnn.experimental.nlp_create_qkv_heads(
        xqkv_fused,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def apply_rope(tensor, rope_mats, transformation_mat, is_decode_mode: bool):
    """
    Apply rotary position embedding (RoPE).

    Args:
        tensor: Input tensor (Q or K)
        rope_mats: Tuple of (cos, sin) matrices
        transformation_mat: Transformation matrix for the mode
        is_decode_mode: Whether in decode mode

    Returns:
        Tensor with RoPE applied
    """
    return ttnn.experimental.rotary_embedding_llama(
        tensor, rope_mats[0], rope_mats[1], transformation_mat, is_decode_mode=is_decode_mode
    )


def concat_heads(tensor, is_decode_mode: bool):
    """
    Concatenate attention heads back to hidden dimension.

    Args:
        tensor: Attention output tensor with separate heads
        is_decode_mode: Whether in decode mode

    Returns:
        Tensor with concatenated heads [batch, seq_len, hidden_size]
    """
    if is_decode_mode:
        tensor = ttnn.transpose(tensor, 1, 2)
    return ttnn.experimental.nlp_concat_heads(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def apply_output_projection(tensor, weights: AttentionWeights, activation_dtype, keep_bf16=False):
    """
    Apply the bias-free output projection (per-device partial sum; the TP all-reduce follows).

    Args:
        tensor: Attention output tensor (bf16 after concat_heads); the caller frees it
        weights: Attention weights container
        activation_dtype: Target dtype for output
        keep_bf16: Skip the bfp8 typecast of the input when ``activation_dtype`` is bf16 (see
            ``attention_bf16_output``: the bf16 x bfp8 matmul then runs HiFi2). With a bfp8 activation dtype (prefill
            above 32K, prefill.py) the cast stays: the output is block-quantised anyway and a bf16 ``[S, 1024]`` input
            would only cost DRAM at that length.

    Returns:
        Output tensor after projection
    """
    if keep_bf16 and activation_dtype == ttnn.bfloat16:
        return ttnn.matmul(tensor, weights.o_proj, dtype=activation_dtype)
    tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)
    out = ttnn.matmul(tensor, weights.o_proj, dtype=activation_dtype)
    tensor.deallocate(True)
    return out


# Per-M_tiles tuned config for the fused attention o_proj matmul + reduce-scatter.
# Tuple layout: (grid_y, M_block, K_block, N_block, chunk_width, subblock_h, subblock_w, num_workers).
# S=1024 (M_tiles=32): min=256us on Tracy (2026-04-23 sweep); see commit for methodology.
_FUSED_MM_RS_CONFIGS = {
    32: (4, 4, 4, 6, 2, 2, 2, 3),  # S=1024
}


def is_shape_fused_mm_rs_supported(tensor) -> bool:
    # #46181: the async fused matmul+reduce_scatter (minimal_matmul_strided_reduce_scatter_async)
    # RACES on Blackhole at M_tiles=32 (S=1024): the reduce-scatter half reads MM output blocks
    # before they are fully written (semaphore/overlap sync bug) -> non-deterministic garbage
    # (absmax ~1e13). It is validated/used on Wormhole (e.g. WH-GLX), so gate the fused path off on
    # Blackhole only and fall back to the correct non-fused matmul+allreduce path there. Remove this
    # gate once the fused-op sync is fixed for Blackhole.
    if "blackhole" in ttnn.get_arch_name():
        return False
    m_tiles = (tensor.shape[-2] + 31) // 32
    return m_tiles in _FUSED_MM_RS_CONFIGS


def apply_output_projection_fused_rs(tensor, weights: AttentionWeights, mesh_config, ccl_manager):
    """Attention output projection + TP reduce-scatter fused into one device op.

    Replaces the sequential `apply_output_projection` + first half of
    `mesh_config.allreduce` (the reduce-scatter). The trailing all-gather and
    padding-trim stay as separate ops in `apply_allgather_and_slice`.

    Internally uses `ttnn.experimental.minimal_matmul_strided_reduce_scatter_async`,
    which overlaps MM compute on one half of the core grid with the RS ring
    traffic on the other half — MM signals RS via on-chip semaphore per output
    block so RS starts forwarding as soon as the first block lands in DRAM.

    Dtypes (end-to-end bf8_b through the TP allreduce):
      - activation cast to bf8_b before MM (DRAM-bandwidth saver)
      - weight is bf8_b (as loaded, row-parallel sharded on K across TP=8)
      - MM math: LoFi (same as `ttnn.matmul` default, via compute_kernel_config)
      - fused-op output inherits input dtype → bf8_b
      - downstream all-gather + padding-slice operate on bf8_b

    Returns the reduce-scattered tensor of shape
    [1, 1, S, padded_hidden_total / TP] in bf8_b.

    Only valid for prefill (TP > 1); the decoder path uses a different pattern.
    """
    TILE = 32
    K = tensor.shape[-1]
    N = weights.o_proj.shape[-1]
    assert K % TILE == 0 and N % TILE == 0, f"K={K}, N={N} must be tile-aligned"

    M_tiles = (tensor.shape[-2] + TILE - 1) // TILE
    K_tiles = K // TILE
    N_tiles = N // TILE

    if M_tiles not in _FUSED_MM_RS_CONFIGS:
        raise ValueError(
            f"No tuned fused o_proj MM+RS config for M_tiles={M_tiles}; "
            f"tuned shapes: {sorted(_FUSED_MM_RS_CONFIGS)}. "
            f"Use apply_output_projection + apply_allreduce for untuned shapes."
        )
    grid_y, m_block, k_block, n_block, chunk_width, subblock_h, subblock_w, num_workers = _FUSED_MM_RS_CONFIGS[M_tiles]

    mm_core_grid = ttnn.CoreCoord(8, grid_y)
    Nt_per_core = N_tiles // mm_core_grid.x
    Mt_per_core = max(1, math.ceil(M_tiles / grid_y))

    tensor = ttnn.typecast(tensor, ttnn.bfloat8_b)

    mm_config = ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=k_block,
        N_block_size=n_block,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=mm_core_grid,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        ccl_manager.mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    mm_out, rs_out = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
        tensor,
        weights.o_proj,
        3,  # scatter on last (hidden) dim
        ccl_manager.get_rs_ping_pong_semaphore(),
        ttnn.CoreCoord(0, grid_y),  # RS cores start just below the MM rows
        compute_kernel_config=compute_config,
        num_links=ccl_manager.num_links,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ccl_manager.topology,
        cluster_axis=mesh_config.tp_axis,
        config=mm_config,
        barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        chunk_width_in_mm_blocks=chunk_width,
        num_workers_per_link=num_workers,
    )

    tensor.deallocate(True)
    mm_out.deallocate(True)
    return rs_out


def apply_allgather_and_slice(rs_out, mesh_config, ccl_manager, hidden_size: int):
    """Complete the attention-output chain after `apply_output_projection_fused_rs`.

    Runs the all-gather (TP) on the scattered output, then drops the padding
    columns that `weights.py` adds for tile alignment of the CCL ops.
    """
    gathered = mesh_config.allgather(rs_out, ccl_manager, axis=mesh_config.tp_axis)
    rs_out.deallocate(True)

    local_hidden = hidden_size // mesh_config.tp
    padded_local_hidden = ((local_hidden + 31) // 32) * 32
    if padded_local_hidden != local_hidden:
        shape = gathered.shape
        sliced = ttnn.slice(
            gathered,
            starts=[0, 0, 0, 0],
            ends=[shape[0], shape[1], shape[2], hidden_size],
            steps=[1, 1, 1, 1],
        )
        gathered.deallocate(True)
        return sliced
    return gathered


def apply_allreduce(tensor, mesh_config, ccl_manager, hidden_size: int):
    """
    Apply tensor parallel allreduce if needed.

    Args:
        tensor: Input tensor
        mesh_config: Mesh configuration
        ccl_manager: Communication manager
        batch_size: Batch size for final reshape
        seq_len: Sequence length for final reshape
        hidden_size: Hidden size for final reshape

    Returns:
        Tensor after allreduce (if TP > 1) or original tensor
    """
    if mesh_config.tp > 1:
        # ttnn.all_reduce (reduce-scatter + all-gather with per-call semaphores), the same collective the MoE block and
        # the decode path use. NOT MeshConfig.allreduce: its all_gather_async driven by the CCLManager's ping-pong
        # semaphores leaves a stale 1/8-of-the-rows block on the last devices of the ring on every other call on
        # P150x8 (measured 2026-09-07: 128 tokens -> devices 5-7, 1024 tokens -> device 7, |err| up to 30 on a sum of
        # 8 replicas; reduce_scatter_minimal_async alone is exact; ttnn.all_reduce is bit-identical on all 8 devices
        # in 12/12 runs). Device 0 was always correct, so a device-0-only PCC never saw it; the corrupted replica fed
        # that device's router/experts and surfaced as run-to-run spread of the decoder PCC. Peak memory: the composite
        # holds input + scattered + gathered (the input is freed afterwards), ~580 MiB at a 64K bfp8 prefill.
        tensor_allreduced = ttnn.all_reduce(
            tensor, num_links=ccl_manager.num_links, topology=ttnn.Topology.Ring, cluster_axis=mesh_config.tp_axis
        )
        tensor.deallocate(True)
        tensor = tensor_allreduced

        # Remove padding added in weights.py for tile-aligned CCL operations.
        # If local_hidden was padded (e.g., 360 -> 384), we need to slice back to original hidden_size.
        local_hidden = hidden_size // mesh_config.tp
        padded_local_hidden = ((local_hidden + 31) // 32) * 32
        if padded_local_hidden != local_hidden:
            # Slice from padded_hidden back to hidden_size on the last dimension.
            # Works for both decode [1, 1, batch, padded_hidden] and prefill [1, batch, seq_len, padded_hidden].
            shape = tensor.shape
            tensor_sliced = ttnn.slice(
                tensor,
                starts=[0, 0, 0, 0],
                ends=[shape[0], shape[1], shape[2], hidden_size],
                steps=[1, 1, 1, 1],
            )
            tensor.deallocate(True)
            tensor = tensor_sliced
    return tensor


def get_mesh_coords(mesh_shape: list[int], row: int = None, col: int = None) -> list:
    """
    Get mesh coordinates for a given mesh shape and optional row and column indices.

    This is used to specify which devices should execute paged cache operations
    when the KV cache is replicated but users are sharded across rows.

    Args:
        mesh_shape: Shape of the mesh as [num_rows, num_cols]
        row: Optional row index to filter (None = all rows)
        col: Optional column index to filter (None = all columns)

    Returns:
        List of ttnn.MeshCoordinate objects for the specified row/column
    """
    if row is not None:
        assert 0 <= row < mesh_shape[0], f"Row index {row} out of bounds for mesh shape {mesh_shape}"
    if col is not None:
        assert 0 <= col < mesh_shape[1], f"Column index {col} out of bounds for mesh shape {mesh_shape}"

    row_select = range(mesh_shape[0]) if row is None else [row]
    col_select = range(mesh_shape[1]) if col is None else [col]
    return [ttnn.MeshCoordinate(r, c) for r in row_select for c in col_select]
