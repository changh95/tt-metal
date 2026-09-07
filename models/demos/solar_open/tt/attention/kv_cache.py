# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.utility_functions import nearest_y
from models.demos.solar_open.config import MeshConfig

from .config import AttentionConfig


def init_kv_cache(
    mesh_device,
    config: AttentionConfig,
    mesh_config: MeshConfig,
    paged_attention_config=None,
    cache_dtype=ttnn.bfloat8_b,
    tensor_cache_path=None,
):
    """
    Initialize KV cache for both paged and non-paged attention.

    The caches start as zeros and are allocated straight on the device with ``ttnn.from_torch`` (DRAM, TILE,
    replicated over the mesh). They deliberately do NOT go through ``ttnn.as_tensor(cache_file_name=...)``: nothing
    ever reads a zero cache back from disk, and the tensorbins would add ~27 GB of zeros per KV configuration
    (paged block count or batch x max_seq_len) to TT_CACHE_PATH and be re-read on every warm start (tech-lead
    decision of 2026-09-07 superseding the ``k_cache_<shape>`` / ``v_cache_<shape>`` cache stems of contract C7).
    Shapes, dtype and memory config are unchanged.

    Args:
        mesh_device: TTNN mesh device
        config: Attention configuration
        mesh_config: Mesh parallelization config
        paged_attention_config: Optional paged attention configuration
        cache_dtype: Data type for cache tensors (default: bfloat8_b)
        tensor_cache_path: Accepted for interface compatibility (Attention passes its weight-cache directory); unused.

    Returns:
        List [k_cache, v_cache]
    """
    # Determine cache shape based on paged vs non-paged attention
    kv_cache_repeats = mesh_device.shape[0] if config.users_row_sharded else 1
    if paged_attention_config:
        # Paged attention cache shape: [max_num_blocks, num_kv_heads, block_size, head_dim]
        cache_shape = [
            paged_attention_config.max_num_blocks * kv_cache_repeats,
            config.num_kv_heads // mesh_device.shape[1],
            paged_attention_config.block_size,
            config.head_dim,
        ]
    else:
        # Standard cache shape: [batch_size, num_kv_heads, max_seq_len, head_dim]
        cache_shape = [
            config.max_local_batch_size * kv_cache_repeats,
            config.num_kv_heads // mesh_device.shape[1],
            config.max_seq_len,
            config.head_dim,
        ]

    # Create K cache
    mesh_mapper = (
        ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, None))
        if config.users_row_sharded
        else ttnn.ReplicateTensorToMesh(mesh_device)
    )
    zeros = torch.zeros(cache_shape)
    k_cache = ttnn.from_torch(
        zeros,
        dtype=cache_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # Create V cache
    v_cache = ttnn.from_torch(
        zeros,
        dtype=cache_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    return [k_cache, v_cache]


def get_kv_memory_config(mesh_device, max_local_batch_size: int, num_local_kv_heads: int, head_dim: int):
    """
    Get sharded memory config for KV tensors in decode mode.

    Args:
        mesh_device: TTNN mesh device
        max_local_batch_size: Maximum local batch size per device
        num_local_kv_heads: Number of KV heads per device
        head_dim: Head dimension

    Returns:
        Sharded memory config for KV tensors
    """
    from .config import ProgramConfig

    # KV tensors are [1, local_batch_size, num_local_kv_heads, head_dim] for decode: one user per core,
    # on the same per-user grid nlp_create_qkv_heads_decode / RoPE use (ProgramConfig.get_decode_user_grid),
    # so the to_memory_config before paged_update_cache is a no-op on every arch and batch size.
    kv_shape = (1, max_local_batch_size, num_local_kv_heads, head_dim)
    kv_shard_height = nearest_y(num_local_kv_heads, ttnn.TILE_SIZE)  # kv heads padded to a tile
    kv_shard_width = kv_shape[3]  # width = head_dim

    kv_core_grid, _ = ProgramConfig.get_decode_user_grid(mesh_device, max_local_batch_size)

    return ttnn.create_sharded_memory_config(
        shape=(kv_shard_height, kv_shard_width),
        core_grid=kv_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
