# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.demos.solar_open.config import MeshConfig

from .config import AttentionConfig, ProgramConfig
from .decode import decode_forward
from .kv_cache import get_kv_memory_config, init_kv_cache
from .prefill import prefill_forward
from .weights import load_attention_weights

__all__ = ["Attention", "AttentionConfig", "ProgramConfig"]


class Attention:
    """
    Generic Attention implementation with automatic decode/prefill dispatch.

    This class provides a clean interface for attention layers. Models provide
    their own ProgramConfig implementations to customize behavior.

    Solar-Open: bias-free, sink-free grouped-query attention with uniform full (causal) attention on every layer;
    ``config.sliding_window`` is passed through unchanged (``None`` for Solar-Open).
    """

    def __init__(
        self,
        mesh_device,
        config: AttentionConfig,
        state_dict,
        ccl_manager,
        mesh_config: MeshConfig,
        program_config: ProgramConfig,
        layer_idx,
        paged_attention_config=None,
        transformation_mats=None,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        create_kv_cache=True,
    ):
        """
        Initialize attention layer.

        Args:
            mesh_device: TTNN mesh device
            config: Attention configuration
            state_dict: State dictionary containing weights
            ccl_manager: Communication manager
            mesh_config: Mesh parallelization config
            program_config: Model-specific program configurations
            layer_idx: Layer index (cache naming / debugging)
            paged_attention_config: Optional paged attention configuration
            transformation_mats: Optional transformation matrices for RoPE
            weight_dtype: Data type for weights (default: bfloat8_b)
            tensor_cache_path: Optional path for weight caching
            create_kv_cache: Whether to create KV cache (default: True)
        """
        self.config = config
        self.mesh_config = mesh_config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.program_config = program_config
        self.layer_idx = layer_idx
        self.transformation_mats = transformation_mats
        self.paged_attention_config = paged_attention_config

        # Load weights
        self.weights = load_attention_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

        # Initialize KV cache
        if create_kv_cache:
            self.kv_cache = init_kv_cache(
                mesh_device=mesh_device,
                config=config,
                mesh_config=mesh_config,
                paged_attention_config=paged_attention_config,
                tensor_cache_path=tensor_cache_path,
            )
            self.layer_past = self.kv_cache  # For tt-transformers compatibility
        else:
            self.kv_cache = None
            self.layer_past = None

        # Get KV memory config for decode mode
        self.kv_mem_cfg = get_kv_memory_config(
            mesh_device,
            config.max_local_batch_size,
            mesh_config.shard_size(config.num_kv_heads),
            config.head_dim,
        )

        # Store references for backward compatibility
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = config.scaling

    def __call__(
        self,
        hidden_states,
        rope_mats,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        user_id=0,
        batch_size=1,
        chunk_page_table=None,
        chunk_start_idx=None,
    ):
        """
        Forward pass - dispatches to decode or prefill.

        Args:
            hidden_states: Input tensor [1, 1, batch (decode) | batch * seq_len (prefill), hidden_size]
            rope_mats: Tuple of (cos, sin) matrices for RoPE
            position_idx: Position index for KV cache update
            page_table: Page table for paged attention (optional)
            kv_cache: External KV cache (optional, uses internal if not provided)
            is_decode: Whether this is decode mode (default: True)
            user_id: User/batch index for KV cache fill in prefill mode (default: 0)
            batch_size: Number of users packed into the prefill input (default: 1)
            chunk_page_table: chunked prefill only -- page-table slice of the chunk's blocks (see prefill_forward)
            chunk_start_idx: chunked prefill only -- absolute start of the chunk (python int; None / 0 = no prefix)

        Returns:
            Attention output with the same shape as hidden_states, all-reduced over the TP axis
        """
        # Use provided kv_cache or internal cache
        cache = kv_cache if kv_cache is not None else self.kv_cache

        # Get transformation matrix for the mode
        mode = "decode" if is_decode else "prefill"
        transformation_mat = self.transformation_mats[mode] if self.transformation_mats else None

        if is_decode:
            return decode_forward(
                hidden_states=hidden_states,
                rope_mats=rope_mats,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                program_config=self.program_config,
                transformation_mat=transformation_mat,
                kv_mem_cfg=self.kv_mem_cfg,
                position_idx=position_idx,
                page_table=page_table,
                ccl_manager=self.ccl_manager,
            )
        else:
            return prefill_forward(
                hidden_states=hidden_states,
                rope_mats=rope_mats,
                user_id=user_id,
                weights=self.weights,
                kv_cache=cache,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                program_config=self.program_config,
                transformation_mat=transformation_mat,
                position_idx=position_idx,
                page_table=page_table,
                ccl_manager=self.ccl_manager,
                batch_size=batch_size,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
            )
