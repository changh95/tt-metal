# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MoE Experts Module (routed experts of Solar-Open).

This module provides a model-agnostic implementation of bias-free GLU MoE experts.
Models provide their own ProgramConfig implementations for customization.

Usage:
    from models.demos.solar_open.tt.experts import Experts, ExpertConfig
    from models.demos.solar_open.tt.expert_configs import solar_open_program_config

    config = ExpertConfig(intermediate_size=1280, num_experts=128, hidden_size=4096, num_experts_per_tok=8)
    program_config = solar_open_program_config(mesh_device)

    experts = Experts(
        mesh_device=mesh_device,
        config=config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=program_config,
    )

    output = experts(hidden_states, routing_weights, is_decode=True, shared_expert=shared_expert)
"""

import ttnn

from models.demos.solar_open.config import MeshConfig, ModeConfig

from .config import ExpertConfig, IndexedRouting, ProgramConfig
from .decode import decode_forward
from .prefill import prefill_forward
from .weights import load_expert_weights

__all__ = ["Experts", "ExpertConfig", "IndexedRouting", "ProgramConfig"]


class Experts:
    """
    MoE expert layer with automatic decode/prefill dispatch.

    Consumes the DENSE ``[tokens, num_experts]`` routing tensor emitted by the router (normalised routing
    weight for the selected experts, 0 elsewhere) and returns the all-reduced MoE output. An optional
    ``shared_expert`` callable is evaluated on the experts' input and its per-device partial is summed into the
    routed partial BEFORE the single TP all_reduce, so routed + shared cost one CCL per layer.

    Single-user decode may instead receive the token's top-k ids and weights as an ``IndexedRouting``
    (``indexed_routing=``, no dense tensor): the experts then run the sparse_matmul indexed/gather mode, visiting
    only the k selected experts with compact ``[1, k, 1, *]`` intermediates (phase 2, ``MoEOptions.indexed_decode``).
    """

    def __init__(
        self,
        mesh_device,
        config: ExpertConfig,
        state_dict,
        ccl_manager,
        mesh_config: MeshConfig,
        program_config: ProgramConfig,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        """
        Initialize expert layers.

        Args:
            mesh_device: TTNN mesh device
            config: Expert configuration
            state_dict: Expert weights dictionary (``gate_up_proj [E, 2I, H]``, ``down_proj [E, H, I]``; ``{}`` loads
                from the tensor cache)
            ccl_manager: Communication manager
            mesh_config: Mesh parallelization configuration
            program_config: Model-specific program configurations
            weight_dtype: Data type for weights (default: bfloat8_b; bfloat4_b is the low-memory option)
            tensor_cache_path: Optional path for weight caching
        """
        self.config = config
        self.mesh_config = mesh_config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.program_config = program_config

        # Load weights
        self.weights = load_expert_weights(
            mesh_device=mesh_device,
            config=config,
            state_dict=state_dict,
            mesh_config=mesh_config,
            weight_dtype=weight_dtype,
            tensor_cache_path=tensor_cache_path,
        )

        # Cache prefill sparsity (created once, reused for all prefill calls)
        self.prefill_sparsity = self._create_prefill_sparsity()

        # For backward compatibility
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size

    def _create_prefill_sparsity(self):
        """Create prefill sparsity mask once and reuse."""
        import torch
        from models.demos.solar_open.config import Mode

        prefill_config = self.mesh_config.get_config(Mode.PREFILL)
        prefill_ep = prefill_config.ep
        tokens_per_ep = self.config.num_experts // prefill_ep

        sparsity = torch.zeros(1, 1, prefill_ep, self.config.num_experts)
        for i in range(prefill_ep):
            sparsity[:, :, i, i * tokens_per_ep : (i + 1) * tokens_per_ep] = torch.ones(1, 1, 1, tokens_per_ep)

        return ttnn.from_torch(
            sparsity,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                dims=(-2, None) if prefill_ep > 1 else (None, None),
                mesh_shape=self.mesh_device.shape,
                mesh_device=self.mesh_device,
            ),
        )

    def __call__(
        self,
        hidden_states,
        topk_expert_weights: ttnn.Tensor = None,
        is_decode: bool = True,
        topk_expert_indices: ttnn.Tensor = None,
        shared_expert=None,
        indexed_routing: IndexedRouting = None,
    ):
        """
        Forward pass - automatically dispatches to decode or prefill.

        Args:
            hidden_states: Input tensor [1, 1, tokens, hidden_size] (decode: one token per user,
                1 <= users <= 32; prefill: tokens = seq_len, a multiple of 32). Consumed.
            topk_expert_weights: Dense router scores [tokens, num_experts] bf16 TILE (0 for unselected experts,
                > 0 for the selected ones; rows normalised by the router). ``num_experts`` INCLUDES the always-on
                slots of a fused shared expert (``self.weights.num_always_on_experts``, routing weight 1.0). May be
                None only together with ``indexed_routing``.
            is_decode: Decode mode
            topk_expert_indices: Top-k expert indices per token (informational; the dense weights carry the routing)
            indexed_routing: decode with exactly one token only: the token's top-k ids / weights as an
                ``IndexedRouting`` (see decode.py ``_decode_forward_indexed``). Not consumed. Mutually exclusive with
                ``topk_expert_weights``; requires EP=1 and no always-on slots (ValueError otherwise -- the caller
                decides the path, see ``tt/mlp.py::MLP.route``).
            shared_expert: Optional ``Callable[[ttnn.Tensor], ttnn.Tensor]``. Called exactly once per decode call on
                the (32-row padded) input ``[1, 1, 32, H]`` (``[1, 1, 1, H]`` for a single user) and once per
                <= ``sequence_chunk_size``-token chunk in prefill on ``[1, 1, chunk, H]``, before that input is
                deallocated. It must return this device's PARTIAL ``[1, 1, rows, H]`` bf16 TILE interleaved tensor
                with the same logical/padded shape; the experts add it in place to their routed partial and
                deallocate it. The shared partial is never all-reduced separately. For the unfused layout only:
                a ValueError is raised when the weights carry always-on slots.

        Returns:
            Expert output tensor [1, 1, tokens, hidden_size] bfloat8_b, replicated (all-reduced over TP)
        """
        if shared_expert is not None and self.weights.num_always_on_experts:
            raise ValueError(
                "the shared expert is fused as an always-on slot of these experts; do not pass shared_expert"
            )
        if indexed_routing is not None:
            if not is_decode:
                raise ValueError("indexed_routing is a decode-only input (prefill routes through the dense tensor)")
            if topk_expert_weights is not None:
                raise ValueError("pass either the dense topk_expert_weights or indexed_routing, not both")
        elif topk_expert_weights is None:
            raise ValueError("topk_expert_weights is required without indexed_routing")
        # Determine mode based on sequence length
        if is_decode:
            return decode_forward(
                hidden_states=hidden_states,
                routing_weights=topk_expert_weights,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                program_config=self.program_config,
                shared_expert=shared_expert,
                indexed_routing=indexed_routing,
                # the sparse_matmul indexed mode still takes a `sparsity` operand it never reads: the cached
                # [1, 1, 1, E] ones of the EP=1 prefill mask serve (ROW_MAJOR bf16, the validated shape)
                sparsity_placeholder=self.prefill_sparsity,
            )
        else:
            return prefill_forward(
                hidden_states=hidden_states,
                routing_weights=topk_expert_weights,
                weights=self.weights,
                config=self.config,
                mesh_config=self.mesh_config,
                mesh_device=self.mesh_device,
                ccl_manager=self.ccl_manager,
                program_config=self.program_config,
                prefill_sparsity=self.prefill_sparsity,
                shared_expert=shared_expert,
            )
