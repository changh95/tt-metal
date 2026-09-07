# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode forward pass for experts: one token per user, 1 <= users <= 32 per step."""

import ttnn
from models.demos.solar_open.config import Mode

from .config import ExpertConfig, ProgramConfig
from .operations import apply_expert_parallel_allreduce, apply_glu, apply_tensor_parallel_allreduce
from .weights import ExpertWeights

# Largest number of decode tokens (users) the batched low-latency path handles in
# one call: all tokens of the step must fit in a single 32-row tile so that the
# expert matmuls stay M=1 tile (same program configs / footprint as batch=1).
MAX_BATCHED_DECODE_TOKENS = 32


def decode_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config: ExpertConfig,
    mesh_config,
    mesh_device,
    ccl_manager,
    program_config: ProgramConfig,
    shared_expert=None,
):
    """
    Decode forward pass: one new token per user.

    users == 1 (hidden_states [1, 1, 1, hidden]) runs the original single-token path;
    1 < users <= 32 (hidden_states [1, 1, users, hidden]) runs _decode_forward_batched.

    The two paths share the fused gate/up -> GLU -> down sequence but are kept separate on purpose: the
    batched path additionally builds the union-of-experts mask, multiplies every (user, expert) output by its
    routing weight and reduces over experts (fast_reduce_nc), all of which the single-user path can skip (its
    selected experts are exactly the active ones and its down output is already the answer). Unifying them
    would cost the batch-1 case those extra passes.

    Args:
        hidden_states: Input tensor [1, 1, users, hidden_size] (consumed)
        routing_weights: Dense router output [users, num_experts] (0 for unselected experts)
        weights: Expert weights
        config: Expert configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        ccl_manager: Communication manager
        program_config: Model-specific program configs
        shared_expert: Optional callable evaluated once on the (padded) input; its per-device partial
            [1, 1, rows, hidden] bf16 is added to the routed partial before the TP all_reduce (see Experts.__call__)

    Returns:
        Expert output [1, batch, 1, hidden_size]
    """
    activation_dtype = ttnn.bfloat8_b
    batch_dim = 1
    seq_dim = 2
    batch_size = hidden_states.shape[batch_dim]
    seq_len = hidden_states.shape[seq_dim]

    # ✅ Use exceptions instead of assertions
    if batch_size != 1:
        raise NotImplementedError(f"Currently only batch_size=1 supported, got {batch_size}")
    if seq_len != 1:
        # Multi-user decode on a single mesh row: hidden_states is [1, 1, users, hidden]
        # (one token per user). Route the whole 32-row tile through the union of the
        # experts selected by any user instead of dispatching tokens to experts.
        if seq_len > MAX_BATCHED_DECODE_TOKENS:
            raise ValueError(
                f"Decode mode supports at most {MAX_BATCHED_DECODE_TOKENS} tokens (users) per step, got {seq_len}"
            )
        return _decode_forward_batched(
            hidden_states,
            routing_weights,
            weights,
            config,
            mesh_config,
            mesh_device,
            ccl_manager,
            program_config,
            shared_expert=shared_expert,
        )

    # Get parallelization config
    mode_config = mesh_config.get_config(Mode.DECODE)
    ep, tp = mode_config.ep, mode_config.tp
    # Prepare inputs for sparse matmul
    sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)

    # EP-specific routing remap for sparsity
    if ep > 1:
        sparsity = ttnn.moe_routing_remap(
            ttnn.reshape(sparsity, (1, sparsity.shape[-1])),
            config.num_experts_per_tok,
            ep,
            mesh_config.ep_axis,
        )
        routing_weights = ttnn.tilize_with_zero_padding(sparsity, use_multicore=True)

    output_tile = ttnn.Tile([32, 32])

    # Shared expert on the still-allocated input [1, 1, 1, H] (the gate/up sparse_matmul below consumes it).
    shared = shared_expert(hidden_states) if shared_expert is not None else None

    # Fused gate/up projection: [1, 1, 1, H] x [1, E, H, 2 * Ip] -> [1, 1, 1, E, 1, 2 * Ip]
    # (Ip = intermediate_padded_per_device; per device the columns are [gate | up], each zero-padded to Ip).
    gate_up = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_up_proj,
        sparsity=sparsity,
        # nnz intentionally omitted (None -> inferred at runtime). Passing a static
        # nnz makes the sparse_matmul in0-mcast receivers loop a fixed count while the
        # sender only mcasts for the *actual* non-zero `sparsity` entries. The decode
        # routing weights (normalised sigmoid over the top-k, scattered) can have <k
        # non-zeros on Blackhole (small weights flush to 0), so a static nnz != actual
        # count and the receivers deadlock in noc_semaphore_wait. Inferring the count is
        # robust. See tenstorrent/tt-metal#45943 (op deadlock) / #45052 (decode hang).
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], weights.gate_up_proj.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=activation_dtype,
    )
    hidden_states.deallocate(True)
    ip = weights.intermediate_padded_per_device
    # Note: reshape/transpose operations return views - do not deallocate originals
    gate_up = ttnn.reshape(gate_up, (batch_size, config.num_experts, 1, 2 * ip))
    gate_up = ttnn.transpose(gate_up, 1, 2)
    gate_up = ttnn.reshape(gate_up, (batch_size, config.num_experts, 2 * ip))
    # Split the fused output at the tile-aligned half: gate = [:, :, :Ip], up = [:, :, Ip:]
    gate = ttnn.slice(gate_up, [0, 0, 0], [batch_size, config.num_experts, ip])
    up = ttnn.slice(gate_up, [0, 0, ip], [batch_size, config.num_experts, 2 * ip])
    gate_up.deallocate(True)

    # GLU: up * silu(gate) in one fused binary op. The zero-padded columns beyond the real intermediate
    # width stay exactly 0 (gate = 0 -> silu = 0).
    down_input = apply_glu(gate, up, config)
    gate.deallocate(True)
    up.deallocate(True)
    # Note: transpose/reshape operations return views - do not deallocate originals
    down_input = ttnn.transpose(down_input, 1, 0)
    down_input = ttnn.reshape(down_input, (1, config.num_experts, seq_len, ip))
    # Down projection
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        # nnz intentionally omitted (None -> inferred at runtime); see the gate/up call above and
        # tenstorrent/tt-metal#45943 / #45052.
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        program_config=program_config.get_decode_down_config(
            down_input.shape[2], weights.down_proj.shape[-1], k=down_input.shape[-1]
        ),
        dtype=activation_dtype,
    )

    down_input.deallocate(True)
    sparsity.deallocate(True)
    # Apply routing weights
    # Note: permute/reshape operations return views - do not deallocate originals
    next_states = ttnn.permute(down, (0, 2, 1, 3))
    next_states = ttnn.reshape(next_states, (batch_size, config.num_experts, config.hidden_size))
    routing_weights = ttnn.permute(routing_weights, (1, 0))
    routing_weights = ttnn.reshape(routing_weights, (batch_size, config.num_experts, 1))

    next_states = ttnn.mul(next_states, routing_weights, output_tensor=next_states)
    routing_weights.deallocate(True)

    # Reduce across experts
    next_states = ttnn.sum(next_states, dim=1)
    # Note: unsqueeze_to_4D typically returns a view, so we don't deallocate the sum result
    next_states = ttnn.unsqueeze_to_4D(next_states)

    # Expert parallel communication
    if ep > 1:
        next_states = apply_expert_parallel_allreduce(next_states, mesh_config, ccl_manager)

    # Note: unsqueeze_to_4D typically returns a view
    next_states = ttnn.unsqueeze_to_4D(next_states)

    # Shared-expert partial [1, 1, 1, H] bf16 into the routed partial [1, 1, 1, H] bfloat8_b, in place, before
    # the TP all_reduce: both are per-device partials over the same intermediate slice, so one CCL sums both.
    if shared is not None:
        next_states = ttnn.add(next_states, shared, output_tensor=next_states)
        shared.deallocate(True)

    # Tensor parallel communication
    if tp > 1:
        # Note: apply_tensor_parallel_allreduce already handles deallocating the input tensor
        next_states = apply_tensor_parallel_allreduce(
            next_states,
            mesh_config,
            mesh_device,
            seq_len,
            ccl_manager,
        )

    # Final reshape
    # Note: reshape typically returns a view, so we don't deallocate the original
    next_states = ttnn.reshape(
        next_states,
        (1, batch_size, seq_len, config.hidden_size),
        (1, batch_size, max(32, seq_len), config.hidden_size),
    )

    return next_states


def _decode_forward_batched(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config: ExpertConfig,
    mesh_config,
    mesh_device,
    ccl_manager,
    program_config: ProgramConfig,
    shared_expert=None,
):
    """
    Multi-user decode (1 < users <= 32) for the low-latency (TP, EP=1) experts.

    Every user contributes one token, so the step is a single 32-row tile
    [1, 1, users, hidden]. Rather than dispatching tokens to experts (which needs
    EP across a mesh axis and all_to_all CCLs), we run the whole tile through the
    *union* of the experts any user selected:

      1. union mask  = sum over users of the dense routing weights -> [1, 1, 1, E]
                       (routing weights are >= 0, so an expert is non-zero iff at
                       least one user picked it; computed on device, trace-safe)
      2. gate/up/down = sparse_matmul over that mask (nnz inferred at runtime)
      3. multiply by the dense per-(user, expert) routing weights so users that did
         not pick an expert get exactly 0 from it, then reduce over experts

    Cost model: each active expert's weight slice is streamed from DRAM once, the
    same as an exact per-token gather. The extra compute (32 rows per active expert
    instead of ~1) is a single tile per matmul and is negligible next to the weight
    streaming at these shapes. The activation footprint is identical to batch=1
    because the M=1 path already pads to a 32-row tile. With top-8 routing and 32
    users the union covers ~112 of 128 experts on average, i.e. the step streams
    almost every expert's weights.

    Args:
        hidden_states: [1, 1, users, hidden_size] (consumed)
        routing_weights: dense router output [users, num_experts] (0 for unselected experts)
        shared_expert: optional callable; evaluated once on the 32-row padded input (see decode_forward)

    Returns:
        Expert output [1, 1, users, hidden_size]
    """
    activation_dtype = ttnn.bfloat8_b
    _, _, num_tokens, hidden_size = hidden_states.shape

    mode_config = mesh_config.get_config(Mode.DECODE)
    ep, tp = mode_config.ep, mode_config.tp
    if ep > 1:
        raise NotImplementedError(f"Batched low-latency decode requires EP=1 (got EP={ep})")

    num_experts = config.num_experts
    output_tile = ttnn.Tile([32, 32])

    # 0. Work on a full 32-row tile. The tile is 32 rows tall regardless of the user count, so a
    #    partial batch costs nothing extra, and the broadcast muls below (routing weights over hidden)
    #    only support full-tile row counts. The padding rows carry zero hidden states and zero routing
    #    weights, so they contribute exactly nothing (also through the shared expert: a zero row stays
    #    zero) and are dropped by the final reshape back to num_tokens rows.
    #    NOTE: for a bf16 TILE input whose padded shape already is 32 rows, ttnn.pad zeroes the tile padding IN PLACE
    #    and returns a view of the caller's tensor (pad.cpp invoke_tile -> fill_implicit_tile_padding); only a bfp8
    #    input gets a real copy. So the "padded" tensors may alias the caller's: hidden_states is consumed anyway, and
    #    the routing tensor is left to the refcount (an explicit deallocate here would free the router's tensor).
    real_tokens = num_tokens
    if num_tokens < ttnn.TILE_SIZE:
        pad_rows = ttnn.TILE_SIZE - num_tokens
        hidden_states = ttnn.pad(hidden_states, padding=[(0, 0), (0, 0), (0, pad_rows), (0, 0)], value=0.0)
        routing_weights = ttnn.pad(routing_weights, padding=[(0, pad_rows), (0, 0)], value=0.0)
        num_tokens = ttnn.TILE_SIZE

    # Shared expert on the padded input [1, 1, 32, H] (consumed by the gate/up sparse_matmul below).
    shared = shared_expert(hidden_states) if shared_expert is not None else None

    # 1. Union-of-experts sparsity mask [1, 1, 1, E], ROW_MAJOR bf16 (+0.0 == inactive).
    expert_hit = ttnn.sum(routing_weights, dim=0, keepdim=True)  # [1, E]
    expert_hit_4d = ttnn.reshape(expert_hit, (1, 1, 1, num_experts))
    sparsity = ttnn.to_layout(expert_hit_4d, ttnn.ROW_MAJOR_LAYOUT)
    expert_hit.deallocate(True)

    # 2a. Fused gate/up projection: [1, 1, T, H] x [1, E, H, 2 * Ip] -> [1, 1, 1, E, T, 2 * Ip] -> [1, E, T, 2 * Ip]
    ip = weights.intermediate_padded_per_device
    gate_up = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_up_proj,
        sparsity=sparsity,
        nnz=None,  # data-dependent union size: must be inferred on device (see decode_forward)
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            num_tokens, weights.gate_up_proj.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=activation_dtype,
    )
    hidden_states.deallocate(True)
    gate_up = ttnn.reshape(gate_up, (1, num_experts, num_tokens, 2 * ip))
    # Split at the tile-aligned half: gate = [..., :Ip], up = [..., Ip:]
    gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, num_experts, num_tokens, ip])
    up = ttnn.slice(gate_up, [0, 0, 0, ip], [1, num_experts, num_tokens, 2 * ip])
    gate_up.deallocate(True)

    # GLU (one fused binary op): [1, E, T, Ip]; padded columns stay exactly 0.
    down_input = apply_glu(gate, up, config)
    gate.deallocate(True)
    up.deallocate(True)

    # 2b. Down projection (input is expert-batched too): [1, E, T, Ip] x [1, E, I, H] -> [1, E, T, H]
    #     (padded K: Ip == padded I of the weight; the extra input columns are zero)
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        program_config=program_config.get_decode_down_config(
            num_tokens, weights.down_proj.shape[-1], k=down_input.shape[-1]
        ),
        dtype=activation_dtype,
    )
    down_input.deallocate(True)
    sparsity.deallocate(True)

    # 3. Per-(user, expert) routing weights [T, E] -> [1, E, T, 1]; zero for unselected pairs.
    token_expert_weights = ttnn.permute(routing_weights, (1, 0))
    token_expert_weights = ttnn.reshape(token_expert_weights, (1, num_experts, num_tokens, 1))
    down = ttnn.mul(down, token_expert_weights, output_tensor=down)
    token_expert_weights.deallocate(True)

    # Reduce over experts: [1, E, T, H] -> [1, 1, T, H]. Keep the result in L1 like the batch=1 path
    # (the residual add and the next norm read it; the default would land in DRAM).
    next_states = ttnn.experimental.fast_reduce_nc(down, dims=[1], memory_config=ttnn.L1_MEMORY_CONFIG)
    next_states = ttnn.unsqueeze_to_4D(next_states)
    down.deallocate(True)

    # 4. Shared-expert partial [1, 1, 32, H] bf16 into the routed partial (bfloat8_b) in place. Both are this
    #    device's partial sums over the same intermediate slice, so the single TP all_reduce below completes
    #    routed + shared at once.
    if shared is not None:
        next_states = ttnn.add(next_states, shared, output_tensor=next_states)
        shared.deallocate(True)

    # Tensor parallel all-reduce (sums the per-device intermediate slices)
    if tp > 1:
        next_states = apply_tensor_parallel_allreduce(
            next_states,
            mesh_config,
            mesh_device,
            num_tokens,
            ccl_manager,
        )

    next_states = ttnn.reshape(
        next_states,
        (1, 1, real_tokens, config.hidden_size),
        (1, 1, ttnn.TILE_SIZE, config.hidden_size),
    )
    return next_states
