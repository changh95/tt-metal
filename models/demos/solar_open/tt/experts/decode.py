# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode forward pass for experts: one token per user, 1 <= users <= 32 per step."""

import ttnn
from models.demos.solar_open.config import Mode

from .config import ExpertConfig, IndexedRouting, ProgramConfig
from .operations import apply_expert_parallel_allreduce, apply_glu, apply_tensor_parallel_allreduce
from .weights import ExpertWeights

# Largest number of decode tokens (users) the batched low-latency path handles in
# one call: all tokens of the step must fit in a single 32-row tile so that the
# expert matmuls stay M=1 tile (same program configs / footprint as batch=1).
MAX_BATCHED_DECODE_TOKENS = 32

# Batched path: apply the per-(user, expert) routing weights to the down INPUT [1, E, 32, Ip] (5 tiles wide) instead
# of the down OUTPUT [1, E, 32, H] (128 tiles wide) -- the down projection is linear, so the two are equal up to the
# bfloat8_b rounding of the product (the dense prefill tail does the same, see prefill.py::_dense_tail). Measured on
# P150 (2026-09-07, per layer, batch 32): the mul costs 12.7 us on the input vs 58.9 us on the output, i.e.
# ~-2.2 ms per 48-layer step. Zero rows (users that did not pick the expert, padding users, inactive experts) stay
# exactly 0 either way. False restores the phase-1 order (mul after the down) for A/B measurements.
ROUTING_WEIGHTS_ON_DOWN_INPUT = True

# Indexed single-user path (_decode_forward_indexed): how the token's [1, 1, 1, k] bf16 TILE routing weights become
# the [1, k, 1, 1] per-expert scalars that the down-input mul broadcasts over each compact GLU row.
#   "transpose": WH transpose to [1, 1, k, 1] + tiled reshape to [1, k, 1, 1] (2 launches);
#   "row_major": untilize + row-major reshape + tilize with zero padding (3 launches).
# Both give bit-identical expert outputs (tests/perf/test_indexed_candidates.py, P150 2026-09-07): conversion alone
# 53 vs 76 us eager (host-bound), whole traced expert block 0.127 vs 0.130 ms per layer -> "transpose".
INDEXED_WEIGHTS_LAYOUT = "transpose"

# Indexed single-user path: WHERE the token's routing weights are applied. True multiplies the [1, k, 1, 1] scalars into
# the compact bfp8 GLU rows (the down INPUT, 5 tiles per expert -- the order of the batched path); False multiplies
# them into the compact down OUTPUT [1, k, 1, H] (128 tiles per expert) -- the order of the phase-1 single-user scan
# path, whose per-expert down outputs the indexed path then reproduces bit for bit (same kernels, program configs and
# unweighted bfp8 inputs). Neither order is bit-equivalent to the scan path: its 128-slot ttnn.sum and the compact
# 8-slot fast_reduce_nc differ by half a bfp8 ulp on ~18 % of the output elements, so the model-level metrics move
# inside the bfp8 rounding-noise band either way. Measured on P150 (2026-09-07, k = 8, tests/perf/test_indexed_candidates.py
# and the b1 ladder in scratchpad/phase2/perf_log.md perf-p2): PCC vs torch fp32 (mean of 4 tokens) scan 0.998220,
# input mul 0.998177, output mul 0.998213; PCC vs the scan result 0.99980 (input) / 0.99996 (output); traced expert
# block 0.133 vs 0.139 ms; real layer 0 traced replay 0.760 vs 0.763 ms; b1 demo 35.8 vs 36.7 ms/step; teacher-forced
# b1 top-1 / decisive / KL 0.9336 / 0.9558 / 0.0355 (input) vs 0.9258 / 0.9646 / 0.0346 (output) vs the scan path's
# 0.9453 / 0.9779 / 0.0316 -- every floor holds for both. True ships (faster, same order as the batched path); False
# is the retained A/B variant.
INDEXED_WEIGHTS_ON_DOWN_INPUT = True


def _indexed_expert_scalars(weights, k, mode=None):
    """``[1, 1, 1, k]`` bf16 TILE routing weights -> ``[1, k, 1, 1]`` bf16 TILE (one scalar tile per selected
    expert; the padding of the k tiles is unspecified in "transpose" mode, zero in "row_major" mode -- either way
    only tile row 0 of the compact rows they scale is read back)."""
    mode = mode or INDEXED_WEIGHTS_LAYOUT
    if mode == "transpose":
        column = ttnn.transpose(weights, -2, -1)  # [1, 1, k, 1]
        return ttnn.reshape(column, (1, k, 1, 1))
    if mode == "row_major":
        rows = ttnn.to_layout(weights, ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, 1, k] one stick
        rows = ttnn.reshape(rows, (1, k, 1, 1))  # k sticks of one element
        return ttnn.to_layout(rows, ttnn.TILE_LAYOUT)  # k zero-padded tiles
    raise ValueError(f"unknown INDEXED_WEIGHTS_LAYOUT {mode!r}")


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
    indexed_routing: IndexedRouting = None,
    sparsity_placeholder=None,
):
    """
    Decode forward pass: one new token per user.

    users == 1 (hidden_states [1, 1, 1, hidden]) runs the original single-token path, or -- when the caller passes
    the token's top-k as ``indexed_routing`` (``MoEOptions.indexed_decode``) -- ``_decode_forward_indexed``;
    1 < users <= 32 (hidden_states [1, 1, users, hidden]) runs _decode_forward_batched. When the weights carry
    always-on slots (``weights.num_always_on_experts``, the fused shared expert: ``config.num_experts`` INCLUDES
    them and the routing tensor has 1.0 in those columns) every user count runs the batched path.

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
        indexed_routing: single user only: the token's top-k ids / weights (``IndexedRouting``) instead of
            ``routing_weights`` (which is then ignored / None); selects ``_decode_forward_indexed``
        sparsity_placeholder: ``[1, 1, 1, num_experts]`` ROW_MAJOR bf16 device tensor handed to the indexed
            sparse_matmuls as their (required, unread) ``sparsity`` operand; required with ``indexed_routing``

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
    if indexed_routing is not None:
        if seq_len != 1:
            raise ValueError(f"indexed_routing routes exactly one token per step, got {seq_len} users")
        if weights.num_always_on_experts:
            raise ValueError("the indexed single-user path does not support always-on expert slots")
        if sparsity_placeholder is None:
            raise ValueError("indexed_routing needs the sparsity_placeholder tensor")
        return _decode_forward_indexed(
            hidden_states,
            indexed_routing,
            weights,
            config,
            mesh_config,
            mesh_device,
            ccl_manager,
            program_config,
            sparsity_placeholder,
            shared_expert=shared_expert,
        )
    if routing_weights is None:
        raise ValueError("routing_weights is required without indexed_routing")
    # Always-on (fused shared) expert slot: every step takes the batched union-of-experts path. The single-user
    # path's HC transposes / permutes on bfp8 are bf16 typecast round trips (transpose.cpp: bfloat8_supported =
    # wh || cn) with no validated contract for a padded expert dim (129 -> 160), and the batched path has no bf16
    # intermediates.
    if seq_len != 1 or weights.num_always_on_experts:
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

    if weights.num_always_on_experts:
        raise NotImplementedError("the single-user decode path does not support always-on expert slots (unreachable)")

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
    # Program config and expert groups travel together (SparseMatmulConfig): an expert-group grid is only legal
    # with its `expert_groups` value and vice versa.
    gate_up_cfg = program_config.get_decode_gate_up_config(
        hidden_states.shape[2], weights.gate_up_proj.shape[3], k=hidden_states.shape[-1]
    )
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
        program_config=gate_up_cfg.program_config,
        expert_groups=gate_up_cfg.expert_groups,
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
    down_cfg = program_config.get_decode_down_config(
        down_input.shape[2], weights.down_proj.shape[-1], k=down_input.shape[-1]
    )
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
        program_config=down_cfg.program_config,
        expert_groups=down_cfg.expert_groups,
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


def _decode_forward_indexed(
    hidden_states,
    indexed_routing: IndexedRouting,
    weights: ExpertWeights,
    config: ExpertConfig,
    mesh_config,
    mesh_device,
    ccl_manager,
    program_config: ProgramConfig,
    sparsity_placeholder,
    shared_expert=None,
):
    """
    Single-user decode in the sparse_matmul INDEXED/GATHER mode (phase 2, profile lever 1).

    The router's top-k ids drive both sparse_matmuls directly (``indices=``: the kernels visit exactly the k
    selected experts, ``bB = indices[i]``) and their outputs are COMPACT ``[1, k, 1, *]`` (slot i = expert
    ``indices[i]``), so relative to the scan path (``decode_forward`` below) there is no 128-slot sparsity scan, no
    zero-filled full-E output (the 16 MB down output and its FILL), no bfp8 typecast / transpose glue between the
    projections and no dense ``[1, E]`` routing tensor (the router skips its scatter):

      1. gate|up: ``[1, 1, 1, H] x [1, E, H, 2 Ip]`` (ids) -> ``[1, k, 1, 2 Ip]``; slice at the tile-aligned half,
         GLU -> ``[1, k, 1, Ip]`` (the compact A of the down projection, no re-layout)
      2. routing weights: ``[1, 1, 1, k]`` -> ``[1, k, 1, 1]`` scalars (``_indexed_expert_scalars``), multiplied
         into the k compact GLU rows (down INPUT, 5 tiles wide -- the down is linear, as in the batched path) when
         ``INDEXED_WEIGHTS_ON_DOWN_INPUT``, else into the k compact down OUTPUT rows (the scan path's order)
      3. down: ``[1, k, 1, Ip] x [1, E, Ip, H]`` (ids, ``is_input_a_sparse``: A slot i pairs with expert
         ``indices[i]``) -> ``[1, k, 1, H]``; ``fast_reduce_nc`` over the k slots -> ``[1, 1, 1, H]``
      4. shared partial added in place, one TP all_reduce (unchanged contract)

    Static k (``config.num_experts_per_tok``) makes every shape static: trace-safe. ``nnz`` must NOT be passed
    (the indexed loop count is k; the op rejects it) and ``sparsity`` is a required operand the kernels never read
    (``sparsity_placeholder``). Numerics: same kernels and program configs as the scan path per expert; with
    ``INDEXED_WEIGHTS_ON_DOWN_INPUT`` the routing weights multiply the bfp8 GLU rows instead of the bfp8 down
    outputs, so the result is equal to the scan path up to bfp8 rounding order; with it off the per-expert down
    outputs and their weighting are the scan path's, only the k-slot reduction order differs (both validated vs
    torch fp32 in tests/perf/test_indexed_candidates.py and at the model level by the teacher-forced test).
    Requires EP=1 (the ids are global expert ids).
    """
    activation_dtype = ttnn.bfloat8_b
    mode_config = mesh_config.get_config(Mode.DECODE)
    ep, tp = mode_config.ep, mode_config.tp
    if ep > 1:
        raise NotImplementedError(f"the indexed single-user decode path requires EP=1 (got EP={ep})")
    k = indexed_routing.top_k
    if k != config.num_experts_per_tok:
        raise ValueError(f"indexed_routing.top_k {k} != num_experts_per_tok {config.num_experts_per_tok}")
    ip = weights.intermediate_padded_per_device
    hidden_size = config.hidden_size
    output_tile = ttnn.Tile([32, 32])

    # Shared expert on the still-allocated input [1, 1, 1, H] (the gate/up sparse_matmul below consumes it).
    shared = shared_expert(hidden_states) if shared_expert is not None else None

    # 1. Fused gate/up over the k selected experts only: [1, 1, 1, H] x [1, E, H, 2 Ip] -> [1, 1, 1, k, 1, 2 Ip]
    #    (expert groups: group i % G computes entry i of the ids; the compact output layout is unchanged)
    gate_up_cfg = program_config.get_decode_gate_up_config(1, weights.gate_up_proj.shape[3], k=hidden_size)
    gate_up = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_up_proj,
        sparsity=sparsity_placeholder,  # required operand, NOT read in indexed mode (see the op's docstring)
        indices=indexed_routing.indices,  # [1, 1, 1, k] uint16 ROW_MAJOR; nnz must stay unset
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_cfg.program_config,
        expert_groups=gate_up_cfg.expert_groups,
        dtype=activation_dtype,
    )
    hidden_states.deallocate(True)
    gate_up = ttnn.reshape(gate_up, (1, k, 1, 2 * ip))  # view
    gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, k, 1, ip])
    up = ttnn.slice(gate_up, [0, 0, 0, ip], [1, k, 1, 2 * ip])
    gate_up.deallocate(True)
    down_input = apply_glu(gate, up, config)  # [1, k, 1, Ip] bfp8; padded columns stay exactly 0
    gate.deallocate(True)
    up.deallocate(True)

    # 2. Routing weights as per-expert scalars: slot i scaled by weights[i], on the compact down input
    #    (INDEXED_WEIGHTS_ON_DOWN_INPUT) or on the compact down output below (the scan path's order).
    expert_scalars = _indexed_expert_scalars(indexed_routing.weights, k)  # [1, k, 1, 1] bf16
    if INDEXED_WEIGHTS_ON_DOWN_INPUT:
        down_input = ttnn.mul(down_input, expert_scalars, output_tensor=down_input)
        expert_scalars.deallocate(True)

    # 3. Down over the same k experts, compact A: [1, k, 1, Ip] x [1, E, Ip, H] -> [1, k, 1, H]
    down_cfg = program_config.get_decode_down_config(1, weights.down_proj.shape[-1], k=ip)
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity_placeholder,
        indices=indexed_routing.indices,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        is_input_b_sparse=True,
        program_config=down_cfg.program_config,
        expert_groups=down_cfg.expert_groups,
        dtype=activation_dtype,
    )
    down_input.deallocate(True)
    if not INDEXED_WEIGHTS_ON_DOWN_INPUT:
        down = ttnn.mul(down, expert_scalars, output_tensor=down)  # [1, k, 1, H] bfp8, slot i x weights[i]
        expert_scalars.deallocate(True)
    next_states = ttnn.experimental.fast_reduce_nc(down, dims=[1], memory_config=ttnn.L1_MEMORY_CONFIG)
    next_states = ttnn.unsqueeze_to_4D(next_states)  # [1, 1, 1, H]
    down.deallocate(True)

    # 4. Shared-expert partial (bf16) into the routed partial (bfp8) in place, then the single TP all_reduce.
    if shared is not None:
        next_states = ttnn.add(next_states, shared, output_tensor=next_states)
        shared.deallocate(True)
    if tp > 1:
        next_states = apply_tensor_parallel_allreduce(next_states, mesh_config, mesh_device, 1, ccl_manager)

    return ttnn.reshape(next_states, (1, 1, 1, hidden_size), (1, 1, ttnn.TILE_SIZE, hidden_size))


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
      3. multiply the GLU output (the down input) by the dense per-(user, expert) routing
         weights so users that did not pick an expert get exactly 0 from it, run the down
         projection and reduce over experts (ROUTING_WEIGHTS_ON_DOWN_INPUT)

    Cost model: each active expert's weight slice is streamed from DRAM once, the
    same as an exact per-token gather. The extra compute (32 rows per active expert
    instead of ~1) is a single tile per matmul and is negligible next to the weight
    streaming at these shapes. The activation footprint is identical to batch=1
    because the M=1 path already pads to a 32-row tile. With top-8 routing and 32
    users the union covers ~112 of 128 experts on average, i.e. the step streams
    almost every expert's weights.

    With always-on slots (fused shared expert) the union mask always contains them (``expert_hit[128] ==
    real_tokens * 1.0``), so nnz >= 1 in every step and the slot's rows are weighted by exactly 1.0.

    Args:
        hidden_states: [1, 1, users, hidden_size] (consumed)
        routing_weights: dense router output [users, num_experts] (0 for unselected experts; ``num_experts``
            includes the always-on slots)
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
    #     With expert groups (ProgramConfig.decode_gate_up_expert_groups) the 32-row activation tile is multicast once
    #     and stays resident while G groups of cores stream every G-th active expert concurrently (phase 3b; the union
    #     of ~72 experts at 32 users: 669 -> 276 us per layer on P150, bit-identical output).
    ip = weights.intermediate_padded_per_device
    gate_up_cfg = program_config.get_decode_gate_up_config(
        num_tokens, weights.gate_up_proj.shape[3], k=hidden_states.shape[-1]
    )
    gate_up = ttnn.sparse_matmul(
        hidden_states,
        weights.gate_up_proj,
        sparsity=sparsity,
        nnz=None,  # data-dependent union size: must be inferred on device (see decode_forward)
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=gate_up_cfg.program_config,
        expert_groups=gate_up_cfg.expert_groups,
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

    # 3. Per-(user, expert) routing weights [T, E] -> [1, E, T, 1]; zero for unselected pairs. Applied to the down
    #    INPUT (ROUTING_WEIGHTS_ON_DOWN_INPUT: 5 tiles wide per expert) or, phase-1 order, to the down OUTPUT (128
    #    tiles wide); the down projection is linear so both give the routed sum up to bfloat8_b rounding.
    token_expert_weights = ttnn.permute(routing_weights, (1, 0))
    token_expert_weights = ttnn.reshape(token_expert_weights, (1, num_experts, num_tokens, 1))
    if ROUTING_WEIGHTS_ON_DOWN_INPUT:
        down_input = ttnn.mul(down_input, token_expert_weights, output_tensor=down_input)
        token_expert_weights.deallocate(True)

    # 2b. Down projection (input is expert-batched too): [1, E, T, Ip] x [1, E, I, H] -> [1, E, T, H]
    #     (padded K: Ip == padded I of the weight; the extra input columns are zero). The grid choice takes the REAL
    #     user count (ProgramConfig.decode_down_batched_min_tokens selects the batched grid / expert groups: phase 3b
    #     runs 11 groups x 8 blocks of 16 tiles from 2 users on -- 216 -> 144 us per layer at the 32-user union; the
    #     legacy configs used 8x4 x 4 tiles below 16 users, whose small union favoured fewer multicast receivers per
    #     validity round trip, and 8x8 x 2 tiles above); per_core_M is 1 tile either way, so the M padding does not
    #     enter the config.
    down_cfg = program_config.get_decode_down_config(real_tokens, weights.down_proj.shape[-1], k=down_input.shape[-1])
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        program_config=down_cfg.program_config,
        expert_groups=down_cfg.expert_groups,
        dtype=activation_dtype,
    )
    down_input.deallocate(True)
    sparsity.deallocate(True)

    if not ROUTING_WEIGHTS_ON_DOWN_INPUT:
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
