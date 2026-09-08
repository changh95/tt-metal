# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill forward pass for experts (seq_len>1).

Trace safety (phase 3a(2), design_traced_prefill.md): on the dense (EP=1) path every split of a prefill chunk takes one
of the MOE_PATH_* paths chosen by ``moe_prefill_path`` from the split length alone (pure, no device state): the dense
bmm for splits up to ``dense_bmm_max_tokens`` (the traced 128-token prefill), the host-planned expert-sorted hot/cold
path for long eager splits (``_sorted_moe_plan`` reads the per-expert token counts back to the host -- never under a
trace capture, the captured plan would be replayed for other prompts), and the static per-expert loop for long splits
listed in ``ProgramConfig.trace_safe_split_lens`` (a traced 1K+ bucket's splits; ~2.5x the sorted path's cost, the
placeholder until a trace-safe sorted plan exists). ``is_trace_safe_prefill_len`` tells ModelArgs which prefill lengths
may be traced. The path's persistent device helpers (``down_proj_padded``, the one-hot identities of the sorted split
lengths) are built on the FIRST prefill call of any length (``_ensure_prefill_helpers``), so no persistent allocation
happens after a trace was captured.
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.solar_open.config import Mode

from .config import ExpertConfig, ProgramConfig
from .operations import (
    apply_expert_parallel_allreduce,
    apply_glu,
    apply_routing_weights,
    apply_sequence_parallel_allgather,
    apply_tensor_parallel_allreduce,
    reduce_experts,
)
from .weights import ExpertWeights


def _reshard_for_sequence_parallel(hidden_states, routing_weights, mesh_config, ccl_manager):
    """
    Convert replicated prefill inputs to SP row-sharded tensors using device-side CCL.

    This avoids host reads (`to_torch/get_device_tensors`) so it is trace-capture safe.
    The input tensors are replicated across rows, so reduce-scatter sums identical values.
    We rescale by 1/sp to recover the original values after sharding.
    """
    sp = mesh_config.get_config(Mode.PREFILL).sp
    if sp <= 1:
        return hidden_states, routing_weights

    cluster_axis = mesh_config.sp_axis
    scale = 1.0 / sp

    hidden_states_sharded = ttnn.reduce_scatter(
        hidden_states,
        dim=2,  # sequence dimension for hidden states: [1, B, S, H]
        cluster_axis=cluster_axis,
        memory_config=hidden_states.memory_config(),
        topology=ccl_manager.topology,
        num_links=ccl_manager.num_links,
    )
    routing_weights_sharded = ttnn.reduce_scatter(
        routing_weights,
        dim=0,  # sequence dimension for routing weights: [S, E]
        cluster_axis=cluster_axis,
        memory_config=routing_weights.memory_config(),
        topology=ccl_manager.topology,
        num_links=ccl_manager.num_links,
    )

    hidden_states_sharded = ttnn.mul(hidden_states_sharded, scale, output_tensor=hidden_states_sharded)
    routing_weights_sharded = ttnn.mul(routing_weights_sharded, scale, output_tensor=routing_weights_sharded)

    # Inputs are replaced by sharded outputs; release replicated tensors early.
    hidden_states.deallocate(True)
    routing_weights.deallocate(True)

    return hidden_states_sharded, routing_weights_sharded


def _process_prefill_chunk(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config: ExpertConfig,
    prefill_sparsity,
    program_config: ProgramConfig,
    ep,
    tp,
    dense_core_grid=None,
):
    """Process a single chunk of the sequence in prefill mode.

    The chunk is processed in `down_split_size` sub-splits along the sequence. For each split the fused
    gate/up projection runs over the EP group's experts, the result is split into its gate and up halves,
    the GLU is applied and the down projection follows; the per-expert outputs are weighted, reduced and
    stream-concatenated. Working per split keeps the peak DRAM footprint at a few split-sized
    [E, split, N] activations rather than chunk-sized ones.
    """
    _, batch_size, seq_len, hidden_size = hidden_states.shape
    activation_dtype = ttnn.bfloat8_b
    TILE_SIZE = 32
    ip = weights.intermediate_padded_per_device
    output_tile = ttnn.Tile([32, 32])
    experts_per_ep = config.num_experts // ep

    # Routing weights: zero the experts owned by other EP groups, then [S, E] -> [B, E, S, 1]
    # Note: prefill_sparsity is cached and reused, don't deallocate it. With EP=1 (every single-row mesh) the mask is
    # all ones and the multiply (plus the ROW_MAJOR -> TILE conversion of the [1, E] operand it forces inside the
    # binary op) would be a no-op launched once per chunk on the prefill hot path.
    if ep > 1:
        prefill_sparsity_reshaped = ttnn.reshape(prefill_sparsity, (1, config.num_experts))
        routing_weights = ttnn.mul(routing_weights, prefill_sparsity_reshaped, output_tensor=routing_weights)
    # EP=1 (single-row meshes, TP only): every device holds all experts, so the MoE runs as dense matmuls --
    # one batched [E, split, H] x [E, H, 2Ip] matmul (or one linear per expert) for gate/up and one batched
    # [E, split, Ip] x [E, Ip, H] matmul for down, or the expert-sorted hot/cold path for long splits. Measured on
    # P150 for a 1024-token split at H=2880 / Ip=384 (the shapes this module was tuned on, tt-metal PR #55589):
    # gate/up 24.5 -> 6.4 (+1.2 concat) ms, down 23.8 -> 3.7 ms versus the sparse_matmul path, whose 1D-multicast
    # kernel keeps the whole M on a few cores and re-streams every expert's weights once per 32-token tile.
    # EP>1 keeps the sparse path with a routing-aware mask for the fused gate/up projection: a 32-token group
    # only needs the experts routed to at least one of its tokens (with top-8 of 128 that is ~112 of 128 on
    # average, so the saving is small), and sparse_matmul's prefill cost is dominated by the per-(group, expert)
    # pair overhead. The down projection keeps the per-expert EP mask: its pairs are few and large, so per-group
    # sparsity would only add pairs. nnz is left to the kernel for the gate/up call -- it must equal
    # count_nonzero exactly when given.
    dense_moe = ep == 1 and dense_core_grid is not None
    if dense_moe:
        _ensure_prefill_helpers(weights, config, program_config)  # idempotent; allocates on the first call only
    group_mask = (
        None if dense_moe else _group_expert_mask(routing_weights, seq_len, config.num_experts)
    )  # [1, S/32, 1, E] row-major
    # Token-major routing weights ([1, 1, S, E], a view) for the dense path's expert-sorted planner; sliced per split.
    routing_tokens_all = ttnn.reshape(routing_weights, (1, 1, seq_len, config.num_experts)) if dense_moe else None
    # Expert-major routing weights [B, E, S, 1] -- the broadcast operand of apply_routing_weights. The reshape
    # [E, S] -> [E, S, 1] re-tiles every element into its own tile row (a real copy: 128 us per 1024 tokens, 32 MiB
    # written per 4096-token chunk). Only the sparse EP path and the dense bmm / per-expert-loop splits read it; the
    # expert-sorted splits (every full 1024-token split of a dense chunk) never do, so the dense path builds it per
    # split on demand (_expert_major_routing) when ELIDE_ROUTING_COPIES is set.
    lazy_routing = dense_moe and ELIDE_ROUTING_COPIES
    if not lazy_routing:
        # Note: permute/reshape operations return views - do not deallocate originals
        routing_weights = ttnn.permute(routing_weights, (1, 0))
        routing_weights = ttnn.reshape(routing_weights, (batch_size, config.num_experts, seq_len, 1))

    # This function consumes hidden_states and routing_weights (the split copies, or the tensors
    # themselves when there is a single split, are released as each split is processed; with lazy_routing the
    # caller's routing tensor is released through its routing_tokens_all view at the end).
    split_size = program_config.get_down_split_size(seq_len)
    if seq_len > split_size:
        hidden_list = ttnn.split(hidden_states, split_size, dim=2)
        hidden_states.deallocate(True)  # the splits are device copies; the chunk is dead from here on
        if lazy_routing:
            routing_list = [None] * len(hidden_list)
        else:
            routing_list = ttnn.split(routing_weights, split_size, dim=2)
            routing_weights.deallocate(True)
    else:
        hidden_list = [hidden_states]
        routing_list = [None] if lazy_routing else [routing_weights]

    # Process each split and stream-concatenate to avoid holding all split outputs.
    next_states_reduced_acc = None
    group_offset = 0
    token_offset = 0
    for hidden_split, routing_split in zip(hidden_list, routing_list):
        split_len = hidden_split.shape[2]
        group_size = split_len // TILE_SIZE

        if dense_moe:
            hidden_4D = ttnn.unsqueeze_to_4D(hidden_split)  # [1, 1, split, H] (view of the split)
            # Path selection (moe_prefill_path, pure): the one-launch dense bmm for short splits, the host-planned
            # sorted hot/cold path for long eager splits, the static per-expert loop for long splits that must stay
            # trace-safe (program_config.trace_safe_split_lens) or for models with too few experts to sort.
            path = moe_prefill_path(split_len, program_config, config.num_experts, weights.num_always_on_experts)
            bmm_config = (
                _dense_bmm_config(dense_core_grid, split_len, weights, program_config.dense_bmm_max_tokens)
                if path == MOE_PATH_DENSE_BMM
                else None
            )
            plan = None
            if path == MOE_PATH_HOST_PLANNED:
                plan = _sorted_moe_plan(
                    routing_tokens_all,
                    token_offset,
                    split_len,
                    config,
                    program_config.dense_bmm_max_tokens,
                    always_on=weights.num_always_on_experts,
                    trace_safe_split_lens=program_config.trace_safe_split_lens,
                )
            LAST_PREFILL_MOE_PATH.update(split=split_len, path=path, sorted=plan is not None)
            if plan is not None:
                next_states_reduced = _sorted_moe_forward(
                    hidden_4D,
                    plan,
                    split_len,
                    weights,
                    config,
                    activation_dtype,
                    dense_core_grid,
                    program_config=program_config,
                )
            else:
                if routing_split is None:  # lazy_routing: this split multiplies per-expert outputs by the weights
                    routing_split = _expert_major_routing(
                        routing_tokens_all, token_offset, split_len, config.num_experts
                    )
                gate_up = _dense_gate_up(
                    hidden_4D,
                    bmm_config,
                    weights,
                    config,
                    activation_dtype,
                    dense_core_grid,
                    program_config=program_config,
                )
                next_states_reduced = _dense_tail(
                    gate_up,
                    routing_split,
                    split_len,
                    weights,
                    config,
                    activation_dtype,
                    dense_core_grid,
                    ip,
                    down_program_config=_dense_down_program_config(program_config, split_len, config.hidden_size, ip),
                )
        else:
            LAST_PREFILL_MOE_PATH.update(split=split_len, path=MOE_PATH_SPARSE_EP, sorted=False)
            # Group tokens into tiles: [1, B, split, H] -> [1, G, 32, H]. This reshape is a view of
            # hidden_split, so deallocating hidden_4D below releases the split itself (intended).
            hidden_4D = ttnn.unsqueeze_to_4D(hidden_split)
            hidden_4D = ttnn.reshape(hidden_4D, (1, group_size, TILE_SIZE, config.hidden_size))
            split_mask = ttnn.slice(
                group_mask, [0, group_offset, 0, 0], [1, group_offset + group_size, 1, config.num_experts]
            )
            group_offset += group_size

            # Fused gate/up projection: [1, G, 32, H] x [1, E, H, 2 * Ip] -> [1, G, 1, E, 32, 2 * Ip]
            # (skipped (group, expert) pairs are zero-filled by the op)
            gate_up = ttnn.sparse_matmul(
                hidden_4D,
                weights.gate_up_proj,
                sparsity=split_mask,
                nnz=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=program_config.get_prefill_gate_up_config(
                    hidden_4D.shape[2], weights.gate_up_proj.shape[3], k=hidden_4D.shape[-1]
                ),
                dtype=activation_dtype,
            )
            hidden_4D.deallocate(True)
            split_mask.deallocate(True)
            # Note: transpose/reshape operations return views - do not deallocate originals
            gate_up = ttnn.transpose(gate_up, 1, 3)
            gate_up = ttnn.reshape(gate_up, (batch_size, config.num_experts, split_len, 2 * ip))
            # Split at the tile-aligned half: gate = [..., :Ip], up = [..., Ip:]
            gate = ttnn.slice(gate_up, [0, 0, 0, 0], [batch_size, config.num_experts, split_len, ip])
            up = ttnn.slice(gate_up, [0, 0, 0, ip], [batch_size, config.num_experts, split_len, 2 * ip])
            gate_up.deallocate(True)
            # GLU: [B, E, split, Ip]; the zero-padded columns stay exactly 0.
            down_input = apply_glu(gate, up, config)
            gate.deallocate(True)
            up.deallocate(True)
            down = ttnn.sparse_matmul(
                down_input,
                weights.down_proj,
                sparsity=prefill_sparsity,
                nnz=experts_per_ep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                is_input_a_sparse=True,
                program_config=program_config.get_prefill_down_config(
                    down_input.shape[2], weights.down_proj.shape[-1], k=down_input.shape[-1]
                ),
                dtype=activation_dtype,
            )
            down_input.deallocate(True)
            # Apply routing weights, reduce across experts
            # Note: reshape returns a view - do not deallocate original
            next_states = ttnn.reshape(down, (batch_size, config.num_experts, split_len, config.hidden_size))
            next_states = apply_routing_weights(next_states, routing_split)
            next_states_reduced = reduce_experts(next_states)
            down.deallocate(True)

        if next_states_reduced_acc is None:
            next_states_reduced_acc = next_states_reduced
        else:
            # ToDo: Replace with slice_write.
            # Concat re-creates the output_tensor every iteration.
            next_states_concat = ttnn.concat([next_states_reduced_acc, next_states_reduced], dim=2)
            next_states_reduced_acc.deallocate(True)
            next_states_reduced.deallocate(True)
            next_states_reduced_acc = next_states_concat
        if routing_split is not None:
            routing_split.deallocate(True)
        token_offset += split_len
    if group_mask is not None:
        group_mask.deallocate(True)
    if routing_tokens_all is not None:
        routing_tokens_all.deallocate(True)

    return next_states_reduced_acc


def _expert_slice(weight, e, rows, cols):
    """Device copy of expert `e` of a [1, E, rows, cols] weight -> [1, 1, rows, cols]. The caller frees it."""
    return ttnn.slice(weight, [0, e, 0, 0], [1, e + 1, rows, cols])


def _concat_expert_slices(weight, expert_ids, rows, cols, dim=1):
    """On-demand copy of the listed experts of a [1, E, rows, cols] weight, concatenated along ``dim``: dim 1 ->
    [1, len(expert_ids), rows, cols] (batched matmul operand), dim 2 -> [1, 1, len(expert_ids) * rows, cols] (the
    K-concatenated down weights of the hot group, HOT_DOWN_KCONCAT).

    Per-expert slices are transient: with one expert the slice itself is returned, otherwise the slices are
    concatenated and freed right away. The caller deallocates the returned tensor after use, so no per-expert
    weight copy outlives the split (the persistent per-expert cache would duplicate every expert weight)."""
    slices = [_expert_slice(weight, e, rows, cols) for e in expert_ids]
    if len(slices) == 1:
        return slices[0]
    cat = ttnn.concat(slices, dim=dim)
    for s in slices:
        s.deallocate(True)
    return cat


def _expert_major_routing(routing_tokens_all, token_offset, split_len, num_experts):
    """[1, E, split, 1] routing weights of one split (the broadcast operand of apply_routing_weights) from the chunk's
    token-major [1, 1, S, E] tensor: slice, transpose to [1, 1, E, split] and the re-tiling copy to [1, E, split, 1].
    Built on demand by the dense path for the splits that read it (dense bmm tail, per-expert loop)."""
    tokens = ttnn.slice(routing_tokens_all, [0, 0, token_offset, 0], [1, 1, token_offset + split_len, num_experts])
    routing_t = ttnn.transpose(tokens, 2, 3)
    if split_len != routing_tokens_all.shape[2]:  # a full-range slice aliases its input
        tokens.deallocate(True)
    # Note: reshape may return a view - do not deallocate routing_t (released with the result / by the refcount)
    return ttnn.reshape(routing_t, (1, num_experts, split_len, 1))


def _dense_gate_up(hidden_4D, bmm_config, weights, config, activation_dtype, dense_core_grid, program_config=None):
    """Fused gate/up projection for a whole split, [1, 1, split, H] -> [1, E, split, 2Ip]. Consumes hidden_4D.
    Short splits (bmm_config given): replicate the activations per expert and run ONE batched matmul (one expert
    per core; 128 separate launches cost ~30 us each on device, which dominates 128-token prefills); with
    ``program_config.dense_activation_bfp8`` the activation is rounded to bfloat8_b before the replication (half the
    broadcast bytes, bfp8 in0 for the bmm -- a numerics switch, off by default). Otherwise one linear per expert over
    the whole split (weight slice taken and freed on demand; ``_expert_linear``: minimal_matmul with
    ``program_config.dense_expert_gate_up_minimal`` or the auto ttnn.linear), concatenated."""
    if bmm_config is not None:
        if program_config is not None and program_config.dense_activation_bfp8 and hidden_4D.dtype != ttnn.bfloat8_b:
            hidden_8 = ttnn.typecast(hidden_4D, ttnn.bfloat8_b)
            hidden_4D.deallocate(True)
            hidden_4D = hidden_8
        hidden_rep = ttnn.repeat(hidden_4D, ttnn.Shape((1, config.num_experts, 1, 1)))
        hidden_4D.deallocate(True)
        gate_up = ttnn.matmul(
            hidden_rep,
            weights.gate_up_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=activation_dtype,
            program_config=bmm_config,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
        hidden_rep.deallocate(True)
        return gate_up
    hidden, n = weights.gate_up_proj.shape[2], weights.gate_up_proj.shape[3]
    gu_config = _dense_expert_gate_up_config(program_config, dense_core_grid, hidden_4D.shape[2], hidden, n)
    per_expert = []
    for e in range(config.num_experts):
        w_e = _expert_slice(weights.gate_up_proj, e, hidden, n)
        per_expert.append(_expert_linear(hidden_4D, w_e, gu_config, dense_core_grid, activation_dtype))
        w_e.deallocate(True)
    hidden_4D.deallocate(True)
    gate_up = ttnn.concat(per_expert, dim=1)
    for t in per_expert:
        t.deallocate(True)
    return gate_up


def _dense_down_program_config(program_config, rows, hidden_size, ip):
    """1D-multicast program config of the dense down bmm ``[1, E, rows, ip] x [1, E, ip, H]``
    (``ProgramConfig.get_dense_down_config``), or None for the ``core_grid=`` auto config: when ``dense_down_cores``
    is unset, or for row counts outside the validated range (``rows`` must be a tile multiple <=
    ``dense_bmm_max_tokens``: the dense tail's split length and the sorted path's cap, 32..256; the per-expert loop
    of longer splits and the hot group keep the auto config)."""
    if program_config is None or rows % ttnn.TILE_SIZE != 0 or rows > program_config.dense_bmm_max_tokens:
        return None
    return program_config.get_dense_down_config(rows, hidden_size, ip)


def _dense_down_matmul(down_input, weights, activation_dtype, dense_core_grid, down_program_config):
    """``down_input`` [1, E, rows, Ip] x down_proj_padded [1, E, Ip, H] -> [1, E, rows, H] with the explicit 1D config
    when given, else ttnn's auto config on ``dense_core_grid``."""
    placement = (
        {"program_config": down_program_config} if down_program_config is not None else {"core_grid": dense_core_grid}
    )
    return ttnn.matmul(
        down_input,
        weights.down_proj_padded,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=activation_dtype,
        compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        **placement,
    )


def _dense_expert_gate_up_config(program_config, dense_core_grid, m, k, n):
    """``ttnn.MinimalMatmulConfig`` of the per-expert fused gate|up linear ``[1, 1, m, k] x [1, 1, k, n]`` of the dense
    per-expert loop / the sorted hot group (``ProgramConfig.get_dense_expert_gate_up_config`` with the dense grid as
    the fit check), or None for ``ttnn.linear``'s auto config."""
    if program_config is None:
        return None
    return program_config.get_dense_expert_gate_up_config(m, k, n, grid=dense_core_grid)


def _hot_down_kconcat_minimal_config(program_config, dense_core_grid, m, k, n):
    """``ttnn.MinimalMatmulConfig`` of the hot group's K-concatenated down ``[1, 1, m, k = n_hot * Ip] x [1, 1, k, n]``
    (``ProgramConfig.get_hot_down_kconcat_config``), or None when the hot down keeps the perf-p1 forms."""
    if program_config is None:
        return None
    return program_config.get_hot_down_kconcat_config(m, k, n, grid=dense_core_grid)


def _expert_linear(hidden_4D, w_e, minimal_config, dense_core_grid, activation_dtype):
    """One expert's fused gate|up over a whole split: ``hidden_4D`` [1, 1, split, K] x ``w_e`` [1, 1, K, N] ->
    [1, 1, split, N] in ``activation_dtype`` (DRAM). ``ttnn.experimental.minimal_matmul`` with ``minimal_config`` when
    given (phase 3: 46 vs 81 us per hot expert at 1024 tokens on P150, ``MinimalMatmulBlocking``), else ``ttnn.linear``
    with ttnn's auto config on ``dense_core_grid`` (a 2D multicast config with in0_block_w 4)."""
    if minimal_config is not None:
        return ttnn.experimental.minimal_matmul(
            hidden_4D,
            w_e,
            config=minimal_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=activation_dtype,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
    return ttnn.linear(
        hidden_4D,
        w_e,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=activation_dtype,
        core_grid=dense_core_grid,
        compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
    )


def _glu(gate, up, config, dtype=None):
    """``apply_glu`` with an explicit output dtype: ``dtype`` None (or the dtype of ``up``) is the plain
    ``apply_glu``; otherwise the same single fused binary op ``up * silu(gate)`` packs its output in ``dtype`` (the hot
    group's K-concatenated down wants bf16 GLU pieces: their 320-B rows make the last-dim concat an exact page copy)."""
    if dtype is None or dtype == up.dtype:
        return apply_glu(gate, up, config)
    activation = getattr(config, "activation", config)
    if activation != "silu":
        raise NotImplementedError(f"activation {activation!r} (only silu is implemented for Solar-Open)")
    return ttnn.mul(up, gate, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)], dtype=dtype)


def _dense_tail(
    gate_up, routing_split, split_len, weights, config, activation_dtype, dense_core_grid, ip, down_program_config=None
):
    """[1, E, split, 2Ip] gate/up -> [1, 1, split, H] MoE output for the split. Consumes gate_up.
    The routing weights are applied to the down INPUT (a fraction of the down output's size; exact since down is
    linear) and the per-expert outputs are reduced over the expert dimension. ``down_program_config`` (from
    ``_dense_down_program_config``) selects the 1D down config; None = auto on ``dense_core_grid``."""
    E = config.num_experts
    gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, E, split_len, ip])
    up = ttnn.slice(gate_up, [0, 0, 0, ip], [1, E, split_len, 2 * ip])
    gate_up.deallocate(True)
    down_input = apply_glu(gate, up, config)  # one fused binary op
    gate.deallocate(True)
    up.deallocate(True)
    down_input = apply_routing_weights(down_input, routing_split)
    down = _dense_down_matmul(down_input, weights, activation_dtype, dense_core_grid, down_program_config)
    down_input.deallocate(True)
    reduced = reduce_experts(down)
    down.deallocate(True)
    return reduced


_SORTED_MOE_DEBUG = os.getenv("SOLAR_OPEN_SORTED_MOE_DEBUG", "0") == "1"
# Last plan chosen by _sorted_moe_plan ({"split", "cap", "hot"}); read by tests to assert which path ran.
LAST_SORTED_MOE_PLAN = {}
_SORTED_MOE_MAX_HOT = 16  # more hot experts than this -> dense per-expert loop for the split
# Cost model (ms per 1024-token split, P150) used to pick the hot/cold threshold on the host. Re-measured for the
# Solar-Open shapes (H=4096 / Ip=160, bfp8) from the tracy device profile of the real-weight layer 0 at 1024 tokens
# (2026-09-07; the shipped values were the H=2880 / Ip=384 numbers of tt-metal PR #55589: 2.5, 0.27, 1.0, 0.25 and
# 0.125): fixed sorted cost ~0.5 ms (planner ops, topk, index/slot glue, untilizes), 0.23 ms per 1024 gathered rows
# (embedding gather 46 + gate/up bmm 49 + down bmm 72 + one-hot scatter matmul 49 + slices/GLU/mul 9 us), hot group
# fixed ~0.3 ms (concats, reduce, adds) + 0.25 ms per hot expert (repeat + bmm + slices), dense per-expert loop
# 0.15 ms (gate/up linear 77 + down 33 + 2 weight slices 18 + GLU/mul/concat share ~20 us). The constants only
# steer the hot/cold split and the sorted-vs-dense choice, never correctness. perf-p1 (2026-09-07): with the hot
# group as per-expert linears (HOT_EXPERTS_PER_EXPERT_LINEAR) its wall time at 1024 tokens fits 0.39 + 0.136 ms x
# n_hot (measured 0.93 / 1.48 / 2.42 ms at 4 / 8 / 15 hot experts vs 1.25 / 2.26 / 4.06 for repeat + bmm, i.e.
# 0.23 + 0.255 x n_hot) -> _HOT_FIXED_MS, _HOT_PER_EXPERT_MS = 0.4, 0.135. Phase 3 (the minimal_matmul gate|up and
# the K-concatenated hot down of SolarOpenProgramConfig) makes the hot group cheaper (~-35 us per expert, ~-0.3 ms at
# 15 hot); the constants are conservative until re-derived from a layer-0 tracy (pinned by tests/unit/test_p1_layout).
_SORTED_FIXED_MS, _SORTED_PER_KROW_MS, _HOT_FIXED_MS, _HOT_PER_EXPERT_MS = 0.5, 0.23, 0.4, 0.135
_DENSE_PER_EXPERT_MS = 0.15  # per-expert cost of the dense loop over a 1024-token split (gate/up + down + slices)
# Measured on P150x8 (PR #55589): the sorted path halves E=128 prefill at ISL >= 1024 but is slower than the dense
# loop for E=32 (~128 routed tokens per expert per 1024, so the gathered rows are not much fewer and the fixed
# cost + host round-trip dominate).
_SORTED_MOE_MIN_EXPERTS = 64

# perf-p1 layout switches (2026-09-07; module constants so an A/B run can flip them, all default ON):
# - HOT_EXPERTS_PER_EXPERT_LINEAR: the hot group's gate/up runs as one ttnn.linear per hot expert over the whole
#   split (full-grid 2D matmul, weight slice taken and freed) stacked along the expert dim, instead of ttnn.repeat of
#   the activations x n_hot + one batched matmul (one expert per core). Same math (per-expert products are exact
#   copies of the dense per-expert loop's).
# - HOT_DOWN_KCONCAT: the hot group's down projection + reduce over the hot experts run as ONE matmul over the
#   concatenated K, [split, n_hot * Ip] x [n_hot * Ip, H] = sum_e act_e @ down_e, instead of a batched matmul and
#   fast_reduce_nc over n_hot bfloat8_b per-expert outputs. Measured on P150 at 1024 tokens (tests/perf/
#   test_layout_candidates.py, wall of the whole hot group, n_hot 4 / 8 / 15): K-concat 1.07 / 1.67 / 2.75 ms vs
#   bmm + fast_reduce_nc 0.93 / 1.48 / 2.42 ms (phase-1 repeat + bmm form: 1.25 / 2.26 / 4.06), PCC vs fp32 equal
#   within 3e-5 -- so it stays OFF (kept as the measured alternative). Phase 3 traced that verdict to the bfp8 act
#   pieces (160-B rows: the concat takes the transpose fallback, 71-204 us) and the legacy 2D config (150 us at 15
#   hot): ``program_config.hot_down_kconcat_minimal`` (SolarOpenProgramConfig default) runs the K-concat over bf16
#   GLU pieces with a minimal_matmul blocking (concat 36 + matmul 117 us at 15 hot vs bmm 394 + reduce 205) and takes
#   precedence over this switch; HOT_DOWN_KCONCAT only matters when that field is None.
# - ELIDE_ROUTING_COPIES: skip the routing-tensor copies the sorted path never reads / can avoid: the expert-major
#   [E, S, 1] re-tiling of the chunk's routing weights (built per split on demand for the dense bmm / per-expert-loop
#   splits only), the [1, E * cap] row-major flattening of the cold index tensor (ttnn.embedding takes [E, cap]) and
#   the hot routing rows' row-major [1, n_hot, split, 1] reshape (replaced by a tile transpose). Bit-identical.
HOT_EXPERTS_PER_EXPERT_LINEAR = True
HOT_DOWN_KCONCAT = False
ELIDE_ROUTING_COPIES = True

# ---------------------------------------------------------------------------------------------------------------------
# Prefill MoE path selection (phase 3a(2), design_traced_prefill.md). One split of the dense (EP=1) path runs as
#   MOE_PATH_DENSE_BMM     splits <= dense_bmm_max_tokens: activations replicated per expert, ONE batched gate|up
#                          matmul, dense down bmm, fast_reduce_nc -- the traced 128-token prefill; static and device-only
#   MOE_PATH_STATIC_LOOP   one ttnn.linear per expert over the whole split + the dense tail: static and device-only
#                          (trace-safe) but ~2.5x the sorted path's cost at 1024 tokens; taken for the splits listed in
#                          program_config.trace_safe_split_lens and for models with < _SORTED_MOE_MIN_EXPERTS experts
#   MOE_PATH_HOST_PLANNED  the expert-sorted hot/cold path planned from a device->host read of the per-expert token
#                          counts (_sorted_moe_plan; it falls back to the per-expert loop when sorting is not cheaper):
#                          every long eager split. NEVER under a trace capture -- the captured plan would be replayed
#                          for other prompts.
# The EP > 1 sparse_matmul path of multi-row meshes (MOE_PATH_SPARSE_EP) uses static masks and is trace-safe.
MOE_PATH_DENSE_BMM = "dense_bmm"
MOE_PATH_STATIC_LOOP = "static_loop"
MOE_PATH_HOST_PLANNED = "host_planned"
MOE_PATH_SPARSE_EP = "sparse_ep"
TRACE_SAFE_MOE_PATHS = frozenset({MOE_PATH_DENSE_BMM, MOE_PATH_STATIC_LOOP, MOE_PATH_SPARSE_EP})
# Path of the LAST split _process_prefill_chunk processed ({"split", "path", "sorted"}); read by tests.
LAST_PREFILL_MOE_PATH = {}


def _pieces(length, size):
    """Piece lengths ``ttnn.split(x, size)`` cuts a ``length``-long axis into: full pieces, then the remainder."""
    return [size] * (length // size) + ([length % size] if length % size else [])


def prefill_split_lens(seq_len, program_config):
    """Distinct split lengths (first-appearance order) the MoE processes for a ``seq_len``-token prefill: the sequence
    is cut into ``sequence_chunk_size`` chunks (prefill_forward) and every chunk into
    ``get_down_split_size(chunk_len)`` splits (_process_prefill_chunk). Pure: mirrors the two ttnn.split calls."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    lens = []
    for chunk_len in _pieces(seq_len, program_config.sequence_chunk_size):
        for split_len in _pieces(chunk_len, program_config.get_down_split_size(chunk_len)):
            if split_len not in lens:
                lens.append(split_len)
    return tuple(lens)


def moe_prefill_path(split_len, program_config, num_experts, always_on=0, dense_moe=True):
    """The MoE path (a MOE_PATH_* constant) _process_prefill_chunk takes for one ``split_len``-token split.
    ``num_experts`` counts the expert slots including ``always_on`` fused shared-expert slots; ``dense_moe`` is False
    on the EP > 1 sparse path of multi-row meshes. Pure."""
    if not dense_moe:
        return MOE_PATH_SPARSE_EP
    if split_len <= program_config.dense_bmm_max_tokens:
        return MOE_PATH_DENSE_BMM
    if num_experts - always_on < _SORTED_MOE_MIN_EXPERTS or split_len in program_config.trace_safe_split_lens:
        return MOE_PATH_STATIC_LOOP
    return MOE_PATH_HOST_PLANNED


def is_trace_safe_prefill_len(seq_len, program_config, num_experts, always_on=0, dense_moe=True):
    """True when every split of a ``seq_len``-token prefill takes a trace-safe MoE path (no device->host read, static
    shapes) -- the condition for listing ``seq_len`` among ModelArgs' traced prefill lengths. Pure."""
    return all(
        moe_prefill_path(split_len, program_config, num_experts, always_on, dense_moe) in TRACE_SAFE_MOE_PATHS
        for split_len in prefill_split_lens(seq_len, program_config)
    )


def _plan_from_counts(routed_counts, split_len, num_experts, always_on=0):
    """Cost-model core of _sorted_moe_plan on the host-side per-expert routed-token counts (torch int64 [E_routed]).

    Returns ``(cost_ms, cap, n_hot)`` for the cheapest cap of (32, 64, 96, 128, 160, 192, 256) below ``split_len``
    with at most _SORTED_MOE_MAX_HOT routed hot experts (count > cap), or None when every cap is over the hot limit
    or none beats the dense per-expert loop (``_DENSE_PER_EXPERT_MS`` x E x split / 1024). Pure function (unit-tested
    on synthetic count distributions)."""
    best = None
    for cap in (32, 64, 96, 128, 160, 192, 256):
        if cap >= split_len:
            # strict: an always-on slot has count == split_len and must stay OUT of the cold mask le(counts, cap)
            break
        hot = int((routed_counts > cap).sum().item())
        if hot > _SORTED_MOE_MAX_HOT:
            continue
        n_hot_group = hot + always_on
        cost = (
            _SORTED_FIXED_MS
            + _SORTED_PER_KROW_MS * (num_experts * cap / 1024)
            + (_HOT_FIXED_MS + _HOT_PER_EXPERT_MS * n_hot_group if n_hot_group else 0.0)
        )
        if best is None or cost < best[0]:
            best = (cost, cap, hot)
    # The sorted path only pays off when the routed rows are few relative to E x split (E=128, top-8: ~64 routed
    # tokens per expert per 1024); when it is not cheaper than the dense per-expert loop (which has no host
    # round-trip) the loop is kept.
    dense_cost = _DENSE_PER_EXPERT_MS * num_experts * split_len / 1024
    if best is None or best[0] >= dense_cost:
        return None
    return best


def _sorted_moe_plan(
    routing_tokens_all, token_offset, split_len, config, dense_bmm_max_tokens, always_on=0, trace_safe_split_lens=()
):
    """Host-side plan for one split from the per-expert routed-token counts (one small device->host read).

    Real MoE routing is skewed (the hottest expert of a 1024-token split often takes a large share of the
    tokens), so the experts are partitioned: HOT experts (count > cap) run dense over the whole split as a small
    batched group, COLD experts run expert-sorted with `cap` gathered rows each. `cap` is chosen from a small cost
    model over the count distribution. Returns (routing^T [1, 1, E, split], cap, hot_ids, cold_mask) or None (use
    the dense per-expert loop when too many experts are hot).

    ``always_on`` trailing slots of the E axis (the fused shared expert) are routed to by EVERY token (count ==
    split_len > every cap): they are always hot, ride in the hot group at no extra launch, are excluded from the hot
    count / _SORTED_MOE_MAX_HOT threshold (which keeps its meaning: routed hot experts) and get 0 in the cold mask
    (never double counted)."""
    E = config.num_experts
    E_routed = E - always_on
    if E_routed < _SORTED_MOE_MIN_EXPERTS:
        return None
    # This does a device->host read of the per-expert counts, so it must never run under trace capture (a captured
    # plan would be replayed for other prompts). moe_prefill_path only routes a split here when it is longer than
    # program_config.dense_bmm_max_tokens (256) AND not listed in program_config.trace_safe_split_lens (the splits of
    # the traced prefill buckets; the only shipped bucket, 128 tokens, never reaches this function) -- asserted again
    # here so no caller can bypass the selection.
    assert split_len > dense_bmm_max_tokens, "the sorted MoE path is for eager (untraced) long splits only"
    assert split_len not in trace_safe_split_lens, f"split {split_len} is trace-safe only (trace_safe_split_lens)"
    routing_tokens = ttnn.slice(routing_tokens_all, [0, 0, token_offset, 0], [1, 1, token_offset + split_len, E])
    routing_t = ttnn.transpose(routing_tokens, 2, 3)  # [1, 1, E, split]
    if split_len != routing_tokens_all.shape[2]:  # a full-range slice aliases its input
        routing_tokens.deallocate(True)
    active = ttnn.gt(routing_t, 0.0)
    counts = ttnn.sum(active, dim=3, keepdim=True)  # [1, 1, E, 1]
    active.deallocate(True)
    # routing weights are replicated across the TP devices, so one device's counts suffice (mesh tensors need a
    # composer for a direct to_torch)
    counts_host = ttnn.to_torch(ttnn.get_device_tensors(counts)[0]).reshape(-1).to(torch.int64)
    routed_counts = counts_host[:E_routed]
    best = _plan_from_counts(routed_counts, split_len, E, always_on)
    if best is None:
        routing_t.deallocate(True)
        counts.deallocate(True)
        return None
    _, cap, n_hot = best
    assert cap < split_len, (cap, split_len)  # the cold mask relies on count(always-on) == split_len > cap
    hot_ids = [int(e) for e in torch.nonzero(routed_counts > cap).reshape(-1).tolist()] + list(range(E_routed, E))
    LAST_SORTED_MOE_PLAN.update(split=split_len, cap=cap, hot=n_hot, always_on=always_on)
    if _SORTED_MOE_DEBUG:
        top = routed_counts.topk(min(4, E_routed)).values.tolist()
        logger.info(
            f"SORTED-MOE split={split_len} cap={cap} hot={n_hot} always_on={always_on} top4={top} "
            f"zero={(routed_counts == 0).sum().item()}"
        )
    # cold mask on device (1.0 for experts handled by the sorted path): no per-split host upload; the always-on
    # slots have count == split_len > cap -> 0.
    cold_mask_t = ttnn.le(counts, float(cap))
    counts.deallocate(True)
    return routing_t, cap, hot_ids, cold_mask_t


def _sorted_moe_forward(
    hidden_4D, plan, split_len, weights, config, activation_dtype, dense_core_grid, program_config=None
):
    """Hot/cold expert-sorted MoE for one split ([1, 1, split, H] -> [1, 1, split, H]); consumes hidden_4D.

    Cold experts: topk over the transposed routing weights gives each expert its `cap` largest-weight tokens (all
    its routed tokens, then zero-weight fillers); ttnn.embedding gathers those rows (and one-hot rows from a cached
    identity), gate/up and down run as batched matmuls over the gathered [E, cap, *] rows only, each row is scaled by
    its slot weight (zeroed for hot experts) and scattered back with one-hot^T @ rows. Hot experts (their routed-token
    count exceeds `cap`): gate/up run as one linear per hot expert over the whole split (weight slices taken on
    demand and freed; HOT_EXPERTS_PER_EXPERT_LINEAR, off = the phase-1 repeat + batched matmul; ``_expert_linear``:
    minimal_matmul with ``program_config.dense_expert_gate_up_minimal`` or the auto ttnn.linear), the GLU outputs are
    weighted by their routing weights, and the down projections + the sum over the hot experts run as ONE
    K-concatenated minimal_matmul over bf16 GLU pieces (``program_config.hot_down_kconcat_minimal``; without it the
    perf-p1 forms: one batched matmul reduced with fast_reduce_nc, or HOT_DOWN_KCONCAT's 2D K-concat). The math
    equals the dense path. ``program_config`` also supplies the 1D config of the cold down bmm (rows = cap, see
    ``_dense_down_program_config``)."""
    routing_t, cap, hot_ids, cold_mask_t = plan
    E, H, ip = config.num_experts, config.hidden_size, weights.intermediate_padded_per_device
    device = weights.gate_up_proj.device()
    table = ttnn.reshape(hidden_4D, (split_len, H))
    if table.dtype != ttnn.bfloat16:  # embedding gathers from a bf16 table
        table16 = ttnn.typecast(table, ttnn.bfloat16)
        hidden_4D.deallocate(True)
        table = table16
        hidden_4D = ttnn.reshape(table, (1, 1, split_len, H))

    # ---- cold experts: sorted / gathered rows ----
    vals, idx = ttnn.topk(routing_t, k=cap, dim=3, largest=True)  # [1, 1, E, cap]
    if hot_ids:
        vals = ttnn.mul(vals, cold_mask_t, output_tensor=vals)  # hot experts contribute via the dense group below
        hot_idx_t = ttnn.from_torch(
            torch.tensor([hot_ids], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )  # [1, n_hot]
        routing_t_table = ttnn.reshape(routing_t, (E, split_len))  # gather table for the hot routing rows
    else:
        routing_t.deallocate(True)
    cold_mask_t.deallocate(True)
    idx_rm = ttnn.to_layout(ttnn.typecast(idx, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, E, cap] uint32
    idx.deallocate(True)
    # ttnn.embedding reads [batch, tokens] indices and returns [batch, 1, tokens, width]: [E, cap] is a 0-cost view of
    # the row-major index tensor (the [1, E * cap] flattening was a 54 us row-major reshape copy per split) and the
    # gathered [E, 1, cap, *] rows are re-viewed to [1, E, cap, H] / [1, 1, E * cap, split] for free (tile-aligned).
    idx_flat = ttnn.reshape(idx_rm, (E, cap) if ELIDE_ROUTING_COPIES else (1, E * cap))
    rows = ttnn.reshape(ttnn.embedding(idx_flat, table, layout=ttnn.TILE_LAYOUT), (1, E, cap, H))
    onehot = ttnn.reshape(
        ttnn.embedding(idx_flat, _eye(weights, split_len), layout=ttnn.TILE_LAYOUT), (1, 1, E * cap, split_len)
    )
    idx_flat.deallocate(True)
    idx_rm.deallocate(True)
    gate_up = ttnn.matmul(
        rows,
        weights.gate_up_proj,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=activation_dtype,
        program_config=_bmm_config(dense_core_grid, cap // 32, H // 32, (2 * ip) // 32),
        compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
    )
    rows.deallocate(True)
    gate = ttnn.slice(gate_up, [0, 0, 0, 0], [1, E, cap, ip])
    up = ttnn.slice(gate_up, [0, 0, 0, ip], [1, E, cap, 2 * ip])
    gate_up.deallocate(True)
    act = apply_glu(gate, up, config)
    gate.deallocate(True)
    up.deallocate(True)
    slot_w = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(vals, ttnn.ROW_MAJOR_LAYOUT), (1, E, cap, 1)), ttnn.TILE_LAYOUT)
    vals.deallocate(True)
    act = ttnn.mul(act, slot_w, output_tensor=act)
    slot_w.deallocate(True)
    down = _dense_down_matmul(
        act, weights, activation_dtype, dense_core_grid, _dense_down_program_config(program_config, cap, H, ip)
    )
    act.deallocate(True)
    # Scatter back: out[split, H] = onehot^T [split, E*cap] @ down[E*cap, H]. A token's 8 (+1) expert contributions
    # sit at K positions e * cap + c that depend on the sorted LAYOUT (c = the token's rank among its expert's routed
    # tokens); with a bf16 destination the partial sums are rounded at K-block boundaries, so the same token gets a
    # slightly different sum when other tokens change its rank (slot / co-batch dependent numerics, measured 2026-09-08
    # in a packed 32 x 128 pass: fillers of one prompt in different slots disagree in 33/720 top-1 steps, PCC min
    # 0.986). SORTED_SCATTER_FP32_DEST accumulates the 8 values in fp32 and rounds once (layout independent).
    out = ttnn.matmul(
        onehot,
        ttnn.reshape(down, (1, 1, E * cap, H)),
        transpose_a=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=activation_dtype,
        core_grid=dense_core_grid,
        compute_kernel_config=_SCATTER_COMPUTE_KERNEL_CONFIG
        if SORTED_SCATTER_FP32_DEST
        else _DENSE_COMPUTE_KERNEL_CONFIG,
    )
    onehot.deallocate(True)
    down.deallocate(True)

    # ---- hot experts: dense over the whole split ----
    if hot_ids:
        n_hot = len(hot_ids)
        kd_rows, kd_cols = weights.down_proj_padded.shape[2], weights.down_proj_padded.shape[3]
        # Phase 3: the hot down as ONE K-concatenated minimal_matmul over bf16 GLU pieces (decided here because the
        # GLU must then pack bf16); None -> the perf-p1 forms below.
        kconcat_config = _hot_down_kconcat_minimal_config(
            program_config, dense_core_grid, split_len, n_hot * kd_rows, kd_cols
        )
        if HOT_EXPERTS_PER_EXPERT_LINEAR:
            # One linear per hot expert over the whole split (the [1, 1, H, 2Ip] weight slice is taken and freed on
            # demand; minimal_matmul blocking or the full-grid auto 2D matmul, _expert_linear), stacked along the
            # expert dim -> [1, n_hot, split, 2Ip]. Replaces ttnn.repeat of the activations x n_hot (120 MB written
            # for 15 experts at 1024 tokens) + one batched matmul with one expert per core (22 TFLOP/s).
            gu_config = _dense_expert_gate_up_config(program_config, dense_core_grid, split_len, H, 2 * ip)
            gu_list = []
            for e in hot_ids:
                w_e = _expert_slice(weights.gate_up_proj, e, H, 2 * ip)
                gu_list.append(_expert_linear(hidden_4D, w_e, gu_config, dense_core_grid, activation_dtype))
                w_e.deallocate(True)
            if n_hot == 1:
                gu_hot = gu_list[0]
            else:
                gu_hot = ttnn.concat(gu_list, dim=1)
                for t_e in gu_list:
                    t_e.deallocate(True)
        else:
            # On-demand [1, n_hot, H, 2Ip] copy of the hot experts' fused gate/up weights (freed after the matmul).
            w_hot = _concat_expert_slices(weights.gate_up_proj, hot_ids, H, 2 * ip)
            hidden_rep = ttnn.repeat(hidden_4D, ttnn.Shape((1, n_hot, 1, 1)))
            gu_hot = ttnn.matmul(
                hidden_rep,
                w_hot,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=activation_dtype,
                core_grid=dense_core_grid,
                compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
            )
            hidden_rep.deallocate(True)
            w_hot.deallocate(True)
        gate_h = ttnn.slice(gu_hot, [0, 0, 0, 0], [1, n_hot, split_len, ip])
        up_h = ttnn.slice(gu_hot, [0, 0, 0, ip], [1, n_hot, split_len, 2 * ip])
        gu_hot.deallocate(True)
        # [1, n_hot, split, Ip]; bf16 for the K-concatenated down (its pieces then concat as exact page copies)
        act_h = _glu(gate_h, up_h, config, dtype=ttnn.bfloat16 if kconcat_config is not None else None)
        gate_h.deallocate(True)
        up_h.deallocate(True)
        # routing weights of the hot experts, [1, n_hot, split, 1]: gather rows of routing^T [E, split] (one op, no
        # per-expert slice program variants) -> [1, n_hot, split] ROW_MAJOR, then tilize as [1, n_hot, 1, split] and
        # transpose the last two dims (tile WH transpose). The row-major reshape to [1, n_hot, split, 1] + tilize it
        # replaces was a 140 us copy per split (ELIDE_ROUTING_COPIES).
        rw_rows = ttnn.embedding(hot_idx_t, routing_t_table, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, n_hot, split]
        if ELIDE_ROUTING_COPIES:
            rw_t = ttnn.to_layout(ttnn.reshape(rw_rows, (1, n_hot, 1, split_len)), ttnn.TILE_LAYOUT)
            rw_hot = ttnn.transpose(rw_t, 2, 3)  # [1, n_hot, split, 1]
            rw_t.deallocate(True)
        else:
            rw_hot = ttnn.to_layout(ttnn.reshape(rw_rows, (1, n_hot, split_len, 1)), ttnn.TILE_LAYOUT)
        rw_rows.deallocate(True)
        act_h = ttnn.mul(act_h, rw_hot, output_tensor=act_h)
        rw_hot.deallocate(True)
        if kconcat_config is not None:
            # sum_e act_e @ down_e as ONE minimal_matmul over the concatenated K: [split, n_hot * Ip] x [n_hot * Ip, H].
            # The bf16 act pieces [1, 1, split, Ip] have 320-B rows, so the last-dim concat is an exact page copy
            # (25 / 28 / 36 us at n_hot 4 / 8 / 15 on P150; bfp8 pieces of 160-B rows take the transpose fallback);
            # the sum over the hot experts moves into the matmul accumulation (no [n_hot, split, H] output, no
            # fast_reduce_nc: 50 / 74 / 117 us vs 109 + 64 / 213 + 117 / 394 + 205, with better PCC).
            if n_hot > 1:
                act_parts = [ttnn.slice(act_h, [0, i, 0, 0], [1, i + 1, split_len, ip]) for i in range(n_hot)]
                act_h.deallocate(True)
                act_cat = ttnn.concat(act_parts, dim=3)  # [1, 1, split, n_hot * Ip]
                for t_e in act_parts:
                    t_e.deallocate(True)
            else:
                act_cat = act_h  # already [1, 1, split, Ip]
            wd_cat = _concat_expert_slices(weights.down_proj_padded, hot_ids, kd_rows, kd_cols, dim=2)
            hot_out = ttnn.experimental.minimal_matmul(  # [1, 1, split, H]
                act_cat,
                wd_cat,
                config=kconcat_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=activation_dtype,
                compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
            )
            act_cat.deallocate(True)
            wd_cat.deallocate(True)
        elif HOT_DOWN_KCONCAT and n_hot > 1:
            # sum_e act_e @ down_e as ONE matmul over the concatenated K: [split, n_hot * Ip] x [n_hot * Ip, H]. The
            # sum over the hot experts moves into the matmul accumulation instead of fast_reduce_nc over n_hot
            # bfloat8_b per-expert outputs ([n_hot, split, H], 60 MB read at 15 x 1024).
            act_parts = [ttnn.slice(act_h, [0, i, 0, 0], [1, i + 1, split_len, ip]) for i in range(n_hot)]
            act_h.deallocate(True)
            act_cat = ttnn.concat(act_parts, dim=3)  # [1, 1, split, n_hot * Ip]
            for t_e in act_parts:
                t_e.deallocate(True)
            wd_cat = _concat_expert_slices(weights.down_proj_padded, hot_ids, kd_rows, kd_cols, dim=2)
            hot_out = _hot_down_kconcat_matmul(act_cat, wd_cat, activation_dtype, dense_core_grid)  # [1, 1, split, H]
            act_cat.deallocate(True)
            wd_cat.deallocate(True)
        else:
            # On-demand [1, n_hot, Ip, H] copy of the hot experts' (K-padded) down weights.
            wd_hot = _concat_expert_slices(weights.down_proj_padded, hot_ids, kd_rows, kd_cols)
            down_h = ttnn.matmul(
                act_h,
                wd_hot,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=activation_dtype,
                core_grid=dense_core_grid,
                compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
            )
            act_h.deallocate(True)
            wd_hot.deallocate(True)
            hot_out = reduce_experts(down_h)  # [1, 1, split, H]
            down_h.deallocate(True)
        out = ttnn.add(out, hot_out, output_tensor=out)
        hot_out.deallocate(True)
        hot_idx_t.deallocate(True)
        routing_t.deallocate(True)
    table.deallocate(True)  # releases the split (view) or the bf16 copy
    return out


# Hot-group K-concatenated down matmul: 2D multicast grid (cores) with the whole per-expert K (Ip = 5 tiles) as the
# in0 block; None -> ttnn's auto config on dense_core_grid (A/B switch).
HOT_DOWN_KCONCAT_CORES = (8, 8)


def _hot_down_kconcat_config(cores, split_len, k, n):
    """MatmulMultiCoreReuseMultiCastProgramConfig for ``[1, 1, split, k] x [1, 1, k, n]`` on ``cores`` (x, y): the
    output tiles split evenly over the grid (per_core_M = Mt / y, per_core_N = Nt / x), in0_block_w = the largest
    divisor of Kt <= 5 (Kt = n_hot x 5), out_subblock 1 x (widest divisor of per_core_N <= 8). None when the shapes
    do not divide or ``cores`` is None / has a zero (caller falls back to the auto config)."""
    if cores is None or min(cores) < 1:
        return None
    core_x, core_y = cores
    Mt, Kt, Nt = split_len // 32, k // 32, n // 32
    if split_len % 32 or k % 32 or n % 32 or Mt % core_y or Nt % core_x:
        return None
    per_core_M, per_core_N = Mt // core_y, Nt // core_x
    in0_block_w = next(d for d in (5, 4, 3, 2, 1) if Kt % d == 0)
    out_subblock_w = next(d for d in (8, 4, 2, 1) if per_core_N % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )


def _hot_down_kconcat_matmul(act_cat, wd_cat, activation_dtype, dense_core_grid, cores=None):
    """``act_cat`` [1, 1, split, n_hot * Ip] x ``wd_cat`` [1, 1, n_hot * Ip, H] -> [1, 1, split, H] (= the sum of the
    hot experts' down projections), with the 2D config of ``_hot_down_kconcat_config`` when the grid fits (cores
    default HOT_DOWN_KCONCAT_CORES capped to the compute grid) else ttnn's auto config on ``dense_core_grid``."""
    if cores is None and HOT_DOWN_KCONCAT_CORES is not None:
        cores = (min(HOT_DOWN_KCONCAT_CORES[0], dense_core_grid.x), min(HOT_DOWN_KCONCAT_CORES[1], dense_core_grid.y))
    cfg = _hot_down_kconcat_config(cores, act_cat.shape[2], act_cat.shape[3], wd_cat.shape[3])
    placement = {"program_config": cfg} if cfg is not None else {"core_grid": dense_core_grid}
    return ttnn.matmul(
        act_cat,
        wd_cat,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=activation_dtype,
        compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        **placement,
    )


def _eye(weights, n, _unused=None):
    """Cached [n, n] bf16 identity on device (one-hot table for the sorted path's scatter matmul). Persistent: built
    for the sorted split lengths by _ensure_prefill_helpers on the first prefill call, before any trace capture."""
    tables = weights.eye_tables
    if tables is None:
        tables = {}
        object.__setattr__(weights, "eye_tables", tables)
    if n not in tables:
        tables[n] = ttnn.from_torch(
            torch.eye(n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=weights.gate_up_proj.device()
        )
    return tables[n]


def _bmm_config(core_grid, mt, kt, nt):
    """MatmulMultiCoreReuseProgramConfig with one [mt x nt]-tile output block per core (batched matmul, one batch
    entry per core). Phase 3 (tests/perf/test_prefill_matmul_candidates.py): the all-expert gate|up bmm
    [1, E, S, 4096] x [1, E, 4096, 320] runs 740 / 748 / 763 / 808 us at S = 32 / 64 / 96 / 128 with this config
    (in0_block_w 4, subblock 1x2) -- flat in S, i.e. bound by streaming the 178 MB of expert weights per launch --
    and no in0_block_w 8-32 / Mt x 2 / 1 x 5 / 8x8-grid variant beats it (751-1033 us, PCC falls with in0_block_w
    >= 16), so it is kept as is; fewer expert visits per token (packed multi-user prefill) or a different kernel are
    the levers for that op."""
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        in0_block_w=next(d for d in (6, 5, 4, 3, 2, 1) if kt % d == 0),
        out_subblock_h=1,
        out_subblock_w=next(d for d in (8, 6, 4, 3, 2, 1) if nt % d == 0),
        per_core_M=mt,
        per_core_N=nt,
    )


# bf16 activations x bfloat8_b weights: HiFi2 keeps full bf8 precision; L1 accumulation in the packer.
# (The Wormhole config class is accepted on Blackhole; validated on P150x8.)
_DENSE_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)
# fp32 destination accumulation for the sorted path's one-hot scatter matmul (SORTED_SCATTER_FP32_DEST, env
# SOLAR_OPEN_SORTED_SCATTER_FP32, default 1): the per-token sum over its expert slots is accumulated in fp32 and rounded
# once, so it no longer depends on where the slots fall relative to the K blocks (see _sorted_moe_forward). Measured
# 2026-09-08 (phase-3a verify): teacher-forced b32 through a 32 x 128 packed pass (T = 4096 sorted MoE) vs the bf16 HF
# reference 0.9219 / 0.9602 / 0.9211 / 0.97775 / 0.99014 / KL 0.0382 -> 0.9336 / 0.9690 / 0.9180 / 0.98195 / 0.99266 /
# 0.0304 and the 8 slot copies of a prompt bit-identical again (1792/1792 vs 1756/1792); real layer 0 prefill_1024 mlp /
# decoder PCC unchanged (0.999913 / 0.999853); TTFT 1K 416 vs 388-401 ms, 8K 2920 vs 2907-2927 (neutral). `0` restores
# the phase-2 bf16-destination form.
_SCATTER_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
SORTED_SCATTER_FP32_DEST = os.getenv("SOLAR_OPEN_SORTED_SCATTER_FP32", "1") == "1"


def _ensure_dense_weights(weights: ExpertWeights):
    """One-time, device-side preparation for the dense prefill path: down_proj with K zero-padded to the tile multiple
    that the GLU output carries (dense matmul checks logical K; the padded activation columns are exactly zero).
    When the per-device intermediate is already tile aligned (Solar-Open: 160 = 5 tiles) this is an alias of
    down_proj and costs no memory. Per-expert weight slices are NOT cached: the dense paths take them on demand."""
    pad_k = weights.intermediate_padded_per_device - weights.intermediate_size_per_device
    down_padded = (
        ttnn.pad(weights.down_proj, padding=[(0, 0), (0, 0), (0, pad_k), (0, 0)], value=0.0)
        if pad_k > 0
        else weights.down_proj
    )
    object.__setattr__(weights, "down_proj_padded", down_padded)  # ExpertWeights is a frozen dataclass


def _sorted_split_lens(program_config: ProgramConfig, num_experts, always_on=0):
    """Split lengths the host-planned sorted path can run for -- the split size of every chunk length
    (``base_down_split_size``; its halved long-context variant only when ``sequence_chunk_size`` exceeds 32K) that
    ``moe_prefill_path`` routes to MOE_PATH_HOST_PLANNED -- i.e. the one-hot identities to prebuild. Pure."""
    chunk = program_config.sequence_chunk_size
    candidates = {min(program_config.get_down_split_size(n), chunk) for n in (min(chunk, 32 * 1024), chunk)}
    return sorted(
        s for s in candidates if moe_prefill_path(s, program_config, num_experts, always_on) == MOE_PATH_HOST_PLANNED
    )


def _ensure_prefill_helpers(weights: ExpertWeights, config: ExpertConfig, program_config: ProgramConfig):
    """Build the dense path's PERSISTENT device helpers on the first prefill call of ANY length -- the eager compile
    pass of every flow -- instead of lazily at the first split that needs them: ``down_proj_padded`` (an alias for
    Solar's tile-aligned Ip, see _ensure_dense_weights) and the ``[split, split]`` bf16 one-hot identities of the
    sorted path's split lengths (2 MiB per layer per device at 1024 tokens). A persistent buffer allocated AFTER a
    trace was captured -- e.g. by the first eager 1K prefill following the 128-token / decode captures -- can land in
    that trace's freed scratch and be clobbered by a later replay (tt-metal #52176); the harness's eager pre-compile
    of every length only mitigates this. Idempotent (dict lookups after the first call)."""
    if weights.down_proj_padded is None:
        _ensure_dense_weights(weights)
    for split_len in _sorted_split_lens(program_config, config.num_experts, weights.num_always_on_experts):
        _eye(weights, split_len)


def _dense_bmm_config(core_grid, split_len, weights: ExpertWeights, max_tokens):
    """One-launch batched matmul config for short splits (one expert's whole [split, 2Ip] output per core), or None
    when the split is too long for the per-core block to fit in L1 (then the sorted / per-expert paths are used)."""
    if split_len > max_tokens:
        return None
    return _bmm_config(
        core_grid, split_len // 32, weights.gate_up_proj.shape[2] // 32, weights.gate_up_proj.shape[3] // 32
    )


def _dense_core_grid(mesh_device, max_width):
    """Core grid for the dense prefill matmuls: the full compute grid height, at most `max_width` wide."""
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=min(grid.x, max_width))


def _group_expert_mask(routing_weights, seq_len, num_experts):
    """[S, E] dense routing weights (0 for unselected experts) -> [1, S/32, 1, E] row-major bf16 mask with 1.0 where
    any token of the 32-token group routes to the expert (the sparse_matmul sparsity layout for a [1, G, 32, K] input).
    """
    groups = seq_len // 32
    grouped = ttnn.reshape(routing_weights, (1, groups, 32, num_experts))  # tile-aligned view
    used = ttnn.sum(grouped, dim=2, keepdim=True)  # [1, G, 1, E], > 0 iff some token in the group uses e
    mask = ttnn.gt(used, 0.0)
    used.deallocate(True)
    mask_rm = ttnn.to_layout(mask, ttnn.ROW_MAJOR_LAYOUT)
    mask.deallocate(True)
    return mask_rm


def prefill_forward(
    hidden_states,
    routing_weights,
    weights: ExpertWeights,
    config: ExpertConfig,
    mesh_config,
    mesh_device,
    ccl_manager,
    program_config: ProgramConfig,
    prefill_sparsity,
    shared_expert=None,
):
    """
    Prefill forward pass - optimized for sequence processing (seq_len>1).

    Args:
        hidden_states: Input tensor [1, batch, seq_len, hidden_size] (consumed)
        routing_weights: Router output [seq_len, num_experts]
        weights: Expert weights
        config: Expert configuration
        mesh_config: Mesh parallelization config
        mesh_device: TTNN mesh device
        ccl_manager: Communication manager
        program_config: Model-specific program configs
        prefill_sparsity: Cached prefill sparsity mask
        shared_expert: Optional callable evaluated once per <= sequence_chunk_size-token chunk on that chunk's
            input; its per-device partial [1, 1, chunk, hidden] bf16 is added to the routed partial of the chunk
            before the TP all_reduce (see Experts.__call__)

    Returns:
        Expert output [1, batch, seq_len, hidden_size]

    Trace safety: a call may be captured in a trace only when ``is_trace_safe_prefill_len(seq_len, program_config,
    config.num_experts, weights.num_always_on_experts)`` holds (every split on a static, device-only MoE path); the
    host-planned sorted path asserts it is never reached for a trace-safe split length.
    """
    batch_dim = 1
    seq_dim = 2
    batch_size = hidden_states.shape[batch_dim]
    seq_len_global = hidden_states.shape[seq_dim]

    if batch_size != 1:
        raise NotImplementedError(f"Currently only batch_size=1 supported, got {batch_size}")

    if seq_len_global <= 1:
        raise ValueError(
            f"Prefill mode requires seq_len>1, got {seq_len_global}. " f"Use decode mode for single tokens."
        )

    TILE_SIZE = 32
    if seq_len_global % TILE_SIZE != 0:
        raise ValueError(
            f"Prefill seq_len must be divisible by {TILE_SIZE} (TILE_SIZE), "
            f"got {seq_len_global}. Please pad your sequence."
        )

    # Get parallelization config
    mode_config = mesh_config.get_config(Mode.PREFILL)
    ep, sp, tp = mode_config.ep, mode_config.sp, mode_config.tp
    if ep > 1 and shared_expert is not None:
        # The shared partial is added per chunk, i.e. before the EP all_reduce, which would count it once per
        # EP group. Prefill always runs EP=1 in the shipped mesh configs; fail loudly rather than silently scale.
        raise NotImplementedError(f"shared_expert with prefill EP={ep} > 1 is not supported")
    if ep > 1 and weights.num_always_on_experts:
        raise NotImplementedError("always-on expert slots need EP=1 (E % ep == 0 is assumed by the EP paths)")
    if shared_expert is not None and weights.num_always_on_experts:
        raise ValueError("the shared expert is fused as an always-on slot of these experts; do not pass shared_expert")

    # Reshard for sequence parallelism if needed
    if sp > 1:
        hidden_states, routing_weights = _reshard_for_sequence_parallel(
            hidden_states, routing_weights, mesh_config, ccl_manager
        )

    # Chunk processing for very long sequences
    chunk_size = program_config.sequence_chunk_size
    if hidden_states.shape[seq_dim] > chunk_size:
        hidden_states_chunks = ttnn.split(hidden_states, chunk_size, dim=seq_dim)
        hidden_states.deallocate(True)
        routing_weights_chunks = ttnn.split(routing_weights, chunk_size, dim=0)
        routing_weights.deallocate(True)
    else:
        hidden_states_chunks = [hidden_states]
        routing_weights_chunks = [routing_weights]

    dense_core_grid = _dense_core_grid(mesh_device, program_config.dense_grid_max_width)

    # Process each chunk and stream-concatenate to reduce peak DRAM usage.
    next_states_acc = None
    for hidden_chunk, routing_chunk in zip(hidden_states_chunks, routing_weights_chunks):
        # Shared expert on the chunk BEFORE the chunk is consumed: [1, 1, chunk, H] bf16 partial (<= 32 MiB at
        # chunk 4096 / H 4096), added in place to the chunk's routed partial below.
        shared = shared_expert(hidden_chunk) if shared_expert is not None else None
        next_states = _process_prefill_chunk(
            hidden_chunk,
            routing_chunk,
            weights,
            config,
            prefill_sparsity,
            program_config,
            ep,
            tp,
            dense_core_grid=dense_core_grid,
        )
        if shared is not None:
            next_states = ttnn.add(next_states, shared, output_tensor=next_states)  # bfloat8_b += bf16, in place
            shared.deallocate(True)
        if next_states_acc is None:
            next_states_acc = next_states
        else:
            next_states_concat = ttnn.concat([next_states_acc, next_states], dim=2)
            next_states_acc.deallocate(True)
            next_states.deallocate(True)
            next_states_acc = next_states_concat
        hidden_chunk.deallocate(True)
        routing_chunk.deallocate(True)
    next_states = next_states_acc

    # Expert parallel communication
    if ep > 1:
        next_states = apply_expert_parallel_allreduce(next_states, mesh_config, ccl_manager)

    # Tensor parallel communication (the single CCL of the MoE: sums the routed + shared partials of all TP devices)
    if tp > 1:
        next_states = apply_tensor_parallel_allreduce(
            next_states,
            mesh_config,
            mesh_device,
            seq_len_global,
            ccl_manager,
        )

    # Sequence parallel all-gather
    if sp > 1:
        next_states = apply_sequence_parallel_allgather(next_states, mesh_config, ccl_manager)

    # Final reshape
    next_states = ttnn.reshape(
        next_states,
        (1, batch_size, seq_len_global, config.hidden_size),
        (1, batch_size, max(32, seq_len_global), config.hidden_size),
    )

    return next_states
