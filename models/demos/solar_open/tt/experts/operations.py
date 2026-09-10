# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Core expert operations - pure functions for composability."""

import ttnn

from ..ccl import FUSED_SITE_MOE


def apply_glu(gate, up, config_or_activation="silu"):
    """``up * act(gate)`` in ONE binary op (consumes nothing; returns a new tensor).

    Solar-Open uses ``act = silu``, i.e. ``up * gate * sigmoid(gate)``: binary_ng applies the SILU unary
    chain to the gate operand while it is being multiplied (probe on P150: PCC 0.99982 vs torch with
    bfloat8_b inputs). Zero-padded intermediate columns stay exactly 0 (gate = 0 -> silu(gate) = 0).

    Args:
        gate: gate projection output ([..., Ip])
        up: up projection output (same shape as gate)
        config_or_activation: an ExpertConfig (its ``activation`` field is used) or the activation name

    Returns:
        Activated tensor with the shape and dtype of ``up``
    """
    activation = getattr(config_or_activation, "activation", config_or_activation)
    if activation != "silu":
        raise NotImplementedError(f"activation {activation!r} (only silu is implemented for Solar-Open)")
    return ttnn.mul(up, gate, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)])


def apply_routing_weights(expert_output, routing_weights):
    """
    Apply routing weights to expert outputs.

    Args:
        expert_output: Output from experts [batch, num_experts, seq_len, hidden]
        routing_weights: Routing weights [batch, num_experts, seq_len, 1]

    Returns:
        Weighted output
    """
    return ttnn.mul(expert_output, routing_weights, output_tensor=expert_output)


def reduce_experts(expert_output):
    """
    Reduce across expert dimension.

    Args:
        expert_output: [batch, num_experts, seq_len, hidden]

    Returns:
        Reduced output [batch, 1, seq_len, hidden]
    """
    return ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(expert_output, dims=[1]))


def apply_expert_parallel_allreduce(tensor, mesh_config, ccl_manager):
    """Apply expert parallel allreduce communication."""
    tensor_allreduced = ttnn.all_reduce(
        tensor, num_links=ccl_manager.num_links, topology=ttnn.Topology.Ring, cluster_axis=mesh_config.ep_axis
    )
    tensor.deallocate(True)
    return tensor_allreduced


def apply_tensor_parallel_allreduce(tensor, mesh_config, mesh_device, seq_len, ccl_manager):
    """
    Apply tensor parallel allreduce communication (sums the per-device partials over the TP axis).

    Deallocates the input tensor.

    Decode partials (``[1, 1, B <= 32, hidden]`` bfp8 in L1, the fast_reduce_nc output + shared-expert add) take the
    fused single-kernel all-reduce when ``SOLAR_OPEN_DECODE_CCL=fused`` (phase 3e / A2, ``tt/ccl.py``): reshard onto
    the 8x4 width-sharded grid, ``all_reduce_async`` on the MoE site's persistent (buffer, semaphore) pair, reshard
    back to L1 interleaved -- the same output contract as ``ttnn.all_reduce`` (not bit-identical: a different,
    deterministic reduction order). Prefill partials (DRAM, > 32 rows) always run the composite ``ttnn.all_reduce``.

    Args:
        tensor: Input tensor to allreduce (this device's partial sum)
        mesh_config: Mesh configuration (provides tp_axis)
        mesh_device: TTNN mesh device
        seq_len: Sequence length (informational)
        ccl_manager: Communication manager (provides num_links and the fused decode all-reduce)

    Returns:
        Allreduced tensor
    """
    if ccl_manager.fused_decode_applies(tensor):
        return ccl_manager.fused_decode_all_reduce_interleaved(tensor, FUSED_SITE_MOE, cluster_axis=mesh_config.tp_axis)
    tensor_allreduced = ttnn.all_reduce(
        tensor, num_links=ccl_manager.num_links, topology=ttnn.Topology.Ring, cluster_axis=mesh_config.tp_axis
    )
    tensor.deallocate(True)

    return tensor_allreduced


def apply_sequence_parallel_allgather(tensor, mesh_config, ccl_manager):
    """Apply sequence parallel allgather communication."""
    tensor_gathered = ttnn.all_gather(
        tensor, dim=-2, num_links=ccl_manager.num_links, topology=ttnn.Topology.Ring, cluster_axis=mesh_config.sp_axis
    )
    tensor.deallocate(True)
    return tensor_gathered
