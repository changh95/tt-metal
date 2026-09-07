# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Solar-Open always-on shared expert.

``SolarOpenMoE.forward`` returns ``experts(x, idx, w) + shared_experts(x)``: the shared expert is a bias-free
``SolarOpenMLP`` with intermediate size ``moe_intermediate_size * n_shared_experts`` (1280), applied unweighted to the
same post-attention-norm input the router sees.  On the TP mesh this module shards that MLP over its intermediate
dimension exactly like the routed experts (160 columns = 5 tiles per device at TP=8), so its down projection yields a
per-device PARTIAL sum and needs no CCL of its own: the routed experts add the partial to their own pre-all_reduce
partial (``Experts.__call__(..., shared_expert=...)``) and the MoE block's single TP all_reduce reduces both at once.
"""

import ttnn
from models.demos.solar_open.utils.general_utils import get_cache_file_name

from .experts.operations import apply_glu


class SharedExpert:
    """``down(silu(gate(x)) * up(x))``, bias-free, TP-sharded over the intermediate dimension.

    Per device: ``w_gate``/``w_up`` ``[1, 1, H, I/tp]`` (column-parallel) and ``w_down`` ``[1, 1, I/tp, H]``
    (row-parallel).  ``__call__`` runs 4 launches (2 linears, one fused SiLU-GLU multiply, the down linear) and returns
    this device's partial over its ``I/tp`` intermediate columns as a bf16 tensor -- the dtype the experts' in-place
    ``ttnn.add(next_states_bfp8, shared_bf16, output_tensor=next_states_bfp8)`` was validated with.

    Weight cache stems (under ``<layer>/mlp/shared_experts``): ``gate_proj_tp{tp}``, ``up_proj_tp{tp}``,
    ``down_proj_tp{tp}``.
    """

    def __init__(self, mesh_device, hf_config, state_dict, mesh_config, dtype=ttnn.bfloat8_b, tensor_cache_path=None):
        """
        Args:
            mesh_device: TTNN mesh device
            hf_config: ``SolarOpenConfig`` (``hidden_size``, ``moe_intermediate_size``, ``n_shared_experts``,
                ``hidden_act``)
            state_dict: ``substate(mlp_state_dict, "shared_experts")`` = ``gate_proj.weight [I, H]``,
                ``up_proj.weight [I, H]``, ``down_proj.weight [H, I]``; ``{}`` loads every tensor from the cache
            mesh_config: ``MeshConfig`` (provides ``tp`` and the column-/row-parallel mesh mappers)
            dtype: weight dtype (``MoEOptions.shared_expert_dtype``; the output is always bf16)
            tensor_cache_path: cache directory for this module's weights (None disables caching)
        """
        activation = getattr(hf_config, "hidden_act", "silu")
        assert activation == "silu", f"SharedExpert implements silu only, hf_config.hidden_act={activation!r}"
        hidden_size = hf_config.hidden_size
        intermediate_size = hf_config.moe_intermediate_size * getattr(hf_config, "n_shared_experts", 1)
        tp = mesh_config.tp
        assert intermediate_size > 0 and intermediate_size % (tp * ttnn.TILE_SIZE) == 0, (
            f"shared-expert intermediate {intermediate_size} must split into tile-aligned slices over TP={tp} "
            f"(Solar-Open: 1280 / 8 = 160 = 5 tiles)"
        )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_size_per_device = intermediate_size // tp
        self.tp = tp

        # nn.Linear stores [out, in]; the device linears consume [in, out] as [1, 1, K, N].
        if state_dict:
            gate = state_dict["gate_proj.weight"].transpose(0, 1).reshape(1, 1, hidden_size, intermediate_size)
            up = state_dict["up_proj.weight"].transpose(0, 1).reshape(1, 1, hidden_size, intermediate_size)
            down = state_dict["down_proj.weight"].transpose(0, 1).reshape(1, 1, intermediate_size, hidden_size)
        else:
            gate = up = down = None

        column_parallel = mesh_config.column_parallel(mesh_device)  # shard the last dim (N) over TP
        row_parallel = mesh_config.row_parallel(mesh_device)  # shard the second-to-last dim (K) over TP

        def load(name, torch_weight, mesh_mapper):
            return ttnn.as_tensor(
                torch_weight,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                mesh_mapper=mesh_mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"{name}_tp{tp}"),
            )

        self.w_gate = load("gate_proj", gate, column_parallel)  # per device [1, 1, H, I/tp]
        self.w_up = load("up_proj", up, column_parallel)  # per device [1, 1, H, I/tp]
        self.w_down = load("down_proj", down, row_parallel)  # per device [1, 1, I/tp, H]

        # bfp8 weights (default): the routed experts' dense-prefill class of config (HiFi2 keeps the full bf16 x bfp8
        # product, bf16 accumulation, L1 packer acc). bf16 weights (SOLAR_OPEN_SHARED_EXPERT_DTYPE=bf16): HiFi4 with
        # fp32 accumulation, otherwise HiFi2 truncates the second bf16 operand and the extra weight precision that
        # option is meant to buy is lost in the matmul (the three linears are tiny: N or K = 160 per device).
        weights_are_bf16 = dtype == ttnn.bfloat16
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4 if weights_are_bf16 else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=weights_are_bf16,
            packer_l1_acc=True,
        )

    def __call__(self, x, is_decode=None):
        """Compute this device's partial of the shared-expert output.

        Args:
            x: ``[1, 1, T, H]`` bf16/bfp8 TILE, replicated over the TP axis -- the post-attention-norm hidden states
                (decode: T <= 32 rows, already padded to 32 by the batched experts, ``[1, 1, 1, H]`` for a single
                user; prefill: T = one expert chunk <= 4096 tokens).  Not consumed.
            is_decode: selects L1 (decode) or DRAM (prefill) for the activations; inferred from the row count when
                None.

        Returns:
            ``[1, 1, T, H]`` bf16 TILE interleaved: the partial sum over this device's ``I/tp`` intermediate columns
            (no CCL; the routed experts add it before their single TP all_reduce).
        """
        if is_decode is None:
            is_decode = x.shape[-2] <= ttnn.TILE_SIZE
        memory_config = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG

        gate = ttnn.linear(
            x,
            self.w_gate,
            dtype=ttnn.bfloat16,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )  # [1, 1, T, I/tp]
        up = ttnn.linear(
            x,
            self.w_up,
            dtype=ttnn.bfloat16,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )  # [1, 1, T, I/tp]
        activated = apply_glu(gate, up)  # up * silu(gate), one binary op
        gate.deallocate(True)
        up.deallocate(True)

        partial = ttnn.linear(
            activated,
            self.w_down,
            dtype=ttnn.bfloat16,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config,
        )  # [1, 1, T, H], partial over this device's I/tp columns
        activated.deallocate(True)
        return partial
