# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Solar-Open MoE MLP: router + routed experts + always-on shared expert, ONE tensor-parallel all_reduce per call.

Mirrors ``SolarOpenMoE.forward`` (``experts(x, idx, w) + shared_experts(x)``, both on the same post-attention-norm
input).  The shared expert's per-device partial is added inside the routed experts, before their all_reduce
(``Experts.__call__(..., shared_expert=...)``), so the block still issues a single CCL.
"""

import functools

import ttnn
from models.demos.solar_open.config import MoEOptions
from models.demos.solar_open.tt.expert_configs import solar_open_program_config
from models.demos.solar_open.utils.general_utils import get_cache_file_name
from models.demos.solar_open.utils.substate import substate

from .experts import ExpertConfig, Experts
from .shared_expert import SharedExpert
from .topk import TopKRouter


class MLP:
    """MoE block of one decoder layer.

    Sub-modules (and the ``state_dict`` / cache sub-paths they own): ``router`` (``gate``), ``experts``
    (``experts``) and ``shared_expert`` (``shared_experts``; None when the config has no shared experts).
    """

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
        tokens_per_device=32,
        moe_options=None,
    ):
        """
        Args:
            mesh_device: TTNN mesh device
            hf_config: ``SolarOpenConfig``
            state_dict: ``substate(layer_state_dict, "mlp")``; ``{}`` loads every tensor from the cache
            ccl_manager: CCL manager used by the experts' all_reduce
            dtype: activation dtype of the surrounding model (the MoE path emits bfp8 regardless; kept for the
                DecoderLayer contract)
            tensor_cache_path: cache directory of this block (``<layer>/mlp``)
            mesh_config: ``MeshConfig`` (TP over the mesh columns)
            tokens_per_device: decode batch per device (the router prebuilds its bias tile for it)
            moe_options: ``MoEOptions``; None selects the Solar-Open defaults
        """
        assert mesh_config is not None, "MLP requires a MeshConfig (Model builds a default one)"
        options = moe_options or MoEOptions()

        # The router and the fused SiLU-GLU only implement the Solar-Open configuration; fail loudly otherwise.
        assert hf_config.hidden_act == "silu", f"unsupported hidden_act {hf_config.hidden_act!r} (silu only)"
        assert (
            getattr(hf_config, "n_group", 1) == 1 and getattr(hf_config, "topk_group", 1) == 1
        ), "the router implements the single-group (n_group == topk_group == 1) selection rule only"
        assert getattr(hf_config, "norm_topk_prob", True), "the fused router always normalises the top-k weights"

        self.router = TopKRouter(
            mesh_device,
            hf_config,
            substate(state_dict, "gate"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "gate"),
            tokens_per_device=tokens_per_device,
            moe_options=options,
        )

        expert_config = ExpertConfig(
            intermediate_size=hf_config.moe_intermediate_size,  # 1280; hf_config.intermediate_size (10240) is unused
            num_experts=hf_config.num_local_experts,
            hidden_size=hf_config.hidden_size,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            activation=hf_config.hidden_act,
        )
        self.experts = Experts(
            mesh_device=mesh_device,
            config=expert_config,
            state_dict=substate(state_dict, "experts"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=solar_open_program_config(mesh_device),
            weight_dtype=options.expert_dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "experts"),
        )

        self.shared_expert = (
            SharedExpert(
                mesh_device,
                hf_config,
                substate(state_dict, "shared_experts"),
                mesh_config,
                dtype=options.shared_expert_dtype,
                tensor_cache_path=get_cache_file_name(tensor_cache_path, "shared_experts"),
            )
            if getattr(hf_config, "n_shared_experts", 0)
            else None
        )

    def __call__(self, hidden_states, is_decode):
        """Route -> (routed experts + shared expert) -> one TP all_reduce.

        Args:
            hidden_states: ``[1, 1, T, H]`` post-attention-norm hidden states, replicated over the TP axis
                (decode: T = users per device, 1..32; prefill: T = seq_len, a multiple of 32)
            is_decode: decode (True) or prefill (False) mode

        Returns:
            ``[1, 1, T, H]`` bfp8, all-reduced: routed-expert output plus the unweighted shared-expert output
        """
        _, routing_weights = self.router(hidden_states, is_decode=is_decode)  # dense [T, E] bf16 TILE
        shared_expert = (
            functools.partial(self.shared_expert, is_decode=is_decode) if self.shared_expert is not None else None
        )
        return self.experts(
            hidden_states,
            topk_expert_weights=routing_weights,
            is_decode=is_decode,
            shared_expert=shared_expert,
        )
