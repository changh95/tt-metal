# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Solar-Open decoder layer: pre-norm attention + residual, pre-norm MoE (router + routed experts + shared expert)
+ residual. Mirrors ``SolarOpenDecoderLayer.forward`` in transformers; every layer is full (non-sliding) attention.
"""

import ttnn
from models.demos.solar_open.utils.general_utils import get_cache_file_name, get_layer_types, get_sliding_window
from models.demos.solar_open.utils.substate import substate

from .attention import Attention, AttentionConfig
from .attention_configs import SolarOpenAttentionProgramConfig
from .mlp import MLP
from .rms_norm import RMSNorm


class DecoderLayer:
    """One Solar-Open transformer block on a TP-sharded mesh.

    Sub-modules and the state-dict keys they consume (relative to ``model.layers.{layer_idx}``):
      - ``input_layernorm`` / ``post_attention_layernorm`` -> :class:`RMSNorm` (T5-style, eps ``rms_norm_eps``)
      - ``self_attn`` -> :class:`Attention` (bias-free, sink-free GQA; per device ``num_heads/tp`` q heads and
        ``num_kv_heads/tp`` kv heads; TP all-reduce at the end of o_proj)
      - ``mlp`` -> :class:`MLP` (router + 128 routed experts + 1 shared expert; ONE TP all-reduce per call)

    ``max_seq_len`` sizes the UNPAGED KV cache only (paged runs take the pool size from
    ``paged_attention_config``). ``tokens_per_device`` is the decode batch the router/experts are sized for
    (== ``max_local_batch_size`` on a single-row mesh). ``moe_options`` (``config.MoEOptions``) carries the
    expert/shared-expert dtypes and the router implementation flags; ``None`` selects the Solar defaults.
    """

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        paged_attention_config=None,
        mesh_config=None,
        create_kv_cache=True,
        transformation_mats=None,
        max_seq_len=1024,
        max_local_batch_size=1,
        users_row_sharded=False,
        tokens_per_device=32,
        moe_options=None,
        router_persistent_token_counts=None,
    ):
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "input_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "input_layernorm"),
            mesh_config=mesh_config,
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "post_attention_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "post_attention_layernorm"),
            mesh_config=mesh_config,
        )
        self.mlp = MLP(
            mesh_device,
            hf_config,
            substate(state_dict, "mlp"),
            ccl_manager,
            dtype=dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
            mesh_config=mesh_config,
            tokens_per_device=tokens_per_device,
            moe_options=moe_options,
            router_persistent_token_counts=router_persistent_token_counts,
        )

        # SolarOpenConfig declares no layer_types: get_layer_types synthesizes ["full_attention"] * num_layers.
        # The sliding-window and attention-bias code paths were removed from this tree, so fail loudly if a
        # config ever asks for them instead of silently running the wrong attention.
        self.attention_type = get_layer_types(hf_config)[layer_idx]
        assert (
            self.attention_type == "full_attention"
        ), f"layer {layer_idx}: {self.attention_type!r} requested but sliding-window attention was removed from this tree"
        assert not getattr(hf_config, "attention_bias", False), "attention biases were removed from this tree"

        attention_config = AttentionConfig(
            hidden_size=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            sliding_window=get_sliding_window(hf_config, layer_idx),  # None for every Solar layer
            max_seq_len=max_seq_len,
            max_local_batch_size=max_local_batch_size,
            users_row_sharded=users_row_sharded,
        )

        self.self_attn = Attention(
            mesh_device=mesh_device,
            config=attention_config,
            state_dict=substate(state_dict, "self_attn"),
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=SolarOpenAttentionProgramConfig(),
            layer_idx=layer_idx,
            paged_attention_config=paged_attention_config,
            transformation_mats=transformation_mats,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "self_attn"),
            create_kv_cache=create_kv_cache,
        )
        self.mesh_device = mesh_device

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
        user_id=0,
        batch_size=1,
    ):
        """Run the block on ``hidden_states`` ``[1, 1, tokens, hidden_size]`` (replicated across TP) and return
        a tensor of the same shape. ``position_embeddings`` are the RoPE cos/sin matrices for this call,
        ``position_idx``/``page_table``/``kv_cache`` drive the KV-cache update, and ``is_decode`` selects the
        decode or prefill kernels in attention and the MoE."""
        seqlen = hidden_states.shape[-2]
        if seqlen > 32 * 1024:
            # Reallocate hidden states to prevent memory fragmentation.
            hidden_states = ttnn.move(hidden_states)

        # hidden_states: [1, 1, tokens/num_rows, hidden_size/num_columns]
        # residual: [1, 1, tokens/num_rows, hidden_size/num_columns]
        residual = hidden_states
        hidden_states_post_norm = self.input_layernorm(hidden_states)

        # additional all_gather (cluster_axis=1) to get [1, 1, global_batch//num_rows, hidden_size]
        # hidden_states_post_norm: [1, 1, tokens/num_rows, hidden_size]
        hidden_states = self.self_attn(
            hidden_states_post_norm,
            rope_mats=position_embeddings,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            user_id=user_id,
            batch_size=batch_size,
        )
        hidden_states_post_norm.deallocate(True)

        # after reduce scatter at end of attn: [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = self._residual_add(residual, hidden_states)
        residual = hidden_states
        hidden_states_post_norm = self.post_attention_layernorm(hidden_states)
        # another all_gather (cluster_axis=1) to get [1, 1, global_batch//num_rows, hidden_size]

        hidden_states = self.mlp(hidden_states_post_norm, is_decode=is_decode)
        hidden_states_post_norm.deallocate(True)

        # TODO: replace all_reduce at end of MLP with reduce_scatter so we get [1, 1, global_batch//num_rows, hidden_size/num_columns]
        hidden_states = self._residual_add(residual, hidden_states)

        return hidden_states

    @staticmethod
    def _residual_add(residual, branch):
        """``residual + branch`` as a NEW bf16 tensor (both inputs are consumed).

        The residual stream is kept in bf16 like the HF reference (each element has its own exponent). The attention
        and MoE branches emit bfloat8_b; writing the sum into that bfp8 tensor (the source demo's in-place add) would
        re-quantise the whole stream to a block float with one exponent per 16 elements at every layer, which on real
        checkpoints wipes out the small channels next to the massive-activation channels of the residual (48 times).
        The sum lands where the branch output lives (L1 in decode, DRAM in prefill); [1, 1, T, 4096] bf16 is 32 MiB at
        T=4096, negligible next to the expert weights.
        """
        out = ttnn.add(residual, branch, dtype=ttnn.bfloat16, memory_config=branch.memory_config())
        residual.deallocate(True)
        branch.deallocate(True)
        return out
