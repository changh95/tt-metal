import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

import random
from loguru import logger
from typing import Optional, Tuple, Union

from transformers import WhisperConfig

from python_api_testing.models.whisper.whisper_common import torch2tt_tensor, tt2torch_tensor, create_padded_tensor, create_unpadded_tensor
from python_api_testing.models.whisper.whisper_decoder_layer import TtWhisperDecoderLayer
from python_api_testing.fused_ops.linear import Linear as TtLinear
from python_api_testing.fused_ops.layernorm import Layernorm as TtLayernorm

from libs import tt_lib as ttm

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    TODO: Implemented in PyTorch for now.
    """
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    TODO: Implemented in PyTorch for now.
    """
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

class WhisperPositionalEmbedding(nn.Embedding):
    """
    TODO: Implemented in PyTorch for now. And Initialized directly from HF reference model.
    """
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__(num_positions, embedding_dim)

    def forward(self, input_ids, past_key_values_length=0):
        return self.weight[past_key_values_length : past_key_values_length + input_ids.shape[-1]]

@dataclass
class TtWhisperDecoderOutput():
    """
    TT implementation of HF Base class for model's outputs
    that may also contain a past key/values (to speed up sequential decoding).
    """

    last_hidden_state: ttm.tensor.Tensor = None
    past_key_values: Optional[Tuple[Tuple[ttm.tensor.Tensor]]] = None
    hidden_states: Optional[Tuple[ttm.tensor.Tensor]] = None
    attentions: Optional[Tuple[ttm.tensor.Tensor]] = None
    cross_attentions: Optional[Tuple[ttm.tensor.Tensor]] = None


class TtWhisperDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    [`WhisperDecoderLayer`]

    Args:
        reference_model: WhisperModel
        device: device: ttm.device.Device
        config: WhisperConfig
    """

    def __init__(
        self,
        state_dict,
        base_address,
        device,
        config: WhisperConfig,
        already_padded_inputs: bool = False
    ):
        super().__init__()

        self.config = config
        self.device = device
        self.state_dict = state_dict

        self.already_padded_inputs = already_padded_inputs

        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        """
        TODO: Embeddings. Implemented from PyTorch for now.
        """
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_tokens.weight = nn.Parameter(state_dict[f"{base_address}.embed_tokens.weight"])

        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)
        self.embed_positions.weight = nn.Parameter(state_dict[f"{base_address}.embed_positions.weight"])

        self.layers = nn.ModuleList([
            TtWhisperDecoderLayer(
                base_address = f"{base_address}.layers.{ind}",
                state_dict = state_dict,
                device = self.device,
                embed_dim = config.d_model,
                num_heads = config.decoder_attention_heads,
                decoder_ffn_dim = config.decoder_ffn_dim,
                config = self.config,
            )
            for ind in range(config.decoder_layers)
        ])

        gamma = self.state_dict[f"{base_address}.layer_norm.weight"]
        gamma = create_padded_tensor(list(gamma.shape), gamma, [1, 1, 32, gamma.shape[-1]], 0, ttm.device.GetHost())
        beta = self.state_dict[f"{base_address}.layer_norm.bias"]
        beta = create_padded_tensor(list(beta.shape), beta, [1, 1, 32, beta.shape[-1]], 0, ttm.device.GetHost())
        tt_gamma = gamma.data()
        tt_beta = beta.data()

        self.layer_norm = TtLayernorm(tt_gamma, tt_beta, 1e-05, 1, config.d_model, self.device, 1)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        """
        TODO: Implemented in PyTorch for now.
        """
        combined_attention_mask = None

        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids:torch.Tensor = None,
        attention_mask:torch.Tensor = None,
        encoder_hidden_states:ttm.tensor.Tensor = None,
        head_mask:torch.Tensor = None,
        cross_attn_head_mask:torch.Tensor=None,
        past_key_values:Optional[Tuple[ttm.tensor.Tensor]]=None,
        inputs_embeds:torch.Tensor=None,
        use_cache:bool=None,
        output_attentions:bool=None,
        output_hidden_states:bool=None,
        return_dict:bool=None,
    ) -> Union[Tuple[ttm.tensor.Tensor], TtWhisperDecoderOutput]:
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        """PyTorch implementation start"""

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[-1] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)
        else:
            positions = self.embed_positions(inputs_embeds, past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + positions

        """PyTorch implementation end"""

        """TTM implementation"""
        if not self.already_padded_inputs:
            # Add padding
            output_tensor_shape = encoder_hidden_states.shape()
            while len(output_tensor_shape) < 4:
                output_tensor_shape.insert(0,1)
            output_tensor_shape[-2] = 1504
            encoder_hidden_states = create_padded_tensor(encoder_hidden_states.shape(), encoder_hidden_states, output_tensor_shape, pad_value=0.0, device=self.device)

        padded_hidden_states = False
        if hidden_states.size(-2) % 32 != 0:
            padded_hidden_states = True

            output_tensor_shape = list(hidden_states.size())
            logger.info(f"Padding Decoder hidden states tensor")

            shape_before_pad = list(hidden_states.size())
            while len(output_tensor_shape) < 4:
                output_tensor_shape.insert(0,1)
            output_tensor_shape[-2] = 32
            hidden_states = create_padded_tensor(list(hidden_states.size()), hidden_states, output_tensor_shape, pad_value=0.0, device=self.device)
        else:
            hidden_states = torch2tt_tensor(hidden_states, self.device)

        if attention_mask is not None:
            if attention_mask.size(-2) % 32 != 0 and attention_mask.size(-1) % 32 != 0:
                logger.info(f"Padding Decoder attention mask tensor")
                output_tensor_shape = list(attention_mask.size())

                while len(output_tensor_shape) < 4:
                    output_tensor_shape.insert(0,1)
                output_tensor_shape[-2] = 32
                output_tensor_shape[-1] = 32
                attention_mask = create_padded_tensor(list(attention_mask.size()), attention_mask, output_tensor_shape, pad_value=0, device=self.device)

            else:
                attention_mask = torch2tt_tensor(attention_mask, self.device)

        # TODO: Dropout not supported for not
        #hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        """TODO: Training not supported for now"""
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            """TODO: Training not supported for now"""
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            """TODO: Training not supported for now"""
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    None,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,  # past_key_value
                )

            else:
                layer_outputs = decoder_layer(
                    hidden_states = hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        H_hidden_states = hidden_states.shape()[-2]
        hidden_states = self.layer_norm(hidden_states,overrideH=H_hidden_states)

        if padded_hidden_states:
            # Unpad hidden states tensor to original size
            input_tensors_shape = [1, *shape_before_pad]
            hidden_states = create_unpadded_tensor(hidden_states, input_tensors_shape)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )

        return TtWhisperDecoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
