# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Remap HuggingFace Qwen3.5-9B state dict to internal format.

Handles:
- Stripping 'model.language_model.' prefix
- Filtering out vision encoder and MTP weights
- Renaming combined in_proj_qkv → qkv_proj (DeltaNet layers; the op uses the fused weight)
- Splitting combined conv1d.weight into separate Q, K, V conv weights (DeltaNet layers)
- Renaming lm_head.weight → output.weight
- Renaming embed_tokens → tok_embeddings
"""
import json
from pathlib import Path
from typing import Dict

import torch

# Layer indices that use full (softmax) attention
FULL_ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}

# DeltaNet QKV split dimensions (used to split the combined conv1d.weight into
# per-stream Q/K/V conv weights — the QKV projection itself stays combined).
# Q: num_key_heads(16) × key_head_dim(128) = 2048
# K: num_key_heads(16) × key_head_dim(128) = 2048
# (V = num_value_heads(32) × value_head_dim(128) = 4096 is the remaining slice)
LINEAR_Q_DIM = 2048
LINEAR_K_DIM = 2048


def remap_qwen36_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap HF Qwen3.5-9B state dict to internal format.

    Args:
        state_dict: Raw HuggingFace state dict loaded from safetensors.

    Returns:
        Remapped state dict with internal naming convention.
    """
    remapped = {}

    for key, tensor in state_dict.items():
        # Filter out vision encoder weights (check original key — no prefix stripping yet)
        if "visual" in key or key.startswith("model.visual"):
            continue
        # Filter out MTP (multi-token prediction) weights (original key)
        if key.startswith("mtp"):
            continue

        # Strip the language-model prefix. Two checkpoint sources produce different
        # prefixes for the same internal weights:
        #   - raw sharded safetensors:           model.language_model.X
        #   - AutoModelForCausalLM.from_pretrained (text-only Qwen3_5ForCausalLM): model.X
        # Strip whichever matches, longest first, so BOTH sources yield identical
        # internal keys. Top-level weights like lm_head.weight have no prefix and are
        # matched against the original `key` below.
        new_key = key
        for prefix in ("model.language_model.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break

        # Rename top-level weights
        if new_key == "embed_tokens.weight":
            remapped["tok_embeddings.weight"] = tensor
            continue
        if key == "lm_head.weight":
            remapped["output.weight"] = tensor
            continue
        # Final norm (model.language_model.norm.weight)
        if new_key == "norm.weight":
            remapped["norm.weight"] = tensor
            continue

        # Handle per-layer weights
        if new_key.startswith("layers."):
            parts = new_key.split(".")
            layer_idx = int(parts[1])
            layer_prefix = f"layers.{layer_idx}"
            sub_key = ".".join(parts[2:])

            # DeltaNet layers: keep ONLY the combined QKV weight. The split q/k/v_proj
            # were dead — the op runs the fused QKV projection from the combined weight
            # (it only read the splits in a fallback reached when qkv_proj_weight is None,
            # which never happens for the 9B).
            if sub_key == "linear_attn.in_proj_qkv.weight":
                remapped[f"{layer_prefix}.linear_attn.qkv_proj.weight"] = tensor  # [8192, 4096]
                continue

            if sub_key == "linear_attn.conv1d.weight":
                conv = tensor  # [8192, 1, 4]
                q_conv = conv[:LINEAR_Q_DIM, :, :]
                k_conv = conv[LINEAR_Q_DIM : LINEAR_Q_DIM + LINEAR_K_DIM, :, :]
                v_conv = conv[LINEAR_Q_DIM + LINEAR_K_DIM :, :, :]
                remapped[f"{layer_prefix}.linear_attn.q_conv.weight"] = q_conv
                remapped[f"{layer_prefix}.linear_attn.k_conv.weight"] = k_conv
                remapped[f"{layer_prefix}.linear_attn.v_conv.weight"] = v_conv
                continue

            # All other keys pass through unchanged
            remapped[new_key] = tensor
            continue

        # Any remaining keys pass through
        remapped[new_key] = tensor

    return remapped


def is_fp8_checkpoint(model_path) -> bool:
    """True when the checkpoint dir holds block-wise FP8 safetensors.

    Detected by a ``*.weight_scale_inv`` entry in the safetensors index (the
    per-block dequant scales that accompany float8_e4m3fn weights). Such a
    checkpoint cannot be loaded via AutoModelForCausalLM here; use
    ``load_qwen36_state_dict_fp8`` instead.
    """
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    except (KeyError, ValueError, OSError):
        return False
    return any(k.endswith(".weight_scale_inv") for k in weight_map)


def load_qwen36_state_dict_fp8(model_path) -> Dict[str, torch.Tensor]:
    """Load Qwen3.5 FP8 weights: block-wise dequant + minimal key remap.

    Produces the SAME internal key scheme as ``remap_qwen36_state_dict`` for the
    shared/simple weights (``layers.N.mlp.*``, ``layers.N.self_attn.*``,
    ``input_layernorm`` / ``post_attention_layernorm``, ``tok_embeddings``,
    ``norm``, ``output``) so ``layer.py``'s substate extraction is unchanged.

    The one deliberate difference vs the single-device remap: GDN ``linear_attn.*``
    projections are kept RAW (fused ``in_proj_qkv``, fused ``conv1d``, plus
    ``in_proj_z`` / ``in_proj_a`` / ``in_proj_b`` / ``out_proj`` / ``A_log`` /
    ``dt_bias`` / ``norm.weight``) — NOT split or renamed — so the tensor-parallel
    GDN weight-prep helpers (prepare_gdn_qkv / prepare_conv_taps) can reorder and
    shard them per device. The TP module loaders branch on this raw layout.
    """
    from safetensors import safe_open

    from models.demos.blackhole.qwen36.tt.tp_common import dequant_fp8_block

    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    file_to_keys: Dict[str, list] = {}
    for key, filename in weight_map.items():
        file_to_keys.setdefault(filename, []).append(key)

    raw: Dict[str, torch.Tensor] = {}
    for filename, keys in file_to_keys.items():
        with safe_open(str(model_path / filename), framework="pt") as sf:
            present = set(sf.keys())
            for key in keys:
                if key in present:
                    raw[key] = sf.get_tensor(key)

    # Dequantize FP8 (skip the scale tensors themselves)
    dequantized: Dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if key.endswith(".weight_scale_inv"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key + "_scale_inv"
            dequantized[key] = (
                dequant_fp8_block(tensor, raw[scale_key]) if scale_key in raw else tensor.to(torch.bfloat16)
            )
        else:
            dequantized[key] = tensor

    state_dict: Dict[str, torch.Tensor] = {}
    for key, tensor in dequantized.items():
        if "visual" in key or key.startswith("mtp"):
            continue
        short = key
        for prefix in ("model.language_model.", "model."):
            if short.startswith(prefix):
                short = short[len(prefix) :]
                break
        if "embed_tokens" in short:
            state_dict["tok_embeddings.weight"] = tensor
        elif key == "lm_head.weight" or short == "lm_head.weight":
            state_dict["output.weight"] = tensor
        else:
            # Everything else (layers.N.mlp.*, self_attn.*, linear_attn.* RAW,
            # input_layernorm/post_attention_layernorm, norm) passes through.
            state_dict[short] = tensor

    return state_dict


# --------------------------------------------------------------------------------------------- MTP head
# The checkpoint's native multi-token-prediction head (config text_config.mtp_num_hidden_layers = 1): ONE full-attention
# Qwen3.5 decoder layer (mtp.layers.0.*) plus the fusion in-projection mtp.fc [dim, 2*dim] and three zero-centered norms
# (mtp.pre_fc_norm_embedding / mtp.pre_fc_norm_hidden / mtp.norm). Both main loaders above DROP every mtp.* key (the
# served text model never runs the head); the speculative-decoding drafter (tt/mtp_head.py) loads them with this helper.
MTP_KEYS = (
    "mtp.fc.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
)


def load_qwen36_mtp_state_dict(model_path, layer_index: int) -> Dict[str, torch.Tensor]:
    """Read ONLY the ``mtp.*`` tensors of a Qwen3.5/3.6/3.8 checkpoint (bf16 safetensors) and remap them so the MTP
    decoder layer can be built by ``Qwen36DecoderLayer(..., layer_num=layer_index)``:

      mtp.layers.0.<sub>  -> layers.{layer_index}.<sub>   (self_attn.*, mlp.*, input_layernorm, post_attention_layernorm)
      mtp.fc.weight, mtp.pre_fc_norm_*.weight, mtp.norm.weight -> unchanged (MTP_KEYS)

    layer_index is a virtual index past the model's real layers (n_layers, i.e. 64 for the 27B) so the per-layer
    tensor cache files (`layers.{layer_index}/...`) never collide with a real layer's. Raises if the checkpoint has no
    MTP head (no mtp.* key) or holds it in a quantized form this loader does not dequantize.
    """
    from safetensors import safe_open

    model_path = Path(model_path)
    with open(model_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    mtp_keys = [k for k in weight_map if k.startswith("mtp.")]
    if not mtp_keys:
        raise KeyError(f"checkpoint {model_path} has no mtp.* tensors (no MTP head)")
    if any(k.endswith(".weight_scale_inv") for k in mtp_keys):
        raise NotImplementedError("quantized (block-FP8) MTP weights are not dequantized by load_qwen36_mtp_state_dict")
    by_file: Dict[str, list] = {}
    for k in mtp_keys:
        by_file.setdefault(weight_map[k], []).append(k)
    out: Dict[str, torch.Tensor] = {}
    for filename, keys in by_file.items():
        with safe_open(str(model_path / filename), framework="pt") as sf:
            for k in keys:
                t = sf.get_tensor(k)
                if k.startswith("mtp.layers.0."):
                    out[f"layers.{layer_index}." + k[len("mtp.layers.0.") :]] = t
                elif k in MTP_KEYS:
                    out[k] = t
                else:
                    raise KeyError(f"unexpected MTP tensor {k} (only one MTP layer, mtp.layers.0, is supported)")
    missing = [k for k in MTP_KEYS if k not in out]
    assert not missing, f"MTP head incomplete: missing {missing}"
    return out
