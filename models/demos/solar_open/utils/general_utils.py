# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the Solar-Open demo: cache-file naming, fabric link defaults and HuggingFace-config
accessors that hide transformers-version differences (where rope_theta lives, absent layer_types /
sliding_window on SolarOpenConfig).
"""

from models.common.utility_functions import is_blackhole


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def get_default_num_links(mesh_device):
    """Default number of fabric links for CCL ops on the given mesh.

    Blackhole exposes 2 fabric links per device; Wormhole exposes 4. Single-row meshes
    (shape[0] == 1) only need 1 link regardless of arch.
    """
    if mesh_device.shape[0] == 1:
        return 1
    return 2 if is_blackhole() else 4


def resolve_rope_theta(hf_config, default=1_000_000.0) -> float:
    """RoPE base frequency of an HF config, independent of the transformers version.

    transformers 5.x folds ``rope_theta`` into ``hf_config.rope_parameters`` (no top-level attribute), older
    configs expose it top-level; ``SolarOpenConfig.default_theta`` is the class default (1e6). A plain
    ``getattr(hf_config, "rope_theta", <number>)`` silently returns the wrong value on 5.x, hence this helper.
    """
    theta = getattr(hf_config, "rope_theta", None)
    if theta is None:
        theta = (getattr(hf_config, "rope_parameters", None) or {}).get("rope_theta")
    if theta is None:
        theta = getattr(hf_config, "default_theta", None)
    if theta is None:
        theta = default
    return float(theta)


def get_layer_types(hf_config) -> list:
    """Per-layer attention type. SolarOpenConfig has no ``layer_types``: every layer is full attention."""
    return list(getattr(hf_config, "layer_types", None) or ["full_attention"] * hf_config.num_hidden_layers)


def get_sliding_window(hf_config, layer_idx):
    """Sliding-window size for ``layer_idx`` or None (always None for Solar-Open: no sliding-attention layers)."""
    if get_layer_types(hf_config)[layer_idx] != "sliding_attention":
        return None
    return getattr(hf_config, "sliding_window", None)
