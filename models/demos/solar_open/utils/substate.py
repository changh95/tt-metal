# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for working with nested state dictionaries.

This module provides helper functions to extract sub-dictionaries from state dicts,
which is useful for loading layer-specific weights and configurations.

Plain dicts take the comprehension fast path. A mapping that implements ``substate`` / ``has_substate`` itself (the
phase-2 ``utils.streaming_loader.LazyStateDict``) is delegated to, so its prefix views never materialise a tensor.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def substate(state: Mapping[str, torch.Tensor], key: str) -> Mapping[str, torch.Tensor]:
    """
    Extract a sub-dictionary from a state dict based on a key prefix.

    Args:
        state: The source state dictionary
        key: The prefix key to filter by (e.g., "q_proj" to get all "q_proj.*" entries)

    Returns:
        A dictionary with the prefix removed from keys

    Example:
        >>> state = {"q_proj.weight": tensor1, "q_proj.bias": tensor2, "k_proj.weight": tensor3}
        >>> substate(state, "q_proj")
        {"weight": tensor1, "bias": tensor2}
    """
    if hasattr(state, "substate"):  # LazyStateDict and its views: a lazy prefix view, no tensor is read
        return state.substate(key)
    prefix = f"{key}."
    prefix_len = len(prefix)

    return {k[prefix_len:]: v for k, v in state.items() if k.startswith(prefix)}


def has_substate(state: Mapping[str, torch.Tensor], key: str) -> bool:
    """
    Check if a state dict contains any keys with the given prefix.

    Args:
        state: The state dictionary to check
        key: The prefix to look for

    Returns:
        True if any key starts with "{key}.", False otherwise
    """
    if hasattr(state, "has_substate"):
        return state.has_substate(key)
    prefix = f"{key}."

    return any(k.startswith(prefix) for k in state)


def indexed_substates(state: Mapping[str, torch.Tensor], key: str) -> list[Mapping[str, torch.Tensor]]:
    """
    Extract a list of indexed sub-states (e.g., "layer.0", "layer.1", ...).

    Args:
        state: The source state dictionary
        key: The base prefix (e.g., "layer" to get "layer.0", "layer.1", etc.)

    Returns:
        A list of sub-state dictionaries, one for each indexed entry

    Example:
        >>> state = {"layer.0.weight": t1, "layer.0.bias": t2, "layer.1.weight": t3}
        >>> indexed_substates(state, "layer")
        [{"weight": t1, "bias": t2}, {"weight": t3}]
    """
    result = []
    for i in itertools.count():
        s = substate(state, f"{key}.{i}")
        if not s:
            return result
        result.append(s)

    return []
