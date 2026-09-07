# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Phase-2 streaming state-dict loader for Solar-Open-100B (DESIGN.md section 4.16) - DESIGN / STUB ONLY.

Phase 1 loads the whole checkpoint with ``AutoModelForCausalLM.from_pretrained(dtype=torch.bfloat16)`` (~205 GB
host RSS, one-time thanks to the ttnn weight cache). This module reserves the phase-2 alternative: a lazy
``Mapping`` over the 42 bf16 safetensors shards that materialises one layer at a time (~4.2 GB peak instead of
205 GB) while presenting exactly the contract-C1 key set to every module, so nothing outside
``ModelArgs.load_state_dict`` and a 3-line hook in ``utils/substate.py`` has to change.

Planned design (not implemented here):
  * ``weight_map`` from ``<snapshot>/model.safetensors.index.json`` gives the shard of every on-disk key.
  * The 384 per-expert keys ``model.layers.{l}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`` of layer l
    are replaced by two virtual keys ``model.layers.{l}.mlp.experts.gate_up_proj`` = stack_e(cat([gate_e, up_e], 0))
    -> [128, 2560, 4096] (gate rows first, NOT interleaved) and ``model.layers.{l}.mlp.experts.down_proj`` =
    stack_e(down_e) -> [128, 4096, 1280]; 18,963 on-disk keys become the 603 keys of contract C1.
  * ``q_proj.weight`` / ``k_proj.weight`` are Meta-permuted on access (``load_checkpoints.reverse_permute``,
    head_dim 128); ``e_score_correction_bias`` stays fp32; everything else is returned raw bf16.
  * An LRU of <= 3 open ``safetensors.safe_open`` handles bounds file descriptors while a layer (2 shards) streams.
  * ``substate(key)`` / ``has_substate(key)`` return prefix views without materialising tensors, so
    ``utils/substate.py`` can delegate to them (``if hasattr(state, "substate"): return state.substate(key)``).
  * Acceptance: tensors identical to the ``from_pretrained`` path for layer 0.

Enabled by ``SOLAR_OPEN_STREAMING_LOAD=1`` in ``ModelArgs.load_state_dict``; until phase 2 lands that raises
``NotImplementedError`` so a run never silently falls back to a different loader.
"""

from collections.abc import Mapping

DESIGN_REFERENCE = "DESIGN.md section 4.16 (utils/streaming_loader.py::LazyStateDict, phase 2)"


class LazyStateDict(Mapping):
    """Lazy transformers-5-layout state dict over an HF snapshot directory (phase 2; see module docstring).

    Constructor signature is frozen by the design so the ``ModelArgs.load_state_dict`` call site does not move:
    ``LazyStateDict(snapshot_dir, head_dim=128, num_experts=128, convert_to_meta=True, prefix="")``.
    """

    def __init__(self, snapshot_dir, head_dim=128, num_experts=128, convert_to_meta=True, prefix=""):
        raise NotImplementedError(
            "SOLAR_OPEN_STREAMING_LOAD=1 requests the streaming state-dict loader, which is a phase-2 item and "
            f"is not implemented yet; see {DESIGN_REFERENCE}. Unset SOLAR_OPEN_STREAMING_LOAD to use the "
            "phase-1 whole-model AutoModelForCausalLM.from_pretrained(dtype=torch.bfloat16) load."
        )

    # Mapping protocol: the phase-2 implementation returns contract-C1 keys (virtual fused expert keys included).
    def __getitem__(self, key):
        raise NotImplementedError(DESIGN_REFERENCE)

    def __iter__(self):
        raise NotImplementedError(DESIGN_REFERENCE)

    def __len__(self):
        raise NotImplementedError(DESIGN_REFERENCE)

    # substate()-aware views (utils/substate.py delegates here once the phase-2 hook is added).
    def substate(self, key) -> "LazyStateDict":
        raise NotImplementedError(DESIGN_REFERENCE)

    def has_substate(self, key) -> bool:
        raise NotImplementedError(DESIGN_REFERENCE)
