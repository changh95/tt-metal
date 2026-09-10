# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Solar-Open MoE MLP: router + routed experts + always-on shared expert, ONE tensor-parallel all_reduce per call.

Mirrors ``SolarOpenMoE.forward`` (``experts(x, idx, w) + shared_experts(x)``, both on the same post-attention-norm
input).  The shared expert's per-device partial is added inside the routed experts, before their all_reduce
(``Experts.__call__(..., shared_expert=...)``), so the block still issues a single CCL. ``MoEOptions.shared_down_bfp8``
(phase 3e / A3) makes the separate module emit that partial in bfloat8_b at decode so the add is a same-dtype op.

``MoEOptions.fuse_shared_expert`` (phase 2, design D2 follow-up) replaces the separate ``SharedExpert`` module by the
"129th expert": the shared shards are appended ON DEVICE as slot ``num_local_experts`` of the routed expert tensors
(``tt/experts/weights.py::fuse_always_on_expert``, no cache change) and the router emits ``[T, E + 1]`` routing
tensors whose last column is the constant 1.0, so the shared expert rides in the routed sparse / dense matmuls and the
union-of-experts reduction (5 launches per layer fewer: 3 linears, the GLU mul and the partial add).

``MoEOptions.indexed_decode`` (phase 2, profile lever 1; default on) sends a SINGLE-user decode step through the
router's ``route_indexed`` (top-k ids + weights, no dense tensor) and the experts' sparse_matmul indexed/gather path
(``tt/experts/decode.py::_decode_forward_indexed``: only the k selected experts are visited, compact outputs). ``route``
is the one place that decides between the two routing forms; batched steps and prefill always use the dense tensor.
"""

import dataclasses
import functools

from loguru import logger

import ttnn
from models.demos.solar_open.config import Mode, MoEOptions
from models.demos.solar_open.tt.expert_configs import solar_open_program_config
from models.demos.solar_open.utils.general_utils import get_cache_file_name
from models.demos.solar_open.utils.substate import substate

from .experts import ExpertConfig, Experts
from .experts.weights import fuse_always_on_expert
from .shared_expert import SharedExpert, load_shared_expert_weights
from .topk import TopKRouter


def indexed_decode_enabled(options: MoEOptions, fuse_shared: bool, decode_ep: int) -> tuple[bool, str]:
    """Whether single-user decode may take the indexed/gather expert path: ``(enabled, reason_if_not)``.

    The path needs the fused router (its ``moe_grouped_topk`` emits the uint16 ids ``ttnn.sparse_matmul(indices=)``
    takes; the ops chain's ``ttnn.topk`` gives uint32), the UNFUSED shared expert (an always-on slot is not a top-k
    selection and the fused layout is pinned to the batched union path) and EP=1 (the ids are global expert ids).
    """
    if not options.indexed_decode:
        return False, "SOLAR_OPEN_INDEXED_DECODE=0"
    if fuse_shared:
        return False, "the fused shared expert (SOLAR_OPEN_FUSE_SHARED_EXPERT=1) runs the batched union path"
    if decode_ep != 1:
        return False, f"decode EP={decode_ep} (the indexed ids are global expert ids; EP=1 required)"
    if options.router_impl != "fused":
        return False, f"router_impl={options.router_impl!r} emits uint32 ids (the fused router's uint16 ids are needed)"
    return True, ""


_SHARED_DOWN_DTYPE_LOGGED = set()


def _log_shared_down_dtype_once(shared_down_bfp8: bool):
    """One INFO line per process for the shared-expert decode partial dtype (48 MLPs share it), so a device run's log
    shows which SOLAR_OPEN_SHARED_DOWN_BFP8 arm it ran (phase 3e / A3)."""
    if shared_down_bfp8 not in _SHARED_DOWN_DTYPE_LOGGED:
        _SHARED_DOWN_DTYPE_LOGGED.add(shared_down_bfp8)
        logger.info(
            "shared-expert decode partial: "
            + (
                "bfloat8_b (SOLAR_OPEN_SHARED_DOWN_BFP8=1)"
                if shared_down_bfp8
                else "bf16 (SOLAR_OPEN_SHARED_DOWN_BFP8=0)"
            )
        )


def _fuse_shared_expert_into_experts(experts: Experts, w_gate, w_up, w_down):
    """Fold the shared expert's shards into ``experts`` as its last, always-on slot (in place).

    After this ``experts.weights`` has ``E + 1`` slots (``ExpertWeights.num_always_on_experts == 1``),
    ``experts.config.num_experts`` / ``experts.num_experts`` count the slots (the routing tensor width every
    reshape in decode.py / prefill.py follows) and the cached prefill sparsity is rebuilt at the new width.
    """
    dec, pre = experts.mesh_config.get_config(Mode.DECODE), experts.mesh_config.get_config(Mode.PREFILL)
    if dec.ep > 1 or pre.ep > 1:
        # experts_per_ep / moe_routing_remap / the EP prefill sparsity assume E % ep == 0 (129 is prime).
        raise NotImplementedError(
            f"always-on expert fusion needs EP=1 in both modes (decode EP={dec.ep}, prefill EP={pre.ep}); "
            "unset SOLAR_OPEN_FUSE_SHARED_EXPERT on multi-row meshes"
        )
    if experts.weights.num_always_on_experts:
        raise ValueError("the shared expert is already fused into these experts")
    experts.weights = fuse_always_on_expert(experts.weights, w_gate, w_up, w_down)
    experts.config = dataclasses.replace(experts.config, num_experts=experts.config.num_experts + 1)
    experts.num_experts = experts.config.num_experts
    experts.prefill_sparsity.deallocate(True)
    experts.prefill_sparsity = experts._create_prefill_sparsity()  # [1, 1, 1, E + 1] ones (EP=1)


class MLP:
    """MoE block of one decoder layer.

    Sub-modules (and the ``state_dict`` / cache sub-paths they own): ``router`` (``gate``), ``experts``
    (``experts``) and ``shared_expert`` (``shared_experts``; None when the config has no shared experts OR when
    ``MoEOptions.fuse_shared_expert`` folded it into ``experts`` as the always-on slot ``num_local_experts`` -- then
    ``experts.weights.num_always_on_experts == 1`` and ``router.num_slots == num_local_experts + 1``).
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
        router_persistent_token_counts=None,
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
            router_persistent_token_counts: token counts whose ``[T, E]`` router helpers are prebuilt and kept
                (``ModelArgs.router_persistent_token_counts``: traced prefill lengths, packed-prefill row counts);
                None = the router's default ``(32, 128)`` plus the decode batch
        """
        assert mesh_config is not None, "MLP requires a MeshConfig (Model builds a default one)"
        options = moe_options or MoEOptions()

        # The router and the fused SiLU-GLU only implement the Solar-Open configuration; fail loudly otherwise.
        assert hf_config.hidden_act == "silu", f"unsupported hidden_act {hf_config.hidden_act!r} (silu only)"
        assert (
            getattr(hf_config, "n_group", 1) == 1 and getattr(hf_config, "topk_group", 1) == 1
        ), "the router implements the single-group (n_group == topk_group == 1) selection rule only"
        assert getattr(hf_config, "norm_topk_prob", True), "the fused router always normalises the top-k weights"

        n_shared_experts = getattr(hf_config, "n_shared_experts", 0)
        fuse_shared = bool(options.fuse_shared_expert and n_shared_experts)
        if fuse_shared and n_shared_experts != 1:
            # The fused MLP would be n_shared * moe_intermediate_size wide and not fit one expert slot.
            raise NotImplementedError(
                f"always-on expert fusion supports n_shared_experts == 1, got {n_shared_experts}; unset "
                "SOLAR_OPEN_FUSE_SHARED_EXPERT"
            )
        self.fuse_shared_expert = fuse_shared
        self.hidden_size = hf_config.hidden_size
        self.indexed_decode, why_not = indexed_decode_enabled(
            options, fuse_shared, mesh_config.get_config(Mode.DECODE).ep
        )
        if options.indexed_decode and not self.indexed_decode:
            logger.debug(f"MLP: single-user decode takes the scan path ({why_not})")

        self.router = TopKRouter(
            mesh_device,
            hf_config,
            substate(state_dict, "gate"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "gate"),
            tokens_per_device=tokens_per_device,
            moe_options=options,
            always_on_slots=1 if fuse_shared else 0,
            persistent_token_counts=router_persistent_token_counts,
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

        # Shared expert: a separate module (phase-1 path, byte for byte unchanged), or fused into the routed experts
        # as their always-on slot. Both read the same three cache stems under <layer>/mlp/shared_experts.
        self.shared_expert = None
        if n_shared_experts:
            shared_cache_path = get_cache_file_name(tensor_cache_path, "shared_experts")
            if fuse_shared:
                w_gate, w_up, w_down = load_shared_expert_weights(
                    mesh_device,
                    hf_config,
                    substate(state_dict, "shared_experts"),
                    mesh_config,
                    dtype=options.shared_expert_dtype,
                    tensor_cache_path=shared_cache_path,
                )
                _fuse_shared_expert_into_experts(self.experts, w_gate, w_up, w_down)
                logger.debug(
                    f"MLP: shared expert fused as always-on slot {hf_config.num_local_experts} of the routed experts "
                    f"({self.experts.config.num_experts} slots, {options.expert_dtype})"
                )
            else:
                self.shared_expert = SharedExpert(
                    mesh_device,
                    hf_config,
                    substate(state_dict, "shared_experts"),
                    mesh_config,
                    dtype=options.shared_expert_dtype,
                    tensor_cache_path=shared_cache_path,
                    decode_down_bfp8=options.shared_down_bfp8,
                )
                _log_shared_down_dtype_once(options.shared_down_bfp8)

    def route(self, hidden_states, is_decode):
        """Run the router in the form the experts will consume: ``(dense, indexed)`` with exactly one of them set.

        ``dense`` is the ``[T, E]`` bf16 TILE routing tensor (``[T, E + 1]`` with the shared expert fused: column E
        is the constant 1.0 and the experts run slot E like any other active expert, so no shared_expert hook is
        passed); ``indexed`` is the ``IndexedRouting`` of a single decode token when ``self.indexed_decode`` (the
        experts then take ``_decode_forward_indexed``). ``hidden_states`` is not consumed.
        """
        if is_decode and self.indexed_decode and hidden_states.logical_volume() // self.hidden_size == 1:
            return None, self.router.route_indexed(hidden_states)
        _, dense = self.router(hidden_states, is_decode=is_decode)
        return dense, None

    def __call__(self, hidden_states, is_decode):
        """Route -> (routed experts + shared expert) -> one TP all_reduce.

        Args:
            hidden_states: ``[1, 1, T, H]`` post-attention-norm hidden states, replicated over the TP axis
                (decode: T = users per device, 1..32; prefill: T = seq_len, a multiple of 32)
            is_decode: decode (True) or prefill (False) mode

        Returns:
            ``[1, 1, T, H]`` bfp8, all-reduced: routed-expert output plus the unweighted shared-expert output
        """
        dense, indexed = self.route(hidden_states, is_decode)

        shared_expert = (
            functools.partial(self.shared_expert, is_decode=is_decode) if self.shared_expert is not None else None
        )
        output = self.experts(
            hidden_states,
            topk_expert_weights=dense,
            is_decode=is_decode,
            shared_expert=shared_expert,
            indexed_routing=indexed,
        )
        if indexed is not None:
            indexed.deallocate()
        return output
