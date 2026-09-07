# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device unit tests for the Solar-Open router (``tt/topk.py::TopKRouter``, design 5.3 / contract C2).

The reference is the HF rule ``SolarOpenMoE.route_tokens_to_experts`` (fp32 sigmoid -> + e_score_correction_bias
for selection -> top-k -> unbiased scores normalised -> * routed_scaling_factor) applied to fp32 logits from the
same bf16 weights. Inputs are RMS-normed random bf16 hidden states, a gate weight of the real scale (std 0.04) and
the real layer-0 ``e_score_correction_bias`` when the checkpoint shard is present (synthetic +-0.01 bias otherwise).

Metrics (probe_results.md, round 2): the fused kernel's in-SFPU sigmoid is accurate to ~5e-4, so expert-set flips
are only tolerated on near-tie tokens whose 8th-vs-9th biased-score margin is below a floor (they are counted and
logged); tokens with a clear margin must agree 100 %. The dense routing tensor must reach PCC >= 0.99 on the agreeing
tokens (and over all tokens from T=128 on, where the near-tie flips cannot dominate it), the weights of agreeing
tokens must match the unbiased normalised scores within a bf16-level absolute tolerance, rows must sum to
``routed_scaling_factor`` and hold exactly k non-zero entries, and all devices must return identical routing.

Run (random weights, no checkpoint needed):
    HF_MODEL=models/demos/solar_open/configs/Solar-Open-100B MESH_DEVICE=P150x8 \\
        pytest models/demos/solar_open/tests/unit/test_router.py -k 1x8
"""

import os
import types
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers.models.solar_open.modeling_solar_open import SolarOpenMoE, SolarOpenTopkRouter

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.solar_open.config import MoEOptions
from models.demos.solar_open.tt.topk import TopKRouter

from ..test_factory import TestFactory, get_num_devices, parametrize_mesh_with_fabric

# Token counts: single user, partial decode batch, full decode batch, traced prefill length, one prefill chunk.
TOKEN_COUNTS = [1, 8, 32, 128, 4096]
# (router_impl, router_fp32_logits): the default first, then the three fallbacks.
ROUTER_VARIANTS = [("fused", True), ("fused", False), ("ops", True), ("ops", False)]
ROUTER_VARIANT_IDS = ["fused-fp32", "fused-bf16", "ops-fp32", "ops-bf16"]

DENSE_PCC_MIN = 0.99  # dense [T, E] routing tensor
# The all-token dense PCC also absorbs the tolerated near-tie flips (one flip moves a whole ~0.125 weight between two
# columns: measured on P150, 1 flip in 8 tokens = 12.5 % of the rows costs 0.016 PCC with otherwise exact weights), so
# it is only asserted from this token count on, where a flip rate up to ~7 % keeps it above DENSE_PCC_MIN; below it
# (T = 1 / 8 / 32) it is logged and the dense PCC is asserted on the agreeing tokens (every clear-margin token).
ALL_TOKEN_PCC_MIN_TOKENS = 128
ROW_SUM_ATOL = 1e-2  # bf16 rounding of 8 normalised weights
# Max |tt_weight - ref_weight| on agreeing tokens (weights are bf16, <= ~0.5): fp32 logits leave bf16 output
# rounding (<= 1e-3) plus the fused kernel's ~5e-4 sigmoid error; bf16 logits add bf16 matmul rounding.
# (A PCC over the gathered weights is NOT asserted: at the real weight scale the top-8 normalised weights of a
# token are nearly uniform, spread ~3e-3, so bf16 rounding alone caps that PCC near 0.95; it is only logged.)
WEIGHT_ATOL = {True: 5e-3, False: 1e-2}
# Tokens whose reference 8th-vs-9th biased-score margin is at least this must select the same expert set; flips
# below it are the kernels' noise floor and are only reported. fp32 logits: fused-kernel sigmoid error ~5e-4
# (probe); bf16 logits add the bf16 logit rounding. With RMS-normed inputs and the real weight scale (logits std
# ~2.5) the sigmoid tail is flat: ~28 % of tokens fall below 1e-3 and ~15 % below 5e-4.
CLEAR_MARGIN = {True: 1e-3, False: 5e-3}

GATE_WEIGHT_STD = 0.04  # real Solar-Open router weight scale
LAYER0_BIAS_SHARD = "model-00002-of-00042.safetensors"
LAYER0_BIAS_KEY = "model.layers.0.mlp.gate.e_score_correction_bias"

# Trace smoke test: the router at T<=128 is three small programs; 32 MiB is generous.
TRACE_REGION_SIZE = 32 * 1024 * 1024
TRACE_TOKEN_COUNTS = [32, 128]

MESH_SHAPES = [(1, 1), (1, 8)]


# --------------------------------------------------------------------------------------------------------------
# Inputs and reference
# --------------------------------------------------------------------------------------------------------------


def _load_layer0_bias(model_path, num_experts):
    """Real ``model.layers.0.mlp.gate.e_score_correction_bias`` (fp32 [E]) from the HF snapshot, or None."""
    if not model_path:
        return None
    shard = Path(model_path) / LAYER0_BIAS_SHARD
    if not shard.is_file():
        return None
    try:
        from safetensors import safe_open

        with safe_open(str(shard), framework="pt") as f:
            if LAYER0_BIAS_KEY not in f.keys():
                return None
            bias = f.get_tensor(LAYER0_BIAS_KEY).float()
    except Exception as exc:  # corrupt / partially downloaded shard: fall back to the synthetic bias
        logger.warning(f"Could not read {LAYER0_BIAS_KEY} from {shard}: {exc}")
        return None
    return bias if tuple(bias.shape) == (num_experts,) else None


def _synthetic_bias(num_experts, generator, step=2**-9):
    """+-5 steps of ``step`` (default 2^-9 -> |bias| <= 0.01, bf16-exact), design C11."""
    return torch.randint(-5, 6, (num_experts,), generator=generator).float() * step


def _rms_normed_hidden(tokens, hidden_size, generator):
    """What the router sees in the model: post_attention_layernorm output (unit RMS), bf16."""
    x = torch.randn(tokens, hidden_size, generator=generator)
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    return x.to(torch.bfloat16)


def _make_router_state(config, bias, generator):
    """State dict of ``substate(mlp, "gate")``: bf16 weight [E, H] (std 0.04) and fp32 bias [E]."""
    weight = (torch.randn(config.num_local_experts, config.hidden_size, generator=generator) * GATE_WEIGHT_STD).to(
        torch.bfloat16
    )
    return {"weight": weight, "e_score_correction_bias": bias.float()}


class HFRouterReference:
    """``SolarOpenTopkRouter`` (fp32 logits) + ``SolarOpenMoE.route_tokens_to_experts`` on the same weights."""

    def __init__(self, config, state_dict):
        self.config = config
        self.gate = SolarOpenTopkRouter(config)
        with torch.no_grad():
            self.gate.weight.copy_(state_dict["weight"].float())
            self.gate.e_score_correction_bias.copy_(state_dict["e_score_correction_bias"].float())
        # route_tokens_to_experts only reads these attributes of the MoE module.
        self._moe = types.SimpleNamespace(
            gate=self.gate,
            n_group=config.n_group,
            topk_group=config.topk_group,
            n_routed_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
        )

    @torch.no_grad()
    def route(self, hidden_bf16):
        """-> (logits fp32 [T, E], indices int64 [T, k] unsorted, weights fp32 [T, k])."""
        logits = self.gate(hidden_bf16)
        indices, weights = SolarOpenMoE.route_tokens_to_experts(self._moe, logits)
        return logits, indices, weights

    @torch.no_grad()
    def selection_margins(self, logits):
        """Per token: biased score of the k-th selected expert minus the (k+1)-th; small = near tie."""
        biased = torch.sigmoid(logits) + self.gate.e_score_correction_bias
        top = biased.topk(self.config.num_experts_per_tok + 1, dim=-1).values
        return top[:, -2] - top[:, -1]


def _dense_from_sparse(indices, weights, num_experts):
    return torch.zeros(indices.shape[0], num_experts, dtype=torch.float32).scatter(1, indices.long(), weights.float())


# --------------------------------------------------------------------------------------------------------------
# Device helpers
# --------------------------------------------------------------------------------------------------------------


def _to_device(hidden_bf16, mesh_device, dtype=ttnn.bfloat16):
    tokens, hidden_size = hidden_bf16.shape
    return ttnn.from_torch(
        hidden_bf16.reshape(1, 1, tokens, hidden_size),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
    )


def _per_device_torch(tt_tensor):
    """One torch tensor per device (the router output is replicated, so all of them must agree)."""
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tt_tensor)]


def _sorted_sets(indices_long):
    return torch.sort(indices_long.long(), dim=-1).values


def _pcc(golden, calculated, threshold):
    passing, message = comp_pcc(golden.float(), calculated.float(), threshold)
    return passing, message


def _check_routing(dense_tt, indices_tt, reference, ref_indices, ref_weights, margins, fp32_logits, label):
    """All C2 / design-5.3 assertions on one device's ``(dense [T, E], indices [T, k])`` output."""
    tokens, num_experts = dense_tt.shape
    top_k = ref_indices.shape[1]
    route_scale = reference.config.routed_scaling_factor
    dense = dense_tt.float()
    indices = indices_tt.long()

    assert tuple(indices.shape) == (tokens, top_k), f"{label}: indices shape {tuple(indices.shape)}"
    assert not torch.isnan(dense).any(), f"{label}: NaN in the dense routing tensor"
    assert (dense >= 0).all(), f"{label}: negative routing weight"

    # Exactly k non-zero entries per row (the experts' union mask and >0 planners rely on it), and they are the
    # experts named by the indices output.
    nonzero_per_row = (dense != 0).sum(dim=-1)
    assert (nonzero_per_row == top_k).all(), f"{label}: non-zero entries per row {nonzero_per_row.tolist()[:16]}..."
    dense_sets = torch.sort(torch.nonzero(dense, as_tuple=False)[:, 1].view(tokens, top_k), dim=-1).values
    tt_sets = _sorted_sets(indices)
    assert torch.equal(dense_sets, tt_sets), f"{label}: dense non-zero columns differ from the indices output"

    # Rows sum to routed_scaling_factor.
    row_sums = dense.sum(dim=-1)
    max_row_sum_err = (row_sums - route_scale).abs().max().item()
    assert max_row_sum_err <= ROW_SUM_ATOL, f"{label}: row sums deviate from {route_scale} by {max_row_sum_err:.4f}"

    # Expert-set agreement, split by the reference's selection margin.
    ref_sets = _sorted_sets(ref_indices)
    same_set = (tt_sets == ref_sets).all(dim=-1)
    clear = margins >= CLEAR_MARGIN[fp32_logits]
    num_clear_mismatch = int((~same_set & clear).sum())
    num_mismatch = int((~same_set).sum())
    near_tie_fraction = float((~clear).float().mean())
    logger.info(
        f"{label}: expert-set agreement {tokens - num_mismatch}/{tokens} "
        f"({num_mismatch} flips, all on near-tie tokens; near-tie tokens {near_tie_fraction:.1%}, "
        f"margin < {CLEAR_MARGIN[fp32_logits]}); row-sum max err {max_row_sum_err:.4f}"
    )
    assert num_clear_mismatch == 0, (
        f"{label}: {num_clear_mismatch} token(s) with a clear selection margin picked a different expert set: "
        f"tokens {torch.nonzero(~same_set & clear).flatten().tolist()[:16]}"
    )

    # Dense PCC: on the agreeing tokens (weight fidelity of the op chain; includes every clear-margin token) and over
    # all tokens (also counts the near-tie flips; asserted from ALL_TOKEN_PCC_MIN_TOKENS on, see there).
    ref_dense = _dense_from_sparse(ref_indices, ref_weights, num_experts)
    all_passing, all_message = _pcc(ref_dense, dense, DENSE_PCC_MIN)
    if same_set.any():
        agree_passing, agree_message = _pcc(ref_dense[same_set], dense[same_set], DENSE_PCC_MIN)
        logger.info(
            f"{label}: dense PCC on {int(same_set.sum())} agreeing tokens {agree_message}, all tokens {all_message}"
        )
        assert (
            agree_passing
        ), f"{label}: dense routing PCC on the agreeing tokens below {DENSE_PCC_MIN}: {agree_message}"
    else:
        # Every token flipped on a near tie (T=1 whose single token is a near tie): the selection is within the
        # tolerated noise floor and there is no agreeing token whose weights could be compared.
        logger.warning(f"{label}: no agreeing token, weight fidelity not measurable (all-token PCC {all_message})")
    assert (
        all_passing or tokens < ALL_TOKEN_PCC_MIN_TOKENS
    ), f"{label}: all-token dense routing PCC below {DENSE_PCC_MIN} ({all_message}); {num_mismatch} near-tie flips"
    max_weight_err = None
    if same_set.any():
        tt_weights = torch.gather(dense[same_set], 1, ref_indices[same_set].long())
        max_weight_err = (tt_weights - ref_weights[same_set]).abs().max().item()
        _, weights_message = _pcc(ref_weights[same_set], tt_weights, DENSE_PCC_MIN)
        logger.info(
            f"{label}: weights on {int(same_set.sum())} agreeing tokens: max |err| {max_weight_err:.2e}, "
            f"PCC (informational) {weights_message}"
        )
        assert max_weight_err <= WEIGHT_ATOL[fp32_logits], (
            f"{label}: routing weights deviate from the unbiased normalised sigmoid scores by {max_weight_err:.3e} "
            f"(allowed {WEIGHT_ATOL[fp32_logits]})"
        )
    return {"mismatch": num_mismatch, "near_tie_fraction": near_tie_fraction, "max_weight_err": max_weight_err}


def _assert_replicated(per_device, label, atol=0.0):
    """Every device must hold the same routing (the experts build per-device partials from it)."""
    first = per_device[0].float()
    for device_index, other in enumerate(per_device[1:], start=1):
        assert torch.allclose(first, other.float(), atol=atol, rtol=0.0), (
            f"{label}: device {device_index} disagrees with device 0 (max abs diff "
            f"{(first - other.float()).abs().max().item():.3e})"
        )


def _expected_index_dtype(router_impl, fp32_logits):
    # Contract C2: fused -> uint16; ops -> uint32 (its ttnn.topk always runs on fp32 scores, also for bf16 logits).
    return ttnn.uint16 if router_impl == "fused" else ttnn.uint32


# --------------------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------------------


@parametrize_mesh_with_fabric(MESH_SHAPES)
@pytest.mark.parametrize("router_impl, fp32_logits", ROUTER_VARIANTS, ids=ROUTER_VARIANT_IDS)
@pytest.mark.parametrize("tokens", TOKEN_COUNTS, ids=[f"T{t}" for t in TOKEN_COUNTS])
def test_router_matches_hf_rule(mesh_device, tokens, router_impl, fp32_logits):
    """TopKRouter vs SolarOpenMoE.route_tokens_to_experts for every impl/dtype variant and token count."""
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    generator = torch.Generator().manual_seed(1234 + tokens)

    bias = _load_layer0_bias(setup["model_args"].model_path, config.num_local_experts)
    bias_source = "real layer-0" if bias is not None else "synthetic"
    if bias is None:
        bias = _synthetic_bias(config.num_local_experts, generator)
    state_dict = _make_router_state(config, bias, generator)
    reference = HFRouterReference(config, state_dict)

    options = MoEOptions(router_impl=router_impl, router_fp32_logits=fp32_logits)
    router = TopKRouter(
        mesh_device, config, state_dict, tensor_cache_path=None, tokens_per_device=32, moe_options=options
    )

    hidden = _rms_normed_hidden(tokens, config.hidden_size, generator)
    logits, ref_indices, ref_weights = reference.route(hidden)
    margins = reference.selection_margins(logits)

    tt_hidden = _to_device(hidden, mesh_device)
    tt_indices, tt_dense = router(tt_hidden, is_decode=tokens <= 32)

    assert tt_dense.dtype == ttnn.bfloat16, f"dense dtype {tt_dense.dtype}"
    assert tt_dense.layout == ttnn.TILE_LAYOUT, f"dense layout {tt_dense.layout}"
    assert tuple(tt_dense.shape) == (tokens, config.num_local_experts), f"dense shape {tuple(tt_dense.shape)}"
    assert tt_indices.dtype == _expected_index_dtype(router_impl, fp32_logits), f"indices dtype {tt_indices.dtype}"

    dense_per_device = _per_device_torch(tt_dense)
    indices_per_device = _per_device_torch(tt_indices)
    label = f"router[{router_impl}, fp32_logits={fp32_logits}, T={tokens}, bias={bias_source}]"
    _assert_replicated(dense_per_device, label + " dense")
    _assert_replicated([t.long() for t in indices_per_device], label + " indices")
    _check_routing(
        dense_per_device[0], indices_per_device[0], reference, ref_indices, ref_weights, margins, fp32_logits, label
    )

    # hidden_states is not consumed by the router (the experts read it next).
    assert tt_hidden.is_allocated(), f"{label}: the router deallocated its input"
    tt_dense.deallocate(True)
    tt_indices.deallocate(True)
    tt_hidden.deallocate(True)


@parametrize_mesh_with_fabric(MESH_SHAPES)
def test_router_bfp8_input(mesh_device):
    """Contract C2 admits a bfloat8_b input: the router widens it to bf16 before the fp32 linear (the block-float
    operand itself is what HF would route in that case, so the reference routes the dequantised bfp8 values; the
    bfp8 -> bf16 typecast is exact). Same metric as test_router_matches_hf_rule at the traced prefill length."""
    tokens = 128
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    generator = torch.Generator().manual_seed(4321)

    bias = _load_layer0_bias(setup["model_args"].model_path, config.num_local_experts)
    if bias is None:
        bias = _synthetic_bias(config.num_local_experts, generator)
    state_dict = _make_router_state(config, bias, generator)
    reference = HFRouterReference(config, state_dict)
    router = TopKRouter(mesh_device, config, state_dict, tensor_cache_path=None, tokens_per_device=32)

    tt_hidden = _to_device(_rms_normed_hidden(tokens, config.hidden_size, generator), mesh_device, dtype=ttnn.bfloat8_b)
    assert tt_hidden.dtype == ttnn.bfloat8_b
    hidden_dequantised = ttnn.to_torch(ttnn.get_device_tensors(tt_hidden)[0]).reshape(tokens, config.hidden_size)
    logits, ref_indices, ref_weights = reference.route(hidden_dequantised.to(torch.bfloat16))
    margins = reference.selection_margins(logits)

    tt_indices, tt_dense = router(tt_hidden, is_decode=False)
    label = f"router[fused, fp32_logits=True, T={tokens}, bfp8 input]"
    _check_routing(
        _per_device_torch(tt_dense)[0],
        _per_device_torch(tt_indices)[0],
        reference,
        ref_indices,
        ref_weights,
        margins,
        True,
        label,
    )
    assert (
        tt_hidden.is_allocated() and tt_hidden.dtype == ttnn.bfloat8_b
    ), f"{label}: the router consumed / altered its input"
    tt_dense.deallocate(True)
    tt_indices.deallocate(True)
    tt_hidden.deallocate(True)


@parametrize_mesh_with_fabric(MESH_SHAPES)
def test_router_bias_is_selection_only(mesh_device):
    """A strong e_score_correction_bias changes WHICH experts are selected but not their (unbiased) weights."""
    tokens = 128
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    generator = torch.Generator().manual_seed(4321)

    # +-5 steps of 2^-3 (|bias| <= 0.625): changes every token's expert set, makes ~97 % of the selection margins
    # clear (>= 1e-3) and separates the biased-normalised from the unbiased weights by ~3.6e-2 (7x WEIGHT_ATOL).
    strong_bias = _synthetic_bias(config.num_local_experts, generator, step=2**-3)
    state_dict = _make_router_state(config, strong_bias, generator)
    biased_reference = HFRouterReference(config, state_dict)
    unbiased_reference = HFRouterReference(
        config, {**state_dict, "e_score_correction_bias": torch.zeros(config.num_local_experts)}
    )

    hidden = _rms_normed_hidden(tokens, config.hidden_size, generator)
    logits, ref_indices, ref_weights = biased_reference.route(hidden)
    _, unbiased_indices, _ = unbiased_reference.route(hidden)
    margins = biased_reference.selection_margins(logits)
    selection_changed = (_sorted_sets(ref_indices) != _sorted_sets(unbiased_indices)).any(dim=-1)
    assert selection_changed.any(), "test setup: the strong bias did not change any token's expert set"
    logger.info(f"bias changes the expert set of {int(selection_changed.sum())}/{tokens} tokens")

    router = TopKRouter(mesh_device, config, state_dict, tensor_cache_path=None, tokens_per_device=32)
    tt_hidden = _to_device(hidden, mesh_device)
    tt_indices, tt_dense = router(tt_hidden, is_decode=False)
    dense = _per_device_torch(tt_dense)[0]
    indices = _per_device_torch(tt_indices)[0]
    # Selection follows the BIASED rule and the weights match the UNBIASED normalised scores (WEIGHT_ATOL) ...
    _check_routing(dense, indices, biased_reference, ref_indices, ref_weights, margins, True, "router[strong bias]")

    # ... and are clearly distinct from the wrong alternative (normalising the biased scores), which the strong
    # bias separates from the correct weights by far more than WEIGHT_ATOL.
    same_set = (_sorted_sets(indices) == _sorted_sets(ref_indices)).all(dim=-1)
    biased_scores = torch.sigmoid(logits) + strong_bias
    wrong_weights = torch.gather(biased_scores, 1, ref_indices)
    wrong_weights = wrong_weights / wrong_weights.sum(dim=-1, keepdim=True)
    tt_weights = torch.gather(dense.float()[same_set], 1, ref_indices[same_set].long())
    err_vs_unbiased = (tt_weights - ref_weights[same_set]).abs().max().item()
    err_vs_biased = (tt_weights - wrong_weights[same_set]).abs().max().item()
    logger.info(
        f"router[strong bias]: max |err| vs unbiased weights {err_vs_unbiased:.2e}, vs biased {err_vs_biased:.2e}"
    )
    assert err_vs_biased > 4 * WEIGHT_ATOL[True], (
        "test setup: the strong bias does not separate biased from unbiased weights " f"({err_vs_biased:.3e})"
    )

    tt_dense.deallocate(True)
    tt_indices.deallocate(True)
    tt_hidden.deallocate(True)


def _parametrize_mesh_for_trace(mesh_shapes):
    """Like ``parametrize_mesh_with_fabric`` but with an explicit trace region so the smoke test is independent
    of the trace-region YAML (``SOLAR_OPEN_TRACE_REGION_SIZE`` still overrides)."""
    num_devices = get_num_devices()
    shapes = [s for s in mesh_shapes if s[0] * s[1] <= num_devices]
    if os.getenv("CI") == "true" and len(shapes) > 1:
        shapes = [max(shapes, key=lambda s: s[0] * s[1])]
    trace_region_size = int(os.getenv("SOLAR_OPEN_TRACE_REGION_SIZE", TRACE_REGION_SIZE))
    params = [
        pytest.param(
            shape,
            {
                "fabric_config": None if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": trace_region_size,
            },
            id=f"{shape[0]}x{shape[1]}",
        )
        for shape in shapes
    ] or [
        pytest.param(
            (1, 1),
            {"fabric_config": None, "trace_region_size": trace_region_size},
            id="1x1",
            marks=pytest.mark.skip(reason="No supported Solar-Open mesh shape fits on this system"),
        )
    ]
    return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)


@_parametrize_mesh_for_trace(MESH_SHAPES)
@pytest.mark.parametrize("tokens", TRACE_TOKEN_COUNTS, ids=[f"T{t}" for t in TRACE_TOKEN_COUNTS])
def test_router_trace_replay(mesh_device, tokens):
    """Capture the default router at the decode batch and the traced prefill length; replays must equal eager."""
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    generator = torch.Generator().manual_seed(99 + tokens)

    bias = _load_layer0_bias(setup["model_args"].model_path, config.num_local_experts)
    if bias is None:
        bias = _synthetic_bias(config.num_local_experts, generator)
    state_dict = _make_router_state(config, bias, generator)
    reference = HFRouterReference(config, state_dict)
    router = TopKRouter(mesh_device, config, state_dict, tensor_cache_path=None, tokens_per_device=32)

    hidden_a = _rms_normed_hidden(tokens, config.hidden_size, generator)
    hidden_b = _rms_normed_hidden(tokens, config.hidden_size, generator)

    # Compile run (eager) on the persistent input, then capture on the same input.
    tt_input = _to_device(hidden_a, mesh_device)
    eager_indices, eager_dense = router(tt_input, is_decode=tokens <= 32)
    eager_dense_a = _per_device_torch(eager_dense)[0].float()
    eager_indices.deallocate(True)
    eager_dense.deallocate(True)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_indices, trace_dense = router(tt_input, is_decode=tokens <= 32)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    def replay(hidden):
        host = ttnn.from_torch(
            hidden.reshape(1, 1, tokens, config.hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, tt_input)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        return _per_device_torch(trace_dense)[0].float(), _per_device_torch(trace_indices)[0].long()

    try:
        # Replay on the captured input reproduces the eager result bit for bit ...
        dense_a, _ = replay(hidden_a)
        assert torch.equal(dense_a, eager_dense_a), "trace replay on the captured input differs from eager"
        # ... and on a new input the trace computes the new routing (not a stale copy), matching the HF rule.
        dense_b, indices_b = replay(hidden_b)
        assert not torch.equal(dense_b, eager_dense_a), "trace replay ignored the new input"
        logits_b, ref_indices_b, ref_weights_b = reference.route(hidden_b)
        _check_routing(
            dense_b,
            indices_b,
            reference,
            ref_indices_b,
            ref_weights_b,
            reference.selection_margins(logits_b),
            True,
            f"router[trace replay, T={tokens}]",
        )
        # Replays are deterministic.
        dense_b_again, _ = replay(hidden_b)
        assert torch.equal(dense_b_again, dense_b), "two replays on the same input differ"
    finally:
        ttnn.release_trace(mesh_device, trace_id)
