# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the expert weight layout adapter (no device).

``prepare_expert_weights_torch`` turns the transformers >= 5 fused expert layout (``gate_up_proj [E, 2I, H]`` with
the gate rows first, ``down_proj [E, H, I]``) into the per-device ``[gate_d | up_d]`` / row-sharded down layout the
sparse and dense expert kernels consume. These tests pin that mapping on tiny tensors, check the shape validation,
and prove that summing the 8 per-device partial products reproduces ``SolarOpenNaiveMoe`` exactly (fp32).
``load_expert_weights`` is exercised with ttnn mocked out to pin the cache stems and the absence of bias tensors.
"""

import dataclasses
from unittest.mock import MagicMock, patch

import pytest
import torch

import ttnn
from models.demos.solar_open.config import mesh_1x8
from models.demos.solar_open.tt.experts.config import ExpertConfig
from models.demos.solar_open.tt.experts.weights import (
    ExpertWeights,
    expert_shard_sizes,
    load_expert_weights,
    prepare_expert_weights_torch,
)

HIDDEN = 128  # != 2I for every intermediate size below, so the [E, H, 2I] layout is distinguishable
NUM_EXPERTS = 4
TP = 8


def _config(intermediate_size):
    return ExpertConfig(
        intermediate_size=intermediate_size, num_experts=NUM_EXPERTS, hidden_size=HIDDEN, num_experts_per_tok=2
    )


def _hf_state_dict(config, seed=0):
    """Random HF-layout expert weights: gate_up_proj [E, 2I, H] (gate rows first), down_proj [E, H, I]."""
    g = torch.Generator().manual_seed(seed)
    E, I, H = config.num_experts, config.intermediate_size, config.hidden_size
    return {
        "gate_up_proj": torch.randn(E, 2 * I, H, generator=g),
        "down_proj": torch.randn(E, H, I, generator=g),
    }


def _device_shards(fused_gate_up, down_proj, config, tp):
    """Emulate the mesh mappers: column-parallel shard of the fused gate/up, row-parallel shard of down."""
    I = config.intermediate_size
    local, padded = expert_shard_sizes(I, tp)
    for d in range(tp):
        gu_d = fused_gate_up[0, :, :, d * 2 * padded : (d + 1) * 2 * padded]  # [E, H, 2 * padded]
        down_d = down_proj[0, :, d * local : (d + 1) * local, :]  # [E, local, H]
        yield d, gu_d[..., :padded], gu_d[..., padded:], down_d


@pytest.mark.parametrize(
    "intermediate_size",
    [32, 256, 1280],
    ids=["I32_padded_4_to_32", "I256_tile_aligned", "I1280_solar_160_per_device"],
)
def test_per_device_layout_is_gate_then_up_then_down_rows(intermediate_size):
    """Shard d of the fused tensor is [gate cols d*I/tp:(d+1)*I/tp | zero pad | up cols ... | zero pad]; shard d of
    down_proj holds the intermediate rows d*I/tp:(d+1)*I/tp of down_proj^T."""
    config = _config(intermediate_size)
    sd = _hf_state_dict(config)
    E, I, H = NUM_EXPERTS, intermediate_size, HIDDEN
    local, padded = expert_shard_sizes(I, TP)
    assert local == I // TP and padded % 32 == 0 and padded >= local

    fused, down = prepare_expert_weights_torch(sd, config, TP)
    assert tuple(fused.shape) == (1, E, H, TP * 2 * padded)
    assert tuple(down.shape) == (1, E, I, H)

    gate_ref = sd["gate_up_proj"][:, :I, :].transpose(1, 2)  # [E, H, I]
    up_ref = sd["gate_up_proj"][:, I:, :].transpose(1, 2)
    down_ref = sd["down_proj"].transpose(1, 2)  # [E, I, H]
    for d, gate_d, up_d, down_d in _device_shards(fused, down, config, TP):
        cols = slice(d * local, (d + 1) * local)
        torch.testing.assert_close(gate_d[..., :local], gate_ref[..., cols], rtol=0, atol=0)
        torch.testing.assert_close(up_d[..., :local], up_ref[..., cols], rtol=0, atol=0)
        assert torch.count_nonzero(gate_d[..., local:]) == 0, "gate padding columns must be zero"
        assert torch.count_nonzero(up_d[..., local:]) == 0, "up padding columns must be zero"
        torch.testing.assert_close(down_d, down_ref[:, cols, :], rtol=0, atol=0)


def test_solar_shapes_have_no_padding():
    """Solar-Open-100B at TP=8: 1280 / 8 = 160 = 5 tiles, so pad_k == 0 and down_proj_padded can alias down_proj."""
    assert expert_shard_sizes(1280, 8) == (160, 160)
    config = ExpertConfig(intermediate_size=1280, num_experts=128, hidden_size=4096, num_experts_per_tok=8)
    assert config.activation == "silu"


@pytest.mark.parametrize(
    "bad_key, bad_shape, message",
    [
        ("gate_up_proj", (NUM_EXPERTS, HIDDEN, 2 * 32), r"gate_up_proj must be \[E, 2I, H\]"),  # interleaved layout
        ("gate_up_proj", (NUM_EXPERTS, 32, HIDDEN), r"gate_up_proj must be \[E, 2I, H\]"),  # gate only
        ("down_proj", (NUM_EXPERTS, 32, HIDDEN), r"down_proj must be \[E, H, I\]"),  # [E, I, H] orientation
    ],
    ids=["gate_up_interleaved_EH2I", "gate_up_half_rows", "down_EIH"],
)
def test_wrong_shapes_raise(bad_key, bad_shape, message, expect_error):
    config = _config(32)
    sd = _hf_state_dict(config)
    sd[bad_key] = torch.zeros(*bad_shape)
    with expect_error(ValueError, message):
        prepare_expert_weights_torch(sd, config, TP)


def _partial_moe_from_device_shards(x, top_k_index, top_k_weights, fused, down, config, tp):
    """The device math in torch: per device d, gate_d/up_d/down_d shards -> weighted per-expert outputs, summed
    over the selected experts, then summed over devices (the TP all_reduce). The zero-padded gate/up columns
    produce zero GLU columns which are dropped before the (unpadded) down shard, exactly like down_proj_padded."""
    T = x.shape[0]
    local, _ = expert_shard_sizes(config.intermediate_size, tp)
    out = torch.zeros(T, config.hidden_size)
    for _, gate_d, up_d, down_d in _device_shards(fused, down, config, tp):
        partial = torch.zeros(T, config.hidden_size)
        for t in range(T):
            for pos in range(top_k_index.shape[1]):
                e = int(top_k_index[t, pos])
                act = torch.nn.functional.silu(x[t] @ gate_d[e]) * (x[t] @ up_d[e])  # [padded]
                partial[t] += top_k_weights[t, pos] * (act[:local] @ down_d[e])
        out += partial
    return out


@pytest.mark.parametrize("intermediate_size", [32, 256], ids=["I32_padded", "I256_aligned"])
def test_sum_of_device_partials_matches_solar_open_naive_moe(intermediate_size):
    """SolarOpenNaiveMoe (HF reference, fp32) == sum over the tp devices of the per-device partial products computed
    from the adapted layout (proves gate-first, per-device column slicing, down orientation and the all_reduce math)."""
    from transformers.models.solar_open.configuration_solar_open import SolarOpenConfig
    from transformers.models.solar_open.modeling_solar_open import SolarOpenNaiveMoe

    config = _config(intermediate_size)
    sd = _hf_state_dict(config, seed=1)
    hf_cfg = SolarOpenConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=intermediate_size,
        n_routed_experts=NUM_EXPERTS,
        num_experts_per_tok=config.num_experts_per_tok,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
    )
    hf_cfg._experts_implementation = "eager"
    reference = SolarOpenNaiveMoe(hf_cfg)
    with torch.no_grad():
        reference.gate_up_proj.copy_(sd["gate_up_proj"])
        reference.down_proj.copy_(sd["down_proj"])

    g = torch.Generator().manual_seed(2)
    T = 6
    x = torch.randn(T, HIDDEN, generator=g)
    top_k_index = torch.stack(
        [torch.randperm(NUM_EXPERTS, generator=g)[: config.num_experts_per_tok] for _ in range(T)]
    )
    top_k_weights = torch.rand(T, config.num_experts_per_tok, generator=g)
    top_k_weights = top_k_weights / top_k_weights.sum(-1, keepdim=True)

    with torch.no_grad():
        expected = reference(x, top_k_index, top_k_weights)
    fused, down = prepare_expert_weights_torch(sd, config, TP)
    actual = _partial_moe_from_device_shards(x, top_k_index, top_k_weights, fused, down, config, TP)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_expert_weights_has_no_bias_fields():
    """The Solar experts are bias-free: no bias tensor may survive in the weight container."""
    names = {f.name for f in dataclasses.fields(ExpertWeights)}
    assert not any("bias" in n for n in names), names
    assert {"gate_up_proj", "down_proj", "intermediate_size_per_device", "intermediate_padded_per_device"} <= names
    assert "down_proj_padded" in names and "eye_tables" in names
    assert not any("per_expert" in n for n in names), "per-expert slices must be transient, not cached"


def test_load_expert_weights_cache_stems_and_shards():
    """load_expert_weights writes exactly two tensors (no bias files) under the contract stems
    ``<path>/gate_up_proj_fused_tp8`` and ``<path>/down_proj``, sharded column-/row-parallel, bfloat8_b by default."""
    config = _config(256)
    sd = _hf_state_dict(config)
    mesh_config = mesh_1x8()
    mesh_device = MagicMock(name="mesh_device")
    col_mapper, row_mapper = object(), object()
    as_tensor_calls = []

    def fake_as_tensor(tensor, **kwargs):
        as_tensor_calls.append((tensor, kwargs))
        return MagicMock(name=kwargs["cache_file_name"])

    with patch("models.demos.solar_open.tt.experts.weights.ttnn") as mock_ttnn, patch.object(
        mesh_config, "column_parallel", return_value=col_mapper
    ), patch.object(mesh_config, "row_parallel", return_value=row_mapper):
        mock_ttnn.as_tensor.side_effect = fake_as_tensor
        weights = load_expert_weights(
            mesh_device, config, sd, mesh_config, tensor_cache_path="layer0/mlp/experts"
        )  # default weight_dtype

    assert len(as_tensor_calls) == 2, [kw["cache_file_name"] for _, kw in as_tensor_calls]
    (gate_up_t, gate_up_kw), (down_t, down_kw) = as_tensor_calls
    assert gate_up_kw["cache_file_name"] == "layer0/mlp/experts/gate_up_proj_fused_tp8"
    assert down_kw["cache_file_name"] == "layer0/mlp/experts/down_proj"
    assert gate_up_kw["mesh_mapper"] is col_mapper and down_kw["mesh_mapper"] is row_mapper
    # the default weight_dtype is bound to the real ttnn enum at definition time (bfloat8_b, not bfloat4_b)
    assert gate_up_kw["dtype"] is ttnn.bfloat8_b and down_kw["dtype"] is ttnn.bfloat8_b
    assert gate_up_kw["layout"] is mock_ttnn.TILE_LAYOUT and down_kw["layout"] is mock_ttnn.TILE_LAYOUT
    assert gate_up_kw["memory_config"] is mock_ttnn.DRAM_MEMORY_CONFIG
    expected_gate_up, expected_down = prepare_expert_weights_torch(sd, config, TP)
    torch.testing.assert_close(gate_up_t, expected_gate_up, rtol=0, atol=0)
    torch.testing.assert_close(down_t, expected_down, rtol=0, atol=0)
    assert weights.intermediate_size_per_device == 256 // TP
    assert weights.intermediate_padded_per_device == 32
    assert weights.down_proj_padded is None and weights.eye_tables is None


def test_load_expert_weights_cache_only_mode_passes_none_tensors():
    """An empty state dict means "load from the tensor cache": both as_tensor calls get tensor=None."""
    config = _config(256)
    mesh_config = mesh_1x8()
    with patch("models.demos.solar_open.tt.experts.weights.ttnn") as mock_ttnn, patch.object(
        mesh_config, "column_parallel"
    ), patch.object(mesh_config, "row_parallel"):
        load_expert_weights(MagicMock(), config, {}, mesh_config, tensor_cache_path="p")
    tensors = [call.args[0] for call in mock_ttnn.as_tensor.call_args_list]
    assert tensors == [None, None]
