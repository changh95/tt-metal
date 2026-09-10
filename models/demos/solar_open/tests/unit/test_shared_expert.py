# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device unit test for models/demos/solar_open/tt/shared_expert.py.

The always-on shared expert is TP-sharded over its 1280-wide intermediate (160 columns per device at TP=8) and
returns each device's PARTIAL down projection without a CCL; the routed experts add that partial to their own
pre-all_reduce partial so one TP all_reduce serves both. This test therefore sums the per-device partials on the
host and compares the sum with the transformers reference ``SolarOpenMLP(config, intermediate_size=1280)``; it also
checks the output dtype (bf16; bfloat8_b for a DECODE call with ``decode_down_bfp8`` = ``SOLAR_OPEN_SHARED_DOWN_BFP8``,
phase 3e / A3 -- prefill calls stay bf16 under the knob), that the per-device outputs really are partials, and the
in-place add into a bfloat8_b accumulator the experts perform on the returned tensor (bf16 -> bfp8 mixed, or bfp8 +=
bfp8 under the knob). The ``down_bfp8`` parametrization covers both arms in one run, independent of the env knob.

Run (random weights, no checkpoint needed):
    HF_MODEL=models/demos/solar_open/configs/Solar-Open-100B MESH_DEVICE=P150x8 \
        pytest models/demos/solar_open/tests/unit/test_shared_expert.py
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.models.solar_open.modeling_solar_open import SolarOpenMLP

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.solar_open.config import MeshConfig, ModeConfig
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.solar_open.tt.shared_expert import SharedExpert

# Random weights at the real init scale (config.initializer_range = 0.02), bf16 activations. Starting values;
# re-baseline ~0.02 below the first measured PCCs (design D13).
PCC_THRESHOLDS = {ttnn.bfloat8_b: 0.98, ttnn.bfloat16: 0.995}
# The experts add the bf16 shared partial in place into their bfp8 routed partial; bfp8 rounding of a bf16 tensor
# keeps a PCC well above this.
INPLACE_ADD_PCC = 0.99
# Real Solar-Open-100B config.json (random weights): the host test needs no checkpoint and no HF_MODEL.
SOLAR_OPEN_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "Solar-Open-100B"


def _reference_shared_expert(config):
    """``SolarOpenMoE.shared_experts`` as built by transformers, with random weights at the real init scale."""
    intermediate_size = config.moe_intermediate_size * config.n_shared_experts
    reference = SolarOpenMLP(config, intermediate_size=intermediate_size).eval()
    init_std = getattr(config, "initializer_range", 0.02)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0, init_std)
    return reference


def _per_device_tensors(tt_tensor):
    return [ttnn.to_torch(device_tensor).float() for device_tensor in ttnn.get_device_tensors(tt_tensor)]


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "num_tokens", [1, 32, 128, 1024], ids=["decode_b1", "decode_b32", "prefill_128", "prefill_1024"]
)
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("down_bfp8", [False, True], ids=["down_bf16", "down_bfp8"])
@parametrize_mesh_with_fabric([(1, 8)])
def test_shared_expert(mesh_device, device_params, num_tokens, weight_dtype, down_bfp8, reset_seeds):
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    mesh_config = setup["mesh_config"]
    hidden_size = config.hidden_size
    is_decode = num_tokens <= ttnn.TILE_SIZE
    # The knob only changes DECODE partials (phase 3e / A3); a prefill call returns bf16 in both arms.
    expected_dtype = ttnn.bfloat8_b if (down_bfp8 and is_decode) else ttnn.bfloat16

    reference = _reference_shared_expert(config)
    hidden_states = torch.randn(1, num_tokens, hidden_size)
    with torch.no_grad():
        reference_output = reference(hidden_states).reshape(1, 1, num_tokens, hidden_size)

    # SolarOpenMLP.state_dict() has exactly the substate(mlp_sd, "shared_experts") keys the module consumes.
    tt_shared_expert = SharedExpert(
        mesh_device,
        config,
        reference.state_dict(),
        mesh_config,
        dtype=weight_dtype,
        tensor_cache_path=None,
        decode_down_bfp8=down_bfp8,
    )
    assert tt_shared_expert.intermediate_size_per_device == (
        config.moe_intermediate_size * config.n_shared_experts
    ) // (mesh_config.tp)
    assert tt_shared_expert.partial_dtype(is_decode) == expected_dtype

    tt_hidden_states = ttnn.from_torch(
        hidden_states.reshape(1, 1, num_tokens, hidden_size),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
    )
    tt_partial = tt_shared_expert(tt_hidden_states, is_decode=is_decode)

    # Contract C4: bf16 output (bfp8 at decode under the knob) with the input's logical shape; the input is not consumed.
    assert tt_partial.dtype == expected_dtype, f"shared partial must be {expected_dtype}, got {tt_partial.dtype}"
    assert tuple(tt_partial.shape) == (1, 1, num_tokens, hidden_size), tuple(tt_partial.shape)
    assert tt_hidden_states.is_allocated(), "SharedExpert must not deallocate its input"

    partials = _per_device_tensors(tt_partial)
    assert len(partials) == mesh_config.tp
    total = torch.stack(partials).sum(0)
    assert torch.isfinite(total).all(), "NaN/Inf in the summed shared-expert output"

    passing, pcc_message = comp_pcc(reference_output, total, PCC_THRESHOLDS[weight_dtype])
    logger.info(
        f"shared expert T={num_tokens} {weight_dtype} partial {tt_partial.dtype}: sum of {len(partials)} device "
        f"partials {pcc_message}"
    )

    if mesh_config.tp > 1:
        # Partial-sum property: each device holds a different 160-column slice, so the per-device outputs differ
        # from each other and from their sum (the experts, not this module, reduce them).
        for i in range(1, len(partials)):
            assert not torch.equal(partials[0], partials[i]), f"device 0 and {i} returned identical partials"
        for i, partial in enumerate(partials):
            partial_passing, partial_pcc = comp_pcc(reference_output, partial, PCC_THRESHOLDS[weight_dtype])
            assert not partial_passing, f"device {i} partial alone already matches the reference ({partial_pcc})"

    assert passing, f"shared expert T={num_tokens} {weight_dtype}: {pcc_message}"

    # The experts consume the partial with ttnn.add(routed_bfp8, shared, output_tensor=routed_bfp8) (decode in L1,
    # prefill in DRAM; shared bf16, or bfp8 at decode under the knob). Reproduce that add on a zero accumulator so a
    # Blackhole dtype/layout rejection shows up here, before the experts tests.
    accumulator = ttnn.zeros(
        tt_partial.shape,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=tt_partial.memory_config(),
    )
    accumulator = ttnn.add(accumulator, tt_partial, output_tensor=accumulator)
    assert accumulator.dtype == ttnn.bfloat8_b
    for i, (partial, accumulated) in enumerate(zip(partials, _per_device_tensors(accumulator))):
        add_passing, add_pcc = comp_pcc(partial, accumulated, INPLACE_ADD_PCC)
        assert add_passing, f"device {i}: in-place {tt_partial.dtype} -> bfp8 add of the shared partial: {add_pcc}"
    if tt_partial.dtype == ttnn.bfloat8_b:
        # 0 + x for a bfp8-representable x re-quantizes to the same block exponents, so the add is expected exact;
        # logged (not asserted) so a packer rounding-mode surprise is recorded rather than failing the gate.
        differing = sum(int((p != a).sum()) for p, a in zip(partials, _per_device_tensors(accumulator)))
        logger.info(f"shared expert T={num_tokens}: bfp8 += bfp8 add differs in {differing} elements (0 = exact)")
    logger.info(f"shared expert T={num_tokens}: in-place {tt_partial.dtype} -> bfp8 add OK")


def test_shared_expert_host_weight_layout():
    """Host-only (no device): pins the TP weight adapter and the partial-sum identity the design relies on.

    Checks, against the transformers ``SolarOpenMLP`` state dict, that ``SharedExpert`` hands ttnn the ``[1, 1, K, N]``
    transposed weights with the column-parallel (gate/up) and row-parallel (down) mesh mappers, the frozen cache
    stems ``gate_proj_tp8`` / ``up_proj_tp8`` / ``down_proj_tp8`` (contract C4), and the same stems with ``None``
    tensors in cache-only mode; then verifies in fp32 torch that the sum over the 8 per-device partials
    ``down_d(silu(gate_d(x)) * up_d(x))`` equals the reference MLP, which is why the experts can add the shared
    partial before their single TP all_reduce.
    """
    torch.manual_seed(0)
    tp = 8
    config = AutoConfig.from_pretrained(SOLAR_OPEN_CONFIG_DIR, trust_remote_code=True)
    mesh_config = MeshConfig((1, tp), decode=ModeConfig(tp=tp, ep=1))
    hidden_size = config.hidden_size
    intermediate_size = config.moe_intermediate_size * config.n_shared_experts
    per_device = intermediate_size // tp
    assert per_device == 160

    reference = _reference_shared_expert(config)
    state_dict = reference.state_dict()
    cache_path = "/cache/model.layers.0/mlp/shared_experts"

    recorded = []

    def fake_as_tensor(tensor, **kwargs):
        recorded.append((tensor, kwargs))
        return MagicMock(name=kwargs["cache_file_name"])

    def fake_shard_mapper(mesh_device, mesh_shape, dims):
        return ("ShardTensor2dMesh", tuple(dims))

    mesh_device = MagicMock()
    mesh_device.shape = (1, tp)
    patches = (
        patch.object(ttnn, "as_tensor", side_effect=fake_as_tensor),
        patch.object(ttnn, "ShardTensor2dMesh", side_effect=fake_shard_mapper),
        patch.object(ttnn, "init_device_compute_kernel_config", return_value=MagicMock()),
    )
    with patches[0], patches[1], patches[2]:
        module = SharedExpert(
            mesh_device, config, state_dict, mesh_config, dtype=ttnn.bfloat8_b, tensor_cache_path=cache_path
        )
    assert module.intermediate_size_per_device == per_device

    (gate, gate_kwargs), (up, up_kwargs), (down, down_kwargs) = recorded
    assert tuple(gate.shape) == (1, 1, hidden_size, intermediate_size)
    assert tuple(up.shape) == (1, 1, hidden_size, intermediate_size)
    assert tuple(down.shape) == (1, 1, intermediate_size, hidden_size)
    assert torch.equal(gate[0, 0], state_dict["gate_proj.weight"].T)
    assert torch.equal(up[0, 0], state_dict["up_proj.weight"].T)
    assert torch.equal(down[0, 0], state_dict["down_proj.weight"].T)
    assert gate_kwargs["cache_file_name"] == f"{cache_path}/gate_proj_tp{tp}"
    assert up_kwargs["cache_file_name"] == f"{cache_path}/up_proj_tp{tp}"
    assert down_kwargs["cache_file_name"] == f"{cache_path}/down_proj_tp{tp}"
    assert gate_kwargs["mesh_mapper"] == up_kwargs["mesh_mapper"] == ("ShardTensor2dMesh", (None, -1))
    assert down_kwargs["mesh_mapper"] == ("ShardTensor2dMesh", (None, -2))
    for kwargs in (gate_kwargs, up_kwargs, down_kwargs):
        assert kwargs["dtype"] == ttnn.bfloat8_b
        assert kwargs["layout"] == ttnn.TILE_LAYOUT
        assert kwargs["memory_config"] == ttnn.DRAM_MEMORY_CONFIG

    # Cache-only mode ({} state dict): the same three stems, no torch tensors.
    recorded.clear()
    with patches[0], patches[1], patches[2]:
        SharedExpert(mesh_device, config, {}, mesh_config, dtype=ttnn.bfloat16, tensor_cache_path=cache_path)
    assert [tensor for tensor, _ in recorded] == [None, None, None]
    assert [kwargs["cache_file_name"].rsplit("/", 1)[1] for _, kwargs in recorded] == [
        f"gate_proj_tp{tp}",
        f"up_proj_tp{tp}",
        f"down_proj_tp{tp}",
    ]
    assert all(kwargs["dtype"] == ttnn.bfloat16 for _, kwargs in recorded)

    # Partial-sum identity: device d owns intermediate columns [d * 160, (d + 1) * 160) of gate/up and the matching
    # rows of down, so the per-device down projections sum to the full MLP output.
    x = torch.randn(4, hidden_size)
    partials = []
    for d in range(tp):
        columns = slice(d * per_device, (d + 1) * per_device)
        activated = torch.nn.functional.silu(x @ gate[0, 0][:, columns]) * (x @ up[0, 0][:, columns])
        partials.append(activated @ down[0, 0][columns, :])
    with torch.no_grad():
        expected = reference(x)
    torch.testing.assert_close(torch.stack(partials).sum(0), expected, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(partials[0], expected), "a single device partial must not equal the full output"
