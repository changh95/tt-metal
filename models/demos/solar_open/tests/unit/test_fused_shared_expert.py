# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the "129th always-on expert" fusion (``MoEOptions.fuse_shared_expert``, phase 2 / design D2 follow-up).

With the flag on, ``tt/mlp.py`` does not build a ``SharedExpert``: the shared expert's per-device shards are appended
ON DEVICE as slot ``num_local_experts`` (= 128) of the routed expert tensors (``tt/experts/weights.py::
fuse_always_on_expert``: ``[1, 129, 4096, 320]`` / ``[1, 129, 160, 4096]`` at TP=8) and the router emits a
``[T, 129]`` routing tensor whose last column is the constant 1.0, so the shared expert rides in the routed sparse /
dense matmuls and the union-of-experts reduction (5 launches per layer fewer). The math is identical to
``SolarOpenMoE.forward`` (``experts(x) + shared_experts(x)``); the numerics differ only in that the shared slot now runs
with the routed experts' bfp8 activations, so fused and unfused outputs are close (PCC >= ``FUSED_VS_UNFUSED_PCC``)
but NOT bit-identical, while the fused WEIGHTS and the routed router columns are bit-identical to the unfused ones.

Device test (``-k 1x8``): two ``MLP`` blocks (fused / unfused) built from the same random ``SolarOpenDecoderLayer``
state dict and run on the same inputs for decode b1 / b8 / b32 (b8: fewer users than
``ProgramConfig.decode_down_batched_min_tokens``, so the batched path runs the 8x4 down grid in both modes) and prefill
128 (dense bmm) / 1024 (expert-sorted hot/cold split) / 4096 (one chunk). Host tests (``-k host``, no device): the concat / typecast / deallocation sequence of
``fuse_always_on_expert`` with ttnn mocked out, its guards, the router's pre-seeded scatter tensor, the MLP-side fusion
guards, and a torch emulation proving that the 129-slot layout with routing column 128 = 1.0 reproduces
``SolarOpenNaiveMoe + SolarOpenMLP``.

The decode b1 case needs ``tt/experts/decode.py`` to send single-user steps of a fused module (``weights.
num_always_on_experts > 0``) through ``_decode_forward_batched`` (the single-user path's bfp8 HC transposes have no
validated padded-C=129 contract; see the phase-2 design notes); a failure of exactly that case points there.

Run (random weights, no checkpoint needed):
    HF_MODEL=models/demos/solar_open/configs/Solar-Open-100B MESH_DEVICE=P150x8 \\
        pytest models/demos/solar_open/tests/unit/test_fused_shared_expert.py -k 1x8
    SOLAR_OPEN_NUM_DEVICES=8 pytest models/demos/solar_open/tests/unit/test_fused_shared_expert.py -k host
"""

import collections
import contextlib
import dataclasses
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.solar_open.config import MoEOptions, mesh_1x8, mesh_2x4
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_batch_seq, parametrize_mesh_with_fabric
from models.demos.solar_open.tests.unit import test_modules as tm
from models.demos.solar_open.tt import mlp as mlp_module
from models.demos.solar_open.tt.experts import prefill as experts_prefill
from models.demos.solar_open.tt.experts.config import ExpertConfig
from models.demos.solar_open.tt.experts.weights import (
    ExpertWeights,
    expert_shard_sizes,
    fuse_always_on_expert,
    prepare_expert_weights_torch,
)
from models.demos.solar_open.tt.mlp import MLP
from models.demos.solar_open.tt.topk import TopKRouter
from models.demos.solar_open.utils.substate import substate

# Fused vs unfused MLP output: the shared slot's activations are bfp8 (routed compute config) instead of bf16 with the
# SharedExpert's HiFi2 config, and the fused sum skips one bfp8 re-quantisation. Design 4.1 expected >= 0.9995 and the
# phase-1 shared expert (auto matmul configs) measured 0.9993-0.9994 (decode b1 / b8 / b32) / 0.9993 (prefill 128). The
# phase-2 1D program configs of the unfused SharedExpert (perf-p0 lever a) moved the UNFUSED output closer to the HF
# reference (decode b1 vs HF 0.99822 -> 0.99896) while the fused slot is unchanged (0.99827), so the two forms now agree
# to 0.9989 at decode b1: the floor is the bfp8-activation class of the fused slot, not the old 0.999 (2026-09-07).
FUSED_VS_UNFUSED_PCC = 0.998
# Routed devices whose full [1, 129, ...] tensors are read back for the bit-identity check (the others check the
# shared slot and one routed slot through device-side slices: a whole bfp8 tensor is ~1 GiB of fp32 on the host).
FULL_WEIGHT_CHECK_DEVICES = (0, -1)


@contextlib.contextmanager
def count_launches(*names):
    """Count calls of the named ``ttnn`` entry points (e.g. ``linear``) made inside the block: yields a Counter."""
    counts = collections.Counter()

    def counting(name, original):
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapper

    with contextlib.ExitStack() as stack:
        for name in names:
            stack.enter_context(mock.patch.object(ttnn, name, counting(name, getattr(ttnn, name))))
        yield counts


def _device_tensors(tt_tensor):
    return ttnn.get_device_tensors(tt_tensor)


def _to_torch(device_tensor):
    return ttnn.to_torch(device_tensor)


# --------------------------------------------------------------------------------------------------------------
# Device test: fused vs unfused MLP on identical weights and inputs
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(1, 8)])
@parametrize_batch_seq(
    [(1, 1), (8, 1), (32, 1), (1, 128), (1, 1024), (1, 4096)],
    ids=["decode_b1", "decode_b8", "decode_b32", "prefill_128", "prefill_1024", "prefill_4096"],
)
def test_fused_shared_expert_equivalence(mesh_device, device_params, batch_size, seq_len, test_thresholds, reset_seeds):
    """Fused (129-slot) vs unfused (128 routed + SharedExpert) MLP: identical weights per device, identical routed
    router columns plus a constant 1.0 column, outputs within ``FUSED_VS_UNFUSED_PCC`` of each other and both above
    the ``mlp`` threshold against ``SolarOpenMoE``, one all_reduce per call, 1 instead of 4 ``ttnn.linear`` launches.
    """
    is_decode = seq_len == 1
    num_tokens = batch_size * seq_len
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    config = setup["config"]
    mesh_config = setup["mesh_config"]
    tp = mesh_config.tp
    hidden_size = config.hidden_size
    num_experts = config.num_local_experts
    if not getattr(config, "n_shared_experts", 0):
        pytest.skip("the config has no shared expert to fuse")
    pcc_threshold = test_thresholds[setup["model_args"].model_name]["decode" if is_decode else "prefill"]["mlp"]

    reference_layer = tm.setup_reference_layer(setup)
    mlp_state = substate(reference_layer.state_dict(), "mlp")
    base_options = MoEOptions.from_env()  # honour the dtype / router flags; pin the fusion flag both ways

    def build(fused):
        return MLP(
            mesh_device,
            config,
            mlp_state,
            setup["ccl_manager"],
            dtype=setup["dtype"],
            tensor_cache_path=None,
            mesh_config=mesh_config,
            tokens_per_device=batch_size if is_decode else 32,
            moe_options=dataclasses.replace(base_options, fuse_shared_expert=fused),
        )

    unfused = build(False)
    fused = build(True)

    # ---- 1. structure and per-device weight identity ----
    assert unfused.shared_expert is not None and fused.shared_expert is None
    assert unfused.experts.weights.num_always_on_experts == 0 and fused.experts.weights.num_always_on_experts == 1
    assert unfused.experts.config.num_experts == num_experts and fused.experts.config.num_experts == num_experts + 1
    assert fused.experts.num_experts == num_experts + 1
    assert unfused.router.num_slots == num_experts and fused.router.num_slots == num_experts + 1
    assert fused.experts.weights.gate_up_proj.dtype == unfused.experts.weights.gate_up_proj.dtype
    ip = unfused.experts.weights.intermediate_padded_per_device
    gate_up_f, down_f = fused.experts.weights.gate_up_proj, fused.experts.weights.down_proj
    gate_up_u, down_u = unfused.experts.weights.gate_up_proj, unfused.experts.weights.down_proj
    assert tuple(gate_up_f.shape) == (1, num_experts + 1, hidden_size, 2 * ip), tuple(gate_up_f.shape)
    assert tuple(down_f.shape) == (1, num_experts + 1, ip, hidden_size), tuple(down_f.shape)
    assert tuple(fused.experts.prefill_sparsity.shape)[-1] == num_experts + 1

    shared = unfused.shared_expert
    # Bit identity of slot E needs equal dtypes; with SOLAR_OPEN_SHARED_EXPERT_DTYPE != SOLAR_OPEN_EXPERT_DTYPE the
    # shared shards are typecast on device (bf16 -> bfp8, or bfp8 -> bfp4) and only match up to that quantisation.
    slot_is_bit_identical = base_options.shared_expert_dtype == base_options.expert_dtype

    def assert_slot_equal(actual, expected, what):
        if slot_is_bit_identical:
            assert torch.equal(actual, expected), what
        else:
            passing, pcc = comp_pcc(expected.float(), actual.float(), 0.99)
            assert (
                passing
            ), f"{what} (typecast {base_options.shared_expert_dtype} -> {base_options.expert_dtype}): {pcc}"

    full_check = {d % tp for d in FULL_WEIGHT_CHECK_DEVICES}
    for d in range(tp):
        gu_f_d, dn_f_d = _device_tensors(gate_up_f)[d], _device_tensors(down_f)[d]
        w_gate_d = _to_torch(_device_tensors(shared.w_gate)[d])[0, 0]
        w_up_d = _to_torch(_device_tensors(shared.w_up)[d])[0, 0]
        w_down_d = _to_torch(_device_tensors(shared.w_down)[d])[0, 0]
        if d in full_check:
            gu_f_t, gu_u_t = _to_torch(gu_f_d)[0], _to_torch(_device_tensors(gate_up_u)[d])[0]
            dn_f_t, dn_u_t = _to_torch(dn_f_d)[0], _to_torch(_device_tensors(down_u)[d])[0]
            assert torch.equal(gu_f_t[:num_experts], gu_u_t), f"device {d}: routed gate/up slots changed by the fusion"
            assert torch.equal(dn_f_t[:num_experts], dn_u_t), f"device {d}: routed down slots changed by the fusion"
            slot_gu, slot_dn = gu_f_t[num_experts], dn_f_t[num_experts]
        else:
            slot_gu = _to_torch(ttnn.slice(gu_f_d, [0, num_experts, 0, 0], [1, num_experts + 1, hidden_size, 2 * ip]))[
                0, 0
            ]
            slot_dn = _to_torch(ttnn.slice(dn_f_d, [0, num_experts, 0, 0], [1, num_experts + 1, ip, hidden_size]))[0, 0]
            slot0_f = _to_torch(ttnn.slice(gu_f_d, [0, 0, 0, 0], [1, 1, hidden_size, 2 * ip]))[0, 0]
            slot0_u = _to_torch(ttnn.slice(_device_tensors(gate_up_u)[d], [0, 0, 0, 0], [1, 1, hidden_size, 2 * ip]))[
                0, 0
            ]
            assert torch.equal(slot0_f, slot0_u), f"device {d}: routed slot 0 changed by the fusion"
        assert_slot_equal(
            slot_gu,
            torch.cat([w_gate_d, w_up_d], dim=-1),
            f"device {d}: slot {num_experts} gate/up != [w_gate_d | w_up_d] of the shared expert",
        )
        assert_slot_equal(slot_dn, w_down_d, f"device {d}: slot {num_experts} down != w_down_d of the shared expert")

    logger.info(f"fused weights: {tuple(gate_up_f.shape)} / {tuple(down_f.shape)}, slot {num_experts} == shared shards")

    # ---- 2. router: routed columns bit-equal, column E == 1.0 ----
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(torch.bfloat16).float()
    replicate = ttnn.ShardTensor2dMesh(dims=(None, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device)

    def upload():
        return ttnn.from_torch(
            hidden_states.reshape(1, 1, num_tokens, hidden_size),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=tm.ACTIVATION_DTYPE,
            mesh_mapper=replicate,
        )

    tt_x = upload()
    _, dense_u = unfused.router(tt_x, is_decode=is_decode)
    _, dense_f = fused.router(tt_x, is_decode=is_decode)
    dense_u_t = _to_torch(_device_tensors(dense_u)[0]).float()
    dense_f_t = _to_torch(_device_tensors(dense_f)[0]).float()
    assert tuple(dense_u.shape) == (num_tokens, num_experts) and tuple(dense_f.shape) == (num_tokens, num_experts + 1)
    assert torch.equal(dense_f_t[:, :num_experts], dense_u_t), "fused router changed the routed columns"
    assert torch.equal(
        dense_f_t[:, num_experts], torch.ones(num_tokens)
    ), f"always-on column must be 1.0 in every row, got {dense_f_t[:, num_experts].unique().tolist()}"
    assert torch.allclose(
        dense_f_t[:, :num_experts].sum(-1), torch.full((num_tokens,), fused.router.route_scale), atol=1e-2
    )
    tm.assert_replicated(dense_f, "fused router output")
    dense_u.deallocate(True)
    dense_f.deallocate(True)
    tt_x.deallocate(True)

    # ---- 3. MLP outputs: fused vs unfused, both vs HF ----
    with torch.no_grad():
        reference = reference_layer.mlp(hidden_states).reshape(num_tokens, hidden_size).float()
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    expected_ccl = {"all_reduce": 1} if tp > 1 else {}

    def run(block, label):
        experts_prefill.LAST_SORTED_MOE_PLAN.clear()
        tt_in = upload()
        with tm.count_ccl_ops() as ccl_counts, count_launches("linear") as launches:
            tt_out = block(tt_in, is_decode=is_decode)
        out = ttnn.to_torch(tt_out, mesh_composer=mesh_composer)[..., :num_tokens, :hidden_size]
        tm.assert_replicated(tt_out, f"{label} MoE output")
        tt_out.deallocate(True)
        plan = dict(experts_prefill.LAST_SORTED_MOE_PLAN)
        assert dict(ccl_counts) == expected_ccl, f"{label}: expected {expected_ccl}, launched {dict(ccl_counts)}"
        return out.reshape(num_tokens, hidden_size).float(), launches["linear"], plan

    out_u, linears_u, plan_u = run(unfused, "unfused")
    out_f, linears_f, plan_f = run(fused, "fused")

    # Launches: the router's linear on both sides; the unfused block adds the SharedExpert's 3 linears per chunk.
    # perf-p1 (HOT_EXPERTS_PER_EXPERT_LINEAR): a split on the expert-sorted path runs one ttnn.linear per hot expert
    # of its plan (the always-on slot is hot in every fused split: count == split_len > cap), so the fused block is
    # NOT linear-free on long prefills -- its saving there is the SharedExpert's 3 linears + GLU + add per chunk.
    # The routed hot experts are decided per split (count > cap) and LAST_SORTED_MOE_PLAN holds only the LAST split's
    # plan (a 4096-token prefill = 4 splits with their own hot counts), so the routed hot linears are not predicted
    # from it: both blocks route identically, hence fused = unfused - 3 x chunks + one always-on linear per sorted split.
    chunks = 1 if is_decode else -(-num_tokens // unfused.experts.program_config.sequence_chunk_size)
    split_len = min(seq_len, unfused.experts.program_config.get_down_split_size(seq_len))
    n_splits = 1 if is_decode else -(-num_tokens // split_len)
    per_expert_linears = experts_prefill.HOT_EXPERTS_PER_EXPERT_LINEAR
    sorted_splits = n_splits if (not is_decode and plan_f) else 0  # the sorted path ran (dense bmm / decode: 0)
    expected_f = linears_u - 3 * chunks + (sorted_splits if per_expert_linears else 0)
    min_u = 1 + 3 * chunks + ((plan_u.get("hot", 0) if plan_u else 0) if per_expert_linears else 0)
    assert linears_f == expected_f, (
        f"fused MLP launched {linears_f} ttnn.linear calls (expected {expected_f}: the unfused block's {linears_u} - 3 x "
        f"{chunks} shared-expert linears + {sorted_splits} always-on hot linears; plans unfused {plan_u}, fused {plan_f})"
    )
    assert linears_u >= min_u, (
        f"unfused MLP launched {linears_u} ttnn.linear calls (expected at least {min_u}: router + 3 x {chunks} + the last "
        f"split's hot experts {plan_u})"
    )
    if not is_decode and plan_f:
        assert plan_f.get("always_on", 0) == 1, plan_f

    passing_equiv, pcc_equiv = comp_pcc(out_u, out_f, FUSED_VS_UNFUSED_PCC)
    max_diff = (out_u - out_f).abs().max().item()
    passing_u, pcc_u = comp_pcc(reference, out_u, pcc_threshold)
    passing_f, pcc_f = comp_pcc(reference, out_f, pcc_threshold)
    logger.info(
        f"fused shared expert ({'decode' if is_decode else 'prefill'} {num_tokens} tokens): fused vs unfused "
        f"{pcc_equiv} (max |diff| {max_diff:.4f}); vs HF unfused {pcc_u}, fused {pcc_f}; linears {linears_u} -> "
        f"{linears_f}; sorted plans unfused {plan_u or None}, fused {plan_f or None}"
    )
    assert passing_u, f"unfused MLP vs SolarOpenMoE below {pcc_threshold}: {pcc_u}"
    assert passing_f, f"fused MLP vs SolarOpenMoE below {pcc_threshold}: {pcc_f}"
    assert passing_equiv, f"fused vs unfused MLP below {FUSED_VS_UNFUSED_PCC}: {pcc_equiv} (max |diff| {max_diff:.4f})"

    # Long splits take the expert-sorted hot/cold path in both modes; with the fusion the always-on slot is hot in
    # every split (count == split_len > every cap) and rides in the hot group. The planner reports it separately once
    # prefill.py distinguishes always-on slots (``always_on`` key); until then it is counted among the hot experts.
    if not is_decode and split_len > unfused.experts.program_config.dense_bmm_max_tokens:
        assert plan_u.get("split") == split_len and plan_f.get("split") == split_len, (plan_u, plan_f)
        if "always_on" in plan_f:
            assert plan_f["always_on"] == 1 and plan_f["hot"] >= 0, plan_f
        else:
            assert plan_f["hot"] >= 1, f"the always-on slot must be a hot expert of every split: {plan_f}"
    else:
        assert not plan_u and not plan_f, (plan_u, plan_f)


# --------------------------------------------------------------------------------------------------------------
# Host tests (no device)
# --------------------------------------------------------------------------------------------------------------

HOST_HIDDEN, HOST_IP = 64, 32  # per-device shapes of the mocked tensors (tile aligned: local == padded)


def _fake_tensor(shape, dtype="bfp8", name="t"):
    return SimpleNamespace(shape=tuple(shape), dtype=dtype, deallocate=MagicMock(name=f"{name}.deallocate"), name=name)


def _fake_routed_weights(num_experts=4, dtype="bfp8", local=HOST_IP, padded=HOST_IP):
    return ExpertWeights(
        gate_up_proj=_fake_tensor((1, num_experts, HOST_HIDDEN, 2 * padded), dtype, "gate_up"),
        down_proj=_fake_tensor((1, num_experts, local, HOST_HIDDEN), dtype, "down"),
        intermediate_size_per_device=local,
        intermediate_padded_per_device=padded,
    )


def _fake_shared_shards(dtype="bfp8"):
    return (
        _fake_tensor((1, 1, HOST_HIDDEN, HOST_IP), dtype, "w_gate"),
        _fake_tensor((1, 1, HOST_HIDDEN, HOST_IP), dtype, "w_up"),
        _fake_tensor((1, 1, HOST_IP, HOST_HIDDEN), dtype, "w_down"),
    )


def _mock_concat(tensors, dim, memory_config=None):
    shape = list(tensors[0].shape)
    shape[dim] = sum(t.shape[dim] for t in tensors)
    out = _fake_tensor(shape, tensors[0].dtype, f"concat(dim={dim})")
    out.parts, out.dim = list(tensors), dim
    return out


def _mock_typecast(tensor, dtype):
    out = _fake_tensor(tensor.shape, dtype, f"typecast({tensor.name})")
    out.source = tensor
    return out


def test_host_fuse_always_on_expert_concat_sequence():
    """ttnn mocked: w_gate / w_up widened to bf16, [w_gate | w_up] along dim 3 in bf16, cast back to the routed dtype,
    then one dim-1 concat per routed tensor; no down cast when the dtypes agree; every handle freed exactly once; the
    returned container keeps the sizes and counts one always-on slot.
    """
    weights = _fake_routed_weights()
    w_gate, w_up, w_down = _fake_shared_shards()
    with patch("models.demos.solar_open.tt.experts.weights.ttnn") as mock_ttnn:
        mock_ttnn.concat.side_effect = _mock_concat
        mock_ttnn.typecast.side_effect = _mock_typecast
        fused = fuse_always_on_expert(weights, w_gate, w_up, w_down)

    casts = mock_ttnn.typecast.call_args_list
    assert len(casts) == 3  # widen gate, widen up, narrow the bf16 concat back to the routed dtype; no down cast
    assert casts[0].args == (w_gate, mock_ttnn.bfloat16) and casts[1].args == (w_up, mock_ttnn.bfloat16)
    calls = mock_ttnn.concat.call_args_list
    assert len(calls) == 3
    gate_wide, up_wide = calls[0].args[0]
    assert gate_wide.source is w_gate and up_wide.source is w_up and calls[0].kwargs["dim"] == 3
    wide_concat = casts[2].args[0]
    assert wide_concat.name == "concat(dim=3)" and wide_concat.dtype is mock_ttnn.bfloat16
    assert casts[2].args[1] == "bfp8"
    shared_gate_up = fused.gate_up_proj.parts[1]
    assert shared_gate_up.source is wide_concat
    assert calls[1].args[0] == [weights.gate_up_proj, shared_gate_up] and calls[1].kwargs["dim"] == 1
    assert calls[2].args[0] == [weights.down_proj, w_down] and calls[2].kwargs["dim"] == 1
    assert all(call.kwargs["memory_config"] is mock_ttnn.DRAM_MEMORY_CONFIG for call in calls)

    assert fused.gate_up_proj.shape == (1, 5, HOST_HIDDEN, 2 * HOST_IP) and fused.gate_up_proj.dtype == "bfp8"
    assert fused.down_proj.shape == (1, 5, HOST_IP, HOST_HIDDEN)
    assert fused.num_always_on_experts == 1
    assert fused.intermediate_size_per_device == HOST_IP and fused.intermediate_padded_per_device == HOST_IP
    assert fused.down_proj_padded is None and fused.eye_tables is None
    for handle in (
        w_gate,
        w_up,
        w_down,
        weights.gate_up_proj,
        weights.down_proj,
        gate_wide,
        up_wide,
        wide_concat,
        shared_gate_up,
    ):
        handle.deallocate.assert_called_once_with(True)
    assert not fused.gate_up_proj.deallocate.called and not fused.down_proj.deallocate.called


def test_host_fuse_always_on_expert_typecasts_mismatched_dtype():
    """bf16 shared shards (SOLAR_OPEN_SHARED_EXPERT_DTYPE=bf16) or bfp4 routed experts: the shared blocks are typecast
    on device to the routed dtype before the dim-1 concats and the pre-cast handles are freed."""
    weights = _fake_routed_weights(dtype="bfp4")
    w_gate, w_up, w_down = _fake_shared_shards(dtype="bfp8")
    with patch("models.demos.solar_open.tt.experts.weights.ttnn") as mock_ttnn:
        mock_ttnn.concat.side_effect = _mock_concat
        mock_ttnn.typecast.side_effect = _mock_typecast
        fused = fuse_always_on_expert(weights, w_gate, w_up, w_down)

    casts = mock_ttnn.typecast.call_args_list
    assert len(casts) == 4  # widen gate, widen up, narrow gate/up to bfp4, down to bfp4
    assert casts[0].args == (w_gate, mock_ttnn.bfloat16) and casts[1].args == (w_up, mock_ttnn.bfloat16)
    gu_cast_args, down_cast_args = casts[2].args, casts[3].args
    assert gu_cast_args[1] == "bfp4" and down_cast_args == (w_down, "bfp4")
    assert gu_cast_args[0].name == "concat(dim=3)" and gu_cast_args[0].dtype is mock_ttnn.bfloat16
    assert fused.gate_up_proj.dtype == "bfp4" and fused.down_proj.dtype == "bfp4"
    assert fused.gate_up_proj.parts[1].source is gu_cast_args[0]  # the cast block, not the bf16 concat
    assert fused.down_proj.parts[1].source is w_down
    for handle in (
        w_gate,
        w_up,
        w_down,
        gu_cast_args[0],
        *gu_cast_args[0].parts,
        weights.gate_up_proj,
        weights.down_proj,
    ):
        handle.deallocate.assert_called_once_with(True)
    # the cast blocks are freed after the dim-1 concats
    fused.gate_up_proj.parts[1].deallocate.assert_called_once_with(True)
    fused.down_proj.parts[1].deallocate.assert_called_once_with(True)


def test_host_fuse_always_on_expert_guards(expect_error):
    with patch("models.demos.solar_open.tt.experts.weights.ttnn") as mock_ttnn:
        mock_ttnn.concat.side_effect = _mock_concat
        # zero-padded per-device intermediate (other TP factors): needs a host-side pad first
        with expect_error(NotImplementedError, "tile-aligned per-device intermediate"):
            fuse_always_on_expert(_fake_routed_weights(local=24, padded=32), *_fake_shared_shards())
        # a shard that is not one slot wide
        w_gate, w_up, w_down = _fake_shared_shards()
        w_up = _fake_tensor((1, 1, HOST_HIDDEN, 2 * HOST_IP), name="w_up")
        with expect_error(ValueError, "w_up must have the per-device shape"):
            fuse_always_on_expert(_fake_routed_weights(), w_gate, w_up, w_down)
        # already prepared for dense prefill
        prepared = dataclasses.replace(
            _fake_routed_weights(), down_proj_padded=_fake_tensor((1, 4, HOST_IP, HOST_HIDDEN))
        )
        with expect_error(ValueError, "before the first dense prefill"):
            fuse_always_on_expert(prepared, *_fake_shared_shards())
        assert mock_ttnn.concat.call_count == 0  # every guard fires before any device op


def test_host_mlp_fusion_guards(expect_error):
    """``_fuse_shared_expert_into_experts`` refuses EP>1 meshes (E=129 is prime: the EP paths assume E % ep == 0) and
    a second fusion; on success it grows the config/width and rebuilds the prefill sparsity."""
    experts = SimpleNamespace(
        mesh_config=mesh_2x4(),
        weights=_fake_routed_weights(),
        config=ExpertConfig(
            intermediate_size=8 * HOST_IP, num_experts=4, hidden_size=HOST_HIDDEN, num_experts_per_tok=2
        ),
        num_experts=4,
        prefill_sparsity=MagicMock(name="prefill_sparsity"),
        _create_prefill_sparsity=MagicMock(return_value="new_sparsity"),
    )
    with expect_error(NotImplementedError, "EP=1"):
        mlp_module._fuse_shared_expert_into_experts(experts, *_fake_shared_shards())
    assert not experts.prefill_sparsity.deallocate.called

    experts.mesh_config = mesh_1x8()
    old_sparsity = experts.prefill_sparsity
    with patch("models.demos.solar_open.tt.experts.weights.ttnn") as mock_ttnn:
        mock_ttnn.concat.side_effect = _mock_concat
        mock_ttnn.typecast.side_effect = _mock_typecast
        mlp_module._fuse_shared_expert_into_experts(experts, *_fake_shared_shards())
    assert experts.weights.num_always_on_experts == 1 and experts.weights.gate_up_proj.shape[1] == 5
    assert experts.config.num_experts == 5 and experts.num_experts == 5
    assert experts.config.num_experts_per_tok == 2 and experts.config.intermediate_size == 8 * HOST_IP
    old_sparsity.deallocate.assert_called_once_with(True)
    experts._create_prefill_sparsity.assert_called_once_with()
    assert experts.prefill_sparsity == "new_sparsity"

    with expect_error(ValueError, "already fused"):
        mlp_module._fuse_shared_expert_into_experts(experts, *_fake_shared_shards())


def test_host_router_scatter_seed():
    """With ``always_on_slots=1`` the router keeps one replicated ``[1, E + 1]`` bf16 row (0 x E, then 1.0), hands the
    row itself out for T=1 and ``ttnn.repeat``s of it for other token counts, and never allocates zeros; without the
    flag it allocates ``[T, E]`` zeros as before."""
    hf_config = SimpleNamespace(
        num_experts_per_tok=8, num_local_experts=128, hidden_size=4096, routed_scaling_factor=1.0
    )
    with patch("models.demos.solar_open.tt.topk.ttnn") as mock_ttnn:
        mock_ttnn.repeat.side_effect = lambda tensor, shape, **kwargs: SimpleNamespace(source=tensor, reps=list(shape))
        router = TopKRouter(MagicMock(name="mesh"), hf_config, {}, tokens_per_device=8, always_on_slots=1)
        assert router.num_experts == 128 and router.num_slots == 129
        (seed_row,), from_torch_kwargs = mock_ttnn.from_torch.call_args
        assert seed_row.dtype == torch.bfloat16 and tuple(seed_row.shape) == (1, 129)
        assert torch.equal(seed_row[0, :128], torch.zeros(128, dtype=torch.bfloat16)) and seed_row[0, 128] == 1.0
        assert from_torch_kwargs["layout"] is mock_ttnn.TILE_LAYOUT and from_torch_kwargs["dtype"] is mock_ttnn.bfloat16
        assert from_torch_kwargs["mesh_mapper"] is mock_ttnn.ReplicateTensorToMesh.return_value
        assert mock_ttnn.zeros.call_count == 0
        # prebuilt seeds for the decode batch, 32 and 128 rows: repeats of the seed row (never the row itself)
        for tokens in (8, 32, 128):
            (_bias_wide, seed), transient = router._get_row_tensors(tokens)
            assert not transient and seed.source is router._seed_row and seed.reps == [tokens, 1]
        (_bias_wide, seed_1), transient = router._get_row_tensors(1)
        assert seed_1 is router._seed_row and not transient
        (_bias_wide, seed_4096), transient = router._get_row_tensors(4096)
        assert transient and seed_4096.reps == [4096, 1] and seed_4096 is not router._seed_row

    with patch("models.demos.solar_open.tt.topk.ttnn") as mock_ttnn:
        router = TopKRouter(MagicMock(name="mesh"), hf_config, {}, tokens_per_device=32)
        assert router.num_slots == 128 and router._seed_row is None and mock_ttnn.from_torch.call_count == 0
        assert sorted(call.args[0] for call in mock_ttnn.zeros.call_args_list) == [[32, 128], [128, 128]]


def _hf_moe_reference(hf_cfg, sd, shared_sd):
    """``SolarOpenNaiveMoe + SolarOpenMLP`` with the given (fp32) weights, as ``SolarOpenMoE.forward`` sums them."""
    from transformers.models.solar_open.modeling_solar_open import SolarOpenMLP, SolarOpenNaiveMoe

    experts = SolarOpenNaiveMoe(hf_cfg)
    shared = SolarOpenMLP(hf_cfg, intermediate_size=hf_cfg.moe_intermediate_size * hf_cfg.n_shared_experts)
    with torch.no_grad():
        experts.gate_up_proj.copy_(sd["gate_up_proj"])
        experts.down_proj.copy_(sd["down_proj"])
        shared.load_state_dict(shared_sd)
    return experts.eval(), shared.eval()


@pytest.mark.parametrize("intermediate_size", [256, 512], ids=["I256_32_per_device", "I512_64_per_device"])
def test_host_fused_layout_reproduces_routed_plus_shared(intermediate_size):
    """Torch emulation of the fused per-device layout: slot E of every device is ``[gate_d | up_d]`` / ``down_d`` of
    the shared MLP, and the ``E + 1``-slot MoE math with routing column E = 1.0 (summed over the TP devices like the
    all_reduce) equals ``SolarOpenNaiveMoe(x, idx, w) + SolarOpenMLP(x)`` -- the identity the device fusion relies on.
    """
    from transformers.models.solar_open.configuration_solar_open import SolarOpenConfig

    torch.manual_seed(0)
    tp, E, H, I, top_k, T = 8, 4, 128, intermediate_size, 2, 6
    local, padded = expert_shard_sizes(I, tp)
    assert local == padded, "the fusion is only defined for tile-aligned per-device intermediates"
    config = ExpertConfig(intermediate_size=I, num_experts=E, hidden_size=H, num_experts_per_tok=top_k)
    sd = {"gate_up_proj": torch.randn(E, 2 * I, H), "down_proj": torch.randn(E, H, I)}
    shared_sd = {
        "gate_proj.weight": torch.randn(I, H),
        "up_proj.weight": torch.randn(I, H),
        "down_proj.weight": torch.randn(H, I),
    }
    hf_cfg = SolarOpenConfig(
        hidden_size=H,
        moe_intermediate_size=I,
        n_routed_experts=E,
        n_shared_experts=1,
        num_experts_per_tok=top_k,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
    )
    hf_cfg._experts_implementation = "eager"
    experts_ref, shared_ref = _hf_moe_reference(hf_cfg, sd, shared_sd)

    x = torch.randn(T, H)
    idx = torch.stack([torch.randperm(E)[:top_k] for _ in range(T)])
    w = torch.rand(T, top_k)
    w = w / w.sum(-1, keepdim=True)
    with torch.no_grad():
        expected = experts_ref(x, idx, w) + shared_ref(x)

    # Per-device fused layout (what ttnn.concat builds on device from the routed and shared shards).
    fused_gu, down = prepare_expert_weights_torch(sd, config, tp)  # [1, E, H, tp * 2 * padded], [1, E, I, H]
    shared_gate, shared_up = shared_sd["gate_proj.weight"].T, shared_sd["up_proj.weight"].T  # [H, I]
    shared_down = shared_sd["down_proj.weight"].T  # [I, H]
    dense = torch.zeros(T, E + 1).scatter_(1, idx, w)
    dense[:, E] = 1.0  # the router's always-on column
    out = torch.zeros(T, H)
    for d in range(tp):
        cols = slice(d * local, (d + 1) * local)
        gu_d = fused_gu[0, :, :, d * 2 * padded : (d + 1) * 2 * padded]  # [E, H, 2 * padded] = [gate_d | up_d]
        slot_gu = torch.cat([shared_gate[:, cols], shared_up[:, cols]], dim=-1).unsqueeze(0)  # [1, H, 2 * padded]
        gu_all = torch.cat([gu_d, slot_gu], dim=0)  # [E + 1, H, 2 * padded]
        down_all = torch.cat([down[0, :, cols, :], shared_down[cols].unsqueeze(0)], dim=0)  # [E + 1, local, H]
        act = torch.nn.functional.silu(x @ gu_all[:, :, :padded]) * (x @ gu_all[:, :, padded:])  # [E + 1, T, padded]
        partial = torch.einsum("etp,eph->eth", act[:, :, :local], down_all)  # [E + 1, T, H]
        out += torch.einsum("te,eth->th", dense, partial)  # routing weights incl. the constant 1.0 for slot E
    # fp32 on both sides; the tolerance only absorbs the different summation order (per-device partials vs HF loops)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

    # and without the always-on column the sum is the routed part only: the shared slot really is what column E adds
    dense[:, E] = 0.0
    with torch.no_grad():
        routed_only = experts_ref(x, idx, w)
    out_routed = torch.zeros(T, H)
    for d in range(tp):
        cols = slice(d * local, (d + 1) * local)
        gu_d = fused_gu[0, :, :, d * 2 * padded : (d + 1) * 2 * padded]
        act = torch.nn.functional.silu(x @ gu_d[:, :, :padded]) * (x @ gu_d[:, :, padded:])
        out_routed += torch.einsum(
            "te,eth->th", dense[:, :E], torch.einsum("etp,eph->eth", act[:, :, :local], down[0, :, cols, :])
        )
    torch.testing.assert_close(out_routed, routed_only, rtol=1e-3, atol=1e-3)
