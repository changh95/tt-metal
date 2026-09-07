# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only regressions for the expert parallelisation and program configs.

``ttnn.moe_routing_remap`` requires ``expert_parallel_size == mesh_shape[cluster_axis]``.
Two places have to hold that up: ``MeshConfig`` must not accept an EP that disagrees with
its own ``ep_axis``, and the decode expert path must forward the configured EP and axis
instead of hardcoded literals. The Solar-Open sparse-matmul grids must also resolve to the
exact-fill rectangles the design assumes. None of this needs a device to check.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models.demos.solar_open.config import MeshConfig, ModeConfig, mesh_1x8, mesh_2x4, mesh_4x4, mesh_4x8
from models.demos.solar_open.tt.expert_configs import SolarOpenProgramConfig, solar_open_program_config
from models.demos.solar_open.tt.experts.config import ExpertConfig
from models.demos.solar_open.tt.experts.decode import decode_forward


class _RemapCallCaptured(Exception):
    """Unwinds decode_forward once the call under test has been recorded."""

    def __init__(self, args, kwargs):
        super().__init__("captured moe_routing_remap call")
        self.captured_args = args
        self.captured_kwargs = kwargs


@pytest.mark.parametrize(
    "mesh_config_factory, expected_ep",
    [
        # 2x4 is the mesh the old hardcoded (nnz=4, ep=4, axis=0) call broke on: it declared
        # 4 partitions while running on 2 rows, and now trips the live mesh-axis TT_FATAL.
        (mesh_2x4, 2),
        # 4x8/4x4 are where the hardcoded literals happened to be right; asserting them here
        # pins that this change is a no-op on the multi-row meshes.
        (mesh_4x8, 4),
        (mesh_4x4, 4),
    ],
    ids=["mesh_2x4", "mesh_4x8", "mesh_4x4"],
)
def test_decode_forward_forwards_configured_ep_and_axis(mesh_config_factory, expected_ep):
    """decode_forward must pass the configured EP and ep_axis to moe_routing_remap.

    Reverting either argument to a literal leaves the operator-level tests green, so assert
    the arguments directly. ttnn is patched out, which stops the forward at the call we care
    about and keeps this runnable without a device.
    """
    mesh_config = mesh_config_factory()
    assert expected_ep == mesh_config.mesh_shape[mesh_config.ep_axis]

    config = ExpertConfig(
        intermediate_size=64,
        num_experts=32,
        hidden_size=64,
        num_experts_per_tok=4,
    )
    reshaped_sparsity = object()

    with patch("models.demos.solar_open.tt.experts.decode.ttnn") as mock_ttnn:
        mock_ttnn.reshape.return_value = reshaped_sparsity

        def capture(*args, **kwargs):
            raise _RemapCallCaptured(args, kwargs)

        mock_ttnn.moe_routing_remap.side_effect = capture

        with pytest.raises(_RemapCallCaptured) as exc_info:  # allow-pytest.raises: control-flow sentinel
            decode_forward(
                hidden_states=SimpleNamespace(shape=[1, 1, 1, config.hidden_size]),
                routing_weights=MagicMock(),
                weights=MagicMock(),
                config=config,
                mesh_config=mesh_config,
                mesh_device=MagicMock(),
                ccl_manager=MagicMock(),
                program_config=MagicMock(),
            )

    assert exc_info.value.captured_args == (
        reshaped_sparsity,
        config.num_experts_per_tok,
        expected_ep,
        mesh_config.ep_axis,
    )
    assert exc_info.value.captured_kwargs == {}


def test_mesh_config_rejects_ep_not_matching_ep_axis(expect_error):
    """EP must agree with its own axis at construction, not as a TT_FATAL mid-forward.

    tp x dp x ep == total_devices and tp <= mesh_shape[tp_axis] both pass here, so before
    the EP check this config built fine and then hard-failed inside moe_routing_remap on the
    first decode step.
    """
    with expect_error(ValueError, r"decode: EP\(2\) != mesh_0_size\(4\)"):
        MeshConfig((4, 8), decode=ModeConfig(tp=8, ep=2))


@pytest.mark.parametrize(
    "mesh_config_factory",
    [mesh_1x8, mesh_2x4, mesh_4x4, mesh_4x8],
    ids=["mesh_1x8", "mesh_2x4", "mesh_4x4", "mesh_4x8"],
)
def test_mesh_config_factories_satisfy_ep_axis_constraint(mesh_config_factory):
    """The EP check must not reject any shipped config.

    Prefill runs EP=1 on multi-row meshes and never reaches the remap, so EP=1 stays
    unconstrained; only EP>1 has to match the axis extent.
    """
    mesh_config = mesh_config_factory()
    ep_dim_size = mesh_config.mesh_shape[mesh_config.ep_axis]

    for mode_config in (mesh_config.decode, mesh_config.prefill):
        assert mode_config.ep == 1 or mode_config.ep == ep_dim_size


# Solar-Open-100B per-device shapes at TP=8: H=4096 (Kt=128), Ip=160 (Kt=5), fused gate|up N=320 (Nt=10),
# down N=4096 (Nt=128).
_H, _IP, _GATE_UP_N, _DOWN_N = 4096, 160, 320, 4096


def _grid(cfg):
    g = cfg.compute_with_storage_grid_size
    return (g.x, g.y), cfg.per_core_N, cfg.in0_block_w, cfg.out_subblock_w


@pytest.mark.parametrize(
    "call, expected",
    [
        # (grid, per_core_N, in0_block_w, out_subblock_w); every grid is an exact fill of ceil(Nt / per_core_N) blocks
        (lambda pc: pc.get_decode_gate_up_config(32, _GATE_UP_N, k=_H), ((5, 2), 1, 32, 1)),
        (lambda pc: pc.get_decode_down_config(1, _DOWN_N, k=_IP), ((8, 4), 4, 5, 1)),
        (lambda pc: pc.get_decode_down_config(8, _DOWN_N, k=_IP), ((8, 4), 4, 5, 1)),
        (lambda pc: pc.get_decode_down_config(16, _DOWN_N, k=_IP), ((8, 8), 2, 5, 1)),
        (lambda pc: pc.get_decode_down_config(32, _DOWN_N, k=_IP), ((8, 8), 2, 5, 1)),
        (lambda pc: pc.get_prefill_gate_up_config(1024, _GATE_UP_N, k=_H), ((5, 2), 1, 32, 1)),
        (lambda pc: pc.get_prefill_down_config(1024, _DOWN_N, k=_IP), ((8, 8), 2, 5, 1)),
    ],
    ids=[
        "decode_gate_up",
        "decode_down_1_user",
        "decode_down_8_users_single_grid",
        "decode_down_16_users_batched_grid",
        "decode_down_32_users_batched_grid",
        "prefill_gate_up_sparse",
        "prefill_down_sparse",
    ],
)
def test_solar_open_program_config_grids_resolve_exactly(call, expected):
    """The shipped SolarOpenProgramConfig values must be the identity under ProgramConfig._build_matmul_config
    (no silent grid shrink, no in0_block_w snap): Nt=10 on 5x2 x 1 tile, Nt=128 on 8x4 x 4 tiles or 8x8 x 2 tiles,
    in0_block_w 32 | Kt=128 and 5 | Kt=5."""
    assert _grid(call(SolarOpenProgramConfig())) == expected


def test_solar_open_program_config_per_core_m_tracks_tokens():
    pc = SolarOpenProgramConfig()
    assert pc.get_decode_gate_up_config(32, _GATE_UP_N, k=_H).per_core_M == 1
    assert pc.get_prefill_gate_up_config(1024, _GATE_UP_N, k=_H).per_core_M == 32
    assert pc.dense_grid_max_width == 12 and pc.dense_bmm_max_tokens == 256
    assert pc.sequence_chunk_size == 4096 and pc.base_down_split_size == 1024


@pytest.mark.parametrize(
    "grid_xy, batched",
    [((13, 10), (8, 8)), ((11, 10), (8, 8)), ((8, 8), (8, 8)), ((7, 8), None), ((8, 7), None)],
    ids=["bh_13x10", "bh_11x10", "wh_8x8", "narrow_7x8", "short_8x7"],
)
def test_solar_open_program_config_factory_selects_batched_down_grid(grid_xy, batched):
    """Compute grids of at least 8x8 keep the 64-core batched down grid; smaller grids drop it (single grid for all
    steps, itself shrunk by the builder if needed)."""
    mesh_device = MagicMock()
    mesh_device.compute_with_storage_grid_size.return_value = SimpleNamespace(x=grid_xy[0], y=grid_xy[1])
    pc = solar_open_program_config(mesh_device)
    assert pc.decode_down_cores_batched == batched
    assert pc.decode_down_cores == (8, 4)


def test_program_config_rejects_bad_dense_knobs(expect_error):
    with expect_error(ValueError, "dense_bmm_max_tokens"):
        SolarOpenProgramConfig(dense_bmm_max_tokens=100)
    with expect_error(ValueError, "dense_grid_max_width"):
        SolarOpenProgramConfig(dense_grid_max_width=0)
