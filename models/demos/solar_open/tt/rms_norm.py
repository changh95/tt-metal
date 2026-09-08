# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""T5-style RMSNorm (``x * rsqrt(mean(x^2) + eps) * weight``) with a width-sharded decode path.

Phase 2 (2026-09-07, device profile 2.1 / 5.4, perf-p0 lever b): ``ttnn.rms_norm`` with the default program config
runs a ``[1, 1, T <= 32, 4096]`` bf16 input on ONE core (57 us per call, 2 norms per layer = 5.5 ms of the 54 ms b1
step). The decode path below reshards the tile onto ``DECODE_NORM_GRID`` (8x4 cores, ``[32, 128]`` shards, 0.8 us),
runs ``LayerNormShardedMultiCoreProgramConfig`` (5.2 us; PCC vs a torch fp32 RMSNorm 0.999995 / 0.999994 at M = 1 /
32 against 0.999996 / 0.999963 for the default kernel, max abs err 0.024 / 0.035 vs 0.105 / 0.178,
tests/perf/test_config_candidates.py) and reshards back to the input's interleaved memory config (0.9 us).

The sharded path is taken only for a tensor whose TILE-PADDED shape is exactly one tile row of ``hidden_size``
columns, i.e. the model's decode inputs ``[1, 1, T <= 32, hidden]`` (``decode_norm_applies``). The worktree ladder's
failure (``TT_FATAL tensor_spec.cpp !shard_grid_fit_error: Shard height 32 must match physical height 1024``) came
from gating on ``x.shape[-2] <= 32`` alone: the rms_norm component test feeds the HF-shaped ``[32, 1, 4096]`` tensor,
whose 32 one-row batches pad to 32 tile rows (physical height 1024), so a ``[32, 128]`` width shard cannot cover it.
Such inputs, prefill inputs (T > 32 rows: the default kernel already spreads the rows over the cores), sharded inputs
and non-bf16 inputs keep the default kernel.
"""

from torch import nn

import ttnn
from models.demos.solar_open.config import MeshConfig, ModeConfig
from models.demos.solar_open.utils.general_utils import get_cache_file_name, get_default_num_links

# Core grid of the width-sharded decode norm: 32 cores x [32, hidden / 32] shards (Solar-Open: 128 = 4 tiles wide,
# block_w 4, subblock_w 4). Measured on P150 (per call, incl. the two reshards): 8x4 5.2 + 1.7 us, 8x8 5.4, 4x4 5.5,
# 8x2 5.8 vs 57.4 us on the default single core. None disables the sharded path (A/B switch).
DECODE_NORM_GRID = (8, 4)


def decode_norm_sharded_configs(hidden_size, grid, cores=None):
    """``(sharded_memory_config, program_config)`` of the width-sharded decode norm for a ``[.., 32, hidden_size]``
    tile, or ``(None, None)`` when the grid (``cores``, default ``DECODE_NORM_GRID``) is disabled, does not fit the
    device's compute grid ``grid``, or does not split ``hidden_size`` into tile-aligned shards."""
    cores = DECODE_NORM_GRID if cores is None else cores
    if cores is None or grid is None:
        return None, None
    core_x, core_y = cores
    num_cores = core_x * core_y
    if core_x > grid.x or core_y > grid.y or hidden_size % (num_cores * ttnn.TILE_SIZE) != 0:
        return None, None
    shard_width = hidden_size // num_cores
    block_w = shard_width // ttnn.TILE_SIZE
    memory_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, shard_width),
        core_grid=ttnn.CoreGrid(y=core_y, x=core_x),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[core_x, core_y],
        subblock_w=max(d for d in (4, 3, 2, 1) if block_w % d == 0),
        block_h=1,
        block_w=block_w,
        inplace=False,
    )
    return memory_config, program_config


def decode_norm_applies(x, hidden_size):
    """True when ``x`` can take the width-sharded decode norm: an interleaved bf16 TILE tensor whose tile-padded
    shape is exactly ONE tile row (32 padded rows over all leading dims) of ``hidden_size`` columns -- the model's
    decode inputs ``[1, 1, T <= 32, hidden]`` (also ``[1, 1, 1, hidden]``: padded to 32 rows). A ``[32, 1, hidden]``
    tensor pads to 32 x 32 rows and a ``[.., 33.., hidden]`` prefill tensor to 64+ rows: both keep the default kernel.
    """
    if x.dtype != ttnn.bfloat16 or x.memory_config().is_sharded():
        return False
    padded = tuple(x.padded_shape)
    if not padded or padded[-1] != hidden_size:
        return False
    rows = 1
    for d in padded[:-1]:
        rows *= d
    return rows == ttnn.TILE_SIZE


class RMSNorm(nn.Module):
    def __init__(
        self, mesh_device, hf_config, state_dict, tensor_cache_path=None, mesh_config=None, sharded_decode=True
    ):
        super().__init__()
        if state_dict:
            torch_weight = state_dict["weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        # Use MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
        self.is_distributed = False  # self.mesh_config.tp > 1
        self.tt_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mesh_config.shard_mapper(mesh_device, mesh_dims=(None, -2))
            if self.is_distributed
            else None,
        )

        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device
        self.hidden_size = hf_config.hidden_size

        # Width-sharded decode path (single-tile-row inputs); (None, None) keeps the default kernel everywhere.
        self.sharded_memory_config, self.sharded_program_config = (
            decode_norm_sharded_configs(hf_config.hidden_size, mesh_device.compute_with_storage_grid_size())
            if sharded_decode
            else (None, None)
        )

    def _forward_decode_sharded(self, x):
        """``x`` ``[.., T <= 32, hidden]`` interleaved -> width-sharded norm -> back to ``x``'s memory config."""
        x_sharded = ttnn.interleaved_to_sharded(x, self.sharded_memory_config)
        y_sharded = ttnn.rms_norm(
            x_sharded,
            weight=self.tt_weight,
            epsilon=self.eps,
            program_config=self.sharded_program_config,
            memory_config=self.sharded_memory_config,
        )
        x_sharded.deallocate(True)
        y = ttnn.sharded_to_interleaved(y_sharded, x.memory_config())
        y_sharded.deallocate(True)
        return y

    def forward(self, x):
        if self.is_distributed:
            activation_grid_bounding_box_size = x.memory_config().shard_spec.grid.bounding_box().grid_size()
            shard_height, shard_width = x.memory_config().shard_spec.shape
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=activation_grid_bounding_box_size,
                subblock_w=1,
                block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
                block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
                inplace=False,
            )
            # If the activation is sharded, we need to use an optimized rmsnorm

            tt_gathered_stats_memory_config = ttnn.create_sharded_memory_config(
                shape=[1, 1, 32, 32 * self.mesh_shape[1]],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
            # Run distributed rmsnorm part 1
            tt_stats = ttnn.rms_norm_pre_all_gather(x, program_config=program_config, dtype=ttnn.bfloat16)

            # AllGather stats
            tt_gathered_stats = ttnn.all_gather(
                tt_stats,
                dim=3,
                num_links=get_default_num_links(self.mesh_device),
                cluster_axis=1,
                mesh_device=self.mesh_device,
                memory_config=tt_gathered_stats_memory_config,
                topology=ttnn.Topology.Ring,
            )
            ttnn.deallocate(tt_stats)

            # Run distributed rmsnorm part 2
            tt_output = ttnn.rms_norm_post_all_gather(
                x,
                tt_gathered_stats,
                program_config=program_config,
                epsilon=self.eps,
                weight=self.tt_weight,
                dtype=ttnn.bfloat16,
                stats=tt_gathered_stats,
            )
            ttnn.deallocate(tt_gathered_stats)
            return tt_output
        if self.sharded_program_config is not None and decode_norm_applies(x, self.hidden_size):
            return self._forward_decode_sharded(x)
        return ttnn.rms_norm(
            x,
            weight=self.tt_weight,
            epsilon=self.eps,
        )
