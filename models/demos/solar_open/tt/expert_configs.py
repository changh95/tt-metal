# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Solar-Open expert program configurations."""

from dataclasses import dataclass

from models.demos.solar_open.tt.experts.config import ProgramConfig


@dataclass
class SolarOpenProgramConfig(ProgramConfig):
    """Solar-Open-100B experts at TP=8: H=4096 (Kt=128), Ip=160 (5 tiles), fused gate|up N=320 (10 tiles),
    down N=4096 (128 tiles).

    All grids resolve exactly in ProgramConfig._build_matmul_config (the sparse-matmul factory requires the cores
    with work to fill the multicast rectangle exactly); the values are performance starting points, never a
    correctness knob. out_subblock_w stays 1 (the only validated kernel variant); wider subblocks for the down
    projection (per_core_N 4 / 2 allow 2 or 4) are a tuning experiment.
    """

    # Decode. The fused gate|up sparse_matmul has N = 10 tiles per device: 10 cores x 1 tile (any larger
    # request shrinks to this). in0_block_w 32 divides Kt = 128 -> 4 K-blocks.
    decode_gate_up_cores: tuple[int, int] = (5, 2)
    decode_gate_up_in0_block_w: int = 32
    # Single-user down projection (N = 128 tiles): 32 cores x per_core_N 4. Kt = 5 is prime, so only 1 or 5
    # divide it; 5 keeps one K-block.
    decode_down_cores: tuple[int, int] = (8, 4)
    decode_down_in0_block_w: int = 5
    # Multi-user steps (>= 16 users): 64 cores x 2 tiles. 128 tiles have no exact-fill rectangle with
    # w <= 13, h <= 10 at one tile per core; 8x8x2 is the widest exact fill. Set to None by
    # solar_open_program_config() when the compute grid is smaller than 8x8.
    decode_down_cores_batched: tuple[int, int] | None = (8, 8)
    decode_down_batched_min_tokens: int = 16

    # Prefill sparse path (EP > 1 only; unreachable on a 1x8 mesh, kept for multi-row meshes)
    prefill_gate_up_cores: tuple[int, int] = (5, 2)
    prefill_gate_up_in0_block_w: int = 32
    prefill_down_cores: tuple[int, int] = (8, 8)
    prefill_down_in0_block_w: int = 5

    # Memory
    sequence_chunk_size: int = 4 * 1024
    base_down_split_size: int = 1024

    # Dense EP=1 prefill: 12x10 = 120 cores -> 128 experts in 2 bmm rounds (13 wide = 130 cores, one round, is a
    # tuning experiment); the one-launch bmm covers splits up to 256 tokens (per-core block M x 10 tiles; 512 is
    # plausible for N = 10 tiles).
    dense_grid_max_width: int = 12
    dense_bmm_max_tokens: int = 256


def solar_open_program_config(mesh_device) -> SolarOpenProgramConfig:
    """Solar-Open expert program config for the device's compute grid.

    Blackhole (13x10 / 11x10) and Wormhole (8x8) grids both fit the 8x8 batched down grid; anything smaller
    falls back to the single-user 8x4 grid (itself shrunk by the builder if needed) for every step.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x >= 8 and grid.y >= 8:
        return SolarOpenProgramConfig()
    return SolarOpenProgramConfig(decode_down_cores_batched=None)
