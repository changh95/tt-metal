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
    with work to fill the multicast rectangle exactly); the values are performance knobs, never a correctness knob
    (the numerics of every value here were re-validated with the component tests and the teacher-forced accuracy
    test, see the README's "Recorded baselines"). Phase-2 tuning (2026-09-07) from the tracy device profile of the
    real-weight layer: gate|up in0_block_w 128 and the widest legal down out_subblock_w per grid.
    """

    # Decode. The fused gate|up sparse_matmul has N = 10 tiles per device: 10 cores x 1 tile (any larger
    # request shrinks to this). in0_block_w 128 = Kt: the whole K as ONE block (was 32 = 4 K-blocks); measured on
    # P150 (2026-09-07, per layer): nnz 8 (batch 1) 158.7 -> 149.6 us, nnz 112 (batch 32) 1162 -> 961 us (-17 %).
    # L1: 128 bf16 in0 tiles (256 KB) + 128 bfp8 in1 tiles per core, double-buffered.
    decode_gate_up_cores: tuple[int, int] = (5, 2)
    decode_gate_up_in0_block_w: int = 128
    # Single-user down projection (N = 128 tiles): 32 cores x per_core_N 4. Kt = 5 is prime, so only 1 or 5
    # divide it; 5 keeps one K-block. out_subblock_w 4 = per_core_N (one compute pass per output block; measured
    # nnz 8: 178.7 -> 151.9 us incl. the output zero-fill).
    decode_down_cores: tuple[int, int] = (8, 4)
    decode_down_in0_block_w: int = 5
    decode_down_subblock_w: int = 4
    # Multi-user steps (>= 16 users): 64 cores x 2 tiles. 128 tiles have no exact-fill rectangle with
    # w <= 13, h <= 10 at one tile per core; 8x8x2 is the widest exact fill. Set to None by
    # solar_open_program_config() when the compute grid is smaller than 8x8. out_subblock_w 2 = per_core_N
    # (measured nnz 112: 437.9 -> 307.7 us, nnz 8: 200.1 -> 190.1).
    decode_down_cores_batched: tuple[int, int] | None = (8, 8)
    decode_down_batched_subblock_w: int = 2
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
    # Dense down bmm [1, E, S, 160] x [1, E, 160, 4096] as a 1D-multicast matmul over 8x8 cores x 2 tiles with the
    # whole K (5 tiles) as one block (get_dense_down_config), used by experts/prefill.py for the dense tail and
    # the sorted path's cold experts at S <= dense_bmm_max_tokens rows. Measured on P150: 396 / 1148 / 1716 us
    # auto (in0_block_w 1) -> 311 / 642 / 1096 us at S = 32 / 128 / 256, PCC vs fp32 at the bfp8 floor; ~-0.5 ms
    # per layer per 128-token split (-24 ms TTFT@128). None restores the auto config (A/B switch).
    dense_down_cores: tuple[int, int] | None = (8, 8)


def solar_open_program_config(mesh_device) -> SolarOpenProgramConfig:
    """Solar-Open expert program config for the device's compute grid.

    Blackhole (13x10 / 11x10) and Wormhole (8x8) grids both fit the 8x8 batched down grid and the 8x8 dense
    down grid; anything smaller falls back to the single-user 8x4 grid (itself shrunk by the builder if needed)
    for every step and to the auto config for the dense prefill down bmm.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x >= 8 and grid.y >= 8:
        return SolarOpenProgramConfig()
    return SolarOpenProgramConfig(decode_down_cores_batched=None, dense_down_cores=None)
