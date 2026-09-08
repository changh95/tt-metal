# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Solar-Open expert program configurations."""

import os
from dataclasses import dataclass, replace

from models.demos.solar_open.tt.experts.config import MinimalMatmulBlocking, ProgramConfig

# Phase-3 prefill expert matmul blockings (tests/perf/test_prefill_matmul_candidates.py on P150, 2026-09-08; per-op
# burst times of the isolated ops on one device, PCC vs the fp32 matmul of the device-rounded operands):
# - per-expert fused gate|up [1024, 4096] x [4096, 320] -> M_block 3 / K_block 16 / N_block 2, subblock 3x2 on 11x5 cores:
#   46 us at 58 TFLOP/s vs 81 us for ttnn.linear's auto config (2D mcast, in0_block_w 4); PCC 0.999902 vs 0.999917
#   (max abs error 0.129 vs 0.152: the same bfp8-output floor, the perf-p1 equality margin is 3e-5).
DENSE_EXPERT_GATE_UP_MINIMAL = MinimalMatmulBlocking(cores=(11, 5), k_block=16, subblock=(3, 2))
# - the same linear with K blocks of 8 tiles on the full 11x10 grid (M_block 3 / N_block 1, subblock 3x1): 55 us and
#   PCC 0.999929 (> the auto config): the fallback if the teacher-forced floors move with the default (preset
#   ``gate_up_alt``). Do NOT use an 8-wide grid for this linear: 8x10 M4 K16 N1 measured 155 us.
DENSE_EXPERT_GATE_UP_MINIMAL_ALT = MinimalMatmulBlocking(cores=(11, 10), k_block=8, subblock=(3, 1))
# - hot group's K-concatenated down [1024, n_hot * 160] x [n_hot * 160, 4096] -> M_block 4 / K_block 5 (one expert's Ip
#   per K block) / N_block 12, subblock 4x2 on 11x10 cores: 50 / 74 / 117 us at n_hot = 4 / 8 / 15 (+ the bf16 act
#   concat 25 / 28 / 36 us) vs the batched matmul + fast_reduce_nc 109 + 64 / 213 + 117 / 394 + 205 us; PCC 0.999950 /
#   0.999950 / 0.999942 vs 0.999931 / 0.999933 / 0.999924 (the sum over the hot experts moves into the accumulation).
HOT_DOWN_KCONCAT_MINIMAL = MinimalMatmulBlocking(cores=(11, 10), k_block=5, subblock=(4, 2))

# A/B presets of the phase-3 prefill knobs for the accuracy / perf gates (SOLAR_OPEN_PREFILL_EXPERT_MM=<name>):
# ``tuned`` (default) = the shipped values, ``phase2`` = the phase-2 forms (auto linear, bmm + reduce, out_subblock_h 1),
# ``gate_up_alt`` = the better-numerics gate|up blocking, ``bfp8_act`` = tuned + the bfp8 activation broadcast of the
# dense 128-token path (a numerics switch: teacher-forced test only).
PREFILL_EXPERT_MM_ENV = "SOLAR_OPEN_PREFILL_EXPERT_MM"
PREFILL_EXPERT_MM_PRESETS = {
    "tuned": {},
    "phase2": dict(dense_expert_gate_up_minimal=None, hot_down_kconcat_minimal=None, dense_down_max_subblock_h=1),
    "gate_up_alt": dict(dense_expert_gate_up_minimal=DENSE_EXPERT_GATE_UP_MINIMAL_ALT),
    "bfp8_act": dict(dense_activation_bfp8=True),
}


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
    # Phase 3 (2026-09-08, one-device micro-benchmarks, see the module constants): the dense down's [Mt x 2] output
    # subblock (bit-identical, -8 % at 96-128 rows), the per-expert fused gate|up of the sorted hot group / per-expert
    # loop as minimal_matmul (81 -> 46 us per expert at 1024 tokens) and the hot down as one K-concatenated
    # minimal_matmul over bf16 GLU pieces (-64 / -159 / -308 us per 1024-token split at 4 / 8 / 15 hot experts, better
    # PCC). The bfp8 activation broadcast of the dense 128-token path (-270 us + -37 us per layer, PCC of the gate|up
    # bmm 0.999896 vs 0.999917) is a numerics change and stays off (SOLAR_OPEN_PREFILL_EXPERT_MM=bfp8_act to A/B it).
    dense_down_max_subblock_h: int = 4
    dense_expert_gate_up_minimal: MinimalMatmulBlocking | None = DENSE_EXPERT_GATE_UP_MINIMAL
    hot_down_kconcat_minimal: MinimalMatmulBlocking | None = HOT_DOWN_KCONCAT_MINIMAL
    dense_activation_bfp8: bool = False
    # Trace-safe splits (phase 3a(2), design_traced_prefill.md): the traced prefill buckets of tt/model_config.py
    # (TRACE_PREFILL_SEQ_LENS: 128 tokens on P150x8) run the dense bmm path, so nothing is listed. A 1K / 2K / 4K bucket
    # (NO-GO in phase 3a: every trace-safe MoE formulation costs more per 1024-token split than a trace saves in host
    # time) would need its 1024-token split here -- ModelArgs refuses a traced length whose splits are not trace-safe --
    # which routes those splits to the static per-expert loop (~2.5x the sorted path's cost) until design (C) lands.
    trace_safe_split_lens: tuple[int, ...] = ()


def prefill_expert_mm_overrides(preset: str | None = None) -> dict:
    """Field overrides of a PREFILL_EXPERT_MM_PRESETS entry: ``preset`` by name, or (None) the one named by the
    SOLAR_OPEN_PREFILL_EXPERT_MM environment variable (unset / empty = ``tuned`` = no overrides)."""
    name = (os.getenv(PREFILL_EXPERT_MM_ENV, "") if preset is None else preset).strip() or "tuned"
    if name not in PREFILL_EXPERT_MM_PRESETS:
        raise ValueError(
            f"{PREFILL_EXPERT_MM_ENV}={name!r}: unknown prefill expert matmul preset, expected one of "
            f"{sorted(PREFILL_EXPERT_MM_PRESETS)}"
        )
    return dict(PREFILL_EXPERT_MM_PRESETS[name])


def solar_open_program_config(mesh_device) -> SolarOpenProgramConfig:
    """Solar-Open expert program config for the device's compute grid.

    Blackhole (13x10 / 11x10) and Wormhole (8x8) grids both fit the 8x8 batched down grid and the 8x8 dense
    down grid; anything smaller falls back to the single-user 8x4 grid (itself shrunk by the builder if needed)
    for every step and to the auto config for the dense prefill down bmm. The phase-3 minimal_matmul blockings need
    their 11-wide grids (Blackhole); on a narrower grid they drop to the auto forms. SOLAR_OPEN_PREFILL_EXPERT_MM
    selects an A/B preset of the phase-3 knobs (prefill_expert_mm_overrides).
    """
    grid = mesh_device.compute_with_storage_grid_size()
    overrides = prefill_expert_mm_overrides()
    if not (grid.x >= 8 and grid.y >= 8):
        overrides.update(decode_down_cores_batched=None, dense_down_cores=None)
    pc = SolarOpenProgramConfig(**overrides)
    drop = {
        name: None
        for name in ("dense_expert_gate_up_minimal", "hot_down_kconcat_minimal")
        if getattr(pc, name) is not None and not getattr(pc, name).fits(grid)
    }
    return replace(pc, **drop) if drop else pc
