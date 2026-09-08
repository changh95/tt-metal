# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""1D in0-multicast matmul program configs for the small dense linears of the step (shared expert, router).

Phase 2 (2026-09-07, device profile section 5.4): for ``[M <= 32, 4096] x [4096, 128..160]`` ttnn's auto choice is a
4-5 core ``MatmulMultiCoreProgramConfig`` (one output tile per core, no multicast, K streamed tile by tile: 90 us per
linear on P150) and for ``[M, 160] x [160, 4096]`` a 1D config with ``in0_block_w`` 1. An explicit
``MatmulMultiCoreReuseMultiCast1DProgramConfig`` -- in0 multicast to the cores that own the N tiles, K in blocks of
``in0_block_w`` tiles, one ``[Mt x per_core_N]`` output block per core -- runs the same math in 11 us (gate / up /
router) and 3 us (down). The numerics differ from the auto configs only through the accumulation order / partial
spills of the K blocks (same fidelity), PROVIDED the call site passes its ``compute_kernel_config`` explicitly: ttnn
raises the auto fidelity of a bf16 x bfp8 matmul to HiFi2 only when neither a program config nor a core grid is given
(``matmul_device_operation.cpp::create_matmul_attributes``) and falls back to LoFi otherwise.
"""

import math

import ttnn


def grid_fits(cores, grid) -> bool:
    """True when the ``(x, y)`` core grid fits the device's compute grid (``CoreCoord`` or anything with ``.x``/``.y``)."""
    return grid is not None and cores[0] <= grid.x and cores[1] <= grid.y


def mcast_1d_linear_config(
    cores,
    rows: int,
    n: int,
    k: int,
    in0_block_w: int,
    out_subblock_w: int = 1,
    fp32_dest_acc: bool = False,
) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    """1D in0-multicast config for ``[.., rows, k] x [k, n]`` with one ``[Mt x per_core_N]`` output block per core.

    The N tiles are split evenly over the ``cores`` grid (raises when they do not divide); ``in0_block_w`` snaps down
    to a divisor of Kt and ``out_subblock_w`` to a divisor of per_core_N within the destination-register budget (8
    tiles, 4 with fp32 accumulation; ``out_subblock_h`` is 1). ``rows`` is the logical row count (1..32 -> Mt 1).
    """
    core_x, core_y = cores
    num_cores = core_x * core_y
    if num_cores < 1:
        raise ValueError(f"1D linear config: need at least one core, got {cores}")
    Mt = max(1, math.ceil(rows / ttnn.TILE_SIZE))
    Kt = math.ceil(k / ttnn.TILE_SIZE)
    Nt = math.ceil(n / ttnn.TILE_SIZE)
    if Nt % num_cores != 0:
        raise ValueError(
            f"1D linear config: N = {n} ({Nt} tiles) must be divisible by the {core_x}x{core_y} = {num_cores} cores"
        )
    per_core_N = Nt // num_cores
    if Kt % in0_block_w != 0:
        in0_block_w = max(d for d in range(1, in0_block_w + 1) if Kt % d == 0)
    max_subblock_w = 4 if fp32_dest_acc else 8
    out_subblock_w = max(d for d in range(1, min(out_subblock_w, max_subblock_w) + 1) if per_core_N % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=Mt,
        out_block_w=per_core_N,
        per_core_M=Mt,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )
