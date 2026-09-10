# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Op-level tests for the indexed/gather mode of ttnn.sparse_matmul.

The new optional `indices` operand is exercised with a hard-coded, NON-MONOTONIC id list, proving
(1) it compiles + runs without hanging (kernel lockstep), (2) it matches a torch reference, (3) the
`is_input_a_sparse=True` compact-A path, (4) bf4 weight addressing by arbitrary group id, (5) that a
program-cache hit re-dispatches a *different* indices buffer correctly, (6) that the dense
sparsity-scan path is value-identical when `indices` is absent, and (7) that the operand's host-side
contract is enforced.

(8) FILL skip (phase 3d): the op no longer runs its `ttnn.zeros_like` FILL pass on the OP-ALLOCATED compact output
of a legacy (non-EGP) indexed call -- the in1 writer visits every entry and every core writes its whole block, so
every output tile is written by the kernel (`sparse_matmul_legacy_indexed_kernel_writes_whole_output`). Those tests
run the op so that its output lands on memory just FREED by a tensor poisoned with values no valid product yields
(bfp8: 12345.0, bf16: NaN) and require torch.equal with the result written into a pre-poisoned optional output (that
path keeps its FILL, unchanged) plus a torch fp32 check. `TT_SPARSE_MATMUL_INDEXED_SKIP_FILL=0` (fresh process)
restores the FILL; the whole file must pass in both arms.
"""

import math
import os

from loguru import logger
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics


def _sparse_pc(m, n, tile_h, tile_w):
    """A 1D-optimized program config that spreads N across a core grid with per_core_N=1, mirroring
    the production gather caller (an MoE expert projection)."""
    nt = int(math.ceil(n / tile_w))
    cx, cy = 1, 1
    for d in range(min(10, nt), 0, -1):
        if nt % d == 0 and nt // d <= 10:
            cx, cy = nt // d, d
            break
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=max(tile_h, m) // tile_h,
        per_core_N=1,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _make_indices(active_ids, device):
    """[1,1,1,num_active] UINT16 ROW_MAJOR device tensor of group ids (compact-output slot order)."""
    t = torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, 1, len(active_ids))
    return ttnn.from_torch(t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _make_sparsity(active_ids, num_experts, device):
    """[1,1,1,E] bf16 ROW_MAJOR mask, nonzero exactly at the active group ids.

    In indexed mode neither kernel reads this tensor -- the indexed loop visits only active groups, so
    there is no validity scan -- but `sparsity` remains a required positional operand of the op.
    """
    s = torch.zeros(1, 1, 1, num_experts, dtype=torch.float32)
    for e in active_ids:
        s[0, 0, 0, e] = 1.0
    return ttnn.from_torch(s.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


@pytest.mark.parametrize("num_experts", [16, 256])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_gather_gate_up(device, num_experts, in1_dtype):
    """gate_up-like: A is dense/broadcast [1,1,1,K], B = expert weights [1,E,K,N], is_input_b_sparse.
    Compact output slot i must equal in0 @ in1[active_ids[i]]."""
    torch.manual_seed(0)
    tile_h, tile_w = 32, 32
    m, k, n = 32, 128, 256
    # A non-monotonic active set (the bf4-addressing-by-arbitrary-index check).
    active_ids = [num_experts - 1, 5, num_experts - 2, 12, 1, num_experts // 2, 7, 0][: min(8, num_experts)]
    num_active = len(active_ids)

    in0 = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    sparsity_t = _make_sparsity(active_ids, num_experts, device)
    indices_t = _make_indices(active_ids, device)

    out_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        indices=indices_t,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=_sparse_pc(m, n, tile_h, tile_w),
    )
    out = ttnn.to_torch(out_t).reshape(num_active, m, n)
    logger.info(f"gate_up gather out shape {tuple(out_t.shape)} -> compact [{num_active}, {m}, {n}]")

    in1_f = in1.float()
    for i, e in enumerate(active_ids):
        ref = torch.matmul(in0[0, 0].float(), in1_f[0, e])
        assert_numeric_metrics(
            ref, out[i].float(), atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.01 * k, pcc_threshold=0.99
        )


@pytest.mark.parametrize("num_experts", [16, 256])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_gather_down(device, num_experts, in1_dtype):
    """down-like: A is COMPACT [1,num_active,1,I] (one row per active expert), B = [1,E,I,H], both
    sparse. Compact output slot i must equal A[i] @ B[active_ids[i]] (A indexed by i, B by id)."""
    torch.manual_seed(1)
    tile_h, tile_w = 32, 32
    m, k, n = 32, 128, 256  # m=intermediate-row tile, k=I, n=H
    active_ids = [num_experts - 1, 3, num_experts - 2, 9, 2, num_experts // 2, 11, 0][: min(8, num_experts)]
    num_active = len(active_ids)

    a_compact = torch.randn((1, num_active, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    a_t = ttnn.from_torch(a_compact, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    sparsity_t = _make_sparsity(active_ids, num_experts, device)
    indices_t = _make_indices(active_ids, device)

    out_t = ttnn.sparse_matmul(
        a_t,
        in1_t,
        sparsity=sparsity_t,
        indices=indices_t,
        is_input_a_sparse=True,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=_sparse_pc(m, n, tile_h, tile_w),
    )
    out = ttnn.to_torch(out_t).reshape(num_active, m, n)
    logger.info(f"down gather out shape {tuple(out_t.shape)} -> compact [{num_active}, {m}, {n}]")

    in1_f = in1.float()
    for i, e in enumerate(active_ids):
        ref = torch.matmul(a_compact[0, i].float(), in1_f[0, e])
        assert_numeric_metrics(
            ref, out[i].float(), atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.01 * k, pcc_threshold=0.99
        )


def test_gather_program_cache_reuses_with_new_indices(device):
    """A cached program must re-dispatch a *different* indices buffer.

    The indices address is patched in override_runtime_arguments; if that patch were missing or wrong,
    the second call would silently gather the first call's groups. Run the same configuration twice
    with two equal-shaped, separately allocated, non-monotonic id lists and require (a) no new program
    cache entry and (b) each output to match its own reference."""
    torch.manual_seed(3)
    tile_h, tile_w = 32, 32
    m, k, n = 32, 128, 256
    num_experts = 16
    active_ids_a = [15, 2, 9, 0, 13, 4, 7, 11]
    active_ids_b = [1, 14, 6, 12, 3, 10, 5, 8]
    num_active = len(active_ids_a)
    assert len(active_ids_b) == num_active
    assert active_ids_a != active_ids_b

    in0 = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)
    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = _sparse_pc(m, n, tile_h, tile_w)

    # Allocate both id lists (and both masks) up front so they land at distinct addresses.
    sparsity_a = _make_sparsity(active_ids_a, num_experts, device)
    indices_a = _make_indices(active_ids_a, device)
    sparsity_b = _make_sparsity(active_ids_b, num_experts, device)
    indices_b = _make_indices(active_ids_b, device)
    assert indices_a.buffer_address() != indices_b.buffer_address()

    def run(sparsity_t, indices_t):
        out_t = ttnn.sparse_matmul(
            in0_t,
            in1_t,
            sparsity=sparsity_t,
            indices=indices_t,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
        )
        return ttnn.to_torch(out_t).reshape(num_active, m, n)

    out_a = run(sparsity_a, indices_a)
    cache_entries_after_first = device.num_program_cache_entries()
    out_b = run(sparsity_b, indices_b)
    cache_entries_after_second = device.num_program_cache_entries()
    assert (
        cache_entries_after_second == cache_entries_after_first
    ), "the second indexed call should reuse the cached program"

    in1_f = in1.float()
    for out, active_ids, tag in ((out_a, active_ids_a, "first"), (out_b, active_ids_b, "second")):
        for i, e in enumerate(active_ids):
            ref = torch.matmul(in0[0, 0].float(), in1_f[0, e])
            logger.info(f"{tag} call, compact slot {i} -> group {e}")
            assert_numeric_metrics(
                ref, out[i].float(), atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.01 * k, pcc_threshold=0.99
            )


def test_indices_absent_is_unchanged(device):
    """With no `indices`, the op must produce the unchanged dense [.., E, M, N] result: the active
    slots hold their products and the skipped slots stay zero-filled. Guards the claim that the
    gather operand is the sole trigger and the legacy path is untouched."""
    torch.manual_seed(2)
    tile_h, tile_w = 32, 32
    m, k, n = 32, 128, 256
    num_experts = 16
    active_ids = [3, 7, 11, 0]

    in0 = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)
    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    sparsity_t = _make_sparsity(active_ids, num_experts, device)

    out_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        nnz=len(active_ids),
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=_sparse_pc(m, n, tile_h, tile_w),
    )
    # Dense expert axis (= E), not compact.
    assert out_t.shape[-3] == num_experts, f"expected dense E={num_experts} axis, got shape {tuple(out_t.shape)}"

    out = ttnn.to_torch(out_t).reshape(num_experts, m, n).float()
    in1_f = in1.float()
    for e in range(num_experts):
        if e in active_ids:
            ref = torch.matmul(in0[0, 0].float(), in1_f[0, e])
            assert_numeric_metrics(
                ref, out[e], atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.01 * k, pcc_threshold=0.99
            )
        else:
            assert torch.count_nonzero(out[e]) == 0, f"inactive expert slot {e} must stay zero-filled"


####################################################################################################
# Host-side contract on the `indices` operand
####################################################################################################


def _contract_inputs(device, num_experts=16, m=32, k=128, n=256):
    torch.manual_seed(4)
    in0 = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)
    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    active_ids = [3, 7, 11, 0]
    sparsity_t = _make_sparsity(active_ids, num_experts, device)
    return in0_t, in1_t, sparsity_t, active_ids, _sparse_pc(m, n, 32, 32), (m, k, n, num_experts)


def _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc, **kwargs):
    return ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        indices=indices_t,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
        **kwargs,
    )


def test_indices_rejects_multi_stick_shape(device, expect_error):
    """The id list is fetched with a single page-0 read, so it must live in one ROW_MAJOR stick.
    A same-volume [1,1,N,1] tensor is N separate one-element pages."""
    in0_t, in1_t, sparsity_t, active_ids, pc, _ = _contract_inputs(device)
    t = torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, len(active_ids), 1)
    indices_t = ttnn.from_torch(t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with expect_error(RuntimeError, "single ROW_MAJOR stick"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc)


def test_indices_rejects_host_tensor(device, expect_error):
    """A host id list must never reach the program factory, which dispatches its buffer address.

    is_allocated() alone is true for host tensors, so the op validates device storage explicitly; the
    generic device-operation launch guard happens to reject this case one layer earlier, which is why
    the message below is the framework's rather than sparse_matmul's. The op's own check still covers
    what the framework does not: an indices tensor resident on a *different* device than the inputs.
    """
    in0_t, in1_t, sparsity_t, active_ids, pc, _ = _contract_inputs(device)
    t = torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, 1, len(active_ids))
    indices_t = ttnn.from_torch(t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)  # no device=

    with expect_error(RuntimeError, "Device Operations expect device tensors as inputs"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc)


def test_indices_rejects_wrong_dtype(device, expect_error):
    in0_t, in1_t, sparsity_t, active_ids, pc, _ = _contract_inputs(device)
    t = torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, 1, len(active_ids))
    indices_t = ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with expect_error(RuntimeError, "indices must be UINT16 dtype"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc)


def test_indices_rejects_more_than_num_groups(device, expect_error):
    """num_active must be <= the number of sparse groups in B, which is B's group axis (not the
    product of A's and B's batch lengths)."""
    in0_t, in1_t, sparsity_t, _, pc, dims = _contract_inputs(device)
    num_experts = dims[3]
    indices_t = _make_indices(list(range(num_experts)) + [0], device)

    with expect_error(RuntimeError, "must be <= the number of sparse groups"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc)


def test_indices_rejects_nnz(device, expect_error):
    """nnz would be silently ignored in indexed mode (the loop count is num_active)."""
    in0_t, in1_t, sparsity_t, active_ids, pc, _ = _contract_inputs(device)
    indices_t = _make_indices(active_ids, device)

    with expect_error(RuntimeError, "must not be supplied together with indices"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc, nnz=len(active_ids))


def test_indices_requires_input_b_sparse(device, expect_error):
    """The ids address B's sparse-group axis, so B must be the sparse operand."""
    torch.manual_seed(6)
    m, k, n = 32, 128, 256
    # A-sparse-only mode: the sparsity length is A's batch length, which is 1 for [1,1,M,K].
    in0_t = ttnn.from_torch(
        torch.randn((1, 1, m, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    in1_t = ttnn.from_torch(
        torch.randn((1, 1, k, n), dtype=torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    sparsity_t = ttnn.from_torch(
        torch.ones((1, 1, 1, 1), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    indices_t = _make_indices([0], device)

    with expect_error(RuntimeError, "requires is_input_b_sparse"):
        ttnn.sparse_matmul(
            in0_t,
            in1_t,
            sparsity=sparsity_t,
            indices=indices_t,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_sparse_pc(m, n, 32, 32),
        )


def test_indices_rejects_mismatched_optional_output(device, expect_error):
    """A preallocated output must have the indexed (compact) shape: a full-E tensor would be left with
    holes, and an undersized one would be written out of bounds."""
    in0_t, in1_t, sparsity_t, active_ids, pc, dims = _contract_inputs(device)
    m, _, n, num_experts = dims
    indices_t = _make_indices(active_ids, device)
    full_e_output = ttnn.from_torch(
        torch.zeros((1, 1, 1, num_experts, m, n), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    with expect_error(RuntimeError, "must match the indexed output shape"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc, optional_output_tensor=full_e_output)


def test_indexed_optional_output(device):
    """The indexed-shaped preallocated output is accepted and receives the gathered results."""
    torch.manual_seed(5)
    tile_h, tile_w = 32, 32
    m, k, n = 32, 128, 256
    num_experts = 16
    active_ids = [13, 1, 8, 4]
    num_active = len(active_ids)

    in0 = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)
    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    sparsity_t = _make_sparsity(active_ids, num_experts, device)
    indices_t = _make_indices(active_ids, device)
    preallocated = ttnn.from_torch(
        torch.full((1, 1, 1, num_active, m, n), 99.0, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    out_t = _run_indexed(
        in0_t, in1_t, sparsity_t, indices_t, _sparse_pc(m, n, tile_h, tile_w), optional_output_tensor=preallocated
    )
    out = ttnn.to_torch(out_t).reshape(num_active, m, n).float()

    in1_f = in1.float()
    for i, e in enumerate(active_ids):
        ref = torch.matmul(in0[0, 0].float(), in1_f[0, e])
        assert_numeric_metrics(
            ref, out[i], atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.01 * k, pcc_threshold=0.99
        )


# ----------------------------------------------------------------------------------------------------------
# 8. FILL skip on the op-allocated compact output of a legacy indexed call (phase 3d, lever A1)
# ----------------------------------------------------------------------------------------------------------
INDEXED_SKIP_FILL_ON = os.environ.get("TT_SPARSE_MATMUL_INDEXED_SKIP_FILL", "1") != "0"
POISON = 12345.0  # bfp8 poison (bfp8 has no NaN); bf16 outputs are poisoned with NaN
H, IP, E, N_GU = (
    4096,
    160,
    128,
    320,
)  # Solar-Open TP=8 expert shapes (hidden, intermediate per device, experts, gate|up N)
SOLAR_IDS = [127, 5, 126, 12, 1, 64, 7, 0]  # non-monotonic top-8
SOLAR_IDS_B = [3, 99, 0, 77, 127, 42, 8, 65]


def _pc_grid(cores, in0_block_w, per_core_N, out_subblock_w=1, per_core_M=1, out_block_h=None, out_subblock_h=1):
    """1D mcast_in0 config with an explicit grid (out_block_w == out_subblock_w, the shape the Solar builder emits)."""
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*cores),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_M if out_block_h is None else out_block_h,
        out_block_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _require_grid(device, cores):
    grid = device.compute_with_storage_grid_size()
    if cores[0] > grid.x or cores[1] > grid.y:
        pytest.skip(f"needs a {cores[0]}x{cores[1]} compute grid, device has {grid.x}x{grid.y}")


def _poison_host(shape, dtype):
    value = float("nan") if dtype == ttnn.bfloat16 else POISON
    return ttnn.from_torch(torch.full(tuple(shape), value, dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def _assert_poison_intact(t, what):
    """Sanity: the poison survives the dtype round trip (otherwise the poisoned tests would prove nothing)."""
    v = ttnn.to_torch(t).float()
    if t.dtype == ttnn.bfloat16:
        assert torch.isnan(v).all(), f"{what}: bf16 poison did not survive"
    else:
        assert (v != 0).all() and (v > 1000).all(), f"{what}: bfp8 poison did not survive (min {v.min()})"


def _assert_exact(got_t, ref_t, what):
    got, ref = ttnn.to_torch(got_t), ttnn.to_torch(ref_t)
    assert got.shape == ref.shape, f"{what}: shape {tuple(got.shape)} vs {tuple(ref.shape)}"
    if not torch.equal(got, ref):
        diff = got != ref
        bad = diff.reshape(-1, diff.shape[-2], diff.shape[-1]).flatten(1).any(1).nonzero().flatten().tolist()
        raise AssertionError(
            f"{what}: {int(diff.sum())} of {diff.numel()} elements differ from the FILLed optional-output result; "
            f"differing compact slots: {bad[:16]}{'...' if len(bad) > 16 else ''}; "
            f"max |diff| {float((got.float() - ref.float()).abs().max())} (poison {POISON} / NaN would show here)"
        )


def _run_on_poisoned_free_memory(device, run, what):
    """`run(**kw)` = an op-allocated legacy indexed sparse_matmul. Runs it once (program compile, output geometry),
    frees the output, poisons the freed hole with a same-spec tensor (bfp8 12345.0 / bf16 NaN), frees that, and runs
    again: the second output must land on the poisoned hole (asserted on the buffer address, otherwise the poison
    proves nothing) and is returned. Every tile the kernel does not write would keep its poison."""
    out0 = run()
    shape, dtype, mem, addr0 = tuple(out0.shape), out0.dtype, out0.memory_config(), out0.buffer_address()
    out0.deallocate(True)
    poison = ttnn.to_device(_poison_host(shape, dtype), device, memory_config=mem)
    addr_p = poison.buffer_address()
    assert addr_p == addr0, (
        f"{what}: the poison tensor landed at {addr_p:#x}, the op output was at {addr0:#x} -- the allocator did not "
        "reuse the hole, so the poisoned run would prove nothing"
    )
    _assert_poison_intact(poison, what)
    poison.deallocate(True)
    out = run()
    assert (
        out.buffer_address() == addr_p
    ), f"{what}: the second op output landed at {out.buffer_address():#x}, the poisoned hole was at {addr_p:#x}"
    return out


def _reference_on_poisoned_optional_output(device, run, shape, dtype, mem):
    """The unchanged path: a caller-supplied (pre-poisoned) output, FILLed by the op and then written by the kernel."""
    pre = ttnn.to_device(_poison_host(shape, dtype), device, memory_config=mem)
    return run(optional_output_tensor=pre)


def _skip_fill_protocol(device, run_a, run_b, what):
    """The two-index protocol on one configuration: (1) the op-owned output of ids A on poisoned freed memory equals
    the FILLed optional-output result of A; (2) after that, ids B on the same poisoned hole equal B's reference (a
    different gather into the same memory, program cache hit) and differ from A. Returns (out_a_torch, out_b_torch)."""
    out_a = _run_on_poisoned_free_memory(device, run_a, f"{what} ids A")
    shape, dtype, mem = tuple(out_a.shape), out_a.dtype, out_a.memory_config()
    ref_a = _reference_on_poisoned_optional_output(device, run_a, shape, dtype, mem)
    _assert_exact(out_a, ref_a, f"{what} ids A (op-owned output on poisoned freed memory)")
    ref_b = _reference_on_poisoned_optional_output(device, run_b, shape, dtype, mem)
    out_a_host = ttnn.to_torch(out_a)
    out_a.deallocate(True)
    out_b = _run_on_poisoned_free_memory(device, run_b, f"{what} ids B")
    _assert_exact(out_b, ref_b, f"{what} ids B (op-owned output on poisoned freed memory)")
    out_b_host = ttnn.to_torch(out_b)
    assert not torch.equal(out_a_host, out_b_host), f"{what}: different ids must give different outputs"
    for t in (out_b, ref_a, ref_b):
        t.deallocate(True)
    return out_a_host, out_b_host


def _check_fp32(out, in0, in1, ids, k, compact_a, what):
    """torch fp32 check of every compact slot (the exactness vs the FILLed path is the real assertion)."""
    out = out.float().reshape(-1, out.shape[-2], out.shape[-1])
    in0_f = in0.float().reshape(-1, in0.shape[-2], in0.shape[-1])
    in1_f = in1.float()
    m = in0.shape[-2]
    for i, e in enumerate(ids):
        a = in0_f[i] if compact_a else in0_f[0]
        ref = torch.matmul(a, in1_f[0, e])
        assert_numeric_metrics(
            ref, out[i][:m], atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.05 * k, pcc_threshold=0.99
        )


def _ids(device, ids):
    return _make_indices(ids, device)


def _dev(device, t, dtype, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=mem)


def _placeholder_mask(device, e):
    """The required (never read in indexed mode) sparsity operand: an all-ones [1,1,1,E] bf16 ROW_MAJOR mask."""
    return _dev(device, torch.ones(1, 1, 1, e, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize(
    "cores, per_core_N, osw",
    [((8, 8), 2, 2), ((8, 4), 4, 4)],
    ids=["8x8_pcn2_osw2_shipped", "8x4_pcn4_osw4_p3b"],
)
def test_skip_fill_solar_indexed_down_compact_a(device, cores, per_core_N, osw):
    """The Solar b1 indexed compact-A down: [1,8,1,160] bfp8 x [1,128,160,4096] bfp8 -> [1,8,1,4096] bfp8 in L1, both
    operands sparse, k = 8 non-monotonic ids, legacy 1D grid (the production config is 8x8 x 2 tiles). Also counts the
    program-cache entries of the first call: one (the sparse matmul only) with the FILL skipped, one or two with
    TT_SPARSE_MATMUL_INDEXED_SKIP_FILL=0 (the FILL program is shared across shapes, so an earlier test may have cached
    it already)."""
    _require_grid(device, cores)
    g = torch.Generator().manual_seed(2026)
    a_compact = torch.randn(1, len(SOLAR_IDS), 1, IP, generator=g)
    w_down = torch.randn(1, E, IP, H, generator=g) * 0.02
    a_t = _dev(device, a_compact, ttnn.bfloat8_b)
    w_t = _dev(device, w_down, ttnn.bfloat8_b)
    mask_t = _placeholder_mask(device, E)
    idx_a, idx_b = _ids(device, SOLAR_IDS), _ids(device, SOLAR_IDS_B)
    pc = _pc_grid(cores, 5, per_core_N, osw)

    def run(idx, **kw):
        return ttnn.sparse_matmul(
            a_t,
            w_t,
            sparsity=mask_t,
            indices=idx,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=pc,
            dtype=ttnn.bfloat8_b,
            **kw,
        )

    n0 = device.num_program_cache_entries()
    first = run(idx_a)
    n1 = device.num_program_cache_entries()
    first.deallocate(True)
    if INDEXED_SKIP_FILL_ON:
        assert n1 - n0 == 1, f"first legacy indexed call: expected 1 new program-cache entry (no FILL), got {n1 - n0}"
    else:
        assert 1 <= n1 - n0 <= 2, f"first legacy indexed call (FILL kept): expected 1-2 new entries, got {n1 - n0}"
    what = f"solar indexed down {cores[0]}x{cores[1]} pcn {per_core_N} osw {osw}"
    out_a, out_b = _skip_fill_protocol(device, lambda **kw: run(idx_a, **kw), lambda **kw: run(idx_b, **kw), what)
    assert tuple(out_a.shape) == (1, len(SOLAR_IDS), 1, H)
    _check_fp32(out_a, a_compact, w_down, SOLAR_IDS, IP, True, what)
    _check_fp32(out_b, a_compact, w_down, SOLAR_IDS_B, IP, True, what)
    logger.info(f"{what}: op-owned output on poisoned freed memory exact vs the FILLed optional output, both id lists")


@pytest.mark.parametrize("out_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16_gemma4_style"])
def test_skip_fill_solar_indexed_gate_up_legacy(device, out_dtype):
    """The Solar indexed gate|up on the legacy 5x2 grid (bcast A [1,1,1,4096] bf16 x [1,128,4096,320] bfp8 ->
    rank-6 [1,1,1,8,1,320]; the production call is EGP, this is the legacy-kernel arm) in bfp8 and, gemma4-style
    (M = 1, bf16 output), with a NaN poison."""
    _require_grid(device, (5, 2))
    g = torch.Generator().manual_seed(2027)
    x = torch.randn(1, 1, 1, H, generator=g).to(torch.bfloat16)
    w_gu = torch.randn(1, E, H, N_GU, generator=g) * 0.02
    x_t = _dev(device, x, ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
    w_t = _dev(device, w_gu, ttnn.bfloat8_b)
    mask_t = _placeholder_mask(device, E)
    idx_a, idx_b = _ids(device, SOLAR_IDS), _ids(device, SOLAR_IDS_B)
    pc = _pc_grid((5, 2), 128, 1)

    def run(idx, **kw):
        return ttnn.sparse_matmul(
            x_t,
            w_t,
            sparsity=mask_t,
            indices=idx,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=pc,
            dtype=out_dtype,
            **kw,
        )

    what = f"solar indexed gate|up legacy 5x2 out {out_dtype}"
    out_a, out_b = _skip_fill_protocol(device, lambda **kw: run(idx_a, **kw), lambda **kw: run(idx_b, **kw), what)
    assert tuple(out_a.shape) == (1, 1, 1, len(SOLAR_IDS), 1, N_GU)
    _check_fp32(out_a, x, w_gu, SOLAR_IDS, H, False, what)
    _check_fp32(out_b, x, w_gu, SOLAR_IDS_B, H, False, what)


@pytest.mark.parametrize("cores, per_core_N, osw", [((8, 3), 1, 1), ((6, 2), 2, 2)], ids=["8x3_pcn1", "6x2_pcn2_osw2"])
def test_skip_fill_gpt_oss_shape(device, cores, per_core_N, osw):
    """gpt-oss expert shape [1,1,32,2880] bf16 x [1,128,2880,768] bfp8 (Kt 90, Nt 24), bf16 output in DRAM (NaN
    poison), k = 4 ids, M = 32 rows per slot."""
    _require_grid(device, cores)
    k, n = 2880, 768
    ids_a, ids_b = [127, 3, 64, 0], [5, 126, 1, 77]
    g = torch.Generator().manual_seed(31)
    x = torch.randn(1, 1, 32, k, generator=g).to(torch.bfloat16)
    w = torch.randn(1, E, k, n, generator=g) * 0.02
    x_t = _dev(device, x, ttnn.bfloat16)
    w_t = _dev(device, w, ttnn.bfloat8_b)
    mask_t = _placeholder_mask(device, E)
    idx_a, idx_b = _ids(device, ids_a), _ids(device, ids_b)
    pc = _pc_grid(cores, 90, per_core_N, osw)

    def run(idx, **kw):
        return ttnn.sparse_matmul(
            x_t,
            w_t,
            sparsity=mask_t,
            indices=idx,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
            dtype=ttnn.bfloat16,
            **kw,
        )

    what = f"gpt-oss indexed {cores[0]}x{cores[1]} pcn {per_core_N} osw {osw} bf16"
    out_a, out_b = _skip_fill_protocol(device, lambda **kw: run(idx_a, **kw), lambda **kw: run(idx_b, **kw), what)
    assert tuple(out_a.shape) == (1, 1, 1, len(ids_a), 32, n)
    _check_fp32(out_a, x, w, ids_a, k, False, what)


@pytest.mark.parametrize("mem", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG], ids=["L1", "DRAM"])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize(
    "cores, per_core_N, osw, per_core_M, out_block_h",
    [((4, 1), 3, 1, 2, 2), ((2, 2), 3, 3, 2, 1), ((5, 1), 2, 2, 1, 1)],
    ids=["4x1_pcn3_osw1_M2", "2x2_pcn3_osw3_M2_obh1", "5x1_pcn2_osw2_M1_exact"],
)
def test_skip_fill_ragged_last_column(device, cores, per_core_N, osw, per_core_M, out_block_h, out_dtype, mem):
    """Ragged N: Nt = 10 tiles on per_core_N 3 -> 4 column blocks, the last one 1 tile wide (the writer's
    last_per_core_N / out_last_subblock_w / padded-skip path), with out_subblock_w 1 (three 1-tile blocks per core)
    and 3 (one 3-tile block whose last subblock is cut to 1 tile); Mt = 2 rows per slot as one 2-row block and as two
    1-row blocks (the bh loop). Control: Nt 10 = 5 exact blocks of 2 tiles on 5x1 with M = 1. Every grid is an exact
    fill (a partially used mcast bounding box would hang the factory). Small operands: E = 16, K = 128 (Kt 4)."""
    _require_grid(device, cores)
    e, k, n = 16, 128, N_GU
    m = 32 * per_core_M
    ids_a, ids_b = [15, 2, 9, 0, 13], [1, 14, 6, 12, 3]
    g = torch.Generator().manual_seed(53)
    x = torch.randn(1, 1, m, k, generator=g).to(torch.bfloat16)
    w = torch.randn(1, e, k, n, generator=g) * 0.05
    x_t = _dev(device, x, ttnn.bfloat16)
    w_t = _dev(device, w, ttnn.bfloat8_b)
    mask_t = _placeholder_mask(device, e)
    idx_a, idx_b = _ids(device, ids_a), _ids(device, ids_b)
    pc = _pc_grid(cores, 4, per_core_N, osw, per_core_M=per_core_M, out_block_h=out_block_h)

    def run(idx, **kw):
        return ttnn.sparse_matmul(
            x_t,
            w_t,
            sparsity=mask_t,
            indices=idx,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=mem,
            program_config=pc,
            dtype=out_dtype,
            **kw,
        )

    what = (
        f"ragged indexed {cores[0]}x{cores[1]} pcn {per_core_N} osw {osw} M {per_core_M} obh {out_block_h} {out_dtype}"
    )
    out_a, out_b = _skip_fill_protocol(device, lambda **kw: run(idx_a, **kw), lambda **kw: run(idx_b, **kw), what)
    assert tuple(out_a.shape) == (1, 1, 1, len(ids_a), m, n)
    _check_fp32(out_a, x, w, ids_a, k, False, what)
    _check_fp32(out_b, x, w, ids_b, k, False, what)
