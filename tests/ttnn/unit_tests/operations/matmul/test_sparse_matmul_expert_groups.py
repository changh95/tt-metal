# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Op-level tests for the expert-group-parallel (EGP) mode of ttnn.sparse_matmul (`expert_groups=G`).

Every case runs the legacy path (`expert_groups=None`) and the EGP path on the SAME device tensors and asserts
EXACT equality of the outputs: EGP only changes the (expert, output block) -> core assignment, keeps the in0 tile
resident in L1 instead of re-multicasting it per expert and decides the validity of every sparsity slot locally,
so the per-output-tile arithmetic is unchanged. Shapes are the Solar-Open TP=8 expert shapes (hidden 4096,
intermediate 160 per device, 128 experts, fused gate|up N = 320) plus the gpt-oss and gemma4-style callers.
"""

import math

from loguru import logger
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics

H, IP, E, N_GU = 4096, 160, 128, 320
TILE = ttnn.Tile([32, 32])


def _pc(cores, in0_block_w, per_core_N, out_subblock_w=1, per_core_M=1, out_block_w=None):
    """1D mcast_in0 config in the shape the Solar expert builder emits (out_block == subblock unless given)."""
    out_block_w = out_subblock_w if out_block_w is None else out_block_w
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*cores),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_M,
        out_block_w=out_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _egp_grid(num_blocks, groups, width=None):
    """Exact-rectangle grid for G x blocks cores: `width` columns (default: the block count) x G rows, or the
    full 11x10 chip when that is the requested size."""
    cores = num_blocks * groups
    if cores == 110:
        return (11, 10)
    width = num_blocks if width is None else width
    assert cores % width == 0, (cores, width)
    return (width, cores // width)


def _mask_torch(nnz, pattern, seed, e=E):
    """[1,1,1,e] bf16 mask with exactly nnz non-zeros: random ids, the first / last nnz slots, or single@k."""
    mask = torch.zeros(1, 1, 1, e)
    if pattern == "random":
        idx = torch.randperm(e, generator=torch.Generator().manual_seed(seed))[:nnz]
    elif pattern == "first":
        idx = torch.arange(nnz)
    elif pattern == "last":
        idx = torch.arange(e - nnz, e)
    elif pattern.startswith("single@"):
        assert nnz == 1
        idx = torch.tensor([int(pattern.split("@")[1])])
    else:
        raise ValueError(pattern)
    mask[..., idx] = 0.125 + 0.5 * torch.rand(nnz, generator=torch.Generator().manual_seed(seed + 1))
    return mask.to(torch.bfloat16), idx.sort().values.tolist()


def _to_dev(device, t, dtype, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=mem)


def _mask_dev(device, nnz, pattern, seed=0, e=E):
    mask, idx = _mask_torch(nnz, pattern, seed, e)
    return _to_dev(device, mask, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), idx


def _assert_exact(egp_t, ref_t, what):
    egp = ttnn.to_torch(egp_t)
    ref = ttnn.to_torch(ref_t)
    assert egp.shape == ref.shape, f"{what}: shape {tuple(egp.shape)} vs {tuple(ref.shape)}"
    if not torch.equal(egp, ref):
        diff = egp != ref
        # Report which expert slots differ (the expert axis is dim -3 in every mode).
        bad = diff.reshape(-1, diff.shape[-2], diff.shape[-1]).flatten(1).any(1).nonzero().flatten().tolist()
        raise AssertionError(
            f"{what}: {int(diff.sum())} of {diff.numel()} elements differ from the legacy result; "
            f"differing expert slots (flattened batch index): {bad[:16]}{'...' if len(bad) > 16 else ''}; "
            f"max |diff| {float((egp.float() - ref.float()).abs().max())}"
        )


def _check_pcc(out_t, in0, in1, idx, k, slots=None):
    """Sanity PCC vs torch fp32 for a few active experts (exactness vs legacy is the real assertion)."""
    out = ttnn.to_torch(out_t).float()
    out = out.reshape(-1, out.shape[-2], out.shape[-1])
    in0_f = in0.float().reshape(-1, in0.shape[-2], in0.shape[-1])
    for j, e in enumerate(idx[:3] if slots is None else slots):
        a = in0_f[0] if in0_f.shape[0] == 1 else in0_f[j]
        ref = torch.matmul(a, in1[0, e].float())
        got = out[e] if slots is None else out[j]
        assert_numeric_metrics(
            ref, got[: ref.shape[0]], atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.05 * k, pcc_threshold=0.99
        )


def _sparse(in0_t, in1_t, mask_t, pc, **kw):
    kw.setdefault("memory_config", ttnn.L1_MEMORY_CONFIG)
    if "optional_output_tensor" not in kw:  # the op rejects output_tile together with an optional output tensor
        kw.setdefault("output_tile", TILE)
    kw.setdefault("dtype", ttnn.bfloat8_b)
    return ttnn.sparse_matmul(in0_t, in1_t, sparsity=mask_t, program_config=pc, **kw)


@pytest.fixture(scope="module")
def solar_torch():
    g = torch.Generator().manual_seed(2026)
    return {
        "x32": torch.randn(1, 1, 32, H, generator=g).to(torch.bfloat16),
        "x1": torch.randn(1, 1, 1, H, generator=g).to(torch.bfloat16),
        "w_gu": torch.randn(1, E, H, N_GU, generator=g) * 0.02,
        "w_down": torch.randn(1, E, IP, H, generator=g) * 0.02,
        "act_down": torch.randn(1, E, 32, IP, generator=g),
    }


def _gate_up_tensors(device, solar_torch, m=32):
    x = solar_torch["x32"] if m == 32 else solar_torch["x1"]
    x_t = _to_dev(device, x, ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
    w_t = _to_dev(device, solar_torch["w_gu"], ttnn.bfloat8_b)
    return x, x_t, w_t


# ----------------------------------------------------------------------------------------------------------
# 1. Solar gate|up, nnz inferred (the b32 union path): [1,1,32,4096] bf16 x [1,128,4096,320] bfp8, out bfp8
# ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nnz", [1, 8, 32, 72, 112, 128])
def test_gate_up_scan_exact(device, solar_torch, nnz):
    x, x_t, w_t = _gate_up_tensors(device, solar_torch)
    patterns = ["random", "first", "last"] + (["single@0", "single@127"] if nnz == 1 else [])
    groups = [1, 2, 5, 11] + ([3, 10] if nnz == 72 else [])
    for pattern in patterns:
        mask_t, idx = _mask_dev(device, nnz, pattern, seed=nnz)
        for bw in (128, 32):
            ref_t = _sparse(x_t, w_t, mask_t, _pc((5, 2), bw, 1))
            for groups_ in groups if bw == 128 else [1, 11]:
                egp_t = _sparse(x_t, w_t, mask_t, _pc(_egp_grid(10, groups_), bw, 1), expert_groups=groups_)
                _assert_exact(egp_t, ref_t, f"gate|up nnz {nnz} {pattern} bw {bw} G {groups_}")
                egp_t.deallocate(True)
            if bw == 128 and pattern == "random":
                _check_pcc(ref_t, x, solar_torch["w_gu"], idx, H)
            ref_t.deallocate(True)
        mask_t.deallocate(True)
    logger.info(f"gate|up scan nnz {nnz}: exact for patterns {patterns}, G {groups}, bw 128/32")


def test_gate_up_m1_exact(device, solar_torch):
    """M = 1 (b1 decode without indices): in0 is padded to one tile, same resident block as M = 32."""
    x, x_t, w_t = _gate_up_tensors(device, solar_torch, m=1)
    mask_t, idx = _mask_dev(device, 8, "random", seed=3)
    ref_t = _sparse(x_t, w_t, mask_t, _pc((5, 2), 128, 1))
    for groups in (1, 11):
        egp_t = _sparse(x_t, w_t, mask_t, _pc(_egp_grid(10, groups), 128, 1), expert_groups=groups)
        _assert_exact(egp_t, ref_t, f"gate|up M=1 nnz 8 G {groups}")
    _check_pcc(ref_t, x, solar_torch["w_gu"], idx, H)


def test_gate_up_out_bf16_exact(device, solar_torch):
    x, x_t, w_t = _gate_up_tensors(device, solar_torch)
    mask_t, _ = _mask_dev(device, 72, "random", seed=5)
    ref_t = _sparse(x_t, w_t, mask_t, _pc((5, 2), 128, 1), dtype=ttnn.bfloat16)
    egp_t = _sparse(x_t, w_t, mask_t, _pc((11, 10), 128, 1), expert_groups=11, dtype=ttnn.bfloat16)
    _assert_exact(egp_t, ref_t, "gate|up nnz 72 out bf16 G 11")


# ----------------------------------------------------------------------------------------------------------
# 4/5. static nnz: expanded output and compact [1, nnz, M, N] output
# ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nnz", [8, 72, 128])
def test_static_nnz_expanded_exact(device, solar_torch, nnz):
    x, x_t, w_t = _gate_up_tensors(device, solar_torch)
    mask_t, _ = _mask_dev(device, nnz, "random", seed=11 + nnz)
    ref_t = _sparse(x_t, w_t, mask_t, _pc((5, 2), 128, 1), nnz=nnz)
    for groups in (1, 5, 11):
        egp_t = _sparse(x_t, w_t, mask_t, _pc(_egp_grid(10, groups), 128, 1), nnz=nnz, expert_groups=groups)
        _assert_exact(egp_t, ref_t, f"static nnz {nnz} expanded G {groups}")
        egp_t.deallocate(True)


@pytest.mark.parametrize("nnz", [8, 72])
def test_static_nnz_compact_exact(device, solar_torch, nnz):
    x, x_t, w_t = _gate_up_tensors(device, solar_torch)

    def compact_out():
        return ttnn.from_torch(
            torch.full((1, nnz, 32, N_GU), 7.0, dtype=torch.bfloat16),  # poison: every slot must be overwritten
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    for pattern in ("first", "last", "random"):
        mask_t, idx = _mask_dev(device, nnz, pattern, seed=17 + nnz)
        ref_t = _sparse(x_t, w_t, mask_t, _pc((5, 2), 128, 1), nnz=nnz, optional_output_tensor=compact_out())
        assert tuple(ref_t.shape) == (1, nnz, 32, N_GU)
        for groups in (2, 5, 11):
            egp_t = _sparse(
                x_t,
                w_t,
                mask_t,
                _pc(_egp_grid(10, groups), 128, 1),
                nnz=nnz,
                expert_groups=groups,
                optional_output_tensor=compact_out(),
            )
            _assert_exact(egp_t, ref_t, f"static nnz {nnz} compact {pattern} G {groups}")
            egp_t.deallocate(True)
        # compact slot j holds expert idx[j] (scan order)
        _check_pcc(ref_t, x, solar_torch["w_gu"], idx, H, slots=idx[:3])
        ref_t.deallocate(True)


# ----------------------------------------------------------------------------------------------------------
# 6. Solar down: A and B sparse, [1,128,32,160] bfp8 x [1,128,160,4096] bfp8, bw 5 (Kt = 5), Nt = 128
# ----------------------------------------------------------------------------------------------------------
DOWN_EGP_CONFIGS = [
    # (per_core_N, out_subblock_w, groups, grid)
    (2, 2, 1, (8, 8)),
    (4, 4, 2, (8, 8)),
    (8, 8, 1, (8, 2)),
    (8, 8, 5, (8, 10)),
    (16, 8, 10, (8, 10)),
]


@pytest.mark.parametrize("nnz", [1, 8, 72, 128])
def test_down_exact(device, solar_torch, nnz):
    act_t = _to_dev(device, solar_torch["act_down"], ttnn.bfloat8_b)
    w_t = _to_dev(device, solar_torch["w_down"], ttnn.bfloat8_b)
    mask_t, idx = _mask_dev(device, nnz, "random", seed=23 + nnz)
    ref_t = _sparse(act_t, w_t, mask_t, _pc((8, 8), 5, 2, 2), is_input_a_sparse=True)
    for pcn, osw, groups, grid in DOWN_EGP_CONFIGS:
        egp_t = _sparse(act_t, w_t, mask_t, _pc(grid, 5, pcn, osw), is_input_a_sparse=True, expert_groups=groups)
        _assert_exact(egp_t, ref_t, f"down nnz {nnz} pcn {pcn} osw {osw} G {groups} grid {grid}")
        egp_t.deallocate(True)
    out = ttnn.to_torch(ref_t).float()
    for e in idx[:3]:
        ref = torch.matmul(solar_torch["act_down"][0, e].float(), solar_torch["w_down"][0, e].float())
        assert_numeric_metrics(
            ref, out[0, e], atol=0.05 * IP, rtol=10.0 * IP, frobenius_threshold=0.05 * IP, pcc_threshold=0.99
        )


# ----------------------------------------------------------------------------------------------------------
# 7/8. indexed / gather mode (the b1 path): compact output addressed by the entry index
# ----------------------------------------------------------------------------------------------------------
def _indices_dev(device, ids):
    t = torch.tensor(ids, dtype=torch.int32).reshape(1, 1, 1, len(ids))
    return ttnn.from_torch(t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


ACTIVE_IDS = [127, 5, 126, 12, 1, 64, 7, 0]  # non-monotonic top-8


def test_indexed_gate_up_exact(device, solar_torch):
    x, x_t, w_t = _gate_up_tensors(device, solar_torch, m=1)
    mask_t = _to_dev(
        device,
        torch.zeros(1, 1, 1, E, dtype=torch.bfloat16).index_fill_(3, torch.tensor(ACTIVE_IDS), 1.0),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    idx_t = _indices_dev(device, ACTIVE_IDS)
    ref_t = _sparse(x_t, w_t, mask_t, _pc((5, 2), 128, 1), indices=idx_t)
    assert tuple(ref_t.shape)[-3:] == (len(ACTIVE_IDS), 1, N_GU)  # [1,1,1,k,1,N]: compact expert axis
    for groups in (1, 2, 5, 11):
        egp_t = _sparse(x_t, w_t, mask_t, _pc(_egp_grid(10, groups), 128, 1), indices=idx_t, expert_groups=groups)
        _assert_exact(egp_t, ref_t, f"indexed gate|up k 8 G {groups}")
    _check_pcc(ref_t, x, solar_torch["w_gu"], ACTIVE_IDS, H, slots=ACTIVE_IDS)


def test_indexed_down_compact_a_exact(device, solar_torch):
    k = len(ACTIVE_IDS)
    a_compact = torch.randn(1, k, 1, IP, generator=torch.Generator().manual_seed(29))
    a_t = _to_dev(device, a_compact, ttnn.bfloat8_b)
    w_t = _to_dev(device, solar_torch["w_down"], ttnn.bfloat8_b)
    mask_t = _to_dev(
        device,
        torch.zeros(1, 1, 1, E, dtype=torch.bfloat16).index_fill_(3, torch.tensor(ACTIVE_IDS), 1.0),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    idx_t = _indices_dev(device, ACTIVE_IDS)
    ref_t = _sparse(a_t, w_t, mask_t, _pc((8, 4), 5, 4, 4), indices=idx_t, is_input_a_sparse=True)
    assert tuple(ref_t.shape) == (1, k, 1, H)
    for groups in (1, 2, 5):
        egp_t = _sparse(
            a_t,
            w_t,
            mask_t,
            _pc((8, 2 * groups), 5, 8, 8),
            indices=idx_t,
            is_input_a_sparse=True,
            expert_groups=groups,
        )
        _assert_exact(egp_t, ref_t, f"indexed down compact-A k 8 pcn 8 G {groups}")
    out = ttnn.to_torch(ref_t).float()
    for j, e in enumerate(ACTIVE_IDS[:3]):
        ref = torch.matmul(a_compact[0, j].float(), solar_torch["w_down"][0, e].float())
        assert_numeric_metrics(
            ref, out[0, j, :1], atol=0.05 * IP, rtol=10.0 * IP, frobenius_threshold=0.05 * IP, pcc_threshold=0.99
        )


# ----------------------------------------------------------------------------------------------------------
# 9. gpt-oss regression shape: [1,1,32,2880] bf16 x [1,128,2880,768] bfp8 (Kt 90, Nt 24), out bf16
# ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nnz", [4, 64])
def test_gpt_oss_shape_exact(device, nnz):
    k, n = 2880, 768
    g = torch.Generator().manual_seed(31)
    x = torch.randn(1, 1, 32, k, generator=g).to(torch.bfloat16)
    w = torch.randn(1, E, k, n, generator=g) * 0.02
    x_t = _to_dev(device, x, ttnn.bfloat16)
    w_t = _to_dev(device, w, ttnn.bfloat8_b)
    mask_t, idx = _mask_dev(device, nnz, "random", seed=37 + nnz)
    # per_core_N 1: 24 blocks -> legacy (8,3); EGP G 2 -> 48 cores (8,6), G 3 -> 72 cores (8,9)
    # per_core_N 2: 12 blocks -> legacy (6,2); EGP G 4 -> 48 cores (8,6), G 6 -> 72 cores (8,9)
    for bw in (90, 1) if nnz == 4 else (90,):
        ref1_t = _sparse(x_t, w_t, mask_t, _pc((8, 3), bw, 1), dtype=ttnn.bfloat16)
        for groups, grid in ((2, (8, 6)), (3, (8, 9))):
            egp_t = _sparse(x_t, w_t, mask_t, _pc(grid, bw, 1), dtype=ttnn.bfloat16, expert_groups=groups)
            _assert_exact(egp_t, ref1_t, f"gpt-oss nnz {nnz} bw {bw} pcn 1 G {groups}")
            egp_t.deallocate(True)
        ref2_t = _sparse(x_t, w_t, mask_t, _pc((6, 2), bw, 2, 2), dtype=ttnn.bfloat16)
        _assert_exact(ref2_t, ref1_t, f"gpt-oss nnz {nnz} bw {bw}: legacy pcn 2 vs legacy pcn 1")
        for groups, grid in ((4, (8, 6)), (6, (8, 9))):
            egp_t = _sparse(x_t, w_t, mask_t, _pc(grid, bw, 2, 2), dtype=ttnn.bfloat16, expert_groups=groups)
            _assert_exact(egp_t, ref2_t, f"gpt-oss nnz {nnz} bw {bw} pcn 2 G {groups}")
            egp_t.deallocate(True)
        if bw == 90:
            _check_pcc(ref1_t, x, w, idx, k)
        ref1_t.deallocate(True)
        ref2_t.deallocate(True)


# ----------------------------------------------------------------------------------------------------------
# 10. gemma4-style: M = 1, static nnz = top_k, expanded [1,1,1,E,32,N] output, out bf16
# ----------------------------------------------------------------------------------------------------------
def test_gemma4_style_exact(device, solar_torch):
    x, x_t, w_t = _gate_up_tensors(device, solar_torch, m=1)
    top_k = 4
    mask_t, _ = _mask_dev(device, top_k, "random", seed=41)
    ref_t = _sparse(x_t, w_t, mask_t, _pc((5, 2), 1, 1), nnz=top_k, dtype=ttnn.bfloat16)
    egp_t = _sparse(x_t, w_t, mask_t, _pc((10, 4), 1, 1), nnz=top_k, dtype=ttnn.bfloat16, expert_groups=4)
    _assert_exact(egp_t, ref_t, "gemma4-style nnz 4 bw 1 G 4")


# ----------------------------------------------------------------------------------------------------------
# 11. program cache: distinct entries per factory, hit on a new mask
# ----------------------------------------------------------------------------------------------------------
def test_program_cache_exact(device, solar_torch):
    x, x_t, w_t = _gate_up_tensors(device, solar_torch)
    mask_a, _ = _mask_dev(device, 72, "random", seed=43)
    mask_b, _ = _mask_dev(device, 34, "random", seed=47)
    pc_legacy, pc_egp = _pc((5, 2), 128, 1), _pc((11, 10), 128, 1)
    n0 = device.num_program_cache_entries()
    ref_a = _sparse(x_t, w_t, mask_a, pc_legacy)
    egp_a = _sparse(x_t, w_t, mask_a, pc_egp, expert_groups=11)
    n1 = device.num_program_cache_entries()
    assert n1 - n0 >= 2, f"expected a legacy and an EGP program cache entry, got {n1 - n0} new entries"
    _assert_exact(egp_a, ref_a, "program cache: first EGP run")
    ref_b = _sparse(x_t, w_t, mask_b, pc_legacy)
    egp_b = _sparse(x_t, w_t, mask_b, pc_egp, expert_groups=11)
    n2 = device.num_program_cache_entries()
    assert n2 == n1, f"a new mask must hit the cached programs, got {n2 - n1} new entries"
    _assert_exact(egp_b, ref_b, "program cache: EGP run on a different mask (cache hit)")
    assert not torch.equal(ttnn.to_torch(egp_a), ttnn.to_torch(egp_b)), "different masks must give different outputs"


# ----------------------------------------------------------------------------------------------------------
# 12. host-side contract
# ----------------------------------------------------------------------------------------------------------
def test_expert_groups_errors(device, solar_torch, expect_error):
    x, x_t, w_t = _gate_up_tensors(device, solar_torch)
    mask_t, _ = _mask_dev(device, 8, "random", seed=53)
    act_t = _to_dev(device, solar_torch["act_down"], ttnn.bfloat8_b)
    w_down_t = _to_dev(device, solar_torch["w_down"], ttnn.bfloat8_b)

    with expect_error(RuntimeError, "expert_groups must be >= 1"):
        _sparse(x_t, w_t, mask_t, _pc((5, 2), 128, 1), expert_groups=0)
    with expect_error(RuntimeError, "exceeds the 11x10 grid"):
        _sparse(x_t, w_t, mask_t, _pc((11, 10), 128, 1), expert_groups=12)
    with expect_error(RuntimeError, "does not fill an exact rectangle"):
        _sparse(x_t, w_t, mask_t, _pc((11, 10), 128, 1), expert_groups=5)
    with expect_error(RuntimeError, "must not exceed the number of sparse slots"):
        _sparse(x_t, w_t, mask_t, _pc((11, 10), 128, 1), expert_groups=129)
    with expect_error(RuntimeError, "requires is_input_b_sparse=true"):
        # A sparse, B dense: A [1,E,32,K] against B [1,E,K,N] with a [1,1,1,E] mask
        _sparse(
            act_t,
            w_down_t,
            mask_t,
            _pc((8, 8), 5, 2, 2),
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            expert_groups=2,
        )
    x4_t = _to_dev(device, torch.randn(1, 4, 32, H).to(torch.bfloat16), ttnn.bfloat16)
    mask4_t = _to_dev(device, torch.ones(1, 4, 1, E, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(RuntimeError, "single outer batch"):
        _sparse(x4_t, w_t, mask4_t, _pc((10, 2), 128, 1), expert_groups=2)
    with expect_error(RuntimeError, "at least 2 cores"):
        # 1 block x 1 group: [1,1,32,4096] x [1,128,4096,32] has a single output block
        w1_t = _to_dev(device, torch.randn(1, E, H, 32) * 0.02, ttnn.bfloat8_b)
        _sparse(x_t, w1_t, mask_t, _pc((1, 1), 128, 1), expert_groups=1)
    # oversize resident in0: per_core_M 8 (M = 256) at bw 128 -> 8 x 128 bf16 tiles = 2 MB > L1
    x256_t = _to_dev(device, torch.randn(1, 1, 256, H).to(torch.bfloat16), ttnn.bfloat16)
    with expect_error(RuntimeError, "circular buffers need"):
        _sparse(x256_t, w_t, mask_t, _pc((10, 2), 128, 1, per_core_M=8), expert_groups=2)
    # broadcast A with both block loops non-trivial (per_core_M 2 blocks x per_core_N 2 blocks)
    x64_t = _to_dev(device, torch.randn(1, 1, 64, H).to(torch.bfloat16), ttnn.bfloat16)
    with expect_error(RuntimeError, "per_core_M == out_block_h"):
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 2),
            in0_block_w=32,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=2,
            per_core_N=2,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        _sparse(x64_t, w_t, mask_t, pc, expert_groups=2)
