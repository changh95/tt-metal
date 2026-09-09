# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Op-level tests for the expert-group-parallel (EGP) mode of ttnn.sparse_matmul (`expert_groups=G`).

Every case runs the legacy path (`expert_groups=None`) and the EGP path on the SAME device tensors and asserts
EXACT equality of the outputs: EGP only changes the (expert, output block) -> core assignment, keeps the in0 tile
resident in L1 instead of re-multicasting it per expert and decides the validity of every sparsity slot locally,
so the per-output-tile arithmetic is unchanged. Shapes are the Solar-Open TP=8 expert shapes (hidden 4096,
intermediate 160 per device, 128 experts, fused gate|up N = 320) plus the gpt-oss and gemma4-style callers.

Kernel zero-fill (sections 13-19): with `expert_groups` set the op no longer runs its host zero-fill (FILL) pass over
the output; the EGP kernels zero-fill every slot nobody computes (group `slot % G`, same tile walk as a computed slot;
issued by the in0 reader by default, or by the in1 writer with `TT_SPARSE_MATMUL_EGP_ZERO_FILL=in1`) and indexed /
compact outputs are fully written by construction. Those tests hand the EGP op a PRE-POISONED `optional_output_tensor`
(bfp8: 12345.0, bf16: NaN -- values no valid product yields) and require torch.equal with the legacy result on a fresh
output, so every stale tile shows up. `TT_SPARSE_MATMUL_EGP_ZERO_FILL=0` (fresh process) restores the phase-3b
behaviour (FILL kept); the whole file must pass in all three arms.
"""

import math
import os

from loguru import logger
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics

H, IP, E, N_GU = 4096, 160, 128, 320
TILE = ttnn.Tile([32, 32])
POISON = 12345.0  # bfp8 poison (bfp8 has no NaN); bf16 outputs are poisoned with NaN
ZERO_FILL_ON = os.environ.get("TT_SPARSE_MATMUL_EGP_ZERO_FILL", "1") != "0"
# Distinct (nnz, pattern) masks of the zero-fill plan: nnz 0 / 1 / 8 / 72 / 128 x random / first / last (+ the two
# single-slot masks at nnz 1; nnz 0 and 128 have one mask each).
ZF_MASKS = [
    (0, "random"),
    (1, "random"),
    (1, "single@0"),
    (1, "single@127"),
    (8, "random"),
    (8, "first"),
    (8, "last"),
    (72, "random"),
    (72, "first"),
    (72, "last"),
    (128, "first"),
]


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


_POISON_HOST = {}


def _poisoned_out(device, shape, dtype=ttnn.bfloat8_b, mem=ttnn.L1_MEMORY_CONFIG):
    """A device output tensor pre-filled with a value no valid slot produces (bfp8: 12345.0, bf16: NaN). Every tile
    must be overwritten by compute or by the writer zero-fill for torch.equal against the legacy result to hold.
    The host copy is converted once per (shape, dtype) and re-uploaded per call (17.8 MB for the down output)."""
    key = (tuple(shape), dtype)
    if key not in _POISON_HOST:
        value = float("nan") if dtype == ttnn.bfloat16 else POISON
        _POISON_HOST[key] = ttnn.from_torch(
            torch.full(tuple(shape), value, dtype=torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT
        )
    return ttnn.to_device(_POISON_HOST[key], device, memory_config=mem)


def _assert_poison_intact(t, what):
    """Sanity: the poison survives the dtype round trip (otherwise the poisoned tests would prove nothing)."""
    v = ttnn.to_torch(t).float()
    if t.dtype == ttnn.bfloat16:
        assert torch.isnan(v).all(), f"{what}: bf16 poison did not survive"
    else:
        assert (v != 0).all() and (v > 1000).all(), f"{what}: bfp8 poison did not survive (min {v.min()})"


def _expected_new_cache_entries(egp_first):
    """Program-cache entries a first EGP call / a first legacy call adds for a new shape. With the writer zero-fill
    the EGP call launches ONE program (no FILL pass); the legacy call launches its FILL (UnaryDeviceOperation) and
    the sparse matmul. With TT_SPARSE_MATMUL_EGP_ZERO_FILL=0 both launch the FILL and share its cache entry."""
    if ZERO_FILL_ON:
        return (1, 2) if egp_first else (2, 1)
    return (2, 1) if egp_first else (2, 1)


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
    """Static nnz with an EXPANDED output: ranks >= nnz and invalid slots must be zero (poisoned EGP output)."""
    x, x_t, w_t = _gate_up_tensors(device, solar_torch)
    mask_t, _ = _mask_dev(device, nnz, "random", seed=11 + nnz)
    ref_t = _sparse(x_t, w_t, mask_t, _pc((5, 2), 128, 1), nnz=nnz)
    for groups in (1, 5, 11):
        egp_t = _sparse(
            x_t,
            w_t,
            mask_t,
            _pc(_egp_grid(10, groups), 128, 1),
            nnz=nnz,
            expert_groups=groups,
            optional_output_tensor=_poisoned_out(device, ref_t.shape),
        )
        _assert_exact(egp_t, ref_t, f"static nnz {nnz} expanded G {groups} (poisoned output)")
        egp_t.deallocate(True)
    # the same with the (A+B sparse) down at the production grid
    act_t = _to_dev(device, solar_torch["act_down"], ttnn.bfloat8_b)
    w_down_t = _to_dev(device, solar_torch["w_down"], ttnn.bfloat8_b)
    ref_d = _sparse(act_t, w_down_t, mask_t, _pc((8, 8), 5, 2, 2), nnz=nnz, is_input_a_sparse=True)
    egp_d = _sparse(
        act_t,
        w_down_t,
        mask_t,
        _pc((11, 8), 5, 16, 8),
        nnz=nnz,
        is_input_a_sparse=True,
        expert_groups=11,
        optional_output_tensor=_poisoned_out(device, ref_d.shape),
    )
    _assert_exact(egp_d, ref_d, f"static nnz {nnz} expanded down pcn16 osw8 G 11 (poisoned output)")


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
        # poisoned indexed-shape output: indexed mode never had a FILL to rely on (every entry is written) and
        # the EGP op now skips the FILL pass outright -- every entry must still be overwritten.
        egp_t = _sparse(
            x_t,
            w_t,
            mask_t,
            _pc(_egp_grid(10, groups), 128, 1),
            indices=idx_t,
            expert_groups=groups,
            optional_output_tensor=_poisoned_out(device, ref_t.shape),
        )
        _assert_exact(egp_t, ref_t, f"indexed gate|up k 8 G {groups} (poisoned output)")
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
            optional_output_tensor=_poisoned_out(device, ref_t.shape),
        )
        _assert_exact(egp_t, ref_t, f"indexed down compact-A k 8 pcn 8 G {groups} (poisoned output)")
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
            # NaN-poisoned bf16 output: Nt 24 with per_core_N 1 -> every core owns one tile column per slot
            egp_t = _sparse(
                x_t,
                w_t,
                mask_t,
                _pc(grid, bw, 1),
                dtype=ttnn.bfloat16,
                expert_groups=groups,
                optional_output_tensor=_poisoned_out(device, ref1_t.shape, ttnn.bfloat16),
            )
            _assert_exact(egp_t, ref1_t, f"gpt-oss nnz {nnz} bw {bw} pcn 1 G {groups} (poisoned output)")
            egp_t.deallocate(True)
        ref2_t = _sparse(x_t, w_t, mask_t, _pc((6, 2), bw, 2, 2), dtype=ttnn.bfloat16)
        _assert_exact(ref2_t, ref1_t, f"gpt-oss nnz {nnz} bw {bw}: legacy pcn 2 vs legacy pcn 1")
        for groups, grid in ((4, (8, 6)), (6, (8, 9))):
            egp_t = _sparse(
                x_t,
                w_t,
                mask_t,
                _pc(grid, bw, 2, 2),
                dtype=ttnn.bfloat16,
                expert_groups=groups,
                optional_output_tensor=_poisoned_out(device, ref2_t.shape, ttnn.bfloat16),
            )
            _assert_exact(egp_t, ref2_t, f"gpt-oss nnz {nnz} bw {bw} pcn 2 G {groups} (poisoned output)")
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
    egp_t = _sparse(
        x_t,
        w_t,
        mask_t,
        _pc((10, 4), 1, 1),
        nnz=top_k,
        dtype=ttnn.bfloat16,
        expert_groups=4,
        optional_output_tensor=_poisoned_out(device, ref_t.shape, ttnn.bfloat16),
    )
    _assert_exact(egp_t, ref_t, "gemma4-style nnz 4 bw 1 G 4 (poisoned output)")


# ----------------------------------------------------------------------------------------------------------
# 11. program cache: distinct entries per factory, hit on a new mask
# ----------------------------------------------------------------------------------------------------------
def test_program_cache_exact(device, solar_torch):
    """Cache entries: the EGP call launches one program (no FILL pass with the writer zero-fill), the legacy call
    its FILL + program; a new mask hits both caches."""
    x, x_t, w_t = _gate_up_tensors(device, solar_torch)
    mask_a, _ = _mask_dev(device, 72, "random", seed=43)
    mask_b, _ = _mask_dev(device, 34, "random", seed=47)
    pc_legacy, pc_egp = _pc((5, 2), 128, 1), _pc((11, 10), 128, 1)
    exp_egp, exp_legacy = _expected_new_cache_entries(egp_first=True)
    n0 = device.num_program_cache_entries()
    egp_a = _sparse(x_t, w_t, mask_a, pc_egp, expert_groups=11)
    n1 = device.num_program_cache_entries()
    assert n1 - n0 == exp_egp, (
        f"first EGP call: expected {exp_egp} new program cache entr{'y' if exp_egp == 1 else 'ies'} "
        f"(zero_fill {'on: sparse matmul only, no FILL program' if ZERO_FILL_ON else 'off: FILL + sparse matmul'}), "
        f"got {n1 - n0}"
    )
    ref_a = _sparse(x_t, w_t, mask_a, pc_legacy)
    n2 = device.num_program_cache_entries()
    assert (
        n2 - n1 == exp_legacy
    ), f"first legacy call: expected {exp_legacy} new entries (FILL + program), got {n2 - n1}"
    _assert_exact(egp_a, ref_a, "program cache: first EGP run")
    ref_b = _sparse(x_t, w_t, mask_b, pc_legacy)
    egp_b = _sparse(x_t, w_t, mask_b, pc_egp, expert_groups=11)
    n3 = device.num_program_cache_entries()
    assert n3 == n2, f"a new mask must hit the cached programs, got {n3 - n2} new entries"
    _assert_exact(egp_b, ref_b, "program cache: EGP run on a different mask (cache hit)")
    assert not torch.equal(ttnn.to_torch(egp_a), ttnn.to_torch(egp_b)), "different masks must give different outputs"
    # the down at the production grid: the FILL program's hash does not include the tensor shape, so the legacy
    # gate|up call above already cached it for this dtype / layout / memory config -> +1 for the legacy down in
    # both arms, and +1 for the EGP down (its own program; with the knob off its FILL is the shared entry).
    act_t = _to_dev(device, solar_torch["act_down"], ttnn.bfloat8_b)
    w_down_t = _to_dev(device, solar_torch["w_down"], ttnn.bfloat8_b)
    egp_d = _sparse(act_t, w_down_t, mask_a, _pc((11, 8), 5, 16, 8), is_input_a_sparse=True, expert_groups=11)
    n4 = device.num_program_cache_entries()
    assert n4 - n3 == 1, f"first EGP down call: expected 1 new entry (its program only), got {n4 - n3}"
    ref_d = _sparse(act_t, w_down_t, mask_a, _pc((8, 8), 5, 2, 2), is_input_a_sparse=True)
    n5 = device.num_program_cache_entries()
    assert n5 - n4 == 1, f"first legacy down call: expected 1 new entry (FILL already cached), got {n5 - n4}"
    _assert_exact(egp_d, ref_d, "program cache: EGP down")


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
    # broadcast A with several row blocks (M 64, per_core_M 1 -> 2 h blocks): the resident in0 is multicast once,
    # so every core must need the same rows (the unguarded case computed rows >= 1 on row block 0's activations)
    with expect_error(RuntimeError, "per_core_M == Mt"):
        _sparse(x64_t, w_t, mask_t, _pc((10, 2), 128, 1), expert_groups=1)
    # writer zero-fill preconditions: per_core_M must divide Mt (M 96 = 3 tiles, per_core_M 4) ...
    act96_t = _to_dev(device, torch.randn(1, E, 96, IP), ttnn.bfloat8_b)
    with expect_error(RuntimeError, "multiple of per_core_M"):
        _sparse(
            act96_t,
            w_down_t,
            mask_t,
            _pc((11, 8), 5, 16, 8, per_core_M=4),
            is_input_a_sparse=True,
            expert_groups=11,
        )
    # ... and the output must be interleaved (the EGP writer addresses interleaved pages)
    sharded_out = ttnn.create_sharded_memory_config(
        (1, E, 32, H),
        core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    with expect_error(RuntimeError, "INTERLEAVED output"):
        _sparse(
            act_t,
            w_down_t,
            mask_t,
            _pc((11, 8), 5, 16, 8),
            is_input_a_sparse=True,
            expert_groups=11,
            memory_config=sharded_out,
        )


# ----------------------------------------------------------------------------------------------------------
# 13. writer zero-fill: poisoned expanded outputs, every nnz x pattern, gate|up (bcast A) in bfp8 and bf16
# ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
def test_zero_fill_poisoned_gate_up(device, solar_torch, out_dtype):
    x, x_t, w_t = _gate_up_tensors(device, solar_torch)
    checked = 0
    for nnz, pattern in ZF_MASKS:
        mask_t, _ = _mask_dev(device, nnz, pattern, seed=61 + nnz)
        ref_t = _sparse(x_t, w_t, mask_t, _pc((5, 2), 128, 1), dtype=out_dtype)
        if nnz == 0:
            assert (ttnn.to_torch(ref_t) == 0).all(), "legacy nnz 0 output must be all zeros"
        for groups in (1, 5, 11):
            out_t = _poisoned_out(device, ref_t.shape, out_dtype)
            if checked == 0:
                _assert_poison_intact(out_t, f"gate|up {out_dtype}")
            egp_t = _sparse(
                x_t,
                w_t,
                mask_t,
                _pc(_egp_grid(10, groups), 128, 1),
                dtype=out_dtype,
                expert_groups=groups,
                optional_output_tensor=out_t,
            )
            _assert_exact(egp_t, ref_t, f"zero-fill gate|up nnz {nnz} {pattern} G {groups} out {out_dtype}")
            egp_t.deallocate(True)
            checked += 1
        ref_t.deallocate(True)
        mask_t.deallocate(True)
    logger.info(f"zero-fill gate|up out {out_dtype}: {checked} poisoned EGP outputs exact vs legacy")


# ----------------------------------------------------------------------------------------------------------
# 14. writer zero-fill: poisoned expanded outputs, the (A+B sparse) down at the production and other grids
# ----------------------------------------------------------------------------------------------------------
ZF_DOWN_CONFIGS = [
    # (per_core_N, out_subblock_w, groups, grid)
    (16, 8, 11, (11, 8)),  # production b32 down (88 cores)
    (8, 8, 5, (8, 10)),
    (8, 8, 4, (8, 8)),
    (2, 2, 1, (8, 8)),
    (4, 4, 2, (8, 8)),
]


@pytest.mark.parametrize("nnz", [0, 1, 8, 72, 128])
def test_zero_fill_poisoned_down(device, solar_torch, nnz):
    act_t = _to_dev(device, solar_torch["act_down"], ttnn.bfloat8_b)
    w_t = _to_dev(device, solar_torch["w_down"], ttnn.bfloat8_b)
    checked = 0
    for nnz_, pattern in ZF_MASKS:
        if nnz_ != nnz:
            continue
        mask_t, _ = _mask_dev(device, nnz, pattern, seed=67 + nnz)
        ref_t = _sparse(act_t, w_t, mask_t, _pc((8, 8), 5, 2, 2), is_input_a_sparse=True)
        if nnz == 0:
            assert (ttnn.to_torch(ref_t) == 0).all(), "legacy nnz 0 output must be all zeros"
        for pcn, osw, groups, grid in ZF_DOWN_CONFIGS:
            out_t = _poisoned_out(device, ref_t.shape)
            if checked == 0:
                _assert_poison_intact(out_t, "down bfp8")
            egp_t = _sparse(
                act_t,
                w_t,
                mask_t,
                _pc(grid, 5, pcn, osw),
                is_input_a_sparse=True,
                expert_groups=groups,
                optional_output_tensor=out_t,
            )
            _assert_exact(egp_t, ref_t, f"zero-fill down nnz {nnz} {pattern} pcn {pcn} osw {osw} G {groups} {grid}")
            egp_t.deallocate(True)
            checked += 1
        ref_t.deallocate(True)
        mask_t.deallocate(True)
    logger.info(f"zero-fill down nnz {nnz}: {checked} poisoned EGP outputs exact vs legacy")


# ----------------------------------------------------------------------------------------------------------
# 15. M = 128 (per_core_M 4: 4 tile rows per slot per core), gate|up and down, poisoned
# ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nnz", [8, 72])
def test_zero_fill_m128(device, solar_torch, nnz):
    g = torch.Generator().manual_seed(71)
    mask_t, _ = _mask_dev(device, nnz, "random", seed=73 + nnz)
    # down: [1,128,128,160] x [1,128,160,4096], per_core_M 4 on both grids. The two 71 MB outputs live in DRAM
    # (two of them do not fit L1 next to the program's CBs); the zero writes then go to DRAM, which the op allows.
    act_t = _to_dev(device, torch.randn(1, E, 128, IP, generator=g), ttnn.bfloat8_b)
    w_t = _to_dev(device, solar_torch["w_down"], ttnn.bfloat8_b)
    ref_d = _sparse(
        act_t,
        w_t,
        mask_t,
        _pc((8, 8), 5, 2, 2, per_core_M=4),
        is_input_a_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert tuple(ref_d.shape) == (1, E, 128, H)
    egp_d = _sparse(
        act_t,
        w_t,
        mask_t,
        _pc((11, 8), 5, 16, 8, per_core_M=4),
        is_input_a_sparse=True,
        expert_groups=11,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        optional_output_tensor=_poisoned_out(device, ref_d.shape, mem=ttnn.DRAM_MEMORY_CONFIG),
    )
    _assert_exact(egp_d, ref_d, f"zero-fill down M 128 nnz {nnz} pcn16 osw8 G 11 (poisoned output)")
    egp_d.deallocate(True)
    act_t.deallocate(True)
    # gate|up: [1,1,128,4096] bf16 x w_gu, per_core_M 4, bw 32 (resident in0 = 4 x 128 tiles = 1 MB). No legacy
    # reference here: the legacy 1D mcast factory returns non-finite values for a broadcast A at per_core_M 4
    # (pre-existing legacy-factory bug, reproduced by the phase-3c M = 128 diagnostic), so the poisoned G 11 output is checked directly --
    # every invalid slot exactly zero, every valid slot vs the fp32 matmul -- and against the EGP G 1 result.
    x128 = torch.randn(1, 1, 128, H, generator=g).to(torch.bfloat16)
    x_t = _to_dev(device, x128, ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
    w_gu_t = _to_dev(device, solar_torch["w_gu"], ttnn.bfloat8_b)
    ref_g = _sparse(x_t, w_gu_t, mask_t, _pc((10, 1), 32, 1, per_core_M=4), expert_groups=1)
    assert tuple(ref_g.shape) == (1, 1, 1, E, 128, N_GU)
    egp_g = _sparse(
        x_t,
        w_gu_t,
        mask_t,
        _pc((11, 10), 32, 1, per_core_M=4),
        expert_groups=11,
        optional_output_tensor=_poisoned_out(device, ref_g.shape),
    )
    _assert_exact(egp_g, ref_g, f"zero-fill gate|up M 128 nnz {nnz} bw 32 G 11 (poisoned output) vs EGP G 1")
    out = ttnn.to_torch(egp_g).float().reshape(E, 128, N_GU)
    _, idx = _mask_torch(nnz, "random", seed=73 + nnz)
    invalid = [e for e in range(E) if e not in idx]
    assert (
        out[invalid] == 0
    ).all(), f"M 128 gate|up: {int((out[invalid] != 0).sum())} non-zero elements in invalid slots"
    w_rt = ttnn.to_torch(w_gu_t).float()
    for e in idx[:4]:
        assert_numeric_metrics(
            x128.float()[0, 0] @ w_rt[0, e],
            out[e],
            atol=0.05 * H,
            rtol=10.0 * H,
            frobenius_threshold=0.05 * H,
            pcc_threshold=0.99,
        )


# ----------------------------------------------------------------------------------------------------------
# 16. ragged last column block: the zero walk must reproduce the writer's last-block / last-subblock guards
# ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nnz", [8, 72])
def test_zero_fill_ragged_last_block(device, solar_torch, nnz):
    act_t = _to_dev(device, solar_torch["act_down"], ttnn.bfloat8_b)
    w_t = _to_dev(device, solar_torch["w_down"], ttnn.bfloat8_b)
    mask_t, _ = _mask_dev(device, nnz, "random", seed=79 + nnz)
    ref_t = _sparse(act_t, w_t, mask_t, _pc((8, 8), 5, 2, 2), is_input_a_sparse=True)
    for pcn, osw, groups, grid, what in (
        # Nt 128 = 10 x 12 + 8: 11 column blocks, the last one 8 tiles = one full 6-tile subblock + a 2-of-6 partial
        (12, 6, 2, (11, 2), "last block 8 = 6 + partial 2"),
        # Nt 128 = 5 x 24 + 8: 6 column blocks, the last one a single full 8-tile subblock of a 3-subblock block
        (24, 8, 5, (6, 5), "last block 8 = one of three subblocks"),
    ):
        egp_t = _sparse(
            act_t,
            w_t,
            mask_t,
            _pc(grid, 5, pcn, osw),
            is_input_a_sparse=True,
            expert_groups=groups,
            optional_output_tensor=_poisoned_out(device, ref_t.shape),
        )
        _assert_exact(egp_t, ref_t, f"zero-fill ragged down nnz {nnz} pcn {pcn} osw {osw} G {groups} ({what})")
        egp_t.deallocate(True)


# ----------------------------------------------------------------------------------------------------------
# 17. one poisoned output reused across eager runs with changing masks (valid -> invalid slots must be re-zeroed)
# ----------------------------------------------------------------------------------------------------------
def test_zero_fill_stale_across_runs(device, solar_torch):
    x, x_t, w_t = _gate_up_tensors(device, solar_torch)
    act_t = _to_dev(device, solar_torch["act_down"], ttnn.bfloat8_b)
    w_down_t = _to_dev(device, solar_torch["w_down"], ttnn.bfloat8_b)
    g = torch.Generator().manual_seed(83)
    perm = torch.randperm(E, generator=g)
    ids_a, ids_b = perm[:72].tolist(), perm[72:80].tolist()  # B disjoint from A

    def mask_from(ids):
        m = torch.zeros(1, 1, 1, E)
        if ids:
            m[..., ids] = 0.125 + 0.5 * torch.rand(len(ids), generator=g)
        return _to_dev(device, m.to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

    masks = [("A nnz 72", mask_from(ids_a)), ("B nnz 8 disjoint", mask_from(ids_b)), ("nnz 0", mask_from([]))]
    masks.append(("A again", masks[0][1]))
    out_gu = _poisoned_out(device, (1, 1, 1, E, 32, N_GU))
    out_dn = _poisoned_out(device, (1, E, 32, H))
    for step, (what, mask_t) in enumerate(masks):
        ref_g = _sparse(x_t, w_t, mask_t, _pc((5, 2), 128, 1))
        egp_g = _sparse(x_t, w_t, mask_t, _pc((11, 10), 128, 1), expert_groups=11, optional_output_tensor=out_gu)
        _assert_exact(egp_g, ref_g, f"stale-reuse gate|up step {step} ({what})")
        ref_d = _sparse(act_t, w_down_t, mask_t, _pc((8, 8), 5, 2, 2), is_input_a_sparse=True)
        egp_d = _sparse(
            act_t,
            w_down_t,
            mask_t,
            _pc((11, 8), 5, 16, 8),
            is_input_a_sparse=True,
            expert_groups=11,
            optional_output_tensor=out_dn,
        )
        _assert_exact(egp_d, ref_d, f"stale-reuse down step {step} ({what})")
        ref_g.deallocate(True)
        ref_d.deallocate(True)


# ----------------------------------------------------------------------------------------------------------
# 18. trace replay with in-place mask updates (how Solar decode runs): the output buffer is fixed in the trace and
#     nothing but the kernel zeroes it between replays
# ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 * 1024 * 1024}], indirect=True)
def test_zero_fill_trace_replay(device, solar_torch):
    x, x_t, w_t = _gate_up_tensors(device, solar_torch)
    act_t = _to_dev(device, solar_torch["act_down"], ttnn.bfloat8_b)
    w_down_t = _to_dev(device, solar_torch["w_down"], ttnn.bfloat8_b)
    g = torch.Generator().manual_seed(89)

    def host_mask(ids):
        m = torch.zeros(1, 1, 1, E)
        if len(ids):
            m[..., ids] = 0.125 + 0.5 * torch.rand(len(ids), generator=g)
        return ttnn.from_torch(m.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    perm = torch.randperm(E, generator=g)
    a_ids, b_ids = perm[:72].tolist(), perm[72:128].tolist()  # A / B disjoint, A u B = all slots
    schedule = [("A", a_ids), ("B", b_ids), ("A", a_ids), ("B", b_ids), ("nnz 0", []), ("nnz 128", list(range(E)))]
    schedule += [("nnz 127", list(range(1, E))), ("nnz 1", [perm[5].item()]), ("nnz 1 @0", [0]), ("nnz 1 @127", [127])]
    while len(schedule) < 40:
        nnz = int(torch.randint(1, 129, (1,), generator=g))
        schedule.append((f"random nnz {nnz}", torch.randperm(E, generator=g)[:nnz].tolist()))

    mask_t = _to_dev(device, ttnn.to_torch(host_mask(a_ids)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)

    def run_all():
        ref_g = _sparse(x_t, w_t, mask_t, _pc((5, 2), 128, 1))
        egp_g = _sparse(x_t, w_t, mask_t, _pc((11, 10), 128, 1), expert_groups=11)
        ref_d = _sparse(act_t, w_down_t, mask_t, _pc((8, 8), 5, 2, 2), is_input_a_sparse=True)
        egp_d = _sparse(act_t, w_down_t, mask_t, _pc((11, 8), 5, 16, 8), is_input_a_sparse=True, expert_groups=11)
        return ref_g, egp_g, ref_d, egp_d

    for t in run_all():  # warm-up: compile + program cache
        t.deallocate(True)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    ref_g, egp_g, ref_d, egp_d = run_all()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    n_cache = device.num_program_cache_entries()
    for i, (what, ids) in enumerate(schedule):
        ttnn.copy_host_to_device_tensor(host_mask(ids), mask_t)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        _assert_exact(egp_g, ref_g, f"trace replay {i} ({what}): EGP gate|up vs in-trace legacy")
        _assert_exact(egp_d, ref_d, f"trace replay {i} ({what}): EGP down vs in-trace legacy")
        # independent reference: a fresh eager legacy call on the same mask
        fresh_g = _sparse(x_t, w_t, mask_t, _pc((5, 2), 128, 1))
        fresh_d = _sparse(act_t, w_down_t, mask_t, _pc((8, 8), 5, 2, 2), is_input_a_sparse=True)
        _assert_exact(egp_g, fresh_g, f"trace replay {i} ({what}): EGP gate|up vs fresh legacy")
        _assert_exact(egp_d, fresh_d, f"trace replay {i} ({what}): EGP down vs fresh legacy")
        fresh_g.deallocate(True)
        fresh_d.deallocate(True)
    assert device.num_program_cache_entries() == n_cache, "replays must not add program cache entries"
    ttnn.release_trace(device, tid)
    logger.info(f"trace replay: {len(schedule)} replays with in-place mask updates, all exact")
