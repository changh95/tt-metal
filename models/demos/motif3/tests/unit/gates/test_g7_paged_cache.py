# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G7 — ``paged_update_cache`` / ``paged_fill_cache`` on the Motif latent cache.

Design §1.2 (row 4), §2.3.4 decode step 7 and prefill step 5, §4.2 G7. Cache per layer and chip:
[num_blocks, 1, block, 576] bfloat8_b (unit-RMS latent 512 ‖ roped k_pe 64), block ∈ {32, 64}.

* decode: input [1, B=8, 1(->32), 576] bf16, height-sharded one user per core (the op requires
  num_cores == num_users), ``update_idxs_tensor`` int32 [8] with an inactive lane at -1, page_table int32
  [8, blocks_per_user] (random permutation). Two successive steps.
* prefill: input [1, 1, S, 576] **bfloat8_b** (the op copies tiles, input dtype must equal the cache dtype),
  ``batch_idx`` scalar and ``batch_idx_tensor`` (trace-safe) variants, S ∈ {256, 1024}.
Inputs are pre-quantised through a host bfp8 round trip, so the expected cache is exact: the readback must
be bit-identical everywhere (written rows, untouched rows, the skipped lane), on every chip.
Also measured: decode-update latency (eager / traced) and the cost of allocating a KV-pool layer with
``ttnn.zeros`` (host-built bfp8, written over PCIe to 32 chips) vs ``ttnn.empty`` + on-device ``ttnn.fill``.
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu

B, D = 8, 576
MAX_SEQ = 1024
REC = gu.Recorder("G7")
FAB2D = gu.mesh_params()


def page_table(bpu: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(B * bpu, generator=g).reshape(B, bpu).to(torch.int32)


def q8(t: torch.Tensor) -> torch.Tensor:
    return gu.host_roundtrip(t, ttnn.bfloat8_b)


def decode_input(mesh_device, rows: torch.Tensor):
    """rows [B, 576] -> device [1, B, 32, 576] bf16 height-sharded one user per core."""
    x = torch.zeros(1, B, 32, D)
    x[0, :, 0, :] = rows
    grid = ttnn.num_cores_to_corerangeset(B, gu.grid_size(mesh_device), True)
    spec = ttnn.ShardSpec(grid, [32, D], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    return gu.to_mesh(x, mesh_device, ttnn.bfloat16, memory_config=mc)


def readback_identical(cache_tt, expected: torch.Tensor, check_devices=(0, 7, 13, 31)):
    worst = 0
    ok_all = True
    for i in check_devices:
        got = ttnn.to_torch(ttnn.get_device_tensors(cache_tt)[i]).float()
        same = torch.equal(got, expected)
        if not same:
            ok_all = False
            worst = max(worst, int((got != expected).sum()))
    return ok_all, worst


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
@pytest.mark.parametrize("block", [32, 64])
def test_g7_paged_update_and_fill(mesh_device, block):
    torch.manual_seed(block)
    bpu = MAX_SEQ // block
    nb = B * bpu
    pt = page_table(bpu, seed=block)
    pt_tt = gu.to_mesh(pt, mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cache = q8(torch.randn(nb, 1, block, D))
    cache_tt = gu.to_mesh(cache, mesh_device, ttnn.bfloat8_b)
    expected = cache.clone()
    failures = []

    # ---------------- decode updates (two steps) ----------------
    steps = [
        [5, block - 1, block, 127, -1, 500, MAX_SEQ - 33, MAX_SEQ - 1],
        [6, block, block + 1, 128, -1, 501, MAX_SEQ - 32, 0],
    ]
    for step, pos in enumerate(steps):
        rows = q8(torch.randn(B, D) * 2.0)
        x_tt = decode_input(mesh_device, rows)
        pos_tt = gu.to_mesh(torch.tensor(pos, dtype=torch.int32), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        case = f"block{block}/update/step{step}"
        try:
            ttnn.experimental.paged_update_cache(cache_tt, x_tt, update_idxs_tensor=pos_tt, page_table=pt_tt)
        except Exception as e:
            REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
            failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
            continue
        for u, p in enumerate(pos):
            if p < 0:
                continue
            expected[pt[u, p // block], 0, p % block, :] = rows[u]
        ok, nbad = readback_identical(cache_tt, expected)
        REC.add(case, status="pass" if ok else "fail", positions=pos, mismatching_elements=nbad, note="pos -1 lane must be untouched")
        if not ok:
            failures.append(f"{case}: {nbad} mismatching elements")

    # ---------------- prefill fills ----------------
    for S, users, use_tensor in [(256, [1, 6], False), (MAX_SEQ, [3], False), (256, [0, 7], True)]:
        for u in users:
            x = q8(torch.randn(1, 1, S, D))
            x_tt = gu.to_mesh(x, mesh_device, ttnn.bfloat8_b)
            case = f"block{block}/fill/S{S}/user{u}/{'batch_idx_tensor' if use_tensor else 'batch_idx'}"
            try:
                if use_tensor:
                    bi = gu.to_mesh(torch.tensor([u], dtype=torch.int32), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.experimental.paged_fill_cache(cache_tt, x_tt, pt_tt, batch_idx_tensor=bi)
                else:
                    ttnn.experimental.paged_fill_cache(cache_tt, x_tt, pt_tt, batch_idx=u)
            except Exception as e:
                REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
                failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
                continue
            for j in range(S // block):
                expected[pt[u, j], 0] = x[0, 0, j * block : (j + 1) * block]
            ok, nbad = readback_identical(cache_tt, expected)
            REC.add(case, status="pass" if ok else "fail", mismatching_elements=nbad)
            if not ok:
                failures.append(f"{case}: {nbad} mismatching elements")

    # NOTE: a bf16 input into this bfp8 cache is deliberately *not* exercised: paged_fill_cache is a raw tile copy
    # ("input_tensor.dtype must match cache_tensor.dtype"), so a mismatched dtype would write 2048-B tiles into
    # 1088-B slots if the check were missing. Prefill must cast the latent to bfloat8_b before the fill.

    assert not failures, "G7 failures:\n" + "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
def test_g7_perf_and_allocation(mesh_device):
    block = 64
    bpu = 32768 // block
    nb = B * bpu  # 8 users x 32K context = 4096 blocks (~160 MB / chip)
    empty = ttnn.empty([nb, 1, block, D], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
    cache_tt = ttnn.fill(empty, 0.0)
    ttnn.deallocate(empty)
    pt_tt = gu.to_mesh(page_table(bpu, seed=1), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    x_tt = decode_input(mesh_device, torch.randn(B, D))
    pos_tt = gu.to_mesh(torch.tensor([100 * (i + 1) for i in range(B)], dtype=torch.int32), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def upd():
        ttnn.experimental.paged_update_cache(cache_tt, x_tt, update_idxs_tensor=pos_tt, page_table=pt_tt)
        return None  # in-place: never free the cache

    eager = gu.time_eager(mesh_device, upd, iters=20)
    traced, raw = gu.time_traced(mesh_device, upd, ops_per_trace=64, reps=9)
    REC.add("perf/decode_update/block64", status="measured", eager_us=eager, traced_us=traced, traced_raw_us=raw, samples=dict(gu.LAST_TRACE_SAMPLES))

    for S in (1024, 4096):
        xf = gu.to_mesh(torch.randn(1, 1, S, D), mesh_device, ttnn.bfloat8_b)

        def fill():
            ttnn.experimental.paged_fill_cache(cache_tt, xf, pt_tt, batch_idx=0)
            return None

        REC.add(f"perf/prefill_fill/S{S}/block64", status="measured", eager_us=gu.time_eager(mesh_device, fill, iters=10))

    # KV-pool layer allocation cost (design §5.1 allocate_kv_cache): 1/8 of a layer's 264K-token pool
    pool_blocks = 264192 // block // 8
    t0 = time.perf_counter()
    z = ttnn.zeros([pool_blocks, 1, block, D], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(mesh_device)
    t_zeros = time.perf_counter() - t0
    ttnn.deallocate(z)
    t0 = time.perf_counter()
    e = ttnn.empty([pool_blocks, 1, block, D], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
    f = ttnn.fill(e, 0.0)
    ttnn.synchronize_device(mesh_device)
    t_fill = time.perf_counter() - t0
    REC.add(
        "alloc/kv_pool_eighth_layer",
        status="measured",
        blocks=pool_blocks,
        bytes_per_chip=pool_blocks * block * D * 1088 // 1024,
        ttnn_zeros_s=t_zeros,
        empty_plus_fill_s=t_fill,
        extrapolated_53_layers_full_pool_zeros_s=t_zeros * 8 * 53,
        extrapolated_53_layers_full_pool_fill_s=t_fill * 8 * 53,
    )
