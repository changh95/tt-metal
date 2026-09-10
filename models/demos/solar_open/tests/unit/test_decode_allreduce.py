# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-level gate of the fused decode all-reduce (``SOLAR_OPEN_DECODE_CCL=fused``, phase 3e / A2, design_decode_levers.md
2.7 item 6) through the PRODUCTION API (``tt/ccl.py::CCLManager``), on the 1x8 mesh:

1. numerics per site against the fp32 sum of the 8 device-held partials and against today's composite ``ttnn.all_reduce``
   on the same operands: every replica identical on all 8 devices, max |err| within ``ERR_FLOOR_MULT`` x the output
   dtype's own requantization floor (A1 measured 1.5x for bfp8, exactly 1 ulp for bf16), mean |err| not above the
   composite's, bit-identical across two calls on the same input;
2. the two production shapes: the attention site's bf16 width-sharded o_proj partial (``[32, 512]`` on 8 cores ->
   typecast bfloat8_b -> fused -> interleaved) and the MoE site's bfp8 L1-interleaved partial (``[1, 1, 32, H]`` and the
   single-user ``[1, 1, 1, H]`` logical shape) through ``fused_decode_all_reduce_interleaved``; ``fused_decode_applies``
   refuses the prefill shapes / DRAM tensors;
3. the stale-replica hazard class: ``GATE_ITERS`` back-to-back [attention, MoE] iterations with the fixed site -> pair
   map and 4 rotating inputs (host read every 10 iterations), then a trace of 10 iterations replayed ``GATE_REPLAYS``
   times with fresh data written IN PLACE into the persistent inputs before every replay (``copy_host_to_device_tensor``;
   the traced pattern of the decode loop): every output identical on the 8 devices and within the error bound;
4. ``reset_fused_semaphores`` between calls is harmless (the Model.switch_mode hook).

    pytest models/demos/solar_open/tests/unit/test_decode_allreduce.py -k 1x8 -x -p no:cacheprovider
    SOLAR_OPEN_NUM_DEVICES=8 pytest models/demos/solar_open/tests/unit/test_decode_allreduce.py -k host   # host only
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.solar_open.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.solar_open.tt.ccl import (
    DECODE_CCL_DEFAULT,
    DECODE_CCL_ENV,
    FUSED_DECODE_GRID,
    FUSED_SITE_ATTENTION,
    FUSED_SITE_MOE,
    CCLManager,
    decode_ccl_mode,
    fused_decode_grid_fits,
)

B, H, TP = 32, 4096, 8
N_INPUTS = 4
GATE_ITERS = int(os.getenv("SOLAR_OPEN_DECODE_CCL_GATE_ITERS", "200"))
GATE_REPLAYS = int(os.getenv("SOLAR_OPEN_DECODE_CCL_GATE_REPLAYS", "40"))
GATE_CHUNK = 10
TRACE_ITERS = 10
ERR_FLOOR_MULT = 4.0  # max |err| bound in units of the output dtype's requantization floor (A1: bfp8 1.5x, bf16 1x)
OPROJ_GRID = (8, 1)  # the decode o_proj emits [32, 512] width shards on 8 cores


# ---------------------------------------------------------------------------------------------------------- host
def test_host_decode_ccl_mode(monkeypatch, expect_error):
    monkeypatch.delenv(DECODE_CCL_ENV, raising=False)
    assert decode_ccl_mode() == DECODE_CCL_DEFAULT == "fused"  # the default since phase 3e / A2
    monkeypatch.setenv(DECODE_CCL_ENV, "")
    assert decode_ccl_mode() == "fused"
    monkeypatch.setenv(DECODE_CCL_ENV, "composite")
    assert decode_ccl_mode() == "composite"
    monkeypatch.setenv(DECODE_CCL_ENV, " Fused ")
    assert decode_ccl_mode() == "fused"
    assert decode_ccl_mode("composite") == "composite"  # an explicit value wins over the environment
    with expect_error(ValueError, DECODE_CCL_ENV):
        decode_ccl_mode("ring")


def test_host_fused_grid_fits():
    class Grid:
        def __init__(self, x, y):
            self.x, self.y = x, y

    assert fused_decode_grid_fits(4096, Grid(11, 10))  # Solar-Open on the P150 compute grid: [32, 128] shards
    assert fused_decode_grid_fits(4096, None)
    assert not fused_decode_grid_fits(4096 + 32, Grid(11, 10))  # not a multiple of 32 cores x 32 columns
    assert not fused_decode_grid_fits(4096, Grid(7, 10))  # 8x4 does not fit a 7-wide grid
    assert FUSED_DECODE_GRID == (8, 4)


# -------------------------------------------------------------------------------------------------------- device
def ws_config(grid, width):
    gx, gy = grid
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, width // (gx * gy)),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class Partials:
    """``n`` rotating per-device partials: device d holds slice d of a ``[1, TP, rows, H]`` randn tensor (dtype-rounded
    at upload) in ``memory_config``; ``ref[i]`` is the fp32 sum of the 8 device-held slices."""

    def __init__(self, mesh_device, dtype, memory_config, seed, n=N_INPUTS, rows=B):
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.memory_config = memory_config
        self.rows = rows
        g = torch.Generator().manual_seed(seed)
        self.host = [torch.randn(1, TP, rows, H, generator=g) for _ in range(n)]
        self.dev = [self.upload(t) for t in self.host]
        self.ref = [self.reference(t) for t in self.host]

    def host_tensor(self, t):
        mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 1), mesh_shape=tuple(self.mesh_device.shape))
        return ttnn.from_torch(t, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

    def upload(self, t):
        return ttnn.to_device(self.host_tensor(t), self.mesh_device, memory_config=self.memory_config)

    def reference(self, t):
        slices = [
            ttnn.to_torch(ttnn.from_torch(t[:, d : d + 1], dtype=self.dtype, layout=ttnn.TILE_LAYOUT)).float()
            for d in range(TP)
        ]
        return torch.stack(slices, 0).sum(0)

    def update(self, i, t):
        """In-place host -> device write into persistent input ``i`` (the decode trace's input pattern)."""
        ttnn.copy_host_to_device_tensor(self.host_tensor(t), self.dev[i])
        self.host[i] = t
        self.ref[i] = self.reference(t)

    def deallocate(self):
        for x in self.dev:
            x.deallocate(True)


def requant_floor(ref, dtype):
    q = ttnn.to_torch(ttnn.from_torch(ref, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()
    d = (q - ref).abs()
    return float(d.max()), float(d.mean())


def replicas(out):
    """(identical on all devices, max |device d - device 0|, device 0's fp32 values)."""
    vals = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
    identical = all(torch.equal(v, vals[0]) for v in vals[1:])
    max_diff = max((float((v.float() - vals[0].float()).abs().max()) for v in vals[1:]), default=0.0)
    return identical, max_diff, vals[0].float()


def check(out, ref, label=""):
    """Replica identity + numerics of device 0 vs the fp32 reference; ``ok`` = identical and within the error bound."""
    identical, replica_max, v0 = replicas(out)
    v0 = v0.reshape(ref.shape)
    err = (v0 - ref).abs()
    floor_max, floor_mean = requant_floor(ref, out.dtype)
    _, pcc_s = comp_pcc(ref, v0, 0.99)
    pcc = float(str(pcc_s).split()[-1]) if not isinstance(pcc_s, float) else pcc_s
    res = {
        "identical": identical,
        "replica_max": replica_max,
        "max_err": float(err.max()),
        "mean_err": float(err.mean()),
        "pcc": pcc,
        "floor_max": floor_max,
        "floor_mean": floor_mean,
        "checksum": float(v0.double().sum()),
    }
    res["ok"] = identical and res["max_err"] <= ERR_FLOOR_MULT * max(floor_max, 1e-6)
    if label:
        logger.info(
            f"[decode_ccl {label}] identical={identical} replica_max={replica_max:.3g} max|err|={res['max_err']:.4g} "
            f"mean|err|={res['mean_err']:.4g} pcc={pcc:.6f} floor(max/mean)={floor_max:.4g}/{floor_mean:.4g} "
            f"checksum={res['checksum']:.6f}"
        )
    return res


def composite(x):
    return ttnn.all_reduce(x, num_links=1, topology=ttnn.Topology.Ring, cluster_axis=1)


@parametrize_mesh_with_fabric([(1, 8)])
def test_fused_decode_allreduce(mesh_device, device_params, reset_seeds):
    if tuple(mesh_device.shape) != (1, 8):
        pytest.skip("the fused decode all-reduce is sized for the 1x8 TP=8 mesh")
    ccl = CCLManager(mesh_device, num_links=1, decode_ccl="fused")
    assert ccl.fused_decode_allreduce
    pool = ccl.ensure_fused_pool(H)  # what Model.__init__ does, before any trace capture
    assert pool is ccl.fused_pool and ttnn.bfloat8_b in pool.buffers
    assert pool.pair(FUSED_SITE_ATTENTION) != pool.pair(FUSED_SITE_MOE)
    oproj_ws = ws_config(OPROJ_GRID, H)
    failures = []

    # Persistent interleaved inputs (the traced pattern writes into them in place); the site layouts are produced
    # inside the chains, like the model's o_proj (width-sharded) and fast_reduce_nc (L1 interleaved) outputs.
    xa = Partials(mesh_device, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, seed=11)
    xm = Partials(mesh_device, ttnn.bfloat8_b, ttnn.L1_MEMORY_CONFIG, seed=12)

    def site_a(x):  # tt/attention/decode.py: o_proj partial bf16 WS 8x1 -> typecast bfp8 -> fused (pair 0) -> s2i
        y = ttnn.to_memory_config(x, oproj_ws)
        z = ttnn.typecast(y, ttnn.bfloat8_b)
        y.deallocate(True)
        r = ccl.fused_decode_all_reduce(z, FUSED_SITE_ATTENTION, cluster_axis=1)
        z.deallocate(True)
        out = ttnn.to_memory_config(r, ttnn.L1_MEMORY_CONFIG)
        r.deallocate(True)
        return out

    def site_a_composite(x):  # today: s2i (here the input is interleaved already) + typecast + composite
        z = ttnn.typecast(x, ttnn.bfloat8_b)
        out = composite(z)
        z.deallocate(True)
        return out

    def site_m(x):  # tt/experts/operations.py (non-consuming variant): i2s 8x4 -> fused (pair 1) -> s2i
        y = ttnn.to_memory_config(x, pool.output_memory_config)
        r = ccl.fused_decode_all_reduce(y, FUSED_SITE_MOE, cluster_axis=1)
        y.deallocate(True)
        out = ttnn.to_memory_config(r, ttnn.L1_MEMORY_CONFIG)
        r.deallocate(True)
        return out

    # 1. numerics per site vs the fp32 sum and vs the composite, determinism ---------------------------------------
    for label, fused_fn, comp_fn, xs in (
        ("A", site_a, site_a_composite, xa),
        ("M", site_m, composite, xm),
    ):
        for i in range(N_INPUTS):
            f1 = fused_fn(xs.dev[i])
            f2 = fused_fn(xs.dev[i])
            c = comp_fn(xs.dev[i])
            rf = check(f1, xs.ref[i], f"{label} fused in{i}")
            rc = check(c, xs.ref[i], f"{label} composite in{i}")
            same = torch.equal(
                ttnn.to_torch(ttnn.get_device_tensors(f1)[0]), ttnn.to_torch(ttnn.get_device_tensors(f2)[0])
            )
            if not rf["ok"]:
                failures.append(f"site {label} input {i}: fused {rf}")
            if not same:
                failures.append(f"site {label} input {i}: two fused calls on the same input differ")
            if rf["mean_err"] > rc["mean_err"] * 1.02 + 1e-6:
                failures.append(
                    f"site {label} input {i}: fused mean |err| {rf['mean_err']:.4g} above the composite's {rc['mean_err']:.4g}"
                )
            assert f1.dtype == c.dtype == ttnn.bfloat8_b and f1.shape == c.shape, (f1.dtype, c.dtype, f1.shape, c.shape)
            assert not f1.memory_config().is_sharded() and f1.memory_config().buffer_type == ttnn.BufferType.L1
            for t in (f1, f2, c):
                t.deallocate(True)

    # 2. the production helpers on the production shapes ---------------------------------------------------------
    # b1 MoE partial: logical [1, 1, 1, H] (one user), padded to the 32-row tile; the helper consumes its input
    x1 = Partials(mesh_device, ttnn.bfloat8_b, ttnn.L1_MEMORY_CONFIG, seed=13, n=1, rows=1)
    assert tuple(x1.dev[0].shape) == (1, 1, 1, H) and tuple(x1.dev[0].padded_shape) == (1, 1, 32, H)
    assert ccl.fused_decode_applies(x1.dev[0])
    out1 = ccl.fused_decode_all_reduce_interleaved(x1.dev[0], FUSED_SITE_MOE, cluster_axis=1)
    r1 = check(out1, x1.ref[0], "M helper b1 [1,1,1,H]")
    assert tuple(out1.shape) == (1, 1, 1, H) and out1.dtype == ttnn.bfloat8_b, (out1.shape, out1.dtype)
    if not r1["ok"]:
        failures.append(f"b1 helper: {r1}")
    out1.deallocate(True)
    # b32 MoE partial through the consuming helper
    x32 = Partials(mesh_device, ttnn.bfloat8_b, ttnn.L1_MEMORY_CONFIG, seed=14, n=1)
    assert ccl.fused_decode_applies(x32.dev[0])
    out32 = ccl.fused_decode_all_reduce_interleaved(x32.dev[0], FUSED_SITE_MOE, cluster_axis=1)
    r32 = check(out32, x32.ref[0], "M helper b32 [1,1,32,H]")
    if not r32["ok"]:
        failures.append(f"b32 helper: {r32}")
    out32.deallocate(True)
    # the attention site's width-sharded input applies too; prefill shapes / DRAM partials do not
    ws_in = ttnn.to_memory_config(xa.dev[0], oproj_ws)
    assert ccl.fused_decode_applies(ws_in, H)
    ws_in.deallocate(True)
    dram_in = ttnn.to_memory_config(xm.dev[0], ttnn.DRAM_MEMORY_CONFIG)
    assert not ccl.fused_decode_applies(dram_in), "a DRAM partial (prefill) must keep the composite"
    dram_in.deallocate(True)
    prefill = ttnn.from_torch(
        torch.zeros(1, 1, 128, H),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    assert not ccl.fused_decode_applies(prefill), "a 128-row partial (prefill) must keep the composite"
    prefill.deallocate(True)
    off = CCLManager(mesh_device, num_links=1, decode_ccl="composite")
    assert not off.fused_decode_allreduce and not off.fused_decode_applies(xm.dev[0]) and off.fused_pool is None

    # 3a. eager gate: GATE_ITERS x [A, M] back-to-back, the fixed site -> pair map, 4 rotating inputs ------------
    bad = 0
    worst = {"A": 0.0, "M": 0.0}
    t0 = time.perf_counter()
    for start in range(0, GATE_ITERS, GATE_CHUNK):
        outs = []
        for it in range(start, min(start + GATE_CHUNK, GATE_ITERS)):
            outs.append((it, "A", site_a(xa.dev[it % N_INPUTS]), xa.ref[it % N_INPUTS]))
            outs.append((it, "M", site_m(xm.dev[it % N_INPUTS]), xm.ref[it % N_INPUTS]))
        ttnn.synchronize_device(mesh_device)
        for it, site, out, ref in outs:
            r = check(out, ref)
            worst[site] = max(worst[site], r["max_err"])
            if not r["ok"]:
                bad += 1
                if bad <= 10:
                    failures.append(f"eager gate iter {it} site {site}: {r}")
            out.deallocate(True)
    logger.info(
        f"[decode_ccl gate eager] {GATE_ITERS} iterations x 2 sites = {2 * GATE_ITERS} outputs, {bad} bad, worst "
        f"max|err| {worst}, {time.perf_counter() - t0:.1f} s"
    )

    # 3b. trace gate: TRACE_ITERS x [A, M] captured, GATE_REPLAYS replays with in-place input updates -----------
    for it in range(2):  # every program has run already (above); warm the exact traced sequence once more
        site_a(xa.dev[it % 2]).deallocate(True)
        site_m(xm.dev[it % 2]).deallocate(True)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        last_a = last_m = None
        for it in range(TRACE_ITERS):
            oa, om = site_a(xa.dev[it % 2]), site_m(xm.dev[it % 2])
            if it < TRACE_ITERS - 1:
                oa.deallocate(True)
                om.deallocate(True)
            else:
                last_a, last_m = oa, om
    finally:
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    idx = (TRACE_ITERS - 1) % 2  # the input the kept outputs were computed from
    g = torch.Generator().manual_seed(7000)
    trace_bad = 0
    walls = []
    checksums = set()
    for r in range(GATE_REPLAYS):
        for i in range(2):
            xa.update(i, torch.randn(1, TP, B, H, generator=g))
            xm.update(i, torch.randn(1, TP, B, H, generator=g))
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        walls.append((time.perf_counter() - t0) * 1e6 / TRACE_ITERS)
        for site, out, ref in (("A", last_a, xa.ref[idx]), ("M", last_m, xm.ref[idx])):
            res = check(out, ref)
            checksums.add((site, res["checksum"]))
            if not res["ok"]:
                trace_bad += 1
                if trace_bad <= 10:
                    failures.append(f"trace gate replay {r} site {site}: {res}")
    ttnn.release_trace(mesh_device, tid)
    assert len(checksums) == 2 * GATE_REPLAYS, "the kept trace outputs must follow the fresh inputs of every replay"
    logger.info(
        f"[decode_ccl gate trace] {GATE_REPLAYS} replays x {TRACE_ITERS} [A, M] iterations, in-place input updates, "
        f"{2 * GATE_REPLAYS} outputs checked, {trace_bad} bad; us per [A, M] iteration min {min(walls):.1f} "
        f"mean {sum(walls) / len(walls):.1f}"
    )

    # 4. the switch_mode hook between calls is harmless --------------------------------------------------------
    ccl.reset_fused_semaphores()
    ra = check(site_a(xa.dev[0]), xa.ref[0], "A after reset")
    rm = check(site_m(xm.dev[0]), xm.ref[0], "M after reset")
    if not (ra["ok"] and rm["ok"]):
        failures.append(f"after reset_fused_semaphores: A {ra} M {rm}")

    logger.info(
        f"[decode_ccl gate] {len(failures)} failures; pool calls {pool.calls}; eager {2 * GATE_ITERS} + trace "
        f"{2 * GATE_REPLAYS} outputs, worst max|err| A {worst['A']:.4g} M {worst['M']:.4g}"
    )
    for f in failures[:20]:
        logger.error(f"[decode_ccl gate] {f}")
    assert not failures, f"{len(failures)} fused decode all-reduce gate failures (first: {failures[0]})"
