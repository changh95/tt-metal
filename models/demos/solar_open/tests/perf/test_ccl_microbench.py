# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode all-reduce micro-benchmark on the 1x8 mesh (profiling helper, SKIPS unless ``SOLAR_OPEN_PERF_PROFILE=1``).

Measures -- instead of estimating -- the fused single-kernel ``ttnn.experimental.all_reduce_async`` against today's
composite ``ttnn.all_reduce`` (ReduceScatterMinimalDirect + AllGather) at the two decode call sites' shape, dtypes and
memory layouts (design_decode_levers.md 2.4 / 2.7, phase 3e stage A1):

* site A = the attention o_proj partial (``tt/attention/decode.py``): ``[1, 1, 32, 4096]`` bf16 WIDTH_SHARDED on the
  8 o_proj cores (``[32, 512]`` shards) -> today: sharded_to_interleaved, typecast bfloat8_b, ``ttnn.all_reduce`` ->
  bfp8 L1 interleaved (the residual add reads it);
* site M = the MoE partial (``tt/experts/operations.py::apply_tensor_parallel_allreduce``): ``[1, 1, 32, 4096]``
  bfloat8_b L1 interleaved (fast_reduce_nc output + the in-place shared-expert add) -> ``ttnn.all_reduce`` -> bfp8 L1
  interleaved.

Sections (``SOLAR_OPEN_CCL_MB_SECTIONS``, default ``KSG``):

* ``K`` kernel sweep: the fused op on a width-sharded ``[32, 128]`` x 8x4 input in bf16 and bfloat8_b over topology
  Ring / Linear x num_links 1 / 2 x fp32_dest_acc on / off x output grid 8x4 ``[32,128]`` / 8x8 ``[32,64]`` / 8x1
  ``[32,512]``, plus the 8x1 (o_proj) input layout, the composite ``ttnn.all_reduce`` (1 / 2 links) and the
  all_gather(dim 0) + fast_reduce_nc pair. Every configuration: REPS eager calls between ``mb_<name>_start/_stop``
  signposts (tracy: per-op, per-device kernel us), one trace of TRACE_CALLS back-to-back calls replayed TRACE_REPLAYS
  times (per-call wall us = what the decode trace pays), numerics of the kept output against the fp32 torch sum of the
  8 device-held inputs (max / mean |diff|, PCC, the output dtype's own requantization floor), replica identity on all
  8 devices, and determinism across two calls on the same input.
* ``S`` site chains: per-site TOTALS incl. the reshards the fused op needs (typecast / interleaved_to_sharded /
  sharded_to_interleaved) vs today's chain, timed the same way; the fused configuration is the K winner (min per-call
  trace wall) or ``SOLAR_OPEN_CCL_MB_FUSED=<ring|linear>,<links>,<fp32 0|1>,<grid>``.
* ``G`` gates (the stale-replica hazard class): >= GATE_ITERS iterations of the production pattern -- site A call, site
  M call, back-to-back without host sync, the fixed (site -> persistent buffer, global semaphore) map, 4 rotating
  inputs -- every output checked bit-exactly across the 8 replicas and against its reference; then a trace of 10
  iterations replayed GATE_REPLAYS times with in-place ``copy_host_to_device_tensor`` input updates before every
  replay; the composite runs the same trace gate as the control.

    SOLAR_OPEN_PERF_PROFILE=1 SOLAR_OPEN_PERF_OUT=/path/ccl_mb.json pytest \
        models/demos/solar_open/tests/perf/test_ccl_microbench.py -k 1x8 -x -p no:cacheprovider
    # per-op kernel us: SOLAR_OPEN_CCL_MB_SECTIONS=KS SOLAR_OPEN_CCL_MB_TRACE_TIMING=0 python -m tracy -r -p -v \
    #     -o <dir> --op-support-count 20000 -m pytest <same>

Persistent state (``Pool``): two (buffer, semaphore) pairs per (grid, input dtype), allocated once before any trace
capture; consecutive calls alternate pairs (the kernel's receiver resets the shared global semaphore to 0 after seeing
ring_size increments, so a pair reused by two in-flight calls of neighbouring devices would over-count -- the design's
per-site alternation is what section G exercises; the single-pair back-to-back stress is a deliberate hang hazard and
is not run).

Recorded 2026-09-10 (stage A1, README "Phase 3e rows", design/phase3/allreduce_micro.md): composite 27.0 us (bfp8) /
34.4 (bf16) per call in lockstep; fused Ring 2 links 8x4 17.2 / 29.0; per site incl. reshards A 30.7 -> 21.1, M 27.1 ->
19.9; fused numerics at or below the composite's error (bf16: the exactly rounded fp32 sum); 0 replica mismatches in K, S
and G. L1 lessons the harness encodes: today's composite needs 512 KB (bf16) / 272 KB (bfp8) of free L1 on EVERY core for
its ReduceScatterMinimalDirect staging shard (measure it with no fused buffer resident), the fused op's CBs on its two
link-worker cores clash with L1 buffers when the lockstep high-water mark is low (prefer the 8x4 grid: 4x less L1 per
core than 8x1 at the same speed), and every program must run once before a trace capture.
"""

import itertools
import json
import os
import re
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.solar_open.tests.test_factory import parametrize_mesh_with_fabric

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(header, message=None):
        logger.info(f"SIGNPOST {header}")


PERF_OUT = os.getenv("SOLAR_OPEN_PERF_OUT", "")
PROFILE = os.getenv("SOLAR_OPEN_PERF_PROFILE", "") == "1"
REPS = int(os.getenv("SOLAR_OPEN_PERF_REPS", "6"))
SECTIONS = os.getenv("SOLAR_OPEN_CCL_MB_SECTIONS", "KSG").upper()
ONLY = os.getenv("SOLAR_OPEN_CCL_MB_ONLY", "")  # regex on K configuration names
TRACE_TIMING = os.getenv("SOLAR_OPEN_CCL_MB_TRACE_TIMING", "1") == "1"
TRACE_CALLS = int(os.getenv("SOLAR_OPEN_CCL_MB_TRACE_CALLS", "20"))
TRACE_REPLAYS = int(os.getenv("SOLAR_OPEN_CCL_MB_TRACE_REPLAYS", "5"))
GATE_ITERS = int(os.getenv("SOLAR_OPEN_CCL_MB_GATE_ITERS", "200"))
GATE_REPLAYS = int(os.getenv("SOLAR_OPEN_CCL_MB_GATE_REPLAYS", "40"))
GATE_CHUNK = 10  # iterations (2 calls each) issued back-to-back before the host reads the outputs
FUSED_PIN = os.getenv("SOLAR_OPEN_CCL_MB_FUSED", "")  # "<ring|linear>,<links>,<fp32 0|1>,<grid>"
B, H, TP = 32, 4096, 8
N_INPUTS = 4  # rotating inputs (changing data between back-to-back calls)
DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
TOPOS = {"ring": ttnn.Topology.Ring, "linear": ttnn.Topology.Linear}
GRIDS = {
    "8x4": (8, 4),
    "8x8": (8, 8),
    "8x1": (8, 1),
}  # (x, y): 8x4 = the decode norm grid, 8x1 = the o_proj output grid


def ws_config(grid, width):
    """Width-sharded L1 memory config of a [.., 32, width] tile on ``grid`` (x, y), row-major shards."""
    gx, gy = GRIDS[grid]
    shard_w = width // (gx * gy)
    assert shard_w % ttnn.TILE_SIZE == 0, (grid, width)
    return ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, shard_w),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def footprint(grid, in_dtype, out_dtype, links):
    """L1 bytes of the fused op at [32, 4096]: persistent buffer per output core, transient scratch CB per link
    worker, output shard per core (formulas of all_reduce_async_program_factory.cpp)."""
    gx, gy = GRIDS[grid]
    cores = gx * gy
    tiles_per_core = H // ttnn.TILE_SIZE // cores
    tile_in = 2048 if in_dtype == ttnn.bfloat16 else 1088
    tile_out = 2048 if out_dtype == ttnn.bfloat16 else 1088
    return {
        "buffer_per_core_B": tiles_per_core * TP * tile_in,
        "buffer_per_device_B": tiles_per_core * TP * tile_in * cores,
        "scratch_cb_per_link_worker_B": -(-cores // links) * tiles_per_core * tile_in,
        "out_shard_per_core_B": tiles_per_core * tile_out,
        "output_cores": cores,
    }


class Pool:
    """Persistent buffers + global semaphores of the fused op: two alternating pairs per (grid, input dtype)."""

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        grid = mesh_device.compute_with_storage_grid_size()
        self.all_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.semaphores = [ttnn.create_global_semaphore(mesh_device, self.all_cores, 0) for _ in range(2)]
        self.buffers = {}
        self.calls = 0

    def buffer(self, grid, in_dtype, k):
        key = (grid, in_dtype)
        if key not in self.buffers:
            cfg = ws_config(grid, H * TP)
            self.buffers[key] = [
                ttnn.from_torch(
                    torch.zeros(1, 1, B, H * TP),
                    device=self.mesh_device,
                    dtype=in_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=cfg,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                for _ in range(2)
            ]
        return self.buffers[key][k % 2]

    def fused(self, x, topo, links, fp32, grid, out_dtype=None, k=None):
        """One fused all-reduce of the width-sharded ``x`` (pair ``k``; consecutive calls alternate by default)."""
        if k is None:
            k = self.calls
        self.calls += 1
        in_dtype = x.dtype
        return ttnn.experimental.all_reduce_async(
            x,
            self.buffer(grid, in_dtype, k),
            cluster_axis=1,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=self.semaphores[k % 2],
            dtype=out_dtype if out_dtype is not None else in_dtype,
            memory_config=ws_config(grid, H),
            topology=TOPOS[topo],
            num_links=links,
            fp32_dest_acc=fp32,
        )

    def reset(self):
        for s in self.semaphores:
            ttnn.reset_global_semaphore_value(s, 0)

    def release(self, grid=None, in_dtype=None):
        """Free persistent buffers (all, or those of one grid / dtype). Today's composite needs 512 KB (bf16) /
        272 KB (bfp8) of free L1 on EVERY core for its ReduceScatterMinimalDirect staging shard, so the fused op's
        buffers must not be resident while it is measured (r1: OOM with 12 buffers alive)."""
        for key in list(self.buffers):
            if (grid is None or key[0] == grid) and (in_dtype is None or key[1] == in_dtype):
                for t in self.buffers.pop(key):
                    t.deallocate(True)

    def release_all_but(self, grid):
        for key in list(self.buffers):
            if key[0] != grid:
                for t in self.buffers.pop(key):
                    t.deallocate(True)

    def release_all_but(self, grid):
        """Free the persistent buffers of the other grids (L1 pressure on the cores every grid shares)."""
        for key in list(self.buffers):
            if key[0] != grid:
                for t in self.buffers.pop(key):
                    t.deallocate(True)


class Inputs:
    """N_INPUTS different [1, 1, 32, 4096] per-device inputs (device d holds slice d of a [1, 8, 32, 4096] tensor) in
    one dtype / memory config, with their fp32 references (sum of the 8 device-held, dtype-rounded slices).

    ``mapper``: "2d" = ``ShardTensor2dMesh(dims=(None, 1))`` (a [1, 8] distribution like the mesh, the topology
    production tensors carry), "1d" = ``ShardTensorToMesh(dim=1)``, "rep" = ``ReplicateTensorToMesh`` of slice 0
    (identical data on every device: timing-only control, reference = 8 x slice 0)."""

    def __init__(self, mesh_device, dtype, memory_config, seed, n=N_INPUTS, mapper="2d"):
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.memory_config = memory_config
        self.mapper = mapper
        g = torch.Generator().manual_seed(seed)
        self.host = [torch.randn(1, TP, B, H, generator=g) for _ in range(n)]
        self.dev = [self.upload(t) for t in self.host]
        self.ref = [self.reference(t) for t in self.host]

    def mesh_mapper(self):
        if self.mapper == "2d":
            return ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 1), mesh_shape=tuple(self.mesh_device.shape))
        if self.mapper == "1d":
            return ttnn.ShardTensorToMesh(self.mesh_device, dim=1)
        return ttnn.ReplicateTensorToMesh(self.mesh_device)

    def host_tensor(self, t):
        if self.mapper == "rep":
            t = t[:, 0:1]
        return ttnn.from_torch(t, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.mesh_mapper())

    def upload(self, t):
        return ttnn.to_device(self.host_tensor(t), self.mesh_device, memory_config=self.memory_config)

    def reference(self, t):
        """fp32 sum over the 8 devices of what each device holds (host-side dtype rounding == the upload's)."""
        slices = []
        for d in range(TP):
            sl = t[:, 0:1] if self.mapper == "rep" else t[:, d : d + 1]
            slices.append(ttnn.to_torch(ttnn.from_torch(sl, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)).float())
        return torch.stack(slices, 0).sum(0)

    def describe(self):
        x = self.dev[0]
        try:
            topo = str(x.tensor_topology())
        except Exception as exc:
            topo = f"n/a ({type(exc).__name__})"
        return (
            f"mapper={self.mapper} shape={tuple(x.shape)} dtype={x.dtype} "
            f"mem={x.memory_config().memory_layout} topology={topo[:160]}"
        )

    def update(self, i, t):
        """In-place host -> device write into persistent input ``i`` (the traced pattern)."""
        ttnn.copy_host_to_device_tensor(self.host_tensor(t), self.dev[i])
        self.host[i] = t
        self.ref[i] = self.reference(t)

    def deallocate(self):
        for x in self.dev:
            x.deallocate(True)


_REQUANT_CACHE = {}


def requant_floor(ref, dtype):
    """The output dtype's own rounding error on the exact answer: max / mean |round_dtype(ref) - ref|."""
    key = (id(ref), dtype)
    if key not in _REQUANT_CACHE:
        q = ttnn.to_torch(ttnn.from_torch(ref, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()
        d = (q - ref).abs()
        _REQUANT_CACHE[key] = (q, float(d.max()), float(d.mean()))
    return _REQUANT_CACHE[key]


def check_output(out, ref, label=""):
    """Replica identity over the 8 devices + numerics of device 0 vs the fp32 reference."""
    vals = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
    identical = all(torch.equal(v, vals[0]) for v in vals[1:])
    replica_max = max(float((v.float() - vals[0].float()).abs().max()) for v in vals[1:]) if len(vals) > 1 else 0.0
    v0 = vals[0].float().reshape(ref.shape)
    err = (v0 - ref).abs()
    q, floor_max, floor_mean = requant_floor(ref, out.dtype)
    err_q = (v0 - q).abs()
    _, pcc_s = comp_pcc(ref, v0, 0.99)
    pcc = float(str(pcc_s).split()[-1]) if not isinstance(pcc_s, float) else pcc_s
    res = {
        "identical": identical,
        "replica_max_diff": replica_max,
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "pcc": pcc,
        "requant_floor_max": floor_max,
        "requant_floor_mean": floor_mean,
        "max_err_vs_requant": float(err_q.max()),
        "frac_ne_requant": float((err_q > 0).float().mean()),
        "checksum": float(v0.double().sum()),
    }
    if label:
        logger.info(
            f"[mb {label}] identical={identical} replica_max={replica_max:.3g} max|err|={res['max_abs_err']:.4g} "
            f"mean|err|={res['mean_abs_err']:.4g} pcc={pcc:.6f} floor(max/mean)={floor_max:.4g}/{floor_mean:.4g} "
            f"max|err vs requant|={res['max_err_vs_requant']:.4g} frac_ne={res['frac_ne_requant']:.4f}"
        )
    return res


class Bench:
    def __init__(self, device, key):
        self.device = device
        self.key = key
        self.results = {}
        self.failures = {}

    def fail(self, name, exc):
        msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:400]}"
        logger.warning(f"[mb {name}] FAILED: {msg}")
        self.failures[name] = msg
        self.dump()

    def eager(self, name, fn, reps=REPS):
        """fn(i) -> output. One compile call, then ``reps`` signposted calls; returns the last output (kept)."""
        try:
            out = fn(0)
            ttnn.synchronize_device(self.device)
            out.deallocate(True)
        except Exception as exc:
            self.fail(name, exc)
            return None
        walls, last = [], None
        signpost(f"mb_{name}_start")
        for i in range(reps):
            t0 = time.perf_counter()
            out = fn(i)
            ttnn.synchronize_device(self.device)
            walls.append((time.perf_counter() - t0) * 1e6)
            if i < reps - 1:
                out.deallocate(True)
            else:
                last = out
        signpost(f"mb_{name}_stop")
        self.results.setdefault(name, {}).update(
            {"eager_wall_us_min": min(walls), "eager_wall_us_mean": sum(walls) / len(walls), "reps": reps}
        )
        logger.info(f"[mb {name}] eager wall min {min(walls):.1f} us mean {sum(walls) / len(walls):.1f} us")
        return last

    def trace(self, name, fn, calls=TRACE_CALLS, replays=TRACE_REPLAYS):
        """Capture ``calls`` back-to-back fn(i) in one trace, replay it; per-call wall us. Returns the kept output
        of the last call (valid after the replays) -- the caller deallocates it."""
        tid = None
        try:
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            last = None
            for i in range(calls):
                out = fn(i)
                if i < calls - 1:
                    out.deallocate(True)
                else:
                    last = out
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            ttnn.synchronize_device(self.device)
            ttnn.execute_trace(self.device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(self.device)
            t0 = time.perf_counter()
            for _ in range(replays):
                ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.device)
            per_call = (time.perf_counter() - t0) / replays / calls * 1e6
            ttnn.release_trace(self.device, tid)
        except Exception as exc:
            self.fail(name + "/trace", exc)
            self.abort_trace(tid)
            return None
        self.results.setdefault(name, {}).update({"trace_us_per_call": per_call, "trace_calls": calls})
        logger.info(f"[mb {name}] trace {calls} calls x {replays} replays: {per_call:.2f} us per call")
        return last

    def abort_trace(self, tid):
        """Best effort: close a capture an exception left open (the next begin_trace_capture would TT_FATAL)."""
        if tid is None:
            return
        for fn in (
            lambda: ttnn.end_trace_capture(self.device, tid, cq_id=0),
            lambda: ttnn.release_trace(self.device, tid),
        ):
            try:
                fn()
            except Exception as exc:
                logger.warning(f"[mb] trace cleanup: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
        try:
            ttnn.synchronize_device(self.device)
        except Exception:
            pass

    def note(self, name, key, value):
        self.results.setdefault(name, {})[key] = value
        logger.info(f"[mb {name}] {key} = {value}")

    def dump(self):
        if not PERF_OUT:
            return
        data = {}
        if os.path.isfile(PERF_OUT):
            try:
                data = json.loads(open(PERF_OUT).read())
            except Exception:
                data = {}
        data[self.key] = {"results": self.results, "failures": self.failures}
        with open(PERF_OUT, "w") as f:
            json.dump(data, f, indent=2)


def composite(x, links=1):
    return ttnn.all_reduce(x, num_links=links, topology=ttnn.Topology.Ring, cluster_axis=1)


def flush_profiler(device):
    """Drain the device profiler buffers (no-op without the profiler build / tracy)."""
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


def gather_reduce(x, links=1):
    """The galaxy 'stable all-reduce': all_gather onto dim 0 + fast_reduce_nc (interleaved, deterministic order)."""
    g = ttnn.all_gather(x, dim=0, num_links=links, topology=ttnn.Topology.Ring, cluster_axis=1)
    out = ttnn.experimental.fast_reduce_nc(g, dims=[0], memory_config=ttnn.L1_MEMORY_CONFIG)
    g.deallocate(True)
    return out


def parse_fused_pin(s):
    topo, links, fp32, grid = s.split(",")
    return {
        "topo": topo.strip().lower(),
        "links": int(links),
        "fp32": fp32.strip() in ("1", "true", "True"),
        "grid": grid,
    }


def measure(bench, pool, name, fn, inputs):
    """Eager reps (signposted) + numerics/replica check + determinism + trace timing of one configuration.
    ``fn(x, k)`` -> output for input tensor ``x`` and pair index ``k``. Never raises: failures go to bench.failures."""
    n = len(inputs.dev)
    out = bench.eager(name, lambda i: fn(inputs.dev[i % n], i))
    if out is None:
        return False
    try:
        res = check_output(out, inputs.ref[(REPS - 1) % n], name)
        out.deallocate(True)
        # determinism: two more calls on input 0 (alternating pairs)
        a = fn(inputs.dev[0], REPS)
        b = fn(inputs.dev[0], REPS + 1)
        ttnn.synchronize_device(bench.device)
        va = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(a)]
        vb = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(b)]
        res["deterministic"] = all(torch.equal(p, q) for p, q in zip(va, vb))
        a.deallocate(True)
        b.deallocate(True)
        bench.results[name].update(res)
    except Exception as exc:
        bench.fail(name + "/check", exc)
        return False
    if TRACE_TIMING:
        out = bench.trace(name, lambda i: fn(inputs.dev[i % n], i))
        if out is not None:
            try:
                tres = check_output(out, inputs.ref[(TRACE_CALLS - 1) % n], name + "/trace")
                out.deallocate(True)
                bench.results[name]["trace_identical"] = tres["identical"]
                bench.results[name]["trace_max_abs_err"] = tres["max_abs_err"]
            except Exception as exc:
                bench.fail(name + "/trace_check", exc)
    flush_profiler(bench.device)
    bench.dump()
    return True


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(1, 8)])
def test_decode_allreduce_microbench(mesh_device, device_params, reset_seeds):
    if not PROFILE:
        pytest.skip(
            "profiling helper: set SOLAR_OPEN_PERF_PROFILE=1 (and run under `python -m tracy -r -p -v -m pytest ...`)"
        )
    if tuple(mesh_device.shape) != (1, 8):
        pytest.skip("sized for the 1x8 mesh")
    bench = Bench(mesh_device, "decode_allreduce")
    pool = Pool(mesh_device)
    grid = mesh_device.compute_with_storage_grid_size()
    bench.note("meta", "compute_grid", [grid.x, grid.y])
    bench.note("meta", "sections", SECTIONS)
    bench.note("meta", "reps_trace_calls_replays", [REPS, TRACE_CALLS, TRACE_REPLAYS])

    # what the fabric makes of a Ring request on cluster axis 1 (a demotion to Linear would make the two arms equal)
    probe = ttnn.from_torch(
        torch.zeros(1, 1, B, H),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    try:
        bench.note("meta", "usable_topology_ring_axis1", str(ttnn.get_usable_topology(probe, ttnn.Topology.Ring, 1)))
        bench.note("meta", "usable_topology_default_axis1", str(ttnn.get_usable_topology(probe, None, 1)))
    except Exception as exc:
        bench.note("meta", "usable_topology_probe_failed", str(exc)[:200])
    probe.deallocate(True)

    fused_cfg = parse_fused_pin(FUSED_PIN) if FUSED_PIN else None

    # ------------------------------------------------------------------ K: kernel sweep
    if "K" in SECTIONS:
        k_names = []
        for dname, dtype in DTYPES.items():
            xin = Inputs(mesh_device, dtype, ws_config("8x4", H), seed=100 + len(k_names))
            bench.note("meta", f"input_{dname}_ws8x4", xin.describe())
            for ogrid in GRIDS:  # one grid's persistent buffer pair alive at a time (L1 hygiene)
                for topo, links, fp32 in itertools.product(TOPOS, (1, 2), (True, False)):
                    name = f"K_{dname}_{topo}_l{links}_fp32{int(fp32)}_{ogrid}"
                    if ONLY and not re.search(ONLY, name):
                        continue
                    bench.results.setdefault(name, {}).update(
                        {"arm": "fused", "dtype": dname, "topo": topo, "links": links, "fp32": fp32, "grid": ogrid}
                    )
                    bench.results[name]["footprint"] = footprint(ogrid, dtype, dtype, links)
                    ok = measure(
                        bench,
                        pool,
                        name,
                        lambda x, k, topo=topo, links=links, fp32=fp32, ogrid=ogrid: pool.fused(
                            x, topo, links, fp32, ogrid, k=k
                        ),
                        xin,
                    )
                    if ok:
                        k_names.append(name)
                pool.release(ogrid, dtype)
            xin.deallocate()
            # the attention site's input layout (8 o_proj cores) into the norm grid, design default config
            xin8 = Inputs(mesh_device, dtype, ws_config("8x1", H), seed=200 + len(k_names))
            name = f"K_{dname}_ring_l1_fp321_8x4_in8x1"
            if not ONLY or re.search(ONLY, name):
                bench.results.setdefault(name, {}).update(
                    {
                        "arm": "fused",
                        "dtype": dname,
                        "topo": "ring",
                        "links": 1,
                        "fp32": True,
                        "grid": "8x4",
                        "in": "8x1",
                    }
                )
                measure(bench, pool, name, lambda x, k: pool.fused(x, "ring", 1, True, "8x4", k=k), xin8)
            xin8.deallocate()
            pool.release()
            # composite baselines and the gather+reduce pair on the L1-interleaved input the sites use today
            # (no fused buffers resident: the composite's RS-direct staging needs 512 / 272 KB free on every core)
            xint = Inputs(mesh_device, dtype, ttnn.L1_MEMORY_CONFIG, seed=300 + len(k_names))
            bench.note("meta", f"input_{dname}_interleaved", xint.describe())
            for links in (1, 2):
                name = f"K_{dname}_composite_l{links}"
                if ONLY and not re.search(ONLY, name):
                    continue
                bench.results.setdefault(name, {}).update({"arm": "composite", "dtype": dname, "links": links})
                measure(bench, pool, name, lambda x, k, links=links: composite(x, links), xint)
            name = f"K_{dname}_agfr_l1"
            if not ONLY or re.search(ONLY, name):
                bench.results.setdefault(name, {}).update(
                    {"arm": "all_gather+fast_reduce_nc", "dtype": dname, "links": 1}
                )
                measure(bench, pool, name, lambda x, k: gather_reduce(x, 1), xint)
            xint.deallocate()
            # timing-only control: the composite on a replicated-topology input (identical data on all devices)
            xrep = Inputs(mesh_device, dtype, ttnn.L1_MEMORY_CONFIG, seed=350 + len(k_names), mapper="rep")
            name = f"K_{dname}_composite_rep_l1"
            if not ONLY or re.search(ONLY, name):
                bench.results.setdefault(name, {}).update(
                    {"arm": "composite (replicated input, timing)", "dtype": dname, "links": 1}
                )
                measure(bench, pool, name, lambda x, k: composite(x, 1), xrep)
            xrep.deallocate()
        # pick the fused configuration for S / G: min trace per-call wall among identical + deterministic + sane numerics
        if fused_cfg is None:
            cands = []
            for name in k_names:
                r = bench.results[name]
                if r.get("identical") and r.get("deterministic") and r.get("trace_identical", True):
                    if r.get("max_abs_err", 1e9) <= 4 * max(r.get("requant_floor_max", 0), 1e-6):
                        cands.append((r.get("trace_us_per_call", r["eager_wall_us_min"]), name))
            if cands:
                cands.sort()
                # near-ties (within 1 us) go to the smallest per-core persistent buffer (8x1 costs 4x the L1 of 8x4)
                lead = cands[0][0]
                near = [c for c in cands if c[0] <= lead + 1.0]
                near.sort(key=lambda c: (bench.results[c[1]]["footprint"]["buffer_per_core_B"], c[0]))
                cands = near + [c for c in cands if c not in near]
                best = bench.results[cands[0][1]]
                fused_cfg = {k: best[k] for k in ("topo", "links", "fp32", "grid")}
                bench.note("meta", "best_fused_K", cands[0][1])
    if fused_cfg is None:
        fused_cfg = {"topo": "ring", "links": 1, "fp32": True, "grid": "8x4"}  # design default
    bench.note("meta", "fused_cfg_for_S_G", fused_cfg)
    fz = fused_cfg
    pool.release_all_but(fz["grid"])

    def fused(x, k, out_dtype=None):
        return pool.fused(x, fz["topo"], fz["links"], fz["fp32"], fz["grid"], out_dtype=out_dtype, k=k)

    # ------------------------------------------------------------------ S: per-site chains incl. reshards
    if "S" in SECTIONS:
        xa = Inputs(mesh_device, ttnn.bfloat16, ws_config("8x1", H), seed=401)  # o_proj output layout
        xm8 = Inputs(mesh_device, ttnn.bfloat8_b, ttnn.L1_MEMORY_CONFIG, seed=402)  # MoE partial today
        xm16 = Inputs(mesh_device, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, seed=403)  # MoE partial, bf16 arm
        xa_rep = Inputs(mesh_device, ttnn.bfloat16, ws_config("8x1", H), seed=404, mapper="rep")  # timing controls
        xm8_rep = Inputs(mesh_device, ttnn.bfloat8_b, ttnn.L1_MEMORY_CONFIG, seed=405, mapper="rep")
        bench.note("meta", "input_S_siteA", xa.describe())

        def A0_today(x, k):  # s2i + typecast bfp8 + composite -> bfp8 L1 interleaved
            y = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            z = ttnn.typecast(y, ttnn.bfloat8_b)
            y.deallocate(True)
            out = composite(z)
            z.deallocate(True)
            return out

        def A0b_today_bf16(x, k):  # attention_bf16_output: s2i + composite bf16
            y = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            out = composite(y)
            y.deallocate(True)
            return out

        def A1_fused_bf16_s2i(x, k):  # fused bf16 -> bf16, back to interleaved for the untouched residual add
            y = fused(x, k)
            out = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)
            y.deallocate(True)
            return out

        def A2_fused_bf16_to_bfp8_s2i(x, k):  # fused bf16 in, bfp8 out (no typecast launch), + s2i
            y = fused(x, k, out_dtype=ttnn.bfloat8_b)
            out = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)
            y.deallocate(True)
            return out

        def A3_typecast_fused_bfp8_s2i(x, k):  # today's wire bytes: typecast bfp8 on the sharded tensor, fused, s2i
            z = ttnn.typecast(x, ttnn.bfloat8_b)
            y = fused(z, k)
            z.deallocate(True)
            out = ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG)
            y.deallocate(True)
            return out

        def A4_fused_bf16_sharded_out(x, k):  # lower bound: the residual add consumes the sharded output (2.5)
            return fused(x, k)

        def M0_today(x, k):
            return composite(x)

        def M1_i2s_fused_s2i(x, k):
            y = ttnn.to_memory_config(x, ws_config(fz["grid"], H))
            z = fused(y, k)
            y.deallocate(True)
            out = ttnn.to_memory_config(z, ttnn.L1_MEMORY_CONFIG)
            z.deallocate(True)
            return out

        def M3_i2s_fused_sharded_out(x, k):
            y = ttnn.to_memory_config(x, ws_config(fz["grid"], H))
            z = fused(y, k)
            y.deallocate(True)
            return z

        def Mx_agfr(x, k):
            return gather_reduce(x)

        # today's chains first: no fused buffer is resident yet (the composite's RS-direct staging shard needs
        # 512 / 272 KB of free L1 on every core), then the fused arms
        chains = [
            ("S_A0_today_s2i_typecast_composite", A0_today, xa),
            ("S_A0b_today_bf16_s2i_composite", A0b_today_bf16, xa),
            ("S_M0_today_composite_bfp8", M0_today, xm8),
            ("S_M0b_today_composite_bf16", M0_today, xm16),
            ("S_Mx_agfr_bfp8", Mx_agfr, xm8),
            ("S_A0rep_today_s2i_typecast_composite", A0_today, xa_rep),
            ("S_M0rep_today_composite_bfp8", M0_today, xm8_rep),
            ("S_A1_fused_bf16_s2i", A1_fused_bf16_s2i, xa),
            ("S_A2_fused_bf16_to_bfp8_s2i", A2_fused_bf16_to_bfp8_s2i, xa),
            ("S_A3_typecast_fused_bfp8_s2i", A3_typecast_fused_bfp8_s2i, xa),
            ("S_A4_fused_bf16_sharded_out", A4_fused_bf16_sharded_out, xa),
            ("S_M1_i2s_fused_bfp8_s2i", M1_i2s_fused_s2i, xm8),
            ("S_M2_i2s_fused_bf16_s2i", M1_i2s_fused_s2i, xm16),
            ("S_M3_i2s_fused_bfp8_sharded_out", M3_i2s_fused_sharded_out, xm8),
        ]
        for name, fn, xin in chains:
            bench.results.setdefault(name, {}).update(
                {"site": name.split("_")[1][0], "fused_cfg": fz, "input": xin.mapper}
            )
            measure(bench, pool, name, fn, xin)
            pool.release()  # L1 hygiene between chains (r2: bf16 8x1 pairs next to bfp8 ones clashed with the CBs)
        for xin in (xa, xm8, xm16, xa_rep, xm8_rep):
            xin.deallocate()

    # ------------------------------------------------------------------ G: replica gates (production pattern)
    gate_failures = []
    if "G" in SECTIONS:
        # the site-A input is kept L1 interleaved here (the in-place host->device update of the trace gate targets an
        # interleaved tensor) and resharded onto the 8 o_proj cores inside the chain, so the CCL sees the real layout
        xa = Inputs(mesh_device, ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, seed=501)
        xm = Inputs(mesh_device, ttnn.bfloat8_b, ttnn.L1_MEMORY_CONFIG, seed=502)

        def site_a_fused(x, k=0):  # site -> pair 0
            w = ttnn.to_memory_config(x, ws_config("8x1", H))
            out = fused(w, 0)
            w.deallocate(True)
            return out

        def site_m_fused(x, k=1):  # site -> pair 1
            y = ttnn.to_memory_config(x, ws_config(fz["grid"], H))
            z = fused(y, 1)
            y.deallocate(True)
            return z

        def site_a_composite(x, k=0):
            w = ttnn.to_memory_config(x, ws_config("8x1", H))
            y = ttnn.to_memory_config(w, ttnn.L1_MEMORY_CONFIG)
            w.deallocate(True)
            z = ttnn.typecast(y, ttnn.bfloat8_b)
            y.deallocate(True)
            out = composite(z)
            z.deallocate(True)
            return out

        def site_m_composite(x, k=1):
            return composite(x)

        def gate_eager(tag, fa, fm, iters):
            """``iters`` iterations of [site A call, site M call] back-to-back (host reads every GATE_CHUNK)."""
            bad = []
            worst = {"a": 0.0, "m": 0.0}
            n_checked = 0
            t0 = time.perf_counter()
            for start in range(0, iters, GATE_CHUNK):
                outs = []
                try:
                    for it in range(start, min(start + GATE_CHUNK, iters)):
                        outs.append((it, "a", fa(xa.dev[it % N_INPUTS]), xa.ref[it % N_INPUTS]))
                        outs.append((it, "m", fm(xm.dev[it % N_INPUTS]), xm.ref[it % N_INPUTS]))
                    ttnn.synchronize_device(mesh_device)
                except Exception as exc:
                    bench.fail(tag + "/eager", exc)
                    bad.append(f"{tag} iter {start}: exception {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
                    break
                for it, site, out, ref in outs:
                    r = check_output(out, ref)
                    n_checked += 1
                    worst[site] = max(worst[site], r["max_abs_err"])
                    floor = max(r["requant_floor_max"], 1e-6)
                    if not r["identical"] or r["max_abs_err"] > 4 * floor:
                        bad.append(
                            f"{tag} iter {it} site {site}: identical={r['identical']} replica_max={r['replica_max_diff']:.4g} "
                            f"max|err|={r['max_abs_err']:.4g} (floor {floor:.4g})"
                        )
                    out.deallocate(True)
                flush_profiler(mesh_device)
                if bad and len(bad) > 5:
                    break
            wall = time.perf_counter() - t0
            bench.note(
                tag,
                "eager_gate",
                {"iters": iters, "checked": n_checked, "bad": len(bad), "wall_s": wall, "worst_max_abs_err": worst},
            )
            for b in bad[:10]:
                logger.error(f"[gate] {b}")
            return bad

        def gate_trace(tag, fa, fm, replays, iters_per_trace=10):
            """Trace of ``iters_per_trace`` [A, M] iterations on inputs 0 / 1 alternating; every replay first writes
            fresh data into the four persistent inputs in place (copy_host_to_device_tensor)."""
            bad = []
            tid = None
            try:
                # warm-up: every program must have run once (kernel binaries uploaded) before a capture -- a cold
                # capture fails with "Writes are not supported during trace capture" (r1, G_composite)
                for it in range(2):
                    fa(xa.dev[it % 2]).deallocate(True)
                    fm(xm.dev[it % 2]).deallocate(True)
                ttnn.synchronize_device(mesh_device)
            except Exception as exc:
                bench.fail(tag + "/warmup", exc)
                return [f"{tag}: warm-up failed: {str(exc).splitlines()[0][:200]}"]
            try:
                tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                last_a = last_m = None
                for it in range(iters_per_trace):
                    oa = fa(xa.dev[it % 2])
                    om = fm(xm.dev[it % 2])
                    if it < iters_per_trace - 1:
                        oa.deallocate(True)
                        om.deallocate(True)
                    else:
                        last_a, last_m = oa, om
                ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
                ttnn.synchronize_device(mesh_device)
            except Exception as exc:
                bench.fail(tag + "/trace_capture", exc)
                bench.abort_trace(tid)
                return [f"{tag}: trace capture failed: {str(exc).splitlines()[0][:200]}"]
            idx = (iters_per_trace - 1) % 2  # the input the kept outputs were computed from
            g = torch.Generator().manual_seed(7000 + len(tag))
            walls = []
            update_mode = "in-place copy_host_to_device_tensor"
            for r in range(replays):
                for i in range(2):
                    try:
                        xa.update(i, torch.randn(1, TP, B, H, generator=g))
                        xm.update(i, torch.randn(1, TP, B, H, generator=g))
                    except Exception as exc:
                        bench.fail(tag + "/in_place_update", exc)
                        return [f"{tag}: in-place input update failed: {exc}"]
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                walls.append((time.perf_counter() - t0) * 1e6 / iters_per_trace)
                flush_profiler(mesh_device)
                for site, out, ref in (("a", last_a, xa.ref[idx]), ("m", last_m, xm.ref[idx])):
                    res = check_output(out, ref)
                    floor = max(res["requant_floor_max"], 1e-6)
                    if not res["identical"] or res["max_abs_err"] > 4 * floor:
                        bad.append(
                            f"{tag} replay {r} site {site}: identical={res['identical']} "
                            f"replica_max={res['replica_max_diff']:.4g} max|err|={res['max_abs_err']:.4g} (floor {floor:.4g})"
                        )
            ttnn.release_trace(mesh_device, tid)
            last_a.deallocate(True)
            last_m.deallocate(True)
            bench.note(
                tag,
                "trace_gate",
                {
                    "replays": replays,
                    "iters_per_trace": iters_per_trace,
                    "bad": len(bad),
                    "us_per_iter_min": min(walls),
                    "us_per_iter_mean": sum(walls) / len(walls),
                    "update": update_mode,
                },
            )
            for b in bad[:10]:
                logger.error(f"[gate] {b}")
            return bad

        bench.note("meta", "input_G_siteA", xa.describe())
        gate_failures += gate_eager("G_fused", site_a_fused, site_m_fused, GATE_ITERS)
        bench.dump()
        gate_failures += gate_trace("G_fused", site_a_fused, site_m_fused, GATE_REPLAYS)
        bench.dump()
        # the control last (the harness must pass today's collective; a failure here is a harness / input finding,
        # not a fused-op one, so it is recorded but excluded from the assertion below); fused buffers released first
        pool.release()
        control_failures = gate_trace("G_composite", site_a_composite, site_m_composite, min(GATE_REPLAYS, 10))
        bench.note("G_composite", "control_failures", len(control_failures))
        bench.dump()
        xa.deallocate()
        xm.deallocate()

    bench.dump()
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
    except Exception:
        pass
    assert not gate_failures, f"{len(gate_failures)} gate failures, first: {gate_failures[0]}"
