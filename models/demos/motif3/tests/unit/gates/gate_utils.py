# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Motif-3 device op gates G1-G8 (design doc §1.2, §1.6, §4.2).

Import-safe: only ``torch``/``ttnn``/stdlib at module import time; nothing here touches a device
until a test calls it with the ``mesh_device`` fixture. No other ``models/demos/**`` package is
imported (design §2.1 import rule).

Conventions used by every gate:
  * Gates run on the logical (4, 8) mesh (design §3.1) with the per-chip Motif shapes of §3.2/§3.3.
    Single-chip ops are fed *replicated* inputs so all 32 chips execute the same program; the
    readback compares device 0 against the torch fp32 golden and checks that all 32 replicas are
    bitwise identical.
  * Latency is wall clock around ``ttnn.synchronize_device`` over N iterations: ``eager`` =
    N back-to-back dispatches; ``traced`` = one trace holding N copies of the op, replayed and
    divided by N (amortises the replay launch; this is the decode-relevant number).
  * Every measurement is appended as one JSON line to ``results/<gate>.jsonl`` (append-only, the
    last record per ``case`` wins in the report).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Iterable

import torch

import ttnn

GATES_DIR = Path(__file__).resolve().parent
RESULTS_DIR = GATES_DIR / "results"

MESH_SHAPE = (4, 8)  # logical mesh: rows = 4 DP groups (cluster_axis 0), cols = TP8 (cluster_axis 1)
TRACE_REGION_SIZE = 96 * 1024 * 1024  # generous: gates capture up to ~64 ops per trace


# --------------------------------------------------------------------------------------------
# device params / fixtures
# --------------------------------------------------------------------------------------------
def device_params(fabric=None, trace_region_size: int = TRACE_REGION_SIZE, **extra) -> dict:
    """device_params for the root-conftest ``mesh_device`` fixture (see tt-metal/conftest.py)."""
    if fabric is None:
        fabric = ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    params = {
        "fabric_config": fabric,
        "trace_region_size": trace_region_size,
        "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
    }
    params.update(extra)
    return params


def mesh_params(fabric=None, **extra):
    """One ``(mesh_device, device_params)`` parametrization tuple for the (4, 8) mesh."""
    return (MESH_SHAPE, device_params(fabric, **extra))


def fabric_name(fabric) -> str:
    return str(fabric).split(".")[-1]


def grid_size(mesh_device):
    return mesh_device.compute_with_storage_grid_size()


def hifi4(fp32_acc: bool = True, approx: bool = False, packer_l1_acc: bool = False):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


def compute_cfg(fidelity: str = "HiFi4", fp32_acc: bool = True, approx: bool = False, packer_l1_acc: bool = False):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


# --------------------------------------------------------------------------------------------
# host <-> device
# --------------------------------------------------------------------------------------------
def to_mesh(t: torch.Tensor, mesh_device, dtype, layout=ttnn.TILE_LAYOUT, memory_config=None, mapper=None, **kw):
    """Replicate a torch tensor onto every chip of the mesh (default) or shard it with ``mapper``."""
    if mapper is None:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
        **kw,
    )


def host_roundtrip(t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT) -> torch.Tensor:
    """Quantise on the host exactly as ``from_torch`` will (e.g. bfloat8_b block exponents), no device."""
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=layout)).to(torch.float32)


def dev_tensors(t) -> list:
    return ttnn.get_device_tensors(t)


def read_dev(t, idx: int = 0) -> torch.Tensor:
    """Read the shard/replica held by device ``idx`` (row-major linear device index)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[idx]).to(torch.float32)


def read_all(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(x).to(torch.float32) for x in ttnn.get_device_tensors(t)]


def replicas_identical(t, idxs: Iterable[int] | None = None) -> tuple[bool, int]:
    """Bitwise identity of all device copies of a replicated result. Returns (ok, n_mismatching_devices)."""
    shards = ttnn.get_device_tensors(t)
    ref = ttnn.to_torch(shards[0]).contiguous()
    bad = 0
    for i, s in enumerate(shards):
        if idxs is not None and i not in idxs:
            continue
        x = ttnn.to_torch(s).contiguous()
        # compare raw bits so NaN payloads / signed zeros count as identical only if bit-identical
        if x.shape != ref.shape or x.dtype != ref.dtype or not torch.equal(_bits(x), _bits(ref)):
            bad += 1
    return bad == 0, bad


def _bits(x: torch.Tensor) -> torch.Tensor:
    if x.dtype in (torch.bfloat16, torch.float16):
        return x.view(torch.int16)
    if x.dtype == torch.float32:
        return x.view(torch.int32)
    return x


# --------------------------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------------------------
def pcc(golden: torch.Tensor, got: torch.Tensor) -> float:
    g = golden.detach().double().flatten()
    c = got.detach().double().flatten()
    if g.numel() != c.numel():
        raise ValueError(f"size mismatch {g.numel()} vs {c.numel()}")
    finite = torch.isfinite(g) & torch.isfinite(c)
    if not bool(finite.all()):
        return float("nan")
    g = g - g.mean()
    c = c - c.mean()
    den = float(g.norm() * c.norm())
    if den == 0.0:
        return 1.0 if torch.allclose(golden.double(), got.double()) else 0.0
    return float((g @ c) / den)


def compare(golden: torch.Tensor, got: torch.Tensor) -> dict:
    g = golden.detach().double()
    c = got.detach().double()
    nonfinite = int((~torch.isfinite(c)).sum())
    diff = (g - c).abs()
    diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, float("inf")))
    ref_scale = float(g.abs().max()) if g.numel() else 0.0
    return {
        "pcc": pcc(golden, got),
        "max_abs": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean()) if diff.numel() else 0.0,
        "ref_absmax": ref_scale,
        "rel_fro": float((g - c).norm() / max(float(g.norm()), 1e-30)) if nonfinite == 0 else float("inf"),
        "nonfinite": nonfinite,
    }


def fmt(stats: dict) -> str:
    return (
        f"pcc={stats['pcc']:.6f} max_abs={stats['max_abs']:.3e} mean_abs={stats['mean_abs']:.3e} "
        f"rel_fro={stats['rel_fro']:.3e} nonfinite={stats['nonfinite']}"
    )


# --------------------------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------------------------
def free(o):
    """Deallocate a tensor or a (nested) tuple/list of tensors (outputs of a timed op)."""
    if isinstance(o, (list, tuple)):
        for x in o:
            free(x)
    elif isinstance(o, ttnn.Tensor):
        try:
            ttnn.deallocate(o)
        except Exception:
            pass


def time_eager(mesh_device, fn: Callable, iters: int = 20, warmup: int = 2) -> float:
    """Mean wall-clock µs per call over ``iters`` back-to-back dispatches (after warmup).

    Outputs are freed right after dispatch (in-order CQ execution makes the reuse safe), so L1-resident
    outputs never accumulate and collide with the op's own static circular buffers.
    """
    for _ in range(warmup):
        free(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        free(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


def _trace_total_us(mesh_device, fn: Callable, n_ops: int, reps: int, keep_outputs: bool = False) -> list[float]:
    """Capture a trace of ``n_ops`` calls, replay ``reps`` times; returns sorted total µs per replay.

    With ``keep_outputs=False`` every output is freed right after its call inside the capture, so all n_ops
    reuse the same output buffer (legal: nothing is allocated between capture and replay/release).
    """
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = []
    try:
        for _ in range(n_ops):
            o = fn()
            if keep_outputs:
                outs.append(o)
            else:
                free(o)
    except BaseException:
        # Never leave the mesh in capture mode: a dangling capture turns every later write into a TT_FATAL and
        # hung close_mesh_device in teardown (observed 2026-10-01, needed a tt-smi reset).
        try:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        except Exception:
            pass
        try:
            ttnn.release_trace(mesh_device, tid)
        except Exception:
            pass
        raise
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            times.append((time.perf_counter() - t0) * 1e6)
    finally:
        ttnn.release_trace(mesh_device, tid)
    free(outs)
    return sorted(times)


LAST_TRACE_SAMPLES: dict = {}


def time_traced(mesh_device, fn: Callable, ops_per_trace: int = 64, reps: int = 9, compile_first: bool = True):
    """Traced per-op latency. Returns (per_op_us, raw_us).

    Two traces are captured with n1 = n2 // 2 and n2 = ops_per_trace back-to-back calls and each is replayed ``reps``
    times (replay + ``synchronize_device``). per_op = (min t(n2) - min t(n1)) / (n2 - n1).
    Calibration (G0/G8, 2026-10-01): ``synchronize_device`` alone costs ~140 µs on the 32-chip mesh and device work
    below that is hidden behind it, so t(n) is flat until n·t_op exceeds ~150 µs (ops < ~2 µs stay hidden even at
    n = 64: then per_op reads ~0 and raw is the bound). Replays are bimodal: besides the fast path, some take a
    ~1/2/3 ms-quantised slow path (host-side sleep while polling), so the *minimum* over reps is used, not the median.
    raw = min t(n2) / n2 is an upper bound. All samples are kept in ``LAST_TRACE_SAMPLES``.
    """
    if compile_first:
        free(fn())
        ttnn.synchronize_device(mesh_device)
    n2 = max(4, ops_per_trace)
    n1 = max(2, n2 // 2)
    t1 = _trace_total_us(mesh_device, fn, n1, reps)
    t2 = _trace_total_us(mesh_device, fn, n2, reps)
    slope = (t2[0] - t1[0]) / (n2 - n1)
    raw = t2[0] / n2
    LAST_TRACE_SAMPLES.clear()
    LAST_TRACE_SAMPLES.update({"n1": n1, "n2": n2, "t1_us": [round(t, 1) for t in t1], "t2_us": [round(t, 1) for t in t2], "slope": slope, "raw": raw})
    return max(slope, 0.0), raw


def time_trace_replay_floor(mesh_device, reps: int = 10) -> float:
    """Total µs for replaying a trivially small trace (one tiny op) + synchronize on the whole mesh."""
    x = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    _o = ttnn.add(x, 1.0)
    ttnn.synchronize_device(mesh_device)
    t = _trace_total_us(mesh_device, lambda: ttnn.add(x, 1.0), 1, reps)
    return t[len(t) // 2]


# --------------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------------
class Recorder:
    """Append-only JSONL results log for one gate."""

    def __init__(self, gate: str):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.gate = gate
        self.path = RESULTS_DIR / f"{gate}.jsonl"
        self.rows: list[dict] = []

    def add(self, case: str, **fields):
        if "traced_us" in fields and "samples" not in fields and LAST_TRACE_SAMPLES:
            fields["samples"] = dict(LAST_TRACE_SAMPLES)
            LAST_TRACE_SAMPLES.clear()
        row = {"gate": self.gate, "case": case, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}
        row = _jsonable(row)
        self.rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"[{self.gate}] {case}: " + ", ".join(f"{k}={v}" for k, v in fields.items() if k != "notes"))
        return row


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return str(x)


def seed_of(*parts) -> int:
    """Deterministic seed from arbitrary parts (Python's str hash is salted per process)."""
    import zlib

    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2**31)


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "False")
