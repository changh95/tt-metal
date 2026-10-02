# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""L1_SMALL regression tests (attention wave-B1 P0; README "Running tests", CONVENTIONS §8). Device only, through the
lock wrapper::

    scripts/devrun.sh -t 1800 -n infra_l1_small -- \
        python -m pytest models/demos/motif3/tests/unit/test_infra_l1_small.py -s -p no:cacheprovider

The hazard: a CCL global semaphore allocated in **main** L1 lands at whatever address is free when its program is first
built and stays there for the life of the program cache; the first later program whose static circular buffers reach
that address throws "Statically allocated circular buffers ... clash with L1 buffers" (a global-layer prefill at
S >= 1024 after any decode step; the bf16-KV FlashMLA decode). The fix: every mesh is opened with ``l1_small_size =
32768`` (``model_config.device_params()``, the vLLM ``"tt"`` config, ``open_motif_mesh``) and ``MotifCCL`` keeps every
CCL's semaphores in L1_SMALL (``tt/ccl.py`` docstring table).

* ``test_ccl_semaphores_stay_in_l1_small`` (4x8, shared ``device_params()``): every draft-1 CCL payload (decode and
  prefill shapes, all MotifCCL paths: direct / ring / line reduce-scatter, native / composite all-gather, the
  AG + local-sum all-reduce, ``ar_exact``, ``ag_dp_rows``, ``partition``) leaves main L1 unchanged on its program-cache
  miss and its hit (outputs freed), the L1_SMALL footprint stays far below the region, and every result equals the
  plain ttnn op bitwise (``ar_exact``: the PolyNorm ``_ar_ag_sum`` bitwise, and the fp64 sum to fp32 rounding) and the
  torch golden, with replicas bitwise identical along the reduced axis. Then the traced (slope method) and eager cost
  of ``MotifCCL.all_reduce`` against ``ttnn.all_reduce`` (the same device programs: traced must not be slower).
* ``test_attention_serving_order_l1_small`` (4x8, shared ``device_params()``): ONE mesh session in serving order for
  a bfp8 and then a bf16 latent cache (MotifAttention, global layer 0 + SWA layer 1, random weights): global prefill at
  S = 1024 and 4096 -> 2 decode steps -> the same global prefills again (output bitwise equal to the first time) -> 2
  decode steps. Every prefill / decode is checked against the fp32 reference (PCC >= 0.999 per user / lane), and at
  the end main L1 holds exactly what it held before the first CCL (nothing persistent left in main L1).

Every test logs the committed fabric topology first (``[motif3.fabric] committed: ...``) and the L1_SMALL size.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Dict, List, Optional

import pytest
import torch

import ttnn
from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, l1_usage, log_fabric, replicas_identical
from models.demos.motif3.tt.model_config import (
    DEFAULT_HF_META_DIR,
    DEFAULT_L1_SMALL_SIZE,
    MotifTTConfig,
    device_params,
    kv_cache_dtype_from_name,
)

HF_META = str(DEFAULT_HF_META_DIR)
MESH = [pytest.param((4, 8), device_params(), id="4x8-shared-device-params")]


def log(msg: str) -> None:
    print(f"[l1small] {msg}", flush=True)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double().flatten()
    b = b.detach().double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = float(a.norm() * b.norm())
    return 1.0 if den == 0.0 else float((a @ b) / den)


def _free(o):
    if isinstance(o, (list, tuple)):
        for t in o:
            _free(t)
    elif isinstance(o, dict):
        for t in o.values():
            _free(t)
    elif isinstance(o, ttnn.Tensor):
        ttnn.deallocate(o)


def _mapper_2d(mesh_device):
    R, C = tuple(mesh_device.shape)
    return ttnn.create_mesh_mapper(
        mesh_device, ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], ttnn.MeshShape(R, C))
    )


def _per_chip(mesh_device, local, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, seed=0, scale=1.0):
    """Distinct data per chip: host ``[R, C, *local]`` -> chip (r, c) holds ``[1, 1, *local]`` (local = (h, w)) or
    ``[1, *local]`` for a rank-3 local shape ``(a, h, w)`` (dims 0 / 1 of the host tensor are the mesh)."""
    R, C = tuple(mesh_device.shape)
    g = torch.Generator().manual_seed(seed)
    host = scale * torch.randn(R, C, *local, generator=g)
    if dtype == ttnn.bfloat16:
        host = host.to(torch.bfloat16).float()
    t = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper_2d(mesh_device),
    )
    return host, t


def _golden(op: str, host: torch.Tensor, cfg: MotifTTConfig, dim: int) -> torch.Tensor:
    """Expected ``[R, C, *local]`` for per-chip inputs ``host [R, C, *local]`` (``dim`` indexes the 4D chip tensor)."""
    a = cfg.axes
    R, C = a.mesh_shape
    ld = dim - 2  # the host tensor drops the two leading chip dims of [1, 1, h, w]
    out = {}
    for r in range(R):
        for c in range(C):
            dp, tp = a.roles(r, c)
            axis = "dp" if op.endswith("dp") else "tp"
            peers = [host[a.coord(d, tp)] for d in range(a.dp_size)] if axis == "dp" else [
                host[a.coord(dp, t)] for t in range(a.tp_size)
            ]
            idx = dp if axis == "dp" else tp
            if op.startswith("ar") or op.startswith("exact"):
                v = torch.stack([p.double() for p in peers]).sum(0)
            elif op.startswith("ag"):
                v = torch.cat(peers, dim=ld)
            elif op.startswith("rs"):
                v = torch.stack([p.double() for p in peers]).sum(0).chunk(len(peers), dim=ld)[idx]
            elif op.startswith("part"):
                v = host[r, c].chunk(len(peers), dim=ld)[idx]
            else:
                raise ValueError(op)
            out[(r, c)] = v
    return torch.stack([torch.stack([out[(r, c)] for c in range(C)]) for r in range(R)])


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_ccl_semaphores_stay_in_l1_small(mesh_device, device_params):
    from models.demos.motif3.tt import polynorm as PN

    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device)
    rep = log_fabric(mesh_device, "l1small ccl")
    log(f"L1_SMALL {rep['l1_small']} B per core; cfg l1_small {cfg.l1_small_size} mesh {cfg.mesh_l1_small_size}")
    assert rep["l1_small"] is not None and rep["l1_small"] >= DEFAULT_L1_SMALL_SIZE, rep
    assert cfg.mesh_l1_small_size == rep["l1_small"]
    ccl = MotifCCL(mesh_device, cfg)
    assert ccl.l1_small_semaphores, "MotifCCL must route semaphores to L1_SMALL on a mesh with the region"
    _, probe_t = _per_chip(mesh_device, (32, 32))
    tp_ring = ccl.is_ring_axis(probe_t, cfg.axes.tp_axis)
    ttnn.deallocate(probe_t)
    ckc = cfg.compute_config("polynorm")
    BF, F32 = ttnn.bfloat16, ttnn.float32
    TILE, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
    # name, MotifCCL fn, plain-ttnn reference fn (or None), golden op, local shape, dtype, layout, gather / scatter dim
    cases = [
        ("AR(tp) wo / MoE out 8x4096 bf16 (direct RS)", lambda x: ccl.ar_tp(x),
         lambda x: ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ar_tp", (8, 4096), BF, TILE, 3),
        ("AR(tp) dense MLP 32x4096 bf16 (direct RS)", lambda x: ccl.ar_tp(x),
         lambda x: ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ar_tp", (32, 4096), BF, TILE, 3),
        ("AR(dp) MoE combine 32x4096 bf16 (DP line: ring factory)", lambda x: ccl.ar_dp(x),
         lambda x: ttnn.all_reduce(x, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ar_dp", (32, 4096), BF, TILE, 3),
        ("AR(dp) MoE combine 32x4096 fp32", lambda x: ccl.ar_dp(x),
         lambda x: ttnn.all_reduce(x, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ar_dp", (32, 4096), F32, TILE, 3),
        ("AR(tp) moments 8x32 fp32 (AG + local sum)", lambda x: ccl.ar_tp(x),
         lambda x: ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ar_tp", (8, 32), F32, TILE, 3),
        ("AR(tp) prefill 128x4096 bf16", lambda x: ccl.ar_tp(x),
         lambda x: ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ar_tp", (128, 4096), BF, TILE, 3),
        ("AR(tp) prefill 1024x4096 bf16", lambda x: ccl.ar_tp(x),
         lambda x: ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ar_tp", (1024, 4096), BF, TILE, 3),
        ("AR(tp) prefill 4096x4096 bf16", lambda x: ccl.ar_tp(x),
         lambda x: ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ar_tp", (4096, 4096), BF, TILE, 3),
        ("RS(dp) MoE prefill 128x4096 bf16", lambda x: ccl.rs_dp(x, 2),
         lambda x: ttnn.reduce_scatter(x, 2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG), "rs_dp", (128, 4096), BF, TILE, 2),
        ("RS(dp) MoE prefill 1024x4096 bf16", lambda x: ccl.rs_dp(x, 2),
         lambda x: ttnn.reduce_scatter(x, 2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG), "rs_dp", (1024, 4096), BF, TILE, 2),
        ("RS(tp) 8x4096 bf16 dim 3 (direct)", lambda x: ccl.reduce_scatter(x, 3, "tp"),
         lambda x: ttnn.reduce_scatter(x, 3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG), "rs_tp", (8, 4096), BF, TILE, 3),
        ("AG(dp) MoE prefill 32x4096 bf16 (native)", lambda x: ccl.ag_dp(x, 2),
         lambda x: ttnn.all_gather(x, 2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ag_dp", (32, 4096), BF, TILE, 2),
        ("AG(dp) 8x4096 TILE (composite -> L1_SMALL composite)", lambda x: ccl.ag_dp(x, 2),
         lambda x: ttnn.all_gather(x, 2, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ag_dp", (8, 4096), BF, TILE, 2),
        ("ag_dp_rows MoE token gather 8x4096", lambda x: ccl.ag_dp_rows(x), None, "ag_dp", (8, 4096), BF, TILE, 2),
        ("AG(tp) argmax rows 32x32 bf16 dim 2", lambda x: ccl.all_gather(x, 2, "tp"),
         lambda x: ttnn.all_gather(x, 2, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ag_tp", (32, 32), BF, TILE, 2),
        ("AG(tp) 1x32 fp32 dim 2 (composite)", lambda x: ccl.all_gather(x, 2, "tp"),
         lambda x: ttnn.all_gather(x, 2, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ag_tp", (1, 32), F32, TILE, 2),
        ("AG(tp) embedding hidden shard 8x512 dim 3", lambda x: ccl.ag_tp(x, 3),
         lambda x: ttnn.all_gather(x, 3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG), "ag_tp", (8, 512), BF, TILE, 3),
        ("partition(dp) 32x4096 TILE (RM round trip)", lambda x: ccl.partition(x, 2, "dp"), None, "part_dp", (32, 4096), BF, TILE, 2),
        ("partition(dp) 128x4096 TILE", lambda x: ccl.partition(x, 2, "dp"),
         lambda x: ttnn.mesh_partition(x, 2, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG), "part_dp", (128, 4096), BF, TILE, 2),
        ("ar_exact(tp) moment sums 8x1 fp32", lambda x: ccl.ar_exact(x, "tp"),
         lambda x: PN._ar_ag_sum(ttnn.clone(x), ccl, mc=ttnn.DRAM_MEMORY_CONFIG, ckc=ckc), "exact_tp", (8, 1), F32, TILE, 3),
        ("ar_exact(tp) moment sums 32x1 fp32", lambda x: ccl.ar_exact(x, "tp"),
         lambda x: PN._ar_ag_sum(ttnn.clone(x), ccl, mc=ttnn.DRAM_MEMORY_CONFIG, ckc=ckc), "exact_tp", (32, 1), F32, TILE, 3),
        ("ar_exact(tp) 8x32 fp32 (dim -3 gather)", lambda x: ccl.ar_exact(x, "tp"), None, "exact_tp", (8, 32), F32, TILE, 3),
        ("ar_exact(dp) 64x1 fp32", lambda x: ccl.ar_exact(x, "dp"), None, "exact_dp", (64, 1), F32, TILE, 3),
    ]  # fmt: skip
    failures, report, outs = [], [], {}
    usage0 = l1_usage(mesh_device)
    # pass 1: MotifCCL only (no plain ttnn CCL yet, so a main-L1 growth can only come from MotifCCL)
    for i, (name, fn, ref_fn, op, local, dtype, layout, dim) in enumerate(cases):
        scale = 1.0 if dtype != F32 else 3.0
        host, x = _per_chip(mesh_device, local, dtype, layout, seed=i, scale=scale)
        try:
            ttnn.synchronize_device(mesh_device)
            u0 = l1_usage(mesh_device)
            out = fn(x)
            ttnn.synchronize_device(mesh_device)
            got = device_tensors_to_torch(out, mesh_device).float()
            _free(out)
            u1 = l1_usage(mesh_device)
            out = fn(x)  # program-cache hit
            ttnn.synchronize_device(mesh_device)
            _free(out)
            u2 = l1_usage(mesh_device)
            d_main, d_main_hit = u1["l1"] - u0["l1"], u2["l1"] - u1["l1"]
            d_small = u1["l1_small"] - u0["l1_small"]
            want = _golden(op, host, cfg, dim)
            g = got.reshape(want.shape)
            exact = op.startswith(("ag", "part"))
            if exact:
                ok_val = torch.equal(g, want.float())
                prec = "exact" if ok_val else f"max diff {(g - want.float()).abs().max().item():.3e}"
            else:
                err = (g.double() - want).abs().max().item()
                ref_max = want.abs().max().item()
                p = _pcc(g, want)
                if op.startswith("exact"):  # fp32 sum of n values, rounded in fp32: relative error ~1e-7
                    ok_val = err <= 4e-6 * max(ref_max, 1.0)
                else:
                    ok_val = p > (0.9999999 if dtype == F32 else 0.9999) and err <= (2e-3 if dtype == F32 else 3e-2) * max(ref_max, 1.0)
                prec = f"pcc {p:.8f} max_err {err:.2e} (|ref| {ref_max:.2f})"
            axis = "dp" if op.endswith("dp") else "tp"
            same = True
            if op.startswith(("ar", "exact")):
                same = bool(torch.equal(got, got.select(cfg.axes.dp_axis if axis == "dp" else cfg.axes.tp_axis, 0)
                                        .unsqueeze(cfg.axes.dp_axis if axis == "dp" else cfg.axes.tp_axis).expand_as(got)))
            ok = d_main == 0 and d_main_hit == 0 and ok_val and same
            line = (f"{'PASS' if ok else 'FAIL'} {name}: main L1 +{d_main} B (hit +{d_main_hit}), L1_SMALL +{d_small} B; "
                    f"{prec}; replicas {same}")  # fmt: skip
            report.append(line)
            if not ok:
                failures.append(line)
            outs[name] = (got, x)
        except Exception as e:  # keep going: one run reports every payload
            failures.append(f"FAIL {name}: {type(e).__name__}: {str(e)[:400]}")
            report.append(failures[-1])
            _free(x)
    usage1 = l1_usage(mesh_device)
    report.append(
        f"INFO pass 1 total: main L1 {usage0['l1']} -> {usage1['l1']} B, L1_SMALL {usage0['l1_small']} -> "
        f"{usage1['l1_small']} B of {rep['l1_small']} B per core (tp ring {tp_ring})"
    )
    if usage1["l1"] != usage0["l1"]:
        failures.append(f"FAIL main L1 grew by {usage1['l1'] - usage0['l1']} B over the MotifCCL pass")
    if usage1["l1_small"] - usage0["l1_small"] > rep["l1_small"] // 4:
        failures.append(f"FAIL the CCL semaphores use {usage1['l1_small'] - usage0['l1_small']} B of L1_SMALL")
    # pass 2: bitwise identity with the plain ttnn op (which may itself put semaphores in main L1: run it last)
    for name, fn, ref_fn, op, local, dtype, layout, dim in cases:
        if name not in outs:
            continue
        got, x = outs[name]
        if ref_fn is not None:
            try:
                r = ref_fn(x)
                ref = device_tensors_to_torch(r, mesh_device).float()
                _free(r)
                bit = torch.equal(got, ref)
                report.append(f"{'PASS' if bit else 'FAIL'} {name}: MotifCCL == plain ttnn bitwise {bit}")
                if not bit:
                    failures.append(f"FAIL {name}: differs from the plain ttnn op (max {(got - ref).abs().max().item():.3e})")
            except Exception as e:
                failures.append(f"FAIL {name} (plain ttnn reference): {type(e).__name__}: {str(e)[:300]}")
        _free(x)
    # pass 3: cost of the L1_SMALL routing (same device programs; MotifCCL's all_reduce is a Python mirror)
    DR = ttnn.DRAM_MEMORY_CONFIG
    for name, local, dtype, mfn, pfn in (
        ("AR(tp) 8x4096 bf16", (8, 4096), BF, ccl.ar_tp, lambda t: ttnn.all_reduce(t, cluster_axis=1, memory_config=DR)),
        ("AR(dp) 32x4096 bf16", (32, 4096), BF, ccl.ar_dp, lambda t: ttnn.all_reduce(t, cluster_axis=0, memory_config=DR)),
        ("AR(tp) 1024x4096 bf16", (1024, 4096), BF, ccl.ar_tp,
         lambda t: ttnn.all_reduce(t, cluster_axis=1, memory_config=DR)),
    ):  # fmt: skip
        _, x = _per_chip(mesh_device, local, dtype, seed=7)
        try:
            tm, tp_ = _traced_us(mesh_device, lambda: mfn(x)), _traced_us(mesh_device, lambda: pfn(x))
            em, ep = _eager_us(mesh_device, lambda: mfn(x)), _eager_us(mesh_device, lambda: pfn(x))
            report.append(
                f"LATENCY {name}: traced MotifCCL {tm:.1f} us vs ttnn.all_reduce {tp_:.1f} us; eager {em:.0f} vs {ep:.0f} us"
            )
            if tm > tp_ * 1.15 + 3.0:
                failures.append(f"FAIL {name}: traced MotifCCL {tm:.1f} us is slower than ttnn.all_reduce {tp_:.1f} us")
        except Exception as e:
            failures.append(f"FAIL latency {name}: {type(e).__name__}: {str(e)[:300]}")
        _free(x)
    print("[l1small] " + "\n[l1small] ".join(report), flush=True)
    assert not failures, "\n".join(failures)


class _Capture:
    """Exception-safe trace capture (a dangling capture hung close_mesh_device once; GATES_RESULTS §11.6)."""

    def __init__(self, mesh_device):
        self.mesh, self.tid = mesh_device, None

    def __enter__(self):
        self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            ttnn.end_trace_capture(self.mesh, self.tid, cq_id=0)
        except Exception:
            if exc_type is None:
                raise
        if exc_type is not None:
            try:
                ttnn.release_trace(self.mesh, self.tid)
            except Exception:
                pass
        return False


def _trace_min_us(mesh_device, fn, n: int, reps: int) -> float:
    with _Capture(mesh_device) as cap:
        for _ in range(n):
            _free(fn())
    try:
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            times.append((time.perf_counter() - t0) * 1e6)
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    return min(times)


def _traced_us(mesh_device, fn, n: int = 32, reps: int = 7) -> float:
    """Traced per-op us by the gates' slope method: (min t(n) - min t(n/2)) / (n/2)."""
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    t1 = _trace_min_us(mesh_device, fn, n // 2, reps)
    t2 = _trace_min_us(mesh_device, fn, n, reps)
    return max((t2 - t1) / (n - n // 2), 0.0)


def _eager_us(mesh_device, fn, iters: int = 10) -> float:
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


# ======================================================================================================================
# attention in serving order, one session, bfp8 then bf16 KV (the P0 regression)
# ======================================================================================================================
ATTN_KEYS = ("wq_a", "q_norm", "wq_b", "wq_b_gate", "wkv_a", "kv_norm", "wkv_b", "lambda_proj", "wo")
EARLY = [(3, 1000), (12, 3000)]  # (lane, prompt length) prefilled first: buckets 1024 and 4096 (DP rows 0 and 1)
LATE = [(21, 2500)]  # a new user prefilled after the decode steps (bucket 4096, with its cache fill; DP row 2)
USERS = EARLY + LATE
STEPS = (2, 2)  # decode steps after the first prefills / after the prefills that follow decode
PCC_MIN = 0.999


def _ref_args():
    from models.demos.motif3.reference.config import MotifArgs

    return MotifArgs.from_hf_config(HF_META)


def _random_attn_tensors(args, seed: int) -> Dict[str, torch.Tensor]:
    """Real-dim GDLA weights (``reference.weights.random_state_dict`` statistics), bf16-representable fp32."""
    g = torch.Generator().manual_seed(seed)
    D, H, hd, v, r = args.hidden_size, args.num_attention_heads, args.head_dim, args.v_head_dim, args.kv_lora_rank

    def linear(o, i):
        return torch.randn(o, i, generator=g) * i**-0.5

    t = {
        "wq_a": linear(args.q_lora_rank, D),
        "q_norm": torch.rand(args.q_lora_rank, generator=g) * 0.5 + 0.5,
        "wq_b": linear(H * hd, args.q_lora_rank),
        "wkv_a": linear(r + args.qk_rope_head_dim, D),
        "kv_norm": torch.rand(r, generator=g) * 0.5 + 0.5,
        "wkv_b": linear(args.num_key_value_heads * (args.qk_nope_head_dim + v), r),
        "lambda_proj": linear(args.n_signal_heads, D),
        "wo": linear(D, args.n_signal_heads * v),
        "wq_b_gate": linear(args.n_signal_heads * v, args.q_lora_rank),
    }
    return {k: x.to(torch.bfloat16).float() for k, x in t.items()}


def _ref_attention(args, layer: int, tensors: Dict[str, torch.Tensor]):
    from models.demos.motif3.reference.modules import GDLAttention

    with torch.device("meta"):
        m = GDLAttention(args, layer)
    m.load_state_dict({f"{k}.weight": v.float() for k, v in tensors.items()}, strict=True, assign=True)
    return m.eval().requires_grad_(False)


def _ref_rope(ref, positions: torch.Tensor):
    from models.demos.motif3.reference.rope import rope_cos_sin

    return rope_cos_sin(ref.inv_freq(), positions, torch.float32)


def _ref_rows(ref, x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Reference output for query rows ``rows`` of a prefill over ``x [S, D]`` (positions 0..S-1): full-sequence K/V,
    attention for the selected rows only (the reference module's own methods, fp32)."""
    from models.demos.motif3.reference.rope import apply_rope

    S = x.shape[0]
    xs = x[rows][None]
    q, gate = ref.project_q(xs)
    q_nope, q_pe = torch.split(q, [ref.nope_dim, ref.rope_dim], dim=-1)
    cos_q, sin_q = _ref_rope(ref, rows[None])
    q_pe = apply_rope(q_pe, cos_q, sin_q)
    cos, sin = _ref_rope(ref, torch.arange(S)[None])
    c, k_pe = ref.project_kv(x[None], cos, sin)
    heads = ref._heads_expanded(q_nope, q_pe, c, k_pe, rows[None], torch.arange(S), x.dtype)
    lam = torch.sigmoid(ref.lambda_proj(xs).float()).to(x.dtype).unsqueeze(-1)
    signal, noise = ref._split_signal_noise(heads)
    diff = (signal - lam * noise) * torch.sigmoid(gate)
    return ref.wo(diff.reshape(1, len(rows), -1))[0]


def _ref_prefill(ref, x: torch.Tensor, rc, rows: Optional[torch.Tensor]) -> torch.Tensor:
    P = x.shape[0]
    if rows is None:
        return ref(x[None], torch.arange(P)[None], rc)[0]
    cos, sin = _ref_rope(ref, torch.arange(P)[None])
    c, k_pe = ref.project_kv(x[None], cos, sin)
    rc.update(c, k_pe, torch.arange(P)[None])
    return _ref_rows(ref, x, rows)


def _rows_to_check(P: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.cat([torch.arange(0, 8), torch.arange(120, 136), torch.randint(136, P - 16, (24,), generator=g),
                      torch.arange(P - 16, P)]).unique()  # fmt: skip


def _replicated(mesh_device, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _alloc_cache(mesh_device, cfg, num_blocks: int, block: int):
    empty = ttnn.empty(
        [num_blocks, 1, block, cfg.kv_latent_dim], cfg.dtypes.kv_cache, ttnn.TILE_LAYOUT, mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )  # fmt: skip
    cache = ttnn.fill(empty, 0.0)
    ttnn.deallocate(empty)
    return cache


def _decode_inputs(mesh_device, cfg, rope, pos_t: List[int], x32: torch.Tensor, pt: torch.Tensor):
    from models.demos.motif3.tt.attention import MotifAttention
    from models.demos.motif3.tt.rope import shard_lanes

    def rows(t, dtype, layout):
        return shard_lanes(t, cfg, mesh_device, dtype=dtype, layout=layout, device=mesh_device)

    x_tt = rows(x32.reshape(cfg.dp, 1, cfg.lanes_per_row, x32.shape[-1]), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    cur = rows(torch.tensor(pos_t, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    pt_tt = rows(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    rot_idx = rope.rot_idxs_device(torch.tensor(pos_t))
    rot = MotifAttention.decode_rope_tables(rope, rot_idx)
    act = MotifAttention.active_mask_from_cur_pos(cur, cfg.lanes_per_row)
    return {"x": x_tt, "cur": cur, "pt": pt_tt, "rot_idx": rot_idx, "rot": rot, "act": act}


def _free_inputs(d):
    _free([d[k] for k in ("x", "cur", "pt", "rot_idx", "act")] + [t for cs in d["rot"].values() for t in cs])


def _per_lane(cfg, mesh_device, out) -> torch.Tensor:
    full = device_tensors_to_torch(out, mesh_device).float()  # [R, C, 1, 1, 8, 4096]
    res = torch.zeros(cfg.max_batch, full.shape[-1])
    for dp in range(cfg.dp):
        r, c = cfg.axes.coord(dp, 0)
        res[cfg.lanes_per_row * dp : cfg.lanes_per_row * (dp + 1)] = full[r, c, 0, 0]
    return res


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_attention_serving_order_l1_small(mesh_device, device_params):
    from models.demos.motif3.reference.cache import LatentKVCache
    from models.demos.motif3.tt import weights as W
    from models.demos.motif3.tt.attention import MotifAttention
    from models.demos.motif3.tt.rope import MotifRope

    cfg0 = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device)
    rep = log_fabric(mesh_device, "l1small attention serving order")
    assert rep["l1_small"] is not None and rep["l1_small"] >= DEFAULT_L1_SMALL_SIZE, rep
    ccl = MotifCCL(mesh_device, cfg0)
    rope = MotifRope(mesh_device, cfg0)
    assert ccl.l1_small_semaphores
    args = _ref_args()
    B, rank, block = cfg0.max_batch, cfg0.kv_lora_rank, cfg0.kv_block_size
    total = sum(STEPS)
    n_blk = {lane: math.ceil((P + total) / block) for lane, P in USERS}
    width = max(max(n_blk.values()), max(cfg0.prefill_page_table_entries(cfg0.prefill_bucket(P)) for _, P in USERS))
    pool = 1 + sum(n_blk.values()) + 2
    pt = torch.zeros(B, width, dtype=torch.int32)
    nxt = 1
    for lane, _ in USERS:
        pt[lane, : n_blk[lane]] = torch.arange(nxt, nxt + n_blk[lane], dtype=torch.int32)
        nxt += n_blk[lane]
    seqs = {lane: torch.randn(P + total, 4096, generator=torch.Generator().manual_seed(lane)).bfloat16().float()
            for lane, P in USERS}  # fmt: skip
    pads = {lane: torch.randn(cfg0.prefill_bucket(P), 4096, generator=torch.Generator().manual_seed(100 + lane))
            .bfloat16().float() for lane, P in USERS}  # fmt: skip
    failures: List[str] = []
    usage0 = None
    for kv_name in ("bfp8", "bf16"):  # both caches in this ONE mesh session
        cfg = dataclasses.replace(cfg0, dtypes=dataclasses.replace(cfg0.dtypes, kv_cache=kv_cache_dtype_from_name(kv_name)))
        L = {}
        for layer in (0, 1):  # 0: global (YaRN, no window), 1: SWA (window 129)
            tensors = _random_attn_tensors(args, seed=900 + layer)
            src = W.DictWeightSource({W.hf_name(layer, f"self_attn.{k}.weight"): v for k, v in tensors.items()})
            attn = MotifAttention(mesh_device, cfg, layer, source=src, ccl=ccl, rope=rope, cache=False,
                                  require_l1_small=True)  # fmt: skip
            L[layer] = {
                "attn": attn,
                "ref": _ref_attention(args, layer, tensors),
                "cache": _alloc_cache(mesh_device, cfg, pool, block),
                "rc": {lane: LatentKVCache(1, P + total, rank, cfg.rope_dim, torch.float32) for lane, P in USERS},
            }
        if usage0 is None:
            ttnn.synchronize_device(mesh_device)
            usage0 = l1_usage(mesh_device)  # before the first CCL program of the session
        first, done = {}, {lane: 0 for lane, _ in USERS}

        def prefill(lane, P, again):
            """``again``: the same prompt once more after decode steps, WITHOUT the cache fill (its bucket padding
            would overwrite the latents decode wrote into the user's last block); the output must be bitwise equal."""
            S = cfg.prefill_bucket(P)
            x = pads[lane].clone()
            x[:P] = seqs[lane][:P]
            x_tt = _replicated(mesh_device, x[None, None], ttnn.bfloat16)
            n_pt = cfg.prefill_page_table_entries(S)
            pt_u = _replicated(mesh_device, pt[lane : lane + 1, :n_pt], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            for layer, M in L.items():
                t0 = time.perf_counter()
                try:
                    kw = {} if again else {"page_table": pt_u, "kv_cache": M["cache"]}
                    out = M["attn"].forward_prefill(x_tt, **kw)
                    ttnn.synchronize_device(mesh_device)
                except Exception as e:  # the static-CB clash this test exists for
                    failures.append(f"{kv_name} L{layer} prefill lane {lane} S={S} again={again}: "
                                    f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")  # fmt: skip
                    continue
                ms = (time.perf_counter() - t0) * 1e3
                got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0]
                same = replicas_identical(out, mesh_device, "tp", cfg.axes) and replicas_identical(
                    out, mesh_device, "dp", cfg.axes
                )
                _free(out)
                tag = f"{kv_name} L{layer} {cfg.layer(layer).attn_kind} prefill lane {lane} P={P} (S={S})"
                if again:
                    bit = torch.equal(got, first[(layer, lane)])
                    log(f"{tag} after decode steps: bitwise equal to the first prefill {bit}; replicas {same}; {ms:.1f} ms")
                    if not (bit and same):
                        failures.append(f"{tag}: repeat bitwise {bit} replicas {same}")
                    continue
                first[(layer, lane)] = got
                rows = _rows_to_check(P, lane) if P > 2048 else None
                want = _ref_prefill(M["ref"], seqs[lane][:P], M["rc"][lane], rows)
                p = _pcc(want, got[rows] if rows is not None else got[:P])
                log(f"{tag}: pcc {p:.6f}; replicas(32) {same}; {ms:.1f} ms")
                if p < PCC_MIN or not same or not bool(torch.isfinite(got[:P]).all()):
                    failures.append(f"{tag}: pcc {p} replicas {same}")
            _free([x_tt, pt_u])

        def decode(tag, users):
            pos_t, x32 = [-1] * B, torch.zeros(B, 4096)
            for lane, P in users:
                pos_t[lane] = P + done[lane]
                x32[lane] = seqs[lane][pos_t[lane]]
            d = _decode_inputs(mesh_device, cfg, rope, pos_t, x32, pt)
            for layer, M in L.items():
                try:
                    out = M["attn"].forward_decode(d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"],
                                                   kv_cache=M["cache"], active=d["act"])  # fmt: skip
                    ttnn.synchronize_device(mesh_device)
                except Exception as e:
                    failures.append(f"{kv_name} L{layer} decode {tag}: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
                    continue
                got = _per_lane(cfg, mesh_device, out)
                same = replicas_identical(out, mesh_device, "tp", cfg.axes)
                _free(out)
                per = {}
                for lane, _ in users:
                    want = M["ref"](x32[lane][None, None], torch.tensor([[pos_t[lane]]]), M["rc"][lane])[0, 0]
                    per[lane] = _pcc(want, got[lane])
                zero = all(float(got[l].abs().max()) == 0.0 for l in range(B) if pos_t[l] < 0)
                log(f"{kv_name} L{layer} {cfg.layer(layer).attn_kind} decode {tag} positions "
                    f"{[pos_t[l] for l, _ in users]}: per-lane pcc {[round(v, 6) for v in per.values()]}; replicas(tp) "
                    f"{same}; inactive lanes zero {zero}")  # fmt: skip
                if min(per.values()) < PCC_MIN or not same or not zero:
                    failures.append(f"{kv_name} L{layer} decode {tag}: pcc {per} same={same} zero={zero}")
            _free_inputs(d)
            for lane, _ in users:
                done[lane] += 1

        for lane, P in EARLY:  # phase 1: global (and SWA) prefills at S = 1024 and 4096, with the cache fill
            prefill(lane, P, again=False)
        for i in range(STEPS[0]):  # phase 2: decode steps (the decode all_reduce creates its CCL programs here)
            decode(f"phase-2 step {i}", EARLY)
        u = l1_usage(mesh_device)
        log(f"{kv_name} after the decode steps: main L1 {u['l1']} B, L1_SMALL {u['l1_small']} B allocated per bank")
        for lane, P in EARLY:  # phase 3a: the same global prefills after decode -- the case that used to throw
            prefill(lane, P, again=True)
        for lane, P in LATE:  # phase 3b: a new user's prefill (with its cache fill) between decode steps
            prefill(lane, P, again=False)
        for i in range(STEPS[1]):  # phase 4: decode all users
            decode(f"phase-4 step {i}", USERS)
        for M in L.values():
            _free(M["cache"])
    ttnn.synchronize_device(mesh_device)
    usage1 = l1_usage(mesh_device)
    log(f"session end: main L1 {usage0['l1']} -> {usage1['l1']} B, L1_SMALL {usage0['l1_small']} -> "
        f"{usage1['l1_small']} B allocated per bank (region {rep['l1_small']} B)")  # fmt: skip
    if usage1["l1"] != usage0["l1"]:
        failures.append(f"main L1 holds {usage1['l1'] - usage0['l1']} B more after the session (persistent L1 state)")
    assert not failures, "\n".join(failures)
