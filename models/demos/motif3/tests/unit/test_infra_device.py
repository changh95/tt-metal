# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests of the shared infra on the BH Galaxy (run ONLY through scripts/devrun.sh):

    scripts/devrun.sh -t 1500 -n infra_device -- \
        python -m pytest models/demos/motif3/tests/unit/test_infra_device.py -s -p no:cacheprovider

* ``test_ccl_payloads_eager_and_trace`` (4x8; FABRIC_2D_TORUS_XY and the FABRIC_1D_RING fallback): every Motif
  draft-1 collective through ``MotifCCL`` with per-chip distinct inputs, checked against torch; replicas bitwise
  identical along the reduced axis (incl. ``ag_dp_rows``, the ROW_MAJOR MoE token gather); then the decode chain
  AR(tp) -> ag_dp_rows -> AR(dp) -> partition(dp) captured in a trace and replayed with new inputs, and the traced
  per-op cost of the token gather (ag_dp_rows vs the TILE gather). Eager / traced latencies are printed; ``cfg.fabric``
  must equal the fabric the mesh was opened with (INFRA-6).
* ``test_mesh_mappers_and_cache`` (4x8): weights.as_tensor role mappings (replicate / tp / dp / 2D / EP)
  reassemble per chip exactly like ``weights.shard_for_device``; cache files are written, then reloaded without
  touching the torch source.
* ``test_rope_device`` (4x8): MotifRope per-lane decode gather (both layouts) and prefill tables vs the host
  tables; composite and rotary_embedding_hf application vs torch.
* ``test_roles_on_8x4`` (8x4): the TP axis is detected as cluster_axis 0 and the role names follow it.
* ``test_program_configs_reproduce_gates`` (4x8): the shared program-config builders and compute roles (INFRA-3/4)
  reproduce gates G1 (paged FlashMLA decode, incl. the SWA causal-edge probe), G2 (SDPA prefill, S = 128 / 1024) and
  G6 (1D-mcast bfp8 expert matmuls): accuracy and traced latency.

Every test logs the committed fabric topology first (``[motif3.fabric] committed: ...``, ``ccl.log_fabric``). The
meshes open with the shared ``device_params()``, i.e. with ``l1_small_size=32768``, so every collective here runs on
MotifCCL's L1_SMALL semaphore paths (``tt/ccl.py``); their main-L1 footprint and bitwise identity with the plain ttnn
ops are asserted separately in ``test_infra_l1_small.py``.
"""

import importlib
import os
import shutil
import time

import pytest
import torch

import ttnn
from models.demos.motif3.tt import weights as W
from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, log_fabric, replicas_identical
from models.demos.motif3.tt.model_config import (
    DEFAULT_HF_META_DIR,
    DEFAULT_TT_CACHE_ROOT,
    MotifTTConfig,
    device_params,
    make_compute_kernel_config,
)
from models.demos.motif3.tt.rope import MotifRope, apply_rope_torch, positions_to_rot_idxs

HF_META = str(DEFAULT_HF_META_DIR)
TRACE = 32 * 1024 * 1024
FABRIC_TYPES = ("TORUS_XY", "TORUS_Y", "TORUS_X", "MESH")


def _pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def _mapper_2d(mesh_device):
    R, C = tuple(mesh_device.shape)
    return ttnn.create_mesh_mapper(
        mesh_device, ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], ttnn.MeshShape(R, C))
    )


def _per_chip(mesh_device, local, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, seed=0, device=True):
    """Distinct data per chip: host [R, C, h, w] -> chip (r, c) holds [1, 1, h, w] = host[r, c]."""
    R, C = tuple(mesh_device.shape)
    g = torch.Generator().manual_seed(seed)
    host = torch.randn(R, C, *local, generator=g)
    if dtype == ttnn.bfloat16:
        host = host.to(torch.bfloat16).float()
    t = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=layout,
        device=mesh_device if device else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
        mesh_mapper=_mapper_2d(mesh_device),
    )
    return host, t


def _readback(t, mesh_device):
    return device_tensors_to_torch(t, mesh_device).float()  # [R, C, 1, 1, h, w]


def _golden(op, host, cfg, dim=2):
    """Expected per-chip outputs [R, C, h', w'] for host [R, C, h, w] of distinct per-chip inputs."""
    a = cfg.axes
    R, C = a.mesh_shape
    out = {}
    for r in range(R):
        for c in range(C):
            dp, tp = a.roles(r, c)
            along_dp = [host[a.coord(d, tp)] for d in range(a.dp_size)]
            along_tp = [host[a.coord(dp, t)] for t in range(a.tp_size)]
            if op == "ag_dp":
                v = torch.cat(along_dp, dim=dim - 2)
            elif op in ("ar_dp", "ar_dp_rsag"):
                v = torch.stack(along_dp).sum(0)
            elif op == "ar_tp":
                v = torch.stack(along_tp).sum(0)
            elif op == "rs_dp":
                v = torch.stack(along_dp).sum(0).chunk(a.dp_size, dim=dim - 2)[dp]
            elif op == "part_dp":
                v = host[r, c].chunk(a.dp_size, dim=dim - 2)[dp]
            else:
                raise ValueError(op)
            out[(r, c)] = v
    return torch.stack([torch.stack([out[(r, c)] for c in range(C)]) for r in range(R)])


def _check(name, dev, ref, exact, dtype):
    """Exact for data movement; for reductions return a precision summary. fp32 reductions are bounded at TF32
    class (measured on BH: the CCL reduction of fp32 inputs is ~1e-3 relative, not exact fp32)."""
    dev = dev.reshape(ref.shape)
    if exact:
        assert torch.equal(dev, ref), f"{name}: max abs diff {(dev - ref).abs().max().item()}"
        return "exact"
    err = (dev - ref).abs().max().item()
    scale = ref.abs().max().item()
    pcc = _pcc(dev, ref)
    tol = 2e-3 if dtype == ttnn.float32 else 3e-2
    assert pcc > (0.9999999 if dtype == ttnn.float32 else 0.9999) and err <= tol * max(
        scale, 1.0
    ), f"{name}: pcc {pcc} err {err}"
    return f"pcc {pcc:.8f} max_err {err:.2e} (max |ref| {scale:.2f})"


def _timed(fn, mesh_device, iters=5):
    fn()
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    ttnn.synchronize_device(mesh_device)
    return out, (time.perf_counter() - t0) / iters * 1e6


def _free(o):
    if isinstance(o, (list, tuple)):
        for x in o:
            _free(x)
    elif isinstance(o, ttnn.Tensor):
        ttnn.deallocate(o)


class _Capture:
    """Exception-safe trace capture: a raise inside the capture still ends and releases it (a dangling capture hung
    close_mesh_device on 2026-10-01 and needed a tt-smi reset; GATES_RESULTS.md §11.6)."""

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


def _trace_min_us(mesh_device, fn, n, reps):
    """Min wall time of replaying a trace of ``n`` back-to-back ``fn()`` calls (outputs freed inside the capture)."""
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


def _eager_us(mesh_device, fn, iters=10, warmup=1):
    """Mean eager us per call with outputs freed right after dispatch (gate_utils.time_eager)."""
    for _ in range(warmup):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


def _traced_us(mesh_device, fn, n=32, reps=7):
    """Traced per-op us by the gates' slope method (tests/unit/gates/gate_utils.time_traced): (t(n) - t(n/2)) / (n/2)
    with the minimum over replays (replays are bimodal; synchronize_device alone costs ~140 us on this mesh)."""
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    t1 = _trace_min_us(mesh_device, fn, n // 2, reps)
    t2 = _trace_min_us(mesh_device, fn, n, reps)
    return max((t2 - t1) / (n - n // 2), 0.0)


def _check_fabric(mesh_device, device_params, cfg, tag):
    """INFRA-6: cfg.fabric is the fabric the mesh was opened with; log what the topology mapper committed."""
    rep = log_fabric(mesh_device, tag)
    want = device_params["fabric_config"]
    assert cfg.fabric == getattr(want, "name", str(want).split(".")[-1]), (cfg.fabric, want)
    assert rep["requested"] == cfg.fabric and rep["committed"] in FABRIC_TYPES, rep
    rings = {"TORUS_XY": (True, True), "TORUS_Y": (True, False), "TORUS_X": (False, True), "MESH": (False, False)}
    assert (rep["tp_ring"], rep["dp_ring"]) == rings[rep["committed"]], rep  # (8, 4) graph: Y = the size-8 TP axis
    return rep


MESH_4x8 = [
    pytest.param((4, 8), device_params("FABRIC_2D_TORUS_XY", TRACE), id="4x8-torus2d"),
    pytest.param((4, 8), device_params("FABRIC_1D_RING", TRACE), id="4x8-ring1d"),
]


@pytest.mark.parametrize("mesh_device, device_params", MESH_4x8, indirect=True)
def test_ccl_payloads_eager_and_trace(mesh_device, device_params):
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device)
    assert cfg.axes.tp_axis == 1 and cfg.axes.dp_axis == 0
    ccl = MotifCCL(mesh_device, cfg)
    assert ccl.cluster_axis("tp") == 1 and ccl.cluster_axis("rows") == 0
    fab = _check_fabric(mesh_device, device_params, cfg, "ccl")
    print(f"\n[infra] fabric={device_params['fabric_config']} report={fab} {cfg.describe()}")

    cases = [
        # name, op, local shape, dtype, layout, exact
        ("AG(dp) moe gather 8x4096 TILE", "ag_dp", (8, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
        ("AG(dp) moe gather 8x4096 RM", "ag_dp", (8, 4096), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, True),
        ("ag_dp_rows moe gather 8x4096 TILE->RM->TILE", "ag_dp_rows", (8, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
        ("ag_dp_rows moe gather 8x4096 RM in", "ag_dp_rows", (8, 4096), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, True),
        ("AR(dp) moe combine 32x4096 fp32", "ar_dp", (32, 4096), ttnn.float32, ttnn.TILE_LAYOUT, False),
        ("RS+AG(dp) fp32 HiFi4/fp32-acc ckc (diag)", "ar_dp_rsag", (32, 4096), ttnn.float32, ttnn.TILE_LAYOUT, False),
        ("AR(dp) moe combine 32x4096 bf16", "ar_dp", (32, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, False),
        ("AR(tp) wo/moe out 8x4096 bf16", "ar_tp", (8, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, False),
        ("AR(tp) polynorm moments 8x32 fp32", "ar_tp", (8, 32), ttnn.float32, ttnn.TILE_LAYOUT, False),
        ("AR(tp) prefill wo 128x4096 bf16", "ar_tp", (128, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, False),
        ("RS(dp) prefill moe 128x4096 bf16", "rs_dp", (128, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, False),
        ("AG(dp) prefill moe 32x4096 bf16", "ag_dp", (32, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
        ("partition(dp) 32x4096 TILE", "part_dp", (32, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
        ("partition(dp) 32x4096 RM", "part_dp", (32, 4096), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, True),
    ]
    fns = {
        "ag_dp": lambda x: ccl.ag_dp(x, 2),
        "ag_dp_rows": ccl.ag_dp_rows,
        "ar_dp": ccl.ar_dp,
        "ar_tp": ccl.ar_tp,
        "rs_dp": lambda x: ccl.rs_dp(x, 2),
        "part_dp": lambda x: ccl.partition(x, 2, "dp"),
        "ar_dp_rsag": lambda x: ccl.all_gather(
            ccl.reduce_scatter(x, 3, "dp", compute_kernel_config=cfg.compute_config("router")), 3, "dp"
        ),
    }
    results, failures = [], []
    for i, (name, op, local, dtype, layout, exact) in enumerate(cases):
        host, x = _per_chip(mesh_device, local, dtype, layout, seed=i)
        try:
            out, us = _timed(lambda: fns[op](x), mesh_device)
            ref = _golden("ag_dp" if op == "ag_dp_rows" else op, host, cfg)
            prec = _check(name, _readback(out, mesh_device), ref, exact, dtype)
            if op in ("ar_dp", "ar_tp", "ar_dp_rsag"):
                assert replicas_identical(out, mesh_device, op[3:5], cfg.axes), f"{name}: replicas differ"
            if op == "ag_dp_rows":
                assert out.layout == ttnn.TILE_LAYOUT and list(out.shape) == [1, 1, 32, 4096], (out.layout, out.shape)
                assert replicas_identical(out, mesh_device, "dp", cfg.axes), f"{name}: replicas differ"
            results.append(f"PASS {name}: {us:.1f} us/op eager; {prec}")
            ttnn.deallocate(out)
        except Exception as e:  # keep going: one run reports every payload
            failures.append(f"FAIL {name}: {type(e).__name__}: {str(e)[:400]}")
            results.append(failures[-1])
        ttnn.deallocate(x)

    # ---- decode chain in a trace: AR(tp) -> AG(dp) -> AR(dp) -> partition(dp) ------------------------
    host0, x = _per_chip(mesh_device, (8, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, seed=100)

    def chain(inp):
        a = ccl.ar_tp(inp)
        g = ccl.ag_dp_rows(a)  # MOE-1: the ROW_MAJOR token gather
        s = ccl.ar_dp(g)
        p = ccl.partition(s, 2, "dp")
        return a, g, s, p

    def check_chain(outs, host):
        a, g, s, p = (_readback(t, mesh_device) for t in outs)
        R, C = cfg.axes.mesh_shape
        _check("chain AR(tp)", a, _golden("ar_tp", host, cfg), False, ttnn.bfloat16)
        a_h = a.reshape(R, C, 8, 4096)
        _check("chain AG(dp)", g, _golden("ag_dp", a_h, cfg), True, ttnn.bfloat16)
        g_h = g.reshape(R, C, 32, 4096)
        _check("chain AR(dp)", s, _golden("ar_dp", g_h, cfg), False, ttnn.bfloat16)
        s_h = s.reshape(R, C, 32, 4096)
        _check("chain partition", p, _golden("part_dp", s_h, cfg), True, ttnn.bfloat16)
        assert replicas_identical(outs[0], mesh_device, "tp", cfg.axes)
        assert replicas_identical(outs[2], mesh_device, "dp", cfg.axes)

    try:
        outs, eager_us = _timed(lambda: chain(x), mesh_device, iters=3)  # compiles everything before capture
        check_chain(outs, host0)
        for t in outs:
            ttnn.deallocate(t)
        with _Capture(mesh_device) as cap:
            outs = chain(x)
        tid = cap.tid
        try:
            for it in range(3):
                host_new, host_t = _per_chip(
                    mesh_device, (8, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, seed=200 + it, device=False
                )
                ttnn.copy_host_to_device_tensor(host_t, x)
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                check_chain(outs, host_new)
            t0 = time.perf_counter()
            for _ in range(20):
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            trace_us = (time.perf_counter() - t0) / 20 * 1e6
        finally:
            ttnn.release_trace(mesh_device, tid)
        results.append(
            f"PASS decode chain AR(tp)+ag_dp_rows+AR(dp)+partition(dp): eager {eager_us:.1f} us, "
            f"traced {trace_us:.1f} us/replay"
        )
    except Exception as e:
        failures.append(f"FAIL decode chain (eager/trace): {type(e).__name__}: {str(e)[:400]}")
        results.append(failures[-1])

    # ---- INFRA-5 acceptance: traced per-op cost of the token gather, ROW_MAJOR path vs TILE (gate G4: 12 vs 62 us)
    try:
        _, xg = _per_chip(mesh_device, (8, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, seed=300)
        t_rows = _traced_us(mesh_device, lambda: ccl.ag_dp_rows(xg))
        t_tile = _traced_us(mesh_device, lambda: ccl.ag_dp(xg, 2))
        L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
        t_rows_dram = _traced_us(mesh_device, lambda: ccl.ag_dp_rows(xg, intermediate_memory_config=DRAM))
        xrm = ttnn.to_layout(xg, ttnn.ROW_MAJOR_LAYOUT)
        t_rm_only = _traced_us(mesh_device, lambda: ccl.all_gather(xrm, 2, "dp"))
        t_rm_l1 = _traced_us(mesh_device, lambda: ccl.all_gather(xrm, 2, "dp", memory_config=L1))
        t_untilize = _traced_us(mesh_device, lambda: ttnn.to_layout(xg, ttnn.ROW_MAJOR_LAYOUT))
        g_rm = ccl.all_gather(xrm, 2, "dp")
        g_rm_l1 = ttnn.to_memory_config(g_rm, L1)
        t_tilize = _traced_us(mesh_device, lambda: ttnn.to_layout(g_rm, ttnn.TILE_LAYOUT))
        t_tilize_l1 = _traced_us(mesh_device, lambda: ttnn.to_layout(g_rm_l1, ttnn.TILE_LAYOUT, memory_config=L1))
        t_tilize_l1_dram = _traced_us(
            mesh_device, lambda: ttnn.to_layout(g_rm_l1, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        results.append(
            f"INFO traced token gather [1,1,8,4096] -> [1,1,32,4096]: ag_dp_rows {t_rows:.1f} us (L1 intermediates, "
            f"default; DRAM intermediates {t_rows_dram:.1f} us) vs TILE all_gather {t_tile:.1f} us; parts: untilize "
            f"{t_untilize:.1f}, RM all_gather "
            f"{t_rm_only:.1f} (to L1 {t_rm_l1:.1f}), tilize DRAM->DRAM {t_tilize:.1f}, L1->L1 {t_tilize_l1:.1f}, "
            f"L1->DRAM {t_tilize_l1_dram:.1f}"
        )
        if t_rows >= t_tile:
            failures.append(f"FAIL ag_dp_rows ({t_rows:.1f} us) is not faster than the TILE gather ({t_tile:.1f} us)")
    except Exception as e:
        failures.append(f"FAIL token-gather timing: {type(e).__name__}: {str(e)[:400]}")
    finally:
        print("[infra] " + "\n[infra] ".join(results))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", [MESH_4x8[0]], indirect=True)
def test_mesh_mappers_and_cache(mesh_device, device_params):
    root = DEFAULT_TT_CACHE_ROOT / f"_infra_test_{os.getpid()}"
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device, tt_cache_root=root)
    _check_fabric(mesh_device, device_params, cfg, "mappers")
    a = cfg.axes
    R, C = a.mesh_shape
    host = torch.arange(8 * 256, dtype=torch.float32).reshape(8, 256)  # fp32: exact values per element
    try:
        for dp_dim, tp_dim in [(None, None), (None, 1), (0, None), (0, 1)]:
            dev = W.as_tensor(host, mesh_device=mesh_device, cfg=cfg, dtype=ttnn.float32, dp_dim=dp_dim, tp_dim=tp_dim)
            got = device_tensors_to_torch(dev, mesh_device)
            for r in range(R):
                for c in range(C):
                    exp = W.shard_for_device(host, a, r, c, dp_dim=dp_dim, tp_dim=tp_dim)
                    assert torch.equal(got[r, c], exp), f"mapping dp={dp_dim} tp={tp_dim} chip {(r, c)}"
            ttnn.deallocate(dev)

        # EP placement: chip (dp, tp) holds experts [12k, 12k+12), k = 8 dp + tp
        ids = W.as_tensor(
            W.local_expert_ids(cfg), mesh_device=mesh_device, cfg=cfg, dtype=ttnn.float32, dp_dim=0, tp_dim=1
        )
        got = device_tensors_to_torch(ids, mesh_device)  # [R, C, 1, 12, 1, 1]
        for r in range(R):
            for c in range(C):
                dp, tp = a.roles(r, c)
                assert got[r, c].reshape(-1).tolist() == [float(e) for e in cfg.experts_of_chip(dp, tp)]

        # cache: build once (files written), reload without calling the source
        sizes = {}
        for name, kw in [("t.rep", {}), ("t.tp", {"tp_dim": 1}), ("t.ep", {"dp_dim": 0, "tp_dim": 1})]:
            src = host if name != "t.ep" else W.local_expert_ids(cfg)
            built = W.as_tensor(
                src, mesh_device=mesh_device, cfg=cfg, dtype=ttnn.bfloat16, cache_name=name, layer=0, **kw
            )
            assert W.is_cached(cfg, name, 0, ttnn.bfloat16, **kw)

            def boom():
                raise AssertionError("cache hit must not materialize the source")

            loaded = W.as_tensor(
                boom, mesh_device=mesh_device, cfg=cfg, dtype=ttnn.bfloat16, cache_name=name, layer=0, **kw
            )
            assert torch.equal(
                device_tensors_to_torch(built, mesh_device), device_tensors_to_torch(loaded, mesh_device)
            )
            path = W.tensorbin_path(
                W.cache_prefix(cfg, name, 0, kw.get("dp_dim"), kw.get("tp_dim")), ttnn.bfloat16, ttnn.TILE_LAYOUT
            )
            sizes[name] = path.stat().st_size
            assert str(path).startswith(str(root / cfg.cache_version_tag / "mesh4x8" / "L00"))
        # informational: replicate caches one unsharded copy; tp / ep store the shards (dedup of DP replicas?)
        print(f"\n[infra] cache file sizes (bytes) for an 8x256 bf16 tensor / EP ids: {sizes}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("mesh_device, device_params", [MESH_4x8[0]], indirect=True)
def test_rope_device(mesh_device, device_params):
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device)
    _check_fabric(mesh_device, device_params, cfg, "rope")
    a = cfg.axes
    R, C = a.mesh_shape
    rope = MotifRope(mesh_device, cfg)
    g = torch.Generator().manual_seed(0)
    pos = torch.tensor(
        [0, 1, 127, 128, 129, 130, 4095, 4096] + torch.randint(0, 32768, (23,), generator=g).tolist() + [-1]
    )
    idx = rope.rot_idxs_device(pos)
    rows = positions_to_rot_idxs(pos, cfg).long()  # [4, 32]
    report = []
    for kind in ("yarn", "plain"):
        tc, ts = rope.host_tables[kind]
        cos, sin = rope.decode_cos_sin(kind, idx, layout="rows")  # [1,1,32,64]
        cb, sb = rope.decode_cos_sin(kind, idx, layout="batch")  # [1,8,1,64]
        gc = device_tensors_to_torch(cos, mesh_device)
        gb = device_tensors_to_torch(cb, mesh_device)
        for r in range(R):
            for c in range(C):
                dp, _ = a.roles(r, c)
                assert torch.equal(gc[r, c, 0, 0], tc[rows[dp]]), f"{kind} rows gather chip {(r, c)}"
                assert torch.equal(gb[r, c, 0, :, 0], tc[rows[dp, :8]]), f"{kind} batch gather chip {(r, c)}"

        # composite rope, lanes on rows: x [1, 10, 32, 64] (same x on every chip; positions differ per row)
        x = torch.randn(1, 10, 32, 64, generator=g).to(torch.bfloat16)
        xd = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        outs = {
            "composite_rows": rope.apply_composite(xd, cos, sin),
            "hf_prefill_mode_rows": rope.apply_hf(xd, cos, sin, is_decode_mode=False),
        }
        for name, o in outs.items():
            got = device_tensors_to_torch(o, mesh_device)
            worst = 1.0
            for r in range(R):
                for c in range(C):
                    dp, _ = a.roles(r, c)
                    ref = apply_rope_torch(x, tc[rows[dp]][None, None], ts[rows[dp]][None, None])
                    worst = min(worst, _pcc(got[r, c].float(), ref.float()))
                    assert (got[r, c].float() - ref.float()).abs().max() < 0.07, f"{kind} {name} chip {(r, c)}"
            assert worst > 0.9999, f"{kind} {name}: pcc {worst}"
            report.append(f"{kind} {name}: min pcc {worst:.6f}")

        # composite rope, lanes on dim 1: x [1, 8, 10, 64] with cos [1, 8, 1, 64]
        xb = torch.randn(1, 8, 10, 64, generator=g).to(torch.bfloat16)
        xbd = ttnn.from_torch(
            xb,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ob = device_tensors_to_torch(rope.apply_composite(xbd, cb, sb), mesh_device)
        worst = 1.0
        for r in range(R):
            for c in range(C):
                dp, _ = a.roles(r, c)
                ref = apply_rope_torch(xb, tc[rows[dp, :8]][None, :, None], ts[rows[dp, :8]][None, :, None])
                worst = min(worst, _pcc(ob[r, c].float(), ref.float()))
        assert worst > 0.9999, f"{kind} composite batch: pcc {worst}"
        report.append(f"{kind} composite_batch: min pcc {worst:.6f}")

        # HF kernel in decode mode: x [1, 8, 10, 64] and cos/sin [1, 8, 1, 64] HEIGHT_SHARDED one lane per core
        cs, ss = rope.decode_cos_sin(kind, idx, layout="batch_sharded")
        xs = ttnn.to_memory_config(xbd, rope.batch_sharded_memory_config())
        od = device_tensors_to_torch(
            ttnn.to_memory_config(rope.apply_hf(xs, cs, ss, is_decode_mode=True), ttnn.DRAM_MEMORY_CONFIG), mesh_device
        )
        worst = 1.0
        for r in range(R):
            for c in range(C):
                dp, _ = a.roles(r, c)
                ref = apply_rope_torch(xb, tc[rows[dp, :8]][None, :, None], ts[rows[dp, :8]][None, :, None])
                worst = min(worst, _pcc(od[r, c].float(), ref.float()))
        assert worst > 0.9999, f"{kind} hf decode-mode sharded: pcc {worst}"
        report.append(f"{kind} hf_decode_mode_sharded: min pcc {worst:.6f}")

        # prefill: positions 0..S-1, x [1, 10, S, 64], HF kernel and composite
        S = 128
        pc, ps = rope.prefill_cos_sin(kind, S)
        xp = torch.randn(1, 10, S, 64, generator=g).to(torch.bfloat16)
        xpd = ttnn.from_torch(
            xp,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        ref = apply_rope_torch(xp, tc[:S][None, None], ts[:S][None, None])
        for name, o in (
            ("hf_prefill", rope.apply_hf(xpd, pc, ps)),
            ("composite_prefill", rope.apply_composite(xpd, pc, ps)),
        ):
            got = device_tensors_to_torch(o, mesh_device)[0, 0].float()
            p = _pcc(got, ref.float())
            assert p > 0.9999, f"{kind} {name}: pcc {p}"
            report.append(f"{kind} {name}: pcc {p:.6f}")
    print("\n[infra] rope " + "\n[infra] rope ".join(report))


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((8, 4), device_params("FABRIC_2D_TORUS_XY", TRACE), id="8x4-torus2d")],
    indirect=True,
)
def test_roles_on_8x4(mesh_device, device_params):
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device)
    _check_fabric(mesh_device, device_params, cfg, "8x4")
    assert cfg.axes.tp_axis == 0 and cfg.axes.dp_axis == 1 and cfg.tp == 8 and cfg.dp == 4
    ccl = MotifCCL(mesh_device, cfg)
    assert ccl.cluster_axis("tp") == 0 and ccl.cluster_axis("dp") == 1
    failures = []
    for i, (op, local, exact) in enumerate(
        [
            ("ar_tp", (8, 4096), False),
            ("ag_dp", (8, 4096), True),
            ("ar_dp", (32, 4096), False),
            ("part_dp", (32, 4096), True),
        ]
    ):
        host, x = _per_chip(mesh_device, local, ttnn.bfloat16, ttnn.TILE_LAYOUT, seed=i)
        fn = {
            "ar_tp": ccl.ar_tp,
            "ar_dp": ccl.ar_dp,
            "ag_dp": lambda t: ccl.ag_dp(t, 2),
            "part_dp": lambda t: ccl.partition(t, 2, "dp"),
        }[op]
        try:
            out = fn(x)
            _check(f"8x4 {op}", _readback(out, mesh_device), _golden(op, host, cfg), exact, ttnn.bfloat16)
            if op == "ag_dp":
                rows = ccl.ag_dp_rows(x)
                _check("8x4 ag_dp_rows", _readback(rows, mesh_device), _golden(op, host, cfg), True, ttnn.bfloat16)
            if op in ("ar_tp", "ar_dp"):
                assert replicas_identical(out, mesh_device, op[-2:], cfg.axes), f"8x4 {op}: replicas differ"
        except Exception as e:
            failures.append(f"FAIL 8x4 {op}: {type(e).__name__}: {str(e)[:300]}")
    print("\n[infra] 8x4 roles: " + ("; ".join(failures) if failures else "AR(tp), AG(dp), AR(dp), partition(dp) PASS"))
    assert not failures, "\n".join(failures)
    ids = W.as_tensor(W.local_expert_ids(cfg), mesh_device=mesh_device, cfg=cfg, dtype=ttnn.float32, dp_dim=0, tp_dim=1)
    got = device_tensors_to_torch(ids, mesh_device)
    R, C = cfg.axes.mesh_shape
    for r in range(R):
        for c in range(C):
            dp, tp = cfg.axes.roles(r, c)
            assert got[r, c].reshape(-1).tolist() == [float(e) for e in cfg.experts_of_chip(dp, tp)]


# ======================================================================================================================
# INFRA-3 / INFRA-4: the shared program configs + compute roles reproduce gates G1, G2 and G6
# ======================================================================================================================
def _replicated(mesh_device, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _host_quant(t, dtype):
    """Quantise on the host exactly as from_torch does (bfp8 block exponents), no device."""
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def _dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _mla_probe(B, NH, seq, positions, window, scale, seed):
    """G1 window / causal-edge probe: the first in-window key (V=+1) must win; the key just outside the window (V=-1)
    and the future key p+1 (V=+3) have larger scores and would dominate if they leaked."""
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(64, generator=g)
    e = e / e.norm()
    a = 16.0
    q = 0.05 * torch.randn(B, NH, 576, generator=g)
    q[:, :, 512:] = a * e
    kv = 0.1 * torch.randn(B, seq, 576, generator=g)
    kv[:, :, 512:] = 0.02 * torch.randn(B, seq, 64, generator=g)

    def key(score):
        return (score / (scale * a)) * e

    for b, p in enumerate(positions):
        lo = max(0, p - window + 1)
        kv[b, lo, :512], kv[b, lo, 512:] = 1.0, key(20.0)
        if lo >= 1:
            kv[b, lo - 1, :512], kv[b, lo - 1, 512:] = -1.0, key(28.0)
        if p + 1 < seq:
            kv[b, p + 1, :512], kv[b, p + 1, 512:] = 3.0, key(36.0)
    return q.bfloat16().float(), kv.bfloat16().float()


@pytest.mark.parametrize("mesh_device, device_params", [MESH_4x8[0]], indirect=True)
def test_program_configs_reproduce_gates(mesh_device, device_params):
    gd = importlib.import_module("models.demos.motif3.tests.unit.gates.goldens")
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device)
    _check_fabric(mesh_device, device_params, cfg, "program_configs")
    assert cfg.compute_grid == (12, 10), cfg.compute_grid
    report, failures = [], []

    def expect(ok, msg):
        report.append(("PASS " if ok else "FAIL ") + msg)
        if not ok:
            failures.append(msg)

    # ---------------- G1: paged FlashMLA decode, cfg.flash_mla_decode_pc() + "sdpa_decode" -----------------------
    B, NH, D, block, seq = 8, 10, 576, 64, 5120
    bpu = seq // block
    swa, glob = cfg.layer(1), cfg.layer(0)
    pc, ckc = cfg.flash_mla_decode_pc(), cfg.compute_config("sdpa_decode")
    g = torch.Generator().manual_seed(11)
    pt = torch.randperm(B * bpu, generator=g).reshape(B, bpu).to(torch.int32)
    pt_tt = _replicated(mesh_device, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def upload_cache(kv):
        paged = torch.empty(B * bpu, 1, block, D)
        paged[pt.reshape(-1).long(), 0] = kv.reshape(B * bpu, block, D)
        return _replicated(mesh_device, paged, cfg.dtypes.kv_cache)

    def mla(q_tt, cache, pos_tt, spec):
        return ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_tt,
            cache,
            None,
            head_dim_v=cfg.kv_lora_rank,
            page_table_tensor=pt_tt,
            cur_pos_tensor=pos_tt,
            scale=spec.softmax_scale,
            sliding_window_size=spec.sliding_window_size,
            program_config=pc,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    q = torch.randn(B, NH, D, generator=g).bfloat16().float()
    kv = torch.randn(B, seq, D, generator=g).bfloat16().float()
    kv_q = _host_quant(kv, cfg.dtypes.kv_cache)
    cache = upload_cache(kv)
    q_tt = _replicated(mesh_device, q.reshape(1, B, NH, D), ttnn.bfloat16)
    positions = [0, 1, 127, 128, 129, 130, 1000, 5000]  # design §4.2 / G1
    pos_tt = _replicated(mesh_device, torch.tensor(positions, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    for name, spec in (("swa", swa), ("global", glob)):
        out = mla(q_tt, cache, pos_tt, spec)
        got = _dev0(out)[0, :, :NH, :]
        want = gd.mla_decode_golden(q, kv_q, positions, spec.softmax_scale, spec.sliding_window_size)
        p = _pcc(got, want)
        same = replicas_identical(out, mesh_device, "tp", cfg.axes) and replicas_identical(
            out, mesh_device, "dp", cfg.axes
        )
        err = (got.double() - want).abs().max().item()
        expect(
            p >= 0.9995 and same,
            f"G1 FlashMLA decode {name}: pcc {p:.6f} max_abs {err:.2e} replicas {same} "
            f"(G1 fp32 acc: 0.99994 SWA / 0.99993 global)",
        )
        ttnn.deallocate(out)
    ttnn.deallocate(cache)

    probe_pos = [129, 159, 160, 161, 255, 256, 1000, 5000]  # window start on / around tile and chunk edges
    qp, kvp = _mla_probe(B, NH, seq, probe_pos, swa.sliding_window_size, swa.softmax_scale, seed=3)
    cache_p = upload_cache(kvp)
    qp_tt = _replicated(mesh_device, qp.reshape(1, B, NH, D), ttnn.bfloat16)
    posp_tt = _replicated(mesh_device, torch.tensor(probe_pos, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    means = _dev0(mla(qp_tt, cache_p, posp_tt, swa))[0, :, :NH, :].mean(dim=(1, 2)).tolist()
    expect(
        all(abs(m - 1.0) < 0.15 for m in means),
        f"G1 SWA window/causal-edge probe user means {[round(m, 3) for m in means]} (all ~ +1)",
    )
    ttnn.deallocate(cache_p)

    # traced latency at a 4K context (G1 fp32 acc, Q DRAM: SWA 30.1 us, global 4K 102.8 us)
    cache = upload_cache(kv)
    pos4k_tt = _replicated(mesh_device, torch.full((B,), 4095, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    for name, spec in (("swa", swa), ("global 4K", glob)):
        t = _traced_us(mesh_device, lambda: mla(q_tt, cache, pos4k_tt, spec), n=32, reps=7)
        report.append(f"INFO G1 FlashMLA decode {name}: traced {t:.1f} us")
    ttnn.deallocate(cache)

    # ---------------- G2: SDPA prefill, cfg.sdpa_prefill_pc(layer, S) + "sdpa_prefill" ---------------------------
    ck_pre = cfg.compute_config("sdpa_prefill")
    for S in (128, 1024):
        g = torch.Generator().manual_seed(S)
        qs = torch.randn(1, 10, S, 192, generator=g).bfloat16().float()
        ks = torch.randn(1, 2, S, 192, generator=g).bfloat16().float()
        vs = torch.randn(1, 2, S, 128, generator=g).bfloat16().float()
        v_pad = torch.zeros(1, 2, S, 192)
        v_pad[..., :128] = vs  # the plain op needs dv == dqk (G2)
        q_s, k_s, v_s = (_replicated(mesh_device, t, ttnn.bfloat16) for t in (qs, ks, v_pad))
        for name, spec in (("swa", swa), ("global", glob)):
            pcfg = cfg.sdpa_prefill_pc(spec, seq_len=S)

            def run():
                return ttnn.transformer.scaled_dot_product_attention(
                    q_s,
                    k_s,
                    v_s,
                    is_causal=True,
                    scale=spec.softmax_scale,
                    sliding_window_size=spec.sliding_window_size,
                    program_config=pcfg,
                    compute_kernel_config=ck_pre,
                )

            out = run()
            full = _dev0(out)[0, :, :S, :]
            want = gd.gqa_prefill_golden(qs[0], ks[0], vs[0], spec.softmax_scale, spec.sliding_window_size)
            p = _pcc(full[..., :128], want)
            pad = full[..., 128:].abs().max().item()
            ttnn.deallocate(out)
            us = _eager_us(mesh_device, run, iters=10)
            expect(
                p >= 0.999 and pad == 0.0,
                f"G2 SDPA prefill S={S} {name} chunks q{pcfg.q_chunk_size}/k{pcfg.k_chunk_size}: pcc {p:.6f} pad cols "
                f"{pad} eager {us:.0f} us (G2 S=1024: 0.99975 SWA / 0.99961 global)",
            )

    # ---------------- G6: 1D-mcast bfp8 expert matmuls, cfg.experts_*_pc() + "experts" ---------------------------
    E, T, H, I = cfg.experts_per_chip, 32, cfg.hidden_size, cfg.moe_intermediate_size
    g = torch.Generator().manual_seed(6)
    w_gu = (0.02 * torch.randn(1, E, H, 2 * I, generator=g)).bfloat16().float()
    w_dn = (0.02 * torch.randn(1, E, I, H, generator=g)).bfloat16().float()
    x = torch.randn(1, 1, T, H, generator=g).bfloat16().float().expand(1, E, T, H).contiguous()
    h = torch.randn(1, E, T, I, generator=g).bfloat16().float()
    ex = [0, 5, 11]  # golden on 3 of the 12 experts (bfp8 blocks never straddle experts, so slicing is exact)
    want_gu = (x[0, ex].double() @ _host_quant(w_gu[:, ex], cfg.dtypes.routed_experts)[0].double()).float()
    want_dn = (h[0, ex].double() @ _host_quant(w_dn[:, ex], cfg.dtypes.routed_experts)[0].double()).float()
    wgu_tt = _replicated(mesh_device, w_gu, cfg.dtypes.routed_experts)
    wdn_tt = _replicated(mesh_device, w_dn, cfg.dtypes.routed_experts)
    x_tt = _replicated(mesh_device, x, ttnn.bfloat16)
    h_tt = _replicated(mesh_device, h, ttnn.bfloat16)
    variants = [("experts role: HiFi4 fp32 acc, packer off", cfg.compute_config("experts"))]
    variants.append(("G6 gate config: packer_l1_acc on", make_compute_kernel_config("HiFi4", True, packer_l1_acc=True)))
    for vname, ck in variants:
        for mm, a_tt, w_tt, pcm, want, gate_us, limit in (
            ("gate_up", x_tt, wgu_tt, cfg.experts_gate_up_pc(), want_gu, 433, 700),
            ("down", h_tt, wdn_tt, cfg.experts_down_pc(), want_dn, 226, 275),
        ):

            def run():
                return ttnn.matmul(a_tt, w_tt, program_config=pcm, compute_kernel_config=ck, dtype=ttnn.bfloat16)

            out = run()
            got = _dev0(out)[0, ex]
            p = _pcc(got, want)
            err = (got - want).abs().max().item()
            ttnn.deallocate(out)
            t = _traced_us(mesh_device, run, n=24, reps=5)
            ok = p >= 0.9999 and t < limit
            msg = f"G6 experts {mm} [{vname}]: pcc {p:.7f} max_abs {err:.3e} traced {t:.0f} us (G6 HiFi4: {gate_us} us)"
            if vname.startswith("experts role"):
                expect(ok, msg)
            else:
                report.append("INFO " + msg)
    print("\n[infra] gates-repro " + "\n[infra] gates-repro ".join(report))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", [MESH_4x8[0]], indirect=True)
def test_ag_dp_rows_kernel_layout(mesh_device, device_params):
    """Phase C D4 (``MOTIF3_AG_ROWS_LAYOUT=kernel``, ``MotifCCL(rows_layout="kernel")``; logs/opt/phaseC/D4):
    ``ag_dp_rows`` with the ``tt/kernels/rm_tile.py`` untilize / tilize is bitwise the ``ttnn.to_layout`` path for
    every decode payload (bf16 [L, W] with L = 8 / 16, W = 4096 / 576 / 32, natural and split order, L1 and DRAM
    outputs; bfp8 falls back to the ops), eager and traced (a trace replayed with new inputs == the eager ops path)."""
    log_fabric(mesh_device, "ag_dp_rows_kernel_layout")
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)
    ops = MotifCCL(mesh_device, cfg, rows_layout="ops")
    ker = MotifCCL(mesh_device, cfg, rows_layout="kernel")
    R, C = (int(s) for s in tuple(mesh_device.shape))
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C))
    g = torch.Generator().manual_seed(45)
    failures = []
    for L, W, halves, dt in ((8, 4096, 1, ttnn.bfloat16), (16, 4096, 1, ttnn.bfloat16), (16, 4096, 2, ttnn.bfloat16),
                             (8, 576, 1, ttnn.bfloat16), (16, 576, 2, ttnn.bfloat16), (8, 32, 1, ttnn.bfloat16),
                             (8, 576, 1, ttnn.bfloat8_b)):
        for mc in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
            xh = torch.randn(R, C, L, W, generator=g)
            x = ttnn.from_torch(xh, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            a = ker.ag_dp_rows(x, memory_config=mc, halves=halves)
            b = ops.ag_dp_rows(x, memory_config=mc, halves=halves)
            ok = bool(torch.equal(device_tensors_to_torch(a, mesh_device), device_tensors_to_torch(b, mesh_device)))
            tag = f"L={L} W={W} halves={halves} {dt} out {mc.buffer_type}"
            print(f"[infra] D4 ag_dp_rows kernel == ops bitwise ({tag}): {ok}")
            if not ok:
                failures.append(tag)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            ttnn.deallocate(x)
    # trace: kernel path captured, replayed with a new input, == the eager ops path
    x = ttnn.from_torch(torch.randn(R, C, 8, 4096, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=mesh_device, mesh_mapper=mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(ker.ag_dp_rows(x, memory_config=ttnn.L1_MEMORY_CONFIG))
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = ker.ag_dp_rows(x, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    try:
        for it in range(3):
            new = ttnn.from_torch(torch.randn(R, C, 8, 4096, generator=g), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(new, x)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ref = ops.ag_dp_rows(x, memory_config=ttnn.L1_MEMORY_CONFIG)
            ok = bool(torch.equal(device_tensors_to_torch(out, mesh_device), device_tensors_to_torch(ref, mesh_device)))
            ttnn.deallocate(ref)
            print(f"[infra] D4 ag_dp_rows kernel traced replay {it} == eager ops bitwise: {ok}")
            if not ok:
                failures.append(f"trace replay {it}")
    finally:
        ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(out)
        ttnn.deallocate(x)
    assert not failures, failures
