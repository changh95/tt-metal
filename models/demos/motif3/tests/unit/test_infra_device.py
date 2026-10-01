# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests of the shared infra on the BH Galaxy (run ONLY through scripts/devrun.sh):

    scripts/devrun.sh -t 1200 -n infra_device -- \
        pytest models/demos/motif3/tests/unit/test_infra_device.py -x

* ``test_ccl_payloads_eager_and_trace`` (4x8; FABRIC_2D_TORUS_XY and the FABRIC_1D_RING fallback): every Motif
  draft-1 collective through ``MotifCCL`` with per-chip distinct inputs, checked against torch; replicas bitwise
  identical along the reduced axis; then the decode chain AR(tp) -> AG(dp) -> AR(dp) -> partition(dp) captured
  in a trace and replayed with new inputs. Eager / traced latencies are printed.
* ``test_mesh_mappers_and_cache`` (4x8): weights.as_tensor role mappings (replicate / tp / dp / 2D / EP)
  reassemble per chip exactly like ``weights.shard_for_device``; cache files are written, then reloaded without
  touching the torch source.
* ``test_rope_device`` (4x8): MotifRope per-lane decode gather (both layouts) and prefill tables vs the host
  tables; composite and rotary_embedding_hf application vs torch.
* ``test_roles_on_8x4`` (8x4): the TP axis is detected as cluster_axis 0 and the role names follow it.
"""

import os
import shutil
import time

import pytest
import torch

import ttnn
from models.demos.motif3.tt import weights as W
from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, replicas_identical
from models.demos.motif3.tt.model_config import DEFAULT_HF_META_DIR, DEFAULT_TT_CACHE_ROOT, MotifTTConfig, device_params
from models.demos.motif3.tt.rope import MotifRope, apply_rope_torch, positions_to_rot_idxs

HF_META = str(DEFAULT_HF_META_DIR)
TRACE = 32 * 1024 * 1024


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
    print(f"\n[infra] fabric={device_params['fabric_config']} {cfg.describe()}")

    cases = [
        # name, op, local shape, dtype, layout, exact
        ("AG(dp) moe gather 8x4096 TILE", "ag_dp", (8, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
        ("AG(dp) moe gather 8x4096 RM", "ag_dp", (8, 4096), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, True),
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
            ref = _golden(op, host, cfg)
            prec = _check(name, _readback(out, mesh_device), ref, exact, dtype)
            if op in ("ar_dp", "ar_tp", "ar_dp_rsag"):
                assert replicas_identical(out, mesh_device, op[3:5], cfg.axes), f"{name}: replicas differ"
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
        g = ccl.ag_dp(a, 2)
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
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs = chain(x)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
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
            f"PASS decode chain AR(tp)+AG(dp)+AR(dp)+partition(dp): eager {eager_us:.1f} us, traced {trace_us:.1f} us/replay"
        )
    except Exception as e:
        failures.append(f"FAIL decode chain (eager/trace): {type(e).__name__}: {str(e)[:400]}")
        results.append(failures[-1])
    finally:
        print("[infra] " + "\n[infra] ".join(results))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", [MESH_4x8[0]], indirect=True)
def test_mesh_mappers_and_cache(mesh_device, device_params):
    root = DEFAULT_TT_CACHE_ROOT / f"_infra_test_{os.getpid()}"
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device, tt_cache_root=root)
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
