# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase D DESIGN-3 stage 1: the dual-NoC all-core sparse expert matmul (``tt/kernels/moe_sparse_mm.py``,
``MOTIF3_DECODE_EXPERT_MM=dualnoc``) against stock ``ttnn.sparse_matmul`` with the production G6 configs (HiFi4 + fp32
dest acc), random bfp8 weights in the production layouts ``[1, 12, 4096, 2560]`` / ``[1, 12, 1280, 4096]``.

* ``test_dualnoc_bitwise_vs_stock``: ``torch.equal`` on all 12 output slices (the inactive ones must be exactly 0)
  for k = 0..12 prefix masks, scattered masks and all-ones, M = 32 and 64, gate_up with fp32 and bf16 outputs, down
  with bf16; every replica identical; ``MOTIF3_SMM_DET`` (default 20) repeats per mask bitwise (determinism).
* ``test_dualnoc_traced_soak``: one captured trace (gate_up + down, the sparse path's pair) replayed
  ``MOTIF3_SMM_SOAK`` (default 1000) times with a random mask (k = 0 included) written into the sparsity tensor before
  each replay; every ``MOTIF3_SMM_SOAK_CHECK``-th replay (default 25) compared bitwise with eager stock. Run it under
  ``scripts/devrun.sh -t 900`` (and once with the watcher on).
* Phase E D3 stage 2 (``MOTIF3_DECODE_EXPERT_MM=fused``): ``test_fused_bitwise_vs_stage1`` (gate_up_routed / down_sum
  against the stage-1 chain decode_sparsity -> dualnoc -> fast_reduce_nc and against stock, h multicast on and off) and
  ``test_fused_traced_soak`` (every replay checked).

    scripts/devrun.sh -t 900 -n smm -- python -m pytest -s models/demos/motif3/tests/unit/test_moe_sparse_mm.py
"""

import os
import random
import time

import pytest
import torch

import ttnn

E = 12
SHAPES = {"gate_up": (4096, 2560), "down": (1280, 4096)}


def _device_params():
    from models.demos.motif3.tt.model_config import device_params

    return device_params("FABRIC_2D_TORUS_XY", 64 * 1024 * 1024)


MESH_PARAMS = [pytest.param((4, 8), _device_params(), id="4x8-torus2d")]


def _check_fabric(mesh_device, tag):
    from models.demos.motif3.tt.ccl import log_fabric

    fab = log_fabric(mesh_device, tag)
    assert str(fab.get("committed")) == "TORUS_XY" and not fab.get("degraded"), fab


def _masks():
    m = {"k0": [], "all": list(range(E))}
    for k in range(1, E):
        m[f"p{k}"] = list(range(k))
    m.update({"s1_11": [11], "s1_5": [5], "s2_hi": [10, 11], "s3": [0, 5, 11], "s6": [1, 3, 4, 7, 8, 10],
              "s9": [0, 1, 2, 4, 5, 7, 8, 9, 11], "s4_odd": [1, 3, 9, 11]})
    return m


class _Env:
    def __init__(self, mesh_device):
        self.mesh = mesh_device
        self.rep = ttnn.ReplicateTensorToMesh(mesh_device)
        from models.demos.motif3.tt.model_config import make_compute_kernel_config

        self.ckc = make_compute_kernel_config("HiFi4", True)

    def dev(self, t, dtype, mc=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=self.mesh, memory_config=mc, mesh_mapper=self.rep)

    def sp_host(self, active):
        s = torch.zeros(1, 1, 1, E)
        for e in active:
            s[..., e] = 0.37 + 0.01 * e
        return s.to(torch.bfloat16)

    def sp(self, active):
        return self.dev(self.sp_host(active), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT)

    def stock(self, kind, a, w, sp, M, dtype):
        from models.demos.motif3.tt.model_config import experts_down_pc, experts_gate_up_pc

        pc = (experts_gate_up_pc if kind == "gate_up" else experts_down_pc)(m_tiles=M // 32)
        y = ttnn.sparse_matmul(a, w, sparsity=sp, program_config=pc, nnz=None, is_input_a_sparse=(kind == "down"),
                               is_input_b_sparse=True, memory_config=ttnn.L1_MEMORY_CONFIG,
                               compute_kernel_config=self.ckc, dtype=dtype)
        return ttnn.reshape(y, (1, E, M, SHAPES[kind][1]))

    @staticmethod
    def host(t, i=0):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[i]).float()


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_dualnoc_bitwise_vs_stock(mesh_device, device_params):
    from models.demos.motif3.tt.kernels.moe_sparse_mm import DualNocSparseMM

    _check_fabric(mesh_device, "smm_bitwise")
    env = _Env(mesh_device)
    n_det = int(os.environ.get("MOTIF3_SMM_DET", "20"))
    torch.manual_seed(0)
    failures, n_cases = [], 0
    for kind, dtypes in (("gate_up", (ttnn.float32, ttnn.bfloat16)), ("down", (ttnn.bfloat16,))):
        K, N = SHAPES[kind]
        w = env.dev((torch.randn(1, E, K, N) * 0.02).to(torch.bfloat16), ttnn.bfloat8_b)
        for dtype in dtypes:
            op = DualNocSparseMM(mesh_device, w, kind=kind, out_dtype=dtype)
            for M in (32, 64):
                shape = (1, 1, M, K) if kind == "gate_up" else (1, E, M, K)
                a = env.dev(torch.randn(*shape).to(torch.bfloat16), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG)
                for name, act in _masks().items():
                    sp = env.sp(act)
                    junk = env.dev(torch.full((1, E, M, N), 7.0), dtype, ttnn.L1_MEMORY_CONFIG)
                    ttnn.deallocate(junk)  # a missing zero-fill would read these 7s back
                    got = op(a, sp, memory_config=ttnn.L1_MEMORY_CONFIG)
                    g = env.host(got)
                    reps_eq = all(torch.equal(g, env.host(got, i)) for i in (7, 19, 31))
                    ttnn.deallocate(got)
                    ref = env.stock(kind, a, w, sp, M, dtype)
                    r = env.host(ref).reshape(g.shape)
                    ttnn.deallocate(ref)
                    tag = f"{kind} {dtype} M={M} {name}"
                    n_cases += 1
                    per = [bool(torch.equal(g[0, e], r[0, e])) for e in range(E)]
                    if not all(per):
                        failures.append(f"{tag}: slices differ {[e for e in range(E) if not per[e]]}, max |d| "
                                        f"{float((g - r).abs().max()):.3g}")
                    if any(bool((g[0, e] != 0).any()) for e in range(E) if e not in act):
                        failures.append(f"{tag}: an inactive slice is not zero")
                    if act and not all(bool((g[0, e] != 0).any()) for e in act):
                        failures.append(f"{tag}: an active slice is all zero")
                    if not reps_eq:
                        failures.append(f"{tag}: replicas differ")
                    if n_det and name in ("all", "p1", "p6", "s3", "k0", "p9"):
                        for _ in range(n_det):
                            o2 = op(a, sp, memory_config=ttnn.L1_MEMORY_CONFIG)
                            same = torch.equal(env.host(o2), g)
                            ttnn.deallocate(o2)
                            if not same:
                                failures.append(f"{tag}: not deterministic")
                                break
                    ttnn.deallocate(sp)
                ttnn.deallocate(a)
            op.deallocate()
        ttnn.deallocate(w)
    print(f"[smm] bitwise vs stock: {n_cases} cases, {len(failures)} failures")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_dualnoc_traced_soak(mesh_device, device_params):
    from models.demos.motif3.tt.kernels.moe_sparse_mm import DualNocSparseMM

    _check_fabric(mesh_device, "smm_soak")
    env = _Env(mesh_device)
    n = int(os.environ.get("MOTIF3_SMM_SOAK", "1000"))
    every = int(os.environ.get("MOTIF3_SMM_SOAK_CHECK", "25"))
    torch.manual_seed(1)
    rng = random.Random(1)
    L1 = ttnn.L1_MEMORY_CONFIG
    M = 32
    wg = env.dev((torch.randn(1, E, 4096, 2560) * 0.02).to(torch.bfloat16), ttnn.bfloat8_b)
    wd = env.dev((torch.randn(1, E, 1280, 4096) * 0.02).to(torch.bfloat16), ttnn.bfloat8_b)
    gu = DualNocSparseMM(mesh_device, wg, kind="gate_up", out_dtype=ttnn.float32)
    dn = DualNocSparseMM(mesh_device, wd, kind="down", out_dtype=ttnn.bfloat16)
    x = env.dev(torch.randn(1, 1, M, 4096).to(torch.bfloat16), ttnn.bfloat16, L1)
    h = env.dev((torch.randn(1, E, M, 1280) * 0.5).to(torch.bfloat16), ttnn.bfloat16, L1)
    sp = env.sp(list(range(E)))

    def step():
        return gu(x, sp, memory_config=L1), dn(h, sp, memory_config=L1)

    for o in step():  # compile (eager) before the capture
        ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    og, od = step()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    failures, ks, t0 = [], [], time.time()
    for i in range(n):
        k = rng.choice([0, 0, 1, 2, 3, 5, 8, 9, 10, 12]) if i % 7 else rng.randint(0, E)
        act = sorted(rng.sample(range(E), k))
        ks.append(k)
        host_sp = ttnn.from_torch(env.sp_host(act), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                  mesh_mapper=env.rep)
        ttnn.copy_host_to_device_tensor(host_sp, sp)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        if i % every == 0 or i == n - 1:
            ttnn.synchronize_device(mesh_device)
            g, d = env.host(og), env.host(od)
            rg = env.stock("gate_up", x, wg, sp, M, ttnn.float32)
            rd = env.stock("down", h, wd, sp, M, ttnn.bfloat16)
            if not torch.equal(g, env.host(rg).reshape(g.shape)) or not torch.equal(d, env.host(rd).reshape(d.shape)):
                failures.append(f"replay {i} (k={k}, {act}): traced dualnoc != eager stock")
            ttnn.deallocate(rg)
            ttnn.deallocate(rd)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, tid)
    print(f"[smm] soak: {n} traced replays in {time.time() - t0:.1f} s, k histogram "
          f"{ {k: ks.count(k) for k in sorted(set(ks))} }, {len(failures)} failures")
    for t in (og, od, x, h, sp, wg, wd):
        ttnn.deallocate(t)
    assert not failures, "\n".join(failures[:20])


# =====================================================================================================================
# Phase E D3 stage 2 (MOTIF3_DECODE_EXPERT_MM=fused): gate_up_routed (sparsity from w_loc in the kernel) and down_sum
# (expert sum in the kernel, h multicast) against the stage-1 chain the decode ran before: decode_sparsity's 4 ops ->
# dualnoc gate_up / down -> fast_reduce_nc (and against stock ttnn.sparse_matmul + fast_reduce_nc).
# =====================================================================================================================
def _wloc_host(act, M, rng, p=0.35):
    w = torch.zeros(1, E, M, 1)
    for e in act:
        rows = [r for r in range(M) if rng.random() < p] or [rng.randrange(M)]
        for r in rows:
            w[0, e, r, 0] = rng.uniform(0.005, 0.4)
    return w


def _sparsity_s1(w_loc):  # MotifMoE.decode_sparsity
    L1 = ttnn.L1_MEMORY_CONFIG
    m = ttnn.max(w_loc, dim=2, keepdim=True, memory_config=L1)
    b = ttnn.typecast(m, ttnn.bfloat16, memory_config=L1)
    ttnn.deallocate(m)
    r = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT, memory_config=L1)
    ttnn.deallocate(b)
    return ttnn.reshape(r, (1, 1, 1, E))


def _reduce_s1(env, y, pdt):  # MotifMoE.reduce_experts
    out = None
    if y.dtype != pdt:
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, int(y.shape[-2]), int(y.shape[-1])]), pdt,
                                             ttnn.TILE_LAYOUT, env.mesh, ttnn.L1_MEMORY_CONFIG)
    return ttnn.experimental.fast_reduce_nc(y, dims=[1], output=out, memory_config=ttnn.L1_MEMORY_CONFIG,
                                            compute_kernel_config=env.ckc)


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_fused_bitwise_vs_stage1(mesh_device, device_params):
    """``torch.equal`` of stage 2's gu and part against stage 1 (dualnoc + decode_sparsity + fast_reduce_nc) and stock,
    the kernel's active set == the stage-1 sparsity == the host mirror, all replicas equal; M = 32 / 64, gate_up fp32 /
    bf16, part bf16 / fp32, prefix / scattered / k = 0 / all masks with sparse and dense row patterns; h multicast on
    and off; ``MOTIF3_SMM_DET`` repeats bitwise."""
    import random

    from models.demos.motif3.tt.kernels.moe_sparse_mm import DualNocSparseMM, sparsity_of

    _check_fabric(mesh_device, "smm_fused_bitwise")
    env = _Env(mesh_device)
    L1 = ttnn.L1_MEMORY_CONFIG
    n_det = int(os.environ.get("MOTIF3_SMM_DET", "20"))
    torch.manual_seed(2)
    rng = random.Random(2)
    wg = env.dev((torch.randn(1, E, 4096, 2560) * 0.02).to(torch.bfloat16), ttnn.bfloat8_b)
    wd = env.dev((torch.randn(1, E, 1280, 4096) * 0.02).to(torch.bfloat16), ttnn.bfloat8_b)
    gus = {dt: DualNocSparseMM(mesh_device, wg, kind="gate_up", out_dtype=dt) for dt in (ttnn.float32, ttnn.bfloat16)}
    dns = {}
    for mc_on in (True, False):
        dns[mc_on] = DualNocSparseMM(mesh_device, wd, kind="down", out_dtype=ttnn.bfloat16)
        dns[mc_on].h_mcast = mc_on
    failures, n_cases = [], 0
    for M in (32, 64):
        x = env.dev(torch.randn(1, 1, M, 4096).to(torch.bfloat16), ttnn.bfloat16, L1)
        h = env.dev((torch.randn(1, E, M, 1280) * 0.5).to(torch.bfloat16), ttnn.bfloat16, L1)
        for gdt, pdt in ((ttnn.float32, ttnn.bfloat16), (ttnn.float32, ttnn.float32), (ttnn.bfloat16, ttnn.bfloat16)):
            for name, act in _masks().items():
                for dense in (False, True):
                    if dense and name not in ("all", "p3", "s6", "k0", "p11"):
                        continue
                    tag = f"M={M} gu={gdt} part={pdt} {name}{' dense' if dense else ''}"
                    wl_h = _wloc_host(act, M, rng, p=1.0 if dense else 0.35)
                    wl = env.dev(wl_h, ttnn.float32, L1)
                    sp1 = _sparsity_s1(wl)
                    g1 = env.host(gus[gdt](x, sp1, memory_config=L1))
                    y1 = dns[True](h, sp1, memory_config=L1)
                    p1t = _reduce_s1(env, y1, pdt)
                    ttnn.deallocate(y1)
                    p1 = env.host(p1t)
                    ttnn.deallocate(p1t)
                    ys = env.stock("down", h, wd, sp1, M, ttnn.bfloat16)
                    pst = _reduce_s1(env, ys, pdt)
                    ttnn.deallocate(ys)
                    ps = env.host(pst)
                    ttnn.deallocate(pst)
                    gst = env.stock("gate_up", x, wg, sp1, M, gdt)
                    gs = env.host(gst).reshape(g1.shape)
                    ttnn.deallocate(gst)
                    for mc_on in (True, False):
                        junk = env.dev(torch.full((1, E, M, 4096), 7.0), ttnn.bfloat16, L1)
                        ttnn.deallocate(junk)  # a missing write would read these 7s back
                        g2t, s2 = gus[gdt].gate_up_routed(x, wl, memory_config=L1)
                        p2t = dns[mc_on].down_sum(h, s2, part_dtype=pdt, memory_config=L1)
                        g2, p2 = env.host(g2t), env.host(p2t)
                        st = env.host(s2).reshape(-1)
                        n_cases += 1
                        t = f"{tag} h_mcast={mc_on}"
                        act_k = [e for e in range(E) if float(st[e]) != 0.0]
                        if act_k != sorted(act) or act_k != sparsity_of(wl_h) or bool((st[E:] != 0).any()):
                            failures.append(f"{t}: active set {act_k} != {sorted(act)}")
                        if not torch.equal(g2, g1) or not torch.equal(g2, gs):
                            failures.append(f"{t}: gu differs (vs s1 {float((g2 - g1).abs().max()):.3g})")
                        if not torch.equal(p2, p1) or not torch.equal(p2, ps):
                            failures.append(f"{t}: part differs (vs s1 {float((p2 - p1).abs().max()):.3g}, "
                                            f"vs stock {float((p2 - ps).abs().max()):.3g})")
                        if not all(torch.equal(p2, env.host(p2t, i)) and torch.equal(g2, env.host(g2t, i))
                                   for i in (7, 19, 31)):
                            failures.append(f"{t}: replicas differ")
                        if n_det and mc_on and not dense and name in ("all", "p1", "s3", "k0", "s9"):
                            for _ in range(n_det):
                                a2, b2 = gus[gdt].gate_up_routed(x, wl, memory_config=L1)
                                c2 = dns[mc_on].down_sum(h, b2, part_dtype=pdt, memory_config=L1)
                                same = torch.equal(env.host(a2), g2) and torch.equal(env.host(c2), p2)
                                for tt_ in (a2, b2, c2):
                                    ttnn.deallocate(tt_)
                                if not same:
                                    failures.append(f"{t}: not deterministic")
                                    break
                        for tt_ in (g2t, s2, p2t):
                            ttnn.deallocate(tt_)
                    ttnn.deallocate(sp1)
                    ttnn.deallocate(wl)
        ttnn.deallocate(x)
        ttnn.deallocate(h)
    for o in list(gus.values()) + list(dns.values()):
        o.deallocate()
    ttnn.deallocate(wg)
    ttnn.deallocate(wd)
    print(f"[smm] fused bitwise vs stage 1 / stock: {n_cases} cases, {len(failures)} failures")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_fused_traced_soak(mesh_device, device_params):
    """One captured trace of the stage-2 pair (gate_up_routed -> down_sum, M = 32) replayed ``MOTIF3_SMM_SOAK`` (default
    1000) times with fresh random routing weights (k = 0 included) written into ``w_loc`` before each replay; EVERY
    replay compared bitwise with the eager stage-1 chain (decode_sparsity -> dualnoc -> fast_reduce_nc). Run it under
    ``scripts/devrun.sh -t 900``."""
    import random

    from models.demos.motif3.tt.kernels.moe_sparse_mm import DualNocSparseMM

    _check_fabric(mesh_device, "smm_fused_soak")
    env = _Env(mesh_device)
    n = int(os.environ.get("MOTIF3_SMM_SOAK", "1000"))
    torch.manual_seed(3)
    rng = random.Random(3)
    L1 = ttnn.L1_MEMORY_CONFIG
    M = int(os.environ.get("MOTIF3_SMM_SOAK_M", "32"))
    wg = env.dev((torch.randn(1, E, 4096, 2560) * 0.02).to(torch.bfloat16), ttnn.bfloat8_b)
    wd = env.dev((torch.randn(1, E, 1280, 4096) * 0.02).to(torch.bfloat16), ttnn.bfloat8_b)
    gu = DualNocSparseMM(mesh_device, wg, kind="gate_up", out_dtype=ttnn.float32)
    dn = DualNocSparseMM(mesh_device, wd, kind="down", out_dtype=ttnn.bfloat16)
    x = env.dev(torch.randn(1, 1, M, 4096).to(torch.bfloat16), ttnn.bfloat16, L1)
    h = env.dev((torch.randn(1, E, M, 1280) * 0.5).to(torch.bfloat16), ttnn.bfloat16, L1)
    wl = env.dev(_wloc_host(list(range(E)), M, rng), ttnn.float32, L1)

    def step():
        g, s = gu.gate_up_routed(x, wl, memory_config=L1)
        p = dn.down_sum(h, s, memory_config=L1)
        ttnn.deallocate(s)
        return g, p

    for o in step():  # compile (eager) before the capture
        ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    og, op_ = step()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    failures, ks, t0 = [], [], time.time()
    for i in range(n):
        k = rng.choice([0, 0, 1, 2, 3, 5, 8, 9, 10, 12]) if i % 7 else rng.randint(0, E)
        act = sorted(rng.sample(range(E), k))
        ks.append(k)
        host_w = ttnn.from_torch(_wloc_host(act, M, rng, p=rng.choice([0.1, 0.35, 1.0])), dtype=ttnn.float32,
                                 layout=ttnn.TILE_LAYOUT, mesh_mapper=env.rep)
        ttnn.copy_host_to_device_tensor(host_w, wl)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        g, p = env.host(og), env.host(op_)
        sp1 = _sparsity_s1(wl)
        rg = gu(x, sp1, memory_config=L1)
        ry = dn(h, sp1, memory_config=L1)
        rp = _reduce_s1(env, ry, ttnn.bfloat16)
        if not torch.equal(g, env.host(rg)) or not torch.equal(p, env.host(rp)) or \
                not torch.equal(p, env.host(op_, 31)):
            failures.append(f"replay {i} (k={k}, {act}): traced fused != eager stage 1")
        for t in (sp1, rg, ry, rp):
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, tid)
    print(f"[smm] fused soak: {n} traced replays (every one checked) in {time.time() - t0:.1f} s, k histogram "
          f"{ {k: ks.count(k) for k in sorted(set(ks))} }, {len(failures)} failures")
    for t in (og, op_, x, h, wl, wg, wd):
        ttnn.deallocate(t)
    gu.deallocate()
    dn.deallocate()
    assert not failures, "\n".join(failures[:20])
