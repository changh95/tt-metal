# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DESIGN-2 replica slots (``MOTIF3_MOE_REPLICAS=r4``, tt/replicas.py; logs/opt/phaseE/DESIGN2), device tests on the
(4, 8) TORUS_XY mesh:

* ``test_replicas_device_sparse_mm``: ``DualNocSparseMM`` with ``w_rep`` (12 native + 4 replica slots) is bitwise the
  16-expert op on the concatenated weights, and with the replica slots inactive bitwise the 12-expert op;
* ``test_replicas_device_router``: the replica-mode router tail on synthetic and skewed scores, M = 32 / 64, with and
  without a lane mask: ``idx`` = the plain kernel's, the on-device assignment (where / load) = ``replicas.assign`` on
  every chip, every chip's ``w_loc`` slot = the plain kernel's weight of that expert (bitwise) when this chip computes
  it and the row is live, else +0; every live (row, expert) pair is computed by exactly one chip;
* ``test_replicas_device_layer``: real layers from the serving TT cache: the replica weights are the cached bytes
  (bitwise vs the home chip's native expert), the 16-slot PolyNorm constants, then ``forward_decode`` r4 vs off on real
  router inputs (tolerance gate: PCC and max |d| per row), r4 with every replica code removed is bitwise off, r4 is
  deterministic (rerun, trace replay == eager); T64 rows vs two T32 steps (reported: the assignment depends on the
  step, so MTP launches keep the replicas off, which ``resolve_moe_replicas`` enforces).

    scripts/devrun.sh -t 900 -n optE_D2_unit -- python -m pytest -s -p no:cacheprovider --timeout=0 -rA \
        models/demos/motif3/tests/unit/test_moe_replicas.py
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.test_moe import (
    E,
    HF_META,
    MESH_PARAMS,
    _Capture,
    _free,
    load_real_inputs,
    moe_from_tt_cache,
    pcc,
    spread_tokens,
    t64_cfg,
    upload_lanes,
    upload_replicated,
    upload_rows16,
    rows16_to_halves,
)
from models.demos.motif3.tt import replicas as RP

K = 8
OUT = Path(os.environ.get("MOTIF3_D2_OUT", "/home/ttuser/hchang/experiments/motif-3/logs/opt/phaseE/DESIGN2/unit"))
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG


def _save(name, res):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(res, indent=1, default=str))


def _cfg(mesh_device, **kw):
    from models.demos.motif3.tt.model_config import MotifTTConfig

    return MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device, **kw)


def _fabric(mesh_device, tag):
    from models.demos.motif3.tt.ccl import log_fabric

    fab = log_fabric(mesh_device, tag)
    assert str(fab.get("committed")) == "TORUS_XY", fab
    return fab


def _shards(t):
    return [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]


def _per_chip(x: torch.Tensor, mesh_device, dtype, layout=ttnn.TILE_LAYOUT, mc=DRAM):
    """``[R, C, ...]`` -> chip (r, c) holds ``x[r, c]`` (leading dim 1)."""
    R, C = tuple(mesh_device.shape)
    mapper = ttnn.create_mesh_mapper(
        mesh_device, ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], ttnn.MeshShape(R, C)))
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh_device, memory_config=mc, mesh_mapper=mapper)


def _table(slots, mesh_device, codes_off=False):
    t = torch.zeros(4, 8, 1, RP.TABLE_WORDS, dtype=torch.int32)
    for k in range(32):
        t[k // 8, k % 8, 0] = torch.tensor(RP.chip_table(slots, k, flex=not codes_off), dtype=torch.int32)
    return _per_chip(t, mesh_device, ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


# =====================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_replicas_device_sparse_mm(mesh_device, device_params):
    from models.demos.motif3.tt.kernels.moe_sparse_mm import DualNocSparseMM

    _fabric(mesh_device, "d2_sparse_mm")
    torch.manual_seed(0)
    rng = random.Random(0)
    REP = ttnn.ReplicateTensorToMesh(mesh_device)

    def dev(t, dtype, mc=L1):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=mc,
                               mesh_mapper=REP)

    wg16 = (torch.randn(1, 16, 4096, 2560) * 0.02).to(torch.bfloat16)
    wd16 = (torch.randn(1, 16, 1280, 4096) * 0.02).to(torch.bfloat16)
    W = {k: dev(v, ttnn.bfloat8_b, DRAM) for k, v in (("g16", wg16), ("d16", wd16), ("g12", wg16[:, :12]),
                                                         ("d12", wd16[:, :12]), ("g4", wg16[:, 12:]),
                                                         ("d4", wd16[:, 12:]))}
    ops = {
        "rep": (DualNocSparseMM(mesh_device, W["g12"], kind="gate_up", out_dtype=ttnn.float32, w_rep=W["g4"]),
                DualNocSparseMM(mesh_device, W["d12"], kind="down", out_dtype=ttnn.bfloat16, w_rep=W["d4"])),
        "e16": (DualNocSparseMM(mesh_device, W["g16"], kind="gate_up", out_dtype=ttnn.float32),
                DualNocSparseMM(mesh_device, W["d16"], kind="down", out_dtype=ttnn.bfloat16)),
        "e12": (DualNocSparseMM(mesh_device, W["g12"], kind="gate_up", out_dtype=ttnn.float32),
                DualNocSparseMM(mesh_device, W["d12"], kind="down", out_dtype=ttnn.bfloat16)),
    }
    res, fails = {}, []
    for M in (32, 64):
        x = dev(torch.randn(1, 1, M, 4096).to(torch.bfloat16), ttnn.bfloat16)
        h16 = (torch.randn(1, 16, M, 1280) * 0.5).to(torch.bfloat16)
        hs = {"16": dev(h16, ttnn.bfloat16), "12": dev(h16[:, :12].contiguous(), ttnn.bfloat16)}
        for case in ("all", "none", "rep_only", "nat_only", "mix", "k0"):
            act = {"all": list(range(16)), "none": [], "rep_only": [12, 13, 14, 15], "nat_only": [0, 3, 7, 11],
                   "mix": [1, 5, 12, 15], "k0": []}[case]
            w = torch.zeros(1, 16, M, 1)
            for e in act:
                for r in range(M):
                    if rng.random() < 0.4 or r == 0:
                        w[0, e, r, 0] = rng.uniform(0.01, 0.4)
            wl16, wl12 = dev(w, ttnn.float32), dev(w[:, :12].contiguous(), ttnn.float32)
            out = {}
            for tag, (gu, dn), wl, h in (("rep", ops["rep"], wl16, hs["16"]), ("e16", ops["e16"], wl16, hs["16"]),
                                         ("e12", ops["e12"], wl12, hs["12"])):
                if tag == "e12" and any(e >= 12 for e in act):
                    continue
                g, sp = gu.gate_up_routed(x, wl, memory_config=L1)
                p = dn.down_sum(h, sp, memory_config=L1)
                out[tag] = (_shards(g)[0].float(), _shards(p)[0].float(), _shards(p)[31].float())
                _free([g, sp, p])
            ok = torch.equal(out["rep"][0], out["e16"][0]) and torch.equal(out["rep"][1], out["e16"][1]) \
                and torch.equal(out["rep"][1], out["rep"][2])
            if "e12" in out:
                ok = ok and torch.equal(out["rep"][0][:, :12], out["e12"][0]) and torch.equal(out["rep"][1], out["e12"][1])
            res[f"M{M}.{case}"] = bool(ok)
            print(f"[d2] sparse_mm w_rep M{M} {case}: bitwise {ok}")
            if not ok:
                fails.append(f"M{M} {case}")
            _free([wl16, wl12])
        _free([x, *hs.values()])
    for g, d in ops.values():
        g.deallocate()
        d.deallocate()
    _free(list(W.values()))
    _save("sparse_mm.json", res)
    assert not fails, fails


# =====================================================================================================================
def _scores(M, mode, rng_t):
    if mode == "uniform":
        return torch.rand(M, E, generator=rng_t)
    if mode == "skew":  # rows prefer the first chips' experts: heavy loads, many reassignments
        s = torch.rand(M, E, generator=rng_t) * 0.3
        s[:, :48] += 0.5 * torch.rand(M, 48, generator=rng_t)
        return s
    if mode == "same":  # every row the same scores: 8 active experts
        return torch.rand(1, E, generator=rng_t).expand(M, E).contiguous()
    raise ValueError(mode)


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_replicas_device_router(mesh_device, device_params):
    from models.demos.motif3.tt import weights as Wt
    from models.demos.motif3.tt.kernels import router_topk as RT

    _fabric(mesh_device, "d2_router")
    cfg = _cfg(mesh_device)
    ids = Wt.as_tensor(Wt.local_expert_ids(cfg), mesh_device=mesh_device, cfg=cfg, dtype=ttnn.float32,
                       cache_name=None, dp_dim=0, tp_dim=1)
    g = torch.Generator().manual_seed(0)
    bias_h = (torch.rand(E, generator=g) - 0.5) * 0.05
    bias = upload_replicated(bias_h.reshape(1, 1, 1, E), mesh_device, dtype=ttnn.float32)
    plan = RP.load_plan(RP.default_plan_path())
    res, fails = {}, []
    for L in ("2", "30"):
        slots = plan["layers"][L]
        code = RP.rep_codes(slots)
        table = _table(slots, mesh_device)
        k_plain = RT.FusedRouterTopK(mesh_device, bias, ids, top_k=K)
        k_rep = RT.FusedRouterTopK(mesh_device, bias, ids, top_k=K, replica_table=table, n_rep=4)
        for M in (32, 64):
            for mode in ("uniform", "skew", "same"):
                for lane_kind in ("none", "mixed", "dead"):
                    s = _scores(M, mode, g)
                    st = upload_replicated(s.reshape(1, 1, M, E), mesh_device, dtype=ttnn.float32)
                    live = [True] * M
                    lane = None
                    if lane_kind != "none":
                        live = [lane_kind == "mixed" and (r % 3 != 1) for r in range(M)]
                        lane = upload_replicated(torch.tensor([1.0 if v else 0.0 for v in live]).reshape(1, 1, M, 1),
                                                 mesh_device, dtype=ttnn.float32)
                    w0, i0 = k_plain(st, scale=2.0, memory_config=L1, want_idx=True)
                    w1, i1, a1 = k_rep(st, scale=2.0, memory_config=L1, want_idx=True, lane_mask=lane,
                                       want_assign=True)
                    I0 = _shards(i0)[0].reshape(M, K).long()
                    ok_idx = all(torch.equal(x.reshape(M, K).long(), I0) for x in _shards(i1))
                    # dense reference weights from the plain kernel (home chip of each expert)
                    D = torch.zeros(M, E)
                    for k, sh in enumerate(_shards(w0)):
                        D[:, 12 * k:12 * k + 12] = sh.float().reshape(12, M).t()
                    act = RP.active_from_rows(I0.tolist(), live)
                    where, load = RP.assign(act, code)
                    exp_assign = RP.assign_words(where, load, slots)
                    ok_assign, ok_w, cover = True, True, torch.zeros(M, E, dtype=torch.long)
                    for k, (ash, wsh) in enumerate(zip(_shards(a1), _shards(w1))):
                        a = ash.reshape(-1).long().tolist()
                        ok_assign &= a == exp_assign
                        got = wsh.float().reshape(16, M).t()
                        exp = torch.zeros(M, 16)
                        for sl, e in enumerate(RP.chip_slots(slots, k)):
                            if where.get(e) == k:
                                for t in range(M):
                                    if live[t]:
                                        exp[t, sl] = D[t, e]
                        for sl, e in enumerate(RP.chip_slots(slots, k)):  # coverage from the device's w_loc
                            cover[:, e] += (got[:, sl] != 0).long()
                        ok_w &= torch.equal(got.view(torch.int32), exp.view(torch.int32))
                    want_cover = torch.zeros(M, E, dtype=torch.long)
                    for t in range(M):
                        if live[t]:
                            want_cover[t, I0[t]] = 1
                    ok_cover = torch.equal(cover, want_cover)
                    key = f"L{L}.M{M}.{mode}.{lane_kind}"
                    base_max = max(sum(1 for e in range(12 * c, 12 * c + 12) if act[e]) for c in range(32))
                    res[key] = dict(idx=ok_idx, assign=ok_assign, w=ok_w, cover=ok_cover, ep32_max=base_max,
                                    rep_max=max(load), n_active=sum(act))
                    print(f"[d2] router {key}: idx {ok_idx} assign {ok_assign} w_loc {ok_w} cover {ok_cover} "
                          f"(max load {base_max} -> {max(load)}, {sum(act)} active)")
                    if not (ok_idx and ok_assign and ok_w and ok_cover):
                        fails.append(key)
                    _free([st, lane, w0, i0, w1, i1, a1])
        k_plain.deallocate()
        k_rep.deallocate()
        _free(table)
    _free([ids, bias])
    _save("router.json", res)
    assert not fails, fails


# =====================================================================================================================
def _rows(t, mesh_device):
    from models.demos.motif3.tt.ccl import device_tensors_to_torch

    return device_tensors_to_torch(t, mesh_device).float()


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_replicas_device_layer(mesh_device, device_params):
    from models.demos.motif3.tt.ccl import MotifCCL
    from models.demos.motif3.tt.moe import resolve_moe_replicas

    _fabric(mesh_device, "d2_layer")
    cfg = _cfg(mesh_device)
    ccl = MotifCCL(mesh_device, cfg)
    data = load_real_inputs()
    res, fails = {}, []
    layers = [int(v) for v in os.environ.get("MOTIF3_D2_LAYERS", "8,32").split(",")]
    for layer in layers:
        t0 = time.time()
        off = moe_from_tt_cache(mesh_device, cfg, ccl, layer, moe_replicas="off")
        rep = moe_from_tt_cache(mesh_device, cfg, ccl, layer, moe_replicas="r4")
        res[f"L{layer}.build_s"] = round(time.time() - t0, 1)
        assert rep.moe_replicas == "r4" and off.moe_replicas == "off"
        slots = rep.rep_slots
        # replica weights == the cached native bytes (dequantized bitwise) of the home chip, chips 0 / 13 / 31
        ok_w = True
        for wn, wr in ((off.w_gate_up, rep.w_rep_gate_up), (off.w_down, rep.w_rep_down)):
            nat = ttnn.get_device_tensors(wn)
            rp = ttnn.get_device_tensors(wr)
            cache = {}
            for k in (0, 13, 31):
                r = ttnn.to_torch(rp[k])
                for j, e in enumerate(slots[k]):
                    h = e // 12
                    if h not in cache:
                        cache[h] = ttnn.to_torch(nat[h])
                    ok_w &= torch.equal(r[0, j], cache[h][0, e % 12])
            cache.clear()
        ok_c = True
        for name, t in rep._rep_consts.items():
            src = {"b": off.pn_consts.c["fp32"]["b"], "D": off.pn_consts.D, "E": off.pn_consts.E}[name]
            ns, rs = _shards(src), _shards(t)
            for k in range(32):
                exp = torch.cat([ns[k]] + [ns[e // 12][:, e % 12:e % 12 + 1] for e in slots[k]], dim=1)
                ok_c &= torch.equal(rs[k], exp)
        res[f"L{layer}.replica_bytes"] = bool(ok_w)
        res[f"L{layer}.consts16"] = bool(ok_c)
        print(f"[d2] L{layer}: replica weights == cached native experts {ok_w}, 16-slot PolyNorm consts {ok_c}")
        if not (ok_w and ok_c):
            fails.append(f"L{layer} weights {ok_w} consts {ok_c}")

        xs = data["layers"][layer]["x"]
        n = xs.shape[0]
        stats = []
        for bi, s0 in enumerate((0, 7, 13)):
            x32 = xs[spread_tokens(n - s0, 32) + s0]
            live = [r % 5 != 3 for r in range(32)] if bi == 1 else [True] * 32
            lane = upload_replicated(torch.tensor([1.0 if v else 0.0 for v in live]).reshape(1, 1, 32, 1),
                                     mesh_device, dtype=ttnn.float32)
            outs = {}
            for tag, m in (("off", off), ("r4", rep), ("r4b", rep)):
                x_tt = upload_lanes(x32, cfg, mesh_device)
                o = m.forward_decode(x_tt, lane_mask=lane)
                outs[tag] = _rows(o, mesh_device)
                _free([o, x_tt])
            a, b = outs["off"], outs["r4"]
            # gathered order: chip (dp, tp) rows = DP row dp's 8 lanes; compare only live lanes
            mask = torch.tensor(live).reshape(4, 8)
            sel = mask.reshape(4, 1, 1, 1, 8, 1).expand(a.shape) if a.dim() == 6 else None
            av, bv = (a[sel], b[sel]) if sel is not None else (a, b)
            p = pcc(av.flatten(), bv.flatten())
            scale = a.abs().amax(-1, keepdim=True).clamp_min(1e-30)
            rel = float((((a - b).abs() / scale)[sel] if sel is not None else (a - b).abs() / scale).max())
            det = bool(torch.equal(outs["r4"], outs["r4b"]))
            stats.append(dict(batch=bi, pcc=p, max_rel=rel, deterministic=det, bitwise_vs_off=bool(torch.equal(a, b))))
            print(f"[d2] L{layer} batch {bi}: r4 vs off PCC {p:.10f}, max |d|/row max {rel:.2e}, r4 rerun bitwise {det}")
            if p < 0.99999 or rel > 4 * 2.0 ** -7 or not det:
                fails.append(f"L{layer} batch {bi}: pcc {p} rel {rel} det {det}")
            _free(lane)
        res[f"L{layer}.r4_vs_off"] = stats
        # every replica code removed (same weights, same 16-slot programs): bitwise the plain path
        saved = rep.router_fused_rep.table
        t_off = _table(slots, mesh_device, codes_off=True)
        rep.router_fused_rep.table = t_off
        rep.router_fused_rep._desc.clear()
        x32 = xs[spread_tokens(n, 32)]
        o1 = _rows(off.forward_decode(upload_lanes(x32, cfg, mesh_device)), mesh_device)
        o2 = _rows(rep.forward_decode(upload_lanes(x32, cfg, mesh_device)), mesh_device)
        home_eq = bool(torch.equal(o1, o2))
        rep.router_fused_rep.table = saved
        rep.router_fused_rep._desc.clear()
        _free(t_off)
        res[f"L{layer}.codes_off_bitwise_vs_off"] = home_eq
        print(f"[d2] L{layer}: r4 with no replica codes == off bitwise {home_eq}")
        if not home_eq:
            fails.append(f"L{layer} codes-off not bitwise")
        # trace replay == eager (r4)
        x_tt = upload_lanes(xs[spread_tokens(n, 32)], cfg, mesh_device)
        _free(rep.forward_decode(x_tt))
        with _Capture(mesh_device) as cap:
            out_t = rep.forward_decode(x_tt)
        tr = []
        try:
            for s0 in (3, 5):
                ttnn.copy_host_to_device_tensor(upload_lanes(xs[spread_tokens(n - s0, 32) + s0], cfg, mesh_device,
                                                             device=False), x_tt)
                ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                got = _rows(out_t, mesh_device)
                o = rep.forward_decode(x_tt)
                tr.append(bool(torch.equal(got, _rows(o, mesh_device))))
                _free(o)
        finally:
            ttnn.release_trace(mesh_device, cap.tid)
            _free([out_t, x_tt])
        res[f"L{layer}.trace_eq_eager"] = tr
        print(f"[d2] L{layer}: r4 trace replay == eager {tr}")
        if not all(tr):
            fails.append(f"L{layer} trace {tr}")
        off.deallocate()
        rep.deallocate()

    # T64 (MTP verify rows) vs two T32 steps with replicas (explicit request on a T64 config); the config path is off
    import dataclasses

    cfg64 = t64_cfg(mesh_device)
    layer = layers[0]
    m64 = moe_from_tt_cache(mesh_device, cfg64, MotifCCL(mesh_device, cfg64), layer, moe_replicas="r4")
    mtp_off = resolve_moe_replicas(None, dataclasses.replace(cfg64, moe_replicas="r4"), module=m64) == "off"
    res["mtp_config_resolves_off"] = mtp_off
    if not mtp_off:
        fails.append("an MTP config (spec_tokens 1) with MOTIF3_MOE_REPLICAS=r4 did not resolve to off")
    xs = data["layers"][layer]["x"]
    x64 = xs[spread_tokens(xs.shape[0], 64)]
    halves = []
    for half in (x64[:32], x64[32:]):
        x_tt = upload_lanes(half, cfg64, mesh_device)
        halves.append(_rows(m64.forward_decode(x_tt), mesh_device))
        _free(x_tt)
    o = _rows(m64.forward_decode(upload_rows16(x64, cfg64, mesh_device)), mesh_device)
    anc, dr = rows16_to_halves(o, cfg64)
    t64_eq = bool(torch.equal(anc, halves[0])) and bool(torch.equal(dr, halves[1]))
    p64 = pcc(torch.cat([anc.flatten(), dr.flatten()]), torch.cat([halves[0].flatten(), halves[1].flatten()]))
    res["t64_rows_eq_t32_with_replicas"] = t64_eq
    res["t64_vs_t32_pcc"] = p64
    print(f"[d2] T64 rows == T32 rows with replicas: {t64_eq} (PCC {p64:.10f}); MTP launches keep replicas off")
    m64.deallocate()
    _save("layer.json", res)
    assert not fails, fails
