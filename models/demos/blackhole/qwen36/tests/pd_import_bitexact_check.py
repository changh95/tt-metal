# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-identity + footprint check of the D-side traced GDN import (pd_transfer.TracedGdnImporter) on synthetic
GDN layers with the served per-device geometry (TP=4: Nv=12 Nk=4 Dk=Dv=128 C=2560 K=4 B=32, fp32 recurrent state).

For every decode slot 0..B-1 a random snapshot (rec [n_dev, L, Nv, Dk, Dv] fp32, taps [n_dev, L, K, C] bf16) is
imported with the traced importer and the WHOLE rec_state / conv_states[m] / conv_hist_packed buffers of sampled
layers are read back and compared (torch.equal) with the host expectation: the previous contents with row `slot`
replaced by the snapshot (packed history via the layer's own _pack_head_tiles at parity slot & 1). After all slots
the full state of EVERY layer is read back and compared. Optionally (PD_IMPORT_REF_MODULE=<path to the pre-change
pd_transfer.py>) the old importer writes a few slots first and its rows are compared byte for byte with the new
importer's rows for the same snapshots; both paths' replay times are logged. Also checks that the [1,1,B,C] /
[B,Nv*4,32,32] reshapes are true views (same buffer address) and counts the device ops of one _body.

Run (half A):
  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 ARCH_NAME=blackhole \
  python_env/bin/python models/demos/blackhole/qwen36/tests/pd_import_bitexact_check.py
Env: PD_IMPORT_L (48 layers), PD_IMPORT_B (32 slots), PD_IMPORT_MESH (1x4), PD_IMPORT_REF_MODULE (old pd_transfer.py),
PD_IMPORT_REF_SLOTS ("0,1,5,30,31"), PD_IMPORT_SAMPLE_LAYERS ("0,mid,last").
"""

import collections
import importlib.util
import json
import os
import threading
import time
import types

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import pd_transfer
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet

_e = lambda n, d: int(os.environ.get(n) or d)
NV, NK, DK, DV, K, B = (
    _e("HIST_NV", 12),
    _e("HIST_NK", 4),
    _e("HIST_DK", 128),
    _e("HIST_DV", 128),
    4,
    _e("PD_IMPORT_B", 32),
)
C = _e("HIST_C", 2 * NK * DK + NV * DV)
L = _e("PD_IMPORT_L", 48)
MESH_SHAPE = tuple(int(v) for v in os.environ.get("PD_IMPORT_MESH", "1x4").lower().split("x"))
REF_MODULE = os.environ.get("PD_IMPORT_REF_MODULE", "")
REF_SLOTS = [int(s) for s in os.environ.get("PD_IMPORT_REF_SLOTS", "0,1,5,30,31").split(",") if s.strip()]


def _load_ref():
    if not REF_MODULE:
        return None
    spec = importlib.util.spec_from_file_location("pd_transfer_ref", REF_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dev(mesh, host, dtype, mapper="replicate"):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if mapper == "replicate" else ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _read(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


class _Host:
    """Host mirror of one layer's buffers, device-major."""

    def __init__(self, n_dev, rec, convs, hist):
        self.rec = rec  # [n_dev, B, Nv, Dk, Dv] fp32
        self.convs = convs  # K x [n_dev, B, C] bf16
        self.hist = hist  # [n_dev, B, Nv, 4, 32, 32] bf16
        self.n_dev = n_dev

    def write(self, dn, slot, rec_l, taps_l):
        """rec_l [n_dev, Nv, Dk, Dv], taps_l [n_dev, K, C]: what the import must leave in row `slot`."""
        self.rec[:, slot] = rec_l
        for m in range(K):
            self.convs[m][:, slot] = taps_l[:, m]
        for d in range(self.n_dev):
            self.hist[d, slot] = dn._pack_head_tiles([taps_l[d, j].reshape(-1) for j in range(K)], parity=slot & 1)


def _make_model(mesh, gen):
    n = mesh.get_num_devices()
    layers, hosts = [], []
    for li in range(L):
        # rec_state / conv_states are replicated in the model (each device holds its own [B, ...]); build them from
        # per-device random data via the shard mapper so every device row differs (dim 0 = device -> [B, ...] each)
        rec_h = torch.randn(n, B, NV, DK, DV, generator=gen, dtype=torch.float32)
        convs_h = [torch.randn(n, B, C, generator=gen).to(torch.bfloat16) for _ in range(K)]
        hist_h = torch.randn(n, B, NV, 4, 32, 32, generator=gen).to(torch.bfloat16)
        dn = types.SimpleNamespace(
            mesh=mesh,
            Nv=NV,
            Nk=NK,
            Dk=DK,
            Dv=DV,
            qkv_dim_tp=C,
            K=K,
            B=B,
            rec_state=_dev(mesh, rec_h.reshape(n * B, NV, DK, DV), ttnn.float32, "shard"),
            conv_states=[_dev(mesh, c.reshape(n, B, C), ttnn.bfloat16, "shard") for c in convs_h],
            conv_hist_packed=_dev(mesh, hist_h.reshape(n * B, NV, 4, 32, 32), ttnn.bfloat16, "shard"),
            _hist_packed_valid=True,
            _decode_fused_conv=True,
        )
        for name in ("_slice_along", "_write_index", "_pack_head_tiles"):
            setattr(dn, name, types.MethodType(getattr(TPGatedDeltaNet, name), dn))
        layers.append(types.SimpleNamespace(is_full_attention=False, attention=dn))
        hosts.append(_Host(n, rec_h.clone(), [c.clone() for c in convs_h], hist_h.clone()))
    model = types.SimpleNamespace(mesh_device=mesh, num_devices=n, layers=layers)
    return model, hosts


def _read_layer(mesh, dn, n):
    rec = _read(mesh, dn.rec_state).view(n, B, NV, DK, DV)
    convs = [_read(mesh, c).view(n, B, C) for c in dn.conv_states]
    hist = _read(mesh, dn.conv_hist_packed).view(n, B, NV, 4, 32, 32)
    return rec, convs, hist


def _compare_layer(mesh, dn, host, li, tag):
    rec, convs, hist = _read_layer(mesh, dn, host.n_dev)
    bad = []
    if not torch.equal(rec, host.rec):
        bad.append(("rec", [int(s) for s in range(B) if not torch.equal(rec[:, s], host.rec[:, s])]))
    for m in range(K):
        if not torch.equal(convs[m], host.convs[m]):
            bad.append((f"tap{m}", [int(s) for s in range(B) if not torch.equal(convs[m][:, s], host.convs[m][:, s])]))
    if not torch.equal(hist, host.hist):
        bad.append(("hist", [int(s) for s in range(B) if not torch.equal(hist[:, s], host.hist[:, s])]))
    if bad:
        logger.error(f"[check] {tag} layer {li}: MISMATCH {bad}")
    return len(bad)


def _op_count(mesh, fn):
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    fn()
    graph = ttnn.graph.end_graph_capture()
    ttnn.synchronize_device(mesh)
    if not isinstance(graph, (list, tuple)):
        graph = json.loads(graph)
    names = [n.get("params", {}).get("name", "") for n in graph if n.get("node_type") == "function_start"]
    # device launches: the prim-level names (ttnn::prim::* / *DeviceOperation); composite wrappers (ttnn::slice,
    # ttnn::fill_cache, ...) are the same launches seen one level up and are not counted
    dev = [x for x in names if x.endswith("Operation") or "prim::" in x]
    counts = collections.Counter(dev)
    return (
        len(dev),
        sorted(counts.items(), key=lambda kv: -kv[1])[:12],
        sorted(collections.Counter(names).items(), key=lambda kv: -kv[1])[:16],
    )


def main():
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(*MESH_SHAPE),
        l1_small_size=24576,
        trace_region_size=_e("PD_IMPORT_TRACE_REGION_MB", 768) * 2**20,
    )
    mesh.enable_program_cache()
    n = mesh.get_num_devices()
    gen = torch.Generator().manual_seed(7)
    logger.info(f"mesh {MESH_SHAPE} ({n} devices) L={L} B={B} Nv={NV} Nk={NK} C={C} K={K}")
    model, hosts = _make_model(mesh, gen)
    dns = [l.attention for l in model.layers]
    sample = sorted({0, L // 2, L - 1})
    snaps = [
        (
            torch.randn(n, L, NV, DK, DV, generator=gen, dtype=torch.float32),
            torch.randn(n, L, K, C, generator=gen).to(torch.bfloat16),
        )
        for _ in range(B)
    ]
    failures = 0
    ref_rows = {}  # slot -> {li: (rec row, [tap rows], hist row)} written by the OLD importer

    # ---- phase A: the pre-change importer on a few slots (reference bytes + its replay time) ----
    ref = _load_ref()
    if ref is not None:
        old = ref.TracedGdnImporter(model)
        for s in REF_SLOTS:
            rec, taps = snaps[s]
            t0 = time.perf_counter()
            old.import_slot(s, rec, taps)
            dt = 1e3 * (time.perf_counter() - t0)
            t0 = time.perf_counter()
            old.import_slot(s, rec, taps)  # warm (trace already captured)
            dt2 = 1e3 * (time.perf_counter() - t0)
            for li, dn in enumerate(dns):
                hosts[li].write(dn, s, rec[:, li], taps[:, li])
            for li in sample:
                failures += _compare_layer(mesh, dns[li], hosts[li], li, f"OLD slot {s}")
            ref_rows[s] = {}
            for li in sample:
                r, cv, h = _read_layer(mesh, dns[li], n)
                ref_rows[s][li] = (r[:, s].clone(), [c[:, s].clone() for c in cv], h[:, s].clone())
            logger.info(f"[old] slot {s}: first (capture) {dt:.0f} ms, warm import {dt2:.1f} ms")
        del old

    # ---- phase B: the new importer on every slot; host prep on a worker thread like the connector's pull ----
    imp = pd_transfer.get_traced_importer(model)
    # view checks: the in-place write targets must alias the decode buffers
    dn0 = dns[0]
    v_hist = ttnn.reshape(dn0.conv_hist_packed, (B, NV * 4, 32, 32))
    v_tap = ttnn.reshape(dn0.conv_states[0], (1, 1, B, C))
    same = v_hist.buffer_address() == dn0.conv_hist_packed.buffer_address() and (
        v_tap.buffer_address() == dn0.conv_states[0].buffer_address()
    )
    logger.info(
        f"[view] hist view addr match {v_hist.buffer_address() == dn0.conv_hist_packed.buffer_address()}, "
        f"tap view addr match {v_tap.buffer_address() == dn0.conv_states[0].buffer_address()}"
    )
    failures += 0 if same else 1
    n_ops, kinds, n_all = _op_count(mesh, lambda: imp._body(0))
    logger.info(f"[ops] new _body: {n_ops} device ops for {L} layers = {n_ops / L:.1f}/layer; by kind {kinds}")
    logger.info(f"[ops] all function_start names (top): {n_all}")
    # the op-count pass wrote the (zero) staging into slot 0: re-import slot 0's data below anyway
    times = []
    for s in range(B):
        rec, taps = snaps[s]
        box = {}

        def prep():
            box["p"] = pd_transfer.prepare_gdn_import(model, rec, taps)

        th = threading.Thread(target=prep)
        t0 = time.perf_counter()
        th.start()
        th.join()
        t_prep = 1e3 * (time.perf_counter() - t0)
        prepared = box["p"]
        t0 = time.perf_counter()
        pd_transfer.import_gdn_slot(model, s, rec, prepared)
        dt = 1e3 * (time.perf_counter() - t0)
        t0 = time.perf_counter()
        pd_transfer.import_gdn_slot(model, s, rec, prepared)  # warm
        dt2 = 1e3 * (time.perf_counter() - t0)
        times.append(dt2)
        for li, dn in enumerate(dns):
            hosts[li].write(dn, s, rec[:, li], taps[:, li])
        f = 0
        for li in sample:
            f += _compare_layer(mesh, dns[li], hosts[li], li, f"NEW slot {s}")
        if s in ref_rows:
            for li in sample:
                r, cv, h = _read_layer(mesh, dns[li], n)
                rr, rcv, rh = ref_rows[s][li]
                ok = (
                    torch.equal(r[:, s], rr)
                    and all(torch.equal(cv[m][:, s], rcv[m]) for m in range(K))
                    and torch.equal(h[:, s], rh)
                )
                if not ok:
                    logger.error(f"[check] slot {s} layer {li}: NEW rows differ from the OLD importer's rows")
                    f += 1
        failures += f
        logger.info(
            f"[new] slot {s}: prep(thread) {t_prep:.1f} ms, first (capture) {dt:.0f} ms, warm import {dt2:.1f} ms; "
            f"sampled layers {sample} {'OK' if f == 0 else 'BAD'}{' (== old path rows)' if s in ref_rows and f == 0 else ''}"
        )
    # fallback path (no prepared object) on one slot
    s = 2
    rec, taps = snaps[s]
    t0 = time.perf_counter()
    pd_transfer.import_gdn_slot(model, s, rec, taps)
    logger.info(f"[new] slot {s} unprepared path: {1e3 * (time.perf_counter() - t0):.1f} ms")
    # ---- final: every layer, every row ----
    t0 = time.perf_counter()
    fin = sum(_compare_layer(mesh, dns[li], hosts[li], li, "FINAL") for li in range(L))
    failures += fin
    logger.info(
        f"[final] all {L} layers x {B} slots read back in {time.perf_counter() - t0:.1f} s: "
        f"{'BIT-IDENTICAL to the host expectation' if fin == 0 else f'{fin} layer mismatches'}"
    )
    times_s = sorted(times)
    logger.info(
        f"[timing] new warm import over {B} slots: min {times_s[0]:.1f} / med {times_s[len(times_s) // 2]:.1f} / max {times_s[-1]:.1f} ms"
    )
    logger.info(f"RESULT: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    ttnn.close_mesh_device(mesh)
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
