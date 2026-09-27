# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Piecewise pooled KV export vs the single-read composer path: bit-exactness and timing, no weights needed.

Random bfp8 paged caches (the served shape: 17 attention layers incl. the MTP head, nkv=1, 64 x 256 blocks) on a 1x4
mesh; for each block list the export is read once with the pool OFF (QWEN36_PD_EXPORT_POOL=0: one power-of-two bucket
through the mesh composer, an exact-count compile above 2048 -- the pre-piecewise path for > 256 blocks) and with the
pool ON (pieces <= 256 blocks through the pooled DMA), and the two results must be torch.equal per layer and side.
Warm timings of both are printed as the before/after per size ("[pd] exported" ms).

    TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 python models/demos/blackhole/qwen36/tests/pd_export_piecewise_check.py

Env: NB (2304 blocks in the caches), NL (17 layers), SIZES (129,257,513,1025,2049), SKIP_COMPOSER=1 (timing only).
"""

import os
import time
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import pd_transfer

NB = int(os.environ.get("NB", "2304"))
NL = int(os.environ.get("NL", "17"))
NKV, BLK, HD = 1, 64, 256
SIZES = [int(x) for x in os.environ.get("SIZES", "129,257,513,1025,2049").split(",")]


def build_caches(mesh, n_dev, nb, nl):
    torch.manual_seed(0)
    layers = []
    for _ in range(nl):
        pair = []
        for _ in range(2):
            src = torch.randn(nb, n_dev * NKV, BLK, HD, dtype=torch.bfloat16)
            pair.append(
                ttnn.from_torch(
                    src,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
                )
            )
        layers.append(
            SimpleNamespace(is_full_attention=True, attention=SimpleNamespace(paged_k=pair[0], paged_v=pair[1]))
        )
    return SimpleNamespace(layers=layers, mesh_device=mesh, num_devices=n_dev)


def export_timed(model, ids, pool_on, reps=2):
    os.environ["QWEN36_PD_EXPORT_POOL"] = "1" if pool_on else "0"
    if not pool_on:
        model._kv_export_pool = None
    out, dts = None, []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = pd_transfer.export_kv_blocks(model, ids)
        dts.append(1e3 * (time.perf_counter() - t0))
    tm = dict(pd_transfer.LAST_EXPORT_TIMING)
    return out, dts, tm


def same(a, b):
    return all(torch.equal(k1, k2) and torch.equal(v1, v2) for (k1, v1), (k2, v2) in zip(a, b))


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
    mesh.enable_program_cache()
    model = build_caches(mesh, 4, NB, NL)
    logger.info(f"caches ready: {NL} layers x 2 x [{NB}, {NKV}, {BLK}, {HD}] bfp8")
    t0 = time.perf_counter()
    os.environ["QWEN36_PD_EXPORT_POOL"] = "1"
    pd_transfer.export_warmup(model, max_bucket=2048)  # capped at the pool's reach (256) + one piecewise export
    logger.info(f"warm-up (pool on) {time.perf_counter() - t0:.1f} s")
    skip_composer = os.environ.get("SKIP_COMPOSER", "0") == "1"
    cases = []
    for n in SIZES:
        cases.append((f"asc{n}", list(range(7, 7 + n))))
    cases.append(("desc257", list(range(300 + 256, 299, -1))))
    cases.append(("frag300", [int(x) for x in torch.randperm(NB)[:300]]))
    cases.append(("tworuns261", list(range(100, 300)) + list(range(700, 761))))
    cases.append(("top129", list(range(NB - 129, NB))))
    bad = 0
    rows = []
    for name, ids in cases:
        piece, dts_p, tm_p = export_timed(model, ids, pool_on=True)
        ref_ms = "-"
        ok = True
        if not skip_composer:
            comp, dts_c, tm_c = export_timed(model, ids, pool_on=False, reps=2)
            ok = same(piece, comp)
            ref_ms = f"{dts_c[-1]:.0f} (first {dts_c[0]:.0f}, {tm_c['read']} bucket {tm_c['bucket']})"
            del comp
        ok &= all(k.is_contiguous() and v.is_contiguous() for k, v in piece)
        bad += not ok
        rows.append((name, len(ids), ok, dts_p[-1], tm_p, ref_ms))
        logger.info(
            f"{name:10s} n={len(ids):5d} {'OK ' if ok else 'BAD'} piecewise {dts_p[-1]:7.0f} ms "
            f"(device {tm_p['device_ms']:.0f} read {tm_p['read_ms']:.0f} host {tm_p['host_ms']:.0f}; pieces {tm_p['pieces']}) "
            f"| composer {ref_ms} ms"
        )
        del piece
    logger.info("| case | blocks | exact | piecewise ms (warm) | pieces | composer/single-bucket ms (warm, first) |")
    logger.info("|---|---|---|---|---|---|")
    for name, n, ok, ms, tm, ref in rows:
        logger.info(f"| {name} | {n} | {'yes' if ok else 'NO'} | {ms:.0f} | {tm['pieces']} | {ref} |")
    os.environ.pop("QWEN36_PD_EXPORT_POOL", None)
    ttnn.close_mesh_device(mesh)
    logger.info("ALL OK" if bad == 0 else f"{bad} MISMATCHES")
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
