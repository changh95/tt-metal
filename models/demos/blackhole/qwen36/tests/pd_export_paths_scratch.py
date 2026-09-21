# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Export-path check without a model: random bfp8 paged caches on a 1x4 mesh, export the same block lists via
runs / blocks / gather and compare bitwise with a host reference; time each path. No weights needed.

    TT_VISIBLE_DEVICES=0,1,6,7 python models/demos/blackhole/qwen36/tests/pd_export_paths_scratch.py

Then (SCRATCH_N1=1, default) a 1x1 mesh phase for the n_dev == 1 pool-aliasing condition.
"""

import os
import time
from types import SimpleNamespace

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import pd_transfer

NB, NKV, BLK, HD, NL = int(os.environ.get("NB", "512")), 1, 64, 256, 16


def _pool_storage_ptrs(model):
    pool = getattr(model, "_kv_export_pool", None)
    return {host.untyped_storage().data_ptr() for host, _ in pool._free} if pool is not None else set()


def _aliases_pool(out, model) -> bool:
    """True when any exported tensor lives in a (released) KvExportPool entry."""
    ptrs = _pool_storage_ptrs(model)
    return any(t.untyped_storage().data_ptr() in ptrs for pair in out for t in pair)


def build_caches(mesh, n_dev, nb, nl):
    """Random bfp8 paged caches sharded over the mesh (dim 1 = device) + the host reference of what the device
    holds, [nb, n_dev*nkv, blk, hd]. Returns (model, host_ref)."""
    torch.manual_seed(0)
    layers, host_ref = [], []
    for _ in range(nl):
        pair = []
        for _ in range(2):
            src = torch.randn(nb, n_dev * NKV, BLK, HD, dtype=torch.bfloat16)  # dim 1 = device
            t = ttnn.from_torch(
                src,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
            )
            # host reference = what the device holds (bfp8-rounded), as [NB, n_dev*nkv, blk, hd]
            back = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).to(torch.bfloat16)
            pair.append((t, back))
        layers.append(
            SimpleNamespace(is_full_attention=True, attention=SimpleNamespace(paged_k=pair[0][0], paged_v=pair[1][0]))
        )
        host_ref.append((pair[0][1], pair[1][1]))
    return SimpleNamespace(layers=layers, mesh_device=mesh, num_devices=n_dev), host_ref


def single_device_phase() -> int:
    """n_dev == 1: the device-major -> block-major permute is a view of the pool entry there, so the export must
    clone explicitly. Check exactness, that no output aliases a pool entry, and that an earlier export's tensors
    survive later exports (which refill the same entry). Returns the number of failures."""
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576)
    mesh.enable_program_cache()
    model, host_ref = build_caches(mesh, 1, 64, 2)
    pd_transfer.export_warmup(model, max_bucket=8)
    bad = 0
    cases = {"contig3": [10, 11, 12], "desc3": [12, 11, 10], "frag3": [38, 57, 50], "one": [7]}
    kept = []
    for mode in ("auto", "blocks"):
        os.environ["QWEN36_PD_EXPORT"] = mode
        for name, ids in cases.items():
            out = pd_transfer.export_kv_blocks(model, ids)
            ok = all(torch.equal(k, hk[ids]) and torch.equal(v, hv[ids]) for (k, v), (hk, hv) in zip(out, host_ref))
            ok &= all(k.is_contiguous() and v.is_contiguous() for k, v in out)
            alias = _aliases_pool(out, model)
            tm = pd_transfer.LAST_EXPORT_TIMING
            logger.info(
                f"n_dev=1 {name:8s} {mode:6s} read={tm['read']:8s} {'OK ' if ok else 'BAD'} "
                f"{'ALIASES POOL' if alias else 'owns memory'}"
            )
            bad += (not ok) + alias
            kept.append((ids, [(k.clone(), v.clone()) for k, v in out], out))
    os.environ.pop("QWEN36_PD_EXPORT", None)
    # every earlier export must still hold its own bytes after all the later ones refilled the pool entry
    stale = sum(
        not all(torch.equal(k, k0) and torch.equal(v, v0) for (k0, v0), (k, v) in zip(snap, out))
        for _, snap, out in kept
    )
    logger.info(f"n_dev=1 earlier exports overwritten by later ones: {stale}/{len(kept)}")
    bad += stale
    ttnn.close_mesh_device(mesh)
    return bad


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=24576)
    mesh.enable_program_cache()
    model, host_ref = build_caches(mesh, 4, NB, NL)
    layers = model.layers
    logger.info("caches ready")
    t0 = time.perf_counter()
    pd_transfer.export_warmup(model, max_bucket=int(os.environ.get("WARM_MAX", "128")))
    logger.info(f"warm-up {time.perf_counter() - t0:.1f} s")
    cases = {
        "contig3": [10, 11, 12],
        "desc3": [12, 11, 10],
        "frag3": [38, 57, 50],
        "top3": [NB - 3, NB - 2, NB - 1],
        "contig16": list(range(100, 116)),
        "frag20": [int(x) for x in torch.randperm(NB)[:20]],
        "contig64": list(range(200, 264)),
        "frag64": [int(x) for x in torch.randperm(NB)[:64]],
        "contig128": list(range(300, 428)),
        "frag128": [int(x) for x in torch.randperm(NB)[:128]],
        "one": [7],
    }
    bad = 0
    for name, ids in cases.items():
        res = {}
        for mode in ("auto", "blocks"):
            os.environ["QWEN36_PD_EXPORT"] = mode
            for _ in range(2):  # second call = warm timing
                t0 = time.perf_counter()
                out = pd_transfer.export_kv_blocks(model, ids)
                dt = time.perf_counter() - t0
            ok = all(torch.equal(k, hk[ids]) and torch.equal(v, hv[ids]) for (k, v), (hk, hv) in zip(out, host_ref))
            ok &= all(k.is_contiguous() and v.is_contiguous() and k.dtype == torch.bfloat16 for k, v in out)
            ok &= not _aliases_pool(out, model)  # outputs must own their memory (pool entry is released)
            tm = dict(pd_transfer.LAST_EXPORT_TIMING)
            res[mode] = (ok, dt, tm)
            bad += not ok
        logger.info(
            f"{name:9s} n={len(ids):3d} runs={len(pd_transfer.coalesce_runs(ids))} "
            + "  ".join(
                f"{m}: {'OK ' if ok else 'BAD'} {tm['total_ms']:7.1f} ms (device {tm['device_ms']:.1f} {tm.get('read', 'read')} {tm['read_ms']:.1f} host {tm['host_ms']:.1f})"
                for m, (ok, dt, tm) in res.items()
            )
        )
    os.environ.pop("QWEN36_PD_EXPORT", None)
    # import round trip with bucket padding: 3 real blocks -> bucket 4, padding rows land in the pad block only
    model.num_devices = 4
    model._pad_kv_block = NB - 1
    before = [
        (
            ttnn.to_torch(l.attention.paged_k, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).to(torch.bfloat16),
            ttnn.to_torch(l.attention.paged_v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).to(torch.bfloat16),
        )
        for l in layers
    ]
    src, dst = [10, 11, 12], [200, 57, 3]
    kv = pd_transfer.export_kv_blocks(model, src)
    pd_transfer.import_warmup(model, max_bucket=8)
    t0 = time.perf_counter()
    pd_transfer.import_kv_blocks(model, dst, kv)
    dt = time.perf_counter() - t0
    ok_imp = True
    for li, l in enumerate(layers):
        for cache, (bk, bv), ref_pair in ((l.attention.paged_k, kv[li], before[li]),):
            pass
        for j, cache in enumerate((l.attention.paged_k, l.attention.paged_v)):
            after = ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).to(torch.bfloat16)
            ref = before[li][j].clone()
            ref[dst] = kv[li][j]  # destination blocks take the exported rows
            mask = torch.ones(NB, dtype=torch.bool)
            mask[NB - 1] = False  # pad block may hold anything
            ok_imp &= torch.equal(after[mask], ref[mask])
    logger.info(
        f"import 3 blocks -> bucket 4 into {dst}: {'OK' if ok_imp else 'BAD'} ({1e3 * dt:.1f} ms); only dst blocks (+pad) changed"
    )
    bad += not ok_imp
    ttnn.close_mesh_device(mesh)
    if os.environ.get("SCRATCH_N1", "1") == "1":
        bad += single_device_phase()
    logger.info("ALL OK" if bad == 0 else f"{bad} MISMATCHES")


if __name__ == "__main__":
    main()
