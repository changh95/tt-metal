# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-identity check of the traced KV import (pd_transfer.TracedKvImporter) against the eager per-cache path and
the host expectation, on synthetic full-attention layers with the served per-device paged-cache geometry (TP=4:
nkv=1, block 64, head_dim 256, 16 layers = 32 caches).

Two identical cache sets are built (random contents); for each block list (single block, ascending/descending runs,
a fragmented list, whole buckets, and lists above the traced bucket cap that go through chunked replays) the same
random payload is imported eagerly (QWEN36_PD_KV_IMPORT_TRACE=0) into set A and traced into set B; all 32 caches of
both sets are read back and compared byte for byte with each other and with the host expectation (payload blocks
written, every other block untouched; the pad block is excluded from the host check since both paths aim their
zero pad rows at it). Timings of both paths are logged.

Run (half A):
  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 ARCH_NAME=blackhole \
  python_env/bin/python models/demos/blackhole/qwen36/tests/pd_kv_import_bitexact_check.py
Env: PD_KV_LAYERS (16), PD_KV_BLOCKS (320 blocks per cache), PD_KV_MESH (1x4), QWEN36_PD_KV_TRACE_MAX_BUCKET (64).
"""

import os
import time
import types

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import pd_transfer

_e = lambda n, d: int(os.environ.get(n) or d)
NKV, BLK, HD = _e("PD_KV_NKV", 1), _e("PD_KV_BLK", 64), _e("PD_KV_HD", 256)
LAYERS = _e("PD_KV_LAYERS", 16)
NUM_BLOCKS = _e("PD_KV_BLOCKS", 320)
MESH_SHAPE = tuple(int(v) for v in os.environ.get("PD_KV_MESH", "1x4").lower().split("x"))


def _shard(mesh, host, dtype):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _read(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def _make_model(mesh, init):
    """init: [n_dev, 2*LAYERS, NUM_BLOCKS, NKV, BLK, HD] bf16 -> fake model with paged caches holding it."""
    n = init.shape[0]
    layers = []
    for li in range(LAYERS):
        att = types.SimpleNamespace(
            paged_k=_shard(mesh, init[:, 2 * li].reshape(n * NUM_BLOCKS, NKV, BLK, HD), ttnn.bfloat16),
            paged_v=_shard(mesh, init[:, 2 * li + 1].reshape(n * NUM_BLOCKS, NKV, BLK, HD), ttnn.bfloat16),
        )
        layers.append(types.SimpleNamespace(is_full_attention=True, attention=att))
    return types.SimpleNamespace(mesh_device=mesh, num_devices=n, layers=layers, _pad_kv_block=NUM_BLOCKS - 1)


def _read_all(mesh, model, n):
    return torch.stack(
        [
            _read(mesh, c).view(n, NUM_BLOCKS, NKV, BLK, HD)
            for l in model.layers
            for c in (l.attention.paged_k, l.attention.paged_v)
        ],
        dim=1,
    )  # [n_dev, 32, NUM_BLOCKS, NKV, BLK, HD]


def _payload(gen, n_dev, n_blocks):
    return [
        (
            torch.randn(n_blocks, n_dev * NKV, BLK, HD, generator=gen).to(torch.bfloat16),
            torch.randn(n_blocks, n_dev * NKV, BLK, HD, generator=gen).to(torch.bfloat16),
        )
        for _ in range(LAYERS)
    ]


def _timed(fn, reps=3):
    fn()  # warm (compiles / captures)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(1e3 * (time.perf_counter() - t0))
    return min(ts)


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE), l1_small_size=24576, trace_region_size=512 * 2**20)
    mesh.enable_program_cache()
    n = mesh.get_num_devices()
    gen = torch.Generator().manual_seed(3)
    pad = NUM_BLOCKS - 1
    logger.info(
        f"mesh {MESH_SHAPE} ({n} devices) layers={LAYERS} blocks={NUM_BLOCKS} nkv={NKV} blk={BLK} hd={HD} pad={pad} "
        f"traced max bucket {pd_transfer.kv_trace_max_bucket()}"
    )
    init = torch.randn(n, 2 * LAYERS, NUM_BLOCKS, NKV, BLK, HD, generator=gen).to(torch.bfloat16)
    model_a = _make_model(mesh, init)  # eager
    model_b = _make_model(mesh, init)  # traced
    expect = init.clone()
    t0 = time.perf_counter()
    os.environ["QWEN36_PD_KV_IMPORT_TRACE"] = "1"
    pd_transfer.import_warmup(model_b, max_bucket=2048)
    logger.info(
        f"[warmup] traced importer: {len(model_b.pd_kv_importer.traces)} traces in {time.perf_counter() - t0:.1f} s"
    )
    cases = [
        ("single", [5]),
        ("asc run x3", [10, 11, 12]),
        ("desc run x5", [40, 39, 38, 37, 36]),
        ("fragmented x16", [3, 9, 7, 8, 20, 21, 22, 23, 50, 51, 52, 53, 54, 55, 56, 57]),
        ("run x64 (max traced bucket)", list(range(100, 164))),
        ("run x100 (bucket 128 -> 2 chunks)", list(range(170, 270))),
        ("desc x130 (bucket 256 -> 4 chunks)", list(range(300, 170, -1))),
        ("single again (pad rows rewritten)", [77]),
    ]
    failures = 0
    for name, ids in cases:
        kv = _payload(gen, n, len(ids))
        os.environ["QWEN36_PD_KV_IMPORT_TRACE"] = "0"
        t_eager = _timed(lambda: pd_transfer.import_kv_blocks(model_a, ids, kv))
        os.environ["QWEN36_PD_KV_IMPORT_TRACE"] = "1"
        t_trace = _timed(lambda: pd_transfer.import_kv_blocks(model_b, ids, kv))  # incl. the host prep
        t0 = time.perf_counter()
        prep = pd_transfer.prepare_kv_import(model_b, kv)  # what the connector's pull thread does
        t_prep = 1e3 * (time.perf_counter() - t0)
        t_trace_prepared = _timed(lambda: pd_transfer.import_kv_blocks(model_b, ids, prep))  # main-thread share
        ttnn.synchronize_device(mesh)
        for li, (k, v) in enumerate(kv):
            for j, t in enumerate((k, v)):
                # [n_blocks, n_dev*nkv, blk, hd] -> device-major rows
                expect[:, 2 * li + j, ids] = t.view(len(ids), n, NKV, BLK, HD).permute(1, 0, 2, 3, 4)
        got_a, got_b = _read_all(mesh, model_a, n), _read_all(mesh, model_b, n)
        keep = [b for b in range(NUM_BLOCKS) if b != pad]  # the pad block takes each path's pad rows (or none)
        same_ab = torch.equal(got_a[:, :, keep], got_b[:, :, keep])
        ok_b = torch.equal(got_b[:, :, keep], expect[:, :, keep])
        ok_a = torch.equal(got_a[:, :, keep], expect[:, :, keep])
        failures += (not same_ab) + (not ok_b) + (not ok_a)
        bad = ""
        if not (same_ab and ok_b and ok_a):
            diff = (got_b != expect).any(dim=(0, 1, 3, 4, 5)).nonzero().flatten().tolist()
            bad = f" BAD: eager==host {ok_a}, traced==host {ok_b}, eager==traced {same_ab}; traced blocks differing {diff[:10]}"
        logger.info(
            f"[kv] {name}: {len(ids)} blocks -> bucket {pd_transfer.kv_import_bucket(len(ids))} ({prep.n_chunks} chunk(s) of "
            f"{prep.chunk}): eager {t_eager:.1f} ms, traced {t_trace:.1f} ms (host prep {t_prep:.1f} ms; prepared -> "
            f"{t_trace_prepared:.1f} ms); {'BIT-IDENTICAL (eager == traced == host)' if not bad else bad}"
        )
    logger.info(f"RESULT: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    ttnn.close_mesh_device(mesh)
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
