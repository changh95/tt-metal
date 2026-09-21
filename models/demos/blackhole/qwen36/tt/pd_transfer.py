# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill/decode disaggregation: host-staged export/import of one request's state (TP model).

A request's state on the Qwen3.x hybrid model is (a) its paged-KV blocks in the 16 full-attention
layers and (b) its Gated DeltaNet state in the 48 linear-attention layers: the fp32 recurrent state
`[Nv/TP, Dk, Dv]` and the K causal-conv taps `[1, qkv_dim_tp]` per layer per chip. Both halves of a
prefill/decode split run the same TP sharding, so everything moves per device (mesh dim 0 = device)
with no reshuffle.

Prefill side: `prefill_paged_slots` snapshots each request's GDN state to the host anyway (the served
plain path writes the decode slot from that snapshot); with `model.pd_gdn_capture` set to a dict it
also parks the snapshot under the request's slot, and `model.pd_skip_gdn_slot_write` skips the slot
write (a pure-prefill instance never decodes). `export_kv_blocks` reads the request's blocks out of the
paged caches.

Snapshot layout (device-major, one host tensor per state type; `GdnSnapshotPool`): `rec` is
`[n_dev, L, Nv, Dk, Dv]` in the model's recurrent-state dtype (fp32 by default) and `taps` is
`[n_dev, L, K, C]` bf16, L = GDN layers, K = conv taps, C = qkv_dim_tp. Device d's shard of every layer
is contiguous, so the D side uploads it with one borrowed row-major transfer per state type and P's
read is one DMA per state type straight into the host buffer. The host buffers come from a pool the
prefill side reuses across requests (the second read into a pinned buffer runs at PCIe rate); the
consumer of a snapshot hands it back with `model.pd_gdn_snapshot_release(rec, taps)` when done.

Decode side: `import_kv_blocks` fills the request's blocks via `paged_fill_cache` over its page-table
row (by default through `TracedKvImporter`: one staged upload + one trace replay per block bucket);
`import_gdn_slot` writes the snapshot into the request's decode slot (by default through
`TracedGdnImporter`: staged uploads + one per-slot trace replay of in-place row writes). Both host
preparations (`prepare_kv_import`, `prepare_gdn_import`) are torch-only and may run off the main thread.
The decode instance then continues the request with one ordinary decode step for the last prompt token
(the prefill side computed h(N-1)).
"""

from __future__ import annotations

import os
import threading
import time

import torch
from loguru import logger

import ttnn


def coalesce_runs(block_ids):
    """Consecutive runs of block ids, ascending or descending, as (lo, hi, descending) with [lo, hi) the
    contiguous region; order of runs = order of block_ids. vLLM's free-block queue hands out descending
    sequences after frees, so both directions matter: [3,4,5,9,10,7] -> [(3,6,F),(9,11,F),(7,8,F)];
    [17,16,15,2,3] -> [(15,18,T),(2,4,F)]."""
    runs = []  # [lo, hi, direction] direction: 0 unknown (single), +1 asc, -1 desc
    for b in block_ids:
        b = int(b)
        if runs:
            lo, hi, d = runs[-1]
            if d >= 0 and b == hi:
                runs[-1] = [lo, hi + 1, 1]
                continue
            if d <= 0 and b == lo - 1:
                runs[-1] = [lo - 1, hi, -1]
                continue
        runs.append([b, b + 1, 0])
    return [(lo, hi, d < 0) for lo, hi, d in runs]


def _reorder_runs(host: torch.Tensor, runs) -> torch.Tensor:
    """Put a device-read tensor whose dim 0 is the concatenation of the runs' contiguous regions into
    block_ids order: descending runs are read ascending on device and flipped here."""
    if not any(desc for _, _, desc in runs):
        return host
    parts, off = [], 0
    for lo, hi, desc in runs:
        n = hi - lo
        part = host[off : off + n]
        parts.append(torch.flip(part, dims=[0]) if desc else part)
        off += n
    return torch.cat(parts, dim=0)


def _copy_off_pool(t: torch.Tensor, pool_buf) -> torch.Tensor:
    """`t` as a contiguous tensor that does not alias `pool_buf` (a borrowed KvExportPool entry, or None for
    the composer read whose host tensor is already a fresh copy)."""
    if pool_buf is not None and t.untyped_storage().data_ptr() == pool_buf.untyped_storage().data_ptr():
        return t.clone(memory_format=torch.contiguous_format)
    return t.contiguous()


def _device_convert() -> bool:
    """QWEN36_PD_DEVICE_CONVERT=0 falls back to host-side tilize/untilize + dtype conversion."""
    return os.environ.get("QWEN36_PD_DEVICE_CONVERT", "1") != "0"


def _upload(model, host: torch.Tensor, dtype, mapper):
    """Host tensor -> device tensor of `dtype` in TILE layout. With device conversion (default) the host
    transfer is a row-major memcpy in the host tensor's own dtype (bf16/fp32) and tilize + typecast run on
    device; otherwise from_torch tilizes/packs on the host."""
    mesh = model.mesh_device
    if not _device_convert():
        return ttnn.from_torch(
            host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
    host_dtype = {torch.bfloat16: ttnn.bfloat16, torch.float32: ttnn.float32}.get(host.dtype)
    if host_dtype is None:
        host = host.to(torch.bfloat16)
        host_dtype = ttnn.bfloat16
    rm = ttnn.from_torch(
        host,
        dtype=host_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    t = ttnn.to_layout(rm, ttnn.TILE_LAYOUT)
    ttnn.deallocate(rm)
    if t.dtype != dtype:
        tc = ttnn.typecast(t, dtype)
        ttnn.deallocate(t)
        t = tc
    return t


def _attention_layers(model):
    return [layer.attention for layer in model.layers if layer.is_full_attention]


def pad_block_ids(block_ids, n, num_blocks):
    """Extend `block_ids` to `n` entries with other VALID block ids (their rows are dropped on the host).
    A single monotonic run is extended in its own direction so it stays one run (the single-slice path);
    anything else repeats the last block. Returns (padded_ids, offset) with the real rows at
    [offset, offset + len(block_ids))."""
    ids = [int(b) for b in block_ids]
    k = n - len(ids)
    if k <= 0:
        return ids, 0
    runs = coalesce_runs(ids)
    if len(runs) == 1:
        lo, hi, desc = runs[0]
        if not desc and hi + k <= num_blocks:
            return ids + list(range(hi, hi + k)), 0
        if desc and lo - k >= 0:
            return ids + list(range(lo - 1, lo - 1 - k, -1)), 0
        if not desc and lo - k >= 0:  # top of the pool: extend below, real rows follow the pad rows
            return list(range(lo - k, lo)) + ids, k
        if desc and hi + k <= num_blocks:
            return list(range(hi + k - 1, hi - 1, -1)) + ids, k
    return ids + [ids[-1]] * k, 0


LAST_EXPORT_TIMING = {}  # diagnostics: the last export_kv_blocks call's device/read/host split (ms)


class KvExportPool:
    """Reusable host buffers for the KV export read, filled by direct DMA (the GdnSnapshotPool pattern).

    An entry is one torch bf16 buffer `[n_dev * n_caches * max_blocks, nkv, blk, hd]` (n_caches = 2 x attention
    layers) plus, per export bucket `n <= max_blocks`, a ROW_MAJOR host mesh tensor made once with
    `ttnn.from_torch(buf[: n_dev * n_caches * n], mesh_mapper=ShardTensorToMesh(dim 0))`: a contiguous prefix of
    the buffer, so from_torch borrows it (dim-0 shards are contiguous chunks) and `ttnn.copy_device_to_host_tensor`
    lands device d's `[n_caches * n, nkv, blk, hd]` straight in the torch memory -- no mesh composer, no host
    concat, no `to_torch` copy. One buffer serves every bucket, so the pages faulted in by the first (warm-up)
    read stay warm for all of them. `export_kv_blocks` borrows an entry for the duration of one call and hands
    it back; its outputs never alias the entry (they are copied off it unconditionally -- at n_dev > 1 the
    device-major -> block-major permute is already a copy, at n_dev == 1 it is a view and an explicit clone is
    taken), so one entry per concurrent export suffices; the pool is capped at `max_entries` (an exhausted
    pool falls back to the composer read). Memory per entry =
    4 MiB x max_blocks at TP4 (1 GiB at the default 256 blocks); buckets above `max_blocks` use the composer.
    Host memory only: safe to create at request time under captured traces.
    """

    def __init__(self, model, max_blocks: int, max_entries: int):
        layers = _attention_layers(model)
        _, nkv, blk, hd = layers[0].paged_k.shape
        self.model = model
        self.n_dev = int(model.num_devices)
        self.n_caches = 2 * len(layers)
        self.row_shape = (int(nkv), int(blk), int(hd))
        self.max_blocks = int(max_blocks)
        self.max_entries = int(max_entries)
        self.mapper = ttnn.ShardTensorToMesh(model.mesh_device, dim=0)
        self._lock = threading.Lock()
        self._free = []  # entries: (host buffer, {bucket: borrowed host mesh tensor})
        self.total = 0

    def entry_nbytes(self) -> int:
        nkv, blk, hd = self.row_shape
        return self.n_dev * self.n_caches * self.max_blocks * nkv * blk * hd * 2

    def _new(self):
        host = torch.empty((self.n_dev * self.n_caches * self.max_blocks, *self.row_shape), dtype=torch.bfloat16)
        self.total += 1
        logger.info(f"[pd] KV export pool: +1 buffer ({self.entry_nbytes() / 2**20:.0f} MiB), {self.total} total")
        return (host, {})

    def acquire(self, n: int):
        """Borrow (entry, host view [n_dev * n_caches * n, nkv, blk, hd], borrowed mesh tensor over that view) for
        bucket `n`, or None when `n` exceeds the pool's reach or every entry is in use."""
        if n > self.max_blocks:
            return None
        with self._lock:
            if self._free:
                entry = self._free.pop()
            elif self.total < self.max_entries:
                entry = self._new()
            else:
                return None
        host, views = entry
        rows = self.n_dev * self.n_caches * n
        view = host[:rows]
        tt = views.get(n)
        if tt is None:
            tt = views[n] = ttnn.from_torch(
                view, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mapper
            )
        return entry, view, tt

    def release(self, entry):
        with self._lock:
            self._free.append(entry)


def _kv_export_pool(model):
    """The model's KvExportPool (created on first use; QWEN36_PD_EXPORT_POOL=0 disables it -> composer read).
    QWEN36_PD_EXPORT_POOL_MAX_BLOCKS (256) bounds the pooled buckets, QWEN36_PD_EXPORT_POOL_ENTRIES (2) the
    number of buffers."""
    if os.environ.get("QWEN36_PD_EXPORT_POOL", "1") != "1":
        return None
    pool = getattr(model, "_kv_export_pool", None)
    if pool is None:
        pool = model._kv_export_pool = KvExportPool(
            model,
            max_blocks=int(os.environ.get("QWEN36_PD_EXPORT_POOL_MAX_BLOCKS", "256")),
            max_entries=int(os.environ.get("QWEN36_PD_EXPORT_POOL_ENTRIES", "2")),
        )
    return pool


def export_kv_blocks(model, block_ids):
    """Read one request's paged-KV blocks off the device.

    Returns, per full-attention layer, a `(k, v)` pair of host tensors shaped
    `[n_blocks, n_dev * n_local_kv_heads, block_size, head_dim]` in torch.bfloat16 (a bf8 cache reads
    back through bf16 exactly), block order = `block_ids` order.

    One device read for the whole request: the blocks of all 32 cache tensors are concatenated on device
    into a single tensor, converted (bfp8 -> bf16, untilize) once, and DMA'd once into a pooled, borrowed
    row-major host buffer (`KvExportPool`, `ttnn.copy_device_to_host_tensor`; 128 blocks = 512 MiB read in
    ~14 ms vs ~330 ms through the composer; ~38 ms for the whole export incl. the host permute below).
    The returned tensors own their memory (never a view of the pool entry, which is released on return).
    Buckets beyond the pool (or QWEN36_PD_EXPORT_POOL=0 /
    QWEN36_PD_DEVICE_CONVERT=0) fall back to the dim-0 mesh composer read (~1.2 GiB/s plus a host cat;
    composing along a middle dim ran at ~0.1 GiB/s and 32 separate reads took seconds).

    Program shapes depend on the block count, so the block list is padded to a power-of-two bucket
    (export_warmup compiles every bucket at startup) and read by one of two fixed-shape paths:
    `runs`  -- the list is one contiguous run: one slice per cache, concat of 32;
    `blocks` -- anything else: one single-block slice per (cache, block), a concat of `bucket` per
    cache, then a concat of 32. Slice offsets are runtime arguments, so neither path compiles per
    block id. (`ttnn.gather` was tried for fragmented lists: it reads the WHOLE cache per call, ~110 ms
    x 32 caches = 3.5 s per request on a 500 MB cache, and returned wrong rows for a bfp8 cache.)
    """
    t0 = time.perf_counter()
    layers = _attention_layers(model)
    n_dev = model.num_devices
    n_real = len(block_ids)
    block_ids = [int(b) for b in block_ids]
    n = export_bucket(n_real) if os.environ.get("QWEN36_PD_EXPORT_BUCKETS", "1") == "1" else n_real
    cache0 = layers[0].paged_k
    num_blocks, nkv, blk, hd = cache0.shape
    block_ids, real_off = pad_block_ids(block_ids, n, int(num_blocks))
    runs = coalesce_runs(block_ids)
    caches = [c for att in layers for c in (att.paged_k, att.paged_v)]
    mode = os.environ.get("QWEN36_PD_EXPORT", "auto")
    if mode == "auto":
        mode = "runs" if len(runs) == 1 else "blocks"
    parts = []
    if mode == "runs":
        for cache in caches:
            for lo, hi, _ in runs:
                parts.append(ttnn.slice(cache, (lo, 0, 0, 0), (hi, nkv, blk, hd)))
    elif mode == "blocks":
        for cache in caches:
            singles = [ttnn.slice(cache, (b, 0, 0, 0), (b + 1, nkv, blk, hd)) for b in block_ids]
            if len(singles) > 1:
                cat = ttnn.concat(singles, dim=0)
                for s_ in singles:
                    ttnn.deallocate(s_)
                parts.append(cat)
            else:
                parts.append(singles[0])
    else:
        raise ValueError(f"QWEN36_PD_EXPORT={mode!r}: expected auto, runs or blocks")
    big = ttnn.concat(parts, dim=0) if len(parts) > 1 else parts[0]  # per device [32*n, nkv, blk, hd]
    if len(parts) > 1:
        for p_ in parts:
            ttnn.deallocate(p_)
    if _device_convert():
        if big.dtype != ttnn.bfloat16:
            b16 = ttnn.typecast(big, ttnn.bfloat16)
            ttnn.deallocate(big)
            big = b16
        rm = ttnn.to_layout(big, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(big)
        big = rm
    t1 = time.perf_counter()
    pool = _kv_export_pool(model) if big.dtype == ttnn.bfloat16 and big.layout == ttnn.ROW_MAJOR_LAYOUT else None
    borrowed = pool.acquire(n) if pool is not None else None
    try:
        if borrowed is not None:
            _, host, host_tt = borrowed  # host: torch view the DMA lands in, [n_dev * 32 * n, nkv, blk, hd]
            ttnn.copy_device_to_host_tensor(big, host_tt, blocking=True)
            read = "dma"
        else:
            host = ttnn.to_torch(big, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)).to(
                torch.bfloat16
            )
            read = "composer"
        ttnn.deallocate(big)
        t2 = time.perf_counter()
        # host: [n_dev * 32 * n, nkv, blk, hd] -> [n_dev, 32, n, nkv, blk, hd]
        host = host.view(n_dev, len(caches), n, nkv, blk, hd)
        out = []
        for li in range(len(layers)):
            pair = []
            for j in (0, 1):
                t = host[:, 2 * li + j]  # [n_dev, n, nkv, blk, hd]
                t = t.permute(1, 0, 2, 3, 4).reshape(n, n_dev * nkv, blk, hd)
                if mode == "runs":
                    t = _reorder_runs(t, runs)
                # The output must own its memory: the pool entry is released below and refilled by the next
                # export. permute+reshape is a copy only when n_dev > 1; at n_dev == 1 it is a view (and
                # _reorder_runs returns its input for ascending runs, .contiguous() is a no-op), so clone then.
                pair.append(_copy_off_pool(t[real_off : real_off + n_real], host if borrowed is not None else None))
            out.append((pair[0], pair[1]))
    finally:
        if borrowed is not None:
            pool.release(borrowed[0])
    t3 = time.perf_counter()
    LAST_EXPORT_TIMING.update(
        n_real=n_real,
        bucket=n,
        mode=mode,
        read=read,
        total_ms=1e3 * (t3 - t0),
        device_ms=1e3 * (t1 - t0),
        read_ms=1e3 * (t2 - t1),
        host_ms=1e3 * (t3 - t2),
    )
    logger.debug(
        f"[pd] exported {n_real} KV blocks (bucket {n}) x {len(out)} layers in {1e3 * (t3 - t0):.1f} ms "
        f"({len(runs)} run(s), {mode}; device {1e3 * (t1 - t0):.1f} ms, {read} read {1e3 * (t2 - t1):.1f} ms, "
        f"host {1e3 * (t3 - t2):.1f} ms)"
    )
    return out


_EXPORT_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def export_bucket(n_blocks: int) -> int:
    for b in _EXPORT_BUCKETS:
        if b >= n_blocks:
            return b
    return n_blocks  # beyond the pre-warmed buckets: exact count (compiles on first use)


def export_warmup(model, max_bucket: int = 2048):
    """Compile the export programs for every bucket up to max_bucket (~2 s per bucket and path)."""
    t0 = time.perf_counter()
    num_blocks = int(_attention_layers(model)[0].paged_k.shape[0])
    for b in _EXPORT_BUCKETS:
        if b > max_bucket:
            break
        # both fixed-shape paths: one contiguous run (single slice per cache) and the per-block path
        export_kv_blocks(model, list(range(1, 1 + b)))
        if b > 1:
            export_kv_blocks(model, [1] * b)
    logger.info(f"[pd] export warm-up: buckets <= {max_bucket} in {time.perf_counter() - t0:.1f} s")


def _pad_block(model, cache):
    pad = getattr(model, "_pad_kv_block", None)
    return int(pad) if pad is not None else int(cache.shape[0]) - 1


def kv_import_traced() -> bool:
    """QWEN36_PD_KV_IMPORT_TRACE=1 (default): import_kv_blocks replays a per-bucket trace (TracedKvImporter); 0 = the
    eager per-cache upload + paged_fill_cache path. Device conversion (QWEN36_PD_DEVICE_CONVERT) is required."""
    return os.environ.get("QWEN36_PD_KV_IMPORT_TRACE", "1") == "1" and _device_convert()


def import_kv_blocks(model, block_ids, kv):
    """Write `kv` (the `export_kv_blocks` layout, or a `PreparedKvImport` of it) into this instance's paged caches at
    `block_ids`.

    Program shapes (tilize, typecast, paged_fill_cache) depend on the block count, so the import is padded to
    the export's power-of-two bucket: the payload rows are followed by zero rows that land in the model's pad
    KV block (whose contents never reach a live request), and `import_warmup` compiles every bucket at start.
    Without it every new prompt-length bucket compiled ~1-2 s inside the first request's TTFT.

    Default path (`kv_import_traced`): `TracedKvImporter` -- one borrowed upload of all caches' payload into a
    persistent staging tensor, the block ids into a persistent page table, and a replayed per-bucket trace of
    tilize + 32 x (slice, paged_fill_cache). Buckets above QWEN36_PD_KV_TRACE_MAX_BUCKET (64) replay the largest
    trace once per chunk of blocks. The eager path below stays as the fallback (QWEN36_PD_KV_IMPORT_TRACE=0)."""
    if kv_import_traced():
        return get_traced_kv_importer(model).import_blocks(block_ids, kv)
    if isinstance(kv, PreparedKvImport):
        kv = kv.kv
    t0 = time.perf_counter()
    n_dev = model.num_devices
    n_real = len(block_ids)
    layers = _attention_layers(model)
    assert len(kv) == len(layers), f"{len(kv)} KV layer pairs for {len(layers)} attention layers"
    cache0 = layers[0].paged_k
    n = export_bucket(n_real) if os.environ.get("QWEN36_PD_IMPORT_BUCKETS", "1") == "1" else n_real
    ids = [int(b) for b in block_ids] + [_pad_block(model, cache0)] * (n - n_real)
    pt = torch.tensor([ids], dtype=torch.int32)  # [1, n]
    page_table_tt = ttnn.from_torch(
        pt,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=model.mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
    )
    mapper = ttnn.ShardTensorToMesh(model.mesh_device, dim=0)
    for att, (k_host, v_host) in zip(layers, kv):
        for host, cache in ((k_host, att.paged_k), (v_host, att.paged_v)):
            nb, ndn, blk, hd = host.shape
            assert nb == n_real, f"{nb} blocks in payload vs {n_real} block ids"
            if n != nb:
                host = torch.cat([host, host.new_zeros((n - nb, ndn, blk, hd))], dim=0)
            nkv = ndn // n_dev
            # [n, n_dev*nkv, blk, hd] -> [n_dev, nkv, n*blk, hd] (one [1, nkv, T, hd] fill per device)
            x = host.view(n, n_dev, nkv, blk, hd).permute(1, 2, 0, 3, 4).reshape(n_dev, nkv, n * blk, hd)
            xt = _upload(model, x.contiguous(), cache.dtype, mapper)
            ttnn.experimental.paged_fill_cache(cache, xt, page_table_tt, batch_idx=0)
            ttnn.deallocate(xt)
    ttnn.deallocate(page_table_tt)
    logger.debug(
        f"[pd] imported {n_real} KV blocks (bucket {n}) x {len(layers)} layers in {1e3 * (time.perf_counter() - t0):.1f} ms"
    )


def import_warmup(model, max_bucket: int = 2048):
    """Compile the KV import programs for every bucket up to max_bucket: zero payloads written into the pad
    block only. With the traced importer this allocates its persistent staging + page tables and captures the
    per-bucket traces (call it at warm-up, never at request time)."""
    t0 = time.perf_counter()
    if kv_import_traced():
        get_traced_kv_importer(model).warmup(max_bucket)
        logger.info(f"[pd] import warm-up (traced): buckets <= {max_bucket} in {time.perf_counter() - t0:.1f} s")
        return
    layers = _attention_layers(model)
    cache0 = layers[0].paged_k
    _, nkv, blk, hd = cache0.shape
    n_dev = model.num_devices
    pad = _pad_block(model, cache0)
    for b in _EXPORT_BUCKETS:
        if b > max_bucket:
            break
        z = torch.zeros((b, n_dev * nkv, blk, hd), dtype=torch.bfloat16)
        import_kv_blocks(model, [pad] * b, [(z, z) for _ in layers])
    logger.info(f"[pd] import warm-up: buckets <= {max_bucket} in {time.perf_counter() - t0:.1f} s")


# --------------------------------------------------------------------------------------
# traced per-bucket KV import
# --------------------------------------------------------------------------------------


def kv_trace_max_bucket() -> int:
    return int(os.environ.get("QWEN36_PD_KV_TRACE_MAX_BUCKET", "64"))


class PreparedKvImport:
    """One request's KV payload in the traced importer's staging order: `host` is
    `[n_chunks, n_dev * n_caches, nkv, chunk * blk, hd]` bf16 with chunk = min(bucket, max traced bucket), n_chunks =
    ceil(n_real / chunk), block b of the payload at chunk b // chunk, rows [(b % chunk) * blk, +blk) of every (device,
    cache) row group; the last chunk's pad blocks are zero rows. Each `host[c]` is contiguous, so its upload is a
    borrowed row-major transfer. Built by
    `prepare_kv_import` (torch ops only: may run on the connector's pull thread); `kv` keeps the payload views for the
    eager path."""

    __slots__ = ("kv", "n_real", "bucket", "chunk", "host")

    def __init__(self, kv, n_real, bucket, chunk, host):
        self.kv = kv
        self.n_real = n_real
        self.bucket = bucket
        self.chunk = chunk
        self.host = host

    @property
    def n_chunks(self) -> int:
        return self.host.shape[0]


def kv_import_bucket(n_real: int) -> int:
    return export_bucket(n_real) if os.environ.get("QWEN36_PD_IMPORT_BUCKETS", "1") == "1" else n_real


def prepare_kv_import(model, kv) -> PreparedKvImport:
    """Host-side preparation of `kv` (the `export_kv_blocks` layout: per layer (k, v) `[n, n_dev * nkv, blk, hd]`) for
    `import_kv_blocks`: one padded, chunked, device-major staging tensor (see PreparedKvImport)."""
    if isinstance(kv, PreparedKvImport):
        return kv
    n_dev = int(model.num_devices)
    n_caches = 2 * len(kv)
    n_real, ndn, blk, hd = kv[0][0].shape
    nkv = ndn // n_dev
    bucket = kv_import_bucket(int(n_real))
    chunk = min(bucket, kv_trace_max_bucket())
    n_chunks = -(-int(n_real) // chunk)  # chunks holding payload blocks; all-pad chunks of the bucket are not replayed
    host = torch.empty(n_chunks, n_dev, n_caches, nkv, chunk, blk, hd, dtype=torch.bfloat16)
    tail = int(n_real) - (n_chunks - 1) * chunk  # payload blocks in the last chunk
    if tail < chunk:
        host[n_chunks - 1, :, :, :, tail:].zero_()  # only the pad rows need defined (zero) bytes
    for li, pair in enumerate(kv):
        for j, t in enumerate(pair):
            if t.shape[0] != n_real:
                raise ValueError(f"KV layer {li}: {t.shape[0]} blocks vs {n_real}")
            src = t.view(n_real, n_dev, nkv, blk, hd)
            for c in range(n_chunks):
                lo, hi = c * chunk, min((c + 1) * chunk, int(n_real))
                # [cnt, n_dev, nkv, blk, hd] -> [n_dev, nkv, cnt, blk, hd]
                host[c, :, 2 * li + j, :, : hi - lo] = src[lo:hi].permute(1, 2, 0, 3, 4)
    host = host.view(n_chunks, n_dev * n_caches, nkv, chunk * blk, hd)
    return PreparedKvImport(kv, int(n_real), bucket, chunk, host)


class TracedKvImporter:
    """Write a request's KV blocks into the paged caches with one trace replay per chunk of blocks.

    Persistent (allocated in `warmup`, i.e. at the connector's post-warm-up hook, never at request time): per traced
    bucket n a ROW_MAJOR bf16 staging tensor `[n_dev * n_caches, nkv, n * blk, hd]` (sharded on dim 0: device d holds
    `[n_caches, nkv, n * blk, hd]`, n MiB at TP4) and an int32 `[1, n]` page table (replicated). The per-bucket trace:
    tilize the staging (1 op) then per cache slice its `[1, nkv, n * blk, hd]` payload (+ typecast for a non-bf16
    cache) and `paged_fill_cache(cache, x, page_table, batch_idx=0)` -- the page table is a TENSOR input read at replay,
    so the block ids are runtime data: `import_blocks` copies them into the persistent page table before each replay
    (row-major memcpy, like the payload). Replaces 32 host uploads + ~64 eager ops with 1 (+1) uploads and one replay.
    Buckets above `kv_trace_max_bucket()` are imported chunk by chunk through the largest trace (the last chunk's
    missing blocks are pad rows aimed at the pad block).
    """

    def __init__(self, model):
        self.model = model
        self.mesh = model.mesh_device
        self.n_dev = int(model.num_devices)
        layers = _attention_layers(model)
        self.caches = [c for att in layers for c in (att.paged_k, att.paged_v)]
        self.n_caches = len(self.caches)
        num_blocks, nkv, blk, hd = self.caches[0].shape
        self.nkv, self.blk, self.hd = int(nkv), int(blk), int(hd)
        self.pad = _pad_block(model, self.caches[0])
        self.max_bucket = kv_trace_max_bucket()
        self.staging: dict[int, tuple] = {}  # bucket -> (kv_rm, page_table)
        self.traces: dict[int, int] = {}
        self.mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0)
        self.replicate = ttnn.ReplicateTensorToMesh(self.mesh)

    def _alloc(self, n: int):
        """Persistent staging + page table for bucket n (page table initialized to the pad block: the compile/capture
        passes write the zero staging there)."""
        if n in self.staging:
            return self.staging[n]
        kv_rm = ttnn.from_torch(
            torch.zeros(self.n_dev * self.n_caches, self.nkv, n * self.blk, self.hd, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mapper,
        )
        pt = ttnn.from_torch(
            torch.full((1, n), self.pad, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.replicate,
        )
        self.staging[n] = (kv_rm, pt)
        return self.staging[n]

    def _body(self, n: int):
        kv_rm, pt = self.staging[n]
        kv_t = ttnn.to_layout(kv_rm, ttnn.TILE_LAYOUT)  # per device [n_caches, nkv, n*blk, hd]
        for i, cache in enumerate(self.caches):
            x = ttnn.slice(kv_t, (i, 0, 0, 0), (i + 1, self.nkv, n * self.blk, self.hd))
            if x.dtype != cache.dtype:
                xc = ttnn.typecast(x, cache.dtype)
                ttnn.deallocate(x)
                x = xc
            ttnn.experimental.paged_fill_cache(cache, x, pt, batch_idx=0)
            ttnn.deallocate(x)
        ttnn.deallocate(kv_t)

    def capture(self, n: int):
        if n in self.traces:
            return
        self._alloc(n)
        t0 = time.perf_counter()
        self._body(n)  # compile pass (writes the zero staging into the pad block)
        ttnn.synchronize_device(self.mesh)
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._body(n)
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        self.traces[n] = tid
        logger.info(f"[pd] captured KV import trace for bucket {n} in {1e3 * (time.perf_counter() - t0):.0f} ms")

    def warmup(self, max_bucket: int = 2048):
        """Allocate the staging and capture the trace of every bucket <= min(max_bucket, max traced bucket)."""
        for b in _EXPORT_BUCKETS:
            if b > max_bucket or b > self.max_bucket:
                break
            self.capture(b)

    def import_blocks(self, block_ids, kv):
        t0 = time.perf_counter()
        prep = prepare_kv_import(self.model, kv)
        n_real = len(block_ids)
        if prep.n_real != n_real:
            raise ValueError(f"{prep.n_real} blocks in payload vs {n_real} block ids")
        chunk = prep.chunk
        ids = [int(b) for b in block_ids] + [self.pad] * (prep.n_chunks * chunk - n_real)
        t1 = time.perf_counter()
        for c in range(prep.n_chunks):
            kv_rm, pt = self._alloc(chunk)
            h = ttnn.from_torch(
                prep.host[c], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mapper
            )
            ttnn.copy_host_to_device_tensor(h, kv_rm)
            p = ttnn.from_torch(
                torch.tensor([ids[c * chunk : (c + 1) * chunk]], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.replicate,
            )
            ttnn.copy_host_to_device_tensor(p, pt)
            if chunk not in self.traces:
                self.capture(chunk)
            ttnn.execute_trace(self.mesh, self.traces[chunk], cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)  # the staging is rewritten by the next chunk / import
            del h, p
        logger.debug(
            f"[pd] imported {n_real} KV blocks (bucket {prep.bucket}) x {self.n_caches // 2} layers in "
            f"{1e3 * (time.perf_counter() - t0):.1f} ms (trace, {prep.n_chunks} chunk(s) of {chunk}; host prep "
            f"{1e3 * (t1 - t0):.1f} ms)"
        )


def get_traced_kv_importer(model) -> "TracedKvImporter":
    imp = getattr(model, "pd_kv_importer", None)
    if imp is None:
        imp = model.pd_kv_importer = TracedKvImporter(model)
    return imp


# --------------------------------------------------------------------------------------
# GDN snapshot: host staging pool (prefill side) + layout helpers
# --------------------------------------------------------------------------------------

_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def gdn_snapshot_dims(model):
    """(n_dev, L, K, (Nv, Dk, Dv), C, rec ttnn dtype, taps ttnn dtype) of the model's GDN state."""
    dn_layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
    dn0 = dn_layers[0]
    rec_shape = tuple(dn0.rec_state.shape)  # [B, Nv, Dk, Dv]
    return (
        model.num_devices,
        len(dn_layers),
        dn0.K,
        (rec_shape[1], rec_shape[2], rec_shape[3]),
        int(dn0.conv_states[0].shape[-1]),
        dn0.rec_state.dtype,
        dn0.conv_states[0].dtype,
    )


class GdnSnapshotPool:
    """Reusable host buffers for the prefill-side GDN snapshot, read by direct DMA.

    Each entry is a pair of torch buffers, `rec` `[n_dev*L, Nv, Dk, Dv]` and `taps` `[n_dev*L*K, 1, C]`, wrapped
    once as ROW_MAJOR host mesh tensors sharded on dim 0 (ttnn.from_torch borrows the torch memory, so the
    ttnn tensor is a view). `read` copies the device-untilized snapshot tensors into an entry with
    `ttnn.copy_device_to_host_tensor` -- no mesh composer, no host concat -- and returns the device-major views
    `rec [n_dev, L, Nv, Dk, Dv]`, `taps [n_dev, L, K, C]`. Reusing an entry keeps its pages faulted in and its
    pinned-memory mapping cached (measured on P150x4: 151 MB fp32 in ~6 ms reused vs ~42 ms into a fresh
    buffer vs ~103 ms through the composer), so consumers return entries with `release` once they have copied
    or uploaded the snapshot. The pool grows to the number of snapshots alive at once (P: one per request of a
    prefill step until the connector stages it).
    """

    def __init__(self, model):
        self.model = model
        n_dev, L, K, (Nv, Dk, Dv), C, rec_dtype, taps_dtype = gdn_snapshot_dims(model)
        self.n_dev, self.L, self.K, self.C = n_dev, L, K, C
        self.rec_shape = (n_dev * L, Nv, Dk, Dv)
        self.taps_shape = (n_dev * L * K, 1, C)
        self.rec_dtype, self.taps_dtype = rec_dtype, taps_dtype
        if rec_dtype not in _TORCH_DTYPE or taps_dtype not in _TORCH_DTYPE:
            raise ValueError(f"GdnSnapshotPool: unsupported GDN state dtypes rec={rec_dtype} taps={taps_dtype}")
        self.mapper = ttnn.ShardTensorToMesh(model.mesh_device, dim=0)
        self._free = []  # entries: (rec_host, rec_tt, taps_host, taps_tt)
        self._busy = {}  # rec_host.data_ptr() -> entry
        self.total = 0

    def _new(self):
        rec_host = torch.empty(self.rec_shape, dtype=_TORCH_DTYPE[self.rec_dtype])
        taps_host = torch.empty(self.taps_shape, dtype=_TORCH_DTYPE[self.taps_dtype])
        rec_tt = ttnn.from_torch(rec_host, dtype=self.rec_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mapper)
        taps_tt = ttnn.from_torch(
            taps_host, dtype=self.taps_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mapper
        )
        self.total += 1
        logger.info(
            f"[pd] GDN snapshot pool: +1 buffer ({(rec_host.numel() * rec_host.element_size() + taps_host.numel() * taps_host.element_size()) / 2**20:.0f} MiB), "
            f"{self.total} total"
        )
        return (rec_host, rec_tt, taps_host, taps_tt)

    def read(self, rec_rm, taps_rm):
        """DMA the device tensors `rec_rm` (per device [L, Nv, Dk, Dv], ROW_MAJOR) and `taps_rm` (per device
        [L*K, 1, C], ROW_MAJOR) into a pool entry; returns the device-major host views (rec, taps)."""
        entry = self._free.pop() if self._free else self._new()
        rec_host, rec_tt, taps_host, taps_tt = entry
        try:
            ttnn.copy_device_to_host_tensor(rec_rm, rec_tt, blocking=True)
            ttnn.copy_device_to_host_tensor(taps_rm, taps_tt, blocking=True)
        except Exception:
            self._free.append(entry)  # a failed DMA must not strand a ~150 MB pinned entry
            raise
        self._busy[rec_host.data_ptr()] = entry
        return rec_host.view(self.n_dev, self.L, *self.rec_shape[1:]), taps_host.view(
            self.n_dev, self.L, self.K, self.C
        )

    def release(self, rec, taps=None):
        """Return the entry `rec` was read into. Unknown tensors are ignored (logged), never re-pooled."""
        entry = self._busy.pop(rec.data_ptr(), None) if isinstance(rec, torch.Tensor) else None
        if entry is None:
            logger.warning("[pd] GDN snapshot pool: release of a snapshot the pool did not hand out; ignored")
            return
        self._free.append(entry)


def as_device_major(rec_snap, conv_snap):
    """Normalize a GDN snapshot to the device-major pair (rec [n_dev, L, Nv, Dk, Dv], taps [n_dev, L, K, C]).
    Accepts that pair as-is, a `PreparedGdnImport` in the `conv_snap` position (host prep done ahead, e.g. on the
    connector's pull thread), or the per-layer lists (rec_snap[li] [n_dev, Nv, Dk, Dv], conv_snap[li][m]
    [n_dev, 1, C]) an older producer emits."""
    if isinstance(conv_snap, PreparedGdnImport):
        return conv_snap.rec, conv_snap.taps
    if isinstance(rec_snap, torch.Tensor) and isinstance(conv_snap, torch.Tensor):
        if rec_snap.dim() != 5 or conv_snap.dim() != 4:
            raise ValueError(
                f"GDN snapshot: expected rec [n_dev, L, Nv, Dk, Dv] and taps [n_dev, L, K, C], got "
                f"{tuple(rec_snap.shape)} and {tuple(conv_snap.shape)}"
            )
        return rec_snap, conv_snap
    rec = torch.stack(list(rec_snap), dim=1)  # [n_dev, L, Nv, Dk, Dv]
    taps = torch.stack([torch.stack([c.reshape(c.shape[0], -1) for c in taps_l], dim=1) for taps_l in conv_snap], dim=1)
    return rec, taps  # taps [n_dev, L, K, C]


def import_gdn_slot(model, slot, rec_snap, conv_snap, mode=None):
    """Write one request's GDN snapshot (device-major: `rec` host `[n_dev, L, Nv, Dk, Dv]`, `taps` host
    `[n_dev, L, K, C]`; the per-layer list form is accepted too) into decode `slot`.

    mode "trace" (default) = TracedGdnImporter: host memcpy into fixed row-major staging tensors + one replayed
    per-slot trace (tilize, fill_cache rows, tap row writes, packed-history row write for all layers). `conv_snap`
    may be a `PreparedGdnImport` (see `prepare_gdn_import`): the host-side preparation was then done ahead of time
    (on another thread) and this call only uploads + replays.
    mode "host" = the served prefill's own slot write (`_write_gdn_slot`: one from_torch + slice/concat/copy
    row write per tensor, ~4 s for 48 layers); "fillcache" (default, QWEN36_PD_GDN_IMPORT) = one batched
    upload of all layers, then per layer an in-place `ttnn.fill_cache` for the recurrent state row and the
    slice/concat/copy row write for the K conv taps (the `_write_index` form the host path uses -- a masked
    `where` does not land the row on TILE-padded taps), then the per-slot packed-history repack.
    """
    mode = mode or os.environ.get("QWEN36_PD_GDN_IMPORT", "trace")
    t0 = time.perf_counter()
    prepared = conv_snap if isinstance(conv_snap, PreparedGdnImport) else None
    rec_snap, conv_snap = as_device_major(rec_snap, conv_snap)
    if mode == "host":
        model._write_gdn_slot(int(slot), rec_snap, conv_snap)
    elif mode == "trace":
        get_traced_importer(model).import_slot(int(slot), rec_snap, conv_snap, prepared=prepared)
    else:
        _import_gdn_slot_fillcache(model, int(slot), rec_snap, conv_snap)
    logger.debug(f"[pd] imported GDN state into slot {slot} ({mode}) in {1e3 * (time.perf_counter() - t0):.1f} ms")


def _import_gdn_slot_fillcache(model, slot, rec, taps):
    mesh = model.mesh_device
    n_dev = model.num_devices
    mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
    dn_layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
    L = len(dn_layers)
    if rec.shape[1] != L or taps.shape[1] != L:
        raise ValueError(f"snapshot has {rec.shape[1]}/{taps.shape[1]} GDN layers, model {L}")
    K = dn_layers[0].K
    dn0 = dn_layers[0]
    # device-major snapshot: one upload per state type, no host reshuffle
    rec_all = rec.reshape(n_dev * L, *rec.shape[2:])  # [n_dev*L, Nv, Dk, Dv]
    rec_dev = _upload(model, rec_all, dn0.rec_state.dtype, mapper)  # per device [L, Nv, Dk, Dv]
    taps_all = taps.reshape(n_dev * L * K, 1, taps.shape[-1])  # [n_dev*L*K, 1, C]
    taps_dev = _upload(model, taps_all, dn0.conv_states[0].dtype, mapper)  # per device [L*K, 1, C]
    for li, dn in enumerate(dn_layers):
        rec_l = dn._slice_along(rec_dev, 0, li, li + 1)  # [1, Nv, Dk, Dv]
        if rec_l.dtype != dn.rec_state.dtype:
            rec_c = ttnn.typecast(rec_l, dn.rec_state.dtype)
            ttnn.deallocate(rec_l)
            rec_l = rec_c
        ttnn.fill_cache(dn.rec_state, rec_l, slot)  # in place: rec_state[slot] = rec_l
        ttnn.deallocate(rec_l)
        for m in range(dn.K):
            c = dn._slice_along(taps_dev, 0, li * K + m, li * K + m + 1)  # [1, 1, C]
            if c.dtype != dn.conv_states[m].dtype:
                c_c = ttnn.typecast(c, dn.conv_states[m].dtype)
                ttnn.deallocate(c)
                c = c_c
            dn._write_index(dn.conv_states[m], c, slot, dim=1)  # consumes c
        if dn.conv_hist_packed is not None and dn._hist_packed_valid:
            dn._sync_conv_hist_packed(slot=slot)
        else:
            dn._sync_conv_hist_packed()
    ttnn.deallocate(rec_dev)
    ttnn.deallocate(taps_dev)


def gdn_state_nbytes(rec_snap, conv_snap):
    rec, taps = as_device_major(rec_snap, conv_snap)
    return rec.numel() * rec.element_size() + taps.numel() * taps.element_size()


def kv_nbytes(kv):
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv)


# --------------------------------------------------------------------------------------
# traced per-slot GDN import
# --------------------------------------------------------------------------------------


class PreparedGdnImport:
    """A GDN snapshot with the D-side host preparation done: the borrowed-upload views of the staging row order plus
    the packed conv-history rows for BOTH slot parities (the decode slot is only known when the request is admitted).
    Built by `prepare_gdn_import` / `GdnHostPacker.prepare` (torch ops only, so it may run on the connector's pull
    thread); consumed by `import_gdn_slot(model, slot, rec, prepared)` (the `conv_snap` position). `rec` / `taps`
    are the device-major views the snapshot API documents, so `as_device_major`, `verify_gdn_slot` and
    `gdn_state_nbytes` take a prepared import as-is."""

    __slots__ = ("rec", "taps", "rec_rows", "taps_rows", "hist")

    def __init__(self, rec, taps, rec_rows, taps_rows, hist):
        self.rec = rec  # [n_dev, L, Nv, Dk, Dv]
        self.taps = taps  # [n_dev, L, K, C]
        self.rec_rows = rec_rows  # [n_dev*L, Nv, Dk, Dv] view of rec (staging row order)
        self.taps_rows = taps_rows  # [n_dev*L*K, 1, C] view of taps
        self.hist = hist  # [2, n_dev*L, Nv*4, 32, 32] bf16 packed history per parity, or None (no packed buffer)


class GdnHostPacker:
    """Host-only half of the traced GDN import: the model's per-device GDN geometry and the vectorized builder of the
    packed conv-history rows (`_pack_head_tiles` for all layers/devices at once). No device tensors, no ttnn calls after
    construction -- safe to use from a worker thread once built (build it on the main thread: `get_gdn_host_packer`)."""

    def __init__(self, model):
        self.dn = [layer.attention for layer in model.layers if not layer.is_full_attention]
        dn0 = self.dn[0]
        self.n_dev, self.L, self.K, (self.Nv, self.Dk, self.Dv), self.C, self.rec_dtype, _ = gdn_snapshot_dims(model)
        self.with_hist = dn0.conv_hist_packed is not None
        Nv, Nk, Dk, Dv = dn0.Nv, dn0.Nk, dn0.Dk, dn0.Dv
        rf, kd = Nv // Nk, Nk * Dk
        rows = []
        for h in range(Nv):
            hk = h // rf
            rows.append(
                torch.cat(
                    [
                        torch.arange(hk * Dk, (hk + 1) * Dk),
                        torch.arange(kd + hk * Dk, kd + (hk + 1) * Dk),
                        torch.arange(2 * kd + h * Dv, 2 * kd + (h + 1) * Dv),
                    ]
                )
            )
        # [Nv, 3*Dk] channel indices: head h reads its q chunk (kv head hk), k chunk and v chunk of a [C] row -- the
        # vectorized form of _pack_head_tiles' per-head concat
        self.gidx = torch.stack(rows)
        self.n_chunks = self.gidx.shape[1] // 32
        self._verified_pack = False
        self._lock = threading.Lock()

    def rows(self, rec, taps):
        """The borrowed-upload views (staging row order) of a device-major snapshot."""
        rec_rows = rec.reshape(self.n_dev * self.L, self.Nv, self.Dk, self.Dv)
        taps_rows = taps.reshape(self.n_dev * self.L * self.K, 1, self.C)
        return rec_rows, taps_rows

    def gathered(self, taps):
        """taps [n_dev, L, K, C] -> the per-head channel chunks [n_dev, L, Nv, K, n_chunks, 32] bf16 (the expensive
        gather; parity-independent)."""
        n = self.n_chunks
        g = taps[..., self.gidx].to(torch.bfloat16).reshape(self.n_dev, self.L, self.K, self.Nv, n, 32)
        return g.permute(0, 1, 3, 2, 4, 5)  # [n_dev, L, Nv, K, n, 32]

    def _scatter(self, g, par, out):
        # channel chunk c of tap j lands at tile row 2c + parity, tile j (see _pack_head_tiles)
        n = self.n_chunks
        out[..., par : 2 * n + par : 2, :] = g

    def hist(self, taps, parity, g=None):
        """Packed history rows of every layer/device at `parity`: [n_dev*L, Nv*4, 32, 32] bf16, contiguous."""
        g = self.gathered(taps) if g is None else g
        out = torch.zeros(self.n_dev, self.L, self.Nv, 4, 32, 32, dtype=torch.bfloat16)
        self._scatter(g, parity, out)
        self._verify(taps, out, parity)
        return out.reshape(self.n_dev * self.L, self.Nv * 4, 32, 32)

    def hist_both(self, taps):
        """Both parities from one gather: [2, n_dev*L, Nv*4, 32, 32] (index [slot & 1] is contiguous)."""
        g = self.gathered(taps)
        out = torch.zeros(2, self.n_dev, self.L, self.Nv, 4, 32, 32, dtype=torch.bfloat16)
        for par in (0, 1):
            self._scatter(g, par, out[par])
        self._verify(taps, out[0], 0)
        return out.reshape(2, self.n_dev * self.L, self.Nv * 4, 32, 32)

    def _verify(self, taps, out, par):
        """One-time check of the vectorized layout against the layer's own scalar packer (host torch only)."""
        if self._verified_pack:
            return
        with self._lock:
            if self._verified_pack:
                return
            ref = self.dn[0]._pack_head_tiles([taps[0, 0, j].reshape(-1) for j in range(self.K)], parity=par)
            if not torch.equal(ref, out[0, 0]):
                raise RuntimeError("vectorized packed-history layout differs from _pack_head_tiles")
            self._verified_pack = True

    def prepare(self, rec, taps) -> PreparedGdnImport:
        """Everything the import needs from the host, for any slot: views in staging row order + both parities of the
        packed history (when the layers carry a packed buffer)."""
        rec, taps = as_device_major(rec, taps)
        rec_rows, taps_rows = self.rows(rec, taps)
        return PreparedGdnImport(rec, taps, rec_rows, taps_rows, self.hist_both(taps) if self.with_hist else None)


def get_gdn_host_packer(model) -> "GdnHostPacker":
    """The model's GdnHostPacker (built on first use; build it on the main thread before handing it to workers)."""
    packer = getattr(model, "pd_gdn_host_packer", None)
    if packer is None:
        packer = model.pd_gdn_host_packer = GdnHostPacker(model)
    return packer


def prepare_gdn_import(model, rec_snap, conv_snap) -> PreparedGdnImport:
    """Host-side preparation of one request's GDN snapshot for `import_gdn_slot` (torch ops only: may run on any thread,
    e.g. the connector's pull worker, once `get_gdn_host_packer(model)` was called on the main thread). Pass the result
    as `import_gdn_slot`'s `conv_snap`."""
    return get_gdn_host_packer(model).prepare(rec_snap, conv_snap)


class TracedGdnImporter:
    """Write a request's GDN snapshot into decode slot ``slot`` with one trace replay.

    The host copies the snapshot into fixed ROW_MAJOR staging tensors (rec fp32, taps bf16, packed history bf16 -- the
    packed rows are built on the host by `GdnHostPacker`, parity ``slot & 1``) and a per-slot trace does tilize + the
    row writes on device, all IN PLACE at the decode buffers' trace-baked addresses:

    * recurrent state: ``ttnn.fill_cache(rec_state, rec_l, slot)`` (rec_state ``[B, Nv, Dk, Dv]``, rec_l ``[1, Nv, Dk, Dv]``);
    * packed conv history: ``ttnn.fill_cache`` into the ``[B, Nv*4, 32, 32]`` view of conv_hist_packed (a true view:
      the last two dims are unchanged), the way ``_sync_conv_hist_packed_device`` writes one slot;
    * conv taps: ``ttnn.where(onehot[slot], tap_row, conv_states[m], output_tensor=conv_states[m])`` -- a masked
      select over the ``[1, B, C]`` tap buffer with the ``[1, 1, C]`` tap row broadcast along B and the buffer itself
      as the (aliased) output, so row ``slot`` takes the new taps and every other row is forwarded unchanged. The
      one-hot row masks (one ``[1, B, C]`` bf16 tensor per slot, 160 KiB each) are persistent, allocated here.
      (``ttnn.update_cache`` was tried first: its program buffers ``32 x C/32`` tiles = 5 MiB of L1 at C = 2560.)

    That is 2 + 2 + 2K = 12 device ops per layer (slice + write each), ~580 per import, vs ~51 per layer before
    (slice/concat/copy rewrites of the WHOLE 3 MB packed-history buffer and of every [1, B, C] tap buffer, the latter
    through untilize/tilize round trips on the tile-padded batch dim). Bytes landing in the slot rows are identical:
    fill_cache and where forward the (already device-tilized, hence canonical) bf16/fp32 values untouched, exactly as
    the old slice/concat/copy did.
    Traces are captured lazily per slot (or up front via ``precapture``); slot indices are baked into them.
    """

    def __init__(self, model):
        self.model = model
        self.mesh = model.mesh_device
        self.n_dev = model.num_devices
        self.packer = get_gdn_host_packer(model)
        self.dn = self.packer.dn
        dn0 = self.dn[0]
        self.L, self.K = self.packer.L, self.packer.K
        self.Nv, self.Dk, self.Dv, self.C = self.packer.Nv, self.packer.Dk, self.packer.Dv, self.packer.C
        self.rec_dtype = dn0.rec_state.dtype
        self.with_hist = dn0.conv_hist_packed is not None
        mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0)

        def stage(shape, torch_dtype, tt_dtype):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch_dtype),
                dtype=tt_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        rec_torch = torch.float32 if self.rec_dtype == ttnn.float32 else torch.bfloat16
        self.rec_rm = stage((self.n_dev * self.L, self.Nv, self.Dk, self.Dv), rec_torch, self.rec_dtype)
        self.taps_rm = stage((self.n_dev * self.L * self.K, 1, self.C), torch.bfloat16, ttnn.bfloat16)
        # 4-D: [L, Nv*4, 32, 32] per device, so a layer's slice is directly fill_cache's [1, Nv*4, 32, 32] input
        self.hist_rm = (
            stage((self.n_dev * self.L, self.Nv * 4, 32, 32), torch.bfloat16, ttnn.bfloat16) if self.with_hist else None
        )
        # one-hot row masks for the tap writes: mask[s][0, b, :] = (b == s), same [1, B, C] shape as conv_states[m]
        B = int(dn0.conv_states[0].shape[-2])
        self.B = B
        eye = torch.eye(B, dtype=torch.bfloat16)
        self.masks = [
            ttnn.from_torch(
                eye[s].reshape(1, B, 1).expand(1, B, self.C).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            for s in range(B)
        ]
        self.traces: dict[int, int] = {}
        self.mapper = mapper
        logger.info(
            f"[pd] TracedGdnImporter: L={self.L} K={self.K} B={B} rec={self.rec_dtype} hist={'on' if self.with_hist else 'off'}; "
            f"staging {self.n_dev * self.L * self.Nv * self.Dk * self.Dv * (4 if rec_torch == torch.float32 else 2) / 2**20:.0f} MiB rec"
        )

    # -- device body (trace-capturable: no host transfers) --
    def _body(self, slot: int):
        rec_t = ttnn.to_layout(self.rec_rm, ttnn.TILE_LAYOUT)
        taps_t = ttnn.to_layout(self.taps_rm, ttnn.TILE_LAYOUT)
        hist_t = ttnn.to_layout(self.hist_rm, ttnn.TILE_LAYOUT) if self.with_hist else None
        K = self.K
        for li, dn in enumerate(self.dn):
            rec_l = dn._slice_along(rec_t, 0, li, li + 1)  # [1, Nv, Dk, Dv]
            ttnn.fill_cache(dn.rec_state, rec_l, slot)  # in place: rec_state[slot] = rec_l
            ttnn.deallocate(rec_l)
            for m in range(K):
                c = dn._slice_along(taps_t, 0, li * K + m, li * K + m + 1)  # [1, 1, C]
                self._write_tap_row(dn.conv_states[m], c, slot)  # consumes c
            if hist_t is not None and dn.conv_hist_packed is not None:
                h = dn._slice_along(hist_t, 0, li, li + 1)  # [1, Nv*4, 32, 32]
                B = dn.conv_hist_packed.shape[0]
                dst = ttnn.reshape(dn.conv_hist_packed, (B, self.Nv * 4, 32, 32))  # view of the trace-baked buffer
                ttnn.fill_cache(dst, h, slot)  # in place: conv_hist_packed[slot] = h
                ttnn.deallocate(h)  # dst is a reshape view: never deallocated
        ttnn.deallocate(rec_t)
        ttnn.deallocate(taps_t)
        if hist_t is not None:
            ttnn.deallocate(hist_t)

    def _write_tap_row(self, conv_state, c, slot: int):
        """conv_state[0, slot, :] = c[0, 0, :] in place (conv_state: the layer's [1, B, C] TILE tap buffer whose address
        the decode trace baked; c: a [1, 1, C] TILE row, consumed). One masked select with the buffer as its own output:
        rows != slot are forwarded, row slot takes the broadcast tap row. Elementwise, tile by tile, so aliasing the
        false-operand and the output is safe (the in-place binary ops work the same way)."""
        assert 0 <= slot < self.B, f"slot {slot} out of range [0,{self.B})"
        ttnn.where(self.masks[slot], c, conv_state, output_tensor=conv_state)
        ttnn.deallocate(c)

    def capture(self, slot: int):
        if slot in self.traces:
            return
        t0 = time.perf_counter()
        self._body(slot)  # compile pass (also a harmless write of the staged data into the slot)
        ttnn.synchronize_device(self.mesh)
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._body(slot)
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        self.traces[slot] = tid
        logger.info(f"[pd] captured GDN import trace for slot {slot} in {1e3 * (time.perf_counter() - t0):.0f} ms")

    def precapture(self, slots):
        for s in slots:
            self.capture(int(s))

    # -- host side --
    def prepare(self, rec, taps) -> PreparedGdnImport:
        """Host preparation for any slot (see GdnHostPacker.prepare); may run on another thread."""
        return self.packer.prepare(rec, taps)

    def _host_hist(self, taps, slot):
        """Packed history rows for all layers/devices at parity slot & 1: [n_dev*L, Nv*4, 32, 32] bf16."""
        return self.packer.hist(taps, slot & 1)

    def _upload(self, host, dst):
        h = ttnn.from_torch(host, dtype=dst.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=self.mapper)
        ttnn.copy_host_to_device_tensor(h, dst)
        return h  # keep alive until the replay is synchronized

    def import_slot(self, slot: int, rec, taps, prepared: "PreparedGdnImport | None" = None):
        """rec: host [n_dev, L, Nv, Dk, Dv]; taps: host [n_dev, L, K, C] (device-major, see module docstring).
        Both are already in the staging tensors' row order, so the uploads are borrowed views (no host copy). With
        `prepared` (a PreparedGdnImport of the same snapshot) the packed-history rows were built ahead of time and the
        main thread only uploads and replays."""
        t0 = time.perf_counter()
        if prepared is None:
            rec_all, taps_all = self.packer.rows(rec, taps)
            hist = self._host_hist(taps, slot) if self.with_hist else None
        else:
            rec_all, taps_all = prepared.rec_rows, prepared.taps_rows
            hist = prepared.hist[slot & 1] if self.with_hist else None
            if self.with_hist and hist is None:
                hist = self._host_hist(taps, slot)
        refs = [self._upload(rec_all, self.rec_rm), self._upload(taps_all, self.taps_rm)]
        if self.with_hist:
            refs.append(self._upload(hist, self.hist_rm))
        t1 = time.perf_counter()
        if slot not in self.traces:
            self.capture(slot)
        ttnn.execute_trace(self.mesh, self.traces[slot], cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh)
        refs.clear()
        for dn in self.dn:
            dn._hist_packed_valid = dn._hist_packed_valid if dn.conv_hist_packed is None else True
        logger.debug(
            f"[pd] traced GDN import slot {slot}: host+upload {1e3 * (t1 - t0):.1f} ms"
            f"{' (prepared)' if prepared is not None else ''}, replay {1e3 * (time.perf_counter() - t1):.1f} ms"
        )


def get_traced_importer(model) -> "TracedGdnImporter":
    imp = getattr(model, "pd_gdn_importer", None)
    if imp is None:
        imp = model.pd_gdn_importer = TracedGdnImporter(model)
    return imp


def verify_gdn_slot(model, slot, rec_snap, conv_snap, tag=""):
    """Read back decode `slot` (recurrent state, conv taps, packed history) and compare with the snapshot."""
    rec_snap, taps_snap = as_device_major(rec_snap, conv_snap)
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
    dn_layers = [layer.attention for layer in model.layers if not layer.is_full_attention]
    n_dev = model.num_devices
    worst_rec, worst_tap, worst_hist = 0.0, 0.0, 0.0
    for li in (0, len(dn_layers) // 2, len(dn_layers) - 1):
        dn = dn_layers[li]
        r = dn._slice_along(dn.rec_state, 0, slot, slot + 1)
        got = ttnn.to_torch(r, mesh_composer=comp).float()  # [n_dev, Nv, Dk, Dv]
        ttnn.deallocate(r)
        worst_rec = max(worst_rec, float((got - rec_snap[:, li].float()).abs().max()))
        for m in range(dn.K):
            c = dn._slice_along(dn.conv_states[m], 1, slot, slot + 1)
            gotc = ttnn.to_torch(c, mesh_composer=comp).float().reshape(n_dev, -1)  # [n_dev, C]
            ttnn.deallocate(c)
            worst_tap = max(worst_tap, float((gotc - taps_snap[:, li, m].float()).abs().max()))
        if dn.conv_hist_packed is not None:
            h = dn._slice_along(dn.conv_hist_packed, 0, slot, slot + 1)
            goth = ttnn.to_torch(h, mesh_composer=comp).to(torch.bfloat16)  # [n_dev, Nv, 4, 32, 32]
            ttnn.deallocate(h)
            ref = torch.stack(
                [
                    dn._pack_head_tiles([taps_snap[d, li, j].reshape(-1) for j in range(dn.K)], parity=slot & 1)
                    for d in range(n_dev)
                ]
            )
            worst_hist = max(worst_hist, float((goth.float() - ref.float()).abs().max()))
    logger.info(
        f"[pd] VERIFY slot {slot} {tag}: max|rec diff| {worst_rec:.3g}, max|tap diff| {worst_tap:.3g}, max|packed-hist diff| {worst_hist:.3g}"
    )
