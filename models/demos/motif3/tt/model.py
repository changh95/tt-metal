# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 model on the BH Galaxy: ``MotifModel`` = embedding -> N decoder layers -> final head (design §2.3,
§3.2-3.3; WAVE_A_REVIEW GEN-1..5, CONV-1..4).

Dataflow (README CONVENTIONS §2-3)::

    decode  tokens [4, 8] uint32 per DP row --embed--> X [1, 4, 8, 4096]
            rot = MotifAttention.decode_rope_tables(rope, rot_idxs)       (once per step, shared by all layers)
            active = MotifAttention.active_mask_from_cur_pos(cur_pos)
            for layer: X = layer.forward_decode(X, rot, cur_pos, page_table, kv[layer], active)
            logits = head.forward_decode(X, row_major=True)               [1, 1, 32, 6880] per chip ("mesh" split)
    prefill tokens [1|4, S] (one user, S = bucket, replicated) --embed--> X [1, 4, S, 4096]
            for layer: X = layer.forward_prefill(X, page_table [1, S / block], kv[layer])   (fills the cache)
            tile = head.forward_prefill(X, last_index)                    the last token's tile row [1, 1, 32, 6880]

Weights and the TT cache (README §7, design §2.3.11, CONV-1..4). Every module uploads through ``weights.as_tensor`` with
a lazy torch source, so a cached tensor never touches the safetensors; the source itself (:class:`LazySource`) opens
the checkpoint index only when a tensor is actually missing from the cache. ``cache`` selects the policy per part
(``global`` = embedding / final norm / LM head, and each decoder layer):

* ``"auto"`` (default) **never writes** into the cache (disk-safe: the full bfp8 cache is ~340 GB, the disk is shared
  with the BF16 download). A part whose completion marker exists (``weights.layer_cache_marker``, written by a
  converter run) is built under :func:`read_only_cache`: every tensor whose cache file exists loads from it, any other
  (an option variant the converter did not build -- e.g. ``layer_kwargs`` A/B knobs, ``vocab_split="tp"`` against
  converted ``"mesh"`` globals, a module cache-name version bump) is uploaded from the HF source without writing,
  listed in ``model.cache_misses`` and logged. Unmarked parts load from the HF source without writing.
* ``"write"`` / ``True``: read and write the cache, then mark each part complete (``weights.mark_layer_cached``, with
  the list of files; an existing marker is never replaced). An incomplete part is written only if the disk guard
  passes (below; else :class:`DiskGuardError`). :func:`convert_weights` is the minimal in-package converter (resumable
  per layer, seconds per layer logged, the same disk guard). The production converter is ``scripts/convert_weights.py``
  (staging + atomic moves, sha256, verification from the cache alone, option variants; ``demo.py --convert`` runs it);
  it builds the same parts with the same constructors (``MotifDecoderLayer``, ``MotifEmbedding``, ``MotifLMHead`` with
  ``cache=True``), so ``"auto"`` loads its output.
* ``"off"`` / ``False``: never use the cache (random weights).

Disk guard (CONV-3; the production scripts' rule): a cache write of a part, or a shard download (below), runs only if
its filesystem keeps :data:`MIN_FREE_GB` (60 GB, ``scripts/convert_weights.py`` ``MIN_FREE_GB``: the BF16
downloader's margin) free after :data:`GUARD_FACTOR` (1.1) x the bytes about to be written (part sizes:
:func:`estimate_part_bytes`, measured on layers 0-2 + globals). An uncached HF **repo id** (``GeneratorSettings.
weights_are_local`` False) is read through :class:`RepoShardSource`: only the shards of tensors that are actually read
are downloaded, one at a time, each after that guard -- never a ``snapshot_download`` of the 630 GB repo.

Layer subsets: ``layers`` (default ``range(cfg.num_layers)``) picks the decoder layers to build, in ascending order;
``cfg.num_layers`` must exceed the largest index (the layer specs come from the config). ``stop_after`` on
:meth:`prefill` / :meth:`decode` runs only the first k built layers (truncated-model checks against intermediate
goldens without building a second model).

KV caches (GEN-2): :meth:`allocate_kv_caches` returns one paged latent cache ``[num_blocks, 1, block, 576]`` per built
layer (``cfg.dtypes.kv_cache``, TILE, DRAM, replicated), allocated with ``ttnn.empty`` + on-device ``ttnn.fill(0)``
(G7: ``ttnn.zeros`` of the pool takes ~20 s); ``num_blocks`` / ``block_size`` are what ``allocate_kv_cache`` received.

Device memory (review finding 10; allocator view, :func:`device_bytes_per_chip`): ``model.part_dram_bytes`` records
the DRAM per chip each part takes -- measured: globals 1.86 GB, dense layer 0.054 GB, MoE layer 0.244 GB, KV 0.162 GB
per layer for the 4129-block pool; 53 layers + the serving pool project to 23.0 of 33.9 GB per chip, and the 53-call
proxy of ``tests/test_model_truncated.py`` ran a 32K prefill and a traced decode (44.9 MiB of the 256 MiB trace
region) in that state.

Import rule (design §2.1): ttnn, torch, stdlib and the motif3 ``tt/`` modules only; the checkpoint (safetensors) and
huggingface_hub are touched lazily, inside :class:`LazySource` / :class:`RepoShardSource`.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import ttnn

from . import weights as W
from .attention import MotifAttention
from .ccl import MotifCCL
from .decoder import MotifDecoderLayer, free_tensors
from .embedding import MotifEmbedding
from .lm_head import MotifLMHead
from .model_config import DEFAULT_HF_META_DIR, MotifTTConfig
from .rope import MotifRope

CACHE_POLICIES = ("auto", "write", "off")

# Disk guard of every write this package makes on its own (cache parts, repo-id shards): the production scripts' floor
# (scripts/convert_weights.py MIN_FREE_GB / GUARD_FACTOR; design §6 option B, CONV-3: the BF16 downloader's margin).
MIN_FREE_GB = 60.0
GUARD_FACTOR = 1.10
# Bytes of one converted part with the serving defaults (scripts/convert_weights.py EST_CORE_BYTES / EST_VARIANT_BYTES,
# measured 2026-10-02 on layers 0-2 + globals): core files, plus the variant files a cfg-default build adds.
EST_CORE_BYTES = {"global": 3_607_113_408, "dense": 355_152_448, "moe": 6_652_399_104}  # global: norm + head + embed
EST_MHC_CONSTS_BYTES = {"motif": 17_152, "stock": 35_840}
EST_EXACT_ROUTER_BYTES = 3_146_112  # moe.router.weight_fp32k_v1
LARGEST_SHARD_BYTES = 8_053_063_840  # the largest safetensors shard of the pinned revision (hf_meta/tree.json)


def _free(*ts) -> None:
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


def _log_default(msg: str) -> None:
    print(f"[motif3.model {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize_cache_policy(cache) -> str:
    if cache is True:
        return "write"
    if cache is False or cache is None:
        return "off"
    if cache not in CACHE_POLICIES:
        raise ValueError(f"cache must be one of {CACHE_POLICIES} or a bool, got {cache!r}")
    return cache


# ================================================================================================================
# disk guard (CONV-3)
# ================================================================================================================
class DiskGuardError(RuntimeError):
    """A write was refused: it would leave less than the free-space floor on its filesystem."""


def free_bytes(path) -> int:
    """Bytes available to this user on ``path``'s filesystem (``df`` Avail); ``path`` may not exist yet."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    st = os.statvfs(p)
    return int(st.f_bavail) * int(st.f_frsize)


def room_ok(free_b: int, need_b: int, min_free_gb: float) -> bool:
    """The guard: writing ``GUARD_FACTOR x need_b`` bytes leaves at least ``min_free_gb`` GB free."""
    return (int(free_b) - GUARD_FACTOR * int(need_b)) / 1e9 >= float(min_free_gb)


def estimate_part_bytes(cfg: MotifTTConfig, layer: Optional[int]) -> int:
    """Cache bytes of one part (``layer=None`` = the globals) as :func:`convert_weights` / ``"write"`` build it (the
    config's defaults: ``cfg.mhc_sinkhorn``, ``cfg.router_logits``, LM head ``"mesh"``, replicated embedding)."""
    if layer is None:
        return EST_CORE_BYTES["global"]
    kind = "moe" if cfg.layer(int(layer)).is_moe else "dense"
    n = EST_CORE_BYTES[kind] + EST_MHC_CONSTS_BYTES.get(str(cfg.mhc_sinkhorn), EST_MHC_CONSTS_BYTES["stock"])
    if kind == "moe" and cfg.router_logits == "exact_fp32":
        n += EST_EXACT_ROUTER_BYTES
    return int(n)


def check_room(path, need_b: int, what: str, *, min_free_gb: float = MIN_FREE_GB, hint: str = "") -> int:
    """Raise :class:`DiskGuardError` unless writing ``need_b`` bytes under ``path`` passes the guard; returns the free
    bytes."""
    free = free_bytes(path)
    if not room_ok(free, need_b, min_free_gb):
        raise DiskGuardError(
            f"disk guard: {what} needs {need_b / 1e9:.2f} GB (x{GUARD_FACTOR:g}) under {path}, which has "
            f"{free / 1e9:.1f} GB free; the floor is {min_free_gb:g} GB{(' -- ' + hint) if hint else ''}"
        )
    return free


# ================================================================================================================
# weight sources
# ================================================================================================================
def _tree_sizes() -> Dict[str, int]:
    """File sizes of the pinned revision (``hf_meta/tree.json``); empty when absent."""
    p = DEFAULT_HF_META_DIR / "tree.json"
    try:
        return {e["path"]: int(e["size"]) for e in json.loads(p.read_text()) if e.get("type") == "file"}
    except Exception:
        return {}


class RepoShardSource:
    """The checkpoint of an uncached HF repo id, fetched **shard by shard, on first use** (never the whole repo).

    Construction fetches only ``model.safetensors.index.json`` (``huggingface_hub.hf_hub_download`` into the HF hub
    cache at ``revision``); an ``HFWeightLoader`` reads that snapshot directory. Reading a tensor whose shard is not
    local downloads that one shard first, after the disk guard on the hub cache's filesystem (``min_free_gb`` free
    after 1.1 x the shard: size from ``hf_meta/tree.json``, else the hub's file metadata, else the largest shard of the
    pinned revision); a refused download raises :class:`DiskGuardError` naming the tensor, the shard and the way out.
    Every download is logged and listed in ``fetched``. ``has`` / ``available`` / ``layer_available`` / ``keys`` are the
    loader's (local view, no download). ``hub``: the ``huggingface_hub`` module (tests inject a fake)."""

    def __init__(self, repo_id: str, revision: Optional[str] = None, *, log: Callable[[str], None] = _log_default,
                 min_free_gb: float = MIN_FREE_GB, cache_dir=None, hub=None):
        self.repo_id, self.revision = str(repo_id), revision
        self._log = log
        self.min_free_gb = float(min_free_gb)
        self.cache_dir = cache_dir
        self._hub = hub
        self.fetched: List[str] = []
        self._sizes = _tree_sizes()
        idx = Path(self._download("model.safetensors.index.json"))
        self.dir = idx.parent
        self._loader = W.HFWeightLoader(self.dir)

    def _hf(self):
        if self._hub is None:
            import huggingface_hub  # lazy (import rule)

            self._hub = huggingface_hub
        return self._hub

    def _download(self, filename: str) -> str:
        return self._hf().hf_hub_download(repo_id=self.repo_id, filename=filename, revision=self.revision,
                                          cache_dir=self.cache_dir)

    def shard_bytes(self, fn: str) -> int:
        if fn in self._sizes:
            return self._sizes[fn]
        try:
            info = self._hf().HfApi().get_paths_info(self.repo_id, [fn], revision=self.revision)
            if info and getattr(info[0], "size", None):
                return int(info[0].size)
        except Exception:
            pass
        return LARGEST_SHARD_BYTES

    def _ensure_local(self, name: str) -> None:
        if name not in self._loader or self._loader.available(name):
            return  # unknown names raise in the loader; local shards need nothing
        fn = self._loader.shard_of(name)
        need = self.shard_bytes(fn)
        free = check_room(
            self.dir, need, f"downloading shard {fn} of {self.repo_id}@{self.revision or 'main'} (for {name})",
            min_free_gb=self.min_free_gb,
            hint="download the checkpoint with scripts/download_weights.py (it keeps its own margin) and point "
                 "MOTIF3_WEIGHTS_DIR / HF_MODEL at it, or convert the part into the TT cache (scripts/convert_weights.py)",
        )
        self._log(f"downloading shard {fn} ({need / 1e9:.2f} GB) of {self.repo_id}@{self.revision or 'main'}: {name} is "
                  f"not in the TT cache ({free / 1e9:.1f} GB free under {self.dir})")
        self._download(fn)
        if not self._loader.shard_present(fn):
            raise W.MissingWeightError(f"shard {fn} is missing or incomplete under {self.dir} after its download")
        self.fetched.append(fn)

    def get(self, name: str, dtype=None):
        self._ensure_local(name)
        return self._loader.get(name, dtype)

    def get_rows(self, name: str, start: int, stop: int, dtype=None):
        self._ensure_local(name)
        return self._loader.get_rows(name, start, stop, dtype)

    def shape(self, name: str):
        self._ensure_local(name)
        return self._loader.shape(name)

    def __contains__(self, name):
        return name in self._loader

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._loader, name)


class LazySource:
    """A weight source that resolves the checkpoint on first use (so a fully cached model never opens it).

    ``weights_dir``: a local checkpoint directory (``HFWeightLoader``); ``repo_id`` (+ ``revision``): an uncached HF
    repo id (``GeneratorSettings.weights_are_local`` False), read through :class:`RepoShardSource` (only the shards of
    the tensors actually read are downloaded, each after the disk guard). Exposes the ``HFWeightLoader`` API
    (``get``, ``get_rows``, ``available``, ``layer_available``, ...)."""

    def __init__(self, weights_dir=None, *, repo_id: Optional[str] = None, revision: Optional[str] = None,
                 log: Callable[[str], None] = _log_default, min_free_gb: float = MIN_FREE_GB, hub=None):
        self.weights_dir = None if weights_dir is None else Path(weights_dir)
        self.repo_id, self.revision = repo_id, revision
        self.min_free_gb = float(min_free_gb)
        self._hub = hub
        self._src = None
        self._log = log
        self.opened = False

    def _resolve(self):
        if self._src is None:
            if self.weights_dir is None and self.repo_id:
                self._log(f"{self.repo_id}@{self.revision or 'main'} is not local: tensors missing from the TT cache are "
                          f"downloaded shard by shard (disk floor {self.min_free_gb:g} GB)")
                self._src = RepoShardSource(self.repo_id, self.revision, log=self._log, min_free_gb=self.min_free_gb,
                                            hub=self._hub)
            else:
                self._src = W.HFWeightLoader(self.weights_dir)
            self.opened = True
        return self._src

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)

    def __contains__(self, name):
        return name in self._resolve()


def device_bytes_per_chip(mesh_device, buffer_type=None) -> Optional[Dict[str, int]]:
    """Allocator bookkeeping of one buffer type (default DRAM) per chip -- mesh buffers are allocated identically on
    every chip: ``{"allocated", "free", "total", "largest_free"}`` bytes (bank values x banks), or None when the
    allocator cannot be queried. No device traffic. ``ttnn.BufferType.TRACE`` gives the trace region's use."""
    bt = ttnn.BufferType.DRAM if buffer_type is None else buffer_type
    try:
        v = ttnn.get_memory_view(mesh_device, bt)
        nb = int(v.num_banks)
        return {"allocated": int(v.total_bytes_allocated_per_bank) * nb, "free": int(v.total_bytes_free_per_bank) * nb,
                "total": int(v.total_bytes_per_bank) * nb,
                "largest_free": int(v.largest_contiguous_bytes_free_per_bank) * nb}
    except Exception:
        return None


# ================================================================================================================
# TT cache
# ================================================================================================================
def layer_cache_complete(cfg: MotifTTConfig, layer: Optional[int]) -> bool:
    """The converter marked this part (``layer=None`` = globals) complete in ``cfg.cache_dir``."""
    return W.layer_cache_marker(cfg, layer).is_file()


@contextlib.contextmanager
def read_only_cache():
    """Build modules against the TT cache **without writing it**: inside the context, ``weights.as_tensor`` loads a
    tensor whose cache file exists and uploads any other from its source with ``cache_name=None`` (nothing written).
    Yields the list of the missing file names (in call order).

    Implementation: every module uploads through the ``weights`` module attribute (``W.as_tensor``), which is swapped
    for the duration (single-threaded model build; restored on exit, also on error). A corrupt existing file is still
    rebuilt in place by ``as_tensor`` (same file, same size). Requested shared change: a ``read_only`` argument of
    ``weights.as_tensor`` would replace the swap."""
    orig = W.as_tensor
    misses: List[str] = []

    def as_tensor_read_only(src, *, mesh_device, cfg, dtype, layout=ttnn.TILE_LAYOUT, memory_config=None, dp_dim=None,
                            tp_dim=None, cache_name=None, layer=None):
        if cache_name is not None and not W.is_cached(cfg, cache_name, layer, dtype, layout, dp_dim, tp_dim):
            misses.append(W.tensorbin_path(W.cache_prefix(cfg, cache_name, layer, dp_dim, tp_dim), dtype, layout).name)
            cache_name = None
        return orig(src, mesh_device=mesh_device, cfg=cfg, dtype=dtype, layout=layout, memory_config=memory_config,
                    dp_dim=dp_dim, tp_dim=tp_dim, cache_name=cache_name, layer=layer)

    W.as_tensor = as_tensor_read_only
    try:
        yield misses
    finally:
        W.as_tensor = orig


def _cache_files(cfg: MotifTTConfig, layer: Optional[int]) -> List[str]:
    d = cfg.cache_dir / ("global" if layer is None else f"L{int(layer):02d}")
    return sorted(p.name for p in d.glob("*.tensorbin")) if d.is_dir() else []


class MotifKVPool:
    """The paged latent pool: one cache ``[num_blocks, 1, block_size, 576]`` per built decoder layer (in
    ``MotifModel.layer_ids`` order). ``layers[i]`` is the ttnn tensor of ``layer_ids[i]``."""

    def __init__(self, layers: List, layer_ids: Sequence[int], num_blocks: int, block_size: int, dtype):
        self.layers = list(layers)
        self.layer_ids = tuple(int(i) for i in layer_ids)
        self.num_blocks, self.block_size, self.dtype = int(num_blocks), int(block_size), dtype

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def deallocate(self) -> None:
        for t in self.layers:
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
        self.layers = []


class MotifModel:
    """Embedding + decoder layers + final head on one mesh (see the module docstring).

    Args:
        mesh_device: the opened mesh (L1_SMALL region required, ``model_config.require_l1_small``).
        cfg: :class:`MotifTTConfig` built with the mesh.
        source: weight source (``HFWeightLoader`` / ``DictWeightSource`` / :class:`LazySource`); default a
            :class:`LazySource` on ``cfg.weights_dir``.
        layers: decoder layer indices to build (ascending; default ``range(cfg.num_layers)``).
        cache: ``"auto"`` | ``"write"`` | ``"off"`` (or a bool), see the module docstring.
        vocab_split: LM-head split (``"mesh"`` default, decision EMB-D1; ``"tp"``).
        layer_kwargs: extra kwargs for every :class:`MotifDecoderLayer` (module A/B knobs).
        log: progress printer (layer load times).
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        *,
        source=None,
        layers: Optional[Iterable[int]] = None,
        cache="auto",
        ccl: Optional[MotifCCL] = None,
        rope: Optional[MotifRope] = None,
        vocab_split: str = "mesh",
        layer_kwargs: Optional[dict] = None,
        log: Optional[Callable[[str], None]] = _log_default,
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.log = log or (lambda m: None)
        self.cache_policy = normalize_cache_policy(cache)
        self.source = source if source is not None else LazySource(cfg.weights_dir, log=self.log)
        ids = list(range(cfg.num_layers)) if layers is None else sorted(int(i) for i in layers)
        if not ids:
            raise ValueError("a MotifModel needs at least one decoder layer")
        if len(set(ids)) != len(ids) or ids[0] < 0 or ids[-1] >= cfg.num_layers:
            raise ValueError(f"layers {ids} must be distinct indices in [0, cfg.num_layers={cfg.num_layers})")
        self.layer_ids = tuple(ids)
        self.ccl = ccl if ccl is not None else MotifCCL(mesh_device, cfg)
        self.rope = rope if rope is not None else MotifRope(mesh_device, cfg)
        self._rope_kinds = tuple(sorted({cfg.layer(l).rope_kind for l in ids}))
        self.load_seconds: Dict[str, float] = {}
        self.cache_misses: Dict[str, List[str]] = {}  # "auto": converted parts' tensors uploaded from the source
        self.part_dram_bytes: Dict[str, int] = {}  # device DRAM per chip each part's weights take (allocator view)

        def globals_(c):
            emb = MotifEmbedding(mesh_device, cfg, source=self.source, ccl=self.ccl, cache=c)
            head = MotifLMHead(mesh_device, cfg, source=self.source, ccl=self.ccl, cache=c, vocab_split=vocab_split)
            return emb, head

        t0 = time.time()
        d0 = device_bytes_per_chip(mesh_device)
        (self.embed, self.head), desc = self._build_part(None, globals_)
        self._note_dram("global", d0)
        self.load_seconds["global"] = time.time() - t0
        self.log(f"globals (embedding, final norm, LM head '{vocab_split}') loaded in {self.load_seconds['global']:.1f} s "
                 f"(TT cache: {desc})")
        self.layers: List[MotifDecoderLayer] = []
        for l in ids:
            t1 = time.time()
            d0 = device_bytes_per_chip(mesh_device)
            layer, desc = self._build_part(
                l, lambda c, l=l: MotifDecoderLayer(mesh_device, cfg, l, source=self.source, ccl=self.ccl,
                                                    rope=self.rope, cache=c, **dict(layer_kwargs or {})))
            self.layers.append(layer)
            self._note_dram(f"L{l:02d}", d0)
            self.load_seconds[f"L{l:02d}"] = time.time() - t1
            spec = cfg.layer(l)
            self.log(f"layer {l} ({spec.kind}) loaded in {self.load_seconds[f'L{l:02d}']:.1f} s (TT cache: {desc})")

    def _note_dram(self, tag: str, before: Optional[Dict[str, int]]) -> None:
        after = device_bytes_per_chip(self.mesh_device)
        if before is not None and after is not None:
            self.part_dram_bytes[tag] = after["allocated"] - before["allocated"]

    # ------------------------------------------------------------------------------------------------------------
    # cache policy
    # ------------------------------------------------------------------------------------------------------------
    def _build_part(self, layer: Optional[int], build: Callable[[bool], object]):
        """``build(cache_flag)`` for one part under the cache policy (module docstring); returns ``(part, desc)``."""
        cfg, tag = self.cfg, "global" if layer is None else f"L{int(layer):02d}"
        part_dir = cfg.cache_dir / tag
        complete = layer_cache_complete(cfg, layer)
        if self.cache_policy == "off":
            return build(False), "off"
        if self.cache_policy == "write":
            if not complete:  # an incomplete part is about to be written: guard the disk first
                check_room(cfg.cache_dir, estimate_part_bytes(cfg, layer), f"writing the TT cache part {tag}",
                           hint="free disk space, or build with cache='auto' (never writes)")
            obj = build(True)
            if not complete:  # never replace an existing marker (scripts/convert_weights.py writes richer ones)
                W.mark_layer_cached(cfg, layer, _cache_files(cfg, layer))
            return obj, f"{part_dir} (write)"
        if not complete:
            return build(False), "not converted (loaded from the HF source, nothing written)"
        with read_only_cache() as misses:
            obj = build(True)
        if misses:
            self.cache_misses[tag] = list(misses)
            return obj, (f"{part_dir}; {len(misses)} tensors not in the cache (option variants) uploaded from the HF "
                         f"source, nothing written: {', '.join(misses[:4])}{' ...' if len(misses) > 4 else ''}")
        return obj, str(part_dir)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    # ------------------------------------------------------------------------------------------------------------
    # KV pool (GEN-2)
    # ------------------------------------------------------------------------------------------------------------
    def allocate_kv_caches(self, num_blocks: int, block_size: int, dtype=None) -> MotifKVPool:
        """One zero-filled paged latent cache per built layer: ``ttnn.empty`` + on-device ``ttnn.fill(0)`` (G7)."""
        dtype = dtype if dtype is not None else self.cfg.dtypes.kv_cache
        shape = [int(num_blocks), 1, int(block_size), int(self.cfg.kv_latent_dim)]
        caches = []
        try:
            for _ in self.layer_ids:
                e = ttnn.empty(shape, dtype, ttnn.TILE_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG)
                caches.append(ttnn.fill(e, 0.0))
                ttnn.deallocate(e)
        except Exception:
            _free(*caches)
            raise
        return MotifKVPool(caches, self.layer_ids, num_blocks, block_size, dtype)

    # ------------------------------------------------------------------------------------------------------------
    # forwards
    # ------------------------------------------------------------------------------------------------------------
    def _check_kv(self, kv_caches, stop: int):
        if kv_caches is None:
            return [None] * stop
        if len(kv_caches) < stop:
            raise ValueError(f"{len(kv_caches)} KV caches for {stop} layers")
        return [kv_caches[i] for i in range(stop)]

    def _stop(self, stop_after: Optional[int]) -> int:
        n = len(self.layers) if stop_after is None else int(stop_after)
        if not 1 <= n <= len(self.layers):
            raise ValueError(f"stop_after {stop_after} outside [1, {len(self.layers)}]")
        return n

    def decode(self, tokens, *, rot_idxs, cur_pos, page_table, kv_caches, return_streams: bool = False,
               stop_after: Optional[int] = None):
        """One decode step for all 32 lanes (trace-safe).

        Args (persistent device inputs, README §3; per DP row): ``tokens [4, 8]`` uint32 ROW_MAJOR
        (``MotifEmbedding.decode_tokens_host``), ``rot_idxs [1, 32]`` uint32 (``MotifRope.rot_idxs_host``), ``cur_pos
        [8]`` int32 (-1 = inactive lane), ``page_table [8, W]`` int32, ``kv_caches`` (:class:`MotifKVPool` or a list,
        one per built layer).

        Returns the device logits ``[1, 1, 32, 6880]`` ROW_MAJOR per chip (read with ``head.logits_to_host``), or with
        ``return_streams`` the residual streams ``[1, 4, 8, 4096]`` after the last layer run."""
        n = self._stop(stop_after)
        kvs = self._check_kv(kv_caches, n)
        X = self.embed.forward_decode(tokens)
        rot = MotifAttention.decode_rope_tables(self.rope, rot_idxs, kinds=self._rope_kinds)
        act = MotifAttention.active_mask_from_cur_pos(cur_pos, self.cfg.lanes_per_row)
        try:
            for layer, kv in zip(self.layers[:n], kvs):
                Xn = layer.forward_decode(X, rot=rot, cur_pos=cur_pos, page_table=page_table, kv_cache=kv, active=act)
                _free(X)
                X = Xn
        finally:
            _free(act, *[t for cs in rot.values() for t in cs])
        if return_streams:
            return X
        logits = self.head.forward_decode(X, row_major=True)
        _free(X)
        return logits

    def prefill(self, tokens, *, page_table=None, kv_caches=None, last_index: Optional[int] = None,
                return_streams: bool = False, stop_after: Optional[int] = None):
        """Prefill of one user (eager). ``tokens``: device ``[1, S]`` / ``[4, S]`` uint32 (``embed.prefill_tokens_device``,
        S = bucket); ``page_table`` ``[1, cfg.prefill_page_table_entries(S)]`` int32 + ``kv_caches`` (or both None: no
        cache fill); ``last_index`` = S_real - 1. Returns the last token's logits tile row ``[1, 1, 32, Vc]`` ROW_MAJOR
        (``head.prefill_logits_to_host(tile, last_index)``), or with ``return_streams`` the streams ``[1, 4, S, 4096]``."""
        n = self._stop(stop_after)
        kvs = self._check_kv(kv_caches, n)
        X = self.embed.forward_prefill(tokens)
        for layer, kv in zip(self.layers[:n], kvs):
            Xn = layer.forward_prefill(X, page_table=page_table if kv is not None else None, kv_cache=kv)
            _free(X)
            X = Xn
        if return_streams:
            return X
        if last_index is None:
            raise ValueError("prefill needs last_index (= prompt length - 1) for the logits")
        tile = self.head.forward_prefill(X, last_index)
        _free(X)
        return tile

    # ------------------------------------------------------------------------------------------------------------
    def deallocate(self) -> None:
        """Free every device weight of the model (embedding, layers, head, RoPE tables)."""
        for layer in self.layers:
            layer.deallocate()
        self.layers = []
        free_tensors(self.embed)
        free_tensors(self.head)
        self.head.close()
        self.rope.release_prefill_tables()
        free_tensors(self.rope)


def convert_weights(
    mesh_device,
    cfg: MotifTTConfig,
    *,
    source=None,
    layers: Optional[Iterable[int]] = None,
    include_globals: bool = True,
    log: Callable[[str], None] = _log_default,
    min_free_gb: float = MIN_FREE_GB,
) -> Dict[str, float]:
    """CONV-1: build the TT weight cache in ``cfg.cache_dir`` (same mesh shape and dtype policy as serving: the cache tag
    includes both), one part at a time, resumable: parts already marked complete are skipped, each finished part is
    marked with ``weights.mark_layer_cached``. Every module is built with ``cache=True`` (which writes exactly its cache
    files) and freed again; no transform is re-derived here. Returns seconds per converted part.

    Disk guard (CONV-3), before each part: the cache filesystem must keep ``min_free_gb`` (60 GB, the production
    scripts' floor) free after 1.1 x the part's size (:func:`estimate_part_bytes`: 6.65 GB per MoE layer, 3.6 GB for the
    globals); otherwise :class:`DiskGuardError` (its ``converted`` attribute holds the seconds of the parts done).
    The production converter, ``scripts/convert_weights.py``, adds staging, sha256 and verification."""
    source = source if source is not None else LazySource(cfg.weights_dir, log=log)
    ids = list(range(cfg.num_layers)) if layers is None else sorted(int(i) for i in layers)
    ccl = MotifCCL(mesh_device, cfg)
    rope = MotifRope(mesh_device, cfg)
    times: Dict[str, float] = {}
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    parts: List[Optional[int]] = ([None] if include_globals else []) + ids
    try:
        for part in parts:
            tag = "global" if part is None else f"L{part:02d}"
            if layer_cache_complete(cfg, part):
                log(f"convert: {tag} already complete, skipped")
                continue
            need = estimate_part_bytes(cfg, part)
            try:
                check_room(cfg.cache_dir, need, f"converting {tag}", min_free_gb=min_free_gb)
            except DiskGuardError as e:
                log(f"convert: STOP before {tag}: {e}")
                e.converted = dict(times)
                raise
            t0 = time.time()
            if part is None:
                emb = MotifEmbedding(mesh_device, cfg, source=source, ccl=ccl, cache=True)
                head = MotifLMHead(mesh_device, cfg, source=source, ccl=ccl, cache=True)
                free_tensors(emb)
                free_tensors(head)
                head.close()
            else:
                layer = MotifDecoderLayer(mesh_device, cfg, part, source=source, ccl=ccl, rope=rope, cache=True)
                layer.deallocate()
            W.mark_layer_cached(cfg, part, _cache_files(cfg, part))
            times[tag] = time.time() - t0
            log(f"convert: {tag} cached in {times[tag]:.1f} s ({free_bytes(cfg.cache_dir) / 1e9:.1f} GB free)")
    finally:
        rope.release_prefill_tables()
        free_tensors(rope)
    return times


__all__ = [
    "CACHE_POLICIES",
    "DiskGuardError",
    "GUARD_FACTOR",
    "LazySource",
    "MIN_FREE_GB",
    "MotifKVPool",
    "MotifModel",
    "RepoShardSource",
    "check_room",
    "convert_weights",
    "device_bytes_per_chip",
    "estimate_part_bytes",
    "free_bytes",
    "layer_cache_complete",
    "normalize_cache_policy",
    "read_only_cache",
    "room_ok",
]
