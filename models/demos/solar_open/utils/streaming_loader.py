# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Phase-2 streaming state-dict loader for Solar-Open-100B (DESIGN.md section 4.16).

Phase 1 loads the whole checkpoint with ``AutoModelForCausalLM.from_pretrained(dtype=torch.bfloat16)`` (393 GB peak
host RSS, one-time thanks to the ttnn weight cache). ``LazyStateDict`` is the alternative selected by
``SOLAR_OPEN_STREAMING_LOAD=1``: a lazy ``Mapping`` over the 42 bf16 safetensors shards that presents exactly the
contract-C1 key set (627 = 3 + 48 x 13 keys) to every module and materialises one tensor per ``__getitem__``, so the
peak host footprint is one layer's transients (~10-16 GB) and nothing outside ``ModelArgs.load_state_dict`` and the
two ``hasattr`` hooks in ``utils/substate.py`` had to change.

How it works:
  * ``weight_map`` from ``<snapshot>/model.safetensors.index.json`` gives the shard of every on-disk key; shard headers
    (u64 LE length + JSON with ``dtype`` / ``shape`` / ``data_offsets``) are parsed lazily and cached.
  * The 384 per-expert keys ``model.layers.{l}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`` of layer l are
    replaced by two virtual keys: ``model.layers.{l}.mlp.experts.gate_up_proj`` = stack_e(cat([gate_e, up_e], 0))
    -> [128, 2560, 4096] (gate rows first, NOT interleaved - exactly transformers' ``solar_open -> qwen2_moe``
    conversion) and ``model.layers.{l}.mlp.experts.down_proj`` = stack_e(down_e) -> [128, 4096, 1280]. Both are
    assembled straight into one preallocated buffer from the per-expert byte ranges (sorted by file offset, read on a
    small thread pool), so the read *is* the fusion and the peak is 1x the output.
  * Tensor bytes are ``os.preadv``-read into freshly allocated torch tensors (anonymous memory, no mmap:
    ``safe_open.get_tensor`` leaves file-backed pages mapped, which is how the 393 GB came about). Bit-exact with
    ``safe_open`` and with the phase-1 ``from_pretrained`` path (probe, 2026-09-07).
  * ``q_proj.weight`` / ``k_proj.weight`` are Meta-permuted on access with ``load_checkpoints.reverse_permute`` (the
    rule of ``convert_hf_qkv_to_meta_format``); ``e_score_correction_bias`` stays fp32; the phase-1 safety-net rule
    (fp32 non-bias stragglers -> bf16) is applied per tensor; everything else is returned as stored (bf16).
  * ``substate(key)`` / ``has_substate(key)`` return prefix *views* sharing the same store without materialising
    anything, so ``utils/substate.py`` delegates to them and ``Model`` / ``DecoderLayer`` / every weight consumer walks
    the loader exactly as it walks a dict.
  * "LRU of one layer": the loader keeps no tensor (every consumer reads each key once and frees it when its
    constructor returns); it keeps a *layer window* of <= 2 open shard fds plus an asynchronous
    ``posix_fadvise(WILLNEED)`` prefetch of the next group's byte ranges so the cold disk read of layer l+1 overlaps
    the bfp8 pack of layer l. ``close()`` drops the window (idempotent; a later access reopens).

Not supported by design: ``items()`` / ``values()`` / ``dict(lazy)`` / ``nn.Module.load_state_dict(lazy)`` would
materialise the whole checkpoint (~205 GB) - the HF reference generators keep the phase-1 path. ``==`` is identity.
"""

from __future__ import annotations

import json
import os
import re
import struct
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import torch
from loguru import logger

from models.tt_transformers.tt.load_checkpoints import reverse_permute

DESIGN_REFERENCE = "DESIGN.md section 4.16 (utils/streaming_loader.py::LazyStateDict, phase 2)"
INDEX_FILE = "model.safetensors.index.json"
# Key suffix of the router's selection bias - the only fp32 tensor of contract C1 (mirrors tt/model_config.py, which
# imports this module lazily; this module stays torch-only so tests can import it without ttnn).
ROUTER_BIAS_SUFFIX = "e_score_correction_bias"
GATE_UP_KEY = "gate_up_proj"
DOWN_KEY = "down_proj"
_EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")
_EXPERT_KEY_RE = re.compile(
    r"^(?P<base>.+\.mlp\.experts)\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_LAYER_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.")
EMBED_PREFIX = "model.embed_tokens."
LM_HEAD_PREFIX = "lm_head."

# Iteration order of the 13 keys of a layer (contract C1 listing order); unknown keys sort after these, alphabetically.
LAYER_KEY_ORDER = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate.weight",
    f"mlp.gate.{ROUTER_BIAS_SUFFIX}",
    f"mlp.experts.{GATE_UP_KEY}",
    f"mlp.experts.{DOWN_KEY}",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
)

_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

_DEFAULT_IO_THREADS = 4


def read_into(fd: int, dst: torch.Tensor, start: int, stop: int) -> int:
    """``preadv`` the byte range ``[start, stop)`` of ``fd`` into the contiguous tensor ``dst`` (zero-copy into its
    anonymous memory; loops over short reads). Returns the number of bytes read."""
    if not dst.is_contiguous():
        raise ValueError("read_into needs a contiguous destination")
    buf = dst.view(torch.uint8).reshape(-1).numpy()
    nbytes = stop - start
    if buf.shape[0] != nbytes:
        raise ValueError(f"destination holds {buf.shape[0]} bytes but the tensor on disk has {nbytes}")
    done = 0
    while done < nbytes:
        n = os.preadv(fd, [buf[done:]], start + done)
        if n <= 0:
            raise OSError(f"short read: {done} of {nbytes} bytes at offset {start}")
        done += n
    return nbytes


class _Entry(NamedTuple):
    dtype: torch.dtype
    shape: tuple
    start: int  # absolute byte offset in the shard file
    stop: int
    dtype_name: str


class _ShardHeader:
    """Parsed safetensors header of one shard: key -> _Entry with absolute byte offsets."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        base = 8 + header_len
        self.entries: dict[str, _Entry] = {}
        for key, spec in header.items():
            if key == "__metadata__":
                continue
            dtype_name = spec["dtype"]
            if dtype_name not in _SAFETENSORS_DTYPES:
                raise ValueError(f"{path.name}: unsupported safetensors dtype {dtype_name!r} for {key}")
            start, stop = spec["data_offsets"]
            self.entries[key] = _Entry(
                _SAFETENSORS_DTYPES[dtype_name], tuple(spec["shape"]), base + start, base + stop, dtype_name
            )


def _resolve_snapshot_dir(snapshot_dir) -> Path:
    """A local directory holding the index, or an HF repo id resolved from the hub cache (HF_HUB_OFFLINE-friendly)."""
    path = Path(snapshot_dir)
    if path.is_dir():
        if not (path / INDEX_FILE).is_file():
            raise FileNotFoundError(
                f"{path / INDEX_FILE} not found: the streaming loader needs a sharded safetensors checkpoint "
                f"(see {DESIGN_REFERENCE})"
            )
        return path
    try:
        from transformers.utils.hub import cached_file

        resolved = cached_file(str(snapshot_dir), INDEX_FILE)
    except Exception as e:  # not a repo id, not cached, offline miss, ...
        raise FileNotFoundError(
            f"{str(snapshot_dir)!r} is neither a checkpoint directory nor an HF repo id resolvable from the hub cache "
            f"({type(e).__name__}: {e})"
        ) from e
    if not resolved:
        raise FileNotFoundError(f"{INDEX_FILE} of {str(snapshot_dir)!r} not found in the HF hub cache")
    # Do not resolve() symlinks: the index symlink points into blobs/, the shards live next to the symlink.
    return Path(resolved).parent


def _sort_key(key: str):
    m = _LAYER_KEY_RE.match(key)
    if m:
        rel = key[m.end() :]
        rank = LAYER_KEY_ORDER.index(rel) if rel in LAYER_KEY_ORDER else len(LAYER_KEY_ORDER)
        return (1, int(m.group(1)), rank, rel)
    if key.startswith(EMBED_PREFIX):
        return (0, 0, 0, key)
    if key.startswith(LM_HEAD_PREFIX):
        return (3, 0, 0, key)
    return (2, 0, 0, key)


def _group_of(key: str) -> tuple:
    """Layer window group of a key: ("embed",), ("layer", l) or ("tail",) (final norm + lm_head)."""
    m = _LAYER_KEY_RE.match(key)
    if m:
        return ("layer", int(m.group(1)))
    if key.startswith(EMBED_PREFIX):
        return ("embed",)
    return ("tail",)


class _Store:
    """State shared by a root ``LazyStateDict`` and all of its prefix views: index, virtual key set, headers, fds,
    layer window, thread pool and statistics."""

    def __init__(self, snapshot_dir, head_dim, num_experts, convert_to_meta, io_threads, prefetch_next_layer):
        self.snapshot_dir = _resolve_snapshot_dir(snapshot_dir)
        index = json.loads((self.snapshot_dir / INDEX_FILE).read_text())
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        if head_dim is None or num_experts is None:
            config = json.loads((self.snapshot_dir / "config.json").read_text())
            head_dim = head_dim if head_dim is not None else config["head_dim"]
            num_experts = num_experts if num_experts is not None else config["num_local_experts"]
        self.head_dim = int(head_dim)
        self.num_experts = int(num_experts)
        self.convert_to_meta = bool(convert_to_meta)
        if io_threads is None:
            io_threads = int(os.getenv("SOLAR_OPEN_STREAMING_THREADS", _DEFAULT_IO_THREADS))
        self.io_threads = max(1, int(io_threads))
        self.prefetch_next_layer = bool(prefetch_next_layer)
        # POSIX_FADV_DONTNEED on the previous group's byte ranges keeps the page cache from holding the whole 205 GB
        # checkpoint; off by default because it also evicts what a following phase-1 load / test run would reuse.
        self.dontneed = os.getenv("SOLAR_OPEN_STREAMING_DONTNEED", "0") == "1"

        self.fused: dict[str, str] = {}  # virtual key -> "...mlp.experts" base
        self.virtual_keys: list[str] = self._collapse_expert_keys()
        self.virtual_set = frozenset(self.virtual_keys)
        self.layer_ids = sorted({g[1] for g in map(_group_of, self.virtual_keys) if g[0] == "layer"})
        self._group_physical: dict[tuple, list[str]] = {}
        for key in self.weight_map:
            self._group_physical.setdefault(_group_of(key), []).append(key)

        self._headers: dict[str, _ShardHeader] = {}
        self._fds: dict[str, int] = {}
        self._pool: ThreadPoolExecutor | None = None
        self._advise_pool: ThreadPoolExecutor | None = None
        self._window: tuple | None = None
        self._prefetched: set[tuple] = set()
        self._read_keys: set[str] = set()
        self._layers_touched: set[int] = set()
        self.stats = {
            "bytes": 0,
            "preads": 0,
            "fused_builds": 0,
            "repeat_reads": 0,
            "layers_touched": 0,
            "seconds": 0.0,
        }

    # -- key set ------------------------------------------------------------------------------------------------------
    def _collapse_expert_keys(self) -> list[str]:
        per_expert: dict[str, dict[tuple, str]] = {}
        virtual = []
        for key in self.weight_map:
            m = _EXPERT_KEY_RE.match(key)
            if m:
                per_expert.setdefault(m["base"], {})[(int(m["expert"]), m["proj"])] = key
            else:
                virtual.append(key)
        expected = {(e, proj) for e in range(self.num_experts) for proj in _EXPERT_PROJS}
        for base, found in per_expert.items():
            if set(found) != expected:
                missing = sorted(expected - set(found))[:4]
                extra = sorted(set(found) - expected)[:4]
                raise ValueError(
                    f"{base}: expected {3 * self.num_experts} per-expert tensors for {self.num_experts} experts, found "
                    f"{len(found)} (missing e.g. {missing}, unexpected e.g. {extra})"
                )
            for kind in (GATE_UP_KEY, DOWN_KEY):
                vkey = f"{base}.{kind}"
                if vkey in self.weight_map:
                    raise ValueError(f"{vkey} is stored fused AND per expert in the checkpoint")
                virtual.append(vkey)
                self.fused[vkey] = base
        return sorted(virtual, key=_sort_key)

    # -- headers / fds ------------------------------------------------------------------------------------------------
    def _shard_path(self, shard: str) -> Path:
        return self.snapshot_dir / shard

    def _header(self, shard: str) -> _ShardHeader:
        header = self._headers.get(shard)
        if header is None:
            path = self._shard_path(shard)
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint shard {path} is missing")
            header = _ShardHeader(path)
            self._check_header_dtypes(shard, header)
            self._headers[shard] = header
        return header

    def _check_header_dtypes(self, shard: str, header: _ShardHeader):
        """Contract C1 from the metadata: every tensor bf16 except the fp32 router biases (warn on stragglers)."""
        stragglers = [
            k for k, e in header.entries.items() if e.dtype_name != "BF16" and not k.endswith(ROUTER_BIAS_SUFFIX)
        ]
        if stragglers:
            logger.warning(
                f"{shard}: {len(stragglers)} non-bf16 tensors besides the router bias (e.g. {stragglers[:3]}); fp32 ones "
                "are cast to bf16 on access (phase-1 safety-net rule)"
            )
        bad_bias = [k for k, e in header.entries.items() if k.endswith(ROUTER_BIAS_SUFFIX) and e.dtype_name != "F32"]
        if bad_bias:
            logger.warning(f"{shard}: router bias not fp32 on disk (contract C1): {bad_bias[:3]}")

    def _entry(self, physical_key: str) -> _Entry:
        shard = self.weight_map.get(physical_key)
        if shard is None:
            raise KeyError(physical_key)
        try:
            return self._header(shard).entries[physical_key]
        except KeyError:
            raise KeyError(
                f"{physical_key} is listed in {INDEX_FILE} for {shard} but the shard header does not have it"
            )

    def _fd(self, shard: str) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            path = self._shard_path(shard)
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint shard {path} is missing")
            fd = os.open(str(path), os.O_RDONLY)
            self._fds[shard] = fd
        return fd

    def _close_fd(self, shard: str):
        fd = self._fds.pop(shard, None)
        if fd is not None:
            os.close(fd)

    def _pool_for_reads(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(self.io_threads, thread_name_prefix="solar-open-loader")
        return self._pool

    @property
    def num_shards(self) -> int:
        return len(set(self.weight_map.values()))

    @property
    def open_fds(self) -> int:
        return len(self._fds)

    # -- layer window / prefetch --------------------------------------------------------------------------------------
    def group_shards(self, group: tuple) -> set[str]:
        return {self.weight_map[k] for k in self._group_physical.get(group, ())}

    def group_byte_ranges(self, group: tuple) -> dict[str, tuple[int, int]]:
        """{shard: (start, stop)} covering every tensor of the group in that shard (headers only)."""
        ranges: dict[str, tuple[int, int]] = {}
        for key in self._group_physical.get(group, ()):
            shard = self.weight_map[key]
            entry = self._header(shard).entries[key]
            start, stop = ranges.get(shard, (entry.start, entry.stop))
            ranges[shard] = (min(start, entry.start), max(stop, entry.stop))
        return ranges

    def next_group(self, group: tuple) -> tuple | None:
        """Build order of Model.__init__: embed -> layers ascending -> tail (norm + lm_head)."""
        if group[0] == "embed":
            return ("layer", self.layer_ids[0]) if self.layer_ids else ("tail",)
        if group[0] == "layer":
            later = [l for l in self.layer_ids if l > group[1]]
            return ("layer", later[0]) if later else ("tail",)
        return None

    def _enter_group(self, group: tuple):
        if group == self._window:
            return
        needed = self.group_shards(group)
        for shard in list(self._fds):
            if shard not in needed:
                self._close_fd(shard)
        previous, self._window = self._window, group
        if group[0] == "layer":
            self._layers_touched.add(group[1])
            self.stats["layers_touched"] = len(self._layers_touched)
        if self.dontneed and previous is not None:
            self._advise_async(previous, os.POSIX_FADV_DONTNEED)
        if self.prefetch_next_layer:
            nxt = self.next_group(group)
            if nxt is not None and nxt not in self._prefetched and self._group_physical.get(nxt):
                self._prefetched.add(nxt)
                self._advise_async(nxt, os.POSIX_FADV_WILLNEED)

    def _advise_async(self, group: tuple, advice: int):
        """posix_fadvise the group's byte ranges on a helper thread (WILLNEED submits ~4 GB of readahead and can take
        a while to return; the caller must not wait for it)."""
        try:
            ranges = self.group_byte_ranges(group)  # header parse on the calling thread (keeps _headers single-writer)
        except OSError as e:
            logger.debug(f"LazyStateDict: skipping fadvise for {group}: {e}")
            return
        if self._advise_pool is None:
            self._advise_pool = ThreadPoolExecutor(1, thread_name_prefix="solar-open-fadvise")
        paths = {shard: str(self._shard_path(shard)) for shard in ranges}
        self._advise_pool.submit(_fadvise_ranges, paths, ranges, advice)

    # -- reads --------------------------------------------------------------------------------------------------------
    def meta(self, key: str) -> tuple[tuple, torch.dtype]:
        """(shape, dtype) of the tensor ``read(key)`` would return, from the shard headers only."""
        if key not in self.virtual_set:
            raise KeyError(key)
        base = self.fused.get(key)
        if base is not None:
            gate0 = self._entry(f"{base}.0.gate_proj.weight")
            down0 = self._entry(f"{base}.0.down_proj.weight")
            intermediate, hidden = gate0.shape
            if key.endswith(GATE_UP_KEY):
                shape, dtype = (self.num_experts, 2 * intermediate, hidden), gate0.dtype
            else:
                shape, dtype = (self.num_experts,) + tuple(down0.shape), down0.dtype
        else:
            entry = self._entry(key)
            shape, dtype = entry.shape, entry.dtype
        return shape, _returned_dtype(key, dtype)

    def read(self, key: str) -> torch.Tensor:
        if key not in self.virtual_set:
            raise KeyError(key)
        if key in self._read_keys:
            self.stats["repeat_reads"] += 1
            logger.warning(
                f"LazyStateDict: {key} read again (every consumer reads each tensor once; this is a fresh disk read)"
            )
        self._read_keys.add(key)
        self._enter_group(_group_of(key))
        t0 = time.perf_counter()
        if key in self.fused:
            tensor = self._build_fused(key)
        else:
            tensor = self._read_physical(key)
            if self.convert_to_meta and ("q_proj.weight" in key or "k_proj.weight" in key):
                # convert_hf_qkv_to_meta_format's rule for weights: n_heads = rows // head_dim (64 q / 8 kv heads).
                tensor = reverse_permute(tensor, tensor.shape[0] // self.head_dim, tensor.shape[0], tensor.shape[1])
        if tensor.dtype != _returned_dtype(key, tensor.dtype):
            tensor = tensor.to(_returned_dtype(key, tensor.dtype))
        self.stats["seconds"] += time.perf_counter() - t0
        return tensor

    def _read_physical(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        entry = self._entry(key)
        tensor = torch.empty(entry.shape, dtype=entry.dtype)
        self.stats["bytes"] += read_into(self._fd(shard), tensor, entry.start, entry.stop)
        self.stats["preads"] += 1
        return tensor

    def _build_fused(self, key: str) -> torch.Tensor:
        base = self.fused[key]
        n_experts = self.num_experts
        gate0 = self._entry(f"{base}.0.gate_proj.weight")
        intermediate, hidden = gate0.shape
        if key.endswith(GATE_UP_KEY):
            out = torch.empty((n_experts, 2 * intermediate, hidden), dtype=gate0.dtype)
            jobs = [(f"{base}.{e}.gate_proj.weight", out[e, :intermediate]) for e in range(n_experts)]
            jobs += [(f"{base}.{e}.up_proj.weight", out[e, intermediate:]) for e in range(n_experts)]
        else:
            down0 = self._entry(f"{base}.0.down_proj.weight")
            out = torch.empty((n_experts,) + tuple(down0.shape), dtype=down0.dtype)
            jobs = [(f"{base}.{e}.down_proj.weight", out[e]) for e in range(n_experts)]
        self._read_many(jobs)
        self.stats["fused_builds"] += 1
        return out

    def _read_many(self, jobs: list[tuple[str, torch.Tensor]]):
        """pread every (physical key -> destination slice) job, sorted by file offset (one sequential sweep per shard),
        on the read pool. Headers and fds are prepared on the calling thread; the workers only pread."""
        for shard in {self.weight_map[k] for k, _ in jobs}:
            self._header(shard)
            self._fd(shard)
        jobs = sorted(jobs, key=lambda j: (self.weight_map[j[0]], self._entry(j[0]).start))

        def one(job):
            key, dst = job
            entry = self._entry(key)
            if tuple(dst.shape) != entry.shape or dst.dtype != entry.dtype:
                raise ValueError(
                    f"{key}: on-disk {entry.shape} {entry.dtype} does not fit the fused slot {tuple(dst.shape)} {dst.dtype}"
                )
            return read_into(self._fds[self.weight_map[key]], dst, entry.start, entry.stop)

        if self.io_threads > 1 and len(jobs) > 1:
            sizes = list(self._pool_for_reads().map(one, jobs))
        else:
            sizes = [one(job) for job in jobs]
        self.stats["bytes"] += sum(sizes)
        self.stats["preads"] += len(sizes)

    # -- lifecycle ----------------------------------------------------------------------------------------------------
    def close(self):
        """Close every fd and stop the helper threads; idempotent, and a later access reopens what it needs."""
        for shard in list(self._fds):
            self._close_fd(shard)
        self._window = None
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._advise_pool is not None:
            self._advise_pool.shutdown(wait=True)
            self._advise_pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _returned_dtype(key: str, on_disk: torch.dtype) -> torch.dtype:
    """Phase-1 safety-net rule: fp32 stragglers become bf16, the router bias stays fp32, everything else is as stored."""
    if on_disk == torch.float32 and not key.endswith(ROUTER_BIAS_SUFFIX):
        return torch.bfloat16
    return on_disk


def _fadvise_ranges(paths: dict[str, str], ranges: dict[str, tuple[int, int]], advice: int):
    """Worker: posix_fadvise each shard's byte range through a private fd (readahead outlives the fd)."""
    for shard, (start, stop) in ranges.items():
        try:
            fd = os.open(paths[shard], os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, start, stop - start, advice)
        except OSError as e:
            logger.debug(f"posix_fadvise({shard}, {start}, {stop - start}, {advice}) failed: {e}")
        finally:
            os.close(fd)


class LazyStateDict(Mapping):
    """Lazy contract-C1 state dict over an HF safetensors snapshot (see the module docstring).

    ``LazyStateDict(snapshot_dir, head_dim=128, num_experts=128, convert_to_meta=True, prefix="")`` is the constructor
    ``ModelArgs.load_state_dict`` uses (``head_dim`` / ``num_experts`` ``None`` = read ``config.json``). Every
    ``__getitem__`` returns a **new** tensor owned by the caller; ``substate(key)`` returns a view sharing this
    loader's store; ``meta(key)`` gives ``(shape, dtype)`` from the headers without reading tensor data.
    """

    def __init__(
        self,
        snapshot_dir,
        head_dim=128,
        num_experts=128,
        convert_to_meta=True,
        prefix="",
        io_threads=None,
        prefetch_next_layer=True,
        *,
        _store: _Store | None = None,
    ):
        if _store is None:
            _store = _Store(snapshot_dir, head_dim, num_experts, convert_to_meta, io_threads, prefetch_next_layer)
        self._store = _store
        self.prefix = prefix
        if prefix:
            self._keys = [k[len(prefix) :] for k in _store.virtual_keys if k.startswith(prefix)]
        else:
            self._keys = list(_store.virtual_keys)
        self._keyset = frozenset(self._keys)

    # -- Mapping protocol ---------------------------------------------------------------------------------------------
    def __getitem__(self, key) -> torch.Tensor:
        if not isinstance(key, str) or key not in self._keyset:
            raise KeyError(f"{self.prefix}{key}")
        return self._store.read(self.prefix + key)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key) -> bool:
        return isinstance(key, str) and key in self._keyset

    def __eq__(self, other) -> bool:
        # Mapping.__eq__ would materialise every tensor of both sides (dict(self.items())); identity is what a loader means.
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"LazyStateDict({len(self)} keys, prefix={self.prefix!r}, snapshot={self._store.snapshot_dir})"

    def __getstate__(self):
        raise TypeError("LazyStateDict holds open file descriptors and cannot be pickled or deep-copied")

    # -- substate views -----------------------------------------------------------------------------------------------
    def substate(self, key: str) -> "LazyStateDict":
        """Prefix view (``key + "."`` stripped) sharing this loader's store; empty (falsy) when nothing matches."""
        return LazyStateDict(self._store.snapshot_dir, prefix=f"{self.prefix}{key}.", _store=self._store)

    def has_substate(self, key: str) -> bool:
        prefix = f"{key}."
        return any(k.startswith(prefix) for k in self._keys)

    # -- metadata / helpers -------------------------------------------------------------------------------------------
    def meta(self, key: str) -> tuple[tuple, torch.dtype]:
        """``(shape, dtype)`` of ``self[key]`` from the shard headers only (no tensor bytes are read)."""
        if key not in self._keyset:
            raise KeyError(f"{self.prefix}{key}")
        return self._store.meta(self.prefix + key)

    @property
    def snapshot_dir(self) -> Path:
        return self._store.snapshot_dir

    @property
    def num_shards(self) -> int:
        return self._store.num_shards

    @property
    def stats(self) -> dict:
        """Shared counters: bytes, preads, fused_builds, repeat_reads, layers_touched, seconds (read + permute time)."""
        return self._store.stats

    @property
    def open_fds(self) -> int:
        return self._store.open_fds

    def layer_keys(self, layer_idx: int) -> list[str]:
        """Absolute virtual keys of ``model.layers.{layer_idx}`` (13 for Solar-Open)."""
        prefix = f"model.layers.{layer_idx}."
        return [k for k in self._store.virtual_keys if k.startswith(prefix)]

    def layer_byte_ranges(self, layer_idx: int) -> dict[str, tuple[int, int]]:
        """{shard: (start, stop)} byte ranges a layer's tensors occupy (what the WILLNEED prefetch touches)."""
        return self._store.group_byte_ranges(("layer", layer_idx))

    def layer_state_dict(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Materialise one layer (keys relative to ``model.layers.{layer_idx}.``, ~4.2 GB for Solar-Open); tests only."""
        prefix = f"model.layers.{layer_idx}."
        return {k[len(prefix) :]: self._store.read(k) for k in self.layer_keys(layer_idx)}

    def close(self):
        """Release fds and helper threads (idempotent). Views stay usable: the next access reopens lazily."""
        self._store.close()
