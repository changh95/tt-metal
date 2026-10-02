#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 TT weight-cache converter (WAVE_A_REVIEW CONV-1..4; design §2.3.11; README CONVENTIONS §7).

Builds ``<TT cache root>/<tag>/mesh4x8/{global,L<nn>}/*.tensorbin`` part by part (``global`` = embedding, final norm and
LM head; ``L<nn>`` = one decoder layer: 2 mHC sites, attention, the two RMSNorms, the dense MLP or the MoE with router,
experts and shared expert). Each part is built by the **serving constructors** with ``cache=True``
(``tt.decoder.MotifDecoderLayer``; ``tt.embedding.MotifEmbedding`` + ``tt.lm_head.MotifLMHead``), so the files are
exactly the ones the model loads, with the same mesh shape and dtype policy (both are in the cache path). No transform
is re-derived here. The mesh is the serving mesh ``(4, 8)`` unless ``--mesh`` says otherwise (``MESH_DEVICE`` is NOT
read: the plugin preset ``BH-Galaxy`` maps to (8, 4), whose cache directory the (4, 8) server never opens).

Two targets, same files (byte-identical, measured):

* ``--target mock`` (default): a **mock** 32-chip Blackhole Galaxy (``TT_METAL_MOCK_CLUSTER_DESC_PATH``), host only. The
  modules run unchanged; ``ttnn.as_tensor`` dumps each file on the host before its (no-op) device write. It never touches
  the chips and takes no device lock, so conversion does not block device work. Run it with the devices hidden::

      scripts/hostrun.sh -t 7200 -n convert -- python /home/ttuser/hchang/experiments/motif-3/scripts/convert_weights.py \
          --layers 0-2 --globals

* ``--target device``: the real mesh, opened as served (``model_config.open_motif_mesh``). Only under the lock::

      scripts/devrun.sh -t 2400 -n convert -- python /home/ttuser/hchang/experiments/motif-3/scripts/convert_weights.py \
          --target device --layers 0-2 --globals

Variants (``--router``, ``--mhc-sinkhorn``, ``--lm-head-split``, ``--embedding``) belong to one part kind each: the
globals take ``--lm-head-split`` / ``--embedding``, MoE layers ``--router`` / ``--mhc-sinkhorn``, dense layers
``--mhc-sinkhorn``. The serving defaults (router ``composite`` and ``cfg.router_logits``, Sinkhorn ``motif`` and
``cfg.mhc_sinkhorn``, LM head ``mesh``, embedding ``replicated``) are **always** built; an option only adds the
non-default choice (``stock`` == ``both``). Defaults: ``--router both`` (D1: the exact-fp32 router weight, +3 MiB per MoE
layer) and ``--mhc-sinkhorn both`` (D2 / MHC-3: the stock fallback constants, +36 KB per layer), so either choice of
those open decisions loads without the BF16 source. Variants only ever add files: a rebuild keeps every variant the part
already records (delete the part directory to drop one).

Per part (resumable, crash-safe):

1. skip it when it is complete: its ``.complete`` marker has this cache tag, ``.convert.json`` exists, every listed
   file exists with the recorded size, and the recorded variants of the part's kind cover the requested ones
   (``--force`` rebuilds). When only variants are missing, step 3 reuses the part's files (hard links) and builds just
   the missing ones;
2. preconditions, checked before anything in the part directory changes: every BF16 tensor of the part is on disk
   (else exit 4) and the cache filesystem keeps ``--min-free-gb`` (60) free after 1.1 x the part's estimated bytes
   (variants included; else exit 3);
3. build the part into a private staging directory next to the parts (``<cache dir>/.convert_staging/<host>-<pid>``,
   same filesystem), free its device tensors;
4. commit: remove the part's ``.complete`` and ``.convert.json`` first (until step 5 the part counts as incomplete:
   ``MotifModel(cache="auto")`` loads it from BF16, a crash leaves it to be rebuilt), ``os.replace`` every file into
   place, delete stale ``.tensorbin`` files that no constructor requested (renamed cache names), then sha256 + fsync
   every file and the directory;
5. write ``.convert.json`` (seconds per module, bytes, per-file sha256, variants, code fingerprint) and the
   ``.complete`` marker that ``MotifModel(cache="auto")`` keys on, each atomically and fsync'd;
6. verify (default; ``--no-verify`` skips it): rebuild the part from the cache alone with a source that raises on any
   read (every tensor the serving constructors need is cached and loads, and every listed file is one of them), then
   re-hash every file against the sha256 of step 4 (a mock load parses a file but its no-op device write never reads
   the data pages); the result goes into ``.convert.json``; a complete but unverified part that fails is rebuilt once.

Code fingerprint: the sha256 of the motif3 sources that produce cached tensors and of the ttnn build that packs them
(:data:`CODE_FINGERPRINT_FILES`). A part records the fingerprint of the code that built **all** its files (a variant
addition onto files of other code records none); ``scripts/stream_weights.py`` deletes BF16 only under parts whose
fingerprint equals the current one. ``--status --json`` reports it, the serving cache directory and per-part estimates.

Exit codes: 0 ok, 1 error, 2 usage, 3 disk guard, 4 BF16 source missing, 5 verification failed.
Runbook: ``docs/WEIGHTS_RUNBOOK.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import fcntl
import gc
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path("/home/ttuser/hchang/experiments/motif-3")
METAL = ROOT / "tt-metal"
DEVICE_LOCK = ROOT / ".device.lock"
DEFAULT_MOCK_DESC = (
    METAL / "tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml"
)
# Free-space floor of every write (both scripts): design §6 option B and CONV-3 keep >= 60 GB, the downloader's own
# default margin. Lower it only as an explicit operator decision (--min-free-gb); see docs/WEIGHTS_RUNBOOK.md §4.
MIN_FREE_GB = 60.0
SERVING_MESH: Tuple[int, int] = (4, 8)
STAGING_DIRNAME = ".convert_staging"
STATS_NAME = ".convert.json"
LOCK_NAME = ".convert.lock"
MARKER_NAME = ".complete"  # weights.layer_cache_marker
FORMAT = "motif3-convert/2"  # /2: per-kind variants, code fingerprint, reuse records (part_status reads /1 too)

# Bytes of the files of a part (measured 2026-10-02 on layers 0-2 + globals; docs/WEIGHTS_RUNBOOK.md §1): the files
# every build has (``EST_CORE_BYTES``) + the files of each variant value it builds (``EST_VARIANT_BYTES``). The disk
# guards of both scripts use estimate x GUARD_FACTOR.
EST_CORE_BYTES = {
    "global": 8_576,  # final_norm.weight
    "dense": 355_152_448,  # attention 192.6 MB, dense MLP 160.5 MB, mHC projections 2.1 MB, norms
    "moe": 6_652_399_104,  # experts 6.42 GB, attention, shared expert, composite router (prefill), local ids, mHC
}
EST_VARIANT_BYTES = {
    ("router", "composite"): 0,  # moe.router.weight + expert_bias: built for either router (prefill uses them): core
    ("router", "exact_fp32"): 3_146_112,  # moe.router.weight_fp32k_v1
    ("mhc_sinkhorn", "motif"): 17_152,  # {mhc_attn,mhc_ffn}.motif_consts
    ("mhc_sinkhorn", "stock"): 35_840,  # {mhc_attn,mhc_ffn}.{alpha,bias,lo,hi}_row
    ("lm_head_split", "mesh"): 1_803_553_728,  # lm_head.weight_mesh__dp0tp3
    ("lm_head_split", "tp"): 1_803_553_728,  # lm_head.weight_tp__tp3 (one copy per TP shard set, like every tp file)
    ("embedding", "replicated"): 1_803_551_104,  # embed.weight__rep
    ("embedding", "sharded"): 1_803_551_104,  # embed.weight__tp1
}
GUARD_FACTOR = 1.10

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_DISK, EXIT_SOURCE, EXIT_VERIFY = 0, 1, 2, 3, 4, 5

KINDS = ("global", "dense", "moe")
ROUTER_CHOICES = ("composite", "exact_fp32", "both")
SINKHORN_CHOICES = ("motif", "stock", "both")
LM_HEAD_CHOICES = ("mesh", "tp", "both")
EMBED_CHOICES = ("replicated", "sharded", "both")
VARIANT_VALUES = {
    "router": ("composite", "exact_fp32"),
    "mhc_sinkhorn": ("motif", "stock"),
    "lm_head_split": ("mesh", "tp"),
    "embedding": ("replicated", "sharded"),
}
VARIANT_KEYS = {"global": ("lm_head_split", "embedding"), "dense": ("mhc_sinkhorn",), "moe": ("router", "mhc_sinkhorn")}
SERVING_DEFAULTS = {"router": "composite", "mhc_sinkhorn": "motif", "lm_head_split": "mesh", "embedding": "replicated"}
_CFG_DEFAULT_ATTR = {"router": "router_logits", "mhc_sinkhorn": "mhc_sinkhorn"}  # env-selectable serving defaults

# The sources whose bytes decide the cached tensors (relative to the tt-metal root): the motif3 modules with cached
# weights + their transforms / dtype policy, and the ttnn build that converts, packs (bfp8) and serializes them.
CODE_FINGERPRINT_FILES = tuple(
    f"models/demos/motif3/tt/{n}"
    for n in ("weights.py", "model_config.py", "decoder.py", "mhc.py", "attention.py", "polynorm.py", "mlp.py",
              "moe.py", "embedding.py", "lm_head.py", "kernels/__init__.py", "kernels/router_fp32.py",
              "kernels/sinkhorn_motif.py")
) + (
    "ttnn/ttnn/operations/core.py",
    "ttnn/ttnn/distributed/distributed.py",
    "ttnn/ttnn/_ttnn.so",
    "build/lib/_ttnncpp.so",
    "build/lib/libtt_metal.so",
)

Part = Optional[int]  # None = the globals, else a decoder layer index


class ConvertError(RuntimeError):
    def __init__(self, msg: str, code: int = EXIT_ERROR):
        super().__init__(msg)
        self.code = code


# ======================================================================================================================
# small helpers (no ttnn)
# ======================================================================================================================
def log(msg: str) -> None:
    print(f"[convert {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def free_bytes(path: os.PathLike) -> int:
    """Bytes available to this (non-root) user on ``path``'s filesystem (what ``df`` shows as Avail)."""
    p = Path(path)
    while not p.exists():
        p = p.parent
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize


def gb(n: float) -> float:
    return float(n) / 1e9


def guarded(need_bytes: int) -> int:
    """The bytes a disk guard reserves for an estimate (x GUARD_FACTOR)."""
    return int(int(need_bytes) * GUARD_FACTOR)


def room_ok(free_b: int, need_b: int, min_free_gb: float) -> bool:
    """The disk guard of both scripts: writing ``need_b`` (already guarded) leaves >= ``min_free_gb`` free."""
    return gb(int(free_b) - int(need_b)) >= float(min_free_gb)


def part_tag(part: Part) -> str:
    return "global" if part is None else f"L{int(part):02d}"


def part_kind(cfg, part: Part) -> str:
    """``global`` | ``dense`` | ``moe``."""
    if part is None:
        return "global"
    return "moe" if cfg.layer(int(part)).is_moe else "dense"


def estimate_part_bytes(kind: str, variants: Optional[Mapping[str, Sequence[str]]] = None) -> int:
    """Bytes of a ``kind`` part built with ``variants`` (``{option: [values]}``; ``None`` = the serving defaults)."""
    if variants is None:
        variants = {k: [SERVING_DEFAULTS[k]] for k in VARIANT_KEYS[kind]}
    n = EST_CORE_BYTES[kind]
    for k in VARIANT_KEYS[kind]:
        for v in variants.get(k) or ():
            n += EST_VARIANT_BYTES.get((k, v), 0)
    return int(n)


def variant_bytes(kind: str, variants: Mapping[str, Sequence[str]]) -> int:
    """Bytes of just the variant files ``variants`` of a ``kind`` part (e.g. the missing ones of a complete part)."""
    return int(sum(EST_VARIANT_BYTES.get((k, v), 0) for k in VARIANT_KEYS[kind] for v in variants.get(k) or ()))


# the default-options sizes (runbook, tests): global 3.607 GB, dense 0.355 GB, moe 6.656 GB
EST_PART_BYTES = {k: estimate_part_bytes(k, {kk: list(VARIANT_VALUES[kk]) if kk in ("router", "mhc_sinkhorn")
                                             else [SERVING_DEFAULTS[kk]] for kk in VARIANT_KEYS[k]}) for k in KINDS}


def parse_layers(spec: Optional[str], n_layers: int) -> List[int]:
    """``"0-52"``, ``"all"``, ``"3,4,10-12"``, ``"36-"``, ``"none"`` / ``""`` -> sorted unique layer indices."""
    if spec is None:
        return []
    s = str(spec).strip().lower()
    if s in ("", "none"):
        return []
    if s == "all":
        return list(range(n_layers))
    out = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            lo, hi = int(a), (int(b) if b.strip() else n_layers - 1)
            if lo > hi:
                raise ValueError(f"empty layer range {tok!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(tok))
    bad = sorted(i for i in out if i < 0 or i >= n_layers)
    if bad:
        raise ValueError(f"layers {bad} outside [0, {n_layers})")
    return sorted(out)


def parse_mesh(spec: str) -> Tuple[int, int]:
    parts = [p for p in re.split(r"[x,() ]+", str(spec).strip()) if p]
    if len(parts) != 2:
        raise ValueError(f"mesh must look like 4x8, got {spec!r}")
    return int(parts[0]), int(parts[1])


def sha256_file(path: Path, bufsize: int = 64 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _hash_and_sync(path: Path, do_hash: bool) -> Optional[str]:
    """sha256 of ``path`` (or None) after making its data durable (fsync)."""
    with open(path, "rb") as f:
        h = None
        if do_hash:
            hh = hashlib.sha256()
            while True:
                b = f.read(64 << 20)
                if not b:
                    break
                hh.update(b)
            h = hh.hexdigest()
        os.fsync(f.fileno())
    return h


def hash_files(paths: Sequence[Path], workers: int = 8, *, sync: bool = False, do_hash: bool = True) -> Dict[str, str]:
    """sha256 of every file (thread pool: hashlib releases the GIL on large updates; ~1.9 GB/s per thread here); with
    ``sync`` every file is also fsync'd (its data is on disk before a marker or a BF16 deletion relies on it)."""
    fn = (lambda p: _hash_and_sync(p, do_hash)) if sync else sha256_file
    with ThreadPoolExecutor(max(1, min(workers, len(paths) or 1))) as ex:
        out = dict(zip((p.name for p in paths), ex.map(fn, paths)))
    return {k: v for k, v in out.items() if v is not None}


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_atomic(path: Path, obj: Any, *, sync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=False))
        if sync:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)
    if sync:
        fsync_dir(path.parent)


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return None


def code_fingerprint(metal: Path = METAL) -> Dict[str, Any]:
    """``{"sha256": <hex>, "files": {path: sha256 | "missing"}}`` over :data:`CODE_FINGERPRINT_FILES` (~0.3 s)."""
    paths = [Path(metal) / rel for rel in CODE_FINGERPRINT_FILES]
    with ThreadPoolExecutor(4) as ex:
        shas = list(ex.map(lambda p: sha256_file(p) if p.is_file() else "missing", paths))
    files = dict(zip(CODE_FINGERPRINT_FILES, shas))
    h = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in files.items()).encode()).hexdigest()
    return {"sha256": h, "files": files}


def devices_visible() -> List[str]:
    try:
        return sorted(os.listdir("/dev/tenstorrent"))
    except FileNotFoundError:
        return []


def device_lock_held_by_someone(lock_path: Path = DEVICE_LOCK) -> bool:
    """True iff the devrun.sh flock is held (by our parent devrun.sh, or by another job). A fresh open file description
    conflicts with every other holder, the parent's included, so "acquirable" means "nobody holds the device lock"."""
    if not lock_path.exists():
        return False
    fd = os.open(str(lock_path), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def under_devrun(max_depth: int = 32) -> bool:
    """True iff an ancestor process runs ``scripts/devrun.sh`` (it holds the device lock for its child). Together
    with :func:`device_lock_held_by_someone` this tells our own lock from another job's."""
    pid = os.getppid()
    for _ in range(max_depth):
        if pid <= 1:
            return False
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            return False
        # the script itself (``bash /.../devrun.sh ...`` or exec'd directly), not a shell whose -c text mentions it
        if any(Path(a).name == "devrun.sh" for a in argv[:2]):
            return True
        m = re.search(r"^PPid:\s+(\d+)", status, re.M)
        if not m:
            return False
        pid = int(m.group(1))
    return False


@contextlib.contextmanager
def converter_lock(cache_dir: Path):
    """One converter per cache directory (mock and device runs alike): an flock on ``<cache_dir>/.convert.lock``.
    ``scripts/stream_weights.py`` takes the same lock while it decides on and performs BF16 deletions."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(cache_dir / LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o664)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConvertError(f"another converter (or a deleting stream_weights.py) holds {cache_dir / LOCK_NAME}",
                               EXIT_ERROR)
        yield
    finally:
        os.close(fd)


# ======================================================================================================================
# weight sources
# ======================================================================================================================
class RaisingSource:
    """A weight source that must never be read: every access raises (proves a part loads from the TT cache alone)."""

    def __init__(self, what: str = "TT cache verification"):
        self.what = what

    def _fail(self, op: str, name: Any = None):
        raise AssertionError(f"{self.what}: the BF16 source was accessed ({op}({name!r})); the TT cache is incomplete")

    def get(self, name, dtype=None):
        self._fail("get", name)

    def get_rows(self, name, start=None, stop=None, dtype=None):
        self._fail("get_rows", name)

    def has(self, name):
        self._fail("has", name)

    def available(self, name):
        self._fail("available", name)

    def shape(self, name):
        self._fail("shape", name)

    def keys(self):
        self._fail("keys")

    def __contains__(self, name):
        self._fail("__contains__", name)

    def layer_available(self, layer):
        self._fail("layer_available", layer)


class CountingSource:
    """Wraps an ``HFWeightLoader`` / ``DictWeightSource``: counts tensors, bytes and seconds of ``get`` / ``get_rows``
    (the BF16 read + materialization cost of a part). Every other attribute is forwarded."""

    def __init__(self, inner):
        self.inner = inner
        self.reset()

    def reset(self) -> None:
        self.n_tensors, self.n_bytes, self.seconds = 0, 0, 0.0

    def _count(self, t, t0):
        self.n_tensors += 1
        self.n_bytes += int(t.numel()) * int(t.element_size())
        self.seconds += time.perf_counter() - t0
        return t

    def get(self, name, dtype=None):
        t0 = time.perf_counter()
        return self._count(self.inner.get(name, dtype), t0)

    def get_rows(self, name, start, stop, dtype=None):
        t0 = time.perf_counter()
        return self._count(self.inner.get_rows(name, start, stop, dtype), t0)

    def __contains__(self, name):
        return name in self.inner

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def missing_source_tensors(source, cfg, part: Part) -> List[str]:
    """HF tensors the part needs that are not (completely) on disk (``HFWeightLoader.available``: index + shard size
    check). Globals: embedding, final norm, LM head; a layer: every ``model.layers.{l}.*`` tensor of the index."""
    if not hasattr(source, "available"):
        return []
    if part is None:
        names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    elif hasattr(source, "layer_names"):
        names = source.layer_names(int(part))
        if not names:
            return [f"model.layers.{int(part)}.* (no tensors in the index)"]
    else:
        return []
    return [n for n in names if not source.available(n)]


# ======================================================================================================================
# recording weights.as_tensor (which files a part touches, hits / misses, seconds per module)
# ======================================================================================================================
@dataclass
class TensorRecord:
    cache_name: Optional[str]
    layer: Optional[int]
    path: Optional[str]
    hit: bool
    seconds: float
    dtype: str
    layout: str
    mapping: str


def module_of(cache_name: Optional[str]) -> str:
    """Module bucket of a cache name: ``attn.v2.wq_b`` -> ``attention``, ``mhc_attn.proj_rows_x128`` -> ``mhc``, ..."""
    if not cache_name:
        return "uncached"
    n = cache_name
    if n.startswith("attn."):
        return "attention"
    if n.startswith("mhc_"):
        return "mhc"
    if n.startswith("moe.shared."):
        return "shared_expert"
    if n.startswith("moe.router."):
        return "router"
    if n.startswith("moe."):
        return "experts"
    if n.startswith("mlp."):
        return "dense_mlp"
    if n.endswith("layernorm.weight") or n.startswith("final_norm"):
        return "norms"
    if n.startswith("embed."):
        return "embedding"
    if n.startswith("lm_head."):
        return "lm_head"
    return n.split(".", 1)[0]


class AsTensorRecorder:
    """Context manager: wraps ``models.demos.motif3.tt.weights.as_tensor`` (every module uploads through the module
    attribute ``W.as_tensor``) to record each call's cache file, hit / miss and duration. Restores it on exit."""

    def __init__(self):
        self.records: List[TensorRecord] = []

    def __enter__(self):
        import ttnn  # noqa: F401  (the wrapped function needs it anyway)
        from models.demos.motif3.tt import weights as W

        self._W = W
        self._orig = W.as_tensor
        orig = self._orig

        def wrapped(src, *, mesh_device, cfg, dtype, **kw):
            import ttnn as _ttnn

            layout = kw.get("layout", _ttnn.TILE_LAYOUT)
            cache_name, layer = kw.get("cache_name"), kw.get("layer")
            dp_dim, tp_dim = kw.get("dp_dim"), kw.get("tp_dim")
            path = None
            hit = False
            if cache_name is not None:
                path = W.tensorbin_path(W.cache_prefix(cfg, cache_name, layer, dp_dim, tp_dim), dtype, layout)
                hit = path.is_file()
            t0 = time.perf_counter()
            out = orig(src, mesh_device=mesh_device, cfg=cfg, dtype=dtype, **kw)
            self.records.append(
                TensorRecord(
                    cache_name=cache_name,
                    layer=layer,
                    path=None if path is None else str(path),
                    hit=bool(hit),
                    seconds=time.perf_counter() - t0,
                    dtype=getattr(dtype, "name", str(dtype)),
                    layout=getattr(layout, "name", str(layout)),
                    mapping=W.mapping_tag(dp_dim, tp_dim),
                )
            )
            return out

        W.as_tensor = wrapped
        return self

    def __exit__(self, *exc):
        self._W.as_tensor = self._orig
        return False


# ======================================================================================================================
# freeing device tensors
# ======================================================================================================================
def free_object(obj, keep: Iterable[Any] = ()) -> int:
    """Free every device tensor ``obj`` holds: its own ``deallocate`` / ``release`` when it has one, then a recursive
    walk over attributes, lists, tuples and dicts of motif3 objects. Objects in ``keep`` (the shared ``MotifCCL`` /
    ``MotifRope`` / config every part references) are never entered. Idempotent; returns the number of tensors the walk
    freed (on top of the module's own release)."""
    import ttnn

    for meth in ("deallocate", "release"):
        f = getattr(obj, meth, None)
        if callable(f):
            try:
                f()
            except Exception as e:  # keep going: the walk below frees what is left
                log(f"  warning: {type(obj).__name__}.{meth}() raised {type(e).__name__}: {str(e)[:160]}")
            break
    close = getattr(obj, "close", None)
    if callable(close) and type(obj).__name__ == "MotifLMHead":
        close()
    n = 0
    seen = {id(k) for k in keep}
    stack = [obj]
    while stack:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        if isinstance(v, ttnn.Tensor):
            try:
                if v.storage_type() == ttnn.StorageType.DEVICE and v.is_allocated():
                    ttnn.deallocate(v)
                    n += 1
            except Exception:
                pass
        elif isinstance(v, (list, tuple, set)):
            stack.extend(v)
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif hasattr(v, "__dict__") and type(v).__module__.startswith("models.demos.motif3"):
            stack.extend(vars(v).values())
    return n


# ======================================================================================================================
# options and the part plan
# ======================================================================================================================
def _expand(value: str, key: str) -> List[str]:
    return list(VARIANT_VALUES[key]) if value == "both" else [value]


def kind_variants(recorded: Optional[Mapping[str, Any]], kind: str) -> Dict[str, List[str]]:
    """The variants of a ``kind`` part from a ``.convert.json`` record (format /1 recorded all four options for every
    part; only the kind's options apply)."""
    rec = recorded or {}
    return {k: [v for v in VARIANT_VALUES[k] if v in (rec.get(k) or [])] for k in VARIANT_KEYS[kind]}


def merge_variants(*vs: Optional[Mapping[str, Sequence[str]]], kind: str) -> Dict[str, List[str]]:
    out = {}
    for k in VARIANT_KEYS[kind]:
        have = set()
        for v in vs:
            have.update((v or {}).get(k) or [])
        out[k] = [x for x in VARIANT_VALUES[k] if x in have]
    return out


@dataclass(frozen=True)
class ConvertOptions:
    """Which module variants to cache, per part kind. The serving defaults are always included (see the module
    docstring); the defaults here add the exact-fp32 router (D1) and the stock Sinkhorn constants (D2)."""

    router: str = "both"  # MoE layers: composite (always) + exact_fp32 (kernels/router_fp32 weight)
    mhc_sinkhorn: str = "both"  # every layer: motif (always) + stock (pre-clamped fallback constants)
    lm_head_split: str = "mesh"  # globals: mesh (EMB-D1 default, always) + tp
    embedding: str = "replicated"  # globals: replicated (always) + sharded (shard_hidden=True)
    hash_files: bool = True

    def __post_init__(self):
        for name, val, ok in (
            ("router", self.router, ROUTER_CHOICES),
            ("mhc_sinkhorn", self.mhc_sinkhorn, SINKHORN_CHOICES),
            ("lm_head_split", self.lm_head_split, LM_HEAD_CHOICES),
            ("embedding", self.embedding, EMBED_CHOICES),
        ):
            if val not in ok:
                raise ValueError(f"{name} must be one of {ok}, got {val!r}")

    def variants_for(self, kind: str, cfg=None) -> Dict[str, List[str]]:
        """``{option: [values]}`` a ``kind`` part must hold: the requested values + the serving defaults (and the
        ``cfg`` defaults ``cfg.router_logits`` / ``cfg.mhc_sinkhorn``, which the decoder builds)."""
        out = {}
        for k in VARIANT_KEYS[kind]:
            want = {SERVING_DEFAULTS[k], *_expand(getattr(self, k), k)}
            attr = _CFG_DEFAULT_ATTR.get(k)
            if cfg is not None and attr:
                want.add(str(getattr(cfg, attr)))
            out[k] = [v for v in VARIANT_VALUES[k] if v in want]
        return out

    def all_variants(self, cfg=None) -> Dict[str, Dict[str, List[str]]]:
        return {kind: self.variants_for(kind, cfg) for kind in KINDS}

    def missing(self, recorded: Optional[Mapping[str, Any]], kind: str, cfg=None) -> Dict[str, List[str]]:
        """Requested variants of a ``kind`` part that ``recorded`` (a ``.convert.json`` ``variants``) does not hold."""
        have = kind_variants(recorded, kind)
        out = {}
        for k, want in self.variants_for(kind, cfg).items():
            miss = [v for v in want if v not in have.get(k, [])]
            if miss:
                out[k] = miss
        return out

    def covered_by(self, recorded: Optional[Mapping[str, Any]], kind: str, cfg=None) -> bool:
        """True iff a ``kind`` part converted with ``recorded`` variants already holds every variant requested here."""
        return bool(recorded) and not self.missing(recorded, kind, cfg)


def build_part(mesh, cfg, part: Part, *, source, ccl, rope, variants: Mapping[str, Sequence[str]],
               on_built: Optional[Callable[[str, float], None]] = None) -> List[Tuple[str, Any]]:
    """Construct the serving objects of one part with ``cache=True`` (their constructors write / load exactly their
    cache files). ``variants``: the part kind's ``{option: [values]}`` (:meth:`ConvertOptions.variants_for`). Returns
    ``[(name, object)]`` for :func:`free_object`. The decoder layer is built with the config defaults (what
    ``MotifModel`` builds); the other values add the files of the non-default options."""
    built: List[Tuple[str, Any]] = []

    def add(name, fn):
        t0 = time.perf_counter()
        obj = fn()
        built.append((name, obj))
        if on_built is not None:
            on_built(name, time.perf_counter() - t0)

    try:
        if part is None:
            from models.demos.motif3.tt.embedding import MotifEmbedding
            from models.demos.motif3.tt.lm_head import MotifLMHead

            for emb in variants["embedding"]:
                add(f"embedding[{emb}]", lambda emb=emb: MotifEmbedding(
                    mesh, cfg, source=source, ccl=ccl, cache=True, shard_hidden=(emb == "sharded")))
            for split in variants["lm_head_split"]:
                add(f"lm_head[{split}]", lambda split=split: MotifLMHead(
                    mesh, cfg, source=source, ccl=ccl, cache=True, vocab_split=split))
            return built
        from models.demos.motif3.tt.decoder import MotifDecoderLayer

        l = int(part)
        spec = cfg.layer(l)
        sks = list(variants["mhc_sinkhorn"])
        base_sinkhorn = cfg.mhc_sinkhorn if cfg.mhc_sinkhorn in sks else sks[0]
        kw = dict(sinkhorn=base_sinkhorn)
        if spec.is_moe:
            rls = list(variants["router"])
            base_router = cfg.router_logits if cfg.router_logits in rls else rls[0]
            kw["router_logits"] = base_router
        add("decoder_layer", lambda: MotifDecoderLayer(mesh, cfg, l, source=source, ccl=ccl, rope=rope, cache=True,
                                                       **kw))
        for sk in sks:
            if sk == base_sinkhorn:
                continue
            from models.demos.motif3.tt.mhc import MHCSite

            for site in ("mhc_attn", "mhc_ffn"):  # the projection is a cache hit; adds the backend's constants
                add(f"{site}[{sk}]", lambda site=site, sk=sk: MHCSite(
                    mesh, cfg, l, site, source=source, cache=True, sinkhorn=sk))
        if spec.is_moe:
            for rl in variants["router"]:
                if rl == kw["router_logits"]:
                    continue
                if rl == "exact_fp32":
                    from models.demos.motif3.tt.kernels.router_fp32 import RouterLogitsFP32

                    add("router[exact_fp32]", lambda: RouterLogitsFP32.from_source(mesh, cfg, l, source=source, cache=True))
                else:  # composite weights (the MoE builds them for either router; kept for symmetry)
                    from models.demos.motif3.tt.moe import MotifRouter

                    add("router[composite]", lambda: MotifRouter(mesh, cfg, l, source=source, cache=True))
        return built
    except BaseException:
        for _, obj in built:
            free_object(obj, keep=(ccl, rope, cfg, mesh, source))
        raise


# ======================================================================================================================
# part status (no device)
# ======================================================================================================================
@dataclass
class PartStatus:
    part: Part
    complete: bool
    reason: str
    files: int = 0
    bytes: int = 0
    verified: Optional[bool] = None
    stats: Optional[dict] = None
    kind: str = ""
    base_ok: bool = False  # marker, tag, .convert.json and every listed file are fine (only variants may be missing)
    missing: Dict[str, List[str]] = field(default_factory=dict)  # requested variants the part lacks
    names: List[str] = field(default_factory=list)  # the files the marker lists
    need_bytes: int = 0  # what converting it now adds: 0 (complete), the missing variants (reuse), or the whole part
    rebuild_bytes: int = 0  # the whole part with requested + recorded variants (a --force rebuild, transiently)

    @property
    def recorded_variants(self) -> Dict[str, List[str]]:
        return kind_variants((self.stats or {}).get("variants"), self.kind) if self.kind else {}

    @property
    def hashes(self) -> int:
        """Listed files with a sha256 in ``.convert.json``."""
        f = (self.stats or {}).get("files") or {}
        return sum(1 for n in self.names if (f.get(n) or {}).get("sha256"))

    @property
    def files_hashed(self) -> Optional[int]:
        v = (self.stats or {}).get("verify") or {}
        return v.get("files_hashed")

    @property
    def fingerprint(self) -> Optional[str]:
        return ((self.stats or {}).get("code") or {}).get("sha256")

    def as_dict(self) -> dict:
        s = self.stats or {}
        return {"part": part_tag(self.part), "layer": self.part, "kind": self.kind, "complete": self.complete,
                "reason": self.reason, "base_ok": self.base_ok, "files": self.files, "bytes": self.bytes,
                "verified": self.verified, "hashes": self.hashes, "files_hashed": self.files_hashed,
                "variants": self.recorded_variants, "missing_variants": self.missing,
                "code_fingerprint": self.fingerprint, "need_bytes": self.need_bytes,
                "rebuild_bytes": self.rebuild_bytes, "build_seconds": (s.get("seconds") or {}).get("build"),
                "converted_at": s.get("converted_at"), "target": s.get("target"), "verify": s.get("verify")}


def part_status(cfg, part: Part, options: ConvertOptions) -> PartStatus:
    """Status of one part in ``cfg.cache_dir`` (no device, no ttnn tensors). Complete iff: the ``.complete`` marker has
    this cache tag, every listed file exists with the size recorded in ``.convert.json``, and the recorded variants of
    the part's kind cover ``options`` (``base_ok`` without the last condition). A marker without ``.convert.json``
    (written by another tool) counts as incomplete: this converter rebuilds it so that every part carries sizes,
    hashes, a fingerprint and a verification record."""
    from models.demos.motif3.tt import weights as W

    kind = part_kind(cfg, part)
    want = options.variants_for(kind, cfg)
    full = estimate_part_bytes(kind, want)

    def bad(reason, **kw):
        st = PartStatus(part, False, reason, kind=kind, need_bytes=full, rebuild_bytes=full, **kw)
        if st.stats:  # a rebuild keeps the recorded variants
            st.need_bytes = st.rebuild_bytes = estimate_part_bytes(kind, merge_variants(want, st.recorded_variants,
                                                                                        kind=kind))
        return st

    marker = W.layer_cache_marker(cfg, part)
    m = read_json(marker)
    if m is None:
        return bad("no .complete marker")
    if m.get("version") != cfg.cache_version_tag:
        return bad(f"marker tag {m.get('version')} != {cfg.cache_version_tag}")
    d = marker.parent
    names = sorted(m.get("tensors") or [])
    if not names:
        return bad("marker lists no files")
    stats = read_json(d / STATS_NAME)
    if stats is None:
        return bad("marker without .convert.json (converted by another tool)")
    sizes = {k: (v or {}).get("bytes") for k, v in (stats.get("files") or {}).items()}
    total = 0
    for n in names:
        p = d / n
        if not p.is_file():
            return bad(f"listed file missing: {n}", stats=stats, names=names)
        sz = p.stat().st_size
        if sizes.get(n) is not None and int(sizes[n]) != sz:
            return bad(f"size of {n} changed ({sz} != {sizes[n]})", stats=stats, names=names)
        total += sz
    v = (stats.get("verify") or {}).get("ok")
    st = PartStatus(part, True, "complete", len(names), total, verified=v, stats=stats, kind=kind, base_ok=True,
                    names=names)
    st.rebuild_bytes = estimate_part_bytes(kind, merge_variants(want, st.recorded_variants, kind=kind))
    miss = options.missing(stats.get("variants"), kind, cfg)
    if miss:  # (``verified`` stays the record of the variants it has)
        st.complete, st.missing = False, miss
        st.reason = f"variants {st.recorded_variants} miss {miss}"
        st.need_bytes = variant_bytes(kind, miss)
    return st


def serving_cache(cfg) -> Dict[str, Any]:
    """The serving cache directory for ``cfg``'s root: the (4, 8) mesh, the pinned checkpoint revision and the default
    dtype policy (what a server started with the defaults opens), and whether ``cfg`` resolves to it."""
    from models.demos.motif3.tt import model_config as MC

    tag = f"motif3-{MC.DEFAULT_WEIGHTS_REVISION[:8]}-c{MC.CACHE_FORMAT_VERSION}-{MC.DtypePolicy().tag}"
    d = Path(cfg.tt_cache_root) / tag / f"mesh{SERVING_MESH[0]}x{SERVING_MESH[1]}"
    return {"serving_mesh_shape": list(SERVING_MESH), "serving_tag": tag, "serving_cache_dir": str(d),
            "is_serving_cache": tuple(cfg.mesh_shape) == SERVING_MESH and cfg.cache_version_tag == tag
            and os.path.realpath(cfg.cache_dir) == os.path.realpath(d)}


# ======================================================================================================================
# the converter
# ======================================================================================================================
class Converter:
    """Converts parts on an opened mesh (mock or real). ``cfg`` decides the cache root / tag / mesh directory.

    Library use (tests): ``Converter(mesh, cfg, source=HFWeightLoader(), options=...).run([None, 0, 1, 2])``."""

    def __init__(self, mesh, cfg, *, source=None, options: Optional[ConvertOptions] = None, target: str = "mock",
                 min_free_gb: float = MIN_FREE_GB, logger: Callable[[str], None] = log,
                 fingerprint: Optional[Dict[str, Any]] = None):
        from models.demos.motif3.tt import weights as W
        from models.demos.motif3.tt.ccl import MotifCCL
        from models.demos.motif3.tt.rope import MotifRope

        if fingerprint is None:  # import every fingerprinted module first: the fingerprint is of the code that runs
            import importlib

            for m in ("decoder", "embedding", "lm_head", "mhc", "moe", "kernels.router_fp32", "kernels.sinkhorn_motif"):
                importlib.import_module(f"models.demos.motif3.tt.{m}")
        self.W = W
        self.mesh = mesh
        self.cfg = cfg
        self.options = options or ConvertOptions()
        self.target = target
        self.min_free_gb = float(min_free_gb)
        self.log = logger
        self.source = CountingSource(source if source is not None else W.HFWeightLoader(cfg.weights_dir))
        self.ccl = MotifCCL(mesh, cfg)
        self.rope = MotifRope(mesh, cfg)
        self.fingerprint = fingerprint or code_fingerprint()
        # staging next to the parts (same filesystem as the files, also when the tag directory is a symlink into
        # another volume, TIS_RUNBOOK §2.4); one subdirectory per process; the converter lock covers it
        cache_dir = Path(os.path.realpath(cfg.cache_dir))
        self.staging_parent = cache_dir / STAGING_DIRNAME
        self.staging_root = self.staging_parent / f"{socket.gethostname()}-{os.getpid()}"
        self.stage_cfg = dataclasses.replace(cfg, tt_cache_root=self.staging_root)
        self.report: Dict[str, dict] = {}

    @property
    def shared(self) -> Tuple[Any, ...]:
        """Objects every part references and :func:`free_object` must not enter."""
        return (self.ccl, self.rope, self.cfg, self.stage_cfg, self.mesh, self.source)

    # ---- paths ----------------------------------------------------------------------------------------------------
    def part_dir(self, part: Part, cfg=None) -> Path:
        cfg = cfg or self.cfg
        return self.W.layer_cache_marker(cfg, part).parent

    def stats_path(self, part: Part) -> Path:
        return self.part_dir(part) / STATS_NAME

    # ---- status ---------------------------------------------------------------------------------------------------
    def status(self, part: Part) -> PartStatus:
        return part_status(self.cfg, part, self.options)

    # ---- guards ---------------------------------------------------------------------------------------------------
    def check_disk(self, part: Part, need_bytes: int) -> Tuple[int, int]:
        need = guarded(need_bytes)
        free = free_bytes(self.cfg.tt_cache_root)
        if not room_ok(free, need, self.min_free_gb):
            raise ConvertError(
                f"disk guard: {part_tag(part)} needs ~{gb(need):.2f} GB, {gb(free):.1f} GB free; converting would leave "
                f"{gb(free - need):.1f} GB < {self.min_free_gb:g} GB (free space first, e.g. scripts/stream_weights.py "
                f"--delete-converted, or lower the floor explicitly with --min-free-gb)",
                EXIT_DISK,
            )
        return free, need

    def check_source(self, part: Part) -> None:
        missing = missing_source_tensors(self.source.inner, self.cfg, part)
        if missing:
            raise ConvertError(
                f"{part_tag(part)}: {len(missing)} BF16 tensors are not on disk (e.g. {missing[:3]}); download the layer "
                f"first (scripts/download_weights.py --layers {'' if part is None else part})",
                EXIT_SOURCE,
            )

    # ---- one part -------------------------------------------------------------------------------------------------
    def convert_part(self, part: Part, *, reuse: bool = False) -> dict:
        """Build one part (step 2-5 of the module docstring). ``reuse``: the part is complete except for variants;
        its files are hard-linked into staging (cache hits) and only the missing variants are built."""
        tag = part_tag(part)
        kind = part_kind(self.cfg, part)
        old = self.status(part)
        want = self.options.variants_for(kind, self.cfg)
        variants = merge_variants(want, old.recorded_variants, kind=kind) if old.stats else want
        reuse_names = list(old.names) if (reuse and old.base_ok) else []
        final_dir = self.part_dir(part)
        # ---- preconditions: nothing in the part directory changes before these pass -------------------------------
        self.check_source(part)
        # the same estimate part_status reports (need_bytes), which scripts/stream_weights.py guards with
        free0, _ = self.check_disk(part, variant_bytes(kind, old.missing) if reuse_names
                                   else estimate_part_bytes(kind, variants))
        stage_dir = self.part_dir(part, self.stage_cfg)
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True)
        if reuse_names:
            try:
                for n in reuse_names:
                    os.link(final_dir / n, stage_dir / n)
            except OSError as e:  # e.g. EXDEV / EPERM: build the whole part instead
                self.log(f"{tag}: cannot hard-link the existing files into staging ({e}); rebuilding the whole part")
                shutil.rmtree(stage_dir)
                stage_dir.mkdir(parents=True)
                reuse_names = []
                free0, _ = self.check_disk(part, estimate_part_bytes(kind, variants))
        self.source.reset()
        module_s: Dict[str, float] = {}
        t0 = time.perf_counter()
        with AsTensorRecorder() as rec:
            built = build_part(self.mesh, self.stage_cfg, part, source=self.source, ccl=self.ccl, rope=self.rope,
                               variants=variants)
        t_build = time.perf_counter() - t0
        for _, obj in built:
            free_object(obj, keep=self.shared)
        del built
        gc.collect()
        if hasattr(self.source.inner, "close"):
            self.source.inner.close()  # drop the safetensors mmaps of this part
        uncached = [r for r in rec.records if r.cache_name is None]
        if uncached:
            raise ConvertError(f"{tag}: {len(uncached)} weight uploads bypass the TT cache (cache_name=None); serving "
                               f"would read the BF16 source for them", EXIT_ERROR)
        for r in rec.records:
            module_s[module_of(r.cache_name)] = module_s.get(module_of(r.cache_name), 0.0) + r.seconds
        files: Dict[str, dict] = {}
        for r in rec.records:
            src = Path(r.path)
            if not str(src).startswith(str(self.staging_root)):
                raise ConvertError(f"{tag}: cache file outside the staging root: {src}")
            if r.layer != part:
                raise ConvertError(f"{tag}: tensor {r.cache_name} was cached for part {part_tag(r.layer)}")
            if src.name in files:
                continue  # the same tensor requested twice (e.g. a shared projection of two mHC variants)
            if not src.is_file():
                raise ConvertError(f"{tag}: expected cache file not written: {src}")
            files[src.name] = {"bytes": src.stat().st_size, "cache_name": r.cache_name, "mapping": r.mapping,
                               "dtype": r.dtype, "layout": r.layout, "seconds": round(r.seconds, 3),
                               "reused": bool(r.hit and src.name in reuse_names)}
        n_new = sum(1 for f in files.values() if not f["reused"])
        # ---- commit: the part is incomplete from here until its marker is back --------------------------------------
        t1 = time.perf_counter()
        final_dir.mkdir(parents=True, exist_ok=True)
        for p in (final_dir / MARKER_NAME, final_dir / STATS_NAME):
            with contextlib.suppress(FileNotFoundError):
                p.unlink()
        fsync_dir(final_dir)
        for name in files:
            os.replace(stage_dir / name, final_dir / name)  # a reused hard link: rename(2) of a file onto itself
        stale = sorted(p.name for p in final_dir.glob("*.tensorbin") if p.name not in files)
        for n in stale:
            (final_dir / n).unlink()
        if stale:
            self.log(f"{tag}: removed {len(stale)} stale files no constructor requested: {stale[:4]}"
                     f"{' ...' if len(stale) > 4 else ''}")
        t_move = time.perf_counter() - t1
        # ---- hash + fsync, stats, marker ----------------------------------------------------------------------------
        t2 = time.perf_counter()
        shas = hash_files([final_dir / n for n in files], sync=True, do_hash=self.options.hash_files)
        for name, h in shas.items():
            files[name]["sha256"] = h
        fsync_dir(final_dir)
        t_hash = time.perf_counter() - t2
        total = sum(f["bytes"] for f in files.values())
        fp = self.fingerprint["sha256"]
        code: Dict[str, Any] = {"sha256": fp, "files": self.fingerprint["files"]}
        if reuse_names and old.fingerprint != fp:  # the reused files were built by other (or unknown) code
            code = {"sha256": None, "files": self.fingerprint["files"], "reused_from": old.fingerprint,
                    "note": f"{len(reuse_names)} files reused from a build by other code; --force rebuilds them"}
        stats = {
            "format": FORMAT,
            "part": tag,
            "layer": part,
            "kind": kind if part is None else f"{self.cfg.layer(int(part)).attn_kind}-attn/{kind}",
            "cache_tag": self.cfg.cache_version_tag,
            "mesh_shape": list(self.cfg.mesh_shape),
            "cache_dir": str(final_dir),
            "target": self.target,
            "variants": variants,
            "defaults": {"router_logits": self.cfg.router_logits, "mhc_sinkhorn": self.cfg.mhc_sinkhorn,
                         "lm_head_split": SERVING_DEFAULTS["lm_head_split"], "embedding": SERVING_DEFAULTS["embedding"]},
            "code": code,
            "converted_at": now_iso(),
            "host": socket.gethostname(),
            "bytes": total,
            "n_files": len(files),
            "reused_files": len(reuse_names),
            "new_files": n_new,
            "removed_stale": stale,
            "source": {"tensors": self.source.n_tensors, "bytes": self.source.n_bytes},
            "seconds": {"build": round(t_build, 2), "move": round(t_move, 3), "hash": round(t_hash, 2),
                        "modules": {k: round(v, 2) for k, v in sorted(module_s.items())}},
            "free_gb_before": round(gb(free0), 1),
            "free_gb_after": round(gb(free_bytes(self.cfg.tt_cache_root)), 1),
            "estimate_bytes": estimate_part_bytes(kind, variants),
            "files": files,
        }
        write_json_atomic(final_dir / STATS_NAME, stats)
        staged_marker = self.W.mark_layer_cached(self.stage_cfg, part, sorted(files))  # same format and tag
        with open(staged_marker, "rb") as f:
            os.fsync(f.fileno())
        os.replace(staged_marker, final_dir / MARKER_NAME)
        fsync_dir(final_dir)
        shutil.rmtree(stage_dir, ignore_errors=True)
        mods = ", ".join(f"{k} {v:.1f}" for k, v in sorted(module_s.items()))
        how = f"{n_new} new + {len(reuse_names)} reused files" if reuse_names else f"{len(files)} files"
        self.log(f"{tag} ({stats['kind']}): {how}, {gb(total):.3f} GB in {t_build:.1f} s "
                 f"({self.source.n_tensors} BF16 tensors, {gb(self.source.n_bytes):.2f} GB; {mods}); "
                 f"move {t_move:.2f} s, sha256 + fsync {t_hash:.1f} s; {stats['free_gb_after']:.1f} GB free")
        return stats

    def verify_part(self, part: Part) -> dict:
        """Rebuild the part from the cache alone (a source that raises on any read) on this mesh, then free it."""
        tag = part_tag(part)
        st = self.status(part)
        if not st.complete:
            raise ConvertError(f"{tag}: cannot verify an incomplete part ({st.reason})", EXIT_VERIFY)
        t0 = time.perf_counter()
        ok, err = True, None
        with AsTensorRecorder() as rec:
            try:
                built = build_part(self.mesh, self.cfg, part, source=RaisingSource(f"verify {tag}"), ccl=self.ccl,
                                   rope=self.rope, variants=st.recorded_variants)
            except AssertionError as e:
                ok, err, built = False, str(e), []
        for _, obj in built:
            free_object(obj, keep=self.shared)
        del built
        gc.collect()
        t_load = time.perf_counter() - t0
        misses = [r.cache_name for r in rec.records if not r.hit]
        if ok and misses:
            ok, err = False, f"cache misses during verification: {misses[:5]}"
        loaded = {Path(r.path).name for r in rec.records if r.path}
        listed = set(st.names)
        if ok and not loaded <= listed:
            ok, err = False, f"loaded files not listed in the marker: {sorted(loaded - listed)[:5]}"
        if ok and not listed <= loaded:
            ok, err = False, f"listed files no constructor loads: {sorted(listed - loaded)[:5]}"
        # integrity: a mock load parses the file but its no-op device write never reads the data pages, so re-hash
        # every listed file against the sha256 recorded at conversion (reads every byte; ~1-2 GB/s)
        sp = self.stats_path(part)
        stats = read_json(sp)
        t1 = time.perf_counter()
        n_hashed = 0
        if ok and stats is not None:
            want = {k: v.get("sha256") for k, v in (stats.get("files") or {}).items() if v.get("sha256")}
            if want:
                got = hash_files([self.part_dir(part) / n for n in sorted(want)])
                bad = sorted(n for n in want if got.get(n) != want[n])
                n_hashed = len(want)
                if bad:
                    ok, err = False, f"sha256 mismatch: {bad[:5]}"
        t_hash = time.perf_counter() - t1
        res = {"ok": ok, "target": self.target, "at": now_iso(), "seconds": round(time.perf_counter() - t0, 2),
               "load_seconds": round(t_load, 2), "hash_seconds": round(t_hash, 2), "tensors_loaded": len(rec.records),
               "files_hashed": n_hashed, "files_listed": len(listed), "error": err}
        if stats is not None:
            stats["verify"] = res
            write_json_atomic(sp, stats)
        self.log(f"{tag}: verify from the cache alone ({self.target}): {'OK' if ok else 'FAILED'} "
                 f"({len(rec.records)} tensors loaded with a raising source in {t_load:.1f} s, {n_hashed}/{len(listed)} "
                 f"files re-hashed in {t_hash:.1f} s){'' if ok else ' -- ' + str(err)}")
        if not ok:
            raise ConvertError(f"{tag}: verification failed: {err}", EXIT_VERIFY)
        return res

    # ---- many parts -----------------------------------------------------------------------------------------------
    def run(self, parts: Sequence[Part], *, force: bool = False, verify: bool = True,
            verify_only: bool = False) -> Dict[str, dict]:
        cache_dir = self.cfg.cache_dir
        self.log(f"cache dir {cache_dir} (root {self.cfg.tt_cache_root}, tag {self.cfg.cache_version_tag}); target "
                 f"{self.target}; parts {[part_tag(p) for p in parts]}; variants {self.options.all_variants(self.cfg)}; "
                 f"min free {self.min_free_gb:g} GB; {gb(free_bytes(self.cfg.tt_cache_root)):.1f} GB free; code "
                 f"fingerprint {self.fingerprint['sha256'][:16]}")
        with converter_lock(cache_dir):
            if self.staging_parent.exists():  # stale staging of crashed runs (the lock excludes live converters)
                for d in self.staging_parent.iterdir():
                    shutil.rmtree(d, ignore_errors=True)
            try:
                for part in parts:
                    tag = part_tag(part)
                    t0 = time.perf_counter()
                    st = self.status(part)
                    entry: Dict[str, Any] = {"part": tag}
                    if verify_only:
                        if not st.complete:  # report and go on: --verify-only --layers all on a partial cache
                            self.log(f"{tag}: not complete ({st.reason}), not verified")
                            entry.update(incomplete=st.reason)
                        else:
                            entry["verify"] = self.verify_part(part)
                    elif st.complete and not force:
                        self.log(f"{tag}: already complete ({st.reason}; {st.files} files, {gb(st.bytes):.3f} GB), "
                                 f"skipped")
                        entry.update(skipped=True, bytes=st.bytes)
                        if verify and st.verified is not True:
                            try:
                                entry["verify"] = self.verify_part(part)
                            except ConvertError as e:  # an unverified complete part that fails: rebuild it once
                                if e.code != EXIT_VERIFY:
                                    raise
                                self.log(f"{tag}: rebuilding after the failed verification")
                                entry.update(skipped=False, stats=self.convert_part(part))
                                entry["verify"] = self.verify_part(part)
                    else:
                        reuse = st.base_ok and bool(st.missing) and not force
                        if reuse:
                            self.log(f"{tag}: adding variants {st.missing} (reusing its {st.files} files)")
                        elif st.reason != "no .complete marker" or force:
                            self.log(f"{tag}: (re)converting: {'forced' if force else st.reason}")
                        entry["stats"] = self.convert_part(part, reuse=reuse)
                        if verify:
                            entry["verify"] = self.verify_part(part)
                    entry["seconds"] = round(time.perf_counter() - t0, 2)
                    self.report[tag] = entry
            finally:
                shutil.rmtree(self.staging_root, ignore_errors=True)
                with contextlib.suppress(OSError):
                    self.staging_parent.rmdir()  # only when empty
        return self.report

    def close(self) -> None:
        try:
            self.rope.release_prefill_tables()
        except Exception:
            pass
        free_object(self.rope, keep=(self.cfg, self.mesh))
        if "models.demos.motif3.tt.mhc" in sys.modules:  # the stock backend's shared per-mesh constants
            sys.modules["models.demos.motif3.tt.mhc"].release_shared(self.mesh)
        if hasattr(self.source.inner, "close"):
            self.source.inner.close()


# ======================================================================================================================
# CLI
# ======================================================================================================================
def build_cfg(mesh, *, cache_root: Optional[str], weights_dir: Optional[str]):
    """The serving config of the opened mesh (``from_hf_config(mesh_device=mesh)``) with every decoder layer
    (``MOTIF3_NUM_LAYERS`` would truncate it) and the optional root / weights overrides."""
    from models.demos.motif3.tt.model_config import MotifTTConfig

    kw: Dict[str, Any] = {}
    if cache_root:
        kw["tt_cache_root"] = Path(cache_root)
    if weights_dir:
        kw["weights_dir"] = Path(weights_dir)
    src = Path(weights_dir) if weights_dir else None
    cfg = MotifTTConfig.from_hf_config(src, mesh_device=mesh, **kw)
    if cfg.num_layers != cfg.num_hidden_layers:
        cfg = MotifTTConfig.from_hf_config(src, mesh_device=mesh, num_layers=cfg.num_hidden_layers, **kw)
    return cfg


def status_document(cfg, parts: Sequence[Part], options: ConvertOptions,
                    fingerprint: Optional[Dict[str, Any]] = None) -> dict:
    """The ``--status --json`` document (``cache_root`` first: scripts/stream_weights.py finds it by that key)."""
    fp = fingerprint or code_fingerprint()
    rows = [part_status(cfg, p, options) for p in parts]
    doc = {"cache_root": str(cfg.tt_cache_root), "cache_tag": cfg.cache_version_tag, "cache_dir": str(cfg.cache_dir),
           "mesh_shape": list(cfg.mesh_shape)}
    doc.update(serving_cache(cfg))
    doc.update({"weights_dir": str(cfg.weights_dir), "variants": options.all_variants(cfg),
                "code_fingerprint": fp["sha256"], "code_files": fp["files"],
                "free_gb": round(gb(free_bytes(cfg.tt_cache_root)), 2), "parts": [r.as_dict() for r in rows]})
    return doc


def print_status(cfg, parts: Sequence[Part], options: ConvertOptions, as_json: bool = False) -> int:
    """Per-part status of ``cfg.cache_dir`` (no device). Returns the number of incomplete parts."""
    doc = status_document(cfg, parts, options)
    if as_json:
        print(json.dumps(doc, indent=1))
        return sum(not r["complete"] for r in doc["parts"])
    tot = 0
    fp = doc["code_fingerprint"]
    for r in doc["parts"]:
        tot += r["bytes"]
        v = {True: "verified", False: "VERIFY FAILED", None: "not verified"}[r["verified"]]
        if r["verified"] is True and r["files_hashed"] != r["files"]:
            v = f"verified, {r['files_hashed'] or 0}/{r['files']} hashed"  # (never released by stream_weights.py)
        sec = r["build_seconds"]
        code = "-" if not r["complete"] else ("code current" if r["code_fingerprint"] == fp else "other code")
        print(f"{r['part']:7s} {'complete' if r['complete'] else 'missing ':8s} {r['files']:3d} files "
              f"{gb(r['bytes']):8.3f} GB  {v:13s} {code:12s} {('build ' + str(sec) + ' s') if sec is not None else '':14s} "
              f"{r['reason']}")
    print(f"total {gb(tot):.2f} GB in {cfg.cache_dir} ({doc['free_gb']:.1f} GB free); serving cache: "
          f"{doc['is_serving_cache']} ({doc['serving_cache_dir']}); code fingerprint {fp[:16]}")
    return sum(not r["complete"] for r in doc["parts"])


def resolve_mesh(arg: Optional[str]) -> Tuple[int, int]:
    """``--mesh`` or the serving mesh (4, 8). ``MESH_DEVICE`` is ignored on purpose (``BH-Galaxy`` means (8, 4))."""
    if arg:
        return parse_mesh(arg)
    env = (os.environ.get("MESH_DEVICE") or "").strip()
    if env:
        try:
            from models.demos.motif3.tt import model_config as MC

            env_shape = tuple(MC.mesh_shape_from_env(SERVING_MESH))
        except Exception:
            env_shape = None
        if env_shape != SERVING_MESH:
            print(f"[convert] note: MESH_DEVICE={env!r} ignored: the converter builds the serving (4, 8) cache unless "
                  f"--mesh says otherwise", file=sys.stderr, flush=True)
    return SERVING_MESH


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="convert_weights.py",
        description="Build the Motif-3 TT weight cache (see the module docstring and docs/WEIGHTS_RUNBOOK.md).",
    )
    ap.add_argument("--layers", default=None, help="decoder layers: 0-52 | all | 3,4,10-12 | 36- | none "
                    "(default: all when --globals is not given either)")
    ap.add_argument("--globals", dest="globals_", action="store_true", help="also convert the globals")
    ap.add_argument("--no-globals", dest="no_globals", action="store_true", help="never convert the globals")
    ap.add_argument("--target", choices=("mock", "device"), default="mock")
    ap.add_argument("--mock-desc", default=str(DEFAULT_MOCK_DESC), help="mock cluster descriptor (--target mock)")
    ap.add_argument("--mesh", default=None, help="mesh shape (default 4x8, the serving mesh; MESH_DEVICE is ignored)")
    ap.add_argument("--cache-root", default=None, help="TT cache root (default: MOTIF3_TT_CACHE_PATH > TT_CACHE_PATH > "
                    "motif-3/tt_cache, the serving order)")
    ap.add_argument("--weights-dir", default=None, help="BF16 checkpoint dir (default: the serving resolution)")
    ap.add_argument("--router", choices=ROUTER_CHOICES, default="both",
                    help="MoE layers; composite is always built (default both: + the exact-fp32 router weight)")
    ap.add_argument("--mhc-sinkhorn", choices=SINKHORN_CHOICES, default="both",
                    help="every layer; motif is always built (default both: + the stock fallback constants)")
    ap.add_argument("--lm-head-split", choices=LM_HEAD_CHOICES, default="mesh",
                    help="globals; mesh is always built (tp / both: + the TP head, 1.8 GB)")
    ap.add_argument("--embedding", choices=EMBED_CHOICES, default="replicated",
                    help="globals; replicated is always built (sharded / both: + the hidden-sharded table, 1.8 GB)")
    ap.add_argument("--no-hash", action="store_true", help="skip the per-file sha256 (the part then never qualifies "
                    "for BF16 deletion by scripts/stream_weights.py)")
    ap.add_argument("--force", action="store_true", help="rebuild complete parts from BF16 (keeps recorded variants)")
    ap.add_argument("--no-verify", action="store_true", help="skip the cache-only rebuild after converting")
    ap.add_argument("--verify-only", action="store_true", help="only verify complete parts")
    ap.add_argument("--status", action="store_true", help="print the per-part status and exit (no device, no mesh)")
    ap.add_argument("--json", action="store_true", help="--status as JSON (cache dir, serving cache, fingerprint, "
                    "per-part records and estimates)")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB, help=f"free-space floor (default {MIN_FREE_GB:g})")
    ap.add_argument("--report", default=None, help="write the run report (JSON) here")
    ap.add_argument("--allow-visible-devices", action="store_true",
                    help="--target mock without scripts/hostrun.sh (devices visible; not recommended)")
    ap.add_argument("--no-lock-check", action="store_true", help="--target device without the devrun.sh lock check")
    a = ap.parse_args(argv)

    if a.layers is None and not a.globals_:
        layer_spec, want_globals = "all", not a.no_globals
    else:
        layer_spec, want_globals = (a.layers or "none"), (a.globals_ and not a.no_globals)
    try:
        options = ConvertOptions(router=a.router, mhc_sinkhorn=a.mhc_sinkhorn, lm_head_split=a.lm_head_split,
                                 embedding=a.embedding, hash_files=not a.no_hash)
        mesh_arg = parse_mesh(a.mesh) if a.mesh else None
    except ValueError as e:
        log(str(e))
        return EXIT_USAGE

    if a.status:  # host-only: the config from the mesh SHAPE (no device, no mesh, no ttnn tensors)
        os.environ.setdefault("LOGURU_LEVEL", "INFO")
        from models.demos.motif3.tt.model_config import MotifTTConfig

        shape = mesh_arg or resolve_mesh(None)
        kw: Dict[str, Any] = {"mesh_shape": shape}
        if a.cache_root:
            kw["tt_cache_root"] = Path(a.cache_root)
        if a.weights_dir:
            kw["weights_dir"] = Path(a.weights_dir)
        cfg = MotifTTConfig.from_hf_config(Path(a.weights_dir) if a.weights_dir else None, **kw)
        layers = parse_layers(layer_spec, cfg.num_hidden_layers)
        if cfg.num_layers != cfg.num_hidden_layers:
            cfg = MotifTTConfig.from_hf_config(Path(a.weights_dir) if a.weights_dir else None,
                                               num_layers=cfg.num_hidden_layers, **kw)
        print_status(cfg, ([None] if want_globals else []) + layers, options, as_json=a.json)
        return EXIT_OK

    # ---- environment guards, BEFORE ttnn is imported (the mock target is selected through the environment) ----------
    if a.target == "mock":
        vis = devices_visible()
        if vis and not a.allow_visible_devices:
            log(f"refusing --target mock with /dev/tenstorrent visible ({len(vis)} entries): run under "
                f"scripts/hostrun.sh (devices hidden), or pass --allow-visible-devices")
            return EXIT_USAGE
        desc = os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH") or a.mock_desc
        if not Path(desc).is_file():
            log(f"mock cluster descriptor not found: {desc}")
            return EXIT_USAGE
        os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(desc)
    else:
        if os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
            log("refusing --target device with TT_METAL_MOCK_CLUSTER_DESC_PATH set")
            return EXIT_USAGE
        if not a.no_lock_check and not (under_devrun() and device_lock_held_by_someone()):
            log(f"refusing --target device outside scripts/devrun.sh (it holds the device lock {DEVICE_LOCK})")
            return EXIT_USAGE
    os.environ.setdefault("LOGURU_LEVEL", "INFO")  # ttnn's per-tensor "Generating cache" debug lines

    import ttnn  # noqa: F401
    from models.demos.motif3.tt import model_config as MC
    from models.demos.motif3.tt.ccl import log_fabric

    shape = mesh_arg or resolve_mesh(None)
    if tuple(shape) != SERVING_MESH:
        log(f"WARNING: mesh {shape} (--mesh): serving uses {SERVING_MESH} (cache directory mesh4x8); this cache is not "
            f"the one the server opens")
    t0 = time.time()
    mesh = MC.open_motif_mesh(shape)
    rc = EXIT_OK
    conv = None
    try:
        log_fabric(mesh, f"convert_weights ({a.target})", printer=log)
        cfg = build_cfg(mesh, cache_root=a.cache_root, weights_dir=a.weights_dir)
        layers = parse_layers(layer_spec, cfg.num_hidden_layers)
        parts: List[Part] = ([None] if want_globals else []) + layers
        serving = serving_cache(cfg)
        log(f"mesh {tuple(mesh.shape)} opened in {time.time() - t0:.1f} s ({a.target}); weights {cfg.weights_dir}; "
            f"cache {cfg.cache_dir} (serving cache: {serving['is_serving_cache']})")
        if not serving["is_serving_cache"]:
            log(f"WARNING: {cfg.cache_dir} is not the serving cache {serving['serving_cache_dir']} (mesh "
                f"{tuple(cfg.mesh_shape)}, tag {cfg.cache_version_tag}; TT_MODEL_WEIGHTS_REVISION / --mesh?)")
        conv = Converter(mesh, cfg, options=options, target=a.target, min_free_gb=a.min_free_gb)
        try:
            report = conv.run(parts, force=a.force, verify=not a.no_verify, verify_only=a.verify_only)
        except ConvertError as e:
            log(f"STOP: {e}")
            rc = e.code
            report = conv.report
        n_conv = sum(1 for v in report.values() if v.get("stats"))
        total_b = sum((v.get("stats") or {}).get("bytes", v.get("bytes", 0)) or 0 for v in report.values())
        log(f"done: {len(report)} parts ({n_conv} converted), {gb(total_b):.2f} GB in the touched parts, "
            f"{time.time() - t0:.1f} s wall; {gb(free_bytes(cfg.tt_cache_root)):.1f} GB free (exit {rc})")
        if a.report:
            write_json_atomic(Path(a.report), {"argv": list(argv if argv is not None else sys.argv[1:]),
                                               "target": a.target, "cache_dir": str(cfg.cache_dir), "exit": rc,
                                               "code_fingerprint": conv.fingerprint["sha256"], "parts": report})
    except ConvertError as e:
        log(f"STOP: {e}")
        rc = e.code
    finally:
        if conv is not None:
            conv.close()
        MC.close_motif_mesh(mesh)
    return rc


if __name__ == "__main__":
    sys.exit(main())
