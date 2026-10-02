#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Disk-limited streaming pipeline for the Motif-3 TT weight cache (WAVE_A_REVIEW CONV-3; design §6 option B).

The BF16 checkpoint (629.7 GB) and the bfp8 TT cache (~344 GB) do not fit on this host's disk together, so the cache is
built one decoder layer at a time and a BF16 shard can be deleted once nothing needs it any more::

    globals, then for L in --layers (ascending):
      1. shards   make sure every shard of L is on disk: scripts/download_weights.py --layers L, with a margin that keeps
                  --min-free-gb (60) free after the download AND after L's cache files
      2. sha256   scripts/verify_shards.py (hashes new shards against hf_meta/tree.json); every shard of L must be "ok",
                  and a record counts only while the file still has the size and mtime that were hashed (a shard that
                  was deleted and downloaded again is hashed again)
      3. golden   layers past the C2 golden's last processed layer (36-52 today):
                  reference/golden_stream.py --resume --layers L --ckpt-dir <checkpoint> (CPU, devices hidden), BEFORE
                  L's shards can go; then the same for every --extra-golden-out run (e.g. the fp32 golden
                  goldens/c2_fp32), in order. A layer counts as golden-done only when EVERY golden lists it
      4. convert  scripts/convert_weights.py --layers L --mesh 4x8 --cache-root <root> --weights-dir <checkpoint>
                  (mock cluster, host only; or --target device under the lock). Only the variant options pass through
                  --convert-arg; the pipeline sets everything else
      5. verify   done by the converter: the part is rebuilt from the TT cache alone with a raising source and every
                  file is re-hashed against the sha256 recorded at conversion; the pipeline requires verify.ok with
                  every file hashed
      6. delete   ONLY with --delete-converted, under the converter lock (no conversion runs meanwhile), after
                  re-reading the converter status: every shard whose decoder layers are all
                    - converted + verified (every listed file re-hashed) in the SERVING cache directory
                      (<root>/<serving tag>/mesh4x8: mesh (4, 8), pinned revision, default dtype policy),
                    - built by the current code (the part's code fingerprint equals the current one; a converted layer
                      whose fingerprint differs is rebuilt first, while its shards are still there),
                    - in the layers_done of the C2 golden and of every --extra-golden-out golden,
                    - not in --keep-bf16 (default 0-35: the bring-up / acceptance-test layers; release them with
                      --keep-bf16 none at sign-off);
                  never a shard holding a global tensor (embedding / final norm / LM head / MTP: the downloader
                  re-fetches those on every call, and the golden's final head needs shard 104), never the smallest
                  shard (TIS's --host-weights-dir check needs one model*.safetensors), never config / tokenizer / index
                  files. The deletion function re-derives all of this itself from the status document.

Without ``--delete-converted`` nothing is deleted; the shards that *would* go are listed. Every step re-checks the free
space (``--min-free-gb``, default 60 GB, on the weights and the cache filesystems) and stops (exit 3) rather than go
below it; a converter disk-guard exit triggers one deletion pass and one retry. The pipeline is restartable: each step
reads the ground truth (shard sizes, ``.verified.json`` + file stats, the golden ``manifest.json``, the converter's
``--status --json``) and skips what is done. ``--dry-run`` prints the plan and the projected free space per step,
applying the same guards (estimates x 1.1, the golden reserve, the variant sizes).

Run it on the host (plain python: it imports no ttnn and opens no device; the steps that do run through
``scripts/hostrun.sh`` / ``scripts/devrun.sh``)::

    python3 /home/ttuser/hchang/experiments/motif-3/scripts/stream_weights.py --dry-run --delete-converted
    nohup python3 .../stream_weights.py --delete-converted > logs/stream/stream_$(date +%s).log 2>&1 &

Exit codes: 0 done, 1 error, 2 usage (also: the converter's cache is not the serving cache), 3 disk guard, 4 download
incomplete / BF16 missing, 5 sha256 mismatch, 6 golden failed, 7 conversion / verification failed.
Runbook: ``docs/WEIGHTS_RUNBOOK.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fcntl
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

ROOT = Path("/home/ttuser/hchang/experiments/motif-3")
SCRIPTS = ROOT / "scripts"
METAL = ROOT / "tt-metal"
PYTHON = METAL / "python_env" / "bin" / "python"
WEIGHTS = ROOT / "weights" / "Motif-3"  # download_weights.py / verify_shards.py DEST (hard-coded there)
TREE = ROOT / "hf_meta" / "tree.json"
GOLDEN_OUT = ROOT / "goldens" / "c2"
HOSTRUN = SCRIPTS / "hostrun.sh"
DEVRUN = SCRIPTS / "devrun.sh"
DOWNLOAD = SCRIPTS / "download_weights.py"
VERIFY = SCRIPTS / "verify_shards.py"
CONVERT = SCRIPTS / "convert_weights.py"
STATE_NAME = ".stream_state.json"
LOCK_NAME = ".stream.lock"
N_LAYERS = 53
N_DENSE = 2  # layers 0-1 dense, 2-52 MoE (config n_dense_first_layers)


def _load_converter():
    """``scripts/convert_weights.py`` as a module (its top level is stdlib only: no ttnn): the size estimates, the disk
    guard, the variant model and the exit codes are the converter's own. One instance per process (the tests load it
    under the same name)."""
    name = "motif3_scripts_convert_weights"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CONVERT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CW = _load_converter()

MIN_FREE_GB = CW.MIN_FREE_GB  # 60 (design §6 option B, CONV-3)
GOLDEN_RESERVE_GB = 0.5  # resume state rewrite (97 MB) + saved states / final head of one call (<= 0.3 GB)
GOLDEN_STEP_GB = 0.1  # projected net growth of the golden per layer (dry run)
GUARD_FACTOR = CW.GUARD_FACTOR
EST_PART_BYTES = CW.EST_PART_BYTES  # default-options part sizes (runbook)
DEFAULT_KEEP_BF16 = "0-35"  # the bring-up layers: decoder-layer / truncated-model / serving-order tests, GEN-7, C3
SERVING_MESH_ARG = "4x8"
SERVING_MESH_DIR = "mesh4x8"
PASS_THROUGH = ("--router", "--mhc-sinkhorn", "--lm-head-split", "--embedding", "--mock-desc")

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_DISK, EXIT_DOWNLOAD, EXIT_SHA, EXIT_GOLDEN, EXIT_CONVERT = 0, 1, 2, 3, 4, 5, 6, 7

Unit = Union[int, str]  # a decoder layer index, or "global:<tensor name>"


class StreamError(RuntimeError):
    def __init__(self, msg: str, code: int = EXIT_ERROR):
        super().__init__(msg)
        self.code = code


def log(msg: str) -> None:
    print(f"[stream {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def gb(n: float) -> float:
    return float(n) / 1e9


def free_bytes(path: os.PathLike) -> int:
    p = Path(path)
    while not p.exists():
        p = p.parent
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize


def parse_layers(spec: Optional[str], n_layers: int = N_LAYERS) -> List[int]:
    if spec is None or str(spec).strip().lower() in ("", "all"):
        return list(range(n_layers))
    if str(spec).strip().lower() == "none":
        return []
    out: Set[int] = set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a), (int(b) if b.strip() else n_layers - 1) + 1))
        else:
            out.add(int(tok))
    bad = sorted(i for i in out if not 0 <= i < n_layers)
    if bad:
        raise ValueError(f"layers {bad} outside [0, {n_layers})")
    return sorted(out)


def layer_ranges(layers: Iterable[int]) -> str:
    """``{0, 1, 2, 5}`` -> ``"0-2,5"`` (logs)."""
    ls = sorted(set(int(l) for l in layers))
    if not ls:
        return "none"
    out, a, b = [], ls[0], ls[0]
    for x in ls[1:] + [None]:
        if x is not None and x == b + 1:
            b = x
            continue
        out.append(f"{a}-{b}" if b > a else f"{a}")
        if x is not None:
            a = b = x
    return ",".join(out)


def part_kind(layer: Optional[int]) -> str:
    if layer is None:
        return "global"
    return "dense" if layer < N_DENSE else "moe"


def part_tag(layer: Optional[int]) -> str:
    return "global" if layer is None else f"L{int(layer):02d}"


def default_cache_root(environ=None) -> Path:
    """The serving resolution of the TT cache root (``generator_api.resolve_tt_cache_path``): ``MOTIF3_TT_CACHE_PATH`` >
    ``TT_CACHE_PATH`` > ``motif-3/tt_cache``."""
    env = os.environ if environ is None else environ
    for name in ("MOTIF3_TT_CACHE_PATH", "TT_CACHE_PATH"):
        v = (env.get(name) or "").strip()
        if v:
            return Path(v)
    return ROOT / "tt_cache"


class _StrictParser(argparse.ArgumentParser):
    def error(self, message):  # no SystemExit from a library call
        raise ValueError(message)


def normalize_convert_args(args: Sequence[str]) -> List[str]:
    """``--convert-arg`` values -> ``--opt=value`` for the variant options (and ``--mock-desc``) only. Everything else
    is the pipeline's (layers, globals, target, mesh, cache root, checkpoint, floor, force) or would make a part
    ineligible for deletion (``--no-hash``, ``--no-verify``): ValueError."""
    p = _StrictParser(prog="--convert-arg", add_help=False, allow_abbrev=False)
    p.add_argument("--router", choices=CW.ROUTER_CHOICES)
    p.add_argument("--mhc-sinkhorn", choices=CW.SINKHORN_CHOICES)
    p.add_argument("--lm-head-split", choices=CW.LM_HEAD_CHOICES)
    p.add_argument("--embedding", choices=CW.EMBED_CHOICES)
    p.add_argument("--mock-desc")
    ns, unknown = p.parse_known_args([str(a) for a in args])
    if unknown:
        raise ValueError(f"--convert-arg {unknown}: only {list(PASS_THROUGH)} pass through to the converter (the "
                         f"pipeline sets --layers/--globals/--target/--mesh/--cache-root/--weights-dir/--min-free-gb/"
                         f"--force itself and every part it relies on is hashed and verified)")
    out = []
    for opt in PASS_THROUGH:
        v = getattr(ns, opt[2:].replace("-", "_"))
        if v is not None:
            out.append(f"{opt}={v}")
    return out


# ======================================================================================================================
# the shard index (pure; no ttnn)
# ======================================================================================================================
class ShardIndex:
    """``model.safetensors.index.json`` + ``hf_meta/tree.json``: which units (decoder layers, global tensors) each shard
    holds, shard sizes, and presence on disk (a shard is present only with its full size)."""

    LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")

    def __init__(self, weight_map: Dict[str, str], sizes: Dict[str, int], weights_dir: Path):
        self.weights_dir = Path(weights_dir)
        self.sizes = {s: int(sizes[s]) for s in set(weight_map.values()) if s in sizes}
        missing = sorted(set(weight_map.values()) - set(self.sizes))
        if missing:
            raise ValueError(f"{len(missing)} shards of the index have no size in tree.json (e.g. {missing[:2]})")
        self.shard_units: Dict[str, Set[Unit]] = {}
        self.layer_shards: Dict[int, Set[str]] = {}
        for name, shard in weight_map.items():
            m = self.LAYER_RE.match(name)
            unit: Unit = int(m.group(1)) if m else f"global:{name}"
            self.shard_units.setdefault(shard, set()).add(unit)
            if m:
                self.layer_shards.setdefault(int(m.group(1)), set()).add(shard)

    @classmethod
    def load(cls, weights_dir: Path = WEIGHTS, tree: Path = TREE) -> "ShardIndex":
        wm = json.loads((Path(weights_dir) / "model.safetensors.index.json").read_text())["weight_map"]
        sizes = {e["path"]: int(e["size"]) for e in json.loads(Path(tree).read_text()) if e.get("type") == "file"}
        return cls(wm, sizes, weights_dir)

    @property
    def shards(self) -> List[str]:
        return sorted(self.shard_units)

    def present(self, shard: str) -> bool:
        p = self.weights_dir / shard
        return p.is_file() and p.stat().st_size == self.sizes[shard]

    def present_shards(self) -> Set[str]:
        return {s for s in self.shards if self.present(s)}

    def missing_of(self, layer: int) -> List[str]:
        return sorted(s for s in self.layer_shards.get(layer, ()) if not self.present(s))

    def global_shards(self) -> Set[str]:
        return {s for s, us in self.shard_units.items() if any(isinstance(u, str) for u in us)}

    def global_tensors_of(self, shard: str) -> List[str]:
        return sorted(u[len("global:"):] for u in self.shard_units[shard] if isinstance(u, str))

    def smallest_shard(self) -> str:
        return min(self.shards, key=lambda s: (self.sizes[s], s))

    def keep_shards(self) -> Set[str]:
        """Never deleted: shards holding any global tensor (embedding / final norm / LM head / MTP) and the smallest
        shard (TIS's --host-weights-dir check needs one model*.safetensors)."""
        return self.global_shards() | {self.smallest_shard()}

    def layers_of(self, shard: str) -> List[int]:
        return sorted(u for u in self.shard_units[shard] if isinstance(u, int))

    def releasable(self, layer: int, keep_layers: Iterable[int] = (), present: Optional[Set[str]] = None) -> List[str]:
        """Shards of ``layer`` that a deletion could ever remove: present, not kept for good, holding no global tensor
        and no layer of ``keep_layers`` (whether their other layers are done is not checked)."""
        kl = set(int(l) for l in keep_layers)
        keep = self.keep_shards()
        out = []
        for s in sorted(self.layer_shards.get(int(layer), ())):
            if s in keep or not (self.present(s) if present is None else s in present):
                continue
            if any(isinstance(u, str) or u in kl for u in self.shard_units[s]):
                continue
            out.append(s)
        return out


def deletable_shards(index: ShardIndex, done_layers: Iterable[int], *, present: Optional[Set[str]] = None,
                     keep: Optional[Set[str]] = None, keep_layers: Iterable[int] = ()) -> List[str]:
    """Shards that may be deleted: present, not in ``keep`` (default ``index.keep_shards()``), holding no global tensor
    and no layer of ``keep_layers``, and every decoder layer they hold is in ``done_layers``. Pure function."""
    done = set(int(l) for l in done_layers)
    kl = set(int(l) for l in keep_layers)
    keep = index.keep_shards() if keep is None else set(keep)
    present = index.present_shards() if present is None else set(present)
    out = []
    for s in index.shards:
        if s not in present or s in keep:
            continue
        units = index.shard_units[s]
        if any(isinstance(u, str) for u in units) or any(u in kl for u in units):
            continue
        if all(u in done for u in units):
            out.append(s)
    return out


# ======================================================================================================================
# the converter's status document (pure readers)
# ======================================================================================================================
def part_entry(status: dict, layer: Optional[int]) -> dict:
    for p in status.get("parts", []):
        if p.get("layer") == layer:
            return p
    return {}


def part_done(p: dict) -> bool:
    """Converted + verified, every listed file with a recorded sha256 and re-hashed by the verification."""
    n = int(p.get("files") or 0)
    return (bool(p.get("complete")) and p.get("verified") is True and n > 0 and p.get("hashes") == n
            and p.get("files_hashed") == n)


def conversion_bytes(p: dict, *, force: bool, full: int) -> int:
    """Bytes a converter run adds for status part ``p`` (what both guards reserve x 1.1): a forced rebuild the whole
    part (transiently), else the converter's own estimate (0 for a complete part that only needs its verification,
    the missing variant files, or the whole part); ``full`` when the part has no record."""
    key = "rebuild_bytes" if force else "need_bytes"
    v = p.get(key)
    return int(full if v is None or (force and not v) else v)


def fingerprint_current(status: dict, p: dict) -> bool:
    """The part was built by the current code (its recorded code fingerprint equals the status' current one)."""
    fp = status.get("code_fingerprint")
    return bool(fp) and p.get("code_fingerprint") == fp


def check_serving_cache(status: dict, cache_root: Path) -> Path:
    """The status document must describe the serving cache under ``cache_root``: mesh (4, 8), the serving tag (pinned
    revision, default dtype policy), ``<cache_root>/<serving tag>/mesh4x8``. Returns that directory; StreamError
    (exit 2) otherwise, e.g. a ``TT_MODEL_WEIGHTS_REVISION`` or an old converter without these fields."""
    tag = status.get("serving_tag")
    problems = []
    if not tag:
        problems.append("the status has no serving_tag (converter too old?)")
    exp = Path(cache_root) / str(tag) / SERVING_MESH_DIR
    if list(status.get("mesh_shape") or []) != list(CW.SERVING_MESH):
        problems.append(f"mesh {status.get('mesh_shape')} != {list(CW.SERVING_MESH)}")
    if tag and status.get("cache_tag") != tag:
        problems.append(f"cache tag {status.get('cache_tag')} != serving tag {tag} (TT_MODEL_WEIGHTS_REVISION?)")
    if os.path.realpath(str(status.get("cache_dir"))) != os.path.realpath(exp):
        problems.append(f"cache dir {status.get('cache_dir')} != {exp}")
    if status.get("is_serving_cache") is not True:
        problems.append("the converter does not call it the serving cache")
    if problems:
        raise StreamError("the converter's cache is not the serving cache: " + "; ".join(problems), EXIT_USAGE)
    return exp


def eligible_layers(status: dict, golden_done: Iterable[int], keep_layers: Iterable[int] = ()) -> Set[int]:
    """Decoder layers whose BF16 may go: done (:func:`part_done`), built by the current code, golden done, not kept."""
    golden = set(int(l) for l in golden_done)
    keep = set(int(l) for l in keep_layers)
    out = set()
    for p in status.get("parts", []):
        l = p.get("layer")
        if l is None or l in keep or l not in golden:
            continue
        if part_done(p) and fingerprint_current(status, p):
            out.add(int(l))
    return out


def delete_shards(index: ShardIndex, shards: Sequence[str], *, allow: bool, status: dict, golden_done: Iterable[int],
                  cache_root: Path, keep_layers: Iterable[int] = (), state: Optional["StreamState"] = None,
                  logger: Callable[[str], None] = log) -> int:
    """Delete ``shards`` from the weights dir. Refuses unless ``allow`` (``--delete-converted``) and ``status`` is the
    serving cache under ``cache_root``; re-derives the eligible layers from ``status`` + ``golden_done`` +
    ``keep_layers`` itself and refuses any shard :func:`deletable_shards` does not list. Only ``model-*.safetensors``
    files are touched; the filesystem is synced first (the converter fsyncs its files too). Returns the bytes freed."""
    if not allow:
        raise StreamError("refusing to delete BF16 shards without --delete-converted", EXIT_USAGE)
    check_serving_cache(status, cache_root)
    done = eligible_layers(status, golden_done, keep_layers)
    ok = set(deletable_shards(index, done, keep_layers=keep_layers))
    for s in shards:
        if s not in ok:
            raise StreamError(f"refusing to delete {s}: not deletable (layers {index.layers_of(s)}: not all converted + "
                              f"verified + current code + golden done, kept, or absent)", EXIT_ERROR)
        if not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", s):
            raise StreamError(f"refusing to delete {s}: not a model shard", EXIT_ERROR)
    if shards:
        os.sync()
    freed = 0
    for s in shards:
        p = index.weights_dir / s
        n = p.stat().st_size
        p.unlink()
        freed += n
        logger(f"  deleted {s} ({gb(n):.2f} GB; layers {index.layers_of(s)})")
        if state is not None:
            state.record_deletion(s, n, index.layers_of(s))
    return freed


# ======================================================================================================================
# persistent state (an audit log; every decision is re-derived from the ground truth)
# ======================================================================================================================
class StreamState:
    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError):
            self.data = {"format": "motif3-stream/1", "layers": {}, "deleted": [], "runs": []}

    def layer(self, l: Union[int, str]) -> dict:
        return self.data["layers"].setdefault(str(l), {})

    def mark(self, l: Union[int, str], step: str, **extra) -> None:
        rec = self.layer(l)
        rec[step] = now_iso()
        rec.update(extra)
        self.save()

    def record_deletion(self, shard: str, nbytes: int, layers: List[int]) -> None:
        self.data["deleted"].append({"shard": shard, "bytes": int(nbytes), "layers": layers, "at": now_iso()})
        self.save()

    def save(self) -> None:
        tmp = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(self.data, indent=1))
        os.replace(tmp, self.path)


# ======================================================================================================================
# ground truth readers
# ======================================================================================================================
def verified_shards(weights_dir: Path) -> Dict[str, str]:
    """``.verified.json`` of scripts/verify_shards.py as {shard: "ok" | "BAD" | "missing" | "stale"}. "ok" holds only
    while the file exists with the size and mtime that were hashed; a changed file (e.g. deleted, then downloaded
    again: verify_shards.py keeps a deleted shard's old record) is "stale" until verify_shards.py hashes it again."""
    try:
        d = json.loads((Path(weights_dir) / ".verified.json").read_text())
    except (FileNotFoundError, ValueError):
        return {}
    out = {}
    for k, v in d.items():
        status = v.get("status") if isinstance(v, dict) else str(v)
        stat = v.get("stat") if isinstance(v, dict) else None
        p = Path(weights_dir) / k
        if not p.is_file():
            out[k] = "missing"
            continue
        if status == "ok":
            st = p.stat()
            if not isinstance(stat, list) or [st.st_size, st.st_mtime] != stat:
                status = "stale"
        out[k] = status
    return out


def golden_last_layer(golden_out: Path) -> Optional[int]:
    try:
        m = json.loads((Path(golden_out) / "manifest.json").read_text())
    except (FileNotFoundError, ValueError):
        return None
    v = m.get("last_layer")
    return None if v is None else int(v)


def golden_done_layers(golden_out: Path) -> Set[int]:
    try:
        m = json.loads((Path(golden_out) / "manifest.json").read_text())
    except (FileNotFoundError, ValueError):
        return set()
    return {int(i) for i in m.get("layers_done", [])}


def goldens_done_layers(golden_outs: Sequence[Path]) -> Set[int]:
    """Layers every golden run of ``golden_outs`` has processed (the intersection; empty for no goldens or when any
    manifest is missing / unreadable, so a missing golden never lets a shard go)."""
    outs = list(golden_outs)
    if not outs:
        return set()
    done = golden_done_layers(outs[0])
    for o in outs[1:]:
        done &= golden_done_layers(o)
    return done


# ======================================================================================================================
# subprocess steps
# ======================================================================================================================
@dataclass
class Runner:
    """Builds and runs the step commands (``dry_run`` prints them instead). ``target``: converter target (mock |
    device). Every converter command is pinned to ``--mesh 4x8 --cache-root <cache_root> --weights-dir <weights_dir>``;
    ``variant_args`` are the normalized ``--convert-arg`` values. ``golden_out`` is the C2 golden; ``extra_golden_outs``
    are further golden runs (e.g. the fp32 one) that are resumed the same way and must also list a layer before its
    BF16 can go."""

    target: str = "mock"
    dry_run: bool = False
    weights_dir: Path = WEIGHTS
    golden_out: Path = GOLDEN_OUT
    cache_root: Path = field(default_factory=default_cache_root)
    variant_args: List[str] = field(default_factory=list)
    min_free_gb: float = MIN_FREE_GB
    timeout_s: int = 7200
    extra_golden_outs: List[Path] = field(default_factory=list)

    def __post_init__(self):
        self.weights_dir, self.golden_out, self.cache_root = Path(self.weights_dir), Path(self.golden_out), Path(
            self.cache_root)
        self.variant_args = normalize_convert_args(self.variant_args)
        self.extra_golden_outs = [Path(o) for o in self.extra_golden_outs]
        seen = {os.path.realpath(self.golden_out)}
        for o in self.extra_golden_outs:
            if os.path.realpath(o) in seen:
                raise ValueError(f"--extra-golden-out {o}: given twice or the same as --golden-out")
            seen.add(os.path.realpath(o))

    @property
    def golden_outs(self) -> List[Path]:
        """Every golden run that gates the deletion: the C2 golden first, then the extra ones in order."""
        return [Path(self.golden_out)] + list(self.extra_golden_outs)

    def run(self, cmd: List[str], what: str, *, timeout: Optional[int] = None) -> int:
        log(f"  $ {' '.join(shlex.quote(str(c)) for c in cmd)}")
        if self.dry_run:
            return 0
        t0 = time.time()
        p = subprocess.run([str(c) for c in cmd], timeout=timeout or self.timeout_s)
        log(f"  {what}: exit {p.returncode} in {time.time() - t0:.0f} s")
        return p.returncode

    def capture(self, cmd: List[str], *, timeout: int = 600) -> str:
        p = subprocess.run([str(c) for c in cmd], timeout=timeout, capture_output=True, text=True)
        if p.returncode != 0:
            raise StreamError(f"{cmd[-1]} failed ({p.returncode}): {p.stderr[-2000:]}")
        return p.stdout

    def _require_downloader_dir(self, what: str) -> None:
        if os.path.realpath(self.weights_dir) != os.path.realpath(WEIGHTS):
            raise StreamError(f"{what} only works on {WEIGHTS} (hard-coded there); this pipeline's checkpoint is "
                              f"{self.weights_dir}", EXIT_USAGE)

    # ---- commands (pure) ----
    def pins(self) -> List[str]:
        return ["--mesh", SERVING_MESH_ARG, "--cache-root", str(self.cache_root), "--weights-dir", str(self.weights_dir)]

    def download_cmd(self, layer: int, margin_gb: float) -> List:
        return [PYTHON, DOWNLOAD, "--layers", str(layer), "--margin-gb", f"{margin_gb:.1f}", "--layers-per-batch", "1"]

    def verify_cmd(self) -> List:
        return [PYTHON, VERIFY]

    def golden_cmd(self, layer: int, out: Optional[Path] = None) -> List:
        out = Path(self.golden_out if out is None else out)
        name = f"golden_L{layer:02d}" if out == Path(self.golden_out) else f"golden_{out.name}_L{layer:02d}"
        return [HOSTRUN, "-t", "3600", "-n", name, "--", "python", "-m",
                "models.demos.motif3.reference.golden_stream", "--resume", "--layers", str(layer), "--out",
                str(out), "--ckpt-dir", str(self.weights_dir)]

    def convert_cmd(self, layer: Optional[int], *, force: bool = False) -> List:
        sel = ["--globals", "--layers", "none"] if layer is None else ["--layers", str(layer)]
        tag = part_tag(layer)
        if self.target == "device":
            cmd = [DEVRUN, "-t", "2400", "-n", f"convert_{tag}", "--", "python", CONVERT, "--target", "device"]
        else:
            cmd = [HOSTRUN, "-t", "3600", "-n", f"convert_{tag}", "--", "python", CONVERT, "--target", "mock"]
        return (cmd + sel + ["--min-free-gb", f"{self.min_free_gb:g}"] + self.pins() + list(self.variant_args)
                + (["--force"] if force else []))

    def status_cmd(self, layers: Sequence[int], with_globals: bool) -> List:
        spec = ",".join(str(l) for l in layers) if layers else "none"
        cmd = [HOSTRUN, "-t", "600", "--", "python", CONVERT, "--status", "--json", "--layers", spec]
        if with_globals:
            cmd.append("--globals")
        return cmd + self.pins() + list(self.variant_args)

    # ---- steps ----
    def download(self, layer: int, margin_gb: float) -> int:
        self._require_downloader_dir("scripts/download_weights.py")
        return self.run(self.download_cmd(layer, margin_gb), f"download layer {layer}", timeout=4 * 3600)

    def verify_sha(self) -> int:
        self._require_downloader_dir("scripts/verify_shards.py")
        return self.run(self.verify_cmd(), "sha256 verification", timeout=3 * 3600)

    def golden(self, layer: int, out: Optional[Path] = None) -> int:
        what = f"golden layer {layer}" + ("" if out is None else f" ({Path(out).name})")
        return self.run(self.golden_cmd(layer, out), what, timeout=3700)

    def convert(self, layer: Optional[int], *, force: bool = False) -> int:
        return self.run(self.convert_cmd(layer, force=force), f"convert {part_tag(layer)}", timeout=3700)

    def cache_status(self, layers: Sequence[int], with_globals: bool) -> dict:
        """The converter's own status (authoritative: cache dir, serving cache, fingerprint, per-part records)."""
        out = self.capture(self.status_cmd(layers, with_globals))
        i = out.find('{\n "cache_root"')  # the status document (ttnn may print other braces, e.g. its Config{...})
        if i < 0:
            raise StreamError(f"no status JSON in the converter output: {out[-500:]}")
        doc, _ = json.JSONDecoder().raw_decode(out[i:])
        return doc


# ======================================================================================================================
# the pipeline
# ======================================================================================================================
@dataclass
class Plan:
    layers: List[int]
    with_globals: bool
    delete: bool
    min_free_gb: float
    run_golden: bool = True  # False (--skip-golden): never run it; those layers' shards then stay until it is done
    keep_layers: Set[int] = field(default_factory=lambda: set(parse_layers(DEFAULT_KEEP_BF16)))


class Pipeline:
    def __init__(self, plan: Plan, runner: Runner, index: Optional[ShardIndex] = None,
                 state_path: Optional[Path] = None):
        self.plan = plan
        self.plan.keep_layers = set(int(l) for l in plan.keep_layers)
        self.runner = runner
        self.index = index or ShardIndex.load(runner.weights_dir)
        self.state = StreamState(state_path or runner.weights_dir / STATE_NAME)
        self.status: dict = {}
        self.cache_dir: Optional[Path] = None

    # ---- disk -----------------------------------------------------------------------------------------------------
    def free_gb(self) -> float:
        return min(gb(free_bytes(self.runner.weights_dir)), gb(free_bytes(self.runner.cache_root)))

    def _room(self, need_bytes: int) -> bool:
        return self.free_gb() - gb(need_bytes) >= self.plan.min_free_gb

    def ensure_room(self, need_bytes: int, what: str) -> None:
        """Keep ``min_free_gb`` after writing ``need_bytes`` (already guarded): deletes eligible shards first when
        allowed."""
        if self._room(need_bytes):
            return
        if self.plan.delete:
            self.delete_consumed()
        if not self._room(need_bytes):
            if not self.plan.delete:
                hint = " (nothing is deleted without --delete-converted)"
            elif self.plan.keep_layers:
                hint = (f" (--keep-bf16 keeps the BF16 of layers {layer_ranges(self.plan.keep_layers)}; release them "
                        f"with --keep-bf16 none once their tests are signed off, or free space elsewhere)")
            else:
                hint = ""
            raise StreamError(
                f"disk guard before {what}: needs {gb(need_bytes):.1f} GB, {self.free_gb():.1f} GB free; would leave "
                f"{self.free_gb() - gb(need_bytes):.1f} GB < {self.plan.min_free_gb:g} GB{hint}",
                EXIT_DISK,
            )

    # ---- ground truth ---------------------------------------------------------------------------------------------
    def refresh_status(self) -> None:
        st = self.runner.cache_status(list(range(N_LAYERS)), True)
        check_serving_cache(st, self.runner.cache_root)
        self.status = st
        self.cache_dir = Path(st["cache_dir"])

    def golden_done(self) -> Set[int]:
        """Layers every golden (the C2 golden and each ``--extra-golden-out``) has processed."""
        return goldens_done_layers(self.runner.golden_outs)

    def done_layers(self) -> Set[int]:
        """The layers whose shards may go now (:func:`eligible_layers`: done, current code, every golden done, not
        kept)."""
        return eligible_layers(self.status, self.golden_done(), self.plan.keep_layers)

    @contextlib.contextmanager
    def converter_lock(self):
        """The converter's per-cache-directory flock (non-blocking): yields False when a converter holds it."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.cache_dir / CW.LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o664)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                got = True
            except BlockingIOError:
                got = False
            yield got
        finally:
            os.close(fd)

    # ---- deletion -------------------------------------------------------------------------------------------------
    def delete_consumed(self) -> int:
        """Under the converter lock: re-read the status, then delete (or, without --delete-converted, list) the shards
        of the eligible layers. Returns the bytes freed."""
        if self.cache_dir is None:
            self.refresh_status()
        with self.converter_lock() as got:
            if not got:
                log(f"  a converter holds {self.cache_dir / CW.LOCK_NAME}: no deletion now")
                return 0
            self.refresh_status()  # no conversion can change a part from here until the lock is released
            golden = self.golden_done()  # the intersection over every golden run
            done = eligible_layers(self.status, golden, self.plan.keep_layers)
            cand = deletable_shards(self.index, done, keep_layers=self.plan.keep_layers)
            if not cand:
                return 0
            size = gb(sum(self.index.sizes[s] for s in cand))
            if not self.plan.delete:
                log(f"  would delete {len(cand)} consumed shards ({size:.1f} GB): {cand[:6]}"
                    f"{' ...' if len(cand) > 6 else ''} (pass --delete-converted)")
                return 0
            if self.runner.dry_run:
                log(f"  [dry-run] would delete {cand}")
                return 0
            freed = delete_shards(self.index, cand, allow=True, status=self.status, golden_done=golden,
                                  cache_root=self.runner.cache_root, keep_layers=self.plan.keep_layers,
                                  state=self.state)
        log(f"  freed {gb(freed):.1f} GB; {self.free_gb():.1f} GB free")
        return freed

    # ---- conversion -----------------------------------------------------------------------------------------------
    def full_estimate(self, part: Optional[int]) -> int:
        kind = part_kind(part)
        var = (self.status.get("variants") or {}).get(kind)
        return CW.estimate_part_bytes(kind, var)

    def convert_step(self, part: Optional[int], *, force: bool = False) -> None:
        """One converter run; its disk-guard exit (3) triggers one deletion pass and one retry. Exit 3 -> 3, 4 -> 4
        (BF16 missing), anything else -> 7."""
        tag = part_tag(part)
        rc = self.runner.convert(part, force=force)
        if rc == CW.EXIT_DISK and self.plan.delete and self.delete_consumed():
            log(f"{tag}: retrying the conversion after the deletion")
            rc = self.runner.convert(part, force=force)
        if rc == CW.EXIT_DISK:
            raise StreamError(f"converter disk guard for {tag} (exit 3): below --min-free-gb {self.plan.min_free_gb:g}",
                              EXIT_DISK)
        if rc == CW.EXIT_SOURCE:
            raise StreamError(f"converter: BF16 tensors of {tag} missing (exit 4)", EXIT_DOWNLOAD)
        if rc != 0:
            raise StreamError(f"conversion of {tag} failed (converter exit {rc})", EXIT_CONVERT)

    # ---- per unit -------------------------------------------------------------------------------------------------
    def do_globals(self) -> None:
        p = part_entry(self.status, None)
        if part_done(p):
            log("global: converted and verified")
            return
        names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
        shards = {s for s in self.index.global_shards() if set(self.index.global_tensors_of(s)) & set(names)}
        missing = sorted(s for s in shards if not self.index.present(s))
        if missing:
            raise StreamError(f"global shards missing: {missing}; run scripts/download_weights.py --layers 0 first "
                              f"(it fetches the globals)", EXIT_DOWNLOAD)
        need = conversion_bytes(p, force=False, full=self.full_estimate(None))
        self.ensure_room(CW.guarded(need), "converting the globals")
        self.convert_step(None)
        self.refresh_status()
        if not part_done(part_entry(self.status, None)) and not self.runner.dry_run:
            raise StreamError("globals not complete + verified (+ hashed) after conversion", EXIT_CONVERT)
        self.state.mark("global", "converted")

    def do_layer(self, l: int) -> None:
        tag = f"L{l:02d}"
        p = part_entry(self.status, l)
        done = part_done(p)
        fp_ok = fingerprint_current(self.status, p)
        # the golden runs (C2 first, then the extra ones) that have not processed L yet
        pending = ([o for o in self.runner.golden_outs if l not in golden_done_layers(o)] if self.plan.run_golden
                   else [])
        golden_needed = bool(pending)
        shards = sorted(self.index.layer_shards.get(l, ()))
        candidate = (self.plan.delete and l not in self.plan.keep_layers
                     and bool(self.index.releasable(l, self.plan.keep_layers)))
        # converted by other code (or a variant addition onto files of other code): rebuild from BF16 before the
        # shards can go, while they are still there
        force = candidate and bool(p.get("base_ok")) and not fp_ok
        if done and not golden_needed and not force:
            log(f"{tag}: converted and verified" + ("" if fp_ok else " (by other code; its BF16 is not released)"))
            self.delete_consumed()
            return
        convert = (not done) or force
        conv_need = conversion_bytes(p, force=force, full=self.full_estimate(l)) if convert else 0
        reserve = int(GOLDEN_RESERVE_GB * 1e9)
        n_res = max(1, len(pending))  # one golden reserve per golden run of this layer
        # 1. shards
        missing = self.index.missing_of(l)
        if missing:
            need = sum(self.index.sizes[s] for s in missing)
            self.ensure_room(need + CW.guarded(conv_need) + n_res * reserve, f"downloading {tag}")
            margin = self.plan.min_free_gb + gb(CW.guarded(conv_need)) + n_res * GOLDEN_RESERVE_GB
            log(f"{tag}: downloading {len(missing)} shards ({gb(need):.1f} GB; margin {margin:.1f} GB)")
            if self.runner.download(l, margin) != 0:
                raise StreamError(f"download of {tag} failed", EXIT_DOWNLOAD)
            still = self.index.missing_of(l)
            if still and not self.runner.dry_run:
                raise StreamError(f"{tag}: shards still missing after the download (disk margin?): {still}",
                                  EXIT_DOWNLOAD)
            self.state.mark(l, "downloaded", shards=shards)
        # 2. sha256 (a record counts only while the file is the one that was hashed)
        ver = verified_shards(self.runner.weights_dir)
        todo = [s for s in shards if ver.get(s) != "ok"]
        if todo:
            log(f"{tag}: sha256 of {len(todo)} shards ({[(s, ver.get(s)) for s in todo][:4]})")
            self.runner.verify_sha()
            ver = verified_shards(self.runner.weights_dir)
        bad = [s for s in shards if ver.get(s) != "ok"]
        if bad and not self.runner.dry_run:
            raise StreamError(f"{tag}: shards not verified ok: {[(s, ver.get(s)) for s in bad]} (delete a BAD shard and "
                              f"re-run to re-download it)", EXIT_SHA)
        # 3. golden(s): the C2 golden, then every extra golden, each resumed for exactly this layer
        for o in pending:
            primary = o == Path(self.runner.golden_out)
            name = "C2 golden" if primary else f"golden {o.name}"
            last = golden_last_layer(o)
            if last is None or last < l - 1:
                raise StreamError(f"{tag}: the {name}'s last layer is {last} ({o}); layer {l} needs {l - 1} first",
                                  EXIT_GOLDEN)
            self.ensure_room(reserve, f"{name} {tag}")
            log(f"{tag}: {name} resume ({o})")
            rc = self.runner.golden(l) if primary else self.runner.golden(l, out=o)
            if rc != 0:
                raise StreamError(f"{name} for {tag} failed (exit {rc})", EXIT_GOLDEN)
            if l not in golden_done_layers(o) and not self.runner.dry_run:
                raise StreamError(f"{name} manifest ({o}) does not list layer {l} after the run", EXIT_GOLDEN)
            self.state.mark(l, "golden" if primary else f"golden:{o.name}")
        # 4 + 5. convert (+ the converter's cache-only verification)
        if convert:
            if force:
                log(f"{tag}: built by other code (fingerprint {str(p.get('code_fingerprint'))[:12]} != current "
                    f"{str(self.status.get('code_fingerprint'))[:12]}): rebuilding it before its BF16 can go")
            self.ensure_room(CW.guarded(conv_need), f"converting {tag}")
            log(f"{tag}: convert ({self.runner.target}{', --force' if force else ''})")
            self.convert_step(l, force=force)
            self.refresh_status()
            q = part_entry(self.status, l)
            if not self.runner.dry_run and not part_done(q):
                raise StreamError(f"{tag} not complete + verified (+ hashed) after conversion: {q.get('reason')}",
                                  EXIT_CONVERT)
            if not self.runner.dry_run and candidate and not fingerprint_current(self.status, q):
                raise StreamError(f"{tag}: converted, but its fingerprint is not the current code's", EXIT_CONVERT)
            self.state.mark(l, "rebuilt" if force else "converted")
        # 6. delete what is consumed now
        self.delete_consumed()

    def run(self) -> int:
        lock = self.runner.weights_dir / LOCK_NAME
        fd = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o664)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise StreamError(f"another stream_weights.py holds {lock}", EXIT_USAGE)
            return self._run()
        finally:
            os.close(fd)

    def _run(self) -> int:
        self.state.data["runs"].append({"started": now_iso(), "argv": sys.argv[1:]})
        self.state.save()
        self.refresh_status()  # also: the converter's cache must be the serving cache
        log(f"layers {layer_ranges(self.plan.layers)} ({len(self.plan.layers)}), globals {self.plan.with_globals}, "
            f"delete {self.plan.delete}, BF16 kept for layers {layer_ranges(self.plan.keep_layers)}, min free "
            f"{self.plan.min_free_gb:g} GB, converter target {self.runner.target}; cache {self.cache_dir}; code "
            f"fingerprint {str(self.status.get('code_fingerprint'))[:16]}; {self.free_gb():.1f} GB free")
        log("goldens (all must list a layer before its BF16 goes): " + "; ".join(
            f"{o} (last layer {golden_last_layer(o)})" for o in self.runner.golden_outs)
            + ("" if self.plan.run_golden else " [--skip-golden: not run]"))
        if not self.plan.delete:
            log("--delete-converted not given: no BF16 shard will be deleted")
        if self.plan.with_globals:
            self.do_globals()
        for l in self.plan.layers:
            self.do_layer(l)
        log(f"done; {self.free_gb():.1f} GB free")
        return EXIT_OK


# ======================================================================================================================
# dry-run disk timeline (pure; estimates; the pipeline's guards)
# ======================================================================================================================
def simulate(index: ShardIndex, layers: Sequence[int], *, with_globals: bool, parts: Dict[Optional[int], dict],
             fingerprint: Optional[str], golden_done: Iterable[int], delete: bool, free_gb0: float, min_free_gb: float,
             run_golden: bool = True, present: Optional[Set[str]] = None, keep_layers: Iterable[int] = (),
             variants: Optional[Dict[str, Dict[str, List[str]]]] = None, n_goldens: int = 1) -> List[dict]:
    """Projected free space after every step of :meth:`Pipeline._run` for the status ``parts`` (``{layer: status part
    record}``; a missing record = not converted) and ``fingerprint`` (the current code's). Applies the pipeline's own
    guards (estimate x 1.1 before a conversion, + the missing shards and the golden reserve before a download, the
    reserve before a golden; deletions first when allowed) and stops at the first guard that trips, as the pipeline
    does (a row with ``ok`` False). ``present``: the shards on disk (default: the real presence). ``golden_done``: the
    layers every golden has processed; ``n_goldens``: golden runs per layer (C2 + the extra ones; reserve and growth
    scale with it)."""
    n_goldens = max(1, int(n_goldens))
    var = variants or CW.ConvertOptions().all_variants()
    status = {"code_fingerprint": fingerprint, "parts": list(parts.values())}
    present = set(index.present_shards() if present is None else present)
    keep_l = set(int(l) for l in keep_layers)
    keep_s = index.keep_shards()
    done = {l for l, p in parts.items() if part_done(p)}
    fresh = {l for l in done if fingerprint_current(status, parts[l])}
    gdone = set(int(l) for l in golden_done)
    reserve = int(GOLDEN_RESERVE_GB * 1e9)
    free = float(free_gb0)
    rows: List[dict] = []

    def full(part):
        kind = part_kind(part)
        return CW.estimate_part_bytes(kind, var.get(kind))

    def add(unit, step, delta, need=None, ok=True):
        rows.append({"unit": unit, "step": step, "delta_gb": round(delta, 2), "free_gb": round(free, 2),
                     "need_gb": None if need is None else round(gb(need), 2), "ok": ok})

    def do_delete(unit):
        nonlocal free
        elig = {l for l in fresh if l is not None and l in gdone and l not in keep_l}
        cand = deletable_shards(index, elig, present=present, keep=keep_s, keep_layers=keep_l)
        if cand:
            present.difference_update(cand)
            freed = gb(sum(index.sizes[s] for s in cand))
            free += freed
            add(unit, f"delete {len(cand)} shards", freed)

    def guard(unit, what, need) -> bool:
        if free - gb(need) >= min_free_gb:
            return True
        if delete:
            do_delete(unit)
        if free - gb(need) >= min_free_gb:
            return True
        add(unit, f"STOP before {what}", 0.0, need, ok=False)
        return False

    add("-", "start", 0.0)
    if with_globals and None not in done:
        nb = conversion_bytes(parts.get(None) or {}, force=False, full=full(None))
        if not guard("global", "convert", CW.guarded(nb)):
            return rows
        free -= gb(nb)
        done.add(None)
        add("global", "convert", -gb(nb))
    for l in layers:
        p = parts.get(l) or {}
        is_done = l in done
        golden_needed = run_golden and l not in gdone
        candidate = delete and l not in keep_l and bool(index.releasable(l, keep_l, present=present))
        force = candidate and (bool(p.get("base_ok")) or is_done) and l not in fresh
        if is_done and not golden_needed and not force:
            if delete:
                do_delete(l)
            continue
        convert = (not is_done) or force
        conv = conversion_bytes(p, force=force, full=full(l)) if convert else 0
        miss = sorted(s for s in index.layer_shards.get(l, ()) if s not in present)
        if miss:
            nb = sum(index.sizes[s] for s in miss)
            n_res = n_goldens if golden_needed else 1
            if not guard(l, f"download of {len(miss)} shards", nb + CW.guarded(conv) + n_res * reserve):
                return rows
            present.update(miss)
            free -= gb(nb)
            add(l, f"download {len(miss)} shards", -gb(nb))
        if golden_needed:
            if not guard(l, "golden", reserve):
                return rows
            gdone.add(l)
            free -= GOLDEN_STEP_GB * n_goldens
            add(l, "golden" if n_goldens == 1 else f"golden x{n_goldens}", -GOLDEN_STEP_GB * n_goldens)
        if convert:
            if not guard(l, "rebuild" if force else "convert", CW.guarded(conv)):
                return rows
            delta = max(0, conv - int(p.get("bytes") or 0)) if force else conv  # a rebuild replaces the part
            free -= gb(delta)
            done.add(l)
            fresh.add(l)
            add(l, "rebuild" if force else "convert", -gb(delta))
        if delete:
            do_delete(l)
    return rows


# ======================================================================================================================
# CLI
# ======================================================================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="stream_weights.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--layers", default="all", help="decoder layers in order (default all = 0-52)")
    ap.add_argument("--no-globals", action="store_true", help="skip the globals part")
    ap.add_argument("--delete-converted", action="store_true",
                    help="DELETE every BF16 shard whose layers are converted + verified + current code + golden done "
                         "and not kept (--keep-bf16); off by default")
    ap.add_argument("--keep-bf16", default=DEFAULT_KEEP_BF16,
                    help=f"layers whose BF16 shards are never deleted (default {DEFAULT_KEEP_BF16}: the bring-up and "
                         f"acceptance-test layers; 'none' releases them, a sign-off decision)")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB, help=f"free-space floor (default {MIN_FREE_GB:g})")
    ap.add_argument("--target", choices=("mock", "device"), default="mock", help="converter target")
    ap.add_argument("--skip-golden", action="store_true", help="do not run the C2 golden (its shards are then kept)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the projected free space; run nothing")
    ap.add_argument("--cache-root", default=None,
                    help="TT cache root the converter writes and the deletion gate checks (default: the serving "
                         "resolution MOTIF3_TT_CACHE_PATH > TT_CACHE_PATH > motif-3/tt_cache)")
    ap.add_argument("--golden-out", default=str(GOLDEN_OUT), help=f"the C2 golden run (default {GOLDEN_OUT})")
    ap.add_argument("--extra-golden-out", action="append", default=[],
                    help="another golden_stream run (repeatable; e.g. the fp32 golden goldens/c2_fp32) resumed for each "
                         "layer right after the C2 golden; a layer's BF16 goes only once every golden lists it")
    ap.add_argument("--state", default=None, help=f"audit log (default <weights dir>/{STATE_NAME})")
    ap.add_argument("--convert-arg", action="append", default=[],
                    help=f"variant option for the converter (repeatable; only {', '.join(PASS_THROUGH)}), e.g. "
                         f"--convert-arg=--lm-head-split=both")
    a = ap.parse_args(argv)

    if os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        log("unset TT_METAL_MOCK_CLUSTER_DESC_PATH: the converter selects the mock cluster itself")
        return EXIT_USAGE
    try:
        layers = parse_layers(a.layers)
        keep = set(parse_layers(a.keep_bf16))  # "none" -> nothing kept; "" / "all" -> everything kept
        runner = Runner(target=a.target, dry_run=a.dry_run, weights_dir=WEIGHTS, golden_out=Path(a.golden_out),
                        cache_root=Path(a.cache_root) if a.cache_root else default_cache_root(),
                        variant_args=list(a.convert_arg), min_free_gb=a.min_free_gb,
                        extra_golden_outs=[Path(o) for o in a.extra_golden_out])
    except ValueError as e:
        log(str(e))
        return EXIT_USAGE
    plan = Plan(layers=layers, with_globals=not a.no_globals, delete=a.delete_converted, min_free_gb=a.min_free_gb,
                run_golden=not a.skip_golden, keep_layers=keep)
    index = ShardIndex.load(runner.weights_dir)
    try:
        if a.dry_run:
            st = runner.cache_status(list(range(N_LAYERS)), True)
            cache_dir = check_serving_cache(st, runner.cache_root)
            parts = {p["layer"]: p for p in st.get("parts", [])}
            done = sorted((l for l, p in parts.items() if part_done(p)), key=lambda x: -1 if x is None else x)
            stale = [l for l in done if l is not None and not fingerprint_current(st, parts[l])]
            last = {str(o): golden_last_layer(o) for o in runner.golden_outs}
            rows = simulate(index, layers, with_globals=not a.no_globals, parts=parts,
                            fingerprint=st.get("code_fingerprint"), golden_done=goldens_done_layers(runner.golden_outs),
                            delete=a.delete_converted, keep_layers=keep,
                            free_gb0=min(gb(free_bytes(runner.weights_dir)), gb(free_bytes(runner.cache_root))),
                            min_free_gb=a.min_free_gb, run_golden=not a.skip_golden, variants=st.get("variants"),
                            n_goldens=len(runner.golden_outs))
            ks = index.keep_shards()
            log(f"cache {cache_dir}: {len(done)} parts complete + verified + hashed "
                f"({[part_tag(l) for l in done]}; built by other code: {[part_tag(l) for l in stale]}); golden last "
                f"layer {last}; BF16 kept for layers {layer_ranges(keep)}; kept shards {sorted(ks)} "
                f"({gb(sum(index.sizes[s] for s in ks)):.1f} GB); floor {a.min_free_gb:g} GB")
            for r in rows:
                print(f"  {str(r['unit']):>6} {r['step']:30s} {r['delta_gb']:+8.1f} GB -> {r['free_gb']:7.1f} GB free"
                      + ("" if r["ok"] else f"   << needs {r['need_gb']:.1f} GB above the {a.min_free_gb:g} GB floor: "
                                            f"the pipeline stops here (exit 3)"))
            low = min(r["free_gb"] for r in rows)
            stop = next((r for r in rows if not r["ok"]), None)
            if stop:
                log(f"projected: stops at {part_tag(stop['unit']) if isinstance(stop['unit'], int) else stop['unit']} "
                    f"({stop['step']}); minimum free {low:.1f} GB")
            else:
                log(f"projected: completes; minimum free {low:.1f} GB, final {rows[-1]['free_gb']:.1f} GB")
            return EXIT_OK
        return Pipeline(plan, runner, index, state_path=Path(a.state) if a.state else None).run()
    except StreamError as e:
        log(f"STOP: {e}")
        return e.code
    except subprocess.TimeoutExpired as e:
        log(f"STOP: step timed out: {e}")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
