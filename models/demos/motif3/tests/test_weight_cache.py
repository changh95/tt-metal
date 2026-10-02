# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT weight-cache conversion tests (WAVE_A_REVIEW CONV-1..4; ``scripts/convert_weights.py``,
``scripts/stream_weights.py``; runbook ``docs/WEIGHTS_RUNBOOK.md``).

Host-only (devices hidden; the root conftest stays active because the file also has device tests; ~1 min, the mock
conversion runs the converter 7 times)::

    scripts/hostrun.sh -n weight_cache_host -- python -m pytest -p no:cacheprovider -q \
        models/demos/motif3/tests/test_weight_cache.py -k host

* ``test_host_parsing_and_options``: layer specs, per-kind variants (the serving defaults always included), size
  estimates, the shared disk guard.
* ``test_host_part_status``: ``part_status`` on a synthetic cache directory (marker / stats / size / tag rules, variant
  coverage per part kind, reuse estimates, hash counts).
* ``test_host_raising_source``: every read of the verification source raises.
* ``test_host_deletion_rules``: ``deletable_shards`` on the real checkpoint index (static: index + tree.json) -- global
  shards, the smallest shard, shards of unfinished or kept layers are never deletable.
* ``test_host_delete_requires_flag``: ``delete_shards`` refuses without ``--delete-converted``, for a non-serving cache,
  and for any shard its own re-derivation (status + golden + keep list) does not allow; config / tokenizer files stay.
* ``test_host_pipeline_deletion_gate``: ``Pipeline.delete_consumed`` with a fake runner: only converted + verified +
  hashed + current-code + golden-done + not-kept layers release shards; nothing while another process holds the
  converter lock, without ``--delete-converted``, or for a non-serving cache.
* ``test_host_pipeline_conversion_steps``: converter exit codes (3 -> one deletion pass + one retry -> 3, 4 -> 4, else
  7) and the rebuild of a layer converted by other code before its shards can go.
* ``test_host_pipeline_extra_golden``: ``--extra-golden-out`` (the fp32 golden): shards go only for layers EVERY golden
  lists; each pending golden is resumed for the layer (C2 first); an extra golden more than one layer behind stops the
  layer (exit 6) before anything runs or is deleted.
* ``test_host_runner_commands``: every converter command is pinned to ``--mesh 4x8 --cache-root --weights-dir``, the
  golden gets ``--ckpt-dir``, only variant options pass ``--convert-arg``, the downloader / verifier refuse another dir.
* ``test_host_verified_shards_stat``: a sha256 record counts only while the file has the size and mtime that were hashed.
* ``test_host_dry_run_timeline``: the projected timeline on a synthetic state (no live disk / golden): it applies the
  converter's guard (same function), stops where the pipeline would, and only deleting makes the stream fit.
* ``test_host_mock_conversion``: ``convert_weights.py --target mock`` (subprocess, mock 32-chip cluster) on layer 0 in a
  scratch root under ``tt_cache/test`` (removed afterwards): a motif-only build, the incremental stock variant (the 22
  files reused: same inodes), ``--force`` (removes a planted stale file, same bytes), exit 3 / 4 leave the part untouched,
  the status JSON (per-kind coverage, serving cache, fingerprint, ``MESH_DEVICE`` ignored), and the files are
  byte-identical to the real cache's layer 0 when that is complete for the default options.

Device (through the lock; ~4 min; ~12 GB of scratch, removed afterwards; needs 40 GB to stay free meanwhile)::

    scripts/devrun.sh -t 2400 -n weight_cache -- python -m pytest -p no:cacheprovider -s \
        models/demos/motif3/tests/test_weight_cache.py -k device

* ``test_device_convert_roundtrip``: needs the real (mock-built) cache's globals + L00-L02 complete for the default
  options (else it skips: nothing to compare against). Converts them with the converter on the real (4, 8) mesh into a
  TEST cache root (``tt_cache/test/weight_cache``), checks EVERY file is byte-identical (sha256) to the mock-built real
  cache, then builds embedding, decoder layers 0-2, LM head, the exact-fp32 router of L02 and the six stock mHC sites
  from the TEST cache alone (a source that raises on any read) and from the BF16 source (``cache=False``) and compares,
  bitwise: every device weight tensor (all chips; 4 sampled chips for the multi-GB ones), the prefill chain
  (embedding -> 3 layers with KV fill -> LM-head tile, S = 128, real prompt tokens), one 32-lane decode step after it
  (heterogeneous positions, own page-table rows), the KV caches, the exact router logits and the stock mHC pre / post;
  every output must also be finite. The TEST root is removed afterwards (``MOTIF3_KEEP_TEST_CACHE=1`` keeps it).
* ``test_device_model_auto_loads_mock_cache``: ``tt.model.MotifModel(cache="auto")`` with a raising source on the
  real (mock-built) cache: loads layers 0-2 + globals without touching the checkpoint and prefills a prompt; then the
  same with the variant choices (stock Sinkhorn, exact-fp32 router: decisions D2 / D1) from the cache alone.

Every device test logs the committed fabric topology first.
"""

from __future__ import annotations

import contextlib
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch

import ttnn
from models.demos.motif3.tt.model_config import DEFAULT_TT_CACHE_ROOT, device_params

PROJECT = Path("/home/ttuser/hchang/experiments/motif-3")
SCRIPTS = PROJECT / "scripts"
TEST_ROOT = DEFAULT_TT_CACHE_ROOT / "test" / "weight_cache"
HOST_SCRATCH = DEFAULT_TT_CACHE_ROOT / "test" / "weight_cache_host"
REAL_ROOT = DEFAULT_TT_CACHE_ROOT
PROMPTS = PROJECT / "goldens" / "c2" / "prompts.json"
MESH = [pytest.param((4, 8), device_params(), id="4x8")]
PARTS = [None, 0, 1, 2]  # globals + layers 0-2
# Temporary test caches (removed in a finally) may take the disk down to this; the persistent floor of the scripts is
# convert_weights.MIN_FREE_GB (60).
TEST_SCRATCH_MIN_FREE_GB = 40.0


def log(msg: str) -> None:
    print(f"[weight_cache] {msg}", flush=True)


def _load_script(name: str):
    """Import ``scripts/<name>.py`` as a module (registered in sys.modules: dataclasses need it; the pipeline loads the
    converter under the same name, so both share one instance)."""
    mod_name = f"motif3_scripts_{name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / 1e9


def _devices_visible() -> bool:
    return bool(os.listdir("/dev/tenstorrent")) if os.path.isdir("/dev/tenstorrent") else False


# ======================================================================================================================
# host-only: converter rules
# ======================================================================================================================
def test_host_parsing_and_options():
    cw = _load_script("convert_weights")
    assert cw.parse_layers("0-2", 53) == [0, 1, 2]
    assert cw.parse_layers("all", 53) == list(range(53))
    assert cw.parse_layers("36-", 53) == list(range(36, 53))
    assert cw.parse_layers("3,4,10-12", 53) == [3, 4, 10, 11, 12]
    assert cw.parse_layers("none", 53) == [] and cw.parse_layers("", 53) == []
    with pytest.raises(ValueError):
        cw.parse_layers("52-60", 53)
    assert cw.parse_mesh("4x8") == (4, 8) and cw.parse_mesh("(8, 4)") == (8, 4)
    assert cw.part_tag(None) == "global" and cw.part_tag(7) == "L07"
    o = cw.ConvertOptions()
    # per kind; the defaults add the exact router (D1) and the stock Sinkhorn constants (D2)
    assert o.variants_for("global") == {"lm_head_split": ["mesh"], "embedding": ["replicated"]}
    assert o.variants_for("dense") == {"mhc_sinkhorn": ["motif", "stock"]}
    assert o.variants_for("moe") == {"router": ["composite", "exact_fp32"], "mhc_sinkhorn": ["motif", "stock"]}
    # a non-"both" option ADDS to the serving defaults, it never replaces them
    assert cw.ConvertOptions(mhc_sinkhorn="stock").variants_for("dense") == {"mhc_sinkhorn": ["motif", "stock"]}
    assert cw.ConvertOptions(router="exact_fp32").variants_for("moe")["router"] == ["composite", "exact_fp32"]
    assert cw.ConvertOptions(lm_head_split="tp").variants_for("global")["lm_head_split"] == ["mesh", "tp"]
    assert cw.ConvertOptions(router="composite", mhc_sinkhorn="motif").variants_for("moe") == {
        "router": ["composite"], "mhc_sinkhorn": ["motif"]}

    class _Cfg:  # the env-selectable config defaults are built too (MOTIF3_ROUTER_LOGITS=exact_fp32)
        router_logits, mhc_sinkhorn = "exact_fp32", "motif"

    assert cw.ConvertOptions(router="composite").variants_for("moe", _Cfg)["router"] == ["composite", "exact_fp32"]
    # coverage is per kind: a global option never makes a layer part incomplete, and vice versa
    rec_l = {"router": ["composite", "exact_fp32"], "mhc_sinkhorn": ["motif", "stock"]}
    assert o.covered_by(rec_l, "moe") and cw.ConvertOptions(lm_head_split="both", embedding="both").covered_by(rec_l, "moe")
    assert cw.ConvertOptions(lm_head_split="both").covered_by({"mhc_sinkhorn": ["motif", "stock"]}, "dense")
    assert not cw.ConvertOptions(lm_head_split="both").covered_by({"lm_head_split": ["mesh"], "embedding": ["replicated"]},
                                                                 "global")
    assert cw.ConvertOptions(router="composite").covered_by({"lm_head_split": ["mesh"], "embedding": ["replicated"]},
                                                            "global")
    # format /1 records (all four options on every part) still read; the old motif-only default now misses "stock"
    old = {"router": ["composite", "exact_fp32"], "mhc_sinkhorn": ["motif"], "lm_head_split": ["mesh"],
           "embedding": ["replicated"]}
    assert o.missing(old, "moe") == {"mhc_sinkhorn": ["stock"]} and o.missing(old, "global") == {}
    assert not o.covered_by(None, "moe") and not o.covered_by({}, "dense")
    with pytest.raises(ValueError):
        cw.ConvertOptions(router="fast")
    # sizes: default parts (measured) and the variant files
    assert cw.EST_PART_BYTES == {"global": 3_607_113_408, "dense": 355_205_440, "moe": 6_655_598_208}
    assert cw.estimate_part_bytes("moe", {"router": ["composite"], "mhc_sinkhorn": ["motif"]}) == 6_652_416_256
    g2 = cw.estimate_part_bytes("global", cw.ConvertOptions(lm_head_split="both", embedding="both").variants_for("global"))
    assert abs(g2 / 1e9 - 7.214) < 0.01, g2  # both global variants: + 2 x 1.8 GB
    assert cw.variant_bytes("moe", {"mhc_sinkhorn": ["stock"]}) == cw.EST_VARIANT_BYTES[("mhc_sinkhorn", "stock")]
    # the shared disk guard
    assert cw.guarded(1_000_000_000) == 1_100_000_000
    assert cw.room_ok(70_000_000_000, 9_000_000_000, 60) and not cw.room_ok(70_000_000_000, 10_000_000_001, 60)
    assert cw.MIN_FREE_GB == 60.0
    sw = _load_script("stream_weights")
    assert sw.CW is cw, "the pipeline must use the converter's estimates and guard"
    assert sw.MIN_FREE_GB == cw.MIN_FREE_GB
    assert sw.part_kind(None) == "global" and sw.part_kind(1) == "dense" and sw.part_kind(2) == "moe"
    assert sw.parse_layers("36-52") == list(range(36, 53)) and sw.parse_layers("none") == []
    assert sw.layer_ranges({0, 1, 2, 5, 7, 8}) == "0-2,5,7-8" and sw.layer_ranges([]) == "none"
    assert sw.Plan(layers=[], with_globals=False, delete=False, min_free_gb=60).keep_layers == set(range(36))


def test_host_part_status(tmp_path):
    from models.demos.motif3.tt import weights as W
    from models.demos.motif3.tt.model_config import MotifTTConfig

    cw = _load_script("convert_weights")
    cfg = MotifTTConfig.from_hf_config(mesh_shape=(4, 8), tt_cache_root=tmp_path, num_layers=53)
    opts = cw.ConvertOptions()
    st = cw.part_status(cfg, 3, opts)
    assert not st.complete and not st.base_ok and st.need_bytes == cw.EST_PART_BYTES["moe"]
    d = W.layer_cache_marker(cfg, 3).parent
    d.mkdir(parents=True)
    names = ["a__rep_dtype_BFLOAT16_layout_TILE.tensorbin", "b__tp1_dtype_BFLOAT8_B_layout_TILE.tensorbin"]
    for i, n in enumerate(names):
        (d / n).write_bytes(b"x" * (100 + i))
    W.mark_layer_cached(cfg, 3, names)
    st = cw.part_status(cfg, 3, opts)
    assert not st.complete and "another tool" in st.reason  # marker without .convert.json -> rebuilt by the converter
    files = {n: {"bytes": 100 + i, "sha256": "0" * 64} for i, n in enumerate(names)}
    stats = {"format": cw.FORMAT, "files": files, "variants": opts.variants_for("moe"), "verify": {"ok": True,
             "files_hashed": 2}, "code": {"sha256": "c" * 64}}
    cw.write_json_atomic(d / cw.STATS_NAME, stats)
    st = cw.part_status(cfg, 3, opts)
    assert st.complete and st.base_ok and st.verified is True and st.files == 2 and st.bytes == 201, st
    assert st.hashes == 2 and st.files_hashed == 2 and st.fingerprint == "c" * 64 and st.need_bytes == 0
    rec = st.as_dict()
    assert rec["kind"] == "moe" and rec["variants"] == opts.variants_for("moe") and rec["missing_variants"] == {}
    # per kind: a globals-only option does not touch a layer part
    assert cw.part_status(cfg, 3, cw.ConvertOptions(lm_head_split="both", embedding="both")).complete
    # a missing variant: base_ok, reuse estimate = just the variant files
    stats["variants"] = {"router": ["composite", "exact_fp32"], "mhc_sinkhorn": ["motif"]}
    cw.write_json_atomic(d / cw.STATS_NAME, stats)
    st = cw.part_status(cfg, 3, opts)
    assert not st.complete and st.base_ok and st.missing == {"mhc_sinkhorn": ["stock"]}, st
    assert st.need_bytes == cw.EST_VARIANT_BYTES[("mhc_sinkhorn", "stock")] and st.verified is True
    assert st.rebuild_bytes == cw.EST_PART_BYTES["moe"]
    stats["variants"] = opts.variants_for("moe")
    del files[names[1]]["sha256"]
    cw.write_json_atomic(d / cw.STATS_NAME, stats)
    assert cw.part_status(cfg, 3, opts).hashes == 1  # (the pipeline requires every listed file hashed)
    (d / names[1]).write_bytes(b"y" * 7)  # size changed
    st = cw.part_status(cfg, 3, opts)
    assert "size" in st.reason and not st.base_ok and st.need_bytes == cw.EST_PART_BYTES["moe"]
    (d / names[1]).unlink()
    assert "missing" in cw.part_status(cfg, 3, opts).reason
    m = json.loads(W.layer_cache_marker(cfg, 3).read_text())
    m["version"] = "motif3-old"
    W.layer_cache_marker(cfg, 3).write_text(json.dumps(m))
    assert "tag" in cw.part_status(cfg, 3, opts).reason
    # the globals: lm_head_split / embedding apply
    gd = W.layer_cache_marker(cfg, None).parent
    gd.mkdir(parents=True)
    (gd / "g.tensorbin").write_bytes(b"g")
    W.mark_layer_cached(cfg, None, ["g.tensorbin"])
    cw.write_json_atomic(gd / cw.STATS_NAME, {"files": {"g.tensorbin": {"bytes": 1, "sha256": "1" * 64}},
                                             "variants": old_format_variants(), "verify": {"ok": True}})
    assert cw.part_status(cfg, None, opts).complete
    st = cw.part_status(cfg, None, cw.ConvertOptions(lm_head_split="both"))
    assert not st.complete and st.missing == {"lm_head_split": ["tp"]}
    assert st.need_bytes == cw.EST_VARIANT_BYTES[("lm_head_split", "tp")]


def old_format_variants() -> dict:
    """``variants`` as format /1 recorded it (all four options on every part)."""
    return {"router": ["composite", "exact_fp32"], "mhc_sinkhorn": ["motif"], "lm_head_split": ["mesh"],
            "embedding": ["replicated"]}


def test_host_raising_source():
    cw = _load_script("convert_weights")
    src = cw.RaisingSource("t")
    for call in (lambda: src.get("x"), lambda: src.get_rows("x", 0, 1), lambda: src.has("x"), lambda: "x" in src,
                 lambda: src.available("x"), lambda: src.shape("x"), lambda: src.keys(), lambda: src.layer_available(1)):
        with pytest.raises(AssertionError):
            call()
    inner = {"w": torch.ones(2, 3, dtype=torch.bfloat16)}

    class _Dict:
        def get(self, name, dtype=None):
            return inner[name]

    c = cw.CountingSource(_Dict())
    c.get("w")
    assert c.n_tensors == 1 and c.n_bytes == 12


# ======================================================================================================================
# host-only: pipeline rules
# ======================================================================================================================
def test_host_deletion_rules():
    sw = _load_script("stream_weights")
    idx = sw.ShardIndex.load()  # static: the index + hf_meta/tree.json (presence is passed explicitly below)
    keep = idx.keep_shards()
    glob = idx.global_shards()
    assert len(idx.shards) == 155
    # the shards that hold embedding / final norm / LM head / MTP: shards 1 and 104 on this checkpoint
    assert glob == {"model-00001-of-00155.safetensors", "model-00104-of-00155.safetensors"}, glob
    assert idx.smallest_shard() in keep and glob <= keep
    present = set(idx.shards)  # pretend everything is on disk
    assert sw.deletable_shards(idx, [], present=present) == []
    every = sw.deletable_shards(idx, range(53), present=present)
    assert set(every) == set(idx.shards) - keep, "with every layer done, all but the kept shards may go"
    # only layer 2 done: its private shards go; shards it shares with layer 1 / 3 stay
    two = set(sw.deletable_shards(idx, [2], present=present))
    assert two and all(idx.layers_of(s) == [2] for s in two), two
    shared = [s for s in idx.layer_shards[2] if set(idx.layers_of(s)) != {2}]
    assert shared and not set(shared) & two
    # layers 0 + 1 done: shard 1 (embedding) stays, shard 2 ([1, 2]) stays until layer 2 is done
    zero_one = set(sw.deletable_shards(idx, [0, 1], present=present))
    assert "model-00001-of-00155.safetensors" not in zero_one and "model-00002-of-00155.safetensors" not in zero_one
    # never a shard that is not present
    assert sw.deletable_shards(idx, range(53), present=set()) == []
    # layer 52 shares shard 104 with the LM head / MTP: never deletable
    assert "model-00104-of-00155.safetensors" not in every
    # kept layers (the default --keep-bf16 0-35): none of their shards, including one shared with layer 36
    kept = sw.deletable_shards(idx, range(53), present=present, keep_layers=range(36))
    assert kept and all(min(idx.layers_of(s)) >= 36 for s in kept), kept
    assert "model-00070-of-00155.safetensors" not in kept  # layers [35, 36]
    # releasable: a layer whose every present shard is kept for good is never a deletion candidate
    assert idx.releasable(0, present=present) == []  # shard 1 holds the embedding
    assert idx.releasable(3, present=present, keep_layers=range(36)) == []
    assert idx.releasable(36, present=present, keep_layers=range(36)) == ["model-00071-of-00155.safetensors",
                                                                         "model-00072-of-00155.safetensors",
                                                                         "model-00139-of-00155.safetensors"]


SERVING_TAG = "motif3-2ed2ed5c-c1-e8s8d8a16r16m16l16v16"
FP = "c" * 64


def _part(layer, *, complete=True, verified=True, files=3, hashes=None, files_hashed=None, fp=FP, base_ok=None,
          bytes_=300, need=None, rebuild=None, kind=None):
    return {"part": "global" if layer is None else f"L{layer:02d}", "layer": layer,
            "kind": kind or ("global" if layer is None else ("dense" if layer < 2 else "moe")), "complete": complete,
            "base_ok": complete if base_ok is None else base_ok, "files": files, "bytes": bytes_,
            "verified": verified, "hashes": files if hashes is None else hashes,
            "files_hashed": files if files_hashed is None else files_hashed, "code_fingerprint": fp,
            "need_bytes": 0 if (complete and need is None) else (need or 0),
            "rebuild_bytes": rebuild or 0, "reason": "complete" if complete else "no .complete marker"}


def _status(root: Path, parts: List[dict], *, fp=FP, tag=SERVING_TAG, mesh=(4, 8), serving=None) -> dict:
    d = Path(root) / tag / f"mesh{mesh[0]}x{mesh[1]}"
    ok = tuple(mesh) == (4, 8) and tag == SERVING_TAG
    return {"cache_root": str(root), "cache_tag": tag, "cache_dir": str(d), "mesh_shape": list(mesh),
            "serving_mesh_shape": [4, 8], "serving_tag": SERVING_TAG,
            "serving_cache_dir": str(Path(root) / SERVING_TAG / "mesh4x8"),
            "is_serving_cache": ok if serving is None else serving, "variants": {}, "code_fingerprint": fp,
            "parts": parts}


def _toy_index(sw, tmp_path: Path):
    """8 shards: 1 = embedding + L0, 2..7 = L1..L6 one each, 8 = LM head (also the smallest)."""
    wm = {"model.embed_tokens.weight": "model-00001-of-00008.safetensors",
          "model.layers.0.a": "model-00001-of-00008.safetensors",
          "lm_head.weight": "model-00008-of-00008.safetensors"}
    for l in range(1, 7):
        wm[f"model.layers.{l}.a"] = f"model-{l + 1:05d}-of-00008.safetensors"
    sizes = {f"model-{i:05d}-of-00008.safetensors": 40 + i for i in range(1, 8)}
    sizes["model-00008-of-00008.safetensors"] = 10
    wdir = tmp_path / "weights"
    wdir.mkdir()
    for s, n in sizes.items():
        (wdir / s).write_bytes(b"\0" * n)
    for f in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        (wdir / f).write_text("{}")
    return sw.ShardIndex(wm, sizes, wdir), wdir


def _verify_all(wdir: Path, idx) -> None:
    """A ``.verified.json`` as scripts/verify_shards.py writes it: every present shard ok, with its current stat."""
    doc = {}
    for s in idx.shards:
        if (wdir / s).is_file():
            st = os.stat(wdir / s)
            doc[s] = {"status": "ok", "stat": [st.st_size, st.st_mtime]}
    (wdir / ".verified.json").write_text(json.dumps(doc))


def _golden(tmp_path: Path, layers) -> Path:
    g = tmp_path / "golden"
    g.mkdir(exist_ok=True)
    (g / "manifest.json").write_text(json.dumps({"layers_done": sorted(layers), "last_layer": max(layers)}))
    return g


def test_host_delete_requires_flag(tmp_path):
    sw = _load_script("stream_weights")
    idx, wdir = _toy_index(sw, tmp_path)
    root = tmp_path / "cache"
    assert idx.keep_shards() == {"model-00001-of-00008.safetensors", "model-00008-of-00008.safetensors"}
    st = _status(root, [_part(l) for l in range(7)])
    golden = range(7)
    cand = sw.deletable_shards(idx, sw.eligible_layers(st, golden))
    assert cand == [f"model-{i:05d}-of-00008.safetensors" for i in range(2, 8)]
    kw = dict(status=st, golden_done=golden, cache_root=root)
    with pytest.raises(sw.StreamError):
        sw.delete_shards(idx, cand[:1], allow=False, **kw)  # no --delete-converted
    assert (wdir / cand[0]).exists()
    with pytest.raises(sw.StreamError):  # the shard of a layer whose golden is not done: re-derived, not trusted
        sw.delete_shards(idx, ["model-00004-of-00008.safetensors"], allow=True, status=st, golden_done=[0, 1],
                         cache_root=root)
    with pytest.raises(sw.StreamError):  # a kept (global) shard
        sw.delete_shards(idx, ["model-00008-of-00008.safetensors"], allow=True, **kw)
    with pytest.raises(sw.StreamError):  # a kept layer
        sw.delete_shards(idx, ["model-00003-of-00008.safetensors"], allow=True, keep_layers=[2], **kw)
    for bad in (_status(root, st["parts"], tag="motif3-deadbeef-c1-x"), _status(root, st["parts"], mesh=(8, 4)),
                _status(tmp_path / "elsewhere", st["parts"])):
        with pytest.raises(sw.StreamError) as e:  # not the serving cache under --cache-root
            sw.delete_shards(idx, cand[:1], allow=True, status=bad, golden_done=golden, cache_root=root)
        assert e.value.code == sw.EXIT_USAGE
    assert all((wdir / s).exists() for s in cand)
    state = sw.StreamState(tmp_path / sw.STATE_NAME)
    freed = sw.delete_shards(idx, cand[:2], allow=True, state=state, logger=log, **kw)
    assert freed == 42 + 43 and not (wdir / cand[0]).exists() and not (wdir / cand[1]).exists()
    left = sorted(p.name for p in wdir.iterdir() if not p.name.startswith("."))
    assert "config.json" in left and "tokenizer.json" in left and "model.safetensors.index.json" in left
    assert json.loads((tmp_path / sw.STATE_NAME).read_text())["deleted"][0]["shard"] == cand[0]


class _FakeRunner:
    """Stands in for ``stream_weights.Runner`` around a status document (no subprocess)."""

    def __new__(cls, sw, doc, **kw):
        class _R(sw.Runner):
            def cache_status(self, layers, with_globals):
                self.calls.append(("status",))
                return json.loads(json.dumps(self.doc))

            def convert(self, layer, *, force=False):
                self.calls.append(("convert", layer, force))
                rc = self.convert_rc.pop(0) if self.convert_rc else 0
                if rc == 0 and self.on_convert is not None:
                    self.on_convert(self, layer, force)
                return rc

            def download(self, layer, margin_gb):
                self.calls.append(("download", layer))
                return 0

            def verify_sha(self):
                self.calls.append(("verify_sha",))
                return 0

            def golden(self, layer, out=None):
                self.calls.append(("golden", layer) if out is None else ("golden", layer, Path(out).name))
                if self.on_golden is not None:
                    self.on_golden(self, layer, out)
                return 0

        r = _R(**kw)
        r.doc, r.calls, r.convert_rc, r.on_convert, r.on_golden = doc, [], [], None, None
        return r


def test_host_pipeline_deletion_gate(tmp_path):
    sw = _load_script("stream_weights")
    idx, wdir = _toy_index(sw, tmp_path)
    root = tmp_path / "cache"
    parts = [_part(0), _part(1), _part(2, verified=False), _part(3, files_hashed=2), _part(4, fp="0" * 64), _part(5),
             _part(6), _part(None)]
    doc = _status(root, parts)
    golden = _golden(tmp_path, [0, 1, 2, 3, 4, 6])  # not 5
    shard = {l: f"model-{l + 1:05d}-of-00008.safetensors" for l in range(1, 7)}

    def pipe(delete=True, keep=(6,), dry_run=False, d=doc):
        r = _FakeRunner(sw, d, weights_dir=wdir, golden_out=golden, cache_root=root, dry_run=dry_run)
        plan = sw.Plan(layers=list(range(7)), with_globals=True, delete=delete, min_free_gb=60, keep_layers=set(keep))
        return sw.Pipeline(plan, r, index=idx, state_path=tmp_path / "state.json"), r

    # eligible: L0 (but its only shard holds the embedding: kept for good) and L1; not L2 (unverified), L3 (not every
    # file re-hashed), L4 (built by other code), L5 (golden missing), L6 (kept, --keep-bf16)
    p, r = pipe()
    p.refresh_status()
    assert p.done_layers() == {0, 1}
    # without --delete-converted: listed, not deleted; the same in a dry run
    for kw in (dict(delete=False), dict(dry_run=True)):
        q, _ = pipe(**kw)
        assert q.delete_consumed() == 0 and (wdir / shard[1]).exists()
    # another converter holds the converter lock: nothing is decided or deleted meanwhile
    lockp = Path(doc["cache_dir"]) / ".convert.lock"
    lockp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lockp), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert p.delete_consumed() == 0 and (wdir / shard[1]).exists()
    finally:
        os.close(fd)
    # a cache that is not the serving one (e.g. TT_MODEL_WEIGHTS_REVISION): refused before anything is decided
    for bad in (_status(root, parts, tag="motif3-deadbeef-c1-x"), _status(root, parts, mesh=(8, 4))):
        q, _ = pipe(d=bad)
        with pytest.raises(sw.StreamError) as e:
            q.delete_consumed()
        assert e.value.code == sw.EXIT_USAGE and (wdir / shard[1]).exists()
    # the real thing: exactly shard 2 (layer 1) goes, re-read under the lock
    n_status = sum(1 for c in r.calls if c == ("status",))
    freed = p.delete_consumed()
    assert freed == idx.sizes[shard[1]] and not (wdir / shard[1]).exists()
    assert sum(1 for c in r.calls if c == ("status",)) == n_status + 1, "the status is re-read under the lock"
    assert all((wdir / shard[l]).exists() for l in range(2, 7))
    # with --keep-bf16 none and L5's golden done, L6 and L5 go too
    golden2 = _golden(tmp_path, range(7))
    p2, _ = pipe(keep=())
    p2.runner.golden_out = golden2
    assert p2.delete_consumed() == idx.sizes[shard[5]] + idx.sizes[shard[6]]
    assert all((wdir / shard[l]).exists() for l in (2, 3, 4))


def test_host_pipeline_conversion_steps(tmp_path):
    sw = _load_script("stream_weights")
    idx, wdir = _toy_index(sw, tmp_path)
    root = tmp_path / "cache"
    golden = _golden(tmp_path, range(7))
    _verify_all(wdir, idx)
    doc = _status(root, [_part(1), _part(None)])

    def pipe(d, delete=True, keep=()):
        r = _FakeRunner(sw, d, weights_dir=wdir, golden_out=golden, cache_root=root)
        plan = sw.Plan(layers=list(range(7)), with_globals=False, delete=delete, min_free_gb=0, keep_layers=set(keep))
        p = sw.Pipeline(plan, r, index=idx, state_path=tmp_path / "state.json")
        p.refresh_status()
        return p, r

    # what the guards reserve: nothing for a complete part that only needs its verification, the converter's own
    # estimate otherwise, the whole part for a forced rebuild
    assert sw.conversion_bytes({"complete": True, "need_bytes": 0}, force=False, full=7) == 0
    assert sw.conversion_bytes({"need_bytes": 35_840}, force=False, full=7) == 35_840
    assert sw.conversion_bytes({}, force=False, full=7) == 7
    assert sw.conversion_bytes({"need_bytes": 0, "rebuild_bytes": 5}, force=True, full=7) == 5
    assert sw.conversion_bytes({"rebuild_bytes": 0}, force=True, full=7) == 7
    # converter exit 3: one deletion pass (L1's shard goes) and one retry; still 3 -> pipeline exit 3
    p, r = pipe(doc)
    r.convert_rc = [3, 3]
    with pytest.raises(sw.StreamError) as e:
        p.convert_step(3)
    assert e.value.code == sw.EXIT_DISK and [c for c in r.calls if c[0] == "convert"] == [("convert", 3, False)] * 2
    assert not (wdir / "model-00002-of-00008.safetensors").exists()
    # nothing to delete: no retry
    p, r = pipe(doc)
    r.convert_rc = [3]
    with pytest.raises(sw.StreamError):
        p.convert_step(3)
    assert len([c for c in r.calls if c[0] == "convert"]) == 1
    for rc, code in ((4, sw.EXIT_DOWNLOAD), (1, sw.EXIT_CONVERT), (5, sw.EXIT_CONVERT)):
        p, r = pipe(doc)
        r.convert_rc = [rc]
        with pytest.raises(sw.StreamError) as e:
            p.convert_step(3)
        assert e.value.code == code, (rc, e.value)
    # a converted layer built by other code is rebuilt (--force) before its shard can go; then it goes
    old = _status(root, [_part(3, fp="0" * 64, rebuild=6_000_000_000), _part(None)])
    p, r = pipe(old)

    def fixed(runner, layer, force):
        assert layer == 3 and force
        runner.doc = _status(root, [_part(3), _part(None)])

    r.on_convert = fixed
    assert (wdir / "model-00004-of-00008.safetensors").exists()
    p.do_layer(3)
    assert ("convert", 3, True) in r.calls and not (wdir / "model-00004-of-00008.safetensors").exists()
    # the same layer kept (--keep-bf16): not rebuilt, not deleted
    (wdir / "model-00004-of-00008.safetensors").write_bytes(b"\0" * idx.sizes["model-00004-of-00008.safetensors"])
    p, r = pipe(old, keep=(3,))
    p.do_layer(3)
    assert not [c for c in r.calls if c[0] == "convert"] and (wdir / "model-00004-of-00008.safetensors").exists()
    # an unconverted layer: converted (no --force), then its shard goes
    p, r = pipe(_status(root, [_part(None)]))
    r.on_convert = lambda runner, layer, force: setattr(runner, "doc", _status(root, [_part(5), _part(None)]))
    p.do_layer(5)
    assert ("convert", 5, False) in r.calls and not (wdir / "model-00006-of-00008.safetensors").exists()


def test_host_pipeline_extra_golden(tmp_path):
    sw = _load_script("stream_weights")
    idx, wdir = _toy_index(sw, tmp_path)
    root = tmp_path / "cache"
    _verify_all(wdir, idx)
    doc = _status(root, [_part(l) for l in range(7)] + [_part(None)])
    shard = {l: f"model-{l + 1:05d}-of-00008.safetensors" for l in range(1, 7)}

    def golden_dir(name, layers):
        g = tmp_path / name
        g.mkdir(exist_ok=True)
        (g / "manifest.json").write_text(json.dumps({"layers_done": sorted(layers), "last_layer": max(layers)}))
        return g

    def advance(runner, layer, out):  # what golden_stream --resume --layers L does to the manifest
        g = Path(runner.golden_out if out is None else out)
        m = json.loads((g / "manifest.json").read_text())
        (g / "manifest.json").write_text(json.dumps({"layers_done": sorted(set(m["layers_done"]) | {layer}),
                                                     "last_layer": layer}))

    def pipe(extra):
        r = _FakeRunner(sw, doc, weights_dir=wdir, golden_out=c2, cache_root=root, extra_golden_outs=extra)
        r.on_golden = advance
        plan = sw.Plan(layers=list(range(7)), with_globals=False, delete=True, min_free_gb=0, keep_layers=set())
        p = sw.Pipeline(plan, r, index=idx, state_path=tmp_path / "state.json")
        p.refresh_status()
        return p, r

    c2 = golden_dir("c2", range(6))  # the C2 golden has 0-5
    fp32 = golden_dir("c2_fp32", range(5))  # the fp32 one 0-4
    with pytest.raises(ValueError):  # the same run twice
        sw.Runner(cache_root=root, golden_out=c2, extra_golden_outs=[tmp_path / "c2"])
    g = [str(c) for c in sw.Runner(cache_root=root, golden_out=c2, extra_golden_outs=[fp32]).golden_cmd(40, fp32)]
    assert g[g.index("--out") + 1] == str(fp32) and g[g.index("-n") + 1] == "golden_c2_fp32_L40"
    assert "--resume" in g and g[g.index("--layers") + 1] == "40" and g[g.index("--ckpt-dir") + 1] == str(sw.WEIGHTS)
    # an extra golden behind by more than one layer: the layer stops (exit 6) before any golden runs or shard goes
    p, r = pipe([golden_dir("stale", range(4))])
    with pytest.raises(sw.StreamError) as e:
        p.do_layer(5)
    assert e.value.code == sw.EXIT_GOLDEN and not [c for c in r.calls if c[0] == "golden"]
    assert all((wdir / shard[l]).exists() for l in range(1, 7))
    # the gate is the intersection: L5 (C2 only) and L6 (neither) stay; L1-L4 go (L0's shard holds the embedding)
    p, r = pipe([fp32])
    assert p.golden_done() == set(range(5)) and p.done_layers() == set(range(5))
    assert p.delete_consumed() == sum(idx.sizes[shard[l]] for l in range(1, 5))
    assert (wdir / shard[5]).exists() and (wdir / shard[6]).exists()
    # L5: only the pending golden (fp32) is resumed, then its shard goes
    p.do_layer(5)
    assert [c for c in r.calls if c[0] == "golden"] == [("golden", 5, "c2_fp32")] and not (wdir / shard[5]).exists()
    # L6: both, the C2 golden first; then the shard goes
    p.do_layer(6)
    assert [c for c in r.calls if c[0] == "golden"][1:] == [("golden", 6), ("golden", 6, "c2_fp32")]
    assert not (wdir / shard[6]).exists() and sw.goldens_done_layers([c2, fp32]) == set(range(7))
    state = json.loads((tmp_path / "state.json").read_text())
    assert "golden:c2_fp32" in state["layers"]["6"] and "golden" in state["layers"]["6"]
    assert sw.goldens_done_layers([]) == set() and sw.goldens_done_layers([c2, tmp_path / "absent"]) == set()
    # the dry-run timeline scales the golden step with the number of goldens
    rows = sw.simulate(idx, [6], with_globals=False, parts={6: _part(6)}, fingerprint=FP, golden_done=range(6),
                       delete=False, free_gb0=100.0, min_free_gb=0.0, present=set(idx.shards), n_goldens=2)
    assert [r["step"] for r in rows if r["unit"] == 6] == ["golden x2"]


def test_host_runner_commands(tmp_path):
    sw = _load_script("stream_weights")
    r = sw.Runner(cache_root=tmp_path, variant_args=["--router=both", "--lm-head-split", "tp"])
    cmd = [str(c) for c in r.convert_cmd(5)]
    i = cmd.index("--mesh")
    assert cmd[i + 1] == "4x8" and cmd[cmd.index("--cache-root") + 1] == str(tmp_path)
    assert cmd[cmd.index("--weights-dir") + 1] == str(sw.WEIGHTS) and cmd[cmd.index("--layers") + 1] == "5"
    assert "--router=both" in cmd and "--lm-head-split=tp" in cmd and "--force" not in cmd
    assert "--force" in [str(c) for c in r.convert_cmd(5, force=True)]
    assert [str(c) for c in r.convert_cmd(None)][-len(r.pins()) - 2:-2] == [str(c) for c in r.pins()]
    st = [str(c) for c in r.status_cmd([0, 1], True)]
    assert "--status" in st and "--json" in st and "--globals" in st and st[st.index("--mesh") + 1] == "4x8"
    assert st[st.index("--cache-root") + 1] == str(tmp_path) and "--lm-head-split=tp" in st
    g = [str(c) for c in r.golden_cmd(40)]
    assert g[g.index("--ckpt-dir") + 1] == str(sw.WEIGHTS) and g[g.index("--layers") + 1] == "40"
    for bad in (["--mesh=8x4"], ["--cache-root=/x"], ["--weights-dir=/x"], ["--no-hash"], ["--no-verify"], ["--force"],
                ["--layers=3"], ["--globals"], ["--target=device"], ["--min-free-gb=1"], ["--router=fast"],
                ["--rout=both"], ["both"]):
        with pytest.raises(ValueError):
            sw.normalize_convert_args(bad)
    assert sw.normalize_convert_args(["--mhc-sinkhorn", "stock", "--embedding=both"]) == ["--mhc-sinkhorn=stock",
                                                                                        "--embedding=both"]
    other = sw.Runner(weights_dir=tmp_path, cache_root=tmp_path)
    for call in (lambda: other.download(3, 60.0), lambda: other.verify_sha()):
        with pytest.raises(sw.StreamError) as e:  # the downloader / verifier work on their hard-coded dir only
            call()
        assert e.value.code == sw.EXIT_USAGE
    assert sw.default_cache_root({}) == sw.ROOT / "tt_cache"
    assert sw.default_cache_root({"TT_CACHE_PATH": "/a", "MOTIF3_TT_CACHE_PATH": "/b"}) == Path("/b")


def test_host_verified_shards_stat(tmp_path):
    sw = _load_script("stream_weights")
    a, b, c = (f"model-{i:05d}-of-00003.safetensors" for i in (1, 2, 3))
    (tmp_path / a).write_bytes(b"x" * 23)
    (tmp_path / c).write_bytes(b"z" * 5)
    st = os.stat(tmp_path / a)
    doc = {a: {"status": "ok", "stat": [12345, 1.0]}, b: {"status": "ok", "stat": [5, 1.0]},
           c: {"status": "BAD", "stat": None}}
    (tmp_path / ".verified.json").write_text(json.dumps(doc))
    v = sw.verified_shards(tmp_path)
    assert v == {a: "stale", b: "missing", c: "BAD"}, v  # a re-downloaded shard is not trusted from its old record
    doc[a]["stat"] = [st.st_size, st.st_mtime]
    (tmp_path / ".verified.json").write_text(json.dumps(doc))
    assert sw.verified_shards(tmp_path)[a] == "ok"
    os.utime(tmp_path / a, (st.st_atime, st.st_mtime + 5))
    assert sw.verified_shards(tmp_path)[a] == "stale"


def test_host_dry_run_timeline():
    sw = _load_script("stream_weights")
    cw = sw.CW
    idx = sw.ShardIndex.load()  # static; the state below is synthetic (independent of the live disk and golden)
    # the 2026-10-02 state: globals + layers 0-35 on disk, golden 0-35 done, globals + L00-L02 converted
    present = {s for s in idx.shards if any(isinstance(u, str) or u <= 35 for u in idx.shard_units[s])}  # 105
    golden = set(range(36))
    est = cw.EST_PART_BYTES

    def parts(fp_layers=FP):
        return {None: _part(None, bytes_=est["global"]), **{l: _part(l, fp=fp_layers, bytes_=est[sw.part_kind(l)],
                                                                 rebuild=est[sw.part_kind(l)]) for l in (0, 1, 2)}}

    kw = dict(with_globals=True, fingerprint=FP, golden_done=golden, min_free_gb=60.0, present=present)

    def stop(rows):
        return next((r for r in rows if not r["ok"]), None)

    # after sign-off (--keep-bf16 none) the deleting stream fits from 61 GB, never below the floor, no download < 36
    rows = sw.simulate(idx, list(range(53)), parts=parts(), delete=True, keep_layers=(), free_gb0=61.0, **kw)
    assert stop(rows) is None, stop(rows)
    low = min(r["free_gb"] for r in rows)
    log(f"projected stream from 61 GB, keep none: min {low:.1f} GB, final {rows[-1]['free_gb']:.1f} GB")
    assert low >= 60.0 and rows[-1]["free_gb"] > 100
    assert not any(r["step"].startswith("download") for r in rows if isinstance(r["unit"], int) and r["unit"] < 36)
    assert any(r["step"] == "golden" and r["unit"] == 36 for r in rows)
    # without deleting it cannot fit (CONV-3): it stops at the first MoE conversion
    s = stop(sw.simulate(idx, list(range(53)), parts=parts(), delete=False, free_gb0=61.0, **kw))
    assert s and s["unit"] == 3 and "convert" in s["step"], s
    # the default --keep-bf16 0-35: nothing local can go, so the same stop
    s = stop(sw.simulate(idx, list(range(53)), parts=parts(), delete=True, keep_layers=range(36), free_gb0=61.0, **kw))
    assert s and s["unit"] == 3, s
    # the guard is the converter's own: 1.1 x the estimate (the first MoE conversion needs 7.32 GB above the floor)
    need = cw.guarded(est["moe"])
    for margin, first_stop in ((-0.01, 3), (+0.01, 4)):
        f0 = 60.0 + need / 1e9 + margin
        s = stop(sw.simulate(idx, [3, 4], parts=parts(), delete=False, free_gb0=f0, **kw))
        assert s and s["unit"] == first_stop, (margin, s)
        assert cw.room_ok(int(f0 * 1e9), need, 60.0) == (first_stop == 4)
    # layers converted by other code are rebuilt before their shards go (and only those whose shards could go)
    rows = sw.simulate(idx, list(range(5)), parts=parts(fp_layers="0" * 64), delete=True, keep_layers=(),
                       free_gb0=80.0, **kw)
    assert [r["unit"] for r in rows if r["step"] == "rebuild"] == [1, 2], rows  # L0: only shard 1 (embedding)
    assert stop(rows) is None


def test_host_mock_conversion():
    """End to end on the mock cluster (subprocesses: the mock target is selected before ttnn starts)."""
    if _devices_visible():
        pytest.skip("devices visible: run under scripts/hostrun.sh")
    from models.demos.motif3.tt.model_config import MotifTTConfig
    from models.demos.motif3.tt.weights import HFWeightLoader

    cw = _load_script("convert_weights")
    if not HFWeightLoader().layer_available(0):
        pytest.skip("layer 0 is not on disk")
    if _free_gb(HOST_SCRATCH) < TEST_SCRATCH_MIN_FREE_GB + 2:
        pytest.skip(f"not enough free disk for a 0.4 GB scratch conversion above {TEST_SCRATCH_MIN_FREE_GB} GB")
    root = HOST_SCRATCH / f"run_{os.getpid()}"
    env = {k: v for k, v in os.environ.items() if k not in ("TT_METAL_MOCK_CLUSTER_DESC_PATH", "MESH_DEVICE")}

    def convert(*extra, env_extra=None):
        rep = root.parent / f"report_{os.getpid()}.json"
        rep.unlink(missing_ok=True)
        cmd = [sys.executable, str(SCRIPTS / "convert_weights.py"), "--layers", "0", "--cache-root", str(root),
               "--min-free-gb", str(TEST_SCRATCH_MIN_FREE_GB), "--report", str(rep), *extra]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env={**env, **(env_extra or {})})
        log(f"convert_weights.py {' '.join(extra)}: exit {p.returncode} in {time.time() - t0:.1f} s")
        out = json.loads(rep.read_text()) if rep.is_file() else None
        rep.unlink(missing_ok=True)
        return p, out

    def status(*extra, env_extra=None):
        cmd = [sys.executable, str(SCRIPTS / "convert_weights.py"), "--status", "--json", "--layers", "0", "--globals",
               "--cache-root", str(root), *extra]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env={**env, **(env_extra or {})})
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        i = p.stdout.find('{\n "cache_root"')
        return json.JSONDecoder().raw_decode(p.stdout[i:])[0], p.stderr

    try:
        # (a) motif-only build: 22 files, hashed, verified, current fingerprint
        p, rep = convert("--mhc-sinkhorn", "motif")
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
        e = rep["parts"]["L00"]
        assert e["verify"]["ok"] and e["stats"]["n_files"] == 22 and e["verify"]["files_hashed"] == 22, e
        assert e["stats"]["target"] == "mock" and e["stats"]["code"]["sha256"] == rep["code_fingerprint"]
        d = Path(rep["cache_dir"]) / "L00"
        inodes = {n: (d / n).stat().st_ino for n in e["stats"]["files"]}
        # (b) the defaults add the stock constants: the 22 files are reused (same inodes), 8 new ones built
        p, rep = convert()
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
        s = rep["parts"]["L00"]["stats"]
        assert s["n_files"] == 30 and s["reused_files"] == 22 and s["new_files"] == 8, s
        assert all((d / n).stat().st_ino == ino for n, ino in inodes.items()), "reused files must not be rewritten"
        assert s["variants"] == {"mhc_sinkhorn": ["motif", "stock"]} and s["code"]["sha256"] == rep["code_fingerprint"]
        assert s["source"]["tensors"] < 20, s["source"]  # only the mHC scalars were read, not the layer
        new = [f for n, f in s["files"].items() if n not in inodes]
        assert sum(f["bytes"] for f in new) == cw.EST_VARIANT_BYTES[("mhc_sinkhorn", "stock")], new
        assert rep["parts"]["L00"]["verify"]["ok"] and rep["parts"]["L00"]["verify"]["files_hashed"] == 30
        shas = {n: f["sha256"] for n, f in s["files"].items()}
        # (c) --force rebuilds from BF16: same bytes, new inodes, a stale (renamed) file is removed
        stale = d / "attn.v1.old__rep_dtype_BFLOAT16_layout_TILE.tensorbin"
        stale.write_bytes(b"stale")
        p, rep = convert("--force")
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
        s = rep["parts"]["L00"]["stats"]
        assert not stale.exists() and s["removed_stale"] == [stale.name], s["removed_stale"]
        assert {n: f["sha256"] for n, f in s["files"].items()} == shas, "a rebuild must give identical bytes"
        assert s["reused_files"] == 0 and any((d / n).stat().st_ino != ino for n, ino in inodes.items())
        marker = (d / ".complete").read_text()
        stats_txt = (d / ".convert.json").read_text()
        # (d) exit 3 (disk guard) and exit 4 (BF16 missing) happen before anything in the part changes
        p, _ = convert("--force", "--min-free-gb", "100000")
        assert p.returncode == cw.EXIT_DISK, p.stdout[-2000:]
        wd = root.parent / f"weights_{os.getpid()}"
        wd.mkdir(exist_ok=True)
        for f in ("config.json", "model.safetensors.index.json"):
            shutil.copy(Path(HFWeightLoader().dir) / f, wd / f)
        p, _ = convert("--force", "--weights-dir", str(wd))
        shutil.rmtree(wd, ignore_errors=True)
        assert p.returncode == cw.EXIT_SOURCE, p.stdout[-2000:]
        assert (d / ".complete").read_text() == marker and (d / ".convert.json").read_text() == stats_txt
        # (e) the status: per kind, the serving cache, the fingerprint; MESH_DEVICE=BH-Galaxy is ignored
        doc, err = status("--lm-head-split", "both", env_extra={"MESH_DEVICE": "BH-Galaxy"})
        assert doc["mesh_shape"] == [4, 8] and "MESH_DEVICE" in err and doc["is_serving_cache"] is True, doc
        assert doc["cache_dir"] == str(Path(doc["serving_cache_dir"]))
        l0 = next(x for x in doc["parts"] if x["layer"] == 0)
        g = next(x for x in doc["parts"] if x["layer"] is None)
        assert l0["complete"] and l0["verified"] is True and l0["hashes"] == 30 and l0["files_hashed"] == 30, l0
        assert l0["code_fingerprint"] == doc["code_fingerprint"] and l0["need_bytes"] == 0
        assert not g["complete"] and g["need_bytes"] == cw.estimate_part_bytes(
            "global", {"lm_head_split": ["mesh", "tp"], "embedding": ["replicated"]})
        doc, _ = status("--mesh", "8x4")
        assert doc["is_serving_cache"] is False and doc["cache_dir"].endswith("mesh8x4")
        # (f) byte identity with the real (mock-built) cache's layer 0 when it is complete for the defaults
        real_cfg = MotifTTConfig.from_hf_config(mesh_shape=(4, 8), tt_cache_root=REAL_ROOT, num_layers=53)
        real = cw.part_status(real_cfg, 0, cw.ConvertOptions())
        if not real.complete:
            log(f"the real cache's L00 is not complete for the default options ({real.reason}): byte comparison "
                f"skipped")
        else:
            theirs = {n: f.get("sha256") for n, f in real.stats["files"].items()}
            assert shas == theirs, sorted(n for n in shas if shas[n] != theirs.get(n))
            log(f"{len(shas)} / {len(theirs)} files byte-identical to the real cache's L00")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        with contextlib.suppress(OSError):
            HOST_SCRATCH.rmdir()


def _real_stats(cfg, part) -> Optional[dict]:
    """``.convert.json`` of a part in the real cache (default root, serving tag), or None."""
    from models.demos.motif3.tt import weights as W
    from models.demos.motif3.tt.model_config import MotifTTConfig

    cfg = cfg or MotifTTConfig.from_hf_config(mesh_shape=(4, 8), tt_cache_root=REAL_ROOT)
    p = W.layer_cache_marker(cfg, part).parent / ".convert.json"
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None


# ======================================================================================================================
# device helpers
# ======================================================================================================================
_KEEP_TYPES = ("MotifCCL", "MotifRope", "MotifTTConfig", "HFWeightLoader", "RaisingSource", "CountingSource")


def _tensor_leaves(obj, prefix: str = "") -> Dict[str, Any]:
    """``{attribute path: device ttnn.Tensor}`` of a module tree (motif3 objects, lists, tuples, dicts; shared
    infrastructure objects are not entered)."""
    out: Dict[str, Any] = {}
    seen = set()

    def walk(v, path):
        if id(v) in seen:
            return
        seen.add(id(v))
        if isinstance(v, ttnn.Tensor):
            if v.storage_type() == ttnn.StorageType.DEVICE and v.is_allocated():
                out[path] = v
            return
        if isinstance(v, dict):
            for k in sorted(v, key=str):
                walk(v[k], f"{path}[{k}]")
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                walk(x, f"{path}[{i}]")
        elif hasattr(v, "__dict__") and type(v).__module__.startswith("models.demos.motif3"):
            if type(v).__name__ in _KEEP_TYPES:
                return
            for k in sorted(vars(v)):
                walk(vars(v)[k], f"{path}.{k}" if path else k)

    walk(obj, prefix)
    return out


def _host(t) -> torch.Tensor:
    return ttnn.to_torch(t)


def _bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    view = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}.get(t.dtype)
    return t.view(view) if view is not None else t


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _spec(t) -> tuple:
    return (tuple(t.shape), str(t.dtype), str(t.layout), str(t.memory_config()))


_BYTES_PER_ELEM = {"BFLOAT16": 2.0, "FLOAT32": 4.0, "UINT32": 4.0, "INT32": 4.0, "UINT16": 2.0, "UINT8": 1.0,
                   "BFLOAT8_B": 1088 / 1024, "BFLOAT4_B": 576 / 1024}


def _chip_bytes(t) -> float:
    n = 1
    for d in t.shape:
        n *= int(d)
    return n * _BYTES_PER_ELEM.get(str(t.dtype).split(".")[-1], 4.0)


def _compare_device_tensors(a, b, *, sample_bytes: float = 2.5e9) -> Dict[str, Any]:
    """Bitwise per chip; tensors above ``sample_bytes`` over the whole mesh are compared on 4 chips."""
    da, db = ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)
    if len(da) != len(db):
        return {"ok": False, "why": f"{len(da)} vs {len(db)} device tensors", "chips": 0}
    idx = list(range(len(da)))
    if _chip_bytes(da[0]) * len(da) > sample_bytes:
        idx = sorted({0, len(da) // 3, (2 * len(da)) // 3, len(da) - 1})
    for i in idx:
        x, y = _host(da[i]), _host(db[i])
        if not _bitwise_equal(x, y):
            return {"ok": False, "why": f"chip {i} differs", "chips": len(idx)}
    return {"ok": True, "chips": len(idx)}


def _mesh_bits(t, mesh) -> List[torch.Tensor]:
    return [_host(d) for d in ttnn.get_device_tensors(t)]


def _same_on_mesh(a, b, mesh) -> bool:
    return all(_bitwise_equal(x, y) for x, y in zip(_mesh_bits(a, mesh), _mesh_bits(b, mesh)))


def _nonfinite(t, mesh) -> int:
    """Non-finite values over every chip's copy (floating dtypes; integer tensors count 0)."""
    n = 0
    for x in _mesh_bits(t, mesh):
        if x.is_floating_point():
            n += int((~torch.isfinite(x.float())).sum())
    return n


def _prompt_ids(name: str) -> List[int]:
    doc = json.loads(PROMPTS.read_text())
    for p in doc["prompts"]:
        if p["name"] == name:
            return [int(i) for i in p["ids"]]
    raise KeyError(name)


def _replicated(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


def _alloc_kv(mesh, cfg, num_blocks: int):
    e = ttnn.empty([num_blocks, 1, cfg.kv_block_size, cfg.kv_latent_dim], cfg.dtypes.kv_cache, ttnn.TILE_LAYOUT, mesh,
                   ttnn.DRAM_MEMORY_CONFIG)
    kv = ttnn.fill(e, 0.0)
    ttnn.deallocate(e)
    return kv


def _free(*ts):
    for t in ts:
        if isinstance(t, (list, tuple)):
            _free(*t)
        elif isinstance(t, dict):
            _free(*t.values())
        elif isinstance(t, ttnn.Tensor) and t.storage_type() == ttnn.StorageType.DEVICE and t.is_allocated():
            ttnn.deallocate(t)


class _Failures(list):
    """A failure list that also logs every entry when it is added (a later exception does not hide earlier ones)."""

    def append(self, msg):
        log(f"FAILURE: {msg}")
        super().append(msg)


class _Build:
    """Embedding + decoder layers + LM head + the variant modules (the exact-fp32 router of L02, the stock mHC sites of
    every layer), built one way (cached with a raising source, or fresh from BF16)."""

    def __init__(self, mesh, cfg, *, source, cache: bool, ccl, rope, layers=(0, 1, 2)):
        from models.demos.motif3.tt.decoder import MotifDecoderLayer
        from models.demos.motif3.tt.embedding import MotifEmbedding
        from models.demos.motif3.tt.kernels.router_fp32 import RouterLogitsFP32
        from models.demos.motif3.tt.lm_head import MotifLMHead
        from models.demos.motif3.tt.mhc import MHCSite

        t0 = time.time()
        self.shared = (ccl, rope, cfg)
        self.embed = MotifEmbedding(mesh, cfg, source=source, ccl=ccl, cache=cache)
        self.head = MotifLMHead(mesh, cfg, source=source, ccl=ccl, cache=cache)
        self.layers = [MotifDecoderLayer(mesh, cfg, l, source=source, ccl=ccl, rope=rope, cache=cache) for l in layers]
        moe = [l for l in layers if cfg.layer(l).is_moe]
        self.router_fp32 = {l: RouterLogitsFP32.from_source(mesh, cfg, l, source=source, cache=cache) for l in moe}
        self.mhc_stock = {(l, site): MHCSite(mesh, cfg, l, site, source=source, cache=cache, sinkhorn="stock")
                          for l in layers for site in ("mhc_attn", "mhc_ffn")}
        self.seconds = time.time() - t0

    def objects(self) -> Dict[str, Any]:
        d = {"embed": self.embed, "head": self.head}
        d.update({f"L{l.layer_idx:02d}": l for l in self.layers})
        d.update({f"router_fp32.L{l:02d}": r for l, r in self.router_fp32.items()})
        d.update({f"mhc_stock.L{l:02d}.{site}": m for (l, site), m in self.mhc_stock.items()})
        return d

    def deallocate(self):
        cw = _load_script("convert_weights")
        for o in self.objects().values():
            cw.free_object(o, keep=self.shared)


# ======================================================================================================================
# device tests
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_device_convert_roundtrip(mesh_device):
    from models.demos.motif3.tt.attention import MotifAttention
    from models.demos.motif3.tt.ccl import MotifCCL, log_fabric
    from models.demos.motif3.tt.rope import MotifRope, shard_lanes
    from models.demos.motif3.tt.weights import HFWeightLoader

    rep = log_fabric(mesh_device, "test_device_convert_roundtrip")
    cw = _load_script("convert_weights")
    src = HFWeightLoader()
    if not all(src.layer_available(l) for l in (0, 1, 2)):
        pytest.skip("layers 0-2 are not all on disk")
    opts = cw.ConvertOptions()
    real_cfg = cw.build_cfg(mesh_device, cache_root=str(REAL_ROOT), weights_dir=None)
    real = {p: cw.part_status(real_cfg, p, opts) for p in PARTS}
    not_ready = [f"{cw.part_tag(p)} ({st.reason})" for p, st in real.items() if not (st.complete and st.hashes == st.files)]
    if not_ready:  # the byte comparison is the point of step 2: never pass with nothing compared
        pytest.skip(f"the mock-built real cache is not complete for the default options: {not_ready} "
                    f"(scripts/convert_weights.py --layers 0-2 --globals first)")
    need = sum(cw.estimate_part_bytes(cw.part_kind(real_cfg, p), opts.variants_for(cw.part_kind(real_cfg, p), real_cfg))
               for p in PARTS) * cw.GUARD_FACTOR / 1e9
    if _free_gb(TEST_ROOT) - need < TEST_SCRATCH_MIN_FREE_GB:
        pytest.skip(f"{_free_gb(TEST_ROOT):.1f} GB free: a {need:.1f} GB test cache would leave < "
                    f"{TEST_SCRATCH_MIN_FREE_GB} GB")
    failures = _Failures()
    cfg = cw.build_cfg(mesh_device, cache_root=str(TEST_ROOT), weights_dir=None)
    if cfg.cache_dir.exists():
        shutil.rmtree(cfg.cache_dir)
    try:
        # ---- 1. convert on the device -----------------------------------------------------------------------------
        conv = cw.Converter(mesh_device, cfg, source=src, target="device", logger=log,
                            min_free_gb=TEST_SCRATCH_MIN_FREE_GB)
        t0 = time.time()
        report = conv.run(PARTS, verify=True)
        conv.close()
        log(f"device conversion of {[cw.part_tag(p) for p in PARTS]}: {time.time() - t0:.1f} s; "
            + ", ".join(f"{k} build {v['stats']['seconds']['build']} s / {v['stats']['bytes'] / 1e9:.3f} GB / verify "
                        f"{v['verify']['seconds']} s" for k, v in report.items()))
        for k, v in report.items():
            if not (v.get("verify") or {}).get("ok") or v["verify"]["files_hashed"] != v["stats"]["n_files"]:
                failures.append(f"{k}: verification failed or incomplete: {v.get('verify')}")
        # ---- 2. byte identity with the mock-built real cache: every file of every part ---------------------------
        n_same = n_cmp = n_total = 0
        for part in PARTS:
            tag = cw.part_tag(part)
            mine = report[tag]["stats"]["files"]
            theirs = real[part].stats["files"]
            n_total += len(mine)
            if real[part].stats.get("target") != "mock":
                log(f"{tag}: the real cache part was built on target {real[part].stats.get('target')}")
            if set(mine) != set(theirs):
                failures.append(f"{tag}: file sets differ (device-only {sorted(set(mine) - set(theirs))[:3]}, mock-only "
                                f"{sorted(set(theirs) - set(mine))[:3]})")
            for n in sorted(set(mine) & set(theirs)):
                n_cmp += 1
                if mine[n].get("sha256") and mine[n].get("sha256") == theirs[n].get("sha256") \
                        and mine[n]["bytes"] == theirs[n]["bytes"]:
                    n_same += 1
                else:
                    failures.append(f"{tag}/{n}: device-built and mock-built files differ")
        if n_cmp != n_total or n_total == 0:
            failures.append(f"compared {n_cmp} of {n_total} device-built files")
        log(f"device-built vs mock-built cache: {n_same}/{n_total} files byte-identical (sha256)")

        # ---- 3. cached (raising source) vs fresh (BF16, cache=False) ---------------------------------------------
        ccl = MotifCCL(mesh_device, cfg)
        rope = MotifRope(mesh_device, cfg)
        cached = _Build(mesh_device, cfg, source=cw.RaisingSource("roundtrip"), cache=True, ccl=ccl, rope=rope)
        fresh = _Build(mesh_device, cfg, source=src, cache=False, ccl=ccl, rope=rope)
        log(f"built from the TT cache alone in {cached.seconds:.1f} s; from BF16 (cache=False) in {fresh.seconds:.1f} s")
        n_t = n_big = 0
        t0 = time.time()
        for name, obj in cached.objects().items():
            la, lb = _tensor_leaves(obj), _tensor_leaves(fresh.objects()[name])
            if set(la) != set(lb):
                failures.append(f"{name}: tensor attributes differ: {sorted(set(la) ^ set(lb))[:5]}")
            for path in sorted(set(la) & set(lb)):
                a, b = la[path], lb[path]
                if _spec(a) != _spec(b):
                    failures.append(f"{name}.{path}: spec {_spec(a)} != {_spec(b)}")
                    continue
                r = _compare_device_tensors(a, b)
                n_t += 1
                n_big += r["chips"] < 32
                if not r["ok"]:
                    failures.append(f"{name}.{path}: {r['why']}")
        log(f"weights: {n_t} device tensors compared bitwise ({n_big} on 4 sampled chips) in {time.time() - t0:.1f} s "
            f"(incl. the exact-fp32 router of L02 and 6 stock mHC sites)")

        # ---- 4. prefill chain + one decode step + the variant modules, bitwise and finite ------------------------
        S, P = 128, 128
        ids = _prompt_ids("en_technical")[:P]
        n_pt = cfg.prefill_page_table_entries(S)
        B = cfg.max_batch
        W = 4
        # lane 0 continues the prefilled user (blocks 1..W); lane l > 0 owns block W + l
        pt = torch.zeros(B, W, dtype=torch.int32)
        pt[0] = torch.arange(1, W + 1, dtype=torch.int32)
        for lane in range(1, B):
            pt[lane, 0] = W + lane
        n_blocks = W + B + 1
        pos = [P] + [(7 * lane) % 60 for lane in range(1, B)]
        pos[5] = -1  # one inactive lane
        dec_ids = torch.tensor(_prompt_ids("python_code")[:B], dtype=torch.int64)
        gen = torch.Generator().manual_seed(1234)
        x_router = torch.randn(1, 1, 32, cfg.hidden_size, generator=gen).to(torch.bfloat16)
        outs: Dict[str, Dict[str, Any]] = {}
        for tag, build in (("cached", cached), ("fresh", fresh)):
            o: Dict[str, Any] = {}
            kvs = [_alloc_kv(mesh_device, cfg, n_blocks) for _ in build.layers]
            pt_pf = _replicated(mesh_device, pt[0:1, :n_pt].contiguous(), ttnn.int32)
            tok = build.embed.prefill_tokens_device(torch.tensor(ids), S)
            X = build.embed.forward_prefill(tok)
            o["prefill.embed"] = X
            for layer, kv in zip(build.layers, kvs):
                X = layer.forward_prefill(X, page_table=pt_pf, kv_cache=kv)
                o[f"prefill.L{layer.layer_idx:02d}"] = X
            o["prefill.head"] = build.head.forward_prefill(X, P - 1)
            # decode step
            tok_d = build.embed.decode_tokens_device(dec_ids)
            # lane-ordered [32] / [32, W]: shard_lanes splits dim 0 over DP -> [8] / [8, W] per chip (README §2)
            cur = shard_lanes(torch.tensor(pos, dtype=torch.int32), cfg, mesh_device, dtype=ttnn.int32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
            pt_d = shard_lanes(pt, cfg, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
            rot_idx = rope.rot_idxs_device(torch.tensor(pos))
            rot = MotifAttention.decode_rope_tables(rope, rot_idx)
            act = MotifAttention.active_mask_from_cur_pos(cur, cfg.lanes_per_row)
            Xd = build.embed.forward_decode(tok_d)
            o["decode.embed"] = Xd
            for layer, kv in zip(build.layers, kvs):
                Xd = layer.forward_decode(Xd, rot=rot, cur_pos=cur, page_table=pt_d, kv_cache=kv, active=act)
                o[f"decode.L{layer.layer_idx:02d}"] = Xd
            o["decode.head"] = build.head.forward_decode(Xd, row_major=True)
            for layer, kv in zip(build.layers, kvs):
                o[f"kv.L{layer.layer_idx:02d}"] = kv
            # the variant modules from the cache: exact router logits, stock mHC pre / post on the decode streams
            xr = _replicated(mesh_device, x_router, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for l, router in build.router_fp32.items():
                o[f"router_fp32.L{l:02d}"] = router(xr)
            for (l, site), m in build.mhc_stock.items():
                x_red, coeffs = m.pre(o["decode.embed"])
                o[f"mhc_stock.L{l:02d}.{site}.pre"] = x_red
                o[f"mhc_stock.L{l:02d}.{site}.post"] = m.post(o["decode.embed"], x_red, coeffs)
            ttnn.synchronize_device(mesh_device)
            o["_inputs"] = [pt_pf, tok, tok_d, cur, pt_d, rot_idx, act, xr, [t for cs in rot.values() for t in cs]]
            outs[tag] = o
        n_same = 0
        keys = [k for k in outs["cached"] if not k.startswith("_")]
        nonfinite = {}
        for k in keys:
            same = _same_on_mesh(outs["cached"][k], outs["fresh"][k], mesh_device)
            n_same += same
            if not same:
                failures.append(f"{k}: cached and fresh outputs differ")
            nf = _nonfinite(outs["cached"][k], mesh_device)  # bitwise equal NaNs must not pass
            if nf:
                nonfinite[k] = nf
        if nonfinite:
            failures.append(f"non-finite values in the outputs: {nonfinite}")
        host_logits = cached.head.prefill_logits_to_host(outs["cached"]["prefill.head"], P - 1).float()
        log(f"forward: {n_same}/{len(keys)} outputs bitwise equal and {len(keys) - len(nonfinite)}/{len(keys)} finite "
            f"(prefill S={S} chain over embedding + L00-L02 + head, one 32-lane decode step, the 3 KV caches, the "
            f"exact-fp32 router logits, 6 stock mHC sites pre + post); prefill argmax {int(host_logits.argmax())}")
        for o in outs.values():
            _free([v for k, v in o.items()])
        cached.deallocate()
        fresh.deallocate()
        rope.release_prefill_tables()
    finally:
        if os.environ.get("MOTIF3_KEEP_TEST_CACHE") != "1":
            shutil.rmtree(TEST_ROOT, ignore_errors=True)
            log(f"removed {TEST_ROOT} ({_free_gb(REAL_ROOT):.1f} GB free)")
    assert not failures, "\n".join(failures[:40])


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_device_model_auto_loads_mock_cache(mesh_device):
    """The serving loader (``MotifModel(cache="auto")``) takes the converter's markers: layers 0-2 + globals of the
    real, mock-built cache load with a source that raises on any read, and a prompt prefills through them; the same
    with the variant choices of decisions D2 (stock Sinkhorn) and D1 (exact-fp32 router)."""
    from models.demos.motif3.tt.ccl import log_fabric

    log_fabric(mesh_device, "test_device_model_auto_loads_mock_cache")
    cw = _load_script("convert_weights")
    try:
        from models.demos.motif3.tt.model import MotifModel
    except ImportError as e:
        pytest.skip(f"tt/model.py not importable: {e}")
    cfg = cw.build_cfg(mesh_device, cache_root=str(REAL_ROOT), weights_dir=None)
    opts = cw.ConvertOptions()
    missing = [cw.part_tag(p) for p in PARTS if not cw.part_status(cfg, p, opts).complete]
    if missing:
        pytest.skip(f"real cache parts {missing} not converted (scripts/convert_weights.py --layers 0-2 --globals)")
    ids = _prompt_ids("chat_default")[:120]
    for what, kw in (("defaults", {}), ("stock sinkhorn + exact-fp32 router",
                                        {"sinkhorn": "stock", "router_logits": "exact_fp32"})):
        t0 = time.time()
        model = MotifModel(mesh_device, cfg, source=cw.RaisingSource(f"MotifModel auto ({what})"), layers=[0, 1, 2],
                           cache="auto", layer_kwargs=kw or None, log=log)
        log(f"MotifModel(layers 0-2, cache='auto', {what}) loaded from the mock-built cache in {time.time() - t0:.1f} s")
        try:
            tok = model.embed.prefill_tokens_device(torch.tensor(ids), 128)
            tile = model.prefill(tok, last_index=len(ids) - 1)
            logits = model.head.prefill_logits_to_host(tile, len(ids) - 1).float()
            _free(tok, tile)
            assert logits.shape[-1] == cfg.vocab_size and bool(torch.isfinite(logits).all()), what
            log(f"prefill through the cached 3-layer model ({what}): logits finite, argmax {int(logits.argmax())}")
        finally:
            model.deallocate()
