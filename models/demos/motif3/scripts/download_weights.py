#!/usr/bin/env python3
"""Layer-ordered, disk-aware downloader for the Motif-3 BF16 checkpoint.

The full checkpoint (629.7 GB, 155 shards) does not fit on this host's disk next to the TT weight cache,
so shards are fetched in layer order (globals, then layer 0, 1, 2, ...) and the download stops when free
space would drop below --margin-gb. Re-run later (after converted layers' shards were deleted) to continue;
already-complete files are skipped.

  python scripts/download_weights.py [--layers 0-52] [--margin-gb 60] [--workers 16]

State: <dest>/.download_state.json lists layers whose shards are all present (and sizes verified).
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

ROOT = "/home/ttuser/hchang/experiments/motif-3"
REPO = "Motif-Technologies/Motif-3"
REVISION = "2ed2ed5cfabffa10fdabb2fc0d0288f8e6de893a"
DEST = f"{ROOT}/weights/Motif-3"
TREE = f"{ROOT}/hf_meta/tree.json"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def free_gb(path="/"):
    # space available to this (non-root) user; shutil.disk_usage().free also counts ext4's root-reserved blocks
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def parse_layers(spec, n_layers):
    if not spec:
        return list(range(n_layers))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="", help="e.g. 0-52 or 3,4,10-12 (default: all)")
    ap.add_argument("--margin-gb", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--layers-per-batch", type=int, default=2)
    args = ap.parse_args()

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download

    index = json.load(open(f"{DEST}/model.safetensors.index.json"))["weight_map"]
    sizes = {e["path"]: e["size"] for e in json.load(open(TREE)) if e["type"] == "file"}

    by_layer = defaultdict(set)
    globals_ = set()
    for name, shard in index.items():
        m = re.match(r"model\.layers\.(\d+)\.", name)
        if m:
            by_layer[int(m.group(1))].add(shard)
        else:
            globals_.add(shard)  # embed, norm, lm_head, mtp
    n_layers = max(by_layer) + 1
    layers = parse_layers(args.layers, n_layers)

    state_path = f"{DEST}/.download_state.json"
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"complete_layers": []}

    def present(f):
        p = f"{DEST}/{f}"
        return os.path.exists(p) and os.path.getsize(p) == sizes[f]

    # ordered work items: globals first, then layers
    items = [("globals", sorted(globals_))] + [(f"layer{L}", sorted(by_layer[L])) for L in layers]
    i = 0
    while i < len(items):
        batch = items[i : i + (1 if items[i][0] == "globals" else args.layers_per_batch)]
        files = sorted({f for _, fs in batch for f in fs if not present(f)})
        need_gb = sum(sizes[f] for f in files) / 1e9
        fr = free_gb()
        names = ",".join(n for n, _ in batch)
        if files and fr - need_gb < args.margin_gb:
            log(f"STOP: {names} needs {need_gb:.1f} GB, free {fr:.1f} GB, margin {args.margin_gb} GB")
            break
        if files:
            log(f"downloading {names}: {len(files)} files, {need_gb:.1f} GB (free {fr:.1f} GB)")
            t0 = time.time()
            for attempt in range(3):
                try:
                    snapshot_download(
                        REPO,
                        revision=REVISION,
                        local_dir=DEST,
                        allow_patterns=files,
                        max_workers=args.workers,
                    )
                except Exception as e:  # network hiccups: retry
                    log(f"  attempt {attempt + 1} error: {e!r}")
                bad = [f for f in files if not present(f)]
                if not bad:
                    break
                log(f"  {len(bad)} files incomplete after attempt {attempt + 1}; retrying")
            else:
                log(f"FAILED {names}: incomplete {bad}")
                sys.exit(1)
            dt = time.time() - t0
            log(f"  done in {dt:.0f}s ({need_gb / max(dt, 1e-6) * 1000:.0f} MB/s)")
        for n, fs in batch:
            if n.startswith("layer") and all(present(f) for f in fs):
                L = int(n[5:])
                if L not in state["complete_layers"]:
                    state["complete_layers"].append(L)
        state["complete_layers"].sort()
        json.dump(state, open(state_path, "w"), indent=1)
        i += len(batch)
    log(f"complete layers: {state['complete_layers']}  free {free_gb():.1f} GB")


if __name__ == "__main__":
    main()
