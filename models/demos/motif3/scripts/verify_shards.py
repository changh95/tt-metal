#!/usr/bin/env python3
"""Verify downloaded Motif-3 shards against the HF LFS sha256 oids (hf_meta/tree.json).

  python scripts/verify_shards.py [--workers 12]

Writes weights/Motif-3/.verified.json {filename: "ok" | "BAD" | "missing"} and prints a summary.
Already-verified files (same size + mtime) are skipped on re-runs.
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

ROOT = "/home/ttuser/hchang/experiments/motif-3"
DEST = f"{ROOT}/weights/Motif-3"


def sha256_file(path, bufsize=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check(args):
    name, oid, size = args
    p = f"{DEST}/{name}"
    if not os.path.exists(p):
        return name, "missing", None
    st = os.stat(p)
    if st.st_size != size:
        return name, "BAD", None
    return name, ("ok" if sha256_file(p) == oid else "BAD"), [st.st_size, st.st_mtime]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    tree = json.load(open(f"{ROOT}/hf_meta/tree.json"))
    want = {e["path"]: (e["lfs"]["oid"], e["size"]) for e in tree if e["type"] == "file" and "lfs" in e}
    state_path = f"{DEST}/.verified.json"
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    todo = []
    for name, (oid, size) in sorted(want.items()):
        p = f"{DEST}/{name}"
        prev = state.get(name)
        if isinstance(prev, dict) and prev.get("status") == "ok" and os.path.exists(p):
            st = os.stat(p)
            if [st.st_size, st.st_mtime] == prev.get("stat"):
                continue
        if os.path.exists(p):
            todo.append((name, oid, size))
    with ProcessPoolExecutor(args.workers) as ex:
        for name, status, stat in ex.map(check, todo):
            state[name] = {"status": status, "stat": stat}
            if status != "ok":
                print(f"{status}: {name}", flush=True)
    json.dump(state, open(state_path, "w"), indent=1)
    ok = sum(1 for v in state.values() if isinstance(v, dict) and v["status"] == "ok")
    bad = [k for k, v in state.items() if isinstance(v, dict) and v["status"] == "BAD"]
    print(f"verified ok: {ok} files; bad: {bad}")


if __name__ == "__main__":
    main()
