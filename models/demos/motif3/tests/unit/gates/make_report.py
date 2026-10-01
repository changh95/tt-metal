# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarise results/*.jsonl (latest record per case) as markdown tables. Host-only, no ttnn import.

    python models/demos/motif3/tests/unit/gates/make_report.py [G1 G3 ...] > /tmp/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SKIP = {"gate", "case", "ts", "per_user_pcc", "positions", "samples", "notes", "user_mean"}


def latest(gate: str) -> dict:
    rows = {}
    p = RESULTS / f"{gate}.jsonl"
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["case"]] = r
    return rows


def fmt(v):
    if isinstance(v, float):
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1000:
            return f"{v:.0f}"
        if a >= 1:
            return f"{v:.2f}"
        if a >= 1e-3:
            return f"{v:.5f}" if a < 0.1 else f"{v:.4f}"
        return f"{v:.2e}"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v[:12]) + (", …" if len(v) > 12 else "") + "]"
    if isinstance(v, str) and len(v) > 160:
        return v[:157] + "…"
    return str(v)


def table(gate: str) -> str:
    rows = latest(gate)
    if not rows:
        return f"_no results for {gate}_\n"
    cols = []
    for r in rows.values():
        for k in r:
            if k not in SKIP and k not in cols:
                cols.append(k)
    out = ["| case | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for case, r in rows.items():
        out.append(f"| {case} | " + " | ".join(fmt(r.get(c, "")) .replace("|", "/") for c in cols) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    gates = sys.argv[1:] or sorted(p.stem for p in RESULTS.glob("G*.jsonl"))
    for g in gates:
        print(f"### {g}\n")
        print(table(g))
