# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Render the jsonl of tests/test_multi_user_regression.py as Markdown.

Default input: every ``generated/solar_open_multi_user_regression/*.jsonl`` of the repository; pass file paths to
render others (the GPT-OSS sweep's ``generated/gpt_oss_multi_user_regression/*.jsonl`` share the schema). The latest
row per (batch, ISL, OSL) wins, so re-running one batch after a fix replaces its cells.

    python models/demos/solar_open/tests/sweep/report.py                       # per-batch tables, all files
    python models/demos/solar_open/tests/sweep/report.py --matrix              # + batch x ISL/OSL matrices
    python models/demos/solar_open/tests/sweep/report.py --matrix --out REPORT.md  a.jsonl b.jsonl

Columns: enc = encoded prompt length (tokens); TTFT first / mean / last = per-user prefill time x 1 / (B+1)/2 / B
(sequential per-user prefill on the 1x8 mesh, the first token is the argmax of each user's prefill logits); step ms /
p99 = decode step mean / p99 without the first (trace-launch) step; QA = keyword accuracy over the whole generation
(ISL 128 only); 1st-tok = users whose first token was not <|think|> / <|content|> (GPT-OSS rows: not <|channel|>);
degen = users flagged by the degeneracy heuristic; content = users that reached <|content|> within OSL (Solar rows).
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

# tests/sweep -> tests -> solar_open -> demos -> models -> repository root
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_ROOT = REPO_ROOT / "generated" / "solar_open_multi_user_regression"

TABLE_COLUMNS = [
    ("batch", "B", "{}"),
    ("isl_nominal", "ISL", "{}"),
    ("osl", "OSL", "{}"),
    ("isl_encoded", "enc", "{}"),
    ("ttft_first_user_ms", "TTFT first ms", "{:.0f}"),
    ("ttft_mean_user_ms", "TTFT mean ms", "{:.0f}"),
    ("ttft_last_user_ms", "TTFT last ms", "{:.0f}"),
    ("decode_step_mean_ms", "step ms", "{:.2f}"),
    ("decode_step_p99_ms", "p99 ms", "{:.2f}"),
    ("tok_s_user", "tok/s/user", "{:.1f}"),
    ("tok_s_aggregate", "tok/s agg", "{:.0f}"),
    ("e2e_s", "e2e s", "{:.1f}"),
    ("qa_accuracy", "QA", "{:.2f}"),
    ("_first_token_failures", "1st-tok fail", "{}"),
    ("_degenerate", "degen", "{}"),
    ("content_reached", "content", "{}"),
    ("status", "status", "{}"),
]
MATRICES = [
    ("decode_step_mean_ms", "decode step ms (mean, steady state)", "{:.1f}"),
    ("ttft_mean_user_ms", "TTFT mean over users (ms)", "{:.0f}"),
    ("ttft_last_user_ms", "TTFT last user (ms, whole batch admitted)", "{:.0f}"),
    ("tok_s_aggregate", "tok/s aggregate", "{:.0f}"),
    ("status", "status", "{}"),
]


def _derive(r):
    """Fill the derived columns: TTFT from the prefill timing (older rows), per-model first-token failure lists."""
    b, per_user = r["batch"], r["prefill_per_user_ms"]
    r.setdefault("ttft_first_user_ms", round(per_user, 1))
    r.setdefault("ttft_mean_user_ms", round(per_user * (b + 1) / 2, 1))
    r.setdefault("ttft_last_user_ms", round(r["prefill_total_s"] * 1000, 1))
    bad_first = r.get("users_bad_first_token")
    if bad_first is None:
        bad_first = r.get("users_not_starting_with_channel")
    r["_first_token_failures"] = len(bad_first or [])
    r["_degenerate"] = len(r.get("degenerate_users") or [])
    return r


def load_rows(path):
    """Latest row per (batch, ISL, OSL), keyed and sorted."""
    rows = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "isl_padded" not in r or "finished_users" not in r:
                continue  # rows from a pre-gate harness revision
            rows[(r["batch"], r["isl_nominal"], r["osl"])] = _derive(r)
    return OrderedDict(sorted(rows.items()))


def _fmt(value, fmt):
    if value is None or value == "":
        return ""
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def _header(name, rows):
    """One-paragraph provenance line: model, mesh, git revisions, tags, time span, context per batch, thermal data."""
    if not rows:
        return f"### {name} (0 cases)\n"
    vals = list(rows.values())
    first, last = min(r["time"] for r in vals), max(r["time"] for r in vals)
    gits = sorted({r.get("git", "?") for r in vals})
    tags = sorted({r.get("tag", "") for r in vals} - {""})
    contexts = sorted({(r["batch"], r.get("context_per_user")) for r in vals})
    temps = [r["board_temp_after_prefill_c"] for r in vals if r.get("board_temp_after_prefill_c") is not None]
    clocks = [r["min_aiclk_after_prefill_mhz"] for r in vals if r.get("min_aiclk_after_prefill_mhz") is not None]
    lines = [
        f"### {name} ({len(rows)} cases)",
        "",
        f"{vals[0].get('model')} on {vals[0].get('mesh')}, git {', '.join(gits)}"
        + (f", tag {', '.join(tags)}" if tags else "")
        + f", runs {first[:16]} .. {last[:16]}; context per user by batch: "
        + ", ".join(f"B{b}={c}" for b, c in contexts)
        + (f"; board temperature after prefill {min(temps):.0f}-{max(temps):.0f} C" if temps else "")
        + (f", min AI clock {min(clocks)} MHz" if clocks else "")
        + (f"; cooldown gate {vals[0]['cooldown_c']} C" if vals[0].get("cooldown_c") else "")
        + ".",
        "",
    ]
    return "\n".join(lines)


def per_batch_tables(rows):
    """One Markdown table per batch size, then the list of failing cells."""
    out = []
    batches = sorted({b for (b, _, _) in rows})
    for batch in batches:
        cases = [r for (b, _, _), r in rows.items() if b == batch]
        out.append(f"#### batch {batch} ({len(cases)} cases)\n")
        out.append("| " + " | ".join(h for _, h, _ in TABLE_COLUMNS) + " |")
        out.append("|" + "---|" * len(TABLE_COLUMNS))
        for r in cases:
            out.append("| " + " | ".join(_fmt(r.get(k), f) for k, _, f in TABLE_COLUMNS) + " |")
        out.append("")
    bad = [(k, r) for k, r in rows.items() if r.get("status") != "ok"]
    if bad:
        out.append("Failures:\n")
        for (b, isl, osl), r in bad:
            bad_first = r.get("users_bad_first_token", r.get("users_not_starting_with_channel"))
            out.append(
                f"- B{b} ISL {isl} OSL {osl}: first-token failures {bad_first}, degenerate {r.get('degenerate_users')}, "
                f"QA {r.get('qa_accuracy')} (misses {r.get('qa_misses')}), sample {r.get('sample_output', '')[:100]!r}"
            )
        out.append("")
    return "\n".join(out)


def matrix(rows, key, title, fmt):
    """batch x (ISL, OSL) matrix of one metric; '-' = not run (ISL + OSL above the batch's context budget, or the
    batch has not been swept), **FAIL** prefix = the cell's checks failed."""
    pairs = sorted({(i, o) for (_, i, o) in rows})
    batches = sorted({b for (b, _, _) in rows})
    out = [
        f"#### {title}\n",
        "| B \\ ISL/OSL | " + " | ".join(f"{i}/{o}" for i, o in pairs) + " |",
        "|---|" + "---|" * len(pairs),
    ]
    for b in batches:
        cells = []
        for i, o in pairs:
            r = rows.get((b, i, o))
            if r is None:
                cells.append("-")
            else:
                prefix = "**FAIL** " if r.get("status") != "ok" and key != "status" else ""
                cells.append(prefix + _fmt(r.get(key), fmt))
        out.append(f"| {b} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def render(files, with_matrix):
    parts = []
    for f in files:
        rows = load_rows(f)
        parts.append(_header(Path(f).name, rows))
        if not rows:
            continue
        parts.append(per_batch_tables(rows))
        if with_matrix:
            for key, title, fmt in MATRICES:
                parts.append(matrix(rows, key, f"{Path(f).name}: {title}", fmt))
    return "\n".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help=f"jsonl files (default: {DEFAULT_ROOT}/*.jsonl)")
    parser.add_argument("--matrix", action="store_true", help="also print batch x ISL/OSL matrices")
    parser.add_argument("--out", help="write the Markdown to this file as well as stdout")
    args = parser.parse_args(argv)
    files = [Path(a) for a in args.files] or sorted(DEFAULT_ROOT.glob("*.jsonl"))
    if not files:
        print(f"no jsonl files given and none under {DEFAULT_ROOT}", file=sys.stderr)
        return 1
    text = render(files, args.matrix)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
