# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Import rule (design §2.1): the shared infra modules import no other ``models/demos/**`` package and open no
device at import time. Checked in a fresh interpreter so other tests' imports do not leak in.

Run device-hidden with ``--noconftest`` (see test_infra_config.py docstring).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TT_METAL = Path(__file__).resolve().parents[5]
MODULES = [
    "models.demos.motif3",
    "models.demos.motif3.tt",
    "models.demos.motif3.tt.model_config",
    "models.demos.motif3.tt.ccl",
    "models.demos.motif3.tt.weights",
    "models.demos.motif3.tt.rope",
]

PROBE = r"""
import importlib, json, sys
mods = json.loads(sys.argv[1])
for m in mods:
    importlib.import_module(m)
demos = sorted(m for m in sys.modules if m.startswith("models.demos.") and not m.startswith("models.demos.motif3"))
other_models = sorted(m for m in sys.modules if m.startswith("models.") and not m.startswith("models.demos"))
print(json.dumps({"demos": demos, "other_models": other_models}))
"""


def test_infra_imports_are_self_contained():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TT_METAL) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(MODULES)],
        cwd=str(TT_METAL),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, res.stderr[-4000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["demos"] == [], f"infra modules pulled in other demo packages: {out['demos']}"
    assert out["other_models"] == [], f"unexpected models.* imports: {out['other_models']}"
    # no device bring-up messages from UMD / metal during import
    for marker in ("Opening user mode device driver", "Starting devices in cluster"):
        assert marker not in res.stderr and marker not in res.stdout
