# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Import rule (design §2.1, README §13): the shared infra and every wave-B module import no other ``models/demos/**``
package, none of vLLM / transformers / huggingface_hub / safetensors, and open no device at import time. Checked in a
fresh interpreter so other tests' imports do not leak in.

``MODULES`` must import; ``OPTIONAL_MODULES`` (the integration wave's decoder / model / generator) are checked as soon
as their file exists, so a new module is covered without editing this list.

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
    # shared infra
    "models.demos.motif3",
    "models.demos.motif3.tt",
    "models.demos.motif3.tt.generator_api",
    "models.demos.motif3.tt.generator_vllm",
    "models.demos.motif3.tt.model_config",
    "models.demos.motif3.tt.ccl",
    "models.demos.motif3.tt.weights",
    "models.demos.motif3.tt.rope",
    # wave-B1 modules (their own tests run the same probe: test_*::test_*import*)
    "models.demos.motif3.tt.mhc",
    "models.demos.motif3.tt.attention",
    "models.demos.motif3.tt.polynorm",
    "models.demos.motif3.tt.mlp",
    "models.demos.motif3.tt.moe",
    "models.demos.motif3.tt.embedding",
    "models.demos.motif3.tt.lm_head",
    # out-of-tree generic_op kernels
    "models.demos.motif3.tt.kernels",
    "models.demos.motif3.tt.kernels.sinkhorn_motif",
    "models.demos.motif3.tt.kernels.router_fp32",
]
# Integration wave (WAVE_A_REVIEW GEN-1..7): checked once the file exists.
OPTIONAL_MODULES = [
    "models.demos.motif3.tt.decoder",
    "models.demos.motif3.tt.model",
    "models.demos.motif3.tt.generator",
]
MOTIF3 = TT_METAL / "models" / "demos" / "motif3"


def _present(module: str) -> bool:
    rel = module.split(".")[3:]  # after models.demos.motif3
    base = MOTIF3.joinpath(*rel)
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()

PROBE = r"""
import importlib, json, sys
mods = json.loads(sys.argv[1])
for m in mods:
    importlib.import_module(m)
demos = sorted(m for m in sys.modules if m.startswith("models.demos.") and not m.startswith("models.demos.motif3"))
other_models = sorted(m for m in sys.modules if m.startswith("models.") and not m.startswith("models.demos"))
heavy = sorted(m for m in ("vllm", "transformers", "huggingface_hub", "safetensors") if m in sys.modules)
print(json.dumps({"demos": demos, "other_models": other_models, "heavy": heavy}))
"""


def test_module_lists_name_existing_files():
    missing = [m for m in MODULES if not _present(m)]
    assert not missing, f"MODULES lists modules that do not exist: {missing}"


def test_infra_imports_are_self_contained():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TT_METAL) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    mods = MODULES + [m for m in OPTIONAL_MODULES if _present(m)]
    print(f"[import rule] checking {len(mods)} modules: {mods}")
    res = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(mods)],
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
    # vLLM / transformers / huggingface_hub / safetensors are imported lazily, inside the functions that need them
    assert out["heavy"] == [], f"shared infra imported heavy packages at module import time: {out['heavy']}"
    # no device bring-up messages from UMD / metal during import
    for marker in ("Opening user mode device driver", "Starting devices in cluster"):
        assert marker not in res.stderr and marker not in res.stdout
