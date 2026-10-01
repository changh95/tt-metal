# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hygiene checks for the test harness itself. They test no model semantics:

* The HF oracle (``hf_reference.load_hf_modules``) never lets ``modeling_motif.py`` run its import-time
  ``kernels.get_kernel("Motif-Technologies/activation")``, so nothing is downloaded and the torch ``RMSNorm`` /
  ``PolyNorm`` are always used. This holds even when a ``kernels`` package is importable or already imported;
  transformers imports it eagerly once it is installed.
* The pytest commands in ``README.md`` keep tt-metal's conftest out and write no junit report. The root
  ``pytest.ini`` addopts would otherwise write the shared ``generated/test_reports/most_recent_tests.xml``.
"""

from __future__ import annotations

import configparser
import importlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from models.demos.motif3.reference import tiny_random_args

from . import hf_reference
from .hf_reference import HF_META_DIR, hf_config_from_args, hf_cpu_flash_attention, load_hf_modules

REF_DIR = Path(__file__).resolve().parents[1]
TT_METAL_ROOT = REF_DIR.parents[3]  # reference -> motif3 -> demos -> models -> tt-metal
HUB_KERNEL_REPO = "Motif-Technologies/activation"


# ---- HF oracle: torch ops only, no hub kernels ----------------------------------------------------------------


def test_hf_oracle_runs_torch_norms():
    """Every submodule of a tiny HF model (all layer kinds) is an HF-file class or a torch.nn built-in. The norms are
    the torch ``MotifRMSNorm`` / ``PolyNormTorch`` / ``GroupedPolyNorm``, so no hub-kernel class can hide in it."""
    cfg_mod, mm = load_hf_modules()
    hf_reference.assert_hf_uses_torch_ops(mm)
    with hf_cpu_flash_attention():
        hf = mm.MotifForCausalLM(hf_config_from_args(tiny_random_args(), cfg_mod))
    kinds = {type(m) for m in hf.modules()}
    foreign = sorted(
        f"{k.__module__}.{k.__qualname__}"
        for k in kinds
        if k.__module__ != mm.__name__ and not k.__module__.startswith("torch.nn.")
    )
    assert not foreign, f"non-HF-file module classes in the oracle: {foreign}"
    assert {mm.MotifRMSNorm, mm.PolyNormTorch, mm.GroupedPolyNorm} <= kinds


def test_blocked_imports_restores_sys_modules(monkeypatch):
    absent, present = "motif3_probe_absent_mod", "motif3_probe_present_mod"
    monkeypatch.delitem(sys.modules, absent, raising=False)
    sentinel = types.ModuleType(present)
    monkeypatch.setitem(sys.modules, present, sentinel)
    with hf_reference.blocked_imports((absent, present)):
        for name in (absent, present):
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(name)
    assert absent not in sys.modules  # not left behind as a None entry that would break a later real import
    assert sys.modules[present] is sentinel


def _drop_package(name: str) -> None:
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]


def test_hf_import_never_loads_hub_kernels(monkeypatch):
    """A fake ``kernels`` package sits in ``sys.modules``, which is where transformers leaves the real one once it is
    installed. The unguarded control import shows that the probe is live: the HF file calls ``get_kernel`` and swaps
    in the hub classes. The harness loader must do neither and must leave ``sys.modules["kernels"]`` as it was."""
    calls = []

    class HubRMSNorm(torch.nn.Module):
        pass

    class HubPolyNorm(torch.nn.Module):
        pass

    def get_kernel(repo_id, *args, **kwargs):
        calls.append(repo_id)
        return types.SimpleNamespace(layers=types.SimpleNamespace(RMSNorm=HubRMSNorm, PolyNorm=HubPolyNorm))

    fake = types.ModuleType("kernels")
    fake.get_kernel = get_kernel
    monkeypatch.setitem(sys.modules, "kernels", fake)

    control, guarded = "motif3_hf_probe_unguarded", "motif3_hf_probe_guarded"
    prev_bytecode = sys.dont_write_bytecode
    try:
        # control: a plain import of the HF file
        pkg = types.ModuleType(control)
        pkg.__path__ = [str(HF_META_DIR)]
        sys.modules[control] = pkg
        sys.dont_write_bytecode = True
        mm_raw = importlib.import_module(f"{control}.modeling_motif")
        sys.dont_write_bytecode = prev_bytecode
        assert calls == [HUB_KERNEL_REPO]
        assert mm_raw.PolyNorm is HubPolyNorm and mm_raw.kernelRMSNorm is HubRMSNorm
        with pytest.raises(RuntimeError, match="hub kernels leaked"):
            hf_reference.assert_hf_uses_torch_ops(mm_raw)

        # the harness loader
        calls.clear()
        _, mm = load_hf_modules(pkg_name=guarded)
        assert calls == [], "modeling_motif.py reached kernels.get_kernel through the harness loader"
        assert mm.activation is None and mm.kernelRMSNorm is None and mm.PolyNormKernel is None
        assert mm.PolyNorm is mm.PolyNormTorch and mm.ACT2CLS["poly_norm"] is mm.PolyNormTorch
        assert sys.modules["kernels"] is fake
    finally:
        sys.dont_write_bytecode = prev_bytecode
        _drop_package(control)
        _drop_package(guarded)


# ---- README test commands --------------------------------------------------------------------------------------

_OPTS_WITH_VALUE = {"-p", "-o", "--override-ini", "-k", "-m", "-c", "--rootdir", "--junitxml", "--junit-xml"}


def _readme_pytest_commands():
    text = (REF_DIR / "README.md").read_text()
    cmds = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.S):
        for line in re.sub(r"\\\n\s*", " ", block).splitlines():
            line = line.strip()
            if line and not line.startswith("#") and re.search(r"\bpytest\b", line):
                cmds.append(line)
    return cmds


def _split_pytest_args(cmd: str):
    """``python -m pytest <args>`` -> (options as a flat list, positional paths)."""
    tokens = shlex.split(cmd)
    it = iter(tokens[tokens.index("pytest") + 1 :])
    opts, paths = [], []
    for tok in it:
        if tok in _OPTS_WITH_VALUE:
            opts += [tok, next(it)]
        elif tok.startswith("-"):
            opts.append(tok)
        else:
            paths.append(tok)
    return opts, paths


def _opt_values(opts, *names):
    return [opts[i + 1] for i, tok in enumerate(opts[:-1]) if tok in names]


def _without_addopts_override(opts):
    for i, tok in enumerate(opts[:-1]):
        if tok in ("-o", "--override-ini") and opts[i + 1].startswith("addopts="):
            return opts[:i] + opts[i + 2 :]
    raise AssertionError(f"no addopts override in {opts}")


def _root_addopts() -> str:
    ini = configparser.ConfigParser(interpolation=None)
    ini.read(TT_METAL_ROOT / "pytest.ini")
    return ini.get("pytest", "addopts", fallback="")


def test_readme_pytest_commands_are_isolated():
    cmds = _readme_pytest_commands()
    assert len(cmds) >= 2, cmds  # full suite + fast subset
    for cmd in cmds:
        opts, paths = _split_pytest_args(cmd)
        assert "--noconftest" in opts, cmd  # tt-metal's root conftest imports ttnn
        assert "no:cacheprovider" in _opt_values(opts, "-p"), cmd
        overrides = [v.split("=", 1)[1] for v in _opt_values(opts, "-o", "--override-ini") if v.startswith("addopts=")]
        assert overrides, f"README command keeps the root pytest.ini addopts ({_root_addopts()!r}): {cmd}"
        assert "--junitxml" not in overrides[-1] and "--junit-xml" not in overrides[-1], cmd
        assert not _opt_values(opts, "--junitxml", "--junit-xml"), cmd
        assert "--import-mode=importlib" in opts, cmd  # what the root addopts set and the suite is validated with
        for p in paths + [o.split("=", 1)[1] for o in opts if o.startswith("--ignore=")]:
            assert (TT_METAL_ROOT / p).exists(), f"README path does not exist: {p}"


def test_readme_pytest_flags_write_no_junit(tmp_path):
    """Run the README flags in a scratch rootdir that holds a copy of the real root ``pytest.ini``. They must write no
    report. As a control, the same flags without the addopts override do write one, so the check is not vacuous.
    The shared report under the tt-metal checkout is never touched."""
    root_ini = TT_METAL_ROOT / "pytest.ini"
    if not root_ini.exists():
        pytest.skip("no root pytest.ini")
    shutil.copy(root_ini, tmp_path / "pytest.ini")
    (tmp_path / "test_probe.py").write_text("def test_probe():\n    assert True\n")
    opts, _ = _split_pytest_args(_readme_pytest_commands()[0])
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}

    def run(flags):
        res = subprocess.run(
            [sys.executable, "-m", "pytest", *flags, "test_probe.py"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
        return sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.xml"))

    assert run(opts) == [], "README pytest flags write a junit report"
    junit = re.search(r"--junit-?xml[= ](\S+)", _root_addopts())
    if junit is None:
        pytest.skip("root pytest.ini addopts write no junit report; nothing to override")
    control = _without_addopts_override(opts)
    assert run(control) == [os.path.normpath(junit.group(1))]
