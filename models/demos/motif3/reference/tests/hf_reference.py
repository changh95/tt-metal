# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the official HF ``modeling_motif.py`` on CPU (test-only helper).

The HF model hard-requires ``attn_implementation="flash_attention_2"`` (modeling_motif.py:1299-1308). flash-attn
is not installed here, so :func:`cpu_flash_attention_forward` re-implements the exact flash-attn semantics that
transformers 5.12.1 requests, and :func:`hf_cpu_flash_attention` temporarily registers it under
``"flash_attention_2"``:

* transformers 5.12.1 ``modeling_flash_attention_utils._process_flash_attention_kwargs`` (lines 642-647) maps
  ``sliding_window=sw`` to ``window_size=(sw - 1, sw - 1)`` ONLY when ``key_length > sw``, always with
  ``causal=True``; HF Motif passes ``sw = config.sliding_window + 1 = 129``.
* flash-attn causal masking is bottom-right aligned: query i of Sq sits at absolute key index ``i + Sk - Sq``.
  With causal, the right window is irrelevant, so key j is visible iff ``i' - (sw - 1) <= j <= i'``.
* GQA: q head h reads kv head ``h // (Hq / Hk)`` (== ``repeat_interleave``).
* Output layout is flash-attn's ``(B, S, H, D)``.

``scores = (q @ k^T) * scale``, softmax and ``P @ V`` run in fp32 and the output is cast to the query dtype.
(The real FA2 kernel additionally rounds P to bf16 before the PV MMA; the reference deliberately uses the exact
fp32 product, so this shim matches the reference rather than FA2's internal rounding.)

Hub kernels are blocked. At import time ``modeling_motif.py:35-46`` runs
``kernels.get_kernel("Motif-Technologies/activation")``. When ``kernels`` is importable, that call downloads from
the HF Hub and swaps ``RMSNorm`` / ``PolyNorm`` for CUDA-kernel classes. ``kernels`` is not installed here, but
transformers 5.12.1 imports it eagerly whenever it is installed (``modeling_utils.py:57`` ->
``integrations/hub_kernels.py:45-59``). So by the time ``modeling_motif`` runs, ``sys.modules["kernels"]`` can
already hold the real package, and a ``setdefault`` guard would be a no-op. :func:`load_hf_modules` therefore
forces ``import kernels`` to fail only while ``modeling_motif`` is imported, then restores the previous entry.
Afterwards it checks that the oracle really uses the torch ops (:func:`assert_hf_uses_torch_ops`).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import types
from pathlib import Path

import torch

HF_META_DIR = Path(os.environ.get("MOTIF3_HF_META_DIR", "/home/ttuser/hchang/experiments/motif-3/hf_meta"))
_PKG = "motif3_hf_reference_pkg"
# Imported by modeling_motif.py at module level and never wanted on CPU: ``kernels`` downloads hub kernels.
HF_BLOCKED_IMPORTS = ("kernels",)
_MISSING = object()


@contextlib.contextmanager
def blocked_imports(names=HF_BLOCKED_IMPORTS):
    """Inside the block ``import <name>`` raises ``ModuleNotFoundError`` (``sys.modules[name] = None``), even if the
    module is installed or already imported. On exit the previous ``sys.modules`` entries are restored."""
    saved = {n: sys.modules.get(n, _MISSING) for n in names}
    for n in names:
        sys.modules[n] = None
    try:
        yield
    finally:
        for n, mod in saved.items():
            if mod is _MISSING:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod


def assert_hf_uses_torch_ops(mdl_mod) -> None:
    """Fail loudly if ``modeling_motif`` picked up hub-kernel norms instead of its torch implementations."""
    picked = {
        "activation": mdl_mod.activation,
        "kernelRMSNorm": mdl_mod.kernelRMSNorm,
        "PolyNormKernel": mdl_mod.PolyNormKernel,
    }
    bad = {k: v for k, v in picked.items() if v is not None}
    if mdl_mod.PolyNorm is not mdl_mod.PolyNormTorch:
        bad["PolyNorm"] = mdl_mod.PolyNorm
    if mdl_mod.ACT2CLS["poly_norm"] is not mdl_mod.PolyNormTorch:
        bad["ACT2CLS['poly_norm']"] = mdl_mod.ACT2CLS["poly_norm"]
    if bad:
        raise RuntimeError(
            f"{mdl_mod.__name__} is not using its torch RMSNorm/PolyNorm (hub kernels leaked in: {bad}); "
            "the HF oracle must run pure torch ops on CPU"
        )


def load_hf_modules(hf_dir: Path = HF_META_DIR, pkg_name: str = _PKG):
    """Import ``configuration_motif`` / ``modeling_motif`` from ``hf_dir`` as the synthetic package ``pkg_name``.
    No files are copied and no bytecode is written next to the sources. Hub kernels are blocked during the import;
    see the module docstring."""
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(hf_dir)]
        sys.modules[pkg_name] = pkg
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        cfg_mod = importlib.import_module(f"{pkg_name}.configuration_motif")
        with blocked_imports(HF_BLOCKED_IMPORTS):
            mdl_mod = importlib.import_module(f"{pkg_name}.modeling_motif")
    finally:
        sys.dont_write_bytecode = prev
    assert_hf_uses_torch_ops(mdl_mod)
    return cfg_mod, mdl_mod


def cpu_flash_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask=None,
    dropout: float = 0.0,
    scaling=None,
    sliding_window=None,
    softcap=None,
    is_causal=None,
    **kwargs,
):
    """Drop-in for transformers' ``flash_attention_forward``: inputs ``(B, H, S, D)``, returns ``((B, S, H, D), None)``."""
    if attention_mask is not None:
        raise NotImplementedError("padding masks are not used by these tests")
    if softcap is not None or dropout:
        raise NotImplementedError("softcap/dropout not supported")
    B, Hq, Sq, Dq = query.shape
    Hk, Sk = key.shape[1], key.shape[2]
    rep = Hq // Hk
    k = key.repeat_interleave(rep, dim=1)
    v = value.repeat_interleave(rep, dim=1)
    causal = module.is_causal if is_causal is None else is_causal
    scale = Dq**-0.5 if scaling is None else scaling
    qi = torch.arange(Sq)[:, None] + (Sk - Sq)  # bottom-right aligned absolute query index
    kj = torch.arange(Sk)[None, :]
    allow = torch.ones(Sq, Sk, dtype=torch.bool)
    if causal:
        allow &= kj <= qi
    if sliding_window is not None and Sk > sliding_window:  # tf 5.12.1: window_size=(sw - 1, sw - 1)
        left = right = sliding_window - 1
        allow &= kj >= qi - left
        if not causal:
            allow &= kj <= qi + right
    scores = (query.float() @ k.float().transpose(-1, -2)) * scale
    scores = scores.masked_fill(~allow, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out = (p @ v.float()).to(query.dtype)
    return out.transpose(1, 2).contiguous(), None


@contextlib.contextmanager
def hf_cpu_flash_attention():
    """Temporarily route ``"flash_attention_2"`` to the CPU shim and skip the flash-attn availability check."""
    from transformers import modeling_utils

    mapping = modeling_utils.AttentionInterface._global_mapping
    prev_fn = mapping.get("flash_attention_2")
    prev_check = modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation
    mapping["flash_attention_2"] = cpu_flash_attention_forward
    modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation = lambda self, *a, **k: "flash_attention_2"
    try:
        yield
    finally:
        mapping["flash_attention_2"] = prev_fn
        modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation = prev_check


def hf_config_from_args(args, cfg_mod=None):
    """HF ``MotifConfig`` with exactly the semantics of reference ``MotifArgs``."""
    cfg_mod = cfg_mod or load_hf_modules()[0]
    cfg = cfg_mod.MotifConfig(**args.to_hf_config_dict())
    cfg._attn_implementation = "flash_attention_2"
    return cfg


def hf_config_from_json(path, cfg_mod=None):
    cfg_mod = cfg_mod or load_hf_modules()[0]
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    cfg = cfg_mod.MotifConfig(**json.loads(path.read_text()))
    cfg._attn_implementation = "flash_attention_2"
    return cfg
