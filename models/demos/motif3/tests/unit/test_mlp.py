# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PolyNorm (tt/polynorm.py) and dense MLP / shared expert (tt/mlp.py) tests (WAVE_A_REVIEW §5.5 MLP-1..3).

Host-only parts (run device-hidden, no root conftest)::

    S=/home/ttuser/hchang/experiments/motif-3/scripts
    # real-activation goldens (CPU reference prefix model, written once to tt_cache/test/mlp): layers 0-2 (~1 min,
    # 119 MB) and layer 4 (layers 0-4 with the routed experts of 2-3, ~10 s with warm page cache, 35 MB)
    $S/hostrun.sh -n mlp_goldens -- python models/demos/motif3/tests/unit/test_mlp.py --make-goldens
    $S/hostrun.sh -n mlp_goldens_l4 -- python models/demos/motif3/tests/unit/test_mlp.py --make-goldens-l4
    # CPU tests (Horner identity, TP-shard emulation vs the reference at real dims, constants and their EP layout,
    # config builders and their compute-grid fallback, PolyNorm config semantics, NaN-safe metrics, device-free import)
    $S/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib -q \
        models/demos/motif3/tests/unit/test_mlp.py -k "not device"

Device tests (only through the lock wrapper)::

    $S/devrun.sh -t 1500 -n mlp -- python -m pytest models/demos/motif3/tests/unit/test_mlp.py -p no:cacheprovider \
        -o addopts="" --import-mode=importlib -s -rA -k "polynorm_device or random_weights or real_weights"
    # opt-in benchmarks: MOTIF3_MLP_VARIANTS=1 (-k variants; MOTIF3_MLP_VARIANTS_ONLY=tp,grouped,matmul,module),
    # MOTIF3_MLP_SWEEP=1 (-k sweep, first sweep), MOTIF3_MLP_PROFILE=1 (-k profile, op-level device kernel times; run
    # with TT_METAL_DEVICE_PROFILER=1 and TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1, or under tracy -- the latter
    # writes ~17 GB of logs for 32 chips)

Device tests:

* ``test_polynorm_device`` -- scalar TP PolyNorm (dense 8 x 1536, shared 8 x 160; distinct data on every chip; real
  layer-0/1/2 coefficients) and grouped PolyNorm (``[1, 12, 32, 1280]`` per chip with each chip's real layer-2
  expert coefficients, incl. the bias clamp) vs an fp64 golden; fp32 / bf16 inputs and modes; L1 / DRAM; traced
  latency; every advertised ``moments`` / ``ar`` / ``horner`` knob (incl. ``moments="pre_ag"`` with both
  all-reduces); the grouped constants rebuilt from the TT cache alone (``from_source`` with a source that raises).
* ``test_mlp_device_random_weights`` -- dense (layer 0) and shared (layer 2) PolyNormMLP with random weights at real
  dims vs the reference ``modules.MLP`` (fp32): decode (each DP row its own 8 lanes; the input's logical rows
  unchanged), ``all_reduce=False`` partials, prefill S = 128; the chunked prefill path (S = 1024 in chunks of 384
  vs unchunked and the reference, chunked partials, S = 16384 in 2 default chunks); bfp8 and bf16 weights; the
  shared ``stats="replicated_gate"`` option; both kinds rebuilt from the TT cache alone (bitwise outputs).
* ``test_mlp_device_real_weights`` -- layers 0 / 1 (dense), 2 and 4 (shared expert; 4 = global + MoE, a PolyNorm
  outlier layer) with real weights on real activations (``post_attention_layernorm.out`` of the reference prefix
  model): decode vs fp32 and HF-bf16 references, PolyNorm in isolation on the TT gate / up, the bf16-PolyNorm decode
  lever (informational, MLP-1), eager + traced latency, trace replay with new inputs, prefill S = 128 / 1024 (fp32
  and bf16 PolyNorm, exact vs TF32-class moments all-reduce), bf16 weights (fp32-faithful), the replicated-gate
  shared expert (layer 2).
* ``test_mlp_device_fused_shared_polynorm`` -- B5: the fused decode shared-expert PolyNorm (``shared_polynorm="fused"``)
  == the composite bitwise on every MoE layer's shared expert (TT cache, real MoE inputs, T32 lanes and T64-shaped
  rows, with and without the TP all-reduce); determinism, T64 rows, trace replay; traced cost.
* ``test_mlp_device_variants`` / ``test_mlp_device_sweep`` / ``test_mlp_device_profile`` (opt-in) -- the measurements
  behind the defaults in tt/mlp.py and tt/polynorm.py.

Every device test logs the committed fabric (``[motif3.fabric] ... committed:``) and reports PCC / max-abs per case.
Metrics: ``stats`` (``comp_pcc`` + an explicit finiteness check) and the NaN-safe ``passes`` / ``worst_of``: a NaN
or Inf output fails every check.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.motif3.tt import weights as W
from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, log_fabric, replicas_identical
from models.demos.motif3.tt.mlp import DECODE_MATMUL_GRIDS, PolyNormMLP, build_decode_program_configs, decode_matmul_pc
from models.demos.motif3.tt.model_config import DEFAULT_HF_META_DIR, PROJECT_ROOT, TILE, MotifTTConfig, device_params
from models.demos.motif3.tt.polynorm import (
    GroupedPolyNormConsts,
    PolyNormCoefficients,
    ScalarPolyNormConsts,
    check_polynorm_semantics,
    grouped_polynorm,
    horner_scale_constants,
    polynorm_output_scale,
    polynorm_tp,
)

HF_META = str(DEFAULT_HF_META_DIR)
GOLDEN_DIR = PROJECT_ROOT / "tt_cache" / "test" / "mlp"
REAL_GOLDEN = GOLDEN_DIR / "real_ffn_goldens_v1.pt"  # layers 0, 1, 2
REAL_GOLDEN_L4 = GOLDEN_DIR / "real_ffn_goldens_L4_v1.pt"  # layer 4 (global + MoE; a PolyNorm outlier layer)
TT_METAL = Path(__file__).resolve().parents[5]
MESH = [pytest.param((4, 8), device_params(), id="4x8-torus2d")]
EPS = 1e-6
RESULTS = []  # (test, case, metrics) lines printed at the end of each test

# Real-golden token layout: prefill rows [0, 1024) (S = 128 uses the first 128), decode lanes = 32 rows after them.
N_PREFILL_TOKENS = 1024
N_DECODE_TOKENS = 32
N_GOLDEN_TOKENS = N_PREFILL_TOKENS + N_DECODE_TOKENS
GOLDEN_PROMPTS = ("chat_default", "en_technical", "ko_passage")


# ============================================================================================================
# torch goldens / metrics
# ============================================================================================================
def poly_golden(g, u, c0, c1, c2, b, eps=EPS):
    """fp64 PolyNorm * up (HF PolyNormTorch / GroupedPolyNorm without the x0.5); c_k, b scalars or broadcastable."""
    g = g.double()
    u = u.double()

    def N(z):
        return z / torch.sqrt(z.pow(2).mean(-1, keepdim=True) + eps)

    return (c0 * N(g**3) + c1 * N(g**2) + c2 * N(g) + b) * u


def stats(ref, got) -> dict:
    """PCC (``models.common.utility_functions.comp_pcc``, README §12), max-abs, |ref| max and relative Frobenius error.

    Non-finite values anywhere give ``pcc=nan``, ``max_abs=inf`` and ``nonfinite`` = their count: ``comp_pcc`` alone
    zeroes NaN / Inf and can still pass (one NaN in 100 values -> 0.990), so finiteness is checked here and every
    acceptance check goes through :func:`passes` (NaN-safe) and :func:`worst_of` (non-finite chips rank worst)."""
    from models.common.utility_functions import comp_pcc

    ref = ref.detach().double()
    got = got.detach().double().reshape(ref.shape)
    bad = int((~torch.isfinite(got)).sum()) + int((~torch.isfinite(ref)).sum())
    fin = ref[torch.isfinite(ref)]
    ref_absmax = float(fin.abs().max()) if fin.numel() else float("nan")
    if bad:
        return dict(pcc=float("nan"), max_abs=float("inf"), ref_absmax=ref_absmax, rel_fro=float("inf"), nonfinite=bad)
    d = (ref - got).abs()
    return dict(
        pcc=float(comp_pcc(ref, got, pcc=0.0)[1]),
        max_abs=float(d.max()),
        ref_absmax=ref_absmax,
        rel_fro=float((ref - got).norm() / max(float(ref.norm()), 1e-30)),
    )


def pcc(a, b) -> float:
    return stats(a, b)["pcc"]


def passes(s: dict, thr: float) -> bool:
    """Acceptance: all values finite and ``pcc >= thr`` (False for a NaN PCC, unlike ``pcc < thr`` checks)."""
    return not s.get("nonfinite") and s["pcc"] >= thr


def worst_of(items) -> dict:
    """The worst chip / case: non-finite or NaN-PCC entries first, then the lowest PCC."""

    def key(s):
        p = s["pcc"]
        ok = not s.get("nonfinite") and p == p
        return (ok, p if p == p else -math.inf)

    return min(items, key=key)


def fmt(s: dict) -> str:
    nf = f" NONFINITE={s['nonfinite']}" if s.get("nonfinite") else ""
    return (f"pcc={s['pcc']:.7f} max_abs={s['max_abs']:.3e} (|ref|max {s['ref_absmax']:.3g}) "
            f"rel_fro={s['rel_fro']:.2e}{nf}")


def record(test: str, case: str, **m):
    line = f"[mlp] {test} | {case} | " + " ".join(
        f"{k}={v:.7g}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()
    )
    RESULTS.append(line)
    print(line, flush=True)


def summary(start: int) -> None:
    """Print this test's result lines (``RESULTS[start:]``; earlier tests' lines are not repeated)."""
    print("\n".join(RESULTS[start:]), flush=True)


# ============================================================================================================
# device helpers (exception-safe trace capture, gates' timing method)
# ============================================================================================================
def _free(o):
    if isinstance(o, (list, tuple)):
        for x in o:
            _free(x)
    elif isinstance(o, dict):
        for x in o.values():
            _free(x)
    elif isinstance(o, ttnn.Tensor):
        try:
            ttnn.deallocate(o)
        except Exception:  # already deallocated
            pass


class _Capture:
    """Exception-safe trace capture (a dangling capture hung close_mesh_device once, GATES_RESULTS §11.6)."""

    def __init__(self, mesh_device):
        self.mesh, self.tid = mesh_device, None

    def __enter__(self):
        self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            ttnn.end_trace_capture(self.mesh, self.tid, cq_id=0)
        except Exception:
            if exc_type is None:
                raise
        if exc_type is not None:
            try:
                ttnn.release_trace(self.mesh, self.tid)
            except Exception:
                pass
        return False


def _trace_min_us(mesh_device, fn, n, reps):
    with _Capture(mesh_device) as cap:
        for _ in range(n):
            _free(fn())
    try:
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            times.append((time.perf_counter() - t0) * 1e6)
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    return min(times)


SYNC_US = 140.0  # synchronize_device on the 32-chip mesh (gate G0 calibration: 133-149 us)


def traced_us(mesh_device, fn, n=None, reps=9, target_us=6000.0, n_max=384):
    """Traced per-call us by the gates' slope method, (min t(n) - min t(n/2)) / (n/2), with ``n`` chosen so one
    replay holds ~``target_us`` of device work (G0: device work below ~150 us hides behind the sync, replays are
    bimodal -> min over ``reps``). A probe trace of 8 calls gives the estimate."""
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    if n is None:
        t8 = _trace_min_us(mesh_device, fn, 8, 3)
        est = max((t8 - SYNC_US) / 8.0, 2.0)
        n = int(min(n_max, max(16, target_us / est)))
        n -= n % 2
    t1 = _trace_min_us(mesh_device, fn, n // 2, reps)
    t2 = _trace_min_us(mesh_device, fn, n, reps)
    return max((t2 - t1) / (n - n // 2), 0.0)


def eager_us(mesh_device, fn, iters=10, warmup=1):
    for _ in range(warmup):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


def to_mesh(host, mesh_device, cfg, *, dtype=ttnn.bfloat16, dp_dim=None, tp_dim=None, layout=ttnn.TILE_LAYOUT,
            device=True):
    return ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device if device else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
        mesh_mapper=W.mesh_mapper(mesh_device, cfg.axes, dp_dim=dp_dim, tp_dim=tp_dim),
    )


def per_chip(t, mesh_device):
    """``[R, C, ...local]`` float32 readback (``[r, c]`` = mesh coordinate)."""
    return device_tensors_to_torch(t, mesh_device).float()


def check_fabric(mesh_device, tag):
    rep = log_fabric(mesh_device, tag)
    assert rep["committed"] is not None, rep
    return rep


def _cfg(mesh_device, **kw):
    return MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device, **kw)


# ============================================================================================================
# real-activation goldens (CPU, host-only; built once)
# ============================================================================================================
def _reference_mlp(source, prefix, inter, dtype):
    """Reference ``modules.MLP`` with the weights of ``prefix`` in ``dtype`` (fp32 = ideal golden, bf16 = HF)."""
    from models.demos.motif3.reference.modules import MLP

    m = MLP(4096, inter, eps=EPS, sigmoid_weight=True, hidden_clamp=1e6, output_scale=0.5)
    sd = {
        "gate_proj.weight": source.get(f"{prefix}.gate_proj.weight"),
        "up_proj.weight": source.get(f"{prefix}.up_proj.weight"),
        "down_proj.weight": source.get(f"{prefix}.down_proj.weight"),
        "act_fn.weight": source.get(f"{prefix}.act_fn.weight").reshape(3),
        "act_fn.bias": source.get(f"{prefix}.act_fn.bias").reshape(1),
    }
    m.load_state_dict({k: v.to(torch.float32) for k, v in sd.items()})
    m.requires_grad_(False)
    return m.to(dtype)


def _ffn_prefix(layer: int) -> str:
    return W.hf_name(layer, "mlp" if layer < 2 else "moe.shared_experts")


def _ffn_inter(layer: int) -> int:
    return 12288 if layer < 2 else 1280


@torch.no_grad()
def make_real_goldens(path: Path = REAL_GOLDEN, layers=(0, 1, 2)) -> Path:
    """Real FFN inputs (``layers.{l}.post_attention_layernorm.out`` for ``l`` in ``layers``, from the reference prefix
    model 0..max(layers) on the rendered prompt set) + reference MLP / shared-expert outputs in fp32 and bf16. CPU
    only. The routed experts of the last layer are skipped (they only feed that layer's output, never the taps); those
    of earlier MoE layers run for real (layer 4's input depends on the MoE outputs of layers 2 and 3)."""
    from models.demos.motif3.reference.golden import TensorRecorder
    from models.demos.motif3.reference.weights import load_reference_model

    layers = tuple(sorted(int(i) for i in layers))
    last = layers[-1]
    prompts_file = TT_METAL / "models/demos/motif3/reference/prompts/rendered.json"
    rendered = {p["name"]: p for p in json.loads(prompts_file.read_text())["prompts"]}
    t0 = time.time()
    model = load_reference_model(layer_ids=tuple(range(last + 1)), dtype=torch.bfloat16, lazy_experts=True)
    if model.model.layers[str(last)].is_moe:
        model.model.layers[str(last)].moe.experts.forward = lambda x, idx, w: torch.zeros(x.shape, dtype=torch.float32)
    want = {f"layers.{i}.post_attention_layernorm.out" for i in layers}
    feats = {i: [] for i in layers}
    names = []
    for name in GOLDEN_PROMPTS:
        ids = torch.tensor([rendered[name]["ids"]])
        S = ids.shape[1]
        rec = TensorRecorder(lambda n: n in want)
        cache = model.new_cache(1, S)
        model(ids, torch.arange(S)[None], cache, "expanded", tap=rec, last_token_only=True)
        for i in layers:
            feats[i].append(rec.tensors[f"layers.{i}.post_attention_layernorm.out"][0])
        names.append((name, S))
        print(f"[mlp goldens] prompt {name} ({S} tokens) done at {time.time() - t0:.0f} s", flush=True)
        if sum(n for _, n in names) >= N_GOLDEN_TOKENS:
            break
    out = {"meta": {"prompts": names, "n_tokens": N_GOLDEN_TOKENS, "version": 1, "layers": list(layers)}, "layers": {}}
    src = W.HFWeightLoader()
    for i in layers:
        f = torch.cat(feats[i], dim=0)[:N_GOLDEN_TOKENS].to(torch.bfloat16).clone()
        prefix, inter = _ffn_prefix(i), _ffn_inter(i)
        ref32 = _reference_mlp(src, prefix, inter, torch.float32)(f.float())
        ref16 = _reference_mlp(src, prefix, inter, torch.bfloat16)(f)
        out["layers"][i] = {"f": f, "ref_fp32": ref32.float().clone(), "ref_bf16": ref16.clone()}
        print(f"[mlp goldens] layer {i}: f {tuple(f.shape)} |f|max {float(f.float().abs().max()):.3g} "
              f"pcc(bf16 ref, fp32 ref) {pcc(ref32, ref16.float()):.7f}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"[mlp goldens] wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_real_goldens(path: Path = REAL_GOLDEN, flag: str = "--make-goldens", required: bool = True):
    if not path.is_file():
        if not required:
            return None
        pytest.skip(f"real goldens missing: run `hostrun.sh -- python {Path(__file__).relative_to(TT_METAL)} "
                    f"{flag}` first")
    return torch.load(path, weights_only=True)


# ============================================================================================================
# CPU tests (no device)
# ============================================================================================================
def test_horner_identity_matches_hf_polynorm():
    """The device formulation (moments + per-row scales + Horner) equals HF's PolyNorm in fp64, incl. TP moments."""
    torch.manual_seed(0)
    I, T = 1280, 16
    g = torch.randn(T, I, dtype=torch.float64) * 3
    g[0, 5] = 25.0  # a spike (one element dominates mean(g^6), study verify_numerics (d))
    u = torch.randn(T, I, dtype=torch.float64)
    coeffs = PolyNormCoefficients.from_tensors(torch.tensor([-0.3, 0.2, 0.5]), torch.tensor([0.15]))
    c0, c1, c2, b = coeffs.c0, coeffs.c1, coeffs.c2, coeffs.b
    ref = poly_golden(g, u, c0, c1, c2, b)
    D, E = horner_scale_constants(torch.tensor(coeffs.by_power, dtype=torch.float64), I, EPS, dtype=torch.float64)
    for tp in (1, 8):  # local stats, and 8 TP shards whose partial sums are added (the all-reduce)
        parts = g.chunk(tp, dim=-1)
        s = torch.stack([sum(p.pow(2 * k).sum(-1) for p in parts) for k in (1, 2, 3)], dim=-1)  # [T, 3]
        a = torch.rsqrt(s * D + E)  # a_k = c_k rsqrt(mean + eps)
        poly = ((a[:, 2:3] * g + a[:, 1:2]) * g + a[:, 0:1]) * g + b
        got = poly * u
        assert torch.allclose(got, ref, rtol=1e-12, atol=1e-12), (tp, float((got - ref).abs().max()))


@pytest.mark.parametrize("layer, inter", [(0, 12288), (2, 1280)])
def test_tp_shards_reproduce_reference_mlp(layer, inter):
    """Host emulation of the per-chip TP math of PolyNormMLP with the module's own weight transforms
    (``weights.mlp_gate_up_for_chip`` / ``mlp_down_for_chip``, x0.5 folded): column-parallel gate|up, 3-moment sums
    added over the 8 shards, Horner with ``a_k = rsqrt(s_k D_k + E_k)``, row-parallel down, partials summed = the
    reference ``modules.MLP`` (fp64, no bf16 cast of the PolyNorm output on either side)."""
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    src = _random_mlp_source(layer, inter, seed=40 + layer)
    p = _ffn_prefix(layer)
    g_w, u_w, d_w = (src.get(f"{p}.{k}_proj.weight").double() for k in ("gate", "up", "down"))
    coeffs = PolyNormCoefficients.from_source(src, p)
    D, E = horner_scale_constants(torch.tensor(coeffs.by_power, dtype=torch.float64), inter, EPS, dtype=torch.float64)
    x = torch.randn(8, 4096, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    n = inter // cfg.tp
    gus = [W.mlp_gate_up_for_chip(g_w, u_w, cfg, t) for t in range(cfg.tp)]  # fp64 in -> fp64 out
    gs = [x @ gu[:, :n] for gu in gus]
    us = [x @ gu[:, n:] for gu in gus]
    s_ = torch.stack([sum(g.pow(2 * k).sum(-1) for g in gs) for k in (1, 2, 3)], dim=-1)  # the TP all-reduce
    a = torch.rsqrt(s_ * D + E)
    y = 0
    for t in range(cfg.tp):
        g = gs[t]
        h = (((a[:, 2:3] * g + a[:, 1:2]) * g + a[:, 0:1]) * g + coeffs.b) * us[t]
        y = y + h @ W.mlp_down_for_chip(d_w, cfg, t)
    gold = (poly_golden(x @ g_w.t(), x @ u_w.t(), coeffs.c0, coeffs.c1, coeffs.c2, coeffs.b) * 0.5) @ d_w.t()
    assert torch.allclose(y, gold, rtol=1e-9, atol=1e-9), float((y - gold).abs().max())
    ref = _reference_mlp(src, p, inter, torch.float64)(x)  # its PolyNorm runs in fp32 (HF: x.float())
    assert torch.allclose(ref, gold, rtol=1e-4, atol=1e-5), float((ref - gold).abs().max())


def test_grouped_constants_host_layout():
    """GroupedPolyNormConsts' host layout equals weights.expert_polynorm_tensors (c0..c2, clamped b), and every chip
    (dp, tp) receives exactly the D / E rows of its own experts [12k, 12k+12), k = 8 dp + tp (mapper emulation)."""
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    g = torch.Generator().manual_seed(9)
    w, bias = torch.randn(cfg.num_experts, 3, generator=g), torch.rand(cfg.num_experts, 1, generator=g) * 2 - 1
    c, b = W.polynorm_coefficients(w, bias, bias_clamp=cfg.polynorm_bias_clamp)
    host = GroupedPolyNormConsts.host_tensors(cfg, c, b)
    want = W.expert_polynorm_tensors(w, bias, cfg)
    for k in ("c0", "c1", "c2", "b"):
        assert torch.equal(host[k], want[k].to(torch.float32)), k
    D, E = horner_scale_constants(torch.stack([c[:, 2], c[:, 1], c[:, 0]], dim=-1), 1280, EPS)
    a = cfg.axes
    for r in range(a.mesh_shape[0]):
        for col in range(a.mesh_shape[1]):
            dp, tp = a.roles(r, col)
            e = list(cfg.experts_of_chip(dp, tp))
            d_chip = W.shard_for_device(host["D"], a, r, col, dp_dim=0, tp_dim=1)  # [3, 12, 1, 1]
            e_chip = W.shard_for_device(host["E"], a, r, col, dp_dim=0, tp_dim=1)
            assert torch.equal(d_chip.reshape(3, 12), D[e].t()) and torch.equal(e_chip.reshape(3, 12), E[e].t())
            c0_chip = W.shard_for_device(host["c0"], a, r, col, dp_dim=0, tp_dim=1).reshape(12)
            assert torch.equal(c0_chip, c[e, 0].to(torch.float32))


def test_coefficients_and_constants():
    w = torch.tensor([0.1, -0.4, 2.0])
    c = PolyNormCoefficients.from_tensors(w, torch.tensor([0.7]))
    assert math.isclose(c.c0, float(torch.sigmoid(w[0])), rel_tol=1e-7) and c.b == pytest.approx(0.7)
    assert c.by_power == (c.c2, c.c1, c.c0)
    cl = PolyNormCoefficients.from_tensors(w, torch.tensor([0.7]), bias_clamp=0.5)
    assert cl.b == pytest.approx(0.5)  # routed-expert clamp (dense / shared never pass bias_clamp)
    D, E = horner_scale_constants(torch.tensor([0.5, 0.25, 1.0]), 1536, 1e-6)
    assert torch.allclose(D, torch.tensor([4 / 1536, 16 / 1536, 1 / 1536]))
    assert torch.allclose(E, torch.tensor([4e-6, 16e-6, 1e-6]))
    with pytest.raises(ValueError):
        horner_scale_constants(torch.tensor([0.0, 0.5, 0.5]), 10, 1e-6)


def test_decode_program_config_builders():
    """The decode matmul configs build for the four dense / shared shapes (config objects only, no device)."""
    dims = {
        ("dense", "gate_up"): (4096, 2 * 1536),
        ("dense", "down"): (1536, 4096),
        ("shared", "gate_up"): (4096, 2 * 160),
        ("shared", "down"): (160, 4096),
    }
    for key, (k, n) in dims.items():
        grid, bw = DECODE_MATMUL_GRIDS[key]
        pc = decode_matmul_pc(k, n, grid, in0_block_w=bw)
        cores = grid[0] * grid[1]
        assert pc.per_core_N * cores >= n // 32 and pc.per_core_N * (cores - 1) < n // 32, key
        assert (k // 32) % pc.in0_block_w == 0, key
    with pytest.raises(ValueError):
        decode_matmul_pc(4096, 320, (12, 10))  # more cores than N tiles


def test_decode_program_configs_fit_compute_grid():
    """``build_decode_program_configs`` (what PolyNormMLP uses): the tuned grids on the 12 x 10 grid; a grid that does
    not fit ``cfg.compute_grid`` (another harvesting) or the matmul shape falls back to the auto config (None), listed
    in the fallbacks; a matmul without a tuned entry is auto without being a fallback. Config objects only."""
    dims = {"gate_up": (4096, 3072), "down": (1536, 4096)}
    pcs, fb = build_decode_program_configs("dense", dims, (12, 10))
    assert fb == [] and all(pc is not None for pc in pcs.values())
    assert pcs["gate_up"].compute_with_storage_grid_size.x == 12 and pcs["gate_up"].compute_with_storage_grid_size.y == 8
    pcs, fb = build_decode_program_configs("dense", dims, (11, 10))  # (12, 8) does not fit, (8, 4) does
    assert pcs["gate_up"] is None and pcs["down"] is not None and len(fb) == 1 and fb[0].startswith("gate_up"), fb
    pcs, fb = build_decode_program_configs("shared", {"gate_up": (4096, 320), "down": (160, 4096)}, (11, 10))
    assert fb == [] and all(pc is not None for pc in pcs.values())
    # a non-default intermediate (2560 -> 320 per chip): the tuned dense grids do not fit those shapes
    pcs, fb = build_decode_program_configs("dense", {"gate_up": (4096, 640), "down": (320, 4096)}, (12, 10))
    assert pcs == {"gate_up": None, "down": None} and len(fb) == 2, fb
    pcs, fb = build_decode_program_configs("dense", {"gate_full": (4096, 12288)}, (12, 10), grids={})
    assert pcs == {"gate_full": None} and fb == []


def test_polynorm_config_semantics():
    """TT applies ``sigmoid(w)`` and one x0.5 output scale (folded into ``W_down``). (a) The local checkpoint configs
    agree: ``polynorm_sigmoid_weight`` absent or True, ``polynorm_output_scale`` present and 0.5 (HF's default when
    absent is 1.0, ``modeling_motif.py:487, 907``; TT and the reference would use 0.5), no per-layer override that
    differs. (b) The helpers honour the fields once ``MotifTTConfig`` carries them (requested shared change). (c) The
    PolyNorm constants' dtype comes from ``cfg.dtypes.polynorm_coeffs`` and must be fp32."""
    from models.demos.motif3.tt.polynorm import _coeff_dtype

    base = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    checked = 0
    for path in (Path(HF_META) / "config.json", Path(base.weights_dir) / "config.json"):
        if not path.is_file():
            continue
        d = json.loads(path.read_text())
        assert d.get("polynorm_sigmoid_weight", True) is True, path
        cfg = MotifTTConfig.from_hf_config(str(path), mesh_shape=(4, 8))
        assert "polynorm_output_scale" in d and float(d["polynorm_output_scale"]) == cfg.polynorm_output_scale == 0.5
        per = d.get("polynorm_output_scale_per_layer") or {}
        assert all(float(v) == cfg.polynorm_output_scale for v in per.values()), per
        assert {polynorm_output_scale(cfg, i) for i in range(cfg.num_layers)} == {0.5}
        check_polynorm_semantics(cfg)
        checked += 1
    assert checked >= 1
    fake = types.SimpleNamespace(polynorm_output_scale=0.5, polynorm_output_scale_per_layer={"3": 0.25, 7: 1.0})
    assert (polynorm_output_scale(fake, 3), polynorm_output_scale(fake, 7), polynorm_output_scale(fake, 4)) == (0.25, 1.0, 0.5)
    with pytest.raises(NotImplementedError):
        check_polynorm_semantics(types.SimpleNamespace(polynorm_sigmoid_weight=False))
    check_polynorm_semantics(types.SimpleNamespace())  # absent = HF default (True)
    assert _coeff_dtype(base) == ttnn.float32
    with pytest.raises(NotImplementedError):
        _coeff_dtype(dataclasses.replace(base, dtypes=dataclasses.replace(base.dtypes, polynorm_coeffs=ttnn.bfloat16)))


def test_metrics_flag_non_finite():
    """The acceptance helpers fail NaN / Inf outputs: one NaN element, an all-Inf output, a NaN chip after the first
    and a NaN PCC (the previous ``pcc < thr`` checks and ``<``-based worst-chip loops passed all of them;
    ``comp_pcc`` alone zeroes non-finite values and passes one NaN in 100 values with PCC 0.990)."""
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(64, 256, generator=g, dtype=torch.float64)
    good = stats(ref, ref + 1e-4 * torch.randn(64, 256, generator=g, dtype=torch.float64))
    assert passes(good, 0.99999) and "nonfinite" not in good
    x = ref.clone()
    x[3, 5] = float("nan")
    one_nan = stats(ref, x)
    assert not passes(one_nan, 0.0) and one_nan["nonfinite"] == 1 and math.isnan(one_nan["pcc"])
    assert "NONFINITE=1" in fmt(one_nan)
    all_inf = stats(ref, torch.full_like(ref, float("inf")))
    assert not passes(all_inf, 0.0) and all_inf["nonfinite"] == ref.numel()
    assert worst_of([good, one_nan, good]) is one_nan and worst_of([good, good, all_inf]) is all_inf
    nan_pcc = dict(good, pcc=float("nan"))
    assert not passes(nan_pcc, 0.5) and worst_of([good, nan_pcc]) is nan_pcc
    lower = dict(good, pcc=0.9)
    assert worst_of([good, lower, good]) is lower and passes(lower, 0.9) and not passes(lower, 0.91)


def test_modules_import_is_self_contained():
    """README §13: tt/polynorm.py and tt/mlp.py import no other models/demos package and open no device."""
    probe = (
        "import importlib, json, sys\n"
        "for m in ('models.demos.motif3.tt.polynorm', 'models.demos.motif3.tt.mlp'):\n"
        "    importlib.import_module(m)\n"
        "demos = sorted(m for m in sys.modules if m.startswith('models.demos.') and not m.startswith('models.demos.motif3'))\n"
        "heavy = sorted(m for m in ('vllm', 'transformers', 'huggingface_hub', 'safetensors') if m in sys.modules)\n"
        "print(json.dumps({'demos': demos, 'heavy': heavy}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TT_METAL) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run([sys.executable, "-c", probe], cwd=str(TT_METAL), env=env, capture_output=True, text=True,
                         timeout=300)
    assert res.returncode == 0, res.stderr[-3000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out == {"demos": [], "heavy": []}, out
    for marker in ("Opening user mode device driver", "Starting devices in cluster"):
        assert marker not in res.stderr and marker not in res.stdout


def _hostdata_heavy(shape, gen, scale=1.0):
    z = torch.randn(*shape, generator=gen)
    chi = torch.randn(*shape, 4, generator=gen).pow(2).sum(-1) / 4
    return (z / chi.sqrt() * scale).clamp(-60, 60)


def test_shared_polynorm_host_numerics():
    """B5 (``tt/kernels/shared_polynorm.py``) on the host: the kernels' algorithm (= the release decode path:
    tile-sequential fp32 moments per chip, chip partials summed in gather order, ``rsqrt(s D + E)``, Horner, ``* up``,
    one bf16 rounding; :func:`emulate_fp32`) at the shared expert's real dims (8 chips x 5 tiles, 32 rows) matches the
    HF PolyNorm in fp64 (:func:`golden_fp64`, synthetic coefficients in the real range and heavy-tailed gates); every
    row depends only on its own inputs (rows 0..15 of a 32-row call == a call on those rows, the T64 / T32 property);
    the D / E constants are the module's :func:`horner_scale_constants`; :func:`plan` sizes and refusals."""
    from models.demos.motif3.tt.kernels import shared_polynorm as SP

    gen = torch.Generator().manual_seed(5)
    tp, n, T = 8, 5, 32
    inter = tp * n * 32
    for case in range(3):
        c = torch.sigmoid(torch.randn(3, generator=gen))  # c of N(g), N(g^2), N(g^3)
        b = float(torch.randn(1, generator=gen) * 0.3)
        D, Ec = horner_scale_constants(c.double(), inter, EPS)
        g = _hostdata_heavy((T, inter), gen, scale=2.0 + case)
        u = torch.randn(T, inter, generator=gen)
        chips = [torch.cat([g[:, k * n * 32:(k + 1) * n * 32], u[:, k * n * 32:(k + 1) * n * 32]], 1) for k in range(tp)]
        got = torch.cat(SP.emulate_fp32(chips, D.float(), Ec.float(), b), 1).double()
        want = SP.golden_fp64(g, u, c, b, eps=EPS)
        st = stats(want, got)
        assert passes(st, 0.99999), st
        # within one bf16 rounding of the fp64 value (+ fp32 noise)
        assert float(((got - want).abs() - (want.abs() * 2.0**-8 + 1e-5)).max()) <= 0, case
        half = SP.emulate_fp32([x[:16] for x in chips], D.float(), Ec.float(), b)
        full = SP.emulate_fp32(chips, D.float(), Ec.float(), b)
        assert all(torch.equal(h, f[:16]) for h, f in zip(half, full))
    p = SP.plan(5, 8)
    assert p["n"] == 5 and p["tp"] == 8 and p["moments_l1"] == 8 * 4096
    assert p["apply_l1"] == (24 + 7 + 5 + 5 + 4) * 4096 + 2 * 2048
    for bad in ((0, 8), (SP.MAX_TILES + 1, 8), (5, 0), (48, 8)):
        with pytest.raises(ValueError, match="fused shared PolyNorm"):
            SP.plan(*bad)


def test_shared_polynorm_host_dispatch(monkeypatch):
    """B5 host side of the MLP wiring (ttnn ops recorded, no device): ``resolve_shared_polynorm`` (config default,
    explicit modes, bad modes; "fused" needs the shared expert, ``stats="tp"``, the fp32 decode PolyNorm, the release
    ``moments`` / ``horner`` / ``ar`` knobs and a width the one-core kernels hold: explicit raises, the config default
    falls back to "composite"); ``_rows`` runs the fused kernels (on the gate_up output, no slices) at decode M = 32 and
    the composite for prefill, M != 32, the bf16 decode PolyNorm, changed ``pn_kw`` and a module without the kernel
    (the default); ``gu`` and ``h`` are freed exactly once; ``taps`` still get gate / up slices."""
    import models.demos.motif3.tt.mlp as MM
    from models.demos.motif3.tt.model_config import SHARED_POLYNORM_MODES

    assert SHARED_POLYNORM_MODES == ("composite", "fused")
    monkeypatch.delenv("MOTIF3_SHARED_POLYNORM", raising=False)
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    assert cfg.shared_polynorm == "fused"  # the default since the B5 gates
    R = MM.resolve_shared_polynorm
    ok = dict(kind="shared", stats="tp", decode_polynorm="fp32", pn_kw=dict(MM.RELEASE_PN_KW), n_local=160)
    assert R(None, cfg, **ok) == "fused" and R("fused", cfg, **ok) == "fused"
    assert R("composite", cfg, **ok) == "composite"
    fz = types.SimpleNamespace(shared_polynorm="fused")
    assert R(None, fz, **ok) == "fused" and R(None, types.SimpleNamespace(), **ok) == "composite"
    for bad in (dict(kind="dense"), dict(stats="replicated_gate"), dict(decode_polynorm="bf16"),
                dict(pn_kw=dict(MM.RELEASE_PN_KW, horner="binary")), dict(pn_kw=dict(MM.RELEASE_PN_KW, ar="all_reduce")),
                dict(n_local=1536), dict(n_local=100)):
        kw = dict(ok)
        kw.update(bad)
        assert R(None, fz, **kw) == "composite", bad
        with pytest.raises(ValueError, match="fused"):
            R("fused", cfg, **kw)
    with pytest.raises(ValueError, match="shared_polynorm"):
        R("kernel", cfg, **ok)

    calls, freed = [], []

    def T(tag, shape):
        return types.SimpleNamespace(tag=tag, shape=list(shape))

    class FakeFused:
        def __call__(self, gu, *, memory_config=None):
            calls.append(("fused", gu.tag, memory_config))
            return T("h_fused", [1, 1, gu.shape[2], 160])

    def linear(a, w, *, dtype=None, **kw):
        calls.append(("linear", w))
        if w == "W_gate_up":
            return T("gu", [1, 1, a.shape[2], 320])
        return T("y", [1, 1, a.shape[2], 4096])

    def slice_(t, start, end, **kw):
        calls.append(("slice", t.tag))
        return T(f"{t.tag}_slice{start[-1]}", [1, 1, end[2] - start[2], end[3] - start[3]])

    def poly(g, u, consts, **kw):
        calls.append(("composite", kw.get("mode"), kw.get("moments"), kw.get("ar")))
        return T("h_comp", g.shape)

    monkeypatch.setattr(MM.ttnn, "linear", linear)
    monkeypatch.setattr(MM.ttnn, "slice", slice_)
    monkeypatch.setattr(MM.ttnn, "deallocate", lambda t, *a, **k: freed.append(getattr(t, "tag", t)))
    monkeypatch.setattr(MM, "polynorm_tp", poly)
    mlp = object.__new__(MM.PolyNormMLP)
    mlp.n_local, mlp.stats, mlp.pn, mlp.ccl, mlp.ckc_mm, mlp.ckc_pn = 160, "tp", "pn", "ccl", "ckc", "ckcpn"
    mlp.pn_exact_ar, mlp.pn_kw, mlp.decode_pc = True, dict(MM.RELEASE_PN_KW), {"gate_up": "pc", "down": "pc"}
    mlp.w_gate_up, mlp.w_down = "W_gate_up", "W_down"
    x = lambda m: T("x", [1, 1, m, 4096])  # noqa: E731
    cases = [  # (pn_fused, M, decode, mode, pn_kw change, want)
        (None, 32, True, "fp32", {}, "composite"),
        (FakeFused(), 32, True, "fp32", {}, "fused"),
        (FakeFused(), 32, False, "fp32", {}, "composite"),  # prefill
        (FakeFused(), 64, True, "fp32", {}, "composite"),  # not one tile row
        (FakeFused(), 32, True, "bf16", {}, "composite"),
        (FakeFused(), 32, True, "fp32", {"ar": "all_reduce"}, "composite"),
    ]
    for fused, m, decode, mode, kwc, want in cases:
        mlp.pn_fused = fused
        mlp.pn_kw = dict(MM.RELEASE_PN_KW, **kwc)
        for taps in (None, {}):
            calls.clear(), freed.clear()
            y = MM.PolyNormMLP._rows(mlp, x(m), mode=mode, decode=decode, all_reduce=False, out_dtype="bf16",
                                     out_mc="L1o", imc="L1", taps=taps)
            assert y.tag == "y"
            kinds = [c[0] for c in calls if c[0] in ("fused", "composite")]
            assert kinds == [want], (fused, m, decode, mode, kwc, calls)
            assert freed.count("gu") == 1, freed
            if want == "fused":
                assert calls[1] == ("fused", "gu", "L1")
                assert [c for c in calls if c[0] == "slice"] == ([] if taps is None else [("slice", "gu")] * 2)
            if taps is None:
                assert freed.count(f"h_{'fused' if want == 'fused' else 'comp'}") == 1, freed
            else:
                assert taps["gate"].tag == "gu_slice0" and taps["up"].tag == "gu_slice160"
                assert taps["act"].tag == ("h_fused" if want == "fused" else "h_comp")
                assert not any(t.startswith("h_") for t in freed)


# ============================================================================================================
# device: PolyNorm
# ============================================================================================================
def _heavy_tailed(shape, gen, scale=1.0):
    """Student-t(4)-like data with a few spikes (real gate rows: one element dominates mean(g^6))."""
    z = torch.randn(*shape, generator=gen)
    chi = torch.randn(*shape, 4, generator=gen).pow(2).sum(-1) / 4
    t = z / chi.sqrt()
    return (t * scale).clamp(-60, 60)


def _real_scalar_coeffs(layer):
    src = W.HFWeightLoader()
    if not src.layer_available(layer):
        return None
    return PolyNormCoefficients.from_source(src, _ffn_prefix(layer))


class _NoReadSource:
    """A weight source that fails on every read: a module built from it must come entirely from the TT cache."""

    def get(self, name, dtype=None):
        raise AssertionError(f"cache hit expected, but {name} was read")

    def get_rows(self, name, *args, **kwargs):
        raise AssertionError(f"cache hit expected, but {name} was read")

    def has(self, name):
        return True

    def __contains__(self, name):
        return True


def _tp_worst(ref_rows, got, a, n):
    """Worst chip of a TP-sharded ``[R, C, 1, 1, T, n]`` readback vs ``ref_rows [dp, 1, T, I]`` (chip (dp, tp) holds
    columns ``[tp n, (tp + 1) n)`` of row ``dp``)."""
    R, C = a.mesh_shape
    out = []
    for r in range(R):
        for c in range(C):
            dp, tp = a.roles(r, c)
            out.append(stats(ref_rows[dp, 0, :, tp * n:(tp + 1) * n], got[r, c, 0, 0]))
    return worst_of(out)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_polynorm_device(mesh_device, device_params):
    r0 = len(RESULTS)
    cfg = _cfg(mesh_device)
    check_fabric(mesh_device, "polynorm")
    ccl = MotifCCL(mesh_device, cfg)
    a = cfg.axes
    R, C = a.mesh_shape
    gen = torch.Generator().manual_seed(1234)
    failures = []
    ckc = cfg.compute_config("polynorm")

    # ---- scalar TP PolyNorm: dense (12288 -> 8 x 1536) and shared (1280 -> 8 x 160), T = 8 rows per DP row ----------
    for layer, inter in ((0, 12288), (1, 12288), (2, 1280)):
        coeffs = _real_scalar_coeffs(layer) or PolyNormCoefficients.from_tensors(
            torch.tensor([-0.5, 0.1, 0.2]), torch.tensor([0.05])
        )
        consts = ScalarPolyNormConsts(mesh_device, cfg, coeffs, inter=inter)
        T = 8
        scale = 1.0 if layer < 2 else 3.0  # dense gates are small (|g| <= ~3), shared / routed larger
        G = _heavy_tailed((a.dp_size, 1, T, inter), gen, scale)  # row dp: its 8 lanes' full intermediate
        U = _heavy_tailed((a.dp_size, 1, T, inter), gen, 1.0)
        G16, U16 = G.bfloat16().float(), U.bfloat16().float()
        ref = poly_golden(G16, U16, coeffs.c0, coeffs.c1, coeffs.c2, coeffs.b)  # [dp, 1, T, I]
        for in_dtype in (ttnn.float32, ttnn.bfloat16):
            g_t = to_mesh(G16 if in_dtype == ttnn.bfloat16 else G, mesh_device, cfg, dtype=in_dtype, dp_dim=0, tp_dim=3)
            u_t = to_mesh(U16 if in_dtype == ttnn.bfloat16 else U, mesh_device, cfg, dtype=in_dtype, dp_dim=0, tp_dim=3)
            ref_in = ref if in_dtype == ttnn.bfloat16 else poly_golden(G, U, coeffs.c0, coeffs.c1, coeffs.c2, coeffs.b)
            for mode in ("fp32", "bf16"):
                for imc_name, imc in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
                    case = f"tp L{layer} I={inter} in={in_dtype.name} mode={mode} {imc_name}"
                    try:
                        fn = lambda: polynorm_tp(g_t, u_t, consts, ccl=ccl, mode=mode, memory_config=imc,
                                                 compute_kernel_config=ckc)
                        h = fn()
                        got = per_chip(h, mesh_device)  # [R, C, 1, 1, T, n]
                        _free(h)
                        worst = _tp_worst(ref_in, got, a, inter // a.tp_size)
                        timed = in_dtype == ttnn.float32 and layer != 1 and (imc_name == "l1" or mode == "fp32")
                        us = traced_us(mesh_device, fn) if timed else float("nan")
                        record("polynorm", case, **worst, traced_us=us)
                        thr = 0.99999 if mode == "fp32" else 0.9999
                        if not passes(worst, thr):
                            failures.append(f"{case}: {fmt(worst)} < {thr}")
                    except Exception as e:
                        failures.append(f"{case}: {type(e).__name__}: {str(e)[:400]}")
                        record("polynorm", case, error=f"{type(e).__name__}: {str(e)[:200]}")
            _free([g_t, u_t])
        consts.deallocate()

    # ---- grouped PolyNorm (routed experts): per chip [1, 12, 32, 1280], each chip its own 12 experts ------------------
    src = W.HFWeightLoader()
    if src.layer_available(2):
        cw, cb = W.polynorm_coefficients(src.get(W.hf_name(2, "moe.experts.act_fn.weight")),
                                         src.get(W.hf_name(2, "moe.experts.act_fn.bias")),
                                         bias_clamp=cfg.polynorm_bias_clamp)
        tag = "real L2"
    else:
        cw = torch.sigmoid(torch.randn(cfg.num_experts, 3, generator=gen))
        cb = (torch.rand(cfg.num_experts, 1, generator=gen) * 2 - 1).clamp(-0.5, 0.5)
        tag = "synthetic"
    gconsts = GroupedPolyNormConsts.from_coefficients(mesh_device, cfg, cw, cb.reshape(-1))
    E, M, I = cfg.experts_per_chip, 32, cfg.moe_intermediate_size
    # host [dp, 96, M, 2I] -> chip (dp, tp) [1, 12, M, 2I] holds experts [12k, 12k+12) of k = 8 dp + tp
    GU = torch.cat([_heavy_tailed((a.dp_size, cfg.num_experts // a.dp_size, M, I), gen, 4.0),
                    _heavy_tailed((a.dp_size, cfg.num_experts // a.dp_size, M, I), gen, 1.5)], dim=-1)
    GU16 = GU.bfloat16().float()
    c_e = cw.reshape(a.dp_size, -1, 1, 1, 3)
    b_e = cb.reshape(a.dp_size, -1, 1, 1)
    ref_g = poly_golden(GU16[..., :I], GU16[..., I:], c_e[..., 0], c_e[..., 1], c_e[..., 2], b_e)  # [dp, 96, M, I]
    gu_t = to_mesh(GU16, mesh_device, cfg, dtype=ttnn.bfloat16, dp_dim=0, tp_dim=1)
    gu32_t = to_mesh(GU16, mesh_device, cfg, dtype=ttnn.float32, dp_dim=0, tp_dim=1)
    for impl, mode, gin, moments in (("horner", "fp32", gu32_t, "sum"), ("horner", "fp32", gu_t, "sum"),
                                     ("horner", "bf16", gu_t, "sum"), ("rms", "fp32", gu32_t, "sum"),
                                     ("rms", "bf16", gu_t, "sum"), ("horner", "fp32", gu32_t, "pre_ag"),
                                     ("horner", "fp32", gu32_t, "sum_fast")):
        case = f"grouped {tag} impl={impl} mode={mode} in={gin.dtype.name}" + (f" moments={moments}" if moments != "sum" else "")
        try:
            fn = lambda: grouped_polynorm(gin, gconsts, inter=I, mode=mode, impl=impl, compute_kernel_config=ckc,  # noqa: E731
                                          moments=moments)
            h = fn()
            got = per_chip(h, mesh_device)  # [R, C, 1, 12, M, I]
            _free(h)
            worst = worst_of([stats(ref_g[a.roles(r, c)[0], 12 * a.roles(r, c)[1]:12 * a.roles(r, c)[1] + 12], got[r, c, 0])
                              for r in range(R) for c in range(C)])
            us = traced_us(mesh_device, fn) if moments == "sum" else float("nan")
            record("polynorm", case, **worst, traced_us=us)
            thr = 0.99999 if (mode == "fp32" and impl == "horner" and moments != "sum_fast") else 0.9995
            if not passes(worst, thr):
                failures.append(f"{case}: {fmt(worst)} < {thr}")
        except Exception as e:
            failures.append(f"{case}: {type(e).__name__}: {str(e)[:400]}")
            record("polynorm", case, error=f"{type(e).__name__}: {str(e)[:200]}")
    _free([gu_t, gu32_t])

    # ---- grouped constants from the TT cache alone (the MoE path: from_source reads act_fn only on a cache miss) ----
    if src.layer_available(2):
        root = GOLDEN_DIR / "wcache_tmp_grouped"
        shutil.rmtree(root, ignore_errors=True)
        cfg_c = dataclasses.replace(cfg, tt_cache_root=root)  # small, private cache root (deleted below)
        case = "grouped real L2 constants: from_source(cache=True), then rebuilt from the TT cache alone"
        try:
            k1 = GroupedPolyNormConsts.from_source(mesh_device, cfg_c, 2, source=src, cache=True)
            files = sorted((cfg_c.cache_dir / "L02").glob("*.tensorbin"))
            k2 = GroupedPolyNormConsts.from_source(mesh_device, cfg_c, 2, source=_NoReadSource(), cache=True)

            def tensors(k):
                return [k.c[m][x] for m in ("fp32", "bf16") for x in ("c0", "c1", "c2", "b")] + [k.D, k.E]

            same = all(torch.equal(per_chip(t1, mesh_device), per_chip(t2, mesh_device))
                       for t1, t2 in zip(tensors(k1), tensors(k2)))
            same_direct = all(torch.equal(per_chip(t1, mesh_device), per_chip(t2, mesh_device))
                              for t1, t2 in zip(tensors(k2), tensors(gconsts)))
            record("polynorm", case, files=len(files), MB=sum(p.stat().st_size for p in files) / 1e6,
                   bitwise_reload=same, bitwise_vs_uncached=same_direct)
            if not (same and same_direct and len(files) == 10):
                failures.append(f"{case}: files {len(files)} reload {same} vs uncached {same_direct}")
            k1.deallocate()
            k2.deallocate()
        except Exception as e:
            failures.append(f"{case}: {type(e).__name__}: {str(e)[:400]}")
            record("polynorm", case, error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            shutil.rmtree(root, ignore_errors=True)
    gconsts.deallocate()

    # ---- scalar TP implementation knobs: every advertised moments / ar / horner combination runs and is accurate ----
    kgen = torch.Generator().manual_seed(4321)
    L1 = ttnn.L1_MEMORY_CONFIG
    for layer, inter in ((0, 12288), (2, 1280)):
        coeffs = _real_scalar_coeffs(layer) or PolyNormCoefficients.from_tensors(
            torch.tensor([-0.5, 0.1, 0.2]), torch.tensor([0.05])
        )
        consts = ScalarPolyNormConsts(mesh_device, cfg, coeffs, inter=inter)
        G = _heavy_tailed((a.dp_size, 1, 8, inter), kgen, 1.0 if layer < 2 else 3.0)
        U = _heavy_tailed((a.dp_size, 1, 8, inter), kgen, 1.0)
        ref = poly_golden(G, U, coeffs.c0, coeffs.c1, coeffs.c2, coeffs.b)
        g_t = to_mesh(G, mesh_device, cfg, dtype=ttnn.float32, dp_dim=0, tp_dim=3)
        u_t = to_mesh(U, mesh_device, cfg, dtype=ttnn.float32, dp_dim=0, tp_dim=3)
        for moments, ar, horner, thr in (("pre_ag", "ag_sum", "mac", 0.99999), ("pre_ag", "all_reduce", "mac", 0.99999),
                                         ("sum", "all_reduce", "mac", 0.99999), ("sum_fast", "ag_sum", "mac", 0.9999),
                                         ("sum", "ag_sum", "binary", 0.99999)):
            case = f"tp knobs L{layer} I={inter} moments={moments} ar={ar} horner={horner} fp32 l1"
            try:
                h = polynorm_tp(g_t, u_t, consts, ccl=ccl, mode="fp32", memory_config=L1, compute_kernel_config=ckc,
                                moments=moments, ar=ar, horner=horner)
                got = per_chip(h, mesh_device)
                _free(h)
                worst = _tp_worst(ref, got, a, inter // a.tp_size)
                record("polynorm", case, **worst)
                if not passes(worst, thr):
                    failures.append(f"{case}: {fmt(worst)} < {thr}")
            except Exception as e:
                failures.append(f"{case}: {type(e).__name__}: {str(e)[:400]}")
                record("polynorm", case, error=f"{type(e).__name__}: {str(e)[:200]}")
        _free([g_t, u_t])
        consts.deallocate()
    summary(r0)
    assert not failures, "\n".join(failures)


# ============================================================================================================
# device: MLP modules
# ============================================================================================================
def _random_mlp_source(layer: int, inter: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    p = _ffn_prefix(layer)

    def lin(o, i):
        return (torch.randn(o, i, generator=g) * i**-0.5).bfloat16()

    return W.DictWeightSource({
        f"{p}.gate_proj.weight": lin(inter, 4096),
        f"{p}.up_proj.weight": lin(inter, 4096),
        f"{p}.down_proj.weight": lin(4096, inter),
        f"{p}.act_fn.weight": torch.randn(3, generator=g).bfloat16(),
        f"{p}.act_fn.bias": (torch.rand(1, generator=g) * 2 - 1).bfloat16(),
    })


def _rows_input(x_lanes, mesh_device, cfg, device=True):
    """``x_lanes [32, 4096]`` (lane order) -> per-DP-row decode input: chip (dp, *) holds lanes 8dp..8dp+7."""
    rows = x_lanes.reshape(cfg.dp, 1, cfg.lanes_per_row, x_lanes.shape[-1])
    return to_mesh(rows.bfloat16(), mesh_device, cfg, dtype=ttnn.bfloat16, dp_dim=0, device=device)


def _check_decode(name, out_t, ref_lanes, mesh_device, cfg, thr, failures, *, partial=False):
    """Row dp of every chip vs reference lanes 8dp..8dp+7; replicas identical over TP (unless ``partial``)."""
    got = per_chip(out_t, mesh_device)  # [R, C, 1, 1, 8, 4096]
    a = cfg.axes
    R, C = a.mesh_shape
    if partial:  # sum the TP partials on the host
        per_row = {}
        for r in range(R):
            for c in range(C):
                dp, _ = a.roles(r, c)
                per_row[dp] = per_row.get(dp, 0) + got[r, c, 0, 0].double()
        outs = [per_row[dp] for dp in range(a.dp_size)]
    else:
        outs = []
        for dp in range(a.dp_size):
            r, c = a.coord(dp, 0)
            outs.append(got[r, c, 0, 0])
        if not replicas_identical(out_t, mesh_device, "tp", a):
            failures.append(f"{name}: TP replicas differ")
    full = torch.cat(outs, dim=0)
    s = stats(ref_lanes, full)
    if not passes(s, thr):
        failures.append(f"{name}: {fmt(s)} < {thr}")
    return s


def _check_prefill(name, out_t, ref, mesh_device, cfg, thr, failures):
    got = per_chip(out_t, mesh_device)
    if not replicas_identical(out_t, mesh_device, "tp", cfg.axes) or not replicas_identical(out_t, mesh_device, "dp", cfg.axes):
        failures.append(f"{name}: replicas differ")
    s = stats(ref, got[0, 0, 0, 0])
    if not passes(s, thr):
        failures.append(f"{name}: {fmt(s)} < {thr}")
    return s


def _chips(t, mesh_device, coords):
    """Local tensors of the chips at mesh ``coords`` only (large prefill outputs: no full 32-chip readback);
    ``ttnn.get_device_tensors`` lists the chips in row-major mesh order."""
    C = int(tuple(mesh_device.shape)[1])
    dts = ttnn.get_device_tensors(t)
    return [ttnn.to_torch(dts[r * C + c]).float() for r, c in coords]


def _prefill_chunk_checks(mlp, ref32, layer, mesh_device, cfg, failures, gen):
    """The chunked prefill path (``S > prefill_row_chunk``; buckets 16K / 32K take it with the default 8192 chunk):
    S = 1024 with chunk 384 (chunks 384 / 384 / 256: ragged tail) vs the reference and vs the unchunked call, the
    chunked TP partials (``all_reduce=False``, summed on the host), and S = 16384 with the default chunk (2 chunks) on
    four chips (corners of the mesh; no 32-chip readback of 128 MB outputs)."""
    a = cfg.axes
    R, C = a.mesh_shape
    kind = mlp.kind
    S = 1024
    x = torch.randn(S, 4096, generator=gen).bfloat16()
    with torch.no_grad():
        want = ref32(x.float())
    xl = to_mesh(x.reshape(1, 1, S, 4096), mesh_device, cfg)
    y_un = mlp.forward_prefill(xl)
    s = _check_prefill(f"L{layer} prefill{S}", y_un, want, mesh_device, cfg, 0.999, failures)
    record("mlp_random", f"L{layer} {kind} bfp8 prefill S={S} (unchunked) vs fp32 ref", **s)
    default_chunk = mlp.prefill_row_chunk
    mlp.prefill_row_chunk = 384
    try:
        y_ch = mlp.forward_prefill(xl)
        s = _check_prefill(f"L{layer} prefill{S} chunk384", y_ch, want, mesh_device, cfg, 0.999, failures)
        u0, c0 = _chips(y_un, mesh_device, [(0, 0)])[0], _chips(y_ch, mesh_device, [(0, 0)])[0]
        bitwise = bool(torch.equal(u0, c0))
        s_uc = stats(u0, c0)
        record("mlp_random", f"L{layer} {kind} bfp8 prefill S={S} chunk 384 (3 chunks, ragged tail) vs fp32 ref", **s,
               bitwise_vs_unchunked=bitwise, pcc_vs_unchunked=s_uc["pcc"], max_abs_vs_unchunked=s_uc["max_abs"])
        if not (bitwise or passes(s_uc, 0.99999)):
            failures.append(f"L{layer} chunked vs unchunked prefill: {fmt(s_uc)}")
        y_p = mlp.forward_prefill(xl, all_reduce=False)  # chunked TP partials
        got = per_chip(y_p, mesh_device)  # [R, C, 1, 1, S, 4096]
        rows = [sum(got[a.coord(dp, tp)][0, 0].double() for tp in range(a.tp_size)) for dp in range(a.dp_size)]
        sp = worst_of([stats(want, r_) for r_ in rows])
        record("mlp_random", f"L{layer} {kind} bfp8 prefill S={S} chunk 384 partials (host TP sum, worst DP row)", **sp)
        if not passes(sp, 0.999):
            failures.append(f"L{layer} chunked prefill partials: {fmt(sp)}")
        _free([y_ch, y_p])
    finally:
        mlp.prefill_row_chunk = default_chunk
    _free([y_un, xl])
    S = 2 * default_chunk  # 16384: two chunks of 8192
    x = torch.randn(S, 4096, generator=gen).bfloat16()
    t0 = time.perf_counter()
    with torch.no_grad():
        want = ref32(x.float())
    t_ref = time.perf_counter() - t0
    xb = to_mesh(x.reshape(1, 1, S, 4096), mesh_device, cfg)
    _free(mlp.forward_prefill(xb))  # compile
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    y = mlp.forward_prefill(xb)
    ttnn.synchronize_device(mesh_device)
    t_dev = (time.perf_counter() - t0) * 1e3
    coords = [(0, 0), (0, C - 1), (R - 1, 0), (R - 1, C - 1)]
    outs = _chips(y, mesh_device, coords)
    same = all(torch.equal(outs[0], o) for o in outs[1:])
    s = stats(want, outs[0].reshape(S, 4096))
    record("mlp_random", f"L{layer} {kind} bfp8 prefill S={S} ({S // default_chunk} chunks of {default_chunk}) vs fp32 ref",
           **s, replicas_identical=same, eager_ms=t_dev, cpu_ref_s=t_ref)
    if not same:
        failures.append(f"L{layer} prefill{S}: replicas {coords} differ")
    if not passes(s, 0.999):
        failures.append(f"L{layer} prefill{S}: {fmt(s)} < 0.999")
    _free([y, xb])


def _cache_roundtrip(mesh_device, cfg, ccl, failures, gen):
    """Build each kind with ``cache=True`` into a private cache root, then again from a source that raises on any read
    (start from the TT cache alone, HF shards gone): bitwise identical outputs, the expected 6 files (gate_up, down,
    polynorm.{D, E, b fp32, b bf16}), lazy host coefficients. Shared expert at real dims (~17 MB of cache); the dense
    kind at a reduced intermediate (2560: ~35 MB instead of 160 MB), which also runs its decode matmuls on the
    auto-config fallback (the tuned dense grids do not fit those shapes)."""
    root = GOLDEN_DIR / "wcache_tmp"
    shutil.rmtree(root, ignore_errors=True)
    try:
        for layer, inter, over in ((2, 1280, {}), (0, 2560, {"intermediate_size": 2560})):
            cfg_c = dataclasses.replace(cfg, tt_cache_root=root, **over)  # small, private cache root (deleted below)
            src = _random_mlp_source(layer, inter, seed=700 + layer)
            ref32 = _reference_mlp(src, _ffn_prefix(layer), inter, torch.float32)
            x = torch.randn(32, 4096, generator=gen).bfloat16()
            xd = _rows_input(x, mesh_device, cfg_c)
            xp = to_mesh(torch.cat([x, x, x, x]).reshape(1, 1, 128, 4096), mesh_device, cfg_c)
            m1 = PolyNormMLP(mesh_device, cfg_c, layer, source=src, ccl=ccl, cache=True)
            files = sorted((cfg_c.cache_dir / f"L{layer:02d}").glob("*.tensorbin"))
            m2 = PolyNormMLP(mesh_device, cfg_c, layer, source=_NoReadSource(), ccl=ccl, cache=True)
            name = f"L{layer} {m2.kind} I={inter} rebuilt from the TT cache alone (source raises on read)"
            y1, y2 = m1.forward_decode(xd), m2.forward_decode(xd)
            bitwise = bool(torch.equal(per_chip(y1, mesh_device), per_chip(y2, mesh_device)))
            s = _check_decode(name, y2, ref32(x.float()), mesh_device, cfg_c, 0.999, failures)
            p1, p2 = m1.forward_prefill(xp), m2.forward_prefill(xp)
            bitwise_p = bool(torch.equal(per_chip(p1, mesh_device), per_chip(p2, mesh_device)))
            try:
                m2.coeffs
                lazy = False
            except AssertionError:
                lazy = True  # the host coefficients were never needed (every constant came from the cache)
            record("mlp_random", f"{name}: decode vs fp32 ref", **s, files=len(files),
                   MB=sum(f.stat().st_size for f in files) / 1e6, bitwise_decode=bitwise, bitwise_prefill128=bitwise_p,
                   coeffs_lazy=lazy, auto_fallbacks=len(m2.decode_pc_fallbacks))
            if not (bitwise and bitwise_p and lazy and len(files) == 6):
                failures.append(f"{name}: files {[f.name for f in files]} bitwise {bitwise}/{bitwise_p} lazy {lazy}")
            if inter == 2560 and len(m2.decode_pc_fallbacks) != 2:
                failures.append(f"{name}: expected 2 auto-config fallbacks, got {m2.decode_pc_fallbacks}")
            _free([y1, y2, p1, p2, xd, xp])
            m1.deallocate()
            m2.deallocate()
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mlp_device_random_weights(mesh_device, device_params):
    r0 = len(RESULTS)
    cfg = _cfg(mesh_device)
    check_fabric(mesh_device, "mlp_random")
    ccl = MotifCCL(mesh_device, cfg)
    a = cfg.axes
    failures = []
    gen = torch.Generator().manual_seed(7)
    gen2 = torch.Generator().manual_seed(8)  # the added cases (the original cases keep their data)
    for layer, inter in ((0, 12288), (2, 1280)):
        src = _random_mlp_source(layer, inter, seed=100 + layer)
        ref32 = _reference_mlp(src, _ffn_prefix(layer), inter, torch.float32)
        ref16 = _reference_mlp(src, _ffn_prefix(layer), inter, torch.bfloat16)
        x_dec = torch.randn(32, 4096, generator=gen).bfloat16()
        x_pre = torch.randn(128, 4096, generator=gen).bfloat16()
        want_dec32, want_pre32 = ref32(x_dec.float()), ref32(x_pre.float())
        want_dec16 = ref16(x_dec).float()
        base = stats(want_dec32, want_dec16)
        record("mlp_random", f"L{layer} reference bf16 vs fp32 (decode rows)", pcc=base["pcc"], max_abs=base["max_abs"])
        xd = _rows_input(x_dec, mesh_device, cfg)
        xp = to_mesh(x_pre.reshape(1, 1, 128, 4096), mesh_device, cfg)
        for wname, wdt in (("bfp8", None), ("bf16", ttnn.bfloat16)):
            mlp = PolyNormMLP(mesh_device, cfg, layer, source=src, ccl=ccl, cache=False, weight_dtype=wdt)
            thr = 0.999 if wname == "bfp8" else 0.9999
            y = mlp.forward_decode(xd)
            s = _check_decode(f"L{layer} {wname} decode", y, want_dec32, mesh_device, cfg, thr, failures)
            _free(y)
            record("mlp_random", f"L{layer} {mlp.kind} {wname} decode vs fp32 ref", **s)
            if wname == "bfp8":  # forward_decode zero-fills x's tile padding in place but never its logical rows
                xin = per_chip(xd, mesh_device)
                same = all(torch.equal(xin[a.coord(dp, tp)][0, 0], x_dec[8 * dp:8 * dp + 8].float())
                           for dp in range(a.dp_size) for tp in range(a.tp_size))
                record("mlp_random", f"L{layer} {mlp.kind} decode input logical rows unchanged", unchanged=same)
                if not same:
                    failures.append(f"L{layer} forward_decode modified its input's logical rows")
            y = mlp.forward_decode(xd, all_reduce=False)
            s = _check_decode(f"L{layer} {wname} decode partial", y, want_dec32, mesh_device, cfg, thr, failures,
                              partial=True)
            _free(y)
            record("mlp_random", f"L{layer} {mlp.kind} {wname} decode partials (host TP sum)", **s)
            y = mlp.forward_prefill(xp)
            s = _check_prefill(f"L{layer} {wname} prefill128", y, want_pre32, mesh_device, cfg, thr, failures)
            _free(y)
            record("mlp_random", f"L{layer} {mlp.kind} {wname} prefill S=128 vs fp32 ref", **s)
            if wname == "bfp8":
                _prefill_chunk_checks(mlp, ref32, layer, mesh_device, cfg, failures, gen2)
            mlp.deallocate()
        if layer >= 2:  # shared-expert alternative: replicated full gate, exact local moments (no moments CCL)
            alt = PolyNormMLP(mesh_device, cfg, layer, source=src, ccl=ccl, cache=False, stats="replicated_gate")
            y = alt.forward_decode(xd)
            s = _check_decode(f"L{layer} replicated_gate decode", y, want_dec32, mesh_device, cfg, 0.999, failures)
            _free(y)
            record("mlp_random", f"L{layer} shared stats=replicated_gate bfp8 decode vs fp32 ref", **s)
            y = alt.forward_prefill(xp)
            s = _check_prefill(f"L{layer} replicated_gate prefill128", y, want_pre32, mesh_device, cfg, 0.999, failures)
            _free(y)
            record("mlp_random", f"L{layer} shared stats=replicated_gate bfp8 prefill S=128 vs fp32 ref", **s)
            alt.deallocate()
        _free([xd, xp])
    _cache_roundtrip(mesh_device, cfg, ccl, failures, gen2)
    summary(r0)
    assert not failures, "\n".join(failures)


def _polynorm_alone(mlp, taps, cfg, mesh_device):
    """PolyNorm in isolation: the TT act (bf16) vs fp64 PolyNorm of the TT gate / up taps, worst DP row."""
    gt, ut, ht = (per_chip(taps[k], mesh_device) for k in ("gate", "up", "act"))
    a = cfg.axes
    c_ = mlp.coeffs
    out = []
    for dp in range(a.dp_size):
        rows = [a.coord(dp, tp) for tp in range(a.tp_size)]
        G = torch.cat([gt[r, c, 0, 0] for r, c in rows], dim=-1)
        U = torch.cat([ut[r, c, 0, 0] for r, c in rows], dim=-1)
        H = torch.cat([ht[r, c, 0, 0] for r, c in rows], dim=-1)
        out.append(stats(poly_golden(G, U, c_.c0, c_.c1, c_.c2, c_.b), H))
    return worst_of(out)


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mlp_device_real_weights(mesh_device, device_params):
    r0 = len(RESULTS)
    gold = load_real_goldens()
    layers = {i: gold["layers"][i] for i in (0, 1, 2)}
    gold4 = load_real_goldens(REAL_GOLDEN_L4, "--make-goldens-l4", required=False)
    cfg = _cfg(mesh_device)
    check_fabric(mesh_device, "mlp_real")
    ccl = MotifCCL(mesh_device, cfg)
    src = W.HFWeightLoader()
    failures = []
    if gold4 is not None:
        layers[4] = gold4["layers"][4]
    else:
        record("mlp_real", "L4", skipped="layer-4 goldens missing (test_mlp.py --make-goldens-l4)")
    for layer, g in layers.items():
        if not src.layer_available(layer):
            record("mlp_real", f"L{layer}", skipped="weights not local")
            continue
        f, ref32, ref16 = g["f"], g["ref_fp32"], g["ref_bf16"].float()
        base = stats(ref32, ref16)
        record("mlp_real", f"L{layer} reference bf16 (HF) vs fp32", pcc=base["pcc"], max_abs=base["max_abs"])
        dec = slice(N_PREFILL_TOKENS, N_PREFILL_TOKENS + N_DECODE_TOKENS)
        xd = _rows_input(f[dec], mesh_device, cfg)
        mlp = PolyNormMLP(mesh_device, cfg, layer, source=src, ccl=ccl, cache=False)
        kind = mlp.kind
        # ---- decode (8 lanes per DP row, rows differ) ------------------------------------------------------------
        y = mlp.forward_decode(xd)
        s = _check_decode(f"L{layer} decode", y, ref32[dec], mesh_device, cfg, 0.999, failures)
        record("mlp_real", f"L{layer} {kind} decode vs fp32 ref", **s)
        s16 = _check_decode(f"L{layer} decode vs bf16", y, ref16[dec], mesh_device, cfg, 0.999, failures)
        record("mlp_real", f"L{layer} {kind} decode vs bf16 (HF) ref", **s16)
        _free(y)
        if kind == "shared":
            y = mlp.forward_decode(xd, all_reduce=False)
            s = _check_decode(f"L{layer} decode partial", y, ref32[dec], mesh_device, cfg, 0.999, failures, partial=True)
            record("mlp_real", f"L{layer} shared decode partials (host TP sum)", **s)
            _free(y)
        # ---- PolyNorm in isolation: TT taps (gate / up fp32, act bf16) vs fp64 on the TT gate / up ---------------
        taps = {}
        y = mlp.forward_decode(xd, taps=taps)
        worst = _polynorm_alone(mlp, taps, cfg, mesh_device)
        _free([y, taps])
        record("mlp_real", f"L{layer} {kind} PolyNorm alone (TT gate/up -> TT act vs fp64)", **worst)
        if not passes(worst, 0.9995):
            failures.append(f"L{layer} PolyNorm alone: {fmt(worst)}")
        # ---- bf16-PolyNorm decode lever (MLP-1: gated on the outlier layers 4 / 51 / 52; informational) -----------
        mlp.decode_polynorm = "bf16"
        try:
            lever = []
            y = mlp.forward_decode(xd)
            sb = _check_decode(f"L{layer} decode bf16 PolyNorm", y, ref32[dec], mesh_device, cfg, 0.999, lever)
            _free(y)
            taps = {}
            y = mlp.forward_decode(xd, taps=taps)
            wb = _polynorm_alone(mlp, taps, cfg, mesh_device)
            _free([y, taps])
            ok = not lever and passes(wb, 0.9995)
            record("mlp_real", f"L{layer} {kind} decode with bf16 PolyNorm (lever) vs fp32 ref", **sb,
                   polynorm_alone_pcc=wb["pcc"], polynorm_alone_max_abs=wb["max_abs"],
                   lever=("meets 0.999 / 0.9995" if ok else "BELOW threshold: keep fp32"))
        finally:
            mlp.decode_polynorm = "fp32"
        # ---- latency (decode eager + traced) --------------------------------------------------------------------
        fn = lambda: mlp.forward_decode(xd)  # noqa: E731
        t_e = eager_us(mesh_device, fn, iters=10)
        t_t = traced_us(mesh_device, fn)
        record("mlp_real", f"L{layer} {kind} decode latency", eager_us=t_e, traced_us=t_t)
        if kind == "shared":
            fnp = lambda: mlp.forward_decode(xd, all_reduce=False)  # noqa: E731
            record("mlp_real", f"L{layer} shared decode latency (partial, no output AR)",
                   eager_us=eager_us(mesh_device, fnp, iters=10), traced_us=traced_us(mesh_device, fnp))
        # ---- trace replay with new inputs (trace safety) ----------------------------------------------------------
        y = mlp.forward_decode(xd)
        _free(y)
        ttnn.synchronize_device(mesh_device)
        with _Capture(mesh_device) as cap:
            y = mlp.forward_decode(xd)
        try:
            for it, sl in enumerate((slice(0, 32), slice(32, 64))):
                host = _rows_input(f[sl], mesh_device, cfg, device=False)
                ttnn.copy_host_to_device_tensor(host, xd)
                ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                s = _check_decode(f"L{layer} trace replay {it}", y, ref32[sl], mesh_device, cfg, 0.999, failures)
                record("mlp_real", f"L{layer} {kind} trace replay #{it} (new lanes) vs fp32 ref", **s)
        finally:
            ttnn.release_trace(mesh_device, cap.tid)
        _free([y, xd])
        # ---- prefill S = 128 / 1024 ------------------------------------------------------------------------------
        for S in (128, 1024):
            xp = to_mesh(f[:S].reshape(1, 1, S, 4096), mesh_device, cfg)
            y = mlp.forward_prefill(xp)
            s = _check_prefill(f"L{layer} prefill{S}", y, ref32[:S], mesh_device, cfg, 0.999, failures)
            _free(y)
            t_e = eager_us(mesh_device, lambda: mlp.forward_prefill(xp), iters=3)
            record("mlp_real", f"L{layer} {kind} prefill S={S} vs fp32 ref", **s, eager_us=t_e)
            if S == 1024:
                mlp.prefill_polynorm = "bf16"
                y = mlp.forward_prefill(xp)
                s = _check_prefill(f"L{layer} prefill{S} bf16pn", y, ref32[:S], mesh_device, cfg, 0.999, failures)
                _free(y)
                t_e = eager_us(mesh_device, lambda: mlp.forward_prefill(xp), iters=3)
                record("mlp_real", f"L{layer} {kind} prefill S={S} bf16 PolyNorm vs fp32 ref", **s, eager_us=t_e)
                mlp.prefill_polynorm = "fp32"
                # moments all-reduce on ttnn.all_reduce's reduce-scatter path (TF32-class fp32 adds) instead of ag_sum
                mlp.pn_exact_ar = False
                mlp.pn_kw["ar"] = "all_reduce"
                y = mlp.forward_prefill(xp)
                s = _check_prefill(f"L{layer} prefill{S} tf32-ar", y, ref32[:S], mesh_device, cfg, 0.999, failures)
                _free(y)
                t_e = eager_us(mesh_device, lambda: mlp.forward_prefill(xp), iters=3)
                record("mlp_real", f"L{layer} {kind} prefill S={S} moments AR on the RS path (TF32 adds)", **s,
                       eager_us=t_e)
                mlp.pn_exact_ar = True
                mlp.pn_kw["ar"] = "ag_sum"
            _free(xp)
        mlp.deallocate()
        # ---- fp32-faithful check: bf16 weights (no bfp8 quantization) ---------------------------------------------
        if layer in (0, 2, 4):
            m16 = PolyNormMLP(mesh_device, cfg, layer, source=src, ccl=ccl, cache=False, weight_dtype=ttnn.bfloat16)
            xd = _rows_input(f[dec], mesh_device, cfg)
            y = m16.forward_decode(xd)
            s = _check_decode(f"L{layer} bf16-weights decode", y, ref32[dec], mesh_device, cfg, 0.9999, failures)
            record("mlp_real", f"L{layer} {kind} bf16 weights decode vs fp32 ref (fp32-faithful mode)", **s)
            _free([y, xd])
            m16.deallocate()
        # ---- shared expert alternative: replicated full gate -> exact local moments, no moments all-reduce --------
        if layer == 2:
            alt = PolyNormMLP(mesh_device, cfg, layer, source=src, ccl=ccl, cache=False, stats="replicated_gate")
            xd = _rows_input(f[dec], mesh_device, cfg)
            for ar in (True, False):
                y = alt.forward_decode(xd, all_reduce=ar)
                s = _check_decode(f"L{layer} replicated_gate ar={ar}", y, ref32[dec], mesh_device, cfg, 0.999,
                                  failures, partial=not ar)
                _free(y)
                fn = lambda ar=ar: alt.forward_decode(xd, all_reduce=ar)  # noqa: E731
                record("mlp_real", f"L{layer} shared stats=replicated_gate decode all_reduce={ar}", **s,
                       eager_us=eager_us(mesh_device, fn, iters=10), traced_us=traced_us(mesh_device, fn))
            xp = to_mesh(f[:128].reshape(1, 1, 128, 4096), mesh_device, cfg)
            y = alt.forward_prefill(xp)
            s = _check_prefill(f"L{layer} replicated_gate prefill128", y, ref32[:128], mesh_device, cfg, 0.999, failures)
            record("mlp_real", f"L{layer} shared stats=replicated_gate prefill S=128", **s)
            _free([y, xp, xd])
            alt.deallocate()
    summary(r0)
    assert not failures, "\n".join(failures)


REAL_MOE_INPUTS = PROJECT_ROOT / "tt_cache" / "test" / "moe" / "real_router_inputs_v1.pt"  # tests/unit/test_moe.py


def _shared_from_tt_cache(mesh_device, cfg, ccl, layer, **kw):
    """The shared expert of MoE layer ``layer`` from the serving TT cache alone (a raising source, nothing written);
    ``None`` when the layer's cache part is not converted."""
    from models.demos.motif3.tt.model import layer_cache_complete, read_only_cache

    if not layer_cache_complete(cfg, layer):
        return None
    with read_only_cache() as misses:
        mlp = PolyNormMLP(mesh_device, cfg, layer, source=_NoReadSource(), ccl=ccl, cache=True, kind="shared", **kw)
    assert misses == [], f"tensors missing from the TT cache: {misses}"
    return mlp


def _rows16_input(x64, mesh_device, cfg, device=True):
    """T64 shape (``tests/unit/test_moe.py`` ``upload_rows16``): anchors ``x64[:32]``, drafts ``x64[32:]`` (lane order)
    -> per DP row ``[1, 1, 16, 4096]`` = ``[the row's 8 anchors | their 8 drafts]``."""
    L = cfg.lanes_per_row
    a = x64[:32].reshape(cfg.dp, L, -1)
    d = x64[32:].reshape(cfg.dp, L, -1)
    rows = torch.cat([a, d], dim=1).reshape(cfg.dp, 1, 2 * L, -1)
    return to_mesh(rows.bfloat16(), mesh_device, cfg, dtype=ttnn.bfloat16, dp_dim=0, device=device)


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@torch.no_grad()
def test_mlp_device_fused_shared_polynorm(mesh_device, device_params):
    """B5 (``shared_polynorm="fused"``, ``tt/kernels/shared_polynorm.py``; docs/OPTIMIZATION_PLAN.md §3.3), the shared
    experts of every MoE layer from the serving TT cache, real MoE inputs (``tests/unit/test_moe.py`` capture; layers
    without a capture take the nearest captured layer's), every chip read back:

    * **bitwise == the composite** (the release decode path): ``forward_decode`` with and without the TP all-reduce,
      on two 32-lane sets and one 64-row (T64-shaped, 16 rows per DP row) set, every layer 2..52 (env
      ``MOTIF3_B5_LAYERS`` = a comma list narrows it; ``MOTIF3_B5_SETS`` = extra 32-lane sets per layer, a soak); the
      PolyNorm output ``h`` (``taps["act"]``) bitwise too (the first build packed fp32 -> bf16 directly and differed
      from the release's RNE typecast on exact bf16 ties: 1 value in ~1e6);
    * layer 2: two fused calls bitwise equal; the 16-row call's rows == the two 8-lane calls' rows (bitwise); trace
      capture with persistent ``x``, replays with new lanes == eager bitwise, two replays equal (no compile after
      capture: a compile inside the capture would fail it);
    * traced cost (informational; asserted fused < composite): the PolyNorm alone and ``forward_decode`` (partial)."""
    if not REAL_MOE_INPUTS.is_file():
        pytest.skip(f"{REAL_MOE_INPUTS} missing (tests/unit/test_moe.py capture)")
    data = torch.load(REAL_MOE_INPUTS, weights_only=True)
    cfg = _cfg(mesh_device)
    fab = check_fabric(mesh_device, "mlp_b5")
    assert str(fab.get("committed")) == "TORUS_XY", fab
    ccl = MotifCCL(mesh_device, cfg)
    L1 = ttnn.L1_MEMORY_CONFIG
    env = os.environ.get("MOTIF3_B5_LAYERS", "").strip()
    layers = [int(v) for v in env.split(",")] if env else [l for l in range(cfg.num_layers) if cfg.layer(l).is_moe]
    gen = torch.Generator().manual_seed(55)
    failures, checked, skipped = [], [], []

    def chips(t):
        return [ttnn.to_torch(c) for c in ttnn.get_device_tensors(t)]

    def same(a, b):
        return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))

    def run(mlp, x, fused, **kw):
        mlp.pn_fused = fused
        y = mlp.forward_decode(x, **kw)
        out = chips(y)
        _free(y)
        return out

    for layer in layers:
        mlp = _shared_from_tt_cache(mesh_device, cfg, ccl, layer, shared_polynorm="fused")
        if mlp is None:
            skipped.append(layer)
            continue
        assert mlp.shared_polynorm == "fused" and mlp.pn_fused is not None
        fused = mlp.pn_fused
        src_l = min(data["layers"], key=lambda l: abs(l - layer))
        xs = data["layers"][src_l]["x"].float()
        pick = xs[torch.randperm(xs.shape[0], generator=gen)[:64]]
        inputs = [("lanes A", _rows_input(pick[:32], mesh_device, cfg)), ("lanes B", _rows_input(pick[32:], mesh_device, cfg)),
                  ("rows16", _rows16_input(pick, mesh_device, cfg))]
        for k in range(int(os.environ.get("MOTIF3_B5_SETS", "0") or 0)):
            extra = xs[torch.randperm(xs.shape[0], generator=gen)[:32]]
            inputs.append((f"extra {k}", _rows_input(extra, mesh_device, cfg)))
        bad = []
        outs = {}
        for name, x in inputs:
            for ar in (False, True):
                yc, yf = run(mlp, x, None, all_reduce=ar), run(mlp, x, fused, all_reduce=ar)
                if not same(yc, yf):
                    nd = sum(int((a != b).sum()) for a, b in zip(yc, yf))
                    bad.append(f"{name} all_reduce={ar}: {nd} values differ")
                outs[(name, ar)] = yf
            tc, tf = {}, {}
            mlp.pn_fused = None
            _free([mlp.forward_decode(x, taps=tc)])
            mlp.pn_fused = fused
            _free([mlp.forward_decode(x, taps=tf)])
            if not same(chips(tc["act"]), chips(tf["act"])):
                bad.append(f"{name}: PolyNorm h differs")
            _free([tc, tf])
        checked.append(layer)
        print(f"[mlp] B5 L{layer} (inputs of L{src_l}): fused == composite bitwise "
              f"{'yes' if not bad else 'NO: ' + '; '.join(bad)}", flush=True)
        failures += [f"L{layer} {b}" for b in bad]
        if layer == layers[0]:
            # determinism, T64 rows == T32 rows
            again = run(mlp, inputs[0][1], fused, all_reduce=False)
            det = same(again, outs[("lanes A", False)])
            L = cfg.lanes_per_row
            r16 = outs[("rows16", False)]
            rows_eq = all(torch.equal(c16[..., :L, :], ca) and torch.equal(c16[..., L:, :], cb)
                          for c16, ca, cb in zip(r16, outs[("lanes A", False)], outs[("lanes B", False)]))
            print(f"[mlp] B5 L{layer}: 2 fused calls bitwise {det}; 16-row call rows == 8-lane calls {rows_eq}", flush=True)
            if not (det and rows_eq):
                failures.append(f"L{layer} determinism {det} / T64 rows {rows_eq}")
            # trace: persistent x, new lanes per replay
            x_dev = inputs[0][1]
            mlp.pn_fused = fused
            _free(mlp.forward_decode(x_dev, all_reduce=False))
            ttnn.synchronize_device(mesh_device)
            with _Capture(mesh_device) as cap:
                y_t = mlp.forward_decode(x_dev, all_reduce=False)
            try:
                for it in range(3):
                    xn = xs[torch.randperm(xs.shape[0], generator=gen)[:32]]
                    ttnn.copy_host_to_device_tensor(_rows_input(xn, mesh_device, cfg, device=False), x_dev)
                    ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                    t1 = chips(y_t)
                    ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                    t2 = chips(y_t)
                    te = run(mlp, x_dev, fused, all_reduce=False)
                    tcmp = run(mlp, x_dev, None, all_reduce=False)
                    ok = same(t1, te) and same(t1, t2) and same(t1, tcmp)
                    print(f"[mlp] B5 L{layer} trace replay {it}: traced == eager {same(t1, te)}, 2 replays equal "
                          f"{same(t1, t2)}, == composite {same(t1, tcmp)}", flush=True)
                    if not ok:
                        failures.append(f"L{layer} trace replay {it}")
            finally:
                ttnn.release_trace(mesh_device, cap.tid)
                _free(y_t)
            # traced cost
            gu = ttnn.linear(ttnn.pad(x_dev, [(0, 0), (0, 0), (0, TILE - L), (0, 0)], 0.0, memory_config=L1),
                             mlp.w_gate_up, dtype=ttnn.float32, memory_config=L1, compute_kernel_config=mlp.ckc_mm,
                             program_config=mlp.decode_pc.get("gate_up"))
            n = mlp.n_local

            def comp():
                a = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, TILE, n], memory_config=L1)
                b = ttnn.slice(gu, [0, 0, 0, n], [1, 1, TILE, 2 * n], memory_config=L1)
                h = polynorm_tp(a, b, mlp.pn, ccl=ccl, mode="fp32", memory_config=L1, compute_kernel_config=mlp.ckc_pn,
                                exact_ar=True, **mlp.pn_kw)
                _free([a, b])
                return h

            t_pc, t_pf = traced_us(mesh_device, comp), traced_us(mesh_device, lambda: fused(gu, memory_config=L1))
            mlp.pn_fused = None
            t_mc = traced_us(mesh_device, lambda: mlp.forward_decode(x_dev, all_reduce=False))
            mlp.pn_fused = fused
            t_mf = traced_us(mesh_device, lambda: mlp.forward_decode(x_dev, all_reduce=False))
            _free(gu)
            record("mlp_b5", f"L{layer} traced", polynorm_composite_us=t_pc, polynorm_fused_us=t_pf,
                   forward_partial_composite_us=t_mc, forward_partial_fused_us=t_mf)
            print(f"[mlp] B5 L{layer} traced: PolyNorm composite {t_pc:.1f} -> fused {t_pf:.1f} us; forward_decode "
                  f"(partial) {t_mc:.1f} -> {t_mf:.1f} us", flush=True)
            if not (t_pf < t_pc and t_mf < t_mc):
                failures.append(f"L{layer} fused not faster: {t_pf:.1f} / {t_pc:.1f}, {t_mf:.1f} / {t_mc:.1f}")
        _free([x for _, x in inputs])
        mlp.deallocate()
    print(f"[mlp] B5 summary: {len(checked)} layers bitwise-checked {checked}; not converted {skipped}; "
          f"{len(failures)} failures", flush=True)
    assert checked, "no layer converted in the TT cache"
    assert not failures, "\n".join(failures)


# ============================================================================================================
# device: opt-in sweep (PolyNorm variants, decode matmul program configs)
# ============================================================================================================
@pytest.mark.timeout(1500)
@pytest.mark.skipif(os.environ.get("MOTIF3_MLP_SWEEP", "0") in ("0", ""), reason="opt-in: MOTIF3_MLP_SWEEP=1")
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mlp_device_sweep(mesh_device, device_params):
    r0 = len(RESULTS)
    cfg = _cfg(mesh_device)
    check_fabric(mesh_device, "mlp_sweep")
    gen = torch.Generator().manual_seed(5)
    ckc = cfg.compute_config("dense_mlp")
    shapes = {
        ("dense", "gate_up"): (4096, 3072),
        ("dense", "down"): (1536, 4096),
        ("shared", "gate_up"): (4096, 320),
        ("shared", "down"): (160, 4096),
    }
    cands = {
        ("dense", "gate_up"): [None, ((12, 8), 8), ((12, 8), 16), ((12, 4), 8), ((8, 6), 8), ((8, 4), 8), ((12, 2), 8)],
        ("dense", "down"): [None, ((8, 4), 8), ((8, 4), 4), ((8, 8), 8), ((12, 6), 8), ((11, 3), 8), ((8, 2), 8)],
        ("shared", "gate_up"): [None, ((10, 1), 8), ((10, 1), 16), ((5, 1), 8), ((2, 1), 8)],
        ("shared", "down"): [None, ((8, 4), 5), ((8, 8), 5), ((12, 6), 5), ((8, 2), 5), ((8, 4), 1)],
    }
    for key, (k, n) in shapes.items():
        w = torch.randn(k, n, generator=gen) * k**-0.5
        x = torch.randn(1, 1, 32, k, generator=gen).bfloat16()
        w_t = to_mesh(w.reshape(1, 1, k, n), mesh_device, cfg, dtype=ttnn.bfloat8_b)
        x_t = to_mesh(x, mesh_device, cfg, dtype=ttnn.bfloat16)
        wq = ttnn.to_torch(ttnn.get_device_tensors(w_t)[0]).float().reshape(k, n)
        want = x.float().reshape(32, k).double() @ wq.double()
        for cand in cands[key]:
            for out_dt in ((ttnn.float32, ttnn.bfloat16) if key[1] == "gate_up" else (ttnn.bfloat16,)):
                case = f"{key[0]}.{key[1]} [{k}x{n}] {'auto' if cand is None else f'grid={cand[0]} bw={cand[1]}'} out={out_dt.name}"
                try:
                    pc = None if cand is None else decode_matmul_pc(k, n, cand[0], in0_block_w=cand[1])
                    fn = lambda: ttnn.linear(x_t, w_t, dtype=out_dt, compute_kernel_config=ckc, program_config=pc,
                                             memory_config=ttnn.L1_MEMORY_CONFIG)
                    o = fn()
                    got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(32, n)
                    _free(o)
                    s = stats(want, got)
                    us = traced_us(mesh_device, fn)
                    mb = k * n * 1088 / 1024 / 1e6
                    record("sweep", case, pcc=s["pcc"], traced_us=us, GBps=mb / max(us, 1e-3) * 1e3)
                except Exception as e:
                    record("sweep", case, error=f"{type(e).__name__}: {str(e)[:200]}")
        _free([w_t, x_t])

    # ---- PolyNorm TP decode chain pieces (where the time goes) --------------------------------------------------
    ccl = MotifCCL(mesh_device, cfg)
    for inter in (12288, 1280):
        n = inter // 8
        coeffs = PolyNormCoefficients.from_tensors(torch.tensor([-0.5, 0.1, 0.2]), torch.tensor([0.05]))
        consts = ScalarPolyNormConsts(mesh_device, cfg, coeffs, inter=inter)
        G = torch.randn(4, 1, 8, inter, generator=gen)
        g_t = to_mesh(G, mesh_device, cfg, dtype=ttnn.float32, dp_dim=0, tp_dim=3)
        u_t = to_mesh(G, mesh_device, cfg, dtype=ttnn.float32, dp_dim=0, tp_dim=3)
        L1 = ttnn.L1_MEMORY_CONFIG
        ckc_pn = cfg.compute_config("polynorm")
        s_t = ttnn.sum(g_t, dim=-1, keepdim=True, compute_kernel_config=ckc_pn)
        m_t = ttnn.concat([s_t, s_t, s_t], dim=1)
        pieces = {
            "mul fp32 [1,1,8,n]": lambda: ttnn.multiply(g_t, g_t, memory_config=L1),
            "sum fp32 [1,1,8,n] (accurate)": lambda: ttnn.sum(g_t, dim=-1, keepdim=True, memory_config=L1,
                                                             compute_kernel_config=ckc_pn),
            "mac col-bcast [1,1,8,n]": lambda: ttnn.mac(g_t, s_t, s_t, memory_config=L1),
            "ar_tp moments [1,3,8,1] fp32": lambda: ccl.ar_tp(m_t, memory_config=L1),
            "slice [1,3,8,1] -> [1,1,8,1]": lambda: ttnn.slice(m_t, [0, 1, 0, 0], [1, 2, 8, 1], memory_config=L1),
            "full polynorm_tp fp32 L1": lambda: polynorm_tp(g_t, u_t, consts, ccl=ccl, mode="fp32", memory_config=L1,
                                                            compute_kernel_config=ckc_pn),
            "full polynorm_tp fp32 L1, concat_full=False": lambda: polynorm_tp(
                g_t, u_t, consts, ccl=ccl, mode="fp32", memory_config=L1, compute_kernel_config=ckc_pn,
                concat_full=False),
        }
        for name, fn in pieces.items():
            try:
                record("sweep", f"polynorm piece I={inter} n={n}: {name}", traced_us=traced_us(mesh_device, fn))
            except Exception as e:
                record("sweep", f"polynorm piece I={inter}: {name}", error=f"{type(e).__name__}: {str(e)[:200]}")
        _free([g_t, u_t, s_t, m_t])
        consts.deallocate()
    summary(r0)


# ============================================================================================================
# device: opt-in variant benchmark v2 (traced latency + accuracy of the implementation knobs)
# ============================================================================================================
@pytest.mark.timeout(1500)
@pytest.mark.skipif(os.environ.get("MOTIF3_MLP_VARIANTS", "0") in ("0", ""), reason="opt-in: MOTIF3_MLP_VARIANTS=1")
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mlp_device_variants(mesh_device, device_params):
    r0 = len(RESULTS)
    cfg = _cfg(mesh_device)
    check_fabric(mesh_device, "mlp_variants")
    ccl = MotifCCL(mesh_device, cfg)
    a = cfg.axes
    R, C = a.mesh_shape
    gen = torch.Generator().manual_seed(21)
    ckc = cfg.compute_config("polynorm")
    L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    only = os.environ.get("MOTIF3_MLP_VARIANTS_ONLY", "")  # comma list of groups: tp,grouped,matmul,module

    def want(group):
        return not only or group in only.split(",")

    # ---- scalar TP PolyNorm composition variants ------------------------------------------------------------
    if want("tp"):
        coeffs = PolyNormCoefficients.from_tensors(torch.tensor([-0.5, 0.1, 0.2]), torch.tensor([0.05]))
        for inter in (12288, 1280):
            consts = ScalarPolyNormConsts(mesh_device, cfg, coeffs, inter=inter)
            n = inter // a.tp_size
            for T in (8, 32):
                G = _heavy_tailed((a.dp_size, 1, T, inter), gen, 1.5)
                U = _heavy_tailed((a.dp_size, 1, T, inter), gen, 1.0)
                ref = poly_golden(G, U, coeffs.c0, coeffs.c1, coeffs.c2, coeffs.b)
                g_t = to_mesh(G, mesh_device, cfg, dtype=ttnn.float32, dp_dim=0, tp_dim=3)
                u_t = to_mesh(U, mesh_device, cfg, dtype=ttnn.float32, dp_dim=0, tp_dim=3)
                for horner in ("mac", "binary"):
                    for ar in ("all_reduce", "ag_sum"):
                        for moments in (("sum", "sum_fast") if (horner, ar) == ("binary", "ag_sum") else ("sum",)):
                            case = f"tp I={inter} T={T} horner={horner} ar={ar} moments={moments} fp32 L1"
                            try:
                                fn = lambda: polynorm_tp(g_t, u_t, consts, ccl=ccl, mode="fp32", memory_config=L1,  # noqa: E731
                                                         compute_kernel_config=ckc, horner=horner, ar=ar,
                                                         moments=moments)
                                h = fn()
                                got = per_chip(h, mesh_device)
                                _free(h)
                                worst = _tp_worst(ref, got, a, n)
                                record("variants", case, pcc=worst["pcc"], max_abs=worst["max_abs"],
                                       traced_us=traced_us(mesh_device, fn))
                            except Exception as e:
                                record("variants", case, error=f"{type(e).__name__}: {str(e)[:300]}")
                _free([g_t, u_t])
            consts.deallocate()

    # ---- grouped PolyNorm in L1 ------------------------------------------------------------------------------
    if want("grouped"):
        cw = torch.sigmoid(torch.randn(cfg.num_experts, 3, generator=gen))
        cb = (torch.rand(cfg.num_experts, generator=gen) - 0.5).clamp(-0.5, 0.5)
        gconsts = GroupedPolyNormConsts.from_coefficients(mesh_device, cfg, cw, cb)
        I = cfg.moe_intermediate_size
        GU = torch.cat([_heavy_tailed((a.dp_size, 96, 32, I), gen, 4.0), _heavy_tailed((a.dp_size, 96, 32, I), gen, 1.5)],
                       dim=-1).bfloat16().float()
        c_e, b_e = cw.reshape(a.dp_size, -1, 1, 1, 3), cb.reshape(a.dp_size, -1, 1, 1)
        ref = poly_golden(GU[..., :I], GU[..., I:], c_e[..., 0], c_e[..., 1], c_e[..., 2], b_e)
        for in_mc_name, in_mc in (("L1", L1), ("DRAM", DRAM)):
            gu_t = ttnn.to_memory_config(to_mesh(GU, mesh_device, cfg, dtype=ttnn.bfloat16, dp_dim=0, tp_dim=1), in_mc)
            for impl, mode, horner, mc_name in (("horner", "fp32", "mac", "L1"), ("horner", "fp32", "binary", "L1"),
                                                ("horner", "bf16", "binary", "L1"), ("rms", "fp32", "-", "L1"),
                                                ("rms", "bf16", "-", "L1"), ("rms", "bf16", "-", "DRAM"),
                                                ("horner", "fp32", "binary", "DRAM")):
                if in_mc_name == "DRAM" and mc_name == "L1" and impl == "rms" and mode == "fp32":
                    continue
                mc = L1 if mc_name == "L1" else DRAM
                case = f"grouped in={in_mc_name} impl={impl} mode={mode} horner={horner} intermediates={mc_name}"
                try:
                    kw = dict(horner=horner) if impl == "horner" else {}
                    fn = lambda: grouped_polynorm(gu_t, gconsts, inter=I, mode=mode, impl=impl, memory_config=mc,  # noqa: E731
                                                  compute_kernel_config=ckc, **kw)
                    h = fn()
                    got = per_chip(h, mesh_device)
                    _free(h)
                    worst = worst_of([stats(ref[a.roles(r, c)[0], 12 * a.roles(r, c)[1]:12 * a.roles(r, c)[1] + 12],
                                            got[r, c, 0]) for r in range(R) for c in range(C)])
                    record("variants", case, pcc=worst["pcc"], max_abs=worst["max_abs"], traced_us=traced_us(mesh_device, fn))
                except Exception as e:
                    record("variants", case, error=f"{type(e).__name__}: {str(e)[:300]}")
            _free(gu_t)
        gconsts.deallocate()

    # ---- decode matmul configs (traced) --------------------------------------------------------------------
    if want("matmul"):
        ckm = cfg.compute_config("dense_mlp")
        cands = {
            (4096, 3072): [None, ((12, 8), 8), ((12, 8), 16), ((12, 2), 8), ((12, 2), 16), ((12, 4), 32)],
            (1536, 4096): [None, ((8, 4), 8), ((8, 4), 12), ((8, 4), 16), ((8, 8), 8), ((8, 8), 16)],
            (4096, 320): [None, ((10, 1), 8), ((10, 1), 16), ((10, 1), 32), ((10, 1), 64), ((5, 1), 32)],
            (160, 4096): [None, ((8, 4), 5), ((8, 8), 5), ((8, 8), 1), ((4, 8), 5)],
            (4096, 1280): [None, ((10, 4), 8), ((10, 4), 16), ((10, 2), 16), ((8, 5), 16)],
            (4096, 160): [None, ((5, 1), 16), ((5, 1), 32)],
        }
        for (k, n), cl in cands.items():
            w_t = to_mesh(torch.randn(1, 1, k, n, generator=gen) * k**-0.5, mesh_device, cfg, dtype=ttnn.bfloat8_b)
            x_t = to_mesh(torch.randn(1, 1, 8, k, generator=gen), mesh_device, cfg)
            for c in cl:
                for out_dt in ((ttnn.float32,) if n != 4096 else (ttnn.bfloat16,)):
                    case = f"matmul [{k}x{n}] {'auto' if c is None else f'grid={c[0]} bw={c[1]}'} out={out_dt.name}"
                    try:
                        pc = None if c is None else decode_matmul_pc(k, n, c[0], in0_block_w=c[1])
                        fn = lambda: ttnn.linear(x_t, w_t, dtype=out_dt, compute_kernel_config=ckm, program_config=pc,  # noqa: E731
                                                 memory_config=L1)
                        us = traced_us(mesh_device, fn)
                        record("variants", case, traced_us=us, GBps=k * n * 1088 / 1024 / max(us, 1e-3) / 1e3)
                    except Exception as e:
                        record("variants", case, error=f"{type(e).__name__}: {str(e)[:200]}")
            _free([w_t, x_t])

    # ---- module-level decode variants (random weights) -------------------------------------------------------
    if want("module"):
        for layer, inter in ((0, 12288), (2, 1280)):
            src = _random_mlp_source(layer, inter, seed=500 + layer)
            ref32 = _reference_mlp(src, _ffn_prefix(layer), inter, torch.float32)
            x = torch.randn(32, 4096, generator=gen).bfloat16()
            want_y = ref32(x.float())
            xd = _rows_input(x, mesh_device, cfg)
            opts = [("default (pad rows, mac Horner, ag_sum moments AR)", {}),
                    ("no row pad", dict(pad_decode_rows=False)),
                    ("ar=all_reduce", dict(polynorm_options=dict(ar="all_reduce"))),
                    ("v0: no row pad + ar=all_reduce", dict(pad_decode_rows=False, polynorm_options=dict(ar="all_reduce"))),
                    ("L1 off (DRAM intermediates)", dict(intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG))]
            if layer >= 2:
                opts += [("stats=replicated_gate", dict(stats="replicated_gate")),
                         ("stats=replicated_gate, no row pad", dict(stats="replicated_gate", pad_decode_rows=False))]
            _free(xd)
            for name, kw in opts:
                case = f"module L{layer} decode {name}"
                mlp = None
                xd = _rows_input(x, mesh_device, cfg)  # fresh input per variant (a failing variant cannot poison it)
                try:
                    mlp = PolyNormMLP(mesh_device, cfg, layer, source=src, ccl=ccl, cache=False, **kw)
                    fails = []
                    y = mlp.forward_decode(xd)
                    s_ = _check_decode(case, y, want_y, mesh_device, cfg, 0.999, fails)
                    _free(y)
                    fn = lambda: mlp.forward_decode(xd)  # noqa: E731
                    t_full = traced_us(mesh_device, fn)
                    fnp = lambda: mlp.forward_decode(xd, all_reduce=False)  # noqa: E731
                    t_part = traced_us(mesh_device, fnp)
                    record("variants", case, pcc=s_["pcc"], max_abs=s_["max_abs"], traced_us=t_full,
                           traced_partial_us=t_part, failures=len(fails))
                except Exception as e:
                    record("variants", case, error=f"{type(e).__name__}: {str(e)[:300]}")
                finally:
                    if mlp is not None:
                        mlp.deallocate()
                    _free(xd)
    summary(r0)


# ============================================================================================================
# device: opt-in op profile (python -m tracy ...; per-op DEVICE KERNEL DURATION from ops_perf_results*.csv)
# ============================================================================================================
MARKER_BASE_W = 3200  # marker op k = ttnn.add on [1, 1, 32, 3200 + 32 k] (a width no real op has)


def _marker(mesh_device, cfg, k):
    t = to_mesh(torch.zeros(1, 1, 32, MARKER_BASE_W + 32 * k), mesh_device, cfg)
    _free(ttnn.add(t, 0.0))
    _free(t)


def profile_variants(mesh_device, cfg, variants, reps=2):
    """Run every ``(name, fn)`` ``reps`` times eagerly, each run preceded by marker op ``k``; flush the device profiler
    between variants. Returns the marker index map for :func:`summarize_profile`."""
    names = {}
    for k, (name, fn) in enumerate(variants):
        names[k] = name
        try:
            for _ in range(reps):
                _marker(mesh_device, cfg, k)
                _free(fn())
            ttnn.synchronize_device(mesh_device)
        except Exception as e:  # one broken variant must not abort the whole profile
            names[k] = f"{name} [ERROR {type(e).__name__}: {str(e)[:120]}]"
            record("profile", name, error=f"{type(e).__name__}: {str(e)[:200]}")
        try:
            ttnn.ReadDeviceProfiler(mesh_device)
        except Exception:
            pass
    return names


def summarize_profile(csv_path, names=None, device_id=0):
    """Per-variant op table from an ``ops_perf_results*.csv``: the ops between the last two markers of each variant
    (device ``device_id``), with DEVICE KERNEL DURATION and core count; returns ``{variant: (total_us, rows)}``."""
    import csv

    rows = [r for r in csv.DictReader(open(csv_path)) if str(r.get("DEVICE ID")) == str(device_id)]
    rows.sort(key=lambda r: int(r.get("GLOBAL CALL COUNT") or 0))
    out, cur, seq = {}, None, []

    def width(r):
        try:
            return int(str(r.get("INPUT_0_X_PAD[LOGICAL]", "0")).split("[")[0])
        except ValueError:
            return 0

    def flush():
        if cur is not None and seq:
            out.setdefault(cur, []).append(list(seq))

    for r in rows:
        w = width(r)
        if r["OP CODE"].startswith("BinaryNg") and w >= MARKER_BASE_W and (w - MARKER_BASE_W) % 32 == 0:
            flush()
            cur, seq = (w - MARKER_BASE_W) // 32, []
            continue
        if cur is not None:
            seq.append(r)
    flush()
    res = {}
    for k, runs in sorted(out.items()):
        last = runs[-1]
        tot = sum(float(r.get("DEVICE KERNEL DURATION [ns]") or 0) for r in last) / 1e3
        name = (names or {}).get(k, f"variant {k}")
        res[name] = (tot, last)
        print(f"== {name}: {len(last)} ops, device kernel total {tot:.1f} us")
        for r in last:
            shp = "x".join(str(r.get(f"INPUT_0_{d}_PAD[LOGICAL]", "")).split("[")[0] for d in "WZYX")
            print(f"   {r['OP CODE'][:42]:42s} cores={r.get('CORE COUNT', ''):>4s} in0={shp:18s} "
                  f"{float(r.get('DEVICE KERNEL DURATION [ns]') or 0) / 1e3:8.1f} us")
    return res


@pytest.mark.timeout(1500)
@pytest.mark.skipif(os.environ.get("MOTIF3_MLP_PROFILE", "0") in ("0", ""), reason="opt-in: MOTIF3_MLP_PROFILE=1 under tracy")
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_mlp_device_profile(mesh_device, device_params):
    """Variants of the TP / grouped PolyNorm and the decode matmuls, one marker-bracketed eager call each (run under
    ``python -m tracy -r -p -v --no-web-server -o <dir> -m pytest ... -k profile``; then
    ``python test_mlp.py --profile-summary <dir>/reports/<ts>/ops_perf_results_<ts>.csv``). A failing variant is
    recorded and skipped (the others still run); the test fails at the end if any variant errored."""
    cfg = _cfg(mesh_device)
    check_fabric(mesh_device, "mlp_profile")
    ccl = MotifCCL(mesh_device, cfg)
    gen = torch.Generator().manual_seed(11)
    ckc = cfg.compute_config("polynorm")
    L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    variants = []
    keep = []
    coeffs = PolyNormCoefficients.from_tensors(torch.tensor([-0.5, 0.1, 0.2]), torch.tensor([0.05]))
    for inter in (12288, 1280):
        consts = ScalarPolyNormConsts(mesh_device, cfg, coeffs, inter=inter)
        G = torch.randn(4, 1, 8, inter, generator=gen)
        g_t = to_mesh(G, mesh_device, cfg, dtype=ttnn.float32, dp_dim=0, tp_dim=3)
        u_t = to_mesh(G, mesh_device, cfg, dtype=ttnn.float32, dp_dim=0, tp_dim=3)
        keep += [consts, g_t, u_t]
        for mom in ("sum", "sum_fast", "pre_ag"):
            for mname, mc in (("L1", L1), ("DRAM", DRAM)):
                if mname == "DRAM" and mom != "sum":
                    continue
                variants.append((f"polynorm_tp I={inter} fp32 moments={mom} {mname}",
                                 lambda g_t=g_t, u_t=u_t, consts=consts, mom=mom, mc=mc: polynorm_tp(
                                     g_t, u_t, consts, ccl=ccl, mode="fp32", memory_config=mc,
                                     compute_kernel_config=ckc, moments=mom)))
        variants.append((f"polynorm_tp I={inter} bf16 L1",
                         lambda g_t=g_t, u_t=u_t, consts=consts: polynorm_tp(
                             g_t, u_t, consts, ccl=ccl, mode="bf16", memory_config=L1, compute_kernel_config=ckc)))
    # grouped
    cw = torch.sigmoid(torch.randn(cfg.num_experts, 3, generator=gen))
    cb = (torch.rand(cfg.num_experts, generator=gen) - 0.5).clamp(-0.5, 0.5)
    gconsts = GroupedPolyNormConsts.from_coefficients(mesh_device, cfg, cw, cb)
    GU = torch.randn(4, 96, 32, 2560, generator=gen) * 2
    gu_t = to_mesh(GU, mesh_device, cfg, dtype=ttnn.bfloat16, dp_dim=0, tp_dim=1)
    keep += [gconsts, gu_t]
    for impl, mode, mom in (("horner", "fp32", "sum"), ("horner", "fp32", "sum_fast"), ("horner", "fp32", "pre_ag"),
                            ("horner", "bf16", "sum"), ("rms", "fp32", "sum"), ("rms", "bf16", "sum")):
        variants.append((f"grouped impl={impl} mode={mode} moments={mom} DRAM",
                         lambda impl=impl, mode=mode, mom=mom: grouped_polynorm(
                             gu_t, gconsts, inter=1280, mode=mode, impl=impl, moments=mom, compute_kernel_config=ckc)))
    try:
        from models.demos.motif3.tt.moe import grouped_polynorm as moe_grouped_polynorm

        for mode in ("fp32", "bf16"):
            variants.append((f"grouped tt/moe.py local G6 composite mode={mode} DRAM",
                             lambda mode=mode: moe_grouped_polynorm(gu_t, gconsts.as_dict(mode), inter=1280, mode=mode,
                                                                    compute_kernel_config=ckc)))
    except Exception as e:  # the MoE module may be mid-edit
        print(f"[mlp] tt/moe.py grouped_polynorm unavailable: {e}")
    # decode matmuls (bfp8 weights, M = 32)
    ckm = cfg.compute_config("dense_mlp")
    for (kind, mm), (k, n) in {("dense", "gate_up"): (4096, 3072), ("dense", "down"): (1536, 4096),
                               ("shared", "gate_up"): (4096, 320), ("shared", "down"): (160, 4096)}.items():
        w_t = to_mesh(torch.randn(1, 1, k, n, generator=gen) * k**-0.5, mesh_device, cfg, dtype=ttnn.bfloat8_b)
        x_t = to_mesh(torch.randn(1, 1, 8, k, generator=gen), mesh_device, cfg)
        keep += [w_t, x_t]
        cands = [("auto", None), ("default", DECODE_MATMUL_GRIDS[(kind, mm)])]
        extra = {("dense", "gate_up"): [((12, 4), 8), ((12, 2), 8), ((8, 4), 8)],
                 ("dense", "down"): [((8, 2), 8), ((8, 8), 8), ((4, 4), 8)],
                 ("shared", "gate_up"): [((5, 1), 8), ((2, 1), 8), ((10, 1), 16)],
                 ("shared", "down"): [((8, 2), 5), ((8, 8), 5), ((4, 4), 5)]}[(kind, mm)]
        cands += [(f"grid={g} bw={b}", (g, b)) for g, b in extra]
        for cname, c in cands:
            pc = None if c is None else decode_matmul_pc(k, n, c[0], in0_block_w=c[1])
            variants.append((f"matmul {kind}.{mm} [{k}x{n}] {cname} out=bf16 L1",
                             lambda x_t=x_t, w_t=w_t, pc=pc: ttnn.linear(x_t, w_t, dtype=ttnn.bfloat16,
                                                                         compute_kernel_config=ckm, program_config=pc,
                                                                         memory_config=L1)))
    # full module decode chains (random weights)
    for layer, inter in ((0, 12288), (2, 1280)):
        src = _random_mlp_source(layer, inter, seed=300 + layer)
        mlp = PolyNormMLP(mesh_device, cfg, layer, source=src, ccl=ccl, cache=False)
        xd = _rows_input(torch.randn(32, 4096, generator=gen), mesh_device, cfg)
        keep += [mlp, xd]
        variants.append((f"PolyNormMLP {mlp.kind} forward_decode", lambda mlp=mlp, xd=xd: mlp.forward_decode(xd)))
        if mlp.kind == "shared":
            variants.append(("PolyNormMLP shared forward_decode all_reduce=False",
                             lambda mlp=mlp, xd=xd: mlp.forward_decode(xd, all_reduce=False)))
    names = profile_variants(mesh_device, cfg, variants)
    path = GOLDEN_DIR / "profile_variant_names.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(names, indent=1))
    print(f"[mlp] profile variant names -> {path} ({len(names)} variants)")
    for o in keep:
        if hasattr(o, "deallocate") and not isinstance(o, ttnn.Tensor):
            o.deallocate()
        else:
            _free(o)
    errors = [n for n in names.values() if "[ERROR " in n]
    assert not errors, "\n".join(errors)


if __name__ == "__main__":
    if "--make-goldens" in sys.argv:
        make_real_goldens()
    elif "--make-goldens-l4" in sys.argv:
        make_real_goldens(REAL_GOLDEN_L4, layers=(4,))
    elif "--profile-summary" in sys.argv:
        csv_path = sys.argv[sys.argv.index("--profile-summary") + 1]
        names_file = GOLDEN_DIR / "profile_variant_names.json"
        names = {int(k): v for k, v in json.loads(names_file.read_text()).items()} if names_file.is_file() else None
        summarize_profile(csv_path, names)
    else:
        print(__doc__)
