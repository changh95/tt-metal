# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact-fp32 router logits kernel (``tt/kernels/router_fp32.py``; WAVE_A_REVIEW D1(b), MOE-2 / MOE-7, GATE-3).

CPU (device free; the root conftest is fine inside scripts/hostrun.sh, devices hidden):

    scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q -s \
        models/demos/motif3/tests/unit/test_router_fp32.py -k "cpu or import"

Real router inputs (CPU, ~2-8 min, ~125 MB under tt_cache/test/router_kernel/; needed by test_device_real_layers):

    scripts/hostrun.sh -t 3000 -- python -m models.demos.motif3.tests.unit.test_router_fp32 capture

Device (scripts/devrun.sh only; ~4 min for all):

    scripts/devrun.sh -t 1500 -n router_fp32 -- python -m pytest models/demos/motif3/tests/unit/test_router_fp32.py \
        -s -p no:cacheprovider -k device

Tests:
* ``test_cpu_*``: the host weight permutation is a bijection onto every worker's (expert group x k group) block; the
  host model of the kernel's data movement (DEST slots, SFPU lanes, SFPTRANSP, reduction) equals ``x @ W^T``; the
  bit-exact fp32 model of the kernel's arithmetic is within the fp32 bound of fp64 and row-independent (the prefill
  shapes); the input contract (leading dims, M = 32 n) and the bf16-valued weight check.
* ``test_import_is_self_contained``: README §13 import rule for this module, in a fresh interpreter.
* ``test_device_debug_modes``: data movement on device, bit exact: mode 1 (every worker's chunk-0 weight tiles as
  copied into DEST) and mode 2 (x[t, k] read by the math RISC-V, broadcast through SFPLOADI, un-permuted).
* ``test_device_random`` (replay loop and the SFPI reference loop): decode shape, random weights at real dims, x
  distinct per chip, DRAM and L1 input / output: worker partials and logits bitwise equal to the host fp32 model on all
  chips, <= 1e-6 relative to max |logit| vs fp64; replicated x -> bitwise identical logits on all chips.
* ``test_device_prefill_random``: prefill shapes (M = 64, 96, 160 with x distinct per chip; M = 4096 replicated), both
  loops, DRAM / L1: every tile row bitwise equal to the host model (pipelined rows never mix receive buffers).
* ``test_device_contract_and_ownership``: ``supports`` / errors for bad shapes and non-interleaved outputs, weight
  tensor validation, ``deallocate`` frees only an owned weight.
* ``test_device_real_layers``: real router weights and real router inputs (layers 2-35): logit error vs fp64, top-8
  set agreement of the exact router and of the FPU composite (G5) with the same device tail (uniform sample and a
  near-tie stress set), bitwise check vs the host model, routing-weight error.
* ``test_device_real_all_tokens``: all 2971 tokens of 11 layers (the MoE test's capture, read only), decode shape (32
  tokens per chip and call) and prefill shape (one 2976-row call per layer): routes of the exact kernel and the FPU
  composite, with this file's G5 tail and with ``tt/moe.py`` ``MotifRouter`` (production tail), compared **directly
  with the reference's own fp32 CPU routes** (``ref_idx`` / ``ref_w``) and with fp64.
* ``test_device_trace_replay_and_cache``: TT-cache path (reload without touching the checkpoint, bitwise equal) and a
  trace of 3 back-to-back launches replayed with new inputs (persistent input, copy_host_to_device_tensor).
* ``test_device_interleaved_back_to_back``: two instances (different weights), three persistent inputs (two decode,
  one prefill shape), 8 launches back to back without a sync, eager, then captured in one trace and replayed 6 times
  with new per-chip data: every output bitwise equal to the host model (cross-launch races, stale args).
* ``test_device_perf``: decode latency (eager and traced, slope method of the gates) of the kernel, the FPU logits, and
  both with the router tail; host cost of an eager call; routes bitwise identical on all chips.
* ``test_device_motif_router_integration``: ``tt/moe.py`` ``MotifRouter`` (imported lazily, read only) with and
  without this kernel as ``logits_fn``: traced cost, agreement, routes identical on all chips (fails on any error).
* ``test_device_perf_prefill``: prefill latency (M = 128, 1024, 4096) of the kernel vs the FPU logits.
* ``test_device_timing_breakdown``: per-core wall-clock stamps of one decode launch (kernels/router_fp32/timing.h).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.motif3.tt.kernels import router_fp32 as R

PROJECT_ROOT = Path("/home/ttuser/hchang/experiments/motif-3")
H, E, K = R.HIDDEN, R.N_EXPERTS, 8
REAL_DIR = Path(os.environ.get("MOTIF3_ROUTER_TEST_DIR", str(PROJECT_ROOT / "tt_cache" / "test" / "router_kernel")))
REAL_FILE = REAL_DIR / "real_router_inputs_L02-35_v1.pt"
REAL_LAYERS = tuple(range(2, 36))
N_UNIFORM, N_NEAR = 384, 64  # per layer: uniform token sample + the most near-tie tokens (stress set)
PROMPTS_JSON = Path(__file__).resolve().parents[2] / "reference" / "prompts" / "messages.json"
MOE_ALL_TOKENS = PROJECT_ROOT / "tt_cache" / "test" / "moe" / "real_router_inputs_v1.pt"  # tests/unit/test_moe.py


# =====================================================================================================================
# real router inputs (CPU capture; run under scripts/hostrun.sh, never while holding the device lock)
# =====================================================================================================================
def capture_real_inputs(out_path: Path = REAL_FILE, layers=REAL_LAYERS) -> Path:
    """Reference prefix model (layers 0..35, bf16 = HF numerics, lazy experts) over the reference prompt set
    (``reference/prompts/messages.json``, 6 chats, 2971 tokens). Per layer L in ``layers``, the router input
    ``layers.L.post_attention_layernorm.out`` (bf16) of ``N_UNIFORM`` uniformly sampled tokens (seed L) and of the
    ``N_NEAR`` tokens with the smallest fp64 8th/9th biased-score gap (real router weight and expert_bias), plus the
    gap of every token. ~8 min CPU, ~125 MB (test input data, regenerable; the TT weight cache of these tests is
    3 MB)."""
    from models.demos.motif3.reference.golden import capture_model_goldens
    from models.demos.motif3.reference.tokenizer import encode_chat, load_tokenizer
    from models.demos.motif3.reference.weights import load_reference_model

    t0 = time.time()
    layers = tuple(sorted(int(i) for i in layers))
    model = load_reference_model(layer_ids=range(max(layers) + 1), dtype=torch.bfloat16, lazy_experts=True)
    tok = load_tokenizer()
    prompts = json.loads(PROMPTS_JSON.read_text())["prompts"]
    want = {f"layers.{L}.post_attention_layernorm.out" for L in layers}
    xs = {L: [] for L in layers}
    lens = []
    for p in prompts:
        ids = encode_chat(p["messages"], tok, add_generation_prompt=p.get("add_generation_prompt", True))
        g = capture_model_goldens(model, torch.tensor([ids]), include=lambda n: n in want)
        for L in layers:
            xs[L].append(g["prefill"][f"layers.{L}.post_attention_layernorm.out"].reshape(-1, H).to(torch.bfloat16))
        lens.append(len(ids))
        print(f"[router_fp32 capture] {p.get('name')}: {len(ids)} tokens, {time.time() - t0:.0f} s", flush=True)
    out = {"meta": {"format": "motif3-router-fp32-inputs/1", "layers": list(layers), "prompt_lens": lens,
                    "n_uniform": N_UNIFORM, "n_near": N_NEAR}, "layers": {}}
    for L in layers:
        x = torch.cat(xs[L], 0)
        moe = model.model.layers[str(L)].moe
        w = moe.router.gate.weight.detach().double()  # [384, 4096] (bf16 values)
        b = moe.expert_bias.detach().double()
        s = torch.sigmoid(x.double() @ w.T) + b
        top9 = torch.topk(s, k=K + 1, dim=-1).values
        gap = (top9[:, K - 1] - top9[:, K]).float()
        g_ = torch.Generator().manual_seed(1000 + L)
        uni = torch.randperm(x.shape[0], generator=g_)[:N_UNIFORM]
        near = torch.argsort(gap)[:N_NEAR]
        out["layers"][L] = {"x": x[uni].contiguous(), "x_near": x[near].contiguous(), "idx_uniform": uni,
                            "idx_near": near, "gap_all": gap}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    torch.save(out, tmp)
    os.replace(tmp, out_path)
    print(f"[router_fp32 capture] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in "
          f"{time.time() - t0:.0f} s")
    return out_path


# =====================================================================================================================
# helpers
# =====================================================================================================================
def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).float()


def random_router(seed: int = 0):
    """bf16-valued ``W^T [4096, 384]`` with real-checkpoint-like scale (sigma_W 0.02) and ``x [n, 4096]`` like
    ``gamma * rmsnorm(h)`` (per-channel scales 0.014 .. 0.34, as the real post_attention_layernorm gamma)."""
    g = torch.Generator().manual_seed(seed)
    w_t = _bf16(0.02 * torch.randn(H, E, generator=g))
    gamma = 0.014 + 0.33 * torch.rand(H, generator=g)
    return w_t, gamma


def random_x(n: int, gamma: torch.Tensor, seed: int, heavy: bool = False) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, H, generator=g)
    if heavy:
        ch = torch.randint(0, H, (8,), generator=g)
        h[:, ch] *= 20.0
    f = h / torch.sqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) * gamma
    return _bf16(f)


def logit_errors(dev: torch.Tensor, x: torch.Tensor, w_t: torch.Tensor) -> dict:
    """Errors of fp32 logits ``dev [n, 384]`` against fp64: max abs; max abs relative to max |logit|, to the logit RMS
    and to the condition-aware scale ``sum_k |x_k w_k|`` of each logit (the fp32 summation bound's scale); PCC."""
    ref = R.golden_logits_fp64(x, w_t)
    err = (dev.double() - ref).abs()
    cond = x.double().abs() @ w_t.double().abs()
    return {
        "max_abs": float(err.max()),
        "rms_err": float(err.pow(2).mean().sqrt()),
        "logit_absmax": float(ref.abs().max()),
        "logit_rms": float(ref.pow(2).mean().sqrt()),
        "rel_to_absmax": float(err.max() / ref.abs().max()),
        "rel_to_rms": float(err.max() / ref.pow(2).mean().sqrt()),
        "rel_to_cond_max": float((err / cond.clamp_min(1e-30)).max()),
        "pcc": _pcc(dev, ref),
    }


def _pcc(a, b) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-300))


def _bits_mismatch(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of fp32 elements whose bit patterns differ (NaN safe)."""
    return int((a.contiguous().float().view(torch.int32) != b.contiguous().float().view(torch.int32)).sum())


# =====================================================================================================================
# CPU
# =====================================================================================================================
def test_cpu_weight_permutation_is_bijection():
    k_idx, e_idx = R._weight_gather_index()
    flat = (k_idx * E + e_idx).flatten()
    assert torch.equal(torch.sort(flat).values, torch.arange(H * E))
    # every worker's 16 tiles hold exactly its experts [128 g, +128) x its k group (k-tiles h, 32 + h, 64 + h, 96 + h)
    order = R.k_order()
    for q in (0, 31, 32, 95):
        g, h = divmod(q, R.K_GROUPS)
        tiles = [R.weight_tile_id(q, cc, i) for cc in range(R.N_CHUNKS) for i in range(R.W_TILES_PER_CHUNK)]
        ks = torch.unique(k_idx[tiles])
        es = torch.unique(e_idx[tiles])
        assert torch.equal(ks, torch.sort(order[h]).values)
        assert torch.equal(es, torch.arange(128 * g, 128 * g + 128))


def test_cpu_layout_model_matches_matmul():
    w_t, gamma = random_router(1)
    x = random_x(R.TOKENS, gamma, seed=2)
    out = R.emulate_layout(x, R.prepare_router_weight(w_t))
    ref = R.golden_logits_fp64(x, w_t)
    assert float((out - ref).abs().max()) <= 1e-12 * float(ref.abs().max())


def test_cpu_fp32_model_bound():
    w_t, gamma = random_router(3)
    x = random_x(R.TOKENS, gamma, seed=4, heavy=True)
    e = logit_errors(R.emulate_device_fp32(x, w_t), x, w_t)
    print(f"[router_fp32 cpu] fp32 model vs fp64: {e}")
    assert e["rel_to_absmax"] < 1e-6
    # fp32 summation bound: |err| <= n u sum|x w| with n = 128 + 32 adds, u = 2^-24
    assert e["rel_to_cond_max"] < 160 * 2.0**-24


def test_cpu_fp32_model_rows_independent():
    """The prefill kernel runs the decode arithmetic on every tile row: the host model of M rows is the concatenation
    of its 32-row results, whatever the block size."""
    w_t, gamma = random_router(5)
    x = random_x(160, gamma, seed=6, heavy=True)
    full = R.emulate_device_fp32(x, w_t, block=64)
    rows = torch.cat([R.emulate_device_fp32(x[i : i + 32], w_t) for i in range(0, 160, 32)])
    assert _bits_mismatch(full, rows) == 0
    assert _bits_mismatch(R.emulate_device_fp32(x.reshape(1, 1, 160, H), w_t, block=1024), full) == 0
    with pytest.raises(ValueError):
        R.emulate_device_fp32(x[:40], w_t)


def test_cpu_input_contract():
    for shape, m in (((1, 1, 32, H), 32), ((32, H), 32), ((1, 64, H), 64), ((1, 1, 4096, H), 4096)):
        assert R.input_rows(shape) == m
    for shape in ((2, 32, H), (1, 2, 32, H), (2, 1, 32, H), (1, 1, 40, H), (1, 1, 16, H), (1, 1, 32, H - 1), (H,),
                  (1, 1, 0, H)):
        with pytest.raises(ValueError):
            R.input_rows(shape)


def test_cpu_bf16_weight_check():
    w_t, _ = random_router(7)
    a = R.prepare_router_weight(w_t)  # bf16-valued fp32
    b = R.prepare_router_weight(w_t.to(torch.bfloat16))
    c = R.prepare_router_weight(w_t.double())
    assert torch.equal(a, b.float()) and torch.equal(a.double(), c)
    bad = w_t.clone()
    bad[17, 5] += 2.0**-20  # not representable in bf16 at |w| ~ 0.02
    with pytest.raises(ValueError, match="bf16-valued"):
        R.prepare_router_weight(bad)
    with pytest.raises(ValueError):
        R.prepare_router_weight(w_t[:, :383])


_IMPORT_PROBE = r"""
import importlib, json, sys
mods = json.loads(sys.argv[1])
for m in mods:
    importlib.import_module(m)
demos = sorted(m for m in sys.modules if m.startswith("models.demos.") and not m.startswith("models.demos.motif3"))
other_models = sorted(m for m in sys.modules if m.startswith("models.") and not m.startswith("models.demos"))
heavy = sorted(m for m in ("vllm", "transformers", "huggingface_hub", "safetensors") if m in sys.modules)
motif3 = sorted(m for m in sys.modules if m.startswith("models.demos.motif3."))
print(json.dumps({"demos": demos, "other_models": other_models, "heavy": heavy, "motif3": motif3}))
"""


def test_import_is_self_contained():
    """README §13 import rule in a fresh interpreter: ``tt.kernels.router_fp32`` imports only ttnn / torch at module
    level (``weights`` is imported lazily inside ``from_source``), and opens no device."""
    tt_metal = Path(__file__).resolve().parents[5]
    mods = ["models.demos.motif3.tt.kernels", "models.demos.motif3.tt.kernels.router_fp32"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tt_metal) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, json.dumps(mods)],
        cwd=str(tt_metal),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, res.stderr[-4000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["demos"] == [] and out["other_models"] == [], out
    assert out["heavy"] == [], out
    assert set(out["motif3"]) == {"models.demos.motif3.tt", *mods}, out["motif3"]
    for marker in ("Opening user mode device driver", "Starting devices in cluster"):
        assert marker not in res.stderr and marker not in res.stdout


# =====================================================================================================================
# device helpers
# =====================================================================================================================
def _device_params():
    from models.demos.motif3.tt.model_config import device_params

    return device_params()


DEVICE = pytest.mark.parametrize("mesh_device, device_params", [((4, 8), _device_params())], indirect=True,
                                 ids=["4x8"])


def _log_fabric(mesh_device, tag):
    from models.demos.motif3.tt.ccl import log_fabric

    log_fabric(mesh_device, tag)


def _mapper(mesh_device):
    Rr, Cc = tuple(mesh_device.shape)
    return ttnn.create_mesh_mapper(
        mesh_device, ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], ttnn.MeshShape(Rr, Cc))
    )


def _per_chip_x(mesh_device, xs: torch.Tensor, memory_config=None):
    """``xs [R, C, M, 4096]`` -> chip (r, c) holds ``[1, 1, M, 4096]`` = xs[r, c] (bf16 TILE, DRAM default)."""
    return ttnn.from_torch(
        xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=_mapper(mesh_device),
    )


def _replicated(mesh_device, t: torch.Tensor, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _per_chip_out(mesh_device, t) -> torch.Tensor:
    from models.demos.motif3.tt.ccl import device_tensors_to_torch

    return device_tensors_to_torch(t, mesh_device)  # [R, C, *local]


def _copy_in(mesh_device, xs: torch.Tensor, dev_x) -> None:
    """New per-chip data ``xs [R, C, M, 4096]`` into the persistent input ``dev_x`` (trace replays)."""
    host = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=_mapper(mesh_device))
    ttnn.copy_host_to_device_tensor(host, dev_x)


def _n_chips(mesh_device) -> int:
    Rr, Cc = tuple(mesh_device.shape)
    return Rr * Cc


def _chip_xs(mesh_device, m: int, gamma, seed: int, heavy=None) -> torch.Tensor:
    """Distinct ``[m, 4096]`` inputs per chip: ``[n_chips, m, 4096]`` (odd chips heavy unless ``heavy`` is given)."""
    n = _n_chips(mesh_device)
    return torch.stack([random_x(m, gamma, seed=seed + i, heavy=(i % 2 == 1) if heavy is None else heavy)
                        for i in range(n)])


def _mismatch_per_chip(mesh_device, out, xs: torch.Tensor, w_t) -> int:
    """Bitwise mismatches of the device logits ``out`` against the host fp32 model of every chip's ``xs[i]``."""
    n = _n_chips(mesh_device)
    m = xs.shape[-2]
    got = _per_chip_out(mesh_device, out).reshape(n, m, E)
    return sum(_bits_mismatch(got[i], R.emulate_device_fp32(xs[i], w_t)) for i in range(n))


# =====================================================================================================================
# device: data movement and arithmetic, decode and prefill shapes
# =====================================================================================================================
@DEVICE
def test_device_debug_modes(mesh_device):
    _log_fabric(mesh_device, "router_fp32 debug modes")
    w_t, gamma = random_router(11)
    w_prep = R.prepare_router_weight(w_t)
    x = random_x(R.TOKENS, gamma, seed=12)
    x_tt = _replicated(mesh_device, x.reshape(1, 1, R.TOKENS, H))
    w_tt = _replicated(mesh_device, w_prep)
    tiles = w_prep.reshape(R.W_TILES, R.TILE, R.TILE)
    # mode 1: worker q's dump tile j = its chunk-0 weight tile j (DEST copy, bit exact)
    rt = R.RouterLogitsFP32(mesh_device, weight_tensor=w_tt, mode=R.MODE_DEBUG_W)
    out = rt(x_tt)
    dbg = _per_chip_out(mesh_device, rt.last_debug)[0, 0].reshape(R.N_WORKERS, R.N_VEC, R.TILE, R.TILE)
    exp = torch.stack([tiles[[R.weight_tile_id(q, 0, i) for i in range(4)]] for q in range(R.N_WORKERS)])
    bad = int((dbg != exp).sum())
    print(f"[router_fp32 debug] mode 1 (weights in DEST): mismatching elements {bad} / {exp.numel()}")
    ttnn.deallocate(out)
    ttnn.deallocate(rt.last_debug)
    # mode 2: dump tile j of worker (g, h) has row t = x[t, k_of(h, 0, j)] (chunk 0 = k-tile h) in every column
    rt2 = R.RouterLogitsFP32(mesh_device, weight_tensor=w_tt, mode=R.MODE_DEBUG_X)
    out2 = rt2(x_tt)
    dbg2 = _per_chip_out(mesh_device, rt2.last_debug)[0, 0].reshape(R.N_WORKERS, R.N_VEC, R.TILE, R.TILE)
    exp2 = torch.empty_like(dbg2)
    for q in range(R.N_WORKERS):
        h = q % R.K_GROUPS
        for j in range(R.N_VEC):
            exp2[q, j] = x[:, R.k_of(h, 0, j)].reshape(R.TILE, 1).expand(R.TILE, R.TILE)
    bad2 = int((dbg2 != exp2).sum())
    print(f"[router_fp32 debug] mode 2 (x broadcast + un-permute): mismatching elements {bad2} / {exp2.numel()}")
    ttnn.deallocate(out2)
    ttnn.deallocate(rt2.last_debug)
    # the caller's weight survives both instances' deallocate (not owned)
    rt.deallocate()
    rt2.deallocate()
    alive = w_tt.is_allocated()
    ttnn.deallocate(w_tt)
    assert bad == 0 and bad2 == 0 and alive


@DEVICE
@pytest.mark.parametrize("mode", [R.MODE_LOGITS, R.MODE_LOGITS_SFPI], ids=["replay", "sfpi"])
def test_device_random(mesh_device, mode):
    _log_fabric(mesh_device, f"router_fp32 random mode {mode}")
    Rr, Cc = tuple(mesh_device.shape)
    w_t, gamma = random_router(21)
    rt = R.RouterLogitsFP32(mesh_device, w_t, debug=True, mode=mode)
    # distinct x per chip
    xs = torch.stack([random_x(R.TOKENS, gamma, seed=100 + i, heavy=(i % 2 == 1)) for i in range(Rr * Cc)])
    x_tt = _per_chip_x(mesh_device, xs.reshape(Rr, Cc, R.TOKENS, H))
    t0 = time.perf_counter()
    out = rt(x_tt)
    ttnn.synchronize_device(mesh_device)
    print(f"[router_fp32 random] first call (compile + run) {time.perf_counter() - t0:.2f} s")
    dev = _per_chip_out(mesh_device, out).reshape(Rr * Cc, R.TOKENS, E)
    dbg = _per_chip_out(mesh_device, rt.last_debug).reshape(Rr * Cc, R.N_WORKERS, R.N_VEC, R.TILE, R.TILE)
    worst = None
    n_bit = 0
    for i in range(Rr * Cc):
        emu = R.emulate_device_fp32(xs[i], w_t)
        n_bit += _bits_mismatch(dev[i], emu)
        e = logit_errors(dev[i], xs[i], w_t)
        if worst is None or e["rel_to_absmax"] > worst["rel_to_absmax"]:
            worst = e
    # worker partials of chip 0 vs the host model (k-group partials)
    emu_parts = _partials_fp32(xs[0], w_t)
    part_bad = _bits_mismatch(dbg[0], emu_parts)
    print(f"[router_fp32 random] chip-0 worker partials != host fp32 model: {part_bad} / {emu_parts.numel()}")
    print(f"[router_fp32 random] logits != host fp32 model (bitwise), all chips: {n_bit} / {dev.numel()}")
    print(f"[router_fp32 random] worst chip vs fp64: {worst}")
    ttnn.deallocate(out)
    # x in L1, logits in L1: same bits
    x_l1 = ttnn.to_memory_config(x_tt, ttnn.L1_MEMORY_CONFIG)
    out_l1 = rt(x_l1, memory_config=ttnn.L1_MEMORY_CONFIG)
    l1_same = _bits_mismatch(_per_chip_out(mesh_device, out_l1).reshape(Rr * Cc, R.TOKENS, E), dev) == 0
    print(f"[router_fp32 random] L1 input + L1 output -> bitwise equal to the DRAM run: {l1_same}")
    for t in (x_l1, out_l1):
        ttnn.deallocate(t)
    # replicated x: identical logits on every chip
    x_rep = _replicated(mesh_device, xs[0].reshape(1, 1, R.TOKENS, H))
    out_rep = rt(x_rep)
    per = _per_chip_out(mesh_device, out_rep).reshape(Rr * Cc, R.TOKENS, E)
    same = all(_bits_mismatch(per[i], per[0]) == 0 for i in range(Rr * Cc))
    print(f"[router_fp32 random] replicated x -> bitwise identical on all {Rr * Cc} chips: {same}")
    rt.deallocate()
    assert worst["rel_to_absmax"] <= 1e-6, worst
    assert worst["pcc"] >= 0.999999, worst
    assert n_bit == 0 and part_bad == 0
    assert same and l1_same


@DEVICE
def test_device_prefill_random(mesh_device):
    """Prefill shapes ``[1, 1, M, 4096]`` (n = M / 32 tile rows, pipelined): every tile row of every chip bitwise equal
    to the host fp32 model, replay and SFPI loops, DRAM and L1 in / out, row-0 worker partials (debug dump); M = 4096
    replicated: identical on all chips and equal to the host model, <= 1e-6 relative to max |logit| vs fp64."""
    _log_fabric(mesh_device, "router_fp32 prefill random")
    Rr, Cc = tuple(mesh_device.shape)
    n = Rr * Cc
    w_t, gamma = random_router(61)
    rt = R.RouterLogitsFP32(mesh_device, w_t)
    rt_sfpi = R.RouterLogitsFP32(mesh_device, weight_tensor=rt.weight, mode=R.MODE_LOGITS_SFPI)
    rt_dbg = R.RouterLogitsFP32(mesh_device, weight_tensor=rt.weight, debug=True)
    bad = {}
    worst, max_abs, min_pcc = 0.0, 0.0, 1.0
    for M in (64, 96, 160):
        xs = _chip_xs(mesh_device, M, gamma, seed=7000 + M)
        x_tt = _per_chip_x(mesh_device, xs.reshape(Rr, Cc, M, H))
        t0 = time.perf_counter()
        out = rt(x_tt)
        ttnn.synchronize_device(mesh_device)
        t_first = time.perf_counter() - t0
        assert tuple(int(d) for d in out.shape) == (1, 1, M, E)
        got = _per_chip_out(mesh_device, out).reshape(n, M, E)
        exp = torch.stack([R.emulate_device_fp32(xs[i], w_t) for i in range(n)])
        bad[f"M{M}_replay"] = _bits_mismatch(got, exp)
        for i in (0, n - 1):
            e = logit_errors(got[i], xs[i], w_t)
            worst = max(worst, e["rel_to_absmax"])
            max_abs = max(max_abs, e["max_abs"])
            min_pcc = min(min_pcc, e["pcc"])
        ttnn.deallocate(out)
        o = rt_sfpi(x_tt)
        bad[f"M{M}_sfpi"] = _bits_mismatch(_per_chip_out(mesh_device, o).reshape(n, M, E), exp)
        ttnn.deallocate(o)
        x_l1 = ttnn.to_memory_config(x_tt, ttnn.L1_MEMORY_CONFIG)
        o = rt(x_l1, memory_config=ttnn.L1_MEMORY_CONFIG)
        bad[f"M{M}_l1"] = _bits_mismatch(_per_chip_out(mesh_device, o).reshape(n, M, E), exp)
        ttnn.deallocate(o)
        ttnn.deallocate(x_l1)
        o = rt_dbg(x_tt)
        dbg = _per_chip_out(mesh_device, rt_dbg.last_debug)[0, 0].reshape(R.N_WORKERS, R.N_VEC, R.TILE, R.TILE)
        bad[f"M{M}_row0_partials"] = _bits_mismatch(dbg, _partials_fp32(xs[0][: R.TOKENS], w_t))
        bad[f"M{M}_debug_logits"] = _bits_mismatch(_per_chip_out(mesh_device, o).reshape(n, M, E), exp)
        _free([o, rt_dbg.last_debug, x_tt])
        print(f"[router_fp32 prefill] M={M}: first call {t_first:.2f} s; mismatches "
              f"{ {k: v for k, v in bad.items() if k.startswith(f'M{M}_')} }")
    # M = 4096 (128 tile rows) replicated
    M = 4096
    x = random_x(M, gamma, seed=7777, heavy=True)
    x_tt = _replicated(mesh_device, x.reshape(1, 1, M, H))
    out = rt(x_tt)
    per = _per_chip_out(mesh_device, out).reshape(n, M, E)
    same = all(_bits_mismatch(per[i], per[0]) == 0 for i in range(n))
    bad["M4096_vs_host"] = _bits_mismatch(per[0], R.emulate_device_fp32(x, w_t))
    e = logit_errors(per[0], x, w_t)
    worst = max(worst, e["rel_to_absmax"])
    max_abs = max(max_abs, e["max_abs"])
    min_pcc = min(min_pcc, e["pcc"])
    print(f"[router_fp32 prefill] M=4096 replicated: identical on all {n} chips {same}; chip 0 vs host model "
          f"{bad['M4096_vs_host']} mismatches; vs fp64 {e}")
    _free([out, x_tt])
    rt.deallocate()
    print(f"[router_fp32 prefill] vs fp64: worst rel error (to max |logit|) {worst:.3e}, max abs {max_abs:.3e}, "
          f"min PCC {min_pcc:.12f}; mismatches {bad}")
    assert same and all(v == 0 for v in bad.values()), bad
    assert worst <= 1e-6 and min_pcc >= 0.999999


@DEVICE
def test_device_contract_and_ownership(mesh_device):
    """Input / output contract and weight ownership: ``[2, 32, 4096]`` (64 tokens in a leading dim) is rejected, not
    truncated; ``[32, 4096]`` and ``[1, 1, 64, 4096]`` work; sharded output memory configs and malformed weight
    tensors are rejected; ``deallocate`` frees only an owned weight."""
    _log_fabric(mesh_device, "router_fp32 contract")
    w_t, gamma = random_router(303)
    rt = R.RouterLogitsFP32(mesh_device, w_t)
    x2 = torch.stack([random_x(R.TOKENS, gamma, seed=7), random_x(R.TOKENS, gamma, seed=8)])  # [2, 32, 4096]
    checks = {}
    x3_tt = _replicated(mesh_device, x2)
    checks["supports([2,32,4096]) is False"] = not R.RouterLogitsFP32.supports(x3_tt)
    with pytest.raises(ValueError):
        rt(x3_tt)
    x64_tt = _replicated(mesh_device, x2.reshape(1, 1, 64, H))
    checks["supports([1,1,64,4096])"] = R.RouterLogitsFP32.supports(x64_tt)
    checks["not supports_decode([1,1,64,4096])"] = not R.RouterLogitsFP32.supports_decode(x64_tt)
    o = rt(x64_tt)
    checks["[1,1,64,4096] = both rows"] = _bits_mismatch(
        _per_chip_out(mesh_device, o)[0, 0].reshape(64, E), R.emulate_device_fp32(x2.reshape(64, H), w_t)) == 0
    ttnn.deallocate(o)
    x2d_tt = _replicated(mesh_device, x2[0])  # [32, 4096]
    checks["supports_decode([32,4096])"] = R.RouterLogitsFP32.supports_decode(x2d_tt)
    o = rt(x2d_tt)
    checks["[32,4096] -> [1,1,32,384]"] = tuple(int(d) for d in o.shape) == (1, 1, R.TOKENS, E) and _bits_mismatch(
        _per_chip_out(mesh_device, o)[0, 0].reshape(R.TOKENS, E), R.emulate_device_fp32(x2[0], w_t)) == 0
    ttnn.deallocate(o)
    x_f32 = _replicated(mesh_device, x2[0].reshape(1, 1, R.TOKENS, H), dtype=ttnn.float32)
    checks["fp32 input rejected"] = not R.RouterLogitsFP32.supports(x_f32)
    sharded = ttnn.create_sharded_memory_config(
        (R.TOKENS, E), core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    x1 = _replicated(mesh_device, x2[0].reshape(1, 1, R.TOKENS, H))
    with pytest.raises(ValueError):
        rt(x1, memory_config=sharded)
    with pytest.raises(ValueError):
        R.RouterLogitsFP32(mesh_device, w_t, output_memory_config=sharded)
    with pytest.raises(ValueError):
        R.RouterLogitsFP32(mesh_device, weight_tensor=x1)  # not the prepared weight
    with pytest.raises(ValueError):
        R.RouterLogitsFP32(mesh_device, w_t + 2.0**-20)  # not bf16-valued
    # ownership: a borrowing instance does not free the owner's weight
    borrow = R.RouterLogitsFP32(mesh_device, weight_tensor=rt.weight)
    checks["borrower does not own"] = not borrow.owns_weight and rt.owns_weight
    borrow.deallocate()
    o = rt(x1)  # still works
    checks["owner works after borrower.deallocate"] = _bits_mismatch(
        _per_chip_out(mesh_device, o)[0, 0].reshape(R.TOKENS, E), R.emulate_device_fp32(x2[0], w_t)) == 0
    ttnn.deallocate(o)
    w = rt.weight
    rt.deallocate()
    checks["owner frees its weight"] = not w.is_allocated()
    with pytest.raises(RuntimeError):
        rt(x1)
    _free([x3_tt, x64_tt, x2d_tt, x_f32, x1])
    print(f"[router_fp32 contract] {checks}")
    assert all(checks.values()), checks


# =====================================================================================================================
# router tail (MOE-2, G5 pipeline) and the FPU composite logits, for comparison
# =====================================================================================================================
def router_tail(logits, bias_tt, scale: float = 2.0):
    """The composite router after the logits (tt/moe.py ``MotifRouter.__call__``): fp32 sigmoid -> + expert_bias ->
    topk(8) on FLOAT32 -> gather the unbiased scores -> / (sum + 1e-20) * scale. Returns (idx, w, biased)."""
    dram = ttnn.DRAM_MEMORY_CONFIG
    scores = ttnn.sigmoid(logits, memory_config=dram)
    biased = ttnn.add(scores, bias_tt, memory_config=dram)
    vals, idx = ttnn.topk(biased, k=K, dim=-1, largest=True, sorted=True, memory_config=dram)
    ttnn.deallocate(vals)
    w = ttnn.gather(scores, -1, idx, memory_config=dram)
    ttnn.deallocate(scores)
    den = ttnn.sum(w, dim=-1, keepdim=True, memory_config=dram)
    den2 = ttnn.add(den, 1e-20, memory_config=dram)
    ttnn.deallocate(den)
    inv = ttnn.reciprocal(den2, memory_config=dram)
    ttnn.deallocate(den2)
    wn = ttnn.multiply(w, inv, memory_config=dram)
    ttnn.deallocate(w)
    ttnn.deallocate(inv)
    ws = ttnn.multiply(wn, scale, memory_config=dram)
    ttnn.deallocate(wn)
    return idx, ws, biased


def _router_ckc():
    from models.demos.motif3.tt.model_config import make_compute_kernel_config

    return make_compute_kernel_config("HiFi4", fp32_acc=True)


def fpu_logits(x_tt, w_t_tt, ckc):
    """Draft-1 composite logits (G5 / MOE-2): ttnn.linear bf16 x bf16, HiFi4, fp32 dest acc, fp32 out."""
    return ttnn.linear(x_tt, w_t_tt, dtype=ttnn.float32, compute_kernel_config=ckc,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)


def golden_route(x: torch.Tensor, w_t: torch.Tensor, bias: torch.Tensor):
    """fp64 router on the bf16-valued inputs: (idx [n, 8], weights [n, 8], biased [n, 384])."""
    logits = R.golden_logits_fp64(x, w_t)
    s = torch.sigmoid(logits)
    biased = s + bias.double()
    _, idx = torch.topk(biased, k=K, dim=-1)
    top = s.gather(-1, idx)
    return idx, top / (top.sum(-1, keepdim=True) + 1e-20) * 2.0, biased


def set_agreement(idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
    """Per token: True iff the two top-8 index *sets* are equal."""
    return (torch.sort(idx_a.long(), -1).values == torch.sort(idx_b.long(), -1).values).all(-1)


def host_route_from_logits(logits: torch.Tensor, bias: torch.Tensor):
    """fp64 sigmoid / bias / topk applied to (device) logits: isolates the logit error from the device tail."""
    s = torch.sigmoid(logits.double())
    _, idx = torch.topk(s + bias.double(), k=K, dim=-1)
    return idx


def _sorted_weights(idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Routing weights reordered by expert id (order-insensitive comparison of equal sets)."""
    return w.float().gather(-1, torch.argsort(idx.long(), dim=-1))


# =====================================================================================================================
# timing (slope method of tests/unit/gates/gate_utils.py, GATES_RESULTS §2)
# =====================================================================================================================
def _free(o):
    if isinstance(o, (list, tuple)):
        for t in o:
            _free(t)
    elif isinstance(o, ttnn.Tensor):
        try:
            ttnn.deallocate(o)
        except Exception:
            pass


def time_eager(mesh_device, fn, iters: int = 20, warmup: int = 2) -> float:
    for _ in range(warmup):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


def _trace_times(mesh_device, fn, n: int, reps: int):
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        for _ in range(n):
            _free(fn())
    except BaseException:
        try:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        except Exception:
            pass
        try:
            ttnn.release_trace(mesh_device, tid)
        except Exception:
            pass
        raise
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            times.append((time.perf_counter() - t0) * 1e6)
    finally:
        ttnn.release_trace(mesh_device, tid)
    return sorted(times)


def time_traced(mesh_device, fn, n2: int = 32, reps: int = 9):
    """(per-call us, raw us): slope between traces of n2 / 2 and n2 back-to-back calls, min over replays."""
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    n1 = n2 // 2
    t1 = _trace_times(mesh_device, fn, n1, reps)
    t2 = _trace_times(mesh_device, fn, n2, reps)
    return max((t2[0] - t1[0]) / (n2 - n1), 0.0), t2[0] / n2


def host_cost(mesh_device, rt, x_tt, n: int = 50) -> dict:
    """Host time of an eager call (us): ``_program`` (key + patch), output alloc + free, ``generic_op`` enqueue, the
    whole ``rt(x)`` enqueue, and with the device sync."""
    m = int(x_tt.shape[-2])
    _free(rt(x_tt))
    ttnn.synchronize_device(mesh_device)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, m, E]), ttnn.float32, ttnn.TILE_LAYOUT, mesh_device,
                                         ttnn.DRAM_MEMORY_CONFIG)
    t0 = time.perf_counter()
    for _ in range(n):
        desc = rt._program(x_tt, out, None, None, m // R.TOKENS)
    t_prog = (time.perf_counter() - t0) / n * 1e6
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, m, E]), ttnn.float32, ttnn.TILE_LAYOUT,
                                                       mesh_device, ttnn.DRAM_MEMORY_CONFIG))
    t_alloc = (time.perf_counter() - t0) / n * 1e6
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.generic_op([x_tt, rt.weight, out], desc)
    t_enq = (time.perf_counter() - t0) / n * 1e6
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.deallocate(rt(x_tt))
    t_call = (time.perf_counter() - t0) / n * 1e6
    ttnn.synchronize_device(mesh_device)
    t_sync = (time.perf_counter() - t0) / n * 1e6
    ttnn.deallocate(out)
    return {"program_us": t_prog, "alloc_free_us": t_alloc, "generic_op_enqueue_us": t_enq, "call_enqueue_us": t_call,
            "call_with_sync_us": t_sync}


# =====================================================================================================================
# device: perf
# =====================================================================================================================
@DEVICE
def test_device_perf(mesh_device):
    """Decode latency per call at 32 tokens (eager and traced): the exact logits kernel alone, the FPU composite logits
    (ttnn.linear, G5), and both with the router tail (sigmoid, bias, topk, gather, renorm); host cost of an eager call;
    exact routes bitwise identical on all chips."""
    _log_fabric(mesh_device, "router_fp32 perf")
    w_t, gamma = random_router(31)
    bias = _bf16(1.1 + 0.03 * torch.randn(E, generator=torch.Generator().manual_seed(32)))
    x = random_x(R.TOKENS, gamma, seed=33)
    x_tt = _replicated(mesh_device, x.reshape(1, 1, R.TOKENS, H))
    w_t_tt = _replicated(mesh_device, w_t)
    bias_tt = _replicated(mesh_device, bias.reshape(1, 1, 1, E), dtype=ttnn.float32)
    rt = R.RouterLogitsFP32(mesh_device, w_t)
    rt_sfpi = R.RouterLogitsFP32(mesh_device, weight_tensor=rt.weight, mode=R.MODE_LOGITS_SFPI)
    ckc = _router_ckc()
    cases = {
        "exact_logits": lambda: rt(x_tt),
        "exact_logits_sfpi_loop": lambda: rt_sfpi(x_tt),
        "fpu_logits": lambda: fpu_logits(x_tt, w_t_tt, ckc),
        "exact_router": lambda: router_tail(rt(x_tt), bias_tt)[:2],
        "fpu_router": lambda: router_tail(fpu_logits(x_tt, w_t_tt, ckc), bias_tt)[:2],
    }
    res = {}
    for name, fn in cases.items():
        eager = time_eager(mesh_device, fn, iters=20)
        traced, raw = time_traced(mesh_device, fn, n2=32, reps=9)
        res[name] = (eager, traced, raw)
        print(f"[router_fp32 perf] {name}: eager {eager:.1f} us, traced {traced:.1f} us/call (raw {raw:.1f})")
    hc = host_cost(mesh_device, rt, x_tt)
    print(f"[router_fp32 perf] eager host cost per decode call: " + ", ".join(f"{k} {v:.1f}" for k, v in hc.items()))
    # routing consistency through the tail: bitwise identical routes on all chips
    idx, w, biased = router_tail(rt(x_tt), bias_tt)
    per_idx = _per_chip_out(mesh_device, idx)
    per_w = _per_chip_out(mesh_device, w)
    same = bool((per_idx == per_idx[0, 0]).all()) and bool(
        (per_w.view(torch.int32) == per_w[0, 0].view(torch.int32)).all()
    )
    print(f"[router_fp32 perf] exact router routes bitwise identical on all chips: {same}")
    _free([idx, w, biased, x_tt, w_t_tt, bias_tt])
    rt_sfpi.deallocate()
    rt.deallocate()
    assert same
    assert res["exact_logits"][1] <= 60.0, res["exact_logits"]


@DEVICE
def test_device_motif_router_integration(mesh_device):
    """``tt/moe.py`` ``MotifRouter`` (the MoE owner's optimized composite: 12-core multicast linear with fused sigmoid,
    padded multi-core topk, fused normalize) with and without this kernel as ``logits_fn``: traced cost with DRAM / L1
    intermediates, agreement vs fp64 on 32 tokens, routes bitwise identical on all chips, and ``MotifRouter.deallocate``
    leaves a borrowed kernel weight alone. Any error fails the test (it used to be print-only)."""
    from models.demos.motif3.tt import weights as W
    from models.demos.motif3.tt.model_config import MotifTTConfig
    from models.demos.motif3.tt.moe import MotifRouter

    _log_fabric(mesh_device, "router_fp32 MotifRouter integration")
    w_t, gamma = random_router(35)
    bias = _bf16(1.1 + 0.03 * torch.randn(E, generator=torch.Generator().manual_seed(36)))
    x = random_x(R.TOKENS, gamma, seed=37)
    x_tt = _replicated(mesh_device, x.reshape(1, 1, R.TOKENS, H))
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)
    L = 2
    src = W.DictWeightSource({W.hf_name(L, "moe.router.gate.weight"): w_t.t().contiguous(),
                              W.hf_name(L, "moe.expert_bias"): bias})
    rt = R.RouterLogitsFP32(mesh_device, w_t)
    borrowed = R.RouterLogitsFP32(mesh_device, weight_tensor=rt.weight)  # MotifRouter.deallocate() frees nothing here
    variants = {
        "moe_router_composite": MotifRouter(mesh_device, cfg, L, source=src, cache=False),
        "moe_router_exact_logits": MotifRouter(mesh_device, cfg, L, source=src, cache=False, logits_fn=borrowed),
    }
    times = {}
    for name, router in variants.items():
        for mc_name, mc in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
            fn = lambda router=router, mc=mc: router(x_tt, memory_config=mc)  # noqa: E731
            traced, raw = time_traced(mesh_device, fn, n2=32, reps=9)
            times[(name, mc_name)] = traced
            print(f"[router_fp32 perf] {name} ({mc_name} intermediates): traced {traced:.1f} us/call (raw {raw:.1f})")
    ia, wa = variants["moe_router_exact_logits"](x_tt)
    ib, wb = variants["moe_router_composite"](x_tt)
    idx_g, w_g, _ = golden_route(x, w_t, bias)
    pa = _per_chip_out(mesh_device, ia)
    pwa = _per_chip_out(mesh_device, wa)
    ag_e = float(set_agreement(pa[0, 0].reshape(R.TOKENS, K), idx_g).float().mean())
    ag_c = float(set_agreement(_per_chip_out(mesh_device, ib)[0, 0].reshape(R.TOKENS, K), idx_g).float().mean())
    same = bool((pa == pa[0, 0]).all()) and bool((pwa.view(torch.int32) == pwa[0, 0].view(torch.int32)).all())
    w_err = float((_sorted_weights(pa[0, 0].reshape(R.TOKENS, K), pwa[0, 0].reshape(R.TOKENS, K))
                   - _sorted_weights(idx_g, w_g)).abs().max())
    print(f"[router_fp32 perf] MotifRouter agreement vs fp64 on these 32 tokens: exact {ag_e:.4f} composite "
          f"{ag_c:.4f}; exact routes bitwise identical on all chips {same}; exact weights max err vs fp64 {w_err:.2e}")
    _free([ia, wa, ib, wb])
    for router in variants.values():
        router.deallocate()
    alive = rt.weight.is_allocated()
    o = rt(x_tt)
    ok = _bits_mismatch(_per_chip_out(mesh_device, o)[0, 0].reshape(R.TOKENS, E), R.emulate_device_fp32(x, w_t)) == 0
    _free([o, x_tt])
    rt.deallocate()
    extra = times[("moe_router_exact_logits", "l1")] - times[("moe_router_composite", "l1")]
    print(f"[router_fp32 perf] exact logits cost inside MotifRouter (L1): +{extra:.1f} us per call")
    assert ag_e == 1.0 and same and w_err < 1e-5
    assert alive and ok


@DEVICE
def test_device_perf_prefill(mesh_device):
    """Prefill latency (traced slope and eager) of the exact kernel vs the FPU composite logits at M = 128, 1024,
    4096 tokens (replicated x)."""
    _log_fabric(mesh_device, "router_fp32 perf prefill")
    w_t, gamma = random_router(41)
    rt = R.RouterLogitsFP32(mesh_device, w_t)
    w_t_tt = _replicated(mesh_device, w_t)
    ckc = _router_ckc()
    rows = []
    for M in (128, 1024, 4096):
        x = random_x(M, gamma, seed=4100 + M)
        x_tt = _replicated(mesh_device, x.reshape(1, 1, M, H))
        n2 = 16 if M <= 1024 else 8
        ex_tr, _ = time_traced(mesh_device, lambda: rt(x_tt), n2=n2, reps=5)
        ex_eg = time_eager(mesh_device, lambda: rt(x_tt), iters=5, warmup=1)
        fp_tr, _ = time_traced(mesh_device, lambda: fpu_logits(x_tt, w_t_tt, ckc), n2=n2, reps=5)
        fp_eg = time_eager(mesh_device, lambda: fpu_logits(x_tt, w_t_tt, ckc), iters=5, warmup=1)
        per_row = ex_tr / (M // R.TOKENS)
        rows.append((M, ex_tr, per_row))
        print(f"[router_fp32 perf prefill] M={M}: exact traced {ex_tr:.1f} us ({per_row:.1f} us per 32 tokens), eager "
              f"{ex_eg:.1f} us | fpu logits traced {fp_tr:.1f} us, eager {fp_eg:.1f} us")
        ttnn.deallocate(x_tt)
    ttnn.deallocate(w_t_tt)
    rt.deallocate()
    assert rows[-1][2] <= 45.0, rows


# =====================================================================================================================
# device: real weights and real router inputs
# =====================================================================================================================
@DEVICE
def test_device_real_layers(mesh_device):
    """Real router weights (layers 2-35, local shards) on real router inputs (reference post_attention_layernorm
    outputs): logit error vs fp64, top-8 set agreement (uniform token sample + near-tie stress set) of the exact
    router and of the FPU composite (G5) with the same device tail, bitwise check vs the host fp32 model."""
    if not REAL_FILE.is_file():
        pytest.skip(f"{REAL_FILE} missing: scripts/hostrun.sh -t 3000 -- python -m "
                    "models.demos.motif3.tests.unit.test_router_fp32 capture (CPU, ~8 min)")
    from models.demos.motif3.tt import weights as W

    _log_fabric(mesh_device, "router_fp32 real layers")
    data = torch.load(REAL_FILE, weights_only=True)
    source = W.HFWeightLoader()
    Rr, Cc = tuple(mesh_device.shape)
    n_slots = Rr * Cc * R.TOKENS
    ckc = _router_ckc()
    rows = []
    tot = {k: [0, 0] for k in ("exact_uni", "exact_near", "fpu_uni", "fpu_near", "exact_host_uni", "exact_host_near",
                               "fpu_host_uni", "fpu_host_near")}
    worst_exact = {"rel_to_absmax": 0.0}
    worst_fpu = {"rel_to_absmax": 0.0}
    bit_bad = bit_tot = 0
    w_err_exact = 0.0
    t_start = time.time()
    for L in REAL_LAYERS:
        if L not in data["layers"] or not source.layer_available(L):
            print(f"[router_fp32 real] layer {L}: skipped (not captured / not local)")
            continue
        d = data["layers"][L]
        w_t, b = W.router_weights(source.get(W.hf_name(L, "moe.router.gate.weight")),
                                  source.get(W.hf_name(L, "moe.expert_bias")))
        X = torch.cat([d["x"], d["x_near"]]).float()
        n_u = d["x"].shape[0]
        n = X.shape[0]
        Xp = torch.zeros(n_slots, H)
        Xp[:n] = X
        x_tt = _per_chip_x(mesh_device, Xp.reshape(Rr, Cc, R.TOKENS, H))
        bias_tt = _replicated(mesh_device, b.reshape(1, 1, 1, E), dtype=ttnn.float32)
        rt = R.RouterLogitsFP32(mesh_device, w_t)
        wt_tt = _replicated(mesh_device, w_t)
        res = {}
        for name, logits_tt in (("exact", rt(x_tt)), ("fpu", fpu_logits(x_tt, wt_tt, ckc))):
            idx_tt, wts_tt, biased_tt = router_tail(logits_tt, bias_tt)
            lg = _per_chip_out(mesh_device, logits_tt).reshape(n_slots, E)[:n]
            ix = _per_chip_out(mesh_device, idx_tt).reshape(n_slots, K)[:n]
            wv = _per_chip_out(mesh_device, wts_tt).reshape(n_slots, K)[:n]
            res[name] = (lg, ix, wv)
            for t in (logits_tt, idx_tt, wts_tt, biased_tt):
                ttnn.deallocate(t)
        idx_g, w_g, _ = golden_route(X, w_t, b)
        row = {"layer": L}
        for name, (lg, ix, wv) in res.items():
            agree = set_agreement(ix, idx_g)
            agree_host = set_agreement(host_route_from_logits(lg, b), idx_g)
            for sub, sl in (("uni", slice(0, n_u)), ("near", slice(n_u, n))):
                tot[f"{name}_{sub}"][0] += int(agree[sl].sum())
                tot[f"{name}_{sub}"][1] += int(agree[sl].numel())
                tot[f"{name}_host_{sub}"][0] += int(agree_host[sl].sum())
                tot[f"{name}_host_{sub}"][1] += int(agree_host[sl].numel())
            e = logit_errors(lg[:n_u], X[:n_u], w_t)
            row[f"{name}_flips_uni"] = int((~agree[:n_u]).sum())
            row[f"{name}_flips_near"] = int((~agree[n_u:]).sum())
            row[f"{name}_rel"] = e["rel_to_absmax"]
            row[f"{name}_max_abs"] = e["max_abs"]
            row[f"{name}_pcc"] = e["pcc"]
            if name == "exact":
                row["logit_rms"] = e["logit_rms"]
                if e["rel_to_absmax"] > worst_exact["rel_to_absmax"]:
                    worst_exact = dict(e, layer=L)
                ok = agree
                if bool(ok.any()):
                    w_err_exact = max(w_err_exact, float((_sorted_weights(ix[ok], wv[ok])
                                                          - _sorted_weights(idx_g[ok], w_g[ok])).abs().max()))
                for bi in range(2):  # bitwise vs the host fp32 model on two 32-token batches
                    emu = R.emulate_device_fp32(Xp[32 * bi : 32 * bi + 32], w_t)
                    bit_bad += _bits_mismatch(lg[32 * bi : 32 * bi + 32], emu)
                    bit_tot += emu.numel()
            elif e["rel_to_absmax"] > worst_fpu["rel_to_absmax"]:
                worst_fpu = dict(e, layer=L)
        rows.append(row)
        print(f"[router_fp32 real] L{L:02d} logit_rms {row['logit_rms']:.2f} | exact: rel {row['exact_rel']:.2e} "
              f"max_abs {row['exact_max_abs']:.2e} pcc {row['exact_pcc']:.9f} flips uni {row['exact_flips_uni']}/{n_u} "
              f"near {row['exact_flips_near']}/{n - n_u} | fpu: rel {row['fpu_rel']:.2e} flips uni "
              f"{row['fpu_flips_uni']}/{n_u} near {row['fpu_flips_near']}/{n - n_u}")
        for t in (x_tt, bias_tt, wt_tt):
            ttnn.deallocate(t)
        rt.deallocate()

    def pct(k):
        a, m = tot[k]
        return 100.0 * a / max(m, 1), m - a, m

    for k in tot:
        p, bad, m = pct(k)
        print(f"[router_fp32 real] agreement {k:16s} {p:8.4f} %  ({bad} flips / {m})")
    print(f"[router_fp32 real] exact logits: worst layer vs fp64 {worst_exact}")
    print(f"[router_fp32 real] fpu logits:   worst layer vs fp64 {worst_fpu}")
    print(f"[router_fp32 real] exact logits != host fp32 model (bitwise): {bit_bad} / {bit_tot}")
    print(f"[router_fp32 real] exact router weights max abs err on agreeing tokens: {w_err_exact:.3e}")
    print(f"[router_fp32 real] {len(rows)} layers in {time.time() - t_start:.0f} s")
    assert rows, "no real layer available"
    assert worst_exact["rel_to_absmax"] <= 1e-6
    assert pct("exact_uni")[0] >= 99.95
    assert bit_bad == 0


@DEVICE
def test_device_real_all_tokens(mesh_device):
    """Every prefill token of the reference prompt set (6 chats, 2971 tokens) for the 11 layers captured by the MoE
    test (``tests/unit/test_moe.py capture``; read only, skipped if absent). Routes compared **directly with the
    reference's own fp32 CPU routes** (``ref_idx``; weights vs ``ref_w``) and with fp64:

    * decode shape (32 tokens per chip and call, 3 calls per layer): exact kernel and FPU composite logits with this
      file's G5 tail, and ``tt/moe.py`` ``MotifRouter`` (production tail) with and without the kernel as ``logits_fn``;
    * prefill shape (one ``[1, 1, 2976, 4096]`` call per layer, replicated): the exact kernel (bitwise equal to its
      decode-shape logits and identical on all chips) and the FPU composite, G5 tail.
    """
    if not MOE_ALL_TOKENS.is_file():
        pytest.skip(f"{MOE_ALL_TOKENS} missing (produced by tests/unit/test_moe.py capture)")
    from models.demos.motif3.tt import weights as W
    from models.demos.motif3.tt.model_config import MotifTTConfig
    from models.demos.motif3.tt.moe import MotifRouter

    _log_fabric(mesh_device, "router_fp32 real all tokens")
    data = torch.load(MOE_ALL_TOKENS, weights_only=True)
    layers = data.get("layers", {})
    source = W.HFWeightLoader()
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)
    Rr, Cc = tuple(mesh_device.shape)
    n_chips = Rr * Cc
    n_slots = n_chips * R.TOKENS
    ckc = _router_ckc()
    names = ("exact", "fpu", "motif_router_exact", "motif_router_composite", "exact_prefill", "fpu_prefill")
    tot = {f"{k}_vs_{r}": [0, 0] for k in names for r in ("ref", "fp64")}
    tot["ref_vs_fp64"] = [0, 0]
    werr = {k: 0.0 for k in names}
    worst, worst_abs, min_pcc = 0.0, 0.0, 1.0
    checks = {"prefill_eq_decode_bits": True, "prefill_identical_on_chips": True}
    t_start = time.time()
    for L in sorted(int(k) for k in layers):
        d = layers[L] if L in layers else layers[str(L)]
        if not isinstance(d, dict) or "x" not in d or "ref_idx" not in d or not source.layer_available(L):
            continue
        X = d["x"].float()
        ref_idx = d["ref_idx"].long()
        ref_w = d["ref_w"].float()
        n_tok = X.shape[0]
        w_t, b = W.router_weights(source.get(W.hf_name(L, "moe.router.gate.weight")),
                                  source.get(W.hf_name(L, "moe.expert_bias")))
        idx_g, _, _ = golden_route(X, w_t, b)
        a_rr = set_agreement(ref_idx, idx_g)
        tot["ref_vs_fp64"][0] += int(a_rr.sum())
        tot["ref_vs_fp64"][1] += n_tok
        rt = R.RouterLogitsFP32(mesh_device, w_t)
        wt_tt = _replicated(mesh_device, w_t)
        bias_tt = _replicated(mesh_device, b.reshape(1, 1, 1, E), dtype=ttnn.float32)
        mr_exact = MotifRouter(mesh_device, cfg, L, source=source, cache=False,
                               logits_fn=R.RouterLogitsFP32(mesh_device, weight_tensor=rt.weight))
        mr_comp = MotifRouter(mesh_device, cfg, L, source=source, cache=False)
        got = {k: ([], []) for k in names}
        exact_logits = []
        for s0 in range(0, n_tok, n_slots):  # decode shape
            chunk = X[s0 : s0 + n_slots]
            n = chunk.shape[0]
            Xp = torch.zeros(n_slots, H)
            Xp[:n] = chunk
            x_tt = _per_chip_x(mesh_device, Xp.reshape(Rr, Cc, R.TOKENS, H))
            for name in ("exact", "fpu"):
                logits_tt = rt(x_tt) if name == "exact" else fpu_logits(x_tt, wt_tt, ckc)
                idx_tt, wts_tt, biased_tt = router_tail(logits_tt, bias_tt)
                got[name][0].append(_per_chip_out(mesh_device, idx_tt).reshape(n_slots, K)[:n])
                got[name][1].append(_per_chip_out(mesh_device, wts_tt).reshape(n_slots, K)[:n])
                if name == "exact":
                    lg = _per_chip_out(mesh_device, logits_tt).reshape(n_slots, E)[:n]
                    exact_logits.append(lg)
                    e = logit_errors(lg, chunk, w_t)
                    worst = max(worst, e["rel_to_absmax"])
                    worst_abs = max(worst_abs, e["max_abs"])
                    min_pcc = min(min_pcc, e["pcc"])
                _free([logits_tt, idx_tt, wts_tt, biased_tt])
            for name, mr in (("motif_router_exact", mr_exact), ("motif_router_composite", mr_comp)):
                ia, wa = mr(x_tt)
                got[name][0].append(_per_chip_out(mesh_device, ia).reshape(n_slots, K)[:n])
                got[name][1].append(_per_chip_out(mesh_device, wa).reshape(n_slots, K)[:n])
                _free([ia, wa])
            ttnn.deallocate(x_tt)
        # prefill shape: one call, all tokens (padded to a multiple of 32), replicated
        m = -(-n_tok // R.TOKENS) * R.TOKENS
        Xp = torch.zeros(m, H)
        Xp[:n_tok] = X
        x_tt = _replicated(mesh_device, Xp.reshape(1, 1, m, H))
        for name in ("exact_prefill", "fpu_prefill"):
            logits_tt = rt(x_tt) if name == "exact_prefill" else fpu_logits(x_tt, wt_tt, ckc)
            idx_tt, wts_tt, biased_tt = router_tail(logits_tt, bias_tt)
            got[name][0].append(_per_chip_out(mesh_device, idx_tt)[0, 0].reshape(m, K)[:n_tok])
            got[name][1].append(_per_chip_out(mesh_device, wts_tt)[0, 0].reshape(m, K)[:n_tok])
            if name == "exact_prefill":
                per = _per_chip_out(mesh_device, logits_tt).reshape(n_chips, m, E)
                checks["prefill_identical_on_chips"] &= all(_bits_mismatch(per[i], per[0]) == 0
                                                            for i in range(0, n_chips, 7))
                checks["prefill_eq_decode_bits"] &= _bits_mismatch(per[0][:n_tok], torch.cat(exact_logits)) == 0
            _free([logits_tt, idx_tt, wts_tt, biased_tt])
        ttnn.deallocate(x_tt)
        line = f"[router_fp32 all-tokens] L{L:02d}: {n_tok} tokens, flips vs ref / vs fp64:"
        for name in names:
            ix = torch.cat(got[name][0]).long()
            wv = torch.cat(got[name][1]).float()
            a_ref = set_agreement(ix, ref_idx)
            a_64 = set_agreement(ix, idx_g)
            tot[f"{name}_vs_ref"][0] += int(a_ref.sum())
            tot[f"{name}_vs_ref"][1] += n_tok
            tot[f"{name}_vs_fp64"][0] += int(a_64.sum())
            tot[f"{name}_vs_fp64"][1] += n_tok
            if bool(a_ref.any()):
                werr[name] = max(werr[name], float((_sorted_weights(ix[a_ref], wv[a_ref])
                                                    - _sorted_weights(ref_idx[a_ref], ref_w[a_ref])).abs().max()))
            line += f" {name} {int((~a_ref).sum())}/{int((~a_64).sum())}"
        print(line)
        mr_exact.deallocate()  # frees nothing of rt (borrowed weight)
        mr_comp.deallocate()
        _free([wt_tt, bias_tt])
        rt.deallocate()
    for k, (a, m) in tot.items():
        if m:
            print(f"[router_fp32 all-tokens] agreement {k:30s} {100.0 * a / m:8.4f} %  ({m - a} flips / {m})")
    print(f"[router_fp32 all-tokens] routing-weight max abs err vs ref_w on agreeing tokens: "
          + ", ".join(f"{k} {v:.2e}" for k, v in werr.items()))
    print(f"[router_fp32 all-tokens] exact logits vs fp64: worst rel error (to max |logit|) {worst:.3e}, max abs "
          f"{worst_abs:.3e}, min PCC {min_pcc:.12f}; {checks}; {time.time() - t_start:.0f} s")
    assert tot["exact_vs_ref"][1] > 0
    for name in ("exact", "motif_router_exact", "exact_prefill"):
        a, m = tot[f"{name}_vs_ref"]
        assert 100.0 * a / m >= 99.99, (name, a, m)
        assert 100.0 * tot[f"{name}_vs_fp64"][0] / m >= 99.95, name
        assert werr[name] <= 1e-5, (name, werr[name])
    assert worst <= 1e-6
    assert all(checks.values()), checks


# =====================================================================================================================
# device: trace, cache, back-to-back launches
# =====================================================================================================================
class _NoSource:
    """A weight source that must not be read (proves a TT-cache hit)."""

    def get(self, name):
        raise AssertionError(f"weight {name} read although the TT cache should have been hit")


@DEVICE
def test_device_trace_replay_and_cache(mesh_device):
    """Decode use: the prepared weight goes through the TT cache (``from_source``; the second construction must not
    touch the checkpoint); a trace holding 3 back-to-back launches is replayed with new per-chip inputs copied into
    the persistent input (``copy_host_to_device_tensor``): every output of every replay is bitwise equal to the host
    fp32 model on all 32 chips (semaphores re-armed across launches)."""
    from models.demos.motif3.tt import weights as W
    from models.demos.motif3.tt.model_config import MotifTTConfig

    _log_fabric(mesh_device, "router_fp32 trace + cache")
    source = W.HFWeightLoader()
    L = 2
    if not source.layer_available(L):
        pytest.skip(f"layer {L} not local")
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device, tt_cache_root=REAL_DIR / "tt_cache")
    rt = R.RouterLogitsFP32.from_source(mesh_device, cfg, L, source=source, cache=True)
    rt_hit = R.RouterLogitsFP32.from_source(mesh_device, cfg, L, source=_NoSource(), cache=True)
    w_t = W.router_weights(source.get(W.hf_name(L, "moe.router.gate.weight")),
                           source.get(W.hf_name(L, "moe.expert_bias")))[0]
    rt_nc = R.RouterLogitsFP32.from_source(mesh_device, cfg, L, source=source, cache=False)  # as_tensor, no cache
    a = _per_chip_out(mesh_device, rt.weight)
    b = _per_chip_out(mesh_device, rt_hit.weight)
    c = _per_chip_out(mesh_device, rt_nc.weight)
    host_w = _bf16(R.prepare_router_weight(w_t))
    cache_ok = torch.equal(a, b) and torch.equal(a, c) and torch.equal(a[0, 0].reshape(host_w.shape), host_w)
    owns = rt.owns_weight and rt_hit.owns_weight and rt_nc.owns_weight
    w_nc = rt_nc.weight
    rt_nc.deallocate()
    owns = owns and not w_nc.is_allocated()
    print(f"[router_fp32 trace] TT cache: reload bitwise equal to the fresh upload and to the host layout: {cache_ok}; "
          f"from_source instances own their weight: {owns}")

    Rr, Cc = tuple(mesh_device.shape)
    gamma = 0.014 + 0.33 * torch.rand(H, generator=torch.Generator().manual_seed(51))
    rounds = [torch.stack([random_x(R.TOKENS, gamma, seed=1000 * r + i) for i in range(Rr * Cc)]) for r in range(3)]
    x_tt = _per_chip_x(mesh_device, rounds[0].reshape(Rr, Cc, R.TOKENS, H))
    ttnn.deallocate(rt_hit(x_tt))  # compile
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        outs = [rt_hit(x_tt) for _ in range(3)]
    except BaseException:
        try:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        finally:
            ttnn.release_trace(mesh_device, tid)
        raise
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    bad = 0
    try:
        for r, xs in enumerate(rounds):
            _copy_in(mesh_device, xs.reshape(Rr, Cc, R.TOKENS, H), x_tt)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            for o in outs:
                bad += _mismatch_per_chip(mesh_device, o, xs, w_t)
            print(f"[router_fp32 trace] replay {r}: mismatching logits vs host fp32 model (3 launches x 32 chips) "
                  f"{bad}")
    finally:
        ttnn.release_trace(mesh_device, tid)
    _free(outs + [x_tt])
    rt.deallocate()
    rt_hit.deallocate()
    assert cache_ok and owns and bad == 0


@DEVICE
def test_device_interleaved_back_to_back(mesh_device):
    """Two instances (different weights) and three persistent inputs (two decode-shape, one 64-token prefill-shape),
    8 launches back to back with no other op and no sync in between (eager), then the same 8 captured in ONE trace
    and replayed 6 times with new per-chip data in every input. Every output must be bitwise equal to the host model:
    a cross-launch race on the receive buffers / semaphores, a decode / prefill program mix-up, or stale runtime args
    on a program-cache hit across instances would show up here (different data in every launch)."""
    _log_fabric(mesh_device, "router_fp32 interleaved back-to-back")
    Rr, Cc = tuple(mesh_device.shape)
    wA, gamma = random_router(101)
    wB, _ = random_router(202)
    inst = {"A": (R.RouterLogitsFP32(mesh_device, wA), wA), "B": (R.RouterLogitsFP32(mesh_device, wB), wB)}
    shapes = {"a": R.TOKENS, "b": R.TOKENS, "p": 2 * R.TOKENS}
    host = {k: _chip_xs(mesh_device, m, gamma, seed=5000 + 1000 * i, heavy=(k == "b")) for i, (k, m) in
            enumerate(shapes.items())}
    dev = {k: _per_chip_x(mesh_device, host[k].reshape(Rr, Cc, shapes[k], H)) for k in shapes}
    pattern = [("A", "a"), ("B", "b"), ("A", "p"), ("B", "a"), ("A", "b"), ("B", "p"), ("A", "a"), ("B", "b")]
    for i, xi in (("A", "a"), ("A", "p"), ("B", "a"), ("B", "p")):  # compile both variants of both instances
        ttnn.deallocate(inst[i][0](dev[xi]))
    ttnn.synchronize_device(mesh_device)

    outs = [inst[i][0](dev[xi]) for i, xi in pattern]  # eager, back to back, no sync
    ttnn.synchronize_device(mesh_device)
    bad_eager = sum(_mismatch_per_chip(mesh_device, o, host[xi], inst[i][1]) for o, (i, xi) in zip(outs, pattern))
    _free(outs)
    print(f"[router_fp32 b2b] eager back-to-back interleaved (8 launches x {Rr * Cc} chips): mismatches {bad_eager}")

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        touts = [inst[i][0](dev[xi]) for i, xi in pattern]
    except BaseException:
        try:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        finally:
            ttnn.release_trace(mesh_device, tid)
        raise
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    bad_trace = 0
    rounds = 6
    try:
        for r in range(rounds):
            cur = {k: _chip_xs(mesh_device, m, gamma, seed=10000 * (r + 1) + 1000 * i, heavy=(r + i) % 2 == 0)
                   for i, (k, m) in enumerate(shapes.items())}
            for k in shapes:
                _copy_in(mesh_device, cur[k].reshape(Rr, Cc, shapes[k], H), dev[k])
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            b = sum(_mismatch_per_chip(mesh_device, o, cur[xi], inst[i][1]) for o, (i, xi) in zip(touts, pattern))
            bad_trace += b
            print(f"[router_fp32 b2b] trace replay {r}: mismatches {b}")
    finally:
        ttnn.release_trace(mesh_device, tid)
    _free(touts + list(dev.values()))
    for rt, _ in inst.values():
        rt.deallocate()
    print(f"[router_fp32 b2b] traced interleaved: {rounds} replays x 8 launches x {Rr * Cc} chips, mismatches "
          f"{bad_trace}")
    assert bad_eager == 0 and bad_trace == 0


# =====================================================================================================================
# device: timing breakdown
# =====================================================================================================================
AICLK_MHZ = 1350.0  # BH AI clock; the RISC wall clock counts these cycles


def timing_breakdown(stamps: torch.Tensor) -> dict:
    """``stamps [96, 16]`` (kernels/router_fp32/timing.h slots) -> microseconds relative to the earliest start."""
    s = stamps.to(torch.int64) & 0xFFFFFFFF
    t0 = int(s[:, [0, 4, 8, 12]].min())

    def us(v):
        return (v - t0).double() / AICLK_MHZ

    ev = {
        "reader_start": us(s[:, 12]),
        "unpack_w0_ready": us(s[:, 1]),
        "unpack_x0_ready": us(s[:, 2]),
        "reads_done": us(s[:, 13]),
        "unpack_x3_ready": us(s[:, 3]),
        "math_pushed(~sfpu_done)": us(s[:, 5]),
        "partials_packed": us(s[:, 9]),
        "rs_writes_done": us(s[:, 10]),
        "rs_sems_done": us(s[:, 11]),
        "rs_all_received": us(s[:, 14]),
        "final_written": us(s[:, 15]),
    }
    return {k: (float(v.min()), float(v.median()), float(v.max())) for k, v in ev.items()}


@DEVICE
def test_device_timing_breakdown(mesh_device):
    """Per-core wall-clock stamps of one decode launch (chip 0): where the kernel's time goes."""
    _log_fabric(mesh_device, "router_fp32 timing")
    w_t, gamma = random_router(41)
    x = random_x(R.TOKENS, gamma, seed=42)
    x_tt = _replicated(mesh_device, x.reshape(1, 1, R.TOKENS, H))
    rt = R.RouterLogitsFP32(mesh_device, w_t, timing=True)
    for it in range(3):
        out = rt(x_tt)
        ttnn.synchronize_device(mesh_device)
        stamps = ttnn.to_torch(ttnn.get_device_tensors(rt.last_timing)[0]).reshape(R.N_WORKERS, -1)
        bd = timing_breakdown(stamps)
        if it == 2:
            for k, (lo, med, hi) in bd.items():
                print(f"[router_fp32 timing] {k:24s} min {lo:8.2f}  median {med:8.2f}  max {hi:8.2f} us")
        ttnn.deallocate(out)
        ttnn.deallocate(rt.last_timing)
    out = rt(x_tt)
    ok = _bits_mismatch(R.emulate_device_fp32(x, w_t), _per_chip_out(mesh_device, out)[0, 0].reshape(R.TOKENS, E)) == 0
    _free([out, rt.last_timing, x_tt])
    rt.deallocate()
    assert ok


def _partials_fp32(x: torch.Tensor, w_t: torch.Tensor) -> torch.Tensor:
    """Host fp32 model of every worker's 4 partial tiles: ``[96, 4, 32, 32]``, worker q = 32 g + h, tile j = experts
    ``128 g + 32 j ..`` of k group h (router_fp32.partials_fp32)."""
    acc = R.partials_fp32(x, w_t)  # [h, t, e]
    out = torch.empty(R.N_WORKERS, R.N_VEC, R.TILE, R.TILE)
    for q in range(R.N_WORKERS):
        g, h = divmod(q, R.K_GROUPS)
        for j in range(R.N_VEC):
            e0 = 128 * g + 32 * j
            out[q, j] = acc[h, :, e0 : e0 + 32]
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        capture_real_inputs()
    else:
        print("usage: python -m models.demos.motif3.tests.unit.test_router_fp32 capture")
