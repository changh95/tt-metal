# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the routed MoE (``tt/moe.py``; design §2.3.7, §3.2-3.3; WAVE_A_REVIEW §5.6 MOE-1..7, decision D1).

Host (CPU only; never touches the chips)::

    S=/home/ttuser/hchang/experiments/motif-3/scripts
    $S/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib -q -s \
        models/demos/motif3/tests/unit/test_moe.py -k host

* ``test_moe_host_dataflow_emulation``: the decode dataflow of ``MotifMoE`` (EP placement through
  ``weights.ep_layout`` / ``shard_for_device``, local mask from ``local_expert_ids``, per-chip combine, AR(dp) +
  partition(dp) + AR(tp)) emulated per chip in torch at real dims reproduces the reference routed output on (4, 8),
  (8, 4) and (1, 8) (48 experts per chip, size-1 DP axis).
* ``test_moe_host_size1_axis_never_frees_inputs``: the free logic of ``forward_decode`` / ``forward_prefill`` with
  stand-in tensors and collectives that hand their input back (as ``MotifCCL`` does on a size-1 axis): the caller's
  ``x`` / ``add_partial`` / taps and the result are never freed, nothing is freed twice, nothing leaks.
* ``test_moe_host_dtype_args``: ``None`` dtype arguments mean the documented bf16 default.
* ``test_moe_host_t64_decode_configs`` (T64, docs/p5_t64/P5_T64_DESIGN.md §4.2, R-E7): the router's per-M decode
  configs == ``model_config.router_decode_pc(m_tiles=)``, the config each decode M selects, the exact-fp32 logits at
  M = 32 / 64 only, the experts' M = 64 configs (gate_up 10 x 8, down 8 x 8, per_core_M 2), ``wide_decode_rows``, and
  ``forward_decode`` refusing a row count the module was not built for before any device op.
* ``test_moe_host_router_mask_scatter`` (A5): ``MotifRouter.route_local`` / ``local_partial`` with the ttnn ops
  emulated in torch: topk's sets (exact ties included) on every chip, weights equal to the gather path to fp32
  rounding, no ``ttnn.gather``, no leak or double free, the scatter path only at decode row counts with constants.
* ``test_moe_host_router_fused_numerics`` (B4): ``kernels/router_topk.emulate_fp32`` (the fused router tail, bit for bit
  the device kernel's algorithm) vs an independent fp64 / sort-based reference: the top-8 sets (exact ties: lower id),
  the weights to fp32 rounding, every chip's local slice, M = 64 rows == M = 32 rows bitwise, the key order;
  ``router_topk.plan`` refusals.
* ``test_moe_host_router_fused_dispatch`` (B4): ``MotifRouter.route_fused`` (taps, frees) and ``local_partial`` taking it
  only at decode row counts (before A5's scatter), ``deallocate``.
* ``test_moe_host_sparse_decode_experts`` (B1): ``local_partial`` with ``decode_experts="sparse"`` and the ttnn ops
  emulated in torch: live rows bitwise == dense, inactive rows 0, the sparsity = the experts live rows route to,
  ``sparse_matmul`` with ``nnz=None`` and the dense program configs, no leak / double free, the dense path kept for
  prefill / other row counts / the dense mode; ``resolve_decode_experts``.
* ``test_moe_host_sparse_lane_mask_wiring`` (B1): ``decode_lane_mask`` (order, frees, size-1 DP axis), the model's
  per-step build and free, decoder -> MoE -> ``local_partial`` hand-over.
* ``test_moe_host_import_clean``: importing ``tt.moe`` loads no other ``models/demos`` package.

Real router inputs (CPU capture, ~5 min, 270 MB; the device tests skip without it)::

    $S/hostrun.sh -t 3000 -- python -m models.demos.motif3.tests.unit.test_moe capture

Device (only through the lock wrapper; (4, 8) mesh, FABRIC_2D_TORUS_XY; each test logs the committed fabric)::

    $S/devrun.sh -t 1500 -n moe -- python -m pytest models/demos/motif3/tests/unit/test_moe.py -s -p no:cacheprovider -k device

Accuracy gates (every MoE output test): PCC >= 0.995 vs the fp32 reference on the reference's routes (>= 0.999 with bf16
experts) **and** a per-token PCC >= 0.999 for every token against the reference experts evaluated on the *device's*
routes (separates expert-path error from near-tie route flips; a zeroed tile or one bad lane fails it); in prefill every
token below 0.999 against the reference routes must be a route flip.

* ``test_moe_device_router_real``: router top-8 set agreement vs fp64 on real router inputs of 11 layers (D1 / MOE-7),
  weights error on agreeing tokens, routes bitwise identical on all 32 chips.
* ``test_moe_device_router_exact``: the exact-fp32 logits kernel (``tt/kernels/router_fp32.py``, another agent's)
  swapped in at decode shape: agreement on every real token of the 11 layers, cost, L2 MoE decode (skips if the kernel
  cannot run here).
* ``test_moe_device_router_variants``: every router op fusion / program config vs the plain G5 composite on all 2971
  real layer-2 tokens at prefill shape *and* in 93 decode-shape calls (the padded top-k and the decode program configs
  only apply there); the module default must give identical sets and weights (asserted).
* ``test_moe_device_polynorm_variants``: grouped PolyNorm impls (shared Horner / rms / local, fp32 / bf16, L1 / DRAM)
  vs an fp64 golden on device gate_up values (thresholds asserted); the multi-core topk.
* ``test_moe_device_decode_random`` (+ ``_8x4``, which also runs prefill there): random weights at real dims (384
  experts), bfp8 and bf16 experts, vs the fp32 reference MoE; replicas identical; gathered tokens exact; ``add_partial``
  / ``reduce_tp=False`` contracts; lane independence (zeroing one lane leaves the other 31 bitwise unchanged).
* ``test_moe_device_submesh_1x8``: the module on a (1, 8) submesh (size-1 DP axis, 48 experts per chip): decode and
  prefill accuracy, the caller's inputs stay allocated (the size-1-axis free fix).
* ``test_moe_device_decode_real[L2|L4]``: real weights and real inputs: default + variants (PolyNorm bf16, fp32
  partials, HF-order combine, PolyNorm impls rms / local, DRAM intermediates), L2 also bf16 experts with
  ``fold_route_scale=False`` (fp32-faithful, PCC >= 0.999); trace capture + replay with new inputs (traced == eager
  bitwise); eager / traced latency, latency variants, a traced per-stage breakdown (floor-aware) and the effective
  expert-weight bandwidth.
* ``test_moe_device_t64_rows[composite|exact_fp32]`` (WP-D D1; T64 verify step, 16 rows per DP row = 64 gathered
  tokens): real layer-2 weights from the serving TT cache; every output row bitwise == the 32-lane module's row of the
  same token on all 32 chips, routes bitwise, the 32-lane path bitwise == B0's committed module, the fp32 reference,
  traced == eager, the static-CB end below an L1 pin, traced cost M = 64 vs 32. Run::

      $S/devrun.sh -t 1500 -n moe_t64 -- python -m pytest models/demos/motif3/tests/unit/test_moe.py -s \
          -p no:cacheprovider -k t64_rows

* ``test_moe_device_router_fused[composite|exact_fp32]`` (B4, ``router_mask="fused"``): the top-8 contract of the
  fused router tail on every real token (sets == the gather path's, 100 %), the kernel == ``emulate_fp32`` bitwise,
  synthetic exact ties, output vs gather, T64 rows == T32 rows, rerun and trace replay bitwise, traced cost.
* ``test_moe_device_sparse_decode_experts`` (B1, ``decode_experts="sparse"``): real L2 / L35 weights and routes at
  M = 32 (1 / 8 / 32 live lanes) and M = 64 (2 / 16 / 64 live rows): live rows bitwise == dense, inactive rows exactly
  ``add_partial``, unmasked == dense on every row, the device lane mask == host, T64 rows == T32 rows, trace replays
  with new tokens and live sets == eager and deterministic, traced cost dense vs sparse.
* ``test_moe_device_prefill``: real layer-2 weights, S = 128 / 2048 real tokens and S = 4096 / 32768 (real tokens
  tiled) vs the reference (and on the device's routes), replicas identical, ``add_partial`` / ``reduce_tp=False``
  contracts (S = 128), ``prefill_polynorm="fp32"`` and ``prefill_pc=False`` (bitwise equal to the default), eager latency
  and a per-stage breakdown at S = 4096.
* ``test_moe_device_block_shared_real``: the MoE block as the decoder will call it -- ``MotifMoE`` + the real shared
  expert (``tt/mlp.py`` ``MotifSharedExpert``, its TP partial handed in as ``add_partial``) vs the reference MoE block
  (routed + shared in fp32), decode and prefill S = 128 / 2048.
* ``test_moe_device_prefill_matmul_variants`` (diagnostics): prefill expert-matmul configs; the module's in1-multicast
  configs must be bitwise identical to the auto config (asserted).

Goldens are the reference package modules (``reference.modules.Router`` / ``RoutedExperts`` / ``MLP``) run in fp32 on
the same bf16-valued weights ("fp32-faithful"); router set agreement is against an fp64 router on the same bf16 inputs.
Traced latencies use the gates' slope method (GATES_RESULTS §2): an op whose ``n/2`` calls stay under the ~150 us
``synchronize_device`` floor reads low or 0, so the breakdown doubles ``n`` until it clears the floor and otherwise
reports the ``t(n)/n`` upper bound.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ttnn

PROJECT_ROOT = Path("/home/ttuser/hchang/experiments/motif-3")
GOLDEN_DIR = Path(os.environ.get("MOTIF3_MOE_TEST_DIR", str(PROJECT_ROOT / "tt_cache" / "test" / "moe")))
REAL_INPUTS = GOLDEN_DIR / "real_router_inputs_v1.pt"
REAL_LAYERS = (2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 35)
PROMPTS_JSON = Path(__file__).resolve().parents[2] / "reference" / "prompts" / "messages.json"
HF_META = str(PROJECT_ROOT / "hf_meta")
H, E, K, I = 4096, 384, 8, 1280
TRACE = 256 * 1024 * 1024
TOKEN_PCC_MIN = 0.999  # every token vs the reference experts on the device's routes (review issue 3)


# ============================================================================================================
# real inputs (CPU capture)
# ============================================================================================================
def capture_real_router_inputs(out_path: Path = REAL_INPUTS, layers=REAL_LAYERS, max_prompts=None) -> Path:
    """CPU only (run under scripts/hostrun.sh, never while holding the device lock).

    Runs the reference prefix model (layers 0..max(layers), bf16 = HF numerics, lazy experts) over the reference
    prompt set (``reference/prompts/messages.json``, 6 chat prompts, 2971 tokens) and stores, per layer in ``layers``,
    the MoE input ``post_attention_layernorm.out`` (bf16, the router input) and the reference routing (indices,
    weights), concatenated over prompts: ``{"meta", "prompt_lens", "layers": {L: {"x", "ref_idx", "ref_w"}}}``.
    """
    from models.demos.motif3.reference.golden import capture_model_goldens
    from models.demos.motif3.reference.tokenizer import encode_chat, load_tokenizer
    from models.demos.motif3.reference.weights import load_reference_model

    layers = tuple(sorted(int(i) for i in layers))
    t0 = time.time()
    model = load_reference_model(layer_ids=range(max(layers) + 1), dtype=torch.bfloat16, lazy_experts=True)
    tok = load_tokenizer()
    prompts = json.loads(PROMPTS_JSON.read_text())["prompts"]
    if max_prompts is not None:
        prompts = prompts[: int(max_prompts)]
    want = set()
    for L in layers:
        want |= {
            f"layers.{L}.post_attention_layernorm.out",
            f"layers.{L}.moe.router.indices",
            f"layers.{L}.moe.router.weights",
        }
    per_layer = {L: {"x": [], "ref_idx": [], "ref_w": []} for L in layers}
    lens, names = [], []
    for p in prompts:
        ids = encode_chat(p["messages"], tok, add_generation_prompt=p.get("add_generation_prompt", True))
        g = capture_model_goldens(model, torch.tensor([ids]), include=lambda n: n in want)
        for L in layers:
            pre = g["prefill"]
            per_layer[L]["x"].append(pre[f"layers.{L}.post_attention_layernorm.out"].reshape(-1, H))
            per_layer[L]["ref_idx"].append(pre[f"layers.{L}.moe.router.indices"].reshape(-1, K).to(torch.int16))
            per_layer[L]["ref_w"].append(pre[f"layers.{L}.moe.router.weights"].reshape(-1, K).float())
        lens.append(len(ids))
        names.append(p.get("name"))
        print(f"[moe capture] prompt {p.get('name')}: {len(ids)} tokens, {time.time() - t0:.0f} s", flush=True)
    out = {
        "meta": {
            "format": "motif3-moe-router-inputs/1",
            "dtype": "bf16",
            "layers": list(layers),
            "prompts": names,
            "source": "reference load_reference_model(bf16, lazy_experts) + capture_model_goldens",
            "seconds": time.time() - t0,
        },
        "prompt_lens": lens,
        "layers": {
            int(L): {k: torch.cat(v, dim=0).contiguous() for k, v in d.items()} for L, d in per_layer.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    torch.save(out, tmp)
    os.replace(tmp, out_path)
    print(f"[moe capture] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s")
    return out_path


def load_real_inputs():
    if not REAL_INPUTS.is_file():
        pytest.skip(
            f"{REAL_INPUTS} missing: run scripts/hostrun.sh -t 3000 -- python -m models.demos.motif3.tests.unit.test_moe "
            "capture (CPU, ~5 min)"
        )
    return torch.load(REAL_INPUTS, weights_only=True)


def require_layer(src, layer: int):
    """README §12: real-weight tests skip (never download) when a layer is not local."""
    if not src.layer_available(layer):
        pytest.skip(f"layer {layer} weights not local")


def ref_routed_all_tokens(source, layer: int, xs: torch.Tensor, *, save: bool = True) -> torch.Tensor:
    """fp32 reference routed output of every captured token of ``layer`` (cached next to the inputs, 48 MB/layer;
    computed in ~20-60 s on the CPU if missing -- precompute with ``python -m ...test_moe golden <layer>``)."""
    path = GOLDEN_DIR / f"ref_routed_L{layer:02d}_v1.pt"
    if path.is_file():
        d = torch.load(path, weights_only=True)
        if d["n"] == xs.shape[0] and torch.equal(d["x_sum"], xs.float().sum(0)):
            return d["routed"]
    t0 = time.time()
    with torch.no_grad():
        routed, idx, w = RefMoE(source, layer)(xs)
    print(f"[moe] reference routed output for {xs.shape[0]} tokens of L{layer}: {time.time() - t0:.1f} s")
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"n": xs.shape[0], "x_sum": xs.float().sum(0), "routed": routed, "idx": idx, "w": w}, path)
    return routed


def spread_tokens(n_total: int, n: int) -> torch.Tensor:
    """``n`` token indices spread evenly over ``[0, n_total)`` (covers every prompt of the set)."""
    return torch.linspace(0, n_total - 1, n).round().long()


# ============================================================================================================
# goldens (pure torch)
# ============================================================================================================
def ref_args():
    from models.demos.motif3.reference.config import MotifArgs

    return MotifArgs.from_hf_config(HF_META)


class RefMoE:
    """fp32 reference routed MoE: ``reference.modules.Router`` + ``RoutedExperts`` (HF loop, fp32 accumulation) over a
    weight source with HF names, weights bf16-valued as stored, math in fp32 ("fp32-faithful" golden)."""

    def __init__(self, source, layer: int, args=None):
        from torch import nn

        from models.demos.motif3.reference.modules import RoutedExperts, Router
        from models.demos.motif3.tt.weights import hf_name

        args = args or ref_args()
        self.layer = layer
        g = lambda s: source.get(hf_name(layer, s))  # noqa: E731
        self.router = Router(args.hidden_size, args.num_experts, args.experts_top_k, args.score_func, args.route_norm,
                             args.route_scale)
        self.router.gate.weight = nn.Parameter(g("moe.router.gate.weight").float(), requires_grad=False)
        self.expert_bias = g("moe.expert_bias").float()
        self.experts = RoutedExperts(args, layer, materialize=False)
        gu, dn = g("moe.experts.gate_up_proj"), g("moe.experts.down_proj")
        self.experts.expert_source = lambda e: (gu[e].float(), dn[e].float())
        self.experts.act_fn.weight = nn.Parameter(g("moe.experts.act_fn.weight").float(), requires_grad=False)
        self.experts.act_fn.bias = nn.Parameter(g("moe.experts.act_fn.bias").float(), requires_grad=False)
        self.gate_w = g("moe.router.gate.weight")

    @torch.no_grad()
    def route(self, x):
        w, idx = self.router(x.float(), self.expert_bias)
        return idx, w

    @torch.no_grad()
    def __call__(self, x, idx=None, w=None):
        """``x [N, 4096]`` -> ``(routed fp32 [N, 4096], idx [N, 8], w [N, 8])`` (``idx``/``w`` given: those routes)."""
        if idx is None:
            idx, w = self.route(x)
        return self.experts(x.float(), idx.long(), w.float()), idx, w


def ref_shared_expert(source, layer: int, args=None):
    """fp32 reference shared expert (``reference.modules.MLP`` built as in ``reference.modules.MoE``) on HF weights."""
    from torch import nn

    from models.demos.motif3.reference.modules import MLP
    from models.demos.motif3.tt.weights import hf_name

    a = args or ref_args()
    m = MLP(a.hidden_size, a.shared_intermediate_size, eps=a.polynorm_eps, sigmoid_weight=a.polynorm_sigmoid_weight,
            hidden_clamp=a.hidden_clamp, output_scale=a.polynorm_output_scale_for_layer(layer))
    g = lambda s: nn.Parameter(source.get(hf_name(layer, f"moe.shared_experts.{s}")).float(),  # noqa: E731
                               requires_grad=False)
    m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight = g("gate_proj.weight"), g("up_proj.weight"), g(
        "down_proj.weight")
    m.act_fn.weight, m.act_fn.bias = g("act_fn.weight"), g("act_fn.bias")
    return m


def router_golden(x, gate_w, bias, k: int = K, scale: float = 2.0):
    """fp64 router on the same bf16-valued x / W / bias: ``(idx [N, k], w [N, k], biased [N, E])``."""
    s = torch.sigmoid(x.double() @ gate_w.double().T)
    biased = s + bias.double().reshape(1, -1)
    idx = torch.topk(biased, k, dim=-1).indices
    w = s.gather(-1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * scale
    return idx, w, biased


def set_agreement(idx_dev, idx_ref, biased_ref):
    """(fraction of tokens with equal top-8 sets, eq mask [N], 8th-9th golden margin [N])."""
    a = torch.sort(idx_dev.long(), dim=-1).values
    b = torch.sort(idx_ref.long(), dim=-1).values
    eq = (a == b).all(dim=-1)
    top = torch.topk(biased_ref, k=idx_ref.shape[-1] + 1, dim=-1).values
    margin = (top[:, -2] - top[:, -1]).double()
    return float(eq.float().mean()), eq, margin


def same_sets(idx_a, idx_b) -> torch.Tensor:
    """[N] bool: the two routes select the same set of experts."""
    return (torch.sort(idx_a.long(), dim=-1).values == torch.sort(idx_b.long(), dim=-1).values).all(dim=-1)


def matched_weight_err(idx_dev, w_dev, idx_ref, w_ref, eq):
    """Max |w_dev - w_ref| over tokens whose sets agree (compared in expert-index order)."""
    if not bool(eq.any()):
        return float("nan")
    od = torch.argsort(idx_dev.long(), dim=-1)
    orf = torch.argsort(idx_ref.long(), dim=-1)
    wd = w_dev.double().gather(-1, od)[eq]
    wr = w_ref.double().gather(-1, orf)[eq]
    return float((wd - wr).abs().max())


def pcc(a, b) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    den = float(a.norm() * b.norm())
    return float((a @ b) / den) if den > 0 else (1.0 if torch.equal(a, b) else 0.0)


def row_pcc(ref, got) -> torch.Tensor:
    """PCC per row of ``[N, W]`` tensors (fp64; a zero row gives 0)."""
    a = ref.double() - ref.double().mean(-1, keepdim=True)
    b = got.double() - got.double().mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-300)


def stats(ref, got) -> dict:
    ref = ref.double()
    got = got.double()
    d = (ref - got).abs()
    per_tok = row_pcc(ref, got) if ref.dim() == 2 else torch.tensor([pcc(ref, got)])
    return {
        "pcc": pcc(ref, got),
        "min_token_pcc": float(per_tok.min()),
        "worst_token": int(per_tok.argmin()),
        "max_abs": float(d.max()),
        "ref_absmax": float(ref.abs().max()),
        "rel_fro": float((ref - got).norm() / max(float(ref.norm()), 1e-30)),
    }


def fmt(s: dict) -> str:
    return (
        f"pcc={s['pcc']:.6f} min_tok_pcc={s['min_token_pcc']:.6f} max_abs={s['max_abs']:.3e} "
        f"(|ref|max {s['ref_absmax']:.2f}) rel_fro={s['rel_fro']:.3e}"
    )


def _maxabs(t: torch.Tensor) -> float:
    return float(t.abs().max()) if t.numel() else 0.0


def frac_not_bf16(t: torch.Tensor) -> float:
    """Fraction of values that bf16 cannot represent (0 for a bf16 tensor, or an fp32 copy of one)."""
    t = t.float()
    return float((t.bfloat16().float() != t).double().mean())


# ============================================================================================================
# random weights at real dims (HF names)
# ============================================================================================================
def _randn_experts(n: int, shape, seed: int, std: float) -> torch.Tensor:
    out = torch.empty(n, *shape, dtype=torch.bfloat16)

    def one(e):
        g = torch.Generator().manual_seed(seed * 1000003 + e)
        out[e].copy_(torch.randn(*shape, generator=g) * std)

    with ThreadPoolExecutor(min(48, os.cpu_count() or 8)) as ex:
        list(ex.map(one, range(n)))
    return out


def random_moe_tensors(layer: int, seed: int = 0, bias_std: float = 0.1) -> dict:
    """Random MoE weights at real dims with the reference's scales (``reference.weights.random_state_dict``): router
    N(0, 1/D), expert_bias 1 + N(0, bias_std), experts N(0, 1/fan_in), PolyNorm w N(0, 1), bias U(-1, 1) (the
    routed +-0.5 clamp is exercised). bf16 values (the checkpoint dtype); generated in parallel (~5 s)."""
    from models.demos.motif3.tt.weights import hf_name

    g = torch.Generator().manual_seed(seed)
    t = {
        "moe.router.gate.weight": (torch.randn(E, H, generator=g) * H**-0.5).bfloat16(),
        "moe.expert_bias": (1.0 + bias_std * torch.randn(E, generator=g)).bfloat16(),
        "moe.experts.act_fn.weight": torch.randn(E, 3, generator=g).bfloat16(),
        "moe.experts.act_fn.bias": (torch.rand(E, 1, generator=g) * 2 - 1).bfloat16(),
        "moe.experts.gate_up_proj": _randn_experts(E, (2 * I, H), seed * 7 + 1, H**-0.5),
        "moe.experts.down_proj": _randn_experts(E, (H, I), seed * 7 + 2, I**-0.5),
    }
    return {hf_name(layer, k): v for k, v in t.items()}


def random_inputs(n: int, seed: int) -> torch.Tensor:
    """MoE inputs like ``post_attention_layernorm`` outputs: rmsnorm(h) * gamma, gamma U(0.5, 1.5), bf16 values."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, H, generator=g)
    gamma = 0.5 + torch.rand(H, generator=g)
    return (h / h.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt() * gamma).bfloat16()


# ============================================================================================================
# host tests (CPU)
# ============================================================================================================
def emulate_decode_dataflow(tensors: dict, layer: int, cfg, x32: torch.Tensor, idx, w) -> torch.Tensor:
    """Torch emulation of ``MotifMoE.forward_decode`` per chip with the given routes ``idx/w [32, 8]``: EP placement
    from ``weights.ep_layout`` + ``shard_for_device(dp_dim=0, tp_dim=1)``, local mask from ``local_expert_ids``, the
    weighted sum of the chip's experts on all 32 tokens, AR over DP, this row's lanes, AR over TP. Returns
    ``[32, 4096]`` fp32 in lane order (row ``dp``'s output for its ``lanes_per_row`` lanes)."""
    from models.demos.motif3.tt import weights as W

    a = cfg.axes
    R, C = a.mesh_shape
    hf = lambda s: tensors[W.hf_name(layer, s)]  # noqa: E731
    gu_ep = W.ep_layout(W.experts_gate_up(hf("moe.experts.gate_up_proj")), cfg)
    dn_ep = W.ep_layout(W.experts_down(hf("moe.experts.down_proj"), cfg.polynorm_output_scale), cfg)
    pn = W.expert_polynorm_tensors(hf("moe.experts.act_fn.weight"), hf("moe.experts.act_fn.bias"), cfg)
    ids = W.local_expert_ids(cfg)
    xf = x32.float()
    part = {}
    for r in range(R):
        for c in range(C):
            loc = lambda t: W.shard_for_device(t, a, r, c, dp_dim=0, tp_dim=1)[0]  # noqa: E731
            gu_c, dn_c = loc(gu_ep).float(), loc(dn_ep).float()  # [e_loc, 4096, 2560], [e_loc, 1280, 4096]
            ids_c = loc(ids).reshape(-1)  # [e_loc]
            w_loc = ((idx.long()[None] == ids_c.long()[:, None, None]).float() * w.float()[None]).sum(-1)  # [e_loc, 32]
            gu = torch.matmul(xf[None], gu_c)  # [e_loc, 32, 2560]
            g, u = gu[..., :I], gu[..., I:]
            c0, c1, c2, b = (loc(pn[k]).reshape(-1, 1, 1) for k in ("c0", "c1", "c2", "b"))
            nrm = lambda z: z / torch.sqrt(z.pow(2).mean(-1, keepdim=True) + cfg.polynorm_eps)  # noqa: E731
            h = (c0 * nrm(g**3) + c1 * nrm(g**2) + c2 * nrm(g) + b) * u
            y = torch.matmul(h, dn_c)  # [e_loc, 32, 4096]
            part[(r, c)] = (w_loc[..., None] * y).sum(0)  # [32, 4096]
            del gu_c, dn_c, gu, g, u, h, y
    out = torch.zeros(32, H)
    for dp in range(a.dp_size):
        acc = torch.zeros(cfg.lanes_per_row, H)
        for tp in range(a.tp_size):
            col_sum = sum(part[a.coord(d, tp)] for d in range(a.dp_size))  # AR over DP (fixed TP column)
            acc += col_sum.reshape(a.dp_size, cfg.lanes_per_row, H)[dp]  # partition(dim=2, "dp") -> AR over TP
        out[cfg.lanes_per_row * dp : cfg.lanes_per_row * (dp + 1)] = acc
    return out


@torch.no_grad()
def test_moe_host_dataflow_emulation():
    from models.demos.motif3.tt.model_config import MotifTTConfig
    from models.demos.motif3.tt.weights import DictWeightSource

    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    L = 2
    tensors = random_moe_tensors(L, seed=3)
    ref = RefMoE(DictWeightSource(tensors), L)
    x32 = random_inputs(32, seed=4)
    want, idx, w = ref(x32)
    got = emulate_decode_dataflow(tensors, L, cfg, x32, idx, w)
    s = stats(want, got)
    print(f"\n[moe host] dataflow emulation (4,8) vs reference routed output: {fmt(s)}")
    assert s["rel_fro"] < 1e-5 and s["pcc"] > 0.999999, s
    # (8, 4) mesh: TP axis is dim 0; (1, 8): size-1 DP axis, 48 experts per chip, 32 lanes in the one row
    for shape in ((8, 4), (1, 8)):
        c2 = MotifTTConfig.from_hf_config(HF_META, mesh_shape=shape)
        s2 = stats(want, emulate_decode_dataflow(tensors, L, c2, x32, idx, w))
        print(f"[moe host] dataflow emulation {shape} (experts/chip {c2.experts_per_chip}): {fmt(s2)}")
        assert s2["rel_fro"] < 1e-5, (shape, s2)


class _FakeT:
    """Stand-in for a ttnn.Tensor in the host free-logic test (no device, no ttnn tensor is created)."""

    _n = 0

    def __init__(self, name, shape=(1, 1, 32, H), dtype=None):
        _FakeT._n += 1
        self.name = f"{name}#{_FakeT._n}"
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype if dtype is not None else ttnn.bfloat16
        self.layout = ttnn.TILE_LAYOUT

    def __repr__(self):
        return self.name


@pytest.mark.parametrize("part_dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("dp1, tp1", [(True, False), (False, True), (True, True), (False, False)],
                         ids=["dp1", "tp1", "dp1tp1", "full"])
def test_moe_host_size1_axis_never_frees_inputs(monkeypatch, dp1, tp1, part_dtype):
    """Review issue 1: on a size-1 mesh axis every MotifCCL collective returns its input (``tt/ccl.py``), so the
    module's frees must skip tensors that came back unchanged. Stand-in tensors and collectives; ``ttnn.deallocate``
    and the few ttnn ops of the decode / prefill glue are recorded instead of run."""
    import models.demos.motif3.tt.moe as M

    pdt = ttnn.float32 if part_dtype == "fp32" else ttnn.bfloat16
    created, freed = [], []

    def new(name, like=None, shape=None, dtype=None):
        t = _FakeT(name, shape or (like.shape if like is not None else (1, 1, 32, H)),
                   dtype or (like.dtype if like is not None else ttnn.bfloat16))
        created.append(t)
        return t

    def concat(ts, dim, **kw):
        shape = list(ts[0].shape)
        shape[dim] = sum(t.shape[dim] for t in ts)
        return new("concat", ts[0], shape=shape)

    monkeypatch.setattr(M.ttnn, "deallocate", lambda t, *a, **k: freed.append(t))
    monkeypatch.setattr(M.ttnn, "add", lambda a, b, **k: new("add", a))
    monkeypatch.setattr(M.ttnn, "typecast", lambda a, dt, **k: new("typecast", a, dtype=dt))
    monkeypatch.setattr(M.ttnn, "slice", lambda a, s, e, **k: new("slice", a, shape=[int(q) - int(p) for p, q in
                                                                                       zip(s, e)]))
    monkeypatch.setattr(M.ttnn, "concat", concat)

    class FakeCCL:  # a size-1 axis hands its input back, like tt/ccl.py
        def _r(self, x, size1, name, shape=None):
            return x if size1 else new(name, x, shape=shape)

        def ag_dp_rows(self, x, memory_config=None):
            return self._r(x, dp1, "ag_dp_rows", (1, 1, 32, H))

        def ar_dp(self, x, **kw):
            return self._r(x, dp1, "ar_dp")

        def partition(self, x, dim, axis, **kw):
            assert axis == "dp"
            return self._r(x, dp1, "partition", (1, 1, x.shape[2] // (1 if dp1 else 4), H))

        def ar_tp(self, x, **kw):
            return self._r(x, tp1, "ar_tp")

        def rs_dp(self, x, dim=2, **kw):
            return self._r(x, dp1, "rs_dp", (1, 1, x.shape[2] // (1 if dp1 else 4), H))

        def ag_dp(self, x, dim=2, **kw):
            return self._r(x, dp1, "ag_dp", (1, 1, x.shape[2] * (1 if dp1 else 4), H))

    moe = M.MotifMoE.__new__(M.MotifMoE)
    moe.ccl = FakeCCL()
    moe.decode_mc = moe.dram = "mc"
    moe.decode_polynorm = moe.prefill_polynorm = "fp32"
    moe.hidden = H
    moe.prefill_chunk = 128
    moe.cfg = SimpleNamespace(dp=1 if dp1 else 4)
    moe.decode_rows = M.DECODE_ROWS  # a module built with a T64 config: 32 and 64 gathered decode rows

    def local_partial(f, *, polynorm, decode, taps=None, memory_config=None, lane_mask=None):
        if taps is not None:
            taps["idx"], taps["w"], taps["w_loc"] = new("idx", f), new("w", f), new("w_loc", f)
        return new("part", f, dtype=pdt)

    moe.local_partial = local_partial

    def check(tag, call, inputs, taps=None):
        created.clear()
        freed.clear()
        out = call()
        protected = list(inputs) + (list(taps.values()) if taps else [])
        ids = [id(t) for t in freed]
        assert len(ids) == len(set(ids)), f"{tag}: double free {freed}"
        bad = [t for t in freed if any(t is p for p in protected) or t is out]
        assert not bad, f"{tag}: freed a caller-owned tensor or the result: {bad}"
        leaked = [t for t in created if t is not out and not any(t is f for f in freed)
                  and not any(t is p for p in protected)]
        assert not leaked, f"{tag}: leaked {leaked}"
        assert out.dtype == (ttnn.bfloat16 if ("reduce_tp=False" not in tag) else out.dtype), (tag, out.dtype)
        return out

    # 8 lanes per DP row (32 on a size-1 DP axis); with 4 DP rows also the T64 step's 16 rows per DP row (M = 64)
    for rows in ((32,) if dp1 else (8, 16)):
        x = _FakeT("x", (1, 1, rows, H))
        for ap_dt in (ttnn.bfloat16, ttnn.float32):
            ap = _FakeT("add_partial", (1, 1, rows, H), dtype=ap_dt)
            check(f"decode {rows} add_partial {ap_dt}", lambda: moe.forward_decode(x, add_partial=ap), [x, ap])
        check(f"decode {rows}", lambda: moe.forward_decode(x), [x])
        check(f"decode {rows} reduce_tp=False", lambda: moe.forward_decode(x, reduce_tp=False), [x])
        taps = {}
        check(f"decode {rows} taps", lambda: moe.forward_decode(x, taps=taps), [x], taps)
    for S in (128, 256):  # one chunk (x itself is the chunk) and two chunks of 128
        xp = _FakeT("x_prefill", (1, 1, S, H))
        check(f"prefill S={S}", lambda: moe.forward_prefill(xp), [xp])
        ap = _FakeT("add_partial", (1, 1, S // moe.cfg.dp, H))
        check(f"prefill S={S} add_partial", lambda: moe.forward_prefill(xp, add_partial=ap), [xp, ap])
        check(f"prefill S={S} reduce_tp=False", lambda: moe.forward_prefill(xp, reduce_tp=False), [xp])


def test_moe_host_dtype_args():
    """Review issue 9: ``None`` dtype arguments select the documented default (bf16), never a silent fp32."""
    from models.demos.motif3.tt.moe import _dtype_of

    assert _dtype_of(None) == ttnn.bfloat16
    assert _dtype_of(None, ttnn.float32) == ttnn.float32
    assert _dtype_of("fp32") == ttnn.float32 and _dtype_of("float32") == ttnn.float32
    assert _dtype_of("BF16") == ttnn.bfloat16 and _dtype_of("bfloat16") == ttnn.bfloat16
    assert _dtype_of(ttnn.float32) == ttnn.float32 and _dtype_of(ttnn.bfloat16) == ttnn.bfloat16
    with pytest.raises(ValueError):
        _dtype_of("int8")


def test_moe_host_t64_decode_configs(monkeypatch):
    """T64 (docs/p5_t64/P5_T64_DESIGN.md §4.2, T4, review edit R-E7), host side of the 64-row decode: the router's
    per-M decode configs equal ``model_config.router_decode_pc(m_tiles=)`` (M = 32 / 64, sigmoid fused or not); the
    config each decode-shape M selects (a replaced ``decode_pc`` runs without its fused twin; other M take the auto
    config); the exact-fp32 logits replacement applies at M = 32 and 64 only; ``experts()`` passes the M = 64 configs
    (gate_up 10 x 8, down 8 x 8, ``per_core_M`` 2) at decode M = 64 and the G6 ones at 32; ``wide_decode_rows``; and
    ``forward_decode`` refuses a row count the module was not built for before any device op."""
    import models.demos.motif3.tt.moe as M
    from models.demos.motif3.tt.model_config import EXPERTS_DOWN_GRID_WIDE, EXPERTS_GATE_UP_GRID, MotifTTConfig

    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    assert M.DECODE_ROWS == (32, 64) and M.EXACT_ROUTER_DECODE_ROWS == (32, 64)
    stub = SimpleNamespace(n_experts=cfg.num_experts, cfg=cfg)
    pcs = {}
    for m in (1, 2):
        for sig in (False, True):
            pc = pcs[m, sig] = M.MotifRouter._decode_pc(stub, sigmoid=sig, m_tiles=m)
            assert repr(pc) == repr(cfg.router_decode_pc(sigmoid=sig, m_tiles=m)), (m, sig)
            assert (pc.per_core_M, pc.out_block_h, pc.out_subblock_h, pc.per_core_N) == (m, m, 1, 1), (m, sig)
    assert repr(M.MotifRouter._decode_pc(stub)) == repr(pcs[1, False])  # the M = 32 default is unchanged
    # the decode-shape selection (the constructor's config attributes, no device)
    r = object.__new__(M.MotifRouter)
    r.decode_pc, r.decode_pc_sigmoid = pcs[1, False], pcs[1, True]
    r._decode_pc_base = r.decode_pc
    r.decode_pcs_wide = {64: (pcs[2, False], pcs[2, True])}
    shape = lambda m: SimpleNamespace(shape=[1, 1, m, H])  # noqa: E731
    assert r._decode_pcs(32) == (pcs[1, False], pcs[1, True]) and r._pc(shape(32)) is pcs[1, False]
    assert r._decode_pcs(64) == (pcs[2, False], pcs[2, True]) and r._pc(shape(64)) is pcs[2, False]
    for m in (8, 96, 128, 2976):
        assert r._decode_pcs(m) == (None, None) and r._pc(shape(m)) is None, m
    other = M.MotifRouter._decode_pc(stub)
    r.decode_pc = other  # a diagnostics config at M = 32: no fused twin (separate sigmoid), M = 64 unaffected
    assert r._decode_pcs(32) == (other, None) and r._decode_pcs(64) == (pcs[2, False], pcs[2, True])
    r.decode_pc = None
    assert r._decode_pcs(32) == (None, None)
    r.logits_fn = None
    assert not r._use_logits_fn(shape(32)) and not r._use_logits_fn(shape(64))
    r.logits_fn = object()
    got = {m: r._use_logits_fn(shape(m)) for m in (32, 64, 96, 128, 4096)}
    assert got == {32: True, 64: True, 96: False, 128: False, 4096: False}, got
    # wide_decode_rows: the T64 step's 64 gathered rows when the config stages it, else 0
    assert M.wide_decode_rows(cfg) == 0
    for mesh_shape in ((4, 8), (8, 4)):
        for mode in ("wide", "auto"):
            c = MotifTTConfig.from_hf_config(HF_META, mesh_shape=mesh_shape, spec_tokens=1, spec_verify=mode)
            assert M.wide_decode_rows(c) == 64, (mesh_shape, mode)
    packed = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8), spec_tokens=1, spec_verify="packed")
    assert M.wide_decode_rows(packed) == 0
    with pytest.raises(ValueError, match="no MoE decode configs"):
        M.wide_decode_rows(SimpleNamespace(dp=4, wide_rows_per_dp=24))
    # experts(): the program configs each M selects (matmuls recorded instead of run)
    seen = []
    monkeypatch.setattr(M.ttnn, "repeat", lambda f, reps, **kw: SimpleNamespace(shape=list(f.shape)))
    monkeypatch.setattr(M.ttnn, "matmul", lambda a, b, **kw: seen.append((b, kw["program_config"])) or a)
    monkeypatch.setattr(M.ttnn, "deallocate", lambda *a, **k: None)
    moe = object.__new__(M.MotifMoE)
    moe.e_loc, moe.inter, moe.hidden, moe.dram = cfg.experts_per_chip, I, H, "dram"
    moe.gate_up_dtype, moe.down_dtype, moe.ckc_experts = None, ttnn.bfloat16, "ckc"
    moe.w_gate_up, moe.w_down, moe.prefill_pc = "W_gate_up", "W_down", True
    moe.polynorm_impl = moe.prefill_polynorm_impl = "horner"
    moe.polynorm = lambda gu, **kw: gu
    moe.pc_gate_up, moe.pc_down = cfg.experts_gate_up_pc(), cfg.experts_down_pc()
    moe.pc_wide = {64: (cfg.experts_gate_up_pc(m_tiles=2), cfg.experts_down_pc(m_tiles=2))}  # as the constructor
    for m, decode, want in ((32, True, (moe.pc_gate_up, moe.pc_down)), (64, True, moe.pc_wide[64]),
                            (96, True, (None, None)), (128, False, (None, None))):  # fmt: skip
        seen.clear()
        M.MotifMoE.experts(moe, shape(m), polynorm="fp32", decode=decode)
        assert [s[0] for s in seen] == ["W_gate_up", "W_down"] and tuple(s[1] for s in seen) == tuple(want), m
    gu64, dn64 = moe.pc_wide[64]
    grid = lambda pc: (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)  # noqa: E731
    assert grid(gu64) == EXPERTS_GATE_UP_GRID and grid(dn64) == EXPERTS_DOWN_GRID_WIDE
    assert (gu64.per_core_M, gu64.per_core_N, dn64.per_core_M, dn64.per_core_N) == (2, 1, 2, 2)
    # forward_decode: a gathered row count the module was not built for raises before any device op
    moe.cfg, moe.layer_idx = SimpleNamespace(dp=4), 2
    moe.ccl = SimpleNamespace(ag_dp_rows=lambda *a, **k: pytest.fail("a device op ran before the row check"))
    moe.decode_rows = (32,)  # built without a T64 config
    with pytest.raises(ValueError, match="wide_rows_per_dp"):
        moe.forward_decode(SimpleNamespace(shape=[1, 1, 16, H]))
    moe.decode_rows = (32, 64)
    for rows in (4, 12, 24, 32):
        with pytest.raises(ValueError, match="gathers"):
            moe.forward_decode(SimpleNamespace(shape=[1, 1, rows, H]))


class _TT:
    """A torch tensor posing as a ttnn tensor in the A5 host emulation (shape / dtype / layout like ttnn's)."""

    def __init__(self, t, dtype=None, layout=None):
        self.t = t
        self.shape = tuple(t.shape)
        self.dtype = dtype if dtype is not None else (ttnn.float32 if t.dtype == torch.float32 else ttnn.bfloat16)
        self.layout = layout if layout is not None else ttnn.TILE_LAYOUT


@torch.no_grad()
def test_moe_host_router_mask_scatter(monkeypatch):
    """A5 (docs/OPTIMIZATION_PLAN.md §3.3; ``router_mask="scatter"``, ``MOTIF3_ROUTER_MASK``), host side:
    ``MotifRouter.route_local`` and ``MotifMoE.local_partial`` with the ttnn ops emulated in torch (same op sequence,
    fp32 math). On random scores with exact 8th / 9th ties, on every chip of the (4, 8) mesh: the scatter path selects
    exactly topk's experts (the union over the 32 chips is topk's set, 8 per token, ties included), its weights match
    the gather path (router + ``local_weights``) to fp32 rounding, it never calls ``ttnn.gather``, frees every
    intermediate once and leaks nothing; ``local_partial`` takes it only at decode row counts that have constants
    (prefill and other M keep the gather path); taps carry ``idx`` / ``sel`` / ``w_loc``."""
    import models.demos.motif3.tt.moe as M
    import models.demos.motif3.tt.weights as W
    from models.demos.motif3.tt.model_config import ROUTER_MASK_MODES, MotifTTConfig

    assert ROUTER_MASK_MODES == ("gather", "scatter", "fused")
    monkeypatch.delenv("MOTIF3_ROUTER_MASK", raising=False)  # the default, whatever the caller exported (review I-2)
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    assert cfg.router_mask == "fused"  # the default since the Phase B eval (B4); gather / scatter are opt-in
    E, K, Mrows = cfg.num_experts, cfg.top_k, 32
    g = torch.Generator().manual_seed(5)
    scores = torch.rand(1, 1, Mrows, E, generator=g)
    bias = (torch.rand(1, 1, 1, E, generator=g) - 0.5) * 0.1
    biased = scores + bias
    for r in (3, 11, 20):  # exact fp32 ties at the 8th / 9th biased value: copy the 8th onto the 9th expert
        order = torch.argsort(biased[0, 0, r], descending=True)
        a, b = int(order[7]), int(order[8])
        scores[0, 0, r, b] = scores[0, 0, r, a] + bias[0, 0, 0, a] - bias[0, 0, 0, b]
        biased[0, 0, r, b] = biased[0, 0, r, a]
    # topk's choice on ties is the device's; emulate it as "the higher id wins" (not torch's), which the mask must follow
    idx = torch.stack([torch.tensor(sorted(range(E), key=lambda j: (-float(biased[0, 0, i, j]), -j))[:K])
                       for i in range(Mrows)]).reshape(1, 1, Mrows, K)
    tie_rows = [r for r in range(Mrows) if (torch.sort(biased[0, 0, r], descending=True).values[7:9].diff() == 0).any()]
    assert tie_rows == [3, 11, 20]

    created, freed, calls = [], [], []

    def new(t, **kw):
        x = _TT(t, **kw)
        created.append(x)
        return x

    def rec(name):
        calls.append(name)

    def mul(a, b, *, input_tensor_b_activations=None, dtype=None, memory_config=None):
        rec("multiply")
        bt = b.t if isinstance(b, _TT) else torch.tensor(float(b))
        if input_tensor_b_activations:
            bt = 1.0 / (bt + 1e-20)
        return new(a.t.float() * bt.float())

    def tsum(a, dim, keepdim=True, **kw):
        rec("sum")
        return new(a.t.sum(dim=dim, keepdim=keepdim))

    def scatter(z, dim, i, src, **kw):
        rec("scatter")
        return new(z.t.clone().scatter_(dim, i.t.long(), src.t), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    monkeypatch.setattr(M.ttnn, "deallocate", lambda t, *a, **k: freed.append(t))
    monkeypatch.setattr(M.ttnn, "multiply", mul)
    monkeypatch.setattr(M.ttnn, "sum", tsum)
    monkeypatch.setattr(M.ttnn, "scatter", scatter)
    monkeypatch.setattr(M.ttnn, "to_layout", lambda a, layout, **k: rec("to_layout") or new(a.t.clone(), layout=layout))
    monkeypatch.setattr(M.ttnn, "gather", lambda a, dim, i, **k: rec("gather") or new(torch.gather(a.t, dim, i.t.long())))
    monkeypatch.setattr(M.ttnn, "typecast", lambda a, dt, **k: rec("typecast") or new(a.t.float()))
    monkeypatch.setattr(M.ttnn, "eq", lambda a, b, **k: rec("eq") or new((a.t == b.t).float()))

    r = object.__new__(M.MotifRouter)
    r.route_scale, r.ckc_eltwise, r.dram, r.fuse_normalize = 2.0, "ckc", "dram", True
    r._scores_topk = lambda f, mc: (new(scores.clone()), new(biased.clone()), new(idx.clone(), dtype=ttnn.uint32))
    lmask_all, ids_all = W.local_expert_mask(cfg), W.local_expert_ids(cfg)  # [dp, 96, 1, 384], [dp, 96, 1, 1]
    consts = (_TT(torch.zeros(1, 1, Mrows, E, dtype=torch.bfloat16)), _TT(torch.ones(1, 1, Mrows, K, dtype=torch.bfloat16)))
    f = _TT(torch.zeros(1, 1, Mrows, H, dtype=torch.bfloat16))
    union = torch.zeros(Mrows, E, dtype=torch.bool)
    for dp in range(cfg.dp):
        for tp in range(cfg.tp):
            sl = slice(cfg.experts_per_chip * tp, cfg.experts_per_chip * (tp + 1))
            lmask = _TT(lmask_all[dp, sl].unsqueeze(0).contiguous())
            moe = object.__new__(M.MotifMoE)
            moe.router, moe.dram, moe.ckc_eltwise, moe.internal_route_scale = r, "dram", "ckc", 1.0
            moe.local_ids = _TT(ids_all[dp, sl].unsqueeze(0).contiguous())
            # scatter path
            created.clear(), freed.clear(), calls.clear()
            w_new = r.route_local(f, lmask, consts, scale=1.0, memory_config="L1")
            assert "gather" not in calls and calls.count("scatter") == 1, calls
            ids = [id(t) for t in freed]
            assert len(ids) == len(set(ids)) and not any(t is w_new for t in freed), "double free / freed the result"
            leaked = [t for t in created if t is not w_new and not any(t is q for q in freed)]
            assert not leaked, leaked
            # gather path (today's router tail + local_weights)
            created.clear(), freed.clear(), calls.clear()
            i_old, w_old = M.MotifRouter.__call__(r, f, scale=1.0, memory_config="L1")
            wl_old = M.MotifMoE.local_weights(moe, i_old, w_old, memory_config="L1")
            assert "gather" in calls
            a, b = w_new.t.reshape(-1, Mrows).t(), wl_old.t.reshape(-1, Mrows).t()  # [M, 12]
            assert torch.equal(a != 0, b != 0), (dp, tp)
            torch.testing.assert_close(a, b, rtol=2e-6, atol=0.0)
            k0 = cfg.experts_per_chip * (cfg.tp * dp + tp)
            union[:, k0:k0 + cfg.experts_per_chip] |= a != 0
    want = torch.zeros(Mrows, E, dtype=torch.bool).scatter_(1, idx.reshape(Mrows, K), True)
    assert torch.equal(union, want) and (union.sum(-1) == K).all()
    # taps of the scatter path: idx and sel stay alive for the caller
    created.clear(), freed.clear()
    taps = {}
    w_t = r.route_local(f, lmask, consts, scale=2.0, taps=taps, memory_config="L1")
    assert set(taps) == {"idx", "sel"} and not any(t is v for t in freed for v in taps.values())
    assert torch.equal(taps["sel"].t.reshape(Mrows, E) != 0, want)
    # local_partial: the scatter path only at decode row counts with constants
    seen = []
    moe = object.__new__(M.MotifMoE)
    moe.dram, moe.internal_route_scale, moe.combine_mode, moe.local_mask = "dram", 1.0, "fold", "LMASK"
    moe.scatter_consts = {32: "C32", 64: "C64"}
    moe.experts = lambda f, **kw: "Y"
    moe.reduce_experts = lambda y, **kw: "PART"

    class _R:
        def route_local(self, f, lm, c, **kw):
            seen.append(("scatter", c))
            return "WLOC"

        def __call__(self, f, **kw):
            seen.append(("gather", int(f.shape[-2])))
            return "IDX", "W"

    moe.router = _R()
    moe.local_weights = lambda i, w, **kw: "WLOC_G"
    monkeypatch.setattr(M, "_free", lambda *a: None)
    for m, decode, want_path in ((32, True, ("scatter", "C32")), (64, True, ("scatter", "C64")),
                                 (96, True, ("gather", 96)), (32, False, ("gather", 32)), (4096, False, ("gather", 4096))):
        seen.clear()
        t = {}
        assert M.MotifMoE.local_partial(moe, SimpleNamespace(shape=[1, 1, m, H]), polynorm="fp32", decode=decode,
                                        taps=t) == "PART"
        assert seen == [want_path], (m, decode, seen)
        assert t["w_loc"] in ("WLOC", "WLOC_G") and (("w" in t) == (want_path[0] == "gather")), t
    moe.scatter_consts = {}  # router_mask="gather": no constants, every decode call takes the gather path
    seen.clear()
    M.MotifMoE.local_partial(moe, SimpleNamespace(shape=[1, 1, 32, H]), polynorm="fp32", decode=True)
    assert seen == [("gather", 32)]


def _router_fused_reference(scores, bias, k, scale):
    """Independent reference for the B4 kernel: top-``k`` of ``scores + bias`` (fp32 values, ranked descending, exact
    ties to the lower id, via Python sorting of fp64-exact values) and fp64 weights ``scale s / (sum s + 1e-20)``.
    ``scores [M, E]`` fp32 -> ``(idx [M, k] long, w [M, k] fp64)``."""
    b = (scores.float() + bias.float().reshape(1, -1)).double()  # fp32 sum (exact in fp64)
    idx, w = [], []
    for r in range(scores.shape[0]):
        order = sorted(range(scores.shape[1]), key=lambda j: (-float(b[r, j]), j))[:k]
        s = scores[r, order].double()
        idx.append(order)
        w.append(s / (s.sum() + 1e-20) * scale)
    return torch.tensor(idx), torch.stack(w)


@torch.no_grad()
def test_moe_host_router_fused_numerics():
    """B4 (docs/OPTIMIZATION_PLAN.md §3.3 "A5 and B4"; ``router_mask="fused"``), host side of the fused router tail
    (``tt/kernels/router_topk.py``; the device kernel matches ``emulate_fp32`` bitwise on all 35,652 real token-layers,
    logs/opt/phaseB/B4): on random fp32 scores with negative biases and exact ties at the 8th value (8th = 9th and
    7th = 8th = 9th), M = 32 and 64:

    * the top-8 ids equal the independent reference (rank by the fp32 biased value, exact ties -> lower id), 8 distinct
      ids per row, rank order;
    * the weights agree with fp64 ``s / sum(s)`` to fp32 rounding (rel <= 4 ulp), ``scale`` 2.0 doubles them exactly;
    * every chip's ``w_loc`` slice (base ``12 k``, 12 experts) is the full row's slice, the union of the 32 chips is
      the top-8 set, everything else exactly 0;
    * rows are independent: the 64-row call's rows equal the two 32-row calls bitwise (the T64 contract);
    * ``order_keys`` orders like the fp32 values (negatives, zeros, denormals) and ``plan`` refuses unsupported shapes."""
    from models.demos.motif3.tt.kernels import router_topk as RT

    E_, K_, El = 384, 8, 12
    g = torch.Generator().manual_seed(11)
    for M_ in (32, 64):
        scores = torch.sigmoid(torch.randn(M_, E_, generator=g) * 2.0).float()
        bias = ((torch.rand(E_, generator=g) - 0.6) * 0.2).float()
        biased = scores + bias
        ties = []
        for r in range(0, M_, 7):  # 8th = 9th (and every 3rd of those also 7th = 8th)
            order = torch.argsort(biased[r], descending=True)
            a, b = int(order[7]), int(order[8])
            scores[r, b] = biased[r, a] - bias[b]
            if (scores[r, b] + bias[b]) != biased[r, a]:
                continue  # not representable as an exact fp32 tie: skip this row
            if r % 3 == 0:
                c = int(order[6])
                scores[r, c] = biased[r, a] - bias[c]
            biased = scores + bias
            ties.append(r)
        srt = torch.sort(scores + bias, dim=-1, descending=True).values
        n_ties = int((srt[:, 7] == srt[:, 8]).sum())
        assert n_ties >= len(ties) // 2 > 0, (n_ties, ties)
        w_full, idx = RT.emulate_fp32(scores.reshape(1, 1, M_, E_), bias, top_k=K_, base=0, e_loc=E_)
        idx = idx.reshape(M_, K_)
        w_full = w_full.reshape(E_, M_).t()  # [M, E]
        ref_i, ref_w = _router_fused_reference(scores, bias, K_, 1.0)
        assert torch.equal(idx, ref_i)
        assert all(len(set(row.tolist())) == K_ for row in idx)
        w_sel = torch.gather(w_full, 1, idx).double()
        rel = ((w_sel - ref_w).abs() / ref_w).max()
        assert rel <= 4 * 2.0 ** -24, float(rel)
        assert int((w_full != 0).sum()) == M_ * K_
        w2, _ = RT.emulate_fp32(scores.reshape(1, 1, M_, E_), bias, top_k=K_, base=0, e_loc=E_, scale=2.0)
        assert torch.equal(w2.reshape(E_, M_).t(), w_full * 2.0)
        union = torch.zeros(M_, E_, dtype=torch.bool)
        for k in range(E_ // El):
            wl, _ = RT.emulate_fp32(scores.reshape(1, 1, M_, E_), bias, top_k=K_, base=El * k, e_loc=El)
            assert tuple(wl.shape) == (1, 1, El, M_, 1)
            sl = wl.reshape(El, M_).t()
            assert torch.equal(sl.view(torch.int32), w_full[:, El * k:El * (k + 1)].view(torch.int32))
            union[:, El * k:El * (k + 1)] = sl != 0
        assert torch.equal(union, torch.zeros(M_, E_, dtype=torch.bool).scatter_(1, idx, True))
        if M_ == 64:
            halves = [RT.emulate_fp32(scores[h * 32:(h + 1) * 32].reshape(1, 1, 32, E_), bias, top_k=K_, base=0,
                                      e_loc=E_)[0].reshape(E_, 32).t() for h in (0, 1)]
            assert torch.equal(torch.cat(halves).view(torch.int32), w_full.view(torch.int32))
    vals = torch.tensor([-3.0, -1.0, -1e-40, -0.0, 0.0, 1e-40, 1e-30, 0.5, 1.0, 7.0], dtype=torch.float32)
    keys = RT.order_keys(vals)
    assert torch.equal(torch.argsort(keys), torch.arange(len(vals))) and len(set(keys.tolist())) == len(vals)
    p = RT.plan(64, 384, 8, 12, (12, 10))
    assert (p["R"], p["NT"], p["n_workers"]) == (2, 12, 64)
    assert RT.plan(32, 384, 8, 48, (12, 10))["E_loc"] == 48  # (1, 8) submesh: 48 experts per chip
    for bad in (dict(M=48), dict(M=128 * 2), dict(n_experts=100), dict(top_k=17), dict(e_loc=63)):
        kw = dict(M=32, n_experts=384, top_k=8, e_loc=12, grid=(12, 10))
        kw.update(bad)
        with pytest.raises(ValueError, match="fused router"):
            RT.plan(**kw)


def test_moe_host_router_fused_dispatch(monkeypatch):
    """B4 host side of the MoE wiring (no device): ``MotifRouter.route_fused`` runs the router matmul (``_scores``) and
    the kernel with the module's internal route scale and memory config; without taps the scores are freed once and the
    result is not; with taps ``idx`` / ``scores`` stay alive (the kernel is asked for ``idx``). ``local_partial`` takes
    the fused path only at the module's decode row counts (before A5's scatter constants), the gather router for
    prefill, for other row counts and without a kernel; ``deallocate`` releases the kernel."""
    import models.demos.motif3.tt.moe as M
    from models.demos.motif3.tt.model_config import ROUTER_MASK_MODES

    assert ROUTER_MASK_MODES == ("gather", "scatter", "fused")
    freed, calls = [], []
    monkeypatch.setattr(M.ttnn, "deallocate", lambda t, *a, **k: freed.append(t))

    class FakeKernel:
        def __init__(self):
            self.released = False

        def __call__(self, scores, *, scale, memory_config, want_idx=False):
            calls.append(("kernel", scores.tag, scale, memory_config, want_idx))
            w = SimpleNamespace(tag="w_loc")
            return (w, SimpleNamespace(tag="idx")) if want_idx else w

        def deallocate(self):
            self.released = True

    r = object.__new__(M.MotifRouter)
    r.route_scale, r.dram = 2.0, "dram"
    r._scores = lambda f, mc: calls.append(("scores", mc)) or SimpleNamespace(tag="scores")
    k = FakeKernel()
    w = r.route_fused(SimpleNamespace(shape=[1, 1, 32, H]), k, scale=1.0, memory_config="L1")
    assert w.tag == "w_loc" and calls == [("scores", "L1"), ("kernel", "scores", 1.0, "L1", False)]
    assert [t.tag for t in freed] == ["scores"]
    calls.clear(), freed.clear()
    taps = {}
    w = r.route_fused(SimpleNamespace(shape=[1, 1, 64, H]), k, taps=taps)
    assert calls == [("scores", "dram"), ("kernel", "scores", 2.0, "dram", True)] and not freed
    assert taps["idx"].tag == "idx" and taps["scores"].tag == "scores"

    seen = []

    class _R:
        def route_fused(self, f, kern, **kw):
            seen.append(("fused", kern))
            return "WLOC_F"

        def route_local(self, f, lm, c, **kw):
            seen.append(("scatter", c))
            return "WLOC_S"

        def __call__(self, f, **kw):
            seen.append(("gather", int(f.shape[-2])))
            return "IDX", "W"

    moe = object.__new__(M.MotifMoE)
    moe.dram, moe.internal_route_scale, moe.combine_mode, moe.local_mask = "dram", 1.0, "fold", "LMASK"
    moe.decode_rows, moe.router, moe.router_fused = (32, 64), _R(), "KERN"
    moe.scatter_consts = {32: "C32", 64: "C64"}  # a fused module never has them; fused must win anyway
    moe.experts = lambda f, **kw: "Y"
    moe.reduce_experts = lambda y, **kw: "PART"
    moe.local_weights = lambda i, w, **kw: "WLOC_G"
    monkeypatch.setattr(M, "_free", lambda *a: None)
    for m, decode, want in ((32, True, ("fused", "KERN")), (64, True, ("fused", "KERN")), (96, True, ("gather", 96)),
                            (32, False, ("gather", 32)), (2048, False, ("gather", 2048))):
        seen.clear()
        t = {}
        assert M.MotifMoE.local_partial(moe, SimpleNamespace(shape=[1, 1, m, H]), polynorm="fp32", decode=decode,
                                        taps=t) == "PART"
        assert seen == [want], (m, decode, seen)
        assert ("w" in t) == (want[0] == "gather"), t
    moe.decode_rows = (32,)  # a module without the T64 step: M = 64 is not a decode row count
    seen.clear()
    M.MotifMoE.local_partial(moe, SimpleNamespace(shape=[1, 1, 64, H]), polynorm="fp32", decode=True)
    assert seen == [("scatter", "C64")]
    moe.router_fused, moe.decode_rows = None, (32, 64)
    seen.clear()
    M.MotifMoE.local_partial(moe, SimpleNamespace(shape=[1, 1, 32, H]), polynorm="fp32", decode=True)
    assert seen == [("scatter", "C32")]
    del moe.router_fused  # a module built before B4 (no attribute)
    moe.scatter_consts = {}
    seen.clear()
    M.MotifMoE.local_partial(moe, SimpleNamespace(shape=[1, 1, 32, H]), polynorm="fp32", decode=True)
    assert seen == [("gather", 32)]
    # deallocate releases the kernel (no device memory of its own)
    moe2 = object.__new__(M.MotifMoE)
    moe2.router = SimpleNamespace(deallocate=lambda: None)
    moe2.pn_consts = SimpleNamespace(deallocate=lambda: None)
    moe2.w_gate_up = moe2.w_down = moe2.local_ids = None
    kk = FakeKernel()
    moe2.router_fused = kk
    M.MotifMoE.deallocate(moe2)
    assert kk.released and moe2.router_fused is None


@torch.no_grad()
def test_moe_host_sparse_decode_experts(monkeypatch):
    """B1 (docs/OPTIMIZATION_PLAN.md §3.3; ``decode_experts="sparse"``, ``MOTIF3_DECODE_EXPERTS``; probe
    logs/opt/phaseA/M6), host side with the ttnn ops emulated in torch (small dims, fp32 math):

    * ``local_partial`` at M = 32 / 64 with 1 / 8 / all live lanes: the masked routing weights are 0 on inactive rows,
      the sparsity holds exactly the local experts some *live* row routes to (an expert routed only by inactive rows
      is skipped), the partial equals the dense path's on every live row bitwise and is 0 on inactive rows;
    * ``sparse_matmul``: ``nnz=None`` always (a static count deadlocks on BH), ``(a, b)`` sparse flags ``(False,
      True)`` then ``(True, True)``, the dense decode program configs of that M, the same compute config and dtypes;
      no ``repeat`` / ``matmul``;
    * ``lane_mask=None``: still exact on every row (inactive rows' experts are then computed too);
    * every intermediate freed once, nothing leaks, the caller's ``f`` / ``lane_mask`` never freed; taps keep the
      masked ``w_loc`` and ``sparsity``;
    * the dense mode, prefill (``decode=False``) and a row count outside ``decode_rows`` keep the dense path and
      ignore the lane mask;
    * ``resolve_decode_experts``: the config default, an explicit mode, bad modes, and "sparse" needs the "fold"
      combine (explicit raises, the config default falls back to "dense")."""
    import models.demos.motif3.tt.moe as M
    from models.demos.motif3.tt.model_config import DECODE_EXPERTS_MODES, MotifTTConfig

    assert DECODE_EXPERTS_MODES == ("dense", "sparse")
    monkeypatch.delenv("MOTIF3_DECODE_EXPERTS", raising=False)
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    assert cfg.decode_experts == "sparse"  # the default since 2026-10-07 (logs/opt/phaseB2/B1-FLIP)
    # ---- resolve_decode_experts ----
    R = M.resolve_decode_experts
    assert R(None, cfg, "fold") == "sparse" and R("sparse", cfg, "fold") == "sparse" and R("dense", cfg, "fold") == "dense"
    assert R(None, cfg, "multiply_sum") == "dense"  # the config default falls back for the HF-order diagnostic module
    sp = SimpleNamespace(decode_experts="sparse")
    assert R(None, sp, "fold") == "sparse" and R(None, sp, "multiply_sum") == "dense" and R("dense", sp, "fold") == "dense"
    assert R(None, SimpleNamespace(), "fold") == "dense"
    with pytest.raises(ValueError, match="combine_mode"):
        R("sparse", cfg, "multiply_sum")
    with pytest.raises(ValueError, match="decode_experts"):
        R("skip", cfg, "fold")

    Hs, Is, El = 64, 16, 12
    created, freed, calls = [], [], []

    def new(t, **kw):
        x = _TT(t, **kw)
        created.append(x)
        return x

    def mul(a, b, *, memory_config=None, **kw):
        calls.append(("multiply",))
        return new(a.t.float() * b.t.float())

    def tmax(a, dim, keepdim=True, memory_config=None):
        calls.append(("max",))
        return new(a.t.amax(dim=dim, keepdim=keepdim))

    def typecast(a, dt, memory_config=None, **kw):
        calls.append(("typecast", dt))
        return new(a.t.to(torch.bfloat16) if dt == ttnn.bfloat16 else a.t.float(), dtype=dt, layout=a.layout)

    def to_layout(a, layout, memory_config=None, **kw):
        calls.append(("to_layout", layout))
        return new(a.t.clone(), dtype=a.dtype, layout=layout)

    def reshape(a, shape, **kw):
        calls.append(("reshape",))
        # a copy (stand-ins have no buffer address): moe._reshape frees the input, the copy is tracked
        return new(a.t.reshape(tuple(int(v) for v in shape)).clone(), dtype=a.dtype, layout=a.layout)

    def sparse_matmul(a, b, *, sparsity, program_config, nnz, is_input_a_sparse, is_input_b_sparse, memory_config,
                      compute_kernel_config, dtype):
        calls.append(("sparse_matmul", program_config, nnz, is_input_a_sparse, is_input_b_sparse,
                      compute_kernel_config, dtype))
        on = sparsity.t.reshape(-1).float() != 0
        assert sparsity.layout == ttnn.ROW_MAJOR_LAYOUT and tuple(sparsity.shape) == (1, 1, 1, El)
        Mr = a.shape[-2]
        out = torch.zeros(El, Mr, b.shape[-1])
        for e in range(El):
            if on[e]:
                ae = a.t[0, e] if is_input_a_sparse else a.t[0, 0]
                out[e] = ae.float() @ b.t[0, e].float()
        if dtype == ttnn.bfloat16:
            out = out.to(torch.bfloat16)
        return new(out.reshape(1, El, Mr, -1) if is_input_a_sparse else out.reshape(1, 1, 1, El, Mr, -1), dtype=dtype)

    def repeat(a, shape, memory_config=None):
        calls.append(("repeat",))
        return new(a.t.expand(1, El, -1, -1).clone())

    def matmul(a, b, *, program_config, compute_kernel_config, dtype, memory_config):
        calls.append(("matmul", program_config, compute_kernel_config, dtype))
        out = torch.stack([a.t[0, e].float() @ b.t[0, e].float() for e in range(El)]).unsqueeze(0)  # per expert
        return new(out.to(torch.bfloat16) if dtype == ttnn.bfloat16 else out, dtype=dtype)

    monkeypatch.setattr(M.ttnn, "deallocate", lambda t, *a, **k: freed.append(t))
    for name, fn in (("multiply", mul), ("max", tmax), ("typecast", typecast), ("to_layout", to_layout),
                     ("reshape", reshape), ("sparse_matmul", sparse_matmul), ("repeat", repeat), ("matmul", matmul)):
        monkeypatch.setattr(M.ttnn, name, fn)
    monkeypatch.setattr(M.ttnn, "Shape", lambda v: list(v))

    g = torch.Generator().manual_seed(11)
    w_gu = _TT(torch.randn(1, El, Hs, 2 * Is, generator=g).to(torch.bfloat16))
    w_dn = _TT(torch.randn(1, El, Is, Hs, generator=g).to(torch.bfloat16))

    def make_moe(mode, M_rows_w_loc):
        moe = object.__new__(M.MotifMoE)
        moe.dram, moe.internal_route_scale, moe.combine_mode, moe.decode_experts = "dram", 1.0, "fold", mode
        moe.decode_rows, moe.e_loc, moe.inter, moe.hidden = (32, 64), El, Is, Hs
        moe.pc_gate_up, moe.pc_down, moe.pc_wide = "PC_GU32", "PC_DN32", {64: ("PC_GU64", "PC_DN64")}
        moe.prefill_pc, moe.ckc_experts, moe.gate_up_dtype, moe.down_dtype = False, "CKC", None, ttnn.bfloat16
        moe.polynorm_impl = moe.prefill_polynorm_impl = "horner"
        moe.w_gate_up, moe.w_down = w_gu, w_dn
        moe.scatter_consts = {}
        moe.router = lambda f, **kw: (new(torch.zeros(1)), new(torch.zeros(1)))
        moe.local_weights = lambda i, w, **kw: new(M_rows_w_loc.clone())

        def polynorm(gu, *, mode, row_scale=None, memory_config=None, impl=None):
            calls.append(("polynorm", impl))
            gg, uu = gu.t[..., :Is].float(), gu.t[..., Is:].float()
            return new((torch.tanh(gg) * (row_scale.t.float() * uu)).to(torch.bfloat16))

        moe.polynorm = polynorm
        moe.reduce_experts = lambda y, **kw: calls.append(("reduce",)) or new(y.t.float().sum(1, keepdim=True))
        return moe

    def routes(Mr, live, seed):
        """w_loc [1, 12, Mr, 1]: every row routes to ~2 local experts (0 elsewhere); expert 11 only by inactive rows."""
        gg = torch.Generator().manual_seed(seed)
        w = torch.zeros(1, El, Mr, 1)
        for r in range(Mr):
            for e in torch.randperm(El - 1, generator=gg)[:2].tolist():
                w[0, e, r, 0] = 0.05 + 0.4 * float(torch.rand(1, generator=gg))
        dead = [r for r in range(Mr) if r not in live]
        if dead:
            w[0, El - 1, dead[0], 0] = 0.3
        return w

    for Mr in (32, 64):
        for nlive in (1, 8, Mr):
            live = sorted(torch.randperm(Mr, generator=g)[:nlive].tolist())
            w_loc = routes(Mr, live, 100 * Mr + nlive)
            mask = torch.zeros(1, 1, Mr, 1)
            mask[0, 0, live, 0] = 1.0
            lm = _TT(mask)
            f = _TT(torch.randn(1, 1, Mr, Hs, generator=g).to(torch.bfloat16))
            # dense reference (today's path)
            calls.clear()
            dense = M.MotifMoE.local_partial(make_moe("dense", w_loc), f, polynorm="fp32", decode=True, lane_mask=lm)
            assert not any(c[0] in ("sparse_matmul", "multiply", "max") for c in calls), calls  # mask ignored
            assert [c[1] for c in calls if c[0] == "matmul"] == (["PC_GU32", "PC_DN32"] if Mr == 32 else
                                                                 ["PC_GU64", "PC_DN64"])
            dense_mm = [c for c in calls if c[0] == "matmul"]
            for use_mask in (True, False):
                moe = make_moe("sparse", w_loc)
                created.clear(), freed.clear(), calls.clear()
                part = M.MotifMoE.local_partial(moe, f, polynorm="fp32", decode=True, lane_mask=lm if use_mask else None)
                tag = (Mr, nlive, use_mask)
                assert not any(c[0] in ("repeat", "matmul") for c in calls), (tag, calls)
                smm = [c for c in calls if c[0] == "sparse_matmul"]
                assert len(smm) == 2 and all(c[2] is None for c in smm), (tag, smm)  # nnz=None always
                assert [(c[3], c[4]) for c in smm] == [(False, True), (True, True)], tag
                assert [c[1] for c in smm] == [c[1] for c in dense_mm], tag  # the dense program configs of this M
                assert [(c[5], c[6]) for c in smm] == [(c[2], c[3]) for c in dense_mm], tag  # compute config, dtypes
                assert ("polynorm", "horner") in calls
                # live rows bitwise == dense; inactive rows 0 with the mask
                assert torch.equal(part.t[0, 0, live], dense.t[0, 0, live]), tag
                if use_mask and nlive < Mr:
                    dead = [r for r in range(Mr) if r not in live]
                    assert float(part.t[0, 0, dead].abs().max()) == 0.0, tag
                if not use_mask:
                    assert torch.equal(part.t, dense.t), tag  # unmasked: exact on every row
                # memory: only the partial survives; caller's tensors never freed; nothing freed twice
                ids = [id(t) for t in freed]
                assert len(ids) == len(set(ids)), tag
                assert not any(t is f or t is lm or t is part for t in freed), tag
                leaked = [t for t in created if t is not part and not any(t is q for q in freed)]
                assert not leaked, (tag, [x.shape for x in leaked])
            # taps: masked w_loc and the sparsity stay alive; sparsity = experts some live row routes to
            moe = make_moe("sparse", w_loc)
            created.clear(), freed.clear()
            taps = {}
            part = M.MotifMoE.local_partial(moe, f, polynorm="fp32", decode=True, lane_mask=lm, taps=taps)
            assert set(taps) == {"idx", "w", "w_loc", "sparsity"}
            assert not any(t is v for t in freed for v in taps.values())
            want_w = w_loc * mask
            assert torch.equal(taps["w_loc"].t, want_w)
            want_on = (want_w[0, :, :, 0] != 0).any(-1)
            got_on = taps["sparsity"].t.reshape(-1).float() != 0
            assert torch.equal(got_on, want_on), (Mr, nlive, got_on, want_on)
            if nlive < Mr:
                assert not bool(got_on[El - 1])  # routed only by an inactive row: skipped
    # dense path kept: prefill, and a row count the decode configs do not cover
    w_loc = routes(96, list(range(96)), 1)
    for decode, Mr in ((False, 32), (True, 96)):
        moe = make_moe("sparse", w_loc[:, :, :Mr])
        calls.clear()
        f = _TT(torch.randn(1, 1, Mr, Hs, generator=g).to(torch.bfloat16))
        M.MotifMoE.local_partial(moe, f, polynorm="bf16", decode=decode, lane_mask=_TT(torch.ones(1, 1, Mr, 1)))
        assert not any(c[0] in ("sparse_matmul", "multiply", "max") for c in calls), (decode, Mr, calls)
        assert sum(c[0] == "matmul" for c in calls) == 2


def test_moe_host_sparse_lane_mask_wiring(monkeypatch):
    """B1 wiring: ``MotifMoE.decode_lane_mask`` (slice the step's ``active`` to 32 columns -> ``ag_dp_rows`` -> the
    first column -> fp32 DRAM; gathered natural order ``L dp + j`` for L = 8 and 16; frees every intermediate once,
    never the caller's ``active``; a size-1 DP axis that hands the input back is not freed twice), the model builds it
    once per step only when a built layer (up to ``stop_after``) runs sparse experts, every decode step function
    (``decode`` / ``decode_spec`` / ``decode_wide``) passes it to every layer and frees it with ``act``, the decoder
    layer hands it to the MoE, and ``forward_decode`` passes it to ``local_partial`` without freeing it."""
    import models.demos.motif3.tt.decoder as D
    import models.demos.motif3.tt.model as MD
    import models.demos.motif3.tt.moe as M

    created, freed = [], []

    def new(t, **kw):
        x = _TT(t, **kw)
        created.append(x)
        return x

    def slc(a, start, end, memory_config=None):
        return new(a.t[tuple(slice(s, e) for s, e in zip(start, end))].clone(), dtype=a.dtype)

    monkeypatch.setattr(M.ttnn, "deallocate", lambda t, *a, **k: freed.append(t))
    monkeypatch.setattr(M.ttnn, "slice", slc)
    monkeypatch.setattr(M.ttnn, "typecast", lambda a, dt, memory_config=None: new(a.t.float(), dtype=dt))
    for L in (8, 16):
        for dp in (4, 1):
            pos = [[(r * 7 + j) % 3 - 1 for j in range(L)] for r in range(dp)]  # -1 = inactive
            acts = [torch.tensor([1.0 if p >= 0 else 0.0 for p in row]).reshape(1, 1, L, 1).expand(1, 1, L, 1024)
                    for row in pos]

            class CCL:
                def ag_dp_rows(self, a, memory_config=None):
                    if dp == 1:
                        return a  # size-1 axis: the input handed back
                    rows = [a.t] + [acts[r][..., :32].to(a.t.dtype) for r in range(1, dp)]  # a = DP row 0's
                    return new(torch.cat(rows, dim=2), dtype=a.dtype)

            act = _TT(acts[0].to(torch.bfloat16))
            created.clear(), freed.clear()
            m = M.MotifMoE.decode_lane_mask(CCL(), act, L)
            want = torch.tensor([1.0 if p >= 0 else 0.0 for row in pos for p in row]).reshape(1, 1, dp * L, 1)
            assert m.dtype == ttnn.float32 and torch.equal(m.t, want), (L, dp)
            ids = [id(t) for t in freed]
            assert len(ids) == len(set(ids)) and not any(t is act or t is m for t in freed), (L, dp)
            assert not [t for t in created if t is not m and not any(t is q for q in freed)], (L, dp)
    # model: built only when a layer (up to stop_after) is sparse
    built = []
    monkeypatch.setattr(MD.MotifMoE, "decode_lane_mask", staticmethod(lambda ccl, act, r: built.append(r) or "LM"))
    model = object.__new__(MD.MotifModel)
    model.ccl = "CCL"
    lay = lambda mode: SimpleNamespace(moe=None if mode is None else SimpleNamespace(decode_experts=mode))  # noqa: E731
    model.layers = [lay(None), lay(None), lay("dense"), lay("sparse")]
    assert model._moe_lane_mask("ACT", 8, 3) is None and built == []
    assert model._moe_lane_mask("ACT", 16, 4) == "LM" and built == [16]
    model.layers = [lay(None), lay("dense")]
    assert model._moe_lane_mask("ACT", 8, 2) is None
    # every decode step function: the mask goes to every layer (only when built) and is freed with act
    got, freed_m = [], []

    class Lay:
        def __init__(self, i, mode):
            self.i, self.moe = i, SimpleNamespace(decode_experts=mode)

        def forward_decode(self, X, **kw):
            got.append(kw)
            return f"X{self.i}"

    model.embed = SimpleNamespace(forward_decode=lambda t: "X")
    model.head = SimpleNamespace(
        vocab_split="mesh", forward_decode=lambda X, **kw: "LG", stream_mean_norm=lambda X: "hn",
        decode_logits=lambda hn, **kw: "lg", logits_rm=lambda lg, **kw: "rm", argmax_decode=lambda lg: "a")
    model.mtp = SimpleNamespace(forward_decode=lambda hn, a, **kw: "m")
    model.rope, model._rope_kinds = None, ("yarn",)
    model.cfg = SimpleNamespace(lanes_per_row=8, wide_rows_per_dp=16, max_batch=32)
    monkeypatch.setattr(MD, "_free", lambda *a: freed_m.extend(a))
    monkeypatch.setattr(MD.MotifAttention, "decode_rope_tables", staticmethod(lambda rope, idx, kinds: {"yarn": ("c",)}))
    monkeypatch.setattr(MD.MotifAttention, "active_mask_from_cur_pos", staticmethod(lambda c, lanes: "ACT"))
    caches = MD.MotifKVPool(["k0", "k1"], (0, 1), 10, 64, "bfp8", mtp="mtp-cache")
    kvw = SimpleNamespace(cur_pos="cur", page_table="pt", lanes_per_row=8, end_step=lambda: None,
                          check_flash_inputs=lambda c, p: None)
    kvw16 = SimpleNamespace(cur_pos="cur", page_table="pt", lanes_per_row=16, end_step=lambda: None)
    steps = {
        "decode": lambda: model.decode("tok", rot_idxs="ri", cur_pos="cur", page_table="pt", kv_caches=caches),
        "decode_kvw": lambda: model.decode("tok", rot_idxs="ri", cur_pos="cur", page_table="pt", kv_caches=caches,
                                           kv_write=kvw),
        "decode_spec": lambda: model.decode_spec("tok", rot_idxs="ri", kv_write=kvw, kv_caches=caches),
        "decode_wide": lambda: model.decode_wide(SimpleNamespace(shape=(4, 16)), rot_idxs="ri", kv_write=kvw16,
                                                 kv_caches=caches),
    }
    for mode in ("dense", "sparse"):
        model.layers = [Lay(0, mode), Lay(1, mode)]
        for name, step in steps.items():
            got.clear(), freed_m.clear(), built.clear()
            step()
            assert len(got) == 2, name
            if mode == "sparse":
                assert all(kw.get("moe_lane_mask") == "LM" for kw in got), (name, got)
                assert built == [16 if name == "decode_wide" else 8] and "LM" in freed_m and "ACT" in freed_m, name
            else:
                assert not any("moe_lane_mask" in kw for kw in got) and built == [], (name, got)  # release calls
    # decoder layer -> MoE
    seen = {}
    layer = object.__new__(D.MotifDecoderLayer)
    layer.is_moe = True
    layer.mhc_attn = layer.mhc_ffn = SimpleNamespace(pre=lambda X: ("R", "C"), post=lambda X, o, c: "X1")
    layer._norm_decode = lambda x, g: "N"
    layer.input_norm = layer.post_attn_norm = None
    layer.attn = SimpleNamespace(forward_decode=lambda a, **kw: "O")
    layer.shared = SimpleNamespace(forward_decode=lambda f, all_reduce: "SP")
    layer.moe = SimpleNamespace(forward_decode=lambda f, add_partial, lane_mask=None: seen.update(lm=lane_mask) or "U")
    monkeypatch.setattr(D, "_free", lambda *a: None, raising=False)
    monkeypatch.setattr(D.ttnn, "deallocate", lambda *a, **k: None)
    D.MotifDecoderLayer.forward_decode(layer, "X", rot=None, cur_pos=None, page_table=None, kv_cache=None,
                                       active="ACT", moe_lane_mask="LM")
    assert seen == {"lm": "LM"}
    # forward_decode -> local_partial, never freeing the mask
    got = {}
    moe = object.__new__(M.MotifMoE)
    moe.cfg, moe.decode_rows, moe.decode_mc, moe.decode_polynorm, moe.dram = SimpleNamespace(dp=4), (32, 64), "L1", "fp32", "dram"
    moe.ccl = SimpleNamespace(ag_dp_rows=lambda x, **k: _TT(torch.zeros(1, 1, 32, 8)), ar_dp=lambda p: p,
                              partition=lambda t, d, a: t, ar_tp=lambda p: p)
    moe.local_partial = lambda f, **kw: got.update(kw) or _TT(torch.zeros(1, 1, 32, 8))
    freed.clear()
    lm = _TT(torch.ones(1, 1, 32, 1))
    moe.forward_decode(_TT(torch.zeros(1, 1, 8, 8)), lane_mask=lm)
    assert got["lane_mask"] is lm and not any(t is lm for t in freed)


def _hf_grouped_polynorm_fp64(g, u, weight, bias, *, bias_clamp, output_scale, eps=1e-6):
    """HF ``GroupedPolyNorm.forward_single`` (modeling_motif.py) per expert in fp64: ``g, u [E, M, I]``, raw
    ``act_fn.weight [E, 3]`` / ``.bias [E, 1]`` -> ``output_scale * (sum_k sigmoid(w_k) N(g^(3-k)) + clamp(b)) * u``."""
    w = torch.sigmoid(weight.float()).double()  # HF: sigmoid(w.float()), the rest in fp64 here
    b = bias.float().double().clamp(-bias_clamp, bias_clamp)
    g, u = g.double(), u.double()

    def N(z):
        return z / torch.sqrt(z.pow(2).mean(-1, keepdim=True) + eps)

    out = []
    for e in range(g.shape[0]):
        poly = w[e, 0] * N(g[e] ** 3) + w[e, 1] * N(g[e] ** 2) + w[e, 2] * N(g[e]) + b[e]
        out.append(poly * u[e] * output_scale)
    return torch.stack(out)


def _fused_case(Mr, seed, *, El=12, Is=I):
    """Synthetic decode gate_up output (bf16-valued fp32, heavy tails like real gates), routing weights (0 for ~30 %),
    raw act_fn weight / bias with several biases outside +-0.5 (so the routed-expert clamp matters)."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(1, El, Mr, 2 * Is, generator=g)
    chi = torch.randn(1, El, Mr, 2 * Is, 4, generator=g).pow(2).sum(-1) / 4
    gu = (z / chi.sqrt() * 3.0).clamp(-60, 60).to(torch.bfloat16).float()
    w = torch.rand(1, El, Mr, 1, generator=g)
    w[w < 0.3] = 0.0
    weight = torch.randn(El, 3, generator=g)
    bias = torch.randn(El, 1, generator=g) * 0.6
    bias[0, 0], bias[1, 0] = 1.7, -2.3  # clamped to +-0.5 by the routed-expert semantics
    return gu, w, weight, bias


@torch.no_grad()
def test_moe_host_fused_polynorm_numerics():
    """B3 (docs/OPTIMIZATION_PLAN.md §3.3; ``moe_polynorm="fused"``; prototype logs/opt/phaseA/M10), host side: the
    fused kernel's algorithm (``kernels.moe_polynorm.emulate_fp32``: G = 4 rank-ordered moment partials,
    ``a = rsqrt(s D + E)``, the routing weight folded into the coefficients, Horner, one bf16 rounding), fed with the
    *production* constants (``weights.polynorm_coefficients`` with ``cfg.polynorm_bias_clamp`` ->
    ``GroupedPolyNormConsts.host_tensors``), against the HF ``GroupedPolyNorm.forward_single`` semantics in fp64:

    * fp32 parity: PCC at the bf16 output-rounding floor (PCC of the fp64 value rounded to bf16, -1e-7) and every
      element within one bf16 rounding of the fp64 value (2^-8 relative, + 1e-5 |w u| for fp32 cancellation inside
      the polynomial), at M = 32 and 64;
    * the routed-expert bias clamp: the clamped bias matches, the unclamped one does not (experts 0 / 1 have |b| > 0.5);
    * the x0.5 output scale stays in ``W_down``: ``h @ experts_down(W, 0.5 x route_scale)`` == HF ``route_scale x w x
      (0.5 PolyNorm(g) u) @ W^T`` (PCC >= 0.99999), and ``plan``'s layout / L1 budget;
    * rows with routing weight 0 are exactly 0; rows are independent (T64 rows 0..31 == the T32 call bitwise)."""
    from models.demos.motif3.tt import weights as W
    from models.demos.motif3.tt.kernels import moe_polynorm as F
    from models.demos.motif3.tt.model_config import MotifTTConfig
    from models.demos.motif3.tt.polynorm import GroupedPolyNormConsts

    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    El = cfg.experts_per_chip
    assert (cfg.polynorm_bias_clamp, cfg.polynorm_output_scale, cfg.route_scale) == (0.5, 0.5, 2.0)
    # ---- plan: the M10 layout (G = 4: 48 workers on 4 grid rows, 10 + 10 tiles each) and its CB budget ----
    p32, p64 = F.plan(El, I, 32), F.plan(El, I, 64)
    assert (p32["G"], p32["n"], p32["R"], p32["n_workers"], p32["rows"]) == (4, 10, 1, 48, 4)
    assert (p64["n"], p64["R"]) == (10, 2) and p32["l1_bytes"] < 200 * 1024 and p64["l1_bytes"] < 360 * 1024
    assert F.worker_cores(48, 12)[13] == (1, 1) and F.worker_cores(50, 12)[-1] == (1, 4)
    for bad in (dict(M=96), dict(M=16), dict(G=3), dict(G=11)):
        kw = dict(E=El, inter=I, M=32, G=4)
        kw.update(bad)
        with pytest.raises(ValueError):
            F.plan(**kw)
    with pytest.raises(ValueError, match="do not fit"):
        F.plan(El, I, 32, G=10, grid=(12, 9))
    out32 = gu32 = w32 = None
    _, _, weight, bias = _fused_case(32, seed=5)
    for Mr in (32, 64):
        gu, w, _, _ = _fused_case(Mr, seed=5 + Mr)
        if Mr == 64:
            gu[:, :, :32], w[:, :, :32] = gu32, w32  # rows 0..31 = the T32 call's
        c, b = W.polynorm_coefficients(weight, bias, bias_clamp=cfg.polynorm_bias_clamp)  # [E, 3], [E, 1]
        assert float(b.abs().max()) <= 0.5 and float(bias.abs().max()) > 0.5
        # the production constants of these 12 experts (host_tensors over all 384, chip 0 = experts 0..11)
        call = torch.cat([c, torch.full((cfg.num_experts - El, 3), 0.5)])
        ball = torch.cat([b.reshape(-1), torch.zeros(cfg.num_experts - El)])
        ht = GroupedPolyNormConsts.host_tensors(cfg, call, ball)
        D = ht["D"][0:3, :El, 0, 0].t()  # [E, 3] moment order g^2, g^4, g^6 (rows 3 dp + k of dp 0)
        Ec = ht["E"][0:3, :El, 0, 0].t()
        bb = ht["b"][0, :El, 0, 0]
        h = F.emulate_fp32(gu, w, D, Ec, bb)
        assert h.dtype == torch.bfloat16 and tuple(h.shape) == (1, El, Mr, I)
        # HF semantics in fp64 (output_scale 1 here: the 0.5 lives in W_down), routing weight on u
        ref = _hf_grouped_polynorm_fp64(gu[0, ..., :I], w[0].double() * gu[0, ..., I:].double(), weight, bias,
                                        bias_clamp=cfg.polynorm_bias_clamp, output_scale=1.0)
        assert torch.allclose(F.golden_fp64(gu, w, c, b.reshape(-1))[0], ref, rtol=1e-10, atol=1e-10)
        hd = h[0].double()
        err = (hd - ref).abs()
        tol = ref.abs() * 2.0**-8 + 1e-5 * (w[0].double() * gu[0, ..., I:].double()).abs() + 1e-30  # + fp32 cancellation
        assert bool((err <= tol).all()), (Mr, float((err - tol).max()))
        floor = pcc(ref, ref.to(torch.bfloat16).double())  # the bf16 output-rounding floor (~0.9999987)
        assert pcc(ref, hd) >= floor - 1e-7, (Mr, pcc(ref, hd), floor)
        # the clamp matters: with the raw bias experts 0 / 1 are far off
        raw = _hf_grouped_polynorm_fp64(gu[0, ..., :I], w[0].double() * gu[0, ..., I:].double(), weight, bias, bias_clamp=1e9,
                                        output_scale=1.0)
        for e in (0, 1):
            live = w[0, e, :, 0] != 0
            assert float((raw[e][live] - hd[e][live]).abs().max()) > 0.1 * float(hd[e][live].abs().max()), e
        # routing weight 0 -> exactly 0
        assert bool((h[0][w[0, :, :, 0] == 0] == 0).all())
        # x0.5 (and route_scale x2) in W_down: h @ W_down_folded == HF route_scale * w * (0.5 PolyNorm u) @ down^T
        gw = torch.Generator().manual_seed(9)
        Hs = 64
        down = torch.randn(El, Hs, I, generator=gw).to(torch.bfloat16)  # HF down_proj [E, H, I]
        wd = W.experts_down(down, cfg.polynorm_output_scale * cfg.route_scale).double()  # [E, I, H]
        y = torch.einsum("emi,eih->emh", hd, wd)
        hf = _hf_grouped_polynorm_fp64(gu[0, ..., :I], gu[0, ..., I:], weight, bias,
                                       bias_clamp=cfg.polynorm_bias_clamp, output_scale=cfg.polynorm_output_scale)
        y_ref = cfg.route_scale * w[0].double() * torch.einsum("emi,ehi->emh", hf, down.double())
        assert pcc(y_ref, y) >= 0.99999, (Mr, pcc(y_ref, y))
        if Mr == 32:
            gu32, w32, out32 = gu.clone(), w.clone(), h.clone()
        else:
            assert torch.equal(h[:, :, :32], out32), "T64 rows 0..31 != the T32 call"


@torch.no_grad()
def test_moe_host_fused_polynorm_dispatch(monkeypatch):
    """B3 host side of the MoE wiring (ttnn ops recorded, no device): ``resolve_moe_polynorm`` (config default,
    explicit modes, bad modes; "fused" needs the fp32 decode PolyNorm, an fp32 gate_up and the "fold" combine: explicit
    raises, the config default falls back to "composite"); ``experts()`` and ``sparse_experts()`` run the fused kernel
    instead of the composite at decode M = 32 / 64 with routing weights, the composite for prefill, for other row
    counts, without routing weights, when the kernel refuses the tensors, and when the module has no kernel (the
    default); ``h`` is freed once, ``gu`` / ``row_scale`` never by the kernel call."""
    import models.demos.motif3.tt.moe as M
    from models.demos.motif3.tt.model_config import MOE_POLYNORM_MODES, MotifTTConfig

    assert MOE_POLYNORM_MODES == ("composite", "fused")
    monkeypatch.delenv("MOTIF3_MOE_POLYNORM", raising=False)
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    assert cfg.moe_polynorm == "fused"  # the default since the Phase B eval
    R = M.resolve_moe_polynorm
    ok = dict(decode_polynorm="fp32", combine_mode="fold")
    assert R(None, cfg, **ok) == "fused" and R("fused", cfg, **ok) == "fused" and R("composite", cfg, **ok) == "composite"
    fz = SimpleNamespace(moe_polynorm="fused")
    assert R(None, fz, **ok) == "fused" and R(None, SimpleNamespace(), **ok) == "composite"
    assert R(None, fz, **ok, gate_up_dtype=ttnn.float32) == "fused"
    for bad in (dict(decode_polynorm="bf16"), dict(combine_mode="multiply_sum"), dict(gate_up_dtype=ttnn.bfloat16)):
        kw = dict(ok)
        kw.update(bad)
        assert R(None, fz, **kw) == "composite", bad
        with pytest.raises(ValueError, match="fused"):
            R("fused", cfg, **kw)
    with pytest.raises(ValueError, match="moe_polynorm"):
        R("kernel", cfg, **ok)

    calls, freed = [], []

    class FakeFused:
        def __init__(self, ok=True):
            self.ok = ok

        def supports(self, gu, w):
            return self.ok

        def __call__(self, gu, w, *, memory_config=None):
            calls.append(("fused", int(gu.shape[-2]), memory_config))
            return SimpleNamespace(shape=[1, 12, int(gu.shape[-2]), I], tag="h_fused")

    monkeypatch.setattr(M.ttnn, "repeat", lambda f, reps, **kw: SimpleNamespace(shape=[1, 12] + list(f.shape[2:])))
    monkeypatch.setattr(M.ttnn, "matmul", lambda a, b, **kw: SimpleNamespace(shape=[1, 12, a.shape[2], 2 * I]))
    monkeypatch.setattr(M.ttnn, "sparse_matmul",
                        lambda a, b, **kw: SimpleNamespace(shape=[1, 12, a.shape[-2], 2 * I]))
    monkeypatch.setattr(M, "_reshape", lambda t, shape: SimpleNamespace(shape=list(shape)))
    monkeypatch.setattr(M.ttnn, "deallocate", lambda t, *a, **k: freed.append(t))
    moe = object.__new__(M.MotifMoE)
    moe.e_loc, moe.inter, moe.hidden, moe.dram = 12, I, H, "dram"
    moe.gate_up_dtype, moe.down_dtype, moe.ckc_experts = None, ttnn.bfloat16, "ckc"
    moe.w_gate_up, moe.w_down, moe.prefill_pc = "W_gate_up", "W_down", False
    moe.polynorm_impl = moe.prefill_polynorm_impl = "horner"
    moe.pc_gate_up = moe.pc_down = "pc"
    moe.pc_wide = {64: ("pc64", "pc64")}
    moe.decode_rows = (32, 64)
    moe.polynorm = lambda gu, **kw: calls.append(("composite", int(gu.shape[-2]))) or SimpleNamespace(
        shape=[1, 12, int(gu.shape[-2]), I], tag="h_comp")
    rs = SimpleNamespace(shape=[1, 12, 32, 1], tag="w")
    f = lambda m: SimpleNamespace(shape=[1, 1, m, H])  # noqa: E731
    cases = [  # (pn_fused, M, decode, row_scale, want)
        (None, 32, True, rs, "composite"),
        (FakeFused(), 32, True, rs, "fused"),
        (FakeFused(), 64, True, rs, "fused"),
        (FakeFused(), 32, False, rs, "composite"),  # prefill
        (FakeFused(), 96, True, rs, "composite"),  # not a decode row count
        (FakeFused(), 32, True, None, "composite"),  # multiply_sum (no routing weights folded)
        (FakeFused(ok=False), 32, True, rs, "composite"),  # contract refused
    ]
    for fused, m, decode, row_scale, want in cases:
        moe.pn_fused = fused
        for path in ("dense", "sparse"):
            if path == "sparse" and (not decode or m not in (32, 64)):
                continue
            calls.clear(), freed.clear()
            if path == "dense":
                M.MotifMoE.experts(moe, f(m), polynorm="fp32", decode=decode, row_scale=row_scale, memory_config="L1")
            else:
                M.MotifMoE.sparse_experts(moe, f(m), "sp", polynorm="fp32", row_scale=row_scale, memory_config="L1")
            kinds = [c[0] for c in calls]
            assert kinds == [want], (fused, m, decode, path, calls)
            if want == "fused":
                assert calls[0][2] == "L1"
            hs = [t for t in freed if getattr(t, "tag", "").startswith("h_")]
            assert len(hs) == 1 and hs[0].tag == ("h_fused" if want == "fused" else "h_comp"), (path, freed)
            assert row_scale is None or not any(t is row_scale for t in freed)
    del moe.pn_fused  # a module built before B3 (no attribute): the composite
    calls.clear()
    M.MotifMoE.experts(moe, f(32), polynorm="fp32", decode=True, row_scale=rs, memory_config="L1")
    assert [c[0] for c in calls] == ["composite"]


def test_moe_host_compact_meta():
    """B2a host row lists (:func:`moe.compact_prefill_meta`), random skewed routes on a (4, 8) layout (32 chips x 12
    local experts, permuted ids), block sizes 32 / 64 / 128:

    * every (token, local expert) assignment appears exactly once on its chip, with its weight bit-copied; rows of an
      expert start at a block boundary, tokens ascending, experts in local order (the dense expert-sum order);
    * pad rows have token 0 / tokv -1 / weight 0; blocks = the one-hot sparsity with exactly ``nb`` nonzeros per chip;
      unused trailing blocks repeat the last expert; ``cblk`` = the expert constants of each block;
    * the emulated compact partial (row values ``f_e(x_t) w``, combined through ``tokv``) == the dense sum over the
      local experts, exactly (integer data);
    * ``need`` == :func:`compact_need_blocks`; fewer blocks than needed and unknown expert ids raise."""
    import models.demos.motif3.tt.moe as M

    g = torch.Generator().manual_seed(5)
    P, El, Kk, S = 32, 12, 8, 512
    perm = torch.randperm(P * El, generator=g)
    local = perm.reshape(P, El)
    # skewed routes: a few hot experts (on chip 3) take a large share; top-8 distinct per token
    pr = torch.ones(P * El)
    pr[local[3, :4]] = 60.0
    idx = torch.stack([torch.multinomial(pr, Kk, replacement=False, generator=g) for _ in range(S)])
    w = torch.rand(S, Kk, generator=g)
    pn = torch.randint(-3, 4, (P, El, 4), generator=g).float()
    val = torch.randint(-4, 5, (P * El, S), generator=g).double()  # f_e(x_t) stand-in per global expert
    wint = torch.randint(1, 4, (S, Kk), generator=g).double()
    for mb in (32, 64, 128):
        need = M.compact_need_blocks(idx, local, mb)
        nb = need + 3
        m = M.compact_prefill_meta(idx, w, local, mb, nb, pn=pn)
        assert m["need"] == need
        R = nb * mb
        tok, tokv, wv, eblk, sp = m["tok"], m["tokv"], m["w"], m["eblk"], m["sparsity"]
        assert tok.shape == (P, R) and tok.dtype == torch.int32 and sp.shape == (P, nb, El)
        assert torch.equal(sp.sum(-1), torch.ones(P, nb)) and int(sp.sum()) == P * nb
        assert torch.equal(sp.argmax(-1), eblk)
        assert torch.equal(m["cblk"], torch.gather(pn, 1, eblk.unsqueeze(-1).expand(P, nb, 4)))
        pad = tokv < 0
        assert bool((tok[pad] == 0).all()) and bool((wv[pad] == 0).all())
        assert torch.equal(tok[~pad].float(), tokv[~pad])
        g2s = {int(local.reshape(-1)[i]): i for i in range(P * El)}
        for p in (0, 3, 17, 31):
            erow = eblk[p].repeat_interleave(mb)  # expert of every row
            seen = 0
            for e in range(El):
                gid = int(local[p, e])
                tk = torch.nonzero((idx == gid).any(-1)).flatten()
                rows = torch.nonzero((erow == e) & ~pad[p]).flatten()
                assert torch.equal(tokv[p, rows].long(), tk), (mb, p, e)  # tokens ascending
                if len(rows):
                    assert int(rows[0]) % mb == 0 and torch.equal(rows, rows[0] + torch.arange(len(rows)))
                    kpos = (idx[tk] == gid).float().argmax(-1)
                    assert torch.equal(wv[p, rows], w[tk, kpos])  # bit copies
                seen += len(rows)
            assert seen == int((~pad[p]).sum())
            # trailing unused blocks repeat the last expert and hold only pad rows
            used = int(((~pad[p]).reshape(nb, mb).any(-1)).nonzero().max()) + 1
            assert bool((eblk[p, used:] == El - 1).all()) or used == nb
            # emulated combine == dense sum over the chip's experts in local order, exactly
            ws = torch.zeros(El, S, dtype=torch.float64)
            for e in range(El):
                gid = int(local[p, e])
                hit = idx == gid
                ws[e] = (wint * hit).sum(-1)
            dense = sum(ws[e] * val[g2s[int(local[p, e])]] for e in range(El))
            rv = torch.zeros(R, dtype=torch.float64)
            for j in torch.nonzero(~pad[p]).flatten().tolist():
                t, e = int(tok[p, j]), int(erow[j])
                kpos = int((idx[t] == int(local[p, e])).float().argmax())
                rv[j] = wint[t, kpos] * val[g2s[int(local[p, e])], t]
            PT = (tokv[p].double().unsqueeze(0) == torch.arange(S, dtype=torch.float64).unsqueeze(1)).double()
            assert torch.equal(PT @ rv, dense), (mb, p)
        with pytest.raises(ValueError, match="busiest chip"):
            M.compact_prefill_meta(idx, w, local, mb, need - 1)
        # the serving fast path (numpy, one pass) == the reference lists packed by compact_upload_rows, word for word
        g2s = M._global_to_slot(local).numpy()
        pn_bits = (pn.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF).numpy()
        ladder = tuple(sorted({need, need + 3, need + 9}))
        n2, nb2, u = M.compact_upload_fast(idx.numpy(), g2s, P, El, mb, ladder, pn_bits, S)
        assert n2 == need and nb2 == need and u.dtype == np.int32 and u.shape == (P, 4, need * mb)
        m2 = M.compact_prefill_meta(idx, w, local, mb, need, pn=pn)
        ref = M.MotifMoE.compact_upload_rows(m2, S, mb, need)
        assert torch.equal(torch.from_numpy(u).long(), ref) and int(ref.min()) >= 0 and int(ref.max()) < 2 ** 31
        # row 2: the routing-weight gather index e S + t into the chip's w_loc (transposed, flattened) -> m["w"]
        for p in (0, 3, 31):
            wl = torch.zeros(El, S)
            for e in range(El):
                hit = idx == int(local[p, e])
                wl[e] = (w * hit).sum(-1)
            v = m2["tokv"][p] >= 0
            assert torch.equal(wl.reshape(-1)[ref[p, 2]][v], m2["w"][p][v])
            assert torch.equal(ref[p, 1][v], m2["tokv"][p][v].long()) and bool((ref[p, 1][~v] == S).all())
        assert M.compact_upload_fast(idx.numpy(), g2s, P, El, mb, (need - 1,), pn_bits, S) == (need, None, None)
    bad = idx.clone()
    bad[0, 0] = P * El + 5
    with pytest.raises(ValueError, match="local expert"):
        M.compact_prefill_meta(bad, w, local, 32, 64)
    with pytest.raises(ValueError, match="distinct"):
        M.compact_need_blocks(idx, torch.zeros(P, El, dtype=torch.long), 32)
    with pytest.raises(ValueError, match="local expert"):
        M.compact_upload_fast(bad.numpy(), M._global_to_slot(local).numpy(), P, El, 32, (64,),
                              np.zeros((P, El, 4), np.int32), S)


def test_moe_host_compact_ladder_and_resolve(monkeypatch):
    """B2a: ``compact_block`` ("auto": 32 up to 2048 rows, 64 above; fixed 32 / 64 / 128; others raise),
    ``compact_ladder`` (floor ``max(e_loc, rows / 4)`` rows, ratio <= 1.25 (+1 at least), last entry = the 2 x rows cap),
    ``compact_bucket`` (smallest entry >= need, None beyond the cap), ``resolve_prefill_moe`` (default dense; compact
    needs fold / bf16 / rms: explicit raises, the config default falls back) and the config knobs."""
    import models.demos.motif3.tt.moe as M
    from models.demos.motif3.tt.model_config import PREFILL_MOE_BLOCKS, PREFILL_MOE_MODES, MotifTTConfig

    assert PREFILL_MOE_MODES == ("dense", "compact") and PREFILL_MOE_BLOCKS == ("auto", "32", "64", "128")
    assert [M.compact_block(r, "auto") for r in (128, 1024, 2048, 4096)] == [32, 32, 32, 64]
    assert [M.compact_block(4096, b) for b in ("32", " 64", 128)] == [32, 64, 128]
    with pytest.raises(ValueError, match="block"):
        M.compact_block(1024, "48")
    for rows, mb in ((1024, 32), (2048, 32), (4096, 64), (4096, 32), (512, 32), (8192, 128)):
        lad = M.compact_ladder(rows, mb, 12)
        assert lad[0] == max(12, -(-rows // (4 * mb))) and lad[-1] == -(-2 * rows // mb)
        assert all(b > a for a, b in zip(lad, lad[1:]))
        assert all(b <= max(a + 1, -(-a * 5 // 4)) for a, b in zip(lad, lad[1:]))
        assert M.compact_bucket(1, lad) == lad[0] and M.compact_bucket(lad[-1], lad) == lad[-1]
        assert M.compact_bucket(lad[-1] + 1, lad) is None
        for need in range(1, lad[-1] + 1):
            b = M.compact_bucket(need, lad)
            assert b >= need and all(x < need for x in lad if x < b)
    assert M.compact_ladder(1024, 32, 12) == (12, 15, 19, 24, 30, 38, 48, 60, 64)
    monkeypatch.delenv("MOTIF3_PREFILL_MOE", raising=False)
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    assert cfg.prefill_moe == "compact" and cfg.prefill_moe_block == "auto" and cfg.prefill_moe_min_rows == 1024
    Rz = M.resolve_prefill_moe
    kw = dict(combine_mode="fold", prefill_polynorm="bf16", prefill_polynorm_impl="rms")
    assert Rz(None, cfg, **kw) == "compact" and Rz("dense", cfg, **kw) == "dense"
    assert Rz(None, SimpleNamespace(), **kw) == "dense"  # a config without the field: the release
    on = SimpleNamespace(prefill_moe="compact")
    assert Rz(None, on, **kw) == "compact"
    for bad in (dict(combine_mode="multiply_sum"), dict(prefill_polynorm="fp32"), dict(prefill_polynorm_impl="horner")):
        k2 = dict(kw, **bad)
        assert Rz(None, on, **k2) == "dense"
        with pytest.raises(ValueError, match="needs"):
            Rz("compact", cfg, **k2)
    with pytest.raises(ValueError, match="prefill_moe"):
        Rz("sparse", cfg, **kw)


class _MT:
    """Per-chip stand-in tensor of the B2a emulation (``chips``: one torch tensor per chip, row-major mesh order)."""

    def __init__(self, chips, dtype=None, layout=None):
        self.chips = [c for c in chips]
        self.dtype, self.layout = dtype, layout

    @property
    def shape(self):
        return list(self.chips[0].shape)

    @property
    def spec(self):
        return ("spec", tuple(self.shape), self.dtype)

    def map(self, fn, **kw):
        return _MT([fn(c) for c in self.chips], dtype=kw.get("dtype", self.dtype), layout=kw.get("layout", self.layout))


def test_moe_host_prefill_compact_emulated(monkeypatch):
    """B2a ``local_partial(decode=False)`` with the ttnn ops emulated per chip on a (1, 2) mesh (4 local experts per chip,
    top-2, small dims, integer data so every sum is exact):

    * compacted partial == the dense math (``sum_e w_loc[e] f_e(x)`` in local expert order) on every chip, exactly;
    * three per-chip uploads, the embedding gather of the chunk's rows, two ``sparse_matmul`` with ``nnz = nb``, ``b``
      sparse only, the decode experts' program configs for ``mb`` rows and a compact preallocated output; the routes are
      read from chip 0 only; every intermediate freed once, the caller's ``f`` never;
    * the busiest chip beyond the ladder cap -> the dense path on the same routes (``dense_cap``), no upload;
    * ``frozen`` with the shape unwarmed -> the dense path (``dense_unwarmed``); ``warm_compact`` compiles the dense path
      once and every ladder entry (pad rows only: a zero partial), after which the chunk runs compacted again;
    * chunks below ``prefill_moe_min_rows``, decode calls and the dense mode never take the compacted path."""
    import models.demos.motif3.tt.moe as M

    Hs, Is, El, Kk, P, Mr = 64, 16, 4, 2, 2, 128
    g = torch.Generator().manual_seed(9)
    local = torch.tensor([[5, 0, 7, 2], [1, 6, 3, 4]])  # chip p's local experts (global ids)
    Wgu = torch.randint(-1, 2, (P, El, Hs, 2 * Is), generator=g).double()
    Wd = torch.randint(-1, 2, (P, El, Is, Hs), generator=g).double()
    pnh = torch.randint(-2, 3, (P, El, 4), generator=g).double()  # c0, c1, c2, b (the emulated PolyNorm uses c0, b)
    x = torch.randint(-2, 3, (Mr, Hs), generator=g).double()
    created, freed, calls = [], [], []

    def new(chips, **kw):
        t = _MT(chips, **kw)
        created.append(t)
        return t

    SHARD, REP = object(), object()

    def from_torch(h, *, dtype, layout, device, memory_config, mesh_mapper):
        calls.append(("upload", dtype, layout, tuple(h.shape)))
        h = h.double() if dtype != ttnn.uint32 else h.long()
        if dtype == ttnn.uint32:
            assert bool(((h >= 0) & (h < 2 ** 31)).all()), "uint32 upload words must be below 2^31"
        if mesh_mapper is REP:
            return new([h.clone() for _ in range(P)], dtype=dtype, layout=layout)
        assert mesh_mapper is SHARD and h.shape[0] == 1
        a = h.shape[1] // P
        return new([h[:, p * a:(p + 1) * a].clone() for p in range(P)], dtype=dtype, layout=layout)

    def to_torch(t):
        assert len(t.chips) == 1, "routes are taken from one chip"
        return t.chips[0].clone()

    def copy_d2h(dev, host, blocking=True):
        calls.append(("read", blocking, dev.dtype))
        host.chips = [c.clone() for c in dev.chips]

    def gather(src, dim, index, memory_config=None):
        calls.append(("gather",))
        assert dim == -1 and index.dtype == ttnn.uint32
        return new([torch.gather(a, -1, b.long()) for a, b in zip(src.chips, index.chips)], dtype=src.dtype)

    def bitcast(t, dtype, memory_config=None):
        calls.append(("bitcast", t.dtype, dtype))
        # only the 16-bit bitcast is exact on device (a uint32 -> fp32 bitcast truncates the mantissa)
        assert t.dtype == ttnn.uint16 and dtype == ttnn.bfloat16
        return new([c.long().to(torch.int16).view(torch.bfloat16).double() for c in t.chips], dtype=dtype,
                   layout=t.layout)

    def typecast(t, dtype, memory_config=None):
        calls.append(("typecast", t.dtype, dtype))
        assert t.dtype == ttnn.uint32
        if dtype == ttnn.float32:  # integers: exact below 2^24
            assert all(bool(((c >= 0) & (c < 2 ** 24)).all()) for c in t.chips)
            return new([c.double() for c in t.chips], dtype=dtype, layout=t.layout)
        assert dtype == ttnn.uint16
        assert all(bool(((c >= 0) & (c < 2 ** 16)).all()) for c in t.chips), "typecast must be exact"
        return new([((c.long() + 2 ** 15) % 2 ** 16 - 2 ** 15) for c in t.chips], dtype=dtype, layout=t.layout)

    def emb(i, tab, *, layout, memory_config):
        calls.append(("embedding",))
        return new([tab.chips[p].reshape(-1, Hs)[i.chips[p].reshape(-1).long()].unsqueeze(0) for p in range(P)])

    def slc(t, a, b, memory_config=None):
        return new([c[tuple(slice(int(s), int(e)) for s, e in zip(a, b))].clone() for c in t.chips], dtype=t.dtype,
                   layout=t.layout)

    def sparse_matmul(a, b, *, sparsity, nnz, is_input_a_sparse, is_input_b_sparse, program_config,
                      compute_kernel_config, dtype, memory_config, optional_output_tensor):
        calls.append(("sparse_matmul", nnz, is_input_a_sparse, is_input_b_sparse, program_config, dtype,
                      optional_output_tensor))
        outs = []
        for p in range(P):
            sp = sparsity.chips[p]
            nb = sp.shape[1]
            assert sp.shape == (1, nb, 1, El) and int((sp != 0).sum()) == nnz == nb
            e_of = sp.reshape(nb, El).argmax(-1)
            outs.append(torch.stack([a.chips[p][0, k] @ b.chips[p][0, int(e_of[k])] for k in range(nb)]).unsqueeze(0))
        return new(outs, dtype=dtype)

    def bcast(op):
        return lambda a, b, **kw: new([op(x_, y_) for x_, y_ in zip(a.chips, b.chips)], dtype=kw.get("dtype"))

    def polynorm(g_, consts, *, inter, mode, eps, compute_kernel_config, memory_config, up, impl,
                 intermediate_memory_config):
        calls.append(("polynorm", mode, impl, isinstance(consts, dict)))
        assert isinstance(consts, dict) and mode == "bf16" and impl == "rms"
        return new([(consts["c0"].chips[p] * g_.chips[p] + consts["b"].chips[p]) * up.chips[p] for p in range(P)])

    def dtt(t, mesh):
        return torch.stack([c for c in t.chips]).reshape(1, P, *t.shape)

    ns = M.ttnn
    for name, fn in (("from_torch", from_torch), ("to_torch", to_torch), ("embedding", emb), ("slice", slc),
                     ("sparse_matmul", sparse_matmul), ("multiply", bcast(torch.mul)),
                     ("eq", bcast(lambda a, b: (a == b).double())), ("matmul", bcast(torch.matmul))):
        monkeypatch.setattr(ns, name, fn)
    monkeypatch.setattr(ns, "get_device_tensors", lambda t: [_MT([c]) for c in t.chips])
    monkeypatch.setattr(ns, "copy_device_to_host_tensor", copy_d2h)
    monkeypatch.setattr(ns, "gather", gather)
    monkeypatch.setattr(ns, "allocate_tensor_on_host", lambda spec, mesh: _MT([]))
    monkeypatch.setattr(ns, "bitcast", bitcast)
    monkeypatch.setattr(ns, "typecast", typecast)
    monkeypatch.setattr(ns, "to_layout", lambda t, layout, memory_config=None: new([c.clone() for c in t.chips],
                                                                                    dtype=t.dtype, layout=layout))
    monkeypatch.setattr(ns, "transpose", lambda t, a, b, memory_config=None: new([c.transpose(a, b).clone()
                                                                                 for c in t.chips], dtype=t.dtype))
    monkeypatch.setattr(ns, "allocate_tensor_on_device", lambda shape, dtype, layout, mesh, mc: ("OUT", tuple(shape)))
    monkeypatch.setattr(ns, "create_mesh_mapper", lambda mesh, conf: SHARD)
    monkeypatch.setattr(ns, "ReplicateTensorToMesh", lambda mesh: REP)
    monkeypatch.setattr(ns, "MeshMapperConfig", lambda *a, **k: None)
    monkeypatch.setattr(ns, "PlacementShard", lambda d: d)
    monkeypatch.setattr(ns, "MeshShape", lambda *a: a)
    monkeypatch.setattr(ns, "Shape", lambda v: list(v))
    monkeypatch.setattr(ns, "deallocate", lambda t, *a, **k: freed.append(t))
    def reshape(t, shape):  # a copy (stand-ins have no buffer): moe._reshape frees the input
        freed.append(t)
        return new([c.reshape(tuple(shape)).clone() for c in t.chips], dtype=t.dtype, layout=t.layout)

    monkeypatch.setattr(M, "_reshape", reshape)
    monkeypatch.setattr(M._pn, "grouped_polynorm", polynorm)
    monkeypatch.setattr(M, "device_tensors_to_torch", dtt)

    moe = object.__new__(M.MotifMoE)
    cfg = SimpleNamespace(experts_gate_up_pc=lambda m_tiles: ("pc_gu", m_tiles),
                          experts_down_pc=lambda m_tiles: ("pc_dn", m_tiles), polynorm_eps=1e-6)
    moe.__dict__.update(
        mesh_device=SimpleNamespace(shape=(1, P)), cfg=cfg, hidden=Hs, inter=Is, e_loc=El, top_k=Kk, dram="DRAM",
        prefill_moe="compact", prefill_moe_block="auto", prefill_moe_min_rows=64, prefill_polynorm="bf16",
        internal_route_scale=1.0, gate_up_dtype=None, down_dtype=ttnn.bfloat16, combine_dtype=ttnn.bfloat16,
        combine_mode="fold", decode_experts="dense", decode_rows=(32,),
        ckc_experts="ckc_e", ckc_polynorm="ckc_p", _pn_host=None, scatter_consts={},
        local_ids=_MT([local[p].double().reshape(1, El, 1, 1) for p in range(P)]),
        pn_consts=SimpleNamespace(c={"bf16": {k: _MT([pnh[p, :, i].reshape(1, El, 1, 1) for p in range(P)])
                                              for i, k in enumerate(("c0", "c1", "c2", "b"))}}),
        w_gate_up=_MT([Wgu[p].unsqueeze(0) for p in range(P)]), w_down=_MT([Wd[p].unsqueeze(0) for p in range(P)]),
    )
    moe.compact_state = M.CompactPrefillState(owner=moe)
    st = moe.compact_state
    routes = {}

    def router(f, *, scale=None, memory_config=None, taps=None):
        i_, w_ = routes["cur"]
        return (new([i_.double().reshape(1, 1, -1, Kk)] * P, dtype=ttnn.uint32),
                new([w_.reshape(1, 1, -1, Kk)] * P, dtype=ttnn.float32))

    moe.router = router
    dense_calls = []
    def local_weights(i_, w_, memory_config=None):  # per chip: w_loc[e, t] = sum_k w[t, k] (idx[t, k] == id_e)
        dense_calls.append("local_weights")
        return new([((i_.chips[p].reshape(-1, 1, Kk) == local[p].double().reshape(1, El, 1)).double()
                     * w_.chips[p].reshape(-1, 1, Kk)).sum(-1).t().reshape(1, El, -1, 1) for p in range(P)],
                   dtype=ttnn.float32)

    moe.local_weights = local_weights
    moe.experts = lambda f, **kw: dense_calls.append("experts") or "Y"
    moe.reduce_experts = lambda y, memory_config=None: dense_calls.append("reduce") or "PART"
    monkeypatch.setattr(M, "_free", lambda *ts: freed.extend(t for t in ts if t is not None))

    def dense_ref(i_, w_):
        out = []
        for p in range(P):
            acc = torch.zeros(Mr, Hs, dtype=torch.float64)
            for e in range(El):
                wl = ((i_ == int(local[p, e])).double() * w_).sum(-1, keepdim=True)  # [Mr, 1]
                gu = x @ Wgu[p, e]
                h = (pnh[p, e, 0] * gu[:, :Is] + pnh[p, e, 3]) * (wl * gu[:, Is:])
                acc = acc + h @ Wd[p, e]
            out.append(acc)
        return out

    f = _MT([x.reshape(1, 1, Mr, Hs)] * P)
    i1 = torch.stack([torch.randperm(2 * El, generator=g)[:Kk] for _ in range(Mr)])
    w1 = torch.randint(1, 4, (Mr, Kk), generator=g).double()
    routes["cur"] = (i1, w1)
    n_created = len(created)
    part = moe.local_partial(f, polynorm="bf16", decode=False)
    assert dense_calls == ["local_weights"] and st.stats["compact"] == 1
    for p, want in enumerate(dense_ref(i1, w1)):
        assert torch.equal(part.chips[p].reshape(Mr, Hs), want), p
    sm = [c for c in calls if c[0] == "sparse_matmul"]
    mb = M.compact_block(Mr, "auto")
    nb = M.compact_bucket(M.compact_need_blocks(i1, local, mb), M.compact_ladder(Mr, mb, El))
    assert [(c[1], c[2], c[3], c[4]) for c in sm] == [(nb, False, True, ("pc_gu", mb // 32)),
                                                       (nb, False, True, ("pc_dn", mb // 32))]
    assert all(c[6][0] == "OUT" and c[6][1][1] == nb for c in sm)
    ups = [c for c in calls if c[0] == "upload"]
    assert [(u[1], u[2]) for u in ups] == [(ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), (ttnn.float32, ttnn.TILE_LAYOUT)]
    assert ups[0][3] == (1, P, 4, nb * mb)  # one per-chip upload: chip p holds [1, 1, 4, rows]
    assert [c[1:] for c in calls if c[0] == "read"] == [(True, ttnn.uint32)]  # one read: idx (w stays on device)
    assert sum(1 for c in calls if c[0] == "embedding") == 1 and sum(1 for c in calls if c[0] == "gather") == 1
    assert [c for c in calls if c[0] == "bitcast"] == [("bitcast", ttnn.uint16, ttnn.bfloat16)]
    ids_freed = [id(t) for t in freed]
    assert len(ids_freed) == len(set(ids_freed)), "a tensor was freed twice"
    assert id(f) not in ids_freed and id(part) not in ids_freed
    live = [t for t in created[n_created:] if id(t) not in set(ids_freed) and t is not part]
    assert [t.shape for t in live] == [[1, 1, Mr, 1]] and st.iota[Mr] is live[0], "only the shared iota column lives"

    # beyond the cap: every route on chip 0's experts, one hot expert -> dense on the same routes
    i2 = torch.stack([torch.tensor([5, 0] if t < 66 else [7, 2]) for t in range(Mr)])  # 3 + 3 + 2 + 2 > 8 blocks
    routes["cur"] = (i2, w1)
    calls.clear()
    dense_calls.clear()
    assert moe.local_partial(f, polynorm="bf16", decode=False) == "PART"
    assert dense_calls == ["local_weights", "experts", "reduce"] and st.stats["dense_cap"] == 1
    assert not [c for c in calls if c[0] == "upload"]
    # frozen + unwarmed -> dense; warm_compact -> compacted again
    dense_calls.clear()
    routes["cur"] = (i1, w1)
    st.frozen = lambda: True
    assert moe.local_partial(f, polynorm="bf16", decode=False) == "PART" and st.stats["dense_unwarmed"] == 1
    dense_calls.clear()
    routes["cur"] = (torch.zeros(Mr, Kk, dtype=torch.long), torch.zeros(Mr, Kk))  # warm-up routes (ignored)
    lad = moe.warm_compact(Mr)
    assert lad == M.compact_ladder(Mr, mb, El) and dense_calls == ["local_weights", "experts", "reduce"] + [
        "local_weights"] * len(lad)
    assert st.warmed == {(Mr, mb, b) for b in lad}
    routes["cur"] = (i1, w1)
    part2 = moe.local_partial(f, polynorm="bf16", decode=False)
    assert st.stats["compact"] == 2  # warm_compact runs _compact_device directly (no counter)
    assert all(torch.equal(a, b) for a, b in zip(part.chips, part2.chips))
    st.frozen = lambda: False
    # below min rows / decode / dense mode: never compacted
    dense_calls.clear()
    moe.prefill_moe_min_rows = 256
    assert moe.local_partial(f, polynorm="bf16", decode=False) == "PART"
    moe.prefill_moe_min_rows = 64
    moe.prefill_moe = "dense"
    assert moe.local_partial(f, polynorm="bf16", decode=False) == "PART"
    moe.prefill_moe = "compact"
    assert not moe.compact_applies(32) and moe.compact_applies(64)
    assert st.stats["compact"] == 2


class _MTA(_MT):
    """:class:`_MT` with ``is_allocated`` (``CompactRows.free``)."""

    def is_allocated(self):
        return True


@pytest.mark.parametrize("combo", [("host", "gather"), ("device", "matmul"), ("device", "gather")],
                         ids=lambda c: "/".join(c))
def test_moe_host_prefill_compact_kernels_emulated(monkeypatch, combo):
    """B2b ``local_partial(decode=False)`` wiring with the ttnn ops, the dispatch kernel and the gather combine emulated
    per chip on a (1, 2) mesh (the B2a emulation's setup: 4 local experts per chip, top-2, integer data):

    * compacted partial == the dense math on every chip, exactly, for each (dispatch, combine) combination;
    * "device": the routes are never read or uploaded; the dispatch kernel runs once per chunk on the router's ``(idx, w)``
      (``w_is_loc=False``) with the layer's meta, and the host reads only its ``need`` tensor; the experts run on the
      first NB blocks (``sparse_matmul`` with ``nnz = NB``); the capacity buffers are freed;
    * "gather": one combine call on the untilized ``y`` with the keys page 1 (the upload's / the dispatch rows'); no
      one-hot ``eq`` / ``matmul``;
    * beyond the cap (NB = 0) -> the dense path on the same routes; frozen + unwarmed -> dense; ``warm_compact``
      compiles the device path at every ladder entry, after which the chunk runs compacted again;
    * every intermediate freed once, the caller's ``f`` never."""
    import models.demos.motif3.tt.moe as M
    from models.demos.motif3.tt.kernels.moe_compact import CompactRows

    disp_mode, comb_mode = combo
    Hs, Is, El, Kk, P, Mr = 64, 16, 4, 2, 2, 128
    g = torch.Generator().manual_seed(9)
    local = torch.tensor([[5, 0, 7, 2], [1, 6, 3, 4]])
    Wgu = torch.randint(-1, 2, (P, El, Hs, 2 * Is), generator=g).double()
    Wd = torch.randint(-1, 2, (P, El, Is, Hs), generator=g).double()
    pnh = torch.randint(-2, 3, (P, El, 4), generator=g).double()
    x = torch.randint(-2, 3, (Mr, Hs), generator=g).double()
    created, freed, calls = [], [], []

    def new(chips, **kw):
        t = _MTA(chips, **kw)
        created.append(t)
        return t

    SHARD, REP = object(), object()

    def from_torch(h, *, dtype, layout, device, memory_config, mesh_mapper):
        calls.append(("upload", dtype, layout, tuple(h.shape)))
        h = h.double() if dtype != ttnn.uint32 else h.long()
        if mesh_mapper is REP:
            return new([h.clone() for _ in range(P)], dtype=dtype, layout=layout)
        assert mesh_mapper is SHARD and h.shape[0] == 1
        a = h.shape[1] // P
        return new([h[:, p * a:(p + 1) * a].clone() for p in range(P)], dtype=dtype, layout=layout)

    def to_torch(t):
        assert len(t.chips) == 1, "one chip's value"
        return t.chips[0].clone()

    def copy_d2h(dev, host, blocking=True):
        calls.append(("read", blocking, dev.dtype, tuple(dev.shape)))
        host.chips = [c.clone() for c in dev.chips]

    def gather(src, dim, index, memory_config=None):
        calls.append(("gather",))
        return new([torch.gather(a, -1, b.long()) for a, b in zip(src.chips, index.chips)], dtype=src.dtype)

    def bitcast(t, dtype, memory_config=None):
        assert t.dtype == ttnn.uint16 and dtype == ttnn.bfloat16
        return new([c.long().to(torch.int16).view(torch.bfloat16).double() for c in t.chips], dtype=dtype,
                   layout=t.layout)

    def typecast(t, dtype, memory_config=None):
        calls.append(("typecast", t.dtype, dtype))
        assert t.dtype == ttnn.uint32
        if dtype == ttnn.float32:
            return new([c.double() for c in t.chips], dtype=dtype, layout=t.layout)
        return new([((c.long() + 2 ** 15) % 2 ** 16 - 2 ** 15) for c in t.chips], dtype=dtype, layout=t.layout)

    def emb(i, tab, *, layout, memory_config):
        calls.append(("embedding", i.dtype, tuple(i.shape)))
        return new([tab.chips[p].reshape(-1, Hs)[i.chips[p].reshape(-1).long()].unsqueeze(0) for p in range(P)])

    def slc(t, a, b, memory_config=None):
        return new([c[tuple(slice(int(s_), int(e_)) for s_, e_ in zip(a, b))].clone() for c in t.chips], dtype=t.dtype,
                   layout=t.layout)

    def sparse_matmul(a, b, *, sparsity, nnz, is_input_a_sparse, is_input_b_sparse, program_config,
                      compute_kernel_config, dtype, memory_config, optional_output_tensor):
        calls.append(("sparse_matmul", nnz))
        outs = []
        for p in range(P):
            sp = sparsity.chips[p]
            nb = sp.shape[1]
            assert sp.shape == (1, nb, 1, El) and int((sp != 0).sum()) == nnz == nb
            e_of = sp.reshape(nb, El).argmax(-1)
            outs.append(torch.stack([a.chips[p][0, k] @ b.chips[p][0, int(e_of[k])] for k in range(nb)]).unsqueeze(0))
        return new(outs, dtype=dtype)

    def bcast(op, name):
        def f_(a, b, **kw):
            calls.append((name,))
            return new([op(x_, y_) for x_, y_ in zip(a.chips, b.chips)], dtype=kw.get("dtype"))

        return f_

    def polynorm(g_, consts, *, inter, mode, eps, compute_kernel_config, memory_config, up, impl,
                 intermediate_memory_config):
        return new([(consts["c0"].chips[p] * g_.chips[p] + consts["b"].chips[p]) * up.chips[p] for p in range(P)])

    def dtt(t, mesh):
        return torch.stack([c for c in t.chips]).reshape(1, P, *t.shape)

    ns = M.ttnn
    for name, fn in (("from_torch", from_torch), ("to_torch", to_torch), ("embedding", emb), ("slice", slc),
                     ("sparse_matmul", sparse_matmul), ("multiply", bcast(torch.mul, "multiply")),
                     ("eq", bcast(lambda a, b: (a == b).double(), "eq")), ("matmul", bcast(torch.matmul, "matmul"))):
        monkeypatch.setattr(ns, name, fn)
    monkeypatch.setattr(ns, "get_device_tensors", lambda t: [_MTA([c]) for c in t.chips])
    monkeypatch.setattr(ns, "copy_device_to_host_tensor", copy_d2h)
    monkeypatch.setattr(ns, "gather", gather)
    monkeypatch.setattr(ns, "allocate_tensor_on_host", lambda spec, mesh: _MTA([]))
    monkeypatch.setattr(ns, "bitcast", bitcast)
    monkeypatch.setattr(ns, "typecast", typecast)
    monkeypatch.setattr(ns, "to_layout", lambda t, layout, memory_config=None: new([c.clone() for c in t.chips],
                                                                                    dtype=t.dtype, layout=layout))
    monkeypatch.setattr(ns, "transpose", lambda t, a, b, memory_config=None: new([c.transpose(a, b).clone()
                                                                                 for c in t.chips], dtype=t.dtype))
    monkeypatch.setattr(ns, "allocate_tensor_on_device", lambda shape, dtype, layout, mesh, mc: ("OUT", tuple(shape)))
    monkeypatch.setattr(ns, "create_mesh_mapper", lambda mesh, conf: SHARD)
    monkeypatch.setattr(ns, "ReplicateTensorToMesh", lambda mesh: REP)
    monkeypatch.setattr(ns, "MeshMapperConfig", lambda *a, **k: None)
    monkeypatch.setattr(ns, "PlacementShard", lambda d: d)
    monkeypatch.setattr(ns, "MeshShape", lambda *a: a)
    monkeypatch.setattr(ns, "Shape", lambda v: list(v))
    monkeypatch.setattr(ns, "deallocate", lambda t, *a, **k: freed.append(t))

    def reshape(t, shape):
        freed.append(t)
        return new([c.reshape(tuple(shape)).clone() for c in t.chips], dtype=t.dtype, layout=t.layout)

    monkeypatch.setattr(M, "_reshape", reshape)
    monkeypatch.setattr(M._pn, "grouped_polynorm", polynorm)
    monkeypatch.setattr(M, "device_tensors_to_torch", dtt)
    monkeypatch.setattr(M, "PREFILL_MOE_DEVICE_ROWS", (Mr,))  # the kernels serve this toy chunk size

    moe = object.__new__(M.MotifMoE)
    cfg = SimpleNamespace(experts_gate_up_pc=lambda m_tiles: ("pc_gu", m_tiles),
                          experts_down_pc=lambda m_tiles: ("pc_dn", m_tiles), polynorm_eps=1e-6)
    moe.__dict__.update(
        mesh_device=SimpleNamespace(shape=(1, P)), cfg=cfg, hidden=Hs, inter=Is, e_loc=El, top_k=Kk, dram="DRAM",
        prefill_moe="compact", prefill_moe_block="auto", prefill_moe_min_rows=64, prefill_polynorm="bf16",
        prefill_moe_dispatch=disp_mode, prefill_moe_combine=comb_mode, _disp_meta="META",
        internal_route_scale=1.0, gate_up_dtype=None, down_dtype=ttnn.bfloat16, combine_dtype=ttnn.bfloat16,
        combine_mode="fold", decode_experts="dense", decode_rows=(32,),
        ckc_experts="ckc_e", ckc_polynorm="ckc_p", _pn_host=None, scatter_consts={},
        local_ids=_MT([local[p].double().reshape(1, El, 1, 1) for p in range(P)]),
        pn_consts=SimpleNamespace(c={"bf16": {k: _MT([pnh[p, :, i].reshape(1, El, 1, 1) for p in range(P)])
                                              for i, k in enumerate(("c0", "c1", "c2", "b"))}}),
        w_gate_up=_MT([Wgu[p].unsqueeze(0) for p in range(P)]), w_down=_MT([Wd[p].unsqueeze(0) for p in range(P)]),
    )
    moe.compact_state = M.CompactPrefillState(owner=moe)
    st = moe.compact_state

    def w_loc_of(i_, w_, p):  # [El, Mr]
        return ((i_.reshape(-1, 1, Kk) == local[p].double().reshape(1, El, 1)).double()
                * w_.reshape(-1, 1, Kk)).sum(-1).t()

    class FakeDispatch:  # the kernel's contract (kernels/moe_compact.py) on the emulated chips
        def __call__(self, idx, w, meta, *, M, mb, ladder, w_is_loc=True):
            calls.append(("dispatch", M, mb, tuple(ladder), w_is_loc, meta))
            assert meta == "META" and not w_is_loc
            i_ = idx.chips[0].reshape(M, Kk).long()
            assert all(torch.equal(c.reshape(M, Kk).long(), i_) for c in idx.chips)
            need, nb, u = M_.compact_upload_fast(i_.numpy(), st.g2s, P, El, mb, tuple(ladder), moe._pn_bits, M)
            cap = ladder[-1]
            rows, wcol, sp, blk = [], [], [], []
            for p in range(P):
                r = torch.zeros(1, 1, 2, cap * mb, dtype=torch.long)
                r[0, 0, 1] = M
                wc = torch.zeros(1, cap, mb, 1, dtype=torch.float64)
                b = torch.zeros(1, cap, 1, 32, dtype=torch.float64)
                if nb:
                    R = nb * mb
                    up = torch.as_tensor(u[p]).long()
                    r[0, 0, :, :R] = up[:2]
                    wl = w_loc_of(i_.double(), w.chips[0].reshape(M, Kk), p).reshape(-1)
                    wv = torch.where(up[1] < M, wl[up[2]], torch.zeros(()))
                    wc.reshape(-1)[:R] = wv
                    bits = up[3, :32 * nb].reshape(nb, 32)
                    b[0, :nb, 0] = bits.to(torch.int16).view(torch.bfloat16).double()
                spv = b.clone()
                spv[..., 12:] = 0
                rows.append(r)
                wcol.append(wc)
                sp.append(spv)
                blk.append(b)
            nd = torch.tensor([need, nb or 0, 0, 0, 0, 0, 0, 0]).reshape(1, 1, 1, 8)
            return CompactRows(new(rows, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
                               new(wcol, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
                               new(sp, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
                               new(blk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                               new([nd] * P, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT))

    class FakeCombine:
        def __call__(self, y_rm, keys, *, M, key_page, out_dtype, memory_config=None):
            calls.append(("combine", key_page, y_rm.layout, keys.shape[-1]))
            outs = []
            for p in range(P):
                y = y_rm.chips[p].reshape(-1, Hs)
                kk = keys.chips[p].reshape(-1, keys.shape[-1])[key_page].long()
                o = torch.zeros(M, Hs, dtype=torch.float64)
                for j in range(y.shape[0]):
                    if int(kk[j]) < M:
                        o[int(kk[j])] += y[j]
                outs.append(o.reshape(1, 1, M, Hs))
            return new(outs, dtype=out_dtype)

    M_ = M
    st.dispatch, st.combine = FakeDispatch(), FakeCombine()
    routes = {}

    def router(f, *, scale=None, memory_config=None, taps=None):
        i_, w_ = routes["cur"]
        return (new([i_.double().reshape(1, 1, -1, Kk)] * P, dtype=ttnn.uint32),
                new([w_.reshape(1, 1, -1, Kk)] * P, dtype=ttnn.float32))

    moe.router = router
    dense_calls = []

    def local_weights(i_, w_, memory_config=None):
        dense_calls.append("local_weights")
        return new([w_loc_of(i_.chips[p].reshape(-1, Kk), w_.chips[p].reshape(-1, Kk), p).reshape(1, El, -1, 1)
                    for p in range(P)], dtype=ttnn.float32)

    moe.local_weights = local_weights
    moe.experts = lambda f, **kw: dense_calls.append("experts") or "Y"
    moe.reduce_experts = lambda y, memory_config=None: dense_calls.append("reduce") or "PART"
    monkeypatch.setattr(M, "_free", lambda *ts: freed.extend(t for t in ts if t is not None))

    def dense_ref(i_, w_):
        out = []
        for p in range(P):
            acc = torch.zeros(Mr, Hs, dtype=torch.float64)
            for e in range(El):
                wl = ((i_ == int(local[p, e])).double() * w_).sum(-1, keepdim=True)
                gu = x @ Wgu[p, e]
                h = (pnh[p, e, 0] * gu[:, :Is] + pnh[p, e, 3]) * (wl * gu[:, Is:])
                acc = acc + h @ Wd[p, e]
            out.append(acc)
        return out

    f = _MTA([x.reshape(1, 1, Mr, Hs)] * P)
    i1 = torch.stack([torch.randperm(2 * El, generator=g)[:Kk] for _ in range(Mr)])
    w1 = torch.randint(1, 4, (Mr, Kk), generator=g).double()
    routes["cur"] = (i1, w1)
    part = moe.local_partial(f, polynorm="bf16", decode=False)
    assert st.stats["compact"] == 1 and st.stats["device_dispatch"] == (1 if disp_mode == "device" else 0)
    for p, want in enumerate(dense_ref(i1, w1)):
        assert torch.equal(part.chips[p].reshape(Mr, Hs), want), p
    mb = M.compact_block(Mr, "auto")
    lad = M.compact_ladder(Mr, mb, El)
    nb = M.compact_bucket(M.compact_need_blocks(i1, local, mb), lad)
    assert [c[1] for c in calls if c[0] == "sparse_matmul"] == [nb, nb]
    reads = [c for c in calls if c[0] == "read"]
    disp = [c for c in calls if c[0] == "dispatch"]
    ups = [c for c in calls if c[0] == "upload"]
    if disp_mode == "device":
        assert disp == [("dispatch", Mr, mb, lad, False, "META")]
        assert reads == [("read", True, ttnn.uint32, (1, 1, 1, 8))]  # only the 32-byte need
        assert not [u for u in ups if u[1] == ttnn.uint32], "the routes are never uploaded"
        assert [c for c in calls if c[0] == "embedding"] == [("embedding", ttnn.uint32, (1, 1, 1, nb * mb))]
    else:
        assert not disp and len(reads) == 1 and reads[0][3] == (1, 1, Mr, Kk)
        assert [(u[1], u[2], u[3]) for u in ups if u[1] == ttnn.uint32] == [(ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT,
                                                                             (1, P, 4, nb * mb))]
    comb = [c for c in calls if c[0] == "combine"]
    if comb_mode == "gather":
        assert comb == [("combine", 1, ttnn.ROW_MAJOR_LAYOUT, (lad[-1] if disp_mode == "device" else nb) * mb)]
        assert not [c for c in calls if c[0] in ("eq", "matmul")]
    else:
        assert not comb and [c[0] for c in calls if c[0] in ("eq", "matmul")] == ["eq", "matmul"]
    ids_freed = [id(t) for t in freed]
    assert len(ids_freed) == len(set(ids_freed)), "a tensor was freed twice"
    assert id(f) not in ids_freed and id(part) not in ids_freed
    live = [t for t in created if id(t) not in set(ids_freed) and t is not part and t is not f]
    if comb_mode == "matmul":
        assert [t.shape for t in live] == [[1, 1, Mr, 1]] and st.iota[Mr] is live[0]
    else:
        assert not live, [t.shape for t in live]
    # beyond the cap -> dense on the same routes
    i2 = torch.stack([torch.tensor([5, 0] if t < 66 else [7, 2]) for t in range(Mr)])
    routes["cur"] = (i2, w1)
    calls.clear()
    dense_calls.clear()
    assert moe.local_partial(f, polynorm="bf16", decode=False) == "PART"
    assert dense_calls == ["local_weights", "experts", "reduce"] and st.stats["dense_cap"] == 1
    assert not [c for c in calls if c[0] in ("sparse_matmul", "combine")]
    # frozen + unwarmed -> dense; warm_compact -> compacted again (same partial)
    routes["cur"] = (i1, w1)
    st.frozen = lambda: True
    dense_calls.clear()
    calls.clear()
    assert moe.local_partial(f, polynorm="bf16", decode=False) == "PART" and st.stats["dense_unwarmed"] == 1
    # an unwarmed chunk size launches no dispatch program after the capture (F3N rule R2)
    assert not [c for c in calls if c[0] in ("dispatch", "sparse_matmul", "combine")]
    routes["cur"] = (torch.zeros(Mr, Kk, dtype=torch.long), torch.zeros(Mr, Kk))
    calls.clear()
    assert moe.warm_compact(Mr) == lad and st.warmed == {(Mr, mb, b) for b in lad}
    if disp_mode == "device":
        assert len([c for c in calls if c[0] == "dispatch"]) == 1
        assert [c[1] for c in calls if c[0] == "sparse_matmul"] == [b for b in lad for _ in range(2)]
    routes["cur"] = (i1, w1)
    part2 = moe.local_partial(f, polynorm="bf16", decode=False)
    assert all(torch.equal(a, b) for a, b in zip(part.chips, part2.chips))
    st.frozen = lambda: False
    # a chunk size outside PREFILL_MOE_DEVICE_ROWS (not validated on device) never reaches the B2b kernels, frozen or
    # not: B2a's host dispatch (one route read + upload) and the one-hot matmul combine, the same partial
    monkeypatch.setattr(M, "PREFILL_MOE_DEVICE_ROWS", (1024, 2048, 4096))
    assert not M.b2b_rows_ok(Mr) and M.b2b_rows_ok(4096) and not M.b2b_rows_ok(3968)
    calls.clear()
    part3 = moe.local_partial(f, polynorm="bf16", decode=False)
    assert all(torch.equal(a, b) for a, b in zip(part.chips, part3.chips))
    assert not [c for c in calls if c[0] in ("dispatch", "combine")]
    assert [c[0] for c in calls if c[0] in ("eq", "matmul")] == ["eq", "matmul"]
    assert len([c for c in calls if c[0] == "read"]) == 1
    ids_freed = [id(t) for t in freed if isinstance(t, _MT)]  # (the dense stand-ins are strings)
    assert len(ids_freed) == len(set(ids_freed)), "a tensor was freed twice"


def test_moe_host_compact_kernel_helpers():
    """B2b host helpers (``kernels/moe_compact.py``) and knob resolution: ``slot_table`` (global id -> ``p E + e``,
    every id held once), ``chip_meta`` (chip index, ids, PolyNorm bf16 bits), ``dispatch_reference`` (the kernel's
    outputs from B2a's upload: tokens, keys, block sparsity with columns 12.. cleared, block words), ``combine_groups``,
    and ``resolve_prefill_moe_kernels`` (explicit > config > B2a defaults; anything else raises)."""
    from models.demos.motif3.tt.kernels.moe_compact import (chip_meta, combine_groups, dispatch_reference,
                                                            slot_table)
    import models.demos.motif3.tt.moe as M

    local = torch.tensor([[5, 0, 7, 2], [1, 6, 3, 4]])
    st = slot_table(local)
    assert st.tolist() == [1, 4, 3, 6, 7, 0, 5, 2]
    with pytest.raises(ValueError, match="exactly once"):
        slot_table(torch.tensor([[0, 1], [1, 3]]))
    bits = torch.arange(2 * 4 * 4).reshape(2, 4, 4) | 0x3F00
    m = chip_meta(local, bits)
    assert m.shape == (2, 64) and m[:, 0].tolist() == [0, 1] and m[1, 1:5].tolist() == [1, 6, 3, 4]
    assert m[1, 16 + 4 * 2 + 3].item() == int(bits[1, 2, 3]) and int(m[:, 5:16].abs().sum()) == 0
    with pytest.raises(ValueError, match="PolyNorm bits"):
        chip_meta(local, bits[:, :3])
    g = torch.Generator().manual_seed(3)
    Mr, Kk, mb, P, El = 128, 2, 32, 2, 4
    idx = torch.stack([torch.randperm(8, generator=g)[:Kk] for _ in range(Mr)])
    g2s = M._global_to_slot(local).numpy()
    pn_bits = (torch.randn(P, El, 4).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF).numpy()
    lad = M.compact_ladder(Mr, mb, El)
    need, nb, u = M.compact_upload_fast(idx.numpy(), g2s, P, El, mb, lad, pn_bits, Mr)
    tix, key, sp, blk = dispatch_reference(u, mb, nb, Mr)
    meta = M.compact_prefill_meta(idx, torch.ones(Mr, Kk), local, mb, nb)
    assert torch.equal(tix, meta["tok"].long()) and torch.equal(key, torch.where(meta["tokv"] >= 0, meta["tokv"].long(),
                                                                                 torch.full_like(key, Mr)))
    assert torch.equal((sp[..., :El] == 0x3F80).float(), meta["sparsity"]) and int(sp[..., 12:].abs().sum()) == 0
    assert torch.equal(blk[..., 16:20], torch.from_numpy(pn_bits).long()[torch.arange(P)[:, None], meta["eblk"]])
    assert combine_groups(1024, 130, 128) == 8 and combine_groups(4096, 130, 128) == 2
    assert combine_groups(8192, 130, 128) == 1 and combine_groups(32, 130, 128) == 16
    cfg = SimpleNamespace(prefill_moe_dispatch="device", prefill_moe_combine="gather")
    assert M.resolve_prefill_moe_kernels(None, None, cfg) == ("device", "gather")
    assert M.resolve_prefill_moe_kernels("host", "matmul", cfg) == ("host", "matmul")
    assert M.resolve_prefill_moe_kernels(None, None, SimpleNamespace()) == ("host", "matmul")
    with pytest.raises(ValueError, match="prefill_moe_dispatch"):
        M.resolve_prefill_moe_kernels("gpu", None, cfg)
    with pytest.raises(ValueError, match="prefill_moe_combine"):
        M.resolve_prefill_moe_kernels(None, "scatter", cfg)


def test_moe_host_model_prefill_moe_rows():
    """B2a model wiring (``MotifModel.compact_moes`` / ``prefill_moe_rows`` / ``warm_prefill_moe``): the chunk sizes of
    the passes' rows (``prefill_chunk``-row chunks plus a remainder) that the compacted path serves; the warm-up reads
    every compacted layer's host constants and compiles each chunk size once, on the first layer; nothing without a
    compacted layer."""
    from models.demos.motif3.tt.model import MotifModel

    log = []

    def moe(name, mode="compact"):
        return SimpleNamespace(
            prefill_moe=mode, prefill_chunk=4096, compact_applies=lambda r: mode == "compact" and int(r) >= 1024,
            prepare_compact=lambda: log.append(("prepare", name)),
            warm_compact=lambda r: log.append(("warm", name, r)) or (12, 15))

    m = object.__new__(MotifModel)
    m.layers = [SimpleNamespace(moe=None), SimpleNamespace(moe=moe("L2")), SimpleNamespace(moe=moe("L3"))]
    assert [x.prefill_chunk for x in m.compact_moes()] == [4096, 4096]
    assert m.prefill_moe_rows([128, 512, 1024, 2048, 4096, 8192, 6144, 1024]) == [1024, 2048, 4096]
    assert m.warm_prefill_moe([128, 1024, 8192]) == {1024: (12, 15), 4096: (12, 15)}
    assert log == [("prepare", "L2"), ("prepare", "L3"), ("warm", "L2", 1024), ("warm", "L2", 4096)]
    m.layers = [SimpleNamespace(moe=moe("L2", "dense"))]
    assert m.compact_moes() == [] and m.prefill_moe_rows([4096]) == [] and m.warm_prefill_moe([4096]) == {}


def _compact_upload_fast_b2a(idx, g2s, P: int, E: int, mb: int, ladder: Tuple[int, ...], pn_bits, rows_m: int):
    """The B2a original of ``compact_upload_fast`` (motif3-opt 5a64c022585), kept as the P2 reference."""
    import numpy as np

    from models.demos.motif3.tt.moe import compact_bucket

    K = int(np.asarray(idx).shape[-1])
    idx = np.asarray(idx).reshape(-1).astype(np.int64, copy=False)
    S_K = idx.size
    pad_key = int(rows_m)
    if S_K and (int(idx.min()) < 0 or int(idx.max()) >= g2s.size):
        raise ValueError("a routed expert id is not any chip's local expert")
    pe = g2s[idx]
    if S_K and int(pe.min()) < 0:
        raise ValueError("a routed expert id is not any chip's local expert")
    cnt = np.bincount(pe, minlength=P * E)
    nblk = ((cnt + (mb - 1)) // mb).reshape(P, E)
    need = int(nblk.sum(1).max()) if S_K else 0
    nb = compact_bucket(need, ladder)
    if nb is None:
        return need, None, None
    rows = int(nb) * int(mb)
    order = np.argsort(pe, kind="stable")  # flattened order is token-major: tokens ascending within each slot
    pe_s = pe[order]
    start = np.cumsum(cnt) - cnt
    blk_end = np.cumsum(nblk, axis=1)
    blk_off = (blk_end - nblk).reshape(-1)
    row = blk_off[pe_s] * mb + (np.arange(S_K) - start[pe_s])
    flat = (pe_s // E) * (4 * rows) + row
    t_s = (order // K).astype(np.int32)
    u = np.zeros((P, 4, rows), dtype=np.int32)
    u[:, 1, :] = int(pad_key)
    uf = u.reshape(-1)
    uf[flat] = t_s
    uf[flat + rows] = t_s
    uf[flat + 2 * rows] = (pe_s % E).astype(np.int32) * int(pad_key) + t_s
    eblk = np.minimum((blk_end[:, :, None] <= np.arange(nb)[None, None, :]).sum(1), E - 1)  # [P, nb]
    blk = np.zeros((P, int(nb), 32), dtype=np.int32)
    np.put_along_axis(blk, eblk[:, :, None], 0x3F80, axis=2)  # bf16 1.0 at the block's expert
    blk[:, :, 16:20] = np.take_along_axis(pn_bits, eblk[:, :, None], axis=1)
    u[:, 3, : 32 * int(nb)] = blk.reshape(P, -1)
    return need, int(nb), u


@pytest.mark.parametrize("S", [128, 1024, 2048, 4096, 8064])
def test_moe_host_compact_upload_fast_p2(S):
    """P2 (logs/opt/phaseC/P2): the rewritten ``compact_upload_fast`` (16-bit radix argsort, one gather per word row)
    returns word for word what the B2a original returns: uniform and skewed routes, ``need`` beyond the cap, 8 seeds."""
    import models.demos.motif3.tt.moe as M

    P, E, K = 32, 12, 8
    G = P * E
    for seed in range(8):
        rng = np.random.default_rng(1000 * S + seed)
        local = rng.permutation(G).reshape(P, E)
        g2s = np.full(G, -1, dtype=np.int64)
        g2s[local.reshape(-1)] = np.arange(G)
        p = (rng.pareto(1.2, G) + 0.02) if seed % 2 else np.ones(G)
        p = p / p.sum()
        idx = np.stack([rng.choice(G, K, replace=False, p=p) for _ in range(S)])
        pn = rng.integers(0, 1 << 16, (P, E, 4)).astype(np.int32)
        mb = M.compact_block(S, "auto")
        for ladder in (M.compact_ladder(S, mb, E), (E,)):
            a = M.compact_upload_fast(idx, g2s, P, E, mb, ladder, pn, S)
            b = _compact_upload_fast_b2a(idx, g2s, P, E, mb, ladder, pn, S)
            assert a[0] == b[0] and a[1] == b[1]
            assert (a[2] is None and b[2] is None) or (a[2].dtype == b[2].dtype and np.array_equal(a[2], b[2]))


def test_moe_host_staged_upload(monkeypatch):
    """P2 ``prefill_moe_upload="staged"`` (``MotifMoE._upload_words``) on a fake (1, 2) mesh: the first call per row
    count is the B2a ``from_torch`` upload and builds the staging (one host mesh tensor, zero-copy views); later calls
    write ``u[p]`` into view ``p`` and copy the host tensor into a fresh device tensor -- chip ``p`` receives ``u[p]``
    exactly (int64 and int32 inputs); a new row count builds its own staging; shards without zero-copy views fall back
    to ``from_torch`` for that row count; ``"from_torch"`` never stages."""
    import models.demos.motif3.tt.moe as M

    P = 2
    calls = []

    class Shard:
        def __init__(self, n, zero_copy=True):
            self.buf, self.zero_copy = torch.zeros(n, dtype=torch.int32), zero_copy

        def to_torch_with_padded_shape(self):
            return self.buf if self.zero_copy else self.buf.clone()

    class Host:
        def __init__(self, n, zero_copy):
            self.shards = [Shard(n, zero_copy) for _ in range(P)]

    class Dev:
        def __init__(self, chips, shape=None):
            self.chips, self.spec = chips, ("spec", shape)

    zero_copy = {"v": True}

    def from_torch(h, *, dtype, layout, device, memory_config, mesh_mapper):
        calls.append(("from_torch", tuple(h.shape)))
        assert dtype == ttnn.uint32 and layout == ttnn.ROW_MAJOR_LAYOUT and h.dtype == torch.int32
        return Dev([h.reshape(P, -1)[p].clone() for p in range(P)], tuple(h.shape))

    def alloc_host(spec, mesh):
        calls.append(("alloc_host", spec))
        return Host(4 * spec[1][-1], zero_copy["v"])

    def alloc_dev(shape, dtype, layout, mesh, mc):
        calls.append(("alloc_dev", tuple(shape)))
        return Dev([None] * P)

    def h2d(h, d):
        calls.append(("h2d",))
        d.chips = [s.buf.clone() for s in h.shards]

    ns = M.ttnn
    monkeypatch.setattr(ns, "from_torch", from_torch)
    monkeypatch.setattr(ns, "allocate_tensor_on_host", alloc_host)
    monkeypatch.setattr(ns, "allocate_tensor_on_device", alloc_dev)
    monkeypatch.setattr(ns, "copy_host_to_device_tensor", h2d)
    monkeypatch.setattr(ns, "get_device_tensors", lambda h: h.shards)
    monkeypatch.setattr(ns, "Shape", lambda v: list(v))

    moe = M.MotifMoE.__new__(M.MotifMoE)
    moe.compact_state = M.CompactPrefillState()
    moe.compact_state.mapper = "SHARD"
    moe.mesh_device, moe.dram = "MESH", "DRAM"
    moe._mesh_rc = lambda: (1, P)
    moe.prefill_moe_upload = "staged"
    g = torch.Generator().manual_seed(3)

    def words(rows, dt):
        return torch.randint(0, 2 ** 31 - 1, (P, 4, rows), generator=g, dtype=torch.int64).to(dt)

    for i, (rows, dt) in enumerate([(64, torch.int64), (64, torch.int64), (64, torch.int32), (96, torch.int32),
                                    (96, torch.int64)]):
        u = words(rows, dt)
        n0 = len(calls)
        U = moe._upload_words(u if i % 2 else u.numpy(), rows)
        for p in range(P):
            assert torch.equal(U.chips[p].long(), u[p].reshape(-1).long()), (i, p)
        kinds = [c[0] for c in calls[n0:]]
        first = i in (0, 3)
        assert kinds == (["from_torch", "alloc_host"] if first else ["alloc_dev", "h2d"]), (i, kinds)
    assert sorted(moe.compact_state.upload_bufs) == [64, 96]
    # no zero-copy views: that row count uploads with from_torch every time
    zero_copy["v"] = False
    for _ in range(2):
        u = words(32, torch.int32)
        U = moe._upload_words(u.numpy(), 32)
        assert all(torch.equal(U.chips[p], u[p].reshape(-1)) for p in range(P))
    assert moe.compact_state.upload_bufs[32] is False and calls[-1][0] == "from_torch"
    # "from_torch": never staged
    moe2 = M.MotifMoE.__new__(M.MotifMoE)
    moe2.__dict__.update(moe.__dict__)
    moe2.compact_state = M.CompactPrefillState()
    moe2.compact_state.mapper = "SHARD"
    moe2.prefill_moe_upload = "from_torch"
    n0 = len(calls)
    for _ in range(2):
        moe2._upload_words(words(64, torch.int32), 64)
    assert [c[0] for c in calls[n0:]] == ["from_torch", "from_torch"] and not moe2.compact_state.upload_bufs


def test_moe_host_import_clean():
    code = (
        "import sys; import models.demos.motif3.tt.moe as m; "
        "bad = sorted(k for k in sys.modules if k.startswith('models.demos.') and not k.startswith('models.demos.motif3'));"
        "print('BAD', bad)"
    )
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("BAD")][-1]
    assert line == "BAD []", line


# ============================================================================================================
# device helpers
# ============================================================================================================
def _free(o, _seen=None):
    """Deallocate tensors in nested lists / tuples / dicts; each tensor once, already-freed ones skipped (taps may
    alias the input on a size-1 axis)."""
    seen = set() if _seen is None else _seen
    if isinstance(o, (list, tuple)):
        for x in o:
            _free(x, seen)
    elif isinstance(o, dict):
        for x in o.values():
            _free(x, seen)
    elif isinstance(o, ttnn.Tensor):
        if id(o) in seen:
            return
        seen.add(id(o))
        if o.is_allocated():
            ttnn.deallocate(o)


class _Capture:
    """Exception-safe trace capture (GATES_RESULTS §11.6: a dangling capture hung close_mesh_device)."""

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


SYNC_FLOOR_US = 200.0  # GATES_RESULTS §2 (G0): synchronize_device alone costs 133-149 us, flat t(n) up to ~185 us


def traced_stats(mesh_device, fn, n=16, reps=7, adapt_to: int = 0) -> dict:
    """Traced per-call cost by the gates' slope method: ``(t(n) - t(n/2)) / (n/2)``, minimum over replays.

    ``floor_limited``: ``t(n/2)`` is under the sync floor, so device work hides behind ``synchronize_device`` and the
    slope underestimates (down to 0, review issue 8). With ``adapt_to`` > n the measurement doubles ``n`` up to that
    while it is floor-limited. ``upper`` = ``t(n)/n``, an upper bound (it includes the sync)."""
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    while True:
        t1 = _trace_min_us(mesh_device, fn, n // 2, reps)
        t2 = _trace_min_us(mesh_device, fn, n, reps)
        limited = t1 < SYNC_FLOOR_US
        if not limited or 2 * n > adapt_to:
            break
        n *= 2
    return {"us": max((t2 - t1) / (n - n // 2), 0.0), "upper": t2 / n, "n": n, "floor_limited": limited}


def traced_us(mesh_device, fn, n=16, reps=7):
    """Traced per-call us (slope method; see :func:`traced_stats`)."""
    return traced_stats(mesh_device, fn, n=n, reps=reps)["us"]


def fmt_traced(st) -> str:
    if not isinstance(st, dict):
        return str(st)
    s = f"{st['us']:.1f}"
    if st["floor_limited"]:
        s += f" (sync-floor limited at n={st['n']}: <= {st['upper']:.1f})"
    return s


def eager_us(mesh_device, fn, iters=5, warmup=1):
    for _ in range(warmup):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


def _setup(mesh_device, tag):
    from models.demos.motif3.tt.ccl import MotifCCL, log_fabric
    from models.demos.motif3.tt.model_config import MotifTTConfig

    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device)
    rep = log_fabric(mesh_device, tag)
    print(f"[moe] {cfg.describe()}")
    return cfg, MotifCCL(mesh_device, cfg), rep


def upload_lanes(x32: torch.Tensor, cfg, mesh_device, device=True):
    """Lane-ordered ``[32, 4096]`` -> per-row ``[1, 1, lanes_per_row, 4096]`` bf16 TILE (row ``dp`` = lanes
    ``8 dp .. 8 dp + 7`` on (4, 8), replicated over TP)."""
    from models.demos.motif3.tt.rope import lanes_to_rows, shard_lanes

    rows = lanes_to_rows(x32.bfloat16(), cfg).reshape(cfg.dp, 1, cfg.lanes_per_row, H)
    return shard_lanes(rows, cfg, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                       device=mesh_device if device else None)


def read_rows(t, cfg, mesh_device):
    """Row outputs ``[1, 1, L, W]`` -> ``([32, W] lane order from TP index 0, replicas identical over TP)``."""
    from models.demos.motif3.tt.ccl import device_tensors_to_torch

    full = device_tensors_to_torch(t, mesh_device)  # [R, C, 1, 1, L, W]
    rows, same = [], True
    for dp in range(cfg.dp):
        r0 = full[cfg.axes.coord(dp, 0)]
        for tp in range(cfg.tp):
            same &= bool(torch.equal(full[cfg.axes.coord(dp, tp)], r0))
        rows.append(r0.reshape(-1, r0.shape[-1]))
    return torch.cat(rows).float(), same


def read_replicated(t, mesh_device, chips=None):
    """A tensor that must be identical on all chips -> (chip-0 value, the chips are bitwise identical). ``chips``:
    linear device indices to read (default all; large prefill outputs read a subset)."""
    shards = ttnn.get_device_tensors(t)
    idx = range(len(shards)) if chips is None else chips
    ref = ttnn.to_torch(shards[0])
    same = True
    for i in idx:
        if i == 0:
            continue
        same &= bool(torch.equal(ttnn.to_torch(shards[i]), ref))
    return ref, same


def read_routes(taps, moe, mesh_device, chips=None):
    """``forward_decode`` / ``local_partial`` taps -> ``(idx [M, K] long, w [M, K] fp32, ok)``: the top-8 ids and
    their routing weights at the module's internal route scale, as the gather path's ``taps["w"]`` holds them.

    With ``router_mask="scatter"`` (A5, ``MOTIF3_ROUTER_MASK=scatter``) the module taps no ``"w"``: ``w`` is then read
    back from every chip's ``w_loc [1, E_loc, M, 1]`` through the module's ``local_ids`` (each expert lives on exactly
    one chip). ``ok``: ``idx`` (and ``w``) identical on the chips read (gather path), or, on the scatter path, ``idx``
    identical on all chips, every selected expert found once and every unselected local weight exactly 0. ``chips``
    limits the replicated reads (gather path only; the scatter path needs every chip)."""
    idx, i_same = read_replicated(taps["idx"], mesh_device, chips=chips)
    idx = idx.reshape(-1, K).long()
    if "w" in taps:
        w, w_same = read_replicated(taps["w"], mesh_device, chips=chips)
        return idx, w.reshape(-1, K).float(), bool(i_same and w_same)
    if chips is not None:
        i_same = read_replicated(taps["idx"], mesh_device)[1]
    M = idx.shape[0]
    dense = torch.full((M, moe.n_experts), float("nan"), dtype=torch.float32)
    seen = torch.zeros(moe.n_experts, dtype=torch.long)
    for s_ids, s_w in zip(ttnn.get_device_tensors(moe.local_ids), ttnn.get_device_tensors(taps["w_loc"])):
        e = ttnn.to_torch(s_ids).reshape(-1).long()
        v = ttnn.to_torch(s_w).float().reshape(len(e), -1)[:, :M]
        dense[:, e] = v.t()
        seen[e] += 1
    w = dense.gather(1, idx)
    unsel = torch.ones_like(dense, dtype=torch.bool).scatter_(1, idx, False)
    ok = bool(i_same) and bool((seen == 1).all()) and bool(torch.isfinite(w).all()) and bool(
        (dense[unsel] == 0).all())
    return idx, w, ok


def upload_replicated(x: torch.Tensor, mesh_device, dtype=ttnn.bfloat16, device=True):
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape[-2:]) if x.dim() == 2 else x,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device if device else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def upload_per_chip(x: torch.Tensor, mesh_device, dtype=ttnn.bfloat16):
    """``[R, C, M, W]`` -> chip (r, c) holds ``[1, 1, M, W]`` = ``x[r, c]``."""
    R, C = tuple(mesh_device.shape)
    mapper = ttnn.create_mesh_mapper(
        mesh_device, ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], ttnn.MeshShape(R, C))
    )
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)


def _device_params():
    from models.demos.motif3.tt.model_config import device_params

    return device_params("FABRIC_2D_TORUS_XY", TRACE)


MESH_PARAMS = [pytest.param((4, 8), _device_params(), id="4x8-torus2d")]


def _check_decode(moe, x32, cfg, mesh_device, ref, tag, *, idx_ref=None, w_ref=None, want=None, biased_ref=None,
                  inspect=None, mesh=None):
    """Run ``forward_decode`` with taps; check gathered tokens, route identity on all chips, replicas, accuracy (vs the
    reference on its own routes and on the device's routes). ``inspect(taps) -> dict`` adds entries (eager taps)."""
    mesh = mesh or mesh_device
    x_tt = upload_lanes(x32, cfg, mesh)
    taps = {}
    out = moe.forward_decode(x_tt, taps=taps)
    got, same_tp = read_rows(out, cfg, mesh)
    f_all, f_same = read_replicated(taps["f_all"], mesh)
    idx_dev, w_dev, routes_same = read_routes(taps, moe, mesh)
    w_dev = w_dev.double() * (moe.route_scale / moe.internal_route_scale)
    res = {
        "tag": tag,
        "replicas_identical_tp": same_tp,
        "gathered_exact": bool(torch.equal(f_all.reshape(-1, H).float(), x32.float())) and f_same,
        "routes_identical_32": routes_same,
    }
    if inspect is not None:
        res.update(inspect(taps))
    if want is None:
        want, idx_ref, w_ref = ref(x32)
    res["vs_ref"] = stats(want, got)
    if biased_ref is not None:
        agr, eq, margin = set_agreement(idx_dev, idx_ref, biased_ref)
        res["set_agreement_vs_fp64"] = agr
        res["mismatch_margins"] = [float(m) for m in margin[~eq]]
        res["weight_err_matched"] = matched_weight_err(idx_dev, w_dev, idx_ref, w_ref, eq)
    # expert-path fidelity independent of near-tie route flips: the reference experts on the device's routes
    want_dev_routes, _, _ = ref(x32, idx_dev, w_dev.float())
    res["vs_ref_device_routes"] = stats(want_dev_routes, got)
    # every token that misses 0.999 against the reference's own routes must be a route flip
    idx_want, _ = ref.route(x32)
    low = (row_pcc(want, got) < TOKEN_PCC_MIN).nonzero().flatten()
    res["low_tokens_not_flipped"] = [int(t) for t in low if bool(same_sets(idx_dev[t:t + 1], idx_want[t:t + 1]))]
    res["input_still_allocated"] = x_tt.is_allocated()
    _free(taps)
    _free(out)
    _free(x_tt)
    return res, got


def _decode_gates(res, failures, pcc_thr=0.995, tok_thr=TOKEN_PCC_MIN):
    """The decode acceptance gates (README §12 + review issue 3) on a :func:`_check_decode` result."""
    tag = res["tag"]
    for k in ("replicas_identical_tp", "gathered_exact", "routes_identical_32"):
        if not res[k]:
            failures.append(f"{tag}: {k} false")
    if res["vs_ref"]["pcc"] < pcc_thr:
        failures.append(f"{tag}: PCC {res['vs_ref']['pcc']:.6f} < {pcc_thr}")
    if res["vs_ref_device_routes"]["min_token_pcc"] < tok_thr:
        failures.append(f"{tag}: min token PCC on the device's routes {res['vs_ref_device_routes']['min_token_pcc']:.6f}"
                        f" < {tok_thr} (token {res['vs_ref_device_routes']['worst_token']})")
    if res["low_tokens_not_flipped"]:
        failures.append(f"{tag}: tokens {res['low_tokens_not_flipped']} miss {tok_thr} without a route flip")
    if not res.get("input_still_allocated", True):
        failures.append(f"{tag}: forward_decode freed its input")


def _report(res):
    keys = [k for k in res if k.startswith("vs_")]
    line = " ".join(f"{k}: {fmt(res[k])};" for k in keys)
    extra = {k: v for k, v in res.items() if not k.startswith("vs_") and k != "tag"}
    print(f"[moe] {res['tag']}: {line} {extra}")


def prefill_device_routes(moe, x_tt):
    """The routes ``forward_prefill`` uses: the router at the module's chunk shape (``min(S, prefill_chunk)`` rows, the
    same program as inside ``local_partial``) -> ``(idx [S, 8] long, w [S, 8] fp64 at route_scale)`` from chip 0."""
    S = int(x_tt.shape[-2])
    C = min(S, moe.prefill_chunk)
    ii, ww = [], []
    for c0 in range(0, S, C):
        c1 = min(S, c0 + C)
        xc = x_tt if (c0 == 0 and c1 == S) else ttnn.slice(x_tt, [0, 0, c0, 0], [1, 1, c1, H],
                                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        i, w = moe.router(xc, scale=moe.internal_route_scale)
        ii.append(read_replicated(i, moe.mesh_device, chips=[0])[0].reshape(-1, K).long())
        ww.append(read_replicated(w, moe.mesh_device, chips=[0])[0].reshape(-1, K).double())
        _free([i, w] + ([xc] if xc is not x_tt else []))
    return torch.cat(ii), torch.cat(ww) * (moe.route_scale / moe.internal_route_scale)


def ref_on_routes(ref, xs, tok, idx, w):
    """Reference routed output for positions with token ids ``tok`` (rows of ``xs``) on per-position routes; when every
    position of a token has the same route (the device router is row-local), each token is computed once."""
    first = {}
    for p, t in enumerate(tok.tolist()):
        first.setdefault(t, p)
    uniq = torch.tensor(sorted(first))
    pos = torch.tensor([first[int(t)] for t in uniq])
    o = torch.argsort(idx, dim=-1)
    ids, ws = idx.gather(-1, o), w.gather(-1, o)
    inv = torch.searchsorted(uniq, tok)
    consistent = bool(torch.equal(ids, ids[pos][inv])) and bool(torch.equal(ws, ws[pos][inv]))
    if consistent:
        want_u, _, _ = ref(xs[uniq], idx[pos], w[pos].float())
        return want_u[inv], True
    want, _, _ = ref(xs[tok], idx, w.float())
    return want, False


def _check_prefill(moe, ref, xs, tok, got, want_ref_routes, routes, failures, tag):
    """Prefill gates on ``got [S, 4096]`` for positions ``tok``: PCC >= 0.995 vs the reference routes, every token
    >= 0.999 on the device's routes (``routes`` from :func:`prefill_device_routes`), low tokens only at route flips."""
    idx_d, w_d = routes
    want_dev, consistent = ref_on_routes(ref, xs, tok, idx_d, w_d)
    s_ref = stats(want_ref_routes, got)
    s_dev = stats(want_dev, got)
    idx_ref, _ = ref.route(xs[tok])
    flipped = ~same_sets(idx_d, idx_ref)
    low = (row_pcc(want_ref_routes, got) < TOKEN_PCC_MIN).nonzero().flatten()
    low_unflipped = [int(p) for p in low if not bool(flipped[p])]
    print(f"[moe] {tag}: vs ref routes {fmt(s_ref)}; on device routes {fmt(s_dev)} (routes position-independent: "
          f"{consistent}); route flips vs the reference router {int(flipped.sum())} of {len(tok)} positions; "
          f"positions < {TOKEN_PCC_MIN} vs ref routes {len(low)}, of them not flipped {low_unflipped[:8]}")
    if s_ref["pcc"] < 0.995:
        failures.append(f"{tag}: PCC {s_ref['pcc']:.6f} < 0.995")
    if s_dev["min_token_pcc"] < TOKEN_PCC_MIN:
        failures.append(f"{tag}: min token PCC on the device's routes {s_dev['min_token_pcc']:.6f} (position "
                        f"{s_dev['worst_token']})")
    if low_unflipped:
        failures.append(f"{tag}: positions {low_unflipped[:8]} miss {TOKEN_PCC_MIN} without a route flip")
    return s_ref, s_dev


# ============================================================================================================
# device tests
# ============================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_router_real(mesh_device, device_params):
    """D1 / MOE-7: top-8 set agreement of the fp32 composite router vs an fp64 router on real router inputs."""
    from models.demos.motif3.tt.ccl import device_tensors_to_torch
    from models.demos.motif3.tt.moe import MotifRouter
    from models.demos.motif3.tt.weights import HFWeightLoader, hf_name

    cfg, ccl, _ = _setup(mesh_device, "moe_router_real")
    data = load_real_inputs()
    src = HFWeightLoader()
    R, C = cfg.axes.mesh_shape
    rows, failures = [], []
    tot_n = tot_eq = 0
    tot_eq_ref = 0
    worst_margin = 0.0
    for L in data["meta"]["layers"]:
        if not all(src.available(hf_name(L, s)) for s in ("moe.router.gate.weight", "moe.expert_bias")):
            continue
        d = data["layers"][L]
        x = d["x"].float()
        n = x.shape[0]
        per_chip = -(-n // (R * C * 32)) * 32
        xp = torch.zeros(R * C * per_chip, H)
        xp[:n] = x
        router = MotifRouter(mesh_device, cfg, L, source=src, cache=False)
        x_tt = upload_per_chip(xp.reshape(R, C, per_chip, H), mesh_device)
        idx_tt, w_tt = router(x_tt)
        idx_dev = device_tensors_to_torch(idx_tt, mesh_device).reshape(-1, K)[:n].long()
        w_dev = device_tensors_to_torch(w_tt, mesh_device).reshape(-1, K)[:n].double()
        _free([x_tt, idx_tt, w_tt])
        # routing invariant: identical inputs -> bitwise identical routes on all 32 chips
        x32 = upload_replicated(x[:32], mesh_device)
        i32, w32 = router(x32)
        _, same_i = read_replicated(i32, mesh_device)
        _, same_w = read_replicated(w32, mesh_device)
        _free([x32, i32, w32])
        router.deallocate()
        gate_w = src.get(hf_name(L, "moe.router.gate.weight"))
        bias = src.get(hf_name(L, "moe.expert_bias"))
        idx_ref, w_ref, biased = router_golden(x, gate_w, bias)
        agr, eq, margin = set_agreement(idx_dev, idx_ref, biased)
        agr_ref, eq_ref, _ = set_agreement(d["ref_idx"], idx_ref, biased)  # the reference's CPU fp32 router vs fp64
        werr = matched_weight_err(idx_dev, w_dev, idx_ref, w_ref, eq)
        mm = margin[~eq]
        worst = float(mm.max()) if mm.numel() else 0.0
        worst_margin = max(worst_margin, worst)
        near = float((margin < 2e-4).double().mean())
        rows.append(
            f"L{L:02d} {cfg.layer(L).attn_kind:6s} n={n} agree={agr * 100:.3f}% ({int((~eq).sum())} flips, max flip "
            f"margin {worst:.2e}, tokens with margin<2e-4: {near * 100:.2f}%) w_err(matched)={werr:.2e} "
            f"ref_fp32_vs_fp64={agr_ref * 100:.3f}% routes_identical_32={same_i and same_w}"
        )
        tot_n += n
        tot_eq += int(eq.sum())
        tot_eq_ref += int(eq_ref.sum())
        if not (same_i and same_w):
            failures.append(f"L{L}: routes differ between chips")
        if agr < 0.985:
            failures.append(f"L{L}: agreement {agr:.4f} < 0.985 (sanity floor; D1 accepts the composite at ~99.6%)")
        if worst > 1e-3:
            failures.append(f"L{L}: a flip with golden margin {worst:.2e} > 1e-3 is not a near-tie")
        if not werr <= 1e-3:
            failures.append(f"L{L}: routing-weight error {werr} on agreeing tokens")
    print("[moe] router set agreement vs fp64 on real inputs (D1):\n[moe]   " + "\n[moe]   ".join(rows))
    if tot_n:
        print(f"[moe] router overall: {tot_eq / tot_n * 100:.3f}% of {tot_n} token-layers "
              f"(reference CPU fp32 router: {tot_eq_ref / tot_n * 100:.3f}%), max flip margin {worst_margin:.2e}")
    assert tot_n > 0, "no real layers available"
    assert not failures, "\n".join(failures)


def _decode_blocks(xs, mesh_device):
    """Every token of ``xs [n, 4096]`` as 32-row decode-shape device blocks (the last one zero-padded)."""
    n = xs.shape[0]
    blocks = []
    for c0 in range(0, n, 32):
        blk = torch.zeros(32, H, dtype=torch.bfloat16)
        blk[: min(32, n - c0)] = xs[c0: c0 + 32]
        blocks.append(upload_replicated(blk, mesh_device))
    return blocks


def _route_blocks(router, blocks, mesh_device, n):
    """Route the decode-shape blocks -> (idx [n, 8] long, w [n, 8] fp64) from chip 0."""
    ii, ww = [], []
    for bt in blocks:
        i, w = router(bt)
        ii.append(read_replicated(i, mesh_device, chips=[0])[0].reshape(-1, K).long())
        ww.append(read_replicated(w, mesh_device, chips=[0])[0].reshape(-1, K).double())
        _free([i, w])
    return torch.cat(ii)[:n], torch.cat(ww)[:n]


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_router_exact(mesh_device, device_params):
    """D1(b): the exact-fp32 router logits kernel (``tt/kernels/router_fp32.py``, another agent's) swapped into the
    MoE router at decode shape: top-8 set agreement vs fp64 on every real token of the 11 captured layers (composite
    vs exact), traced router cost, and the layer-2 MoE decode with the exact router."""
    try:
        from models.demos.motif3.tt.kernels.router_fp32 import RouterLogitsFP32
    except Exception as e:  # the kernel is optional (built in parallel)
        pytest.skip(f"exact-fp32 router kernel unavailable: {type(e).__name__}: {e}")
    from models.demos.motif3.tt.moe import MotifMoE, MotifRouter
    from models.demos.motif3.tt.weights import HFWeightLoader, hf_name

    cfg, ccl, _ = _setup(mesh_device, "moe_router_exact")
    data = load_real_inputs()
    src = HFWeightLoader()
    require_layer(src, 2)
    # probe: the kernel is another agent's work in progress; if it cannot build / run here, skip (do not fail)
    try:
        probe = RouterLogitsFP32.from_source(mesh_device, cfg, 2, source=src, cache=False)
        xp = upload_replicated(data["layers"][2]["x"][:32], mesh_device)
        _free(probe(xp))
        _free(xp)
        probe.deallocate()
    except Exception as e:
        pytest.skip(f"exact-fp32 router kernel does not run here: {type(e).__name__}: {str(e)[:300]}")
    rows, failures = [], []
    tot = {"composite": [0, 0], "exact_fp32": [0, 0]}
    for L in data["meta"]["layers"]:
        if not src.layer_available(L):
            rows.append(f"L{L:02d} skipped (weights not local)")
            continue
        xs = data["layers"][L]["x"]
        n = xs.shape[0]
        idx64, _, biased = router_golden(xs.float(), src.get(hf_name(L, "moe.router.gate.weight")),
                                         src.get(hf_name(L, "moe.expert_bias")))
        blocks = _decode_blocks(xs, mesh_device)
        line = f"L{L:02d}"
        for name in ("composite", "exact_fp32"):
            fn = RouterLogitsFP32.from_source(mesh_device, cfg, L, source=src, cache=False) if name != "composite" else None
            router = MotifRouter(mesh_device, cfg, L, source=src, cache=False, logits_fn=fn)
            try:
                got, _ = _route_blocks(router, blocks, mesh_device, n)
                agr, eq, margin = set_agreement(got, idx64, biased)
                tot[name][0] += int(eq.sum())
                tot[name][1] += n
                line += f" {name} {agr * 100:.3f}% ({int((~eq).sum())} flips)"
                if L == 2:
                    x32 = upload_replicated(xs[spread_tokens(n, 32)], mesh_device)
                    t = traced_us(mesh_device, lambda: router(x32), n=32, reps=7)
                    line += f" [router traced {t:.1f} us]"
                    _free(x32)
            except Exception as e:
                line += f" {name} error {type(e).__name__}: {str(e)[:200]}"
                failures.append(line)
            router.deallocate()
        _free(blocks)
        rows.append(line)
    summary = "; ".join(f"{k} {v[0] / max(v[1], 1) * 100:.3f}% of {v[1]}" for k, v in tot.items())
    print("[moe] router agreement vs fp64 (decode-shape calls, real tokens):\n[moe]   " + "\n[moe]   ".join(rows)
          + f"\n[moe]   overall: {summary}")
    # the layer-2 MoE decode with the exact router
    L = 2
    xs = data["layers"][L]["x"]
    x32 = xs[spread_tokens(xs.shape[0], 32)]
    ref = RefMoE(src, L)
    want, _, _ = ref(x32)
    idx64, w64, biased = router_golden(x32.float(), src.get(hf_name(L, "moe.router.gate.weight")),
                                       src.get(hf_name(L, "moe.expert_bias")))
    moe = MotifMoE(mesh_device, cfg, L, source=src, ccl=ccl, cache=False, router_logits="exact_fp32")
    res, _ = _check_decode(moe, x32, cfg, mesh_device, ref, "real L2 exact_fp32 router", want=want, idx_ref=idx64,
                           w_ref=w64, biased_ref=biased)
    _report(res)
    _decode_gates(res, failures)
    x_tt = upload_lanes(x32, cfg, mesh_device)
    t_us = traced_us(mesh_device, lambda: moe.forward_decode(x_tt), n=8, reps=5)
    print(f"[moe] L2 decode with exact_fp32 router: traced {t_us:.0f} us/call")
    _free(x_tt)
    moe.deallocate()
    if tot["exact_fp32"][1] and tot["exact_fp32"][0] < tot["composite"][0]:
        failures.append("exact router agreement below the composite's")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_router_variants(mesh_device, device_params):
    """Router op fusions vs the plain G5 composite on all 2971 real layer-2 tokens, at prefill shape (one 2976-row call)
    and at decode shape (93 calls of 32 rows: the padded top-k and the decode program configs apply only there). Each
    knob and the module default must give identical top-8 sets and weights (|dw| <= 1e-5) at both shapes (review
    issue 7: the identity is now checked on every token, not on 32); costs are traced at decode shape."""
    from models.demos.motif3.tt.model_config import mcast1d_matmul_pc
    from models.demos.motif3.tt.moe import MotifRouter
    from models.demos.motif3.tt.weights import HFWeightLoader, hf_name

    cfg, ccl, _ = _setup(mesh_device, "moe_router_variants")
    src = HFWeightLoader()
    L = 2
    require_layer(src, L)
    data = load_real_inputs()
    xs = data["layers"][L]["x"]
    n = xs.shape[0]
    x32 = xs[spread_tokens(n, 32)]
    x_tt = upload_replicated(x32, mesh_device)
    xall = torch.zeros(-(-n // 32) * 32, H, dtype=torch.bfloat16)
    xall[:n] = xs
    xall_tt = upload_replicated(xall, mesh_device)
    blocks = _decode_blocks(xs, mesh_device)
    router = MotifRouter(mesh_device, cfg, L, source=src, cache=False)
    idx64, _, biased64 = router_golden(xs.float(), src.get(hf_name(L, "moe.router.gate.weight")),
                                       src.get(hf_name(L, "moe.expert_bias")))

    def ordered(a, b):
        o = torch.argsort(a, dim=-1)  # order-free comparison (sorted=False returns the set in any order)
        return a.gather(-1, o), b.gather(-1, o)

    def run_prefill_shape():
        i, w = router(xall_tt)
        a, _ = read_replicated(i, mesh_device, chips=[0])
        b, _ = read_replicated(w, mesh_device, chips=[0])
        _free([i, w])
        return ordered(a.reshape(-1, K)[:n].long(), b.reshape(-1, K)[:n].double())

    def run_decode_shape():
        return ordered(*_route_blocks(router, blocks, mesh_device, n))

    pads = dict(router._pads)
    default_pc = router.decode_pc

    def configure(sorted_=True, sig=False, sig_pc=False, norm=False, pad=False, pc=False):
        router.topk_sorted, router.fuse_sigmoid, router.sigmoid_in_pc = sorted_, sig, sig_pc
        router.fuse_normalize = norm
        router._pads = dict(pads) if pad else {}
        router.decode_pc = default_pc if pc else None

    configure()
    base_p = run_prefill_shape()
    base_d = run_decode_shape()
    base_agr = set_agreement(base_d[0], idx64, biased64)[0]
    rows, failures = [], []
    for name, kw in [
        ("plain composite", {}),
        ("topk sorted=False", {"sorted_": False}),
        ("linear activation='sigmoid' (a separate unary_chain op in ttnn)", {"sig": True}),
        ("sigmoid as the decode program config's fused_activation", {"sig_pc": True, "pc": True}),
        ("fused normalize", {"norm": True}),
        ("topk width padded to 1024 (decode)", {"pad": True}),
        ("12-core linear program config (decode)", {"pc": True}),
        ("module default", {"sig": True, "sig_pc": True, "norm": True, "pad": True, "pc": True}),
    ]:
        configure(**kw)
        try:
            ip, wp = run_prefill_shape()
            idd, wd = run_decode_shape()
            same_p = float((ip == base_p[0]).all(-1).double().mean())
            same_d = float((idd == base_d[0]).all(-1).double().mean())
            eq_p = (ip == base_p[0]).all(-1)
            eq_d = (idd == base_d[0]).all(-1)
            wdiff = max(_maxabs((wp - base_p[1])[eq_p]), _maxabs((wd - base_d[1])[eq_d]))
            agr = set_agreement(idd, idx64, biased64)[0]
            t = traced_us(mesh_device, lambda: router(x_tt), n=32, reps=7)
            rows.append(f"{name}: traced {t:.1f} us/call; sets equal to plain on {same_p * 100:.3f}% (prefill shape) / "
                        f"{same_d * 100:.3f}% (decode shape) of {n} tokens, max |dw| {wdiff:.2e}; decode-shape "
                        f"agreement vs fp64 {agr * 100:.3f}% (plain {base_agr * 100:.3f}%)")
            if same_p < 1.0 or same_d < 1.0 or wdiff > 1e-5:
                rows[-1] += "  (NOT identical)"
                failures.append(rows[-1])
        except Exception as e:
            rows.append(f"{name}: error {type(e).__name__}: {str(e)[:300]}")
            failures.append(rows[-1])
    configure(True, True, True, True, True, True)
    # router linear program configs (decode shape; sigmoid separate so the configs are comparable)
    router.sigmoid_in_pc = False
    for name, grid, bw in [("auto", None, None), ("mcast1d 12x1 k8", (12, 1), 8), ("mcast1d 12x1 k16", (12, 1), 16),
                           ("mcast1d 12x1 k32 (module)", (12, 1), 32), ("mcast1d 6x1 k16", (6, 1), 16)]:
        router.decode_pc = None if grid is None else mcast1d_matmul_pc(grid, E // 32, bw, H // 32)
        try:
            t_lin = traced_us(mesh_device, lambda: router.route_logits(x_tt), n=32, reps=7)
            t_all = traced_us(mesh_device, lambda: router(x_tt), n=32, reps=7)
            idd, wd = run_decode_shape()
            same = bool(torch.equal(idd, base_d[0])) and float((wd - base_d[1]).abs().max()) <= 1e-6
            agr = set_agreement(idd, idx64, biased64)[0]
            rows.append(f"linear pc {name}: linear {t_lin:.1f} us, full router {t_all:.1f} us; decode-shape routes "
                        f"identical to plain on all {n} tokens: {same}; set agreement vs fp64 {agr * 100:.3f}%")
        except Exception as e:
            rows.append(f"linear pc {name}: error {type(e).__name__}: {str(e)[:300]}")
            if "module" in name:
                failures.append(rows[-1])
    router.decode_pc = default_pc
    router.sigmoid_in_pc = True
    # the pieces of the plain composite (traced, decode shape; small ops adapt n to clear the sync floor)
    lg = router.route_logits(x_tt)
    sc = ttnn.sigmoid(lg)
    bi = ttnn.add(sc, router.bias)
    _, ix = ttnn.topk(bi, k=K, dim=-1, largest=True, sorted=True)
    wg = ttnn.gather(sc, -1, ix)
    pieces = {
        "linear fp32-out": lambda: router.route_logits(x_tt),
        "sigmoid": lambda: ttnn.sigmoid(lg),
        "add bias": lambda: ttnn.add(sc, router.bias),
        "topk(8) of 384": lambda: ttnn.topk(bi, k=K, dim=-1, largest=True, sorted=True),
        "gather": lambda: ttnn.gather(sc, -1, ix),
        "sum(8) fp32": lambda: ttnn.sum(wg, dim=-1, keepdim=True, compute_kernel_config=router.ckc_eltwise),
    }
    for name, fn in pieces.items():
        try:
            rows.append(f"piece {name}: traced {fmt_traced(traced_stats(mesh_device, fn, n=32, reps=7, adapt_to=128))}"
                        " us")
        except Exception as e:
            rows.append(f"piece {name}: error {type(e).__name__}: {str(e)[:200]}")
    _free([lg, sc, bi, ix, wg, x_tt, xall_tt, blocks])
    router.deallocate()
    print("[moe] router variants (L2 real inputs):\n[moe]   " + "\n[moe]   ".join(rows))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_polynorm_variants(mesh_device, device_params):
    """Grouped PolyNorm (decode, ``[1,12,32,2560]`` from the real layer-2 gate_up of real inputs) in fp32 / bf16
    intermediates, DRAM vs L1 intermediates: accuracy vs an fp64 golden on the same device gate_up values (asserted:
    fp32 PCC >= 0.99999 and every token >= 0.9999; bf16 PCC >= 0.9999 and every token >= 0.999) and traced cost;
    plus the multi-core topk (width padded to 1024; the sets must equal the width-384 sets)."""
    from models.demos.motif3.tt.moe import MotifMoE
    from models.demos.motif3.tt.weights import HFWeightLoader, expert_polynorm_tensors, hf_name, shard_for_device

    cfg, ccl, _ = _setup(mesh_device, "moe_polynorm_variants")
    src = HFWeightLoader()
    L = 2
    require_layer(src, L)
    data = load_real_inputs()
    xs = data["layers"][L]["x"]
    x32 = xs[spread_tokens(xs.shape[0], 32)]
    moe = MotifMoE(mesh_device, cfg, L, source=src, ccl=ccl, cache=False)
    f_tt = upload_replicated(x32, mesh_device)
    x12 = ttnn.repeat(f_tt, ttnn.Shape([1, moe.e_loc, 1, 1]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gu = ttnn.matmul(x12, moe.w_gate_up, program_config=moe.pc_gate_up, compute_kernel_config=moe.ckc_experts,
                     dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gu_h = ttnn.to_torch(ttnn.get_device_tensors(gu)[0]).double().reshape(12, 32, 2 * I)
    pn = expert_polynorm_tensors(src.get(hf_name(L, "moe.experts.act_fn.weight")),
                                 src.get(hf_name(L, "moe.experts.act_fn.bias")), cfg)
    r0, c0_ = cfg.axes.coord(0, 0)
    cst = {k: shard_for_device(v, cfg.axes, r0, c0_, dp_dim=0, tp_dim=1).double().reshape(12, 1, 1)
           for k, v in pn.items()}
    g, u = gu_h[..., :I], gu_h[..., I:]
    nrm = lambda z: z / torch.sqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)  # noqa: E731
    want = (cst["c0"] * nrm(g**3) + cst["c1"] * nrm(g**2) + cst["c2"] * nrm(g) + cst["b"]) * u
    gu_bf = ttnn.typecast(gu, ttnn.bfloat16)
    L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    rows, failures = [], []
    thr = {"fp32": (0.99999, 0.9999), "bf16": (0.9999, 0.999)}
    for impl, mode, mc in (("horner", "fp32", L1), ("rms", "fp32", L1), ("local", "fp32", L1), ("horner", "fp32", DRAM),
                           ("horner", "bf16", L1), ("rms", "bf16", L1), ("local", "bf16", L1)):
        inp = gu if mode == "fp32" else gu_bf
        fn = lambda: moe.polynorm(inp, mode=mode, memory_config=mc, impl=impl)  # noqa: E731
        tag = f"polynorm {impl} {mode} intermediates in {'L1' if mc == L1 else 'DRAM'}"
        try:
            h = fn()
            got = ttnn.to_torch(ttnn.get_device_tensors(h)[0]).double().reshape(12, 32, I)
            _free(h)
            s = stats(want.reshape(-1, I), got.reshape(-1, I))
            t = traced_us(mesh_device, fn, n=8, reps=5)
            rows.append(f"{tag}: traced {t:.1f} us; {fmt(s)}")
            if s["pcc"] < thr[mode][0] or s["min_token_pcc"] < thr[mode][1]:
                failures.append(f"{tag}: pcc {s['pcc']:.7f} / min token {s['min_token_pcc']:.7f} below {thr[mode]}")
        except Exception as e:
            rows.append(f"{tag}: error {type(e).__name__}: {str(e)[:300]}")
            failures.append(rows[-1])
    # multi-core topk: biased scores padded to width 1024 with -inf (the op's multi-core path needs a power of two)
    lg = moe.router.route_logits(f_tt)
    sc = ttnn.sigmoid(lg)
    bi = ttnn.add(sc, moe.router.bias)
    pad = ttnn.from_torch(torch.full((1, 1, 32, 1024 - E), float("-inf")), dtype=ttnn.float32,
                          layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    for name, fn in [
        ("topk 384 sorted", lambda: ttnn.topk(bi, k=K, dim=-1, largest=True, sorted=True)),
        ("concat to 1024 + topk", lambda: ttnn.topk(ttnn.concat([bi, pad], dim=-1), k=K, dim=-1, largest=True,
                                                     sorted=True)),
    ]:
        try:
            rows.append(f"{name}: traced {traced_us(mesh_device, fn, n=32, reps=7):.1f} us")
        except Exception as e:
            rows.append(f"{name}: error {type(e).__name__}: {str(e)[:300]}")
    try:
        _, i1 = ttnn.topk(bi, k=K, dim=-1, largest=True, sorted=True)
        wide = ttnn.concat([bi, pad], dim=-1)
        _, i2 = ttnn.topk(wide, k=K, dim=-1, largest=True, sorted=True)
        a = ttnn.to_torch(ttnn.get_device_tensors(i1)[0]).long().reshape(-1, K).sort(-1).values
        b = ttnn.to_torch(ttnn.get_device_tensors(i2)[0]).long().reshape(-1, K).sort(-1).values
        same = bool(torch.equal(a, b))
        rows.append(f"topk 1024-padded sets equal to width-384 sets (32 tokens; all 2971 in router_variants): {same}")
        if not same:
            failures.append(rows[-1])
        _free([i1, i2, wide])
    except Exception as e:
        rows.append(f"topk compare: error {type(e).__name__}: {str(e)[:300]}")
        failures.append(rows[-1])
    _free([lg, sc, bi, pad, f_tt, x12, gu, gu_bf])
    moe.deallocate()
    print("[moe] polynorm / topk variants:\n[moe]   " + "\n[moe]   ".join(rows))
    assert not failures, "\n".join(failures)


def _decode_contracts(moe, x32, got, cfg, mesh_device, failures, tag):
    """``reduce_tp=False`` returns the column partial (sum over TP = output); ``add_partial`` is summed over TP before
    AR(tp); the caller's tensors stay allocated."""
    from models.demos.motif3.tt.ccl import device_tensors_to_torch

    R, C = cfg.axes.mesh_shape
    Lr = cfg.lanes_per_row
    x_tt = upload_lanes(x32, cfg, mesh_device)
    part = moe.forward_decode(x_tt, reduce_tp=False)
    pfull = device_tensors_to_torch(part, mesh_device).float()  # [R, C, 1, 1, Lr, 4096]
    psum = torch.cat([sum(pfull[cfg.axes.coord(dp, tp)] for tp in range(cfg.tp)).reshape(-1, H)
                      for dp in range(cfg.dp)])
    s_part = stats(got, psum)
    g = torch.Generator().manual_seed(9)
    extra = (0.05 * torch.randn(R, C, 1, Lr, H, generator=g)).bfloat16()
    ex_tt = upload_per_chip(extra.reshape(R, C, Lr, H), mesh_device)
    out2 = moe.forward_decode(x_tt, add_partial=ex_tt)
    got2, same2 = read_rows(out2, cfg, mesh_device)
    exp2 = got + torch.cat([sum(extra[cfg.axes.coord(dp, tp)].float() for tp in range(cfg.tp)).reshape(-1, H)
                            for dp in range(cfg.dp)])
    s_add = stats(exp2, got2)
    alive = x_tt.is_allocated() and ex_tt.is_allocated()
    print(f"[moe] {tag} contracts: sum_tp(reduce_tp=False partial) vs output: {fmt(s_part)}; add_partial: {fmt(s_add)} "
          f"replicas={same2}; inputs still allocated {alive}")
    if s_part["pcc"] < 0.9999 or s_add["pcc"] < 0.9999 or not same2 or not alive:
        failures.append(f"{tag} contracts: partial {s_part['pcc']:.6f} add {s_add['pcc']:.6f} same {same2} "
                        f"alive {alive}")
    _free([x_tt, part, ex_tt, out2])


def _prefill_random_checks(moe, ref, cfg, mesh_device, failures, tag, S=512, seed=7):
    """Random-weight prefill: accuracy (reference routes + device routes), replicas, add_partial / reduce_tp=False."""
    from models.demos.motif3.tt.ccl import device_tensors_to_torch

    x = random_inputs(S, seed=seed)
    want, _, _ = ref(x)
    x_tt = upload_replicated(x, mesh_device)
    out = moe.forward_prefill(x_tt)
    got, same = read_replicated(out, mesh_device)
    got = got.reshape(S, H).float()
    routes = prefill_device_routes(moe, x_tt)
    tok = torch.arange(S)
    _check_prefill(moe, ref, x, tok, got, want, routes, failures, f"{tag} prefill S={S}")
    R, C = cfg.axes.mesh_shape
    g = torch.Generator().manual_seed(3)
    extra = (0.05 * torch.randn(R, C, S // cfg.dp, H, generator=g)).bfloat16()
    ex_tt = upload_per_chip(extra, mesh_device)
    out2 = moe.forward_prefill(x_tt, add_partial=ex_tt)
    got2, same2 = read_replicated(out2, mesh_device)
    add = torch.cat([sum(extra[cfg.axes.coord(dp, t)].float() for t in range(cfg.tp)) for dp in range(cfg.dp)])
    s2 = stats(got + add, got2.reshape(S, H).float())
    part = moe.forward_prefill(x_tt, reduce_tp=False)
    pf = device_tensors_to_torch(part, mesh_device).float().reshape(R, C, S // cfg.dp, H)
    psum = torch.cat([sum(pf[cfg.axes.coord(dp, t)] for t in range(cfg.tp)) for dp in range(cfg.dp)])
    s3 = stats(got, psum)
    alive = x_tt.is_allocated() and ex_tt.is_allocated()
    print(f"[moe] {tag} prefill S={S}: identical on all chips {same}; add_partial {fmt(s2)} replicas {same2}; "
          f"reduce_tp=False sum {fmt(s3)}; inputs still allocated {alive}")
    if not (same and same2 and alive) or s2["pcc"] < 0.9999 or s3["pcc"] < 0.9999:
        failures.append(f"{tag} prefill contracts: same {same}/{same2} alive {alive} add {s2['pcc']:.6f} "
                        f"part {s3['pcc']:.6f}")
    _free([x_tt, out, ex_tt, out2, part])


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_decode_random(mesh_device, device_params):
    """Random weights at real dims (384 experts): bfp8 (production) and bf16 (fp32-faithful check) experts."""
    from models.demos.motif3.tt.moe import MotifMoE
    from models.demos.motif3.tt.weights import DictWeightSource

    cfg, ccl, _ = _setup(mesh_device, "moe_decode_random")
    L = 2
    t0 = time.time()
    tensors = random_moe_tensors(L, seed=0)
    src = DictWeightSource(tensors)
    ref = RefMoE(src, L)
    x32 = random_inputs(32, seed=1)
    want, idx_ref, w_ref = ref(x32)
    idx64, w64, biased = router_golden(x32.float(), tensors[f"model.layers.{L}.moe.router.gate.weight"],
                                       tensors[f"model.layers.{L}.moe.expert_bias"])
    print(f"[moe] random weights + reference: {time.time() - t0:.1f} s")
    failures = []
    for edt, thr in ((ttnn.bfloat8_b, 0.995), (ttnn.bfloat16, 0.999)):
        t0 = time.time()
        moe = MotifMoE(mesh_device, cfg, L, source=src, ccl=ccl, cache=False, experts_dtype=edt)
        t_load = time.time() - t0
        res, got = _check_decode(moe, x32, cfg, mesh_device, ref, f"random/{edt.name}", want=want, idx_ref=idx64,
                                 w_ref=w64, biased_ref=biased)
        res["load_s"] = round(t_load, 1)
        _report(res)
        _decode_gates(res, failures, pcc_thr=thr)
        if edt == ttnn.bfloat8_b:
            _decode_contracts(moe, x32, got, cfg, mesh_device, failures, "random/bfp8")
            # lane independence (inactive lanes): zero one lane's input -> every other lane's output is bitwise equal
            x_z = x32.clone()
            x_z[5] = 0
            xz_tt = upload_lanes(x_z, cfg, mesh_device)
            out_z = moe.forward_decode(xz_tt)
            got_z, _ = read_rows(out_z, cfg, mesh_device)
            others = [i for i in range(32) if i != 5]
            indep = bool(torch.equal(got_z[others], got[others])) and bool(torch.isfinite(got_z).all())
            print(f"[moe] lane independence (lane 5 zeroed): other 31 lanes bitwise equal {indep}; zero lane "
                  f"max |out| {float(got_z[5].abs().max()):.3e}")
            if not indep:
                failures.append("zeroing one lane changed other lanes' outputs (or produced non-finite values)")
            _free([xz_tt, out_z])
        moe.deallocate()
    assert not failures, "\n".join(failures)


def _device_params_8x4():
    from models.demos.motif3.tt.model_config import device_params

    return device_params("FABRIC_2D_TORUS_XY", TRACE)


@pytest.mark.parametrize("mesh_device, device_params", [pytest.param((8, 4), _device_params_8x4(), id="8x4-torus2d")],
                         indirect=True)
@torch.no_grad()
def test_moe_device_decode_random_8x4(mesh_device, device_params):
    """(8, 4) mesh (the plugin's BH-Galaxy preset): the TP axis is cluster_axis 0; the same module (role-based axes,
    EP placement through ``weights.ep_layout`` + ``as_tensor(dp_dim=0, tp_dim=1)``) must match the reference in decode
    and prefill (RS(dp) / AR(tp) / AG(dp) ordering, ``add_partial`` / ``reduce_tp=False``)."""
    from models.demos.motif3.tt.moe import MotifMoE
    from models.demos.motif3.tt.weights import DictWeightSource

    cfg, ccl, _ = _setup(mesh_device, "moe_decode_random_8x4")
    assert cfg.axes.tp_axis == 0 and cfg.axes.dp_axis == 1, cfg.axes
    L = 2
    tensors = random_moe_tensors(L, seed=0)
    src = DictWeightSource(tensors)
    ref = RefMoE(src, L)
    x32 = random_inputs(32, seed=1)
    want, _, _ = ref(x32)
    idx64, w64, biased = router_golden(x32.float(), tensors[f"model.layers.{L}.moe.router.gate.weight"],
                                       tensors[f"model.layers.{L}.moe.expert_bias"])
    moe = MotifMoE(mesh_device, cfg, L, source=src, ccl=ccl, cache=False)
    res, _ = _check_decode(moe, x32, cfg, mesh_device, ref, "random/8x4/bfp8", want=want, idx_ref=idx64, w_ref=w64,
                           biased_ref=biased)
    _report(res)
    failures = []
    _decode_gates(res, failures)
    _prefill_random_checks(moe, ref, cfg, mesh_device, failures, "random/8x4/bfp8")
    moe.deallocate()
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_submesh_1x8(mesh_device, device_params):
    """Review issue 1 on silicon: the module on a (1, 8) submesh (size-1 DP axis: ``ag_dp_rows`` / ``ar_dp`` /
    ``partition`` / ``rs_dp`` / ``ag_dp`` hand their input back; 48 experts per chip, 32 lanes in the one row). Decode
    and prefill (two chunks) match the reference, and the caller's inputs stay allocated."""
    from models.demos.motif3.tt.ccl import MotifCCL, log_fabric
    from models.demos.motif3.tt.model_config import MotifTTConfig
    from models.demos.motif3.tt.moe import MotifMoE
    from models.demos.motif3.tt.weights import DictWeightSource

    sub = mesh_device.create_submesh(ttnn.MeshShape(1, 8))
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=sub)
    log_fabric(sub, "moe_submesh_1x8")
    assert cfg.dp == 1 and cfg.tp == 8 and cfg.experts_per_chip == 48, cfg.describe()
    ccl = MotifCCL(sub, cfg)
    L = 2
    tensors = random_moe_tensors(L, seed=0)
    src = DictWeightSource(tensors)
    ref = RefMoE(src, L)
    x32 = random_inputs(32, seed=1)
    want, _, _ = ref(x32)
    idx64, w64, biased = router_golden(x32.float(), tensors[f"model.layers.{L}.moe.router.gate.weight"],
                                       tensors[f"model.layers.{L}.moe.expert_bias"])
    moe = MotifMoE(sub, cfg, L, source=src, ccl=ccl, cache=False)
    failures = []
    res, got = _check_decode(moe, x32, cfg, sub, ref, "random/1x8 submesh/bfp8", want=want, idx_ref=idx64, w_ref=w64,
                             biased_ref=biased)
    _report(res)
    _decode_gates(res, failures)
    _decode_contracts(moe, x32, got, cfg, sub, failures, "random/1x8")
    moe.prefill_chunk = 256  # two chunks at S = 512: the chunk slices are freed, x is not
    _prefill_random_checks(moe, ref, cfg, sub, failures, "random/1x8", S=512)
    moe.deallocate()
    assert not failures, "\n".join(failures)


def _breakdown(mesh_device, moe, x_tt, reps=5):
    """Traced per-stage cost of one decode MoE call (each stage timed alone on fixed inputs, the module's path and
    memory placement). Small stages double ``n`` (up to 128) until their ``n/2`` calls clear the sync floor; a stage
    that stays under it reports the ``t(n)/n`` upper bound (review issue 8)."""
    mc = moe.decode_mc
    ccl = moe.ccl
    fold = moe.combine_mode == "fold"
    f_all = ccl.ag_dp_rows(x_tt, memory_config=mc)
    idx, w = moe.router(f_all, scale=moe.internal_route_scale, memory_config=mc)
    w_loc = moe.local_weights(idx, w, memory_config=mc)
    x12 = ttnn.repeat(f_all, ttnn.Shape([1, moe.e_loc, 1, 1]), memory_config=mc)
    gu_dtype = ttnn.float32 if moe.decode_polynorm == "fp32" else ttnn.bfloat16
    mm_gu = lambda: ttnn.matmul(x12, moe.w_gate_up, program_config=moe.pc_gate_up,  # noqa: E731
                                compute_kernel_config=moe.ckc_experts, dtype=gu_dtype, memory_config=mc)
    gu = mm_gu()
    pn = lambda: moe.polynorm(gu, mode=moe.decode_polynorm, row_scale=w_loc if fold else None,  # noqa: E731
                              memory_config=mc)
    h = pn()
    mm_dn = lambda: ttnn.matmul(h, moe.w_down, program_config=moe.pc_down,  # noqa: E731
                                compute_kernel_config=moe.ckc_experts, dtype=moe.down_dtype, memory_config=mc)
    y = mm_dn()
    comb = (lambda: moe.reduce_experts(y)) if fold else (lambda: moe.combine(y, w_loc))  # noqa: E731
    part = comb()
    red = ccl.ar_dp(part)
    mine = ccl.partition(red, 2, "dp")
    where = "L1" if mc == ttnn.L1_MEMORY_CONFIG else "DRAM"
    stages = [
        ("gather ag_dp_rows", lambda: ccl.ag_dp_rows(x_tt, memory_config=mc), 16),
        ("router", lambda: moe.router(f_all, scale=moe.internal_route_scale, memory_config=mc), 16),
        ("local mask", lambda: moe.local_weights(idx, w, memory_config=mc), 32),
        ("repeat x12", lambda: ttnn.repeat(f_all, ttnn.Shape([1, moe.e_loc, 1, 1]), memory_config=mc), 32),
        ("gate_up matmul", mm_gu, 8),
        (f"polynorm {moe.polynorm_impl} {moe.decode_polynorm}{' +w' if fold else ''} ({where})", pn, 8),
        ("down matmul", mm_dn, 8),
        ("combine " + ("fast_reduce_nc" if fold else "multiply+sum"), comb, 32),
        ("ar_dp", lambda: ccl.ar_dp(part), 16),
        ("partition dp", lambda: ccl.partition(red, 2, "dp"), 32),
        ("ar_tp", lambda: ccl.ar_tp(mine), 16),
    ]
    out = {}
    for name, fn, nn_ in stages:
        try:
            out[name] = traced_stats(mesh_device, fn, n=nn_, reps=reps, adapt_to=128)
        except Exception as e:  # keep going: report what works
            out[name] = f"error {type(e).__name__}: {str(e)[:200]}"
    _free([f_all, idx, w, w_loc, x12, gu, h, y, part, red, mine])
    return out


@pytest.mark.parametrize("layer", [2, 4], ids=["L2", "L4"])
@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_decode_real(mesh_device, device_params, layer):
    """Real layer weights + real router inputs: accuracy gates, variants, trace replay, latency."""
    from models.demos.motif3.tt.moe import MotifMoE
    from models.demos.motif3.tt.weights import HFWeightLoader, hf_name

    cfg, ccl, fab = _setup(mesh_device, f"moe_decode_real_L{layer}")
    src = HFWeightLoader()
    require_layer(src, layer)
    data = load_real_inputs()
    d = data["layers"][layer]
    n = d["x"].shape[0]
    sel = spread_tokens(n, 32)
    x32 = d["x"][sel]
    t0 = time.time()
    ref = RefMoE(src, layer)
    want, idx_ref, w_ref = ref(x32)
    idx64, w64, biased = router_golden(x32.float(), src.get(hf_name(layer, "moe.router.gate.weight")),
                                       src.get(hf_name(layer, "moe.expert_bias")))
    t_ref = time.time() - t0
    t0 = time.time()
    moe = MotifMoE(mesh_device, cfg, layer, source=src, ccl=ccl, cache=False)
    t_load = time.time() - t0
    print(f"[moe] L{layer}: reference {t_ref:.1f} s, module load (no TT cache) {t_load:.1f} s, "
          f"{moe.expert_weight_bytes_per_chip / 1e6:.1f} MB experts per chip")
    failures = []
    kw = dict(want=want, idx_ref=idx64, w_ref=w64, biased_ref=biased)
    res, got = _check_decode(moe, x32, cfg, mesh_device, ref, f"real L{layer} default (bfp8, polynorm fp32, combine "
                             "bf16)", **kw)
    _report(res)
    _decode_gates(res, failures)

    def part_info(taps):
        """Is the per-chip partial a true fp32 sum? A chip's partial row is the sum of the token's experts *on that
        chip*: usually none (exactly 0) or one (one bf16 term, exactly bf16), so only rows with >= 2 local experts can
        hold non-bf16 values. Checked on those rows, over all 32 chips."""
        from models.demos.motif3.tt.ccl import device_tensors_to_torch

        R, C = cfg.axes.mesh_shape
        p = device_tensors_to_torch(taps["part"], mesh_device).float().reshape(R, C, -1, H)  # [R, C, 32, 4096]
        wl = device_tensors_to_torch(taps["w_loc"], mesh_device).float().reshape(R, C, moe.e_loc, -1)  # [R, C, 12, 32]
        multi = (wl != 0).sum(2) >= 2  # [R, C, 32]
        rows = p[multi]
        return {"part_dtype": str(taps["part"].dtype).split(".")[-1], "multi_expert_rows": int(multi.sum()),
                "part_not_bf16_multi": round(frac_not_bf16(rows), 3) if rows.numel() else None}

    # variants (same weights): PolyNorm bf16 intermediates (the 6.6 ms/step lever), fp32 partials end to end, the
    # literal HF combine order (multiply + sum), the other PolyNorm impls, DRAM intermediates (decode_l1=False)
    default = {"decode_polynorm": "fp32", "combine_dtype": ttnn.bfloat16, "combine_mode": "fold",
               "polynorm_impl": "horner", "decode_mc": ttnn.L1_MEMORY_CONFIG}
    for vname, attrs in (
        ("polynorm bf16", {"decode_polynorm": "bf16"}),
        ("fp32 partials (combine_dtype fp32)", {"combine_dtype": ttnn.float32}),
        ("HF-order combine (multiply_sum)", {"combine_mode": "multiply_sum"}),
        ("polynorm impl rms", {"polynorm_impl": "rms"}),
        ("polynorm impl local", {"polynorm_impl": "local"}),
        ("DRAM intermediates (decode_l1=False)", {"decode_mc": ttnn.DRAM_MEMORY_CONFIG}),
    ):
        for k, v in attrs.items():
            setattr(moe, k, v)
        try:
            r2, got2 = _check_decode(moe, x32, cfg, mesh_device, ref, f"real L{layer} {vname}", inspect=part_info, **kw)
            r2["bitwise_equal_default"] = bool(torch.equal(got2, got))
            _report(r2)
            _decode_gates(r2, failures)
            if "fp32 partials" in vname and (r2["part_dtype"] != "FLOAT32" or not r2["multi_expert_rows"]
                                             or r2["part_not_bf16_multi"] < 0.5):
                failures.append(f"{vname}: the per-chip partial is not a true fp32 sum ({r2['part_dtype']}; on the "
                                f"{r2['multi_expert_rows']} rows with >= 2 local experts {r2['part_not_bf16_multi']} of "
                                "values are not bf16-representable)")
            if "DRAM" in vname and not r2["bitwise_equal_default"]:
                print(f"[moe] note: {vname} is not bitwise equal to the L1 default")
        except Exception as e:
            failures.append(f"variant {vname}: {type(e).__name__}: {str(e)[:300]}")
        for k, v in default.items():
            setattr(moe, k, v)

    # fp32-faithful real-weight check (non-default fold_route_scale=False, bf16 experts): PCC >= 0.999
    if layer == 2:
        moe16 = MotifMoE(mesh_device, cfg, layer, source=src, ccl=ccl, cache=False, experts_dtype=ttnn.bfloat16,
                         fold_route_scale=False)
        r16, _ = _check_decode(moe16, x32, cfg, mesh_device, ref, f"real L{layer} bf16 experts, fold_route_scale=False",
                               **kw)
        _report(r16)
        _decode_gates(r16, failures, pcc_thr=0.999)
        moe16.deallocate()

    # trace: capture once, replay with new inputs (persistent input tensor), compare with eager
    x_tt = upload_lanes(x32, cfg, mesh_device)
    out_e = moe.forward_decode(x_tt)
    _free(out_e)
    with _Capture(mesh_device) as cap:
        out_t = moe.forward_decode(x_tt)
    try:
        for it, s0 in enumerate((1, 2)):
            sel2 = spread_tokens(n - s0, 32) + s0
            x_new = d["x"][sel2]
            ttnn.copy_host_to_device_tensor(upload_lanes(x_new, cfg, mesh_device, device=False), x_tt)
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            got_t, same_t = read_rows(out_t, cfg, mesh_device)
            out_e2 = moe.forward_decode(x_tt)
            got_e2, _ = read_rows(out_e2, cfg, mesh_device)
            _free(out_e2)
            taps2 = {}
            out_e3 = moe.forward_decode(x_tt, taps=taps2)  # the routes of the replayed inputs
            idx2, w2, _ = read_routes(taps2, moe, mesh_device, chips=[0])  # gather or scatter (A5) router path
            w2 = w2.double() * (moe.route_scale / moe.internal_route_scale)
            _free([taps2, out_e3])
            want2, _, _ = ref(x_new)
            want2_dev, _, _ = ref(x_new, idx2, w2.float())
            s_t = stats(want2, got_t)
            s_td = stats(want2_dev, got_t)
            exact = bool(torch.equal(got_t, got_e2))
            print(f"[moe] L{layer} trace replay {it}: vs ref {fmt(s_t)}; on device routes {fmt(s_td)}; traced == eager "
                  f"bitwise: {exact}; replicas {same_t}")
            if s_t["pcc"] < 0.995 or s_td["min_token_pcc"] < TOKEN_PCC_MIN or not same_t:
                failures.append(f"trace replay {it}: {fmt(s_t)} / device routes {fmt(s_td)} same={same_t}")
            if not exact:
                failures.append(f"trace replay {it}: traced output differs from eager")
        t0 = time.perf_counter()
        for _ in range(10):
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        replay_us = (time.perf_counter() - t0) / 10 * 1e6
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    _free(out_t)

    # latency: eager, traced (slope), per-stage breakdown, effective expert-weight bandwidth
    fn = lambda: moe.forward_decode(x_tt)  # noqa: E731
    e_us = eager_us(mesh_device, fn, iters=5)
    t_us = traced_us(mesh_device, fn, n=8, reps=5)
    lat_var = {}
    for vname, obj, attr, val in (("polynorm bf16", moe, "decode_polynorm", "bf16"),
                                  ("fp32 partials", moe, "combine_dtype", ttnn.float32),
                                  ("router sigmoid as a separate op", moe.router, "sigmoid_in_pc", False),
                                  ("DRAM intermediates", moe, "decode_mc", ttnn.DRAM_MEMORY_CONFIG)):
        old = getattr(obj, attr)
        setattr(obj, attr, val)
        try:
            lat_var[vname] = traced_us(mesh_device, fn, n=8, reps=5)
        except Exception as e:
            lat_var[vname] = f"error {type(e).__name__}: {str(e)[:200]}"
        setattr(obj, attr, old)
    print(f"[moe] L{layer} traced decode variants (us/call): default {t_us:.0f}; " + "; ".join(
        f"{k} {v:.0f}" if isinstance(v, float) else f"{k} {v}" for k, v in lat_var.items()))
    bd = _breakdown(mesh_device, moe, x_tt)
    mm = [bd.get("gate_up matmul"), bd.get("down matmul")]
    bw = (moe.expert_weight_bytes_per_chip / (sum(m["us"] for m in mm) * 1e-6) / 1e9
          if all(isinstance(m, dict) for m in mm) else None)
    print(f"[moe] L{layer} decode latency: eager {e_us:.0f} us/call, traced {t_us:.0f} us/call (slope), single "
          f"replay+sync {replay_us:.0f} us; fabric committed {fab.get('committed')} (dp_ring={fab.get('dp_ring')})")
    print(f"[moe] L{layer} traced breakdown (us): " + ", ".join(f"{k} {fmt_traced(v)}" for k, v in bd.items()))
    tot = sum(v["us"] for v in bd.values() if isinstance(v, dict))
    print(f"[moe] L{layer} breakdown sum {tot:.0f} us vs the full call {t_us:.0f} us")
    if bw is not None:
        print(f"[moe] L{layer} expert weights {moe.expert_weight_bytes_per_chip / 1e6:.1f} MB/chip in "
              f"{sum(m['us'] for m in mm):.0f} us -> {bw:.0f} GB/s effective")
    _free(x_tt)
    moe.deallocate()
    assert not failures, "\n".join(failures)


# ============================================================================================================
# T64: the 64-row verify step (docs/p5_t64/P5_T64_DESIGN.md §4.2, T4; R-E7)
# ============================================================================================================
class _RaisingSource:
    """A weight source that must never be read: the T64 tests load every tensor from the serving TT cache."""

    def _fail(self, *a, **k):
        raise AssertionError(f"the BF16 source was read: {a}")

    get = get_rows = shape = available = has = keys = layer_available = _fail

    def __contains__(self, name):
        self._fail(name)


def t64_cfg(mesh_device, **kw):
    """The mesh's config with the T64 step staged (``spec_tokens=1``, ``spec_verify="wide"``; ``ring_gather`` "safe",
    the default): modules built with it allocate their 64-row constants in their constructors (F3N rule R3)."""
    from models.demos.motif3.tt.model_config import MotifTTConfig

    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device, spec_tokens=1, spec_verify="wide", **kw)
    assert cfg.wide_rows_per_dp == 16 and cfg.dp * cfg.wide_rows_per_dp == 64 and cfg.ring_gather == "safe"
    return cfg


def l1_pin(mesh_device):
    """A one-page L1 tensor allocated before any new program runs. L1 buffers are allocated top-down, so it sits at the
    top of main L1, just below L1_SMALL (the CCL semaphores), and a program whose static circular buffers reach it fails
    with tt-metal's "clash" error: the static-CB-end check of every new T64 program (P5_T64_DESIGN.md §0.3 item 7; the
    method of track B's ``probe_sp1_cbend.py``). Free it at the end of the test."""
    pin = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                          memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    try:
        print(f"[moe] L1 pin page at {pin.buffer_address()} B (static CBs of every program below must end under it)")
    except Exception:  # report only
        pass
    return pin


def b0_module(name: str):
    """``tt/<name>.py`` as committed at git ``HEAD`` (B0), imported under a private name with the package context of
    the current modules (its relative imports resolve to the current shared infra): the reference of the "32-lane path
    bitwise unchanged" checks. Skips without git."""
    import importlib.util

    root = Path(__file__).resolve().parents[5]  # tt-metal (or a snapshot of it with a .git pointer)
    rel = f"models/demos/motif3/tt/{name}.py"
    try:
        src = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{rel}"], capture_output=True, text=True,
                             check=True).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"git show HEAD:{rel} failed: {e}")
    spec = importlib.util.spec_from_loader(f"models.demos.motif3.tt._b0_{name}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "models.demos.motif3.tt"
    mod.__file__ = str(root / rel)
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def moe_from_tt_cache(mesh_device, cfg, ccl, layer: int, *, module=None, **kw):
    """``MotifMoE`` of ``layer`` (``module``'s, default ``tt.moe``) from the serving TT cache alone (a raising source,
    nothing written); skips when the part is not converted."""
    from models.demos.motif3.tt.model import layer_cache_complete, read_only_cache

    if module is None:
        from models.demos.motif3.tt import moe as module
    if not layer_cache_complete(cfg, layer):
        pytest.skip(f"TT-cache part L{layer:02d} is not converted ({cfg.cache_dir})")
    t0 = time.time()
    with read_only_cache() as misses:
        moe = module.MotifMoE(mesh_device, cfg, layer, source=_RaisingSource(), ccl=ccl, cache=True, **kw)
    assert misses == [], f"tensors missing from the TT cache: {misses}"
    print(f"[moe] L{layer} ({module.__name__}, {kw}) loaded from the TT cache alone in {time.time() - t0:.1f} s")
    return moe


def upload_rows16(x64: torch.Tensor, cfg, mesh_device, device=True):
    """T64 rows: lane-ordered anchors ``x64[:32]`` and drafts ``x64[32:]`` -> per DP row ``[1, 1, 16, 4096]`` bf16 TILE
    = ``[the row's 8 anchors | their 8 drafts]`` (lanes ``8 dp + j``), replicated over TP."""
    from models.demos.motif3.tt.rope import shard_lanes

    L = cfg.lanes_per_row
    a = x64[:32].reshape(cfg.dp, L, -1)
    d = x64[32:].reshape(cfg.dp, L, -1)
    rows = torch.cat([a, d], dim=1).reshape(cfg.dp, 1, 2 * L, -1).bfloat16()
    return shard_lanes(rows, cfg, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                       device=mesh_device if device else None)


def rows16_to_halves(t16: torch.Tensor, cfg):
    """Per-chip T64 rows ``[R, C, ..., 16, W]`` -> ``(anchors [R, C, ..., 8, W], drafts [R, C, ..., 8, W])``."""
    L = cfg.lanes_per_row
    return t16[..., :L, :], t16[..., L:, :]


def natural_to_lanes(t64: torch.Tensor, cfg) -> torch.Tensor:
    """Gathered T64 rows in natural order (row ``16 dp + j``) ``[64, ...]`` -> ``[anchors of lanes 0..31 | drafts of
    lanes 0..31]`` ``[64, ...]``."""
    L = cfg.lanes_per_row
    r = t64.reshape(cfg.dp, 2, L, *t64.shape[1:])
    return torch.cat([r[:, 0].reshape(cfg.dp * L, *t64.shape[1:]), r[:, 1].reshape(cfg.dp * L, *t64.shape[1:])])


@pytest.mark.parametrize("router_logits", ["composite", "exact_fp32"])
@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_t64_rows(mesh_device, device_params, router_logits):
    """WP-D (D1) module test of the T64 verify step's MoE (P5_T64_DESIGN.md §4.2, T4; review edit R-E7), real layer-2
    weights from the serving TT cache, both router settings: 64 real router inputs as 16 rows per DP row (``[8 anchors
    | 8 drafts]``, M = 64 gathered tokens) vs the same tokens as two 32-lane steps (anchors, drafts):

    * every output row on all 32 chips **bitwise** equal to the 32-lane module's row (router per_core_M 2 + fused
      sigmoid or the exact-fp32 kernel at M = 64, gate_up / down per_core_M 2 with down on 8 x 8, 64-row top-k pad),
      and the same with the down projection on 8 x 4 cores (G16-lite's other grid);
    * the gathered tokens in natural order (``16 dp + j``) and the routes (indices and weights) bitwise equal to the
      32-lane routes of the same tokens, identical on all 32 chips; replicas identical over TP;
    * the 32-lane path bitwise equal to B0's committed module (git ``HEAD``; output on 32 chips and routes);
    * the output vs the fp32 reference routed output of these tokens (PCC >= 0.995);
    * a trace of the 16-row step replayed with new inputs == the eager step bitwise;
    * every new program's static CBs end below a one-page L1 pin (no clash error), and the traced cost of M = 64 vs
      M = 32 (G16-lite: 1204.1 vs 1038.6 us with the composite router; informational).
    A module built without a T64 config refuses the 16-row input (host test ``test_moe_host_t64_decode_configs``)."""
    from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, log_fabric

    cfg = t64_cfg(mesh_device)
    log_fabric(mesh_device, f"moe_t64_{router_logits}")
    print(f"[moe] {cfg.describe()}")
    ccl = MotifCCL(mesh_device, cfg)
    pin = l1_pin(mesh_device)
    layer = 2
    moe = moe_from_tt_cache(mesh_device, cfg, ccl, layer, router_logits=router_logits)
    assert moe.decode_rows == (32, 64) and sorted(moe.router._pads) == [32, 64], (moe.decode_rows, moe.router._pads)
    assert (moe.router.logits_fn is not None) == (router_logits == "exact_fp32")
    data = load_real_inputs()
    xs = data["layers"][layer]["x"]
    n = xs.shape[0]
    sel = spread_tokens(n, 64)
    x64 = xs[sel]  # 64 distinct real tokens: anchors = lanes' rows 0..31, drafts = 32..63
    failures = []

    def run(x_tt):
        taps = {}
        out = moe.forward_decode(x_tt, taps=taps)
        t = device_tensors_to_torch(out, mesh_device)  # [R, C, 1, 1, L, H]
        f_all, f_same = read_replicated(taps["f_all"], mesh_device)
        idx, w, r_same = read_routes(taps, moe, mesh_device)
        _free([taps, out])
        return t, f_all.reshape(-1, H).float(), idx, w, f_same and r_same

    # the 32-lane module: anchors, then drafts
    ref32 = []
    for half in (x64[:32], x64[32:]):
        x_tt = upload_lanes(half, cfg, mesh_device)
        ref32.append(run(x_tt))
        _free(x_tt)
    # the 32-lane path is bitwise B0's: the committed module (git HEAD) on the anchors
    moe_b0 = moe_from_tt_cache(mesh_device, cfg, ccl, layer, module=b0_module("moe"), router_logits=router_logits)
    x_tt = upload_lanes(x64[:32], cfg, mesh_device)
    taps = {}
    o = moe_b0.forward_decode(x_tt, taps=taps)
    b0_out = device_tensors_to_torch(o, mesh_device)
    b0_idx, b0_w, _ = read_routes(taps, moe_b0, mesh_device, chips=[0])
    _free([taps, o, x_tt])
    moe_b0.deallocate()
    same_b0 = bool(torch.equal(b0_out, ref32[0][0])) and bool(torch.equal(b0_idx, ref32[0][2])) and bool(
        torch.equal(b0_w, ref32[0][3]))
    print(f"[moe] 32-lane path ({router_logits}) bitwise == B0's committed module (output on 32 chips, routes): "
          f"{same_b0}")
    if not same_b0:
        failures.append(f"{router_logits}: the 32-lane path differs from B0's module")
    # the 16-row step (eager; compiles every M = 64 program while the L1 pin is allocated)
    x16 = upload_rows16(x64, cfg, mesh_device)
    t16, f64, idx64, w64, same64 = run(x16)
    anchors, drafts = rows16_to_halves(t16, cfg)
    rows_eq = (bool(torch.equal(anchors, ref32[0][0])), bool(torch.equal(drafts, ref32[1][0])))
    diff = max(float((anchors.float() - ref32[0][0].float()).abs().max()),
               float((drafts.float() - ref32[1][0].float()).abs().max()))
    gathered_ok = bool(torch.equal(natural_to_lanes(f64, cfg), x64.bfloat16().float()))
    routes_eq = bool(torch.equal(natural_to_lanes(idx64, cfg), torch.cat([ref32[0][2], ref32[1][2]]))) and bool(
        torch.equal(natural_to_lanes(w64, cfg), torch.cat([ref32[0][3], ref32[1][3]])))
    rep_tp = all(bool(torch.equal(t16[r, c], t16[r, 0])) for r in range(t16.shape[0]) for c in range(t16.shape[1]))
    finite = bool(torch.isfinite(t16.float()).all())
    print(f"[moe] T64 {router_logits}: output rows bitwise == 32-lane (anchors, drafts) {rows_eq} (max |diff| "
          f"{diff:.3e}) on all 32 chips; gathered tokens in natural order exact {gathered_ok}; routes (idx, w) "
          f"bitwise == 32-lane {routes_eq}, identical on 32 chips {same64}; replicas over TP identical {rep_tp}; "
          f"finite {finite}")
    if not (all(rows_eq) and gathered_ok and routes_eq and same64 and rep_tp and finite):
        failures.append(f"{router_logits}: rows {rows_eq} (max |diff| {diff:.3e}), gathered {gathered_ok}, routes "
                        f"{routes_eq}, replicas {same64}/{rep_tp}, finite {finite}")
    # the down projection of M = 64 on 8 x 4 cores (G16-lite's other grid, the fallback of 8 x 8): the same rows
    from models.demos.motif3.tt.model_config import EXPERTS_DOWN_GRID

    gu64, dn64 = moe.pc_wide[64]
    moe.pc_wide[64] = (gu64, cfg.experts_down_pc(m_tiles=2, grid=EXPERTS_DOWN_GRID))
    try:
        t16_84 = run(x16)[0]
    finally:
        moe.pc_wide[64] = (gu64, dn64)
    same84 = bool(torch.equal(t16_84, t16))
    print(f"[moe] T64 {router_logits}: down on 8 x 4 cores (per_core_N 4) gives the same rows bitwise: {same84}")
    if not same84:
        failures.append(f"{router_logits}: the 8 x 4 down grid changes the M = 64 rows")
    # accuracy vs the fp32 reference routed output of these tokens (cached golden of all L2 tokens), lane order
    golden = GOLDEN_DIR / f"ref_routed_L{layer:02d}_v1.pt"
    if golden.is_file():
        g = torch.load(golden, weights_only=True)
        if g["n"] == n and torch.equal(g["x_sum"], xs.float().sum(0)):
            want = g["routed"][sel]
            rows = []
            for half in (anchors, drafts):  # chip (dp, tp=0) holds DP row dp's 8 lanes
                rows.append(torch.cat([half[cfg.axes.coord(dp, 0)].reshape(-1, H) for dp in range(cfg.dp)]))
            s = stats(want, torch.cat(rows).float())
            print(f"[moe] T64 {router_logits} vs the fp32 reference routed output (64 tokens): {fmt(s)}")
            if s["pcc"] < 0.995:
                failures.append(f"{router_logits}: PCC vs reference {s['pcc']:.6f} < 0.995")
        else:
            print(f"[moe] {golden.name} does not match the captured inputs; reference check skipped")
    else:
        print(f"[moe] {golden.name} missing; reference check skipped")
    # trace: capture the 16-row step once, replay with new inputs, compare with eager (bitwise)
    with _Capture(mesh_device) as cap:
        out_t = moe.forward_decode(x16)
    try:
        for it, s0 in enumerate((1, 2)):
            x_new = xs[spread_tokens(n - s0, 64) + s0]
            ttnn.copy_host_to_device_tensor(upload_rows16(x_new, cfg, mesh_device, device=False), x16)
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            got_t = device_tensors_to_torch(out_t, mesh_device)
            out_e = moe.forward_decode(x16)
            got_e = device_tensors_to_torch(out_e, mesh_device)
            _free(out_e)
            exact = bool(torch.equal(got_t, got_e))
            print(f"[moe] T64 {router_logits} trace replay {it}: traced == eager bitwise {exact}")
            if not exact:
                failures.append(f"{router_logits}: trace replay {it} differs from eager")
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
        _free(out_t)
    # traced cost (informational; slope method): M = 64 vs M = 32
    x32 = upload_lanes(x64[:32], cfg, mesh_device)
    st64 = traced_stats(mesh_device, lambda: moe.forward_decode(x16), n=8, reps=5, adapt_to=16)
    st32 = traced_stats(mesh_device, lambda: moe.forward_decode(x32), n=8, reps=5, adapt_to=16)
    print(f"[moe] T64 {router_logits} traced: M = 64 {fmt_traced(st64)} us vs M = 32 {fmt_traced(st32)} us per call "
          f"(G16-lite composite: 1204.1 vs 1038.6 us)")
    _free([x32, x16, pin])
    moe.deallocate()
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_decode_ccl_rs(mesh_device, device_params):
    """Phase C D4 (``decode_ccl="rs"``, ``MOTIF3_MOE_DECODE_CCL``; logs/opt/phaseC/D4): the decode combine as row fold +
    ONE reduce-scatter over DP + unfold (``tt/kernels/row_fold.py``) instead of AR(dp) + partition. Real layer-2 weights
    from the serving TT cache, a T64-staged config (M = 32 and 64), 64 real tokens, a random bf16 TP ``add_partial``:

    * the fold / unfold kernels bitwise equal to the logical reshapes (``ttnn.reshape``), unfold padding rows zero;
      ``fold_add`` bitwise ``ttnn.add(R, ttnn.reshape(S))``; ``reduce_tp=False`` + ``ar_tp`` close to the output;
    * "rs" vs "ar": the same DP sums in another order (not bitwise): max |diff| and the differing-word fraction reported,
      PCC >= 0.99999; both vs the fp32 reference routed output (PCC >= 0.995);
    * "rs": replicas over TP identical, finite; T64 rows (16 per DP row) bitwise equal to the 32-lane rows (the T32 / T64
      contract: a row's DP sum is reduced on its own chip in both);
    * "rs": a trace replayed with new inputs == eager bitwise; determinism soak: ``MOTIF3_D4_SOAK`` (default 200) replays
      of one captured trace, every output bitwise equal to the first;
    * traced cost of the module (rs vs ar, M = 32 and 64; informational)."""
    from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, log_fabric
    from models.demos.motif3.tt.kernels.row_fold import RowFold

    cfg = t64_cfg(mesh_device)
    log_fabric(mesh_device, "moe_d4_rs")
    ccl = MotifCCL(mesh_device, cfg)
    pin = l1_pin(mesh_device)
    failures = []
    # ---- kernels vs ttnn.reshape (pure data movement) ----
    rf = RowFold(mesh_device)
    mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], ttnn.MeshShape(*cfg.axes.mesh_shape)))
    R, C = cfg.axes.mesh_shape
    g = torch.Generator().manual_seed(4)
    for L in (8, 16):
        for mc in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
            Ph = torch.randn(R, C, 4 * L, H, generator=g).bfloat16()
            P = ttnn.from_torch(Ph, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                mesh_mapper=mapper, memory_config=mc)
            Rt = ttnn.from_torch(Ph[..., : H // 4].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=mesh_device, mesh_mapper=mapper, memory_config=mc)
            q, u = rf.fold(P, L, memory_config=mc), rf.unfold(Rt, L, memory_config=mc)
            qh = device_tensors_to_torch(q, mesh_device)
            uh = device_tensors_to_torch(u, mesh_device)
            v = ttnn.reshape(u, (1, 1, 32, H), (1, 1, 32, H))  # a view with the padding rows logical
            pad = device_tensors_to_torch(v, mesh_device)[..., L:, :]
            Sh = torch.randn(R, C, L, H, generator=g).bfloat16()
            St = ttnn.from_torch(Sh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                 mesh_mapper=mapper, memory_config=mc)
            fa = device_tensors_to_torch(rf.fold_add(Rt, St, L, memory_config=mc), mesh_device)
            sf = ttnn.reshape(St, (1, 1, 4 * L, H // 4))
            ad = ttnn.add(Rt, sf, memory_config=mc)
            ok_a = bool(torch.equal(fa, device_tensors_to_torch(ad, mesh_device)))
            _free([St, sf, ad])
            print(f"[moe] D4 fold_add L={L} {mc.buffer_type}: == ttnn.add(R, reshape(S)) bitwise {ok_a}")
            if not ok_a:
                failures.append(f"fold_add L={L} {mc.buffer_type}")
            ok_q = bool(torch.equal(qh, Ph.reshape(R, C, 1, 4, 4 * L, H // 4)))
            ok_u = bool(torch.equal(uh, Ph[..., : H // 4].reshape(R, C, 1, 1, L, H)))
            ok_p = bool((pad == 0).all())
            print(f"[moe] D4 row fold L={L} {mc.buffer_type}: fold == reshape {ok_q}, unfold == reshape {ok_u}, "
                  f"unfold padding zero {ok_p}")
            if not (ok_q and ok_u and ok_p):
                failures.append(f"row fold L={L} {mc.buffer_type}: fold {ok_q} unfold {ok_u} padding {ok_p}")
            _free([P, Rt, q, u])
    # ---- the module ----
    layer = 2
    moe_ar = moe_from_tt_cache(mesh_device, cfg, ccl, layer, decode_ccl="ar")
    moe_rs = moe_from_tt_cache(mesh_device, cfg, ccl, layer, decode_ccl="rs")
    assert moe_rs.decode_rs_applies(8) and moe_rs.decode_rs_applies(16) and not moe_ar.decode_rs_applies(8)
    data = load_real_inputs()
    xs = data["layers"][layer]["x"]
    n = xs.shape[0]
    sel = spread_tokens(n, 64)
    x64 = xs[sel]
    tp_mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(0), ttnn.PlacementShard(1)], ttnn.MeshShape(R, C)))

    def partial(L, seed):
        t = (torch.randn(R, C, L, H, generator=torch.Generator().manual_seed(seed)) * 0.02).bfloat16()
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                               mesh_mapper=tp_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x32 = upload_lanes(x64[:32], cfg, mesh_device)
    x32d = upload_lanes(x64[32:], cfg, mesh_device)
    x16 = upload_rows16(x64, cfg, mesh_device)
    ap8 = partial(8, 11)
    ap8d = partial(8, 12)
    # T64 add_partial = [the anchors' partial | the drafts' partial] per DP row
    a8h = device_tensors_to_torch(ap8, mesh_device).reshape(R, C, 8, H)
    a8dh = device_tensors_to_torch(ap8d, mesh_device).reshape(R, C, 8, H)
    ap16 = ttnn.from_torch(torch.cat([a8h, a8dh], dim=2).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=mesh_device, mesh_mapper=tp_mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def run(moe, x, ap):
        o = moe.forward_decode(x, add_partial=ap)
        t = device_tensors_to_torch(o, mesh_device)
        _free(o)
        return t

    for ap_name, (a, ad) in (("none", (None, None)), ("partial", (ap8, ap8d))):
        o_ar = run(moe_ar, x32, a)
        o_rs = run(moe_rs, x32, a)
        o_rs_d = run(moe_rs, x32d, ad)
        diff = (o_rs.float() - o_ar.float()).abs()
        frac = float((o_rs != o_ar).float().mean())
        s = stats(o_ar.float().reshape(-1, H), o_rs.float().reshape(-1, H))
        rep_tp = all(bool(torch.equal(o_rs[r, c], o_rs[r, 0])) for r in range(R) for c in range(C))
        finite = bool(torch.isfinite(o_rs.float()).all())
        print(f"[moe] D4 rs vs ar (add_partial {ap_name}): max |diff| {float(diff.max()):.3e}, words differing "
              f"{frac:.4f}, {fmt(s)}; rs replicas over TP identical {rep_tp}, finite {finite}")
        if s["pcc"] < 0.99999 or not rep_tp or not finite:
            failures.append(f"rs vs ar ({ap_name}): pcc {s['pcc']:.7f}, replicas {rep_tp}, finite {finite}")
        # T64 rows == the 32-lane rows (rs)
        t16 = run(moe_rs, x16, ap16 if a is not None else None)
        anchors, drafts = rows16_to_halves(t16, cfg)
        rows_eq = (bool(torch.equal(anchors, o_rs)), bool(torch.equal(drafts, o_rs_d)))
        print(f"[moe] D4 rs T64 rows bitwise == 32-lane rows (anchors, drafts; add_partial {ap_name}): {rows_eq}")
        if not all(rows_eq):
            failures.append(f"rs T64 rows ({ap_name}): {rows_eq}")
        if a is None:  # accuracy vs the fp32 reference routed output
            golden = GOLDEN_DIR / f"ref_routed_L{layer:02d}_v1.pt"
            if golden.is_file():
                gref = torch.load(golden, weights_only=True)
                if gref["n"] == n and torch.equal(gref["x_sum"], xs.float().sum(0)):
                    want = gref["routed"][sel][:32]
                    for nm, t in (("ar", o_ar), ("rs", o_rs)):
                        got = torch.cat([t[cfg.axes.coord(dp, 0)].reshape(-1, H) for dp in range(cfg.dp)])
                        sr = stats(want, got.float())
                        print(f"[moe] D4 {nm} vs the fp32 reference routed output (32 tokens): {fmt(sr)}")
                        if sr["pcc"] < 0.995:
                            failures.append(f"{nm}: PCC vs reference {sr['pcc']:.6f}")
    # reduce_tp=False: the unfolded column partial; closing it with AR(tp) gives the module output up to rounding
    pt = moe_rs.forward_decode(x32, add_partial=ap8, reduce_tp=False)
    closed = ccl.ar_tp(pt)
    o_close = device_tensors_to_torch(closed, mesh_device)
    o_full = run(moe_rs, x32, ap8)
    _free([pt, closed])
    s_close = stats(o_full.float().reshape(-1, H), o_close.float().reshape(-1, H))
    print(f"[moe] D4 rs reduce_tp=False + ar_tp vs the module output: {fmt(s_close)}")
    if s_close["pcc"] < 0.99999:
        failures.append(f"rs reduce_tp=False: pcc {s_close['pcc']:.7f}")
    # ---- trace: replay with new inputs == eager; determinism soak ----
    with _Capture(mesh_device) as cap:
        out_t = moe_rs.forward_decode(x32, add_partial=ap8)
    try:
        for it, s0 in enumerate((1, 2)):
            x_new = xs[spread_tokens(n - s0, 32) + s0]
            ttnn.copy_host_to_device_tensor(upload_lanes(x_new, cfg, mesh_device, device=False), x32)
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            got_t = device_tensors_to_torch(out_t, mesh_device)
            out_e = moe_rs.forward_decode(x32, add_partial=ap8)
            got_e = device_tensors_to_torch(out_e, mesh_device)
            _free(out_e)
            exact = bool(torch.equal(got_t, got_e))
            print(f"[moe] D4 rs trace replay {it}: traced == eager bitwise {exact}")
            if not exact:
                failures.append(f"rs trace replay {it} differs from eager")
        n_soak = int(os.environ.get("MOTIF3_D4_SOAK", "200"))
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
        first = device_tensors_to_torch(out_t, mesh_device)
        bad = 0
        t0 = time.time()
        for _ in range(n_soak):
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            if not torch.equal(device_tensors_to_torch(out_t, mesh_device), first):
                bad += 1
        print(f"[moe] D4 rs determinism soak: {n_soak} trace replays, {bad} differ from the first "
              f"({time.time() - t0:.1f} s)")
        if bad:
            failures.append(f"rs determinism soak: {bad} / {n_soak} replays differ")
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
        _free(out_t)
    # ---- traced cost (informational) ----
    for nm, moe in (("ar", moe_ar), ("rs", moe_rs)):
        st32 = traced_stats(mesh_device, lambda: moe.forward_decode(x32, add_partial=ap8), n=8, reps=5, adapt_to=16)
        st64 = traced_stats(mesh_device, lambda: moe.forward_decode(x16, add_partial=ap16), n=8, reps=5, adapt_to=16)
        print(f"[moe] D4 {nm} traced forward_decode (+ add_partial): M = 32 {fmt_traced(st32)} us, M = 64 "
              f"{fmt_traced(st64)} us")
    _free([x32, x32d, x16, ap8, ap8d, ap16, pin])
    moe_ar.deallocate()
    moe_rs.deallocate()
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("router_logits", ["composite", "exact_fp32"])
@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_router_mask_scatter(mesh_device, device_params, router_logits):
    """A5 (docs/OPTIMIZATION_PLAN.md §3.3; ``router_mask="scatter"``; probe logs/opt/phaseA/A5), real weights from the
    serving TT cache, both routers, on the T64 config (32 and 64 decode rows):

    * the top-8 contract: on every real router input (2971 tokens; all 11 captured layers with the composite router,
      layer 2 with the exact one) the experts the scatter path weights (union of ``w_loc`` over the 32 chips) are
      exactly the gather path's, 8 per token, and the weights agree to <= 1e-6 relative (measured: 1-4 fp32 ulp);
    * the module output at M = 32 vs the gather module: PCC >= 0.99999, max |d| <= 2 bf16 ulp of the row's scale;
    * T64: the 16-row step's rows bitwise equal the two 32-lane steps' rows (scatter module), and a rerun is bitwise.
    Prefill is not affected (decode only)."""
    from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, log_fabric

    cfg = t64_cfg(mesh_device)
    log_fabric(mesh_device, f"moe_router_mask_{router_logits}")
    ccl = MotifCCL(mesh_device, cfg)
    data = load_real_inputs()
    layers = list(data["meta"]["layers"]) if router_logits == "composite" else [2]
    L1 = ttnn.L1_MEMORY_CONFIG
    E_loc = cfg.experts_per_chip
    failures = []

    def full_wloc(t, M):
        a = device_tensors_to_torch(t, mesh_device)  # [R, C, 1, 12, M, 1]
        out = torch.zeros(M, E, dtype=torch.float32)
        for dp in range(cfg.dp):
            for tp in range(cfg.tp):
                k0 = E_loc * (cfg.tp * dp + tp)
                out[:, k0:k0 + E_loc] = a[cfg.axes.coord(dp, tp)].reshape(E_loc, M).t().float()
        return out

    for layer in layers:
        # explicit "gather": the reference module must not follow MOTIF3_ROUTER_MASK (review I-2)
        old = moe_from_tt_cache(mesh_device, cfg, ccl, layer, router_logits=router_logits, router_mask="gather")
        new = moe_from_tt_cache(mesh_device, cfg, ccl, layer, router_logits=router_logits, router_mask="scatter")
        assert old.router_mask == "gather" and new.router_mask == "scatter" and sorted(new.scatter_consts) == [32, 64]
        xs = data["layers"][layer]["x"]
        n = xs.shape[0]
        blocks = _decode_blocks(xs, mesh_device)
        set_eq, max_rel = 0, 0.0
        for bi, bt in enumerate(blocks):
            nv = min(32, n - 32 * bi)
            got = {}
            for tag, m in (("old", old), ("new", new)):
                taps = {}
                part = m.local_partial(bt, polynorm=m.decode_polynorm, decode=True, taps=taps, memory_config=L1)
                got[tag] = full_wloc(taps["w_loc"], 32)[:nv]
                _free([taps, part])
            a, b = got["old"], got["new"]
            ok = ((a != 0) == (b != 0)).all(-1) & ((b != 0).sum(-1) == K)
            set_eq += int(ok.sum())
            nz = a != 0
            if nz.any():
                max_rel = max(max_rel, float(((a - b).abs()[nz] / a.abs()[nz]).max()))
        _free(blocks)
        print(f"[moe] A5 L{layer} {router_logits}: top-8 sets equal {set_eq}/{n}, max rel |dw| {max_rel:.2e}")
        if set_eq != n or max_rel > 1e-6:
            failures.append(f"L{layer} {router_logits}: sets {set_eq}/{n}, max rel {max_rel:.2e}")
        if layer == layers[0]:
            sel = spread_tokens(n, 64)
            x64 = xs[sel]
            outs = {}
            for tag, m in (("old", old), ("new", new)):
                x_tt = upload_lanes(x64[:32], cfg, mesh_device)
                o = m.forward_decode(x_tt)
                outs[tag] = device_tensors_to_torch(o, mesh_device).float()
                _free([o, x_tt])
            p = pcc(outs["old"].flatten(), outs["new"].flatten())
            scale = outs["old"].abs().amax(-1, keepdim=True).clamp_min(1e-30)
            rel = float(((outs["old"] - outs["new"]).abs() / scale).max())
            print(f"[moe] A5 L{layer} {router_logits}: output vs gather PCC {p:.10f}, max |d| / row max {rel:.2e}")
            if p < 0.99999 or rel > 2 * 2.0 ** -7:
                failures.append(f"L{layer} {router_logits}: output PCC {p} rel {rel}")
            halves = []
            for half in (x64[:32], x64[32:]):
                x_tt = upload_lanes(half, cfg, mesh_device)
                o = new.forward_decode(x_tt)
                halves.append(device_tensors_to_torch(o, mesh_device))
                _free([o, x_tt])
            reps = []
            for _ in range(2):
                x16 = upload_rows16(x64, cfg, mesh_device)
                o = new.forward_decode(x16)
                reps.append(device_tensors_to_torch(o, mesh_device))
                _free([o, x16])
            anchors, drafts = rows16_to_halves(reps[0], cfg)
            t64_eq = bool(torch.equal(anchors, halves[0])) and bool(torch.equal(drafts, halves[1]))
            rerun_eq = bool(torch.equal(reps[0], reps[1]))
            print(f"[moe] A5 L{layer} {router_logits}: T64 rows == T32 rows bitwise {t64_eq}, rerun bitwise {rerun_eq}")
            if not (t64_eq and rerun_eq):
                failures.append(f"L{layer} {router_logits}: T64 rows {t64_eq} rerun {rerun_eq}")
        old.deallocate()
        new.deallocate()
    assert not failures, failures


@pytest.mark.parametrize("router_logits", ["composite", "exact_fp32"])
@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_router_fused(mesh_device, device_params, router_logits):
    """B4 (docs/OPTIMIZATION_PLAN.md §3.3 "A5 and B4"; ``router_mask="fused"``, ``tt/kernels/router_topk.py``; results
    logs/opt/phaseB/B4), real weights from the serving TT cache, both routers, on the T64 config (32 and 64 rows):

    * the top-8 contract: on every real router input (2971 tokens; all 11 captured layers with the composite router,
      layer 2 with the exact one) the experts the fused path weights (union of ``w_loc`` over the 32 chips) are exactly
      the gather path's, 8 per token (100 %), the kernel's ``idx`` is ttnn.topk's set, and the weights agree to
      <= 1e-6 relative;
    * the kernel is its host model: every chip's ``w_loc`` and the ``idx`` equal ``router_topk.emulate_fp32`` on the
      device's scores bitwise (scores replicated on all chips);
    * synthetic exact ties (composite run): 8th = 9th and 7th = 8th = 9th rows select the lower ids;
    * the module output at M = 32 vs the gather module: PCC >= 0.99999, max |d| <= 2 bf16 ulp of the row's scale;
    * T64: the 16-row step's rows bitwise equal the two 32-lane steps' rows, a rerun is bitwise, a trace replay with new
      inputs equals eager; traced router + mask cost fused vs gather (informational). Prefill is not affected."""
    from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, log_fabric
    from models.demos.motif3.tt.kernels import router_topk as RT

    cfg = t64_cfg(mesh_device)
    fab = log_fabric(mesh_device, f"moe_router_fused_{router_logits}")
    assert str(fab.get("committed")) == "TORUS_XY", fab
    ccl = MotifCCL(mesh_device, cfg)
    data = load_real_inputs()
    layers = list(data["meta"]["layers"]) if router_logits == "composite" else [2]
    L1 = ttnn.L1_MEMORY_CONFIG
    E_loc = cfg.experts_per_chip
    failures = []

    def full_wloc(t, M):
        a = device_tensors_to_torch(t, mesh_device)  # [R, C, 1, 12, M, 1]
        out = torch.zeros(M, E, dtype=torch.float32)
        for dp in range(cfg.dp):
            for tp in range(cfg.tp):
                k0 = E_loc * (cfg.tp * dp + tp)
                out[:, k0:k0 + E_loc] = a[cfg.axes.coord(dp, tp)].reshape(E_loc, M).t().float()
        return out

    def chip0(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])

    if router_logits == "composite":  # synthetic exact ties on the kernel alone (zero bias)
        ids = moe_from_tt_cache(mesh_device, cfg, ccl, 2, router_mask="gather")
        g = torch.Generator().manual_seed(0)
        s = torch.rand(32, E, generator=g) * 0.5
        srt = torch.sort(s, dim=-1, descending=True)
        for r in range(16):
            s[r, srt.indices[r, 8]] = s[r, srt.indices[r, 7]]
        for r in range(16, 24):
            s[r, srt.indices[r, 6]] = s[r, srt.indices[r, 7]]
            s[r, srt.indices[r, 8]] = s[r, srt.indices[r, 7]]
        st = upload_replicated(s.reshape(1, 1, 32, E), mesh_device, dtype=ttnn.float32)
        zb = upload_replicated(torch.zeros(1, 1, 1, E), mesh_device, dtype=ttnn.float32)
        kern = RT.FusedRouterTopK(mesh_device, zb, ids.local_ids, top_k=K)
        w_t, i_t = kern(st, scale=1.0, memory_config=L1, want_idx=True)
        ew, ei = RT.emulate_fp32(s.reshape(1, 1, 32, E), torch.zeros(E), top_k=K, base=0, e_loc=E)
        ok_i = bool(torch.equal(chip0(i_t).reshape(32, K).long(), ei.reshape(32, K)))
        ok_w = bool(torch.equal(full_wloc(w_t, 32).view(torch.int32), ew.reshape(E, 32).t().view(torch.int32)))
        print(f"[moe] B4 synthetic ties: idx == emulation (lower id) {ok_i}, w_loc bitwise {ok_w}")
        if not (ok_i and ok_w):
            failures.append(f"synthetic ties: idx {ok_i} w {ok_w}")
        _free([st, zb, w_t, i_t])
        kern.deallocate()
        ids.deallocate()

    for layer in layers:
        # explicit modes: neither module may follow MOTIF3_ROUTER_MASK (review I-2)
        old = moe_from_tt_cache(mesh_device, cfg, ccl, layer, router_logits=router_logits, router_mask="gather")
        new = moe_from_tt_cache(mesh_device, cfg, ccl, layer, router_logits=router_logits, router_mask="fused")
        assert old.router_mask == "gather" and old.router_fused is None
        assert new.router_mask == "fused" and new.router_fused is not None and new.scatter_consts == {}
        xs = data["layers"][layer]["x"]
        n = xs.shape[0]
        bias = chip0(new.router.bias).reshape(-1).float()[:E]
        blocks = _decode_blocks(xs, mesh_device)
        set_eq, idx_eq, emu_eq, max_rel, rep_ok = 0, 0, 0, 0.0, True
        for bi, bt in enumerate(blocks):
            nv = min(32, n - 32 * bi)
            got = {}
            taps_new = {}
            for tag, m, taps in (("old", old, {}), ("new", new, taps_new)):
                part = m.local_partial(bt, polynorm=m.decode_polynorm, decode=True, taps=taps, memory_config=L1)
                got[tag] = full_wloc(taps["w_loc"], 32)[:nv]
                if tag == "old":
                    i_old = chip0(taps["idx"]).reshape(32, -1)[:nv, :K].long()
                    _free([taps, part])
                else:
                    _free(part)
            sc, same = read_replicated(taps_new["scores"], mesh_device)
            rep_ok &= bool(same)
            i_new = chip0(taps_new["idx"]).reshape(32, K).long()
            _free(taps_new)
            ew, ei = RT.emulate_fp32(sc.float().reshape(1, 1, 32, E), bias, top_k=K, base=0, e_loc=E,
                                     scale=new.internal_route_scale)
            ew = ew.reshape(E, 32).t()[:nv]
            emu_eq += int(((ew.view(torch.int32) == got["new"].view(torch.int32)).all(-1)
                           & (ei.reshape(32, K)[:nv] == i_new[:nv]).all(-1)).sum())
            a, b = got["old"], got["new"]
            ok = ((a != 0) == (b != 0)).all(-1) & ((b != 0).sum(-1) == K)
            set_eq += int(ok.sum())
            ma = torch.zeros(nv, E, dtype=torch.bool).scatter_(1, i_old, True)
            mb = torch.zeros(nv, E, dtype=torch.bool).scatter_(1, i_new[:nv], True)
            idx_eq += int((ma == mb).all(-1).sum())
            nz = a != 0
            if nz.any():
                max_rel = max(max_rel, float(((a - b).abs()[nz] / a.abs()[nz]).max()))
        _free(blocks)
        print(f"[moe] B4 L{layer} {router_logits}: top-8 sets equal {set_eq}/{n}, idx sets {idx_eq}/{n}, kernel == "
              f"emulation {emu_eq}/{n}, scores replicated {rep_ok}, max rel |dw| {max_rel:.2e}")
        if set_eq != n or idx_eq != n or emu_eq != n or not rep_ok or max_rel > 1e-6:
            failures.append(f"L{layer} {router_logits}: sets {set_eq} idx {idx_eq} emu {emu_eq} rep {rep_ok} "
                            f"rel {max_rel:.2e} (n {n})")
        if layer == layers[0]:
            sel = spread_tokens(n, 64)
            x64 = xs[sel]
            outs = {}
            for tag, m in (("old", old), ("new", new)):
                x_tt = upload_lanes(x64[:32], cfg, mesh_device)
                o = m.forward_decode(x_tt)
                outs[tag] = device_tensors_to_torch(o, mesh_device).float()
                _free([o, x_tt])
            p = pcc(outs["old"].flatten(), outs["new"].flatten())
            scale = outs["old"].abs().amax(-1, keepdim=True).clamp_min(1e-30)
            rel = float(((outs["old"] - outs["new"]).abs() / scale).max())
            print(f"[moe] B4 L{layer} {router_logits}: output vs gather PCC {p:.10f}, max |d| / row max {rel:.2e}")
            if p < 0.99999 or rel > 2 * 2.0 ** -7:
                failures.append(f"L{layer} {router_logits}: output PCC {p} rel {rel}")
            halves = []
            for half in (x64[:32], x64[32:]):
                x_tt = upload_lanes(half, cfg, mesh_device)
                o = new.forward_decode(x_tt)
                halves.append(device_tensors_to_torch(o, mesh_device))
                _free([o, x_tt])
            reps = []
            for _ in range(2):
                x16 = upload_rows16(x64, cfg, mesh_device)
                o = new.forward_decode(x16)
                reps.append(device_tensors_to_torch(o, mesh_device))
                _free([o, x16])
            anchors, drafts = rows16_to_halves(reps[0], cfg)
            t64_eq = bool(torch.equal(anchors, halves[0])) and bool(torch.equal(drafts, halves[1]))
            rerun_eq = bool(torch.equal(reps[0], reps[1]))
            # trace: capture the 64-row step once, replay with new tokens == eager
            x16 = upload_rows16(x64, cfg, mesh_device)
            _free(new.forward_decode(x16))  # compile before the capture (F3N rule R2)
            trace_eq = []
            with _Capture(mesh_device) as cap:
                out_t = new.forward_decode(x16)
            try:
                for s0 in (1, 2):
                    x_new = xs[spread_tokens(n - s0, 64) + s0]
                    ttnn.copy_host_to_device_tensor(upload_rows16(x_new, cfg, mesh_device, device=False), x16)
                    ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                    got_t = device_tensors_to_torch(out_t, mesh_device)
                    o = new.forward_decode(x16)
                    trace_eq.append(bool(torch.equal(got_t, device_tensors_to_torch(o, mesh_device))))
                    _free(o)
            finally:
                ttnn.release_trace(mesh_device, cap.tid)
                _free(out_t)
            print(f"[moe] B4 L{layer} {router_logits}: T64 rows == T32 rows bitwise {t64_eq}, rerun bitwise {rerun_eq}"
                  f", trace replay == eager {trace_eq}")
            if not (t64_eq and rerun_eq and all(trace_eq)):
                failures.append(f"L{layer} {router_logits}: T64 rows {t64_eq} rerun {rerun_eq} trace {trace_eq}")
            if router_logits == "composite":  # traced cost (informational; slope method)
                for M in (32, 64):
                    xb = upload_replicated(xs[spread_tokens(n, M)], mesh_device)
                    t_new = traced_stats(mesh_device, lambda: new.router.route_fused(
                        xb, new.router_fused, scale=new.internal_route_scale, memory_config=L1), n=32, reps=7,
                        adapt_to=128)

                    def gather_path():
                        i, w = old.router(xb, scale=old.internal_route_scale, memory_config=L1)
                        wl = old.local_weights(i, w, memory_config=L1)
                        _free([i, w])
                        return wl

                    t_old = traced_stats(mesh_device, gather_path, n=32, reps=7, adapt_to=128)
                    print(f"[moe] B4 traced router + mask M = {M}: fused {fmt_traced(t_new)} us vs gather "
                          f"{fmt_traced(t_old)} us per layer")
                    _free(xb)
            _free(x16)
        old.deallocate()
        new.deallocate()
    assert not failures, failures


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_sparse_decode_experts(mesh_device, device_params):
    """B1 (docs/OPTIMIZATION_PLAN.md §3.3; ``decode_experts="sparse"``; probe logs/opt/phaseA/M6), real layer-2 /
    layer-35 weights from the serving TT cache on the T64 config, real router inputs:

    * the lane mask: ``MotifMoE.decode_lane_mask`` from ``MotifAttention.active_mask_host`` == the live rows in the
      gathered order (M = 32 and 64), on every chip;
    * M = 32 with 1 / 8 / 32 live lanes and M = 64 with 2 / 16 / 64 live rows (3 batches each): the sparse output's live
      rows bitwise == the dense module's (``add_partial`` included), inactive rows exactly the closed ``add_partial``
      (the all-inactive output; 0 without it),
      no NaN / Inf, replicas identical; the sparsity = the experts the live rows route to (busiest chip reported);
      ``lane_mask=None``: every row bitwise == dense;
    * T64: the 16-row step's rows bitwise == the two 32-lane steps' rows (sparse, all rows live);
    * trace: one capture with persistent ``x`` / lane-mask inputs, replays with new tokens *and* new live sets ==
      eager bitwise, and two replays are bitwise equal (determinism);
    * traced cost per call (informational): dense vs sparse at 1 / 8 / 32 live lanes."""
    from models.demos.motif3.tt.attention import MotifAttention
    from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, log_fabric
    from models.demos.motif3.tt.moe import MotifMoE

    cfg = t64_cfg(mesh_device)
    fab = log_fabric(mesh_device, "moe_sparse_decode_experts")
    assert str(fab.get("committed")) == "TORUS_XY", fab
    ccl = MotifCCL(mesh_device, cfg)
    data = load_real_inputs()
    L = cfg.lanes_per_row
    failures = []
    g = torch.Generator().manual_seed(2024)
    REP = ttnn.ReplicateTensorToMesh(mesh_device)

    def mask_host(live_nat, M):
        m = torch.zeros(1, 1, M, 1)
        m[0, 0, live_nat, 0] = 1.0
        return ttnn.from_torch(m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=REP)

    def lanes_of(t, M):
        """forward_decode output -> rows in lane order ([32, H]; M = 64: anchors of lanes 0..31 | drafts), replicas."""
        full = device_tensors_to_torch(t, mesh_device)  # [R, C, 1, 1, Lr, H]
        same, rows = True, []
        for dp in range(cfg.dp):
            r0 = full[cfg.axes.coord(dp, 0)]
            for tp in range(cfg.tp):
                same &= bool(torch.equal(full[cfg.axes.coord(dp, tp)], r0))
            rows.append(r0.reshape(-1, H))
        rows = torch.stack(rows)  # [dp, Lr, H]
        if M == 32:
            return rows.reshape(32, H).float(), same
        return torch.cat([rows[:, :L].reshape(32, H), rows[:, L:].reshape(32, H)]).float(), same

    def nat_of_lane_rows(M, lane_rows):
        """lane-order row ids (M = 64: anchors 0..31, drafts 32..63) -> natural gathered ids (Lr dp + j)."""
        if M == 32:
            return sorted(lane_rows)
        out = []
        for r in lane_rows:
            lane, draft = r % 32, r // 32
            dp, j = divmod(lane, L)
            out.append(2 * L * dp + L * draft + j)
        return sorted(out)

    for layer in (2, 35):
        if layer not in data["layers"]:
            continue
        moe = moe_from_tt_cache(mesh_device, cfg, ccl, layer)
        assert moe.decode_experts in ("dense", "sparse") and moe.decode_rows == (32, 64)
        xs = data["layers"][layer]["x"].float()
        n = xs.shape[0]

        # ---- the lane mask builder vs host (gathered natural order), M = 32 and 64 ----
        for M, Lr in ((32, L), (64, 2 * L)):
            pos = torch.full((cfg.dp * Lr,), -1, dtype=torch.int32)
            live = torch.randperm(cfg.dp * Lr, generator=g)[: max(1, M // 3)]
            pos[live] = 100
            act = MotifAttention.active_mask_host(pos, cfg, mesh_device, device=mesh_device, rows_per_dp=Lr)
            lm = MotifMoE.decode_lane_mask(ccl, act, Lr)
            got, same = read_replicated(lm, mesh_device)
            want = (pos >= 0).float().reshape(1, 1, M, 1)
            ok = bool(same) and torch.equal(got.float().reshape(1, 1, M, 1), want) and lm.dtype == ttnn.float32
            print(f"[moe] B1 L{layer} decode_lane_mask M={M}: == host {ok}")
            if not ok:
                failures.append(f"L{layer} lane mask M={M}")
            _free([act, lm])

        # ---- eager: sparse vs dense at real routes ----
        for M, cs in ((32, (1, 8, 32)), (64, (2, 16, 64))):
            for c in cs:
                for b in range(3):
                    tok = torch.randperm(n, generator=g)[:M]
                    x = xs[0].expand(M, H).clone()  # inactive rows: some real hidden state (token 0)
                    if M == 32:
                        lane_rows = sorted(torch.randperm(32, generator=g)[:c].tolist())
                    else:
                        lanes = sorted(torch.randperm(32, generator=g)[: c // 2].tolist())
                        lane_rows = lanes + [32 + l for l in lanes]  # anchor + draft of each live lane
                    x[lane_rows] = xs[tok[: len(lane_rows)]]
                    x_tt = upload_lanes(x, cfg, mesh_device) if M == 32 else upload_rows16(x, cfg, mesh_device)
                    add = upload_lanes(xs[tok[:32]] * 0.01, cfg, mesh_device) if M == 32 else None
                    nat = nat_of_lane_rows(M, lane_rows)
                    lm = ttnn.to_device(mask_host(nat, M), mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    outs = {}
                    for mode, mask in (("dense", lm), ("sparse", lm), ("sparse_nomask", None)):
                        moe.decode_experts = "dense" if mode == "dense" else "sparse"
                        taps = {} if mode == "sparse" else None
                        o = moe.forward_decode(x_tt, add_partial=add, lane_mask=mask, taps=taps)
                        outs[mode], same = lanes_of(o, M)
                        if not same:
                            failures.append(f"L{layer} M{M} c{c} {mode}: replicas differ")
                        if taps is not None:
                            act_k = torch.stack([ttnn.to_torch(t).float().reshape(-1)
                                                 for t in ttnn.get_device_tensors(taps["sparsity"])]) != 0
                            if b == 0:
                                print(f"[moe] B1 L{layer} M={M} c={c}: experts on the busiest chip "
                                      f"{int(act_k.sum(1).max())}, mean {float(act_k.sum(1).float().mean()):.2f}")
                            _free(taps)
                        _free(o)
                    # every row inactive: the output is the closed add_partial alone (0 without it)
                    z = ttnn.to_device(mask_host([], M), mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    o = moe.forward_decode(x_tt, add_partial=add, lane_mask=z)
                    zero_ref = lanes_of(o, M)[0]
                    _free([o, z])
                    d, s_, s0 = outs["dense"], outs["sparse"], outs["sparse_nomask"]
                    dead = [r for r in range(M) if r not in lane_rows]
                    ok_live = bool(torch.equal(s_[lane_rows], d[lane_rows]))
                    ok_dead = (not dead) or bool(torch.equal(s_[dead], zero_ref[dead]))
                    if add is None:
                        ok_dead = ok_dead and float(zero_ref.abs().max()) == 0.0
                    ok_nm = bool(torch.equal(s0, d))
                    finite = bool(torch.isfinite(s_).all())
                    if b == 0 or not (ok_live and ok_dead and ok_nm and finite):
                        print(f"[moe] B1 L{layer} M={M} c={c} b{b}: live == dense {ok_live}, inactive == add_partial "
                              f"{ok_dead}, unmasked == dense {ok_nm}, finite {finite}")
                    if not (ok_live and ok_dead and ok_nm and finite):
                        failures.append(f"L{layer} M{M} c{c} b{b}: live {ok_live} dead {ok_dead} nomask {ok_nm} "
                                        f"finite {finite}")
                    _free([x_tt, add, lm])

        # ---- T64 rows == T32 rows (sparse, all live) ----
        moe.decode_experts = "sparse"
        x64 = xs[torch.randperm(n, generator=g)[:64]]
        halves = []
        for half in (x64[:32], x64[32:]):
            x_tt = upload_lanes(half, cfg, mesh_device)
            o = moe.forward_decode(x_tt)
            halves.append(lanes_of(o, 32)[0])
            _free([o, x_tt])
        x16 = upload_rows16(x64, cfg, mesh_device)
        o = moe.forward_decode(x16)
        r64 = lanes_of(o, 64)[0]
        _free([o, x16])
        t64_eq = bool(torch.equal(r64[:32], halves[0])) and bool(torch.equal(r64[32:], halves[1]))
        print(f"[moe] B1 L{layer}: T64 rows == T32 rows bitwise {t64_eq}")
        if not t64_eq:
            failures.append(f"L{layer}: T64 rows != T32 rows")

        # ---- trace: persistent x / lane mask, new tokens and live sets per replay ----
        if layer == 2:
            x_dev = upload_lanes(xs[:32], cfg, mesh_device)
            lm_dev = ttnn.to_device(mask_host(list(range(32)), 32), mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            o_e = moe.forward_decode(x_dev, lane_mask=lm_dev)  # eager warm (compiles)
            _free(o_e)
            with _Capture(mesh_device) as cap:
                out_t = moe.forward_decode(x_dev, lane_mask=lm_dev)
            try:
                for it, c in enumerate((1, 8, 32, 3)):
                    live = sorted(torch.randperm(32, generator=g)[:c].tolist())
                    x = xs[0].expand(32, H).clone()
                    x[live] = xs[torch.randperm(n, generator=g)[:c]]
                    ttnn.copy_host_to_device_tensor(upload_lanes(x, cfg, mesh_device, device=False), x_dev)
                    ttnn.copy_host_to_device_tensor(mask_host(live, 32), lm_dev)
                    ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                    t1 = device_tensors_to_torch(out_t, mesh_device)
                    ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                    t2 = device_tensors_to_torch(out_t, mesh_device)
                    o_e = moe.forward_decode(x_dev, lane_mask=lm_dev)
                    te = device_tensors_to_torch(o_e, mesh_device)
                    _free(o_e)
                    moe.decode_experts = "dense"
                    o_d = moe.forward_decode(x_dev)
                    td = lanes_of(o_d, 32)[0]
                    _free(o_d)
                    moe.decode_experts = "sparse"
                    eq_e, det = bool(torch.equal(t1, te)), bool(torch.equal(t1, t2))
                    t1l = lanes_of(out_t, 32)[0]
                    eq_d = bool(torch.equal(t1l[live], td[live]))
                    print(f"[moe] B1 trace replay {it} (c={c}): traced == eager {eq_e}, 2 replays equal {det}, "
                          f"live == dense {eq_d}")
                    if not (eq_e and det and eq_d):
                        failures.append(f"trace replay {it}: eager {eq_e} det {det} dense {eq_d}")
            finally:
                ttnn.release_trace(mesh_device, cap.tid)
                _free(out_t)
            # traced cost (informational): dense vs sparse, same tokens, 1 / 8 / 32 live lanes
            for c in (1, 8, 32):
                live = list(range(c))
                x = xs[0].expand(32, H).clone()
                x[live] = xs[torch.randperm(n, generator=g)[:c]]
                ttnn.copy_host_to_device_tensor(upload_lanes(x, cfg, mesh_device, device=False), x_dev)
                ttnn.copy_host_to_device_tensor(mask_host(live, 32), lm_dev)
                st = {}
                for mode in ("dense", "sparse"):
                    moe.decode_experts = mode
                    st[mode] = traced_stats(mesh_device, lambda: moe.forward_decode(x_dev, lane_mask=lm_dev), n=8,
                                            reps=5, adapt_to=16)
                print(f"[moe] B1 L{layer} traced c={c}: dense {fmt_traced(st['dense'])} us, sparse "
                      f"{fmt_traced(st['sparse'])} us per call (M6: 1035 / 524 / 631 / 904)")
            _free([x_dev, lm_dev])
        moe.deallocate()
    assert not failures, failures


def _chips(t):
    """Every chip's local tensor (fp64), in device order."""
    return [ttnn.to_torch(c).double() for c in ttnn.get_device_tensors(t)]


def _pn_golden_from_consts(gu, w, D, Ec, b):
    """fp64 grouped PolyNorm from the device constants of one chip: ``gu [1, E, M, 2 I]``, ``w [1, E, M, 1]``,
    ``D, Ec [3, E, 1, 1]`` (moment order g^2, g^4, g^6), ``b [1, E, 1, 1]`` -> ``(sum_m a_m g^m + b) (w u)`` with
    ``a_m = (s_m D_m + E_m)^-1/2 = c_m rsqrt(mean(g^(2m)) + eps)``."""
    I2 = gu.shape[-1] // 2
    g, u = gu[..., :I2], gu[..., I2:]
    D, Ec, b = D.reshape(1, 3, -1, 1, 1), Ec.reshape(1, 3, -1, 1, 1), b.reshape(1, -1, 1, 1)
    poly = b.expand(g.shape).clone()
    for m in range(3):
        s_m = (g ** (2 * (m + 1))).sum(-1, keepdim=True)
        a_m = (s_m * D[:, m] + Ec[:, m]).rsqrt()
        poly = poly + a_m * g ** (m + 1)
    return poly * (w * u)


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_fused_polynorm(mesh_device, device_params):
    """B3 (docs/OPTIMIZATION_PLAN.md §3.3; ``moe_polynorm="fused"``; prototype logs/opt/phaseA/M10), real layer-2 /
    layer-35 weights from the serving TT cache on the T64 config, real router inputs, every chip checked:

    * kernel (after B1's sparsity ops, the regression case of a mailbox-based build): every value finite; the fused
      ``h`` vs an fp64 golden built from each chip's own device constants (``D``, ``E``, clamped ``b``; asserted on
      the routed (expert, token) rows: PCC >= 0.99999 per chip and every row >= 0.9999) and vs the composite ``h`` (asserted: at
      most one bf16 rounding apart, |d| <= 2^-7 |h| + 1e-6, and < 0.1 % of the values differ; counts reported); rows
      with routing weight 0 exactly 0; two calls bitwise equal; M = 64: gate_up rows 0..31 == M = 32 and fused rows
      0..31 == the M = 32 call bitwise;
    * module: ``forward_decode`` fused vs composite at M = 32 and the T64 step (PCC >= 0.9999 every lane row), T64
      rows == the two T32 calls bitwise (fused), B1 sparse + fused (unmasked) == dense + fused bitwise; trace capture
      with persistent ``x``, replays with new tokens == eager bitwise, two replays equal;
    * static CBs end below an L1 pin; traced cost (informational): PolyNorm composite vs fused at M = 32 / 64 and
      ``forward_decode`` composite vs fused."""
    from models.demos.motif3.tt.ccl import MotifCCL, device_tensors_to_torch, log_fabric

    cfg = t64_cfg(mesh_device)
    fab = log_fabric(mesh_device, "moe_fused_polynorm")
    assert str(fab.get("committed")) == "TORUS_XY", fab
    ccl = MotifCCL(mesh_device, cfg)
    data = load_real_inputs()
    pin = l1_pin(mesh_device)
    L1 = ttnn.L1_MEMORY_CONFIG
    failures, rows = [], []
    g = torch.Generator().manual_seed(77)

    def note(msg, ok=True):
        rows.append(msg)
        print(f"[moe] B3 {msg}")
        if not ok:
            failures.append(msg)

    for layer in (2, 35):
        if layer not in data["layers"]:
            continue
        moe = moe_from_tt_cache(mesh_device, cfg, ccl, layer, moe_polynorm="fused")
        assert moe.moe_polynorm == "fused" and moe.pn_fused is not None and moe.decode_rows == (32, 64)
        fused = moe.pn_fused
        xs = data["layers"][layer]["x"].float()
        n = xs.shape[0]
        consts = [_chips(t) for t in (moe.pn_consts.D, moe.pn_consts.E, moe.pn_consts.c["fp32"]["b"])]
        tok = torch.randperm(n, generator=g)[:64]
        h32 = None
        for M in (32, 64):
            f = upload_replicated(xs[tok[:M]], mesh_device)
            idx, w = moe.router(f, scale=moe.internal_route_scale, memory_config=L1)
            w_loc = moe.local_weights(idx, w, memory_config=L1)
            x12 = ttnn.repeat(f, ttnn.Shape([1, moe.e_loc, 1, 1]), memory_config=L1)
            pc = moe.pc_gate_up if M == 32 else moe.pc_wide[64][0]
            gu = ttnn.matmul(x12, moe.w_gate_up, program_config=pc, compute_kernel_config=moe.ckc_experts,
                             dtype=ttnn.float32, memory_config=L1)
            hc = moe.polynorm(gu, mode="fp32", row_scale=w_loc, memory_config=L1)
            # B1's sparsity ops (max, fp32 -> bf16 typecast, to_layout) first: a read_tile_value build of the kernel
            # returned all-NaN h after them (stale compute-thread mailbox state); the constants now come as tiles
            _free(moe.decode_sparsity(w_loc, memory_config=L1))
            hf = fused(gu, w_loc, memory_config=L1)
            hf2 = fused(gu, w_loc, memory_config=L1)
            gus, ws, hcs, hfs, hf2s = (_chips(t) for t in (gu, w_loc, hc, hf, hf2))
            worst_pcc, worst_tok, n_diff, n_tot, max_ulp_viol, zero_ok, finite = 1.0, 1.0, 0, 0, 0.0, True, True
            for i in range(len(gus)):  # NaN-safe: a non-finite value fails "finite", never slips through min / max
                finite &= bool(torch.isfinite(hfs[i]).all()) and bool(torch.isfinite(hcs[i]).all())
                want = _pn_golden_from_consts(gus[i], ws[i], consts[0][i], consts[1][i], consts[2][i])
                routed = (ws[i] != 0).reshape(-1)  # rows of (expert, token) with a routing weight (else h == 0)
                st = stats(want.reshape(-1, I)[routed], hfs[i].reshape(-1, I)[routed])
                worst_pcc = st["pcc"] if not st["pcc"] >= worst_pcc else worst_pcc
                worst_tok = st["min_token_pcc"] if not st["min_token_pcc"] >= worst_tok else worst_tok
                d = (hfs[i] - hcs[i]).abs()
                n_diff += int((d != 0).sum())
                n_tot += d.numel()
                v = float((d - (hcs[i].abs() * 2.0**-7 + 1e-6)).max())
                max_ulp_viol = v if not v <= max_ulp_viol else max_ulp_viol
                zero_ok &= bool((hfs[i][(ws[i] == 0).expand_as(hfs[i])] == 0).all())
            det = all(torch.equal(a, b) for a, b in zip(hfs, hf2s))
            note(f"L{layer} M={M} kernel: vs fp64 worst chip PCC {worst_pcc:.8f}, worst token {worst_tok:.7f}; vs "
                 f"composite {n_diff} / {n_tot} values differ ({n_diff / n_tot:.2e}), beyond 1 bf16 rounding "
                 f"{max_ulp_viol:.3g}; w = 0 rows exactly 0 {zero_ok}; all finite {finite}; 2 calls bitwise {det}",
                 finite and worst_pcc >= 0.99999 and worst_tok >= 0.9999 and max_ulp_viol <= 0
                 and n_diff / n_tot < 1e-3 and zero_ok and det)
            if M == 32:
                h32, gu32 = hfs, gus
                st_c = traced_stats(mesh_device, lambda: moe.polynorm(gu, mode="fp32", row_scale=w_loc,
                                                                      memory_config=L1), n=16, reps=5, adapt_to=64)
            else:
                same_gu = all(torch.equal(a[:, :, :32], b) for a, b in zip(gus, gu32))
                same_h = all(torch.equal(a[:, :, :32], b) for a, b in zip(hfs, h32))
                note(f"L{layer} M=64 rows 0..31: gate_up == M=32 {same_gu}, fused h == M=32 call {same_h}",
                     (not same_gu) or same_h)
                st_c = traced_stats(mesh_device, lambda: moe.polynorm(gu, mode="fp32", row_scale=w_loc,
                                                                      memory_config=L1), n=16, reps=5, adapt_to=64)
            st_f = traced_stats(mesh_device, lambda: fused(gu, w_loc, memory_config=L1), n=16, reps=5, adapt_to=64)
            note(f"L{layer} M={M} traced PolyNorm: composite {fmt_traced(st_c)} us, fused {fmt_traced(st_f)} us "
                 f"(M10: 127.1 -> 25.3 / 192.5 -> 56.8)")
            _free([f, idx, w, w_loc, x12, gu, hc, hf, hf2])

        # ---- module: forward_decode fused vs composite, T64 rows, sparse + fused ----
        def run(x_tt, mode, experts="dense"):
            moe.pn_fused = fused if mode == "fused" else None
            moe.decode_experts = experts
            o = moe.forward_decode(x_tt)
            full = device_tensors_to_torch(o, mesh_device)
            _free(o)
            return full

        def lane_rows(full, M):
            rws = torch.stack([full[cfg.axes.coord(dp, 0)].reshape(-1, H) for dp in range(cfg.dp)])
            if M == 32:
                return rws.reshape(32, H)
            Lr = cfg.lanes_per_row
            return torch.cat([rws[:, :Lr].reshape(32, H), rws[:, Lr:].reshape(32, H)])

        x64 = xs[torch.randperm(n, generator=g)[:64]]
        halves = []
        for half in (x64[:32], x64[32:]):
            x_tt = upload_lanes(half, cfg, mesh_device)
            oc, of = run(x_tt, "composite"), run(x_tt, "fused")
            osf = run(x_tt, "fused", "sparse")
            a, b = lane_rows(oc, 32).double(), lane_rows(of, 32).double()
            tp = row_pcc(a, b)
            fin = bool(torch.isfinite(b).all()) and bool(torch.isfinite(osf.float()).all())
            note(f"L{layer} forward_decode M=32: fused vs composite PCC {pcc(a, b):.7f}, min lane {float(tp.min()):.7f}"
                 f", sparse+fused == dense+fused {torch.equal(osf, of)}, finite {fin}",
                 fin and float(tp.min()) >= 0.9999 and torch.equal(osf, of))
            halves.append(lane_rows(of, 32))
            _free(x_tt)
        x16 = upload_rows16(x64, cfg, mesh_device)
        o64c, o64 = lane_rows(run(x16, "composite"), 64), lane_rows(run(x16, "fused"), 64)
        t64_eq = bool(torch.equal(o64[:32], halves[0])) and bool(torch.equal(o64[32:], halves[1]))
        tp = row_pcc(o64c.double(), o64.double())
        fin = bool(torch.isfinite(o64.float()).all())
        note(f"L{layer} T64 (fused): rows == T32 rows bitwise {t64_eq}; vs composite min row PCC {float(tp.min()):.7f}"
             f", finite {fin}", fin and t64_eq and float(tp.min()) >= 0.9999)
        _free(x16)

        # ---- trace (layer 2): persistent x, new tokens per replay; traced cost composite vs fused ----
        if layer == 2:
            moe.pn_fused, moe.decode_experts = fused, "dense"
            x_dev = upload_lanes(xs[:32], cfg, mesh_device)
            _free(moe.forward_decode(x_dev))  # eager warm (compiles)
            with _Capture(mesh_device) as cap:
                out_t = moe.forward_decode(x_dev)
            try:
                for it in range(3):
                    x = xs[torch.randperm(n, generator=g)[:32]]
                    ttnn.copy_host_to_device_tensor(upload_lanes(x, cfg, mesh_device, device=False), x_dev)
                    ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                    t1 = device_tensors_to_torch(out_t, mesh_device)
                    ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
                    t2 = device_tensors_to_torch(out_t, mesh_device)
                    o_e = moe.forward_decode(x_dev)
                    te = device_tensors_to_torch(o_e, mesh_device)
                    _free(o_e)
                    note(f"trace replay {it}: traced == eager {torch.equal(t1, te)}, 2 replays equal "
                         f"{torch.equal(t1, t2)}", torch.equal(t1, te) and torch.equal(t1, t2))
            finally:
                ttnn.release_trace(mesh_device, cap.tid)
                _free(out_t)
            x16 = upload_rows16(x64, cfg, mesh_device)
            for M, xt in ((32, x_dev), (64, x16)):
                st = {}
                for mode in ("composite", "fused"):
                    moe.pn_fused = fused if mode == "fused" else None
                    st[mode] = traced_stats(mesh_device, lambda: moe.forward_decode(xt), n=8, reps=5, adapt_to=16)
                note(f"L{layer} forward_decode traced M={M}: composite {fmt_traced(st['composite'])} us, fused "
                     f"{fmt_traced(st['fused'])} us per call")
            moe.pn_fused = fused
            _free([x_dev, x16])
        moe.deallocate()
    _free(pin)
    print("[moe] B3 summary:\n[moe]   " + "\n[moe]   ".join(rows))
    assert rows, "no layer ran"
    assert not failures, failures


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_prefill(mesh_device, device_params):
    """MOE-6: masked-dense prefill, real layer-2 weights; S = 128 / 2048 real tokens, 4096 / 32768 tiled. Gates: PCC vs
    the reference routes, every position >= 0.999 on the device's routes (review issue 3), low positions only at route
    flips; at S = 2048 / 4096 also ``prefill_polynorm="fp32"`` and ``prefill_pc=False`` (bitwise equal to the
    in1-multicast default)."""
    from models.demos.motif3.tt.moe import MotifMoE
    from models.demos.motif3.tt.weights import HFWeightLoader

    layer = 2
    cfg, ccl, _ = _setup(mesh_device, "moe_prefill")
    src = HFWeightLoader()
    require_layer(src, layer)
    data = load_real_inputs()
    xs = data["layers"][layer]["x"]
    n = xs.shape[0]
    want_all = ref_routed_all_tokens(src, layer, xs)  # [n, 4096] fp32 golden of every real token (cached)
    ref = RefMoE(src, layer)
    moe = MotifMoE(mesh_device, cfg, layer, source=src, ccl=ccl, cache=False)
    sizes = [int(s) for s in os.environ.get("MOTIF3_MOE_PREFILL_S", "128,2048,4096,32768").split(",")]
    failures = []
    for S in sizes:
        tok = torch.arange(S) % n
        x = xs[tok]
        x_tt = upload_replicated(x, mesh_device)
        try:
            t0 = time.perf_counter()
            out = moe.forward_prefill(x_tt)
            ttnn.synchronize_device(mesh_device)
            first_s = time.perf_counter() - t0
            chips = None if S <= 4096 else [0, 11, 22, 31]
            got, same = read_replicated(out, mesh_device, chips=chips)
            got = got.reshape(S, H).float()
            _free(out)
            routes = prefill_device_routes(moe, x_tt)
            _check_prefill(moe, ref, xs, tok, got, want_all[tok], routes, failures, f"prefill S={S}")
            us = eager_us(mesh_device, lambda: moe.forward_prefill(x_tt), iters=2 if S <= 4096 else 1, warmup=0)
            print(f"[moe] prefill S={S}: identical on {'all 32' if chips is None else chips} chips: {same}; input "
                  f"still allocated {x_tt.is_allocated()}; eager {us / 1e3:.1f} ms (first call incl. compile "
                  f"{first_s:.1f} s)")
            if not same or not x_tt.is_allocated():
                failures.append(f"prefill S={S}: replicas {same} input allocated {x_tt.is_allocated()}")
            if S == 128:
                # contracts: add_partial ([1,1,S/4,4096] TP partial of this DP row's rows) and reduce_tp=False
                R, Cc = cfg.axes.mesh_shape
                g = torch.Generator().manual_seed(11)
                extra = (0.05 * torch.randn(R, Cc, S // 4, H, generator=g)).bfloat16()
                ex_tt = upload_per_chip(extra, mesh_device)
                out2 = moe.forward_prefill(x_tt, add_partial=ex_tt)
                got2, same2 = read_replicated(out2, mesh_device)
                add = torch.cat([sum(extra[cfg.axes.coord(dp, tp)].float() for tp in range(cfg.tp))
                                 for dp in range(cfg.dp)])  # [S, 4096]: row block dp gets sum_tp extra[dp, tp]
                s_add = stats(got + add, got2.reshape(S, H).float())
                part = moe.forward_prefill(x_tt, reduce_tp=False)  # [1,1,S/4,4096] per chip (DP-reduced partials)
                from models.demos.motif3.tt.ccl import device_tensors_to_torch

                pf = device_tensors_to_torch(part, mesh_device).float().reshape(R, Cc, S // 4, H)
                psum = torch.cat([sum(pf[cfg.axes.coord(dp, tp)] for tp in range(cfg.tp)) for dp in range(cfg.dp)])
                s_part = stats(got, psum)
                print(f"[moe] prefill contracts S={S}: add_partial {fmt(s_add)} replicas {same2}; "
                      f"sum_tp(reduce_tp=False) vs output {fmt(s_part)}; add_partial still allocated "
                      f"{ex_tt.is_allocated()}")
                if s_add["pcc"] < 0.9999 or not same2 or s_part["pcc"] < 0.9999 or not ex_tt.is_allocated():
                    failures.append(f"prefill contracts: add {s_add['pcc']:.6f} part {s_part['pcc']:.6f} {same2}")
                _free([ex_tt, out2, part])
            if S in (2048, 4096):
                # prefill_pc=False (auto matmul configs): bitwise equal to the in1-multicast default
                moe.prefill_pc = False
                out_a = moe.forward_prefill(x_tt)
                moe.prefill_pc = True
                got_a, _ = read_replicated(out_a, mesh_device, chips=[0])
                _free(out_a)
                same_a = bool(torch.equal(got_a.reshape(S, H).float(), got))
                print(f"[moe] prefill S={S} prefill_pc=False (auto configs) bitwise equal to the default: {same_a}")
                if not same_a:
                    failures.append(f"prefill S={S}: auto-config output differs from the in1-multicast default")
            if S == 2048:
                # prefill_polynorm="fp32" (fp32 gate_up output through the in1-multicast config); same routes
                moe.prefill_polynorm = "fp32"
                try:
                    out_f = moe.forward_prefill(x_tt)
                    got_f, same_f = read_replicated(out_f, mesh_device, chips=[0, 31])
                    _free(out_f)
                    _check_prefill(moe, ref, xs, tok, got_f.reshape(S, H).float(), want_all[tok], routes, failures,
                                   f"prefill S={S} prefill_polynorm=fp32")
                    if not same_f:
                        failures.append(f"prefill S={S} fp32 polynorm: replicas differ")
                finally:
                    moe.prefill_polynorm = "bf16"
            if S == int(os.environ.get("MOTIF3_MOE_PREFILL_BREAKDOWN_S", "4096")):
                print(f"[moe] prefill S={S} eager breakdown (ms): " + ", ".join(
                    f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                    for k, v in _prefill_breakdown(mesh_device, moe, x_tt).items()))
        except Exception as e:
            failures.append(f"prefill S={S}: {type(e).__name__}: {str(e)[:500]}")
            print(f"[moe] prefill S={S}: FAILED {type(e).__name__}: {str(e)[:500]}")
        _free(x_tt)
    moe.deallocate()
    assert not failures, "\n".join(failures)


def _chips_equal(a, b):
    """(mismatching values, all values) between two per-chip tensors of the same shape, over all chips."""
    sa, sb = ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)
    bad = n = 0
    for x, y in zip(sa, sb):
        tx, ty = ttnn.to_torch(x).float(), ttnn.to_torch(y).float()
        bad += int((tx != ty).sum()) if tx.shape == ty.shape else tx.numel()
        n += tx.numel()
    return bad, n


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_prefill_compact(mesh_device, device_params):
    """B2a (docs/OPTIMIZATION_PLAN.md §3.3 B2; ``prefill_moe="compact"``; prototype logs/opt/phaseA/m7), real weights of
    layers 2 and 35 (``MOTIF3_B2A_LAYERS``) from the serving TT cache, real router inputs tiled to S = 1024 / 2048 /
    4096 / 8192 (``MOTIF3_B2A_S``):

    * the compacted local partial of every chunk == the dense one on all 32 chips (0 mismatching values), and
      ``forward_prefill`` compacted == dense (replicated output, every chip);
    * a second compacted run is bitwise equal (determinism); the chunks really ran compacted (state counters);
    * after ``frozen`` turns on, an unwarmed shape falls back to the dense path (same output), and after
      ``warm_compact`` the same chunk runs compacted again;
    * eager ``forward_prefill`` time, dense vs compacted (informational)."""
    from models.demos.motif3.tt.moe import compact_block, compact_ladder

    cfg, ccl, fab = _setup(mesh_device, "moe_prefill_compact")
    assert "TORUS_XY" in str(fab.get("committed")), fab
    data = load_real_inputs()
    layers = [int(v) for v in os.environ.get("MOTIF3_B2A_LAYERS", "2,35").split(",")]
    sizes = [int(v) for v in os.environ.get("MOTIF3_B2A_S", "1024,2048,4096,8192").split(",")]
    failures = []
    for layer in layers:
        moe = moe_from_tt_cache(mesh_device, cfg, ccl, layer, prefill_moe="compact")
        st = moe.compact_state
        xs = data["layers"][layer]["x"]
        n = xs.shape[0]
        try:
            for S in sizes:
                tag = f"L{layer} S={S}"
                x_tt = upload_replicated(xs[torch.arange(S) % n], mesh_device)
                C = min(S, moe.prefill_chunk)
                xc = x_tt if C == S else ttnn.slice(x_tt, [0, 0, 0, 0], [1, 1, C, H])
                try:
                    moe.prefill_moe = "dense"
                    pd = moe.local_partial(xc, polynorm=moe.prefill_polynorm, decode=False)
                    od = moe.forward_prefill(x_tt)
                    moe.prefill_moe = "compact"
                    n0 = st.stats["compact"]
                    pc = moe.local_partial(xc, polynorm=moe.prefill_polynorm, decode=False)
                    oc = moe.forward_prefill(x_tt)
                    oc2 = moe.forward_prefill(x_tt)
                    ran = st.stats["compact"] - n0
                    bad_p, n_p = _chips_equal(pd, pc)
                    bad_o, n_o = _chips_equal(od, oc)
                    bad_r, _ = _chips_equal(oc, oc2)
                    shapes = sorted(k for k in st.blocks if k[0] == C)
                    print(f"[moe] B2a {tag}: compacted chunks {ran} (want {1 + 2 * (S // C)}), shapes {shapes}; local "
                          f"partial mismatches {bad_p}/{n_p}; forward_prefill mismatches {bad_o}/{n_o}; rerun "
                          f"mismatches {bad_r}")
                    if ran != 1 + 2 * (S // C) or bad_p or bad_o or bad_r:
                        failures.append(f"{tag}: ran {ran} partial {bad_p} fwd {bad_o} rerun {bad_r}")
                    _free([pd, od, pc, oc, oc2])
                    # frozen: an unwarmed shape runs dense (same output); warm_compact makes it compacted again
                    mb = compact_block(C, moe.prefill_moe_block)
                    st.warmed.clear()
                    st.frozen = lambda: True
                    u0, c0 = st.stats["dense_unwarmed"], st.stats["compact"]
                    pf = moe.local_partial(xc, polynorm=moe.prefill_polynorm, decode=False)
                    fell = st.stats["dense_unwarmed"] - u0 == 1 and st.stats["compact"] == c0
                    t0 = time.perf_counter()
                    ladder = moe.warm_compact(C)
                    warm_s = time.perf_counter() - t0
                    pw = moe.local_partial(xc, polynorm=moe.prefill_polynorm, decode=False)
                    ran_w = st.stats["compact"] - c0 == 1
                    st.frozen = lambda: False
                    bad_f, _ = _chips_equal(pf, pw)
                    print(f"[moe] B2a {tag}: frozen + unwarmed -> dense {fell}; warm_compact({C}) ladder {ladder} "
                          f"(mb {mb}) in {warm_s:.1f} s, then compacted {ran_w}; mismatches {bad_f}")
                    if not (fell and ran_w) or bad_f or ladder != compact_ladder(C, mb, moe.e_loc):
                        failures.append(f"{tag}: frozen fallback {fell} warmed {ran_w} mismatches {bad_f}")
                    _free([pf, pw])
                    moe.prefill_moe = "dense"
                    t_d = eager_us(mesh_device, lambda: moe.forward_prefill(x_tt), iters=2)
                    moe.prefill_moe = "compact"
                    t_c = eager_us(mesh_device, lambda: moe.forward_prefill(x_tt), iters=2)
                    print(f"[moe] B2a {tag}: eager forward_prefill dense {t_d / 1e3:.2f} ms, compacted {t_c / 1e3:.2f} "
                          f"ms (x{t_c / t_d:.3f})")
                except Exception as e:
                    failures.append(f"{tag}: {type(e).__name__}: {str(e)[:500]}")
                    print(f"[moe] B2a {tag}: FAILED {type(e).__name__}: {str(e)[:500]}")
                    import traceback

                    traceback.print_exc()
                finally:
                    if xc is not x_tt:
                        _free(xc)
                    _free(x_tt)
            print(f"[moe] B2a L{layer}: state counters {st.stats}, shapes {dict(sorted(st.blocks.items()))}")
        finally:
            moe.deallocate()
    assert not failures, "\n".join(failures)


B2B_COMBOS = (("host", "matmul"), ("host", "gather"), ("device", "matmul"), ("device", "gather"))


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_prefill_compact_kernels(mesh_device, device_params):
    """B2b (docs/OPTIMIZATION_PLAN.md §3.3 B2; ``MOTIF3_PREFILL_MOE_DISPATCH`` / ``MOTIF3_PREFILL_MOE_COMBINE``;
    logs/opt/phaseB/B2b), real weights of layers 2 and 35 (``MOTIF3_B2B_LAYERS``), real router inputs tiled to S = 1024
    / 2048 / 4096 (``MOTIF3_B2B_S``; a size outside ``moe.PREFILL_MOE_DEVICE_ROWS``, e.g. 3968, runs the dispatch kernel
    directly as a diagnostic while the model itself keeps such chunks on B2a's host path):

    * the dispatch kernel (``kernels.moe_compact.CompactDispatch``) writes exactly B2a's host lists on all 32 chips
      (tokens, keys, block sparsity and PolyNorm words, the rows' routing weights == ``w_loc`` bits, need / NB), from
      ``w_loc`` and from the router's ``w`` alike;
    * every (dispatch, combine) combination: the compacted local partial == the dense one on all 32 chips, and
      ``forward_prefill`` == dense; a second run is bitwise equal; the chunks ran compacted (state counters; the device
      dispatch counted);
    * device dispatch, frozen state: an unwarmed shape falls back to dense (same output); ``warm_compact`` makes it
      compacted again;
    * a chunk of one repeated token (every route on the same 8 experts) exceeds the ladder's cap: the device dispatch
      reports NB = 0 and the dense path runs (same output);
    * eager ``forward_prefill`` time per combination (informational)."""
    from models.demos.motif3.tt.kernels.moe_compact import dispatch_reference
    from models.demos.motif3.tt.moe import b2b_rows_ok, compact_block, compact_ladder, compact_upload_fast

    cfg, ccl, fab = _setup(mesh_device, "moe_prefill_compact_kernels")
    assert "TORUS_XY" in str(fab.get("committed")), fab
    data = load_real_inputs()
    layers = [int(v) for v in os.environ.get("MOTIF3_B2B_LAYERS", "2,35").split(",")]
    sizes = [int(v) for v in os.environ.get("MOTIF3_B2B_S", "1024,2048,4096").split(",")]
    R_, C_ = tuple(mesh_device.shape)
    P = R_ * C_
    failures = []

    def bits(t):
        t = t.contiguous()
        if t.dtype == torch.float32:
            return t.view(torch.int32).long()
        if t.dtype == torch.bfloat16:
            return t.view(torch.int16).long() & 0xFFFF
        return t.long()

    def chips(t):
        from models.demos.motif3.tt.ccl import device_tensors_to_torch

        return device_tensors_to_torch(t, mesh_device)

    def set_mode(moe, d, c):
        moe.prefill_moe_dispatch, moe.prefill_moe_combine = d, c
        moe.prepare_compact()

    for layer in layers:
        moe = moe_from_tt_cache(mesh_device, cfg, ccl, layer, prefill_moe="compact", prefill_moe_dispatch="device",
                                prefill_moe_combine="gather")
        st = moe.compact_state
        xs = data["layers"][layer]["x"]
        n = xs.shape[0]
        try:
            for S in sizes:
                tag = f"L{layer} S={S}"
                x_tt = upload_replicated(xs[torch.arange(S) % n], mesh_device)
                C = min(S, moe.prefill_chunk)
                xc = x_tt if C == S else ttnn.slice(x_tt, [0, 0, 0, 0], [1, 1, C, H])
                try:
                    # --- the dispatch kernel vs the host lists ---
                    set_mode(moe, "device", "gather")
                    mb = compact_block(C, moe.prefill_moe_block)
                    ladder = compact_ladder(C, mb, moe.e_loc)
                    idx, w = moe.router(xc, scale=moe.internal_route_scale)
                    w_loc = moe.local_weights(idx, w)
                    hi = moe._read_routes(idx, C)
                    need, nb, u = compact_upload_fast(hi, st.g2s, P, moe.e_loc, mb, ladder, moe._pn_bits, C)
                    bad_k = []
                    for src, is_loc in ((w_loc, True), (w, False)):
                        out = st.dispatch(idx, src, moe._disp_meta, M=C, mb=mb, ladder=ladder, w_is_loc=is_loc)
                        nd = chips(out.need).reshape(P, 8).long()
                        if not (bool((nd[:, 0] == need).all()) and bool((nd[:, 1] == (nb or 0)).all())):
                            bad_k.append(f"need {nd[:2, :3].tolist()} want {need} / {nb}")
                        if nb is not None:
                            R = nb * mb
                            tix, key, sp, blk = dispatch_reference(u, mb, nb, C)
                            rw = chips(out.rows).reshape(P, 2, -1).long()
                            spd = bits(chips(out.sp).reshape(P, -1, 32)[:, :nb])
                            blkd = bits(chips(out.blk).reshape(P, -1, 32)[:, :nb])
                            wl = chips(w_loc).reshape(P, moe.e_loc, C).float()
                            cnt = np.bincount(st.g2s[hi.reshape(-1)], minlength=P * moe.e_loc).reshape(P, moe.e_loc)
                            ends = np.cumsum((cnt + mb - 1) // mb, axis=1)
                            eb = np.minimum((ends[:, :, None] <= np.arange(nb)[None, None, :]).sum(1), moe.e_loc - 1)
                            erow = torch.from_numpy(eb).long().repeat_interleave(mb, dim=1)
                            wref = torch.where(key < C, wl[torch.arange(P)[:, None], erow, tix.clamp(max=C - 1)],
                                               torch.zeros(()))
                            wcd = chips(out.wcol).reshape(P, -1)[:, :R].float()
                            for name, ok in (("tix", torch.equal(rw[:, 0, :R], tix)),
                                             ("key", torch.equal(rw[:, 1, :R], key)), ("sp", torch.equal(spd, sp)),
                                             ("blk", torch.equal(blkd, blk)), ("w", torch.equal(bits(wcd), bits(wref)))):
                                if not ok:
                                    bad_k.append(f"{name} ({'w_loc' if is_loc else 'w'})")
                        out.free()
                    # a ladder whose cap is below the need: NB = 0 (the dense path), the need still exact
                    small = tuple(v for v in ladder if v < need) or None
                    if small:
                        out = st.dispatch(idx, w, moe._disp_meta, M=C, mb=mb, ladder=small, w_is_loc=False)
                        nd = chips(out.need).reshape(P, 8).long()
                        if not (bool((nd[:, 0] == need).all()) and bool((nd[:, 1] == 0).all())):
                            bad_k.append(f"overflow need {nd[:2, :3].tolist()} want {need} / 0")
                        out.free()
                    _free([idx, w, w_loc])
                    print(f"[moe] B2b {tag}: dispatch kernel need {need} NB {nb} (cap {small[-1] if small else None}"
                          f" -> NB 0): "
                          f"{'exact' if not bad_k else 'MISMATCH ' + ', '.join(bad_k)}")
                    if bad_k:
                        failures.append(f"{tag}: dispatch kernel {bad_k}")
                    # --- every combination vs dense ---
                    moe.prefill_moe = "dense"
                    pd = moe.local_partial(xc, polynorm=moe.prefill_polynorm, decode=False)
                    od = moe.forward_prefill(x_tt)
                    moe.prefill_moe = "compact"
                    for d, c in B2B_COMBOS:
                        set_mode(moe, d, c)
                        n0, n1 = st.stats["compact"], st.stats["device_dispatch"]
                        pc = moe.local_partial(xc, polynorm=moe.prefill_polynorm, decode=False)
                        oc = moe.forward_prefill(x_tt)
                        oc2 = moe.forward_prefill(x_tt)
                        ran = st.stats["compact"] - n0
                        ran_d = st.stats["device_dispatch"] - n1
                        want_d = ran if d == "device" and b2b_rows_ok(C) else 0
                        bad_p, n_p = _chips_equal(pd, pc)
                        bad_o, n_o = _chips_equal(od, oc)
                        bad_r, _ = _chips_equal(oc, oc2)
                        print(f"[moe] B2b {tag} {d}/{c}: compacted chunks {ran} (device {ran_d}, want {1 + 2 * (S // C)});"
                              f" partial mismatches {bad_p}/{n_p}; forward_prefill {bad_o}/{n_o}; rerun {bad_r}")
                        if ran != 1 + 2 * (S // C) or ran_d != want_d or bad_p or bad_o or bad_r:
                            failures.append(f"{tag} {d}/{c}: ran {ran} dev {ran_d} partial {bad_p} fwd {bad_o} "
                                            f"rerun {bad_r}")
                        _free([pc, oc, oc2])
                    _free([pd, od])
                    # --- device dispatch: frozen fallback, warm-up ---
                    set_mode(moe, "device", "gather")
                    pw0 = moe.local_partial(xc, polynorm=moe.prefill_polynorm, decode=False)
                    st.warmed.clear()
                    st.frozen = lambda: True
                    u0, c0 = st.stats["dense_unwarmed"], st.stats["compact"]
                    pf = moe.local_partial(xc, polynorm=moe.prefill_polynorm, decode=False)
                    fell = st.stats["dense_unwarmed"] - u0 == 1 and st.stats["compact"] == c0
                    t0 = time.perf_counter()
                    lad = moe.warm_compact(C)
                    warm_s = time.perf_counter() - t0
                    pw = moe.local_partial(xc, polynorm=moe.prefill_polynorm, decode=False)
                    ran_w = st.stats["compact"] - c0 == 1
                    st.frozen = lambda: False
                    bad_f, _ = _chips_equal(pf, pw)
                    bad_f0, _ = _chips_equal(pw0, pw)
                    print(f"[moe] B2b {tag}: frozen + unwarmed -> dense {fell}; warm_compact({C}) ladder {lad} in "
                          f"{warm_s:.1f} s, then compacted {ran_w}; mismatches {bad_f} / {bad_f0}")
                    if not (fell and ran_w) or bad_f or bad_f0 or st.warmed != {(C, mb, b) for b in lad}:
                        failures.append(f"{tag}: frozen fallback {fell} warmed {ran_w} mismatches {bad_f} {bad_f0}")
                    _free([pw0, pf, pw])
                    # --- eager timing per combination (informational) ---
                    moe.prefill_moe = "dense"
                    t_d = eager_us(mesh_device, lambda: moe.forward_prefill(x_tt), iters=2)
                    moe.prefill_moe = "compact"
                    ts = []
                    for d, c in B2B_COMBOS:
                        set_mode(moe, d, c)
                        ts.append(f"{d}/{c} {eager_us(mesh_device, lambda: moe.forward_prefill(x_tt), iters=2) / 1e3:.2f}")
                    print(f"[moe] B2b {tag}: eager forward_prefill ms: dense {t_d / 1e3:.2f}; " + "; ".join(ts))
                except Exception as e:
                    failures.append(f"{tag}: {type(e).__name__}: {str(e)[:500]}")
                    print(f"[moe] B2b {tag}: FAILED {type(e).__name__}: {str(e)[:500]}")
                    import traceback

                    traceback.print_exc()
                finally:
                    if xc is not x_tt:
                        _free(xc)
                    _free(x_tt)
            # --- beyond the cap: one repeated token routes every row to the same 8 experts ---
            Sx = min(sizes)
            x_rep = upload_replicated(xs[:1].expand(Sx, -1).contiguous(), mesh_device)
            try:
                set_mode(moe, "device", "gather")
                moe.prefill_moe = "dense"
                pd = moe.local_partial(x_rep, polynorm=moe.prefill_polynorm, decode=False)
                moe.prefill_moe = "compact"
                ir, wr = moe.router(x_rep, scale=moe.internal_route_scale)
                mbx = compact_block(Sx, moe.prefill_moe_block)
                need_x = compact_upload_fast(moe._read_routes(ir, Sx), st.g2s, P, moe.e_loc, mbx,
                                             compact_ladder(Sx, mbx, moe.e_loc), moe._pn_bits, Sx)[0]
                _free([ir, wr])
                want_cap = int(need_x > compact_ladder(Sx, mbx, moe.e_loc)[-1])
                k0 = st.stats["dense_cap"]
                pc = moe.local_partial(x_rep, polynorm=moe.prefill_polynorm, decode=False)
                capped = st.stats["dense_cap"] - k0
                bad, _ = _chips_equal(pd, pc)
                print(f"[moe] B2b L{layer} repeated token S={Sx}: need {need_x} blocks, dense_cap {capped} (want "
                      f"{want_cap}), mismatches {bad}")
                if capped != want_cap or bad:
                    failures.append(f"L{layer} repeated token: dense_cap {capped} mismatches {bad}")
                _free([pd, pc])
            finally:
                _free(x_rep)
            print(f"[moe] B2b L{layer}: state counters {st.stats}")
        finally:
            moe.deallocate()
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_block_shared_real(mesh_device, device_params):
    """The MoE block as the decoder calls it (review issue 4): ``MotifMoE`` + the real shared expert
    (``tt/mlp.py`` ``MotifSharedExpert``, another agent's module; its TP partial ``all_reduce=False`` handed in as
    ``add_partial`` so one ``ar_tp`` closes both) vs the reference MoE block (routed + shared in fp32, HF
    ``MoE.forward``), real layer-2 weights; decode (32 lanes) and prefill S = 128 / 2048 (shared expert on
    ``dp_slice``)."""
    from models.demos.motif3.tt.mlp import MotifSharedExpert
    from models.demos.motif3.tt.moe import MotifMoE
    from models.demos.motif3.tt.weights import HFWeightLoader

    cfg, ccl, _ = _setup(mesh_device, "moe_block_shared_real")
    src = HFWeightLoader()
    L = 2
    require_layer(src, L)
    xs = load_real_inputs()["layers"][L]["x"]
    n = xs.shape[0]
    moe = MotifMoE(mesh_device, cfg, L, source=src, ccl=ccl, cache=False)
    shared = MotifSharedExpert(mesh_device, cfg, L, source=src, ccl=ccl, cache=False)
    ref = RefMoE(src, L)
    shref = ref_shared_expert(src, L)
    failures = []
    # decode: 32 lanes; shared partial first, then the MoE closes both with one ar_tp
    x32 = xs[spread_tokens(n, 32)]
    routed, idx_r, _ = ref(x32)
    sh = shref(x32.float()).float()
    want = routed + sh
    x_tt = upload_lanes(x32, cfg, mesh_device)
    part = shared.forward_decode(x_tt, all_reduce=False)
    taps = {}
    out = moe.forward_decode(x_tt, add_partial=part, taps=taps)
    got, same = read_rows(out, cfg, mesh_device)
    idx_d, w_d, _ = read_routes(taps, moe, mesh_device, chips=[0])
    w_d = w_d.double() * (moe.route_scale / moe.internal_route_scale)
    want_dev = ref(x32, idx_d, w_d.float())[0] + sh
    s, s_dev = stats(want, got), stats(want_dev, got)
    alive = x_tt.is_allocated() and part.is_allocated()
    print(f"[moe] MoE block decode (routed + shared via add_partial) vs reference MoE: {fmt(s)}; on device routes "
          f"{fmt(s_dev)}; replicas {same}; inputs still allocated {alive}")
    if s["pcc"] < 0.995 or s_dev["min_token_pcc"] < TOKEN_PCC_MIN or not same or not alive:
        failures.append(f"block decode: {fmt(s)} / {fmt(s_dev)} same {same} alive {alive}")
    _free([taps, x_tt, part, out])
    # prefill: shared expert on this DP row's rows (dp_slice), routed closes with add_partial
    for S in (128, 2048):
        x = xs[:S]
        tok = torch.arange(S)
        sh_p = shref(x.float()).float()
        want_p = ref(x)[0] + sh_p
        xp_tt = upload_replicated(x, mesh_device)
        sl = moe.dp_slice(xp_tt)
        sp = shared.forward_prefill(sl, all_reduce=False)
        outp = moe.forward_prefill(xp_tt, add_partial=sp)
        gp, samep = read_replicated(outp, mesh_device)
        gp = gp.reshape(S, H).float()
        routes = prefill_device_routes(moe, xp_tt)
        want_pd = ref_on_routes(ref, x, tok, *routes)[0] + sh_p
        sp_, sp_dev = stats(want_p, gp), stats(want_pd, gp)
        print(f"[moe] MoE block prefill S={S} vs reference MoE: {fmt(sp_)}; on device routes {fmt(sp_dev)}; identical "
              f"on all chips {samep}")
        if sp_["pcc"] < 0.995 or sp_dev["min_token_pcc"] < TOKEN_PCC_MIN or not samep:
            failures.append(f"block prefill S={S}: {fmt(sp_)} / {fmt(sp_dev)} same {samep}")
        _free([xp_tt, sl, sp, outp])
    moe.deallocate()
    shared.deallocate()
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH_PARAMS, indirect=True)
@torch.no_grad()
def test_moe_device_prefill_matmul_variants(mesh_device, device_params):
    """Prefill expert matmuls (C = 4096 rows): batched ``[1,12,C,K] @ [1,12,K,N]`` with the auto config vs per-expert
    ``[1,1,C,K] @ [1,1,K,N]`` (weight slice + matmul) with the auto config and with 2D-multicast configs, and batched
    in1-multicast configs. The module's in1-multicast configs (:func:`prefill_experts_pc`) must run and be bitwise
    identical to the auto config (asserted); the other configs are reported."""
    from models.demos.motif3.tt.moe import MotifMoE, prefill_experts_pc
    from models.demos.motif3.tt.weights import DictWeightSource

    cfg, ccl, _ = _setup(mesh_device, "moe_prefill_matmul_variants")
    L = 2
    moe = MotifMoE(mesh_device, cfg, L, source=DictWeightSource(random_moe_tensors(L, seed=0)), ccl=ccl, cache=False)
    C = int(os.environ.get("MOTIF3_MOE_PREFILL_C", "4096"))
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    x = upload_replicated(random_inputs(C, seed=5), mesh_device)
    h = upload_replicated(torch.randn(C, I).bfloat16(), mesh_device)
    ck = moe.ckc_experts
    rows, failures = [], []

    def t(name, fn, iters=2):
        try:
            us = eager_us(mesh_device, fn, iters=iters, warmup=1)
            rows.append(f"{name}: {us / 1e3:.2f} ms")
            return us
        except Exception as e:
            rows.append(f"{name}: error {type(e).__name__}: {str(e)[:260]}")
            return None

    def chip0(tt_out):
        v = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        _free(tt_out)
        return v

    x12 = ttnn.repeat(x, ttnn.Shape([1, 12, 1, 1]), memory_config=DRAM)
    h12 = ttnn.repeat(h, ttnn.Shape([1, 12, 1, 1]), memory_config=DRAM)
    t("batched gate_up auto", lambda: ttnn.matmul(x12, moe.w_gate_up, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                                  memory_config=DRAM))
    t("batched down auto", lambda: ttnn.matmul(h12, moe.w_down, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                               memory_config=DRAM))
    w_gu0 = ttnn.slice(moe.w_gate_up, [0, 0, 0, 0], [1, 1, H, 2 * I], memory_config=DRAM)
    w_dn0 = ttnn.slice(moe.w_down, [0, 0, 0, 0], [1, 1, I, H], memory_config=DRAM)
    t("weight slice gate_up (1 expert)", lambda: ttnn.slice(moe.w_gate_up, [0, 3, 0, 0], [1, 4, H, 2 * I],
                                                            memory_config=DRAM))
    t("weight slice down (1 expert)", lambda: ttnn.slice(moe.w_down, [0, 3, 0, 0], [1, 4, I, H], memory_config=DRAM))
    t("per-expert gate_up auto (1 expert)", lambda: ttnn.matmul(x, w_gu0, compute_kernel_config=ck,
                                                                dtype=ttnn.bfloat16, memory_config=DRAM))
    t("per-expert down auto (1 expert)", lambda: ttnn.matmul(h, w_dn0, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                                             memory_config=DRAM))

    def pc2d(gx, gy, m_tiles, n_tiles, in0_bw, sub_h, sub_w, obh=None, obw=None):
        pm, pn = m_tiles // gy, n_tiles // gx
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=in0_bw,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            out_block_h=obh or pm,
            out_block_w=obw or pn,
            per_core_M=pm,
            per_core_N=pn,
            transpose_mcast=False,
            fused_activation=None,
        )

    Mt = C // 32
    for gx, gy, bw, sh, sw, obh in ((10, 8, 2, 1, 4, None), (10, 8, 4, 1, 4, 8), (10, 8, 2, 2, 2, 4), (8, 8, 4, 2, 2, 8)):
        if (2 * I // 32) % gx or Mt % gy:
            continue
        pc = pc2d(gx, gy, Mt, 2 * I // 32, bw, sh, sw, obh)
        t(f"per-expert gate_up 2D {gx}x{gy} k{bw} sub{sh}x{sw} out_block_h {obh or Mt // gy}", lambda: ttnn.matmul(
            x, w_gu0, program_config=pc, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=DRAM))
    for gx, gy, bw, sh, sw, obh in ((8, 8, 2, 1, 4, 4), (8, 8, 4, 2, 2, 8), (8, 8, 2, 1, 4, 8), (8, 4, 2, 1, 4, 4)):
        if (H // 32) % gx or Mt % gy:
            continue
        pc = pc2d(gx, gy, Mt, H // 32, bw, sh, sw, obh)
        t(f"per-expert down 2D {gx}x{gy} k{bw} sub{sh}x{sw} out_block_h {obh}", lambda: ttnn.matmul(
            h, w_dn0, program_config=pc, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=DRAM))
    # batched, 1D multicast of in1 (mcast_in0=False): every core owns per_core_M rows of every expert's output
    ref_gu = chip0(ttnn.matmul(x12, moe.w_gate_up, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=DRAM))
    ref_dn = chip0(ttnn.matmul(h12, moe.w_down, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=DRAM))

    def pc1d_in1(gx, gy, pm, pn, bw, sh, sw, obh, obw):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=bw, out_subblock_h=sh,
            out_subblock_w=sw, out_block_h=obh, out_block_w=obw, per_core_M=pm, per_core_N=pn, fuse_batch=False,
            fused_activation=None, mcast_in0=False)

    module_cfgs = (
        ("gate_up (module config)", x12, moe.w_gate_up, ref_gu, prefill_experts_pc(Mt, 2 * I // 32, out_block_w=20)),
        ("down (module config)", h12, moe.w_down, ref_dn, prefill_experts_pc(Mt, H // 32, out_block_w=16)),
    )
    for tag, inp, w, ref, n_t, cfgs in (
        ("gate_up", x12, moe.w_gate_up, ref_gu, 2 * I // 32,
         ((8, 8, 2, 4, 1, 4, 2, 40), (8, 4, 4, 4, 1, 4, 4, 20), (8, 8, 2, 4, 2, 2, 2, 40))),
        ("down", h12, moe.w_down, ref_dn, H // 32,
         ((8, 8, 2, 4, 1, 4, 2, 32), (8, 4, 4, 4, 1, 4, 4, 16), (8, 8, 2, 4, 2, 2, 2, 32))),
    ):
        for gx, gy, pm, bw, sh, sw, obh, obw in cfgs:
            if Mt % pm or n_t % obw:
                continue
            name = f"batched {tag} 1D mcast-in1 {gx}x{gy} per_core_M {pm} k{bw} sub{sh}x{sw} out_block {obh}x{obw}"
            try:
                pc = pc1d_in1(gx, gy, pm, n_t, bw, sh, sw, obh, obw)
                fn = lambda: ttnn.matmul(inp, w, program_config=pc, compute_kernel_config=ck,  # noqa: E731
                                         dtype=ttnn.bfloat16, memory_config=DRAM)
                got = chip0(fn())
                same = bool(torch.equal(got, ref))
                st = stats(ref.reshape(-1, ref.shape[-1]), got.reshape(-1, ref.shape[-1]))
                t(name, fn)
                rows[-1] += f" (vs auto: bitwise {same}, pcc {st['pcc']:.7f})"
            except Exception as e:
                rows.append(f"{name}: error {type(e).__name__}: {str(e)[:260]}")
    for tag, inp, w, ref, pc in module_cfgs:
        name = f"batched {tag}: {pc}"
        try:
            if pc is None:
                raise RuntimeError(f"prefill_experts_pc gives no config for C = {C}")
            fn = lambda: ttnn.matmul(inp, w, program_config=pc, compute_kernel_config=ck,  # noqa: E731
                                     dtype=ttnn.bfloat16, memory_config=DRAM)
            same = bool(torch.equal(chip0(fn()), ref))
            t(f"batched {tag}", fn)
            rows[-1] += f" (vs auto: bitwise {same})"
            if not same:
                failures.append(rows[-1])
        except Exception as e:
            rows.append(f"{name}: error {type(e).__name__}: {str(e)[:260]}")
            failures.append(rows[-1])
    _free([x, h, x12, h12, w_gu0, w_dn0])
    moe.deallocate()
    print(f"[moe] prefill matmul variants (C = {C} rows, bfp8 weights, HiFi4):\n[moe]   " + "\n[moe]   ".join(rows))
    assert not failures, "\n".join(failures)


def _prefill_breakdown(mesh_device, moe, x_tt):
    """Eager per-stage ms of one prefill chunk (``min(S, prefill_chunk)`` rows) + the CCLs on the full S."""
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    S = int(x_tt.shape[-2])
    C = min(S, moe.prefill_chunk)
    xc = x_tt if C == S else ttnn.slice(x_tt, [0, 0, 0, 0], [1, 1, C, H])
    out = {}

    def t(name, fn, iters=2):
        try:
            out[name] = eager_us(mesh_device, fn, iters=iters, warmup=1) / 1e3
        except Exception as e:
            out[name] = f"error {type(e).__name__}: {str(e)[:160]}"

    from models.demos.motif3.tt.moe import prefill_experts_pc

    idx, w = moe.router(xc, scale=moe.internal_route_scale)
    w_loc = moe.local_weights(idx, w)
    x12 = ttnn.repeat(xc, ttnn.Shape([1, moe.e_loc, 1, 1]), memory_config=DRAM)
    gdt = ttnn.float32 if moe.prefill_polynorm == "fp32" else ttnn.bfloat16
    pc_gu = prefill_experts_pc(C // 32, 2 * moe.inter // 32, out_block_w=20) if moe.prefill_pc else None
    pc_dn = prefill_experts_pc(C // 32, moe.hidden // 32, out_block_w=16) if moe.prefill_pc else None
    gu = ttnn.matmul(x12, moe.w_gate_up, program_config=pc_gu, compute_kernel_config=moe.ckc_experts, dtype=gdt,
                     memory_config=DRAM)
    h = moe.polynorm(gu, mode=moe.prefill_polynorm, row_scale=w_loc, memory_config=DRAM)
    y = ttnn.matmul(h, moe.w_down, program_config=pc_dn, compute_kernel_config=moe.ckc_experts, dtype=moe.down_dtype,
                    memory_config=DRAM)
    cfg_tag = "in1-mcast cfg" if pc_gu is not None else "auto cfg"
    t(f"router[{C}]", lambda: moe.router(xc, scale=moe.internal_route_scale))
    t("local mask", lambda: moe.local_weights(idx, w))
    t("repeat x12", lambda: ttnn.repeat(xc, ttnn.Shape([1, moe.e_loc, 1, 1]), memory_config=DRAM))
    t(f"gate_up matmul ({cfg_tag})", lambda: ttnn.matmul(x12, moe.w_gate_up, program_config=pc_gu,
                                                         compute_kernel_config=moe.ckc_experts, dtype=gdt,
                                                         memory_config=DRAM))
    for impl in ("horner", "rms"):
        t(f"polynorm {impl} {moe.prefill_polynorm} +w", lambda: moe.polynorm(
            gu, mode=moe.prefill_polynorm, row_scale=w_loc, memory_config=DRAM, impl=impl))
    t(f"down matmul ({cfg_tag})", lambda: ttnn.matmul(h, moe.w_down, program_config=pc_dn,
                                                      compute_kernel_config=moe.ckc_experts, dtype=moe.down_dtype,
                                                      memory_config=DRAM))
    t("fast_reduce_nc", lambda: moe.reduce_experts(y))
    part = upload_replicated(torch.zeros(S, H), mesh_device)
    t(f"rs_dp[{S}]", lambda: moe.ccl.rs_dp(part, 2))
    rs = moe.ccl.rs_dp(part, 2)
    t(f"ar_tp[{S // 4}]", lambda: moe.ccl.ar_tp(rs))
    t(f"ag_dp[{S // 4}]", lambda: moe.ccl.ag_dp(rs, 2))
    _free([idx, w, w_loc, x12, gu, h, y, part, rs])
    if xc is not x_tt:
        _free(xc)
    return out


if __name__ == "__main__":
    # CPU-only helpers (run under scripts/hostrun.sh):
    #   python -m models.demos.motif3.tests.unit.test_moe capture        real router inputs of REAL_LAYERS
    #   python -m models.demos.motif3.tests.unit.test_moe golden 2 4     fp32 reference routed outputs (prefill test)
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        with torch.no_grad():
            capture_real_router_inputs()
    elif len(sys.argv) > 1 and sys.argv[1] == "golden":
        from models.demos.motif3.tt.weights import HFWeightLoader

        _data = torch.load(REAL_INPUTS, weights_only=True)
        _src = HFWeightLoader()
        for _L in [int(a) for a in sys.argv[2:]] or [2]:
            ref_routed_all_tokens(_src, _L, _data["layers"][_L]["x"])
