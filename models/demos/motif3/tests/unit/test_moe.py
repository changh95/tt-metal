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

    def local_partial(f, *, polynorm, decode, taps=None, memory_config=None):
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

    rows = 32 if dp1 else 8
    x = _FakeT("x", (1, 1, rows, H))
    for ap_dt in (ttnn.bfloat16, ttnn.float32):
        ap = _FakeT("add_partial", (1, 1, rows, H), dtype=ap_dt)
        check(f"decode add_partial {ap_dt}", lambda: moe.forward_decode(x, add_partial=ap), [x, ap])
    check("decode", lambda: moe.forward_decode(x), [x])
    check("decode reduce_tp=False", lambda: moe.forward_decode(x, reduce_tp=False), [x])
    taps = {}
    check("decode taps", lambda: moe.forward_decode(x, taps=taps), [x], taps)
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
    idx_dev, idx_same = read_replicated(taps["idx"], mesh)
    w_dev, w_same = read_replicated(taps["w"], mesh)
    idx_dev = idx_dev.reshape(-1, K).long()
    w_dev = w_dev.reshape(-1, K).double() * (moe.route_scale / moe.internal_route_scale)
    res = {
        "tag": tag,
        "replicas_identical_tp": same_tp,
        "gathered_exact": bool(torch.equal(f_all.reshape(-1, H).float(), x32.float())) and f_same,
        "routes_identical_32": idx_same and w_same,
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
            idx2 = read_replicated(taps2["idx"], mesh_device, chips=[0])[0].reshape(-1, K).long()
            w2 = read_replicated(taps2["w"], mesh_device, chips=[0])[0].reshape(-1, K).double() * (
                moe.route_scale / moe.internal_route_scale)
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
    idx_d = read_replicated(taps["idx"], mesh_device, chips=[0])[0].reshape(-1, K).long()
    w_d = read_replicated(taps["w"], mesh_device, chips=[0])[0].reshape(-1, K).double() * (
        moe.route_scale / moe.internal_route_scale)
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
