# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Truncated-model tests (WAVE_A_REVIEW GEN-7(b), design §4.5): the real model, layers 0..N-1, end to end.

``test_model_truncated_prefill``: ``MotifModel`` with layers 0..7 (real weights) from the token ids of the 6 C2 prompts:
the 4-stream residual after layer 3 and after layer 7 vs the golden states (README §12: truncated-model state >= 0.99);
the early-exit top-1 of those states at **all 2971 positions** (the reference's bf16 head on the TT state vs on the
golden state; the floor: the same head on the reference's own fp32 run, ``goldens/c2/fp32_sanity_L0-7``); and the
last-token logits of the 8-layer model (TT head) vs the reference head on the golden state after layer 7.

``test_model_truncated_generator[N]`` (N = 4, 8): the serving stack -- ``MotifGenerator.create`` (settings as the bridge
builds them) behind the **real bridge** ``MotifForCausalLM`` (``generator_vllm``), plugin-style calls only:

* ``allocate_kv_cache`` (vLLM's 53-layer hint, a truncated run allocates N), ``warmup_model_prefill`` (every bucket),
  ``warmup_model_decode`` eager then traced (the capture);
* 30 requests on 30 of the 32 state slots (two slots stay empty = inactive lanes), each a C2 prompt prefix of its own
  length (heterogeneous positions 100..883), prefilled through the bridge in batches with block tables whose tails hold
  stale ids of OTHER live requests' blocks (the bridge must zero them, or the bucket padding corrupts their KV);
* ``DECODE_STEPS`` teacher-forced decode steps over all 32 rows (traced), with a ``slot_remap`` permutation at one
  step (row i reads the state slot remap[i]; the bridge's LaneMap follows it), one request finishing early (its row
  turns inactive), and two steps run eager and traced with the same inputs (must be bitwise equal: trace replay with
  new inputs == eager);
* every prefill / decode logits row vs the reference head on the golden state after layer N-1 (PCC, calibrated top-1);
* decode step latency (traced replay alone, the full ``decode_forward`` with host I/O, eager) and the trace size.

``test_model_cache_roundtrip``: CONV-1..4 (convert layer 0 into a scratch cache, rebuild from it alone) and the
``cache="auto"`` read-only rule (a converted part built with an option the converter did not build writes nothing).

``test_model_teacher_forced`` (opt-in ``MOTIF3_RUN_TF=1``): 36 real layers at the serving geometry, the early-exit head
after layer 35 vs the golden heads, the decision-D1 router A/B, and a 53-layer full-scale proxy (finding 10).

Early-exit top-1 metric (review finding 1). A row **agrees** when the candidate's argmax is the reference argmax or
ties it exactly in the reference logits (``argmax_ties``: bf16 logits tie often); rows are binned by the reference's
top-1 - top-2 logit margin. A truncated model's head is flat, and rows with a small margin flip under ANY change of
numerics, so the overall rate is a coarse floor and the hard check is on rows with a clear margin. Calibration on all
2971 C2 positions (reference bf16 head; ties count as agreement):

=========================================  ==========  ==============  ==============
candidate vs the bf16 golden                overall     margin > 0.25   margin > 0.5
=========================================  ==========  ==============  ==============
reference fp32 run, after layer 3           0.9926      0.9996          0.9995
TT 8-layer prefill state, after layer 3     0.9845      0.9991          0.9995
reference fp32 run, after layer 7           0.9842      1.0000          1.0000
TT 8-layer prefill state, after layer 7     0.9764      0.9987          0.9995
reference fp32 run, after layer 35          0.9448      0.9771          0.9886  (exact argmax 0.9307; > 1.0: 0.9936)
=========================================  ==========  ==============  ==============

After layer 35 the reference's own fp32 run misses the decision-D1 target "top-1 >= 95 %" against its bf16 run: that
target is below the noise floor of an early-exit head at this depth (it is meant for the full model). The 36-layer test
therefore reports the D1 rate and asserts against the floor (``TF_*`` constants below); the fp32 run of layers 0-35 is
``MOTIF3_GOLDEN_FP32_DIR`` (made with ``python -m models.demos.motif3.reference.golden_stream --dtype fp32 --layers 0-35
--save-layers 35 --final-head --threads 32 --out <dir>``, 4.5 min on the host; see ``fp32_golden_or_none``).
Measured 2026-10-02 (``logs/dev/20261002_030247_integ2_tf36_ab_proxy53.log``), 990 teacher-forced predictions vs the
bf16 golden, tie-aware overall / margin > 0.5 / margin > 1.0: composite router 0.9374 / 0.9871 / 0.9944, exact-fp32
router 0.9253 / 0.9842 / 0.9925, the fp32 run on the same rows 0.9414 / 0.9842 / 0.9888; vs the fp32 run, overall:
composite 0.9323, exact 0.9111, the bf16 golden itself 0.9293. The state after layer 35 is as far from the fp32 run as
the bf16 golden is (per-token relative error median 5.2 % vs 4.3 %, p99 27.5 % vs 27.7 %).

Run (device; the generator test is ~3-5 min per N)::

    scripts/devrun.sh -t 2400 -n model_truncated -- python -m pytest models/demos/motif3/tests/test_model_truncated.py \
        -s -p no:cacheprovider --timeout=0
    # host-only checks of the disk guards / read-only cache / prefill page table (devices hidden)
    scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q models/demos/motif3/tests/test_model_truncated.py -k host
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import ttnn
from models.demos.motif3.tt.model_config import device_params

GOLDEN_DIR = Path(os.environ.get("MOTIF3_GOLDEN_STREAM_DIR", "/home/ttuser/hchang/experiments/motif-3/goldens/c2"))
GOLDEN_FP32_L7_DIR = GOLDEN_DIR / "fp32_sanity_L0-7"  # the reference's fp32 run of layers 0-7 (states after 3, 7)
GOLDEN_FP32_DIR = Path(os.environ.get(
    "MOTIF3_GOLDEN_FP32_DIR",
    "/home/ttuser/hchang/experiments/motif-3/tt_cache/test/integration/goldens_c2_fp32_L0-35"))  # head + state after 35
MAX_MODEL_LEN = 2048
NUM_BLOCKS = 512
BLOCK = 64
DECODE_STEPS = 10
STATE_PCC_MIN = 0.99  # README §12: truncated model state after several layers
LOGIT_PCC_MIN = 0.99
# early-exit top-1 (module docstring): margin bands, the hard check above TOP1_MARGIN, coarse overall floors
MARGIN_EDGES = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0)
TOP1_MARGIN = 0.5
TOP1_MIN_ABOVE = 0.995  # misses allowed above the margin: max(TOP1_SLACK_ROWS, 0.5 % of those rows)
TOP1_SLACK_ROWS = 2
TOP1_MIN_OVERALL_STATE = 0.95  # all positions, the reference head on the TT state (layers 3 / 7)
TOP1_MIN_OVERALL_E2E = 0.90  # generator rows (TT head on the TT state; near prompt ends)
MESH = [pytest.param((4, 8), device_params(), id="4x8")]


# ============================================================================================================
# host helpers
# ============================================================================================================
def stats(ref: torch.Tensor, got: torch.Tensor) -> dict:
    from models.common.utility_functions import comp_pcc

    r, g = ref.double(), got.double()
    nonfinite = int((~torch.isfinite(g)).sum())
    _, pcc = comp_pcc(r, g, 0.0)
    return {"pcc": float(pcc) if nonfinite == 0 else float("nan"), "max_abs": float((r - g).abs().max()),
            "nonfinite": nonfinite}


def fmt(s: dict) -> str:
    return f"pcc={s['pcc']:.6f} max_abs={s['max_abs']:.3e}" + (f" NONFINITE={s['nonfinite']}" if s["nonfinite"] else "")


def all_at_least(values, threshold) -> bool:
    """NaN-safe ``min(values) >= threshold`` (a NaN PCC -- non-finite logits -- fails)."""
    return all(v >= threshold for v in values)


def load_goldens(layers):
    """``({layer: {prompt: [S, 4, 4096] bf16}}, [StreamPrompt])``."""
    from models.demos.motif3.reference import golden_stream as gs

    pfile = GOLDEN_DIR / "prompts.json"
    missing = [str(p) for p in [pfile] + [gs.state_path(GOLDEN_DIR, l) for l in layers] if not p.is_file()]
    if missing:
        pytest.skip(f"C2 golden-stream files missing: {missing}")
    prompts = gs.load_prompt_set(pfile)
    sha = gs.prompt_set_sha256(prompts)
    out = {}
    for l in layers:
        states, meta = gs.load_tensors(gs.state_path(GOLDEN_DIR, l))
        assert meta.get("prompt_sha256") == sha and meta.get("mode", {}).get("dtype") == "bf16", (l, meta.get("mode"))
        out[l] = {k: v[0] for k, v in states.items()}
    return out, prompts


def fp32_states_or_none(directory: Path, layers, sha: str):
    """``{layer: {prompt: [S, 4, 4096]}}`` of a reference fp32 run (calibration only), or None if absent/mismatched."""
    from models.demos.motif3.reference import golden_stream as gs

    out = {}
    for l in layers:
        p = gs.state_path(directory, l)
        if not p.is_file():
            return None
        states, meta = gs.load_tensors(p)
        if meta.get("prompt_sha256") != sha or meta.get("mode", {}).get("dtype") != "fp32":
            return None
        out[l] = {k: v[0] for k, v in states.items()}
    return out


def fp32_golden_or_none(layer: int, sha: str):
    """The head (top-32 format) after ``layer`` of the reference's fp32 run ``MOTIF3_GOLDEN_FP32_DIR``, or None."""
    from models.demos.motif3.reference import golden_stream as gs

    lp, _ = gs.head_paths(GOLDEN_FP32_DIR, layer)
    if not lp.is_file():
        return None
    t, meta = gs.load_tensors(lp)
    if meta.get("prompt_sha256") != sha or meta.get("mode", {}).get("dtype") != "fp32":
        return None
    return t


def real_source_or_skip(layers):
    from models.demos.motif3.tt.weights import HFWeightLoader

    try:
        src = HFWeightLoader()
    except FileNotFoundError as e:
        pytest.skip(f"no local checkpoint: {e}")
    missing = [l for l in layers if not src.layer_available(l)]
    if missing:
        pytest.skip(f"checkpoint layers {missing} are not local")
    for n in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        if not src.available(n):
            pytest.skip(f"{n} is not local")
    return src


class RefHead:
    """The reference's final head in its bf16 numerics (``MotifModel.reduce_streams`` -> ``RMSNorm`` -> ``lm_head``
    -> fp32), on CPU, for streams ``[N, 4, 4096]``."""

    def __init__(self, src):
        from models.demos.motif3.reference.modules import RMSNorm

        self.norm = RMSNorm(4096, 1e-5).to(torch.bfloat16)
        with torch.no_grad():
            self.norm.weight.copy_(src.get("model.norm.weight").to(torch.bfloat16))
        self.lm = src.get("lm_head.weight").to(torch.bfloat16)

    @torch.no_grad()
    def __call__(self, streams: torch.Tensor) -> torch.Tensor:
        h = self.norm(streams.to(torch.bfloat16).mean(dim=1))
        return torch.nn.functional.linear(h, self.lm).float()


# ============================================================================================================
# early-exit top-1 metric (module docstring; review finding 1)
# ============================================================================================================
def _agree_topk(ids_k, vals_k, argmax, am):
    """Tie-aware agreement of argmaxes ``am [N]`` with a reference given as top-32 ``ids_k`` / ``vals_k`` (descending)
    and its full-vocab ``argmax``: ``am`` is the argmax, or has exactly the reference's max logit."""
    hit = ids_k == am[:, None]
    at = torch.where(hit, vals_k, torch.full_like(vals_k, float("-inf"))).max(-1).values
    return (am == argmax) | (at == vals_k[:, 0]), hit.any(-1)


def top1_full(ref: torch.Tensor, got: torch.Tensor) -> dict:
    """Full-vocab reference logits ``ref [N, V]`` vs candidate ``got [N, V]``, per row: ``agree`` (tie-aware),
    ``exact``, ``margin`` (reference top-1 - top-2), ``err32`` (max |ref - got| over the reference top-32)."""
    ref, got = ref.float(), got.float()
    am = got.argmax(-1)
    top = ref.topk(32, dim=-1)
    exact = am == ref.argmax(-1)
    agree = exact | (ref.gather(1, am[:, None])[:, 0] == top.values[:, 0])
    err32 = (top.values - got.gather(1, top.indices)).abs().max(-1).values
    return {"agree": agree, "exact": exact, "margin": top.values[:, 0] - top.values[:, 1], "err32": err32}


def top1_topk(ids_k, vals_k, argmax, got: torch.Tensor) -> dict:
    """A golden head's rows (``topk_ids`` / ``topk_logits`` [N, 32] descending, full-vocab ``argmax`` [N]) vs candidate
    full-vocab logits ``got [N, V]``: the fields of :func:`top1_full` (``err32`` over the golden top-32) plus
    ``pcc32`` (the candidate's logits at the golden top-32 ids vs the golden's)."""
    vals_k, got = vals_k.float(), got.float()
    am = got.argmax(-1)
    agree, _ = _agree_topk(ids_k, vals_k, argmax, am)
    dev_k = got.gather(1, ids_k)
    pcc32 = torch.tensor([stats(vals_k[j], dev_k[j])["pcc"] for j in range(vals_k.shape[0])])
    return {"agree": agree, "exact": am == argmax, "margin": vals_k[:, 0] - vals_k[:, 1],
            "err32": (dev_k - vals_k).abs().max(-1).values, "pcc32": pcc32}


def floor_topk(ids_k, vals_k, argmax, am_other) -> dict:
    """Another reference run's argmax ``am_other`` vs a golden head (top-32 format): the noise floor rows."""
    vals_k = vals_k.float()
    agree, _ = _agree_topk(ids_k, vals_k, argmax, am_other)
    return {"agree": agree, "exact": am_other == argmax, "margin": vals_k[:, 0] - vals_k[:, 1]}


def cat_rows(parts) -> dict:
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def bands(m: dict, edges=MARGIN_EDGES) -> dict:
    """``{"n", "exact", "agree", "above": {t: (rows, misses)}, "band": {label: (rows, agree)}}``."""
    a, mg = m["agree"].bool(), m["margin"].float()
    out = {"n": int(a.numel()), "exact": float(m["exact"].float().mean()), "agree": float(a.float().mean()),
           "above": {}, "band": {}}
    for t in edges:
        sel = mg > t
        out["above"][t] = (int(sel.sum()), int((~a[sel]).sum()))
    cuts = [(None, 0.0)] + list(zip(edges, list(edges[1:]) + [float("inf")]))
    for lo, hi in cuts:
        sel = (mg <= 0) if lo is None else ((mg > lo) & (mg <= hi))
        label = "tie" if lo is None else f"({lo:g},{hi:g}]"
        out["band"][label] = (int(sel.sum()), float(a[sel].float().mean()) if bool(sel.any()) else float("nan"))
    return out


def fmt_bands(b: dict) -> str:
    above = "; ".join(f">{t:g}: {(n - k) / n if n else float('nan'):.4f} ({n} rows, {k} miss)"
                      for t, (n, k) in b["above"].items())
    band = " ".join(f"{lab} {a:.3f}/{n}" for lab, (n, a) in b["band"].items())
    return f"{b['n']} rows: exact {b['exact']:.4f}, tie-aware {b['agree']:.4f} | margin {above} | bands {band}"


def top1_failures(tag: str, m: dict, *, min_overall: float, margin: float = TOP1_MARGIN,
                  min_above: float = TOP1_MIN_ABOVE, slack_rows: int = TOP1_SLACK_ROWS) -> list:
    """The calibrated top-1 checks: tie-aware overall >= ``min_overall``; above ``margin`` at most
    ``max(slack_rows, (1 - min_above) x rows)`` misses."""
    b = bands(m)
    out = []
    if not b["agree"] >= min_overall:
        out.append(f"{tag}: tie-aware top-1 {b['agree']:.4f} < {min_overall} ({fmt_bands(b)})")
    rows, miss = bands(m, (margin,))["above"][margin]
    allowed = max(int(slack_rows), int((1.0 - min_above) * rows))
    if miss > allowed:
        out.append(f"{tag}: {miss} of {rows} rows with a reference margin > {margin:g} disagree (allowed {allowed}; "
                   f"{fmt_bands(b)})")
    return out


# ============================================================================================================
# device helpers
# ============================================================================================================
def _free(*ts):
    for t in ts:
        if t is None:
            continue
        if isinstance(t, (list, tuple)):
            _free(*t)
        elif isinstance(t, dict):
            _free(*t.values())
        elif t.is_allocated():
            ttnn.deallocate(t)


def read_streams(t, S: int) -> torch.Tensor:
    v = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return v[0, :, :S].permute(1, 0, 2).contiguous()


def bucket_of(S: int) -> int:
    b = 128
    while b < S:
        b *= 2
    return b


def mem_line(mesh_device, tag: str) -> dict:
    """DRAM / trace-region use per chip (allocator view), printed."""
    from models.demos.motif3.tt.model import device_bytes_per_chip

    d = device_bytes_per_chip(mesh_device)
    t = device_bytes_per_chip(mesh_device, ttnn.BufferType.TRACE)
    if d is not None:
        print(f"[mem] {tag}: DRAM per chip allocated {d['allocated'] / 1e9:.2f} GB, free {d['free'] / 1e9:.2f} GB of "
              f"{d['total'] / 1e9:.2f} GB (largest free block {d['largest_free'] / 1e9:.2f} GB)"
              + (f"; trace region allocated {t['allocated'] / 2**20:.1f} MiB of {t['total'] / 2**20:.1f} MiB"
                 if t is not None else ""))
    return {"dram": d, "trace": t}


@torch.no_grad()
def head_compare(head: RefHead, ref_states: torch.Tensor, got_states: torch.Tensor, chunk: int = 256) -> dict:
    """The reference head on ``got_states`` vs on ``ref_states`` (``[N, 4, 4096]`` each), in row chunks."""
    parts = []
    for c0 in range(0, ref_states.shape[0], chunk):
        parts.append(top1_full(head(ref_states[c0: c0 + chunk]), head(got_states[c0: c0 + chunk])))
    return cat_rows(parts)


# ============================================================================================================
# (1) model prefill vs the golden states after layers 3 and 7
# ============================================================================================================
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@torch.no_grad()
def test_model_truncated_prefill(mesh_device, device_params):
    from models.demos.motif3.reference import golden_stream as gs
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.model import MotifModel
    from models.demos.motif3.tt.model_config import MotifTTConfig

    goldens, prompts = load_goldens((3, 7))
    src = real_source_or_skip(range(8))
    log_fabric(mesh_device, "model_truncated_prefill")
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device, max_model_len=MAX_MODEL_LEN, num_layers=8)
    t0 = time.time()
    model = MotifModel(mesh_device, cfg, source=src, layers=range(8), cache="auto")
    try:
        print(f"[truncated] model layers 0..7 loaded in {time.time() - t0:.1f} s ({model.load_seconds})")
        head = RefHead(src)
        failures = []
        agg = {3: ([], []), 7: ([], [])}
        ref_last, got_last = [], []
        for p in prompts:
            S = len(p.ids)
            B = bucket_of(S)
            tok = model.embed.prefill_tokens_device(torch.tensor(p.ids, dtype=torch.int32), B)
            X = model.embed.forward_prefill(tok)
            _free(tok)
            t1 = time.time()
            for i, layer in enumerate(model.layers):
                Xn = layer.forward_prefill(X)
                _free(X)
                X = Xn
                if i in agg:
                    got = read_streams(X, S)
                    ref = goldens[i][p.name].float()
                    s = stats(ref, got)
                    print(f"[truncated] {p.name:18s} S={S:4d}: streams after layer {i} {fmt(s)}")
                    if not s["pcc"] >= STATE_PCC_MIN:
                        failures.append(f"{p.name} after layer {i}: {fmt(s)}")
                    agg[i][0].append(ref)
                    agg[i][1].append(got)
            tile = model.head.forward_prefill(X, S - 1)
            logits = model.head.prefill_logits_to_host(tile, S - 1).float()
            _free(tile, X)
            ref_last.append(head(goldens[7][p.name][S - 1:S].float())[0])
            got_last.append(logits)
            print(f"[truncated] {p.name}: 8-layer prefill {time.time() - t1:.2f} s (bucket {B}); last-token logits "
                  f"top-1 dev {int(logits.argmax())} ref {int(ref_last[-1].argmax())}")
    finally:
        model.deallocate()

    # ---- states: PCC over all prompts, then the early-exit top-1 at all positions (state error only: the CPU head) ----
    fp32 = fp32_states_or_none(GOLDEN_FP32_L7_DIR, (3, 7), gs.prompt_set_sha256(prompts))
    t0 = time.time()
    for i, (r, g) in agg.items():
        ref_all, got_all = torch.cat(r), torch.cat(g)
        s = stats(ref_all, got_all)
        print(f"[truncated] all prompts ({ref_all.shape[0]} tokens): streams after layer {i} {fmt(s)}")
        m = head_compare(head, ref_all, got_all)
        b = bands(m)
        print(f"[truncated] early-exit top-1 after layer {i}, reference head on the TT state vs on the golden state: "
              f"{fmt_bands(b)}")
        failures += top1_failures(f"top-1 after layer {i} (all positions)", m, min_overall=TOP1_MIN_OVERALL_STATE)
        if fp32 is not None:  # the floor: the same head on the reference's own fp32 run
            f_all = torch.cat([fp32[i][p.name].float() for p in prompts])
            print(f"[truncated]   floor after layer {i}, reference fp32 run vs the bf16 golden: "
                  f"{fmt_bands(bands(head_compare(head, ref_all, f_all)))}")
    print(f"[truncated] head calibration in {time.time() - t0:.1f} s")
    m = top1_full(torch.stack(ref_last), torch.stack(got_last))
    pccs = [stats(r, g)["pcc"] for r, g in zip(ref_last, got_last)]
    print(f"[truncated] last-token logits (8 layers, TT head) vs the reference head on the golden state after layer 7: "
          f"pcc min {min(pccs):.6f} mean {sum(pccs) / len(pccs):.6f}; {fmt_bands(bands(m))}")
    if not all_at_least(pccs, LOGIT_PCC_MIN):
        failures.append(f"last-token logits pcc {pccs}")
    assert not failures, "\n".join(failures)


# ============================================================================================================
# (2) generator + real bridge: 32-lane decode, heterogeneous positions, inactive lanes, slot remap, trace replay
# ============================================================================================================
class _Req:
    def __init__(self, rid, prompt, q0, blocks):
        self.rid, self.prompt, self.q0, self.blocks = rid, prompt, q0, blocks
        self.active = True

    @property
    def ids(self):
        return self.prompt.ids


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("num_layers", [4, 8], ids=["N4", "N8"])
@torch.no_grad()
def test_model_truncated_generator(mesh_device, device_params, num_layers):
    from models.demos.motif3.tt import generator_api as api
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.generator_vllm import MotifForCausalLM
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR

    N = int(num_layers)
    goldens, prompts = load_goldens((N - 1,))
    gst = goldens[N - 1]
    src = real_source_or_skip(range(N))
    log_fabric(mesh_device, f"model_truncated_generator N={N}")
    head = RefHead(src)
    settings = api.GeneratorSettings(max_batch_size=api.NUM_LANES, max_seq_len=MAX_MODEL_LEN, num_layers=N,
                                     weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=BLOCK,
                                     weights_source="test (local snapshot)")
    t0 = time.time()
    gen = MotifGenerator.create(hf_config=None, mesh_device=mesh_device, settings=settings)
    t_create = time.time() - t0
    bridge = None
    try:
        bridge = MotifForCausalLM(gen, settings)
        kv = bridge.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK, api.KV_LATENT_DIM), torch.bfloat16, api.NUM_HIDDEN_LAYERS)
        W = kv.page_table_width
        t0 = time.time()
        bridge.warmup_model_prefill(kv, enable_trace=False)
        t_wp = time.time() - t0
        t0 = time.time()
        bridge.warmup_model_decode(kv, enable_trace=False, max_batch_size=api.NUM_LANES, num_blocks=W)
        bridge.warmup_model_decode(kv, enable_trace=True, max_batch_size=api.NUM_LANES, num_blocks=W)
        t_wd = time.time() - t0
        assert gen.trace_captured
        print(f"[gen N={N}] create {t_create:.1f} s, warmup prefill {t_wp:.1f} s ({len(gen.cfg.prefill_buckets)} "
              f"buckets), warmup decode + capture {t_wd:.1f} s; W={W}")
        mem_line(mesh_device, f"N={N} after warmup + capture")
        failures = _generator_body(mesh_device, gen, bridge, kv, W, N, prompts, gst, head, api)
        bridge.release_persistent_capture()
        assert not gen.trace_captured
    finally:
        gen.close()  # releases a live trace too (also on failure)
    assert not failures, "\n".join(failures)


def _generator_body(mesh_device, gen, bridge, kv, W, N, prompts, gst, head, api) -> list:
    # ---- requests: 30 slots used, 2 empty (inactive lanes) ----------------------------------------------------------
    n_req = 30
    g = torch.Generator().manual_seed(7 + N)
    perm = (torch.randperm(NUM_BLOCKS - 1, generator=g) + 1).tolist()
    reqs, nxt = [], 0
    for r in range(n_req):
        p = prompts[r % len(prompts)]
        q0 = len(p.ids) - DECODE_STEPS - 3 * (r // len(prompts)) - 1
        nb = math.ceil((q0 + DECODE_STEPS) / BLOCK)
        reqs.append(_Req(r, p, q0, perm[nxt: nxt + nb]))
        nxt += nb
    slot_of = {r: r for r in range(n_req)}  # request -> vLLM state slot (rows are slots when no remap)
    stale_ids = perm[:40]  # ids of live requests' blocks: what stale block-table tails hold

    def table_row(req, filler_seed):
        row = torch.zeros(W, dtype=torch.int32)
        row[: len(req.blocks)] = torch.tensor(req.blocks, dtype=torch.int32)
        gg = torch.Generator().manual_seed(filler_seed)
        tail = torch.tensor(stale_ids, dtype=torch.int32)[torch.randint(len(stale_ids), (W,), generator=gg)]
        need = len(req.blocks)
        row[need:] = tail[need:]  # stale ids of OTHER requests' blocks past this request's blocks
        return row

    # ---- prefill through the bridge, in batches of 8 rows -----------------------------------------------------------
    pf_ref, pf_got, failures = [], [], []
    t0 = time.time()
    for b0 in range(0, n_req, 8):
        batch = reqs[b0: b0 + 8]
        lens = np.array([q.q0 for q in batch], dtype=np.int64)
        tokens = torch.full((len(batch), int(lens.max()) + 5), 7, dtype=torch.int32)  # stale values past each length
        for i, q in enumerate(batch):
            tokens[i, : q.q0] = torch.tensor(q.ids[: q.q0], dtype=torch.int32)
        table = torch.stack([table_row(q, 100 + q.rid) for q in batch])
        out = bridge.prefill_forward(tokens=tokens, page_table=table, kv_cache=kv, prompt_lens=lens,
                                     empty_slots=[slot_of[q.rid] for q in batch])
        assert tuple(out.shape) == (len(batch), 1, api.VOCAB_SIZE)
        for i, q in enumerate(batch):
            pf_got.append(out[i, 0].float())
            pf_ref.append(head(gst[q.prompt.name][q.q0 - 1: q.q0].float())[0])
    t_pf = time.time() - t0
    pf_pcc = [stats(r, g)["pcc"] for r, g in zip(pf_ref, pf_got)]
    pf_m = top1_full(torch.stack(pf_ref), torch.stack(pf_got))
    print(f"[gen N={N}] prefill {n_req} requests (lengths {min(q.q0 for q in reqs)}..{max(q.q0 for q in reqs)}) in "
          f"{t_pf:.1f} s: logits pcc min {min(pf_pcc):.6f} mean {sum(pf_pcc) / n_req:.6f}; {fmt_bands(bands(pf_m))}")
    if not all_at_least(pf_pcc, LOGIT_PCC_MIN):  # NaN-safe (review finding 5)
        failures.append(f"prefill logits pcc below {LOGIT_PCC_MIN} (or non-finite): {[round(p, 6) for p in pf_pcc]}")

    # ---- decode: 32 rows, slot remap at REMAP_STEP, request 5 stops at STOP_STEP, eager == traced at CHECK_STEPS -----
    REMAP_STEP, STOP_STEP, CHECK_STEPS = 4, 6, (2, 8)
    stop_req = 5
    remap = list(range(api.NUM_LANES))
    rng = torch.Generator().manual_seed(99)
    remap = [remap[i] for i in torch.randperm(api.NUM_LANES, generator=rng).tolist()]  # a full permutation
    dec_ref, dec_got, dec_tag = [], [], []
    exact_steps = []
    step_times = []
    for t in range(DECODE_STEPS):
        if t == STOP_STEP:
            reqs[stop_req].active = False
        slot_remap = torch.tensor(remap, dtype=torch.int32) if t == REMAP_STEP else None
        # row i carries the request currently in state slot (remap[i] if remapped else i)
        req_in_slot = {s: q for q in reqs for s in [slot_of[q.rid]] if q.active}
        rows_src = remap if slot_remap is not None else list(range(api.NUM_LANES))
        tokens = torch.zeros(api.NUM_LANES, 1, dtype=torch.int32)
        pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
        table = torch.zeros(api.NUM_LANES, W, dtype=torch.int32)
        row_req = {}
        for i, s in enumerate(rows_src):
            q = req_in_slot.get(s)
            if q is None:
                continue
            p = q.q0 + t
            tokens[i, 0] = q.ids[p]
            pos[i] = p
            table[i] = table_row(q, 1000 * t + q.rid)
            row_req[i] = q
        kw = dict(tokens=tokens, start_pos=pos, page_table=table, kv_cache=kv, slot_remap=slot_remap)
        if t in CHECK_STEPS:
            assert slot_remap is None
            out_e = bridge.decode_forward(enable_trace=False, **kw).clone()
        t1 = time.perf_counter()
        out = bridge.decode_forward(enable_trace=True, **kw)
        step_times.append(time.perf_counter() - t1)
        if t in CHECK_STEPS:
            same = torch.equal(out, out_e)
            rows = sorted(row_req)
            diff = float((out[rows] - out_e[rows]).abs().max())
            exact_steps.append((t, same, diff))
            if not same:
                failures.append(f"step {t}: traced != eager (max diff on active rows {diff:.3e})")
        assert tuple(out.shape) == (api.NUM_LANES, 1, api.VOCAB_SIZE)
        for i, q in row_req.items():
            dec_got.append(out[i, 0].float())
            dec_ref.append(head(gst[q.prompt.name][q.q0 + t: q.q0 + t + 1].float())[0])
            dec_tag.append((t, q.rid, i))
        if slot_remap is not None:  # accepted: slot i now holds what slot remap[i] held
            for q in reqs:
                if slot_of[q.rid] in remap:
                    slot_of[q.rid] = remap.index(slot_of[q.rid])
    n = len(dec_tag)
    pccs = [stats(r, g)["pcc"] for r, g in zip(dec_ref, dec_got)]
    m = top1_full(torch.stack(dec_ref), torch.stack(dec_got))
    bad = [(dec_tag[i], round(pccs[i], 6)) for i in range(n) if not pccs[i] >= LOGIT_PCC_MIN]
    print(f"[gen N={N}] decode {DECODE_STEPS} steps, {n} (request, step) rows, positions "
          f"{min(q.q0 for q in reqs)}..{max(q.q0 for q in reqs) + DECODE_STEPS - 1}: logits pcc min {min(pccs):.6f} "
          f"mean {sum(pccs) / n:.6f}; eager==traced {exact_steps}")
    print(f"[gen N={N}] decode top-1 vs the reference head on the golden state after layer {N - 1}: {fmt_bands(bands(m))}")
    for step in (REMAP_STEP - 1, REMAP_STEP, REMAP_STEP + 1):
        idx = [i for i in range(n) if dec_tag[i][0] == step]
        print(f"[gen N={N}]   step {step}{' (slot remap)' if step == REMAP_STEP else ''}: {len(idx)} rows, pcc min "
              f"{min(pccs[i] for i in idx):.6f}, tie-aware top-1 {int(m['agree'][idx].sum())}/{len(idx)}")
    if bad:
        failures.append(f"decode rows below {LOGIT_PCC_MIN} (or non-finite): {bad[:10]} ({len(bad)} total)")
    both = cat_rows([pf_m, m])
    failures += top1_failures(f"N={N} prefill + decode top-1", both, min_overall=TOP1_MIN_OVERALL_E2E)

    # ---- latency: traced replay alone, decode_forward with host I/O, eager -------------------------------------------
    # the last decode step again, in lane order (rewrites the same KV slots with the same latents)
    lt = torch.zeros(api.NUM_LANES, dtype=torch.int32)
    lp = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
    lpt = torch.zeros(api.NUM_LANES, W, dtype=torch.int32)
    for q in reqs:
        if not q.active:
            continue
        lane = bridge._lanes.lane_of_slot(slot_of[q.rid])
        p = q.q0 + DECODE_STEPS - 1
        lt[lane], lp[lane] = q.ids[p], p
        nb = p // BLOCK + 1
        lpt[lane, :nb] = torch.tensor(q.blocks[:nb], dtype=torch.int32)
    batch = api.DecodeBatch(tokens=lt, positions=lp, page_table=lpt)
    pool = kv.device_cache
    for _ in range(2):
        gen.decode_forward(batch, kv_cache=pool, enable_trace=True)
    replay, full, eager = [], [], []
    for _ in range(15):
        t1 = time.perf_counter()
        ttnn.execute_trace(mesh_device, gen._trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        replay.append(time.perf_counter() - t1)
    for _ in range(15):
        t1 = time.perf_counter()
        gen.decode_forward(batch, kv_cache=pool, enable_trace=True)
        full.append(time.perf_counter() - t1)
    for _ in range(3):
        t1 = time.perf_counter()
        gen.decode_forward(batch, kv_cache=pool, enable_trace=False)
        eager.append(time.perf_counter() - t1)
    print(f"[gen N={N}] LATENCY decode step ({N} layers, 32 lanes, contexts up to "
          f"{max(q.q0 for q in reqs) + DECODE_STEPS}): trace replay + sync median {statistics.median(replay) * 1e3:.2f} ms "
          f"(min {min(replay) * 1e3:.2f}); decode_forward (host inputs + replay + [32, 220160] logits to host) median "
          f"{statistics.median(full) * 1e3:.2f} ms (min {min(full) * 1e3:.2f}); bridge steps median "
          f"{statistics.median(step_times) * 1e3:.2f} ms; eager median {statistics.median(eager) * 1e3:.1f} ms")
    return failures


# ============================================================================================================
# (3) TT weight cache round trip (CONV-1..4): convert, then build from the cache without touching the checkpoint
# ============================================================================================================
class _GuardSource:
    """Forwards to ``src`` but raises on any read of a forbidden tensor name prefix (proves a part came from the TT
    cache alone)."""

    def __init__(self, src, forbid: str):
        self.src, self.forbid, self.reads = src, forbid, []

    def _check(self, name):
        if str(name).startswith(self.forbid):
            raise AssertionError(f"read {name} although its layer is converted")
        self.reads.append(name)

    def get(self, name, *a, **k):
        self._check(name)
        return self.src.get(name, *a, **k)

    def get_rows(self, name, *a, **k):
        self._check(name)
        return self.src.get_rows(name, *a, **k)

    def __getattr__(self, name):
        return getattr(self.src, name)


# a 0.36 GB scratch part, deleted at the end of the test: an explicit, lower floor than the 60 GB production one
ROUNDTRIP_MIN_FREE_GB = 20.0


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@torch.no_grad()
def test_model_cache_roundtrip(mesh_device, device_params, tmp_path_factory):
    """``model.convert_weights`` writes layer 0 (dense, 0.36 GB) into a scratch cache root and marks it complete; a
    ``MotifModel(cache="auto")`` then builds layer 0 from the TT cache with a source that raises on any layer-0 read
    (the globals still come from the checkpoint: not converted, nothing written for them), and its prefill matches the
    golden state after layer 0. Then the read-only rule of ``"auto"`` (review finding 4): layer 0 built with
    ``sinkhorn="stock"`` (an option variant the converter did not build) loads the converted tensors, uploads the
    missing stock constants from the checkpoint, lists them in ``cache_misses`` and writes no file. The scratch cache is
    deleted afterwards (disk: CONV-3)."""
    import shutil

    from models.demos.motif3.tt import weights as W
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.model import MotifModel, convert_weights, layer_cache_complete
    from models.demos.motif3.tt.model_config import MotifTTConfig

    goldens, prompts = load_goldens((0,))
    src = real_source_or_skip((0,))
    log_fabric(mesh_device, "model_cache_roundtrip")
    root = Path("/home/ttuser/hchang/experiments/motif-3/tt_cache/test/integration") / f"roundtrip_{os.getpid()}"
    try:
        cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device, max_model_len=MAX_MODEL_LEN, num_layers=1,
                                           tt_cache_root=root)
        t0 = time.time()
        times = convert_weights(mesh_device, cfg, source=src, layers=[0], include_globals=False,
                                min_free_gb=ROUNDTRIP_MIN_FREE_GB)
        l0 = cfg.cache_dir / "L00"
        files = sorted(p.name for p in l0.glob("*.tensorbin"))
        size = sum(p.stat().st_size for p in l0.glob("*.tensorbin"))
        print(f"[cache] converted layer 0 in {time.time() - t0:.1f} s: {len(files)} files, {size / 1e6:.1f} MB; "
              f"complete {layer_cache_complete(cfg, 0)}; times {times}")
        assert layer_cache_complete(cfg, 0) and not layer_cache_complete(cfg, None)
        marker = W.layer_cache_marker(cfg, 0).read_text()
        assert cfg.cache_version_tag in marker
        again = convert_weights(mesh_device, cfg, source=src, layers=[0], include_globals=False,
                                min_free_gb=ROUNDTRIP_MIN_FREE_GB)
        assert again == {}, again  # resumable: a complete part is skipped
        guard = _GuardSource(src, "model.layers.0.")
        t0 = time.time()
        model = MotifModel(mesh_device, cfg, source=guard, layers=[0], cache="auto")
        try:
            print(f"[cache] model (layer 0 from the TT cache, globals from the checkpoint) built in "
                  f"{time.time() - t0:.1f} s; source reads: {sorted(set(guard.reads))}; misses {model.cache_misses}")
            assert all(not n.startswith("model.layers.") for n in guard.reads)
            assert not model.cache_misses, model.cache_misses
            assert not (cfg.cache_dir / "global").exists() or not any((cfg.cache_dir / "global").iterdir())
            p = prompts[0]
            S = len(p.ids)
            tok = model.embed.prefill_tokens_device(torch.tensor(p.ids, dtype=torch.int32), bucket_of(S))
            X = model.prefill(tok, return_streams=True)
            got = read_streams(X, S)
            _free(tok, X)
        finally:
            model.deallocate()
        s = stats(goldens[0][p.name].float(), got)
        print(f"[cache] {p.name}: streams after layer 0 (cached weights) {fmt(s)}")
        assert s["pcc"] >= 0.995

        # ---- "auto" never writes (finding 4): an option variant the converter did not build -----------------------
        before = {q.name: q.stat().st_size for q in l0.iterdir()}
        model = MotifModel(mesh_device, cfg, source=src, layers=[0], cache="auto", layer_kwargs={"sinkhorn": "stock"})
        try:
            after = {q.name: q.stat().st_size for q in l0.iterdir()}
            misses = model.cache_misses.get("L00", [])
            print(f"[cache] layer 0 with sinkhorn='stock' under cache='auto': {len(misses)} tensors uploaded from the "
                  f"checkpoint ({misses}); files before {len(before)}, after {len(after)}")
            assert after == before, sorted(set(after) ^ set(before))
            assert misses and all("_row" in n for n in misses), misses
            tok = model.embed.prefill_tokens_device(torch.tensor(p.ids, dtype=torch.int32), bucket_of(S))
            X = model.prefill(tok, return_streams=True)
            got = read_streams(X, S)
            _free(tok, X)
        finally:
            model.deallocate()
        s = stats(goldens[0][p.name].float(), got)
        print(f"[cache] {p.name}: streams after layer 0 (stock Sinkhorn, mixed cache / checkpoint) {fmt(s)}")
        assert s["pcc"] >= 0.995
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ============================================================================================================
# (3b) host-only: disk guards (findings 2, 3), the read-only cache (finding 4), the prefill page table (finding 7)
# ============================================================================================================
def _host_cfg(tmp_path):
    from models.demos.motif3.tt.model_config import MotifTTConfig

    return MotifTTConfig.from_hf_config(mesh_shape=(4, 8), tt_cache_root=tmp_path / "ttc", max_model_len=2048)


def test_host_disk_guard_and_part_estimates(tmp_path, monkeypatch):
    """The converter / ``cache="write"`` guard: free - 1.1 x part >= 60 GB; MoE parts 6.65 GB, globals 3.6 GB. A
    convert_weights call on a nearly full disk stops before building anything (fake mesh / modules)."""
    from models.demos.motif3.tt import model as M

    cfg = _host_cfg(tmp_path)
    assert M.estimate_part_bytes(cfg, None) == 3_607_113_408
    assert 6.6e9 < M.estimate_part_bytes(cfg, 2) < 6.7e9 and 3.5e8 < M.estimate_part_bytes(cfg, 0) < 3.6e8
    moe = M.estimate_part_bytes(cfg, 2)
    assert M.room_ok(80e9, moe, 60.0) and not M.room_ok(66e9, moe, 60.0)  # 66 - 7.3 < 60: the old guard passed this
    monkeypatch.setattr(M, "free_bytes", lambda path: int(64e9))
    with pytest.raises(M.DiskGuardError, match="floor is 60"):
        M.check_room(tmp_path, moe, "a MoE layer")
    M.check_room(tmp_path, M.estimate_part_bytes(cfg, 0), "a dense layer")  # 64 - 0.39 >= 60

    built = []

    class _Rope:
        def release_prefill_tables(self):
            pass

    monkeypatch.setattr(M, "MotifCCL", lambda *a, **k: object())
    monkeypatch.setattr(M, "MotifRope", lambda *a, **k: _Rope())
    monkeypatch.setattr(M, "MotifDecoderLayer", lambda *a, **k: built.append(a) or pytest.fail("built a layer"))
    with pytest.raises(M.DiskGuardError) as ei:
        M.convert_weights(None, cfg, source=object(), layers=[2], include_globals=False)
    assert ei.value.converted == {} and not built
    assert not M.layer_cache_complete(cfg, 2)


def _fake_repo(tmp_path):
    """A 2-shard safetensors repo (layer 0 tensors in shard 1, layer 1 in shard 2) + a fake huggingface_hub."""
    from safetensors.torch import save_file

    repo = tmp_path / "repo"
    repo.mkdir()
    shards = {"model-00001-of-00002.safetensors": {"model.layers.0.w": torch.ones(4, 4)},
              "model-00002-of-00002.safetensors": {"model.layers.1.w": torch.full((2, 3), 2.0)}}
    wm = {}
    for fn, tensors in shards.items():
        save_file(tensors, str(repo / fn))
        wm.update({k: fn for k in tensors})
    (repo / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    snap = tmp_path / "hub" / "models--org--fake" / "snapshots" / "abc1234"

    class _Hub:
        calls = []

        @staticmethod
        def hf_hub_download(repo_id, filename, revision=None, cache_dir=None):
            _Hub.calls.append(filename)
            snap.mkdir(parents=True, exist_ok=True)
            (snap / filename).write_bytes((repo / filename).read_bytes())
            return str(snap / filename)

    return _Hub


def test_host_repo_id_source_fetches_only_needed_shards(tmp_path, monkeypatch):
    """Finding 3: an uncached repo id never ``snapshot_download``s the repo. The index first, then only the shard of a
    tensor that is read; a shard that would cross the 60 GB floor is refused with a pointer to download_weights.py."""
    from models.demos.motif3.tt import model as M

    hub = _fake_repo(tmp_path)
    monkeypatch.setattr(M, "free_bytes", lambda path: int(500e9))
    src = M.LazySource(repo_id="org/fake", revision="abc1234", hub=hub, log=lambda m: None)
    assert hub.calls == [] and not src.opened  # nothing before the first use
    assert src.has("model.layers.1.w") and not src.layer_available(1)
    assert hub.calls == ["model.safetensors.index.json"]
    t = src.get("model.layers.1.w")
    assert torch.equal(t, torch.full((2, 3), 2.0))
    assert hub.calls == ["model.safetensors.index.json", "model-00002-of-00002.safetensors"]
    src.get("model.layers.1.w")  # local now: no second download
    assert len(hub.calls) == 2 and src.fetched == ["model-00002-of-00002.safetensors"]
    monkeypatch.setattr(M, "free_bytes", lambda path: int(61e9))  # 61 GB - 1.1 x shard (8 GB fallback) < 60 GB
    monkeypatch.setattr(M, "_tree_sizes", lambda: {})
    src2 = M.LazySource(repo_id="org/fake", revision="abc1234", hub=hub, log=lambda m: None)
    src2._resolve()._sizes = {}
    with pytest.raises(M.DiskGuardError, match="download_weights.py"):
        src2.get_rows("model.layers.0.w", 0, 2)
    assert "model-00001-of-00002.safetensors" not in hub.calls


def test_host_read_only_cache(tmp_path, monkeypatch):
    """Finding 4: inside ``read_only_cache`` a cached tensor loads from its file (cache name kept) and a missing one is
    uploaded with ``cache_name=None`` (nothing written); the module attribute is restored afterwards."""
    from models.demos.motif3.tt import model as M
    from models.demos.motif3.tt import weights as W

    cfg = _host_cfg(tmp_path)
    seen = []

    def recorder(src, *, mesh_device, cfg, dtype, layout=ttnn.TILE_LAYOUT, memory_config=None, dp_dim=None,
                 tp_dim=None, cache_name=None, layer=None):
        seen.append(cache_name)
        return cache_name

    monkeypatch.setattr(W, "as_tensor", recorder)
    have = W.tensorbin_path(W.cache_prefix(cfg, "attn.v2.wo", 3, None, 1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    have.parent.mkdir(parents=True, exist_ok=True)
    have.write_bytes(b"x")
    kw = dict(mesh_device=None, cfg=cfg, dtype=ttnn.bfloat16, layer=3)
    with M.read_only_cache() as misses:
        assert W.as_tensor(None, cache_name="attn.v2.wo", tp_dim=1, **kw) == "attn.v2.wo"
        assert W.as_tensor(None, cache_name="moe.experts.down_x1", dp_dim=0, tp_dim=1, **kw) is None
        assert W.as_tensor(None, cache_name=None, **kw) is None
    assert seen == ["attn.v2.wo", None, None]
    assert misses == [W.tensorbin_path(W.cache_prefix(cfg, "moe.experts.down_x1", 3, 0, 1), ttnn.bfloat16,
                                       ttnn.TILE_LAYOUT).name]
    assert W.as_tensor is recorder
    with pytest.raises(RuntimeError):
        with M.read_only_cache():
            raise RuntimeError("boom")
    assert W.as_tensor is recorder


def test_host_demo_convert_runs_the_production_converter(monkeypatch):
    """Finding 2: ``demo.py --convert`` hands the conversion to ``scripts/convert_weights.py`` (its 60 GB floor, staging
    and verification) in a child process -- mock target with the devices hidden -- and returns its exit code."""
    from models.demos.motif3.demo import demo

    script = demo.production_converter()
    if script is None:
        pytest.skip("scripts/convert_weights.py not present")
    calls = []
    real_listdir = os.listdir
    monkeypatch.setattr(demo.subprocess, "call", lambda cmd: calls.append(cmd) or 3)
    monkeypatch.setattr(demo.os, "listdir", lambda p: [] if str(p) == "/dev/tenstorrent" else real_listdir(p))
    rc = demo.convert(demo.parse_args(["--num-layers", "4", "--convert"]), 4, "/w/Motif-3", None)
    assert rc == 3 and len(calls) == 1
    cmd = calls[0]
    assert cmd[1] == str(script) and cmd[cmd.index("--target") + 1] == "mock"
    assert cmd[cmd.index("--layers") + 1] == "0-3" and "--globals" in cmd and cmd[-1] == "/w/Motif-3"


def test_host_demo_top1_report_is_tie_aware():
    from models.demos.motif3.demo.demo import top1_vs_golden

    ids = torch.tensor([[5, 6, 7], [1, 2, 3], [9, 8, 4]])
    vals = torch.tensor([[2.0, 2.0, 1.0], [3.0, 1.0, 0.0], [1.0, 0.9, 0.0]])
    am = torch.tensor([5, 1, 9])
    m = top1_vs_golden(torch.tensor([6, 1, 8]), am, ids, vals)  # row 0: tie (agrees), row 1: exact, row 2: miss
    assert m["exact"] == pytest.approx(1 / 3) and m["agree"] == pytest.approx(2 / 3)
    assert m["n_sel"] == 1 and m["agree_sel"] == 1.0 and m["in5"] == 1.0


def test_host_prefill_page_table_zeroes_the_tail():
    """Finding 7: the generator writes the bucket padding through the null block whatever the caller left past the
    request's own blocks."""
    from models.demos.motif3.tt.generator import prefill_page_table_host

    stale = torch.tensor([5, 9, 13, 21, 34, 55, 89, 144], dtype=torch.int32)
    pt = prefill_page_table_host(stale, entries=4, seq_len=100, block_size=64)  # bucket 256: 4 entries, 2 own
    assert pt.tolist() == [[5, 9, 0, 0]] and pt.dtype == torch.int32
    assert prefill_page_table_host(stale, 2, 128, 64).tolist() == [[5, 9]]
    assert prefill_page_table_host(stale[:1], 4, 60, 64).tolist() == [[5, 0, 0, 0]]
    assert prefill_page_table_host(stale, 16, 1000, 64).tolist() == [stale.tolist() + [0] * 8]  # short table


# ============================================================================================================
# (4) model-level teacher-forced metric: 36 layers vs the golden early-exit head after layer 35 (opt-in, ~25 min)
# ============================================================================================================
TF_LAYERS = int(os.environ.get("MOTIF3_TF_LAYERS", "36"))
TF_STEPS = int(os.environ.get("MOTIF3_TF_STEPS", "32"))
TF_MAX_MODEL_LEN = int(os.environ.get("MOTIF3_TF_MAX_MODEL_LEN", "32768"))  # serving: 9 buckets, page-table width 512
TF_NUM_BLOCKS = int(os.environ.get("MOTIF3_TF_NUM_BLOCKS", "4129"))  # the serving pool (README §3)
TF_ROUTER_AB = os.environ.get("MOTIF3_TF_ROUTER_AB", "1") == "1"  # decision D1: exact_fp32 vs composite decode router
TF_PROXY53 = os.environ.get("MOTIF3_TF_PROXY53", "1") == "1"  # finding 10: the 53-layer full-scale proxy
# Acceptance after layer 35 (calibrated against the reference's own floor, module docstring; decided before the run):
# (a) vs the bf16 golden, rows with a margin > TF_MARGIN: tie-aware agreement >= TF_MIN_ABOVE (floor 0.9936);
# (b) vs the reference's fp32 run (when present): TT agrees with it at most TF_FLOOR_SLACK below the bf16 golden's own
#     agreement with it, on the same rows, overall and above margin TOP1_MARGIN (fp32 margins).
TF_MARGIN = 1.0
TF_MIN_ABOVE = 0.98
TF_FLOOR_SLACK = 0.02
TF_FLOOR_SLACK_OVERALL = 0.03
D1_TARGET = 0.95


def _q(t: torch.Tensor, qs=(0.5, 0.9, 0.99, 1.0)) -> str:
    v = torch.quantile(t.double(), torch.tensor(qs, dtype=torch.float64))
    return " / ".join(f"{float(x):.2e}" for x in v)


def _token_rel(ref: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    """Per-token relative error ``||got - ref|| / ||ref||`` over the 4 x 4096 stream values: ``[S]``."""
    r, g = ref.double().flatten(1), got.double().flatten(1)
    return (g - r).norm(dim=1) / r.norm(dim=1).clamp_min(1e-30)


class _Goldens:
    """The bf16 golden head after layer N-1 and, when present, the reference's fp32 run (head + state)."""

    def __init__(self, N: int, prompts):
        from models.demos.motif3.reference import golden_stream as gs

        sha = gs.prompt_set_sha256(prompts)
        lp, _ = gs.head_paths(GOLDEN_DIR, N - 1)
        self.bf16 = gs.load_tensors(lp)[0]
        self.fp32 = fp32_golden_or_none(N - 1, sha)
        st = fp32_states_or_none(GOLDEN_FP32_DIR, (N - 1,), sha)
        self.fp32_state = None if st is None else st[N - 1]
        print(f"[tf] goldens: bf16 head after layer {N - 1}; fp32 run: "
              f"{'head + state' if self.fp32 is not None else 'absent'} ({GOLDEN_FP32_DIR})")

    def rows(self, which: str, name: str, pos):
        g = self.bf16 if which == "bf16" else self.fp32
        return g[f"{name}.topk_ids"][pos], g[f"{name}.topk_logits"][pos].float(), g[f"{name}.argmax"][pos]

    def compare(self, name: str, pos, logits: torch.Tensor) -> dict:
        """Candidate logits ``[n, V]`` at positions ``pos`` of prompt ``name``: metrics vs the bf16 golden and, when
        present, vs the fp32 run, with the floors (the other run's argmax) on the same rows."""
        out = {"bf16": top1_topk(*self.rows("bf16", name, pos), logits)}
        if self.fp32 is not None:
            ids, vals, am = self.rows("bf16", name, pos)
            ids32, vals32, am32 = self.rows("fp32", name, pos)
            out["bf16_floor"] = floor_topk(ids, vals, am, am32)  # the fp32 run vs the bf16 golden
            out["fp32"] = top1_topk(ids32, vals32, am32, logits)
            out["fp32_floor"] = floor_topk(ids32, vals32, am32, am)  # the bf16 golden vs the fp32 run
        return out


def tf_checks(tag: str, parts: list, failures: list) -> dict:
    """Print and check (a) / (b) above on rows ``parts`` (dicts of :meth:`_Goldens.compare`)."""
    m = {k: cat_rows([p[k] for p in parts]) for k in parts[0]}
    b = bands(m["bf16"])
    print(f"[tf] {tag} vs the bf16 golden: {fmt_bands(b)}; top-32 max err median/p90/p99/max {_q(m['bf16']['err32'])}; "
          f"top-32 pcc p1/p10/median {_q(m['bf16']['pcc32'], (0.01, 0.1, 0.5))}")
    failures += top1_failures(f"{tag} vs bf16 golden", m["bf16"], min_overall=0.0, margin=TF_MARGIN,
                              min_above=TF_MIN_ABOVE)
    line = f"[tf] {tag}: D1 target top-1 >= {D1_TARGET} vs the bf16 golden: TT exact {b['exact']:.4f}"
    if "fp32" in m:
        fb = bands(m["bf16_floor"])
        line += f", the reference's own fp32 run {fb['exact']:.4f} (floor)"
        print(f"[tf]   floor: the fp32 run vs the bf16 golden on the same rows: {fmt_bands(fb)}")
        t32, f32 = bands(m["fp32"]), bands(m["fp32_floor"])
        print(f"[tf]   {tag} vs the fp32 run: {fmt_bands(t32)}")
        print(f"[tf]   floor: the bf16 golden vs the fp32 run on the same rows: {fmt_bands(f32)}")
        for lab, tt_v, fl_v, slack in (
            ("overall", t32["agree"], f32["agree"], TF_FLOOR_SLACK_OVERALL),
            (f"margin > {TOP1_MARGIN:g}", *[(n - k) / max(n, 1) for n, k in (t32["above"][TOP1_MARGIN],
                                                                          f32["above"][TOP1_MARGIN])], TF_FLOOR_SLACK),
        ):
            ok = tt_v >= fl_v - slack
            print(f"[tf]   (b) {lab}: TT vs fp32 {tt_v:.4f}, bf16 golden vs fp32 {fl_v:.4f} -> "
                  f"{'ok' if ok else 'FAIL'} (slack {slack})")
            if not ok:
                failures.append(f"{tag}: TT agrees with the fp32 run {tt_v:.4f} ({lab}), the bf16 golden "
                                f"{fl_v:.4f}: more than {slack} below the reference's own floor")
    print(line)
    return m


@torch.no_grad()
def depth_sweep(model, prompts, N: int, gold: _Goldens, failures: list) -> None:
    """Prefill every C2 prompt through the TT layers 0..N-1 (no KV) and compare the 4-stream state with the golden after
    every saved depth: full-state PCC and the per-token relative error (median / p90 / p99 / max; massive-activation
    tokens dominate the PCC); at layer N-1 also vs the reference's fp32 run (and the bf16 golden's own error vs it).
    Then the reference's own bf16 head (CPU) on the TT state after layer N-1 vs the golden heads at ALL positions
    (the state error alone, without the TT head): :func:`tf_checks`."""
    from models.demos.motif3.reference import golden_stream as gs

    depths = [l for l in (0, 1, 2, 3, 4, 7, 8, 15, 16, 23, 24, 31, 32, 35) if l < N]
    g = {l: {k: v[0] for k, v in gs.load_tensors(gs.state_path(GOLDEN_DIR, l))[0].items()} for l in depths}
    rel = {l: [] for l in depths}
    pcc = {l: ([], []) for l in depths}
    final = {}
    t0 = time.time()
    for p in prompts:
        S = len(p.ids)
        tok = model.embed.prefill_tokens_device(torch.tensor(p.ids, dtype=torch.int32), bucket_of(S))
        X = model.embed.forward_prefill(tok)
        _free(tok)
        for i, layer in enumerate(model.layers):
            Xn = layer.forward_prefill(X)
            _free(X)
            X = Xn
            if i in g:
                got = read_streams(X, S)
                ref = g[i][p.name].float()
                rel[i].append(_token_rel(ref, got))
                pcc[i][0].append(ref)
                pcc[i][1].append(got)
                if i == N - 1:
                    final[p.name] = got
        _free(X)
    print(f"[depth N={N}] prefill sweep of {len(prompts)} prompts in {time.time() - t0:.1f} s")
    for l in depths:
        r = torch.cat(rel[l])
        s = stats(torch.cat(pcc[l][0]), torch.cat(pcc[l][1]))
        print(f"[depth N={N}] after layer {l:2d}: state pcc={s['pcc']:.6f}; per-token rel err median/p90/p99/max "
              f"{_q(r)}")
    del pcc
    if gold.fp32_state is not None and final:
        tt_rel = torch.cat([_token_rel(gold.fp32_state[p.name].float(), final[p.name]) for p in prompts])
        bf_rel = torch.cat([_token_rel(gold.fp32_state[p.name].float(), g[N - 1][p.name].float()) for p in prompts])
        print(f"[depth N={N}] after layer {N - 1} vs the reference's fp32 run: per-token rel err median/p90/p99/max "
              f"TT {_q(tt_rel)}; the bf16 golden itself {_q(bf_rel)}")
    if not final:
        return
    head = RefHead(model.source)
    parts = []
    for p in prompts:
        x = final[p.name]
        for c0 in range(0, x.shape[0], 256):
            pos = torch.arange(c0, min(c0 + 256, x.shape[0]))
            parts.append(gold.compare(p.name, pos, head(x[c0: c0 + 256])))
    tf_checks(f"reference head on the TT state after layer {N - 1} (all {sum(len(p.ids) for p in prompts)} positions)",
              parts, failures)


def _set_decode_router(model, exact_fns) -> None:
    """Decision-D1 A/B in one session: ``exact_fns`` = the routers' exact-fp32 kernels (decode uses them) or None
    (composite decode logits: the MoE built with ``router_logits="exact_fp32"`` keeps the composite weights, prefill
    uses them; ``MotifRouter.logits_fn`` selects the decode path)."""
    for i, layer in enumerate(model.layers):
        if layer.moe is not None:
            layer.moe.router.logits_fn = None if exact_fns is None else exact_fns[i]


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@torch.no_grad()
def test_model_teacher_forced(mesh_device, device_params):
    """``MOTIF3_RUN_TF=1`` only. The generator with ``MOTIF3_TF_LAYERS`` (36) real layers at the serving geometry
    (max_model_len 32768: 9 buckets, page-table width 512; the 4129-block pool), then:

    1. memory: DRAM per chip after the weights and the pool, the trace region after the capture;
    2. :func:`depth_sweep` (prefill states vs the goldens at every saved depth; the reference head on the TT state at
       all positions vs the golden heads, checks (a) / (b));
    3. teacher forcing on 30 lanes (5 offsets of each C2 prompt): prefill of each prefix + ``MOTIF3_TF_STEPS`` traced
       decode steps, every next-token prediction (TT head) vs the golden heads: checks (a) / (b); next-token accuracy;
    4. ``MOTIF3_TF_ROUTER_AB`` (on): the model is built with ``router_logits="exact_fp32"``; step 3 runs with the exact
       decode router and again with the composite one (``cfg.router_logits`` default; the checks apply to it);
    5. ``MOTIF3_TF_PROXY53`` (on): :func:`full_scale_proxy`."""
    if os.environ.get("MOTIF3_RUN_TF") != "1":
        pytest.skip("opt-in: MOTIF3_RUN_TF=1 (loads 36 real layers, ~25 min)")
    from models.demos.motif3.reference import golden_stream as gs
    from models.demos.motif3.tt import generator_api as api
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR

    N = TF_LAYERS
    lp, _ = gs.head_paths(GOLDEN_DIR, N - 1)
    if not lp.is_file():
        pytest.skip(f"no golden head after layer {N - 1}")
    prompts = gs.load_prompt_set(GOLDEN_DIR / "prompts.json")
    real_source_or_skip(range(N))
    log_fabric(mesh_device, f"model_teacher_forced N={N}")
    gold = _Goldens(N, prompts)
    mem0 = mem_line(mesh_device, "mesh opened")
    settings = api.GeneratorSettings(max_batch_size=api.NUM_LANES, max_seq_len=TF_MAX_MODEL_LEN, num_layers=N,
                                     weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=BLOCK)
    kw = {"layer_kwargs": {"router_logits": "exact_fp32"}} if TF_ROUTER_AB else {}
    t0 = time.time()
    gen = MotifGenerator.create(hf_config=None, mesh_device=mesh_device, settings=settings, **kw)
    t_create = time.time() - t0
    model = gen.model
    exact_fns = [l.moe.router.logits_fn if l.moe is not None else None for l in model.layers]
    failures = []
    try:
        mem1 = mem_line(mesh_device, f"{N} layers loaded")
        pool = gen.allocate_kv_cache(num_blocks=TF_NUM_BLOCKS, block_size=BLOCK, num_layers=N)
        mem2 = mem_line(mesh_device, f"KV pool {N} x [{TF_NUM_BLOCKS}, 1, {BLOCK}, 576]")
        W = gen.cfg.kv_blocks_per_seq
        t0 = time.time()
        gen.warmup_prefill(kv_cache=pool, enable_trace=False)
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=W)
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
        mem3 = mem_line(mesh_device, f"warmup ({len(gen.cfg.prefill_buckets)} buckets) + decode capture (W={W})")
        print(f"[tf N={N}] create {t_create:.1f} s; warmup + capture {time.time() - t0:.1f} s ({gen.timings})")
        memory_projection(model, mem0, mem1, mem2, mem3, N)
        if os.environ.get("MOTIF3_TF_DEPTH_SWEEP", "1") == "1":
            depth_sweep(model, prompts, N, gold, failures)
        arms = [("exact_fp32", exact_fns), ("composite", None)] if TF_ROUTER_AB else [(gen.cfg.router_logits, None)]
        results = {}
        for name, fns in arms:
            if TF_ROUTER_AB:
                _set_decode_router(model, fns)
                if gen.trace_captured and name != arms[0][0]:  # a new decode router: recompile eagerly, recapture
                    gen.release_traces()
                    gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=W)
                    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
            results[name] = tf_arm(gen, pool, W, prompts, gold, N, name, api)
        default = gen.cfg.router_logits if gen.cfg.router_logits in results else list(results)[-1]
        for name, (parts, info) in results.items():
            sub = None if name == default else []  # the serving default's checks are asserted, the other arm reported
            m = tf_checks(f"TT decode router {name} ({info})", parts, failures if sub is None else sub)
            if sub is not None:
                print(f"[tf]   (router {name} is not the serving default: its checks are reported only: "
                      f"{sub or 'all pass'})")
            results[name] = m
        if len(results) == 2:
            a, c = bands(results["exact_fp32"]["bf16"]), bands(results["composite"]["bf16"])
            print(f"[tf] D1 router A/B vs the bf16 golden: exact_fp32 tie-aware {a['agree']:.4f} / margin > {TF_MARGIN:g} "
                  f"{1 - a['above'][TF_MARGIN][1] / max(a['above'][TF_MARGIN][0], 1):.4f}; composite {c['agree']:.4f} / "
                  f"{1 - c['above'][TF_MARGIN][1] / max(c['above'][TF_MARGIN][0], 1):.4f}")
        if TF_PROXY53:
            failures += full_scale_proxy(gen, pool, W, prompts, N, mesh_device, api)
    finally:
        _set_decode_router(model, exact_fns)  # the MoE owns (and frees) its exact kernels
        gen.close()
    assert not failures, "\n".join(failures)


def tf_arm(gen, pool, W, prompts, gold: _Goldens, N: int, tag: str, api):
    """30 lanes (5 offsets of each C2 prompt): prefill of each prefix, then ``TF_STEPS`` traced teacher-forced decode
    steps; every prediction compared with the golden heads. Returns ``(parts, info)``."""
    lanes, nxt = [], 1
    for r in range(30):
        p = prompts[r % len(prompts)]
        q0 = max(16, len(p.ids) - TF_STEPS - 1 - 40 * (r // len(prompts)))
        nb = math.ceil((q0 + TF_STEPS + 1) / BLOCK)
        lanes.append(dict(lane=(r % 4) * 8 + r // 4, p=p, q0=q0, blocks=list(range(nxt, nxt + nb))))
        nxt += nb
    parts, tgt = [], []

    def record(name, position, logits):
        parts.append(gold.compare(name, torch.tensor([position]), logits.float()[None]))
        tgt.append((int(logits.float().argmax()), int(gold.bf16[f"{name}.argmax"][position]),
                    int(gold.bf16[f"{name}.target_ids"][position])))

    t0 = time.time()
    for d in lanes:
        pt = torch.zeros(W, dtype=torch.int32)
        own = math.ceil(d["q0"] / BLOCK)
        pt[:own] = torch.tensor(d["blocks"][:own], dtype=torch.int32)
        logits = gen.prefill_forward(api.PrefillRequest(lane=d["lane"], tokens=torch.tensor(d["p"].ids[: d["q0"]],
                                                        dtype=torch.int32), page_table=pt), kv_cache=pool)
        record(d["p"].name, d["q0"] - 1, logits)
    t_pf = time.time() - t0
    step_ms = []
    for t in range(TF_STEPS):
        tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
        pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
        table = torch.zeros(api.NUM_LANES, W, dtype=torch.int32)
        for d in lanes:
            q = d["q0"] + t
            tokens[d["lane"]], pos[d["lane"]] = d["p"].ids[q], q
            table[d["lane"], : q // BLOCK + 1] = torch.tensor(d["blocks"][: q // BLOCK + 1], dtype=torch.int32)
        t1 = time.perf_counter()
        out = gen.decode_forward(api.DecodeBatch(tokens=tokens, positions=pos, page_table=table), kv_cache=pool,
                                 enable_trace=True)
        step_ms.append((time.perf_counter() - t1) * 1e3)
        for d in lanes:
            record(d["p"].name, d["q0"] + t, out[d["lane"]])
    valid = [x for x in tgt if x[2] >= 0]
    acc_dev = sum(a == t for a, _, t in valid) / len(valid)
    acc_ref = sum(g == t for _, g, t in valid) / len(valid)
    info = (f"{len(tgt)} predictions: {len(lanes)} prefills ({t_pf:.1f} s) + {TF_STEPS} decode steps (median "
            f"{statistics.median(step_ms):.1f} ms/step); next-token accuracy TT {acc_dev:.4f} vs golden {acc_ref:.4f}")
    print(f"[tf N={N}] arm {tag}: {info}")
    return parts, info


def memory_projection(model, mem0, mem1, mem2, mem3, N: int) -> None:
    """Finding 10: the measured DRAM per chip of the globals, a dense layer, a MoE layer and the KV pool, projected to
    the 53-layer model with the serving pool; the decode trace bytes per layer."""
    pdb = model.part_dram_bytes
    moe = [v for k, v in pdb.items() if k != "global" and model.cfg.layer(int(k[1:])).is_moe]
    dense = [v for k, v in pdb.items() if k != "global" and not model.cfg.layer(int(k[1:])).is_moe]
    if not moe or mem0["dram"] is None or mem2["dram"] is None:
        print("[mem] no allocator view: projection skipped")
        return
    g, dmed, mmed = pdb.get("global", 0), statistics.median(dense) if dense else 0, statistics.median(moe)
    kv_layer = (mem2["dram"]["allocated"] - mem1["dram"]["allocated"]) / N
    proj_w = g + 2 * dmed + 51 * mmed
    proj = mem0["dram"]["allocated"] + proj_w + 53 * kv_layer
    tot = mem0["dram"]["total"]
    tr = mem3["trace"]["allocated"] if mem3["trace"] is not None else None
    print(f"[mem] per chip: globals {g / 1e9:.2f} GB, dense layer {dmed / 1e9:.3f} GB, MoE layer {mmed / 1e9:.3f} GB "
          f"(min {min(moe) / 1e9:.3f}, max {max(moe) / 1e9:.3f}), KV per layer {kv_layer / 1e9:.3f} GB; projected 53 "
          f"layers + pool: weights {proj_w / 1e9:.2f} GB + KV {53 * kv_layer / 1e9:.2f} GB + base "
          f"{mem0['dram']['allocated'] / 1e9:.2f} GB = {proj / 1e9:.2f} GB of {tot / 1e9:.2f} GB "
          f"({(tot - proj) / 1e9:.2f} GB left for activations)"
          + (f"; decode trace {tr / 2**20:.1f} MiB for {N} layers = {tr / N / 2**20:.2f} MiB per layer -> "
             f"~{tr / N * 53 / 2**20:.0f} MiB at 53 layers" if tr else ""))


def full_scale_proxy(gen, pool, W, prompts, N: int, mesh_device, api) -> list:
    """Finding 10 (53 layers are not on disk yet): the loaded layers plus 17 aliases make a 53-call model with the
    memory of the real one -- layer ``36 + i`` runs the module of layer ``16 + i`` (same kind: l % 4 preserved) on its
    OWN KV cache of the serving geometry, and a DRAM ballast stands in for the 17 missing layers' weights (17 x the
    median MoE layer, measured). Then, in that state: a 32000-token prefill (bucket 32768) through all 53 calls, a
    53-call decode step eager and as a trace (capture fits the trace region; replay == eager bitwise; latency), and the
    decode at position 32000 vs the prefill of 32001 tokens (two code paths over the same 32K history, PCC)."""
    from models.demos.motif3.tt.model import MotifKVPool

    model, cfg = gen.model, gen.cfg
    if N != 36 or len(model.layers) != 36:
        print(f"[proxy53] needs the 36-layer model, have {len(model.layers)}: skipped")
        return []
    failures = []
    base = list(model.layers)
    alias = [16 + i for i in range(17)]
    moe = [v for k, v in model.part_dram_bytes.items() if k != "global" and cfg.layer(int(k[1:])).is_moe]
    ballast, extra_kv, tid = [], [], None
    S = 32000
    ids = [i for _ in range(20) for p in prompts for i in p.ids][: S + 1]
    blk0 = 1000  # past the teacher-forcing lanes' blocks
    nb = math.ceil((S + 1) / BLOCK)
    blocks_a, blocks_b = list(range(blk0, blk0 + nb)), list(range(blk0 + nb, blk0 + 2 * nb))
    try:
        gen.release_traces()  # measure the 53-call trace alone
        need = int(17 * statistics.median(moe)) if moe else 0
        chunk = 1 << 29
        while need > 0:
            rows = max(32, min(chunk, need) // (4096 * 2) // 32 * 32)
            ballast.append(ttnn.empty([1, 1, rows, 4096], ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device,
                                      ttnn.DRAM_MEMORY_CONFIG))
            need -= rows * 4096 * 2
        for _ in alias:
            e = ttnn.empty([pool.num_blocks, 1, pool.block_size, cfg.kv_latent_dim], pool.dtype, ttnn.TILE_LAYOUT,
                           mesh_device, ttnn.DRAM_MEMORY_CONFIG)
            extra_kv.append(ttnn.fill(e, 0.0))
            ttnn.deallocate(e)
        pool53 = MotifKVPool(list(pool.layers) + extra_kv, list(model.layer_ids) + [36 + i for i in range(17)],
                             pool.num_blocks, pool.block_size, pool.dtype)
        model.layers = base + [base[l] for l in alias]
        mem_line(mesh_device, "proxy53: 36 layers + 17 aliases, ballast for 17 layers' weights, 53-layer serving pool")

        def prefill(n_tok, blocks):
            bucket = cfg.prefill_bucket(n_tok)
            tok = model.embed.prefill_tokens_device(torch.tensor(ids[:n_tok], dtype=torch.int32), bucket)
            pt = torch.zeros(1, cfg.prefill_page_table_entries(bucket), dtype=torch.int32)
            own = math.ceil(n_tok / BLOCK)
            pt[0, :own] = torch.tensor(blocks[:own], dtype=torch.int32)
            ptt = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                  mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
            t1 = time.time()
            tile = model.prefill(tok, page_table=ptt, kv_caches=pool53, last_index=n_tok - 1)
            out = model.head.prefill_logits_to_host(tile, n_tok - 1).float()
            dt = time.time() - t1
            _free(tok, ptt, tile)
            return out, dt

        lg, dt = prefill(S, blocks_a)
        print(f"[proxy53] prefill S={S} (bucket {cfg.prefill_bucket(S)}) through 53 layer calls: {dt:.2f} s; logits "
              f"finite {bool(torch.isfinite(lg).all())}")
        if not bool(torch.isfinite(lg).all()):
            failures.append("proxy53: non-finite prefill logits")
        tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
        posv = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
        table = torch.zeros(api.NUM_LANES, W, dtype=torch.int32)
        tokens[0], posv[0] = ids[S], S
        table[0, : S // BLOCK + 1] = torch.tensor(blocks_a[: S // BLOCK + 1], dtype=torch.int32)
        batch = api.DecodeBatch(tokens=tokens, positions=posv, page_table=table)
        gen._write_inputs(batch)
        d = gen._inputs

        def step():
            return model.decode(d["tokens"], rot_idxs=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_caches=pool53)

        t1 = time.perf_counter()
        out = step()
        eager = model.head.logits_to_host(out).float()
        t_eager = time.perf_counter() - t1
        _free(out)
        tr0 = mem_line(mesh_device, "proxy53 before the 53-call capture")["trace"]
        ttnn.synchronize_device(mesh_device)
        tid, tout = gen._capture(pool53)
        tr1 = mem_line(mesh_device, "proxy53 53-call decode trace captured")["trace"]
        gen._write_inputs(batch)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        traced = model.head.logits_to_host(tout).float()
        same = torch.equal(traced[0], eager[0])
        reps = []
        for _ in range(10):
            t1 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            reps.append(time.perf_counter() - t1)
        if tr0 is not None and tr1 is not None:
            print(f"[proxy53] 53-call decode trace: {(tr1['allocated'] - tr0['allocated']) / 2**20:.1f} MiB of the "
                  f"{tr1['total'] / 2**20:.0f} MiB trace region")
        print(f"[proxy53] decode step, 53 layer calls, one lane at position {S} (32 lanes traced): replay + sync median "
              f"{statistics.median(reps) * 1e3:.2f} ms (min {min(reps) * 1e3:.2f}); eager {t_eager * 1e3:.0f} ms; traced "
              f"== eager bitwise {same}")
        if not same:
            failures.append("proxy53: traced 53-call decode != eager")
        ttnn.release_trace(mesh_device, tid)
        tid = None
        _free(tout)
        ref, dt = prefill(S + 1, blocks_b)
        s = stats(ref, eager[0])
        print(f"[proxy53] decode at {S} (FlashMLA over the paged 32K history) vs prefill of {S + 1} tokens (SDPA), "
              f"53 calls: {fmt(s)}; top-1 {'same' if int(ref.argmax()) == int(eager[0].argmax()) else 'DIFF'} "
              f"(prefill {dt:.2f} s)")
        if not s["pcc"] >= LOGIT_PCC_MIN:
            failures.append(f"proxy53: decode at {S} vs prefill of {S + 1}: {fmt(s)}")
    except Exception as e:  # report: an out-of-memory here IS the finding
        failures.append(f"proxy53 failed: {type(e).__name__}: {e}")
        print(f"[proxy53] FAILED: {type(e).__name__}: {e}")
    finally:
        if tid is not None:
            try:
                ttnn.release_trace(mesh_device, tid)
            except Exception:
                pass
        model.layers = base
        _free(ballast, extra_kv)
    return failures
