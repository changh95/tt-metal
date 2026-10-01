# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 GDLA attention (``tt/attention.py``; design §2.3.4, §3.2-3.3; WAVE_A_REVIEW §5.4 ATTN-1..7).

Host-only (exact fp64 algebra of the per-chip dataflow + weight transforms vs the CPU reference; no device)::

    S=/home/ttuser/hchang/experiments/motif-3/scripts
    $S/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib -q \
        models/demos/motif3/tests/unit/test_attention.py -k cpu
    $S/hostrun.sh -- python models/demos/motif3/tests/unit/test_attention.py --long   # (re)build the long goldens

Device (lock wrapper; every test logs the committed fabric first)::

    $S/devrun.sh -t 1500 -n attention -- python -m pytest models/demos/motif3/tests/unit/test_attention.py \
        -k "not cpu" -s -p no:cacheprovider

Goldens: ``models/demos/motif3/reference`` ``GDLAttention`` (HF semantics) in **fp32** with the same bf16-valued weights
and inputs ("fp32-faithful"; the HF-bf16 module is reported next to it for context). Real weights: layers 0 (global,
YaRN) and 1 (SWA, window 129) from the local checkpoint; real inputs: the reference prefix model's
``layers.{0,1}.input_layernorm.out`` on a 2080-token chat prompt (``tt_cache/test/attention/real_inputs_L0-1.pt``,
made by :func:`make_real_inputs`, 34 MB). Random weights follow ``reference.weights.random_state_dict`` at real dims.
Long prefills (S = 8K-32K) compare a subset of query rows against a row-subset golden built from the reference
module's own methods (full-sequence K/V, selected query rows), cached on disk under a key of the reference sources and
the exact inputs (:func:`long_golden_key`).

Leak sensitivity (a zero cache slot only adds e^-m to the softmax denominator, so zero-filled caches cannot catch a
causal / window leak): every decode test fills every cache slot that is not history -- slots at and after a lane's
position, the null block 0, unassigned and spare blocks -- with plausible *stale* latents (other tokens' unit-RMS
latents and roped k_pe, :func:`stale_bank`), and prefills pad their bucket with non-zero inputs. FlashMLA is also
checked alone against fp64 goldens on its own Q and cache, per lane, and must match the exact mask better than every
plausible wrong one (key p+1 visible, window 128 / 130, window dropped / applied) wherever that one differs
(:func:`mla_goldens`, :func:`mla_score`); negative controls run the kernel with such a wrong mask on the same Q / cache
and must be flagged on every lane where it is distinguishable. Per-lane floors are 0.999 (a 128- or 130-key window
moves random-weight lanes to 0.994-0.998; the worst correct lane seen is 0.99945). Real weights cannot detect a window
off-by-one at all (their latents are too similar: 0.99993-0.999999 between the masks); the random-weight cases do.

L1_SMALL: every device test opens the mesh with ``l1_small_size`` (``MESH``; a local workaround until
``model_config.device_params()`` sets it, requested), except :func:`test_attention_l1_small_hazard`, which pins the
static-CB clash a decode step causes with the shared default. :func:`test_attention_serving_session` runs the serving
order (prefill, decode, global prefill at S >= 1024, decode) for bfp8 / bf16 KV caches and blocks 64 / 32.

Every device test prints ``[attn] ...`` lines with PCC / max-abs per case and latencies (eager / traced).
"""

from __future__ import annotations

import hashlib
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

import ttnn
from models.demos.motif3.tt import weights as W
from models.demos.motif3.tt.attention import (
    L1_SMALL_WARNING,
    RECOMMENDED_L1_SMALL_SIZE,
    MotifAttention,
    _AttnSource,
    hf_order_from_virtual,
    l1_small_bytes,
    lambda_expansion,
    latent_kv_weight_for_chip,
    latent_q_weight,
    noise_expansion,
    virtual_head_order,
    w_uk_virtual_for_chip,
    w_uv_virtual_for_chip,
    wq_b_gate_for_chip,
    wq_b_virtual_for_chip,
)
from models.demos.motif3.tt.model_config import DEFAULT_HF_META_DIR, PROJECT_ROOT, MotifTTConfig, device_params

HF_META = str(DEFAULT_HF_META_DIR)
TEST_DIR = PROJECT_ROOT / "tt_cache" / "test" / "attention"
REAL_INPUTS = TEST_DIR / "real_inputs_L0-1.pt"
ATTN_KEYS = ("wq_a", "q_norm", "wq_b", "wq_b_gate", "wkv_a", "kv_norm", "wkv_b", "lambda_proj", "wo")
# The shared device_params() sets no l1_small_size (requested); until it does, the module's tests open the mesh with
# the L1_SMALL region the module requires (tt/attention.py docstring).
L1_SMALL_SIZE = RECOMMENDED_L1_SMALL_SIZE
MESH = [pytest.param((4, 8), device_params(l1_small_size=L1_SMALL_SIZE), id="4x8-torus2d-l1small")]
MESH_SHARED_DEFAULT = [pytest.param((4, 8), device_params(), id="4x8-torus2d-shared-device-params")]
PCC_MIN = 0.999  # module output (aggregate over lanes / rows)
LANE_PCC_MIN = 0.999  # per lane / user / position
ISO_LANE_PCC_MIN = 0.9995  # FlashMLA alone vs fp64 on its own Q / cache, per lane
DISC_MIN = 2e-3  # a wrong-mask golden is "distinguishable" from the exact one when 1 - pcc(exact, wrong) > DISC_MIN


def log(msg: str) -> None:
    print(f"[attn] {msg}", flush=True)


# ======================================================================================================================
# pure torch helpers (no device)
# ======================================================================================================================
def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double().flatten()
    b = b.detach().double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = float(a.norm() * b.norm())
    return 1.0 if den == 0.0 else float((a @ b) / den)


def stats(ref: torch.Tensor, got: torch.Tensor) -> Dict[str, float]:
    r, g = ref.double(), got.double()
    return {
        "pcc": pcc(r, g),
        "max_abs": float((r - g).abs().max()),
        "rel_fro": float((r - g).norm() / max(float(r.norm()), 1e-30)),
        "ref_absmax": float(r.abs().max()),
        "nonfinite": int((~torch.isfinite(g)).sum()),
    }


def fmt(s: Dict[str, float]) -> str:
    return (
        f"pcc {s['pcc']:.6f} max_abs {s['max_abs']:.3e} rel_fro {s['rel_fro']:.3e} (|ref|max {s['ref_absmax']:.3f})"
        + (f" NONFINITE {s['nonfinite']}" if s["nonfinite"] else "")
    )


def ref_args(**kw):
    from models.demos.motif3.reference.config import MotifArgs

    return MotifArgs.from_hf_config(HF_META, **kw)


def random_attn_tensors(args, seed: int) -> Dict[str, torch.Tensor]:
    """Real-dim GDLA weights with ``reference.weights.random_state_dict`` statistics, bf16-representable fp32."""
    g = torch.Generator().manual_seed(seed)
    D, H, hd, v, r = args.hidden_size, args.num_attention_heads, args.head_dim, args.v_head_dim, args.kv_lora_rank

    def linear(o, i):
        return torch.randn(o, i, generator=g) * i**-0.5

    t = {
        "wq_a": linear(args.q_lora_rank, D),
        "q_norm": torch.rand(args.q_lora_rank, generator=g) * 0.5 + 0.5,
        "wq_b": linear(H * hd, args.q_lora_rank),
        "wkv_a": linear(r + args.qk_rope_head_dim, D),
        "kv_norm": torch.rand(r, generator=g) * 0.5 + 0.5,
        "wkv_b": linear(args.num_key_value_heads * (args.qk_nope_head_dim + v), r),
        "lambda_proj": linear(args.n_signal_heads, D),
        "wo": linear(D, args.n_signal_heads * v),
        "wq_b_gate": linear(args.n_signal_heads * v, args.q_lora_rank),
    }
    return {k: x.to(torch.bfloat16).float() for k, x in t.items()}


def real_attn_tensors(layer: int) -> Dict[str, torch.Tensor]:
    loader = W.HFWeightLoader()
    if not loader.layer_available(layer):
        pytest.skip(f"layer {layer} is not fully downloaded")
    return {k: loader.get(W.hf_name(layer, f"self_attn.{k}.weight")) for k in ATTN_KEYS}


def hf_source(tensors: Dict[str, torch.Tensor], layer: int) -> W.DictWeightSource:
    return W.DictWeightSource({W.hf_name(layer, f"self_attn.{k}.weight"): v for k, v in tensors.items()})


def ref_attention(args, layer: int, tensors: Dict[str, torch.Tensor], dtype=torch.float32):
    from models.demos.motif3.reference.modules import GDLAttention

    with torch.device("meta"):
        m = GDLAttention(args, layer)
    m.load_state_dict({f"{k}.weight": v.to(dtype) for k, v in tensors.items()}, strict=True, assign=True)
    return m.eval().requires_grad_(False)


def ref_rope(ref, positions: torch.Tensor, dtype):
    from models.demos.motif3.reference.rope import rope_cos_sin

    return rope_cos_sin(ref.inv_freq(), positions, dtype)


def ref_latents(ref, x: torch.Tensor, positions: torch.Tensor):
    """Reference cache entries for ``x [T, D]`` at ``positions [T]``: unit-RMS latent ``n`` (gamma-free, what the TT
    cache stores) and roped ``k_pe`` (``c = gamma * n`` is the reference cache entry)."""
    from models.demos.motif3.reference.rope import apply_rope

    dt = x.dtype
    kv = torch.nn.functional.linear(x, ref.wkv_a.weight)
    c_raw, kpe = kv[..., : ref.kv_rank], kv[..., ref.kv_rank :]
    cf = c_raw.float()
    n = (cf * torch.rsqrt(cf.pow(2).mean(-1, keepdim=True) + ref.kv_norm.eps)).to(dt)
    cos, sin = ref_rope(ref, positions[None], dt)
    k_pe = apply_rope(kpe[None, :, None, :], cos, sin)[0, :, 0, :]
    return n, k_pe


def load_real_inputs(layer: int, n: int) -> torch.Tensor:
    if not REAL_INPUTS.is_file():
        pytest.skip(f"{REAL_INPUTS} missing: run make_real_inputs() on the host (hostrun.sh)")
    d = torch.load(REAL_INPUTS, weights_only=True)
    x = d[f"L{layer}"]
    if x.shape[0] < n:
        pytest.skip(f"real inputs have {x.shape[0]} < {n} tokens")
    return x[:n].float()


def make_real_inputs(n_tokens: int = 2080, path: Path = REAL_INPUTS) -> Path:
    """Host-only (hostrun.sh): reference bf16 prefix model L0-1 on a chat prompt over motif3/README.md; stores the
    normalized attention inputs ``layers.{0,1}.input_layernorm.out`` (bf16)."""
    from models.demos.motif3.reference.golden import capture_model_goldens
    from models.demos.motif3.reference.tokenizer import encode_chat, load_tokenizer
    from models.demos.motif3.reference.weights import load_reference_model

    text = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize the following engineering notes and list open risks.\n\n" + text},
    ]
    ids = torch.tensor([encode_chat(msgs, load_tokenizer())[:n_tokens]])
    model = load_reference_model(layer_ids=(0, 1), dtype=torch.bfloat16, lazy_experts=True)
    with torch.no_grad():
        g = capture_model_goldens(model, ids, include=lambda k: k.endswith("input_layernorm.out"))
    out = {
        "input_ids": ids[0].clone(),
        "L0": g["prefill"]["layers.0.input_layernorm.out"][0].clone(),
        "L1": g["prefill"]["layers.1.input_layernorm.out"][0].clone(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return path


def stale_bank(ref, weights_kind: str, layer: int, n: int = 1024, seed: int = 0) -> torch.Tensor:
    """``[n, 576]`` plausible stale cache rows -- what a freed block of an earlier request holds: unit-RMS latents and
    roped k_pe of *other* tokens (random inputs, or real inputs at random offsets for real weights) at random
    positions, through the same reference projection."""
    g = torch.Generator().manual_seed(1000 + seed)
    if weights_kind == "real":
        real = load_real_inputs(layer, 2080)
        x = real[torch.randint(0, real.shape[0], (n,), generator=g)]
    else:
        x = torch.randn(n, 4096, generator=g).to(torch.bfloat16).float()
    nn_, kpe = ref_latents(ref, x, torch.randint(0, 32768, (n,), generator=g))
    return torch.cat([nn_, kpe], -1)


def stale_paged(bank: torch.Tensor, pool: int, block: int, seed: int = 1) -> torch.Tensor:
    """Host paged cache ``[pool, 1, block, 576]`` with every slot (incl. the null block 0) holding a stale row."""
    g = torch.Generator().manual_seed(2000 + seed)
    idx = torch.randint(0, bank.shape[0], (pool * block,), generator=g)
    return bank[idx].reshape(pool, 1, block, bank.shape[-1]).contiguous()


def _gather_cache(cache_host: torch.Tensor, pt_row: torch.Tensor, positions, block: int) -> torch.Tensor:
    """Rows ``[len(positions), 576]`` of a paged cache readback for one user's page-table row (vectorized)."""
    pos = positions if torch.is_tensor(positions) else torch.as_tensor(list(positions), dtype=torch.long)
    pos = pos.long()
    return cache_host[pt_row.long()[pos // block], 0, pos % block]


def mask_variants(p: int, window: Optional[int], swa_window: int = 129) -> Dict[str, Tuple[int, int]]:
    """Inclusive key ranges ``[lo, hi]`` of the exact decode mask at position ``p`` (``window`` keys incl. the current
    one, ``None`` = full causal) and of the plausible wrong masks that differ from it: a causal leak (key p+1 visible),
    an off-by-one window (128 / 130), the window dropped (SWA) or applied (global)."""
    lo = 0 if window is None else max(0, p - window + 1)
    out = {"exact": (lo, p), "key p+1 visible": (lo, p + 1)}
    alts = (
        {f"window {window - 1}": window - 1, f"window {window + 1}": window + 1, "window dropped": None}
        if window is not None
        else {f"window {swa_window} applied": swa_window}
    )
    for name, w in alts.items():
        lo2 = 0 if w is None else max(0, p - w + 1)
        if lo2 != lo:
            out[name] = (lo2, p)
    return out


def legacy_sdpa_window_mask(S: int, window: int, tile: int = 32) -> torch.Tensor:
    """``[S, S]`` bool "allowed" mask that tt-metal's legacy (fp32-acc) SDPA prefill kernel builds for a causal
    sliding window, emulated from ``generate_causal_sliding_window_mask`` (``sdpa/device/kernels/dataflow/
    dataflow_common.hpp``): per (Q tile, K tile) it uses the window of the Q tile's *first* row for the "fully
    allowed" test (``k_tile_start >= min_window_start``), so rows 1..31 of a Q tile also see up to 31 keys older than
    their own window; partially covered tiles get the exact per-row diagonal mask. The streaming kernel (fp32 acc off)
    builds the exact mask."""
    q = torch.arange(S)
    exact = (q[None, :] <= q[:, None]) & (q[None, :] > q[:, None] - window)
    allowed = torch.zeros(S, S, dtype=torch.bool)
    nt = S // tile
    for qt in range(nt):
        q0, q1 = qt * tile, qt * tile + tile - 1
        min_ws = max(0, q0 - window + 1)
        min_we, max_we = q0 + 1, q1 + 1  # exclusive
        for kt in range(nt):
            k0, k1 = kt * tile, kt * tile + tile - 1
            rows, cols = slice(q0, q1 + 1), slice(k0, k1 + 1)
            if k1 < min_ws or k0 >= max_we:
                continue  # fully masked
            if k0 >= min_ws and k1 < min_we:
                allowed[rows, cols] = True  # "fully allowed" judged on the first row's window only
            else:
                allowed[rows, cols] = exact[rows, cols]
    return allowed


def _rmsnorm(x: torch.Tensor, eps: float, w: Optional[torch.Tensor] = None) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w


def _mask(q_pos: torch.Tensor, k_pos: torch.Tensor, window: Optional[int]) -> torch.Tensor:
    m = k_pos[None, :] <= q_pos[:, None]
    if window is not None:
        m &= k_pos[None, :] > q_pos[:, None] - window
    return m


def emulate_tt(cfg: MotifTTConfig, tensors: Dict[str, torch.Tensor], layer: int, x: torch.Tensor, mode: str):
    """Torch emulation of the TT per-chip dataflow (both forms), summed over the 8 TP chips (= ``all_reduce(tp)``).

    ``mode="decode"``: absorbed MQA over the 576-wide latent, virtual head order, combine on contiguous slices;
    ``mode="prefill"``: expanded GQA from ``n @ E_pref`` with Q reordered to the HF order. Positions ``0..T-1``.
    Uses exactly the module's weight transforms; math in the dtype of ``x`` / ``tensors`` (fp64 for exactness)."""
    from models.demos.motif3.reference.rope import apply_rope, rope_cos_sin
    from models.demos.motif3.tt.rope import inv_freq_for_kind

    spec = cfg.layer(layer)
    T = x.shape[0]
    pos = torch.arange(T)
    cos, sin = rope_cos_sin(inv_freq_for_kind(cfg, spec.rope_kind), pos[None], x.dtype)
    src = _AttnSource(hf_source(tensors, layer), layer)
    H, G, r, Sg = cfg.q_heads_per_chip, cfg.kv_groups_per_chip, cfg.grouped_ratio, cfg.signal_heads_per_chip
    nope, rd, v, rank = cfg.qk_nope_head_dim, cfg.rope_dim, cfg.v_head_dim, cfg.kv_lora_rank
    eps = cfg.rms_norm_eps
    mask = _mask(pos, pos, spec.sliding_window_size)
    E = lambda_expansion(cfg).to(x.dtype)
    Wq = latent_q_weight(src, cfg)
    out = torch.zeros(T, cfg.hidden_size, dtype=x.dtype)

    def rope(t):  # [T, h, 64]
        return apply_rope(t[None], cos, sin)[0]

    def attend(q, k, val):  # q [T, d], k [T, d], val [T, dv]
        s = (q @ k.T).masked_fill(~mask, float("-inf"))
        return torch.softmax(s, dim=-1) @ val

    X = noise_expansion(cfg).to(x.dtype)
    for tp in range(cfg.tp):
        cq_n = _rmsnorm(x @ Wq, eps)  # gamma_q is folded into wq_b / wq_b_gate
        kvl = x @ latent_kv_weight_for_chip(src, cfg, tp)
        n = _rmsnorm(kvl[:, :rank], eps)
        k_pe = rope(kvl[:, rank : rank + rd][:, None, :])[:, 0]
        lam = kvl[:, rank + rd : rank + rd + E.shape[0]]
        q = (cq_n @ wq_b_virtual_for_chip(src, cfg, tp, spec.softmax_scale)).reshape(T, H, cfg.head_dim)
        g = torch.sigmoid(cq_n @ wq_b_gate_for_chip(src, cfg, tp))
        q_nope, q_pe = q[..., :nope], rope(q[..., nope:])
        if mode == "decode":
            q_lat = torch.einsum("thn,hnr->thr", q_nope, w_uk_virtual_for_chip(src, cfg, tp)[0])
            Q = torch.cat([q_lat, q_pe], -1)
            K = torch.cat([n, k_pe], -1)
            w_uv = w_uv_virtual_for_chip(src, cfg, tp)[0]
            U = torch.stack([attend(Q[:, h], K, n) @ w_uv[h] for h in range(H)], 1)  # [T, 10, 128] virtual
            u_flat = U.reshape(T, H * v)
            u_sig, u_noise = u_flat[:, : Sg * v], u_flat[:, Sg * v :] @ X
        else:
            Qv = torch.cat([q_nope, q_pe], -1)
            Qh = torch.cat([Qv[:, a:b] for a, b in hf_order_from_virtual(cfg)], 1)  # HF local order
            kvx = n @ W.prefill_kv_expansion_for_chip(src["wkv_b"], src["kv_norm"], cfg, tp)
            blk = kvx.shape[1] // G
            outs = []
            for h in range(H):
                gi = h // cfg.heads_per_group
                k_g = torch.cat([kvx[:, gi * blk : gi * blk + nope], k_pe], -1)
                v_g = kvx[:, gi * blk + nope : (gi + 1) * blk]
                outs.append(attend(Qh[:, h], k_g, v_g)[:, :v])
            u_flat = torch.cat(outs, -1)  # HF order
            hpg = cfg.heads_per_group
            u_sig = torch.cat([u_flat[:, hpg * gi * v : (hpg * gi + r) * v] for gi in range(G)], -1)
            noise = [u_flat[:, (hpg * gi + r) * v : (hpg * gi + r + 1) * v] for gi in range(G)]
            u_noise = torch.cat([nz for nz in noise for _ in range(r)], -1)
        d = (u_sig - torch.sigmoid(lam @ E) * u_noise) * g
        out += d @ W.wo_for_chip(src["wo"], cfg, tp)
    return out


# ---- long-prefill row-subset goldens (cached on disk, keyed by the reference sources and the exact inputs) ----------
LONG_S = (8192, 16384, 32768)
LONG_GOLDEN_FORMAT = 2  # bump when ref_rows / long_case change meaning
REFERENCE_SOURCES = ("modules.py", "rope.py", "config.py", "cache.py")


def ref_rows(ref, x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Reference GDLA output for query rows ``rows`` of a prefill over ``x [S, D]`` (positions 0..S-1): full-sequence
    latents / K / V, attention only for the selected rows (the reference module's own methods, fp32)."""
    S = x.shape[0]
    xs = x[rows][None]
    q, gate = ref.project_q(xs)
    q_nope, q_pe = torch.split(q, [ref.nope_dim, ref.rope_dim], dim=-1)
    from models.demos.motif3.reference.rope import apply_rope

    cos_q, sin_q = ref_rope(ref, rows[None], x.dtype)
    q_pe = apply_rope(q_pe, cos_q, sin_q)
    cos, sin = ref_rope(ref, torch.arange(S)[None], x.dtype)
    c, k_pe = ref.project_kv(x[None], cos, sin)
    heads = ref._heads_expanded(q_nope, q_pe, c, k_pe, rows[None], torch.arange(S), x.dtype)
    lam = torch.sigmoid(ref.lambda_proj(xs).float()).to(x.dtype).unsqueeze(-1)
    signal, noise = ref._split_signal_noise(heads)
    diff = (signal - lam * noise) * torch.sigmoid(gate)
    return ref.wo(diff.reshape(1, len(rows), -1))[0]


def check_rows(P: int, seed: int) -> torch.Tensor:
    """Query rows checked against a row-subset golden: the first rows, the window edge, random rows, the last rows."""
    g = torch.Generator().manual_seed(seed)
    return torch.cat([torch.arange(0, 8), torch.arange(120, 136), torch.randint(136, P - 16, (24,), generator=g),
                      torch.arange(P - 16, P)]).unique()  # fmt: skip


def ref_prefill_into(ref, x: torch.Tensor, rc, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Reference prefill of ``x [P, D]`` (positions 0..P-1) into the reference cache ``rc``; returns the output
    ``[P, D]``, or only ``rows`` (:func:`ref_rows`) when given -- the full fp32 expanded form materializes
    ``[80, P, P]`` scores (2 GB at P = 2500)."""
    P = x.shape[0]
    if rows is None:
        return ref(x[None], torch.arange(P)[None], rc)[0]
    cos, sin = ref_rope(ref, torch.arange(P)[None], x.dtype)
    c, k_pe = ref.project_kv(x[None], cos, sin)
    rc.update(c, k_pe, torch.arange(P)[None])
    return ref_rows(ref, x, rows)


def long_case(args, layer: int, S: int):
    """Deterministic long-prefill case: random weights / inputs and the query rows to check."""
    tensors = random_attn_tensors(args, seed=70 + layer)
    g = torch.Generator().manual_seed(S + layer)
    x = torch.randn(S, 4096, generator=g).to(torch.bfloat16).float()
    rows = torch.cat([torch.arange(0, 8), torch.arange(120, 136), torch.randint(136, S - 16, (24,), generator=g),
                      torch.arange(S - 16, S)]).unique()  # fmt: skip
    return tensors, x, rows


def reference_fingerprint() -> str:
    """Hash of the reference package sources the goldens depend on (a reference change invalidates cached goldens)."""
    from models.demos.motif3 import reference

    root = Path(reference.__file__).resolve().parent
    h = hashlib.sha1()
    for name in REFERENCE_SOURCES:
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return h.hexdigest()[:16]


def long_golden_key(layer: int, S: int, tensors: Dict[str, torch.Tensor], x: torch.Tensor, rows: torch.Tensor) -> str:
    """Cache key of a long golden: golden format, reference sources, layer kind, and the exact weight / input / row
    bytes (so a changed seed, RNG or reference silently invalidates the file instead of being compared against)."""
    h = hashlib.sha1()
    h.update(f"fmt{LONG_GOLDEN_FORMAT}|ref{reference_fingerprint()}|L{layer}|S{S}".encode())
    for k in ATTN_KEYS:
        h.update(tensors[k].float().contiguous().numpy().tobytes())
    h.update(x.float().contiguous().numpy().tobytes())
    h.update(rows.long().contiguous().numpy().tobytes())
    return h.hexdigest()


def long_golden(args, layer: int, S: int, tensors=None, x=None, rows=None):
    """Row-subset golden of :func:`long_case`, cached as ``tt_cache/test/attention/long_golden_L<l>_S<S>.pt`` (~1 MB)
    together with its :func:`long_golden_key`; a file whose key differs is rebuilt (~10-60 s of CPU)."""
    if tensors is None:
        tensors, x, rows = long_case(args, layer, S)
    key = long_golden_key(layer, S, tensors, x, rows)
    path = TEST_DIR / f"long_golden_L{layer}_S{S}.pt"
    if path.is_file():
        d = torch.load(path, weights_only=True)
        if d.get("key") == key:
            return d["rows"], d["want"], d["n_rows"]
        log(f"long golden {path.name}: stale (key mismatch) -> rebuilding")
    ref = ref_attention(args, layer, tensors, torch.float32)
    want = ref_rows(ref, x, rows)
    n_ref, _ = ref_latents(ref, x, torch.arange(S))
    d = {"key": key, "rows": rows, "want": want, "n_rows": n_ref[rows].clone()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, path)
    return d["rows"], d["want"], d["n_rows"]


# ======================================================================================================================
# CPU tests (exact algebra; no device)
# ======================================================================================================================
@pytest.mark.parametrize("layer", [0, 1], ids=["global_L0", "swa_L1"])
def test_cpu_dataflow_matches_reference_fp64(layer):
    """The module's weight transforms + per-chip dataflow (decode and prefill forms) reproduce the reference GDLA
    in fp64 at real dims (T = 160 > window): head order, scale folding, absorb / un-absorb, kv expansion, lambda
    expansion, signal / noise selection, gate, wo row split."""
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    args = ref_args(q_path_fp32=False)  # keep the q path in the module dtype (fp64)
    t64 = {k: v.double() for k, v in random_attn_tensors(args, seed=100 + layer).items()}
    ref = ref_attention(args, layer, t64, torch.float64)
    T = 160
    g = torch.Generator().manual_seed(7)
    x = torch.randn(T, cfg.hidden_size, generator=g, dtype=torch.float64)
    want = ref(x[None], torch.arange(T)[None])[0]
    for mode in ("decode", "prefill"):
        got = emulate_tt(cfg, t64, layer, x, mode)
        s = stats(want, got)
        log(f"cpu fp64 L{layer} {mode}: {fmt(s)}")
        # rope runs in fp32 inside the reference (apply_rope upcasts to fp32), so ~1e-7 relative is the floor
        assert s["rel_fro"] < 2e-6, f"L{layer} {mode}: {fmt(s)}"


def test_cpu_head_orders():
    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8))
    assert virtual_head_order(cfg) == [0, 1, 2, 3, 5, 6, 7, 8, 4, 9]
    order = virtual_head_order(cfg)
    flat = [i for a, b in hf_order_from_virtual(cfg) for i in range(a, b)]
    assert [order[i] for i in flat] == list(range(10))  # reordering the virtual heads gives the HF local order
    E = lambda_expansion(cfg)
    assert E.shape == (64, 1024) and torch.equal(E.sum(0), torch.ones(1024)) and E[8:].abs().sum() == 0
    X = noise_expansion(cfg)
    assert X.shape == (256, 1024) and torch.equal(X.sum(0), torch.ones(1024))
    u = torch.randn(3, 256)
    want = torch.cat([u[:, :128]] * 4 + [u[:, 128:]] * 4, -1)
    assert torch.equal(u @ X, want)


def test_cpu_mask_variants_and_golden_key():
    """The wrong-mask alternatives of the FlashMLA check and the long-golden cache key (pure python / torch)."""
    v = mask_variants(200, 129)
    assert v == {"exact": (72, 200), "key p+1 visible": (72, 201), "window 128": (73, 200), "window 130": (71, 200),
                 "window dropped": (0, 200)}  # fmt: skip
    assert mask_variants(128, 129) == {"exact": (0, 128), "key p+1 visible": (0, 129), "window 128": (1, 128)}
    assert mask_variants(5, 129) == {"exact": (0, 5), "key p+1 visible": (0, 6)}
    assert mask_variants(300, None) == {
        "exact": (0, 300),
        "key p+1 visible": (0, 301),
        "window 129 applied": (172, 300),
    }
    assert mask_variants(100, None) == {"exact": (0, 100), "key p+1 visible": (0, 101)}
    # the golden key follows every input byte and the reference sources, and is stable otherwise
    args = ref_args()
    t = random_attn_tensors(args, seed=1)
    x = torch.randn(64, 4096, generator=torch.Generator().manual_seed(0)).bfloat16().float()
    rows = torch.arange(0, 64, 7)
    k = long_golden_key(0, 64, t, x, rows)
    assert k == long_golden_key(0, 64, t, x.clone(), rows.clone())
    x2 = x.clone()
    x2[3, 5] += 1.0
    t2 = dict(t, wo=t["wo"] * 2)
    assert len({k, long_golden_key(1, 64, t, x, rows), long_golden_key(0, 64, t, x2, rows),
                long_golden_key(0, 64, t2, x, rows), long_golden_key(0, 64, t, x, rows[:-1])}) == 5  # fmt: skip
    assert len(reference_fingerprint()) == 16


def test_cpu_legacy_sdpa_window_mask_emulation():
    """The emulated legacy-kernel window mask (upstream bug analysis): a superset of the exact causal window, at most
    31 extra (older) keys per row, exact on the first row of every Q tile; e.g. row 287 sees keys from 128 instead of
    159 (Q tile 256..287 judges the K tile 128..159 on row 256's window [128, 256])."""
    S, Wn = 512, 129
    q = torch.arange(S)
    exact = (q[None, :] <= q[:, None]) & (q[None, :] > q[:, None] - Wn)
    legacy = legacy_sdpa_window_mask(S, Wn)
    assert bool((legacy | exact).eq(legacy).all())  # superset
    extra = (legacy & ~exact).sum(-1)
    assert int(extra.max()) == 31 and int(extra[::32].max()) == 0 and int(extra[:Wn].max()) == 0
    assert int(legacy[287].nonzero().min()) == 128 and int(exact[287].nonzero().min()) == 159


def test_cpu_import_rule():
    """``tt/attention.py`` imports no other ``models/demos/**`` package and nothing heavy at import time (README §13;
    checked in a fresh interpreter, like ``test_infra_import.py``)."""
    import json
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[5]
    probe = (
        "import importlib, json, sys; importlib.import_module('models.demos.motif3.tt.attention'); "
        "print(json.dumps({'demos': sorted(m for m in sys.modules if m.startswith('models.demos.') and not "
        "m.startswith('models.demos.motif3')), 'heavy': sorted(m for m in ('vllm', 'transformers', 'huggingface_hub', "
        "'safetensors') if m in sys.modules)}))"
    )
    env = dict(
        os.environ, PYTHONPATH=str(root) + (os.pathsep + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")
    )
    res = subprocess.run(
        [sys.executable, "-c", probe], cwd=str(root), env=env, capture_output=True, text=True, timeout=300
    )
    assert res.returncode == 0, res.stderr[-3000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out == {"demos": [], "heavy": []}, out
    assert "Opening user mode device driver" not in res.stderr + res.stdout


# ======================================================================================================================
# device helpers
# ======================================================================================================================
def _setup(mesh_device, tag: str):
    from models.demos.motif3.tt.ccl import MotifCCL, log_fabric
    from models.demos.motif3.tt.rope import MotifRope

    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device)
    rep = log_fabric(mesh_device, f"attention {tag}")
    assert rep["committed"] is not None
    log(f"{tag}: L1_SMALL {l1_small_bytes(mesh_device)} B per core")
    return cfg, MotifCCL(mesh_device, cfg), MotifRope(mesh_device, cfg)


def _replicated(mesh_device, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _host_quant(t: torch.Tensor, dtype) -> torch.Tensor:
    """Exactly the values ``from_torch(dtype)`` stores (bfp8 block exponents depend on the tile layout of ``t``; rows
    of a tile are quantized independently, so stale rows never change the history rows' values)."""
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def _upload_cache(mesh_device, paged_q: torch.Tensor, dtype):
    """Replicated paged cache from host values that are already representable in ``dtype`` (:func:`_host_quant`)."""
    return _replicated(mesh_device, paged_q, dtype)


def _dev(t, idx: int = 0) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[idx]).float()


def _rows(mesh_device, cfg, t: torch.Tensor, dtype, layout):
    """Lane-ordered host tensor ``[32, ...]`` -> per DP row ``[8, ...]`` (dim 0 sharded over DP, replicated over TP)."""
    from models.demos.motif3.tt.rope import shard_lanes

    return shard_lanes(t, cfg, mesh_device, dtype=dtype, layout=layout, device=mesh_device)


def _lane_x(mesh_device, cfg, x32: torch.Tensor):
    """``x32 [32, 4096]`` (lane order) -> decode input ``[1, 1, 8, 4096]`` per DP row."""
    rows = x32.reshape(cfg.dp, 1, cfg.lanes_per_row, x32.shape[-1])
    return _rows(mesh_device, cfg, rows, ttnn.bfloat16, ttnn.TILE_LAYOUT)


def _per_lane_out(cfg, mesh_device, out) -> torch.Tensor:
    """Decode output -> ``[32, 4096]`` lane order from TP index 0 of each row."""
    from models.demos.motif3.tt.ccl import device_tensors_to_torch

    full = device_tensors_to_torch(out, mesh_device).float()  # [R, C, 1, 1, 8, 4096]
    res = torch.zeros(cfg.max_batch, full.shape[-1])
    for dp in range(cfg.dp):
        r, c = cfg.axes.coord(dp, 0)
        res[cfg.lanes_per_row * dp : cfg.lanes_per_row * (dp + 1)] = full[r, c, 0, 0]
    return res


def _alloc_cache(mesh_device, cfg, num_blocks: int, block: int):
    empty = ttnn.empty(
        [num_blocks, 1, block, cfg.kv_latent_dim],
        cfg.dtypes.kv_cache,
        ttnn.TILE_LAYOUT,
        mesh_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    cache = ttnn.fill(empty, 0.0)
    ttnn.deallocate(empty)
    return cache


def _chip_index(cfg, dp: int, tp: int = 0) -> int:
    """``ttnn.get_device_tensors`` index (row-major mesh coordinate) of the chip with roles (dp, tp)."""
    r, c = cfg.axes.coord(dp, tp)
    return r * cfg.axes.mesh_shape[1] + c


def _free(o):
    if isinstance(o, (list, tuple)):
        for t in o:
            _free(t)
    elif isinstance(o, ttnn.Tensor):
        ttnn.deallocate(o)


def _l1_report(mesh_device) -> str:
    """Non-DRAM buffers of the mesh's first device (diagnostics for the L1_SMALL requirement; private ttnn reports
    API). Device ids are per process, not per mesh: a mesh opened after another one has no device 0."""
    try:
        bufs = list(ttnn._ttnn.reports.get_buffers(mesh_device))
        first = min((b.device_id for b in bufs), default=None)
        bufs = [b for b in bufs if b.device_id == first]
    except Exception as e:  # pragma: no cover - API drift
        return f"(buffer report unavailable: {type(e).__name__})"
    l1 = sorted(int(b.address) for b in bufs if str(b.buffer_type).endswith(".L1"))
    small = [b for b in bufs if "L1_SMALL" in str(b.buffer_type)]
    return f"main-L1 buffers {len(l1)} (lowest at {l1[0] if l1 else None} B), L1_SMALL buffers {len(small)}"


class _Capture:
    """Exception-safe trace capture (a dangling capture hung close_mesh_device once; GATES_RESULTS §11.6)."""

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


def _trace_min_us(mesh_device, fn, n: int, reps: int) -> float:
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


def _traced_us(mesh_device, fn, n: int = 16, reps: int = 7) -> Dict[str, float]:
    """Slope method of the gates (GATES_RESULTS §2): (min t(n) - min t(n/2)) / (n/2); raw = min t(n) / n."""
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    t1 = _trace_min_us(mesh_device, fn, n // 2, reps)
    t2 = _trace_min_us(mesh_device, fn, n, reps)
    return {"slope_us": max((t2 - t1) / (n - n // 2), 0.0), "raw_us": t2 / n, "t_half": t1, "t_full": t2}


def _eager_us(mesh_device, fn, iters: int = 10, warmup: int = 2) -> float:
    for _ in range(warmup):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


def _tensors_for(weights_kind: str, layer: int, args):
    if weights_kind == "real":
        return {k: v.float() for k, v in real_attn_tensors(layer).items()}
    return random_attn_tensors(args, seed=1000 + layer)


def _inputs_for(weights_kind: str, layer: int, n: int, seed: int) -> torch.Tensor:
    if weights_kind == "real":
        return load_real_inputs(layer, n)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 4096, generator=g).to(torch.bfloat16).float()


def _decode_step_inputs(mesh_device, cfg, rope, pos_t: Sequence[int], x32: torch.Tensor, pt: torch.Tensor):
    """Per-step device inputs of :meth:`MotifAttention.forward_decode` from lane-ordered host values."""
    x_tt = _lane_x(mesh_device, cfg, x32)
    cur_tt = _rows(mesh_device, cfg, torch.tensor(list(pos_t), dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    pt_tt = _rows(mesh_device, cfg, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    rot_idx = rope.rot_idxs_device(torch.tensor(list(pos_t)))
    rot = MotifAttention.decode_rope_tables(rope, rot_idx)
    act = MotifAttention.active_mask_from_cur_pos(cur_tt, cfg.lanes_per_row)
    return {"x": x_tt, "cur": cur_tt, "pt": pt_tt, "rot_idx": rot_idx, "rot": rot, "act": act}


def _free_step(d):
    _free([d[k] for k in ("x", "cur", "pt", "rot_idx", "act")] + [t for cs in d["rot"].values() for t in cs])


def mla_goldens(cfg, q_tt, caches_host: Dict[int, torch.Tensor], pt, pos_t, window, swa_window: int = 129):
    """fp64 FlashMLA goldens on the TT's own Q (``q_tt`` = ``taps["q_mla"]``) and cache contents after this step's
    update (``caches_host[dp]`` = readback of chip (dp, tp=0)) for every active lane: ``{lane: (pos, {mask name:
    [H, 512]})}`` for the exact mask and the plausible wrong ones (:func:`mask_variants`)."""
    H, rank, block = cfg.q_heads_per_chip, cfg.kv_lora_rank, cfg.kv_block_size
    out = {}
    for dp in range(cfg.dp):
        q = _dev(q_tt, _chip_index(cfg, dp))[0, :, :H].double()
        for j in range(cfg.lanes_per_row):
            lane = cfg.lanes_per_row * dp + j
            p = pos_t[lane]
            if p < 0:
                continue
            variants = mask_variants(p, window, swa_window)
            if (p + 1) // block >= pt.shape[1]:
                variants.pop("key p+1 visible")
            hi = max(b for _, b in variants.values())
            k_all = _gather_cache(caches_host[dp], pt[lane], torch.arange(hi + 1), block).double()
            out[lane] = (
                p,
                {
                    name: torch.softmax(q[j] @ k_all[a : b + 1].T, dim=-1) @ k_all[a : b + 1, :rank]
                    for name, (a, b) in variants.items()
                },
            )
    return out


def mla_score(cfg, o_tt, goldens):
    """Score a FlashMLA output ``[1, L, H, 512]`` per DP row against :func:`mla_goldens`. Returns ``(aggregate stats
    vs the exact mask, {lane: pcc vs exact}, {lane: best-matching mask among the exact one and the distinguishable
    wrong ones}, number of distinguishable (lane, wrong mask) pairs)``. A wrong mask is distinguishable when its golden
    differs from the exact one by ``1 - pcc > DISC_MIN`` (far above the kernel's bf16-statistics noise, ~1e-4)."""
    H = cfg.q_heads_per_chip
    o_rows = {dp: _dev(o_tt, _chip_index(cfg, dp))[0, :, :H].double() for dp in range(cfg.dp)}
    want, got, lane_pcc, best, n_disc = [], [], {}, {}, 0
    for lane, (p, gold) in goldens.items():
        o = o_rows[lane // cfg.lanes_per_row][lane % cfg.lanes_per_row]
        ex = gold["exact"]
        want.append(ex)
        got.append(o)
        scores = {"exact": pcc(ex, o)}
        for name, g_alt in gold.items():
            if name != "exact" and 1.0 - pcc(ex, g_alt) > DISC_MIN:
                scores[name] = pcc(g_alt, o)
                n_disc += 1
        lane_pcc[lane] = scores["exact"]
        best[lane] = max(scores, key=scores.get)
    return stats(torch.stack(want), torch.stack(got)), lane_pcc, best, n_disc


def mla_distinguishable_lanes(goldens, name: str) -> List[int]:
    """Lanes whose ``name`` wrong-mask golden is distinguishable from the exact one (see :func:`mla_score`)."""
    return [l for l, (_, g) in goldens.items() if name in g and 1.0 - pcc(g["exact"], g[name]) > DISC_MIN]


# ======================================================================================================================
# device: prefill (S = 128 / 1024) + cache fill
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("weights_kind", ["random", "real"])
@pytest.mark.parametrize("layer", [0, 1], ids=["global_L0", "swa_L1"])
def test_attention_prefill(mesh_device, device_params, layer, weights_kind):
    """Prefill S = 128 / 1024 with the default (``sdpa_prefill`` role) and the fp32-acc opt-in, the composite-RoPE
    fallback (random weights), cache fill into a stale-filled pool: filled rows vs the reference latents, every block
    outside the user's page table bitwise untouched."""
    from models.demos.motif3.tt.ccl import replicas_identical

    cfg, ccl, rope = _setup(mesh_device, f"prefill L{layer} {weights_kind}")
    args = ref_args()
    tensors = _tensors_for(weights_kind, layer, args)
    attn = MotifAttention(mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=False)
    attn_fp32 = MotifAttention(  # opt-in A/B: fp32 dest acc where no window is in effect (legacy SDPA kernel)
        mesh_device,
        cfg,
        layer,
        source=hf_source(tensors, layer),
        ccl=ccl,
        rope=rope,
        cache=False,
        sdpa_prefill_fp32_acc="auto",
    )
    attn_c = (  # G8 fallback: composite RoPE (random weights only)
        MotifAttention(mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=False,
                       rope_mode="composite")  # fmt: skip
        if weights_kind == "random"
        else None
    )
    ref32 = ref_attention(args, layer, tensors, torch.float32)
    ref16 = ref_attention(args, layer, tensors, torch.bfloat16)
    block = cfg.kv_block_size
    bank = stale_bank(ref32, weights_kind, layer, n=512, seed=layer)
    failures = []
    if attn_c is not None:  # composite RoPE in the decode layout ([1, 10, 8, 64] vs 32-row "rows" tables)
        g0 = torch.Generator().manual_seed(4)
        xr = torch.randn(1, 10, 8, 64, generator=g0).to(torch.bfloat16).float()
        pos = torch.tensor([0, 1, 127, 128, 129, 1000, 4095, 32767] * cfg.dp)
        cos, sin = rope.decode_cos_sin(attn.kind, rope.rot_idxs_device(pos), layout="rows")
        xr_tt = _replicated(mesh_device, xr, ttnn.bfloat16)
        a_hf, a_c = _dev(attn._rope(xr_tt, cos, sin)), _dev(attn_c._rope(xr_tt, cos, sin))
        tc, ts = rope.host_tables[attn.kind]
        from models.demos.motif3.tt.rope import apply_rope_torch

        want_r = apply_rope_torch(xr, tc[pos[:8]][None, None], ts[pos[:8]][None, None]).float()
        p_hf, p_c = pcc(want_r, a_hf), pcc(want_r, a_c)
        log(f"rope decode layout L{layer}: fused hf pcc {p_hf:.6f}, composite pcc {p_c:.6f}")
        if min(p_hf, p_c) < 0.9999:
            failures.append(f"rope decode layout: hf {p_hf} composite {p_c}")
        _free([xr_tt, cos, sin])
    for S in (128, 1024):
        x = _inputs_for(weights_kind, layer, S, seed=S + layer)
        pos = torch.arange(S)
        want = ref32(x[None], pos[None])[0]
        hf16 = ref16(x[None].to(torch.bfloat16), pos[None])[0].float()
        nb = S // block
        g = torch.Generator().manual_seed(S)
        pool = 1 + nb + 3
        pt = (torch.randperm(pool - 1, generator=g)[:nb] + 1).to(torch.int32)[None]  # block 0 = null block
        stale = _host_quant(stale_paged(bank, pool, block, seed=S), cfg.dtypes.kv_cache)
        cache = _upload_cache(mesh_device, stale, cfg.dtypes.kv_cache)
        x_tt = _replicated(mesh_device, x[None, None], ttnn.bfloat16)
        pt_tt = _replicated(mesh_device, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        out_fp32 = attn_fp32.forward_prefill(x_tt)  # opt-in variant; no cache fill
        s_fp32 = stats(want, _dev(out_fp32)[0, 0])
        ttnn.deallocate(out_fp32)
        s_c = None
        if attn_c is not None:
            out_c = attn_c.forward_prefill(x_tt)
            s_c = stats(want, _dev(out_c)[0, 0])
            ttnn.deallocate(out_c)
            if s_c["pcc"] < PCC_MIN:
                failures.append(f"S={S} composite rope: {fmt(s_c)}")
        win, ckc = attn.prefill_sdpa_window_and_config(S)
        win32, ckc32 = attn_fp32.prefill_sdpa_window_and_config(S)
        out = attn.forward_prefill(x_tt, page_table=pt_tt, kv_cache=cache)
        got = _dev(out)[0, 0]
        s = stats(want, got)
        s16 = stats(want, hf16)
        same = replicas_identical(out, mesh_device, "tp", cfg.axes) and replicas_identical(
            out, mesh_device, "dp", cfg.axes
        )
        # cache contents (M18): TT holds the gamma-free unit-RMS latent; compare n * gamma_kv with the reference c_kv.
        # Blocks outside the page table (null block + 3 spare) must keep their stale content bitwise.
        n_ref, kpe_ref = ref_latents(ref32, x, pos)
        gam = tensors["kv_norm"].float()
        outside = sorted(set(range(pool)) - set(pt[0].tolist()))
        cs, untouched = [], True
        for chip in (0, 31):
            cache_h = _dev(cache, chip)
            rows = _gather_cache(cache_h, pt[0], torch.arange(S), block)
            cs.append(
                (stats(n_ref * gam, rows[:, : cfg.kv_lora_rank] * gam), stats(kpe_ref, rows[:, cfg.kv_lora_rank :]))
            )
            untouched &= torch.equal(cache_h[outside], stale[outside])
        eager = _eager_us(mesh_device, lambda: attn.forward_prefill(x_tt, page_table=pt_tt, kv_cache=cache), iters=3)
        eager32 = _eager_us(mesh_device, lambda: attn_fp32.forward_prefill(x_tt), iters=3)
        log(
            f"prefill L{layer} {cfg.layer(layer).attn_kind} {weights_kind} S={S}: TT vs ref fp32 {fmt(s)}; "
            f"(default: sdpa window {win} fp32_acc {ckc.fp32_dest_acc_en}) "
            + (f"[composite-RoPE variant: pcc {s_c['pcc']:.6f}] " if s_c is not None else "")
            + f"[fp32-acc opt-in (window {win32} fp32_acc {ckc32.fp32_dest_acc_en}): pcc {s_fp32['pcc']:.6f} "
            f"max_abs {s_fp32['max_abs']:.3e}, eager {eager32:.0f} us] "
            f"[HF-bf16 vs fp32: pcc {s16['pcc']:.6f} max_abs {s16['max_abs']:.3e}]; replicas(32) identical {same}; "
            f"cache c_kv pcc {cs[0][0]['pcc']:.6f} max_abs {cs[0][0]['max_abs']:.3e}, k_pe pcc {cs[0][1]['pcc']:.6f} "
            f"(chip31 c_kv pcc {cs[1][0]['pcc']:.6f}); blocks outside the page table untouched {untouched}; "
            f"eager {eager:.0f} us (incl. cache fill)"
        )
        ok = (
            s["pcc"] >= PCC_MIN
            and s["nonfinite"] == 0
            and s_fp32["pcc"] >= PCC_MIN
            and same
            and untouched
            and min(c[0]["pcc"] for c in cs) >= PCC_MIN
            and min(c[1]["pcc"] for c in cs) >= PCC_MIN
        )
        if not ok:
            failures.append(
                f"S={S}: {fmt(s)} fp32-opt-in {s_fp32['pcc']:.5f} same={same} untouched={untouched} "
                f"cache={[(c[0]['pcc'], c[1]['pcc']) for c in cs]}"
            )
        _free([out, x_tt, pt_tt, cache])
    assert not failures, "\n".join(failures)


# ======================================================================================================================
# device: decode, 32 lanes with heterogeneous positions (window edges, inactive lanes), host-filled history + stale rest
# ======================================================================================================================
DECODE_POS = [
    0, 1, 5, 64, 127, 128, 129, 130,  # row 0: around the 129-key window edge
    131, 200, 255, 256, 257, 383, 511, 640,  # row 1
    700, 1000, 1023, 1024, 1500, 1800, 1999, 63,  # row 2
    -1, 9, 300, 129, 1201, 1995, -1, 1777,  # row 3: two inactive lanes
]  # fmt: skip
DECODE_STEPS = 2


def _lane_sequences(weights_kind: str, layer: int, positions: Sequence[int], steps: int) -> List[torch.Tensor]:
    """Per-lane input sequences ``[p + steps, 4096]`` (history ``[:p]``, decode tokens ``[p:]``), distinct per lane."""
    seqs = []
    if weights_kind == "real":
        real = load_real_inputs(layer, 2080)
        for lane, p in enumerate(positions):
            n = max(p, 0) + steps
            off = (97 * lane) % (real.shape[0] - n + 1)
            seqs.append(real[off : off + n])
        return seqs
    for lane, p in enumerate(positions):
        g = torch.Generator().manual_seed(10_000 * layer + lane)
        seqs.append(torch.randn(max(p, 0) + steps, 4096, generator=g).to(torch.bfloat16).float())
    return seqs


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("weights_kind", ["random", "real"])
@pytest.mark.parametrize("layer", [0, 1], ids=["global_L0", "swa_L1"])
def test_attention_decode(mesh_device, device_params, layer, weights_kind):
    from models.demos.motif3.reference.cache import LatentKVCache
    from models.demos.motif3.reference.golden import TensorRecorder
    from models.demos.motif3.tt.ccl import replicas_identical

    cfg, ccl, rope = _setup(mesh_device, f"decode L{layer} {weights_kind}")
    args = ref_args()
    tensors = _tensors_for(weights_kind, layer, args)
    attn = MotifAttention(mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=False)
    ref = ref_attention(args, layer, tensors, torch.float32)
    gam = tensors["kv_norm"].float()
    B, block, rank = cfg.max_batch, cfg.kv_block_size, cfg.kv_lora_rank
    positions = list(DECODE_POS)
    active = [p >= 0 for p in positions]
    seqs = _lane_sequences(weights_kind, layer, positions, DECODE_STEPS)
    window = cfg.layer(layer).sliding_window_size

    # ---- page tables: W blocks per lane from a shuffled pool (block 0 = null); inactive lanes -> all null ----------
    Wd = math.ceil((max(positions) + DECODE_STEPS) / block)
    pool = 1 + B * Wd
    g = torch.Generator().manual_seed(5)
    perm = (torch.randperm(pool - 1, generator=g) + 1).to(torch.int32)
    pt = torch.zeros(B, Wd, dtype=torch.int32)
    for lane in range(B):
        if active[lane]:
            pt[lane] = perm[lane * Wd : (lane + 1) * Wd]

    # ---- history: TT paged cache (exact host quantization; every non-history slot stale) + two reference caches ----
    paged = stale_paged(stale_bank(ref, weights_kind, layer, seed=layer), pool, block, seed=layer)
    lat = {}
    for lane, p in enumerate(positions):
        if p <= 0:
            continue
        n, kpe = ref_latents(ref, seqs[lane][:p], torch.arange(p))
        lat[lane] = (n, kpe)
        idx = torch.arange(p)
        paged[pt[lane, idx // block].long(), 0, idx % block] = torch.cat([n, kpe], -1)
    paged_q = _host_quant(paged, cfg.dtypes.kv_cache)
    cache = _upload_cache(mesh_device, paged_q, cfg.dtypes.kv_cache)
    max_len = max(positions) + DECODE_STEPS
    ref_cache = {k: LatentKVCache(B, max_len, rank, cfg.rope_dim, torch.float32) for k in ("exact", "quant")}
    for lane, (n, kpe) in lat.items():
        p = positions[lane]
        rows_q = _gather_cache(paged_q, pt[lane], torch.arange(p), block)
        for k, (nn_, kk) in {"exact": (n, kpe), "quant": (rows_q[:, :rank], rows_q[:, rank:])}.items():
            ref_cache[k].c[lane, :p] = nn_ * gam
            ref_cache[k].k_pe[lane, :p] = kk
            ref_cache[k].seq_lens[lane] = p

    failures = []
    for step in range(DECODE_STEPS):
        pos_t = [p + step if a else -1 for p, a in zip(positions, active)]
        x32 = torch.stack([seqs[l][pos_t[l]] if active[l] else torch.zeros(4096) for l in range(B)])
        # reference (inactive lanes run at their own empty slot 0 and are ignored)
        rpos = torch.tensor([[max(p, 0)] for p in pos_t])
        want = {}
        ref_taps = None
        for k in ("exact", "quant"):
            rec = TensorRecorder() if k == "exact" else None
            want[k] = ref(x32[:, None], rpos, ref_cache[k], tap=rec)[:, 0]
            ref_taps = rec.tensors if rec is not None else ref_taps
        d = _decode_step_inputs(mesh_device, cfg, rope, pos_t, x32, pt)
        if step == 0:  # the mask is mandatory (FlashMLA leaves skipped lanes' rows unwritten); raises before any op
            with pytest.raises(ValueError, match="active"):
                attn.forward_decode(d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache,
                                    active=None)  # fmt: skip
        act_host = MotifAttention.active_mask_host(torch.tensor(pos_t), cfg, mesh_device, device=mesh_device)
        same_mask = torch.equal(_dev(d["act"]), _dev(act_host)) and torch.equal(_dev(d["act"], 31), _dev(act_host, 31))
        taps = {}
        out = attn.forward_decode(
            d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache, active=d["act"], taps=taps
        )
        got = _per_lane_out(cfg, mesh_device, out)
        same = replicas_identical(out, mesh_device, "tp", cfg.axes)
        idx = [l for l in range(B) if active[l]]
        caches_h = {}
        for dp in range(cfg.dp):
            di = _chip_index(cfg, dp)
            assert _dev(d["cur"], di).long().tolist() == pos_t[cfg.lanes_per_row * dp : cfg.lanes_per_row * (dp + 1)]
            caches_h[dp] = _dev(cache, di)
        # FlashMLA alone: TT's own Q and cache contents, fp64 goldens (scale folded -> 1.0), exact vs wrong masks
        gold = mla_goldens(cfg, taps["q_mla"], caches_h, pt, pos_t, window)
        kern, kern_lane, best, n_disc = mla_score(cfg, taps["o_lat"], gold)
        disc_fail = [f"lane {l} pos {pos_t[l]} best '{b}'" for l, b in best.items() if b != "exact"]
        controls = []
        if step == 0 and weights_kind == "random":
            # negative controls on the same Q / cache: the kernel run with a deliberately wrong mask must be flagged
            # (best match != exact) on every lane where that mask is distinguishable -- the check has teeth on device
            cur_p1 = _rows(mesh_device, cfg, torch.tensor([p + 1 if p >= 0 else -1 for p in pos_t], dtype=torch.int32),
                           ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
            wrong = (
                {f"window {window - 1}": (window - 1, d["cur"]), f"window {window + 1}": (window + 1, d["cur"])}
                if window is not None
                else {"window 129 applied": (129, d["cur"]), "key p+1 visible": (None, cur_p1)}
            )
            for name, (w_bad, cur_bad) in wrong.items():
                o_bad = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                    taps["q_mla"],
                    cache,
                    None,
                    head_dim_v=rank,
                    page_table_tensor=d["pt"],
                    cur_pos_tensor=cur_bad,
                    scale=1.0,
                    sliding_window_size=w_bad,
                    program_config=attn.decode_pc,
                    compute_kernel_config=attn.ckc_sdpa_decode,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                _, _, best_bad, _ = mla_score(cfg, o_bad, gold)
                ttnn.deallocate(o_bad)
                lanes_c = mla_distinguishable_lanes(gold, name)
                flagged = [l for l in lanes_c if best_bad[l] != "exact"]
                matched = [l for l in lanes_c if best_bad[l] == name]
                controls.append((name, len(lanes_c), len(flagged), len(matched)))
            ttnn.deallocate(cur_p1)
        _free([taps.pop(k) for k in list(taps)])
        ctl_ok = all(n > 0 and f == n for _, n, f, _ in controls)
        ctl_txt = "; ".join(f"control '{nm}': flagged on {f}/{n} distinguishable lanes ({m} best-match it)"
                            for nm, n, f, m in controls)  # fmt: skip
        s_e = stats(want["exact"][idx], got[idx])
        s_q = stats(want["quant"][idx], got[idx])
        lane_pcc = {l: pcc(want["exact"][l], got[l]) for l in idx}
        worst = min(lane_pcc, key=lane_pcc.get)
        kworst = min(kern_lane, key=kern_lane.get)
        zero_inactive = all(float(got[l].abs().max()) == 0.0 for l in range(B) if not active[l])
        # the cache rows written by this step (M18): n * gamma vs reference c_kv, k_pe vs reference k_pe. Lane l's
        # row is written only on the chips of its DP row l // 8 (device index of (dp, tp=0))
        rows = []
        for dp in range(cfg.dp):
            lanes_dp = [l for l in idx if l // cfg.lanes_per_row == dp]
            rows += [caches_h[dp][int(pt[l, pos_t[l] // block]), 0, pos_t[l] % block] for l in lanes_dp]
        rows = torch.stack(rows)
        c_s = stats(ref_taps["c_kv"][idx, 0], rows[:, :rank] * gam)
        k_s = stats(ref_taps["k_pe"][idx, 0], rows[:, rank:])
        log(
            f"decode L{layer} {cfg.layer(layer).attn_kind} {weights_kind} step {step}: TT vs ref fp32 (exact history) "
            f"{fmt(s_e)}; vs ref on the {cfg.dtypes.kv_cache_name} history pcc {s_q['pcc']:.6f} max_abs "
            f"{s_q['max_abs']:.3e}; worst lane {worst} (pos {pos_t[worst]}) pcc {lane_pcc[worst]:.6f}; replicas(tp) "
            f"identical {same}; inactive rows zero {zero_inactive}; device/host active mask equal {same_mask}; written "
            f"cache c_kv pcc {c_s['pcc']:.6f} k_pe pcc {k_s['pcc']:.6f}; FlashMLA alone vs fp64 on its own Q/cache: "
            f"{fmt(kern)}, worst lane {kworst} (pos {pos_t[kworst]}) pcc {kern_lane[kworst]:.6f}; exact mask is the "
            f"best match against {n_disc} distinguishable wrong masks: {not disc_fail}"
            + (f"; {ctl_txt}" if ctl_txt else "")
        )
        ok = (
            s_e["pcc"] >= PCC_MIN
            and s_e["nonfinite"] == 0
            and min(lane_pcc.values()) >= LANE_PCC_MIN
            and same
            and zero_inactive
            and same_mask
            and c_s["pcc"] >= PCC_MIN
            and k_s["pcc"] >= PCC_MIN
            and min(kern_lane.values()) >= ISO_LANE_PCC_MIN
            and not disc_fail
            and (n_disc > 0 if weights_kind == "random" else True)
            and ctl_ok
        )
        if not ok:
            failures.append(
                f"step {step}: {fmt(s_e)} worst lane {worst} {lane_pcc[worst]:.5f} same={same} zero={zero_inactive} "
                f"mask={same_mask} cache c {c_s['pcc']:.5f} k {k_s['pcc']:.5f} kernel worst lane {kworst} "
                f"{kern_lane[kworst]:.6f} n_disc {n_disc} {disc_fail[:4]} controls {controls}"
            )
        _free([out, act_host])
        _free_step(d)
    assert not failures, "\n".join(failures)


# ======================================================================================================================
# device: prefill of 4 users into the paged cache, then teacher-forced decode steps (below / above the window)
# ======================================================================================================================
USERS = [(3, 100), (12, 128), (21, 700), (30, 1000)]  # (lane, prompt length): one user per DP row


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("weights_kind", ["real"])
@pytest.mark.parametrize("layer", [0, 1], ids=["global_L0", "swa_L1"])
def test_attention_prefill_then_decode(mesh_device, device_params, layer, weights_kind):
    """Prompts padded to their bucket with *non-zero* inputs (the fill writes plausible latents beyond the prompt, as a
    pad token does in serving) into a stale-filled pool, then 3 decode steps: the padding / stale slots after each
    position must stay masked."""
    from models.demos.motif3.reference.cache import LatentKVCache
    from models.demos.motif3.tt.ccl import replicas_identical

    cfg, ccl, rope = _setup(mesh_device, f"prefill->decode L{layer} {weights_kind}")
    args = ref_args()
    tensors = _tensors_for(weights_kind, layer, args)
    attn = MotifAttention(mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=False)
    ref = ref_attention(args, layer, tensors, torch.float32)
    B, block = cfg.max_batch, cfg.kv_block_size
    steps = 3
    real = _inputs_for(weights_kind, layer, 2080, seed=0)
    offs = [0, 300, 900, 1050]  # distinct prompts per user (windows of the real sequence)
    Wd = math.ceil((max(p for _, p in USERS) + steps) / block)
    pool = 1 + len(USERS) * Wd
    g = torch.Generator().manual_seed(9)
    perm = (torch.randperm(pool - 1, generator=g) + 1).to(torch.int32)
    pt = torch.zeros(B, Wd, dtype=torch.int32)
    stale = _host_quant(stale_paged(stale_bank(ref, weights_kind, layer, seed=7), pool, block, seed=7),
                        cfg.dtypes.kv_cache)  # fmt: skip
    cache = _upload_cache(mesh_device, stale, cfg.dtypes.kv_cache)
    ref_cache = LatentKVCache(
        len(USERS), max(p for _, p in USERS) + steps, cfg.kv_lora_rank, cfg.rope_dim, torch.float32
    )
    failures = []
    for u, ((lane, P), off) in enumerate(zip(USERS, offs)):
        pt[lane] = perm[u * Wd : (u + 1) * Wd]
        S = cfg.prefill_bucket(P)
        # padding rows: real tokens in shuffled order (non-zero, plausible, not the continuation)
        x = real[torch.randperm(real.shape[0], generator=torch.Generator().manual_seed(50 + u))[:S]].clone()
        x[:P] = real[off : off + P]
        want = ref(x[None, :P], torch.arange(P)[None], ref_cache.user_view(u))[0]
        x_tt = _replicated(mesh_device, x[None, None], ttnn.bfloat16)
        pt_u = _replicated(mesh_device, pt[lane : lane + 1, : cfg.prefill_page_table_entries(S)], ttnn.int32,
                           ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        out = attn.forward_prefill(x_tt, page_table=pt_u, kv_cache=cache)
        s = stats(want, _dev(out)[0, 0, :P])
        log(f"prefill->decode L{layer} user lane {lane} P={P} (bucket {S}, non-zero padding) prefill: {fmt(s)}")
        if s["pcc"] < PCC_MIN:
            failures.append(f"prefill lane {lane}: {fmt(s)}")
        _free([out, x_tt, pt_u])
    for step in range(steps):
        pos_t = [-1] * B
        x32 = torch.zeros(B, 4096)
        for (lane, P), off in zip(USERS, offs):
            pos_t[lane] = P + step
            x32[lane] = real[off + P + step]
        want = ref(
            torch.stack([x32[l] for l, _ in USERS])[:, None],
            torch.tensor([[P + step] for _, P in USERS]),
            ref_cache,
        )[:, 0]
        d = _decode_step_inputs(mesh_device, cfg, rope, pos_t, x32, pt)
        out = attn.forward_decode(
            d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache, active=d["act"]
        )
        got = _per_lane_out(cfg, mesh_device, out)
        same = replicas_identical(out, mesh_device, "tp", cfg.axes)
        lanes = [l for l, _ in USERS]
        per = [stats(want[i], got[l]) for i, l in enumerate(lanes)]
        s = stats(want, got[lanes])
        zero = all(float(got[l].abs().max()) == 0.0 for l in range(B) if pos_t[l] < 0)
        log(
            f"prefill->decode L{layer} step {step} positions {[pos_t[l] for l in lanes]}: {fmt(s)}; per-user pcc "
            f"{[round(p['pcc'], 6) for p in per]}; replicas(tp) {same}; inactive zero {zero}"
        )
        if s["pcc"] < PCC_MIN or min(p["pcc"] for p in per) < LANE_PCC_MIN or not same or not zero:
            failures.append(f"decode step {step}: {fmt(s)} per {[p['pcc'] for p in per]} same={same} zero={zero}")
        _free(out)
        _free_step(d)
    assert not failures, "\n".join(failures)


# ======================================================================================================================
# device: one session in serving order (prefill, decode, prefill after decode, decode), bfp8 / bf16 KV, block 64 / 32
# ======================================================================================================================
SESSION_VARIANTS = [
    pytest.param("bfp8", 64, id="bfp8-block64"),
    pytest.param("bf16", 64, id="bf16-block64"),
    pytest.param("bfp8", 32, id="bfp8-block32"),
]
SESSION_EARLY = [(3, 100), (12, 1000)]  # (lane, prompt length) prefilled before the first decode step
SESSION_LATE = [(21, 2500), (30, 700)]  # prefilled after decode steps ran (buckets 4096 / 1024)
SESSION_STEPS = (2, 2)  # decode steps after the early / the late prefills


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("kv_name, block", SESSION_VARIANTS)
def test_attention_serving_session(mesh_device, device_params, kv_name, block):
    """One mesh session in serving order, both layer kinds (L0 global, L1 SWA), random weights, stale-filled pool:
    prefill A (bucket 128) and B (1024) -> 2 decode steps (A, B) -> prefill C (4096: a global prefill at S >= 1024
    after a decode step, the case the CCL semaphores broke without L1_SMALL) and D (1024), and B's prompt again
    (output bitwise equal to its first prefill: no dependence on the session history) -> 2 decode steps (A-D).
    Every prefill and decode step vs the fp32 reference (per-user >= 0.999); written latents vs the reference cache;
    spare blocks bitwise untouched. Variants: bfp8 / bf16 KV (the MOTIF3_KV_CACHE_DTYPE lever), block 64 / 32 (the
    sizes the bridge accepts). C's page table has 40 (64) / 79 (32) real blocks and the null block beyond, as the
    bridge pads it, so the bucket's padding rows land in the null block."""
    import dataclasses

    from models.demos.motif3.reference.cache import LatentKVCache
    from models.demos.motif3.tt.ccl import replicas_identical
    from models.demos.motif3.tt.model_config import kv_cache_dtype_from_name

    cfg0, ccl, rope = _setup(mesh_device, f"serving session kv {kv_name} block {block}")
    kv_dtype = kv_cache_dtype_from_name(kv_name)
    cfg = dataclasses.replace(cfg0, kv_block_size=block, dtypes=dataclasses.replace(cfg0.dtypes, kv_cache=kv_dtype))
    args = ref_args()
    B, rank = cfg.max_batch, cfg.kv_lora_rank
    users = SESSION_EARLY + SESSION_LATE
    total = sum(SESSION_STEPS)
    n_blk = {lane: math.ceil((P + total) / block) for lane, P in users}
    Wd = max(max(n_blk.values()), max(cfg.prefill_page_table_entries(cfg.prefill_bucket(P)) for _, P in users))
    n_spare = 2
    pool = 1 + sum(n_blk.values()) + n_spare
    perm = (torch.randperm(pool - 1, generator=torch.Generator().manual_seed(17)) + 1).to(torch.int32)
    pt = torch.zeros(B, Wd, dtype=torch.int32)
    off = 0
    for lane, _ in users:
        pt[lane, : n_blk[lane]] = perm[off : off + n_blk[lane]]
        off += n_blk[lane]
    spare = sorted(perm[off:].tolist())
    assert len(spare) == n_spare
    seqs, pads = {}, {}
    for lane, P in users:
        S = cfg.prefill_bucket(P)
        seqs[lane] = torch.randn(P + total, 4096, generator=torch.Generator().manual_seed(lane)).bfloat16().float()
        pads[lane] = torch.randn(S, 4096, generator=torch.Generator().manual_seed(100 + lane)).bfloat16().float()
    L = {}
    for layer in (0, 1):
        tensors = random_attn_tensors(args, seed=600 + layer)
        attn = MotifAttention(
            mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=False
        )
        ref = ref_attention(args, layer, tensors, torch.float32)
        stale = _host_quant(stale_paged(stale_bank(ref, "random", layer, n=512, seed=20 + layer), pool, block,
                                        seed=20 + layer), kv_dtype)  # fmt: skip
        L[layer] = {
            "attn": attn,
            "ref": ref,
            "gam": tensors["kv_norm"].float(),
            "stale": stale,
            "cache": _upload_cache(mesh_device, stale, kv_dtype),
            "rc": {lane: LatentKVCache(1, P + total, rank, cfg.rope_dim, torch.float32) for lane, P in users},
        }
    failures, first_out, done = [], {}, {lane: 0 for lane, _ in users}

    def prefill(lane, P, fill=True):
        S = cfg.prefill_bucket(P)
        x = pads[lane].clone()
        x[:P] = seqs[lane][:P]
        x_tt = _replicated(mesh_device, x[None, None], ttnn.bfloat16)
        n_pt = cfg.prefill_page_table_entries(S)
        pt_u = _replicated(mesh_device, pt[lane : lane + 1, :n_pt], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        for layer, M in L.items():
            kw = {"page_table": pt_u, "kv_cache": M["cache"]} if fill else {}
            t0 = time.perf_counter()
            out = M["attn"].forward_prefill(x_tt, **kw)
            ttnn.synchronize_device(mesh_device)
            ms = (time.perf_counter() - t0) * 1e3
            got = _dev(out)[0, 0]
            same = replicas_identical(out, mesh_device, "tp", cfg.axes) and replicas_identical(
                out, mesh_device, "dp", cfg.axes
            )
            _free(out)
            tag = f"session {kv_name}/{block} L{layer} {cfg.layer(layer).attn_kind} prefill lane {lane} P={P} (S={S})"
            if not fill:  # repeat of an earlier prompt: the output must not depend on what ran in between
                bitwise = torch.equal(got, first_out[(layer, lane)])
                log(f"{tag} again after decode steps: bitwise equal to its first prefill {bitwise}; {ms:.1f} ms")
                if not bitwise:
                    failures.append(f"{tag}: repeat not bitwise equal")
                continue
            first_out[(layer, lane)] = got
            rows_chk = check_rows(P, seed=lane) if P > 2048 else None  # row-subset golden for the long prompt
            want = ref_prefill_into(M["ref"], seqs[lane][:P], M["rc"][lane], rows_chk)
            s = stats(want, got[rows_chk] if rows_chk is not None else got[:P])
            cache_h = _dev(M["cache"], 0)
            rows = _gather_cache(cache_h, pt[lane], torch.arange(P), block)
            c_s = stats(M["rc"][lane].c[0, :P], rows[:, :rank] * M["gam"])
            k_s = stats(M["rc"][lane].k_pe[0, :P], rows[:, rank:])
            untouched = torch.equal(cache_h[spare], M["stale"][spare])
            log(
                f"{tag}: {fmt(s)}; replicas(32) {same}; cache c_kv pcc {c_s['pcc']:.6f} k_pe pcc {k_s['pcc']:.6f}; "
                f"spare blocks untouched {untouched}; {ms:.1f} ms"
            )
            if (
                s["pcc"] < PCC_MIN
                or s["nonfinite"]
                or not same
                or not untouched
                or min(c_s["pcc"], k_s["pcc"]) < PCC_MIN
            ):
                failures.append(f"{tag}: {fmt(s)} same={same} untouched={untouched} c {c_s['pcc']} k {k_s['pcc']}")
        _free([x_tt, pt_u])

    def decode(lanes, tag):
        pos_t, x32 = [-1] * B, torch.zeros(B, 4096)
        for lane in lanes:
            P = dict(users)[lane]
            pos_t[lane] = P + done[lane]
            x32[lane] = seqs[lane][pos_t[lane]]
        d = _decode_step_inputs(mesh_device, cfg, rope, pos_t, x32, pt)
        for layer, M in L.items():
            out = M["attn"].forward_decode(
                d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=M["cache"], active=d["act"]
            )
            got = _per_lane_out(cfg, mesh_device, out)
            same = replicas_identical(out, mesh_device, "tp", cfg.axes)
            _free(out)
            want = torch.stack(
                [M["ref"](x32[l][None, None], torch.tensor([[pos_t[l]]]), M["rc"][l])[0, 0] for l in lanes]
            )
            s = stats(want, got[lanes])
            per = [pcc(want[i], got[l]) for i, l in enumerate(lanes)]
            zero = all(float(got[l].abs().max()) == 0.0 for l in range(B) if pos_t[l] < 0)
            log(
                f"session {kv_name}/{block} L{layer} {cfg.layer(layer).attn_kind} decode {tag} positions "
                f"{[pos_t[l] for l in lanes]}: {fmt(s)}; per-user pcc {[round(p, 6) for p in per]}; replicas(tp) "
                f"{same}; inactive zero {zero}"
            )
            if s["pcc"] < PCC_MIN or min(per) < LANE_PCC_MIN or s["nonfinite"] or not same or not zero:
                failures.append(f"decode {tag} L{layer}: {fmt(s)} per {per} same={same} zero={zero}")
        _free_step(d)
        for lane in lanes:
            done[lane] += 1

    early, late = [l for l, _ in SESSION_EARLY], [l for l, _ in SESSION_LATE]
    for lane, P in SESSION_EARLY:
        prefill(lane, P)
    for i in range(SESSION_STEPS[0]):
        decode(early, f"phase-2 step {i}")
    log(f"session {kv_name}/{block} after the first decode steps: {_l1_report(mesh_device)}")
    for lane, P in SESSION_LATE:  # the serving case: new prefills between decode steps (global S = 4096 / 1024)
        prefill(lane, P)
    prefill(*SESSION_EARLY[1], fill=False)
    for i in range(SESSION_STEPS[1]):
        decode(early + late, f"phase-4 step {i}")
    log(f"session {kv_name}/{block} end: {_l1_report(mesh_device)}")
    for M in L.values():
        _free(M["cache"])
    assert not failures, "\n".join(failures)


# ======================================================================================================================
# device: pin of the L1_SMALL requirement with the shared device_params() (no l1_small_size today)
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH_SHARED_DEFAULT, indirect=True)
def test_attention_l1_small_hazard(mesh_device, device_params):
    """With the shared ``device_params()`` the mesh has no L1_SMALL region: the module must warn (and raise with
    ``require_l1_small=True``), and after one decode step the decode all_reduce's CCL semaphores sit in main L1 at
    ~1.0 MB, so a global-layer prefill at S = 1024 throws a static-CB clash. xfails with that message while the shared
    default lacks ``l1_small_size``; once it has one, the same sequence must pass with a bitwise-identical prefill."""
    cfg, ccl, rope = _setup(mesh_device, "L1_SMALL hazard (shared device_params)")
    size = l1_small_bytes(mesh_device)
    args = ref_args()
    t0, t1 = random_attn_tensors(args, seed=1000), random_attn_tensors(args, seed=1001)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        a0 = MotifAttention(mesh_device, cfg, 0, source=hf_source(t0, 0), ccl=ccl, rope=rope, cache=False)
    warned = any(L1_SMALL_WARNING in str(w.message) for w in rec)
    raised = None
    if size == 0:
        with pytest.raises(RuntimeError, match="L1_SMALL") as ei:
            MotifAttention(mesh_device, cfg, 0, source=hf_source(t0, 0), ccl=ccl, rope=rope, cache=False,
                           require_l1_small=True)  # fmt: skip
        raised = str(ei.value).split(".")[0]
    a1 = MotifAttention(mesh_device, cfg, 1, source=hf_source(t1, 1), ccl=ccl, rope=rope, cache=False)
    x = torch.randn(1024, 4096, generator=torch.Generator().manual_seed(3)).bfloat16().float()
    x_tt = _replicated(mesh_device, x[None, None], ttnn.bfloat16)
    o = a0.forward_prefill(x_tt)
    before = _dev(o)[0, 0]
    _free(o)
    # one SWA decode step (8 active lanes on row 0), small stale-filled pool
    block, B = cfg.kv_block_size, cfg.max_batch
    pos_t = [100 + 7 * l if l < 8 else -1 for l in range(B)]
    Wd = math.ceil((max(pos_t) + 1) / block)
    pt = torch.zeros(B, Wd, dtype=torch.int32)
    pt[:8] = (torch.arange(8 * Wd, dtype=torch.int32) + 1).reshape(8, Wd)
    cache = _alloc_cache(mesh_device, cfg, 1 + 8 * Wd, block)
    x32 = torch.randn(B, 4096, generator=torch.Generator().manual_seed(4)).bfloat16().float()
    d = _decode_step_inputs(mesh_device, cfg, rope, pos_t, x32, pt)
    _free(a1.forward_decode(d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache,
                            active=d["act"]))  # fmt: skip
    ttnn.synchronize_device(mesh_device)
    l1 = _l1_report(mesh_device)
    err = None
    try:
        o = a0.forward_prefill(x_tt)
        ttnn.synchronize_device(mesh_device)
        after = _dev(o)[0, 0]
        _free(o)
    except Exception as e:  # TT_THROW (static CBs clash with the leftover L1 semaphores)
        err = next((ln.strip() for ln in str(e).splitlines() if "clash" in ln), str(e).splitlines()[0])
    log(
        f"L1_SMALL hazard: l1_small {size} B, module warned {warned}, require_l1_small raised {raised!r}; after one "
        f"decode step {l1}; global prefill S=1024 after the decode: {'EXCEPTION ' + err if err else 'OK'}"
    )
    _free([x_tt, cache])
    _free_step(d)
    assert warned == (size == 0), f"warning expected iff the mesh has no L1_SMALL region (size {size})"
    if err is not None:
        assert size == 0, f"static-CB clash although the mesh has an L1_SMALL region: {err}"
        pytest.xfail(f"shared device_params() opens the mesh without l1_small_size (requested): {err}")
    assert torch.equal(before, after)


# ======================================================================================================================
# device: decode at long context (FlashMLA over up to 32K keys: ~256 k-chunks of bf16 running statistics)
# ======================================================================================================================
LONG_DECODE_POS = [8191, 16384, 24000, 32767]  # one long lane per DP row; the other lanes are inactive


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("layer", [0, 1], ids=["global_L0", "swa_L1"])
def test_attention_decode_long_context(mesh_device, device_params, layer):
    from models.demos.motif3.reference.cache import LatentKVCache

    cfg, ccl, rope = _setup(mesh_device, f"decode long context L{layer}")
    args = ref_args()
    tensors = random_attn_tensors(args, seed=300 + layer)
    attn = MotifAttention(mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=False)
    ref = ref_attention(args, layer, tensors, torch.float32)
    gam = tensors["kv_norm"].float()
    B, block, rank = cfg.max_batch, cfg.kv_block_size, cfg.kv_lora_rank
    lanes = [8 * dp + 3 for dp in range(cfg.dp)]
    pos = [-1] * B
    for l, p in zip(lanes, LONG_DECODE_POS):
        pos[l] = p
    Wd = cfg.kv_blocks_per_seq  # 512 = the serving page-table width
    nblk = [math.ceil((p + 1) / block) for p in LONG_DECODE_POS]
    pool = 1 + sum(nblk)
    g = torch.Generator().manual_seed(31)
    perm = (torch.randperm(pool - 1, generator=g) + 1).to(torch.int32)
    pt = torch.zeros(B, Wd, dtype=torch.int32)
    off = 0
    for l, nb in zip(lanes, nblk):
        pt[l, :nb] = perm[off : off + nb]
        off += nb
    paged = stale_paged(stale_bank(ref, "random", layer, seed=30 + layer), pool, block, seed=30 + layer)
    ref_cache = LatentKVCache(len(lanes), max(LONG_DECODE_POS) + 1, rank, cfg.rope_dim, torch.float32)
    xs = []
    for u, (l, p) in enumerate(zip(lanes, LONG_DECODE_POS)):
        gl = torch.Generator().manual_seed(4000 + u)
        seq = torch.randn(p + 1, 4096, generator=gl).to(torch.bfloat16).float()
        n, kpe = ref_latents(ref, seq[:p], torch.arange(p))
        idx = torch.arange(p)
        paged[pt[l, idx // block].long(), 0, idx % block] = torch.cat([n, kpe], -1)
        xs.append(seq[p])
    paged_q = _host_quant(paged, cfg.dtypes.kv_cache)
    for u, (l, p) in enumerate(zip(lanes, LONG_DECODE_POS)):
        rq = _gather_cache(paged_q, pt[l], torch.arange(p), block)
        ref_cache.c[u, :p] = rq[:, :rank] * gam
        ref_cache.k_pe[u, :p] = rq[:, rank:]
        ref_cache.seq_lens[u] = p
    cache = _upload_cache(mesh_device, paged_q, cfg.dtypes.kv_cache)
    # absorbed form (MQA over the latent; same math, ~1e-5 in fp32): the expanded form would materialize
    # [4, 80, 32K, 192] fp32 K / V (14 GB) on the host
    want = ref(torch.stack(xs)[:, None], torch.tensor(LONG_DECODE_POS)[:, None], ref_cache, mode="absorbed")[:, 0]
    x32 = torch.zeros(B, 4096)
    for l, x in zip(lanes, xs):
        x32[l] = x
    d = _decode_step_inputs(mesh_device, cfg, rope, pos, x32, pt)

    def fn():
        return attn.forward_decode(
            d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache, active=d["act"]
        )

    out = fn()
    got = _per_lane_out(cfg, mesh_device, out)
    _free(out)
    per = {p: pcc(want[u], got[l]) for u, (l, p) in enumerate(zip(lanes, LONG_DECODE_POS))}
    s = stats(want, got[lanes])
    tr = _traced_us(mesh_device, fn, n=8, reps=5)
    log(
        f"decode long context L{layer} {cfg.layer(layer).attn_kind} positions {LONG_DECODE_POS}: TT vs ref fp32 (bfp8 "
        f"history, stale tails) {fmt(s)}; per-position pcc {[round(v, 6) for v in per.values()]}; "
        f"traced {tr['slope_us']:.0f} us/call (lanes at <= 32K, W = {Wd})"
    )
    _free([cache])
    _free_step(d)
    assert s["pcc"] >= PCC_MIN and min(per.values()) >= LANE_PCC_MIN, fmt(s)


# ======================================================================================================================
# device: TT weight-cache round trip (cache=True writes, a rebuild loads without touching the source)
# ======================================================================================================================
class _NoTouchSource:
    def get(self, name, dtype=None):
        raise AssertionError(f"cache hit expected, but {name} was read")

    def has(self, name):
        return True


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_attention_weight_cache_roundtrip(mesh_device, device_params):
    import dataclasses
    import shutil

    cfg, ccl, rope = _setup(mesh_device, "weight cache")
    root = TEST_DIR / "wcache_tmp"
    shutil.rmtree(root, ignore_errors=True)
    cfg = dataclasses.replace(cfg, tt_cache_root=root)  # small, private cache root (deleted below)
    args = ref_args()
    layer = 1
    tensors = random_attn_tensors(args, seed=77)
    try:
        a1 = MotifAttention(mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=True)
        files = sorted(p.name for p in (cfg.cache_dir / f"L{layer:02d}").glob("*.tensorbin"))
        size = sum(p.stat().st_size for p in (cfg.cache_dir / f"L{layer:02d}").glob("*.tensorbin"))
        a2 = MotifAttention(mesh_device, cfg, layer, source=_NoTouchSource(), ccl=ccl, rope=rope, cache=True)
        x = torch.randn(128, 4096, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16).float()
        x_tt = _replicated(mesh_device, x[None, None], ttnn.bfloat16)
        o1, o2 = _dev(a1.forward_prefill(x_tt)), _dev(a2.forward_prefill(x_tt))
        log(f"weight cache: {len(files)} files, {size / 1e6:.1f} MB for one layer ({files}); reload bitwise "
            f"{torch.equal(o1, o2)}")  # fmt: skip
        assert torch.equal(o1, o2) and len(files) == 10
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ======================================================================================================================
# device: (8, 4) mesh orientation (TP = mesh dim 0): prefill + one decode step vs the reference
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params",
                         [pytest.param((8, 4), device_params(l1_small_size=L1_SMALL_SIZE), id="8x4-torus2d-l1small")],
                         indirect=True)  # fmt: skip
def test_attention_mesh_8x4(mesh_device, device_params):
    from models.demos.motif3.reference.cache import LatentKVCache
    from models.demos.motif3.tt.ccl import replicas_identical

    cfg, ccl, rope = _setup(mesh_device, "8x4")
    assert cfg.axes.tp_axis == 0 and cfg.tp == 8 and cfg.dp == 4
    args = ref_args()
    layer = 1
    tensors = random_attn_tensors(args, seed=88)
    attn = MotifAttention(mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=False)
    ref = ref_attention(args, layer, tensors, torch.float32)
    block, B = cfg.kv_block_size, cfg.max_batch
    P, lanes = 300, [5, 14, 23, 28]  # one user per DP row, all prefilled with the same prompt
    S = cfg.prefill_bucket(P)
    x = torch.randn(P + 1, 4096, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16).float()
    xp = torch.randn(S, 4096, generator=torch.Generator().manual_seed(12)).to(torch.bfloat16).float()  # pad != 0
    xp[:P] = x[:P]
    Wd = max(cfg.prefill_page_table_entries(S), math.ceil((P + 1) / block))
    pool = 1 + len(lanes) * Wd
    perm = (torch.randperm(pool - 1, generator=torch.Generator().manual_seed(3)) + 1).to(torch.int32)
    pt = torch.zeros(B, Wd, dtype=torch.int32)
    cache = _alloc_cache(mesh_device, cfg, pool, block)
    x_tt = _replicated(mesh_device, xp[None, None], ttnn.bfloat16)
    for u, l in enumerate(lanes):
        pt[l] = perm[u * Wd : (u + 1) * Wd]
        pt_u = _replicated(mesh_device, pt[l : l + 1, : cfg.prefill_page_table_entries(S)], ttnn.int32,
                           ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        out = attn.forward_prefill(x_tt, page_table=pt_u, kv_cache=cache)
        if u == 0:
            s_pre = stats(ref(x[None, :P], torch.arange(P)[None])[0], _dev(out)[0, 0, :P])
            same_pre = replicas_identical(out, mesh_device, "tp", cfg.axes) and replicas_identical(
                out, mesh_device, "dp", cfg.axes
            )
        _free([out, pt_u])
    rc = LatentKVCache(1, P + 1, cfg.kv_lora_rank, cfg.rope_dim, torch.float32)
    ref(x[None, :P], torch.arange(P)[None], rc)
    want = ref(x[None, P : P + 1], torch.tensor([[P]]), rc)[0, 0]
    pos = [-1] * B
    x32 = torch.zeros(B, 4096)
    for l in lanes:
        pos[l], x32[l] = P, x[P]
    d = _decode_step_inputs(mesh_device, cfg, rope, pos, x32, pt)
    out = attn.forward_decode(d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache,
                              active=d["act"])  # fmt: skip
    got = _per_lane_out(cfg, mesh_device, out)
    per = [pcc(want, got[l]) for l in lanes]
    same = replicas_identical(out, mesh_device, "tp", cfg.axes)
    zero = all(float(got[l].abs().max()) == 0.0 for l in range(B) if pos[l] < 0)
    log(
        f"8x4 mesh L{layer}: prefill {fmt(s_pre)} replicas(32) {same_pre}; decode at {P} per-row pcc "
        f"{[round(p, 6) for p in per]} replicas(tp) {same} inactive zero {zero}"
    )
    assert s_pre["pcc"] >= PCC_MIN and same_pre and min(per) >= LANE_PCC_MIN and same and zero


# ======================================================================================================================
# device: trace capture of one full decode step (rot gather + active mask + attention), replayed with NEW inputs,
# including a NEW page table (block-boundary crossing + remapped blocks)
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("layer", [1, 0], ids=["swa_L1", "global_L0"])
def test_attention_decode_trace_replay(mesh_device, device_params, layer):
    """Persistent device inputs (x, cur_pos, rot_idxs, page_table) -> one trace with the per-step helpers
    (``decode_rope_tables``, ``active_mask_from_cur_pos``) and ``forward_decode``. Replayed twice with different
    positions / lanes (one lane becomes inactive, an inactive one becomes active) / x **and page table**: between the
    replays lane 4 crosses a block boundary (127 -> 128: its next page-table entry goes from the null block to a fresh
    block) and lanes 9 / 17 / 26 get an in-window block remapped to a spare block with different contents. Each replay
    must equal the eager step bitwise (outputs and the whole cache afterwards), inactive rows stay 0, and the remap must
    change the remapped lanes' outputs (a control run with the old page table differs)."""
    from models.demos.motif3.tt.rope import shard_lanes

    cfg, ccl, rope = _setup(mesh_device, f"decode trace replay L{layer}")
    args = ref_args()
    tensors = random_attn_tensors(args, seed=900 + layer)
    attn = MotifAttention(mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=False)
    ref = ref_attention(args, layer, tensors, torch.float32)
    B, block = cfg.max_batch, cfg.kv_block_size
    pos_a = list(DECODE_POS)
    pos_b = [p + 1 if p >= 0 else -1 for p in pos_a]
    pos_b[1], pos_b[24] = -1, 0  # lane 1 drops out, lane 24 (inactive in A) starts at position 0
    assert pos_a[4] == 127 and pos_b[4] == 128  # lane 4 crosses into its block 2
    remap = [9, 17, 26]
    Wd = math.ceil((max(pos_a) + 2) / block)
    n_spare = 1 + len(remap)  # lane 4's fresh block + one spare per remapped lane
    pool = 1 + B * Wd + n_spare
    g = torch.Generator().manual_seed(21)
    perm = (torch.randperm(pool - 1, generator=g) + 1).to(torch.int32)
    pt_b = perm[: B * Wd].reshape(B, Wd).clone()
    spare = perm[B * Wd :].tolist()
    pt_b[4, 2] = spare[0]
    pt_a = pt_b.clone()
    pt_a[4, 2:] = 0  # step A: lane 4 has blocks 0-1 only (positions <= 127); the rest of its row is the null block
    for i, l in enumerate(remap):  # the block before the one holding position pos_b[l] (inside the window)
        j = pos_b[l] // block - 1
        assert j >= 0 and pos_a[l] // block == j + 1
        pt_b[l, j] = spare[1 + i]
    seqs = _lane_sequences("random", layer, [max(p, 1) for p in pos_a], 2)
    paged = stale_paged(stale_bank(ref, "random", layer, seed=40 + layer), pool, block, seed=40 + layer)
    for lane, p in enumerate(pos_a):
        if p <= 0:
            continue
        n, kpe = ref_latents(ref, seqs[lane][:p], torch.arange(p))
        idx = torch.arange(p)
        paged[pt_a[lane, idx // block].long(), 0, idx % block] = torch.cat([n, kpe], -1)
    paged_q = _host_quant(paged, cfg.dtypes.kv_cache)

    def x_of(pos, salt):
        return torch.stack([seqs[l][min(max(p, 0), seqs[l].shape[0] - 1)] * (1.0 + 0.1 * salt) for l, p in
                            enumerate(pos)]).to(torch.bfloat16).float()  # fmt: skip

    def host_inputs(pos, xs, pt):
        rows = xs.reshape(cfg.dp, 1, cfg.lanes_per_row, 4096)
        return {
            "x": shard_lanes(rows, cfg, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None),
            "cur": shard_lanes(torch.tensor(pos, dtype=torch.int32), cfg, mesh_device, dtype=ttnn.int32, device=None),
            "rot": rope.rot_idxs_host(torch.tensor(pos)),
            "pt": shard_lanes(pt, cfg, mesh_device, dtype=ttnn.int32, device=None),
        }

    steps = [(pos_a, x_of(pos_a, 0), pt_a), (pos_b, x_of(pos_b, 1), pt_b)]

    def step(dev, kv_cache):
        rot = MotifAttention.decode_rope_tables(rope, dev["rot"])
        act = MotifAttention.active_mask_from_cur_pos(dev["cur"], cfg.lanes_per_row)
        out = attn.forward_decode(
            dev["x"], rot=rot, cur_pos=dev["cur"], page_table=dev["pt"], kv_cache=kv_cache, active=act
        )
        _free([act] + [t for cs in rot.values() for t in cs])
        return out

    def to_dev(h):
        return {k: ttnn.to_device(v, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for k, v in h.items()}

    def run_eager(plan):
        cache = _upload_cache(mesh_device, paged_q, cfg.dtypes.kv_cache)
        outs = []
        for pos, xs, pt in plan:
            dev = to_dev(host_inputs(pos, xs, pt))
            out = step(dev, cache)
            outs.append(_per_lane_out(cfg, mesh_device, out))
            _free([out] + list(dev.values()))
        return outs, cache

    eager, cache_e = run_eager(steps)
    control, cache_c = run_eager([steps[0], (pos_b, steps[1][1], pt_a)])  # step B with the OLD page table
    _free(cache_c)
    # traced run: persistent inputs, one capture, two replays with new inputs (page table included)
    cache_t = _upload_cache(mesh_device, paged_q, cfg.dtypes.kv_cache)
    dev = to_dev(host_inputs(*steps[0]))
    scratch = _upload_cache(mesh_device, paged_q, cfg.dtypes.kv_cache)
    _free(step(dev, scratch))  # compile with the same shapes on a scratch cache (programs cached)
    _free(scratch)
    ttnn.synchronize_device(mesh_device)
    with _Capture(mesh_device) as cap:
        out_t = step(dev, cache_t)
    failures = []
    try:
        for i, (pos, xs, pt) in enumerate(steps):
            h = host_inputs(pos, xs, pt)
            for k in ("x", "cur", "rot", "pt"):
                ttnn.copy_host_to_device_tensor(h[k], dev[k])
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            got = _per_lane_out(cfg, mesh_device, out_t)
            exact = torch.equal(got, eager[i])
            zero = all(float(got[l].abs().max()) == 0.0 for l in range(B) if pos[l] < 0)
            nonzero = all(float(got[l].abs().max()) > 0.0 for l in range(B) if pos[l] >= 0)
            log(
                f"decode trace replay L{layer} step {i}: traced == eager bitwise {exact} (max diff "
                f"{float((got - eager[i]).abs().max()):.3e}); inactive rows zero {zero}; active rows non-zero {nonzero}"
            )
            if not (exact and zero and nonzero):
                failures.append(f"step {i}: exact={exact} zero={zero} nonzero={nonzero}")
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    # the whole cache after both steps: traced == eager bitwise on one chip per DP row; lane 4's write went to the
    # fresh block (row 0 = the step-B latent), never to the null block (still stale)
    cache_same, fresh_written, null_stale = True, True, True
    for dp in range(cfg.dp):
        ce, ct = _dev(cache_e, _chip_index(cfg, dp)), _dev(cache_t, _chip_index(cfg, dp))
        cache_same &= torch.equal(ce, ct)
        null_stale &= torch.equal(ct[0], paged_q[0])
        if dp == 0:
            fresh_written = not torch.equal(ct[spare[0], 0, 0], paged_q[spare[0], 0, 0])
        del ce, ct
    diff_remap = {l: float((eager[1][l] - control[1][l]).abs().max()) for l in remap}
    same_other = all(torch.equal(eager[1][l], control[1][l]) for l in range(B) if l not in remap + [4])
    log(
        f"decode trace replay L{layer}: cache traced == eager bitwise {cache_same}; lane 4 fresh block written "
        f"{fresh_written}; null block untouched {null_stale}; remapped lanes' outputs vs old page table (control) max "
        f"diff {[f'{v:.3e}' for v in diff_remap.values()]}; other lanes identical to the control {same_other}"
    )
    _free([cache_e, cache_t, out_t] + list(dev.values()))
    if not (cache_same and fresh_written and null_stale and same_other and min(diff_remap.values()) > 1e-3):
        failures.append(
            f"cache_same={cache_same} fresh={fresh_written} null={null_stale} other={same_other} remap={diff_remap}"
        )
    assert not failures, "\n".join(failures)


# ======================================================================================================================
# device: decode latency per module call (eager / traced), 4K context
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_attention_decode_latency(mesh_device, device_params):
    cfg, ccl, rope = _setup(mesh_device, "decode latency")
    args = ref_args()
    B, block, ctx = cfg.max_batch, cfg.kv_block_size, 4096
    Wd = ctx // block
    pool = 1 + B * Wd
    cache = _alloc_cache(mesh_device, cfg, pool, block)
    g = torch.Generator().manual_seed(3)
    pt = (torch.randperm(pool - 1, generator=g) + 1).to(torch.int32).reshape(B, Wd)
    pos = [ctx - 1] * B
    d = _decode_step_inputs(mesh_device, cfg, rope, pos, torch.randn(B, 4096, generator=g).bfloat16().float(), pt)
    for layer in (1, 0):
        tensors = random_attn_tensors(args, seed=50 + layer)
        attn = MotifAttention(
            mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=False
        )

        def fn():
            return attn.forward_decode(
                d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache, active=d["act"]
            )

        eager = _eager_us(mesh_device, fn, iters=10)
        tr = _traced_us(mesh_device, fn, n=16, reps=7)
        log(
            f"decode latency L{layer} {cfg.layer(layer).attn_kind} ctx {ctx}: eager {eager:.0f} us/call, traced "
            f"{tr['slope_us']:.1f} us/call (raw {tr['raw_us']:.1f}; replay 8 calls {tr['t_half']:.0f} us, 16 calls "
            f"{tr['t_full']:.0f} us)"
        )
        del attn
    _free(cache)
    _free_step(d)


def _decode_op_sequence(attn: MotifAttention, x, rot, cur, pt, cache, act):
    """The ops of ``MotifAttention.forward_decode`` one by one on real intermediates (for the per-op breakdown):
    ``[(name, thunk)]``, every thunk re-runs one op on stored inputs. Mirrors forward_decode (keep in sync)."""
    cos, sin = attn._rot_tables(rot)
    ops = []
    pc = attn.decode_pcs

    def op(name, fn):
        ops.append((name, fn))
        return fn()

    eps, ck_n, ck_h, ck_l = attn.cfg.rms_norm_eps, attn.ckc_norm, attn.ckc_heads, attn.ckc_latent
    split = ttnn.experimental.nlp_create_q_heads_split
    cq = op("x@Wq_lat (fp32 out)", lambda: attn._linear(x, attn.w_q_lat, ckc=ck_l, pc=pc["q_lat"], dtype=ttnn.float32))
    kvl = op("x@Wkv_lat", lambda: attn._linear(x, attn.w_kv_lat, ckc=ck_l, pc=pc["kv_lat"]))
    cq_n = op("rms_norm q (fp32, weightless)", lambda: ttnn.rms_norm(cq, epsilon=eps, compute_kernel_config=ck_n))
    q = op("cq@wq_b", lambda: attn._linear(cq_n, attn.w_q_b, ckc=ck_h, pc=pc["wq_b"]))
    g = op("cq@wq_b_gate+sigmoid", lambda: attn._linear(cq_n, attn.w_gate, ckc=ck_h, pc=pc["gate"],
                                                        activation="sigmoid"))  # fmt: skip
    c_raw, rest = op("split kvl [c_raw|rest]", lambda: split(kvl, num_heads=1, split_head_dim=attn.rank))
    kpe, lam = op("split rest [kpe|lam]", lambda: split(rest, num_heads=1, split_head_dim=attn.rope_dim))
    n = op("rms_norm kv", lambda: ttnn.rms_norm(c_raw, epsilon=eps, compute_kernel_config=ck_n))
    qn, qp = op("q heads split", lambda: split(q, num_heads=attn.H, split_head_dim=attn.nope))
    q_lat = op("bmm W_UK", lambda: ttnn.matmul(qn, attn.w_uk, program_config=pc["w_uk"], compute_kernel_config=ck_h))
    q_pe = op("rope q", lambda: attn._rope(qp, cos, sin))
    qh = op("concat Q", lambda: ttnn.concat([q_lat, q_pe], dim=-1))
    q_mla = op("transpose Q", lambda: ttnn.transpose(qh, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    k_pe = op("rope k", lambda: attn._rope(kpe, cos, sin))
    kv_row = op("concat kv", lambda: ttnn.concat([n, k_pe], dim=-1))
    kv_upd = op("transpose kv -> sharded", lambda: ttnn.transpose(kv_row, 1, 2, memory_config=attn.update_mc))

    def upd():  # in place: never hand the cache back to _free
        ttnn.experimental.paged_update_cache(cache, kv_upd, update_idxs_tensor=cur, page_table=pt)
        return None

    op("paged_update_cache", upd)
    o = op("FlashMLA decode", lambda: ttnn.transformer.paged_flash_multi_latent_attention_decode(
        q_mla, cache, None, head_dim_v=attn.rank, page_table_tensor=pt, cur_pos_tensor=cur, scale=1.0,
        sliding_window_size=attn.window, program_config=attn.decode_pc, compute_kernel_config=attn.ckc_sdpa_decode,
        memory_config=ttnn.DRAM_MEMORY_CONFIG))  # fmt: skip
    oh = op("transpose O", lambda: ttnn.transpose(o, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    u = op("bmm W_UV", lambda: ttnn.matmul(oh, attn.w_uv, program_config=pc["w_uv"], compute_kernel_config=ck_h))
    uf = op("concat heads", lambda: ttnn.experimental.nlp_concat_heads(u))
    us, nz = op("split U [sig|noise]", lambda: split(uf, num_heads=1, split_head_dim=attn.Sg * attn.vdim))
    un = op("noise@X", lambda: attn._linear(nz, attn.noise_expand, ckc=ck_h))
    ve = op("sigmoid(lam@E)", lambda: attn._linear(lam, attn.lam_expand, ckc=ck_h, activation="sigmoid"))
    d = op("addcmul", lambda: ttnn.addcmul(us, ve, un, value=-1.0))
    dg = op("mul gate", lambda: ttnn.multiply(d, g))
    dgm = op("where(active)", lambda: ttnn.where(act, dg, 0.0))
    part = op("D@wo", lambda: attn._linear(dgm, attn.w_o, ckc=ck_h, pc=pc["wo"]))
    op("all_reduce(tp)", lambda: attn.ccl.ar_tp(part))
    return ops


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_attention_decode_breakdown(mesh_device, device_params):
    """Traced per-op cost of the decode sequence (each op alone in a trace, slope method), 4K context."""
    cfg, ccl, rope = _setup(mesh_device, "decode breakdown")
    args = ref_args()
    B, block, ctx = cfg.max_batch, cfg.kv_block_size, 4096
    Wd = ctx // block
    pool = 1 + B * Wd
    cache = _alloc_cache(mesh_device, cfg, pool, block)
    g = torch.Generator().manual_seed(3)
    pt = (torch.randperm(pool - 1, generator=g) + 1).to(torch.int32).reshape(B, Wd)
    pos = [ctx - 1] * B
    d = _decode_step_inputs(mesh_device, cfg, rope, pos, torch.randn(B, 4096, generator=g).bfloat16().float(), pt)
    for layer in (1, 0):
        tensors = random_attn_tensors(args, seed=50 + layer)
        attn = MotifAttention(
            mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=False
        )
        ops = _decode_op_sequence(attn, d["x"], d["rot"], d["cur"], d["pt"], cache, d["act"])
        rows, total = [], 0.0
        for name, fn in ops:
            tr = _traced_us(mesh_device, fn, n=32, reps=5)
            us = tr["slope_us"] if tr["slope_us"] > 0.5 else tr["raw_us"]
            total += us
            rows.append(f"{name}: {us:.1f}")
        log(
            f"decode breakdown L{layer} {cfg.layer(layer).attn_kind} ctx {ctx} (traced us/op; sum {total:.0f}): "
            + "; ".join(rows)
        )
        del attn


# ======================================================================================================================
# device: long prefill buckets (S = 8K / 16K / 32K; ATTN-7), row-subset golden; default role vs fp32-acc opt-in
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("S", LONG_S)
def test_attention_prefill_long(mesh_device, device_params, S):
    cfg, ccl, rope = _setup(mesh_device, f"prefill long S={S}")
    args = ref_args()
    block = cfg.kv_block_size
    failures = []
    for layer in (1, 0):
        tensors, x, rows = long_case(args, layer, S)
        rows, want, n_rows = long_golden(args, layer, S, tensors, x, rows)
        variants = {"default (sdpa_prefill role)": {}}
        if cfg.layer(layer).sliding_window_size is None:  # SWA at S >= 256: the opt-in is the role anyway
            variants["fp32-acc opt-in"] = {"sdpa_prefill_fp32_acc": "auto"}
        g = torch.Generator().manual_seed(S + layer + 1)
        nb = S // block
        pool = 1 + nb
        pt = (torch.randperm(pool - 1, generator=g) + 1).to(torch.int32)[None]
        x_tt = _replicated(mesh_device, x[None, None], ttnn.bfloat16)
        pt_tt = _replicated(mesh_device, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        for vname, kw in variants.items():
            attn = MotifAttention(
                mesh_device, cfg, layer, source=hf_source(tensors, layer), ccl=ccl, rope=rope, cache=False, **kw
            )
            fill = not kw
            cache = _alloc_cache(mesh_device, cfg, pool, block) if fill else None
            fkw = {"page_table": pt_tt, "kv_cache": cache} if fill else {}
            t0 = time.perf_counter()
            out = attn.forward_prefill(x_tt, **fkw)
            ttnn.synchronize_device(mesh_device)
            dt = (time.perf_counter() - t0) * 1e3
            got_full = _dev(out)[0, 0]
            got = got_full[rows]
            s = stats(want, got)
            finite = bool(torch.isfinite(got_full).all())
            ttnn.deallocate(out)
            ts = []  # min of 3 synced eager calls (a single call picks up host contention from the other agents)
            for _ in range(3):
                t0 = time.perf_counter()
                out = attn.forward_prefill(x_tt, **fkw)
                ttnn.synchronize_device(mesh_device)
                ts.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(out)
            dt2 = min(ts)
            c_txt, c_ok = "", True
            if fill:  # cache spot check (M18) on the selected rows: the gamma-free unit-RMS latent
                crow = _gather_cache(_dev(cache, 0), pt[0], rows, block)
                c_s = stats(n_rows, crow[:, : cfg.kv_lora_rank])
                c_txt, c_ok = f"; cache latent pcc {c_s['pcc']:.6f}", c_s["pcc"] >= PCC_MIN
                ttnn.deallocate(cache)
            log(
                f"prefill long L{layer} {cfg.layer(layer).attn_kind} S={S} {vname}: rows {len(rows)} {fmt(s)}; all rows "
                f"finite {finite}{c_txt}; eager {dt2:.1f} ms/call{' incl. cache fill' if fill else ''} (min of 3, max "
                f"{max(ts):.1f}; first call {dt:.0f} ms incl. compile)"
            )
            if s["pcc"] < PCC_MIN or not finite or not c_ok:
                failures.append(f"L{layer} S={S} {vname}: {fmt(s)} finite={finite}{c_txt}")
            del attn
        _free([x_tt, pt_tt])
    assert not failures, "\n".join(failures)


# ======================================================================================================================
# device: upstream bug pin -- SDPA prefill with sliding_window_size + fp32 dest acc (why SWA prefill keeps the G2 role)
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.xfail(
    reason="tt-metal SDPA prefill: sliding_window_size with fp32_dest_acc_en=True returns wrong values for S >= 256 "
    "(PCC 0.958-0.989 for every chunk config; no window or fp32 acc off: 0.9997-0.99999). fp32 acc selects the legacy "
    "compute kernel (sdpa_program_factory.cpp can_use_streaming_compute), whose reader builds the window mask with "
    "generate_causal_sliding_window_mask (sdpa/device/kernels/dataflow/dataflow_common.hpp): a K tile counts as fully "
    "allowed when k_tile_start >= min_window_start, the window start of the Q tile's FIRST row, so rows 1..31 of each Q "
    "tile also see up to 31 keys older than their window (fix: compare with max_window_start). Confirmed on device: the "
    "output matches that emulated mask (legacy_sdpa_window_mask) at PCC 0.999995, and the same kernel with the window "
    "as an explicit bf16 attn_mask is exact (0.999995). MotifAttention therefore never uses fp32 acc with a window.",
    strict=False,
)
def test_sdpa_prefill_window_fp32_acc_upstream_bug(mesh_device, device_params):
    cfg, _, _ = _setup(mesh_device, "sdpa window+fp32 bug pin")
    S = 512
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 10, S, 192, generator=g).bfloat16().float() * 192**-0.5
    k = torch.randn(1, 2, S, 192, generator=g).bfloat16().float()
    v = torch.zeros(1, 2, S, 192)
    v[..., :128] = torch.randn(1, 2, S, 128, generator=g).bfloat16().float()
    pos = torch.arange(S)
    allow = (pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - 129)

    def golden(mask):
        return torch.stack(
            [
                torch.softmax((q[0, h] @ k[0, h // 5].T).masked_fill(~mask, float("-inf")), -1) @ v[0, h // 5]
                for h in range(10)
            ]
        )[..., :128]

    want, want_legacy = golden(allow), golden(legacy_sdpa_window_mask(S, 129))
    q_tt, k_tt, v_tt = (_replicated(mesh_device, t, ttnn.bfloat16) for t in (q, k, v))
    mask_tt = _replicated(mesh_device, torch.zeros(1, 1, S, S).masked_fill(~allow, float("-inf")), ttnn.bfloat16)

    def run(fp32: bool, explicit_mask: bool):
        ckc = ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
        )
        kw = {"attn_mask": mask_tt, "is_causal": False} if explicit_mask else {"is_causal": True}
        out = ttnn.transformer.scaled_dot_product_attention(
            q_tt,
            k_tt,
            v_tt,
            scale=1.0,
            sliding_window_size=None if explicit_mask else 129,
            program_config=cfg.sdpa_prefill_pc("swa", seq_len=S),
            compute_kernel_config=ckc,
            **kw,
        )
        got = _dev(out)[0, :, :S, :128]
        ttnn.deallocate(out)
        return got

    got = run(True, False)  # the bug: sliding_window_size + fp32 acc (legacy kernel)
    p, p_leg = pcc(want, got), pcc(want_legacy, got)
    p_stream = pcc(want, run(False, False))  # control: fp32 acc off (streaming kernel)
    p_mask = pcc(want, run(True, True))  # the same window as an explicit bf16 mask, fp32 acc (legacy kernel)
    log(
        f"sdpa prefill window 129 + fp32 acc, S={S}: pcc vs exact window {p:.6f}, vs the emulated legacy-kernel mask "
        f"(first-row 'fully allowed' test, legacy_sdpa_window_mask) {p_leg:.6f}; controls: fp32 acc off (streaming) "
        f"{p_stream:.6f}, explicit bf16 window mask + fp32 acc {p_mask:.6f} (upstream bug pin; expected >= 0.999 when "
        f"fixed)"
    )
    _free([q_tt, k_tt, v_tt, mask_tt])
    assert p_stream >= PCC_MIN
    assert p >= PCC_MIN


if __name__ == "__main__":  # host-only helpers: scripts/hostrun.sh -- python <this file> [--real-inputs] [--long]
    import sys

    if "--real-inputs" in sys.argv:
        print(make_real_inputs())
    if "--long" in sys.argv:
        _args = ref_args()
        for _S in LONG_S:
            for _layer in (1, 0):
                _t0 = time.time()
                long_golden(_args, _layer, _S)
                print(f"long golden L{_layer} S={_S}: {time.time() - _t0:.1f} s", flush=True)
