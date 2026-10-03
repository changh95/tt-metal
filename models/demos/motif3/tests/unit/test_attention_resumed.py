# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Work packages 2a and 2b of the features design (``docs/features/FEATURES_DESIGN.md`` §3.2, §3.5, §3.6, §5.2; README
§15-§17): the attention for resumed / chunked prefill, the decode KV-write hook and the MTP layer.

WP2a:

* ``MotifAttention(spec=, weight_prefix=)`` (``tt.attention.resolve_attn_layer``): the defaults for decoder layers and
  for the MTP layer (``cfg.mtp_layer_spec()``, ``model.mtp_layers.0.self_attn``), and the guards that keep a
  non-canonical spec or weight prefix out of the TT weight cache.
* ``MotifAttention.fill_kv``: the KV-only latent fill (MTP prefill, chunked fills). It writes through ``-1``-skip fill
  tables, at any position: offset RoPE comes from ``MotifRope.chunk_rope_tables``.
* Draft 1 is unchanged. The module at commit :data:`DRAFT1_COMMIT`, loaded from git, and the current one have bitwise
  equal host weight transforms, and on device bitwise equal prefill and decode outputs and caches.

WP2b (``forward_prefill(chunk=PrefillChunkInputs)``, ``forward_decode(kv_write=)``):

* host: ``chunk_host_tables`` on the design's worked examples (explicit plans at A = 64 and A = 128, so they hold
  whatever alignment / cost table the config has) and its guards, including tables that disagree with each other;
  ``PrefillChunkInputs.write`` (in-place rewrite; ``regather=False`` drops the stale RoPE rows); the recorded op
  sequences of the sp0 / sp1 global / sp1 SWA chunk paths (sp0 == draft 1; global fills before the chunked SDPA with
  the G9 role; SWA tail slices -> square window-129 SDPA -> fill; no leaks; guards raise before any op) and of the
  decode hook; every design schedule through an fp64 emulation of the chunk dataflow on an emulated paged cache,
  driven only by the host tables (rows == reference, own blocks == single-shot latent, shared / other blocks
  untouched, negative controls);
* device (``test_wp2b_*``): the design §5.2 schedules of a 4096-token prompt on L0 (global) and L1 (SWA), random and
  real weights with the bfp8 cache, random weights with a bf16 cache, against the draft-1 single shot and the fp32
  reference (sp0 chunks bitwise draft 1, SWA sp1 rows bitwise the single shot wherever that is exact, final caches
  bitwise on 32 chips, negative controls); the warm-up of every ``(path, bucket)`` compiles every program a real chunk
  needs and writes nothing; the decode hook with a draft-1 writer and a KV-R writer; an sp1 chunk captured in a trace
  at one start and replayed at another; a cost report.

Every chunk input is the prompt rows plus JUNK bucket-padding rows (:func:`chunk_input_rows`): serving pads a bucket
with padding tokens, never with the prompt's next rows. With the true next rows as padding, an intermediate chunk's
fill would already write the correct latents into the request's own partial last block, and a continuation that
failed to rewrite that block (design §3.1, "unaligned 1348": block 21) would pass every check; the schedule tests run
that sabotage as a negative control.

Host only (root conftest active, devices hidden; the ``-k cpu`` tests)::

    S=/home/ttuser/hchang/experiments/motif-3/scripts
    $S/hostrun.sh -n attn_resumed_host -- python -m pytest -p no:cacheprovider -q \
        models/demos/motif3/tests/unit/test_attention_resumed.py -k cpu

Device (one mesh open per test; ~6 min for all)::

    $S/devrun.sh -t 2400 -n attn_resumed -- python -m pytest \
        models/demos/motif3/tests/unit/test_attention_resumed.py -k "not cpu" -s -p no:cacheprovider

Goldens: the reference ``GDLAttention`` with the same bf16-valued weights and inputs, in fp32 (fp64 for the exact host
algebra), as in ``test_attention.py``. Its helpers are reused from there. Decoder layers 0 (global, YaRN) and 1 (SWA)
use random weights, and in WP2b also their real weights (layers 0-3 are BF16 on disk) with the real inputs of the first
4096 tokens of a chat prompt (:func:`load_long_real_inputs`). The MTP layer uses its real weights (checkpoint shard
104, kept in BF16) and realistic inputs: the
first half of the reference ``MotifMTP``, ``input_layernorm(input_proj([hn_p | embed_norm(embed(t_{p+1}))]))``, on the
C2 prompt ``en_technical``. ``hn_p`` is taken from the C2 golden's final hidden states
(:func:`make_mtp_attention_inputs`, cached under ``tt_cache/test/attention_resumed``). Every device check prints an
``[attn-resumed]`` line.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.test_attention import (
    ATTN_KEYS,
    HF_META,
    MESH,
    _Capture,
    _decode_step_inputs,
    _dev,
    _free,
    _free_step,
    _gather_cache,
    _host_quant,
    _per_lane_out,
    _replicated,
    _setup,
    _upload_cache,
    fmt,
    make_real_inputs,
    pcc,
    random_attn_tensors,
    real_attn_tensors,
    ref_args,
    ref_latents,
    stale_bank,
    stale_paged,
    stats,
)
from models.demos.motif3.tt import attention as A
from models.demos.motif3.tt import prefill_plan as PP
from models.demos.motif3.tt import weights as W
from models.demos.motif3.tt.attention import (
    MTP_ATTN_PREFIX,
    ChunkHostTables,
    MotifAttention,
    PrefillChunkInputs,
    _AttnSource,
    attn_weight_prefix,
    canonical_attn_spec,
    chunk_host_tables,
    hf_order_from_virtual,
    lambda_expansion,
    latent_kv_weight_for_chip,
    latent_q_weight,
    max_sp1_bucket,
    noise_expansion,
    resolve_attn_layer,
    virtual_head_order,
    w_uk_virtual_for_chip,
    w_uv_virtual_for_chip,
    warmup_chunk_host_tables,
    warmup_start,
    wq_b_gate_for_chip,
    wq_b_virtual_for_chip,
)
from models.demos.motif3.tt.generator_api import cdiv
from models.demos.motif3.tt.model_config import PROJECT_ROOT, MotifTTConfig
from models.demos.motif3.tt.rope import chunk_rot_rows, cos_sin_table, inv_freq_for_kind

TEST_DIR = PROJECT_ROOT / "tt_cache" / "test" / "attention_resumed"
C2_DIR = PROJECT_ROOT / "goldens" / "c2"
MTP_PROMPT = "en_technical"  # 893 tokens: 892 MTP rows (positions 0 .. 891)
MTP_INPUTS = TEST_DIR / f"mtp_attn_inputs_{MTP_PROMPT}.pt"
MTP_INPUTS_FORMAT = 1  # bump when make_mtp_attention_inputs changes meaning
METAL_ROOT = Path(__file__).resolve().parents[5]
DRAFT1_COMMIT = "01d0781cb56"  # last commit of tt/attention.py before WP2a: draft 1, validated end to end
DRAFT1_MODULE = "models.demos.motif3.tt._attention_draft1"
OUT_PCC_MIN = 0.999  # module output vs the fp32 reference (test_attention.py's bar)
LANE_PCC_MIN = 0.999  # per decode lane
LAT_PCC_MIN = 0.9999  # latent cache rows vs the reference c_kv / gamma and k_pe (gate G14's bar)
# Relative Frobenius error bound next to every PCC bar: pcc() is 1.0 when one side has zero variance, so an all-zero
# output (nothing computed / written) would otherwise pass. PCC 0.999 at matched norms is a relative error of ~0.045.
REL_MAX = 0.1
# 32 decode lanes for the draft-1 comparison: window edges, block edges, two inactive lanes; < 320 (5 blocks of 64)
DRAFT1_DECODE_POS = [
    0, 1, 5, 63, 64, 127, 128, 129,
    130, 131, 192, 200, 255, 256, 257, 299,
    -1, 9, 100, 140, 160, 180, 220, 240,
    250, 260, 270, 280, 290, 295, -1, 298,
]  # fmt: skip
# resumed rows (s, e) -> one chunk each (design §3.1 worked examples + one more start of the 512 bucket)
CHUNK_CASES = [(640, 1000), (1348, 3000), (64, 900), (1280, 1700)]


def log(msg: str) -> None:
    print(f"[attn-resumed] {msg}", flush=True)


# ======================================================================================================================
# pure torch helpers
# ======================================================================================================================
def host_cfg(**kw) -> MotifTTConfig:
    return MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8), **kw)


def ref_attention_spec(args, layer: int, tensors: Dict[str, torch.Tensor], dtype=torch.float32, *, swa=None):
    """Reference ``GDLAttention(args, layer, swa=swa)`` with ``tensors`` in ``dtype``. The reference ``MotifMTP``
    builds its attention as ``GDLAttention(args, 53, swa=True)``."""
    from models.demos.motif3.reference.modules import GDLAttention

    with torch.device("meta"):
        m = GDLAttention(args, layer, swa=swa)
    m.load_state_dict({f"{k}.weight": v.to(dtype) for k, v in tensors.items()}, strict=True, assign=True)
    return m.eval().requires_grad_(False)


def source_for(tensors: Dict[str, torch.Tensor], prefix: str) -> W.DictWeightSource:
    return W.DictWeightSource({f"{prefix}.{k}.weight": v for k, v in tensors.items()})


def real_mtp_tensors() -> Dict[str, torch.Tensor]:
    """The 9 attention tensors of the MTP layer (``model.mtp_layers.0.self_attn.*``, shard 104), as fp32."""
    try:
        loader = W.HFWeightLoader()
        return {k: loader.get(f"{MTP_ATTN_PREFIX}.{k}.weight").float() for k in ATTN_KEYS}
    except (W.MissingWeightError, FileNotFoundError) as e:
        pytest.skip(f"MTP attention weights not on disk: {e}")


def _rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _mask(pos: torch.Tensor, window) -> torch.Tensor:
    m = pos[None, :] <= pos[:, None]
    if window is not None:
        m &= pos[None, :] > pos[:, None] - window
    return m


def emulate_attention(cfg: MotifTTConfig, spec, src: _AttnSource, x: torch.Tensor, mode: str) -> torch.Tensor:
    """Torch emulation of the TT per-chip dataflow, summed over the 8 TP chips (``all_reduce(tp)``), positions
    ``0 .. T-1``: ``test_attention.emulate_tt`` for any ``LayerSpec`` and weight source (the MTP layer). ``"decode"``
    is the absorbed MQA over the 576-wide latent, ``"prefill"`` the expanded GQA from ``n @ E_pref``. It uses exactly
    the module's weight transforms, with math in the dtype of ``x`` (fp64 for exactness)."""
    from models.demos.motif3.reference.rope import apply_rope, rope_cos_sin

    T = x.shape[0]
    pos = torch.arange(T)
    cos, sin = rope_cos_sin(inv_freq_for_kind(cfg, spec.rope_kind), pos[None], x.dtype)
    H, G, r, Sg = cfg.q_heads_per_chip, cfg.kv_groups_per_chip, cfg.grouped_ratio, cfg.signal_heads_per_chip
    nope, rd, v, rank = cfg.qk_nope_head_dim, cfg.rope_dim, cfg.v_head_dim, cfg.kv_lora_rank
    eps = cfg.rms_norm_eps
    mask = _mask(pos, spec.sliding_window_size)
    E = lambda_expansion(cfg).to(x.dtype)
    X = noise_expansion(cfg).to(x.dtype)
    Wq = latent_q_weight(src, cfg)
    out = torch.zeros(T, cfg.hidden_size, dtype=x.dtype)

    def rope(t):  # [T, h, 64]
        return apply_rope(t[None], cos, sin)[0]

    def attend(q, k, val):
        s = (q @ k.T).masked_fill(~mask, float("-inf"))
        return torch.softmax(s, dim=-1) @ val

    for tp in range(cfg.tp):
        cq_n = _rmsnorm(x @ Wq, eps)
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
            U = torch.stack([attend(Q[:, h], K, n) @ w_uv[h] for h in range(H)], 1)
            u_flat = U.reshape(T, H * v)
            u_sig, u_noise = u_flat[:, : Sg * v], u_flat[:, Sg * v :] @ X
        else:
            Qv = torch.cat([q_nope, q_pe], -1)
            Qh = torch.cat([Qv[:, a:b] for a, b in hf_order_from_virtual(cfg)], 1)
            kvx = n @ W.prefill_kv_expansion_for_chip(src["wkv_b"], src["kv_norm"], cfg, tp)
            blk = kvx.shape[1] // G
            outs = []
            for h in range(H):
                gi = h // cfg.heads_per_group
                k_g = torch.cat([kvx[:, gi * blk : gi * blk + nope], k_pe], -1)
                v_g = kvx[:, gi * blk + nope : (gi + 1) * blk]
                outs.append(attend(Qh[:, h], k_g, v_g)[:, :v])
            u_flat = torch.cat(outs, -1)
            hpg = cfg.heads_per_group
            u_sig = torch.cat([u_flat[:, hpg * gi * v : (hpg * gi + r) * v] for gi in range(G)], -1)
            noise = [u_flat[:, (hpg * gi + r) * v : (hpg * gi + r + 1) * v] for gi in range(G)]
            u_noise = torch.cat([nz for nz in noise for _ in range(r)], -1)
        d = (u_sig - torch.sigmoid(lam @ E) * u_noise) * g
        out += d @ W.wo_for_chip(src["wo"], cfg, tp)
    return out


def emulate_latent(cfg: MotifTTConfig, spec, src: _AttnSource, x: torch.Tensor, positions: torch.Tensor):
    """:meth:`MotifAttention.fill_kv`'s per-chip dataflow: ``[tp, T, 576]`` = ``[rms_norm(c_raw) | rope(kpe)]`` of
    the rows ``x [T, D]`` at the absolute ``positions [T]``."""
    from models.demos.motif3.reference.rope import apply_rope, rope_cos_sin

    cos, sin = rope_cos_sin(inv_freq_for_kind(cfg, spec.rope_kind), positions[None], x.dtype)
    rank, rd = cfg.kv_lora_rank, cfg.rope_dim
    out = []
    for tp in range(cfg.tp):
        kvl = x @ latent_kv_weight_for_chip(src, cfg, tp)
        n = _rmsnorm(kvl[:, :rank], cfg.rms_norm_eps)
        k_pe = apply_rope(kvl[:, rank : rank + rd][None, :, None, :], cos, sin)[0, :, 0, :]
        out.append(torch.cat([n, k_pe], -1))
    return torch.stack(out)


def draft1_attention_module():
    """``tt/attention.py`` as of :data:`DRAFT1_COMMIT` (draft 1), loaded from git as :data:`DRAFT1_MODULE`. Its
    relative imports resolve to the current shared infra. Skips when git or the commit is unavailable, e.g. in a
    packaged copy."""
    if DRAFT1_MODULE in sys.modules:
        return sys.modules[DRAFT1_MODULE]
    rel = "models/demos/motif3/tt/attention.py"
    try:
        res = subprocess.run(
            ["git", "-C", str(METAL_ROOT), "show", f"{DRAFT1_COMMIT}:{rel}"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as e:
        pytest.skip(f"git unavailable ({e}): cannot load the draft-1 attention module")
    if res.returncode != 0:
        pytest.skip(f"draft-1 attention module not in git: {res.stderr.strip()[:300]}")
    spec = importlib.util.spec_from_loader(DRAFT1_MODULE, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "models.demos.motif3.tt"
    mod.__file__ = f"<git {DRAFT1_COMMIT}>/{rel}"
    exec(compile(res.stdout, mod.__file__, "exec"), mod.__dict__)
    sys.modules[DRAFT1_MODULE] = mod
    return mod


def make_mtp_attention_inputs(path: Path = MTP_INPUTS, prompt: str = MTP_PROMPT) -> Path:
    """Host only. Builds the MTP attention input ``input_layernorm(input_proj([hn_p | embed_norm(embed(t_{p+1}))]))``
    for positions ``0 .. S-2`` of a C2 prompt, in bf16 like the reference ``MotifMTP`` (``reference/modules.py``).
    ``hn_p`` is the C2 golden's post-final-norm hidden state; the embedding and the MTP tensors come from the
    checkpoint (shards 1, 104). Stores ``{"x": [S-1, 4096] bf16, "ids", ...}``; ~7 MB."""
    from safetensors import safe_open

    from models.demos.motif3.reference.config import MotifArgs

    prompts = json.loads((C2_DIR / "prompts.json").read_text())["prompts"]
    entry = next((p for p in prompts if p["name"] == prompt), None)
    if entry is None:
        raise KeyError(f"prompt {prompt!r} not in {C2_DIR / 'prompts.json'}")
    ids = torch.tensor(entry["ids"], dtype=torch.long)
    with safe_open(str(C2_DIR / "final" / "hidden_after_layer_52.safetensors"), "pt") as f:
        hn = f.get_tensor(f"{prompt}.final_hidden")[0]  # [S, 4096], post final norm
    if hn.shape[0] != ids.numel():
        raise ValueError(f"golden hidden rows {hn.shape[0]} != prompt tokens {ids.numel()}")
    loader = W.HFWeightLoader()
    P = "model.mtp_layers.0"
    eps = MotifArgs.from_hf_config(HF_META).rms_norm_eps
    bf = torch.bfloat16

    def rmsnorm(x, w):  # reference RMSNorm in bf16: fp32 statistics, cast back, then weight * x (bf16)
        xf = x.float()
        return w.to(bf) * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(bf)

    emb = torch.cat([loader.get_rows("model.embed_tokens.weight", int(t), int(t) + 1) for t in ids[1:]]).to(bf)
    e = rmsnorm(emb, loader.get(f"{P}.embed_norm.weight"))
    h_in = torch.cat([hn[:-1].to(bf), e], dim=-1)  # concat order [hidden, embed] (reference MotifMTP)
    h = torch.nn.functional.linear(h_in.float(), loader.get(f"{P}.input_proj.weight").float()).to(bf)
    x = rmsnorm(h, loader.get(f"{P}.input_layernorm.weight")).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": MTP_INPUTS_FORMAT, "prompt": prompt, "ids": ids, "x": x}, path)
    return path


def load_mtp_attention_inputs() -> torch.Tensor:
    """``[S-1, 4096]`` fp32 (bf16 values) MTP attention inputs (:func:`make_mtp_attention_inputs`, built on first
    use). Skips when the C2 golden or the checkpoint tensors are missing."""
    if MTP_INPUTS.is_file():
        d = torch.load(MTP_INPUTS, weights_only=True)
        if d.get("format") == MTP_INPUTS_FORMAT:
            return d["x"].float()
    try:
        make_mtp_attention_inputs()
    except (FileNotFoundError, KeyError, W.MissingWeightError) as e:
        pytest.skip(f"cannot build the MTP attention inputs: {e}")
    return torch.load(MTP_INPUTS, weights_only=True)["x"].float()


# ======================================================================================================================
# host tests
# ======================================================================================================================
def test_cpu_layer_identity_defaults():
    """Default spec / weight prefix per layer index: decoder layers are draft 1; the MTP layer gets
    ``cfg.mtp_layer_spec()``, which the reference ``GDLAttention(args, 53, swa=True)`` agrees with (scale, RoPE table,
    window), and ``model.mtp_layers.0.self_attn``."""
    from models.demos.motif3.reference.rope import inv_freq_for_layer as ref_inv_freq

    cfg = host_cfg()
    assert (cfg.num_hidden_layers, cfg.mtp_layer_idx, cfg.num_nextn_predict_layers) == (53, 53, 1)
    for l in range(cfg.num_layers):
        assert canonical_attn_spec(cfg, l) is cfg.layer(l)
        assert attn_weight_prefix(cfg, l) == f"model.layers.{l}.self_attn" == W.hf_name(l, "self_attn")
        spec, prefix = resolve_attn_layer(cfg, l)
        assert spec is cfg.layer(l) and prefix == f"model.layers.{l}.self_attn"
    mtp = cfg.mtp_layer_spec()
    assert canonical_attn_spec(cfg, 53) == mtp
    assert (mtp.idx, mtp.is_global, mtp.is_moe, mtp.window, mtp.rope_kind, mtp.attn_kind) == (
        53, False, False, 129, "plain", "swa")  # fmt: skip
    assert mtp.softmax_scale == cfg.head_dim**-0.5 and abs(mtp.softmax_scale - 0.07216878) < 1e-8
    assert attn_weight_prefix(cfg, 53) == MTP_ATTN_PREFIX == "model.mtp_layers.0.self_attn"
    assert resolve_attn_layer(cfg, 53) == (mtp, MTP_ATTN_PREFIX)
    # the MTP attention has exactly layer 1's SWA constants (only the index and the weights differ)
    assert dataclasses.replace(cfg.layer(1), idx=53) == mtp
    args = ref_args()
    assert args.softmax_scale(53, swa=True) == mtp.softmax_scale and not args.uses_yarn(53, swa=True)
    assert args.effective_sliding_window == mtp.window
    assert torch.equal(inv_freq_for_kind(cfg, mtp.rope_kind), ref_inv_freq(args, 53, swa=True))
    for bad in (-1, 54, 100):
        for fn in (canonical_attn_spec, attn_weight_prefix, resolve_attn_layer):
            with pytest.raises(ValueError):
                fn(cfg, bad)
    # truncated configs: unbuilt layers have no spec but keep their HF prefix; the MTP layer stays 53
    t = host_cfg(num_layers=4)
    assert canonical_attn_spec(t, 3) == cfg.layer(3) and t.mtp_layer_idx == 53 and canonical_attn_spec(t, 53) == mtp
    assert attn_weight_prefix(t, 10) == "model.layers.10.self_attn"
    with pytest.raises(ValueError, match="pass spec="):
        canonical_attn_spec(t, 10)
    # a checkpoint without an MTP layer has no layer 53
    n0 = host_cfg(num_nextn_predict_layers=0)
    for fn in (canonical_attn_spec, attn_weight_prefix):
        with pytest.raises(ValueError):
            fn(n0, 53)


def test_cpu_layer_identity_guards():
    """``resolve_attn_layer`` (runs before anything touches the device): explicit canonical values pass with the
    cache on (what ``tt/mtp.py`` passes). An index mismatch, a non-canonical spec or prefix with the cache on, a wrong
    spec type and malformed prefixes raise. Non-canonical values pass with ``cache=False``."""
    cfg = host_cfg()
    mtp = cfg.mtp_layer_spec()
    assert resolve_attn_layer(cfg, 53, spec=mtp, weight_prefix=MTP_ATTN_PREFIX, cache=True) == (mtp, MTP_ATTN_PREFIX)
    assert resolve_attn_layer(cfg, 53, weight_prefix=f" {MTP_ATTN_PREFIX}. ")[1] == MTP_ATTN_PREFIX
    assert resolve_attn_layer(cfg, 1, spec=cfg.layer(1), weight_prefix="model.layers.1.self_attn")[0] is cfg.layer(1)
    with pytest.raises(ValueError, match="spec.idx"):
        resolve_attn_layer(cfg, 1, spec=mtp, cache=False)
    odd = dataclasses.replace(cfg.layer(1), softmax_scale=0.1)
    with pytest.raises(ValueError, match="cache=False"):
        resolve_attn_layer(cfg, 1, spec=odd)
    assert resolve_attn_layer(cfg, 1, spec=odd, cache=False) == (odd, "model.layers.1.self_attn")
    with pytest.raises(ValueError, match="cache=False"):  # MTP weights under L05/attn.v2.*
        resolve_attn_layer(cfg, 5, weight_prefix=MTP_ATTN_PREFIX)
    assert resolve_attn_layer(cfg, 5, weight_prefix=MTP_ATTN_PREFIX, cache=False) == (cfg.layer(5), MTP_ATTN_PREFIX)
    with pytest.raises(ValueError, match="cache=False"):  # global constants (YaRN, scale 0.1447) under L53
        resolve_attn_layer(cfg, 53, spec=dataclasses.replace(cfg.layer(0), idx=53))
    with pytest.raises(ValueError, match="cache=False"):  # layer 1's weights under L53
        resolve_attn_layer(cfg, 53, weight_prefix="model.layers.1.self_attn")
    t = host_cfg(num_layers=4)  # an index the config does not build: explicit spec, never the cache
    with pytest.raises(ValueError):
        resolve_attn_layer(t, 10)
    with pytest.raises(ValueError, match="cache=False"):
        resolve_attn_layer(t, 10, spec=cfg.layer(10))
    assert resolve_attn_layer(t, 10, spec=cfg.layer(10), cache=False) == (cfg.layer(10), "model.layers.10.self_attn")
    with pytest.raises(TypeError):
        resolve_attn_layer(cfg, 1, spec={"idx": 1}, cache=False)
    for bad in ("", ".", "  ", "model..self_attn", ".model.layers.1.self_attn"):
        with pytest.raises(ValueError, match="weight_prefix"):
            resolve_attn_layer(cfg, 1, weight_prefix=bad, cache=False)


def test_cpu_attn_source_names():
    """``_AttnSource`` asks the source for ``{prefix}.{name}.weight`` once per tensor. The default prefix gives the
    draft-1 names; every MTP attention tensor exists in the checkpoint index under :data:`MTP_ATTN_PREFIX`."""

    class Recorder:
        def __init__(self):
            self.calls: List[str] = []

        def get(self, name, dtype=None):
            self.calls.append(name)
            return torch.zeros(1)

    rec = Recorder()
    s = _AttnSource(rec, 7)
    for k in _AttnSource.NAMES:
        s[k], s[k]
    assert rec.calls == [f"model.layers.7.self_attn.{k}.weight" for k in _AttnSource.NAMES]
    assert [s.name(k) for k in _AttnSource.NAMES] == [A._attn_name(7, f"{k}.weight") for k in _AttnSource.NAMES]
    assert tuple(_AttnSource.NAMES) == tuple(ATTN_KEYS)
    rec2 = Recorder()
    m = _AttnSource(rec2, 53, MTP_ATTN_PREFIX)
    m["wq_a"], m["wo"]
    assert rec2.calls == [f"{MTP_ATTN_PREFIX}.wq_a.weight", f"{MTP_ATTN_PREFIX}.wo.weight"]
    try:
        weight_map = W.HFWeightLoader().weight_map
    except FileNotFoundError:
        pytest.skip("no checkpoint index")
    names = [m.name(k) for k in _AttnSource.NAMES]
    assert all(n in weight_map for n in names), [n for n in names if n not in weight_map]
    assert {weight_map[n] for n in names} == {"model-00104-of-00155.safetensors"}


@pytest.mark.parametrize("weights_kind", ["random", "real"])
def test_cpu_mtp_dataflow_matches_reference_fp64(weights_kind):
    """The module's weight transforms + per-chip dataflow (decode and prefill forms) with the MTP spec and weight
    prefix reproduce the reference ``GDLAttention(args, 53, swa=True)`` in fp64 at T = 160 > window (scale folding
    0.0722, plain RoPE theta 1e4, window 129, head order, absorb / expansion, lambda / noise, wo row split). ``real`` =
    the checkpoint's MTP attention tensors (shard 104)."""
    cfg = host_cfg()
    spec = cfg.mtp_layer_spec()
    args = ref_args(q_path_fp32=False)  # keep the q path in the module dtype (fp64)
    t = real_mtp_tensors() if weights_kind == "real" else random_attn_tensors(args, seed=153)
    t64 = {k: v.double() for k, v in t.items()}
    ref = ref_attention_spec(args, 53, t64, torch.float64, swa=True)
    src = _AttnSource(source_for(t64, MTP_ATTN_PREFIX), 53, MTP_ATTN_PREFIX)
    T = 160
    x = torch.randn(T, cfg.hidden_size, generator=torch.Generator().manual_seed(53), dtype=torch.float64)
    want = ref(x[None], torch.arange(T)[None])[0]
    for mode in ("decode", "prefill"):
        s = stats(want, emulate_attention(cfg, spec, src, x, mode))
        log(f"cpu fp64 MTP L53 {weights_kind} {mode}: {fmt(s)}")
        assert s["rel_fro"] < 2e-6, f"MTP {weights_kind} {mode}: {fmt(s)}"  # fp32 RoPE / norms in the reference


@pytest.mark.parametrize("layer", [0, 1, 53], ids=["global_L0", "swa_L1", "mtp_L53"])
def test_cpu_fill_kv_latent_rows_fp64(layer):
    """``fill_kv``'s dataflow at absolute positions that do not start at 0 (resumed chunks, packed rows) equals the
    reference cache entries (``c_kv / gamma`` and roped ``k_pe``) in fp64. It is identical on the 8 TP chips (the
    latent columns of ``Wkv_lat`` are replicated), and a row's latent depends only on its own input row and position:
    the rows of a later chunk filled alone are the rows of a single-shot fill."""
    from models.demos.motif3.reference.rope import rope_cos_sin

    cfg = host_cfg()
    args = ref_args(q_path_fp32=False)
    spec, prefix = resolve_attn_layer(cfg, layer, cache=False)
    t64 = {k: v.double() for k, v in random_attn_tensors(args, seed=200 + layer).items()}
    ref = ref_attention_spec(args, layer, t64, torch.float64, swa=True if layer == 53 else None)
    src = _AttnSource(source_for(t64, prefix), layer, prefix)
    g = torch.Generator().manual_seed(300 + layer)
    positions = torch.cat(
        [torch.arange(6976, 6976 + 64), torch.tensor([0, 1, 127, 128, 129, 4095, 4096, 32767]),
         torch.randint(0, 32768, (24,), generator=g)]  # fmt: skip
    )
    x = torch.randn(positions.numel(), cfg.hidden_size, generator=g, dtype=torch.float64)
    lat = emulate_latent(cfg, spec, src, x, positions)
    assert all(torch.equal(lat[0], lat[tp]) for tp in range(cfg.tp)), "latent differs between TP chips"
    cos, sin = rope_cos_sin(ref.inv_freq(), positions[None], torch.float64)
    c, kpe = ref.project_kv(x[None], cos, sin)
    gam = t64["kv_norm"]
    rank = cfg.kv_lora_rank
    s_n, s_k = stats(c[0] / gam, lat[0, :, :rank]), stats(kpe[0], lat[0, :, rank:])
    log(f"cpu fp64 fill_kv L{layer} ({spec.rope_kind}): n {fmt(s_n)}; k_pe {fmt(s_k)}")
    assert s_n["rel_fro"] < 2e-6 and s_k["rel_fro"] < 2e-6
    sub = slice(0, 64)
    lat_sub = emulate_latent(cfg, spec, src, x[sub], positions[sub])
    assert torch.allclose(lat_sub[0], lat[0, sub], rtol=0, atol=1e-12)


def test_cpu_chunk_rot_rows():
    """Host side of the offset-RoPE gather: chunk table rows from ``prefill_plan.rope_positions`` are ``int32 [1, C]``.
    Real rows sit at their positions and padded rows clamp to the last table row. The rows a gather returns are the
    host table rows of those positions (exact per position). Bad inputs raise."""
    cfg = host_cfg()
    P = cfg.max_model_len
    tables = {k: cos_sin_table(inv_freq_for_kind(cfg, k), torch.arange(P)) for k in ("yarn", "plain")}
    seen = set()
    for s, e in [(0, 1000), (640, 1000), (64, 900), (1348, 3000), (1472, 1500), (30000, 32768), (0, 16736)]:
        plan = cfg.plan_prefill_row(s, e)
        for ch in plan.chunks:
            idx = chunk_rot_rows(PP.rope_positions(ch, P), P)
            assert idx.dtype == torch.int32 and tuple(idx.shape) == (1, ch.bucket)
            real = ch.end - ch.start
            assert torch.equal(idx[0, :real], torch.arange(ch.start, ch.end, dtype=torch.int32))
            assert torch.equal(idx[0, real:], torch.arange(ch.end, ch.start + ch.bucket).clamp(max=P - 1).int())
            for kind, (c_all, s_all) in tables.items():
                c_pos, s_pos = cos_sin_table(inv_freq_for_kind(cfg, kind), idx[0].long())
                assert torch.equal(c_all[idx[0].long()], c_pos) and torch.equal(s_all[idx[0].long()], s_pos)
            seen.add((ch.path, ch.start + ch.bucket > P))
    assert ("sp1", True) in seen and ("sp0", False) in seen  # a clamped sp1 chunk and a cold chunk were covered
    packed = torch.arange(256).repeat(32)  # 32 requests' rows concatenated (one packed fill)
    assert torch.equal(chunk_rot_rows(packed, P)[0], packed.int())
    for bad in ([], list(range(31)), [-1] + list(range(31)), list(range(P - 8, P + 24))):
        with pytest.raises(ValueError):
            chunk_rot_rows(bad, P)
    with pytest.raises(TypeError):
        chunk_rot_rows(torch.zeros(32), P)


def test_cpu_draft1_host_transforms_unchanged():
    """Every host tensor the constructor uploads comes out bitwise equal from the draft-1 module (git) and the current
    one, with the same cache names / version. So for decoder layers the serving TT cache (``L00`` ... ``L52``) stays
    valid, and the MTP layer uses the same transforms."""
    d1 = draft1_attention_module()
    cfg = host_cfg()
    args = ref_args()
    assert d1.ATTN_CACHE_VERSION == A.ATTN_CACHE_VERSION and d1._CACHE == A._CACHE == "attn.v2"
    assert d1.virtual_head_order(cfg) == virtual_head_order(cfg)
    assert d1.hf_order_from_virtual(cfg) == hf_order_from_virtual(cfg)
    assert torch.equal(d1.lambda_expansion(cfg), lambda_expansion(cfg))
    assert torch.equal(d1.noise_expansion(cfg), noise_expansion(cfg))
    for layer in (0, 1, 53):
        spec, prefix = resolve_attn_layer(cfg, layer)
        t = random_attn_tensors(args, seed=400 + layer)
        new = _AttnSource(source_for(t, prefix), layer, prefix)
        # draft 1 can only name decoder layers: feed it the same tensors under its own (model.layers.<l>) names
        old = d1._AttnSource(source_for(t, W.hf_name(layer, "self_attn")), layer)
        assert torch.equal(latent_q_weight(new, cfg), d1.latent_q_weight(old, cfg))
        for tp in range(cfg.tp):
            for fn in ("latent_kv_weight_for_chip", "wq_b_gate_for_chip", "w_uk_virtual_for_chip",
                       "w_uv_virtual_for_chip"):  # fmt: skip
                assert torch.equal(getattr(A, fn)(new, cfg, tp), getattr(d1, fn)(old, cfg, tp)), (layer, fn, tp)
            assert torch.equal(
                wq_b_virtual_for_chip(new, cfg, tp, spec.softmax_scale),
                d1.wq_b_virtual_for_chip(old, cfg, tp, spec.softmax_scale),
            )
        if layer < cfg.num_layers:
            assert spec is cfg.layer(layer)  # draft 1: self.spec = cfg.layer(layer_idx)
    log("draft-1 host transforms: bitwise equal for L0, L1 and (same transforms) L53")


class _FT:
    """Fake device tensor for :class:`_FakeTTNN`: shape, dtype and lineage (the inputs / weights it was computed
    from), so a recorded op describes its operands exactly."""

    def __init__(self, shape: Sequence[int], dtype: str = "bf16", src=()):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.src = frozenset(src)
        self.data: Optional[torch.Tensor] = None  # host values (uploads / copies of :class:`_FakeTTNN` only)
        self.on_device = True

    def desc(self):
        return ("T", self.shape, self.dtype, tuple(sorted(self.src)))


def _desc(v):
    if isinstance(v, _FT):
        return v.desc()
    if isinstance(v, (list, tuple)):
        return tuple(_desc(i) for i in v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return ("obj", type(v).__name__, getattr(v, "q_chunk_size", None), getattr(v, "k_chunk_size", None))


class _FakeTTNN:
    """Records every ttnn call the attention methods make (op, operand descriptors, kwargs) and returns fake tensors
    of the right shape / dtype / lineage. Patched in as the module-global ``ttnn`` of ``tt/attention.py`` (and of the
    draft-1 module) to compare op sequences on the host. ``made`` lists every tensor an op returned and ``freed``
    every deallocation (by identity), for the leak checks."""

    DRAM_MEMORY_CONFIG = "DRAM"
    TILE_LAYOUT, ROW_MAJOR_LAYOUT = "TILE", "RM"
    float32, bfloat16, bfloat8_b, int32, uint32 = "fp32", "bf16", "bfp8", "i32", "u32"
    Tensor = _FT

    def __init__(self):
        self.calls: List[Any] = []
        self.made: List[_FT] = []
        self.freed: List[_FT] = []
        self.experimental = SimpleNamespace(
            nlp_create_q_heads_split=self._split,
            rotary_embedding_hf=lambda x, cos, sin, **kw: self._out("rotary_embedding_hf", (x, cos, sin), kw, x.shape),
            nlp_concat_heads=self._concat_heads,
            paged_fill_cache=lambda *a, **kw: self._rec("paged_fill_cache", a, kw),
            paged_update_cache=lambda *a, **kw: self._rec("paged_update_cache", a, kw),
        )
        self.transformer = SimpleNamespace(
            scaled_dot_product_attention=lambda q, k, v, **kw: self._out(
                "scaled_dot_product_attention", (q, k, v), kw, q.shape[:-1] + (v.shape[-1],)
            ),
            chunked_scaled_dot_product_attention=lambda q, k, v, pt, **kw: self._out(
                "chunked_scaled_dot_product_attention", (q, k, v, pt), kw, q.shape
            ),
            paged_flash_multi_latent_attention_decode=lambda q, c, m, **kw: self._out(
                "paged_flash_multi_latent_attention_decode", (q, c, m), kw, q.shape[:-1] + (kw["head_dim_v"],)
            ),
        )

    def new(self, shape, dtype, src) -> _FT:
        t = _FT(shape, dtype, src)
        self.made.append(t)
        return t

    def _rec(self, op, args, kw):
        self.calls.append((op, _desc(tuple(args)), tuple(sorted((k, _desc(v)) for k, v in kw.items()))))

    def _out(self, op, args, kw, shape, dtype=None):
        self._rec(op, args, kw)
        ts = [a for a in args if isinstance(a, _FT)]
        return self.new(shape, dtype or ts[0].dtype, frozenset().union(*[t.src for t in ts]))

    def _split(self, x, num_heads, split_head_dim, **kw):
        self._rec("nlp_create_q_heads_split", (x,), dict(kw, num_heads=num_heads, split_head_dim=split_head_dim))
        T, Wd = x.shape[-2], x.shape[-1] // num_heads
        h = 1 if num_heads == 1 else num_heads
        return self.new((1, h, T, split_head_dim), x.dtype, x.src), self.new(
            (1, h, T, Wd - split_head_dim), x.dtype, x.src
        )

    def _concat_heads(self, u, **kw):
        return self._out("nlp_concat_heads", (u,), kw, (1, 1, u.shape[2], u.shape[1] * u.shape[3]))

    def linear(self, x, w, **kw):
        return self._out("linear", (x, w), kw, x.shape[:-1] + (w.shape[-1],), kw.get("dtype"))

    def matmul(self, a, b, **kw):
        return self._out("matmul", (a, b), kw, a.shape[:-1] + (b.shape[-1],))

    def rms_norm(self, x, **kw):
        return self._out("rms_norm", (x,), kw, x.shape)

    def concat(self, ts, dim, **kw):
        shape = list(ts[0].shape)
        shape[dim] = sum(t.shape[dim] for t in ts)
        return self._out("concat", tuple(ts), {"dim": dim, **kw}, shape)

    def slice(self, t, start, end, *a, **kw):
        if isinstance(start, _FT):  # tensor-args slice: output = input / num_devices along slice_dim
            shape = list(t.shape)
            shape[kw["slice_dim"]] //= kw["num_devices"]
            return self._out("slice", (t, start, end), kw, shape)
        return self._out("slice", (t, tuple(start), tuple(end)), kw, [e - s for s, e in zip(start, end)])

    def repeat(self, t, reps):
        return self._out("repeat", (t, tuple(reps)), {}, [s * r for s, r in zip(t.shape, reps)])

    def Shape(self, dims):
        return tuple(dims)

    def transpose(self, t, d1, d2, **kw):
        shape = list(t.shape)
        shape[d1], shape[d2] = shape[d2], shape[d1]
        return self._out("transpose", (t, d1, d2), kw, shape)

    def typecast(self, t, dtype):
        return self._out("typecast", (t, dtype), {}, t.shape, dtype)

    def addcmul(self, a, b, c, **kw):
        return self._out("addcmul", (a, b, c), kw, a.shape)

    def multiply(self, a, b, **kw):
        return self._out("multiply", (a, b), kw, a.shape)

    def where(self, m, a, b, **kw):
        return self._out("where", (m, a, b), kw, a.shape)

    def deallocate(self, t, *a):
        self.freed.append(t)
        self._rec("deallocate", (t,), {})

    # ---- host <-> device copies (PrefillChunkInputs.upload / write); tensors carry their values in .data ----
    def ReplicateTensorToMesh(self, mesh_device):
        return ("replicate",)

    def from_torch(self, t, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None):
        self._rec("from_torch", (), {"dtype": dtype, "layout": layout, "device": device is not None})
        out = (
            self.new(tuple(t.shape), dtype, {"upload"}) if device is not None else _FT(tuple(t.shape), dtype, {"host"})
        )
        out.data, out.on_device = t.clone(), device is not None
        return out

    def copy_host_to_device_tensor(self, src, dst, *a, **kw):
        assert not src.on_device and dst.on_device, "copy_host_to_device_tensor: host -> device only"
        assert (src.shape, src.dtype) == (dst.shape, dst.dtype), (src.shape, src.dtype, dst.shape, dst.dtype)
        dst.data = src.data.clone()
        self._rec("copy_host_to_device_tensor", (src, dst), {})

    def leaks(self, keep=()) -> Tuple[List[_FT], List[_FT]]:
        """``(never freed, freed twice or not made by an op)`` among the op outputs, ``keep`` (returned / tapped
        tensors) excluded."""
        made = {id(t): t for t in self.made}
        kept = {id(t) for t in keep}
        counts: Dict[int, int] = {}
        for t in self.freed:
            counts[id(t)] = counts.get(id(t), 0) + 1
        never = [t for i, t in made.items() if i not in kept and counts.get(i, 0) == 0]
        bad = [t for t in self.freed if id(t) not in made or counts[id(t)] > 1 or id(t) in kept]
        return never, bad


def _fake_attention(mod, cfg: MotifTTConfig, spec, fake: _FakeTTNN, *, kind_tables: bool = True):
    """A ``mod.MotifAttention`` built without a device: every attribute the forward methods read, with fake
    weights of the per-chip shapes and sentinel configs (identical for the draft-1 and the current module)."""
    a = object.__new__(mod.MotifAttention)
    H, hd, Sg, v = cfg.q_heads_per_chip, cfg.head_dim, cfg.signal_heads_per_chip, cfg.v_head_dim
    D, Q, R, rd = cfg.hidden_size, cfg.q_lora_rank, cfg.kv_lora_rank, cfg.rope_dim

    def w(name, *shape):
        return _FT(shape, "bf16", {name})

    a.cfg, a.spec, a.layer_idx = cfg, spec, spec.idx
    a.H, a.G, a.r, a.Sg = H, cfg.kv_groups_per_chip, cfg.grouped_ratio, Sg
    a.nope, a.rope_dim, a.vdim, a.rank, a.latent_dim = cfg.qk_nope_head_dim, rd, v, R, cfg.kv_latent_dim
    a.window, a.scale, a.kind, a.lanes = spec.sliding_window_size, float(spec.softmax_scale), spec.rope_kind, 8
    a.rope_mode, a.sdpa_prefill_fp32_acc, a.dtype = "hf", False, "bf16"
    for role in ("latent", "heads", "norm", "sdpa_decode", "sdpa_prefill", "sdpa_prefill_fp32", "rope"):
        setattr(a, f"ckc_{role}", f"ckc:{role}")
    a.ckc_sdpa_sp1_global = "ckc:sdpa_prefill_fp32"  # the sp1 global role (G9), as the constructor sets it
    a.decode_pc, a.update_mc = "pc:flash_mla", "mc:update"
    a.decode_pcs = {k: f"pc:{k}" for k in ("q_lat", "kv_lat", "wq_b", "gate", "w_uk", "w_uv")}
    a.decode_pcs["wo"] = None
    a.w_q_lat, a.w_kv_lat = w("w_q_lat", D, Q), w("w_kv_lat", D, R + rd + 64)
    a.w_q_b, a.w_gate = w("w_q_b", Q, H * hd), w("w_gate", Q, Sg * v)
    a.w_uk, a.w_uv = w("w_uk", 1, H, cfg.qk_nope_head_dim, R), w("w_uv", 1, H, R, v)
    a.w_kv_expand, a.w_o = w("w_kv_expand", R, cfg.kv_groups_per_chip * 320), w("w_o", Sg * v, D)
    a.lam_expand, a.noise_expand = w("lam_expand", 64, Sg * v), w("noise_expand", cfg.kv_groups_per_chip * v, Sg * v)
    tables = {k: (_FT((1, 1, 32768, rd), "bf16", {f"cos_{k}"}), _FT((1, 1, 32768, rd), "bf16", {f"sin_{k}"}))
              for k in ("yarn", "plain")}  # fmt: skip
    a.rope = SimpleNamespace(
        prefill_cos_sin=lambda kind, S: (_FT((1, 1, S, rd), "bf16", {f"cos_{kind}"}),
                                         _FT((1, 1, S, rd), "bf16", {f"sin_{kind}"})),  # fmt: skip
        tables=tables,
    )

    def ar_tp(t):
        fake.calls.append(("ar_tp", _desc((t,)), ()))
        return fake.new(t.shape, t.dtype, t.src)

    a.ccl = SimpleNamespace(ar_tp=ar_tp)
    return a


def _subsequence(sub: Sequence[Any], seq: Sequence[Any]) -> bool:
    it = iter(seq)
    return all(any(s == x for x in it) for s in sub)


def test_cpu_op_sequence_unchanged_vs_draft1(monkeypatch):
    """Host proof of "draft 1 bitwise": with a recording fake ``ttnn``, the draft-1 module (git) and the current one
    issue the identical op sequence -- every op, operand shape / dtype / lineage, kwarg (program and compute configs)
    and deallocation -- for ``forward_prefill`` (S 128 / 1024, with and without the cache fill, a wider page table)
    and ``forward_decode`` on global and SWA layers. And ``fill_kv``'s ops are exactly the kv-path ops of
    ``forward_prefill`` (an ordered subsequence with identical operands), so its rows are those of a prefill."""
    d1 = draft1_attention_module()
    cfg = host_cfg()
    for layer in (0, 1):
        spec = cfg.layer(layer)
        runs = {}
        for name, mod in (("draft1", d1), ("new", A)):
            fake = _FakeTTNN()
            monkeypatch.setattr(mod, "ttnn", fake)
            attn = _fake_attention(mod, cfg, spec, fake)
            cache = _FT((20, 1, 64, 576), "bfp8", {"cache"})
            for S in (128, 1024):
                x = _FT((1, 1, S, 4096), "bf16", {"x"})
                attn.forward_prefill(x, page_table=_FT((1, S // 64), "i32", {"pt"}), kv_cache=cache)
                attn.forward_prefill(x)
                attn.forward_prefill(x, page_table=_FT((1, S // 64 + 3), "i32", {"pt"}), kv_cache=cache)
            rot = {k: (_FT((1, 1, 32, 64), "bf16", {f"cos_{k}"}), _FT((1, 1, 32, 64), "bf16", {f"sin_{k}"}))
                   for k in ("yarn", "plain")}  # fmt: skip
            attn.forward_decode(
                _FT((1, 1, 8, 4096), "bf16", {"x"}), rot=rot, cur_pos=_FT((8,), "i32", {"cur"}),
                page_table=_FT((8, 5), "i32", {"pt"}), kv_cache=cache, active=_FT((1, 1, 8, 1024), "bf16", {"act"}),
            )  # fmt: skip
            runs[name] = list(fake.calls)
        assert len(runs["new"]) > 100
        assert runs["draft1"] == runs["new"], next(
            (i, a, b) for i, (a, b) in enumerate(zip(runs["draft1"], runs["new"])) if a != b
        )
        # negative control: the recording sees configs (another norm compute config changes the sequence)
        fake = _FakeTTNN()
        monkeypatch.setattr(A, "ttnn", fake)
        bad = _fake_attention(A, cfg, spec, fake)
        bad.ckc_norm = "ckc:other"
        bad.forward_prefill(_FT((1, 1, 128, 4096), "bf16", {"x"}), page_table=_FT((1, 2), "i32", {"pt"}),
                            kv_cache=_FT((20, 1, 64, 576), "bfp8", {"cache"}))  # fmt: skip
        assert fake.calls != runs["new"][: len(fake.calls)]
        # fill_kv == the kv path of forward_prefill (same ops, same operands, same order)
        fake = _FakeTTNN()
        monkeypatch.setattr(A, "ttnn", fake)
        attn = _fake_attention(A, cfg, spec, fake)
        x, pt = _FT((1, 1, 1024, 4096), "bf16", {"x"}), _FT((1, 16), "i32", {"pt"})
        cache = _FT((20, 1, 64, 576), "bfp8", {"cache"})
        attn.forward_prefill(x, page_table=pt, kv_cache=cache)
        pf = [c for c in fake.calls if c[0] != "deallocate"]
        fake.calls.clear()
        attn.fill_kv(x, fill_pt=pt, kv_cache=cache)
        fk = [c for c in fake.calls if c[0] != "deallocate"]
        n_dealloc = sum(c[0] == "deallocate" for c in fake.calls)
        assert [c[0] for c in fk] == ["linear", "nlp_create_q_heads_split", "nlp_create_q_heads_split", "rms_norm",
                                      "rotary_embedding_hf", "concat", "typecast", "paged_fill_cache"], fk  # fmt: skip
        assert _subsequence(fk, pf), (fk, pf)
        assert n_dealloc == 9  # kvl, rest, c_raw, lam, kpe, the bf16 row, the cast row, n, k_pe: nothing leaks
        log(f"op sequences L{layer}: draft-1 == current ({len(runs['new'])} calls); fill_kv = kv path of the prefill")


def test_cpu_mtp_attention_inputs():
    """The realistic MTP attention inputs (built on first use): finite, one row per position ``0 .. S-2``, and they
    are distinct per row."""
    x = load_mtp_attention_inputs()
    assert x.shape[1] == 4096 and x.shape[0] >= 512, tuple(x.shape)
    assert bool(torch.isfinite(x).all()) and torch.equal(x, x.bfloat16().float())
    assert float(x.std(-1).min()) > 0 and torch.unique(x[:64], dim=0).shape[0] == 64
    log(f"MTP attention inputs {tuple(x.shape)} ({MTP_PROMPT}): rms {float(x.pow(2).mean().sqrt()):.4f}")


# ======================================================================================================================
# work package 2b: resumed (sp1) prefill -- schedules, fp64 emulation of the dataflow, host tests
# ======================================================================================================================
@dataclass(frozen=True)
class Schedule:
    """One request prefilled in vLLM-scheduled steps ``(start, end)`` (``PrefillRequest.start`` / ``end``)."""

    name: str
    steps: Tuple[Tuple[int, int], ...]
    cached: int = 0  # positions [0, cached) are in the cache before the first step (a prefix-cache hit)
    align: Optional[int] = None  # plan with this resume alignment instead of cfg.prefill_resume_alignment

    @property
    def end(self) -> int:
        return self.steps[-1][1]


def make_schedules(S: int, unaligned: int, hit_a128: int) -> List[Schedule]:
    """The features design §5.2 schedules of an ``S``-token prompt: single shot (draft 1); two halves; half cached +
    half; ``S / 128`` chunks of 128; a hit planned with A = 128, so ``c0 < w0`` (the recomputed rows ``[c0, w0)`` lie in
    a shared block the fill must skip); an unaligned vLLM chunk end; a one-block hit (``c0 = 0``: sp0 with block 0
    shared)."""
    h = S // 2
    return [
        Schedule("single", ((0, S),)),
        Schedule("2x_half", ((0, h), (h, S))),
        Schedule("half_cached+half", ((h, S),), cached=h),
        Schedule(f"{S // 128}x128", tuple((i, i + 128) for i in range(0, S, 128))),
        Schedule(f"hit{hit_a128}_A128", ((hit_a128, S),), cached=hit_a128, align=128),
        Schedule(f"unaligned{unaligned}", ((0, unaligned), (unaligned, S))),
        Schedule("hit64", ((64, S),), cached=64),
    ]


DEVICE_S = 4096
DEVICE_SCHEDULES = make_schedules(DEVICE_S, unaligned=1348, hit_a128=960)  # design §5.2: hit 960 -> c0 896
HOST_S = 1024
HOST_SCHEDULES = make_schedules(HOST_S, unaligned=337, hit_a128=960)


def plan_step(cfg: MotifTTConfig, s: int, e: int, align: Optional[int] = None) -> "PP.RowPlan":
    """``cfg.plan_prefill_row(s, e)``, or the same planner with another resume alignment."""
    if align is None:
        return cfg.plan_prefill_row(s, e)
    cost = PP.prefill_cost_model(cfg.prefill_cost_table, sp1_s_per_row_key=cfg.prefill_sp1_s_per_row_key)
    return PP.plan_prefill_row(
        s, e, block_size=cfg.kv_block_size, align=align, buckets=cfg.prefill_span_buckets,
        span_cap=cfg.max_prefill_span, swa_tail=cfg.prefill_swa_tail, cost=cost,
    )  # fmt: skip


def step_page_table(pt_full: torch.Tensor, end: int, block: int, width: int = 512) -> torch.Tensor:
    """The page-table row of a step ending at ``end``: the request's ids of blocks ``[0, cdiv(end, bs))``, then 0
    (vLLM allocates blocks as the request grows; the bridge zeroes the rest)."""
    out = torch.zeros(width, dtype=torch.int32)
    n = cdiv(end, block)
    out[:n] = pt_full[:n]
    return out


def schedule_chunks(cfg: MotifTTConfig, sched: Schedule):
    """``[((s, e), plan, chunk)]`` in execution order."""
    out = []
    for s, e in sched.steps:
        plan = plan_step(cfg, s, e, sched.align)
        out += [((s, e), plan, ch) for ch in plan.chunks]
    return out


def schedule_rows(cfg: MotifTTConfig, schedules: Sequence[Schedule]) -> int:
    """Input rows every chunk of ``schedules`` reads (real rows plus the last chunk's bucket padding)."""
    return max(ch.start + ch.bucket for sc in schedules for _, _, ch in schedule_chunks(cfg, sc))


def chunk_input_rows(x: torch.Tensor, junk: Optional[torch.Tensor], start: int, bucket: int, end: int) -> torch.Tensor:
    """The ``bucket`` input rows of a chunk at ``start`` with real rows ``[start, end)``: ``x[start : end]``, then
    ``junk[end : start + bucket]`` for the bucket padding rows (what serving feeds is padding tokens, never the
    prompt's next rows: with those, an intermediate chunk's fill would already write the next chunk's true latents
    into the own partial last block, and a continuation that skipped it would go unnoticed). ``junk=None`` keeps
    ``x``'s rows (the blind variant, for the negative controls)."""
    out = x[start : start + bucket].clone()
    if junk is not None and end < start + bucket:
        out[end - start :] = junk[end : start + bucket]
    return out


def skip_own_partial_block(bs: int):
    """Sabotage for the negative controls (``sabotage(k, (s, e), chunk, host) -> host``): the chunk that holds an
    unaligned step start ``s`` does not rewrite the request's own partial block ``s // bs`` (its fill entry -> -1). The
    table stays self-consistent (a builder with a wrong ``w0``); only the end-to-end checks can see it."""

    def sabotage(k, se, ch, host):
        s = int(se[0])
        if s % bs == 0 or not ch.start <= s < ch.end:
            return host
        f = host.fill.clone()
        f[0, s // bs - ch.start // bs] = -1
        return dataclasses.replace(host, fill=f)

    return sabotage


def explicit_plan(s: int, e: int, chunks: Sequence[Tuple[int, int]], *, block: int, align: int) -> "PP.RowPlan":
    """The ``RowPlan`` of row ``(s, e)`` with the given ``[(start, bucket), ...]`` chunks (the design's worked
    examples), independent of the planner's cost model and of the config's resume alignment. The chunks must follow
    the planner's rules: consecutive, ``align``-aligned starts from ``c0`` (0 below the 128-row tail), covering
    ``[c0, e)``, only the last one padded."""
    c0 = s // align * align
    c0 = 0 if c0 < PP.DEFAULT_SWA_TAIL else c0
    assert chunks[0][0] == c0 and all(a % align == 0 for a, _ in chunks), (c0, chunks)
    assert all(a1 + C1 == a2 for (a1, C1), (a2, _) in zip(chunks, chunks[1:])) and chunks[-1][0] + chunks[-1][1] >= e
    cps = tuple(
        PP.ChunkPlan(start=a, bucket=C, end=min(e, a + C), path=PP.SP0 if a == 0 else PP.SP1, last=i == len(chunks) - 1)
        for i, (a, C) in enumerate(chunks)
    )
    return PP.RowPlan(start=s, end=e, w0=s // block * block, c0=c0, chunks=cps, block_size=block, align=align)


def swa_exact_rows_from(cfg: MotifTTConfig, spec, chunk, single_shot_len: int, *, prefix_end: int, bf16_cache: bool):
    """First chunk-local row from which an sp1 SWA chunk's output rows must be BITWISE the draft-1 single shot's (of
    ``single_shot_len`` rows), or ``None`` when no row is exact.

    The square ``[tail ‖ chunk]`` SDPA and the single shot run the same q / k chunk sizes; when the square's chunk grid
    (it starts at position ``a - T``) lines up with the single shot's (``a - T`` a multiple of q and k), a row whose
    129-key window lies inside the chunk (local row ``>= T``) runs exactly the single shot's tiles on bitwise equal
    K / V / Q rows (the latent is row-local), so its output is bitwise equal (gate G10 at op level; this module's
    probe at the module level, ``a % 128 == 0``). The first ``T`` rows read the tail from the cache: exact too only
    with a bf16 cache whose tail blocks TT itself wrote in this schedule (``a - T >= prefix_end``, the end of a
    host-written cached prefix); a bfp8 tail is a quantized copy of the latent."""
    T = int(cfg.prefill_swa_tail)
    if spec.sliding_window_size is None or int(single_shot_len) < int(spec.sliding_window_size) or not chunk.is_sp1:
        return None
    sq = cfg.resumed_prefill_pc(spec, chunk.bucket)
    ss = cfg.sdpa_prefill_pc(spec, seq_len=int(single_shot_len))
    q, k = int(sq.q_chunk_size), int(sq.k_chunk_size)
    if (q, k) != (int(ss.q_chunk_size), int(ss.k_chunk_size)) or (chunk.start - T) % q or (chunk.start - T) % k:
        return None
    return 0 if bf16_cache and chunk.start - T >= int(prefix_end) else T


def row_pccs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PCC of every row of ``a``, ``b`` ``[R, D]`` (fp64)."""
    a, b = a.double(), b.double()
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-300)


def _emu_project(cfg: MotifTTConfig, spec, src: _AttnSource, x: torch.Tensor, positions: torch.Tensor, tp: int):
    """Per-chip projections of rows ``x [T, D]`` at absolute ``positions [T]`` with the module's weight transforms
    (math in ``x.dtype``): ``q_nope [T, H, 128]``, roped ``q_pe [T, H, 64]`` (virtual order, scale folded), the latent
    ``[T, 576]`` = ``[n | roped k_pe]``, lambda logits ``[T, 64]``, ``g = sigmoid(gate)`` ``[T, 1024]``."""
    from models.demos.motif3.reference.rope import apply_rope, rope_cos_sin

    cos, sin = rope_cos_sin(inv_freq_for_kind(cfg, spec.rope_kind), positions[None], x.dtype)

    def rope(t):  # [T, h, 64]
        return apply_rope(t[None], cos, sin)[0]

    T, eps = x.shape[0], cfg.rms_norm_eps
    rank, rd, nope = cfg.kv_lora_rank, cfg.rope_dim, cfg.qk_nope_head_dim
    cq_n = _rmsnorm(x @ latent_q_weight(src, cfg), eps)
    kvl = x @ latent_kv_weight_for_chip(src, cfg, tp)
    n = _rmsnorm(kvl[:, :rank], eps)
    k_pe = rope(kvl[:, rank : rank + rd][:, None, :])[:, 0]
    lam = kvl[:, rank + rd : rank + rd + cfg.n_signal_heads]
    q = (cq_n @ wq_b_virtual_for_chip(src, cfg, tp, spec.softmax_scale)).reshape(T, cfg.q_heads_per_chip, cfg.head_dim)
    g = torch.sigmoid(cq_n @ wq_b_gate_for_chip(src, cfg, tp))
    return q[..., :nope], rope(q[..., nope:]), torch.cat([n, k_pe], -1), lam, g


def _emu_combine(cfg: MotifTTConfig, src: _AttnSource, tp: int, u_sig, u_noise, lam, g) -> torch.Tensor:
    d = (u_sig - torch.sigmoid(lam @ lambda_expansion(cfg).to(u_sig.dtype)) * u_noise) * g
    return d @ W.wo_for_chip(src["wo"], cfg, tp)


def emulate_sp1_global(cfg, spec, src, x: torch.Tensor, positions: torch.Tensor, keys: torch.Tensor, start: int):
    """fp64 emulation of the sp1 global dataflow, summed over the 8 TP chips: rows ``x [C, D]`` (RoPE at
    ``positions``) attend, absorbed over the latent, the cache rows ``keys [P, 576]`` of positions ``0 .. P-1`` (read
    through the SDPA table; the chunk's own rows are in it: fill first), row ``i`` keys ``[0, start + i]`` (the
    kernel's index mask); ``V = keys[:, :512]``; then ``W_UV'``, the virtual-order differential, gate and ``wo``."""
    C, H, v, rank = x.shape[0], cfg.q_heads_per_chip, cfg.v_head_dim, cfg.kv_lora_rank
    Sg = cfg.signal_heads_per_chip
    mask = torch.arange(keys.shape[0])[None, :] <= (start + torch.arange(C))[:, None]
    X = noise_expansion(cfg).to(x.dtype)
    out = torch.zeros(C, cfg.hidden_size, dtype=x.dtype)
    for tp in range(cfg.tp):
        q_nope, q_pe, _, lam, g = _emu_project(cfg, spec, src, x, positions, tp)
        Q = torch.cat([torch.einsum("thn,hnr->thr", q_nope, w_uk_virtual_for_chip(src, cfg, tp)[0]), q_pe], -1)
        w_uv = w_uv_virtual_for_chip(src, cfg, tp)[0]
        U = []
        for h in range(H):
            s = (Q[:, h] @ keys.T).masked_fill(~mask, float("-inf"))
            U.append((torch.softmax(s, -1) @ keys)[:, :rank] @ w_uv[h])
        u_flat = torch.stack(U, 1).reshape(C, H * v)
        out += _emu_combine(cfg, src, tp, u_flat[:, : Sg * v], u_flat[:, Sg * v :] @ X, lam, g)
    return out


def emulate_sp1_swa(cfg, spec, src, x: torch.Tensor, positions: torch.Tensor, tail: torch.Tensor):
    """fp64 emulation of the sp1 SWA dataflow, summed over the 8 TP chips: the square ``[tail ‖ chunk]`` latent (``tail
    [T, 576]`` = the cache rows of the ``T`` positions before the chunk) -> ``E_pref`` expansion -> ``Q_cat = [first
    T rows of Q | Q]`` (HF head order) -> causal + window attention in square coordinates -> rows ``[T, T + C)`` ->
    the draft-1 epilogue."""
    C, T, H, G, r = x.shape[0], tail.shape[0], cfg.q_heads_per_chip, cfg.kv_groups_per_chip, cfg.grouped_ratio
    nope, v, rank, hpg = cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank, cfg.heads_per_group
    mask = _mask(torch.arange(T + C), spec.sliding_window_size)
    out = torch.zeros(C, cfg.hidden_size, dtype=x.dtype)
    for tp in range(cfg.tp):
        q_nope, q_pe, lat, lam, g = _emu_project(cfg, spec, src, x, positions, tp)
        lat_cat = torch.cat([tail, lat])
        kvx = lat_cat[:, :rank] @ W.prefill_kv_expansion_for_chip(src["wkv_b"], src["kv_norm"], cfg, tp)
        Qv = torch.cat([q_nope, q_pe], -1)
        Qh = torch.cat([Qv[:, a:b] for a, b in hf_order_from_virtual(cfg)], 1)
        Qc = torch.cat([Qh[:T], Qh])
        blk = kvx.shape[1] // G
        outs = []
        for h in range(H):
            gi = h // hpg
            k_g = torch.cat([kvx[:, gi * blk : gi * blk + nope], lat_cat[:, rank:]], -1)
            v_g = kvx[:, gi * blk + nope : (gi + 1) * blk]
            s = (Qc[:, h] @ k_g.T).masked_fill(~mask, float("-inf"))
            outs.append((torch.softmax(s, -1) @ v_g)[T:, :v])
        u_flat = torch.cat(outs, -1)
        u_sig = torch.cat([u_flat[:, hpg * gi * v : (hpg * gi + r) * v] for gi in range(G)], -1)
        noise = [u_flat[:, (hpg * gi + r) * v : (hpg * gi + r + 1) * v] for gi in range(G)]
        out += _emu_combine(cfg, src, tp, u_sig, torch.cat([nz for nz in noise for _ in range(r)], -1), lam, g)
    return out


def emulate_chunk(cfg, spec, src, x_rows: torch.Tensor, tables: ChunkHostTables, cache: torch.Tensor) -> torch.Tensor:
    """fp64 emulation of ``forward_prefill(x, chunk=PrefillChunkInputs.upload(tables), kv_cache)``: the output rows
    ``[C, D]``; ``cache [N, bs, 576]`` (one copy: the latent is the same on every chip) is updated like the fill table
    says, before the SDPA on global layers and after it on SWA layers. All addressing goes through ``tables``."""
    bs, C, a = tables.block_size, tables.bucket, tables.start
    pos = tables.rope.long() if tables.is_sp1 else torch.arange(C)
    lat = emulate_latent(cfg, spec, src, x_rows, pos)[0]

    def fill():
        for j, b in enumerate(tables.fill[0].tolist()):
            if b >= 0:
                cache[b] = lat[j * bs : (j + 1) * bs]

    if not tables.is_sp1:
        out = emulate_attention(cfg, spec, src, x_rows, "prefill")  # draft 1: positions 0 .. C-1
        fill()
        return out
    if spec.sliding_window_size is None:
        fill()
        keys = cache[tables.sdpa[0].long()].reshape(-1, cache.shape[-1])[: a + C]
        return emulate_sp1_global(cfg, spec, src, x_rows, pos, keys, a)
    out = emulate_sp1_swa(cfg, spec, src, x_rows, pos, cache[tables.tail.long()].reshape(-1, cache.shape[-1]))
    fill()
    return out


def test_cpu_chunk_host_tables():
    """``chunk_host_tables`` on the features design's worked examples (§3.1 table, Appendix A; A = bs = 64) and on rows
    planned with A = 128 (gate G9's per-bucket q / k give A = 128: ``c0 < w0``, so the recomputed rows' shared block is
    skipped): path, start, fill table (-1 for shared and pure-padding blocks), SDPA table (real ids, then 0, width 640),
    start index, RoPE rows, SWA tail blocks and their tensor-args slice bounds. The plans are explicit
    (:func:`explicit_plan`), so the expectations hold whatever ``cfg.prefill_resume_alignment`` and cost table are; the
    config's own plans of many rows give tables with the same invariants. The warm-up tables write nothing, read only
    block 0 and stop at :func:`max_sp1_bucket`. Malformed tables raise, including tables that disagree with each other
    (a fill id the SDPA table does not read, a tail that is not the SDPA table's blocks before the start, real rows
    roped at other positions, holes / padding / the null block in the fill table)."""
    cfg = host_cfg()
    bs, Wp, P, T = cfg.kv_block_size, cfg.sp1_page_table_width, cfg.max_model_len, cfg.prefill_swa_tail
    pt = (torch.arange(512, dtype=torch.int32) * 7 + 3) % 4000 + 1  # distinct ids >= 1 (block id = 7 j + 4 mod 4000)
    ids = pt.tolist()

    def tables(s, e, chunks, k=0, align=64):
        plan = explicit_plan(s, e, chunks, block=bs, align=align)
        return plan, chunk_host_tables(cfg, plan, plan.chunks[k], step_page_table(pt, e, bs))

    # ---- A = 64: the design's worked examples ----
    _, t = tables(0, 1000, [(0, 1024)])  # cold: blocks 0-15 real
    assert (t.path, t.start, t.bucket, t.end) == ("sp0", 0, 1024, 1000) and t.sdpa is None and t.tail is None
    assert t.fill[0].tolist() == ids[:16]
    _, t = tables(64, 900, [(0, 1024)])  # hit of one block: sp0, block 0 shared (-1)
    assert t.path == "sp0" and t.fill[0].tolist() == [-1] + ids[1:15] + [-1]
    _, t = tables(640, 1000, [(640, 512)])  # 640 cached: sp1 (640, 512); fill blocks 10-15, then 2 padding blocks
    assert (t.path, t.start, t.bucket, t.end) == ("sp1", 640, 512, 1000)
    assert t.fill[0].tolist() == ids[10:16] + [-1, -1]
    assert tuple(t.sdpa.shape) == (1, Wp) == (1, 640) and t.sdpa[0, :16].tolist() == ids[:16]
    assert int(t.sdpa[0, 16:].abs().sum()) == 0 and int(t.sdpa.min()) >= 0
    assert t.start_idx.tolist() == [640] and torch.equal(t.rope, torch.arange(640, 1152, dtype=torch.int32))
    assert t.tail.tolist() == ids[8:10]
    b = t.tail_bounds(cfg.kv_latent_dim)
    assert [(s.tolist(), e.tolist()) for s, e in b] == [([i, 0, 0, 0], [i + 1, 1, bs, 576]) for i in ids[8:10]]
    _, t = tables(1348, 3000, [(1344, 2048)])  # unaligned continuation: block 21 own (rows 1344-1347 rewritten)
    assert (t.start, t.bucket) == (1344, 2048) and t.fill[0].tolist() == ids[21:47] + [-1] * 6
    assert t.tail.tolist() == ids[19:21]
    _, t = tables(1472, 1500, [(1472, 128)])  # preempted 1200 + 300, 1472 cached: block 23
    assert t.fill[0].tolist() == [ids[23], -1] and t.tail.tolist() == ids[21:23]
    _, t = tables(6976, 9000, [(6976, 2048)])  # Appendix A: blocks 109-140, tail 107-108, SDPA 0-140
    assert (t.start, t.bucket) == (6976, 2048) and t.fill[0].tolist() == ids[109:141]
    assert t.tail.tolist() == ids[107:109] and t.sdpa[0, :141].tolist() == ids[:141] and int(t.sdpa[0, 141:].sum()) == 0
    # the 16,736-token cold prompt: sp0 + 2 sp1 chunks; the last one's padding blocks are skipped
    chunks16k = [(0, 8192), (8192, 8192), (16384, 512)]
    plan = explicit_plan(0, 16736, chunks16k, block=bs, align=64)
    hs = [chunk_host_tables(cfg, plan, ch, step_page_table(pt, 16736, bs)) for ch in plan.chunks]
    assert [h.path for h in hs] == ["sp0", "sp1", "sp1"] and [h.start for h in hs] == [0, 8192, 16384]
    assert hs[2].fill[0].tolist() == ids[256:262] + [-1, -1] and hs[2].tail.tolist() == ids[254:256]
    # ---- A = 128: c0 < w0 (the recomputed rows [c0, w0) lie in a shared block: read, never written) ----
    _, t = tables(1348, 3000, [(1280, 2048)], align=128)  # c0 1280, w0 1344: block 20 skipped, block 21 own
    assert (t.start, t.bucket, t.end) == (1280, 2048, 3000) and t.fill[0].tolist() == [-1] + ids[21:47] + [-1] * 5
    assert t.tail.tolist() == ids[18:20] and t.sdpa[0, :47].tolist() == ids[:47] and int(t.sdpa[0, 47:].sum()) == 0
    assert torch.equal(t.rope, torch.arange(1280, 3328, dtype=torch.int32))
    _, t = tables(960, 4096, [(896, 4096)], align=128)  # the device schedule "hit960_A128": c0 896 < w0 960
    assert t.fill[0].tolist() == [-1] + ids[15:64] + [-1] * 14 and t.tail.tolist() == ids[12:14]
    # ---- the config's planner (any A, cost table): the same invariants ----
    for s, e in [(0, 1000), (64, 900), (640, 1000), (1348, 3000), (1472, 1500), (6976, 9000), (0, 16736),
                 (30000, 32768), (130, 4000), (4095, 4097)]:  # fmt: skip
        plan = cfg.plan_prefill_row(s, e)
        for ch in plan.chunks:
            h = chunk_host_tables(cfg, plan, ch, step_page_table(pt, e, bs))
            a, C, n = ch.start, ch.bucket, cdiv(ch.end, bs)
            want_fill = [ids[a // bs + j] if plan.w0 // bs <= a // bs + j < n else -1 for j in range(C // bs)]
            assert h.fill[0].tolist() == want_fill, (s, e, a, C)
            if ch.is_sp1:
                assert h.sdpa[0, :n].tolist() == ids[:n] and int(h.sdpa[0, n:].abs().sum()) == 0
                assert h.tail.tolist() == ids[(a - T) // bs : a // bs] and h.start_idx.tolist() == [a]
                assert torch.equal(h.rope, torch.arange(a, a + C, dtype=torch.int32).clamp(max=P - 1))
    # ---- warm-up: nothing written, only the null block read, at an aligned start >= the tail ----
    a_w = warmup_start(cfg)
    assert a_w >= T and a_w % cfg.prefill_resume_alignment == 0
    assert max_sp1_bucket(cfg) == cfg.max_prefill_span == 8192  # 128 + 8192 <= 32768
    for path in ("sp0", "sp1"):
        for C in cfg.prefill_span_buckets:
            w = warmup_chunk_host_tables(cfg, path, C)
            assert w.path == path and w.bucket == C and bool((w.fill == -1).all())
            if path == "sp1":
                assert w.start == a_w and int(w.sdpa.abs().sum()) == 0 and w.tail.tolist() == [0, 0]
                assert w.end == min(a_w + C, P) and int(w.rope.max()) <= P - 1 and int(w.rope[0]) == a_w
    # span cap 32768 / max_model_len 8192: the largest bucket cannot be an sp1 chunk (square of 128 + C rows)
    for kw, want in ((dict(prefill_span_cap=32768), 16384), (dict(max_model_len=8192), 4096)):
        c2 = host_cfg(**kw)
        top = c2.prefill_span_buckets[-1]
        assert max_sp1_bucket(c2) == want and top == c2.max_prefill_span > want, (kw, c2.prefill_span_buckets)
        assert warmup_chunk_host_tables(c2, "sp0", top).bucket == top
        assert warmup_chunk_host_tables(c2, "sp1", want).end == min(warmup_start(c2) + want, c2.max_model_len)
        with pytest.raises(ValueError, match="max_sp1_bucket"):
            warmup_chunk_host_tables(c2, "sp1", top)
    # ---- malformed tables raise ----
    good = tables(640, 1000, [(640, 512)])[1]  # fill ids[10:16] + [-1, -1] (entries 0-5 written), tail ids[8:10]

    def fill_with(**entries):
        f = good.fill.clone()
        for j, v in entries.items():
            f[0, int(j[1:])] = v
        return f

    def sdpa_with(j, v):
        t_ = good.sdpa.clone()
        t_[0, j] = v
        return t_

    bad = [
        dict(sdpa=good.sdpa.clone().index_fill_(1, torch.tensor([3]), -1)),  # -1 in an SDPA table (D5)
        dict(start_idx=torch.tensor([576], dtype=torch.int32)),  # start index != start
        dict(start=608, start_idx=torch.tensor([608], dtype=torch.int32)),  # start not a block multiple
        dict(sdpa=good.sdpa[:, :8].contiguous()),  # table does not cover a + C
        dict(fill=good.fill[:, :4].contiguous()),  # wrong fill width
        dict(rope=good.rope[:256].contiguous()),  # RoPE rows != bucket
        dict(tail=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int32)),  # tail past the start
        dict(path="sp2"),
        dict(fill=good.fill.to(torch.int64)),
        # cross-table / structure (review finding 4)
        dict(fill=fill_with(j1=ids[100])),  # the fill writes a block the SDPA table does not read there
        dict(sdpa=sdpa_with(12, ids[200])),  # the SDPA table reads another block than the fill writes (block 12)
        dict(sdpa=sdpa_with(3, 0)),  # a cached-prefix position on the null block
        dict(fill=fill_with(j0=0)),  # a write into the null block
        dict(fill=fill_with(j2=-1)),  # a hole in the written run
        dict(fill=fill_with(j5=-1)),  # the block of the last real row (999) not written
        dict(fill=fill_with(j6=ids[16])),  # a pure padding block written
        dict(tail=good.tail.flip(0).contiguous()),  # tail blocks out of position order
        dict(tail=torch.tensor(ids[7:9], dtype=torch.int32)),  # the tail one block early
        dict(rope=good.rope + 1),  # real rows roped one position late
        dict(rope=good.rope.clamp(max=900)),  # real rows clamped
    ]
    for kw in bad:
        with pytest.raises((ValueError, TypeError)):
            dataclasses.replace(good, **kw)
    sp0 = tables(64, 900, [(0, 1024)])[1]  # fill [-1, ids 1-14, -1]
    with pytest.raises(ValueError):  # an sp0 chunk carries no sp1 tables
        dataclasses.replace(tables(0, 1000, [(0, 1024)])[1], sdpa=good.sdpa)
    for j, v in ((3, -1), (14, -1), (15, ids[15]), (1, 0)):  # hole, last real block skipped, padding, null block
        f = sp0.fill.clone()
        f[0, j] = v
        with pytest.raises(ValueError):
            dataclasses.replace(sp0, fill=f)
    log("chunk host tables: worked examples (A = 64 / 128), planner invariants, warm-up tables and the guards ok")


def _fake_chunk(cfg: MotifTTConfig, host: ChunkHostTables) -> PrefillChunkInputs:
    """:class:`PrefillChunkInputs` of :class:`_FT` fake device tensors with the device inputs' shapes."""
    C, rd = host.bucket, cfg.rope_dim
    fill = _FT((1, C // host.block_size), "i32", {"fill_pt"})
    if not host.is_sp1:
        return PrefillChunkInputs(host.path, 0, C, host.end, fill)
    rot = {k: (_FT((1, 1, C, rd), "bf16", {f"cos_{k}_chunk"}), _FT((1, 1, C, rd), "bf16", {f"sin_{k}_chunk"}))
           for k in ("yarn", "plain")}  # fmt: skip
    tail = tuple((_FT((4,), "i32", {f"tail{j}_s"}), _FT((4,), "i32", {f"tail{j}_e"})) for j in range(host.tail.numel()))
    return PrefillChunkInputs(
        host.path, host.start, C, host.end, fill, rot=rot, sdpa_pt=_FT(tuple(host.sdpa.shape), "i32", {"sdpa_pt"}),
        start_idx=_FT((1,), "i32", {"start_idx"}), tail_bounds=tail, rot_idx=_FT((1, C), "u32", {"rot_idx"}),
    )  # fmt: skip


def _kw(call) -> Dict[str, Any]:
    return dict(call[2])


def test_cpu_sp1_op_sequences(monkeypatch):
    """Host proof of the chunk paths' structure (recording fake ``ttnn``, as for draft 1):

    * an sp0 chunk issues exactly the draft-1 ``forward_prefill(page_table=fill_pt)`` sequence (bitwise draft 1);
    * sp1 global: the chunk's latent is filled BEFORE ``chunked_scaled_dot_product_attention(Q [1,10,C,576], K = V =
      the cache, sdpa_pt, chunk_start_idx_tensor=start_idx, scale 1)`` with the G9 compute role (fp32 dest acc) and
      ``cfg.resumed_prefill_pc``; the output keeps columns ``[:512]``; RoPE uses the chunk's gathered tables;
    * sp1 SWA: two tensor-args dim-0 slices of the cache (``num_devices`` = its block count) -> concat -> bf16; the
      square SDPA ``[1, 10, 128 + C, 192]`` with window 129, the ``sdpa_prefill`` role and ``cfg.resumed_prefill_pc``;
      rows ``[128, 128 + C)``; the fill after the SDPA;
    * no leaks: every op output is freed exactly once (except the result), and neither the cache nor any chunk input
      (shared by all layers) is ever freed; ``fill_kv(chunk=)`` == ``fill_kv(fill_pt=, rot=)``;
    * the guards raise before any op: misaligned global start, missing cache, wrong rows, wrong tail, both forms."""
    cfg = host_cfg()
    bs = cfg.kv_block_size
    pt = torch.arange(1, 513, dtype=torch.int32)
    N = 200
    for layer in (0, 1):
        spec = cfg.layer(layer)
        glob = spec.sliding_window_size is None
        cache = _FT((N, 1, bs, 576), "bfp8", {"cache"})
        # ---- sp0 chunk == draft-1 call with the fill table ----
        plan = cfg.plan_prefill_row(64, 900)
        inp = _fake_chunk(cfg, chunk_host_tables(cfg, plan, plan.chunks[0], step_page_table(pt, 900, bs)))
        fake = _FakeTTNN()
        monkeypatch.setattr(A, "ttnn", fake)
        attn = _fake_attention(A, cfg, spec, fake)
        x = _FT((1, 1, 1024, 4096), "bf16", {"x"})
        out = attn.forward_prefill(x, chunk=inp, kv_cache=cache)
        seq_chunk = list(fake.calls)
        assert not any(fake.leaks(keep=[out]))
        fake.calls.clear()
        attn.forward_prefill(x, page_table=inp.fill_pt, kv_cache=cache)
        assert seq_chunk == fake.calls and len(seq_chunk) > 60
        # ---- sp1 chunk (640, 1000) -> (640, 512) ----
        plan = cfg.plan_prefill_row(640, 1000)
        host = chunk_host_tables(cfg, plan, plan.chunks[0], step_page_table(pt, 1000, bs))
        inp = _fake_chunk(cfg, host)
        C = host.bucket
        fake = _FakeTTNN()
        monkeypatch.setattr(A, "ttnn", fake)
        attn = _fake_attention(A, cfg, spec, fake)
        x = _FT((1, 1, C, 4096), "bf16", {"x"})
        out = attn.forward_prefill(x, chunk=inp, kv_cache=cache)
        ops = [c[0] for c in fake.calls]
        never, bad = fake.leaks(keep=[out])
        assert not never and not bad, (never, bad)
        assert tuple(out.shape) == (1, 1, C, 4096) and ops.count("ar_tp") == 1
        rope_calls = [c for c in fake.calls if c[0] == "rotary_embedding_hf"]
        kind = spec.rope_kind
        assert len(rope_calls) == 2 and all(
            c[1][1] == _desc(inp.rot[kind][0]) and c[1][2] == _desc(inp.rot[kind][1]) for c in rope_calls
        )
        assert ops.count("paged_fill_cache") == 1
        fill_call = fake.calls[ops.index("paged_fill_cache")]
        assert fill_call[1][0] == _desc(cache) and fill_call[1][2] == _desc(inp.fill_pt)
        if glob:
            assert "scaled_dot_product_attention" not in ops and ops.count("chunked_scaled_dot_product_attention") == 1
            i_sdpa = ops.index("chunked_scaled_dot_product_attention")
            assert ops.index("paged_fill_cache") < i_sdpa  # fill FIRST: the chunk's own keys come from the cache
            call = fake.calls[i_sdpa]
            q_d, k_d, v_d, pt_d = call[1]
            assert q_d[1] == (1, cfg.q_heads_per_chip, C, cfg.kv_latent_dim) and k_d == v_d == _desc(cache)
            assert pt_d == _desc(inp.sdpa_pt)
            kw = _kw(call)
            assert kw["chunk_start_idx_tensor"] == _desc(inp.start_idx) and kw["scale"] == 1.0
            assert kw["compute_kernel_config"] == "ckc:sdpa_prefill_fp32"
            assert kw["program_config"][2:] == cfg.sp1_global_chunks(C)
            nxt = fake.calls[i_sdpa + 2]  # after deallocate(q_abs): the [:512] slice
            assert nxt[0] == "slice" and nxt[1][1:] == ((0, 0, 0, 0), (1, cfg.q_heads_per_chip, C, cfg.kv_lora_rank))
        else:
            tails = [c for c in fake.calls if c[0] == "slice" and _kw(c).get("slice_dim") == 0]
            assert len(tails) == 2 and all(c[1][0] == _desc(cache) and _kw(c)["num_devices"] == N for c in tails)
            assert [c[1][1:] for c in tails] == [(_desc(s), _desc(e)) for s, e in inp.tail_bounds]
            assert "chunked_scaled_dot_product_attention" not in ops and ops.count("scaled_dot_product_attention") == 1
            i_sdpa = ops.index("scaled_dot_product_attention")
            call = fake.calls[i_sdpa]
            T = cfg.prefill_swa_tail
            assert call[1][0][1] == (1, cfg.q_heads_per_chip, T + C, cfg.head_dim)
            assert call[1][1][1] == call[1][2][1] == (1, cfg.kv_groups_per_chip, T + C, cfg.head_dim)
            kw = _kw(call)
            assert kw["sliding_window_size"] == 129 and kw["is_causal"] is True and kw["scale"] == 1.0
            assert kw["compute_kernel_config"] == "ckc:sdpa_prefill" and kw["program_config"][2:] == (128, 128)
            assert ops.index("paged_fill_cache") > ops.index("ar_tp") > i_sdpa  # fill after the SDPA, as sp0
            o_slice = fake.calls[i_sdpa + 4]  # after deallocate(q_cat, K, V_pad): the chunk rows
            assert o_slice[0] == "slice" and o_slice[1][1:] == ((0, 0, T, 0), (1, cfg.q_heads_per_chip, T + C, 128))
            typecasts = [c for c in fake.calls if c[0] == "typecast"]
            assert typecasts[0][1][1] == "bf16" and typecasts[-1][1][1] == "bfp8"  # tail -> bf16; fill row -> cache
        # fill_kv(chunk=) == fill_kv(fill_pt=, rot=)
        fake.calls.clear()
        attn.fill_kv(x, chunk=inp, kv_cache=cache)
        a_calls = list(fake.calls)
        fake.calls.clear()
        attn.fill_kv(x, fill_pt=inp.fill_pt, rot=inp.rot, kv_cache=cache)
        assert a_calls == fake.calls
        # ---- guards: raise before any op ----
        fake.calls.clear()
        qc, kc = cfg.sp1_global_chunks(C)
        misaligned = dataclasses.replace(inp, start=3 * max(qc, kc) + 32)
        bad_cases = [
            (dict(chunk=inp), ValueError),  # no kv_cache
            (dict(chunk=inp, kv_cache=cache, page_table=inp.fill_pt), ValueError),  # both forms
            (dict(chunk=dataclasses.replace(inp, bucket=2 * C), kv_cache=cache), ValueError),  # rows != bucket
            (dict(chunk=dataclasses.replace(inp, rot=None), kv_cache=cache), ValueError),  # no RoPE rows
            (dict(chunk="not a chunk", kv_cache=cache), TypeError),
        ]
        if glob:
            bad_cases.append((dict(chunk=misaligned, kv_cache=cache), ValueError))  # R6 / G9
            bad_cases.append((dict(chunk=dataclasses.replace(inp, sdpa_pt=None), kv_cache=cache), ValueError))
        else:
            bad_cases.append((dict(chunk=dataclasses.replace(inp, tail_bounds=inp.tail_bounds[:1]), kv_cache=cache),
                              ValueError))  # fmt: skip
            bad_cases.append((dict(chunk=dataclasses.replace(inp, start=64), kv_cache=cache), ValueError))  # < tail
        for kw, exc in bad_cases:
            with pytest.raises(exc):
                attn.forward_prefill(x, **kw)
        if not glob:  # a square [tail | chunk] longer than max_model_len (span cap 32768) is refused
            P = cfg.max_model_len
            with pytest.raises(ValueError, match="max_model_len"):
                attn.forward_prefill(_FT((1, 1, P, 4096), "bf16", {"x"}), chunk=dataclasses.replace(inp, bucket=P),
                                     kv_cache=cache)  # fmt: skip
        assert not fake.calls, fake.calls[:3]
        log(
            f"sp1 op sequence L{layer} ({spec.attn_kind}): {len(ops)} calls, "
            f"{'fill -> chunked SDPA (fp32 acc)' if glob else 'tail slices -> square SDPA (window 129) -> fill'}; "
            f"no leaks; guards raise before any op"
        )


def test_cpu_decode_kv_write_hook(monkeypatch):
    """``forward_decode(kv_write=)``: ``None`` keeps the draft-1 sequence (the op-sequence test above). With a writer,
    the draft-1 write ``[transpose to the 8-core shard, free kv_row, paged_update_cache, free]`` is replaced by
    exactly one ``writer.write(kv_row, kv_cache, cur_pos=, page_table=)`` and the free of ``kv_row``, at the same
    place (after the q path, before FlashMLA); everything else is identical. No leaks; a writer without ``write``
    raises."""
    cfg = host_cfg()
    for layer in (0, 1):
        spec = cfg.layer(layer)
        fake = _FakeTTNN()
        monkeypatch.setattr(A, "ttnn", fake)
        attn = _fake_attention(A, cfg, spec, fake)
        rot = {k: (_FT((1, 1, 32, 64), "bf16", {f"cos_{k}"}), _FT((1, 1, 32, 64), "bf16", {f"sin_{k}"}))
               for k in ("yarn", "plain")}  # fmt: skip
        kw = dict(rot=rot, cur_pos=_FT((8,), "i32", {"cur"}), page_table=_FT((8, 5), "i32", {"pt"}))
        kw.update(kv_cache=_FT((20, 1, 64, 576), "bfp8", {"cache"}), active=_FT((1, 1, 8, 1024), "bf16", {"act"}))
        x = _FT((1, 1, 8, 4096), "bf16", {"x"})
        attn.forward_decode(x, **kw)
        seq0 = list(fake.calls)

        class Writer:
            def __init__(self):
                self.n = 0

            def write(self, kv_row, kv_cache, *, cur_pos, page_table):
                self.n += 1
                kws = (("cur_pos", _desc(cur_pos)), ("page_table", _desc(page_table)))
                fake.calls.append(("kv_write", _desc((kv_row, kv_cache)), kws))

        w = Writer()
        fake = _FakeTTNN()
        monkeypatch.setattr(A, "ttnn", fake)
        attn = _fake_attention(A, cfg, spec, fake)
        out = attn.forward_decode(x, kv_write=w, **kw)
        seq1 = list(fake.calls)
        assert w.n == 1 and not any(fake.leaks(keep=[out]))
        i = next(j for j, c in enumerate(seq0) if c[0] == "transpose" and _kw(c).get("memory_config") == "mc:update")
        assert [c[0] for c in seq0[i : i + 4]] == ["transpose", "deallocate", "paged_update_cache", "deallocate"]
        assert seq1[:i] == seq0[:i] and seq1[i + 2 :] == seq0[i + 4 :]
        assert seq1[i] == ("kv_write", (seq0[i][1][0], _desc(kw["kv_cache"])),
                           (("cur_pos", _desc(kw["cur_pos"])), ("page_table", _desc(kw["page_table"]))))  # fmt: skip
        assert seq1[i + 1][0] == "deallocate" and seq1[i + 1][1][0] == seq0[i][1][0]  # kv_row freed after the write
        assert [c[0] for c in seq1].index("paged_flash_multi_latent_attention_decode") > i
        with pytest.raises(TypeError):
            attn.forward_decode(x, kv_write=object(), **kw)
        log(f"decode kv_write hook L{layer}: one write call replaces the draft-1 update; the rest is identical")


class _FakeRope:
    """The ``MotifRope`` chunk API on :class:`_FakeTTNN` tensors: index tensors carry their rows (``.data``); a
    gathered table carries the rows it was gathered from, so a test sees which positions a layer would rope with."""

    def __init__(self, fake: _FakeTTNN, max_positions: int, dim: int = 64):
        self.fake, self.P, self.dim = fake, int(max_positions), int(dim)
        self.gathers = 0

    def chunk_rot_idxs_device(self, positions):
        return self.fake.from_torch(chunk_rot_rows(positions, self.P), dtype="u32", device=True)

    def chunk_rot_idxs_host(self, positions):
        return self.fake.from_torch(chunk_rot_rows(positions, self.P), dtype="u32")

    def chunk_rope_tables(self, rot_idx, kinds=None):
        self.gathers += 1
        C = int(rot_idx.shape[-1])
        out = {}
        for k in kinds if kinds is not None else ("yarn", "plain"):
            pair = []
            for part in ("cos", "sin"):
                t = self.fake.new((1, 1, C, self.dim), "bf16", {f"{part}_{k}_gather{self.gathers}"})
                t.data = rot_idx.data.clone()  # the positions this table holds
                pair.append(t)
            out[k] = tuple(pair)
        return out


def test_cpu_chunk_inputs_write(monkeypatch):
    """``PrefillChunkInputs.upload`` / ``write`` / ``free`` on the host (recording fake ``ttnn``, tensors carry their
    values): ``write`` copies every table of the next chunk into the persistent tensors in place (the same objects),
    always frees the previous chunk's gathered RoPE rows, and with ``regather=True`` gathers the new positions. With
    ``regather=False`` it leaves ``rot = None``, so an eager ``forward_prefill(chunk=)`` / ``fill_kv(chunk=)`` raises
    before any op (it used to rope q_pe, k_pe and the cache rows at the OLD positions silently) until tables of the new
    positions are passed in (``dataclasses.replace(inp, rot=...)``, as a captured prefill does). Inputs of another
    ``(path, bucket)``, tail-block count or block size are refused; ``free`` frees every tensor exactly once."""
    cfg = host_cfg()
    bs, D = cfg.kv_block_size, cfg.kv_latent_dim
    fake = _FakeTTNN()
    monkeypatch.setattr(A, "ttnn", fake)
    rope = _FakeRope(fake, cfg.max_model_len)
    mesh = SimpleNamespace(name="fake mesh")
    pt = torch.arange(1, 513, dtype=torch.int32)

    def host_of(s, e):
        plan = cfg.plan_prefill_row(s, e)
        assert len(plan.chunks) == 1, plan
        return chunk_host_tables(cfg, plan, plan.chunks[0], step_page_table(pt, e, bs))

    h1, h2, h3 = host_of(1024, 1536), host_of(2048, 2560), host_of(3072, 3500)
    assert {(h.path, h.bucket) for h in (h1, h2, h3)} == {("sp1", 512)}

    def holds(inp, h) -> bool:
        tails = [(s.data, e.data) for s, e in inp.tail_bounds]
        return (
            torch.equal(inp.fill_pt.data, h.fill) and torch.equal(inp.sdpa_pt.data, h.sdpa)
            and torch.equal(inp.start_idx.data, h.start_idx) and torch.equal(inp.rot_idx.data[0], h.rope)
            and len(tails) == len(h.tail_bounds(D))
            and all(torch.equal(a, s) and torch.equal(b, e) for (a, b), (s, e) in zip(tails, h.tail_bounds(D)))
            and (inp.start, inp.end) == (h.start, h.end)
        )  # fmt: skip

    def rot_rows(inp):
        return [t.data for cs in inp.rot.values() for t in cs]

    inp = PrefillChunkInputs.upload(mesh, cfg, rope, h1)
    persistent = [inp.fill_pt, inp.sdpa_pt, inp.start_idx, inp.rot_idx] + [t for p in inp.tail_bounds for t in p]
    assert holds(inp, h1) and all(torch.equal(r[0], h1.rope) for r in rot_rows(inp))
    # ---- regather=True: in place, old rows freed, new rows gathered ----
    old = [t for cs in inp.rot.values() for t in cs]
    inp.write(mesh, cfg, rope, h2, regather=True)
    assert holds(inp, h2) and all(torch.equal(r[0], h2.rope) for r in rot_rows(inp))
    now = [inp.fill_pt, inp.sdpa_pt, inp.start_idx, inp.rot_idx] + [t for p in inp.tail_bounds for t in p]
    assert all(a is b for a, b in zip(now, persistent)), "write must keep the persistent tensors (trace addresses)"
    assert all(sum(f is t for f in fake.freed) == 1 for t in old)
    # ---- regather=False: rot dropped (and freed), so an eager call cannot rope at the old positions ----
    old = [t for cs in inp.rot.values() for t in cs]
    inp.write(mesh, cfg, rope, h3, regather=False)
    assert holds(inp, h3) and inp.rot is None
    assert all(sum(f is t for f in fake.freed) == 1 for t in old)
    for layer in (0, 1):
        attn = _fake_attention(A, cfg, cfg.layer(layer), fake)
        x, cache = _FT((1, 1, 512, 4096), "bf16", {"x"}), _FT((200, 1, bs, D), "bfp8", {"cache"})
        n_calls = len(fake.calls)
        with pytest.raises(ValueError, match="regather=False"):
            attn.forward_prefill(x, chunk=inp, kv_cache=cache)
        with pytest.raises(ValueError, match="regather=False"):
            attn.fill_kv(x, chunk=inp, kv_cache=cache)
        assert len(fake.calls) == n_calls, "the guard must raise before any op"
        tables = rope.chunk_rope_tables(inp.rot_idx)  # what a captured prefill gathers inside the trace
        attn.forward_prefill(x, chunk=dataclasses.replace(inp, rot=tables), kv_cache=cache)
        roped = [c for c in fake.calls[n_calls:] if c[0] == "rotary_embedding_hf"]
        kind = cfg.layer(layer).rope_kind
        assert len(roped) == 2 and all(c[1][1] == _desc(tables[kind][0]) for c in roped)
        assert torch.equal(tables[kind][0].data[0], h3.rope)
    # ---- refusals ----
    with pytest.raises(ValueError):  # another bucket
        inp.write(mesh, cfg, rope, host_of(4096, 4200), regather=False)
    with pytest.raises(ValueError):  # an sp0 chunk
        plan0 = cfg.plan_prefill_row(0, 500)
        inp.write(mesh, cfg, rope, chunk_host_tables(cfg, plan0, plan0.chunks[0], step_page_table(pt, 500, bs)))
    n_calls = len(fake.calls)
    with pytest.raises(ValueError):  # inputs holding another number of tail blocks
        dataclasses.replace(inp, tail_bounds=inp.tail_bounds[:1]).write(mesh, cfg, rope, h2, regather=False)
    odd = dataclasses.replace(h2)
    object.__setattr__(odd, "block_size", 32)  # bypasses the dataclass check: write itself must refuse it
    with pytest.raises(ValueError, match="block size"):
        inp.write(mesh, cfg, rope, odd, regather=False)
    assert len(fake.calls) == n_calls and holds(inp, h3), "a refused write must not copy anything"
    # ---- free: every device tensor once ----
    held = inp.tensors()
    inp.free()
    assert inp.tensors() == [] and all(sum(f is t for f in fake.freed) == 1 for t in held)
    log("PrefillChunkInputs.write: in place, stale RoPE rows freed; regather=False -> eager calls refused; free ok")


def emulate_schedule(cfg, spec, src, sched: Schedule, x, junk, pt_full, init, sabotage=None):
    """Every chunk of ``sched`` through :func:`emulate_chunk` on a copy of the cache ``init``, with chunk inputs
    :func:`chunk_input_rows` (prompt rows + ``junk`` padding rows). ``sabotage(k, (s, e), chunk, host) -> host`` may
    replace a chunk's tables (negative controls). Returns ``(rows [S, D] (NaN where not computed), cache, chunks)``."""
    bs, S = cfg.kv_block_size, sched.end
    cache = init.clone()
    got = torch.full((S, cfg.hidden_size), float("nan"), dtype=x.dtype)
    chunks = schedule_chunks(cfg, sched)
    for k, ((s, e), plan, ch) in enumerate(chunks):
        host = chunk_host_tables(cfg, plan, ch, step_page_table(pt_full, e, bs))
        if sabotage is not None:
            host = sabotage(k, (s, e), ch, host)
        out = emulate_chunk(cfg, spec, src, chunk_input_rows(x, junk, ch.start, ch.bucket, ch.end), host, cache)
        got[ch.start : ch.end] = out[: ch.end - ch.start]
    return got, cache, chunks


@pytest.mark.parametrize("layer", [0, 1], ids=["global_L0", "swa_L1"])
def test_cpu_resumed_schedules_fp64(layer):
    """The features design §5.2 schedules (scaled to a 1024-token prompt) through the fp64 emulation of the module's
    chunk dataflow, driven only by ``chunk_host_tables`` on an emulated paged cache (stale rows everywhere, a cached
    prefix written "by another request" (+1e-9 so a rewrite is visible)); chunk inputs are the prompt rows plus junk
    bucket-padding rows (:func:`chunk_input_rows`), so an intermediate chunk leaves junk latents in the own partial
    block:

    * every row of every chunk (recomputed rows included) equals the reference ``GDLAttention`` single shot (fp64);
    * every block the request owns ends up holding the single-shot latent; shared blocks below ``w0`` and every other
      block (spares, null block 0) are untouched;
    * negative controls: an SWA chunk with a wrong tail (null block) is wrong exactly on its first 128 rows (the window
      straddles the chunk start) and bitwise unchanged after them; a global chunk with the cached prefix zeroed in its
      SDPA table is wrong; the "unaligned" schedule whose continuation does not rewrite the own partial block
      (:func:`skip_own_partial_block`) is caught (own blocks keep junk latents; on the global layer the rows attend
      them), while with the prompt's true rows as padding the same sabotage is invisible (why the padding is junk)."""
    cfg = host_cfg()
    args = ref_args(q_path_fp32=False)
    spec, prefix = resolve_attn_layer(cfg, layer, cache=False)
    t64 = {k: v.double() for k, v in random_attn_tensors(args, seed=600 + layer).items()}
    ref = ref_attention_spec(args, layer, t64, torch.float64)
    src = _AttnSource(source_for(t64, prefix), layer, prefix)
    S, bs, D = HOST_S, cfg.kv_block_size, cfg.hidden_size
    rows = schedule_rows(cfg, HOST_SCHEDULES)
    g = torch.Generator().manual_seed(700 + layer)
    x = torch.randn(rows, D, generator=g, dtype=torch.float64)
    junk = torch.randn(rows, D, generator=torch.Generator().manual_seed(750 + layer), dtype=torch.float64)
    want = ref(x[None, :S], torch.arange(S)[None])[0]
    lat_true = emulate_latent(cfg, spec, src, x[:S], torch.arange(S))[0]  # [S, 576]
    nreq = cdiv(S, bs)
    pool = 1 + cdiv(rows, bs) + 4
    pt_full = _page_table(pool, nreq, seed=800 + layer)
    stale = torch.randn(pool, bs, cfg.kv_latent_dim, generator=g, dtype=torch.float64)
    glob = spec.sliding_window_size is None
    seen_unaligned = False
    for sched in HOST_SCHEDULES:
        chunks = schedule_chunks(cfg, sched)
        w0 = chunks[0][1].w0
        init = stale.clone()
        for j in range(sched.cached // bs):  # cached prefix: written by another request
            init[int(pt_full[j])] = lat_true[j * bs : (j + 1) * bs] + 1e-9 * torch.randn(bs, 576, generator=g)
        got, cache, _ = emulate_schedule(cfg, spec, src, sched, x, junk, pt_full, init)
        c0 = chunks[0][2].start  # rows [0, c0) are cached and never computed
        assert bool(torch.isfinite(got[c0:]).all())
        s_out = stats(want[c0:], got[c0:])
        shared = [int(b) for b in pt_full[: w0 // bs]]
        own = [int(b) for b in pt_full[w0 // bs : nreq]]
        others = sorted(set(range(pool)) - set(pt_full.tolist()))

        def own_ok(c):
            return torch.allclose(c[own].reshape(-1, 576), lat_true[w0:], rtol=0, atol=1e-12)

        shared_ok = torch.equal(cache[shared], init[shared])
        others_ok = torch.equal(cache[others], init[others])
        log(
            f"cpu fp64 L{layer} {sched.name}: chunks {[(c.start, c.bucket, c.path) for _, _, c in chunks][:6]}"
            f"{' ...' if len(chunks) > 6 else ''}; out {fmt(s_out)}; own blocks == single-shot latent {own_ok(cache)}, "
            f"shared {len(shared)} untouched {shared_ok}, others untouched {others_ok}"
        )
        assert s_out["rel_fro"] < 1e-6 and s_out["nonfinite"] == 0, (sched.name, fmt(s_out))
        assert own_ok(cache) and shared_ok and others_ok, sched.name
        if sched.name.startswith("unaligned"):  # negative control: the own partial block not rewritten
            seen_unaligned = True
            assert any(ch.end % bs and ch.end < ch.start + ch.bucket for _, _, ch in chunks[:-1]), chunks
            skip = skip_own_partial_block(bs)
            b_got, b_cache, _ = emulate_schedule(cfg, spec, src, sched, x, junk, pt_full, init, sabotage=skip)
            blind_got, blind_cache, _ = emulate_schedule(cfg, spec, src, sched, x, None, pt_full, init, sabotage=skip)
            s_bad, s_blind = stats(want[c0:], b_got[c0:]), stats(want[c0:], blind_got[c0:])
            log(
                f"cpu fp64 L{layer} negative control (continuation skips the own partial block): junk padding -> rows "
                f"{fmt(s_bad)}, own blocks == single shot {own_ok(b_cache)}; true next rows as padding -> rows "
                f"rel {s_blind['rel_fro']:.2e}, own blocks == single shot {own_ok(blind_cache)} (blind)"
            )
            assert not own_ok(b_cache), "the sabotage must leave junk latents in the own partial block"
            assert not glob or s_bad["rel_fro"] > 1e-2, "global rows must see the junk keys"
            assert own_ok(blind_cache) and s_blind["rel_fro"] < 1e-6  # documents why the padding rows are junk
        if sched.name == "half_cached+half":  # negative controls on a cached-prefix chunk
            (s, e), plan, ch = chunks[0]
            host = chunk_host_tables(cfg, plan, ch, step_page_table(pt_full, e, bs))
            xr = chunk_input_rows(x, junk, ch.start, ch.bucket, ch.end)
            ok_out = emulate_chunk(cfg, spec, src, xr, host, init.clone())
            if glob:
                zeroed = host.sdpa.clone().index_fill_(1, torch.arange(w0 // bs), 0)
                with pytest.raises(ValueError, match="null block"):  # the host tables refuse it already ...
                    dataclasses.replace(host, sdpa=zeroed)
                bad = _unchecked_replace(host, sdpa=zeroed)
                bad_out = emulate_chunk(cfg, spec, src, xr, bad, init.clone())  # ... and the dataflow shows why
                s_bad = stats(want[ch.start : ch.end], bad_out[: ch.end - ch.start])
                log(f"cpu fp64 L{layer} negative control (cached prefix -> null block): {fmt(s_bad)}")
                assert s_bad["rel_fro"] > 1e-2
            else:
                T = cfg.prefill_swa_tail
                with pytest.raises(ValueError, match="tail"):  # refused by the host tables, run anyway below
                    dataclasses.replace(host, tail=torch.zeros_like(host.tail))
                bad = _unchecked_replace(host, tail=torch.zeros_like(host.tail))
                bad_out = emulate_chunk(cfg, spec, src, xr, bad, init.clone())
                s_head = stats(want[ch.start : ch.start + T], bad_out[:T])
                same_after = torch.equal(bad_out[T:], ok_out[T:])
                log(f"cpu fp64 L{layer} negative control (wrong tail): rows [a, a+{T}) {fmt(s_head)}; later rows "
                    f"unchanged {same_after}")  # fmt: skip
                assert s_head["rel_fro"] > 1e-2 and same_after
    assert seen_unaligned


def _unchecked_replace(host: ChunkHostTables, **changes) -> ChunkHostTables:
    """``dataclasses.replace`` without ``__post_init__``: tables the guards refuse, for negative controls that run
    them anyway (on the host emulation or on the device) to show what the guard prevents."""
    out = dataclasses.replace(host)
    for k, v in changes.items():
        object.__setattr__(out, k, v)
    return out


# ======================================================================================================================
# device helpers
# ======================================================================================================================
def _all_chips(t, mesh_device) -> torch.Tensor:
    from models.demos.motif3.tt.ccl import device_tensors_to_torch

    return device_tensors_to_torch(t, mesh_device).float()  # [R, C, ...]


def _rows_by_dp(t, cfg, mesh_device) -> torch.Tensor:
    """Readback of chip (dp, tp=0) for every DP row and of chip (0, tp=7) (decode writes differ per DP row)."""
    from models.demos.motif3.tests.unit.test_attention import _chip_index

    idx = [_chip_index(cfg, dp, 0) for dp in range(cfg.dp)] + [_chip_index(cfg, 0, cfg.tp - 1)]
    return torch.stack([_dev(t, i) for i in idx])


def decode_page_table_width(positions: Sequence[int], block: int, multiple: int = 8) -> int:
    """Decode page-table width for lanes at ``positions``: ``cdiv(max + 1, block)`` rounded up to ``multiple`` (8, as
    the serving width 512). FlashMLA decode runs ``k_chunk = 128`` key chunks over ``W x block`` keys and has no check
    that this is a multiple of the chunk: with W = 5 (320 keys) the lanes in the last, partial chunk (positions 256 ..
    319) read entry 5, past the table, i.e. garbage blocks (outputs up to 5e35, different on every call; seen on this
    Galaxy 2026-10-02)."""
    w = cdiv(max(positions) + 1, block)
    return -(-w // multiple) * multiple


def _bf16_rows(n: int, seed: int) -> torch.Tensor:
    return torch.randn(n, 4096, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16).float()


def _page_table(pool: int, n: int, seed: int) -> torch.Tensor:
    """``int32 [n]`` distinct block ids from ``1 .. pool-1`` (block 0 = the null block)."""
    return (torch.randperm(pool - 1, generator=torch.Generator().manual_seed(seed))[:n] + 1).to(torch.int32)


def _good(s: Dict[str, float], pcc_min: float) -> bool:
    """A reference comparison passes: PCC bar, relative error bound (:data:`REL_MAX`), all finite."""
    return s["pcc"] >= pcc_min and s["rel_fro"] < REL_MAX and s["nonfinite"] == 0


def _latent_check(cache_h, pt_row, positions, ref, x_rows, gam, rank: int):
    """Stats of the cache rows at ``positions`` (through ``pt_row``) vs the reference latents of ``x_rows``."""
    rows = _gather_cache(cache_h, pt_row, positions, int(cache_h.shape[-2]))
    n_ref, kpe_ref = ref_latents(ref, x_rows, positions)
    return stats(n_ref * gam, rows[:, :rank] * gam), stats(kpe_ref, rows[:, rank:])


def _check_draft1(mesh_device, cfg, rope, old, new, bank, tag: str) -> List[str]:
    """Draft 1 bitwise: forward_prefill (S 128, 1024, cache fill) and forward_decode (32 lanes) of the git draft-1
    module and of the current module give bitwise equal outputs and caches."""
    fails, block, kvdt = [], cfg.kv_block_size, cfg.dtypes.kv_cache
    for S in (128, 1024):
        x = _bf16_rows(S, seed=10 + S)
        nb = S // block
        pool = 1 + nb + 3
        pt = _page_table(pool, nb, seed=S)[None]
        stale = _host_quant(stale_paged(bank, pool, block, seed=S), kvdt)
        c_old, c_new = _upload_cache(mesh_device, stale, kvdt), _upload_cache(mesh_device, stale, kvdt)
        x_tt = _replicated(mesh_device, x[None, None], ttnn.bfloat16)
        pt_tt = _replicated(mesh_device, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        o_old = old.forward_prefill(x_tt, page_table=pt_tt, kv_cache=c_old)
        o_new = new.forward_prefill(x_tt, page_table=pt_tt, kv_cache=c_new)
        eq_out = torch.equal(_all_chips(o_old, mesh_device), _all_chips(o_new, mesh_device))
        eq_cache = torch.equal(_all_chips(c_old, mesh_device), _all_chips(c_new, mesh_device))
        log(f"{tag} draft1 vs new prefill S={S}: output bitwise {eq_out}, cache bitwise {eq_cache} (32 chips)")
        if not (eq_out and eq_cache):
            fails.append(f"{tag} prefill S={S}: output bitwise {eq_out}, cache bitwise {eq_cache}")
        _free([o_old, o_new, c_old, c_new, x_tt, pt_tt])
    pos = list(DRAFT1_DECODE_POS)
    Wd = decode_page_table_width(pos, block)
    pool = 1 + cfg.max_batch * Wd
    perm = _page_table(pool, pool - 1, seed=77)
    pt = torch.zeros(cfg.max_batch, Wd, dtype=torch.int32)
    for lane, p in enumerate(pos):
        if p >= 0:
            pt[lane] = perm[lane * Wd : (lane + 1) * Wd]
    stale = _host_quant(stale_paged(bank, pool, block, seed=78), kvdt)
    c_old, c_new = _upload_cache(mesh_device, stale, kvdt), _upload_cache(mesh_device, stale, kvdt)
    d = _decode_step_inputs(mesh_device, cfg, rope, pos, _bf16_rows(cfg.max_batch, seed=79), pt)
    kw = dict(rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], active=d["act"])
    o_old = old.forward_decode(d["x"], kv_cache=c_old, **kw)
    o_new = new.forward_decode(d["x"], kv_cache=c_new, **kw)
    eq_out = torch.equal(_all_chips(o_old, mesh_device), _all_chips(o_new, mesh_device))
    eq_cache = torch.equal(_rows_by_dp(c_old, cfg, mesh_device), _rows_by_dp(c_new, cfg, mesh_device))
    log(
        f"{tag} draft1 vs new decode (32 lanes, 2 inactive): output bitwise {eq_out} (32 chips), "
        f"cache bitwise {eq_cache}"
    )
    if not (eq_out and eq_cache):
        fails.append(f"{tag} decode: output bitwise {eq_out}, cache bitwise {eq_cache}")
    _free([o_old, o_new, c_old, c_new])
    _free_step(d)
    return fails


def _check_fill_sp0(mesh_device, cfg, attn, ref, gam, bank, tag: str) -> List[str]:
    """fill_kv at sp0 (positions 0 .. S-1) vs forward_prefill's fill, bitwise; -1 fill tables (leading shared blocks,
    trailing padding block) skip exactly those blocks, also in forward_prefill, whose output stays bitwise the same;
    the latent vs the reference."""
    fails, block, kvdt, rank = [], cfg.kv_block_size, cfg.dtypes.kv_cache, cfg.kv_lora_rank
    for S in (128, 1024):
        x = _bf16_rows(S, seed=20 + S)
        nb = S // block
        pool = 1 + nb + 3
        pt = _page_table(pool, nb, seed=30 + S)[None]
        skip = pt.clone()
        skip[0, : (1 if nb <= 2 else 2)] = -1  # shared full blocks below w0
        if nb > 2:
            skip[0, -1] = -1  # a pure padding block
        stale = _host_quant(stale_paged(bank, pool, block, seed=40 + S), kvdt)
        caches = {k: _upload_cache(mesh_device, stale, kvdt) for k in ("prefill", "fill", "prefill_skip", "fill_skip")}
        x_tt = _replicated(mesh_device, x[None, None], ttnn.bfloat16)
        pt_tt = _replicated(mesh_device, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        skip_tt = _replicated(mesh_device, skip, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        o_full = attn.forward_prefill(x_tt, page_table=pt_tt, kv_cache=caches["prefill"])
        taps: Dict[str, ttnn.Tensor] = {}
        attn.fill_kv(x_tt, fill_pt=pt_tt, kv_cache=caches["fill"], taps=taps)
        o_skip = attn.forward_prefill(x_tt, page_table=skip_tt, kv_cache=caches["prefill_skip"])
        attn.fill_kv(x_tt, fill_pt=skip_tt, kv_cache=caches["fill_skip"])
        h = {k: _all_chips(c, mesh_device) for k, c in caches.items()}
        eq_fill = torch.equal(h["fill"], h["prefill"])
        eq_out = torch.equal(_all_chips(o_skip, mesh_device), _all_chips(o_full, mesh_device))
        written = pt[0][skip[0] >= 0].long()
        want = stale.clone()
        want[written] = h["prefill"][0, 0][written]
        want = want.expand_as(h["prefill"])
        eq_skip = torch.equal(h["prefill_skip"], want) and torch.equal(h["fill_skip"], want)
        kv = _dev(taps["kv_row"])[0, 0]  # bf16 latent before the typecast
        n_ref, kpe_ref = ref_latents(ref, x, torch.arange(S))
        s_n, s_k = stats(n_ref * gam, kv[:, :rank] * gam), stats(kpe_ref, kv[:, rank:])
        c_n, c_k = _latent_check(h["fill"][0, 0], pt[0], torch.arange(S), ref, x, gam, rank)
        log(
            f"{tag} fill_kv sp0 S={S}: cache == forward_prefill's bitwise {eq_fill} (32 chips); -1 table "
            f"{skip[0].tolist()}: skipped blocks untouched + rest == full fill {eq_skip}, forward_prefill output "
            f"unchanged bitwise {eq_out}; latent bf16 n {fmt(s_n)} k_pe pcc {s_k['pcc']:.6f}; cache rows n pcc "
            f"{c_n['pcc']:.6f} k_pe pcc {c_k['pcc']:.6f}"
        )
        if not (eq_fill and eq_out and eq_skip):
            fails.append(f"{tag} sp0 S={S}: fill==prefill {eq_fill}, skip {eq_skip}, output unchanged {eq_out}")
        if not all(_good(t, LAT_PCC_MIN) for t in (s_n, s_k, c_n, c_k)):
            fails.append(f"{tag} sp0 S={S} latent: {fmt(s_n)} / {fmt(s_k)} / cache {c_n['pcc']} {c_k['pcc']}")
        _free([o_full, o_skip, x_tt, pt_tt, skip_tt, taps["kv_row"], *caches.values()])
    return fails


def _check_chunks(mesh_device, cfg, rope, attn, ref, gam, bank, tag: str) -> List[str]:
    """fill_kv of a resumed chunk alone (rows ``[a, a + C)``, offset RoPE gathered from ``prefill_plan.rope_positions``,
    ``prefill_plan.fill_table``). Every block the fill table names holds bitwise the rows a single-shot fill of
    ``[0, a + C)`` writes; every other block (shared blocks below ``w0``, padding, spares, null) is untouched; the rows
    match the reference latents; the gathered tables are the host rows bitwise; a second start of the same bucket
    compiles no program."""
    fails, block, kvdt, rank = [], cfg.kv_block_size, cfg.dtypes.kv_cache, cfg.kv_lora_rank
    P = cfg.max_model_len
    plans = [(s, e, cfg.plan_prefill_row(s, e)) for s, e in CHUNK_CASES]
    rows = max(ch.start + ch.bucket for _, _, p in plans for ch in p.chunks)
    x_all = _bf16_rows(rows, seed=50)
    pool = (
        1 + cdiv(rows, block) + 3
    )  # one pool shape for every case (as in serving): programs depend on the bucket only
    new_programs: Dict[int, List[int]] = {}
    for i, (s, e, plan) in enumerate(plans):
        assert len(plan.chunks) == 1, plan
        ch = plan.chunks[0]
        a, C = ch.start, ch.bucket
        nbl = cdiv(e, block)  # the request's blocks
        pt_row = _page_table(pool, nbl, seed=60 + i)
        fill = PP.fill_table(pt_row, ch, plan.w0, block)  # int32 [C / bs], -1 = skip
        single = torch.full((cdiv(a + C, block),), -1, dtype=torch.int32)
        single[:nbl] = pt_row  # single-shot fill of [0, a + C): the request's blocks, nothing past them
        stale = _host_quant(stale_paged(bank, pool, block, seed=70 + i), kvdt)
        c_chunk, c_single = _upload_cache(mesh_device, stale, kvdt), _upload_cache(mesh_device, stale, kvdt)
        x1_tt = _replicated(mesh_device, x_all[: a + C][None, None], ttnn.bfloat16)
        s_tt = _replicated(mesh_device, single[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        attn.fill_kv(x1_tt, fill_pt=s_tt, kv_cache=c_single)  # positions 0 .. a+C-1 (sp0 tables)
        pos = PP.rope_positions(ch, P)
        idx_tt = rope.chunk_rot_idxs_device(pos)
        rot = rope.chunk_rope_tables(idx_tt)
        tab_ok = all(
            torch.equal(_dev(t)[0, 0], rope.host_tables[k][j][pos.long()].float())
            for k, cs in rot.items()
            for j, t in enumerate(cs)
        )
        xc_tt = _replicated(mesh_device, x_all[a : a + C][None, None], ttnn.bfloat16)
        f_tt = _replicated(mesh_device, fill[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.synchronize_device(mesh_device)
        n0 = mesh_device.num_program_cache_entries()
        attn.fill_kv(xc_tt, fill_pt=f_tt, kv_cache=c_chunk, rot=rot)
        ttnn.synchronize_device(mesh_device)
        new_programs.setdefault(C, []).append(mesh_device.num_program_cache_entries() - n0)
        hc, hs = _all_chips(c_chunk, mesh_device), _all_chips(c_single, mesh_device)
        wb = fill[fill >= 0].long()
        others = sorted(set(range(pool)) - set(wb.tolist()))
        bitwise = torch.equal(hc[:, :, wb], hs[:, :, wb])
        untouched = torch.equal(hc[:, :, others], stale[others].expand_as(hc[:, :, others]))
        logical = [a // block + j for j in range(C // block) if int(fill[j]) >= 0]
        positions = torch.cat([torch.arange(b * block, (b + 1) * block) for b in logical])
        c_n, c_k = _latent_check(hc[0, 0], pt_row, positions, ref, x_all[positions], gam, rank)
        diff = float((hc[:, :, wb] - hs[:, :, wb]).abs().max())
        log(
            f"{tag} chunk s={s} e={e} -> ({ch.path}, start {a}, bucket {C}), w0 {plan.w0}, fill {fill.tolist()}: "
            f"written blocks == single-shot fill bitwise {bitwise} (max |diff| {diff:.3e}), others untouched "
            f"{untouched}, rope tables == host rows {tab_ok}; rows vs reference n pcc {c_n['pcc']:.6f} k_pe pcc "
            f"{c_k['pcc']:.6f}; programs compiled by this fill {new_programs[C][-1]}"
        )
        if not (bitwise and untouched and tab_ok):
            fails.append(f"{tag} chunk ({s}, {e}): bitwise {bitwise} untouched {untouched} tables {tab_ok}")
        if not (_good(c_n, LAT_PCC_MIN) and _good(c_k, LAT_PCC_MIN)):
            fails.append(f"{tag} chunk ({s}, {e}) latent: n {fmt(c_n)} / k_pe {fmt(c_k)}")
        _free([c_chunk, c_single, x1_tt, s_tt, idx_tt, xc_tt, f_tt] + [t for cs in rot.values() for t in cs])
    for C, counts in new_programs.items():
        if any(n != 0 for n in counts[1:]):
            fails.append(f"{tag}: bucket {C} compiled programs for a later chunk start: {counts}")
    return fails


# ======================================================================================================================
# device tests
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_wp2a_decoder_layers(mesh_device, device_params):
    """Global L0 and SWA L1, random weights. (1) Draft 1 bitwise: the git draft-1 module and the current one give
    bitwise equal forward_prefill and forward_decode outputs and caches. (2) fill_kv at sp0 equals forward_prefill's
    fill bitwise, and ``-1`` fill tables skip exactly their blocks. (3) Resumed chunks filled alone through the
    prefill_plan tables and the offset-RoPE gather: bitwise the single-shot rows, nothing else touched, one program
    set per bucket."""
    from models.demos.motif3.tt.decoder import free_tensors

    d1 = draft1_attention_module()
    cfg, ccl, rope = _setup(mesh_device, "wp2a decoder layers")
    args = ref_args()
    failures: List[str] = []
    for layer in (0, 1):
        tag = f"L{layer} {cfg.layer(layer).attn_kind}"
        tensors = random_attn_tensors(args, seed=500 + layer)
        src = source_for(tensors, W.hf_name(layer, "self_attn"))
        new = MotifAttention(mesh_device, cfg, layer, source=src, ccl=ccl, rope=rope, cache=False)
        old = d1.MotifAttention(mesh_device, cfg, layer, source=src, ccl=ccl, rope=rope, cache=False)
        ref = ref_attention_spec(args, layer, tensors)
        gam = tensors["kv_norm"].float()
        bank = stale_bank(ref, "random", layer, n=512, seed=layer)
        failures += _check_draft1(mesh_device, cfg, rope, old, new, bank, tag)
        free_tensors(old)
        failures += _check_fill_sp0(mesh_device, cfg, new, ref, gam, bank, tag)
        failures += _check_chunks(mesh_device, cfg, rope, new, ref, gam, bank, tag)
        free_tensors(new)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_wp2a_mtp_layer(mesh_device, device_params):
    """The MTP layer's attention, ``MotifAttention(cfg.mtp_layer_idx, spec=cfg.mtp_layer_spec(),
    weight_prefix=MTP_ATTN_PREFIX)``, with the real MTP weights (shard 104) and realistic MTP inputs. Checks: prefill
    (S = 512 > window) vs the reference ``GDLAttention(args, 53, swa=True)`` in fp32; fill_kv rows vs the reference
    cache entries and bitwise equal to forward_prefill's; a 32-lane decode step vs the reference. The decode history
    comes from ONE packed fill_kv: 32 requests' rows concatenated, with per-row positions from the offset-RoPE gather
    and the concatenated fill tables."""
    from models.demos.motif3.tt.ccl import replicas_identical
    from models.demos.motif3.tt.decoder import free_tensors

    cfg, ccl, rope = _setup(mesh_device, "wp2a mtp layer")
    args = ref_args()
    tensors = real_mtp_tensors()
    x_all = load_mtp_attention_inputs()
    spec = cfg.mtp_layer_spec()
    L = cfg.mtp_layer_idx
    attn = MotifAttention(
        mesh_device, cfg, L, source=source_for(tensors, MTP_ATTN_PREFIX), ccl=ccl, rope=rope, cache=False,
        spec=spec, weight_prefix=MTP_ATTN_PREFIX,
    )  # fmt: skip
    assert (attn.layer_idx, attn.window, attn.kind, attn.scale, attn.weight_prefix) == (
        53, 129, "plain", spec.softmax_scale, MTP_ATTN_PREFIX)  # fmt: skip
    ref = ref_attention_spec(args, L, tensors, swa=True)
    gam = tensors["kv_norm"].float()
    block, kvdt, rank = cfg.kv_block_size, cfg.dtypes.kv_cache, cfg.kv_lora_rank
    bank = stale_bank(ref, "random", L, n=512, seed=L)
    failures: List[str] = []

    # ---- (1) prefill S = 512 + fill_kv --------------------------------------------------------------------------
    S = 512
    x = x_all[:S]
    want = ref(x[None], torch.arange(S)[None])[0]
    nb = S // block
    pool = 1 + nb + 3
    pt = _page_table(pool, nb, seed=90)[None]
    stale = _host_quant(stale_paged(bank, pool, block, seed=91), kvdt)
    c_pf, c_fk = _upload_cache(mesh_device, stale, kvdt), _upload_cache(mesh_device, stale, kvdt)
    x_tt = _replicated(mesh_device, x[None, None], ttnn.bfloat16)
    pt_tt = _replicated(mesh_device, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    out = attn.forward_prefill(x_tt, page_table=pt_tt, kv_cache=c_pf)
    s_out = stats(want, _dev(out)[0, 0])
    same = replicas_identical(out, mesh_device, "tp", cfg.axes) and replicas_identical(out, mesh_device, "dp", cfg.axes)
    attn.fill_kv(x_tt, fill_pt=pt_tt, kv_cache=c_fk)
    h_pf, h_fk = _all_chips(c_pf, mesh_device), _all_chips(c_fk, mesh_device)
    eq_fill = torch.equal(h_pf, h_fk)
    c_n, c_k = _latent_check(h_fk[0, 0], pt[0], torch.arange(S), ref, x, gam, rank)
    log(
        f"MTP L53 prefill S={S} (real weights, {MTP_PROMPT} inputs): TT vs ref fp32 {fmt(s_out)}; replicas(32) "
        f"identical {same}; fill_kv cache == forward_prefill's bitwise {eq_fill}; cache rows vs ref c_kv/gamma pcc "
        f"{c_n['pcc']:.6f} max_abs {c_n['max_abs']:.3e}, k_pe pcc {c_k['pcc']:.6f}"
    )
    if not (_good(s_out, OUT_PCC_MIN) and same):
        failures.append(f"MTP prefill: {fmt(s_out)} replicas identical {same}")
    if not (eq_fill and _good(c_n, LAT_PCC_MIN) and _good(c_k, LAT_PCC_MIN)):
        failures.append(f"MTP fill_kv: == prefill {eq_fill}, n {fmt(c_n)}, k_pe {fmt(c_k)}")
    _free([out, c_pf, c_fk, x_tt, pt_tt])

    # ---- (2) decode over a packed fill_kv history: lane l = request l, history rows 0 .. 255, decodes at p_l -------
    Hn, B = 256, cfg.max_batch
    per = Hn // block
    pool = 1 + B * per + 3
    pt_lanes = _page_table(pool, B * per, seed=92).reshape(B, per)  # lane l's 4 blocks
    stale = _host_quant(stale_paged(bank, pool, block, seed=93), kvdt)
    cache = _upload_cache(mesh_device, stale, kvdt)
    xh_tt = _replicated(mesh_device, x_all[:Hn].repeat(B, 1)[None, None], ttnn.bfloat16)  # [1, 1, 8192, 4096]
    fill_tt = _replicated(mesh_device, pt_lanes.reshape(1, -1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)  # row i -> i // 64
    idx_tt = rope.chunk_rot_idxs_device(torch.arange(Hn).repeat(B))
    rot = rope.chunk_rope_tables(idx_tt, kinds=(attn.kind,))
    attn.fill_kv(xh_tt, fill_pt=fill_tt, kv_cache=cache, rot=rot)
    hist = _dev(cache)
    lat_pccs, lat_ok = [], True
    for lane in (0, 13, 31):
        c_n, c_k = _latent_check(hist, pt_lanes[lane], torch.arange(Hn), ref, x_all[:Hn], gam, rank)
        lat_pccs += [c_n["pcc"], c_k["pcc"]]
        lat_ok &= _good(c_n, LAT_PCC_MIN) and _good(c_k, LAT_PCC_MIN)
    positions = [129 + 4 * lane for lane in range(B)]  # 129 .. 253: the 129-key window is in effect everywhere
    d = _decode_step_inputs(mesh_device, cfg, rope, positions, x_all[positions], pt_lanes)
    out = attn.forward_decode(
        d["x"], rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], kv_cache=cache, active=d["act"]
    )
    got = _per_lane_out(cfg, mesh_device, out)
    want_dec = ref(x_all[:Hn][None], torch.arange(Hn)[None])[0][positions]  # row p: keys [p - 128, p]
    s_dec = stats(want_dec, got)
    lane_pcc = [pcc(want_dec[l], got[l]) for l in range(B)]
    lane_ok = all(_good(stats(want_dec[l], got[l]), LANE_PCC_MIN) for l in range(B))
    log(
        f"MTP L53 decode (32 lanes at 129..253, history from one packed fill_kv of {B} x {Hn} rows): TT vs ref fp32 "
        f"{fmt(s_dec)}; worst lane pcc {min(lane_pcc):.6f} (lane {lane_pcc.index(min(lane_pcc))}); packed history "
        f"rows vs ref (lanes 0, 13, 31) min pcc {min(lat_pccs):.6f}"
    )
    if not (_good(s_dec, OUT_PCC_MIN) and lane_ok):
        failures.append(f"MTP decode: {fmt(s_dec)} worst lane {min(lane_pcc)}")
    if not lat_ok:
        failures.append(f"MTP packed fill_kv history: min latent pcc {min(lat_pccs)}")
    _free([out, cache, xh_tt, fill_tt, idx_tt] + [t for cs in rot.values() for t in cs])
    _free_step(d)
    free_tensors(attn)
    assert not failures, "\n".join(failures)


# ======================================================================================================================
# device tests, work package 2b: resumed (sp1) prefill and the decode KV-write hook
# ======================================================================================================================
LONG_REAL_INPUTS = TEST_DIR / f"real_inputs_L0-1_{DEVICE_S}.pt"
SP1_OUT_PCC_MIN = 0.9995  # a schedule's computed rows vs the fp32 reference, aggregate (design §5.2)
# Worst single row vs the fp32 reference. Design §5.2 asks per-row >= 0.9995, which the validated draft-1 single shot
# itself misses on this Galaxy (4096-token prompt, bfp8 cache: SWA random worst row 0.99946 with 98.1 % of rows >=
# 0.9995, SWA real worst 0.99913, global random worst 0.99934), so the bar is test_attention.py's per-position 0.999
# (relaxation reported for the design / GATES_RESULTS record). No PCC bar sees a one-key window error (~3e-5): the
# SWA rows also get an exact check, bitwise the draft-1 single shot wherever that holds (swa_exact_rows_from).
SP1_ROW_PCC_MIN = 0.999
CHIPS = (0, 13, 31)  # replicas read back where all 32 would be too large (prefill outputs are replicated)
NEG_PCC_MAX = 0.99  # a negative control is "detected" when its rows fall below this


def load_long_real_inputs(layer: int, n: int) -> torch.Tensor:
    """Real normalized attention inputs (``layers.{0,1}.input_layernorm.out`` of the reference bf16 prefix model) of
    the first ``n <= 4096`` tokens of the README chat prompt: ``test_attention.make_real_inputs`` at 4096 tokens,
    cached as :data:`LONG_REAL_INPUTS` (built on first use, ~15 s of CPU)."""
    if not LONG_REAL_INPUTS.is_file():
        make_real_inputs(n_tokens=DEVICE_S, path=LONG_REAL_INPUTS)
    x = torch.load(LONG_REAL_INPUTS, weights_only=True)[f"L{layer}"]
    if x.shape[0] < n:
        pytest.skip(f"{LONG_REAL_INPUTS} has {x.shape[0]} < {n} tokens")
    return x[:n].float()


def _inputs(layer: int, weights_kind: str, rows: int, seed: int) -> torch.Tensor:
    """``[rows, 4096]`` bf16-valued inputs: the 4096 real prompt rows (real weights) or random rows. Rows past 4096 are
    only bucket padding (the real rows repeated; the schedule runs replace every padding row by :func:`_junk_rows`)."""
    if weights_kind == "real":
        x = load_long_real_inputs(layer, DEVICE_S)
        return torch.cat([x] * cdiv(rows, DEVICE_S))[:rows].contiguous()
    return _bf16_rows(rows, seed)


def _junk_rows(layer: int, weights_kind: str, rows: int, seed: int) -> torch.Tensor:
    """``[rows, 4096]`` bucket-padding rows (:func:`chunk_input_rows`): random rows of another seed (random weights),
    or the real prompt rows in a random order (real weights: realistic magnitudes at the wrong positions)."""
    if weights_kind == "real":
        x = load_long_real_inputs(layer, DEVICE_S)
        return x[torch.randint(0, x.shape[0], (rows,), generator=torch.Generator().manual_seed(seed))].contiguous()
    return _bf16_rows(rows, seed)


def _kv_name(dtype) -> str:
    return "bf16" if dtype == ttnn.bfloat16 else "bfp8" if dtype == ttnn.bfloat8_b else str(dtype)


def _chips(t, idx: Sequence[int] = CHIPS) -> torch.Tensor:
    return torch.stack([_dev(t, i) for i in idx])


def _rep_i32(mesh_device, t: torch.Tensor):
    return _replicated(mesh_device, t, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)


def _write_prefix(paged: torch.Tensor, lat: torch.Tensor, pt_full: torch.Tensor, n: int, block: int) -> torch.Tensor:
    """``paged [N, 1, bs, 576]`` with the whole blocks of positions ``[0, n)`` holding ``lat`` rows: a cached prefix
    written by another request (host reference latents, so a rewrite by TT shows up bitwise)."""
    out = paged.clone()
    for j in range(n // block):
        out[int(pt_full[j]), 0] = lat[j * block : (j + 1) * block]
    return out


def _row_report(want: torch.Tensor, got: torch.Tensor) -> Dict[str, float]:
    s = stats(want, got)
    r = row_pccs(want, got)
    s.update(
        row_min=float(r.min()), row_p01=float(r.quantile(0.01)) if r.numel() > 1 else float(r.min()),
        row_med=float(r.median()), frac_9995=float((r >= 0.9995).double().mean()), worst_row=int(r.argmin()),
    )  # fmt: skip
    return s


def _fmt_rows(s: Dict[str, float]) -> str:
    return (
        f"pcc {s['pcc']:.6f} rel {s['rel_fro']:.2e}; rows min {s['row_min']:.5f} (row {s['worst_row']}) p01 "
        f"{s['row_p01']:.5f} med {s['row_med']:.6f}, >= 0.9995: {100 * s['frac_9995']:.1f} %"
        + (f" NONFINITE {s['nonfinite']}" if s["nonfinite"] else "")
    )


def _run_chunk(
    mesh_device, cfg, rope, attn, x, host: ChunkHostTables, cache, progs: Dict, *, keep: bool = False, junk=None
):
    """One ``forward_prefill(chunk=)`` from host tables with input rows :func:`chunk_input_rows` (``junk`` = the
    bucket-padding rows); returns the output (kept on the device when ``keep``, else the host rows of chip 0
    ``[C, D]``) and records the programs it compiled under ``(layer kind, cache dtype, path, bucket)``."""
    inp = PrefillChunkInputs.upload(mesh_device, cfg, rope, host)
    rows = chunk_input_rows(x, junk, host.start, host.bucket, host.end)
    x_tt = _replicated(mesh_device, rows[None, None], ttnn.bfloat16)
    ttnn.synchronize_device(mesh_device)
    n0 = mesh_device.num_program_cache_entries()
    o = attn.forward_prefill(x_tt, chunk=inp, kv_cache=cache)
    ttnn.synchronize_device(mesh_device)
    progs.setdefault((attn.spec.attn_kind, _kv_name(cache.dtype), host.path, host.bucket), []).append(
        mesh_device.num_program_cache_entries() - n0
    )
    inp.free()
    if keep:
        return o, x_tt
    out = _dev(o)[0, 0]
    _free([o, x_tt])
    return out


def _run_schedule(
    mesh_device, cfg, rope, attn, old, sched: Schedule, ctx: Dict[str, Any], progs: Dict, tag: str, *, sabotage=None
) -> Tuple[List[str], Dict[str, Any]]:
    """One schedule on a fresh cache (stale rows everywhere; the cached prefix = host reference latents; chunk inputs
    with junk bucket-padding rows). Checks: the first chunk, when sp0, is bitwise the git draft-1 module with the same
    fill table (single shot: bitwise the draft-1 single shot, outputs on 3 chips, cache on 32); every computed row vs
    the fp32 reference; SWA sp1 rows bitwise the draft-1 single shot wherever that is exact
    (:func:`swa_exact_rows_from`: rows past the tail-straddling 128 of every 128-aligned chunk, all of its rows with a
    bf16 cache whose tail TT wrote); the final cache on all 32 chips: the request's own blocks bitwise the TT single
    shot's, the shared blocks and every other block bitwise untouched. ``sabotage(k, (s, e), chunk, host) -> host``
    replaces a chunk's tables (negative controls). Returns ``(failures, results)``."""
    fails: List[str] = []
    bs, kvdt, D = cfg.kv_block_size, ctx["kvdt"], cfg.hidden_size
    S, x, pt_full, pool = sched.end, ctx["x"], ctx["pt_full"], ctx["pool"]
    chunks = schedule_chunks(cfg, sched)
    w0, c0 = chunks[0][1].w0, chunks[0][2].start
    init = ctx["stale"]
    if sched.cached:
        init = _host_quant(_write_prefix(init, ctx["lat_ref"], pt_full, sched.cached, bs), kvdt)
    cache = _upload_cache(mesh_device, init, kvdt)
    got = torch.full((S, D), float("nan"))
    sp1_rows = torch.zeros(S, dtype=torch.bool)
    exact: List[Tuple[int, int, int, bool]] = []  # (chunk start, first exact local row, exact rows, bitwise)
    rep_ok = None
    for k, ((s, e), plan, ch) in enumerate(chunks):
        host = chunk_host_tables(cfg, plan, ch, step_page_table(pt_full, e, bs))
        if sabotage is not None:
            host = sabotage(k, (s, e), ch, host)
        o, x_tt = _run_chunk(mesh_device, cfg, rope, attn, x, host, cache, progs, keep=True, junk=ctx["junk"])
        got[ch.start : ch.end] = _dev(o)[0, 0, : ch.end - ch.start]
        sp1_rows[ch.start : ch.end] |= ch.is_sp1
        lo = swa_exact_rows_from(
            cfg, attn.spec, ch, ctx["S"], prefix_end=sched.cached, bf16_cache=kvdt == ttnn.bfloat16
        )
        if lo is not None and ch.start + lo < ch.end:
            r = slice(ch.start + lo, ch.end)
            exact.append((ch.start, lo, ch.end - ch.start - lo, torch.equal(got[r], ctx["ss_out"][r])))
        if k == 0 and not ch.is_sp1:  # sp0: bitwise draft 1
            if sched.name == "single":
                eq_out = torch.equal(_chips(o), ctx["ss_chips"])
                eq_cache = torch.equal(_all_chips(cache, mesh_device), ctx["ss_cache"])
                where = "draft-1 single shot (cache on 32 chips)"
            else:
                c_d1 = _upload_cache(mesh_device, init, kvdt)
                pt_d1 = _rep_i32(mesh_device, host.fill)
                o_d1 = old.forward_prefill(x_tt, page_table=pt_d1, kv_cache=c_d1)
                eq_out = torch.equal(_chips(o), _chips(o_d1))
                eq_cache = torch.equal(_chips(cache), _chips(c_d1))
                _free([o_d1, c_d1, pt_d1])
                where = f"git draft-1 forward_prefill(page_table=fill table {host.fill[0, :3].tolist()}...)"
            log(f"{tag} {sched.name}: sp0 chunk (0, {ch.bucket}) vs {where}: output bitwise {eq_out}, cache {eq_cache}")
            if not (eq_out and eq_cache):
                fails.append(f"{tag} {sched.name}: sp0 chunk not bitwise draft 1 (output {eq_out}, cache {eq_cache})")
        if k == len(chunks) - 1:
            r_ = _chips(o)
            rep_ok = all(torch.equal(r_[0], r_[i]) for i in range(1, r_.shape[0]))
        _free([o, x_tt])
    hc = _all_chips(cache, mesh_device)
    ss_cache = ctx["ss_cache"]
    nreq = cdiv(S, bs)
    shared, own = pt_full[: w0 // bs].long(), pt_full[w0 // bs : nreq].long()
    others = torch.tensor(sorted(set(range(pool)) - set(pt_full[:nreq].tolist())), dtype=torch.long)
    own_eq = torch.equal(hc[:, :, own], ss_cache[:, :, own])
    own_diff = float((hc[:, :, own] - ss_cache[:, :, own]).abs().max())
    shared_eq = torch.equal(hc[:, :, shared], init[shared].expand_as(hc[:, :, shared]))
    others_eq = torch.equal(hc[:, :, others], init[others].expand_as(hc[:, :, others]))
    s_ref = _row_report(ctx["want"][c0:], got[c0:])
    r_ss = row_pccs(ctx["ss_out"][c0:], got[c0:])
    s_ss = stats(ctx["ss_out"][c0:], got[c0:])
    exact_ok = all(eq for *_, eq in exact)
    msg = (
        f"{tag} {sched.name}: chunks {[(c.start, c.bucket, c.path) for _, _, c in chunks][:5]}"
        f"{' ...' if len(chunks) > 5 else ''} (w0 {w0}, c0 {c0}); rows [{c0}, {S}) vs fp32 ref {_fmt_rows(s_ref)}"
    )
    if sp1_rows.any():
        msg += f"; sp1 rows only: {_fmt_rows(_row_report(ctx['want'][sp1_rows], got[sp1_rows]))}"
    msg += (
        f"; vs TT single shot pcc {s_ss['pcc']:.6f} (rows min {float(r_ss.min()):.5f})"
        + (
            f"; SWA rows bitwise the single shot: {sum(n for _, _, n, _ in exact)} rows of {len(exact)} chunks "
            f"(from local row {sorted({lo for _, lo, _, _ in exact})}) {exact_ok}"
            if exact
            else ""
        )
        + f"; last-chunk replicas {rep_ok}; final cache (32 chips): own {len(own)} blocks == single shot bitwise "
        f"{own_eq} (max |diff| {own_diff:.2e}), shared {len(shared)} untouched {shared_eq}, other {len(others)} "
        f"untouched {others_eq}"
    )
    log(msg)
    rows_ok = s_ref["pcc"] >= SP1_OUT_PCC_MIN and s_ref["row_min"] >= SP1_ROW_PCC_MIN and _good(s_ref, 0.0)
    if not rows_ok:
        fails.append(f"{tag} {sched.name}: rows vs reference {_fmt_rows(s_ref)}")
    if not exact_ok:
        fails.append(f"{tag} {sched.name}: SWA rows not bitwise the single shot (start, from, rows, eq): {exact}")
    if not (own_eq and shared_eq and others_eq and rep_ok):
        fails.append(f"{tag} {sched.name}: cache own {own_eq} shared {shared_eq} others {others_eq}, replicas {rep_ok}")
    _free(cache)
    res = dict(rows_ok=rows_ok, exact_ok=exact_ok, own_eq=own_eq, shared_eq=shared_eq, others_eq=others_eq)
    res.update(n_exact=sum(n for _, _, n, _ in exact), chunks=len(chunks))
    return fails, res


def _negative_controls(mesh_device, cfg, rope, attn, old, ctx: Dict[str, Any], progs: Dict, tag: str) -> List[str]:
    """(1) The cached-prefix chunk ``(S/2, S)`` of "half_cached+half" once with its own tables and once broken:
    global, the cached prefix's SDPA-table entries pointed at the null block; SWA, the two tail blocks pointed at the
    null block. Both tables are refused by ``ChunkHostTables`` (checked) and are run unchecked to show what that
    prevents: the broken run must be detected (rows vs reference below :data:`NEG_PCC_MAX`); for SWA only the chunk's
    first 128 rows may change (their window reaches into the tail), the later rows stay bitwise equal. (2) The
    "unaligned" schedule whose continuation chunk does not rewrite the request's own partial block
    (:func:`skip_own_partial_block`, a self-consistent table): with junk padding rows the schedule check must fail --
    own blocks hold junk latents (both kinds) and, on the global layer, the rows attend them."""
    bs, kvdt, S = cfg.kv_block_size, ctx["kvdt"], DEVICE_S
    h = S // 2
    init = _host_quant(_write_prefix(ctx["stale"], ctx["lat_ref"], ctx["pt_full"], h, bs), kvdt)
    plan = cfg.plan_prefill_row(h, S)
    ch = plan.chunks[0]
    host = chunk_host_tables(cfg, plan, ch, step_page_table(ctx["pt_full"], S, bs))
    glob = attn.window is None
    change = (
        dict(sdpa=host.sdpa.clone().index_fill_(1, torch.arange(h // bs), 0))
        if glob
        else dict(tail=torch.zeros_like(host.tail))
    )
    try:
        dataclasses.replace(host, **change)
        refused = False
    except ValueError:
        refused = True
    bad = _unchecked_replace(host, **change)
    outs = []
    for tb in (host, bad):
        cache = _upload_cache(mesh_device, init, kvdt)
        outs.append(_run_chunk(mesh_device, cfg, rope, attn, ctx["x"], tb, cache, progs, junk=ctx["junk"]))
        outs[-1] = outs[-1][: ch.end - ch.start]
        _free(cache)
    good, wrong = outs
    want = ctx["want"][ch.start : ch.end]
    T = cfg.prefill_swa_tail
    if glob:
        s = stats(want, wrong)
        ok = s["pcc"] < NEG_PCC_MAX and refused
        log(
            f"{tag} negative control (cached prefix -> null block in the SDPA table; refused by the host tables "
            f"{refused}): rows {fmt(s)}; detected {ok}"
        )
    else:
        s = stats(want[:T], wrong[:T])
        same_after = torch.equal(wrong[T:], good[T:])
        ok = s["pcc"] < NEG_PCC_MAX and same_after and refused
        log(
            f"{tag} negative control (tail -> null block; refused by the host tables {refused}): rows [a, a+{T}) "
            f"{fmt(s)}; rows [a+{T}, e) bitwise unchanged {same_after}; detected {ok}"
        )
    fails = [] if ok else [f"{tag}: negative control not detected ({fmt(s)}, host tables refused {refused})"]
    sched = next(sc for sc in DEVICE_SCHEDULES if sc.name.startswith("unaligned"))
    f_sab, res = _run_schedule(
        mesh_device, cfg, rope, attn, old, sched, ctx, progs, f"{tag} [sabotage: own partial block not rewritten]",
        sabotage=skip_own_partial_block(bs),
    )  # fmt: skip
    caught = not res["own_eq"] and (res["rows_ok"] is False or not glob)
    log(
        f"{tag} negative control (continuation skips the own partial block, junk padding): schedule check fails "
        f"{bool(f_sab)} (own blocks == single shot {res['own_eq']}, rows ok {res['rows_ok']}); detected {caught}"
    )
    if not caught:
        fails.append(f"{tag}: own-partial-block sabotage not detected: {res}")
    return fails


def _resumed_case(
    mesh_device, cfg, ccl, rope, d1, layer: int, weights_kind: str, progs: Dict, kv_dtype=None
) -> List[str]:
    from models.demos.motif3.tt.decoder import free_tensors

    t0 = time.time()
    args = ref_args()
    kvdt = kv_dtype if kv_dtype is not None else cfg.dtypes.kv_cache
    tag = f"L{layer} {cfg.layer(layer).attn_kind} {weights_kind}" + (
        "" if kv_dtype is None else f" {_kv_name(kvdt)}-KV"
    )
    if weights_kind == "real":
        tensors = {k: v.float() for k, v in real_attn_tensors(layer).items()}
    else:
        tensors = random_attn_tensors(args, seed=900 + layer)
    src = source_for(tensors, W.hf_name(layer, "self_attn"))
    attn = MotifAttention(mesh_device, cfg, layer, source=src, ccl=ccl, rope=rope, cache=False)
    old = d1.MotifAttention(mesh_device, cfg, layer, source=src, ccl=ccl, rope=rope, cache=False)
    ref = ref_attention_spec(args, layer, tensors)
    S, bs = DEVICE_S, cfg.kv_block_size
    rows = schedule_rows(cfg, DEVICE_SCHEDULES)
    x = _inputs(layer, weights_kind, rows, seed=910 + layer)
    junk = _junk_rows(layer, weights_kind, rows, seed=940 + layer)
    with torch.no_grad():
        want = ref(x[None, :S], torch.arange(S)[None])[0]
        n_ref, kpe_ref = ref_latents(ref, x[:S], torch.arange(S))
    pool = 1 + cdiv(rows, bs) + 8
    pt_full = _page_table(pool, S // bs, seed=920 + layer)
    bank = stale_bank(ref, weights_kind, layer, n=512, seed=layer)
    stale = _host_quant(stale_paged(bank, pool, bs, seed=930 + layer), kvdt)
    log(f"{tag}: reference ({S} rows) + stale pool of {pool} blocks ({_kv_name(kvdt)}) in {time.time() - t0:.1f} s")

    # ---- TT single shot (git draft 1): the bitwise reference of the cache rows and of the sp0 path --------------
    c_ss = _upload_cache(mesh_device, stale, kvdt)
    x_ss = _replicated(mesh_device, x[:S][None, None], ttnn.bfloat16)
    pt_ss = _rep_i32(mesh_device, pt_full[None])
    o_ss = old.forward_prefill(x_ss, page_table=pt_ss, kv_cache=c_ss)
    ctx = dict(x=x, junk=junk, want=want, lat_ref=torch.cat([n_ref, kpe_ref], -1), pt_full=pt_full, pool=pool)
    ctx.update(stale=stale, kvdt=kvdt, S=S)
    ctx.update(ss_out=_dev(o_ss)[0, 0], ss_chips=_chips(o_ss), ss_cache=_all_chips(c_ss, mesh_device))
    log(f"{tag} TT single shot (draft 1, S={S}) vs fp32 ref: {_fmt_rows(_row_report(want, ctx['ss_out']))}")
    _free([o_ss, x_ss, pt_ss, c_ss])

    fails: List[str] = []
    for sched in DEVICE_SCHEDULES:
        fails += _run_schedule(mesh_device, cfg, rope, attn, old, sched, ctx, progs, tag)[0]
    if weights_kind == "random" and kv_dtype is None:
        fails += _negative_controls(mesh_device, cfg, rope, attn, old, ctx, progs, tag)
    free_tensors(old)
    free_tensors(attn)
    log(f"{tag}: done in {time.time() - t0:.1f} s")
    return fails


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_wp2b_resumed_prefill(mesh_device, device_params):
    """Resumed (sp1) prefill of global L0 and SWA L1 against the 4096-token single shot (draft 1) and the fp32
    reference, over the features design §5.2 schedules (:data:`DEVICE_SCHEDULES`): two halves; half cached (by another
    request) + half; 32 chunks of 128; hit 960 planned with A = 128 (c0 896 < w0 960: the shared block is recomputed,
    read, never written); an unaligned chunk end 1348 (its padding rows are junk, so the continuation must rewrite
    block 21, the request's own partial block); a one-block hit (sp0, block 0 shared). Random and real weights (layers
    0-1, BF16 on disk; real weights with the real 4096-token inputs) on the bfp8 cache, and random weights on a bf16
    cache. The SWA window straddles every sp1 chunk start (its first 128 rows read the cached tail).

    Checks per schedule: rows vs the reference (aggregate PCC >= 0.9995, every row >= 0.999: :data:`SP1_ROW_PCC_MIN`);
    SWA sp1 rows bitwise the draft-1 single shot wherever that is exact (rows past the first 128 of every 128-aligned
    chunk; with the bf16 cache all rows of the chunks whose tail TT wrote); sp0 chunks bitwise draft 1; the final cache
    on 32 chips: own blocks bitwise the single shot's, shared and other blocks untouched. Also: no program compiled
    for a later start of a ``(layer kind, cache dtype, path, bucket)``; negative controls (broken tail / cached prefix,
    refused by the host tables and detected when run; a continuation that skips the own partial block, detected)."""
    d1 = draft1_attention_module()
    cfg, ccl, rope = _setup(mesh_device, "wp2b resumed prefill")
    torch.set_num_threads(max(8, min(32, (os.cpu_count() or 8) // 2)))
    progs: Dict[Tuple[str, str, str, int], List[int]] = {}
    failures: List[str] = []
    for weights_kind, kv_dtype in (("random", None), ("real", None), ("random", ttnn.bfloat16)):
        for layer in (0, 1):
            failures += _resumed_case(mesh_device, cfg, ccl, rope, d1, layer, weights_kind, progs, kv_dtype=kv_dtype)
    later = {k: v[1:] for k, v in progs.items() if any(n != 0 for n in v[1:])}
    summary = {k: (v[0], len(v) - 1, max(v[1:], default=0)) for k, v in sorted(progs.items())}
    log(
        f"programs compiled per (layer kind, cache, path, bucket): (first call, later calls, max of the later) {summary}"
    )
    if later:
        failures.append(f"programs compiled after the first chunk of a (layer kind, cache, path, bucket): {later}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_wp2b_warmup_compiles_every_program(mesh_device, device_params):
    """Design D12 as the generator uses it: on a fresh program cache, the warm-up of ``(path, C)``
    (``PrefillChunkInputs.warmup`` + ``forward_prefill(chunk=)`` + ``fill_kv(chunk=)``, the MTP KV-only fill)
    compiles every program a real chunk of that ``(path, C)`` needs: a real chunk afterwards (sp0 at start 0; sp1 at
    start 2048 over a cached prefix; its inputs uploaded, ``forward_prefill`` and ``fill_kv``) compiles none. The
    warm-up writes nothing (cache bitwise untouched on 3 chips; prefill writes are replicated) and its outputs are
    finite. Global L0 and SWA L1 (random weights), buckets 256 and 8192: the schedule test's "no program at a later
    start" check sees only buckets that run at two starts there."""
    from models.demos.motif3.tt.decoder import free_tensors

    cfg, ccl, rope = _setup(mesh_device, "wp2b warm-up")
    args = ref_args()
    bs, kvdt, A_ = cfg.kv_block_size, cfg.dtypes.kv_cache, cfg.prefill_resume_alignment
    Cs, a_real = (256, 8192), 2048
    rows = a_real + max(Cs)
    pool = 1 + cdiv(rows, bs) + 4  # one pool shape for warm-up and real chunks (as in serving)
    pt_full = _page_table(pool, cdiv(rows, bs), seed=1200)
    x = _bf16_rows(rows, seed=1201)
    g = torch.Generator().manual_seed(1202)
    paged = _host_quant(torch.randn(pool, 1, bs, cfg.kv_latent_dim, generator=g).bfloat16().float(), kvdt)
    failures: List[str] = []
    for layer in (0, 1):
        tag = f"L{layer} {cfg.layer(layer).attn_kind}"
        tensors = random_attn_tensors(args, seed=1210 + layer)
        attn = MotifAttention(mesh_device, cfg, layer, source=source_for(tensors, W.hf_name(layer, "self_attn")),
                              ccl=ccl, rope=rope, cache=False)  # fmt: skip
        for C in Cs:
            for path in ("sp0", "sp1"):
                cache = _upload_cache(mesh_device, paged, kvdt)
                xw = _replicated(mesh_device, x[:C][None, None], ttnn.bfloat16)
                ttnn.synchronize_device(mesh_device)
                n0 = mesh_device.num_program_cache_entries()
                w = PrefillChunkInputs.warmup(mesh_device, cfg, rope, path, C)
                o = attn.forward_prefill(xw, chunk=w, kv_cache=cache)
                attn.fill_kv(xw, chunk=w, kv_cache=cache)
                ttnn.synchronize_device(mesh_device)
                n_warm = mesh_device.num_program_cache_entries() - n0
                fin_w = bool(torch.isfinite(_dev(o)).all())
                untouched = torch.equal(_chips(cache), paged.expand(len(CHIPS), *paged.shape))
                a_w = w.start
                _free([o, xw])
                w.free()
                a = 0 if path == "sp0" else a_real  # a real chunk of the same (path, bucket)
                plan = explicit_plan(a, a + C, [(a, C)], block=bs, align=A_)
                host = chunk_host_tables(cfg, plan, plan.chunks[0], step_page_table(pt_full, a + C, bs))
                xc = _replicated(mesh_device, x[a : a + C][None, None], ttnn.bfloat16)
                ttnn.synchronize_device(mesh_device)
                n1 = mesh_device.num_program_cache_entries()
                inp = PrefillChunkInputs.upload(mesh_device, cfg, rope, host)
                o = attn.forward_prefill(xc, chunk=inp, kv_cache=cache)
                attn.fill_kv(xc, chunk=inp, kv_cache=cache)
                ttnn.synchronize_device(mesh_device)
                n_real = mesh_device.num_program_cache_entries() - n1
                fin_r = bool(torch.isfinite(_dev(o)).all())
                _free([o, xc, cache])
                inp.free()
                log(
                    f"{tag} warm-up ({path}, {C}) at start {a_w}: compiled {n_warm} programs, cache untouched "
                    f"{untouched}, finite {fin_w}; then a real chunk ({a}, {C}) + fill_kv compiled {n_real} "
                    f"(must be 0), finite {fin_r}"
                )
                if n_real or not (untouched and fin_w and fin_r):
                    failures.append(
                        f"{tag} ({path}, {C}): real chunk compiled {n_real} programs after the warm-up; warm-up "
                        f"untouched {untouched} finite {fin_w}; real finite {fin_r}"
                    )
        free_tensors(attn)
    assert not failures, "\n".join(failures)


def _decode_diff(a: torch.Tensor, b: torch.Tensor, cfg: MotifTTConfig, pos: Sequence[int]) -> str:
    """Where two decode outputs ``[R, C, 1, 1, 8, D]`` (all chips) differ: NaN / Inf counts and the lanes (NaN-aware
    comparison), split into active and inactive lanes."""
    same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    lanes = []
    for dp in range(cfg.dp):
        for l in range(cfg.lanes_per_row):
            r_c = [cfg.axes.coord(dp, tp) for tp in range(cfg.tp)]
            if not all(bool(same[r, c, 0, 0, l].all()) for r, c in r_c):
                lanes.append(cfg.lanes_per_row * dp + l)
    fin = torch.isfinite(a) & torch.isfinite(b)
    md = float((a - b)[fin].abs().max()) if bool(fin.any()) else float("nan")
    return (
        f"NaN {int(torch.isnan(a).sum())}/{int(torch.isnan(b).sum())}, Inf {int(torch.isinf(a).sum())}/"
        f"{int(torch.isinf(b).sum())}, differing active lanes {[l for l in lanes if pos[l] >= 0]}, inactive "
        f"{[l for l in lanes if pos[l] < 0]}, max |diff| (finite) {md:.3e}"
    )


class _RowWriter:
    """Test double of ``tt/kv_write.py``'s ``row`` mode (draft 1 through the hook): this DP row's 8 lanes, one
    ``paged_update_cache`` through the per-row ``cur_pos`` / ``page_table``."""

    def __init__(self, attn: MotifAttention):
        self.mc = attn.update_mc
        self.calls = 0

    def write(self, kv_row, kv_cache, *, cur_pos, page_table):
        self.calls += 1
        u = ttnn.transpose(kv_row, 1, 2, memory_config=self.mc)
        ttnn.experimental.paged_update_cache(kv_cache, u, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.deallocate(u)


class _AllWriter:
    """Test double of the KV-R ``all`` mode (gate G13a recipe, design §3.5): ``ccl.ag_dp_rows`` of the row's latent
    ``[1, 1, 8, 576]`` -> ``[1, 1, 32, 576]`` (lane order) -> 32-core height-sharded -> one 32-lane update with the
    replicated ``cur_all [32]`` / ``pt_all [32, W]``: every chip writes every lane."""

    def __init__(self, mesh_device, cfg: MotifTTConfig, ccl, positions: Sequence[int], pt: torch.Tensor):
        grid = mesh_device.compute_with_storage_grid_size()
        self.ccl, self.calls = ccl, 0
        self.mc32 = ttnn.create_sharded_memory_config(
            shape=(32, cfg.kv_latent_dim), core_grid=ttnn.num_cores_to_corerangeset(cfg.max_batch, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )  # fmt: skip
        self.cur_all = _rep_i32(mesh_device, torch.tensor(list(positions), dtype=torch.int32))
        self.pt_all = _rep_i32(mesh_device, pt.contiguous())

    def write(self, kv_row, kv_cache, *, cur_pos, page_table):
        self.calls += 1
        g = self.ccl.ag_dp_rows(kv_row)  # [1, 1, 32, 576], lane order 8 dp + l
        u = ttnn.transpose(g, 1, 2, memory_config=self.mc32)
        ttnn.deallocate(g)
        ttnn.experimental.paged_update_cache(kv_cache, u, update_idxs_tensor=self.cur_all, page_table=self.pt_all)
        ttnn.deallocate(u)

    def free(self):
        _free([self.cur_all, self.pt_all])


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_wp2b_decode_kv_write_hook(mesh_device, device_params):
    """``forward_decode(kv_write=)`` on global L0 and SWA L1 (random weights, 32 lanes, 2 inactive): a writer that
    reproduces the draft-1 write gives bitwise the ``kv_write=None`` output and cache (32 chips); a KV-R writer
    (``ag_dp_rows`` + one 32-lane update, the G13a recipe) gives the same output bitwise and a cache that holds every
    lane's row on every chip (identical on all 32 chips: the KV-R invariant)."""
    from models.demos.motif3.tt.decoder import free_tensors

    cfg, ccl, rope = _setup(mesh_device, "wp2b decode kv_write hook")
    args = ref_args()
    block, kvdt = cfg.kv_block_size, cfg.dtypes.kv_cache
    failures: List[str] = []
    pos = list(DRAFT1_DECODE_POS)
    Wd = decode_page_table_width(pos, block)
    pool = 1 + cfg.max_batch * Wd
    perm = _page_table(pool, pool - 1, seed=177)
    pt = torch.zeros(cfg.max_batch, Wd, dtype=torch.int32)
    for lane, p in enumerate(pos):
        if p >= 0:
            pt[lane] = perm[lane * Wd : (lane + 1) * Wd]
    for layer in (0, 1):
        tag = f"L{layer} {cfg.layer(layer).attn_kind}"
        tensors = random_attn_tensors(args, seed=950 + layer)
        attn = MotifAttention(mesh_device, cfg, layer, source=source_for(tensors, W.hf_name(layer, "self_attn")),
                              ccl=ccl, rope=rope, cache=False)  # fmt: skip
        ref = ref_attention_spec(args, layer, tensors)
        stale = _host_quant(
            stale_paged(stale_bank(ref, "random", layer, n=256, seed=layer), pool, block, seed=178), kvdt
        )
        d = _decode_step_inputs(mesh_device, cfg, rope, pos, _bf16_rows(cfg.max_batch, seed=179 + layer), pt)
        kw = dict(rot=d["rot"], cur_pos=d["cur"], page_table=d["pt"], active=d["act"])
        caches = {k: _upload_cache(mesh_device, stale, kvdt) for k in ("none", "row", "all")}
        row_w, all_w = _RowWriter(attn), _AllWriter(mesh_device, cfg, ccl, pos, pt)
        outs = {
            "none": attn.forward_decode(d["x"], kv_cache=caches["none"], **kw),
            "row": attn.forward_decode(d["x"], kv_cache=caches["row"], kv_write=row_w, **kw),
            "all": attn.forward_decode(d["x"], kv_cache=caches["all"], kv_write=all_w, **kw),
        }
        h = {k: _all_chips(c, mesh_device) for k, c in caches.items()}
        o = {k: _all_chips(t, mesh_device) for k, t in outs.items()}
        eq_row = torch.equal(o["row"], o["none"]) and torch.equal(h["row"], h["none"])
        eq_all_out = torch.equal(o["all"], o["none"])
        # KV-R expectation: every chip = the stale pool with every active lane's row (the row-mode writes of its DP row)
        want = stale.clone()
        for lane, p in enumerate(pos):
            if p >= 0:
                r, c = cfg.axes.coord(cfg.lane_row(lane), 0)
                b = int(pt[lane, p // block])
                want[b, 0, p % block] = h["none"][r, c, b, 0, p % block]
        kvr = torch.equal(h["all"], want.expand_as(h["all"]))
        log(
            f"{tag} decode kv_write hook: draft-1 writer == kv_write=None bitwise (output + cache, 32 chips) {eq_row} "
            f"({row_w.calls} call); KV-R writer: output == row mode bitwise {eq_all_out}, every chip holds all "
            f"{sum(p >= 0 for p in pos)} active lanes' rows (32 chips identical, bitwise) {kvr}"
        )
        if not (eq_row and eq_all_out):
            d_row, d_all = _decode_diff(o["row"], o["none"], cfg, pos), _decode_diff(o["all"], o["none"], cfg, pos)
            log(f"{tag} output differences: row vs none {d_row}; all vs none {d_all}")
        if not (eq_row and eq_all_out and kvr and row_w.calls == 1 and all_w.calls == 1):
            failures.append(f"{tag}: row writer {eq_row}, KV-R output {eq_all_out}, KV-R cache {kvr}")
        _free(list(outs.values()) + list(caches.values()))
        all_w.free()
        _free_step(d)
        free_tensors(attn)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_wp2b_sp1_trace_replay(mesh_device, device_params):
    """An sp1 chunk is trace-safe end to end (global L0 and SWA L1, random weights, bucket 512): capture
    ``rope.chunk_rope_tables(rot_idx)`` + ``forward_prefill(chunk=)`` at start 1024 with persistent inputs, rewrite
    them in place for start 2048 (``PrefillChunkInputs.write(regather=False)`` + the input rows), replay: output (3
    chips) and cache (2 chips) bitwise equal to an eager call at 2048, no program compiled during the capture (the
    start, the block ids and the RoPE rows are device data: one program set per ``(path, bucket)``, design D12). After
    the rewrite the inputs hold no eager RoPE rows (``rot is None``): an eager call refuses them instead of roping at
    the old start."""
    from models.demos.motif3.tt.decoder import free_tensors

    cfg, ccl, rope = _setup(mesh_device, "wp2b sp1 trace replay")
    args = ref_args()
    bs, kvdt = cfg.kv_block_size, cfg.dtypes.kv_cache
    C, starts = 512, (1024, 2048)
    failures: List[str] = []
    for layer in (0, 1):
        tag = f"L{layer} {cfg.layer(layer).attn_kind}"
        tensors = random_attn_tensors(args, seed=960 + layer)
        attn = MotifAttention(mesh_device, cfg, layer, source=source_for(tensors, W.hf_name(layer, "self_attn")),
                              ccl=ccl, rope=rope, cache=False)  # fmt: skip
        ref = ref_attention_spec(args, layer, tensors)
        rows = starts[1] + C
        x = _bf16_rows(rows, seed=970 + layer)
        with torch.no_grad():
            want = ref(x[None], torch.arange(rows)[None])[0]
            n_ref, kpe_ref = ref_latents(ref, x, torch.arange(rows))
        pool = 1 + rows // bs + 4
        pt_full = _page_table(pool, rows // bs, seed=980 + layer)
        stale = stale_paged(stale_bank(ref, "random", layer, n=256, seed=layer), pool, bs, seed=990 + layer)
        init = _host_quant(_write_prefix(stale, torch.cat([n_ref, kpe_ref], -1), pt_full, starts[1], bs), kvdt)
        hosts = []
        for a in starts:
            plan = explicit_plan(a, a + C, [(a, C)], block=bs, align=cfg.prefill_resume_alignment)
            hosts.append(chunk_host_tables(cfg, plan, plan.chunks[0], step_page_table(pt_full, a + C, bs)))
        inp = PrefillChunkInputs.upload(mesh_device, cfg, rope, hosts[0])  # persistent inputs
        x_dev = _replicated(mesh_device, x[starts[0] : starts[0] + C][None, None], ttnn.bfloat16)
        scratch = _upload_cache(mesh_device, init, kvdt)
        _free([attn.forward_prefill(x_dev, chunk=inp, kv_cache=scratch), scratch])  # compile before the capture
        c_tr = _upload_cache(mesh_device, init, kvdt)
        ttnn.synchronize_device(mesh_device)
        n0 = mesh_device.num_program_cache_entries()
        with _Capture(mesh_device) as cap:
            rot = rope.chunk_rope_tables(inp.rot_idx)
            o_tr = attn.forward_prefill(x_dev, chunk=dataclasses.replace(inp, rot=rot), kv_cache=c_tr)
        n_cap = mesh_device.num_program_cache_entries() - n0
        inp.write(mesh_device, cfg, rope, hosts[1], regather=False)
        dropped = inp.rot is None
        try:  # the old start's eager RoPE rows are gone: an eager call must refuse the inputs, before any device op
            attn.forward_prefill(x_dev, chunk=inp, kv_cache=c_tr)
            refused = False
        except ValueError:
            refused = True
        x_host = ttnn.from_torch(x[starts[1] : starts[1] + C][None, None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))  # fmt: skip
        ttnn.copy_host_to_device_tensor(x_host, x_dev)
        try:
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            got, got_cache = _chips(o_tr), _chips(c_tr, (0, 31))
        finally:
            ttnn.release_trace(mesh_device, cap.tid)
        c_e = _upload_cache(mesh_device, init, kvdt)
        progs: Dict = {}
        o_e, x_e = _run_chunk(mesh_device, cfg, rope, attn, x, hosts[1], c_e, progs, keep=True)
        eq_out, eq_cache = torch.equal(got, _chips(o_e)), torch.equal(got_cache, _chips(c_e, (0, 31)))
        s = stats(want[starts[1] : starts[1] + C], got[0, 0, 0])
        log(
            f"{tag} sp1 trace: captured at {starts[0]}, replayed at {starts[1]}: output == eager bitwise {eq_out}, "
            f"cache == eager bitwise {eq_cache}; programs during capture {n_cap}, eager call at the new start "
            f"{list(progs.values())}; replayed rows vs fp32 ref {fmt(s)}; write(regather=False) dropped the eager "
            f"RoPE rows {dropped}, eager call with them refused {refused}"
        )
        ok = eq_out and eq_cache and n_cap == 0 and list(progs.values()) == [[0]] and _good(s, OUT_PCC_MIN)
        if not (ok and dropped and refused):
            failures.append(
                f"{tag}: trace replay output {eq_out} cache {eq_cache} programs {n_cap} / {progs}; rot dropped "
                f"{dropped}, eager call refused {refused}"
            )
        _free([o_tr, c_tr, o_e, x_e, c_e, x_dev] + [t for cs in rot.values() for t in cs])
        inp.free()
        free_tensors(attn)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_wp2b_sp1_cost(mesh_device, device_params):
    """Report: eager wall time of one attention layer call (synchronized, min of 3 after a warm-up call), sp0 vs sp1,
    global L0 and SWA L1 (random weights, bfp8 cache), buckets 128 / 512 / 2048 / 8192, sp1 at starts 128 and 8192 (the
    prefix the global chunked SDPA streams). The op-level costs are gates G9 / G10's; this is the module (projections,
    fill, CCL and dispatch included) for the chunk cost model (``cfg.prefill_cost_table``,
    ``prefill_plan.prefill_cost_model``). Asserts only finite outputs and no program compiled by a timed call."""
    from models.demos.motif3.tt.decoder import free_tensors

    cfg, ccl, rope = _setup(mesh_device, "wp2b sp1 cost")
    args = ref_args()
    bs, kvdt = cfg.kv_block_size, cfg.dtypes.kv_cache
    Cs, starts = (128, 512, 2048, 8192), (128, 8192)
    rows = max(starts) + max(Cs)
    pool = 1 + rows // bs + 4
    pt_full = _page_table(pool, rows // bs, seed=1100)
    g = torch.Generator().manual_seed(1101)
    paged = torch.randn(pool, 1, bs, cfg.kv_latent_dim, generator=g).bfloat16().float()
    x = _bf16_rows(rows, seed=1102)
    table: Dict[str, Dict[str, float]] = {}
    failures: List[str] = []
    for layer in (0, 1):
        kind = cfg.layer(layer).attn_kind
        tensors = random_attn_tensors(args, seed=1110 + layer)
        attn = MotifAttention(mesh_device, cfg, layer, source=source_for(tensors, W.hf_name(layer, "self_attn")),
                              ccl=ccl, rope=rope, cache=False)  # fmt: skip
        cache = _upload_cache(mesh_device, paged, kvdt)
        for C in Cs:
            for a in (0,) + starts:
                plan = explicit_plan(a, a + C, [(a, C)], block=bs, align=cfg.prefill_resume_alignment)
                host = chunk_host_tables(cfg, plan, plan.chunks[0], step_page_table(pt_full, a + C, bs))
                inp = PrefillChunkInputs.upload(mesh_device, cfg, rope, host)
                x_tt = _replicated(mesh_device, x[a : a + C][None, None], ttnn.bfloat16)
                o = attn.forward_prefill(x_tt, chunk=inp, kv_cache=cache)  # warm-up / compile
                finite = bool(torch.isfinite(_dev(o)).all())
                _free(o)
                ttnn.synchronize_device(mesh_device)
                n0, times = mesh_device.num_program_cache_entries(), []
                for _ in range(3):
                    t0 = time.perf_counter()
                    _free(attn.forward_prefill(x_tt, chunk=inp, kv_cache=cache))
                    ttnn.synchronize_device(mesh_device)
                    times.append((time.perf_counter() - t0) * 1e3)
                dn = mesh_device.num_program_cache_entries() - n0
                key = f"{host.path}@{a}" if a else "sp0"
                table.setdefault(f"{kind} C={C}", {})[key] = min(times)
                if not finite or dn:
                    failures.append(f"{kind} C={C} {key}: finite {finite}, programs compiled by timed calls {dn}")
                _free(x_tt)
                inp.free()
        _free(cache)
        free_tensors(attn)
    for k, v in table.items():
        log(f"cost {k}: " + ", ".join(f"{p} {ms:.2f} ms" for p, ms in v.items()))
    assert not failures, "\n".join(failures)
