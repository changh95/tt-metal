# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden inputs/outputs for TT module tests.

Every reference module reports its intermediates through ``tap(name, tensor)``; :class:`TensorRecorder` collects
them (detached CPU clones). The two entry points cover the common cases:

* :func:`capture_layer_goldens` - one decoder layer on given (or :func:`random_streams`) 4-stream inputs:
  prefill of the first ``S - decode_steps`` tokens with a fresh latent KV cache, then ``decode_steps`` single-token
  decode steps against that cache. Captures mHC pre/post (h_pre, h_post, H_res, reduced input, combined output),
  attention internals (q, roped q_pe / k_pe, cached latent c, lambda, gate, per-head outputs, output), dense MLP or
  router (logits / scores / indices / weights) + MoE outputs, and the layer output.
* :func:`capture_model_goldens` - a (prefix) model on real token ids (e.g. a chat-templated prompt): per-layer
  inputs/outputs with realistic activations, every module tap, final stream mean / norm / logits, plus decode
  steps.

A golden is a plain dict ``{"meta": {...}, "prefill": {name: tensor}, "decode": [{name: tensor}, ...]}`` saved
with :func:`save_golden` (``torch.save``) and loadable with ``torch.load(path, weights_only=True)``.

Tap names (per decoder layer, prefix ``layers.{i}.`` in model goldens):
  x_in, x_mid, x_out                       4-stream residual [B,S,4,D] in/after-attention/out
  mhc_attn.{mixes,h_pre,h_post,h_res,x_reduced}, input_layernorm.out
  self_attn.{q_latent,q_nope,q_pe,c_kv,k_pe,lambda,gate,attn_heads|q_lat+attn_latent_heads+diff_latent,diff,gated,out}
  mhc_ffn.{mixes,h_pre,h_post,h_res,x_reduced}, post_attention_layernorm.out
  mlp.{gate,up,act,out}  |  moe.router.{logits,scores,indices,weights}, moe.routed_out,
                             moe.shared_experts.{gate,up,act,out}, moe.out
Model level: embed, final.stream_mean, final.norm, logits.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Sequence

import torch

from .cache import LatentKVCache
from .config import MotifArgs
from .modules import DecoderLayer, MotifForCausalLM


class TensorRecorder:
    """``tap`` callable storing detached CPU clones, optionally filtered by name."""

    def __init__(self, include: Optional[Callable[[str], bool]] = None):
        self.tensors: Dict[str, torch.Tensor] = {}
        self.include = include

    def __call__(self, name: str, t: torch.Tensor) -> None:
        if self.include is None or self.include(name):
            self.tensors[name] = t.detach().to("cpu").clone()


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two tensors (fp64)."""
    a = a.detach().double().flatten()
    b = b.detach().double().flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float((a @ b) / denom)


def _layer_meta(args: MotifArgs, layer_idx: int) -> dict:
    return dict(
        layer_idx=layer_idx,
        kind=args.layer_kind(layer_idx),
        is_swa=args.is_swa_layer(layer_idx),
        window=args.attention_window(layer_idx),
        softmax_scale=args.softmax_scale(layer_idx),
        uses_yarn=args.uses_yarn(layer_idx),
        is_moe=args.is_moe_layer(layer_idx),
    )


def _args_meta(args: MotifArgs) -> dict:
    d = asdict(args)
    d["eos_token_ids"] = list(d["eos_token_ids"])
    return d


def random_streams(
    args: MotifArgs,
    batch: int,
    seq_len: int,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    embed_weight: Optional[torch.Tensor] = None,
    stream_noise: float = 0.5,
) -> torch.Tensor:
    """Random-but-realistic layer input ``[B, S, E, D]``.

    Rows of ``embed_weight`` (the real embedding table if given, else N(0, 1)) for random tokens, replicated to the E
    streams (exactly the layer-0 input) and then perturbed per stream with relative noise ``stream_noise`` (streams
    diverge after the first mHC mixes) and a per-token log-normal scale. For truly realistic deep-layer inputs use
    :func:`capture_model_goldens` on a real prompt instead.
    """
    g = torch.Generator().manual_seed(seed)
    E, D = args.mhc_expansion_rate, args.hidden_size
    if embed_weight is not None:
        ids = torch.randint(0, embed_weight.shape[0], (batch, seq_len), generator=g)
        base = embed_weight[ids].float()
    else:
        base = torch.randn(batch, seq_len, D, generator=g)
    rms = base.pow(2).mean(-1, keepdim=True).sqrt()
    x = base.unsqueeze(2).expand(batch, seq_len, E, D).clone()
    x = x + stream_noise * rms.unsqueeze(2) * torch.randn(batch, seq_len, E, D, generator=g)
    x = x * torch.exp(0.25 * torch.randn(batch, seq_len, 1, 1, generator=g))
    return x.to(dtype)


@torch.no_grad()
def capture_layer_goldens(
    layer: DecoderLayer,
    x_streams: torch.Tensor,
    decode_steps: int = 0,
    attn_mode: str = "expanded",
    include: Optional[Callable[[str], bool]] = None,
    full_reference: bool = True,
) -> dict:
    """Prefill ``x_streams[:, :S - decode_steps]`` then decode the remaining rows one token at a time.

    ``x_streams [B, S, E, D]`` must be in the layer's dtype; positions are ``arange(S)``. With ``full_reference``
    the layer output of a cache-free full-sequence pass is stored as ``full.x_out`` (decode-step outputs must
    reproduce its last rows). The final latent cache is stored as ``cache.{c_kv,k_pe}``.
    """
    args = layer.args
    B, S = x_streams.shape[:2]
    n_pre = S - decode_steps
    if n_pre <= 0:
        raise ValueError("need at least one prefill token")
    positions = torch.arange(S)[None, :].expand(B, S)
    cache = LatentKVCache(
        B, S, args.kv_lora_rank, args.qk_rope_head_dim, x_streams.dtype, args.attention_window(layer.layer_idx)
    )
    rec = TensorRecorder(include)
    layer(x_streams[:, :n_pre], positions[:, :n_pre], cache, attn_mode, tap=rec)
    prefill = dict(rec.tensors, positions=positions[:, :n_pre].clone())
    decode: List[Dict[str, torch.Tensor]] = []
    for t in range(n_pre, S):
        rec = TensorRecorder(include)
        layer(x_streams[:, t : t + 1], positions[:, t : t + 1], cache, attn_mode, tap=rec)
        decode.append(dict(rec.tensors, positions=positions[:, t : t + 1].clone()))
    golden = dict(
        meta=dict(
            _layer_meta(args, layer.layer_idx),
            dtype=str(x_streams.dtype).replace("torch.", ""),
            attn_mode=attn_mode,
            batch=B,
            prefill_len=n_pre,
            decode_steps=decode_steps,
            args=_args_meta(args),
        ),
        prefill=prefill,
        decode=decode,
        cache=dict(c_kv=cache.c.clone(), k_pe=cache.k_pe.clone()),
    )
    if full_reference:
        golden["full"] = dict(x_out=layer(x_streams, positions, None, attn_mode).clone(), positions=positions.clone())
    return golden


@torch.no_grad()
def capture_model_goldens(
    model: MotifForCausalLM,
    input_ids: torch.Tensor,
    decode_ids: Optional[torch.Tensor] = None,
    attn_mode: str = "expanded",
    include: Optional[Callable[[str], bool]] = None,
    max_seq_len: Optional[int] = None,
) -> dict:
    """Prefill ``input_ids [B, S]`` through the (prefix) model, then teacher-forced decode of ``decode_ids [B, n]``.

    Records every tap (filter with ``include``), e.g. ``layers.2.moe.router.indices`` or ``layers.0.x_out``.
    For a partial model (``layer_ids=range(k)``) ``logits`` are early-exit logits of layer k-1.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids[None]
    B, S = input_ids.shape
    n_dec = 0 if decode_ids is None else decode_ids.shape[1]
    cache = model.new_cache(B, max_seq_len or (S + n_dec))
    rec = TensorRecorder(include)
    positions = torch.arange(S)[None, :].expand(B, S)
    model(input_ids, positions, cache, attn_mode, tap=rec)
    prefill = dict(rec.tensors, input_ids=input_ids.clone(), positions=positions.clone())
    decode = []
    for t in range(n_dec):
        rec = TensorRecorder(include)
        pos = torch.full((B, 1), S + t, dtype=torch.long)
        model(decode_ids[:, t : t + 1], pos, cache, attn_mode, tap=rec)
        decode.append(dict(rec.tensors, input_ids=decode_ids[:, t : t + 1].clone(), positions=pos))
    args = model.args
    return dict(
        meta=dict(
            dtype=str(model.dtype).replace("torch.", ""),
            attn_mode=attn_mode,
            layers=[_layer_meta(args, i) for i in model.model.layer_ids],
            batch=B,
            prefill_len=S,
            decode_steps=n_dec,
            args=_args_meta(args),
        ),
        prefill=prefill,
        decode=decode,
    )


@torch.no_grad()
def capture_expert_goldens(moe, x: torch.Tensor, experts: Optional[Sequence[int]] = None) -> dict:
    """Per-expert I/O of a MoE layer for the tokens the router sends to each expert.

    ``x [B, S, D]`` is the MoE input (``post_attention_layernorm.out``). Returns
    ``{"router": {...}, "experts": {e: {"tokens", "slots", "weights", "x", "gate", "up", "act", "out"}}}`` where
    ``out = expert_e(x[tokens])`` before the routing weight; the MoE output is
    ``dtype(sum_e index_add(out.float() * weights) + shared.float())``.
    """
    rec = TensorRecorder()
    xf = x.reshape(-1, x.shape[-1])
    weights, indices = moe.router(xf, moe.expert_bias, tap=rec)
    chosen = sorted(set(indices.flatten().tolist())) if experts is None else list(experts)
    out = {}
    for e in chosen:
        tok, slot = torch.nonzero(indices == e, as_tuple=True)
        er = TensorRecorder()
        moe.experts.expert_forward(xf[tok], e, tap=er)
        out[int(e)] = dict(er.tensors, tokens=tok, slots=slot, weights=weights[tok, slot], x=xf[tok].clone())
    return dict(router=rec.tensors, experts=out)


def save_golden(golden: dict, path: os.PathLike) -> str:
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(golden, path)
    return path


def load_golden(path: os.PathLike) -> dict:
    return torch.load(os.fspath(path), weights_only=True)


def layer_input_from_model(golden: dict, layer_idx: int, step: Optional[int] = None) -> torch.Tensor:
    """The recorded 4-stream input of ``layer_idx`` (prefill, or decode step ``step``) from a model golden."""
    src = golden["prefill"] if step is None else golden["decode"][step]
    return src[f"layers.{layer_idx}.x_in"]


def tensors_with_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in d.items() if k.startswith(prefix)}


def summarize(golden: dict, keys: Optional[Sequence[str]] = None) -> str:
    lines = [f"meta: { {k: v for k, v in golden['meta'].items() if k != 'args'} }"]
    for k, v in golden["prefill"].items():
        if keys is None or k in keys:
            lines.append(f"  {k}: {tuple(v.shape)} {str(v.dtype).replace('torch.', '')}")
    lines.append(f"  decode steps: {len(golden['decode'])}")
    return "\n".join(lines)


# =================================================================================================
# real-weight export (CLI)
# =================================================================================================
DEFAULT_PROMPT_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": (
            "Tenstorrent Blackhole Galaxy systems connect thirty-two accelerator chips in a two-dimensional mesh. "
            "Explain, step by step, how a mixture-of-experts transformer with three hundred eighty-four routed "
            "experts and one shared expert per layer could be partitioned across such a mesh, which collective "
            "communication patterns are required to dispatch tokens to experts and to combine their outputs, and "
            "how sliding-window attention layers differ from global attention layers in their key-value cache "
            "requirements when the context grows to many thousands of tokens. Keep the answer concise but "
            "technically precise, mention the numerical precision concerns for the sigmoid router with its expert "
            "bias, for the polynomial normalization inside the experts, and for the Sinkhorn iterations of the "
            "hyper-connection mixing matrices, and finish with a short list of the three most important things to "
            "validate first when porting such a model."
        ),
    },
]


def export_real_goldens(
    out_dir: os.PathLike,
    layer_ids: Sequence[int] = (0, 1, 2),
    dtype: torch.dtype = torch.bfloat16,
    messages: Optional[Sequence[dict]] = None,
    decode_steps: int = 4,
    attn_mode: str = "expanded",
    ckpt_dir: Optional[os.PathLike] = None,
) -> List[str]:
    """Real-weight goldens for a prefix model ``layer_ids = 0..k``: one file per layer (layer-relative tap names,
    e.g. ``x_in``, ``self_attn.c_kv``, ``moe.router.indices``) plus ``model_*.pt`` (embed, final norm, logits).

    The prompt (chat-templated, longer than the 129-key window) is prefilled except its last ``decode_steps``
    tokens, which are then decoded one at a time against the cache (teacher forcing). Routed experts are fetched
    lazily, so only the experts the router selects are read from disk.
    """
    from .tokenizer import encode_chat, load_tokenizer
    from .weights import load_reference_model

    ids = sorted(int(i) for i in layer_ids)
    if ids != list(range(len(ids))):
        raise ValueError("real goldens need a prefix model: layer_ids must be 0..k")
    model = load_reference_model(ckpt_dir, ids, dtype=dtype, lazy_experts=True)
    tok_ids = torch.tensor([encode_chat(messages or DEFAULT_PROMPT_MESSAGES, load_tokenizer(ckpt_dir))])
    n_pre = tok_ids.shape[1] - decode_steps
    g = capture_model_goldens(
        model, tok_ids[:, :n_pre], tok_ids[:, n_pre:] if decode_steps else None, attn_mode=attn_mode
    )
    tag = f"{str(dtype).replace('torch.', '')}_{attn_mode}"
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in ids:
        p = f"layers.{i}."
        layer_golden = dict(
            meta=dict(
                _layer_meta(model.args, i),
                dtype=g["meta"]["dtype"],
                attn_mode=attn_mode,
                prefill_len=g["meta"]["prefill_len"],
                decode_steps=decode_steps,
                args=g["meta"]["args"],
            ),
            prefill=dict(tensors_with_prefix(g["prefill"], p), positions=g["prefill"]["positions"]),
            decode=[dict(tensors_with_prefix(d, p), positions=d["positions"]) for d in g["decode"]],
        )
        paths.append(save_golden(layer_golden, os.path.join(out_dir, f"layer{i}_{tag}.pt")))
    model_keys = ("embed", "final.stream_mean", "final.norm", "logits", "input_ids", "positions")
    model_golden = dict(
        meta=g["meta"],
        prefill={k: v for k, v in g["prefill"].items() if k in model_keys},
        decode=[{k: v for k, v in d.items() if k in model_keys} for d in g["decode"]],
    )
    paths.append(save_golden(model_golden, os.path.join(out_dir, f"model_L0-{ids[-1]}_{tag}.pt")))
    return paths


def _main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Export Motif-3 real-weight goldens (CPU, no device).")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--layers", type=int, default=3, help="prefix depth k: layers 0..k-1 (default 3)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--decode-steps", type=int, default=4)
    ap.add_argument("--attn-mode", choices=["expanded", "absorbed"], default="expanded")
    ap.add_argument("--ckpt-dir", default=None)
    a = ap.parse_args(argv)
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    with torch.no_grad():
        for path in export_real_goldens(
            a.out, range(a.layers), dtype, decode_steps=a.decode_steps, attn_mode=a.attn_mode, ckpt_dir=a.ckpt_dir
        ):
            print(path, f"{os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    _main()
