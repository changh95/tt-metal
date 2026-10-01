# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Streaming full-model golden: a prompt set through all 53 Motif-3 layers, one decoder layer's weights at a time.

The whole checkpoint never has to be on disk (or in RAM) at once: layers 36-52 are downloaded only after the early
shards are deleted, so the golden is computed layer by layer and is resumable from the saved residual streams::

    embed -> for L in layers: load layer L (lazy experts) -> x_p = layer_L(x_p) for every prompt p -> free -> checkpoint
    after layer 52: mean over the 4 streams -> final RMSNorm -> lm_head -> per-position (teacher-forced) logits

Every step calls the reference modules exactly as ``MotifForCausalLM.forward`` does (``layer(x, positions, None,
attn_mode)`` with ``positions = arange(S)``, then ``reduce_streams``, ``norm``, ``lm_head(h).float()``); nothing is
re-implemented. The result is therefore the reference's ordinary full forward of each prompt (B=1, no KV cache, one
full-sequence prefill = teacher forcing), bit for bit, which ``tests/test_golden_stream.py`` checks on a tiny random
6-layer model (with an interrupted + resumed run) and on layers 0-3 of the real checkpoint.

Numerics (``--dtype``; recorded in the manifest and in every file's metadata):
  * ``bf16`` (default): bf16 weights and activations with the reference's default numerics, i.e. HF
    ``modeling_motif.py``'s bf16 cast points: ``q_path_fp32=True`` (wq_a/q_norm/wq_b in fp32), ``mhc_mix_fp32=False``
    (bf16 mHC mixes), ``attn_mode="expanded"`` (fp32 scores/softmax/PV, one cast), routed experts accumulated in fp32.
  * ``fp32``: every weight upcast to fp32, the "ideal" golden (intended for smaller sanity runs).

Prompt set (``prompts/``): ``messages.json`` holds the source conversations; ``--render-prompts`` renders them with the
Motif chat template (``tokenizer.encode_chat``) into ``rendered.json`` (token ids, text, per-token roles) and one
``<name>.txt`` per prompt. Roles (``ROLE_NAMES``): template glue / system / user / assistant / assistant_think / tool,
so teacher-forced metrics can be restricted to the assistant region (position t predicts token t + 1).

Output layout (``--out``; safetensors files carry their metadata as JSON under the key ``"json"``):
  manifest.json                        mode, prompt set, layers done, per-layer timings / RSS / state stats, runs, heads
  prompts.json                         frozen copy of the rendered prompt set that the states belong to
  states/embed.safetensors             {name: [1, S, 4096]} embedding (the layer-0 input is 4 copies of it)
  states/after_layer_XX.safetensors    {name: [1, S, 4, 4096]} 4-stream residual after layer XX (model dtype: bf16)
  resume/state.safetensors             the same after the last processed layer (``--resume`` continues from it)
  final/logits_after_layer_52.safetensors   per prompt ``{name}.<key>``, S positions (position t predicts t + 1):
        topk_ids [S, 32] int64 / topk_logits [S, 32] fp32 (descending), argmax [S] int64 (full vocab, first max),
        argmax_ties [S] (#vocab entries equal to the max: bf16 logits tie often), logsumexp [S] fp32,
        target_ids [S] (ids[t + 1], -1 at the last position), target_logit / target_logprob [S] fp32 (NaN there),
        target_rank [S] (#logits > target logit; 0 = tie-aware top-1; -1 there), target_in_assistant [S] bool
  final/hidden_after_layer_52.safetensors   {name}.final_hidden [1, S, 4096] fp32 (post-norm = lm_head input),
                                            {name}.stream_mean [1, S, 4096] fp32 (mean of the 4 streams)
  final/*_after_layer_XX.safetensors (XX < 52)   the same head applied early (``--final-head``): early-exit logits of
                                            the truncated model embed -> layers 0..XX -> mean -> norm -> lm_head

CLI (from the tt-metal root, ``python_env/bin/python -m models.demos.motif3.reference.golden_stream``)::

    --render-prompts                                   # prompts/messages.json -> prompts/rendered.json + *.txt
    --layers 0-35 --out /home/ttuser/hchang/experiments/motif-3/goldens/c2          # fresh run (embed + layers 0..35)
    --resume --layers 36-52 --out /home/ttuser/hchang/experiments/motif-3/goldens/c2  # continue; head after 52
    --resume --out ...                                 # = --layers auto: every consecutive downloaded layer
    --resume --final-head --out ...                    # early-exit head after the last processed layer only
    --status --out ...
    --dtype fp32 --layers 0-3 --prompts chat_default,multi_turn_chat --save-layers 3 --final-head --out /tmp/x

A layer runs only if all its shards are complete (safetensors header check) and, when the checkpoint directory has a
``.download_state.json``, it is listed in ``complete_layers``. The final head needs ``model.norm.weight`` and
``lm_head.weight`` (shard 104, which also holds layer-52 tensors, so it is present whenever layer 52 is). On this host
the layer outputs do not depend on the torch thread count (7, 16 and 32 threads give identical bits); the count is
still recorded and ``--resume`` reuses it.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from .config import MotifArgs
from .golden import _layer_meta
from .modules import DecoderLayer, MotifForCausalLM
from .weights import MissingWeightsError, MotifCheckpoint

FORMAT = "motif3-golden-stream/1"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_MESSAGES = PROMPTS_DIR / "messages.json"
DEFAULT_PROMPT_SET = PROMPTS_DIR / "rendered.json"
DEFAULT_OUT_DIR = Path(os.environ.get("MOTIF3_GOLDEN_STREAM_DIR", "/home/ttuser/hchang/experiments/motif-3/goldens/c2"))
DEFAULT_SAVE_LAYERS = (0, 1, 2, 3, 4, 7, 8, 15, 16, 23, 24, 31, 32, 35, 39, 47, 51, 52)
TOPK = 32
ROLE_NAMES = ("template", "system", "user", "assistant", "assistant_think", "tool")
ASSISTANT_ROLES = (3, 4)
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
_DTYPE_NAMES = {v: k for k, v in DTYPES.items()}

Log = Callable[[str], None]


def _log(msg: str) -> None:
    print(f"[golden_stream {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =================================================================================================
# prompt set
# =================================================================================================
@dataclass
class StreamPrompt:
    """One prompt of the set: token ids and (optionally) per-token role ids (``ROLE_NAMES``)."""

    name: str
    ids: List[int]
    roles: Optional[List[int]] = None

    def __post_init__(self):
        self.ids = [int(i) for i in self.ids]
        if not self.name or "." in self.name:
            raise ValueError(f"prompt name {self.name!r} must be non-empty and contain no '.'")
        if not self.ids:
            raise ValueError(f"prompt {self.name!r} has no tokens")
        if self.roles is not None:
            self.roles = [int(r) for r in self.roles]
            if len(self.roles) != len(self.ids):
                raise ValueError(f"prompt {self.name!r}: {len(self.roles)} roles for {len(self.ids)} tokens")

    def assistant_mask(self) -> torch.Tensor:
        """Bool ``[S]``: token t was produced by the assistant (content, ``<think>`` block or its ``<|endofturn|>``)."""
        if self.roles is None:
            return torch.zeros(len(self.ids), dtype=torch.bool)
        return torch.tensor([r in ASSISTANT_ROLES for r in self.roles], dtype=torch.bool)


def prompt_set_sha256(prompts: Sequence[StreamPrompt]) -> str:
    """Identity of a prompt set for the computation: names and token ids, in order (roles are metadata)."""
    blob = json.dumps([[p.name, p.ids] for p in prompts], separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _prompts_from_doc(doc: dict, names: Optional[Sequence[str]] = None) -> List[StreamPrompt]:
    prompts = [StreamPrompt(p["name"], p["ids"], p.get("roles")) for p in doc["prompts"]]
    if names is not None:
        by_name = {p.name: p for p in prompts}
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise KeyError(f"unknown prompt(s) {unknown}; the set has {sorted(by_name)}")
        prompts = [by_name[n] for n in names]
    if len({p.name for p in prompts}) != len(prompts):
        raise ValueError("duplicate prompt names")
    return prompts


def load_prompt_set(
    path: os.PathLike = DEFAULT_PROMPT_SET, names: Optional[Sequence[str]] = None
) -> List[StreamPrompt]:
    """Prompts of a rendered prompt set (``prompts/rendered.json`` or a run's ``prompts.json``), optionally a subset."""
    return _prompts_from_doc(json.loads(Path(path).read_text(encoding="utf-8")), names)


def _special_ids(tokenizer) -> Dict[str, int]:
    names = dict(
        startofturn="<|startofturn|>",
        endofturn="<|endofturn|>",
        system="<|system|>",
        user="<|user|>",
        assistant="<|assistant|>",
        tool="<|tool|>",
        think="<think>",
        end_think="</think>",
    )
    return {k: int(tokenizer.convert_tokens_to_ids(v)) for k, v in names.items()}


def token_roles(ids: Sequence[int], tokenizer) -> List[int]:
    """Per-token role ids (``ROLE_NAMES``) from the template's ``<|startofturn|><|role|> ... <|endofturn|>`` turns.

    A turn's text and its ``<|endofturn|>`` get the turn's role; ``<|beginoftext|>``, ``<|startofturn|>``, the role tag
    and ``<|endoftext|>`` are ``template``. In an assistant turn ``<think> ... </think>`` (inclusive) is
    ``assistant_think`` (for a generation prompt that is the trailing ``<think>``).
    """
    sid = _special_ids(tokenizer)
    role_of_tag = {sid["system"]: 1, sid["user"]: 2, sid["assistant"]: 3, sid["tool"]: 5}
    roles: List[int] = []
    cur, in_think, expect_tag = 0, False, False
    for t in ids:
        if expect_tag:
            cur, expect_tag = role_of_tag.get(int(t), 0), False
            roles.append(0)
        elif t == sid["startofturn"]:
            roles.append(0)
            expect_tag = True
        elif cur == 0:
            roles.append(0)
        else:
            if cur == 3 and t == sid["think"]:
                in_think = True
            roles.append(4 if in_think else cur)
            if t == sid["end_think"]:
                in_think = False
            if t == sid["endofturn"]:
                cur, in_think = 0, False
    return roles


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _dumps_compact_ints(doc: dict) -> str:
    """``json.dumps(indent=1)``, but the ``ids`` / ``roles`` lists of every prompt on one line each."""
    doc = copy.deepcopy(doc)
    stash: Dict[str, str] = {}
    for p in doc["prompts"]:
        for key in ("ids", "roles"):
            if p.get(key) is not None:
                tag = f"@@{p['name']}.{key}@@"
                stash[json.dumps(tag)] = json.dumps(p[key], separators=(",", ":"))
                p[key] = tag
    text = json.dumps(doc, ensure_ascii=False, indent=1)
    for tag, val in stash.items():
        text = text.replace(tag, val, 1)
    return text + "\n"


def render_prompt_set(
    messages_path: os.PathLike = DEFAULT_MESSAGES,
    out_dir: Optional[os.PathLike] = PROMPTS_DIR,
    tokenizer=None,
    ckpt_dir: Optional[os.PathLike] = None,
) -> dict:
    """Render ``messages.json`` with the Motif chat template. Writes ``rendered.json`` + ``<name>.txt`` into ``out_dir``
    (``None``: write nothing) and returns the rendered document."""
    from .tokenizer import encode_chat, load_tokenizer
    from .weights import DEFAULT_WEIGHTS_DIR

    tok = tokenizer or load_tokenizer(ckpt_dir)
    src = json.loads(Path(messages_path).read_text(encoding="utf-8"))
    rendered = []
    for p in src["prompts"]:
        gen = bool(p.get("add_generation_prompt", False))
        ids = encode_chat(p["messages"], tok, add_generation_prompt=gen)
        text = tok.apply_chat_template(list(p["messages"]), add_generation_prompt=gen, tokenize=False)
        if tok(text, add_special_tokens=False)["input_ids"] != ids:
            raise RuntimeError(f"prompt {p['name']}: re-tokenizing the rendered text does not give the chat ids")
        roles = token_roles(ids, tok)
        rendered.append(
            dict(
                name=p["name"],
                description=p.get("description", ""),
                add_generation_prompt=gen,
                n_tokens=len(ids),
                n_assistant_tokens=sum(r in ASSISTANT_ROLES for r in roles),
                messages=p["messages"],
                text=text,
                ids=ids,
                roles=roles,
            )
        )
    tok_dir = Path(ckpt_dir) if ckpt_dir is not None else DEFAULT_WEIGHTS_DIR
    doc = dict(
        version=1,
        about=(
            "Rendered Motif-3 streaming-golden prompt set (generated by golden_stream.render_prompt_set from "
            "messages.json; do not edit by hand). roles[t] indexes role_names; position t of a teacher-forced "
            "forward predicts ids[t + 1]."
        ),
        role_names=list(ROLE_NAMES),
        tokenizer=dict(
            tokenizer_json_sha256=_sha256_file(tok_dir / "tokenizer.json"),
            chat_template_sha256=_sha256_file(tok_dir / "chat_template.jinja"),
        ),
        sha256=prompt_set_sha256(_prompts_from_doc(dict(prompts=rendered))),
        n_tokens_total=sum(p["n_tokens"] for p in rendered),
        prompts=rendered,
    )
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rendered.json").write_text(_dumps_compact_ints(doc), encoding="utf-8")
        for p in rendered:
            (out / f"{p['name']}.txt").write_text(p["text"] + "\n", encoding="utf-8")
    return doc


# =================================================================================================
# model pieces (reference modules, loaded one at a time)
# =================================================================================================
def load_decoder_layer(
    ckpt: MotifCheckpoint, args: MotifArgs, layer_idx: int, dtype: torch.dtype, lazy_experts: bool = True
) -> DecoderLayer:
    """Reference ``DecoderLayer`` with real weights, loaded like ``weights.load_reference_model`` loads each layer
    (meta construction, ``assign``-load; bf16 tensors stay memory-mapped). Lazy experts are read on first use and
    cached for the whole layer (every prompt reuses them)."""
    with torch.device("meta"):
        layer = DecoderLayer(args, layer_idx, materialize_experts=not lazy_experts)
    sd = ckpt.layer_state_dict(layer_idx, dtype, include_experts=not lazy_experts, strip_prefix=True)
    layer.load_state_dict(sd, strict=True, assign=True)
    if lazy_experts and layer.is_moe:
        layer.moe.experts.expert_source = ckpt.expert_source(layer_idx, dtype, cache_size=args.num_experts)
    layer.requires_grad_(False)
    return layer


def _head_shell(args: MotifArgs, ckpt: MotifCheckpoint, names: Sequence[str], dtype: torch.dtype) -> MotifForCausalLM:
    """A ``MotifForCausalLM`` without decoder layers holding only ``names`` (embed / final norm / lm_head); the other
    parameters stay on the meta device. Its ``model.embed_tokens``, ``expand_streams``, ``reduce_streams``,
    ``model.norm`` and ``lm_head`` are the reference's own code."""
    with torch.device("meta"):
        shell = MotifForCausalLM(args, layer_ids=[], materialize_experts=False)
    missing, unexpected = shell.load_state_dict({n: ckpt.get(n, dtype) for n in names}, strict=False, assign=True)
    assert not unexpected, unexpected
    return shell


# =================================================================================================
# files
# =================================================================================================
def _umask() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError):
        pass
    return 0o022


def _save_tensors(path: Path, tensors: Dict[str, torch.Tensor], meta: dict) -> int:
    """Atomic safetensors write (temp file + rename); the file gets the usual umask permissions."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file({k: v.detach().contiguous() for k, v in tensors.items()}, str(tmp), metadata={"json": json.dumps(meta)})
    os.chmod(tmp, 0o666 & ~_umask())  # safetensors creates 0600 files
    os.replace(tmp, path)
    return path.stat().st_size


def load_tensors(path: os.PathLike) -> Tuple[Dict[str, torch.Tensor], dict]:
    """``({key: tensor}, metadata)`` of a golden-stream safetensors file (tensors are private, aligned copies)."""
    from safetensors import safe_open

    with safe_open(os.fspath(path), framework="pt", device="cpu") as f:
        meta = json.loads((f.metadata() or {}).get("json", "{}"))
        tensors = {k: f.get_tensor(k).clone() for k in f.keys()}
    return tensors, meta


def state_path(out_dir: os.PathLike, layer_idx: int) -> Path:
    return Path(out_dir) / "states" / f"after_layer_{layer_idx:02d}.safetensors"


def head_paths(out_dir: os.PathLike, layer_idx: int) -> Tuple[Path, Path]:
    d = Path(out_dir) / "final"
    return d / f"logits_after_layer_{layer_idx:02d}.safetensors", d / f"hidden_after_layer_{layer_idx:02d}.safetensors"


def load_states(out_dir: os.PathLike, layer_idx: int) -> Dict[str, torch.Tensor]:
    """``{prompt name: [1, S, 4, D]}`` saved after ``layer_idx``."""
    return load_tensors(state_path(out_dir, layer_idx))[0]


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _mem_gb() -> Dict[str, float]:
    """Current RSS split (GB): ``rss`` = anon + file-backed (memory-mapped shards, reclaimable) + shmem."""
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                key = line.split(":", 1)[0]
                if key in ("VmRSS", "RssAnon", "RssFile", "VmHWM"):
                    out[{"VmRSS": "rss", "RssAnon": "anon", "RssFile": "file", "VmHWM": "hwm"}[key]] = round(
                        int(line.split()[1]) / 2**20, 3
                    )
    except OSError:
        out["hwm"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
    return out


# =================================================================================================
# layer selection
# =================================================================================================
def downloaded_layers(ckpt: MotifCheckpoint) -> Optional[List[int]]:
    """``complete_layers`` of the checkpoint's ``.download_state.json`` (``None`` if there is no such file)."""
    path = ckpt.dir / ".download_state.json"
    if not path.exists():
        return None
    return sorted(int(i) for i in json.loads(path.read_text()).get("complete_layers", []))


def layer_available(ckpt: MotifCheckpoint, layer_idx: int, downloaded: Optional[Sequence[int]] = None) -> bool:
    if downloaded is not None and layer_idx not in downloaded:
        return False
    return ckpt.layer_is_local(layer_idx)


def resolve_layers(
    spec: Union[str, Iterable[int]],
    start: int,
    num_layers: int,
    available: Optional[Callable[[int], bool]] = None,
) -> List[int]:
    """Layers to run next. ``spec``: ``"auto"`` (every consecutive available layer from ``start``), ``"all"``
    (``start..num_layers-1``), ``"a-b"``, ``"a-"``, ``"a"``, ``"a,b,c"`` or a sequence of ints. The result must be
    contiguous and begin at ``start`` (the layer after the last processed one)."""
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s == "auto":
            out = []
            for i in range(start, num_layers):
                if available is not None and not available(i):
                    break
                out.append(i)
            return out
        if s == "all":
            ids = list(range(start, num_layers))
        elif "-" in s:
            a, b = s.split("-", 1)
            ids = list(range(int(a), int(b) + 1 if b else num_layers))
        else:
            ids = [int(x) for x in s.split(",") if x.strip()]
    else:
        ids = [int(x) for x in spec]
    if not ids:
        return []
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError(f"layers must be contiguous and ascending, got {ids}")
    if ids[0] != start:
        have = "a fresh run starts from the embedding" if start == 0 else f"the saved state is after layer {start - 1}"
        raise ValueError(f"the next layer to process is {start} ({have}), but the requested layers start at {ids[0]}")
    if ids[-1] >= num_layers:
        raise ValueError(f"layer {ids[-1]} does not exist (the model has {num_layers} layers)")
    if available is not None:
        missing = [i for i in ids if not available(i)]
        if missing:
            raise MissingWeightsError(
                f"layer(s) {missing} are not completely downloaded (shard header check / .download_state.json); "
                f"run the available prefix with --layers auto and --resume later"
            )
    return ids


# =================================================================================================
# the streaming run
# =================================================================================================
_MODE_KEYS = ("dtype", "attn_mode", "q_path_fp32", "mhc_mix_fp32")


def _mode(args: MotifArgs, dtype: torch.dtype, attn_mode: str) -> dict:
    return dict(
        dtype=_DTYPE_NAMES[dtype],
        attn_mode=attn_mode,
        q_path_fp32=bool(args.q_path_fp32),
        mhc_mix_fp32=bool(args.mhc_mix_fp32),
        forward="MotifForCausalLM.forward per prompt: B=1, positions 0..S-1, no KV cache (teacher-forced prefill)",
    )


def _same_mode(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in _MODE_KEYS)


def _state_stats(x: torch.Tensor) -> dict:
    xf = x.float()
    return dict(
        rms=round(float(xf.pow(2).mean().sqrt()), 6),
        absmax=round(float(xf.abs().max()), 6),
        finite=bool(torch.isfinite(xf).all()),
    )


def _clean_previous_run(out: Path) -> None:
    """Remove the files a previous run in ``out`` wrote (only those; other content of ``out`` is left alone)."""
    for sub, pattern in (("states", "*.safetensors*"), ("resume", "state.safetensors*"), ("final", "*.safetensors*")):
        for p in (out / sub).glob(pattern):
            p.unlink()
    for name in ("manifest.json", "prompts.json"):
        if (out / name).exists():
            (out / name).unlink()


@torch.no_grad()
def apply_final_head(
    shell: MotifForCausalLM, prompt: StreamPrompt, x: torch.Tensor, topk: int = TOPK
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict]:
    """Mean over the 4 streams -> final norm -> lm_head (exactly ``MotifModel``/``MotifForCausalLM.forward``), reduced
    to the stored per-position quantities. Returns ``(logit tensors, hidden tensors, summary)``."""
    stream_mean = shell.model.reduce_streams(x)
    h = shell.model.norm(stream_mean)
    logits = shell.lm_head(h).float()[0]  # [S, V]
    S = logits.shape[0]
    top_v, top_i = torch.topk(logits, min(topk, logits.shape[-1]), dim=-1)
    target = torch.tensor(prompt.ids[1:] + [-1], dtype=torch.long)
    valid = target >= 0
    t_logit = logits.gather(1, target.clamp(min=0)[:, None])[:, 0]
    lse = torch.logsumexp(logits, dim=-1)
    rank = (logits > t_logit[:, None]).sum(-1)
    nan = torch.tensor(float("nan"))
    t_logit = torch.where(valid, t_logit, nan)
    rank = torch.where(valid, rank, torch.full_like(rank, -1))
    in_asst = torch.cat([prompt.assistant_mask()[1:], torch.zeros(1, dtype=torch.bool)])
    lt = dict(
        topk_ids=top_i,
        topk_logits=top_v,
        argmax=logits.argmax(-1),
        argmax_ties=(logits == top_v[:, :1]).sum(-1),
        logsumexp=lse,
        target_ids=target,
        target_logit=t_logit,
        target_logprob=t_logit - lse,
        target_rank=rank,
        target_in_assistant=in_asst,
    )
    ht = dict(final_hidden=h.float(), stream_mean=stream_mean.float())

    def region(mask):
        n = int(mask.sum())
        if n == 0:
            return dict(n=0)
        r, lp = rank[mask], (t_logit - lse)[mask]
        return dict(
            n=n,
            top1=round(float((r == 0).float().mean()), 4),
            top5=round(float((r < 5).float().mean()), 4),
            top32=round(float((r < topk).float().mean()), 4),
            nll=round(float(-lp.mean()), 4),
        )

    summary = dict(
        positions=S,
        all=region(valid),
        assistant=region(valid & in_asst),
        last_position_top5_ids=[int(i) for i in top_i[-1, :5]],
        finite=bool(torch.isfinite(logits).all()),
    )
    return lt, ht, summary


@torch.no_grad()
def run_stream(
    out_dir: os.PathLike = DEFAULT_OUT_DIR,
    layers: Union[str, Iterable[int]] = "auto",
    *,
    resume: bool = False,
    prompts: Optional[Sequence[StreamPrompt]] = None,
    prompt_set_path: Optional[os.PathLike] = None,
    ckpt_dir: Optional[os.PathLike] = None,
    dtype: Optional[torch.dtype] = None,
    attn_mode: Optional[str] = None,
    save_layers: Optional[Iterable[int]] = DEFAULT_SAVE_LAYERS,
    final_head: Optional[bool] = None,
    overwrite: bool = False,
    threads: Optional[int] = None,
    argv: Optional[Sequence[str]] = None,
    log: Log = _log,
) -> dict:
    """Run (or continue) the streaming golden; returns the manifest.

    Fresh run (``resume=False``): embeds ``prompts`` (default: the full prompt set at ``prompt_set_path``) and runs
    ``layers`` starting at 0. ``resume=True``: continues from ``out_dir/resume/state.safetensors`` with the frozen
    prompt set, dtype and attention mode of that run; ``layers`` must start right after the last processed layer.
    ``final_head``: ``None`` = after the model's last layer only; ``True`` = also after the last layer processed by
    this call (early-exit logits of a truncated run); ``False`` = never. ``save_layers=None`` saves every layer.
    """
    t_run = time.perf_counter()
    out = Path(out_dir)
    ckpt = MotifCheckpoint(ckpt_dir)
    args = ckpt.args()
    n_layers = args.num_hidden_layers
    config_sha = _sha256_file(ckpt.dir / "config.json")
    manifest_path, resume_path = out / "manifest.json", out / "resume" / "state.safetensors"
    if threads is not None:
        torch.set_num_threads(int(threads))
    peak_anon = 0.0

    if resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"no resume checkpoint at {resume_path}")
        states, rmeta = load_tensors(resume_path)  # cloned: fresh, aligned buffers like the in-process stream
        if rmeta.get("format") != FORMAT:
            raise ValueError(f"{resume_path} is not a {FORMAT} checkpoint")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen = load_prompt_set(out / "prompts.json")
        if prompts is not None and prompt_set_sha256(prompts) != prompt_set_sha256(frozen):
            raise ValueError("resume: the given prompts differ from the run's frozen prompts.json")
        prompts = frozen
        if prompt_set_sha256(prompts) != rmeta["prompt_sha256"]:
            raise ValueError("resume: prompts.json does not match the resume checkpoint")
        rec_dtype = DTYPES[rmeta["mode"]["dtype"]]
        if dtype is not None and dtype != rec_dtype:
            raise ValueError(f"resume: run dtype is {rmeta['mode']['dtype']}, not {_DTYPE_NAMES[dtype]}")
        if attn_mode is not None and attn_mode != rmeta["mode"]["attn_mode"]:
            raise ValueError(f"resume: run attn_mode is {rmeta['mode']['attn_mode']}, not {attn_mode}")
        dtype, attn_mode = rec_dtype, rmeta["mode"]["attn_mode"]
        if not _same_mode(_mode(args, dtype, attn_mode), rmeta["mode"]):
            raise ValueError(f"resume: numerics changed: {_mode(args, dtype, attn_mode)} vs {rmeta['mode']}")
        if config_sha != rmeta.get("config_sha256"):
            raise ValueError("resume: the checkpoint's config.json changed since the run started")
        if set(states) != {p.name for p in prompts}:
            raise ValueError("resume: checkpoint prompts do not match prompts.json")
        start = int(rmeta["last_layer"]) + 1
        if manifest.get("threads") not in (None, torch.get_num_threads()):
            log(f"WARNING: run used {manifest['threads']} threads, now {torch.get_num_threads()} (rounding may differ)")
        log(f"resuming after layer {start - 1} ({len(prompts)} prompts, {rmeta['mode']['dtype']}, {attn_mode})")
    else:
        if (manifest_path.exists() or resume_path.exists()) and not overwrite:
            raise FileExistsError(f"{out} already holds a streaming-golden run; pass resume=True or overwrite=True")
        dtype = dtype or torch.bfloat16
        attn_mode = attn_mode or "expanded"
        prompt_doc = None
        if prompts is None:
            prompt_doc = json.loads(Path(prompt_set_path or DEFAULT_PROMPT_SET).read_text(encoding="utf-8"))
            prompts = _prompts_from_doc(prompt_doc)
        else:
            prompts = list(prompts)
            if prompt_set_path is not None:  # a subset of a rendered set: keep its text / messages / roles
                src = json.loads(Path(prompt_set_path).read_text(encoding="utf-8"))
                by_name = {p["name"]: p for p in src["prompts"]}
                if all(p.name in by_name and by_name[p.name]["ids"] == p.ids for p in prompts):
                    prompt_doc = dict(src, prompts=[by_name[p.name] for p in prompts])
        if not prompts or len({p.name for p in prompts}) != len(prompts):
            raise ValueError("need at least one prompt and unique prompt names")
        if prompt_doc is None:  # ad-hoc prompts (tests): store just what the run needs
            prompt_doc = dict(
                version=1,
                role_names=list(ROLE_NAMES),
                prompts=[dict(name=p.name, n_tokens=len(p.ids), ids=p.ids, roles=p.roles) for p in prompts],
            )
        start = 0
        manifest = dict(
            format=FORMAT,
            mode=_mode(args, dtype, attn_mode),
            checkpoint=dict(dir=str(ckpt.dir), config_sha256=config_sha, num_hidden_layers=n_layers),
            prompt_set=dict(
                file="prompts.json",
                sha256=prompt_set_sha256(prompts),
                n_tokens_total=sum(len(p.ids) for p in prompts),
                prompts=[dict(name=p.name, n_tokens=len(p.ids)) for p in prompts],
            ),
            threads=torch.get_num_threads(),
            torch=torch.__version__,
            saved_layers=[],
            files={},
            layers_done=[],
            last_layer=None,
            layer_log=[],
            heads={},
            runs=[],
        )

    save_set = None if save_layers is None else {int(i) for i in save_layers}
    downloaded = downloaded_layers(ckpt)
    layer_ids = resolve_layers(layers, start, n_layers, lambda i: layer_available(ckpt, i, downloaded))
    end = layer_ids[-1] if layer_ids else manifest["last_layer"]  # the layer a head would follow
    want_head = end is not None and final_head is not False and (end == n_layers - 1 or final_head is True)
    if not layer_ids and not (want_head and str(end) not in manifest["heads"]):
        why = f"layer {start} is not available yet" if start < n_layers else "every layer and the head are done"
        log(f"nothing to run ({why}); last processed layer: {manifest['last_layer']}")
        return manifest
    if not resume:  # nothing is written before the request is known to be valid
        _clean_previous_run(out)
        out.mkdir(parents=True, exist_ok=True)
        prompt_doc = dict(
            prompt_doc, sha256=prompt_set_sha256(prompts), n_tokens_total=manifest["prompt_set"]["n_tokens_total"]
        )
        (out / "prompts.json").write_text(_dumps_compact_ints(prompt_doc), encoding="utf-8")
    run_rec = dict(
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
        argv=list(argv) if argv is not None else None,
        resume=bool(resume),
        layers=[layer_ids[0], layer_ids[-1]] if layer_ids else [],
        save_layers="all" if save_set is None else sorted(save_set),
        threads=torch.get_num_threads(),
    )
    log(
        f"{len(prompts)} prompts / {sum(len(p.ids) for p in prompts)} tokens, {_DTYPE_NAMES[dtype]} {attn_mode}, "
        + (f"layers {layer_ids[0]}..{layer_ids[-1]}, " if layer_ids else f"head after layer {end}, ")
        + f"{torch.get_num_threads()} threads -> {out}"
    )

    def finish_manifest():
        mem = _mem_gb()
        run_rec.update(
            finished=time.strftime("%Y-%m-%d %H:%M:%S"),
            wall_s=round(time.perf_counter() - t_run, 2),
            peak_rss_gb=mem.get("hwm"),
            peak_anon_gb=round(peak_anon, 3),
        )
        manifest["runs"].append(run_rec)
        manifest["files"] = {
            str(p.relative_to(out)): p.stat().st_size
            for p in [f for sub in ("states", "resume", "final") for f in sorted((out / sub).glob("*.safetensors"))]
            + [out / "prompts.json"]
            if p.exists()
        }
        manifest["saved_layers"] = sorted(
            int(p.stem.rsplit("_", 1)[1]) for p in out.glob("states/after_layer_*.safetensors")
        )
        manifest["total_bytes"] = sum(manifest["files"].values())
        _write_json(manifest_path, manifest)

    # ---- layer-0 input -------------------------------------------------------------------------------------------
    if not resume:
        t0 = time.perf_counter()
        shell = _head_shell(args, ckpt, ["model.embed_tokens.weight"], dtype)
        embeds, states = {}, {}
        for p in prompts:
            h = shell.model.embed_tokens(torch.tensor([p.ids], dtype=torch.long))
            embeds[p.name] = h
            states[p.name] = shell.model.expand_streams(h)
        del shell
        _save_tensors(
            out / "states" / "embed.safetensors",
            embeds,
            dict(format=FORMAT, kind="embed [1, S, D]; layer-0 input = 4 identical streams", mode=manifest["mode"]),
        )
        del embeds
        log(f"embedded {len(prompts)} prompts in {time.perf_counter() - t0:.1f}s")

    # ---- decoder layers --------------------------------------------------------------------------------------------
    for L in layer_ids:
        t0 = time.perf_counter()
        layer_ckpt = MotifCheckpoint(ckpt.dir)  # own shard handles: dropped (unmapped) with the layer
        layer = load_decoder_layer(layer_ckpt, args, L, dtype)
        t1 = time.perf_counter()
        for p in prompts:
            x = states[p.name]
            B, S = x.shape[:2]
            positions = torch.arange(S)[None, :].expand(B, S)  # as MotifModel.forward
            states[p.name] = layer(x, positions, None, attn_mode)
            peak_anon = max(peak_anon, _mem_gb().get("anon", 0.0))
        t2 = time.perf_counter()
        mem_layer = _mem_gb()
        del layer, layer_ckpt, x, positions
        gc.collect()
        stats = {name: _state_stats(x) for name, x in states.items()}
        meta = dict(
            format=FORMAT,
            kind="x_out [1, S, E=4, D]",
            layer=L,
            last_layer=L,
            mode=manifest["mode"],
            prompt_sha256=manifest["prompt_set"]["sha256"],
            config_sha256=config_sha,
        )
        saved = None
        if save_set is None or L in save_set:
            saved = _save_tensors(state_path(out, L), states, meta)
        _save_tensors(resume_path, states, meta)
        t3 = time.perf_counter()
        manifest["layers_done"] = sorted(set(manifest["layers_done"]) | {L})
        manifest["last_layer"] = L
        entry = dict(
            _layer_meta(args, L),
            load_s=round(t1 - t0, 2),
            compute_s=round(t2 - t1, 2),
            save_s=round(t3 - t2, 2),
            mem_gb_end_of_compute=mem_layer,
            mem_gb_after_free=_mem_gb(),
            saved_bytes=saved,
            stats=stats,
        )
        manifest["layer_log"] = [e for e in manifest["layer_log"] if e["layer_idx"] != L] + [entry]
        _write_json(manifest_path, manifest)
        bad = [n for n, s in stats.items() if not s["finite"]]
        log(
            f"layer {L:2d} {args.layer_kind(L):12s} load {t1 - t0:5.1f}s compute {t2 - t1:6.1f}s save {t3 - t2:4.1f}s "
            f"rss {mem_layer.get('rss', 0):6.1f} GB (anon {mem_layer.get('anon', 0):5.1f}) "
            f"rms {min(s['rms'] for s in stats.values()):.3g}..{max(s['rms'] for s in stats.values()):.3g}"
            + (f" NON-FINITE: {bad}" if bad else "")
        )
        if bad:
            finish_manifest()
            raise FloatingPointError(f"non-finite states after layer {L}: {bad}")

    # ---- final head --------------------------------------------------------------------------------------------------
    last = manifest["last_layer"]
    if want_head and (layer_ids or str(last) not in manifest["heads"]):
        t0 = time.perf_counter()
        shell = _head_shell(args, ckpt, ["model.norm.weight", "lm_head.weight"], dtype)
        logit_t, hidden_t, summaries = {}, {}, {}
        for p in prompts:
            lt, ht, summ = apply_final_head(shell, p, states[p.name])
            logit_t.update({f"{p.name}.{k}": v for k, v in lt.items()})
            hidden_t.update({f"{p.name}.{k}": v for k, v in ht.items()})
            summaries[p.name] = summ
            peak_anon = max(peak_anon, _mem_gb().get("anon", 0.0))
        del shell
        meta = dict(
            format=FORMAT,
            after_layer=last,
            early_exit=last != n_layers - 1,
            topk=TOPK,
            mode=manifest["mode"],
            prompt_sha256=manifest["prompt_set"]["sha256"],
            role_names=list(ROLE_NAMES),
        )
        lp, hp = head_paths(out, last)
        _save_tensors(lp, logit_t, meta)
        _save_tensors(hp, hidden_t, meta)
        manifest["heads"][str(last)] = dict(
            logits=str(lp.relative_to(out)),
            hidden=str(hp.relative_to(out)),
            early_exit=last != n_layers - 1,
            seconds=round(time.perf_counter() - t0, 2),
            summary=summaries,
        )
        log(
            f"final head after layer {last} in {time.perf_counter() - t0:.1f}s: "
            + ", ".join(f"{n} top1(asst)={s['assistant'].get('top1')}" for n, s in summaries.items())
        )

    finish_manifest()
    log(
        f"done in {run_rec['wall_s']:.1f}s, peak RSS {run_rec['peak_rss_gb']} GB (anon {run_rec['peak_anon_gb']} GB), "
        f"{manifest['total_bytes'] / 1e9:.2f} GB on disk"
    )
    return manifest


# =================================================================================================
# CLI
# =================================================================================================
def _status(out: Path) -> str:
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    lines = [
        f"{out}: {m['mode']['dtype']} {m['mode']['attn_mode']}, last layer {m['last_layer']}, "
        f"{m['prompt_set']['n_tokens_total']} tokens in {len(m['prompt_set']['prompts'])} prompts, "
        f"{m.get('total_bytes', 0) / 1e9:.2f} GB",
        f"layers done: {m['layers_done']}",
        f"heads: {sorted(m['heads'], key=int)}",
    ]
    for r in m["runs"]:
        lines.append(f"run {r['started']}: layers {r['layers']} {r.get('wall_s')}s peak RSS {r.get('peak_rss_gb')} GB")
    return "\n".join(lines)


def _parse_save_layers(spec: str) -> Optional[List[int]]:
    s = spec.strip().lower()
    if s == "all":
        return None
    if s in ("none", ""):
        return []
    out: List[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="python -m models.demos.motif3.reference.golden_stream",
        description="Streaming, resumable full-model Motif-3 golden (CPU, no device). See the module docstring.",
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR), help=f"output directory (default {DEFAULT_OUT_DIR})")
    ap.add_argument("--layers", default="auto", help="0-35 | 36-52 | 36- | all | auto (consecutive downloaded layers)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/resume/state.safetensors")
    ap.add_argument("--overwrite", action="store_true", help="fresh run into an --out that holds a previous run")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default=None, help="bf16 (default) | fp32 (sanity runs)")
    ap.add_argument("--attn-mode", choices=["expanded", "absorbed"], default=None, help="default expanded (HF)")
    ap.add_argument("--prompts", default=None, help="comma-separated subset of the prompt set (fresh runs)")
    ap.add_argument("--prompt-set", default=str(DEFAULT_PROMPT_SET), help="rendered prompt set (fresh runs)")
    ap.add_argument(
        "--save-layers",
        default=",".join(str(i) for i in DEFAULT_SAVE_LAYERS),
        help="layers whose output states are saved: list/ranges | all | none (the resume checkpoint is always kept)",
    )
    ap.add_argument("--final-head", action="store_true", help="also apply the head after the last layer of this call")
    ap.add_argument("--ckpt-dir", default=None, help="checkpoint directory (default MOTIF3_WEIGHTS_DIR)")
    ap.add_argument("--threads", type=int, default=None, help="torch threads (default: recorded on resume, else torch)")
    ap.add_argument("--render-prompts", action="store_true", help="render prompts/messages.json and exit")
    ap.add_argument("--status", action="store_true", help="print the run status of --out and exit")
    a = ap.parse_args(argv)
    out = Path(a.out)

    if a.render_prompts:
        doc = render_prompt_set(ckpt_dir=a.ckpt_dir)
        for p in doc["prompts"]:
            print(f"{p['name']:20s} {p['n_tokens']:5d} tokens ({p['n_assistant_tokens']} assistant)")
        print(f"total {doc['n_tokens_total']} tokens, sha256 {doc['sha256'][:16]} -> {DEFAULT_PROMPT_SET}")
        return 0
    if a.status:
        print(_status(out))
        return 0

    threads = a.threads
    if threads is None and a.resume and (out / "manifest.json").exists():
        threads = json.loads((out / "manifest.json").read_text(encoding="utf-8")).get("threads")
    prompts = None
    if a.prompts:
        if a.resume:
            ap.error("--prompts applies to fresh runs; a resumed run uses its frozen prompts.json")
        prompts = load_prompt_set(a.prompt_set, [n.strip() for n in a.prompts.split(",") if n.strip()])
    try:
        run_stream(
            out,
            a.layers,
            resume=a.resume,
            prompts=prompts,
            prompt_set_path=a.prompt_set,
            ckpt_dir=a.ckpt_dir,
            dtype=DTYPES[a.dtype] if a.dtype else None,
            attn_mode=a.attn_mode,
            save_layers=_parse_save_layers(a.save_layers),
            final_head=True if a.final_head else None,
            overwrite=a.overwrite,
            threads=threads,
            argv=argv,
        )
    except (MissingWeightsError, FileExistsError, FileNotFoundError, ValueError) as e:
        print(f"golden_stream: error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
