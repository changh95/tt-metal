# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weights for the Motif-3 reference: lazy safetensors loading, HF <-> reference names, random init.

* :class:`MotifCheckpoint` resolves tensors through ``model.safetensors.index.json`` and opens shards lazily
  (memory-mapped; nothing is read until a tensor is touched). Asking for a tensor whose shard is not on disk
  raises :class:`MissingWeightsError` naming the shard, so partial downloads fail loudly instead of silently.
* Reference parameter names equal the HF checkpoint names, except the mHC projections which are stored merged:
  ``mhc_{attn,ffn}.proj_merged.weight = cat([proj_pre, proj_post, proj_res])`` (rows 4 + 4 + 16).
* :func:`load_reference_model` builds only the requested layers (on the meta device, then ``assign``-loads the
  tensors), optionally fetching routed experts lazily per expert (``lazy_experts=True``) so a real MoE layer
  touches only the experts the router picks.
* :func:`random_state_dict` makes "realistic" random weights for tiny configs.
"""

from __future__ import annotations

import json
import os
import re
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from .config import MotifArgs
from .modules import MotifForCausalLM, MotifMTP

DEFAULT_WEIGHTS_DIR = Path(
    os.environ.get("MOTIF3_WEIGHTS_DIR", "/home/ttuser/hchang/experiments/motif-3/weights/Motif-3")
)
_MHC_PARTS = ("proj_pre", "proj_post", "proj_res")
_MHC_RE = re.compile(r"^(.*mhc_(?:attn|ffn))\.(proj_pre|proj_post|proj_res)\.weight$")
_EXPERT_TENSORS = ("moe.experts.gate_up_proj", "moe.experts.down_proj")


class MissingWeightsError(KeyError):
    """A requested tensor is not in the checkpoint index, or its shard is not present locally."""

    def __str__(self):  # KeyError would repr() the message
        return str(self.args[0]) if self.args else ""


# =================================================================================================
# name mapping
# =================================================================================================
def hf_to_reference_state_dict(hf_sd: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """HF/checkpoint names -> reference names (merges the three mHC projections)."""
    out: Dict[str, torch.Tensor] = {}
    pending: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, t in hf_sd.items():
        m = _MHC_RE.match(name)
        if m:
            pending.setdefault(m.group(1), {})[m.group(2)] = t
        else:
            out[name] = t
    for prefix, parts in pending.items():
        missing = [p for p in _MHC_PARTS if p not in parts]
        if missing:
            raise MissingWeightsError(f"{prefix}: missing mHC projection(s) {missing}")
        out[f"{prefix}.proj_merged.weight"] = torch.cat([parts[p] for p in _MHC_PARTS], dim=0)
    return out


def reference_to_hf_state_dict(ref_sd: Mapping[str, torch.Tensor], n_streams: int = 4) -> Dict[str, torch.Tensor]:
    """Reference names -> HF/checkpoint names (splits ``proj_merged`` into pre/post/res)."""
    E = n_streams
    out: Dict[str, torch.Tensor] = {}
    for name, t in ref_sd.items():
        if name.endswith(".proj_merged.weight"):
            prefix = name[: -len(".proj_merged.weight")]
            if t.shape[0] != E * E + 2 * E:
                raise ValueError(f"{name}: expected {E * E + 2 * E} rows, got {t.shape[0]}")
            out[f"{prefix}.proj_pre.weight"] = t[:E]
            out[f"{prefix}.proj_post.weight"] = t[E : 2 * E]
            out[f"{prefix}.proj_res.weight"] = t[2 * E :]
        else:
            out[name] = t
    return out


# =================================================================================================
# checkpoint access
# =================================================================================================
class MotifCheckpoint:
    """Lazy, index-driven access to a (possibly partially downloaded) Motif-3 safetensors checkpoint."""

    def __init__(self, ckpt_dir: Optional[os.PathLike] = None):
        self.dir = Path(ckpt_dir) if ckpt_dir is not None else DEFAULT_WEIGHTS_DIR
        index_path = self.dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"no model.safetensors.index.json under {self.dir}")
        self.weight_map: Dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        self._handles = {}
        self._shard_status: Dict[str, Tuple[Tuple[int, int], bool]] = {}

    # ---- metadata ----------------------------------------------------------------------------------
    def args(self, **overrides) -> MotifArgs:
        return MotifArgs.from_hf_config(self.dir, **overrides)

    def shard_is_complete(self, fn: str) -> bool:
        """True if shard ``fn`` exists and is at least as long as its safetensors header declares (a shard that
        is still being downloaded/copied is treated as absent). Cached per (size, mtime)."""
        path = self.dir / fn
        try:
            st = path.stat()
        except FileNotFoundError:
            return False
        key = (st.st_size, st.st_mtime_ns)
        cached = self._shard_status.get(fn)
        if cached is not None and cached[0] == key:
            return cached[1]
        ok = False
        try:
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                if 0 < n <= st.st_size - 8:
                    header = json.loads(f.read(n))
                    end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
                    ok = st.st_size >= 8 + n + end
        except (OSError, ValueError, KeyError, struct.error):
            ok = False
        self._shard_status[fn] = (key, ok)
        return ok

    @property
    def local_files(self) -> set:
        """Shards that are fully present right now (re-evaluated on every call; downloads may be in flight)."""
        return {fn for fn in set(self.weight_map.values()) if self.shard_is_complete(fn)}

    def is_local(self, name: str) -> bool:
        fn = self.weight_map.get(name)
        return fn is not None and self.shard_is_complete(fn)

    def names_with_prefix(self, prefix: str) -> List[str]:
        return sorted(n for n in self.weight_map if n.startswith(prefix))

    def layer_names(self, layer_idx: int, include_experts: bool = True) -> List[str]:
        names = self.names_with_prefix(f"model.layers.{layer_idx}.")
        if not include_experts:
            names = [n for n in names if not n.endswith(_EXPERT_TENSORS)]
        return names

    def layer_is_local(self, layer_idx: int, include_experts: bool = True) -> bool:
        names = self.layer_names(layer_idx, include_experts)
        return bool(names) and all(self.is_local(n) for n in names)

    def local_layers(self, num_layers: Optional[int] = None, include_experts: bool = True) -> List[int]:
        n = num_layers if num_layers is not None else self.args().num_hidden_layers
        return [i for i in range(n) if self.layer_is_local(i, include_experts)]

    def describe_local(self) -> str:
        n = self.args().num_hidden_layers
        parts = [f"layers {self.local_layers(n)} (complete)"]
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            parts.append(f"{name}: {'yes' if self.is_local(name) else 'no'}")
        mtp = self.names_with_prefix("model.mtp_layers.")
        parts.append(f"mtp: {'yes' if mtp and all(self.is_local(x) for x in mtp) else 'no'}")
        return "; ".join(parts)

    # ---- tensor access -----------------------------------------------------------------------------
    def _require(self, name: str) -> str:
        if name not in self.weight_map:
            raise MissingWeightsError(f"{name!r} is not a tensor of the checkpoint at {self.dir}")
        fn = self.weight_map[name]
        if not self.shard_is_complete(fn):
            state = "incomplete (still downloading?)" if (self.dir / fn).exists() else "not present"
            raise MissingWeightsError(
                f"tensor {name!r} lives in shard {fn}, which is {state} under {self.dir} "
                f"({len(self.local_files)}/{len(set(self.weight_map.values()))} shards local). "
                f"Locally available: {self.describe_local()}. "
                f"Download that shard or restrict the request to local layers."
            )
        return fn

    def _handle(self, fn: str):
        if fn not in self._handles:
            from safetensors import safe_open

            self._handles[fn] = safe_open(str(self.dir / fn), framework="pt", device="cpu")
        return self._handles[fn]

    def get(self, name: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Full tensor (memory-mapped bf16 unless ``dtype`` forces a converted copy)."""
        t = self._handle(self._require(name)).get_tensor(name)
        return t if dtype is None or t.dtype == dtype else t.to(dtype)

    def get_rows(self, name: str, index, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """``tensor[index]`` along dim 0 without materializing the rest (e.g. one expert of a fused tensor)."""
        t = self._handle(self._require(name)).get_slice(name)[index]
        return t if dtype is None or t.dtype == dtype else t.to(dtype)

    def get_many(self, names: Iterable[str], dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        names = list(names)
        missing = [n for n in names if not self.is_local(n)]
        if missing:
            self._require(missing[0])  # raises with a descriptive message
        return {n: self.get(n, dtype) for n in names}

    # ---- reference-named state dicts -----------------------------------------------------------------
    def layer_state_dict(
        self,
        layer_idx: int,
        dtype: Optional[torch.dtype] = None,
        include_experts: bool = True,
        strip_prefix: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Reference-named tensors of decoder layer ``layer_idx`` (``model.layers.{i}.`` prefix unless stripped)."""
        names = self.layer_names(layer_idx, include_experts)
        if not names:
            raise MissingWeightsError(f"layer {layer_idx} has no tensors in the checkpoint index")
        sd = hf_to_reference_state_dict(self.get_many(names, dtype))
        if strip_prefix:
            p = f"model.layers.{layer_idx}."
            sd = {k[len(p) :]: v for k, v in sd.items()}
        return sd

    def mtp_state_dict(self, dtype: Optional[torch.dtype] = None, mtp_idx: int = 0) -> Dict[str, torch.Tensor]:
        """``MotifMTP``-named tensors (``model.mtp_layers.{k}.`` stripped)."""
        p = f"model.mtp_layers.{mtp_idx}."
        names = self.names_with_prefix(p)
        if not names:
            raise MissingWeightsError(f"no MTP layer {mtp_idx} in the checkpoint index")
        return {k[len(p) :]: v for k, v in self.get_many(names, dtype).items()}

    def expert_source(self, layer_idx: int, dtype: Optional[torch.dtype] = None, cache_size: int = 64):
        """``e -> (gate_up [2I, D], down [D, I])`` reading single experts from the fused tensors (LRU-cached)."""
        gu_name = f"model.layers.{layer_idx}.moe.experts.gate_up_proj"
        dn_name = f"model.layers.{layer_idx}.moe.experts.down_proj"
        self._require(gu_name)
        self._require(dn_name)
        lru: "OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

        def source(e: int):
            if e in lru:
                lru.move_to_end(e)
                return lru[e]
            w = (self.get_rows(gu_name, e, dtype), self.get_rows(dn_name, e, dtype))
            lru[e] = w
            if len(lru) > cache_size:
                lru.popitem(last=False)
            return w

        return source


def load_reference_model(
    ckpt_dir: Optional[os.PathLike] = None,
    layer_ids: Optional[Iterable[int]] = (0, 1, 2),
    dtype: torch.dtype = torch.bfloat16,
    lazy_experts: bool = False,
    args: Optional[MotifArgs] = None,
    checkpoint: Optional[MotifCheckpoint] = None,
) -> MotifForCausalLM:
    """Build a (partial) reference model with real weights.

    ``layer_ids=(0, 1, 2)`` builds the exact early-exit prefix model that the local shards support (embed, layers
    0-2, final norm, lm_head). bf16 tensors are used in place (memory-mapped); other dtypes are converted copies.
    """
    ckpt = checkpoint or MotifCheckpoint(ckpt_dir)
    args = args or ckpt.args()
    ids = list(range(args.num_hidden_layers)) if layer_ids is None else sorted(int(i) for i in layer_ids)
    with torch.device("meta"):
        model = MotifForCausalLM(args, ids, materialize_experts=not lazy_experts)
    sd = {
        "model.embed_tokens.weight": ckpt.get("model.embed_tokens.weight", dtype),
        "model.norm.weight": ckpt.get("model.norm.weight", dtype),
        "lm_head.weight": ckpt.get("lm_head.weight", dtype),
    }
    for i in ids:
        sd.update(ckpt.layer_state_dict(i, dtype, include_experts=not lazy_experts))
    model.load_state_dict(sd, strict=True, assign=True)
    if lazy_experts:
        for i in ids:
            layer = model.model.layers[str(i)]
            if layer.is_moe:
                layer.moe.experts.expert_source = ckpt.expert_source(i, dtype)
    model.requires_grad_(False)
    return model


def load_mtp(
    ckpt_dir: Optional[os.PathLike] = None,
    dtype: torch.dtype = torch.bfloat16,
    args: Optional[MotifArgs] = None,
    checkpoint: Optional[MotifCheckpoint] = None,
) -> MotifMTP:
    ckpt = checkpoint or MotifCheckpoint(ckpt_dir)
    args = args or ckpt.args()
    with torch.device("meta"):
        mtp = MotifMTP(args)
    mtp.load_state_dict(ckpt.mtp_state_dict(dtype), strict=True, assign=True)
    mtp.requires_grad_(False)
    return mtp


# =================================================================================================
# random weights for tiny configs
# =================================================================================================
def random_state_dict(
    args: MotifArgs, layer_ids: Optional[Iterable[int]] = None, seed: int = 0, include_mtp: bool = False
) -> Dict[str, torch.Tensor]:
    """fp32, reference-named, "realistic" random weights for ``MotifForCausalLM(args, layer_ids)``.

    Scales keep every nonlinearity in a non-degenerate regime: linears ~ N(0, 1/fan_in), norm gammas in
    [0.5, 1.5], mHC alphas ~0.3-0.6 with N(0, 0.3) biases (non-trivial H_res), PolyNorm biases in [-1, 1] (so the
    routed +-0.5 clamp is exercised while dense/shared biases are not clamped), expert_bias = 1 + N(0, 0.1)
    (changes top-k selection vs. the unbiased scores for some tokens).
    """
    g = torch.Generator().manual_seed(seed)
    ids = list(range(args.num_hidden_layers)) if layer_ids is None else sorted(int(i) for i in layer_ids)
    D, V, E = args.hidden_size, args.vocab_size, args.mhc_expansion_rate

    def normal(*shape, std=1.0, mean=0.0):
        return torch.randn(*shape, generator=g) * std + mean

    def uniform(*shape, lo, hi):
        return torch.rand(*shape, generator=g) * (hi - lo) + lo

    def linear(out_f, in_f):
        return normal(out_f, in_f, std=in_f**-0.5)

    def gamma(n):
        return uniform(n, lo=0.5, hi=1.5)

    def mlp(prefix, inter):
        return {
            f"{prefix}.gate_proj.weight": linear(inter, D),
            f"{prefix}.up_proj.weight": linear(inter, D),
            f"{prefix}.down_proj.weight": linear(D, inter),
            f"{prefix}.act_fn.weight": normal(3),
            f"{prefix}.act_fn.bias": uniform(1, lo=-1.0, hi=1.0),
        }

    def attn(prefix):
        H, hd, v, r = args.num_attention_heads, args.head_dim, args.v_head_dim, args.kv_lora_rank
        sd = {
            f"{prefix}.wq_a.weight": linear(args.q_lora_rank, D),
            f"{prefix}.q_norm.weight": uniform(args.q_lora_rank, lo=0.5, hi=1.0),
            f"{prefix}.wq_b.weight": linear(H * hd, args.q_lora_rank),
            f"{prefix}.wkv_a.weight": linear(r + args.qk_rope_head_dim, D),
            f"{prefix}.kv_norm.weight": uniform(r, lo=0.5, hi=1.0),
            f"{prefix}.wkv_b.weight": linear(args.num_key_value_heads * (args.qk_nope_head_dim + v), r),
            f"{prefix}.lambda_proj.weight": linear(args.n_signal_heads, D),
            f"{prefix}.wo.weight": linear(D, args.n_signal_heads * v),
        }
        if args.elementwise_attn_output_gate:
            sd[f"{prefix}.wq_b_gate.weight"] = linear(args.n_signal_heads * v, args.q_lora_rank)
        return sd

    def mhc(prefix):
        return {
            f"{prefix}.proj_merged.weight": linear(E * E + 2 * E, E * D),
            f"{prefix}.rms_norm.weight": gamma(E * D),
            f"{prefix}.bias_pre": normal(E, std=0.3),
            f"{prefix}.bias_post": normal(E, std=0.3),
            f"{prefix}.bias_res": normal(E, E, std=0.3),
            f"{prefix}.alpha_pre": uniform(1, lo=0.3, hi=0.6),
            f"{prefix}.alpha_post": uniform(1, lo=0.3, hi=0.6),
            f"{prefix}.alpha_res": uniform(1, lo=0.3, hi=0.6),
        }

    sd = {
        "model.embed_tokens.weight": normal(V, D),
        "model.norm.weight": gamma(D),
        "lm_head.weight": linear(V, D),
    }
    for i in ids:
        p = f"model.layers.{i}"
        sd.update(attn(f"{p}.self_attn"))
        sd[f"{p}.input_layernorm.weight"] = gamma(D)
        sd[f"{p}.post_attention_layernorm.weight"] = gamma(D)
        if args.mhc_enabled:
            sd.update(mhc(f"{p}.mhc_attn"))
            sd.update(mhc(f"{p}.mhc_ffn"))
        if args.is_moe_layer(i):
            Ne, I = args.num_experts, args.moe_intermediate_size
            sd[f"{p}.moe.router.gate.weight"] = linear(Ne, D)
            if args.load_balance_coeff is not None:
                sd[f"{p}.moe.expert_bias"] = normal(Ne, std=0.1, mean=1.0)
            sd[f"{p}.moe.experts.gate_up_proj"] = normal(Ne, 2 * I, D, std=D**-0.5)
            sd[f"{p}.moe.experts.down_proj"] = normal(Ne, D, I, std=I**-0.5)
            sd[f"{p}.moe.experts.act_fn.weight"] = normal(Ne, 3)
            sd[f"{p}.moe.experts.act_fn.bias"] = uniform(Ne, 1, lo=-1.0, hi=1.0)
            if args.num_shared_experts > 0:
                sd.update(mlp(f"{p}.moe.shared_experts", args.shared_intermediate_size))
        else:
            sd.update(mlp(f"{p}.mlp", args.intermediate_size))
    if include_mtp:
        p = "mtp"
        sd[f"{p}.embed_norm.weight"] = gamma(D)
        sd[f"{p}.input_proj.weight"] = linear(D, 2 * D)
        sd[f"{p}.input_layernorm.weight"] = gamma(D)
        sd[f"{p}.post_attention_layernorm.weight"] = gamma(D)
        sd[f"{p}.final_layernorm.weight"] = gamma(D)
        sd.update(attn(f"{p}.self_attn"))
        sd.update(mlp(f"{p}.mlp", args.intermediate_size))
    return sd


def round_state_dict(sd: Mapping[str, torch.Tensor], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Round every tensor to ``dtype`` and back to fp32 (so two models built in ``dtype`` hold identical values)."""
    return {k: v.to(dtype).to(torch.float32) for k, v in sd.items()}


def build_random_model(
    args: MotifArgs,
    layer_ids: Optional[Iterable[int]] = None,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    state_dict: Optional[Mapping[str, torch.Tensor]] = None,
) -> MotifForCausalLM:
    """Random-weight reference model in ``dtype`` (weights from :func:`random_state_dict` unless given)."""
    model = MotifForCausalLM(args, layer_ids).to(dtype)
    sd = state_dict if state_dict is not None else random_state_dict(args, layer_ids, seed)
    sd = {k: v for k, v in sd.items() if not k.startswith("mtp.")}
    model.load_state_dict(sd, strict=True)
    return model
