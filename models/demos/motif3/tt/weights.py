# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 weights on TT: (1) lazy HF loading, (2) torch-side transforms, (3) cached mesh upload.

Conventions used by every transform below
------------------------------------------
* HF ``nn.Linear`` weights are ``[out, in]``. Every transform returns the **device orientation** ``[in, out]``
  (``y = x @ W``, as ``ttnn.matmul(x, W)`` / ``ttnn.linear(x, W)`` expect).
* "Per-chip" transforms take a TP index ``tp`` (0..7) and return that chip's block. :func:`stack_tp` concatenates
  the 8 blocks along one dim so that a TP mapper on that dim (``as_tensor(..., tp_dim=d)``) hands chip ``tp``
  exactly ``f(tp)``. Rows of the DP axis receive identical copies unless ``dp_dim`` is given.
* Math is fp32 on the host; tensors are rounded **once** to the device dtype at upload (design §2.3.11).
* Head bookkeeping (design §2.3.4; HF ``modeling_motif.py:745-775``): q head ``h`` -> group ``g = h // 5``; heads
  ``5g..5g+3`` are signal with signal index ``s = 4g + j``; ``5g + 4`` is the noise head. TP chip ``tp`` owns groups
  ``{2tp, 2tp+1}`` = q heads ``[10tp, 10tp+10)`` = signal heads ``[8tp, 8tp+8)``; all slices are contiguous.
* Expert placement (design §2.3.7): EP over all 32 chips, chip linear index ``k = dp * 8 + tp`` (orientation
  independent, ``MeshAxes.chip_index``) holds routed experts ``[12k, 12k + 12)`` in increasing order. The host
  layout for that placement is :func:`ep_layout` (``[E, ...] -> [dp, E/dp, ...]``) + ``as_tensor(dp_dim=0, tp_dim=1)``,
  which gives every chip a ``[1, 12, ...]`` tensor.

Cache (design §2.3.11): :func:`as_tensor` wraps ``ttnn.as_tensor(cache_file_name=...)`` with
``<TT_CACHE_PATH>/<version-tag>/mesh<R>x<C>/{L<nn>|global}/<name>__<mapping>`` (+ ttnn's
``_dtype_<D>_layout_<L>.tensorbin`` suffix). The torch source may be a callable, so a cache hit never touches the
safetensors. Bump ``model_config.CACHE_FORMAT_VERSION`` whenever a transform here changes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

import ttnn

from .model_config import DEFAULT_HF_META_DIR, MeshAxes, MotifTTConfig, resolve_weights_dir

TorchSource = Union[torch.Tensor, Callable[[], torch.Tensor]]


# ============================================================================================================
# (1) Loading
# ============================================================================================================
class MissingWeightError(KeyError):
    """A tensor is not in the checkpoint index, or its shard is not (completely) on disk."""

    def __str__(self):
        return str(self.args[0]) if self.args else ""


def hf_name(layer: Optional[int], suffix: str) -> str:
    """``model.layers.{layer}.{suffix}`` (``layer=None`` returns ``suffix`` unchanged, e.g. ``lm_head.weight``)."""
    return suffix if layer is None else f"model.layers.{int(layer)}.{suffix}"


# ---- the MTP layer's names (features design §3.6.5; README §17) ---------------------------------------------------
MTP_LAYERS_PREFIX = "model.mtp_layers"  # the checkpoint's MTP layers (Motif-3: one, model.mtp_layers.0, shard 104)
MTP_ATTN_TENSORS = ("wq_a", "q_norm", "wq_b", "wq_b_gate", "wkv_a", "kv_norm", "wkv_b", "lambda_proj", "wo")
MTP_NORMS = ("embed_norm", "input_layernorm", "post_attention_layernorm", "final_layernorm")


def mtp_name(suffix: Optional[str] = None, mtp_idx: int = 0) -> str:
    """``model.mtp_layers.{mtp_idx}.{suffix}``; ``suffix=None`` gives the module path ``model.mtp_layers.{mtp_idx}``
    (e.g. ``mtp_name("self_attn")`` is the MTP layer's ``MotifAttention(weight_prefix=)``)."""
    p = f"{MTP_LAYERS_PREFIX}.{int(mtp_idx)}"
    return p if suffix is None else f"{p}.{suffix}"


def mtp_tensor_names(mtp_idx: int = 0) -> List[str]:
    """The 19 checkpoint tensors of MTP layer ``mtp_idx``, sorted: the 9 GDLA attention tensors, the dense PolyNorm
    MLP (``gate_proj`` / ``up_proj`` / ``down_proj`` + ``act_fn.{weight,bias}``), ``input_proj`` and the 4 RMSNorms
    (reference ``MotifMTP``; all in shard 104 of revision 2ed2ed5c)."""
    s = [f"self_attn.{k}.weight" for k in MTP_ATTN_TENSORS]
    s += [f"mlp.{k}_proj.weight" for k in ("gate", "up", "down")] + ["mlp.act_fn.weight", "mlp.act_fn.bias"]
    s += ["input_proj.weight"] + [f"{n}.weight" for n in MTP_NORMS]
    return sorted(mtp_name(x, mtp_idx) for x in s)


def mtp_names_in(source, mtp_idx: int = 0) -> List[str]:
    """MTP tensors a weight source lists (``source.keys()``: the checkpoint index, or a dict's keys), sorted."""
    p = mtp_name(None, mtp_idx) + "."
    return sorted(n for n in source.keys() if n.startswith(p))


def mtp_layer_available(source, mtp_idx: int = 0) -> bool:
    """Every tensor of MTP layer ``mtp_idx`` that ``tt.mtp.MotifMTP`` reads (:func:`mtp_tensor_names`, all 19) is listed
    by the source and readable (``source.available``: an ``HFWeightLoader`` also checks that the shard is completely on
    disk). A source that lists only some of them does not count: the layer would fail at load time."""
    return all(source.available(n) for n in mtp_tensor_names(mtp_idx))


class HFWeightLoader:
    """Lazy, index-driven access to the HF safetensors checkpoint (design §2.3.11; study 01 §6.2, §9.3).

    * Opens each shard once with ``safetensors.safe_open`` (memory-mapped; nothing is read until used).
    * :meth:`get_rows` reads a dim-0 slice through ``get_slice`` (e.g. 12 experts of the fused
      ``[384, 2560, 4096]`` tensor) without materializing the rest.
    * A shard counts as present only if it exists with the size listed in ``hf_meta/tree.json`` (downloads in
      progress are therefore never opened). :meth:`layer_available` / :meth:`available_layers` report which
      decoder layers are fully local; ``.download_state.json`` (``complete_layers``) is exposed as-is.
    """

    def __init__(self, weights_dir: Optional[Union[str, os.PathLike]] = None, *, tree_json: Optional[Path] = None):
        self.dir = Path(weights_dir) if weights_dir is not None else resolve_weights_dir()
        index_path = self.dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"no model.safetensors.index.json under {self.dir}")
        self.weight_map: Dict[str, str] = json.loads(index_path.read_text())["weight_map"]
        self._handles: Dict[str, object] = {}
        tree = Path(tree_json) if tree_json is not None else DEFAULT_HF_META_DIR / "tree.json"
        self._sizes: Dict[str, int] = {}
        if tree.is_file():
            try:
                self._sizes = {
                    e["path"]: int(e["size"]) for e in json.loads(tree.read_text()) if e.get("type") == "file"
                }
            except Exception:  # pragma: no cover - malformed tree.json: fall back to existence checks
                self._sizes = {}

    # ---- metadata ----------------------------------------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def keys(self) -> List[str]:
        return sorted(self.weight_map)

    def shard_of(self, name: str) -> str:
        if name not in self.weight_map:
            raise MissingWeightError(f"{name!r} is not a tensor of the checkpoint at {self.dir}")
        return self.weight_map[name]

    def shard_present(self, fn: str) -> bool:
        p = self.dir / fn
        if not p.is_file():
            return False
        expected = self._sizes.get(fn)
        return expected is None or p.stat().st_size == expected

    def available(self, name: str) -> bool:
        return name in self.weight_map and self.shard_present(self.weight_map[name])

    def layer_names(self, layer: int) -> List[str]:
        p = f"model.layers.{int(layer)}."
        return sorted(n for n in self.weight_map if n.startswith(p))

    def layer_available(self, layer: int) -> bool:
        names = self.layer_names(layer)
        return bool(names) and all(self.available(n) for n in names)

    def available_layers(self, num_layers: int = 53) -> List[int]:
        return [i for i in range(num_layers) if self.layer_available(i)]

    def download_state_layers(self) -> List[int]:
        p = self.dir / ".download_state.json"
        if not p.is_file():
            return []
        try:
            return sorted(int(i) for i in json.loads(p.read_text()).get("complete_layers", []))
        except Exception:  # pragma: no cover
            return []

    # ---- tensor access ---------------------------------------------------------------------------------------
    def _handle(self, name: str):
        fn = self.shard_of(name)
        if fn not in self._handles:
            if not self.shard_present(fn):
                raise MissingWeightError(
                    f"tensor {name!r} lives in shard {fn}, which is missing or incomplete under {self.dir} "
                    f"(complete layers per .download_state.json: {self.download_state_layers()})"
                )
            from safetensors import safe_open

            self._handles[fn] = safe_open(str(self.dir / fn), framework="pt", device="cpu")
        return self._handles[fn]

    def shape(self, name: str) -> Tuple[int, ...]:
        return tuple(self._handle(name).get_slice(name).get_shape())

    def get(self, name: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Full tensor (bf16 as stored unless ``dtype`` is given)."""
        t = self._handle(name).get_tensor(name)
        return t if dtype is None or t.dtype == dtype else t.to(dtype)

    def get_rows(self, name: str, start: int, stop: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """``tensor[start:stop]`` along dim 0, reading only those rows."""
        t = self._handle(name).get_slice(name)[int(start) : int(stop)]
        return t if dtype is None or t.dtype == dtype else t.to(dtype)

    def close(self) -> None:
        self._handles.clear()


class DictWeightSource:
    """Same read API as :class:`HFWeightLoader` over an in-memory ``{hf_name: tensor}`` dict (random weights).

    Reference-named mHC tensors (``mhc_*.proj_merged.weight``, see ``reference/weights.py``) are accepted too;
    :func:`mhc_projection_from_source` handles both spellings.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor]):
        self.tensors = dict(tensors)

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def has(self, name: str) -> bool:
        return name in self.tensors

    def keys(self) -> List[str]:
        return sorted(self.tensors)

    def available(self, name: str) -> bool:
        return name in self.tensors

    def shape(self, name: str) -> Tuple[int, ...]:
        return tuple(self.tensors[name].shape)

    def get(self, name: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if name not in self.tensors:
            raise MissingWeightError(f"{name!r} not in the weight dict")
        t = self.tensors[name]
        return t if dtype is None or t.dtype == dtype else t.to(dtype)

    def get_rows(self, name: str, start: int, stop: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.get(name, dtype)[int(start) : int(stop)]


# ============================================================================================================
# (2) Torch-side transforms
# ============================================================================================================
def _f32(t: torch.Tensor) -> torch.Tensor:
    """Host math precision: fp32, or fp64 when the input already is fp64 (exactness tests)."""
    return t if t.dtype == torch.float64 else t.to(torch.float32)


def stack_tp(fn: Callable[[int], torch.Tensor], cfg: MotifTTConfig, dim: int = -1) -> torch.Tensor:
    """``cat([fn(0), ..., fn(tp-1)], dim)``: the host tensor whose TP shard on ``dim`` is ``fn(tp)``."""
    return torch.cat([fn(t) for t in range(cfg.tp)], dim=dim)


# ---- GDLA attention (design §2.3.4; study 01 §4.1-4.5) -------------------------------------------------
def split_wkv_a(wkv_a: torch.Tensor, cfg: MotifTTConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """``wkv_a [576, 4096]`` -> ``(W_DKV [512, 4096], W_KR [64, 4096])`` (latent rows 0..511, k_pe rows 512..575;
    the rope rows are already half-split, no permutation; study 01 §3.10)."""
    r = cfg.kv_lora_rank
    assert wkv_a.shape[0] == r + cfg.rope_dim, wkv_a.shape
    return wkv_a[:r], wkv_a[r:]


def split_wq_b(wq_b: torch.Tensor, cfg: MotifTTConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """``wq_b [80*192, 1024]`` -> ``(W_UQ [80, 128, 1024], W_QR [80, 64, 1024])``: head ``h`` occupies rows
    ``[192h, 192h + 192)``, of which ``+0..127`` are nope and ``+128..191`` rope (study 01 §4.1)."""
    w = wq_b.reshape(cfg.n_heads, cfg.head_dim, wq_b.shape[-1])
    return w[:, : cfg.qk_nope_head_dim], w[:, cfg.qk_nope_head_dim :]


def split_wkv_b(wkv_b: torch.Tensor, cfg: MotifTTConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """``wkv_b [16*256, 512]`` -> ``(W_UK [16, 128, 512], W_UV [16, 128, 512])``: group ``g`` rows ``[256g, 256g+256)``,
    ``+0..127`` = W_UK,g and ``+128..255`` = W_UV,g (study 01 §4.1)."""
    w = wkv_b.reshape(cfg.n_kv_heads, cfg.qk_nope_head_dim + cfg.v_head_dim, wkv_b.shape[-1])
    return w[:, : cfg.qk_nope_head_dim], w[:, cfg.qk_nope_head_dim :]


def fold_kv_norm(w_uk: torch.Tensor, w_uv: torch.Tensor, kv_norm_gamma: torch.Tensor):
    """Fold the ``kv_norm`` gamma into the up-projections column-wise: ``W' = W diag(gamma)`` (fp32).

    The cache then stores the gamma-free unit-RMS latent ``n = c_raw / rms(c_raw)`` (better bfp8 conditioning, study
    01 §4.5 / N11) and ``k_nope = W_UK' n``, ``v = W_UV' n`` reproduce HF's ``wkv_b(kv_norm(c_raw))`` exactly in real
    arithmetic (design §2.3.4 weights table; ``kv_norm`` runs weightless on device)."""
    g = _f32(kv_norm_gamma)[None, None, :]
    return _f32(w_uk) * g, _f32(w_uv) * g


def signal_order_for_chip(cfg: MotifTTConfig, tp: int) -> List[int]:
    """All 64 signal indices with chip ``tp``'s 8 first (``8tp .. 8tp+7``), the rest in increasing order."""
    local = list(cfg.chip_heads(tp).signal_heads)
    return local + [s for s in range(cfg.n_signal_heads) if s not in local]


def latent_projection(
    wq_a: torch.Tensor, wkv_a: torch.Tensor, lambda_proj: torch.Tensor, lambda_rows: Optional[Sequence[int]] = None
) -> torch.Tensor:
    """Fused latent projection ``W_lat = [wq_a; wkv_a; lambda_proj]^T`` -> ``[4096, 1664]`` fp32 (design §2.3.4).

    Output columns of ``x @ W_lat``: ``[0, 1024)`` cq (-> q_norm), ``[1024, 1536)`` c_raw (-> weightless rms_norm =
    the cached latent n), ``[1536, 1600)`` raw k_pe (-> RoPE), ``[1600, 1664)`` lambda logits in ``lambda_rows`` order
    (default ``s = 0..63``).
    """
    lam = lambda_proj if lambda_rows is None else lambda_proj[list(lambda_rows)]
    return torch.cat([_f32(wq_a), _f32(wkv_a), _f32(lam)], dim=0).t().contiguous()


def latent_projection_for_chip(
    wq_a: torch.Tensor, wkv_a: torch.Tensor, lambda_proj: torch.Tensor, cfg: MotifTTConfig, tp: int
) -> torch.Tensor:
    """:func:`latent_projection` with chip ``tp``'s 8 lambda rows first, so ``sigmoid(lam[:, 1600:1608])`` is the
    local signal heads' lambda with the **same** slice on every chip (no per-chip offset in a trace). The other 56
    lambda columns are unused. Upload with ``stack_tp(..., dim=1)`` + ``tp_dim=1``."""
    return latent_projection(wq_a, wkv_a, lambda_proj, signal_order_for_chip(cfg, tp))


def wq_b_for_chip(wq_b: torch.Tensor, cfg: MotifTTConfig, tp: int, *, layout: str = "split") -> torch.Tensor:
    """Chip ``tp``'s q up-projection ``[1024, 1920]`` for its 10 q heads (design §2.3.4: rows ``[1920tp, +1920)``).

    * ``layout="split"`` (default): columns ``[0, 1280)`` = q_nope of local heads 0..9 (``h_loc * 128 + d``),
      ``[1280, 1920)`` = q_pe (``h_loc * 64 + d``) -> two contiguous slices, no per-head interleave.
    * ``layout="interleaved"``: HF order, ``h_loc * 192 + d`` (nope 0..127, rope 128..191 per head).
    """
    H = cfg.q_heads_per_chip
    heads = cfg.chip_heads(tp).q_heads
    w = _f32(wq_b).reshape(cfg.n_heads, cfg.head_dim, -1)[heads.start : heads.stop]  # [10, 192, 1024]
    if layout == "interleaved":
        rows = w.reshape(H * cfg.head_dim, -1)
    elif layout == "split":
        nope = w[:, : cfg.qk_nope_head_dim].reshape(H * cfg.qk_nope_head_dim, -1)
        rope = w[:, cfg.qk_nope_head_dim :].reshape(H * cfg.rope_dim, -1)
        rows = torch.cat([nope, rope], dim=0)
    else:
        raise ValueError(f"unknown wq_b layout {layout!r}")
    return rows.t().contiguous()


def wq_b_gate_for_chip(wq_b_gate: torch.Tensor, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """Elementwise gate projection of chip ``tp``'s 8 signal heads: ``wq_b_gate[1024tp : +1024]^T`` -> ``[1024, 1024]``
    (output column ``s_loc * 128 + d``; gate = sigmoid of it, design §2.3.4 step 4)."""
    s = cfg.chip_heads(tp).signal_heads
    v = cfg.v_head_dim
    return _f32(wq_b_gate)[s.start * v : s.stop * v].t().contiguous()


def kv_up_for_chip(
    wkv_b: torch.Tensor, kv_norm_gamma: torch.Tensor, cfg: MotifTTConfig, tp: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """gamma-folded ``(W_UK' [2, 128, 512], W_UV' [2, 128, 512])`` of chip ``tp``'s groups (rows ``[512tp, +512)``)."""
    w_uk, w_uv = fold_kv_norm(*split_wkv_b(wkv_b, cfg), kv_norm_gamma)
    g = cfg.chip_heads(tp).groups
    return w_uk[g.start : g.stop].contiguous(), w_uv[g.start : g.stop].contiguous()


def absorb_weights_for_chip(
    wkv_b: torch.Tensor, kv_norm_gamma: torch.Tensor, cfg: MotifTTConfig, tp: int
) -> torch.Tensor:
    """Decode absorb (design §2.3.4 step 6): ``q_lat[g] = q_nope[g] @ W_UK'_g`` with ``W_UK'_g [128, 512]`` -> per chip
    ``[2, 128, 512]`` (bmm over the 2 local groups; ``q_nope.view(2, 5L, 128)``)."""
    return kv_up_for_chip(wkv_b, kv_norm_gamma, cfg, tp)[0]


def unabsorb_weights_for_chip(
    wkv_b: torch.Tensor, kv_norm_gamma: torch.Tensor, cfg: MotifTTConfig, tp: int
) -> torch.Tensor:
    """Decode un-absorb (design §2.3.4 step 10): ``D[g] = d_lat[g] @ W_UV'_g^T`` -> per chip ``[2, 512, 128]``."""
    return kv_up_for_chip(wkv_b, kv_norm_gamma, cfg, tp)[1].transpose(-1, -2).contiguous()


def absorb_weights_per_head_for_chip(
    wkv_b: torch.Tensor, kv_norm_gamma: torch.Tensor, cfg: MotifTTConfig, tp: int
) -> torch.Tensor:
    """Per-head absorb matrices ``[10, 128, 512]`` (``W_UK'_{h // 5}`` repeated for the 5 heads of each group; the
    vLLM-fork / FlashMLA ``W_UK_T`` layout) for a per-head bmm of ``q_nope [10, L, 128]``."""
    return absorb_weights_for_chip(wkv_b, kv_norm_gamma, cfg, tp).repeat_interleave(cfg.heads_per_group, dim=0)


def absorbed_q_weights_per_head_for_chip(
    wq_b: torch.Tensor, wkv_b: torch.Tensor, kv_norm_gamma: torch.Tensor, cfg: MotifTTConfig, tp: int
) -> torch.Tensor:
    """Fully absorbed q projection per local head: ``q_lat_h = cq @ (W_UQ,h^T W_UK'_{g(h)})`` -> ``[10, 1024, 512]``.

    Optional (study 01 §4.2): it removes the nope projection + absorb bmm but holds 5.2M params per head
    (~21 MB bf16 per chip-layer vs ~0.4 MB), so draft 1 uses :func:`absorb_weights_for_chip` instead."""
    w_uq, _ = split_wq_b(_f32(wq_b), cfg)  # [80, 128, 1024]
    heads = cfg.chip_heads(tp).q_heads
    w_uk_h = absorb_weights_per_head_for_chip(wkv_b, kv_norm_gamma, cfg, tp)  # [10, 128, 512]
    return torch.einsum("hni,hnr->hir", w_uq[heads.start : heads.stop], w_uk_h).contiguous()


def prefill_kv_expansion_for_chip(
    wkv_b: torch.Tensor, kv_norm_gamma: torch.Tensor, cfg: MotifTTConfig, tp: int, *, v_pad_to: Optional[int] = None
) -> torch.Tensor:
    """Prefill expansion from the cached latent (design §2.3.4 prefill step 2): ``n @ E`` -> per local group
    ``[k_nope (128) | v (128) | 0 (v_pad_to - 128)]``, i.e. ``E [512, 2 * (128 + v_pad_to)]`` (v_pad_to defaults to
    head_dim = 192, so each group block is 320 wide). Then ``K_g = cat(blk[:, :128], k_pe)`` [S, 192] and the padded
    ``V_g = blk[:, 128:320]`` [S, 192] (SDPA prefill needs V's head dim == Q's)."""
    v_pad_to = cfg.head_dim if v_pad_to is None else int(v_pad_to)
    w_uk, w_uv = kv_up_for_chip(wkv_b, kv_norm_gamma, cfg, tp)  # [2, 128, 512] each
    blocks = []
    for g in range(w_uk.shape[0]):
        zeros = torch.zeros(v_pad_to - cfg.v_head_dim, w_uk.shape[-1], dtype=w_uk.dtype)
        blocks.append(torch.cat([w_uk[g], w_uv[g], zeros], dim=0))  # [128 + v_pad_to, 512]
    return torch.cat(blocks, dim=0).t().contiguous()


def wo_for_chip(wo: torch.Tensor, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """Row-parallel output projection: input columns of chip ``tp``'s 8 signal heads, ``wo[:, 1024tp : +1024]^T``
    -> ``[1024, 4096]`` (row ``s_loc * 128 + d``); the partial sums are closed with ``all_reduce(tp)``."""
    s = cfg.chip_heads(tp).signal_heads
    v = cfg.v_head_dim
    return _f32(wo)[:, s.start * v : s.stop * v].t().contiguous()


# ---- mHC (design §2.3.3) -----------------------------------------------------------------------------------
def mhc_fused_projection(
    proj_pre: torch.Tensor, proj_post: torch.Tensor, proj_res: torch.Tensor, rms_gamma: torch.Tensor
) -> torch.Tensor:
    """``fn = (W_pre || W_post || W_res) * gamma_mhc`` -> ``[24, 16384]`` fp32 (rows: pre 0..3, post 4..7,
    res 8..23 with ``res[4i + j]`` = H[i][j]); fold in fp32, round once to bf16 at upload (design §2.3.3 step 1;
    = the fork's tilelang ``fn`` buffer). On device ``mixes = (X @ fn^T) * rsqrt(sum(X^2) / 16384 + 1e-6)``,
    which equals ``rms_norm(X; gamma) @ (W_pre || W_post || W_res)^T`` (the rsqrt commutes with the linear map)."""
    w = torch.cat([_f32(proj_pre), _f32(proj_post), _f32(proj_res)], dim=0)
    return w * _f32(rms_gamma)[None, :]


def mhc_projection_blocks(fn: torch.Tensor, n_streams: int = 4, pad_to: int = 32) -> torch.Tensor:
    """Per-stream device blocks ``[1, 4, 4096, 32]``: block ``i`` = ``fn[:, 4096i : 4096(i+1)]^T`` with the 24 real
    columns zero-padded to 32, so ``sum_dim1(matmul(X [1,4,T,4096], blocks))`` = ``X_flat @ fn^T`` (design §2.3.3)."""
    n_out, width = fn.shape
    d = width // n_streams
    blocks = fn.reshape(n_out, n_streams, d).permute(1, 2, 0)  # [4, 4096, 24]
    if pad_to > n_out:
        blocks = torch.cat([blocks, torch.zeros(n_streams, d, pad_to - n_out, dtype=blocks.dtype)], dim=-1)
    return blocks.unsqueeze(0).contiguous()


def mhc_projection_from_source(source, prefix: str) -> torch.Tensor:
    """``(W_pre || W_post || W_res)`` ``[24, 16384]`` from HF names (``{prefix}.proj_pre.weight`` ...) or the reference
    name ``{prefix}.proj_merged.weight``. ``prefix`` e.g. ``model.layers.3.mhc_attn``."""
    merged = f"{prefix}.proj_merged.weight"
    if source.has(merged):
        return source.get(merged)
    return torch.cat([source.get(f"{prefix}.proj_{p}.weight") for p in ("pre", "post", "res")], dim=0)


def mhc_scalars(source, prefix: str) -> Dict[str, torch.Tensor]:
    """fp32 ``alpha_{pre,post,res}`` [1], ``bias_pre``/``bias_post`` [4], ``bias_res`` [16] (row-major ``4i + j``),
    the inputs of the Sinkhorn constants builder (``mhc_split_sinkhorn`` consts, design §2.3.3 step 2)."""
    out = {k: _f32(source.get(f"{prefix}.{k}")).reshape(-1) for k in ("alpha_pre", "alpha_post", "alpha_res")}
    out["bias_pre"] = _f32(source.get(f"{prefix}.bias_pre")).reshape(-1)
    out["bias_post"] = _f32(source.get(f"{prefix}.bias_post")).reshape(-1)
    out["bias_res"] = _f32(source.get(f"{prefix}.bias_res")).reshape(-1)
    return out


# ---- PolyNorm, dense MLP, shared expert (design §2.3.5, §2.3.6) ----------------------------------------------
def polynorm_coefficients(
    weight: torch.Tensor, bias: torch.Tensor, *, bias_clamp: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(c = sigmoid(w) fp32 [..., 3], b fp32 [..., 1])`` with ``poly = c0 N(g^3) + c1 N(g^2) + c2 N(g) + b``.

    ``bias_clamp`` (0.5) applies to **routed experts only** (``GroupedPolyNorm``); the dense and shared ``PolyNormTorch``
    bias is not clamped (HF ``modeling_motif.py:49-80`` vs ``112-137``). The x0.5 output scale is *not* here: it is
    folded into the down projection (:func:`mlp_down_for_chip`, :func:`experts_down`), which is exact."""
    c = torch.sigmoid(_f32(weight))
    b = _f32(bias)
    if bias_clamp is not None:
        b = b.clamp(-float(bias_clamp), float(bias_clamp))
    return c, b


def mlp_gate_up_for_chip(gate: torch.Tensor, up: torch.Tensor, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """Column-parallel fused ``[gate_c^T | up_c^T]`` -> ``[4096, 2 * I/8]`` (gate first). ``I`` comes from the weight
    (12288 dense -> 1536 per chip; 1280 shared -> 160 per chip). PolyNorm moments are then local partial sums that
    need ``all_reduce(tp)`` (design §2.3.5)."""
    inter = gate.shape[0]
    n = inter // cfg.tp
    sl = slice(n * tp, n * tp + n)
    return torch.cat([_f32(gate)[sl].t(), _f32(up)[sl].t()], dim=1).contiguous()


def mlp_down_for_chip(
    down: torch.Tensor, cfg: MotifTTConfig, tp: int, output_scale: Optional[float] = None
) -> torch.Tensor:
    """Row-parallel ``down[:, I/8 tp : +I/8]^T * output_scale`` -> ``[I/8, 4096]`` (x0.5 PolyNorm output scale folded:
    ``(0.5 h) W = h (0.5 W)``, exact for a power of two)."""
    scale = cfg.polynorm_output_scale if output_scale is None else float(output_scale)
    inter = down.shape[1]
    n = inter // cfg.tp
    return (_f32(down)[:, n * tp : n * tp + n].t() * scale).contiguous()


# ---- routed experts (design §2.3.7, §2.3.11) ------------------------------------------------------------------
def experts_of_chip(cfg: MotifTTConfig, dp: int, tp: int) -> range:
    """Routed experts on chip (dp, tp): ``[12k, 12k + 12)`` with ``k = dp * 8 + tp``."""
    return cfg.experts_of_chip(dp, tp)


def experts_gate_up(gate_up: torch.Tensor) -> torch.Tensor:
    """``gate_up_proj [n, 2560, 4096]`` -> ``[n, 4096, 2560]``: columns ``0..1279`` gate, ``1280..2559`` up (HF
    ``chunk(2)``, ``modeling_motif.py:946-951``). Pure transpose: stays in the source dtype (bf16), exact."""
    return gate_up.transpose(-1, -2).contiguous()


def experts_down(down: torch.Tensor, output_scale: float = 0.5) -> torch.Tensor:
    """``down_proj [n, 4096, 1280]`` -> ``[n, 1280, 4096] * 0.5`` (output scale folded; exact in bf16)."""
    return (down.transpose(-1, -2) * output_scale).contiguous()


def ep_layout(t: torch.Tensor, cfg: MotifTTConfig) -> torch.Tensor:
    """All-expert tensor ``[E, ...]`` -> ``[dp, E/dp, ...]`` so that ``as_tensor(dp_dim=0, tp_dim=1)`` gives chip
    (dp, tp) the slab ``[1, 12, ...]`` of experts ``[12k, 12k + 12)``, ``k = dp * 8 + tp`` (index ``[dp, 12 tp + j]`` =
    expert ``96 dp + 12 tp + j``)."""
    if t.shape[0] != cfg.num_experts:
        raise ValueError(f"expected {cfg.num_experts} experts, got {t.shape[0]}")
    return t.reshape(cfg.dp, cfg.num_experts // cfg.dp, *t.shape[1:])


def local_expert_ids(cfg: MotifTTConfig) -> torch.Tensor:
    """``[dp, 96, 1, 1]`` fp32 global expert ids in EP layout (chip gets ``[1, 12, 1, 1]``) for the decode mask
    ``eq(gathered_idx, local_expert_ids)`` (GPT-OSS gather pattern; fp32 keeps ids exact)."""
    return ep_layout(torch.arange(cfg.num_experts, dtype=torch.float32).reshape(-1, 1, 1), cfg)


def expert_polynorm_tensors(weight: torch.Tensor, bias: torch.Tensor, cfg: MotifTTConfig) -> Dict[str, torch.Tensor]:
    """Per-expert PolyNorm constants in EP layout, each ``[dp, 96, 1, 1]`` fp32 (chip gets ``[1, 12, 1, 1]``):
    ``c0, c1, c2`` = sigmoid(w) of the x^3 / x^2 / x terms and ``b`` = clamp(bias, +-0.5) (routed only)."""
    c, b = polynorm_coefficients(weight, bias, bias_clamp=cfg.polynorm_bias_clamp)  # [E, 3], [E, 1]
    out = {f"c{k}": ep_layout(c[:, k].reshape(-1, 1, 1).contiguous(), cfg) for k in range(3)}
    out["b"] = ep_layout(b.reshape(-1, 1, 1).contiguous(), cfg)
    return out


# ---- router (design §2.3.7) -------------------------------------------------------------------------------------
def router_weights(
    gate_w: torch.Tensor, expert_bias: torch.Tensor, *, pad_to: Optional[int] = None, pad_bias: float = -1e9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(W_router^T [4096, N] fp32, expert_bias [N] fp32)``. bf16 weights on device with fp32 output (HiFi4 +
    fp32 dest acc); bias added in fp32 and used for *selection only*. ``pad_to=512`` appends zero columns whose
    bias is ``pad_bias`` so they are never selected (topk width 384 -> 512)."""
    w = _f32(gate_w).t().contiguous()
    b = _f32(expert_bias).reshape(-1)
    if pad_to is not None and pad_to > w.shape[1]:
        extra = pad_to - w.shape[1]
        w = torch.cat([w, torch.zeros(w.shape[0], extra, dtype=w.dtype)], dim=1)
        b = torch.cat([b, torch.full((extra,), float(pad_bias), dtype=b.dtype)])
    return w, b


# ---- embedding, LM head, norms (design §2.3.2, §2.3.8) -------------------------------------------------------------
def lm_head_for_chip(lm_head: torch.Tensor, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """Vocab block ``tp``: ``lm_head[27520 tp : +27520]^T`` -> ``[4096, 27520]`` (host assembles ``[32, 220160]`` with
    ``ConcatMesh2dToTensor``; vocab is replicated over DP rows)."""
    n = cfg.vocab_per_chip
    return _f32(lm_head)[n * tp : n * tp + n].t().contiguous()


def norm_weight(gamma: torch.Tensor, tile: int = 32) -> torch.Tensor:
    """RMSNorm gamma ``[dim]`` -> ``[1, 1, dim / 32, 32]`` (upload ROW_MAJOR bf16; the ``ttnn.rms_norm`` weight layout
    used by ``models/common/rmsnorm.py``)."""
    d = gamma.shape[-1]
    return _f32(gamma).reshape(1, 1, d // tile, tile)


# ---- MTP input projection (features design §3.6.2; tt/mtp.py) --------------------------------------------------------
def mtp_input_proj_rows(cfg: MotifTTConfig, tp: int, in_features: Optional[int] = None) -> List[int]:
    """Input columns of ``cat[hn, e]`` (``in_features`` = 2 x 4096) that chip ``tp`` multiplies: the "interleaved" K
    split, ``[k tp, k tp + k)`` of the hidden half and the same range of the embedding half, ``k = 4096 / tp_size`` =
    512. The chip's input is then ``cat[partition(hn), partition(e)]`` ``[T, 1024]``: two tile-aligned per-chip slices
    and one concat, never the ``[T, 8192]`` concat (whose temporary is 0.5 GB per chip at T = 32768)."""
    full = 2 * cfg.hidden_size if in_features is None else int(in_features)
    half = full // 2
    if full % 2 or half % cfg.tp:
        raise ValueError(f"input_proj in_features {full} does not split into 2 x {cfg.tp} slices")
    k = half // cfg.tp
    lo = k * int(tp)
    return list(range(lo, lo + k)) + list(range(half + lo, half + lo + k))


def mtp_input_proj_for_chip(input_proj: torch.Tensor, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """Chip ``tp``'s block of the MTP ``input_proj`` (HF ``[4096, 8192]``, ``h = cat[hn, e] @ W^T``): the rows of
    ``W^T`` for :func:`mtp_input_proj_rows` -> ``[1024, 4096]`` (hidden-half rows first, fp32). Upload
    ``stack_tp(..., dim=0)`` with ``tp_dim=0``; the per-chip partial products are closed with ``all_reduce(tp)``
    (exact algebra: the K sum is split into 8 disjoint parts)."""
    D, K = input_proj.shape
    if D != cfg.hidden_size or K != 2 * cfg.hidden_size:
        want = (cfg.hidden_size, 2 * cfg.hidden_size)
        raise ValueError(f"input_proj has shape {tuple(input_proj.shape)}, expected {want}")
    rows = torch.tensor(mtp_input_proj_rows(cfg, tp, K), dtype=torch.long)
    return _f32(input_proj).t()[rows].contiguous()


# ============================================================================================================
# (3) Mesh upload with cache
# ============================================================================================================
def mapping_tag(dp_dim: Optional[int] = None, tp_dim: Optional[int] = None) -> str:
    """Cache-name suffix for a mapping: ``rep``, ``tp1``, ``dp0``, ``dp0tp1`` (role based, orientation free)."""
    if dp_dim is None and tp_dim is None:
        return "rep"
    return (f"dp{dp_dim}" if dp_dim is not None else "") + (f"tp{tp_dim}" if tp_dim is not None else "")


def mesh_mapper(mesh_device, axes: MeshAxes, *, dp_dim: Optional[int] = None, tp_dim: Optional[int] = None):
    """Mapper for the role-based placement:

    * replicate (both ``None``): ``ttnn.ReplicateTensorToMesh`` -- its wrapper type makes ``ttnn.as_tensor`` cache
      **one** unsharded copy instead of 32;
    * ``tp_dim`` only: shard that tensor dim over the TP axis, replicate over DP ("shard over cols");
    * ``dp_dim`` only: shard over the DP axis, replicate over TP ("shard over rows");
    * both: 2D shard; with :func:`ep_layout` (``dp_dim=0, tp_dim=1``) this is the EP32 expert placement.
    """
    if dp_dim is None and tp_dim is None:
        return ttnn.ReplicateTensorToMesh(mesh_device)
    dims = axes.mesh_dims(dp_dim=dp_dim, tp_dim=tp_dim)
    return ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate() if d is None else ttnn.PlacementShard(d) for d in dims],
            ttnn.MeshShape(*axes.mesh_shape),
        ),
    )


def shard_for_device(
    t: torch.Tensor, axes: MeshAxes, row: int, col: int, *, dp_dim: Optional[int] = None, tp_dim: Optional[int] = None
) -> torch.Tensor:
    """Host emulation of :func:`mesh_mapper`: the local tensor chip (row, col) receives (tests, host-side cache
    builders). Chip (row, col) has roles ``(dp, tp) = axes.roles(row, col)``."""
    dp, tp = axes.roles(row, col)
    out = t
    if dp_dim is not None:
        out = out.chunk(axes.dp_size, dim=dp_dim)[dp]
    if tp_dim is not None:
        out = out.chunk(axes.tp_size, dim=tp_dim)[tp]
    return out


def tensorbin_path(cache_prefix: Union[str, os.PathLike], dtype, layout) -> Path:
    """The file ``ttnn.as_tensor(cache_file_name=prefix)`` writes: ``<prefix>_dtype_<D>_layout_<L>.tensorbin``."""
    return Path(f"{cache_prefix}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")


def cache_prefix(cfg: MotifTTConfig, name: str, layer: Optional[int], dp_dim=None, tp_dim=None) -> Path:
    """``<cache_dir>/{L<nn>|global}/<name>__<mapping>``."""
    return cfg.cache_file(f"{name}__{mapping_tag(dp_dim, tp_dim)}", layer)


def _materialize(src: TorchSource) -> torch.Tensor:
    t = src() if callable(src) else src
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"weight source produced {type(t)}, expected torch.Tensor")
    return t.contiguous()


def as_tensor(
    src: TorchSource,
    *,
    mesh_device,
    cfg: MotifTTConfig,
    dtype,
    layout=ttnn.TILE_LAYOUT,
    memory_config=None,
    dp_dim: Optional[int] = None,
    tp_dim: Optional[int] = None,
    cache_name: Optional[str] = None,
    layer: Optional[int] = None,
):
    """Upload a (lazily built) host tensor with a role-based mesh mapping, through the TT weight cache.

    Args:
        src: the full host tensor, or a zero-arg callable returning it (only called on a cache miss).
        dtype / layout / memory_config: device format (memory_config defaults to DRAM interleaved; a cached file
            reloads in the memory config it was built with -- re-shard with ``ttnn.to_memory_config`` if needed).
        dp_dim / tp_dim: tensor dims sharded over the DP / TP axes (see :func:`mesh_mapper`).
        cache_name: e.g. ``"attn.wq_b"``; ``None`` disables caching (random weights). The cache key also contains
            the layer, the mapping tag, dtype and layout, and (via ``cfg.cache_dir``) checkpoint revision,
            ``CACHE_FORMAT_VERSION``, dtype policy and mesh shape.
        layer: decoder layer index, or ``None`` for model-global tensors.
    """
    mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    mapper = mesh_mapper(mesh_device, cfg.axes, dp_dim=dp_dim, tp_dim=tp_dim)
    if cache_name is None:
        return ttnn.from_torch(
            _materialize(src), dtype=dtype, layout=layout, device=mesh_device, memory_config=mc, mesh_mapper=mapper
        )
    prefix = cache_prefix(cfg, cache_name, layer, dp_dim, tp_dim)
    path = tensorbin_path(prefix, dtype, layout)
    if path.is_file():
        try:
            return ttnn.load_tensor(path, device=mesh_device)
        except RuntimeError as e:  # corrupt / incompatible file: rebuild below (ttnn.as_tensor overwrites it)
            print(f"[motif3.weights] cache load failed for {path}: {e}; rebuilding")
    return ttnn.as_tensor(
        _materialize(src),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=mc,
        cache_file_name=str(prefix),
        mesh_mapper=mapper,
    )


def is_cached(
    cfg: MotifTTConfig, cache_name: str, layer: Optional[int], dtype, layout=ttnn.TILE_LAYOUT, dp_dim=None, tp_dim=None
) -> bool:
    return tensorbin_path(cache_prefix(cfg, cache_name, layer, dp_dim, tp_dim), dtype, layout).is_file()


def layer_cache_marker(cfg: MotifTTConfig, layer: Optional[int]) -> Path:
    """Completion marker written by the converter after every tensor of a layer is cached (resumable conversion)."""
    sub = "global" if layer is None else f"L{int(layer):02d}"
    return cfg.cache_dir / sub / ".complete"


def mark_layer_cached(cfg: MotifTTConfig, layer: Optional[int], names: Iterable[str] = ()) -> Path:
    p = layer_cache_marker(cfg, layer)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"layer": layer, "tensors": sorted(names), "version": cfg.cache_version_tag}, indent=1))
    return p


__all__ = [
    "DictWeightSource",
    "HFWeightLoader",
    "MissingWeightError",
    "absorb_weights_for_chip",
    "absorb_weights_per_head_for_chip",
    "absorbed_q_weights_per_head_for_chip",
    "as_tensor",
    "cache_prefix",
    "ep_layout",
    "expert_polynorm_tensors",
    "experts_down",
    "experts_gate_up",
    "experts_of_chip",
    "fold_kv_norm",
    "hf_name",
    "is_cached",
    "kv_up_for_chip",
    "latent_projection",
    "latent_projection_for_chip",
    "layer_cache_marker",
    "lm_head_for_chip",
    "local_expert_ids",
    "mapping_tag",
    "mark_layer_cached",
    "mesh_mapper",
    "mhc_fused_projection",
    "mhc_projection_blocks",
    "mhc_projection_from_source",
    "mhc_scalars",
    "MTP_ATTN_TENSORS",
    "MTP_LAYERS_PREFIX",
    "MTP_NORMS",
    "mlp_down_for_chip",
    "mlp_gate_up_for_chip",
    "mtp_input_proj_for_chip",
    "mtp_input_proj_rows",
    "mtp_layer_available",
    "mtp_name",
    "mtp_names_in",
    "mtp_tensor_names",
    "norm_weight",
    "polynorm_coefficients",
    "prefill_kv_expansion_for_chip",
    "router_weights",
    "shard_for_device",
    "signal_order_for_chip",
    "split_wkv_a",
    "split_wkv_b",
    "split_wq_b",
    "stack_tp",
    "tensorbin_path",
    "unabsorb_weights_for_chip",
    "wo_for_chip",
    "wq_b_for_chip",
    "wq_b_gate_for_chip",
]
