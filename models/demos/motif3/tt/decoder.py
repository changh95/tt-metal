# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 decoder layer on the BH Galaxy: ``MotifDecoderLayer`` (design §2.3.3-2.3.7, §3.2-3.3; WAVE_A_REVIEW GEN-5).

One Motif-3 block on the 4-stream residual (HF ``_forward_with_mhc``; reference ``modules.DecoderLayer``)::

    x_red, c1 = mhc_attn.pre(X)                 # coefficients + stream reduce            [1, 4, T, D] -> [1, 1, T, D]
    a  = input_layernorm(x_red)                 # RMSNorm 1e-5
    o  = attention(a)                           # GDLA, closed with AR(tp)                 [1, 1, T, D]
    X1 = mhc_attn.post(X, o, c1)                # H @ X + h_post (x) o                     [1, 4, T, D]
    y_red, c2 = mhc_ffn.pre(X1)
    f  = post_attention_layernorm(y_red)
    u  = dense MLP(f)                           (layers 0-1, closed with AR(tp))
       | MoE(f, add_partial=shared(f))          (layers 2-52; one AR(tp) closes routed + shared expert)
    X2 = mhc_ffn.post(X1, u, c2)

Every sub-module is the wave-B1 module (``tt/mhc.py``, ``tt/attention.py``, ``tt/mlp.py``, ``tt/moe.py``); this file only
owns the two RMSNorms (gamma ``[1, 1, 128, 32]`` bf16 ROW_MAJOR, cache names ``input_layernorm.weight`` /
``post_attention_layernorm.weight``) and the wiring. Module defaults come from the config (``cfg.mhc_sinkhorn``,
``cfg.router_logits``; README §10 rule 1); keyword dicts pass module-specific overrides through (A/B experiments).

Tensor contract (README CONVENTIONS §3):

* decode: ``X [1, 4, 8, 4096]`` bf16 TILE DRAM (the 8 lanes of the chip's DP row, replicated over its 8 TP chips) ->
  ``X' [1, 4, 8, 4096]``; per-step inputs shared by all layers (built ONCE per step by the model): ``rot`` (
  ``MotifAttention.decode_rope_tables``), ``active`` (``MotifAttention.active_mask_from_cur_pos``), ``cur_pos [8]``
  int32, ``page_table [8, W]`` int32; this layer's paged latent cache. Trace-safe: fixed shapes, no host round trip.
* prefill (one user, S = bucket, replicated on all 32 chips): ``X [1, 4, S, 4096]`` -> ``X' [1, 4, S, 4096]``;
  ``page_table [1, >= S / block]`` int32 + this layer's cache (``None`` / ``None`` skips the cache fill: layer tests).
  Resumed / chunked prefill (README §15): ``chunk=`` = the chunk's ``attention.PrefillChunkInputs`` instead of
  ``page_table`` (rows at positions ``[chunk.start, chunk.start + S)``; sp1 chunks read the cached prefix).
* decode with KV-R / the speculative split (README §16): ``kv_write=`` = the step's ``kv_write.DecodeKVWrite``, shared
  by every layer (``cur_pos`` / ``page_table`` are then its tensors); ``None`` = draft 1.

Inputs are never deallocated (README §10 rule 3); the returned streams are a new tensor. The decode norms run
width-sharded on 8 x 4 cores (``cfg.decode_norm_configs()``: ~6 us + reshards instead of 65 us on one core,
README §12 gotchas); prefill norms are interleaved (S rows spread over the grid).

L1 between calls (README §10 rule 7): mHC decode keeps ``w_pre`` / ``w_post`` (~100 KB per chip) in L1 between
``pre`` and ``post``, i.e. across the attention and the FFN sublayer. ``mhc_kwargs={"mix_l1": False}`` moves them to
DRAM (+20 us per site) if a sublayer's static circular buffers ever clash with them.

Import rule (design §2.1): ttnn and the motif3 ``tt/`` modules only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import ttnn

from . import weights as W
from .attention import MotifAttention
from .ccl import MotifCCL
from .mhc import MHCSite
from .mlp import MotifDenseMLP, MotifSharedExpert
from .model_config import MotifTTConfig
from .moe import MotifMoE
from .rope import MotifRope

INPUT_NORM = "input_layernorm.weight"
POST_ATTN_NORM = "post_attention_layernorm.weight"


def _free(*ts) -> None:
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


def free_tensors(obj, _seen=None) -> int:
    """Deallocate every ``ttnn.Tensor`` held by ``obj``'s attributes (one level, plus lists / tuples / dicts of
    tensors); returns the count. For modules without a ``deallocate`` (attention, embedding). Idempotent."""
    n = 0
    vals = list(vars(obj).values()) if hasattr(obj, "__dict__") else []
    stack = vals
    while stack:
        v = stack.pop()
        if isinstance(v, ttnn.Tensor):
            try:
                if v.is_allocated():
                    ttnn.deallocate(v)
                    n += 1
            except Exception:
                pass
        elif isinstance(v, (list, tuple)):
            stack.extend(v)
        elif isinstance(v, dict):
            stack.extend(v.values())
    return n


class MotifDecoderLayer:
    """Decoder layer ``layer_idx`` (dense MLP for layers 0-1, MoE + shared expert for 2-52; global / SWA attention per
    ``cfg.layer(l)``). See the module docstring.

    Args:
        mesh_device: the opened (4, 8) (or (8, 4)) mesh (with an L1_SMALL region: ``model_config.device_params()``).
        cfg: :class:`MotifTTConfig` (``cfg.layer(layer_idx)`` must exist: ``layer_idx < cfg.num_layers``).
        layer_idx: decoder layer index.
        source: ``HFWeightLoader`` / ``DictWeightSource`` (HF names; read lazily, never on a TT-cache hit).
        ccl: shared :class:`MotifCCL`; rope: shared :class:`MotifRope` (prefill tables, decode gather).
        cache: TT weight cache (README §7; ``False`` for random weights: nothing written).
        sinkhorn / router_logits: module defaults override (``None`` = ``cfg.mhc_sinkhorn`` / ``cfg.router_logits``).
        mhc_kwargs / attention_kwargs / mlp_kwargs / moe_kwargs / shared_kwargs: extra constructor kwargs of the
            sub-modules (A/B experiments; the defaults are the measured module defaults).
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        layer_idx: int,
        *,
        source,
        ccl: MotifCCL,
        rope: MotifRope,
        cache: bool = True,
        sinkhorn: Optional[str] = None,
        router_logits: Optional[str] = None,
        mhc_kwargs: Optional[Dict[str, Any]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        mlp_kwargs: Optional[Dict[str, Any]] = None,
        moe_kwargs: Optional[Dict[str, Any]] = None,
        shared_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.layer_idx = l = int(layer_idx)
        self.spec = cfg.layer(l)
        self.ccl = ccl
        self.rope = rope
        self.is_moe = bool(self.spec.is_moe)
        sinkhorn = sinkhorn if sinkhorn is not None else cfg.mhc_sinkhorn
        router_logits = router_logits if router_logits is not None else cfg.router_logits

        mk = dict(mhc_kwargs or {})
        self.mhc_attn = MHCSite(mesh_device, cfg, l, "mhc_attn", source=source, cache=cache, sinkhorn=sinkhorn, **mk)
        self.mhc_ffn = MHCSite(mesh_device, cfg, l, "mhc_ffn", source=source, cache=cache, sinkhorn=sinkhorn, **mk)
        ak = dict(require_l1_small=True)
        ak.update(attention_kwargs or {})
        self.attn = MotifAttention(mesh_device, cfg, l, source=source, ccl=ccl, rope=rope, cache=cache, **ak)
        self.mlp = self.moe = self.shared = None
        if self.is_moe:
            mo = dict(router_logits=router_logits)
            mo.update(moe_kwargs or {})
            self.moe = MotifMoE(mesh_device, cfg, l, source=source, ccl=ccl, cache=cache, **mo)
            self.shared = MotifSharedExpert(mesh_device, cfg, l, source=source, ccl=ccl, cache=cache,
                                            **dict(shared_kwargs or {}))
        else:
            self.mlp = MotifDenseMLP(mesh_device, cfg, l, source=source, ccl=ccl, cache=cache, **dict(mlp_kwargs or {}))

        def norm(name):
            return W.as_tensor(
                lambda: W.norm_weight(source.get(W.hf_name(l, name))),
                mesh_device=mesh_device,
                cfg=cfg,
                dtype=cfg.dtypes.norms,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                cache_name=(name if cache else None),
                layer=l,
            )

        self.input_norm = norm(INPUT_NORM)  # [1, 1, 128, 32] bf16 ROW_MAJOR, replicated
        self.post_attn_norm = norm(POST_ATTN_NORM)
        self.eps = float(cfg.rms_norm_eps)
        self.ckc_norm = cfg.compute_config("norm")
        self.norm_mc, self.norm_pc = cfg.decode_norm_configs()  # width-sharded 8 x 4 (README §5)

    # ------------------------------------------------------------------------------------------------------------
    # norms
    # ------------------------------------------------------------------------------------------------------------
    def _norm_decode(self, x, gamma):
        """``[1, 1, 8, 4096]`` DRAM -> width-sharded rms_norm -> DRAM (lm_head's measured pattern)."""
        xs = ttnn.to_memory_config(x, self.norm_mc)
        ys = ttnn.rms_norm(
            xs,
            epsilon=self.eps,
            weight=gamma,
            program_config=self.norm_pc,
            memory_config=self.norm_mc,
            compute_kernel_config=self.ckc_norm,
        )
        _free(xs)
        y = ttnn.to_memory_config(ys, ttnn.DRAM_MEMORY_CONFIG)
        _free(ys)
        return y

    def _norm_prefill(self, x, gamma):
        return ttnn.rms_norm(
            x, epsilon=self.eps, weight=gamma, compute_kernel_config=self.ckc_norm, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    # ------------------------------------------------------------------------------------------------------------
    # forwards
    # ------------------------------------------------------------------------------------------------------------
    def forward_decode(
        self, X, *, rot, cur_pos, page_table, kv_cache, active, taps: Optional[dict] = None, kv_write=None,
        moe_lane_mask=None,
    ):
        """One decode step of this layer for the 8 lanes of each DP row (trace-safe; see the module docstring).
        ``taps`` (eager debugging only) receives the intermediates ``x_red``, ``attn_in``, ``attn_out``, ``x_mid``,
        ``ffn_in``, ``ffn_out`` (not freed). ``kv_write``: the step's ``tt.kv_write.DecodeKVWrite`` (KV-R / the
        speculative split, README §16), passed to the attention unchanged; ``None`` = the draft-1 8-lane update (bitwise
        draft 1). With it, ``cur_pos`` / ``page_table`` must be ``kv_write.cur_pos`` / ``kv_write.page_table``.
        ``moe_lane_mask``: the step's MoE live-row mask (B1 sparse decode experts; ``MotifMoE.decode_lane_mask``, built
        once per step by the model; not consumed), handed to the MoE; dense layers and the dense MoE ignore it."""
        x_red, c1 = self.mhc_attn.pre(X)
        a = self._norm_decode(x_red, self.input_norm)
        kw = {} if kv_write is None else {"kv_write": kv_write}
        o = self.attn.forward_decode(a, rot=rot, cur_pos=cur_pos, page_table=page_table, kv_cache=kv_cache,
                                     active=active, **kw)
        X1 = self.mhc_attn.post(X, o, c1)  # frees c1
        y_red, c2 = self.mhc_ffn.pre(X1)
        f = self._norm_decode(y_red, self.post_attn_norm)
        if self.is_moe:
            part = self.shared.forward_decode(f, all_reduce=False)  # this chip's TP partial (bf16)
            u = self.moe.forward_decode(f, add_partial=part, lane_mask=moe_lane_mask)  # routed + shared, one AR(tp)
            _free(part)
        else:
            u = self.mlp.forward_decode(f)
        X2 = self.mhc_ffn.post(X1, u, c2)  # frees c2
        if taps is not None:
            taps.update(x_red=x_red, attn_in=a, attn_out=o, x_mid=X1, ffn_red=y_red, ffn_in=f, ffn_out=u)
        else:
            _free(x_red, a, o, y_red, f, u, X1)
        return X2

    def forward_prefill(self, X, *, page_table=None, kv_cache=None, taps: Optional[dict] = None, chunk=None):
        """Prefill of one user (eager): ``X [1, 4, S, 4096]`` -> ``[1, 4, S, 4096]``; fills this layer's cache for
        positions ``0 .. S-1`` through the first ``cfg.prefill_page_table_entries(S)`` entries of ``page_table``
        (``kv_cache=None`` skips the fill). ``taps`` as in :meth:`forward_decode`.

        ``chunk``: one chunk of a resumed / chunked prefill (features design §3.7.1; README §15): the chunk's
        ``tt.attention.PrefillChunkInputs`` (fill table, and for an sp1 chunk the SDPA table, start, offset-RoPE rows
        and SWA tail bounds), built once per chunk and shared by every layer, in place of ``page_table``. ``X`` holds
        the chunk's bucket rows at positions ``[chunk.start, chunk.start + S)``. Only the attention reads it (every
        other sub-module is row-local); an sp0 chunk is the draft-1 call with the chunk's fill table, an sp1 chunk
        needs ``kv_cache`` (it reads the cached prefix)."""
        if chunk is not None and page_table is not None:
            raise ValueError("forward_prefill: pass chunk= or page_table=, not both (the chunk carries its fill table)")
        x_red, c1 = self.mhc_attn.pre(X)
        a = self._norm_prefill(x_red, self.input_norm)
        if chunk is not None:
            o = self.attn.forward_prefill(a, chunk=chunk, kv_cache=kv_cache)
        else:
            o = self.attn.forward_prefill(a, page_table=page_table if kv_cache is not None else None,
                                          kv_cache=kv_cache)
        X1 = self.mhc_attn.post(X, o, c1)
        y_red, c2 = self.mhc_ffn.pre(X1)
        f = self._norm_prefill(y_red, self.post_attn_norm)
        if self.is_moe:
            sl = self.moe.dp_slice(f)  # this DP row's S / 4 rows: the rows the shared partial must cover
            part = self.shared.forward_prefill(sl, all_reduce=False)
            _free(sl)
            u = self.moe.forward_prefill(f, add_partial=part)
            _free(part)
        else:
            u = self.mlp.forward_prefill(f)
        X2 = self.mhc_ffn.post(X1, u, c2)
        if taps is not None:
            taps.update(x_red=x_red, attn_in=a, attn_out=o, x_mid=X1, ffn_red=y_red, ffn_in=f, ffn_out=u)
        else:
            _free(x_red, a, o, y_red, f, u, X1)
        return X2

    # ------------------------------------------------------------------------------------------------------------
    def deallocate(self) -> None:
        """Free this layer's device weights (the shared ccl / rope stay)."""
        self.mhc_attn.release()
        self.mhc_ffn.release()
        free_tensors(self.attn)
        if self.moe is not None:
            self.moe.deallocate()
        if self.shared is not None:
            self.shared.deallocate()
        if self.mlp is not None:
            self.mlp.deallocate()
        _free(self.input_norm, self.post_attn_norm)
        self.input_norm = self.post_attn_norm = None


__all__ = ["INPUT_NORM", "POST_ATTN_NORM", "MotifDecoderLayer", "free_tensors"]
