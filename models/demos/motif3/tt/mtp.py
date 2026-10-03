# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 MTP layer (``model.mtp_layers.0``) on the BH Galaxy: the drafter of MTP self-speculative decoding (features
design ``docs/features/FEATURES_DESIGN.md`` §3.6, §3.8, D9, D10; README CONVENTIONS §17; work package 3).

Math (reference ``modules.MotifMTP``; fork ``motif_mtp.py:160-170``, training ``model.py:865-876``)::

    e   = embed_norm(embed(t_{p+1}))                  # the main model's embedding table; RMSNorm 1e-5
    h   = input_proj(cat[hn_p, e])                    # [8192] -> [4096]; hn_p = the main model's post-final-norm hidden
    h  += GDLA(input_layernorm(h))                    # SWA ("all" mode): window 129, plain RoPE, scale 192^-0.5
    h  += MLP(post_attention_layernorm(h))            # dense PolyNorm, intermediate 12288, output scale 0.5
    out = final_layernorm(h);  logits = lm_head(out)  # the main model's LM head

Concat order ``[hidden, embed]`` (DeepSeek-V3 uses the reverse), no ``hnorm``, one plain pre-norm residual block (no
mHC). Position ``p`` consumes ``(hn_p, t_{p+1})``, writes its own latent at ``p`` into the MTP cache and predicts
``t_{p+2}``.

TT dataflow per chip (TP index ``tp``; ``k = 4096 / 8 = 512``)::

    e_n  = rms_norm(embed_rows(t), gamma_embed)                     [1, 1, T, 4096]
    x_in = concat(partition_tp(hn), partition_tp(e_n))              [1, 1, T, 1024]  hidden cols [k tp, +k) | embed cols
    h    = ar_tp(x_in @ W_in[tp])                                   [1, 1, T, 4096]  W_in[tp] = rows of input_proj^T
    a    = rms_norm(h, gamma_input)
    decode (every lane of every decode step; features design §3.8.1, the "T32-spec" trace):
         h1 = h + attn.forward_decode(a)  ->  h2 = h1 + mlp(rms_norm(h1, gamma_post))
         out = rms_norm(h2, gamma_final)
         m  = head.argmax_decode(head.decode_logits(out))           [1, 1, 1, 32] uint32, lane order
    prefill, KV-only (serving; D9): attn.fill_kv(a, chunk=, kv_cache=)       no SDPA, no MLP, no head
    prefill, full (tests, acceptance estimates): attn.forward_prefill(a) -> ... -> out [1, 1, S, 4096]

* ``input_proj`` is K-sharded over TP ("interleaved" split, ``weights.mtp_input_proj_rows``): chip ``tp`` holds the
  1024 rows of ``W^T`` for hidden columns ``[512 tp, +512)`` and the same embedding columns, so its input is two
  tile-aligned per-chip slices and one concat, and ``ar_tp`` closes the 8 partial products. (The design's contiguous
  split partitions a ``[T, 8192]`` concat, a 0.5 GB per-chip temporary at T = 32768; a replicated 67 MB weight would
  stream 8x the bytes per step.) bf16 weights (``cfg.dtypes.attention``), HiFi4 + fp32 acc (``attn_heads`` role, the
  ``wo`` analog), bf16 partials.
* Attention: ``MotifAttention(cfg.mtp_layer_idx, spec=cfg.mtp_layer_spec(), weight_prefix=model.mtp_layers.0.
  self_attn)`` (work package 2a) -- the decoder layers' module and transforms, its own latent cache (the MTP cache of
  the pool, same block ids as the 53 main caches). MLP: ``PolyNormMLP(kind="dense", prefix=model.mtp_layers.0.mlp)``
  (bfp8 weights, output scale ``polynorm_output_scale(cfg, 53)`` = 0.5 folded into ``W_down``). Norms: bf16 gammas;
  decode width-sharded (``cfg.decode_norm_configs()``), prefill interleaved. Embedding and LM head are the main
  model's modules (``embed=``, ``head=``; the MTP layer owns no table).
* Decode inputs, all from the speculative decode step (no host round trip): ``hn`` = ``head.stream_mean_norm(X)``
  ``[1, 1, 8, 4096]`` (this DP row's lanes; ``head.decode_logits(hn)`` does not consume it), ``tokens`` = the main
  argmax ``head.argmax_decode(logits)`` ``[1, 1, 1, 32]`` uint32 (lane order, every chip), and the step's ``rot``
  (must hold the ``"plain"`` tables), ``cur_pos``, ``page_table``, ``active`` (and ``kv_write``) of the main layers:
  the MTP layer writes its latent at the same lanes and positions. Inactive lanes produce don't-care tokens.
* Prefill inputs: ``hn`` = ``head.stream_mean_norm(X)`` of the chunk ``[1, 1, C, 4096]`` (replicated), next tokens
  ``[1, C]`` uint32 (``embed.rows_tokens_device(mtp_next_tokens(...), C)``: ``t_{p+1}`` per row; only the last known
  row -- the request's last prefill row or a vLLM chunk end -- takes the host argmax of its logits, features design
  §3.6.3, §3.7.1; :func:`mtp_next_tokens` refuses a stand-in where the token is known), and the chunk's
  ``PrefillChunkInputs`` (``chunk=``: its fill table, ``-1`` = skip, and RoPE rows; checked against ``C`` by the
  attention) or the bare ``fill_pt`` / ``rot`` (``None`` = positions ``0 .. C-1``, else ``rope.chunk_rope_tables``).
  Rows are processed in chunks of ``cfg.prefill_row_chunk`` (8192) up to the attention fill, so a 32K single-shot fill
  keeps its temporaries bounded.

TT weight cache (part ``L53`` = ``cfg.mtp_layer_idx``; README §7, §17): the attention's ``attn.v2.*`` (10 files), the
MLP's ``mlp.gate_up`` / ``mlp.down`` / ``mlp.polynorm.*``, and this module's ``mtp.v1.input_proj`` (``__tp0``) plus
``mtp.v1.<norm>.weight`` for the 4 norms (``__rep``, ROW_MAJOR). Bump :data:`MTP_CACHE_VERSION` when a transform here
changes. ``scripts/convert_weights.py --mtp`` builds the part (kind ``mtp``) with this constructor and ``cache=True``:
21 files, 420,183,552 bytes; about 0.06 GB of device DRAM per chip.

Measured on this Galaxy (``tests/unit/test_mtp.py``: real weights from part L53, C2 golden inputs, PCC vs the fp32
reference ``MotifMTP``; gate G14):

* KV-only fill at C = 128 (sp0, a shared block -1), 1024 (sp1 rows at 1024 .. 2047, a padding block -1) and 4096:
  cache ``n * gamma`` 0.999967, ``k_pe`` 0.999969-0.999970; skipped blocks untouched; 4 row chunks == one pass
  bitwise. Eager wall time 1.8 / 1.9 / 6.7 ms per call at C = 128 / 1024 / 8192.
* Decode, 32 lanes at positions 129 .. 510 over KV-only-filled histories whose slots p and p + 1 were then poisoned:
  GDLA 0.999952 (worst lane 0.999828), MLP on its own input 0.999949 (0.999899), output 0.999895 (0.999794), argmax ==
  the fp32 reference 32 / 32. The decode's KV write rewrites each lane's p on its own DP row (latent ``n * gamma``
  0.999960, ``k_pe`` 0.999964 vs the reference) and nothing else on any of the 32 chips (``kv_write`` host model).
  Traced 997 us per step, of which the shared head (``ag_dp_rows`` + project + argmax) 391 us; a trace replay at p + 1
  with new inputs == an eager step from the same cache state, bitwise (tokens, hidden, all 32 cache copies).
* Lane independence (``test_mtp_decode_kv_writes``): odd lanes inactive, every lane moved to the next DP row,
  ``kv_write`` mode ``all``, and an ``all_split`` packed verify with the partners on other DP rows at p + 1 (GDLA
  0.999926, output 0.999894, argmax 16 / 16 vs the reference) all give bitwise the row-mode rows, and every chip's
  cache matches the host model of the mode; a partner row == the same row computed by an ordinary step, bitwise.
* Full block over the 2965 C2 rows (prefill form): output 0.999825 (per-row min 0.99931); argmax == the fp32 reference
  on 98.95 % of the rows and on all 2704 rows without a bf16 tie. Acceptance (teacher-forced on the golden hidden
  states, K = 1): 0.7558 all rows, 0.8340 on-policy, vs 0.7599 / 0.8365 for the CPU bf16 reference.

Import rule (design §2.1): ttnn, torch and the motif3 ``tt/`` modules only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

import ttnn

from . import weights as W
from .attention import MotifAttention, PrefillChunkInputs
from .ccl import MotifCCL
from .mlp import PolyNormMLP
from .model_config import TILE, MotifTTConfig, mcast1d_matmul_pc
from .rope import MotifRope

# Cache-name prefix of every tensor this module itself writes (the attention and the MLP keep their own names).
MTP_CACHE_VERSION = 1
_CACHE = f"mtp.v{MTP_CACHE_VERSION}"
INPUT_PROJ = "input_proj.weight"
# Decode program config of the input projection [1, 1, 8 | 32, 1024] @ [1024, 4096] (M = one tile row): 1D multicast
# on 8 x 4 cores, 4 of the 128 output tiles each, in0_block_w 8 (the dense MLP's down-projection pattern).
INPUT_PROJ_DECODE_GRID = (8, 4)
INPUT_PROJ_DECODE_IN0_BLOCK_W = 8
TAP_NAMES = ("e", "e_norm", "x_in", "h", "attn_in", "attn_out", "h_mid", "ffn_in", "ffn_out", "h_out", "out", "logits")


def _free(*ts) -> None:
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


def mtp_next_tokens(tokens, start: int, end: int, next_after_end: Optional[int] = None) -> torch.Tensor:
    """Host: the MTP layer's input tokens ``t_{p+1}`` for the prefill rows ``p in [start, end)`` -> int32
    ``[end - start]`` = ``tokens[start + 1 : end + 1]``, the last one supplied by ``next_after_end`` when the token at
    position ``end`` is not known (features design §3.6.3, §3.7.1).

    ``tokens``: the request's KNOWN tokens, positions ``0 .. n-1`` (``PrefillRequest.tokens``: cached prefix, earlier
    chunks, new tokens and, for a resumed request, the generated ones; no padding); ``0 <= start < end <= n``.

    * ``end < n``: the chunk ends inside the known tokens (a generator-internal chunk end), so row ``end - 1`` takes
      the known ``tokens[end]``. ``next_after_end`` may be omitted; when given it must equal ``tokens[end]``
      (``ValueError`` otherwise): a stand-in there would store a wrong MTP latent at an aligned chunk end, i.e. in a
      full block that vLLM may cache and share.
    * ``end == n``: row ``end - 1`` is the last known position (the request's last prefill row, or a vLLM chunk end),
      so ``next_after_end`` is required: the host argmax of the logits at ``end - 1`` (the token a greedy request
      commits; lowest-index tie rule, ``torch.argmax``).

    Upload with ``MotifEmbedding.rows_tokens_device(ids, bucket)`` (pads to the bucket)."""
    t = torch.as_tensor(tokens).reshape(-1)
    s, e, n = int(start), int(end), int(t.numel())
    if not 0 <= s < e <= n:
        raise ValueError(f"rows [{s}, {e}) outside the {n} known tokens")
    if e < n:
        nxt = int(t[e])
        if next_after_end is not None and int(next_after_end) != nxt:
            raise ValueError(
                f"the token at position {e} is known ({nxt}): row {e - 1} must take it, not next_after_end="
                f"{int(next_after_end)} (only the last known row takes the host-argmax stand-in; design §3.7.1)"
            )
    elif next_after_end is None:
        raise ValueError(
            f"rows [{s}, {e}) end at the last known token: pass next_after_end (the host argmax of the logits at "
            f"position {e - 1}; design §3.6.3)"
        )
    else:
        nxt = int(next_after_end)
    return torch.cat([t[s + 1 : e].to(torch.int32), torch.tensor([nxt], dtype=torch.int32)])


def input_proj_decode_pc(
    cfg: MotifTTConfig, grid=INPUT_PROJ_DECODE_GRID, in0_block_w: int = INPUT_PROJ_DECODE_IN0_BLOCK_W
):
    """1D-multicast program config of the decode input projection ``[1, 1, 32, 1024] @ [1024, 4096]``, or ``None``
    (ttnn's auto config) when ``grid`` does not fit ``cfg.compute_grid``."""
    gx, gy = int(grid[0]), int(grid[1])
    if gx > cfg.compute_grid[0] or gy > cfg.compute_grid[1]:
        return None
    k_tiles = 2 * cfg.hidden_size // cfg.tp // TILE  # 32
    return mcast1d_matmul_pc((gx, gy), cfg.hidden_size // TILE, int(in0_block_w), k_tiles, fuse_batch=True)


def mtp_cache_complete(cfg: MotifTTConfig) -> bool:
    """The converter marked the MTP part (``L53``) complete in ``cfg.cache_dir`` (``weights.layer_cache_marker``)."""
    return W.layer_cache_marker(cfg, cfg.mtp_layer_idx).is_file()


def mtp_weights_available(cfg: MotifTTConfig, source=None) -> bool:
    """MTP weights can be loaded without a download: the TT-cache part ``L53`` is complete, or ``source`` (default an
    ``HFWeightLoader`` on ``cfg.weights_dir``; index + shard sizes only, no tensor read) lists all 19
    ``model.mtp_layers.0.*`` tensors this module reads, each on disk (``weights.mtp_layer_available``: an index that
    lists only some of them is refused). Host only, never raises (``False`` on any error)."""
    try:
        if int(cfg.num_nextn_predict_layers) < 1:
            return False
        if mtp_cache_complete(cfg):
            return True
        src = source if source is not None else W.HFWeightLoader(cfg.weights_dir)
        return W.mtp_layer_available(src)
    except Exception:
        return False


class MotifMTP:
    """The MTP layer ``model.mtp_layers.0`` (module docstring).

    Args:
        mesh_device: the opened mesh (an L1_SMALL region is required, as for every decoder layer).
        cfg: :class:`MotifTTConfig` of the mesh (``cfg.num_nextn_predict_layers >= 1``).
        source: weight source with HF names ``model.mtp_layers.0.*`` (``HFWeightLoader`` / ``DictWeightSource``); read
            lazily, never on a TT-cache hit.
        ccl / rope: the model's shared :class:`MotifCCL` / :class:`MotifRope` (prefill tables, decode gathers).
        embed / head: the main model's :class:`MotifEmbedding` / :class:`MotifLMHead` (the MTP layer has no own table
            or head). ``None`` is allowed only for building the weights (the converter); every forward needs them.
        cache: TT weight cache (part ``L53``); ``False`` for random weights (nothing written).
        attention_kwargs / mlp_kwargs: extra constructor kwargs of the attention / the MLP (A/B knobs).
        input_proj_program_config: decode program config of the input projection (``"default"`` =
            :func:`input_proj_decode_pc`, ``None`` = ttnn's auto config, or a config object).
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        *,
        source,
        ccl: MotifCCL,
        rope: Optional[MotifRope] = None,
        embed=None,
        head=None,
        cache: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        mlp_kwargs: Optional[Dict[str, Any]] = None,
        input_proj_program_config: Any = "default",
    ):
        if int(cfg.num_nextn_predict_layers) < 1:
            raise ValueError("this checkpoint has no MTP layer (num_nextn_predict_layers = 0)")
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.ccl = ccl
        self.rope = rope
        self.embed = embed
        self.head = head
        self.layer_idx = L = int(cfg.mtp_layer_idx)  # 53: the reference layer index and the TT-cache part L53
        self.spec = cfg.mtp_layer_spec()
        self.prefix = W.mtp_name()  # model.mtp_layers.0
        self.hidden = int(cfg.hidden_size)
        self.lanes = int(cfg.lanes_per_row)
        self.eps = float(cfg.rms_norm_eps)
        self.row_chunk = int(cfg.prefill_row_chunk)

        ak = dict(require_l1_small=True)
        ak.update(attention_kwargs or {})
        self.attn = MotifAttention(
            mesh_device, cfg, L, source=source, ccl=ccl, rope=rope, cache=cache, spec=self.spec,
            weight_prefix=W.mtp_name("self_attn"), **ak,
        )  # fmt: skip
        self.mlp = PolyNormMLP(
            mesh_device, cfg, L, source=source, ccl=ccl, kind="dense", prefix=W.mtp_name("mlp"), cache=cache,
            **dict(mlp_kwargs or {}),
        )  # fmt: skip

        def input_proj():
            w = source.get(W.mtp_name(INPUT_PROJ))  # [4096, 8192], read once for the 8 chip blocks
            return W.stack_tp(lambda tp: W.mtp_input_proj_for_chip(w, cfg, tp), cfg, dim=0)  # [8192, 4096]

        self.w_in = W.as_tensor(
            input_proj,
            mesh_device=mesh_device,
            cfg=cfg,
            dtype=cfg.dtypes.attention,
            tp_dim=0,  # chip tp: [1024, 4096]
            cache_name=f"{_CACHE}.input_proj" if cache else None,
            layer=L,
        )

        def norm(name: str):
            return W.as_tensor(
                lambda: W.norm_weight(source.get(W.mtp_name(f"{name}.weight"))),
                mesh_device=mesh_device,
                cfg=cfg,
                dtype=cfg.dtypes.norms,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                cache_name=f"{_CACHE}.{name}.weight" if cache else None,
                layer=L,
            )

        self.norms: Dict[str, Any] = {n: norm(n) for n in W.MTP_NORMS}  # [1, 1, 128, 32] bf16 ROW_MAJOR, replicated
        self.ckc_norm = cfg.compute_config("norm")
        self.ckc_in = cfg.compute_config("attn_heads")
        self.norm_mc, self.norm_pc = cfg.decode_norm_configs()  # width-sharded 8 x 4 (README §5)
        pc = input_proj_program_config
        self.decode_pc_in = input_proj_decode_pc(cfg) if isinstance(pc, str) and pc == "default" else pc

    # ==================================================================================================================
    # building blocks
    # ==================================================================================================================
    def _need(self, what: str):
        mod = getattr(self, what)
        if mod is None:
            raise ValueError(f"MotifMTP was built without {what}= (weights only); pass the main model's {what} module")
        return mod

    def _norm(self, x, name: str, *, decode: bool):
        """RMSNorm with this layer's gamma ``name``: decode width-sharded on 8 x 4 cores (``[1, 1, 8, 4096]``, the
        decoder's measured pattern), prefill interleaved (rows spread over the grid). Not consumed."""
        gamma = self.norms[name]
        if not decode:
            return ttnn.rms_norm(
                x, epsilon=self.eps, weight=gamma, compute_kernel_config=self.ckc_norm,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # fmt: skip
        xs = ttnn.to_memory_config(x, self.norm_mc)
        ys = ttnn.rms_norm(
            xs, epsilon=self.eps, weight=gamma, program_config=self.norm_pc, memory_config=self.norm_mc,
            compute_kernel_config=self.ckc_norm,
        )  # fmt: skip
        _free(xs)
        y = ttnn.to_memory_config(ys, ttnn.DRAM_MEMORY_CONFIG)
        _free(ys)
        return y

    def _input_proj(self, hn, e_norm, *, decode: bool, taps=None):
        """``ar_tp(cat[partition_tp(hn), partition_tp(e_norm)] @ W_in[tp])`` -> ``h [1, 1, T, 4096]`` bf16 (identical on
        the TP chips). Not consumed."""
        hp = self.ccl.partition(hn, 3, "tp")  # [1, 1, T, 512]: hidden columns [512 tp, +512)
        ep = self.ccl.partition(e_norm, 3, "tp")
        x_in = ttnn.concat([hp, ep], dim=-1)  # [1, 1, T, 1024] = the chip's K slice (weights.mtp_input_proj_rows)
        _free(hp, ep)
        kw = {"program_config": self.decode_pc_in} if (decode and self.decode_pc_in is not None) else {}
        part = ttnn.linear(
            x_in, self.w_in, dtype=ttnn.bfloat16, compute_kernel_config=self.ckc_in,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw,
        )  # fmt: skip
        if taps is not None:
            taps["x_in"] = x_in
        else:
            _free(x_in)
        h = self.ccl.ar_tp(part)
        _free(part)
        return h

    def _block_input_rows(self, hn, e, *, decode: bool, want_h: bool, taps=None):
        """``(h, a)`` of rows ``e`` (embedding rows, consumed) and ``hn``: ``h = input_proj(cat[hn, embed_norm(e)])``
        (``None`` unless ``want_h``) and ``a = input_layernorm(h)`` (the attention input)."""
        en = self._norm(e, "embed_norm", decode=decode)
        if taps is not None:
            taps["e"], taps["e_norm"] = e, en
        else:
            _free(e)
        h = self._input_proj(hn, en, decode=decode, taps=taps)
        if taps is None:
            _free(en)
        a = self._norm(h, "input_layernorm", decode=decode)
        if taps is not None:
            taps["h"] = h  # (kept: the caller frees the taps)
        if not want_h:
            if taps is None:
                _free(h)
            h = None
        return h, a

    def _block_input_prefill(self, hn, next_tokens, *, want_h: bool, taps=None):
        """Prefill ``(h, a)`` for ``C`` rows, in row chunks of ``cfg.prefill_row_chunk`` (8192) when ``C`` is larger
        (row-local ops: the chunked result equals the unchunked one; the ``[C, 4096]`` outputs are concatenated)."""
        embed = self._need("embed")
        C = int(hn.shape[-2])
        if tuple(int(d) for d in hn.shape) != (1, 1, C, self.hidden) or C % TILE:
            raise ValueError(
                f"MTP prefill expects hn [1, 1, C, {self.hidden}] with C % {TILE} == 0, got {list(hn.shape)}"
            )
        if tuple(int(d) for d in next_tokens.shape) != (1, C):
            raise ValueError(f"MTP prefill expects next tokens [1, {C}] (one per row), got {list(next_tokens.shape)}")
        c = self.row_chunk
        if C <= c:
            return self._block_input_rows(hn, embed.embed_rows(next_tokens), decode=False, want_h=want_h, taps=taps)
        if taps is not None:
            raise ValueError(f"taps are only supported for unchunked MTP prefill (C {C} > prefill_row_chunk {c})")
        hs: List[Any] = []
        as_: List[Any] = []
        for s0 in range(0, C, c):
            s1 = min(C, s0 + c)
            hn_c = ttnn.slice(hn, [0, 0, s0, 0], [1, 1, s1, self.hidden])
            tok_c = ttnn.slice(next_tokens, [0, s0], [1, s1])
            h_c, a_c = self._block_input_rows(hn_c, embed.embed_rows(tok_c), decode=False, want_h=want_h)
            _free(hn_c, tok_c)
            hs.append(h_c)
            as_.append(a_c)
        a = ttnn.concat(as_, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _free(*as_)
        h = None
        if want_h:
            h = ttnn.concat(hs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _free(*hs)
        return h, a

    # ==================================================================================================================
    # prefill
    # ==================================================================================================================
    def fill_kv_prefill(
        self, hn, next_tokens, *, kv_cache, fill_pt=None, rot=None, chunk=None, taps: Optional[Dict[str, Any]] = None
    ) -> None:
        """KV-only MTP prefill of one chunk (eager; features design D9, §3.6.3): writes the MTP latent of every row
        into ``kv_cache`` and nothing else (no SDPA, MLP or head). The latent at ``p`` depends only on ``(hn_p,
        t_{p+1})`` and ``p``, so a chunk's rows need no history.

        Args:
            hn: ``head.stream_mean_norm(X)`` of the main chunk, ``[1, 1, C, 4096]`` bf16 replicated (not consumed).
            next_tokens: ``[1, C]`` uint32 ROW_MAJOR replicated: ``t_{p+1}`` of each row (:func:`mtp_next_tokens` +
                ``embed.rows_tokens_device``; padding rows any valid id).
            kv_cache: the MTP layer's paged latent cache (``[N, 1, block, 576]``, the pool's MTP cache).
            chunk: the serving form (features design §3.7.1): the chunk's :class:`~.attention.PrefillChunkInputs`
                (the same object the main layers get), passed on as ``MotifAttention.fill_kv(chunk=)``, which uses its
                fill table and RoPE rows and checks the rows against it (``C == chunk.bucket``; an sp1 chunk needs its
                ``"plain"`` RoPE rows). ``C != chunk.bucket`` is refused here already, before any device op.
            fill_pt / rot: the lower-level form, instead of ``chunk`` (never both): the fill table ``[1, >= C /
                block]`` int32 (``-1`` = skip) and RoPE rows (``None`` = positions ``0 .. C-1``; else ``(cos, sin)`` /
                ``{kind: (cos, sin)}`` with ``"plain"``), e.g. several requests' rows filled in one call.
            taps: eager debugging only; receives the intermediates (:data:`TAP_NAMES` up to ``attn_in``) and the
                attention's ``kv_row`` (bf16 latent before the cast).
        """
        if chunk is not None:
            if not isinstance(chunk, PrefillChunkInputs):
                raise TypeError(f"chunk must be a PrefillChunkInputs, got {type(chunk).__name__}")
            if fill_pt is not None or rot is not None:
                raise ValueError("fill_kv_prefill: pass chunk= or fill_pt= / rot=, not both (the chunk has its tables)")
            C = int(hn.shape[-2])
            if C != int(chunk.bucket):
                raise ValueError(f"fill_kv_prefill: hn has {C} rows, the chunk's bucket is {chunk.bucket}")
            tables = {"chunk": chunk}
        elif fill_pt is None:
            raise ValueError("fill_kv_prefill needs chunk= (or fill_pt=)")
        else:
            tables = {"fill_pt": fill_pt, "rot": rot}
        if kv_cache is None:
            raise ValueError("fill_kv_prefill needs kv_cache= (the MTP cache)")
        _, a = self._block_input_prefill(hn, next_tokens, want_h=False, taps=taps)
        attn_taps = {} if taps is not None else None
        try:
            self.attn.fill_kv(a, kv_cache=kv_cache, taps=attn_taps, **tables)
        finally:
            if taps is not None:
                taps["attn_in"] = a
                taps["kv_row"] = attn_taps.get("kv_row")
            else:
                _free(a)

    def forward_prefill(self, hn, next_tokens, *, page_table=None, kv_cache=None, rot=None, taps=None):
        """The full MTP block on ``S`` prefill rows at positions ``0 .. S-1`` (eager; tests and acceptance estimates --
        serving uses :meth:`fill_kv_prefill`): ``out = final_layernorm(...)`` ``[1, 1, S, 4096]`` bf16 replicated.
        ``hn`` / ``next_tokens`` as :meth:`fill_kv_prefill`; ``page_table`` + ``kv_cache`` (or both ``None``) fill the
        MTP cache as the attention's prefill does; ``rot`` = the attention's (default positions ``0 .. S-1``). Logits:
        :meth:`rows_argmax`."""
        h, a = self._block_input_prefill(hn, next_tokens, want_h=True, taps=taps)
        o = self.attn.forward_prefill(
            a, page_table=page_table if kv_cache is not None else None, kv_cache=kv_cache, rot=rot
        )
        return self._block_tail(h, a, o, decode=False, taps=taps)

    def _block_tail(self, h, a, o, *, decode: bool, taps=None):
        """``h + o`` -> ``+ mlp(post_attention_layernorm)`` -> ``final_layernorm``; consumes ``h``, ``a``, ``o``."""
        h1 = ttnn.add(h, o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        f = self._norm(h1, "post_attention_layernorm", decode=decode)
        u = self.mlp.forward_decode(f) if decode else self.mlp.forward_prefill(f)
        h2 = ttnn.add(h1, u, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = self._norm(h2, "final_layernorm", decode=decode)
        if taps is not None:
            taps.update(h=h, attn_in=a, attn_out=o, h_mid=h1, ffn_in=f, ffn_out=u, h_out=h2, out=out)
        else:
            _free(h, a, o, h1, f, u, h2)
        return out

    def rows_argmax(self, out, n_rows: Optional[int] = None) -> torch.Tensor:
        """Host ``int64 [n_rows]``: ``argmax(lm_head(out))`` of the first ``n_rows`` rows of a prefill-shaped ``out
        [1, 1, S, 4096]`` (:meth:`forward_prefill`), one 32-row tile at a time through the shared head's
        :meth:`~MotifLMHead.project` + :meth:`~MotifLMHead.argmax_decode` (vocab split "mesh"; the device tie rule:
        lowest index). Eager, tests / acceptance only (integer slices: one program per tile position)."""
        head = self._need("head")
        if head.vocab_split != "mesh":
            raise ValueError("rows_argmax needs the 'mesh' vocab split (32 rows per argmax_decode call)")
        S = int(out.shape[-2])
        n = S if n_rows is None else int(n_rows)
        if not 0 < n <= S or S % TILE:
            raise ValueError(f"n_rows {n} outside (0, {S}] or S {S} not a multiple of {TILE}")
        toks = []
        for r0 in range(0, n, TILE):
            t = ttnn.slice(out, [0, 0, r0, 0], [1, 1, r0 + TILE, self.hidden]) if S != TILE else out
            logits = head.project(t)
            ids = head.argmax_decode(logits)
            toks.append(head.tokens_to_host(ids)[:TILE])
            _free(logits, ids, t if t is not out else None)
        return torch.cat(toks)[:n]

    # ==================================================================================================================
    # decode
    # ==================================================================================================================
    def forward_decode(
        self,
        hn,
        tokens,
        *,
        rot,
        cur_pos,
        page_table,
        kv_cache,
        active,
        kv_write=None,
        return_hidden: bool = False,
        taps: Optional[Dict[str, Any]] = None,
    ):
        """One MTP decode step on the 8 lanes of each DP row (trace-safe; features design §3.8.1).

        Args:
            hn: ``head.stream_mean_norm(X)`` of the main model ``[1, 1, 8, 4096]`` bf16 (this DP row's lanes; not
                consumed).
            tokens: the main model's argmax ``head.argmax_decode(logits)`` ``[1, 1, 1, 32]`` uint32 ROW_MAJOR, lane
                order, on every chip (``[1, 1, 1, 8]`` per row with the "tp" vocab split); not consumed. The MTP input
                ``t_{p+1}`` of lane ``l`` at position ``cur_pos[l]``.
            rot / cur_pos / page_table / active / kv_write: the step's tensors shared with the main layers
                (``MotifAttention.forward_decode``; ``rot`` must hold the ``"plain"`` tables); ``kv_write`` is passed to
                the attention only when given. ``kv_cache``: the MTP cache (updated at ``cur_pos``).
            return_hidden: also return ``out`` (the ``final_layernorm`` output ``[1, 1, 8, 4096]``, the LM-head input;
                the caller frees it): acceptance diagnostics and trace tests.
            taps: eager debugging only (never inside a trace); receives :data:`TAP_NAMES` (not freed).

        Returns the MTP argmax ``m`` ``[1, 1, 1, 32]`` uint32 ROW_MAJOR, lane order, identical on every chip
        (``head.argmax_decode``; read with ``head.tokens_to_host``): the draft for position ``cur_pos[l] + 2``
        (``(m, out)`` with ``return_hidden``)."""
        embed, head = self._need("embed"), self._need("head")
        if int(hn.shape[-2]) != self.lanes or int(hn.shape[-1]) != self.hidden:
            raise ValueError(f"MTP decode expects hn [1, 1, {self.lanes}, {self.hidden}], got {list(hn.shape)}")
        e = embed.embed_rows_from_device(tokens)  # [1, 1, 8, 4096]
        h, a = self._block_input_rows(hn, e, decode=True, want_h=True, taps=taps)
        kw = {} if kv_write is None else {"kv_write": kv_write}
        o = self.attn.forward_decode(
            a, rot=rot, cur_pos=cur_pos, page_table=page_table, kv_cache=kv_cache, active=active, **kw
        )
        out = self._block_tail(h, a, o, decode=True, taps=taps)
        logits = head.decode_logits(out, consume=taps is None and not return_hidden)  # [1, 1, 32, 6880] ("mesh")
        m = head.argmax_decode(logits)
        if taps is not None:
            taps["logits"] = logits
        else:
            _free(logits)
        return (m, out) if return_hidden else m

    # ==================================================================================================================
    def deallocate(self) -> None:
        """Free this layer's device weights (the shared embedding / head / ccl / rope stay)."""
        for v in list(vars(self.attn).values()):
            if isinstance(v, ttnn.Tensor) and v.is_allocated():
                ttnn.deallocate(v)
        self.mlp.deallocate()
        _free(self.w_in, *self.norms.values())
        self.w_in = None
        self.norms = {}


__all__ = [
    "INPUT_PROJ",
    "INPUT_PROJ_DECODE_GRID",
    "INPUT_PROJ_DECODE_IN0_BLOCK_W",
    "MTP_CACHE_VERSION",
    "MotifMTP",
    "TAP_NAMES",
    "input_proj_decode_pc",
    "mtp_cache_complete",
    "mtp_next_tokens",
    "mtp_weights_available",
]
