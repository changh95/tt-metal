# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 block-diffusion drafter (z-lab/Qwen3.8-27B-DFlash2, Apache-2.0) for speculative decoding on the 1x4 mesh.

Reference semantics (z-lab/dflash ``dflash/model.py`` ``DFlash2DraftModel`` + the vLLM PR #52816 ``qwen3_dflash2.py``;
every formula below was read off that code and pinned by ``tests/test_dflash2_cpu.py`` against the official module):

  * Context features: ``target_hidden = hidden_norm(fc(cat_l aux_l))`` with ``aux_l`` = the TARGET model's residual
    stream AFTER decoder layer ``l`` for ``l in target_layer_ids`` (HF ``hidden_states[l+1]``; vLLM sets
    ``aux_hidden_state_layers = [l+1]``). ``fc`` [dim, 5*dim] has no bias; ``hidden_norm`` is a plain-weight RMSNorm.
  * Context K/V of drafter layer ``i`` are PROJECTIONS of ``target_hidden``: ``K_i = RoPE(k_norm_i(k_proj_i(t)))``,
    ``V_i = v_proj_i(t)`` at the context token's absolute position -- no input_layernorm, no conv, no drafter layer
    runs over the context. They are written to the drafter's paged KV once per committed token.
  * Query block per request (``block_size`` = 8 rows): ``[anchor (= the last committed token, the verify grid's row 0)
    , mask_token_id x 7]`` embedded with the TARGET's embedding table (row 248070 is the mask embedding), positions
    ``P .. P+7``. Layer ``i``: ``h = in_ln(x)``; ``(h', c) = attention_conv.prepare(h)`` (two-tap grouped dynamic conv
    over the block, see ``_conv``); ``a = o_proj(attn(q_norm(q(h')), K_i ++ k_norm(k(h')), V_i ++ v(h')))`` with the
    block's own K/V appended (RoPE'd at the block positions); ``x += attention_conv.finish(a, c)``; the same around
    the SwiGLU MLP with ``mlp_conv``. Attention is NON-causal inside the block (config ``is_causal: false`` wins over
    the ``sliding_attention`` layer type in every reference implementation) with a sliding window of 2048 over the
    context (query_pos - key_pos < 2048; irrelevant below 2048 tokens; the device SDPA has no window: documented).
  * Conv (``GroupedDynamicCausalConv``, kernel 2 taps, groups of 16 channels): ``dyn = kernel_projection(h)`` viewed
    ``[2 (pre/post), 2 (tap), 320 (groups)]``; ``out_t = sum_tap (base[side][tap] + expand16(dyn[side][tap]_t)) *
    x_{t-tap}`` with ``x_{t-1} = 0`` for the block's first row (block-local, stateless); the POST kernel
    (``dyn[1]``) is computed from the sublayer INPUT ``h`` and applied to the sublayer OUTPUT.
  * Head: ``hidden = norm(x)`` (rows 1..7 = the draft positions), ``logits = lm_head_target(hidden)`` (the drafter
    has no LM head; vocab = the target's), candidates = top-16 per position; selector: ``H = hidden_projection(hidden)``
    [7, 256]; walking from ``pred = anchor``: ``score(b) = logit_t(b) + <A(pred) * H_t, B(b)>`` over the 16 candidates
    ``b`` of position ``t`` (A/B = predecessor/successor codebooks [V, 256]); greedy ``argmax`` -> ``pred``.
  * Draft d_1..d_7 = the walked path; the verify grid (tt/verify_grid.py) is exactly the block: row 0 = anchor,
    rows 1..k = drafts. After a step the aux rows of the committed grid rows (s, 0..a_s) become new context.

Device side (``DFlash2Drafter``, TP=4): weights bfp8 (q/k/v/o heads and MLP columns sharded per device, fc
column-parallel + all-gather, conv kernel projections replicated), the residual REPLICATED [1,1,R,dim] per device
(R = 8w block rows), sub-layer partials all-reduced (reduce-scatter + all-gather), RMSNorms as explicit fp32
mean-of-squares ops (a plain ttnn.rms_norm over a 5120-wide interleaved row sizes its static CBs into the persistent
L1 buffers, logs/mtp_smoke1.log). The block's K/V go through ``paged_update_cache(num_tokens=8)`` into the drafter's
paged KV (5 layers x [k, v], the model's page table / block size, bfp8) and ONE virtual-user paged SDPA decode per
layer attends every row to the context + the whole block (cur_pos = P+7). Context writes: ``commit_rows`` (traced per
verify plan, reads ``plan.out_aux``) and ``write_context`` (prompt / payload: paged_fill_cache per block run). The
selector runs on host from the per-device top-32 readback ([TP, R, 64] values + ids) and the [R, 256] projected hidden.
Trace rules (tests/VERIFY_W32_AUDIT.md): every buffer allocated before any capture, every program compiled eagerly
before any capture, per-step values DMA'd into persistent buffers.
"""
import json
import math
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt import verify_grid as vg
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, apply_partial_rope_prefill
from models.demos.blackhole.qwen36.tt.verify_step import _EXACT_MM
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import get_block_size

DFLASH2_REPO = "z-lab/Qwen3.8-27B-DFlash2"


def dflash2_snapshot_dir():
    """The local HF snapshot of the drafter (``DFLASH2_MODEL`` = a directory or hub id; default the z-lab repo, offline)."""
    p = os.environ.get("DFLASH2_MODEL", DFLASH2_REPO)
    if os.path.isfile(os.path.join(p, "config.json")):
        return p
    from huggingface_hub import snapshot_download

    return snapshot_download(p, local_files_only=True)


class DFlash2Config:
    """The drafter's config.json (+ dflash_config) as attributes; every value the module depends on is read here."""

    def __init__(self, path):
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        d = c["dflash_config"]
        self.path = path
        self.n_layers = int(c["num_hidden_layers"])
        self.dim = int(c["hidden_size"])
        self.n_heads = int(c["num_attention_heads"])
        self.n_kv_heads = int(c["num_key_value_heads"])
        self.head_dim = int(c.get("head_dim") or self.dim // self.n_heads)  # 128 (explicit in config.json)
        self.inter = int(c["intermediate_size"])
        self.vocab_size = int(c["vocab_size"])
        self.eps = float(c["rms_norm_eps"])
        rp = c.get("rope_parameters") or {}
        self.rope_theta = float(rp.get("rope_theta", c.get("rope_theta", 10_000_000)))
        self.block_size = int(d["block_size"])
        self.conv_group_size = int(d["conv_group_size"])
        self.conv_kernel_size = int(d["conv_kernel_size"])
        self.mask_token_id = int(d["mask_token_id"])
        self.selector_rank = int(d["selector_rank"])
        self.selector_top_k = int(d["selector_top_k"])
        self.target_layer_ids = [int(v) for v in d["target_layer_ids"]]
        self.layer_types = c.get("layer_types") or ["full_attention"] * self.n_layers
        # z-lab model.py: the top-level ``is_causal`` (False for this checkpoint) overrides the layer-type default
        # (sliding layers would be causal); vLLM PR 52816 ``_dflash_layer_causal`` resolves the same way.
        ic = c.get("is_causal", d.get("causal"))
        self.is_causal = [(lt == "sliding_attention") if ic is None else bool(ic) for lt in self.layer_types]
        sw = c.get("sliding_window") if c.get("use_sliding_window", True) else None
        self.sliding_window = [sw if lt == "sliding_attention" else None for lt in self.layer_types]
        self.n_groups = self.dim // self.conv_group_size
        self.n_draft = self.block_size - 1
        assert self.conv_kernel_size == 2, "the two-tap conv is what the device path implements"
        assert len(set(self.is_causal)) == 1 and len(set(self.sliding_window)) == 1, "uniform layer types expected"


def load_dflash2_state_dict(path):
    """All drafter tensors (bf16) from the single safetensors file, keyed as in the checkpoint."""
    from safetensors import safe_open

    out = {}
    with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as sf:
        for k in sf.keys():
            out[k] = sf.get_tensor(k)
    return out


def load_target_embed_and_head(ckpt_dir):
    """The TARGET's embedding table and LM head (bf16 [V, dim]) -- the drafter uses both."""
    from safetensors import safe_open

    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    emb_key = next(k for k in wm if k.endswith("embed_tokens.weight"))
    lm_key = next(k for k in wm if k.endswith("lm_head.weight"))
    with safe_open(os.path.join(ckpt_dir, wm[emb_key]), framework="pt") as sf:
        emb = sf.get_tensor(emb_key)
    with safe_open(os.path.join(ckpt_dir, wm[lm_key]), framework="pt") as sf:
        head = sf.get_tensor(lm_key)
    return emb, head


# ============================================================================================== host reference
def _rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _rope(x, positions, theta):
    """x [S, H, HD] fp32, full head_dim rotation, HF rotate-half (cos/sin of cat(freqs, freqs))."""
    hd = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    freqs = torch.outer(positions.float(), inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    r1, r2 = x[..., : hd // 2], x[..., hd // 2 :]
    return x * cos + torch.cat([-r2, r1], dim=-1) * sin


def grouped_dynamic_conv(x, dyn, base, group_size):
    """z-lab ``_grouped_dynamic_convolve`` for one block. x [L, dim]; dyn [L, taps, groups]; base [taps, dim].
    out_t = sum_tap (base[tap] + repeat16(dyn[t, tap])) * x_{t-tap}, x_{<0} = 0 (the block's first row has one tap)."""
    L, dim = x.shape
    taps = base.shape[0]
    groups = dim // group_size
    out = torch.zeros_like(x)
    for tap in range(taps):
        shifted = x if tap == 0 else torch.cat([torch.zeros(tap, dim, dtype=x.dtype), x[:-tap]], dim=0)
        k = base[tap].reshape(1, groups, group_size) + dyn[:, tap].reshape(L, groups, 1)
        out = out + k.reshape(L, dim) * shifted
    return out


class DFlash2HostReference:
    """fp32 torch DFlash2 drafter straight from the safetensors: the context projector, the block forward (conv +
    non-causal attention over context + block), the head and the candidate selector; validated bit-for-bit (fp32) vs
    the official ``DFlash2DraftModel`` in tests/test_dflash2_cpu.py.

    Context state per request: ``state = new_state()`` then ``append_context(state, aux[N, 5*dim], positions[N])``
    (the prompt, then every step's committed rows). ``draft(state, anchor, position) -> DraftResult``."""

    def __init__(self, cfg: DFlash2Config, sd, embed, lm_head):
        self.cfg = cfg
        f = lambda k: sd[k].float()
        self.fc = f("fc.weight")  # [dim, 5*dim]
        self.hidden_norm = f("hidden_norm.weight")
        self.norm = f("norm.weight")
        self.layers = []
        for i in range(cfg.n_layers):
            p = f"layers.{i}."
            self.layers.append(
                dict(
                    in_ln=f(p + "input_layernorm.weight"),
                    post_ln=f(p + "post_attention_layernorm.weight"),
                    q=f(p + "self_attn.q_proj.weight"),
                    k=f(p + "self_attn.k_proj.weight"),
                    v=f(p + "self_attn.v_proj.weight"),
                    o=f(p + "self_attn.o_proj.weight"),
                    q_norm=f(p + "self_attn.q_norm.weight"),
                    k_norm=f(p + "self_attn.k_norm.weight"),
                    gate=f(p + "mlp.gate_proj.weight"),
                    up=f(p + "mlp.up_proj.weight"),
                    down=f(p + "mlp.down_proj.weight"),
                    conv_a_base=f(p + "attention_conv.base_kernel"),  # [2 (pre/post), taps, dim]
                    conv_a_kp=f(p + "attention_conv.kernel_projection.weight"),  # [2*taps*groups, dim]
                    conv_m_base=f(p + "mlp_conv.base_kernel"),
                    conv_m_kp=f(p + "mlp_conv.kernel_projection.weight"),
                )
            )
        self.sel_hp = f("candidate_selector.hidden_projection.weight")  # [rank, dim]
        self.sel_pred = sd["candidate_selector.predecessor_codebook"]  # bf16 [V, rank] (gathered rows -> fp32)
        self.sel_succ = sd["candidate_selector.successor_codebook"]
        self.embed = embed  # bf16 [V, dim] (target)
        self.lm_head = lm_head  # bf16 [V, dim] (target)
        assert tuple(self.embed.shape) == (cfg.vocab_size, cfg.dim), self.embed.shape

    # ---------------------------------------------------------------------------------------- context
    def project_context(self, aux, positions):
        """aux [N, 5*dim] (any float dtype) -> per layer (K [N, NKV, HD] normed + RoPE'd, V [N, NKV, HD]) fp32."""
        cfg = self.cfg
        t = _rms(torch.as_tensor(aux).float() @ self.fc.T, self.hidden_norm, cfg.eps)  # [N, dim]
        positions = torch.as_tensor(positions, dtype=torch.long)
        out = []
        for L in self.layers:
            k = (t @ L["k"].T).view(-1, cfg.n_kv_heads, cfg.head_dim)
            v = (t @ L["v"].T).view(-1, cfg.n_kv_heads, cfg.head_dim)
            k = _rope(_rms(k, L["k_norm"], cfg.eps), positions, cfg.rope_theta)
            out.append((k, v))
        return out

    @staticmethod
    def new_state():
        return {"pos": [], "kv": None}

    def append_context(self, state, aux, positions):
        kv = self.project_context(aux, positions)
        positions = [int(p) for p in positions]
        if state["kv"] is None:
            state["kv"] = kv
            state["pos"] = positions
        else:
            # a re-written position (rejected rows re-committed) replaces the old entry, like the paged cache
            keep = torch.tensor([p not in set(positions) for p in state["pos"]], dtype=torch.bool)
            state["kv"] = [
                (torch.cat([k0[keep], k1]), torch.cat([v0[keep], v1])) for (k0, v0), (k1, v1) in zip(state["kv"], kv)
            ]
            state["pos"] = [p for p, kp in zip(state["pos"], keep.tolist()) if kp] + positions

    # ---------------------------------------------------------------------------------------- block forward
    def _attn_mask(self, q_pos, k_pos, causal, window):
        vis = torch.ones(len(q_pos), len(k_pos), dtype=torch.bool)
        qp, kp = q_pos[:, None], k_pos[None, :]
        if causal:
            vis &= kp <= qp
        if window is not None:
            vis &= (qp - kp) < window
            if not causal:
                vis &= (kp - qp) < window
        return vis

    @torch.no_grad()
    def block_forward(self, state, tokens, positions):
        """tokens [L] (anchor + masks), positions [L] -> (hidden [L, dim] post-norm fp32, per-layer block K/V)."""
        cfg = self.cfg
        L = len(tokens)
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        positions = torch.as_tensor(positions, dtype=torch.long)
        x = self.embed[tokens].float()
        ctx_pos = torch.tensor(state["pos"], dtype=torch.long) if state["pos"] else torch.zeros(0, dtype=torch.long)
        g = cfg.n_heads // cfg.n_kv_heads
        for li, Lw in enumerate(self.layers):
            h = _rms(x, Lw["in_ln"], cfg.eps)
            dyn = (h @ Lw["conv_a_kp"].T).view(L, 2, cfg.conv_kernel_size, cfg.n_groups)
            hc = grouped_dynamic_conv(h, dyn[:, 0], Lw["conv_a_base"][0], cfg.conv_group_size)
            q = _rms((hc @ Lw["q"].T).view(L, cfg.n_heads, cfg.head_dim), Lw["q_norm"], cfg.eps)
            k = _rms((hc @ Lw["k"].T).view(L, cfg.n_kv_heads, cfg.head_dim), Lw["k_norm"], cfg.eps)
            v = (hc @ Lw["v"].T).view(L, cfg.n_kv_heads, cfg.head_dim)
            q = _rope(q, positions, cfg.rope_theta)
            k = _rope(k, positions, cfg.rope_theta)
            if state["kv"] is not None:
                K = torch.cat([state["kv"][li][0], k])
                V = torch.cat([state["kv"][li][1], v])
                kpos = torch.cat([ctx_pos, positions])
            else:
                K, V, kpos = k, v, positions
            Kh = K.repeat_interleave(g, dim=1).permute(1, 0, 2)  # [NH, n, HD]
            Vh = V.repeat_interleave(g, dim=1).permute(1, 0, 2)
            scores = torch.matmul(q.permute(1, 0, 2), Kh.transpose(1, 2)) * (cfg.head_dim**-0.5)
            vis = self._attn_mask(positions, kpos, cfg.is_causal[li], cfg.sliding_window[li])
            scores = scores.masked_fill(~vis[None], float("-inf"))
            out = torch.matmul(torch.softmax(scores, dim=-1), Vh).permute(1, 0, 2).reshape(L, -1)
            a = out @ Lw["o"].T
            a = grouped_dynamic_conv(a, dyn[:, 1], Lw["conv_a_base"][1], cfg.conv_group_size)
            x = x + a
            h2 = _rms(x, Lw["post_ln"], cfg.eps)
            dyn2 = (h2 @ Lw["conv_m_kp"].T).view(L, 2, cfg.conv_kernel_size, cfg.n_groups)
            hc2 = grouped_dynamic_conv(h2, dyn2[:, 0], Lw["conv_m_base"][0], cfg.conv_group_size)
            m = (torch.nn.functional.silu(hc2 @ Lw["gate"].T) * (hc2 @ Lw["up"].T)) @ Lw["down"].T
            m = grouped_dynamic_conv(m, dyn2[:, 1], Lw["conv_m_base"][1], cfg.conv_group_size)
            x = x + m
        return _rms(x, self.norm, cfg.eps)

    def logits(self, hidden):
        out = torch.empty(hidden.shape[0], self.cfg.vocab_size, dtype=torch.float32)
        step = 32768
        for a in range(0, self.cfg.vocab_size, step):
            out[:, a : a + step] = hidden @ self.lm_head[a : a + step].float().T
        return out

    def select(self, hidden, logits, anchor, candidates=None):
        """The candidate selector (z-lab ``CandidateSelector.select``, greedy). hidden/logits: the draft rows
        [n, dim] / [n, V]. candidates [n, top_k] (optional, default the logits' top-k). Returns (path [n],
        candidates [n, top_k], scores [n, top_k])."""
        k = self.cfg.selector_top_k
        if candidates is None:
            candidates = torch.topk(logits, k, dim=-1).indices
        unary = logits.gather(-1, candidates)  # [n, k]
        hp = hidden @ self.sel_hp.T  # [n, rank]
        path, scores = selector_walk(hp, candidates, unary, anchor, self.sel_pred, self.sel_succ)
        return path, candidates, scores

    @torch.no_grad()
    def draft(self, state, anchor, position, n_draft=None):
        """One draft step at block positions position..position+block_size-1 -> dict(tokens [n_draft], candidates,
        logits [n_draft, V], hidden [n_draft, dim])."""
        cfg = self.cfg
        n = cfg.n_draft if n_draft is None else int(n_draft)
        toks = [int(anchor)] + [cfg.mask_token_id] * n
        pos = torch.arange(int(position), int(position) + n + 1)
        hidden = self.block_forward(state, toks, pos)[1:]  # the draft rows
        logits = self.logits(hidden)
        path, cands, scores = self.select(hidden, logits, anchor)
        return {"tokens": path, "candidates": cands, "scores": scores, "logits": logits, "hidden": hidden}


def selector_walk(hp, cands, unary, anchor, pred_cb, succ_cb):
    """The greedy candidate-selector walk shared by the host reference and the device path. hp [n, rank] float
    (hidden_projection of the draft rows), cands [n, k] long, unary [n, k] float (the candidates' logits), anchor int,
    pred_cb / succ_cb [V, rank] (bf16 tables). Returns (path [n], scores [n, k])."""
    pred = int(anchor)
    path, scores_all = [], []
    for t in range(hp.shape[0]):
        a = pred_cb[pred].float() * hp[t].float()
        b = succ_cb[cands[t]].float()
        scores = unary[t].float() + b @ a
        idx = int(torch.argmax(scores))
        pred = int(cands[t, idx])
        path.append(pred)
        scores_all.append(scores)
    return path, torch.stack(scores_all)


# ============================================================================================== device side
_BYTES = {}


def _bytes_per_elem(dtype):
    if not _BYTES:
        _BYTES.update(
            {
                ttnn.bfloat16: 2.0,
                ttnn.float32: 4.0,
                ttnn.bfloat8_b: 1.0625,
                ttnn.bfloat4_b: 0.5625,
                ttnn.int32: 4.0,
                ttnn.uint32: 4.0,
                ttnn.uint16: 2.0,
            }
        )
    return _BYTES.get(dtype, 2.0)


def tensor_bytes(t):
    """Per-device bytes of a (mesh) tensor from its padded shape and dtype."""
    return int(math.prod(int(v) for v in t.padded_shape) * _bytes_per_elem(t.dtype))


class _StepBufs:
    def __init__(self):
        self.tok = self.cos = self.sin = self.pos0 = self.pt = None
        self.pt_host = None
        self.chunks = []  # (a, b, pt_rows [b-a, blocks], cur_rows [b-a])
        self.shift = None  # [1,1,R,R] 0/1 block-local row shift
        self.spread = []  # per kv head [1,1,w*32,R] 0/1 (row s*32 + h*8 + j <- row s*8 + j)
        self.trace_id = None
        self.out_vals = self.out_idx = self.out_hp = self.out_logits = None


class _CommitBufs:
    def __init__(self):
        self.cos = self.sin = None  # [1,1,R,HD] rows (position P_s + j)
        self.spread = []  # per kv head [1,1,w*32,R] (row s*32 + h*T + j <- row s*T + j)
        self.trace_id = None
        self.tmp_aux = None  # compile-time stand-in for plan.out_aux


class DFlash2Drafter:
    """DFlash2 drafter on the mesh (module docstring). Build AFTER ``allocate_kv_caches`` and BEFORE any trace capture.

    API (D side):
      * ``allocate_kv(model, num_blocks)`` -> ``[[k, v]] * n_layers`` (paged, model block size + 1 pad block, bfp8),
        passed as ``kv_caches=`` or allocated by the constructor; ``kv_caches`` / ``kv_layers`` expose them for a
        pd_transfer-style block export/import (the same block ids as the target's caches).
      * ``projector.project(aux, positions)`` -> per layer (K [N, n_kv_heads, HD] roped, V) host tensors (all heads).
      * ``write_context(slot, positions, aux)`` (prompt / imported aux rows; eager, block fill) and
        ``import_context_kv(slot, positions, K_layers, V_layers)`` (a payload of projected K/V).
      * ``bind_plan(plan)`` / ``compile_commit(plan)`` / ``capture_commit(plan)`` / ``commit(plan, positions)``: the
        traced context write of the verify grid's rows from ``plan.out_aux`` (rows r = s*T + j at P_s + j).
      * ``compile_step(w)`` / ``capture_step(w)`` / ``draft(w, anchors, positions)`` -> (drafts [w][7], candidates,
        info); the traced block forward at width w (R = 8w rows) + the host selector.
    """

    def __init__(
        self,
        model,
        page_tables=None,
        widths=(),
        cfg_path=None,
        keep_logits=False,
        num_blocks=None,
        kv_caches=None,
        sel_topk_local=32,
    ):
        self.model = model
        self.mesh = mesh = model.mesh_device
        self.args = args = model.args
        self.tt_ccl = model.tt_ccl
        self.nd = model.num_devices
        assert self.nd > 1, "the DFlash2 drafter is TP only"
        assert model._paged_kv_caches, "allocate_kv_caches first (the drafter KV mirrors the main cache's blocks)"
        t0 = time.perf_counter()
        path = cfg_path or dflash2_snapshot_dir()
        self.cfg = cfg = DFlash2Config(path)
        assert cfg.dim == args.dim and cfg.vocab_size == args.vocab_size, "drafter/target dims"
        assert cfg.n_heads % self.nd == 0 and cfg.n_kv_heads % self.nd == 0
        self.NH = cfg.n_heads // self.nd  # 8
        self.NKV = cfg.n_kv_heads // self.nd  # 2
        self.HD = cfg.head_dim  # 128
        self.dim = cfg.dim
        self.B = cfg.block_size  # 8 query rows per user
        self.keep_logits = bool(keep_logits)
        self.rep = ttnn.ReplicateTensorToMesh(mesh)
        self.per_shard = args.vocab_size // self.nd
        self.grid_w = getattr(args, "decode_grid_w", 8)
        self.k_local = int(sel_topk_local)  # per-device per-half top-k (host merges to the selector's top-16)
        self.eps = cfg.eps
        assert self.NKV * self.B <= tpc.TILE_SIZE, "paged_update_cache(num_tokens) needs num_kv_heads * T <= 32"
        self.causal = cfg.is_causal[0]
        if cfg.sliding_window[0] is not None:
            logger.warning(
                f"[dflash2] sliding window {cfg.sliding_window[0]} of the reference is NOT applied on device (the paged "
                "SDPA decode attends to the whole context); identical below that context length"
            )
        self._host_refs = []
        self.weight_bytes = 0
        self._load_weights(load_dflash2_state_dict(path), args.weight_cache_path())
        logger.info(
            f"[dflash2] weights on device in {time.perf_counter() - t0:.1f}s ({self.weight_bytes / 2**20:.0f} MiB/chip)"
        )

        # --- paged KV: n_layers x [k, v], the main cache's block count + 1 pad block, bfp8 ---
        k0 = model._paged_kv_caches[0][0]
        self.block_size = get_block_size(model._paged_kv_caches)
        self.n_main_blocks = int(k0.shape[0])
        self.pad_block = self.n_main_blocks
        self.kv_caches = list(kv_caches) if kv_caches is not None else self.allocate_kv(model, num_blocks, cfg)
        assert len(self.kv_caches) == cfg.n_layers
        for k, v in self.kv_caches:
            assert tuple(k.shape) == (self.n_main_blocks + 1, self.NKV, self.block_size, self.HD), tuple(k.shape)
        self.kv_bytes = sum(tensor_bytes(k) + tensor_bytes(v) for k, v in self.kv_caches)
        self.kv_dtype = self.kv_caches[0][0].dtype

        # --- page tables / step buffers ---
        self.page_tables = None
        if page_tables is not None:
            pt = page_tables if isinstance(page_tables, torch.Tensor) else torch.as_tensor(page_tables)
            self.page_tables = pt.to(torch.int32)
            assert self.page_tables.shape[1] % 8 == 0
        grid = mesh.compute_with_storage_grid_size()
        self.n_cores = grid.x * grid.y
        self.sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )
        self._kv_cfg_cache = {}
        self.sb = {}
        if widths:
            assert self.page_tables is not None, "draft-step widths need the decode page tables"
            self.build_step_buffers(widths, self.page_tables)
        self._plans = {}
        self.stats = {"draft_steps": 0, "draft_wall": 0.0, "select_wall": 0.0, "commit_steps": 0, "commit_wall": 0.0}
        self.projector = DFlash2ContextProjector(self)
        assert getattr(model, "dflash2_drafter", None) is None, "the model already has a DFlash2 drafter"
        model.dflash2_drafter = self  # tt/pd_transfer.py registers kv_layers / pad_block as KV group "dflash2"
        logger.info(
            f"[dflash2] drafter ready: widths {sorted(self.sb)} kv {cfg.n_layers} x 2 x {list(self.kv_caches[0][0].shape)} "
            f"{self.kv_dtype} ({self.kv_bytes / 2**20:.0f} MiB/chip) block attention {'causal' if self.causal else 'non-causal'}"
        )

    # ------------------------------------------------------------------------------------------ weights
    def _as(self, t, dtype, name, mapper=None, layout=ttnn.TILE_LAYOUT, cache=None):
        out = ttnn.as_tensor(
            t,
            dtype=dtype,
            device=self.mesh,
            mesh_mapper=mapper or self.rep,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(str(cache / name) if cache is not None else None),
        )
        self.weight_bytes += tensor_bytes(out)
        return out

    def _load_weights(self, sd, cache_root):
        cfg, nd = self.cfg, self.nd
        cache = cache_root / "dflash2"  # per-tensor bfp8 cache files (bf16 -> bfp8 conversion once)
        os.makedirs(cache, exist_ok=True)
        NH, NKV, HD, dim = self.NH, self.NKV, self.HD, self.dim
        shard_col = ttnn.ShardTensorToMesh(self.mesh, dim=-1)
        shard_row = ttnn.ShardTensorToMesh(self.mesh, dim=-2)
        bf = lambda k: sd[k].to(torch.bfloat16)
        row4 = lambda t: t.reshape(1, 1, 1, -1).to(torch.bfloat16)
        self.layers = []
        kv_all = []  # the context projector's fused [dim, n_layers * 2 * NKV*HD] per device
        for i in range(cfg.n_layers):
            p = f"layers.{i}."
            q_t = bf(p + "self_attn.q_proj.weight").T  # [dim, NH_tot*HD]
            k_t = bf(p + "self_attn.k_proj.weight").T  # [dim, NKV_tot*HD]
            v_t = bf(p + "self_attn.v_proj.weight").T
            fused = torch.cat(
                [
                    torch.cat(
                        [
                            q_t[:, d * NH * HD : (d + 1) * NH * HD],
                            k_t[:, d * NKV * HD : (d + 1) * NKV * HD],
                            v_t[:, d * NKV * HD : (d + 1) * NKV * HD],
                        ],
                        dim=-1,
                    )
                    for d in range(nd)
                ],
                dim=-1,
            )  # [dim, nd * (NH+2NKV)*HD]; device d gets its q heads then its kv heads
            kv_all.append(
                torch.cat(
                    [
                        torch.cat(
                            [k_t[:, d * NKV * HD : (d + 1) * NKV * HD], v_t[:, d * NKV * HD : (d + 1) * NKV * HD]],
                            dim=-1,
                        )
                        for d in range(nd)
                    ],
                    dim=-1,
                )
            )  # [dim, nd * 2*NKV*HD]
            L = dict(
                wqkv=self._as(fused, ttnn.bfloat8_b, f"l{i}.wqkv", shard_col, cache=cache),
                wo=self._as(
                    bf(p + "self_attn.o_proj.weight").T.contiguous(), ttnn.bfloat8_b, f"l{i}.wo", shard_row, cache=cache
                ),
                w1=self._as(
                    bf(p + "mlp.gate_proj.weight").T.contiguous(), ttnn.bfloat8_b, f"l{i}.w1", shard_col, cache=cache
                ),
                w3=self._as(
                    bf(p + "mlp.up_proj.weight").T.contiguous(), ttnn.bfloat8_b, f"l{i}.w3", shard_col, cache=cache
                ),
                w2=self._as(
                    bf(p + "mlp.down_proj.weight").T.contiguous(), ttnn.bfloat8_b, f"l{i}.w2", shard_row, cache=cache
                ),
                kp_a=self._as(
                    bf(p + "attention_conv.kernel_projection.weight").T.contiguous(),
                    ttnn.bfloat8_b,
                    f"l{i}.kp_a",
                    cache=cache,
                ),
                kp_m=self._as(
                    bf(p + "mlp_conv.kernel_projection.weight").T.contiguous(),
                    ttnn.bfloat8_b,
                    f"l{i}.kp_m",
                    cache=cache,
                ),
                in_ln=self._as(row4(sd[p + "input_layernorm.weight"]), ttnn.bfloat16, f"l{i}.in_ln", cache=cache),
                post_ln=self._as(
                    row4(sd[p + "post_attention_layernorm.weight"]), ttnn.bfloat16, f"l{i}.post_ln", cache=cache
                ),
                q_norm=self._as(row4(sd[p + "self_attn.q_norm.weight"]), ttnn.bfloat16, f"l{i}.q_norm", cache=cache),
                k_norm=self._as(row4(sd[p + "self_attn.k_norm.weight"]), ttnn.bfloat16, f"l{i}.k_norm", cache=cache),
                base_a=[
                    [
                        self._as(
                            row4(sd[p + "attention_conv.base_kernel"][s, t]),
                            ttnn.bfloat16,
                            f"l{i}.base_a{s}{t}",
                            cache=cache,
                        )
                        for t in range(2)
                    ]
                    for s in range(2)
                ],
                base_m=[
                    [
                        self._as(
                            row4(sd[p + "mlp_conv.base_kernel"][s, t]), ttnn.bfloat16, f"l{i}.base_m{s}{t}", cache=cache
                        )
                        for t in range(2)
                    ]
                    for s in range(2)
                ],
            )
            self.layers.append(L)
        self.w_kv_all = self._as(
            torch.cat(
                [
                    torch.cat(
                        [kv_all[i][:, d * 2 * NKV * HD : (d + 1) * 2 * NKV * HD] for i in range(cfg.n_layers)], dim=-1
                    )
                    for d in range(nd)
                ],
                dim=-1,
            ),
            ttnn.bfloat8_b,
            "kv_all",
            shard_col,
            cache=cache,
        )  # per device [dim, n_layers*2*NKV*HD]: layer i at cols i*2*NKV*HD, k heads first then v heads
        self.w_fc = self._as(
            bf("fc.weight").T.contiguous(), ttnn.bfloat8_b, "fc", shard_col, cache=cache
        )  # [5*dim, dim/TP]
        self.hidden_norm = self._as(row4(sd["hidden_norm.weight"]), ttnn.bfloat16, "hidden_norm", cache=cache)
        self.final_norm = self._as(row4(sd["norm.weight"]), ttnn.bfloat16, "norm", cache=cache)
        self.w_hp = self._as(
            bf("candidate_selector.hidden_projection.weight").T.contiguous(), ttnn.bfloat16, "sel_hp", cache=cache
        )
        # group expansion E [groups, dim]: E[g, g*16 + i] = 1 (dyn [R, groups] @ E -> per-channel kernel)
        G, gs = cfg.n_groups, cfg.conv_group_size
        E = torch.zeros(1, 1, G, dim)
        for g in range(G):
            E[0, 0, g, g * gs : (g + 1) * gs] = 1.0
        self.E = self._as(E.to(torch.bfloat16), ttnn.bfloat16, "E", cache=cache)
        # host selector tables (bf16, ~254 MiB each) -- the walk runs on host from the device top-k readback
        self.sel_pred = sd["candidate_selector.predecessor_codebook"]
        self.sel_succ = sd["candidate_selector.successor_codebook"]

    @staticmethod
    def allocate_kv(model, num_blocks=None, cfg=None, kv_dtype=ttnn.bfloat8_b):
        """n_layers x [k, v] paged caches [num_blocks + 1, n_kv_heads/TP, block_size, head_dim] (bfp8 by default,
        whatever the main cache's dtype), one pad block past the main cache's block count (never in a page table),
        replicated shape per device (each device holds its own kv heads). Allocate BEFORE any capture."""
        cfg = cfg or DFlash2Config(dflash2_snapshot_dir())
        k0 = model._paged_kv_caches[0][0]
        nb = int(k0.shape[0]) if num_blocks is None else int(num_blocks)
        assert nb == int(k0.shape[0]), f"drafter KV block count {nb} must equal the main cache's {k0.shape[0]}"
        shape = [nb + 1, cfg.n_kv_heads // model.num_devices, int(k0.shape[2]), cfg.head_dim]
        rep = ttnn.ReplicateTensorToMesh(model.mesh_device)

        def _mk():
            try:
                return ttnn.zeros(
                    shape,
                    dtype=kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=model.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            except Exception:  # noqa: BLE001 -- ttnn.zeros without bfp8 support: host path
                return ttnn.as_tensor(
                    torch.zeros(shape, dtype=torch.bfloat16),
                    device=model.mesh_device,
                    dtype=kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rep,
                )

        return [[_mk(), _mk()] for _ in range(cfg.n_layers)]

    @property
    def kv_layers(self):
        """[(k, v)] per drafter layer -- extra attention layers for a pd_transfer-style block export/import."""
        return [tuple(kv) for kv in self.kv_caches]

    def dram_report(self, tokens=None):
        """Per-chip bytes: weights, the allocated KV pool and the KV pool at ``tokens`` positions (bfp8)."""
        per_tok = self.cfg.n_layers * 2 * self.NKV * self.HD * _bytes_per_elem(self.kv_dtype)
        rep = {"weights_bytes": self.weight_bytes, "kv_pool_bytes": self.kv_bytes, "kv_bytes_per_token": per_tok}
        if tokens is not None:
            rep[f"kv_bytes_at_{tokens}_tokens"] = int(per_tok * tokens)
        return rep

    # ------------------------------------------------------------------------------------------ helpers
    def _up(self, t, dtype, layout):
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=self.rep
        )

    def _dma(self, host_t, dst, dtype, layout):
        h = ttnn.from_torch(host_t, dtype=dtype, layout=layout, device=None, mesh_mapper=self.rep)
        ttnn.copy_host_to_device_tensor(h, dst)
        self._host_refs.append(h)

    def _sync(self):
        ttnn.synchronize_device(self.mesh)
        self._host_refs = []

    def _kv_cfg(self, B):
        cfg = self._kv_cfg_cache.get(B)
        if cfg is None:
            cols = next(c for c in range(min(8, B), 0, -1) if B % c == 0)
            cfg = ttnn.create_sharded_memory_config(
                shape=(tpc.TILE_SIZE, self.HD),
                core_grid=ttnn.CoreGrid(x=cols, y=B // cols),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self._kv_cfg_cache[B] = cfg
        return cfg

    def _pc(self, m, k, n, act=None):
        return tpc.small_m_progcfg(m, k, n, act, self.grid_w)

    def _mm(self, x, w, act=None, out=ttnn.L1_MEMORY_CONFIG, exact=False):
        """[1,1,R,K] x [K,N] weight (interleaved) on the small-M 1D mcast config; x is moved to L1 interleaved."""
        pc = self._pc(x.shape[-2], x.shape[-1], w.shape[-1], act)
        # NOT tpc.matmul_1d_decode: its to_memory_config(x, L1) of an already L1-interleaved x returns a NEW tensor
        # object aliasing x's buffer, which it then deallocates -- x must survive here (the conv reuses it).
        return ttnn.linear(
            x, w, compute_kernel_config=_EXACT_MM if exact else tpc.COMPUTE_HIFI2, program_config=pc, memory_config=out
        )

    def _rms(self, x, w, out=ttnn.L1_MEMORY_CONFIG):
        """Plain-weight RMSNorm of [1,1,R,dim] rows as explicit ops with fp32 statistics (see module docstring)."""
        L1 = ttnn.L1_MEMORY_CONFIG
        x32 = ttnn.typecast(x, ttnn.float32, memory_config=L1)
        sq = ttnn.multiply(x32, x32, memory_config=L1)
        ms = ttnn.mean(sq, dim=-1, keepdim=True, memory_config=L1, compute_kernel_config=_EXACT_MM)
        ttnn.deallocate(sq)
        ms_e = ttnn.add(ms, self.eps, memory_config=L1)
        ttnn.deallocate(ms)
        rs = ttnn.rsqrt(ms_e, memory_config=L1)
        ttnn.deallocate(ms_e)
        y = ttnn.multiply(x32, rs, memory_config=L1)
        ttnn.deallocate(x32)
        ttnn.deallocate(rs)
        y16 = ttnn.typecast(y, ttnn.bfloat16, memory_config=L1)
        ttnn.deallocate(y)
        out_t = ttnn.multiply(y16, w, memory_config=out)
        ttnn.deallocate(y16)
        return out_t

    def _all_gather(self, frac):
        """[1,1,R,dim/TP] fractured -> replicated [1,1,R,dim] DRAM; consumes."""
        g = ttnn.experimental.all_gather_async(
            frac,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.tt_ccl.get_num_links(1),
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        ttnn.deallocate(frac)
        return g

    def _all_reduce(self, partial):
        """Per-device partial [1,1,R,dim] -> replicated sum [1,1,R,dim] DRAM (reduce-scatter + all-gather); consumes."""
        frac = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._all_gather(frac)

    def _conv(self, x, dyn, base, side, shift):
        """One grouped dynamic conv application over the block rows. x [1,1,R,dim]; dyn [1,1,R,2*2*groups] (the
        kernel projection of the sublayer INPUT); base[side][tap] [1,1,1,dim]; shift [1,1,R,R] 0/1 (row t <- t-1 inside
        each user's block). out_t = (base0 + E(dyn[side,0])_t) * x_t + (base1 + E(dyn[side,1])_t) * x_{t-1}."""
        L1 = ttnn.L1_MEMORY_CONFIG
        R, G = x.shape[-2], self.cfg.n_groups
        ks = []
        for tap in range(2):
            c0 = (side * 2 + tap) * G
            d = ttnn.slice(dyn, (0, 0, 0, c0), (1, 1, R, c0 + G), memory_config=L1)
            kf = ttnn.linear(
                d, self.E, compute_kernel_config=_EXACT_MM, program_config=self._pc(R, G, self.dim), memory_config=L1
            )
            ttnn.deallocate(d)
            kb = ttnn.add(kf, base[side][tap], memory_config=L1)
            ttnn.deallocate(kf)
            ks.append(kb)
        xs = ttnn.matmul(shift, x, compute_kernel_config=_EXACT_MM, memory_config=L1)
        t0 = ttnn.multiply(ks[0], x, memory_config=L1)
        t1 = ttnn.multiply(ks[1], xs, memory_config=L1)
        for t in (ks[0], ks[1], xs):
            ttnn.deallocate(t)
        out = ttnn.add(t0, t1, memory_config=L1)
        ttnn.deallocate(t0)
        ttnn.deallocate(t1)
        return out

    def _kv_heads(self, kv, col0, L, cos, sin, is_k):
        """Per kv head [1,1,R,HD] slices of columns col0.. of kv [1,1,R,*]; K: k_norm + RoPE(rows cos/sin)."""
        L1 = ttnn.L1_MEMORY_CONFIG
        R, HD = kv.shape[-2], self.HD
        outs = []
        for h in range(self.NKV):
            c = col0 + h * HD
            t = ttnn.slice(kv, (0, 0, 0, c), (1, 1, R, c + HD), memory_config=L1)
            if is_k:
                n = ttnn.multiply(ttnn.rms_norm(t, epsilon=self.eps, memory_config=L1), L["k_norm"], memory_config=L1)
                ttnn.deallocate(t)
                t = apply_partial_rope_prefill(n, cos, sin, 1, HD)  # [1,1,R,HD] L1
                ttnn.deallocate(n)
            outs.append(t)
        return outs

    def _spread_update(self, cache, heads, spreads, w, pos0, page_table, T):
        """heads: per kv head [1,1,R,HD]; spreads: per head [1,1,w*32,R] 0/1 -> [1,w,32,HD] shard rows h*T + j ->
        paged_update_cache(num_tokens=T) at pos0 [w] through page_table [w, blocks]. Consumes heads."""
        L1 = ttnn.L1_MEMORY_CONFIG
        acc = None
        for h, t in enumerate(heads):
            part = ttnn.matmul(spreads[h], t, compute_kernel_config=_EXACT_MM, memory_config=L1)
            ttnn.deallocate(t)
            if acc is None:
                acc = part
            else:
                s = ttnn.add(acc, part, memory_config=L1)
                ttnn.deallocate(acc)
                ttnn.deallocate(part)
                acc = s
        sp4 = ttnn.reshape(acc, (1, w, tpc.TILE_SIZE, self.HD))
        sh = ttnn.to_memory_config(sp4, self._kv_cfg(w))
        ttnn.deallocate(sp4)
        if sp4 is not acc:
            ttnn.deallocate(acc)
        ttnn.experimental.paged_update_cache(cache, sh, update_idxs_tensor=pos0, page_table=page_table, num_tokens=T)
        ttnn.deallocate(sh)

    @staticmethod
    def spread_matrices(w, T, R, NKV):
        """Per kv head h: [1,1,w*32,R] 0/1 with row s*32 + h*T + j <- grid row s*T + j (float32; exact in bf16)."""
        out = []
        for h in range(NKV):
            m = torch.zeros(1, 1, w * tpc.TILE_SIZE, R, dtype=torch.float32)
            for s in range(w):
                for j in range(T):
                    m[0, 0, s * tpc.TILE_SIZE + h * T + j, s * T + j] = 1.0
            out.append(m)
        return out

    @staticmethod
    def shift_matrix(w, T, R):
        """[1,1,R,R] 0/1: row s*T + j <- row s*T + j - 1 for j >= 1 (block-local previous row), zero for j = 0."""
        m = torch.zeros(1, 1, R, R, dtype=torch.float32)
        for s in range(w):
            for j in range(1, T):
                m[0, 0, s * T + j, s * T + j - 1] = 1.0
        return m

    # ------------------------------------------------------------------------------------------ context projection
    def project_kv(self, aux, cos, sin):
        """Device: aux [1,1,N,5*dim] replicated -> per layer (K heads [NKV x [1,1,N,HD]] normed+roped, V heads).
        cos/sin [1,1,N,HD] of the rows' positions. fc column-parallel + all-gather, hidden_norm, one fused K/V GEMM."""
        t_frac = self._mm(aux, self.w_fc, out=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,N,dim/TP]
        t = self._all_gather(t_frac)
        tn = self._rms(t, self.hidden_norm)
        ttnn.deallocate(t)
        kv = self._mm(tn, self.w_kv_all)  # [1,1,N,n_layers*2*NKV*HD] L1
        ttnn.deallocate(tn)
        out = []
        stride = 2 * self.NKV * self.HD
        for i, L in enumerate(self.layers):
            ks = self._kv_heads(kv, i * stride, L, cos, sin, True)
            vs = self._kv_heads(kv, i * stride + self.NKV * self.HD, L, cos, sin, False)
            out.append((ks, vs))
        ttnn.deallocate(kv)
        return out

    def _rope_rows(self, positions):
        pos = torch.as_tensor(positions, dtype=torch.int32).reshape(-1)
        return vg.rope_cos_sin(pos.clamp(min=0), self.HD, self.cfg.rope_theta)

    def write_context(self, slot, positions, aux, page_table_row=None):
        """Eager: project the aux rows ``aux`` [N, 5*dim] (host, any float dtype) at ``positions`` [N] (a contiguous
        run starting at a block boundary, e.g. the prompt 0..N-1) and write the K/V of every drafter layer into the
        blocks of ``page_table_row`` (default ``page_tables[slot]``) with paged_fill_cache (whole blocks: the rows past
        N in the last block hold zeros, never read before the block K/V overwrite them)."""
        t0 = time.perf_counter()
        pos = torch.as_tensor(positions, dtype=torch.int64).reshape(-1)
        N = int(pos.numel())
        assert N >= 1 and torch.equal(pos, torch.arange(int(pos[0]), int(pos[0]) + N)), "contiguous positions"
        assert int(pos[0]) % self.block_size == 0, "write_context: the run must start at a block boundary"
        pt_row = self.page_tables[int(slot)] if page_table_row is None else torch.as_tensor(page_table_row)
        pt_row = pt_row.reshape(-1).to(torch.int32)
        aux_t = torch.as_tensor(aux).to(torch.bfloat16).reshape(1, 1, N, 5 * self.dim)
        S = -(-N // self.block_size) * self.block_size
        if S > N:
            aux_t = torch.cat([aux_t, torch.zeros(1, 1, S - N, 5 * self.dim, dtype=torch.bfloat16)], dim=2)
        pos_pad = torch.cat([pos, torch.zeros(S - N, dtype=torch.int64)])
        aux_tt = self._up(aux_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        cos_t, sin_t = self._rope_rows(pos_pad)
        cos_tt = self._up(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        sin_tt = self._up(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        kvs = self.project_kv(aux_tt, cos_tt, sin_tt)
        blk0 = int(pos[0]) // self.block_size
        nblk = S // self.block_size
        fill_pt = self._up(pt_row[blk0 : blk0 + nblk].reshape(1, nblk).contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        for (k_cache, v_cache), (ks, vs) in zip(self.kv_caches, kvs):
            for cache, heads in ((k_cache, ks), (v_cache, vs)):
                # [1,1,S,HD] per head -> [1,NKV,S,HD] in the cache dtype -> whole-block fill
                hs = [ttnn.reshape(h, (1, 1, S, self.HD)) for h in heads]
                cat = ttnn.concat(hs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG) if self.NKV > 1 else hs[0]
                fill = ttnn.typecast(cat, self.kv_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.experimental.paged_fill_cache(cache, fill, fill_pt, batch_idx=0)
                ttnn.deallocate(fill)
                if cat is not hs[0]:
                    ttnn.deallocate(cat)
                for h in heads:
                    ttnn.deallocate(h)
        self._sync()
        for t in (aux_tt, cos_tt, sin_tt, fill_pt):
            ttnn.deallocate(t)
        self.stats["commit_wall"] += time.perf_counter() - t0
        return N

    def import_context_kv(self, slot, positions, k_layers, v_layers, page_table_row=None):
        """A payload of already-projected K/V (host: per layer K [N, n_kv_heads, HD] roped/normed bf16, V) for the rows
        at ``positions`` (contiguous, block-aligned start): this device's kv heads are sliced and block-filled."""
        pos = torch.as_tensor(positions, dtype=torch.int64).reshape(-1)
        N = int(pos.numel())
        assert int(pos[0]) % self.block_size == 0 and torch.equal(pos, torch.arange(int(pos[0]), int(pos[0]) + N))
        pt_row = self.page_tables[int(slot)] if page_table_row is None else torch.as_tensor(page_table_row)
        pt_row = pt_row.reshape(-1).to(torch.int32)
        S = -(-N // self.block_size) * self.block_size
        blk0 = int(pos[0]) // self.block_size
        nblk = S // self.block_size
        fill_pt = self._up(pt_row[blk0 : blk0 + nblk].reshape(1, nblk).contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        shard_heads = ttnn.ShardTensorToMesh(self.mesh, dim=1)
        for (k_cache, v_cache), K, V in zip(self.kv_caches, k_layers, v_layers):
            for cache, t in ((k_cache, K), (v_cache, V)):
                t = torch.as_tensor(t).to(torch.bfloat16).reshape(N, self.cfg.n_kv_heads, self.HD).permute(1, 0, 2)
                if S > N:
                    t = torch.cat([t, torch.zeros(self.cfg.n_kv_heads, S - N, self.HD, dtype=torch.bfloat16)], dim=1)
                dev = ttnn.from_torch(
                    t.reshape(1, self.cfg.n_kv_heads, S, self.HD).contiguous(),
                    dtype=self.kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=shard_heads,
                )
                ttnn.experimental.paged_fill_cache(cache, dev, fill_pt, batch_idx=0)
                ttnn.deallocate(dev)
        self._sync()
        ttnn.deallocate(fill_pt)

    # ------------------------------------------------------------------------------------------ per-plan commit
    def bind_plan(self, plan):
        """Persistent buffers of the traced context commit of a verify plan (w, T): rows r = s*T + j of plan.out_aux
        are written at P_s + j (plan.cur_pos[0] = P_s, plan.page_table). Allocate BEFORE any capture."""
        w, T, R = plan.w, plan.T, plan.R
        assert self.NKV * T <= tpc.TILE_SIZE, f"num_kv_heads * T = {self.NKV * T} > 32 (paged_update_cache)"
        c = _CommitBufs()
        cos0, sin0 = self._rope_rows(torch.zeros(R, dtype=torch.int32))
        c.cos = self._up(cos0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        c.sin = self._up(sin0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        c.spread = [self._up(m, ttnn.bfloat16, ttnn.TILE_LAYOUT) for m in self.spread_matrices(w, T, R, self.NKV)]
        c.tmp_aux = self._up(torch.zeros(1, 1, R, 5 * self.dim, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._plans[id(plan)] = c

    def _commit_body(self, plan, aux):
        c = self._plans[id(plan)]
        w, T = plan.w, plan.T
        kvs = self.project_kv(aux, c.cos, c.sin)
        for (k_cache, v_cache), (ks, vs) in zip(self.kv_caches, kvs):
            self._spread_update(k_cache, ks, c.spread, w, plan.cur_pos[0], plan.page_table, T)
            self._spread_update(v_cache, vs, c.spread, w, plan.cur_pos[0], plan.page_table, T)

    def _upload_commit(self, plan, positions):
        c = self._plans[id(plan)]
        rows = vg.row_positions([max(0, int(p)) for p in positions], plan.T, plan.R)
        cos_t, sin_t = self._rope_rows(rows)
        self._dma(cos_t, c.cos, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._dma(sin_t, c.sin, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    def compile_commit(self, plan):
        """Eager commit on the compile-time aux stand-in (writes garbage KV at the plan's current cur_pos rows)."""
        t0 = time.perf_counter()
        c = self._plans[id(plan)]
        self._upload_commit(plan, [8] * plan.w)
        self._commit_body(plan, c.tmp_aux)
        self._sync()
        logger.info(f"[dflash2] commit ({plan.w},T={plan.T}) compiled in {time.perf_counter() - t0:.1f}s")

    def capture_commit(self, plan):
        """Capture the commit trace reading plan.out_aux (the verify trace's output: capture the verify step first)."""
        c = self._plans[id(plan)]
        assert c.trace_id is None
        aux = getattr(plan, "out_aux", None)
        assert aux is not None, "plan.out_aux missing: capture a verify step that keeps the aux rows first"
        assert tuple(aux.shape) == (1, 1, plan.R, 5 * self.dim), tuple(aux.shape)
        self._upload_commit(plan, [8] * plan.w)
        self._sync()
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._commit_body(plan, aux)
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self._sync()
        c.trace_id = tid

    def commit(self, plan, positions):
        """After a verify step at positions P_s (the step's own positions, BEFORE the accept advance): write the aux
        rows of every grid row (s, j) at P_s + j into the drafter KV. Rows past a_s are overwritten by the next block /
        commit before they are ever attended (the block SDPA is bounded by its cur_pos)."""
        t0 = time.perf_counter()
        c = self._plans[id(plan)]
        self._upload_commit(plan, positions)
        if c.trace_id is None:
            self._commit_body(plan, plan.out_aux)
        else:
            ttnn.execute_trace(self.mesh, c.trace_id, cq_id=0, blocking=False)
        self._sync()
        self.stats["commit_steps"] += 1
        self.stats["commit_wall"] += time.perf_counter() - t0

    # ------------------------------------------------------------------------------------------ draft step
    def build_step_buffers(self, widths, page_tables):
        pt = page_tables if isinstance(page_tables, torch.Tensor) else torch.as_tensor(page_tables)
        pt = pt.to(torch.int32)
        if self.page_tables is None:
            self.page_tables = pt
        for w in sorted(set(int(v) for v in widths)):
            if w in self.sb:
                continue
            R = w * self.B
            b = _StepBufs()
            b.tok = self._up(torch.zeros(1, R, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            cos0, sin0 = self._rope_rows(torch.zeros(R, dtype=torch.int32))
            b.cos = self._up(cos0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            b.sin = self._up(sin0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            b.pos0 = self._up(torch.zeros(w, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            b.pt = self._up(pt[:w].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            b.pt_host = pt[:w].clone()
            n_split = -(-R // self.n_cores)
            per = -(-R // n_split)
            pt_rows = pt[:w].repeat_interleave(self.B, dim=0).contiguous()
            for cidx in range(n_split):
                a, e = cidx * per, min(R, (cidx + 1) * per)
                if a >= e:
                    break
                b.chunks.append(
                    (
                        a,
                        e,
                        self._up(pt_rows[a:e].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                        self._up(torch.zeros(e - a, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    )
                )
            b.shift = self._up(self.shift_matrix(w, self.B, R), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            b.spread = [
                self._up(m, ttnn.bfloat16, ttnn.TILE_LAYOUT) for m in self.spread_matrices(w, self.B, R, self.NKV)
            ]
            self.sb[w] = b

    def set_page_table(self, w, page_table):
        b = self.sb[w]
        pt = (page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)).to(torch.int32)
        assert tuple(pt.shape) == tuple(b.pt_host.shape)
        if not torch.equal(pt, b.pt_host):
            b.pt_host = pt.clone()
            self._dma(pt.contiguous(), b.pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            rows = pt.repeat_interleave(self.B, dim=0)
            for a, e, pt_c, _ in b.chunks:
                self._dma(rows[a:e].contiguous(), pt_c, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def _upload_step(self, w, anchors, positions):
        """anchors[s] = the row-0 token (t'_s), positions[s] = P_s (the block start). positions[s] = -1: padding user
        (token 0 at KV position -1: the update and the SDPA skip it)."""
        b, B = self.sb[w], self.B
        R = w * B
        toks = torch.zeros(1, R, dtype=torch.int32)
        rows = torch.zeros(R, dtype=torch.int32)
        cur = torch.zeros(R, dtype=torch.int32)
        pos0 = torch.zeros(w, dtype=torch.int32)
        for s in range(w):
            p = int(positions[s])
            if p < 0:
                pos0[s] = -1
                cur[s * B : (s + 1) * B] = -1
                continue
            toks[0, s * B] = int(anchors[s])
            toks[0, s * B + 1 : (s + 1) * B] = self.cfg.mask_token_id
            rows[s * B : (s + 1) * B] = torch.arange(p, p + B)
            cur[s * B : (s + 1) * B] = torch.arange(p, p + B) if self.causal else (p + B - 1)
            pos0[s] = p
        self._dma(toks, b.tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._dma(pos0, b.pos0, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cos_t, sin_t = self._rope_rows(rows)
        self._dma(cos_t, b.cos, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._dma(sin_t, b.sin, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        for a, e, _, pos_c in b.chunks:
            self._dma(cur[a:e].contiguous(), pos_c, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def step_forward(self, w):
        """The traced block forward for w users (R = 8w rows). Returns (topk_vals, topk_idx, hp, logits|None)."""
        m, b = self.model, self.sb[w]
        L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
        R, NH, NKV, HD = w * self.B, self.NH, self.NKV, self.HD
        x_e = m.embd(b.tok)  # [1,R,dim/TP]
        x_e = ttnn.reshape(x_e, (1, 1, R, x_e.shape[-1]))
        x_e = ttnn.to_memory_config(x_e, DRAM)
        x = self._all_gather(x_e)  # replicated residual [1,1,R,dim] DRAM
        for li, L in enumerate(self.layers):
            # --- attention sub-layer ---
            h = self._rms(x, L["in_ln"])
            dyn = self._mm(h, L["kp_a"])  # [1,1,R,4*groups]
            hc = self._conv(h, dyn, L["base_a"], 0, b.shift)
            ttnn.deallocate(h)
            qkv = self._mm(hc, L["wqkv"], out=DRAM)  # [1,1,R,(NH+2NKV)*HD]
            ttnn.deallocate(hc)
            q_flat = ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, R, NH * HD), memory_config=L1)
            q_rm = ttnn.to_layout(q_flat, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(q_flat)
            q_rm4 = ttnn.reshape(q_rm, (1, R, NH, HD))
            q = ttnn.to_layout(q_rm4, ttnn.TILE_LAYOUT)
            ttnn.deallocate(q_rm4)
            if q_rm4 is not q_rm:
                ttnn.deallocate(q_rm)
            q = ttnn.multiply(ttnn.rms_norm(q, epsilon=self.eps, memory_config=L1), L["q_norm"], memory_config=L1)
            q = apply_partial_rope_decode(q, b.cos, b.sin, NH, R, HD)  # [1,R,NH,HD] DRAM
            ks = self._kv_heads(qkv, NH * HD, L, b.cos, b.sin, True)
            vs = self._kv_heads(qkv, NH * HD + NKV * HD, L, b.cos, b.sin, False)
            ttnn.deallocate(qkv)
            k_cache, v_cache = self.kv_caches[li]
            self._spread_update(k_cache, ks, b.spread, w, b.pos0, b.pt, self.B)
            self._spread_update(v_cache, vs, b.spread, w, b.pos0, b.pt, self.B)
            outs = []
            for a, e, pt_c, pos_c in b.chunks:
                q_c = q if (a == 0 and e == R) else ttnn.slice(q, (0, a, 0, 0), (1, e, NH, HD))
                outs.append(
                    ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        q_c,
                        k_cache,
                        v_cache,
                        page_table_tensor=pt_c,
                        cur_pos_tensor=pos_c,
                        scale=HD**-0.5,
                        program_config=self.sdpa_cfg,
                        memory_config=L1,
                    )
                )
                if q_c is not q:
                    ttnn.deallocate(q_c)
            ttnn.deallocate(q)
            attn = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=1, memory_config=L1)
            if len(outs) > 1:
                for o in outs:
                    ttnn.deallocate(o)
            a_rm = ttnn.to_layout(attn, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(attn)
            a_rm2 = ttnn.reshape(a_rm, (1, 1, R, NH * HD))
            a_flat = ttnn.to_layout(a_rm2, ttnn.TILE_LAYOUT, memory_config=L1)
            ttnn.deallocate(a_rm2)
            if a_rm2 is not a_rm:
                ttnn.deallocate(a_rm)
            o_part = self._mm(a_flat, L["wo"], out=DRAM)  # partial [1,1,R,dim]
            ttnn.deallocate(a_flat)
            o = self._all_reduce(o_part)
            oc = self._conv(o, dyn, L["base_a"], 1, b.shift)
            ttnn.deallocate(o)
            ttnn.deallocate(dyn)
            x2 = ttnn.add(x, oc, memory_config=DRAM)
            ttnn.deallocate(x)
            ttnn.deallocate(oc)
            x = x2
            # --- MLP sub-layer ---
            h = self._rms(x, L["post_ln"])
            dyn = self._mm(h, L["kp_m"])
            hc = self._conv(h, dyn, L["base_m"], 0, b.shift)
            ttnn.deallocate(h)
            g = self._mm(hc, L["w1"], act=ttnn.UnaryOpType.SILU)
            u = self._mm(hc, L["w3"])
            ttnn.deallocate(hc)
            gu = ttnn.multiply(g, u, memory_config=L1)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            d_part = self._mm(gu, L["w2"], out=DRAM)
            ttnn.deallocate(gu)
            d = self._all_reduce(d_part)
            dc = self._conv(d, dyn, L["base_m"], 1, b.shift)
            ttnn.deallocate(d)
            ttnn.deallocate(dyn)
            x2 = ttnn.add(x, dc, memory_config=DRAM)
            ttnn.deallocate(x)
            ttnn.deallocate(dc)
            x = x2
        hn = self._rms(x, self.final_norm, out=DRAM)  # [1,1,R,dim] replicated
        ttnn.deallocate(x)
        hp = ttnn.linear(hn, self.w_hp, compute_kernel_config=_EXACT_MM, memory_config=DRAM)  # [1,1,R,rank]
        logits = ttnn.linear(hn, m.lm_head_weight)  # vocab-sharded [1,1,R,V/TP]
        ttnn.deallocate(hn)
        V = logits.shape[-1]
        Vp = 1 << (V - 1).bit_length()  # 62080 -> 65536: two power-of-two halves for the multi-core top-k
        padded = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, Vp - V)], value=-3.0e38)
        if not self.keep_logits:
            ttnn.deallocate(logits)
            logits = None
        half = Vp // 2
        vals, idxs = [], []
        for hh in range(2):
            part = ttnn.slice(padded, (0, 0, 0, hh * half), (1, 1, R, (hh + 1) * half), memory_config=DRAM)
            v_t, i_t = ttnn.topk(part, k=self.k_local, dim=-1)
            ttnn.deallocate(part)
            vals.append(v_t)
            idxs.append(i_t)
        ttnn.deallocate(padded)
        return vals, idxs, hp, logits

    def compile_step(self, w):
        t0 = time.perf_counter()
        self._upload_step(w, [1] * w, [8] * w)
        outs = self.step_forward(w)
        self._sync()
        self._free_outs(outs)
        logger.info(f"[dflash2] draft step w={w} (R={w * self.B}) compiled in {time.perf_counter() - t0:.1f}s")

    @staticmethod
    def _free_outs(outs):
        vals, idxs, hp, logits = outs
        for t in list(vals) + list(idxs) + [hp, logits]:
            if t is not None:
                ttnn.deallocate(t)

    def capture_step(self, w):
        b = self.sb[w]
        assert b.trace_id is None
        self._upload_step(w, [1] * w, [8] * w)
        self._sync()
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        b.out_vals, b.out_idx, b.out_hp, b.out_logits = self.step_forward(w)
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self._sync()
        b.trace_id = tid

    def release(self):
        for b in self.sb.values():
            if b.trace_id is not None:
                ttnn.release_trace(self.mesh, b.trace_id)
                b.trace_id = None
        for c in self._plans.values():
            if c.trace_id is not None:
                ttnn.release_trace(self.mesh, c.trace_id)
                c.trace_id = None

    def _read_candidates(self, w, vals, idxs, hp):
        """Device top-k readback -> per user per draft row: (cands [7,16] long, unary [7,16] float, hp [7,rank])."""
        comp0 = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        R, B = w * self.B, self.B
        v_all, i_all = [], []
        half = (1 << (self.per_shard - 1).bit_length()) // 2
        for hh in range(2):
            v = ttnn.to_torch(vals[hh], mesh_composer=comp0).float().reshape(self.nd, -1, self.k_local)[:, :R]
            i = ttnn.to_torch(idxs[hh], mesh_composer=comp0).to(torch.int64).reshape(self.nd, -1, self.k_local)[:, :R]
            v_all.append(v)
            i_all.append(i + hh * half)
        v_all = torch.cat(v_all, dim=-1)  # [nd, R, 2k]
        i_all = torch.cat(i_all, dim=-1) + (torch.arange(self.nd) * self.per_shard).reshape(self.nd, 1, 1)
        v_all = v_all.permute(1, 0, 2).reshape(R, -1)  # [R, nd*2k]
        i_all = i_all.permute(1, 0, 2).reshape(R, -1)
        top = torch.topk(v_all, self.cfg.selector_top_k, dim=-1)
        cands = i_all.gather(-1, top.indices)  # [R, 16]
        unary = top.values
        hp_t = ttnn.to_torch(ttnn.get_device_tensors(hp)[0]).float().reshape(-1, self.cfg.selector_rank)[:R]
        out = []
        for s in range(w):
            r0 = s * B + 1
            out.append((cands[r0 : r0 + B - 1], unary[r0 : r0 + B - 1], hp_t[r0 : r0 + B - 1]))
        return out

    def draft(self, w, anchors, positions, pad=None, observer=None):
        """One traced block draft for w users: anchors[s] = t'_s (row-0 token), positions[s] = P_s. Returns
        (drafts [w][7], candidates [w][7,16] long, unary [w][7,16] float). pad[s]: padding user (drafts zeros)."""
        t0 = time.perf_counter()
        b = self.sb[w]
        pad = [False] * w if pad is None else [bool(p) for p in pad]
        pos = [-1 if pad[s] else int(positions[s]) for s in range(w)]
        self._upload_step(w, anchors, pos)
        if b.trace_id is None:
            outs = self.step_forward(w)
            self._sync()
            vals, idxs, hp, logits = outs
        else:
            ttnn.execute_trace(self.mesh, b.trace_id, cq_id=0, blocking=False)
            self._sync()
            vals, idxs, hp, logits = b.out_vals, b.out_idx, b.out_hp, b.out_logits
        t1 = time.perf_counter()
        per_user = self._read_candidates(w, vals, idxs, hp)
        if observer is not None:
            observer(w, anchors, pos, per_user, logits)
        if b.trace_id is None:
            self._free_outs((vals, idxs, hp, logits))
        drafts, cands_out, unary_out = [], [], []
        for s in range(w):
            cands, unary, hp_s = per_user[s]
            if pad[s]:
                drafts.append([0] * (self.B - 1))
            else:
                path, _ = selector_walk(hp_s, cands, unary, int(anchors[s]), self.sel_pred, self.sel_succ)
                drafts.append(path)
            cands_out.append(cands)
            unary_out.append(unary)
        self.stats["draft_steps"] += 1
        self.stats["draft_wall"] += time.perf_counter() - t0
        self.stats["select_wall"] += time.perf_counter() - t1
        return drafts, cands_out, unary_out

    def read_logits(self, w, logits=None):
        """Full draft logits [R, V] float of the last step (keep_logits; the trace's output or the eager tensor)."""
        lg = logits if logits is not None else self.sb[w].out_logits
        assert lg is not None
        out = ttnn.to_torch(lg, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3)).float()
        return out.reshape(-1, out.shape[-1])[: w * self.B].clone()

    def time_step_replays(self, w, n=50):
        """Traced draft-step wall (upload + replay + top-k/hp readback + host selector), median / min ms."""
        ms = []
        for i in range(n):
            t0 = time.perf_counter()
            self.draft(w, [(100 + i + s) % 1000 for s in range(w)], [16 + 8 * i + s for s in range(w)])
            ms.append(1e3 * (time.perf_counter() - t0))
        ms.sort()
        return ms[len(ms) // 2], ms[0]

    def time_commit_replays(self, plan, n=50):
        ms = []
        for i in range(n):
            t0 = time.perf_counter()
            self.commit(plan, [16 + 8 * i + s for s in range(plan.w)])
            ms.append(1e3 * (time.perf_counter() - t0))
        ms.sort()
        return ms[len(ms) // 2], ms[0]


class DFlash2ContextProjector:
    """Host-facing context projection through the device weights: ``project(aux [N, 5*dim], positions [N])`` ->
    per layer (K [N, n_kv_heads, HD] roped + normed bf16 (heads gathered over the mesh), V [N, n_kv_heads, HD]).
    head_dim = 128 (config.json ``head_dim``), n_kv_heads = 8, RoPE theta 1e7 (drafter config), full-dim rotation.
    The same device ops feed ``DFlash2Drafter.write_context`` / ``commit`` (which keep the K/V on device)."""

    def __init__(self, drafter):
        self.d = drafter
        self.head_dim = drafter.HD
        self.n_kv_heads = drafter.cfg.n_kv_heads

    def project(self, aux, positions):
        """aux: HOST [N, 5*dim] rows (uploaded replicated) -> per layer (K, V) host bf16 [N, n_kv_heads, HD]."""
        d = self.d
        aux_t = torch.as_tensor(aux).to(torch.bfloat16).reshape(1, 1, -1, 5 * d.dim)
        aux_tt = d._up(aux_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        out = self.project_device(aux_tt, positions)
        ttnn.deallocate(aux_tt)
        return out

    def project_device(self, aux_rep, positions):
        """aux_rep: the REPLICATED device tensor [1,1,N,5*dim] bf16 (the prefill hook's all-gathered rows, left
        alone) -> per layer (K, V) host bf16 [N, n_kv_heads, HD] in global head order (tt/aux_hidden.py fast path)."""
        d = self.d
        N = int(aux_rep.shape[-2])
        pos = torch.as_tensor(positions).reshape(-1)[:N]
        cos_t, sin_t = d._rope_rows(pos)
        cos_tt = d._up(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        sin_tt = d._up(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        kvs = d.project_kv(aux_rep, cos_tt, sin_tt)
        d._sync()
        comp = ttnn.ConcatMeshToTensor(d.mesh, dim=1)
        out = []
        for ks, vs in kvs:
            res = []
            for heads in (ks, vs):
                hs = [ttnn.reshape(h, (1, 1, N, d.HD)) for h in heads]
                cat = ttnn.concat(hs, dim=1) if d.NKV > 1 else hs[0]  # [1,NKV,N,HD] per device
                t = ttnn.to_torch(cat, mesh_composer=comp)  # [1, n_kv_heads, N, HD]
                res.append(t.reshape(self.n_kv_heads, N, d.HD).permute(1, 0, 2).contiguous().to(torch.bfloat16))
                if cat is not hs[0]:
                    ttnn.deallocate(cat)
                for h in heads:
                    ttnn.deallocate(h)
            out.append(tuple(res))
        for t in (cos_tt, sin_tt):
            ttnn.deallocate(t)
        return out
