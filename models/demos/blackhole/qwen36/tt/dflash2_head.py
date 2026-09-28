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
import os

import torch

DFLASH2_REPO = "z-lab/Qwen3.8-27B-DFlash2"
_SDPA_PT_BLOCKS = 32


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
        pred = int(anchor)
        path, scores_all = [], []
        for t in range(hidden.shape[0]):
            a = self.sel_pred[pred].float() * hp[t]
            b = self.sel_succ[candidates[t]].float()  # [k, rank]
            scores = unary[t] + b @ a
            idx = int(torch.argmax(scores))
            pred = int(candidates[t, idx])
            path.append(pred)
            scores_all.append(scores)
        return path, candidates, torch.stack(scores_all)

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
