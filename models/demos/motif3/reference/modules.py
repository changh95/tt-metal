# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch Motif-3 modules (golden reference, CPU, no ttnn).

Numerics follow the official HF ``modeling_motif.py`` op-for-op (same cast points, same op order) so that a
model built in bf16 reproduces HF's bf16 rounding and a model built in fp32 is the "ideal" golden. Where HF is
known to be stale w.r.t. the fork/training it would be called out here; for the shipped HF file there are no
semantic differences (see README.md "Semantics decisions"), only the precision knobs in ``MotifArgs``
(``q_path_fp32``, ``mhc_mix_fp32``).

Parameter names mirror the HF checkpoint (``self_attn.wq_a.weight``, ``moe.experts.gate_up_proj``, ...) except the
mHC projections, which are stored merged as ``proj_merged.weight`` = cat(proj_pre, proj_post, proj_res) rows
(fork/training layout, ``weights.py`` maps names).

Every ``forward`` accepts an optional ``tap(name, tensor)`` callable used by ``golden.py`` to capture
intermediates; it is never needed for normal use.

Shapes: B batch, S new tokens, T keys, D hidden, E mHC streams (4), H q heads (80), Hkv kv heads (16),
G differential groups (16), gr signal heads per group (4), n_sig = G * gr (64).
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .cache import LatentKVCache, MotifKVCache
from .config import MotifArgs
from .rope import apply_rope, inv_freq_for_layer, rope_cos_sin

Tap = Optional[Callable[[str, torch.Tensor], None]]


def _sub(tap: Tap, prefix: str) -> Tap:
    if tap is None:
        return None
    return lambda name, t: tap(f"{prefix}.{name}", t)


def _rec(tap: Tap, name: str, t: torch.Tensor) -> None:
    if tap is not None:
        tap(name, t)


# =================================================================================================
# Norms and activations
# =================================================================================================
class RMSNorm(nn.Module):
    """HF ``MotifRMSNorm``: fp32 statistics, cast back to the INPUT dtype, then ``weight * x``.

    bf16 in -> bf16 out (two roundings: normalized x, then the gamma product). fp32 in (the q path) -> fp32 out
    because ``bf16 weight * fp32`` promotes.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        xf = x.to(torch.float32)
        variance = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.eps)
        return self.weight * xf.to(in_dtype)


def _poly_rms(z: torch.Tensor, eps: float) -> torch.Tensor:
    """PolyNorm's per-row normalization ``z / sqrt(mean(z^2) + eps)`` (a reduction over the intermediate dim)."""
    return z / torch.sqrt(z.pow(2).mean(-1, keepdim=True) + eps)


class PolyNorm(nn.Module):
    """Dense / shared-expert PolyNorm (HF ``PolyNormTorch``; training ``FusedMulPolyNorm``).

    ``poly(g) = s(w0) N(g^3) + s(w1) N(g^2) + s(w2) N(g) + b`` with ``s`` = sigmoid, fp32 math;
    ``forward_mul`` returns ``dtype(poly(g) * up)`` (the multiply is in fp32, one downcast). NO bias clamp here:
    training's FeedForward never clamps (fork motif_docs.md "Fixed discrepancies" #1).
    """

    def __init__(self, eps: float = 1e-6, sigmoid_weight: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3) / 3)
        self.bias = nn.Parameter(torch.zeros(1))
        self.eps = eps
        self.sigmoid_weight = sigmoid_weight

    def coefficients(self) -> torch.Tensor:
        w = self.weight.float()
        return torch.sigmoid(w) if self.sigmoid_weight else w

    def poly(self, xf: torch.Tensor) -> torch.Tensor:
        w = self.coefficients()
        return (
            w[0] * _poly_rms(xf**3, self.eps)
            + w[1] * _poly_rms(xf**2, self.eps)
            + w[2] * _poly_rms(xf, self.eps)
            + self.bias.float()
        )

    def forward_mul(self, x: torch.Tensor, mul: torch.Tensor) -> torch.Tensor:
        return (self.poly(x.float()) * mul.float()).to(x.dtype)


class GroupedPolyNorm(nn.Module):
    """Routed-expert PolyNorm (HF ``GroupedPolyNorm``; training ``GroupedExpertsPolyNorm``).

    Per-expert ``weight [E, 3]`` / ``bias [E, 1]``. Bias clamped to +-polynorm_bias_clamp (0.5), gate/up/result
    clamped to +-hidden_clamp (1e6, a no-op in practice), result scaled by polynorm_output_scale (0.5); fp32 math,
    a single cast to the activation dtype at the end. (HF scales before the cast, the fork after; for a
    power-of-two scale the results are identical.)
    """

    def __init__(
        self,
        num_experts: int,
        eps: float = 1e-6,
        sigmoid_weight: bool = True,
        bias_clamp: Optional[float] = None,
        output_scale: float = 1.0,
        hidden_clamp: Optional[float] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_experts, 3) / 3)
        self.bias = nn.Parameter(torch.zeros(num_experts, 1))
        self.eps = eps
        self.sigmoid_weight = sigmoid_weight
        self.bias_clamp = bias_clamp
        self.output_scale = output_scale
        self.hidden_clamp = hidden_clamp

    def coefficients(self, expert_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(sigmoid(weight[e]) [3] fp32, clamped bias[e] [1] fp32)."""
        w = self.weight[expert_idx].float()
        if self.sigmoid_weight:
            w = torch.sigmoid(w)
        b = self.bias[expert_idx].float()
        if self.bias_clamp is not None:
            b = b.clamp(-self.bias_clamp, self.bias_clamp)
        return w, b

    def forward_single(self, x: torch.Tensor, mul: torch.Tensor, expert_idx: int) -> torch.Tensor:
        out_dtype = x.dtype
        w, b = self.coefficients(expert_idx)
        xf = x.float()
        mf = mul.float()
        if self.hidden_clamp is not None:
            xf = xf.clamp(-self.hidden_clamp, self.hidden_clamp)
            mf = mf.clamp(-self.hidden_clamp, self.hidden_clamp)
        poly = (
            w[0] * _poly_rms(xf**3, self.eps) + w[1] * _poly_rms(xf**2, self.eps) + w[2] * _poly_rms(xf, self.eps) + b
        )
        result = poly * mf
        if self.hidden_clamp is not None:
            result = result.clamp(-self.hidden_clamp, self.hidden_clamp)
        result = result * self.output_scale
        return result.to(out_dtype)


# =================================================================================================
# mHC
# =================================================================================================
def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Sinkhorn-Knopp in fp32: ``exp(clamp(M, -20, 20))``, then ``iters`` x (row-normalize, column-normalize),
    sums clamped at 1e-8 (HF modeling_motif.py:226-233 == training mhc.py:191-216). With peaked logits 20
    iterations leave rows only approximately normalized (columns are exact), so count and order are spec."""
    m = logits.float().clamp(-20.0, 20.0).exp()
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        m = m / m.sum(dim=-2, keepdim=True).clamp(min=1e-8)
    return m


class MHCLayer(nn.Module):
    """Manifold-constrained Hyper-Connections block (one per sublayer).

    ``x [B, S, E, D]`` -> ``h_pre [B,S,E]``, ``h_post [B,S,E]``, ``h_res [B,S,E,E]`` (all fp32):
      n      = RMSNorm_{eps=1e-6}(x.reshape(B, S, E*D))       (own gamma)
      p      = n @ proj_merged^T   rows = [pre(E) | post(E) | res(E*E, row-major i*E + j)]
      h_pre  = sigmoid(clamp(alpha_pre  * p_pre  + bias_pre,  -10, 10))
      h_post = coeff * sigmoid(clamp(alpha_post * p_post + bias_post, -10, 10))     (coeff = 1.0)
      h_res  = sinkhorn(alpha_res * p_res + bias_res, 20)
    ``pre(x)`` = sum_i h_pre[i] x[i] (fp32 accumulate, one cast); ``post(x, out)`` = h_res @ x + h_post (x) out
    in fp32 with a single cast to the residual dtype (HF modeling_motif.py:1224-1226, :1242-1244).
    """

    def __init__(
        self,
        n_streams: int,
        dim: int,
        sinkhorn_iters: int = 20,
        h_post_coeff: float = 1.0,
        rms_eps: float = 1e-6,
        mix_fp32: bool = False,
    ):
        super().__init__()
        E = n_streams
        self.n_streams = E
        self.dim = dim
        self.sinkhorn_iters = sinkhorn_iters
        self.h_post_coeff = float(h_post_coeff)
        self.mix_fp32 = mix_fp32
        self.proj_merged = nn.Linear(E * dim, E * E + 2 * E, bias=False)
        self.rms_norm = RMSNorm(E * dim, eps=rms_eps)
        self.bias_pre = nn.Parameter(torch.zeros(E))
        self.bias_post = nn.Parameter(torch.zeros(E))
        self.bias_res = nn.Parameter(torch.zeros(E, E))
        self.alpha_pre = nn.Parameter(torch.zeros(1))
        self.alpha_post = nn.Parameter(torch.zeros(1))
        self.alpha_res = nn.Parameter(torch.zeros(1))

    def folded_mix_weight(self) -> torch.Tensor:
        """fp32 ``proj_merged * rms_gamma`` (the fork's ``fn`` buffer; TT can use it with an fp32 inv-RMS)."""
        return self.proj_merged.weight.float() * self.rms_norm.weight.float()[None, :]

    def mixes(self, x: torch.Tensor) -> torch.Tensor:
        """Raw projections ``p [B, S, E*E + 2E]`` in fp32."""
        B, S, E, D = x.shape
        x_flat = x.reshape(B, S, E * D)
        if self.mix_fp32:
            # fork tilelang semantics: gamma folded into the projection, fp32 GEMM on the raw stream, fp32 RMS scale
            xf = x_flat.float()
            inv_rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.rms_norm.eps)
            return F.linear(xf, self.folded_mix_weight()) * inv_rms
        # HF semantics: n (activation dtype) @ W (activation dtype) -> activation dtype -> fp32. Three GEMMs on row
        # slices of the merged weight, i.e. exactly HF's proj_pre / proj_post / proj_res.
        n = self.rms_norm(x_flat)
        W = self.proj_merged.weight
        return torch.cat(
            [F.linear(n, W[:E]).float(), F.linear(n, W[E : 2 * E]).float(), F.linear(n, W[2 * E :]).float()], dim=-1
        )

    def forward(self, x: torch.Tensor, tap: Tap = None):
        B, S, E, _ = x.shape
        p = self.mixes(x)
        p_pre, p_post, p_res = p[..., :E], p[..., E : 2 * E], p[..., 2 * E :].reshape(B, S, E, E)
        h_pre = torch.sigmoid((self.alpha_pre * p_pre + self.bias_pre).clamp(-10.0, 10.0))
        h_post = self.h_post_coeff * torch.sigmoid((self.alpha_post * p_post + self.bias_post).clamp(-10.0, 10.0))
        h_res = sinkhorn(self.alpha_res * p_res + self.bias_res, self.sinkhorn_iters)
        _rec(tap, "mixes", p)
        _rec(tap, "h_pre", h_pre)
        _rec(tap, "h_post", h_post)
        _rec(tap, "h_res", h_res)
        return h_pre, h_post, h_res

    @staticmethod
    def pre(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
        """``[B,S,E,D] -> [B,S,D]``: fp32 weighted sum over streams, cast to the stream dtype."""
        return (x * h_pre.unsqueeze(-1)).sum(dim=2).to(x.dtype)

    @staticmethod
    def post(x: torch.Tensor, out: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor) -> torch.Tensor:
        """``x' = h_res @ x + h_post (x) out`` in fp32, one cast to the stream dtype."""
        res = torch.einsum("bsij,bsjd->bsid", h_res, x.float())
        post = h_post.unsqueeze(-1) * out.float().unsqueeze(2)
        return (res + post).to(x.dtype)


# =================================================================================================
# Attention
# =================================================================================================
def attention_mask(q_pos: torch.Tensor, k_pos: torch.Tensor, window: Optional[int]) -> torch.Tensor:
    """Boolean ``[B, 1, S, T]`` mask from absolute positions (``q_pos [B, S]``, ``k_pos [B, T]`` or ``[T]``).

    Key t is visible to query s iff ``k_pos <= q_pos`` and, for SWA (``window`` = keys INCLUDING the current
    one, 129 for Motif-3), ``k_pos > q_pos - window`` i.e. ``k_pos >= q_pos - 128``. This is flash-attn
    ``causal=True, window_size=(window - 1, *)`` with bottom-right alignment.
    """
    if k_pos.dim() == 1:
        k_pos = k_pos[None, :]
    qp = q_pos[:, :, None]
    kp = k_pos[:, None, :]
    allowed = kp <= qp
    if window is not None:
        allowed = allowed & (kp > qp - window)
    return allowed[:, None]


def sdpa_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    out_dtype: torch.dtype,
    q_chunk: int = 2048,
) -> torch.Tensor:
    """Exact attention core: ``softmax(scale * q k^T + mask) v`` with fp32 scores, softmax and PV.

    ``q [B, H, S, Dq]``; ``k [B, Hk, T, Dq]`` / ``v [B, Hk, T, Dv]`` with ``Hk`` in {H, 1} (broadcast);
    ``mask [B, 1, S, T]`` bool. Returns ``[B, S, H, Dv]`` cast once to ``out_dtype`` (like a flash-attn output).
    """
    kf = k.float().transpose(-1, -2)
    vf = v.float()
    outs = []
    for s0 in range(0, q.shape[2], q_chunk):
        s1 = min(s0 + q_chunk, q.shape[2])
        scores = (q[:, :, s0:s1].float() @ kf) * scale
        scores = scores.masked_fill(~mask[:, :, s0:s1], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        outs.append(p @ vf)
    out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
    return out.to(out_dtype).transpose(1, 2)


class GDLAttention(nn.Module):
    """Grouped Differential Latent Attention (HF ``MotifGDLAttention``, diff_v2, elementwise gate).

    Head layout: q head ``h`` belongs to differential group ``g = h // 5`` and uses KV head ``h // 5``; heads
    ``5g..5g+3`` are signal, ``5g+4`` is noise. Signal head ``s = 4g + j`` indexes ``lambda_proj``, the elementwise
    gate and the ``wo`` input columns: ``out_s = sigmoid(gate_s) * (O_{5g+j} - sigmoid(lambda_s) O_{5g+4})``.

    Two mathematically equivalent forms (``mode``):
      * ``"expanded"``: K = [W_UK c ; k_pe] (192), V = W_UV c (128), GQA 5 q heads per KV head (HF/FA2 form).
      * ``"absorbed"``: MQA over the latent: q_lat_h = W_UK,g(h)^T q_nope_h (512), keys [c ; k_pe] (576),
        values c (512); the differential combine happens in latent space, then W_UV,g is applied once per signal
        head (64 up-projections instead of 80). W_UV cannot be folded into wo because of the elementwise gate.
    Both read the same :class:`LatentKVCache` (c = kv_norm(latent) incl. gamma, roped k_pe), in any cache dtype.
    In fp32 the two forms agree to ~1e-5. In bf16 they round differently (same math); only "expanded" reproduces
    HF's bf16 rounding, which is why it is the default everywhere (forward, ``MotifGenerator``, goldens).
    """

    def __init__(self, args: MotifArgs, layer_idx: int, *, swa: Optional[bool] = None):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.is_swa = args.is_swa_layer(layer_idx) if swa is None else bool(swa)
        self.window = args.effective_sliding_window if self.is_swa else None
        self.scale = args.softmax_scale(layer_idx, swa=self.is_swa)
        self.uses_yarn = args.uses_yarn(layer_idx, swa=self.is_swa)
        self.rope_theta = args.rope_theta_for_layer(layer_idx, swa=self.is_swa)
        self.q_path_fp32 = args.q_path_fp32

        D, H, Hkv = args.hidden_size, args.num_attention_heads, args.num_key_value_heads
        self.n_heads, self.n_kv_heads = H, Hkv
        self.head_dim, self.rope_dim, self.nope_dim = args.head_dim, args.qk_rope_head_dim, args.qk_nope_head_dim
        self.v_dim, self.kv_rank = args.v_head_dim, args.kv_lora_rank
        self.n_groups, self.grouped_ratio, self.n_signal = args.num_noise_heads, args.grouped_ratio, args.n_signal_heads
        self.kv_group_size = args.kv_group_size

        self.wq_a = nn.Linear(D, args.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.wq_b = nn.Linear(args.q_lora_rank, H * self.head_dim, bias=False)
        self.wq_b_gate = (
            nn.Linear(args.q_lora_rank, self.n_signal * self.v_dim, bias=False)
            if args.elementwise_attn_output_gate
            else None
        )
        self.wkv_a = nn.Linear(D, self.kv_rank + self.rope_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_rank, eps=args.rms_norm_eps)
        self.wkv_b = nn.Linear(self.kv_rank, Hkv * (self.nope_dim + self.v_dim), bias=False)
        self.lambda_proj = nn.Linear(D, self.n_signal, bias=False)
        self.wo = nn.Linear(self.n_signal * self.v_dim, D, bias=False)

        # KV head serving each signal head s = g * gr + j (q head g * (gr + 1) + j); used by the absorbed form.
        # (explicit CPU device: models are built under torch.device("meta") when loading real checkpoints)
        signal_q_heads = torch.tensor(
            [g * (self.grouped_ratio + 1) + j for g in range(self.n_groups) for j in range(self.grouped_ratio)],
            device="cpu",
        )
        noise_q_heads = torch.tensor(
            [g * (self.grouped_ratio + 1) + self.grouped_ratio for g in range(self.n_groups)], device="cpu"
        )
        self._signal_kv_head = signal_q_heads // self.kv_group_size
        # The latent-space differential needs each group's signal and noise heads to share one KV head (true for
        # Motif-3: 5 q heads per KV head == 5 heads per differential group).
        self.latent_diff_ok = bool(
            torch.equal(
                self._signal_kv_head, (noise_q_heads // self.kv_group_size).repeat_interleave(self.grouped_ratio)
            )
        )

    # ---- helpers -------------------------------------------------------------------------------------
    def inv_freq(self) -> torch.Tensor:
        return inv_freq_for_layer(self.args, self.layer_idx, swa=self.is_swa)

    def w_uk_uv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-KV-head absorption matrices from ``wkv_b [Hkv * (nope + v), r]``: W_UK [Hkv, nope, r], W_UV [Hkv, v, r]."""
        w = self.wkv_b.weight.view(self.n_kv_heads, self.nope_dim + self.v_dim, self.kv_rank)
        return w[:, : self.nope_dim], w[:, self.nope_dim :]

    def project_q(self, x: torch.Tensor, tap: Tap = None):
        """q ``[B,S,H,hd]`` (activation dtype) and gate logits ``[B,S,n_sig,v]`` (HF modeling_motif.py:646-656).

        Taps ``q_latent`` = q_norm output as fed to ``wq_b`` (fp32 when ``q_path_fp32``; the gate sees its bf16 cast).
        """
        B, S, _ = x.shape
        dtype = x.dtype
        if self.q_path_fp32:
            cq = self.q_norm(F.linear(x.float(), self.wq_a.weight.float()))  # fp32 (bf16 gamma * fp32 -> fp32)
            q = F.linear(cq, self.wq_b.weight.float()).view(B, S, self.n_heads, self.head_dim).to(dtype)
            _rec(tap, "q_latent", cq)
            cq = cq.to(dtype)
        else:
            cq = self.q_norm(self.wq_a(x))
            q = self.wq_b(cq).view(B, S, self.n_heads, self.head_dim)
            _rec(tap, "q_latent", cq)
        gate = self.wq_b_gate(cq).view(B, S, self.n_signal, self.v_dim) if self.wq_b_gate is not None else None
        return q, gate

    def project_kv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Cache entries: ``c = kv_norm(c_raw) [B,S,r]`` and roped ``k_pe [B,S,rope]``."""
        kv = self.wkv_a(x)
        c_raw, k_pe = torch.split(kv, [self.kv_rank, self.rope_dim], dim=-1)
        k_pe = apply_rope(k_pe.unsqueeze(2), cos, sin).squeeze(2)
        c = self.kv_norm(c_raw.contiguous())
        return c, k_pe

    # ---- attention cores ---------------------------------------------------------------------------
    def _heads_expanded(self, q_nope, q_pe, c_keys, kpe_keys, q_pos, k_pos, out_dtype):
        """Per-head outputs ``[B,S,H,v]`` via expanded GQA attention."""
        B, T, _ = c_keys.shape
        kv = self.wkv_b(c_keys).view(B, T, self.n_kv_heads, self.nope_dim + self.v_dim)
        k_nope, v = torch.split(kv, [self.nope_dim, self.v_dim], dim=-1)
        k = torch.cat([k_nope, kpe_keys.unsqueeze(2).expand(-1, -1, self.n_kv_heads, -1)], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)  # [B,H,S,hd]
        k = k.transpose(1, 2).repeat_interleave(self.kv_group_size, dim=1)  # [B,H,T,hd]
        v = v.transpose(1, 2).repeat_interleave(self.kv_group_size, dim=1)  # [B,H,T,v]
        return sdpa_fp32(q, k, v, attention_mask(q_pos, k_pos, self.window), self.scale, out_dtype)

    def _latent_heads_absorbed(self, q_nope, q_pe, c_keys, kpe_keys, q_pos, k_pos, out_dtype):
        """Per-head LATENT outputs ``o_lat [B,S,H,r] = sum_t p_{h,t} c_t`` via MQA over the latent cache."""
        w_uk, _ = self.w_uk_uv()
        w_uk_h = w_uk.repeat_interleave(self.kv_group_size, dim=0)  # [H, nope, r]
        q_lat = torch.einsum("bshn,hnr->bshr", q_nope, w_uk_h)  # [B,S,H,r]
        q = torch.cat([q_lat, q_pe], dim=-1).transpose(1, 2)  # [B,H,S,r+rope]
        k = torch.cat([c_keys, kpe_keys], dim=-1).unsqueeze(1)  # [B,1,T,r+rope]
        v = c_keys.unsqueeze(1)  # [B,1,T,r]
        return q_lat, sdpa_fp32(q, k, v, attention_mask(q_pos, k_pos, self.window), self.scale, out_dtype)

    def _split_signal_noise(self, heads: torch.Tensor):
        """``[B,S,H,d] -> signal [B,S,n_sig,d]``, noise repeated to ``[B,S,n_sig,d]`` (HF modeling_motif.py:753-763)."""
        B, S, _, d = heads.shape
        grouped = heads.view(B, S, self.n_groups, self.grouped_ratio + 1, d)
        signal = grouped[:, :, :, : self.grouped_ratio, :].reshape(B, S, self.n_signal, d)
        noise = grouped[:, :, :, self.grouped_ratio :, :].reshape(B, S, self.n_groups, d)
        return signal, torch.repeat_interleave(noise, dim=2, repeats=self.grouped_ratio)

    # ---- forward ---------------------------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: Optional[LatentKVCache] = None,
        mode: str = "expanded",
        tap: Tap = None,
    ) -> torch.Tensor:
        """``x [B,S,D]`` (normalized attention input), ``positions [B,S]`` absolute -> ``[B,S,D]``.

        With ``cache`` the new latents are written at ``positions`` first and the queries attend over every cached
        slot up to the newest position (prefill, chunked prefill and decode are the same code path).
        """
        if mode not in ("expanded", "absorbed"):
            raise ValueError(f"unknown attention mode {mode!r}")
        B, S, _ = x.shape
        dtype = x.dtype
        q, gate = self.project_q(x, tap)
        q_nope, q_pe = torch.split(q, [self.nope_dim, self.rope_dim], dim=-1)
        cos, sin = rope_cos_sin(self.inv_freq(), positions, dtype)
        q_pe = apply_rope(q_pe, cos, sin)
        c, k_pe = self.project_kv(x, cos, sin)
        lam = self.lambda_proj(x)  # [B,S,n_sig]
        _rec(tap, "q_nope", q_nope)
        _rec(tap, "q_pe", q_pe)
        _rec(tap, "c_kv", c)
        _rec(tap, "k_pe", k_pe)
        _rec(tap, "lambda", lam)
        if gate is not None:
            _rec(tap, "gate", gate)

        if cache is not None:
            cache.update(c, k_pe, positions)
            c_keys, kpe_keys, k_pos = cache.keys(int(positions.max()) + 1)
            # The cache dtype is storage precision only: entries are rounded to it on write and read back in the
            # activation dtype, so both attention forms compute exactly as with a cache in the model dtype (an fp32
            # cache on a bf16 model is lossless; a bf16 cache on an fp32 model emulates a bf16 KV cache). Contiguous,
            # so the GEMMs see one memory layout whatever the cache dtype: with B > 1 a strided cache view takes
            # another fp32 summation path than the contiguous copy a dtype cast makes.
            c_keys, kpe_keys = c_keys.to(dtype).contiguous(), kpe_keys.to(dtype).contiguous()
        else:
            c_keys, kpe_keys, k_pos = c, k_pe, positions

        lam_scale = torch.sigmoid(lam.float()).to(dtype).unsqueeze(-1)  # [B,S,n_sig,1]
        if mode == "expanded":
            heads = self._heads_expanded(q_nope, q_pe, c_keys, kpe_keys, positions, k_pos, dtype)
            _rec(tap, "attn_heads", heads)
            signal, noise = self._split_signal_noise(heads)
            diff = signal - lam_scale * noise  # [B,S,n_sig,v]
        else:
            if not self.latent_diff_ok:
                raise NotImplementedError("latent-space differential requires each group to share one KV head")
            q_lat, o_lat = self._latent_heads_absorbed(q_nope, q_pe, c_keys, kpe_keys, positions, k_pos, dtype)
            _rec(tap, "q_lat", q_lat)
            _rec(tap, "attn_latent_heads", o_lat)
            signal, noise = self._split_signal_noise(o_lat)
            diff_lat = signal - lam_scale * noise  # [B,S,n_sig,r]
            _rec(tap, "diff_latent", diff_lat)
            _, w_uv = self.w_uk_uv()
            diff = torch.einsum("bsnr,nvr->bsnv", diff_lat, w_uv[self._signal_kv_head])  # [B,S,n_sig,v]
        _rec(tap, "diff", diff)
        if gate is not None:
            diff = diff * torch.sigmoid(gate)
        _rec(tap, "gated", diff)
        out = self.wo(diff.reshape(B, S, self.n_signal * self.v_dim))
        _rec(tap, "out", out)
        return out


# =================================================================================================
# FFN: dense MLP, router, MoE
# =================================================================================================
class MLP(nn.Module):
    """Dense PolyNorm MLP (layers 0-1, shared expert, MTP): HF ``MotifMLP`` (modeling_motif.py:473-501).

    ``down( dtype(poly(clamp(gate)) * clamp(up)) * output_scale )``; the scale is applied after the downcast.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        eps: float = 1e-6,
        sigmoid_weight: bool = True,
        hidden_clamp: Optional[float] = None,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = PolyNorm(eps=eps, sigmoid_weight=sigmoid_weight)
        self.hidden_clamp = hidden_clamp
        self.output_scale = float(output_scale)

    def forward(self, x: torch.Tensor, tap: Tap = None) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.hidden_clamp is not None:
            gate = gate.clamp(-self.hidden_clamp, self.hidden_clamp)
            up = up.clamp(-self.hidden_clamp, self.hidden_clamp)
        h = self.act_fn.forward_mul(gate, up)
        if self.output_scale != 1.0:
            h = h * self.output_scale
        out = self.down_proj(h)
        _rec(tap, "gate", gate)
        _rec(tap, "up", up)
        _rec(tap, "act", h)
        _rec(tap, "out", out)
        return out


class Router(nn.Module):
    """Token-choice top-k router (HF ``TokenChoiceTopKRouter``; fork FusedTopKBiasRouter).

    fp32 GEMM on fp32-upcast input and gate weight -> sigmoid (fp32) -> top-k on ``scores + expert_bias``
    (selection only) -> weights = UNBIASED scores of the selected experts -> renormalize (+1e-20) -> * route_scale.
    """

    def __init__(
        self, hidden_size: int, num_experts: int, top_k: int, score_func: str, route_norm: bool, route_scale: float
    ):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts, self.top_k = num_experts, top_k
        self.score_func, self.route_norm, self.route_scale = score_func, route_norm, route_scale

    def forward(self, x: torch.Tensor, expert_bias: Optional[torch.Tensor] = None, tap: Tap = None):
        """``x [N, D]`` -> ``(weights [N, k] fp32, indices [N, k] int64)``."""
        logits = F.linear(x.to(torch.float32), self.gate.weight.to(torch.float32))
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = F.softmax(logits, dim=1)
        if expert_bias is not None:
            _, indices = torch.topk(scores + expert_bias, k=self.top_k, dim=1)
            weights = scores.gather(dim=1, index=indices)
        else:
            weights, indices = torch.topk(scores, k=self.top_k, dim=1)
        if self.route_norm:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        _rec(tap, "logits", logits)
        _rec(tap, "scores", scores)
        _rec(tap, "indices", indices)
        _rec(tap, "weights", weights)
        return weights, indices


class RoutedExperts(nn.Module):
    """Routed experts with fused ``gate_up_proj [E, 2I, D]`` (rows 0..I-1 gate, I..2I-1 up) and
    ``down_proj [E, D, I]`` exactly as stored in the checkpoint.

    Weights are either module parameters or, for real checkpoints, fetched lazily per expert through
    ``expert_source(e) -> (gate_up [2I, D], down [D, I])`` (set by ``weights.py``; avoids loading 12 GB/layer).
    """

    def __init__(self, args: MotifArgs, layer_idx: int, materialize: bool = True):
        super().__init__()
        E, D, I = args.num_experts, args.hidden_size, args.moe_intermediate_size
        self.num_experts, self.hidden_size, self.intermediate_size = E, D, I
        if materialize:
            self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * I, D))
            self.down_proj = nn.Parameter(torch.empty(E, D, I))
        else:
            self.gate_up_proj = None
            self.down_proj = None
        self.expert_source: Optional[Callable[[int], Tuple[torch.Tensor, torch.Tensor]]] = None
        self.act_fn = GroupedPolyNorm(
            E,
            eps=args.polynorm_eps,
            sigmoid_weight=args.polynorm_sigmoid_weight,
            bias_clamp=args.polynorm_bias_clamp,
            output_scale=args.polynorm_output_scale_for_layer(layer_idx),
            hidden_clamp=args.hidden_clamp,
        )

    def expert_weights(self, e: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.gate_up_proj is not None:
            return self.gate_up_proj[e], self.down_proj[e]
        if self.expert_source is None:
            raise RuntimeError("RoutedExperts has neither materialized weights nor an expert_source")
        return self.expert_source(e)

    def expert_forward(self, x: torch.Tensor, e: int, tap: Tap = None) -> torch.Tensor:
        """One expert on ``x [n, D]`` -> ``[n, D]`` (activation dtype): bf16 GEMMs, fp32 PolyNorm, bf16 act."""
        gate_up_w, down_w = self.expert_weights(e)
        gate_up = x @ gate_up_w.T
        gate, up = gate_up.chunk(2, dim=-1)
        act = self.act_fn.forward_single(gate.contiguous(), up, e)
        y = act @ down_w.T
        _rec(tap, "gate", gate)
        _rec(tap, "up", up)
        _rec(tap, "act", act)
        _rec(tap, "out", y)
        return y

    def forward(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``x [N, D]``, ``indices/weights [N, k]`` -> fp32 ``[N, D]`` = sum_k w_k * expert_k(x).

        Same loop as HF ``MotifExperts.forward`` (modeling_motif.py:929-944): experts in ascending order,
        ``y.float() * w`` accumulated with ``index_add_`` into an fp32 buffer.
        """
        out = torch.zeros_like(x, dtype=torch.float32)
        expert_mask = F.one_hot(indices, num_classes=self.num_experts).permute(2, 1, 0)  # [E, k, N]
        for e in torch.nonzero(expert_mask.sum(dim=(1, 2))).flatten().tolist():
            slot, tok = torch.where(expert_mask[e])
            y = self.expert_forward(x[tok], e)
            out.index_add_(0, tok, y.float() * weights[tok, slot, None].float())
        return out


class MoE(nn.Module):
    """Router + routed experts + shared expert (HF ``MoE``, modeling_motif.py:954-1008).

    ``out = dtype( sum_k w_k expert_k(x) [fp32] + shared(x).float() )``.
    """

    def __init__(self, args: MotifArgs, layer_idx: int, materialize_experts: bool = True):
        super().__init__()
        self.router = Router(
            args.hidden_size, args.num_experts, args.experts_top_k, args.score_func, args.route_norm, args.route_scale
        )
        self.experts = RoutedExperts(args, layer_idx, materialize=materialize_experts)
        self.shared_experts = (
            MLP(
                args.hidden_size,
                args.shared_intermediate_size,
                eps=args.polynorm_eps,
                sigmoid_weight=args.polynorm_sigmoid_weight,
                hidden_clamp=args.hidden_clamp,
                output_scale=args.polynorm_output_scale_for_layer(layer_idx),
            )
            if args.num_shared_experts > 0
            else None
        )
        if args.load_balance_coeff is not None:
            self.expert_bias = nn.Parameter(torch.zeros(args.num_experts), requires_grad=False)
        else:
            self.expert_bias = None

    def forward(self, x: torch.Tensor, tap: Tap = None) -> torch.Tensor:
        B, S, D = x.shape
        xf = x.reshape(-1, D)
        weights, indices = self.router(xf, self.expert_bias, tap=_sub(tap, "router"))
        routed = self.experts(xf, indices, weights)
        _rec(tap, "routed_out", routed.view(B, S, D))
        total = routed
        if self.shared_experts is not None:
            shared = self.shared_experts(xf, tap=_sub(tap, "shared_experts"))
            total = routed + shared.float()
        out = total.view(B, S, D).to(x.dtype)
        _rec(tap, "out", out)
        return out


# =================================================================================================
# Decoder layer and model
# =================================================================================================
class DecoderLayer(nn.Module):
    """One Motif-3 block on the 4-stream residual ``X [B, S, E, D]`` (HF ``_forward_with_mhc``):

        h_pre, h_post, H_res = mhc_attn(X);  a = input_layernorm(pre(X));  o = GDLA(a)
        X1 = post(X, o) ;  h_pre', h_post', H_res' = mhc_ffn(X1);  f = post_attention_layernorm(pre(X1))
        X2 = post(X1, MLP(f) | MoE(f))
    With ``mhc_enabled=False`` it is a plain pre-norm residual block on ``[B, S, D]``.
    """

    def __init__(self, args: MotifArgs, layer_idx: int, materialize_experts: bool = True):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.self_attn = GDLAttention(args, layer_idx)
        self.is_moe = args.is_moe_layer(layer_idx)
        if self.is_moe:
            self.moe = MoE(args, layer_idx, materialize_experts=materialize_experts)
        else:
            self.mlp = MLP(
                args.hidden_size,
                args.intermediate_size,
                eps=args.polynorm_eps,
                sigmoid_weight=args.polynorm_sigmoid_weight,
                hidden_clamp=args.hidden_clamp,
                output_scale=args.polynorm_output_scale_for_layer(layer_idx),
            )
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mhc_enabled = args.mhc_enabled
        if self.mhc_enabled:
            mhc_kwargs = dict(
                n_streams=args.mhc_expansion_rate,
                dim=args.hidden_size,
                sinkhorn_iters=args.mhc_sinkhorn_iters,
                h_post_coeff=args.mhc_h_post_coeff,
                rms_eps=args.mhc_rms_eps,
                mix_fp32=args.mhc_mix_fp32,
            )
            self.mhc_attn = MHCLayer(**mhc_kwargs)
            self.mhc_ffn = MHCLayer(**mhc_kwargs)

    def ffn(self, f: torch.Tensor, tap: Tap = None) -> torch.Tensor:
        return self.moe(f, tap=_sub(tap, "moe")) if self.is_moe else self.mlp(f, tap=_sub(tap, "mlp"))

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: Optional[LatentKVCache] = None,
        attn_mode: str = "expanded",
        tap: Tap = None,
    ) -> torch.Tensor:
        _rec(tap, "x_in", x)
        if not self.mhc_enabled:
            a = self.input_layernorm(x)
            h = x + self.self_attn(a, positions, cache, attn_mode, tap=_sub(tap, "self_attn"))
            _rec(tap, "x_mid", h)
            out = h + self.ffn(self.post_attention_layernorm(h), tap)
            _rec(tap, "x_out", out)
            return out

        h_pre, h_post, h_res = self.mhc_attn(x, tap=_sub(tap, "mhc_attn"))
        x_red = MHCLayer.pre(x, h_pre)
        a = self.input_layernorm(x_red)
        _rec(tap, "mhc_attn.x_reduced", x_red)
        _rec(tap, "input_layernorm.out", a)
        o = self.self_attn(a, positions, cache, attn_mode, tap=_sub(tap, "self_attn"))
        x1 = MHCLayer.post(x, o, h_post, h_res)
        _rec(tap, "x_mid", x1)

        h_pre2, h_post2, h_res2 = self.mhc_ffn(x1, tap=_sub(tap, "mhc_ffn"))
        y_red = MHCLayer.pre(x1, h_pre2)
        f = self.post_attention_layernorm(y_red)
        _rec(tap, "mhc_ffn.x_reduced", y_red)
        _rec(tap, "post_attention_layernorm.out", f)
        u = self.ffn(f, tap)
        x2 = MHCLayer.post(x1, u, h_post2, h_res2)
        _rec(tap, "x_out", x2)
        return x2


class MotifModel(nn.Module):
    """Embedding -> expand to E identical streams -> decoder layers -> mean over streams -> final RMSNorm.

    ``layer_ids`` selects which decoder layers exist (default: all). ``forward`` runs the built layers in
    ascending order, so a model built with ``layer_ids=range(n)`` is an exact prefix ("early-exit") model.
    """

    def __init__(self, args: MotifArgs, layer_ids: Optional[Iterable[int]] = None, materialize_experts: bool = True):
        super().__init__()
        self.args = args
        ids = list(range(args.num_hidden_layers)) if layer_ids is None else sorted(int(i) for i in layer_ids)
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleDict({str(i): DecoderLayer(args, i, materialize_experts) for i in ids})
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    @property
    def layer_ids(self) -> list:
        return [int(k) for k in self.layers.keys()]

    def expand_streams(self, h: torch.Tensor) -> torch.Tensor:
        if not self.args.mhc_enabled:
            return h
        return h.unsqueeze(2).expand(-1, -1, self.args.mhc_expansion_rate, -1).contiguous()

    def reduce_streams(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2) if self.args.mhc_enabled else x

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        cache: Optional[MotifKVCache] = None,
        attn_mode: str = "expanded",
        tap: Tap = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_streams: bool = False,
    ):
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        B, S, _ = h.shape
        if positions is None:
            positions = torch.arange(S)[None, :].expand(B, S)
        _rec(tap, "embed", h)
        x = self.expand_streams(h)
        for i, layer in self.layers.items():
            layer_cache = cache.get(int(i)) if cache is not None else None
            x = layer(x, positions, layer_cache, attn_mode, tap=_sub(tap, f"layers.{i}"))
        h = self.reduce_streams(x)
        _rec(tap, "final.stream_mean", h)
        h = self.norm(h)
        _rec(tap, "final.norm", h)
        return (h, x) if return_streams else h


class MotifForCausalLM(nn.Module):
    """``MotifModel`` + untied ``lm_head``; logits are returned in fp32 (HF modeling_motif.py:1667-1668)."""

    def __init__(self, args: MotifArgs, layer_ids: Optional[Iterable[int]] = None, materialize_experts: bool = True):
        super().__init__()
        self.args = args
        self.model = MotifModel(args, layer_ids, materialize_experts)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.requires_grad_(False)

    @property
    def dtype(self) -> torch.dtype:
        return self.model.norm.weight.dtype

    def new_cache(self, batch_size: int, max_seq_len: int, dtype: Optional[torch.dtype] = None) -> MotifKVCache:
        """Latent KV cache for the built layers. ``dtype`` (default: the model dtype) is storage precision only;
        attention reads the entries back in the activation dtype."""
        return MotifKVCache.create(self.args, self.model.layer_ids, batch_size, max_seq_len, dtype or self.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        cache: Optional[MotifKVCache] = None,
        attn_mode: str = "expanded",
        tap: Tap = None,
        last_token_only: bool = False,
    ) -> torch.Tensor:
        h = self.model(input_ids, positions, cache, attn_mode, tap)
        if last_token_only:
            h = h[:, -1:]
        logits = self.lm_head(h).float()
        _rec(tap, "logits", logits)
        return logits


class MotifMTP(nn.Module):
    """Multi-token-prediction head (``model.mtp_layers.0.*``; fork motif_mtp.py:106-170, training model.py:865-876).

    ``h = input_proj(cat([h_main_postnorm, embed_norm(embed(t+1))]))``, then one plain pre-norm block (no mHC)
    with SWA attention in "all" mode (window 129, plain RoPE, scale 192^-0.5) and a dense PolyNorm MLP, then
    ``final_layernorm``. Logits use the main model's ``lm_head``. The block's KV cache is its own
    :class:`LatentKVCache` (pass ``layer_idx = num_hidden_layers``).
    """

    def __init__(self, args: MotifArgs):
        super().__init__()
        D = args.hidden_size
        self.args = args
        self.layer_idx = args.num_hidden_layers
        self.embed_norm = RMSNorm(D, eps=args.rms_norm_eps)
        self.input_proj = nn.Linear(2 * D, D, bias=False)
        self.input_layernorm = RMSNorm(D, eps=args.rms_norm_eps)
        self.self_attn = GDLAttention(args, self.layer_idx, swa=True)
        self.post_attention_layernorm = RMSNorm(D, eps=args.rms_norm_eps)
        self.mlp = MLP(
            D,
            args.intermediate_size,
            eps=args.polynorm_eps,
            sigmoid_weight=args.polynorm_sigmoid_weight,
            hidden_clamp=None,  # fork dense MotifMLP gets no hidden_clamp (no-op either way)
            output_scale=args.polynorm_output_scale_for_layer(self.layer_idx),
        )
        self.final_layernorm = RMSNorm(D, eps=args.rms_norm_eps)
        self.requires_grad_(False)

    def forward(
        self,
        h_main: torch.Tensor,
        next_embeds: torch.Tensor,
        positions: torch.Tensor,
        cache: Optional[LatentKVCache] = None,
        attn_mode: str = "expanded",
        tap: Tap = None,
    ) -> torch.Tensor:
        """``h_main [B,S,D]`` = main model post-norm hidden at positions p, ``next_embeds`` = embed(token p+1)."""
        e = self.embed_norm(next_embeds)
        h = self.input_proj(torch.cat([h_main, e], dim=-1))
        _rec(tap, "input_proj.out", h)
        h = h + self.self_attn(self.input_layernorm(h), positions, cache, attn_mode, tap=_sub(tap, "self_attn"))
        h = h + self.mlp(self.post_attention_layernorm(h), tap=_sub(tap, "mlp"))
        out = self.final_layernorm(h)
        _rec(tap, "out", out)
        return out
