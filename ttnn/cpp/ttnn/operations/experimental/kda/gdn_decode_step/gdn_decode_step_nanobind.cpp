// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "gdn_decode_step_nanobind.hpp"

#include "gdn_decode_step.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::kda::gdn_decode_step::detail {

void bind_gdn_decode_step(nb::module_& mod) {
    ttnn::bind_function<"gdn_decode_step", "ttnn.experimental.kda.">(
        mod,
        R"doc(
        One decode step (B = 1) of the gated delta rule with a fused gated RMSNorm, one core per value head.
        For value head ``h`` (key head ``h // (Nv/Nk)``), with the token in row 0 of every tile:
            qn = l2norm(q) * scale, kn = l2norm(k)
            S  = S * exp(g[h]);  delta = beta[h] * (v - kn @ S);  S += kn^T @ delta
            o  = qn @ S;  out = o / sqrt(mean(o^2) + norm_epsilon) * weight
        ``state`` is updated in place. Rows 1..31 of q/k/v are ignored (masked to zero).
        Args:
            qkv (ttnn.Tensor): ``[1, 1, 2*Nk*Dk + Nv*Dv]`` BFLOAT16 TILE, post conv+silu, laid out ``[q | k | v]``.
            beta (ttnn.Tensor): ``[1, 1, Nv]`` FLOAT32 or BFLOAT16 update strengths (sigmoid already applied).
            g (ttnn.Tensor): ``[1, 1, Nv]`` FLOAT32 or BFLOAT16 log decays.
            state (ttnn.Tensor): ``[1, Nv, Dk, Dv]`` FLOAT32 recurrent state, updated in place.
            weight (ttnn.Tensor): ``[Dv]`` BFLOAT16 gated-norm weight.
            num_value_heads, num_key_heads, key_dim, value_dim (int).
        Keyword Args:
            scale (float, optional): query scale, defaults to ``key_dim ** -0.5``.
            l2_epsilon (float): l2-norm epsilon (default 1e-6). norm_epsilon (float): RMSNorm epsilon (default 1e-6).
            memory_config, compute_kernel_config, output_dtype (FLOAT32 or BFLOAT16, default BFLOAT16).
            conv_hist (ttnn.Tensor, optional): fused-conv mode: packed conv history ``[Nv, 4, 32, 32]`` BFLOAT16, one
                tile per (value head, slot); row c of a tile is channel chunk c of the head's ``[q | k | v]`` row, slot 3
                is the newest token. ``qkv`` is then the full projection row ``[1, 1, W]`` = ``[q | k | v | z | a | b]``,
                ``beta`` is dt_bias and ``g`` is -exp(A_log) (both volume Nv). The op computes the 4-tap causal conv +
                SiLU, beta = sigmoid(b), decay = exp(-exp(A) * softplus(a + dt_bias)), gates the output with silu(z) and
                shifts the packed history in place (slot0 <- slot1, ..., slot3 <- new token).
            conv_taps (ttnn.Tensor, optional): packed taps ``[Nv, 4, 32, 32]`` BFLOAT16 in the same layout (tap 0 = oldest).
            qkvz_dim (int): column offset of the a|b block in the projection row (= 2*Nk*Dk + 2*Nv*Dv).
            num_tokens (int): T = k + 1 rows per user for the MULTI-TOKEN (speculative-verify) mode of the fused-conv
                op (default 1 = the one-token step above; T in 2..32 with group_users*T <= 32). Row grid of ``qkv``,
                ``qkv_prev`` and the output: tensor row ``r = s*T + j`` is user ``s`` (state / history slot ``s``),
                offset ``j``; rows ``>= B*T`` are padding (R tile-padded, B = ``accept.volume()``).
                ``qkv`` ``[1, 1, R, W]`` (or ``[1, R, W]``) BFLOAT16 TILE holds THIS step's projection rows
                ``[q | k | v | z | a | b]``: row ``j = 0`` of a user is its last committed token (the bonus token),
                rows ``j = 1..k`` the new drafts. Per user s the op
                (1) COMMITS: runs the recurrence (history shift + conv + SiLU, gates, state update) over
                    ``qkv_prev`` rows ``j = 1..accept[s]`` and then ``qkv`` row ``j = 0``, exactly as T=1 steps would
                    (same per-token op order: bit-identical state / packed history), writing ``state[s]`` and the
                    packed history slot ``s`` ONCE (no output rows for the prev rows);
                (2) DRAFTS: continues the recurrence over ``qkv`` rows ``j = 1..k`` in L1 only (nothing written back);
                (3) OUTPUT: writes the gated-norm output of every ``qkv`` row ``j = 0..k`` to output row ``s*T + j``
                    (= the T=1 op's output on the state after committing rows ``0..j-1``); padding rows are 0.
                The packed history keeps the T=1 parity convention (user s's chunks at tile rows ``2c + (s & 1)``),
                so ``conv_hist`` is interchangeable between the two modes without repacking.
            qkv_prev (ttnn.Tensor, optional): multi-token mode: the PREVIOUS step's ``qkv`` of this layer (same shape /
                dtype / layout as ``qkv``; zeros at the first step, where accept must be 0).
            accept (ttnn.Tensor, optional): multi-token mode: ``[B]`` or ``[1, B]`` UINT32 / INT32 ROW_MAJOR (DRAM):
                ``accept[s]`` = number of last step's drafts accepted for user s, clamped in-kernel to ``[0, T-1]``.
        Returns:
            ttnn.Tensor: ``[1, 1, Nv*Dv]`` normalized output (row 0 valid, padding rows zero). Fused-conv batched:
            ``[1, B, Nv*Dv]`` (row b = user b). Multi-token: ``[1, 1, R, Nv*Dv]`` (qkv's rank; row ``s*T + j``).
        )doc",
        &ttnn::experimental::kda::gdn_decode_step,
        nb::arg("qkv").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("state").noconvert(),
        nb::arg("weight").noconvert(),
        nb::arg("num_value_heads"),
        nb::arg("num_key_heads"),
        nb::arg("key_dim"),
        nb::arg("value_dim"),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("l2_epsilon") = 1e-6f,
        nb::arg("norm_epsilon") = 1e-6f,
        nb::arg("memory_config") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("output_dtype") = ttnn::DataType::BFLOAT16,
        nb::arg("conv_hist") = nb::none(),
        nb::arg("conv_taps") = nb::none(),
        nb::arg("qkvz_dim") = 0,
        nb::arg("num_tokens") = 1,
        nb::arg("qkv_prev") = nb::none(),
        nb::arg("accept") = nb::none());
}

}  // namespace ttnn::operations::experimental::kda::gdn_decode_step::detail
