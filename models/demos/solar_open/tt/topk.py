# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Solar-Open top-k router (design D1, contract C2).

Numerics, identical to ``SolarOpenMoE.route_tokens_to_experts`` with ``n_group == topk_group == 1``::

    logits = x @ gate.weight.T                           # fp32 output from bf16 operands
    s      = sigmoid(logits)
    sel    = s + e_score_correction_bias                 # bias is used for SELECTION only
    idx    = topk(sel, k)                                # unsorted
    w      = s[idx] / (sum(s[idx]) + 1e-20) * routed_scaling_factor   # UNBIASED scores, normalised

The router emits the DENSE ``[tokens, num_experts]`` bf16 TILE routing tensor that ``tt/experts`` consumes
(``w`` at the k selected expert columns, 0 elsewhere) plus the ``[tokens, k]`` indices, which the experts ignore.

Two implementations share the linear and the scatter (``MoEOptions.router_impl``):

* ``"fused"`` (default, 3 launches): ``ttnn.linear`` -> ``moe_grouped_topk`` (single-group path, fp32 inside the
  kernel) -> ``ttnn.scatter``. The fused op needs a bias tensor with the SAME logical shape as the scores, so a
  ``[tokens, num_experts]`` copy of the bias is kept per distinct token count.
* ``"ops"`` (fallback, ~10 launches, exact ``ttnn.sigmoid``): [typecast bf16 logits to fp32 ->] sigmoid -> add
  bias -> topk -> gather -> normalise -> typecast bf16 -> scatter. ``ttnn.scatter`` rejects fp32 TILE tensors,
  hence the typecast before it; the indices are uint32 (``ttnn.topk`` on fp32).

``MoEOptions.router_fp32_logits`` selects fp32 (default) or bf16 logits (and, for the fused op, bias); the
selection math runs in fp32 in both implementations (the fused kernel upcasts internally, the ops chain typecasts).
fp32 matters because a bf16 sigmoid score near 1 has a resolution of 2^-8 = 0.004, coarser than the selection
bias (real layer 0: |bias| <= 0.002) and than the gaps between the top experts' scores; the logits themselves are
always accumulated in fp32 (see ``compute_config``).

Trace safety: every per-call tensor is produced by a device op. The per-token-count bias copy and the bf16 zeros
that seed the scatter are persistent for the traced token counts (prebuilt for the decode batch, 32 and the traced
prefill length 128; other token counts <= ``_KEEP_ROW_TENSORS_UP_TO`` are built lazily and kept), so captured
graphs never allocate them. Longer token counts build them per call and free them afterwards (two device ops; the
128K fp32 bias copy is 64 MiB per device); such lengths must only run eagerly.
"""

import ttnn
from models.demos.solar_open.config import MoEOptions
from models.demos.solar_open.utils.general_utils import get_cache_file_name

# Per-token-count helper tensors ([T, E] bias copy for the fused op, [T, E] bf16 zeros for the scatter) are kept
# for T up to this bound and built-and-freed per call above it. Only the traced token counts need to persist (the
# decode batch, 32 and the traced prefill length 128, prebuilt in __init__); keeping every prefill length up to 4096
# would retain up to ~5 MiB per layer per distinct length (250 MiB per device over a 1K/2K/4K length sweep x 48 layers).
_KEEP_ROW_TENSORS_UP_TO = 128

# Token counts at or below this run the router in L1 (decode and the traced prefill@128); longer ones in DRAM.
_L1_MAX_TOKENS = 128

# HF adds 1e-20 to the top-k sum before normalising (SolarOpenMoE.route_tokens_to_experts).
_NORM_EPSILON = 1e-20


class TopKRouter:
    """Solar-Open MoE router: dense ``[tokens, num_experts]`` bf16 routing weights from ``[1, 1, tokens, hidden]``.

    Args:
        mesh_device: the mesh; the router weight and bias are replicated on every device (each device needs the
            scores of all experts to build the union mask used by the experts).
        hf_config: ``SolarOpenConfig`` (reads ``num_experts_per_tok``, ``num_local_experts``, ``hidden_size``,
            ``routed_scaling_factor``).
        state_dict: ``substate(mlp_state_dict, "gate")`` = ``{"weight": [E, H] bf16,
            "e_score_correction_bias": [E] fp32}``; an empty dict is cache-only mode (both tensors are read
            from ``tensor_cache_path``; a missing bias file raises, Solar always ships the bias).
        tensor_cache_path: cache stem directory; files ``weight`` and ``e_score_correction_bias`` (ttnn appends
            the dtype/layout suffix, so the fp32 and bf16 bias variants never collide).
        tokens_per_device: decode batch per device; its helper tensors are prebuilt so decode can be traced.
        moe_options: ``MoEOptions`` (``router_impl``, ``router_fp32_logits``); ``None`` means the defaults.
    """

    def __init__(
        self, mesh_device, hf_config, state_dict, tensor_cache_path=None, tokens_per_device=32, moe_options=None
    ):
        options = moe_options or MoEOptions()
        self.mesh_device = mesh_device
        self.top_k = hf_config.num_experts_per_tok
        self.num_experts = hf_config.num_local_experts
        self.hidden_dim = hf_config.hidden_size
        self.route_scale = float(getattr(hf_config, "routed_scaling_factor", 1.0))
        self.impl = options.router_impl
        self.fp32_logits = options.router_fp32_logits
        self.logits_dtype = ttnn.float32 if self.fp32_logits else ttnn.bfloat16
        # The ops chain adds the bias to fp32 scores (``_ops_topk``), so its bias is fp32 whatever the logits dtype;
        # the fused op takes the bias in the logits dtype and upcasts it itself.
        self.bias_dtype = ttnn.float32 if self.impl == "ops" else self.logits_dtype

        # Explicit replication: the linear needs all num_experts output columns on every device.
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        weight = state_dict["weight"].transpose(0, 1) if state_dict else None  # [H, E]
        bias = state_dict["e_score_correction_bias"].reshape(1, -1).float() if state_dict else None  # [1, E] fp32
        self.weight = ttnn.as_tensor(
            weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=replicate,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
        )
        self.bias = ttnn.as_tensor(
            bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.bias_dtype,
            mesh_mapper=replicate,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=get_cache_file_name(tensor_cache_path, "e_score_correction_bias"),
        )
        # HiFi4 keeps the full bf16 operand precision. The K=4096 dot product is ALWAYS accumulated in fp32, also
        # for bf16 logits: with fp32_dest_acc_en=False the accumulation itself runs in bf16 and the logits are off
        # by up to 0.65 (mean 0.15, measured on P150), which flips ~40 % of the tokens' expert sets; with fp32
        # accumulation the bf16 logits are exact up to their own output rounding (<= 0.03).
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # {tokens: (bias_wide, zeros)}: decode batch, the batched-decode padding width and the traced prefill length.
        self._row_tensors = {}
        for tokens in {tokens_per_device, 32, 128}:
            self._get_row_tensors(tokens, keep=True)

    def _get_row_tensors(self, tokens, keep=False):
        """Return ``((bias_wide, zeros), transient)`` for ``tokens`` rows.

        ``bias_wide`` is the ``[tokens, E]`` bias copy the fused op needs (``None`` for the ops impl, which
        broadcasts the ``[1, E]`` bias itself); ``zeros`` is the ``[tokens, E]`` bf16 TILE tensor the scatter
        starts from (scatter is out-of-place, so it is reused across calls). ``transient`` tells the caller to
        free both after use (token counts above ``_KEEP_ROW_TENSORS_UP_TO`` and not forced with ``keep``).
        """
        cached = self._row_tensors.get(tokens)
        if cached is not None:
            return cached, False
        bias_wide = None
        if self.impl == "fused":
            bias_wide = self.bias if tokens == 1 else ttnn.repeat(self.bias, [tokens, 1])
        zeros = ttnn.zeros(
            [tokens, self.num_experts],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        row_tensors = (bias_wide, zeros)
        if keep or tokens <= _KEEP_ROW_TENSORS_UP_TO:
            self._row_tensors[tokens] = row_tensors
            return row_tensors, False
        return row_tensors, True

    def _fused_topk(self, logits, bias_wide):
        """``moe_grouped_topk`` single-group path: -> (weights bf16 [T, k], indices uint16 [T, k])."""
        return ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits,
            bias_wide,
            n_groups=1,
            summed_experts_per_group=1,
            topk_groups=1,
            n_activated_experts=self.top_k,
            route_scale=self.route_scale,
            epsilon=_NORM_EPSILON,
            score_func="sigmoid",
        )

    def _ops_topk(self, logits):
        """Pure ttnn chain with the same numerics: -> (weights bf16 [T, k], indices uint32 [T, k])."""
        if logits.dtype != ttnn.float32:
            # Select in fp32 like the fused kernel: bf16 sigmoid scores (ulp 2^-8 near 1) would swallow the bias and
            # tie the top experts (measured on P150: 35/128 flips vs 3/128 with fp32 scores from the same logits).
            logits = ttnn.typecast(logits, ttnn.float32)
            scores = ttnn.sigmoid(logits)
            logits.deallocate(True)
        else:
            scores = ttnn.sigmoid(logits)
        biased = ttnn.add(scores, self.bias)  # [T, E] + [1, E] row broadcast, fp32; selection only
        _values, indices = ttnn.topk(biased, k=self.top_k, dim=-1, sorted=False)
        _values.deallocate(True)
        biased.deallocate(True)
        selected = ttnn.gather(scores, dim=-1, index=indices)  # UNBIASED scores of the selected experts
        scores.deallocate(True)
        total = ttnn.sum(selected, dim=-1, keepdim=True)
        denominator = ttnn.add(total, _NORM_EPSILON)
        total.deallocate(True)
        weights = ttnn.div(selected, denominator)
        selected.deallocate(True)
        denominator.deallocate(True)
        if self.route_scale != 1.0:
            scaled = ttnn.mul(weights, self.route_scale)
            weights.deallocate(True)
            weights = scaled
        if weights.dtype != ttnn.bfloat16:
            # ttnn.scatter does not support fp32 TILE tensors; the dense output is bf16 anyway.
            weights_bf16 = ttnn.typecast(weights, ttnn.bfloat16)
            weights.deallocate(True)
            weights = weights_bf16
        return weights, indices

    def __call__(self, hidden_states, is_decode=True):
        """Route ``hidden_states`` ``[1, 1, tokens, hidden]`` (bf16 or bfp8 TILE, replicated) to experts.

        Returns ``(indices, dense)``: ``indices`` ``[tokens, k]`` uint16 (fused) or uint32 (ops), unsorted
        and informational only; ``dense`` ``[tokens, num_experts]`` bf16 TILE with the normalised unbiased
        sigmoid score of each selected expert and 0 elsewhere (rows sum to ``routed_scaling_factor``).
        ``hidden_states`` is not consumed; ``is_decode`` is accepted for interface symmetry with the other
        modules (the memory placement follows the token count). No host reads.
        """
        # ttnn.Tensor.volume() is the tile-PADDED volume (32 rows for a [1,1,1,H] tensor); the logical token count
        # must drive the [T, E] helper tensors, or T=1/8 would meet a [32, E] bias copy (fused op shape check).
        tokens = hidden_states.logical_volume() // self.hidden_dim
        x = ttnn.reshape(hidden_states, (-1, self.hidden_dim))  # [tokens, hidden] view of the caller's tensor
        memory_config = ttnn.L1_MEMORY_CONFIG if tokens <= _L1_MAX_TOKENS else ttnn.DRAM_MEMORY_CONFIG

        # HF routes on the full-precision hidden states. The production input is bf16 (the residual stream is bf16),
        # but the contract admits bfp8: a bfp8 operand (7-bit mantissa shared per 16 elements) would move the logits
        # by several times the ~2e-3 the bf16 operand costs and push more tokens across the 8th/9th-score margin, so
        # a bfp8 input is widened to bf16 first (one extra launch, never taken by the model itself).
        x_cast = None
        if x.dtype == ttnn.bfloat8_b:
            x_cast = ttnn.typecast(x, ttnn.bfloat16)
            x = x_cast

        logits = ttnn.linear(
            x,
            self.weight,
            dtype=self.logits_dtype,
            memory_config=memory_config,
            compute_kernel_config=self.compute_config,
        )
        if x_cast is not None:
            x_cast.deallocate(True)
        (bias_wide, zeros), transient = self._get_row_tensors(tokens)
        if self.impl == "fused":
            weights, indices = self._fused_topk(logits, bias_wide)
        else:
            weights, indices = self._ops_topk(logits)
        logits.deallocate(True)

        dense = ttnn.scatter(zeros, dim=-1, index=indices, src=weights, memory_config=memory_config)
        weights.deallocate(True)
        if transient:
            zeros.deallocate(True)
            if bias_wide is not None and bias_wide is not self.bias:
                bias_wide.deallocate(True)
        return indices, dense
