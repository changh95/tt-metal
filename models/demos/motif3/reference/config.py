# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model arguments for the Motif-3 CPU golden reference.

``MotifArgs`` mirrors every ``config.json`` field the model actually consumes (see
docs/study/01_motif_reference.md §2) plus the constants that are hard-coded in the HF modeling file
(mHC RMSNorm eps 1e-6, PolyNorm eps 1e-6). Defaults are the released Motif-3 values, so
``MotifArgs()`` is the real 314B configuration.

Layer schedule (HF ``modeling_motif.py:566-578`` == fork ``motif.py:489-507`` == training
``model.py:185-203``): a layer is SWA iff ``layer_idx % sliding_window_period != 0`` (pattern
"interleave"). NOTE: the prose in the fork's ``motif_docs.md`` ("(layer_idx + 1) % period") is stale;
all three codebases use ``layer_idx % period``.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union

_DEFAULT_ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 64.0,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "rope_theta": 10000.0,
    "apply_yarn_scaling": False,
}


@dataclass
class MotifArgs:
    # ---- shapes -----------------------------------------------------------------------------------
    vocab_size: int = 220160
    hidden_size: int = 4096
    num_hidden_layers: int = 53
    # GDLA attention: 80 q heads = 16 groups x (4 signal + 1 noise); 16 KV heads (one per group).
    num_attention_heads: int = 80
    num_key_value_heads: int = 16
    num_noise_heads: int = 16
    head_dim: int = 192  # qk head dim = nope (128) + rope (64)
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    attention_cls: str = "gdla"
    diff_v2: bool = True
    elementwise_attn_output_gate: bool = True
    headwise_attn_output_gate: bool = False
    attention_dropout: float = 0.0

    # ---- FFN / MoE ----------------------------------------------------------------------------------
    hidden_act: str = "poly_norm"
    intermediate_size: int = 12288  # dense layers 0, 1 (and MTP)
    moe_intermediate_size: int = 1280  # routed + shared experts
    num_experts: int = 384
    experts_top_k: int = 8
    num_shared_experts: int = 1
    n_dense_first_layers: int = 2
    interleave_moe_layer_step: int = 1
    score_func: str = "sigmoid"
    route_norm: bool = True
    route_scale: float = 2.0
    load_balance_coeff: Optional[float] = 1e-4  # not None => the MoE owns an ``expert_bias`` (selection only)
    score_before_experts: bool = False
    polynorm_sigmoid_weight: bool = True
    polynorm_output_scale: float = 0.5
    polynorm_output_scale_per_layer: dict = field(default_factory=dict)  # fork motif.py:1246-1252
    polynorm_bias_clamp: Optional[float] = 0.5  # routed experts only (fork motif_docs.md "Fixed discrepancies" #1)
    hidden_clamp: Optional[float] = 1e6  # effectively a no-op at real activation ranges
    polynorm_eps: float = 1e-6  # hard-coded (HF modeling_motif.py:56, :99)

    # ---- norms ------------------------------------------------------------------------------------
    rms_norm_eps: float = 1e-5  # input/post-attn/q/kv/final norms
    mhc_rms_eps: float = 1e-6  # hard-coded mHC RMSNorm eps (HF modeling_motif.py:188)

    # ---- mHC ----------------------------------------------------------------------------------------
    mhc_enabled: bool = True
    mhc_expansion_rate: int = 4
    mhc_sinkhorn_iters: int = 20
    mhc_identity_init: bool = False  # init only
    mhc_h_post_alpha_end: float = 0.0  # h_post = (1 + alpha_end) * sigmoid(.) = 1.0 * sigmoid(.)

    # ---- attention windows / RoPE ------------------------------------------------------------------
    use_sliding_window: bool = True
    sliding_window: Optional[int] = 128  # config value; the effective window is sliding_window + 1 keys
    sliding_window_pattern: str = "interleave"
    sliding_window_period: int = 4
    rope_theta: float = 10000.0
    swa_rope_theta: Optional[float] = 10000.0
    max_position_embeddings: int = 262144
    original_seq_len: int = 4096  # softmax-scale mscale (top-level, as HF/fork/training)
    rope_factor: float = 64.0
    mscale: float = 1.0
    rope_scaling: Optional[dict] = field(default_factory=lambda: dict(_DEFAULT_ROPE_SCALING))

    # ---- misc ---------------------------------------------------------------------------------------
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = 1
    bos_token_id: int = 1
    pad_token_id: int = 0
    eos_token_ids: tuple = (0, 3, 6)  # generation_config.json

    # ---- numerics knobs (only matter when running in bf16; defaults = HF modeling_motif.py) ---------
    # Each knob switches ONE part of the computation to the fork's precision. Neither reproduces the fork's
    # numerics as a whole: TF32 router/mHC GEMMs, routing weights folded into GEMM2, the bf16 shared + routed
    # add and the kernels' bf16 P are not modelled (README "Known deviations" #4).
    # HF computes wq_a / q_norm / wq_b in fp32 (modeling_motif.py:647-650). The fork and training run
    # them in the activation dtype (bf16). False uses the fork's bf16 q path.
    q_path_fp32: bool = True
    # HF rounds the mHC RMSNorm output and the 24 projection outputs to bf16 (modeling_motif.py:238-243).
    # The fork's default tilelang path folds the RMSNorm gamma into the projection and produces the 24
    # mixes in fp32 (fork motif.py:302-360, layers/mhc.py:224-329). True keeps the 24 mixes in fp32 like the
    # fork (the fork's TF32 GEMM rounding is not modelled).
    mhc_mix_fp32: bool = False

    def __post_init__(self):
        if isinstance(self.eos_token_ids, list):
            self.eos_token_ids = tuple(self.eos_token_ids)
        if self.rope_scaling is not None:
            self.rope_scaling = dict(self.rope_scaling)
        self.validate()

    # ---------------------------------------------------------------------------------------------
    # validation
    # ---------------------------------------------------------------------------------------------
    def validate(self) -> None:
        if self.attention_cls != "gdla" or not self.diff_v2:
            raise NotImplementedError("Only GDLA with diff_v2=True exists for Motif-3 (HF modeling_motif.py:612-624)")
        if self.headwise_attn_output_gate:
            raise NotImplementedError("headwise_attn_output_gate is not used by Motif-3")
        if self.hidden_act != "poly_norm":
            raise NotImplementedError("Motif-3 uses hidden_act='poly_norm'")
        if self.score_func not in ("sigmoid", "softmax"):
            raise ValueError(f"unknown score_func {self.score_func!r}")
        if self.score_before_experts:
            raise NotImplementedError("score_before_experts=True is not used by Motif-3")
        if self.num_noise_heads <= 0 or (self.num_attention_heads - self.num_noise_heads) % self.num_noise_heads:
            raise ValueError("num_attention_heads - num_noise_heads must be a multiple of num_noise_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.num_attention_heads % self.heads_per_group:
            raise ValueError("num_attention_heads must be a multiple of grouped_ratio + 1")
        if self.qk_rope_head_dim % 2 or self.qk_rope_head_dim >= self.head_dim:
            raise ValueError("qk_rope_head_dim must be even and smaller than head_dim")
        if self.sliding_window_pattern not in ("interleave", "all"):
            raise ValueError(f"unknown sliding_window_pattern {self.sliding_window_pattern!r}")

    # ---------------------------------------------------------------------------------------------
    # derived quantities
    # ---------------------------------------------------------------------------------------------
    @property
    def grouped_ratio(self) -> int:
        """Signal heads per differential group (4 for Motif-3; HF modeling_motif.py:556)."""
        return (self.num_attention_heads - self.num_noise_heads) // self.num_noise_heads

    @property
    def heads_per_group(self) -> int:
        """q heads per differential group: grouped_ratio signal heads + 1 noise head (5)."""
        return self.grouped_ratio + 1

    @property
    def n_signal_heads(self) -> int:
        return self.grouped_ratio * self.num_noise_heads

    @property
    def qk_nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def kv_group_size(self) -> int:
        """q heads per KV head (GQA repeat factor, 5 for Motif-3)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def mhc_h_post_coeff(self) -> float:
        """h_post multiplier (HF modeling_motif.py:1090; fork motif.py:1282) -> 1.0 for Motif-3."""
        return 1.0 + float(self.mhc_h_post_alpha_end)

    @property
    def effective_sliding_window(self) -> Optional[int]:
        """Number of keys an SWA query attends to, INCLUDING itself: sliding_window + 1 = 129.

        Training: flash-attn ``window_size=(sliding_window, 0)``. HF/fork pass ``sliding_window + 1``
        to backends that subtract one (HF modeling_motif.py:571; fork motif.py:499).
        """
        if not self.use_sliding_window or self.sliding_window is None:
            return None
        return int(self.sliding_window) + 1

    def is_swa_layer(self, layer_idx: int) -> bool:
        if self.effective_sliding_window is None:
            return False
        if self.sliding_window_pattern == "all":
            return True
        return layer_idx % self.sliding_window_period != 0

    def attention_window(self, layer_idx: int) -> Optional[int]:
        """Keys attended per query including the current one (129 on SWA layers), None = full causal."""
        return self.effective_sliding_window if self.is_swa_layer(layer_idx) else None

    def is_moe_layer(self, layer_idx: int) -> bool:
        """HF modeling_motif.py:1068-1073 / fork motif.py:1239-1245."""
        if self.interleave_moe_layer_step == 0 or self.num_experts == 0:
            return False
        return layer_idx >= self.n_dense_first_layers and (layer_idx + 1) % self.interleave_moe_layer_step == 0

    @property
    def yarn_mscale(self) -> float:
        """DeepSeek-style YaRN mscale folded into the softmax scale: 0.1 * mscale * ln(factor) + 1."""
        return 0.1 * self.mscale * math.log(self.rope_factor) + 1.0

    def softmax_scale(self, layer_idx: int, *, swa: Optional[bool] = None) -> float:
        """head_dim^-0.5, times mscale^2 on full-attention layers when max_pos > original_seq_len.

        HF modeling_motif.py:579-585 == fork motif.py:516-522 == training model.py:298-308.
        Motif-3: SWA 0.07216878, global 0.14467963.
        """
        is_swa = self.is_swa_layer(layer_idx) if swa is None else swa
        scale = self.head_dim**-0.5
        if (not is_swa) and self.max_position_embeddings > self.original_seq_len:
            scale = scale * self.yarn_mscale * self.yarn_mscale
        return scale

    def uses_yarn(self, layer_idx: int, *, swa: Optional[bool] = None) -> bool:
        """True if this layer's RoPE table is the YaRN-interpolated one.

        SWA layers use plain RoPE with ``swa_rope_theta`` when it is set (HF modeling_motif.py:626-632);
        otherwise they share the global table, exactly like HF/fork.
        """
        is_swa = self.is_swa_layer(layer_idx) if swa is None else swa
        if is_swa and self.swa_rope_theta is not None:
            return False
        return self.rope_type == "yarn"

    @property
    def rope_type(self) -> str:
        if isinstance(self.rope_scaling, dict):
            return self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))
        return "default"

    def rope_theta_for_layer(self, layer_idx: int, *, swa: Optional[bool] = None) -> float:
        is_swa = self.is_swa_layer(layer_idx) if swa is None else swa
        if is_swa and self.swa_rope_theta is not None:
            return float(self.swa_rope_theta)
        if self.rope_type == "yarn":
            return float(self.rope_scaling.get("rope_theta", self.rope_theta))
        return float(self.rope_theta)

    def yarn_params(self) -> dict:
        """YaRN parameters exactly as HF ``MotifRotaryEmbedding.__init__`` reads them (:337-359)."""
        rs = self.rope_scaling if isinstance(self.rope_scaling, dict) else {}
        return dict(
            theta=float(rs.get("rope_theta", self.rope_theta)),
            factor=float(rs.get("factor", self.rope_factor)),
            original_seq_len=int(rs.get("original_max_position_embeddings", self.original_seq_len)),
            beta_fast=float(rs.get("beta_fast", 32)),
            beta_slow=float(rs.get("beta_slow", 1)),
            max_seq_len=int(self.max_position_embeddings),
        )

    def polynorm_output_scale_for_layer(self, layer_idx: int) -> float:
        per_layer = self.polynorm_output_scale_per_layer or {}
        value = per_layer.get(layer_idx, per_layer.get(str(layer_idx)))
        return float(value) if value is not None else float(self.polynorm_output_scale)

    @property
    def shared_intermediate_size(self) -> int:
        # Training/fork: moe_intermediate_size * num_shared_experts. HF uses moe_intermediate_size (equal for
        # Motif-3's single shared expert).
        return self.moe_intermediate_size * self.num_shared_experts

    def layer_kind(self, layer_idx: int) -> str:
        return (
            ("swa" if self.is_swa_layer(layer_idx) else "global")
            + "/"
            + ("moe" if self.is_moe_layer(layer_idx) else "dense")
        )

    # ---------------------------------------------------------------------------------------------
    # (de)serialization
    # ---------------------------------------------------------------------------------------------
    @classmethod
    def from_hf_config(cls, config: Union[str, Path, dict], **overrides) -> "MotifArgs":
        """Build from an HF ``config.json`` (path to the file or its directory, or a dict).

        Keys that the model does not consume (``max_window_layers``, ``k_ratio``, ``initializer_range``,
        ``_debug_force_load_balance``, ...) are ignored. ``generation_config.json`` next to the config,
        if present, provides ``eos_token_ids``/``bos``/``pad``.
        """
        gen_cfg = {}
        if isinstance(config, (str, Path)):
            path = Path(config)
            if path.is_dir():
                path = path / "config.json"
            cfg = json.loads(path.read_text())
            gen_path = path.parent / "generation_config.json"
            if gen_path.exists():
                gen_cfg = json.loads(gen_path.read_text())
        else:
            cfg = copy.deepcopy(dict(config))
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in cfg.items() if k in names}
        if not cfg.get("use_sliding_window", True):
            kwargs["sliding_window"] = None  # HF MotifConfig: sliding_window if use_sliding_window else None
        if "eos_token_id" in gen_cfg or "eos_token_id" in cfg:
            eos = gen_cfg.get("eos_token_id", cfg.get("eos_token_id"))
            kwargs["eos_token_ids"] = tuple(eos) if isinstance(eos, (list, tuple)) else (int(eos),)
        if "bos_token_id" in gen_cfg:
            kwargs["bos_token_id"] = int(gen_cfg["bos_token_id"])
        if "pad_token_id" in gen_cfg:
            kwargs["pad_token_id"] = int(gen_cfg["pad_token_id"])
        kwargs.update(overrides)
        return cls(**kwargs)

    def to_hf_config_dict(self) -> dict:
        """kwargs for the HF ``MotifConfig`` so that the HF model has exactly these semantics."""
        d = asdict(self)
        for k in ("eos_token_ids", "polynorm_eps", "mhc_rms_eps", "q_path_fp32", "mhc_mix_fp32"):
            d.pop(k)
        d["eos_token_id"] = int(self.eos_token_ids[0]) if self.eos_token_ids else None
        d["rope_scaling"] = copy.deepcopy(self.rope_scaling)
        d["output_router_logits"] = False
        d["_debug_force_load_balance"] = False
        return d

    def replace(self, **changes) -> "MotifArgs":
        d = {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self)}
        d.update(changes)
        return type(self)(**d)

    def summary(self) -> str:
        kinds = [self.layer_kind(i) for i in range(self.num_hidden_layers)]
        return (
            f"MotifArgs(layers={self.num_hidden_layers}, hidden={self.hidden_size}, heads={self.num_attention_heads}"
            f"/{self.num_key_value_heads}kv ({self.num_noise_heads} groups x {self.grouped_ratio}+1), "
            f"experts={self.num_experts} top{self.experts_top_k}, window={self.effective_sliding_window}, "
            f"kinds={kinds})"
        )


def tiny_random_args(**overrides: Any) -> MotifArgs:
    """A small Motif-3-shaped config for fast CPU tests.

    Keeps every structural ratio that matters for correctness: 4 signal + 1 noise head per group with one KV
    head per group, top-8 routing with a shared expert and expert_bias, 4 mHC streams with 20 Sinkhorn
    iterations, the real window (128 + 1 keys) and the real YaRN parameters (factor 64, original 4096,
    max_pos 262144, beta 32/1; with a 16-dim rope slice the correction range is [2, 6], so the ramp is
    non-trivial). Six layers cover every layer kind: 0 global/dense, 1 swa/dense, 2-3 swa/moe,
    4 global/moe, 5 swa/moe.
    """
    base = dict(
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=20,
        num_key_value_heads=4,
        num_noise_heads=4,
        head_dim=48,
        qk_rope_head_dim=16,
        v_head_dim=32,
        q_lora_rank=96,
        kv_lora_rank=64,
        intermediate_size=384,
        moe_intermediate_size=64,
        num_experts=32,
        experts_top_k=8,
        num_shared_experts=1,
        n_dense_first_layers=2,
    )
    base.update(overrides)
    return MotifArgs(**base)
