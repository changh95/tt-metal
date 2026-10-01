# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Composite PolyNorm for Motif-3 on TT (design §2.3.5; WAVE_A_REVIEW §5.5 MLP-1 / MLP-2).

Semantics (HF ``modeling_motif.py:49-80`` ``PolyNormTorch`` and ``:112-137`` ``GroupedPolyNorm``; reference
``modules.PolyNorm`` / ``GroupedPolyNorm``), with the reduction over the intermediate dim ``I`` and fp32 math::

    N(z) = z / sqrt(mean(z^2) + eps)                                   eps = 1e-6 (cfg.polynorm_eps)
    poly = c0 N(g^3) + c1 N(g^2) + c2 N(g) + b,   c_k = sigmoid(w_k)    (w = act_fn.weight [3] or [E, 3])
    h    = poly * up                                                   x 0.5 output scale folded into W_down

* **Scalar** PolyNorm (dense MLP layers 0-1, shared expert): Python-scalar ``c_k`` and ``b``, **no** bias clamp.
* **Grouped** PolyNorm (routed experts): per-expert ``c_k`` / ``b`` as ``[1, E, 1, 1]`` (E = 12 per chip), bias
  clamped to +-0.5 (``weights.expert_polynorm_tensors``).
* ``hidden_clamp`` (1e6) is a no-op at Motif's ranges (|g| <= ~20-100, verify_numerics (d)) and is omitted.

Device formulation ("moments + Horner", the default, ``impl="horner"``)
------------------------------------------------------------------------
Since ``N(g^k) = g^k * rsqrt(mean(g^(2k)) + eps)``, PolyNorm needs three per-row moments and is then a cubic in ``g``
with per-row coefficients ``a_k = c_(3-k) * rsqrt(m_k + eps)``::

    s   = [sum g^2, sum g^4, sum g^6]                  per row, fp32 ("accurate" SFPU reduce, not TF32)
    s   = all_reduce(s, "tp")                          only for TP-sharded intermediates (dense 1536 / shared 160 per chip)
    a_k = rsqrt(s_k * D_k + E_k)                       D_k = 1 / (I c_k^2), E_k = eps / c_k^2  =>  a_k = c_k rsqrt(s_k/I + eps)
    t   = mac(g, a3, a2); t = mac(t, g, a1); t = mac(t, g, b)          = a3 g^3 + a2 g^2 + a1 g + b (Horner, SFPU fp32)
    h   = t * up                                       -> bf16 (never block float, study 01 N3)

Every elementwise product is an SFPU fp32 op (binary_ng / ternary ``mac`` take the SFPU path; the FPU would truncate
fp32 operands to TF32), and ``ttnn.sum`` on fp32 runs the accurate SFPU reduce (``fast_and_approximate_mode=False`` +
fp32 dest acc, ``polynorm`` role), so the statistics are fp32-exact. The 3-moment all-reduce is the reduction the
Motif vLLM fork uses for TP > 1 (study 01 §5.2; verify_moe (e)). Accuracy (device, vs fp64): fp32 mode sits at the
bf16 output-rounding floor (PCC 0.999998, max-abs = half a bf16 ulp of the largest output); bf16 mode PCC 0.99998.

Decode op count and cost (TP stats, defaults ``moments="sum", ar="ag_sum", horner="mac"``): 3 multiplies + concat +
sum, pad + all_gather + sum (the exact moments all-reduce), mac + rsqrt, 3 slices, 3 mac, multiply = 17 small ops.
Traced on this Galaxy (fp32, L1 intermediates, 32 logical rows): dense 8 x 1536 per chip 90.7 us, shared 8 x 160
73.8 us (``ttnn.all_reduce`` for the moments instead: +15 / +26 us; 8 logical rows instead of 32: +14 / +11 us, from
implicit-padding fills inside the reductions). In trace mode every op costs ~5-6 us of dispatch on top of its
kernel, so the op count dominates: a 6-op binary Horner is +17 us slower than the 3 ternary ``mac`` (whose kernels
are ~4x slower than binary ops at these sizes); the FPU (TF32) moment sum saves only ~4 us. The fused PolyNorm kernel
is the v1 lever.

The alternative ``impl="rms"`` (grouped only) is the G6 composite (``N(z) = ttnn.rms_norm(z, eps)``, weightless) with
the coefficient multiply-adds fused into ``ttnn.mac``: 2 multiplies + 3 rms_norm + 3 mac + multiply. ``rms_norm``
squares and reduces on the FPU (TF32-class operands); in fp32 mode its accuracy equals the Horner path at the bf16
output floor. Grouped ``[1, 12, 32, 1280]`` per chip, traced, L1 intermediates: horner fp32 115.8 us, rms fp32
115.1 us, rms bf16 81.1 us (PCC 0.999998 / 0.999998 / 0.999991); DRAM intermediates are 2-3x slower (the
elementwise ops become DRAM-bandwidth bound: 24 us per fp32 op on 2 MB tensors).

Modes: ``"fp32"`` = fp32 intermediates (design §1.5 decode default; inputs typecast unless they already are fp32,
e.g. a gate_up matmul with ``dtype=float32``); ``"bf16"`` = bf16 Horner intermediates with fp32 moments (the
prefill lever, PCC-gated).

TP collectives go through :class:`~models.demos.motif3.tt.ccl.MotifCCL`. ``ar="ag_sum"`` (default) all-gathers the
zero-padded ``[1, 3, T, 32]`` moment tile over TP and sums it with the accurate fp32 reduce (exact for any T, 3 ops;
``moments="pre_ag"``, whose stats are one tile wide, first slices the sum column: +1 op). ``ar="all_reduce"`` uses
``ttnn.all_reduce``: exact (all-gather + local sum) for decode-sized payloads, but its reduce-scatter path for
``T > 32`` adds fp32 at TF32 class -- with ``exact_ar=True`` (default) such prefill payloads are routed to ``ag_sum``.

Interfaces (shapes per chip; all inputs TILE; outputs DRAM interleaved unless ``memory_config`` says otherwise; no
input is consumed):

* :func:`polynorm_tp(g, u, consts, *, ccl=None, mode="fp32")` -- scalar PolyNorm, ``g, u [1, 1, T, n]`` -> ``h
  [1, 1, T, n]`` bf16. ``ccl`` given: moments all-reduced over TP (``n = I / 8``); ``ccl=None``: local stats (``n = I``).
* :func:`grouped_polynorm(gu, consts, *, inter, mode="fp32")` -- routed experts, ``gu [1, E, M, 2 inter]`` (gate |
  up) or ``(g, u)`` -> ``[1, E, M, inter]`` bf16. Same call signature as the MoE module's local copy; ``consts`` is a
  :class:`GroupedPolyNormConsts` (all impls) or a mapping with ``c0, c1, c2, b`` ``[1, E, 1, 1]`` tensors (``impl="rms"``).
* :class:`ScalarPolyNormConsts`, :class:`GroupedPolyNormConsts` -- the device constants (built once per layer). Both
  go through the TT weight cache when given cache names and read the checkpoint's ``act_fn.{weight,bias}`` only on a
  cache miss (lazy sources), so a model can start from the TT cache alone (HF shards deleted, WAVE_A_REVIEW D4).
  fp32 constants use ``cfg.dtypes.polynorm_coeffs`` (must be fp32); the bf16 mode's constants are bf16.
* :func:`polynorm_output_scale(cfg, layer)` / :func:`check_polynorm_semantics(cfg)` -- the x0.5 output scale of a layer
  (folded into ``W_down`` by the callers) and the supported-semantics guard (sigmoid coefficients). ``MotifTTConfig``
  parses neither ``polynorm_sigmoid_weight`` nor ``polynorm_output_scale_per_layer`` yet (requested shared change);
  both helpers honour them when the config carries them (revision 2ed2ed5c: absent / ``{}``, i.e. sigmoid + one 0.5).

Trace safety: fixed shapes, per-layer constant slices only, no host round trips; constants are device tensors
created in the constructors (none are created inside the forward functions).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch

import ttnn

from . import weights as W
from .model_config import TILE, MotifTTConfig

POLYNORM_MODES = ("fp32", "bf16")
POLYNORM_IMPLS = ("horner", "rms")


def _free(*ts) -> None:
    for t in ts:
        if t is not None:
            ttnn.deallocate(t)


def _memory_configs(rows: int, memory_config, intermediate_memory_config):
    """(output mc, intermediate mc): output ``memory_config`` (default DRAM); intermediates
    ``intermediate_memory_config``, else ``memory_config`` when given, else L1 for decode-sized inputs (<= 32 rows;
    measured 2-3x faster than DRAM for the grouped PolyNorm) and DRAM otherwise."""
    out_mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
    if intermediate_memory_config is not None:
        imc = intermediate_memory_config
    elif memory_config is not None:
        imc = memory_config
    else:
        imc = ttnn.L1_MEMORY_CONFIG if int(rows) <= 32 else ttnn.DRAM_MEMORY_CONFIG
    return out_mc, imc


def _check_mode(mode: str) -> None:
    if mode not in POLYNORM_MODES:
        raise ValueError(f"polynorm mode must be one of {POLYNORM_MODES}, got {mode!r}")


def _math_dtype(mode: str):
    return ttnn.float32 if mode == "fp32" else ttnn.bfloat16


def _coeff_dtype(cfg: MotifTTConfig):
    """``cfg.dtypes.polynorm_coeffs`` (README §4: fp32). The Horner scales ``D, E`` feed fp32 moments and the fp32-mode
    Horner tail, so anything else is rejected rather than silently mixing dtypes in the ternary ops."""
    dt = cfg.dtypes.polynorm_coeffs
    if dt != ttnn.float32:
        raise NotImplementedError(f"PolyNorm constants must be fp32 (cfg.dtypes.polynorm_coeffs = {dt})")
    return dt


# ============================================================================================================
# Config semantics (HF modeling_motif.py:483-498, 905-907; reference config.polynorm_output_scale_for_layer)
# ============================================================================================================
def polynorm_output_scale(cfg: MotifTTConfig, layer_idx: int) -> float:
    """The PolyNorm output scale of layer ``layer_idx`` (0.5 for Motif-3), folded into ``W_down`` by the callers.

    ``cfg.polynorm_output_scale``, unless the config carries a per-layer override (the fork's
    ``polynorm_output_scale_per_layer``, ``{layer: scale}``; the reference supports it). ``MotifTTConfig`` does not parse
    that field yet (requested shared change), so today this is always the global value -- correct for revision
    2ed2ed5c, whose ``config.json`` has ``polynorm_output_scale_per_layer: {}``."""
    per_layer = getattr(cfg, "polynorm_output_scale_per_layer", None) or {}
    v = per_layer.get(int(layer_idx), per_layer.get(str(int(layer_idx))))
    return float(cfg.polynorm_output_scale if v is None else v)


def check_polynorm_semantics(cfg: MotifTTConfig) -> None:
    """Reject PolyNorm configs this implementation does not support: ``polynorm_sigmoid_weight=False`` (raw
    coefficients; the Horner form needs ``c_k > 0``, which sigmoid outputs guarantee). A no-op until ``MotifTTConfig``
    parses the field (requested shared change; HF default True, revision 2ed2ed5c does not set it)."""
    if not bool(getattr(cfg, "polynorm_sigmoid_weight", True)):
        raise NotImplementedError("polynorm_sigmoid_weight=False is not supported (TT PolyNorm applies sigmoid(w))")


# ============================================================================================================
# Host-side coefficients
# ============================================================================================================
@dataclass(frozen=True)
class PolyNormCoefficients:
    """Scalar PolyNorm coefficients (fp32 values as python floats): ``poly = c0 N(g^3) + c1 N(g^2) + c2 N(g) + b``.

    ``c_k = sigmoid(act_fn.weight[k])`` in fp32 (HF ``_coeffs``), ``b = act_fn.bias`` (dense / shared: unclamped)."""

    c0: float
    c1: float
    c2: float
    b: float

    @classmethod
    def from_tensors(cls, weight: torch.Tensor, bias: torch.Tensor, *, bias_clamp: Optional[float] = None):
        c, b = W.polynorm_coefficients(weight.reshape(-1)[:3], bias.reshape(-1)[:1], bias_clamp=bias_clamp)
        c = c.to(torch.float32)
        b = b.to(torch.float32)
        return cls(float(c[0]), float(c[1]), float(c[2]), float(b[0]))

    @classmethod
    def from_source(cls, source, prefix: str, *, bias_clamp: Optional[float] = None):
        """``prefix`` = e.g. ``model.layers.0.mlp`` or ``model.layers.2.moe.shared_experts`` (reads ``act_fn.*``)."""
        return cls.from_tensors(source.get(f"{prefix}.act_fn.weight"), source.get(f"{prefix}.act_fn.bias"), bias_clamp=bias_clamp)

    @property
    def by_power(self) -> Tuple[float, float, float]:
        """Coefficients of ``N(g), N(g^2), N(g^3)`` (moment order k = 1, 2, 3) = ``(c2, c1, c0)``."""
        return (self.c2, self.c1, self.c0)


def horner_scale_constants(
    c_by_power: torch.Tensor, inter: int, eps: float, dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(D, E)`` with ``D_k = 1 / (inter c_k^2)``, ``E_k = eps / c_k^2`` (fp64 math, returned in ``dtype``), so that
    ``rsqrt(sum_k * D_k + E_k) = c_k rsqrt(sum_k / inter + eps)`` for ``c_k > 0`` (sigmoid outputs always are).
    ``c_by_power``: ``[..., 3]`` coefficients of ``N(g), N(g^2), N(g^3)``."""
    c = c_by_power.to(torch.float64)
    if bool((c <= 0).any()):
        raise ValueError("PolyNorm coefficients must be > 0 (sigmoid weights); the Horner form needs c_k > 0")
    D = 1.0 / (float(inter) * c * c)
    E = float(eps) / (c * c)
    return D.to(dtype), E.to(dtype)


# ============================================================================================================
# Device constants
# ============================================================================================================
def _upload(src, mesh_device, cfg: MotifTTConfig, dtype, *, dp_dim=None, tp_dim=None, cache_name=None, layer=None):
    """``weights.as_tensor`` of a host tensor or a zero-arg callable (only called on a cache miss)."""
    return W.as_tensor(
        src if callable(src) else src.contiguous(),
        mesh_device=mesh_device,
        cfg=cfg,
        dtype=dtype,
        dp_dim=dp_dim,
        tp_dim=tp_dim,
        cache_name=cache_name,
        layer=layer,
    )


class _Lazy:
    """Memoized zero-arg callable (host constants built at most once, and only if some cache lookup misses)."""

    def __init__(self, fn):
        self.fn, self.value, self.done = fn, None, False

    def __call__(self):
        if not self.done:
            self.value, self.done = self.fn(), True
            self.fn = None  # release the closure (e.g. the weight source)
        return self.value


class ScalarPolyNormConsts:
    """Device constants of one scalar PolyNorm site (dense MLP or shared expert), replicated on every chip.

    * ``D``, ``E``: ``[1, 3, 1, 1]`` fp32 (moment order ``g^2, g^4, g^6`` <-> ``N(g), N(g^2), N(g^3)``), see
      :func:`horner_scale_constants` -- ``inter`` is the FULL intermediate size (12288 / 1280), also under TP.
    * ``b``: ``[1, 1, 1, 1]`` fp32 and bf16 (the Horner tail; a tensor so all layers share one program).
    * ``coeffs``: the host :class:`PolyNormCoefficients` (lazy when built from a callable).

    ``coeffs`` may be a :class:`PolyNormCoefficients` or a zero-arg callable returning one (e.g. reading
    ``act_fn.{weight,bias}`` from the checkpoint). With ``cache_name`` (e.g. ``"mlp.polynorm"``) the four tensors are
    cached as ``<cache_name>.{D,E,b}`` (replicated; ``b`` in fp32 and bf16) and the callable runs only on a cache miss
    or on the first :attr:`coeffs` access."""

    def __init__(self, mesh_device, cfg: MotifTTConfig, coeffs, *, inter: int, eps: Optional[float] = None,
                 cache_name: Optional[str] = None, layer: Optional[int] = None):
        if isinstance(coeffs, PolyNormCoefficients):
            self._coeffs = _Lazy(lambda: coeffs)
        elif callable(coeffs):
            self._coeffs = _Lazy(coeffs)
        else:
            raise TypeError(f"coeffs must be PolyNormCoefficients or a callable, got {type(coeffs)}")
        self.inter = int(inter)
        self.eps = float(cfg.polynorm_eps if eps is None else eps)
        dt = _coeff_dtype(cfg)

        def build():
            c = self.coeffs
            D, E = horner_scale_constants(torch.tensor(c.by_power, dtype=torch.float64), self.inter, self.eps)
            b = torch.tensor([[[[c.b]]]], dtype=torch.float32)
            return {"D": D.reshape(1, 3, 1, 1).contiguous(), "E": E.reshape(1, 3, 1, 1).contiguous(), "b": b}

        host = _Lazy(build)
        name = (lambda k: f"{cache_name}.{k}") if cache_name else (lambda k: None)
        up = lambda k, d: _upload(lambda: host()[k], mesh_device, cfg, d, cache_name=name(k), layer=layer)  # noqa: E731
        self.D = up("D", dt)
        self.E = up("E", dt)
        self.b = {"fp32": up("b", dt), "bf16": up("b", ttnn.bfloat16)}

    @property
    def coeffs(self) -> PolyNormCoefficients:
        return self._coeffs()

    def deallocate(self) -> None:
        _free(self.D, self.E, *self.b.values())


class GroupedPolyNormConsts:
    """Device constants of the routed experts' grouped PolyNorm on every chip (EP32 placement, ``[1, E, 1, 1]``
    per chip for chip ``k = 8 dp + tp`` = experts ``[12k, 12k+12)``).

    * ``c0, c1, c2, b``: ``[1, E, 1, 1]`` in fp32 and bf16 (``self.c["fp32"]["c0"]`` ...; ``b`` = clamp(bias, +-0.5));
      ``as_dict(mode)`` gives the MoE module's ``{c0, c1, c2, b}`` mapping.
    * ``D``, ``E``: ``[3, E, 1, 1]`` fp32 Horner scale constants (dim 0 = moment order ``g^2, g^4, g^6``).

    Build with :meth:`from_source` (checkpoint ``model.layers.{l}.moe.experts.act_fn.{weight,bias}``, ``[384, 3]`` /
    ``[384, 1]``, read only on a cache miss) or :meth:`from_coefficients` (host ``c [384, 3]``, ``b [384]`` after
    sigmoid / clamp). ``c`` may also be a zero-arg callable returning ``(c, b)`` (then ``b`` is omitted). Cache names
    ``moe.experts.polynorm.{c0,c1,c2,b}`` (shared with tt/moe.py) and ``moe.experts.polynorm.{D,E}``."""

    def __init__(self, mesh_device, cfg: MotifTTConfig, c, b: Optional[torch.Tensor] = None, *,
                 inter: Optional[int] = None, eps: Optional[float] = None, cache: bool = False,
                 layer: Optional[int] = None):
        self.inter = int(inter if inter is not None else cfg.moe_intermediate_size)
        self.eps = float(cfg.polynorm_eps if eps is None else eps)
        self.e_loc = cfg.experts_per_chip
        ep = dict(dp_dim=0, tp_dim=1)
        name = (lambda k: f"moe.experts.polynorm.{k}") if cache else (lambda k: None)
        if callable(c):
            coeff_fn = c
        else:
            if b is None:
                raise ValueError("GroupedPolyNormConsts: b is required when c is a tensor")
            coeff_fn = lambda: (c, b)  # noqa: E731
        host = _Lazy(lambda: self.host_tensors(cfg, *coeff_fn(), inter=self.inter, eps=self.eps))
        dt32 = _coeff_dtype(cfg)
        up = lambda k, dt: _upload(lambda: host()[k], mesh_device, cfg, dt, cache_name=name(k), layer=layer, **ep)  # noqa: E731
        self.c = {mode: {k: up(k, dt) for k in ("c0", "c1", "c2", "b")}
                  for mode, dt in (("fp32", dt32), ("bf16", ttnn.bfloat16))}
        self.D = up("D", dt32)
        self.E = up("E", dt32)

    @staticmethod
    def host_tensors(cfg: MotifTTConfig, c: torch.Tensor, b: torch.Tensor, *, inter: Optional[int] = None,
                     eps: Optional[float] = None):
        """Host layout (pure torch) of the device constants, all mapped with ``dp_dim=0, tp_dim=1``:

        * ``c0, c1, c2, b``: ``[dp, 96, 1, 1]`` fp32 = ``weights.ep_layout`` of the per-expert values (identical to
          ``weights.expert_polynorm_tensors``); chip ``k = 8 dp + tp`` gets ``[1, 12, 1, 1]``;
        * ``D, E``: ``[dp * 3, 96, 1, 1]`` with row ``3 dp + k`` = moment ``k`` (``g^2, g^4, g^6``), so chip (dp, tp)
          gets ``[3, 12, 1, 1]``."""
        inter = int(inter if inter is not None else cfg.moe_intermediate_size)
        eps = float(cfg.polynorm_eps if eps is None else eps)
        n_exp, dp = cfg.num_experts, cfg.dp
        c = c.reshape(n_exp, 3).to(torch.float32)
        b = b.reshape(n_exp).to(torch.float32)

        def ep_col(v):  # [384] -> [dp, 96, 1, 1]
            return W.ep_layout(v.reshape(-1, 1, 1).contiguous(), cfg)

        out = {"c0": ep_col(c[:, 0]), "c1": ep_col(c[:, 1]), "c2": ep_col(c[:, 2]), "b": ep_col(b)}
        Dh, Eh = horner_scale_constants(torch.stack([c[:, 2], c[:, 1], c[:, 0]], dim=-1), inter, eps)  # [384, 3]

        def stack_rows(v):  # [384, 3] -> [dp * 3, 96, 1, 1]
            v = v.reshape(dp, n_exp // dp, 3).permute(0, 2, 1)  # [dp, 3, 96]
            return v.reshape(dp * 3, n_exp // dp, 1, 1).contiguous()

        out["D"], out["E"] = stack_rows(Dh), stack_rows(Eh)
        return out

    @classmethod
    def from_coefficients(cls, mesh_device, cfg: MotifTTConfig, c: torch.Tensor, b: torch.Tensor, **kw):
        return cls(mesh_device, cfg, c, b, **kw)

    @classmethod
    def from_source(cls, mesh_device, cfg: MotifTTConfig, layer_idx: int, *, source, cache: bool = True, **kw):
        """Reads ``model.layers.{l}.moe.experts.act_fn.weight [384, 3]`` / ``.bias [384, 1]`` (sigmoid, clamp +-0.5)
        -- lazily: with ``cache=True`` and every constant cached, the source is never touched."""
        check_polynorm_semantics(cfg)

        def coeffs():
            return W.polynorm_coefficients(
                source.get(W.hf_name(layer_idx, "moe.experts.act_fn.weight")),
                source.get(W.hf_name(layer_idx, "moe.experts.act_fn.bias")),
                bias_clamp=cfg.polynorm_bias_clamp,
            )

        return cls(mesh_device, cfg, coeffs, cache=cache, layer=layer_idx, **kw)

    def as_dict(self, mode: str = "fp32"):
        _check_mode(mode)
        return dict(self.c[mode])

    def deallocate(self) -> None:
        for d in self.c.values():
            _free(*d.values())
        _free(self.D, self.E)


# ============================================================================================================
# Kernels (composites)
# ============================================================================================================
def _to_dtype(t, dtype, mc):
    """``t`` in ``dtype`` (a new tensor, or ``t`` itself when it already is) + whether a copy was made."""
    if t.dtype == dtype:
        return t, False
    return ttnn.typecast(t, dtype, memory_config=mc), True


MOMENT_IMPLS = ("sum", "sum_fast", "pre_ag")


def _moment_sums(gm, *, stack_dim: int, mc, ckc, concat_full: Optional[bool] = None, impl: str = "sum"):
    """``[sum g^2, sum g^4, sum g^6]`` of ``gm [.., T, n]`` (fp32 or bf16 values) over the last dim, stacked on
    ``stack_dim``: ``[.., 3 at stack_dim, .., T, w]`` fp32 (``w`` = 1, or 32 for ``impl="pre_ag"`` with the sum in
    column 0; every consumer -- :func:`_ar_moments` (both impls) and :func:`_slice_dim` -- reads column 0 only). The
    squares are formed in fp32 on the SFPU (exact for bf16 inputs).

    ``impl``: "sum" = ``ttnn.sum`` accurate fp32 SFPU reduce (default); "sum_fast" = the FPU reduce (operands
    truncated to TF32: ~5e-4 relative bias on the moments, ~2.5e-4 on PolyNorm); "pre_ag" =
    ``rms_norm_pre_all_gather`` (accurate SFPU path) on the stacked ``[g, g^2, g^3]`` (fused square + sum).
    Decode-sized inputs (one tile row) concat first and reduce once (fewer ops); larger inputs reduce each power and
    concat the small sums (less traffic)."""
    if impl not in MOMENT_IMPLS:
        raise ValueError(f"moment impl must be one of {MOMENT_IMPLS}, got {impl!r}")
    rows = int(gm.shape[-2])
    if concat_full is None:
        concat_full = rows <= 32
    g2 = ttnn.multiply(gm, gm, dtype=ttnn.float32, memory_config=mc)
    if impl == "pre_ag":
        gf, gf_new = _to_dtype(gm, ttnn.float32, mc)
        g3 = ttnn.multiply(g2, gf, memory_config=mc)
        z = ttnn.concat([gf, g2, g3], dim=stack_dim, memory_config=mc)
        _free(g2, g3)
        if gf_new:
            _free(gf)
        s = ttnn.rms_norm_pre_all_gather(z, dtype=ttnn.float32, compute_kernel_config=ckc, memory_config=mc)
        _free(z)
        return s
    fast = impl == "sum_fast"
    g4 = ttnn.multiply(g2, g2, memory_config=mc)
    g6 = ttnn.multiply(g4, g2, memory_config=mc)
    if concat_full:
        q = ttnn.concat([g2, g4, g6], dim=stack_dim, memory_config=mc)
        _free(g2, g4, g6)
        s = ttnn.sum(q, dim=-1, keepdim=True, memory_config=mc, compute_kernel_config=ckc,
                     fast_and_approximate_mode=fast)
        _free(q)
        return s
    sums = []
    for p in (g2, g4, g6):
        sums.append(ttnn.sum(p, dim=-1, keepdim=True, memory_config=mc, compute_kernel_config=ckc,
                             fast_and_approximate_mode=fast))
        _free(p)
    s = ttnn.concat(sums, dim=stack_dim, memory_config=mc)
    _free(*sums)
    return s


AR_IMPLS = ("all_reduce", "ag_sum")


def _ar_ag_sum(s, ccl, *, mc, ckc):
    """Exact fp32 TP all-reduce of moment sums ``s [.., T, w]`` held in column 0: zero-pad the width-1 sums to one
    tile -> native (tile aligned) ``all_gather(dim=-1, "tp")`` -> accurate fp32 ``ttnn.sum(dim=-1)`` (sums 8 values +
    31 x 8 zeros). ``w = 1`` (moments "sum" / "sum_fast"), or ``w = 32`` (moments "pre_ag": ``rms_norm_pre_all_gather``
    stats are one tile wide with the sum in column 0; that column is sliced out first, +1 op).

    ``ttnn.pad`` within the tile padding is an in-place fill + view of its input (``pad.cpp`` ``invoke_tile``), so
    ``sp`` aliases ``s`` / ``s1``; deallocating both is safe (buffer deallocation is idempotent)."""
    shape = [int(d) for d in s.shape]
    if shape[-1] not in (1, TILE):
        raise ValueError(f"ag_sum expects moment sums of width 1 or {TILE} (sum in column 0), got {shape[-1]}")
    s1 = s
    if shape[-1] != 1:
        s1 = ttnn.slice(s, [0] * len(shape), shape[:-1] + [1], memory_config=mc)
    pad = [(0, 0)] * (len(shape) - 1) + [(0, TILE - 1)]
    sp = ttnn.pad(s1, pad, 0.0, memory_config=mc)
    gat = ccl.all_gather(sp, len(shape) - 1, "tp", memory_config=mc)
    _free(sp)
    if s1 is not s:
        _free(s1)
    out = ttnn.sum(gat, dim=-1, keepdim=True, memory_config=mc, compute_kernel_config=ckc)
    _free(gat)
    return out


def _ar_moments(s, ccl, *, exact: bool, mc, impl: str = "all_reduce", ckc=None):
    """All-reduce the moment sums ``s [.., 3, T, w]`` over TP (``w`` = 1, or 32 for ``moments="pre_ag"``; the sums are
    in column 0).

    * ``"all_reduce"``: ``ttnn.all_reduce``. Decode-sized payloads (T <= 32) take its all-gather + local-sum path
      (exact fp32). Larger ones would take the reduce-scatter path, whose fp32 adds are TF32-class; with ``exact``
      they go through ``"ag_sum"`` instead.
    * ``"ag_sum"``: :func:`_ar_ag_sum` (exact for any T; 3 ops instead of the composite all-reduce's ~7)."""
    if ccl is None:
        return s
    if impl not in AR_IMPLS:
        raise ValueError(f"ar impl must be one of {AR_IMPLS}, got {impl!r}")
    if impl == "ag_sum" or (exact and int(s.shape[-2]) > 32):
        return _ar_ag_sum(s, ccl, mc=mc, ckc=ckc)
    return ccl.ar_tp(s, memory_config=mc)


def _slice_dim(t, dim: int, k: int, mc):
    """``t[.., k:k+1 at dim, .., :, 0:1]`` (the per-row scale of moment ``k``; stats tiles keep it in column 0)."""
    shape = [int(x) for x in t.shape]
    start = [0] * len(shape)
    end = list(shape)
    start[dim], end[dim] = k, k + 1
    end[-1] = 1
    return ttnn.slice(t, start, end, memory_config=mc)


HORNER_IMPLS = ("mac", "binary")


def _horner(gx, a, b, *, stack_dim: int, mc, impl: str = "mac"):
    """``a3 g^3 + a2 g^2 + a1 g + b`` with ``a [.., 3 at stack_dim, .., T, 1]`` (k = 1, 2, 3) and ``b`` broadcast.
    ``impl``: "mac" = 3 ternary ``ttnn.mac`` (fewest ops); "binary" = 6 binary_ng ops (each ~4x cheaper on device
    than a ternary mac at decode sizes)."""
    if impl not in HORNER_IMPLS:
        raise ValueError(f"horner impl must be one of {HORNER_IMPLS}, got {impl!r}")
    a1, a2, a3 = (_slice_dim(a, stack_dim, k, mc) for k in range(3))
    if impl == "mac":
        t = ttnn.mac(gx, a3, a2, memory_config=mc)
        _free(a3, a2)
        t2 = ttnn.mac(t, gx, a1, memory_config=mc)
        _free(t, a1)
        t3 = ttnn.mac(t2, gx, b, memory_config=mc)
        _free(t2)
        return t3
    t = ttnn.multiply(gx, a3, memory_config=mc)
    _free(a3)
    t2 = ttnn.add(t, a2, memory_config=mc)
    _free(t, a2)
    t = ttnn.multiply(t2, gx, memory_config=mc)
    _free(t2)
    t2 = ttnn.add(t, a1, memory_config=mc)
    _free(t, a1)
    t = ttnn.multiply(t2, gx, memory_config=mc)
    _free(t2)
    t2 = ttnn.add(t, b, memory_config=mc)
    _free(t)
    return t2


def _poly_horner(g, consts_D, consts_E, b_by_mode, *, mode: str, stack_dim: int, ccl, exact_ar: bool, mc, ckc,
                 concat_full: Optional[bool] = None, moments: str = "sum", g_stats=None, horner: str = "mac",
                 ar: str = "all_reduce"):
    """Shared core of the scalar (TP / local) and grouped Horner PolyNorm: returns ``poly`` (no ``* up``).

    The moments always come from exact fp32 squares (``multiply(g, g, dtype=float32)``: a bf16 x bf16 product is
    exact in fp32), so "bf16" mode only rounds the Horner chain, never the statistics."""
    md = _math_dtype(mode)
    gx, gx_new = _to_dtype(g, md, mc)  # Horner operand in the math dtype
    gm = gx if mode == "fp32" else g  # moment source (fp32 or bf16 values; squares are formed in fp32)
    if g_stats is not None:
        gm = g_stats
    s = _moment_sums(gm, stack_dim=stack_dim, mc=mc, ckc=ckc, concat_full=concat_full, impl=moments)
    s2 = _ar_moments(s, ccl, exact=exact_ar, mc=mc, impl=ar, ckc=ckc)
    if s2 is not s:
        _free(s)
    v = ttnn.mac(s2, consts_D, consts_E, memory_config=mc)
    _free(s2)
    a = ttnn.rsqrt(v, memory_config=mc)
    _free(v)
    if md != ttnn.float32:
        a16 = ttnn.typecast(a, md, memory_config=mc)
        _free(a)
        a = a16
    t = _horner(gx, a, b_by_mode[mode], stack_dim=stack_dim, mc=mc, impl=horner)
    _free(a)
    if gx_new:
        _free(gx)
    return t


def _times_up(t, u, *, out_dtype, mc):
    h = ttnn.multiply(t, u, dtype=out_dtype, memory_config=mc)
    return h


def polynorm_tp(
    g,
    u,
    consts: ScalarPolyNormConsts,
    *,
    ccl=None,
    mode: str = "fp32",
    out_dtype=ttnn.bfloat16,
    memory_config=None,
    compute_kernel_config=None,
    exact_ar: bool = True,
    concat_full: Optional[bool] = None,
    moments: str = "sum",
    g_stats=None,
    horner: str = "mac",
    ar: str = "ag_sum",
    intermediate_memory_config=None,
):
    """Scalar PolyNorm ``h = poly(g) * u`` on ``g, u [1, 1, T, n]`` (TILE; bf16 or fp32) -> ``[1, 1, T, n]``
    ``out_dtype`` (bf16).

    ``g_stats`` (optional, ``[1, 1, T, consts.inter]``): the FULL gate, from which the moments are taken locally
    (exact, no CCL; e.g. a replicated gate projection) while the Horner chain runs on the local shard ``g``; ``ccl``
    is then ignored.

    ``ccl`` (a :class:`MotifCCL`): the intermediate is TP-sharded (``n = consts.inter / 8``) and the three moments are
    all-reduced over TP (one ``[1, 3, T, 1]`` fp32 CCL). ``ccl=None``: local statistics (``n = consts.inter``).
    ``mode``: "fp32" (fp32 intermediates) | "bf16" (bf16 Horner, fp32 moments). ``compute_kernel_config``: the
    ``polynorm`` role (HiFi4 + fp32 dest acc, required for the accurate fp32 reduce). ``memory_config``: the output
    (default DRAM); intermediates: ``intermediate_memory_config``, else ``memory_config``, else L1 for T <= 32.
    ``moments`` / ``horner`` / ``ar``: implementation knobs (module docstring). Inputs are not consumed."""
    _check_mode(mode)
    out_mc, mc = _memory_configs(int(g.shape[-2]), memory_config, intermediate_memory_config)
    n = int(g.shape[-1])
    if g_stats is not None:
        if int(g_stats.shape[-1]) != consts.inter:
            raise ValueError(f"g_stats width {int(g_stats.shape[-1])} != intermediate {consts.inter}")
        t = _poly_horner(g, consts.D, consts.E, consts.b, mode=mode, stack_dim=1, ccl=None, exact_ar=False, mc=mc,
                         ckc=compute_kernel_config, concat_full=concat_full, moments=moments, g_stats=g_stats,
                         horner=horner)
        h = _times_up(t, u, out_dtype=out_dtype, mc=out_mc)
        _free(t)
        return h
    if ccl is None and n != consts.inter:
        raise ValueError(f"local-stat PolyNorm needs the full intermediate ({consts.inter}), got width {n}")
    if ccl is not None and n * ccl.axis_size("tp") != consts.inter:
        raise ValueError(f"TP PolyNorm: width {n} x tp {ccl.axis_size('tp')} != intermediate {consts.inter}")
    t = _poly_horner(g, consts.D, consts.E, consts.b, mode=mode, stack_dim=1, ccl=ccl, exact_ar=exact_ar, mc=mc,
                     ckc=compute_kernel_config, concat_full=concat_full, moments=moments, horner=horner, ar=ar)
    h = _times_up(t, u, out_dtype=out_dtype, mc=out_mc)
    _free(t)
    return h


def _grouped_rms(g, u, c, *, eps: float, ckc, mc, out_dtype, out_mc=None):
    """G6 composite with fused multiply-adds: ``(c0 N(g^3) + c1 N(g^2) + c2 N(g) + b) * u``, ``N`` = weightless
    rms_norm (FPU statistics, TF32-class)."""
    g2 = ttnn.multiply(g, g, memory_config=mc)
    g3 = ttnn.multiply(g2, g, memory_config=mc)
    n3 = ttnn.rms_norm(g3, epsilon=eps, compute_kernel_config=ckc, memory_config=mc)
    _free(g3)
    t = ttnn.mac(n3, c["c0"], c["b"], memory_config=mc)
    _free(n3)
    n2 = ttnn.rms_norm(g2, epsilon=eps, compute_kernel_config=ckc, memory_config=mc)
    _free(g2)
    t2 = ttnn.mac(n2, c["c1"], t, memory_config=mc)
    _free(n2, t)
    n1 = ttnn.rms_norm(g, epsilon=eps, compute_kernel_config=ckc, memory_config=mc)
    t3 = ttnn.mac(n1, c["c2"], t2, memory_config=mc)
    _free(n1, t2)
    h = ttnn.multiply(t3, u, dtype=out_dtype, memory_config=out_mc or mc)
    _free(t3)
    return h


def grouped_polynorm(
    gu,
    consts: Union[GroupedPolyNormConsts, Mapping[str, object]],
    *,
    inter: Optional[int] = None,
    mode: str = "fp32",
    eps: float = 1e-6,
    compute_kernel_config=None,
    memory_config=None,
    out_dtype=ttnn.bfloat16,
    impl: Optional[str] = None,
    up=None,
    moments: str = "sum",
    horner: str = "mac",
    intermediate_memory_config=None,
):
    """Routed-expert PolyNorm * up: ``gu [1, E, M, 2 inter]`` (gate ``[..., :inter]`` | up ``[..., inter:]``, the G6
    gate_up matmul output, bf16 or fp32) -> ``h [1, E, M, inter]`` in ``out_dtype`` (bf16, never block float).
    Alternatively pass the gate as ``gu`` and the up half as ``up=`` (both ``[1, E, M, inter]``).

    ``consts``: :class:`GroupedPolyNormConsts` (``impl`` "horner" (default) or "rms"), or a mapping with ``c0, c1, c2,
    b`` ``[1, E, 1, 1]`` tensors in the math dtype of ``mode`` (tt/moe.py's dict; ``impl`` "rms" only). ``mode``:
    "fp32" | "bf16" intermediates. ``eps`` is used by ``impl="rms"`` (the Horner constants carry their own eps).
    ``memory_config``: the output (default DRAM); intermediates: ``intermediate_memory_config``, else
    ``memory_config``, else L1 for M <= 32 (decode; 2-3x faster than DRAM). Same call signature as tt/moe.py's local
    ``grouped_polynorm``; consumes nothing."""
    _check_mode(mode)
    out_mc, mc = _memory_configs(int(gu.shape[-2]), memory_config, intermediate_memory_config)
    is_consts = isinstance(consts, GroupedPolyNormConsts)
    if impl is None:
        impl = "horner" if is_consts else "rms"
    if impl not in POLYNORM_IMPLS:
        raise ValueError(f"impl must be one of {POLYNORM_IMPLS}, got {impl!r}")
    if impl == "horner" and not is_consts:
        raise ValueError("impl='horner' needs GroupedPolyNormConsts (it carries the D / E Horner constants)")

    if up is None:
        B, E, M, N = (int(s) for s in gu.shape)
        if inter is None:
            inter = N // 2
        if N != 2 * inter:
            raise ValueError(f"gate_up width {N} != 2 x {inter}")
        g = ttnn.slice(gu, [0, 0, 0, 0], [B, E, M, inter], memory_config=mc)
        u = ttnn.slice(gu, [0, 0, 0, inter], [B, E, M, 2 * inter], memory_config=mc)
        own_gu = True
    else:
        g, u = gu, up
        own_gu = False
    if is_consts and int(g.shape[-1]) != consts.inter:
        raise ValueError(f"grouped PolyNorm width {int(g.shape[-1])} != constants' intermediate {consts.inter}")

    md = _math_dtype(mode)
    if impl == "rms":
        c = consts.as_dict(mode) if is_consts else consts
        gx, gx_new = _to_dtype(g, md, mc)
        # up keeps its dtype: the final fp32 x bf16 multiply is exact in fp32 (validated equal to a typecast up)
        h = _grouped_rms(gx, u, c, eps=eps, ckc=compute_kernel_config, mc=mc, out_dtype=out_dtype, out_mc=out_mc)
        if gx_new:
            _free(gx)
    else:
        t = _poly_horner(g, consts.D, consts.E, {k: v["b"] for k, v in consts.c.items()}, mode=mode, stack_dim=0,
                         ccl=None, exact_ar=False, mc=mc, ckc=compute_kernel_config, moments=moments, horner=horner)
        h = _times_up(t, u, out_dtype=out_dtype, mc=out_mc)
        _free(t)
    if own_gu:
        _free(g, u)
    return h


__all__ = [
    "AR_IMPLS",
    "GroupedPolyNormConsts",
    "HORNER_IMPLS",
    "MOMENT_IMPLS",
    "POLYNORM_IMPLS",
    "POLYNORM_MODES",
    "PolyNormCoefficients",
    "ScalarPolyNormConsts",
    "check_polynorm_semantics",
    "grouped_polynorm",
    "horner_scale_constants",
    "polynorm_output_scale",
    "polynorm_tp",
]
