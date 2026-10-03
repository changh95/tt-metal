# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact on-device sampling for Motif-3 (vLLM semantics) inside the decode trace.

Design, measurements and the integration recipe: ``docs/sampling/DEVICE_SAMPLER.md`` (decision record:
``docs/sampling/MEASURE_AND_CONTRACT.md``). Built from stock ttnn ops only (no C++ change, no ``_ttnn.so`` rebuild):
``ttnn.sampling`` keeps the global top 32, computes in bf16 and draws with an 8-bit uniform, so it truncates Motif's
top-p 0.95 nucleus and must not be used for Motif.

Semantics (per lane, vLLM 0.26 ``Sampler`` = ``apply_temperature`` -> ``apply_top_k_top_p`` -> random sample)
--------------------------------------------------------------------------------------------------------------
* **greedy** (``T < 1e-5``, vLLM ``_SAMPLING_EPS``): the lowest token id among the maxima of the raw logits, bitwise
  ``torch.argmax`` of the row (the same rule as ``MotifLMHead.argmax_decode``).
* **temperature** first: probabilities ``p = softmax(x / T)`` over the **full** vocabulary in fp32, as
  ``exp((x - M) * (1 / T)) / Z`` with the exact full-vocab normaliser ``Z`` (one fp32 partial sum per chip).
* **top-k** (``1 <= k < vocab``): keep every token whose logit is ``>=`` the k-th largest logit (vLLM keeps all ties of
  the k-th value: ``logits_sort < kth`` is masked).
* **top-p** on the top-k-renormalised distribution: walk the tokens by logit descending and keep rank ``r`` iff its
  exclusive prefix mass is ``< top_p * mass(top-k)`` (vLLM's ascending ``cumsum <= 1 - p`` mask in descending form);
  the first token is always kept (vLLM's "at least one"; ``top_k == 1`` arrives with ``top_p = 0``).
* **draw**: inverse CDF over the kept tokens in walk order: ``token = kept[#{j : C_j <= u * mass_kept}]`` with a 24-bit
  uniform ``u``. Tokens with identical logits at the top-p boundary are ordered by ``ttnn.topk`` (deterministic,
  unspecified); vLLM orders them with an unstable ``torch.sort``, so any fixed tie order implements its semantics
  (MEASURE_AND_CONTRACT §1.6). Inside the kept set ties do not change the distribution.

RNG (counter based, slot addressable, no hidden device state)
-------------------------------------------------------------
``u = (lowbias32(ctr) >> 8) * 2^-24`` with the per-lane, per-step counter ``ctr = key + pos * 0x9E3779B1 (mod 2^32)``
written by :meth:`MotifDeviceSampler.set_positions` (``pos`` = the step's input position, i.e. the decode
``start_pos``; the sampled token is the one for ``pos + 1``). Seeded requests: ``key = seed_key(seed)`` (splitmix64 of
the 64-bit seed), so a lane's draw is a pure function of ``(seed, position)``: identical across lanes, batches,
replays, traced / eager and lane relocations. Unseeded requests (``seed=None``): a fresh host-random counter every
step (vLLM's unseeded sampling is not reproducible either). The hash runs on device (``lowbias32``: 3 xor-shifts, 2
exact int32 multiplies); :func:`uniform_from_counter` is its bit-exact host twin (used by the host fallback).

Device pipeline (every op stock ttnn, all parameters persistent device tensors, fixed shapes, trace safe)
-------------------------------------------------------------------------------------------------------
Input: the LM head's decode logits ``[1, 1, 32, 6880]`` bf16 TILE on every chip ("mesh" vocab split: rows = the 32
lanes in lane order, chip (r, c) holds vocab block ``b = r * C + c``; a ROW_MAJOR input is tilized first).

1. local candidates: ``topk(logits, K)`` per chip (``K = local_k`` = 64), ids ``6880 b + i`` as fp32 (exact < 2^24),
   all-gathered over TP then DP -> values ``V [32, 32 K]`` and ids ``I``; ``M = max V`` (the global max).
2. normalisers: per chip ``s = sum exp((x - M) / T)`` over its 6880 logits and, with ``logprobs``, ``s1 = sum exp(x -
   M)`` (raw, T = 1), both as one accurate fp32 SFPU reduction of the two exponent rows stacked; the partials of all
   chips are all-gathered (one pair) and summed exactly: ``Z``, ``Z1``.
3. window: ``topk(V, W)`` (``W = window`` = 512, sorted) -> ``SV`` and the columns ``sidx``; ``PR = exp((SV - M) / T) /
   Z`` (full-vocab probabilities).
4. exclusive prefix sums ``EX = PR @ striu(1)`` as two bf16 matmuls on ``PR ~ hi + mid``. The FPU accumulates a
   K-tile below fp32 precision (measured: 2^-13 on a probe row, >= 8.7e-4 in real decoding), so ``EX`` only
   PROPOSES the two decisions below; step 7 re-checks both with exact fp32 SFPU sums.
5. masks: ``kth`` (one-hot of ``k - 1``), ``keep_k = SV >= kth``, ``p_eff = top_p * mass(keep_k)``, ``keep = (EX <
   p_eff) * keep_k``, ``n_kept``, ``mass_kept`` (exact SFPU sum); then one exact repair step: the first dropped token
   is kept when its exact exclusive prefix ``mass_kept`` is still ``< p_eff`` (the FPU prefix runs high).
6. draw: ``j = min(#{EX <= u * mass_kept}, n_kept) - 1``, column ``sidx[j]`` -> id (two one-hot reductions); greedy
   lanes: ``min(I + 2^24 [V < M])``; blended per lane.
7. certificate (the coverage check) -- ``flag = 1`` unless all of:
   * every token at or above the lane's decisive value is a candidate inside the window: ``max(every chip's K-th
     candidate, the window's last value) < v`` with ``v`` = the max (greedy), the k-th value (top-k) or the last kept
     value (top-p);
   * the walk ends inside the window: ``k <= W`` (top-k) or window mass ``>= p_eff + 2^-20`` (top-p; so ``top_p = 1``
     without top-k, whose support is the whole vocabulary, is always flagged);
   * the decisions are exact (sampled lanes): the last kept token's exclusive prefix ``mass_kept - PR[n - 1] <
     p_eff``, the first dropped one's ``mass_kept >= p_eff`` (unless the top-k set is exhausted), and ``EX[j] <=
     u * mass_kept < EX[j] + PR[j]`` with ``EX[j] = sum(PR[:j])`` -- all exact fp32 SFPU reductions.
   A flagged lane's device token must be replaced (:meth:`MotifDeviceSampler.resolve`: exact host resample from the
   logits with the same ``u``).
8. outputs: ``tokens [1, 1, 1, 32]`` uint32 ROW_MAJOR (lane order, identical on every chip; the ``argmax_decode``
   contract) and ``info [1, 1, 8, 32]`` fp32 ROW_MAJOR (rows :data:`INFO_ROWS`: token, flag, raw logprob, n_kept, u,
   mass_kept, window mass, k-th value), read from one chip with one small transfer (:meth:`read`); optional per-lane
   cumulative flag counter (:meth:`coverage_counts`).

Exactness (DEVICE_SAMPLER.md §3-§4): on every certified lane the kept set equals the fp64 nucleus (up to boundary ties)
and the token equals the fp64 inverse CDF fed the same ``u``. Candidate-coverage flags at K = 64 (all saved real rows
through the host twin :func:`emulate_device`): 0.029 % of the rows of Motif's own T=1.0 traces (~0.9 % of 32-lane
steps), 0 at T = 0.6, 0.76 % of teacher-forced assistant rows at T = 1.0. In the real 53-layer decode trace all flags
together (coverage + exactness checks, after the cut repair) hit 0.04 % of the lane-steps with ``top_p < 1``.

Import rule (README §13): stdlib, numpy, torch, ttnn and the motif3 ``tt/`` infra only; the host functions are pure
torch / numpy (CPU-tested without a device).
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import ttnn

from .ccl import MotifCCL
from .lm_head import HostShardReader, vocab_block_of_coord, vocab_per_shard
from .model_config import TILE, MotifTTConfig

# ============================================================================================================
# constants
# ============================================================================================================
SAMPLING_EPS = 1e-5  # vLLM v1 Sampler._SAMPLING_EPS: temperature below it -> greedy
GOLDEN32 = 0x9E3779B1  # Weyl increment of the per-lane counter: ctr = key + pos * GOLDEN32 (mod 2^32)
LOWBIAS_M1 = 0x21F0AAAD  # lowbias32 (C. Wellons) multipliers
LOWBIAS_M2 = 0x735A2D97
U24 = 2.0**-24
MASK32 = 0xFFFFFFFF
DEFAULT_LOCAL_K = 64
DEFAULT_WINDOW = 512
LOCAL_K_CHOICES = (32, 64, 128)
WINDOW_CHOICES = (256, 512, 1024, 2048)
# The certificate requires the window mass to exceed p_eff by this margin (fp32 rounding of a 512-term sum is
# ~1e-7; a top_p of 1.0 without top-k can then never be certified: its support is the whole vocabulary).
CERT_MARGIN = 2.0**-20
# top_p is clamped from below so that the first token is always kept (vLLM: "at least one"; top_k == 1 -> top_p 0).
MIN_TOP_P = 2.0**-60
# top_p >= 1 (off) is sent to the device as 2: every exclusive prefix (<= the top-k mass) is below 2 * mass_k, so the
# whole top-k set is kept, and a lane without top-k is never certified (its support is the whole vocabulary).
TOP_P_OFF = 2.0
BIG = 3.0e38  # finite "infinity" for masks (fp32)
ID_SENTINEL = float(1 << 24)  # > every vocab id, exact in fp32: "not a maximum" in the greedy min
INFO_ROWS = ("token", "flag", "logprob", "n_kept", "u", "mass_kept", "window_mass", "kth")
INFO_INDEX = {name: i for i, name in enumerate(INFO_ROWS)}


# ============================================================================================================
# host side: RNG (bit-exact twin of the device hash)
# ============================================================================================================
def lowbias32(x) -> np.ndarray:
    """lowbias32 integer hash on uint32 (numpy, vectorized): ``x ^= x >> 16; x *= 0x21f0aaad; x ^= x >> 15;
    x *= 0x735a2d97; x ^= x >> 15`` (mod 2^32). The device runs the same ops in int32 (exact SFPU ``mul_int32``,
    logical right shifts)."""
    h = np.asarray(x, dtype=np.uint64) & MASK32
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(LOWBIAS_M1)) & MASK32
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(LOWBIAS_M2)) & MASK32
    h ^= h >> np.uint64(15)
    return h.astype(np.uint32)


def uniform_from_counter(ctr) -> np.ndarray:
    """The device uniform for counters ``ctr`` (uint32): ``(lowbias32(ctr) >> 8) * 2^-24`` as float32 in [0, 1), 24
    random bits. Bit-exact with :meth:`MotifDeviceSampler.sample`'s ``u``."""
    return (lowbias32(ctr) >> np.uint32(8)).astype(np.float32) * np.float32(U24)


def seed_key(seed: int) -> int:
    """32-bit key of a request seed (any Python int; vLLM seeds are 64-bit): the high half of splitmix64(seed mod
    2^64). Deterministic and well mixed, so neighbouring seeds give unrelated lanes."""
    z = (int(seed) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    return int(z >> 32)


def lane_counter(key: int, position: int) -> int:
    """The per-step counter of a seeded lane: ``(key + position * 0x9E3779B1) mod 2^32``."""
    return (int(key) + int(position) * GOLDEN32) & MASK32


def _i32(x: int) -> int:
    """uint32 bit pattern -> the int32 with the same bits (device int32 tensors)."""
    x &= MASK32
    return x - (1 << 32) if x >= (1 << 31) else x


# ============================================================================================================
# host side: per-lane parameters (vLLM / vllm-tt-plugin conventions)
# ============================================================================================================
@dataclass(frozen=True)
class LaneParams:
    """Normalised sampling parameters of one lane.

    ``top_k = 0`` means top-k off (the plugin sends ``vocab_size`` for "off"; ``<= 0`` / ``None`` / ``>= vocab`` are
    off too). ``top_p`` is in ``[0, 1]`` (1 = off). ``seed`` None = unseeded. ``greedy`` iff ``temperature < 1e-5``
    (empty / padding slots arrive as ``temperature 0, top_k 1, top_p 1``: greedy)."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return not (self.temperature >= SAMPLING_EPS)

    @property
    def inv_temperature(self) -> float:
        return 1.0 if self.greedy else 1.0 / float(self.temperature)

    @property
    def top_k_active(self) -> bool:
        return (not self.greedy) and self.top_k > 0

    @property
    def full_support(self) -> bool:
        """Sampled without top-k and ``top_p >= 1``: the support is the whole vocabulary (never device certified)."""
        return (not self.greedy) and self.top_k <= 0 and self.top_p >= 1.0


GREEDY_LANE = LaneParams()


def normalize_lane(temperature=None, top_p=None, top_k=None, seed=None, *, vocab_size: int) -> LaneParams:
    """One lane's raw values (plugin ``TTSamplingParams`` entries) -> :class:`LaneParams`.

    ``temperature`` None -> 1.0 (vLLM's default), must be finite and >= 0; ``top_p`` None -> 1.0, must lie in
    ``[0, 1]``; ``top_k`` None / ``<= 0`` / ``>= vocab_size`` -> off; ``seed`` int or None."""
    t = 1.0 if temperature is None else float(temperature)
    if not math.isfinite(t) or t < 0:
        raise ValueError(f"temperature must be finite and >= 0, got {temperature!r}")
    p = 1.0 if top_p is None else float(top_p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"top_p must lie in [0, 1], got {top_p!r}")
    k = 0 if top_k is None else int(top_k)
    if k <= 0 or k >= int(vocab_size):
        k = 0
    s = None if seed is None else int(seed)
    return LaneParams(temperature=t, top_p=p, top_k=k, seed=s)


def normalize_lanes(temperature, top_p, top_k, seeds, *, vocab_size: int, num_lanes: int) -> List[LaneParams]:
    """Lane-ordered sequences (lists, tuples, tensors or None = all default) -> ``num_lanes`` :class:`LaneParams`."""

    def seq(v, name):
        if v is None:
            return [None] * num_lanes
        if isinstance(v, torch.Tensor):
            v = v.reshape(-1).tolist()
        v = list(v)
        if len(v) != num_lanes:
            raise ValueError(f"{name} has {len(v)} entries, expected {num_lanes} (one per lane)")
        return v

    T, P, K, S = seq(temperature, "temperature"), seq(top_p, "top_p"), seq(top_k, "top_k"), seq(seeds, "seeds")
    return [normalize_lane(T[i], P[i], K[i], S[i], vocab_size=vocab_size) for i in range(num_lanes)]


def lane_lists_from_rows(
    sampling_params, lanes: Sequence[int], *, num_lanes: int = 32
) -> Tuple[list, list, list, list]:
    """Bridge helper: the plugin's per-ROW ``TTSamplingParams`` (duck-typed: ``temperature``, ``top_p``, ``top_k``,
    ``seed`` sequences of length B, the rows of the step) and the lane of each row -> lane-ordered ``(temperature,
    top_p, top_k, seeds)`` lists of length ``num_lanes`` for :meth:`MotifDeviceSampler.set_params`. Lanes without a row
    get the padding-slot defaults (greedy: temperature 0, top_p 1, top_k 1, seed None)."""

    def col(name):
        v = getattr(sampling_params, name)
        return v.reshape(-1).tolist() if isinstance(v, torch.Tensor) else list(v)

    T, P, K, S = col("temperature"), col("top_p"), col("top_k"), col("seed")
    if not (len(T) == len(P) == len(K) == len(S) == len(lanes)):
        raise ValueError(f"sampling params ({len(T)}, {len(P)}, {len(K)}, {len(S)}) do not match {len(lanes)} rows")
    t, p, k, sd = [0.0] * num_lanes, [1.0] * num_lanes, [1] * num_lanes, [None] * num_lanes
    for row, lane in enumerate(lanes):
        t[lane], p[lane], k[lane], sd[lane] = T[row], P[row], K[row], S[row]
    return t, p, k, sd


def device_param_columns(lanes: Sequence[LaneParams], window: int) -> Dict[str, torch.Tensor]:
    """The per-lane device parameter columns (float32 ``[n]`` each) of :class:`MotifDeviceSampler`:

    * ``greedy`` 0/1; ``inv_t`` 1/T (1 for greedy lanes, which keeps the normaliser finite);
    * ``top_p`` clamped to ``[MIN_TOP_P, 1)`` (the first token is always kept); ``top_p >= 1`` (top-p off) becomes
      :data:`TOP_P_OFF` = 2, so the walk keeps the whole top-k set: the fp32 exclusive prefix saturates at 1.0 and
      would otherwise drop the top-k tokens below fp32 resolution that vLLM keeps (its ascending cumsum is > 0);
      greedy lanes get ``MIN_TOP_P`` (their certificate's window-mass condition then always holds);
    * ``k_active`` 0/1 (sampled lane with top-k); ``k_idx`` = ``min(k, W) - 1`` (0 when off);
    * ``k_in_win`` 0/1 (top-k active and ``k <= W``; a larger k is flagged for the host)."""
    W = int(window)
    g = torch.tensor([1.0 if l.greedy else 0.0 for l in lanes], dtype=torch.float32)
    inv_t = torch.tensor([l.inv_temperature for l in lanes], dtype=torch.float64).to(torch.float32)
    top_p = torch.tensor(
        [MIN_TOP_P if l.greedy else (TOP_P_OFF if l.top_p >= 1.0 else max(l.top_p, MIN_TOP_P)) for l in lanes],
        dtype=torch.float32,
    )
    ka = torch.tensor([1.0 if l.top_k_active else 0.0 for l in lanes], dtype=torch.float32)
    kidx = torch.tensor([float(min(l.top_k, W) - 1) if l.top_k_active else 0.0 for l in lanes], dtype=torch.float32)
    kin = torch.tensor([1.0 if (l.top_k_active and l.top_k <= W) else 0.0 for l in lanes], dtype=torch.float32)
    return {"greedy": g, "inv_t": inv_t, "top_p": top_p, "k_active": ka, "k_idx": kidx, "k_in_win": kin}


def k_offset_rows(cols: Dict[str, torch.Tensor], window: int) -> torch.Tensor:
    """``[n, W]`` float32 additive mask of the top-k threshold: ``0`` at column ``k_idx`` of a top-k lane, ``-BIG``
    everywhere else, so ``kth = max(SV + k_ofs)`` is the k-th window value (top-k lanes) or ``-BIG`` (keep all)."""
    n = int(cols["k_active"].numel())
    ofs = torch.full((n, int(window)), -BIG, dtype=torch.float32)
    act = cols["k_active"] > 0
    rows = torch.nonzero(act).reshape(-1)
    ofs[rows, cols["k_idx"][rows].long()] = 0.0
    return ofs


# ============================================================================================================
# host side: exact reference / fallback sampler (fp64, vLLM semantics)
# ============================================================================================================
def greedy_token(row: torch.Tensor) -> int:
    """``torch.argmax`` of a logits row (lowest index among the maxima)."""
    return int(torch.argmax(row.reshape(-1).float()))


def sorted_order(row: torch.Tensor) -> torch.Tensor:
    """The reference walk order of a logits row: logit descending, ties by token id ascending."""
    return torch.sort(row.reshape(-1).float(), descending=True, stable=True).indices


class UndecidedOrder(ValueError):
    """A partial walk order that does not decide the kept set (the cut may lie beyond its end)."""


def exact_nucleus(
    row: torch.Tensor, lane: LaneParams, *, order: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The exact kept set of one logits row under vLLM semantics, in fp64: ``(ids [n] int64 in walk order, q [n] fp64
    renormalised over the kept set)``. Greedy lanes return ``([argmax], [1.0])``.

    ``order``: the walk order -- default :func:`sorted_order` (logit desc, id asc). It may be a **prefix** of a
    logit-descending order of the whole vocabulary (e.g. the device's sorted window, to reproduce its boundary tie
    choice; every token outside it must have a logit ``<=`` its last one): :class:`UndecidedOrder` is raised when that
    prefix does not decide the kept set (top-k set not inside it, or the top-p walk reaching its end below ``p``)."""
    x = row.reshape(-1).double()
    if lane.greedy:
        return torch.tensor([greedy_token(row)], dtype=torch.int64), torch.ones(1, dtype=torch.float64)
    if order is None:
        order = sorted_order(row)
    order = order.reshape(-1).to(torch.int64)
    xs = x[order]
    if bool((xs[1:] > xs[:-1]).any()):
        raise ValueError("order is not a logit-descending order")
    n_walk = int(order.numel())
    full = n_walk == int(x.numel())
    logp = (x - x.max()) / float(lane.temperature)
    lse = torch.logsumexp(logp, 0)
    ps = torch.exp(logp[order] - lse)  # full-vocab probabilities along the walk
    if lane.top_k_active:
        if lane.top_k > n_walk:
            raise UndecidedOrder(f"top_k {lane.top_k} exceeds the walk ({n_walk} ids)")
        kth = float(xs[lane.top_k - 1])
        keep_k = xs >= kth
        if not full and int((x >= kth).sum()) != int(keep_k.sum()):
            raise UndecidedOrder("the walk does not contain every token tied with the k-th logit")
        mass_k = float(ps[keep_k].sum())
    else:
        keep_k = torch.ones_like(xs, dtype=torch.bool)
        mass_k = 1.0
    excl = torch.cumsum(ps, 0) - ps
    if lane.top_p >= 1.0:
        # top-p off: vLLM masks only ``cumsum <= 0`` (zero-probability tokens); keep the whole top-k set
        keep = keep_k & (ps > 0)
    else:
        keep = (excl < float(lane.top_p) * mass_k) & keep_k
    keep[0] = True
    n = int(keep.sum())
    if not bool(keep[:n].all()):
        raise AssertionError("the kept set is not a prefix of the walk")
    if not full and not lane.top_k_active and n == n_walk and float(ps.sum()) < float(lane.top_p):
        raise UndecidedOrder("the top-p walk reaches the end of the given order below top_p")
    q = ps[:n] / ps[:n].sum()
    return order[:n].clone(), q


def inverse_cdf(q: torch.Tensor, u: float) -> int:
    """Index ``#{j : C_j <= u}`` (clamped to ``len(q) - 1``) of the fp64 CDF of ``q`` -- the device's draw rule."""
    C = torch.cumsum(q.double(), 0)
    j = int(torch.searchsorted(C, torch.tensor([float(u)], dtype=torch.float64), right=True)[0])
    return min(j, int(q.numel()) - 1)


def exact_sample(row: torch.Tensor, lane: LaneParams, u: float, *, order: Optional[torch.Tensor] = None) -> int:
    """Exact sample of one row for the uniform ``u`` (inverse CDF over :func:`exact_nucleus` in walk order)."""
    ids, q = exact_nucleus(row, lane, order=order)
    if lane.greedy:
        return int(ids[0])
    return int(ids[inverse_cdf(q, u)])


def raw_logprob(row: torch.Tensor, token: int) -> float:
    """vLLM ``raw_logprobs`` of ``token``: ``log_softmax`` of the unprocessed logits (T = 1), fp64 -> float."""
    x = row.reshape(-1).double()
    return float(x[int(token)] - torch.logsumexp(x, 0))


def fallback_sample(row: torch.Tensor, lane: LaneParams, u: float, *, prefix: int = 4096) -> int:
    """The host fallback for a flagged lane: an exact sample with the lane's device uniform ``u``.

    Equal to :func:`exact_sample` (reference order: logit desc, id asc), computed fast (~1-3 ms per row): greedy ->
    argmax; otherwise the walk over the tokens strictly above the ``prefix``-th largest logit (``torch.topk``) when
    that prefix decides the kept set, else a full sort. Full-vocabulary support (``top_p = 1``, no top-k) uses the
    inverse CDF in **token-id order** instead (no sort: an exact sample of the same distribution, another u -> token
    map). Deterministic in ``(row, lane, u)``."""
    if lane.greedy:
        return greedy_token(row)
    x = row.reshape(-1).double()
    if lane.full_support:
        logp = (x - x.max()) / float(lane.temperature)
        p = torch.exp(logp - torch.logsumexp(logp, 0))
        return inverse_cdf(p, u)
    V = int(x.numel())
    n0 = min(V, max(int(prefix), 2 * lane.top_k if lane.top_k_active else 0))
    if n0 < V:
        xf = row.reshape(-1).float()  # bf16 logits are exact in fp32 (a faster top-k than fp64)
        idx = torch.sort(torch.topk(xf, n0, sorted=False).indices).values  # the n0 largest, by id
        idx = idx[torch.sort(x[idx], descending=True, stable=True).indices]  # logit desc, id asc
        v_last = x[idx[-1]]
        idx = idx[x[idx] > v_last]  # every token above v_last: a prefix of the reference order
        if idx.numel() > 0:
            try:
                ids, q = exact_nucleus(row, lane, order=idx)
                return int(ids[inverse_cdf(q, u)])
            except UndecidedOrder:
                pass
    return exact_sample(row, lane, u)


# ============================================================================================================
# host side: fp32 emulation of the device pipeline (CPU tests, design checks on saved logits)
# ============================================================================================================
def emulate_device(
    logits: torch.Tensor,
    lanes: Sequence[LaneParams],
    ctr: Sequence[int],
    *,
    local_k: int = DEFAULT_LOCAL_K,
    window: int = DEFAULT_WINDOW,
    num_chips: int = 32,
) -> Dict[str, torch.Tensor]:
    """fp32 torch emulation of :meth:`MotifDeviceSampler.sample` on host logits ``[B, V]`` (bf16) for lanes ``lanes``
    and counters ``ctr``: the same candidate sets, normaliser, window, masks, draw and certificate, with the reference
    tie order (logit desc, id asc) where the device uses ``ttnn.topk``'s. Returns ``token, flag, n_kept, u, mass_kept,
    window_mass, kth, logprob`` (``[B]`` each). Accumulation order differs from the device (sums are fp32 either way),
    so results agree except at measure-zero rounding boundaries and tie permutations."""
    x = logits.float()
    B, V = x.shape
    Vc = V // num_chips
    K, W = int(local_k), int(window)
    cols = device_param_columns(lanes, W)
    inv_t = cols["inv_t"].view(B, 1)
    xc = x.view(B, num_chips, Vc)
    # per-chip top-K (reference tie order: value desc, local id asc)
    vals, lidx = torch.sort(xc, dim=2, descending=True, stable=True)
    cv = vals[:, :, :K]
    cid = (lidx[:, :, :K] + (torch.arange(num_chips) * Vc).view(1, num_chips, 1)).float()
    Vf = cv.reshape(B, -1)
    If = cid.reshape(B, -1)
    M = Vf.max(1, keepdim=True).values
    d = x - M
    s = torch.exp(d * inv_t).view(B, num_chips, Vc).sum(2)  # per-chip fp32 partials
    Z = s.sum(1, keepdim=True)
    Z1 = torch.exp(d).view(B, num_chips, Vc).sum(2).sum(1, keepdim=True)
    # window: top-W of the candidates (value desc, id asc)
    o1 = torch.sort(If, dim=1, stable=True).indices
    Vs, Is = Vf.gather(1, o1), If.gather(1, o1)
    o2 = torch.sort(Vs, dim=1, descending=True, stable=True).indices[:, :W]
    SV, SI = Vs.gather(1, o2), Is.gather(1, o2)
    PR = torch.exp((SV - M) * inv_t) / Z
    EX = (torch.cumsum(PR.double(), 1) - PR.double()).float()  # exclusive prefix (device: fp32-accumulated matmul)
    iota = torch.arange(W, dtype=torch.float32).view(1, W)
    ka = cols["k_active"].view(B, 1)
    kth = (SV + k_offset_rows(cols, W)).max(1, keepdim=True).values
    keep_k = (SV >= kth).float()
    mass_k = (PR * keep_k).sum(1, keepdim=True)
    p_eff = cols["top_p"].view(B, 1) * torch.where(ka > 0, mass_k, torch.ones_like(mass_k))
    n_kept = ((EX < p_eff).float() * keep_k).sum(1, keepdim=True)
    keep = (iota < n_kept).float()
    mass_kept = (PR * keep).sum(1, keepdim=True)
    u = torch.from_numpy(uniform_from_counter(np.asarray(ctr, dtype=np.uint64))).view(B, 1)
    j = torch.minimum((EX <= u * mass_kept).float().sum(1, keepdim=True), n_kept) - 1
    oh = (iota == j).float()
    tok_s = (SI * oh).sum(1, keepdim=True)
    x_s = (SV * oh).sum(1, keepdim=True)
    g = torch.where(Vf < M, torch.full_like(If, ID_SENTINEL), If).min(1, keepdim=True).values
    gr = cols["greedy"].view(B, 1)
    tok = torch.where(gr > 0, g, tok_s)
    last_mask = (torch.arange(num_chips * K) % K == K - 1).view(1, -1)
    kchip = torch.where(last_mask, Vf, torch.full_like(Vf, -BIG)).max(1, keepdim=True).values
    last_val = torch.where(keep > 0, SV, torch.full_like(SV, BIG)).min(1, keepdim=True).values
    total = PR.sum(1, keepdim=True)
    win_last = SV.min(1, keepdim=True).values
    vthr = torch.where(gr > 0, M, torch.where(ka > 0, kth, last_val))
    c1 = (torch.maximum(kchip, win_last) < vthr).float()
    c2 = torch.where(ka > 0, cols["k_in_win"].view(B, 1), (total >= p_eff + CERT_MARGIN).float())
    flag = 1.0 - c1 * c2
    x_tok = torch.where(gr > 0, M, x_s)
    lp = (x_tok - M) - torch.log(Z1)
    return {
        "token": tok.view(-1).long(),
        "flag": flag.view(-1),
        "n_kept": n_kept.view(-1),
        "u": u.view(-1),
        "mass_kept": mass_kept.view(-1),
        "window_mass": total.view(-1),
        "kth": kth.view(-1),
        "logprob": lp.view(-1),
        "order": SI.long(),
    }


# ============================================================================================================
# device module
# ============================================================================================================
@dataclass
class SamplerOutput:
    """Device outputs of one :meth:`MotifDeviceSampler.sample` call (trace outputs when captured).

    ``tokens``: ``[1, 1, 1, 32]`` uint32 ROW_MAJOR, lane order, identical on every chip (``MotifLMHead.argmax_decode``
    contract: ``head.tokens_to_host`` reads it; device consumers such as the MTP layer take it as is).
    ``info``: ``[1, 1, 8, 32]`` fp32 ROW_MAJOR, rows :data:`INFO_ROWS` (read with :meth:`MotifDeviceSampler.read`).
    ``debug``: intermediates kept alive with ``sample(..., debug=True)`` (eager diagnostics only)."""

    tokens: Any
    info: Any
    debug: Optional[Dict[str, Any]] = None

    def tensors(self) -> List[Any]:
        out = [self.tokens, self.info]
        if self.debug:
            out += list(self.debug.values())
        return out


@dataclass
class SampleResult:
    """Host view of one step's sampler output (lane order, ``[32]`` each).

    ``tokens`` int64, ``flags`` bool (True = the device token is not certified exact: :meth:`MotifDeviceSampler.resolve`
    replaces it), ``logprobs`` float32 (raw logprob of the token, vLLM ``raw_logprobs``; 0 without ``logprobs``),
    ``n_kept`` / ``mass_kept`` / ``window_mass`` / ``kth`` / ``u`` float32 (diagnostics), ``resolved``: lanes whose
    token came from the host fallback."""

    tokens: torch.Tensor
    flags: torch.Tensor
    logprobs: torch.Tensor
    n_kept: torch.Tensor
    u: torch.Tensor
    mass_kept: torch.Tensor
    window_mass: torch.Tensor
    kth: torch.Tensor
    resolved: List[int] = field(default_factory=list)

    @staticmethod
    def from_info(info: torch.Tensor) -> "SampleResult":
        a = info.reshape(len(INFO_ROWS), -1).float()
        g = lambda n: a[INFO_INDEX[n]].clone()  # noqa: E731
        return SampleResult(
            tokens=g("token").round().to(torch.int64),
            flags=g("flag") > 0.5,
            logprobs=g("logprob"),
            n_kept=g("n_kept"),
            u=g("u"),
            mass_kept=g("mass_kept"),
            window_mass=g("window_mass"),
            kth=g("kth"),
        )


class MotifDeviceSampler:
    """Exact on-device sampler for the 32 decode lanes (module docstring; ``docs/sampling/DEVICE_SAMPLER.md``).

    Args:
        mesh_device / cfg: the opened mesh and its :class:`MotifTTConfig` (README §10 pattern).
        ccl: a :class:`MotifCCL` (shared with the model; default a new one).
        max_lanes: decode lanes (32: the decode trace always runs all 32, ``cfg.max_batch``).
        local_k: per-chip candidates K (32, 64 (default) or 128); the candidate set is ``32 K`` per lane.
        window: sorted global window W walked for top-k / top-p (256, 512 (default), 1024, 2048; ``<= 32 K``).
        logprobs: also compute the raw (T = 1) logprob of the sampled token (one more fp32 exp-sum pass per chip,
            needed when the plugin routes a ``logprobs=0`` request to device sampling).
        rng: ``"device"`` (default: the counter hash runs inside the trace; the host writes one int32 counter per lane
            per step) or ``"host"`` (the host writes the uniform itself, bit-identical values; 11 fewer device ops).
        coverage_counter: keep a persistent per-lane count of flagged steps on the device (:meth:`coverage_counts`).
        rng_seed: seed of the host RNG behind unseeded lanes' counters (None = OS entropy).
        memory_config: outputs (DRAM interleaved).
        l1_intermediates: keep the intermediates in L1 (interleaved) instead of DRAM (all freed inside the call).
        impl: implementation variants (A/B on device; the defaults are the measured best, DEVICE_SAMPLER.md §5):
            ``rowsum`` "stack" (with logprobs: both full-row exp sums as one two-tile-row reduction) or "row" (one
            single-core W reduction each); ``fused_exp`` (exp as the multiply's post-activation);
            ``prefix`` "2mm" (two matmuls, hi and mid) or "1mm" (one matmul of [mid | hi] with [U; U]);
            ``fuse`` (fold five small ops into their neighbours as unary pre/post activations); ``verify`` (the exact
            re-check of the cut and the draw; False only to MEASURE its cost -- the sampler is then not exact);
            ``repair`` (move the proposed top-p cut by one token when the exact sums say so, instead of flagging).

    Lifecycle (all device tensors are allocated here, before any trace capture):
        ``set_params(temperature, top_p, top_k, seeds)`` (on change; host compare, no device write when equal),
        ``set_positions(positions)`` (every step: the RNG counters), ``out = sample(logits)`` (eager or inside the
        decode trace capture; one eager call before capture compiles every program), ``res = read(out)`` (host),
        ``resolve(res, logits_host)`` when ``res.flags.any()``."""

    DEFAULT_IMPL = {
        "rowsum": "stack", "fused_exp": False, "prefix": "2mm", "fuse": True, "verify": True, "repair": True,
    }  # fmt: skip

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        *,
        ccl: Optional[MotifCCL] = None,
        max_lanes: int = 32,
        local_k: int = DEFAULT_LOCAL_K,
        window: int = DEFAULT_WINDOW,
        logprobs: bool = True,
        rng: str = "device",
        coverage_counter: bool = True,
        rng_seed: Optional[int] = None,
        memory_config=None,
        l1_intermediates: bool = False,
        impl: Optional[Dict[str, Any]] = None,
    ):
        if int(max_lanes) != TILE or int(cfg.max_batch) != TILE:
            raise ValueError(f"the sampler runs the {TILE} decode lanes of the trace (max_lanes={max_lanes}, "
                             f"cfg.max_batch={cfg.max_batch})")
        if int(local_k) not in LOCAL_K_CHOICES:
            raise ValueError(f"local_k must be one of {LOCAL_K_CHOICES}, got {local_k}")
        n_chips = int(cfg.axes.num_chips)
        if int(window) not in WINDOW_CHOICES or int(window) > n_chips * int(local_k):
            raise ValueError(f"window must be one of {WINDOW_CHOICES} and <= {n_chips} x local_k, got {window}")
        if rng not in ("device", "host"):
            raise ValueError(f"rng must be 'device' or 'host', got {rng!r}")
        if tuple(mesh_device.shape) != tuple(cfg.axes.mesh_shape):
            raise ValueError(f"cfg is for mesh {cfg.axes.mesh_shape}, device mesh is {tuple(mesh_device.shape)}")
        self.impl = dict(self.DEFAULT_IMPL)
        self.impl.update(impl or {})
        if self.impl["rowsum"] not in ("stack", "row") or self.impl["prefix"] not in ("1mm", "2mm"):
            raise ValueError(f"bad impl {self.impl}")
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.ccl = ccl if ccl is not None else MotifCCL(mesh_device, cfg)
        self.lanes = TILE
        self.K = int(local_k)
        self.W = int(window)
        self.num_chips = n_chips
        self.NC = n_chips * self.K
        self.vc = vocab_per_shard(cfg, "mesh")
        self.vocab_size = int(cfg.vocab_size)
        self.logprobs = bool(logprobs)
        self.rng = rng
        self.memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        self.imc = ttnn.L1_MEMORY_CONFIG if l1_intermediates else ttnn.DRAM_MEMORY_CONFIG
        self.ckc = cfg.compute_config("eltwise")  # HiFi4, fp32 dest acc (accurate SFPU reductions), approx off
        self._rng = np.random.default_rng(rng_seed if rng_seed is not None else secrets.randbits(64))
        self._readers: Dict[tuple, HostShardReader] = {}
        # steps / flagged_*: every read() (flags of ALL lanes: inactive and full-support lanes included);
        # resolved_steps / resolved_lanes: what resolve() actually re-sampled (the steps that paid the logits read)
        self.stats = {"steps": 0, "flagged_steps": 0, "flagged_lanes": 0, "resolved_steps": 0, "resolved_lanes": 0,
                      "param_uploads": 0}
        rep = self._rep = ttnn.ReplicateTensorToMesh(mesh_device)
        L, W, NC = self.lanes, self.W, self.NC

        def up(t, dtype, mapper=rep):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)

        row = lambda v: v.reshape(1, 1, 1, -1).expand(1, 1, L, -1).contiguous()  # noqa: E731
        # ---- constants (allocated once: trace safe) ----
        self._iota_w = up(row(torch.arange(W, dtype=torch.float32)), ttnn.float32)
        self._iota_n = up(row(torch.arange(NC, dtype=torch.float32)), ttnn.float32)
        self._mask_last = up(row((torch.arange(NC) % self.K == self.K - 1).float()), ttnn.float32)
        e = torch.zeros(1, 1, L, TILE)
        self._e0 = up(e.clone().index_fill_(3, torch.tensor([0]), 1.0), ttnn.float32)
        self._e1 = up(e.clone().index_fill_(3, torch.tensor([1]), 1.0), ttnn.float32)
        cols = torch.arange(n_chips * TILE)
        self._sel_z = up(row((cols % TILE == 0).float()), ttnn.float32)
        self._sel_z1 = up(row((cols % TILE == 1).float()), ttnn.float32)
        # stacked normalisers ([64, ...]: rows 0-31 the T-scaled partials, rows 32-63 the raw ones)
        self._e0_64 = up(torch.zeros(1, 1, 2 * L, TILE).index_fill_(3, torch.tensor([0]), 1.0), ttnn.float32)
        self._sel_z64 = up((cols % TILE == 0).float().reshape(1, 1, 1, -1).expand(1, 1, 2 * L, -1).contiguous(),
                           ttnn.float32)
        # strictly upper triangular: (PR @ Us)[j] = sum_{i < j} PR[i], the exclusive prefix
        Us = torch.triu(torch.ones(W, W), diagonal=1)
        if self.impl["prefix"] == "1mm":
            Us = torch.cat([Us, Us], dim=0)
        self._U = up(Us.reshape(1, 1, -1, W), ttnn.bfloat16)
        self._m1 = up(torch.full((1, 1, L, 1), _i32(LOWBIAS_M1), dtype=torch.int32), ttnn.int32)
        self._m2 = up(torch.full((1, 1, L, 1), _i32(LOWBIAS_M2), dtype=torch.int32), ttnn.int32)
        # per-chip vocab offset 6880 b (b = mesh linear index r * C + c, the LM head's "mesh" split)
        R, C = cfg.axes.mesh_shape
        off = torch.zeros(R, 1, L, C, dtype=torch.float32)
        for r in range(R):
            for c in range(C):
                off[r, 0, :, c] = float(vocab_block_of_coord(cfg, r, c, "mesh") * self.vc)
        self._chip_off = up(off, ttnn.float32, ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 3), mesh_shape=(R, C)))
        # ---- persistent per-lane parameters ([1, 1, 32, 1], replicated) ----
        z = torch.zeros(1, 1, L, 1)
        self._p = {k: up(z, ttnn.float32) for k in ("greedy", "inv_t", "top_p", "k_active", "k_in_win")}
        self._p["k_ofs"] = up(torch.full((1, 1, L, W), -BIG), ttnn.float32)  # top-k threshold mask (k_offset_rows)
        if rng == "device":
            self._ctr = up(torch.zeros(1, 1, L, 1, dtype=torch.int32), ttnn.int32)
            self._u = None
        else:
            self._ctr = None
            self._u = up(z, ttnn.float32)
        self._counter = up(z, ttnn.float32) if coverage_counter else None
        # host-side state
        self._lane_params: List[LaneParams] = [GREEDY_LANE] * L
        self._host_cols: Optional[Dict[str, torch.Tensor]] = None
        self._last_raw = None
        self._keys: List[Optional[int]] = [None] * L
        self._last_ctr = np.zeros(L, dtype=np.uint32)
        self.set_params([0.0] * L, [1.0] * L, [0] * L, [None] * L)
        self.set_positions([0] * L)

    # ---- host -> device parameters ------------------------------------------------------------------------
    def _host_tensor(self, t: torch.Tensor, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._rep)

    @property
    def lane_params(self) -> List[LaneParams]:
        return list(self._lane_params)

    def set_params(self, temperature, top_p, top_k, seeds, *, force: bool = False) -> bool:
        """Per-lane sampling parameters in **lane order** (sequences / tensors of length 32; padding lanes greedy).
        Conventions as the plugin sends them (:func:`normalize_lane`): temperature 0 = greedy, top_k ``vocab_size``
        (or <= 0 / None) = off, top_p 1.0 = off, seed int or None. Uploads the device columns only when they changed
        (or ``force``); returns True when it wrote the device. Not traceable (host -> device copies): call it before
        the step's replay. Seeds only affect :meth:`set_positions` (host-side counters)."""
        raw = tuple(tuple(v.reshape(-1).tolist()) if isinstance(v, torch.Tensor) else (None if v is None else tuple(v))
                    for v in (temperature, top_p, top_k, seeds))
        if not force and raw == self._last_raw:  # the common case: the plugin resends unchanged parameters
            return False
        lanes = normalize_lanes(temperature, top_p, top_k, seeds, vocab_size=self.vocab_size, num_lanes=self.lanes)
        self._last_raw = raw
        self._lane_params = lanes
        self._keys = [None if l.seed is None else seed_key(l.seed) for l in lanes]
        self._key_mask = np.array([k is not None for k in self._keys])
        self._key_arr = np.array([0 if k is None else k for k in self._keys], dtype=np.uint64)
        cols = device_param_columns(lanes, self.W)
        if not force and self._host_cols is not None and all(torch.equal(cols[k], self._host_cols[k]) for k in cols):
            return False
        for k, v in cols.items():
            if k == "k_idx":
                continue
            ttnn.copy_host_to_device_tensor(self._host_tensor(v.reshape(1, 1, self.lanes, 1), ttnn.float32), self._p[k])
        kofs = k_offset_rows(cols, self.W).reshape(1, 1, self.lanes, self.W)
        ttnn.copy_host_to_device_tensor(self._host_tensor(kofs, ttnn.float32), self._p["k_ofs"])
        self._host_cols = cols
        self.stats["param_uploads"] += 1
        return True

    def set_lane_params(self, params: Sequence[LaneParams], *, force: bool = False) -> bool:
        """:meth:`set_params` from :class:`LaneParams` (one per lane)."""
        if len(params) != self.lanes:
            raise ValueError(f"expected {self.lanes} lanes, got {len(params)}")
        return self.set_params([p.temperature for p in params], [p.top_p for p in params],
                               [p.top_k for p in params], [p.seed for p in params], force=force)

    def counters_for(self, positions) -> np.ndarray:
        """The uint32 RNG counters :meth:`set_positions` would write for ``positions`` (lane order): seeded lanes
        ``key + pos * 0x9E3779B1``, unseeded lanes fresh host-random values (advances the host RNG)."""
        pos = np.asarray(positions.numpy() if isinstance(positions, torch.Tensor) else positions, dtype=np.int64)
        pos = pos.reshape(-1)
        if pos.shape[0] != self.lanes:
            raise ValueError(f"expected {self.lanes} positions, got {pos.shape[0]}")
        seeded = self._key_mask
        ctr = (self._key_arr + np.maximum(pos, 0).astype(np.uint64) * np.uint64(GOLDEN32)) & np.uint64(MASK32)
        if not seeded.all():
            rnd = self._rng.integers(0, 1 << 32, size=self.lanes, dtype=np.uint64)
            ctr = np.where(seeded, ctr, rnd)
        return ctr.astype(np.uint32)

    def set_counters(self, ctr) -> None:
        """Write explicit uint32 RNG counters (lane order; tests and replays): one ``[1, 1, 32, 1]`` host -> device copy
        (~0.15 ms). Not traceable. With ``rng="host"`` the uniforms of the counters are written instead (bit-identical
        to the device hash)."""
        c = (np.asarray(ctr, dtype=np.uint64).reshape(-1) & np.uint64(MASK32)).astype(np.uint32)
        if c.shape[0] != self.lanes:
            raise ValueError(f"expected {self.lanes} counters, got {c.shape[0]}")
        if self._ctr is not None:
            t = torch.from_numpy(c.view(np.int32).copy()).reshape(1, 1, self.lanes, 1)
            ttnn.copy_host_to_device_tensor(self._host_tensor(t, ttnn.int32), self._ctr)
        else:
            u = torch.from_numpy(uniform_from_counter(c)).reshape(1, 1, self.lanes, 1)
            ttnn.copy_host_to_device_tensor(self._host_tensor(u, ttnn.float32), self._u)
        self._last_ctr = c

    def set_positions(self, positions) -> np.ndarray:
        """Per-step RNG input: ``positions`` = each lane's decode input position (``start_pos``; -1 / anything for
        inactive lanes) in lane order. Writes the counters (one ``[1, 1, 32, 1]`` host -> device copy) and returns
        them. Not traceable: call it before every replay of a trace that holds :meth:`sample`."""
        ctr = self.counters_for(positions)
        self.set_counters(ctr)
        return ctr

    def last_uniforms(self) -> np.ndarray:
        """The host twin of the device ``u`` of the counters written last (float32 ``[32]``)."""
        return uniform_from_counter(self._last_ctr)

    # ---- the device pipeline (stages; each returns new tensors and records its intermediates in ``tmp``) ---------
    def _red(self, op, x, dim: int = 3):
        return op(x, dim=dim, keepdim=True, compute_kernel_config=self.ckc, memory_config=self.imc)

    def _exp(self, a, b):
        """``exp(a * b)`` in fp32 (accurate exp; optionally fused as the multiply's post-activation)."""
        mc = self.imc
        if self.impl["fused_exp"]:
            return [ttnn.multiply(a, b, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)],
                                  memory_config=mc)]
        p = ttnn.multiply(a, b, memory_config=mc)
        return [p, ttnn.exp(p, fast_and_approximate_mode=False, memory_config=mc)]

    def _stage_local(self, x, tmp) -> Dict[str, Any]:
        mc = self.imc
        T = tmp.append
        tv, ti = ttnn.topk(x, k=self.K, dim=-1, largest=True, sorted=True, memory_config=mc)
        T(tv), T(ti)
        tif = ttnn.typecast(ti, ttnn.float32, memory_config=mc)
        T(tif)
        gid = ttnn.add(tif, self._chip_off, memory_config=mc)  # global ids, exact in fp32 (< 2^24)
        T(gid)
        a = self.ccl.all_gather(tv, 3, "tp", memory_config=mc)
        T(a)
        Vb = self.ccl.all_gather(a, 3, "dp", memory_config=mc)
        T(Vb)
        b = self.ccl.all_gather(gid, 3, "tp", memory_config=mc)
        T(b)
        I = self.ccl.all_gather(b, 3, "dp", memory_config=mc)
        T(I)
        V = ttnn.typecast(Vb, ttnn.float32, memory_config=mc)
        T(V)
        M = self._red(ttnn.max, V)
        T(M)
        return {"Vb": Vb, "V": V, "I": I, "M": M}

    def _stage_norm(self, x, M, tmp) -> Dict[str, Any]:
        """Full-vocab normalisers. Per chip ``s = sum exp((x - M) / T)`` (and, with logprobs, ``s1 = sum exp(x - M)``)
        over the chip's 6880 logits -- accurate fp32 SFPU reductions; a W reduction of one tile row runs on one core,
        so ``impl["rowsum"] = "stack"`` reduces both exponent rows stacked as two tile rows in ONE call (two cores) --
        then the per-chip partials (column 0 of a tile per partial row) are all-gathered and summed exactly
        (adding zeros): ``Z [1, 1, 32, 1]`` and ``Z1``."""
        mc, P = self.imc, self._p
        T = tmp.append
        x32 = ttnn.typecast(x, ttnn.float32, memory_config=mc)
        T(x32)
        d = ttnn.subtract(x32, M, memory_config=mc)
        T(d)
        L = self.lanes
        if self.logprobs and self.impl["rowsum"] == "stack":
            a = ttnn.multiply(d, P["inv_t"], memory_config=mc)
            T(a)
            st = ttnn.concat([a, d], dim=2, memory_config=mc)  # [1, 1, 64, 6880]: T-scaled rows, then raw rows
            T(st)
            e = ttnn.exp(st, fast_and_approximate_mode=False, memory_config=mc)
            T(e)
            s2 = self._red(ttnn.sum, e)  # [1, 1, 64, 1]
            T(s2)
            part = ttnn.multiply(self._e0_64, s2, memory_config=mc)  # partial in column 0, zeros elsewhere
            T(part)
            sel = self._sel_z64
        else:
            es = self._exp(d, P["inv_t"])
            tmp.extend(es)
            sz = self._red(ttnn.sum, es[-1])
            T(sz)
            part = ttnn.multiply(self._e0, sz, memory_config=mc)
            T(part)
            if self.logprobs:
                e1 = ttnn.exp(d, fast_and_approximate_mode=False, memory_config=mc)
                T(e1)
                s1 = self._red(ttnn.sum, e1)
                T(s1)
                q = ttnn.multiply(self._e1, s1, memory_config=mc)
                T(q)
                part = ttnn.add(part, q, memory_config=mc)
                T(part)
            sel = self._sel_z
        g = self.ccl.all_gather(part, 3, "tp", memory_config=mc)
        T(g)
        S = self.ccl.all_gather(g, 3, "dp", memory_config=mc)
        T(S)
        zm = ttnn.multiply(S, sel, memory_config=mc)
        T(zm)
        Zs = self._red(ttnn.sum, zm)
        T(Zs)
        out = {"S": S}
        if self.logprobs and self.impl["rowsum"] == "stack":
            Z = ttnn.slice(Zs, [0, 0, 0, 0], [1, 1, L, 1], memory_config=mc)
            Z1 = ttnn.slice(Zs, [0, 0, L, 0], [1, 1, 2 * L, 1], memory_config=mc)
            T(Z), T(Z1)
            out.update(Z=Z, Z1=Z1)
        else:
            out["Z"] = Zs
            if self.logprobs:
                z1m = ttnn.multiply(S, self._sel_z1, memory_config=mc)
                T(z1m)
                Z1 = self._red(ttnn.sum, z1m)
                T(Z1)
                out["Z1"] = Z1
        return out

    def _stage_window(self, Vb, M, Z, tmp) -> Dict[str, Any]:
        mc, P = self.imc, self._p
        T = tmp.append
        SVb, sidx = ttnn.topk(Vb, k=self.W, dim=-1, largest=True, sorted=True, memory_config=mc)
        T(SVb), T(sidx)
        SV = ttnn.typecast(SVb, ttnn.float32, memory_config=mc)
        T(SV)
        sidx_f = ttnn.typecast(sidx, ttnn.float32, memory_config=mc)
        T(sidx_f)
        dw = ttnn.subtract(SV, M, memory_config=mc)
        T(dw)
        es = self._exp(dw, P["inv_t"])
        tmp.extend(es)
        PR = ttnn.divide(es[-1], Z, memory_config=mc)
        T(PR)
        return {"SV": SV, "sidx": sidx_f, "PR": PR}

    def _stage_prefix(self, PR, tmp) -> Dict[str, Any]:
        """Proposed exclusive prefix sums ``EX[j] ~ sum_{i < j} PR[i]``: ``PR ~ hi + mid`` (bf16 parts, relative error
        <= 2^-17), bf16 x {0, 1} matmuls with fp32 dest. The FPU rounds inside a K-tile (~10 significant bits
        observed, DEVICE_SAMPLER.md §3.3), so ``EX`` only proposes the cut and the draw; :meth:`_stage_select`
        re-checks both with exact fp32 SFPU sums."""
        mc = self.imc
        T = tmp.append
        hi = ttnn.typecast(PR, ttnn.bfloat16, memory_config=mc)
        T(hi)
        hi32 = ttnn.typecast(hi, ttnn.float32, memory_config=mc)
        T(hi32)
        r1 = ttnn.subtract(PR, hi32, memory_config=mc)
        T(r1)
        mid = ttnn.typecast(r1, ttnn.bfloat16, memory_config=mc)
        T(mid)

        def mm(a):
            o = ttnn.matmul(a, self._U, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=mc)
            T(o)
            return o

        if self.impl["prefix"] == "1mm":
            cat = ttnn.concat([mid, hi], dim=3, memory_config=mc)
            T(cat)
            EX = mm(cat)
        else:
            EX = ttnn.add(mm(mid), mm(hi), memory_config=mc)
            T(EX)
        return {"EX": EX}

    def _stage_uniform(self, tmp):
        if self._u is not None:
            return self._u
        mc = self.imc
        T = tmp.append
        h = self._ctr
        for shift, mul in ((16, self._m1), (15, self._m2), (15, None)):
            s = ttnn.logical_right_shift(h, shift, memory_config=mc)
            T(s)
            h = ttnn.bitwise_xor(h, s, memory_config=mc)
            T(h)
            if mul is not None:
                h = ttnn.multiply(h, mul, memory_config=mc)
                T(h)
        hs = ttnn.logical_right_shift(h, 8, memory_config=mc)
        T(hs)
        hf = ttnn.typecast(hs, ttnn.float32, memory_config=mc)
        T(hf)
        u = ttnn.multiply(hf, U24, memory_config=mc)
        T(u)
        return u

    def _stage_select(self, d: Dict[str, Any], tmp) -> Dict[str, Any]:
        """top-k / top-p masks, the draw, the greedy token, the certificate and the logprob (all ``[1, 1, 32, 1]``)."""
        mc, P = self.imc, self._p
        red = self._red

        def T(t):
            tmp.append(t)
            return t

        SV, PR, EX, V, I, M = d["SV"], d["PR"], d["EX"], d["V"], d["I"], d["M"]
        fuse = bool(self.impl["fuse"])
        # top-k: keep every value >= the k-th largest (all ties); off -> -BIG (keep all)
        kth = T(red(ttnn.max, T(ttnn.add(SV, P["k_ofs"], memory_config=mc))))  # k-th value, or -BIG (top-k off)
        keep_k = T(ttnn.ge(SV, kth, memory_config=mc))
        mass_k = T(red(ttnn.sum, T(ttnn.multiply(PR, keep_k, memory_config=mc))))
        # top-p on the top-k-renormalised mass: keep rank r iff its exclusive prefix < top_p * mass_k
        p_eff = T(ttnn.multiply(P["top_p"], T(ttnn.where(P["k_active"], mass_k, 1.0, memory_config=mc)),
                                memory_config=mc))
        # the proposed count; the kept set is the PREFIX of that length by construction (the FPU-accumulated EX need
        # not be monotone), whose exactness the certificate re-checks with exact sums
        n_kept = T(red(ttnn.sum, T(ttnn.multiply(T(ttnn.lt(EX, p_eff, memory_config=mc)), keep_k, memory_config=mc))))
        keep = T(ttnn.lt(self._iota_w, n_kept, memory_config=mc))
        mass_kept = T(red(ttnn.sum, T(ttnn.multiply(PR, keep, memory_config=mc))))
        if self.impl["repair"]:
            # one-step exact repair of the cut. The FPU prefix runs high (~1e-3 near 1 in real decoding), so the
            # proposal mostly drops one token too many: the first dropped token (rank n, inside the top-k set) is kept
            # iff its exact exclusive prefix (mass_kept) < p_eff. The certificate (v_a, v_b) checks the repaired cut;
            # a proposal off by more, or one token too long, stays flagged.
            m_n = T(ttnn.multiply(T(ttnn.eq(self._iota_w, n_kept, memory_config=mc)), keep_k, memory_config=mc))
            in_k = T(red(ttnn.sum, m_n))  # 1 iff rank n lies inside the window and the top-k set
            up = T(ttnn.multiply(T(ttnn.lt(mass_kept, p_eff, memory_config=mc)), in_k, memory_config=mc))
            n_kept = T(ttnn.add(n_kept, up, memory_config=mc))
            keep = T(ttnn.lt(self._iota_w, n_kept, memory_config=mc))
            mass_kept = T(red(ttnn.sum, T(ttnn.multiply(PR, keep, memory_config=mc))))
        # draw: j = #{C_j <= u * mass_kept} = #{EX_j <= u * mass_kept} - 1, clamped to the kept prefix
        u = self._stage_uniform(tmp)
        target = T(ttnn.multiply(u, mass_kept, memory_config=mc))
        cnt = T(red(ttnn.sum, T(ttnn.le(EX, target, memory_config=mc))))
        if fuse:
            j = T(ttnn.minimum(cnt, n_kept, activations=[_UP(ttnn.UnaryOpType.ADD_UNARY_SFPU, -1.0)],
                               memory_config=mc))
        else:
            j = T(ttnn.subtract(T(ttnn.minimum(cnt, n_kept, memory_config=mc)), 1.0, memory_config=mc))
        oh_j = T(ttnn.eq(self._iota_w, j, memory_config=mc))
        pstar = T(red(ttnn.sum, T(ttnn.multiply(d["sidx"], oh_j, memory_config=mc))))  # column in the candidate row
        tok_s = T(red(ttnn.sum, T(ttnn.multiply(I, T(ttnn.eq(self._iota_n, pstar, memory_config=mc)),
                                                memory_config=mc))))
        # greedy: the lowest id among the maxima (exact fp32 min over ids < 2^24)
        g = T(red(ttnn.min, T(ttnn.where(T(ttnn.lt(V, M, memory_config=mc)), ID_SENTINEL, I, memory_config=mc))))
        tok = T(ttnn.where(P["greedy"], g, tok_s, memory_config=mc))
        # certificate: every token at or above the lane's decisive value is a candidate inside the window
        # (greedy: the max; top-k: the k-th value; top-p: the last kept value), and the walk ends inside the window
        # (top-p: window mass >= p_eff + margin; top-k: k <= W)
        kchip = T(red(ttnn.max, T(ttnn.where(self._mask_last, V, -BIG, memory_config=mc))))  # max_c (K-th of chip c)
        last_val = T(red(ttnn.min, T(ttnn.where(keep, SV, BIG, memory_config=mc))))
        total = T(red(ttnn.sum, PR))
        win_last = T(red(ttnn.min, SV))
        vthr = T(ttnn.where(P["greedy"], M, T(ttnn.where(P["k_active"], kth, last_val, memory_config=mc)),
                            memory_config=mc))
        c1 = T(ttnn.lt(T(ttnn.maximum(kchip, win_last, memory_config=mc)), vthr, memory_config=mc))
        if fuse:  # (total - p_eff) >= margin
            reach = T(ttnn.subtract(total, p_eff, activations=[_UP(ttnn.UnaryOpType.UNARY_GE, CERT_MARGIN)],
                                    memory_config=mc))
        else:
            reach = T(ttnn.ge(total, T(ttnn.add(p_eff, CERT_MARGIN, memory_config=mc)), memory_config=mc))
        c2 = T(ttnn.where(P["k_active"], P["k_in_win"], reach, memory_config=mc))
        # exactness of the two prefix decisions (the matmul prefix EX accumulates inside the FPU below fp32 precision
        # -- ~10 significant bits observed): re-check the cut and the draw with exact fp32 SFPU sums.
        #   cut:  the last kept token's exclusive prefix (mass_kept - PR[n-1]) < p_eff, and the first dropped one's
        #         (mass_kept) >= p_eff unless the whole top-k set is kept;
        #   draw: EX[j] <= target < EX[j] + PR[j] with EX[j] = sum(PR[:j]).
        if not self.impl["verify"]:  # measurement only (DEVICE_SAMPLER.md §3.3): NOT exact without the check
            return self._finish_select(d, tmp, tok=tok, c1=c1, c2=c2, n_kept=n_kept, u=u, mass_kept=mass_kept,
                                       total=total, kth=kth, keep=keep, kchip=kchip, last_val=last_val, oh_j=oh_j)
        pr_last = T(red(ttnn.min, T(ttnn.where(keep, PR, BIG, memory_config=mc))))
        v_a = T(ttnn.lt(T(ttnn.subtract(mass_kept, pr_last, memory_config=mc)), p_eff, memory_config=mc))
        # first dropped token: mass_kept >= p_eff, or the whole top-k set kept (mass_kept == mass_k bitwise: the same
        # elements summed in the same order); greedy lanes pass all three checks by construction (p_eff ~ 0)
        v_b = T(ttnn.ge(mass_kept, T(ttnn.minimum(p_eff, mass_k, memory_config=mc)), memory_config=mc))
        ex_j = T(red(ttnn.sum, T(ttnn.multiply(PR, T(ttnn.lt(self._iota_w, j, memory_config=mc)), memory_config=mc))))
        pr_j = T(red(ttnn.sum, T(ttnn.multiply(PR, oh_j, memory_config=mc))))
        if fuse:  # dlt = target - EX[j]: 0 <= dlt < PR[j]
            dlt = T(ttnn.subtract(target, ex_j, memory_config=mc))
            v_c = T(ttnn.multiply(dlt, T(ttnn.lt(dlt, pr_j, memory_config=mc)),
                                  input_tensor_a_activations=[_UP(ttnn.UnaryOpType.GEZ)], memory_config=mc))
        else:
            v_c = T(ttnn.multiply(T(ttnn.le(ex_j, target, memory_config=mc)),
                                  T(ttnn.lt(target, T(ttnn.add(ex_j, pr_j, memory_config=mc)), memory_config=mc)),
                                  memory_config=mc))
        v = T(ttnn.multiply(T(ttnn.multiply(v_a, v_b, memory_config=mc)), v_c, memory_config=mc))
        c2 = T(ttnn.multiply(c2, v, memory_config=mc))
        return self._finish_select(d, tmp, tok=tok, c1=c1, c2=c2, n_kept=n_kept, u=u, mass_kept=mass_kept, total=total,
                                   kth=kth, keep=keep, kchip=kchip, last_val=last_val, oh_j=oh_j)

    def _finish_select(self, d, tmp, *, tok, c1, c2, n_kept, u, mass_kept, total, kth, keep, kchip, last_val, oh_j):
        """The flag and the raw logprob (the end of :meth:`_stage_select`)."""
        mc, P = self.imc, self._p
        red = self._red
        fuse = bool(self.impl["fuse"])
        SV, M = d["SV"], d["M"]

        def T(t):
            tmp.append(t)
            return t

        if fuse:
            flag = T(ttnn.multiply(c1, c2, activations=[_UP(ttnn.UnaryOpType.LOGICAL_NOT_UNARY)], memory_config=mc))
        else:
            flag = T(ttnn.rsub(T(ttnn.multiply(c1, c2, memory_config=mc)), 1.0, memory_config=mc))
        out = {"tok": tok, "flag": flag, "n_kept": n_kept, "u": u, "mass_kept": mass_kept, "total": total,
               "kth": kth, "keep": keep, "kchip": kchip, "last_val": last_val}
        # raw logprob of the token (T = 1): x_tok - M - log Z1
        if self.logprobs:
            x_s = T(red(ttnn.sum, T(ttnn.multiply(SV, oh_j, memory_config=mc))))
            x_tok = T(ttnn.where(P["greedy"], M, x_s, memory_config=mc))
            if fuse:  # x_tok - (M + log Z1)
                mz = T(ttnn.add(M, d["Z1"], input_tensor_b_activations=[_UP(ttnn.UnaryOpType.LOG)], memory_config=mc))
                out["lp"] = T(ttnn.subtract(x_tok, mz, memory_config=mc))
            else:
                lz = T(ttnn.log(d["Z1"], fast_and_approximate_mode=False, memory_config=mc))
                out["lp"] = T(ttnn.subtract(T(ttnn.subtract(x_tok, M, memory_config=mc)), lz, memory_config=mc))
        else:
            out["lp"] = T(ttnn.multiply(tok, 0.0, memory_config=mc))
        return out

    def _stage_outputs(self, o: Dict[str, Any], tmp) -> Tuple[Any, Any]:
        mc, omc = self.imc, self.memory_config
        T = tmp.append
        if self._counter is not None:
            ttnn.add(self._counter, o["flag"], output_tensor=self._counter)
        rows = [o["tok"], o["flag"], o["lp"], o["n_kept"], o["u"], o["mass_kept"], o["total"], o["kth"]]
        packed = ttnn.concat(rows, dim=3, memory_config=mc)  # [1, 1, 32, 8]
        T(packed)
        pt = ttnn.transpose(packed, 2, 3, memory_config=mc)  # [1, 1, 8, 32]
        T(pt)
        info = ttnn.untilize_with_unpadding(pt, [0, 0, len(INFO_ROWS) - 1, self.lanes - 1], memory_config=omc)
        tok_t = ttnn.transpose(o["tok"], 2, 3, memory_config=mc)  # [1, 1, 1, 32]
        T(tok_t)
        tok_u = ttnn.typecast(tok_t, ttnn.uint32, memory_config=mc)
        T(tok_u)
        tokens = ttnn.untilize_with_unpadding(tok_u, [0, 0, 0, self.lanes - 1], memory_config=omc)
        return tokens, info

    def prepare_logits(self, logits, tmp) -> Any:
        """The sampler's input view of the decode logits: TILE bf16 ``[1, 1, 32, 6880]`` (a ROW_MAJOR input is tilized,
        the new tensor recorded in ``tmp``). bf16 only: the candidates, the certificate and the host side (fallback,
        twin) all work on the values the host reads, so fp32 logits (``MotifLMHead(logits_dtype=ttnn.float32)``) would
        have to be rounded to bf16 here -- greedy would then no longer be ``torch.argmax`` of the host logits and the
        device draws would follow another distribution than the fp32 fallback (review 2026-10-03: this used to be a
        silent ``typecast``)."""
        if tuple(int(d) for d in logits.shape) != (1, 1, self.lanes, self.vc):
            raise ValueError(f"expected decode logits [1, 1, {self.lanes}, {self.vc}], got {list(logits.shape)}")
        if logits.dtype != ttnn.bfloat16:
            raise ValueError(
                f"MotifDeviceSampler needs bf16 decode logits, got {logits.dtype}: rounding them to bf16 here would "
                "break greedy == torch.argmax of the host logits and the exactness of the host fallback"
            )
        x = logits
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=self.imc)
            tmp.append(x)
        return x

    def stages(self, logits, tmp) -> Dict[str, Any]:
        """Every intermediate of one sampler call (the stages of :meth:`sample`, without the output packing)."""
        x = self.prepare_logits(logits, tmp)
        d: Dict[str, Any] = {"x": x}
        d.update(self._stage_local(x, tmp))
        d.update(self._stage_norm(x, d["M"], tmp))
        d.update(self._stage_window(d["Vb"], d["M"], d["Z"], tmp))
        d.update(self._stage_prefix(d["PR"], tmp))
        d.update(self._stage_select(d, tmp))
        return d

    def sample(self, logits, *, debug: bool = False) -> SamplerOutput:
        """Sample one token per lane from the decode logits (trace safe; ``logits`` is not consumed).

        ``logits``: the LM head's decode output ``[1, 1, 32, 6880]`` bf16 per chip ("mesh" split), TILE (preferred:
        ``MotifLMHead.decode_logits`` / ``forward_decode`` without ``row_major``) or ROW_MAJOR (tilized here, one more
        op). Uses the parameters and counters written last. Returns :class:`SamplerOutput` (fresh device tensors).
        ``debug=True`` (eager only) also keeps the intermediates (``V``, ``I``, ``SV``, ``sidx``, ``PR``, ``EX``,
        ...)."""
        tmp: List[Any] = []
        d = self.stages(logits, tmp)
        tokens, info = self._stage_outputs(d, tmp)
        dbg = None
        keep_ids = set()
        if debug:
            dbg = {k: d[k] for k in ("V", "I", "M", "Z", "SV", "sidx", "PR", "EX", "keep", "kchip", "last_val", "u")}
            if self.logprobs:
                dbg["Z1"] = d["Z1"]
            keep_ids = {id(t) for t in dbg.values()}
        _free_unique(tmp, keep=keep_ids | {id(logits)} | ({id(self._u)} if self._u is not None else set()))
        return SamplerOutput(tokens=tokens, info=info, debug=dbg)

    # ---- host reads -------------------------------------------------------------------------------------------
    def _reader(self, t) -> HostShardReader:
        key = HostShardReader.spec_key(t)
        r = self._readers.get(key)
        if r is None:
            r = self._readers[key] = HostShardReader(self.mesh_device, t)
        return r

    def read(self, out: SamplerOutput, *, all_chips: bool = False) -> SampleResult:
        """Blocking read of ``out.info`` from ONE chip (1 KB; ~0.2-0.3 ms: a read of every chip's copy costs ~1-3 ms
        because it ends in a full-mesh finish) -> :class:`SampleResult`. Waits for the step that produced it.
        ``all_chips``: read every chip's copy (persistent staging) and attach them as ``res.per_chip`` ``[n_chips, 8,
        32]`` (tests: identical on all chips)."""
        if all_chips:
            r = self._reader(out.info)
            ttnn.copy_device_to_host_tensor(out.info, r.host, blocking=True)
            res = SampleResult.from_info(r.views[0])
            res.per_chip = torch.stack([v.reshape(len(INFO_ROWS), -1).float().clone() for v in r.views])
        else:
            res = SampleResult.from_info(ttnn.to_torch(ttnn.get_device_tensors(out.info)[0]))
        self.stats["steps"] += 1
        nf = int(res.flags.sum())
        self.stats["flagged_lanes"] += nf
        self.stats["flagged_steps"] += int(nf > 0)
        return res

    def resolve(
        self,
        res: SampleResult,
        logits_host: Union[torch.Tensor, Callable[[], torch.Tensor]],
        *,
        lanes: Optional[Sequence[int]] = None,
        active: Optional[Sequence[bool]] = None,
    ) -> SampleResult:
        """Exact host fallback (in place): every flagged lane (or ``lanes``) gets :func:`fallback_sample` of its host
        logits row with the step's device uniform ``res.u`` (24-bit, exact in fp32; bit-identical to the host twin of
        the step's counters) and its raw logprob, under the lane parameters set last. ``logits_host``: ``[32, vocab]``
        in lane order (``MotifLMHead.logits_to_host``) or a callable returning it (called only when a lane needs it).
        ``active``: lanes to consider (default all; e.g. ``positions >= 0``)."""
        todo = [int(l) for l in (lanes if lanes is not None else torch.nonzero(res.flags).reshape(-1).tolist())]
        if active is not None:
            todo = [l for l in todo if bool(active[l])]
        if not todo:
            return res
        lg = logits_host() if callable(logits_host) else logits_host
        if tuple(lg.shape) != (self.lanes, self.vocab_size):
            raise ValueError(f"host logits must be [{self.lanes}, {self.vocab_size}], got {tuple(lg.shape)}")
        u = res.u
        for l in todo:
            lane = self._lane_params[l]
            tok = fallback_sample(lg[l], lane, float(u[l]))
            res.tokens[l] = tok
            res.logprobs[l] = raw_logprob(lg[l], tok)
            res.resolved.append(l)
        self.stats["resolved_lanes"] += len(todo)
        self.stats["resolved_steps"] += 1
        return res

    # ---- coverage counter -------------------------------------------------------------------------------------
    def coverage_counts(self) -> Optional[torch.Tensor]:
        """Per-lane number of flagged steps since construction / :meth:`reset_coverage` (the device counter; a
        blocking read of one chip). None without ``coverage_counter``."""
        if self._counter is None:
            return None
        return ttnn.to_torch(ttnn.get_device_tensors(self._counter)[0]).float().reshape(-1)[: self.lanes].clone()

    def reset_coverage(self) -> None:
        if self._counter is not None:
            z = torch.zeros(1, 1, self.lanes, 1)
            ttnn.copy_host_to_device_tensor(self._host_tensor(z, ttnn.float32), self._counter)

    # ---- teardown -----------------------------------------------------------------------------------------------
    def deallocate(self) -> None:
        """Free every persistent device tensor (constants, parameters, counter). Release traces holding
        :meth:`sample` first."""
        ts = [self._iota_w, self._iota_n, self._mask_last, self._e0, self._e1, self._sel_z, self._sel_z1, self._U,
              self._e0_64, self._sel_z64,
              self._m1, self._m2, self._chip_off, *self._p.values()]
        ts += [t for t in (self._ctr, self._u, self._counter) if t is not None]
        for t in ts:
            ttnn.deallocate(t)
        self._readers.clear()


def _UP(op, *param):
    """``ttnn.UnaryWithParam`` shorthand for fused activations."""
    return ttnn.UnaryWithParam(op, *param) if param else ttnn.UnaryWithParam(op)


def _free_unique(ts, keep=frozenset()) -> None:
    """Deallocate each distinct tensor of ``ts`` once (identity), except those whose ``id`` is in ``keep``."""
    seen = set()
    for t in ts:
        i = id(t)
        if i in seen or i in keep or t is None:
            continue
        seen.add(i)
        ttnn.deallocate(t)


__all__ = [
    "CERT_MARGIN",
    "DEFAULT_LOCAL_K",
    "DEFAULT_WINDOW",
    "GOLDEN32",
    "GREEDY_LANE",
    "INFO_INDEX",
    "INFO_ROWS",
    "LaneParams",
    "MotifDeviceSampler",
    "SAMPLING_EPS",
    "SampleResult",
    "TOP_P_OFF",
    "UndecidedOrder",
    "SamplerOutput",
    "device_param_columns",
    "emulate_device",
    "exact_nucleus",
    "exact_sample",
    "fallback_sample",
    "greedy_token",
    "inverse_cdf",
    "k_offset_rows",
    "lane_counter",
    "lane_lists_from_rows",
    "lowbias32",
    "normalize_lane",
    "normalize_lanes",
    "raw_logprob",
    "seed_key",
    "sorted_order",
    "uniform_from_counter",
]
