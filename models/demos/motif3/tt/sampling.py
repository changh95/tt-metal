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
* **full vocabulary** (sampled, ``top_p = 1``, top-k off; lead decision 4): **Gumbel-max over all 220,160 tokens** --
  ``token = argmax_v (x_v - M) / T - log E_v`` with ``E_v = -log(1 - W_v)`` an Exp(1) variate from a per-(seed,
  position, token id) hash (below): an exact sample of ``softmax(x / T)`` (the exponential race), computed as a
  per-chip local argmax over the chip's 6,880-token slice and a global argmax over the 32 chips. Such a lane never
  falls back to the host.

RNG (counter based, slot addressable, no hidden device state)
-------------------------------------------------------------
Two 32-bit counters per lane and step, written by :meth:`MotifDeviceSampler.set_positions` (``pos`` = the step's input
position, i.e. the decode ``start_pos``; the sampled token is the one for ``pos + 1``)::

    c1 = k1 + pos * 0x9E3779B1,  c2 = k2 + pos * 0x85EBCA6B  (mod 2^32),  (k1, k2) = splitmix64(seed) (high, low)

so a seeded lane's draws are a pure function of the 64-bit ``(seed, position)``: identical across lanes, batches,
replays, traced / eager and lane relocations. Unseeded requests (``seed=None``): 64 counter bits every step from the
sampler's host RNG (``rng_seed``; the vLLM bridge passes vLLM's ``--seed``, so a server run repeats its unseeded draws
for the same sequence of steps, as vLLM's host sampler does with its global torch generator; None = OS entropy).

* inverse-CDF lanes (device): ``u = (lowbias32(c1) >> 8) * 2^-24`` (``lowbias32``: 3 xor-shifts, 2 exact int32
  multiplies, on device; :func:`uniform_from_counter` is the bit-exact host twin).
* host fallback (lead decision 5): the 64-bit stream ``U = ((lowbias32(c1) << 32 | lowbias32(c2)) >> 11) * 2^-53``
  (:func:`uniform64`): its top 24 bits ARE the device ``u`` (``floor(U * 2^24) = u * 2^24``), the rest refine the draw
  inside the device's 2^-24 cell, so the fallback picks the device token whenever no CDF step falls inside that cell,
  and resolves the 24-bit grid otherwise (53-bit resolution instead of 24: no token of a large support is undrawable).
* Gumbel lanes: per token id ``v`` two words ``h1 = lowbias32(c1 ^ K_v)`` (``K_v = lowbias32(v + 0x632BE5AB)``, a
  per-chip constant) and ``h2 = mix(h1 ^ c2)`` (``mix(y) = y * 0x846CA68B; y ^ (y >> 16)``), then ``W = ((h1 >> 8) +
  ((h2 >> 9) + 1/2) * 2^-23) * 2^-24`` in ``(0, 1]``: 47 bits, uniform with relative precision ~2^-24 down to 2^-48,
  so ``E = -log1p(-W)`` (the device's accurate fp32 ``log1p``) is exact in relative terms where the race is decided
  (small ``E``). :func:`gumbel_words` / :func:`gumbel_sample` are the host twin (fp64).

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
   * the walk ends inside the window: ``k < W`` (top-k; at ``k = W`` the k-th value IS the window's last value, so
     the first condition can never hold) or window mass ``>= p_eff + 2^-20`` (top-p; ``top_p = 1`` without top-k,
     whose support is the whole vocabulary, never passes: it takes the Gumbel path below instead);
   * the decisions are exact (sampled lanes): the last kept token's exclusive prefix ``mass_kept - PR[n - 1] <
     p_eff``, the first dropped one's ``mass_kept >= p_eff`` (unless the top-k set is exhausted), and ``EX[j] <=
     u * mass_kept < EX[j] + PR[j]`` with ``EX[j] = sum(PR[:j])`` -- all exact fp32 SFPU reductions.
   A flagged lane's device token must be replaced (:meth:`MotifDeviceSampler.resolve`: exact host resample from the
   logits with the same uniform, refined to 53 bits: :func:`uniform64`).
7b. Gumbel lanes (``gumbel``, default on): per chip, over the 6,880 logits, the two hash words, ``W``, ``F = log1p(-W)
   = -E``, the score ``s = (x - M) / T - log(-F)`` (one fused op), its max, the lowest local id attaining it and the
   raw ``x - M`` there; the (score, global id, raw) triples of all chips are gathered (one all-gather pair of a
   ``[96, 32]`` tile) and the global winner is the max score (lowest id among exact ties). Blended per lane: token,
   ``flag = 0`` (exact by construction), raw logprob, ``n_kept = vocab``. ~0.5 ms per step, every step.
8. outputs: ``tokens [1, 1, 1, 32]`` uint32 ROW_MAJOR (lane order, identical on every chip; the ``argmax_decode``
   contract) and ``info [1, 1, 8, 32]`` fp32 ROW_MAJOR (rows :data:`INFO_ROWS`: token, flag, raw logprob, n_kept, u,
   mass_kept, window mass, k-th value), read from one chip with one small transfer (:meth:`read`); optional per-lane
   cumulative flag counter (:meth:`coverage_counts`).

Exactness (DEVICE_SAMPLER.md §3-§4): on every certified lane the kept set equals the fp64 nucleus (up to boundary ties)
and the token equals the fp64 inverse CDF fed the same ``u``. Candidate-coverage flags at K = 64 (all saved real rows
through the host twin :func:`emulate_device`): 0.029 % of the rows of Motif's own T=1.0 traces (~0.9 % of 32-lane
steps), 0 at T = 0.6, 0.76 % of teacher-forced assistant rows at T = 1.0. In the real 53-layer decode trace all flags
together (coverage + exactness checks, after the cut repair) hit 0.04 % of the lane-steps with ``top_p < 1``. Gumbel
lanes: the device token equals the fp64 Gumbel-max of the same hash words (:func:`gumbel_sample`) except where two
scores agree to fp32 rounding (~1e-7 relative); the distribution is ``softmax(x / T)`` up to that rounding and the
47-bit ``W`` (DEVICE_SAMPLER.md §10).

Top-k lanes (with or without top-p) are certified only when ``k < W`` and no chip holds 64 (``local_k``) or more of
the top-k set: Motif's vocabulary is split into 32 contiguous 6,880-token blocks and the top tokens of a row cluster
by id. On the 32 real rows every row is certified at ``top_k <= 64``; 2 / 6 / 11 / 17 / 28 / 31 of the 32 rows are
flagged at ``top_k`` 100 / 128 / 200 / 256 / 400 / 500, and every row at ``top_k >= 511`` (also with ``top_p = 1``:
such a lane is a top-k lane, not a Gumbel lane). Those lanes stay exact through the host fallback, at ~7 ms of host
time per flagged lane (DEVICE_SAMPLER.md §2, §7).

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
GOLDEN32 = 0x9E3779B1  # Weyl increment of the first per-lane counter: c1 = k1 + pos * GOLDEN32 (mod 2^32)
GOLDEN32_2 = 0x85EBCA6B  # Weyl increment of the second counter: c2 = k2 + pos * GOLDEN32_2 (odd: full period)
LOWBIAS_M1 = 0x21F0AAAD  # lowbias32 (C. Wellons) multipliers
LOWBIAS_M2 = 0x735A2D97
GUMBEL_MIX = 0x846CA68B  # odd multiplier of the Gumbel second word: h2 = y * MIX; h2 ^= h2 >> 16 (y = h1 ^ c2)
GUMBEL_KEY_SALT = 0x632BE5AB  # per-token key K_v = lowbias32(v + salt) (mod 2^32)
CTR2_SALT = 0x5851F42D  # c2 derived from c1 when a caller gives only c1 (tests / replays): lowbias32(c1 ^ salt)
U24 = 2.0**-24
U53 = 2.0**-53
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


def _splitmix64(seed: int) -> int:
    z = (int(seed) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def seed_keys(seed: int) -> Tuple[int, int]:
    """The two 32-bit keys ``(k1, k2)`` of a request seed (any Python int; vLLM seeds are 64-bit): the high and low
    halves of splitmix64(seed mod 2^64), a bijection of the 64-bit seed (two seeds never share both keys)."""
    z = _splitmix64(seed)
    return int(z >> 32), int(z & MASK32)


def seed_key(seed: int) -> int:
    """32-bit key ``k1`` of a request seed: the high half of splitmix64(seed mod 2^64) (:func:`seed_keys`).
    Deterministic and well mixed, so neighbouring seeds give unrelated lanes."""
    return seed_keys(seed)[0]


def lane_counter(key: int, position: int) -> int:
    """The first per-step counter of a seeded lane: ``(k1 + position * 0x9E3779B1) mod 2^32``."""
    return (int(key) + int(position) * GOLDEN32) & MASK32


def lane_counters(seed: int, position: int) -> Tuple[int, int]:
    """Both per-step counters ``(c1, c2)`` of a lane with ``seed`` at ``position`` (the decode input position)."""
    k1, k2 = seed_keys(seed)
    pos = max(int(position), 0)
    return (k1 + pos * GOLDEN32) & MASK32, (k2 + pos * GOLDEN32_2) & MASK32


def derived_ctr2(c1) -> np.ndarray:
    """The second counter assumed for callers that give only ``c1`` (tests, replays of 1-D counter arrays):
    ``lowbias32(c1 ^ CTR2_SALT)``. Serving always writes both counters (:meth:`MotifDeviceSampler.counters_for`)."""
    return lowbias32((np.asarray(c1, dtype=np.uint64) & MASK32) ^ np.uint64(CTR2_SALT))


def as_counter_pairs(ctr) -> np.ndarray:
    """``uint32 [n, 2]`` counters ``(c1, c2)`` from ``[n, 2]`` pairs or ``[n]`` first counters (``c2`` derived)."""
    a = np.asarray(ctr.numpy() if isinstance(ctr, torch.Tensor) else ctr, dtype=np.uint64) & np.uint64(MASK32)
    if a.ndim == 1:
        a = np.stack([a, derived_ctr2(a).astype(np.uint64)], axis=1)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"counters must be [n] or [n, 2], got {tuple(a.shape)}")
    return a.astype(np.uint32)


def uniform64(ctr) -> np.ndarray:
    """The host fallback's uniform (lead decision 5): float64 ``[n]`` in ``[0, 1)`` with 53 random bits from the
    64-bit stream ``lowbias32(c1) << 32 | lowbias32(c2)`` of the lane's counters (``ctr`` as :func:`as_counter_pairs`).
    Its top 24 bits are the device uniform: ``floor(uniform64 * 2^24) == uniform_from_counter(c1) * 2^24`` exactly, so
    the fallback draw lies in the device draw's 2^-24 cell and only refines it."""
    c = as_counter_pairs(ctr)
    hi = lowbias32(c[:, 0]).astype(np.uint64)
    lo = lowbias32(c[:, 1]).astype(np.uint64)
    k = ((hi << np.uint64(32)) | lo) >> np.uint64(11)  # 53 bits, exact in float64
    return k.astype(np.float64) * U53


def gumbel_token_keys(ids) -> np.ndarray:
    """``K_v = lowbias32(v + GUMBEL_KEY_SALT)`` (uint32) of token ids ``v``: the per-token input of the Gumbel hash
    (a persistent per-chip device constant)."""
    v = np.asarray(ids, dtype=np.uint64) & np.uint64(MASK32)
    return lowbias32((v + np.uint64(GUMBEL_KEY_SALT)) & np.uint64(MASK32))


def gumbel_words(c1: int, c2: int, ids) -> Tuple[np.ndarray, np.ndarray]:
    """The two Gumbel hash words ``(h1, h2)`` (uint32 ``[len(ids)]``) of token ids for one lane's counters:
    ``h1 = lowbias32(c1 ^ K_v)``, ``y = h1 ^ c2``, ``h2 = (y * GUMBEL_MIX) ^ ((y * GUMBEL_MIX) >> 16)`` (mod 2^32).
    Bit-exact with the device."""
    k = gumbel_token_keys(ids).astype(np.uint64)
    h1 = lowbias32(k ^ np.uint64(int(c1) & MASK32)).astype(np.uint64)
    y = (h1 ^ np.uint64(int(c2) & MASK32)) * np.uint64(GUMBEL_MIX) & np.uint64(MASK32)
    h2 = y ^ (y >> np.uint64(16))
    return h1.astype(np.uint32), h2.astype(np.uint32)


def gumbel_uniform(h1, h2) -> np.ndarray:
    """``W = ((h1 >> 8) + ((h2 >> 9) + 1/2) * 2^-23) * 2^-24`` in float64 (exact: 47 significant bits), in ``(0, 1)``:
    uniform with resolution 2^-47 below 2^-24 and relative precision ~2^-24 above (the device rounds the sum to fp32,
    which matters only near ``W = 1``, where ``E = -log(1 - W)`` is large and the race is not decided)."""
    a = (np.asarray(h1, dtype=np.uint64) >> np.uint64(8)).astype(np.float64)
    b = (np.asarray(h2, dtype=np.uint64) >> np.uint64(9)).astype(np.float64)
    return (a + (b + 0.5) * 2.0**-23) * U24


def gumbel_scores(row: torch.Tensor, lane: "LaneParams", ctr_pair, *, ids=None) -> Tuple[np.ndarray, np.ndarray]:
    """fp64 Gumbel-max scores ``s_v = (x_v - M) / T - log(-log1p(-W_v))`` of a full-support lane (the device's race,
    exactly): returns ``(ids, scores)``. ``ids`` (default every token) may be any superset of the tokens that can win,
    e.g. the finite-logit tokens of a crafted row."""
    x = row.reshape(-1).double().numpy()
    if ids is None:
        ids = np.arange(x.shape[0], dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    c1, c2 = (int(v) for v in np.asarray(ctr_pair, dtype=np.uint64).reshape(-1)[:2])
    h1, h2 = gumbel_words(c1, c2, ids)
    w = gumbel_uniform(h1, h2)
    t = 1.0 if lane.greedy else float(lane.temperature)
    with np.errstate(divide="ignore"):
        s = (x[ids] - x.max()) / t - np.log(-np.log1p(-w))
    return ids, s


def gumbel_sample(row: torch.Tensor, lane: "LaneParams", ctr_pair, *, ids=None) -> int:
    """The host twin of a Gumbel lane's draw (fp64): the token with the largest :func:`gumbel_scores` (lowest id
    among exact ties) -- an exact sample of ``softmax(x / T)`` keyed by the lane's counters ``(c1, c2)``."""
    ids, s = gumbel_scores(row, lane, ctr_pair, ids=ids)
    best = np.flatnonzero(s == s.max())
    return int(ids[best].min())


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


def device_param_columns(lanes: Sequence[LaneParams], window: int, *, gumbel: bool = True) -> Dict[str, torch.Tensor]:
    """The per-lane device parameter columns (float32 ``[n]`` each) of :class:`MotifDeviceSampler`:

    * ``greedy`` 0/1; ``inv_t`` 1/T (1 for greedy lanes, which keeps the normaliser finite);
    * ``top_p`` clamped to ``[MIN_TOP_P, 1)`` (the first token is always kept); ``top_p >= 1`` (top-p off) becomes
      :data:`TOP_P_OFF` = 2, so the walk keeps the whole top-k set: the fp32 exclusive prefix saturates at 1.0 and
      would otherwise drop the top-k tokens below fp32 resolution that vLLM keeps (its ascending cumsum is > 0);
      greedy lanes get ``MIN_TOP_P`` (their certificate's window-mass condition then always holds);
    * ``k_active`` 0/1 (sampled lane with top-k); ``k_idx`` = ``min(k, W) - 1`` (0 when off);
    * ``k_in_win`` 0/1 (top-k active and ``k < W``; a larger k is flagged for the host -- ``k = W`` too: its k-th
      value is the window's last value, which the coverage certificate needs strictly below it);
    * ``gumbel`` 0/1: a full-support lane (sampled, top-k off, ``top_p >= 1``) whose token is the full-vocab
      Gumbel-max (only with ``gumbel``; otherwise such a lane is flagged for the host)."""
    W = int(window)
    g = torch.tensor([1.0 if l.greedy else 0.0 for l in lanes], dtype=torch.float32)
    inv_t = torch.tensor([l.inv_temperature for l in lanes], dtype=torch.float64).to(torch.float32)
    top_p = torch.tensor(
        [MIN_TOP_P if l.greedy else (TOP_P_OFF if l.top_p >= 1.0 else max(l.top_p, MIN_TOP_P)) for l in lanes],
        dtype=torch.float32,
    )
    ka = torch.tensor([1.0 if l.top_k_active else 0.0 for l in lanes], dtype=torch.float32)
    kidx = torch.tensor([float(min(l.top_k, W) - 1) if l.top_k_active else 0.0 for l in lanes], dtype=torch.float32)
    kin = torch.tensor([1.0 if (l.top_k_active and l.top_k < W) else 0.0 for l in lanes], dtype=torch.float32)
    gm = torch.tensor([1.0 if (gumbel and l.full_support) else 0.0 for l in lanes], dtype=torch.float32)
    return {"greedy": g, "inv_t": inv_t, "top_p": top_p, "k_active": ka, "k_idx": kidx, "k_in_win": kin,
            "gumbel": gm}  # fmt: skip


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
    """The host fallback for a flagged lane: an exact sample with the lane's uniform ``u`` (:meth:`MotifDeviceSampler.
    resolve` passes the 53-bit :func:`uniform64` of the step's counters, whose top 24 bits are the device ``u``).

    Equal to :func:`exact_sample` (reference order: logit desc, id asc), computed fast (~1-3 ms per row): greedy ->
    argmax; otherwise the walk over the tokens strictly above the ``prefix``-th largest logit (``torch.topk``) when
    that prefix decides the kept set, else a full sort. Full-vocabulary support (``top_p = 1``, no top-k) uses the
    inverse CDF in **token-id order** instead (no sort: an exact sample of the same distribution, another u -> token
    map; only for a sampler without the Gumbel path -- with it such lanes are never flagged, and their host twin is
    :func:`gumbel_sample`). Deterministic in ``(row, lane, u)``."""
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
    ctr,
    *,
    local_k: int = DEFAULT_LOCAL_K,
    window: int = DEFAULT_WINDOW,
    num_chips: int = 32,
    gumbel: bool = True,
) -> Dict[str, torch.Tensor]:
    """fp32 torch emulation of :meth:`MotifDeviceSampler.sample` on host logits ``[B, V]`` (bf16) for lanes ``lanes``
    and counters ``ctr`` (``[B]`` first counters or ``[B, 2]`` pairs, :func:`as_counter_pairs`): the same candidate
    sets, normaliser, window, masks, draw and certificate, with the reference tie order (logit desc, id asc) where the
    device uses ``ttnn.topk``'s. Full-support lanes take the Gumbel path with ``gumbel`` (token = the fp64
    :func:`gumbel_sample`, flag 0, ``n_kept`` = vocab). Returns ``token, flag, n_kept, u, mass_kept, window_mass, kth,
    logprob`` (``[B]`` each). Accumulation order differs from the device (sums are fp32 either way), so results agree
    except at measure-zero rounding boundaries and tie permutations."""
    x = logits.float()
    B, V = x.shape
    Vc = V // num_chips
    K, W = int(local_k), int(window)
    cp = as_counter_pairs(ctr)
    if cp.shape[0] != B:
        raise ValueError(f"{cp.shape[0]} counters for {B} rows")
    cols = device_param_columns(lanes, W, gumbel=gumbel)
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
    u = torch.from_numpy(uniform_from_counter(cp[:, 0])).view(B, 1)
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
    gl = torch.nonzero(cols["gumbel"] > 0).reshape(-1).tolist()
    if gl:  # full-support lanes: the full-vocab Gumbel-max (fp64 twin of the device race), never flagged
        tok = tok.clone()
        for l in gl:
            t = gumbel_sample(logits[l], lanes[l], cp[l])
            tok[l, 0] = float(t)
            x_tok[l, 0] = x[l, t]
            flag[l, 0] = 0.0
            n_kept[l, 0] = float(V)
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
    ``n_kept`` / ``mass_kept`` / ``window_mass`` / ``kth`` / ``u`` float32 (diagnostics; ``n_kept`` = vocab on Gumbel
    lanes), ``resolved``: lanes whose token came from the host fallback, ``counters``: the step's ``uint32 [32, 2]``
    RNG counters (attached by :meth:`MotifDeviceSampler.read`; the host fallback's 53-bit uniform and the Gumbel twin
    derive from them)."""

    tokens: torch.Tensor
    flags: torch.Tensor
    logprobs: torch.Tensor
    n_kept: torch.Tensor
    u: torch.Tensor
    mass_kept: torch.Tensor
    window_mass: torch.Tensor
    kth: torch.Tensor
    resolved: List[int] = field(default_factory=list)
    counters: Optional[np.ndarray] = None

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
        rng: ``"device"`` (default: the counter hash runs inside the trace; the host writes the two int32 counters per
            lane per step, one ``[32, 32]`` tile) or ``"host"`` (the host writes the uniform itself, bit-identical
            values; 11 fewer device ops; with ``gumbel`` the counters are written too).
        gumbel: the full-vocab Gumbel-max path for ``top_p = 1`` lanes without top-k (lead decision 4; default on,
            ~0.5 ms per step). Off: such lanes are flagged and sampled on the host every step.
        coverage_counter: keep a persistent per-lane count of flagged steps on the device (:meth:`coverage_counts`).
        rng_seed: seed of the host RNG behind unseeded lanes' counters (None = OS entropy; the vLLM bridge passes
            vLLM's ``--seed``).
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
        gumbel: bool = True,
        coverage_counter: bool = True,
        rng_seed: Optional[int] = None,
        memory_config=None,
        l1_intermediates: bool = False,
        impl: Optional[Dict[str, Any]] = None,
    ):
        if int(max_lanes) != TILE or int(cfg.max_batch) != TILE:
            raise ValueError(
                f"the sampler runs the {TILE} decode lanes of the trace (max_lanes={max_lanes}, "
                f"cfg.max_batch={cfg.max_batch})"
            )
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
        self.gumbel = bool(gumbel)
        self.memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        self.imc = ttnn.L1_MEMORY_CONFIG if l1_intermediates else ttnn.DRAM_MEMORY_CONFIG
        self.ckc = cfg.compute_config("eltwise")  # HiFi4, fp32 dest acc (accurate SFPU reductions), approx off
        self._rng = np.random.default_rng(secrets.randbits(64) if rng_seed is None else int(rng_seed) & ((1 << 64) - 1))
        self._readers: Dict[tuple, HostShardReader] = {}
        # steps / flagged_*: every read() (flags of ALL lanes: inactive and full-support lanes included);
        # resolved_steps / resolved_lanes: what resolve() actually re-sampled (the steps that paid the logits read)
        self.stats = {
            "steps": 0,
            "flagged_steps": 0,
            "flagged_lanes": 0,
            "resolved_steps": 0,
            "resolved_lanes": 0,
            "param_uploads": 0,
        }
        rep = self._rep = ttnn.ReplicateTensorToMesh(mesh_device)
        L, W, NC = self.lanes, self.W, self.NC

        def up(t, dtype, mapper=rep):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

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
        self._sel_z64 = up(
            (cols % TILE == 0).float().reshape(1, 1, 1, -1).expand(1, 1, 2 * L, -1).contiguous(), ttnn.float32
        )
        # strictly upper triangular: (PR @ Us)[j] = sum_{i < j} PR[i], the exclusive prefix
        Us = torch.triu(torch.ones(W, W), diagonal=1)
        if self.impl["prefix"] == "1mm":
            Us = torch.cat([Us, Us], dim=0)
        self._U = up(Us.reshape(1, 1, -1, W), ttnn.bfloat16)
        self._m1 = up(torch.full((1, 1, L, 1), _i32(LOWBIAS_M1), dtype=torch.int32), ttnn.int32)
        self._m2 = up(torch.full((1, 1, L, 1), _i32(LOWBIAS_M2), dtype=torch.int32), ttnn.int32)
        # per-chip vocab offset 6880 b (b = mesh linear index r * C + c, the LM head's "mesh" split)
        R, C = cfg.axes.mesh_shape
        chips = ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 3), mesh_shape=(R, C))
        off = torch.zeros(R, 1, L, C, dtype=torch.float32)
        for r in range(R):
            for c in range(C):
                off[r, 0, :, c] = float(vocab_block_of_coord(cfg, r, c, "mesh") * self.vc)
        self._chip_off = up(off, ttnn.float32, chips)
        if self.gumbel:
            # per-chip Gumbel token keys K_v (int32 bit patterns) of the chip's vocab block, the same row for every
            # lane; the local ids 0..6879; the second-word multiplier
            keys = torch.zeros(R, 1, L, C * self.vc, dtype=torch.int32)
            for r in range(R):
                for c in range(C):
                    b = vocab_block_of_coord(cfg, r, c, "mesh")
                    k = gumbel_token_keys(np.arange(b * self.vc, (b + 1) * self.vc)).view(np.int32)
                    keys[r, 0, :, c * self.vc : (c + 1) * self.vc] = torch.from_numpy(k.copy())
            self._gkey = up(keys, ttnn.int32, chips)
            self._iota_vc = up(row(torch.arange(self.vc, dtype=torch.float32)), ttnn.float32)
            self._m3 = up(torch.full((1, 1, L, 1), _i32(GUMBEL_MIX), dtype=torch.int32), ttnn.int32)
        else:
            self._gkey = self._iota_vc = self._m3 = None
        # ---- persistent per-lane parameters ([1, 1, 32, 1], replicated) ----
        z = torch.zeros(1, 1, L, 1)
        self._p = {k: up(z, ttnn.float32) for k in ("greedy", "inv_t", "top_p", "k_active", "k_in_win", "gumbel")}
        self._p["k_ofs"] = up(torch.full((1, 1, L, W), -BIG), ttnn.float32)  # top-k threshold mask (k_offset_rows)
        # RNG inputs: the two counters (c1, c2) in columns 0 / 1 of one [32, 32] int32 tile (one host write per step);
        # rng="host" writes the uniform instead (and the counters too when the Gumbel path needs them)
        self._ctr = up(torch.zeros(1, 1, L, TILE, dtype=torch.int32), ttnn.int32) if (rng == "device" or self.gumbel) \
            else None  # fmt: skip
        self._u = up(z, ttnn.float32) if rng == "host" else None
        self._counter = up(z, ttnn.float32) if coverage_counter else None
        # host-side state
        self._lane_params: List[LaneParams] = [GREEDY_LANE] * L
        self._host_cols: Optional[Dict[str, torch.Tensor]] = None
        self._last_raw = None
        self._keys: List[Optional[Tuple[int, int]]] = [None] * L
        self._last_ctr = np.zeros((L, 2), dtype=np.uint32)
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
        raw = tuple(
            tuple(v.reshape(-1).tolist()) if isinstance(v, torch.Tensor) else (None if v is None else tuple(v))
            for v in (temperature, top_p, top_k, seeds)
        )
        if not force and raw == self._last_raw:  # the common case: the plugin resends unchanged parameters
            return False
        lanes = normalize_lanes(temperature, top_p, top_k, seeds, vocab_size=self.vocab_size, num_lanes=self.lanes)
        self._last_raw = raw
        self._lane_params = lanes
        self._keys = [None if l.seed is None else seed_keys(l.seed) for l in lanes]
        self._key_mask = np.array([k is not None for k in self._keys])
        self._key_arr = np.array([(0, 0) if k is None else k for k in self._keys], dtype=np.uint64).reshape(-1, 2)
        cols = device_param_columns(lanes, self.W, gumbel=self.gumbel)
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
        return self.set_params(
            [p.temperature for p in params],
            [p.top_p for p in params],
            [p.top_k for p in params],
            [p.seed for p in params],
            force=force,
        )

    def counters_for(self, positions) -> np.ndarray:
        """The ``uint32 [32, 2]`` RNG counters ``(c1, c2)`` :meth:`set_positions` would write for ``positions`` (lane
        order): seeded lanes ``(k1 + pos * 0x9E3779B1, k2 + pos * 0x85EBCA6B)`` with ``(k1, k2) =``
        :func:`seed_keys` of the seed, unseeded lanes 64 fresh host-random bits (advances the host RNG)."""
        pos = np.asarray(positions.numpy() if isinstance(positions, torch.Tensor) else positions, dtype=np.int64)
        pos = pos.reshape(-1)
        if pos.shape[0] != self.lanes:
            raise ValueError(f"expected {self.lanes} positions, got {pos.shape[0]}")
        seeded = self._key_mask
        p = np.maximum(pos, 0).astype(np.uint64)
        inc = np.array([GOLDEN32, GOLDEN32_2], dtype=np.uint64)
        ctr = (self._key_arr + p[:, None] * inc[None, :]) & np.uint64(MASK32)
        if not seeded.all():
            rnd = self._rng.integers(0, 1 << 32, size=(self.lanes, 2), dtype=np.uint64)
            ctr = np.where(seeded[:, None], ctr, rnd)
        return ctr.astype(np.uint32)

    def set_counters(self, ctr) -> np.ndarray:
        """Write explicit RNG counters (lane order; ``[32, 2]`` pairs, or ``[32]`` first counters whose ``c2`` is
        :func:`derived_ctr2`; tests and replays): one ``[1, 1, 32, 32]`` host -> device copy (~0.15 ms). Not
        traceable. With ``rng="host"`` the uniforms of ``c1`` are written as well (bit-identical to the device
        hash). Returns the ``uint32 [32, 2]`` pairs written."""
        c = as_counter_pairs(ctr)
        if c.shape[0] != self.lanes:
            raise ValueError(f"expected {self.lanes} counters, got {c.shape[0]}")
        if self._ctr is not None:
            t = torch.zeros(1, 1, self.lanes, TILE, dtype=torch.int32)
            t[0, 0, :, :2] = torch.from_numpy(c.view(np.int32).copy())
            ttnn.copy_host_to_device_tensor(self._host_tensor(t, ttnn.int32), self._ctr)
        if self._u is not None:
            u = torch.from_numpy(uniform_from_counter(c[:, 0])).reshape(1, 1, self.lanes, 1)
            ttnn.copy_host_to_device_tensor(self._host_tensor(u, ttnn.float32), self._u)
        self._last_ctr = c
        return c

    def set_positions(self, positions) -> np.ndarray:
        """Per-step RNG input: ``positions`` = each lane's decode input position (``start_pos``; -1 / anything for
        inactive lanes) in lane order. Writes the counters (one ``[1, 1, 32, 32]`` host -> device copy) and returns
        them (``uint32 [32, 2]``). Not traceable: call it before every replay of a trace that holds :meth:`sample`."""
        return self.set_counters(self.counters_for(positions))

    def last_uniforms(self) -> np.ndarray:
        """The host twin of the device ``u`` of the counters written last (float32 ``[32]``)."""
        return uniform_from_counter(self._last_ctr[:, 0])

    def last_counters(self) -> np.ndarray:
        """The ``uint32 [32, 2]`` counters written last."""
        return self._last_ctr.copy()

    # ---- the device pipeline (stages; each returns new tensors and records its intermediates in ``tmp``) ---------
    def _red(self, op, x, dim: int = 3):
        return op(x, dim=dim, keepdim=True, compute_kernel_config=self.ckc, memory_config=self.imc)

    def _exp(self, a, b):
        """``exp(a * b)`` in fp32 (accurate exp; optionally fused as the multiply's post-activation)."""
        mc = self.imc
        if self.impl["fused_exp"]:
            return [ttnn.multiply(a, b, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, 0.0)], memory_config=mc)]
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
        a = None  # (x - M) / T, kept for the Gumbel path when it is materialized here
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
            if len(es) == 2:
                a = es[0]
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
        out = {"S": S, "dx": d, "a": a}
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

    def _lowbias32(self, h, tmp):
        """Device ``lowbias32`` of an int32 tensor (any shape broadcasting against the ``[32, 1]`` multipliers): 3
        logical xor-shifts and 2 exact int32 multiplies (mod 2^32), intermediates recorded in ``tmp``."""
        mc = self.imc
        T = tmp.append
        for shift, mul in ((16, self._m1), (15, self._m2), (15, None)):
            s = ttnn.logical_right_shift(h, shift, memory_config=mc)
            T(s)
            h = ttnn.bitwise_xor(h, s, memory_config=mc)
            T(h)
            if mul is not None:
                h = ttnn.multiply(h, mul, memory_config=mc)
                T(h)
        return h

    def _stage_counters(self, tmp) -> Dict[str, Any]:
        """The two per-lane counters ``c1`` / ``c2`` (``[1, 1, 32, 1]`` int32) sliced out of the packed counter tile."""
        if self._ctr is None:
            return {}
        L, mc = self.lanes, self.imc
        c1 = ttnn.slice(self._ctr, [0, 0, 0, 0], [1, 1, L, 1], memory_config=mc)
        c2 = ttnn.slice(self._ctr, [0, 0, 0, 1], [1, 1, L, 2], memory_config=mc)
        tmp.extend([c1, c2])
        return {"c1": c1, "c2": c2}

    def _stage_uniform(self, c1, tmp):
        """``u = (lowbias32(c1) >> 8) * 2^-24`` (``[1, 1, 32, 1]`` fp32), or the host-written uniform (rng="host")."""
        if self._u is not None:
            return self._u
        mc = self.imc
        T = tmp.append
        h = self._lowbias32(c1, tmp)
        hs = ttnn.logical_right_shift(h, 8, memory_config=mc)
        T(hs)
        hf = ttnn.typecast(hs, ttnn.float32, memory_config=mc)
        T(hf)
        u = ttnn.multiply(hf, U24, memory_config=mc)
        T(u)
        return u

    def _stage_gumbel(self, d: Dict[str, Any], tmp) -> Dict[str, Any]:
        """The full-vocab Gumbel-max of every lane (only Gumbel lanes use it; module docstring, step 7b).

        Per chip over its 6880 logits: ``h1 = lowbias32(c1 ^ K_v)``, ``h2 = mix(h1 ^ c2)``, ``W`` (47 bits, ``(0,
        1]``), ``F = log1p(-W) = -E`` (accurate fp32 log1p: relative ~1e-7 down to W = 2^-48), the score ``s = (x - M)
        / T - log(-F)`` (one subtract with the fused ``[NEG, LOG]`` input activations), the local max, the lowest local
        id attaining it and the raw ``x - M`` there. The per-chip triples (score, global id, raw) are packed as three
        row blocks of one ``[96, 32]`` tile (column 0), all-gathered over TP then DP, and reduced: ``g_score`` = the
        global max, ``g_tok`` = the lowest global id among the chips attaining it, ``g_dx`` = its raw ``x - M``."""
        mc, P, red = self.imc, self._p, self._red

        def T(t):
            tmp.append(t)
            return t

        dx = d["dx"]
        a = d.get("a")
        if a is None:
            a = T(ttnn.multiply(dx, P["inv_t"], memory_config=mc))
        h1 = self._lowbias32(T(ttnn.bitwise_xor(self._gkey, d["c1"], memory_config=mc)), tmp)
        y = T(ttnn.multiply(T(ttnn.bitwise_xor(h1, d["c2"], memory_config=mc)), self._m3, memory_config=mc))
        h2 = T(ttnn.bitwise_xor(y, T(ttnn.logical_right_shift(y, 16, memory_config=mc)), memory_config=mc))
        A = T(ttnn.typecast(T(ttnn.logical_right_shift(h1, 8, memory_config=mc)), ttnn.float32, memory_config=mc))
        B = T(ttnn.typecast(T(ttnn.logical_right_shift(h2, 9, memory_config=mc)), ttnn.float32, memory_config=mc))
        Bf = T(ttnn.add(B, 0.5, activations=[_UP(ttnn.UnaryOpType.MUL_UNARY_SFPU, 2.0**-23)], memory_config=mc))
        Wn = T(ttnn.add(A, Bf, activations=[_UP(ttnn.UnaryOpType.MUL_UNARY_SFPU, -U24)], memory_config=mc))  # -W
        F = T(ttnn.log1p(Wn, memory_config=mc))  # log(1 - W) = -E
        s = T(
            ttnn.subtract(
                a,
                F,
                input_tensor_b_activations=[_UP(ttnn.UnaryOpType.NEG), _UP(ttnn.UnaryOpType.LOG)],
                memory_config=mc,
            )
        )  # (x - M) / T - log E
        m = T(red(ttnn.max, s))
        nm = T(ttnn.lt(s, m, memory_config=mc))  # 1 strictly below the local max
        li = T(red(ttnn.min, T(ttnn.where(nm, ID_SENTINEL, self._iota_vc, memory_config=mc))))
        dw = T(red(ttnn.max, T(ttnn.where(nm, -BIG, dx, memory_config=mc))))
        gid = T(ttnn.add(li, self._chip_off, memory_config=mc))  # global id, exact in fp32 (< 2^24)
        q = T(
            ttnn.concat(
                [T(ttnn.multiply(self._e0, v, memory_config=mc)) for v in (m, gid, dw)], dim=2, memory_config=mc
            )
        )  # [1, 1, 96, 32]: score / id / raw blocks, column 0
        G = T(self.ccl.all_gather(T(self.ccl.all_gather(q, 3, "tp", memory_config=mc)), 3, "dp", memory_config=mc))
        L = self.lanes
        Gs, Gi, Gd = (T(ttnn.slice(G, [0, 0, k * L, 0], [1, 1, (k + 1) * L, int(G.shape[3])], memory_config=mc))
                      for k in range(3))  # fmt: skip
        gmax = T(red(ttnn.max, T(ttnn.where(self._sel_z, Gs, -BIG, memory_config=mc))))
        win = T(ttnn.multiply(T(ttnn.eq(Gs, gmax, memory_config=mc)), self._sel_z, memory_config=mc))
        g_tok = T(red(ttnn.min, T(ttnn.where(win, Gi, BIG, memory_config=mc))))
        g_dx = T(red(ttnn.max, T(ttnn.where(win, Gd, -BIG, memory_config=mc))))
        return {"g_tok": g_tok, "g_dx": g_dx, "g_score": gmax}

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
        p_eff = T(
            ttnn.multiply(P["top_p"], T(ttnn.where(P["k_active"], mass_k, 1.0, memory_config=mc)), memory_config=mc)
        )
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
        u = self._stage_uniform(d.get("c1"), tmp)
        target = T(ttnn.multiply(u, mass_kept, memory_config=mc))
        cnt = T(red(ttnn.sum, T(ttnn.le(EX, target, memory_config=mc))))
        if fuse:
            j = T(ttnn.minimum(cnt, n_kept, activations=[_UP(ttnn.UnaryOpType.ADD_UNARY_SFPU, -1.0)], memory_config=mc))
        else:
            j = T(ttnn.subtract(T(ttnn.minimum(cnt, n_kept, memory_config=mc)), 1.0, memory_config=mc))
        oh_j = T(ttnn.eq(self._iota_w, j, memory_config=mc))
        pstar = T(red(ttnn.sum, T(ttnn.multiply(d["sidx"], oh_j, memory_config=mc))))  # column in the candidate row
        tok_s = T(
            red(ttnn.sum, T(ttnn.multiply(I, T(ttnn.eq(self._iota_n, pstar, memory_config=mc)), memory_config=mc)))
        )
        # greedy: the lowest id among the maxima (exact fp32 min over ids < 2^24)
        g = T(red(ttnn.min, T(ttnn.where(T(ttnn.lt(V, M, memory_config=mc)), ID_SENTINEL, I, memory_config=mc))))
        tok = T(ttnn.where(P["greedy"], g, tok_s, memory_config=mc))
        # certificate: every token at or above the lane's decisive value is a candidate inside the window
        # (greedy: the max; top-k: the k-th value; top-p: the last kept value), and the walk ends inside the window
        # (top-p: window mass >= p_eff + margin; top-k: k < W)
        kchip = T(red(ttnn.max, T(ttnn.where(self._mask_last, V, -BIG, memory_config=mc))))  # max_c (K-th of chip c)
        last_val = T(red(ttnn.min, T(ttnn.where(keep, SV, BIG, memory_config=mc))))
        total = T(red(ttnn.sum, PR))
        win_last = T(red(ttnn.min, SV))
        vthr = T(
            ttnn.where(P["greedy"], M, T(ttnn.where(P["k_active"], kth, last_val, memory_config=mc)), memory_config=mc)
        )
        c1 = T(ttnn.lt(T(ttnn.maximum(kchip, win_last, memory_config=mc)), vthr, memory_config=mc))
        if fuse:  # (total - p_eff) >= margin
            reach = T(
                ttnn.subtract(total, p_eff, activations=[_UP(ttnn.UnaryOpType.UNARY_GE, CERT_MARGIN)], memory_config=mc)
            )
        else:
            reach = T(ttnn.ge(total, T(ttnn.add(p_eff, CERT_MARGIN, memory_config=mc)), memory_config=mc))
        c2 = T(ttnn.where(P["k_active"], P["k_in_win"], reach, memory_config=mc))
        # exactness of the two prefix decisions (the matmul prefix EX accumulates inside the FPU below fp32 precision
        # -- ~10 significant bits observed): re-check the cut and the draw with exact fp32 SFPU sums.
        #   cut:  the last kept token's exclusive prefix (mass_kept - PR[n-1]) < p_eff, and the first dropped one's
        #         (mass_kept) >= p_eff unless the whole top-k set is kept;
        #   draw: EX[j] <= target < EX[j] + PR[j] with EX[j] = sum(PR[:j]).
        if not self.impl["verify"]:  # measurement only (DEVICE_SAMPLER.md §3.3): NOT exact without the check
            return self._finish_select(
                d,
                tmp,
                tok=tok,
                c1=c1,
                c2=c2,
                n_kept=n_kept,
                u=u,
                mass_kept=mass_kept,
                total=total,
                kth=kth,
                keep=keep,
                kchip=kchip,
                last_val=last_val,
                oh_j=oh_j,
            )
        pr_last = T(red(ttnn.min, T(ttnn.where(keep, PR, BIG, memory_config=mc))))
        v_a = T(ttnn.lt(T(ttnn.subtract(mass_kept, pr_last, memory_config=mc)), p_eff, memory_config=mc))
        # first dropped token: mass_kept >= p_eff, or the whole top-k set kept (mass_kept == mass_k bitwise: the same
        # elements summed in the same order); greedy lanes pass all three checks by construction (p_eff ~ 0)
        v_b = T(ttnn.ge(mass_kept, T(ttnn.minimum(p_eff, mass_k, memory_config=mc)), memory_config=mc))
        ex_j = T(red(ttnn.sum, T(ttnn.multiply(PR, T(ttnn.lt(self._iota_w, j, memory_config=mc)), memory_config=mc))))
        pr_j = T(red(ttnn.sum, T(ttnn.multiply(PR, oh_j, memory_config=mc))))
        if fuse:  # dlt = target - EX[j]: 0 <= dlt < PR[j]
            dlt = T(ttnn.subtract(target, ex_j, memory_config=mc))
            v_c = T(
                ttnn.multiply(
                    dlt,
                    T(ttnn.lt(dlt, pr_j, memory_config=mc)),
                    input_tensor_a_activations=[_UP(ttnn.UnaryOpType.GEZ)],
                    memory_config=mc,
                )
            )
        else:
            v_c = T(
                ttnn.multiply(
                    T(ttnn.le(ex_j, target, memory_config=mc)),
                    T(ttnn.lt(target, T(ttnn.add(ex_j, pr_j, memory_config=mc)), memory_config=mc)),
                    memory_config=mc,
                )
            )
        v = T(ttnn.multiply(T(ttnn.multiply(v_a, v_b, memory_config=mc)), v_c, memory_config=mc))
        c2 = T(ttnn.multiply(c2, v, memory_config=mc))
        return self._finish_select(
            d,
            tmp,
            tok=tok,
            c1=c1,
            c2=c2,
            n_kept=n_kept,
            u=u,
            mass_kept=mass_kept,
            total=total,
            kth=kth,
            keep=keep,
            kchip=kchip,
            last_val=last_val,
            oh_j=oh_j,
        )

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
        if self.gumbel:  # Gumbel lanes: the full-vocab race's token, exact by construction (never flagged)
            gm = P["gumbel"]
            tok = T(ttnn.where(gm, d["g_tok"], tok, memory_config=mc))
            flag = T(ttnn.where(gm, 0.0, flag, memory_config=mc))
            n_kept = T(ttnn.where(gm, float(self.vocab_size), n_kept, memory_config=mc))
        out = {
            "tok": tok,
            "flag": flag,
            "n_kept": n_kept,
            "u": u,
            "mass_kept": mass_kept,
            "total": total,
            "kth": kth,
            "keep": keep,
            "kchip": kchip,
            "last_val": last_val,
        }
        # raw logprob of the token (T = 1): x_tok - M - log Z1
        if self.logprobs:
            x_s = T(red(ttnn.sum, T(ttnn.multiply(SV, oh_j, memory_config=mc))))
            if self.gumbel:  # the Gumbel winner's raw logit: M + (x - M), exact (bf16 logits)
                x_s = T(ttnn.where(P["gumbel"], T(ttnn.add(M, d["g_dx"], memory_config=mc)), x_s, memory_config=mc))
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
        d.update(self._stage_counters(tmp))
        if self.gumbel:
            d.update(self._stage_gumbel(d, tmp))
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
            if self.gumbel:
                dbg.update({k: d[k] for k in ("g_tok", "g_score", "g_dx")})
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

    def read(
        self, out: SamplerOutput, *, all_chips: bool = False, mesh_sync: bool = False, count: bool = True
    ) -> SampleResult:
        """Blocking read of ``out.info`` from ONE chip (1 KB; ~0.2-0.3 ms: a read of every chip's copy costs ~1-3 ms
        because it ends in a full-mesh finish) -> :class:`SampleResult` (with ``counters`` = the counters written
        last, i.e. the step's). Waits for the step that produced it. ``mesh_sync``: read every chip's copy into
        persistent staging instead (a full-mesh finish: every read enqueued before it has landed too -- the spec step's
        non-blocking ``a`` / ``m`` reads); ``all_chips``: the same, and attach every chip's copy as ``res.per_chip``
        ``[n_chips, 8, 32]`` (tests: identical on all chips). ``count=False``: leave :attr:`stats` alone (warmup)."""
        if all_chips or mesh_sync:
            r = self._reader(out.info)
            ttnn.copy_device_to_host_tensor(out.info, r.host, blocking=True)
            res = SampleResult.from_info(r.views[0])
            if all_chips:
                res.per_chip = torch.stack([v.reshape(len(INFO_ROWS), -1).float().clone() for v in r.views])
        else:
            res = SampleResult.from_info(ttnn.to_torch(ttnn.get_device_tensors(out.info)[0]))
        res.counters = self._last_ctr.copy()
        if count:
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
        logits row with the step's 53-bit uniform :func:`uniform64` (lead decision 5: its top 24 bits are the device
        uniform ``res.u``, so the draw only refines the device's 2^-24 cell) and its raw logprob, under the lane
        parameters set last; a Gumbel lane (``lanes`` given explicitly: they are never flagged) gets its host twin
        :func:`gumbel_sample` with the same counters. The counters are ``res.counters`` (the step's, attached by
        :meth:`read`). ``logits_host``: ``[32, vocab]`` in lane order (``MotifLMHead.logits_to_host``) or a callable
        returning it (called only when a lane needs it). ``active``: lanes to consider (default all; e.g.
        ``positions >= 0``)."""
        todo = [int(l) for l in (lanes if lanes is not None else torch.nonzero(res.flags).reshape(-1).tolist())]
        if active is not None:
            todo = [l for l in todo if bool(active[l])]
        if not todo:
            return res
        lg = logits_host() if callable(logits_host) else logits_host
        if tuple(lg.shape) != (self.lanes, self.vocab_size):
            raise ValueError(f"host logits must be [{self.lanes}, {self.vocab_size}], got {tuple(lg.shape)}")
        ctr = res.counters if res.counters is not None else self._last_ctr
        u64 = uniform64(ctr)
        for l in todo:
            lane = self._lane_params[l]
            if self.gumbel and lane.full_support:
                tok = gumbel_sample(lg[l], lane, ctr[l])
            else:
                tok = fallback_sample(lg[l], lane, float(u64[l]))
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
        ts = [
            self._iota_w,
            self._iota_n,
            self._mask_last,
            self._e0,
            self._e1,
            self._sel_z,
            self._sel_z1,
            self._U,
            self._e0_64,
            self._sel_z64,
            self._m1,
            self._m2,
            self._chip_off,
            *self._p.values(),
        ]
        ts += [t for t in (self._ctr, self._u, self._counter, self._gkey, self._iota_vc, self._m3) if t is not None]
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
    "GOLDEN32_2",
    "GREEDY_LANE",
    "GUMBEL_KEY_SALT",
    "GUMBEL_MIX",
    "INFO_INDEX",
    "INFO_ROWS",
    "LaneParams",
    "MotifDeviceSampler",
    "SAMPLING_EPS",
    "SampleResult",
    "TOP_P_OFF",
    "UndecidedOrder",
    "SamplerOutput",
    "as_counter_pairs",
    "derived_ctr2",
    "device_param_columns",
    "emulate_device",
    "exact_nucleus",
    "exact_sample",
    "fallback_sample",
    "greedy_token",
    "gumbel_sample",
    "gumbel_scores",
    "gumbel_token_keys",
    "gumbel_uniform",
    "gumbel_words",
    "inverse_cdf",
    "k_offset_rows",
    "lane_counter",
    "lane_counters",
    "lane_lists_from_rows",
    "lowbias32",
    "normalize_lane",
    "normalize_lanes",
    "raw_logprob",
    "seed_key",
    "seed_keys",
    "sorted_order",
    "uniform64",
    "uniform_from_counter",
]
