# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 GDLA attention on the BH Galaxy (design §2.3.4, §3.2-3.3; WAVE_A_REVIEW §5.4 ATTN-1..7).

Grouped Differential Latent Attention: 80 q heads = 16 groups x (4 signal + 1 noise), one shared 576-wide latent
KV head ``[n (512, unit RMS) | roped k_pe (64)]`` per token and layer, per-signal-head ``sigmoid(lambda)``
differential, elementwise sigmoid output gate, ``wo``. TP8 over the mesh columns: chip ``tp`` owns q heads
``[10 tp, 10 tp + 10)`` = KV groups ``{2 tp, 2 tp + 1}`` = signal heads ``[8 tp, 8 tp + 8)``; ``wo`` partial sums are
closed with ``all_reduce(tp)``. The latent projection is replicated (no CCL before attention).

Module boundary (README CONVENTIONS §3): input = the *normalized* attention input (``input_layernorm`` output; the
decoder owns that norm), output = the attention output after ``wo`` and ``all_reduce(tp)``:

* decode: ``x [1, 1, L, 4096]`` bf16 TILE DRAM (L = 8 lanes of this chip's DP row, replicated in the row)
  -> ``[1, 1, L, 4096]`` bf16 TILE DRAM, bitwise identical on the 8 TP chips of a row;
* prefill (one user, S = bucket, replicated on all 32 chips): ``x [1, 1, S, 4096]`` -> ``[1, 1, S, 4096]``.

Decode (all 53 layers; absorbed MLA over the paged latent cache; per chip, lanes on tile rows)::

    cq   = x @ Wq_lat [4096,1024] (fp32 out)    kvl = x @ Wkv_lat [4096,640] = [c_raw 512 | kpe 64 | lam 64]
    cq_n = rms_norm(cq) (gamma_q folded into wq_b / wq_b_gate)    n = rms_norm(c_raw) (gamma_kv folded into W_UK/W_UV)
    q    = cq_n @ wq_b'  [1024, 10 x 192]      g = sigmoid(cq_n @ wq_b_gate') [1024, 8 x 128]
    q_nope [1,10,L,128], q_pe [1,10,L,64] = nlp_create_q_heads_split(q)        (heads on dim 1, lanes on rows)
    q_lat = q_nope @ W_UK'_h [1,10,128,512]    q_pe, k_pe = rotary_embedding_hf (prefill mode, cos/sin rows = lanes)
    Q = transpose(concat(q_lat, q_pe), 1, 2) -> [1, L, 10, 576]          (FlashMLA layout: lanes on dim 1)
    paged_update_cache(cache, transpose(concat(n, k_pe)) [1,L,1,576] height-sharded one lane per core, cur_pos, pt)
    O = paged_flash_multi_latent_attention_decode(Q, cache, head_dim_v=512, scale=1.0, window 129 | None,
                                                  G1 program config (k_chunk 128), sdpa_decode role) [1, L, 10, 512]
    U = transpose(O, 1, 2) @ W_UV'_h [1,10,512,128] -> nlp_concat_heads -> [U_sig 1024 | U_noise 2 x 128]
    D = U_sig - sigmoid(lam @ E) * (U_noise @ X);  out = all_reduce_tp(where(active, D * g, 0) @ wo [1024, 4096])

Prefill (expanded GQA, one user)::

    same projections; K/V from the latent: n @ E_pref [512, 2 x 320] -> per group [k_nope | v | 0_64]
    (nlp_create_q_heads_split -> k_nope [1,2,S,128], V padded [1,2,S,192]); K = [k_nope | k_pe x 2 groups]
    O = scaled_dot_product_attention(Q [1,10,S,192] (HF head order), K, V_pad, causal, scale=1.0, window 129 | None,
                                     G2 program config)[..., :128] -> nlp_concat_heads
    same differential / gate / wo / all_reduce; the cache gets typecast(concat(n, k_pe)) via paged_fill_cache with the
    bucket's first S / block_size page-table entries (M10).

Numerics / design decisions:

* **Softmax scale folded into q** (FEASIBILITY_REPORT §4.3 M1, verify_attention d1): the decode / prefill kernels
  truncate the scale of the running-max correction to bf16 (-0.54 % for 192^-0.5, -0.10 % for the global scale).
  ``wq_b`` is multiplied by the layer's scale on the host (fp32, rounded once) and both kernels get ``scale=1.0``
  (exact in bf16). RoPE and the absorb are linear, so the scaled q_pe / q_lat are exactly s * q. ``q_norm``'s gamma is
  folded into the input rows of ``wq_b`` and ``wq_b_gate`` the same way (a weighted ``rms_norm`` costs 18 us traced at
  decode shapes, a weightless one ~6 us).
* **Head order.** Decode keeps the 10 local q heads in the *virtual* order ``[s_0..s_7 | noise_g0, noise_g1]``
  (local HF heads ``0,1,2,3,5,6,7,8,4,9``; all heads share the one latent KV head, so the order is free). After the
  un-absorb + ``nlp_concat_heads`` the 8 signal heads are one contiguous ``[L, 1024]`` channel region in ``wo`` / gate
  order and the noise heads the last 256 columns (one channel split, no slices). Prefill reorders Q to the HF order (5
  consecutive heads per KV group, the GQA mapping of the SDPA op) with one dim-1 concat.
* **Differential after the un-absorb** (HF expanded-form rounding: per-head 128-dim outputs, then
  ``signal - bf16(sigmoid(lambda)) * noise``); lambda and the noise heads are expanded to the 8 x 128 signal
  columns with exact 0/1 matmuls (``E``, ``X``; ``sigmoid(lam @ E) == sigmoid(lam) @ E`` because every column of ``E``
  selects one logit).
* q path: ``x @ Wq_lat`` in fp32 (HF runs wq_a / q_norm / wq_b in fp32), q_norm in fp32, TF32-class operands into
  ``wq_b`` / ``wq_b_gate``; kv path and lambda in bf16 like HF. KV cache = gamma-free unit-RMS latent in
  ``cfg.dtypes.kv_cache`` (bfp8; bf16 via ``MOTIF3_KV_CACHE_DTYPE``); gamma_kv lives in ``W_UK'`` / ``W_UV'`` /
  ``E_pref`` (``weights.fold_kv_norm``).
* Prefill SDPA: the shared ``sdpa_prefill`` role on every layer by default (G2: HiFi4, fp32 acc off -> the streaming
  SDPA kernel; README §4, §10 rule 1). The QK scores / softmax statistics are then bf16: module PCC 0.99965-0.99978
  (random) / 0.99996-0.99997 (real) on global layers and 0.99956-0.99973 on SWA layers, against HF-bf16's 0.99998 (HF
  keeps the scores in fp32). ``sdpa_prefill_fp32_acc="auto"`` is the measured opt-in for window-free calls: fp32 dest
  acc keeps the scores in fp32 (``qk_im_df``; module PCC 0.99996-0.99999) but makes ttnn fall back to the legacy,
  non-streaming compute kernel (``sdpa_program_factory.cpp`` ``can_use_streaming_compute``), which G2 never validated,
  costs +30-55 % per global-layer call at 8K-32K (8K 19.6 vs 15.1 ms, 16K 45.3 vs 32.5, 32K 137.8 vs 88.8: ~0.7 s TTFT
  over the 14 global layers of a 32K prompt) and needs 1.38 MB of static circular buffers (role: 1.17 MB). It is a
  layer-level decision (decoder-layer PCC / teacher-forced top-1), so it stays off here until a shared role exists
  (requested: ``sdpa_prefill_fp32``). SWA prefill cannot use fp32 acc at all (upstream bug, pinned by
  ``test_sdpa_prefill_window_fp32_acc_upstream_bug``): the legacy kernel's ``generate_causal_sliding_window_mask``
  (``sdpa/device/kernels/dataflow/dataflow_common.hpp``) counts a K tile as fully allowed when it lies in the window of
  the Q tile's *first* row (``k_tile_start >= min_window_start``; should be ``max_window_start``), so rows 1..31 of
  every Q tile also attend to up to 31 keys older than their window -- confirmed on device (output == that emulated mask
  at PCC 0.999995; the same kernel with the window as an explicit bf16 ``attn_mask`` is exact). Once fixed upstream,
  fp32 acc would lift SWA prefill from ~0.9996 to ~0.99999. Decode FlashMLA keeps its scores / statistics in bf16
  regardless (``im_df`` is hard-coded in ``sdpa_decode_program_factory.cpp``): that kernel floor (op PCC ~0.9999 at
  Motif shapes) dominates the decode error and is roughly doubled by the differential's cancellation.
* Inactive lanes (``cur_pos = -1``): skipped by ``paged_update_cache`` and FlashMLA, which leaves their output rows
  unwritten; ``forward_decode`` therefore requires the per-step ``active`` mask and replaces those rows of the ``wo``
  input by 0 with ``ttnn.where(active, D * g, 0)`` (design §2.3.4 step 12), so nothing non-finite can reach ``wo`` /
  the all-reduce / the residual streams.
* **L1_SMALL is required** (device params ``l1_small_size``; ``RECOMMENDED_L1_SMALL_SIZE``). The decode
  ``all_reduce(tp)`` of ``[1, 1, 8, 4096]`` takes ttnn's direct reduce-scatter, whose program creates its global
  semaphores while its ~512 KB L1-sharded staging buffer is live; without an L1_SMALL region they land in main L1 at
  ~1,047,424 B and stay there for the life of the cached program (``reduce_scatter_minimal_direct_factory.cpp``
  ``prefer_l1_small_buffer_type``; the all-gather's barrier semaphore likewise). Every later program whose static
  circular buffers end above that address then throws ``TT_THROW ... Statically allocated circular buffers ... clash
  with L1 buffers``: global-layer prefill at S >= 1024 (CB region up to 1.17 MB with the role, 1.38 MB with fp32
  acc), and the bf16-KV FlashMLA decode (1.09 MB). In serving (prefills between decode steps) that is the first
  prefill after any decode step. With ``l1_small_size=32768`` all of these pass and the prefill output is bitwise
  independent of the session history (``test_attention_serving_session``). The constructor warns when the mesh has no
  L1_SMALL region (``require_l1_small=True`` raises); ``test_attention_l1_small_hazard`` pins the failure.

Decode matmul program configs (measured on this Galaxy, traced, ``[1,1,8,K] @ [K,N]``): 1D multicast
(``model_config.mcast1d_matmul_pc``) for ``Wq_lat`` (8x4 cores, 62 -> 28 us), ``Wkv_lat`` (5x4, 59 -> 21 us), ``wq_b``
(12x5, 20 -> 12 us), ``wq_b_gate`` (8x4, 17 -> 8 us); ``wo`` keeps the auto config (25 us, the 1D configs are slower);
the per-head bmms use ``MatmulMultiCoreReuseProgramConfig`` (one head per core; ``W_UV`` 65 -> 7 us, ``W_UK`` 20 ->
8 us). They are built by :func:`decode_matmul_program_configs`, a module-local helper until ``model_config`` has
shared builders for them (requested: ``cfg.attn_decode_matmul_pc(name)`` with exactly these parameters). Prefill uses
ttnn's auto configs (M = S rows).

Fallback paths kept: ``rope_mode="composite"`` (``x cos + (x @ R) sin``, G8 fallback); ``matmul_program_configs``
overrides (``None`` entries = auto config).

Measured on this Galaxy (``tests/unit/test_attention.py``, fabric committed TORUS_Y, mesh opened with
``l1_small_size=32768``; PCC vs the fp32 reference): prefill (default role) global 0.99965-0.99978 random / 0.99996-
0.99997 real (S = 128 ... 32768), SWA 0.99956-0.99973 (fp32-acc opt-in on window-free calls: 0.99996-0.99999);
decode global 0.99975 (random) / 0.99997 (real), SWA 0.99961-0.99976, worst lane 0.99945; FlashMLA alone vs fp64 on
its own Q / cache 0.99987-0.99999 (worst lane 0.99976), the exact mask the best match on every lane, negative controls
(window 128 / 130 / 129-on-global, key p+1 visible) flagged on every distinguishable lane; latent cache vs reference
``c_kv / gamma`` 0.99997 (bf16 KV 0.999997); bf16 KV and block 32 in the serving order (prefill, decode, prefill after
decode, decode) per user >= 0.9995.
Traced decode per call at 4K context: SWA 282 us, global 354 us (FlashMLA 30 / 102 us, AR(tp) 34 us); eager ~2.8 ms.
Eager prefill per call (incl. cache fill): S = 32768 SWA 41.7 ms, global 88.8 ms (fp32-acc opt-in 137.8 ms).
Deviations from the wave-B action list, with reasons: ATTN-4 uses ``rotary_embedding_hf`` in *prefill* mode on the
heads-on-dim-1 / lanes-on-rows q_pe ``[1, 10, L, 64]`` with ``decode_cos_sin(layout="rows")`` (same kernel, row t
rotated by lane t's position; the decode-mode variant would need a transpose + reshard of q_pe and k_pe, 4 extra ops);
ATTN-1 uses ``wq_b_for_chip(layout="interleaved")`` (one ``nlp_create_q_heads_split`` yields q_nope and q_pe) and
per-head copies of the per-group ``W_UK'`` / ``W_UV'`` (one head per core in the bmm); the ``active`` mask is applied to
the ``wo`` input (1024 columns) instead of the output (identical result, half the cost).

Import rule: only ``torch``, ``ttnn`` and the motif3 shared infra (``model_config``, ``weights``, ``ccl``, ``rope``).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

import ttnn

from . import weights as W
from .ccl import MotifCCL
from .model_config import TILE, MotifTTConfig, make_compute_kernel_config, mcast1d_matmul_pc
from .rope import MotifRope

# Cache-name prefix of every tensor this module writes to the TT weight cache. Bump the version whenever a transform
# below changes (head order, scale / gamma folding, layouts), so stale .tensorbin files are never loaded silently.
ATTN_CACHE_VERSION = 2
_CACHE = f"attn.v{ATTN_CACHE_VERSION}"
MASK_WIDTH = 1024  # width of the decode ``active`` mask that matches the ``wo`` input (8 signal heads x 128)
# L1_SMALL region (bytes per core) the mesh must be opened with (module docstring, "L1_SMALL is required"). 32 KB =
# 512 CCL semaphores of 64 B. Measured for attention alone: 16 after the decode all_reduce (direct reduce-scatter 12 +
# all-gather) and the 128 / 1024 prefill buckets, 18 after the 4096 bucket (large prefill reduce-scatters take the ring
# path, whose 4 semaphores stay in main L1: all_reduce has no use_l1_small_for_semaphores). Verified on this Galaxy.
RECOMMENDED_L1_SMALL_SIZE = 32768


def l1_small_bytes(mesh_device) -> Optional[int]:
    """Per-core size of the mesh's L1_SMALL allocator region (0 = the mesh was opened without ``l1_small_size``), or
    ``None`` if it cannot be queried (no real device)."""
    try:
        return int(ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_per_bank)
    except Exception:
        return None


L1_SMALL_WARNING = "Motif-3 attention: the mesh has no L1_SMALL region"


def check_l1_small(mesh_device, *, require: bool = False) -> Optional[int]:
    """Warn (``require=True``: raise) when the mesh has no L1_SMALL region: the CCL semaphores then fragment main L1
    and the first global-layer prefill (or bf16-KV decode) after a decode step throws a static-CB clash (module
    docstring). Returns :func:`l1_small_bytes`. The warning is a ``RuntimeWarning`` issued at the caller of the
    constructor, so Python's default filter shows it once per call site (not once per layer)."""
    size = l1_small_bytes(mesh_device)
    if size == 0:
        msg = (
            f"{L1_SMALL_WARNING}. CCL global semaphores (decode all_reduce / direct reduce-scatter, all-gather) are "
            "then allocated in main L1 around 1.0 MB and stay there, so the first global-layer prefill at S >= 1024 "
            "(or bf16-KV FlashMLA decode) after any decode step throws 'Statically allocated circular buffers ... "
            f"clash with L1 buffers'. Open the mesh with l1_small_size={RECOMMENDED_L1_SMALL_SIZE} (pytest: "
            'device_params(l1_small_size=...); vLLM: --additional-config \'{"tt": {"l1_small_size": ...}}\').'
        )
        if require:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return size


# ======================================================================================================================
# head bookkeeping (pure python / torch; also used by the tests)
# ======================================================================================================================
def virtual_head_order(cfg: MotifTTConfig) -> List[int]:
    """Local q-head order used on device: the chip's signal heads first (signal order ``s_loc = 4 g_loc + j``, i.e.
    local HF heads ``5 g + j``), then one noise head per local group (``5 g + 4``): ``[0,1,2,3,5,6,7,8,4,9]``."""
    G, r, hpg = cfg.kv_groups_per_chip, cfg.grouped_ratio, cfg.heads_per_group
    return [hpg * g + j for g in range(G) for j in range(r)] + [hpg * g + r for g in range(G)]


def virtual_head_groups(cfg: MotifTTConfig) -> List[int]:
    """Local KV group of each virtual head."""
    return [h // cfg.heads_per_group for h in virtual_head_order(cfg)]


def hf_order_from_virtual(cfg: MotifTTConfig) -> List[Tuple[int, int]]:
    """Dim-1 slices ``(start, stop)`` of a virtual-order head tensor whose concatenation is the HF local order
    (``[s(g0) x4, n(g0), s(g1) x4, n(g1)]``)."""
    G, r = cfg.kv_groups_per_chip, cfg.grouped_ratio
    out = []
    for g in range(G):
        out.append((r * g, r * (g + 1)))
        out.append((G * r + g, G * r + g + 1))
    return out


# ======================================================================================================================
# host-side weight transforms (fp32; built on the skeleton transforms of tt/weights.py, ATTN-1)
# ======================================================================================================================
def _attn_name(layer_idx: int, suffix: str) -> str:
    return W.hf_name(layer_idx, f"self_attn.{suffix}")


class _AttnSource:
    """The 9 GDLA tensors of one layer from a weight source (lazy, loaded once)."""

    NAMES = ("wq_a", "q_norm", "wq_b", "wq_b_gate", "wkv_a", "kv_norm", "wkv_b", "lambda_proj", "wo")

    def __init__(self, source, layer_idx: int):
        self.source, self.layer_idx = source, layer_idx
        self._t: Dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._t:
            self._t[key] = self.source.get(_attn_name(self.layer_idx, f"{key}.weight"))
        return self._t[key]


def latent_q_weight(src: _AttnSource, cfg: MotifTTConfig) -> torch.Tensor:
    """``Wq_lat = wq_a^T`` ``[4096, 1024]`` (columns ``[0, 1024)`` of ``weights.latent_projection_for_chip``;
    identical on every chip -> replicated)."""
    return W.latent_projection_for_chip(src["wq_a"], src["wkv_a"], src["lambda_proj"], cfg, 0)[:, : cfg.q_lora_rank]


def latent_kv_weight_for_chip(src: _AttnSource, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """``Wkv_lat`` ``[4096, 640]`` = columns ``[1024, 1664)`` of ``weights.latent_projection_for_chip``: ``[c_raw 512 |
    k_pe 64 | lambda 64]`` with chip ``tp``'s 8 lambda columns first (``[576, 584)``)."""
    full = W.latent_projection_for_chip(src["wq_a"], src["wkv_a"], src["lambda_proj"], cfg, tp)
    return full[:, cfg.q_lora_rank :].contiguous()


def _q_norm_gamma(src: _AttnSource, w: torch.Tensor) -> torch.Tensor:
    g = src["q_norm"]
    g = g.to(w.dtype) if w.dtype == torch.float64 else g.to(torch.float32)
    return g[:, None]


def wq_b_virtual_for_chip(src: _AttnSource, cfg: MotifTTConfig, tp: int, scale: float) -> torch.Tensor:
    """``weights.wq_b_for_chip(layout="interleaved")`` ``[1024, 10 x 192]`` with the head blocks in
    :func:`virtual_head_order`, the softmax scale and the ``q_norm`` gamma (input rows) folded in (exact algebra; one
    rounding at upload)."""
    H, hd = cfg.q_heads_per_chip, cfg.head_dim
    w = W.wq_b_for_chip(src["wq_b"], cfg, tp, layout="interleaved")
    w = w.reshape(w.shape[0], H, hd)[:, virtual_head_order(cfg), :].reshape(w.shape[0], H * hd)
    return (w * _q_norm_gamma(src, w) * float(scale)).contiguous()


def wq_b_gate_for_chip(src: _AttnSource, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """``weights.wq_b_gate_for_chip`` ``[1024, 8 x 128]`` with the ``q_norm`` gamma folded into the input rows."""
    w = W.wq_b_gate_for_chip(src["wq_b_gate"], cfg, tp)
    return (w * _q_norm_gamma(src, w)).contiguous()


def w_uk_virtual_for_chip(src: _AttnSource, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """Absorb matrices per virtual head ``[1, 10, 128, 512]`` (``weights.absorb_weights_for_chip``, gamma_kv folded):
    ``q_lat_h = q_nope_h @ W_UK'_{g(h)}``."""
    w = W.absorb_weights_for_chip(src["wkv_b"], src["kv_norm"], cfg, tp)  # [2, 128, 512]
    return w[virtual_head_groups(cfg)].unsqueeze(0).contiguous()


def w_uv_virtual_for_chip(src: _AttnSource, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """Un-absorb matrices per virtual head ``[1, 10, 512, 128]`` (``weights.unabsorb_weights_for_chip``):
    ``U_h = o_lat_h @ W_UV'_{g(h)}^T``."""
    w = W.unabsorb_weights_for_chip(src["wkv_b"], src["kv_norm"], cfg, tp)  # [2, 512, 128]
    return w[virtual_head_groups(cfg)].unsqueeze(0).contiguous()


def lambda_expansion(cfg: MotifTTConfig) -> torch.Tensor:
    """``E [64, 1024]``: ``lam[:, :64] @ E`` repeats signal head ``s``'s lambda logit over its 128 output columns (rows
    >= 8 are zero, so the other chips' 56 lambda columns of ``Wkv_lat`` drop out). Every column holds exactly one 1, so
    ``sigmoid(lam @ E) == sigmoid(lam) @ E`` exactly (fp32 accumulation of one term)."""
    Sg, v = cfg.signal_heads_per_chip, cfg.v_head_dim
    E = torch.zeros(cfg.n_signal_heads, Sg * v)
    for s in range(Sg):
        E[s, s * v : (s + 1) * v] = 1.0
    return E


def noise_expansion(cfg: MotifTTConfig) -> torch.Tensor:
    """``X [G x 128, 8 x 128]``: ``[noise_g0 | noise_g1] @ X`` puts group ``g``'s noise head under each of its 4 signal
    heads (signal order). Exact (0/1, one term per column)."""
    G, r, v = cfg.kv_groups_per_chip, cfg.grouped_ratio, cfg.v_head_dim
    X = torch.zeros(G * v, G * r * v)
    eye = torch.eye(v)
    for s in range(G * r):
        g = s // r
        X[g * v : (g + 1) * v, s * v : (s + 1) * v] = eye
    return X


def decode_matmul_program_configs(cfg: MotifTTConfig) -> Dict[str, Any]:
    """Decode (M = one tile of lanes) program configs measured on this Galaxy (module docstring). Entries whose grid
    does not fit ``cfg.compute_grid`` fall back to ``None`` (ttnn's auto config).

    Module-local helper (README §10 rule 1 wants ``cfg.*_pc()`` builders; ``model_config`` has none for these yet):
    the four 1D-multicast configs already go through the shared ``mcast1d_matmul_pc``; the two per-head bmm configs
    are ``MatmulMultiCoreReuseProgramConfig`` literals. Requested shared builder: ``cfg.attn_decode_matmul_pc(name)``
    returning exactly these (grid, ``in0_block_w``, subblock, ``per_core_N``) per name."""
    gx, gy = cfg.compute_grid
    K, Kq, H = cfg.hidden_size // TILE, cfg.q_lora_rank // TILE, cfg.q_heads_per_chip

    def mc(grid, n_tiles, ibw, k_tiles):
        if grid[0] > gx or grid[1] > gy:
            return None
        return mcast1d_matmul_pc(grid, n_tiles, ibw, k_tiles)

    def reuse(n_tiles, ibw, sub_w):
        if gx * gy < H:  # one head (batch entry) per core
            return None
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=sub_w,
            per_core_M=1,
            per_core_N=n_tiles,
        )

    q_cols = cfg.q_lora_rank // TILE  # 32
    kv_cols = (cfg.kv_lora_rank + cfg.rope_dim + cfg.n_signal_heads) // TILE  # 20
    qb_cols = H * cfg.head_dim // TILE  # 60
    gate_cols = cfg.signal_heads_per_chip * cfg.v_head_dim // TILE  # 32
    return {
        "q_lat": mc((8, 4), q_cols, 8, K),
        "kv_lat": mc((5, 4), kv_cols, 16, K),
        "wq_b": mc((12, 5), qb_cols, 4, Kq),
        "gate": mc((8, 4), gate_cols, 8, Kq),
        "wo": None,  # auto config: 25 us, faster than every 1D config tried
        "w_uk": reuse(cfg.kv_lora_rank // TILE, 4, 2),
        "w_uv": reuse(cfg.v_head_dim // TILE, 4, 2),
    }


# ======================================================================================================================
# the module
# ======================================================================================================================
RotArg = Union[None, "ttnn.Tensor", Tuple[Any, Any], Dict[str, Tuple[Any, Any]]]


class MotifAttention:
    """GDLA attention of decoder layer ``layer_idx`` (design §2.3.4). See the module docstring for the dataflow.

    Args:
        mesh_device: the opened (4, 8) (or (8, 4)) mesh.
        cfg: ``MotifTTConfig`` built with this mesh.
        layer_idx: decoder layer (window / scale / RoPE kind from ``cfg.layer(layer_idx)``).
        source: ``weights.HFWeightLoader`` or ``weights.DictWeightSource`` (HF names ``model.layers.{l}.self_attn.*``).
        ccl: ``MotifCCL`` of the mesh (``all_reduce(tp)`` after ``wo``).
        rope: ``MotifRope`` (needed for prefill tables and when ``forward_decode`` gets raw rot indices).
        cache: write / read the TT weight cache (``False`` for random weights: nothing is written).
        rope_mode: ``"hf"`` (fused ``rotary_embedding_hf``, G8 decision) or ``"composite"`` (fallback).
        matmul_program_configs: overrides of :func:`decode_matmul_program_configs` (keys ``q_lat``, ``kv_lat``,
            ``wq_b``, ``gate``, ``wo``, ``w_uk``, ``w_uv``; ``None`` = ttnn's auto config). Decode only.
        sdpa_prefill_fp32_acc: ``False`` (default): the shared ``sdpa_prefill`` role (G2: HiFi4, fp32 acc off,
            streaming SDPA kernel) for every prefill call. ``"auto"``: opt-in A/B for the layer-level decision -- fp32
            dest accumulation for every prefill SDPA call that has no window mask in effect (global layers; SWA layers
            at S <= window - 1 = 128, where the 129-key window masks nothing and is dropped), the role otherwise; fp32
            acc keeps the QK scores in fp32 (``qk_im_df``) but selects the legacy non-streaming kernel (module
            docstring: +40-55 % per global call at 16K-32K, 1.38 MB static CBs, not gate-validated). ``True``: always
            fp32 acc (diagnosis only: with ``sliding_window_size`` the legacy kernel returns wrong values for every
            S >= 256 and chunk config, PCC 0.958-0.989; ``test_sdpa_prefill_window_fp32_acc_upstream_bug``). The
            fp32 variant uses a module-local compute config until ``model_config`` has a shared role for it.
        require_l1_small: raise instead of warning when the mesh has no L1_SMALL region (module docstring,
            :func:`check_l1_small`).
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        layer_idx: int,
        *,
        source,
        ccl: MotifCCL,
        rope: Optional[MotifRope] = None,
        cache: bool = True,
        rope_mode: str = "hf",
        matmul_program_configs: Optional[Dict[str, Any]] = None,
        sdpa_prefill_fp32_acc: Union[str, bool] = False,
        require_l1_small: bool = False,
    ):
        if rope_mode not in ("hf", "composite"):
            raise ValueError(f"rope_mode must be 'hf' or 'composite', got {rope_mode!r}")
        if sdpa_prefill_fp32_acc not in ("auto", True, False):
            raise ValueError(f"sdpa_prefill_fp32_acc must be 'auto', True or False, got {sdpa_prefill_fp32_acc!r}")
        self.l1_small_bytes = check_l1_small(mesh_device, require=require_l1_small)
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.spec = cfg.layer(self.layer_idx)
        self.ccl = ccl
        self.rope = rope
        self.rope_mode = rope_mode
        self.sdpa_prefill_fp32_acc = sdpa_prefill_fp32_acc
        self.decode_pcs = decode_matmul_program_configs(cfg)
        self.decode_pcs.update(matmul_program_configs or {})

        # ---- per-chip geometry ------------------------------------------------------------------------------------
        self.H = cfg.q_heads_per_chip  # 10
        self.G = cfg.kv_groups_per_chip  # 2
        self.r = cfg.grouped_ratio  # 4
        self.Sg = cfg.signal_heads_per_chip  # 8
        self.nope, self.rope_dim, self.vdim = cfg.qk_nope_head_dim, cfg.rope_dim, cfg.v_head_dim  # 128, 64, 128
        self.rank = cfg.kv_lora_rank  # 512
        self.latent_dim = cfg.kv_latent_dim  # 576
        self.window = self.spec.sliding_window_size  # 129 | None
        self.scale = float(self.spec.softmax_scale)
        self.kind = self.spec.rope_kind  # "yarn" | "plain"
        self.lanes = cfg.lanes_per_row

        # ---- compute / program configs (README §4-5) --------------------------------------------------------------
        self.ckc_latent = cfg.compute_config("attn_latent")
        self.ckc_heads = cfg.compute_config("attn_heads")
        self.ckc_norm = cfg.compute_config("norm")
        self.ckc_sdpa_decode = cfg.compute_config("sdpa_decode")
        self.ckc_sdpa_prefill = cfg.compute_config("sdpa_prefill")  # G2-validated: HiFi4, fp32 acc off (default)
        role = cfg.compute_role("sdpa_prefill")
        # opt-in only (sdpa_prefill_fp32_acc="auto"/True): module-local until model_config has the requested
        # "sdpa_prefill_fp32" role (the sdpa_prefill role with fp32 dest acc on)
        self.ckc_sdpa_prefill_fp32 = make_compute_kernel_config(
            role.fidelity, True, approx=role.approx, packer_l1_acc=role.packer_l1_acc
        )
        self.ckc_rope = rope.ckc if rope is not None else cfg.compute_config("rope")
        self.decode_pc = cfg.flash_mla_decode_pc()  # G1: k_chunk 128, mandatory (ATTN-2)
        self.dtype = cfg.dtypes.activations

        # ---- weights (ATTN-1) -----------------------------------------------------------------------------------
        src = _AttnSource(source, self.layer_idx)
        L = self.layer_idx
        dt = cfg.dtypes.attention

        def up(fn, name, **kw):
            return W.as_tensor(
                fn,
                mesh_device=mesh_device,
                cfg=cfg,
                dtype=kw.pop("dtype", dt),
                layout=kw.pop("layout", ttnn.TILE_LAYOUT),
                cache_name=f"{_CACHE}.{name}" if cache else None,
                layer=L,
                **kw,
            )

        def stack(fn, dim):
            return lambda: W.stack_tp(fn, cfg, dim=dim)

        s = self.scale
        self.w_q_lat = up(lambda: latent_q_weight(src, cfg), "wq_lat")  # [4096, 1024] replicated
        self.w_kv_lat = up(stack(lambda tp: latent_kv_weight_for_chip(src, cfg, tp), 1), "wkv_lat", tp_dim=1)
        self.w_q_b = up(stack(lambda tp: wq_b_virtual_for_chip(src, cfg, tp, s), 1), "wq_b", tp_dim=1)
        self.w_gate = up(stack(lambda tp: wq_b_gate_for_chip(src, cfg, tp), 1), "wq_b_gate", tp_dim=1)
        self.w_uk = up(stack(lambda tp: w_uk_virtual_for_chip(src, cfg, tp), 1), "w_uk", tp_dim=1)
        self.w_uv = up(stack(lambda tp: w_uv_virtual_for_chip(src, cfg, tp), 1), "w_uv", tp_dim=1)
        self.w_kv_expand = up(
            stack(lambda tp: W.prefill_kv_expansion_for_chip(src["wkv_b"], src["kv_norm"], cfg, tp), 1),
            "kv_expand",
            tp_dim=1,
        )
        self.w_o = up(stack(lambda tp: W.wo_for_chip(src["wo"], cfg, tp), 0), "wo", tp_dim=0)
        self.lam_expand = up(lambda: lambda_expansion(cfg), "lam_expand")  # constants, replicated
        self.noise_expand = up(lambda: noise_expansion(cfg), "noise_expand")

        # decode KV-update input: [1, L, 1(32), 576] HEIGHT_SHARDED, one lane per core (G7 / ATTN-3)
        grid = mesh_device.compute_with_storage_grid_size()
        self._update_cores = ttnn.num_cores_to_corerangeset(self.lanes, grid, row_wise=True)
        self.update_mc = ttnn.create_sharded_memory_config(
            shape=(TILE, self.latent_dim),
            core_grid=self._update_cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # ==================================================================================================================
    # per-step helpers (call ONCE per decode step, shared by all layers)
    # ==================================================================================================================
    @staticmethod
    def decode_rope_tables(rope: MotifRope, rot_idxs, kinds: Sequence[str] = ("yarn", "plain")) -> Dict[str, Tuple]:
        """``{kind: (cos, sin)}`` ``[1, 1, 32, 64]`` TILE (row t = lane t of this DP row; ``rope.decode_cos_sin(...,
        layout="rows")``) from the per-step ``rot_idxs [1, 32]`` uint32 device tensor. Trace-safe (2 embedding gathers
        per kind). Pass the dict as ``rot=`` to every layer's :meth:`forward_decode`."""
        return {k: rope.decode_cos_sin(k, rot_idxs, layout="rows") for k in kinds}

    @staticmethod
    def active_mask_from_cur_pos(cur_pos, lanes: int = 8, width: int = MASK_WIDTH):
        """Trace-safe ``[1, 1, L, width]`` bf16 TILE 0/1 mask (1 = lane active, ``cur_pos >= 0``) from the ``[L]`` int32
        ROW_MAJOR ``cur_pos`` device tensor. Build it once per step and pass it as ``active=`` to every layer
        (``width`` 1024 = the ``wo`` input width: a full-width predicate is 2x cheaper than a broadcast one; ``width=1``
        gives the broadcastable ``[1, 1, L, 1]``)."""
        p = ttnn.reshape(cur_pos, (1, 1, lanes, 1))
        p = ttnn.typecast(p, ttnn.float32)
        t = ttnn.to_layout(p, ttnn.TILE_LAYOUT)
        ttnn.deallocate(p)
        m = ttnn.ge(t, 0.0)
        ttnn.deallocate(t)
        if m.dtype != ttnn.bfloat16:
            m2 = ttnn.typecast(m, ttnn.bfloat16)
            ttnn.deallocate(m)
            m = m2
        if width != 1:
            m2 = ttnn.repeat(m, ttnn.Shape([1, 1, 1, int(width)]))
            ttnn.deallocate(m)
            m = m2
        return m

    @staticmethod
    def active_mask_host(
        positions: torch.Tensor, cfg: MotifTTConfig, mesh_device, *, device=None, width: int = MASK_WIDTH
    ):
        """Host (``device=None``, for ``ttnn.copy_host_to_device_tensor``) or device mesh tensor ``[1, 1, L, width]``
        bf16 per DP row from the 32 lane positions (``-1`` = inactive), lane order."""
        pos = torch.as_tensor(positions).reshape(cfg.max_batch)
        rows = (pos >= 0).to(torch.float32).reshape(cfg.dp, 1, cfg.lanes_per_row, 1).expand(-1, -1, -1, int(width))
        return _shard_rows(rows.contiguous(), cfg, mesh_device, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    # ==================================================================================================================
    # shared building blocks
    # ==================================================================================================================
    def _linear(self, x, w, *, ckc, pc=None, dtype=None, activation=None):
        kw = {}
        if pc is not None:
            kw["program_config"] = pc
        if activation is not None:
            kw["activation"] = activation
        return ttnn.linear(
            x, w, dtype=dtype or self.dtype, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw
        )

    def _pc(self, name: str, decode: bool):
        return self.decode_pcs.get(name) if decode else None

    def _project(self, x, decode: bool):
        """Shared projections. ``x [1,1,T,4096]`` -> ``q`` ``[1,1,T,1920]`` (virtual head order, scaled), ``g`` =
        sigmoid(gate) ``[1,1,T,1024]``, ``n`` ``[1,1,T,512]``, ``kpe`` ``[1,1,T,64]`` (raw), ``lam`` ``[1,1,T,64]``
        (lambda logits, the chip's 8 first)."""
        cq = self._linear(x, self.w_q_lat, ckc=self.ckc_latent, pc=self._pc("q_lat", decode), dtype=ttnn.float32)
        kvl = self._linear(x, self.w_kv_lat, ckc=self.ckc_latent, pc=self._pc("kv_lat", decode))
        cq_n = ttnn.rms_norm(cq, epsilon=self.cfg.rms_norm_eps, compute_kernel_config=self.ckc_norm)
        ttnn.deallocate(cq)
        q = self._linear(cq_n, self.w_q_b, ckc=self.ckc_heads, pc=self._pc("wq_b", decode))
        g = self._linear(cq_n, self.w_gate, ckc=self.ckc_heads, pc=self._pc("gate", decode), activation="sigmoid")
        ttnn.deallocate(cq_n)
        # channel splits (one op each; tile-aligned regions): [c_raw | kpe lam] -> [kpe | lam]
        c_raw, rest = ttnn.experimental.nlp_create_q_heads_split(kvl, num_heads=1, split_head_dim=self.rank)
        ttnn.deallocate(kvl)
        kpe, lam = ttnn.experimental.nlp_create_q_heads_split(rest, num_heads=1, split_head_dim=self.rope_dim)
        ttnn.deallocate(rest)
        n = ttnn.rms_norm(c_raw, epsilon=self.cfg.rms_norm_eps, compute_kernel_config=self.ckc_norm)
        ttnn.deallocate(c_raw)
        return q, g, n, kpe, lam

    def _rope(self, x, cos, sin):
        if self.rope_mode == "hf":
            return ttnn.experimental.rotary_embedding_hf(
                x, cos, sin, is_decode_mode=False, compute_kernel_config=self.ckc_rope
            )
        if self.rope is None:
            raise ValueError("rope_mode='composite' needs a MotifRope")
        T = int(x.shape[-2])
        if int(cos.shape[-2]) != T:  # decode "rows" tables carry 32 rows for L = 8 lanes: broadcast needs equal rows
            cos = ttnn.slice(cos, [0, 0, 0, 0], [1, 1, T, int(cos.shape[-1])])
            sin = ttnn.slice(sin, [0, 0, 0, 0], [1, 1, T, int(sin.shape[-1])])
            out = self.rope.apply_composite(x, cos, sin)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            return out
        return self.rope.apply_composite(x, cos, sin)

    def _combine(self, u_sig, u_noise, g, lam, active=None):
        """``(u_sig - sigmoid(lam @ E) * u_noise) * g`` ``[1,1,T,1024]`` (signal order = ``wo`` input order); inactive
        rows -> 0 when ``active`` is given (always in decode; prefill has no inactive rows). Consumes ``u_sig``,
        ``u_noise``, ``g``, ``lam``."""
        v_exp = self._linear(lam, self.lam_expand, ckc=self.ckc_heads, activation="sigmoid")
        ttnn.deallocate(lam)
        d = ttnn.addcmul(u_sig, v_exp, u_noise, value=-1.0)
        for t in (u_sig, u_noise, v_exp):
            ttnn.deallocate(t)
        dg = ttnn.multiply(d, g)
        ttnn.deallocate(d)
        ttnn.deallocate(g)
        if active is not None:
            masked = ttnn.where(active, dg, 0.0)
            ttnn.deallocate(dg)
            dg = masked
        return dg

    def _rot_tables(self, rot: RotArg):
        if isinstance(rot, dict):
            return rot[self.kind]
        if isinstance(rot, (tuple, list)):
            return rot[0], rot[1]
        if rot is None:
            raise ValueError("forward_decode needs rot= (cos, sin) | {kind: (cos, sin)} | rot_idxs tensor")
        if self.rope is None:
            raise ValueError("raw rot indices need a MotifRope (pass rope= at construction)")
        return self.rope.decode_cos_sin(self.kind, rot, layout="rows")

    # ==================================================================================================================
    # decode
    # ==================================================================================================================
    def forward_decode(
        self,
        x,
        *,
        rot: RotArg,
        cur_pos,
        page_table,
        kv_cache,
        active,
        taps: Optional[Dict[str, Any]] = None,
    ):
        """One decode step for the L = 8 lanes of this chip's DP row (trace-safe; design §2.3.4 decode steps 1-12).

        Args:
            x: normalized input ``[1, 1, L, 4096]`` bf16 TILE DRAM (replicated in the row).
            rot: this step's RoPE tables: ``{kind: (cos, sin)}`` from :meth:`decode_rope_tables` (preferred, built once
                per step), a ``(cos, sin)`` pair ``[1,1,32,64]`` for this layer's kind, or the ``rot_idxs [1, 32]``
                uint32 tensor (gathered here).
            cur_pos: ``[L]`` int32 ROW_MAJOR (per row; ``-1`` = inactive lane: skipped by the cache update and
                FlashMLA).
            page_table: ``[L, W]`` int32 ROW_MAJOR (per row).
            kv_cache: this layer's paged latent cache ``[N, 1, block, 576]`` (``cfg.dtypes.kv_cache``, TILE, DRAM);
                updated in place at ``cur_pos``.
            active: **required** ``[1, 1, L, 1024]`` (or ``[1, 1, L, 1]``) bf16 0/1 mask
                (:meth:`active_mask_from_cur_pos` or :meth:`active_mask_host`, built once per step for all layers);
                inactive rows of the output are exactly 0. FlashMLA leaves the output rows of skipped lanes unwritten,
                so there is no unmasked variant.
            taps: optional dict that receives intermediate tensors (debug / tests; never inside a trace).

        Returns ``[1, 1, L, 4096]`` bf16 TILE DRAM, identical on the TP chips of the row.
        """
        if active is None:
            raise ValueError(
                "forward_decode needs the per-step active mask (MotifAttention.active_mask_from_cur_pos(cur_pos) or "
                "active_mask_host): FlashMLA leaves the rows of skipped lanes (cur_pos = -1) unwritten"
            )
        cos, sin = self._rot_tables(rot)
        q, g, n, kpe, lam = self._project(x, decode=True)

        # ---- Q [1, L, 10, 576] = [q_nope @ W_UK' | rope(q_pe)] (lanes on dim 1, heads on rows) -------------------
        q_nope, q_pe = ttnn.experimental.nlp_create_q_heads_split(q, num_heads=self.H, split_head_dim=self.nope)
        ttnn.deallocate(q)
        q_lat = ttnn.matmul(
            q_nope,
            self.w_uk,
            program_config=self._pc("w_uk", True),
            compute_kernel_config=self.ckc_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_nope)
        q_pe_r = self._rope(q_pe, cos, sin)
        ttnn.deallocate(q_pe)
        q_heads = ttnn.concat([q_lat, q_pe_r], dim=-1)  # [1, 10, L, 576]
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_pe_r)
        q_mla = ttnn.transpose(q_heads, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, L, 10, 576]
        ttnn.deallocate(q_heads)

        # ---- KV write: [n | rope(kpe)] -> [1, L, 1, 576] height-sharded one lane per core (G7, ATTN-3) -------------
        k_pe = self._rope(kpe, cos, sin)
        ttnn.deallocate(kpe)
        kv_row = ttnn.concat([n, k_pe], dim=-1)  # [1, 1, L, 576]
        ttnn.deallocate(n)
        ttnn.deallocate(k_pe)
        kv_upd = ttnn.transpose(kv_row, 1, 2, memory_config=self.update_mc)  # [1, L, 1, 576]
        if taps is not None:
            taps["kv_row"] = kv_row
        else:
            ttnn.deallocate(kv_row)
        ttnn.experimental.paged_update_cache(kv_cache, kv_upd, update_idxs_tensor=cur_pos, page_table=page_table)
        ttnn.deallocate(kv_upd)

        # ---- FlashMLA decode (G1 config, scale folded into q, ATTN-2) ----------------------------------------------
        o_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_mla,
            kv_cache,
            None,
            head_dim_v=self.rank,
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos,
            scale=1.0,
            sliding_window_size=self.window,
            program_config=self.decode_pc,
            compute_kernel_config=self.ckc_sdpa_decode,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, L, 10, 512]
        if taps is not None:
            taps["q_mla"] = q_mla
            taps["o_lat"] = o_lat
        else:
            ttnn.deallocate(q_mla)

        # ---- un-absorb per head, differential, gate, mask, wo, AR(tp) ------------------------------------------
        o_heads = ttnn.transpose(o_lat, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 10, L, 512]
        if taps is None:
            ttnn.deallocate(o_lat)
        u = ttnn.matmul(
            o_heads,
            self.w_uv,
            program_config=self._pc("w_uv", True),
            compute_kernel_config=self.ckc_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 10, L, 128]
        ttnn.deallocate(o_heads)
        u_flat = ttnn.experimental.nlp_concat_heads(u, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, L, 1280]
        ttnn.deallocate(u)
        u_sig, noise = ttnn.experimental.nlp_create_q_heads_split(
            u_flat, num_heads=1, split_head_dim=self.Sg * self.vdim
        )  # [1,1,L,1024], [1,1,L,256]
        if taps is not None:
            taps["u_flat"] = u_flat
        else:
            ttnn.deallocate(u_flat)
        u_noise = self._linear(noise, self.noise_expand, ckc=self.ckc_heads)  # [1, 1, L, 1024]
        ttnn.deallocate(noise)
        dg = self._combine(u_sig, u_noise, g, lam, active)
        part = self._linear(dg, self.w_o, ckc=self.ckc_heads, pc=self._pc("wo", True))
        ttnn.deallocate(dg)
        out = self.ccl.ar_tp(part)
        ttnn.deallocate(part)
        return out

    # ==================================================================================================================
    # prefill
    # ==================================================================================================================
    def prefill_sdpa_window_and_config(self, S: int):
        """``(sliding_window_size, compute_kernel_config)`` of the prefill SDPA for bucket ``S``: the window is dropped
        when it cannot mask anything (``S < window``, i.e. the 128 bucket on SWA layers: keys ``[p - 128, p]`` cover
        ``[0, p]``). The config is the ``sdpa_prefill`` role by default; with ``sdpa_prefill_fp32_acc="auto"`` fp32
        accumulation is used exactly when no window is in effect (``True``: always)."""
        window = self.window if (self.window is not None and S >= self.window) else None
        mode = self.sdpa_prefill_fp32_acc
        fp32 = mode is True or (mode == "auto" and window is None)
        return window, (self.ckc_sdpa_prefill_fp32 if fp32 else self.ckc_sdpa_prefill)

    def forward_prefill(
        self,
        x,
        *,
        page_table=None,
        kv_cache=None,
        rot: RotArg = None,
        taps: Optional[Dict[str, Any]] = None,
    ):
        """Prefill of one user (eager; design §2.3.4 prefill, §3.3). ``x [1, 1, S, 4096]`` bf16 (S = bucket, positions
        ``0..S-1``, replicated on all chips) -> ``[1, 1, S, 4096]`` bf16 (identical on all chips).

        Args:
            page_table: ``[1, n]`` int32 ROW_MAJOR, the user's blocks; the fill uses exactly the first
                ``cfg.prefill_page_table_entries(S)`` entries (a wider table is sliced here; pass the exact width to
                keep one program shape per bucket, M10). ``None`` (with ``kv_cache=None``) skips the cache fill.
            kv_cache: this layer's paged cache; gets ``[n | rope(k_pe)]`` for positions ``0..S-1`` (ATTN-6).
            rot: optional ``(cos, sin)`` ``[1, 1, >=S, 64]``; default ``rope.prefill_cos_sin(kind, S)``.
        """
        S = int(x.shape[-2])
        if rot is None:
            if self.rope is None:
                raise ValueError("forward_prefill needs a MotifRope or rot=(cos, sin)")
            cos, sin = self.rope.prefill_cos_sin(self.kind, S)
        else:
            cos, sin = self._rot_tables(rot)
        q, g, n, kpe, lam = self._project(x, decode=False)

        # ---- Q [1, 10, S, 192] in HF head order (GQA: 5 consecutive q heads per KV group) -----------------------
        q_nope, q_pe = ttnn.experimental.nlp_create_q_heads_split(q, num_heads=self.H, split_head_dim=self.nope)
        ttnn.deallocate(q)
        q_pe_r = self._rope(q_pe, cos, sin)
        ttnn.deallocate(q_pe)
        q_virt = ttnn.concat([q_nope, q_pe_r], dim=-1)  # [1, 10, S, 192], virtual order
        ttnn.deallocate(q_nope)
        ttnn.deallocate(q_pe_r)
        hd = self.cfg.head_dim
        parts = [ttnn.slice(q_virt, [0, a, 0, 0], [1, b, S, hd]) for a, b in hf_order_from_virtual(self.cfg)]
        ttnn.deallocate(q_virt)
        q_full = ttnn.concat(parts, dim=1)
        for t in parts:
            ttnn.deallocate(t)

        # ---- K / V_pad from the latent (expanded form, V zero-padded 128 -> 192, G2) --------------------------------
        k_pe = self._rope(kpe, cos, sin)  # [1, 1, S, 64]
        ttnn.deallocate(kpe)
        kvx = self._linear(n, self.w_kv_expand, ckc=self.ckc_heads)  # [1, 1, S, 2 x 320]
        k_nope, v_pad = ttnn.experimental.nlp_create_q_heads_split(kvx, num_heads=self.G, split_head_dim=self.nope)
        ttnn.deallocate(kvx)
        k_pe_g = ttnn.repeat(k_pe, ttnn.Shape([1, self.G, 1, 1]))
        k_full = ttnn.concat([k_nope, k_pe_g], dim=-1)  # [1, 2, S, 192]
        ttnn.deallocate(k_nope)
        ttnn.deallocate(k_pe_g)

        # ---- SDPA (G2 program config, scale folded into q, ATTN-5) -------------------------------------------------
        window, ckc = self.prefill_sdpa_window_and_config(S)
        o = ttnn.transformer.scaled_dot_product_attention(
            q_full,
            k_full,
            v_pad,
            is_causal=True,
            scale=1.0,
            sliding_window_size=window,
            program_config=self.cfg.sdpa_prefill_pc(self.spec, seq_len=S),
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 10, S, 192] (columns 128..191 = 0)
        ttnn.deallocate(q_full)
        ttnn.deallocate(k_full)
        ttnn.deallocate(v_pad)
        o_v = ttnn.slice(o, [0, 0, 0, 0], [1, self.H, S, self.vdim])
        ttnn.deallocate(o)
        u_flat = ttnn.experimental.nlp_concat_heads(o_v, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,S,1280] HF order
        ttnn.deallocate(o_v)
        # HF order: [s(g) x4 | n(g)] per group -> signal columns (2 slices + concat) and noise x4 (concat of 8)
        v, r, hpg = self.vdim, self.r, self.cfg.heads_per_group
        sig = [ttnn.slice(u_flat, [0, 0, 0, hpg * gi * v], [1, 1, S, (hpg * gi + r) * v]) for gi in range(self.G)]
        noise = [
            ttnn.slice(u_flat, [0, 0, 0, (hpg * gi + r) * v], [1, 1, S, (hpg * gi + r + 1) * v]) for gi in range(self.G)
        ]
        if taps is not None:
            taps["u_flat"] = u_flat
        else:
            ttnn.deallocate(u_flat)
        u_sig = ttnn.concat(sig, dim=-1)
        u_noise = ttnn.concat([nz for nz in noise for _ in range(r)], dim=-1)
        for t in sig + noise:
            ttnn.deallocate(t)
        dg = self._combine(u_sig, u_noise, g, lam)
        part = self._linear(dg, self.w_o, ckc=self.ckc_heads)
        ttnn.deallocate(dg)
        out = self.ccl.ar_tp(part)
        ttnn.deallocate(part)

        # ---- cache fill (ATTN-6): typecast to the cache dtype (raw tile copy), bucket's first S/bs entries -----
        if kv_cache is not None:
            if page_table is None:
                raise ValueError("forward_prefill: kv_cache given without page_table")
            kv_row = ttnn.concat([n, k_pe], dim=-1)  # [1, 1, S, 576]
            if kv_row.dtype != kv_cache.dtype:
                kv_cast = ttnn.typecast(kv_row, kv_cache.dtype)
                ttnn.deallocate(kv_row)
                kv_row = kv_cast
            n_pt = self.cfg.prefill_page_table_entries(S)
            if int(page_table.shape[-1]) < n_pt:
                raise ValueError(
                    f"prefill page table has {int(page_table.shape[-1])} entries, bucket {S} needs {n_pt} "
                    f"(pad with the null block 0 beyond the user's blocks)"
                )
            pt = page_table
            if int(page_table.shape[-1]) != n_pt:
                pt = ttnn.slice(page_table, [0, 0], [1, n_pt])
            ttnn.experimental.paged_fill_cache(kv_cache, kv_row, pt, batch_idx=0)
            if pt is not page_table:
                ttnn.deallocate(pt)
            ttnn.deallocate(kv_row)
        ttnn.deallocate(n)
        ttnn.deallocate(k_pe)
        return out


# ======================================================================================================================
# small helpers
# ======================================================================================================================
def _shard_rows(rows: torch.Tensor, cfg: MotifTTConfig, mesh_device, dtype, layout, device=None):
    """``[dp, ...]`` host tensor -> mesh tensor with DP row ``r`` holding ``rows[r]`` (``[1, ...]``), replicated over TP
    (same placement as ``rope.shard_lanes``, any layout / dtype)."""
    dims = cfg.axes.mesh_dims(dp_dim=0, tp_dim=None)
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate() if d is None else ttnn.PlacementShard(d) for d in dims],
            ttnn.MeshShape(*cfg.axes.mesh_shape),
        ),
    )
    return ttnn.from_torch(
        rows,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if device is not None else None,
        mesh_mapper=mapper,
    )


__all__ = [
    "ATTN_CACHE_VERSION",
    "L1_SMALL_WARNING",
    "MASK_WIDTH",
    "MotifAttention",
    "RECOMMENDED_L1_SMALL_SIZE",
    "check_l1_small",
    "decode_matmul_program_configs",
    "hf_order_from_virtual",
    "l1_small_bytes",
    "lambda_expansion",
    "latent_kv_weight_for_chip",
    "latent_q_weight",
    "noise_expansion",
    "virtual_head_groups",
    "virtual_head_order",
    "w_uk_virtual_for_chip",
    "w_uv_virtual_for_chip",
    "wq_b_gate_for_chip",
    "wq_b_virtual_for_chip",
]
