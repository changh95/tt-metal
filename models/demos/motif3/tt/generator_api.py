# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The contract between the vLLM bridge and the Motif-3 TT runtime.

Layering::

    vLLM 0.26 + vllm-tt-plugin         rows, state slots, block tables, plugin kwargs, host sampling
      └─ tt/generator_vllm.py          MotifForCausalLM: translates rows/slots -> lanes, validates, gathers logits
           └─ MotifGenerator (here)    lanes, host torch tensors in / host logits out, one opaque KV handle
                └─ tt/generator.py     integration wave: mesh, weights, KV tensors, decode trace, warmup

The bridge owns everything that is vLLM-specific. A generator never sees vLLM rows, state slots,
``slot_remap``, sampling parameters or plugin keyword arguments; it only sees *lanes*.

Lanes (design 00 §2.3.10, §3.1)
-------------------------------
There are ``NUM_LANES = 32`` decode lanes. Lane ``l`` belongs to DP group ``l // LANES_PER_GROUP`` (8 lanes per
group): the size-4 mesh axis indexes the group, and the group's 8 chips (the size-8 TP axis) run its 8 lanes.
A decode step writes a lane's new latent only on its own group's chips (unless KV-R is on, below), so a request must
keep its lane for its whole life. The bridge guarantees that (``LaneMap`` in ``generator_vllm.py`` follows vLLM's
``slot_remap``). Prefill writes the latent on every chip, so a request that is re-prefilled (preemption) may get any
lane, and a chunked prompt may change lane between chunks.

KV cache (design 00 §1.5, §2.3.4; study 05 §7.3)
------------------------------------------------
One paged latent cache per decoder layer, logical shape ``[num_blocks, 1, block_size, KV_LATENT_DIM]`` with
``KV_LATENT_DIM = 576 = kv_lora_rank 512 (unit-RMS latent, kv_norm gamma folded into W_UK/W_UV) ‖ RoPE'd k_pe 64``.
It is replicated on all 32 chips (one block-id space, DP=1 in vLLM). vLLM owns the block ids: block 0 is vLLM's
null block (padding in every page table, never holds a valid position); real blocks are ``1 .. num_blocks-1``.
Device dtype is ``settings.kv_cache_dtype`` (``"bfp8"`` -> ``ttnn.bfloat8_b``, ``"bf16"`` -> ``ttnn.bfloat16``),
TILE layout, DRAM. The torch dtype vLLM hands the bridge is bookkeeping only.

``num_blocks`` is whatever ``allocate_kv_cache`` receives, never a config constant: the plugin allocates
``plugin_num_blocks(pool + NULL_BLOCK_RESERVE_TOKENS, block_size, max_num_seqs)`` blocks, i.e. 4129 for the default
pool of 262,144 tokens, block 64 and ``--max-num-seqs 32`` (4105 with ``--max-num-seqs 8``).
:func:`expected_num_blocks` reproduces that number for planning and tests only.

Positions
---------
A position is the absolute token index in the request (0-based). For decode, ``positions[l]`` is the index of
``tokens[l]`` = the KV slot that token's latent is written to; the returned logits predict index
``positions[l] + 1``. Its latent goes to block ``page_table[l, p // block_size]``, row ``p % block_size``.
``-1`` marks an inactive lane: no KV write, no reads, its logits row is ignored.

Host tensors only
-----------------
Every tensor crossing this interface is a CPU torch tensor. Inputs: ``torch.int32``. Outputs: logits as
``torch.float32`` or ``torch.bfloat16`` (vLLM's sampler upcasts), never padded past ``vocab_size``; argmax ids as
``torch.int32``.

Features: chunked prefill, prefix caching, MTP speculative decoding (docs/features/FEATURES_DESIGN.md §2)
---------------------------------------------------------------------------------------------------------
All three are off unless the bridge turns them on in :class:`GeneratorSettings` (from vLLM's scheduler config); a
generator that does not override the new members keeps the draft-1 behaviour.

* **Resumed prefill rows.** :class:`PrefillRequest` carries ``start`` (vLLM ``num_computed_tokens``): positions
  ``[0, start)`` are already in the cache. Full blocks below ``floor(start / block_size)`` are READ-ONLY (they may be
  shared through vLLM's prefix cache); the generator writes ``[floor(start / bs) * bs, end)`` plus bucket padding
  inside the request's own last block (or the never-read null block 0). The bridge makes **one**
  :meth:`MotifGenerator.prefill_forward_batch` call per plugin prefill step; the generator plans the rows
  (``tt/prefill_plan.py``: resume alignment ``A``, internal chunks of at most ``max_prefill_span`` rows, fill tables
  with ``-1`` for shared blocks) and runs them writer-first (a row may hit blocks another row of the same call
  writes).
* **KV-R** (``GeneratorSettings.kv_replicated``, on whenever prefix caching is): every decode KV write (53 layers +
  the MTP layer) lands on all 32 chips, so a prefix hit on a block another DP row decode-wrote reads valid KV.
* **MTP speculative decoding** (``spec_tokens = 1``): the MTP layer (``model.mtp_layers.0``, reference layer index
  53) keeps its own latent cache, indexed by the same vLLM block ids (the generator allocates it next to the 53
  layers; vLLM keeps accounting 53). In a speculating launch the bridge calls
  :meth:`MotifGenerator.decode_forward_spec` for every decode step: ordinary steps (no drafts, host logits) and
  verify steps (one draft per lane at most, argmax ids only).

Features: packed prefill (P5) and full-batch speculative verify (T64) (docs/p5_t64/P5_T64_DESIGN.md)
----------------------------------------------------------------------------------------------------
Both leave the per-call contract unchanged; :class:`GeneratorSettings` carries their knobs.

* **Packed prefill** (``packed_prefill``, ``MOTIF3_PACKED_PREFILL``; §3): the generator may run the short chunks of one
  :meth:`MotifGenerator.prefill_forward_batch` call together, as B segments of S rows in one pass of ``T = B * S``
  rows (``pk0``: chunks at start 0; ``pk1``: resumed chunks at one common start). The rows' logits, KV writes and
  read-only blocks are exactly those of the per-row contract. Knobs: ``packed_prefill_max_seg`` /
  ``packed_prefill_max_tokens`` / ``packed_prefill_pk1`` / ``packed_warmup``.
* **Speculative verify mode** (``spec_verify``, ``MOTIF3_SPEC_VERIFY``; §2.2, §4): ``"packed"`` runs drafts on idle
  lanes of the 32-lane trace (T32), ``"wide"`` runs every step in a 64-row trace (T64: per DP row 8 anchors + the 8
  lanes' drafts), ``"auto"`` captures both and uses T64 only for verify steps whose drafts do not fit idle lanes.
  :class:`SpecDecodeBatch` / :class:`SpecDecodeResult` stay as they are; the bridge asks
  :meth:`MotifGenerator.drafts_all_lanes` whether every live lane may draft.

Import rule (design 00 §2.1): this module is imported by vLLM's API server, its registry-inspection subprocess and
EngineCore before any mesh exists. It imports only the standard library and torch, never ttnn or other
``models/demos/**`` packages, and never touches a device (``huggingface_hub`` is imported lazily, only to locate the
HF-cache snapshot of a repo-id ``HF_MODEL``). ``tt/model_config.py`` imports the KV-pool and weights-location helpers
from here, so the bridge and the TT config share one implementation.
"""

from __future__ import annotations

import abc
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import torch

# ----------------------------------------------------------------------------------------------------------------
# Draft-1 constants (design 00 §1.5, §2.3, §3.1; study 01 §2)
# ----------------------------------------------------------------------------------------------------------------
NUM_LANES = 32  # decode lanes = the batch the decode trace is captured for
LANES_PER_GROUP = 8  # lanes per DP group (mesh row of the logical (4, 8) mesh)
NUM_DP_GROUPS = NUM_LANES // LANES_PER_GROUP  # 4
MESH_SHAPES = ((4, 8), (8, 4))  # logical (4, 8) preferred; the plugin's "BH-Galaxy" preset opens (8, 4)

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576 = 18 tiles
NUM_HIDDEN_LAYERS = 53
VOCAB_SIZE = 220160
MAX_CONTEXT = 32768  # draft-1 max_model_len (largest prefill bucket)
MIN_PREFILL_BUCKET = 128
# The last prefill bucket is max_model_len itself, so it must be a whole number of SDPA prefill chunks (256 on global
# layers, gate G2) and of KV blocks (<= 64): vLLM's --max-model-len must be a multiple of this (BRIDGE-3).
MAX_MODEL_LEN_ALIGNMENT = 256
# Block sizes the paged latent ops are validated with on this Galaxy: G1 (paged FlashMLA decode) and G7 (paged
# update / fill) ran blocks 32 and 64 only. Re-add 128 only after GATE-4 covers it (WAVE_A_REVIEW M4, BRIDGE-1).
SUPPORTED_BLOCK_SIZES = (32, 64)
DEFAULT_BLOCK_SIZE = 64

KV_CACHE_DTYPES = ("bfp8", "bf16")
DEFAULT_KV_CACHE_DTYPE = "bfp8"
_TILE = 32
_TILE_BYTES = {"bfp8": 1088, "bf16": 2048}  # one 32x32 tile: bfp8_b = 1024 mantissa + 64 exponent bytes

# ---- KV pool sizing shared by the bridge (vllm-tt-plugin contract) and MotifTTConfig ---------------------------
DEFAULT_KV_POOL_TOKENS = 262144  # usable pool = TIS max_tokens_all_users_override (design 00 §5.2)
KV_POOL_ALIGNMENT = 128  # pool multiple of every supported block (and of 128) -> the reserve below adds exactly 1 block
NULL_BLOCK_RESERVE_TOKENS = 32  # <= one block for every supported block size: pays for vLLM's null block 0
MAX_KV_POOL_TOKENS = 4 * 1024 * 1024

# ---- Device memory regions the mesh must be opened with (vllm-tt-plugin "tt" additional config) ---------------
# L1_SMALL bytes per core. Every CCL keeps its global semaphores (64 B each) there: ttnn's direct reduce-scatter and
# all-gather pick L1_SMALL by themselves when the region exists, and tt/ccl.py MotifCCL routes every other CCL path
# there explicitly. Without the region the semaphores land in main L1 at whatever address is free when the program is
# first built (~1.0 MB next to a decode all_reduce's 512 KB staging buffer) and stay there for the life of the program
# cache; the first later program whose static circular buffers reach that address throws "Statically allocated
# circular buffers ... clash with L1 buffers": a global-layer prefill at S >= 1024 after any decode step, or the
# bf16-KV FlashMLA decode (tt/attention.py docstring, "L1_SMALL is required"). 32 KiB = 512 semaphores; attention alone
# uses 16-18. pytest: model_config.device_params() sets it; vLLM: --additional-config '{"tt": {"l1_small_size": ...}}'
# (vllm-tt-plugin worker.py device_params_from_tt_config); TIS: override_tt_config.l1_small_size.
L1_SMALL_SIZE = 32768

# The "tt" additional config a Motif-3 server is launched with (design 00 §5.1; TIS override_tt_config):
#   --additional-config '{"tt": {"trace_mode": "decode_only", "trace_region_size": 268435456,
#                                "fabric_config": "FABRIC_2D_TORUS_XY", "dispatch_core_axis": "col",
#                                "l1_small_size": 32768}}'
# Only l1_small_size is enforced (check_tt_config); the rest are the validated defaults (decode traced, prefill eager).
SERVING_TT_CONFIG = {
    "trace_mode": "decode_only",
    "trace_region_size": 268435456,
    "fabric_config": "FABRIC_2D_TORUS_XY",
    "dispatch_core_axis": "col",
    "l1_small_size": L1_SMALL_SIZE,
}


def serving_additional_config(**overrides: Any) -> Dict[str, Any]:
    """``{"tt": SERVING_TT_CONFIG | overrides}``: the value of vLLM's ``--additional-config`` for Motif-3 (pass it
    through ``json.dumps`` on a command line)."""
    tt = dict(SERVING_TT_CONFIG)
    tt.update(overrides)
    return {"tt": tt}


def check_tt_config(tt_config: Optional[Mapping[str, Any]], *, where: str = "--additional-config") -> Dict[str, Any]:
    """Validate the plugin's ``"tt"`` additional config for Motif-3 and return it as a dict.

    Raises ``ValueError`` unless ``l1_small_size >= L1_SMALL_SIZE`` (the plugin opens the mesh with exactly the
    ``l1_small_size`` given there, and with none when the key is absent; see :data:`L1_SMALL_SIZE` for why the model
    cannot run without it). ``tt_config`` is the ``"tt"`` object of vLLM's ``additional_config`` (``None`` or ``{}``
    when the server was started without one)."""
    tt = dict(tt_config or {})
    fix = (
        f'pass --additional-config \'{{"tt": {{..., "l1_small_size": {L1_SMALL_SIZE}}}}}\' (TIS: override_tt_config '
        f"l1_small_size: {L1_SMALL_SIZE}); recommended: {SERVING_TT_CONFIG}"
    )
    raw = tt.get("l1_small_size")
    if raw is None:
        raise ValueError(
            f"Motif-3 needs an L1_SMALL region of >= {L1_SMALL_SIZE} B per core for its CCL semaphores, but {where} "
            f'has no "l1_small_size" (the mesh would open without one); {fix}'
        )
    try:
        size = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: l1_small_size must be an integer byte count, got {raw!r}; {fix}") from None
    if size < L1_SMALL_SIZE:
        raise ValueError(f"{where}: l1_small_size={size} is below the {L1_SMALL_SIZE} B Motif-3 needs; {fix}")
    return tt


def prefill_buckets(max_seq_len: int = MAX_CONTEXT, min_bucket: int = MIN_PREFILL_BUCKET) -> Tuple[int, ...]:
    """Power-of-two prefill lengths ``min_bucket .. max_seq_len`` (the last one is ``max_seq_len`` itself).

    Every bucket must be compiled by ``warmup_prefill`` before the decode trace is captured: a prefill shape first
    compiled after capture can corrupt the trace (vllm-tt-plugin ``model_runner.py:3735-3745``).
    """
    out, b = [], int(min_bucket)
    while b < max_seq_len:
        out.append(b)
        b *= 2
    out.append(int(max_seq_len))
    return tuple(out)


def kv_cache_bytes_per_chip(
    num_blocks: int, block_size: int, num_layers: int, kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE
) -> int:
    """Device bytes of the paged latent pool on ONE chip (the pool is replicated on every chip).

    ``num_layers x num_blocks x (block_size / 32) x 18 tiles x tile bytes`` (bfp8 1088 B, bf16 2048 B per tile).
    Draft 1 (4129 blocks of 64, 53 layers, bfp8) = 8.57 GB per chip (design 00 §1.1).
    """
    if kv_cache_dtype not in _TILE_BYTES:
        raise ValueError(f"kv_cache_dtype must be one of {KV_CACHE_DTYPES}, got {kv_cache_dtype!r}")
    if block_size % _TILE:
        raise ValueError(f"block_size {block_size} is not a multiple of the {_TILE}-row tile")
    tiles_per_block = (block_size // _TILE) * (KV_LATENT_DIM // _TILE)
    return int(num_layers) * int(num_blocks) * tiles_per_block * _TILE_BYTES[kv_cache_dtype]


def plugin_num_blocks(max_tokens_all_users: int, block_size: int, max_num_seqs: int) -> int:
    """The block count vllm-tt-plugin allocates (``worker.get_num_available_blocks_tt``, AR model, no hybrid
    headroom): ``ceil((max_tokens_all_users + block_size * max_num_seqs) / block_size)``; vLLM then keeps block 0."""
    return cdiv(int(max_tokens_all_users) + int(block_size) * int(max_num_seqs), int(block_size))


def expected_num_blocks(
    pool_tokens: int = DEFAULT_KV_POOL_TOKENS, block_size: int = DEFAULT_BLOCK_SIZE, max_num_seqs: int = NUM_LANES
) -> int:
    """``num_blocks`` that ``allocate_kv_cache`` receives under the bridge's sizing: the usable pool plus
    ``NULL_BLOCK_RESERVE_TOKENS`` from ``get_max_tokens_all_users``, plus the plugin's ``block_size * max_num_seqs``
    reservation, in whole blocks. 4129 for the defaults (262,144 / 64 / 32), 4105 with ``max_num_seqs=8``.

    Planning only: a generator must use the ``num_blocks`` it is given (WAVE_A_REVIEW M2/M3)."""
    return plugin_num_blocks(int(pool_tokens) + NULL_BLOCK_RESERVE_TOKENS, block_size, max_num_seqs)


def kv_pool_tokens_from_env(environ: Optional[Mapping[str, str]] = None) -> int:
    """Usable KV pool in tokens: ``MOTIF3_KV_POOL_TOKENS`` (default 262,144), a multiple of 128."""
    env = os.environ if environ is None else environ
    raw = env.get("MOTIF3_KV_POOL_TOKENS")
    if raw is None or raw.strip() == "":
        return DEFAULT_KV_POOL_TOKENS
    raw = raw.strip()
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"MOTIF3_KV_POOL_TOKENS must be a positive decimal token count, got {raw!r}")
    tokens = int(raw)
    if tokens % KV_POOL_ALIGNMENT or not KV_POOL_ALIGNMENT <= tokens <= MAX_KV_POOL_TOKENS:
        raise ValueError(
            f"MOTIF3_KV_POOL_TOKENS={tokens} must be a multiple of {KV_POOL_ALIGNMENT} in "
            f"[{KV_POOL_ALIGNMENT}, {MAX_KV_POOL_TOKENS}]"
        )
    return tokens


def check_block_size(block_size: int) -> int:
    """``block_size`` if it is in ``SUPPORTED_BLOCK_SIZES``, else ``ValueError`` (vLLM's default 16 included)."""
    block_size = int(block_size)
    if block_size not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            f"Motif-3 needs --block-size in {SUPPORTED_BLOCK_SIZES} (the block sizes the paged latent ops are "
            f"validated with on this Galaxy, gates G1/G7; 64 is the default), got {block_size}. vLLM's default of 16 "
            f"is not usable on TT."
        )
    return block_size


def check_max_model_len(max_model_len: int) -> int:
    """``max_model_len`` if it is in ``[MAX_MODEL_LEN_ALIGNMENT, MAX_CONTEXT]`` and a multiple of
    ``MAX_MODEL_LEN_ALIGNMENT`` (256), else ``ValueError``. The last prefill bucket is ``max_model_len`` itself, so it
    must be a whole number of SDPA prefill chunks (256 on global layers) and of KV blocks (BRIDGE-3, M6)."""
    n = int(max_model_len)
    if not 1 <= n <= MAX_CONTEXT:
        raise ValueError(f"max_model_len must be in [1, {MAX_CONTEXT}] in draft 1, got {n}")
    if n % MAX_MODEL_LEN_ALIGNMENT:
        hint = max(MAX_MODEL_LEN_ALIGNMENT, n // MAX_MODEL_LEN_ALIGNMENT * MAX_MODEL_LEN_ALIGNMENT)
        raise ValueError(
            f"max_model_len={n} must be a multiple of {MAX_MODEL_LEN_ALIGNMENT}: the last prefill bucket is "
            f"max_model_len itself and must align to the SDPA prefill chunk (256 on global layers) and the KV block; "
            f"pass e.g. --max-model-len {hint}"
        )
    return n


# ----------------------------------------------------------------------------------------------------------------
# Features: chunked prefill, prefix caching, MTP speculative decoding (docs/features/FEATURES_DESIGN.md §1.3, §2)
# ----------------------------------------------------------------------------------------------------------------
# Class-capability switches of the bridge (read at import time, identical in the API server and EngineCore). They only
# allow vLLM to enable a feature; vLLM's own flags decide (GeneratorSettings.chunked_prefill / prefix_caching /
# spec_tokens carry what vLLM enabled).
FEATURE_SWITCHES = ("MOTIF3_PREFIX_CACHING", "MOTIF3_CHUNKED_PREFILL", "MOTIF3_SPEC_DECODE")
# Span cap (MOTIF3_PREFILL_MAX_BUCKET): the largest sp0/sp1 bucket a resumed-prefill generator compiles; longer spans
# are split into chunks inside one prefill_forward_batch call (design D8). 32768 compiles every draft-1 bucket (no
# forced split; the planner's cost-based head split still turns 16,736 into 16384 + 512).
DEFAULT_PREFILL_SPAN_CAP = 8192
# Resume alignment A = lcm(block, every q_chunk / k_chunk) of the sp1 global op: 128 with gate G9's per-bucket chunks
# (prefill_plan.DEFAULT_SP1_GLOBAL_CHUNKS: 128/128 at C = 128 and C >= 2048, 64/64 at 256-1024; lead decision F5), so
# the vLLM budget = threshold = 8192 - 128 = 8064. Only for checks that run before the generator exists (and
# FEATURE_VLLM_ARGS); the generator's ``prefill_alignment`` is authoritative (test_prefill_plan checks they agree).
DEFAULT_PREFILL_ALIGNMENT = 128
# Chunk budget target (MOTIF3_CHUNK_BUDGET; OPTIMIZATION_PLAN.md §3.3 A1a): None = the span cap's own budget
# (prefill_plan.recommended_budget: span cap - A = 8064, the code default until the TIS A/B decides); an integer asks
# for that budget, made alignment-aware by recommended_budget (rounded down to a multiple of A, at most cap - A).
# vLLM's --max-num-batched-tokens / --long-prefill-token-threshold set the real budget: the knob only derives the
# flags (generator_vllm.feature_vllm_args, the launch scripts) and makes check_serving_config warn when they differ.
DEFAULT_CHUNK_BUDGET: Optional[int] = None
SUPPORTED_SPEC_TOKENS = (0, 1)  # MTP draft tokens per step (num_speculative_tokens): K = 1 only
MTP_LAYER_IDX = NUM_HIDDEN_LAYERS  # the MTP layer is reference layer 53 (TT-cache part "L53")
# Decode KV-write modes (tt/kv_write.py): "row" = draft 1 (one 8-lane update per DP row), "row_split" = speculation
# without KV-R (two 8-lane calls: anchors / plain lanes, then draft lanes), "all" = KV-R (gather the 8-lane latent
# over DP, one 32-lane update on every chip), "all_split" = KV-R + speculation (production). See kv_write_mode().
KV_WRITE_MODES = ("row", "row_split", "all", "all_split")
# Speculative verify (MOTIF3_SPEC_VERIFY; docs/p5_t64/P5_T64_DESIGN.md §2.2, §4):
#   "packed" (default, S1): a draft runs on an idle lane of the 32-lane spec trace (T32); drafts that do not fit run
#            in a second T32 replay (the overflow pass);
#   "wide"   (S3): the 64-row trace (T64) alone serves every step (the one-trace fallback);
#   "auto":  both traces, each captured once at warmup: T32 for ordinary steps and for verify steps whose drafts all
#            fit idle lanes, one T64 replay for the other verify steps (no overflow pass for bridge traffic).
# "wide" / "auto" need MotifTTConfig.ring_gather == "safe" (F3N rule R1; the config refuses anything else).
SPEC_VERIFY_MODES = ("packed", "wide", "auto")
WIDE_SPEC_VERIFY_MODES = ("wide", "auto")  # the modes that stage and capture the T64 trace
# T64 rows: per DP row [8 anchors | 8 drafts] (a draft on its owner's DP row at n + 1, with the owner's page-table row),
# still one 32-row tile row; the gathered step has 64 rows (split order: 0..31 = the anchors in lane order, 32..63 =
# the drafts).
WIDE_ROWS = 2 * NUM_LANES  # 64
WIDE_ROWS_PER_GROUP = 2 * LANES_PER_GROUP  # 16
# T64 drafting policy (§4.5, §4.7; review edits R-E3, R-E9). In "auto" the bridge drafts every live lane once the live
# lanes reach c* = verify_plan.crossover_lanes(alpha, MotifTTConfig.wide_step_ratio) in [17, 33] (33 = never), or
# GeneratorSettings.wide_min_lanes (MOTIF3_WIDE_MIN_LANES) when set; below it the idle-lane budget stays. alpha is the
# running acceptance pulled toward a prior (smoothed_acceptance): a server whose first traffic is a 32-request burst
# must still draft, or the idle-lane budget alone would never let it measure alpha.
WIDE_MIN_LANES_NEVER = NUM_LANES + 1  # 33: "never draft every lane"
DEFAULT_SPEC_ALPHA_PRIOR = 0.85  # alpha_0
SPEC_ALPHA_PRIOR_WEIGHT = 64  # n_0: the prior weighs as much as 64 verified drafts
# Packed multi-row prefill (P5; docs/p5_t64/P5_T64_DESIGN.md §3; MOTIF3_PACKED_PREFILL is off in the code and on in
# both TIS specs since gates CP-P, CP9-P and E2E-P passed). The short chunks of one prefill_forward_batch call run
# together: B segments of S rows each (B in PACK_BATCHES, dummy segments fill B), T = B * S rows, an existing prefill
# bucket; only the SDPA, the RoPE rows and the KV fill are per segment. "pk0" = sp0 segments (start 0), "pk1" = sp1
# segments at one common start. The generator warms every packed shape (MotifTTConfig.packed_prefill_shapes()) before
# the decode capture; after it, a pass of an unwarmed shape runs as solo chunks (packing never refuses a call).
PACKED_PASS_KINDS = ("pk0", "pk1")
PACK_SEG_BUCKETS = (64, 128, 256, 512, 1024)  # pk0 segment rows S: the smallest one >= the chunk's rows
PACK_SP1_SEG_BUCKETS = (128, 256, 512, 1024)  # pk1 S (S = 64 would need a 64/64 config for the 192-row SWA square)
PACK_BATCHES = (2, 4, 8, 16, 32)  # B: a pass holds at most the NUM_LANES rows of one call
# Review edit R-E2: a pk1 pass gathers its SWA tails once ("shared": all B tail sets identical) or per segment
# ("distinct"). They are different programs, so both are warmed per pk1 shape and the shape key names the variant.
PK1_TAIL_VARIANTS = ("shared", "distinct")
DEFAULT_PACKED_PREFILL_MAX_SEG = 1024  # MOTIF3_PACKED_PREFILL_MAX_SEG: the largest packed segment S
DEFAULT_PACKED_PREFILL_MAX_TOKENS = 8192  # MOTIF3_PACKED_PREFILL_MAX_TOKENS: the largest packed pass T (<= span cap)
PACKED_WARMUP_MODES = ("attention", "full")  # MOTIF3_PACKED_WARMUP: attention-only warm calls per shape | a full pass
DEFAULT_PACKED_WARMUP = "attention"
# Keys of the bridge's captured vLLM scheduler config (``GeneratorSettings.from_env(serving=...)``).
SERVING_KEYS = (
    "block_size",
    "enable_chunked_prefill",
    "max_num_batched_tokens",
    "long_prefill_token_threshold",
    "enable_prefix_caching",
    "prefix_match_unit",
    "spec_tokens",
)

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _parse_switch(name: str, raw: Optional[str], default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return bool(default)
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{name} must be one of {_TRUE + _FALSE}, got {raw!r}")


def feature_switch_from_env(name: str, environ: Optional[Mapping[str, str]] = None, default: bool = True) -> bool:
    """One of :data:`FEATURE_SWITCHES` (``1/true/yes/on`` or ``0/false/no/off``; unset = ``default``). A typo
    raises instead of silently turning a feature off."""
    if name not in FEATURE_SWITCHES:
        raise ValueError(f"unknown feature switch {name!r}; known: {FEATURE_SWITCHES}")
    env = os.environ if environ is None else environ
    return _parse_switch(name, env.get(name), default)


def kv_replicated_decode_from_env(environ: Optional[Mapping[str, str]] = None) -> Optional[bool]:
    """``MOTIF3_KV_REPLICATED_DECODE``: ``auto`` / unset -> None (= on iff prefix caching), ``1`` -> True (forced on),
    ``0`` -> False (refused by :class:`GeneratorSettings` when prefix caching is on)."""
    env = os.environ if environ is None else environ
    raw = env.get("MOTIF3_KV_REPLICATED_DECODE")
    if raw is None or raw.strip().lower() in ("", "auto"):
        return None
    return _parse_switch("MOTIF3_KV_REPLICATED_DECODE", raw, False)


def check_prefill_span_cap(cap: int, max_seq_len: int = MAX_CONTEXT) -> int:
    """``cap`` if it is a power of two in ``[MIN_PREFILL_BUCKET, MAX_CONTEXT]`` or ``max_seq_len`` itself, else
    ``ValueError``. The effective cap is ``min(cap, max_seq_len)``, always one of ``prefill_buckets(max_seq_len)``."""
    c = int(cap)
    pow2 = c >= MIN_PREFILL_BUCKET and c & (c - 1) == 0 and c <= MAX_CONTEXT
    if not (pow2 or c == int(max_seq_len)):
        raise ValueError(
            f"prefill span cap {c} must be a power of two in [{MIN_PREFILL_BUCKET}, {MAX_CONTEXT}] (a prefill bucket) "
            f"or max_seq_len {max_seq_len}"
        )
    return c


def prefill_span_cap_from_env(environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """``MOTIF3_PREFILL_MAX_BUCKET`` (a power of two in [128, 32768]) or None when unset (the generator's default:
    :data:`DEFAULT_PREFILL_SPAN_CAP` with resumed prefill, else ``max_seq_len``)."""
    env = os.environ if environ is None else environ
    v = _env_int(env, "MOTIF3_PREFILL_MAX_BUCKET")
    return None if v is None else check_prefill_span_cap(v, MAX_CONTEXT)


def check_chunk_budget(budget: int) -> int:
    """``budget`` if it is an integer in ``[MIN_PREFILL_BUCKET, MAX_CONTEXT]`` (the A1a chunk-budget target), else
    ``ValueError``. Alignment is applied later (``prefill_plan.recommended_budget``), not required here."""
    n = int(budget)
    if not MIN_PREFILL_BUCKET <= n <= MAX_CONTEXT:
        raise ValueError(
            f"the chunk budget (MOTIF3_CHUNK_BUDGET) must be 'auto' or a token count in [{MIN_PREFILL_BUCKET}, "
            f"{MAX_CONTEXT}], got {budget!r}"
        )
    return n


def chunk_budget_from_env(environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """``MOTIF3_CHUNK_BUDGET``: unset, empty or ``auto`` = None (:data:`DEFAULT_CHUNK_BUDGET`: the span cap's own
    budget, 8064), else the requested per-step token budget (:func:`check_chunk_budget`; e.g. 4096 for A1a)."""
    env = os.environ if environ is None else environ
    raw = env.get("MOTIF3_CHUNK_BUDGET")
    if raw is None or raw.strip().lower() in ("", "auto"):
        return DEFAULT_CHUNK_BUDGET
    v = _env_int(env, "MOTIF3_CHUNK_BUDGET")
    return check_chunk_budget(v)


def packed_prefill_from_env(environ: Optional[Mapping[str, str]] = None) -> bool:
    """``MOTIF3_PACKED_PREFILL`` (default off; both TIS specs set ``1`` since gates CP-P / CP9-P / E2E-P passed):
    packed multi-row prefill (P5, docs/p5_t64/P5_T64_DESIGN.md §3)."""
    env = os.environ if environ is None else environ
    return _parse_switch("MOTIF3_PACKED_PREFILL", env.get("MOTIF3_PACKED_PREFILL"), False)


def check_packed_prefill_max_seg(max_seg: int) -> int:
    """``max_seg`` if it is one of :data:`PACK_SEG_BUCKETS` (the segment sizes gate G15 validates), else
    ``ValueError``. Segments of more rows never pack: their chunks run solo."""
    s = int(max_seg)
    if s not in PACK_SEG_BUCKETS:
        raise ValueError(
            f"the largest packed prefill segment (MOTIF3_PACKED_PREFILL_MAX_SEG) must be one of {PACK_SEG_BUCKETS}, "
            f"got {max_seg!r}"
        )
    return s


def check_packed_prefill_max_tokens(max_tokens: int) -> int:
    """``max_tokens`` if it is a power of two in ``[MIN_PREFILL_BUCKET, MAX_CONTEXT]`` (a packed pass of ``T = B * S``
    rows runs the bucket-T programs), else ``ValueError``. The effective cap is ``min(max_tokens, span cap)``."""
    t = int(max_tokens)
    if not (MIN_PREFILL_BUCKET <= t <= MAX_CONTEXT and t & (t - 1) == 0):
        raise ValueError(
            f"the largest packed prefill pass (MOTIF3_PACKED_PREFILL_MAX_TOKENS) must be a power of two in "
            f"[{MIN_PREFILL_BUCKET}, {MAX_CONTEXT}], got {max_tokens!r}"
        )
    return t


def packed_prefill_max_seg_from_env(environ: Optional[Mapping[str, str]] = None) -> int:
    """``MOTIF3_PACKED_PREFILL_MAX_SEG``: the largest packed segment S, one of :data:`PACK_SEG_BUCKETS` (default
    :data:`DEFAULT_PACKED_PREFILL_MAX_SEG` = 1024). It caps the pk0 and the pk1 segment sizes."""
    env = os.environ if environ is None else environ
    v = _env_int(env, "MOTIF3_PACKED_PREFILL_MAX_SEG")
    return DEFAULT_PACKED_PREFILL_MAX_SEG if v is None else check_packed_prefill_max_seg(v)


def packed_prefill_max_tokens_from_env(environ: Optional[Mapping[str, str]] = None) -> int:
    """``MOTIF3_PACKED_PREFILL_MAX_TOKENS``: the largest packed pass ``T = B * S`` (a power of two; default
    :data:`DEFAULT_PACKED_PREFILL_MAX_TOKENS` = 8192)."""
    env = os.environ if environ is None else environ
    v = _env_int(env, "MOTIF3_PACKED_PREFILL_MAX_TOKENS")
    return DEFAULT_PACKED_PREFILL_MAX_TOKENS if v is None else check_packed_prefill_max_tokens(v)


def packed_prefill_pk1_from_env(environ: Optional[Mapping[str, str]] = None) -> bool:
    """``MOTIF3_PACKED_PREFILL_PK1`` (default on): pack resumed (sp1) chunks that share a start (P5b). Off: those chunks
    run solo (gate G15a's fallback for the pk1 SWA square)."""
    env = os.environ if environ is None else environ
    return _parse_switch("MOTIF3_PACKED_PREFILL_PK1", env.get("MOTIF3_PACKED_PREFILL_PK1"), True)


def packed_warmup_from_env(environ: Optional[Mapping[str, str]] = None) -> str:
    """``MOTIF3_PACKED_WARMUP``: ``attention`` (default: per packed shape, the attention of one global and one SWA layer
    on zeros, ~1 s per boot) or ``full`` (one full packed pass per shape, ~60-70 s)."""
    env = os.environ if environ is None else environ
    v = (env.get("MOTIF3_PACKED_WARMUP") or "").strip().lower() or DEFAULT_PACKED_WARMUP
    if v not in PACKED_WARMUP_MODES:
        raise ValueError(f"MOTIF3_PACKED_WARMUP must be one of {PACKED_WARMUP_MODES}, got {v!r}")
    return v


# B6a (docs/OPT_PHASE_A_REVIEW.md §7.1-7.2; logs/opt/phaseB/B6a): the per-decode-step host input staging of the
# bridge and the generator (default "fast" since the B6a gates). "release" = the release code; "fast" = the same device
# inputs built with fewer host ops
# (cached mesh mappers, page-table work only on the used columns, persistent inputs whose values did not change are not
# copied again, cached lane index tensors). The device sees bit-identical inputs either way.
HOST_STAGING_MODES = ("release", "fast")
DEFAULT_HOST_STAGING = "fast"


def check_host_staging(mode: Any, *, name: str = "MOTIF3_HOST_STAGING") -> str:
    """``mode`` lower-cased and stripped if it is one of :data:`HOST_STAGING_MODES` (``""`` / None = the default),
    else ``ValueError`` naming ``name``."""
    v = str(mode if mode is not None else "").strip().lower() or DEFAULT_HOST_STAGING
    if v not in HOST_STAGING_MODES:
        raise ValueError(f"{name} must be one of {HOST_STAGING_MODES}, got {mode!r}")
    return v


def host_staging_from_env(environ: Optional[Mapping[str, str]] = None) -> str:
    """``MOTIF3_HOST_STAGING``: ``fast`` (default) or ``release`` (B6a, :data:`HOST_STAGING_MODES`); case and blanks
    ignored, anything else raises ``ValueError``."""
    env = os.environ if environ is None else environ
    return check_host_staging(env.get("MOTIF3_HOST_STAGING"))


# B6a: how the generator waits for a replayed decode trace (default "spin" since the B6a gates). "block" (the
# release) = the step's blocking read sleeps
# until the device is done (~85 ms of idle core: under the schedutil governor the host code that follows runs ~2.5-3x
# slower, logs/opt/phaseB/B6a); "spin" = the calling thread first polls (time.sleep(0): the GIL is released every
# iteration) until a few ms before the predicted end of the replay, then makes the same blocking read. Host only.
HOST_WAIT_MODES = ("block", "spin")
DEFAULT_HOST_WAIT = "spin"


def check_host_wait(mode: Any, *, name: str = "MOTIF3_HOST_WAIT") -> str:
    """``mode`` lower-cased and stripped if it is one of :data:`HOST_WAIT_MODES` (``""`` / None = the default), else
    ``ValueError`` naming ``name``."""
    v = str(mode if mode is not None else "").strip().lower() or DEFAULT_HOST_WAIT
    if v not in HOST_WAIT_MODES:
        raise ValueError(f"{name} must be one of {HOST_WAIT_MODES}, got {mode!r}")
    return v


# B6b: vLLM asynchronous scheduling for the non-MTP launches (OPTIMIZATION_PLAN.md §3.3 B6, OPT_PHASE_A_REVIEW.md §7.2).
# "on": the bridge declares ``supports_async_decode`` (decode-reload contract v1, full adapter): a device-sampled decode
# step returns before its 1 KB read (``read_from_device=False``), and a steady step (``reload_inputs=False``) takes its
# tokens from the previous step's read and its positions from the previous step + 1, so vLLM's scheduling and the
# plugin's input build for step k + 1 run while step k replays. The device runs the same trace on the same inputs.
# "off" (default) = the release: every decode reloads its inputs and reads its result before returning. The MTP launch
# keeps "off" (the plugin refuses async scheduling with speculation unless supports_async_spec_decode).
ASYNC_DECODE_MODES = ("off", "on")
DEFAULT_ASYNC_DECODE = "off"


def check_async_decode(mode: Any, *, name: str = "MOTIF3_ASYNC_DECODE") -> str:
    """``mode`` lower-cased and stripped if it is one of :data:`ASYNC_DECODE_MODES` (``""`` / None = the default),
    else ``ValueError`` naming ``name``."""
    v = str(mode if mode is not None else "").strip().lower() or DEFAULT_ASYNC_DECODE
    if v not in ASYNC_DECODE_MODES:
        raise ValueError(f"{name} must be one of {ASYNC_DECODE_MODES}, got {mode!r}")
    return v


def async_decode_from_env(environ: Optional[Mapping[str, str]] = None) -> str:
    """``MOTIF3_ASYNC_DECODE``: ``off`` (default) or ``on`` (B6b, :data:`ASYNC_DECODE_MODES`); case and blanks ignored,
    anything else raises ``ValueError``."""
    env = os.environ if environ is None else environ
    return check_async_decode(env.get("MOTIF3_ASYNC_DECODE"))


# E3 / B7: the host thread a trace capture runs on. A capture keeps thousands of small host objects alive for the
# trace's lifetime; made on the serving thread they fragment glibc's main malloc arena and every later eager prefill
# pass of a dispatch-bound shape (128-512 rows, packed passes) runs ~20-30 ms slower per live trace (logs/opt/phaseB/B7,
# E3b-E3d). "worker" = each capture on a short-lived worker thread (its own malloc arena; joined before the capture
# returns); "main" = the calling thread (the release, and the default: served with MOTIF3_PREFILL_TRACE=128, worker
# captures cost +60-75 ms TTFT at 1K and +15-30 ms at 4K, logs/opt/phaseB/B7/report.md). The device receives the same
# commands either way. "dedicated" (B7-FIX, logs/opt/phaseB2/B7-FIX): every capture on ONE long-lived daemon thread that
# never exits, so its arena never goes back to glibc's free list (a short-lived "worker" thread's arena is inherited
# by the next thread started, e.g. the serving engine's I/O threads). In process (E3f, n = 8) eager sp0 128 / 1024 /
# 4096 after all captures: main +42 / +57 / +17 ms, dedicated +2 / +5 / +2 ms. Served A/B pending (device fault).
CAPTURE_THREAD_MODES = ("main", "worker", "dedicated")
DEFAULT_CAPTURE_THREAD = "main"


def check_capture_thread(mode: Any, *, name: str = "MOTIF3_CAPTURE_THREAD") -> str:
    """``mode`` lower-cased and stripped if it is one of :data:`CAPTURE_THREAD_MODES` (``""`` / None = the default),
    else ``ValueError`` naming ``name``."""
    v = str(mode if mode is not None else "").strip().lower() or DEFAULT_CAPTURE_THREAD
    if v not in CAPTURE_THREAD_MODES:
        raise ValueError(f"{name} must be one of {CAPTURE_THREAD_MODES}, got {mode!r}")
    return v


def capture_thread_from_env(environ: Optional[Mapping[str, str]] = None) -> str:
    """``MOTIF3_CAPTURE_THREAD``: :data:`CAPTURE_THREAD_MODES`; case and blanks ignored, anything else raises."""
    env = os.environ if environ is None else environ
    return check_capture_thread(env.get("MOTIF3_CAPTURE_THREAD"))


# B7: traced prefill of small solo chunks (OPTIMIZATION_PLAN.md §3.3 B7, OPT_PHASE_A_REVIEW.md §7.1 M8; prototype
# logs/opt/phaseA/m8; results logs/opt/phaseB/B7). "off" = the release: every prefill chunk runs eagerly (~0.71 s of
# host dispatch for a 128-row chunk whose device time is ~0.21 s). A comma list of buckets (from PREFILL_TRACE_BUCKETS;
# "on" = "128"; "128" is the default since its gates passed) = at the decode capture the generator also captures one trace per (sp0, b) and
# (sp1, b) of those buckets (embedding -> layers -> LM-head tile row, plus a second small trace for the MTP layer's
# KV-only fill on a speculating launch) and replays it for every solo chunk of that shape. Traced == eager bitwise.
# 1024 is not offered: it is device bound (M8: 1.001 -> 0.984 s) and its trace would cost ~50 MiB of the trace region.
PREFILL_TRACE_BUCKETS = (128, 256, 512)
DEFAULT_PREFILL_TRACE = "128"


def check_prefill_trace(value: Any, *, name: str = "MOTIF3_PREFILL_TRACE") -> str:
    """The canonical form of a traced-prefill setting: ``"off"`` or an ascending comma list of buckets from
    :data:`PREFILL_TRACE_BUCKETS` (``"128"``, ``"128,256"``); ``"on"`` = ``"128"``; ``""`` / None = the default. Case
    and blanks are ignored; anything else raises ``ValueError`` naming ``name``."""
    v = str(value if value is not None else "").strip().lower() or DEFAULT_PREFILL_TRACE
    if v == "off":
        return "off"
    if v == "on":
        return str(PREFILL_TRACE_BUCKETS[0])
    out = set()
    for part in v.split(","):
        p = part.strip()
        if not p.isdigit() or int(p) not in PREFILL_TRACE_BUCKETS:
            raise ValueError(
                f"{name} must be 'off', 'on' or a comma list of buckets from {PREFILL_TRACE_BUCKETS}, got {value!r}"
            )
        out.add(int(p))
    return ",".join(str(b) for b in sorted(out))


def prefill_trace_buckets(value: Any, *, name: str = "MOTIF3_PREFILL_TRACE") -> Tuple[int, ...]:
    """The traced prefill buckets of a setting (:func:`check_prefill_trace`): ``()`` for ``"off"``."""
    v = check_prefill_trace(value, name=name)
    return () if v == "off" else tuple(int(b) for b in v.split(","))


def prefill_trace_from_env(environ: Optional[Mapping[str, str]] = None) -> str:
    """``MOTIF3_PREFILL_TRACE`` (B7): ``128`` (default), ``off``, ``on`` (= ``128``) or a comma list of buckets from
    :data:`PREFILL_TRACE_BUCKETS`; canonical form of :func:`check_prefill_trace`."""
    env = os.environ if environ is None else environ
    return check_prefill_trace(env.get("MOTIF3_PREFILL_TRACE"))


def spec_verify_from_env(environ: Optional[Mapping[str, str]] = None) -> str:
    """``MOTIF3_SPEC_VERIFY``: ``packed`` (default, S1: drafts on idle lanes of the 32-lane trace), ``wide`` (the
    64-row trace alone, S3) or ``auto`` (both traces; the 64-row one only for verify steps whose drafts do not fit idle
    lanes). See :data:`SPEC_VERIFY_MODES`."""
    env = os.environ if environ is None else environ
    v = (env.get("MOTIF3_SPEC_VERIFY") or "").strip().lower() or "packed"
    if v not in SPEC_VERIFY_MODES:
        raise ValueError(f"MOTIF3_SPEC_VERIFY must be one of {SPEC_VERIFY_MODES}, got {v!r}")
    return v


def check_wide_min_lanes(lanes: int) -> int:
    """``lanes`` if it is in ``[1, WIDE_MIN_LANES_NEVER]`` (33 = never draft every lane), else ``ValueError``."""
    n = int(lanes)
    if not 1 <= n <= WIDE_MIN_LANES_NEVER:
        raise ValueError(
            f"the T64 drafting threshold (MOTIF3_WIDE_MIN_LANES) must be a live-lane count in "
            f"[1, {WIDE_MIN_LANES_NEVER}] ({WIDE_MIN_LANES_NEVER} = never), got {lanes!r}"
        )
    return n


def wide_min_lanes_from_env(environ: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """``MOTIF3_WIDE_MIN_LANES``: unset = None (the generator's acceptance-based ``c*``), else the live-lane count from
    which an ``auto`` launch drafts every lane (:func:`check_wide_min_lanes`)."""
    env = os.environ if environ is None else environ
    v = _env_int(env, "MOTIF3_WIDE_MIN_LANES")
    return None if v is None else check_wide_min_lanes(v)


def check_wide_step_ratio(ratio: float) -> float:
    """``ratio`` as a float if it is a finite T64 / T32-spec step-time ratio ``>= 1`` (the rule
    ``MotifTTConfig.validate`` applies to ``wide_step_ratio``), else ``ValueError`` (NaN and inf included)."""
    if isinstance(ratio, bool):
        raise TypeError(f"the T64 / T32 step ratio (MOTIF3_WIDE_STEP_RATIO) must be a number >= 1, got {ratio!r}")
    r = float(ratio)
    if not (math.isfinite(r) and r >= 1.0):
        raise ValueError(f"the T64 / T32 step ratio (MOTIF3_WIDE_STEP_RATIO) must be finite and >= 1, got {ratio!r}")
    return r


def wide_step_ratio_from_env(environ: Optional[Mapping[str, str]] = None) -> Optional[float]:
    """``MOTIF3_WIDE_STEP_RATIO``: unset = None (``MotifTTConfig.wide_step_ratio`` keeps
    ``model_config.DEFAULT_WIDE_STEP_RATIO``, 1.21), else the T64 step / T32-spec step device-time ratio ``r`` the
    ``auto`` drafting crossover ``c*`` uses (:func:`check_wide_step_ratio`: a finite float >= 1)."""
    env = os.environ if environ is None else environ
    raw = env.get("MOTIF3_WIDE_STEP_RATIO")
    if raw is None or raw.strip() == "":
        return None
    try:
        r = float(raw.strip())
    except ValueError:
        raise ValueError(f"MOTIF3_WIDE_STEP_RATIO must be a number >= 1, got {raw!r}") from None
    return check_wide_step_ratio(r)


def check_spec_alpha_prior(prior: float) -> float:
    """``prior`` as a float if it is an acceptance rate in ``[0, 1]``, else ``ValueError`` (NaN included)."""
    if isinstance(prior, bool):
        raise TypeError(f"the acceptance prior must be a number in [0, 1], got {prior!r}")
    p = float(prior)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"the acceptance prior must be in [0, 1], got {prior!r}")
    return p


def smoothed_acceptance(
    accepted: int,
    verified: int,
    *,
    prior: float = DEFAULT_SPEC_ALPHA_PRIOR,
    weight: int = SPEC_ALPHA_PRIOR_WEIGHT,
) -> float:
    """The drafting policy's acceptance estimate (review edit R-E3): ``(accepted + weight * prior) / (verified +
    weight)``. That is ``prior`` before any draft was verified and tends to the measured rate ``accepted / verified``.
    ``accepted`` / ``verified``: drafts accepted / verified so far (the bridge's ``SpecStats``), ``0 <= accepted <=
    verified``; ``prior``: ``GeneratorSettings.spec_alpha_prior``; ``weight``: :data:`SPEC_ALPHA_PRIOR_WEIGHT`."""
    a, v, w, p = int(accepted), int(verified), int(weight), check_spec_alpha_prior(prior)
    if not 0 <= a <= v:
        raise ValueError(f"need 0 <= accepted <= verified, got accepted={accepted}, verified={verified}")
    if w < 0:
        raise ValueError(f"the prior weight must be >= 0, got {weight}")
    if v + w == 0:
        return p
    return (a + w * p) / (v + w)


def kv_write_mode(kv_replicated: bool, spec: bool) -> str:
    """The decode KV-write mode (:data:`KV_WRITE_MODES`): ``row`` (draft 1), ``row_split`` (speculation, no KV-R),
    ``all`` (KV-R), ``all_split`` (KV-R + speculation). The split modes write anchor / plain lanes and draft lanes in
    two ``paged_update_cache`` calls: two users writing ``p`` and ``p + 1`` of one block in one call race on the
    shared tile (design §3.5)."""
    return ("all" if kv_replicated else "row") + ("_split" if spec else "")


def check_generator_features(generator: "MotifGenerator", settings: "GeneratorSettings") -> None:
    """Refuse a generator that cannot serve what vLLM enabled (design §1.5, last row), and a generator whose resumed
    prefill geometry is inconsistent. Raises ``ValueError``."""
    name = type(generator).__name__
    if settings.resumed_prefill and not generator.supports_resumed_prefill:
        raise ValueError(
            f"vLLM enabled {'chunked prefill' if settings.chunked_prefill else 'prefix caching'} but {name} has no "
            f"resumed prefill (supports_resumed_prefill is False); launch with --no-enable-chunked-prefill "
            f"--no-enable-prefix-caching or set MOTIF3_CHUNKED_PREFILL=0 MOTIF3_PREFIX_CACHING=0"
        )
    if settings.spec_decode and not generator.supports_spec_decode:
        raise ValueError(
            f"vLLM enabled speculative decoding (num_speculative_tokens={settings.spec_tokens}) but {name} has no "
            f"decode_forward_spec (supports_spec_decode is False); drop --speculative-config or set "
            f"MOTIF3_SPEC_DECODE=0"
        )
    span, longest = int(generator.max_prefill_span), int(generator.max_prefill_len)
    if span < MIN_PREFILL_BUCKET or span > longest:
        raise ValueError(f"{name}.max_prefill_span {span} outside [{MIN_PREFILL_BUCKET}, max_prefill_len {longest}]")
    if generator.supports_resumed_prefill:
        A = int(generator.prefill_alignment)
        if A < 1 or (settings.block_size is not None and A % int(settings.block_size)):
            raise ValueError(f"{name}.prefill_alignment {A} must be a positive multiple of the block size")
    elif span < min(longest, int(settings.max_seq_len)):
        raise ValueError(f"{name} splits spans at {span} rows but has no resumed prefill to run the continuations")


# ----------------------------------------------------------------------------------------------------------------
# Weights location (one precedence order for the bridge and MotifTTConfig; design 00 §2.3.11; WAVE_A_REVIEW M7)
# ----------------------------------------------------------------------------------------------------------------
class WeightsLocation(NamedTuple):
    """Where the checkpoint is. ``path`` is a local directory when ``is_local``; otherwise a HF repo id that the
    generator must resolve (download) at ``revision`` itself, or ``None`` (nothing given: use the generator's
    default snapshot). ``source`` names the rule that matched (logged by the bridge)."""

    path: Optional[str]
    source: str
    revision: Optional[str]
    is_local: bool


def _looks_like_repo_id(value: str) -> bool:
    """``org/name`` (one slash, no leading path marker): what HF_MODEL holds when it is not a snapshot directory."""
    v = value.strip()
    if not v or v.startswith(("/", ".", "~")):
        return False
    parts = v.split("/")
    return len(parts) == 2 and all(parts)


def _hf_hub_cache_dir(environ: Optional[Mapping[str, str]] = None) -> Path:
    """The HF hub cache: ``HF_HUB_CACHE``, else ``$HF_HOME/hub``, else huggingface_hub's default."""
    env = os.environ if environ is None else environ
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"]).expanduser()
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser() / "hub"
    try:
        from huggingface_hub import constants as _hf_constants

        return Path(_hf_constants.HF_HUB_CACHE)
    except Exception:  # pragma: no cover - huggingface_hub missing
        return Path.home() / ".cache" / "huggingface" / "hub"


def hf_cache_snapshot(
    repo_id: str, revision: Optional[str] = None, *, cache_dir: Optional[Union[str, os.PathLike]] = None
) -> Optional[Path]:
    """Local HF-cache snapshot directory of ``repo_id`` at ``revision`` (full or >= 7-char commit hash, a branch/tag
    with a ``refs/`` entry, or ``None`` = ``main``), or ``None`` if it is not in the cache (never downloads).
    ``cache_dir`` defaults to :func:`_hf_hub_cache_dir` (``HF_HUB_CACHE`` / ``HF_HOME`` / huggingface_hub).

    A pinned ``snapshot_download(revision=<sha>)`` leaves no ``refs/main`` (design 00 §2.3.11), so with no revision
    and no ``refs/main`` the single cached snapshot, if there is exactly one, is used."""
    cache = Path(cache_dir) if cache_dir is not None else _hf_hub_cache_dir()
    repo_dir = cache / ("models--" + repo_id.strip().replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    cands = sorted(p for p in snapshots.iterdir() if p.is_dir() and (p / "config.json").is_file())
    rev = (revision or "").strip() or "main"
    ref = repo_dir / "refs" / rev
    if ref.is_file():
        rev = ref.read_text().strip()
    match = [p for p in cands if p.name == rev or (len(rev) >= 7 and p.name.startswith(rev))]
    if not match and not revision and len(cands) == 1:
        match = cands
    return match[0] if len(match) == 1 else None


def resolve_weights_location(hf_config: Any = None, environ: Optional[Mapping[str, str]] = None) -> WeightsLocation:
    """The checkpoint location, first match wins:

    1. ``MOTIF3_WEIGHTS_DIR`` (must be a directory; a set but missing directory raises);
    2. ``HF_MODEL`` when it is a directory (TIS sets it to a symlinked snapshot dir);
    3. ``HF_MODEL`` as a repo id: its local HF-cache snapshot at ``TT_MODEL_WEIGHTS_REVISION``; if the snapshot is
       not cached, the repo id itself (``is_local=False``: the generator must download it at that revision);
    4. ``hf_config._name_or_path`` (vLLM's ``--model``) when it is a directory, or a repo id resolved as in 3;
    5. nothing: ``WeightsLocation(None, "default", ...)`` (``MotifTTConfig`` then uses its local default snapshot).

    ``HF_MODEL`` holding a path (``/...``, ``./...``) that does not exist raises: a typo must not silently fall
    through to another checkpoint."""
    env = os.environ if environ is None else environ
    revision = (env.get("TT_MODEL_WEIGHTS_REVISION") or "").strip() or None

    explicit = (env.get("MOTIF3_WEIGHTS_DIR") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_dir():
            raise ValueError(f"MOTIF3_WEIGHTS_DIR={explicit!r} is not a directory")
        return WeightsLocation(str(p), "MOTIF3_WEIGHTS_DIR", revision, True)

    def from_value(value: str, name: str) -> Optional[WeightsLocation]:
        p = Path(value).expanduser()
        if p.is_dir():
            return WeightsLocation(str(p), name, revision, True)
        if _looks_like_repo_id(value):
            snap = hf_cache_snapshot(value, revision, cache_dir=_hf_hub_cache_dir(env))
            if snap is not None:
                return WeightsLocation(
                    str(snap), f"{name} repo id {value}@{revision or 'main'} (HF cache)", revision, True
                )
            return WeightsLocation(
                value.strip(), f"{name} repo id {value}@{revision or 'main'} (not cached)", revision, False
            )
        return None

    hf_model = (env.get("HF_MODEL") or "").strip()
    if hf_model:
        loc = from_value(hf_model, "HF_MODEL")
        if loc is None:
            raise ValueError(f"HF_MODEL={hf_model!r} is neither a directory nor a 'org/name' HF repo id")
        return loc

    name = getattr(hf_config, "_name_or_path", None) or getattr(hf_config, "name_or_path", None)
    if name:
        loc = from_value(str(name), "hf_config._name_or_path")
        if loc is not None:
            return loc
    return WeightsLocation(None, "default", revision, False)


def resolve_tt_cache_path(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """TT weight-cache root: ``MOTIF3_TT_CACHE_PATH`` > ``TT_CACHE_PATH`` (TIS sets it under its host volume) > None
    (the generator's / ``MotifTTConfig``'s default ``motif-3/tt_cache``). The Motif-specific variable lets the
    converter and a TIS server share one converted cache without the symlink of TIS_RUNBOOK §2.4 (WAVE_A_REVIEW CONV-2;
    the cache path inside is ``<root>/<version-tag>/mesh<R>x<C>/...``)."""
    env = os.environ if environ is None else environ
    for name in ("MOTIF3_TT_CACHE_PATH", "TT_CACHE_PATH"):
        v = (env.get(name) or "").strip()
        if v:
            return v
    return None


# TT weight-cache policy of the model build (MOTIF3_TT_CACHE_POLICY -> GeneratorSettings.tt_cache_policy ->
# MotifModel(cache=...) in MotifGenerator.create; tt/model.py module docstring, README §7). Per part (globals, each
# decoder layer, the MTP layer): "auto" (default) loads a part a converter marked complete and never writes (an unmarked
# part loads from the checkpoint); "write" also converts an unmarked part, writes it under the cache root and marks it
# complete (disk guard first: tt/model.py MIN_FREE_GB), so the first start writes the cache and later starts load it;
# "off" never reads or writes the cache (every tensor from the checkpoint). tt/model.py CACHE_POLICIES lists the same.
TT_CACHE_POLICIES = ("auto", "write", "off")
DEFAULT_TT_CACHE_POLICY = "auto"


def check_tt_cache_policy(policy: str) -> str:
    """``policy`` if it is one of :data:`TT_CACHE_POLICIES`, else ``ValueError``."""
    if policy not in TT_CACHE_POLICIES:
        raise ValueError(
            f"the TT weight-cache policy (MOTIF3_TT_CACHE_POLICY) must be one of {TT_CACHE_POLICIES}, got {policy!r}"
        )
    return policy


def tt_cache_policy_from_env(environ: Optional[Mapping[str, str]] = None) -> str:
    """``MOTIF3_TT_CACHE_POLICY``: ``auto`` (default: complete parts load from the TT cache, nothing is written),
    ``write`` (parts that are not complete are converted and written under the cache root, then load from it on later
    starts) or ``off`` (no TT cache). See :data:`TT_CACHE_POLICIES`."""
    env = os.environ if environ is None else environ
    v = (env.get("MOTIF3_TT_CACHE_POLICY") or "").strip().lower() or DEFAULT_TT_CACHE_POLICY
    if v not in TT_CACHE_POLICIES:
        raise ValueError(f"MOTIF3_TT_CACHE_POLICY must be one of {TT_CACHE_POLICIES}, got {v!r}")
    return v


# ----------------------------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------------------------
def _env_int(environ: Mapping[str, str], name: str) -> Optional[int]:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    raw = raw.strip()
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{name} must be a positive decimal integer, got {raw!r}")
    return int(raw)


@dataclass(frozen=True)
class GeneratorSettings:
    """Serving-time choices the bridge hands to ``MotifGenerator.create`` (all validated).

    The integration wave maps them onto the TT config with ``MotifTTConfig.from_settings(settings,
    mesh_device=mesh_device, hf_config=hf_config)`` (``tt/model_config.py``): config from
    ``<weights_path>/config.json``, ``num_layers``, ``max_model_len = max_seq_len``, ``max_num_seqs =
    max_batch_size`` (the decode batch stays ``max_batch = NUM_LANES = 32``), the KV dtype, the block size when known,
    the fabric from the device. The KV pool geometry itself arrives later through ``allocate_kv_cache``.

    Fields:
        max_batch_size: vLLM ``max_num_seqs`` (1..32). The decode trace always runs ``NUM_LANES`` lanes; this
            only bounds how many lanes can be active (and sizes the plugin's per-sequence block reservation).
        max_seq_len: vLLM ``max_model_len`` (``MAX_MODEL_LEN_ALIGNMENT``..``MAX_CONTEXT``, a multiple of 256):
            largest prompt and largest decode position + 1.
        num_layers: decoder layers to run (``hf_config.num_hidden_layers``, or fewer for a truncated bring-up
            run via ``MOTIF3_NUM_LAYERS``; the final norm and LM head always run).
        kv_cache_dtype: ``"bfp8"`` (default) or ``"bf16"`` (``MOTIF3_KV_CACHE_DTYPE``; bf16 is the A/B lever of
            design 00 §7.1 risk 8 and needs a smaller pool).
        weights_path: the checkpoint per :func:`resolve_weights_location` -- ``MOTIF3_WEIGHTS_DIR`` > ``HF_MODEL``
            (directory) > the HF-cache snapshot of a repo-id ``HF_MODEL`` at ``TT_MODEL_WEIGHTS_REVISION`` >
            ``hf_config._name_or_path``. A local directory, except for an uncached repo id (``weights_source`` then
            says "not cached" and the generator must download it at ``weights_revision``); ``None`` = the
            generator's default snapshot.
        weights_revision: ``TT_MODEL_WEIGHTS_REVISION`` (pinned snapshot when ``HF_MODEL`` is a repo id).
        cache_path: TT weight-cache root (:func:`resolve_tt_cache_path`: ``MOTIF3_TT_CACHE_PATH`` > ``TT_CACHE_PATH``),
            or None for the generator's default.
        tt_cache_policy: how the model build uses the TT cache (``MOTIF3_TT_CACHE_POLICY``, :data:`TT_CACHE_POLICIES`):
            ``"auto"`` (default; never writes), ``"write"`` (converts and writes the parts that are not complete, disk
            guard first) or ``"off"``. ``MotifGenerator.create`` passes it to ``MotifModel(cache=...)`` unless its
            model kwargs carry ``cache``.
        optimizations: plugin ``tt.optimizations`` (None | "performance" | "accuracy"); draft 1 may ignore it.
        block_size: vLLM ``--block-size`` when the bridge could see it at model init (BRIDGE-4), else None. Only a
            hint for ``create``: ``allocate_kv_cache(block_size=...)`` is authoritative.
        weights_source: which precedence rule produced ``weights_path`` (logged by the bridge).

    Feature fields (docs/features/FEATURES_DESIGN.md §2.1; the defaults are draft 1). They describe what vLLM enabled
    (the bridge captures vLLM's scheduler config in ``get_max_tokens_all_users``) plus the Motif environment knobs:
        chunked_prefill: vLLM ``enable_chunked_prefill`` (after the plugin's policy). Rows may start at any position.
        prefix_caching: vLLM ``enable_prefix_caching``. Rows may start at a block multiple and share read-only blocks.
        max_num_batched_tokens: vLLM's per-step token budget (None = unknown / chunking off).
        long_prefill_token_threshold: vLLM's per-request chunk cap (0 = none).
        spec_tokens: 0, or 1 = MTP self-speculation with K = 1 (``num_speculative_tokens``).
        kv_replicated_decode: KV-R (every decode KV write on all 32 chips). None = ``auto`` = on iff prefix caching
            (``MOTIF3_KV_REPLICATED_DECODE``); False with prefix caching on is refused (stale cross-row KV, §3.4).
        prefill_span_cap: largest prefill bucket (``MOTIF3_PREFILL_MAX_BUCKET``); None = the generator's default
            (:meth:`resolved_prefill_span_cap`).
        packed_prefill: packed multi-row prefill (``MOTIF3_PACKED_PREFILL``, default off; P5,
            docs/p5_t64/P5_T64_DESIGN.md §3); a generator without it ignores the flag (rows run one after another).
        spec_verify: the verify mode of a speculating launch (``MOTIF3_SPEC_VERIFY``, :data:`SPEC_VERIFY_MODES`):
            ``"packed"`` (drafts on idle lanes of the 32-lane trace), ``"wide"`` (the 64-row trace alone) or ``"auto"``
            (both traces, the 64-row one for verify steps whose drafts do not fit idle lanes). A generator without the
            64-row trace must refuse ``"wide"`` and ``"auto"`` in ``create``. Without speculation it has no effect.

    Packed-prefill knobs (P5, §3.3-§3.5, §3.9; they matter only with ``packed_prefill``):
        packed_prefill_max_seg: the largest packed segment S (``MOTIF3_PACKED_PREFILL_MAX_SEG``, one of
            :data:`PACK_SEG_BUCKETS`, default 1024); a chunk of more rows runs solo.
        packed_prefill_max_tokens: the largest packed pass ``T = B * S`` (``MOTIF3_PACKED_PREFILL_MAX_TOKENS``, a
            power of two, default 8192; the effective cap is also bounded by the prefill span cap).
        packed_prefill_pk1: also pack resumed (sp1) chunks that share a start (``MOTIF3_PACKED_PREFILL_PK1``, default
            on); off, they run solo.
        packed_warmup: ``"attention"`` (default; attention-only warm calls per packed shape) or ``"full"`` (one full
            packed pass per shape) (``MOTIF3_PACKED_WARMUP``, :data:`PACKED_WARMUP_MODES`).

    T64 drafting knobs (§4.5, §4.7; they matter only with ``spec_verify`` ``"auto"``):
        wide_min_lanes: the live-lane count from which every live lane drafts (``MOTIF3_WIDE_MIN_LANES``, ``[1, 33]``,
            33 = never); None (default) = the generator's ``c*`` from the acceptance and the T64 / T32 step ratio.
        spec_alpha_prior: the acceptance prior ``alpha_0`` (default 0.85) of :func:`smoothed_acceptance`; also the
            acceptance :meth:`MotifGenerator.drafts_all_lanes` assumes before any draft was verified.
        wide_step_ratio: the T64 / T32-spec step ratio ``r`` of ``c*`` (``MOTIF3_WIDE_STEP_RATIO``, a finite float
            >= 1); None (default) keeps ``MotifTTConfig.wide_step_ratio`` at ``model_config.DEFAULT_WIDE_STEP_RATIO``
            (1.21). ``MotifTTConfig.from_settings`` maps it.
    """

    max_batch_size: int = NUM_LANES
    max_seq_len: int = MAX_CONTEXT
    num_layers: int = NUM_HIDDEN_LAYERS
    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE
    weights_path: Optional[str] = None
    weights_revision: Optional[str] = None
    cache_path: Optional[str] = None
    tt_cache_policy: str = DEFAULT_TT_CACHE_POLICY
    optimizations: Optional[str] = None
    block_size: Optional[int] = None
    weights_source: Optional[str] = None
    # ---- features (all validated; defaults = draft 1) --------------------------------------------------------
    chunked_prefill: bool = False
    prefix_caching: bool = False
    max_num_batched_tokens: Optional[int] = None
    long_prefill_token_threshold: int = 0
    spec_tokens: int = 0
    kv_replicated_decode: Optional[bool] = None
    prefill_span_cap: Optional[int] = None
    packed_prefill: bool = False
    spec_verify: str = "packed"
    # ---- packed prefill (P5) and T64 verify knobs (docs/p5_t64/P5_T64_DESIGN.md §3.9, §4.5-§4.7) -----------------
    packed_prefill_max_seg: int = DEFAULT_PACKED_PREFILL_MAX_SEG
    packed_prefill_max_tokens: int = DEFAULT_PACKED_PREFILL_MAX_TOKENS
    packed_prefill_pk1: bool = True
    packed_warmup: str = DEFAULT_PACKED_WARMUP
    wide_min_lanes: Optional[int] = None
    spec_alpha_prior: float = DEFAULT_SPEC_ALPHA_PRIOR
    wide_step_ratio: Optional[float] = None

    def __post_init__(self):
        if not 1 <= int(self.max_batch_size) <= NUM_LANES:
            raise ValueError(f"max_batch_size must be in [1, {NUM_LANES}], got {self.max_batch_size}")
        if not 1 <= int(self.max_seq_len) <= MAX_CONTEXT:
            raise ValueError(f"max_seq_len must be in [1, {MAX_CONTEXT}] in draft 1, got {self.max_seq_len}")
        check_max_model_len(self.max_seq_len)
        if int(self.num_layers) < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.kv_cache_dtype not in KV_CACHE_DTYPES:
            raise ValueError(f"kv_cache_dtype must be one of {KV_CACHE_DTYPES}, got {self.kv_cache_dtype!r}")
        check_tt_cache_policy(self.tt_cache_policy)
        if self.optimizations not in (None, "performance", "accuracy"):
            raise ValueError(f"optimizations must be None, 'performance' or 'accuracy', got {self.optimizations!r}")
        if self.block_size is not None:
            check_block_size(self.block_size)
        for name in ("chunked_prefill", "prefix_caching", "packed_prefill", "packed_prefill_pk1"):
            if getattr(self, name) not in (True, False):
                raise TypeError(f"{name} must be a bool, got {getattr(self, name)!r}")
        if self.kv_replicated_decode not in (None, True, False):
            raise TypeError(f"kv_replicated_decode must be None (auto) or a bool, got {self.kv_replicated_decode!r}")
        if self.max_num_batched_tokens is not None and int(self.max_num_batched_tokens) < 1:
            raise ValueError(f"max_num_batched_tokens must be >= 1, got {self.max_num_batched_tokens}")
        if int(self.long_prefill_token_threshold) < 0:
            raise ValueError(f"long_prefill_token_threshold must be >= 0, got {self.long_prefill_token_threshold}")
        if int(self.spec_tokens) not in SUPPORTED_SPEC_TOKENS:
            raise ValueError(
                f"spec_tokens must be one of {SUPPORTED_SPEC_TOKENS} (Motif-3 MTP drafts K = 1), got {self.spec_tokens}"
            )
        if self.prefix_caching and self.kv_replicated_decode is False:
            raise ValueError(
                "prefix caching needs KV-R (MOTIF3_KV_REPLICATED_DECODE=0 refused): decode writes a lane's KV only on "
                "its DP row, so a prefix hit on a block another row decode-wrote would read stale KV, and the MoE "
                "reduce-scatter spreads the error to every row (docs/features/FEATURES_DESIGN.md §3.4)"
            )
        if self.prefill_span_cap is not None:
            check_prefill_span_cap(self.prefill_span_cap, self.max_seq_len)
        if self.spec_verify not in SPEC_VERIFY_MODES:
            raise ValueError(f"spec_verify must be one of {SPEC_VERIFY_MODES}, got {self.spec_verify!r}")
        check_packed_prefill_max_seg(self.packed_prefill_max_seg)
        check_packed_prefill_max_tokens(self.packed_prefill_max_tokens)
        if self.packed_warmup not in PACKED_WARMUP_MODES:
            raise ValueError(f"packed_warmup must be one of {PACKED_WARMUP_MODES}, got {self.packed_warmup!r}")
        if self.wide_min_lanes is not None:
            check_wide_min_lanes(self.wide_min_lanes)
        check_spec_alpha_prior(self.spec_alpha_prior)
        if self.wide_step_ratio is not None:
            check_wide_step_ratio(self.wide_step_ratio)

    @property
    def weights_are_local(self) -> bool:
        """``weights_path`` is a local checkpoint directory (False for an uncached repo id or None)."""
        return self.weights_path is not None and Path(self.weights_path).is_dir()

    # ---- resolved feature facts -------------------------------------------------------------------------------
    @property
    def resumed_prefill(self) -> bool:
        """vLLM may send prefill rows with ``start > 0`` (chunked prefill or prefix caching is on)."""
        return bool(self.chunked_prefill or self.prefix_caching)

    @property
    def spec_decode(self) -> bool:
        """MTP self-speculative decoding is on (``spec_tokens > 0``)."""
        return int(self.spec_tokens) > 0

    @property
    def kv_replicated(self) -> bool:
        """KV-R resolved: ``kv_replicated_decode``, or ``prefix_caching`` when it is None (``auto``)."""
        return bool(self.prefix_caching) if self.kv_replicated_decode is None else bool(self.kv_replicated_decode)

    @property
    def kv_write_mode(self) -> str:
        """:func:`kv_write_mode` of these settings: ``row`` | ``row_split`` | ``all`` | ``all_split``."""
        return kv_write_mode(self.kv_replicated, self.spec_decode)

    @property
    def mtp_kv_layers(self) -> int:
        """Extra latent caches the generator allocates next to the ``num_layers`` main ones: 1 with speculation."""
        return 1 if self.spec_decode else 0

    def resolved_prefill_span_cap(self, supports_resumed_prefill: bool) -> int:
        """The span cap a generator uses: ``min(prefill_span_cap or DEFAULT_PREFILL_SPAN_CAP, max_seq_len)`` when it
        supports resumed prefill (spans above it are split internally, design D8), else ``max_seq_len`` (draft 1:
        one bucket per prompt). Raises when a cap below ``max_seq_len`` was asked of a generator that cannot split."""
        L = int(self.max_seq_len)
        if supports_resumed_prefill:
            cap = DEFAULT_PREFILL_SPAN_CAP if self.prefill_span_cap is None else int(self.prefill_span_cap)
            return min(cap, L)
        if self.prefill_span_cap is not None and min(int(self.prefill_span_cap), L) < L:
            raise ValueError(
                f"prefill_span_cap {self.prefill_span_cap} < max_seq_len {L} needs a generator with resumed prefill "
                f"(spans are split into sp0 + sp1 chunks)"
            )
        return L

    @classmethod
    def from_env(
        cls,
        hf_config: Any,
        *,
        max_batch_size: int,
        max_seq_len: int,
        optimizations: Optional[str] = None,
        block_size: Optional[int] = None,
        environ: Optional[Mapping[str, str]] = None,
        serving: Optional[Mapping[str, Any]] = None,
    ) -> "GeneratorSettings":
        """Resolve settings from the vLLM arguments plus the documented environment variables.

        ``serving`` is the vLLM scheduler config the bridge captured (keys :data:`SERVING_KEYS`, all optional; an
        unknown key raises): ``block_size`` (used when ``block_size`` is None), ``enable_chunked_prefill``,
        ``max_num_batched_tokens``, ``long_prefill_token_threshold``, ``enable_prefix_caching``,
        ``prefix_match_unit`` (checked by ``prefill_plan.check_scheduler_config``, not stored) and ``spec_tokens``
        (vLLM ``num_speculative_tokens`` after the platform published ``effective_k``). None = draft 1. Environment:
        ``MOTIF3_TT_CACHE_POLICY``, ``MOTIF3_KV_REPLICATED_DECODE``, ``MOTIF3_PREFILL_MAX_BUCKET``,
        ``MOTIF3_PACKED_PREFILL``, ``MOTIF3_PACKED_PREFILL_MAX_SEG``, ``MOTIF3_PACKED_PREFILL_MAX_TOKENS``,
        ``MOTIF3_PACKED_PREFILL_PK1``, ``MOTIF3_PACKED_WARMUP``, ``MOTIF3_SPEC_VERIFY``, ``MOTIF3_WIDE_MIN_LANES``,
        ``MOTIF3_WIDE_STEP_RATIO`` (``spec_alpha_prior`` keeps its default)."""
        env = os.environ if environ is None else environ
        hf_layers = int(getattr(hf_config, "num_hidden_layers", NUM_HIDDEN_LAYERS))
        env_layers = _env_int(env, "MOTIF3_NUM_LAYERS")
        num_layers = hf_layers if env_layers is None else env_layers
        if not 1 <= num_layers <= hf_layers:
            raise ValueError(f"MOTIF3_NUM_LAYERS={num_layers} outside [1, {hf_layers}]")
        kv_dtype = (env.get("MOTIF3_KV_CACHE_DTYPE") or DEFAULT_KV_CACHE_DTYPE).strip().lower()
        loc = resolve_weights_location(hf_config, env)
        sv = dict(serving or {})
        unknown = sorted(set(sv) - set(SERVING_KEYS))
        if unknown:
            raise ValueError(f"unknown serving config keys {unknown}; known: {SERVING_KEYS}")
        if block_size is None and sv.get("block_size") is not None:
            block_size = int(sv["block_size"])

        def opt_int(key: str) -> Optional[int]:
            return None if sv.get(key) is None else int(sv[key])

        return cls(
            max_batch_size=int(max_batch_size),
            max_seq_len=int(max_seq_len),
            num_layers=num_layers,
            kv_cache_dtype=kv_dtype,
            weights_path=loc.path,
            weights_revision=loc.revision,
            cache_path=resolve_tt_cache_path(env),
            tt_cache_policy=tt_cache_policy_from_env(env),
            optimizations=optimizations,
            block_size=None if block_size is None else int(block_size),
            weights_source=loc.source,
            chunked_prefill=bool(sv.get("enable_chunked_prefill") or False),
            prefix_caching=bool(sv.get("enable_prefix_caching") or False),
            max_num_batched_tokens=opt_int("max_num_batched_tokens"),
            long_prefill_token_threshold=int(sv.get("long_prefill_token_threshold") or 0),
            spec_tokens=int(sv.get("spec_tokens") or 0),
            kv_replicated_decode=kv_replicated_decode_from_env(env),
            prefill_span_cap=prefill_span_cap_from_env(env),
            packed_prefill=packed_prefill_from_env(env),
            spec_verify=spec_verify_from_env(env),
            packed_prefill_max_seg=packed_prefill_max_seg_from_env(env),
            packed_prefill_max_tokens=packed_prefill_max_tokens_from_env(env),
            packed_prefill_pk1=packed_prefill_pk1_from_env(env),
            packed_warmup=packed_warmup_from_env(env),
            wide_min_lanes=wide_min_lanes_from_env(env),
            wide_step_ratio=wide_step_ratio_from_env(env),
        )


def kv_cache_dtype_from_env(environ: Optional[Mapping[str, str]] = None) -> str:
    """``MOTIF3_KV_CACHE_DTYPE`` (default ``"bfp8"``), validated."""
    env = os.environ if environ is None else environ
    value = (env.get("MOTIF3_KV_CACHE_DTYPE") or DEFAULT_KV_CACHE_DTYPE).strip().lower()
    if value not in KV_CACHE_DTYPES:
        raise ValueError(f"MOTIF3_KV_CACHE_DTYPE must be one of {KV_CACHE_DTYPES}, got {value!r}")
    return value


# ----------------------------------------------------------------------------------------------------------------
# Per-call inputs
# ----------------------------------------------------------------------------------------------------------------
def _check_int32(name: str, t: torch.Tensor, ndim: int) -> None:
    if not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or t.ndim != ndim or t.device.type != "cpu":
        raise TypeError(
            f"{name} must be a CPU torch.int32 tensor with {ndim} dim(s), got "
            f"{type(t).__name__} {getattr(t, 'dtype', None)} shape {tuple(getattr(t, 'shape', ()))}"
        )


@dataclass(frozen=True)
class PrefillRequest:
    """One prefill row: a new prompt, a chunk of a long prompt, or the uncached rest after a prefix-cache hit.

    Attributes:
        lane: destination lane in ``[0, NUM_LANES)``. Its DP group (``lane // 8``) runs every later decode step of
            this request. Prefill writes the latent on every chip, so the lane only matters to lane-owned state (none:
            the paged cache is the only state, and a chunked request may change lane between chunks).
        tokens: ``torch.int32 [end]``, ALL the request's tokens at positions ``0 .. end-1`` (``1 <= end <=
            max_seq_len``), unpadded: the cached prefix, earlier chunks and this chunk's new tokens. For a request
            resumed after preemption this is prompt + every generated token.
        page_table: ``torch.int32 [W]``, the request's vLLM block ids in position order (W = the server's
            page-table width, ``min(ceil(max_seq_len / block_size), num_blocks)``). Entries
            ``0 .. ceil(end / block_size) - 1`` are real blocks (>= 1); the rest are 0 (null block). The bridge zeroes
            that tail itself: vLLM's persistent block-table rows keep stale ids there, often of blocks other live
            requests own now. Never write through a page-table entry the bridge did not hand over. The same block id
            may appear in several rows of one call (a shared prefix), never twice in one row.
        start: positions ``[0, start)`` are already in the cache (vLLM ``num_computed_tokens``: a prefix-cache hit,
            a multiple of the block size, or the end of the previous chunk, any integer); ``0 <= start < end``.

    Contract (features design §2.1): full blocks below ``floor(start / block_size)`` are READ-ONLY (they may be
    cached and shared with other requests); the generator writes positions ``[floor(start / bs) * bs, end)`` and may
    write bucket padding only into the request's own last block (decode overwrites it before reading) or the null
    block 0 (never read: only dropped padding rows read it) -- never into another block. The logits returned are those
    of position ``end - 1`` (also for an intermediate chunk: the plugin does not say which chunk is the last).
    ``start = 0`` is draft 1.
    """

    lane: int
    tokens: torch.Tensor
    page_table: torch.Tensor
    start: int = 0

    def __post_init__(self):
        if not 0 <= int(self.lane) < NUM_LANES:
            raise ValueError(f"lane must be in [0, {NUM_LANES}), got {self.lane}")
        _check_int32("tokens", self.tokens, 1)
        _check_int32("page_table", self.page_table, 1)
        if self.tokens.shape[0] < 1:
            raise ValueError("empty prompt")
        if not 0 <= int(self.start) < int(self.tokens.shape[0]):
            raise ValueError(
                f"start must be in [0, end={int(self.tokens.shape[0])}) (positions [0, start) are cached and at least "
                f"one position is computed), got {self.start}"
            )

    @property
    def seq_len(self) -> int:
        """``end``: one past the last position of this row (the logits are for ``end - 1``)."""
        return int(self.tokens.shape[0])

    @property
    def end(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def resumed(self) -> bool:
        """``start > 0``: the row reads a cached prefix (or recomputes part of it)."""
        return int(self.start) > 0

    @property
    def num_new_tokens(self) -> int:
        """Positions vLLM scheduled in this row: ``end - start``."""
        return self.end - int(self.start)


@dataclass(frozen=True)
class DecodeBatch:
    """One decode step for all ``NUM_LANES`` lanes, in lane order.

    Attributes:
        tokens: ``torch.int32 [NUM_LANES]``; the input token of each lane (0 for inactive lanes).
        positions: ``torch.int32 [NUM_LANES]``; absolute position of that token (= its KV write slot), ``-1`` for
            an inactive lane. Active positions are ``< max_seq_len``.
        page_table: ``torch.int32 [NUM_LANES, W]``; each lane's vLLM block ids for positions ``0 .. p``, zero after
            entry ``p // block_size`` (stale vLLM ids are zeroed by the bridge) and all zero for inactive lanes. W is
            fixed for the life of the server (the width ``warmup_decode`` was given), so it can be a trace input.
    """

    tokens: torch.Tensor
    positions: torch.Tensor
    page_table: torch.Tensor

    def __post_init__(self):
        _check_int32("tokens", self.tokens, 1)
        _check_int32("positions", self.positions, 1)
        _check_int32("page_table", self.page_table, 2)
        if self.tokens.shape[0] != NUM_LANES or self.positions.shape[0] != NUM_LANES:
            raise ValueError(f"decode inputs must have {NUM_LANES} lanes")
        if self.page_table.shape[0] != NUM_LANES:
            raise ValueError(f"page_table must have {NUM_LANES} rows, got {tuple(self.page_table.shape)}")

    @property
    def active(self) -> torch.Tensor:
        """``torch.bool [NUM_LANES]``: lanes that carry a request this step."""
        return self.positions >= 0

    @property
    def page_table_width(self) -> int:
        return int(self.page_table.shape[1])


@dataclass(frozen=True)
class SpecDecodeBatch:
    """One decode step of a speculating launch (``spec_tokens = 1``), all ``NUM_LANES`` lanes in OWNER-lane order
    (lane = the request's own lane, as in :class:`DecodeBatch`). Ordinary steps carry no draft; verify steps carry at
    most one draft per lane.

    Attributes:
        tokens: ``torch.int32 [NUM_LANES]``: the anchor token (the last committed token) of each lane; 0 for inactive
            lanes.
        positions: ``torch.int32 [NUM_LANES]``: the anchor's position ``n`` (its KV write slot); ``-1`` = inactive.
            Active positions are ``< max_seq_len``, and ``< max_seq_len - 1`` on drafted lanes.
        draft_tokens: ``torch.int32 [NUM_LANES]``: the draft for position ``n + 1``; ``-1`` = no draft (always on an
            ordinary step and on inactive lanes).
        page_table: ``torch.int32 [NUM_LANES, W]``: each lane's block ids for positions ``0 .. n`` (``0 .. n + 1``
            on drafted lanes), zero after; all zero for inactive lanes. W is the decode trace's fixed width.

    Idle lanes (``positions == -1``) belong to nobody this step: the generator may borrow them to run drafts (packed
    verify, design §3.8.2).
    """

    tokens: torch.Tensor
    positions: torch.Tensor
    draft_tokens: torch.Tensor
    page_table: torch.Tensor

    def __post_init__(self):
        _check_int32("tokens", self.tokens, 1)
        _check_int32("positions", self.positions, 1)
        _check_int32("draft_tokens", self.draft_tokens, 1)
        _check_int32("page_table", self.page_table, 2)
        for name in ("tokens", "positions", "draft_tokens"):
            if getattr(self, name).shape[0] != NUM_LANES:
                raise ValueError(
                    f"spec decode {name} must have {NUM_LANES} lanes, got {tuple(getattr(self, name).shape)}"
                )
        if self.page_table.shape[0] != NUM_LANES:
            raise ValueError(f"page_table must have {NUM_LANES} rows, got {tuple(self.page_table.shape)}")
        if bool((self.positions < -1).any()):
            raise ValueError("positions must be -1 (inactive) or >= 0")
        if bool((self.draft_tokens < -1).any()):
            raise ValueError("draft_tokens must be -1 (no draft) or a token id >= 0")
        if bool(((self.draft_tokens >= 0) & (self.positions < 0)).any()):
            raise ValueError("a draft on an inactive lane (draft_tokens >= 0 where positions == -1)")

    @property
    def active(self) -> torch.Tensor:
        """``torch.bool [NUM_LANES]``: lanes that carry a request this step (their owners)."""
        return self.positions >= 0

    @property
    def has_draft(self) -> torch.Tensor:
        """``torch.bool [NUM_LANES]``: owner lanes whose draft must be verified this step."""
        return self.draft_tokens >= 0

    @property
    def num_drafts(self) -> int:
        return int(self.has_draft.sum())

    @property
    def is_verify(self) -> bool:
        """At least one draft: the generator returns ``a1`` / ``m1`` for the drafted lanes."""
        return self.num_drafts > 0

    @property
    def idle_lanes(self) -> Tuple[int, ...]:
        """Lanes no request uses this step (``positions == -1``), ascending: the packed verify's partner lanes."""
        return tuple(int(i) for i in torch.nonzero(self.positions < 0).reshape(-1).tolist())

    @property
    def page_table_width(self) -> int:
        return int(self.page_table.shape[1])

    @classmethod
    def from_decode_batch(cls, batch: "DecodeBatch") -> "SpecDecodeBatch":
        """The ordinary step (no drafts) of a speculating launch for ``batch``."""
        return cls(
            tokens=batch.tokens,
            positions=batch.positions,
            draft_tokens=torch.full((NUM_LANES,), -1, dtype=torch.int32),
            page_table=batch.page_table,
        )

    def anchors(self) -> "DecodeBatch":
        """The anchor rows alone as a :class:`DecodeBatch` (drafts dropped)."""
        return DecodeBatch(tokens=self.tokens, positions=self.positions, page_table=self.page_table)


@dataclass(frozen=True)
class SpecDecodeResult:
    """What :meth:`MotifGenerator.decode_forward_spec` returns, in OWNER-lane order (fresh host tensors).

    Attributes:
        logits: ``[NUM_LANES, vocab]`` float32 / bfloat16 host logits of the anchor rows (the next-token logits at
            position ``n``, exactly what ``decode_forward`` returns) when ``want_logits``, else None (a verify step
            needs only the ids and skips the logits read).
        argmax: ``torch.int32 [NUM_LANES, 2]``: column 0 = ``a0``, the target argmax at the anchor (the token at
            ``n + 1``); column 1 = ``a1``, the target argmax at the draft row (the token at ``n + 2`` if the draft is
            accepted). Lowest-index tie rule (vLLM's host greedy).
        mtp_argmax: ``torch.int32 [NUM_LANES, 2]``: the MTP layer's predictions. Column 0 = ``m0`` from the anchor
            row (MTP input ``(hn_n, a0)`` at ``n``: a draft for ``n + 2``); column 1 = ``m1`` from the draft row
            (``(hn_{n+1}, a1)`` at ``n + 1``: a draft for ``n + 3``).

    Column 1 of both is meaningful only on drafted lanes; inactive lanes' rows are unspecified (ignore them).
    """

    logits: Optional[torch.Tensor]
    argmax: torch.Tensor
    mtp_argmax: torch.Tensor

    def __post_init__(self):
        for name in ("argmax", "mtp_argmax"):
            t = getattr(self, name)
            _check_int32(name, t, 2)
            if tuple(t.shape) != (NUM_LANES, 2):
                raise ValueError(f"{name} must be [{NUM_LANES}, 2], got {tuple(t.shape)}")
        if self.logits is not None:
            lg = self.logits
            if not isinstance(lg, torch.Tensor) or lg.device.type != "cpu" or not lg.is_floating_point():
                raise TypeError(f"logits must be None or CPU floating-point, got {type(lg).__name__}")
            if lg.ndim != 2 or lg.shape[0] != NUM_LANES:
                raise ValueError(f"logits must be [{NUM_LANES}, vocab], got {tuple(lg.shape)}")


def check_spec_result(name: str, result: Any, *, want_logits: bool, vocab_size: int) -> SpecDecodeResult:
    """Validate :meth:`MotifGenerator.decode_forward_spec`'s output: a :class:`SpecDecodeResult`, logits
    ``[NUM_LANES, vocab_size]`` exactly when ``want_logits`` (None otherwise: the logits read costs ~2.7 ms)."""
    if not isinstance(result, SpecDecodeResult):
        raise TypeError(f"{name} must return a SpecDecodeResult, got {type(result).__name__}")
    if want_logits:
        if result.logits is None:
            raise ValueError(f"{name}: want_logits=True but no logits were returned")
        check_logits(name, result.logits, (NUM_LANES, int(vocab_size)))
    elif result.logits is not None:
        raise ValueError(f"{name}: want_logits=False but logits were returned (skip the ~2.7 ms logits read)")
    return result


def check_prefill_batch(requests: Sequence[Any]) -> Tuple[PrefillRequest, ...]:
    """The rows of one ``prefill_forward_batch`` call: 1 .. ``NUM_LANES`` :class:`PrefillRequest` on distinct lanes."""
    reqs = tuple(requests)
    if not reqs:
        raise ValueError("prefill_forward_batch got no rows")
    if len(reqs) > NUM_LANES:
        raise ValueError(f"prefill_forward_batch got {len(reqs)} rows; at most {NUM_LANES}")
    for i, r in enumerate(reqs):
        if not isinstance(r, PrefillRequest):
            raise TypeError(f"row {i} must be a PrefillRequest, got {type(r).__name__}")
    lanes = [int(r.lane) for r in reqs]
    if len(set(lanes)) != len(lanes):
        raise ValueError(f"prefill rows must use distinct lanes, got {lanes}")
    return reqs


# ----------------------------------------------------------------------------------------------------------------
# The runtime interface
# ----------------------------------------------------------------------------------------------------------------
class MotifGenerator(abc.ABC):
    """What the integration wave implements in ``tt/generator.py`` (default class path
    ``models.demos.motif3.tt.generator:MotifGenerator``, overridable with ``MOTIF3_GENERATOR_CLASS``).

    Call order under vLLM (vllm-tt-plugin ``worker.py:221-469``, ``model_runner.py:678-708, 3727-3781``):

    1. ``create(hf_config=..., mesh_device=..., settings=...)`` once, after the plugin opened the mesh
       (``initialize_vllm_model``). Loads/converts weights; must not allocate the KV pool. The bridge then calls
       :func:`check_generator_features` (a feature vLLM enabled must be supported).
    2. ``allocate_kv_cache(num_blocks=, block_size=, num_layers=)`` once.
    3. ``warmup_prefill(enable_trace=False)`` -> ``warmup_decode(enable_trace=False, page_table_width=W)`` ->
       [``warmup_prefill(enable_trace=True)`` only with plugin ``trace_mode="all"``] ->
       ``warmup_decode(enable_trace=True, page_table_width=W)`` (decode trace capture). Warmup is skipped when
       the plugin runs with ``enable_model_warmup=false`` (bring-up).
    4. Serving: any interleaving of prefill steps and decode steps; ``release_lane`` when a request on that lane
       finished or was preempted.
       * prefill: ONE ``prefill_forward_batch`` call per plugin step with all of the step's rows (new, resumed and
         chunk-continuation rows; at most ``NUM_LANES`` rows and ``max_num_batched_tokens`` new tokens). Draft-1
         bridges called ``prefill_forward`` once per row instead; that remains valid for ``start = 0`` rows.
       * decode: ``decode_forward`` (non-speculating launch), or ``decode_forward_spec`` for EVERY decode step of a
         speculating launch (``settings.spec_tokens = 1``; one decode trace serves ordinary and verify steps, two
         with ``settings.spec_verify="auto"``). Before proposing the next step's drafts the bridge asks
         ``drafts_all_lanes``.
    5. ``release_traces()`` at shutdown while the mesh is still open (the plugin closes the mesh afterwards).

    Concurrency: calls are strictly sequential (one EngineCore thread). Every method may raise; a raise must leave
    no partially-applied host state behind (the bridge commits its own lane bookkeeping only after success).

    Feature capabilities (defaults keep draft-1 generators working): ``supports_resumed_prefill``,
    ``prefill_alignment``, ``max_prefill_span``, ``supports_spec_decode``; the T64 drafting answer
    ``drafts_all_lanes`` (default False: the bridge's idle-lane draft budget).
    """

    # ---- construction -----------------------------------------------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def create(cls, *, hf_config: Any, mesh_device: Any, settings: GeneratorSettings) -> "MotifGenerator":
        """Build the runtime on an already-open mesh.

        Build the TT config with ``MotifTTConfig.from_settings(settings, mesh_device=mesh_device,
        hf_config=hf_config)``: it reads ``<settings.weights_path>/config.json`` (+ ``generation_config.json`` for
        the EOS set) when the weights are local, else the ``hf_config`` object (safe since INFRA-1: ``rope_scaling``
        or transformers-5 ``rope_parameters``, EOS from the generation config next to ``_name_or_path``), keeps the
        decode batch at ``NUM_LANES`` whatever ``max_batch_size`` is, and takes the fabric from the device.

        Args:
            hf_config: vLLM's ``model_config.hf_config``: the trust-remote-code ``MotifConfig`` instance (dynamic
                class; duck-type it, never ``isinstance``). ``hf_config.architectures`` has already been rewritten
                to ``["TTMotifForCausalLM"]`` by the plugin.
            mesh_device: the ``ttnn.MeshDevice`` the plugin opened from ``MESH_DEVICE``: shape (4, 8) or (8, 4)
                (the TP axis is the size-8 dim), fabric ``FABRIC_2D_TORUS_XY`` by default, with an L1_SMALL region of
                at least :data:`L1_SMALL_SIZE` bytes per core (``"l1_small_size"`` in the plugin's ``"tt"`` config;
                the bridge refuses a mesh without it before calling ``create``). Standalone runs open the mesh with
                ``model_config.open_motif_mesh()`` (same fabric, dispatch axis, trace region and L1_SMALL); a
                generator may assert the region with ``model_config.require_l1_small(mesh_device)``.
            settings: validated ``GeneratorSettings``.
        """

    # ---- static facts the bridge validates against ----------------------------------------------------------
    @property
    @abc.abstractmethod
    def num_layers(self) -> int:
        """Decoder layers this generator runs (``settings.num_layers``) = KV caches it allocates."""

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Width of every logits row (``hf_config.vocab_size``, 220160 for Motif-3)."""

    @property
    def num_lanes(self) -> int:
        """Lanes of every decode step; must be ``NUM_LANES``."""
        return NUM_LANES

    @property
    def max_prefill_len(self) -> int:
        """Longest row (``end``) ``prefill_forward`` / ``prefill_forward_batch`` accept; >= ``settings.max_seq_len``.
        Draft 1: its largest bucket. With internal chunking (``max_prefill_span < max_prefill_len``) longer rows are
        split into chunks of at most ``max_prefill_span`` rows."""
        return MAX_CONTEXT

    # ---- feature capabilities (docs/features/FEATURES_DESIGN.md §2.1) ------------------------------------------
    @property
    def supports_resumed_prefill(self) -> bool:
        """``prefill_forward_batch`` accepts rows with ``start > 0`` (chunked prefill, prefix caching). Required
        whenever ``settings.chunked_prefill`` or ``settings.prefix_caching`` is set."""
        return False

    @property
    def prefill_alignment(self) -> int:
        """Resume alignment ``A`` = ``lcm(block_size, q_chunk, k_chunk)`` of the resumed (sp1) programs: every chunk
        the generator runs starts at a multiple of it, so a row resumed at ``start`` recomputes up to ``A - 1``
        cached positions (``prefill_plan``). 0 = no resumed prefill. The bridge checks vLLM's chunk budget against it.
        """
        return 0

    @property
    def max_prefill_span(self) -> int:
        """The span cap: the largest prefill bucket this generator compiles. Rows longer than it (after the
        alignment floor) are split into several chunks inside one call (design D8)."""
        return self.max_prefill_len

    @property
    def supports_spec_decode(self) -> bool:
        """``decode_forward_spec`` is implemented (MTP self-speculation, K = 1). Required when
        ``settings.spec_tokens > 0``."""
        return False

    def drafts_all_lanes(self, live_lanes: Sequence[int], acceptance: Optional[float] = None) -> bool:
        """Whether the bridge may propose a draft for EVERY live lane of the next decode step (T64 verify,
        docs/p5_t64/P5_T64_DESIGN.md §4.5, §4.7; review edits R-E3, R-E9). Host only: no device work, no state change.

        Args:
            live_lanes: the lanes that carry a request in the next step (lane ids in ``[0, NUM_LANES)``, as the bridge's
                draft budget sees them; duplicates count once).
            acceptance: the bridge's acceptance estimate (:func:`smoothed_acceptance`, prior
                ``settings.spec_alpha_prior``), or None before any draft was verified (then the generator assumes
                ``settings.spec_alpha_prior``).

        Returns:
            True: every live lane may draft; the generator verifies drafts that do not fit idle lanes in one 64-row step
            (``spec_verify`` "wide" / "auto"). False: the bridge keeps its idle-lane budget (with KV-R ``32 - live``
            drafts in all, without it ``8 - active`` per DP row), so every draft fits the 32-lane trace. This default
            (no 64-row trace) and ``spec_verify="packed"`` answer False. A generator with the 64-row trace answers True
            in ``"wide"`` and, in ``"auto"``, True iff the live lanes reach ``settings.wide_min_lanes`` when set, else
            ``c*`` = ``verify_plan.crossover_lanes(acceptance, cfg.wide_step_ratio)`` (17..33, 33 = never).
        """
        return False

    # ---- KV pool ----------------------------------------------------------------------------------------------
    @abc.abstractmethod
    def allocate_kv_cache(self, *, num_blocks: int, block_size: int, num_layers: int) -> Any:
        """Allocate the paged latent pool and return an opaque handle.

        Allocates ``num_layers`` (== ``self.num_layers``) tensors, each of logical shape
        ``[num_blocks, 1, block_size, KV_LATENT_DIM]``, dtype per ``settings.kv_cache_dtype`` (``ttnn.bfloat8_b`` or
        ``ttnn.bfloat16``), TILE layout, DRAM, replicated on every chip of the mesh, zero-filled. Block 0 is the
        null block. ``block_size`` is in ``SUPPORTED_BLOCK_SIZES``. Bytes per chip:
        ``kv_cache_bytes_per_chip(num_blocks, block_size, num_layers, settings.kv_cache_dtype)``.

        ``num_blocks`` / ``block_size`` given here are authoritative (4129 / 64 for the default serving flags; 4105
        with ``--max-num-seqs 8``): record them in the TT config (``cfg.set_kv_geometry(num_blocks, block_size)``),
        never recompute them from the config. Allocate with ``ttnn.empty`` + on-device ``ttnn.fill(0)`` per layer
        (gate G7: ``ttnn.zeros`` of the pool takes ~20 s).

        With ``settings.spec_tokens = 1`` the generator also allocates the MTP layer's cache, same shape and dtype,
        indexed by the same block ids (so it travels with prefix hits and is freed with the request). ``num_layers``
        stays vLLM's count of main layers; the per-chip bytes are then
        ``kv_cache_bytes_per_chip(num_blocks, block_size, num_layers + settings.mtp_kv_layers, dtype)``.

        The handle is passed back unchanged as ``kv_cache=`` to every later call; the bridge never looks inside.
        """

    # ---- forwards ---------------------------------------------------------------------------------------------
    @abc.abstractmethod
    def prefill_forward(self, request: PrefillRequest, *, kv_cache: Any, enable_trace: bool = False) -> torch.Tensor:
        """Prefill one row; return the logits of its last position ``end - 1``.

        Draft 1 (``request.start == 0``): pads ``request.tokens`` to the smallest bucket ``>= S``, computes the full
        forward, and writes the latent of positions ``0 .. S-1`` into the request's blocks on every chip. Positions
        ``S .. bucket-1`` may be written into the request's own last block (decode overwrites them before they are
        read) or into null block 0 (never read); no other block may be written. KV of other lanes/requests must be
        left untouched. A generator without resumed prefill may refuse ``start > 0`` (``NotImplementedError``); one
        with it treats ``prefill_forward(r)`` as ``prefill_forward_batch([r])[0]`` (the :class:`PrefillRequest`
        contract: nothing below ``floor(start / bs) * bs`` is written).

        ``enable_trace`` is True only with plugin ``trace_mode="all"``; draft-1 implementations run eager anyway.

        Returns:
            Host logits for position ``S-1``: ``torch.float32`` or ``torch.bfloat16``, shape ``[vocab_size]``.
        """

    def prefill_forward_batch(
        self, requests: Sequence[PrefillRequest], *, kv_cache: Any, enable_trace: bool = False
    ) -> torch.Tensor:
        """Prefill all rows of one plugin step (ONE call per step); return host logits ``[B, vocab_size]`` (float32 or
        bfloat16), row ``i`` = position ``requests[i].end - 1``, in INPUT order.

        Rows (:func:`check_prefill_batch`): 1 .. ``NUM_LANES`` :class:`PrefillRequest` on distinct lanes; each obeys
        the :class:`PrefillRequest` contract (read-only blocks below ``floor(start / bs)``, writes from
        ``floor(start / bs) * bs``, padding only into the own last block or null block 0). Rows of one call may share
        read-only prefix blocks, and a row may READ blocks another row of the same call WRITES (vLLM caches full
        blocks when it allocates them, so a request admitted later in the step can hit them): run writers first
        (``prefill_plan.order_prefill_requests``) or layer-synchronously. Two rows never write the same block. No
        per-lane / per-slot state may cross calls (a request may change lane between chunks): the paged cache is the
        only cross-chunk state. With ``settings.spec_tokens`` the MTP layer's cache is written for the same positions
        (KV-only: its entry at ``p`` needs ``t_{p+1}``, which for a row's last position is the host argmax of the
        returned logits), so every prefilled position has an MTP entry (design G8).

        With ``settings.packed_prefill`` (P5, docs/p5_t64/P5_T64_DESIGN.md §3) the generator may run short chunks of
        several rows in one packed pass; every row still gets exactly the logits, KV writes and read-only blocks of
        this contract (a reader still runs after the writers of the blocks it reads), and the output order is unchanged.

        Default (draft-1 generators): every row must have ``start == 0`` (else ``NotImplementedError``); then no row
        reads the cache, so input order is writer-first, and the rows run one by one through ``prefill_forward``.
        A generator with ``supports_resumed_prefill`` overrides this (``prefill_plan.plan_prefill_batch``)."""
        reqs = check_prefill_batch(requests)
        resumed = [i for i, r in enumerate(reqs) if r.resumed]
        if resumed:
            raise NotImplementedError(
                f"{type(self).__name__}: rows {resumed} resume at start > 0 (chunked prefill / prefix caching), "
                f"which needs a generator with supports_resumed_prefill and its own prefill_forward_batch"
            )
        outs = []
        for r in reqs:
            logits = self.prefill_forward(r, kv_cache=kv_cache, enable_trace=enable_trace)
            outs.append(check_logits(f"{type(self).__name__}.prefill_forward", logits, (int(self.vocab_size),)))
        return torch.stack(outs)

    @abc.abstractmethod
    def decode_forward(self, batch: DecodeBatch, *, kv_cache: Any, enable_trace: bool) -> torch.Tensor:
        """One decode step for every lane.

        For each active lane ``l`` (``batch.positions[l] = p >= 0``): writes the latent of ``batch.tokens[l]`` at
        position ``p`` (block ``batch.page_table[l, p // block_size]``, row ``p % block_size``) on the chips of DP
        group ``l // 8`` -- on all 32 chips when ``settings.kv_replicated`` (KV-R: then a later prefix hit from any
        lane reads valid KV) --, attends over positions ``0 .. p`` (global layers) or ``max(0, p - 128) .. p`` (SWA
        layers, 129 keys), and produces the next-token logits. Inactive lanes write nothing.

        ``enable_trace=True``: copy the inputs into the persistent device tensors and replay the trace captured by
        ``warmup_decode(enable_trace=True)``; if no trace was captured (warmup disabled), run eager instead of
        capturing one lazily (a capture behind later first-time prefill compiles could be corrupted).

        Returns:
            Host logits ``torch.float32`` or ``torch.bfloat16`` of shape ``[NUM_LANES, vocab_size]`` in lane order.
            Rows of inactive lanes are ignored (any value, including non-finite). The tensor is handed to vLLM's
            sampler, so it must not alias a host buffer that a later call overwrites (return a fresh tensor).
        """

    def decode_forward_spec(
        self, batch: SpecDecodeBatch, *, kv_cache: Any, enable_trace: bool, want_logits: bool
    ) -> SpecDecodeResult:
        """One decode step of a speculating launch (``settings.spec_tokens = 1``; design §3.8). Every decode step of
        such a launch comes here: ordinary steps (no drafts, ``want_logits=True`` for host sampling) and verify steps
        (``want_logits=False``: the plugin needs only argmax ids).

        For each active owner lane ``l`` (anchor token ``t`` at ``n = batch.positions[l]``):
          * the main layers (``num_layers``, 53) + the MTP layer write the anchor's latent at ``n`` (block
            ``page_table[l, n // bs]``);
          * ``a0`` = target argmax at ``n``, and the MTP layer runs on ``(hn_n, a0)`` at ``n``: ``m0``;
          * when ``draft_tokens[l] = d >= 0``: ``d`` is evaluated at ``n + 1`` through the SAME page-table row (its KV
            lands in ``l``'s blocks at ``n + 1``; it attends over ``0 .. n + 1`` including the anchor's ``n``):
            ``a1`` = target argmax at ``n + 1``, ``m1`` = MTP on ``(hn_{n+1}, a1)``.
        KV writes land on all 32 chips with ``settings.kv_replicated`` (KV-R), else on the owner's DP row only. A
        rejected draft's KV at ``n + 1`` is garbage, overwritten by the next step's anchor before any read; vLLM never
        caches it (it is beyond ``num_computed_tokens``).

        Layout is the generator's business (design §3.8.2, "packed" verify): a draft runs on an IDLE lane (one whose
        ``positions`` is -1; with KV-R on any DP row, without it on the owner's row) with a copy of the owner's
        page-table row; anchors and drafts write the cache in two separate ``paged_update_cache`` calls (two users
        writing ``p`` and ``p + 1`` of one tile in one call race). Drafts that do not fit the idle lanes are run in
        an overflow pass on their own lanes at ``n + 1``: every draft must be evaluated (if ``d == a0`` the plugin
        commits ``a1``). The result is the same as running every row on its own lane.

        With ``settings.spec_verify`` ``"wide"`` / ``"auto"`` (T64, docs/p5_t64/P5_T64_DESIGN.md §4) a verify step
        may instead run as one 64-row step: per DP row the 8 lanes' anchors, then their drafts, each draft on its
        owner's DP row with the owner's page-table row at ``n + 1`` (no idle lanes needed, no overflow pass); in
        ``"auto"`` only when the drafts do not all fit idle lanes. Same result, same contract.

        ``enable_trace``: replay the decode trace (captured by ``warmup_decode(enable_trace=True)``; the overflow
        pass replays it again), or run eager when none was captured.

        Returns:
            :class:`SpecDecodeResult` in owner-lane order (``check_spec_result``): ``argmax [32, 2]`` = ``(a0, a1)``,
            ``mtp_argmax [32, 2]`` = ``(m0, m1)``, ``logits [32, vocab]`` only when ``want_logits``. Fresh tensors.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement speculative decoding")

    # ---- warmup -----------------------------------------------------------------------------------------------
    @abc.abstractmethod
    def warmup_prefill(self, *, kv_cache: Any, enable_trace: bool) -> None:
        """Compile every prefill bucket (``prefill_buckets(settings.max_seq_len)``) once.

        No request is live during warmup, so it may write any block of the pool (all-zero page tables keep every
        write in null block 0). Must leave no lane state behind. With ``enable_trace=True`` (plugin
        ``trace_mode="all"``, called before decode capture) a draft-1 generator may return immediately.

        With resumed prefill: compile every ``(path, bucket)`` the generator can run, i.e. sp0 and sp1 for every
        bucket ``<= max_prefill_span`` (sp1 at a start of 128 with all-zero SDPA tables and all ``-1`` fill tables,
        so nothing real is written), plus the MTP layer's KV-only prefill fill when ``settings.spec_tokens``. Shapes
        depend only on ``(path, bucket)`` (starts, block ids, RoPE rows and the LM-head row are device tensors), so
        nothing compiles after the decode capture (a prefill program compiled after capture can corrupt the trace).
        With ``settings.packed_prefill`` also every packed shape (``MotifTTConfig.packed_prefill_shapes()``: pk0
        ``(T, S)`` and pk1 ``(T, S)`` with both SWA tail variants); a packed shape left unwarmed runs as solo chunks
        after the capture.
        """

    @abc.abstractmethod
    def warmup_decode(self, *, kv_cache: Any, enable_trace: bool, page_table_width: int) -> None:
        """Prepare decode for page tables of width ``page_table_width`` (fixed for the server's lifetime).

        ``enable_trace=False``: run one eager decode step (compiles every decode program, stages the persistent
        input tensors ``tokens [32]``, ``positions [32]``, ``page_table [32, W]``). ``enable_trace=True``: capture
        the decode trace (embed -> layers -> LM head; logits read outside the trace). All lanes inactive or
        writing only block 0; no lane state may survive.

        In a speculating launch the one trace is the spec trace (main layers with the split KV write, LM head, main
        argmax, MTP layer, MTP argmax) that serves ordinary, verify and overflow steps; the KV-write mode
        (``settings.kv_write_mode``) is fixed at capture. ``settings.spec_verify="wide"``: the one trace is the 64-row
        T64 trace instead; ``"auto"``: both, each staged and run eagerly once before the first capture, then captured
        once (T32 spec trace first) and never re-captured while serving (F3N rules R2-R5,
        docs/p5_t64/P5_T64_DESIGN.md §2.3).
        """

    # ---- lifecycle --------------------------------------------------------------------------------------------
    def release_lane(self, lane: int) -> None:
        """The request on ``lane`` finished or was preempted. Drop lane-owned model state (none in draft 1: the
        KV lives in vLLM's blocks; the G1-fallback SWA ring would be reset here; the speculation bookkeeping --
        retained argmax / MTP ids per lane -- lives in the bridge, not here).

        The plugin delivers it with the NEXT scheduler step (a request finishing in the last step of a burst is
        released only when traffic resumes), but always before that lane is handed to another request's prefill.
        Lane-owned state should still be (re)initialised by ``prefill_forward`` rather than rely on this call."""
        return None

    def release_traces(self) -> None:
        """Release captured traces (called at shutdown with the mesh still open)."""
        return None

    def close(self) -> None:
        """Release everything this generator owns on the device (standalone use; vLLM only calls
        ``release_traces``)."""
        self.release_traces()


def check_logits(name: str, logits: Any, shape: Tuple[int, ...]) -> torch.Tensor:
    """Validate a generator's logits output (host, floating point, exact shape)."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"{name} must return a host torch.Tensor of logits, got {type(logits).__name__}")
    if logits.device.type != "cpu" or not logits.is_floating_point():
        raise TypeError(f"{name} must return CPU floating-point logits, got {logits.dtype} on {logits.device}")
    if tuple(logits.shape) != tuple(shape):
        raise ValueError(f"{name} returned logits of shape {tuple(logits.shape)}, expected {tuple(shape)}")
    return logits


def cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_KV_CACHE_DTYPE",
    "DEFAULT_KV_POOL_TOKENS",
    "DEFAULT_PACKED_PREFILL_MAX_SEG",
    "DEFAULT_PACKED_PREFILL_MAX_TOKENS",
    "DEFAULT_PACKED_WARMUP",
    "DEFAULT_CHUNK_BUDGET",
    "DEFAULT_PREFILL_ALIGNMENT",
    "DEFAULT_PREFILL_SPAN_CAP",
    "DEFAULT_SPEC_ALPHA_PRIOR",
    "DEFAULT_TT_CACHE_POLICY",
    "DecodeBatch",
    "FEATURE_SWITCHES",
    "GeneratorSettings",
    "KV_CACHE_DTYPES",
    "KV_LATENT_DIM",
    "KV_LORA_RANK",
    "KV_POOL_ALIGNMENT",
    "KV_WRITE_MODES",
    "L1_SMALL_SIZE",
    "LANES_PER_GROUP",
    "MAX_CONTEXT",
    "MAX_KV_POOL_TOKENS",
    "MAX_MODEL_LEN_ALIGNMENT",
    "MESH_SHAPES",
    "MIN_PREFILL_BUCKET",
    "MTP_LAYER_IDX",
    "MotifGenerator",
    "NULL_BLOCK_RESERVE_TOKENS",
    "NUM_DP_GROUPS",
    "NUM_HIDDEN_LAYERS",
    "NUM_LANES",
    "PACKED_PASS_KINDS",
    "PACKED_WARMUP_MODES",
    "PACK_BATCHES",
    "PACK_SEG_BUCKETS",
    "PACK_SP1_SEG_BUCKETS",
    "PK1_TAIL_VARIANTS",
    "PrefillRequest",
    "QK_ROPE_HEAD_DIM",
    "SERVING_KEYS",
    "SERVING_TT_CONFIG",
    "SPEC_ALPHA_PRIOR_WEIGHT",
    "SPEC_VERIFY_MODES",
    "SUPPORTED_BLOCK_SIZES",
    "SUPPORTED_SPEC_TOKENS",
    "SpecDecodeBatch",
    "SpecDecodeResult",
    "TT_CACHE_POLICIES",
    "VOCAB_SIZE",
    "WIDE_MIN_LANES_NEVER",
    "WIDE_ROWS",
    "WIDE_ROWS_PER_GROUP",
    "WIDE_SPEC_VERIFY_MODES",
    "WeightsLocation",
    "cdiv",
    "check_block_size",
    "check_generator_features",
    "check_logits",
    "check_max_model_len",
    "check_packed_prefill_max_seg",
    "check_packed_prefill_max_tokens",
    "check_chunk_budget",
    "check_prefill_batch",
    "check_prefill_span_cap",
    "check_spec_alpha_prior",
    "check_spec_result",
    "check_tt_cache_policy",
    "check_tt_config",
    "check_wide_min_lanes",
    "check_wide_step_ratio",
    "chunk_budget_from_env",
    "expected_num_blocks",
    "feature_switch_from_env",
    "hf_cache_snapshot",
    "kv_cache_bytes_per_chip",
    "kv_cache_dtype_from_env",
    "kv_pool_tokens_from_env",
    "kv_replicated_decode_from_env",
    "kv_write_mode",
    "packed_prefill_from_env",
    "packed_prefill_max_seg_from_env",
    "packed_prefill_max_tokens_from_env",
    "packed_prefill_pk1_from_env",
    "packed_warmup_from_env",
    "plugin_num_blocks",
    "prefill_buckets",
    "prefill_span_cap_from_env",
    "resolve_tt_cache_path",
    "resolve_weights_location",
    "serving_additional_config",
    "smoothed_acceptance",
    "spec_verify_from_env",
    "HOST_STAGING_MODES",
    "DEFAULT_HOST_STAGING",
    "check_host_staging",
    "host_staging_from_env",
    "HOST_WAIT_MODES",
    "DEFAULT_HOST_WAIT",
    "check_host_wait",
    "ASYNC_DECODE_MODES",
    "DEFAULT_ASYNC_DECODE",
    "check_async_decode",
    "async_decode_from_env",
    "CAPTURE_THREAD_MODES",
    "DEFAULT_CAPTURE_THREAD",
    "check_capture_thread",
    "capture_thread_from_env",
    "PREFILL_TRACE_BUCKETS",
    "DEFAULT_PREFILL_TRACE",
    "check_prefill_trace",
    "prefill_trace_buckets",
    "prefill_trace_from_env",
    "tt_cache_policy_from_env",
    "wide_min_lanes_from_env",
    "wide_step_ratio_from_env",
]
