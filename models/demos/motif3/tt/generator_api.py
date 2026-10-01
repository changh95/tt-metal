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
A decode step writes a lane's new latent only on its own group's chips, so a request must keep its lane for its
whole life. The bridge guarantees that (``LaneMap`` in ``generator_vllm.py`` follows vLLM's ``slot_remap``).
Prefill writes the latent on every chip, so a request that is re-prefilled (preemption) may get any lane.

KV cache (design 00 §1.5, §2.3.4; study 05 §7.3)
------------------------------------------------
One paged latent cache per decoder layer, logical shape ``[num_blocks, 1, block_size, KV_LATENT_DIM]`` with
``KV_LATENT_DIM = 576 = kv_lora_rank 512 (unit-RMS latent, kv_norm gamma folded into W_UK/W_UV) ‖ RoPE'd k_pe 64``.
It is replicated on all 32 chips (one block-id space, DP=1 in vLLM). vLLM owns the block ids: block 0 is vLLM's
null block (padding in every page table, never holds a valid position); real blocks are ``1 .. num_blocks-1``.
Device dtype is ``settings.kv_cache_dtype`` (``"bfp8"`` -> ``ttnn.bfloat8_b``, ``"bf16"`` -> ``ttnn.bfloat16``),
TILE layout, DRAM. The torch dtype vLLM hands the bridge is bookkeeping only.

Positions
---------
A position is the absolute token index in the request (0-based). For decode, ``positions[l]`` is the index of
``tokens[l]`` = the KV slot that token's latent is written to; the returned logits predict index
``positions[l] + 1``. Its latent goes to block ``page_table[l, p // block_size]``, row ``p % block_size``.
``-1`` marks an inactive lane: no KV write, no reads, its logits row is ignored.

Host tensors only
-----------------
Every tensor crossing this interface is a CPU torch tensor. Inputs: ``torch.int32``. Outputs: logits as
``torch.float32`` or ``torch.bfloat16`` (vLLM's sampler upcasts), never padded past ``vocab_size``.

Import rule (design 00 §2.1): this module is imported by vLLM's API server, its registry-inspection subprocess and
EngineCore before any mesh exists. It imports only the standard library and torch, never ttnn or other
``models/demos/**`` packages, and never touches a device.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

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
SUPPORTED_BLOCK_SIZES = (32, 64, 128)  # multiples of the 32-row tile that the paged ops are tested with
DEFAULT_BLOCK_SIZE = 64

KV_CACHE_DTYPES = ("bfp8", "bf16")
DEFAULT_KV_CACHE_DTYPE = "bfp8"
_TILE = 32
_TILE_BYTES = {"bfp8": 1088, "bf16": 2048}  # one 32x32 tile: bfp8_b = 1024 mantissa + 64 exponent bytes


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

    The integration wave maps them onto ``MotifTTConfig.from_hf_config(hf_config, mesh_device=..., num_layers=...,
    max_model_len=max_seq_len, ...)``; the KV pool geometry itself arrives later through ``allocate_kv_cache``.

    Fields:
        max_batch_size: vLLM ``max_num_seqs`` (1..32). The decode trace always runs ``NUM_LANES`` lanes; this
            only bounds how many lanes can be active.
        max_seq_len: vLLM ``max_model_len`` (1..``MAX_CONTEXT``): largest prompt and largest decode position + 1.
        num_layers: decoder layers to run (``hf_config.num_hidden_layers``, or fewer for a truncated bring-up
            run via ``MOTIF3_NUM_LAYERS``; the final norm and LM head always run).
        kv_cache_dtype: ``"bfp8"`` (default) or ``"bf16"`` (``MOTIF3_KV_CACHE_DTYPE``; bf16 is the A/B lever of
            design 00 §7.1 risk 8 and needs a smaller pool).
        weights_path: ``HF_MODEL`` (snapshot dir or repo id), else ``hf_config._name_or_path``.
        weights_revision: ``TT_MODEL_WEIGHTS_REVISION`` (pinned snapshot when ``weights_path`` is a repo id).
        cache_path: ``TT_CACHE_PATH`` (TT weight-cache root), or None for the generator's default.
        optimizations: plugin ``tt.optimizations`` (None | "performance" | "accuracy"); draft 1 may ignore it.
    """

    max_batch_size: int = NUM_LANES
    max_seq_len: int = MAX_CONTEXT
    num_layers: int = NUM_HIDDEN_LAYERS
    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE
    weights_path: Optional[str] = None
    weights_revision: Optional[str] = None
    cache_path: Optional[str] = None
    optimizations: Optional[str] = None

    def __post_init__(self):
        if not 1 <= int(self.max_batch_size) <= NUM_LANES:
            raise ValueError(f"max_batch_size must be in [1, {NUM_LANES}], got {self.max_batch_size}")
        if not 1 <= int(self.max_seq_len) <= MAX_CONTEXT:
            raise ValueError(f"max_seq_len must be in [1, {MAX_CONTEXT}] in draft 1, got {self.max_seq_len}")
        if int(self.num_layers) < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.kv_cache_dtype not in KV_CACHE_DTYPES:
            raise ValueError(f"kv_cache_dtype must be one of {KV_CACHE_DTYPES}, got {self.kv_cache_dtype!r}")
        if self.optimizations not in (None, "performance", "accuracy"):
            raise ValueError(f"optimizations must be None, 'performance' or 'accuracy', got {self.optimizations!r}")

    @classmethod
    def from_env(
        cls,
        hf_config: Any,
        *,
        max_batch_size: int,
        max_seq_len: int,
        optimizations: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "GeneratorSettings":
        """Resolve settings from the vLLM arguments plus the documented environment variables."""
        env = os.environ if environ is None else environ
        hf_layers = int(getattr(hf_config, "num_hidden_layers", NUM_HIDDEN_LAYERS))
        env_layers = _env_int(env, "MOTIF3_NUM_LAYERS")
        num_layers = hf_layers if env_layers is None else env_layers
        if not 1 <= num_layers <= hf_layers:
            raise ValueError(f"MOTIF3_NUM_LAYERS={num_layers} outside [1, {hf_layers}]")
        kv_dtype = (env.get("MOTIF3_KV_CACHE_DTYPE") or DEFAULT_KV_CACHE_DTYPE).strip().lower()
        weights = (
            env.get("HF_MODEL") or getattr(hf_config, "_name_or_path", None) or getattr(hf_config, "name_or_path", None)
        )
        return cls(
            max_batch_size=int(max_batch_size),
            max_seq_len=int(max_seq_len),
            num_layers=num_layers,
            kv_cache_dtype=kv_dtype,
            weights_path=str(weights) if weights else None,
            weights_revision=env.get("TT_MODEL_WEIGHTS_REVISION") or None,
            cache_path=env.get("TT_CACHE_PATH") or None,
            optimizations=optimizations,
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
    """One prompt to prefill (draft 1 prefills one request per call, eager, padded to a bucket).

    Attributes:
        lane: destination lane in ``[0, NUM_LANES)``. Its DP group (``lane // 8``) runs every later decode step of
            this request. Draft-1 prefill writes the latent on every chip, so the lane only matters to lane-owned
            state (none in draft 1; the G1-fallback SWA ring and a v1 per-group prefill would use it).
        tokens: ``torch.int32 [S]``, the request's tokens at positions ``0 .. S-1`` (``1 <= S <= max_seq_len``),
            unpadded. For a request resumed after preemption this is prompt + every generated token.
        page_table: ``torch.int32 [W]``, the request's vLLM block ids in position order (W = the server's
            page-table width, ``min(ceil(max_seq_len / block_size), num_blocks)``). Entries
            ``0 .. ceil(S / block_size) - 1`` are real blocks (>= 1); the rest are 0 (null block), so bucket-padding
            writes past them land in block 0. The bridge zeroes that tail itself: vLLM's persistent block-table rows
            keep stale ids there, often of blocks other live requests own now. Never write through a page-table
            entry the bridge did not hand over.
    """

    lane: int
    tokens: torch.Tensor
    page_table: torch.Tensor

    def __post_init__(self):
        if not 0 <= int(self.lane) < NUM_LANES:
            raise ValueError(f"lane must be in [0, {NUM_LANES}), got {self.lane}")
        _check_int32("tokens", self.tokens, 1)
        _check_int32("page_table", self.page_table, 1)
        if self.tokens.shape[0] < 1:
            raise ValueError("empty prompt")

    @property
    def seq_len(self) -> int:
        return int(self.tokens.shape[0])


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


# ----------------------------------------------------------------------------------------------------------------
# The runtime interface
# ----------------------------------------------------------------------------------------------------------------
class MotifGenerator(abc.ABC):
    """What the integration wave implements in ``tt/generator.py`` (default class path
    ``models.demos.motif3.tt.generator:MotifGenerator``, overridable with ``MOTIF3_GENERATOR_CLASS``).

    Call order under vLLM (vllm-tt-plugin ``worker.py:221-469``, ``model_runner.py:678-708, 3727-3781``):

    1. ``create(hf_config=..., mesh_device=..., settings=...)`` once, after the plugin opened the mesh
       (``initialize_vllm_model``). Loads/converts weights; must not allocate the KV pool.
    2. ``allocate_kv_cache(num_blocks=, block_size=, num_layers=)`` once.
    3. ``warmup_prefill(enable_trace=False)`` -> ``warmup_decode(enable_trace=False, page_table_width=W)`` ->
       [``warmup_prefill(enable_trace=True)`` only with plugin ``trace_mode="all"``] ->
       ``warmup_decode(enable_trace=True, page_table_width=W)`` (decode trace capture). Warmup is skipped when
       the plugin runs with ``enable_model_warmup=false`` (bring-up).
    4. Serving: any interleaving of ``prefill_forward`` (one request per call) and ``decode_forward``;
       ``release_lane`` when a request on that lane finished or was preempted.
    5. ``release_traces()`` at shutdown while the mesh is still open (the plugin closes the mesh afterwards).

    Concurrency: calls are strictly sequential (one EngineCore thread). Every method may raise; a raise must leave
    no partially-applied host state behind (the bridge commits its own lane bookkeeping only after success).
    """

    # ---- construction -----------------------------------------------------------------------------------------
    @classmethod
    @abc.abstractmethod
    def create(cls, *, hf_config: Any, mesh_device: Any, settings: GeneratorSettings) -> "MotifGenerator":
        """Build the runtime on an already-open mesh.

        Args:
            hf_config: vLLM's ``model_config.hf_config``: the trust-remote-code ``MotifConfig`` instance (dynamic
                class; duck-type it, never ``isinstance``). ``hf_config.architectures`` has already been rewritten
                to ``["TTMotifForCausalLM"]`` by the plugin.
            mesh_device: the ``ttnn.MeshDevice`` the plugin opened from ``MESH_DEVICE``: shape (4, 8) or (8, 4)
                (the TP axis is the size-8 dim), fabric ``FABRIC_2D_TORUS_XY`` by default.
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
        """Longest prompt ``prefill_forward`` accepts (its largest bucket); >= ``settings.max_seq_len``."""
        return MAX_CONTEXT

    # ---- KV pool ----------------------------------------------------------------------------------------------
    @abc.abstractmethod
    def allocate_kv_cache(self, *, num_blocks: int, block_size: int, num_layers: int) -> Any:
        """Allocate the paged latent pool and return an opaque handle.

        Allocates ``num_layers`` (== ``self.num_layers``) tensors, each of logical shape
        ``[num_blocks, 1, block_size, KV_LATENT_DIM]``, dtype per ``settings.kv_cache_dtype`` (``ttnn.bfloat8_b`` or
        ``ttnn.bfloat16``), TILE layout, DRAM, replicated on every chip of the mesh, zero-filled. Block 0 is the
        null block. ``block_size`` is in ``SUPPORTED_BLOCK_SIZES``. Bytes per chip:
        ``kv_cache_bytes_per_chip(num_blocks, block_size, num_layers, settings.kv_cache_dtype)``.

        The handle is passed back unchanged as ``kv_cache=`` to every later call; the bridge never looks inside.
        """

    # ---- forwards ---------------------------------------------------------------------------------------------
    @abc.abstractmethod
    def prefill_forward(self, request: PrefillRequest, *, kv_cache: Any, enable_trace: bool = False) -> torch.Tensor:
        """Prefill one request; return the logits of its last token.

        Pads ``request.tokens`` to the smallest bucket ``>= S``, computes the full forward, and writes the latent of
        positions ``0 .. S-1`` into the request's blocks on every chip. Positions ``S .. bucket-1`` may be written
        into the request's own last block (decode overwrites them before they are read) or into null block 0
        (never read); no other block may be written. KV of other lanes/requests must be left untouched.

        ``enable_trace`` is True only with plugin ``trace_mode="all"``; draft-1 implementations run eager anyway.

        Returns:
            Host logits for position ``S-1``: ``torch.float32`` or ``torch.bfloat16``, shape ``[vocab_size]``.
        """

    @abc.abstractmethod
    def decode_forward(self, batch: DecodeBatch, *, kv_cache: Any, enable_trace: bool) -> torch.Tensor:
        """One decode step for every lane.

        For each active lane ``l`` (``batch.positions[l] = p >= 0``): writes the latent of ``batch.tokens[l]`` at
        position ``p`` (block ``batch.page_table[l, p // block_size]``, row ``p % block_size``) on the chips of DP
        group ``l // 8``, attends over positions ``0 .. p`` (global layers) or ``max(0, p - 128) .. p`` (SWA layers,
        129 keys), and produces the next-token logits. Inactive lanes write nothing.

        ``enable_trace=True``: copy the inputs into the persistent device tensors and replay the trace captured by
        ``warmup_decode(enable_trace=True)``; if no trace was captured (warmup disabled), run eager instead of
        capturing one lazily (a capture behind later first-time prefill compiles could be corrupted).

        Returns:
            Host logits ``torch.float32`` or ``torch.bfloat16`` of shape ``[NUM_LANES, vocab_size]`` in lane order.
            Rows of inactive lanes are ignored (any value, including non-finite). The tensor is handed to vLLM's
            sampler, so it must not alias a host buffer that a later call overwrites (return a fresh tensor).
        """

    # ---- warmup -----------------------------------------------------------------------------------------------
    @abc.abstractmethod
    def warmup_prefill(self, *, kv_cache: Any, enable_trace: bool) -> None:
        """Compile every prefill bucket (``prefill_buckets(settings.max_seq_len)``) once.

        No request is live during warmup, so it may write any block of the pool (all-zero page tables keep every
        write in null block 0). Must leave no lane state behind. With ``enable_trace=True`` (plugin
        ``trace_mode="all"``, called before decode capture) a draft-1 generator may return immediately.
        """

    @abc.abstractmethod
    def warmup_decode(self, *, kv_cache: Any, enable_trace: bool, page_table_width: int) -> None:
        """Prepare decode for page tables of width ``page_table_width`` (fixed for the server's lifetime).

        ``enable_trace=False``: run one eager decode step (compiles every decode program, stages the persistent
        input tensors ``tokens [32]``, ``positions [32]``, ``page_table [32, W]``). ``enable_trace=True``: capture
        the decode trace (embed -> layers -> LM head; logits read outside the trace). All lanes inactive or
        writing only block 0; no lane state may survive.
        """

    # ---- lifecycle --------------------------------------------------------------------------------------------
    def release_lane(self, lane: int) -> None:
        """The request on ``lane`` finished or was preempted. Drop lane-owned model state (none in draft 1: the
        KV lives in vLLM's blocks; the G1-fallback SWA ring would be reset here).

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
    "DecodeBatch",
    "GeneratorSettings",
    "KV_CACHE_DTYPES",
    "KV_LATENT_DIM",
    "KV_LORA_RANK",
    "LANES_PER_GROUP",
    "MAX_CONTEXT",
    "MESH_SHAPES",
    "MIN_PREFILL_BUCKET",
    "MotifGenerator",
    "NUM_DP_GROUPS",
    "NUM_HIDDEN_LAYERS",
    "NUM_LANES",
    "PrefillRequest",
    "QK_ROPE_HEAD_DIM",
    "SUPPORTED_BLOCK_SIZES",
    "VOCAB_SIZE",
    "cdiv",
    "check_logits",
    "kv_cache_bytes_per_chip",
    "kv_cache_dtype_from_env",
    "prefill_buckets",
]
