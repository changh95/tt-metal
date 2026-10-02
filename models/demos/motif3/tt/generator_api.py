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
``torch.float32`` or ``torch.bfloat16`` (vLLM's sampler upcasts), never padded past ``vocab_size``.

Import rule (design 00 §2.1): this module is imported by vLLM's API server, its registry-inspection subprocess and
EngineCore before any mesh exists. It imports only the standard library and torch, never ttnn or other
``models/demos/**`` packages, and never touches a device (``huggingface_hub`` is imported lazily, only to locate the
HF-cache snapshot of a repo-id ``HF_MODEL``). ``tt/model_config.py`` imports the KV-pool and weights-location helpers
from here, so the bridge and the TT config share one implementation.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple, Union

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
        optimizations: plugin ``tt.optimizations`` (None | "performance" | "accuracy"); draft 1 may ignore it.
        block_size: vLLM ``--block-size`` when the bridge could see it at model init (BRIDGE-4), else None. Only a
            hint for ``create``: ``allocate_kv_cache(block_size=...)`` is authoritative.
        weights_source: which precedence rule produced ``weights_path`` (logged by the bridge).
    """

    max_batch_size: int = NUM_LANES
    max_seq_len: int = MAX_CONTEXT
    num_layers: int = NUM_HIDDEN_LAYERS
    kv_cache_dtype: str = DEFAULT_KV_CACHE_DTYPE
    weights_path: Optional[str] = None
    weights_revision: Optional[str] = None
    cache_path: Optional[str] = None
    optimizations: Optional[str] = None
    block_size: Optional[int] = None
    weights_source: Optional[str] = None

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
        if self.optimizations not in (None, "performance", "accuracy"):
            raise ValueError(f"optimizations must be None, 'performance' or 'accuracy', got {self.optimizations!r}")
        if self.block_size is not None:
            check_block_size(self.block_size)

    @property
    def weights_are_local(self) -> bool:
        """``weights_path`` is a local checkpoint directory (False for an uncached repo id or None)."""
        return self.weights_path is not None and Path(self.weights_path).is_dir()

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
    ) -> "GeneratorSettings":
        """Resolve settings from the vLLM arguments plus the documented environment variables."""
        env = os.environ if environ is None else environ
        hf_layers = int(getattr(hf_config, "num_hidden_layers", NUM_HIDDEN_LAYERS))
        env_layers = _env_int(env, "MOTIF3_NUM_LAYERS")
        num_layers = hf_layers if env_layers is None else env_layers
        if not 1 <= num_layers <= hf_layers:
            raise ValueError(f"MOTIF3_NUM_LAYERS={num_layers} outside [1, {hf_layers}]")
        kv_dtype = (env.get("MOTIF3_KV_CACHE_DTYPE") or DEFAULT_KV_CACHE_DTYPE).strip().lower()
        loc = resolve_weights_location(hf_config, env)
        return cls(
            max_batch_size=int(max_batch_size),
            max_seq_len=int(max_seq_len),
            num_layers=num_layers,
            kv_cache_dtype=kv_dtype,
            weights_path=loc.path,
            weights_revision=loc.revision,
            cache_path=resolve_tt_cache_path(env),
            optimizations=optimizations,
            block_size=None if block_size is None else int(block_size),
            weights_source=loc.source,
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

        ``num_blocks`` / ``block_size`` given here are authoritative (4129 / 64 for the default serving flags; 4105
        with ``--max-num-seqs 8``): record them in the TT config (``cfg.set_kv_geometry(num_blocks, block_size)``),
        never recompute them from the config. Allocate with ``ttnn.empty`` + on-device ``ttnn.fill(0)`` per layer
        (gate G7: ``ttnn.zeros`` of the pool takes ~20 s).

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
    "DEFAULT_KV_POOL_TOKENS",
    "DecodeBatch",
    "GeneratorSettings",
    "KV_CACHE_DTYPES",
    "KV_LATENT_DIM",
    "KV_LORA_RANK",
    "KV_POOL_ALIGNMENT",
    "L1_SMALL_SIZE",
    "LANES_PER_GROUP",
    "MAX_CONTEXT",
    "MAX_KV_POOL_TOKENS",
    "MAX_MODEL_LEN_ALIGNMENT",
    "MESH_SHAPES",
    "MIN_PREFILL_BUCKET",
    "MotifGenerator",
    "NULL_BLOCK_RESERVE_TOKENS",
    "NUM_DP_GROUPS",
    "NUM_HIDDEN_LAYERS",
    "NUM_LANES",
    "PrefillRequest",
    "QK_ROPE_HEAD_DIM",
    "SERVING_TT_CONFIG",
    "SUPPORTED_BLOCK_SIZES",
    "VOCAB_SIZE",
    "WeightsLocation",
    "cdiv",
    "check_block_size",
    "check_logits",
    "check_max_model_len",
    "check_tt_config",
    "expected_num_blocks",
    "hf_cache_snapshot",
    "kv_cache_bytes_per_chip",
    "kv_cache_dtype_from_env",
    "kv_pool_tokens_from_env",
    "plugin_num_blocks",
    "prefill_buckets",
    "resolve_tt_cache_path",
    "resolve_weights_location",
    "serving_additional_config",
]
