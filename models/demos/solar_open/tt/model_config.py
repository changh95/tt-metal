# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Solar-Open-100B ModelArgs, compatible with the tt_transformers Generator / create_tt_model interface.

ModelArgs resolves the checkpoint (HF_MODEL), loads the HuggingFace SolarOpenConfig and tokenizer, owns the MoE flag
bundle (MoEOptions), the generation stop set ({2, 24, 25} from generation_config.json), the chat-template
encoding, and the on-disk ttnn weight cache (directory naming + completion marker). load_state_dict is the single
host weight path (contract C1): whole-model bf16 from_pretrained -> Meta-permuted q/k -> fp32 router bias kept, or,
with SOLAR_OPEN_STREAMING_LOAD=1, the per-layer streaming LazyStateDict that presents the same keys lazily.
"""

import dataclasses
import gc
import json
import os
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.demos.solar_open.config import MoEOptions
from models.demos.solar_open.tt.expert_configs import SolarOpenProgramConfig
from models.demos.solar_open.tt.experts.prefill import is_trace_safe_prefill_len
from models.demos.solar_open.utils.general_utils import resolve_rope_theta
from models.tt_transformers.tt.common import (
    calculate_prefill_warmup_seq_lens,
    cap_seq_lens_to_max_prefill_chunk_size,
    get_all_padded_prefill_lengths,
    get_base_model_name,
    get_padded_prefill_len,
)
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

# HF repo id used when HF_MODEL is unset. Its basename is the model name and transformers resolves it from the HF
# hub cache; a local directory or symlink literally named "Solar-Open-100B" works the same way.
DEFAULT_HF_MODEL = "upstage/Solar-Open-100B"
MODEL_NAME = "Solar-Open-100B"
# Key suffix of the router's selection bias - the only fp32 tensor in the checkpoint; it must stay fp32 (C1).
ROUTER_BIAS_SUFFIX = "e_score_correction_bias"

# Traced prefill lengths per (model, device name) -- phase 3a(2), design_traced_prefill.md. 128 tokens on P150x8 is the
# validated bucket (its MoE runs the dense bmm path, trace-safe by construction; 150-168 ms TTFT). 1K / 2K / 4K are
# NO-GO for phase 3a: a whole-forward trace removes at most the 55-65 ms of exposed host time of a 1K prefill (~360 ms
# kernel floor vs 413-426 ms eager), while every trace-safe MoE formulation costs more per 1024-token split than the
# host-planned expert-sorted path saves (static per-expert loop +13 ms/layer; device-planned sorted (C) +1.2..+3.7
# ms/layer until the sparse_matmul kernel work of item 4b). Adding a length here requires ALL of: (a) a trace-safe MoE
# path for its splits -- experts/prefill.py is_trace_safe_prefill_len: every split <= dense_bmm_max_tokens or listed in
# SolarOpenProgramConfig.trace_safe_split_lens (ModelArgs refuses the entry otherwise); (b) the trace region of design
# 6.3 (160 MB for {128, 1K, 2K}, 256 MB with 4K; models/model_trace_region_sizes.yaml solar-open-100b); (c) for 4096
# also capped_warmup_seq_len >= 4096 (the list is capped by it); (d) the prepare-all-then-record-all capture order of
# design 6.4 (tt-metal #52176); (e) the traced-vs-eager equivalence test of tests/unit/test_traced_prefill.py for the
# bucket. The router helpers of every entry are persistent (router_persistent_token_counts).
TRACE_PREFILL_SEQ_LENS = {MODEL_NAME: {"P150x8": [128]}}  # format: model_name: {device_name: [sequence_lengths]}
# Expert slots of Solar-Open-100B: validates the trace table when no HF config is loaded (dummy weights).
_SOLAR_NUM_EXPERTS = 128

# ---------------------------------------------------------------------------------------------------------------------
# Phase 3a(1): packed / batched multi-user prefill on the single-row (1x8) mesh -- scratchpad design_packed_prefill.md.
#
# The tt_transformers Generator already has a complete batched-prefill path (slot placement, -1 page-table rows for
# unused slots, per-user last-token extraction) that ``ModelArgs.disable_batched_prefill`` switches off. Everything
# below the Generator is token-major: the MoE sees ``[1, 1, B*S, H]`` and never looks at users, the router works on
# ``T = B*S`` rows and ``attention/prefill.py`` reshapes to ``[B, 1, S, -1]`` and fills the paged KV cache per user
# with its own page-table row. Packing B short prompts into one forward therefore replaces B dense-bmm 128-token
# passes (~180 ms kernel each, 1.4 ms/token) by one expert-sorted pass at ~0.4 ms/token.
#
# Knobs (environment, read once by ModelArgs):
#   SOLAR_OPEN_BATCHED_PREFILL=1              turn the batched path on (default OFF: byte-identical to phase 2)
#   SOLAR_OPEN_BATCHED_PREFILL_TOKENS=<n>     token budget of one packed pass, B_mb x S <= n (default 1024 = 8 x 128;
#                                             4096 = 32 x 128 in one pass; clamped to 256..8192, see the v1 cap below)
#   SOLAR_OPEN_BATCHED_PREFILL_MAX_SEQ_LEN=<n> largest PADDED per-user prefill length that is packed (default 128:
#                                             the 1K+ buckets already run the sorted MoE per user at a flat per-token
#                                             cost, so packing them buys little TTFT-last and costs TTFT-mean)
BATCHED_PREFILL_MIN_TOKENS = 256
# v1 runs the tt_transformers batched path, whose head applies norm + lm_head to ALL B*S rows ([T, 24576] bf16 per
# device = 50 MiB at 1K, 201 MiB at 4K, 400 MiB at 8K) and reads one 32-row tile per user back; 8K tokens per pass
# is the DRAM-safe cap until a gather head exists (design 2.6).
BATCHED_PREFILL_MAX_TOKENS = 8 * 1024
BATCHED_PREFILL_DEFAULT_TOKENS = 1024
BATCHED_PREFILL_DEFAULT_MAX_SEQ_LEN = 128
# Device batch sizes the tt_transformers batched-prefill path pads a pass to (generator.SUPPORTED_PREFILL_BATCH_SIZES).
BATCHED_PREFILL_BATCH_SIZES = (1, 2, 4, 8, 16, 32)


def _env_flag(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not an integer") from exc


def padded_prefill_batch(num_users):
    """Rows the tt_transformers batched-prefill pass allocates for ``num_users`` users re-slotted to ``0..num_users-1``
    (the smallest supported device batch >= the user count; the parent pads tokens / page-table rows up to it)."""
    num_users = int(num_users)
    return next((b for b in BATCHED_PREFILL_BATCH_SIZES if b >= num_users), num_users)


@dataclasses.dataclass(frozen=True)
class BatchedPrefillOptions:
    """Policy knobs of the packed multi-user prefill (see the module comment above for the env names)."""

    enabled: bool = False
    tokens_per_pass: int = BATCHED_PREFILL_DEFAULT_TOKENS
    max_seq_len: int = BATCHED_PREFILL_DEFAULT_MAX_SEQ_LEN

    def __post_init__(self):
        if self.tokens_per_pass % 32 != 0 or not (
            BATCHED_PREFILL_MIN_TOKENS <= self.tokens_per_pass <= BATCHED_PREFILL_MAX_TOKENS
        ):
            raise ValueError(
                f"tokens_per_pass must be a multiple of 32 in [{BATCHED_PREFILL_MIN_TOKENS}, "
                f"{BATCHED_PREFILL_MAX_TOKENS}], got {self.tokens_per_pass}"
            )
        if self.max_seq_len < 32 or self.max_seq_len % 32 != 0:
            raise ValueError(f"max_seq_len must be a positive multiple of 32, got {self.max_seq_len}")

    @classmethod
    def from_env(cls):
        enabled = _env_flag("SOLAR_OPEN_BATCHED_PREFILL", False)
        tokens = _env_int("SOLAR_OPEN_BATCHED_PREFILL_TOKENS", BATCHED_PREFILL_DEFAULT_TOKENS)
        clamped = min(max(tokens, BATCHED_PREFILL_MIN_TOKENS), BATCHED_PREFILL_MAX_TOKENS)
        if clamped != tokens:
            logger.warning(
                f"SOLAR_OPEN_BATCHED_PREFILL_TOKENS={tokens} clamped to {clamped} (the v1 batched path runs lm_head on "
                f"every token of a pass; {BATCHED_PREFILL_MAX_TOKENS} tokens is its DRAM-safe cap)"
            )
        max_seq_len = _env_int("SOLAR_OPEN_BATCHED_PREFILL_MAX_SEQ_LEN", BATCHED_PREFILL_DEFAULT_MAX_SEQ_LEN)
        return cls(enabled=enabled, tokens_per_pass=clamped, max_seq_len=max_seq_len)

    def users_per_pass(self, seq_len):
        """Users of padded length ``seq_len`` that fit one pass (0 when the length is not packed at all)."""
        seq_len = int(seq_len)
        if seq_len > self.max_seq_len or seq_len > self.tokens_per_pass:
            return 0
        return self.tokens_per_pass // seq_len

    def describe(self):
        return (
            f"batched prefill {'ON' if self.enabled else 'OFF'} (tokens_per_pass={self.tokens_per_pass}, "
            f"max_seq_len={self.max_seq_len})"
        )


@dataclasses.dataclass(frozen=True)
class BatchedPrefillPass:
    """One packed pass: ``users`` (positions in the caller's request batch, re-slotted to ``0..len-1`` on the device)
    sharing the padded per-user length ``seq_len``."""

    users: tuple
    seq_len: int

    @property
    def padded_batch(self):
        return padded_prefill_batch(len(self.users))

    @property
    def num_tokens(self):
        return self.padded_batch * self.seq_len


@dataclasses.dataclass(frozen=True)
class BatchedPrefillPlan:
    """Packed passes (shortest bucket first) plus the users that stay on the per-user path."""

    passes: tuple
    sequential: tuple

    @property
    def packed_users(self):
        return tuple(u for p in self.passes for u in p.users)


def plan_batched_prefill(prompt_lens, options: BatchedPrefillOptions, num_cached=None):
    """Group the users of one prefill call into packed passes.

    A user is packable when it has no cached prefix and its padded prefill length (``get_padded_prefill_len``, the
    bucket the tt_transformers Generator pads to: 128 / 1024 / 2K / ...) is at most ``options.max_seq_len``. Users of
    one bucket are packed in request order, ``options.users_per_pass(S)`` per pass, the remainder forming a smaller
    last pass; a bucket with a single packable user is not a pass (the Generator's batched path needs B > 1 and the
    per-user path keeps its 128-token trace). Buckets are ordered shortest first so the many short users are not held
    behind a long one. Users that are not packed are returned in request order in ``sequential``.
    """
    lens = [int(n) for n in (prompt_lens.tolist() if isinstance(prompt_lens, torch.Tensor) else prompt_lens)]
    cached = [0] * len(lens) if num_cached is None else [int(n) for n in num_cached]
    if len(cached) != len(lens):
        raise ValueError(f"num_cached has {len(cached)} entries for {len(lens)} users")
    if not options.enabled:
        return BatchedPrefillPlan(passes=(), sequential=tuple(range(len(lens))))

    buckets = {}
    sequential = []
    for user, (length, n_cached) in enumerate(zip(lens, cached)):
        padded = get_padded_prefill_len(length - n_cached) if length > n_cached else None
        if n_cached != 0 or padded is None or options.users_per_pass(padded) < 2:
            sequential.append(user)
            continue
        buckets.setdefault(padded, []).append(user)

    passes = []
    for seq_len in sorted(buckets):
        users = buckets[seq_len]
        if len(users) < 2:
            sequential.extend(users)
            continue
        per_pass = options.users_per_pass(seq_len)
        for start in range(0, len(users), per_pass):
            chunk = users[start : start + per_pass]
            if len(chunk) == 1:
                sequential.append(chunk[0])  # a trailing single user runs on the per-user path (traceable at 128)
            else:
                passes.append(BatchedPrefillPass(users=tuple(chunk), seq_len=seq_len))
    return BatchedPrefillPlan(passes=tuple(passes), sequential=tuple(sorted(sequential)))


class ModelArgs:
    """Solar-Open ModelArgs compatible with the tt_transformers create_tt_model / Generator interface."""

    def __init__(
        self,
        mesh_device,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,
    ):
        self.mesh_device = mesh_device
        self.dummy_weights = dummy_weights
        self.max_batch_size = max_batch_size
        if self.max_batch_size > 32:
            # More than 32 users needs users_row_sharded: each mesh row decodes its own 32-user
            # slice (nlp_create_qkv_heads_decode / nlp_concat_heads_decode / on-device sampling all
            # cap at 32 users per device).
            if self.mesh_device.shape[0] == 1:
                raise ValueError(
                    f"max_batch_size={self.max_batch_size} exceeds the 32 users a single mesh row can decode; "
                    "single-row meshes (e.g. 1x8) support batch sizes up to 32."
                )
            assert (
                self.max_batch_size % self.mesh_device.shape[0] == 0
            ), "max_batch_size must be divisible by the number of device rows"
            self.max_local_batch_size = self.max_batch_size // self.mesh_device.shape[0]
        else:
            self.max_local_batch_size = self.max_batch_size
        self.max_seq_len = max_seq_len
        if optimizations is not None:
            logger.warning("Solar-Open doesn't support any performance optimizations - ignoring optimizations argument")
        self.optimizations = None
        self.cache_hf = cache_hf

        # Checkpoint location: HF_MODEL (tt_transformers standard) or the HF repo id.
        model_path = os.getenv("HF_MODEL", DEFAULT_HF_MODEL)
        self.model_path = model_path
        self.weights_path = model_path

        logger.info(
            f"Using Solar-Open model from: {self.model_path}"
            f"{' (dummy weights — no checkpoint load)' if self.dummy_weights else ''}"
        )

        # MoE knobs (expert dtypes, router impl) from the SOLAR_OPEN_* environment; set in BOTH branches because the
        # expert dtype is part of the weight-cache directory name (weight_cache_path is used by dummy-weight tests too).
        self.moe_options = MoEOptions.from_env()

        if self.dummy_weights:
            # Skip loading the HF config for testing; tests build their own AutoConfig from model_path.
            logger.info("Using dummy weights mode - skipping HuggingFace config loading")
            self.hf_config = None
            self.vocab_size = None
            self.n_layers = None
            self.head_dim = None
            self.rope_theta = None
            self.rope_scaling = None
        else:
            # SolarOpenConfig is native in transformers >= 5.12 (model_type "solar_open").
            self.hf_config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            self.vocab_size = self.hf_config.vocab_size  # 196608
            self.n_layers = self.hf_config.num_hidden_layers  # 48
            # head_dim is an explicit config field (128); hidden_size // num_attention_heads would give 64.
            self.head_dim = (
                getattr(self.hf_config, "head_dim", None)
                or self.hf_config.hidden_size // self.hf_config.num_attention_heads
            )
            # Informational: Model resolves theta / YaRN itself from hf_config (rope_parameters on transformers 5.x).
            self.rope_theta = resolve_rope_theta(self.hf_config)  # 1e6
            self.rope_scaling = dict(getattr(self.hf_config, "rope_parameters", None) or {})

        # Attributes the Generator expects
        self.max_prefill_chunk_size = 128 * 1024
        self.model_name = Path(self.model_path).name
        assert self.model_name == MODEL_NAME, (
            f"Unrecognized model name {self.model_name!r} inferred from model path {self.model_path}. "
            f"Use HF_MODEL={DEFAULT_HF_MODEL} (HF repo id) or a directory/symlink literally named {MODEL_NAME} "
            "(HF hub snapshot directories are hashes, so symlink them)."
        )
        self.max_context_len = max_seq_len  # Context length for tt_transformers compatibility

        if self.dummy_weights:
            # Skip tokenizer loading for testing
            self.tokenizer = None
            self.processor = None
            self.stop_token_ids = set()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path, trust_remote_code=True)
            self.processor = None  # text-only model, no vision processor
            self.stop_token_ids = self._load_stop_token_ids()

        # Phase 3a(1): packed / batched multi-user prefill (module comment above, design_packed_prefill.md). The
        # tt_transformers Generator batches equal-length users into one forward when ``disable_batched_prefill`` is
        # False; the Solar policy (which users, how many per pass, never traced) lives in ``plan_batched_prefill`` and
        # ``tt/model.py: prefill_forward_text_batched``. Default OFF (SOLAR_OPEN_BATCHED_PREFILL=1 flips it): with the
        # flag on, a plain Generator call with several equal-length users runs one batched pass (host sampling only --
        # the on-device sampling contract of the batched path is not implemented, keep sampling_params=None) and any
        # attempt to TRACE such a pass is refused (packed_prefill_trace_shapes).
        self.batched_prefill = BatchedPrefillOptions.from_env()
        self.disable_batched_prefill = not self.batched_prefill.enabled
        # (padded_batch, per-user seq_len) pairs whose batched prefill may be traced. Empty until the traced-prefill
        # work provides a trace-safe MoE for T = B x S: the expert-sorted MoE reads the per-expert counts back to the
        # host (experts/prefill.py _sorted_moe_plan) and the router's [T, E] helpers are transient above 128 rows, so a
        # captured 8 x 128 pass would replay garbage. Model.prepare_prefill_inputs_trace refuses any other batched
        # capture; prefill_forward_text_batched runs those passes eagerly. Owned by the traced-prefill implementer.
        self.packed_prefill_trace_shapes = set()
        self.capped_warmup_seq_len = 2048
        self.trace_prefill_supported_seq_lens = self.get_trace_prefill_supported_seq_lens()

    def _load_stop_token_ids(self) -> set:
        """Generation stop set: generation_config.json ``eos_token_id`` (list or int) union the tokenizer's eos.

        Solar stops on {2 <|endoftext|>, 24 <|flush|>, 25 <|calls|>} while ``tokenizer.eos_token_id`` is 2 only and
        ``<|end|>`` (21) closes a message without ending the turn - the demo must use this set, not the tokenizer eos.
        Falls back to ``{tokenizer.eos_token_id}`` when generation_config.json is unavailable.
        """
        stop_ids = set()
        tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
        if tokenizer_eos is not None:
            stop_ids.add(int(tokenizer_eos))
        try:
            eos = GenerationConfig.from_pretrained(self.model_path).eos_token_id
        except Exception as e:  # missing generation_config.json, offline hub miss, ...
            logger.warning(
                f"Could not read generation_config.json from {self.model_path} ({type(e).__name__}: {e}); "
                f"stop tokens fall back to the tokenizer eos {sorted(stop_ids)}"
            )
            return stop_ids
        if eos is not None:
            stop_ids.update(int(t) for t in ([eos] if isinstance(eos, int) else eos))
        logger.info(f"Solar-Open stop token ids: {sorted(stop_ids)}")
        return stop_ids

    def get_warmup_prefill_supported_seq_lens(self):
        DEFAULT_VALUE = self.capped_warmup_seq_len
        # This dictionary is used to override the default ceil warmup prefill value
        model_specific_ceil_warmup_lengths = {
            # e.g. "Solar-Open-100B": 4096
        }

        max_seq_len_to_warmup = model_specific_ceil_warmup_lengths.get(self.base_model_name, DEFAULT_VALUE)
        if max_seq_len_to_warmup > self.capped_warmup_seq_len:
            max_seq_len_to_warmup = self.capped_warmup_seq_len

        to_warmup_seq_lens = calculate_prefill_warmup_seq_lens(
            max_seq_len_to_warmup, self.trace_prefill_supported_seq_lens
        )

        to_warmup_seq_lens = self.filter_warmup_seq_lens(to_warmup_seq_lens)

        return to_warmup_seq_lens

    def filter_warmup_seq_lens(self, to_warmup_seq_lens):
        # Warmup lengths come from https://github.com/tenstorrent/tt-metal/pull/33143; single-user prefill on 1x8 is
        # capped at 64K (demo limitation), so nothing at or above 64K is warmed up.
        for seq_len in to_warmup_seq_lens:
            if seq_len >= 64 * 1024:
                to_warmup_seq_lens = to_warmup_seq_lens[: to_warmup_seq_lens.index(seq_len)]
                break
        return to_warmup_seq_lens

    @property
    def base_model_name(self):
        return get_base_model_name(self.model_name)

    def can_enable_trace(self, prefill_seq_len, num_cached_tokens=0):
        """
        This function is used to determine if trace should be enabled for the prefill.
        Tracing is used only for certain sequence lengths, because for bigger sequence lengths, op2op gaps are already small, so we don't need tracing.
        # TODO: Support chunked prefill with tracing - https://github.com/tenstorrent/tt-metal/issues/32056
        """

        allowed_seq_lens = self.trace_prefill_supported_seq_lens

        return (
            prefill_seq_len in allowed_seq_lens
            and prefill_seq_len <= self.max_prefill_chunk_size
            and prefill_seq_len <= self.max_seq_len
            and num_cached_tokens == 0
        )

    def can_enable_batched_prefill_trace(self, padded_batch, prefill_seq_len):
        """Whether a packed pass of ``padded_batch`` users x ``prefill_seq_len`` tokens may replay a trace.

        Only shapes listed in ``packed_prefill_trace_shapes`` (none today, see __init__) and whose per-user length is
        itself a traced length. ``prefill_forward_text_batched`` forces eager execution for every other pass; the
        Generator's own ``can_enable_trace(prefill_seq_len)`` cannot see the batch and would trace a 32 x 128 pass.
        """
        return (int(padded_batch), int(prefill_seq_len)) in self.packed_prefill_trace_shapes and self.can_enable_trace(
            int(prefill_seq_len)
        )

    def get_trace_prefill_supported_seq_lens(self):
        """Prefill lengths whose forward is captured in a trace on this (model, device): the TRACE_PREFILL_SEQ_LENS
        entry, capped at ``capped_warmup_seq_len`` (a 4096 entry is silently dropped unless the cap is raised too) and
        checked against the experts' path selection: every split of a traced length must take a trace-safe MoE path
        (``is_trace_safe_prefill_len``), else the capture would record the host-planned sorted path's plan for ONE
        prompt and replay it for all others. No default for other (model, device) pairs: eager prefill."""
        # TODO: https://github.com/tenstorrent/tt-metal/issues/32818
        result = TRACE_PREFILL_SEQ_LENS.get(self.model_name, {}).get(determine_device_name(self.mesh_device))
        if result is None:
            return []
        seq_lens = cap_seq_lens_to_max_prefill_chunk_size(list(result), self.capped_warmup_seq_len)
        self._check_trace_safe_prefill_lens(seq_lens)
        return seq_lens

    def _check_trace_safe_prefill_lens(self, seq_lens):
        """Raise ValueError when a traced prefill length has a split the experts would run on the host-planned
        expert-sorted MoE path (see TRACE_PREFILL_SEQ_LENS). Uses the shipped SolarOpenProgramConfig defaults (the
        chunk / split sizes and trace_safe_split_lens do not depend on the compute grid) and the expert count of the
        HF config (Solar's 128 without one); the fused shared expert's always-on slot does not change the selection."""
        program_config = SolarOpenProgramConfig()
        num_experts = self.hf_config.num_local_experts if self.hf_config is not None else _SOLAR_NUM_EXPERTS
        dense_moe = self.mesh_device.shape[0] == 1  # single-row meshes run the dense (EP=1) prefill MoE
        for seq_len in seq_lens:
            if not is_trace_safe_prefill_len(seq_len, program_config, num_experts, dense_moe=dense_moe):
                raise ValueError(
                    f"traced prefill length {seq_len} is not trace-safe: a split of it takes the host-planned "
                    "expert-sorted MoE path (a device->host read per split, experts/prefill.py _sorted_moe_plan). "
                    "List the split length in SolarOpenProgramConfig.trace_safe_split_lens (static per-expert loop) "
                    "or provide a trace-safe sorted plan before adding the length to TRACE_PREFILL_SEQ_LENS."
                )

    @property
    def batched_prefill_token_counts(self):
        """Row counts ``T = padded_batch x per-user length`` of the packed multi-user prefill passes
        (``BatchedPrefillOptions``: every padded prefill length <= ``max_seq_len`` times every device batch size >= 2
        that fits the pass token budget); empty while the batched prefill is off. The router keeps its ``[T, E]``
        helpers for them (``router_persistent_token_counts``) so a pass neither rebuilds them per layer nor -- once its
        shape is admitted to ``packed_prefill_trace_shapes`` -- allocates them inside a capture."""
        options = getattr(self, "batched_prefill", None)
        if options is None or not options.enabled:
            return []
        counts = set()
        for seq_len in get_all_padded_prefill_lengths(options.max_seq_len):
            for batch in BATCHED_PREFILL_BATCH_SIZES:
                if batch >= 2 and batch * seq_len <= options.tokens_per_pass:
                    counts.add(batch * seq_len)
        return sorted(counts)

    @property
    def router_persistent_token_counts(self):
        """Token counts whose ``[T, E]`` router helpers ``TopKRouter`` prebuilds and keeps (design_traced_prefill.md
        6.1): the decode batch, the batched-decode padding width 32, every traced prefill length and the packed-prefill
        row counts. ``Model`` hands it to every layer's router (``TopKRouter(persistent_token_counts=...)``): a token
        count captured in a trace with transient helpers would record their allocation into the trace."""
        counts = {
            self.max_local_batch_size,
            32,
            *self.trace_prefill_supported_seq_lens,
            *self.batched_prefill_token_counts,
        }
        return sorted(counts)

    def encode_prompt(self, prompt_text, instruct=False, system_prompt_text=None, **template_kwargs):
        """Encode a prompt (str, or an already-built list of chat messages) through Solar's chat template.

        Returns a flat list[int]. The template is always applied (``add_generation_prompt=True`` appends
        ``<|begin|>assistant``); the tt_transformers ``instruct`` flag is kept for interface compatibility only.
        Template kwargs default to ``reasoning_effort=$SOLAR_OPEN_REASONING_EFFORT`` ("high"; "low"/"minimal"
        prepend an empty think block) and ``default_system_prompt=$SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT`` ("1");
        explicit ``template_kwargs`` win. The template injects the current date via ``strftime_now``, so token ids
        change from day to day - compare golden outputs at the string level.
        """
        assert not instruct, "Solar-Open always applies its chat template; instruct=True is not a separate mode"
        kw = {
            "reasoning_effort": os.getenv("SOLAR_OPEN_REASONING_EFFORT", "high"),
            "default_system_prompt": os.getenv("SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT", "1") == "1",
            **template_kwargs,
        }
        if isinstance(prompt_text, str):
            chat = []
            if system_prompt_text:
                chat.append({"role": "system", "content": system_prompt_text})
            if prompt_text:
                chat.append({"role": "user", "content": prompt_text})
        else:
            # prompt_text is already a list of chat messages
            chat = prompt_text
        encoded = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, **kw)

        # Normalize whatever apply_chat_template(tokenize=True) returns into a flat List[int].
        # Across tokenizer/transformers versions this may be a List[int], a tokenizers.Encoding,
        # a list of Encodings, or a BatchEncoding/dict ({"input_ids": ...}); a non-list form makes
        # downstream torch.tensor(...) raise "Could not infer dtype of tokenizers.Encoding".
        raw_type = type(encoded).__name__
        # BatchEncoding / dict -> take input_ids
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"] if "input_ids" in encoded else getattr(encoded, "input_ids")

        def _to_ids(obj):
            if hasattr(obj, "ids"):  # tokenizers.Encoding
                return list(obj.ids)
            if isinstance(obj, (list, tuple)):
                flat = []
                for item in obj:
                    flat.append(item) if isinstance(item, int) else flat.extend(_to_ids(item))
                return flat
            return obj

        encoded = _to_ids(encoded)
        if not (isinstance(encoded, list) and (len(encoded) == 0 or isinstance(encoded[0], int))):
            logger.warning(
                f"[solar-open encode_prompt] unexpected token container: raw={raw_type}, "
                f"normalized={type(encoded).__name__}"
            )
        return encoded

    @staticmethod
    def load_state_dict(weights_path, dummy_weights=False, convert_to_meta_format=True):
        """Load the model state dict in the layout every TT module expects (contract C1).

        Phase 1 loads the whole checkpoint with ``AutoModelForCausalLM.from_pretrained(dtype=torch.bfloat16)``
        (~205 GB host RSS; one-time thanks to the ttnn weight cache). transformers >= 5.12 applies the
        ``solar_open -> qwen2_moe`` conversion mapping, so the 384 per-expert tensors of every layer arrive fused as
        ``mlp.experts.gate_up_proj [128, 2560, 4096]`` (rows 0:1280 gate, 1280:2560 up - NOT interleaved) and
        ``mlp.experts.down_proj [128, 4096, 1280]``; ``SolarOpenDecoderLayer(config).state_dict()`` yields the same
        keys, so one weight path serves tests and demo. ``mlp.gate.e_score_correction_bias`` is fp32 on disk and
        MUST stay fp32 (the router adds it to fp32 sigmoid scores; |bias| <= 0.01 is below the bf16 ulp near 0.5).

        ``SOLAR_OPEN_STREAMING_LOAD=1`` (phase 2, DESIGN.md 4.16) returns a ``utils.streaming_loader.LazyStateDict``
        instead: the same contract-C1 keys (fused expert tensors, Meta-permuted q/k, fp32 bias) over the safetensors
        shards, each tensor read on access and never retained, so a cold cache build peaks at one layer's transients
        instead of 393 GB. The returned object is a ``Mapping``, ``substate()``-aware, and bit-identical to the
        phase-1 tensors (tests/unit/test_streaming_loader.py); it must never go through the dict rebuild below.

        Args:
            weights_path (str or Path): checkpoint directory, symlink or HF repo id.
            dummy_weights (bool): If True, returns ``{}`` (every module then builds random / cached tensors).
            convert_to_meta_format (bool): If True, permute q_proj/k_proj to Meta RoPE order (head_dim 128).
                Set to False when loading for HuggingFace reference models.
        """
        if dummy_weights:
            return {}
        if os.getenv("SOLAR_OPEN_STREAMING_LOAD") == "1":
            # Phase 2: per-layer streaming over the safetensors shards (DESIGN.md 4.16). head_dim / num_experts come
            # from the HF config (head_dim is an explicit field: hidden_size // num_attention_heads would give 64).
            from models.demos.solar_open.utils.streaming_loader import LazyStateDict

            hf_config = AutoConfig.from_pretrained(weights_path, trust_remote_code=True)
            head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
            state_dict = LazyStateDict(
                weights_path,
                head_dim=head_dim,
                num_experts=hf_config.num_local_experts,
                convert_to_meta=convert_to_meta_format,
            )
            _validate_state_dict_layout(state_dict)  # metadata only on a LazyStateDict (meta()); no tensor is read
            logger.info(
                f"Streaming loader: {len(state_dict)} tensors over {state_dict.num_shards} shards from "
                f"{state_dict.snapshot_dir}; tensors are read per access (peak host RSS ~ one layer)"
            )
            return state_dict

        # bf16 straight from disk (the checkpoint is bf16): no fp32 intermediate, and the safety-net cast below is a
        # no-op that rebinds references instead of copying 205 GB. `dtype=` is the transformers 5.x spelling
        # (`torch_dtype` is deprecated, `low_cpu_mem_usage` is ignored).
        model = AutoModelForCausalLM.from_pretrained(weights_path, dtype=torch.bfloat16)
        head_dim = model.config.head_dim
        state_dict = model.state_dict()
        # Drop the HF module graph/buffers now that we hold the weight tensors. state_dict shares
        # storage with the params, so after this the peak host footprint is bounded by the bf16
        # weights themselves rather than weights + a live HF model object.
        del model
        gc.collect()
        if convert_to_meta_format:
            logger.info("Converting QKV weights from HuggingFace to Meta format for RoPE")
            state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
        # Safety net: ensure bf16 weights. With the bf16 load above this is skipped entirely; it only casts
        # genuine fp32 stragglers and never the router bias (contract C1).
        if state_dict["model.norm.weight"].dtype != torch.bfloat16:
            state_dict = {
                k: (v.to(torch.bfloat16) if v.dtype == torch.float32 and not k.endswith(ROUTER_BIAS_SUFFIX) else v)
                for k, v in tqdm(state_dict.items(), desc="Converting to bfloat16")
            }
        _validate_state_dict_layout(state_dict)
        return state_dict

    # ttnn weight cache: dtype string used in the directory name (attention / lm_head dtype; the expert dtype is
    # appended separately from MoEOptions so bfp8 and bfp4 expert caches never share .tensorbin files).
    _DTYPE_STR = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}

    def weight_cache_root(self) -> Path:
        """Directory holding the tensor caches: TT_CACHE_PATH if set, else the checkpoint directory when HF_MODEL is
        a local directory, else ~/.cache/tenstorrent/<model_name> (an HF repo id must never mkdir "upstage/..." in
        the CWD). Does not create anything."""
        env_cache_dir = os.getenv("TT_CACHE_PATH")
        if env_cache_dir:
            return Path(env_cache_dir)
        model_dir = Path(self.model_path)
        if model_dir.is_dir():
            return model_dir
        return Path.home() / ".cache" / "tenstorrent" / self.model_name

    def weight_cache_path(self, dtype):
        """Cache directory for (dtype, expert dtype, mesh shape), created on demand:
        ``<root>/tensor_cache_{bf16|bfp8|bfp4}_exp{bfp8|bfp4}_(rows, cols)``."""
        dtype_str = self._DTYPE_STR[dtype]
        mesh_shape = tuple(self.mesh_device.shape)  # "(1, 8)" for both ttnn.MeshShape and plain tuples
        cache_path = (
            self.weight_cache_root() / f"tensor_cache_{dtype_str}_exp{self.moe_options.expert_dtype_str}_{mesh_shape}"
        )
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory: {cache_path}")
        return cache_path

    # Name of the marker file dropped into a weight-cache directory once every weight
    # for that (model, dtype, expert dtype, mesh shape) has been materialized to disk.
    WEIGHT_CACHE_MARKER = ".weights_complete"
    # Cache-format version embedded in the marker. Bump this whenever the set/naming/layout of
    # cached weight tensors changes in a way that an existing cache would not satisfy (e.g. a
    # weight tensor is added or renamed without changing the layer count). A marker written by an
    # older format is then rejected -> the run cold-loads and regenerates the cache, rather than
    # skipping the load and hard-failing in ttnn.as_tensor(None, ...) on a missing .tensorbin.
    # (Mirrors DeepSeek's WEIGHT_CACHE_FORMAT_VERSION in deepseek_v3/utils/weight_config.py.)
    # v3: expert gate/up projections cached fused (experts/weights.py gate_up_proj_fused_tp*).
    # v4: Solar-Open tensor set - no bias/sink files, router under mlp/gate/ (weight + e_score_correction_bias),
    #     new mlp/shared_experts/{gate_proj,up_proj,down_proj}_tp*, marker records the MoE options.
    WEIGHT_CACHE_FORMAT_VERSION = 4

    def weight_cache_is_complete(self, dtype):
        """True when the on-disk ttnn weight cache for this (model, dtype, expert dtype, mesh shape) was
        fully built by a previous run with the same MoE options.

        When True, ttnn.as_tensor loads every weight from its cached .tensorbin and the HF
        state_dict is never read, so the caller can skip the expensive from_pretrained host
        load entirely (the load that OOMs/hangs during prefill, #48509) without needing the
        manual --skip-model-load flag. Set SOLAR_OPEN_FORCE_MODEL_LOAD=1 to force a fresh load
        (e.g. to regenerate the cache)."""
        if os.getenv("SOLAR_OPEN_FORCE_MODEL_LOAD") == "1":
            return False
        cache_path = self.weight_cache_path(dtype)
        marker = cache_path / self.WEIGHT_CACHE_MARKER
        if not marker.is_file():
            return False
        try:
            meta = json.loads(marker.read_text())
        except (ValueError, OSError):
            return False
        # Reject a stale marker: an older cache format, a different model, a partial
        # (num_layers-limited) build whose cache does not cover the full model we are about to
        # construct, or tensors built with different MoE options (router impl / dtypes). A rejected
        # marker falls back to a cold load (which regenerates the cache) rather than skipping the
        # load and crashing on a missing/renamed .tensorbin.
        if meta.get("format_version") != self.WEIGHT_CACHE_FORMAT_VERSION:
            return False
        # A cache built for MORE layers than requested (a full 48-layer cache serving a num_layers-limited debug run)
        # holds every tensorbin the partial model needs; only a cache built for FEWER layers is stale.
        if meta.get("model_name") != self.model_name or (meta.get("n_layers") or 0) < self.n_layers:
            return False
        if meta.get("moe") != self.moe_options.marker_fields():
            return False
        # Belt-and-suspenders: the cache dir must still actually hold tensor files.
        return any(cache_path.glob("*.tensorbin"))

    def mark_weight_cache_complete(self, dtype):
        """Record that the ttnn weight cache for this (model, dtype, expert dtype, mesh shape) was fully
        built, so subsequent runs can skip the HF state_dict load (see weight_cache_is_complete)."""
        cache_path = self.weight_cache_path(dtype)
        marker = cache_path / self.WEIGHT_CACHE_MARKER
        try:
            marker.write_text(
                json.dumps(
                    {
                        "format_version": self.WEIGHT_CACHE_FORMAT_VERSION,
                        "model_name": self.model_name,
                        "n_layers": self.n_layers,
                        "dtype": str(dtype),
                        "moe": self.moe_options.marker_fields(),
                    }
                )
            )
            logger.info(f"Marked ttnn weight cache complete: {marker}")
        except OSError as e:
            logger.warning(f"Could not write weight-cache completion marker {marker}: {e}")

    def get_model_config(self):
        """Return model configuration dict"""
        return {
            "vocab_size": self.vocab_size,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "max_batch_size": self.max_batch_size,
        }

    def get_state_dict_prefix(self, prefix, layer_idx):
        """Get state dict prefix for layer weights"""
        if layer_idx is None:
            return prefix
        return f"{prefix}layers.{layer_idx}."

    @property
    def max_grid_size(self):
        """Return maximum grid size for the device"""
        return ttnn.CoreGrid(y=8, x=8)  # Standard grid size


def _shape_dtype(state_dict, key):
    """``(shape, dtype)`` of ``state_dict[key]``: from the headers on a ``LazyStateDict`` (``meta()``, no tensor bytes
    read - a layer-0 expert tensor is 2.7 GB), from the tensor on a plain dict."""
    if hasattr(state_dict, "meta"):
        shape, dtype = state_dict.meta(key)
        return tuple(shape), dtype
    tensor = state_dict[key]
    return tuple(tensor.shape), tensor.dtype


def _validate_state_dict_layout(state_dict):
    """Fail loudly, before any device work, if the checkpoint did not arrive in the contract-C1 layout.

    The most likely misconfiguration is an older transformers that leaves the 128 per-expert
    ``mlp.experts.{e}.*`` tensors unfused; the TT expert loader would only notice after the mesh is open.
    Works on a plain dict and on the streaming ``LazyStateDict`` (metadata only, see ``_shape_dtype``).
    """
    layer0 = "model.layers.0."
    required = (
        "self_attn.q_proj.weight",
        "mlp.gate.weight",
        f"mlp.gate.{ROUTER_BIAS_SUFFIX}",
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        "mlp.shared_experts.gate_proj.weight",
    )
    missing = [layer0 + k for k in required if layer0 + k not in state_dict]
    if missing:
        raise ValueError(
            f"State dict is not in the fused transformers>=5 Solar-Open layout (missing {missing}). Per-expert "
            "mlp.experts.{e}.gate_proj/up_proj/down_proj keys need transformers >= 5.12 (solar_open -> qwen2_moe "
            "conversion mapping fuses them to gate_up_proj [E, 2I, H] / down_proj [E, H, I])."
        )
    gate_up_shape, _ = _shape_dtype(state_dict, layer0 + "mlp.experts.gate_up_proj")
    down_shape, _ = _shape_dtype(state_dict, layer0 + "mlp.experts.down_proj")
    if (
        len(gate_up_shape) != 3
        or len(down_shape) != 3
        or gate_up_shape[0] != down_shape[0]  # E
        or gate_up_shape[1] != 2 * down_shape[2]  # 2I vs I
        or gate_up_shape[2] != down_shape[1]  # H
    ):
        raise ValueError(
            f"Expected mlp.experts.gate_up_proj [E, 2I, H] and down_proj [E, H, I]; got {gate_up_shape} and "
            f"{down_shape}"
        )
    _, bias_dtype = _shape_dtype(state_dict, layer0 + f"mlp.gate.{ROUTER_BIAS_SUFFIX}")
    if bias_dtype != torch.float32:
        raise ValueError(f"mlp.gate.{ROUTER_BIAS_SUFFIX} must stay fp32 (contract C1), got {bias_dtype}")


def determine_device_name(mesh_device):
    """
    Determine device name based on number of devices and architecture.

    Args:
        mesh_device (MeshDevice): MeshDevice object

    Returns:
        str: Device name (e.g., "CPU", "N150", "P100", etc.)

    Raises:
        ValueError: If architecture or device count is unsupported
    """
    num_devices = mesh_device.get_num_devices() if mesh_device else 0
    arch_name = ttnn.get_arch_name()
    dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None  # CoreCoord with (x, y)

    if num_devices == 0:
        return "CPU"

    if is_blackhole():
        dict_device_names = {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",  # P100 DRAM grid is 7x1, P150 is 8x1
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }
    elif is_wormhole_b0():
        dict_device_names = {
            1: "N150",
            2: "N300",
            4: "N150x4",
            8: "T3K",
            32: "TG",
        }
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    if num_devices in dict_device_names:
        return dict_device_names[num_devices]
    else:
        raise ValueError(f"Unsupported number of devices: {num_devices} for {arch_name}")
