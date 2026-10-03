# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only contract tests for the Motif-3 vLLM bridge (design 00 §4.7, §5.1; study 05 §3, §7; features design
``docs/features/FEATURES_DESIGN.md`` §1.2, §1.5, §2.3, §3.9, §5.1).

Nothing here opens a device. Run them in a device-hidden namespace (the module skips itself otherwise)::

    scripts/hostrun.sh -n bridge -- python -m pytest -p no:cacheprovider -q \\
        models/demos/motif3/tests/test_generator_vllm_host.py

Coverage:

* import hygiene: the bridge imports no ttnn, no vLLM, no other ``models/demos`` package and not the TT runtime;
* registration: ``TT_MODEL_CLASS_OVERRIDES`` registers the bare and the ``TT`` names; ``vllm_metadata.json`` /
  ``EXTRA_MODELS_DIR`` registers only ``TTMotifForCausalLM`` (the bare-name trap); vLLM's registry inspection
  classifies the class as a plain text-generation model;
* the real chain on the Motif checkpoint config: ``ModelConfig`` (trust_remote_code) resolves ``MotifForCausalLM`` to
  the bridge, ``VllmConfig`` runs ``TTPlatform.check_and_update_config`` with our capabilities, the plugin sizes the
  pool, calls ``get_kv_cache_spec``, vLLM builds the KV config, the runner derives the ``(N, 1, 64, 576)`` hint and
  the bridge allocates it; vLLM's ``BlockPool`` then holds 32 users x (8192 tokens + 1 block) after its null block;
  the tokenizer resolves with the reasoning / tool tokens as single ids;
* pool math, spec and allocation contracts, warmup order, release hooks, decode-reload contract v1 rejections;
* prefill/decode plumbing through the bridge with a fake ``MotifGenerator`` that emulates the device semantics the
  bridge relies on (KV replicated per DP group, decode writes only its lane's group unless KV-R is on, bucket-padded
  prefill writes, resumed prefill planned with ``tt/prefill_plan.py`` and run writer-first, packed verify with idle
  partner lanes, a deterministic fake MTP), driven by the plugin's own state-slot bookkeeping
  (``TTModelRunner._alloc_prefill_state_slots`` / ``_decode_state_slot_remap`` / ...), its speculative helpers
  (``_spec_row_state``, ``_step_verifies``, ``_spec_candidate_block``, ``accept_greedy_drafts``,
  ``_committed_positions``) and by block-table rows carrying stale ids, as vLLM's reused rows do;
* the features (§5.1 bridge suite): capabilities per feature switch, ``spec_plan`` (plan, refusals, the PS-1 guard,
  the per-chip MTP bytes), the scheduler-config capture and fail-fast checks, the cross-DP-row prefix-hit hazard
  (wrong without KV-R whichever lane hits, right with it), same-step hits (writer-first), a chunked prompt changing
  lane between chunks, internal span splits, the verify / propose bookkeeping, the idle-lane budget, PS-1 hold-back;
* the real engine end to end: ``vllm.LLM`` -> TTPlatform -> TTWorker -> TTScheduler / TTModelRunner -> bridge -> fake
  generator, with only the mesh open/close patched (this is what found the stale block-table ids), with every feature
  off (draft 1) and with chunked prefill + prefix caching + MTP speculation on (small budget, every token checked);
* the Motif reasoning and tool parser plugin files, loaded the way ``--*-parser-plugin`` loads them, and the
  documented ``vllm serve`` flags parsed by vLLM's own CLI parser.

The CPU golden package (``models/demos/motif3/reference``) is not needed: the fake generator's "model" is a
deterministic function of the token sequence, which is all the plumbing tests need.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _visible_tt_devices():
    try:
        return sorted(os.listdir("/dev/tenstorrent"))
    except (FileNotFoundError, NotADirectoryError):
        return []


if _visible_tt_devices() and os.environ.get("MOTIF3_HOST_TEST_ALLOW_DEVICES") != "1":
    pytest.skip(
        "host-only vLLM bridge tests must run with the Tenstorrent devices hidden: "
        "unshare -Urm --propagation private bash -c 'mount -t tmpfs none /dev/tenstorrent && pytest ...' "
        "(or set MOTIF3_HOST_TEST_ALLOW_DEVICES=1 on a machine where touching the devices is harmless)",
        allow_module_level=True,
    )

from models.demos.motif3.tt import generator_api as api  # noqa: E402
from models.demos.motif3.tt import generator_vllm as gv  # noqa: E402
from models.demos.motif3.tt import prefill_plan  # noqa: E402

METAL_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
_PROJECT = METAL_ROOT.parent
_CANDIDATE_DIRS = [
    os.environ.get("MOTIF3_HF_DIR"),
    os.environ.get("HF_MODEL"),
    str(_PROJECT / "weights" / "Motif-3"),
    str(_PROJECT / "hf_meta"),
]
ALL_ON = {name: "1" for name in api.FEATURE_SWITCHES}
ALL_OFF = {name: "0" for name in api.FEATURE_SWITCHES}
CAPS_ON = gv.model_capabilities_from_env(ALL_ON)
CAPS_OFF = gv.model_capabilities_from_env(ALL_OFF)
PLACEHOLDER = -1  # vllm_tt_plugin.spec_decode.PLACEHOLDER_TOKEN_ID


def _motif_dir(require_tokenizer: bool = False) -> Path:
    for cand in _CANDIDATE_DIRS:
        if not cand:
            continue
        p = Path(cand)
        if (p / "config.json").is_file() and (p / "configuration_motif.py").is_file():
            if not require_tokenizer or (p / "tokenizer.json").is_file():
                return p
    pytest.skip(f"no Motif-3 config{' + tokenizer' if require_tokenizer else ''} dir among {_CANDIDATE_DIRS}")


@pytest.fixture(autouse=True)
def _fresh_bridge_process_state(monkeypatch):
    """``get_max_tokens_all_users`` records the scheduler config it saw (module globals: a process-wide fact in
    EngineCore) for ``initialize_vllm_model``; every test starts without one, and without the feature env knobs."""
    monkeypatch.setattr(gv, "_SEEN_VLLM_BLOCK_SIZE", None)
    monkeypatch.setattr(gv, "_SEEN_VLLM_SERVING", None)
    for var in (
        "MOTIF3_KV_REPLICATED_DECODE",
        "MOTIF3_PREFILL_MAX_BUCKET",
        "MOTIF3_PACKED_PREFILL",
        "MOTIF3_SPEC_VERIFY",
    ):
        monkeypatch.delenv(var, raising=False)


@contextlib.contextmanager
def loguru_messages():
    """Collect the bridge's loguru messages (vLLM's logging capture does not see loguru)."""
    from loguru import logger

    seen = []
    handle = logger.add(lambda m: seen.append(m.record["message"]), level="DEBUG")
    try:
        yield seen
    finally:
        logger.remove(handle)


# ================================================================================================================
# Fake generator: the device semantics the bridge depends on, on the host
# ================================================================================================================
PAD_GARBAGE = -99  # bucket-padding rows a prefill writes into the request's own last block
NO_TOKEN = -5  # argmax / MTP ids of lanes that ran nothing this step: the bridge must never hand them out


def next_token(seq, vocab: int) -> int:
    """The fake model: a deterministic, position-sensitive function of the whole token sequence.

    Never below 100, so it never emits a Motif special / stop token (ids 0-84 are control tokens)."""
    s = torch.as_tensor(seq, dtype=torch.int64).reshape(-1)
    w = torch.arange(1, s.numel() + 1, dtype=torch.int64)
    return 100 + int(((s + 7) * w * w).sum().item() % (vocab - 100))


def wrong_token(token: int, vocab: int) -> int:
    """A token guaranteed to differ from ``token`` (what a prefill over stale KV answers)."""
    return 100 + ((int(token) - 100 + 1) % (vocab - 100))


def mtp_token(seq, vocab: int) -> int:
    """The fake MTP layer. ``seq`` = the tokens through position ``n + 1`` (the anchor's sequence plus the target's
    argmax at ``n``); returns the draft for ``n + 2``: the fake model's own next token on ~70 % of sequences (a
    deterministic hash), another token otherwise, so verify steps see accepted and rejected drafts."""
    truth = next_token(seq, vocab)
    n = int(torch.as_tensor(seq).numel())
    return truth if (truth * 7 + n * 3) % 10 < 7 else wrong_token(truth, vocab)


def one_hot(token: int, vocab: int, dtype=torch.float32) -> torch.Tensor:
    out = torch.full((vocab,), -1.0, dtype=dtype)
    out[token] = 1.0
    return out


def greedy_continuation(prompt, n: int, vocab: int):
    seq, out = list(prompt), []
    for _ in range(n):
        out.append(next_token(seq, vocab))
        seq.append(out[-1])
    return out


class FakeMotifGenerator(api.MotifGenerator):
    """Emulates what matters to the bridge, draft 1 and the features:

    * the latent pool is one copy per DP group (the real pool is replicated on every chip; a group's 8 chips agree):
      prefill writes every copy, decode writes only the copy of its lane's group ``lane // 8``, or every copy with
      KV-R (``settings.kv_replicated``);
    * prefill rows (any ``start``) are planned with ``tt/prefill_plan.py`` exactly as the generator plans them
      (alignment ``A = 64``, the span cap, chunks) and run writer-first in ONE call. An sp1 row (``c0 > 0``) READS
      positions ``[0, w0)`` from the cache through its page table (the full blocks below its write floor: the global
      layers read them as keys -- the fill skips them, so even the recomputed rows ``[c0, w0)`` come from the cache --
      and the SWA tail lies inside them), from EVERY group's copy: on device the prefill is replicated and the MoE
      reduce-scatter mixes the rows, so one stale copy corrupts every row (features design review R7). Copies that
      disagree give a wrong token. An sp0 row reads nothing. The row writes ``[w0, end)`` on every copy, never a block
      below ``w0``, and scribbles bucket padding only into its own last block;
    * a step's "logits" are a one-hot of ``next_token`` over the sequence read back through the page table (decode:
      from the lane's group copy), so a wrong lane group, page table, position or token changes the prediction;
    * speculation (``decode_forward_spec``): every active owner lane writes its anchor at ``n`` and reads ``0 .. n``
      (``a0``); a draft runs on an IDLE partner lane (any group with KV-R, the owner's group without), writes at
      ``n + 1`` through the owner's page-table row and reads ``0 .. n + 1`` from the partner's group copy (``a1``);
      drafts without a partner run in an overflow pass on their own lane. ``m = mtp_token(...)``;
    * inactive-lane rows are NaN logits and ``NO_TOKEN`` ids, so the bridge must never hand them to vLLM.
    """

    ALIGN = 64

    def __init__(self, settings: api.GeneratorSettings, vocab_size: int, hf_config=None, mesh_device=None):
        self.settings = settings
        self._vocab = int(vocab_size)
        self.hf_config = hf_config
        self.mesh_device = mesh_device
        self.kv = None
        self.handle = None
        self.block_size = None
        self.alloc_args = None
        self.writer_first = True  # False = input order (negative control for same-step hits)
        self.prefills = []  # (lane, end, enable_trace) per row, in execution order
        self.prefill_rows = []  # one record per row (start, end, c0, w0, paths, consistent, ...)
        self.prefill_calls = []  # one record per prefill_forward_batch call
        self.decode_steps = []  # (active lanes, enable_trace) of every decode step (plain and spec)
        self.spec_steps = []  # one record per decode_forward_spec call
        self.warmups = []  # ("prefill"|"decode", enable_trace, page_table_width)
        self.released_lanes = []
        self.traces_released = 0
        self.fail_next_decode = False

    @classmethod
    def create(cls, *, hf_config, mesh_device, settings):
        return cls(settings, int(getattr(hf_config, "vocab_size", api.VOCAB_SIZE)), hf_config, mesh_device)

    @property
    def num_layers(self) -> int:
        return self.settings.num_layers

    @property
    def vocab_size(self) -> int:
        return self._vocab

    @property
    def supports_resumed_prefill(self) -> bool:
        return True

    @property
    def prefill_alignment(self) -> int:
        return self.ALIGN

    @property
    def max_prefill_span(self) -> int:
        return self.settings.resolved_prefill_span_cap(True)

    @property
    def supports_spec_decode(self) -> bool:
        return True

    def allocate_kv_cache(self, *, num_blocks, block_size, num_layers):
        assert num_layers == self.num_layers
        self.alloc_args = dict(num_blocks=num_blocks, block_size=block_size, num_layers=num_layers)
        self.block_size = int(block_size)
        self.kv = torch.full((api.NUM_DP_GROUPS, num_blocks, block_size), -1, dtype=torch.int64)
        self.handle = ("fake-latent-pool", id(self))
        return self.handle

    def _read(self, group: int, page_table: torch.Tensor, upto: int) -> torch.Tensor:
        pos = torch.arange(upto + 1)
        return self.kv[group, page_table[pos // self.block_size].long(), pos % self.block_size]

    def _write(self, lane: int, page_table: torch.Tensor, pos: int, token: int) -> None:
        """A decode KV write from ``lane`` at ``pos`` through ``page_table``: every group copy with KV-R, else the
        lane's group only."""
        block, row = int(page_table[pos // self.block_size]), pos % self.block_size
        assert block >= 1, "decode position on the null block"
        if self.settings.kv_replicated:
            self.kv[:, block, row] = int(token)
        else:
            self.kv[lane // api.LANES_PER_GROUP, block, row] = int(token)

    # ---- prefill ------------------------------------------------------------------------------------------------
    def prefill_forward(self, request, *, kv_cache, enable_trace=False):
        return self.prefill_forward_batch([request], kv_cache=kv_cache, enable_trace=enable_trace)[0]

    def prefill_forward_batch(self, requests, *, kv_cache, enable_trace=False):
        assert kv_cache is self.handle
        reqs = api.check_prefill_batch(requests)
        if any(r.resumed for r in reqs):
            assert self.settings.resumed_prefill, "a resumed row on a launch without chunked prefill / prefix caching"
        cap = self.max_prefill_span
        plans, order = prefill_plan.plan_prefill_batch(
            reqs,
            block_size=self.block_size,
            align=self.prefill_alignment,
            buckets=prefill_plan.span_buckets(self.settings.max_seq_len, cap),
            span_cap=cap,
        )
        if not self.writer_first:
            order = list(range(len(reqs)))
        written, rows, out = set(), [], [None] * len(reqs)
        for i in order:
            out[i], rec = self._prefill_row(reqs[i], plans[i], enable_trace, written)
            rows.append(rec)
        self.prefill_calls.append(dict(rows=rows, order=list(order)))
        return torch.stack(out)

    def _prefill_row(self, req, plan, enable_trace, written):
        s, e, bs, pt = int(req.start), req.seq_len, self.block_size, req.page_table
        need = api.cdiv(e, bs)
        assert bool((pt[:need] >= 1).all()), "prompt positions on the null block"
        # Contract: the tail is zero. vLLM's rows can carry stale ids of OTHER requests' blocks there, and the
        # bucket-padding writes below would land in them.
        assert bool((pt[need:] == 0).all()), f"prefill page-table tail not zeroed: {pt.tolist()}"
        tokens = req.tokens.long()
        c0, w0 = plan.c0, plan.w0
        cached = w0 if c0 > 0 else 0  # sp1 reads the blocks below the write floor from the cache; sp0 reads nothing
        seq, consistent, reads_same_call = tokens, True, False
        if cached:
            pos = torch.arange(cached)
            blocks = pt[pos // bs].long()
            copies = self.kv[:, blocks, pos % bs]  # every chip's copy
            consistent = bool((copies == copies[:1]).all())
            seq = torch.cat([copies[0], tokens[cached:]])
            reads_same_call = bool(set(blocks.tolist()) & written)
        pos = torch.arange(w0, e)
        self.kv[:, pt[pos // bs].long(), pos % bs] = tokens[w0:e]  # every chip; never a block below w0
        last = plan.chunks[-1]
        pad = torch.arange(e, min(last.start + last.bucket, need * bs))
        if pad.numel():  # bucket padding: only inside the own last block (fill table -1 past it)
            self.kv[:, pt[pad // bs].long(), pad % bs] = PAD_GARBAGE
        written.update(int(b) for b in pt[w0 // bs : need].tolist())
        pred = next_token(seq, self._vocab) if consistent else wrong_token(next_token(tokens, self._vocab), self._vocab)
        self.prefills.append((req.lane, e, enable_trace))
        rec = dict(
            lane=req.lane,
            start=s,
            end=e,
            c0=c0,
            w0=w0,
            paths=plan.paths,
            buckets=plan.buckets,
            consistent=consistent,
            reads_same_call=reads_same_call,
            tokens=tokens.tolist(),
        )
        self.prefill_rows.append(rec)
        return one_hot(pred, self._vocab, torch.bfloat16), rec

    # ---- decode -------------------------------------------------------------------------------------------------
    def _check_lane_inputs(self, batch, drafted=None):
        active = batch.active
        assert bool((batch.tokens[~active] == 0).all()) and bool((batch.page_table[~active] == 0).all())
        for lane in torch.nonzero(active).reshape(-1).tolist():
            p, pt = int(batch.positions[lane]), batch.page_table[lane]
            top = p + 1 if drafted is not None and bool(drafted[lane]) else p
            assert int(pt[top // self.block_size]) >= 1, "decode position on the null block"
            assert bool((pt[top // self.block_size + 1 :] == 0).all()), f"decode page-table tail not zeroed: {pt}"

    def decode_forward(self, batch, *, kv_cache, enable_trace):
        assert kv_cache is self.handle
        assert isinstance(batch, api.DecodeBatch)
        assert not self.settings.spec_decode, "a speculating launch runs every decode step through decode_forward_spec"
        if self.fail_next_decode:
            self.fail_next_decode = False
            raise RuntimeError("injected decode failure")
        self._check_lane_inputs(batch)
        lanes = torch.nonzero(batch.active).reshape(-1).tolist()
        out = torch.full((api.NUM_LANES, self._vocab), float("nan"))
        for lane in lanes:  # every write of the step precedes every read (the update runs before FlashMLA)
            self._write(lane, batch.page_table[lane], int(batch.positions[lane]), int(batch.tokens[lane]))
        for lane in lanes:
            seq = self._read(lane // api.LANES_PER_GROUP, batch.page_table[lane], int(batch.positions[lane]))
            out[lane] = one_hot(next_token(seq, self._vocab), self._vocab)
        self.decode_steps.append((lanes, enable_trace))
        return out

    def decode_forward_spec(self, batch, *, kv_cache, enable_trace, want_logits):
        assert kv_cache is self.handle
        assert isinstance(batch, api.SpecDecodeBatch)
        assert self.settings.spec_decode, "decode_forward_spec on a launch without speculation"
        if self.fail_next_decode:
            self.fail_next_decode = False
            raise RuntimeError("injected decode failure")
        drafted = batch.has_draft
        self._check_lane_inputs(batch, drafted)
        V, G = self._vocab, api.LANES_PER_GROUP
        owners = torch.nonzero(batch.active).reshape(-1).tolist()
        n = {lane: int(batch.positions[lane]) for lane in owners}
        idle = list(batch.idle_lanes)
        partner, overflow = {}, []
        for lane in torch.nonzero(drafted).reshape(-1).tolist():  # packed verify (features design §3.8.2)
            cands = idle if self.settings.kv_replicated else [x for x in idle if x // G == lane // G]
            if cands:
                partner[lane] = cands[0]
                idle.remove(cands[0])
            else:
                overflow.append(lane)
        argmax = torch.full((api.NUM_LANES, 2), NO_TOKEN, dtype=torch.int32)
        mtp = torch.full((api.NUM_LANES, 2), NO_TOKEN, dtype=torch.int32)
        logits = torch.full((api.NUM_LANES, V), float("nan")) if want_logits else None
        pt = batch.page_table
        # Pass 1: call A writes the anchors, call B the packed drafts, then every row reads.
        for lane in owners:
            self._write(lane, pt[lane], n[lane], int(batch.tokens[lane]))
        for lane, p in partner.items():
            self._write(p, pt[lane], n[lane] + 1, int(batch.draft_tokens[lane]))
        for lane in owners:
            seq = self._read(lane // G, pt[lane], n[lane]).tolist()
            a0 = next_token(seq, V)
            argmax[lane, 0], mtp[lane, 0] = a0, mtp_token(seq + [a0], V)
            if logits is not None:
                logits[lane] = one_hot(a0, V)
        for lane, p in partner.items():
            seq = self._read(p // G, pt[lane], n[lane] + 1).tolist()
            a1 = next_token(seq, V)
            argmax[lane, 1], mtp[lane, 1] = a1, mtp_token(seq + [a1], V)
        # Pass 2 (overflow): the remaining drafts on their own lanes at n + 1.
        for lane in overflow:
            self._write(lane, pt[lane], n[lane] + 1, int(batch.draft_tokens[lane]))
        for lane in overflow:
            seq = self._read(lane // G, pt[lane], n[lane] + 1).tolist()
            a1 = next_token(seq, V)
            argmax[lane, 1], mtp[lane, 1] = a1, mtp_token(seq + [a1], V)
        self.decode_steps.append((owners, enable_trace))
        self.spec_steps.append(
            dict(
                owners=owners,
                drafted=torch.nonzero(drafted).reshape(-1).tolist(),
                partners=dict(partner),
                overflow=list(overflow),
                want_logits=want_logits,
                trace=enable_trace,
            )
        )
        return api.SpecDecodeResult(logits=logits, argmax=argmax, mtp_argmax=mtp)

    def warmup_prefill(self, *, kv_cache, enable_trace):
        assert kv_cache is self.handle
        self.warmups.append(("prefill", enable_trace, None))

    def warmup_decode(self, *, kv_cache, enable_trace, page_table_width):
        assert kv_cache is self.handle
        self.warmups.append(("decode", enable_trace, page_table_width))

    def release_lane(self, lane):
        self.released_lanes.append(int(lane))

    def release_traces(self):
        self.traces_released += 1


class FakeDraft1Generator(FakeMotifGenerator):
    """The draft-1 generator: ``MotifGenerator``'s feature defaults (no resumed prefill, no speculation, one bucket
    per prompt), bucket padding through the zeroed page-table tail into null block 0."""

    created = 0
    supports_resumed_prefill = api.MotifGenerator.supports_resumed_prefill
    prefill_alignment = api.MotifGenerator.prefill_alignment
    max_prefill_span = api.MotifGenerator.max_prefill_span
    supports_spec_decode = api.MotifGenerator.supports_spec_decode
    prefill_forward_batch = api.MotifGenerator.prefill_forward_batch
    decode_forward_spec = api.MotifGenerator.decode_forward_spec

    @classmethod
    def create(cls, *, hf_config, mesh_device, settings):
        cls.created += 1
        return super().create(hf_config=hf_config, mesh_device=mesh_device, settings=settings)

    def prefill_forward(self, request, *, kv_cache, enable_trace=False):
        assert kv_cache is self.handle
        assert isinstance(request, api.PrefillRequest) and request.start == 0
        s, bs, pt = request.seq_len, self.block_size, request.page_table
        assert bool((pt[: api.cdiv(s, bs)] >= 1).all()), "prompt positions on the null block"
        assert bool((pt[api.cdiv(s, bs) :] == 0).all()), f"prefill page-table tail not zeroed: {pt.tolist()}"
        pos = torch.arange(s)
        self.kv[:, pt[pos // bs].long(), pos % bs] = request.tokens.long()  # every chip
        bucket = next(b for b in api.prefill_buckets(self.settings.max_seq_len) if b >= s)
        pad = torch.arange(s, min(bucket, pt.numel() * bs))
        if pad.numel():
            self.kv[:, pt[pad // bs].long(), pad % bs] = PAD_GARBAGE  # overwritten by decode before any read
        self.prefills.append((request.lane, s, enable_trace))
        return one_hot(next_token(request.tokens, self._vocab), self._vocab, torch.bfloat16)


def _bridge(num_slots=32, max_seq_len=1024, num_layers=3, vocab=4096, kv_dtype="bfp8", gen_cls=None, **features):
    settings = api.GeneratorSettings(
        max_batch_size=num_slots, max_seq_len=max_seq_len, num_layers=num_layers, kv_cache_dtype=kv_dtype, **features
    )
    gen = (gen_cls or FakeMotifGenerator)(settings, vocab)
    return gv.MotifForCausalLM(gen, settings), gen


# ================================================================================================================
# vLLM-side driver: the parts of TTModelRunner that talk to the model, with the plugin's own slot bookkeeping
# ================================================================================================================
class PluginDriver:
    """Feeds the bridge exactly what ``vllm_tt_plugin.model_runner`` / ``async_decode`` feed it.

    State slots come from the plugin's ``TTModelRunner`` methods, called unbound on a stand-in runner (the plugin's
    own ``tests/test_state_slots.py`` pattern). Blocks come from a refcounted free list that never hands out block 0
    (vLLM's null block); a prefix-cache hit shares the source request's leading blocks. Every token is checked
    against the fake model's ground truth. ``spec_decode`` runs one decode step of a speculating launch the way the
    runner builds it (``_spec_row_state`` -> ``_step_verifies`` / PS-1 hold-back -> ``_spec_candidate_block`` ->
    ``decode_forward`` -> ``accept_greedy_drafts`` -> commit -> ``propose_draft_tokens`` -> ``_publish_draft``).
    """

    def __init__(self, bridge, kv, *, num_slots, block_size, num_blocks, width, vocab, stale_tails=True, ps1=False):
        from vllm_tt_plugin.model_runner import TTModelRunner

        self.R = TTModelRunner
        self.bridge, self.kv = bridge, kv
        self.num_slots, self.bs, self.width, self.vocab = num_slots, block_size, width, vocab
        self.max_len = int(bridge.settings.max_seq_len)
        self.runner = SimpleNamespace(
            tt_per_lane_max_num_seqs=num_slots,
            _req_state_slot={},
            _pending_state_slot_settle=None,
            _pending_state_slot_moves=None,
            requests={},
            model=bridge,
        )
        self.free = list(range(num_blocks - 1, 0, -1))
        self.refs = {}
        self.seqs = {}
        self.blocks = {}
        self.order = []  # persistent-batch row order of running (decoding) requests
        self.remaps = 0
        self.stale_tails = stale_tails
        self.stale_rng = random.Random(99)
        self.sample_rng = random.Random(5)
        self.lane_of = {}  # request -> the lane it was prefilled on (must not change until it is re-prefilled)
        self.lane_checks = 0
        # Speculation: the runner's per-request state.
        self.sampled = set()  # requests with a temperature: never speculable
        self.drafts = {}  # request -> the drafts the scheduler verifies next step
        self.counts = {}  # request -> accepted count of its previous step (absent = 1)
        self.ps1 = ps1  # SpecPlan.verify_requires_speculable_rows
        self.stats = collections.Counter()
        self.offers = []  # drafts offered per propose call

    def add(self, rid, prompt, sampled=False):
        self.seqs[rid] = list(prompt)
        self.blocks[rid] = []
        if sampled:
            self.sampled.add(rid)

    def _grow(self, rid, n_tokens):
        while len(self.blocks[rid]) < api.cdiv(n_tokens, self.bs):
            b = self.free.pop()
            self.refs[b] = 1
            self.blocks[rid].append(b)

    def share_prefix(self, rid, src, n_blocks):
        """A prefix-cache hit: ``rid``'s first ``n_blocks`` blocks are ``src``'s (same tokens)."""
        assert not self.blocks[rid] and self.seqs[rid][: n_blocks * self.bs] == self.seqs[src][: n_blocks * self.bs]
        for b in self.blocks[src][:n_blocks]:
            self.refs[b] += 1
            self.blocks[rid].append(b)

    def _row_table(self, rid):
        row = torch.zeros(self.width, dtype=torch.int32)
        n = len(self.blocks[rid])
        row[:n] = torch.tensor(self.blocks[rid], dtype=torch.int32)
        if self.stale_tails:
            # vLLM's persistent block table is not cleared when a row is reused: entries past a request's own
            # blocks can hold stale ids. Emulate the worst case, blocks that other live requests own now.
            own = set(self.blocks[rid])
            others = sorted({b for r, owned in self.blocks.items() if r != rid for b in owned} - own)
            k = min(self.width - n, len(others), self.stale_rng.randrange(0, 4))
            if k:
                row[n : n + k] = torch.tensor(self.stale_rng.sample(others, k), dtype=torch.int32)
        return row

    def _sample(self, rid, argmax):
        """Host sampling: greedy rows take the argmax; a sampled request draws something else."""
        if rid not in self.sampled:
            return int(argmax)
        return 100 + ((int(argmax) - 100 + 1 + self.sample_rng.randrange(1000)) % (self.vocab - 100))

    def _check_and_append(self, rids, logits):
        for rid, row in zip(rids, logits, strict=True):
            want = next_token(self.seqs[rid], self.vocab)
            got = int(torch.nan_to_num(row.float(), nan=-1e9).argmax())
            assert got == want, f"request {rid}: sampled {got}, ground truth {want} (len {len(self.seqs[rid])})"
            self.seqs[rid].append(self._sample(rid, got))

    def prefill(self, rids, starts=None, ends=None, slots=None, check=True):
        """One prefill step: row ``i`` = request ``rids[i]``, positions ``[starts[i], ends[i])`` (default: the whole
        sequence from 0). A row whose end is its sequence length is final and commits a token."""
        rids = list(rids)
        if slots is None:
            slots = self.R._alloc_prefill_state_slots(self.runner, rids)
        else:
            for r, s in zip(rids, slots, strict=True):
                self.runner._req_state_slot[r] = int(s)
        self.runner.requests.update(dict.fromkeys(rids))
        starts = [0] * len(rids) if starts is None else [int(s) for s in starts]
        ends = [len(self.seqs[r]) for r in rids] if ends is None else [int(e) for e in ends]
        for r, e in zip(rids, ends, strict=True):
            self._grow(r, e)
        tokens = torch.full((len(rids), max(ends)), 4321, dtype=torch.int32)  # stale past each row's end
        for i, r in enumerate(rids):
            tokens[i, : ends[i]] = torch.tensor(self.seqs[r][: ends[i]], dtype=torch.int32)
        for r, slot in zip(rids, slots, strict=True):
            self.lane_of[r] = self.bridge._lanes.lane_of_slot(slot)
        out = self.bridge.prefill_forward(
            tokens=tokens,
            page_table=torch.stack([self._row_table(r) for r in rids]),
            kv_cache=self.kv,
            enable_trace=False,
            prompt_lens=np.array(ends, dtype=np.int64),
            start_pos=np.array(starts, dtype=np.int32),
            empty_slots=list(slots),
        )
        assert tuple(out.shape) == (len(rids), 1, self.vocab)
        results, done = [], []
        for i, r in enumerate(rids):
            got = int(torch.nan_to_num(out[i, -1].float(), nan=-1e9).argmax())
            want = next_token(self.seqs[r][: ends[i]], self.vocab)
            results.append((got, want))
            if check:
                assert (
                    got == want
                ), f"request {r}: prefill [{starts[i]}, {ends[i]}) predicted {got}, ground truth {want}"
            self.drafts.pop(r, None)
            self.counts.pop(r, None)
            if ends[i] == len(self.seqs[r]):  # the final chunk commits the sampled token
                self.seqs[r].append(self._sample(r, got))
                done.append(r)
        # After a prefill step the running requests come back behind the prefilled rows (vLLM re-adds them).
        self.order = done + [r for r in self.order if r not in rids]
        return results if not check else slots

    def _decode_common(self, rows):
        remap = self.R._decode_state_slot_remap(self.runner, rows)
        pt = torch.zeros(self.num_slots, self.width, dtype=torch.int32)
        for i, r in enumerate(rows):
            self._grow(r, len(self.seqs[r]) + 1)  # the anchor at n and the scheduler's lookahead for a draft at n + 1
            pt[i] = self._row_table(r)
        kwargs = dict(
            page_table=pt,
            kv_cache=self.kv,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
            enable_trace=True,
            read_from_device=False,
        )
        if remap is not None:
            kwargs["slot_remap"] = remap
            self.remaps += 1
        return kwargs

    def _after_decode(self, rows):
        self.R.note_decode_state_slots_settled(self.runner)
        for r in rows:  # a request keeps its lane (= its DP group's KV copy) for as long as it lives
            assert self.bridge._lanes.lane_of_slot(self.runner._req_state_slot[r]) == self.lane_of[r], r
            self.lane_checks += 1
        self.order = rows

    def decode(self, rows=None):
        rows = list(self.order if rows is None else rows)
        kwargs = self._decode_common(rows)
        tokens = torch.zeros(self.num_slots, 1, dtype=torch.int32)
        pos = torch.full((self.num_slots,), -1, dtype=torch.int32)
        for i, r in enumerate(rows):
            tokens[i, 0], pos[i] = self.seqs[r][-1], len(self.seqs[r]) - 1
        out = self.bridge.decode_forward(tokens=tokens, start_pos=pos, **kwargs)
        self._after_decode(rows)
        assert tuple(out.shape) == (self.num_slots, 1, self.vocab)
        self._check_and_append(rows, out[: len(rows), -1, :])
        return kwargs.get("slot_remap")

    def spec_decode(self, rows=None):
        """One decode step of a speculating launch (K = 1), built and finished as ``TTModelRunner`` does it."""
        from vllm_tt_plugin.async_decode import _verify_output_tensor
        from vllm_tt_plugin.model_runner import _step_verifies
        from vllm_tt_plugin.spec_decode import accept_greedy_drafts

        rows = list(self.order if rows is None else rows)
        n, B = len(rows), self.num_slots
        drafts, num_valid, counts = self.R._spec_row_state(
            self.counts, {r: self.drafts[r] for r in rows if r in self.drafts}, rows, 1
        )
        held_back = self.ps1 and any(r in self.sampled for r in rows)
        verify = not held_back and _step_verifies(True, num_valid, counts, n)
        kwargs = self._decode_common(rows)
        tok1 = torch.tensor([[self.seqs[r][-1]] for r in rows], dtype=torch.int32)
        pos1 = torch.tensor([len(self.seqs[r]) - 1 for r in rows], dtype=torch.int32)
        if verify:
            tok2, pos2 = self.R._spec_candidate_block(drafts, num_valid, tok1, pos1)
            tokens = torch.zeros(B, 2, dtype=torch.int32)  # padding rows: token 0, position -1 in every column
            positions = torch.full((B, 2), -1, dtype=torch.int32)
            nv, cnt = torch.zeros(B, dtype=torch.int32), torch.ones(B, dtype=torch.int32)
            dr = torch.full((B, 1), PLACEHOLDER, dtype=torch.int32)
            tokens[:n], positions[:n], nv[:n], cnt[:n], dr[:n] = tok2, pos2, num_valid, counts, drafts
            out = self.bridge.decode_forward(
                tokens=tokens, start_pos=positions, num_valid_drafts=nv, accepted_counts=cnt, spec_mode="argmax_ids",
                **kwargs,
            )  # fmt: skip
            ids = _verify_output_tensor(out, "MotifForCausalLM", "argmax_ids")
            assert tuple(ids.shape) == (B, 2) and ids.dtype == torch.int32
            self._after_decode(rows)
            committed, ccounts = accept_greedy_drafts(ids, dr, nv)
            for i, r in enumerate(rows):
                c = int(ccounts[i])
                if r in self.sampled:
                    self.stats["sampled_rows_verified"] += 1  # what PS-1 prevents: a sampled row commits the argmax
                for t in committed[i, :c].tolist():
                    want = next_token(self.seqs[r], self.vocab)
                    assert t == want, f"request {r}: verify committed {t}, ground truth {want}"
                    self.seqs[r].append(int(t))
                if int(nv[i]):
                    self.stats[f"draft_count{c}"] += 1
                self.counts[r] = c
                self.drafts.pop(r, None)
            self.stats["verify_steps"] += 1
            self._propose(committed, self.R._committed_positions(positions, 2), ccounts, rows)
        else:
            tokens = torch.zeros(B, 1, dtype=torch.int32)
            positions = torch.full((B,), -1, dtype=torch.int32)
            tokens[:n], positions[:n] = tok1, pos1
            out = self.bridge.decode_forward(tokens=tokens, start_pos=positions, **kwargs)
            self._after_decode(rows)
            assert torch.is_tensor(out) and tuple(out.shape) == (B, 1, self.vocab)
            committed = torch.full((B, 2), PLACEHOLDER, dtype=torch.int32)
            committed[:, 0] = 0
            for i, r in enumerate(rows):
                want = next_token(self.seqs[r], self.vocab)
                got = int(torch.nan_to_num(out[i, -1].float(), nan=-1e9).argmax())
                assert got == want, f"request {r}: decode predicted {got}, ground truth {want}"
                committed[i, 0] = self._sample(r, got)
                self.seqs[r].append(int(committed[i, 0]))
                self.counts.pop(r, None)  # one token per row: every count is back to 1
                self.drafts.pop(r, None)  # a held-back step drops the scheduled drafts unverified
            self.stats["ordinary_steps"] += 1
            positions2 = torch.full((B, 2), -1, dtype=torch.int32)
            for i, r in enumerate(rows):  # _committed_positions_from_state: the committed token's position
                positions2[i] = torch.tensor([len(self.seqs[r]) - 1, len(self.seqs[r])], dtype=torch.int32)
            self._propose(committed, positions2, torch.ones(B, dtype=torch.int32), rows)

    def _propose(self, committed, positions, counts, rows):
        """``TTModelRunner._propose_model_drafts`` + ``_publish_draft``."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        B = self.num_slots
        out = self.bridge.propose_draft_tokens(1, committed, positions, counts, hidden=None)
        assert isinstance(out, DraftOutput)
        assert out.draft_token_ids.dtype == torch.int32 and tuple(out.draft_token_ids.shape) == (B, 1)
        assert out.num_valid is not None and out.num_valid.dtype == torch.int32 and tuple(out.num_valid.shape) == (B,)
        assert int(out.num_valid[len(rows) :].sum()) == 0, "a padding row was offered a draft"
        self.offers.append(int(out.num_valid.sum()))
        held = self.ps1 and any(r in self.sampled for r in rows)
        for i, r in enumerate(rows):
            offered = int(out.num_valid[i])
            usable = max(0, min(offered, self.max_len - len(self.seqs[r])))
            ids = [int(t) for t in out.draft_token_ids[i, :usable].tolist()]
            assert all(0 <= t < self.vocab for t in ids), ids
            if offered and r in self.sampled:
                self.stats["sampled_rows_offered"] += 1
            if ids and r not in self.sampled and not held:
                self.drafts[r] = ids
                self.stats["published"] += 1
            else:
                self.drafts.pop(r, None)

    def _release(self, rid, preempted):
        self.R._release_model_request(self.runner, rid)
        out = SimpleNamespace(
            finished_req_ids=set() if preempted else {rid}, preempted_req_ids={rid} if preempted else None
        )
        self.R._release_dead_state_slots(self.runner, out)
        for b in self.blocks[rid]:
            self.refs[b] -= 1
            if not self.refs[b]:
                del self.refs[b]
                self.free.append(b)
        self.blocks[rid] = []
        self.drafts.pop(rid, None)
        self.counts.pop(rid, None)
        if rid in self.order:  # condense: the last row moves into the hole
            i = self.order.index(rid)
            last = self.order.pop()
            if last != rid:
                self.order[i] = last

    def finish(self, rid):
        self._release(rid, preempted=False)
        self.runner.requests.pop(rid, None)
        self.seqs.pop(rid)
        self.blocks.pop(rid)
        self.sampled.discard(rid)

    def preempt(self, rid):
        self._release(rid, preempted=True)  # keeps its tokens; resumes with a full re-prefill


# ================================================================================================================
# 1. Import hygiene and registration
# ================================================================================================================
def test_bridge_import_is_device_free_and_lazy():
    """vLLM imports the bridge in the API server, the registry subprocess and EngineCore before any mesh exists."""
    probe = (
        "import json, sys\n"
        "import models.demos.motif3.tt.generator_vllm as m\n"
        "import models.demos.motif3.vllm_plugins as p\n"
        "mods = sorted(k for k in sys.modules if k.startswith('models.'))\n"
        "print(json.dumps({'models': mods, 'ttnn': 'ttnn' in sys.modules, 'vllm': 'vllm' in sys.modules,"
        " 'plugin': 'vllm_tt_plugin' in sys.modules,"
        " 'tokens': m.MotifForCausalLM.get_max_tokens_all_users(max_model_len=32768, max_num_seqs=32)}))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(METAL_ROOT))
    env.pop("MOTIF3_KV_POOL_TOKENS", None)
    res = subprocess.run([sys.executable, "-c", probe], cwd=METAL_ROOT, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
    info = json.loads(res.stdout.strip().splitlines()[-1])
    # The project rule (design 00 §2.1): no other models/** package at import time (demo imports opened the cluster in
    # the prior port). ttnn itself is allowed, though the bridge and generator_api do not need it.
    foreign = [
        m for m in info["models"] if m not in ("models", "models.demos") and not m.startswith("models.demos.motif3")
    ]
    assert not foreign, f"the bridge import pulled in other models/** packages: {foreign}"
    # The TT runtime (mesh, weights) is imported lazily by initialize_vllm_model, and vLLM / the plugin are never
    # imported here (spec_plan / verify / propose import vllm_tt_plugin inside the call).
    heavy = {"models.demos.motif3.tt.generator", "models.demos.motif3.tt.model", "models.demos.motif3.tt.weights"}
    assert not heavy & set(info["models"]) and info["vllm"] is False and info["plugin"] is False
    assert info["tokens"] == 262144 + gv.NULL_BLOCK_RESERVE_TOKENS


def test_vllm_metadata_json_names_the_bridge():
    meta = json.loads((PACKAGE_DIR / "vllm_metadata.json").read_text())
    assert meta == {"arch": gv.ARCHITECTURE, "main_class": gv.MAIN_CLASS}
    module, cls_name = meta["main_class"].split(":")
    import importlib

    assert getattr(importlib.import_module(module), cls_name) is gv.MotifForCausalLM
    assert gv.TT_MODEL_CLASS_OVERRIDES == "MotifForCausalLM=models.demos.motif3.tt.generator_vllm:MotifForCausalLM"


def _patched_registry(monkeypatch):
    from vllm.model_executor.models.registry import ModelRegistry

    registered = {}
    monkeypatch.setattr(ModelRegistry, "get_supported_archs", staticmethod(lambda: list(registered)))
    monkeypatch.setattr(
        ModelRegistry, "register_model", staticmethod(lambda arch, target: registered.__setitem__(arch, target))
    )
    return registered


def test_tt_model_class_overrides_registers_bare_and_tt_names(monkeypatch):
    import vllm.config  # noqa: F401  (finish vLLM init before the plugin package)
    import vllm_tt_plugin.platform as tt_platform

    registered = _patched_registry(monkeypatch)
    monkeypatch.delenv("EXTRA_MODELS_DIR", raising=False)
    monkeypatch.setenv("TT_MODEL_CLASS_OVERRIDES", gv.TT_MODEL_CLASS_OVERRIDES)
    tt_platform.register_tt_models()
    assert registered["MotifForCausalLM"] == gv.MAIN_CLASS
    assert registered["TTMotifForCausalLM"] == gv.MAIN_CLASS


def test_extra_models_dir_alone_registers_only_the_tt_name(monkeypatch, tmp_path):
    """The bare-name trap (design 00 §1.4): a bundle alone leaves ``MotifForCausalLM`` to upstream vLLM."""
    import vllm.config  # noqa: F401
    import vllm_tt_plugin.platform as tt_platform

    bundle = tmp_path / "motif-3-bh-galaxy"
    bundle.mkdir()
    (bundle / "vllm_metadata.json").write_text((PACKAGE_DIR / "vllm_metadata.json").read_text())
    registered = _patched_registry(monkeypatch)
    monkeypatch.delenv("TT_MODEL_CLASS_OVERRIDES", raising=False)
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))  # the plugin appends the bundle folder
    tt_platform.register_tt_models()
    assert registered["TTMotifForCausalLM"] == gv.MAIN_CLASS
    assert "MotifForCausalLM" not in registered

    from vllm.model_executor.models.registry import _PREVIOUSLY_SUPPORTED_MODELS

    assert "MotifForCausalLM" in _PREVIOUSLY_SUPPORTED_MODELS  # why the override is needed


FORBIDDEN_CLASS_ATTRS = (
    "is_hybrid",
    "has_inner_state",
    "is_attention_free",
    "supports_multimodal",
    "supports_pp",
    "is_pooling_model",
    "has_noops",
    "attn_type",
    "supports_transcription",
    "requires_raw_input_tokens",
    "supports_mamba_prefix_caching",
    "_HYBRID_KV_CACHE_GROUPS_ENABLED",
    "tt_supported_decode_batch_sizes",
    "already_warmed_up_prefill",
    "note_state_slots_moved",
)


def test_vllm_inspects_a_plain_text_generation_model():
    from vllm.model_executor.models.registry import _ModelInfo

    info = _ModelInfo.from_model_cls(gv.MotifForCausalLM)
    assert info.is_text_generation_model and not info.is_pooling_model
    assert not (info.is_hybrid or info.has_inner_state or info.is_attention_free or info.has_noops)
    assert not (info.supports_multimodal or info.supports_pp or info.supports_transcription)
    assert info.attn_type == "decoder"
    present = [a for a in FORBIDDEN_CLASS_ATTRS if hasattr(gv.MotifForCausalLM, a)]
    assert not present, f"vLLM/plugin read these with getattr; do not define them: {present}"
    with pytest.raises(TypeError):
        gv.MotifForCausalLM(vllm_config=object())  # no vLLM-native constructor


def test_model_capabilities_are_explicit_class_level():
    caps = gv.MotifForCausalLM.model_capabilities
    assert caps == gv.model_capabilities_from_env()  # the import-time environment (identical in every process)
    assert caps["supports_device_penalties"] is False  # plugin default for an absent key is True
    for key in ("supports_async_decode", "supports_sample_on_device", "supports_async_spec_decode"):
        assert caps[key] is False, key
    assert caps["output_tokens_per_step"] == 1
    assert "max_device_top_k" not in caps and "fabric_config" not in caps
    assert gv.MotifForCausalLM.decode_input_update_contract == 1
    # Every switch off is exactly the draft-1 declaration.
    assert CAPS_OFF == {
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
        "supports_device_penalties": False,
        "supports_spec_decode": False,
        "supports_async_spec_decode": False,
        "output_tokens_per_step": 1,
    }


def test_feature_switches_gate_the_capabilities():
    """Features design §1.2: three bisection switches; they only allow a feature (vLLM's flags enable it)."""
    assert (
        CAPS_ON["supports_prefix_caching"] and CAPS_ON["supports_chunked_prefill"] and CAPS_ON["supports_spec_decode"]
    )
    assert CAPS_ON["spec_requirements"] == ("device_propose", "hidden_feed")
    assert CAPS_ON["spec_hidden_handoff"] == ("on_device",)  # keeps supports_narrow_decode
    assert CAPS_ON["supports_async_spec_decode"] is False and CAPS_ON["output_tokens_per_step"] == 1
    for name in api.FEATURE_SWITCHES:
        caps = gv.model_capabilities_from_env({**ALL_OFF, name: "on"})
        on = {k for k in ("supports_prefix_caching", "supports_chunked_prefill", "supports_spec_decode") if caps[k]}
        assert len(on) == 1, (name, on)
        assert ("spec_requirements" in caps) == (name == "MOTIF3_SPEC_DECODE")
    unset = gv.model_capabilities_from_env({})
    assert unset["supports_chunked_prefill"] is gv.FEATURE_SWITCH_DEFAULT
    assert gv.feature_switches({"MOTIF3_PREFIX_CACHING": " YES "})["MOTIF3_PREFIX_CACHING"] is True
    with pytest.raises(ValueError, match="MOTIF3_SPEC_DECODE"):
        gv.model_capabilities_from_env({"MOTIF3_SPEC_DECODE": "maybe"})  # a typo never silently turns a feature off
    # The normalisation the plugin applies to the two spec lists accepts them (a str would be refused).
    from vllm_tt_plugin.spec_decode import HIDDEN_HANDOFFS, SPEC_REQUIREMENTS, normalize_declared_values

    assert (
        normalize_declared_values(CAPS_ON["spec_requirements"], SPEC_REQUIREMENTS, "r") == CAPS_ON["spec_requirements"]
    )
    assert normalize_declared_values(CAPS_ON["spec_hidden_handoff"], HIDDEN_HANDOFFS, "h") == ("on_device",)


@pytest.mark.parametrize("value", ["1", "0", None, "bogus"])
def test_class_capabilities_follow_the_import_time_environment(value):
    """The plugin reads the dict from the CLASS in the API server and in EngineCore: the switches are read when the
    module is imported, in each process."""
    probe = (
        "import json\n"
        "from models.demos.motif3.tt.generator_vllm import MotifForCausalLM as M\n"
        "print(json.dumps(M.model_capabilities))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(METAL_ROOT))
    for name in api.FEATURE_SWITCHES:
        env.pop(name, None)
        if value is not None:
            env[name] = value
    res = subprocess.run([sys.executable, "-c", probe], cwd=METAL_ROOT, env=env, capture_output=True, text=True)
    if value == "bogus":
        assert res.returncode != 0 and "MOTIF3_" in res.stderr
        return
    assert res.returncode == 0, res.stderr[-3000:]
    caps = json.loads(res.stdout.strip().splitlines()[-1])
    on = {"1": True, "0": False, None: gv.FEATURE_SWITCH_DEFAULT}[value]
    for key in ("supports_prefix_caching", "supports_chunked_prefill", "supports_spec_decode"):
        assert caps[key] is on, key
    assert ("spec_requirements" in caps) is on


# ================================================================================================================
# 2. The real vLLM chain on the Motif config
# ================================================================================================================
@pytest.fixture(scope="module")
def motif_vllm_config(tmp_path_factory):
    """``VllmConfig`` for the real Motif-3 config, built the way ``vllm serve`` builds it on the TT platform, with
    every feature switch OFF (draft 1).

    ``ModelConfig`` inspects the registered class in vLLM's registry subprocess, which imports the bridge in a fresh
    interpreter (it inherits this device-hidden namespace and ``PYTHONPATH``)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TT_MODEL_CLASS_OVERRIDES", gv.TT_MODEL_CLASS_OVERRIDES)
        mp.setenv("VLLM_CACHE_ROOT", str(tmp_path_factory.mktemp("vllm_cache")))
        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.setenv("PYTHONPATH", str(METAL_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        for var in ("EXTRA_MODELS_DIR", "MOTIF3_KV_POOL_TOKENS", "MOTIF3_NUM_LAYERS", "MOTIF3_KV_CACHE_DTYPE"):
            mp.delenv(var, raising=False)
        mp.setattr(gv.MotifForCausalLM, "model_capabilities", CAPS_OFF)
        from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig

        import vllm_tt_plugin.platform as tt_platform

        tt_platform.register_tt_models()
        model_config = ModelConfig(model=str(_motif_dir()), trust_remote_code=True, max_model_len=32768, seed=0)
        resolved_arch = model_config.architecture
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=CacheConfig(block_size=64, enable_prefix_caching=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=32,
                max_num_batched_tokens=8192,
                max_model_len=32768,
                is_encoder_decoder=False,
                enable_chunked_prefill=True,
            ),
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            # l1_small_size is mandatory (get_max_tokens_all_users refuses a "tt" config without it)
            additional_config={"tt": {"trace_mode": "decode_only", "l1_small_size": api.L1_SMALL_SIZE}},
        )
        yield SimpleNamespace(vllm_config=vllm_config, model_config=model_config, resolved_arch=resolved_arch)


def test_vllm_resolves_motif_config_to_the_bridge(motif_vllm_config):
    from vllm.model_executor.model_loader import get_model_architecture

    vc, mc = motif_vllm_config.vllm_config, motif_vllm_config.model_config
    # ModelConfig (trust_remote_code) resolved the BARE name to us, not to TransformersMoEForCausalLM.
    assert motif_vllm_config.resolved_arch == "MotifForCausalLM"
    assert mc.hf_config.model_type == "Motif" and mc.trust_remote_code
    assert mc.hf_config.architectures == ["TTMotifForCausalLM"]  # rewritten by TTPlatform
    model_cls, _ = get_model_architecture(mc)
    assert model_cls is gv.MotifForCausalLM
    assert vc.parallel_config.worker_cls == "vllm_tt_plugin.worker.TTWorker"
    # Capabilities (every switch off) took effect at config time.
    assert vc.cache_config.enable_prefix_caching is False
    assert vc.scheduler_config.enable_chunked_prefill is False
    assert vc.scheduler_config.max_num_batched_tokens >= 32768
    assert not vc.scheduler_config.async_scheduling
    # What vLLM derives from the Motif config (study 05 §13).
    assert mc.is_moe and not mc.use_mla and not mc.uses_mrope
    assert mc.get_vocab_size() == 220160 and mc.max_model_len == 32768
    assert mc.get_num_layers_by_block_type(vc.parallel_config, "attention") == 53
    assert mc.get_sliding_window() == 128  # why the default FullAttentionSpec would be wrong


def test_kv_spec_pool_and_allocation_chain(motif_vllm_config):
    """Plugin sizing -> spec hook -> vLLM KV config -> runner hint -> bridge allocation -> vLLM BlockPool."""
    from vllm.config import set_current_vllm_config
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    from vllm_tt_plugin.model_runner import TTModelRunner
    from vllm_tt_plugin.worker import (
        TTWorker,
        _available_kv_cache_memory_bytes_for_num_blocks,
        get_num_available_blocks_tt,
    )

    vc, mc = motif_vllm_config.vllm_config, motif_vllm_config.model_config
    with set_current_vllm_config(vc):  # what EngineCore's init_device sees
        num_blocks = get_num_available_blocks_tt(vc, 32)
    assert num_blocks == (262144 + 64 * 32) // 64 + 1 == 4129

    spec = TTWorker._try_get_spec_from_model_hook(SimpleNamespace(model_config=mc, vllm_config=vc))
    assert len(spec) == 53
    assert all(isinstance(s, MLAAttentionSpec) for s in spec.values())
    first = spec["model.layers.0.self_attn"]
    assert (first.block_size, first.num_kv_heads, first.head_size, first.dtype) == (64, 1, 576, torch.bfloat16)
    assert first.sliding_window is None and first.page_size_bytes == 64 * 576 * 2

    available = _available_kv_cache_memory_bytes_for_num_blocks(vc, spec, num_blocks)
    vc.cache_config.num_gpu_blocks_override = num_blocks
    kv_cache_config = get_kv_cache_configs(vc, [spec], [available])[0]
    assert kv_cache_config.num_blocks == num_blocks
    assert len(kv_cache_config.kv_cache_groups) == 1
    assert len(kv_cache_config.kv_cache_groups[0].layer_names) == 53

    runner = SimpleNamespace(num_devices=32, tt_data_parallel_size=1)
    runner._kv_cache_shape = functools.partial(TTModelRunner._kv_cache_shape, runner)
    per_layer = TTModelRunner._build_per_layer_specs(runner, kv_cache_config, 53)
    assert [s[0] for s in per_layer] == [(4129, 1, 64, 576)] * 53
    assert [s[2] for s in per_layer] == list(range(53))

    settings = api.GeneratorSettings(max_batch_size=32, max_seq_len=32768, num_layers=53)
    gen = FakeMotifGenerator(settings, 220160)
    bridge = gv.MotifForCausalLM(gen, settings)
    kv = bridge.allocate_kv_cache_per_layer(per_layer)
    assert kv.shape == (4129, 1, 64, 576) and kv.num_layers == 53 and kv.kv_cache_dtype == "bfp8"
    assert kv.page_table_width == min(api.cdiv(32768, 64), num_blocks) == 512  # = plugin max_num_blocks_per_req
    assert kv.bytes_per_chip == 53 * 4129 * 2 * 18 * 1088 == 8_571_407_616  # design 00 §1.1: 8.57 GB
    assert kv.mtp_layers == 0 and kv.device_layers == 53
    assert gen.alloc_args == dict(num_blocks=4129, block_size=64, num_layers=53)

    pool = BlockPool(num_gpu_blocks=num_blocks, enable_caching=False, hash_block_size=64)
    per_user = api.cdiv(262144 // 32 + 1, 64)  # 8192 tokens + the next token's slot
    assert pool.get_num_free_blocks() == num_blocks - 1 == 32 * per_user
    # Without the reserve the plugin would allocate one block too few for the 32nd user.
    assert gv.plugin_num_blocks(262144, 64, 32) - 1 < 32 * per_user


def test_motif_tt_config_from_vllm_hf_config(motif_vllm_config, monkeypatch):
    """INFRA-1 + INFRA-2 on the real vLLM chain. ``MotifTTConfig`` built from vLLM's ``hf_config`` object (what
    ``MotifGenerator.create`` receives; transformers 5 moved YaRN into ``rope_parameters``), from a plain
    ``AutoConfig(trust_remote_code)`` object and from ``config.json`` are identical (fields, layer schedule, YaRN
    ``inv_freq``), and the config's expected KV block count is the one the plugin really allocates (4129)."""
    from transformers import AutoConfig
    from vllm.config import set_current_vllm_config

    from models.demos.motif3.tt.model_config import MotifTTConfig
    from models.demos.motif3.tt.rope import inv_freq_for_kind
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    for var in ("MOTIF3_MAX_MODEL_LEN", "MOTIF3_WEIGHTS_DIR", "HF_MODEL", "TT_MODEL_WEIGHTS_REVISION", "MOTIF3_FABRIC"):
        monkeypatch.delenv(var, raising=False)
    vc, mc = motif_vllm_config.vllm_config, motif_vllm_config.model_config
    path = _motif_dir()
    by_path = MotifTTConfig.from_hf_config(str(path), mesh_shape=(4, 8))
    by_vllm = MotifTTConfig.from_hf_config(mc.hf_config, mesh_shape=(4, 8))
    by_auto = MotifTTConfig.from_hf_config(
        AutoConfig.from_pretrained(str(path), trust_remote_code=True), mesh_shape=(4, 8)
    )
    ref = {f.name: getattr(by_path, f.name) for f in dataclasses.fields(by_path)}
    for name, cfg in (("vllm hf_config", by_vllm), ("AutoConfig", by_auto)):
        got = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
        assert got == ref, f"{name}: {[k for k in ref if got[k] != ref[k]]}"
        assert cfg.layers == by_path.layers, name
        assert torch.equal(inv_freq_for_kind(cfg, "yarn"), inv_freq_for_kind(by_path, "yarn")), name
    assert by_vllm.rope_type == "yarn" and by_vllm.yarn_factor == 64.0 and by_vllm.eos_token_ids == (0, 3, 6)

    with set_current_vllm_config(vc):
        num_blocks = get_num_available_blocks_tt(vc, 32)
    assert by_vllm.kv_num_blocks == num_blocks == 4129 == api.expected_num_blocks()
    settings = api.GeneratorSettings.from_env(mc.hf_config, max_batch_size=32, max_seq_len=32768, block_size=64)
    tt_cfg = MotifTTConfig.from_settings(settings, hf_config=mc.hf_config, mesh_shape=(4, 8))
    assert tt_cfg.kv_cache_shape == (num_blocks, 1, 64, 576) and tt_cfg.kv_blocks_per_seq == 512
    assert tt_cfg.kv_cache_bytes_per_chip() == api.kv_cache_bytes_per_chip(num_blocks, 64, 53, "bfp8")


def test_tokenizer_resolves_with_trust_remote_code(motif_vllm_config):
    from vllm.tokenizers import cached_tokenizer_from_config

    _motif_dir(require_tokenizer=True)
    tok = cached_tokenizer_from_config(motif_vllm_config.model_config)
    assert len(tok) == 220160
    assert tok.convert_tokens_to_ids(["<think>", "</think>", "<tool_call>", "</tool_call>"]) == [11, 12, 13, 14]
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    on = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    off = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert on.endswith("<|startofturn|><|assistant|><think>")
    assert off.endswith("<|startofturn|><|assistant|><think></think>")


def _mtp_cache_marker(tmp_path: Path) -> Path:
    """A TT weight-cache root holding a converted MTP part (``L53/.complete``): ``spec_plan`` then finds the MTP
    weights whatever checkpoint shards this host keeps (hermetic)."""
    cache = tmp_path / "tt_cache"
    part = cache / "motif3-test" / "mesh4x8" / "L53"
    part.mkdir(parents=True, exist_ok=True)
    (part / ".complete").write_text("{}")
    return cache


def _engine_config(monkeypatch, tmp_path, caps, **engine_kwargs):
    """``EngineArgs(...).create_engine_config()`` for the Motif config on the TT platform with ``caps`` as the
    class capabilities (what ``vllm serve`` builds in the API server, and EngineCore again in ``init_device``)."""
    from vllm.engine.arg_utils import EngineArgs

    import vllm_tt_plugin.platform as tt_platform

    monkeypatch.setenv("TT_MODEL_CLASS_OVERRIDES", gv.TT_MODEL_CLASS_OVERRIDES)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm_cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("PYTHONPATH", str(METAL_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for var in ("EXTRA_MODELS_DIR", "MOTIF3_KV_POOL_TOKENS", "MOTIF3_NUM_LAYERS", "MOTIF3_KV_CACHE_DTYPE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gv.MotifForCausalLM, "model_capabilities", caps)
    monkeypatch.setenv("MOTIF3_TT_CACHE_PATH", str(_mtp_cache_marker(tmp_path)))  # spec_plan finds the MTP weights
    tt_platform.register_tt_models()
    args = dict(
        model=str(_motif_dir()),
        trust_remote_code=True,
        max_model_len=32768,
        max_num_seqs=32,
        block_size=64,
        seed=0,
        additional_config={"tt": dict(api.SERVING_TT_CONFIG)},
    )
    args.update(engine_kwargs)
    return EngineArgs(**args).create_engine_config()


PRODUCTION_ENGINE_ARGS = dict(
    enable_chunked_prefill=True,
    max_num_batched_tokens=8128,
    long_prefill_token_threshold=8128,
    enable_prefix_caching=True,
    speculative_config=dict(gv.SPECULATIVE_CONFIG),
    async_scheduling=False,
)


def test_vllm_config_keeps_the_features_the_capabilities_allow(monkeypatch, tmp_path):
    """Features design §1.1-§1.2 on the real chain: with the switches on, vLLM's flags enable chunked prefill (8128 /
    8128), prefix caching and the model-owned MTP drafter; the platform admits our ``spec_plan`` (K = 1, PS-1 when the
    installed plugin has it); ``init_device``'s pool sizing captures it all and counts the MTP layer."""
    from vllm.config import set_current_vllm_config

    from vllm_tt_plugin.config import get_tt_spec_plan

    vc = _engine_config(monkeypatch, tmp_path, CAPS_ON, **PRODUCTION_ENGINE_ARGS)
    sched, cache = vc.scheduler_config, vc.cache_config
    assert sched.enable_chunked_prefill is True and cache.enable_prefix_caching is True
    assert (sched.max_num_batched_tokens, sched.long_prefill_token_threshold) == (8128, 8128)
    assert not sched.async_scheduling
    plan = get_tt_spec_plan(vc)
    assert plan is not None and plan.effective_k == 1 and vc.speculative_config.num_speculative_tokens == 1
    assert plan.accept_modes == ("argmax_ids",) and plan.drafter_state == "internal" and plan.supports_narrow_decode
    assert plan.lanes_per_request == 2 and plan.extra_bytes_per_seq == 0 and plan.extra_bytes_per_token == 612
    if gv.spec_plan_supports_speculable_rows():
        assert plan.verify_requires_speculable_rows is True
    with set_current_vllm_config(vc), loguru_messages() as seen:
        tokens = gv.MotifForCausalLM.get_max_tokens_all_users(num_devices=32, max_model_len=32768, max_num_seqs=32)
    assert tokens == 262144 + gv.NULL_BLOCK_RESERVE_TOKENS
    assert gv._SEEN_VLLM_SERVING == {
        "block_size": 64,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 8128,
        "long_prefill_token_threshold": 8128,
        "enable_prefix_caching": True,
        "prefix_match_unit": None,
        "spec_tokens": 1,
    }
    assert not [m for m in seen if m.startswith("Motif-3 serving config: ") and "chunked_prefill=" not in m]
    settings = api.GeneratorSettings.from_env(
        vc.model_config.hf_config, max_batch_size=32, max_seq_len=32768, serving=gv._SEEN_VLLM_SERVING
    )
    assert settings.resumed_prefill and settings.spec_decode and settings.kv_replicated
    assert settings.kv_write_mode == "all_split" and settings.mtp_kv_layers == 1

    # A draft length above 1 is served as K = 1: the platform publishes effective_k back.
    vc3 = _engine_config(
        monkeypatch, tmp_path, CAPS_ON, speculative_config={**gv.SPECULATIVE_CONFIG, "num_speculative_tokens": 3}
    )
    assert vc3.speculative_config.num_speculative_tokens == 1 and get_tt_spec_plan(vc3).effective_k == 1


def test_vllm_config_refuses_what_the_switches_do_not_allow(monkeypatch, tmp_path):
    """MOTIF3_*=0 is draft 1: the platform turns chunked prefill and prefix caching off and refuses speculation."""
    off = _engine_config(
        monkeypatch, tmp_path, CAPS_OFF, enable_chunked_prefill=True, enable_prefix_caching=True,
        max_num_batched_tokens=8128,
    )  # fmt: skip
    assert off.scheduler_config.enable_chunked_prefill is False and off.cache_config.enable_prefix_caching is False
    assert off.scheduler_config.long_prefill_token_threshold == 0
    with pytest.raises(Exception, match="supports_spec_decode"):
        _engine_config(monkeypatch, tmp_path, CAPS_OFF, speculative_config=dict(gv.SPECULATIVE_CONFIG))
    with pytest.raises(Exception, match="MTP layer"):  # spec_plan refuses any method but the model-owned drafter
        _engine_config(
            monkeypatch, tmp_path, CAPS_ON,
            speculative_config={"method": "ngram", "num_speculative_tokens": 1, "prompt_lookup_max": 3},
        )  # fmt: skip


# ================================================================================================================
# 3. Pool math, spec and allocation contracts
# ================================================================================================================
@pytest.mark.parametrize("block_size", api.SUPPORTED_BLOCK_SIZES)
def test_null_block_reserve_is_exactly_one_block(block_size, monkeypatch):
    monkeypatch.delenv("MOTIF3_KV_POOL_TOKENS", raising=False)
    tokens = gv.MotifForCausalLM.get_max_tokens_all_users(num_devices=32, max_model_len=32768, max_num_seqs=32)
    assert tokens == 262144 + gv.NULL_BLOCK_RESERVE_TOKENS
    blocks = gv.plugin_num_blocks(tokens, block_size, 32)
    assert blocks == 262144 // block_size + 32 + 1  # pool + one output block per user + vLLM's null block


def test_get_max_tokens_all_users_validation(monkeypatch):
    f = gv.MotifForCausalLM.get_max_tokens_all_users
    for var in ("MOTIF3_KV_POOL_TOKENS", "MOTIF3_KV_CACHE_DTYPE", "MOTIF3_NUM_LAYERS", "MOTIF3_KV_MAX_GB_PER_CHIP"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="tt_data_parallel"):
        f(num_devices=32, tt_data_parallel=4)
    with pytest.raises(ValueError, match="32-chip"):
        f(num_devices=8)
    with pytest.raises(ValueError, match="max-num-seqs"):
        f(num_devices=32, max_num_seqs=64)
    with pytest.raises(ValueError, match="max-model-len"):
        f(num_devices=32, max_model_len=262144)  # vLLM's derived default for Motif
    for bad in ("lots", "1000", "-5"):
        monkeypatch.setenv("MOTIF3_KV_POOL_TOKENS", bad)
        with pytest.raises(ValueError, match="MOTIF3_KV_POOL_TOKENS"):
            f(num_devices=32)
    monkeypatch.delenv("MOTIF3_KV_POOL_TOKENS")
    for bad_len in (5000, 32767, 128):  # BRIDGE-3: last bucket = max_model_len must be a whole number of SDPA chunks
        with pytest.raises(ValueError, match="multiple of 256"):
            f(num_devices=32, max_model_len=bad_len)
    with pytest.raises(ValueError, match="multiple of 256"):
        api.GeneratorSettings(max_seq_len=1000)
    assert f(num_devices=32, max_model_len=4096, max_num_seqs=32) == 262144 + 32
    monkeypatch.setenv("MOTIF3_KV_POOL_TOKENS", "16384")
    with pytest.raises(ValueError, match="does not fit"):
        f(num_devices=32, max_model_len=32768)
    monkeypatch.setenv("MOTIF3_KV_POOL_TOKENS", "393216")
    assert f(num_devices=32, max_model_len=32768, max_num_seqs=32) == 393216 + 32
    # A bf16 latent pool of 262,144 tokens needs ~16.2 GB per chip: refused before the weights load.
    monkeypatch.delenv("MOTIF3_KV_POOL_TOKENS")
    monkeypatch.setenv("MOTIF3_KV_CACHE_DTYPE", "bf16")
    with pytest.raises(ValueError, match="GB per chip"):
        f(num_devices=32, max_model_len=32768, max_num_seqs=32)
    monkeypatch.setenv("MOTIF3_KV_POOL_TOKENS", "131072")
    assert f(num_devices=32, max_model_len=32768, max_num_seqs=32) == 131072 + 32
    # A truncated bring-up run accounts only the layers it runs.
    monkeypatch.delenv("MOTIF3_KV_POOL_TOKENS")
    monkeypatch.setenv("MOTIF3_NUM_LAYERS", "3")
    assert f(num_devices=32, max_model_len=32768) == 262144 + 32


def test_get_max_tokens_all_users_rejects_bad_block_size_early(monkeypatch):
    """Inside EngineCore's init_device the current VllmConfig is visible, so --block-size 16 fails before loading."""
    from vllm.config import set_current_vllm_config

    monkeypatch.delenv("MOTIF3_KV_POOL_TOKENS", raising=False)
    fake = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=53)),
    )
    with set_current_vllm_config(fake), pytest.raises(ValueError, match="block-size"):
        gv.MotifForCausalLM.get_max_tokens_all_users(num_devices=32, max_model_len=32768)


def test_get_max_tokens_all_users_requires_l1_small(monkeypatch):
    """Attention P0: the plugin opens the mesh with the "tt" config's l1_small_size (none when absent), and the CCL
    semaphores need >= 32 KiB of L1_SMALL. A server started without it fails in init_device, before the weights load,
    with the fix in the message."""
    from vllm.config import set_current_vllm_config

    monkeypatch.delenv("MOTIF3_KV_POOL_TOKENS", raising=False)

    def fake(tt):
        return SimpleNamespace(
            cache_config=SimpleNamespace(block_size=64),
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=53)),
            additional_config=tt,
        )

    f = gv.MotifForCausalLM.get_max_tokens_all_users
    for bad in ({}, {"tt": {"trace_mode": "decode_only"}}, {"tt": {"l1_small_size": 16384}}):
        with set_current_vllm_config(fake(bad)), pytest.raises(ValueError, match="l1_small_size") as ei:
            f(num_devices=32, max_model_len=32768, max_num_seqs=32)
        assert "32768" in str(ei.value)
    with set_current_vllm_config(fake({"tt": dict(api.SERVING_TT_CONFIG)})):
        assert f(num_devices=32, max_model_len=32768, max_num_seqs=32) == 262144 + gv.NULL_BLOCK_RESERVE_TOKENS
    with set_current_vllm_config(fake({"tt": "not a dict"})), pytest.raises(ValueError, match="JSON object"):
        f(num_devices=32, max_model_len=32768)


def test_kv_cache_bytes_per_chip():
    assert api.kv_cache_bytes_per_chip(4129, 64, 53, "bfp8") == 8_571_407_616
    assert api.kv_cache_bytes_per_chip(4129, 64, 53, "bf16") == 4129 * 64 * 576 * 2 * 53
    assert api.kv_cache_bytes_per_chip(4129, 64, 14, "bfp8") * 53 == api.kv_cache_bytes_per_chip(4129, 64, 53) * 14
    with pytest.raises(ValueError):
        api.kv_cache_bytes_per_chip(10, 16, 1)


def _fake_vllm_config(block_size=64, cache_dtype="auto", num_layers=53, hf_overrides=None):
    hf = SimpleNamespace(model_type="Motif", kv_lora_rank=512, qk_rope_head_dim=64, num_hidden_layers=num_layers)
    for k, v in (hf_overrides or {}).items():
        setattr(hf, k, v)
    model_config = SimpleNamespace(
        hf_config=hf,
        hf_text_config=hf,
        dtype=torch.bfloat16,
        get_num_layers_by_block_type=lambda parallel_config, block_type="attention": num_layers,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(block_size=block_size, cache_dtype=cache_dtype),
        parallel_config=SimpleNamespace(),
    )


def test_kv_cache_spec_contract():
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    from vllm_tt_plugin.model_runner import _parse_layer_index

    spec = gv.MotifForCausalLM.get_kv_cache_spec(_fake_vllm_config())
    assert list(spec) == [f"model.layers.{i}.self_attn" for i in range(53)]  # no MTP layer: it is model-owned
    assert [_parse_layer_index(k) for k in spec] == list(range(53))
    assert len({v for v in spec.values()}) == 1  # uniform -> one vLLM KV cache group
    s = spec["model.layers.52.self_attn"]
    assert isinstance(s, MLAAttentionSpec) and (s.num_kv_heads, s.head_size, s.block_size) == (1, 576, 64)
    for bs in api.SUPPORTED_BLOCK_SIZES:
        assert (
            gv.MotifForCausalLM.get_kv_cache_spec(_fake_vllm_config(block_size=bs))[
                "model.layers.0.self_attn"
            ].block_size
            == bs
        )
    with pytest.raises(ValueError, match="block-size"):
        gv.MotifForCausalLM.get_kv_cache_spec(_fake_vllm_config(block_size=16))  # vLLM's default
    with pytest.raises(ValueError, match="576"):
        gv.MotifForCausalLM.get_kv_cache_spec(_fake_vllm_config(hf_overrides={"kv_lora_rank": 128}))
    fp8 = gv.MotifForCausalLM.get_kv_cache_spec(_fake_vllm_config(cache_dtype="fp8"))["model.layers.0.self_attn"]
    assert fp8.dtype != torch.bfloat16  # vLLM bookkeeping follows --kv-cache-dtype; the device dtype does not


def test_allocate_kv_cache_contract(monkeypatch):
    monkeypatch.delenv("MOTIF3_KV_MAX_GB_PER_CHIP", raising=False)
    bridge, gen = _bridge(num_layers=53, max_seq_len=32768)
    with pytest.raises(ValueError, match="FullAttentionSpec"):
        bridge.allocate_kv_cache((4129, 1, 64, 192), torch.bfloat16, 53)  # the plugin's default-spec hint
    with pytest.raises(ValueError, match="576"):
        bridge.allocate_kv_cache((4129, 16, 64, 576), torch.bfloat16, 53)
    with pytest.raises(ValueError, match="block-size"):
        bridge.allocate_kv_cache((4129, 1, 16, 576), torch.bfloat16, 53)
    with pytest.raises(ValueError, match="generator runs"):
        bridge.allocate_kv_cache((4129, 1, 64, 576), torch.bfloat16, 52)
    with pytest.raises(ValueError, match="uniform"):
        bridge.allocate_kv_cache_per_layer(
            [((4129, 1, 64, 576), torch.bfloat16, 0), ((4129, 1, 32, 576), torch.bfloat16, 1)]
        )
    with pytest.raises(ValueError, match="share"):
        bridge.allocate_kv_cache_per_layer(
            [((4129, 1, 64, 576), torch.bfloat16, 0), ((4129, 1, 64, 576), torch.bfloat16, 0)]
        )
    assert gen.alloc_args is None  # nothing reached the generator
    kv = bridge.allocate_kv_cache((4129, 1, 64, 576), torch.bfloat16, 53)
    assert isinstance(kv, gv.MotifKVCache) and kv.device_cache is gen.handle
    with pytest.raises(RuntimeError, match="twice"):
        bridge.allocate_kv_cache((4129, 1, 64, 576), torch.bfloat16, 53)
    # Truncated bring-up run: vLLM still accounts 53 layers; the generator allocates the 3 it runs.
    bridge3, gen3 = _bridge(num_layers=3, max_seq_len=32768)
    kv3 = bridge3.allocate_kv_cache_per_layer([((4129, 1, 64, 576), torch.bfloat16, i) for i in range(53)])
    assert kv3.num_layers == 3 and kv3.vllm_num_layers == 53 and gen3.alloc_args["num_layers"] == 3
    # Memory budget: a bf16 pool of this size does not fit next to the weights.
    bridge_bf16, _ = _bridge(num_layers=53, max_seq_len=32768, kv_dtype="bf16")
    with pytest.raises(ValueError, match="GB per chip"):
        bridge_bf16.allocate_kv_cache((4129, 1, 64, 576), torch.bfloat16, 53)
    other = gv.MotifKVCache(1, 64, 1, 1, "bfp8", None, 1, 0, None)
    with pytest.raises(ValueError, match="allocate_kv_cache returned"):
        bridge.decode_forward(
            tokens=torch.zeros(32, 1, dtype=torch.int32),
            start_pos=torch.full((32,), -1),
            page_table=torch.zeros(32, 512, dtype=torch.int32),
            kv_cache=other,
        )


def test_allocation_and_pool_sizing_count_the_mtp_cache(monkeypatch):
    """Speculation adds one model-owned [N, 1, bs, 576] cache: vLLM keeps accounting 53 layers (the generator is
    asked for 53 and adds the MTP cache itself), the bridge's per-chip checks count 54 (+612 B per token per chip in
    bfp8 = SpecPlan.extra_bytes_per_token)."""
    from vllm.config import set_current_vllm_config

    for var in ("MOTIF3_KV_MAX_GB_PER_CHIP", "MOTIF3_KV_POOL_TOKENS", "MOTIF3_KV_CACHE_DTYPE", "MOTIF3_NUM_LAYERS"):
        monkeypatch.delenv(var, raising=False)
    bridge, gen = _bridge(num_layers=53, max_seq_len=32768, spec_tokens=1)
    kv = bridge.allocate_kv_cache((4129, 1, 64, 576), torch.bfloat16, 53)
    assert gen.alloc_args == dict(num_blocks=4129, block_size=64, num_layers=53)
    assert kv.mtp_layers == 1 and kv.device_layers == 54 and kv.num_layers == 53
    assert kv.bytes_per_chip == api.kv_cache_bytes_per_chip(4129, 64, 54, "bfp8") == 54 * 4129 * 2 * 18 * 1088
    assert kv.bytes_per_chip - api.kv_cache_bytes_per_chip(4129, 64, 53) == 4129 * 64 * gv.mtp_extra_bytes_per_token(
        "bfp8"
    )
    # A KV budget between the 53- and the 54-layer pool refuses the speculating launch only.
    gb = (api.kv_cache_bytes_per_chip(4129, 64, 53) + api.kv_cache_bytes_per_chip(4129, 64, 54)) / 2e9
    monkeypatch.setenv("MOTIF3_KV_MAX_GB_PER_CHIP", f"{gb:.6f}")
    with pytest.raises(ValueError, match="MTP layer"):
        _bridge(num_layers=53, max_seq_len=32768, spec_tokens=1)[0].allocate_kv_cache((4129, 1, 64, 576), None, 53)
    _bridge(num_layers=53, max_seq_len=32768)[0].allocate_kv_cache((4129, 1, 64, 576), None, 53)
    f = gv.MotifForCausalLM.get_max_tokens_all_users
    with set_current_vllm_config(_serving_vllm_config(spec_k=1)), pytest.raises(ValueError, match="MTP layer"):
        f(num_devices=32, max_model_len=32768, max_num_seqs=32)
    with set_current_vllm_config(_serving_vllm_config(spec_k=0)):
        assert f(num_devices=32, max_model_len=32768, max_num_seqs=32) == 262144 + 32


# ================================================================================================================
# 4. initialize_vllm_model
# ================================================================================================================
@pytest.fixture
def fake_generator_class(monkeypatch):
    module = types.ModuleType("motif3_host_test_fake_generator")
    module.FakeMotifGenerator = FakeMotifGenerator
    module.FakeDraft1Generator = FakeDraft1Generator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("MOTIF3_GENERATOR_CLASS", f"{module.__name__}:FakeMotifGenerator")
    for var in ("MOTIF3_NUM_LAYERS", "MOTIF3_KV_CACHE_DTYPE", "TT_CACHE_PATH", "MOTIF3_TT_CACHE_PATH",
                "TT_MODEL_WEIGHTS_REVISION"):  # fmt: skip
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(FakeDraft1Generator, "created", 0)
    return FakeMotifGenerator


@pytest.fixture(scope="module")
def motif_hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(_motif_dir()), trust_remote_code=True)


def test_initialize_vllm_model_builds_the_generator(fake_generator_class, motif_hf_config, monkeypatch):
    snapshot = str(_motif_dir())
    monkeypatch.setenv("HF_MODEL", snapshot)
    monkeypatch.setenv("TT_CACHE_PATH", "/tt_cache/motif3")
    monkeypatch.delenv("MOTIF3_WEIGHTS_DIR", raising=False)
    mesh = SimpleNamespace(shape=(4, 8))
    model = gv.MotifForCausalLM.initialize_vllm_model(
        motif_hf_config, mesh, max_batch_size=32, max_seq_len=32768, tt_data_parallel=1, optimizations=None
    )
    assert isinstance(model, gv.MotifForCausalLM)
    gen = model.generator
    assert isinstance(gen, fake_generator_class) and gen.mesh_device is mesh and gen.hf_config is motif_hf_config
    s = gen.settings
    assert (s.max_batch_size, s.max_seq_len, s.num_layers, s.kv_cache_dtype) == (32, 32768, 53, "bfp8")
    assert (s.weights_path, s.weights_source, s.cache_path) == (snapshot, "HF_MODEL", "/tt_cache/motif3")
    assert s.weights_are_local and s.block_size is None
    assert not (s.chunked_prefill or s.prefix_caching or s.spec_decode)  # no scheduler config seen: draft 1
    assert model.vocab_size == 220160
    monkeypatch.setenv("HF_MODEL", "/snapshots/does-not-exist")  # a path typo must not fall through to another source
    with pytest.raises(ValueError, match="HF_MODEL"):
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768)
    monkeypatch.setenv("HF_MODEL", snapshot)
    with pytest.raises(ValueError, match="multiple of 256"):  # BRIDGE-3, also before any weight is loaded
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 5000)

    # The plugin's BH-Galaxy preset opens (8, 4); the model detects the TP axis itself.
    gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, SimpleNamespace(shape=(8, 4)), 32, 32768)
    monkeypatch.setenv("MOTIF3_NUM_LAYERS", "3")
    small = gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 8, 4096)
    assert (small.generator.num_layers, small.settings.max_batch_size, small._lanes.num_slots) == (3, 8, 8)
    monkeypatch.delenv("MOTIF3_NUM_LAYERS")

    with pytest.raises(ValueError, match="MESH_DEVICE"):
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, SimpleNamespace(shape=(1, 32)), 32, 32768)
    with pytest.raises(ValueError, match="DP=1"):
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768, tt_data_parallel=4)
    with pytest.raises(ValueError, match="max_batch_size"):
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 64, 32768)
    with pytest.raises(ValueError, match="max_seq_len"):
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 65536)
    monkeypatch.setenv("MOTIF3_GENERATOR_CLASS", "models.demos.motif3.tt.generator_api:GeneratorSettings")
    with pytest.raises(TypeError, match="MotifGenerator"):
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768)


def test_block_size_reaches_generator_settings(fake_generator_class, motif_hf_config, monkeypatch):
    """BRIDGE-4: vLLM's --block-size reaches GeneratorSettings, either from a current VllmConfig or from the one
    get_max_tokens_all_users saw in init_device (initialize_vllm_model runs in load_model, with no current config).
    The allocation hint stays authoritative: a different block size there only warns."""
    from vllm.config import set_current_vllm_config

    monkeypatch.setenv("HF_MODEL", str(_motif_dir()))
    for var in ("MOTIF3_KV_POOL_TOKENS", "MOTIF3_KV_MAX_GB_PER_CHIP", "MOTIF3_WEIGHTS_DIR"):
        monkeypatch.delenv(var, raising=False)
    mesh = SimpleNamespace(shape=(4, 8))
    fake = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=32),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=53)),
    )
    with set_current_vllm_config(fake):
        assert gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768).settings.block_size == 32
        gv.MotifForCausalLM.get_max_tokens_all_users(num_devices=32, max_model_len=32768, max_num_seqs=32)
    model = gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768)  # what load_model sees
    assert model.settings.block_size == 32
    kv = model.allocate_kv_cache((4129, 1, 64, 576), torch.bfloat16, 53)  # hint wins (warning logged)
    assert model.generator.alloc_args["block_size"] == 64 and kv.block_size == 64
    with pytest.raises(ValueError, match="block-size"):
        api.GeneratorSettings(block_size=128)  # BRIDGE-1


def test_initialize_vllm_model_requires_an_l1_small_mesh(fake_generator_class, motif_hf_config, monkeypatch):
    """The bridge checks the mesh the plugin really opened (whatever launcher started it) before create()."""
    monkeypatch.setenv("HF_MODEL", str(_motif_dir()))
    monkeypatch.delenv("MOTIF3_WEIGHTS_DIR", raising=False)
    mesh = SimpleNamespace(shape=(4, 8))
    assert gv._mesh_l1_small_bytes(mesh) is None  # a host fake is never queried (ttnn is not even needed)
    for size in (0, 16384):
        monkeypatch.setattr(gv, "_mesh_l1_small_bytes", lambda m, size=size: size)
        with pytest.raises(ValueError, match="l1_small_size"):
            gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768)
    for size in (32768, 65536, None):
        monkeypatch.setattr(gv, "_mesh_l1_small_bytes", lambda m, size=size: size)
        model = gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768)
        assert model.generator.mesh_device is mesh


def _serving_vllm_config(
    *, block_size=64, chunked=True, budget=8128, threshold=8128, prefix=True, unit=None, spec_k=1, tt=None
):
    """A partial VllmConfig with the scheduler facts ``serving_config_of`` reads (what init_device sees)."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size, enable_prefix_caching=prefix, prefix_match_unit=unit),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=chunked, max_num_batched_tokens=budget, long_prefill_token_threshold=threshold
        ),
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=spec_k, method=gv.SPEC_METHOD) if spec_k is not None else None
        ),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=53)),
        additional_config={"tt": dict(api.SERVING_TT_CONFIG) if tt is None else tt},
    )


def test_serving_config_reaches_the_generator_settings(fake_generator_class, motif_hf_config, monkeypatch):
    """Features design §3.9 item 2-3: init_device's get_max_tokens_all_users captures vLLM's scheduler config;
    initialize_vllm_model (load_model, no current config) builds GeneratorSettings from it."""
    from vllm.config import set_current_vllm_config

    monkeypatch.setenv("HF_MODEL", str(_motif_dir()))
    for var in ("MOTIF3_KV_POOL_TOKENS", "MOTIF3_KV_MAX_GB_PER_CHIP", "MOTIF3_WEIGHTS_DIR"):
        monkeypatch.delenv(var, raising=False)
    mesh = SimpleNamespace(shape=(4, 8))
    with set_current_vllm_config(_serving_vllm_config(budget=8128, threshold=8128, unit=64)):
        gv.MotifForCausalLM.get_max_tokens_all_users(num_devices=32, max_model_len=32768, max_num_seqs=32)
    assert gv._SEEN_VLLM_SERVING["spec_tokens"] == 1 and gv._SEEN_VLLM_SERVING["prefix_match_unit"] == 64
    model = gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768)
    s = model.settings
    assert (s.chunked_prefill, s.prefix_caching, s.spec_tokens) == (True, True, 1)
    assert (s.max_num_batched_tokens, s.long_prefill_token_threshold, s.block_size) == (8128, 8128, 64)
    assert s.kv_replicated and s.kv_write_mode == "all_split" and model._spec
    assert model.generator.settings is s
    # MOTIF3_PREFILL_MAX_BUCKET reaches the generator (its span cap); 32768 restores single-shot buckets.
    monkeypatch.setenv("MOTIF3_PREFILL_MAX_BUCKET", "2048")
    model = gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768)
    assert model.settings.prefill_span_cap == 2048 and model.generator.max_prefill_span == 2048


def test_initialize_refuses_a_generator_without_the_enabled_features(
    fake_generator_class, motif_hf_config, monkeypatch
):
    """Features design §1.5 last row: a generator class that keeps MotifGenerator's "unsupported" default for a
    feature vLLM enabled is refused BEFORE create() loads the weights; a lying instance right after create()."""
    from vllm.config import set_current_vllm_config

    monkeypatch.setenv("HF_MODEL", str(_motif_dir()))
    monkeypatch.delenv("MOTIF3_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("MOTIF3_GENERATOR_CLASS", "motif3_host_test_fake_generator:FakeDraft1Generator")
    mesh = SimpleNamespace(shape=(4, 8))
    cases = (
        (dict(chunked=True, prefix=False, spec_k=None), "chunked prefill"),
        (dict(chunked=False, prefix=True, spec_k=None), "prefix caching"),
        (dict(chunked=False, prefix=False, spec_k=1), "speculative decoding"),
    )
    for kw, what in cases:
        with set_current_vllm_config(_serving_vllm_config(**kw)), pytest.raises(ValueError, match=what) as ei:
            gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768)
        assert "MOTIF3_" in str(ei.value)
    assert FakeDraft1Generator.created == 0  # refused before any weight was loaded
    with set_current_vllm_config(_serving_vllm_config(chunked=False, prefix=False, spec_k=None)):
        draft1 = gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768)  # features off: fine
    assert isinstance(draft1.generator, FakeDraft1Generator) and FakeDraft1Generator.created == 1

    class Liar(FakeMotifGenerator):  # overrides the property (unknowable before an instance exists), returns False
        @property
        def supports_spec_decode(self):
            return False

    spec_settings = api.GeneratorSettings(spec_tokens=1)
    gv.precheck_generator_class(Liar, spec_settings)  # an overridden property is only known on the instance
    with pytest.raises(ValueError, match="supports_spec_decode"):
        gv.MotifForCausalLM(Liar(spec_settings, 4096), spec_settings)
    with pytest.raises(ValueError, match="resumed prefill"):
        gv.MotifForCausalLM(*_draft1_pair(chunked_prefill=True))


def _draft1_pair(**features):
    settings = api.GeneratorSettings(max_seq_len=1024, num_layers=3, **features)
    return FakeDraft1Generator(settings, 4096), settings


def test_real_generator_class_imports_device_free():
    """GEN-7(c): the default MOTIF3_GENERATOR_CLASS module imports without a device and without other demo packages
    (checked once the integration wave's tt/generator.py exists)."""
    path = METAL_ROOT / "models" / "demos" / "motif3" / "tt" / "generator.py"
    if not path.is_file():
        pytest.skip("models/demos/motif3/tt/generator.py does not exist yet (integration wave)")
    module, _, cls = gv.DEFAULT_GENERATOR_CLASS.partition(":")
    probe = (
        "import json, sys, importlib\n"
        f"m = importlib.import_module({module!r})\n"
        "from models.demos.motif3.tt.generator_api import MotifGenerator\n"
        f"c = getattr(m, {cls!r})\n"
        "mods = sorted(k for k in sys.modules if k.startswith('models.'))\n"
        "print(json.dumps({'models': mods, 'subclass': issubclass(c, MotifGenerator), 'vllm': 'vllm' in sys.modules}))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(METAL_ROOT))
    res = subprocess.run([sys.executable, "-c", probe], cwd=METAL_ROOT, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
    info = json.loads(res.stdout.strip().splitlines()[-1])
    foreign = [
        m for m in info["models"] if m not in ("models", "models.demos") and not m.startswith("models.demos.motif3")
    ]
    assert not foreign and info["subclass"] and info["vllm"] is False, info
    for marker in ("Opening user mode device driver", "Starting devices in cluster"):
        assert marker not in res.stderr and marker not in res.stdout


def test_default_generator_class_serves_the_default_capabilities():
    """Final bridge <-> generator wiring (features design §6 merge order): with the switches unset the class allows
    all three features (``FEATURE_SWITCH_DEFAULT`` is on), and the default ``MOTIF3_GENERATOR_CLASS``
    (``tt/generator.py``) serves every one of them: ``precheck_generator_class`` accepts it for chunked prefill +
    prefix caching + speculation (it overrides each "unsupported" default), and every generator call the bridge makes
    exists with the keywords the bridge passes. In a subprocess: the real class imports ttnn."""
    assert gv.FEATURE_SWITCH_DEFAULT is True
    assert gv.model_capabilities_from_env({}) == CAPS_ON
    path = METAL_ROOT / "models" / "demos" / "motif3" / "tt" / "generator.py"
    if not path.is_file():
        pytest.skip("models/demos/motif3/tt/generator.py does not exist yet (integration wave)")
    module, _, cls = gv.DEFAULT_GENERATOR_CLASS.partition(":")
    calls = {  # generator call -> the keywords generator_vllm passes (or the positional names it relies on)
        "create": ["hf_config", "mesh_device", "settings"],
        "allocate_kv_cache": ["num_blocks", "block_size", "num_layers"],
        "prefill_forward_batch": ["requests", "kv_cache", "enable_trace"],
        "decode_forward": ["batch", "kv_cache", "enable_trace"],
        "decode_forward_spec": ["batch", "kv_cache", "enable_trace", "want_logits"],
        "warmup_prefill": ["kv_cache", "enable_trace"],
        "warmup_decode": ["kv_cache", "enable_trace", "page_table_width"],
        "release_lane": ["lane"],
        "release_traces": [],
        "close": [],
    }
    overridable = ("supports_resumed_prefill", "prefill_alignment", "max_prefill_span", "supports_spec_decode",
                   "prefill_forward_batch", "decode_forward_spec")  # fmt: skip
    probe = (
        "import importlib, inspect, json\n"
        "from models.demos.motif3.tt import generator_api as api, generator_vllm as gv\n"
        f"c = getattr(importlib.import_module({module!r}), {cls!r})\n"
        "s = api.GeneratorSettings(block_size=64, chunked_prefill=True, prefix_caching=True,\n"
        "    max_num_batched_tokens=8128, long_prefill_token_threshold=8128, spec_tokens=1)\n"
        "gv.precheck_generator_class(c, s)\n"
        f"calls = {calls!r}\n"
        "missing = {n: [k for k in kw if k not in inspect.signature(getattr(c, n)).parameters]\n"
        "           for n, kw in calls.items()}\n"
        f"over = [n for n in {overridable!r} if inspect.getattr_static(c, n) is not api.MotifGenerator.__dict__[n]]\n"
        "print(json.dumps({'missing': {k: v for k, v in missing.items() if v}, 'overrides': over,\n"
        "                  'kv_write_mode': s.kv_write_mode}))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(METAL_ROOT))
    res = subprocess.run([sys.executable, "-c", probe], cwd=METAL_ROOT, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
    info = json.loads(res.stdout.strip().splitlines()[-1])
    assert info["missing"] == {}, info
    assert sorted(info["overrides"]) == sorted(overridable), info
    assert info["kv_write_mode"] == "all_split"


def test_supported_block_sizes_are_the_gated_ones():
    """BRIDGE-1: only the block sizes gates G1 / G7 validated (32, 64)."""
    assert api.SUPPORTED_BLOCK_SIZES == (32, 64)
    for bs in (16, 128, 48):
        with pytest.raises(ValueError, match="block-size"):
            gv.validate_block_size(bs)
        with pytest.raises(ValueError, match="block-size"):
            gv.MotifForCausalLM.get_kv_cache_spec(_fake_vllm_config(block_size=bs))
    bridge, _ = _bridge()
    with pytest.raises(ValueError, match="block-size"):
        bridge.allocate_kv_cache((4129, 1, 128, 576), torch.bfloat16, 3)


def test_weights_location_precedence(tmp_path):
    """BRIDGE-2: MOTIF3_WEIGHTS_DIR > HF_MODEL (dir) > HF-cache snapshot of a repo-id HF_MODEL at
    TT_MODEL_WEIGHTS_REVISION > hf_config._name_or_path; the same order MotifTTConfig uses."""
    f = api.resolve_weights_location
    a, b, c = (tmp_path / n for n in "abc")
    for d in (a, b, c):
        d.mkdir()
    hub = tmp_path / "hub"
    sha = "2ed2ed5cfabffa10fdabb2fc0d0288f8e6de893a"
    snap = hub / "models--Motif-Technologies--Motif-3" / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    cfg_obj = SimpleNamespace(_name_or_path=str(c))
    base = {"HF_HUB_CACHE": str(hub)}
    assert f(cfg_obj, base) == api.WeightsLocation(str(c), "hf_config._name_or_path", None, True)
    assert f(None, base) == api.WeightsLocation(None, "default", None, False)
    assert f(cfg_obj, {**base, "HF_MODEL": str(b)}).path == str(b)
    assert f(cfg_obj, {**base, "HF_MODEL": str(b), "MOTIF3_WEIGHTS_DIR": str(a)}).source == "MOTIF3_WEIGHTS_DIR"
    repo = {**base, "HF_MODEL": "Motif-Technologies/Motif-3", "TT_MODEL_WEIGHTS_REVISION": sha}
    loc = f(cfg_obj, repo)
    assert (loc.path, loc.is_local, loc.revision) == (str(snap), True, sha) and "HF cache" in loc.source
    assert f(cfg_obj, {**repo, "TT_MODEL_WEIGHTS_REVISION": sha[:8]}).path == str(snap)  # short hashes resolve too
    (hub / "models--Motif-Technologies--Motif-3" / "refs").mkdir()
    (hub / "models--Motif-Technologies--Motif-3" / "refs" / "main").write_text(sha)
    assert f(cfg_obj, {**base, "HF_MODEL": "Motif-Technologies/Motif-3"}).path == str(snap)  # refs/main
    missing = f(cfg_obj, {**repo, "TT_MODEL_WEIGHTS_REVISION": "0" * 40})
    assert (missing.path, missing.is_local) == ("Motif-Technologies/Motif-3", False) and "not cached" in missing.source
    with pytest.raises(ValueError, match="MOTIF3_WEIGHTS_DIR"):
        f(cfg_obj, {**base, "MOTIF3_WEIGHTS_DIR": str(tmp_path / "nope")})
    with pytest.raises(ValueError, match="HF_MODEL"):
        f(cfg_obj, {**base, "HF_MODEL": str(tmp_path / "nope")})
    s = api.GeneratorSettings.from_env(cfg_obj, max_batch_size=8, max_seq_len=4096, environ=repo)
    assert (s.weights_path, s.weights_revision, s.weights_are_local) == (str(snap), sha, True)
    assert "HF cache" in s.weights_source and s.block_size is None


# ================================================================================================================
# 5. Prefill / decode plumbing with the plugin's slot bookkeeping
# ================================================================================================================
def _allocated_bridge(num_slots, block_size=32, num_blocks=640, max_seq_len=1024, vocab=4096, ps1=False, **features):
    bridge, gen = _bridge(num_slots=num_slots, max_seq_len=max_seq_len, vocab=vocab, **features)
    kv = bridge.allocate_kv_cache((num_blocks, 1, block_size, 576), torch.bfloat16, 3)
    driver = PluginDriver(
        bridge,
        kv,
        num_slots=num_slots,
        block_size=block_size,
        num_blocks=num_blocks,
        width=kv.page_table_width,
        vocab=vocab,
        ps1=ps1,
    )
    return bridge, gen, kv, driver


@pytest.mark.parametrize("num_slots", [32, 8])
def test_prefill_decode_plumbing_follows_slots(num_slots):
    """Random serving traffic: prefills into free slots, decodes with row reorders (non-identity slot_remap),
    finishes with condense, preemption + resume. Every sampled token must match the fake model's ground truth,
    which it only can if each request keeps its DP group between decode steps."""
    bridge, gen, kv, d = _allocated_bridge(num_slots)
    rng = random.Random(1234 + num_slots)
    next_id, waiting, preempted = 0, [], []

    def new_request():
        nonlocal next_id
        rid = f"r{next_id}"
        next_id += 1
        d.add(rid, [rng.randrange(1, 4000) for _ in range(rng.randrange(1, 70))])
        return rid

    for _ in range(min(3, num_slots)):
        waiting.append(new_request())
    d.prefill(waiting)
    for _ in range(60):
        if not d.order:
            d.prefill([new_request()])
        running = len(d.order)
        if preempted and running < num_slots and rng.random() < 0.3:
            d.prefill([preempted.pop(0)])  # resume: full re-prefill of prompt + generated tokens
        elif running < num_slots and rng.random() < 0.35:
            d.prefill([new_request() for _ in range(rng.randrange(1, min(4, num_slots - running) + 1))])
        rows = list(d.order)
        if rng.random() < 0.4:
            rng.shuffle(rows)  # any row order must work: the remap tells the bridge who is where
        d.decode(rows)
        if d.order and rng.random() < 0.2:
            d.finish(rng.choice(d.order))
        if len(d.order) > 1 and rng.random() < 0.1:
            victim = rng.choice(d.order)
            d.preempt(victim)
            preempted.append(victim)
    assert d.remaps > 5, "the traffic never exercised a non-identity slot_remap"
    assert d.lane_checks > 100, "decode never re-checked lane stability"
    assert gen.released_lanes, "finish/preempt must release the request's lane"
    lanes = bridge._lanes.slot_to_lane
    assert len(set(lanes)) == num_slots and all(0 <= lane < api.NUM_LANES for lane in lanes)


def test_lanes_spread_over_dp_groups_and_stay_put():
    bridge, gen, kv, d = _allocated_bridge(32)
    for i in range(4):
        d.add(f"a{i}", [5 + i] * (10 + i))
    slots = d.prefill([f"a{i}" for i in range(4)])
    assert slots == [0, 1, 2, 3]
    assert [lane for lane, _, _ in gen.prefills] == [0, 8, 16, 24]  # one request per DP group
    assert len(gen.prefill_calls) == 1 and len(gen.prefill_calls[0]["rows"]) == 4  # ONE batch call for the step
    d.decode()
    assert gen.decode_steps[-1][0] == [0, 8, 16, 24]
    d.decode(["a3", "a2", "a1", "a0"])  # rows reversed -> remap; lanes do not move
    assert gen.decode_steps[-1][0] == [0, 8, 16, 24]
    assert [bridge._lanes.lane_of_slot(d.runner._req_state_slot[f"a{i}"]) for i in range(4)] == [0, 8, 16, 24]


def test_decode_failure_does_not_commit_the_remap():
    bridge, gen, kv, d = _allocated_bridge(8)
    for i in range(3):
        d.add(f"b{i}", [11 * (i + 1)] * 5)
    d.prefill(["b0", "b1", "b2"])
    d.decode()
    before = bridge._lanes.slot_to_lane
    rows = ["b2", "b0", "b1"]
    remap = d.R._decode_state_slot_remap(d.runner, rows)
    assert remap is not None
    gen.fail_next_decode = True
    with pytest.raises(RuntimeError, match="injected"):
        bridge.decode_forward(
            tokens=torch.zeros(8, 1, dtype=torch.int32),
            start_pos=torch.tensor([len(d.seqs[r]) - 1 for r in rows] + [-1] * 5, dtype=torch.int32),
            page_table=torch.stack([d._row_table(r) for r in rows] + [torch.zeros(d.width, dtype=torch.int32)] * 5),
            kv_cache=kv,
            slot_remap=remap,
            reload_inputs=True,
        )
    assert bridge._lanes.slot_to_lane == before  # a refused decode never moved anything
    d.runner._pending_state_slot_settle = None  # the plugin drops the pending map when the call raised
    d.runner._pending_state_slot_moves = None
    d.decode(rows)  # and the next attempt works from the unchanged state
    d.decode()


def test_decode_contract_rejections():
    bridge, gen, kv, d = _allocated_bridge(8)
    d.add("c0", [3, 4, 5])
    d.prefill(["c0"])
    base = dict(
        tokens=torch.zeros(8, 1, dtype=torch.int32),
        start_pos=torch.tensor([3] + [-1] * 7, dtype=torch.int32),
        page_table=torch.stack([d._row_table("c0")] + [torch.zeros(d.width, dtype=torch.int32)] * 7),
        kv_cache=kv,
    )
    with pytest.raises(TypeError, match="reset_batch"):
        bridge.decode_forward(**base, reset_batch=False)
    with pytest.raises(NotImplementedError, match="reload"):
        bridge.decode_forward(**base, reload_inputs=False, reload_page_table=True)
    with pytest.raises(ValueError, match="reload_page_table"):
        bridge.decode_forward(**base, reload_inputs=True, reload_page_table=True)
    with pytest.raises(NotImplementedError, match="host"):
        bridge.decode_forward(**base, sampling_params=SimpleNamespace(temperature=[0.0] * 8))
    with pytest.raises(NotImplementedError, match="page_tables_per_layer"):
        bridge.decode_forward(**base, page_tables_per_layer=[base["page_table"]])
    with pytest.raises(NotImplementedError, match="speculative"):
        bridge.decode_forward(**base, num_valid_drafts=torch.zeros(8, dtype=torch.int32))
    with pytest.raises(NotImplementedError, match="one token per row"):
        bridge.decode_forward(**{**base, "tokens": torch.zeros(8, 2, dtype=torch.int32)})
    with pytest.raises(ValueError, match="permutation"):
        bridge.decode_forward(**base, slot_remap=torch.tensor([0, 0, 1, 2, 3, 4, 5, 6], dtype=torch.int32))
    with pytest.raises(ValueError, match="null block"):
        bridge.decode_forward(**{**base, "start_pos": torch.tensor([40] + [-1] * 7, dtype=torch.int32)})
    with pytest.raises(ValueError, match="block ids"):
        bad = base["page_table"].clone()
        bad[0, 0] = kv.num_blocks
        bridge.decode_forward(**{**base, "page_table": bad})
    with pytest.raises(RuntimeError, match="speculative"):
        bridge.propose_draft_tokens(1, torch.zeros(8, 2, dtype=torch.int32), torch.zeros(8, 2), torch.ones(8))
    assert gen.decode_steps == []  # nothing reached the generator
    out = bridge.decode_forward(**base, enable_trace=False, reload_sampling_params=False, reset_sampling_state=False)
    assert tuple(out.shape) == (8, 1, 4096) and gen.decode_steps[-1] == ([0], False)
    # Narrower page tables are padded, wider ones must only carry null-block zeros past the context.
    wide = torch.nn.functional.pad(base["page_table"], (0, 7))
    bridge.decode_forward(**{**base, "page_table": wide, "start_pos": torch.tensor([4] + [-1] * 7, dtype=torch.int32)})
    assert bridge.read_decode_output(out) is out and bridge.read_decode_output(out, async_read=True) == (out, [])
    assert bridge.process_decode_output_host(out) is out
    with pytest.raises(NotImplementedError):
        bridge.process_decode_output_host(out, is_tokens=True)


def test_prefill_contract():
    bridge, gen, kv, d = _allocated_bridge(8)
    d.add("p0", list(range(1, 41)))
    d.add("p1", list(range(100, 107)))
    d._grow("p0", 40)
    d._grow("p1", 7)
    tokens = torch.full((2, 40), 4321, dtype=torch.int32)
    tokens[0, :40] = torch.tensor(d.seqs["p0"], dtype=torch.int32)
    tokens[1, :7] = torch.tensor(d.seqs["p1"], dtype=torch.int32)
    pt = torch.stack([d._row_table("p0"), d._row_table("p1")])
    base = dict(tokens=tokens, page_table=pt, kv_cache=kv, prompt_lens=np.array([40, 7]), empty_slots=[3, 5])
    with pytest.raises(NotImplementedError, match="prefix caching"):  # draft-1 launch: every row starts at 0
        bridge.prefill_forward(**base, start_pos=np.array([16, 0], dtype=np.int32))
    with pytest.raises(NotImplementedError, match="host"):
        bridge.prefill_forward(**base, sampling_params=SimpleNamespace())
    with pytest.raises(ValueError, match="distinct"):
        bridge.prefill_forward(**{**base, "empty_slots": [3, 3]})
    with pytest.raises(ValueError, match="null block"):
        bridge.prefill_forward(**{**base, "page_table": torch.zeros_like(pt)})
    with pytest.raises(ValueError, match="slot"):
        bridge.prefill_forward(**{**base, "empty_slots": [3, 8]})
    assert gen.prefills == []
    out = bridge.prefill_forward(**base, start_pos=np.zeros(2, dtype=np.int32), enable_trace=True)
    assert tuple(out.shape) == (2, 1, 4096) and out.dtype == torch.bfloat16
    lanes = bridge._lanes.slot_to_lane
    assert gen.prefills == [(lanes[3], 40, True), (lanes[5], 7, True)]  # stale tokens past prompt_lens were cut
    assert int(out[0, -1].argmax()) == next_token(d.seqs["p0"], 4096)
    assert int(out[1, -1].argmax()) == next_token(d.seqs["p1"], 4096)
    # With resumed prefill on, start_pos must stay inside [0, end).
    rb, rgen, rkv, rd = _allocated_bridge(8, chunked_prefill=True)
    rd.add("q", list(range(1, 41)))
    rd._grow("q", 40)
    with pytest.raises(ValueError, match="start_pos"):
        rb.prefill_forward(
            tokens=torch.tensor([rd.seqs["q"]], dtype=torch.int32), page_table=rd._row_table("q")[None], kv_cache=rkv,
            prompt_lens=np.array([40]), start_pos=np.array([40], dtype=np.int32), empty_slots=[0],
        )  # fmt: skip
    assert rgen.prefills == []


def test_stale_block_table_tails_are_zeroed():
    """vLLM's persistent block-table rows keep stale ids past a reused row's length (seen in the end-to-end test
    below). The generator must only ever see zeros there, or a bucket-padded prefill overwrites another request's KV.
    """
    for gen_cls in (FakeMotifGenerator, FakeDraft1Generator):
        bridge, gen = _bridge(num_slots=8, gen_cls=gen_cls)
        kv = bridge.allocate_kv_cache((640, 1, 32, 576), torch.bfloat16, 3)
        seq = list(range(100, 140))  # 40 tokens = 2 blocks of 32
        row = torch.zeros(kv.page_table_width, dtype=torch.int32)
        row[:2] = torch.tensor([5, 6], dtype=torch.int32)
        row[2:4] = torch.tensor([77, 78], dtype=torch.int32)  # stale: blocks another request owns now
        out = bridge.prefill_forward(
            tokens=torch.tensor([seq], dtype=torch.int32),
            page_table=row[None],
            kv_cache=kv,
            prompt_lens=np.array([40]),
            start_pos=np.zeros(1, dtype=np.int32),
            empty_slots=[0],
        )
        assert int(out[0, -1].argmax()) == next_token(seq, 4096)
        assert bool((gen.kv[:, [77, 78]] == -1).all())  # the bucket padding (positions 40..127) never reached them
        tokens = torch.zeros(8, 1, dtype=torch.int32)
        pos = torch.full((8,), -1, dtype=torch.int32)
        pt = torch.zeros(8, kv.page_table_width, dtype=torch.int32)
        tokens[0, 0], pos[0], pt[0] = 5, 40, row
        pt[3, 0] = 78  # junk on an inactive row
        bridge.decode_forward(tokens=tokens, start_pos=pos, page_table=pt, kv_cache=kv, reload_inputs=True)
        assert bool((gen.kv[:, [77, 78]] == -1).all())  # the fake also asserts every tail it saw was zero


def test_warmup_and_lifecycle_hooks():
    bridge, gen, kv, d = _allocated_bridge(32)
    with pytest.raises(RuntimeError, match="prefill warmup"):
        bridge.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=32, num_blocks=kv.page_table_width)
    with pytest.raises(ValueError, match="device"):
        bridge.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=True)
    # The plugin's two-phase warmup with trace_mode="decode_only" (model_runner.py:3727-3781).
    bridge.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=False)
    bridge.warmup_model_decode(
        kv_cache=kv, enable_trace=False, max_batch_size=32, num_blocks=kv.page_table_width, can_sample_on_device=False
    )
    bridge.warmup_model_decode(
        kv_cache=kv, enable_trace=True, max_batch_size=32, num_blocks=kv.page_table_width, can_sample_on_device=False
    )
    assert gen.warmups == [("prefill", False, None), ("decode", False, 32), ("decode", True, 32)]
    assert kv.page_table_width == api.cdiv(1024, 32)
    d.add("w0", [1, 2, 3])
    slot = d.prefill(["w0"])[0]
    d.decode()
    d.finish("w0")
    assert gen.released_lanes == [bridge._lanes.lane_of_slot(slot)]
    gen.stats = {"prefill_calls": 1, "verify_steps": 0}  # the real generator's counters (MotifGenerator.stats)
    with loguru_messages() as seen:
        bridge.release_persistent_capture()
    assert gen.traces_released == 1
    assert "Motif-3 generator: {'prefill_calls': 1, 'verify_steps': 0}" in seen, seen
    bridge.close()
    assert gen.traces_released == 2


def test_lane_map_unit():
    lm = gv.LaneMap(32)
    assert lm.slot_to_lane[:8] == (0, 8, 16, 24, 1, 9, 17, 25)
    assert sorted(lm.slot_to_lane) == list(range(32))
    assert [lm.group_of_slot(s) for s in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
    remap = list(range(32))
    remap[0], remap[5] = 5, 0
    assert lm.decode_lanes(32, remap)[:6] == [9, 8, 16, 24, 1, 0]
    assert lm.slot_to_lane[0] == 0  # not committed yet
    lm.commit(torch.tensor(remap, dtype=torch.int32))
    assert lm.slot_to_lane[0] == 9 and lm.slot_to_lane[5] == 0
    lm.commit(None)
    assert lm.slot_to_lane[0] == 9
    with pytest.raises(ValueError):
        lm.decode_lanes(33)
    with pytest.raises(ValueError):
        lm.commit(list(range(31)))
    with pytest.raises(ValueError):
        gv.LaneMap(33)
    small = gv.LaneMap(5)
    assert small.slot_to_lane == (0, 8, 16, 24, 1)


# ================================================================================================================
# 6. Resumed / chunked prefill and prefix caching (features design §2.3, §3.4, §3.7.2, §5.1 tests 1-3)
# ================================================================================================================
def test_resumed_rows_reach_one_prefill_forward_batch_call():
    """A step mixing a new row, a prefix hit and a chunk continuation is ONE generator call; each row carries vLLM's
    ``num_computed_tokens`` as ``PrefillRequest.start``, its tokens up to the chunk end and a zero-tailed page table;
    logits come back in input order."""
    bridge, gen, kv, d = _allocated_bridge(8, chunked_prefill=True, prefix_caching=True)
    rng = random.Random(3)
    d.add("a", [rng.randrange(100, 4000) for _ in range(300)])
    d.prefill(["a"])  # a cold 300-token prompt: blocks 0..9 (bs 32)
    d.add("hit", d.seqs["a"][:256] + [rng.randrange(100, 4000) for _ in range(70)])
    d.share_prefix("hit", "a", 8)  # vLLM's prefix cache: 8 full blocks
    d.add("long", [rng.randrange(100, 4000) for _ in range(500)])
    d.prefill(["long"], ends=[250])  # chunk 1 of a chunked prompt (unaligned end)
    d.add("new", [7] * 33)
    calls = len(gen.prefill_calls)
    d.prefill(["hit", "new", "long"], starts=[256, 0, 250], ends=[326, 33, 500])
    assert len(gen.prefill_calls) == calls + 1, "one plugin prefill step must be one prefill_forward_batch call"
    rows = {r["lane"]: r for r in gen.prefill_calls[-1]["rows"]}
    lanes = {rid: d.lane_of[rid] for rid in ("hit", "new", "long")}
    assert (rows[lanes["hit"]]["start"], rows[lanes["hit"]]["end"], rows[lanes["hit"]]["c0"]) == (256, 326, 256)
    assert rows[lanes["hit"]]["paths"] == ("sp1",)  # reads the 256 cached positions, computes 70
    assert (rows[lanes["long"]]["start"], rows[lanes["long"]]["c0"], rows[lanes["long"]]["w0"]) == (250, 192, 224)
    assert rows[lanes["new"]]["paths"] == ("sp0",)
    assert all(r["consistent"] for r in gen.prefill_rows)
    for rid in ("hit", "new", "long"):
        assert len(d.seqs[rid]) == {"hit": 327, "new": 34, "long": 501}[rid]


@pytest.mark.parametrize("kv_replicated", [False, True])
@pytest.mark.parametrize("hit_slot", [2, 0])
def test_cross_dp_row_prefix_hit_needs_kv_replicated_decode(kv_replicated, hit_slot):
    """Features design §3.4 / §5.1 test (1), review R7: request A decodes on lane 0 (DP row 0); request B (A's
    prompt + A's answer + a new turn) hits A's blocks, decode-written ones included. Without KV-R only row 0's chips
    hold the decode-written KV, and the replicated prefill (MoE RS(dp)) mixes every row: B's answer is wrong whichever
    lane B lands on, B on lane 16 (row 2) and B on lane 0 (row 0, after A finished) alike. With KV-R it is right.

    (Prefix caching itself refuses KV-R off, GeneratorSettings / check_scheduler_config; the hazard is staged with a
    chunked-prefill launch, whose resumed rows the same code serves.)"""
    features = dict(chunked_prefill=True, kv_replicated_decode=True if kv_replicated else None)
    bridge, gen, kv, d = _allocated_bridge(32, **features)
    assert bridge.settings.kv_replicated is kv_replicated
    rng = random.Random(17)
    d.add("A", [rng.randrange(100, 4000) for _ in range(100)])
    assert d.prefill(["A"], slots=[0]) == [0] and d.lane_of["A"] == 0
    for _ in range(60):
        d.decode()  # positions 100..159 decode-written (row 0 only without KV-R)
    assert len(d.seqs["A"]) == 161
    d.add("B", d.seqs["A"][:160] + [rng.randrange(100, 4000) for _ in range(30)])
    d.share_prefix("B", "A", 5)  # 5 full blocks of 32 = positions 0..159: a hit of 160 tokens
    if hit_slot == 0:
        d.finish("A")  # B reuses A's lane (and DP row); the shared blocks stay alive through B's reference
    results = d.prefill(["B"], starts=[160], slots=[hit_slot], check=False)
    rec = gen.prefill_rows[-1]
    assert (rec["start"], rec["c0"], rec["lane"]) == (160, 128, bridge._lanes.lane_of_slot(hit_slot))
    [(got, want)] = results
    assert rec["consistent"] is kv_replicated
    assert (got == want) is kv_replicated, f"KV-R {kv_replicated}: predicted {got}, ground truth {want}"


@pytest.mark.parametrize("writer_first", [True, False])
def test_same_step_prefix_hit_is_served_writer_first(writer_first):
    """Features design D7 / §3.7.2 / §5.1 test (2): vLLM caches a request's full blocks when it allocates them, so a
    request admitted later in the SAME step hits blocks another row of the same call is about to compute, and the
    plugin's row order need not put the writer first. One batch call, writer-first, is right; input order (the
    negative control) reads unwritten KV."""
    bridge, gen, kv, d = _allocated_bridge(8, chunked_prefill=True, prefix_caching=True)
    gen.writer_first = writer_first
    rng = random.Random(5)
    d.add("writer", [rng.randrange(100, 4000) for _ in range(300)])
    d.add("reader", d.seqs["writer"][:256] + [rng.randrange(100, 4000) for _ in range(40)])
    d._grow("writer", 300)
    d.share_prefix("reader", "writer", 8)
    results = d.prefill(["reader", "writer"], starts=[256, 0], check=False)  # the reader comes first in the step
    assert len(gen.prefill_calls) == 1 and len(gen.prefill_calls[0]["rows"]) == 2
    assert gen.prefill_calls[0]["order"] == ([1, 0] if writer_first else [0, 1])
    reader = next(r for r in gen.prefill_rows if r["start"] == 256)
    assert reader["reads_same_call"] is writer_first
    assert [got == want for got, want in results] == [writer_first, True]


def test_chunked_prompt_may_change_lane_between_chunks():
    """Features design §3.7.1 / §5.1 test (3): ``_alloc_prefill_state_slots`` re-picks a slot each prefill step, so a
    chunked prompt's chunks may run on different lanes (DP rows). Prefill writes every chip, the paged cache is the
    only cross-chunk state, and the request decodes on its LAST chunk's lane. Unaligned chunk ends recompute from the
    alignment floor and rewrite only from the write floor."""
    bridge, gen, kv, d = _allocated_bridge(8, chunked_prefill=True)
    rng = random.Random(23)
    d.add("c", [rng.randrange(100, 4000) for _ in range(700)])
    d.add("other", [9] * 20)
    d.prefill(["c"], ends=[300], slots=[0])  # lane 0 (row 0)
    d.prefill(["other", "c"], starts=[0, 300], ends=[20, 520], slots=[0, 5])  # c moves to lane 9 (row 1)
    d.prefill(["c"], starts=[520], ends=[700], slots=[3])  # and to lane 24 (row 3) for the final chunk
    c_rows = [r for r in gen.prefill_rows if r["end"] in (300, 520, 700)]
    assert [r["lane"] for r in c_rows] == [0, 9, 24]
    assert [(r["start"], r["c0"], r["w0"]) for r in c_rows] == [(0, 0, 0), (300, 256, 288), (520, 512, 512)]
    assert len(d.seqs["c"]) == 701 and d.order[0] == "c"
    for _ in range(5):
        d.decode()  # decodes on lane 24, its prefill slot 3
    assert gen.decode_steps[-1][0] == sorted([24, bridge._lanes.lane_of_slot(d.runner._req_state_slot["other"])])


def test_long_span_is_split_inside_one_call():
    """Features design D8: spans above the span cap (``MOTIF3_PREFILL_MAX_BUCKET``) are split inside ONE generator
    call into an sp0 head and sp1 continuations; the bridge sends the row as is."""
    bridge, gen, kv, d = _allocated_bridge(8, prefill_span_cap=256)
    assert gen.max_prefill_span == 256
    d.add("long", [random.Random(1).randrange(100, 4000) for _ in range(900)])
    d.prefill(["long"])  # start 0: the bridge needs no resumed-prefill launch for an internal split
    (rec,) = gen.prefill_rows
    assert rec["paths"] == ("sp0", "sp1", "sp1", "sp1") and rec["buckets"] == (256, 256, 256, 256)
    for _ in range(3):
        d.decode()


def test_serving_config_fail_fast():
    """Features design §1.5 on the captured scheduler config (raises on wrong-output configurations, warns on slow
    ones), as get_max_tokens_all_users runs it in init_device."""
    from vllm.config import set_current_vllm_config

    ok = gv.serving_config_of(_serving_vllm_config())
    assert gv.check_serving_config(ok, max_model_len=32768) == []  # 8128 / 8128 at A = 64: clean
    with pytest.raises(ValueError, match="KV-R"):
        gv.check_serving_config(ok, max_model_len=32768, environ={"MOTIF3_KV_REPLICATED_DECODE": "0"})
    with pytest.raises(ValueError, match="prefix-match-unit"):
        gv.check_serving_config(gv.serving_config_of(_serving_vllm_config(unit=128)), max_model_len=32768)
    with pytest.raises(ValueError, match="K = 1"):
        gv.check_serving_config(gv.serving_config_of(_serving_vllm_config(spec_k=2)), max_model_len=32768)
    warn = gv.check_serving_config(
        gv.serving_config_of(_serving_vllm_config(budget=2048, threshold=1000)), max_model_len=32768
    )
    assert any("long_prefill_token_threshold 1000" in w for w in warn) and any("2048 <" in w for w in warn)
    span = gv.check_serving_config(
        gv.serving_config_of(_serving_vllm_config(budget=32768, threshold=0)), max_model_len=32768
    )
    assert any("span cap" in w for w in span)  # TIS's unpinned max_context budget: rows are split internally
    single_shot = gv.serving_config_of(_serving_vllm_config(budget=32704, threshold=0))  # 32768 - A
    assert (
        gv.check_serving_config(single_shot, max_model_len=32768, environ={"MOTIF3_PREFILL_MAX_BUCKET": "32768"}) == []
    )
    off = gv.serving_config_of(_serving_vllm_config(chunked=False, prefix=False, spec_k=None))
    assert gv.check_serving_config(off, max_model_len=32768) == [] and off["spec_tokens"] == 0
    assert gv.serving_config_of(SimpleNamespace(model_config=None)) is None
    # The same checks run in init_device: prefix caching with KV-R forced off fails before the weights load.
    f = gv.MotifForCausalLM.get_max_tokens_all_users
    with set_current_vllm_config(_serving_vllm_config()), pytest.MonkeyPatch.context() as mp:
        mp.setenv("MOTIF3_KV_REPLICATED_DECODE", "0")
        with pytest.raises(ValueError, match="KV-R"):
            f(num_devices=32, max_model_len=32768, max_num_seqs=32)
    with set_current_vllm_config(_serving_vllm_config(budget=2048, threshold=2048)), loguru_messages() as seen:
        f(num_devices=32, max_model_len=32768, max_num_seqs=32)
    assert any("pin 8128" in m for m in seen), seen


# ================================================================================================================
# 7. MTP speculative decoding: spec_plan, verify / propose (features design §1.2, §2.3, §3.8-§3.10, §5.1 test 4)
# ================================================================================================================
def test_spec_plan_contract(monkeypatch, tmp_path):
    """``spec_plan``: K = 1 whatever was asked, packed verify (2 lanes per request), the MTP cache as
    ``extra_bytes_per_token`` PER CHIP (612 B bfp8 / 1152 B bf16), narrow decode, and PS-1 when the plugin has it."""
    from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

    for var in ("MOTIF3_WEIGHTS_DIR", "HF_MODEL", "TT_CACHE_PATH", "MOTIF3_TT_CACHE_PATH", "MOTIF3_KV_CACHE_DTYPE"):
        monkeypatch.delenv(var, raising=False)
    weights = _fake_checkpoint(tmp_path / "ckpt")
    vc = _spec_vllm_config(weights)
    for k in (1, 3, 7):
        plan = gv.MotifForCausalLM.spec_plan(vc, 32, k)
        assert isinstance(plan, SpecPlan), plan
        assert plan.effective_k == 1 and plan.block_width == 2 and plan.accepted_counts_range == (1, 2)
        assert plan.lanes_per_request == 2 and plan.extra_bytes_per_seq == 0 and plan.extra_bytes_per_token == 612
        assert plan.accept_modes == ("argmax_ids",) and plan.drafter_state == "internal"
        assert plan.supports_narrow_decode is True and plan.drafter_target_cache_requires == ()
        if gv.spec_plan_supports_speculable_rows():
            assert plan.verify_requires_speculable_rows is True
    monkeypatch.setenv("MOTIF3_KV_CACHE_DTYPE", "bf16")
    assert gv.MotifForCausalLM.spec_plan(vc, 32, 1).extra_bytes_per_token == 1152
    monkeypatch.setenv("MOTIF3_KV_CACHE_DTYPE", "fp4")  # a bad env value is a refusal, never a raise
    assert isinstance(gv.MotifForCausalLM.spec_plan(vc, 32, 1), SpecReject)
    monkeypatch.delenv("MOTIF3_KV_CACHE_DTYPE")
    # The unit: one more [N, 1, bs, 576] layer per chip, i.e. the 54- minus the 53-layer pool per pool token.
    for dtype in api.KV_CACHE_DTYPES:
        per_token = api.kv_cache_bytes_per_chip(4129, 64, 54, dtype) - api.kv_cache_bytes_per_chip(4129, 64, 53, dtype)
        assert per_token == 4129 * 64 * gv.mtp_extra_bytes_per_token(dtype)
    assert gv.mtp_extra_bytes_per_token("bfp8") == 576 * 1088 // 1024 and gv.mtp_extra_bytes_per_token("bf16") == 1152
    # Without the plugin's PS-1 field the plan omits it (and the bridge warns that sampled rows commit the argmax).
    monkeypatch.setattr(gv, "spec_plan_supports_speculable_rows", lambda: False)
    with loguru_messages() as seen:
        old = gv.MotifForCausalLM.spec_plan(vc, 32, 1)
    assert isinstance(old, SpecPlan) and getattr(old, "verify_requires_speculable_rows", False) is False
    assert any("PS-1" in m for m in seen)


def _fake_checkpoint(d: Path, *, mtp=True, shard=True, nextn=1):
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": "Motif", "num_nextn_predict_layers": nextn}))
    wm = {"model.embed_tokens.weight": "model-00001-of-00155.safetensors"}
    if mtp:
        wm.update(
            {f"{gv.MTP_WEIGHT_PREFIX}{n}": "model-00104-of-00155.safetensors" for n in ("input_proj", "x.weight")}
        )
    (d / gv.CHECKPOINT_INDEX).write_text(json.dumps({"weight_map": wm}))
    if shard:
        (d / "model-00104-of-00155.safetensors").write_bytes(b"")
    return d


def _spec_vllm_config(path, method=gv.SPEC_METHOD, nextn=1):
    hf = SimpleNamespace(_name_or_path=str(path), num_nextn_predict_layers=nextn)
    return SimpleNamespace(
        model_config=SimpleNamespace(model=str(path), hf_config=hf, hf_text_config=hf),
        speculative_config=SimpleNamespace(method=method, num_speculative_tokens=1),
    )


def test_spec_plan_refusals_never_raise(monkeypatch, tmp_path):
    """Every refusal is a SpecReject the plugin quotes; ``spec_plan`` never raises (SPEC_DECODE_CONTRACT §2)."""
    from vllm_tt_plugin.spec_decode import SpecReject

    for var in ("MOTIF3_WEIGHTS_DIR", "HF_MODEL", "TT_CACHE_PATH", "MOTIF3_TT_CACHE_PATH", "MOTIF3_KV_CACHE_DTYPE"):
        monkeypatch.delenv(var, raising=False)
    good = _spec_vllm_config(_fake_checkpoint(tmp_path / "good"))
    plan = gv.MotifForCausalLM.spec_plan

    def rejected(out, *, k=None, why=""):
        assert isinstance(out, SpecReject), out
        assert why in out.reason, out.reason
        if k is not None:
            assert out.supported_k == k
        return out

    rejected(plan(good, 32, 0), k=(1,), why="K = 1")
    rejected(plan(good, 33, 1), k=(), why="max_num_seqs=33")
    rejected(plan(_spec_vllm_config(tmp_path / "good", method="ngram"), 32, 1), k=(1,), why="custom_class")
    rejected(plan(_spec_vllm_config(_fake_checkpoint(tmp_path / "nomtp", mtp=False)), 32, 1), k=(), why="maps no")
    rejected(plan(_spec_vllm_config(_fake_checkpoint(tmp_path / "noshard", shard=False)), 32, 1), why="lacks")
    rejected(plan(_spec_vllm_config(tmp_path / "good", nextn=0), 32, 1), why="num_nextn_predict_layers=0")
    monkeypatch.setenv("MOTIF3_WEIGHTS_DIR", str(tmp_path / "typo"))  # resolve_weights_location raises: a refusal
    rejected(plan(good, 32, 1), why="MOTIF3_WEIGHTS_DIR")
    monkeypatch.delenv("MOTIF3_WEIGHTS_DIR")
    rejected(plan(object(), "many", 1), why="spec_plan failed")
    assert not isinstance(plan(None, 32, 1), SpecReject)  # no config at all: nothing to refuse on

    # Where the weights may come from: a converted TT cache part L53, the checkpoint, or a download.
    status = gv.mtp_weights_status
    nomtp = _spec_vllm_config(tmp_path / "nomtp")
    assert status(nomtp, {})[0] is False
    cache = tmp_path / "tt_cache"
    (cache / "motif3-2ed2ed5c-c1-x" / "mesh4x8" / "L53").mkdir(parents=True)
    assert status(nomtp, {"TT_CACHE_PATH": str(cache)})[0] is False  # no .complete marker: not converted
    (cache / "motif3-2ed2ed5c-c1-x" / "mesh4x8" / "L53" / ".complete").write_text("{}")
    ok, why = status(nomtp, {"MOTIF3_TT_CACHE_PATH": str(cache)})
    assert ok and "L53" in why
    hub = tmp_path / "hub"
    ok, why = status(None, {"HF_MODEL": "Motif-Technologies/Motif-3", "HF_HUB_CACHE": str(hub)})
    assert ok and "downloaded" in why
    assert status(good, {})[0] is True
    assert status(_spec_vllm_config(_motif_dir()), {})[0] in (True, False)  # the real snapshot: whatever is on disk


def test_verify_and_propose_bookkeeping():
    """Features design §3.8.3 / §3.9 item 4 / §5.1 test (4) on one request: the first decode after a prefill is an
    ordinary step whose MTP row yields the first draft (m0); a verify answers (a0, a1) per owner lane, and the next
    draft is m1 after an accepted draft (count 2), m0 after a rejected one (count 1); padding rows, stale rows and
    sampled rows are declined."""
    from vllm_tt_plugin.async_decode import _verify_output_tensor

    bridge, gen, kv, d = _allocated_bridge(4, spec_tokens=1)
    V = 4096
    d.add("g", [101, 102, 103, 104, 105])
    d.prefill(["g"])
    d.spec_decode()  # ordinary: nothing to verify after a prefill
    assert gen.spec_steps[-1]["want_logits"] is True and gen.spec_steps[-1]["drafted"] == []
    seq = d.seqs["g"]
    assert d.drafts["g"] == [mtp_token(seq, V)]  # m0 of the step = the MTP on (hn_n, a0 = the committed token)
    for _ in range(60):  # until the fake MTP has produced both an accepted and a rejected draft
        if d.stats["draft_count2"] and d.stats["draft_count1"] and d.stats["verify_steps"] >= 12:
            break
        draft = d.drafts.get("g")
        d.spec_decode()
        st = gen.spec_steps[-1]
        if draft:
            assert st["want_logits"] is False and st["drafted"] == st["owners"] == [d.lane_of["g"]]
            assert len(st["partners"]) == 1 and not st["overflow"]
            c = d.counts["g"]
            expected = mtp_token(d.seqs["g"], V)  # m1 (count 2) or m0 (count 1): the MTP after the committed tail
            assert d.drafts.get("g") == [expected], (c, d.drafts.get("g"), expected)
    assert d.stats["draft_count2"] > 0 and d.stats["draft_count1"] > 0, d.stats
    s = bridge.spec_stats
    assert s.verify_steps == d.stats["verify_steps"] and s.ordinary_steps == d.stats["ordinary_steps"]
    assert s.accepted == d.stats["draft_count2"] and s.rejected == d.stats["draft_count1"]

    # Direct propose calls: padding / stale / sampled rows decline; bad shapes raise.
    lane = d.lane_of["g"]
    st = bridge._retained[lane]
    n1 = st.pos0 + 1
    committed = torch.full((4, 2), PLACEHOLDER, dtype=torch.int32)
    committed[0, 0] = st.argmax[0]
    pos = torch.full((4, 2), -1, dtype=torch.int32)
    pos[0] = torch.tensor([n1, n1 + 1])
    out = bridge.propose_draft_tokens(1, committed, pos, torch.ones(4, dtype=torch.int32))
    assert out.num_valid.tolist() == [1, 0, 0, 0] and int(out.draft_token_ids[0, 0]) == st.mtp[0]
    assert out.draft_token_ids[1:].eq(PLACEHOLDER).all()
    committed[0, 0] = wrong_token(st.argmax[0], V)  # a sampled row committed something else
    before = bridge.spec_stats.declined_mismatch
    assert bridge.propose_draft_tokens(1, committed, pos, torch.ones(4, dtype=torch.int32)).num_valid.tolist()[0] == 0
    assert bridge.spec_stats.declined_mismatch == before + 1
    committed[0, 0] = st.argmax[0]
    stale = pos.clone()
    stale[0] += 5
    assert bridge.propose_draft_tokens(1, committed, stale, torch.ones(4, dtype=torch.int32)).num_valid[0] == 0
    with pytest.raises(ValueError, match="K = 2"):
        bridge.propose_draft_tokens(2, committed, pos, torch.ones(4, dtype=torch.int32))
    with pytest.raises(ValueError, match=r"\[B, 2\]"):
        bridge.propose_draft_tokens(1, committed[:, :1], pos[:, :1], torch.ones(4, dtype=torch.int32))
    with pytest.raises(ValueError, match="accepted_counts"):
        bridge.propose_draft_tokens(1, committed, pos, torch.full((4,), 3, dtype=torch.int32))
    # A verify answers argmax ids only: padding rows (0, -1), undrafted rows (a0, -1).
    d.add("h", [7, 8, 9])
    d.prefill(["h"])
    d.drafts["g"] = [mtp_token(d.seqs["g"], V)]
    kwargs = d._decode_common(["g", "h"])
    tokens = torch.zeros(4, 2, dtype=torch.int32)
    positions = torch.full((4, 2), -1, dtype=torch.int32)
    tokens[0] = torch.tensor([d.seqs["g"][-1], d.drafts["g"][0]])
    positions[0] = torch.tensor([len(d.seqs["g"]) - 1, len(d.seqs["g"])])
    tokens[1, 0], positions[1, 0] = d.seqs["h"][-1], len(d.seqs["h"]) - 1
    out = bridge.decode_forward(
        tokens=tokens, start_pos=positions, num_valid_drafts=torch.tensor([1, 0, 0, 0], dtype=torch.int32),
        accepted_counts=torch.ones(4, dtype=torch.int32), spec_mode="argmax_ids", **kwargs,
    )  # fmt: skip
    ids = _verify_output_tensor(out, "MotifForCausalLM", "argmax_ids")
    assert out.hidden is None and ids.dtype == torch.int32
    assert int(ids[0, 0]) == next_token(d.seqs["g"], V) and int(ids[1, 0]) == next_token(d.seqs["h"], V)
    assert int(ids[1, 1]) == PLACEHOLDER and ids[2:].tolist() == [[0, PLACEHOLDER]] * 2
    assert bridge._retained[d.lane_of["h"]].argmax == (int(ids[1, 0]),)


def test_verify_input_validation():
    bridge, gen, kv, d = _allocated_bridge(4, spec_tokens=1)
    d.add("v", [11, 12, 13, 14])
    d.prefill(["v"])
    n = len(d.seqs["v"]) - 1
    kwargs = d._decode_common(["v"])
    tokens = torch.tensor([[d.seqs["v"][-1], 200], [0, 0], [0, 0], [0, 0]], dtype=torch.int32)
    positions = torch.tensor([[n, n + 1], [-1, -1], [-1, -1], [-1, -1]], dtype=torch.int32)
    nv, cnt = torch.tensor([1, 0, 0, 0], dtype=torch.int32), torch.ones(4, dtype=torch.int32)
    base = dict(kwargs, tokens=tokens, start_pos=positions, num_valid_drafts=nv, accepted_counts=cnt)
    with pytest.raises(NotImplementedError, match="argmax_ids"):
        bridge.decode_forward(**base, spec_mode="logits")
    with pytest.raises(ValueError, match="missing"):
        bridge.decode_forward(**{**base, "accepted_counts": None}, spec_mode="argmax_ids")
    with pytest.raises(ValueError, match="without spec_mode"):
        bridge.decode_forward(tokens=tokens, start_pos=positions[:, 0], **kwargs)
    with pytest.raises(ValueError, match=r"\[B, 2\]"):
        bridge.decode_forward(**{**base, "tokens": tokens[:, :1]}, spec_mode="argmax_ids")
    with pytest.raises(ValueError, match="int32"):  # the plugin's own side-tensor check
        bridge.decode_forward(**{**base, "num_valid_drafts": nv.long()}, spec_mode="argmax_ids")
    with pytest.raises(ValueError, match=r"anchor position \+ 1"):
        bridge.decode_forward(
            **{**base, "start_pos": positions + torch.tensor([[0, 1]] + [[0, 0]] * 3)}, spec_mode="argmax_ids"
        )
    with pytest.raises(ValueError, match="padding row"):
        bridge.decode_forward(
            **{**base, "num_valid_drafts": torch.tensor([1, 1, 0, 0], dtype=torch.int32)}, spec_mode="argmax_ids"
        )
    with pytest.raises(ValueError, match="position -1"):
        undrafted = dict(base, num_valid_drafts=torch.zeros(4, dtype=torch.int32))
        bridge.decode_forward(**undrafted, spec_mode="argmax_ids")
    with pytest.raises(ValueError, match="draft token"):
        bridge.decode_forward(
            **{
                **base,
                "tokens": tokens.clone().index_put_(
                    (torch.tensor([0]), torch.tensor([1])), torch.tensor([5000], dtype=torch.int32)
                ),
            },
            spec_mode="argmax_ids",
        )
    assert gen.spec_steps == []  # nothing reached the generator
    # A failing generator commits nothing: no remap, no retained ids.
    gen.fail_next_decode = True
    retained = dict(bridge._retained)
    with pytest.raises(RuntimeError, match="injected"):
        bridge.decode_forward(**base, spec_mode="argmax_ids")
    assert bridge._retained == retained and bridge.spec_stats.verify_steps == 0
    bridge.decode_forward(**base, spec_mode="argmax_ids")
    assert bridge.spec_stats.verify_steps == 1 and gen.spec_steps[-1]["drafted"] == [d.lane_of["v"]]
    # A generator returning garbage ids for an active lane is refused before anything is committed.
    gen.decode_forward_spec = lambda batch, **kw: api.SpecDecodeResult(
        logits=None,
        argmax=torch.full((32, 2), 999999, dtype=torch.int32),
        mtp_argmax=torch.zeros(32, 2, dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="outside"):
        bridge.decode_forward(**base, spec_mode="argmax_ids")
    # release_request drops the lane's retained entry.
    slot = d.runner._req_state_slot["v"]
    assert d.lane_of["v"] in bridge._retained
    bridge.release_request(slot)
    assert d.lane_of["v"] not in bridge._retained and gen.released_lanes[-1] == d.lane_of["v"]


@pytest.mark.parametrize("kvr", [True, False])
def test_spec_decode_is_lossless_through_the_plugin_loop(kvr):
    """Random greedy traffic through the plugin's speculative loop (ordinary and verify steps, reorders, finishes,
    preemption + re-prefill, admissions): every committed token is the fake model's greedy token (lossless), drafts
    are accepted and rejected, and the generator's packed verify finds partner lanes (on other DP rows only with
    KV-R)."""
    features = dict(spec_tokens=1, prefix_caching=kvr, chunked_prefill=kvr)
    bridge, gen, kv, d = _allocated_bridge(16, **features)
    assert bridge.settings.kv_replicated is kvr and bridge.settings.kv_write_mode == (
        "all_split" if kvr else "row_split"
    )
    rng = random.Random(41 + kvr)
    next_id, preempted = 0, []

    def new_request():
        nonlocal next_id
        rid = f"s{next_id}"
        next_id += 1
        d.add(rid, [rng.randrange(100, 4000) for _ in range(rng.randrange(1, 60))])
        return rid

    d.prefill([new_request() for _ in range(5)])
    for step in range(120):
        running = len(d.order)
        if preempted and running < 16 and rng.random() < 0.2:
            d.prefill([preempted.pop(0)])
        elif running < 12 and rng.random() < 0.2:
            d.prefill([new_request() for _ in range(rng.randrange(1, 4))])
        if not d.order:
            d.prefill([new_request()])
        rows = list(d.order)
        if rng.random() < 0.3:
            rng.shuffle(rows)
        d.spec_decode(rows)
        if len(d.order) > 3 and rng.random() < 0.08:
            d.finish(rng.choice(d.order))
        if len(d.order) > 3 and rng.random() < 0.04:
            victim = rng.choice(d.order)
            d.preempt(victim)
            preempted.append(victim)
    # Ordinary steps are rare once drafts flow: one row with a draft makes the whole step a verify.
    assert d.stats["verify_steps"] > 30 and d.stats["ordinary_steps"] >= 1, d.stats
    assert d.stats["draft_count2"] > 20 and d.stats["draft_count1"] > 5, d.stats
    assert d.remaps > 5 and d.lane_checks > 200
    pairs = [(o, p) for st in gen.spec_steps for o, p in st["partners"].items()]
    assert pairs and all(st["overflow"] == [] for st in gen.spec_steps)
    cross = sum(o // 8 != p // 8 for o, p in pairs)
    assert (cross > 0) if kvr else (cross == 0), (cross, len(pairs))
    s = bridge.spec_stats
    assert s.verify_steps + s.ordinary_steps == len(gen.spec_steps) and s.accepted == d.stats["draft_count2"]


@pytest.mark.parametrize("kvr", [True, False])
def test_draft_budget_caps_drafts_to_idle_lanes(kvr):
    """A draft runs on an idle lane; past the idle lanes the generator would need a second trace replay (its overflow
    pass), so the bridge offers at most 32 - live drafts (KV-R: any idle lane) or 8 - live per DP row (without)."""
    features = dict(spec_tokens=1, prefix_caching=kvr)
    bridge, gen, kv, d = _allocated_bridge(32, **features)
    rng = random.Random(8)
    rids = [f"b{i}" for i in range(28)]
    for rid in rids:
        d.add(rid, [rng.randrange(100, 4000) for _ in range(rng.randrange(1, 30))])
    for i in range(0, 28, 7):
        d.prefill(rids[i : i + 7])
    for _ in range(20):
        d.spec_decode()
    live_groups = collections.Counter(d.lane_of[r] // 8 for r in rids)
    cap = 32 - 28 if kvr else sum(max(0, 8 - live_groups[g]) for g in range(4))
    assert max(d.offers) <= cap and max(d.offers) > 0, (d.offers, cap)
    assert all(st["overflow"] == [] for st in gen.spec_steps)
    assert bridge.spec_stats.declined_budget > 0
    # Fair: the rotating start lets every row draft now and then.
    drafted_lanes = {o for st in gen.spec_steps for o in st["drafted"]}
    assert len(drafted_lanes) > cap


@pytest.mark.parametrize("ps1", [True, False])
def test_ps1_holds_back_verify_while_a_sampled_request_is_live(ps1):
    """Features design §3.10 / lead decision PS-1: with ``verify_requires_speculable_rows`` the plugin sends no verify
    while a sampled request is in the batch (every step is the ordinary decode, sampled with its own parameters) and
    publishes no draft; without it a verify carries the sampled row, which commits the argmax (the hazard PS-1
    removes). Greedy rows stay lossless either way, and speculation resumes once the sampled request leaves."""
    bridge, gen, kv, d = _allocated_bridge(8, ps1=ps1, spec_tokens=1)
    d.add("t", [300, 301, 302], sampled=True)
    for i in range(3):
        d.add(f"g{i}", [400 + i] * (5 + i))
    d.prefill(["t", "g0", "g1", "g2"])
    for _ in range(15):
        d.spec_decode()
    if ps1:
        assert d.stats["verify_steps"] == 0 and d.stats["sampled_rows_verified"] == 0 and d.stats["published"] == 0
        assert all(st["want_logits"] for st in gen.spec_steps)
    else:
        assert d.stats["verify_steps"] > 0 and d.stats["sampled_rows_verified"] > 0
    if ps1:  # a held-back step commits the sample, never the retained argmax: the bridge offers nothing
        assert d.stats["sampled_rows_offered"] == 0
    else:  # a verify commits the argmax for the sampled row too; the plugin (_publish_draft) drops the offer
        assert d.stats["sampled_rows_offered"] > 0 and "t" not in d.drafts
    assert bridge.spec_stats.declined_mismatch > 0  # the sampled row's committed token is not the argmax
    d.finish("t")
    before = d.stats["verify_steps"]
    for _ in range(6):
        d.spec_decode()
    assert d.stats["verify_steps"] > before  # speculation resumes without the sampled request


# ================================================================================================================
# 8. Motif parser plugins (loaded exactly like --reasoning-parser-plugin / --tool-parser-plugin)
# ================================================================================================================
@pytest.fixture(scope="module")
def motif_tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(str(_motif_dir(require_tokenizer=True)), trust_remote_code=True)


@pytest.fixture(scope="module")
def motif_parsers():
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    from models.demos.motif3 import vllm_plugins as vp

    ReasoningParserManager.import_reasoning_parser(vp.REASONING_PARSER_PLUGIN)
    ToolParserManager.import_tool_parser(vp.TOOL_PARSER_PLUGIN)
    reasoning = ReasoningParserManager.get_reasoning_parser(vp.REASONING_PARSER_NAME)
    tools = ToolParserManager.get_tool_parser(vp.TOOL_PARSER_NAME)
    # import_*_parser swallows exceptions, so check the classes really come from our files.
    assert reasoning.__module__ == "motif_reasoning_parser" and reasoning.__name__ == "MotifReasoningParser"
    assert tools.__module__ == "motif_tool_parser" and tools.__name__ == "MotifToolParser"
    assert ToolParserManager.get_tool_parser("motif_hermes") is tools
    return SimpleNamespace(reasoning=reasoning, tools=tools)


def _chat_request(**kwargs):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], model="Motif-3", **kwargs)


def test_parser_plugin_files_and_cli_args():
    from models.demos.motif3 import vllm_plugins as vp

    for path in (vp.REASONING_PARSER_PLUGIN, vp.TOOL_PARSER_PLUGIN):
        text = Path(path).read_text()
        assert "SPDX-License-Identifier: Apache-2.0" in text and "github.com/MotifTechnologies/vllm" in text
        assert "import ttnn" not in text and "from models" not in text
    args = vp.vllm_cli_args()
    assert args[args.index("--reasoning-parser") + 1] == "motif"
    assert args[args.index("--tool-call-parser") + 1] == "motif" and "--enable-auto-tool-choice" in args


def test_vllm_serve_cli_accepts_the_motif_flags():
    """The documented launch flags (design 00 §5.1) and the parser-plugin flags parse in vLLM 0.26's ``vllm serve``
    parser, and the plugins load and validate exactly as ``api_server.setup_server`` does it."""
    from vllm.entrypoints.openai.api_server import validate_api_server_args
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from models.demos.motif3 import vllm_plugins as vp

    tt = dict(api.SERVING_TT_CONFIG)  # the documented "tt" config, incl. the mandatory l1_small_size
    assert tt == {"trace_mode": "decode_only", "trace_region_size": 268435456, "fabric_config": "FABRIC_2D_TORUS_XY",
                  "dispatch_core_axis": "col", "l1_small_size": 32768}  # fmt: skip
    argv = ["--model", str(_motif_dir()), "--trust-remote-code", "--max-num-seqs", "32", "--block-size", "64"]
    argv += ["--max-model-len", "32768", "--no-enable-prefix-caching", "--additional-config", json.dumps({"tt": tt})]
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(argv + vp.vllm_cli_args())
    validate_parsed_serve_args(args)
    assert (args.max_num_seqs, args.block_size, args.max_model_len) == (32, 64, 32768)
    assert args.trust_remote_code and args.enable_prefix_caching is False
    assert args.additional_config == {"tt": tt} == api.serving_additional_config()
    assert gv.check_tt_config(args.additional_config["tt"])["l1_small_size"] == gv.L1_SMALL_SIZE
    assert (args.reasoning_parser, args.reasoning_parser_plugin) == ("motif", vp.REASONING_PARSER_PLUGIN)
    assert (args.tool_call_parser, args.tool_parser_plugin) == ("motif", vp.TOOL_PARSER_PLUGIN)
    assert args.enable_auto_tool_choice
    ToolParserManager.import_tool_parser(args.tool_parser_plugin)
    ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)
    validate_api_server_args(args)
    assert args.reasoning_parser in ReasoningParserManager.list_registered()


def test_vllm_serve_cli_accepts_the_feature_flags():
    """Features design §1.1 (with the lead decision threshold = budget): ``gv.FEATURE_VLLM_ARGS`` parse in ``vllm
    serve`` and carry the values the bridge's checks call clean."""
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    tt = {**api.SERVING_TT_CONFIG, "decode_interleave_prefill_steps": 1, "decode_interleave_decode_steps": 1}
    argv = ["--model", str(_motif_dir()), "--trust-remote-code", "--max-num-seqs", "32", "--block-size", "64"]
    argv += ["--max-model-len", "32768", "--additional-config", json.dumps({"tt": tt}), *gv.FEATURE_VLLM_ARGS]
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(argv)
    validate_parsed_serve_args(args)
    assert args.enable_chunked_prefill is True and args.enable_prefix_caching is True and args.async_scheduling is False
    assert (args.max_num_batched_tokens, args.long_prefill_token_threshold) == (8128, 8128)
    spec = args.speculative_config if isinstance(args.speculative_config, dict) else json.loads(args.speculative_config)
    assert spec == {
        "method": "custom_class",
        "model": "vllm_tt_plugin.model_owned_drafter",
        "num_speculative_tokens": 1,
    }
    serving = gv.serving_config_of(_serving_vllm_config(budget=8128, threshold=8128))
    assert gv.check_serving_config(serving, max_model_len=32768) == []


def test_reasoning_parser_splits_think_blocks(motif_parsers, motif_tokenizer):
    tok, req = motif_tokenizer, _chat_request()
    parser = motif_parsers.reasoning(tok)
    assert (parser.start_token_id, parser.end_token_id) == (11, 12)
    # Motif's generation prompt already opened <think>, so outputs usually start inside the block.
    assert parser.extract_reasoning("Add the numbers.</think>2 + 2 = 4.", req) == ("Add the numbers.", "2 + 2 = 4.")
    assert parser.extract_reasoning("<think>plan</think>answer", req) == ("plan", "answer")
    ids = tok.encode("x</think>y", add_special_tokens=False)
    assert parser.is_reasoning_end(ids) and not parser.is_reasoning_end(
        tok.encode("<think>x", add_special_tokens=False)
    )
    off = motif_parsers.reasoning(tok, chat_template_kwargs={"enable_thinking": False})
    text = "No thinking here </think> literally."
    assert off.extract_reasoning(text, req) == (None, text)
    assert off.is_reasoning_end([]) is True  # never gates tool calls when thinking is off

    # Streaming, token by token, as the OpenAI server feeds it.
    out_text = "Let me check: 2 + 2.</think>The answer is 4."
    out_ids = tok.encode(out_text, add_special_tokens=False)
    assert 12 in out_ids
    stream = motif_parsers.reasoning(tok)
    reasoning, content, prev_text, prev_ids = "", "", "", []
    for tid in out_ids:
        cur_ids = prev_ids + [tid]
        cur_text = tok.decode(cur_ids)
        delta = stream.extract_reasoning_streaming(
            prev_text, cur_text, cur_text[len(prev_text) :], prev_ids, cur_ids, [tid]
        )
        if delta is not None:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
        prev_text, prev_ids = cur_text, cur_ids
    assert (reasoning, content) == ("Let me check: 2 + 2.", "The answer is 4.")


def test_tool_parser_extracts_and_repairs_tool_calls(motif_parsers, motif_tokenizer):
    parser = motif_parsers.tools(motif_tokenizer)
    req = _chat_request()
    good = 'Checking the weather.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Seoul"}}\n</tool_call>'
    info = parser.extract_tool_calls(good, req)
    assert info.tools_called and info.content == "Checking the weather.\n"
    assert [c.function.name for c in info.tool_calls] == ["get_weather"]
    assert json.loads(info.tool_calls[0].function.arguments) == {"city": "Seoul"}
    # Motif's repair ladder: a missing "[" around a string list, and two calls in one turn.
    bad = (
        '<tool_call>{"name": "search", "arguments": {"queries": "tt-metal", "vllm"}}</tool_call>\n'
        '<tool_call>{"name": "fetch", "arguments": {"urls": ["http://x"]}}}}</tool_call>'
    )
    info = parser.extract_tool_calls(bad, req)
    assert [c.function.name for c in info.tool_calls] == ["search", "fetch"]
    assert json.loads(info.tool_calls[0].function.arguments) == {"queries": ["tt-metal", "vllm"]}
    assert json.loads(info.tool_calls[1].function.arguments) == {"urls": ["http://x"]}
    plain = parser.extract_tool_calls("Just an answer.", req)
    assert not plain.tools_called and plain.content == "Just an answer."
    broken = '<tool_call>{"name": totally broken</tool_call>'
    assert parser.extract_tool_calls(broken, req).tools_called is False

    # Streaming in 5-character deltas: the name streams early, the arguments once the block closes.
    stream = motif_parsers.tools(motif_tokenizer)
    names, args, prev = [], "", ""
    for i in range(0, len(good), 5):
        cur = good[: i + 5]
        delta = stream.extract_tool_calls_streaming(prev, cur, cur[len(prev) :], [], [], [], req)
        for call in (delta.tool_calls if delta is not None else []) or []:
            fn = call.function if isinstance(call.function, dict) else call.function.model_dump()
            names += [fn["name"]] if fn.get("name") else []
            args += fn.get("arguments") or ""
        prev = cur
    assert names == ["get_weather"] and json.loads(args) == {"city": "Seoul"}


def test_reasoning_then_tool_call(motif_parsers, motif_tokenizer):
    """The server runs the reasoning parser first, then the tool parser on the content."""
    req = _chat_request()
    call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Busan"}}\n</tool_call>'
    output = "I need the weather.</think>\n" + call
    reasoning, content = motif_parsers.reasoning(motif_tokenizer).extract_reasoning(output, req)
    assert reasoning == "I need the weather."
    info = motif_parsers.tools(motif_tokenizer).extract_tool_calls(content, req)
    assert info.tools_called and json.loads(info.tool_calls[0].function.arguments) == {"city": "Busan"}


# ================================================================================================================
# 9. The real vLLM engine, end to end on the host
# ================================================================================================================
class _FakeMesh:
    """Stands in for the ``ttnn.MeshDevice`` the plugin would open; the runner itself never touches it."""

    shape = (4, 8)

    def get_num_devices(self):
        return 32

    def get_submeshes(self):
        return []


def _offline_engine_env(monkeypatch, tmp_path, caps):
    import vllm_tt_plugin.worker as tt_worker

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # EngineCore in this process (keeps the patches)
    monkeypatch.setenv("TT_MODEL_CLASS_OVERRIDES", gv.TT_MODEL_CLASS_OVERRIDES)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm_cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("MESH_DEVICE", "(4, 8)")
    monkeypatch.setenv("PYTHONPATH", str(METAL_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for var in ("EXTRA_MODELS_DIR", "MOTIF3_KV_POOL_TOKENS", "MOTIF3_KV_CACHE_DTYPE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(gv.MotifForCausalLM, "model_capabilities", caps)
    meshes, opened_with = [], []

    def open_fake_mesh(tt_config, trace_mode, *args, **kwargs):
        # what the real open_mesh_device would pass to ttnn.open_mesh_device
        opened_with.append(tt_worker.device_params_from_tt_config(tt_config, trace_mode))
        meshes.append(_FakeMesh())
        return meshes[-1]

    monkeypatch.setattr(tt_worker, "open_mesh_device", open_fake_mesh)
    monkeypatch.setattr(tt_worker, "close_mesh_device", lambda *args, **kwargs: None)
    return meshes, opened_with


def _shutdown(llm):
    try:
        llm.llm_engine.engine_core.shutdown()
    except RuntimeError as exc:
        # vLLM's cleanup_dist_env_and_memory() ends with torch.accelerator.empty_cache(), which raises on a host
        # without a torch accelerator (a TT host included); the model-side teardown already ran before it.
        if "accelerator" not in str(exc):
            raise


def test_vllm_offline_engine_end_to_end(fake_generator_class, monkeypatch, tmp_path):
    """``vllm.LLM`` -> TTPlatform -> TTWorker -> TTScheduler / TTModelRunner -> MotifForCausalLM -> fake generator,
    every feature switch off (draft 1).

    Everything is vLLM's and the plugin's real code except opening and closing the mesh (there is no device here).
    Twelve greedy requests of different lengths on ``max_num_seqs=8`` run in several waves, so state slots are reused
    and rows are reordered; every generated token must be the fake model's next token.
    """
    model_dir = _motif_dir(require_tokenizer=True)
    meshes, opened_with = _offline_engine_env(monkeypatch, tmp_path, CAPS_OFF)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_dir),
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=8,
        block_size=64,
        enable_prefix_caching=False,
        seed=0,
        additional_config={"tt": {"trace_mode": "decode_only", "l1_small_size": api.L1_SMALL_SIZE}},
    )
    try:
        # the plugin opens the mesh with the L1_SMALL region the model's CCL semaphores need
        assert opened_with and opened_with[0].get("l1_small_size") == api.L1_SMALL_SIZE, opened_with
        bridge = llm.llm_engine.model_executor.driver_worker.model_runner.model
        assert isinstance(bridge, gv.MotifForCausalLM)
        gen = bridge.generator
        assert isinstance(gen, fake_generator_class) and gen.mesh_device is meshes[0]
        assert not (gen.settings.chunked_prefill or gen.settings.prefix_caching or gen.settings.spec_decode)
        width = api.cdiv(4096, 64)
        assert gen.warmups == [("prefill", False, None), ("decode", False, width), ("decode", True, width)]
        assert gen.alloc_args == dict(num_blocks=gv.plugin_num_blocks(262144 + 32, 64, 8), block_size=64, num_layers=53)

        rng = random.Random(7)
        prompts = [
            {"prompt_token_ids": [1, 5, 3] + [rng.randrange(100, 200000) for _ in range(rng.randrange(1, 200))]}
            for _ in range(12)
        ]
        params = [SamplingParams(temperature=0.0, max_tokens=rng.randrange(1, 12), ignore_eos=True) for _ in prompts]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        for prompt, p, out in zip(prompts, params, outputs, strict=True):
            assert list(out.outputs[0].token_ids) == greedy_continuation(
                prompt["prompt_token_ids"], p.max_tokens, api.VOCAB_SIZE
            )
        assert len(gen.prefills) == len(prompts) and not any(trace for *_, trace in gen.prefills)
        assert all(r["start"] == 0 for r in gen.prefill_rows)
        assert gen.decode_steps and all(trace for _, trace in gen.decode_steps)  # trace_mode=decode_only
        assert max(len(lanes) for lanes, _ in gen.decode_steps) <= 8 and not gen.spec_steps

        # The plugin delivers a finished request's release_request with the NEXT step's scheduler output, so a
        # second batch flushes the releases of the first one (and reuses its freed slots and lanes).
        extra = {"prompt_token_ids": [1, 5, 3, 4321, 8765]}
        out = llm.generate([extra], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), use_tqdm=False)
        assert list(out[0].outputs[0].token_ids) == greedy_continuation(extra["prompt_token_ids"], 3, api.VOCAB_SIZE)
        assert len(gen.released_lanes) >= len(prompts)  # every finished request of the first batch released its lane
    finally:
        _shutdown(llm)
    assert gen.traces_released == 1  # release_persistent_capture at shutdown


def test_vllm_offline_engine_with_chunked_prefill_prefix_caching_and_mtp(fake_generator_class, monkeypatch, tmp_path):
    """Features design §5.1 test (5): the real engine with ``--enable-chunked-prefill --enable-prefix-caching
    --speculative-config <model-owned MTP drafter, K = 1>``, a small unaligned budget (300) and threshold (256) and a
    256-row span cap, so every mechanism runs: chunk continuations at unaligned starts, internal span splits,
    same-step prefix hits (shared prefixes submitted together), multi-turn hits on decode-written blocks (KV-R), full
    re-hits, verify steps with accepted and rejected drafts, PS-1 hold-back for a sampled request. Every greedy token
    is checked against the fake model."""
    model_dir = _motif_dir(require_tokenizer=True)
    meshes, _ = _offline_engine_env(monkeypatch, tmp_path, CAPS_ON)
    monkeypatch.setenv("MOTIF3_PREFILL_MAX_BUCKET", "256")
    monkeypatch.setenv("MOTIF3_TT_CACHE_PATH", str(_mtp_cache_marker(tmp_path)))  # spec_plan finds the MTP weights

    from vllm import LLM, SamplingParams

    tt = {"trace_mode": "decode_only", "l1_small_size": api.L1_SMALL_SIZE, "decode_interleave_prefill_steps": 1,
          "decode_interleave_decode_steps": 1}  # fmt: skip
    llm = LLM(
        model=str(model_dir),
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=8,
        block_size=64,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=300,
        long_prefill_token_threshold=256,
        speculative_config=dict(gv.SPECULATIVE_CONFIG),
        async_scheduling=False,
        seed=0,
        additional_config={"tt": tt},
    )
    V = api.VOCAB_SIZE
    try:
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        bridge = runner.model
        gen = bridge.generator
        s = gen.settings
        assert isinstance(gen, fake_generator_class) and gen.mesh_device is meshes[0]
        assert (s.chunked_prefill, s.prefix_caching, s.spec_tokens, s.kv_replicated) == (True, True, 1, True)
        assert (s.max_num_batched_tokens, s.long_prefill_token_threshold, s.prefill_span_cap) == (300, 256, 256)
        assert runner._num_speculative_tokens == 1 and runner._spec_supports_narrow_decode is True
        ps1 = gv.spec_plan_supports_speculable_rows()
        assert getattr(runner, "_spec_verify_requires_speculable_rows", False) is ps1
        rng = random.Random(11)

        def rand(n):
            return [rng.randrange(100, 200000) for _ in range(n)]

        def check(prompts, params, outs):
            for prompt, p, out in zip(prompts, params, outs, strict=True):
                if p.temperature == 0:
                    assert list(out.outputs[0].token_ids) == greedy_continuation(prompt, p.max_tokens, V)
                else:
                    assert len(out.outputs[0].token_ids) == p.max_tokens

        def run(prompts, params):
            outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
            check(prompts, params, outs)
            return [list(o.outputs[0].token_ids) for o in outs]

        head = [1, 5, 3]
        shared = head + rand(700)
        # Phase 1: shared prefixes submitted together (same-step hits), one long prompt (chunks + splits), short ones.
        p1 = [shared + rand(rng.randrange(20, 90)) for _ in range(5)] + [head + rand(1500)]
        p1 += [head + rand(rng.randrange(5, 60)) for _ in range(4)]
        k1 = [80, 80, 80] + [rng.randrange(8, 30) for _ in range(len(p1) - 3)]
        out1 = run(p1, [SamplingParams(temperature=0.0, max_tokens=k, ignore_eos=True) for k in k1])
        rows1 = list(gen.prefill_rows)
        assert any(r["start"] > 0 and r["start"] % 64 for r in rows1), "no unaligned chunk continuation"
        assert any(len(r["paths"]) > 1 for r in rows1), "no internal span split"
        assert any(r["reads_same_call"] for r in rows1), "no same-step prefix hit"
        stats = bridge.spec_stats
        assert stats.verify_steps > 0 and stats.accepted > 0 and stats.rejected > 0, stats

        # Phase 2: multi-turn - the prompt carries the previous answer, so the hit covers decode-written blocks.
        n_rows = len(gen.prefill_rows)
        p2 = [p1[i] + out1[i] + rand(40) for i in range(3)]
        run(p2, [SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True) for _ in p2])
        rows2 = gen.prefill_rows[n_rows:]
        for i, prompt in enumerate(p2):
            mine = [r for r in rows2 if r["tokens"] == prompt[: r["end"]]]
            assert mine and min(r["start"] for r in mine) > len(p1[i]), (i, [(r["start"], r["end"]) for r in mine])
        assert all(r["consistent"] for r in gen.prefill_rows), "a prefill read KV that differs between DP rows"

        # Phase 3: a sampled request lives through every decode step: with PS-1 nothing verifies.
        n_spec = len(gen.spec_steps)
        p3 = [head + rand(30)] + [head + rand(rng.randrange(10, 50)) for _ in range(3)]
        k3 = [SamplingParams(temperature=1.0, max_tokens=40, ignore_eos=True, seed=3)]
        k3 += [SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=True) for _ in p3[1:]]
        run(p3, k3)
        steps3 = gen.spec_steps[n_spec:]
        assert steps3
        if ps1:
            assert all(not st["drafted"] for st in steps3), "a verify step carried the sampled request (PS-1)"

        # Phase 4: identical prompts again - full-prefix hits, same greedy answers.
        n_rows = len(gen.prefill_rows)
        again = run(
            [p1[3], p1[4]], [SamplingParams(temperature=0.0, max_tokens=k1[i], ignore_eos=True) for i in (3, 4)]
        )
        assert again == [out1[3], out1[4]]
        rows4 = gen.prefill_rows[n_rows:]
        assert rows4 and all(r["start"] >= 640 for r in rows4), [(r["start"], r["end"]) for r in rows4]
        assert (
            all(trace for _, trace in gen.decode_steps)
            and gen.spec_steps
            and not any(st["overflow"] for st in gen.spec_steps)
        )
        rows = gen.prefill_rows
        summary = dict(
            prefill_calls=len(gen.prefill_calls),
            prefill_rows=len(rows),
            resumed_rows=sum(r["start"] > 0 for r in rows),
            unaligned_starts=sum(r["start"] % 64 > 0 for r in rows),
            recomputed_rows=sum(r["start"] - r["c0"] > 0 and r["c0"] > 0 for r in rows),
            multi_chunk_rows=sum(len(r["paths"]) > 1 for r in rows),
            same_step_hits=sum(r["reads_same_call"] for r in rows),
            reordered_calls=sum(c["order"] != sorted(c["order"]) for c in gen.prefill_calls),
            spec=bridge.spec_stats.as_dict(),
            cross_row_partners=sum(o // 8 != p // 8 for st in gen.spec_steps for o, p in st["partners"].items()),
        )
        print("MOTIF3_E2E_FEATURES", json.dumps(summary))
    finally:
        _shutdown(llm)
    assert gen.traces_released == 1
