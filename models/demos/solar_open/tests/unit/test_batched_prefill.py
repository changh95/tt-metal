# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed / batched multi-user prefill (phase 3a(1), scratchpad design_packed_prefill.md).

Two layers of tests:

* ``test_host_*`` -- no device: the policy knobs (``BatchedPrefillOptions.from_env``), the pure microbatch plan
  (``plan_batched_prefill``: buckets, users per pass, trailing single user, cached prefixes, ordering), the driver
  (``prefill_forward_text_batched`` against a fake Generator: re-slotting to ``0..m-1`` with the users' own page-table
  rows, the ``disable_batched_prefill`` flag lifted only for packed passes and restored, traces forced off, output
  scattered back to request order, per-user sequential remainder, delegation when nothing is packable) and the
  trace-safety guard of ``Model.prepare_prefill_inputs_trace``.

      SOLAR_OPEN_NUM_DEVICES=8 pytest models/demos/solar_open/tests/unit/test_batched_prefill.py -k host

* ``test_batched_vs_sequential_prefill`` -- 1x8 mesh, real weights, one device process: the same B users prefilled
  (a) sequentially on today's per-user path and (b) packed through the tt_transformers batched path, per-user
  last-token logits compared (full-vocab PCC, KL, top-1 on decisive users) and the paged KV checked by K teacher-forced
  decode steps over each prefill's cache (a wrong per-user page mapping collapses the affected user's decode logits,
  not just adds noise). Cases: B in {2, 8, 32} x 128 tokens (T = 256 dense-bmm MoE, 1024 / 4096 expert-sorted MoE),
  B = 4 x 1024 (T = 4096, the 1K bucket opted in through max_seq_len), and 8 x 128 in TWO passes of 4 (users 4..7
  re-slotted to device rows 0..3: exercises the page-table-row mapping under re-slotting).

      pytest models/demos/solar_open/tests/unit/test_batched_prefill.py -k "1x8 and b8_s128 and not x2" \\
          -x -p no:cacheprovider      # one case; timeout 1800 per the device rules

Not bit-identical by construction, and not at the bfp8 floor either (measured 2026-09-08, real layer 0,
tests/test_layer0_batched_prefill.py): every row-wise matmul of a packed pass (qkv, o_proj, router, lm_head, the MoE
paths) gets ttnn's auto program config for T = B x S rows instead of S rows, so its bf16 partial sums accumulate in
another order (attention rows PCC 0.9997 packed vs per-user, MoE 0.9997, whole layer 0.9997, min row 0.996; both forms
equally close to HF: 0.99959 / 0.99970 attention, 0.99997 / 0.99996 MoE) and the router's near-tie flips make the
difference discrete; over 48 layers the full-model first-token logits of a packed pass land at PCC 0.973-0.988 / KL
0.01-0.14 against the sequential pass (b2 / b8 / b32 x 128, 4 x 1024), i.e. the batch-1-vs-batch-32-decode class of
difference (the first decode step over the packed KV lands at PCC 0.95-0.98 and converges over the following steps),
while a wrong user / RoPE / page mapping collapses a user to PCC < 0.5. The floors below are therefore no-garbage
consistency floors (PCC >= 0.9 per user, per-user KL <= 1.0 and mean KL <= 0.25, top-1 equal for users whose sequential
margin is >= 3 logits, distinct users); the ground-truth gate of the packed path is
tests/accuracy/test_teacher_forced.py -k b32 with SOLAR_OPEN_BATCHED_PREFILL=1 (the whole 32 x 128 batch as ONE pass
against the bf16 HF reference; phase-3a: 0.9297 / 0.9690 / 0.9211 / 0.97969 / 0.99064 / 0.0312 sequential -- see
the README "Recorded baselines" phase-3a rows for the packed values).

Phase 3c (slot independence, design_traced_prefill.md G1): ``Model.ttnn_prefill_forward`` marks a packed pass
(``experts/prefill.py: packed_prefill_pass``) and the expert-sorted MoE then plans its hot / cold sets ONCE per
4096-token chunk (``_sorted_moe_chunk_plan``, arithmetic in ``experts/sorted_plan.py``, host tests in
tests/unit/test_sorted_moe_chunk_plan.py) instead of per 1024-token split, so a user's numerics no longer depend on
the split its slot falls into (the phase-3a failure of tests/test_multi_user_consistency.py with the flag). The
optional GATHER head (``BatchedPrefillOptions.head == "gather"``, env SOLAR_OPEN_BATCHED_PREFILL_HEAD) runs the pass
through Solar's own ``Model.packed_prefill_pass``: norm + lm_head on the 32 gathered last-token rows and ONE readback
per pass (design_packed_prefill.md 2.6 item 1); the device case ``b32_s128_gather`` compares it against the sequential
arm with the same floors as the "full" head. The single-user prefill path (batch_size 1) is untouched by both.
"""

import json
import math
import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.solar_open.tt import model_config as mc
from models.demos.solar_open.tt.model import (
    Model,
    batched_prefill_flag,
    prefill_forward_text_batched,
    summarize_batched_prefill_log,
)
from models.tt_transformers.tt.common import get_padded_prefill_len, preprocess_inputs_prefill

PROMPTS_128 = "models/demos/solar_open/demo/sample_prompts/input_data_questions_ko_en_prefill_128.json"
LONG_CONTEXT_FILE = "models/tt_transformers/demo/sample_prompts/input_data_long_4k.json"
# No-garbage consistency floors (see the module docstring): the packed pass is a different-but-equivalent numerical path.
# Measured 2026-09-08 (real weights): first-token logits per-user PCC min 0.9729 (32 x 128), KL mean 0.01-0.14 / max
# 0.52; the FIRST teacher-forced decode step over the packed KV vs the sequential KV per-user PCC min 0.948 (32 x 128),
# 0.953 (4 x 1024), KL mean 0.07-0.13, converging over the next steps (2 x 128: 0.974 -> 0.987 -> 0.994 -> 0.996); one
# first-token top-1 flip at a 2.25-logit sequential margin (8 x 128). A wrong user / RoPE / page mapping collapses a
# user to PCC < 0.5 and flips nearly every large-margin top-1.
PCC_MIN = 0.9  # per-user full-vocab PCC, sequential vs packed, prefill logits and every compared decode step
KL_MAX = 1.0  # per-user KL(sequential || packed) (`<|think|>` / `<|content|>` near ties dominate the first token)
KL_MEAN_MAX = 0.25  # mean over the users
DECISIVE_MARGIN = 3.0  # sequential top-1 margin (logits) above which a top-1 flip is not a near tie
DECODE_STEPS = 4


# ---------------------------------------------------------------------------------------------------------------------
# Host tests
# ---------------------------------------------------------------------------------------------------------------------

BATCHED_ENV = (
    "SOLAR_OPEN_BATCHED_PREFILL",
    "SOLAR_OPEN_BATCHED_PREFILL_TOKENS",
    "SOLAR_OPEN_BATCHED_PREFILL_MAX_SEQ_LEN",
    "SOLAR_OPEN_BATCHED_PREFILL_HEAD",
)


@pytest.fixture
def clean_batched_env(monkeypatch):
    for var in BATCHED_ENV:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_host_options_defaults_and_env(clean_batched_env, expect_error):
    opts = mc.BatchedPrefillOptions.from_env()
    assert opts == mc.BatchedPrefillOptions(enabled=False, tokens_per_pass=1024, max_seq_len=128)
    assert opts.users_per_pass(128) == 8 and opts.users_per_pass(1024) == 0
    assert "OFF" in opts.describe()

    clean_batched_env.setenv("SOLAR_OPEN_BATCHED_PREFILL", "1")
    clean_batched_env.setenv("SOLAR_OPEN_BATCHED_PREFILL_TOKENS", "4096")
    clean_batched_env.setenv("SOLAR_OPEN_BATCHED_PREFILL_MAX_SEQ_LEN", "1024")
    opts = mc.BatchedPrefillOptions.from_env()
    assert opts == mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=4096, max_seq_len=1024)
    assert opts.users_per_pass(128) == 32 and opts.users_per_pass(1024) == 4 and opts.users_per_pass(2048) == 0

    for value in ("0", "false", "off", ""):
        clean_batched_env.setenv("SOLAR_OPEN_BATCHED_PREFILL", value)
        assert not mc.BatchedPrefillOptions.from_env().enabled
    # The v1 token budget is clamped to the lm_head-on-all-rows safe range, never rejected.
    clean_batched_env.setenv("SOLAR_OPEN_BATCHED_PREFILL_TOKENS", "65536")
    assert mc.BatchedPrefillOptions.from_env().tokens_per_pass == mc.BATCHED_PREFILL_MAX_TOKENS
    clean_batched_env.setenv("SOLAR_OPEN_BATCHED_PREFILL_TOKENS", "32")
    assert mc.BatchedPrefillOptions.from_env().tokens_per_pass == mc.BATCHED_PREFILL_MIN_TOKENS
    # phase 3c: the head of a packed pass ("full" = tt_transformers batched path, "gather" = Model.packed_prefill_pass)
    assert mc.BatchedPrefillOptions.from_env().head == "full" and "head=full" in mc.BatchedPrefillOptions().describe()
    clean_batched_env.setenv("SOLAR_OPEN_BATCHED_PREFILL_HEAD", "Gather")
    assert mc.BatchedPrefillOptions.from_env().head == "gather"
    clean_batched_env.setenv("SOLAR_OPEN_BATCHED_PREFILL_HEAD", "slice")
    with expect_error(ValueError, "SOLAR_OPEN_BATCHED_PREFILL_HEAD='slice' is not one of"):
        mc.BatchedPrefillOptions.from_env()
    clean_batched_env.delenv("SOLAR_OPEN_BATCHED_PREFILL_HEAD", raising=False)
    with expect_error(ValueError, "head must be one of"):
        mc.BatchedPrefillOptions(head="slice")
    clean_batched_env.setenv("SOLAR_OPEN_BATCHED_PREFILL_TOKENS", "abc")
    with expect_error(ValueError, "SOLAR_OPEN_BATCHED_PREFILL_TOKENS='abc' is not an integer"):
        mc.BatchedPrefillOptions.from_env()
    with expect_error(ValueError, "tokens_per_pass must be a multiple of 32"):
        mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=1000)
    with expect_error(ValueError, "max_seq_len must be a positive multiple of 32"):
        mc.BatchedPrefillOptions(enabled=True, max_seq_len=100)


def test_host_padded_prefill_batch():
    assert [mc.padded_prefill_batch(n) for n in (1, 2, 3, 4, 5, 8, 9, 16, 17, 32)] == [1, 2, 4, 4, 8, 8, 16, 16, 32, 32]
    p = mc.BatchedPrefillPass(users=(0, 1, 2), seq_len=128)
    assert p.padded_batch == 4 and p.num_tokens == 512


def test_host_plan_buckets_and_microbatches():
    on = mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=1024, max_seq_len=128)
    # The batch-32 demo: 32 x (78..84 -> 128) tokens -> four passes of 8 users, in request order, nothing sequential.
    plan = mc.plan_batched_prefill([80] * 32, on)
    assert [p.users for p in plan.passes] == [tuple(range(s, s + 8)) for s in range(0, 32, 8)]
    assert all(p.seq_len == 128 and p.padded_batch == 8 for p in plan.passes)
    assert plan.sequential == () and plan.packed_users == tuple(range(32))
    # One pass of 32 with a 4096-token budget.
    plan = mc.plan_batched_prefill(torch.tensor([80] * 32), mc.BatchedPrefillOptions(True, 4096, 128))
    assert [(len(p.users), p.padded_batch) for p in plan.passes] == [(32, 32)]
    # Remainder pass: 5 users at 4 per pass -> one pass of 4 and a trailing single user on the per-user path.
    plan = mc.plan_batched_prefill([80] * 5, mc.BatchedPrefillOptions(True, 512, 128))
    assert [p.users for p in plan.passes] == [(0, 1, 2, 3)] and plan.sequential == (4,)
    # 6 users at 4 per pass -> a pass of 4 and a pass of 2 (device batch 2).
    plan = mc.plan_batched_prefill([80] * 6, mc.BatchedPrefillOptions(True, 512, 128))
    assert [(p.users, p.padded_batch) for p in plan.passes] == [((0, 1, 2, 3), 4), ((4, 5), 2)]


def test_host_plan_mixed_buckets_and_fallbacks(expect_error):
    opts = mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=4096, max_seq_len=1024)
    lens = [80, 900, 80, 900, 3000, 80, 80, 80]  # buckets 128, 1024, 128, 1024, 4096, 128, 128, 128
    plan = mc.plan_batched_prefill(lens, opts)
    # Shortest bucket first; the 4K user is above max_seq_len -> sequential.
    assert [(p.users, p.seq_len, p.padded_batch) for p in plan.passes] == [((0, 2, 5, 6, 7), 128, 8), ((1, 3), 1024, 2)]
    assert plan.sequential == (4,)
    # A bucket with a single packable user is not a pass (the Generator's batched path needs B > 1).
    plan = mc.plan_batched_prefill([80, 900, 80], opts)
    assert [p.users for p in plan.passes] == [(0, 2)] and plan.sequential == (1,)
    # Users with a cached prefix are never packed (the batched path feeds full prompts).
    plan = mc.plan_batched_prefill([80, 80, 80], opts, num_cached=[0, 16, 0])
    assert [p.users for p in plan.passes] == [(0, 2)] and plan.sequential == (1,)
    with expect_error(ValueError, "num_cached has 1 entries for 2 users"):
        mc.plan_batched_prefill([80, 80], opts, num_cached=[0])
    # Disabled -> everything sequential, in request order.
    plan = mc.plan_batched_prefill(lens, mc.BatchedPrefillOptions(enabled=False))
    assert plan.passes == () and plan.sequential == tuple(range(len(lens)))
    # max_seq_len 128 keeps the 1K bucket sequential even with a 4K budget (the design's default: flat per-token cost).
    plan = mc.plan_batched_prefill([900] * 4, mc.BatchedPrefillOptions(True, 4096, 128))
    assert plan.passes == () and plan.sequential == (0, 1, 2, 3)


class _FakeModel:
    """Records every ``packed_prefill_pass`` (gather head) call of the driver; same fake logits as the generator."""

    users_row_sharded = False
    _supports_on_device_sampling = False
    sampling = None

    def __init__(self, vocab):
        self.vocab_size = vocab
        self.pass_calls = []

    def packed_prefill_pass(self, tokens, prompt_lens, page_table, kv_cache, seq_len, padded_batch):
        self.pass_calls.append(
            {
                "batch": int(tokens.shape[0]),
                "lens": list(prompt_lens),
                "page_rows": page_table[:, 0].tolist(),
                "kv_cache": kv_cache,
                "seq_len": int(seq_len),
                "padded_batch": int(padded_batch),
            }
        )
        out = torch.zeros(tokens.shape[0], 1, self.vocab_size)
        out[:, 0, 0] = tokens[:, 0].float()
        return out


class _FakeGenerator:
    """Records every prefill_forward_text call; returns logits whose column 0 carries the user's first token id."""

    VOCAB = 8

    def __init__(self, options, trace_shapes=()):
        args = SimpleNamespace(disable_batched_prefill=not options.enabled, vocab_size=self.VOCAB)
        args.batched_prefill = options
        args.packed_prefill_trace_shapes = set(trace_shapes)
        args.can_enable_batched_prefill_trace = lambda b, s: (int(b), int(s)) in args.packed_prefill_trace_shapes
        self.model_args = [args]
        self.model = [_FakeModel(self.VOCAB)]
        self.data_parallel = 1
        self.calls = []
        self.warmups = []
        self.mode = None
        self._slots_prefilled_since_decode = {7}

    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, greedy_only=False):
        self.warmups.append({"trace": enable_trace, "sample": can_sample_on_device})

    def prefill_forward_text(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        start_pos=None,
        sampling_params=None,
        warmup_prefill=True,
        **kwargs,
    ):
        self.calls.append(
            {
                "batch": int(tokens.shape[0]),
                "flag_disable_batched": self.model_args[0].disable_batched_prefill,
                "slots": list(empty_slots),
                "trace": enable_trace,
                "page_rows": page_table[:, 0].tolist() if page_table is not None else None,
                "lens": list(prompt_lens),
                "start_pos": start_pos,
                "warmup": warmup_prefill,
                "sampling": sampling_params,
            }
        )
        self._slots_prefilled_since_decode.update(int(s) for s in empty_slots)
        out = torch.zeros(tokens.shape[0], 1, self.VOCAB)
        out[:, 0, 0] = tokens[:, 0].float()
        return out


def _fake_inputs(num_users):
    tokens = torch.arange(100, 100 + num_users).reshape(num_users, 1).repeat(1, 128)
    page_table = torch.arange(num_users).reshape(num_users, 1).repeat(1, 4) * 10
    return tokens, page_table


def test_host_driver_packs_reslots_and_scatters():
    options = mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=1024, max_seq_len=1024)
    gen = _FakeGenerator(options)
    tokens, page_table = _fake_inputs(8)
    lens = [80, 900, 80, 900, 3000, 80, 80, 80]
    out = prefill_forward_text_batched(
        gen, tokens, page_table=page_table, kv_cache="kv", prompt_lens=lens, enable_trace=True, warmup_prefill=True
    )
    # Output rows come back in request order whatever the pass order was.
    assert out.shape == (8, 1, _FakeGenerator.VOCAB)
    assert out[:, 0, 0].tolist() == [float(100 + u) for u in range(8)]
    # One packed pass of the 128 bucket (users 0, 2, 5, 6, 7 re-slotted to 0..4 with THEIR page-table rows, device
    # batch 8), traces forced off, the flag lifted; the 1K users (one per 1024-token pass -> not a pass) and the 4K
    # user (above max_seq_len) one per-user call each with the caller's trace setting, their real slot and the flag
    # back to disabled.
    packed = gen.calls[0]
    assert packed["batch"] == 5 and packed["slots"] == [0, 1, 2, 3, 4] and packed["page_rows"] == [0, 20, 50, 60, 70]
    assert packed["lens"] == [80] * 5 and packed["trace"] is False and packed["flag_disable_batched"] is False
    assert packed["start_pos"] is None and packed["sampling"] is None and packed["warmup"] is True
    rest = gen.calls[1:]
    assert [c["slots"] for c in rest] == [[1], [3], [4]]
    assert [c["page_rows"] for c in rest] == [[10], [30], [40]]
    assert all(c["batch"] == 1 and c["trace"] is True and c["flag_disable_batched"] is True for c in rest)
    assert all(c["warmup"] is False for c in rest)
    # The flag is restored to the configured value and the decode-side slot bookkeeping holds the REAL slots.
    assert gen.model_args[0].disable_batched_prefill is False
    assert gen._slots_prefilled_since_decode == {7} | set(range(8))
    # Pass log: one packed record + three per-user records, TTFT monotone in pass order.
    log = gen.batched_prefill_pass_log
    assert [r.packed for r in log] == [True, False, False, False]
    assert log[0].users == (0, 2, 5, 6, 7) and log[0].slots == (0, 2, 5, 6, 7) and log[0].padded_batch == 8
    assert log[0].seq_len == 128 and log[0].traced is False
    assert all(a.ttft_s <= b.ttft_s for a, b in zip(log, log[1:]))
    summary = summarize_batched_prefill_log(log, 8)
    assert summary["first"] == log[0].ttft_s and summary["last"] == log[-1].ttft_s
    assert summary["per_user"][0] == log[0].ttft_s and summary["per_user"][4] == log[3].ttft_s


def test_host_driver_microbatches_and_real_slots():
    options = mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=1024, max_seq_len=128)
    gen = _FakeGenerator(options)
    tokens, page_table = _fake_inputs(20)
    slots = list(range(12, 32))  # users sit in slots 12..31: the passes must NOT span to slot 31
    out = prefill_forward_text_batched(
        gen, tokens, page_table=page_table, kv_cache=None, prompt_lens=[100] * 20, empty_slots=slots, enable_trace=True
    )
    assert out[:, 0, 0].tolist() == [float(100 + u) for u in range(20)]
    assert [c["batch"] for c in gen.calls] == [8, 8, 4]
    assert all(c["slots"] == list(range(c["batch"])) for c in gen.calls)
    assert gen.calls[2]["page_rows"] == [160, 170, 180, 190]
    assert [r.padded_batch for r in gen.batched_prefill_pass_log] == [8, 8, 4]
    assert gen.batched_prefill_pass_log[2].slots == (28, 29, 30, 31)
    assert gen._slots_prefilled_since_decode == {7} | set(slots)


def test_host_driver_gather_head_uses_the_model_pass(expect_error):
    """head == "gather": packed passes run Model.packed_prefill_pass (re-slotted users, their own page-table rows,
    the model's kv_cache, the pass's padded batch), never the Generator's batched path; the remainder stays
    sequential through the Generator; the pass log records the head; the Generator is put in PREFILL mode."""
    from models.tt_transformers.tt.common import Mode as GeneratorMode

    options = mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=512, max_seq_len=128, head="gather")
    gen = _FakeGenerator(options)
    tokens, page_table = _fake_inputs(9)
    kv_cache = [["layer-kv"]]  # the Generator's per-model list
    out = prefill_forward_text_batched(
        gen,
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[80] * 8 + [900],
        empty_slots=list(range(9)),
        enable_trace=True,
        warmup_prefill=True,
    )
    assert torch.equal(out[:, 0, 0], tokens[:, 0].float())
    passes = gen.model[0].pass_calls
    assert [p["batch"] for p in passes] == [4, 4] and [p["padded_batch"] for p in passes] == [4, 4]
    assert passes[1]["page_rows"] == [40, 50, 60, 70] and passes[1]["lens"] == [80] * 4
    assert all(p["seq_len"] == 128 and p["kv_cache"] == ["layer-kv"] for p in passes)
    assert gen.mode == GeneratorMode.PREFILL
    assert gen.warmups == [{"trace": True, "sample": False}]  # once, on the first call
    # the 900-token user is sequential through the Generator (flag lowered, no warmup again)
    assert len(gen.calls) == 1 and gen.calls[0]["batch"] == 1 and gen.calls[0]["slots"] == [8]
    assert gen.calls[0]["flag_disable_batched"] is True and gen.calls[0]["warmup"] is False
    log = gen.batched_prefill_pass_log
    assert [(r.packed, r.traced, r.head) for r in log] == [
        (True, False, "gather"),
        (True, False, "gather"),
        (False, True, "full"),
    ]
    assert gen._slots_prefilled_since_decode == {7} | set(range(9))
    # the gather head needs the paged KV cache
    with expect_error(ValueError, "the gather head needs the paged KV cache"):
        prefill_forward_text_batched(_FakeGenerator(options), *_fake_inputs(4), prompt_lens=[80] * 4)
    # the default "full" head keeps the Generator's batched path (no model pass)
    gen = _FakeGenerator(mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=512, max_seq_len=128))
    prefill_forward_text_batched(gen, *_fake_inputs(4), prompt_lens=[80] * 4)
    assert (
        gen.model[0].pass_calls == [] and gen.calls[0]["batch"] == 4 and gen.batched_prefill_pass_log[0].head == "full"
    )


def test_host_driver_delegates_when_nothing_is_packable():
    options = mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=1024, max_seq_len=128)
    tokens, page_table = _fake_inputs(4)
    # Equal-length 2K users: the policy declines, so ONE Generator call with the batched flag lowered (the Generator
    # would otherwise batch them itself), caller's trace setting and start_pos forwarded.
    gen = _FakeGenerator(options)
    prefill_forward_text_batched(
        gen, tokens, page_table=page_table, prompt_lens=[2000] * 4, enable_trace=True, start_pos=[0, 0, 0, 0]
    )
    assert len(gen.calls) == 1
    call = gen.calls[0]
    assert call["batch"] == 4 and call["flag_disable_batched"] is True and call["trace"] is True
    assert call["start_pos"] == [0, 0, 0, 0] and call["slots"] == [0, 1, 2, 3]
    assert gen.model_args[0].disable_batched_prefill is False  # restored to the configured value
    assert [r.packed for r in gen.batched_prefill_pass_log] == [False]
    assert gen.batched_prefill_pass_log[0].users == (0, 1, 2, 3)
    # On-device sampling requested -> delegation (the batched sampling contract is not implemented).
    gen = _FakeGenerator(options)
    prefill_forward_text_batched(gen, tokens, page_table=page_table, prompt_lens=[80] * 4, sampling_params="sp")
    assert len(gen.calls) == 1 and gen.calls[0]["sampling"] == "sp" and gen.calls[0]["flag_disable_batched"] is True
    # No page table -> delegation (the unpaged fill would put re-slotted users into the wrong cache rows).
    gen = _FakeGenerator(options)
    prefill_forward_text_batched(gen, tokens, page_table=None, prompt_lens=[80] * 4)
    assert len(gen.calls) == 1 and gen.calls[0]["batch"] == 4
    # Options disabled -> delegation, flag stays disabled.
    gen = _FakeGenerator(mc.BatchedPrefillOptions(enabled=False))
    prefill_forward_text_batched(gen, tokens, page_table=page_table, prompt_lens=[80] * 4)
    assert len(gen.calls) == 1 and gen.calls[0]["flag_disable_batched"] is True
    assert gen.model_args[0].disable_batched_prefill is True
    # Explicit options override the model args (the device test uses this to run one arm per call).
    gen = _FakeGenerator(mc.BatchedPrefillOptions(enabled=False))
    prefill_forward_text_batched(gen, tokens, page_table=page_table, prompt_lens=[80] * 4, options=options)
    assert len(gen.calls) == 1 and gen.calls[0]["batch"] == 4 and gen.calls[0]["flag_disable_batched"] is False
    assert gen.model_args[0].disable_batched_prefill is True


def test_host_driver_trace_only_for_listed_shapes():
    options = mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=1024, max_seq_len=128)
    tokens, page_table = _fake_inputs(8)
    gen = _FakeGenerator(options)
    prefill_forward_text_batched(gen, tokens, page_table=page_table, prompt_lens=[80] * 8, enable_trace=True)
    assert gen.calls[0]["trace"] is False and gen.batched_prefill_pass_log[0].traced is False
    gen = _FakeGenerator(options, trace_shapes={(8, 128)})
    prefill_forward_text_batched(gen, tokens, page_table=page_table, prompt_lens=[80] * 8, enable_trace=True)
    assert gen.calls[0]["trace"] is True and gen.batched_prefill_pass_log[0].traced is True
    prefill_forward_text_batched(gen, tokens, page_table=page_table, prompt_lens=[80] * 8, enable_trace=False)
    assert gen.calls[1]["trace"] is False


def test_host_batched_prefill_flag_restores(expect_error):
    a = SimpleNamespace(disable_batched_prefill=True)
    b = SimpleNamespace(disable_batched_prefill=False)
    with batched_prefill_flag([a, b], True):
        assert a.disable_batched_prefill is False and b.disable_batched_prefill is False
    assert a.disable_batched_prefill is True and b.disable_batched_prefill is False
    with expect_error(RuntimeError, "boom"):
        with batched_prefill_flag([a, b], False):
            assert a.disable_batched_prefill is True and b.disable_batched_prefill is True
            raise RuntimeError("boom")
    assert a.disable_batched_prefill is True and b.disable_batched_prefill is False


def test_host_prepare_prefill_inputs_trace_guard(expect_error):
    calls = []
    stub = SimpleNamespace(
        args=SimpleNamespace(packed_prefill_trace_shapes=set()),
        prepare_inputs_prefill=lambda tokens, **kw: calls.append((tuple(tokens.shape), kw)) or "host_inputs",
    )
    page_table = torch.zeros(8, 2, dtype=torch.int32)
    # A batched capture of an unlisted shape is refused before any host/device work.
    with expect_error(RuntimeError, "Refusing to capture a batched prefill trace for 8 users x 128 tokens"):
        Model.prepare_prefill_inputs_trace(
            stub, torch.zeros(8, 128, dtype=torch.long), page_table=page_table, batch_size=8
        )
    assert calls == []
    # Single-user captures are untouched (today's traced 128-token prefill).
    assert Model.prepare_prefill_inputs_trace(
        stub, torch.zeros(1, 128, dtype=torch.long), page_table=page_table[:1]
    ) == ("host_inputs")
    assert calls[-1][1]["trace_enabled"] is True and "batch_size" not in calls[-1][1]
    # A listed shape passes through with its batch_size.
    stub.args.packed_prefill_trace_shapes.add((8, 128))
    assert (
        Model.prepare_prefill_inputs_trace(
            stub, torch.zeros(8, 128, dtype=torch.long), page_table=page_table, batch_size=8
        )
        == "host_inputs"
    )
    assert calls[-1][1]["batch_size"] == 8
    # Models built without ModelArgs (unit tests) have no ``args``: the guard still refuses batched captures.
    bare = SimpleNamespace(prepare_inputs_prefill=lambda tokens, **kw: "host_inputs")
    with expect_error(RuntimeError, "Refusing to capture a batched prefill trace for 2 users x 128 tokens"):
        Model.prepare_prefill_inputs_trace(
            bare, torch.zeros(2, 128, dtype=torch.long), page_table=page_table[:2], batch_size=2
        )


def test_host_model_args_fields(monkeypatch):
    """ModelArgs exposes the knobs the Generator and the driver read (no checkpoint: dummy weights, mocked device)."""
    from unittest.mock import MagicMock

    for var in BATCHED_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mc, "determine_device_name", lambda mesh_device: "P150x8")
    config_dir = os.path.join(os.path.dirname(mc.__file__), "..", "configs", "Solar-Open-100B")
    monkeypatch.setenv("HF_MODEL", os.path.normpath(config_dir))
    mesh = MagicMock(name="mesh_device_1x8")
    mesh.shape = (1, 8)
    args = mc.ModelArgs(mesh_device=mesh, dummy_weights=True, max_batch_size=32, max_seq_len=8192)
    assert args.disable_batched_prefill is True and args.batched_prefill.enabled is False
    assert args.packed_prefill_trace_shapes == set()
    assert args.can_enable_batched_prefill_trace(8, 128) is False
    monkeypatch.setenv("SOLAR_OPEN_BATCHED_PREFILL", "1")
    args = mc.ModelArgs(mesh_device=mesh, dummy_weights=True, max_batch_size=32, max_seq_len=8192)
    assert args.disable_batched_prefill is False and args.batched_prefill.enabled is True
    # Listing a shape is not enough on its own: its per-user length must be a traced length too (128 on P150x8).
    args.packed_prefill_trace_shapes.add((8, 128))
    args.packed_prefill_trace_shapes.add((4, 1024))
    assert args.can_enable_batched_prefill_trace(8, 128) is True
    assert args.can_enable_batched_prefill_trace(4, 1024) is False


# ---------------------------------------------------------------------------------------------------------------------
# Device test (1x8, real weights): batched vs sequential prefill logits and KV per user
# ---------------------------------------------------------------------------------------------------------------------


def _load_prompts_128(num_users):
    with open(PROMPTS_128) as f:
        entries = json.load(f)
    prompts = [e["prompt"] for e in entries]
    assert len(prompts) >= num_users, f"{PROMPTS_128} holds {len(prompts)} prompts, need {num_users}"
    return prompts[:num_users]


def _load_prompts_1024(num_users, tokenizer, context_tokens=800, stride=300):
    """``num_users`` DISTINCT ~1K-token prompts: consecutive windows of the cached Gutenberg text of the long-context
    demo prompt file, each followed by its own question (distinct users make a slot mix-up visible)."""
    from models.tt_transformers.demo.simple_text_demo import load_inputs

    contexts, _ = load_inputs(LONG_CONTEXT_FILE, 1, instruct=False)
    ids = tokenizer.encode(contexts[0], add_special_tokens=False)
    needed = (num_users - 1) * stride + context_tokens
    assert len(ids) >= needed, f"context of {len(ids)} tokens is too short for {num_users} windows ({needed})"
    questions = [
        "Summarize the passage above in two sentences.",
        "Who is speaking in the passage above, and to whom?",
        "List three concrete objects or places named in the passage above.",
        "What emotion dominates the passage above? Quote the words that show it.",
        "Which sentence of the passage above best states its main idea?",
        "Rewrite the last sentence of the passage above in plain modern English.",
        "Name every person mentioned in the passage above.",
        "What happens next, judging from the passage above?",
    ]
    prompts = []
    for u in range(num_users):
        window = tokenizer.decode(ids[u * stride : u * stride + context_tokens])
        prompts.append("```" + window + "```\n\n" + questions[u % len(questions)])
    return prompts


def _clear_kv(models, generator, mesh_device):
    for m in models:
        m.clear_kv_caches()
    generator.prev_page_table = None
    ttnn.synchronize_device(mesh_device)


def _decode_steps(generator, tt_kv_cache, page_table, forced, current_pos):
    """Eager teacher-forced decode: ``forced[:, t]`` fed at step t. Returns logits [T, B, V] fp32."""
    out = []
    pos = current_pos.clone()
    for t in range(forced.shape[1]):
        logits, _ = generator.decode_forward(
            forced[:, t].clone(),
            pos,
            enable_trace=False,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=None,
        )
        out.append(logits.reshape(forced.shape[0], -1).float().clone())
        pos += 1
    return torch.stack(out)


def _pcc(a, b):
    a_c = a - a.mean(dim=-1, keepdim=True)
    b_c = b - b.mean(dim=-1, keepdim=True)
    return (a_c * b_c).sum(-1) / (a_c.norm(dim=-1) * b_c.norm(dim=-1) + 1e-12)


def _kl(ref, other):
    """KL(ref || other) over the vocab in fp32, per row."""
    log_p = torch.log_softmax(ref.float(), dim=-1)
    log_q = torch.log_softmax(other.float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def _compare_rows(name, seq_logits, bat_logits):
    """Per-user PCC / KL / decisive top-1 agreement between the two arms; asserts the floors."""
    pcc = _pcc(seq_logits, bat_logits)
    kl = _kl(seq_logits, bat_logits)
    top2 = seq_logits.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    decisive = margin >= DECISIVE_MARGIN
    same_top1 = seq_logits.argmax(-1) == bat_logits.argmax(-1)
    logger.info(
        f"[{name}] per-user PCC min {pcc.min():.5f} mean {pcc.mean():.5f}; KL max {kl.max():.4f} mean {kl.mean():.4f}; "
        f"top-1 equal {int(same_top1.sum())}/{len(same_top1)} (decisive users {int(decisive.sum())}, of which equal "
        f"{int((same_top1 & decisive).sum())}); margins {[round(m, 2) for m in margin.tolist()]}"
    )
    assert pcc.min() >= PCC_MIN, f"{name}: per-user logits PCC {pcc.tolist()} below {PCC_MIN}"
    assert kl.max() <= KL_MAX, f"{name}: per-user KL {kl.tolist()} above {KL_MAX}"
    assert kl.mean() <= KL_MEAN_MAX, f"{name}: mean KL {kl.mean():.4f} above {KL_MEAN_MAX}"
    flipped = [u for u in range(len(same_top1)) if decisive[u] and not same_top1[u]]
    assert (
        not flipped
    ), f"{name}: decisive users {flipped} changed their top-1 token (margins {margin[flipped].tolist()})"


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "batch_size, seq_len, tokens_per_pass, head",
    [
        (2, 128, 256, "full"),  # T = 256: dense-bmm MoE in both arms
        (8, 128, 1024, "full"),  # one pass of 8 (the demo's default microbatch): T = 1024 expert-sorted MoE
        (32, 128, 4096, "full"),  # one pass of 32: T = 4096, the whole batch-32 demo in one forward
        (
            4,
            1024,
            4096,
            "full",
        ),  # the 1K bucket opted in (max_seq_len 1024): 4 distinct ~1K prompts in one 4096-token pass
        (8, 128, 512, "full"),  # TWO passes of 4: users 4..7 re-slotted to device rows 0..3
        (32, 128, 4096, "gather"),  # phase 3c: the same 32 x 128 pass through Model.packed_prefill_pass (gather head)
    ],
    ids=["b2_s128", "b8_s128", "b32_s128", "b4_s1024", "b8_s128_x2", "b32_s128_gather"],
)
@parametrize_mesh_with_fabric([(1, 8)])
def test_batched_vs_sequential_prefill(
    mesh_device, device_params, batch_size, seq_len, tokens_per_pass, head, state_dict
):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] != 1 or mesh_shape[1] < 8:
        pytest.skip(f"validated on 1x8 meshes (TP=8), got {mesh_shape}")
    # Imported here: the demo module's import chain opens the UMD cluster, which the host tests above must not do.
    from models.demos.solar_open.demo.text_demo import prepare_solar_open_generator_args
    from models.tt_transformers.tt.generator import Generator

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    max_seq_len = 4096
    block_size = 64
    page_params = {
        "page_block_size": block_size,
        "page_max_num_blocks_per_dp": batch_size * (max_seq_len // block_size),
    }
    model_args, models, page_table, tt_kv_cache, tokenizer, _processor, _paged_cfg = prepare_solar_open_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=setup["mesh_config"],
        state_dict=state_dict,
        users_row_sharded=False,
    )
    generator = Generator(models, model_args, mesh_device, processor=None, tokenizer=tokenizer)
    vocab = model_args[0].vocab_size

    prompts = _load_prompts_128(batch_size) if seq_len == 128 else _load_prompts_1024(batch_size, tokenizer)
    input_tokens, _encoded, decoding_pos, _prefill_lens = preprocess_inputs_prefill(
        prompts, tokenizer, model_args, instruct=False, max_generated_tokens=64, max_prefill_len=max_seq_len
    )
    tokens = torch.stack(input_tokens).view(batch_size, -1)
    lens = [int(n) for n in decoding_pos]
    padded = [get_padded_prefill_len(n) for n in lens]
    assert padded == [seq_len] * batch_size, f"prompt lengths {lens} do not all pad to {seq_len}: {padded}"
    logger.info(f"{batch_size} users, prompt lengths {lens} (padded {seq_len}), tokens_per_pass {tokens_per_pass}")

    # (a) Sequential, today's per-user path (the Generator's batched path explicitly disabled for the call).
    _clear_kv(models, generator, mesh_device)
    t0 = time.perf_counter()
    with batched_prefill_flag(model_args, False):
        seq_logits = generator.prefill_forward_text(
            tokens, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=lens, enable_trace=False
        )
    ttnn.synchronize_device(mesh_device)
    seq_wall = time.perf_counter() - t0
    seq_logits = seq_logits.reshape(batch_size, -1).float()[:, :vocab]
    forced = seq_logits.argmax(-1, keepdim=True)  # step-0 greedy token of the sequential run
    forced = torch.cat([forced] + [torch.zeros_like(forced)] * (DECODE_STEPS - 1), dim=1)
    current_pos = torch.tensor(lens)
    seq_decode = _decode_steps(generator, tt_kv_cache, page_table, forced[:, :1], current_pos)
    for t in range(1, DECODE_STEPS):  # greedy continuation of the sequential run, fed to both arms
        forced[:, t] = seq_decode[-1, :, :vocab].argmax(-1)
        seq_decode = torch.cat(
            [seq_decode, _decode_steps(generator, tt_kv_cache, page_table, forced[:, t : t + 1], current_pos + t)]
        )
    seq_decode = seq_decode[..., :vocab]

    # (b) Packed: the same users through prefill_forward_text_batched with an explicit policy for this case.
    options = mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=tokens_per_pass, max_seq_len=seq_len, head=head)
    expected_passes = math.ceil(batch_size / options.users_per_pass(seq_len))
    _clear_kv(models, generator, mesh_device)
    t0 = time.perf_counter()
    bat_logits = prefill_forward_text_batched(
        generator,
        tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=lens,
        enable_trace=False,
        options=options,
    )
    ttnn.synchronize_device(mesh_device)
    bat_wall = time.perf_counter() - t0
    log = generator.batched_prefill_pass_log
    assert len(log) == expected_passes and all(r.packed and not r.traced and r.head == head for r in log), log
    # phase 3c: a packed pass of several 1024-token splits is planned once per chunk (Model marks the pass)
    from models.demos.solar_open.tt.experts import prefill as experts_prefill

    last_plan = dict(experts_prefill.LAST_SORTED_MOE_PLAN)
    logger.info(f"sorted-MoE plan of the last split (mode {experts_prefill.SORTED_MOE_PLAN}): {last_plan}")
    if experts_prefill.SORTED_MOE_PLAN == "auto" and options.users_per_pass(seq_len) * seq_len > 1024:
        assert last_plan.get("per_chunk") is True, f"a packed multi-split pass must be planned per chunk: {last_plan}"
    assert sorted(u for r in log for u in r.users) == list(range(batch_size))
    summary = summarize_batched_prefill_log(log, batch_size)
    logger.info(
        f"sequential prefill {seq_wall * 1000:.0f} ms (eager, {batch_size} users); packed {bat_wall * 1000:.0f} ms in "
        f"{len(log)} pass(es) "
        + ", ".join(f"{len(r.users)}x{r.seq_len}={r.duration_s * 1000:.0f}ms" for r in log)
        + f"; TTFT first/mean/last {summary['first'] * 1000:.0f}/{summary['mean'] * 1000:.0f}/{summary['last'] * 1000:.0f} ms"
    )
    bat_logits = bat_logits.reshape(batch_size, -1).float()[:, :vocab]
    _compare_rows("prefill logits", seq_logits, bat_logits)

    # KV correctness: the same forced tokens over the packed cache must reproduce the sequential decode logits per
    # user (a wrong page mapping or a slot mix-up collapses that user's rows, not just adds noise).
    bat_decode = _decode_steps(generator, tt_kv_cache, page_table, forced, current_pos)[..., :vocab]
    for t in range(DECODE_STEPS):
        _compare_rows(f"decode step {t + 1} logits", seq_decode[t], bat_decode[t])
    # Every user must be its own: rows of different users differ in the sequential run, so a duplicated row in the
    # packed run (user b served user a's blocks) fails the pairwise check even when both looked plausible above.
    if batch_size > 1:
        cross = _pcc(bat_decode[-1].unsqueeze(1), bat_decode[-1].unsqueeze(0))  # [B, B]
        off_diag = cross[~torch.eye(batch_size, dtype=torch.bool)]
        seq_cross = _pcc(seq_decode[-1].unsqueeze(1), seq_decode[-1].unsqueeze(0))[
            ~torch.eye(batch_size, dtype=torch.bool)
        ]
        logger.info(f"cross-user decode logit PCC max: packed {off_diag.max():.4f}, sequential {seq_cross.max():.4f}")
        assert off_diag.max() <= max(0.999, seq_cross.max().item()), "two users of the packed run share decode logits"
