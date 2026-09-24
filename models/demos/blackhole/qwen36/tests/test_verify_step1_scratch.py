# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device): first verify step at (w, T) -- eager vs traced vs the plain decode, per user.

Prefills w users (3 prompts cycled, grouped <= 8), takes the plain decode's next token per user (eager
_forward_decode at width w), re-prefills, runs ONE verify step eagerly (accept 0, random drafts) and compares row 0's
argmax per user with the decode token; re-prefills, runs the same step through the trace and compares again; also
compares eager vs traced argmax rows for every row. Pins whether a (w, T) failure is a body bug (eager already wrong),
a trace-only bug, or a state (prefill) issue. Env: VERIFY_W (32), VERIFY_T (4), QWEN36_VERIFY_ATTN, QWEN36_VERIFY_GDN_STUB.
"""
import os
import random

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_verify_step_scratch import (
    BPU,
    CHUNK,
    PROMPTS,
    DecodeRef,
    _pin_aiclk,
    _prefill,
)
from models.demos.blackhole.qwen36.tt import verify_grid as vg
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.verify_step import VerifyStep

BMAX = 32
W = int(os.environ.get("VERIFY_W", "32"))
T = int(os.environ.get("VERIFY_T", "4"))


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_step1(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    _pin_aiclk(1200)
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt_ids = [tok(p, return_tensors="pt").input_ids.to(torch.int32) for p in PROMPTS]
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    vs = ref = None
    try:
        vs = VerifyStep(model, W, T, page_tables[:W])
        vs.compile()
        ref = DecodeRef(model, W, page_tables[:W])
        ref.compile()
        pt_full = torch.arange(BMAX * BPU, dtype=torch.int32).reshape(1, -1)
        prev = model._bind_gdn_prefill_scratch()
        try:
            model.capture_prefill_trace_chunked(device, pt_full, chunk_size=CHUNK, capture_chunk_trace=True)
        finally:
            model._unbind_gdn_prefill_scratch(prev)
        model.warmup_gdn_slot_write()
        for layer in model.layers:
            if not layer.is_full_attention and hasattr(layer.attention, "warmup_hist_device_pack"):
                layer.attention.warmup_hist_device_pack()
        ttnn.synchronize_device(device)
        _prefill(model, prompt_ids, page_tables, 1)
        vs.capture()
        ref.capture()
        rng = random.Random(7)
        drafts = [[rng.randrange(model.vocab_size) for _ in range(T - 1)] for _ in range(W)]

        # plain decode next token per user (traced decode reference, one step)
        lens, first = _prefill(model, prompt_ids, page_tables, W)
        nxt = ref.step(first, lens)
        logger.info(f"[step1] decode next tokens: {nxt[:12]}...")

        def run_verify(eager):
            lens2, first2 = _prefill(model, prompt_ids, page_tables, W)
            assert first2 == first
            tokens = [[first[s]] + drafts[s] for s in range(W)]
            rows = vs.run(tokens, lens2, [0] * W, eager=eager)
            return [int(rows[vg.row(s, 0, T)]) for s in range(W)], rows

        row0_eager, rows_eager = run_verify(True)
        row0_traced, rows_traced = run_verify(False)
        bad_e = [s for s in range(W) if row0_eager[s] != nxt[s]]
        bad_t = [s for s in range(W) if row0_traced[s] != nxt[s]]
        diff_rows = [r for r in range(vs.plan.R) if int(rows_eager[r]) != int(rows_traced[r])]
        logger.info(f"[step1] ({W},{T}) row0 eager != decode for users {bad_e} ({len(bad_e)}/{W})")
        logger.info(f"[step1] ({W},{T}) row0 traced != decode for users {bad_t} ({len(bad_t)}/{W})")
        logger.info(
            f"[step1] ({W},{T}) eager vs traced argmax rows differ at {len(diff_rows)}/{vs.plan.R} rows: {diff_rows[:20]}"
        )
        # same-prompt users must agree with each other in every mode
        for name, r0 in (("eager", row0_eager), ("traced", row0_traced)):
            groups = {}
            for s in range(W):
                groups.setdefault(s % len(PROMPTS), set()).add(r0[s])
            logger.info(
                f"[step1] {name} row0 per prompt group (distinct values): {[sorted(v) for v in groups.values()]}"
            )
        print(
            f"STEP1_RESULT w={W} T={T} eager_bad={len(bad_e)} traced_bad={len(bad_t)} eager_vs_traced_rows_diff={len(diff_rows)}"
        )
    finally:
        if vs is not None:
            vs.release()
        if ref is not None:
            ref.release()
        model.pd_gdn_capture = None
        model.free_kv_caches()
