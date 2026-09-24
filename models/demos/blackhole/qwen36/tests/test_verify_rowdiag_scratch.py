# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device): find the row-position dependent op of the R>32 verify body. All w users get the SAME prompt and
the SAME drafts, so after every layer the per-user row blocks must be bitwise identical; the first layer where a user's
block differs from user 0's names the culprit (VERIFY_W, VERIFY_T; stub GDN by default)."""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_verify_step_scratch import BPU, CHUNK, PROMPTS, _pin_aiclk, _prefill
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.verify_step import VerifyStep

BMAX = 32
W = int(os.environ.get("VERIFY_W", "8"))
T = int(os.environ.get("VERIFY_T", "8"))


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_rowdiag(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    _pin_aiclk(1200)
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    ids = tok(PROMPTS[1], return_tensors="pt").input_ids.to(torch.int32)
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    vs = None
    try:
        vs = VerifyStep(model, W, T, page_tables[:W])
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
        vs.compile()
        lens, first = _prefill(model, [ids] * 3, page_tables, W)  # every user: prompt 1
        assert len(set(first)) == 1, first
        drafts = [11, 12, 13, 14, 15, 16, 17][: T - 1]
        tokens = [[first[0]] + drafts for _ in range(W)]
        report = []

        def row_check(name, x):
            ttnn.synchronize_device(device)
            h = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float().reshape(-1, x.shape[-1])[: vs.plan.R]
            base = h[:T]
            diffs = []
            for s in range(1, W):
                blk = h[s * T : (s + 1) * T]
                if not torch.equal(blk, base):
                    diffs.append((s, float((blk - base).abs().max()), int((blk != base).sum())))
            line = f"[rowdiag] {name}: shape={tuple(x.shape)} users_differing_from_user0={diffs}"
            logger.info(line)
            report.append(line)

        vs.plan.upload(tokens, lens, [0] * W)
        idx, val = vs.forward(row_check=row_check)
        ttnn.synchronize_device(device)
        ttnn.deallocate(idx)
        ttnn.deallocate(val)
        print("ROWDIAG_REPORT\n" + "\n".join(report))
    finally:
        if vs is not None:
            vs.release()
        model.pd_gdn_capture = None
        model.free_kv_caches()
