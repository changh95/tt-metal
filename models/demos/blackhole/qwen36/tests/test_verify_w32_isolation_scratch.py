# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device): isolation experiments of the w=32 wedge (tests/VERIFY_W32_AUDIT.md), harness-like flow.

VERIFY_ISO = E1  : DecodeRef(32) with the test's traced argmax tail, capture + 50 replays, no VerifyPlan.
             E1b : DecodeRef(32) with the harness output path (replicated logits, no argmax ops), capture + 50 replays.
             E2  : VerifyPlan(32,4) allocation + compile + capture (no verify replay), then DecodeRef(32) capture + 50 replays.
             E3  : E2 + 50 verify replays (QWEN36_SDPA_DEC_MAX_CORES_PER_HEAD may be set by the caller).
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_verify_step_scratch import BPU, CHUNK, DecodeRef, _pin_aiclk, _prefill
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.verify_step import VerifyStep
from models.tt_transformers.tt.common import copy_host_to_device

BMAX = 32
ISO = os.environ.get("VERIFY_ISO", "E1")
N = int(os.environ.get("VERIFY_ISO_REPLAYS", "50"))


class DecodeRefHarness(DecodeRef):
    """The harness's decode output: replicated full logits via _lm_head (no argmax ops in the trace)."""

    def _fwd(self, dev):
        cos, sin = unpack_rope(dev[2])
        logits = self.model._forward_decode(dev[0], cos, sin, dev[1], dev[3], sharded_lm_head=False)
        return logits, None

    def step(self, tokens, positions):
        host = self.model.prepare_decode_inputs_host(
            torch.tensor(tokens, dtype=torch.int32).reshape(self.w, 1),
            torch.tensor(positions, dtype=torch.int32),
            self.pt,
        )
        copy_host_to_device(host_tensors=host, device_tensors=self.dev)
        ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh)
        full = ttnn.to_torch(ttnn.get_device_tensors(self.out[0])[0]).float().reshape(-1, self.model.args.vocab_size)
        return full[: self.w].argmax(-1).tolist()

    def time_replays(self, n):
        ms = []
        for i in range(n):
            t0 = time.perf_counter()
            self.step([100 + i] * self.w, [9 + i] * self.w)
            ms.append(1e3 * (time.perf_counter() - t0))
        ms.sort()
        return ms[len(ms) // 2], ms[0]


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_w32_isolation(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    _pin_aiclk(int(os.environ.get("VERIFY_FORCE_AICLK_MHZ", "1200")))
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    vs = ref = None
    try:
        if ISO in ("E2", "E3"):
            vs = VerifyStep(model, 32, 4, page_tables[:32])
            logger.info("[iso] VerifyPlan(32,4) allocated before the prefill captures")
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
        logger.info("[iso] prefill warm-up done")
        ids = torch.randint(1000, 100000, (1, 40), dtype=torch.int32)
        _prefill(model, [ids], page_tables, 1)
        logger.info("[iso] warm 1-user prefill done")
        if vs is not None:
            vs.compile()
            vs.capture()
            logger.info(
                f"[iso] verify (32,4) compiled + captured (R={vs.plan.R}, gdn={'kernel' if vs.plan.gdn_kernel else 'stub'})"
            )
        ref = (DecodeRefHarness if ISO == "E1b" else DecodeRef)(model, 32, page_tables[:32])
        ref.setup()
        logger.info(f"[iso] DecodeRef(32) captured ({type(ref).__name__})")
        for chunk in range(N // 10):
            med, mn = ref.time_replays(10)
            logger.info(f"[iso] decode w32 replays {10 * (chunk + 1)}/{N}: med {med:.2f} min {mn:.2f} ms")
        if ISO == "E3":
            for chunk in range(N // 10):
                med, mn = vs.time_replays(10)
                logger.info(f"[iso] verify (32,4) replays {10 * (chunk + 1)}/{N}: med {med:.2f} min {mn:.2f} ms")
        print(f"ISO_RESULT {ISO} completed")
    finally:
        if vs is not None:
            vs.release()
        if ref is not None:
            ref.release()
        model.pd_gdn_capture = None
        model.free_kv_caches()
