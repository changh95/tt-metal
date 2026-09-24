# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device): isolation experiments of the w=32 wedge (tests/VERIFY_W32_AUDIT.md), harness-like flow.

VERIFY_ISO = E1  : DecodeRef(32) with the test's traced argmax tail, capture + 50 replays, no VerifyPlan.
             E1b : DecodeRef(32) with the harness output path (replicated logits, no argmax ops), capture + 50 replays.
             E2  : VerifyPlan(32,4) allocation + compile + capture (no verify replay), then DecodeRef(32) capture + 50 replays.
             E3  : E2 + 50 verify replays (QWEN36_SDPA_DEC_MAX_CORES_PER_HEAD may be set by the caller).
             E4  : two live decode traces (w=8 then w=32), replays interleaved, no verify code.
             E5  : E4 with release_trace of the w=8 trace before the w=32 capture (the harness discipline).
             E6  : (a) prefill-TRACE replays interleaved with decode w32 / verify (32,4) replays: 5 rounds of
                   {8 x prefill_traced_chunked into the B=1 scratch (masked-bucket trace replay + KV fill, NO slot write),
                    10 decode replays, 10 verify replays}.
             E7  : (b) served-D pattern, eager: GDN slot writes (import_gdn_slot mode=fillcache) + KV block imports
                   (eager import_kv_blocks) into 8 slots interleaved with the same replays, no prefill trace replays.
             E8  : (b) with the traced importers the served D engine uses (import_gdn_slot mode=trace, traced KV import).
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
        if ISO in ("E2", "E3", "E6", "E7", "E8"):
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
        snap = kv0 = None
        if ISO in ("E7", "E8"):
            from models.demos.blackhole.qwen36.tt import pd_transfer as pdt

            if ISO == "E8":
                pdt.import_warmup(model, max_bucket=64)  # traced KV importer: staging buffers + per-bucket traces
                pdt.get_traced_importer(model)  # traced GDN importer (per-slot traces)
                logger.info("[iso] traced importers warmed (served D engine pattern)")
            model.pd_gdn_capture = {}
        _prefill(model, [ids], page_tables, 1)
        logger.info("[iso] warm 1-user prefill done")
        if ISO in ("E7", "E8"):
            snap = model.pd_gdn_capture.pop(0)  # (rec_snap, conv_snap) of slot 0, host device-major
            kv0 = pdt.export_kv_blocks(model, page_tables[0].tolist())
            logger.info(f"[iso] slot-0 GDN snapshot + {len(page_tables[0])} KV blocks exported for re-import")
        if vs is not None:
            vs.compile()
            vs.capture()
            logger.info(
                f"[iso] verify (32,4) compiled + captured (R={vs.plan.R}, gdn={'kernel' if vs.plan.gdn_kernel else 'stub'})"
            )
        if ISO in ("E4", "E5"):
            # two decode traces, w=8 then w=32; E4 keeps both alive and interleaves their replays, E5 releases w8 first
            ref8 = DecodeRef(model, 8, page_tables[:8])
            ref8.setup()
            logger.info("[iso] DecodeRef(8) captured")
            if ISO == "E5":
                med, mn = ref8.time_replays(N)
                logger.info(f"[iso] decode w8 replays {N}: med {med:.2f} ms; releasing its trace")
                ref8.release()
            ref = DecodeRef(model, 32, page_tables[:32])
            ref.setup()
            logger.info("[iso] DecodeRef(32) captured")
            for rnd in range(N // 10):
                if ISO == "E4":
                    med8, _ = ref8.time_replays(10)
                    med32, _ = ref.time_replays(10)
                    logger.info(
                        f"[iso] round {rnd + 1}: decode w8 {med8:.2f} ms, w32 {med32:.2f} ms (both traces alive)"
                    )
                else:
                    med32, _ = ref.time_replays(10)
                    logger.info(
                        f"[iso] decode w32 replays {10 * (rnd + 1)}/{N}: med {med32:.2f} ms (w8 trace released)"
                    )
            if ISO == "E4":
                ref8.release()
            print(f"ISO_RESULT {ISO} completed")
            return
        ref = (DecodeRefHarness if ISO == "E1b" else DecodeRef)(model, 32, page_tables[:32])
        ref.setup()
        logger.info(f"[iso] DecodeRef(32) captured ({type(ref).__name__})")
        if ISO in ("E6", "E7", "E8"):
            for rnd in range(5):
                t0 = time.perf_counter()
                if ISO == "E6":
                    prev = model._bind_gdn_prefill_scratch()
                    try:
                        for u in range(8):
                            model.prefill_traced_chunked(ids, page_tables[u : u + 1], actual_len=ids.shape[1])
                    finally:
                        model._unbind_gdn_prefill_scratch(prev)
                    ttnn.synchronize_device(device)
                    what = "8 x prefill-trace replays (no slot write)"
                else:
                    mode = "fillcache" if ISO == "E7" else "trace"
                    for u in range(8, 16):
                        pdt.import_gdn_slot(model, u, snap[0], snap[1], mode=mode)
                        pdt.import_kv_blocks(model, page_tables[u].tolist(), kv0)
                    ttnn.synchronize_device(device)
                    what = f"8 x GDN slot write ({mode}) + KV block import ({'traced' if ISO == 'E8' else 'eager'})"
                t1 = time.perf_counter()
                med32, _ = ref.time_replays(10)
                medv, _ = vs.time_replays(10)
                logger.info(
                    f"[iso] round {rnd + 1}: {what} {1e3 * (t1 - t0):.0f} ms, then decode w32 {med32:.2f} ms, verify (32,4) {medv:.2f} ms"
                )
            print(f"ISO_RESULT {ISO} completed")
            return
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
