# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-logits reference harness for the qwen36 SMALL-M prefill matmul path (item J: all-gather + 1D mcast matmul
instead of the AGMM / 2D configs for buckets <= QWEN36_PREFILL_SMALLM_MAX; serving default 128).

The K accumulation order of the 1D kernel differs from the AGMM's, so the bar is NOT bit-identity for the short
buckets:
  * PCC >= 0.999 of the prefill logits (full vocab, position actual_len-1) vs the reference on every prompt, and
  * identical greedy tokens over STEPS (>= 8) decode steps on the first GREEDY_WIDTH (4 >= 3) prompts.
Prompts: ISL 100 / 128 (bucket 128) take the new path; the CONTROL prompts (``QWEN36_SMALLM_CONTROL_ISLS``, default
200,256,400 = buckets 256 and 512 under the MAX=128 default) stay on the AGMM / 2D path and must match bit-exactly
(max|d| == 0). Checking MAX=256 instead: QWEN36_PREFILL_SMALLM_MAX=256 QWEN36_SMALLM_CONTROL_ISLS=400 (that check
FAILED on 2026-09-24 -- the ISL-256 prompt's greedy tokens diverged at decode step 3 -- hence the 128 default).

Modes, selected by ``QWEN36_SMALLM_REF=save|check`` (``QWEN36_SMALLM_REF_DIR`` = output dir):
  * ``save``  -- run on the REFERENCE tree (or with the new path disabled), dump logits + greedy tokens.
  * ``check`` -- run on the changed tree, assert the bars above (prints the per-prompt PCC).

Served path exactly as profile_prefill_decode.py: Qwen36Model, chunked prefill trace warm-up (captures the masked
bucket traces), prefill_paged_slots per prompt, traced decode with copy_host_to_device per step.

Run (half A):
  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.8-27B HF_HUB_OFFLINE=1 ARCH_NAME=blackhole \
  TT_QWEN35_TEXT_VER=qwen36_blackhole TT_DECODE_BUCKETING=1 QWEN36_PLAIN_GDN_SLOT_DEVICE_COPY_FORCE=1 \
  QWEN36_GDN_HIST_DEVICE_PACK=1 QWEN36_SMALLM_REF=save QWEN36_SMALLM_REF_DIR=logs/itemJ_ref \
  python_env/bin/python -m pytest models/demos/blackhole/qwen36/tests/test_prefill_smallm_ref_scratch.py -x -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS

MODE = os.environ.get("QWEN36_SMALLM_REF", "save")
REF_DIR = os.environ.get("QWEN36_SMALLM_REF_DIR", "logs/itemJ_ref")
STEPS = int(os.environ.get("QWEN36_SMALLM_STEPS", "8"))
ISLS = [int(x) for x in os.environ.get("QWEN36_SMALLM_ISLS", "100,128,200,256,400").split(",")]
GREEDY_WIDTH = 4  # the first 4 prompts (>= 3), a power-of-two decode bucket
PCC_BAR = float(os.environ.get("QWEN36_SMALLM_PCC_BAR", "0.999"))
# Prompts asserted bit-exact (unchanged AGMM / 2D path): buckets 256 and 512 under the MAX=128 serving default.
CONTROL_ISLS = {int(x) for x in os.environ.get("QWEN36_SMALLM_CONTROL_ISLS", "200,256,400").split(",") if x}
BMAX = 32
BPU = 72  # same KV geometry as profile_prefill_decode.py


def _pcc(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def _save_or_check(name, tensors):
    os.makedirs(REF_DIR, exist_ok=True)
    path = os.path.join(REF_DIR, f"{name}.pt")
    if MODE == "save":
        torch.save({"tensors": tensors}, path)
        logger.info(f"[ref] saved {len(tensors)} tensors -> {path}")
        return
    ref = torch.load(path)["tensors"]
    missing = [k for k in ref if k not in tensors]
    assert not missing, f"missing keys in the new run: {missing}"
    bad = []
    min_pcc = 1.0
    for k, v in tensors.items():
        r = ref[k]
        if r.shape != v.shape:
            bad.append((k, f"shape ref {tuple(r.shape)} vs new {tuple(v.shape)}"))
            continue
        if k.endswith("_logits"):
            p = _pcc(r, v)
            maxd = (r.float() - v.float()).abs().max().item()
            top_same = int(r.float().argmax()) == int(v.float().argmax())
            isl = int(k.split("_")[1])
            control = isl in CONTROL_ISLS
            min_pcc = min(min_pcc, p) if not control else min_pcc
            logger.info(
                f"[ref] {k}: pcc={p:.6f} max|d|={maxd:.4f} top1 same={top_same}{' (control, bit-exact)' if control else ''}"
            )
            if control and maxd != 0.0:
                bad.append((k, f"control prompt not bit-exact (max|d| {maxd})"))
            if p < PCC_BAR:
                bad.append((k, f"pcc {p:.6f} < {PCC_BAR}"))
            if not top_same:
                bad.append((k, "top-1 differs"))
        else:
            if not torch.equal(r, v):
                bad.append((k, f"tokens differ: ref {r.tolist()} vs new {v.tolist()}"))
            else:
                logger.info(f"[ref] {k}: identical ({v.numel()} values)")
    for k, msg in bad:
        logger.error(f"[ref] FAIL {k}: {msg}")
    logger.info(f"[ref] min prefill-logits PCC (small-M prompts) = {min_pcc:.6f} (bar {PCC_BAR})")
    assert not bad, f"prefill reference check failed: {len(bad)} entries"
    logger.info(f"[ref] PASS: all {len(tensors)} entries of {name} meet the bar (min PCC {min_pcc:.6f})")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_model_prefill_smallm_reference(mesh_device):
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.tt_transformers.tt.common import copy_host_to_device

    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    logger.info(f"[ref] model load {time.perf_counter() - t0:.1f}s BPU={BPU} isls={ISLS} steps={STEPS} mode={MODE}")
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    out = {}
    try:
        t0 = time.perf_counter()
        pt_full = torch.arange(BMAX * BPU, dtype=torch.int32).reshape(1, -1)
        prev = model._bind_gdn_prefill_scratch()
        try:
            model.capture_prefill_trace_chunked(device, pt_full, chunk_size=2048, capture_chunk_trace=True)
        finally:
            model._unbind_gdn_prefill_scratch(prev)
        model.warmup_gdn_slot_write()
        for layer in model.layers:
            if not layer.is_full_attention and hasattr(layer.attention, "warmup_hist_device_pack"):
                layer.attention.warmup_hist_device_pack()
        ttnn.synchronize_device(device)
        logger.info(f"[ref] prefill warmup {time.perf_counter() - t0:.1f}s (bucket traces: {sorted(model._mb_traces)})")

        # One prompt per slot: full prefill logits saved per ISL.
        first_tok = {}
        for slot, isl in enumerate(ISLS):
            gen = torch.Generator().manual_seed(1000 + isl)
            ids = torch.randint(1000, 100000, (1, isl), dtype=torch.int32, generator=gen)
            t0 = time.perf_counter()
            lg = model.prefill_paged_slots([ids], page_tables[slot : slot + 1], [slot], valid_lens=[isl])
            ttnn.synchronize_device(device)
            dt = time.perf_counter() - t0
            logits = lg[0].reshape(-1)[: model.vocab_size].float().clone()
            first_tok[slot] = int(logits.argmax())
            out[f"prefill_{isl}_logits"] = logits
            logger.info(f"[ref] prefill isl={isl} slot={slot} {1e3 * dt:.1f} ms top={first_tok[slot]}")

        # Greedy generation on the first GREEDY_WIDTH prompts: traced decode bucket of that width, argmax fed back.
        gw = GREEDY_WIDTH
        cur = torch.tensor([[first_tok[u]] for u in range(gw)], dtype=torch.int32)
        pos = torch.tensor([ISLS[u] for u in range(gw)], dtype=torch.int32)
        seq = [cur.reshape(-1).clone()]
        pt = page_tables[:gw]
        # eager compile step, then the trace (served idiom); the eager step's output is used as step 0
        dev = copy_host_to_device(model.prepare_decode_inputs_host(cur, pos, page_table=pt), mesh_device=device)
        lg0, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        ttnn.synchronize_device(device)
        lt = model.process_output_decode(lg0, gw).to(torch.bfloat16)
        nxt = lt.float().argmax(-1).reshape(-1)[:gw].to(torch.int32)
        seq.append(nxt.clone())
        logger.info(f"[ref] greedy step=0 (eager) tokens={nxt.tolist()}")
        cur = nxt.reshape(gw, 1)
        pos = pos + 1
        host = model.prepare_decode_inputs_host(cur, pos, page_table=pt)
        dev = copy_host_to_device(host, mesh_device=device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        lg, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        for s in range(1, STEPS + 1):
            host = model.prepare_decode_inputs_host(cur, pos, page_table=pt)
            copy_host_to_device(host_tensors=host[:3], device_tensors=dev[:3])
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            lt = model.process_output_decode(lg, gw).to(torch.bfloat16)
            nxt = lt.float().argmax(-1).reshape(-1)[:gw].to(torch.int32)
            seq.append(nxt.clone())
            logger.info(f"[ref] greedy step={s} tokens={nxt.tolist()}")
            cur = nxt.reshape(gw, 1)
            pos = pos + 1
        ttnn.release_trace(device, tid)
        out["greedy_tokens"] = torch.stack(seq, dim=1)  # [gw, STEPS+2]
    finally:
        model.pd_gdn_capture = None
        model.free_kv_caches()
    _save_or_check("model_prefill_smallm", out)
