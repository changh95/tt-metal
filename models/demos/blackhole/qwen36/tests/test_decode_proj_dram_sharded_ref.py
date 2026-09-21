# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode-logits reference harness for the qwen36 decode PROJECTION matmul relayout (DRAM-sharded multi-reader
kernel for MLP gate/up/down, GDN qkvzab/out, attention qkv/wo).

The K-accumulation order of the new kernel differs from the 1D mcast_in0 kernel, so the bar is NOT bit-identity:
  * PCC >= 0.9999 of the decode logits vs the reference at widths 1 and 32 for STEPS traced steps, and
  * identical greedy tokens over >= 8 decode steps on 3 (here 4) prompts.

Modes, selected by ``QWEN36_PROJ_REF=save|check`` (``QWEN36_PROJ_REF_DIR`` = output dir):
  * ``save``  -- run on the REFERENCE tree, dump the logits + greedy tokens.
  * ``check`` -- run on the changed tree, assert the PCC / token bars above (prints the per-step PCC).

Served path exactly as profile_prefill_decode.py / test_attention_decode_layout_bitexact.py: Qwen36Model, chunked
prefill trace warm-up, prefill_paged_slots for every slot, per-width decode trace with copy_host_to_device per step.

Run (half A):
  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.8-27B HF_HUB_OFFLINE=1 ARCH_NAME=blackhole \
  TT_QWEN35_TEXT_VER=qwen36_blackhole TT_DECODE_BUCKETING=1 QWEN36_PLAIN_GDN_SLOT_DEVICE_COPY_FORCE=1 \
  QWEN36_GDN_HIST_DEVICE_PACK=1 QWEN36_PROJ_REF=save QWEN36_PROJ_REF_DIR=logs/itemA_ref \
  python_env/bin/python -m pytest models/demos/blackhole/qwen36/tests/test_decode_proj_dram_sharded_ref.py -x -s
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS

MODE = os.environ.get("QWEN36_PROJ_REF", "save")
REF_DIR = os.environ.get("QWEN36_PROJ_REF_DIR", "logs/itemA_ref")
MODEL_WIDTHS = [int(w) for w in os.environ.get("QWEN36_PROJ_WIDTHS", "1,32").split(",")]
STEPS = int(os.environ.get("QWEN36_PROJ_STEPS", "8"))
GREEDY_WIDTH = 4  # 3 prompts + 1 (bucket widths are powers of two)
PCC_BAR = float(os.environ.get("QWEN36_PROJ_PCC_BAR", "0.9999"))

ISL = 128
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
            min_pcc = min(min_pcc, p)
            top_r, top_v = r.float().argmax(-1), v.float().argmax(-1)
            same_top = int((top_r == top_v).sum())
            logger.info(
                f"[ref] {k}: pcc={p:.6f} max|d|={(r.float() - v.float()).abs().max().item():.4f} "
                f"top1 identical {same_top}/{top_r.numel()}"
            )
            if p < PCC_BAR:
                bad.append((k, f"pcc {p:.6f} < {PCC_BAR}"))
            if same_top != top_r.numel():
                bad.append((k, f"top-1 differs on {top_r.numel() - same_top} rows"))
        else:
            if not torch.equal(r, v):
                bad.append((k, f"tokens differ: ref {r.tolist()} vs new {v.tolist()}"))
            else:
                logger.info(f"[ref] {k}: identical ({v.numel()} values)")
    for k, msg in bad:
        logger.error(f"[ref] FAIL {k}: {msg}")
    logger.info(f"[ref] min logits PCC over all steps/widths = {min_pcc:.6f} (bar {PCC_BAR})")
    assert not bad, f"decode reference check failed: {len(bad)} entries"
    logger.info(f"[ref] PASS: all {len(tensors)} entries of {name} meet the bar (min PCC {min_pcc:.6f})")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_model_decode_proj_reference(mesh_device):
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.tt_transformers.tt.common import copy_host_to_device

    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    logger.info(f"[ref] model load {time.perf_counter() - t0:.1f}s BPU={BPU} widths={MODEL_WIDTHS} steps={STEPS}")
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    out = {}

    def run_traced(width, tokens_fn, positions_fn, pt, on_step):
        """One eager step (program-cache warm-up; trace capture cannot compile), then capture the width bucket
        once and run STEPS steps with per-step host inputs (served idiom). The eager step's logits are recorded
        under step index -1 by the caller's on_step(-1, ...)."""
        dev0 = model.prepare_inputs_decode(tokens_fn(-1), positions_fn(-1), pt)
        lg0, _ = model.ttnn_decode_forward(dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3])
        ttnn.synchronize_device(device)
        on_step(-1, model.process_output_decode(lg0, width).to(torch.bfloat16))
        host = model.prepare_decode_inputs_host(tokens_fn(0), positions_fn(0), page_table=pt)
        dev = copy_host_to_device(host, mesh_device=device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        lg, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        for s in range(STEPS):
            host = model.prepare_decode_inputs_host(tokens_fn(s), positions_fn(s), page_table=pt)
            copy_host_to_device(host_tensors=host[:3], device_tensors=dev[:3])
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            on_step(s, model.process_output_decode(lg, width).to(torch.bfloat16))
        ttnn.release_trace(device, tid)

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
        logger.info(f"[ref] prefill warmup {time.perf_counter() - t0:.1f}s")

        # Prefill EVERY slot with its own fixed prompt so all 32 users have real KV + GDN state.
        t0 = time.perf_counter()
        first_tok = {}
        for slot in range(BMAX):
            gen = torch.Generator().manual_seed(1000 + slot)
            ids = torch.randint(1000, 100000, (1, ISL), dtype=torch.int32, generator=gen)
            lg = model.prefill_paged_slots([ids], page_tables[slot : slot + 1], [slot], valid_lens=[ISL])
            first_tok[slot] = int(lg[0].reshape(-1)[: model.vocab_size].float().argmax())
        out["prefill_top"] = torch.tensor([first_tok[s] for s in range(BMAX)])
        ttnn.synchronize_device(device)
        logger.info(f"[ref] {BMAX} slot prefills {time.perf_counter() - t0:.1f}s")

        # (1) fixed-input traced steps at widths 1 / 32: full logits saved per step.
        for width in MODEL_WIDTHS:
            base_tokens = torch.tensor([[100 + u] for u in range(width)], dtype=torch.int32)
            base_pos = torch.tensor([ISL + (u % 5) for u in range(width)], dtype=torch.int32)

            def on_step(s, lt, width=width):
                tag = "eager" if s < 0 else f"s{s}"
                out[f"w{width}_{tag}_logits"] = lt[:width].clone()
                logger.info(f"[ref] width={width} step={tag} top ids (first 4): {lt.float().argmax(-1)[:4].tolist()}")

            # eager step at s=-1 uses positions base_pos (tokens base_tokens - 1); traced steps s>=0 follow.
            run_traced(width, lambda s: base_tokens + s, lambda s: base_pos + 1 + s, page_tables[:width], on_step)

        # (2) greedy generation: prompts = slots 0..3 (prefilled above), first token = prefill argmax,
        #     then STEPS steps feeding back the argmax (traced width-4 bucket).
        gw = GREEDY_WIDTH
        cur = torch.tensor([[first_tok[u]] for u in range(gw)], dtype=torch.int32)
        seq = [cur.reshape(-1).clone()]
        # The fixed-step section above advanced slots 0..width-1 by STEPS+1 positions with synthetic tokens; the
        # greedy section is still a deterministic function of the same state, so positions just continue.
        pos0 = torch.tensor([ISL + 5 + STEPS + 2] * gw, dtype=torch.int32)

        def greedy_step(s, lt):
            nxt = lt.float().argmax(-1).reshape(-1)[:gw].to(torch.int32)
            cur.copy_(nxt.reshape(gw, 1))
            seq.append(nxt.clone())
            logger.info(f"[ref] greedy step={s} tokens={nxt.tolist()}")

        run_traced(gw, lambda s: cur.clone(), lambda s: pos0 + 1 + s, page_tables[:gw], greedy_step)
        out["greedy_tokens"] = torch.stack(seq, dim=1)  # [gw, STEPS+1]
    finally:
        model.pd_gdn_capture = None
        model.free_kv_caches()
    _save_or_check("model_decode_proj", out)
