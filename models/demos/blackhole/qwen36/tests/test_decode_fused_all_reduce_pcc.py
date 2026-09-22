# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy gate for the qwen36 decode fused all-reduce (replicated residual) path, on the SERVED model path.

Two modes, ``QWEN36_AR_REF=save|check`` (``QWEN36_AR_REF_DIR`` = reference dir):

* ``save``  -- run on the reference tree (or with QWEN36_DECODE_FUSED_AR=0), dump per-step greedy decode logits.
* ``check`` -- run on the changed tree; every step's logits must reach PCC >= PCC_MIN (default 0.998, the measured
  reduction-order noise floor, see below) against the saved ones and the greedy tokens must be identical for every
  prompt over all steps. QWEN36_AR_DECODE_TOPOLOGY=linear|ring forces the decode CCL topology after the prefills (the
  noise-floor control: old path, different summation order). QWEN36_AR_NEW_TAG names the saved check-run dump.

Flow: Qwen36Model as served (BMAX=32 slots, paged KV, prefill chunk trace); 3 real prompts are prefilled once into the
32 slots (slot u <- prompt u % 3), then greedy decode (argmax fed back) traced per width exactly as
profile_prefill_decode.py captures it: width 32 (all rows) then width 1 (row 0 continues prompt 0), STEPS steps each.

Run (half B):
  TT_VISIBLE_DEVICES=2,3,4,5 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.8-27B HF_HUB_OFFLINE=1 ARCH_NAME=blackhole \
  TT_QWEN35_TEXT_VER=qwen36_blackhole QWEN36_AR_REF=save QWEN36_AR_REF_DIR=logs/itemB_ref \
  python_env/bin/python -m pytest models/demos/blackhole/qwen36/tests/test_decode_fused_all_reduce_pcc.py -x -s
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS

MODE = os.environ.get("QWEN36_AR_REF", "save")
REF_DIR = os.environ.get("QWEN36_AR_REF_DIR", "logs/itemB_ref")
STEPS = int(os.environ.get("QWEN36_AR_STEPS", "10"))
# Gate: identical greedy tokens AND per-row logits PCC >= PCC_MIN on every step. The default is the measured
# summation-order noise floor of this model, NOT 0.9999: on the reference tree, merely switching the OLD decode
# reduce-scatter/all-gather from Ring to Linear (a different order for the same 4-way bf16 sum, nothing else) gives
# step PCC 0.99906-0.99975 (w32) / 0.99955-0.99970 (w1), per-row min 0.99809, tokens identical; the fused
# all_reduce_async lands in the same band (step 0.99886-0.99975 / 0.99935-0.99967, per-row min 0.99807, tokens
# identical). The bf16 residual stream + GDN recurrence amplify 1-ulp differences to that level, so 0.9999 is not
# reachable for any change of the reduction order. QWEN36_AR_PCC_MIN overrides.
PCC_MIN = float(os.environ.get("QWEN36_AR_PCC_MIN", "0.998"))
BMAX = 32
BPU = 72
PROMPTS = [
    "The quick brown fox jumps over the lazy dog. Explain in two sentences why this pangram is famous:",
    "Q: What is the capital of France, and which river flows through it?\nA:",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
]


def _pcc(a, b):
    # float64: an fp32 dot over 32 x 152k logits is only good to ~1e-3, useless against a 0.9999 bar
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_decode_fused_all_reduce_pcc(mesh_device):
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.tt_transformers.tt.common import copy_host_to_device

    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    logger.info(
        f"[ar] model load {time.perf_counter() - t0:.1f}s mode={MODE} steps={STEPS} "
        f"fused_ar={getattr(model.args, 'decode_fused_all_reduce', None)}"
    )
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt_ids = [torch.tensor(tok(p)["input_ids"], dtype=torch.int32).reshape(1, -1) for p in PROMPTS]
    lens = [int(p.shape[1]) for p in prompt_ids]
    logger.info(f"[ar] prompt lengths {lens}")

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
        logger.info(f"[ar] prefill warmup {time.perf_counter() - t0:.1f}s")

        # Prefill every slot ONCE, up front (re-prefilling a slot after a decode sequence was observed to hand later
        # decodes a broken state on this tree; the vLLM warmup order -- traces first, then prefill -- did the same in
        # this harness), then capture each width's trace on its real first-step inputs (the eager compile run and the
        # capture pass advance the GDN state with that same token twice more: a fixed, deterministic perturbation shared
        # by the reference and the check) and replay STEPS greedy steps.
        first_tok = [None] * BMAX
        t0 = time.perf_counter()
        for slot in range(BMAX):
            p = slot % len(PROMPTS)
            lg = model.prefill_paged_slots([prompt_ids[p]], page_tables[slot : slot + 1], [slot], valid_lens=[lens[p]])
            first_tok[slot] = int(lg[0].reshape(-1)[: model.vocab_size].float().argmax())
        ttnn.synchronize_device(device)
        logger.info(f"[ar] {BMAX} slot prefills {time.perf_counter() - t0:.1f}s")
        # Control knob for the noise-floor experiment: QWEN36_AR_DECODE_TOPOLOGY=linear switches the DECODE CCLs (the
        # reduce-scatter's summation order, and the all-gathers) to Linear after the prefills, leaving prefill untouched.
        _topo = os.environ.get("QWEN36_AR_DECODE_TOPOLOGY")
        if _topo:
            _t = {"linear": ttnn.Topology.Linear, "ring": ttnn.Topology.Ring}[_topo.lower()]
            model.args.ccl_topology = lambda: _t
            _ar = getattr(model.tt_ccl, "decode_all_reduce", None)
            if _ar is not None:
                _ar.topology = _t
            logger.info(f"[ar] decode CCL topology forced to {_t}")
        # per-row running state: next token to feed and its position (row u holds prompt u % 3)
        next_tok = list(first_tok)
        next_pos = [lens[u % len(PROMPTS)] for u in range(BMAX)]

        def greedy_decode(width, tag):
            """Capture the width bucket (rows 0..width-1) on its current inputs, replay STEPS times with greedy argmax
            fed back; records logits + tokens per step."""
            pt = page_tables[:width]
            tokens = torch.tensor([[next_tok[u]] for u in range(width)], dtype=torch.int32)
            positions = torch.tensor(next_pos[:width], dtype=torch.int32)
            dev0 = model.prepare_inputs_decode(tokens, positions, pt)
            model.ttnn_decode_forward(dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3])
            ttnn.synchronize_device(device)
            host = model.prepare_decode_inputs_host(tokens, positions, page_table=pt)
            dev = copy_host_to_device(host, mesh_device=device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            lg = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
            if isinstance(lg, tuple):
                lg = lg[0]
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            logits_all, toks_all = [], [tokens.reshape(-1).clone()]
            for st in range(STEPS):
                tokens = torch.tensor([[next_tok[u]] for u in range(width)], dtype=torch.int32)
                positions = torch.tensor(next_pos[:width], dtype=torch.int32)
                host = model.prepare_decode_inputs_host(tokens, positions, page_table=pt)
                copy_host_to_device(host_tensors=host[:3], device_tensors=dev[:3])
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(device)
                lt = model.process_output_decode(lg, width).reshape(width, -1)[:, : model.vocab_size].to(torch.bfloat16)
                logits_all.append(lt.clone())
                nxt = lt.float().argmax(-1)
                for u in range(width):
                    next_tok[u] = int(nxt[u])
                    next_pos[u] += 1
                toks_all.append(nxt.to(torch.int32).clone())
            ttnn.release_trace(device, tid)
            out[tag] = {"logits": torch.stack(logits_all), "tokens": torch.stack(toks_all)}
            for u in range(min(3, width)):
                logger.info(f"[ar] {tag} row {u}: {tok.decode(out[tag]['tokens'][:, u].tolist())!r}")

        greedy_decode(BMAX, "w32")  # rows 0..31 = prompts u % 3
        greedy_decode(1, "w1")  # row 0 continues prompt 0's sequence at width 1
    finally:
        model.pd_gdn_capture = None
        model.free_kv_caches()

    os.makedirs(REF_DIR, exist_ok=True)
    path = os.path.join(REF_DIR, "decode_greedy_ref.pt")
    if MODE == "save":
        torch.save(out, path)
        logger.info(f"[ar] saved reference -> {path}")
        return
    ref = torch.load(path)
    _new_tag = os.environ.get("QWEN36_AR_NEW_TAG", "new")
    torch.save(out, os.path.join(REF_DIR, f"decode_greedy_{_new_tag}.pt"))
    worst = 1.0
    bad = []
    for tag, cur in out.items():
        r = ref[tag]
        same_tok = torch.equal(r["tokens"], cur["tokens"])
        pccs = [_pcc(r["logits"][s], cur["logits"][s]) for s in range(STEPS)]
        row_pccs = [
            min(_pcc(r["logits"][s][u], cur["logits"][s][u]) for u in range(cur["logits"].shape[1]))
            for s in range(STEPS)
        ]
        worst = min(worst, min(pccs), min(row_pccs))
        logger.info(
            f"[ar] {tag}: tokens_identical={same_tok} step PCC min={min(pccs):.6f} (per-row min {min(row_pccs):.6f}) "
            f"max|d|={(r['logits'].float() - cur['logits'].float()).abs().max().item():.4f}"
        )
        print(
            f"AR_PCC_STEPS {tag} step_pcc={[round(v, 6) for v in pccs]} row_min_pcc={[round(v, 6) for v in row_pccs]}",
            flush=True,
        )
        if not same_tok:
            nd = int((r["tokens"] != cur["tokens"]).sum())
            print(f"AR_PCC_TOKENS {tag}: {nd} of {r['tokens'].numel()} greedy tokens differ", flush=True)
        if not same_tok:
            bad.append(f"{tag}: tokens differ")
        if min(row_pccs) < PCC_MIN:
            bad.append(f"{tag}: PCC {min(row_pccs):.6f} < {PCC_MIN}")
    print(f"AR_PCC_RESULT worst_pcc={worst:.6f} bad={bad}", flush=True)
    assert not bad, bad
