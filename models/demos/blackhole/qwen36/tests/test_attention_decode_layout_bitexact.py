# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-identity harness for the qwen36 decode attention LAYOUT cleanup (cos/sin [1,1,B,rd] packing, dropped
pad/i2s/s2i/copy/reshape plumbing in attention/tp.py decode).

Two modes, selected by ``QWEN36_LAYOUT_REF=save|check`` (``QWEN36_LAYOUT_REF_DIR`` = output dir):

* ``save``  — run on the REFERENCE commit, dump outputs.
* ``check`` — run on the changed tree, assert every saved tensor is reproduced EXACTLY (torch.equal on bf16 bits).

Tests:
* ``test_attention_layer_decode_bitexact``: one real full-attention layer (weights from the HF snapshot), paged KV,
  eager decode at widths 1/8/32 for several steps with per-user positions; saves the layer output per step and a
  hash of the paged K/V cache; also records the per-step device-op list (graph capture) to report ops removed.
* ``test_model_decode_bitexact``: the served path (Qwen36Model, prefill_paged_slots for every slot, per-width decode
  trace exactly as profile_prefill_decode.py) — logits hash + top-1 ids per step at widths 1 and 32.

Run (half A):
  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.8-27B HF_HUB_OFFLINE=1 ARCH_NAME=blackhole \
  TT_QWEN35_TEXT_VER=qwen36_blackhole QWEN36_LAYOUT_REF=save QWEN36_LAYOUT_REF_DIR=logs/itemD_ref \
  python_env/bin/python -m pytest models/demos/blackhole/qwen36/tests/test_attention_decode_layout_bitexact.py -x -s
"""
import hashlib
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp

MODE = os.environ.get("QWEN36_LAYOUT_REF", "save")
REF_DIR = os.environ.get("QWEN36_LAYOUT_REF_DIR", "logs/itemD_ref")
LAYER_WIDTHS = [int(w) for w in os.environ.get("QWEN36_LAYOUT_WIDTHS", "1,8,32").split(",")]
MODEL_WIDTHS = [int(w) for w in os.environ.get("QWEN36_LAYOUT_MODEL_WIDTHS", "1,32").split(",")]
STEPS = int(os.environ.get("QWEN36_LAYOUT_STEPS", "4"))


def _sha(t):
    t = t.contiguous()
    return hashlib.sha256(
        t.view(torch.int16).numpy().tobytes() if t.dtype == torch.bfloat16 else t.numpy().tobytes()
    ).hexdigest()


def _save_or_check(name, tensors, extra=None):
    """tensors: dict name -> torch tensor. save: write; check: torch.equal against the saved file."""
    os.makedirs(REF_DIR, exist_ok=True)
    path = os.path.join(REF_DIR, f"{name}.pt")
    if MODE == "save":
        torch.save({"tensors": tensors, "extra": extra or {}}, path)
        logger.info(f"[ref] saved {len(tensors)} tensors -> {path}")
        return
    ref = torch.load(path)["tensors"]
    bad = []
    for k, v in tensors.items():
        r = ref[k]
        if r.shape != v.shape or r.dtype != v.dtype or not torch.equal(r, v):
            d = (r.float() - v.float()).abs().max().item() if r.shape == v.shape else float("nan")
            bad.append((k, tuple(r.shape), tuple(v.shape), d))
    missing = [k for k in ref if k not in tensors]
    for k, rs, vs, d in bad:
        logger.error(f"[ref] MISMATCH {k}: ref {rs} vs new {vs}, max|diff|={d}")
    assert not bad and not missing, f"bit-identity broken: {len(bad)} mismatching, missing {missing}"
    logger.info(f"[ref] BIT-IDENTICAL: all {len(tensors)} tensors of {name} equal the reference ({path})")


def _op_names(graph):
    """Device-op launch names from a ttnn graph capture (best effort: falls back to every function_start name)."""
    if isinstance(graph, str):
        graph = json.loads(graph)
    names = [n.get("params", {}).get("name", "") for n in graph if n.get("node_type") == "function_start"]
    dev = [n.split("::")[-1] for n in names if "DeviceOperation" in n or "::prim::" in n]
    if dev:
        return dev
    logger.warning(f"[ops] no device-op names matched; distinct function_start names: {sorted(set(names))[:80]}")
    return names


@torch.no_grad()
@parametrize_mesh_tp()
def test_attention_layer_decode_bitexact(mesh_device, reset_seeds, ensure_gc):
    from models.demos.blackhole.qwen36.tests.test_factory import load_attn_layer, model_path, tp_composer
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, rot_mats_decode
    from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    os.environ.setdefault("HF_MODEL", model_path())
    BMAX = 32
    args = Qwen36ModelArgs(mesh_device, max_batch_size=BMAX, max_seq_len=512)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    NH, NKV, HD, rd = args.n_local_heads, args.n_local_kv_heads, args.head_dim, args.rope_head_dim
    block, bpu = 64, 4
    logger.info(f"layer={li} NH={NH} NKV={NKV} HD={HD} rd={rd} widths={LAYER_WIDTHS} steps={STEPS} mode={MODE}")
    sd = load_attn_layer(args.CKPT_DIR, li)
    tt_ccl = TT_CCL(mesh_device)
    tw = load_attention_weights_tp(mesh_device, sd, args)
    attn = TPAttention(mesh_device, args, tw, tt_ccl)

    def mk_cache():
        return ttnn.from_torch(
            torch.zeros(BMAX * bpu, NKV, block, HD, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def rep(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
        )

    k_cache, v_cache = mk_cache(), mk_cache()
    attn.set_paged_kv_cache(k_cache, v_cache)
    comp = tp_composer(mesh_device)
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(BMAX)])

    out = {}
    ops = {}
    g = torch.Generator().manual_seed(1234)
    for B in LAYER_WIDTHS:
        pt_tt = rep(page_table[:B], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for s in range(STEPS):
            x = torch.randn(1, 1, B, args.dim, generator=g).to(torch.bfloat16)
            pos = torch.tensor(
                [(7 * u + 3 * s + 5) % (bpu * block - STEPS - 1) + s for u in range(B)], dtype=torch.int32
            )
            cos, sin = rot_mats_decode(mesh_device, rd, args.max_seq_len, args.rope_theta, pos)
            cur_tt = rep(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            capture = s == 0
            if capture:
                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            y = attn.forward_decode(rep(x), cur_tt, cos, sin, page_table=pt_tt)
            if capture:
                try:
                    ops[B] = _op_names(ttnn.graph.end_graph_capture())
                except Exception as e:  # best effort
                    logger.warning(f"graph capture failed: {e}")
                    ops[B] = []
            out[f"B{B}_s{s}_out"] = ttnn.to_torch(y, mesh_composer=comp).to(torch.bfloat16).cpu()
            ttnn.deallocate(y)
            # rope-only probe on a random q (the decode rope helper is exercised standalone as well)
            if s == 0:
                q = torch.randn(1, B, NH, HD, generator=g).to(torch.bfloat16)
                qr = apply_partial_rope_decode(rep(q), cos, sin, NH, B, rd)
                out[f"B{B}_rope_q"] = (
                    ttnn.to_torch(qr, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
                    .to(torch.bfloat16)[:1]
                    .cpu()
                )
                ttnn.deallocate(qr)
        kc = ttnn.to_torch(ttnn.get_device_tensors(k_cache)[0]).to(torch.bfloat16)
        vc = ttnn.to_torch(ttnn.get_device_tensors(v_cache)[0]).to(torch.bfloat16)
        out[f"B{B}_kcache_hash"] = torch.tensor([int(_sha(kc)[:15], 16)])
        out[f"B{B}_vcache_hash"] = torch.tensor([int(_sha(vc)[:15], 16)])
        out[f"B{B}_kcache_rows"] = kc[: bpu * min(B, 2)].reshape(-1)[:65536].clone()
        out[f"B{B}_vcache_rows"] = vc[: bpu * min(B, 2)].reshape(-1)[:65536].clone()
        logger.info(f"[ops] width {B}: {len(ops[B])} device ops in forward_decode: {ops[B]}")
    os.makedirs(REF_DIR, exist_ok=True)
    with open(os.path.join(REF_DIR, f"layer_ops_{MODE}.json"), "w") as f:
        json.dump({str(k): v for k, v in ops.items()}, f, indent=1)
    _save_or_check("attention_layer_decode", out, extra={"ops": {str(k): v for k, v in ops.items()}})


# ---------------------------------------------------------------------------------------------------------------
# Served-path model test (mirrors profile_prefill_decode.py)
# ---------------------------------------------------------------------------------------------------------------
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS  # noqa: E402

ISL = 128
BMAX = 32
BPU = 72  # blocks per user, same KV geometry as profile_prefill_decode.py (ISLs 128/4096, 50 steps)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_model_decode_bitexact(mesh_device):
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

        # prefill EVERY slot with its own fixed prompt so all 32 users have real KV + GDN state
        t0 = time.perf_counter()
        for slot in range(BMAX):
            gen = torch.Generator().manual_seed(1000 + slot)
            ids = torch.randint(1000, 100000, (1, ISL), dtype=torch.int32, generator=gen)
            lg = model.prefill_paged_slots([ids], page_tables[slot : slot + 1], [slot], valid_lens=[ISL])
            out[f"prefill_slot{slot}_top"] = lg[0].reshape(-1)[: model.vocab_size].float().argmax().reshape(1)
        ttnn.synchronize_device(device)
        logger.info(f"[ref] {BMAX} slot prefills {time.perf_counter() - t0:.1f}s")

        for width in MODEL_WIDTHS:
            tokens = torch.tensor([[100 + u] for u in range(width)], dtype=torch.int32)
            positions = torch.tensor([ISL + (u % 5) for u in range(width)], dtype=torch.int32)  # per-user positions
            pt = page_tables[:width]
            dev0 = model.prepare_inputs_decode(tokens, positions, pt)
            lg0, _ = model.ttnn_decode_forward(dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3])
            ttnn.synchronize_device(device)
            e = model.process_output_decode(lg0, width).to(torch.bfloat16)
            out[f"w{width}_eager_hash"] = torch.tensor([int(_sha(e)[:15], 16)])
            out[f"w{width}_eager_top"] = e.float().argmax(-1).reshape(-1)
            out[f"w{width}_eager_rows"] = e[[0, width - 1]].reshape(-1).clone()

            host = model.prepare_decode_inputs_host(tokens, positions + 1, page_table=pt)
            dev = copy_host_to_device(host, mesh_device=device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            lg, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            for s in range(STEPS):
                host = model.prepare_decode_inputs_host(tokens + s, positions + 2 + s, page_table=pt)
                copy_host_to_device(host_tensors=host[:3], device_tensors=dev[:3])
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(device)
                lt = model.process_output_decode(lg, width).to(torch.bfloat16)
                out[f"w{width}_s{s}_hash"] = torch.tensor([int(_sha(lt)[:15], 16)])
                out[f"w{width}_s{s}_top"] = lt.float().argmax(-1).reshape(-1)
                out[f"w{width}_s{s}_rows"] = lt[[0, width - 1]].reshape(-1).clone()
                logger.info(f"[ref] width={width} step={s} top ids (first 4): {out[f'w{width}_s{s}_top'][:4].tolist()}")
            ttnn.release_trace(device, tid)
    finally:
        model.pd_gdn_capture = None
        model.free_kv_caches()
    _save_or_check("model_decode_traced", out)
