# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: masked-bucket GDN conv through the fused KDA op (qkv_causal_conv1d_silu) vs the 22-op FIR chain.

test_kda_masked_conv_vs_fir   one real GDN layer, TP=4: for buckets 128/256/2048 and several valid lengths, compares
                              (1) the fused op's q/k/v on the REAL rows against the FIR conv output (PCC + max|d|),
                              (2) the decode conv window (new_state) bit-for-bit,
                              (3) the whole layer output + carried conv/recurrent state, masked-KDA vs masked-FIR,
                              (4) the eager int-valid_len path vs the gdn_masks (traced-body) path bit-for-bit, and
                              times the fused op at T=128 for several channel chunk sizes.
test_kda_masked_e2e_tokens    served prefill (prefill_paged_slots, traced masked buckets) + eager greedy decode on 3
                              prompts; writes the greedy tokens to $QWEN36_KDA_TOKENS_OUT and, when
                              $QWEN36_KDA_TOKENS_REF is set, asserts they equal that reference (saved on the
                              pre-change commit).
Run on one half: TT_VISIBLE_DEVICES=2,3,4,5 MESH_DEVICE=P150x4 + the served env (see scripts/profile_tp8.sh).
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import (
    load_gdn_layer,
    model_path,
    parametrize_mesh_tp,
    shard_to_device,
)
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt import masked_bucket_trace as mbt
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.tt_transformers.tt.ccl import TT_CCL


def _host(t, comp):
    return ttnn.to_torch(t, mesh_composer=comp).float()


def _cmp(name, ref, out, pcc_min=0.9999, bit=False):
    _, p = comp_pcc(ref, out, pcc_min)
    d = float((ref - out).abs().max())
    nmm = int((ref != out).sum())
    tag = "BIT-IDENTICAL" if nmm == 0 else f"mismatches={nmm}"
    logger.info(f"KDA_MASKED {name}: PCC={p} max|d|={d:.6g} {tag}")
    assert float(p) >= pcc_min, f"{name}: PCC {p} < {pcc_min}"
    if bit:
        assert nmm == 0, f"{name}: expected bit-identical, {nmm} mismatches (max|d|={d:.6g})"
    return nmm == 0


@torch.no_grad()
@parametrize_mesh_tp()
def test_kda_masked_conv_vs_fir(mesh_device, reset_seeds, ensure_gc):
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    tt_ccl = TT_CCL(mesh_device)
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    assert gdn._gdn_conv_kda and gdn._gdn_flat_qkv, "test needs QWEN36_GDN_CONV=kda (the serving default)"
    gdn._stable_state = True  # served state: persistent carry, in-place writeback (what gdn_masks requires)
    gdn.reset_state()
    comp = ttnn.ConcatMeshToTensor(mesh_device, dim=-1)
    K, C = gdn.K, gdn.qkv_dim_tp
    kd = gdn.key_dim_tp
    n_dev = mesh_device.get_num_devices()

    def set_carry(seed):
        torch.manual_seed(seed)
        carry = torch.randn(1, K - 1, C * n_dev, dtype=torch.bfloat16)
        ttnn.copy(shard_to_device(mesh_device, carry, dim=-1), gdn.conv_carry)
        return carry

    def one_hots(valid_len, bucket):
        sx, sc = mbt.host_conv_sel_split(valid_len, bucket, K)
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        return (
            ttnn.from_torch(sx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep),
            ttnn.from_torch(sc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep),
        )

    all_bit = True
    for bucket in (128, 256, 2048):
        x = torch.randn(1, 1, bucket, args.dim, dtype=torch.bfloat16)
        x_tt = shard_to_device(mesh_device, x, dim=-1)
        valid_lens = sorted({1, 2, 3, 4, 5, 31, 32, 33, bucket // 2 + 7, bucket - 3, bucket - 1, bucket})
        for vl in valid_lens:
            set_carry(1000 + bucket + vl)
            qkv, z, a, b = gdn._project_qkvzab(x_tt, bucket, out_mc=ttnn.L1_MEMORY_CONFIG)
            for t in (z, a, b):
                ttnn.deallocate(t)
            # ---- (1)+(2): conv op level ----
            conv_f, ns_f = _causal_conv1d_fir(
                qkv,
                None,
                None,
                K,
                mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                conv_state=gdn.conv_carry,
                weight_taps=tw["conv_taps"],
                bias_dev=None,
                valid_len=vl,
            )
            sel_x, sel_c = one_hots(vl, bucket)
            (q_k, k_k, v_k), ns_k = gdn._conv1d_prefill_kda(qkv, bucket, gdn.conv_carry, conv_sel=(sel_x, sel_c))
            conv_fh = _host(conv_f, comp)

            # per-device concat of [1,T,C] shards: q/k/v shards live at [d*C : d*C+kd] etc.
            def _slices(lo, hi):
                return torch.cat([conv_fh[..., d * C + lo : d * C + hi] for d in range(n_dev)], dim=-1)

            for name, ref_h, out_t in (
                ("q", _slices(0, kd), q_k),
                ("k", _slices(kd, 2 * kd), k_k),
                ("v", _slices(2 * kd, C), v_k),
            ):
                out_h = _host(out_t, comp)
                all_bit &= _cmp(f"b{bucket} vl={vl} conv {name} real rows", ref_h[:, :vl], out_h[:, :vl])
                assert torch.isfinite(out_h).all(), "pad rows must stay finite (masked downstream)"
            _cmp(f"b{bucket} vl={vl} new_state", _host(ns_f, comp), _host(ns_k, comp), bit=True)
            for t in (conv_f, ns_f, q_k, k_k, v_k, ns_k, sel_x, sel_c, qkv):
                ttnn.deallocate(t)

        # ---- (3) whole layer, masked FIR vs masked KDA, + (4) eager int valid_len vs gdn_masks ----
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        for vl in (1, 2, 5, bucket // 2 + 7, bucket - 1, bucket):
            m = mbt.host_masks(vl, bucket)
            mask_f32 = ttnn.from_torch(
                m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
            )
            mask_q = ttnn.from_torch(
                m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
            )
            conv_sel = ttnn.from_torch(
                mbt.host_conv_sel(vl, bucket, K),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=rep,
            )
            sel_x, sel_c = one_hots(vl, bucket)

            def run(kda_masked, masks):
                gdn._kda_masked = kda_masked
                gdn.reset_state_inplace()
                set_carry(77 + vl)
                torch.manual_seed(5)
                ttnn.copy(
                    ttnn.from_torch(
                        torch.randn(1, gdn.Nv, gdn.Dk, gdn.Dv) * 0.1,
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        mesh_mapper=rep,
                    ),
                    gdn.rec_state,
                )
                if masks:
                    out = gdn.forward_prefill(x_tt, chunk_size=args.gdn_chunk_size, valid_len=bucket, gdn_masks=masks)
                else:
                    out = gdn.forward_prefill(x_tt, chunk_size=args.gdn_chunk_size, valid_len=vl)
                o = _host(out, comp)[0, 0]
                ttnn.deallocate(out)
                return o, _host(gdn.conv_carry, comp), ttnn.to_torch(gdn.rec_state, mesh_composer=comp).float()

            o_fir, c_fir, r_fir = run(False, (mask_f32, mask_q, conv_sel, None))
            o_kda, c_kda, r_kda = run(True, (mask_f32, mask_q, None, (sel_x, sel_c)))
            o_eag, c_eag, r_eag = run(True, None)
            _cmp(f"b{bucket} vl={vl} LAYER out masked-KDA vs masked-FIR (real rows)", o_fir[:vl], o_kda[:vl])
            _cmp(f"b{bucket} vl={vl} LAYER conv_carry KDA vs FIR", c_fir, c_kda, bit=True)
            _cmp(f"b{bucket} vl={vl} LAYER rec_state KDA vs FIR", r_fir, r_kda)
            _cmp(f"b{bucket} vl={vl} LAYER out eager-int vs gdn_masks", o_kda, o_eag, bit=True)
            _cmp(f"b{bucket} vl={vl} LAYER conv_carry eager-int vs gdn_masks", c_kda, c_eag, bit=True)
            _cmp(f"b{bucket} vl={vl} LAYER rec_state eager-int vs gdn_masks", r_kda, r_eag, bit=True)
            for t in (mask_f32, mask_q, conv_sel, sel_x, sel_c):
                ttnn.deallocate(t)
        ttnn.deallocate(x_tt)
    logger.info(f"KDA_MASKED conv real rows bit-identical to FIR everywhere: {all_bit}")

    # ---- channel chunk size: numerics (must be bit-identical) + timing at T=128 ----
    T = 128
    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
    x_tt = shard_to_device(mesh_device, x, dim=-1)
    qkv, z, a, b = gdn._project_qkvzab(x_tt, T, out_mc=ttnn.L1_MEMORY_CONFIG)
    for t in (z, a, b):
        ttnn.deallocate(t)
    set_carry(3)
    sel_x, sel_c = one_hots(100, T)
    ref = None
    for chunk in (512, 256, 128, 64, 32):
        if C % chunk:
            continue
        (q_k, k_k, v_k), ns_k = gdn._conv1d_prefill_kda(qkv, T, gdn.conv_carry, conv_sel=(sel_x, sel_c), chunk=chunk)
        outs = [_host(t, comp) for t in (q_k, k_k, v_k, ns_k)]
        if ref is None:
            ref = outs
        else:
            for name, r, o in zip(("q", "k", "v", "ns"), ref, outs):
                assert torch.equal(r, o), f"chunk {chunk}: {name} differs from chunk 512"
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            (q2, k2, v2), ns2 = gdn._conv1d_prefill_kda(qkv, T, gdn.conv_carry, conv_sel=(sel_x, sel_c), chunk=chunk)
            for t in (q2, k2, v2, ns2):
                ttnn.deallocate(t)
        ttnn.synchronize_device(mesh_device)
        us = 1e6 * (time.perf_counter() - t0) / n
        logger.info(
            f"KDA_MASKED T=128 chunk={chunk}: fused conv + window select {us:.0f} us/call (wall, incl. dispatch)"
        )
        for t in (q_k, k_k, v_k, ns_k):
            ttnn.deallocate(t)
    # FIR for reference timing
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(20):
        c, ns = _causal_conv1d_fir(
            qkv,
            None,
            None,
            K,
            mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            conv_state=gdn.conv_carry,
            weight_taps=tw["conv_taps"],
            bias_dev=None,
            valid_len=100,
        )
        ttnn.deallocate(c)
        ttnn.deallocate(ns)
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"KDA_MASKED T=128 FIR chain: {1e6 * (time.perf_counter() - t0) / 20:.0f} us/call (wall, incl. dispatch)"
    )


# ---------------------------------------------------------------------------------------------------------------------
# Host-only (no device): the split one-hot must reproduce the FIR one-hot over concat(carry, x) exactly.
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bucket", [128, 256, 2048])
@pytest.mark.parametrize("actual_len", [1, 2, 3, 4, 5, 33, 127, 128])
def test_host_conv_sel_split_matches_concat_select(bucket, actual_len):
    K = 4
    if actual_len > bucket:
        pytest.skip("actual_len > bucket")
    torch.manual_seed(0)
    carry = torch.randn(1, K - 1, 40)
    x = torch.randn(1, bucket, 40)
    sel = mbt.host_conv_sel(actual_len, bucket, K)
    sel_x, sel_c = mbt.host_conv_sel_split(actual_len, bucket, K)
    assert tuple(sel_x.shape) == (1, K - 1, bucket) and tuple(sel_c.shape) == (1, K - 1, K - 1)
    ref = sel @ torch.cat([carry, x], dim=1)
    out = sel_x @ x + sel_c @ carry
    assert torch.equal(ref, out)
    # every window row has exactly one 1.0, in exactly one of the two selectors
    assert torch.equal(sel_x.sum(-1) + sel_c.sum(-1), torch.ones(1, K - 1))
    assert torch.equal(ref, x[:, actual_len - (K - 1) : actual_len]) if actual_len >= K - 1 else True
    # sel_c only reaches into the carry when the window starts before the bucket
    assert bool(sel_c.any()) == (actual_len < K - 1)


# ---------------------------------------------------------------------------------------------------------------------
# End-to-end greedy tokens through the served prefill path (+ eager decode), for a before/after comparison.
# ---------------------------------------------------------------------------------------------------------------------
_PROMPTS = [
    "Write a haiku about autumn rain.",
    "Translate to French: The quick brown fox jumps over the lazy dog. Then count the words in the "
    "translated sentence and explain any grammatical choices you made, in detail, paragraph by paragraph.",
    None,  # long prompt (> 2048 tokens: chunk trace + masked tail), built from the tokenizer below
]


def _build_prompts(tok):
    def _ids(text):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, enable_thinking=False
        )
        return [int(x) for x in (ids["input_ids"] if hasattr(ids, "keys") else ids)]

    para = (
        "The history of the printing press is a story of incremental refinements: movable type, oil-based inks, "
        "the screw press adapted from wine making, and the standardization of paper sizes. Each step lowered the "
        "cost of a page and widened the circle of readers. "
    )
    # Long prompt: > 2048 tokens so the served path runs the 2048 chunk trace + a masked tail (128 bucket),
    # ending with the proper generation prompt. Grow the filler word by word until the target length is hit.
    target = 2048 + 77
    body = para * 40
    while len(_ids(body + "Summarize the text above in one sentence.")) < target:
        body += "and again, "
    long_ids = _ids(body + "Summarize the text above in one sentence.")
    return [_ids(_PROMPTS[0]), _ids(_PROMPTS[1]), long_ids]


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_kda_masked_e2e_tokens(mesh_device):
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.tt_transformers.tt.common import copy_host_to_device

    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    BMAX, STEPS = 8, 8
    BPU = 40  # 2560 tokens per user >= 2048+77 prompt + decode
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = _build_prompts(tok)
    logger.info(f"[e2e] prompt lengths {[len(p) for p in prompts]}")
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    result = {}
    try:
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
        logger.info(f"[e2e] traces: mb={sorted(model._mb_traces)} chunk={model._chunked_trace_id is not None}")

        N = len(prompts)
        first, full_logits = [], []
        for u, ids in enumerate(prompts):
            lg = model.prefill_paged_slots(
                [torch.tensor([ids], dtype=torch.int32)], page_tables[u : u + 1], [u], valid_lens=[len(ids)]
            )[0]
            lg = lg.reshape(-1)[: model.vocab_size].float()
            first.append(int(lg.argmax()))
            result[f"prefill_logits_{u}"] = [round(float(v), 4) for v in lg[:8]]
            full_logits.append(lg.clone())
        ttnn.synchronize_device(device)
        # determinism: the same prompt prefilled again (another slot) must give bit-identical logits
        lg_again = (
            model.prefill_paged_slots(
                [torch.tensor([prompts[0]], dtype=torch.int32)],
                page_tables[N : N + 1],
                [N],
                valid_lens=[len(prompts[0])],
            )[0]
            .reshape(-1)[: model.vocab_size]
            .float()
        )
        lg_first = (
            model.prefill_paged_slots(
                [torch.tensor([prompts[0]], dtype=torch.int32)],
                page_tables[N + 1 : N + 2],
                [N + 1],
                valid_lens=[len(prompts[0])],
            )[0]
            .reshape(-1)[: model.vocab_size]
            .float()
        )
        det = bool(torch.equal(lg_again, lg_first))
        logger.info(f"[e2e] prefill determinism (prompt 0 twice): {'BIT-IDENTICAL' if det else 'DIFFERS'}")
        result["deterministic"] = det
        assert det, "served prefill is not bit-deterministic"
        toks = [[t] for t in first]
        tokens = torch.tensor([[t] for t in first], dtype=torch.int32)
        pt = page_tables[:N]
        for s in range(STEPS):
            positions = torch.tensor([len(prompts[u]) + s for u in range(N)], dtype=torch.int32)
            host = model.prepare_decode_inputs_host(tokens, positions, page_table=pt)
            dev = copy_host_to_device(host, mesh_device=device)
            lg, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
            out = model.process_output_decode(lg, N)  # [N,1,vocab]
            nxt = out[:, 0, :].argmax(dim=-1)
            for u in range(N):
                toks[u].append(int(nxt[u]))
            tokens = nxt.to(torch.int32).reshape(N, 1)
        for u in range(N):
            result[f"tokens_{u}"] = toks[u]
            logger.info(f"[e2e] user {u} (len {len(prompts[u])}): {toks[u]} -> {tok.decode(toks[u])!r}")
    finally:
        model.free_kv_caches()
    out_path = os.environ.get("QWEN36_KDA_TOKENS_OUT")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)
        torch.save(full_logits, out_path + ".logits.pt")
        logger.info(f"[e2e] wrote {out_path} (+ .logits.pt)")
    ref_path = os.environ.get("QWEN36_KDA_TOKENS_REF")
    if ref_path:
        with open(ref_path) as f:
            ref = json.load(f)
        bad = []
        for u in range(len(prompts)):
            same = ref[f"tokens_{u}"] == result[f"tokens_{u}"]
            logger.info(f"[e2e] user {u} tokens {'IDENTICAL' if same else 'DIFFER'} vs reference")
            if not same:
                bad.append(u)
            for k in (f"prefill_logits_{u}",):
                d = max(abs(a - b) for a, b in zip(ref[k], result[k]))
                logger.info(f"[e2e] user {u} first-8 prefill logits max|d| vs ref = {d:.4g}")
        if os.path.exists(ref_path + ".logits.pt"):
            ref_logits = torch.load(ref_path + ".logits.pt")
            for u in range(len(prompts)):
                _, p = comp_pcc(ref_logits[u], full_logits[u], 0.9999)
                d = float((ref_logits[u] - full_logits[u]).abs().max())
                logger.info(f"[e2e] user {u} FULL prefill logits vs ref: PCC={p} max|d|={d:.4g}")
        assert not bad, f"greedy tokens differ from reference for users {bad}"


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 24576, "trace_region_size": 64 * 1024 * 1024}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_kda_masked_conv_device_time(mesh_device, reset_seeds, ensure_gc):
    """Device time (trace replay, no host dispatch) of the fused conv + window select per channel chunk size vs the
    FIR chain, at T=128 and T=256, one GDN layer."""
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=4096)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = load_gdn_layer(args.CKPT_DIR, li)
    tt_ccl = TT_CCL(mesh_device)
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
    gdn._stable_state = True
    gdn.reset_state()
    K, C = gdn.K, gdn.qkv_dim_tp
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    NIT = 16
    for T in (128, 256, 512, 1024):
        x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)
        x_tt = shard_to_device(mesh_device, x, dim=-1)
        qkv, z, a, b = gdn._project_qkvzab(x_tt, T, out_mc=ttnn.L1_MEMORY_CONFIG)
        for t in (z, a, b):
            ttnn.deallocate(t)
        sx, sc = mbt.host_conv_sel_split(T - 5, T, K)
        sel_x = ttnn.from_torch(sx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        sel_c = ttnn.from_torch(sc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        conv_sel = ttnn.from_torch(
            mbt.host_conv_sel(T - 5, T, K),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )

        def body_kda(chunk):
            (q, k, v), ns = gdn._conv1d_prefill_kda(qkv, T, gdn.conv_carry, conv_sel=(sel_x, sel_c), chunk=chunk)
            for t in (q, k, v, ns):
                ttnn.deallocate(t)

        def body_fir():
            c, ns = _causal_conv1d_fir(
                qkv,
                None,
                None,
                K,
                mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                conv_state=gdn.conv_carry,
                weight_taps=tw["conv_taps"],
                bias_dev=None,
                conv_sel=conv_sel,
            )
            ttnn.deallocate(c)
            ttnn.deallocate(ns)

        variants = [(f"kda chunk={c}", (lambda c=c: body_kda(c))) for c in (512, 256, 128, 64, 32) if C % c == 0]
        variants.append(("fir chain", body_fir))
        variants.append((f"kda auto chunk={gdn._kda_chunk_for(T)}", lambda: body_kda(None)))
        for name, body in variants:
            body()  # compile
            ttnn.synchronize_device(mesh_device)
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for _ in range(NIT):
                body()
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            best = 1e9
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                best = min(best, time.perf_counter() - t0)
            ttnn.release_trace(mesh_device, tid)
            logger.info(
                f"KDA_MASKED DEVICE T={T} {name:22s}: {1e6 * best / NIT:.1f} us per conv (trace replay, {NIT} iters)"
            )
        for t in (qkv, sel_x, sel_c, conv_sel, x_tt):
            ttnn.deallocate(t)
