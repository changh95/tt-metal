# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunked single-user prefill (phase 3d, ``tt/chunked_prefill.py``) against the unchunked pass on the WHOLE model
(1x8, real weights through the demo's model path; warm cache: 1-2 min load, then ~30 s).

Two arms over the SAME ~7K-token prompt (raw tokenizer, no chat template: the token ids do not depend on the date)
through the tt_transformers ``Generator``:

  (a) one 8192-token pass (``max_prefill_chunk_size = 8192``, the phase-3c path) into page-table row A;
  (b) the Generator's chunk loop with ``max_prefill_chunk_size`` = 4096 (two chunks) or 2048 (four) into row B (the 64K
      cases: 60000 tokens of the full book padded to 65536 -- ``chunk32k`` = the demo's 64K parity pair, one 64K pass vs
      two 32K chunks, no-garbage floors because the unchunked arm is the phase-3c long pass with its own per-pass rules;
      ``chunk32k_bf16`` = the same pair with that pass held at bf16 attention output; ``chunk16k_vs_32k`` = two chunked
      arms, exact; see the CASES comment for the recorded digits and the attribution): chunk
      i > 0 writes its K / V through ``chunk_page_table`` and attends with
      ``ttnn.transformer.chunked_scaled_dot_product_attention`` over the cache prefix, RoPE rows offset by the chunk
      start (``Model.prepare_inputs_prefill`` / ``attention/prefill.py``).

Asserts: the chunk loop ran exactly the schedule ``chunked_prefill.chunk_schedule`` predicts (chunk starts, page-table
slice widths, chunk lengths); the last-token logits of the two arms agree (PCC, KL, decisive top-1); the K / V blocks
of the compared layers agree on every device (one KV head per device at TP=8) within bfp8 noise -- separately for the
chunk-0 range (legacy ops: expected near-exact) and for the later chunks (a missing RoPE offset or a wrong block
mapping collapses their PCC); one teacher-forced decode step over each cache agrees; the single-user chunks were NOT
marked as a packed pass for the sorted-MoE planner. Not bit-identical by construction: the chunked SDPA accumulates
the online softmax over the prefix in another order and the row-count-dependent projections take other auto program
configs -- measured bit-equal nevertheless at 8K (2 / 4 chunks) and between chunk sizes at 64K, while the single 64K pass
differs from layer 1 on. Every measured value is logged; the floors are per case (CASES; design D13).

    pytest models/demos/solar_open/tests/test_chunked_prefill.py -k 1x8 -p no:cacheprovider          # all 5 cases, ~12 min
    pytest models/demos/solar_open/tests/test_chunked_prefill.py -k "1x8 and (chunk4k or chunk2k)" -x  # the 8K pair, ~1 min
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.solar_open.tt import chunked_prefill as cp
from models.tt_transformers.tt.common import get_padded_prefill_len

BLOCK_SIZE = 64
# Layer 1 is compared so a divergence that starts right after layer 0 (the only measured 64K-pass difference, see the
# case comments) is attributed to the layer-0 block and not to the accumulated residual.
COMPARED_LAYERS = (0, 1, 23, 47)
FILE_16K = "models/tt_transformers/demo/sample_prompts/input_data_long_16k.json"
FILE_128K = "models/tt_transformers/demo/sample_prompts/input_data_long_128k.json"  # the full book: 102,605 raw tokens
# Cases. ``a_chunk`` None = arm A is the single unchunked pass, else arm A is chunked too (self-consistency pair);
# ``bf16_above`` pins attention/prefill.py::ATTENTION_BFP8_OUTPUT_ABOVE_TOKENS for the case (None = the shipped rule);
# ``floors`` = (logits PCC, logits KL, KV PCC, KV bit-equal fraction, decode PCC), every value is logged before it is judged.
# 7000 tokens pad to 8192; the last prompt token lies in chunk 1 of 2 (4096-token chunks) / chunk 3 of 4 (2048), so every
# chunk of the padded prefill runs and the chunk holding the last token is a chunked-SDPA chunk. The 64K cases (phase 3d /
# P1 / R1) are the demo's 64K parity pair on the whole model: 60000 raw tokens of the full text pad to 65536.
# Recorded 2026-09-09 (phase 3d, ledgers scratchpad/phase3d/{P1,R1}/runs.txt):
#   chunk4k / chunk2k (8K: one pass vs 2 x 4096 / 4 x 2048): logits PCC 1.000002, KL 0.00000, every K / V range of layers
#     0 / 23 / 47 on all 8 devices max |diff| 0.0000 and bit-equal 1.0000, decode PCC 0.999998 -> TIGHT floors.
#   chunk16k_vs_32k (two CHUNKED arms, 2 x 32768 vs 4 x 16384, no dtype difference): PCC 1.000003, KL 0.00000, every K / V
#     range bit-equal 1.0000, decode PCC 0.999997 -> the chunk loop is exact against itself at any chunk size -> TIGHT.
#   chunk32k_bf16 (one 64K pass HELD at bf16 attention output vs 2 x 32768): layer 0 K / V bit-equal 1.0000 in BOTH ranges
#     (embedding, norm, qkv projection and the RoPE rows >= 32768 are exact), but layers 23 / 47 differ already in the
#     chunk-0 range (bit-equal 0.10-0.35, K PCC min 0.9986 / 0.9994, V 0.9942 / 0.9865), logits PCC 0.996885, KL 0.01413,
#     same top-1, decode PCC 0.994995 -> the divergence is seeded inside the layer block of the UNCHUNKED 64K pass, not by
#     the chunk mechanism: the per-pass rules a 32K chunk never triggers (the bfp8 attention output -- pinned off here --,
#     and the auto program configs of the row-count-dependent matmuls: the o_proj `ttnn.matmul` of attention/operations.py
#     carries no program config, so a 65536-row pass and a 32768-row chunk block their K accumulation differently; the MoE
#     cuts every pass into 4096-token chunks first, so its split size and hot / cold plan do not depend on the pass
#     length, and `ttnn.move` above 32K relocates without arithmetic). Floors = those digits with margin.
#   chunk32k (the SHIPPED pair: the phase-3c single 64K pass with its bfp8 attention output vs two 32K chunks): logits PCC
#     0.968027 / 0.974618 (two runs), KL 0.07808 / 0.07369, same top-1 ('.\n\n', margin 1.625 / 1.000), decode step PCC
#     0.977309; the K / V of the deep layers DIVERGE between the two arms although the logits agree -- layer 23 K 0.86910 /
#     V 0.64166 (chunk-0 positions) and V 0.88658 (chunked positions), layer 47 V 0.64161 / 0.78207 (layer 0 bit-equal): the
#     unchunked arm's bfp8 attention output changes the residual stream from layer 0 on, so the two arms compute DIFFERENT
#     deep-layer activations of the same prompt (chunk32k_bf16, which holds that arm at bf16, has K / V PCC >= 0.98). So
#     this pair is gated on its logits / top-1 / decode only (no-garbage floors below), and its K / V PCC is logged, not
#     judged (floor 0.0). NOT expected bit-equal: the chunked path is the higher-fidelity computation of a > 32K prompt.
TIGHT = (0.999, 0.01, 0.999, 0.99, 0.999)
BF16_PAIR = (0.99, 0.05, 0.98, 0.0, 0.99)
SHIPPED_64K = (0.95, 0.25, 0.0, 0.0, 0.9)
CASES = {
    "chunk4k": dict(
        file=FILE_16K, max_seq_len=8 * 1024, prompt=7000, a_chunk=None, b_chunk=4096, floors=TIGHT, bf16_above=None
    ),
    "chunk2k": dict(
        file=FILE_16K, max_seq_len=8 * 1024, prompt=7000, a_chunk=None, b_chunk=2048, floors=TIGHT, bf16_above=None
    ),
    "chunk32k": dict(
        file=FILE_128K,
        max_seq_len=64 * 1024,
        prompt=60000,
        a_chunk=None,
        b_chunk=32768,
        floors=SHIPPED_64K,
        bf16_above=None,
    ),
    "chunk32k_bf16": dict(
        file=FILE_128K,
        max_seq_len=64 * 1024,
        prompt=60000,
        a_chunk=None,
        b_chunk=32768,
        floors=BF16_PAIR,
        bf16_above=1 << 30,
    ),
    "chunk16k_vs_32k": dict(
        file=FILE_128K, max_seq_len=64 * 1024, prompt=60000, a_chunk=32768, b_chunk=16384, floors=TIGHT, bf16_above=None
    ),
}
DECISIVE_MARGIN = 3.0  # unchunked top-1 margin (logits) above which a top-1 flip is not a near tie


def _pcc(a, b):
    a_c = a - a.mean(dim=-1, keepdim=True)
    b_c = b - b.mean(dim=-1, keepdim=True)
    return (a_c * b_c).sum(-1) / (a_c.norm(dim=-1) * b_c.norm(dim=-1) + 1e-12)


def _kl(ref, other):
    log_p = torch.log_softmax(ref.float(), dim=-1)
    log_q = torch.log_softmax(other.float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def _prompt_tokens(tokenizer, context_file, num_tokens):
    """``[1, num_tokens]`` ids of the cached Gutenberg context of the long-context demo file (raw tokenizer)."""
    from models.tt_transformers.demo.simple_text_demo import load_inputs

    contexts, _ = load_inputs(context_file, 1, instruct=False)
    ids = tokenizer.encode(contexts[0], add_special_tokens=False)
    assert len(ids) >= num_tokens, f"{context_file} yields {len(ids)} tokens, need {num_tokens}"
    return torch.tensor(ids[:num_tokens], dtype=torch.long).reshape(1, num_tokens)


def _prefill(generator, tt_kv_cache, page_table, tokens, length, mesh_device):
    generator.prev_page_table = None
    t0 = time.perf_counter()
    logits = generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[length],
        enable_trace=False,
        warmup_prefill=False,
    )
    ttnn.synchronize_device(mesh_device)
    return logits.reshape(1, -1).float(), time.perf_counter() - t0


def _decode_step(generator, tt_kv_cache, page_table, token, pos):
    generator.prev_page_table = None
    logits, _ = generator.decode_forward(
        torch.tensor([token]),
        torch.tensor([pos]),
        enable_trace=False,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        sampling_params=None,
    )
    return logits.reshape(1, -1).float()


def _read_kv_blocks(model, page_table_row, num_blocks, layers):
    """``{layer: (k, v)}`` with ``k`` / ``v`` ``[devices, num_blocks * block_size, head_dim]`` fp32: the user's first
    ``num_blocks`` blocks in position order, per device (each device holds one KV head at TP=8)."""
    blocks = page_table_row[0, :num_blocks].long()
    out = {}
    for layer in layers:
        k_cache, v_cache = model.layers[layer].self_attn.layer_past
        ks, vs = [], []
        for dev_k, dev_v in zip(ttnn.get_device_tensors(k_cache), ttnn.get_device_tensors(v_cache)):
            k = ttnn.to_torch(dev_k).float()[blocks]  # [num_blocks, 1, block_size, head_dim]
            v = ttnn.to_torch(dev_v).float()[blocks]
            ks.append(k.reshape(-1, k.shape[-1]))
            vs.append(v.reshape(-1, v.shape[-1]))
        out[layer] = (torch.stack(ks), torch.stack(vs))
    return out


def _compare_kv(name, ref, other, lo, hi):
    """PCC of positions ``[lo, hi)`` per device (flattened over positions x head_dim); returns the min over devices."""
    a = ref[:, lo:hi].reshape(ref.shape[0], -1)
    b = other[:, lo:hi].reshape(other.shape[0], -1)
    assert a.abs().sum() > 0, f"{name}: the unchunked cache blocks [{lo}, {hi}) are all zero (fill did not happen)"
    assert b.abs().sum() > 0, f"{name}: the chunked cache blocks [{lo}, {hi}) are all zero (fill did not happen)"
    pcc = _pcc(a, b)
    equal = (a == b).float().mean().item()
    logger.info(
        f"[{name}] positions [{lo}, {hi}): PCC per device min {pcc.min():.6f} mean {pcc.mean():.6f}, max |diff| "
        f"{(a - b).abs().max():.4f}, bit-equal fraction {equal:.4f}"
    )
    return pcc.min().item(), equal


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
@parametrize_mesh_with_fabric([(1, 8)])
def test_chunked_vs_unchunked_prefill(mesh_device, device_params, case, state_dict, monkeypatch):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] != 1 or mesh_shape[1] < 8:
        pytest.skip(f"validated on 1x8 meshes (TP=8), got {mesh_shape}")
    cfg = CASES[case]
    MAX_SEQ_LEN, PROMPT_TOKENS, a_chunk, b_chunk = cfg["max_seq_len"], cfg["prompt"], cfg["a_chunk"], cfg["b_chunk"]
    PCC_MIN, KL_MAX, KV_PCC_MIN, KV_BIT_EQUAL_MIN, DECODE_PCC_MIN = cfg["floors"]
    # Imported here: the demo module's import chain opens the UMD cluster, which host-only collection must not do.
    from models.demos.solar_open.demo.text_demo import prepare_solar_open_generator_args
    from models.demos.solar_open.tt.attention import prefill as attention_prefill
    from models.demos.solar_open.tt.experts import prefill as experts_prefill
    from models.tt_transformers.tt.generator import Generator

    if cfg["bf16_above"] is not None:
        monkeypatch.setattr(attention_prefill, "ATTENTION_BFP8_OUTPUT_ABOVE_TOKENS", cfg["bf16_above"])
    logger.info(
        f"case {case}: arm A {'unchunked' if a_chunk is None else f'{a_chunk}-token chunks'} vs arm B {b_chunk}-token "
        f"chunks over {PROMPT_TOKENS} tokens padded to {MAX_SEQ_LEN}; attention bfp8 output above "
        f"{attention_prefill.ATTENTION_BFP8_OUTPUT_ABOVE_TOKENS} tokens per pass; floors {cfg['floors']}"
    )
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    blocks_per_user = MAX_SEQ_LEN // BLOCK_SIZE
    page_params = {"page_block_size": BLOCK_SIZE, "page_max_num_blocks_per_dp": 2 * blocks_per_user}
    model_args, models, _pt, tt_kv_cache, tokenizer, _processor, _cfg = prepare_solar_open_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=1,
        optimizations=None,
        max_seq_len=MAX_SEQ_LEN,
        page_params=page_params,
        paged_attention=True,
        mesh_config=setup["mesh_config"],
        state_dict=state_dict,
        users_row_sharded=False,
    )
    generator = Generator(models, model_args, mesh_device, processor=None, tokenizer=tokenizer)
    model = models[0]
    vocab = model_args[0].vocab_size
    assert get_padded_prefill_len(PROMPT_TOKENS) == MAX_SEQ_LEN
    tokens = _prompt_tokens(tokenizer, cfg["file"], PROMPT_TOKENS)
    # Two disjoint block ranges of the pool, one per arm, so both caches can be read back and compared.
    pt_a = torch.arange(0, blocks_per_user, dtype=torch.int32).reshape(1, -1)
    pt_b = torch.arange(blocks_per_user, 2 * blocks_per_user, dtype=torch.int32).reshape(1, -1)
    schedule_a = cp.chunk_schedule(MAX_SEQ_LEN, a_chunk or MAX_SEQ_LEN, BLOCK_SIZE, PROMPT_TOKENS - 1)
    schedule_b = cp.chunk_schedule(MAX_SEQ_LEN, b_chunk, BLOCK_SIZE, PROMPT_TOKENS - 1)
    logger.info(f"chunk schedule A: {schedule_a}")
    logger.info(f"chunk schedule B ({b_chunk}-token chunks): {schedule_b}")
    assert len(schedule_b) == MAX_SEQ_LEN // b_chunk and schedule_b[-1].is_last
    assert schedule_a[-1].end == schedule_b[-1].end

    # Record every model-level prefill call the Generator makes (chunk starts, page-table slices, lengths).
    calls = []
    original_forward = model.ttnn_prefill_forward

    def recording_forward(*args, **kwargs):
        x = args[0] if args else kwargs["x"]
        chunk_pt = kwargs.get("chunk_page_table")
        calls.append(
            {
                "seq_len": int(x.shape[-2]),
                "chunk_start_idx": kwargs.get("chunk_start_idx"),
                "chunk_blocks": None if chunk_pt is None else int(chunk_pt.shape[-1]),
            }
        )
        return original_forward(*args, **kwargs)

    def check_calls(schedule, chunk):
        if chunk is None:
            assert len(calls) == 1 and calls[0]["seq_len"] == MAX_SEQ_LEN and calls[0]["chunk_blocks"] is None, calls
            assert calls[0]["chunk_start_idx"] in (None, 0), calls
        else:
            assert [c["chunk_start_idx"] for c in calls] == [c.start for c in schedule], calls
            assert all(c["seq_len"] == chunk for c in calls), calls
            assert [c["chunk_blocks"] for c in calls] == [c.block_end - c.block_start for c in schedule], calls

    monkeypatch.setattr(model, "ttnn_prefill_forward", recording_forward)
    model.clear_kv_caches()
    ttnn.synchronize_device(mesh_device)

    # arm A (unchunked: one pass of the padded tokens, the phase-3c path -- or the coarser chunking of a self-consistency pair)
    model_args[0].max_prefill_chunk_size = a_chunk or MAX_SEQ_LEN
    logits_a, wall_a = _prefill(generator, tt_kv_cache, pt_a, tokens, PROMPT_TOKENS, mesh_device)
    check_calls(schedule_a, a_chunk)
    n_a = len(calls)
    kv_a = _read_kv_blocks(model, pt_a, schedule_b[-1].block_end, COMPARED_LAYERS)

    # arm B: the Generator's chunk loop
    calls.clear()
    model_args[0].max_prefill_chunk_size = b_chunk
    logits_b, wall_b = _prefill(generator, tt_kv_cache, pt_b, tokens, PROMPT_TOKENS, mesh_device)
    logger.info(f"prefill wall: A x{n_a} {wall_a * 1000:.0f} ms, B x{len(calls)} {wall_b * 1000:.0f} ms")
    check_calls(schedule_b, b_chunk)
    last_plan = dict(experts_prefill.LAST_SORTED_MOE_PLAN)
    logger.info(f"sorted-MoE plan of the last split (mode {experts_prefill.SORTED_MOE_PLAN}): {last_plan}")
    assert (
        last_plan.get("per_chunk") is not True
    ), f"a single-user chunk must not be planned as a packed pass: {last_plan}"
    kv_b = _read_kv_blocks(model, pt_b, schedule_b[-1].block_end, COMPARED_LAYERS)

    problems = []

    # Last-token logits
    a, b = logits_a[:, :vocab], logits_b[:, :vocab]
    pcc = _pcc(a, b).item()
    kl = _kl(a, b).item()
    top2 = a.topk(2, dim=-1).values[0]
    margin = (top2[0] - top2[1]).item()
    top1_a, top1_b = int(a.argmax(-1)), int(b.argmax(-1))
    logger.info(
        f"[prefill logits] PCC {pcc:.6f}, KL {kl:.5f}, top-1 {top1_a} vs {top1_b} ({tokenizer.decode([top1_a])!r} vs "
        f"{tokenizer.decode([top1_b])!r}), arm-A margin {margin:.3f}, max |diff| {(a - b).abs().max():.4f}"
    )
    if pcc < PCC_MIN:
        problems.append(f"prefill logits PCC {pcc:.5f} below {PCC_MIN}")
    if kl > KL_MAX:
        problems.append(f"prefill logits KL {kl:.4f} above {KL_MAX}")
    if margin >= DECISIVE_MARGIN and top1_a != top1_b:
        problems.append(f"decisive top-1 flipped ({top1_a} -> {top1_b}) at margin {margin:.3f}")

    # K / V blocks: arm B's chunk-0 range (legacy ops in both arms when A is unchunked) and its chunked-SDPA range
    first_chunk_end = schedule_b[0].end
    total = schedule_b[-1].end
    for layer in COMPARED_LAYERS:
        for name, ref, other in (("K", kv_a[layer][0], kv_b[layer][0]), ("V", kv_a[layer][1], kv_b[layer][1])):
            tag = f"layer {layer} {name}"
            head, head_eq = _compare_kv(tag, ref, other, 0, first_chunk_end)
            tail, tail_eq = _compare_kv(tag, ref, other, first_chunk_end, total)
            if head < KV_PCC_MIN:
                problems.append(f"{tag}: chunk-0 blocks PCC {head:.5f} below {KV_PCC_MIN}")
            if tail < KV_PCC_MIN:
                problems.append(f"{tag}: chunked blocks PCC {tail:.5f} below {KV_PCC_MIN} (RoPE offset / fill?)")
            if head_eq < KV_BIT_EQUAL_MIN:
                problems.append(f"{tag}: chunk-0 blocks bit-equal fraction {head_eq:.4f} below {KV_BIT_EQUAL_MIN}")
            if tail_eq < KV_BIT_EQUAL_MIN:
                problems.append(f"{tag}: chunked blocks bit-equal fraction {tail_eq:.4f} below {KV_BIT_EQUAL_MIN}")

    # One teacher-forced decode step over each cache (the same token at the same position)
    dec_a = _decode_step(generator, tt_kv_cache, pt_a, top1_a, PROMPT_TOKENS)[:, :vocab]
    dec_b = _decode_step(generator, tt_kv_cache, pt_b, top1_a, PROMPT_TOKENS)[:, :vocab]
    dec_pcc = _pcc(dec_a, dec_b).item()
    dec_kl = _kl(dec_a, dec_b).item()
    dec_top2 = dec_a.topk(2, dim=-1).values[0]
    dec_margin = (dec_top2[0] - dec_top2[1]).item()
    logger.info(
        f"[decode step 1] PCC {dec_pcc:.6f}, KL {dec_kl:.5f}, top-1 {int(dec_a.argmax(-1))} vs {int(dec_b.argmax(-1))}, "
        f"arm-A margin {dec_margin:.3f}"
    )
    if dec_pcc < DECODE_PCC_MIN:
        problems.append(f"decode logits PCC {dec_pcc:.5f} below {DECODE_PCC_MIN}")
    if dec_margin >= DECISIVE_MARGIN and int(dec_a.argmax(-1)) != int(dec_b.argmax(-1)):
        problems.append("decisive decode top-1 flipped over the chunked cache")
    assert not problems, "; ".join(problems)
