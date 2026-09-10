# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Teacher-forced whole-model accuracy of the TT Solar-Open-100B against the bf16 HF reference (design 5.7 / 7.2-5).

The reference is generated OFF the device by ``gen_reference.py`` (the 100B bf16 model on the CPU needs ~205 GB of
host RAM, so it never shares a process - or the box - with a whole-model TT load): per prompt, the chat-templated
token ids (fixed date), the greedy 64-token continuation and the per-step logits. This test replays exactly those
token ids through the TT model - prefill the prompt, then decode step by step feeding the REFERENCE token (not the
device's own) - and compares the device logits of every step with the reference:

  top-1 agreement       argmax(TT) == reference token (also restricted to the "decisive" steps whose reference
                        top-1 margin is >= 0.5 logit, i.e. above the bf16 quantisation of the logits themselves)
  top-5 overlap         |top-5(TT) & top-5(ref)| / 5
  PCC of the top-64     device logits gathered at the reference's top-64 indices vs the reference values
  full-vocab PCC, KL    KL(ref || TT) over the 196608-way softmax in fp32

plus the device's own free-running greedy continuation vs the reference continuation (first divergence position). The
b32 case puts the prompts in all 32 slots (prompt i in slots i, i+4, ...) so the batched union-of-experts decode
path is measured too and the copies of one prompt can be checked against each other. The packed32 case (phase 3d /
A3) is b32 with the 32 prompts prefilled as ONE 32 x 128 packed pass through the driver ``tt/model.py:
prefill_forward_text_batched`` (``PACKED_GATE_OPTIONS``: 4096 tokens per pass, the demo's default) instead of 32
sequential per-user prefills -- the accuracy gate of the packed path, same floors (the ``b1`` / ``b32`` cases stay
sequential: ``ModelArgs.disable_batched_prefill`` is always True, only the driver packs).

Reading the numbers: a wrong FIRST step (prefill) points at the embedding / lm_head / chat template; a drift that
grows with the step index points at RoPE or the KV cache; a flat per-step noise with occasional near-tie flips is the
bfp8 / fp32-selection numerics floor. Thresholds are keyed by the expert dtype (bfp8 default, bfp4 A/B).

    timeout 10800 python models/demos/solar_open/tests/accuracy/gen_reference.py      # host only, once per checkpoint
    pytest models/demos/solar_open/tests/accuracy/test_teacher_forced.py -k "b1 and 1x8"
    pytest models/demos/solar_open/tests/accuracy/test_teacher_forced.py -k "b32 and 1x8"        # sequential prefill
    pytest models/demos/solar_open/tests/accuracy/test_teacher_forced.py -k "packed32 and 1x8"   # one packed 32 x 128 pass

``SOLAR_OPEN_TF_REFERENCE`` points at the reference file (default ``$TT_CACHE_PATH/teacher_forced_reference.pt``);
``SOLAR_OPEN_TF_REPORT_DIR`` (optional) receives a markdown report per case.
"""

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.solar_open.config import MoEOptions
from models.demos.solar_open.demo.text_demo import prepare_solar_open_generator_args
from models.demos.solar_open.tests.accuracy.gen_reference import default_reference_path, render_prompt_ids
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.solar_open.tt.model import prefill_forward_text_batched
from models.demos.solar_open.tt.model_config import BatchedPrefillOptions
from models.tt_transformers.tt.generator import Generator

# Thresholds (design D13: ~0.02 below the first measured values; recorded in README.md "Recorded baselines"), keyed by
# the routed-expert dtype (SOLAR_OPEN_EXPERT_DTYPE; the bfp4 variant is a documented A/B option, not the default).
# Measured 2026-09-07 on P150x8 (1x8, TP=8; bfp8 attention / lm_head, bf16 residual, fused fp32 router) with the
# 4-prompt x 64-step reference of gen_reference.py, cases b1 / b32:
#   bfp8 experts (default)
#     top-1 agreement            0.9297 / 0.9180 (per prompt 0.875-0.969 / 0.906-0.938; every flip takes the reference's
#                                2nd choice, 13 of 18 (b1) / 12 of 21 (b32) flipped steps are bf16 near ties < 0.5 logit)
#     top-1 on decisive steps    0.9779 / 0.9602 (226 steps with a reference margin >= 0.5; flipped margins <= 1.25)
#     top-5 overlap              0.9281 / 0.9227
#     top-64 logit PCC mean      0.98177 / 0.98222 (per-step min 0.81 / 0.83 at exact-tie steps)
#     full-vocab logit PCC mean  0.99215 / 0.99219, no growth of the error with the step index
#     KL(ref || TT) mean         0.0347 / 0.0324 (max 0.44 at the KO capital prompt's first token)
#     slot copies (b32)          logit PCC min 0.99995, 1792/1792 identical top-1 tokens
#   bfp4 experts (SOLAR_OPEN_EXPERT_DTYPE=bfp4)
#     top-1 agreement            0.9062 / 0.8945 (per prompt 0.859-0.938 / 0.859-0.922); decisive 0.9425 / 0.9425
#     top-5 overlap              0.8977 / 0.8906; top-64 PCC 0.97061 / 0.97154; full-vocab PCC 0.98747 / 0.98755
#     KL(ref || TT) mean         0.0694 / 0.0681 (max 2.87: bfp4 puts <|content|> above <|think|> at step 0 of all
#                                4 prompts, also against 2.0-3.75-logit reference margins - it answers directly under
#                                reasoning_effort=low where the bf16 model re-opens a think block); slot copies 0.99994
# A real defect looks very different: a wrong first token on every prompt with bfp8 (embedding / lm_head / template),
# a per-step PCC that decays with the position (RoPE / KV) or KL >> 0.1 everywhere; the floors keep that margin.
THRESHOLDS = {
    "bfp8": {
        "top1": 0.90,  # aggregate over all prompts x steps
        "top1_per_prompt": 0.85,  # design 5.7 value; measured min 0.875 (KO capital prompt, 9 near-tie steps of 64)
        "top1_decisive": 0.94,
        "top5": 0.90,
        "pcc_top64": 0.96,
        "pcc_full": 0.97,
        "kl_max_mean": 0.06,
    },
    "bfp4": {
        "top1": 0.87,
        "top1_per_prompt": 0.84,
        "top1_decisive": 0.92,
        "top5": 0.87,
        "pcc_top64": 0.95,
        "pcc_full": 0.96,
        "kl_max_mean": 0.10,
    },
}
MIN_REPLICA_PCC = 0.97  # copies of one prompt in other slots vs its first slot (b32); a leak collapses this below 0.3
# packed32: the pass shape of the packed arm, pinned here (not the env): all 32 slots in ONE 4096-token pass = one MoE
# chunk, one hot / cold plan for the whole set (the demo's default since phase 3d / A3; P2 2026-09-09 measured this
# pass at 0.9336 / 0.9690 / 0.9180 / 0.98195 / 0.99266 / KL 0.03039, slot copies 1792 / 1792).
PACKED_GATE_OPTIONS = BatchedPrefillOptions(enabled=True, tokens_per_pass=4096, max_seq_len=128)
DECISIVE_MARGIN = 0.5  # reference top-1 margin (logits) above which a top-1 flip is not a bf16 near tie


def _clear_kv_caches(models):
    for m in models:
        for layer in m.layers:
            k_cache, v_cache = layer.self_attn.layer_past
            ttnn.mul(k_cache, 0, output_tensor=k_cache)
            ttnn.mul(v_cache, 0, output_tensor=v_cache)


def _prefill(generator, models, tt_kv_cache, page_table, prompt_ids_per_slot, packed=False):
    """Clear the KV cache and prefill every slot with its token ids (eager prefill). Returns (logits [B, V] fp32, pos).

    ``packed``: ONE packed pass of all slots through the driver (``PACKED_GATE_OPTIONS``, asserted); otherwise the plain
    Generator, which prefills per user (``disable_batched_prefill`` is always True).
    """
    batch = len(prompt_ids_per_slot)
    _clear_kv_caches(models)
    generator.prev_page_table = None
    lens = [len(ids) for ids in prompt_ids_per_slot]
    tokens = torch.zeros(batch, max(lens), dtype=torch.long)
    for b, ids in enumerate(prompt_ids_per_slot):
        tokens[b, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    if packed:
        logits = prefill_forward_text_batched(
            generator,
            tokens,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=lens,
            enable_trace=False,
            options=PACKED_GATE_OPTIONS,
        )
        log = generator.batched_prefill_pass_log
        assert (
            len(log) == 1
            and log[0].packed
            and not log[0].traced
            and len(log[0].users) == batch
            and log[0].seq_len == 128
        ), f"expected ONE packed {batch} x 128 pass, got {log}"
    else:
        logits = generator.prefill_forward_text(
            tokens, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=lens, enable_trace=False
        )
    return logits.reshape(batch, -1).float(), torch.tensor(lens)


def _decode_step(generator, tt_kv_cache, page_table, tokens, current_pos):
    logits, _ = generator.decode_forward(
        tokens, current_pos, enable_trace=True, page_table=page_table, kv_cache=tt_kv_cache, sampling_params=None
    )
    return logits.reshape(tokens.shape[0], -1)


def _teacher_forced_logits(generator, models, tt_kv_cache, page_table, prompt_ids_per_slot, forced, packed=False):
    """Prefill, then feed ``forced[b, t - 1]`` at decode step t. Returns (logits [B, T, V] bf16, step times)."""
    batch, num_steps = forced.shape
    logits0, current_pos = _prefill(generator, models, tt_kv_cache, page_table, prompt_ids_per_slot, packed)
    vocab = logits0.shape[-1]
    out = torch.empty(batch, num_steps, vocab, dtype=torch.bfloat16)  # the device emits bf16 logits: lossless
    out[:, 0] = logits0.to(torch.bfloat16)
    times = []
    for t in range(1, num_steps):
        t0 = time.perf_counter()
        logits = _decode_step(generator, tt_kv_cache, page_table, forced[:, t - 1].clone(), current_pos)
        times.append(time.perf_counter() - t0)
        out[:, t] = logits[:, :vocab].to(torch.bfloat16)
        current_pos += 1
    return out, times


def _greedy_tokens(generator, models, tt_kv_cache, page_table, prompt_ids_per_slot, num_steps, packed=False):
    """Free-running greedy continuation (host argmax of the device logits). Returns tokens [B, T]."""
    batch = len(prompt_ids_per_slot)
    logits0, current_pos = _prefill(generator, models, tt_kv_cache, page_table, prompt_ids_per_slot, packed)
    vocab = logits0.shape[-1]
    tokens = torch.empty(batch, num_steps, dtype=torch.long)
    tokens[:, 0] = logits0.argmax(-1)
    for t in range(1, num_steps):
        logits = _decode_step(generator, tt_kv_cache, page_table, tokens[:, t - 1].clone(), current_pos)
        tokens[:, t] = logits[:, :vocab].float().argmax(-1)
        current_pos += 1
    return tokens


def _pcc_rows(a, b):
    a_c = a - a.mean(dim=-1, keepdim=True)
    b_c = b - b.mean(dim=-1, keepdim=True)
    return (a_c * b_c).sum(-1) / (a_c.norm(dim=-1) * b_c.norm(dim=-1) + 1e-12)


def compare_with_reference(tt_logits, ref):
    """Per-step metrics of device logits [T, V] (any float dtype) against one reference prompt entry."""
    tt = tt_logits.float()
    ref_logits = ref["logits"].float()
    ref_top_idx, ref_top_val = ref["top_indices"], ref["top_values"]
    ref_tokens = ref["gen_ids"]
    tt_top1 = tt.argmax(-1)
    top5_tt = tt.topk(5, dim=-1).indices
    top5_ref = ref_top_idx[:, :5]
    metrics = {
        "top1": tt_top1 == ref_tokens,
        "top5_overlap": (top5_tt.unsqueeze(-1) == top5_ref.unsqueeze(-2)).any(-1).float().mean(-1),
        "pcc_top64": _pcc_rows(tt.gather(-1, ref_top_idx), ref_top_val),
        "pcc_full": _pcc_rows(tt, ref_logits),
        "kl": (torch.softmax(ref_logits, -1) * (torch.log_softmax(ref_logits, -1) - torch.log_softmax(tt, -1))).sum(-1),
        "ref_margin": ref_top_val[:, 0] - ref_top_val[:, 1],
        "tt_top1": tt_top1,
    }
    tt_top2 = tt.topk(2, dim=-1).values
    metrics["tt_margin"] = tt_top2[:, 0] - tt_top2[:, 1]
    # Reference rank of the device's top-1 token (0 = agreement; > 63 = outside the stored top-64)
    hit = ref_top_idx == tt_top1.unsqueeze(-1)
    metrics["tt_top1_ref_rank"] = torch.where(hit.any(-1), hit.float().argmax(-1), torch.full_like(tt_top1, 64))
    return metrics


def summarize(name, m, first_stop):
    steps = m["top1"].numel()
    decisive = m["ref_margin"] >= DECISIVE_MARGIN
    disagree = (~m["top1"]).nonzero().flatten().tolist()
    upto = slice(0, first_stop + 1) if first_stop >= 0 else slice(0, steps)
    half = steps // 2
    logger.info(
        f"[{name}] top-1 agreement {m['top1'].float().mean():.4f} ({int(m['top1'].sum())}/{steps}; decisive steps "
        f"{int((m['top1'] & decisive).sum())}/{int(decisive.sum())}; first/second half "
        f"{m['top1'][:half].float().mean():.3f} / {m['top1'][half:].float().mean():.3f}"
        + (
            f"; up to the stop token at step {first_stop}: {m['top1'][upto].float().mean():.4f}"
            if first_stop >= 0
            else ""
        )
        + f"), top-5 overlap {m['top5_overlap'].mean():.4f}, top-64 PCC mean {m['pcc_top64'].mean():.5f} / min "
        f"{m['pcc_top64'].min():.5f}, full-vocab PCC mean {m['pcc_full'].mean():.5f} / min {m['pcc_full'].min():.5f} "
        f"(steps 0-7 {m['pcc_full'][:8].mean():.5f}, last 8 {m['pcc_full'][-8:].mean():.5f}), KL mean "
        f"{m['kl'].mean():.5f} / max {m['kl'].max():.5f}; disagreeing steps {disagree[:16]}"
        + (
            " with ref margin "
            + ", ".join(f"{m['ref_margin'][i]:.3f}" for i in disagree[:16])
            + " and TT-top-1 reference rank "
            + ", ".join(str(int(m["tt_top1_ref_rank"][i])) for i in disagree[:16])
            if disagree
            else ""
        )
    )
    return {
        "steps": steps,
        "top1": m["top1"].float().mean().item(),
        "top1_decisive": (m["top1"] & decisive).sum().item() / max(1, int(decisive.sum())),
        "n_decisive": int(decisive.sum()),
        "top1_upto_stop": m["top1"][upto].float().mean().item(),
        "top5": m["top5_overlap"].mean().item(),
        "pcc_top64_mean": m["pcc_top64"].mean().item(),
        "pcc_top64_min": m["pcc_top64"].min().item(),
        "pcc_full_mean": m["pcc_full"].mean().item(),
        "pcc_full_min": m["pcc_full"].min().item(),
        "pcc_full_first8": m["pcc_full"][:8].mean().item(),
        "pcc_full_last8": m["pcc_full"][-8:].mean().item(),
        "kl_mean": m["kl"].mean().item(),
        "kl_max": m["kl"].max().item(),
        "disagree": disagree,
    }


def _first_divergence(a, b):
    diff = (a != b).nonzero().flatten()
    return int(diff[0]) if diff.numel() else int(a.numel())


def _load_reference():
    path = default_reference_path()
    if not path.is_file():
        pytest.skip(
            f"teacher-forced reference {path} missing: generate it on the host (no device process running) with "
            "`timeout 10800 python models/demos/solar_open/tests/accuracy/gen_reference.py --out <path>` and point "
            "SOLAR_OPEN_TF_REFERENCE at it"
        )
    ref = torch.load(path, weights_only=False)
    assert ref.get("format") == 1, f"unknown reference format {ref.get('format')}"
    logger.info(f"reference {path}: {ref['meta']}")
    return ref


def _write_report(report_dir, case, ref, tokenizer, rows, greedy, per_slot, summaries, decode_ms):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    meta = ref["meta"]
    lines = [
        f"# Teacher-forced accuracy, case {case}",
        "",
        f"Reference: {meta['model_path']} bf16 CPU (transformers {meta['transformers']}, eager attention / experts), "
        f"reasoning_effort={meta['reasoning_effort']}, default_system_prompt={meta['default_system_prompt']}, "
        f"date {meta['date_string']}, {meta['num_new_tokens']} greedy tokens per prompt, created {meta['created']}.",
        f"Device: teacher-forced decode over the reference tokens; greedy decode {decode_ms:.1f} ms/step.",
        "",
        "| prompt | slots | top-1 | top-1 decisive | top-5 | top-64 PCC mean/min | full PCC mean/min | KL mean/max | "
        "TT greedy diverges at | TF first flip |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += rows
    for p, entry in enumerate(ref["prompts"]):
        s = summaries[p]
        ref_text = tokenizer.decode(entry["gen_ids"].tolist(), skip_special_tokens=False)
        tt_text = tokenizer.decode(greedy[p].tolist(), skip_special_tokens=False)
        stop = entry["first_stop_pos"]
        lines += [
            "",
            f"## Prompt {entry['index']}: {entry['text']}",
            "",
            f"- prompt tokens: {len(entry['prompt_ids'])}; reference stop token at step {stop if stop >= 0 else 'none'}",
            f"- HF greedy ({len(entry['gen_ids'])} tokens): `{ref_text!r}`",
            f"- TT greedy ({greedy[p].numel()} tokens): `{tt_text!r}`",
            f"- first divergence of the TT greedy continuation: step {_first_divergence(greedy[p], entry['gen_ids'])} "
            f"of {entry['gen_ids'].numel()}; teacher-forced top-1 flips at steps {s['disagree']}",
            f"- per-slot top-1 agreement: {per_slot[p]}",
        ]
    (report_dir / f"teacher_forced_{case}.md").write_text("\n".join(lines) + "\n")
    logger.info(f"report written to {report_dir / f'teacher_forced_{case}.md'}")


def _save_device_outputs(report_dir, case, ref, tt_logits, greedy, all_metrics, decode_ms):
    """Persist the device logits (bf16, first slot of each prompt) and per-step metrics for offline A/B comparisons."""
    path = Path(report_dir) / f"teacher_forced_{case}_device.pt"
    torch.save(
        {
            "format": 1,
            "case": case,
            "reference_meta": ref["meta"],
            "prompt_indices": [int(p["index"]) for p in ref["prompts"]],
            "tt_logits": torch.stack(tt_logits),  # [P, T, V] bf16
            "tt_greedy": torch.stack(greedy),  # [P, T]
            "metrics": [{k: v.clone() for k, v in m.items()} for m in all_metrics],
            "decode_ms_per_step": decode_ms,
            "expert_dtype": MoEOptions.from_env().expert_dtype_str,
        },
        path,
    )
    logger.info(f"device logits / metrics written to {path} ({path.stat().st_size / 2**20:.0f} MiB)")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "batch_size, max_seq_len, packed_prefill",
    [(1, 4 * 1024, False), (32, 8 * 1024, False), (32, 8 * 1024, True)],
    ids=["b1", "b32", "packed32"],
)
@parametrize_mesh_with_fabric([(1, 8)])
def test_teacher_forced(mesh_device, device_params, batch_size, max_seq_len, packed_prefill, state_dict):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] != 1 or mesh_shape[1] < 8:
        pytest.skip(f"validated on 1x8 meshes (TP=8), got {mesh_shape}")
    ref = _load_reference()
    meta = ref["meta"]
    prompts = ref["prompts"]
    num_prompts = len(prompts)
    num_steps = int(prompts[0]["gen_ids"].numel())
    assert all(int(p["gen_ids"].numel()) == num_steps for p in prompts)

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    block_size = 64
    page_params = {
        "page_block_size": block_size,
        "page_max_num_blocks_per_dp": batch_size * (max_seq_len // block_size),
    }
    model_args, models, page_table, tt_kv_cache, tokenizer, _processor, _paged_cfg = prepare_solar_open_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=setup["mesh_config"],
        state_dict=state_dict,
        users_row_sharded=False,
    )
    generator = Generator(models, model_args, mesh_device, processor=None, tokenizer=tokenizer)
    assert model_args[0].vocab_size == meta["vocab_size"]
    assert all(a.disable_batched_prefill for a in model_args), "a plain Generator call must stay per-user (phase 3d)"
    case_name = "packed32" if packed_prefill else f"b{batch_size}"
    logger.info(
        f"case {case_name}: prefill "
        + (
            f"as ONE packed {batch_size} x 128 pass through the driver ({PACKED_GATE_OPTIONS.describe()})"
            if packed_prefill
            else "sequential per user through the plain Generator"
        )
        + f"; ModelArgs policy {model_args[0].batched_prefill.describe()}"
    )

    # The device run must tokenize exactly like the reference: same tokenizer, same template kwargs, same fixed date.
    for entry in prompts:
        kw = dict(
            reasoning_effort=meta["reasoning_effort"],
            default_system_prompt=meta["default_system_prompt"],
            strftime_now=lambda fmt: meta["date_string"],
        )
        via_model_args = model_args[0].encode_prompt(entry["text"], **kw)
        via_helper = render_prompt_ids(
            tokenizer, entry["text"], meta["reasoning_effort"], meta["default_system_prompt"], meta["date_string"]
        )
        assert via_model_args == entry["prompt_ids"].tolist() == via_helper, (
            f"prompt {entry['index']} tokenizes differently on the device host than in the reference "
            f"({len(via_model_args)} vs {entry['prompt_ids'].numel()} tokens)"
        )

    # Slot layout: b1 runs the prompts one after another; b32 tiles them over the 32 slots (prompt p in slots p, p+4, ..).
    if batch_size == 1:
        layouts = [[p] for p in range(num_prompts)]
    else:
        layouts = [[s % num_prompts for s in range(batch_size)]]

    tt_logits = [None] * num_prompts  # first slot of each prompt: [T, V] bf16
    per_slot_top1 = [[] for _ in range(num_prompts)]
    replica_notes = []
    greedy = [None] * num_prompts
    step_times = []
    args = (generator, models, tt_kv_cache, page_table)
    for layout in layouts:
        slot_ids = [prompts[p]["prompt_ids"].tolist() for p in layout]
        forced = torch.stack([prompts[p]["gen_ids"] for p in layout])  # [B, T]
        logits, times = _teacher_forced_logits(*args, slot_ids, forced, packed=packed_prefill)
        step_times += times
        greedy_tokens = _greedy_tokens(*args, slot_ids, num_steps, packed=packed_prefill)
        for slot, p in enumerate(layout):
            if tt_logits[p] is None:
                tt_logits[p] = logits[slot]
                greedy[p] = greedy_tokens[slot]
            m = compare_with_reference(logits[slot], prompts[p])
            per_slot_top1[p].append(round(m["top1"].float().mean().item(), 4))
            if slot != layout.index(p):  # a copy of the prompt in another slot: compare with the first copy
                pcc = _pcc_rows(logits[slot].float(), tt_logits[p].float())
                same_top1 = int((logits[slot].float().argmax(-1) == tt_logits[p].float().argmax(-1)).sum())
                same_greedy = int((greedy_tokens[slot] == greedy[p]).sum())
                replica_notes.append((p, slot, pcc.min().item(), same_top1, same_greedy))
    if replica_notes:
        worst = min(replica_notes, key=lambda r: r[2])
        logger.info(
            f"slot copies of the same prompt vs its first slot: logit PCC min {worst[2]:.5f} (prompt {worst[0]} slot "
            f"{worst[1]}); teacher-forced top-1 identical in {sum(r[3] for r in replica_notes)} of "
            f"{len(replica_notes) * num_steps} (slot, step) pairs; greedy tokens identical in "
            f"{sum(r[4] for r in replica_notes)} of {len(replica_notes) * num_steps}"
        )
    steady = step_times[2:] if len(step_times) > 2 else step_times
    decode_ms = 1000 * sum(steady) / max(1, len(steady))
    logger.info(
        f"teacher-forced decode (batch {batch_size}, {case_name}): {decode_ms:.1f} ms/step over {len(steady)} steps"
    )

    # Per-prompt and aggregate metrics against the reference.
    summaries, rows, all_metrics = [], [], []
    for p, entry in enumerate(prompts):
        m = compare_with_reference(tt_logits[p], entry)
        all_metrics.append(m)
        name = f"prompt {entry['index']} {entry['text']!r}"
        s = summarize(name, m, entry["first_stop_pos"])
        summaries.append(s)
        div = _first_divergence(greedy[p], entry["gen_ids"])
        first_flip = s["disagree"][0] if s["disagree"] else num_steps
        logger.info(
            f"[{name}] HF greedy : {tokenizer.decode(entry['gen_ids'].tolist(), skip_special_tokens=False)!r}\n"
            f"[{name}] TT greedy : {tokenizer.decode(greedy[p].tolist(), skip_special_tokens=False)!r}\n"
            f"[{name}] TT greedy diverges from the reference at step {div} of {num_steps} (teacher-forced first "
            f"top-1 flip at step {first_flip}); reference stop token at step {entry['first_stop_pos']}"
        )
        rows.append(
            f"| {entry['index']} {entry['text']} | {len(per_slot_top1[p])} | {s['top1']:.4f} | "
            f"{s['top1_decisive']:.4f} ({s['n_decisive']} steps) | {s['top5']:.4f} | {s['pcc_top64_mean']:.5f} / "
            f"{s['pcc_top64_min']:.5f} | {s['pcc_full_mean']:.5f} / {s['pcc_full_min']:.5f} | {s['kl_mean']:.4f} / "
            f"{s['kl_max']:.4f} | {div} | {first_flip} |"
        )
    top1_all = torch.cat([m["top1"] for m in all_metrics])
    decisive_all = torch.cat([m["ref_margin"] >= DECISIVE_MARGIN for m in all_metrics])
    top5_all = torch.cat([m["top5_overlap"] for m in all_metrics])
    pcc64_all = torch.cat([m["pcc_top64"] for m in all_metrics])
    pccf_all = torch.cat([m["pcc_full"] for m in all_metrics])
    kl_all = torch.cat([m["kl"] for m in all_metrics])
    agg = {
        "top1": top1_all.float().mean().item(),
        "top1_decisive": (top1_all & decisive_all).sum().item() / max(1, int(decisive_all.sum())),
        "top5": top5_all.mean().item(),
        "pcc_top64_mean": pcc64_all.mean().item(),
        "pcc_top64_min": pcc64_all.min().item(),
        "pcc_full_mean": pccf_all.mean().item(),
        "pcc_full_min": pccf_all.min().item(),
        "kl_mean": kl_all.mean().item(),
        "kl_max": kl_all.max().item(),
    }
    logger.info(
        f"[all {num_prompts} prompts x {num_steps} steps, batch {batch_size}, {case_name}] top-1 agreement "
        f"{agg['top1']:.4f} "
        f"({int(top1_all.sum())}/{top1_all.numel()}; decisive {agg['top1_decisive']:.4f} of {int(decisive_all.sum())}), "
        f"top-5 overlap {agg['top5']:.4f}, top-64 PCC mean {agg['pcc_top64_mean']:.5f} / min {agg['pcc_top64_min']:.5f}, "
        f"full-vocab PCC mean {agg['pcc_full_mean']:.5f} / min {agg['pcc_full_min']:.5f}, KL mean {agg['kl_mean']:.5f} / "
        f"max {agg['kl_max']:.5f}"
    )
    rows.append(
        f"| all | {sum(len(x) for x in per_slot_top1)} | {agg['top1']:.4f} | {agg['top1_decisive']:.4f} "
        f"({int(decisive_all.sum())} steps) | {agg['top5']:.4f} | {agg['pcc_top64_mean']:.5f} / {agg['pcc_top64_min']:.5f} "
        f"| {agg['pcc_full_mean']:.5f} / {agg['pcc_full_min']:.5f} | {agg['kl_mean']:.4f} / {agg['kl_max']:.4f} | - | - |"
    )
    report_dir = os.getenv("SOLAR_OPEN_TF_REPORT_DIR")
    if report_dir:
        _write_report(report_dir, case_name, ref, tokenizer, rows, greedy, per_slot_top1, summaries, decode_ms)
        _save_device_outputs(report_dir, case_name, ref, tt_logits, greedy, all_metrics, decode_ms)

    expert_dtype = MoEOptions.from_env().expert_dtype_str
    th = THRESHOLDS[expert_dtype]
    logger.info(f"thresholds for {expert_dtype} experts: {th}")
    problems = []
    if agg["top1"] < th["top1"]:
        problems.append(f"top-1 agreement {agg['top1']:.4f} < {th['top1']}")
    for entry, s in zip(prompts, summaries):
        if s["top1"] < th["top1_per_prompt"]:
            problems.append(
                f"prompt {entry['index']}: top-1 agreement {s['top1']:.4f} < {th['top1_per_prompt']} "
                f"(flipped steps {s['disagree']})"
            )
    if agg["top1_decisive"] < th["top1_decisive"]:
        problems.append(f"top-1 agreement on decisive steps {agg['top1_decisive']:.4f} < {th['top1_decisive']}")
    if agg["top5"] < th["top5"]:
        problems.append(f"top-5 overlap {agg['top5']:.4f} < {th['top5']}")
    if agg["pcc_top64_mean"] < th["pcc_top64"]:
        problems.append(f"mean top-64 logit PCC {agg['pcc_top64_mean']:.4f} < {th['pcc_top64']}")
    if agg["pcc_full_mean"] < th["pcc_full"]:
        problems.append(f"mean full-vocab logit PCC {agg['pcc_full_mean']:.4f} < {th['pcc_full']}")
    if agg["kl_mean"] > th["kl_max_mean"]:
        problems.append(f"mean KL(ref || TT) {agg['kl_mean']:.4f} > {th['kl_max_mean']}")
    if replica_notes and min(r[2] for r in replica_notes) < MIN_REPLICA_PCC:
        problems.append(f"slot copies of one prompt disagree: logit PCC min {min(r[2] for r in replica_notes):.4f}")
    assert not problems, "Teacher-forced accuracy below threshold:\n" + "\n".join(problems)
