# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Cross-user isolation check for multi-user decode on single-row meshes (batch 32, users not row-sharded).

Every user's computation must be independent of which slot it occupies and of what the other users
are doing. The two multi-user mechanisms this checks are exactly the ones that fail *silently* when
they go wrong: the per-user Q/K/V shard placement in attention decode (RoPE / KV update / SDPA all
read "their" user from a core they compute themselves) and the union-of-experts MoE path (an
expert selected by another user must contribute exactly zero to this user).

Greedy token sequences are NOT a usable signal here: with bfloat8 activations the model routinely has
near-tie top-1 candidates, so any last-bit numerical difference (run-to-run CCL reduction order, traced
vs eager prefill, ...) flips a token and the sequences diverge. Instead every run is teacher-forced
with the same tokens and the per-step *logits* of each prompt are compared across slots:

  run A1 / A2: the 32 prompts in slot order, twice (same-slot baseline for numerical noise),
  run B:       the same prompts rotated by a few slots,
  run C:       one prompt alone in a middle slot with every other slot holding a filler prompt.

A prompt's logits at every step must be as close across slots as they are across two identical runs
(PCC ~1, top-1 agreement at the baseline rate). Requires real weights (HF_MODEL); runs on a 1x8 mesh.

Two prefill arms, one case each (phase 3d / A3):

  ``b32``       sequential per-user prefills through the plain ``Generator`` (``ModelArgs.disable_batched_prefill`` is
                always True since phase 3d, so this is what the regression harness, vLLM and every plain Generator
                caller get); the phase-2 floors: a slot change is indistinguishable from a same-slot repeat.
  ``packed32``  every ``_prefill`` is ONE 32 x 128 packed pass through the driver ``tt/model.py:
                prefill_forward_text_batched`` with ``PACKED_GATE_OPTIONS`` (4096 tokens per pass = the demo's default:
                one 4096-token MoE chunk of four 1024-token splits) -- the slot-independence gate of the packed path
                with its own floors (``PACKED_*`` below). Phase 3a failed the slot check (rotated slots PCC min
                0.91-0.94, 34-38 / 768 top-1 flips; fillers of one prompt in different splits 24-33 / 720): the
                expert-sorted MoE planned its hot / cold experts per split from the 8 users of that split. Phase 3c
                plans a packed pass once per chunk (``experts/prefill.py: _sorted_moe_chunk_plan``, marked by
                ``Model.ttnn_prefill_forward``; the test asserts the plan really ran), so every slot of the pass sees
                the same hot / cold sets: same slots and fillers are exact (0 flips), the rotation costs a few near-tie
                flips (P2 2026-09-09: 9 / 768, PCC min 0.979 on that day's ids; A3 on the pinned 09-08 ids: 18 / 768 at
                margins <= 0.5, PCC min 0.989). What remains by construction is the co-batch dependence:
                a user's logits depend on the SET of users in its pass (hot / cold is decided on all of them), which the
                lone-prompt check exposes (prompt 0 among 31 copies of a filler: P2 2 / 24 flips at margins <= 0.875,
                PCC min 0.973). The packed floors are those measurements with margin (design D13), not the sequential
                exactness; the co-batch dependence is accepted and documented (README "Known limitations").

    pytest models/demos/solar_open/tests/test_multi_user_consistency.py -k "1x8 and b32"        # sequential arm
    pytest models/demos/solar_open/tests/test_multi_user_consistency.py -k "1x8 and packed32"   # packed arm
    pytest models/demos/solar_open/tests/test_multi_user_consistency.py -k 1x8                  # both

The prompts are chat-templated with the template date pinned (``pinned_template_date``: 2026-09-08;
``SOLAR_OPEN_TEMPLATE_DATE`` overrides), so the greedy continuations and the flip counts reproduce across days.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams
from models.demos.solar_open.demo.text_demo import prepare_solar_open_generator_args
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.solar_open.tt import packed_prefill
from models.demos.solar_open.tt.experts import prefill as experts_prefill
from models.demos.solar_open.tt.model import prefill_forward_text_batched
from models.demos.solar_open.tt.model_config import BatchedPrefillOptions
from models.tt_transformers.demo.simple_text_demo import load_inputs
from models.tt_transformers.tt.common import preprocess_inputs_prefill
from models.tt_transformers.tt.generator import Generator

# 16 Korean + 16 English short-answer prompts: the same set the batch-32 demo case uses.
PROMPTS_FILE = "models/demos/solar_open/demo/sample_prompts/input_data_questions_ko_en_prefill_128.json"
FILLER_PROMPT = "Write one sentence about the weather."
# The packed arm's pass shape, pinned here (not the env): the 32 users of a run in ONE 4096-token pass = one MoE chunk,
# one hot / cold plan for the whole set (the demo's default since phase 3d / A3). With 1024-token passes the rotation
# would move users between passes and the co-batch dependence would fail the rotated-slot floor by design.
PACKED_GATE_OPTIONS = BatchedPrefillOptions(enabled=True, tokens_per_pass=4096, max_seq_len=128)
# Packed-arm floors (phase 3d / A3, derived from the P2 measurements of 2026-09-09 with margin; the sequential arm keeps
# the exact phase-2 floors in the test body):
# Rotated slots: the SET of users is unchanged, so the plan is identical, but the row order of the sorted MoE changes and
# the fp32 / bfp8 accumulation order with it -- last-bit noise that flips only near-tie steps. The count is input
# dependent: P2 (2026-09-09 ids) 9 of 768 at margins <= 0.375, PCC min 0.97903; A3 (the pinned 2026-09-08 ids) 18 of 768
# at margins <= 0.5, PCC min 0.98854 -> the floor is a count bound with margin plus a near-tie bound on EVERY flipped step.
PACKED_MAX_FLIPS_ROTATED = 32  # of 768 (prompt, step) pairs (~4 %)
PACKED_MAX_FLIP_MARGIN_ROTATED = 1.0  # every flipped step is a near tie (top-1 margin, logits); measured <= 0.5
PACKED_MIN_STEP_PCC_ROTATED = 0.97  # a real cross-user leak collapses per-step PCC below 0.7 (observed < 0.3)
PACKED_MAX_FLIPS_LONE = 3  # of 24; P2: 2 (the co-batch dependence of hot / cold)
PACKED_MAX_FLIP_MARGIN_LONE = 1.0  # every flipped step is a near tie (top-1 margin, logits); P2: <= 0.875
PACKED_MIN_STEP_PCC_LONE = 0.96  # P2: 0.97326
PACKED_MIN_STEP_PCC_SAME = 0.97  # same slots / fillers vs each other; P2: >= 0.99996 (0 flips required, exact)


def _clear_kv_caches(models):
    for m in models:
        for layer in m.layers:
            k_cache, v_cache = layer.self_attn.layer_past
            ttnn.mul(k_cache, 0, output_tensor=k_cache)
            ttnn.mul(v_cache, 0, output_tensor=v_cache)


def _prefill(
    generator, models, model_args, tt_kv_cache, page_table, tokenizer, prompts, num_tokens, max_seq_len, packed=False
):
    """Clear the KV cache and prefill all users. Returns (prefill logits [B, vocab], decoding positions).

    ``packed``: the driver with ``PACKED_GATE_OPTIONS`` (one eager 32 x 128 pass, asserted); otherwise the plain
    Generator, which prefills per user (``disable_batched_prefill`` is always True).
    """
    batch = len(prompts)
    _clear_kv_caches(models)
    generator.prev_page_table = None
    input_tokens, _encoded, decoding_pos, _prefill_lens = preprocess_inputs_prefill(
        prompts, tokenizer, model_args, instruct=False, max_generated_tokens=num_tokens, max_prefill_len=max_seq_len
    )
    input_tokens = torch.stack(input_tokens).view(batch, -1)
    if packed:
        logits = prefill_forward_text_batched(
            generator,
            input_tokens,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
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
        assert all(
            a.disable_batched_prefill for a in model_args
        ), "the sequential arm needs the per-user Generator path"
        logits = generator.prefill_forward_text(
            input_tokens, page_table=page_table, kv_cache=tt_kv_cache, prompt_lens=decoding_pos, enable_trace=False
        )
    return logits.reshape(batch, -1).float(), torch.tensor(decoding_pos)


def _generate_greedy(
    generator,
    models,
    model_args,
    tt_kv_cache,
    page_table,
    tokenizer,
    prompts,
    num_tokens,
    sampling,
    max_seq_len,
    packed=False,
):
    """On-device greedy generation (the demo path). Returns per-user token lists and decode step times."""
    batch = len(prompts)
    logits, current_pos = _prefill(
        generator, models, model_args, tt_kv_cache, page_table, tokenizer, prompts, num_tokens, max_seq_len, packed
    )
    out_tok = torch.argmax(logits, dim=-1)
    outputs = [[int(out_tok[b])] for b in range(batch)]
    step_times = []
    for _ in range(num_tokens - 1):
        t0 = time.perf_counter()
        out_tok, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=True,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=sampling,
        )
        step_times.append(time.perf_counter() - t0)
        current_pos += 1
        for b in range(batch):
            outputs[b].append(int(out_tok[b]))
    return outputs, step_times


def _teacher_forced_logits(
    generator,
    models,
    model_args,
    tt_kv_cache,
    page_table,
    tokenizer,
    prompts,
    forced,
    num_tokens,
    max_seq_len,
    packed=False,
):
    """Prefill, then feed `forced[b, t]` as user b's token at decode step t (host-side logits).

    Returns logits [B, num_tokens, vocab] (float16 on host): step 0 is the prefill output, step t the
    decode output after consuming forced[:, t - 1].
    """
    batch = len(prompts)
    logits0, current_pos = _prefill(
        generator, models, model_args, tt_kv_cache, page_table, tokenizer, prompts, num_tokens, max_seq_len, packed
    )
    vocab = logits0.shape[-1]
    all_logits = torch.empty(batch, num_tokens, vocab, dtype=torch.float16)
    all_logits[:, 0] = logits0.half()
    for t in range(1, num_tokens):
        tokens = forced[:, t - 1].clone()
        logits, _ = generator.decode_forward(
            tokens, current_pos, enable_trace=True, page_table=page_table, kv_cache=tt_kv_cache, sampling_params=None
        )
        all_logits[:, t] = logits.reshape(batch, -1)[:, :vocab].half()
        current_pos += 1
    return all_logits


def _compare(name, ref, other, ref_slots, other_slots):
    """Per-(prompt, step) logit PCC, top-1 agreement and the reference top-1 margin between two runs; slots map
    prompt -> slot in each run. Returns (pcc [P, T], agree [P, T], margin [P, T])."""
    pccs, agree, margins = [], [], []
    for p_ref, p_other in zip(ref_slots, other_slots):
        a = ref[p_ref].float()
        b = other[p_other].float()
        # A NaN / inf logit row makes every comparison below vacuous (PCC nan compares False against any floor), so
        # it is a failure of its own -- measured once on 2026-09-08 (C2 filler repeat: PCC nan, 53/744 flips).
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), (
            f"[{name}] non-finite logits: ref slot {p_ref} finite={bool(torch.isfinite(a).all())}, "
            f"other slot {p_other} finite={bool(torch.isfinite(b).all())}"
        )
        a_c = a - a.mean(dim=-1, keepdim=True)
        b_c = b - b.mean(dim=-1, keepdim=True)
        pcc = (a_c * b_c).sum(-1) / (a_c.norm(dim=-1) * b_c.norm(dim=-1) + 1e-12)
        pccs.append(pcc)
        agree.append(a.argmax(-1) == b.argmax(-1))
        top2 = a.topk(2, dim=-1).values
        margins.append(top2[:, 0] - top2[:, 1])
    pccs = torch.stack(pccs)  # [P, T]
    agree = torch.stack(agree)
    margins = torch.stack(margins)
    disagreeing_margins = margins[~agree]
    logger.info(
        f"[{name}] logit PCC min {pccs.min():.5f} mean {pccs.mean():.5f}; top-1 agreement "
        f"{agree.float().mean():.4f} ({int((~agree).sum())} of {agree.numel()} disagree"
        + (f", their top-1 margin <= {disagreeing_margins.max():.3f})" if disagreeing_margins.numel() else ")")
    )
    return pccs, agree, margins


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "batch_size, num_tokens, rotation, lone_slot, prefill_mode",
    [(32, 24, 5, 7, "sequential"), (32, 24, 5, 7, "packed")],
    ids=["b32", "packed32"],
)
@parametrize_mesh_with_fabric([(1, 8)])
def test_multi_user_isolation(
    mesh_device,
    device_params,
    batch_size,
    num_tokens,
    rotation,
    lone_slot,
    prefill_mode,
    state_dict,
    pinned_template_date,
):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] != 1 or mesh_shape[1] < 8:
        pytest.skip(f"multi-user single-row decode is validated on 1x8 meshes, got {mesh_shape}")

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    max_seq_len = 8 * 1024
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
    assert all(getattr(m, "sampling", None) is not None for m in models), "on-device sampling expected on 1x8"
    sampling = SamplingParams(
        temperature=[0.0] * batch_size,
        top_k=[1] * batch_size,
        top_p=[1.0] * batch_size,
        enable_log_probs=[False] * batch_size,
        num_logprobs=[0] * batch_size,
    )
    prompts, _ = load_inputs(PROMPTS_FILE, batch_size, instruct=False)
    prompts = list(prompts)[:batch_size]
    args = (generator, models, model_args, tt_kv_cache, page_table, tokenizer)
    packed_arm = prefill_mode == "packed"
    assert all(a.disable_batched_prefill for a in model_args), "a plain Generator call must stay per-user (phase 3d)"
    logger.info(
        f"prefill arm: {prefill_mode} -> "
        + (
            f"ONE packed {batch_size} x 128 pass per run through the driver ({PACKED_GATE_OPTIONS.describe()})"
            if packed_arm
            else "sequential per-user prefills through the plain Generator"
        )
        + f"; ModelArgs policy {model_args[0].batched_prefill.describe()}; sorted-MoE plan mode "
        f"{experts_prefill.SORTED_MOE_PLAN}"
    )
    experts_prefill.LAST_SORTED_MOE_PLAN.clear()

    # 1. Greedy generation (the demo path): sanity-check outputs and get a natural continuation per prompt
    #    to teacher-force below.
    greedy, times = _generate_greedy(*args, prompts, num_tokens, sampling, max_seq_len, packed=packed_arm)
    last_plan = dict(experts_prefill.LAST_SORTED_MOE_PLAN)
    logger.info(f"sorted-MoE plan of the last prefill split: {last_plan or None}")
    if (
        packed_arm
        and experts_prefill.SORTED_MOE_PLAN in ("auto", "chunk")
        and packed_prefill.packed_seq_numerics_level() < 2
    ):
        # the 32 x 128 pass is one 4096-token chunk of four sorted splits: it must have been planned per chunk (phase
        # 3c), otherwise the slot floors below measure the phase-3a per-split dependence again
        assert (
            last_plan.get("per_chunk") is True and last_plan.get("n_splits") == 4
        ), f"the packed pass was not planned per chunk: {last_plan}"
    steady = times[2:] if len(times) > 2 else times
    logger.info(
        f"greedy batch {batch_size}: decode {1000 * sum(steady) / max(1, len(steady)):.1f} ms/step steady-state "
        f"-> {batch_size / (sum(steady) / max(1, len(steady))):.0f} tok/s aggregate"
    )
    for u in range(batch_size):  # all users: the continuations are the teacher-forced inputs below
        logger.info(f"user {u} prompt: {prompts[u]!r}\n  -> {tokenizer.decode(greedy[u])!r}")
    forced = torch.tensor(greedy, dtype=torch.long)  # [B, T]

    # 2. Teacher-forced logits for the same inputs in different slot layouts.
    in_order = list(range(batch_size))
    lg_a1 = _teacher_forced_logits(*args, prompts, forced, num_tokens, max_seq_len, packed=packed_arm)
    lg_a2 = _teacher_forced_logits(*args, prompts, forced, num_tokens, max_seq_len, packed=packed_arm)

    rotated_slots = [(i + rotation) % batch_size for i in in_order]  # prompt i -> slot rotated_slots[i]
    prompts_b = [None] * batch_size
    forced_b = torch.empty_like(forced)
    for i in in_order:
        prompts_b[rotated_slots[i]] = prompts[i]
        forced_b[rotated_slots[i]] = forced[i]
    lg_b = _teacher_forced_logits(*args, prompts_b, forced_b, num_tokens, max_seq_len, packed=packed_arm)

    prompts_c = [FILLER_PROMPT] * batch_size
    prompts_c[lone_slot] = prompts[0]
    forced_c = forced[0].unsqueeze(0).expand(batch_size, -1).clone()  # every slot consumes prompt 0's continuation
    lg_c1 = _teacher_forced_logits(*args, prompts_c, forced_c, num_tokens, max_seq_len, packed=packed_arm)
    lg_c2 = _teacher_forced_logits(*args, prompts_c, forced_c, num_tokens, max_seq_len, packed=packed_arm)

    # 3. Compare. Identical repeats (A1 vs A2, C1 vs C2) measure the numerical noise floor of the device
    #    (the CCL reductions are not bit-reproducible run to run, and a last-bit change flips near-tie
    #    top-k expert choices, so even same-slot repeats show occasional logit PCC dips). A slot layout
    #    change must not look any different from that; a real mixing bug gives PCC ~0 for the users hit.
    pcc_base, agree_base, _m = _compare("A1 vs A2 (same slots, baseline)", lg_a1, lg_a2, in_order, in_order)
    pcc_rot, agree_rot, margin_rot = _compare("A1 vs B (rotated slots)", lg_a1, lg_b, in_order, rotated_slots)
    pcc_lone, agree_lone, margin_lone = _compare(
        "A1 vs C1 (prompt 0 alone in slot %d)" % lone_slot, lg_a1, lg_c1, [0], [lone_slot]
    )
    filler_slots = [s for s in range(batch_size) if s != lone_slot]
    pcc_fbase, agree_fbase, _m = _compare(
        "C1 vs C2 fillers (same slots, baseline)", lg_c1, lg_c2, filler_slots, filler_slots
    )
    pcc_fill, agree_fill, _m = _compare(
        "C1 fillers vs each other (same prompt, different slots)",
        lg_c1,
        lg_c1,
        [filler_slots[0]] * (len(filler_slots) - 1),
        filler_slots[1:],
    )

    if packed_arm:
        # Packed floors (phase 3d / A3; module constants): exact where the phase-3c per-chunk plan makes the pass slot
        # independent (same-set same-slot repeats, fillers of one prompt in different slots of ONE pass), a bounded
        # number of near-tie flips where the layout change or the changed SET of pass mates re-decides hot / cold.
        MIN_MEAN_PCC = 0.98
        problems = []
        for name, pcc, agree, margins, max_flips, min_step_pcc, max_margin in [
            ("same slots (A1 vs A2)", pcc_base, agree_base, None, 0, PACKED_MIN_STEP_PCC_SAME, None),
            (
                "rotated slots",
                pcc_rot,
                agree_rot,
                margin_rot,
                PACKED_MAX_FLIPS_ROTATED,
                PACKED_MIN_STEP_PCC_ROTATED,
                PACKED_MAX_FLIP_MARGIN_ROTATED,
            ),
            (
                "lone prompt",
                pcc_lone,
                agree_lone,
                margin_lone,
                PACKED_MAX_FLIPS_LONE,
                PACKED_MIN_STEP_PCC_LONE,
                PACKED_MAX_FLIP_MARGIN_LONE,
            ),
            ("filler same slots (C1 vs C2)", pcc_fbase, agree_fbase, None, 0, PACKED_MIN_STEP_PCC_SAME, None),
            ("filler slots", pcc_fill, agree_fill, None, 0, PACKED_MIN_STEP_PCC_SAME, None),
        ]:
            mean_pcc = pcc.mean().item()
            if mean_pcc < MIN_MEAN_PCC:
                problems.append(f"{name}: mean logit PCC {mean_pcc:.4f} < {MIN_MEAN_PCC}")
            if pcc.min() < min_step_pcc:
                bad = (pcc < min_step_pcc).nonzero().tolist()
                problems.append(
                    f"{name}: logit PCC collapsed at (prompt, step) {bad[:10]} (min {pcc.min():.3f} < {min_step_pcc})"
                )
            disagree = int((~agree).sum())
            if disagree > max_flips:
                problems.append(f"{name}: {disagree} of {agree.numel()} top-1 flips, allowed {max_flips}")
            if max_margin is not None and disagree and margins[~agree].max() > max_margin:
                problems.append(
                    f"{name}: a flipped step has top-1 margin {margins[~agree].max():.3f} > {max_margin} (not a near tie)"
                )
        assert not problems, "Packed prefill: cross-user contamination beyond the packed floors:\n" + "\n".join(
            problems
        )
        logger.info(
            f"All {batch_size} users are slot-independent over {num_tokens} teacher-forced steps within the packed "
            f"floors (rotated <= {PACKED_MAX_FLIPS_ROTATED} / 768 near-tie flips, PCC >= {PACKED_MIN_STEP_PCC_ROTATED}; "
            f"lone prompt <= {PACKED_MAX_FLIPS_LONE} / 24 near-tie flips, PCC >= {PACKED_MIN_STEP_PCC_LONE}; same slots "
            "and fillers exact)"
        )
        return

    # Thresholds (design D13: ~0.02 below the first measured Solar-Open-100B values; recorded in README.md
    # "Recorded baselines"). Measured 2026-09-07 on P150x8 (1x8, TP=8; bfp8 experts / attention / lm_head, bf16
    # residual, ttnn.all_reduce CCLs), 32 users x 24 teacher-forced steps, two runs:
    #   same-slot repeat A1 vs A2:            logit PCC min 0.99997 mean 1.00000, top-1 agreement 768/768
    #   rotated slots A1 vs B:                logit PCC min 0.99997 mean 1.00000, top-1 agreement 768/768
    #   prompt 0 alone in slot 7 (A1 vs C1):  logit PCC min 0.99999 mean 1.00000, top-1 agreement 24/24
    #   filler same-slot repeat C1 vs C2:     logit PCC min 1.00000 mean 1.00000, top-1 agreement 744/744
    #   fillers vs each other (31 slots):     logit PCC min 1.00000 mean 1.00000, top-1 agreement 720/720
    # i.e. the device is run-to-run reproducible to ~1e-5 in logit PCC and a slot layout change is
    # indistinguishable from a same-slot repeat. (The GPT-OSS source model measured mean PCC 0.993-1.000 and top-1
    # agreement 0.95-1.00 here on its reduce_scatter + all_gather CCL chain; Solar's all_reduce chain is
    # reproducible, see README.md.) A real cross-user leak collapses per-step PCC far below 0.7 (observed < 0.3 with
    # the 13-wide-grid placement bug), so the floors keep >= 0.6 of margin to the failure they guard while sitting
    # ~0.02 below the measured noise floor. The relative bounds allow only that noise on top of the same-slot baseline.
    MIN_MEAN_PCC, MAX_MEAN_PCC_DROP = 0.98, 0.01  # measured mean 1.00000, drop 0.00000 (design values kept)
    MIN_STEP_PCC = (
        0.97  # measured min 0.99997; one near-tie expert flip costs <= ~0.015 (README) -> one flip of headroom
    )
    # Top-1 agreement is checked as a disagreement COUNT: at most (1 - MIN_TOP1_AGREEMENT) of the (prompt, step) pairs
    # (measured 0 of 3024 pairs), with a floor of one pair so the 24-pair lone-prompt comparison tolerates a single
    # near-tie flip; likewise at most MAX_TOP1_AGREEMENT_DROP more pairs (floor one) than the same-slot baseline.
    MIN_TOP1_AGREEMENT, MAX_TOP1_AGREEMENT_DROP = 0.98, 0.02  # measured 1.0000 / drop 0.0000 (were 0.85 / 0.08)
    problems = []
    for name, pcc, agree, pcc_ref, agree_ref in [
        ("rotated slots", pcc_rot, agree_rot, pcc_base, agree_base),
        ("lone prompt", pcc_lone, agree_lone, pcc_base[0:1], agree_base[0:1]),
        ("filler slots", pcc_fill, agree_fill, pcc_fbase, agree_fbase),
    ]:
        mean_pcc, ref_mean_pcc = pcc.mean().item(), pcc_ref.mean().item()
        # Absolute floors (gross mixing) and baseline-relative bounds (noise-level differences only).
        if mean_pcc < MIN_MEAN_PCC or mean_pcc < ref_mean_pcc - MAX_MEAN_PCC_DROP:
            problems.append(f"{name}: mean logit PCC {mean_pcc:.4f} vs same-slot baseline {ref_mean_pcc:.4f}")
        if pcc.min() < MIN_STEP_PCC:
            bad = (pcc < MIN_STEP_PCC).nonzero().tolist()
            problems.append(f"{name}: logit PCC collapsed at (prompt, step) {bad[:10]} (min {pcc.min():.3f})")
        pairs, ref_pairs = agree.numel(), agree_ref.numel()
        disagree, ref_disagree = int((~agree).sum()), int((~agree_ref).sum())
        max_disagree = max(1, int((1 - MIN_TOP1_AGREEMENT) * pairs))
        max_extra = max(1, int(MAX_TOP1_AGREEMENT_DROP * pairs))
        if disagree > max_disagree or disagree > ref_disagree * pairs / ref_pairs + max_extra:
            problems.append(
                f"{name}: top-1 agreement {1 - disagree / pairs:.4f} ({disagree} of {pairs} pairs disagree, "
                f"allowed {max_disagree}) vs same-slot baseline {1 - ref_disagree / ref_pairs:.4f} "
                f"({ref_disagree} of {ref_pairs})"
            )
    assert not problems, "Cross-user contamination detected:\n" + "\n".join(problems)
    logger.info(f"All {batch_size} users are slot-independent over {num_tokens} teacher-forced steps")
