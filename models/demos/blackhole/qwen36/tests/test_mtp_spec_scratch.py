# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device): speculative decoding with the checkpoint's native MTP head as the drafter (milestone M3).

Flow per (w, k) config on REAL prompts (3 base + 8 GSM8K + 3 code): prefill_paged_slots (with the MTP prefill hook
filling the head's KV) -> plain greedy reference stream through the traced decode (same process) -> re-prefill ->
loop { draft k tokens per user with the MTP head (traced step, chained) -> verify step (T = k+1) -> accept/commit }
until every user has >= MTP_MIN_TOKENS committed tokens. Asserts the committed streams are bitwise the plain decode
streams (any divergence is a plumbing bug: the oracle test proved the verify mechanism draft-independent) and reports
the acceptance length, wall per step (verify / select / draft), tokens/s vs the plain decode in the same process.
Optional probe (MTP_PCC=1): the device draft logits vs a torch fp32 host reference of the head (PCC, argmax agreement)
over a 128-token prompt + 8 verify steps, w=1, k=3.

Warm-up order (tests/VERIFY_W32_AUDIT.md): every program the process runs -- decode references, verify bodies, MTP
draft steps, MTP selects, MTP prefills -- is compiled eagerly BEFORE any trace is captured.

  TT_VISIBLE_DEVICES=2,3,4,5 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.8-27B ... pytest tests/test_mtp_spec_scratch.py -s
Env: MTP_CONFIGS ("1,1;1,2;1,3;8,1;8,2;32,1" as w,k), MTP_MIN_TOKENS (64), MTP_W1_PROMPTS ("0,1,2,3,4,5,6,11,12,13"),
     MTP_PCC (1), MTP_TIMING (1), MTP_TIMING_REPLAYS (50), MTP_OUT (json path).
"""
import json
import os
import random
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_verify_step_scratch import (
    AICLK_MHZ,
    POST_PREFILL_SLEEP_MS,
    PREFILL_GROUP,
    PREFILL_GROUP_IDLE_S,
    DecodeRef,
    _pin_aiclk,
    _stream_compare,
)
from models.demos.blackhole.qwen36.tt import verify_grid as vg
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.mtp_head import MTPHead, MTPHostReference
from models.demos.blackhole.qwen36.tt.verify_step import VerifyStep

BMAX = 32
BPU = 8  # blocks per user = 512 positions (prompt <= ~300 + 64 generated + k)
CHUNK = 2048
CONFIGS = [
    tuple(int(v) for v in c.split(","))
    for c in os.environ.get("MTP_CONFIGS", "1,1;1,2;1,3;8,1;8,2;32,1").split(";")
    if c
]
MIN_TOKENS = int(os.environ.get("MTP_MIN_TOKENS", "64"))
W1_PROMPTS = [int(v) for v in os.environ.get("MTP_W1_PROMPTS", "0,1,2,3,4,5,6,11,12,13").split(",") if v]
DO_PCC = os.environ.get("MTP_PCC", "1") == "1"
DO_TIMING = os.environ.get("MTP_TIMING", "1") == "1"
N_REPLAYS = int(os.environ.get("MTP_TIMING_REPLAYS", "50"))
OUT_JSON = os.environ.get("MTP_OUT", "/home/eslim/experiments/qwen36/logs/mtp_spec_result.json")
# Extra draft sources run after the MTP run of every config (each re-prefills): "oracle" (the plain decode's
# continuation) and/or "random" -- the committed streams must be identical across sources (draft independence).
MECH_POLICIES = [p for p in os.environ.get("MTP_MECH_POLICIES", "").split(",") if p]
# Near-tie probe (default on): at every divergence from the plain decode, the PLAIN decode path's logit gap between
# its own token and the committed one (an eager decode forward at that position).
DO_NEARTIE = os.environ.get("MTP_NEARTIE", "1") == "1"
GSM8K_PARQUET = os.environ.get(
    "MTP_GSM8K_PARQUET",
    "/home/eslim/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet",
)

BASE_PROMPTS = [
    "The capital of France is",
    "Write a short story about a robot who learns to paint. Once upon a time,",
    "Q: What is 17 * 23? Let's think step by step.\nA:",
]
CODE_PROMPTS = [
    ("raw", 'def fibonacci(n):\n    """Return the n-th Fibonacci number (0-indexed)."""\n'),
    (
        "raw",
        "# Python 3\nfrom typing import List\n\n\ndef binary_search(arr: List[int], target: int) -> int:\n"
        '    """Return the index of target in the sorted list arr, or -1 if absent."""\n',
    ),
    (
        "chat",
        "Write a Python function that checks whether a string is a palindrome, with a docstring and two example calls.",
    ),
]


def _chat(tok, text):
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_prompts(tok):
    """[(type, text)] : 3 base, 8 GSM8K (chat template), 3 code."""
    prompts = [("base", p) for p in BASE_PROMPTS]
    import pandas as pd

    df = pd.read_parquet(GSM8K_PARQUET)
    for i in range(8):
        q = str(df.question[i]) + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        prompts.append(("gsm8k", _chat(tok, q)))
    for kind, text in CODE_PROMPTS:
        prompts.append(("code", _chat(tok, text) if kind == "chat" else text))
    return prompts


def _pcc(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def _prefill(model, ids, page_tables, users):
    """prefill_paged_slots of the given users (ids[i] -> slot users[i]); groups of PREFILL_GROUP; (lens, first)."""
    lens = [t.shape[1] for t in ids]
    logits = []
    for g0 in range(0, len(users), PREFILL_GROUP):
        g1 = min(len(users), g0 + PREFILL_GROUP)
        if g0 > 0:
            time.sleep(PREFILL_GROUP_IDLE_S)
        logits += model.prefill_paged_slots(
            ids[g0:g1], page_tables[users[g0:g1]], list(users[g0:g1]), valid_lens=lens[g0:g1]
        )
        ttnn.synchronize_device(model.mesh_device)
    if POST_PREFILL_SLEEP_MS > 0:
        time.sleep(POST_PREFILL_SLEEP_MS / 1e3)
    first = [int(lg.reshape(-1)[: model.vocab_size].float().argmax()) for lg in logits]
    return lens, first


def _reference_stream(ref, first, lens, n_ref):
    """Plain greedy decode of w users through the traced decode: streams [w][1 + n_ref], wall seconds."""
    w = len(first)
    streams = [[first[s]] for s in range(w)]
    pos, cur = list(lens), list(first)
    t0 = time.perf_counter()
    for _ in range(n_ref):
        nxt = ref.step(cur, pos)
        for s in range(w):
            streams[s].append(nxt[s])
        pos = [p + 1 for p in pos]
        cur = nxt
    return streams, time.perf_counter() - t0


def _spec_loop(vs, head, w, k, lens, first, min_tokens, observer=None, max_steps=None, policy="mtp", ref_streams=None):
    """Draft -> verify -> commit until every user has >= min_tokens committed tokens. policy: "mtp" (the head),
    "oracle" (the plain decode's continuation ref_streams) or "random" (rejected drafts). Returns a dict."""
    T = k + 1
    ctrl = vg.VerifyController(T=T, run=vs.run, positions=list(lens), last=list(first))
    vs.reset_sequence()
    rng = random.Random(4321 + w * 10 + k)

    def make_drafts():
        if policy == "mtp":
            return head.draft(w, k, ctrl.last, ctrl.positions, observer=observer)
        out = []
        for s in range(w):
            n_done = len(ctrl.committed[s])
            if policy == "oracle":
                d = list(ref_streams[s][n_done : n_done + k])
                while len(d) < k:
                    d.append(rng.randrange(1000))
            else:
                d = [rng.randrange(1000) for _ in range(k)]
            out.append(d)
        return out

    if policy == "mtp":
        head.begin_batch(w, list(range(w)))
    t_draft = t_verify = t_select = 0.0
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    drafts = make_drafts()
    t_draft += time.perf_counter() - t0
    n_steps = 0
    limit = max_steps if max_steps is not None else 4 * min_tokens
    while min(len(c) for c in ctrl.committed) < min_tokens and n_steps < limit:
        t0 = time.perf_counter()
        accepts = ctrl.step(drafts)
        t1 = time.perf_counter()
        if policy == "mtp":
            head.select_hidden(vs.plan, accepts)
        t2 = time.perf_counter()
        drafts = make_drafts()
        t3 = time.perf_counter()
        t_verify += t1 - t0
        t_select += t2 - t1
        t_draft += t3 - t2
        n_steps += 1
    wall = time.perf_counter() - t_all
    committed_total = sum(len(c) - 1 for c in ctrl.committed)
    accepted = sum(sum(a) for a in ctrl.accept_history)
    return {
        "steps": n_steps,
        "wall_s": wall,
        "verify_s": t_verify,
        "select_s": t_select,
        "draft_s": t_draft,
        "committed_total": committed_total,
        "drafts_accepted": accepted,
        "accept_len_mean": committed_total / max(1, n_steps * w),  # committed tokens per user per step (incl. bonus)
        "accept_rate": accepted / max(1, n_steps * w * k),
        "accept_hist": [sum(1 for st in ctrl.accept_history for a in st if a == j) for j in range(k + 1)],
        "streams": [list(c) for c in ctrl.committed],
        "accept_history": ctrl.accept_history,
    }


def _neartie_probe(model, ref, w, ids, page_tables, users, ref_streams, mism, prefill_fn):
    """For every divergence (user s, token index i, got, exp): re-prefill the w users, replay the plain decode up to
    the step that produced token i and run that step EAGERLY on the same decode programs to read the full logits ->
    the plain path's logit of its own token (exp, its argmax) minus the committed token's (got), plus got's rank.
    A gap of a few bf16 ulps of the logit magnitude is a greedy near-tie the R>32 verify numerics may flip."""
    from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
    from models.tt_transformers.tt.common import copy_host_to_device

    lens, first = prefill_fn(model, ids, page_tables, users)
    by_step = {}
    for s, i, got, exp in mism:
        by_step.setdefault(i - 1, []).append((s, i, got, exp))  # token i is produced by decode step i-1
    out = []
    pos, cur = list(lens), list(first)
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=3)
    for t in range(max(by_step) + 1):
        if t in by_step:
            host = model.prepare_decode_inputs_host(
                torch.tensor(cur, dtype=torch.int32).reshape(w, 1),
                torch.tensor(pos, dtype=torch.int32),
                page_tables[:w],
            )
            copy_host_to_device(host_tensors=host, device_tensors=ref.dev)
            cos, sin = unpack_rope(ref.dev[2])
            logits = model._forward_decode(ref.dev[0], cos, sin, ref.dev[1], ref.dev[3], sharded_lm_head=True)
            ttnn.synchronize_device(model.mesh_device)
            lg = ttnn.to_torch(logits, mesh_composer=comp).float().reshape(-1, model.vocab_size)[:w]
            ttnn.deallocate(logits)
            nxt = lg.argmax(-1).tolist()
            for s, i, got, exp in by_step[t]:
                row = lg[s]
                top2 = torch.topk(row, 2).values
                rank_got = int((row > row[got]).sum())
                out.append(
                    {
                        "user": s,
                        "token_index": i,
                        "got": got,
                        "exp": exp,
                        "decode_argmax_here": nxt[s],
                        "logit_exp": float(row[exp]),
                        "logit_got": float(row[got]),
                        "gap": float(row[exp] - row[got]),
                        "top2_gap": float(top2[0] - top2[1]),
                        "rank_got": rank_got,
                    }
                )
        else:
            nxt = ref.step(cur, pos)
        for s in range(w):
            assert nxt[s] == ref_streams[s][t + 1] or (t in by_step), (s, t, nxt[s], ref_streams[s][t + 1])
        pos = [p + 1 for p in pos]
        cur = nxt
    return out


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_mtp_spec(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    _pin_aiclk(AICLK_MHZ)
    results = {"configs": {}, "timing": {}, "pcc": None, "prompts": []}

    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    logger.info(f"[mtp] model load {time.perf_counter() - t0:.1f}s layers={len(model.layers)}")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = build_prompts(tok)
    prompt_ids = [tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(torch.int32) for _, p in prompts]
    results["prompts"] = [{"type": t, "len": int(i.shape[1]), "head": p[:60]} for (t, p), i in zip(prompts, prompt_ids)]
    logger.info(f"[mtp] prompts: {[(t, int(i.shape[1])) for (t, _), i in zip(prompts, prompt_ids)]}")
    max_T = max(k for _, k in CONFIGS) + 1
    for (t, _), ids in zip(prompts, prompt_ids):
        assert ids.shape[1] + MIN_TOKENS + 3 * max_T + 8 <= BPU * BLOCK_SIZE, f"{t} prompt too long for {BPU} blocks"
    buckets = sorted(set(Qwen36Model._mask_bucket_for(int(i.shape[1])) for i in prompt_ids))
    # PCC probe prompt: exactly 128 tokens of GSM8K text (bucket 128)
    probe_ids = None
    if DO_PCC:
        import pandas as pd

        df = pd.read_parquet(GSM8K_PARQUET)
        text = " ".join(str(df.question[i]) for i in range(8, 20))
        probe_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :128].to(torch.int32)
        assert probe_ids.shape[1] == 128, probe_ids.shape
        buckets = sorted(set(buckets) | {128})

    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    widths = sorted(set(w for w, _ in CONFIGS) | ({1} if DO_PCC else set()))
    steps, refs = {}, {}
    head = None
    try:
        # --- persistent buffers BEFORE any capture: verify plans (keep_hidden), the MTP head, its selectors ---
        t0 = time.perf_counter()
        head = MTPHead(model, page_tables, widths=widths, buckets=buckets, keep_logits=DO_PCC)
        plan_keys = list(dict.fromkeys(CONFIGS + ([(1, 3)] if DO_PCC else [])))
        for w, k in plan_keys:
            steps[(w, k)] = VerifyStep(model, w, k + 1, page_tables[:w], keep_hidden=True)
            head.bind_plan(steps[(w, k)].plan)
        logger.info(f"[mtp] head + {len(steps)} verify plans allocated in {time.perf_counter() - t0:.1f}s")

        # --- COMPILE EVERYTHING FIRST (VERIFY_W32_AUDIT.md rule) ---
        for w in widths:
            refs[w] = DecodeRef(model, w, page_tables[:w])
            refs[w].compile()
            head.compile_step(w)
        for (w, k), vs in steps.items():
            t1 = time.perf_counter()
            vs.compile()
            head.compile_select(vs.plan)
            logger.info(f"[mtp] verify ({w},T={k + 1}) R={vs.plan.R} compiled in {time.perf_counter() - t1:.1f}s")
        for b in buckets:
            head.compile_prefill(b)
        ttnn.synchronize_device(device)

        # --- prefill warm-up as the served path (chunk trace + masked-bucket traces + slot writes) ---
        t0 = time.perf_counter()
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
        logger.info(f"[mtp] prefill warmup {time.perf_counter() - t0:.1f}s")
        # the MTP prefill rides on the main prefill from here on; untimed warm 1-user prefill (lazy allocations)
        model.prefill_hidden_hook = head.prefill_hook
        _prefill(model, [prompt_ids[0]], page_tables, [0])

        # --- captures: decode references, verify bodies, MTP draft steps ---
        for w in widths:
            refs[w].capture()
            head.capture_step(w)
        for (w, k), vs in steps.items():
            t1 = time.perf_counter()
            vs.capture()
            logger.info(f"[mtp] verify ({w},T={k + 1}) captured in {time.perf_counter() - t1:.1f}s")

        # --- PCC / argmax-agreement probe of the draft logits vs the host reference ---
        if DO_PCC:
            t0 = time.perf_counter()
            ref_head = MTPHostReference(model.args.CKPT_DIR, model.args)
            logger.info(f"[mtp] host reference loaded in {time.perf_counter() - t0:.1f}s")
            w, k = 1, 3
            vs = steps[(w, k)]
            head.probe = True
            lens, first = _prefill(model, [probe_ids], page_tables, [0])
            tok_shift, rows = head.probe_prefill[0]
            state = ref_head.new_state()
            ref_head.forward(state, tok_shift.tolist(), rows, list(range(len(tok_shift))))
            probe = {"pcc": [], "agree": [], "top1_dev": [], "top1_ref": [], "steps": []}
            pending = {}

            def observer(phase, j, toks, poss, drafts):
                if phase == "pre":
                    pending["h"] = head.read_hidden_in(1)
                    return
                dev = head.read_logits(1)[0]
                ref_logits, _ = ref_head.forward(state, [toks[0]], pending["h"], [poss[0]])
                pcc = _pcc(dev, ref_logits[0])
                d_ref = int(ref_logits[0].argmax())
                probe["pcc"].append(pcc)
                probe["agree"].append(int(drafts[0]) == d_ref)
                probe["steps"].append(
                    {"pos": poss[0], "tok": toks[0], "chain_j": j, "pcc": pcc, "dev": int(drafts[0]), "ref": d_ref}
                )

            ref_streams, _ = _reference_stream(refs[w], first, lens, 8 * k + 12)
            lens, first = _prefill(model, [probe_ids], page_tables, [0])
            res = _spec_loop(vs, head, w, k, lens, first, min_tokens=10**9, observer=observer, max_steps=8)
            head.probe = False
            i, n = _stream_compare(res["streams"][0], ref_streams[0])
            pcc_t = torch.tensor(probe["pcc"])
            results["pcc"] = {
                "n_draft_steps": len(probe["pcc"]),
                "pcc_mean": float(pcc_t.mean()),
                "pcc_min": float(pcc_t.min()),
                "argmax_agree_rate": sum(probe["agree"]) / max(1, len(probe["agree"])),
                "exact_vs_decode": i is None,
                "accept_len_mean": res["accept_len_mean"],
                "steps": probe["steps"],
            }
            logger.info(
                f"[mtp] PCC probe (128-token prompt, 8 verify steps, k=3): {len(probe['pcc'])} draft steps, "
                f"PCC mean {float(pcc_t.mean()):.5f} min {float(pcc_t.min()):.5f}, argmax agreement "
                f"{results['pcc']['argmax_agree_rate']:.3f}, exact vs decode={i is None}, "
                f"accept len {res['accept_len_mean']:.2f}"
            )
            del ref_head

        # --- exactness + acceptance + tokens/s per config ---
        types = [t for t, _ in prompts]
        for w, k in CONFIGS:
            vs = steps[(w, k)]
            T = k + 1
            n_ref = MIN_TOKENS + 3 * T + 2
            if w == 1:
                assignments = [[p] for p in W1_PROMPTS]  # one prompt per run
            else:
                assignments = [[s % len(prompts) for s in range(w)]]
            cfg = {"R": vs.plan.R, "runs": []}
            for users_prompts in assignments:
                ids = [prompt_ids[p] for p in users_prompts]
                users = list(range(w))
                lens, first = _prefill(model, ids, page_tables, users)
                ref_streams, ref_wall = _reference_stream(refs[w], first, lens, n_ref)
                same_prompt_ok = all(
                    ref_streams[s] == ref_streams[users_prompts.index(users_prompts[s])] for s in range(w)
                )
                lens2, first2 = _prefill(model, ids, page_tables, users)
                assert first2 == first, "prefill first token not reproducible"
                res = _spec_loop(vs, head, w, k, lens2, first2, MIN_TOKENS)
                mism = []
                for s in range(w):
                    i, n = _stream_compare(res["streams"][s], ref_streams[s])
                    if i is not None:
                        mism.append((s, i, res["streams"][s][i], ref_streams[s][i]))
                # per prompt type acceptance
                by_type = {}
                for s in range(w):
                    t = types[users_prompts[s]]
                    acc = [st[s] for st in res["accept_history"]]
                    d = by_type.setdefault(t, {"steps": 0, "committed": 0, "accepted": 0})
                    d["steps"] += len(acc)
                    d["committed"] += len(res["streams"][s]) - 1
                    d["accepted"] += sum(acc)
                for d in by_type.values():
                    d["accept_len_mean"] = d["committed"] / max(1, d["steps"])
                spec_tps = res["committed_total"] / res["wall_s"]
                dec_tps = w * n_ref / ref_wall
                run = {
                    "prompts": users_prompts,
                    "prompt_types": [types[p] for p in users_prompts],
                    "lens": lens,
                    "ref_same_prompt_identical": same_prompt_ok,
                    "mismatch_vs_decode": mism,
                    "exact_vs_decode": not mism,
                    "spec_tok_s_aggregate": spec_tps,
                    "spec_tok_s_per_user": spec_tps / w,
                    "decode_tok_s_aggregate": dec_tps,
                    "decode_tok_s_per_user": dec_tps / w,
                    "decode_ms_per_step": 1e3 * ref_wall / n_ref,
                    "spec_ms_per_step": 1e3 * res["wall_s"] / max(1, res["steps"]),
                    "verify_ms_per_step": 1e3 * res["verify_s"] / max(1, res["steps"]),
                    "select_ms_per_step": 1e3 * res["select_s"] / max(1, res["steps"]),
                    "draft_ms_per_step": 1e3 * res["draft_s"] / max(1, res["steps"] + 1),
                    "by_type": by_type,
                    **{kk: v for kk, v in res.items() if kk not in ("streams",)},
                    "streams": res["streams"],
                    "ref_streams": ref_streams,
                    "text_user0": tok.decode(res["streams"][0][:40]),
                }
                if mism and DO_NEARTIE:
                    probe_rows = _neartie_probe(model, refs[w], w, ids, page_tables, users, ref_streams, mism, _prefill)
                    run["neartie"] = probe_rows
                    logger.info(
                        f"[mtp] ({w},k={k}) NEAR-TIE probe (plain decode logits at each divergence): "
                        + "; ".join(
                            f"u{r['user']}@{r['token_index']}: exp {r['exp']} {r['logit_exp']:.3f} vs got {r['got']} "
                            f"{r['logit_got']:.3f} gap {r['gap']:.3f} (rank of got {r['rank_got']})"
                            for r in probe_rows
                        )
                    )
                for pol in MECH_POLICIES:
                    lens3, first3 = _prefill(model, ids, page_tables, users)
                    res_p = _spec_loop(vs, head, w, k, lens3, first3, MIN_TOKENS, policy=pol, ref_streams=ref_streams)
                    ident = all(_stream_compare(res_p["streams"][s], res["streams"][s])[0] is None for s in range(w))
                    mism_p = [
                        (s,) + tuple(_stream_compare(res_p["streams"][s], ref_streams[s])[:1])
                        for s in range(w)
                        if _stream_compare(res_p["streams"][s], ref_streams[s])[0] is not None
                    ]
                    run[f"policy_{pol}"] = {
                        "steps": res_p["steps"],
                        "accept_len_mean": res_p["accept_len_mean"],
                        "identical_to_mtp_stream": ident,
                        "mismatch_vs_decode": mism_p,
                        "streams": res_p["streams"],
                    }
                    logger.info(
                        f"[mtp] ({w},k={k}) drafts={pol}: {res_p['steps']} steps, accept len {res_p['accept_len_mean']:.2f}; "
                        f"committed streams identical to the MTP-draft run: {ident}; mismatches vs decode: {len(mism_p)}"
                    )
                cfg["runs"].append(run)
                logger.info(
                    f"[mtp] ({w},k={k}) prompts {users_prompts if w <= 8 else 'all 14 (mod)'}: {res['steps']} steps, "
                    f"accept len {res['accept_len_mean']:.2f} tok/user/step (rate {res['accept_rate']:.2f}, hist {res['accept_hist']}), "
                    f"spec {spec_tps:.1f} tok/s vs decode {dec_tps:.1f} tok/s (x{spec_tps / dec_tps:.2f}); "
                    f"per step: verify {run['verify_ms_per_step']:.1f} + select {run['select_ms_per_step']:.1f} + "
                    f"draft {run['draft_ms_per_step']:.1f} ms = {run['spec_ms_per_step']:.1f} (decode {run['decode_ms_per_step']:.1f}); "
                    f"exact={not mism} {mism[:3]}; same-prompt users identical={same_prompt_ok}"
                )
                if not mism:
                    logger.info(f"[mtp]   user0 text: {run['text_user0']!r}")
            cfg["exact_vs_decode"] = all(r["exact_vs_decode"] for r in cfg["runs"])
            tot_steps = sum(r["steps"] for r in cfg["runs"])
            cfg["accept_len_mean"] = sum(r["committed_total"] for r in cfg["runs"]) / max(1, tot_steps * w)
            cfg["by_type"] = {}
            for r in cfg["runs"]:
                for t, d in r["by_type"].items():
                    a = cfg["by_type"].setdefault(t, {"steps": 0, "committed": 0, "accepted": 0})
                    for kk in ("steps", "committed", "accepted"):
                        a[kk] += d[kk]
            for d in cfg["by_type"].values():
                d["accept_len_mean"] = d["committed"] / max(1, d["steps"])
            cfg["spec_tok_s_aggregate"] = sum(r["spec_tok_s_aggregate"] for r in cfg["runs"]) / len(cfg["runs"])
            cfg["decode_tok_s_aggregate"] = sum(r["decode_tok_s_aggregate"] for r in cfg["runs"]) / len(cfg["runs"])
            results["configs"][f"{w},{k}"] = cfg
            by_type_txt = ", ".join(f"{t}: {d['accept_len_mean']:.2f}" for t, d in cfg["by_type"].items())
            logger.info(
                f"[mtp] ({w},k={k}) RESULT exact_vs_plain_decode={cfg['exact_vs_decode']} accept_len {cfg['accept_len_mean']:.2f} "
                f"by type {{{by_type_txt}}} "
                f"spec {cfg['spec_tok_s_aggregate']:.1f} vs decode {cfg['decode_tok_s_aggregate']:.1f} tok/s"
            )

        # --- timing: traced draft step per width (upload + replay + argmax readback), MTP stats ---
        if DO_TIMING:
            for w in widths:
                med, mn = head.time_step_replays(w, N_REPLAYS)
                results["timing"][f"mtp_step_w{w}"] = {"median_ms": med, "min_ms": mn}
                logger.info(f"[mtp] TIMING draft step w={w} traced x{N_REPLAYS}: med {med:.2f} min {mn:.2f} ms")
            results["timing"]["mtp_stats"] = dict(head.stats)
            st = head.stats
            logger.info(
                f"[mtp] TIMING totals: {st['draft_steps']} draft steps {1e3 * st['draft_wall'] / max(1, st['draft_steps']):.2f} ms each; "
                f"{st['select_calls']} selects {1e3 * st['select_wall'] / max(1, st['select_calls']):.2f} ms each; "
                f"{st['prefill_calls']} MTP prefills {1e3 * st['prefill_wall'] / max(1, st['prefill_calls']):.1f} ms each (eager)"
            )
    finally:
        model.prefill_hidden_hook = None
        for vs in steps.values():
            vs.release()
        for r in refs.values():
            r.release()
        if head is not None:
            head.release()
        model.pd_gdn_capture = None
        model.free_kv_caches()
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=1)
        logger.info(f"[mtp] results -> {OUT_JSON}")
    if results["pcc"] is not None:
        p = results["pcc"]
        print(
            f"MTP_PCC steps={p['n_draft_steps']} pcc_mean={p['pcc_mean']:.5f} pcc_min={p['pcc_min']:.5f} "
            f"agree={p['argmax_agree_rate']:.3f} exact={p['exact_vs_decode']}"
        )
    for key, cfg in results["configs"].items():
        print(
            f"MTP_EXACT {key}: exact_vs_decode={cfg['exact_vs_decode']} accept_len={cfg['accept_len_mean']:.3f} "
            + " ".join(f"{t}={d['accept_len_mean']:.3f}" for t, d in cfg["by_type"].items())
            + f" spec_tok_s={cfg['spec_tok_s_aggregate']:.1f} decode_tok_s={cfg['decode_tok_s_aggregate']:.1f}"
        )
    print("MTP_TIMING " + json.dumps(results["timing"]))
    # Exactness: bitwise vs the plain decode for every plan on the fused-AR path (R <= 32: the verify body runs the
    # decode step's own ops). Above 32 rows the body runs the fractured reduce-scatter path whose numerics are not the
    # fused all-reduce's (verify_step.py), so its greedy stream may flip at bf16 near-ties: such a config passes when
    # every divergence is a near-tie of the plain decode's own logits (gap <= 2 bf16 ulps at |logit| 16..32 = 0.25) and
    # the draft-policy runs (when requested) commit identical streams (logs/mtp_diag32.log: 15/15 divergences at gap
    # 0.125/0.25/0.0, MTP == oracle == random streams).
    for key, cfg in results["configs"].items():
        if cfg["exact_vs_decode"]:
            continue
        assert (
            cfg["R"] > 32
        ), f"config {key} (R={cfg['R']}, decode-numerics path) committed stream != plain greedy decode"
        for r in cfg["runs"]:
            if r["mismatch_vs_decode"]:
                assert "neartie" in r, f"config {key}: divergences without the near-tie probe (MTP_NEARTIE=0)"
                bad = [x for x in r["neartie"] if x["gap"] > 0.25]
                assert not bad, f"config {key}: divergences that are NOT near-ties: {bad}"
            for pol in MECH_POLICIES:
                assert r[f"policy_{pol}"][
                    "identical_to_mtp_stream"
                ], f"config {key}: {pol} drafts commit a different stream"
    if results["pcc"] is not None:
        assert results["pcc"]["pcc_mean"] >= 0.99, f"draft logits PCC {results['pcc']['pcc_mean']}"
