# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device): speculative decoding with the DFlash2 block-diffusion drafter (milestone M4), half B P150x4.

Flow per (w, k) config on the 14 real prompts of test_mtp_spec_scratch (3 base + 8 GSM8K + 3 code): prefill_paged_slots
(EAGER masked bucket; the target's residual stream after layers 5/19/33/47/61 is captured per prompt row by
PrefillAuxCapture) -> plain greedy reference stream through the traced decode (same process) -> re-prefill -> the
prompt's aux rows become the drafter's context KV (write_context) -> loop { traced block draft (7 drafts, the first k
used) -> verify step T = k+1 (VerifyStepAux keeps every grid row's aux hiddens: plan.out_aux) -> accept/commit ->
traced commit of the grid rows' aux into the drafter KV } until every user has >= DF_MIN_TOKENS committed tokens.
Asserts the committed streams == the plain decode (bitwise for R <= 32, near-tie-bounded above, as the MTP test) and
reports acceptance length per prompt type, verify / draft / commit ms per step, tokens/s vs the plain decode.
Optional probe (DF_PCC=1): device draft logits of the 7 draft rows vs the fp32 host reference fed the SAME aux rows
(PCC, selected-path agreement) over a 128-token prompt + 8 steps at w=1, k=7.

Warm-up order (tests/VERIFY_W32_AUDIT.md): every program (decode references, drafter steps, verify bodies, drafter
commits) is compiled eagerly, one prompt per masked bucket is prefilled, THEN the traces are captured.

  TT_VISIBLE_DEVICES=2,3,4,5 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.8-27B ... pytest tests/test_dflash2_spec_scratch.py -s
Env: DF_CONFIGS ("1,7;1,3;4,7;8,7;8,3;16,3;32,3" as w,k), DF_MIN_TOKENS (64), DF_W1_PROMPTS, DF_PCC (1), DF_TIMING (1),
     DF_TIMING_REPLAYS (30), DF_OUT (json path), DF_NEARTIE (1), DF_PROBE_STEPS (8).
"""
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_mtp_spec_scratch import (
    GSM8K_PARQUET,
    _neartie_probe,
    _pcc,
    _prefill,
    _reference_stream,
    build_prompts,
)
from models.demos.blackhole.qwen36.tests.test_verify_step_scratch import (
    AICLK_MHZ,
    DecodeRef,
    _pin_aiclk,
    _stream_compare,
)
from models.demos.blackhole.qwen36.tt import aux_hidden as ah
from models.demos.blackhole.qwen36.tt import verify_grid as vg
from models.demos.blackhole.qwen36.tt.dflash2_head import (
    DFlash2Drafter,
    DFlash2HostReference,
    load_dflash2_state_dict,
    load_target_embed_and_head,
    selector_walk,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.verify_step import VerifyStep

BMAX = 32
BPU = 8  # blocks per user = 512 positions
CHUNK = 2048
CONFIGS = [
    tuple(int(v) for v in c.split(","))
    for c in os.environ.get("DF_CONFIGS", "1,7;1,3;4,7;8,7;8,3;16,3;32,3").split(";")
    if c
]
MIN_TOKENS = int(os.environ.get("DF_MIN_TOKENS", "64"))
W1_PROMPTS = [int(v) for v in os.environ.get("DF_W1_PROMPTS", "0,1,2,3,4,5,6,11,12,13").split(",") if v]
DO_PCC = os.environ.get("DF_PCC", "1") == "1"
DO_TIMING = os.environ.get("DF_TIMING", "1") == "1"
N_REPLAYS = int(os.environ.get("DF_TIMING_REPLAYS", "30"))
PROBE_STEPS = int(os.environ.get("DF_PROBE_STEPS", "8"))
DO_NEARTIE = os.environ.get("DF_NEARTIE", "1") == "1"
OUT_JSON = os.environ.get("DF_OUT", "/home/eslim/experiments/qwen36/logs/dflash2_spec_result.json")
KV_TOKENS_REPORT = 1_052_672


class AuxRowsHook:
    """``model.prefill_aux_hook`` of the harness: the aux rows of every prefilled segment gathered to host per decode
    slot ([n, 5*dim] bf16, segments concatenated in position order) -- the D-side stand-in for the served hook
    (tt/aux_hidden.py DFlash2ContextPrefillHook projects instead of storing)."""

    def __init__(self, model):
        self.model = model
        self.rows = {}
        self.enabled = False

    def __call__(self, user_ctx, aux_frac, token_buf, actual_len, bucket, chunk_start):
        if user_ctx is None or not self.enabled:
            return
        slot = int(user_ctx[1])
        rows = ah.aux_rows_to_host(self.model, aux_frac, n_valid=int(actual_len), fractured=True)
        if int(chunk_start) == 0:
            self.rows[slot] = rows
        else:
            self.rows[slot] = torch.cat([self.rows[slot], rows])


def _prefill_with_aux(model, ids, page_tables, users, cap):
    """_prefill + the per-user aux rows [len_u, 5*dim] bf16 through the model's prefill_aux_hook."""
    cap.rows = {}
    cap.enabled = True
    try:
        lens, first = _prefill(model, ids, page_tables, users)
    finally:
        cap.enabled = False
    aux = [cap.rows[int(u)] for u in users]
    for i, a in enumerate(aux):
        assert a.shape[0] == lens[i], (a.shape, lens[i])
    cap.rows = {}
    return lens, first, aux


def _spec_loop(vs, head, w, k, lens, first, aux, min_tokens, observer=None, max_steps=None, ref_state_cb=None):
    """Draft -> verify -> commit until every user has >= min_tokens committed tokens. aux[s]: the prompt's aux rows."""
    T = k + 1
    ctrl = vg.VerifyController(T=T, run=vs.run, positions=list(lens), last=list(first))
    vs.reset_sequence()
    t_ctx = time.perf_counter()
    for s in range(w):
        head.write_context(s, torch.arange(lens[s]), aux[s])
    t_ctx = time.perf_counter() - t_ctx
    t_draft = t_verify = t_commit = 0.0
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    drafts7, _, _ = head.draft(w, ctrl.last, ctrl.positions, observer=observer)
    t_draft += time.perf_counter() - t0
    n_steps = 0
    limit = max_steps if max_steps is not None else 4 * min_tokens
    while min(len(c) for c in ctrl.committed) < min_tokens and n_steps < limit:
        pos_before = list(ctrl.positions)
        t0 = time.perf_counter()
        accepts = ctrl.step([d[:k] for d in drafts7])
        t1 = time.perf_counter()
        head.commit(vs.plan, pos_before)
        t2 = time.perf_counter()
        if ref_state_cb is not None:
            ref_state_cb(pos_before, accepts)
        drafts7, _, _ = head.draft(w, ctrl.last, ctrl.positions, observer=observer)
        t3 = time.perf_counter()
        t_verify += t1 - t0
        t_commit += t2 - t1
        t_draft += t3 - t2
        n_steps += 1
    wall = time.perf_counter() - t_all
    committed_total = sum(len(c) - 1 for c in ctrl.committed)
    accepted = sum(sum(a) for a in ctrl.accept_history)
    return {
        "steps": n_steps,
        "wall_s": wall,
        "context_s": t_ctx,
        "verify_s": t_verify,
        "commit_s": t_commit,
        "draft_s": t_draft,
        "committed_total": committed_total,
        "drafts_accepted": accepted,
        "accept_len_mean": committed_total / max(1, n_steps * w),
        "accept_rate": accepted / max(1, n_steps * w * k),
        "accept_hist": [sum(1 for st in ctrl.accept_history for a in st if a == j) for j in range(k + 1)],
        "streams": [list(c) for c in ctrl.committed],
        "accept_history": ctrl.accept_history,
    }


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_dflash2_spec(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    _pin_aiclk(AICLK_MHZ)
    results = {"configs": {}, "timing": {}, "pcc": None, "prompts": [], "dram": None}

    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    logger.info(f"[df2] model load {time.perf_counter() - t0:.1f}s layers={len(model.layers)}")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = build_prompts(tok)
    prompt_ids = [tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(torch.int32) for _, p in prompts]
    results["prompts"] = [{"type": t, "len": int(i.shape[1]), "head": p[:60]} for (t, p), i in zip(prompts, prompt_ids)]
    logger.info(f"[df2] prompts: {[(t, int(i.shape[1])) for (t, _), i in zip(prompts, prompt_ids)]}")
    max_T = max(k for _, k in CONFIGS) + 1
    for (t, _), ids in zip(prompts, prompt_ids):
        assert ids.shape[1] + MIN_TOKENS + 3 * max_T + 16 <= BPU * BLOCK_SIZE, f"{t} prompt too long for {BPU} blocks"
    probe_ids = None
    if DO_PCC:
        import pandas as pd

        df = pd.read_parquet(GSM8K_PARQUET)
        text = " ".join(str(df.question[i]) for i in range(8, 20))
        probe_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :128].to(torch.int32)
        assert probe_ids.shape[1] == 128
    bucket_reps = {}
    for i, ids in enumerate(prompt_ids + ([probe_ids] if probe_ids is not None else [])):
        bucket_reps.setdefault(Qwen36Model._mask_bucket_for(int(ids.shape[1])), ids)

    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    widths = sorted(set(w for w, _ in CONFIGS) | ({1} if DO_PCC else set()))
    steps, refs = {}, {}
    head = None
    try:
        # --- persistent buffers BEFORE any capture: the drafter (+ its KV), the verify plans, the commit buffers ---
        t0 = time.perf_counter()
        head = DFlash2Drafter(model, page_tables, widths=widths, keep_logits=DO_PCC)
        cfg = head.cfg
        plan_keys = list(dict.fromkeys(CONFIGS + ([(1, 7)] if DO_PCC else [])))
        for w, k in plan_keys:
            steps[(w, k)] = VerifyStep(
                model, w, k + 1, page_tables[:w], keep_aux_hidden=True, aux_layers=tuple(cfg.target_layer_ids)
            )
            head.bind_plan(steps[(w, k)].plan)
        # the model's aux copies of every prefill path (set BEFORE the prefill warm-up captures) + the harness hook
        model.prefill_aux_layers = tuple(cfg.target_layer_ids)
        cap = AuxRowsHook(model)
        model.prefill_aux_hook = cap
        results["dram"] = head.dram_report(KV_TOKENS_REPORT)
        logger.info(
            f"[df2] drafter + {len(steps)} verify plans allocated in {time.perf_counter() - t0:.1f}s; DRAM {results['dram']}"
        )

        # --- COMPILE EVERYTHING FIRST ---
        for w in widths:
            refs[w] = DecodeRef(model, w, page_tables[:w])
            refs[w].compile()
            head.compile_step(w)
        for (w, k), vs in steps.items():
            t1 = time.perf_counter()
            vs.compile()
            head.compile_commit(vs.plan)
            logger.info(
                f"[df2] verify ({w},T={k + 1}) R={vs.plan.R} + commit compiled in {time.perf_counter() - t1:.1f}s"
            )
        ttnn.synchronize_device(device)

        # --- prefill warm-up as the served path, then one prefill per masked bucket (eager bucket programs) ---
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
        for b, ids in sorted(bucket_reps.items()):
            _prefill_with_aux(model, [ids], page_tables, [0], cap)
            logger.info(f"[df2] warm prefill bucket {b}")
        # compile the context fill (projector matmuls at S rows + paged_fill_cache over S/64 blocks) for EVERY prompt
        # length class before the captures: the fill program depends on the block count (a 93-token prompt = 2 blocks
        # cost a 4.8 s post-capture JIT in logs/df2_full1.log, "context write 4841 ms")
        max_len = max(int(i.shape[1]) for i in prompt_ids + ([probe_ids] if probe_ids is not None else []))
        for S in range(BLOCK_SIZE, -(-max_len // BLOCK_SIZE) * BLOCK_SIZE + 1, BLOCK_SIZE):
            head.write_context(0, torch.arange(S), torch.zeros(S, 5 * cfg.dim))
        logger.info(f"[df2] prefill warmup {time.perf_counter() - t0:.1f}s")

        # --- captures: decode references, drafter steps, verify bodies, then the commits (they read plan.out_aux) ---
        for w in widths:
            refs[w].capture()
            head.capture_step(w)
        for (w, k), vs in steps.items():
            t1 = time.perf_counter()
            vs.capture()
            head.capture_commit(vs.plan)
            logger.info(f"[df2] verify ({w},T={k + 1}) + commit captured in {time.perf_counter() - t1:.1f}s")

        # --- PCC probe: device draft logits / selected path vs the host reference on the same aux rows ---
        if DO_PCC:
            t0 = time.perf_counter()
            sd = load_dflash2_state_dict(cfg.path)
            emb, lm = load_target_embed_and_head(model.args.CKPT_DIR)
            ref = DFlash2HostReference(cfg, sd, emb, lm)
            del sd
            logger.info(f"[df2] host reference loaded in {time.perf_counter() - t0:.1f}s")
            w, k = 1, 7
            vs = steps[(w, k)]
            probe = {"pcc": [], "agree": [], "path_agree": [], "steps": []}
            state = {"ref": None}

            def observer(w_, anchors, pos, per_user, logits):
                dev = head.read_logits(w_, logits)[1:]  # the 7 draft rows [7, V]
                r = ref.draft(state["ref"], int(anchors[0]), int(pos[0]))
                pccs = [_pcc(dev[t], r["logits"][t]) for t in range(cfg.n_draft)]
                cands, unary, hp = per_user[0]
                dev_path, _ = selector_walk(hp, cands, unary, int(anchors[0]), head.sel_pred, head.sel_succ)
                agree = [int(a == b) for a, b in zip(dev_path, r["tokens"])]
                argmax_agree = [int(int(dev[t].argmax()) == int(r["logits"][t].argmax())) for t in range(cfg.n_draft)]
                probe["pcc"].extend(pccs)
                probe["agree"].extend(argmax_agree)
                probe["path_agree"].extend(agree)
                probe["steps"].append(
                    {
                        "pos": int(pos[0]),
                        "anchor": int(anchors[0]),
                        "pcc": pccs,
                        "dev_path": dev_path,
                        "ref_path": r["tokens"],
                    }
                )

            def ref_state_cb(pos_before, accepts):
                # the committed rows only (0..a_s): the device writes all T rows but the rejected positions are
                # overwritten by the next block's K/V before they are attended; the host state has no such overwrite
                n = int(accepts[0]) + 1
                aux_rows = vs.plan.read_aux(n)
                ref.append_context(state["ref"], aux_rows, torch.arange(pos_before[0], pos_before[0] + n))

            lens, first, aux = _prefill_with_aux(model, [probe_ids], page_tables, [0], cap)
            ref_streams, _ = _reference_stream(refs[w], first, lens, PROBE_STEPS * (k + 1) + 4)
            lens, first, aux = _prefill_with_aux(model, [probe_ids], page_tables, [0], cap)
            assert lens[0] == 128
            state["ref"] = ref.new_state()
            ref.append_context(state["ref"], aux[0], torch.arange(lens[0]))
            res = _spec_loop(
                vs,
                head,
                w,
                k,
                lens,
                first,
                aux,
                10**9,
                observer=observer,
                max_steps=PROBE_STEPS,
                ref_state_cb=ref_state_cb,
            )
            i, n = _stream_compare(res["streams"][0], ref_streams[0])
            pcc_t = torch.tensor(probe["pcc"])
            results["pcc"] = {
                "n_draft_rows": len(probe["pcc"]),
                "pcc_mean": float(pcc_t.mean()),
                "pcc_min": float(pcc_t.min()),
                "argmax_agree_rate": sum(probe["agree"]) / max(1, len(probe["agree"])),
                "path_agree_rate": sum(probe["path_agree"]) / max(1, len(probe["path_agree"])),
                "exact_vs_decode": i is None,
                "accept_len_mean": res["accept_len_mean"],
                "steps": probe["steps"],
            }
            logger.info(
                f"[df2] PCC probe (128-token prompt, {res['steps']} verify steps, k=7): {len(probe['pcc'])} draft rows, "
                f"PCC mean {float(pcc_t.mean()):.5f} min {float(pcc_t.min()):.5f}, argmax agreement "
                f"{results['pcc']['argmax_agree_rate']:.3f}, selected-path agreement {results['pcc']['path_agree_rate']:.3f}, "
                f"exact vs decode={i is None}, accept len {res['accept_len_mean']:.2f}"
            )
            del ref

        # --- exactness + acceptance + tokens/s per config ---
        types = [t for t, _ in prompts]
        for w, k in CONFIGS:
            vs = steps[(w, k)]
            T = k + 1
            n_ref = MIN_TOKENS + 3 * T + 2
            assignments = [[p] for p in W1_PROMPTS] if w == 1 else [[s % len(prompts) for s in range(w)]]
            cfg_res = {"R": vs.plan.R, "runs": []}
            for users_prompts in assignments:
                ids = [prompt_ids[p] for p in users_prompts]
                users = list(range(w))
                lens, first, _ = _prefill_with_aux(model, ids, page_tables, users, cap)
                ref_streams, ref_wall = _reference_stream(refs[w], first, lens, n_ref)
                same_prompt_ok = all(
                    ref_streams[s] == ref_streams[users_prompts.index(users_prompts[s])] for s in range(w)
                )
                lens2, first2, aux = _prefill_with_aux(model, ids, page_tables, users, cap)
                assert first2 == first, "prefill first token not reproducible"
                res = _spec_loop(vs, head, w, k, lens2, first2, aux, MIN_TOKENS)
                mism = []
                for s in range(w):
                    i, n = _stream_compare(res["streams"][s], ref_streams[s])
                    if i is not None:
                        mism.append((s, i, res["streams"][s][i], ref_streams[s][i]))
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
                    "commit_ms_per_step": 1e3 * res["commit_s"] / max(1, res["steps"]),
                    "draft_ms_per_step": 1e3 * res["draft_s"] / max(1, res["steps"] + 1),
                    "context_ms": 1e3 * res["context_s"],
                    "by_type": by_type,
                    **{kk: v for kk, v in res.items() if kk not in ("streams",)},
                    "streams": res["streams"],
                    "ref_streams": ref_streams,
                    "text_user0": tok.decode(res["streams"][0][:40]),
                }
                if mism and DO_NEARTIE:
                    cap.enabled = False
                    probe_rows = _neartie_probe(model, refs[w], w, ids, page_tables, users, ref_streams, mism, _prefill)
                    run["neartie"] = probe_rows
                    logger.info(
                        f"[df2] ({w},k={k}) NEAR-TIE probe: "
                        + "; ".join(
                            f"u{r['user']}@{r['token_index']}: exp {r['exp']} {r['logit_exp']:.3f} vs got {r['got']} "
                            f"{r['logit_got']:.3f} gap {r['gap']:.3f} (rank of got {r['rank_got']})"
                            for r in probe_rows
                        )
                    )
                cfg_res["runs"].append(run)
                logger.info(
                    f"[df2] ({w},k={k}) prompts {users_prompts if w <= 8 else 'all 14 (mod)'}: {res['steps']} steps, "
                    f"accept len {res['accept_len_mean']:.2f} tok/user/step (rate {res['accept_rate']:.2f}, hist {res['accept_hist']}), "
                    f"spec {spec_tps:.1f} tok/s vs decode {dec_tps:.1f} tok/s (x{spec_tps / dec_tps:.2f}); "
                    f"per step: verify {run['verify_ms_per_step']:.1f} + commit {run['commit_ms_per_step']:.1f} + "
                    f"draft {run['draft_ms_per_step']:.1f} ms = {run['spec_ms_per_step']:.1f} (decode {run['decode_ms_per_step']:.1f}); "
                    f"context write {run['context_ms']:.0f} ms; exact={not mism} {mism[:3]}; same-prompt users identical={same_prompt_ok}"
                )
                if not mism:
                    logger.info(f"[df2]   user0 text: {run['text_user0']!r}")
            cfg_res["exact_vs_decode"] = all(r["exact_vs_decode"] for r in cfg_res["runs"])
            tot_steps = sum(r["steps"] for r in cfg_res["runs"])
            cfg_res["accept_len_mean"] = sum(r["committed_total"] for r in cfg_res["runs"]) / max(1, tot_steps * w)
            cfg_res["by_type"] = {}
            for r in cfg_res["runs"]:
                for t, d in r["by_type"].items():
                    a = cfg_res["by_type"].setdefault(t, {"steps": 0, "committed": 0, "accepted": 0})
                    for kk in ("steps", "committed", "accepted"):
                        a[kk] += d[kk]
            for d in cfg_res["by_type"].values():
                d["accept_len_mean"] = d["committed"] / max(1, d["steps"])
            cfg_res["spec_tok_s_aggregate"] = sum(r["spec_tok_s_aggregate"] for r in cfg_res["runs"]) / len(
                cfg_res["runs"]
            )
            cfg_res["decode_tok_s_aggregate"] = sum(r["decode_tok_s_aggregate"] for r in cfg_res["runs"]) / len(
                cfg_res["runs"]
            )
            for key in (
                "verify_ms_per_step",
                "commit_ms_per_step",
                "draft_ms_per_step",
                "spec_ms_per_step",
                "decode_ms_per_step",
            ):
                cfg_res[key] = sum(r[key] for r in cfg_res["runs"]) / len(cfg_res["runs"])
            results["configs"][f"{w},{k}"] = cfg_res
            by_type_txt = ", ".join(f"{t}: {d['accept_len_mean']:.2f}" for t, d in cfg_res["by_type"].items())
            logger.info(
                f"[df2] ({w},k={k}) RESULT exact_vs_plain_decode={cfg_res['exact_vs_decode']} accept_len {cfg_res['accept_len_mean']:.2f} "
                f"by type {{{by_type_txt}}} spec {cfg_res['spec_tok_s_aggregate']:.1f} vs decode {cfg_res['decode_tok_s_aggregate']:.1f} tok/s"
            )

        # --- timing: traced draft step / commit per width, drafter stats ---
        if DO_TIMING:
            for w in widths:
                med, mn = head.time_step_replays(w, N_REPLAYS)
                results["timing"][f"draft_step_w{w}"] = {"median_ms": med, "min_ms": mn, "rows": w * head.B}
                logger.info(
                    f"[df2] TIMING draft step w={w} (R={w * head.B}) traced x{N_REPLAYS}: med {med:.2f} min {mn:.2f} ms"
                )
            for (w, k), vs in steps.items():
                med, mn = head.time_commit_replays(vs.plan, N_REPLAYS)
                results["timing"][f"commit_w{w}_T{k + 1}"] = {"median_ms": med, "min_ms": mn}
                logger.info(f"[df2] TIMING commit ({w},T={k + 1}) traced x{N_REPLAYS}: med {med:.2f} min {mn:.2f} ms")
            results["timing"]["stats"] = dict(head.stats)
    finally:
        model.prefill_aux_hook = None
        model.prefill_aux_layers = ()
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
        logger.info(f"[df2] results -> {OUT_JSON}")
    if results["pcc"] is not None:
        p = results["pcc"]
        print(
            f"DF_PCC rows={p['n_draft_rows']} pcc_mean={p['pcc_mean']:.5f} pcc_min={p['pcc_min']:.5f} "
            f"argmax_agree={p['argmax_agree_rate']:.3f} path_agree={p['path_agree_rate']:.3f} exact={p['exact_vs_decode']} "
            f"accept_len={p['accept_len_mean']:.2f}"
        )
    for key, c in results["configs"].items():
        print(
            f"DF_EXACT {key}: R={c['R']} exact_vs_decode={c['exact_vs_decode']} accept_len={c['accept_len_mean']:.3f} "
            + " ".join(f"{t}={d['accept_len_mean']:.3f}" for t, d in c["by_type"].items())
            + f" spec_tok_s={c['spec_tok_s_aggregate']:.1f} decode_tok_s={c['decode_tok_s_aggregate']:.1f} "
            f"verify_ms={c['verify_ms_per_step']:.1f} commit_ms={c['commit_ms_per_step']:.1f} draft_ms={c['draft_ms_per_step']:.1f}"
        )
    print("DF_TIMING " + json.dumps(results["timing"]))
    print("DF_DRAM " + json.dumps(results["dram"]))
    for key, c in results["configs"].items():
        if c["exact_vs_decode"]:
            continue
        assert c["R"] > 32, f"config {key} (R={c['R']}, decode-numerics path) committed stream != plain greedy decode"
        for r in c["runs"]:
            if r["mismatch_vs_decode"]:
                assert "neartie" in r, f"config {key}: divergences without the near-tie probe (DF_NEARTIE=0)"
                bad = [x for x in r["neartie"] if x["gap"] > 0.25]
                assert not bad, f"config {key}: divergences that are NOT near-ties: {bad}"
    if results["pcc"] is not None:
        assert results["pcc"]["pcc_mean"] >= 0.99, f"draft logits PCC {results['pcc']['pcc_mean']}"
