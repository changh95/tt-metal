# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device, half B): adaptive draft length of the DFlash2 band (tt/spec_serving.py acceptance rule) measured
in-process through the SERVED loop (SpecDecoder + the Engine of test_spec_serving_scratch: stable slots, the hold
protocol, the P-side hooks + payload import per admission) on real text (the 14 prompts of test_mtp_spec_scratch)
and on RANDOM-token prompts of 128 / 1024 / 4096 tokens with 128- and 1024-token outputs, at w = 1 and w = 4.

One policy per process (SPEC_POLICY):
  k7           hybrid, the DFlash2 band at T = 8 (k = 7), the acceptance rule off  (= the v12 served default)
  k3           hybrid with QWEN36_SPEC_K=3: every plan (w,4), the block drafter's first 3 path tokens in the band
  mtp_mode     hybrid forced into the MTP plans everywhere (context limit 1): the MTP head at (w,4) + the hybrid's
               per-step DFlash2 commit (what the low band of the adaptive policy runs)
  mtp          the mtp drafter alone, K = 3 (the v10 served stack's loop)
  adaptive     hybrid + the acceptance rule (QWEN36_SPEC_DFLASH2_ADAPTIVE=1, the QWEN36_SPEC_ADAPT_* knobs of the env)
  adaptive_df2 as adaptive with QWEN36_SPEC_ADAPT_DRAFTER=dflash2 (the block drafter at k = 3 in the low band)
Every committed stream must be BITWISE the plain traced decode of its prompt (every plan has R <= 32). Reports per
workload item: steps, ms/step, tokens, ms/token, tok/s, accepted drafts per user-step, the steps per (plan, drafter,
mode), the rule's switches; and the DFlash2 selector margin vs acceptance per draft position (why drafts are not
trimmed per user: tt/spec_serving.py).

  scripts/spec_adapt_run.sh adaptive           (-> logs/adapt/adapt_adaptive.log / .json)
Env: SPEC_POLICY, SPEC_OUT, ADAPT_OSL_LONG (1024), ADAPT_RAND_LENS (128,1024,4096), ADAPT_W4 (1), ADAPT_REAL_W1 (1).
"""

import json
import math
import os
import random
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_mtp_spec_scratch import _prefill, build_prompts
from models.demos.blackhole.qwen36.tests.test_spec_serving_scratch import BMAX, CHUNK, Engine
from models.demos.blackhole.qwen36.tests.test_verify_step_scratch import (
    AICLK_MHZ,
    DecodeRef,
    _pin_aiclk,
    _stream_compare,
)
from models.demos.blackhole.qwen36.tt import aux_hidden as ah
from models.demos.blackhole.qwen36.tt import pd_transfer
from models.demos.blackhole.qwen36.tt import spec_serving as ss
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.mtp_head import MTPHead
from models.demos.blackhole.qwen36.tt.spec_decoder import HybridHeads, RowChunkedProjector, SpecDecoder

BPU = 96  # blocks per slot = 6144 positions (a 4096-token prompt + 1024 tokens + slack)
POLICY = os.environ.get("SPEC_POLICY", "adaptive")
OSL_LONG = int(os.environ.get("ADAPT_OSL_LONG", "1024"))
OSL_SHORT = 128
RAND_LENS = [int(v) for v in os.environ.get("ADAPT_RAND_LENS", "128,1024,4096").split(",") if v]
DO_W4 = os.environ.get("ADAPT_W4", "1") == "1"
DO_REAL_W1 = os.environ.get("ADAPT_REAL_W1", "1") == "1"
OUT_JSON = os.environ.get("SPEC_OUT", f"/home/eslim/experiments/qwen36/logs/adapt/adapt_{POLICY}.json")
POLICIES = {
    "k7": ("hybrid", 7, {"QWEN36_SPEC_DFLASH2_ADAPTIVE": "0"}),
    "k3": ("hybrid", 3, {"QWEN36_SPEC_DFLASH2_ADAPTIVE": "0"}),
    "mtp_mode": (
        "hybrid",
        7,
        {"QWEN36_SPEC_DFLASH2_ADAPTIVE": "0", "QWEN36_SPEC_DFLASH2_MAX_CTX": "1", "QWEN36_SPEC_DFLASH2_CTX_HYST": "0"},
    ),
    "mtp": ("mtp", 3, {}),
    "adaptive": ("hybrid", 7, {"QWEN36_SPEC_DFLASH2_ADAPTIVE": "1"}),
    "adaptive_df2": ("hybrid", 7, {"QWEN36_SPEC_DFLASH2_ADAPTIVE": "1", "QWEN36_SPEC_ADAPT_DRAFTER": "dflash2"}),
}


class Recorder:
    """The SpecDecoder behind the Engine, recording every step's (plan, drafter, low, accepts, times) and the DFlash2
    selector margins of the drafts each step verified."""

    def __init__(self, spec, df2_head):
        self._spec = spec
        self._df2 = df2_head
        self.steps = []
        self.margin_pairs = []  # (position, margin, accepted) of every DFlash2 draft that was verified
        self._prev_margins = None  # margins of the drafts proposed by the previous step ([w][7]) + its live mask
        self._prev_drafter = None

    def __getattr__(self, name):
        return getattr(self._spec, name)

    def step(self, tokens, positions, page_table, rows, drafts, eligible=True, flush=False):
        res = self._spec.step(tokens, positions, page_table, rows, drafts, eligible=eligible, flush=flush)
        if res is None:
            self.steps.append(None)
            self._prev_margins = None
            return None
        if self._prev_margins is not None and self._prev_drafter == "dflash2" and not res.flush:
            marg, live_prev = self._prev_margins
            for s in range(min(len(marg), res.w)):
                if live_prev[s] and rows[s] is not None and drafts[s]:
                    n = len(drafts[s])
                    for j in range(min(n, len(marg[s]))):
                        self.margin_pairs.append((j, float(marg[s][j]), int(j < res.accepts[s])))
        self.steps.append(
            dict(
                plan=str(res.plan),
                drafter=res.drafter,
                low=res.low,
                long=res.long,
                flush=res.flush,
                migrated=res.migrated,
                accepts=[int(a) for a in res.accepts],
                live=[rows[s] is not None for s in range(res.w)],
                times=dict(res.times_ms),
            )
        )
        if res.drafter == "dflash2" and self._df2 is not None:
            self._prev_margins = (
                [list(m) for m in self._df2.last_margins],
                [rows[s] is not None for s in range(res.w)],
            )
        else:
            self._prev_margins = None
        self._prev_drafter = res.drafter
        return res


def _random_prompt(length, seed):
    rng = random.Random(1_000_003 * length + seed)
    return torch.tensor([[rng.randrange(1000, 120_000) for _ in range(length)]], dtype=torch.int32)


NEARTIE_GAP = float(
    os.environ.get("ADAPT_NEARTIE_GAP", "0.5")
)  # plain-logit gap (exp - got) up to which a flip is a near-tie


def _neartie_probe_w1(model, ref1, prompt, page_tables, ref_stream, probes, prefill_fn):
    """The plain decode's logits at every divergence of ONE prompt: re-prefill it as user 0, replay the plain decode at
    width 1 TEACHER-FORCED on the width-8 reference stream (so the probed prefix is exactly the one the committed stream
    shared with the reference), read the full logits eagerly (the same decode programs) at each probed step and record,
    per (token index, got, exp): the plain logit of the reference token minus the committed token's, the top-2 gap, the
    rank of the committed token, the bf16 ulp at that magnitude -- and, for the reference side, how often the width-1
    argmax differs from the width-8 reference stream before the probed index (a width-dependent plain decode would
    show up here). ``probes`` = [(i, got, exp)], token i is produced by decode step i - 1."""
    from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
    from models.tt_transformers.tt.common import copy_host_to_device

    lens, first = prefill_fn(model, [prompt], page_tables, [0])
    assert first[0] == ref_stream[0], (first, ref_stream[:2])
    by_step = {}
    for i, got, exp in probes:
        by_step.setdefault(i - 1, []).append((i, got, exp))
    out, w1_vs_w8 = [], []
    pos = lens[0]
    comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=3)
    for t in range(max(by_step) + 1):
        cur = ref_stream[t]  # teacher forcing: the reference prefix
        if t in by_step:
            host = model.prepare_decode_inputs_host(
                torch.tensor([cur], dtype=torch.int32).reshape(1, 1),
                torch.tensor([pos], dtype=torch.int32),
                page_tables[:1],
            )
            copy_host_to_device(host_tensors=host, device_tensors=ref1.dev)
            cos, sin = unpack_rope(ref1.dev[2])
            logits = model._forward_decode(ref1.dev[0], cos, sin, ref1.dev[1], ref1.dev[3], sharded_lm_head=True)
            ttnn.synchronize_device(model.mesh_device)
            row = ttnn.to_torch(logits, mesh_composer=comp).float().reshape(-1, model.vocab_size)[0]
            ttnn.deallocate(logits)
            nxt = int(row.argmax())
            for i, got, exp in by_step[t]:
                top2 = torch.topk(row, 2).values
                mag = max(abs(float(row[exp])), abs(float(row[got])), 1e-9)
                ulp = 2.0 ** (math.floor(math.log2(mag)) - 7)  # bf16: 8 significand bits
                out.append(
                    {
                        "token_index": i,
                        "got": got,
                        "exp": exp,
                        "w1_eager_argmax": nxt,
                        "logit_exp": float(row[exp]),
                        "logit_got": float(row[got]),
                        "gap": float(row[exp] - row[got]),
                        "gap_ulps": float(row[exp] - row[got]) / ulp,
                        "top2_gap": float(top2[0] - top2[1]),
                        "rank_got": int((row > row[got]).sum()),
                        "w1_vs_w8_disagreements_before": len(w1_vs_w8),
                    }
                )
        else:
            nxt = ref1.step([cur], [pos])[0]
        if nxt != ref_stream[t + 1]:
            w1_vs_w8.append((t + 1, nxt, ref_stream[t + 1]))
        pos += 1
    return out, w1_vs_w8


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_adapt(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    assert POLICY in POLICIES, f"SPEC_POLICY={POLICY}: expected one of {sorted(POLICIES)}"
    drafter, K, pol_env = POLICIES[POLICY]
    env = {k: v for k, v in os.environ.items() if k.startswith("QWEN36_SPEC_")}
    env.update(pol_env)
    device = mesh_device
    device.enable_program_cache()
    _pin_aiclk(AICLK_MHZ)
    results = {"policy": POLICY, "drafter": drafter, "k": K, "env": env, "items": [], "aiclk_mhz": AICLK_MHZ}
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    logger.info(
        f"[adapt] model load {time.perf_counter() - t0:.1f}s policy={POLICY} drafter={drafter} K={K} env={pol_env}"
    )
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = build_prompts(tok)
    prompt_ids = [tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(torch.int32) for _, p in prompts]
    types = [t for t, _ in prompts]
    n_real = len(prompt_ids)
    rand_idx = {}  # (length, seed) -> prompt index
    for L in RAND_LENS:
        for seed in range(4 if DO_W4 else 1):
            rand_idx[(L, seed)] = len(prompt_ids)
            prompt_ids.append(_random_prompt(L, seed))
            types.append(f"rand{L}")
    # the workload: (name, prompt indices, tokens per user)
    items = []
    if DO_REAL_W1:
        items += [(f"w1/real/{types[p]}{p}", [p], OSL_SHORT) for p in range(n_real)]
    for L in RAND_LENS:
        items.append((f"w1/rand{L}/o{OSL_SHORT}", [rand_idx[(L, 0)]], OSL_SHORT))
        items.append((f"w1/rand{L}/o{OSL_LONG}", [rand_idx[(L, 0)]], OSL_LONG))
    if DO_W4:
        items.append(("w4/real/gsm8k", [3, 4, 5, 6], OSL_SHORT))
        items.append(("w4/real/mixed", [0, 7, 11, 12], OSL_SHORT))
        for L in RAND_LENS:
            idx = [rand_idx[(L, s)] for s in range(4)]
            items.append((f"w4/rand{L}/o{OSL_SHORT}", idx, OSL_SHORT))
            items.append((f"w4/rand{L}/o{OSL_LONG}", idx, OSL_LONG))
    need = {}  # prompt index -> reference length
    for _, idx, n in items:
        for p in idx:
            need[p] = max(need.get(p, 0), n + 16)
    for p, n in need.items():
        assert prompt_ids[p].shape[1] + n + 8 <= BPU * BLOCK_SIZE, (p, prompt_ids[p].shape, n)
    # prompts above the chunk are prefilled in 2048-token chunks (the hooks see CHUNK segments), the rest masked
    buckets = sorted(
        set(Qwen36Model._mask_bucket_for(int(prompt_ids[p].shape[1])) for p in need if prompt_ids[p].shape[1] <= CHUNK)
        | {CHUNK}
    )
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    uses_df2 = drafter in ("dflash2", "hybrid")
    uses_mtp = drafter in ("mtp", "hybrid")
    spec = None
    refs = {}
    hook = None
    try:
        df2_head = mtp_head = None
        if uses_df2:
            from models.demos.blackhole.qwen36.tt.dflash2_head import DFlash2Drafter

            df2_head = DFlash2Drafter(model, page_tables=None, widths=())
            model.prefill_aux_layers = ah.aux_layers_for(model)
        if uses_mtp:
            mtp_head = MTPHead(model, page_tables=None, widths=(), buckets=buckets, sdpa_pt_blocks=BPU)
        head = HybridHeads(mtp_head, df2_head) if drafter == "hybrid" else (df2_head or mtp_head)
        ladder = ss.Ladder.from_env(drafter, K, BMAX, env)
        results["ladder"] = [f"{p}:{ladder.drafter_for(p)}" for p in ladder.plans]
        results["low_ladder"] = [f"{p}:{ladder.drafter_for(p, low=True)}" for p in ladder.low_plans]
        results["adapt"] = None if ladder.adapt is None else ladder.adapt.__dict__
        spec_dec = SpecDecoder(model, head, ladder, BMAX, page_tables.shape[1], log_every=100)
        spec = Recorder(spec_dec, df2_head)
        for w in (
            1,
            8,
            32,
        ):  # 1: the near-tie probe's eager replay; 8: the reference streams; 32: the Engine's plain steps
            refs[w] = DecodeRef(model, w, page_tables[:w])
            refs[w].compile()
        spec_dec.compile()
        if uses_df2:
            hook = ah.DFlash2ContextPrefillHook(model, RowChunkedProjector(df2_head.projector), block_size=BLOCK_SIZE)
            hook.compile(buckets)
            pd_transfer.kv_group_import_warmup(model, 128)
        if uses_mtp:
            for b in buckets:
                mtp_head.compile_prefill(b)
        ttnn.synchronize_device(device)
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
        if uses_df2:
            model.prefill_aux_hook = hook
        if uses_mtp:
            model.prefill_hidden_hook = mtp_head.prefill_hook

        def _clear_state():
            if uses_df2:
                ah.kv_group_stage_store(model).clear()
            if uses_mtp:
                mtp_head.pending_rows.clear()

        # one warm prefill per masked bucket / length class (the eager bucket programs), the group imports per block
        # count, then the captures
        seen_b = set()
        for p in sorted(need, key=lambda p: prompt_ids[p].shape[1]):
            b = Qwen36Model._mask_bucket_for(int(prompt_ids[p].shape[1]))
            if b in seen_b:
                continue
            seen_b.add(b)
            _prefill(model, [prompt_ids[p]], page_tables, [0])
            _clear_state()
        if uses_df2:
            for nblk in sorted(set(-(-int(prompt_ids[p].shape[1]) // BLOCK_SIZE) for p in need)):
                z = torch.zeros(nblk, ah.DFLASH2_KV_HEADS, BLOCK_SIZE, ah.DFLASH2_HEAD_DIM, dtype=torch.bfloat16)
                meta = {
                    "n_layers": ah.DFLASH2_N_LAYERS,
                    "kv_heads": ah.DFLASH2_KV_HEADS,
                    "head_dim": ah.DFLASH2_HEAD_DIM,
                    "block_size": BLOCK_SIZE,
                    "block_index": list(range(nblk)),
                    "first_pos": 0,
                    "n_tokens": nblk * BLOCK_SIZE,
                }
                pd_transfer.import_kv_groups(
                    model, page_tables[0].tolist(), {"dflash2": ([(z, z)] * ah.DFLASH2_N_LAYERS, meta)}
                )
            ttnn.synchronize_device(device)
        for r in refs.values():
            r.capture()
        spec_dec.capture()
        logger.info(
            f"[adapt] warm-up done {time.perf_counter() - t0:.1f}s; ladder {results['ladder']} low {results['low_ladder']}"
        )

        # --- plain reference streams (width 8 batches; the plain decode is width independent: test_spec_serving) ---
        ref_streams = {}
        order = sorted(need, key=lambda p: -need[p])
        t_ref = time.perf_counter()
        for g0 in range(0, len(order), 8):
            group = order[g0 : g0 + 8]
            ids = [prompt_ids[p] for p in group]
            users = list(range(len(ids)))
            lens, first = _prefill(model, ids, page_tables, users)
            _clear_state()
            pos, cur = list(lens), list(first)
            streams = [[t] for t in first]
            n_ref = max(need[p] for p in group)
            for _ in range(n_ref):
                toks = cur + [0] * (8 - len(cur))
                ps = pos + [-1] * (8 - len(pos))
                nxt = refs[8].step(toks, ps)[: len(cur)]
                for s in range(len(cur)):
                    streams[s].append(nxt[s])
                pos = [p + 1 for p in pos]
                cur = nxt
            for s, p in enumerate(group):
                ref_streams[p] = streams[s]
        logger.info(f"[adapt] {len(ref_streams)} reference streams in {time.perf_counter() - t_ref:.1f}s")
        results["random_text_sample"] = {f"rand{L}": tok.decode(ref_streams[rand_idx[(L, 0)]][1:41]) for L in RAND_LENS}
        logger.info(f"[adapt] what the model writes after a random prompt: {results['random_text_sample']}")

        # --- the workload ---
        all_mism = []
        for name, idx, n_tok in items:
            eng = Engine(model, spec, refs[32], page_tables, prompt_ids, _prefill, drafter=drafter)
            stats0 = dict(spec_dec.state.stats)
            n0 = len(spec.steps)
            for i, p in enumerate(idx):
                eng.submit(f"{name}#{i}", p, n_tok)
            t_run = time.perf_counter()
            n_steps = 0
            while eng.pending_join or eng.live_count() > 0:
                eng.step()
                n_steps += 1
                assert n_steps < 20 * n_tok + 50, name
            wall = time.perf_counter() - t_run
            steps = spec.steps[n0:]
            spec_ms = sum(e["ms"] for e in eng.log)
            tokens = sum(e["tokens"] for e in eng.log)
            user_steps = sum(len(e["live"]) for e in eng.log)
            by_mode = {}
            for e, rec in zip(eng.log, steps):
                key = (
                    "plain"
                    if rec is None
                    else f"{rec['plan']}:{rec['drafter']}{'/low' if rec['low'] else ''}{'/long' if rec['long'] else ''}{'/flush' if rec['flush'] else ''}"
                )
                d = by_mode.setdefault(key, {"n": 0, "ms": 0.0, "tokens": 0, "users": 0, "accepted": 0})
                d["n"] += 1
                d["ms"] += e["ms"]
                d["tokens"] += e["tokens"]
                d["users"] += len(e["live"])
                if rec is not None:
                    d["accepted"] += sum(a for a, lv in zip(rec["accepts"], rec["live"]) if lv)
            mism = []
            for rid, r in eng.req.items():
                ref = ref_streams[r["prompt"]]
                i_bad, _ = _stream_compare(r["stream"], ref)
                if i_bad is not None:
                    mism.append((rid, i_bad, r["stream"][i_bad], ref[i_bad]))
                    all_mism.append((name, rid, i_bad, r["stream"][i_bad], ref[i_bad]))
            st = spec_dec.state.stats
            dstats = {k: st[k] - stats0.get(k, 0) for k in st}
            item = {
                "name": name,
                "w": len(idx),
                "prompts": idx,
                "prompt_lens": [int(prompt_ids[p].shape[1]) for p in idx],
                "tokens_per_user": n_tok,
                "steps": len(eng.log),
                "wall_s": wall,
                "step_ms_sum": spec_ms,
                "tokens": tokens,
                "ms_per_token": spec_ms / max(1, tokens),
                "tok_s": 1e3 * tokens / max(1e-9, spec_ms),
                "tok_per_user_step": tokens / max(1, user_steps),
                "by_mode": by_mode,
                "stats": dstats,
                "mismatch": mism,
                "text0": tok.decode(next(iter(eng.req.values()))["stream"][:24]),
            }
            results["items"].append(item)
            modes_txt = ", ".join(
                f"{k}: {d['n']} steps {d['ms'] / d['n']:.1f} ms {d['tokens'] / max(1, d['users']):.2f} tok/u/step"
                for k, d in sorted(by_mode.items(), key=lambda kv: -kv[1]["n"])
            )
            logger.info(
                f"[adapt] {POLICY} {name}: {item['steps']} steps, {tokens} tokens, {item['ms_per_token']:.2f} ms/token, "
                f"{item['tok_s']:.1f} tok/s, {item['tok_per_user_step']:.2f} tok/user/step; exact={not mism} {mism[:2]}; "
                f"adapt switches {dstats.get('adapt_switches', 0)} (left {dstats.get('adapt_left', 0)}, probes "
                f"{dstats.get('adapt_probes', 0)}, flushes {dstats.get('adapt_flushes', 0)}, deferred "
                f"{dstats.get('adapt_deferred', 0)}); flushes {dstats['flushes']} migrations {dstats['migrations']}; "
                f"modes: {modes_txt}"
            )
        results["mismatch"] = all_mism
        results["exact"] = not all_mism
        results["stats_total"] = dict(spec_dec.state.stats)
        # --- near-tie probe of every divergence (the plain decode's logits there; reference side: w1 vs w8 plain) ---
        # A divergence is near-tie-bounded when the plain decode's logit of its own token exceeds the committed token's
        # by <= NEARTIE_GAP (DFLASH2_RESULTS.md 9.4: 0.5 = 4 bf16 ulps at |logit| 16..32 observed at >= 3k context;
        # the R = 64 fractured flips are <= 0.25) and the committed token is the plain decode's runner-up (rank <= 1).
        uniq = {}
        for name, rid, i_bad, got, exp in all_mism:
            p = None
            for it in results["items"]:
                if it["name"] == name:
                    p = it["prompts"][int(rid.rsplit("#", 1)[1])]
            uniq.setdefault((p, i_bad, got, exp), []).append((name, rid))
        neartie, w1w8 = [], {}
        t_probe = time.perf_counter()
        for p in sorted(set(k[0] for k in uniq)):
            probes = sorted((i, got, exp) for (pp, i, got, exp) in uniq if pp == p)
            rows_, dis = _neartie_probe_w1(model, refs[1], prompt_ids[p], page_tables, ref_streams[p], probes, _prefill)
            _clear_state()
            for row_ in rows_:
                row_["prompt"] = p
                row_["prompt_len"] = int(prompt_ids[p].shape[1])
                row_["items"] = uniq[(p, row_["token_index"], row_["got"], row_["exp"])]
                row_["neartie"] = row_["gap"] <= NEARTIE_GAP and row_["rank_got"] <= 1
            neartie.extend(rows_)
            w1w8[p] = dis
            logger.info(
                f"[adapt] near-tie probe prompt {p} ({int(prompt_ids[p].shape[1])} tokens): {rows_}; w1-vs-w8 plain disagreements {dis[:5]}"
            )
        results["neartie"] = neartie
        results["neartie_gap"] = NEARTIE_GAP
        results["w1_vs_w8_plain"] = {str(p): d for p, d in w1w8.items()}
        results["neartie_bounded"] = all(r["neartie"] for r in neartie)
        logger.info(
            f"[adapt] near-tie probe {time.perf_counter() - t_probe:.1f}s: {len(neartie)} unique divergences, "
            f"{sum(r['neartie'] for r in neartie)} near-tie-bounded (gap <= {NEARTIE_GAP}, rank <= 1); "
            f"w1-vs-w8 plain disagreements {sum(len(d) for d in w1w8.values())}"
        )
        # --- selector margin vs acceptance (DFlash2 drafts) ---
        pairs = spec.margin_pairs
        if pairs:
            by_pos = {}
            for j, m, a in pairs:
                d = by_pos.setdefault(j, {"n": 0, "acc": 0, "m_acc": 0.0, "m_rej": 0.0, "n_rej": 0})
                d["n"] += 1
                d["acc"] += a
                if a:
                    d["m_acc"] += m
                else:
                    d["m_rej"] += m
                    d["n_rej"] += 1
            margins = sorted(m for _, m, _ in pairs)
            q = [margins[int(len(margins) * f)] for f in (0.25, 0.5, 0.75)]

            def rate(lo, hi):
                sel = [a for _, m, a in pairs if lo <= m < hi]
                return (len(sel), sum(sel) / max(1, len(sel)))

            results["margin"] = {
                "n": len(pairs),
                "by_position": {
                    j: {
                        "n": d["n"],
                        "accept_rate": d["acc"] / d["n"],
                        "mean_margin_accepted": d["m_acc"] / max(1, d["acc"]),
                        "mean_margin_rejected": d["m_rej"] / max(1, d["n_rej"]),
                    }
                    for j, d in sorted(by_pos.items())
                },
                "quartiles": q,
                "accept_rate_by_margin_quartile": [
                    rate(-1e9, q[0]),
                    rate(q[0], q[1]),
                    rate(q[1], q[2]),
                    rate(q[2], 1e9),
                ],
            }
            logger.info(f"[adapt] selector margin vs acceptance: {results['margin']}")
    finally:
        model.prefill_hidden_hook = None
        model.prefill_aux_hook = None
        model.prefill_aux_layers = ()
        for r in refs.values():
            r.release()
        if spec is not None:
            spec._spec.release()
        model.mtp_head = None
        model.dflash2_drafter = None
        model.pd_kv_group_stage = None
        model.pd_kv_groups = None
        model.free_kv_caches()
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=1, default=str)
        logger.info(f"[adapt] results -> {OUT_JSON}")
    for it in results["items"]:
        print(
            f"ADAPT {POLICY} {it['name']:24s} w={it['w']} steps={it['steps']:4d} tokens={it['tokens']:5d} "
            f"ms/tok={it['ms_per_token']:6.2f} tok/s={it['tok_s']:7.1f} tok/u/step={it['tok_per_user_step']:.2f} "
            f"switches={it['stats'].get('adapt_switches', 0)} exact={not it['mismatch']}"
        )
    for r in results.get("neartie", []):
        print(
            f"ADAPT_NEARTIE {POLICY} prompt={r['prompt']} len={r['prompt_len']} idx={r['token_index']} got={r['got']} exp={r['exp']} "
            f"gap={r['gap']:.3f} ({r['gap_ulps']:.0f} ulp) top2_gap={r['top2_gap']:.3f} rank_got={r['rank_got']} "
            f"w1_argmax={r['w1_eager_argmax']} w1_vs_w8_before={r['w1_vs_w8_disagreements_before']} neartie={r['neartie']} items={len(r['items'])}"
        )
    print(
        f"ADAPT_TOTAL {POLICY} exact={results['exact']} neartie_bounded={results.get('neartie_bounded')} "
        f"divergences={len(all_mism)} unique={len(results.get('neartie', []))} "
        f"w1_vs_w8_plain={sum(len(d) for d in results.get('w1_vs_w8_plain', {}).values())} stats={results['stats_total']}"
    )
    assert results[
        "neartie_bounded"
    ], f"committed streams != plain decode beyond greedy near-ties: {results['neartie'][:5]}"
