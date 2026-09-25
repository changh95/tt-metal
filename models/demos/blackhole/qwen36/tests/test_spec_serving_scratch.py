# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device, half B): the SERVED speculative-decoding loop (tt/spec_decoder.py + tt/spec_serving.py) driven
like the D engine drives it -- users joining and leaving decode slots, bucket / T changes (qkv_prev migration), the
admission-hold flush protocol, padding users, catch-up draft steps from the prefill's hidden row, plain-forced steps
(a non-greedy user in the batch) and plain decode above the ladder -- against the plain traced decode of every user
(DecodeRef at width 32 with -1 pad rows = the served decode). Every committed stream must be BITWISE the plain one.

  scripts/mtp_spec_run.sh spec_serving1 TEST=models/demos/blackhole/qwen36/tests/test_spec_serving_scratch.py
Env: SPEC_MIN_TOKENS (24), SPEC_LADDER (QWEN36_SPEC_LADDER default), SPEC_K (3), SPEC_OUT (json path).
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
from models.demos.blackhole.qwen36.tests.test_mtp_spec_scratch import _prefill, build_prompts
from models.demos.blackhole.qwen36.tests.test_verify_step_scratch import (
    AICLK_MHZ,
    DecodeRef,
    _pin_aiclk,
    _stream_compare,
)
from models.demos.blackhole.qwen36.tt import spec_serving as ss
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.mtp_head import MTPHead
from models.demos.blackhole.qwen36.tt.spec_decoder import SpecDecoder

BMAX = 32
BPU = 8  # blocks per slot = 512 positions
CHUNK = 2048
MIN_TOKENS = int(os.environ.get("SPEC_MIN_TOKENS", "24"))
K = int(os.environ.get("SPEC_K", "3"))
OUT_JSON = os.environ.get("SPEC_OUT", "/home/eslim/experiments/qwen36/logs/spec_serving_result.json")


class Engine:
    """The D engine over the device: stable slots, the scheduler's hold protocol, the runner's step (spec or plain)."""

    def __init__(self, model, spec, plain_ref, page_tables, prompt_ids, prefill_fn):
        self.model, self.spec, self.plain, self.pt, self.prompt_ids, self.prefill = (
            model,
            spec,
            plain_ref,
            page_tables,
            prompt_ids,
            prefill_fn,
        )
        self.owner = [None] * BMAX
        self.req = {}
        self.pending_join = []
        self.log = []

    def submit(self, rid, prompt_idx, want, greedy=True):
        self.req[rid] = dict(prompt=prompt_idx, want=want, greedy=greedy, stream=None, drafts=[], pos=None, last=None)
        self.pending_join.append(rid)

    def live_count(self):
        return sum(o is not None for o in self.owner)

    def _admit(self, rids):
        """The prefill step of the joining requests (into their slots; the MTP hook stores their hidden rows)."""
        slots = []
        for rid in rids:
            slot = self.owner.index(None)
            self.owner[slot] = rid
            slots.append(slot)
        lens, first = self.prefill(self.model, [self.prompt_ids[self.req[r]["prompt"]] for r in rids], self.pt, slots)
        for rid, n, t in zip(rids, lens, first):
            r = self.req[rid]
            r["stream"], r["pos"], r["last"], r["drafts"] = [t], n, t, []

    def step(self, forbid_hold=False):
        hold = self.spec.hold_info()
        ready = self.pending_join[: self.owner.count(None)]
        flush = False
        if ready and hold.pending_any and not forbid_hold and self.live_count() > 0:
            forces_plain = any(not self.req[r]["greedy"] for r in ready)
            crossing = hold.slots_before_crossing is not None and len(ready) > hold.slots_before_crossing
            if forces_plain or crossing:
                flush, ready = True, []
        if ready:
            for r in ready:
                self.pending_join.remove(r)
            self._admit(ready)
        assert self.live_count() > 0
        eligible = all(self.req[o]["greedy"] for o in self.owner if o is not None)
        rows = list(self.owner)
        tokens = torch.zeros(BMAX, 1, dtype=torch.int32)
        positions = torch.full((BMAX,), -1, dtype=torch.int32)
        pt = torch.zeros(BMAX, self.pt.shape[1], dtype=torch.int32)
        drafts = [None] * BMAX
        for s, o in enumerate(rows):
            if o is None:
                continue
            r = self.req[o]
            tokens[s, 0], positions[s], pt[s] = r["last"], r["pos"], self.pt[s]
            drafts[s] = list(r["drafts"])
        t0 = time.perf_counter()
        res = self.spec.step(tokens, positions, pt, rows, drafts, eligible=eligible, flush=flush)
        committed = {}
        if res is None:
            nxt = self.plain.step(tokens.reshape(-1).tolist(), positions.tolist())  # the served width-32 decode
            for s, o in enumerate(rows):
                if o is not None:
                    committed[s] = [nxt[s]]
            mode = "plain"
        else:
            for s in range(res.w):
                if rows[s] is not None:
                    committed[s] = res.committed[s]
                    assert 1 <= len(committed[s]) <= 1 + len(drafts[s] or []), (s, committed[s], drafts[s])
            mode = f"spec{res.plan}{' flush' if res.flush else ''}{' migrate' if res.migrated else ''}"
        wall = time.perf_counter() - t0
        for s, toks in committed.items():
            r = self.req[rows[s]]
            r["stream"].extend(toks)
            r["pos"] += len(toks)
            r["last"] = toks[-1]
            r["drafts"] = list(res.next_drafts[s]) if res is not None else []
        for s in range(BMAX):
            o = self.owner[s]
            if o is not None and len(self.req[o]["stream"]) - 1 >= self.req[o]["want"]:
                self.owner[s] = None
        self.log.append(
            dict(mode=mode, live=sorted(committed), ms=1e3 * wall, tokens=sum(len(t) for t in committed.values()))
        )
        return res


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_spec_serving(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    _pin_aiclk(AICLK_MHZ)
    results = {"scenario": [], "exact": None}
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    logger.info(f"[spec-test] model load {time.perf_counter() - t0:.1f}s")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = build_prompts(tok)
    prompt_ids = [tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(torch.int32) for _, p in prompts]
    for ids in prompt_ids:
        assert ids.shape[1] + MIN_TOKENS + 4 * (K + 1) + 8 <= BPU * BLOCK_SIZE
    buckets = sorted(set(Qwen36Model._mask_bucket_for(int(i.shape[1])) for i in prompt_ids))
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    spec = None
    refs = {}
    try:
        # --- persistent buffers before any capture: the head (prefill-only, as the served allocate), the decoder ---
        head = MTPHead(model, page_tables=None, widths=(), buckets=buckets, sdpa_pt_blocks=32)
        ladder = ss.Ladder.from_spec(os.environ.get("SPEC_LADDER", ss.DEFAULT_LADDER), K, BMAX)
        spec = SpecDecoder(model, head, ladder, BMAX, page_tables.shape[1], log_every=20)
        # --- compile everything first ---
        for w in (1, 8, 32):
            refs[w] = DecodeRef(model, w, page_tables[:w])
            refs[w].compile()
        spec.compile()
        for b in buckets:
            head.compile_prefill(b)
        ttnn.synchronize_device(device)
        # --- prefill warm-up as served, hook installed, warm prefill ---
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
        model.prefill_hidden_hook = head.prefill_hook
        _prefill(model, [prompt_ids[0]], page_tables, [0])
        head.pending_rows.clear()
        # --- captures ---
        for r in refs.values():
            r.capture()
        spec.capture()

        # --- plain reference streams per prompt (width 8, two batches) + width independence 1 vs 8 vs 32 ---
        n_ref = MIN_TOKENS + 4 * (K + 1) + 4
        ref_streams = {}
        for g0 in range(0, len(prompt_ids), 8):
            ids = prompt_ids[g0 : g0 + 8]
            users = list(range(len(ids)))
            lens, first = _prefill(model, ids, page_tables, users)
            head.pending_rows.clear()
            pos, cur = list(lens), list(first)
            streams = [[t] for t in first]
            # width 8 (pad to 8 users when fewer)
            for _ in range(n_ref):
                toks = cur + [0] * (8 - len(cur))
                ps = pos + [-1] * (8 - len(pos))
                nxt = refs[8].step(toks, ps)[: len(cur)]
                for s in range(len(cur)):
                    streams[s].append(nxt[s])
                pos = [p + 1 for p in pos]
                cur = nxt
            for s, p in enumerate(range(g0, g0 + len(ids))):
                ref_streams[p] = streams[s]
        lens, first = _prefill(model, [prompt_ids[0]], page_tables, [0])
        head.pending_rows.clear()
        s1, pos, cur = [first[0]], lens[0], first[0]
        for _ in range(n_ref):
            nxt = refs[1].step([cur], [pos])[0]
            s1.append(nxt)
            pos += 1
            cur = nxt
        lens, first = _prefill(model, [prompt_ids[0]], page_tables, [0])
        head.pending_rows.clear()
        s32, pos, cur = [first[0]], lens[0], first[0]
        for _ in range(n_ref):
            nxt = refs[32].step([cur] + [0] * 31, [pos] + [-1] * 31)[0]
            s32.append(nxt)
            pos += 1
            cur = nxt
        width_indep = s1 == ref_streams[0] == s32
        logger.info(f"[spec-test] plain decode width independence (1 vs 8 vs 32) for prompt 0: {width_indep}")
        results["width_independent"] = width_indep

        # --- the scenario: 3 -> 9 -> 17 users (bucket changes, the 8->9 band-up hold + flush, plain above 16), a
        # non-greedy user forcing plain steps, then the drain (band-down migrations) ---
        eng = Engine(model, spec, refs[32], page_tables, prompt_ids, _prefill)
        rng = random.Random(7)
        n_prompts = len(prompt_ids)
        waves = [(0, 3), (3, 6), (7, 8), (30, 2)]
        i = 0
        submitted = {}
        for at, cnt in waves:
            for _ in range(cnt):
                submitted.setdefault(at, []).append(
                    (f"r{i}", i % n_prompts, rng.randrange(MIN_TOKENS, MIN_TOKENS + 12), True)
                )
                i += 1
        submitted.setdefault(16, []).append(("sampled", 1, 6, False))  # forces plain steps while present
        step = 0
        t_all = time.perf_counter()
        while submitted or eng.pending_join or eng.live_count() > 0:
            for rid, p, want, greedy in submitted.pop(step, []):
                eng.submit(rid, p, want, greedy)
            if eng.live_count() == 0 and not eng.pending_join:
                step += 1
                continue
            eng.step()
            step += 1
            assert step < 400
        wall = time.perf_counter() - t_all
        mism = []
        for rid, r in eng.req.items():
            ref = ref_streams[r["prompt"]]
            i_bad, n = _stream_compare(r["stream"], ref)
            if i_bad is not None:
                mism.append((rid, i_bad, r["stream"][i_bad], ref[i_bad]))
        st = spec.state.stats
        modes = [e["mode"] for e in eng.log]
        by_mode = {}
        for e in eng.log:
            d = by_mode.setdefault(e["mode"], {"n": 0, "ms": 0.0, "tokens": 0, "users": 0})
            d["n"] += 1
            d["ms"] += e["ms"]
            d["tokens"] += e["tokens"]
            d["users"] += len(e["live"])
        for m, d in sorted(by_mode.items()):
            logger.info(
                f"[spec-test]   {m:28s} steps {d['n']:3d}  {d['ms'] / d['n']:6.1f} ms/step  "
                f"{d['tokens'] / max(1, d['users']):.2f} tok/user/step  ({d['users'] / d['n']:.1f} users)"
            )
        results["by_mode"] = by_mode
        results["scenario"] = eng.log
        results["stats"] = st
        results["mismatch"] = mism
        results["exact"] = not mism
        tot_tokens = sum(e["tokens"] for e in eng.log)
        logger.info(
            f"[spec-test] scenario: {len(eng.log)} steps in {wall:.1f}s, {tot_tokens} tokens, stats {st}, "
            f"mismatches {mism[:5]}; modes: {sorted(set(modes))}"
        )
        # --- protocol violation: a crossing the scheduler did not hold must raise, never corrupt ---
        eng2 = Engine(model, spec, refs[32], page_tables, prompt_ids, _prefill)
        for j in range(8):
            eng2.submit(f"v{j}", j % n_prompts, 200)
        eng2.step()
        eng2.step()
        assert spec.state.plan == ss.Plan(8, 4) and spec.state.max_pending() >= 1
        eng2.submit("v9", 3, 200)
        raised = False
        if spec.state.max_pending() > 2:
            try:
                eng2.step(forbid_hold=True)
            except ss.SpecProtocolError:
                raised = True
        results["protocol_error_raised"] = raised
        logger.info(f"[spec-test] unheld band-up crossing raised SpecProtocolError: {raised}")
    finally:
        model.prefill_hidden_hook = None
        for r in refs.values():
            r.release()
        if spec is not None:
            spec.release()
        model.mtp_head = None
        model.free_kv_caches()
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=1, default=str)
    print(
        f"SPEC_SERVING exact={results['exact']} width_independent={results.get('width_independent')} stats={results.get('stats')}"
    )
    assert results.get("width_independent"), "plain decode differs across widths 1 / 8 / 32"
    assert results["exact"], f"committed streams != plain decode: {results['mismatch'][:5]}"
    assert (
        results["stats"]["flushes"] >= 1
        and results["stats"]["migrations"] >= 1
        and results["stats"]["plain_steps"] >= 1
    )
