# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device, half B): the SERVED speculative-decoding loop (tt/spec_decoder.py + tt/spec_serving.py) driven
like the D engine drives it -- users joining and leaving decode slots, bucket / T changes (qkv_prev migration), the
admission-hold flush protocol, padding users, catch-up draft steps from the prefill's hidden row, plain-forced steps
(a non-greedy user in the batch) and plain decode above the ladder -- against the plain traced decode of every user
(DecodeRef at width 32 with -1 pad rows = the served decode). Every committed stream must be BITWISE the plain one
for the tokens produced by R <= 32 plans; a token produced by a fractured (R > 32) plan may differ only at a greedy
near-tie of the plain decode (logit gap <= 0.25: the near-tie probe of test_mtp_spec_scratch).

Drafter (SPEC_DRAFTER): ``mtp`` = the MTP head (hidden row from the prefill hook, catch-up steps); ``dflash2`` = the
DFlash2 block drafter with the SERVED context path: the P-side prefill hook (aux_hidden.DFlash2ContextPrefillHook)
stages the prompt's context K/V, pd_transfer.export_kv_groups lays them out as the payload KV group and
pd_transfer.import_kv_groups writes them into the drafter caches at the request's blocks (what the connector's drain
does on D), then SpecDecoder.note_context (the runner's admission report). One request joins WITHOUT context (its
payload lacked the group): it must decode with no drafts and still stay bitwise. ``hybrid`` = both drafters resident
(HybridHeads), both prefill hooks installed, both state paths taken per admission, the ladder switching drafters by
width: the scenario ramps 3 -> 5 -> 9 -> 17 users (the 4 -> 5 DFlash2 -> MTP crossing, the 8 -> 9 MTP band-up, plain
above 16) and drains back (5 -> 4: MTP -> DFlash2 by migration); every stream must be bitwise the plain decode.

  scripts/spec_scenario_run.sh B spec_serving_hybrid1 SPEC_DRAFTER=hybrid

  scripts/mtp_spec_run.sh spec_serving1 TEST=models/demos/blackhole/qwen36/tests/test_spec_serving_scratch.py
  scripts/mtp_spec_run.sh spec_serving_df2 SPEC_DRAFTER=dflash2 TEST=...test_spec_serving_scratch.py
Env: SPEC_DRAFTER (mtp), SPEC_MIN_TOKENS (24), SPEC_LADDER (the drafter's default), SPEC_ALLOW_FRACTURED (unset / 0 / 1
= QWEN36_SPEC_ALLOW_FRACTURED), SPEC_K (3 mtp / 7 dflash2), SPEC_NO_CONTEXT_REQ (r5), SPEC_OUT (json path),
SPEC_DFLASH2_MAX_CTX / SPEC_DFLASH2_CTX_HYST (= QWEN36_SPEC_DFLASH2_MAX_CTX / _CTX_HYST: the hybrid context rule; a
small limit such as 100 / 20 makes the scenario's users grow past it mid-generation -> the (w,8) DFlash2 -> (w,4) MTP
switches with the state machine's own flush, and the way back when the long users leave; every stream bitwise).
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
from models.demos.blackhole.qwen36.tests.test_mtp_spec_scratch import _neartie_probe, _prefill, build_prompts
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

BMAX = 32
BPU = 8  # blocks per slot = 512 positions
CHUNK = 2048
DRAFTER = os.environ.get("SPEC_DRAFTER", "mtp")
MIN_TOKENS = int(os.environ.get("SPEC_MIN_TOKENS", "24"))
K = int(os.environ.get("SPEC_K", "7" if DRAFTER in ("dflash2", "hybrid") else "3"))
USES_DFLASH2 = DRAFTER in ("dflash2", "hybrid")
USES_MTP = DRAFTER in ("mtp", "hybrid")
NO_CONTEXT_REQ = os.environ.get("SPEC_NO_CONTEXT_REQ", "r5")  # dflash2: this request joins without context K/V
OUT_JSON = os.environ.get("SPEC_OUT", "/home/eslim/experiments/qwen36/logs/spec_serving_result.json")


class Engine:
    """The D engine over the device: stable slots, the scheduler's hold protocol, the runner's step (spec or plain)."""

    def __init__(self, model, spec, plain_ref, page_tables, prompt_ids, prefill_fn, drafter="mtp"):
        self.model, self.spec, self.plain, self.pt, self.prompt_ids, self.prefill = (
            model,
            spec,
            plain_ref,
            page_tables,
            prompt_ids,
            prefill_fn,
        )
        self.drafter = drafter
        self.owner = [None] * BMAX
        self.req = {}
        self.pending_join = []
        self.log = []
        self.context_imports = []  # dflash2: (rid, slot, n_tokens, blocks, ms)

    def submit(self, rid, prompt_idx, want, greedy=True):
        self.req[rid] = dict(
            prompt=prompt_idx, want=want, greedy=greedy, stream=None, drafts=[], pos=None, last=None, src=[0]
        )
        self.pending_join.append(rid)

    def live_count(self):
        return sum(o is not None for o in self.owner)

    def _admit(self, rids):
        """The prefill step of the joining requests (into their slots; the MTP hook stores their hidden rows / the
        DFlash2 hook stages their context K/V, exported + imported as the connector would and reported per slot)."""
        slots = []
        for rid in rids:
            slot = self.owner.index(None)
            self.owner[slot] = rid
            slots.append(slot)
        lens, first = self.prefill(self.model, [self.prompt_ids[self.req[r]["prompt"]] for r in rids], self.pt, slots)
        for rid, n, t in zip(rids, lens, first):
            r = self.req[rid]
            r["stream"], r["pos"], r["last"], r["drafts"], r["src"] = [t], n, t, [], [0]
        if self.drafter in ("dflash2", "hybrid"):
            for rid, slot, n in zip(rids, slots, lens):
                block_ids = self.pt[slot].tolist()
                t0 = time.perf_counter()
                groups = pd_transfer.export_kv_groups(self.model, slot, block_ids, n)  # P side (staged by the hook)
                assert "dflash2" in groups, f"the prefill hook staged no context for slot {slot}"
                kv, meta = groups["dflash2"]
                assert meta["n_tokens"] == n and meta["first_pos"] == 0, meta
                if rid == NO_CONTEXT_REQ:
                    self.spec.note_context(slot, rid, None)  # payload without the group: decodes with no drafts
                    continue
                pd_transfer.import_kv_groups(self.model, block_ids, {"dflash2": (kv, meta)})  # D side (the drain)
                ttnn.synchronize_device(self.model.mesh_device)
                self.spec.note_context(slot, rid, meta)
                self.context_imports.append((rid, slot, n, len(meta["block_index"]), 1e3 * (time.perf_counter() - t0)))

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
            mode = f"spec{res.plan}[{res.drafter}{'/long' if res.long else ''}]{' flush' if res.flush else ''}{' migrate' if res.migrated else ''}"
        wall = time.perf_counter() - t0
        for s, toks in committed.items():
            r = self.req[rows[s]]
            r["stream"].extend(toks)
            r["src"].extend([res.plan.R if res is not None else 0] * len(toks))  # the grid rows that produced them
            r["pos"] += len(toks)
            r["last"] = toks[-1]
            r["drafts"] = list(res.next_drafts[s]) if res is not None else []
            if (
                self.drafter in ("dflash2", "hybrid")
                and rows[s] == NO_CONTEXT_REQ
                and res is not None
                and res.drafter == "dflash2"
            ):
                assert r["drafts"] == [], (rows[s], r["drafts"])  # no context -> no DFlash2 drafts, ever
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
    results = {"scenario": [], "exact": None, "drafter": DRAFTER, "k": K}
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
    hook = None
    try:
        # --- persistent buffers before any capture: the drafter (as the served allocate_kv_cache builds it), the
        # decoder (its ladder from the env knobs, as qwen36_vllm._spec_prepare) ---
        df2_head = mtp_head = None
        if USES_DFLASH2:
            from models.demos.blackhole.qwen36.tt.dflash2_head import DFlash2Drafter

            df2_head = DFlash2Drafter(model, page_tables=None, widths=())
            model.prefill_aux_layers = ah.aux_layers_for(model)  # BEFORE the prefill warm-up captures
        if USES_MTP:
            mtp_head = MTPHead(model, page_tables=None, widths=(), buckets=buckets, sdpa_pt_blocks=32)
        head = HybridHeads(mtp_head, df2_head) if DRAFTER == "hybrid" else (df2_head or mtp_head)
        ladder = ss.Ladder.for_drafter(
            DRAFTER,
            K,
            BMAX,
            spec=os.environ.get("SPEC_LADDER"),
            allow_fractured=os.environ.get("SPEC_ALLOW_FRACTURED"),
            dflash2_max_ctx=os.environ.get("SPEC_DFLASH2_MAX_CTX"),
            ctx_hysteresis=os.environ.get("SPEC_DFLASH2_CTX_HYST"),
        )
        results["ladder"] = [f"{p}:{ladder.drafter_for(p)}" for p in ladder.plans]
        results["long_ladder"] = [f"{p}:{ladder.drafter_for(p, long=True)}" for p in ladder.long_plans]
        results["ctx_rule"] = {"max_ctx": ladder.dflash2_max_ctx, "hysteresis": ladder.ctx_hysteresis}
        results["fractured_plans"] = [str(p) for p in ladder.fractured_plans]
        spec = SpecDecoder(model, head, ladder, BMAX, page_tables.shape[1], log_every=20)
        # --- compile everything first ---
        for w in (1, 8, 32):
            refs[w] = DecodeRef(model, w, page_tables[:w])
            refs[w].compile()
        spec.compile()
        if USES_DFLASH2:
            hook = ah.DFlash2ContextPrefillHook(model, RowChunkedProjector(df2_head.projector), block_size=BLOCK_SIZE)
            hook.compile(sorted(set(buckets) | {CHUNK}))
            pd_transfer.kv_group_import_warmup(model, 256)
        if USES_MTP:
            for b in buckets:
                mtp_head.compile_prefill(b)
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
        if USES_DFLASH2:
            model.prefill_aux_hook = hook
        if USES_MTP:
            model.prefill_hidden_hook = mtp_head.prefill_hook
        _prefill(model, [prompt_ids[0]], page_tables, [0])

        def _clear_state():
            """Drop drafter state a reference prefill produced (the served path consumes it per request)."""
            if USES_DFLASH2:
                store = ah.kv_group_stage_store(model)
                store.clear()
            if USES_MTP:
                mtp_head.pending_rows.clear()

        _clear_state()
        if USES_DFLASH2:
            # the group import of every block bucket a prompt needs, before the captures (the eager paged fills)
            for nblk in sorted(set(-(-int(i.shape[1]) // BLOCK_SIZE) for i in prompt_ids)):
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
            _clear_state()
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
        _clear_state()
        s1, pos, cur = [first[0]], lens[0], first[0]
        for _ in range(n_ref):
            nxt = refs[1].step([cur], [pos])[0]
            s1.append(nxt)
            pos += 1
            cur = nxt
        lens, first = _prefill(model, [prompt_ids[0]], page_tables, [0])
        _clear_state()
        s32, pos, cur = [first[0]], lens[0], first[0]
        for _ in range(n_ref):
            nxt = refs[32].step([cur] + [0] * 31, [pos] + [-1] * 31)[0]
            s32.append(nxt)
            pos += 1
            cur = nxt
        width_indep = s1 == ref_streams[0] == s32
        logger.info(f"[spec-test] plain decode width independence (1 vs 8 vs 32) for prompt 0: {width_indep}")
        results["width_independent"] = width_indep

        # --- the scenario: 3 -> 6 -> 9 -> 17 users (bucket changes incl. the widest T=8 bucket at 6 users -- the
        # fractured (8,T=8) plan of the dflash2 ladder -- the 8->9 band-up hold + flush, plain above 16), a non-greedy
        # user forcing plain steps, then the drain (band-down migrations) ---
        eng = Engine(model, spec, refs[32], page_tables, prompt_ids, _prefill, drafter=DRAFTER)
        rng = random.Random(7)
        n_prompts = len(prompt_ids)
        # hybrid: 3 -> 5 -> 9 -> 17 (the 4 -> 5 DFlash2 -> MTP crossing, then the MTP band-up, then plain) and back
        waves = (
            [(0, 3), (2, 2), (5, 4), (9, 8), (32, 2)]
            if DRAFTER == "hybrid"
            else [(0, 3), (2, 3), (5, 3), (9, 8), (32, 2)]
        )
        i = 0
        submitted = {}
        for at, cnt in waves:
            for _ in range(cnt):
                submitted.setdefault(at, []).append(
                    (f"r{i}", i % n_prompts, rng.randrange(MIN_TOKENS, MIN_TOKENS + 12), True)
                )
                i += 1
        submitted.setdefault(18, []).append(("sampled", 1, 6, False))  # forces plain steps while present
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
                mism.append((rid, i_bad, r["stream"][i_bad], ref[i_bad], r["src"][i_bad]))
        # classify: a divergence is acceptable only if the user ran a fractured (R > 32) plan at or before the token
        # (its KV / GDN state then carries the fractured path's ulp-level rounding, so a later step of any width may
        # flip a greedy near-tie) AND the plain decode's logits are a near-tie there (gap <= 0.25)
        fractured_user = {rid: max(r["src"][: i_bad + 1]) > 32 for rid, i_bad, *_ in mism for r in [eng.req[rid]]}
        bad = [m for m in mism if not fractured_user[m[0]]]
        neartie = []
        for rid, i_bad, got, exp, R in mism:
            if not fractured_user[rid]:
                continue
            p = eng.req[rid]["prompt"]
            rows_ = _neartie_probe(
                model, refs[1], 1, [prompt_ids[p]], page_tables, [0], [ref_streams[p]], [(0, i_bad, got, exp)], _prefill
            )
            _clear_state()
            for row_ in rows_:
                row_["rid"], row_["R"] = rid, R
            neartie.extend(rows_)
            logger.info(f"[spec-test] near-tie probe {rid}@{i_bad} (R={R}): {rows_}")
        bad += [x for x in neartie if x["gap"] > 0.25]
        results["mismatch_all"] = mism
        results["neartie"] = neartie
        results["context_imports"] = eng.context_imports
        results["no_context_user_steps"] = spec.n_no_context
        results["steps_by_drafter"] = {str(k): v for k, v in spec.n_steps_by_drafter.items()}
        if DRAFTER == "hybrid":
            results["hybrid_keep_steps_total"] = spec.head.mtp.stats.get("keep_steps", 0)
            results["hybrid_keep_wall_ms"] = 1e3 * spec.head.mtp.stats.get("keep_wall", 0.0)
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
        results["mismatch"] = bad
        results["exact"] = not bad
        results["bitwise"] = not mism
        tot_tokens = sum(e["tokens"] for e in eng.log)
        logger.info(
            f"[spec-test] scenario ({DRAFTER}): {len(eng.log)} steps in {wall:.1f}s, {tot_tokens} tokens, stats {st}, "
            f"divergences {mism[:5]} (rid, token, got, exp, R of the producing step; near-tie-bounded after a fractured "
            f"plan: {len(mism) - len(bad)}, BAD: {bad[:5]}); "
            f"modes: {sorted(set(modes))}; no-context user-steps {spec.n_no_context}; steps by drafter "
            f"{spec.n_steps_by_drafter}"
        )
        # --- protocol violation: a crossing the scheduler did not hold must raise, never corrupt ---
        eng2 = Engine(model, spec, refs[32], page_tables, prompt_ids, _prefill, drafter=DRAFTER)
        for j in range(8):
            eng2.submit(f"v{j}", j % n_prompts, 200)
        eng2.step()
        eng2.step()
        plan8, plan9 = ladder.plan_for(8), ladder.plan_for(9)
        assert spec.state.plan == plan8 and spec.state.max_pending() >= 1
        eng2.submit("v9", 3, 200)
        raised = False
        if spec.state.max_pending() > (plan9.k if plan9 is not None else 0):
            try:
                eng2.step(forbid_hold=True)
            except ss.SpecProtocolError:
                raised = True
        results["protocol_error_raised"] = raised
        logger.info(f"[spec-test] unheld band-up crossing raised SpecProtocolError: {raised}")
    finally:
        model.prefill_hidden_hook = None
        model.prefill_aux_hook = None
        model.prefill_aux_layers = ()
        for r in refs.values():
            r.release()
        if spec is not None:
            spec.release()
        model.mtp_head = None
        model.dflash2_drafter = None
        model.pd_kv_group_stage = None
        model.pd_kv_groups = None
        model.free_kv_caches()
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=1, default=str)
    print(
        f"SPEC_SERVING drafter={DRAFTER} ladder={results.get('ladder')} exact={results['exact']} "
        f"bitwise={results.get('bitwise')} neartie_flips={len(results.get('neartie') or [])} "
        f"width_independent={results.get('width_independent')} stats={results.get('stats')} "
        f"no_context_user_steps={results.get('no_context_user_steps')} steps_by_drafter={results.get('steps_by_drafter')}"
    )
    assert results.get("width_independent"), "plain decode differs across widths 1 / 8 / 32"
    assert results["exact"], f"committed streams != plain decode beyond fractured near-ties: {results['mismatch'][:5]}"
    if not results.get("fractured_plans"):
        assert results.get("bitwise"), f"R <= 32 ladder must be bitwise: {results.get('mismatch_all')[:5]}"
    assert (
        results["stats"]["flushes"] >= 1
        and results["stats"]["migrations"] >= 1
        and results["stats"]["plain_steps"] >= 1
    )
    if DRAFTER == "hybrid":
        assert set(results["steps_by_drafter"]) == {"mtp", "dflash2"}, results["steps_by_drafter"]
        assert results["stats"]["drafter_switches"] >= 2, results["stats"]
        if ladder.has_ctx_rule and ladder.dflash2_max_ctx < 400:
            # a limit inside the scenario's context range: users grow past it (mode changes; the long ladder's plans
            # run; the machine's own flush fires only when > 3 rows are pending at a crossing no admission hold
            # covers -- the CPU tests drive that case deterministically, here it is logged in the stats)
            assert results["stats"]["ctx_switches"] >= 1, results["stats"]
            assert any("/long" in e["mode"] for e in eng.log), "no step ran the long ladder"
