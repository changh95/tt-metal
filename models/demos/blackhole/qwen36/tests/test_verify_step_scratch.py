# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device): speculative-decoding VERIFY step -- oracle-drafter exactness + timing (milestone M2).

Per (w, T) config: prefill w users (3 prompts, user s -> prompt s % 3) with prefill_paged_slots, record the plain
greedy decode stream of every user through the served decode forward (traced, width w), re-prefill, and drive the
traced verify step (tt/verify_step.py) with three draft policies -- random tokens, the true greedy continuation
(oracle), and mixtures -- through verify_grid.VerifyController until every user has >= VERIFY_MIN_TOKENS committed
tokens. Asserts: the committed streams of the three policies are identical to each other (the accept / lazy-prefix /
KV-overwrite mechanism is draft-independent) and equal to the plain decode stream (reported per config; a mismatch is
diagnosed with the row's top-2 logit gap when it is a numerics near-tie). Then the timing table: traced verify step
(50 replays) per config vs the traced decode step at w in {1, 8, 32}, plus an eager per-section breakdown.

  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.8-27B ... pytest tests/test_verify_step_scratch.py -s
Env: VERIFY_CONFIGS ("1,8;8,8;32,4;32,8"), VERIFY_MIN_TOKENS (32), VERIFY_POLICIES ("random,oracle,mixed"),
     VERIFY_TIMING (1), VERIFY_TIMING_REPLAYS (50), VERIFY_EXACT (1), VERIFY_DECODE_WIDTHS ("1,8,32"),
     QWEN36_VERIFY_GDN_STUB (0/1: force the per-token stub instead of the multi-token kernel), VERIFY_OUT (json path).
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
from models.demos.blackhole.qwen36.tt import verify_grid as vg
from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.verify_step import VerifyStep, argmax_sharded_rows, combine_sharded_argmax
from models.tt_transformers.tt.common import copy_host_to_device

BMAX = 32
CONFIGS = [
    tuple(int(v) for v in c.split(",")) for c in os.environ.get("VERIFY_CONFIGS", "1,8;8,8;32,4;32,8").split(";") if c
]
MIN_TOKENS = int(os.environ.get("VERIFY_MIN_TOKENS", "32"))
POLICIES = [p for p in os.environ.get("VERIFY_POLICIES", "random,oracle,mixed").split(",") if p]
DO_TIMING = os.environ.get("VERIFY_TIMING", "1") == "1"
DO_EXACT = os.environ.get("VERIFY_EXACT", "1") == "1"
N_REPLAYS = int(os.environ.get("VERIFY_TIMING_REPLAYS", "50"))
DECODE_WIDTHS = [int(v) for v in os.environ.get("VERIFY_DECODE_WIDTHS", "1,8,32").split(",") if v]
DEBUG_EAGER = os.environ.get("VERIFY_DEBUG_EAGER", "0") == "1"
# VERIFY_TRACKER_AUDIT=1 (+ TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1): log-and-skip every replay
# that would corrupt a live buffer (see test_verify_w32_isolation_scratch._install_tracker_audit); the dump lists the
# programs compiled after a capture (program-cache buffers) with their allocation sites.
TRACKER_AUDIT = os.environ.get("VERIFY_TRACKER_AUDIT", "0") == "1"
OUT_JSON = os.environ.get("VERIFY_OUT", "/home/eslim/experiments/qwen36/logs/verify_step_result.json")
CHUNK = 2048
BPU = 8  # blocks per user (512 tokens: prompt <= 128 + <= ~3*MIN_TOKENS generated); multiple of 8 for the SDPA stick

PROMPTS = [
    "The capital of France is",
    "Write a short story about a robot who learns to paint. Once upon a time,",
    "Q: What is 17 * 23? Let's think step by step.\nA:",
]


class DecodeRef:
    """The served single-token decode forward at width w, traced, with the same two-stage argmax as the verify step."""

    def __init__(self, model, w, page_table):
        self.model, self.w, self.pt = model, w, page_table
        self.mesh = model.mesh_device
        self.per_shard = model.args.vocab_size // model.num_devices
        self.dev = None
        self.tid = None
        self.out = None

    def _fwd(self, dev):
        cos, sin = unpack_rope(dev[2])
        logits = self.model._forward_decode(dev[0], cos, sin, dev[1], dev[3], sharded_lm_head=True)
        idx, val = argmax_sharded_rows(logits)
        ttnn.deallocate(logits)
        return idx, val

    def compile(self):
        """Allocate the device inputs and compile the decode programs eagerly (BEFORE any trace capture: a program
        compiled after a capture owns buffers in that trace's freed range, tests/VERIFY_W32_AUDIT.md)."""
        toks = torch.full((self.w, 1), 1, dtype=torch.int32)
        pos = torch.full((self.w,), 8, dtype=torch.int32)
        self.dev = self.model.prepare_inputs_decode(toks, pos, self.pt)
        idx, val = self._fwd(self.dev)
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(idx)
        ttnn.deallocate(val)

    def capture(self):
        assert self.dev is not None, "compile() first"
        self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out = self._fwd(self.dev)
        ttnn.end_trace_capture(self.mesh, self.tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)

    def setup(self):
        self.compile()
        self.capture()

    def step(self, tokens, positions):
        host = self.model.prepare_decode_inputs_host(
            torch.tensor(tokens, dtype=torch.int32).reshape(self.w, 1),
            torch.tensor(positions, dtype=torch.int32),
            self.pt,
        )
        copy_host_to_device(host_tensors=host, device_tensors=self.dev)
        ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh)
        return combine_sharded_argmax(self.mesh, self.out[0], self.out[1], self.w, self.per_shard).tolist()

    def time_replays(self, n):
        ms = []
        for i in range(n):
            host = self.model.prepare_decode_inputs_host(
                torch.full((self.w, 1), 100 + i, dtype=torch.int32),
                torch.full((self.w,), 9 + i, dtype=torch.int32),
                self.pt,
            )
            t0 = time.perf_counter()
            copy_host_to_device(host_tensors=host, device_tensors=self.dev)
            ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
            ms.append(1e3 * (time.perf_counter() - t0))
        ms.sort()
        return ms[len(ms) // 2], ms[0]

    def release(self):
        if self.tid is not None:
            ttnn.release_trace(self.mesh, self.tid)
            self.tid = None


# Hang mitigations (2026-09-24, run verify_smoke3: the FIRST decode-trace replay after an 8-user prefill_paged_slots
# wedged chip 0 of half A -- PCIe reads 0xffffffff, board reset needed; the signature of the documented burst-prefill ->
# decode-trace SDPA wedge, root-caused to AICLK throttling steps mid-kernel and fixed in serving by pinning the clock,
# qwen36_vllm._pin_aiclk). The test pins the clock itself (VERIFY_FORCE_AICLK_MHZ, default 1200, 0 = off) and idles
# VERIFY_POST_PREFILL_SLEEP_MS (default 600) after every prefill before any trace replays.
AICLK_MHZ = int(os.environ.get("VERIFY_FORCE_AICLK_MHZ", "1200"))
POST_PREFILL_SLEEP_MS = float(os.environ.get("VERIFY_POST_PREFILL_SLEEP_MS", "600"))
PREFILL_GROUP = int(os.environ.get("VERIFY_PREFILL_GROUP", "8"))
PREFILL_GROUP_IDLE_S = float(os.environ.get("VERIFY_PREFILL_GROUP_IDLE_S", "1.0"))


def _pin_aiclk(mhz):
    if mhz <= 0:
        return
    try:
        import pyluwen

        chips = pyluwen.detect_chips()
        for chip in chips:
            chip.arc_msg(0x33, wait_for_done=True, arg0=mhz, arg1=0, timeout=2.0)  # FORCE_AICLK
        logger.info(f"[verify] pinned {len(chips)} chip(s) AICLK to {mhz} MHz (FORCE_AICLK; persists until reset / 0)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[verify] AICLK pin {mhz} MHz failed: {e!r}")


def _prefill(model, prompt_ids, page_tables, w):
    """prefill_paged_slots of users 0..w-1 (user s -> prompt s % 3); returns (positions [w], first tokens [w])."""
    ids = [prompt_ids[s % len(prompt_ids)] for s in range(w)]
    lens = [t.shape[1] for t in ids]
    # groups of <= PREFILL_GROUP users with an idle between groups: a 32-user prefill_paged_slots burst followed by a
    # trace replay wedged half A three times (the served monolithic server also prefills in small groups; D never does)
    logits = []
    for g0 in range(0, w, PREFILL_GROUP):
        g1 = min(w, g0 + PREFILL_GROUP)
        if g0 > 0:
            time.sleep(PREFILL_GROUP_IDLE_S)
        logits += model.prefill_paged_slots(ids[g0:g1], page_tables[g0:g1], list(range(g0, g1)), valid_lens=lens[g0:g1])
        ttnn.synchronize_device(model.mesh_device)
    if POST_PREFILL_SLEEP_MS > 0:
        time.sleep(POST_PREFILL_SLEEP_MS / 1e3)
    first = [int(lg.reshape(-1)[: model.vocab_size].float().argmax()) for lg in logits]
    return lens, first


def _stream_compare(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, n
    return None, n


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_verify_step(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    _pin_aiclk(AICLK_MHZ)
    if TRACKER_AUDIT:
        from models.demos.blackhole.qwen36.tests import test_verify_w32_isolation_scratch as iso

        iso._install_tracker_audit(device)
        _audit = iso._AUDIT
    results = {
        "configs": {},
        "timing": {},
        "env": {
            k: os.environ.get(k)
            for k in ("QWEN36_VERIFY_GDN_STUB", "QWEN36_DECODE_DRAM_SHARDED", "QWEN36_DECODE_FUSED_AR")
        },
    }

    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    logger.info(f"[verify] model load {time.perf_counter() - t0:.1f}s layers={len(model.layers)}")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt_ids = [tok(p, return_tensors="pt").input_ids.to(torch.int32) for p in PROMPTS]
    logger.info(f"[verify] prompt lengths {[t.shape[1] for t in prompt_ids]}")

    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    steps, refs = {}, {}
    try:
        # --- persistent verify buffers BEFORE any trace capture (trace-hazard rule) ---
        t0 = time.perf_counter()
        for w, T in CONFIGS:
            steps[(w, T)] = VerifyStep(model, w, T, page_tables[:w])
        logger.info(f"[verify] plans allocated {time.perf_counter() - t0:.1f}s")

        # --- COMPILE EVERYTHING FIRST (VERIFY_W32_AUDIT.md): every program this process will run -- the decode bodies
        # at the reference widths and the verify bodies -- is compiled eagerly BEFORE any trace capture, so no
        # program-cache buffer lands inside a parked trace's freed-intermediate range. The captures below compile
        # nothing new. (The served slot-write set is compiled by the prefill warm-up, as in the served order.) ---
        if DO_EXACT:
            for w in sorted(set(w for w, _ in CONFIGS)):
                refs[w] = DecodeRef(model, w, page_tables[:w])
                refs[w].compile()
            for (w, T), vs in steps.items():
                t0 = time.perf_counter()
                vs.compile()
                logger.info(
                    f"[verify] ({w},{T}) R={vs.plan.R} compiled in {time.perf_counter() - t0:.1f}s (before any capture)"
                )
            ttnn.synchronize_device(device)

        # --- prefill warmup (chunk trace + masked buckets + slot write + hist pack), as the served path ---
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
        logger.info(f"[verify] prefill warmup {time.perf_counter() - t0:.1f}s")
        # untimed warm prefill (lazy allocations happen here, not inside a measured / compared region)
        if TRACKER_AUDIT:
            _audit["phase"] = "warm_prefill_1user"
        _prefill(model, prompt_ids, page_tables, 1)
        if TRACKER_AUDIT:
            _audit["phase"] = "captures"

        # --- served order (qwen36_vllm: prefill warm-up -> decode trace capture), then the verify traces. Only for
        # the exactness phase: the timing phase captures/releases one trace at a time (see below) ---
        if DO_EXACT:
            for w in sorted(refs):
                refs[w].capture()
            for (w, T), vs in steps.items():
                t1 = time.perf_counter()
                vs.capture()
                logger.info(f"[verify] ({w},{T}) R={vs.plan.R} captured in {time.perf_counter() - t1:.1f}s")

        # --- exactness per config ---
        if DO_EXACT:
            for (w, T), vs in steps.items():
                k = T - 1
                rng = random.Random(1234 + w * 10 + T)
                if TRACKER_AUDIT:
                    _audit["phase"] = f"prefill_{w}users_for_reference"
                lens, first = _prefill(model, prompt_ids, page_tables, w)
                if TRACKER_AUDIT:
                    _audit["phase"] = "reference_decode_replays"
                n_ref = MIN_TOKENS + 3 * T + 2
                ref_streams = [[first[s]] for s in range(w)]
                pos = list(lens)
                cur = list(first)
                for _ in range(n_ref):
                    nxt = refs[w].step(cur, pos)
                    for s in range(w):
                        ref_streams[s].append(nxt[s])
                    pos = [p + 1 for p in pos]
                    cur = nxt
                same_prompt_ok = all(
                    ref_streams[s] == ref_streams[s % len(PROMPTS)] for s in range(w)
                )  # users of one prompt decode identically
                logger.info(
                    f"[verify] ({w},{T}) reference decode: {n_ref} steps; users of the same prompt identical={same_prompt_ok}; "
                    f"user0 text: {tok.decode(ref_streams[0][:24])!r}"
                )
                cfg_res = {"R": vs.plan.R, "ref_same_prompt_identical": same_prompt_ok, "policies": {}}
                streams_by_policy = {}
                for policy in POLICIES:
                    if TRACKER_AUDIT:
                        _audit["phase"] = f"prefill_{w}users_for_{policy}"
                    lens, first = _prefill(model, prompt_ids, page_tables, w)
                    if TRACKER_AUDIT:
                        _audit["phase"] = f"verify_replays_{policy}"
                    assert (
                        first == [ref_streams[s][0] for s in range(w)] or TRACKER_AUDIT
                    ), "prefill first token not reproducible"
                    if DEBUG_EAGER:
                        # is the state after this re-prefill good? one plain decode step must reproduce the reference
                        chk = refs[w].step(first, lens)
                        bad_c = [s_ for s_ in range(w) if chk[s_] != ref_streams[s_][1]]
                        logger.info(f"[verify-dbg] plain decode after re-prefill != reference for users {bad_c}")
                        lens, first = _prefill(model, prompt_ids, page_tables, w)
                    vs.reset_sequence()  # qkv_prev := 0 for the new batch (kernel contract: zeros at the first step)
                    if DEBUG_EAGER:
                        q0 = ttnn.to_torch(
                            ttnn.get_device_tensors(vs.plan.gdn_qkv_prev[next(iter(vs.plan.gdn_qkv_prev))])[0]
                        ).float()
                        logger.info(
                            f"[verify-dbg] qkv_prev[layer0] after reset: max|x|={q0.abs().max().item():.3g} nonfinite={int((~torch.isfinite(q0)).sum())}"
                        )
                    run_fn = vs.run
                    if DEBUG_EAGER:
                        # diagnostic: run every step eagerly too and log where the traced argmax rows differ
                        def run_fn(tokens, positions, accept_prev, _vs=vs, _w=w, _T=T, _ref=ref_streams):
                            logger.info(
                                f"[verify-dbg] inputs: positions={list(positions)} accept={list(accept_prev)} tokens[:4]={[list(t) for t in tokens[:4]]} attn={_vs.plan.attn_mode} kernel={_vs.plan.gdn_kernel is not None}"
                            )
                            if os.environ.get("VERIFY_DEBUG_ROWDIAG", "0") == "1" and len(ctrl.accept_history) == 0:
                                # per-layer row diagnostic on the FAILING flow: users of the same prompt (s % 3) must have
                                # bitwise identical T-row blocks after every layer; log the layers where one differs
                                npr = len(PROMPTS)

                                def row_check(name, x):
                                    ttnn.synchronize_device(device)
                                    h = (
                                        ttnn.to_torch(ttnn.get_device_tensors(x)[0])
                                        .float()
                                        .reshape(-1, x.shape[-1])[: _vs.plan.R]
                                    )
                                    diffs = []
                                    for s_ in range(npr, _w):
                                        b = s_ % npr
                                        # row 0 only: rows 1..k carry per-user random drafts and legitimately differ
                                        blk, base = h[s_ * _T], h[b * _T]
                                        if not torch.equal(blk, base):
                                            diffs.append(
                                                (
                                                    s_,
                                                    round(float((blk - base).abs().max()), 4),
                                                    int((blk != base).sum()),
                                                )
                                            )
                                    if diffs:
                                        logger.info(
                                            f"[verify-rowdiag] {name}: users differing from their prompt's first user: {diffs[:10]}"
                                        )
                                    else:
                                        logger.info(f"[verify-rowdiag] {name}: all same-prompt users identical")

                                def debug_head(logits, idx_t, val_t):
                                    ttnn.synchronize_device(device)
                                    nd = model.num_devices
                                    R_ = _vs.plan.R
                                    lg = ttnn.to_torch(
                                        logits, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
                                    ).float()
                                    lg = lg.reshape(nd, -1, lg.shape[-1])[:, :R_]  # [nd, R, V/nd]
                                    idxs = (
                                        ttnn.to_torch(idx_t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                                        .reshape(nd, -1)[:, :R_]
                                        .to(torch.int64)
                                    )
                                    vals = (
                                        ttnn.to_torch(val_t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                                        .float()
                                        .reshape(nd, -1)[:, :R_]
                                    )
                                    host_idx = lg.argmax(-1)
                                    host_val = lg.max(-1).values
                                    bad_idx = [
                                        (d, r)
                                        for d in range(nd)
                                        for r in range(R_)
                                        if int(host_idx[d, r]) != int(idxs[d, r])
                                    ]
                                    bad_val = [
                                        (d, r)
                                        for d in range(nd)
                                        for r in range(R_)
                                        if float(host_val[d, r]) != float(vals[d, r])
                                    ]
                                    # logits row-0 consistency across same-prompt users (rows s*T)
                                    npr_ = len(PROMPTS)
                                    lg_diff = [
                                        s_
                                        for s_ in range(npr_, _w)
                                        if not torch.equal(lg[:, s_ * _T], lg[:, (s_ % npr_) * _T])
                                    ]
                                    logger.info(
                                        f"[verify-head] logits row0 differs from prompt's first user for users {lg_diff}; "
                                        f"device argmax != host argmax at (dev,row) {bad_idx[:12]} ({len(bad_idx)} total); "
                                        f"device max != host max at {bad_val[:12]} ({len(bad_val)} total)"
                                    )
                                    per_shard_ = model.args.vocab_size // nd
                                    d_win = torch.argmax(vals, dim=0)
                                    dev_tok = (d_win * per_shard_ + idxs[d_win, torch.arange(R_)]).tolist()
                                    h_win = torch.argmax(host_val, dim=0)
                                    host_tok = (h_win * per_shard_ + host_idx[h_win, torch.arange(R_)]).tolist()
                                    logger.info(
                                        f"[verify-head] row0 tokens device={[dev_tok[s_ * _T] for s_ in range(_w)]} host={[host_tok[s_ * _T] for s_ in range(_w)]}"
                                    )

                                _vs.plan.upload(tokens, positions, accept_prev)
                                idx_d, val_d = _vs.forward(row_check=row_check, debug_head=debug_head)
                                ttnn.synchronize_device(device)
                                ttnn.deallocate(idx_d)
                                ttnn.deallocate(val_d)
                                _vs.plan._host_refs = []
                                _prefill(model, prompt_ids, page_tables, _w)
                            eager_rows = _vs.run(tokens, positions, accept_prev, eager=True)
                            logger.info(
                                f"[verify-dbg] eager row0 all users: {[int(eager_rows[vg.row(s_, 0, _T)]) for s_ in range(_w)]}"
                            )
                            for rep in range(int(os.environ.get("VERIFY_DEBUG_REPEATS", "0"))):
                                # determinism probe: re-prefill (same state) and run the same eager step again
                                _prefill(model, prompt_ids, page_tables, _w)
                                again = _vs.run(tokens, positions, accept_prev, eager=True)
                                nd = [r for r in range(_vs.plan.R) if int(again[r]) != int(eager_rows[r])]
                                logger.info(
                                    f"[verify-dbg] eager repeat {rep + 1}: rows differing from the first eager run: {len(nd)} {nd[:16]}; row0: {[int(again[vg.row(s_, 0, _T)]) for s_ in range(_w)]}"
                                )
                            _prefill(model, prompt_ids, page_tables, _w)
                            traced_rows = _vs.run(tokens, positions, accept_prev)
                            diff = [r for r in range(_vs.plan.R) if int(eager_rows[r]) != int(traced_rows[r])]
                            row0_e = [int(eager_rows[vg.row(s_, 0, _T)]) for s_ in range(_w)]
                            exp = [
                                _ref[s_][len(ctrl.committed[s_])] if len(ctrl.committed[s_]) < len(_ref[s_]) else -1
                                for s_ in range(_w)
                            ]
                            bad_e = [s_ for s_ in range(_w) if row0_e[s_] != exp[s_]]
                            logger.info(
                                f"[verify-dbg] step {len(ctrl.accept_history) + 1}: eager vs traced rows differ at {len(diff)} rows "
                                f"{diff[:12]}; eager row0 != plain decode for users {bad_e}"
                            )
                            return traced_rows

                    ctrl = vg.VerifyController(T=T, run=run_fn, positions=lens, last=first)
                    n_steps = 0
                    t_start = time.perf_counter()
                    while min(len(c) for c in ctrl.committed) < MIN_TOKENS and n_steps < 4 * MIN_TOKENS:
                        drafts = []
                        for s in range(w):
                            n_done = len(ctrl.committed[s])
                            true_next = ref_streams[s][n_done : n_done + k]
                            while len(true_next) < k:  # past the reference: any token (will be rejected)
                                true_next = true_next + [rng.randrange(model.vocab_size)]
                            if policy == "random":
                                d = [rng.randrange(model.vocab_size) for _ in range(k)]
                            elif policy == "oracle":
                                d = list(true_next)
                            else:
                                m = rng.randrange(k + 1)
                                d = list(true_next[:m]) + [rng.randrange(model.vocab_size) for _ in range(k - m)]
                            drafts.append(d)
                        ctrl.step(drafts)
                        n_steps += 1
                    wall = time.perf_counter() - t_start
                    streams_by_policy[policy] = [list(c) for c in ctrl.committed]
                    mism = []
                    for s in range(w):
                        i, n = _stream_compare(ctrl.committed[s], ref_streams[s])
                        if i is not None:
                            mism.append((s, i, ctrl.committed[s][i], ref_streams[s][i]))
                    accepted = sum(sum(a) for a in ctrl.accept_history)
                    committed_total = sum(len(c) - 1 for c in ctrl.committed)
                    logger.info(
                        f"[verify] ({w},{T}) policy={policy}: {n_steps} steps in {wall:.1f}s, committed {committed_total} tokens "
                        f"({committed_total / max(1, n_steps * w):.2f}/user/step, {accepted} drafts accepted), "
                        f"min stream {min(len(c) for c in ctrl.committed)}; mismatches vs plain decode: {len(mism)} {mism[:4]}"
                    )
                    cfg_res["policies"][policy] = {
                        "steps": n_steps,
                        "wall_s": wall,
                        "committed_total": committed_total,
                        "drafts_accepted": accepted,
                        "mismatch_vs_decode": mism,
                        "streams": [list(c) for c in ctrl.committed],
                    }
                cfg_res["ref_streams"] = ref_streams
                # mechanism: identical committed streams across policies (compare the common prefix per user)
                pols = list(streams_by_policy)
                mech_ok = True
                for s in range(w):
                    base = streams_by_policy[pols[0]][s]
                    for p in pols[1:]:
                        i, n = _stream_compare(base, streams_by_policy[p][s])
                        if i is not None:
                            mech_ok = False
                            logger.error(
                                f"[verify] ({w},{T}) MECHANISM MISMATCH user {s}: {pols[0]} vs {p} at token {i}"
                            )
                cfg_res["mechanism_identical_across_policies"] = mech_ok
                cfg_res["exact_vs_decode"] = all(not v["mismatch_vs_decode"] for v in cfg_res["policies"].values())
                results["configs"][f"{w},{T}"] = cfg_res
                logger.info(
                    f"[verify] ({w},{T}) RESULT mechanism_identical={mech_ok} exact_vs_plain_decode={cfg_res['exact_vs_decode']}"
                )

        # --- timing: exactly ONE trace alive at a time (the harness discipline; see tests/VERIFY_W32_AUDIT.md):
        # decode width -> time -> release; per plan: sub-trace sections (each released), capture, time, release ---
        if DO_TIMING:
            for r in refs.values():
                r.release()
            refs.clear()
            for vs in steps.values():
                vs.release()
            for w in DECODE_WIDTHS:
                r = DecodeRef(model, w, page_tables[:w])
                r.setup()
                med, mn = r.time_replays(N_REPLAYS)
                r.release()
                results["timing"][f"decode_w{w}"] = {"median_ms": med, "min_ms": mn}
                logger.info(f"[verify] TIMING decode w={w} traced x{N_REPLAYS}: med {med:.2f} min {mn:.2f} ms")
            for (w, T), vs in steps.items():
                sec = vs.time_sections_traced(N_REPLAYS)  # TRACED sub-traces (one layer of each kind, embed, head)
                vs.compile()
                vs.capture()
                med, mn = vs.time_replays(N_REPLAYS)
                vs.release()
                results["timing"][f"verify_w{w}_T{T}"] = {
                    "R": vs.plan.R,
                    "median_ms": med,
                    "min_ms": mn,
                    "traced_sections_ms": sec,
                }
                dec = results["timing"].get(f"decode_w{w}", {}).get("median_ms")
                logger.info(
                    f"[verify] TIMING verify (w={w},T={T},R={vs.plan.R}) attn={sec['attn_mode']} traced x{N_REPLAYS}: "
                    f"med {med:.2f} min {mn:.2f} ms" + (f" = {med / dec:.2f}x decode w{w} ({dec:.2f})" if dec else "")
                )
                logger.info(
                    f"[verify] SECTIONS traced (w={w},T={T}) ms: attn_layer[{sec['attn_mode']}]={sec['attn_T']:.2f} "
                    f"attn_offsets_T={sec['attn_offsets_T']:.2f} attn_1={sec['attn_1']:.2f} "
                    f"per_offset={sec['attn_layer_per_offset']:.3f}"
                    + (f" attn_batched={sec['attn_batched']:.2f}" if "attn_batched" in sec else "")
                    + f" gdn_layer={sec['gdn']:.2f} embed={sec['embed']:.2f} head={sec['head']:.2f} | "
                    f"x{sec['n_attn']} attn={sec['attn_layers_total']:.1f} ({100 * sec['attn_layers_total'] / med:.0f}%), "
                    f"x{sec['n_gdn']} gdn={sec['gdn_layers_total']:.1f} ({100 * sec['gdn_layers_total'] / med:.0f}%), "
                    f"embed+head={sec['embed'] + sec['head']:.1f}, "
                    f"per-offset loop would be {sec['attn_offset_loop_total']:.1f} ({100 * sec['attn_offset_loop_total'] / med:.0f}%)"
                )
    finally:
        if TRACKER_AUDIT:
            iso._audit_dump(
                os.environ.get("VERIFY_AUDIT_OUT", "/home/eslim/experiments/qwen36/logs/tracker_audit_exactness.txt")
            )
        for vs in steps.values():
            vs.release()
        for r in refs.values():
            r.release()
        model.pd_gdn_capture = None
        model.free_kv_caches()
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=1)
        logger.info(f"[verify] results -> {OUT_JSON}")
    print("VERIFY_RESULT " + json.dumps({k: v for k, v in results["timing"].items()}))
    for key, cfg in results["configs"].items():
        print(
            f"VERIFY_EXACT {key}: mechanism_identical={cfg['mechanism_identical_across_policies']} "
            f"exact_vs_decode={cfg['exact_vs_decode']} "
            + " ".join(
                f"{p}:steps={v['steps']},acc={v['drafts_accepted']},mism={len(v['mismatch_vs_decode'])}"
                for p, v in cfg["policies"].items()
            )
        )
    if DO_EXACT:
        assert all(
            c["mechanism_identical_across_policies"] for c in results["configs"].values()
        ), "draft-dependent commits"
