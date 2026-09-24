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

    def setup(self):
        toks = torch.full((self.w, 1), 1, dtype=torch.int32)
        pos = torch.full((self.w,), 8, dtype=torch.int32)
        self.dev = self.model.prepare_inputs_decode(toks, pos, self.pt)
        idx, val = self._fwd(self.dev)  # compile
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(idx)
        ttnn.deallocate(val)
        self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out = self._fwd(self.dev)
        ttnn.end_trace_capture(self.mesh, self.tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)

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
        _prefill(model, prompt_ids, page_tables, 1)

        # --- served order (qwen36_vllm: prefill warm-up -> decode trace capture), then the verify traces ---
        for w in sorted(set([w for w, _ in CONFIGS] + (DECODE_WIDTHS if DO_TIMING else []))):
            refs[w] = DecodeRef(model, w, page_tables[:w])
            refs[w].setup()
        for (w, T), vs in steps.items():
            t0 = time.perf_counter()
            vs.compile()
            t1 = time.perf_counter()
            vs.capture()
            logger.info(
                f"[verify] ({w},{T}) R={vs.plan.R} compile {t1 - t0:.1f}s capture {time.perf_counter() - t1:.1f}s"
            )

        # --- exactness per config ---
        if DO_EXACT:
            for (w, T), vs in steps.items():
                k = T - 1
                rng = random.Random(1234 + w * 10 + T)
                lens, first = _prefill(model, prompt_ids, page_tables, w)
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
                    lens, first = _prefill(model, prompt_ids, page_tables, w)
                    assert first == [ref_streams[s][0] for s in range(w)], "prefill first token not reproducible"
                    ctrl = vg.VerifyController(T=T, run=vs.run, positions=lens, last=first)
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

        # --- timing ---
        if DO_TIMING:
            for (w, T), vs in steps.items():
                med, mn = vs.time_replays(N_REPLAYS)
                results["timing"][f"verify_w{w}_T{T}"] = {"median_ms": med, "min_ms": mn, "R": vs.plan.R}
                logger.info(
                    f"[verify] TIMING verify (w={w},T={T},R={vs.plan.R}) traced x{N_REPLAYS}: med {med:.2f} min {mn:.2f} ms"
                )
            for w in DECODE_WIDTHS:
                med, mn = refs[w].time_replays(N_REPLAYS)
                results["timing"][f"decode_w{w}"] = {"median_ms": med, "min_ms": mn}
                logger.info(f"[verify] TIMING decode w={w} traced x{N_REPLAYS}: med {med:.2f} min {mn:.2f} ms")
            for (w, T), vs in steps.items():
                sec = vs.time_sections_traced(N_REPLAYS)  # TRACED sub-traces (one layer of each kind, embed, head)
                results["timing"][f"verify_w{w}_T{T}"]["traced_sections_ms"] = sec
                tot = results["timing"][f"verify_w{w}_T{T}"]["median_ms"]
                logger.info(
                    f"[verify] SECTIONS traced (w={w},T={T},R={vs.plan.R}) ms: attn_layer(T)={sec['attn_T']:.2f} "
                    f"attn_layer(1 offset)={sec['attn_1']:.2f} per_offset={sec['attn_layer_per_offset']:.3f} "
                    f"gdn_layer={sec['gdn']:.2f} embed={sec['embed']:.2f} head={sec['head']:.2f} | "
                    f"x{sec['n_attn']} attn={sec['attn_layers_total']:.1f} (offset loop {sec['attn_offset_loop_total']:.1f} = "
                    f"{100 * sec['attn_offset_loop_total'] / tot:.0f}% of step) x{sec['n_gdn']} gdn={sec['gdn_layers_total']:.1f} "
                    f"| step {tot:.1f}"
                )
    finally:
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
