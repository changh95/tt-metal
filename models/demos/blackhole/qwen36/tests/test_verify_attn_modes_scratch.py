# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device, model weights): the BATCHED attention middle of the verify step vs the per-offset reference.

Per (w, T): two VerifySteps on the same model -- attn_mode "offsets" (T per-offset decode passes per attention layer,
the reference) and "batched" (one T-row paged_update_cache per cache + one virtual-user SDPA per layer). With
ATTN_EXACT=1 (w <= 8 only: never prefill 32 users in-process): prefill w users, run one EAGER step in each mode on the
same inputs (re-prefill in between: the GDN state commit is not idempotent) with the per-layer row_check hook ->
per-layer bitwise / PCC comparison and the per-row argmax; then the TRACED steps driven by VerifyController with the
same seeded random drafts -> committed streams must be identical. Timing: traced step (both modes), the traced decode
step at width w, and the traced per-section sub-traces (attention layer: offsets T / offsets 1 / batched).

  TT_VISIBLE_DEVICES=2,3,4,5 MESH_DEVICE=P150x4 ... pytest tests/test_verify_attn_modes_scratch.py -s
Env: ATTN_CONFIGS ("1,8;8,8"), ATTN_EXACT (1), ATTN_REPLAYS (30), ATTN_STEPS (6), ATTN_OUT (json path),
     QWEN36_SDPA_DEC_MAX_CORES_PER_HEAD=1 recommended for w=32 configs (B=32 back-to-back SDPA wedge suspicion).
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
CONFIGS = [tuple(int(v) for v in c.split(",")) for c in os.environ.get("ATTN_CONFIGS", "1,8;8,8").split(";") if c]
DO_EXACT = os.environ.get("ATTN_EXACT", "1") == "1"
N_REPLAYS = int(os.environ.get("ATTN_REPLAYS", "30"))
N_STEPS = int(os.environ.get("ATTN_STEPS", "6"))
OUT_JSON = os.environ.get("ATTN_OUT", "/home/eslim/experiments/qwen36/logs/verify_attn_modes_result.json")
CHUNK = 2048
BPU = 8
PROMPTS = [
    "The capital of France is",
    "Write a short story about a robot who learns to paint. Once upon a time,",
    "Q: What is 17 * 23? Let's think step by step.\nA:",
]
AICLK_MHZ = int(os.environ.get("VERIFY_FORCE_AICLK_MHZ", "1200"))
POST_PREFILL_SLEEP_MS = float(os.environ.get("VERIFY_POST_PREFILL_SLEEP_MS", "600"))


def _pin_aiclk(mhz):
    if mhz <= 0:
        return
    try:
        import pyluwen

        chips = pyluwen.detect_chips()
        for chip in chips:
            chip.arc_msg(0x33, wait_for_done=True, arg0=mhz, arg1=0, timeout=2.0)
        logger.info(f"[attn] pinned {len(chips)} chip(s) AICLK to {mhz} MHz")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[attn] AICLK pin failed: {e!r}")


class DecodeRef:
    def __init__(self, model, w, page_table):
        self.model, self.w, self.pt = model, w, page_table
        self.mesh = model.mesh_device
        self.per_shard = model.args.vocab_size // model.num_devices
        self.dev = self.tid = self.out = None

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
        idx, val = self._fwd(self.dev)
        ttnn.synchronize_device(self.mesh)
        ttnn.deallocate(idx)
        ttnn.deallocate(val)
        self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out = self._fwd(self.dev)
        ttnn.end_trace_capture(self.mesh, self.tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)

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


def _prefill(model, prompt_ids, page_tables, w):
    ids = [prompt_ids[s % len(prompt_ids)] for s in range(w)]
    lens = [t.shape[1] for t in ids]
    logits = []
    for g0 in range(0, w, 8):
        g1 = min(w, g0 + 8)
        if g0 > 0:
            time.sleep(1.0)
        logits += model.prefill_paged_slots(ids[g0:g1], page_tables[g0:g1], list(range(g0, g1)), valid_lens=lens[g0:g1])
        ttnn.synchronize_device(model.mesh_device)
    if POST_PREFILL_SLEEP_MS > 0:
        time.sleep(POST_PREFILL_SLEEP_MS / 1e3)
    first = [int(lg.reshape(-1)[: model.vocab_size].float().argmax()) for lg in logits]
    return lens, first


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    if a.std() == 0 or b.std() == 0:
        return float(torch.equal(a, b))
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _make_step(model, w, T, pt, mode):
    prev = os.environ.get("QWEN36_VERIFY_ATTN")
    os.environ["QWEN36_VERIFY_ATTN"] = mode
    try:
        vs = VerifyStep(model, w, T, pt)
    finally:
        if prev is None:
            os.environ.pop("QWEN36_VERIFY_ATTN", None)
        else:
            os.environ["QWEN36_VERIFY_ATTN"] = prev
    assert vs.plan.attn_mode == mode
    return vs


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_verify_attn_modes(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    _pin_aiclk(AICLK_MHZ)
    results = {"configs": {}, "timing": {}}
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    logger.info(f"[attn] model load {time.perf_counter() - t0:.1f}s layers={len(model.layers)}")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt_ids = [tok(p, return_tensors="pt").input_ids.to(torch.int32) for p in PROMPTS]
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    steps, refs = {}, {}
    exact_cfgs = [(w, T) for (w, T) in CONFIGS if w <= 8] if DO_EXACT else []
    try:
        for w, T in CONFIGS:
            steps[(w, T, "offsets")] = _make_step(model, w, T, page_tables[:w], "offsets")
            steps[(w, T, "batched")] = _make_step(model, w, T, page_tables[:w], "batched")
        if exact_cfgs:
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
            logger.info(f"[attn] prefill warmup {time.perf_counter() - t0:.1f}s")
            _prefill(model, prompt_ids, page_tables, 1)
        for w in sorted(set(w for w, _ in CONFIGS)):
            refs[w] = DecodeRef(model, w, page_tables[:w])
            refs[w].setup()

        # --- eager exactness (per layer) BEFORE the traces are captured: eager runs mutate nothing the traces need
        for w, T in exact_cfgs:
            k = T - 1
            rng = random.Random(777 + w * 10 + T)
            caps = {}
            argmaxes = {}
            for mode in ("offsets", "batched"):
                vs = steps[(w, T, mode)]
                lens, first = _prefill(model, prompt_ids, page_tables, w)
                toks = [[first[s]] + [rng.randrange(model.vocab_size) for _ in range(k)] for s in range(w)]
                rng = random.Random(777 + w * 10 + T)  # same drafts for both modes
                cap = {}

                def hook(name, t, cap=cap):
                    cap[name] = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).clone()

                vs.plan.upload(toks, lens, [0] * w)
                idx, val = vs.forward(row_check=hook)
                ttnn.synchronize_device(device)
                argmaxes[mode] = combine_sharded_argmax(device, idx, val, vs.plan.R, vs.per_shard).tolist()
                ttnn.deallocate(idx)
                ttnn.deallocate(val)
                vs.plan._host_refs = []
                caps[mode] = cap
            layer_rows = []
            first_diff = None
            for name in caps["offsets"]:
                a, b = caps["offsets"][name], caps["batched"][name]
                eq = torch.equal(a, b)
                pcc = _pcc(a, b)
                layer_rows.append((name, eq, pcc))
                if not eq and first_diff is None:
                    first_diff = name
            n_eq = sum(1 for _, eq, _ in layer_rows if eq)
            min_pcc = min(p for _, _, p in layer_rows)
            am_eq = argmaxes["offsets"] == argmaxes["batched"]
            logger.info(
                f"[attn] ({w},{T}) EAGER offsets vs batched: {n_eq}/{len(layer_rows)} layer outputs bitwise equal, "
                f"min pcc {min_pcc:.6f}, first differing = {first_diff}, argmax rows equal = {am_eq}"
            )
            for name, eq, pcc in layer_rows:
                if "attn" in name or not eq:
                    logger.info(f"[attn]    {name}: equal={eq} pcc={pcc:.6f}")
            results["configs"][f"{w},{T}"] = {
                "eager_layers_equal": n_eq,
                "eager_layers": len(layer_rows),
                "eager_min_pcc": min_pcc,
                "eager_first_diff": first_diff,
                "eager_argmax_equal": am_eq,
                "layers": [(n, bool(e), p) for n, e, p in layer_rows],
            }

        # --- compile + capture both modes
        for key, vs in steps.items():
            t0 = time.perf_counter()
            vs.compile()
            t1 = time.perf_counter()
            vs.capture()
            logger.info(f"[attn] {key} R={vs.plan.R} compile {t1 - t0:.1f}s capture {time.perf_counter() - t1:.1f}s")

        # --- traced multi-step exactness: same seeded drafts through VerifyController in both modes
        for w, T in exact_cfgs:
            k = T - 1
            streams = {}
            accepts = {}
            for mode in ("offsets", "batched"):
                vs = steps[(w, T, mode)]
                rng = random.Random(4242 + w * 10 + T)
                lens, first = _prefill(model, prompt_ids, page_tables, w)
                ctrl = vg.VerifyController(T=T, run=vs.run, positions=lens, last=first)
                for _ in range(N_STEPS):
                    drafts = []
                    for s in range(w):
                        # mix of plausible (the true continuation would need the reference stream; use last token +
                        # random) drafts so that some get accepted only when both modes agree
                        drafts.append([rng.randrange(model.vocab_size) for _ in range(k)])
                    ctrl.step(drafts)
                streams[mode] = [list(c) for c in ctrl.committed]
                accepts[mode] = [list(a) for a in ctrl.accept_history]
            same = streams["offsets"] == streams["batched"]
            logger.info(
                f"[attn] ({w},{T}) TRACED {N_STEPS} steps, committed streams identical across modes = {same}; "
                f"tokens/user offsets={[len(c) for c in streams['offsets']]} batched={[len(c) for c in streams['batched']]}; "
                f"user0 offsets text: {tok.decode(streams['offsets'][0][:16])!r}"
            )
            results["configs"][f"{w},{T}"]["traced_streams_identical"] = same
            results["configs"][f"{w},{T}"]["streams"] = streams

        # --- timing
        for w in sorted(refs):
            med, mn = refs[w].time_replays(N_REPLAYS)
            results["timing"][f"decode_w{w}"] = {"median_ms": med, "min_ms": mn}
            logger.info(f"[attn] TIMING decode w={w} traced x{N_REPLAYS}: med {med:.2f} min {mn:.2f} ms")
        for w, T in CONFIGS:
            for mode in ("offsets", "batched"):
                vs = steps[(w, T, mode)]
                med, mn = vs.time_replays(N_REPLAYS)
                results["timing"][f"verify_w{w}_T{T}_{mode}"] = {"median_ms": med, "min_ms": mn, "R": vs.plan.R}
                logger.info(f"[attn] TIMING verify (w={w},T={T},R={vs.plan.R}) {mode}: med {med:.2f} min {mn:.2f} ms")
            sec = steps[(w, T, "batched")].time_sections_traced(N_REPLAYS)
            results["timing"][f"verify_w{w}_T{T}_batched"]["traced_sections_ms"] = sec
            dec = results["timing"][f"decode_w{w}"]["median_ms"]
            logger.info(
                f"[attn] SECTIONS traced (w={w},T={T}) ms/layer: attn offsets(T)={sec['attn_offsets_T']:.3f} "
                f"attn offsets(1)={sec['attn_1']:.3f} attn batched={sec.get('attn_batched', float('nan')):.3f} "
                f"gdn={sec['gdn']:.3f} embed={sec['embed']:.3f} head={sec['head']:.3f} | x{sec['n_attn']} attn layers: "
                f"offsets {sec['n_attn'] * sec['attn_offsets_T']:.2f} vs batched {sec['n_attn'] * sec.get('attn_batched', float('nan')):.2f} "
                f"(one-pass floor {sec['n_attn'] * sec['attn_1']:.2f}) | decode w={w} step {dec:.2f}"
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
        logger.info(f"[attn] results -> {OUT_JSON}")
    print("ATTN_RESULT " + json.dumps(results["timing"]))
    for key, cfg in results["configs"].items():
        print(
            f"ATTN_EXACT {key}: eager_layers_equal={cfg['eager_layers_equal']}/{cfg['eager_layers']} "
            f"min_pcc={cfg['eager_min_pcc']:.6f} argmax_equal={cfg['eager_argmax_equal']} "
            f"traced_streams_identical={cfg.get('traced_streams_identical')}"
        )
