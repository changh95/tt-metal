# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy on-device sampling equivalence: multi-chip top-k path (QWEN36_SAMPLING_FORCE_ARGMAX=0, the default)
vs the forced full-vocab argmax sampler (QWEN36_SAMPLING_FORCE_ARGMAX=1), on the SERVED decode path.

Builds the model like profile_prefill_decode.py (Qwen36Model.from_pretrained, batched GDN state + paged KV,
prefill_paged_slots per user, per-width traced decode with per-step host input refresh, per-bucket in-trace
sampler), then for every decode step runs BOTH samplers on the SAME logits tensor and reads the logits back to
host as a third, exact reference (float argmax with lowest-index tie-break = the contract of both device paths):

  W1          each of the 3 prompts alone in slot 0, STEPS steps at bucket width 1
  W8          8 users (prompt i % 3) in slots 0..7, STEPS steps at width 8
  TRANSITION  32 users prefilled; 16 steps at width 8, 16 at width 32 (rows 8..31 start decoding), then 8 more at
              width 8 and 8 at width 32 with every bucket's decode + sampling traces kept alive and multiplexed
              exactly as served decode bucketing does (_tt_allow_decode_trace_buffer_reuse).

Each configuration runs three times: fed with the argmax-path token (today's served behaviour), fed with the
top-k-path token (the new default), and the top-k feed again (determinism). Assertions:
  * per step and row, argmax-path token == top-k-path token == host reference, unless the host logits show an
    exact bf16 tie at the maximum (then both device tokens must be among the tied ids; ties are reported);
  * the top-k-fed sequence equals the argmax-fed sequence up to the first tie divergence (if any);
  * the two top-k-fed runs are identical token for token (determinism).
The per-step param-upload no-change guard (TTSampling/TTPenalties.reset_params) is checked at the start by
counting copy_host_to_device_tensor calls over repeated identical greedy parameter sets.

Run (half A, TP=4):
  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.8-27B HF_HUB_OFFLINE=1 ARCH_NAME=blackhole \
  TT_QWEN35_TEXT_VER=qwen36_blackhole TT_DECODE_BUCKETING=1 \
    pytest models/demos/blackhole/qwen36/tests/test_sampling_topk_vs_argmax.py -x -s
Env: SAMP_EQUIV_STEPS (32) decode steps per phase-less configuration; SAMP_EQUIV_CONFIGS (default "w1").
Result lines are prefixed SAMP_EQUIV for grepping.

STATUS 2026-09-24: default configs = "w1" only. In this in-process harness every request after the first one
decodes garbage (bit-deterministic, any slot, any slot-write mode 0/1/2, with or without the slot-write / hist-pack
warm-ups; the prefill's own first token stays correct and the slot-verify probe shows the written GDN slot equal to
the scratch) and the first width>=4 decode after that wedged the half twice -- see logs/samp_diag_r{2,3,4}*.log and
tests/samp_diag_scratch.py. The sampler comparison itself is unaffected (both paths and the host reference see the
same logits, and the garbage-flat distributions produce MORE exact ties than real text), but "w8" and "transition"
stay opt-in (SAMP_EQUIV_CONFIGS=w1,w8,transition) until that handoff issue is understood. The served stack is
unaffected: the P/D decode instance imports its GDN state (pd_transfer) and never runs this handoff.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingParams, format_sampling_params
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.tt_transformers.tt.common import copy_host_to_device

BMAX = 32
STEPS = int(os.environ.get("SAMP_EQUIV_STEPS", "32"))
CONFIGS = [c for c in os.environ.get("SAMP_EQUIV_CONFIGS", "w1").split(",") if c]
CHUNK = 2048  # qwen36_vllm._PREFILL_WARMUP_CHUNK
PROMPTS = [
    "The capital of France is",
    "Write a short haiku about the ocean at dawn.",
    "Explain in one paragraph why the sky is blue, then list three planets.",
]
# Blocks per user: prompt + slack + the longest run (transition = 48 steps), rounded to a multiple of 8.
_need = 128 + 256 + 2 * STEPS
BPU = ((-(-_need // BLOCK_SIZE) + 7) // 8) * 8


def _host_greedy(row: torch.Tensor):
    """Lowest-index argmax of a float row and the number of exact maxima (1 = no tie)."""
    mx = row.max()
    tied = (row == mx).nonzero().reshape(-1)
    return int(tied[0]), int(tied.numel()), tied.tolist()


class _WidthTraces:
    """Decode trace + both per-bucket sampler traces for one bucket width, kept alive like served bucketing.

    Two phases, in the served order (generator_interface.warmup_decode_buckets): ``compile`` EVERY width eagerly
    first, then ``capture`` each -- a program compile or persistent buffer allocated after a capture aliases that
    trace's freed intermediates and corrupts its replays (qwen36_vllm.warmup_model_prefill)."""

    def __init__(self, model, device, width, page_table):
        self.model, self.device, self.width = model, device, width
        self.page_table = page_table
        self.tid = None

    def compile(self, tokens, positions):
        model, device = self.model, self.device
        ts = model.sampling.tt_sampling
        model.sampling.set_trace_bucket(self.width)
        dev0 = model.prepare_inputs_decode(tokens, positions, self.page_table)
        lg0 = model.ttnn_decode_forward(
            dev0[0], dev0[1], rot_mat_idxs=dev0[2], page_table=dev0[3], on_device_logits=True
        )
        for flag in (True, False):
            ts._force_argmax_sampling = flag
            out = model.sampling.sample(lg0, enable_trace=False)
            for t in out if isinstance(out, tuple) else (out,):
                if t is not None:
                    ttnn.deallocate(t)
        ttnn.synchronize_device(device)
        ttnn.deallocate(lg0)
        for t in dev0:
            if t is not None:
                ttnn.deallocate(t)

    def capture(self, tokens, positions):
        model, device = self.model, self.device
        ts = model.sampling.tt_sampling
        model.sampling.set_trace_bucket(self.width)
        host = model.prepare_decode_inputs_host(tokens, positions, page_table=self.page_table)
        self.dev = copy_host_to_device(host, mesh_device=device)
        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        self.lg = model.ttnn_decode_forward(
            self.dev[0], self.dev[1], rot_mat_idxs=self.dev[2], page_table=self.dev[3], on_device_logits=True
        )
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        ttnn.synchronize_device(device)
        # Per-bucket sampling traces bound to THIS bucket's logits tensor, one slot per force_argmax key.
        ttnn.execute_trace(device, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        for flag in (True, False):
            ts._force_argmax_sampling = flag
            model.sampling.sample(self.lg, enable_trace=True, skip_precompile=True)
            ttnn.synchronize_device(device)

    def step(self, tokens, positions):
        """One served-style decode step: refresh inputs, replay decode, sample with both paths, host reference.

        Returns dict(argmax=[W], topk=[W], host=[W], ties={row: [tied ids]})."""
        model, device, W = self.model, self.device, self.width
        model.sampling.set_trace_bucket(W)
        host = model.prepare_decode_inputs_host(tokens, positions, page_table=self.page_table)
        copy_host_to_device(host_tensors=host[:3], device_tensors=self.dev[:3])
        ttnn.execute_trace(device, self.tid, cq_id=0, blocking=False)
        out = {}
        ts = model.sampling.tt_sampling
        for name, flag in (("argmax", True), ("topk", False)):
            ts._force_argmax_sampling = flag
            sampled = model.sampling.sample(self.lg, enable_trace=True)
            tok = sampled[0] if isinstance(sampled, tuple) else sampled
            out[name] = model.process_output_decode(tok, W, is_tokens=True).clone().tolist()
        # Host reference from the same (padded, vocab-sharded) logits: shard d holds global ids
        # [d*V/nd, (d+1)*V/nd) in its first V/nd columns; anything past that is padding.
        full = ttnn.to_torch(self.lg, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=3)).float()
        nd = model.num_devices
        V = model.args.vocab_size
        P = full.shape[-1] // nd
        real = V // nd
        rows = full.reshape(-1, full.shape[-1])[:W]
        logits = torch.cat([rows[:, d * P : d * P + real] for d in range(nd)], dim=1)  # [W, V], global order
        out["host"], out["ties"] = [], {}
        for r in range(W):
            tok, n_tied, tied = _host_greedy(logits[r])
            out["host"].append(tok)
            if n_tied > 1:
                out["ties"][r] = tied
        return out

    def release(self):
        if self.tid is not None:
            ttnn.release_trace(self.device, self.tid)


def _run(model, device, tok, name, prompt_rows, phases, feed, page_tables, traces, records):
    """Prefill `prompt_rows` (slot -> prompt index), then decode through `phases` = [(width, steps), ...] replaying
    the pre-captured `traces[width]` (all bucket widths are compiled and captured before any request, as
    generator_interface.warmup_decode_buckets does for serving: a decode forward advances the GDN state).

    `feed` selects which sampler's token continues each row ("argmax" or "topk"). Returns per-row token
    sequences per path plus the per-step mismatch/tie log; appends SAMP_EQUIV lines to `records`."""
    n_rows = len(prompt_rows)
    ids = [tok(PROMPTS[prompt_rows[r]], return_tensors="pt").input_ids.to(torch.int32) for r in range(n_rows)]
    lens = [int(x.shape[1]) for x in ids]
    t0 = time.perf_counter()
    lg = model.prefill_paged_slots(ids, page_tables[:n_rows], list(range(n_rows)), valid_lens=lens)
    ttnn.synchronize_device(device)
    first = [int(lg[r].reshape(-1)[: model.vocab_size].float().argmax()) for r in range(n_rows)]
    logger.info(f"[{name}/{feed}] prefill {n_rows} users {time.perf_counter() - t0:.1f}s lens={lens} first={first}")

    tokens = torch.tensor(first, dtype=torch.int32).reshape(n_rows, 1)
    positions = torch.tensor(lens, dtype=torch.int32)
    seq = {k: [[] for _ in range(n_rows)] for k in ("argmax", "topk", "host")}
    events = []  # (phase, step, row, argmax_tok, topk_tok, host_tok, tied_ids or None)
    step_ms = []
    for phase_idx, (width, steps) in enumerate(phases):
        tr = traces[width]
        for s in range(steps):
            t0 = time.perf_counter()
            out = tr.step(tokens[:width], positions[:width])
            step_ms.append(1e3 * (time.perf_counter() - t0))
            for r in range(width):
                a, b, h = out["argmax"][r], out["topk"][r], out["host"][r]
                seq["argmax"][r].append(a)
                seq["topk"][r].append(b)
                seq["host"][r].append(h)
                tied = out["ties"].get(r)
                if not (a == b == h) or tied is not None:
                    events.append((phase_idx, s, r, a, b, h, tied))
                tokens[r, 0] = a if feed == "argmax" else b
            positions[:width] += 1
    med = sorted(step_ms)[len(step_ms) // 2] if step_ms else float("nan")
    logger.info(f"[{name}/{feed}] {len(step_ms)} steps, median {med:.1f} ms/step incl. both samplers + host read")
    for ev in events:
        phase_idx, s, r, a, b, h, tied = ev
        kind = "TIE" if tied is not None else "MISMATCH"
        line = (
            f"SAMP_EQUIV {name}/{feed} {kind} phase={phase_idx} step={s} row={r} argmax={a} topk={b} host={h}"
            f" tied_ids={tied}"
        )
        records.append(line)
        logger.warning(line)
    for r in range(n_rows):
        text = tok.decode(seq["topk"][r][: min(len(seq["topk"][r]), 24)])
        logger.info(f"[{name}/{feed}] row {r} (prompt {prompt_rows[r]}): {text!r}")
    return seq, events


def _check_param_upload_guard(model, monkeypatch, records):
    """Repeated identical greedy parameter sets must not re-upload k/p/temp/tie-mask or the penalties."""
    calls = {"n": 0}
    real = ttnn.copy_host_to_device_tensor

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", counted)
    n_rows = model.sampling.tt_sampling.max_batch_size
    params = format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0, seed=None), n_rows)
    assert not model.sampling.tt_sampling._allow_force_argmax_sampling, "test expects the default (top-k) knob"
    model.sampling.reset_sampling_params(params)
    first = calls["n"]
    assert not model.sampling.tt_sampling.force_argmax_sampling
    for _ in range(3):
        model.sampling.reset_sampling_params(params)
    repeat = calls["n"] - first
    monkeypatch.undo()
    line = f"SAMP_EQUIV param_upload_guard first_call_uploads={first} repeat_uploads_over_3_calls={repeat}"
    records.append(line)
    logger.info(line)
    assert repeat == 0, f"unchanged greedy params re-uploaded {repeat} tensors over 3 reset_sampling_params calls"


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_greedy_topk_matches_argmax(mesh_device, monkeypatch):
    if not _MULTI:
        pytest.skip("TP path only")
    device = mesh_device
    device.enable_program_cache()
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    logger.info(f"[samp] model load {time.perf_counter() - t0:.1f}s mesh={_MESH_SHAPE} BPU={BPU} steps={STEPS}")
    if model.sampling is None:
        pytest.skip("on-device sampling unsupported on this mesh/vocab")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)

    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    records = []
    failures = []
    traces = {}
    try:
        # served prefill warm-up (Qwen36ForCausalLM.warmup_model_prefill)
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

        _check_param_upload_guard(model, monkeypatch, records)

        configs = []
        if "w1" in CONFIGS:
            configs += [(f"w1_p{i}", [i], [(1, STEPS)]) for i in range(len(PROMPTS))]
        if "w8" in CONFIGS:
            configs.append(("w8", [i % len(PROMPTS) for i in range(8)], [(8, STEPS)]))
        if "transition" in CONFIGS:
            half = max(STEPS // 2, 1)
            quarter = max(STEPS // 4, 1)
            configs.append(
                (
                    "transition",
                    [i % len(PROMPTS) for i in range(32)],
                    [(8, half), (32, half), (8, quarter), (32, quarter)],
                )
            )

        # Served-style decode warm-up: compile + capture every needed bucket width (decode trace and both
        # sampler traces) BEFORE any request; the traces stay alive and are multiplexed for the whole test.
        widths = sorted({w for _, _, phases in configs for w, _ in phases})
        t0 = time.perf_counter()

        def _dummy(w):
            return torch.tensor([[100 + u] for u in range(w)], dtype=torch.int32), torch.full(
                (w,), 64, dtype=torch.int32
            )

        for w in widths:  # 1. compile every width (decode + both sampler paths) ...
            traces[w] = _WidthTraces(model, device, w, page_tables[:w])
            traces[w].compile(*_dummy(w))
        for w in widths:  # 2. ... then capture every width's traces
            traces[w].capture(*_dummy(w))
        logger.info(
            f"[samp] decode warm-up (widths {widths}, decode + 2 sampler traces each) {time.perf_counter() - t0:.1f}s"
        )

        for name, prompt_rows, phases in configs:
            runs = {}
            for tag, feed in (("A", "argmax"), ("T1", "topk"), ("T2", "topk")):
                runs[tag] = _run(
                    model, device, tok, f"{name}.{tag}", prompt_rows, phases, feed, page_tables, traces, records
                )
            n_rows = len(prompt_rows)
            # 1. per-step three-way agreement except at host-visible ties (both device tokens within the tie set)
            for tag in runs:
                for phase_idx, s, r, a, b, h, tied in runs[tag][1]:
                    if tied is None:
                        failures.append(f"{name}.{tag}: step {s} row {r} argmax={a} topk={b} host={h} with NO tie")
                    elif a not in tied or b not in tied:
                        failures.append(f"{name}.{tag}: step {s} row {r} argmax={a} topk={b} outside tie set {tied}")
            # 2. top-k-fed sequence == argmax-fed sequence up to the first tie divergence
            seqA, seqT = runs["A"][0], runs["T1"][0]
            for r in range(n_rows):
                a_seq, t_seq = seqA["argmax"][r], seqT["topk"][r]
                n = min(len(a_seq), len(t_seq))
                diverge = next((i for i in range(n) if a_seq[i] != t_seq[i]), None)
                if diverge is not None:
                    # a divergence is only acceptable at (or after) a tie on this row (in either run)
                    first_tie = min(
                        (ev[1] for tag in ("A", "T1") for ev in runs[tag][1] if ev[2] == r and ev[6] is not None),
                        default=None,
                    )
                    if first_tie is None or first_tie > diverge:
                        failures.append(
                            f"{name}: row {r} topk-fed diverges from argmax-fed at step {diverge} "
                            f"({a_seq[diverge]} vs {t_seq[diverge]}) without a preceding tie"
                        )
                    records.append(f"SAMP_EQUIV {name} row={r} topk_fed_diverges_at={diverge} first_tie={first_tie}")
                else:
                    records.append(f"SAMP_EQUIV {name} row={r} topk_fed==argmax_fed steps={n}")
            # 3. determinism: the two top-k-fed runs are identical token for token, on every path
            seqT2 = runs["T2"][0]
            for r in range(n_rows):
                for path in ("argmax", "topk", "host"):
                    if seqT[path][r] != seqT2[path][r]:
                        first = next(i for i, (x, y) in enumerate(zip(seqT[path][r], seqT2[path][r])) if x != y)
                        failures.append(
                            f"{name}: row {r} {path} tokens differ between two top-k-fed runs at step {first}"
                        )
            det = all(seqT[p][r] == seqT2[p][r] for r in range(n_rows) for p in ("argmax", "topk", "host"))
            n_ties = sum(1 for ev in runs["A"][1] if ev[-1] is not None)
            n_mis = sum(1 for ev in runs["A"][1] if ev[-1] is None)
            records.append(
                f"SAMP_EQUIV {name} rows={n_rows} phases={phases} argmax_vs_topk_mismatches_no_tie={n_mis} "
                f"ties={n_ties} deterministic={det}"
            )
    finally:
        if model.sampling is not None:
            model.sampling.reset_trace()
        for tr in traces.values():
            tr.release()
        model.pd_gdn_capture = None
        model.free_kv_caches()
    for line in records:
        print(line)
    assert not failures, "\n".join(failures)
