# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Grouped traced prefill (model.capture_prefill_group_traces + the grouping in prefill_paged_slots) against the
per-user traced path it replaces, on the served warm-up (chunk trace + masked-bucket traces + grouped traces).

test_grouped_prefill_matches_per_user
    For prompt sets of B in {2, 4, 8} same-bucket users with mixed real lengths (buckets 128 / 256 / 512) plus one
    mixed-bucket step: prefill every user through the per-user path (N=1 calls) and through ONE grouped call
    (N=B), both with the P/D producer capture on (host GDN snapshots) and the host decode-slot write; compare the
    next-token logits, the GDN snapshots (rec + taps) and the greedy tokens of STEPS decode steps per user.
    Passes when every user's greedy continuation is identical; logs exact-match / max|d| / PCC of the rest.

test_grouped_prefill_step_timing
    [PREFILL_TIMING]-style wall clock of one N=8 x 128-token step, grouped vs per-user (QWEN36_PREFILL_GROUP_USE).

Run (half A): MESH_DEVICE=P150x4 TT_VISIBLE_DEVICES=0,1,6,7 QWEN36_PREFILL_GROUP_TRACE=1 QWEN36_PREFILL_TIMING=1 \
    pytest models/demos/blackhole/qwen36/tests/test_prefill_grouped_trace.py -x -s
First run under the trace-allocation tracker (TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1): every
execute_trace then RAISES listing the buffers a replay would corrupt instead of wedging the chips.
Env: QWEN36_GRP_TEST_STEPS (8), QWEN36_GRP_TEST_SETS (all) -- comma list of set names to run.
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

BMAX = 8
BPU = 16  # blocks per user: 1024 positions (bucket 512 prompts + decode steps)
STEPS = int(os.environ.get("QWEN36_GRP_TEST_STEPS", "8"))
CHUNK = 2048

# name -> per-user real lengths (a chat prompt fills slot 0 of every set, see _prompt_sets)
PCC_MIN = 0.999  # correctness bar: PCC of logits / GDN state vs the per-user path (not bit-identical: M = B*bucket)
SETS = {
    "b128_B8": [None, 5, 17, 64, 100, 127, 128, 33],
    "b256_B4": [129, 200, 256, 140],
    "b512_B2": [300, 511],
    "b128_B8_full": [128] * 8,
    "mixed": [None, 40, 128, 129, 250, 600],  # 3 x 128-bucket, 2 x 256-bucket, one 1024-bucket single
}


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if a.numel() < 2 or float(a.std()) == 0 or float(b.std()) == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _cmp(name, got, ref):
    exact = torch.equal(got, ref)
    d = float((got.float() - ref.float()).abs().max())
    p = _pcc(got, ref)
    logger.info(f"[grp] {name}: {'EXACT' if exact else f'max|d|={d:.3g} pcc={p:.6f}'}")
    return exact, d, p


def _prompt_sets(tok):
    chat = tok.apply_chat_template(
        [{"role": "user", "content": "List the planets of the solar system in order from the sun, one line each."}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    chat = [int(x) for x in (chat["input_ids"] if hasattr(chat, "keys") else chat)]
    # Real text (QWEN36_GRP_TEST_RANDOM=0, default): T-token windows of the 4k sample document at a per-row offset, so
    # the greedy continuations have real margins. Uniform-random token ids (=1) give near-flat logits whose argmax
    # flips on the ~1e-3 relative noise of the M = B*bucket matmul programs vs the small-M per-user programs
    # (QWEN36_PREFILL_SMALLM_MAX=128: a different K accumulation order, not bit-identical -- tp_common.py item J).
    import json

    doc = json.load(open("models/demos/blackhole/qwen36/demo/sample_prompts/input_data_long_4k.json"))[0]["prompt"]
    doc_ids = tok(doc, add_special_tokens=False)["input_ids"]
    random_ids = os.environ.get("QWEN36_GRP_TEST_RANDOM", "0") == "1"
    g = torch.Generator().manual_seed(1234)
    out = {}
    for si, (name, lens) in enumerate(SETS.items()):
        rows = []
        for i, T in enumerate(lens):
            if T is None:
                rows.append(torch.tensor([chat], dtype=torch.int32))
            elif random_ids:
                rows.append(torch.randint(1000, 100000, (1, T), generator=g, dtype=torch.int32))
            else:
                off = (97 * i + 331 * si) % max(1, len(doc_ids) - T)
                rows.append(torch.tensor([doc_ids[off : off + T]], dtype=torch.int32))
        out[name] = rows
    return out


def _decode_rows(model, firsts, lens, page_tables, steps):
    """Greedy-decode every row of the batched state for `steps` steps (row u: first token firsts[u], position
    lens[u] + s). Returns per-row token lists. Eager decode at width BMAX."""
    n = len(firsts)
    toks = list(firsts)
    out = [[] for _ in range(n)]
    for s in range(steps):
        tokens = torch.zeros((BMAX, 1), dtype=torch.int32)
        positions = torch.zeros((BMAX,), dtype=torch.int32)
        for u in range(BMAX):
            tokens[u, 0], positions[u] = 100 + u, 64
        for u in range(n):
            tokens[u, 0], positions[u] = toks[u], lens[u] + s
        dev = model.prepare_inputs_decode(tokens, positions, page_tables)
        o, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        lg = model.process_output_decode(o, BMAX)[:, 0, : model.vocab_size].float()
        for u in range(n):
            toks[u] = int(lg[u].argmax())
            out[u].append(toks[u])
    return out


def _snap_copy(model, slot):
    rec, taps = model.pd_gdn_capture.pop(slot)
    rec_c, taps_c = rec.clone(), taps.clone()
    model.pd_gdn_snapshot_release(rec, taps)
    return rec_c, taps_c


def _setup(mesh_device):
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE * 2)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    # +1: the scratch pad block the bucket traces' fixed-width fill tables point padding rows at.
    kv_shape = [BMAX * BPU + 1, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    assert model._grp_spec, "QWEN36_PREFILL_GROUP_TRACE is off; nothing to test (set QWEN36_PREFILL_GROUP_TRACE=1)"
    # Compile-first rule (tests/VERIFY_W32_AUDIT.md): every program this process runs is compiled BEFORE the first
    # trace capture. The eager decode at width BMAX (the reference/greedy continuation below) is compiled here; the
    # grouped bodies + readouts are compiled inside capture_prefill_trace_chunked (_prepare_prefill_group_traces)
    # before the chunk-trace capture. The round-3 runs compiled the decode after the captures and hung on the first
    # grouped replay (logs/round3/itemK_test.log, logs/grp_test_all.log).
    _decode_rows(model, [100], [64], page_tables, 1)
    # ... and the per-user request path's own programs (eager masked bucket, the host GDN snapshot's concat/untilize,
    # the logits readout untilize, the host decode-slot write's slice/tilize/concat/copy): one 1-user prefill through
    # prefill_paged_slots while no trace exists (the tracker listed 19 program-cache buffers of exactly these ops,
    # compiled at the first request AFTER the captures, before the first bucket replay: logs/grp_trk2.log).
    model.pd_gdn_capture = {}
    model.prefill_paged_slots([torch.full((1, 5), 1000, dtype=torch.int32)], page_tables[:1], [0], valid_lens=[5])
    rec0, taps0 = model.pd_gdn_capture.pop(0)
    # The host decode-slot write's programs are keyed by the SLOT (slice/concat ranges, fill_cache batch_idx), so
    # every slot this test writes is compiled here (the tracker's second list: 14 slot-1 programs, logs/grp_trk3.log).
    for slot in range(1, BMAX):
        model._write_gdn_slot(slot, rec0, taps0)
    model.pd_gdn_snapshot_release(rec0, taps0)
    model.pd_gdn_capture = None
    ttnn.synchronize_device(device)
    # The served warm-up (Qwen36ForCausalLM.warmup_model_prefill): chunk trace + bucket traces + grouped traces
    # against the persistent B=1 scratch.
    t0 = time.perf_counter()
    pt_full = torch.arange(BMAX * BPU, dtype=torch.int32).reshape(1, -1)
    prev = model._bind_gdn_prefill_scratch()
    try:
        model.capture_prefill_trace_chunked(device, pt_full, chunk_size=CHUNK, capture_chunk_trace=True)
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    ttnn.synchronize_device(device)
    logger.info(
        f"[grp] warm-up {time.perf_counter() - t0:.1f}s: bucket traces {sorted(model._mb_traces)}, "
        f"grouped traces {sorted(model._grp_traces)}"
    )
    assert model._grp_traces, "no grouped trace was captured"
    return model, tok, page_tables


@run_for_blackhole()
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_grouped_prefill_matches_per_user(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    model, tok, page_tables = _setup(mesh_device)
    only = os.environ.get("QWEN36_GRP_TEST_SETS")
    only = {s.strip() for s in only.split(",")} if only else None
    failures = []
    summary = []
    try:
        for name, rows in _prompt_sets(tok).items():
            if only and name not in only:
                continue
            n = len(rows)
            lens = [int(r.shape[1]) for r in rows]
            slots = list(range(n))

            # ---- reference: per-user path (N=1 calls) ----
            model.pd_gdn_capture = {}
            ref_lg, ref_snap = [], []
            t0 = time.perf_counter()
            for u in range(n):
                lg = model.prefill_paged_slots([rows[u]], page_tables[u : u + 1], [u], valid_lens=[lens[u]])
                assert model._last_prefill_groups == []
                ref_lg.append(lg[0].reshape(-1)[: model.vocab_size].float().clone())
                ref_snap.append(_snap_copy(model, u))
            ttnn.synchronize_device(mesh_device)
            t_ref = time.perf_counter() - t0
            ref_tok = _decode_rows(model, [int(l.argmax()) for l in ref_lg], lens, page_tables, STEPS)

            # ---- grouped: one call with all users ----
            model.pd_gdn_capture = {}
            t0 = time.perf_counter()
            lgs = model.prefill_paged_slots(rows, page_tables[:n], slots, valid_lens=lens)
            ttnn.synchronize_device(mesh_device)
            t_grp = time.perf_counter() - t0
            groups = list(model._last_prefill_groups)
            grp_lg = [lg.reshape(-1)[: model.vocab_size].float().clone() for lg in lgs]
            grp_snap = [_snap_copy(model, u) for u in range(n)]
            grp_tok = _decode_rows(model, [int(l.argmax()) for l in grp_lg], lens, page_tables, STEPS)
            model.pd_gdn_capture = None

            logger.info(f"[grp] set {name}: lens={lens} groups={groups} per-user {t_ref:.2f}s grouped {t_grp:.2f}s")
            assert groups, f"set {name}: the grouped call did not use a grouped trace"
            n_exact_lg = n_exact_rec = n_exact_taps = 0
            worst = (1.0, 1.0, 1.0)
            # Per-GDN-layer divergence of user 0 (rec [n_dev, L, ...] / taps [n_dev, L, K, C]): where along the depth the
            # grouped rows first leave the per-user numerics (layer 0 exact => the divergence is accumulated rounding of
            # the M = B*bucket per-token programs, not a row/mask/conv bug).
            r_g, r_r = grp_snap[0][0].float(), ref_snap[0][0].float()
            t_g, t_r = grp_snap[0][1].float(), ref_snap[0][1].float()
            per_layer = [
                (li, float((r_g[:, li] - r_r[:, li]).abs().max()), float((t_g[:, li] - t_r[:, li]).abs().max()))
                for li in range(r_g.shape[1])
            ]
            first_bad = next((li for li, dr, dt in per_layer if dr > 0 or dt > 0), None)
            logger.info(
                f"[grp] {name} u0 per-GDN-layer max|d| rec/taps: first differing layer = {first_bad}; "
                + " ".join(f"L{li}:{dr:.3g}/{dt:.3g}" for li, dr, dt in per_layer[:6])
                + " ... "
                + " ".join(f"L{li}:{dr:.3g}/{dt:.3g}" for li, dr, dt in per_layer[-3:])
            )
            for u in range(n):
                e1, d1, p1 = _cmp(f"{name} u{u} logits", grp_lg[u], ref_lg[u])
                e2, d2, p2 = _cmp(f"{name} u{u} gdn.rec", grp_snap[u][0], ref_snap[u][0])
                e3, d3, p3 = _cmp(f"{name} u{u} gdn.taps", grp_snap[u][1], ref_snap[u][1])
                n_exact_lg += e1
                n_exact_rec += e2
                n_exact_taps += e3
                worst = (min(worst[0], p1), min(worst[1], p2), min(worst[2], p3))
                same = grp_tok[u] == ref_tok[u]
                logger.info(
                    f"[grp] {name} u{u} T={lens[u]}: tokens {'SAME' if same else 'DIFF'} "
                    f"ref={tok.decode(ref_tok[u])!r} grp={tok.decode(grp_tok[u])!r}"
                )
                if not same:
                    failures.append((name, u, lens[u], ref_tok[u], grp_tok[u]))
            if min(worst) < PCC_MIN:
                failures.append((name, "pcc", worst))
            summary.append(
                f"{name}: users={n} groups={groups} tokens_same={sum(grp_tok[u] == ref_tok[u] for u in range(n))}/{n} "
                f"exact logits/rec/taps={n_exact_lg}/{n_exact_rec}/{n_exact_taps} of {n} "
                f"min pcc logits/rec/taps={worst[0]:.6f}/{worst[1]:.6f}/{worst[2]:.6f} "
                f"per-user {1e3 * t_ref:.0f} ms grouped {1e3 * t_grp:.0f} ms"
            )
        for line in summary:
            logger.info(f"[grp] SUMMARY {line}")
        print("GRP_RESULT " + " | ".join(summary))
    finally:
        model.pd_gdn_capture = None
        model.free_kv_caches()
    assert not failures, f"greedy tokens differ / PCC < {PCC_MIN} for {len(failures)} user(s)/set(s): {failures}"


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_grouped_prefill_step_timing(mesh_device):
    """Wall clock of one N=8 x 128-token prefill step, grouped vs per-user, P/D producer configuration
    (pd_gdn_capture on: host snapshots parked, no decode-slot write). Prints GRP_TIMING."""
    if not _MULTI:
        pytest.skip("TP path only")
    model, tok, page_tables = _setup(mesh_device)
    g = torch.Generator().manual_seed(7)
    results = {}
    try:
        model.pd_skip_gdn_slot_write = True
        for isl in (128, 100):
            rows = [torch.randint(1000, 100000, (1, isl), generator=g, dtype=torch.int32) for _ in range(BMAX)]
            lens = [isl] * BMAX
            for mode in ("per_user", "grouped", "per_user", "grouped"):
                model._grp_use = mode == "grouped"
                times = []
                for rep in range(4):
                    model.pd_gdn_capture = {}
                    t0 = time.perf_counter()
                    model.prefill_paged_slots(rows, page_tables, list(range(BMAX)), valid_lens=lens)
                    ttnn.synchronize_device(mesh_device)
                    times.append(1e3 * (time.perf_counter() - t0))
                    for u in range(BMAX):
                        model.pd_gdn_snapshot_release(*model.pd_gdn_capture.pop(u))
                    if rep == 0:
                        logger.info(f"[grp] N=8 isl={isl} {mode}: groups={model._last_prefill_groups}")
                key = f"N8_isl{isl}_{mode}"
                results[key] = min(results.get(key, 1e9), min(times))
                logger.info(f"[grp] N=8 isl={isl} {mode}: {' '.join(f'{t:.0f}' for t in times)} ms")
    finally:
        model.pd_gdn_capture = None
        model.pd_skip_gdn_slot_write = False
        model._grp_use = True
        model.free_kv_caches()
    print("GRP_TIMING " + " ".join(f"{k}={v:.1f}ms" for k, v in results.items()))
