# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving-order run of the generator in ONE mesh session (README §12; WAVE_A_REVIEW GEN-7, attention P0):

    warmup (every prefill bucket, eager decode, decode trace capture)
    -> prefill A (bucket 256)                -> decode A (traced)
    -> prefill B (S = 1410: bucket 2048, global layer 0 at S >= 1024 AFTER decode steps: the L1_SMALL hazard)
    -> decode A + B (traced)
    -> prefill C (S = 546: bucket 1024) with A's and B's live block ids in the page-table tail past C's own blocks
       (the generator must zero them: A's and B's KV blocks stay bitwise unchanged)  -> decode A + B + C (traced)
    -> re-prefill A's prompt on another lane / other blocks: logits bitwise equal to the first prefill of A
    -> decode A + B + C + A' (traced)

with the real model, layers 0..3 (``MotifGenerator``, real weights). Every prefill and decode logits row is checked
against the reference: the C2 golden state after layer 3 (A, C: real C2 prompts) or the CPU reference prefix model
(layers 0-3, bf16, ``reference.load_reference_model``) on the long prompt B (en_technical + math_word_problem, 1420
tokens), each through the reference head (mean -> RMSNorm -> lm_head). Decode is teacher-forced (the next input token
is the prompt's own next token).

Run::

    scripts/devrun.sh -t 2400 -n serving_order -- python -m pytest models/demos/motif3/tests/test_serving_order.py \
        -s -p no:cacheprovider --timeout=0
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.motif3.tt.model_config import device_params

GOLDEN_DIR = Path(os.environ.get("MOTIF3_GOLDEN_STREAM_DIR", "/home/ttuser/hchang/experiments/motif-3/goldens/c2"))
N_LAYERS = 4
MAX_MODEL_LEN = 2048
NUM_BLOCKS = 512
BLOCK = 64
LOGIT_PCC_MIN = 0.99
MESH = [pytest.param((4, 8), device_params(), id="4x8")]


def stats(ref, got) -> dict:
    from models.common.utility_functions import comp_pcc

    r, g = ref.double(), got.double()
    nonfinite = int((~torch.isfinite(g)).sum())
    _, pcc = comp_pcc(r, g, 0.0)
    return {"pcc": float(pcc) if nonfinite == 0 else float("nan"), "max_abs": float((r - g).abs().max()),
            "nonfinite": nonfinite}


class RefHead:
    """Reference bf16 final head (``reduce_streams`` -> ``RMSNorm`` -> ``lm_head`` -> fp32) on ``[N, 4, 4096]``."""

    def __init__(self, src):
        from models.demos.motif3.reference.modules import RMSNorm

        self.norm = RMSNorm(4096, 1e-5).to(torch.bfloat16)
        with torch.no_grad():
            self.norm.weight.copy_(src.get("model.norm.weight").to(torch.bfloat16))
        self.lm = src.get("lm_head.weight").to(torch.bfloat16)

    @torch.no_grad()
    def __call__(self, streams):
        return torch.nn.functional.linear(self.norm(streams.to(torch.bfloat16).mean(dim=1)), self.lm).float()


class _Req:
    def __init__(self, name, ids, S, lane, blocks, states):
        self.name, self.ids, self.S, self.lane, self.blocks, self.states = name, ids, S, lane, blocks, states
        self.t = 0  # decode steps done

    @property
    def pos(self):
        return self.S + self.t


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@torch.no_grad()
def test_serving_order(mesh_device, device_params):
    from models.demos.motif3.reference import golden_stream as gs
    from models.demos.motif3.tt import generator_api as api
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR
    from models.demos.motif3.tt.weights import HFWeightLoader

    try:
        src = HFWeightLoader()
    except FileNotFoundError as e:
        pytest.skip(f"no local checkpoint: {e}")
    if not all(src.layer_available(l) for l in range(N_LAYERS)):
        pytest.skip("checkpoint layers 0-3 are not local")
    if not gs.state_path(GOLDEN_DIR, N_LAYERS - 1).is_file():
        pytest.skip("C2 golden state after layer 3 missing")
    log_fabric(mesh_device, "serving_order")
    prompts = {p.name: p for p in gs.load_prompt_set(GOLDEN_DIR / "prompts.json")}
    golden3 = {k: v[0] for k, v in gs.load_tensors(gs.state_path(GOLDEN_DIR, N_LAYERS - 1))[0].items()}
    head = RefHead(src)

    # ---- the long prompt B and its CPU reference (layers 0-3, bf16 = the C2 numerics) ---------------------------------
    ids_b = prompts["en_technical"].ids + prompts["math_word_problem"].ids  # 1420 tokens
    t0 = time.time()
    from models.demos.motif3.reference.weights import load_reference_model

    ref_model = load_reference_model(layer_ids=tuple(range(N_LAYERS)), dtype=torch.bfloat16, lazy_experts=True)
    _, xb = ref_model.model(torch.tensor([ids_b]), return_streams=True)  # streams after layer 3 [1, S, 4, 4096]
    states_b = xb[0].contiguous()
    del ref_model, xb
    print(f"[serving] CPU reference for the {len(ids_b)}-token prompt B in {time.time() - t0:.1f} s")

    settings = api.GeneratorSettings(max_batch_size=api.NUM_LANES, max_seq_len=MAX_MODEL_LEN, num_layers=N_LAYERS,
                                     weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=BLOCK)
    t0 = time.time()
    gen = MotifGenerator.create(hf_config=None, mesh_device=mesh_device, settings=settings)
    print(f"[serving] generator ({N_LAYERS} layers) created in {time.time() - t0:.1f} s")
    try:  # the generator (trace, pool, weights) is released on any outcome (review finding 9)
        failures = _serving_order_body(gen, mesh_device, api, prompts, golden3, head, ids_b, states_b)
    finally:
        gen.close()
    assert not failures, "\n".join(failures)


def kv_blocks(pool, blocks) -> list:
    """Chip 0's copy of the given KV blocks of every layer (``[n, 1, block, 576]`` each)."""
    idx = torch.tensor(blocks, dtype=torch.long)
    return [ttnn.to_torch(ttnn.get_device_tensors(t)[0])[idx].clone() for t in pool.layers]


def _serving_order_body(gen, mesh_device, api, prompts, golden3, head, ids_b, states_b) -> list:
    pool = gen.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BLOCK, num_layers=N_LAYERS)
    W = math.ceil(MAX_MODEL_LEN / BLOCK)
    t0 = time.time()
    gen.warmup_prefill(kv_cache=pool, enable_trace=False)
    gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=W)
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
    print(f"[serving] warmup + capture in {time.time() - t0:.1f} s: {gen.timings}")

    free_blocks = list(range(1, NUM_BLOCKS))
    failures, log = [], []
    live = {}

    def new_req(name, ids, S, lane, states):
        nb = math.ceil((S + 16) / BLOCK)
        blocks = [free_blocks.pop(0) for _ in range(nb)]
        return _Req(name, ids, S, lane, blocks, states)

    def check(tag, ref_logits, got):
        s = stats(ref_logits, got.float())
        top1 = int(got.float().argmax()) == int(ref_logits.argmax())
        log.append((tag, s["pcc"], top1))
        print(f"[serving] {tag}: logits pcc={s['pcc']:.6f} max_abs={s['max_abs']:.3e} top-1 {'ok' if top1 else 'DIFF'}")
        if not s["pcc"] >= LOGIT_PCC_MIN:
            failures.append(f"{tag}: {s}")

    def prefill(req, tag, stale_tail=()):
        pt = torch.zeros(W, dtype=torch.int32)
        own = math.ceil(req.S / BLOCK)
        pt[:own] = torch.tensor(req.blocks[:own], dtype=torch.int32)
        if stale_tail:  # what vLLM's persistent block table can hold past a request's blocks: other live requests' ids
            tail = torch.tensor(list(stale_tail), dtype=torch.int32).repeat(W)[: W - own]
            pt[own:] = tail
        t1 = time.time()
        logits = gen.prefill_forward(
            api.PrefillRequest(lane=req.lane, tokens=torch.tensor(req.ids[: req.S], dtype=torch.int32), page_table=pt),
            kv_cache=pool,
        )
        dt = time.time() - t1
        check(f"{tag} prefill S={req.S} (bucket {gen.cfg.prefill_bucket(req.S)}, {dt * 1e3:.0f} ms)",
              head(req.states[req.S - 1: req.S].float())[0], logits)
        return logits

    def decode(steps, tag):
        for _ in range(steps):
            tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
            pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
            table = torch.zeros(api.NUM_LANES, W, dtype=torch.int32)
            for r in live.values():
                tokens[r.lane] = r.ids[r.pos]
                pos[r.lane] = r.pos
                table[r.lane, : len(r.blocks)] = torch.tensor(r.blocks, dtype=torch.int32)
            t1 = time.perf_counter()
            out = gen.decode_forward(api.DecodeBatch(tokens=tokens, positions=pos, page_table=table), kv_cache=pool,
                                     enable_trace=True)
            dt = time.perf_counter() - t1
            for r in live.values():
                check(f"{tag} decode {r.name}@lane{r.lane} pos {r.pos} ({dt * 1e3:.1f} ms)",
                      head(r.states[r.pos: r.pos + 1].float())[0], out[r.lane])
                r.t += 1

    pa = prompts["chat_default"]
    A = new_req("A", pa.ids, len(pa.ids) - 13, 0, golden3[pa.name])  # 12 decode steps in all
    logits_a = prefill(A, "A")
    live["A"] = A
    decode(3, "A")
    B = new_req("B", ids_b, len(ids_b) - 10, 9, states_b)  # 9 decode steps
    prefill(B, "B (global layer 0 at bucket 2048 after decode)")
    live["B"] = B
    decode(4, "A+B")
    pc = prompts["python_code"]
    C = new_req("C", pc.ids, len(pc.ids) - 6, 17, golden3[pc.name])  # 5 decode steps
    others = A.blocks + B.blocks  # live blocks: C's bucket padding (positions 576..1023) must not reach them
    kv_before = kv_blocks(pool, others)
    prefill(C, "C (stale tail: A's and B's live block ids past C's own blocks)", stale_tail=others)
    kv_after = kv_blocks(pool, others)
    intact = all(torch.equal(a, b) for a, b in zip(kv_before, kv_after))
    print(f"[serving] A's and B's {len(others)} KV blocks after C's prefill with their ids in its page-table tail: "
          f"bitwise unchanged on all {len(kv_before)} layers {intact}")
    if not intact:
        failures.append("prefill C wrote through stale page-table entries into A's / B's KV blocks")
    del kv_before, kv_after
    live["C"] = C
    decode(3, "A+B+C")
    A2 = new_req("A2", pa.ids, A.S, 25, golden3[pa.name])
    logits_a2 = prefill(A2, "A' (A's prompt again, lane 25, new blocks)")
    same = torch.equal(logits_a, logits_a2)
    print(f"[serving] re-prefill of A after {sum(r.t for r in live.values())} decode rows: logits bitwise equal {same}")
    if not same:
        failures.append(f"re-prefill of A differs: max {float((logits_a.float() - logits_a2.float()).abs().max()):.3e}")
    live["A2"] = A2
    decode(2, "A+B+C+A'")
    pccs = [p for _, p, _ in log]
    print(f"[serving] {len(log)} logits rows: pcc min {min(pccs):.6f} mean {sum(pccs) / len(pccs):.6f}; top-1 "
          f"{sum(t for *_, t in log)}/{len(log)}")
    return failures


# ============================================================================================================
# long context: every bucket up to 32768 warmed, a ~17K-token prefill after decode steps, decode at long context
# ============================================================================================================
LONG_MAX_MODEL_LEN = 32768
LONG_S = 17000


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@torch.no_grad()
def test_serving_order_long_context(mesh_device, device_params):
    """Serving order at long context with the 4-layer real model and ``max_model_len`` 32768 (all 9 buckets warmed
    before the capture): prefill S=1000 (lane 0) -> decode -> prefill S=17000 (bucket 32768: global layer 0 at 32K
    after decode) on lane 9 -> decode both -> prefill of the same 17000 tokens + 1 on lane 18. No CPU reference exists
    at 17K tokens (the reference attention would need ~100 GB), so the checks are self-consistency on the device:
    the decode logits at position S (after a prefill of S tokens) vs the prefill logits of the S+1-token prompt (two
    different code paths over the same 17K-token history: FlashMLA over the paged cache vs SDPA prefill), and the
    1000-token request's decode vs its own prefill of one token more. Tokens: the 6 C2 prompts concatenated 6 times."""
    from models.demos.motif3.reference import golden_stream as gs
    from models.demos.motif3.tt import generator_api as api
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR
    from models.demos.motif3.tt.weights import HFWeightLoader

    try:
        src = HFWeightLoader()
    except FileNotFoundError as e:
        pytest.skip(f"no local checkpoint: {e}")
    if not all(src.layer_available(l) for l in range(N_LAYERS)):
        pytest.skip("checkpoint layers 0-3 are not local")
    log_fabric(mesh_device, "serving_order_long_context")
    prompts = gs.load_prompt_set(GOLDEN_DIR / "prompts.json")
    ids = [i for _ in range(6) for p in prompts for i in p.ids][: LONG_S + 8]
    settings = api.GeneratorSettings(max_batch_size=api.NUM_LANES, max_seq_len=LONG_MAX_MODEL_LEN, num_layers=N_LAYERS,
                                     weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=BLOCK)
    gen = MotifGenerator.create(hf_config=None, mesh_device=mesh_device, settings=settings)
    try:  # released on any outcome (review finding 9)
        failures = _long_context_body(gen, mesh_device, api, ids)
    finally:
        gen.close()
    assert not failures, "\n".join(failures)


def _long_context_body(gen, mesh_device, api, ids) -> list:
    nb_long = math.ceil((LONG_S + 8) / BLOCK)
    num_blocks = 1 + 3 * nb_long + 64
    pool = gen.allocate_kv_cache(num_blocks=num_blocks, block_size=BLOCK, num_layers=N_LAYERS)
    W = math.ceil(LONG_MAX_MODEL_LEN / BLOCK)
    t0 = time.time()
    gen.warmup_prefill(kv_cache=pool, enable_trace=False)
    t_wp = time.time() - t0
    gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=W)
    gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=W)
    print(f"[long] warmup prefill of {len(gen.cfg.prefill_buckets)} buckets {t_wp:.1f} s "
          f"({ {k: round(v, 1) for k, v in gen.timings.items() if k.startswith('warmup_prefill_')} }); capture "
          f"{gen.timings.get('capture_decode_s', 0):.1f} s")
    nxt = [1]

    def blocks(n):
        b = list(range(nxt[0], nxt[0] + n))
        nxt[0] += n
        return b

    def prefill(lane, S, blk):
        pt = torch.zeros(W, dtype=torch.int32)
        own = math.ceil(S / BLOCK)
        pt[:own] = torch.tensor(blk[:own], dtype=torch.int32)
        t1 = time.time()
        out = gen.prefill_forward(api.PrefillRequest(lane=lane, tokens=torch.tensor(ids[:S], dtype=torch.int32),
                                                     page_table=pt), kv_cache=pool)
        print(f"[long] prefill lane {lane} S={S} (bucket {gen.cfg.prefill_bucket(S)}): {(time.time() - t1) * 1e3:.0f} ms")
        return out.float()

    def decode(rows):
        """rows: {lane: (pos, blocks)} -> logits [32, V]"""
        tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
        pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
        table = torch.zeros(api.NUM_LANES, W, dtype=torch.int32)
        for lane, (p, blk) in rows.items():
            tokens[lane], pos[lane] = ids[p], p
            n = p // BLOCK + 1
            table[lane, :n] = torch.tensor(blk[:n], dtype=torch.int32)
        t1 = time.perf_counter()
        out = gen.decode_forward(api.DecodeBatch(tokens=tokens, positions=pos, page_table=table), kv_cache=pool,
                                 enable_trace=True)
        print(f"[long] decode {sorted(rows)} at {[rows[l][0] for l in sorted(rows)]}: "
              f"{(time.perf_counter() - t1) * 1e3:.1f} ms")
        return out.float()

    failures = []

    def same_dist(tag, a, b):
        s = stats(a, b)
        top1 = int(a.argmax()) == int(b.argmax())
        print(f"[long] {tag}: pcc={s['pcc']:.6f} max_abs={s['max_abs']:.3e} top-1 {'same' if top1 else 'DIFF'}")
        if not s["pcc"] >= LOGIT_PCC_MIN:
            failures.append(f"{tag}: {s}")

    S1, b1 = 1000, blocks(math.ceil(1010 / BLOCK))
    prefill(0, S1, b1)
    d1 = decode({0: (S1, b1)})  # position 1000
    ref1 = prefill(1, S1 + 1, blocks(math.ceil(1010 / BLOCK)))  # the same history + 1 token, other lane / blocks
    same_dist("S=1000: decode at 1000 vs prefill of 1001 tokens", ref1, d1[0])
    b2 = blocks(nb_long)
    prefill(9, LONG_S, b2)  # bucket 32768, global layer 0 at 32K after decode steps
    d2 = decode({0: (S1 + 1, b1), 9: (LONG_S, b2)})
    ref2 = prefill(18, LONG_S + 1, blocks(nb_long))
    same_dist(f"S={LONG_S}: decode at {LONG_S} vs prefill of {LONG_S + 1} tokens", ref2, d2[9])
    d3 = decode({0: (S1 + 2, b1), 9: (LONG_S + 1, b2)})
    finite = bool(torch.isfinite(d3[0]).all() and torch.isfinite(d3[9]).all())
    print(f"[long] next step finite: {finite}")
    if not finite:  # review finding 8: asserted, not only printed
        failures.append(f"decode at {S1 + 2} / {LONG_S + 1}: non-finite logits")
    return failures
