# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two-process round trip of the speculative-decoding state (MTP head KV + hidden row) through the P/D transfer.

Role P (PDMTP_ROLE=P): a prefill instance with QWEN36_SPEC_MTP semantics -- the MTP prefill hook installed, GDN
capture on, no decode-slot writes -- prefills the test prompts, stages each request through the plugin connector's
real producer path (`_WorkerSide.stage_after_step`: `pd_transfer.export_kv_blocks` over the 17 attention layers,
`export_mtp_hidden`, `pack_payload` version 2) and writes the staged bytes + headers to PDMTP_DIR with a manifest.
Role D (PDMTP_ROLE=D): a decode instance with the head (draft widths, verify plans) prefills the same prompts locally
(the in-process reference: head KV / hidden row readback + the speculative loop of test_mtp_spec_scratch), then
feeds every staged payload through the connector's consumer drain (`_drain_fetched`: `import_kv_blocks` of all 17
layers) + the runner-side slot import (`import_gdn_slot`, `import_mtp_hidden`) into OTHER slots / block ids, and
asserts: the imported head KV blocks and hidden row equal the locally computed ones bitwise (torch.equal on
readback), the speculative loop from the imported state commits the same stream as the local loop, and both equal
the plain traced decode (R <= 32 plans). Cases: 8 users x 128 tokens (w=8, k=1), one 4k+ prompt (2 chunks + tail,
w=1, k=2), one 8k prompt (4 chunks, w=1, k=2). Cost lines: the connector's own "[pd] staged" / "[pd] pulled" logs
and PDMTP_COST summaries (bytes, export / import ms) + PDMTP_PREFILL (MTP prefill ms on P).

Run (half A):  scripts/pd_mtp_run.sh <tag> PDMTP_ROLE=P   then   scripts/pd_mtp_run.sh <tag> PDMTP_ROLE=D
Env: PDMTP_DIR (payload dir), PDMTP_MIN_TOKENS (48), PDMTP_CASES ("a,b,c").
"""
import json
import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_mtp_spec_scratch import (
    GSM8K_PARQUET,
    _prefill,
    _reference_stream,
    _spec_loop,
)
from models.demos.blackhole.qwen36.tests.test_verify_step_scratch import (
    AICLK_MHZ,
    DecodeRef,
    _pin_aiclk,
    _stream_compare,
)
from models.demos.blackhole.qwen36.tt import pd_transfer
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.mtp_head import MTPHead
from models.demos.blackhole.qwen36.tt.verify_step import VerifyStep

ROLE = os.environ.get("PDMTP_ROLE", "P").upper()
PAYLOAD_DIR = os.environ.get("PDMTP_DIR", "/home/eslim/experiments/qwen36/logs/pdmtp_payload")
MIN_TOKENS = int(os.environ.get("PDMTP_MIN_TOKENS", "48"))
CASES = [c for c in os.environ.get("PDMTP_CASES", "a,b,c").split(",") if c]
BMAX = 8
BPU = 136  # blocks per user = 8704 positions (8k prompt + generation)
CHUNK = 2048
N_MAIN_LAYERS = 16

# (name, users, prompt tokens, w, k): a = 8 short prompts (bucket 128), b = 2 chunks + tail, c = 4 exact chunks
CASE_DEFS = {"a": ("a", 8, 128, 8, 1), "b": ("b", 1, CHUNK * 2 + 100, 1, 2), "c": ("c", 1, CHUNK * 4, 1, 2)}
PERM_SHIFT = 3  # D imports user u's payload into slot (u + 3) % BMAX (other slot, other block ids)


class FakeEngine:
    """Stands in for the Mooncake TransferEngine (the bytes never leave this host: they go through a file)."""

    def register_memory(self, ptr, nbytes):
        return 0

    def transfer_sync_read(self, segment, dst, src, nbytes):
        raise AssertionError("no pull in the file-based hand-off")


def _connector_module():
    from vllm_tt_plugin.kv_connector import tt_mooncake_connector as mc

    return mc


def _worker(mc, producer, model):
    stub = SimpleNamespace(
        _is_producer=producer, _is_consumer=not producer, _side_host="127.0.0.1", _side_port=0, _block_size=BLOCK_SIZE
    )
    w = mc._WorkerSide(stub)
    w.engine = FakeEngine()
    w.pool = mc._HostBufferPool(w.engine, w._engine_lock, "staging" if producer else "receive", shm=False)
    w.model = model
    w.runner = SimpleNamespace(_req_state_slot={}, pd_pending_gdn={})
    return w


def build_prompt_ids(tok):
    """Prompt token ids per case from GSM8K text: 8 distinct 128-token prompts, one 4196-token, one 8192-token."""
    import pandas as pd

    df = pd.read_parquet(GSM8K_PARQUET)
    out = {}
    if "a" in CASES:
        ids = []
        for u in range(8):
            text = " ".join(str(df.question[i]) for i in range(30 + 6 * u, 30 + 6 * u + 6))
            t = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :128].to(torch.int32)
            assert t.shape[1] == 128, t.shape
            ids.append(t)
        out["a"] = ids
    long_text = " ".join(str(df.question[i]) for i in range(100, 100 + 400))
    long_ids = tok(long_text, return_tensors="pt", add_special_tokens=False).input_ids.to(torch.int32)
    for c in ("b", "c"):
        if c in CASES:
            n = CASE_DEFS[c][2]
            assert long_ids.shape[1] >= n, (long_ids.shape, n)
            out[c] = [long_ids[:, :n].contiguous()]
    return out


def _head_kv_readback(model, block_ids):
    """(k, v) host bf16 of the MTP layer (the last export pair) + the main layers, for the given blocks."""
    kv = pd_transfer.export_kv_blocks(model, block_ids)
    assert len(kv) == N_MAIN_LAYERS + 1, len(kv)
    return kv


def _blocks(page_tables, slot, T):
    n_blocks = -(-T // BLOCK_SIZE)
    return [int(b) for b in page_tables[slot][:n_blocks]]


def _common_setup(device, widths):
    device.enable_program_cache()
    _pin_aiclk(AICLK_MHZ)
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE)
    logger.info(f"[pdmtp] model load {time.perf_counter() - t0:.1f}s")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = build_prompt_ids(tok)
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    buckets = {128, CHUNK}
    for c, ids in prompts.items():
        for t in ids:
            tail = t.shape[1] % CHUNK
            if t.shape[1] < CHUNK:
                buckets.add(Qwen36Model._mask_bucket_for(t.shape[1]))
            elif tail:
                buckets.add(Qwen36Model._mask_bucket_for(tail))
    return model, tok, prompts, page_tables, sorted(buckets)


def _prefill_warmup(model, device, page_tables):
    """The served prefill warm-up: chunk trace + masked-bucket traces + slot writes + device history repack."""
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
    logger.info(f"[pdmtp] prefill warmup {time.perf_counter() - t0:.1f}s")


# ============================================================================================== role P
def run_producer(device):
    mc = _connector_module()
    model, tok, prompts, page_tables, buckets = _common_setup(device, widths=())
    os.makedirs(PAYLOAD_DIR, exist_ok=True)
    manifest = {"cases": [], "buckets": buckets}
    head = None
    try:
        # prefill-only head (a P instance never drafts), programs compiled before any capture
        head = MTPHead(model, page_tables=None, widths=(), buckets=buckets, sdpa_pt_blocks=BPU)
        for b in buckets:
            head.compile_prefill(b)
        _prefill_warmup(model, device, page_tables)
        model.prefill_hidden_hook = head.prefill_hook
        _prefill(model, [prompts[CASES[0]][0]], page_tables, [0])  # warm 1-user prefill (lazy allocations)
        head.pending_rows.clear()
        # the producer role as the connector's bind_runner sets it, then its export warm-up (post_warmup)
        model.pd_gdn_capture = {}
        model.pd_skip_gdn_slot_write = True
        pd_transfer.export_warmup(model, max_bucket=256)
        w = _worker(mc, True, model)

        for c in CASES:
            name, n_users, T, wdt, k = CASE_DEFS[c]
            ids = prompts[c]
            users = list(range(n_users))
            calls0, wall0 = head.stats["prefill_calls"], head.stats["prefill_wall"]
            t0 = time.perf_counter()
            lens, first = _prefill(model, ids, page_tables, users)
            t_pf = time.perf_counter() - t0
            mtp_calls = head.stats["prefill_calls"] - calls0
            mtp_ms = 1e3 * (head.stats["prefill_wall"] - wall0) / n_users
            print(
                f"PDMTP_PREFILL case={name} T={T} users={n_users}: prefill_paged_slots {1e3 * t_pf / n_users:.1f} ms/req "
                f"incl. MTP prefill {mtp_ms:.1f} ms/req ({mtp_calls // n_users} hook call(s)/req, eager)",
                flush=True,
            )
            case = {"name": name, "T": T, "w": wdt, "k": k, "users": users, "first": first, "reqs": []}
            meta = mc.TTMooncakeConnectorMetadata()
            for u in users:
                rid = f"{name}-u{u}"
                w.runner._req_state_slot[rid] = u
                meta.stage.append(
                    mc.StageReq(req_id=rid, block_ids=[int(b) for b in page_tables[u]], num_tokens=T, transfer_id=rid)
                )
            st0 = dict(w.stats)
            w.stage_after_step(meta)
            n_st = w.stats["staged"] - st0["staged"]
            assert n_st == n_users, f"staged {n_st} of {n_users}"
            print(
                f"PDMTP_COST case={name} T={T} staged={n_st} bytes/req={(w.stats['staged_bytes'] - st0['staged_bytes']) // n_users} "
                f"stage_ms/req={(w.stats['stage_ms'] - st0['stage_ms']) / n_users:.1f} "
                f"(last export: {pd_transfer.LAST_EXPORT_TIMING})",
                flush=True,
            )
            for u in users:
                rid = f"{name}-u{u}"
                st = w._staged.pop(rid)
                path = os.path.join(PAYLOAD_DIR, f"{rid}.pt")
                torch.save({"buf": st.buf[: st.nbytes].clone(), "header": st.header}, path)
                w.pool.release(st.buf)
                hdr = st.header
                assert hdr["version"] == 2 and hdr["n_attn_layers"] == N_MAIN_LAYERS, hdr
                assert hdr["mtp"] == {"n_layers": 1, "hidden": True}, hdr.get("mtp")
                case["reqs"].append(
                    {
                        "req_id": rid,
                        "slot": u,
                        "path": path,
                        "nbytes": st.nbytes,
                        "digest": mc.payload_digest(st.buf, st.nbytes),
                        "tokens": ids[u].reshape(-1).tolist(),
                    }
                )
            manifest["cases"].append(case)
        with open(os.path.join(PAYLOAD_DIR, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        print(f"PDMTP_P_DONE cases={[c['name'] for c in manifest['cases']]} dir={PAYLOAD_DIR}", flush=True)
    finally:
        model.prefill_hidden_hook = None
        model.pd_gdn_capture = None
        model.pd_skip_gdn_slot_write = False
        model.free_kv_caches()


# ============================================================================================== role D
def run_consumer(device):
    mc = _connector_module()
    with open(os.path.join(PAYLOAD_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    cases = [c for c in manifest["cases"] if c["name"] in CASES]
    widths = sorted(set(c["w"] for c in cases))
    plan_keys = sorted(set((c["w"], c["k"]) for c in cases))
    model, tok, prompts, page_tables, buckets = _common_setup(device, widths)
    buckets = sorted(set(buckets) | set(manifest["buckets"]))
    steps, refs = {}, {}
    head = None
    failures = []
    try:
        # --- persistent buffers, then compile everything (VERIFY_W32_AUDIT.md), then the prefill warm-up + captures
        head = MTPHead(model, page_tables, widths=widths, buckets=buckets)
        for w, k in plan_keys:
            steps[(w, k)] = VerifyStep(model, w, k + 1, page_tables[:w], keep_hidden=True)
            head.bind_plan(steps[(w, k)].plan)
        for w in widths:
            refs[w] = DecodeRef(model, w, page_tables[:w])
            refs[w].compile()
            head.compile_step(w)
        for (w, k), vs in steps.items():
            vs.compile()
            head.compile_select(vs.plan)
        for b in buckets:
            head.compile_prefill(b)
        ttnn.synchronize_device(device)
        _prefill_warmup(model, device, page_tables)
        model.prefill_hidden_hook = head.prefill_hook
        _prefill(model, [prompts[CASES[0]][0]], page_tables, [0])
        for w in widths:
            refs[w].capture()
            head.capture_step(w)
        for vs in steps.values():
            vs.capture()
        # the consumer's post_warmup: KV import traces, GDN host packer, per-slot GDN import traces
        pd_transfer.import_warmup(model, max_bucket=256)
        pd_transfer.get_gdn_host_packer(model)
        pd_transfer.get_traced_importer(model).precapture(range(BMAX))
        wk = _worker(mc, False, model)

        for case in cases:
            name, T, w, k = case["name"], case["T"], case["w"], case["k"]
            ids = [torch.tensor([r["tokens"]], dtype=torch.int32) for r in case["reqs"]]
            users = list(range(len(ids)))
            vs = steps[(w, k)]
            n_ref = MIN_TOKENS + 3 * (k + 1) + 2
            # --- local reference: prefill (hook), readback, plain decode stream, speculative stream
            lens, first = _prefill(model, ids, page_tables, users)
            assert first == case["first"], f"case {name}: first token differs from P's ({first} vs {case['first']})"
            ref_kv = [_head_kv_readback(model, _blocks(page_tables, u, T)) for u in users]
            ref_hidden = [head.get_hidden_in(u).clone() for u in users]
            plain, _ = _reference_stream(refs[w], first, lens, n_ref)
            lens2, first2 = _prefill(model, ids, page_tables, users)
            assert first2 == first
            res_local = _spec_loop(vs, head, w, k, lens2, first2, MIN_TOKENS)
            for s in users:
                i, _ = _stream_compare(res_local["streams"][s], plain[s])
                assert i is None, f"case {name}: local speculative stream != plain decode at {i} (user {s})"
            # --- scrub the destination slots so a stale local state cannot pass as an import
            dst = [(u + PERM_SHIFT) % BMAX for u in users] if len(users) > 1 else [0]
            scrub_ids = [torch.flip(t, dims=[1]).contiguous() for t in ids]
            _prefill(model, scrub_ids, page_tables, dst)
            # --- import every payload through the connector drain + the runner-side slot import
            t_imp = {}
            for u, slot in zip(users, dst):
                r = case["reqs"][u]
                t0 = time.perf_counter()
                saved = torch.load(r["path"])
                buf, header = saved["buf"], saved["header"]
                assert mc.payload_digest(buf, header["nbytes"]) == r["digest"], "payload file corrupt"
                rr = mc.RecvReq(r["req_id"], [int(b) for b in page_tables[slot]], "127.0.0.1", 0, r["req_id"], T)
                wk._inflight[rr.req_id] = rr
                wk._fetched.put(
                    mc._Fetched(
                        rr, buf, header, (lambda: None), "pull", 0.0, time.perf_counter() - t0, None, None, None
                    )
                )
                st0 = dict(wk.stats)
                wk._drain_fetched()
                entry = wk.runner.pd_pending_gdn.pop(rr.req_id)
                rec, gdn, extra = entry[0], entry[1], entry[4]
                t1 = time.perf_counter()
                pd_transfer.import_gdn_slot(model, slot, rec, gdn)
                assert extra["mtp_hidden"] is not None, "payload without the MTP hidden row"
                pd_transfer.import_mtp_hidden(model, slot, extra["mtp_hidden"])
                ttnn.synchronize_device(device)
                t_imp[u] = (
                    wk.stats["import_ms"] - st0["import_ms"],
                    1e3 * (time.perf_counter() - t1),
                    header["nbytes"],
                )
                wk.take_finished(set())
            print(
                f"PDMTP_COST case={name} T={T} pulled={len(users)} bytes/req={sum(v[2] for v in t_imp.values()) // len(users)} "
                f"kv_import_ms/req={sum(v[0] for v in t_imp.values()) / len(users):.1f} "
                f"gdn+hidden_import_ms/req={sum(v[1] for v in t_imp.values()) / len(users):.1f}",
                flush=True,
            )
            # --- bit-exactness of the imported state vs the locally computed one
            for u, slot in zip(users, dst):
                got = _head_kv_readback(model, _blocks(page_tables, slot, T))
                for li, ((k_ref, v_ref), (k_got, v_got)) in enumerate(zip(ref_kv[u], got)):
                    if not (torch.equal(k_ref, k_got) and torch.equal(v_ref, v_got)):
                        failures.append(f"case {name} user {u} -> slot {slot}: KV layer {li} differs")
                if not torch.equal(ref_hidden[u], head.get_hidden_in(slot)):
                    failures.append(f"case {name} user {u} -> slot {slot}: hidden row differs")
            logger.info(
                f"[pdmtp] case {name}: head KV (layer {N_MAIN_LAYERS}) + 16 main layers + hidden row torch.equal on "
                f"{len(users)} import(s): {'OK' if not [f for f in failures if f.startswith(f'case {name}')] else 'FAIL'}"
            )
            # --- speculative loop from the imported state (rows = slots; user u sits at slot dst[u])
            if len(users) > 1:
                lens_i = [0] * BMAX
                first_i = [0] * BMAX
                for u, slot in zip(users, dst):
                    lens_i[slot], first_i[slot] = lens[u], first[u]
                res_imp = _spec_loop(vs, head, w, k, lens_i, first_i, MIN_TOKENS)
                for u, slot in zip(users, dst):
                    i, n = _stream_compare(res_imp["streams"][slot], res_local["streams"][u])
                    if i is not None or n < MIN_TOKENS:
                        failures.append(
                            f"case {name} user {u} -> slot {slot}: imported stream differs at {i} ({n} cmp)"
                        )
            else:
                res_imp = _spec_loop(vs, head, w, k, lens, first, MIN_TOKENS)
                i, n = _stream_compare(res_imp["streams"][0], res_local["streams"][0])
                if i is not None or n < MIN_TOKENS:
                    failures.append(f"case {name}: imported stream differs at {i} ({n} cmp)")
            print(
                f"PDMTP_EXACT case={name} T={T} w={w} k={k} R={vs.plan.R}: local_spec==plain_decode=True "
                f"imported_spec==local_spec={not [f for f in failures if f.startswith(f'case {name}') and 'stream' in f]} "
                f"kv+hidden_bitwise={not [f for f in failures if f.startswith(f'case {name}') and 'stream' not in f]} "
                f"accept_len local {res_local['accept_len_mean']:.2f} imported {res_imp['accept_len_mean']:.2f} "
                f"text {tok.decode(res_imp['streams'][dst[0]][:24])!r}",
                flush=True,
            )
        st = head.stats
        print(
            f"PDMTP_TIMING D: {st['prefill_calls']} MTP prefill hook calls {1e3 * st['prefill_wall'] / max(1, st['prefill_calls']):.1f} ms each; "
            f"{st['draft_steps']} draft steps {1e3 * st['draft_wall'] / max(1, st['draft_steps']):.2f} ms",
            flush=True,
        )
    finally:
        model.prefill_hidden_hook = None
        for vs in steps.values():
            vs.release()
        for r in refs.values():
            r.release()
        if head is not None:
            head.release()
        model.free_kv_caches()
    assert not failures, "\n".join(failures)


@run_for_blackhole()
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_pd_mtp_transfer(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    if ROLE == "P":
        run_producer(mesh_device)
    elif ROLE == "D":
        run_consumer(mesh_device)
    else:
        pytest.fail(f"PDMTP_ROLE={ROLE}: expected P or D")
