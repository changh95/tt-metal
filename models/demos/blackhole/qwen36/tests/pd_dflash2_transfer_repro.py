# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two-process round trip of the DFlash2 drafter's CONTEXT K/V through the P/D transfer (payload version 3, KV group
"dflash2"; tt/aux_hidden.py, tt/pd_transfer.py kv groups, the plugin's TTMooncakeConnector).

Role P (PDDF_ROLE=P): a prefill instance with QWEN36_SPEC_DRAFTER=dflash2 semantics -- ``model.prefill_aux_layers`` =
the drafter's target layers, ``model.prefill_aux_hook`` = ``DFlash2ContextPrefillHook`` over the (stub) context
projector, GDN capture on, no decode-slot writes -- prefills the test prompts (eager masked buckets, traced 2048
chunks), stages each request through the connector's real producer path (``_WorkerSide.stage_after_step`` ->
``pd_transfer.export_kv_blocks`` + ``export_kv_groups`` + ``pack_payload`` version 3) and writes the staged bytes,
headers and its own staged context rows to PDDF_DIR with a manifest.
Role D (PDDF_ROLE=D): a decode instance with the (stub) drafter caches registered as KV group "dflash2" prefills the
same prompts locally with the same hook (traced masked buckets: QWEN36_PREFILL_BUCKET_TRACE=1 -> the cross-path
reference rows; plus the eager chunked path for the long prompts), then feeds every staged payload through the
connector's consumer drain (``_drain_fetched``: ``import_kv_blocks`` + ``import_kv_groups``) into OTHER slots / block
ids and asserts: the drafter cache blocks read back equal the locally projected rows and P's staged rows bitwise
(torch.equal), for every shipped position; entry[4]["kv_groups"]["dflash2"] carries the group's metadata.
Cases: 8 users x 128 tokens (one masked bucket), one 4196-token prompt (2 chunks + tail), one 8192-token prompt
(4 exact chunks; the context window ships the last 2048 positions = 32 of 128 blocks).
Cost lines: the connector's "[pd] staged" / "[pd] pulled" logs and PDDF_COST / PDDF_PREFILL summaries.

Run (half A):  scripts/pd_mtp_run.sh <tag> TEST=models/demos/blackhole/qwen36/tests/pd_dflash2_transfer_repro.py PDDF_ROLE=P
          then scripts/pd_mtp_run.sh <tag> TEST=... PDDF_ROLE=D QWEN36_PREFILL_BUCKET_TRACE=1
Env: PDDF_DIR (payload dir), PDDF_CASES ("a,b,c"), PDDF_BASELINE=1 (role P: no aux hook -> the plain 16-layer producer,
the byte / cost baseline of the same blocks), QWEN36_DFLASH2_CONTEXT_WINDOW (2048).
"""
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

CASES = [c for c in os.environ.get("PDDF_CASES", "a,b,c").split(",") if c]
os.environ.setdefault("PDMTP_CASES", ",".join(CASES))  # the shared prompt builder reads it at import

from models.common.utility_functions import run_for_blackhole  # noqa: E402
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS  # noqa: E402
from models.demos.blackhole.qwen36.tests.dflash2_stub import (  # noqa: E402
    StubContextProjector,
    allocate_stub_drafter_kv,
)
from models.demos.blackhole.qwen36.tests.pd_mtp_transfer_repro import (  # noqa: E402
    BMAX,
    BPU,
    CASE_DEFS,
    CHUNK,
    N_MAIN_LAYERS,
    PERM_SHIFT,
    _common_setup,
    _connector_module,
    _prefill_warmup,
    _worker,
)
from models.demos.blackhole.qwen36.tests.test_mtp_spec_scratch import _prefill  # noqa: E402
from models.demos.blackhole.qwen36.tt import aux_hidden as ah  # noqa: E402
from models.demos.blackhole.qwen36.tt import pd_transfer  # noqa: E402

ROLE = os.environ.get("PDDF_ROLE", "P").upper()
PAYLOAD_DIR = os.environ.get("PDDF_DIR", "/home/eslim/experiments/qwen36/logs/pddf_payload")
BASELINE = os.environ.get("PDDF_BASELINE", "0") == "1"
GROUP = "dflash2"


def _install_hook(model, buckets):
    """The P-side DFlash2 prefill path on this instance: aux layers (before the prefill warm-up captures), the stub
    projector, the context hook, its programs compiled for every segment width (compile-first)."""
    model.prefill_aux_layers = ah.aux_layers_for(model)
    projector = StubContextProjector(model)
    hook = ah.DFlash2ContextPrefillHook(model, projector, block_size=BLOCK_SIZE)
    hook.compile(sorted(set(buckets) | {CHUNK}))
    model.prefill_aux_hook = hook
    return hook


def _rows_of(stage):
    return [(k.clone(), v.clone()) for k, v in stage.rows()], int(stage.first_pos), int(stage.n_tokens)


def _rows_equal(a, b):
    return len(a) == len(b) and all(torch.equal(ka, kb) and torch.equal(va, vb) for (ka, va), (kb, vb) in zip(a, b))


# ============================================================================================== role P
def run_producer(device):
    mc = _connector_module()
    model, tok, prompts, page_tables, buckets = _common_setup(device, widths=())
    os.makedirs(PAYLOAD_DIR, exist_ok=True)
    manifest = {"cases": [], "buckets": buckets, "baseline": BASELINE}
    hook = None
    try:
        if not BASELINE:
            hook = _install_hook(model, buckets)
        _prefill_warmup(model, device, page_tables)
        _prefill(model, [prompts[CASES[0]][0]], page_tables, [0])  # warm 1-user prefill (lazy allocations)
        if hook is not None:
            hook.store.clear()
            hook.stats.update(calls=0, skipped=0, wall=0.0, gather=0.0, project=0.0, tokens=0)
        model.pd_gdn_capture = {}
        model.pd_skip_gdn_slot_write = True
        pd_transfer.export_warmup(model, max_bucket=256)
        w = _worker(mc, True, model)

        for c in CASES:
            name, n_users, T, _, _ = CASE_DEFS[c]
            ids = prompts[c]
            users = list(range(n_users))
            st0 = dict(hook.stats) if hook is not None else None
            t0 = time.perf_counter()
            lens, first = _prefill(model, ids, page_tables, users)
            t_pf = time.perf_counter() - t0
            if hook is not None:
                d = {k: hook.stats[k] - st0[k] for k in ("calls", "skipped", "wall", "gather", "project", "tokens")}
                print(
                    f"PDDF_PREFILL case={name} T={T} users={n_users}: prefill_paged_slots {1e3 * t_pf / n_users:.1f} ms/req; "
                    f"aux hook {d['calls'] / n_users:.0f} call(s)/req ({d['skipped'] / n_users:.0f} skipped by the window), "
                    f"{1e3 * d['wall'] / n_users:.1f} ms/req = gather {1e3 * d['gather'] / n_users:.1f} + "
                    f"project(stub, incl. host readback) {1e3 * d['project'] / n_users:.1f}; "
                    f"{d['tokens'] / n_users:.0f} positions staged/req",
                    flush=True,
                )
            else:
                print(
                    f"PDDF_PREFILL case={name} T={T} users={n_users}: prefill_paged_slots {1e3 * t_pf / n_users:.1f} ms/req (baseline)",
                    flush=True,
                )
            case = {"name": name, "T": T, "users": users, "first": first, "reqs": []}
            local_rows = {}
            if hook is not None:
                for u in users:
                    local_rows[u] = _rows_of(hook.store[u])  # a copy: the export pops the stage
            meta = mc.TTMooncakeConnectorMetadata()
            for u in users:
                rid = f"{name}-u{u}"
                w.runner._req_state_slot[rid] = u
                meta.stage.append(
                    mc.StageReq(req_id=rid, block_ids=[int(b) for b in page_tables[u]], num_tokens=T, transfer_id=rid)
                )
            s0 = dict(w.stats)
            w.stage_after_step(meta)
            n_st = w.stats["staged"] - s0["staged"]
            assert n_st == n_users, f"staged {n_st} of {n_users}"
            print(
                f"PDDF_COST case={name} T={T} staged={n_st} bytes/req={(w.stats['staged_bytes'] - s0['staged_bytes']) // n_users} "
                f"stage_ms/req={(w.stats['stage_ms'] - s0['stage_ms']) / n_users:.1f} "
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
                assert hdr["version"] == 3 and hdr["n_attn_layers"] == N_MAIN_LAYERS, hdr
                assert "mtp" not in hdr, hdr.get("mtp")
                if BASELINE:
                    assert "kv_groups" not in hdr, hdr.get("kv_groups")
                else:
                    g = hdr["kv_groups"][GROUP]
                    rows, first_pos, n_tok = local_rows[u]
                    assert g["n_layers"] == ah.DFLASH2_N_LAYERS and g["kv_heads"] == ah.DFLASH2_KV_HEADS, g
                    assert g["head_dim"] == ah.DFLASH2_HEAD_DIM and g["block_size"] == BLOCK_SIZE, g
                    assert g["first_pos"] == first_pos and g["n_tokens"] == n_tok == T - first_pos, (
                        g,
                        first_pos,
                        n_tok,
                        T,
                    )
                    n_blk = -(-T // BLOCK_SIZE) - first_pos // BLOCK_SIZE
                    assert g["block_index"] == list(range(first_pos // BLOCK_SIZE, first_pos // BLOCK_SIZE + n_blk)), g
                    # the payload's group tensors are exactly the staged rows in block layout
                    groups = mc.unpack_kv_groups(st.buf[: st.nbytes], hdr)
                    gkv, _ = groups[GROUP]
                    pay_rows = [
                        (pd_transfer.kv_blocks_to_rows(k, n_tok), pd_transfer.kv_blocks_to_rows(v, n_tok))
                        for k, v in gkv
                    ]
                    assert _rows_equal(pay_rows, rows), f"{rid}: payload group rows != staged rows"
                    torch.save({"rows": rows, "first_pos": first_pos, "n_tokens": n_tok}, path + ".rows")
                case["reqs"].append(
                    {
                        "req_id": rid,
                        "slot": u,
                        "path": path,
                        "nbytes": st.nbytes,
                        "digest": mc.payload_digest(st.buf, st.nbytes),
                        "tokens": ids[u].reshape(-1).tolist(),
                        "kv_group": hdr.get("kv_groups", {}).get(GROUP),
                    }
                )
            manifest["cases"].append(case)
        with open(os.path.join(PAYLOAD_DIR, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        print(f"PDDF_P_DONE cases={[c['name'] for c in manifest['cases']]} dir={PAYLOAD_DIR}", flush=True)
    finally:
        model.prefill_aux_hook = None
        model.prefill_aux_layers = ()
        model.pd_gdn_capture = None
        model.pd_skip_gdn_slot_write = False
        model.free_kv_caches()


# ============================================================================================== role D
def run_consumer(device):
    mc = _connector_module()
    with open(os.path.join(PAYLOAD_DIR, "manifest.json")) as f:
        manifest = json.load(f)
    assert not manifest.get(
        "baseline"
    ), "the D role needs a payload set with the dflash2 group (PDDF_BASELINE unset on P)"
    cases = [c for c in manifest["cases"] if c["name"] in CASES]
    model, tok, prompts, page_tables, buckets = _common_setup(device, widths=())
    failures = []
    hook = None
    try:
        # --- the drafter's caches (KV group), the local hook, then the prefill warm-up (captures) and the importers
        group = allocate_stub_drafter_kv(model, BMAX * BPU)
        hook = _install_hook(model, buckets)
        _prefill_warmup(model, device, page_tables)
        _prefill(model, [prompts[CASES[0]][0]], page_tables, [0])
        hook.store.clear()
        pd_transfer.import_warmup(model, max_bucket=256)
        pd_transfer.kv_group_import_warmup(model, max_bucket=256)
        pd_transfer.get_gdn_host_packer(model)
        pd_transfer.get_traced_importer(model).precapture(range(BMAX))
        wk = _worker(mc, False, model)
        traced_buckets = bool(model._mb_traces)
        logger.info(
            f"[pddf] D prefill path: masked buckets {'traced' if traced_buckets else 'eager'}, chunk trace replays"
        )

        for case in cases:
            name, T = case["name"], case["T"]
            ids = [torch.tensor([r["tokens"]], dtype=torch.int32) for r in case["reqs"]]
            users = list(range(len(ids)))
            # --- local reference: the same prefill (traced chunk path + masked buckets) -> staged rows
            lens, first = _prefill(model, ids, page_tables, users)
            assert first == case["first"], f"case {name}: first token differs from P's ({first} vs {case['first']})"
            local = {u: _rows_of(hook.pop(u)) for u in users}
            eager_ok = None
            if T > CHUNK:
                # the eager chunked path (no chunk trace) must stage the same rows as the traced replays
                tid, model._chunked_trace_id = model._chunked_trace_id, None
                try:
                    _prefill(model, ids, page_tables, users)
                finally:
                    model._chunked_trace_id = tid
                eager = {u: _rows_of(hook.pop(u)) for u in users}
                eager_ok = all(_rows_equal(eager[u][0], local[u][0]) and eager[u][1:] == local[u][1:] for u in users)
                if not eager_ok:
                    failures.append(f"case {name}: eager chunked path rows != traced chunk path rows")
            # --- scrub the destination slots / blocks with other prompts (their staged rows are discarded)
            dst = [(u + PERM_SHIFT) % BMAX for u in users] if len(users) > 1 else [PERM_SHIFT]
            scrub_ids = [torch.flip(t, dims=[1]).contiguous() for t in ids]
            _prefill(model, scrub_ids, page_tables, dst)
            for s in dst:
                hook.pop(s)
            # --- import every payload through the connector drain + the runner-side slot import
            t_imp = {}
            metas = {}
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
                s0 = dict(wk.stats)
                wk._drain_fetched()
                entry = wk.runner.pd_pending_gdn.pop(rr.req_id)
                rec, gdn, extra = entry[0], entry[1], entry[4]
                t1 = time.perf_counter()
                pd_transfer.import_gdn_slot(model, slot, rec, gdn)
                ttnn.synchronize_device(device)
                assert extra["mtp_hidden"] is None, "a dflash2 payload must carry no MTP row"
                assert GROUP in extra["kv_groups"], extra
                metas[u] = extra["kv_groups"][GROUP]
                t_imp[u] = (wk.stats["import_ms"] - s0["import_ms"], 1e3 * (time.perf_counter() - t1), header["nbytes"])
                wk.take_finished(set())
            print(
                f"PDDF_COST case={name} T={T} pulled={len(users)} bytes/req={sum(v[2] for v in t_imp.values()) // len(users)} "
                f"kv+group_import_ms/req={sum(v[0] for v in t_imp.values()) / len(users):.1f} "
                f"gdn_import_ms/req={sum(v[1] for v in t_imp.values()) / len(users):.1f}",
                flush=True,
            )
            # --- bit-exactness: drafter cache blocks at the destination == local rows == P's staged rows
            for u, slot in zip(users, dst):
                m = metas[u]
                rows_l, first_pos, n_tok = local[u]
                if (int(m["first_pos"]), int(m["n_tokens"])) != (first_pos, n_tok):
                    failures.append(
                        f"case {name} user {u}: window {m['first_pos']}+{m['n_tokens']} vs local {first_pos}+{n_tok}"
                    )
                    continue
                blocks = [int(page_tables[slot][i]) for i in m["block_index"]]
                got = pd_transfer.read_kv_group_blocks(model, group, blocks)
                got_rows = [
                    (pd_transfer.kv_blocks_to_rows(k, n_tok), pd_transfer.kv_blocks_to_rows(v, n_tok)) for k, v in got
                ]
                p_saved = torch.load(case["reqs"][u]["path"] + ".rows")
                ok_local = _rows_equal(got_rows, rows_l)
                ok_p = _rows_equal(got_rows, p_saved["rows"]) and p_saved["first_pos"] == first_pos
                if not ok_local:
                    diffs = [
                        (j, float((k.float() - kl.float()).abs().max()), float((v.float() - vl.float()).abs().max()))
                        for j, ((k, v), (kl, vl)) in enumerate(zip(got_rows, rows_l))
                    ]
                    failures.append(f"case {name} user {u} -> slot {slot}: imported drafter K/V != local rows {diffs}")
                if not ok_p:
                    failures.append(f"case {name} user {u} -> slot {slot}: imported drafter K/V != P's staged rows")
            ok = not [f for f in failures if f.startswith(f"case {name}")]
            n_blk_ship = len(metas[users[0]]["block_index"])
            print(
                f"PDDF_EXACT case={name} T={T} users={len(users)}: imported_drafter_kv==local_rows==P_rows={ok} "
                f"window first_pos={metas[users[0]]['first_pos']} n_tokens={metas[users[0]]['n_tokens']} "
                f"blocks_shipped={n_blk_ship}/{-(-T // BLOCK_SIZE)} eager_chunked==traced={eager_ok} "
                f"masked_buckets_D={'traced' if traced_buckets else 'eager'} first_token_match=True",
                flush=True,
            )
        st = hook.stats
        print(
            f"PDDF_TIMING D: {st['calls']} aux hook calls, {1e3 * st['wall'] / max(1, st['calls']):.1f} ms each "
            f"(gather {1e3 * st['gather'] / max(1, st['calls']):.1f}, project {1e3 * st['project'] / max(1, st['calls']):.1f})",
            flush=True,
        )
    finally:
        model.prefill_aux_hook = None
        model.prefill_aux_layers = ()
        model.free_kv_caches()
    assert not failures, "\n".join(failures)


@run_for_blackhole()
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_pd_dflash2_transfer(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    if ROLE == "P":
        run_producer(mesh_device)
    elif ROLE == "D":
        run_consumer(mesh_device)
    else:
        pytest.fail(f"PDDF_ROLE={ROLE}: expected P or D")
