# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B7 device gates: traced prefill (``MOTIF3_PREFILL_TRACE``; docs/OPTIMIZATION_PLAN.md §3.3 B7, prototype
logs/opt/phaseA/m8, results logs/opt/phaseB/B7) at 53 layers, in the serving order of the bridge:

    warmup_prefill (every shape eager) -> warmup_decode(eager: stages the decode paths AND the traced prefill
    shapes' persistent inputs, F3N R3) -> eager references -> warmup_decode(trace: the decode traces, then the
    prefill traces, R5) -> the same calls again (traced) -> ...

``test_prefill_trace_plain`` (the production launch: chunked prefill, prefix caching / KV-R, packed prefill, device
sampling, ``("plain", "all")`` decode trace):

* traced == eager bitwise (host logits) for sp0 128 rows of 128 / 77 / 5 tokens, rows with an sp1 128 tail
  (``[8064, 8192)``: the second vLLM step of an 8K prompt under the 8064-token budget; a prefix hit resumed at 4096
  -> 4160), and the same
  calls run eagerly AFTER the capture (``chunk_observer`` forces eager) are bitwise equal too;
* decode after a traced prefill == decode after the eager prefill of the same blocks (6 traced decode steps each);
* no program compiled by any capture or afterwards (F3N R2), the traced shapes counted, the trace bytes of each
  prefill trace and the trace region in all;
* a soak: traced sp0 128 / traced tail / eager packed pass (8 x 100 tokens) / traced decode steps, interleaved,
  every repeat bitwise equal to the first;
* timing (report): ``prefill_forward_batch`` wall of each case eager before the capture, traced, eager after.

``test_prefill_trace_spec`` (the MTP launch: + the MTP layer, ``MOTIF3_SPEC_VERIFY=auto``: T32-spec + T64 decode
traces, the default 256 MiB trace region): both prefill traces fit next to the two decode traces; traced == eager
bitwise for the logits AND the KV written (main layers 0 / 1 / 52 and the MTP cache, chip 0, every block the row
wrote), sp0 128 and the sp1 tail; spec decode steps after a traced prefill == after the eager prefill.

Run (each ~10-15 min)::

    scripts/devrun.sh -t 2700 -n b7_dev -- env OMP_WAIT_POLICY=PASSIVE MOTIF3_PREFILL_TRACE=128 \
        python -m pytest models/demos/motif3/tests/test_prefill_trace_device.py -s -p no:cacheprovider --timeout=0

``MOTIF3_B7_OUT`` (optional): a directory for the JSON report of each test. ``MOTIF3_PREFILL_TRACE`` defaults to 128
here (the test sets it when unset).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Any, Dict, List

import pytest
import torch

from models.demos.motif3.tt import generator_api as api
from models.demos.motif3.tt import prefill_plan as PP

NL = 53
BS, MAX_LEN, NUM_BLOCKS = 64, 32768, 4129
WIDTH = MAX_LEN // BS
OUT_DIR = os.environ.get("MOTIF3_B7_OUT")


def log(msg: str) -> None:
    print(f"[b7 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dump(name: str, rep: Dict[str, Any]) -> None:
    if OUT_DIR:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
            json.dump(rep, f, indent=1, default=str)


class Blocks:
    def __init__(self):
        self.next = 1

    def take(self, n: int) -> List[int]:
        assert self.next + n <= NUM_BLOCKS
        out = list(range(self.next, self.next + n))
        self.next += n
        return out


def new_row(gen_: torch.Generator, blocks: Blocks, length: int) -> Dict[str, Any]:
    return {"ids": torch.randint(100, 200000, (length,), generator=gen_, dtype=torch.int32),
            "blocks": blocks.take(api.cdiv(length, BS) + 1)}  # fmt: skip


def req(row, start: int, end: int, lane: int = 0) -> api.PrefillRequest:
    pt = torch.zeros(WIDTH, dtype=torch.int32)
    n = api.cdiv(end, BS)
    pt[:n] = torch.tensor(row["blocks"][:n], dtype=torch.int32)
    return api.PrefillRequest(lane=lane, tokens=row["ids"][:end].clone(), page_table=pt, start=start)


def _session(monkeypatch, *, spec: bool):
    import ttnn

    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.generator import MotifGenerator
    from models.demos.motif3.tt.model_config import DEFAULT_WEIGHTS_DIR, open_motif_mesh

    if not os.environ.get("MOTIF3_PREFILL_TRACE"):
        monkeypatch.setenv("MOTIF3_PREFILL_TRACE", "128")
    mesh = open_motif_mesh()
    rep = log_fabric(mesh, "b7_prefill_trace")
    assert "TORUS_XY" in str(rep.get("committed")).upper() and not rep.get("degraded"), rep
    A = api.DEFAULT_PREFILL_ALIGNMENT
    budget = PP.recommended_budget(api.DEFAULT_PREFILL_SPAN_CAP, A)
    settings = api.GeneratorSettings(
        max_batch_size=api.NUM_LANES, max_seq_len=MAX_LEN, num_layers=NL, kv_cache_dtype="bfp8",
        weights_path=str(DEFAULT_WEIGHTS_DIR), block_size=BS, weights_source="TT cache (b7)", chunked_prefill=True,
        prefix_caching=True, max_num_batched_tokens=budget, long_prefill_token_threshold=budget,
        spec_tokens=1 if spec else 0, spec_verify="auto" if spec else "packed", packed_prefill=True,
    )  # fmt: skip
    gen = MotifGenerator.create(hf_config=None, mesh_device=mesh, settings=settings)
    return ttnn, mesh, gen, rep


def _nprog(mesh) -> int:
    return int(mesh.num_program_cache_entries())


def _timed(ttnn, mesh, fn):
    t0 = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(mesh)
    return time.perf_counter() - t0, out


def _eager(gen):
    """Force eager solo chunks after the capture (``chunk_observer`` set: the traced path is skipped)."""

    class _E:
        def __enter__(self):
            gen.chunk_observer = lambda job, ch, X: None

        def __exit__(self, *a):
            gen.chunk_observer = None

    return _E()


def _kv_rows(ttnn, pool, blocks, mtp: bool) -> List[torch.Tensor]:
    """Chip 0's copy of ``blocks`` of main layers 0, 1, 52 (and the MTP cache)."""
    idx = torch.tensor(blocks, dtype=torch.long)
    caches = [pool.layers[i] for i in (0, 1, NL - 1)] + ([pool.mtp] if mtp else [])
    return [ttnn.to_torch(ttnn.get_device_tensors(t)[0])[idx].clone() for t in caches]


@pytest.mark.timeout(3600)
@torch.no_grad()
def test_prefill_trace_plain(monkeypatch):
    ttnn, mesh, gen, fab = _session(monkeypatch, spec=False)
    R: Dict[str, Any] = {"fabric": {k: str(v) for k, v in fab.items()}, "env": {
        k: v for k, v in os.environ.items() if k.startswith(("MOTIF3_", "OMP_"))}}  # fmt: skip
    fails: List[str] = []
    try:
        assert gen.prefill_trace_buckets, "MOTIF3_PREFILL_TRACE is off"
        pool = gen.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BS, num_layers=NL)
        gen.warmup_prefill(kv_cache=pool, enable_trace=False)
        gen.enable_device_sampling()
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=WIDTH)
        staged = sorted(gen.prefill_traces)
        assert staged == [(PP.SP0, 128), (PP.SP1, 128)], staged
        g = torch.Generator().manual_seed(7077)
        blocks = Blocks()
        rows = {"sp0_128_full": new_row(g, blocks, 128), "sp0_128_77": new_row(g, blocks, 77),
                "sp0_128_5": new_row(g, blocks, 5), "tail_8192": new_row(g, blocks, 8192),
                "tail_4160": new_row(g, blocks, 4160), "sp0_256_200": new_row(g, blocks, 200)}  # fmt: skip
        calls = {"sp0_128_full": (0, 128), "sp0_128_77": (0, 77), "sp0_128_5": (0, 5), "tail_8192": (8064, 8192),
                 "tail_4160": (4096, 4160), "sp0_256_200": (0, 200)}  # fmt: skip
        # the cached prefixes: a hit at 4096; the first vLLM step of an 8K prompt under the 8064-token budget
        gen.prefill_forward_batch([req(rows["tail_4160"], 0, 4096)], kv_cache=pool)
        gen.prefill_forward_batch([req(rows["tail_8192"], 0, 8064)], kv_cache=pool)
        pk_rows = [new_row(g, blocks, 100) for _ in range(8)]
        pk_reqs = [req(r_, 0, 100, lane=i) for i, r_ in enumerate(pk_rows)]
        N = 5
        ref, t_eager = {}, {}
        for name, (s, e) in calls.items():
            ts = []
            for _ in range(N + 1):
                dt, out = _timed(ttnn, mesh, lambda: gen.prefill_forward_batch([req(rows[name], s, e)], kv_cache=pool))
                ts.append(dt)
                if name not in ref:
                    ref[name] = out[0].clone()
                elif not torch.equal(out[0], ref[name]):
                    fails.append(f"eager {name} not deterministic before the capture")
            t_eager[name] = statistics.median(ts[1:])
            rows[name]["last_logits"] = ref[name]
        ts = []
        for _ in range(N + 1):
            dt, out = _timed(ttnn, mesh, lambda: gen.prefill_forward_batch(pk_reqs, kv_cache=pool))
            ts.append(dt)
        pk_ref = out.clone()
        t_eager["packed_8x100"] = statistics.median(ts[1:])
        R["eager_before_capture_s"] = t_eager
        log(f"eager before the capture: { {k: round(v, 4) for k, v in t_eager.items()} }")

        n0 = _nprog(mesh)
        tr0 = gen._trace_region_used()
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=WIDTH)
        n1 = _nprog(mesh)
        traced = {k: t.traced for k, t in gen.prefill_traces.items()}
        R["captures"] = {f"{k[0]}:{k[1]}": {"traced": t.traced, "trace_MiB_per_chip": t.trace_bytes / 2**20,
                                             "capture_s": gen.timings.get(f"capture_prefill_{k[0]}_{k[1]}_s")}
                         for k, t in gen.prefill_traces.items()}  # fmt: skip
        R["trace_region_MiB_per_chip"] = {"before_captures": (tr0 or 0) / 2**20,
                                          "after_captures": (gen._trace_region_used() or 0) / 2**20,
                                          "free_after": (gen._trace_region_free() or 0) / 2**20}  # fmt: skip
        R["programs"] = {"before_captures": n0, "after_captures": n1}
        log(f"captures: {R['captures']}; trace region {R['trace_region_MiB_per_chip']}; programs {n0} -> {n1}")
        assert all(traced.values()), f"a prefill shape was not captured: {traced}"
        if n1 != n0:
            fails.append(f"the captures compiled {n1 - n0} programs")

        # traced (and eager after the capture) == eager before, bitwise; timings
        t_tr, t_ea = {}, {}
        for name, (s, e) in calls.items():
            c0 = gen.stats["traced_prefill_chunks"]
            ts, te = [], []
            for _ in range(N + 1):
                dt, out = _timed(ttnn, mesh, lambda: gen.prefill_forward_batch([req(rows[name], s, e)], kv_cache=pool))
                ts.append(dt)
                if not torch.equal(out[0], ref[name]):
                    fails.append(f"traced {name} != eager")
                with _eager(gen):
                    dt, out = _timed(ttnn, mesh,
                                     lambda: gen.prefill_forward_batch([req(rows[name], s, e)], kv_cache=pool))
                te.append(dt)
                if not torch.equal(out[0], ref[name]):
                    fails.append(f"eager after the capture {name} != eager before")
            n_tr = gen.stats["traced_prefill_chunks"] - c0
            want = 0 if name == "sp0_256_200" else N + 1
            if n_tr != want:
                fails.append(f"{name}: {n_tr} traced chunks, expected {want}")
            t_tr[name], t_ea[name] = statistics.median(ts[1:]), statistics.median(te[1:])
        ts = []
        for _ in range(N + 1):
            dt, out = _timed(ttnn, mesh, lambda: gen.prefill_forward_batch(pk_reqs, kv_cache=pool))
            ts.append(dt)
            if not torch.equal(out, pk_ref):
                fails.append("packed 8 x 100 after the capture != before")
        t_ea["packed_8x100"] = statistics.median(ts[1:])
        R["traced_s"], R["eager_after_capture_s"] = t_tr, t_ea
        log(f"traced: { {k: round(v, 4) for k, v in t_tr.items()} }")
        log(f"eager after the capture: { {k: round(v, 4) for k, v in t_ea.items()} }")

        # decode after a traced prefill == after the eager prefill (same blocks)
        def decode_n(row, plen, n, lane=0):
            tok = int(torch.argmax(row["last_logits"]))
            outs = []
            pt = torch.zeros(api.NUM_LANES, WIDTH, dtype=torch.int32)
            nb = api.cdiv(plen + n, BS)
            pt[lane, :nb] = torch.tensor(row["blocks"][:nb], dtype=torch.int32)
            for i in range(n):
                tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
                pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
                tokens[lane], pos[lane] = tok, plen + i
                lg = gen.decode_forward(api.DecodeBatch(tokens=tokens, positions=pos, page_table=pt), kv_cache=pool,
                                        enable_trace=True)  # fmt: skip
                outs.append(lg[lane].clone())
                tok = int(torch.argmax(lg[lane]))
            return torch.stack(outs)

        R["decode_compat"] = {}
        for name in ("sp0_128_77", "tail_4160", "tail_8192"):
            s, e = calls[name]
            with _eager(gen):
                gen.prefill_forward_batch([req(rows[name], s, e)], kv_cache=pool)
            d_e = decode_n(rows[name], e, 6)
            gen.prefill_forward_batch([req(rows[name], s, e)], kv_cache=pool)  # traced, same blocks
            d_t = decode_n(rows[name], e, 6)
            ok = bool(torch.equal(d_e, d_t))
            R["decode_compat"][name] = ok
            if not ok:
                fails.append(f"decode after the traced prefill of {name} != after the eager one")
        log(f"decode compat: {R['decode_compat']}")

        # soak: traced prefill <-> traced decode <-> eager packed passes
        bad = []
        n_soak = int(os.environ.get("MOTIF3_B7_SOAK", "10"))
        d_ref = None
        for it in range(n_soak):
            for name in ("sp0_128_77", "tail_4160", "tail_8192"):
                s, e = calls[name]
                o = gen.prefill_forward_batch([req(rows[name], s, e)], kv_cache=pool)
                if not torch.equal(o[0], ref[name]):
                    bad.append((it, name))
                d = decode_n(rows["sp0_128_77"], 77, 1)
                if d_ref is None:
                    d_ref = d
                elif not torch.equal(d, d_ref):
                    bad.append((it, name, "decode"))
            if not torch.equal(gen.prefill_forward_batch(pk_reqs, kv_cache=pool), pk_ref):
                bad.append((it, "packed"))
        R["soak"] = {"iters": n_soak, "mismatches": bad}
        if bad:
            fails.append(f"soak mismatches {bad[:5]}")
        R["programs"]["end"] = _nprog(mesh)
        if R["programs"]["end"] != n1:
            fails.append(f"{R['programs']['end'] - n1} programs compiled after the captures")
        R["stats"] = {k: v for k, v in gen.stats.items() if "prefill" in k or "packed" in k or k == "decode_steps"}
        log(f"soak {R['soak']}; programs {R['programs']}; stats {R['stats']}")
    finally:
        R["fails"] = fails
        dump("plain", R)
        try:
            gen.close()
        finally:
            from models.demos.motif3.tt.model_config import close_motif_mesh

            close_motif_mesh(mesh)
    assert not fails, "\n".join(fails)


@pytest.mark.timeout(3600)
@torch.no_grad()
def test_prefill_trace_spec(monkeypatch):
    monkeypatch.setenv("OMP_WAIT_POLICY", os.environ.get("OMP_WAIT_POLICY", "PASSIVE"))
    ttnn, mesh, gen, fab = _session(monkeypatch, spec=True)
    R: Dict[str, Any] = {"fabric": {k: str(v) for k, v in fab.items()}, "env": {
        k: v for k, v in os.environ.items() if k.startswith(("MOTIF3_", "OMP_"))}}  # fmt: skip
    fails: List[str] = []
    try:
        assert gen.spec_launch and gen.mtp_enabled and gen.serving_paths == [("spec", "all_split"), ("wide", "all_split")]
        region = gen.cfg.trace_region_size
        R["trace_region_size"] = region
        pool = gen.allocate_kv_cache(num_blocks=NUM_BLOCKS, block_size=BS, num_layers=NL)
        gen.warmup_prefill(kv_cache=pool, enable_trace=False)
        gen.enable_device_sampling()
        gen.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=WIDTH)
        g = torch.Generator().manual_seed(7078)
        blocks = Blocks()
        rows = {"sp0_128_77": new_row(g, blocks, 77), "tail_4160": new_row(g, blocks, 4160),
                "tail_8192": new_row(g, blocks, 8192)}  # fmt: skip
        calls = {"sp0_128_77": (0, 77), "tail_4160": (4096, 4160), "tail_8192": (8064, 8192)}
        gen.prefill_forward_batch([req(rows["tail_4160"], 0, 4096)], kv_cache=pool)
        gen.prefill_forward_batch([req(rows["tail_8192"], 0, 8064)], kv_cache=pool)
        ref, kv_ref = {}, {}
        for name, (s, e) in calls.items():
            ref[name] = gen.prefill_forward_batch([req(rows[name], s, e)], kv_cache=pool)[0].clone()
            w = rows[name]["blocks"][s // BS: api.cdiv(e, BS)]
            kv_ref[name] = (w, _kv_rows(ttnn, pool, w, True))
        n0 = _nprog(mesh)
        gen.warmup_decode(kv_cache=pool, enable_trace=True, page_table_width=WIDTH)
        n1 = _nprog(mesh)
        used, free = gen._trace_region_used() or 0, gen._trace_region_free() or 0
        R["captures"] = {f"{k[0]}:{k[1]}": {"traced": t.traced, "mtp_trace": t.mtp_trace_id is not None,
                                             "trace_MiB_per_chip": t.trace_bytes / 2**20}
                         for k, t in gen.prefill_traces.items()}  # fmt: skip
        R["trace_region_MiB_per_chip"] = {"used": used / 2**20, "free": free / 2**20, "size": region / 2**20}
        R["programs"] = {"before_captures": n0, "after_captures": n1}
        log(f"MTP launch captures {R['captures']}; trace region {R['trace_region_MiB_per_chip']}; programs {n0}->{n1}")
        if not all(t.traced and t.mtp_trace_id is not None for t in gen.prefill_traces.values()):
            fails.append(f"the prefill traces did not fit next to the T32-spec + T64 traces: {R['captures']}")
        if n1 != n0:
            fails.append(f"the captures compiled {n1 - n0} programs")
        R["eq"] = {}
        for name, (s, e) in calls.items():
            c0, m0 = gen.stats["traced_prefill_chunks"], gen.stats["mtp_fills"]
            out = gen.prefill_forward_batch([req(rows[name], s, e)], kv_cache=pool)[0]
            w, kv_e = kv_ref[name]
            kv_t = _kv_rows(ttnn, pool, w, True)
            eq = {"logits": bool(torch.equal(out, ref[name])),
                  "kv": [bool(torch.equal(a, b)) for a, b in zip(kv_e, kv_t)],
                  "traced_chunks": gen.stats["traced_prefill_chunks"] - c0, "mtp_fills": gen.stats["mtp_fills"] - m0}
            R["eq"][name] = eq
            if not (eq["logits"] and all(eq["kv"])) or eq["traced_chunks"] != 1:
                fails.append(f"{name}: traced != eager on the MTP launch: {eq}")
        log(f"MTP launch traced vs eager: {R['eq']}")

        def spec_n(row, plen, n, lane=0):
            tok = int(torch.argmax(row["last"]))
            pt = torch.zeros(api.NUM_LANES, WIDTH, dtype=torch.int32)
            nb = api.cdiv(plen + n + 1, BS)
            pt[lane, :nb] = torch.tensor(row["blocks"][:nb], dtype=torch.int32)
            outs = []
            for i in range(n):
                tokens = torch.zeros(api.NUM_LANES, dtype=torch.int32)
                pos = torch.full((api.NUM_LANES,), -1, dtype=torch.int32)
                tokens[lane], pos[lane] = tok, plen + i
                res = gen.decode_forward_spec(
                    api.SpecDecodeBatch(tokens=tokens, positions=pos, page_table=pt,
                                        draft_tokens=torch.full((api.NUM_LANES,), -1, dtype=torch.int32)),
                    kv_cache=pool, enable_trace=True, want_logits=False)  # fmt: skip
                a, m = int(res.argmax[lane, 0]), int(res.mtp_argmax[lane, 0])  # m reads the MTP cache the prefill filled
                outs.append((a, m))
                tok = a
            return outs

        R["spec_decode_compat"] = {}
        for name in ("sp0_128_77", "tail_4160"):
            s, e = calls[name]
            rows[name]["last"] = ref[name]
            with _eager(gen):
                gen.prefill_forward_batch([req(rows[name], s, e)], kv_cache=pool)
            d_e = spec_n(rows[name], e, 6)
            gen.prefill_forward_batch([req(rows[name], s, e)], kv_cache=pool)
            d_t = spec_n(rows[name], e, 6)
            R["spec_decode_compat"][name] = {"eq": d_e == d_t, "eager": d_e, "traced": d_t}
            if d_e != d_t:
                fails.append(f"spec decode after the traced prefill of {name} != after the eager one: {d_e} {d_t}")
        log(f"spec decode compat: {R['spec_decode_compat']}")
        R["programs"]["end"] = _nprog(mesh)
        if R["programs"]["end"] != n1:
            fails.append(f"{R['programs']['end'] - n1} programs compiled after the captures")
    finally:
        R["fails"] = fails
        dump("spec", R)
        try:
            gen.close()
        finally:
            from models.demos.motif3.tt.model_config import close_motif_mesh

            close_motif_mesh(mesh)
    assert not fails, "\n".join(fails)
