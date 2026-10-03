# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G11 — chunk I/O of resumed prefill: chunk fills through ``-1``-skip page tables and the offset-RoPE gather
(FEATURES_DESIGN §4 G11, D5, §3.1 item 4, §3.2.4).

(a) ``paged_fill_cache(cache [520, 1, 64, 576] bfp8, x [1, 1, C, 576] bfp8, fill_pt [1, C/64] int32, batch_idx=0)`` for
    C in {128, 2048, 8192} with fill tables: all real ids / leading -1 (shared full blocks below w0) / trailing -1 (pure
    bucket padding past e) / mixed (-1 runs inside) / a concatenated 4-row table (P5 packed prefill: row i of x maps
    through entry i // 64, four requests' segments each with their own -1 runs). Pass (G7 method): every written row
    bit-exact, every -1 block and every other block unchanged, on chips 0, 13 and 31. The x rows are pre-quantized
    through a host bfp8 round trip, so the expected cache is exact. Plus the eager cost per C.

(b) Offset RoPE: ``ttnn.embedding(idx [1, C] uint32, table [1, 1, 32768, 64] bf16 ROW_MAJOR, layout=TILE)`` ->
    reshape ``[1, 1, C, 64]`` for idx = min(c0 + i, 32767), c0 in {0, 128, 24576, 32704} (the last clamps), yarn and
    plain tables. Pass: bitwise equal to the host table rows; ``rotary_embedding_hf`` (prefill mode, rope role) on q_pe
    ``[1, 10, C, 64]`` / k_pe ``[1, 1, C, 64]`` with the gathered tables PCC >= 0.99999 vs x cos + rotate_half(x) sin;
    at c0 = 0 the result is bitwise the draft-1 path (TILE tables of rows 0..C-1); one gather program per C over all c0;
    the gather + rotary captured in a trace and replayed at other c0 (index rewritten in place) == eager bitwise.

Fail (a) -> per-block fills.

Run::

    scripts/devrun.sh -t 1800 -n g11 -- python -m pytest models/demos/motif3/tests/unit/gates/test_g11_chunk_io.py \
        -s -p no:cacheprovider
    scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q models/demos/motif3/tests/unit/gates/test_g11_chunk_io.py -k host
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

QUICK = gu.env_flag("MOTIF3_GATES_QUICK")
REC = gu.Recorder("G11_quick" if QUICK else "G11")
MESH = gu.mesh_params(trace_region_size=256 << 20, l1_small_size=32768)


def capture_trace(mesh_device, fn):
    """Exception-safe trace capture of ``fn()``: returns ``(trace_id, fn's result)``. If ``fn`` raises inside the
    capture, the capture is ended *and* the trace released before re-raising (a dangling capture hung
    close_mesh_device once, GATES_RESULTS §11.6)."""
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        out = fn()
    except BaseException:
        try:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        finally:
            ttnn.release_trace(mesh_device, tid)
        raise
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    return tid, out

BS, D = 64, 576
N_BLOCKS = 520
CHECK_CHIPS = (0, 13, 31)
FILL_CS = (128, 2048) if QUICK else (128, 2048, 8192)
ROPE_CS = (128,) if QUICK else (128, 2048, 8192)
C0S = (0, 32704) if QUICK else (0, 128, 24576, 32704)
MAX_POS = 32768
ROPE_PCC = 0.99999


# ----------------------------------------------------------------------------------------------------------------
# host builders
# ----------------------------------------------------------------------------------------------------------------
def fill_tables(C: int, seed: int):
    """``{name: [C / 64] int32}`` fill tables over distinct real block ids (1..N-1); -1 = skip."""
    nb = C // BS
    g = torch.Generator().manual_seed(seed)
    ids = (torch.randperm(N_BLOCKS - 1, generator=g)[:nb] + 1).to(torch.int32)
    out = {"all_real": ids.clone()}
    k = max(1, nb // 4)
    lead = ids.clone()
    lead[:k] = -1
    out["leading_neg1"] = lead
    trail = ids.clone()
    trail[nb - k :] = -1
    out["trailing_neg1"] = trail
    mixed = ids.clone()
    if nb >= 2:
        mixed[0] = -1
        mixed[nb - 1] = -1
    if nb >= 8:
        mixed[2:4] = -1
        mixed[nb // 2] = -1
    out["mixed"] = mixed
    if nb >= 4:  # four packed rows of nb/4 blocks each: [shared -1 | own | padding -1] per row
        cat = ids.clone()
        seg = nb // 4
        for r in range(4):
            a = r * seg
            if seg >= 4:
                cat[a : a + r % 2 + 1] = -1  # 1-2 shared blocks
                cat[a + seg - 1] = -1  # one pure-padding block
            elif seg >= 2:
                cat[a] = -1
        out["concat_4row"] = cat
    return out


def expected_after_fill(cache: torch.Tensor, x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """cache [N, 1, 64, 576], x [C, 576] -> the cache after ``paged_fill_cache`` (row i through entry i // 64)."""
    out = cache.clone()
    for j, b in enumerate(table.tolist()):
        if b >= 0:
            out[b, 0] = x[j * BS : (j + 1) * BS]
    return out


def rope_tables():
    """HF tables cos / sin [32768, 64] (bf16-rounded) for the yarn (global) and plain (SWA / MTP) kinds."""
    pos = torch.arange(MAX_POS)
    return {"yarn": gd.cos_sin(gd.yarn_inv_freq(), pos), "plain": gd.cos_sin(gd.plain_inv_freq(), pos)}


def offset_index(c0: int, C: int) -> torch.Tensor:
    return torch.clamp(torch.arange(c0, c0 + C), max=MAX_POS - 1).to(torch.int32)


def test_g11_host_tables_selfcheck():
    for C in FILL_CS:
        t = fill_tables(C, seed=C)
        assert set(t) >= {"all_real", "leading_neg1", "trailing_neg1", "mixed"}
        for name, tab in t.items():
            assert tab.numel() == C // BS and int(tab.max()) < N_BLOCKS
            real = tab[tab >= 0]
            assert real.unique().numel() == real.numel(), name
            if name != "all_real":
                assert int((tab < 0).sum()) >= 1, name
    cache = torch.randn(N_BLOCKS, 1, BS, D)
    x = torch.randn(128, D)
    tab = torch.tensor([-1, 7], dtype=torch.int32)
    e = expected_after_fill(cache, x, tab)
    assert torch.equal(e[7, 0], x[64:]) and torch.equal(e[:7], cache[:7]) and torch.equal(e[8:], cache[8:])
    assert int(offset_index(32704, 128)[-1]) == MAX_POS - 1 and int(offset_index(0, 4)[3]) == 3


# ----------------------------------------------------------------------------------------------------------------
# device
# ----------------------------------------------------------------------------------------------------------------
def rep_host(t: torch.Tensor, mesh_device, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))


def cache_matches(cache_tt, expected: torch.Tensor, chips=CHECK_CHIPS):
    worst = 0
    for i in chips:
        got = ttnn.to_torch(ttnn.get_device_tensors(cache_tt)[i]).float()
        if not torch.equal(got, expected):
            worst = max(worst, int((got != expected).sum()))
    return worst == 0, worst


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g11a_chunk_fill(mesh_device):
    try:
        from models.demos.motif3.tt.ccl import log_fabric

        log_fabric(mesh_device, "G11")
    except Exception as e:  # pragma: no cover
        print(f"[G11] fabric report unavailable: {e}")
    failures = []
    g = torch.Generator().manual_seed(11)
    expected = gu.host_roundtrip(torch.randn(N_BLOCKS, 1, BS, D, generator=g), ttnn.bfloat8_b)
    cache = gu.to_mesh(expected, mesh_device, ttnn.bfloat8_b)
    ok0, _ = cache_matches(cache, expected)
    assert ok0, "initial cache upload is not bit-exact"
    for C in FILL_CS:
        tables = fill_tables(C, seed=C)
        for name, tab in tables.items():
            case = f"C{C}/{name}"
            x = gu.host_roundtrip(torch.randn(1, 1, C, D, generator=g) * (1.0 + C / 4096), ttnn.bfloat8_b)
            x_tt = gu.to_mesh(x, mesh_device, ttnn.bfloat8_b)
            pt_tt = gu.to_mesh(tab[None].contiguous(), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            try:
                ttnn.experimental.paged_fill_cache(cache, x_tt, pt_tt, batch_idx=0)
            except Exception as e:
                REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}", table=tab.tolist())
                failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
                gu.free([x_tt, pt_tt])
                continue
            expected = expected_after_fill(expected, x[0, 0], tab)
            ok, nbad = cache_matches(cache, expected)
            n_skip = int((tab < 0).sum())
            rec = dict(mismatching_elements=nbad, blocks=tab.numel(), skipped=n_skip, table_head=tab[:8].tolist())
            if name == "all_real":

                def fill():
                    ttnn.experimental.paged_fill_cache(cache, x_tt, pt_tt, batch_idx=0)
                    return None

                rec["eager_us"] = gu.time_eager(mesh_device, fill, iters=10)  # rewrites the same rows: still exact
            REC.add(case, status="pass" if ok else "fail", **rec)
            if not ok:
                failures.append(f"{case}: {nbad} mismatching elements (skip entries {n_skip})")
            gu.free([x_tt, pt_tt])
    ok, nbad = cache_matches(cache, expected, chips=tuple(range(32)))
    REC.add("final/all_32_chips", status="pass" if ok else "fail", mismatching_elements=nbad)
    if not ok:
        failures.append(f"final cache differs on some chip ({nbad} elements)")
    assert not failures, "G11a failures:\n" + "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g11b_offset_rope(mesh_device):
    failures = []
    tabs = rope_tables()
    dev_tabs = {
        k: tuple(gu.to_mesh(t[None, None].contiguous(), mesh_device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for t in v)
        for k, v in tabs.items()
    }
    rope_ckc = gu.compute_cfg("HiFi4", fp32_acc=True, approx=False)  # the rope role (G8)
    g = torch.Generator().manual_seed(110)

    def gather(idx_tt, kind, C):
        out = []
        for t in dev_tabs[kind]:
            e = ttnn.embedding(idx_tt, t, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out.append(ttnn.reshape(e, (1, 1, C, 64)))
        return out

    for C in ROPE_CS:
        idx_tt = gu.to_mesh(offset_index(0, C)[None].contiguous(), mesh_device, ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        qh = torch.randn(1, 10, C, 64, generator=g).bfloat16().float()
        kh = torch.randn(1, 1, C, 64, generator=g).bfloat16().float()
        q_tt = gu.to_mesh(qh, mesh_device, ttnn.bfloat16)
        k_tt = gu.to_mesh(kh, mesh_device, ttnn.bfloat16)
        ttnn.synchronize_device(mesh_device)
        n0 = mesh_device.num_program_cache_entries()
        n_first = None  # programs after the first (kind, c0 = 0) case: no later offset or kind may add one
        eager_rot = {}
        for kind in ("yarn", "plain"):
            cos_h, sin_h = tabs[kind]
            for c0 in C0S:
                case = f"C{C}/{kind}/c0_{c0}"
                idx = offset_index(c0, C)
                ttnn.copy_host_to_device_tensor(rep_host(idx[None].contiguous(), mesh_device, ttnn.uint32), idx_tt)
                try:
                    cos_t, sin_t = gather(idx_tt, kind, C)
                    gc, gs = gu.read_dev(cos_t, 0)[0, 0], gu.read_dev(sin_t, 0)[0, 0]
                    gc31 = gu.read_dev(cos_t, 31)[0, 0]
                    rq = ttnn.experimental.rotary_embedding_hf(q_tt, cos_t, sin_t, is_decode_mode=False, compute_kernel_config=rope_ckc)
                    rk = ttnn.experimental.rotary_embedding_hf(k_tt, cos_t, sin_t, is_decode_mode=False, compute_kernel_config=rope_ckc)
                    oq, ok_ = gu.read_dev(rq, 0), gu.read_dev(rk, 0)
                    gu.free([rq, rk])
                except Exception as e:
                    REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
                    failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
                    continue
                want_c, want_s = cos_h[idx.long()], sin_h[idx.long()]
                bit_ok = torch.equal(gc, want_c) and torch.equal(gs, want_s) and torch.equal(gc31, want_c)
                sq = gu.compare(gd.rope_golden(qh, want_c, want_s), oq)
                sk = gu.compare(gd.rope_golden(kh, want_c, want_s), ok_)
                ok = bit_ok and sq["pcc"] >= ROPE_PCC and sk["pcc"] >= ROPE_PCC
                rec = dict(tables_bitwise=bit_ok, pcc_q=sq["pcc"], pcc_k=sk["pcc"], max_abs_q=sq["max_abs"], clamped=c0 + C > MAX_POS)
                eager_rot[(kind, c0)] = (oq, ok_)
                if c0 == 0:  # the draft-1 path: TILE tables of rows 0..C-1 uploaded from the host
                    tc = gu.to_mesh(cos_h[:C][None, None].contiguous(), mesh_device, ttnn.bfloat16)
                    ts = gu.to_mesh(sin_h[:C][None, None].contiguous(), mesh_device, ttnn.bfloat16)
                    rq = ttnn.experimental.rotary_embedding_hf(q_tt, tc, ts, is_decode_mode=False, compute_kernel_config=rope_ckc)
                    rec["c0_0_bitwise_eq_draft1_tables"] = bool(torch.equal(gu.read_dev(rq, 0), oq))
                    ok &= rec["c0_0_bitwise_eq_draft1_tables"]
                    gu.free([rq, tc, ts])
                gu.free([cos_t, sin_t])
                REC.add(case, status="pass" if ok else "fail", **rec)
                if not ok:
                    failures.append(f"{case}: tables_bitwise={bit_ok} pcc_q={sq['pcc']:.7f} pcc_k={sk['pcc']:.7f}")
                if n_first is None:
                    ttnn.synchronize_device(mesh_device)
                    n_first = mesh_device.num_program_cache_entries()
        ttnn.synchronize_device(mesh_device)
        n1 = mesh_device.num_program_cache_entries()
        if n_first is not None and n1 != n_first:
            failures.append(f"C{C}: {n1 - n_first} programs compiled for later offsets / kinds (expected 0)")

        # ---- trace: gather + rotary captured at c0 = 128, replayed at the other offsets ----
        kind = "yarn"
        ttnn.copy_host_to_device_tensor(rep_host(offset_index(128, C)[None].contiguous(), mesh_device, ttnn.uint32), idx_tt)

        def step():
            cos_t, sin_t = gather(idx_tt, kind, C)
            rq = ttnn.experimental.rotary_embedding_hf(q_tt, cos_t, sin_t, is_decode_mode=False, compute_kernel_config=rope_ckc)
            gu.free([cos_t, sin_t])
            return rq

        tr_ok, tr_detail = True, []
        tid, tout = capture_trace(mesh_device, lambda: step())
        try:
            for c0 in C0S:
                ttnn.copy_host_to_device_tensor(rep_host(offset_index(c0, C)[None].contiguous(), mesh_device, ttnn.uint32), idx_tt)
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                eq = torch.equal(gu.read_dev(tout, 0), eager_rot[(kind, c0)][0])
                tr_detail.append((c0, eq))
                tr_ok &= eq
        finally:
            ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(tout)
        ttnn.synchronize_device(mesh_device)
        n2 = mesh_device.num_program_cache_entries()
        gather_us = gu.time_eager(mesh_device, lambda: gather(idx_tt, kind, C), iters=20)
        gather_tr, raw = gu.time_traced(mesh_device, lambda: gather(idx_tt, kind, C), ops_per_trace=32, reps=9)
        # programs: per C the gather (embedding [+ reshape]) and the two rotary shapes; none may depend on c0 or kind
        progs_ok = n_first is not None and n1 == n_first and n2 == n1
        REC.add(f"C{C}/program_cache_and_trace", status="pass" if (tr_ok and progs_ok) else "fail",
                programs_first_case=(n_first - n0) if n_first is not None else None,
                programs_added_by_other_offsets_and_kinds=(n1 - n_first) if n_first is not None else None,
                programs_added_by_trace=n2 - n1, trace_replay=tr_detail,
                gather_eager_us=gather_us, gather_traced_us=gather_tr, gather_traced_raw_us=raw)
        if not tr_ok:
            failures.append(f"C{C}: traced gather+rotary != eager: {tr_detail}")
        if n2 != n1:
            failures.append(f"C{C}: the trace capture compiled {n2 - n1} program(s)")
        gu.free([idx_tt, q_tt, k_tt])
    assert not failures, "G11b failures:\n" + "\n".join(failures)
