# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G10 — resumed ("sp1") SWA attention: the 128-token tail from the paged cache and the square ``[tail | chunk]``
windowed SDPA (FEATURES_DESIGN §4 G10, D3, §3.2.3).

(a) Tail gather: ``ttnn.slice(cache [4129, 1, 64, 576] bfp8, start_t [blk, 0, 0, 0], end_t [blk + 1, 1, 64, 576],
    slice_dim=0, num_devices=4129)`` -- the bounds live in persistent ``[4]`` int32 device tensors, so one program
    serves every block id and the gather is trace-safe (the LM-head precedent, ``tt/lm_head.py``). Checked: bitwise
    equal to the host block for blk in {1, 2, 2063, 4127, 4128} + 3 random ids, eager and inside a trace replayed with
    rewritten bounds; the program cache grows by one slice program over all ids; then the tail itself (two block
    slices -> ``concat(dim=2)`` -> bf16) eager and traced, with its cost.

(b) Square windowed SDPA: ``scaled_dot_product_attention(Q_cat [1,10,128+C,192] (rows < 128 zero), K_cat / V_cat
    [1,2,128+C,192] = [tail 128 | chunk C], is_causal=True, sliding_window_size=129, scale=1.0, q/k 128/128,
    sdpa_prefill)`` -> rows [128, 128+C). Square row 128 + i is absolute position p = s + i; causal + window give it
    exactly keys [p - 128, p]. For C in {128, 1024, 8192} and s in {128, 4096, 30720} (24576 for C = 8192, design R6):
    PCC >= 0.999 vs fp32 window attention at absolute positions and vs the draft-1 single-shot SDPA rows [s, s + C)
    (also whether the two are bitwise equal); probes at chunk rows {0, 1, 63, 64, 127, C - 1}: key p - 128 attended
    (output +1 in the probe's value dims), p - 129 not (-1 would leak), p + 1 not (+3 would leak), |delta| <= 1e-3; the
    window-128 / window-130 negative controls must be detected. Cost: eager us of the square call vs the draft-1 SWA
    SDPA of a C-row chunk ((1 + 128/C)x expected).

Fail (a) -> S-C (C++ patch passing ``sliding_window_size`` through the chunked wrappers; needs a ttnn rebuild).

Run::

    scripts/devrun.sh -t 2400 -n g10 -- python -m pytest models/demos/motif3/tests/unit/gates/test_g10_swa_tail.py \
        -s -p no:cacheprovider
    scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q models/demos/motif3/tests/unit/gates/test_g10_swa_tail.py -k host
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

QUICK = gu.env_flag("MOTIF3_GATES_QUICK")
REC = gu.Recorder("G10_quick" if QUICK else "G10")
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

BS, D_LAT = 64, 576
N_BIG = 4129  # serving pool size per layer (MOTIF3_KV_POOL_TOKENS 262144 / 64 + 1)
NQ, NKV, DQK, DV = 10, 2, 192, 128
TAIL = 128
WINDOW = gd.WINDOW  # 129 keys including the current one
PCC_PASS = 0.999
PROBE_TOL = 1e-3
SQUARE_CASES = [(128, 128), (128, 4096), (128, 30720), (1024, 128), (1024, 4096), (1024, 30720),
                (8192, 128), (8192, 4096), (8192, 24576)]  # (C, s), s + C <= 32768 (R6)  # fmt: skip
if QUICK:
    SQUARE_CASES = [(128, 128), (1024, 4096)]


def probe_rows(C: int):
    return sorted({0, 1, 63, 64, 127, C - 1})


# ----------------------------------------------------------------------------------------------------------------
# host builders (pure torch)
# ----------------------------------------------------------------------------------------------------------------
def random_seq(S: int, seed: int):
    """Full-sequence expanded GQA tensors: Q [NQ, S, 192] (scale folded), K [NKV, S, 192], V [NKV, S, 192] (cols
    128.. zero, the draft-1 V padding)."""
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(NQ, S, DQK, generator=g) * gd.SCALE_SWA).bfloat16().float()
    k = torch.randn(NKV, S, DQK, generator=g).bfloat16().float()
    v = torch.zeros(NKV, S, DQK)
    v[..., :DV] = torch.randn(NKV, S, DV, generator=g).bfloat16().float()
    return q, k, v


def to_square(q_chunk, k, v, s: int, C: int):
    """[tail | chunk] layout: Q_cat = [0_128 | Q_chunk], K_cat / V_cat = keys [s - 128, s + C)."""
    qc = torch.cat([torch.zeros(q_chunk.shape[0], TAIL, q_chunk.shape[-1]), q_chunk], dim=1)
    return qc, k[:, s - TAIL : s + C], v[:, s - TAIL : s + C]


@torch.no_grad()
def window_golden(q, k, v, q0: int, window: int = WINDOW, block: int = 512):
    """fp32 sliding-window GQA attention: q ``[NQ, R, d]`` rows at absolute positions q0 .. q0 + R - 1 over the full
    sequence k / v ``[NKV, S, d]``; row p sees keys [p - window + 1, p]. Returns ``[NQ, R, dv]``."""
    nq, R, _ = q.shape
    rep = nq // k.shape[0]
    out = torch.empty(nq, R, v.shape[-1])
    for i0 in range(0, R, block):
        i1 = min(R, i0 + block)
        lo, hi = max(0, q0 + i0 - (window - 1)), q0 + i1
        kp = torch.arange(lo, hi)
        qp = torch.arange(q0 + i0, q0 + i1)
        allow = (kp[None] <= qp[:, None]) & (kp[None] >= qp[:, None] - (window - 1))
        for g in range(k.shape[0]):
            sc = q[g * rep : (g + 1) * rep, i0:i1].float() @ k[g, lo:hi].float().T
            sc = sc.masked_fill(~allow[None], float("-inf"))
            out[g * rep : (g + 1) * rep, i0:i1] = torch.softmax(sc, -1) @ v[g, lo:hi].float()
    return out


@torch.no_grad()
def square_emulation(qc, kc, vc, window: int = WINDOW):
    """Host emulation of the device call on the square layout (causal + window on local indices)."""
    return window_golden(qc, kc, vc, 0, window)


def probe_square(C: int, rows, *, a: float = 16.0, seed: int = 0):
    """Square-layout probe tensors. Probe r (chunk row i, square row 128 + i, absolute p = s + i) queries along its own
    orthonormal direction e_r with magnitude a; its designated keys carry score along e_r only and value only in its
    16 value dims D_r = [16 r, 16 r + 16): local key i (= p - 128, score 20, V +1, must be attended), i - 1 (= p - 129,
    score 28, V -1: a window leak dominates), 129 + i (= p + 1, score 36, V +3: a causal leak dominates). The other
    keys carry ~0 score, so a correct row outputs +1 in D_r. Returns (Q_cat, K_cat, V_cat, dims)."""
    T = TAIL + C
    g = torch.Generator().manual_seed(seed)
    E, _ = torch.linalg.qr(torch.randn(DQK, DQK, generator=g))
    q = 0.05 * torch.randn(NQ, T, DQK, generator=g)
    q[:, :TAIL] = 0.0
    k = 0.01 * torch.randn(NKV, T, DQK, generator=g)
    v = torch.zeros(NKV, T, DQK)
    v[..., :DV] = 0.05 * torch.randn(NKV, T, DV, generator=g)
    dims = []
    for r, i in enumerate(rows):
        e = E[:, r]
        row = TAIL + i
        q[:, row] = a * e
        d = slice(16 * r, 16 * r + 16)
        dims.append(d)

        def key(j, score, val):
            k[:, j] += (score / a) * e
            v[:, j, d] = val

        key(i, 20.0, 1.0)
        if i - 1 >= 0:
            key(i - 1, 28.0, -1.0)
        if row + 1 < T:
            key(row + 1, 36.0, 3.0)
    return q.bfloat16().float(), k.bfloat16().float(), v.bfloat16().float(), dims


def probe_verdict(out_rows, rows, dims):
    """out_rows [NQ, C, dv] (chunk rows): per probe the mean over heads / D_r dims and a verdict."""
    res = []
    for r, i in enumerate(rows):
        m = out_rows[:, i, dims[r]]
        mean, dev = float(m.mean()), float((m - 1.0).abs().max())
        if dev <= PROBE_TOL:
            verdict = "ok"
        elif abs(mean + 1.0) < 0.5:
            verdict = "WINDOW_LEAK"
        elif abs(mean - 3.0) < 0.75:
            verdict = "CAUSAL_LEAK"
        else:
            verdict = f"OFF({mean:.3f})"
        res.append((i, round(mean, 5), round(dev, 6), verdict))
    return res


# ----------------------------------------------------------------------------------------------------------------
# host self-checks
# ----------------------------------------------------------------------------------------------------------------
def test_g10_host_square_selfcheck():
    """The square layout reproduces window attention at absolute positions; the probes and their negative controls
    discriminate (on a host emulation of the device semantics)."""
    S, s, C = 256 + 64, 192, 128
    q, k, v = random_seq(S, seed=1)
    want = window_golden(q[:, s : s + C], k, v, s)
    qc, kc, vc = to_square(q[:, s : s + C], k, v, s, C)
    got = square_emulation(qc, kc, vc)[:, TAIL:]
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
    rows = probe_rows(C)
    qp, kp, vp, dims = probe_square(C, rows)
    ok = probe_verdict(square_emulation(qp, kp, vp)[:, TAIL:], rows, dims)
    assert all(x[3] == "ok" for x in ok), ok
    bad128 = probe_verdict(square_emulation(qp, kp, vp, window=128)[:, TAIL:], rows, dims)
    bad130 = probe_verdict(square_emulation(qp, kp, vp, window=130)[:, TAIL:], rows, dims)
    assert all(x[3] != "ok" for x in bad128), bad128
    assert all(x[3] == "WINDOW_LEAK" for x in bad130 if x[0] >= 1), bad130


# ----------------------------------------------------------------------------------------------------------------
# device
# ----------------------------------------------------------------------------------------------------------------
def rep_host(t: torch.Tensor, mesh_device, dtype=ttnn.int32):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))


class Bounds:
    """Persistent tensor-args slice bounds of one cache block ([blk, 0, 0, 0] / [blk + 1, 1, 64, 576])."""

    def __init__(self, mesh_device):
        self.mesh = mesh_device
        z = torch.zeros(4, dtype=torch.int32)
        self.start = gu.to_mesh(z, mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.end = gu.to_mesh(z, mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def set(self, blk: int):
        ttnn.copy_host_to_device_tensor(rep_host(torch.tensor([blk, 0, 0, 0], dtype=torch.int32), self.mesh), self.start)
        ttnn.copy_host_to_device_tensor(rep_host(torch.tensor([blk + 1, 1, BS, D_LAT], dtype=torch.int32), self.mesh), self.end)


def slice_block(cache, b: Bounds, n_blocks: int):
    return ttnn.slice(cache, b.start, b.end, slice_dim=0, num_devices=n_blocks, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def gather_tail(cache, b0: Bounds, b1: Bounds, n_blocks: int, concat_first: bool = True):
    """The design's tail: two block slices -> concat(dim=2) -> [1, 1, 128, 576] -> bf16."""
    t0 = slice_block(cache, b0, n_blocks)
    t1 = slice_block(cache, b1, n_blocks)
    if concat_first:
        cat = ttnn.concat([t0, t1], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gu.free([t0, t1])
        out = ttnn.typecast(cat, ttnn.bfloat16) if cat.dtype != ttnn.bfloat16 else cat
        if out is not cat:
            ttnn.deallocate(cat)
        return out
    a0 = ttnn.typecast(t0, ttnn.bfloat16)
    a1 = ttnn.typecast(t1, ttnn.bfloat16)
    gu.free([t0, t1])
    out = ttnn.concat([a0, a1], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gu.free([a0, a1])
    return out


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g10a_tail_gather(mesh_device):
    try:
        from models.demos.motif3.tt.ccl import log_fabric

        log_fabric(mesh_device, "G10")
    except Exception as e:  # pragma: no cover
        print(f"[G10] fabric report unavailable: {e}")
    failures = []
    t0 = time.time()
    g = torch.Generator().manual_seed(10)
    host = gu.host_roundtrip(torch.randn(N_BIG, 1, BS, D_LAT, generator=g), ttnn.bfloat8_b)  # bfp8-exact values
    cache = gu.to_mesh(host, mesh_device, ttnn.bfloat8_b)
    print(f"[G10] cache [{N_BIG},1,64,576] bfp8 built and uploaded in {time.time() - t0:.1f} s")
    blocks = [1, 2, 2063, 4127, 4128] + torch.randperm(N_BIG - 1, generator=g)[:3].add(1).tolist()
    if QUICK:
        blocks = [1, 4128]

    # ---- eager: one block per call, one program over all ids ----
    b0 = Bounds(mesh_device)
    ttnn.synchronize_device(mesh_device)
    n0 = mesh_device.num_program_cache_entries()
    eager_ok, detail = True, []
    for blk in blocks:
        b0.set(blk)
        try:
            out = slice_block(cache, b0, N_BIG)
            shape = tuple(int(x) for x in out.shape)
            per_chip = [ttnn.to_torch(t).float() for t in (ttnn.get_device_tensors(out)[i] for i in (0, 13, 31))]
            ttnn.deallocate(out)
        except Exception as e:
            REC.add(f"slice_eager/blk{blk}", status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
            failures.append(f"slice blk {blk}: {type(e).__name__}: {str(e)[:200]}")
            eager_ok = False
            continue
        want = host[blk : blk + 1]
        ok = shape == (1, 1, BS, D_LAT) and all(torch.equal(x, want) for x in per_chip)
        detail.append((blk, ok))
        eager_ok &= ok
        REC.add(f"slice_eager/blk{blk}", status="pass" if ok else "fail", out_shape=list(shape),
                neighbour_equal_prev=bool(torch.equal(per_chip[0], host[blk - 1 : blk])) if blk > 0 else None)
        if not ok:
            failures.append(f"slice blk {blk}: not bitwise equal to the host block (shape {shape})")
    ttnn.synchronize_device(mesh_device)
    n1 = mesh_device.num_program_cache_entries()
    REC.add("slice_eager/program_cache", status="pass" if n1 - n0 == 1 else "fail", programs_added=n1 - n0,
            blocks=blocks, all_bitwise=eager_ok)
    if n1 - n0 != 1:
        failures.append(f"slice: {n1 - n0} programs over {len(blocks)} block ids (expected 1)")

    # ---- traced: one slice captured, replayed with rewritten bounds ----
    b0.set(blocks[0])
    tid, tout = capture_trace(mesh_device, lambda: slice_block(cache, b0, N_BIG))
    try:
        tr_ok = True
        for blk in blocks[::-1]:
            b0.set(blk)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            ok = torch.equal(gu.read_dev(tout, 0), host[blk : blk + 1]) and torch.equal(gu.read_dev(tout, 31), host[blk : blk + 1])
            tr_ok &= ok
        REC.add("slice_traced/rewritten_bounds", status="pass" if tr_ok else "fail", blocks=blocks[::-1])
        if not tr_ok:
            failures.append("traced slice with rewritten bounds != host block")
    finally:
        ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(tout)

    # ---- the tail: 2 slices -> concat(dim 2) -> bf16, eager + traced + cost ----
    b1 = Bounds(mesh_device)
    pairs = [(1, 2), (4127, 4128), (2063, 17), (4128, 1)] + [tuple(torch.randperm(N_BIG - 1, generator=g)[:2].add(1).tolist())]
    for concat_first in (True, False):
        mode = "concat_bfp8_then_bf16" if concat_first else "bf16_then_concat"
        try:
            b0.set(pairs[0][0])
            b1.set(pairs[0][1])
            out = gather_tail(cache, b0, b1, N_BIG, concat_first)
            ttnn.deallocate(out)
        except Exception as e:
            REC.add(f"tail/{mode}", status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
            if concat_first:
                continue
            failures.append(f"tail gather {mode}: {type(e).__name__}: {str(e)[:200]}")
            continue
        ok_e = True
        for p0, p1 in pairs:
            b0.set(p0)
            b1.set(p1)
            out = gather_tail(cache, b0, b1, N_BIG, concat_first)
            want = torch.cat([host[p0], host[p1]], dim=1)[None]  # [1, 1, 128, 576]
            ok_e &= out.dtype == ttnn.bfloat16 and torch.equal(gu.read_dev(out, 0), want) and torch.equal(gu.read_dev(out, 20), want)
            ttnn.deallocate(out)
        eager_us = gu.time_eager(mesh_device, lambda: gather_tail(cache, b0, b1, N_BIG, concat_first), iters=20)
        tid, tout = capture_trace(mesh_device, lambda: gather_tail(cache, b0, b1, N_BIG, concat_first))
        ok_t = True
        try:
            for p0, p1 in pairs[::-1]:
                b0.set(p0)
                b1.set(p1)
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                want = torch.cat([host[p0], host[p1]], dim=1)[None]
                ok_t &= torch.equal(gu.read_dev(tout, 0), want) and torch.equal(gu.read_dev(tout, 31), want)
        finally:
            ttnn.release_trace(mesh_device, tid)
        ttnn.deallocate(tout)
        traced_us, raw = gu.time_traced(mesh_device, lambda: gather_tail(cache, b0, b1, N_BIG, concat_first),
                                        ops_per_trace=32, reps=9)
        REC.add(f"tail/{mode}", status="pass" if (ok_e and ok_t) else "fail", eager_bitwise=ok_e, traced_bitwise=ok_t,
                pairs=pairs, eager_us=eager_us, traced_us=traced_us, traced_raw_us=raw,
                note="per sp1 chunk once (all 39 SWA layers share the tail block ids, one page table per model) -- "
                     "but each SWA layer slices its own cache, so x39 per chunk")
        if not (ok_e and ok_t):
            failures.append(f"tail gather {mode}: eager {ok_e} traced {ok_t}")
    assert not failures, "G10a failures:\n" + "\n".join(failures)


def sdpa_pc(mesh_device, q=128, k=128):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=gu.grid_size(mesh_device), q_chunk_size=q, k_chunk_size=k, exp_approx_mode=False
    )


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g10b_square_window_sdpa(mesh_device):
    torch.set_num_threads(max(8, min(32, (os.cpu_count() or 8) // 2)))
    ckc = gu.compute_cfg("HiFi4", fp32_acc=False, approx=False)  # sdpa_prefill role: never fp32 acc with a window
    prog = sdpa_pc(mesh_device)
    failures = []

    def sdpa(q_tt, k_tt, v_tt, window=WINDOW):
        return ttnn.transformer.scaled_dot_product_attention(
            q_tt, k_tt, v_tt, is_causal=True, scale=1.0, sliding_window_size=window, program_config=prog,
            compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    timed_C = set()
    for C, s in SQUARE_CASES:
        case = f"C{C}/s{s}"
        S = s + C
        q, k, v = random_seq(S, seed=C * 7 + s)
        qc, kc, vc = to_square(q[:, s:], k, v, s, C)
        want = window_golden(q[:, s:], k, v, s)[..., :DV]
        try:
            tq, tk, tv = (gu.to_mesh(t[None], mesh_device, ttnn.bfloat16) for t in (qc, kc, vc))
            o = sdpa(tq, tk, tv)
            sq = gu.read_dev(o, 0)[0, :, TAIL:, :DV]
            same_rep = gu.replicas_identical(o, (0, 7, 31))[0] if C <= 1024 else None
            ttnn.deallocate(o)
            if C not in timed_C:
                us_sq = gu.time_eager(mesh_device, lambda: sdpa(tq, tk, tv), iters=5 if C >= 8192 else 10, warmup=1)
            gu.free([tq, tk, tv])
            # draft-1 single shot over the whole sequence, rows [s, s + C)
            fq, fk, fv = (gu.to_mesh(t[None], mesh_device, ttnn.bfloat16) for t in (q, k, v))
            o = sdpa(fq, fk, fv)
            ss = gu.read_dev(o, 0)[0, :, s:, :DV]
            ttnn.deallocate(o)
            gu.free([fq, fk, fv])
        except Exception as e:
            REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
            failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
            continue
        st_sq = gu.compare(want, sq)
        st_ss = gu.compare(want, ss)
        st_x = gu.compare(ss, sq)
        ok = st_sq["pcc"] >= PCC_PASS and st_ss["pcc"] >= PCC_PASS and st_sq["nonfinite"] == 0 and same_rep is not False
        rec = dict(pcc_square_vs_golden=st_sq["pcc"], pcc_single_shot_vs_golden=st_ss["pcc"], pcc_square_vs_single_shot=st_x["pcc"],
                   square_bitwise_eq_single_shot=bool(torch.equal(sq, ss)), max_abs=st_sq["max_abs"], replicas_identical=same_rep)
        if C not in timed_C:
            # the draft-1 SWA SDPA over a C-row chunk (positions 0..C-1) for the cost factor
            q0, k0, v0 = random_seq(C, seed=C)
            tq, tk, tv = (gu.to_mesh(t[None], mesh_device, ttnn.bfloat16) for t in (q0, k0, v0))
            win0 = WINDOW if C >= WINDOW else None
            us_sp0 = gu.time_eager(mesh_device, lambda: sdpa(tq, tk, tv, win0), iters=5 if C >= 8192 else 10, warmup=1)
            gu.free([tq, tk, tv])
            rec.update(square_eager_us=us_sq, draft1_chunk_eager_us=us_sp0, cost_factor=us_sq / us_sp0,
                       expected_factor=1 + TAIL / C)
            timed_C.add(C)
        REC.add(case, status="pass" if ok else "fail", **rec)
        if not ok:
            failures.append(f"{case}: square {gu.fmt(st_sq)} single-shot pcc {st_ss['pcc']:.6f}")

        # ---- probes (square layout; position-independent, so one probe run per C at the first s) ----
        if s == SQUARE_CASES[[c for c, _ in SQUARE_CASES].index(C)][1]:
            rows = probe_rows(C)
            qp, kp, vp, dims = probe_square(C, rows, seed=C)
            tq, tk, tv = (gu.to_mesh(t[None], mesh_device, ttnn.bfloat16) for t in (qp, kp, vp))
            windows = (WINDOW, 128, 130) if C == 128 else (WINDOW,)
            for w in windows:
                o = sdpa(tq, tk, tv, w)
                got = gu.read_dev(o, 0)[0, :, TAIL:, :DV]
                ttnn.deallocate(o)
                verdicts = probe_verdict(got, rows, dims)
                good = all(x[3] == "ok" for x in verdicts)
                if w == WINDOW:
                    REC.add(f"probe/C{C}/window{w}", status="pass" if good else "fail", rows=rows, verdicts=verdicts)
                    if not good:
                        failures.append(f"probe C{C}: {verdicts}")
                else:
                    detected = all(x[3] != "ok" for x in verdicts if (w == 128 or x[0] >= 1))
                    REC.add(f"probe/C{C}/negative_window{w}", status="detected" if detected else "NOT_detected", verdicts=verdicts)
                    if not detected:
                        failures.append(f"negative control window {w} not detected: {verdicts}")
            gu.free([tq, tk, tv])
    assert not failures, "G10b failures:\n" + "\n".join(failures)
