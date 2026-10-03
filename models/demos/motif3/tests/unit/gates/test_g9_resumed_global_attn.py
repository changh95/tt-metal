# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G9 — resumed ("sp1") global attention over the paged latent cache (FEATURES_DESIGN §4 G9, D1, D2, §3.2.2).

The sp1 global layer runs absorbed attention straight on the paged latent cache, with no C++ change:

    O = ttnn.transformer.chunked_scaled_dot_product_attention(
            Q [1, 10, C, 576] bf16 (q_lat 512 | roped q_pe 64, softmax scale folded into Q),
            K = V = cache [N, 1, 64, 576] bfloat8_b (the layer's latent cache: unit-RMS n 512 | roped k_pe 64),
            page_table [1, 640] int32 (real ids for blocks [0, cdiv(s + C, 64)), then the null block 0, never -1),
            chunk_start_idx_tensor = [s] int32 (device tensor: one program per (C, q, k) for every start),
            scale=1.0, SDPAProgramConfig(12x10, q, k, exp_approx False), sdpa_prefill role)  -> [1, 10, C, 576]
    O_lat = O[..., :512]   (columns 512..575 = the attention-weighted k_pe, discarded by the model)

Query row i sits at absolute position s + i and attends keys [0, s + i] (the chunk's own keys come from the cache, so
the caller fills the chunk first). Golden: fp32 causal latent attention on the host-dequantized bfp8 cache with V = K
(columns [:512] are the model's V = K[..., :512]).

Cases (design §4): C in {128, 512, 2048, 8192} x s in {128, 2048, 8192, 24448} (s + C <= 32768) x (q, k) in {(64, 64),
(128, 128)} plus (32, 64) at C = 128, plus s = 8256 (= 64 mod 128, only an A = 64 config may serve it) for q, k <= 64,
x compute role (``MOTIF3_G9_ROLES``, default both): ``fp32acc`` (HiFi4 + fp32 dest acc, the window-free
``sdpa_prefill_fp32`` role, legacy kernel; the gating role) and ``bf16acc`` (the ``sdpa_prefill`` role, streaming
kernel; measured at PCC 0.998 over the 576-wide latent, below the bar, kept for the cost comparison). Per case: PCC on
columns [:512] (pass >= 0.999, target >= 0.9995), worst row
(>= 0.998), columns [512:576], non-finite count, replicas, eager us (and traced us for C <= 512). Per (C, q, k): the
program cache grows by exactly 1 over all starts, and a trace captured at s = 128 replays bitwise equal to eager at
every other start (start and page table rewritten in place). L1 overflow of a config is recorded ``unsupported_L1``.

Comparisons (test_g9_reference_ops): the legacy scalar-start MLA op ``chunked_flash_mla_prefill`` (one program per
start; the kill criterion is "K = V op > 1.5x the MLA scalar op at C = 2048, s = 8192" -> G9b C++ patch), the draft-1
expanded global SDPA (sp0) at the same buckets (cost model baseline), a bf16-cache L1/accuracy probe, and two negative
controls (start not a multiple of q_chunk, page table shifted by one block) that must be *detected* as wrong.

Run (device)::

    scripts/devrun.sh -t 3000 -n g9 -- python -m pytest models/demos/motif3/tests/unit/gates/test_g9_resumed_global_attn.py \
        -s -p no:cacheprovider
    # host self-checks of the golden / tables (devices hidden; keep the root conftest for the indirect fixtures)
    scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q \
        models/demos/motif3/tests/unit/gates/test_g9_resumed_global_attn.py -k host
"""

from __future__ import annotations

import math
import os
import time

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

QUICK = gu.env_flag("MOTIF3_GATES_QUICK")  # smoke run: a few cases, results in G9_quick.jsonl
REC = gu.Recorder("G9_quick" if QUICK else "G9")
# serving device params: FABRIC_2D_TORUS_XY, 256 MiB trace region, L1_SMALL 32 KiB (CCL semaphores; design §4 setup)
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

NQ, D, DV, BS = 10, 576, 512, 64
W_SDPA = 640  # round_up_8(cdiv(32768 + 8192, 64)): fixed sp1 SDPA page-table width (design §3.1 item 4)
MAX_POS = 32768
N_BLOCKS = 520  # 1 null + 510 blocks cover s + C <= 32640, plus spare
CS = (128, 2048) if QUICK else (128, 512, 2048, 8192)
if os.environ.get("MOTIF3_G9_CS"):  # e.g. "256,1024,4096": the remaining prefill buckets for the per-bucket (q, k) table
    CS = tuple(int(c) for c in os.environ["MOTIF3_G9_CS"].split(",") if c.strip())
SS = (128, 8192) if QUICK else (128, 2048, 8192, 24448)
SS_A64 = (8256,)  # a start = 64 mod 128: only an A = 64 config (q, k <= 64) may serve it (validates A = lcm(64, q, k))
QK_MAIN = ((64, 64), (128, 128))
QK_SMALL = ((32, 64),)  # C = 128 only (occupancy for short suffixes)
PCC_PASS, PCC_TARGET, ROW_PASS = 0.999, 0.9995, 0.998
Q_SCALE = gd.SCALE_GLOBAL  # the model folds the softmax scale into q (scale=1.0 in the op)


def cases():
    out = []
    for C in CS:
        for s in SS:
            if s + C > MAX_POS:
                continue
            out.append((C, s))
    return out


def qk_configs(C: int):
    return list(QK_MAIN) + (list(QK_SMALL) if C == 128 else [])


def align_of(q: int, k: int, bs: int = BS) -> int:
    return math.lcm(bs, q, k)


# ----------------------------------------------------------------------------------------------------------------
# host data (pure torch)
# ----------------------------------------------------------------------------------------------------------------
class Doc:
    """One virtual sequence of MAX_POS - 128 latent rows stored in a shuffled paged cache.

    ``kv_virt [P, 576]`` holds the dequantized values the device cache stores for virtual position p; virtual block v
    lives in physical block ``phys[v]`` (ids 1..N-1, block 0 = the null block)."""

    def __init__(self, seed: int = 9, n_blocks: int = N_BLOCKS, quantize=None):
        g = torch.Generator().manual_seed(seed)
        nv = min((MAX_POS - 128) // BS, n_blocks - 1)  # 510 virtual blocks at the real size
        self.phys = (torch.randperm(n_blocks - 1, generator=g)[:nv] + 1).to(torch.int32)
        paged = torch.randn(n_blocks, 1, BS, D, generator=g)  # unit-RMS-like latent | k_pe, every block distinct
        self.paged = quantize(paged) if quantize is not None else paged.bfloat16().float()
        self.kv_virt = self.paged[self.phys.long(), 0].reshape(nv * BS, D)
        self.n_blocks = n_blocks

    def sdpa_table(self, end: int, width: int = W_SDPA, shift: int = 0) -> torch.Tensor:
        """``[1, width]`` int32: real ids for virtual blocks [0, cdiv(end, 64)), then 0 (the SDPA reader has no skip)."""
        n = -(-end // BS)
        pt = torch.zeros(1, width, dtype=torch.int32)
        pt[0, :n] = self.phys[shift : shift + n]
        return pt


def make_q(C: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, NQ, C, D, generator=g) * Q_SCALE).bfloat16().float()


def sample_rows(C: int, seed: int) -> torch.Tensor:
    """All rows for C <= 2048; at C = 8192 both ends, every 256-row seam (q-chunk boundaries of every config) and 1792
    random rows (~2.3K rows; the full 8K x 32K golden is ~6 TFLOP on the host)."""
    if C <= 2048:
        return torch.arange(C)
    g = torch.Generator().manual_seed(seed)
    rows = set(range(0, 128)) | set(range(C - 128, C))
    for k in range(1, C // 256):
        rows |= {256 * k - 1, 256 * k}
    rows |= set(torch.randperm(C, generator=g)[:1792].tolist())
    return torch.tensor(sorted(rows))


@torch.no_grad()
def latent_golden(q: torch.Tensor, kv_virt: torch.Tensor, s: int, rows: torch.Tensor, block: int = 128) -> torch.Tensor:
    """fp32 causal latent attention with V = K: q ``[NQ, C, 576]``, rows (local indices) -> ``[NQ, len(rows), 576]``.
    Row i (absolute position s + i) attends keys [0, s + i]."""
    out = torch.empty(q.shape[0], len(rows), D, dtype=torch.float32)
    for b0 in range(0, len(rows), block):
        r = rows[b0 : b0 + block]
        p = s + r
        kmax = int(p.max()) + 1
        K = kv_virt[:kmax].float()
        sc = q[:, r].float() @ K.T  # [NQ, nr, kmax]
        mask = torch.arange(kmax)[None, :] > p[:, None]
        sc.masked_fill_(mask[None], float("-inf"))
        out[:, b0 : b0 + len(r)] = torch.softmax(sc, dim=-1) @ K
    return out


def row_pccs(want: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    """Per query row PCC over (heads x columns): want/got ``[NQ, R, d]`` -> ``[R]``."""
    w = want.double().permute(1, 0, 2).reshape(want.shape[1], -1)
    g = got.double().permute(1, 0, 2).reshape(got.shape[1], -1)
    w = w - w.mean(dim=1, keepdim=True)
    g = g - g.mean(dim=1, keepdim=True)
    return (w * g).sum(1) / (w.norm(dim=1) * g.norm(dim=1)).clamp_min(1e-30)


# ----------------------------------------------------------------------------------------------------------------
# device helpers
# ----------------------------------------------------------------------------------------------------------------
def pc(mesh_device, q: int, k: int):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=gu.grid_size(mesh_device), q_chunk_size=q, k_chunk_size=k, exp_approx_mode=False
    )


def ckc_prefill():
    return gu.compute_cfg("HiFi4", fp32_acc=False, approx=False)  # the sdpa_prefill role (G2)


# Compute roles swept by the main test: "fp32acc" = the window-free "sdpa_prefill_fp32" role (HiFi4, fp32 dest acc:
# legacy non-streaming kernel) -- the gating role; "bf16acc" = "sdpa_prefill" (HiFi4, bf16 dest: streaming kernel).
# The first full run (2026-10-02 17:01) showed bf16 dest accumulation over the 576-wide latent at PCC 0.998 (below the
# bar at every case) and fp32 dest at 0.99993+, so the gate is decided on fp32acc. MOTIF3_G9_ROLES picks the roles.
ROLES = {"fp32acc": dict(fidelity="HiFi4", fp32_acc=True, approx=False), "bf16acc": dict(fidelity="HiFi4", fp32_acc=False, approx=False)}
GATING_ROLE = "fp32acc"


def selected_roles():
    names = [r.strip() for r in os.environ.get("MOTIF3_G9_ROLES", "fp32acc,bf16acc").split(",") if r.strip()]
    bad = [r for r in names if r not in ROLES]
    if bad:
        raise ValueError(f"MOTIF3_G9_ROLES: unknown role(s) {bad}; known {list(ROLES)}")
    return names


def rep_host(t: torch.Tensor, mesh_device, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    """Host-side replicated mesh tensor for ``ttnn.copy_host_to_device_tensor`` into a persistent input."""
    return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))


class Inputs:
    """Persistent device inputs of the sp1 call (rewritten in place, as the generator will do)."""

    def __init__(self, mesh_device, doc: Doc):
        self.mesh = mesh_device
        self.pt = gu.to_mesh(torch.zeros(1, W_SDPA, dtype=torch.int32), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.start = gu.to_mesh(torch.zeros(1, dtype=torch.int32), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.doc = doc

    def set(self, s: int, C: int, *, shift: int = 0):
        ttnn.copy_host_to_device_tensor(rep_host(self.doc.sdpa_table(s + C, shift=shift), self.mesh, ttnn.int32), self.pt)
        ttnn.copy_host_to_device_tensor(rep_host(torch.tensor([s], dtype=torch.int32), self.mesh, ttnn.int32), self.start)


def is_l1_error(msg: str) -> bool:
    return "beyond max L1 size" in msg or "clash with L1 buffers" in msg or "Statically allocated circular buffers" in msg


def replicas_equal(t, idxs) -> bool:
    shards = ttnn.get_device_tensors(t)
    ref = ttnn.to_torch(shards[idxs[0]])
    return all(torch.equal(ttnn.to_torch(shards[i]), ref) for i in idxs[1:])


# ----------------------------------------------------------------------------------------------------------------
# host self-checks (no device): the golden and the tables the device cases rely on
# ----------------------------------------------------------------------------------------------------------------
def test_g9_host_golden_selfcheck():
    """The latent golden equals a dense masked reference; the SDPA table never holds -1 and maps virtual blocks."""
    torch.manual_seed(0)
    doc = Doc(seed=1, n_blocks=40)
    s, C = 128, 64
    q = torch.randn(NQ, C, D) * 0.1
    rows = torch.arange(C)
    got = latent_golden(q, doc.kv_virt[: s + C], s, rows, block=16)
    K = doc.kv_virt[: s + C]
    sc = q @ K.T
    mask = torch.arange(s + C)[None, :] > (s + torch.arange(C))[:, None]
    ref = torch.softmax(sc.masked_fill(mask[None], float("-inf")), -1) @ K
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()
    pt = doc.sdpa_table(s + C, width=16)
    assert int(pt.min()) >= 0 and int((pt[0, : 3] == doc.phys[:3]).all()) == 1 and int(pt[0, 3:].abs().sum()) == 0
    # the paged layout: virtual position p lives in physical block phys[p // 64], row p % 64
    p = 77
    assert torch.equal(doc.kv_virt[p], doc.paged[int(doc.phys[p // BS]), 0, p % BS])
    assert sample_rows(8192, 0).max() == 8191 and sample_rows(512, 0).numel() == 512
    assert align_of(64, 64) == 64 and align_of(128, 128) == 128 and align_of(32, 64) == 64


# ----------------------------------------------------------------------------------------------------------------
# device: accuracy, program cache, trace replay, cost
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g9_resumed_global_attention(mesh_device):
    torch.set_num_threads(max(8, min(32, (os.cpu_count() or 8) // 2)))
    try:
        from models.demos.motif3.tt.ccl import log_fabric

        log_fabric(mesh_device, "G9")
    except Exception as e:  # pragma: no cover - report only
        print(f"[G9] fabric report unavailable: {e}")
    t0 = time.time()
    doc = Doc(seed=9, quantize=lambda t: gu.host_roundtrip(t, ttnn.bfloat8_b))
    cache = gu.to_mesh(doc.paged, mesh_device, ttnn.bfloat8_b)
    inp = Inputs(mesh_device, doc)
    ckc = ckc_prefill()
    print(f"[G9] cache [{N_BLOCKS},1,64,576] bfp8 uploaded in {time.time() - t0:.1f} s")
    failures = []
    golden_cache = {}

    for C in CS:
        q_host = make_q(C, seed=C)
        q_dev = gu.to_mesh(q_host, mesh_device, ttnn.bfloat16)
        rows = sample_rows(C, seed=C)
        for role, (qc, kc) in [(r, qk) for r in selected_roles() for qk in qk_configs(C)]:
            ckc = gu.compute_cfg(**ROLES[role])
            gating = role == GATING_ROLE
            # bf16acc keeps the case names of the first run (C/qk/s); other roles insert the role name
            cfg_name = f"C{C}/q{qc}_k{kc}" if role == "bf16acc" else f"C{C}/q{qc}_k{kc}/{role}"
            extra = SS_A64 if (align_of(qc, kc) == 64 and not QUICK) else ()
            s_list = sorted(s for s in SS + extra if s + C <= MAX_POS)
            prog = pc(mesh_device, qc, kc)

            def fn():
                return ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_dev,
                    cache,
                    cache,
                    inp.pt,
                    chunk_start_idx_tensor=inp.start,
                    scale=1.0,
                    program_config=prog,
                    compute_kernel_config=ckc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            ttnn.synchronize_device(mesh_device)
            n0 = mesh_device.num_program_cache_entries()
            eager_out = {}
            config_ok = True
            for s in s_list:
                case = f"{cfg_name}/s{s}"
                inp.set(s, C)
                try:
                    out = fn()
                    full = gu.read_dev(out, 0)[0]  # [NQ, C, 576]
                    same = replicas_equal(out, (0, 13, 31) if C >= 8192 else (0, 7, 13, 20, 31)) if s == s_list[0] else None
                    ttnn.deallocate(out)
                    iters = 3 if C >= 8192 else (5 if C >= 2048 else 10)
                    eager_us = gu.time_eager(mesh_device, fn, iters=iters, warmup=1)
                except Exception as e:
                    msg = str(e)
                    l1 = is_l1_error(msg)
                    REC.add(case, status="unsupported_L1" if l1 else "error", error=f"{type(e).__name__}: {msg[:400]}",
                            align=align_of(qc, kc))
                    if not l1 and gating:
                        failures.append(f"{case}: {type(e).__name__}: {msg[:200]}")
                    config_ok = False
                    break
                eager_out[s] = full
                key = (C, s)
                if key not in golden_cache:
                    tg = time.time()
                    golden_cache[key] = latent_golden(q_host[0], doc.kv_virt, s, rows)
                    print(f"[G9] golden C={C} s={s} rows={len(rows)}: {time.time() - tg:.1f} s")
                want = golden_cache[key]
                got = full[:, rows]
                st_lat = gu.compare(want[..., :DV], got[..., :DV])
                st_pe = gu.compare(want[..., DV:], got[..., DV:])
                rp = row_pccs(want[..., :DV], got[..., :DV])
                worst = float(rp.min())
                worst_row = int(rows[int(rp.argmin())])
                ok = (st_lat["pcc"] >= PCC_PASS and worst >= ROW_PASS and st_lat["nonfinite"] == 0
                      and int((~torch.isfinite(full)).sum()) == 0 and (same is None or same))
                # attention FLOPs (K = V = latent: QK 576 + PV 576) and the causal K/V DRAM stream (R10: no sharing
                # between (head, q-chunk) work units in causal mode; 2 x 612 B per key in bfp8)
                flops = 2.0 * NQ * C * (s + C / 2) * (D + D)
                dram = NQ * (C / qc) * (s + C / 2) * 2 * 612
                REC.add(
                    case,
                    status="pass" if ok else "fail",
                    pcc=st_lat["pcc"],
                    target_met=st_lat["pcc"] >= PCC_TARGET,
                    worst_row_pcc=worst,
                    worst_row=worst_row,
                    pcc_kpe_cols=st_pe["pcc"],
                    max_abs=st_lat["max_abs"],
                    nonfinite=int((~torch.isfinite(full)).sum()),
                    replicas_identical=same,
                    rows_checked=len(rows),
                    eager_us=eager_us,
                    tflops=flops / (eager_us * 1e-6) / 1e12,
                    dram_gb=dram / 1e9,
                    dram_GBps=dram / (eager_us * 1e-6) / 1e9,
                    align=align_of(qc, kc),
                    role=role,
                    gating=gating,
                )
                if not ok and gating:
                    failures.append(f"{case}: {gu.fmt(st_lat)} worst_row={worst:.5f}@{worst_row} replicas={same}")
            if not config_ok:
                continue
            ttnn.synchronize_device(mesh_device)
            n1 = mesh_device.num_program_cache_entries()

            # ---- trace: capture at s = 128, replay at every start (inputs rewritten in place) ----
            trace_ok, trace_detail = True, []
            inp.set(s_list[0], C)
            tid, tout = None, None
            try:
                tid, tout = capture_trace(mesh_device, lambda: fn())
                for s in s_list:
                    inp.set(s, C)
                    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                    rep = gu.read_dev(tout, 0)[0]
                    eq = torch.equal(rep, eager_out[s])
                    trace_detail.append((s, eq))
                    trace_ok &= eq
                traced_us = None
                ttnn.release_trace(mesh_device, tid)
                tid = None
                ttnn.deallocate(tout)
                tout = None
                if C <= 512:  # eager is dispatch-bound here: the traced per-call time is the device cost
                    inp.set(s_list[-1], C)
                    traced_us, raw = gu.time_traced(mesh_device, fn, ops_per_trace=16, reps=7, compile_first=False)
            except Exception as e:
                trace_ok = False
                trace_detail.append(f"{type(e).__name__}: {str(e)[:300]}")
                traced_us = None
            finally:
                if tid is not None:
                    ttnn.release_trace(mesh_device, tid)
                if tout is not None:
                    gu.free(tout)
            ttnn.synchronize_device(mesh_device)
            n2 = mesh_device.num_program_cache_entries()
            progs_ok = (n1 - n0) == 1 and n2 == n1
            REC.add(
                f"{cfg_name}/program_cache_and_trace",
                status="pass" if (progs_ok and trace_ok) else "fail",
                programs_added_over_starts=n1 - n0,
                programs_added_by_trace=n2 - n1,
                starts=s_list,
                trace_replay_bitwise_eq_eager=trace_detail,
                traced_us_last_start=traced_us,
                align=align_of(qc, kc),
                role=role,
            )
            if not progs_ok:
                failures.append(f"{cfg_name}: program cache +{n1 - n0} over {len(s_list)} starts (+{n2 - n1} by trace)")
            if not trace_ok:
                failures.append(f"{cfg_name}: trace replay != eager: {trace_detail}")
            eager_out.clear()
        ttnn.deallocate(q_dev)
        for key in [k for k in golden_cache if k[0] == C]:
            del golden_cache[key]

    assert not failures, "G9 failures:\n" + "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g9_reference_ops(mesh_device):
    """MLA scalar-start op (kill criterion), draft-1 expanded global SDPA (sp0 baseline), bf16 cache, negative controls."""
    torch.set_num_threads(max(8, min(32, (os.cpu_count() or 8) // 2)))
    doc = Doc(seed=9, quantize=lambda t: gu.host_roundtrip(t, ttnn.bfloat8_b))
    cache = gu.to_mesh(doc.paged, mesh_device, ttnn.bfloat8_b)
    inp = Inputs(mesh_device, doc)
    ckc = ckc_prefill()
    failures = []

    # ---- (1) K = V flexible op vs the MLA scalar-start op: same start, same Q, same cache ----
    mla_cases = ((2048, 8192),) if QUICK else ((2048, 8192), (8192, 8192), (8192, 24448), (2048, 24448), (128, 24448), (512, 8192))
    for C, s in mla_cases:
        q_host = make_q(C, seed=C)
        q_dev = gu.to_mesh(q_host, mesh_device, ttnn.bfloat16)
        for qc, kc in QK_MAIN:
            prog = pc(mesh_device, qc, kc)
            case = f"mla_vs_kv/C{C}/s{s}/q{qc}_k{kc}"
            inp.set(s, C)
            pt_scalar = gu.to_mesh(doc.sdpa_table(s + C), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

            def f_kv():
                return ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_dev, cache, cache, inp.pt, chunk_start_idx_tensor=inp.start, scale=1.0, program_config=prog,
                    compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            def f_mla():
                return ttnn.transformer.chunked_flash_mla_prefill(
                    q_dev, cache, DV, pt_scalar, s, scale=1.0, program_config=prog, compute_kernel_config=ckc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)

            try:
                o_kv = f_kv()
                a = gu.read_dev(o_kv, 0)[0, :, :, :DV]
                ttnn.deallocate(o_kv)
                o_mla = f_mla()
                b = gu.read_dev(o_mla, 0)[0, :, :, :DV]
                ttnn.deallocate(o_mla)
                iters = 3 if C >= 8192 else 6
                us_kv = gu.time_eager(mesh_device, f_kv, iters=iters, warmup=1)
                us_mla = gu.time_eager(mesh_device, f_mla, iters=iters, warmup=1)
            except Exception as e:
                msg = str(e)
                REC.add(case, status="unsupported_L1" if is_l1_error(msg) else "error", error=f"{type(e).__name__}: {msg[:400]}")
                ttnn.deallocate(pt_scalar)
                continue
            ratio = us_kv / us_mla
            st = gu.compare(b, a)
            kill = C == 2048 and s == 8192 and ratio > 1.5
            REC.add(
                case,
                status="kill_G9b" if kill else "measured",
                kv_eager_us=us_kv,
                mla_scalar_eager_us=us_mla,
                ratio_kv_over_mla=ratio,
                lat_cols_bitwise_equal=bool(torch.equal(a, b)),
                pcc_kv_vs_mla=st["pcc"],
                max_abs_kv_vs_mla=st["max_abs"],
            )
            if kill:
                failures.append(f"{case}: K=V op {us_kv:.0f} us > 1.5 x MLA scalar {us_mla:.0f} us -> G9b")
            ttnn.deallocate(pt_scalar)
        ttnn.deallocate(q_dev)

    # ---- (2) draft-1 expanded global SDPA (sp0, d = 192, q256/k256; q128/k128 at S = 128) at the same buckets ----
    for C in CS:
        g = torch.Generator().manual_seed(100 + C)
        qx = gu.to_mesh(torch.randn(1, NQ, C, 192, generator=g) * 0.1447, mesh_device, ttnn.bfloat16)
        kx = gu.to_mesh(torch.randn(1, 2, C, 192, generator=g), mesh_device, ttnn.bfloat16)
        vx = gu.to_mesh(torch.randn(1, 2, C, 192, generator=g), mesh_device, ttnn.bfloat16)
        qc = kc = 128 if C == 128 else 256
        prog = pc(mesh_device, qc, kc)

        def f_sp0():
            return ttnn.transformer.scaled_dot_product_attention(
                qx, kx, vx, is_causal=True, scale=1.0, program_config=prog, compute_kernel_config=ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)

        try:
            us = gu.time_eager(mesh_device, f_sp0, iters=3 if C >= 8192 else 8, warmup=1)
            tr = gu.time_traced(mesh_device, f_sp0, ops_per_trace=16, reps=7)[0] if C <= 512 else None
            REC.add(f"sp0_expanded_global/C{C}", status="measured", eager_us=us, traced_us=tr, q_chunk=qc, k_chunk=kc)
        except Exception as e:
            REC.add(f"sp0_expanded_global/C{C}", status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
        gu.free([qx, kx, vx])

    # ---- (3) bf16 latent cache (MOTIF3_KV_CACHE_DTYPE=bf16): L1 fit and accuracy at C = 2048, s = 8192 ----
    doc16 = Doc(seed=9)
    cache16 = gu.to_mesh(doc16.paged, mesh_device, ttnn.bfloat16)
    inp16 = Inputs(mesh_device, doc16)
    C, s = 2048, 8192
    q_host = make_q(C, seed=C)
    q_dev = gu.to_mesh(q_host, mesh_device, ttnn.bfloat16)
    rows = sample_rows(C, seed=C)
    want = latent_golden(q_host[0], doc16.kv_virt, s, rows)
    for qc, kc in QK_MAIN:
        case = f"bf16_cache/C{C}/s{s}/q{qc}_k{kc}"
        inp16.set(s, C)
        prog = pc(mesh_device, qc, kc)

        def f16():
            return ttnn.transformer.chunked_scaled_dot_product_attention(
                q_dev, cache16, cache16, inp16.pt, chunk_start_idx_tensor=inp16.start, scale=1.0, program_config=prog,
                compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        try:
            o = f16()
            got = gu.read_dev(o, 0)[0][:, rows]
            ttnn.deallocate(o)
            us = gu.time_eager(mesh_device, f16, iters=5, warmup=1)
            st = gu.compare(want[..., :DV], got[..., :DV])
            REC.add(case, status="measured", pcc=st["pcc"], worst_row_pcc=float(row_pccs(want[..., :DV], got[..., :DV]).min()),
                    eager_us=us)
        except Exception as e:
            msg = str(e)
            REC.add(case, status="unsupported_L1" if is_l1_error(msg) else "error", error=f"{type(e).__name__}: {msg[:400]}")
    gu.free([cache16, q_dev])

    # ---- (3b) fp32 dest accumulation (the window-free "sdpa_prefill_fp32" role, legacy non-streaming kernel): an
    #      accuracy remedy if bf16 QK scores limit long prefixes; L1 fit and cost ----
    ckc32 = gu.compute_cfg("HiFi4", fp32_acc=True, approx=False)
    for C, s in (((2048, 8192),) if QUICK else ((2048, 8192), (8192, 24448))):
        q_host = make_q(C, seed=C)
        q_dev = gu.to_mesh(q_host, mesh_device, ttnn.bfloat16)
        rows = sample_rows(C, seed=C)
        want = latent_golden(q_host[0], doc.kv_virt, s, rows)
        for qc, kc in QK_MAIN:
            case = f"fp32_acc/C{C}/s{s}/q{qc}_k{kc}"
            inp.set(s, C)
            prog = pc(mesh_device, qc, kc)

            def f32():
                return ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_dev, cache, cache, inp.pt, chunk_start_idx_tensor=inp.start, scale=1.0, program_config=prog,
                    compute_kernel_config=ckc32, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            try:
                o = f32()
                got = gu.read_dev(o, 0)[0][:, rows]
                ttnn.deallocate(o)
                us = gu.time_eager(mesh_device, f32, iters=3 if C >= 8192 else 5, warmup=1)
                st = gu.compare(want[..., :DV], got[..., :DV])
                REC.add(case, status="measured", pcc=st["pcc"], eager_us=us,
                        worst_row_pcc=float(row_pccs(want[..., :DV], got[..., :DV]).min()))
            except Exception as e:
                msg = str(e)
                REC.add(case, status="unsupported_L1" if is_l1_error(msg) else "error", error=f"{type(e).__name__}: {msg[:400]}")
        ttnn.deallocate(q_dev)

    # ---- (4) negative controls at C = 128, q = k = 64: both must be detected as wrong ----
    C = 128
    q_host = make_q(C, seed=C)
    q_dev = gu.to_mesh(q_host, mesh_device, ttnn.bfloat16)
    rows = torch.arange(C)
    prog = pc(mesh_device, 64, 64)

    def f_neg():
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            q_dev, cache, cache, inp.pt, chunk_start_idx_tensor=inp.start, scale=1.0, program_config=prog,
            compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # start 96 with q_chunk 64: the kernels compute start // q_chunk = 1 chunk, i.e. they silently answer as if the
    # chunk started at 64 (one third of row 0's keys missing); a shifted page table reads the wrong blocks
    for name, s, shift in (("start_not_multiple_of_q_chunk", 96, 0), ("page_table_shifted_one_block", 2048, 1)):
        inp.set(s, C, shift=shift)
        try:
            o = f_neg()
            got = gu.read_dev(o, 0)[0]
            ttnn.deallocate(o)
            want = latent_golden(q_host[0], doc.kv_virt, s, rows)
            st = gu.compare(want[..., :DV], got[:, :, :DV])
            rec = dict(pcc=st["pcc"], start=s)
            if shift == 0:
                floor_s = (s // 64) * 64
                st_f = gu.compare(latent_golden(q_host[0], doc.kv_virt, floor_s, rows)[..., :DV], got[:, :, :DV])
                rec.update(pcc_vs_floored_start=st_f["pcc"], floored_start=floor_s)
            detected = st["pcc"] < PCC_PASS
            REC.add(f"negative_control/{name}", status="detected" if detected else "NOT_detected",
                    note="the flexible start has no device-side alignment check (design D1, R6)", **rec)
            if not detected:
                failures.append(f"negative control {name} not detected: pcc {st['pcc']:.6f}")
        except Exception as e:
            REC.add(f"negative_control/{name}", status="raised", error=f"{type(e).__name__}: {str(e)[:300]}")

    assert not failures, "G9 reference-op failures:\n" + "\n".join(failures)
