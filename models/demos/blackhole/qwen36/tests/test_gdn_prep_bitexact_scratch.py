# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: bit-exact regression + traced timing of the fused GDN chunk op (ChunkGdnPrep + ChunkGdnScan) at the
Qwen3.8-27B TP=4 prefill shape, for the round-4 item 12 step 1 kernel change (chunk_gdn_prep.cpp invert_block:
merged block-diagonal Horner). The prep's T_inv is consumed by the scan, so (o, final_state) cover it.

Modes, selected by ``QWEN36_PREP_REF=save|check`` (``QWEN36_PREP_REF_DIR`` = output dir; default ``check`` so a run
without the variable can never overwrite a saved reference; ``save`` refuses to overwrite unless
``QWEN36_PREP_REF_OVERWRITE=1``):
  * ``save``  -- run on the REFERENCE kernel (the original chunk_gdn_prep.cpp), dump o/final_state per case.
  * ``check`` -- run on the changed tree (op cache entries cleared), assert torch.equal on the raw bits and also
                 report the strict bitwise (int16/int32 view) mismatch count, so a signed-zero-only difference is
                 visible separately from a value difference.
Both modes print ``PREP_BENCH_RESULT`` with the per-call time of 50 traced replays (prep + scan + adapter glue).

Per device: flat q/k [1,T,4*128] bf16, v [1,T,12*128] bf16, beta/g [1,T,12] bf16, chunk 32, 12 value heads.
Cases: 3 seeds at T=2048 (the served chunk) with two g distributions, plus one T=4096.

  QWEN36_PREP_REF=save  pytest models/demos/blackhole/qwen36/tests/test_gdn_prep_bitexact_scratch.py -s
  QWEN36_PREP_REF=check pytest models/demos/blackhole/qwen36/tests/test_gdn_prep_bitexact_scratch.py -s
"""

import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import (
    build_fused_const_tiles,
    chunk_gated_delta_rule_fused_adapter,
)

MODE = os.environ.get("QWEN36_PREP_REF", "check")
assert MODE in ("save", "check"), f"QWEN36_PREP_REF must be save|check, got {MODE!r}"
OVERWRITE = os.environ.get("QWEN36_PREP_REF_OVERWRITE", "0") == "1"
REF_DIR = os.environ.get("QWEN36_PREP_REF_DIR", "/home/eslim/experiments/qwen36/logs/r4_prep_ref")
REPLAYS = int(os.environ.get("QWEN36_PREP_REPLAYS", "50"))
BATCHES = int(os.environ.get("QWEN36_PREP_BATCHES", "5"))

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2, "fabric_config": ttnn.FabricConfig.FABRIC_1D}]
NK, DK, NV, DV = 4, 128, 12, 128
# (name, T, seed, g distribution)
CASES = [
    ("t2048_s0_mild", 2048, 0, "mild"),
    ("t2048_s1_mild", 2048, 1, "mild"),
    ("t2048_s2_harsh", 2048, 2, "harsh"),
    ("t4096_s3_harsh", 4096, 3, "harsh"),
]


def _pcc(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def _bits(t):
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16)
    if t.dtype == torch.float32:
        return t.contiguous().view(torch.int32)
    return t


def _save_or_check(name, tensors):
    os.makedirs(REF_DIR, exist_ok=True)
    path = os.path.join(REF_DIR, f"{name}.pt")
    if MODE == "save":
        assert OVERWRITE or not os.path.exists(
            path
        ), f"{path} exists; refusing to overwrite a saved reference (set QWEN36_PREP_REF_OVERWRITE=1 to replace it)"
        torch.save({k: v.clone().contiguous() for k, v in tensors.items()}, path)
        logger.info(f"[ref] saved {len(tensors)} tensors -> {path}")
        return
    ref = torch.load(path)
    bad = []
    for k, v in tensors.items():
        r = ref[k]
        if r.dtype != v.dtype or r.shape != v.shape:
            bad.append(f"{k}: dtype/shape ref {r.dtype}{tuple(r.shape)} vs new {v.dtype}{tuple(v.shape)}")
            continue
        nbits = int((_bits(r) != _bits(v)).sum())
        equal = torch.equal(r, v)
        nd = (r.float() - v.float()).abs()
        logger.info(
            f"[ref] {name}/{k}: torch.equal={equal} bitwise mismatches={nbits}/{v.numel()} "
            f"max|d|={nd.max().item():.3e} pcc={_pcc(r, v):.8f}"
        )
        if not equal:
            bad.append(f"{k}: value mismatches={int((nd != 0).sum())} max|d|={nd.max().item():.3e}")
        elif nbits:
            logger.warning(f"[ref] {name}/{k}: values equal but {nbits} signed-zero bit differences")
    assert not bad, f"{name}: NOT bit-identical to {path}:\n  " + "\n  ".join(bad)
    logger.info(f"[ref] {name}: {len(tensors)} tensors bit-identical to {path}")


def _inputs(T, seed, gdist):
    torch.manual_seed(seed)
    q = torch.randn(1, T, NK * DK, dtype=torch.bfloat16)
    k = torch.randn(1, T, NK * DK, dtype=torch.bfloat16)
    v = torch.randn(1, T, NV * DV, dtype=torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, T, NV)).to(torch.bfloat16)
    if gdist == "mild":
        g = (-torch.nn.functional.softplus(torch.randn(1, T, NV)) * 0.1).to(torch.bfloat16)
    else:  # the model's g = -exp(A) * softplus(a + dt_bias): heavier decay, wider spread
        g = (-torch.exp(0.3 * torch.randn(1, 1, NV)) * torch.nn.functional.softplus(torch.randn(1, T, NV))).to(
            torch.bfloat16
        )
    return q, k, v, beta, g


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_gdn_prep_bitexact(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    rep = ttnn.ReplicateTensorToMesh(mesh)
    consts = build_fused_const_tiles(mesh)

    def up(t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

    def first(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])

    bench = {}
    for name, T, seed, gdist in CASES:
        q, k, v, beta, g = _inputs(T, seed, gdist)
        tq, tk, tv, tb, tg = up(q), up(k), up(v), up(beta), up(g)
        # Explicit zero initial state (as the model's persistent rec_state): initial_state=None makes the op build
        # its zero state with a host write, which TT_FATALs inside the trace capture below.
        ts0 = up(torch.zeros(1, NV, DK, DV), ttnn.float32)

        def run():
            return chunk_gated_delta_rule_fused_adapter(
                tq,
                tk,
                tv,
                tb,
                tg,
                scale=1.0 / math.sqrt(DK),
                initial_state=ts0,
                device=mesh,
                qkv_head_dims=(NK, DK, NV, DV),
                return_o_bh=True,
                const_tiles=consts,
            )

        o, fs = run()
        ttnn.synchronize_device(mesh)
        o_t, fs_t = first(o), first(fs)
        finite = bool(torch.isfinite(o_t.float()).all()) and bool(torch.isfinite(fs_t.float()).all())
        logger.info(
            f"[prep] {name}: o {tuple(o_t.shape)} {o_t.dtype} fs {tuple(fs_t.shape)} {fs_t.dtype} finite={finite}"
        )
        assert finite, f"{name}: non-finite output"
        # Determinism within the run (two eager calls must agree bit for bit).
        o2, fs2 = run()
        ttnn.synchronize_device(mesh)
        assert torch.equal(first(o2), o_t) and torch.equal(first(fs2), fs_t), f"{name}: op is not deterministic"
        ttnn.deallocate(o2)
        ttnn.deallocate(fs2)
        _save_or_check(name, {"o": o_t, "final_state": fs_t})
        ttnn.deallocate(o)
        ttnn.deallocate(fs)

        if T == 2048 and "t2048" not in bench:
            # 50 traced replays of prep + scan (+ adapter reshapes): the prep gain shows as the delta vs the reference.
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            o, fs = run()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            assert torch.equal(first(o), o_t) and torch.equal(
                first(fs), fs_t
            ), f"{name}: traced replay differs from eager"
            # BATCHES x REPLAYS: report min and median of the per-batch means (process-to-process noise is a few %).
            pers = []
            for _ in range(BATCHES):
                t0 = time.perf_counter()
                for _ in range(REPLAYS):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                pers.append(1e6 * (time.perf_counter() - t0) / REPLAYS)
            bench["t2048"] = (min(pers), sorted(pers)[len(pers) // 2])
            ttnn.release_trace(mesh, tid)
            ttnn.deallocate(o)
            ttnn.deallocate(fs)
        for t in (tq, tk, tv, tb, tg, ts0):
            ttnn.deallocate(t)
    print(
        f"PREP_BENCH_RESULT mode={MODE} T=2048 prep+scan traced us/call min={bench['t2048'][0]:.1f} "
        f"med={bench['t2048'][1]:.1f} ({BATCHES}x{REPLAYS} replays)"
    )
