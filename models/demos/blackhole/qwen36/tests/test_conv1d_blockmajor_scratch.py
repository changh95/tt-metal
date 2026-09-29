# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Round-4 item 11 (conv1d block-major work order): bit-exactness + timing of ttnn.experimental.kda.qkv_causal_conv1d_silu
on the REAL served shapes of the Qwen3.8-27B GDN prefill at TP=4 (C = 2560 = 512 + 512 + 1536, K = 4 taps, HiFi4 with
fp32 DEST accumulation, exactly the call gdn/tp.py _conv1d_prefill_kda makes), for every (T, channel_chunk_size) the
model's _kda_chunk_for picks: (128, 32) (256, 64) (512, 128) (1024, 256) (2048, 512), TILE input (QWEN36_KDA_TILE_IN=1,
the served default) plus the ROW_MAJOR-input kernel on the 2048-row chunk.

Modes (QWEN36_CONV_REF=save|check, QWEN36_CONV_REF_DIR, default logs/r4_conv1d_ref):
  save   on the UNPATCHED tree: dump q/k/v of every shape (per-device concat) + a CPU float reference sanity check.
  check  on the patched tree: torch.equal against the dump for every shape (the design claims bit-identical outputs).
Both modes then time the op: one trace of N_OPS back-to-back ops per shape, replayed REPS times (>= 30), per-op us.
Prints CONV1D_RESULT <shape>=<us> ... and CONV1D_EXACT for the check mode.

Run (half A):
  TT_VISIBLE_DEVICES=0,1,6,7 MESH_DEVICE=P150x4 QWEN36_CONV_REF=save \
  python_env/bin/python -m pytest models/demos/blackhole/qwen36/tests/test_conv1d_blockmajor_scratch.py -x -s
"""

import os
import time

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS

MODE = os.environ.get("QWEN36_CONV_REF", "check")
assert MODE in ("save", "check"), MODE
REF_DIR = os.environ.get("QWEN36_CONV_REF_DIR", "logs/r4_conv1d_ref")
N_OPS = int(os.environ.get("QWEN36_CONV_NOPS", "48"))
REPS = int(os.environ.get("QWEN36_CONV_REPS", "30"))
KD, VD = 512, 1536  # key_dim_tp, value_dim_tp at TP=4 (16 x 128 / 4, 48 x 128 / 4)
C = 2 * KD + VD  # 2560 = gdn_qkv_dim_tp
K = 4
# (T, channel_chunk_size) as gdn/tp.py _kda_chunk_for picks on the 110-core BH grid, plus the RM-input kernel once.
SHAPES = [(128, 32, True), (256, 64, True), (512, 128, True), (1024, 256, True), (2048, 512, True), (2048, 512, False)]

_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)


def _inputs(T, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, C, generator=g).to(torch.bfloat16)
    hist = torch.randn(1, K - 1, C, generator=g).to(torch.bfloat16)
    taps = [(0.5 * torch.randn(1, 1, C, generator=g)).to(torch.bfloat16) for _ in range(K)]
    return x, hist, taps


def _cpu_ref(x, hist, taps):
    window = torch.cat((hist.float(), x.float()), dim=1)
    conv = sum(window[:, j : j + x.shape[1]] * taps[j].float() for j in range(K))
    return [t.to(torch.bfloat16) for t in F.silu(conv).split((KD, KD, VD), dim=-1)]


def _run(x_tt, hist_tt, taps_tt, chunk):
    return ttnn.experimental.kda.qkv_causal_conv1d_silu(
        x_tt,
        hist_tt,
        taps_tt[0],
        taps_tt[1],
        taps_tt[2],
        taps_tt[3],
        KD,
        KD,
        VD,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=chunk),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_CFG,
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_conv1d_blockmajor(mesh_device):
    device = mesh_device
    device.enable_program_cache()
    rep = ttnn.ReplicateTensorToMesh(device)
    comp = ttnn.ConcatMeshToTensor(device, dim=0)
    os.makedirs(REF_DIR, exist_ok=True)
    path = os.path.join(REF_DIR, "conv1d_qkv.pt")
    ref = torch.load(path)["tensors"] if MODE == "check" else None
    out = {}
    times = {}
    exact_all = True
    for T, chunk, tile_in in SHAPES:
        name = f"T{T}_c{chunk}_{'tile' if tile_in else 'rm'}"
        x, hist, taps = _inputs(T, 1000 + T + (0 if tile_in else 1))
        layout = ttnn.TILE_LAYOUT if tile_in else ttnn.ROW_MAJOR_LAYOUT
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, mesh_mapper=rep)
        hist_tt = ttnn.from_torch(hist, dtype=ttnn.bfloat16, layout=layout, device=device, mesh_mapper=rep)
        taps_tt = [
            ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep)
            for t in taps
        ]
        q, k, v = _run(x_tt, hist_tt, taps_tt, chunk)  # compile
        ttnn.synchronize_device(device)
        got = [ttnn.to_torch(t, mesh_composer=comp).clone() for t in (q, k, v)]
        for t in (q, k, v):
            ttnn.deallocate(t)
        # CPU sanity (fp32 CPU conv vs HiFi4/fp32-acc device: informational, not the bar).
        cpu = _cpu_ref(x, hist, taps)
        n_dev = got[0].shape[0]
        maxd = max(float((g[:1].float() - c.float()).abs().max()) for g, c in zip(got, cpu))
        same_dev = all(torch.equal(g[:1], g[d : d + 1]) for g in got for d in range(n_dev))
        logger.info(
            f"[conv1d] {name}: Mt={T // 32} blocks={C // chunk} max|dev-cpu|={maxd:.4g} replicas identical={same_dev}"
        )
        assert same_dev, f"{name}: device replicas differ"
        for nm, g in zip(("q", "k", "v"), got):
            out[f"{name}_{nm}"] = g
            if ref is not None:
                r = ref[f"{name}_{nm}"]
                eq = torch.equal(r, g)
                nmis = int((r.float() != g.float()).sum())
                logger.info(
                    f"[conv1d] {name} {nm}: {'EXACT' if eq else f'DIFF ({nmis} mismatches, max|d|={float((r.float() - g.float()).abs().max()):.4g})'}"
                )
                exact_all &= eq
        # Timing: N_OPS back-to-back ops in one trace, REPS replays.
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(N_OPS):
            o = _run(x_tt, hist_tt, taps_tt, chunk)
            for t in o:
                ttnn.deallocate(t)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)  # warm replay
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            ts.append((time.perf_counter() - t0) * 1e6 / N_OPS)
        ttnn.release_trace(device, tid)
        ts.sort()
        times[name] = ts[len(ts) // 2]
        logger.info(
            f"[conv1d] {name}: per-op us median={ts[len(ts) // 2]:.1f} min={ts[0]:.1f} max={ts[-1]:.1f} ({REPS} replays x {N_OPS} ops)"
        )
        for t in (x_tt, hist_tt, *taps_tt):
            ttnn.deallocate(t)
    if MODE == "save":
        assert not os.path.exists(path) or os.environ.get("QWEN36_CONV_REF_OVERWRITE") == "1", f"{path} exists"
        torch.save({"tensors": out}, path)
        logger.info(f"[conv1d] saved {len(out)} tensors -> {path}")
    print("CONV1D_RESULT " + " ".join(f"{k}={v:.1f}us" for k, v in times.items()))
    if MODE == "check":
        print(f"CONV1D_EXACT {'ALL' if exact_all else 'FAIL'} ({len(out)} tensors vs {path})")
        assert exact_all, "conv1d outputs are not bit-identical to the reference"
