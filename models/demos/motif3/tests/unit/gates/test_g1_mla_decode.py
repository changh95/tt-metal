# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G1 — ``ttnn.transformer.paged_flash_multi_latent_attention_decode`` at Motif decode shapes.

Design §1.2 (row 1), §2.3.4 decode step 8, §4.2 G1, §7.2 Q1. Per chip (one TP column of one DP row):
  Q  [1, B=8 users, NH=10 heads, 576]  (absorbed q_lat 512 ‖ roped q_pe 64), bf16
  K  paged latent cache [num_blocks, 1, block, 576] bfloat8_b (V = first 512 columns, ``v=None``)
  head_dim_v = 512, nkv = 1, scale 0.07216878 (SWA) / 0.14467963 (global), sliding_window_size 129 / None.

Golden: fp64 masked attention over the *quantised* cache (so the comparison isolates the op), plus
an end-to-end PCC against the unquantised cache. "Window exact" is checked with probe caches whose
boundary keys dominate the softmax: the key just inside the window (V=+1), the key just outside
(V=-1, larger score) and the first future key p+1 (V=+3, largest score). The output must be ≈ +1.

Pass (design §4.2): PCC ≥ 0.999 vs torch on the quantised cache, probes exact, all 32 replicas identical.
"""

from __future__ import annotations

import math

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

B = 8
NH = 10
D_LAT = 512
D_ROPE = 64
D_QK = D_LAT + D_ROPE
POSITIONS = [0, 1, 127, 128, 129, 130, 1000, 5000]  # design §4.2
PROBE_POSITIONS = [129, 159, 160, 161, 255, 256, 1000, 5000]  # window start on/around tile & chunk edges
SEQ = 5120  # >= 5001 keys, a multiple of every block size tested
PCC_MIN = 0.999

REC = gu.Recorder("G1")
FAB2D = gu.mesh_params()


# ----------------------------------------------------------------------------------------------
# host-side builders
# ----------------------------------------------------------------------------------------------
def make_page_table(blocks_per_user: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(B * blocks_per_user, generator=g)
    return perm.reshape(B, blocks_per_user).to(torch.int32)


def to_paged(kv: torch.Tensor, page_table: torch.Tensor, block: int) -> torch.Tensor:
    """kv [B, S, D] -> paged [B*bpu, 1, block, D] with paged[page_table[b, j]] = kv[b, j*block:(j+1)*block]."""
    bpu = page_table.shape[1]
    paged = torch.empty(B * bpu, 1, block, kv.shape[-1], dtype=kv.dtype)
    paged[page_table.reshape(-1).long(), 0] = kv.reshape(B * bpu, block, kv.shape[-1])
    return paged


def random_case(seed: int):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, NH, D_QK, generator=g).bfloat16().float()
    kv = torch.randn(B, SEQ, D_QK, generator=g).bfloat16().float()  # unit-RMS latent ‖ k_pe
    return q, kv


def probe_case(positions, window, seed: int):
    """Boundary-key probe: scores are carried by the 64 rope dims only, so V (first 512 dims) is free."""
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(D_ROPE, generator=g)
    e = e / e.norm()
    scale = gd.SCALE_SWA if window is not None else gd.SCALE_GLOBAL
    a = 16.0  # query magnitude along e
    q = 0.05 * torch.randn(B, NH, D_QK, generator=g)
    q[:, :, D_LAT:] = a * e
    kv = 0.1 * torch.randn(B, SEQ, D_QK, generator=g)
    kv[:, :, D_LAT:] = 0.02 * torch.randn(B, SEQ, D_ROPE, generator=g)

    def key(score):  # rope part producing a given pre-softmax score
        return (score / (scale * a)) * e

    for b, p in enumerate(positions):
        lo = 0 if window is None else max(0, p - window + 1)
        kv[b, lo, :D_LAT] = 1.0  # first in-window key: dominant, V = +1
        kv[b, lo, D_LAT:] = key(20.0)
        if window is not None and lo - 1 >= 0:  # just outside the window: would dominate if leaked
            kv[b, lo - 1, :D_LAT] = -1.0
            kv[b, lo - 1, D_LAT:] = key(28.0)
        if p + 1 < SEQ:  # first future key (causal edge): would dominate if leaked
            kv[b, p + 1, :D_LAT] = 3.0
            kv[b, p + 1, D_LAT:] = key(36.0)
    return q.bfloat16().float(), kv.bfloat16().float()


# ----------------------------------------------------------------------------------------------
# device helpers
# ----------------------------------------------------------------------------------------------
def q_memory_configs(mesh_device, sharded: bool, nh_pad: int):
    if not sharded:
        return ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    grid = gu.grid_size(mesh_device)
    cores = ttnn.num_cores_to_corerangeset(B, grid, row_wise=True)
    rows = ((nh_pad + 31) // 32) * 32
    q_mc = ttnn.create_sharded_memory_config(
        shape=(rows, D_QK),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    o_mc = ttnn.create_sharded_memory_config(
        shape=(rows, D_LAT),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    return q_mc, o_mc


def program_config(mesh_device, k_chunk: int, max_cores_per_head_batch: int = 16):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=gu.grid_size(mesh_device),
        q_chunk_size=0,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
        max_cores_per_head_batch=max_cores_per_head_batch,
    )


def mla_compute_cfg(fp32_acc: bool = False):
    return gu.compute_cfg("HiFi4", fp32_acc=fp32_acc, approx=False, packer_l1_acc=False)


class Setup:
    """Device tensors for one (cache, block) configuration."""

    def __init__(self, mesh_device, kv: torch.Tensor, block: int, cache_dtype, seed: int):
        self.block = block
        self.page_table = make_page_table(SEQ // block, seed)
        self.kv_q = gu.host_roundtrip(kv, cache_dtype) if cache_dtype != ttnn.bfloat16 else kv.bfloat16().float()
        paged = to_paged(kv, self.page_table, block)
        self.cache = gu.to_mesh(paged, mesh_device, cache_dtype)
        self.pt = gu.to_mesh(self.page_table, mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def run_op(mesh_device, q_tt, setup: Setup, pos_tt, scale, window, pc, ckc, out_mc):
    return ttnn.transformer.paged_flash_multi_latent_attention_decode(
        q_tt,
        setup.cache,
        None,
        head_dim_v=D_LAT,
        page_table_tensor=setup.pt,
        cur_pos_tensor=pos_tt,
        scale=scale,
        sliding_window_size=window,
        program_config=pc,
        compute_kernel_config=ckc,
        memory_config=out_mc,
    )


def upload_q(mesh_device, q: torch.Tensor, nh_pad: int, q_mc):
    qp = torch.zeros(1, B, nh_pad, D_QK)
    qp[0, :, :NH] = q
    return gu.to_mesh(qp, mesh_device, ttnn.bfloat16, memory_config=q_mc)


def upload_pos(mesh_device, positions):
    return gu.to_mesh(torch.tensor(positions, dtype=torch.int32), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def readback(out_tt) -> torch.Tensor:
    o = gu.read_dev(out_tt, 0)  # [1, B, nh(pad), 512]
    return o[0, :, :NH, :]


# ----------------------------------------------------------------------------------------------
# accuracy
# ----------------------------------------------------------------------------------------------
WIN_SCALE = [
    ("swa", gd.WINDOW, gd.SCALE_SWA),  # the real SWA layer config
    ("global", None, gd.SCALE_GLOBAL),  # the real global layer config
    ("swa_globalscale", gd.WINDOW, gd.SCALE_GLOBAL),
    ("global_swascale", None, gd.SCALE_SWA),
]


@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
@pytest.mark.parametrize(
    "block, k_chunk, q_sharded, nh_pad, fp32_acc",
    [
        (64, 128, False, 10, False),
        (32, 128, False, 10, False),
        pytest.param(
            64,
            0,
            False,
            10,
            False,
            marks=pytest.mark.xfail(
                reason="known tt-metal bug (found by this gate): with k_chunk_size=0 (DYNAMIC_CHUNK_SIZE, the op default "
                "without a program config) sdpa_flash_decode.cpp fuses only the sliding-window mask when the window-start "
                "chunk is also the last chunk, dropping the causal mask -> keys > cur_pos leak",
                strict=False,
            ),
        ),  # k_chunk 0 = dynamic chunking
        (64, 128, True, 10, False),  # DeepSeek-style height-sharded Q / output
        (64, 256, False, 10, False),
        (64, 128, False, 10, True),
    ],
    ids=["b64_k128_qdram", "b32_k128_qdram", "b64_kdyn_qdram", "b64_k128_qshard", "b64_k256_qdram", "b64_k128_fp32acc"],
)
def test_g1_mla_decode_accuracy(mesh_device, block, k_chunk, q_sharded, nh_pad, fp32_acc):
    torch.manual_seed(0)
    cfg = f"block={block} k_chunk={k_chunk} q={'hsharded' if q_sharded else 'dram'} nh_pad={nh_pad} fp32_acc={fp32_acc}"
    q_mc, o_mc = q_memory_configs(mesh_device, q_sharded, nh_pad)
    pc = program_config(mesh_device, k_chunk)
    ckc = mla_compute_cfg(fp32_acc)
    failures = []

    # ---- random data, design positions ----
    q, kv = random_case(seed=1)
    st = Setup(mesh_device, kv, block, ttnn.bfloat8_b, seed=2)
    q_tt = upload_q(mesh_device, q, nh_pad, q_mc)
    pos_tt = upload_pos(mesh_device, POSITIONS)
    for name, window, scale in WIN_SCALE:
        case = f"acc/{name}/{cfg}"
        try:
            out_tt = run_op(mesh_device, q_tt, st, pos_tt, scale, window, pc, ckc, o_mc)
            got = readback(out_tt)
        except Exception as e:
            REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
            failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
            continue
        want = gd.mla_decode_golden(q, st.kv_q, POSITIONS, scale, window)
        want_e2e = gd.mla_decode_golden(q, kv, POSITIONS, scale, window)
        s = gu.compare(want, got)
        s_e2e = gu.compare(want_e2e, got)
        per_user = [gu.pcc(want[b], got[b]) for b in range(B)]
        same, nbad = gu.replicas_identical(out_tt)
        ok = s["pcc"] >= PCC_MIN and min(per_user) >= PCC_MIN - 0.004 and s["nonfinite"] == 0 and same
        REC.add(
            case,
            status="pass" if ok else "fail",
            pcc=s["pcc"],
            max_abs=s["max_abs"],
            pcc_e2e_vs_unquantised=s_e2e["pcc"],
            min_user_pcc=min(per_user),
            per_user_pcc=[round(x, 6) for x in per_user],
            replicas_identical=same,
            positions=POSITIONS,
        )
        if not ok:
            failures.append(f"{case}: {gu.fmt(s)} min_user_pcc={min(per_user):.5f} replicas_ok={same}")

    # ---- inactive lane (-1) must be skipped without disturbing the other lanes ----
    pos_skip = list(POSITIONS)
    pos_skip[2] = -1
    pos_skip_tt = upload_pos(mesh_device, pos_skip)
    try:
        out_tt = run_op(mesh_device, q_tt, st, pos_skip_tt, gd.SCALE_SWA, gd.WINDOW, pc, ckc, o_mc)
        got = readback(out_tt)
        want = gd.mla_decode_golden(q, st.kv_q, pos_skip, gd.SCALE_SWA, gd.WINDOW)
        keep = [b for b in range(B) if pos_skip[b] >= 0]
        s = gu.compare(want[keep], got[keep])
        ok = s["pcc"] >= PCC_MIN and s["nonfinite"] == 0
        REC.add(f"skip_minus1/{cfg}", status="pass" if ok else "fail", pcc_active_lanes=s["pcc"], max_abs=s["max_abs"])
        if not ok:
            failures.append(f"skip_minus1/{cfg}: {gu.fmt(s)}")
    except Exception as e:
        REC.add(f"skip_minus1/{cfg}", status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
        failures.append(f"skip_minus1/{cfg}: {type(e).__name__}")

    # ---- window / causal edge probes ----
    for name, window, scale in WIN_SCALE[:2]:
        positions = PROBE_POSITIONS
        qp, kvp = probe_case(positions, window, seed=3)
        stp = Setup(mesh_device, kvp, block, ttnn.bfloat8_b, seed=4)
        qp_tt = upload_q(mesh_device, qp, nh_pad, q_mc)
        posp_tt = upload_pos(mesh_device, positions)
        case = f"probe/{name}/{cfg}"
        try:
            out_tt = run_op(mesh_device, qp_tt, stp, posp_tt, scale, window, pc, ckc, o_mc)
            got = readback(out_tt)
        except Exception as e:
            REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
            failures.append(f"{case}: {type(e).__name__}")
            continue
        want = gd.mla_decode_golden(qp, stp.kv_q, positions, scale, window)
        # diagnose each user: mean of V-output ≈ +1 (correct), -1 (window leak), +3 (causal leak), ~0 (edge dropped)
        means = got.mean(dim=(1, 2)).tolist()
        verdict = []
        for m in means:
            if abs(m - 1.0) < 0.15:
                verdict.append("ok")
            elif abs(m + 1.0) < 0.5:
                verdict.append("WINDOW_LEAK")
            elif abs(m - 3.0) < 0.75:
                verdict.append("CAUSAL_LEAK")
            else:
                verdict.append(f"OFF({m:.2f})")
        s = gu.compare(want, got)
        ok = all(v == "ok" for v in verdict) and s["max_abs"] < 0.1
        REC.add(
            case,
            status="pass" if ok else "fail",
            max_abs=s["max_abs"],
            pcc=s["pcc"],
            positions=positions,
            user_mean=[round(m, 3) for m in means],
            verdict=verdict,
        )
        if not ok:
            failures.append(f"{case}: verdict={verdict} max_abs={s['max_abs']:.3e}")

    assert not failures, "G1 accuracy failures:\n" + "\n".join(failures)


# ----------------------------------------------------------------------------------------------
# latency (eager + traced) at 4K / 32K context
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device, device_params", [FAB2D], indirect=True, ids=["4x8_torus2d"])
@pytest.mark.parametrize(
    "q_sharded, fp32_acc", [(False, False), (True, False), (False, True)], ids=["qdram", "qshard", "qdram_fp32acc"]
)
def test_g1_mla_decode_perf(mesh_device, q_sharded, fp32_acc):
    block = 64
    q_mc, o_mc = q_memory_configs(mesh_device, q_sharded, NH)
    ckc = mla_compute_cfg(fp32_acc)
    floor = gu.time_trace_replay_floor(mesh_device)
    REC.add("perf/trace_replay_floor", us=floor)
    q = torch.randn(B, NH, D_QK).bfloat16().float()
    q_tt = upload_q(mesh_device, q, NH, q_mc)
    for ctx in (4096, 32768):
        bpu = ctx // block
        nb = B * bpu
        # device-side allocation (host-built bfp8 zeros of this size would cost ~minutes over PCIe on 32 chips)
        empty = ttnn.empty([nb, 1, block, D_QK], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        cache = ttnn.fill(empty, 0.0)
        ttnn.deallocate(empty)
        pt = gu.to_mesh(make_page_table(bpu, seed=7), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos_tt = upload_pos(mesh_device, [ctx - 1] * B)
        kv_bytes = B * ctx * D_QK * 1088 / 1024
        for name, window, scale in WIN_SCALE[:2]:
            if ctx > 4096 and window is not None:
                continue  # SWA cost is context independent
            # k_chunk 512 overflows L1 (static CBs 1.70-1.75 MB > 1.5 MB, measured 2026-10-01); 256 / 0 clash with a
            # height-sharded Q's L1 buffer, so the sharded-Q variant only runs 128.
            for k_chunk in (128,) if (q_sharded or fp32_acc) else (128, 256, 0):
                pc = program_config(mesh_device, k_chunk)

                def fn():
                    return ttnn.transformer.paged_flash_multi_latent_attention_decode(
                        q_tt,
                        cache,
                        None,
                        head_dim_v=D_LAT,
                        page_table_tensor=pt,
                        cur_pos_tensor=pos_tt,
                        scale=scale,
                        sliding_window_size=window,
                        program_config=pc,
                        compute_kernel_config=ckc,
                        memory_config=o_mc,
                    )

                case = f"perf/{name}/ctx{ctx}/k_chunk{k_chunk}/{'qshard' if q_sharded else 'qdram'}{'/fp32acc' if fp32_acc else ''}"
                try:
                    eager = gu.time_eager(mesh_device, fn, iters=20)
                    tr_slope, tr_raw = gu.time_traced(mesh_device, fn, ops_per_trace=64, reps=9)
                except Exception as e:
                    REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
                    continue
                read = kv_bytes if window is None else B * 256 * D_QK * 1088 / 1024
                REC.add(
                    case,
                    status="measured",
                    eager_us=eager,
                    traced_us=tr_slope,
                    traced_raw_us=tr_raw,
                    kv_bytes_read=read,
                    eff_GBps=read / (max(tr_slope, 1e-3) * 1e-6) / 1e9,
                )
        ttnn.deallocate(cache)
