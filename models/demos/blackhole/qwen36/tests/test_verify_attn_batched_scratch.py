# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device, no model weights): building blocks of the BATCHED attention middle of the speculative-decoding
verify step (one KV write + one SDPA per layer instead of T per-offset passes).

  MESH_DEVICE=P150x4 TT_VISIBLE_DEVICES=2,3,4,5 pytest models/demos/blackhole/qwen36/tests/test_verify_attn_batched_scratch.py -s

Probes (PROBE lines):
  rmsnorm_rows      rms_norm(+weight) on [1,1,R,HD] rows == the decode-shaped [1,B,32(pad),HD] rows (bitwise)
  rope_rows         partial rope of q [1,R,NH,HD] / k [1,1,R,HD] with per-row cos/sin == the per-offset decode rope
  q_roundtrip       [1,1,R,NH*HD] -> RM -> [1,R,NH,HD] -> TILE (pad) and back are exact
  spread_matmul     0/1 [w*32,R] @ [R,HD] spreads rows s*T+j to (s, j) exactly
  sdpa_virtual      paged SDPA decode with B' = R virtual users (row = (user, token), cur_pos P_s+j, duplicated page
                    table rows) vs T per-offset calls at B = w: bitwise / PCC, at short and multi-chunk positions
  update_multi      paged_update_cache(num_tokens=T) (if the build has it) == T sequential single-row calls, bitwise,
                    incl. spans crossing a tile / block boundary and the bf8 cache
  timing            traced: T per-offset SDPA calls at B=w vs the virtual-user SDPA (B'=R, split at <= 110 users)
Env: PROBE_W (8), PROBE_T (8), PROBE_TIMING (1).
"""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tt import verify_grid as vg
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, apply_partial_rope_prefill

NH, NKV, HD, RD = 6, 1, 256, 64  # Qwen3.8-27B at TP=4
BLK = 64
NBLOCKS = 512
W = int(os.environ.get("PROBE_W", "8"))
T = int(os.environ.get("PROBE_T", "8"))
DO_TIMING = os.environ.get("PROBE_TIMING", "1") == "1"
_EXACT_MM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)


def _rep(mesh, t, dtype, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, memory_config=mem, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _t0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() == 0 or a.std() == 0 or b.std() == 0:
        return float(torch.equal(a, b))
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _kv_shard(B):
    cols = next(c for c in range(min(8, B), 0, -1) if B % c == 0)
    return ttnn.create_sharded_memory_config(
        shape=(32, HD),
        core_grid=ttnn.CoreGrid(x=cols, y=B // cols),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _sdpa_cfg(mesh, max_cores=None):
    g = mesh.compute_with_storage_grid_size()
    kw = {"max_cores_per_head_batch": max_cores} if max_cores else {}
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0, **kw
    )


def _sdpa(mesh, q, k, v, pt, pos, max_cores=None):
    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        k,
        v,
        page_table_tensor=pt,
        cur_pos_tensor=pos,
        scale=HD**-0.5,
        program_config=_sdpa_cfg(mesh, max_cores),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_verify_attn_batched_probes(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    torch.manual_seed(0)
    w, Tk = W, T
    R = vg.grid_rows(w, Tk)
    assert R == w * Tk
    results = {}
    _L1 = ttnn.L1_MEMORY_CONFIG

    # ---------------- rms_norm rows ----------------
    try:
        x = torch.randn(w, Tk, HD).to(torch.bfloat16)
        wgt = (torch.rand(1, 1, 1, HD) + 0.5).to(torch.bfloat16)
        wt = _rep(mesh, wgt, ttnn.bfloat16)
        # reference: per offset j, decode-shaped [1,w,1(pad 32),HD]
        ref = torch.zeros(w, Tk, HD, dtype=torch.bfloat16)
        for j in range(Tk):
            xj = torch.zeros(1, w, 32, HD, dtype=torch.bfloat16)
            xj[0, :, 0, :] = x[:, j]
            xt = _rep(mesh, xj, ttnn.bfloat16, mem=_L1)
            y = ttnn.multiply(ttnn.rms_norm(xt, epsilon=1e-6, memory_config=_L1), wt, memory_config=_L1)
            ref[:, j] = _t0(y)[0, :, 0, :]
        xr = _rep(mesh, x.reshape(1, 1, R, HD), ttnn.bfloat16, mem=_L1)
        yr = ttnn.multiply(ttnn.rms_norm(xr, epsilon=1e-6, memory_config=_L1), wt, memory_config=_L1)
        got = _t0(yr).reshape(w, Tk, HD)
        results["rmsnorm_rows"] = f"exact={torch.equal(got, ref)} pcc={_pcc(got, ref):.6f}"
        # q-shaped: [1,R,NH(pad 32),HD] vs per-offset [1,w,NH(pad 32),HD]
        xq = torch.randn(w, Tk, NH, HD).to(torch.bfloat16)
        refq = torch.zeros(w, Tk, NH, HD, dtype=torch.bfloat16)
        for j in range(Tk):
            xj = torch.zeros(1, w, 32, HD, dtype=torch.bfloat16)
            xj[0, :, :NH, :] = xq[:, j]
            xt = ttnn.from_torch(
                xj[:, :, :NH],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=_L1,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            y = ttnn.multiply(ttnn.rms_norm(xt, epsilon=1e-6, memory_config=_L1), wt, memory_config=_L1)
            refq[:, j] = _t0(y)[0, :, :NH, :]
        xt = _rep(mesh, xq.reshape(1, R, NH, HD), ttnn.bfloat16, mem=_L1)
        y = ttnn.multiply(ttnn.rms_norm(xt, epsilon=1e-6, memory_config=_L1), wt, memory_config=_L1)
        gotq = _t0(y)[0, :, :NH, :].reshape(w, Tk, NH, HD)
        results["rmsnorm_q_rows"] = f"exact={torch.equal(gotq, refq)} pcc={_pcc(gotq, refq):.6f}"
    except Exception as e:  # noqa: BLE001
        results["rmsnorm_rows"] = f"FAIL {str(e)[:300]}"

    # ---------------- rope rows ----------------
    try:
        theta = 10000000.0
        pos = torch.tensor([100 + 37 * s for s in range(w)], dtype=torch.int32)
        q = torch.randn(w, Tk, NH, HD).to(torch.bfloat16)
        k = torch.randn(w, Tk, HD).to(torch.bfloat16)
        refq = torch.zeros_like(q)
        refk = torch.zeros_like(k)
        for j in range(Tk):
            cos, sin = vg.rope_cos_sin(pos + j, RD, theta)
            cos_t, sin_t = _rep(mesh, cos, ttnn.bfloat16), _rep(mesh, sin, ttnn.bfloat16)
            qt = _rep(mesh, q[:, j].reshape(1, w, NH, HD), ttnn.bfloat16, mem=_L1)
            refq[:, j] = _t0(apply_partial_rope_decode(qt, cos_t, sin_t, NH, w, RD))[0, :, :NH]
            kt = _rep(mesh, k[:, j].reshape(1, w, 1, HD), ttnn.bfloat16, mem=_L1)
            refk[:, j] = _t0(apply_partial_rope_decode(kt, cos_t, sin_t, 1, w, RD))[0, :, 0]
        rows_pos = torch.tensor([int(pos[s]) + j for s in range(w) for j in range(Tk)], dtype=torch.int32)
        cos, sin = vg.rope_cos_sin(rows_pos, RD, theta)
        cos_t, sin_t = _rep(mesh, cos, ttnn.bfloat16), _rep(mesh, sin, ttnn.bfloat16)
        qt = _rep(mesh, q.reshape(1, R, NH, HD), ttnn.bfloat16, mem=_L1)
        gotq = _t0(apply_partial_rope_decode(qt, cos_t, sin_t, NH, R, RD))[0, :, :NH].reshape(w, Tk, NH, HD)
        kt = _rep(mesh, k.reshape(1, 1, R, HD), ttnn.bfloat16, mem=_L1)
        gotk = _t0(apply_partial_rope_prefill(kt, cos_t, sin_t, 1, RD))[0, 0].reshape(w, Tk, HD)
        results["rope_rows"] = (
            f"q exact={torch.equal(gotq, refq)} pcc={_pcc(gotq, refq):.6f} | "
            f"k(prefill-shaped) exact={torch.equal(gotk, refk)} pcc={_pcc(gotk, refk):.6f}"
        )
    except Exception as e:  # noqa: BLE001
        results["rope_rows"] = f"FAIL {str(e)[:300]}"

    # ---------------- q round trip ----------------
    try:
        qf = torch.randn(1, 1, R, NH * HD).to(torch.bfloat16)
        qt = _rep(mesh, qf, ttnn.bfloat16, mem=_L1)
        rm = ttnn.to_layout(qt, ttnn.ROW_MAJOR_LAYOUT)
        rm4 = ttnn.reshape(rm, (1, R, NH, HD))
        q4 = ttnn.to_layout(rm4, ttnn.TILE_LAYOUT)
        got = _t0(q4)
        ok1 = torch.equal(got[0, :, :NH], qf.reshape(R, NH, HD)) and tuple(q4.shape) == (1, R, NH, HD)
        # back: [1,R,NH(pad),HD] -> RM -> [1,1,R,NH*HD] -> TILE
        rmb = ttnn.to_layout(q4, ttnn.ROW_MAJOR_LAYOUT)
        rmb2 = ttnn.reshape(rmb, (1, 1, R, NH * HD))
        back = ttnn.to_layout(rmb2, ttnn.TILE_LAYOUT)
        ok2 = torch.equal(_t0(back), qf)
        results["q_roundtrip"] = f"fwd exact={ok1} padded={tuple(q4.padded_shape)} back exact={ok2}"
    except Exception as e:  # noqa: BLE001
        results["q_roundtrip"] = f"FAIL {str(e)[:300]}"

    # ---------------- spread matmul ----------------
    try:
        k = torch.randn(1, 1, R, HD).to(torch.bfloat16)
        spread = torch.zeros(1, 1, w * 32, R, dtype=torch.bfloat16)
        for s in range(w):
            for j in range(Tk):
                spread[0, 0, s * 32 + j, s * Tk + j] = 1.0
        out = ttnn.matmul(
            _rep(mesh, spread, ttnn.bfloat16),
            _rep(mesh, k, ttnn.bfloat16, mem=_L1),
            compute_kernel_config=_EXACT_MM,
            memory_config=_L1,
        )
        got = _t0(out).reshape(w, 32, HD)
        ok = torch.equal(got[:, :Tk], k.reshape(w, Tk, HD)) and bool((got[:, Tk:] == 0).all())
        v4 = ttnn.reshape(out, (1, w, 32, HD))
        sh = ttnn.to_memory_config(v4, _kv_shard(w))
        results["spread_matmul"] = f"exact={ok} view={tuple(v4.shape)} sharded={sh.memory_config().is_sharded()}"
    except Exception as e:  # noqa: BLE001
        results["spread_matmul"] = f"FAIL {str(e)[:300]}"

    # ---------------- KV cache + SDPA virtual users ----------------
    kc = ((torch.randn(NBLOCKS, NKV, BLK, HD)) * 0.5).to(torch.bfloat16)
    vc = ((torch.randn(NBLOCKS, NKV, BLK, HD)) * 0.5).to(torch.bfloat16)
    keys = _rep(mesh, kc, ttnn.bfloat8_b)
    values = _rep(mesh, vc, ttnn.bfloat8_b)
    bpu = 32  # blocks per user (2048 tokens)
    pt = torch.zeros(w, bpu, dtype=torch.int32)
    for s in range(w):
        pt[s] = torch.arange(bpu) + s * bpu
    pt_t = _rep(mesh, pt, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pt_rows = pt.repeat_interleave(Tk, dim=0)  # [R, bpu]
    pt_rows_t = _rep(mesh, pt_rows, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    for tag, base in (("short", 40), ("mid", 300), ("long", 1500), ("edge", 1024 - 3)):
        try:
            pos = torch.tensor([base + 5 * s for s in range(w)], dtype=torch.int32)
            q = (torch.randn(w, Tk, NH, HD) * 0.3).to(torch.bfloat16)
            ref = torch.zeros(w, Tk, NH, HD, dtype=torch.bfloat16)
            for j in range(Tk):
                qt = _rep(mesh, q[:, j].reshape(1, w, NH, HD), ttnn.bfloat16)
                pj = _rep(mesh, pos + j, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                o = _sdpa(mesh, qt, keys, values, pt_t, pj)
                ref[:, j] = _t0(o)[0, :, :NH]
            qr = _rep(mesh, q.reshape(1, R, NH, HD), ttnn.bfloat16)
            rows_pos = torch.tensor([int(pos[s]) + j for s in range(w) for j in range(Tk)], dtype=torch.int32)
            pr = _rep(mesh, rows_pos, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            outs = []
            n_split = -(-R // 110)
            per = -(-R // n_split)
            for c in range(n_split):
                a, b = c * per, min(R, (c + 1) * per)
                if n_split == 1:
                    o = _sdpa(mesh, qr, keys, values, pt_rows_t, pr)
                else:
                    qs = ttnn.slice(qr, (0, a, 0, 0), (1, b, NH, HD))
                    pts = _rep(mesh, pt_rows[a:b], ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                    prs = _rep(mesh, rows_pos[a:b], ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                    o = _sdpa(mesh, qs, keys, values, pts, prs)
                outs.append(_t0(o)[0, :, :NH])
            got = torch.cat(outs, 0).reshape(w, Tk, NH, HD)
            # host reference (fp32) for both
            kf = kc.float()
            vf = vc.float()
            hq = torch.zeros(w, Tk, NH, HD)
            for s in range(w):
                for j in range(Tk):
                    L = int(pos[s]) + j + 1
                    blocks = pt[s, : -(-L // BLK)]
                    K = kf[blocks, 0].reshape(-1, HD)[:L]
                    V = vf[blocks, 0].reshape(-1, HD)[:L]
                    sc = (q[s, j].float() @ K.T) * HD**-0.5
                    hq[s, j] = torch.softmax(sc, -1) @ V
            results[f"sdpa_virtual_{tag}"] = (
                f"B'={R} splits={n_split} exact={torch.equal(got, ref)} pcc(got,ref)={_pcc(got, ref):.6f} "
                f"pcc(ref,host)={_pcc(ref, hq):.5f} pcc(got,host)={_pcc(got, hq):.5f} maxabs={float((got.float() - ref.float()).abs().max()):.4f}"
            )
        except Exception as e:  # noqa: BLE001
            results[f"sdpa_virtual_{tag}"] = f"FAIL {str(e)[:400]}"

    # ---------------- multi-token paged_update_cache ----------------
    doc = getattr(ttnn.experimental.paged_update_cache, "__doc__", "") or ""
    if "num_tokens" in doc:
        for tag, base in (
            ("intile", 40),
            ("cross_tile", 60),
            ("cross_block", 60 + 64 * 3),
            ("tile_end", 31 - Tk + 1 + 64),
        ):
            try:
                c1 = ttnn.from_torch(
                    kc,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                c2 = ttnn.from_torch(
                    kc,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                pos = torch.tensor([base + 3 * s for s in range(w)], dtype=torch.int32)
                new = torch.randn(w, Tk, HD).to(torch.bfloat16)
                # reference: T sequential single-row calls
                for j in range(Tk):
                    xj = torch.zeros(1, w, 32, HD, dtype=torch.bfloat16)
                    xj[0, :, 0] = new[:, j]
                    sh = ttnn.to_memory_config(_rep(mesh, xj, ttnn.bfloat16), _kv_shard(w))
                    pj = _rep(mesh, pos + j, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.experimental.paged_update_cache(c1, sh, update_idxs_tensor=pj, page_table=pt_t)
                # one multi-token call
                xm = torch.zeros(1, w, 32, HD, dtype=torch.bfloat16)
                xm[0, :, :Tk] = new
                shm = ttnn.to_memory_config(_rep(mesh, xm, ttnn.bfloat16), _kv_shard(w))
                p0 = _rep(mesh, pos, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                ttnn.experimental.paged_update_cache(c2, shm, update_idxs_tensor=p0, page_table=pt_t, num_tokens=Tk)
                a, b = _t0(c1), _t0(c2)
                changed = int((a != _t0(keys)).sum().item())
                results[
                    f"update_multi_{tag}"
                ] = f"exact={torch.equal(a, b)} rows_changed_elems={changed} (expect {w * Tk * HD}) maxabs={float((a.float() - b.float()).abs().max()):.4f}"
            except Exception as e:  # noqa: BLE001
                results[f"update_multi_{tag}"] = f"FAIL {str(e)[:400]}"
    else:
        results["update_multi"] = "SKIP (build without num_tokens)"

    # ---------------- timing ----------------
    if DO_TIMING:
        try:
            base = 300
            pos = torch.tensor([base + 5 * s for s in range(w)], dtype=torch.int32)
            q = (torch.randn(w, Tk, NH, HD) * 0.3).to(torch.bfloat16)
            qj_t = [_rep(mesh, q[:, j].reshape(1, w, NH, HD), ttnn.bfloat16) for j in range(Tk)]
            pj_t = [_rep(mesh, pos + j, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT) for j in range(Tk)]
            qr = _rep(mesh, q.reshape(1, R, NH, HD), ttnn.bfloat16)
            rows_pos = torch.tensor([int(pos[s]) + j for s in range(w) for j in range(Tk)], dtype=torch.int32)
            pr = _rep(mesh, rows_pos, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            n_split = -(-R // 110)
            per = -(-R // n_split)
            splits = []
            for c in range(n_split):
                a, b = c * per, min(R, (c + 1) * per)
                splits.append(
                    (
                        a,
                        b,
                        _rep(mesh, pt_rows[a:b], ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                        _rep(mesh, rows_pos[a:b], ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
                    )
                )
            mc_ref = 1 if w >= 32 else None  # B=32 back-to-back SDPA with 3 cores/head is the suspected wedge site

            def body_ref():
                outs = []
                for j in range(Tk):
                    outs.append(_sdpa(mesh, qj_t[j], keys, values, pt_t, pj_t[j], max_cores=mc_ref))
                return outs

            def body_virt():
                outs = []
                for a, b, pts, prs in splits:
                    qs = qr if n_split == 1 else ttnn.slice(qr, (0, a, 0, 0), (1, b, NH, HD))
                    outs.append(_sdpa(mesh, qs, keys, values, pts, prs))
                return outs

            def timed(body, name, n=30):
                outs = body()
                ttnn.synchronize_device(mesh)
                for o in outs:
                    ttnn.deallocate(o)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                outs = body()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                ms = []
                for _ in range(n):
                    t0 = time.perf_counter()
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    ms.append(1e3 * (time.perf_counter() - t0))
                ttnn.release_trace(mesh, tid)
                ms.sort()
                return ms[len(ms) // 2]

            t_ref = timed(body_ref, "ref")
            t_virt = timed(body_virt, "virt")
            results["timing"] = (
                f"(w={w},T={Tk}) traced ms: {Tk} SDPA calls at B={w} (max_cores={mc_ref}) = {t_ref:.3f} | "
                f"virtual-user SDPA B'={R} in {n_split} call(s) = {t_virt:.3f}"
            )
        except Exception as e:  # noqa: BLE001
            results["timing"] = f"FAIL {str(e)[:400]}"

    for k_, v_ in results.items():
        logger.info(f"PROBE {k_}: {v_}")
        print(f"PROBE {k_}: {v_}")
