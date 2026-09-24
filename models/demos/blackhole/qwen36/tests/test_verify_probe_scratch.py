# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: device capability probes for the speculative-decoding verify step (no model weights).

Each probe answers one question the verify-step design depends on; results are logged as PROBE lines.
  MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_verify_probe_scratch.py -s
"""

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp


def _rep(mesh, t, dtype, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, memory_config=mem, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _t0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


@parametrize_mesh_tp()
def test_verify_probes(mesh_device):
    mesh = mesh_device
    results = {}

    # 1. broadcast multiply of a per-user [B,1,1,1] 0/1 mask over [B, Nv, Dk, Dv] (fp32 and bf16)
    for dt, tdt in ((ttnn.float32, torch.float32), (ttnn.bfloat16, torch.bfloat16)):
        try:
            S = torch.randn(8, 8, 128, 128).to(tdt)
            m = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=tdt).reshape(8, 1, 1, 1)
            out = ttnn.multiply(_rep(mesh, S, dt), _rep(mesh, m, dt))
            got = _t0(out)
            ok = torch.equal(got, S * m)
            results[f"bcast_mul_{dt}"] = f"ok exact={ok}"
        except Exception as e:  # noqa: BLE001
            results[f"bcast_mul_{dt}"] = f"FAIL {str(e)[:200]}"

    # 1b. add of the masked pieces (exactness of x + 0)
    try:
        S = torch.randn(8, 8, 128, 128)
        m = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.float32).reshape(8, 1, 1, 1)
        a = ttnn.multiply(_rep(mesh, S, ttnn.float32), _rep(mesh, m, ttnn.float32))
        b = ttnn.multiply(_rep(mesh, S * 2, ttnn.float32), _rep(mesh, 1 - m, ttnn.float32))
        got = _t0(ttnn.add(a, b))
        results["masked_select_fp32"] = f"exact={torch.equal(got, S * m + (S * 2) * (1 - m))}"
    except Exception as e:  # noqa: BLE001
        results["masked_select_fp32"] = f"FAIL {str(e)[:200]}"

    # 2. slice dim 0 of a [T,1,w,rd] TILE tensor -> [1,1,w,rd]
    try:
        c = torch.randn(8, 1, 8, 64).to(torch.bfloat16)
        ct = _rep(mesh, c, ttnn.bfloat16)
        s = ttnn.slice(ct, (3, 0, 0, 0), (4, 1, 8, 64))
        results["slice_dim0"] = f"shape={tuple(s.shape)} exact={torch.equal(_t0(s), c[3:4])}"
    except Exception as e:  # noqa: BLE001
        results["slice_dim0"] = f"FAIL {str(e)[:200]}"

    # 3. 0/1 gather matmul exactness in bf16 ([w,R] @ [R,W]), fp32 acc
    try:
        R, w, W = 256, 32, 4128
        T = R // w
        x = torch.randn(1, 1, R, W).to(torch.bfloat16)
        sel = torch.zeros(1, 1, w, R, dtype=torch.bfloat16)
        for s_ in range(w):
            sel[0, 0, s_, s_ * T + 3] = 1.0
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        out = ttnn.matmul(_rep(mesh, sel, ttnn.bfloat16), _rep(mesh, x, ttnn.bfloat16), compute_kernel_config=cfg)
        got = _t0(out)
        ref = torch.stack([x[0, 0, s_ * T + 3] for s_ in range(w)]).reshape(1, 1, w, W)
        results["gather_matmul_hifi4"] = f"exact={torch.equal(got, ref)}"
        cfg2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        out2 = ttnn.matmul(_rep(mesh, sel, ttnn.bfloat16), _rep(mesh, x, ttnn.bfloat16), compute_kernel_config=cfg2)
        results["gather_matmul_hifi2"] = f"exact={torch.equal(_t0(out2), ref)}"
        # scatter back [R,w] @ [w,W] and sum of T of them
        selT = torch.zeros(1, 1, R, w, dtype=torch.bfloat16)
        for s_ in range(w):
            selT[0, 0, s_ * T + 3, s_] = 1.0
        back = ttnn.matmul(_rep(mesh, selT, ttnn.bfloat16), out, compute_kernel_config=cfg)
        gb = _t0(back)
        refb = torch.zeros(1, 1, R, W, dtype=torch.bfloat16)
        for s_ in range(w):
            refb[0, 0, s_ * T + 3] = x[0, 0, s_ * T + 3]
        results["scatter_matmul"] = f"exact={torch.equal(gb, refb)}"
    except Exception as e:  # noqa: BLE001
        results["gather_matmul"] = f"FAIL {str(e)[:200]}"

    # 4. sharded rms_norm with block_h = R/32 on the decode norm grid + all_gather into R-row sharded config
    try:
        from models.demos.blackhole.qwen36.tt import tp_common as tpc
        from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

        args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=32, max_seq_len=4096)
        ccl = tpc.Qwen36CCL(mesh, args)
        grid = args.attn_input_grid
        dim = args.dim
        nd = mesh.get_num_devices()
        for R in (64, 128, 256):
            block_w = dim // grid.num_cores // 32
            sub_w = next(s for s in (4, 3, 2, 1) if block_w % s == 0)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[grid.x, grid.y],
                subblock_w=sub_w,
                block_h=R // 32,
                block_w=block_w,
                inplace=False,
            )
            mc = ttnn.create_sharded_memory_config(
                (R, dim // grid.num_cores),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            xf = torch.randn(1, 1, R, dim).to(torch.bfloat16)
            # fractured input [1,1,R,dim/nd] per device
            xt = ttnn.from_torch(
                xf,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            )
            g = ttnn.experimental.all_gather_async(
                xt,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=ccl.get_num_links(1),
                topology=args.ccl_topology(),
                memory_config=mc,
                barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            gg = _t0(g)
            ag_ok = torch.equal(gg, xf)
            wgt = torch.randn(1, 1, 1, dim).to(torch.bfloat16)
            wt = _rep(mesh, wgt, ttnn.bfloat16)
            y = ttnn.rms_norm(
                g,
                epsilon=1e-6,
                weight=wt,
                program_config=pc,
                memory_config=mc,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
                ),
            )
            yy = _t0(y).float()
            ref = xf.float() * torch.rsqrt(xf.float().pow(2).mean(-1, keepdim=True) + 1e-6) * wgt.float()
            err = (yy - ref).abs().max().item()
            results[f"ag_norm_R{R}"] = f"ag_exact={ag_ok} norm_maxerr={err:.4f}"
            ttnn.deallocate(y)
            ttnn.deallocate(g)
    except Exception as e:  # noqa: BLE001
        results["ag_norm"] = f"FAIL {str(e)[:300]}"

    # 5. argmax + max on a row-major [1,1,R,V] bf16 tensor
    try:
        R, V = 256, 62080
        lg = torch.randn(1, 1, R, V).to(torch.bfloat16)
        lt = _rep(mesh, lg, ttnn.bfloat16)
        rm = ttnn.to_layout(lt, ttnn.ROW_MAJOR_LAYOUT)
        idx = ttnn.argmax(rm, dim=-1, keepdim=False)
        got = _t0(idx).reshape(-1)[:R].to(torch.int64)
        ref = lg.float().reshape(R, V).argmax(-1)
        results["argmax_rm"] = f"shape={tuple(idx.shape)} match={int((got == ref).sum())}/{R}"
        mx = ttnn.max(lt, dim=-1)
        gm = _t0(mx).float().reshape(-1)[:R]
        refm = lg.float().reshape(R, V).max(-1).values
        results["max_tile"] = f"shape={tuple(mx.shape)} match={int((gm == refm).sum())}/{R}"
    except Exception as e:  # noqa: BLE001
        results["argmax"] = f"FAIL {str(e)[:300]}"

    # 6. paged_update_cache race: T virtual users of one real user writing rows of the SAME tile in one call
    try:
        NKV, HD, BLK = 1, 256, 64
        nblocks = 8
        cache = _rep(mesh, torch.zeros(nblocks, NKV, BLK, HD), ttnn.bfloat16)
        T = 8
        kt = torch.arange(1, T + 1, dtype=torch.float32).reshape(1, T, 1, 1).expand(1, T, NKV, HD).contiguous()
        k = _rep(mesh, kt.to(torch.bfloat16), ttnn.bfloat16)
        cols = next(c for c in range(min(8, T), 0, -1) if T % c == 0)
        shard = ttnn.create_sharded_memory_config(
            shape=(32, HD),
            core_grid=ttnn.CoreGrid(x=cols, y=T // cols),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k_sh = ttnn.to_memory_config(k, shard)
        pos = torch.tensor([100 + j for j in range(T)], dtype=torch.int32)  # same tile (rows 4..11 of block 1)
        pt = torch.tensor([[0, 1, 2, 3]] * T, dtype=torch.int32)
        pos_t = _rep(mesh, pos, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pt_t = _rep(mesh, pt, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        lost = []
        for trial in range(5):
            ttnn.experimental.paged_update_cache(cache, k_sh, update_idxs_tensor=pos_t, page_table=pt_t)
            c = _t0(cache).float()
            rows = c[1, 0, 100 - 64 : 100 - 64 + T, 0]
            lost.append(int((rows != torch.arange(1, T + 1)).sum()))
        results["update_cache_same_tile_race"] = f"lost_rows_per_trial={lost} (0 = no race)"
        # control: distinct tiles (positions spread across tiles)
        pos2 = torch.tensor([64 * (j % 4) + 32 * (j // 4) + 5 for j in range(T)], dtype=torch.int32)
        pos2_t = _rep(mesh, pos2, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.paged_update_cache(cache, k_sh, update_idxs_tensor=pos2_t, page_table=pt_t)
        c = _t0(cache).float()
        ok = all(c[int(pt[0, p // 64]), 0, p % 64, 0].item() == j + 1 for j, p in enumerate(pos2.tolist()))
        results["update_cache_distinct_tiles"] = f"ok={ok}"
    except Exception as e:  # noqa: BLE001
        results["update_cache"] = f"FAIL {str(e)[:300]}"

    # 7. nlp_create_qkv_heads_decode at B=8 and B=1 (rows of a [1,1,B,4608] L1 tensor)
    try:
        for B in (1, 8, 32):
            NH, NKV, HD = 16, 1, 256
            x = torch.randn(1, 1, B, (NH + 2 * NKV) * HD).to(torch.bfloat16)
            xt = _rep(mesh, x, ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                xt, num_heads=NH, num_kv_heads=NKV, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
            )
            results[f"heads_decode_B{B}"] = f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
            for t in (q, k, v, xt):
                ttnn.deallocate(t)
    except Exception as e:  # noqa: BLE001
        results["heads_decode"] = f"FAIL {str(e)[:300]}"

    for k_, v_ in results.items():
        logger.info(f"PROBE {k_}: {v_}")
    print("PROBE_RESULTS " + " | ".join(f"{k_}: {v_}" for k_, v_ in results.items()))
