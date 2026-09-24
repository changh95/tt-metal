# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: row-position dependence of the ops the R>32 verify body adds over the decode body (no model weights).

For each candidate op the input holds IDENTICAL rows at different grid positions (row 8..15 == row 32..39 == row 56..63
of a 64-row block); the op is exact iff those output rows are bitwise equal, and iff they equal the M=32 (block_h 1 /
per_core_M 1) result for the same rows. PROBE lines report both.
"""
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt import tp_common as tpc


def _rep(mesh, t, dtype, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, memory_config=mem, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _t0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _dup_rows(R, W, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(R, W, generator=g)
    x[32:40] = x[8:16]
    x[56:64] = x[8:16]
    return x.to(torch.bfloat16)


def _same(out, groups=((8, 16), (32, 40), (56, 64))):
    a = out[groups[0][0] : groups[0][1]]
    return all(torch.equal(a, out[lo:hi]) for lo, hi in groups[1:])


@parametrize_mesh_tp()
def test_rowpos_probes(mesh_device):
    mesh = mesh_device
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

    args = Qwen36ModelArgs(mesh_device=mesh, max_batch_size=32, max_seq_len=4096)
    dim = args.dim
    res = {}
    R = 64

    # --- sharded RMSNorm, block_h = 2 vs 1 on the decode attn norm grid ---
    for name, grid in (("attn", args.attn_input_grid), ("lm_head", args.lm_head_core_grid)):
        x = _dup_rows(R, dim)
        w = torch.randn(1, 1, 1, dim).to(torch.bfloat16)
        wt = _rep(mesh, w, ttnn.bfloat16)
        block_w = dim // grid.num_cores // 32
        sub_w = next(s for s in (4, 3, 2, 1) if block_w % s == 0)

        def run(rows, bh):
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[grid.x, grid.y],
                subblock_w=sub_w,
                block_h=bh,
                block_w=block_w,
                inplace=False,
            )
            mc = ttnn.create_sharded_memory_config(
                (rows.shape[0], dim // grid.num_cores),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            xt = ttnn.to_memory_config(_rep(mesh, rows.reshape(1, 1, rows.shape[0], dim), ttnn.bfloat16), mc)
            y = ttnn.rms_norm(
                xt,
                epsilon=1e-6,
                weight=wt,
                program_config=pc,
                memory_config=mc,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
                ),
            )
            return _t0(ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)).reshape(rows.shape[0], dim)

        y64 = run(x, 2)
        y32 = run(x[:32], 1)
        res[f"rmsnorm_{name}_bh2_rows_equal"] = _same(y64)
        res[f"rmsnorm_{name}_bh2_vs_bh1"] = torch.equal(y64[:32], y32)

    # --- 1D mcast matmul at per_core_M = 2 (small_m_progcfg) vs the decode config (per_core_M 1) ---
    gw = args.decode_grid_w
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    for name, K, N, wdt, act, dec_cfg in (
        ("mlp_w1", dim, args.hidden_dim // 4, ttnn.bfloat4_b, ttnn.UnaryOpType.SILU, args.mlp_w1_decode_1d_progcfg),
        ("mlp_w2", args.hidden_dim // 4, dim, ttnn.bfloat8_b, None, args.mlp_w2_decode_1d_progcfg),
        ("attn_qkv", dim, args.attn_qkv_fused_dim_tp, ttnn.bfloat8_b, None, args.attn_qkv_decode_1d_progcfg),
        (
            "gdn_qkvz",
            dim,
            args.gdn_qkvzab_dim_tp + (32 - args.gdn_qkvzab_dim_tp % 32) % 32,
            ttnn.bfloat8_b,
            None,
            args.gdn_qkvz_decode_1d_progcfg,
        ),
        ("wo", args.attn_out_dim_tp, dim, ttnn.bfloat8_b, None, args.attn_wo_decode_1d_progcfg),
    ):
        x = _dup_rows(R, K, seed=1)
        wgt = _rep(mesh, torch.randn(K, N).to(torch.bfloat16), wdt)
        ck = cfg if name.startswith("mlp") else tpc.COMPUTE_HIFI2
        pc64 = tpc.small_m_progcfg(R, K, N, fused_activation=act, grid_w=gw)
        y64 = _t0(
            ttnn.linear(
                _rep(mesh, x.reshape(1, 1, R, K), ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG),
                wgt,
                compute_kernel_config=ck,
                program_config=pc64,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        ).reshape(R, N)
        y32 = _t0(
            ttnn.linear(
                _rep(mesh, x[:32].reshape(1, 1, 32, K), ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG),
                wgt,
                compute_kernel_config=ck,
                program_config=dec_cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        ).reshape(32, N)
        res[f"mm1d_{name}_M64_rows_equal"] = _same(y64)
        res[f"mm1d_{name}_M64_vs_decodecfg_M32"] = torch.equal(y64[:32], y32)
        # the DRAM-sharded decode kernels (served default QWEN36_DECODE_DRAM_SHARDED=all) vs 1D at M=32: informational
        if name == "mlp_w1":
            res[f"mm1d_{name}_M64_vs_decodecfg_M32_maxabs"] = float((y64[:32].float() - y32.float()).abs().max())

    # --- LM head: ttnn.linear auto config at M=64 vs M=32 on a vocab shard ---
    V = args.vocab_size // 4
    x = _dup_rows(R, dim, seed=2)
    wgt = _rep(mesh, torch.randn(dim, V).to(torch.bfloat16), ttnn.bfloat8_b)
    y64 = _t0(ttnn.linear(_rep(mesh, x.reshape(1, 1, R, dim), ttnn.bfloat16), wgt)).reshape(R, V)
    y32 = _t0(ttnn.linear(_rep(mesh, x[:32].reshape(1, 1, 32, dim), ttnn.bfloat16), wgt)).reshape(32, V)
    res["lm_head_auto_M64_rows_equal"] = _same(y64)
    res["lm_head_auto_M64_vs_M32"] = torch.equal(y64[:32], y32)

    # --- LM head auto config and the 1D configs at M=128 / 256 vs M=32 (rows 8..15 duplicated into every 32-row block)
    for M in (128, 256):
        xx = torch.randn(M, dim).to(torch.bfloat16)
        for b in range(1, M // 32):
            xx[b * 32 + 8 : b * 32 + 16] = xx[8:16]
        y = _t0(ttnn.linear(_rep(mesh, xx.reshape(1, 1, M, dim), ttnn.bfloat16), wgt)).reshape(M, V)
        y32 = _t0(ttnn.linear(_rep(mesh, xx[:32].reshape(1, 1, 32, dim), ttnn.bfloat16), wgt)).reshape(32, V)
        res[f"lm_head_auto_M{M}_rows_equal"] = all(
            torch.equal(y[b * 32 + 8 : b * 32 + 16], y[8:16]) for b in range(1, M // 32)
        )
        res[f"lm_head_auto_M{M}_vs_M32"] = torch.equal(y[:32], y32)
        for name, K, N, wdt, act, dec_cfg in (
            ("gdn_qkvz", dim, 4128, ttnn.bfloat8_b, None, args.gdn_qkvz_decode_1d_progcfg),
            ("mlp_w1", dim, args.hidden_dim // 4, ttnn.bfloat4_b, ttnn.UnaryOpType.SILU, args.mlp_w1_decode_1d_progcfg),
        ):
            xk = torch.randn(M, K).to(torch.bfloat16)
            for b in range(1, M // 32):
                xk[b * 32 + 8 : b * 32 + 16] = xk[8:16]
            wk = _rep(mesh, torch.randn(K, N).to(torch.bfloat16), wdt)
            ck = cfg if name.startswith("mlp") else tpc.COMPUTE_HIFI2
            pc = tpc.small_m_progcfg(M, K, N, fused_activation=act, grid_w=gw)
            ym = _t0(
                ttnn.linear(
                    _rep(mesh, xk.reshape(1, 1, M, K), ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG),
                    wk,
                    compute_kernel_config=ck,
                    program_config=pc,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            ).reshape(M, N)
            y3 = _t0(
                ttnn.linear(
                    _rep(mesh, xk[:32].reshape(1, 1, 32, K), ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG),
                    wk,
                    compute_kernel_config=ck,
                    program_config=dec_cfg,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            ).reshape(32, N)
            res[f"mm1d_{name}_M{M}_rows_equal"] = all(
                torch.equal(ym[b * 32 + 8 : b * 32 + 16], ym[8:16]) for b in range(1, M // 32)
            )
            res[f"mm1d_{name}_M{M}_vs_decodecfg_M32"] = torch.equal(ym[:32], y3)

    # --- to_layout + argmax / max at 64 rows vs 32 ---
    lg = _dup_rows(R, V, seed=3)
    lt = _rep(mesh, lg.reshape(1, 1, R, V), ttnn.bfloat16)
    idx = _t0(ttnn.argmax(ttnn.to_layout(lt, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=False)).reshape(-1)[:R]
    mx = _t0(ttnn.max(lt, dim=-1)).reshape(-1)[:R]
    res["argmax_rows_equal"] = _same(idx.reshape(R, 1)) and _same(mx.reshape(R, 1))

    for k, v in res.items():
        logger.info(f"PROBE {k}: {v}")
    print("PROBE_RESULTS " + " | ".join(f"{k}={v}" for k, v in res.items()))
