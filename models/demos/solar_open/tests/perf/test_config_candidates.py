# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device validation of the explicit program configs of the production call sites against a torch fp32 reference.

Runs on ONE device (1x1 mesh) with random weights at the real Solar-Open TP=8 shapes and compares each explicit
config against the auto / phase-1 config it replaced: numerics (PCC and max abs error vs the fp32 matmul of the
device-rounded operands, plus PCC / max abs diff between the two device results) and host wall time (synchronised,
REPS runs; the difference between two configs is the kernel-time difference since the launch overhead is the same).
The candidate must not be worse than the auto config by more than 2e-4 in PCC vs the reference. Results are logged
and, with ``SOLAR_OPEN_PERF_OUT=<json>``, written to that file.

  * ``dense_down``: ``ProgramConfig.get_dense_down_config`` (1D mcast, 8x8, in0_block_w = Kt) for the dense prefill
    down bmm ``[1, E, S, Ip] x [1, E, Ip, H]`` vs ``core_grid=`` auto (which picks in0_block_w = 1) -- call sites
    ``experts/prefill.py::_dense_tail`` and the cold down of ``_sorted_moe_forward`` (perf-p0 lever e).
  * ``qkv``: ``SolarOpenAttentionProgramConfig`` decode qkv (8x5) through ``nlp_create_qkv_heads_decode`` vs the auto
    matmul, in four variants: in0_block_w 16 WITHOUT a compute config (what ttnn then runs: LoFi -- the phase-2 profile
    stage's candidate), 16 and 8 with the auto fidelity restated (HiFi2, ``get_decode_qkv_compute_config``) and 16
    with fp32 destination accumulation -- call site ``attention/decode.py`` (lever d).
  * ``o_proj``: decode out projection (8x8, per_core_N 2, in0_block_w 4, out_subblock_w 2) on the WIDTH-sharded
    ``nlp_concat_heads_decode`` output vs the auto linear -- not adopted (no gain in the production layout).
  * ``o_proj_8x8_interleaved``: the phase-3e B1 lever (``SOLAR_OPEN_ATTENTION_OUT_GRID=8x8``): the same (8x8) config
    from an L1-INTERLEAVED in0 (``sharded_to_interleaved`` moved in front of the matmul) with the HiFi2 compute config
    restated, as a two-op chain vs the production auto linear + reshard chain -- PCC vs the fp32 reference (floor:
    not worse than auto by 2e-4) and the chain wall times -- call site ``attention/decode.py::decode_output_projection``.
  * ``shared``: the shared expert's gate (= up) and down linears with ``shared_expert_program_configs`` vs auto at
    M = 1 / 32 / 128 rows (lever a).
  * ``router``: the router linear (bf16 x bf16 -> fp32, HiFi4, fp32 acc) with ``router_linear_program_config`` vs
    auto at M = 1 / 32 / 128 (lever c).
  * ``rms_norm``: the width-sharded decode norm of ``decode_norm_sharded_configs`` (incl. its two reshards) vs the
    default single-core kernel and a torch fp32 RMSNorm at M = 1 / 32 (lever b), then the ``RMSNorm`` module itself
    (``sharded_decode`` on vs off) on the decode shapes, the component test's HF-shaped ``[32, 1, H]`` input and a
    128-row prefill input: the gate ``decode_norm_applies`` must shard exactly the single-tile-row shapes.
  * ``egp``: the phase-3b expert-group decode sparse_matmul configs of ``SolarOpenProgramConfig`` (gate|up 11 groups on
    11x10, batched down 11 groups x 16 tiles on 11x8, the b1 indexed gate|up) vs the legacy configs of the same tree
    (``DECODE_EGP_LEGACY`` = ``SOLAR_OPEN_DECODE_EGP=off``) at union sizes 8 / 72 / 128: the results must be IDENTICAL
    (max abs diff 0, the kernels compute the same per-tile math) -- call sites ``experts/decode.py``.
  * ``p3c_down_indexed``: the phase-3c grid of the b1 indexed compact-A down (``decode_down_indexed_cores`` 8x8 x 2
    tiles, out_subblock_w 2) vs the phase-3b one (the single-user 8x4 x 4 tiles, ``SOLAR_OPEN_DECODE_EGP=p3b``) on the
    same compact bfp8 GLU rows and top-8 ids: IDENTICAL required -- call site ``experts/decode.py::_decode_forward_indexed``.

    SOLAR_OPEN_PERF_OUT=/path/cand.json pytest models/demos/solar_open/tests/perf/test_config_candidates.py \
        -k 1x1 -x -p no:cacheprovider
"""

import json
import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.solar_open.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.solar_open.tt.attention_configs import SolarOpenAttentionProgramConfig
from models.demos.solar_open.tt.expert_configs import DECODE_EGP_LEGACY, SolarOpenProgramConfig, decode_egp_overrides
from models.demos.solar_open.tt.experts.prefill import _DENSE_COMPUTE_KERNEL_CONFIG, _dense_core_grid
from models.demos.solar_open.tt.rms_norm import RMSNorm, decode_norm_applies, decode_norm_sharded_configs
from models.demos.solar_open.tt.shared_expert import shared_expert_program_configs
from models.demos.solar_open.tt.topk import router_linear_program_config

PERF_OUT = os.getenv("SOLAR_OPEN_PERF_OUT", "")
REPS = int(os.getenv("SOLAR_OPEN_PERF_REPS", "10"))
H, IP, E = 4096, 160, 128
QKV_N, O_K = 1280, 1024  # per device: 8 q heads + 1 k + 1 v head of 128; o_proj K = 8 heads x 128
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 8, 1, 128
EPS = 1e-5


def _time(device, fn, reps=REPS):
    out = fn()
    ttnn.synchronize_device(device)
    out.deallocate(True)
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        walls.append((time.perf_counter() - t0) * 1e3)
        out.deallocate(True)
    return {"wall_ms_min": min(walls), "wall_ms_mean": sum(walls) / len(walls)}


def _compare(name, auto, cand, results, reference=None, tolerance=2e-4):
    """Numerics of the candidate vs the auto config on identical inputs. With a torch ``reference`` (fp32 matmul of the
    device-rounded operands) both device results are scored against it and the candidate must not be worse than the
    auto config by more than ``tolerance`` in PCC (two device configs differ from each other by their accumulation
    order / spill format, so their mutual PCC alone does not say which one is closer to the exact result)."""
    auto_t = ttnn.to_torch(auto).float().reshape(reference.shape if reference is not None else -1)
    cand_t = ttnn.to_torch(cand).float().reshape(auto_t.shape)
    _, pcc_mutual = comp_pcc(auto_t, cand_t, 0.0)
    rec = {
        "pcc_cand_vs_auto": pcc_mutual,
        "max_abs_diff_cand_vs_auto": (auto_t - cand_t).abs().max().item(),
        "identical": bool(torch.equal(auto_t, cand_t)),
    }
    if reference is not None:
        _, rec["pcc_auto_vs_ref"] = comp_pcc(reference, auto_t, 0.0)
        _, rec["pcc_cand_vs_ref"] = comp_pcc(reference, cand_t, 0.0)
        rec["max_abs_err_auto"] = (reference - auto_t).abs().max().item()
        rec["max_abs_err_cand"] = (reference - cand_t).abs().max().item()
    results[name] = rec
    logger.info(f"[cand {name}] {json.dumps(rec)}")
    if reference is not None:
        assert (
            rec["pcc_cand_vs_ref"] >= rec["pcc_auto_vs_ref"] - tolerance
        ), f"{name}: candidate PCC vs reference {rec['pcc_cand_vs_ref']} worse than auto {rec['pcc_auto_vs_ref']}"


def _ab(device, name, auto, cand, results, reference):
    """Numerics + timing of ``cand`` vs ``auto`` (callables returning a device tensor) against ``reference``."""
    ref, got = auto(), cand()
    _compare(name, ref, got, results, reference=reference)
    ref.deallocate(True)
    got.deallocate(True)
    results[f"{name}_auto"] = _time(device, auto)
    results[f"{name}_cand"] = _time(device, cand)
    logger.info(
        f"[cand {name}] auto {results[f'{name}_auto']['wall_ms_min']:.3f} ms -> "
        f"cand {results[f'{name}_cand']['wall_ms_min']:.3f} ms (min of {REPS})"
    )


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(1, 1)])
def test_config_candidates(mesh_device, device_params, reset_seeds):
    device = mesh_device
    g = torch.Generator().manual_seed(11)
    results = {}
    grid = device.compute_with_storage_grid_size()
    logger.info(f"compute grid {grid.x}x{grid.y}")

    def up(t, dtype=ttnn.bfloat8_b, mem=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=mem)

    # ---------------- 1. dense prefill down bmm: auto core_grid vs get_dense_down_config ----------------
    pc = SolarOpenProgramConfig(dense_down_cores=(8, 8))  # explicit: independent of the shipped default (A/B switch)
    dense_grid = _dense_core_grid(device, pc.dense_grid_max_width)
    w_down = up(torch.randn(1, E, IP, H, generator=g) * 0.02)
    w_down_t = ttnn.to_torch(w_down).float()  # bfp8-rounded operand as the device sees it
    for s in (32, 128, 256):
        act = up(torch.randn(1, E, s, IP, generator=g))
        ref_down = torch.matmul(ttnn.to_torch(act).float(), w_down_t)
        cfg = pc.get_dense_down_config(s, H, IP)
        assert cfg is not None
        auto = lambda: ttnn.matmul(
            act,
            w_down,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=dense_grid,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
        cand = lambda: ttnn.matmul(
            act,
            w_down,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            program_config=cfg,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
        _ab(device, f"dense_down_s{s}", auto, cand, results, ref_down)
        act.deallocate(True)
    w_down.deallocate(True)

    # ---------------- 2. attention decode qkv projection through nlp_create_qkv_heads_decode ----------------
    batch = 32
    x = up(torch.randn(1, 1, batch, H, generator=g), dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
    w_qkv = up(torch.randn(1, 1, H, QKV_N, generator=g) * 0.02)
    apc0 = SolarOpenAttentionProgramConfig()
    batch_grid, _ = apc0.get_decode_user_grid(device, batch)
    heads_mem = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, HEAD_DIM),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    def qkv_auto():
        return ttnn.matmul(x, w_qkv, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

    def heads(fused):
        return ttnn.experimental.nlp_create_qkv_heads_decode(
            fused, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, memory_config=heads_mem
        )

    ref_qkv = torch.matmul(ttnn.to_torch(x).float(), ttnn.to_torch(w_qkv).float())
    variants = [
        ("bw16_lofi", 16, False, False),  # program config without compute config -> ttnn falls back to LoFi
        ("bw16_hifi2", 16, True, False),
        ("bw8_hifi2", 8, True, False),
        ("bw16_hifi2_fp32acc", 16, True, True),
    ]
    for tag, bw, with_compute, fp32_acc in variants:
        apc = SolarOpenAttentionProgramConfig(
            decode_qkv_cores=(8, 5), decode_qkv_in0_block_w=bw, decode_qkv_fp32_dest_acc=fp32_acc
        )
        qkv_cfg = apc.get_decode_qkv_config(batch, QKV_N, H)
        compute = apc.get_decode_qkv_compute_config(device.arch()) if with_compute else None
        logger.info(f"qkv {tag}: {qkv_cfg} compute {compute}")

        def qkv_cand(qkv_cfg=qkv_cfg, compute=compute):
            return ttnn.matmul(
                x,
                w_qkv,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=qkv_cfg,
                compute_kernel_config=compute,
            )

        ref_f, got_f = qkv_auto(), qkv_cand()
        if tag == "bw16_lofi":
            logger.info(f"qkv output shard spec auto {ref_f.memory_config()} cand {got_f.memory_config()}")
        _compare(f"qkv_{tag}", ref_f, got_f, results, reference=ref_qkv, tolerance=2e-4 if tag != "bw16_lofi" else 1.0)
        for name, r, c in zip(("q", "k", "v"), heads(ref_f), heads(got_f)):
            _compare(f"qkv_{tag}_heads_{name}", r, c, results)  # the head split must accept the candidate's layout
            r.deallocate(True)
            c.deallocate(True)
        ref_f.deallocate(True)
        got_f.deallocate(True)
        if tag == "bw16_lofi":
            results["qkv_auto"] = _time(device, qkv_auto)
        results[f"qkv_{tag}_cand"] = _time(device, qkv_cand)
        logger.info(
            f"[cand qkv {tag}] auto {results['qkv_auto']['wall_ms_min']:.3f} ms -> "
            f"cand {results[f'qkv_{tag}_cand']['wall_ms_min']:.3f} ms"
        )
    x.deallocate(True)
    w_qkv.deallocate(True)

    # ---------------- 3. attention decode o_proj on the width-sharded concat_heads output (not adopted) ----------------
    apc_o = SolarOpenAttentionProgramConfig(
        decode_out_cores=(8, 8), decode_out_in0_block_w=4, decode_out_out_subblock_w=2
    )
    w_o = up(torch.randn(1, 1, O_K, H, generator=g) * 0.02)
    concat_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(NUM_HEADS, grid, row_wise=True),
            (batch, HEAD_DIM),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    sdpa_il = up(torch.randn(1, 1, batch, O_K, generator=g), dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
    sdpa_out = ttnn.interleaved_to_sharded(sdpa_il, concat_mem)  # == nlp_concat_heads_decode's output layout
    sdpa_il.deallocate(True)
    out_cfg = apc_o.get_decode_out_config(batch, H, O_K)
    logger.info(f"o_proj candidate config: {out_cfg}")

    def o_auto():
        return ttnn.linear(sdpa_out, w_o, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

    def o_cand():
        return ttnn.linear(
            sdpa_out,
            w_o,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=out_cfg,
            compute_kernel_config=apc_o.get_decode_qkv_compute_config(device.arch()),  # HiFi2 like the auto config
        )

    ref_oproj = torch.matmul(ttnn.to_torch(sdpa_out).float(), ttnn.to_torch(w_o).float())
    ref_o, got_o = o_auto(), o_cand()
    _compare("o_proj", ref_o, got_o, results, reference=ref_oproj)
    # the production chain continues with to_memory_config(L1 interleaved) + typecast(bfp8): must accept both outputs
    for t in (ref_o, got_o):
        il = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
        tc = ttnn.typecast(il, ttnn.bfloat8_b)
        il.deallocate(True)
        tc.deallocate(True)
        t.deallocate(True)
    results["o_proj_auto"] = _time(device, o_auto)
    results["o_proj_cand"] = _time(device, o_cand)
    logger.info(
        f"[cand o_proj] auto {results['o_proj_auto']['wall_ms_min']:.3f} ms -> cand {results['o_proj_cand']['wall_ms_min']:.3f} ms"
    )

    # ---------------- 3b. phase 3e B1 slice 2: o_proj (8x8) from an INTERLEAVED in0, HiFi2 restated ----------------
    # Production legacy chain = auto linear on the width-sharded concat output + to_memory_config(L1 interleaved);
    # candidate chain = sharded_to_interleaved first, then the explicit (8, 8) config (per_core_N 2, in0_block_w 4,
    # out_subblock_w 2) straight into an L1-interleaved output. Both chains end in the same layout, so their wall
    # times are the per-layer difference of the lever (design_decode_levers.md 3.3: 24.0 -> ~12.9 us incl. reshard).
    apc_oi = SolarOpenAttentionProgramConfig(decode_out_cores=(8, 8))  # the SOLAR_OPEN_ATTENTION_OUT_GRID=8x8 values
    oi_cfg = apc_oi.get_decode_out_config(batch, H, O_K)
    oi_compute = apc_oi.get_decode_out_compute_config(device.arch())
    logger.info(f"o_proj interleaved-in0 candidate config: {oi_cfg} compute {oi_compute}")

    def o_auto_chain():
        t = ttnn.linear(sdpa_out, w_o, dtype=ttnn.bfloat16, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
        il = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
        t.deallocate(True)
        return il

    def o_interleaved_chain():
        il = ttnn.sharded_to_interleaved(sdpa_out, ttnn.L1_MEMORY_CONFIG)
        t = ttnn.linear(
            il,
            w_o,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=oi_cfg,
            compute_kernel_config=oi_compute,
        )
        il.deallocate(True)
        return t

    ref_oi, got_oi = o_auto_chain(), o_interleaved_chain()
    assert got_oi.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    _compare("o_proj_8x8_interleaved", ref_oi, got_oi, results, reference=ref_oproj)
    ref_oi.deallocate(True)
    got_oi.deallocate(True)
    results["o_proj_8x8_interleaved_auto"] = _time(device, o_auto_chain)
    results["o_proj_8x8_interleaved_cand"] = _time(device, o_interleaved_chain)
    logger.info(
        f"[cand o_proj_8x8_interleaved] auto chain {results['o_proj_8x8_interleaved_auto']['wall_ms_min']:.3f} ms -> "
        f"interleaved-in0 chain {results['o_proj_8x8_interleaved_cand']['wall_ms_min']:.3f} ms"
    )
    sdpa_out.deallocate(True)
    w_o.deallocate(True)

    # ---------------- 4. shared expert linears (bf16 x bfp8, HiFi2): auto vs shared_expert_program_configs ----------------
    shared_compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    w_sg = up(torch.randn(1, 1, H, IP, generator=g) * 0.02)
    w_sd = up(torch.randn(1, 1, IP, H, generator=g) * 0.02)
    w_sg_t, w_sd_t = ttnn.to_torch(w_sg).float(), ttnn.to_torch(w_sd).float()

    def lin(a, w, pc=None):
        return lambda: ttnn.linear(
            a,
            w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=shared_compute,
            program_config=pc,
        )

    for m in (1, 32, 128):
        xs = up(torch.randn(1, 1, m, H, generator=g), dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
        gate_up_cfg, down_cfg = shared_expert_program_configs(m, H, IP, grid, max_rows=128)
        assert gate_up_cfg is not None and down_cfg is not None, (m, gate_up_cfg, down_cfg)
        logger.info(f"shared m={m}: gate/up {gate_up_cfg} down {down_cfg}")
        _ab(
            device,
            f"shared_gate_m{m}",
            lin(xs, w_sg),
            lin(xs, w_sg, gate_up_cfg),
            results,
            ttnn.to_torch(xs).float() @ w_sg_t,
        )
        # down input: a GLU-like bf16 activation of the real shape [1, 1, m, IP]
        a = up(torch.randn(1, 1, m, IP, generator=g), dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
        _ab(
            device,
            f"shared_down_m{m}",
            lin(a, w_sd),
            lin(a, w_sd, down_cfg),
            results,
            ttnn.to_torch(a).float() @ w_sd_t,
        )
        a.deallocate(True)
        xs.deallocate(True)
    w_sg.deallocate(True)
    w_sd.deallocate(True)

    # ---------------- 5. router linear (bf16 x bf16 -> fp32, HiFi4, fp32 acc): auto vs router_linear_program_config ----------------
    router_compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    w_r = up(torch.randn(1, 1, H, E, generator=g) * 0.04, dtype=ttnn.bfloat16)
    w_r_t = ttnn.to_torch(w_r).float().reshape(H, E)

    def router(a, pc=None):
        return lambda: ttnn.linear(
            a,
            w_r,
            dtype=ttnn.float32,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=router_compute,
            program_config=pc,
        )

    for m in (1, 32, 128):
        x4 = up(torch.randn(1, 1, m, H, generator=g), dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
        xr = ttnn.reshape(x4, (-1, H))  # the router's [tokens, hidden] view
        cfg = router_linear_program_config(m, E, H, grid, max_tokens=128)
        assert cfg is not None, m
        logger.info(f"router m={m}: {cfg}")
        _ab(device, f"router_m{m}", router(xr), router(xr, cfg), results, ttnn.to_torch(xr).float() @ w_r_t)
        x4.deallocate(True)
    w_r.deallocate(True)

    # ---------------- 6. rms_norm: default single-core kernel vs the width-sharded decode config (+ 2 reshards) ----------------
    mem, npc = decode_norm_sharded_configs(H, grid, cores=(8, 4))
    assert mem is not None and npc is not None
    w_norm_t = torch.randn(H, generator=g) * 0.1 + 1.0
    w_norm = up(w_norm_t.reshape(1, 1, H // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    w_norm_rounded = ttnn.to_torch(w_norm).float().reshape(1, 1, 1, H)
    for m in (1, 32):
        xn = up(torch.randn(1, 1, m, H, generator=g), dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
        xt = ttnn.to_torch(xn).float()
        ref_norm = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + EPS) * w_norm_rounded

        def norm_auto():
            return ttnn.rms_norm(xn, weight=w_norm, epsilon=EPS)

        def norm_cand():
            xs = ttnn.interleaved_to_sharded(xn, mem)
            ys = ttnn.rms_norm(xs, weight=w_norm, epsilon=EPS, program_config=npc, memory_config=mem)
            xs.deallocate(True)
            y = ttnn.sharded_to_interleaved(ys, ttnn.L1_MEMORY_CONFIG)
            ys.deallocate(True)
            return y

        _ab(device, f"rms_norm_m{m}", norm_auto, norm_cand, results, ref_norm)
        xn.deallocate(True)
    w_norm.deallocate(True)

    # ---------------- 7. the RMSNorm module's own gate (decode_norm_applies) on the production and the test shapes ----------------
    # The worktree ladder failed on the component test's HF-shaped [32, 1, H] decode input (32 one-row batches pad to
    # 32 tile rows: a [32, 128] width shard cannot cover the physical height 1024). The module must route only the
    # single-tile-row shapes through the sharded kernel and every other shape through the default one, with the
    # same numerics as the sharded_decode=False instance on all of them.
    norm_cfg = SimpleNamespace(rms_norm_eps=EPS, hidden_size=H)
    norm_sd = {"weight": w_norm_t.to(torch.bfloat16)}
    norm_on = RMSNorm(device, norm_cfg, norm_sd, sharded_decode=True)
    norm_off = RMSNorm(device, norm_cfg, norm_sd, sharded_decode=False)
    assert norm_on.sharded_program_config is not None and norm_off.sharded_program_config is None
    for shape, expect_sharded in (
        ((1, 1, 1, H), True),
        ((1, 1, 32, H), True),
        ((1, 1, 16, H), True),
        ((32, 1, H), False),  # the component test's decode_b32 input (physical height 1024)
        ((1, 32, 1, H), False),
        ((1, 1, 128, H), False),
    ):
        xt = torch.randn(*shape, generator=g)
        for mem in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
            xn = up(xt, dtype=ttnn.bfloat16, mem=mem)
            assert decode_norm_applies(xn, H) is expect_sharded, (shape, mem, tuple(xn.padded_shape))
            xr = ttnn.to_torch(xn).float()
            ref_norm = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + EPS) * w_norm_rounded.reshape(H)
            y_on, y_off = norm_on(xn), norm_off(xn)
            if expect_sharded:  # the sharded path must hand back the caller's interleaved memory config
                assert y_on.memory_config() == xn.memory_config(), (shape, y_on.memory_config())
            tag = f"rms_norm_module_{'x'.join(map(str, shape))}_{'l1' if mem == ttnn.L1_MEMORY_CONFIG else 'dram'}"
            _compare(tag, y_off, y_on, results, reference=ref_norm, tolerance=2e-5)
            y_on.deallocate(True)
            y_off.deallocate(True)
            xn.deallocate(True)

    # ---------------- 8. expert groups (phase 3b): the shipped EGP decode configs vs the legacy ones, identical ----------------
    # Same random bfp8 weights, the same union masks; EGP (expert_groups 11) vs legacy (None) must agree bit for bit.
    pc_egp, pc_legacy = SolarOpenProgramConfig(), SolarOpenProgramConfig(**DECODE_EGP_LEGACY)
    assert pc_egp.decode_gate_up_expert_groups == 11 and pc_legacy.decode_gate_up_expert_groups is None
    tile = ttnn.Tile([32, 32])
    w_gu = up(torch.randn(1, E, H, 2 * IP, generator=g) * 0.02)
    w_dn = up(torch.randn(1, E, IP, H, generator=g) * 0.02)
    w_gu_t, w_dn_t = ttnn.to_torch(w_gu).float(), ttnn.to_torch(w_dn).float()
    x32 = up(torch.randn(1, 1, 32, H, generator=g), dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
    x1 = up(torch.randn(1, 1, 1, H, generator=g), dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
    x32_t, x1_t = ttnn.to_torch(x32).float(), ttnn.to_torch(x1).float()

    def sparse_gate_up(x, mask, cfg, users):
        return lambda: ttnn.sparse_matmul(
            x,
            w_gu,
            sparsity=mask,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=tile,
            program_config=cfg.get_decode_gate_up_config(users, 2 * IP, k=H).program_config,
            expert_groups=cfg.get_decode_gate_up_config(users, 2 * IP, k=H).expert_groups,
            dtype=ttnn.bfloat8_b,
        )

    def sparse_down(act, mask, cfg, users):
        return lambda: ttnn.sparse_matmul(
            act,
            w_dn,
            sparsity=mask,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=tile,
            is_input_a_sparse=True,
            program_config=cfg.get_decode_down_config(users, H, k=IP).program_config,
            expert_groups=cfg.get_decode_down_config(users, H, k=IP).expert_groups,
            dtype=ttnn.bfloat8_b,
        )

    for nnz in (8, 72, 128):
        idx = torch.randperm(E, generator=g)[:nnz]
        mask_t = torch.zeros(1, 1, 1, E)
        mask_t[..., idx] = 0.125
        mask = up(mask_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mem=ttnn.L1_MEMORY_CONFIG)
        active = torch.zeros(E)
        active[idx] = 1.0
        # gate|up at 32 users (the b32 union) and at 1 user (the scan path): [1, E, 32, 2Ip], inactive experts all zero
        for users, x, x_t in ((32, x32, x32_t), (1, x1, x1_t)):
            ref = torch.matmul(x_t.reshape(1, 1, -1, H).expand(1, E, -1, H), w_gu_t) * active.reshape(1, E, 1, 1)
            _ab(
                device,
                f"egp_gate_up_m{users}_nnz{nnz}",
                sparse_gate_up(x, mask, pc_legacy, users),
                sparse_gate_up(x, mask, pc_egp, users),
                results,
                ref.reshape(1, 1, 1, E, -1, 2 * IP),
            )
            assert results[f"egp_gate_up_m{users}_nnz{nnz}"][
                "identical"
            ], f"EGP gate|up differs at nnz {nnz}, M {users}"
        # batched down: A [1, E, 32, Ip] (zero rows for the inactive experts) at 32 and 16 users (both the EGP grid)
        act_t = torch.randn(1, E, 32, IP, generator=g) * active.reshape(1, E, 1, 1)
        act = up(act_t, mem=ttnn.L1_MEMORY_CONFIG)
        ref_dn = torch.matmul(ttnn.to_torch(act).float(), w_dn_t)
        for users in (32, 16):
            _ab(
                device,
                f"egp_down_u{users}_nnz{nnz}",
                sparse_down(act, mask, pc_legacy, users),
                sparse_down(act, mask, pc_egp, users),
                results,
                ref_dn,
            )
            assert results[f"egp_down_u{users}_nnz{nnz}"]["identical"], f"EGP down differs at nnz {nnz}, users {users}"
        act.deallocate(True)
        mask.deallocate(True)
    # the b1 indexed gate|up (k = 8 top-k ids, compact [1, k, 1, 2Ip] output): EGP vs legacy, identical
    ids = torch.randperm(E, generator=g)[:8]
    indices = up(ids.reshape(1, 1, 1, 8).to(torch.int32), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
    placeholder = up(torch.ones(1, 1, 1, E), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def indexed_gate_up(cfg):
        c = cfg.get_decode_gate_up_config(1, 2 * IP, k=H)
        return lambda: ttnn.sparse_matmul(
            x1,
            w_gu,
            sparsity=placeholder,
            indices=indices,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=tile,
            program_config=c.program_config,
            expert_groups=c.expert_groups,
            dtype=ttnn.bfloat8_b,
        )

    ref_idx = torch.matmul(x1_t.reshape(1, 1, 1, H).expand(1, 8, 1, H), w_gu_t[:, ids])
    _ab(
        device,
        "egp_gate_up_indexed_k8",
        indexed_gate_up(pc_legacy),
        indexed_gate_up(pc_egp),
        results,
        ref_idx.reshape(1, 1, 1, 8, 1, 2 * IP),
    )
    assert results["egp_gate_up_indexed_k8"]["identical"]

    # ---------------- 9. phase 3c: the indexed compact-A down on 8x8 x 2 tiles vs the 8x4 x 4 tiles grid, identical ----------------
    # The shipped b1 down: A = the k = 8 compact bfp8 GLU rows [1, 8, 1, Ip], both operands sparse, output [1, 8, 1, H].
    pc_p3b = SolarOpenProgramConfig(**decode_egp_overrides("p3b"))
    assert pc_p3b.decode_down_indexed_cores is None and pc_egp.decode_down_indexed_cores == (8, 8)
    act_c = up(torch.randn(1, 8, 1, IP, generator=g), mem=ttnn.L1_MEMORY_CONFIG)

    def indexed_down(cfg):
        c = cfg.get_decode_down_config(1, H, k=IP, indexed=True)
        return lambda: ttnn.sparse_matmul(
            act_c,
            w_dn,
            sparsity=placeholder,
            indices=indices,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=tile,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=c.program_config,
            expert_groups=c.expert_groups,
            dtype=ttnn.bfloat8_b,
        )

    p3b_key = pc_p3b.get_decode_down_config(1, H, k=IP, indexed=True).program_config
    p3c_key = pc_egp.get_decode_down_config(1, H, k=IP, indexed=True).program_config
    assert (p3b_key.compute_with_storage_grid_size.x, p3b_key.compute_with_storage_grid_size.y, p3b_key.per_core_N) == (
        8,
        4,
        4,
    )
    assert (p3c_key.compute_with_storage_grid_size.x, p3c_key.compute_with_storage_grid_size.y, p3c_key.per_core_N) == (
        8,
        8,
        2,
    )
    ref_dn_idx = torch.matmul(ttnn.to_torch(act_c).float(), w_dn_t[:, ids])  # [1, 8, 1, H]
    _ab(device, "p3c_down_indexed_k8", indexed_down(pc_p3b), indexed_down(pc_egp), results, ref_dn_idx)
    assert results["p3c_down_indexed_k8"][
        "identical"
    ], "the 8x8 x 2 tiles indexed down differs from the 8x4 x 4 tiles one"
    for t in (w_gu, w_dn, x32, x1, indices, placeholder, act_c):
        t.deallocate(True)

    if PERF_OUT:
        data = json.loads(open(PERF_OUT).read()) if os.path.isfile(PERF_OUT) else {}
        data["config_candidates"] = results
        with open(PERF_OUT, "w") as f:
            json.dump(data, f, indent=2)
    logger.info(json.dumps(results, indent=1))
