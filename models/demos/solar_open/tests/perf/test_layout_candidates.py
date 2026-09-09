# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device validation of the perf-p1 LAYOUT levers of the expert paths against a torch fp32 reference (one device).

Runs on a 1x1 mesh with random bfloat8_b weights at the real Solar-Open TP=8 shapes (H = 4096, Ip = 160, E = 128) and
compares each new form against the phase-1 form it replaces: numerics (PCC vs the fp32 result of the device-rounded
operands, PCC / max abs diff between the two device results) and synchronised host wall time (REPS runs; the launch
overhead is shared, so the difference is the kernel-time difference). Results are logged and, with
``SOLAR_OPEN_PERF_OUT=<json>``, written to that file.

  * ``decode_routing``: batched decode, routing weights multiplied into the down INPUT [1, E, 32, Ip]
    (``experts/decode.py`` ROUTING_WEIGHTS_ON_DOWN_INPUT) vs into the down OUTPUT [1, E, 32, H] (phase 1), both
    followed by the down sparse_matmul over the union mask and fast_reduce_nc (lever a).
  * ``hot_group_n<hot>``: the sorted prefill path's hot group at 1024 tokens with 4 / 8 / 15 hot experts: per-expert
    ``ttnn.linear`` gate/up + K-concatenated down (``experts/prefill.py`` HOT_EXPERTS_PER_EXPERT_LINEAR +
    HOT_DOWN_KCONCAT, explicit 2D config and auto) vs repeat + batched matmul + batched down + fast_reduce_nc
    (phase 1), plus the per-expert linears + batched down + fast_reduce_nc variant (the shipped form) -- the timings
    feed ``_HOT_FIXED_MS`` / ``_HOT_PER_EXPERT_MS`` (lever b).
  * ``cold_index`` / ``hot_routing_rows``: the copy-free index / routing-row layouts of ELIDE_ROUTING_COPIES must be
    bit-identical to the phase-1 reshapes (lever c).

    SOLAR_OPEN_PERF_OUT=/path/cand.json pytest models/demos/solar_open/tests/perf/test_layout_candidates.py \
        -k 1x1 -x -p no:cacheprovider
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.solar_open.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.solar_open.tt.expert_configs import SolarOpenProgramConfig
from models.demos.solar_open.tt.experts.operations import apply_glu, reduce_experts
from models.demos.solar_open.tt.experts.prefill import (
    _DENSE_COMPUTE_KERNEL_CONFIG,
    _concat_expert_slices,
    _dense_core_grid,
    _expert_slice,
    _hot_down_kconcat_config,
    _hot_down_kconcat_matmul,
)

PERF_OUT = os.getenv("SOLAR_OPEN_PERF_OUT", "")
REPS = int(os.getenv("SOLAR_OPEN_PERF_REPS", "5"))
H, IP, E = 4096, 160, 128
SPLIT = 1024


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


def _compare(name, old, new, results, reference=None, tolerance=2e-4, shape=None):
    """Numerics of the new form vs the phase-1 form on identical inputs; with a torch ``reference`` (fp32 of the
    device-rounded operands) the new form must not be worse than the old one by more than ``tolerance`` in PCC."""
    old_t = ttnn.to_torch(old).float()
    new_t = ttnn.to_torch(new).float()
    if shape is not None:
        old_t, new_t = old_t.reshape(shape), new_t.reshape(shape)
    _, pcc_mutual = comp_pcc(old_t, new_t, 0.0)
    rec = {
        "pcc_new_vs_old": pcc_mutual,
        "max_abs_diff_new_vs_old": (old_t - new_t).abs().max().item(),
        "identical": bool(torch.equal(old_t, new_t)),
    }
    if reference is not None:
        ref = reference.reshape(old_t.shape)
        _, rec["pcc_old_vs_ref"] = comp_pcc(ref, old_t, 0.0)
        _, rec["pcc_new_vs_ref"] = comp_pcc(ref, new_t, 0.0)
        rec["max_abs_err_old"] = (ref - old_t).abs().max().item()
        rec["max_abs_err_new"] = (ref - new_t).abs().max().item()
    results[name] = rec
    logger.info(f"[cand {name}] {json.dumps(rec)}")
    if reference is not None:
        assert (
            rec["pcc_new_vs_ref"] >= rec["pcc_old_vs_ref"] - tolerance
        ), f"{name}: new form PCC vs reference {rec['pcc_new_vs_ref']} worse than the old form {rec['pcc_old_vs_ref']}"
    return rec


def _ab(device, name, old, new, results, reference=None, shape=None, tolerance=2e-4):
    """Numerics + timing of ``new`` vs ``old`` (callables returning a device tensor)."""
    a, b = old(), new()
    _compare(name, a, b, results, reference=reference, tolerance=tolerance, shape=shape)
    a.deallocate(True)
    b.deallocate(True)
    results[f"{name}_old"] = _time(device, old)
    results[f"{name}_new"] = _time(device, new)
    logger.info(
        f"[cand {name}] old {results[f'{name}_old']['wall_ms_min']:.3f} ms -> "
        f"new {results[f'{name}_new']['wall_ms_min']:.3f} ms (min of {REPS})"
    )


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(1, 1)])
def test_layout_candidates(mesh_device, device_params, reset_seeds):
    device = mesh_device
    g = torch.Generator().manual_seed(23)
    results = {}
    grid = device.compute_with_storage_grid_size()
    dense_grid = _dense_core_grid(device, SolarOpenProgramConfig().dense_grid_max_width)
    logger.info(f"compute grid {grid.x}x{grid.y}, dense grid {dense_grid}")

    def up(t, dtype=ttnn.bfloat8_b, mem=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=mem)

    # ---------------- 1. batched decode: routing weights on the down input vs on the down output ----------------
    pc = SolarOpenProgramConfig()
    users = 32
    active_pool = torch.randperm(E, generator=g)[:80]  # the users draw their top-8 from 80 experts: union ~ 70
    routing = torch.zeros(users, E)
    for u in range(users):
        picks = active_pool[torch.randperm(80, generator=g)[:8]]
        w = torch.rand(8, generator=g)
        routing[u, picks] = w / w.sum()
    union_mask = (routing.sum(0) > 0).float()
    logger.info(f"decode routing: union of experts {int(union_mask.sum())}")
    glu = torch.randn(1, E, users, IP, generator=g) * union_mask.reshape(1, E, 1, 1)  # inactive experts: 0 rows
    w_down = up(torch.randn(1, E, IP, H, generator=g) * 0.02)
    x_in = up(glu, mem=ttnn.L1_MEMORY_CONFIG)
    routing_tt = up(routing, dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
    sparsity = up(
        union_mask.reshape(1, 1, 1, E), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mem=ttnn.L1_MEMORY_CONFIG
    )
    x_t, w_t, r_t = ttnn.to_torch(x_in).float(), ttnn.to_torch(w_down).float(), ttnn.to_torch(routing_tt).float()
    ref_decode = torch.einsum("te,eth->th", r_t, torch.matmul(x_t[0], w_t[0]))  # sum_e r[t, e] (x_e @ W_e)[t]
    down_cfg = pc.get_decode_down_config(users, H, k=IP)
    tile = ttnn.Tile([32, 32])

    def weights_bcast():
        tw = ttnn.permute(routing_tt, (1, 0))
        return ttnn.reshape(tw, (1, E, users, 1))

    def down_of(x):
        return ttnn.sparse_matmul(
            x,
            w_down,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=tile,
            is_input_a_sparse=True,
            program_config=down_cfg.program_config,
            expert_groups=down_cfg.expert_groups,
            dtype=ttnn.bfloat8_b,
        )

    def decode_old():  # phase 1: down, then mul on [1, E, 32, H], then reduce
        x = ttnn.clone(x_in)
        tw = weights_bcast()
        d = down_of(x)
        x.deallocate(True)
        d = ttnn.mul(d, tw, output_tensor=d)
        tw.deallocate(True)
        out = ttnn.experimental.fast_reduce_nc(d, dims=[1], memory_config=ttnn.L1_MEMORY_CONFIG)
        d.deallocate(True)
        return out

    def decode_new():  # lever a: mul on [1, E, 32, Ip], then down, then reduce
        x = ttnn.clone(x_in)
        tw = weights_bcast()
        x = ttnn.mul(x, tw, output_tensor=x)
        tw.deallocate(True)
        d = down_of(x)
        x.deallocate(True)
        out = ttnn.experimental.fast_reduce_nc(d, dims=[1], memory_config=ttnn.L1_MEMORY_CONFIG)
        d.deallocate(True)
        return out

    _ab(device, "decode_routing", decode_old, decode_new, results, reference=ref_decode, shape=(users, H))
    for t in (x_in, routing_tt, sparsity, w_down):
        t.deallocate(True)

    # ---------------- 2. sorted prefill hot group at 1024 tokens: per-expert linears + K-concat down vs repeat/bmm ----------------
    E_POOL = 16  # the hot experts are drawn from this many expert slots (memory)
    hidden = up(torch.randn(1, 1, SPLIT, H, generator=g), dtype=ttnn.bfloat16)
    w_gu = up(torch.randn(1, E_POOL, H, 2 * IP, generator=g) * 0.02)
    w_dn = up(torch.randn(1, E_POOL, IP, H, generator=g) * 0.02)
    routing_rows_t = torch.rand(E_POOL, SPLIT) * (torch.rand(E_POOL, SPLIT) < 0.6)  # hot experts: ~60 % of the tokens
    routing_rows = up(routing_rows_t, dtype=ttnn.bfloat16)  # [E_POOL, split] = routing^T table of the plan
    hid_t, wgu_t, wdn_t, rr_t = (ttnn.to_torch(t).float() for t in (hidden, w_gu, w_dn, routing_rows))

    def torch_hot(hot_ids):
        out = torch.zeros(SPLIT, H)
        for e in hot_ids:
            gu = hid_t[0, 0] @ wgu_t[0, e]
            act = gu[:, IP:] * torch.nn.functional.silu(gu[:, :IP]) * rr_t[e].reshape(-1, 1)
            out += act @ wdn_t[0, e]
        return out

    def rw_hot_of(hot_ids, n_hot, transpose_form):
        hot_idx_t = ttnn.from_torch(
            torch.tensor([hot_ids], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        rw_rows = ttnn.embedding(hot_idx_t, routing_rows, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, n_hot, split]
        if transpose_form:
            rw_t = ttnn.to_layout(ttnn.reshape(rw_rows, (1, n_hot, 1, SPLIT)), ttnn.TILE_LAYOUT)
            rw = ttnn.transpose(rw_t, 2, 3)
            rw_t.deallocate(True)
        else:
            rw = ttnn.to_layout(ttnn.reshape(rw_rows, (1, n_hot, SPLIT, 1)), ttnn.TILE_LAYOUT)
        rw_rows.deallocate(True)
        hot_idx_t.deallocate(True)
        return rw

    def hot_old(hot_ids):
        n_hot = len(hot_ids)
        w_hot = _concat_expert_slices(w_gu, hot_ids, H, 2 * IP)
        hidden_rep = ttnn.repeat(hidden, ttnn.Shape((1, n_hot, 1, 1)))
        gu_hot = ttnn.matmul(
            hidden_rep,
            w_hot,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=dense_grid,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
        hidden_rep.deallocate(True)
        w_hot.deallocate(True)
        gate_h = ttnn.slice(gu_hot, [0, 0, 0, 0], [1, n_hot, SPLIT, IP])
        up_h = ttnn.slice(gu_hot, [0, 0, 0, IP], [1, n_hot, SPLIT, 2 * IP])
        gu_hot.deallocate(True)
        act_h = apply_glu(gate_h, up_h, "silu")
        gate_h.deallocate(True)
        up_h.deallocate(True)
        rw = rw_hot_of(hot_ids, n_hot, transpose_form=False)
        act_h = ttnn.mul(act_h, rw, output_tensor=act_h)
        rw.deallocate(True)
        wd_hot = _concat_expert_slices(w_dn, hot_ids, IP, H)
        down_h = ttnn.matmul(
            act_h,
            wd_hot,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=dense_grid,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
        act_h.deallocate(True)
        wd_hot.deallocate(True)
        out = reduce_experts(down_h)
        down_h.deallocate(True)
        return out

    def hot_new(hot_ids, kconcat=True, cores=None):
        n_hot = len(hot_ids)
        gu_list = []
        for e in hot_ids:
            w_e = _expert_slice(w_gu, e, H, 2 * IP)
            gu_list.append(
                ttnn.linear(
                    hidden,
                    w_e,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                    core_grid=dense_grid,
                    compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
                )
            )
            w_e.deallocate(True)
        gu_hot = ttnn.concat(gu_list, dim=1)
        for t in gu_list:
            t.deallocate(True)
        gate_h = ttnn.slice(gu_hot, [0, 0, 0, 0], [1, n_hot, SPLIT, IP])
        up_h = ttnn.slice(gu_hot, [0, 0, 0, IP], [1, n_hot, SPLIT, 2 * IP])
        gu_hot.deallocate(True)
        act_h = apply_glu(gate_h, up_h, "silu")
        gate_h.deallocate(True)
        up_h.deallocate(True)
        rw = rw_hot_of(hot_ids, n_hot, transpose_form=True)
        act_h = ttnn.mul(act_h, rw, output_tensor=act_h)
        rw.deallocate(True)
        if kconcat:
            parts = [ttnn.slice(act_h, [0, i, 0, 0], [1, i + 1, SPLIT, IP]) for i in range(n_hot)]
            act_h.deallocate(True)
            act_cat = ttnn.concat(parts, dim=3)
            for t in parts:
                t.deallocate(True)
            wd_cat = _concat_expert_slices(w_dn, hot_ids, IP, H, dim=2)
            out = _hot_down_kconcat_matmul(act_cat, wd_cat, ttnn.bfloat8_b, dense_grid, cores=cores)
            act_cat.deallocate(True)
            wd_cat.deallocate(True)
            return out
        wd_hot = _concat_expert_slices(w_dn, hot_ids, IP, H)
        down_h = ttnn.matmul(
            act_h,
            wd_hot,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            core_grid=dense_grid,
            compute_kernel_config=_DENSE_COMPUTE_KERNEL_CONFIG,
        )
        act_h.deallocate(True)
        wd_hot.deallocate(True)
        out = reduce_experts(down_h)
        down_h.deallocate(True)
        return out

    for n_hot in (4, 8, 15):
        hot_ids = list(range(n_hot))
        ref_hot = torch_hot(hot_ids)
        cfg = _hot_down_kconcat_config((8, 8), SPLIT, n_hot * IP, H)
        logger.info(f"hot group n_hot={n_hot}: K-concat 2D config {cfg}")
        _ab(
            device,
            f"hot_group_n{n_hot}",
            lambda: hot_old(hot_ids),
            lambda: hot_new(hot_ids),
            results,
            reference=ref_hot,
            shape=(SPLIT, H),
        )
        # variants of the new form: K-concat down with ttnn's auto config, and per-expert linears + bmm down + reduce
        ref_new = hot_new(hot_ids)
        for tag, fn in (
            ("kconcat_auto", lambda: hot_new(hot_ids, cores=(0, 0))),
            ("bmm_reduce", lambda: hot_new(hot_ids, kconcat=False)),
        ):
            got = fn()
            _compare(
                f"hot_group_n{n_hot}_{tag}", ref_new, got, results, reference=ref_hot, shape=(SPLIT, H), tolerance=1.0
            )
            got.deallocate(True)
            results[f"hot_group_n{n_hot}_{tag}_time"] = _time(device, fn)
            logger.info(
                f"[cand hot_group_n{n_hot} {tag}] {results[f'hot_group_n{n_hot}_{tag}_time']['wall_ms_min']:.3f} ms"
            )
        ref_new.deallocate(True)

    # ---------------- 3. copy-free layouts of ELIDE_ROUTING_COPIES: bit-identical to the phase-1 reshapes ----------------
    cap = 128
    idx_t = torch.randint(0, SPLIT, (1, 1, E, cap), generator=g)
    idx_rm = up(idx_t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    table = ttnn.reshape(hidden, (SPLIT, H))  # TILE bf16 table like the production path

    # (the reshaped index tensors are left to the refcount: the [E, cap] one is a view sharing idx_rm's buffer)
    def gather_old():
        idx_flat = ttnn.reshape(idx_rm, (1, E * cap))
        return ttnn.reshape(ttnn.embedding(idx_flat, table, layout=ttnn.TILE_LAYOUT), (1, E, cap, H))

    def gather_new():
        idx2 = ttnn.reshape(idx_rm, (E, cap))
        return ttnn.reshape(ttnn.embedding(idx2, table, layout=ttnn.TILE_LAYOUT), (1, E, cap, H))

    _ab(device, "cold_index", gather_old, gather_new, results, shape=(E * cap, H))
    assert results["cold_index"]["identical"], "the [E, cap] index gather must be bit-identical to the [1, E * cap] one"
    idx_rm.deallocate(True)

    rw_old = rw_hot_of(list(range(15)), 15, transpose_form=False)
    rw_new = rw_hot_of(list(range(15)), 15, transpose_form=True)
    _compare("hot_routing_rows", rw_old, rw_new, results, shape=(15 * SPLIT,))
    assert results["hot_routing_rows"][
        "identical"
    ], "the transposed routing rows must be bit-identical to the reshaped ones"
    assert list(rw_new.shape) == [1, 15, SPLIT, 1], rw_new.shape
    rw_old.deallocate(True)
    rw_new.deallocate(True)
    results["hot_routing_rows_old"] = _time(device, lambda: rw_hot_of(list(range(15)), 15, transpose_form=False))
    results["hot_routing_rows_new"] = _time(device, lambda: rw_hot_of(list(range(15)), 15, transpose_form=True))
    logger.info(
        f"[cand hot_routing_rows] old {results['hot_routing_rows_old']['wall_ms_min']:.3f} ms -> "
        f"new {results['hot_routing_rows_new']['wall_ms_min']:.3f} ms"
    )
    for t in (hidden, w_gu, w_dn, routing_rows):
        t.deallocate(True)

    if PERF_OUT:
        data = json.loads(open(PERF_OUT).read()) if os.path.isfile(PERF_OUT) else {}
        data["layout_candidates"] = results
        with open(PERF_OUT, "w") as f:
            json.dump(data, f, indent=2)
    logger.info(json.dumps(results, indent=1))
