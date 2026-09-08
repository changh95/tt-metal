# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device validation of the perf-p2 INDEXED/GATHER single-user expert path against the scan path and a torch fp32
reference (one device).

Runs on a 1x1 mesh with random bfloat8_b weights at the real Solar-Open TP=8 per-device shapes (H = 4096, Ip = 160,
E = 128, k = 8) and calls the production ``tt/experts/decode.py::decode_forward`` twice per token: with the dense
``[1, E]`` routing tensor (phase-1 scan path) and with an ``IndexedRouting`` (``_decode_forward_indexed``,
``MoEOptions.indexed_decode``). Checks, per token: PCC of both forms vs the fp32 result of the device-rounded operands
(the indexed form must not be worse than the scan form by more than ``TOLERANCE``), PCC / max abs diff between the two
device results, that the two weight-layout variants of the indexed path (``decode.INDEXED_WEIGHTS_LAYOUT``
"transpose" / "row_major") are bit-identical, and the same for the routing-weight placement
(``decode.INDEXED_WEIGHTS_ON_DOWN_INPUT``: on the compact GLU rows vs on the compact down outputs, the scan path's
order -- the latter is expected to reproduce the scan result bit for bit or nearly so). Timing: synchronised eager wall (min of REPS) and a trace capture +
replay of each form (the decode step is traced in production), plus the isolated cost of the routing conversions
(uint16 TILE -> ROW_MAJOR ids as ``TopKRouter.route_indexed`` does it; ``_indexed_expert_scalars`` in both modes).
Results are logged as ``[cand ...]`` lines and, with ``SOLAR_OPEN_PERF_OUT=<json>``, written to that file.

    SOLAR_OPEN_PERF_OUT=/path/cand.json pytest models/demos/solar_open/tests/perf/test_indexed_candidates.py \
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
from models.demos.solar_open.config import MeshConfig, ModeConfig
from models.demos.solar_open.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.solar_open.tt.expert_configs import solar_open_program_config
from models.demos.solar_open.tt.experts import decode as experts_decode
from models.demos.solar_open.tt.experts.config import ExpertConfig, IndexedRouting
from models.demos.solar_open.tt.experts.weights import ExpertWeights

PERF_OUT = os.getenv("SOLAR_OPEN_PERF_OUT", "")
REPS = int(os.getenv("SOLAR_OPEN_PERF_REPS", "10"))
TRACE_REPLAYS = int(os.getenv("SOLAR_OPEN_PERF_TRACE_REPLAYS", "50"))
H, IP, E, K = 4096, 160, 128, 8
TOKENS = 4  # distinct random routings / inputs
TOLERANCE = 2e-4  # the indexed form's PCC vs fp32 may not be below the scan form's by more than this
MIN_PCC = 0.99


def _sync(device):
    ttnn.synchronize_device(device)


def _time_eager(device, fn, reps=REPS):
    out = fn()
    _sync(device)
    out.deallocate(True)
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        walls.append((time.perf_counter() - t0) * 1e3)
        out.deallocate(True)
    return {"eager_ms_min": min(walls), "eager_ms_mean": sum(walls) / len(walls)}


def _time_trace(device, fn, replays=TRACE_REPLAYS):
    """Capture ``fn`` (device ops only) in a trace and replay it: ms per replay, non-blocking back-to-back and
    blocking (min)."""
    out = fn()
    _sync(device)
    out.deallocate(True)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    _sync(device)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(replays):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    _sync(device)
    nonblocking = (time.perf_counter() - t0) / replays * 1e3
    blocking = []
    for _ in range(replays):
        t1 = time.perf_counter()
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        blocking.append((time.perf_counter() - t1) * 1e3)
    ttnn.release_trace(device, trace_id)
    out.deallocate(True)
    return {"trace_ms_nonblocking": nonblocking, "trace_ms_blocking_min": min(blocking)}


def _log(results, name, rec):
    results[name] = rec
    logger.info(f"[cand {name}] {json.dumps(rec)}")


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric([(1, 1)])
def test_indexed_candidates(mesh_device, device_params, reset_seeds):
    device = mesh_device
    g = torch.Generator().manual_seed(29)
    results = {}
    shipped_layout = experts_decode.INDEXED_WEIGHTS_LAYOUT
    shipped_on_input = experts_decode.INDEXED_WEIGHTS_ON_DOWN_INPUT
    grid = device.compute_with_storage_grid_size()
    logger.info(f"compute grid {grid.x}x{grid.y}")

    def up(t, dtype=ttnn.bfloat8_b, mem=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, device=device, dtype=dtype, layout=layout, memory_config=mem)

    # ---------------- production-shaped random experts (per-device TP=8 shapes) ----------------
    w_gu = up(torch.randn(1, E, H, 2 * IP, generator=g) * 0.02)  # [1, E, H, 2 Ip] = [gate | up]
    w_down = up(torch.randn(1, E, IP, H, generator=g) * 0.02)  # [1, E, Ip, H]
    w_gu_dev = ttnn.to_torch(w_gu).float()  # device-rounded (bfp8) operands for the fp32 reference
    w_down_dev = ttnn.to_torch(w_down).float()
    weights = ExpertWeights(
        gate_up_proj=w_gu, down_proj=w_down, intermediate_size_per_device=IP, intermediate_padded_per_device=IP
    )
    config = ExpertConfig(intermediate_size=1280, num_experts=E, hidden_size=H, num_experts_per_tok=K)
    mesh_config = MeshConfig((1, 1), decode=ModeConfig(tp=1, ep=1))
    program_config = solar_open_program_config(device)
    placeholder = up(torch.ones(1, 1, 1, E), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def run_scan(x, dense):
        # decode_forward consumes x; the dense routing tensor is cloned because the scan path frees its permuted
        # view (one extra ~2 KB launch, charged to the scan form)
        return experts_decode.decode_forward(
            ttnn.clone(x), ttnn.clone(dense), weights, config, mesh_config, device, None, program_config
        )

    def run_indexed(x, routing, mode, on_input=True):
        experts_decode.INDEXED_WEIGHTS_LAYOUT = mode
        experts_decode.INDEXED_WEIGHTS_ON_DOWN_INPUT = on_input
        return experts_decode.decode_forward(
            ttnn.clone(x),
            None,
            weights,
            config,
            mesh_config,
            device,
            None,
            program_config,
            indexed_routing=routing,
            sparsity_placeholder=placeholder,
        )

    # ---------------- 0. routing conversions in isolation ----------------
    idx_t = torch.randperm(E, generator=g)[:K]
    w_t = torch.rand(K, generator=g)
    w_t = (w_t / w_t.sum()).to(torch.bfloat16)
    idx_tile = up(
        idx_t.reshape(1, K).to(torch.int32), dtype=ttnn.uint16, mem=ttnn.L1_MEMORY_CONFIG
    )  # as moe_grouped_topk emits
    idx_rm = ttnn.to_layout(idx_tile, ttnn.ROW_MAJOR_LAYOUT)
    idx_rm = ttnn.reshape(idx_rm, (1, 1, 1, K))
    assert idx_rm.layout == ttnn.ROW_MAJOR_LAYOUT and idx_rm.dtype == ttnn.uint16
    idx_back = ttnn.to_torch(idx_rm).reshape(K).long()
    assert torch.equal(idx_back, idx_t), f"uint16 ids through untilize: {idx_back.tolist()} != {idx_t.tolist()}"
    results["ids_untilize"] = _time_eager(device, lambda: ttnn.to_layout(idx_tile, ttnn.ROW_MAJOR_LAYOUT))
    logger.info(f"[cand ids_untilize] {json.dumps(results['ids_untilize'])}")
    w_tile = up(w_t.reshape(1, 1, 1, K), dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
    for mode in ("transpose", "row_major"):
        scalars = experts_decode._indexed_expert_scalars(w_tile, K, mode)
        assert tuple(scalars.shape) == (1, K, 1, 1) and scalars.layout == ttnn.TILE_LAYOUT, scalars.shape
        got = ttnn.to_torch(scalars).reshape(K).to(torch.bfloat16)
        assert torch.equal(got, w_t), f"{mode}: scalars {got.tolist()} != {w_t.tolist()}"
        scalars.deallocate(True)
        results[f"scalars_{mode}"] = _time_eager(
            device, lambda m=mode: experts_decode._indexed_expert_scalars(w_tile, K, m)
        )
        logger.info(f"[cand scalars_{mode}] {json.dumps(results[f'scalars_{mode}'])}")

    # ---------------- 1. full single-user expert path: scan vs indexed vs fp32 ----------------
    pcc_scan, pcc_idx, pcc_ow = [], [], []
    for t in range(TOKENS):
        x_t = torch.randn(1, 1, 1, H, generator=g).to(torch.bfloat16)
        idx_t = torch.randperm(E, generator=g)[:K]
        w_t = torch.rand(K, generator=g)
        w_t = (w_t / w_t.sum()).to(torch.bfloat16)  # bf16-exact weights, as the router emits them
        dense_t = torch.zeros(1, E)
        dense_t[0, idx_t] = w_t.float()
        # fp32 reference of the device-rounded operands
        xf = x_t.float().reshape(H)
        gu = torch.einsum("k,ekn->en", xf, w_gu_dev[0, idx_t])  # [K, 2 Ip]
        act = gu[:, IP:] * torch.nn.functional.silu(gu[:, :IP])  # [K, Ip]
        ref = torch.einsum("e,ek,ekn->n", w_t.float(), act, w_down_dev[0, idx_t])  # [H]

        x = up(x_t, dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)
        dense = up(dense_t, dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG)  # [1, E]
        routing = IndexedRouting(
            indices=up(
                idx_t.reshape(1, 1, 1, K).to(torch.int32),
                dtype=ttnn.uint16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mem=ttnn.L1_MEMORY_CONFIG,
            ),
            weights=up(w_t.reshape(1, 1, 1, K), dtype=ttnn.bfloat16, mem=ttnn.L1_MEMORY_CONFIG),
            top_k=K,
        )

        out_scan = run_scan(x, dense)
        out_tr = run_indexed(x, routing, "transpose")  # weights on the down input
        out_rm = run_indexed(x, routing, "row_major")
        out_ow = run_indexed(x, routing, "transpose", on_input=False)  # weights on the down output (scan order)
        scan_t = ttnn.to_torch(out_scan).float().reshape(-1)[:H]
        tr_t = ttnn.to_torch(out_tr).float().reshape(-1)[:H]
        rm_t = ttnn.to_torch(out_rm).float().reshape(-1)[:H]
        ow_t = ttnn.to_torch(out_ow).float().reshape(-1)[:H]
        _, p_scan = comp_pcc(ref, scan_t, 0.0)
        _, p_tr = comp_pcc(ref, tr_t, 0.0)
        _, p_ow = comp_pcc(ref, ow_t, 0.0)
        _, p_mutual = comp_pcc(scan_t, tr_t, 0.0)
        _, p_mutual_ow = comp_pcc(scan_t, ow_t, 0.0)
        rec = {
            "shape_scan": list(out_scan.shape),
            "shape_indexed": list(out_tr.shape),
            "pcc_scan_vs_ref": p_scan,
            "pcc_indexed_vs_ref": p_tr,
            "pcc_indexed_output_vs_ref": p_ow,
            "pcc_indexed_vs_scan": p_mutual,
            "pcc_indexed_output_vs_scan": p_mutual_ow,
            "max_abs_err_scan": (ref - scan_t).abs().max().item(),
            "max_abs_err_indexed": (ref - tr_t).abs().max().item(),
            "max_abs_err_indexed_output": (ref - ow_t).abs().max().item(),
            "max_abs_diff_indexed_vs_scan": (scan_t - tr_t).abs().max().item(),
            "max_abs_diff_indexed_output_vs_scan": (scan_t - ow_t).abs().max().item(),
            "n_diff_indexed_output_vs_scan": int((scan_t != ow_t).sum().item()),
            "layout_variants_identical": bool(torch.equal(tr_t, rm_t)),
            "output_variant_identical_to_scan": bool(torch.equal(scan_t, ow_t)),
            "ref_std": ref.std().item(),
        }
        _log(results, f"token{t}", rec)
        assert tuple(out_tr.shape) == tuple(out_scan.shape) == tuple(out_ow.shape) == (1, 1, 1, H), rec
        assert rec["layout_variants_identical"], "the two INDEXED_WEIGHTS_LAYOUT variants must agree bit for bit"
        assert p_tr >= MIN_PCC and p_tr >= p_scan - TOLERANCE, f"indexed path PCC {p_tr} vs scan {p_scan}"
        assert p_ow >= MIN_PCC and p_ow >= p_scan - TOLERANCE, f"indexed (output mul) PCC {p_ow} vs scan {p_scan}"
        pcc_scan.append(p_scan)
        pcc_idx.append(p_tr)
        pcc_ow.append(p_ow)
        for o in (out_scan, out_tr, out_rm, out_ow):
            o.deallocate(True)

        if t == 0:
            # ---------------- 2. timing on the first token (eager + traced) ----------------
            for name, fn in (
                ("scan", lambda: run_scan(x, dense)),
                ("indexed_transpose", lambda: run_indexed(x, routing, "transpose")),
                ("indexed_row_major", lambda: run_indexed(x, routing, "row_major")),
                ("indexed_output_mul", lambda: run_indexed(x, routing, "transpose", on_input=False)),
            ):
                rec = _time_eager(device, fn)
                rec.update(_time_trace(device, fn))
                _log(results, f"time_{name}", rec)
        x.deallocate(True)
        dense.deallocate(True)
        routing.deallocate()

    experts_decode.INDEXED_WEIGHTS_LAYOUT = shipped_layout
    experts_decode.INDEXED_WEIGHTS_ON_DOWN_INPUT = shipped_on_input
    summary = {
        "pcc_scan_vs_ref_mean": sum(pcc_scan) / len(pcc_scan),
        "pcc_indexed_vs_ref_mean": sum(pcc_idx) / len(pcc_idx),
        "pcc_indexed_output_vs_ref_mean": sum(pcc_ow) / len(pcc_ow),
        "tokens": TOKENS,
    }
    _log(results, "summary", summary)
    if PERF_OUT:
        with open(PERF_OUT, "w") as f:
            json.dump(results, f, indent=1)
        logger.info(f"wrote {PERF_OUT}")
