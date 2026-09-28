# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (device): the verify step's AUX HIDDEN capture for the DFlash2 drafter (VerifyStep(keep_aux_hidden=True)).

VAUX_MODE=exact (default): per plan in VAUX_CONFIGS ("1,8;8,8;32,4"), after a real prefill of the grid's users:
  (a) an EAGER verify forward whose row_check hook reads the residual after every aux layer straight off the device
      (the reference rows) -- its own out_aux (filled by the same copy ops) must equal them bitwise;
  (b) the GDN state is restored to its pre-step values (persistent device snapshots allocated before any capture,
      values moved with ttnn.copy; qkv_prev re-zeroed) and the TRACED replay of the same step must reproduce the
      eager reference rows bitwise in plan.out_aux (and the same argmax rows).
  Flow rule (tests/VERIFY_W32_AUDIT.md): every program compiled before the first capture; all captures before the
  users' prefill; no prefill-trace replay afterwards.
VAUX_MODE=timing: per config, the traced step time (VAUX_REPLAYS=50 replays, median) of the plan WITH and WITHOUT
  keep_aux_hidden (one live trace at a time) -> the capture's cost per verify step, plus the eager per-section split.

Run (half A): scripts/verify_step_run.sh <tag> TEST=models/demos/blackhole/qwen36/tests/test_verify_aux_scratch.py VAUX_MODE=exact
"""
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_verify_step_scratch import AICLK_MHZ, PROMPTS, _pin_aiclk, _prefill
from models.demos.blackhole.qwen36.tt import aux_hidden as ah
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.verify_step import VerifyStep, combine_sharded_argmax

MODE = os.environ.get("VAUX_MODE", "exact")
CONFIGS = [tuple(int(v) for v in c.split(",")) for c in os.environ.get("VAUX_CONFIGS", "1,8;8,8;32,4").split(";") if c]
N_REPLAYS = int(os.environ.get("VAUX_REPLAYS", "50"))
OUT_JSON = os.environ.get("VAUX_OUT", "/home/eslim/experiments/qwen36/logs/verify_aux_result.json")
BMAX = 32
BPU = 8
CHUNK = 2048


def _setup(device):
    device.enable_program_cache()
    _pin_aiclk(AICLK_MHZ)
    t0 = time.perf_counter()
    model = Qwen36Model.from_pretrained(device, max_batch_size=BMAX, max_seq_len=BPU * BLOCK_SIZE)
    logger.info(f"[vaux] model load {time.perf_counter() - t0:.1f}s")
    page_tables = torch.stack([torch.arange(u * BPU, (u + 1) * BPU, dtype=torch.int32) for u in range(BMAX)])
    kv_shape = [BMAX * BPU, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=BMAX)
    return model, page_tables


def _prefill_warmup(model, device):
    """The served prefill warm-up (chunk trace + masked-bucket traces + slot-write programs)."""
    t0 = time.perf_counter()
    pt_full = torch.arange(BMAX * BPU, dtype=torch.int32).reshape(1, -1)
    prev = model._bind_gdn_prefill_scratch()
    try:
        model.capture_prefill_trace_chunked(device, pt_full, chunk_size=CHUNK, capture_chunk_trace=True)
    finally:
        model._unbind_gdn_prefill_scratch(prev)
    model.warmup_gdn_slot_write()
    for layer in model.layers:
        if not layer.is_full_attention and hasattr(layer.attention, "warmup_hist_device_pack"):
            layer.attention.warmup_hist_device_pack()
    ttnn.synchronize_device(device)
    logger.info(f"[vaux] prefill warmup {time.perf_counter() - t0:.1f}s")


class GdnSnapshot:
    """Persistent device copies of every GDN layer's decode state (allocated BEFORE any capture): save/restore by
    ttnn.copy (values only, the state buffers keep their trace-baked addresses)."""

    def __init__(self, model):
        self.model = model
        self.items = []  # (state tensor, snapshot tensor)
        for layer in model.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            for t in [dn.rec_state, dn.conv_hist_packed] + list(dn.conv_states):
                if t is not None:
                    self.items.append((t, ttnn.clone(t)))
        ttnn.synchronize_device(model.mesh_device)
        logger.info(f"[vaux] GDN snapshot buffers: {len(self.items)} tensors")

    def save(self):
        for t, s in self.items:
            ttnn.copy(t, s)
        ttnn.synchronize_device(self.model.mesh_device)

    def restore(self):
        for t, s in self.items:
            ttnn.copy(s, t)
        ttnn.synchronize_device(self.model.mesh_device)


def _host_rows(model, x):
    """Host bf16 [R, dim] of a residual tensor: replicated (device 0) or fractured (gathered along the hidden dim)."""
    if x.shape[-1] == model.args.dim:
        rows = ttnn.to_torch(ttnn.get_device_tensors(x)[0])
    else:
        rows = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=3))
    return rows.to(torch.bfloat16).reshape(-1, rows.shape[-1]).clone()


def _step_inputs(w, T, lens, i=0):
    tokens = [[(1000 + 7 * s + 3 * j + i) % 60000 for j in range(T)] for s in range(w)]
    positions = [int(lens[s]) for s in range(w)]
    accept = [0] * w
    return tokens, positions, accept


def run_exact(device):
    model, page_tables = _setup(device)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt_ids = [tok(p, return_tensors="pt").input_ids.to(torch.int32) for p in PROMPTS]
    steps = {}
    results = {}
    failures = []
    try:
        # persistent buffers + compile-first
        for w, T in CONFIGS:
            steps[(w, T)] = VerifyStep(model, w, T, page_tables[:w], keep_hidden=True, keep_aux_hidden=True)
        snap = GdnSnapshot(model)
        for vs in steps.values():
            vs.compile()
        _prefill_warmup(model, device)
        _prefill(model, prompt_ids, page_tables, 1)  # the warm 1-user prefill (lazy allocations)
        for vs in steps.values():
            vs.capture()
        w_max = max(w for w, _ in CONFIGS)
        lens, first = _prefill(model, prompt_ids, page_tables, w_max)
        aux_layers = ah.aux_layers_for(model)
        dim = model.args.dim
        for (w, T), vs in steps.items():
            plan = vs.plan
            tokens, positions, accept = _step_inputs(w, T, lens)
            # ---- (a) eager step at state S0 with the row_check reference
            snap.save()
            vs.reset_sequence()
            ref = {}

            def row_check(name, x):
                for li in aux_layers:
                    if name.startswith(f"layer{li}_"):
                        ref[li] = _host_rows(model, x)

            plan.upload(tokens, positions, accept)
            idx, val = vs.forward(row_check=row_check)
            ttnn.synchronize_device(device)
            eager_arg = combine_sharded_argmax(device, idx, val, plan.R, vs.per_shard)
            ttnn.deallocate(idx)
            ttnn.deallocate(val)
            vs._drop_last_hidden()
            plan._host_refs = []
            ref_rows = torch.cat([ref[li] for li in aux_layers], dim=-1)  # [R, n_aux*dim]
            eager_aux = plan.read_aux()
            ok_a = torch.equal(eager_aux, ref_rows)
            # ---- (b) restore S0, traced replay of the same step
            snap.restore()
            vs.reset_sequence()
            traced_arg = vs.run(tokens, positions, accept)
            traced_aux = plan.read_aux()
            ok_b = torch.equal(traced_aux, ref_rows)
            ok_arg = torch.equal(eager_arg, traced_arg)
            nz = int((ref_rows.float().abs().sum(-1) > 0).sum())
            per_layer = [
                bool(torch.equal(traced_aux[:, k * dim : (k + 1) * dim], ref[li])) for k, li in enumerate(aux_layers)
            ]
            results[f"{w},{T}"] = {
                "R": plan.R,
                "fused_ar": plan.fused_ar,
                "eager_aux_eq_ref": ok_a,
                "traced_aux_eq_ref": ok_b,
                "argmax_eager_eq_traced": ok_arg,
                "rows_nonzero": nz,
                "per_layer_traced_eq": per_layer,
                "max_abs_diff_traced": float((traced_aux.float() - ref_rows.float()).abs().max()),
            }
            print(
                f"VAUX_EXACT config=({w},{T}) R={plan.R} fused_ar={plan.fused_ar} eager_out_aux==ref={ok_a} "
                f"traced_out_aux==ref={ok_b} per_layer={per_layer} argmax_eager==traced={ok_arg} nonzero_rows={nz}/{plan.R}",
                flush=True,
            )
            if not (ok_a and ok_b and ok_arg):
                failures.append(f"({w},{T}): eager {ok_a} traced {ok_b} argmax {ok_arg}")
    finally:
        for vs in steps.values():
            vs.release()
        model.free_kv_caches()
    with open(OUT_JSON, "w") as f:
        json.dump({"mode": "exact", "results": results}, f, indent=1)
    assert not failures, "\n".join(failures)


def run_timing(device):
    model, page_tables = _setup(device)
    res = {}
    plans = []
    try:
        for w, T in CONFIGS:
            for aux in (False, True):
                vs = VerifyStep(model, w, T, page_tables[:w], keep_hidden=True, keep_aux_hidden=aux)
                plans.append(((w, T), aux, vs))
        for _, aux, vs in plans:
            vs.compile(profile=True)
            vs.section_times_eager = dict(vs.section_times or {})
        for (w, T), aux, vs in plans:
            vs.capture()
            med, mn = vs.time_replays(N_REPLAYS)
            vs.release()
            key = f"{w},{T}"
            res.setdefault(key, {"R": vs.plan.R})
            res[key]["aux" if aux else "plain"] = {
                "median_ms": med,
                "min_ms": mn,
                "eager_sections": vs.section_times_eager,
            }
            print(
                f"VAUX_TIMING config=({w},{T}) R={vs.plan.R} keep_aux_hidden={aux}: traced step {med:.2f} ms (min {mn:.2f}); "
                f"eager aux_copies {1e3 * vs.section_times_eager.get('aux_copies', 0.0):.2f} ms",
                flush=True,
            )
        for key, r in res.items():
            if "aux" in r and "plain" in r:
                d = r["aux"]["median_ms"] - r["plain"]["median_ms"]
                r["delta_ms"] = d
                kb = r["R"] * len(ah.DFLASH2_TARGET_LAYERS) * model.args.dim * 2 / 1024
                print(
                    f"VAUX_COST config=({key}) R={r['R']}: +{d:.2f} ms per verify step for {kb:.0f} KB of aux rows "
                    f"({r['plain']['median_ms']:.2f} -> {r['aux']['median_ms']:.2f} ms)",
                    flush=True,
                )
    finally:
        for _, _, vs in plans:
            vs.release()
        model.free_kv_caches()
    with open(OUT_JSON, "w") as f:
        json.dump({"mode": "timing", "results": res}, f, indent=1)


@run_for_blackhole()
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_verify_aux(mesh_device):
    if not _MULTI:
        pytest.skip("TP path only")
    if MODE == "exact":
        run_exact(mesh_device)
    elif MODE == "timing":
        run_timing(mesh_device)
    else:
        pytest.fail(f"VAUX_MODE={MODE}")
