# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 3g / D1: bisection of the packed-prefill first-token bias (real weights, 1x8, backlog item 8).

Three arms on the same prompts (users of the 128-token KO/EN set on the pinned 2026-09-08 ids, reasoning_effort low):

* A -- the production sequential per-user prefill (``Generator.prefill_forward_text``, one user per call, 32-row head);
* B -- packed passes of two users through ``prefill_forward_text_batched`` (T = 256: dense-bmm MoE, no sorted plan;
  the tt_transformers batched path runs norm + lm_head on all T rows); ``0+0`` packs user 0 with a copy of itself
  (pass-mate independence: row-wise ops must give bit-identical rows for the copy);
* C -- ONE user (B = 1, T = 128) through the packed CODE PATH: ``prepare_inputs_prefill`` + ``ttnn_prefill_forward``
  with ``get_last_token=-1`` (head on all 128 rows) -- isolates the code path from the row count.

Hooks on the residual stream and on every block of every layer (host copies of device 0's tensor; the device tensors
are left to the model) give per layer and per op: PCC, max |diff|, signed mean diff, the relative scale of B along A
(``<B, A> / <A, A> - 1``: a magnitude bias shows as a consistent sign), the fraction of rows that differ at all, the
router's expert-set flips, for A vs B, A vs C and B(0+1) vs B(0+0). The head is then run on IDENTICAL hidden states
through the 32-row sequential head, the T-row "full" head (T = 256 / 1024 / 4096) and the gather head; every logits
vector is ranked against the bf16 HF first-token reference (``tests/accuracy/gen_prefill_reference.py``) and the
``<|think|>`` (22) / ``<|content|>`` (23) gap is reported.

    SOLAR_OPEN_BISECT_OUT=<dir> pytest models/demos/solar_open/tests/test_packed_bias_bisect.py -k 1x8 -x -p no:cacheprovider

Env: ``SOLAR_OPEN_BISECT_USERS`` (default ``0,1,12,19``), ``SOLAR_OPEN_BISECT_PASSES`` (default ``0+1,12+19,0+0``),
``SOLAR_OPEN_BISECT_HEAD_T`` (default ``256,1024,4096``). Diagnostic: it asserts only that the hook path reproduces the
production logits bit for bit; everything else is written to ``<out>/bisect_results.json`` and ``<out>/tables.md``.

Phase 3g / D2 adds arm **D**: arm B's first two passes again with the "sequential numerics" knob
(``SOLAR_OPEN_PACKED_PREFILL_SEQ_NUMERICS``, ``tt/packed_numerics.py``) at ``SOLAR_OPEN_BISECT_FIX_LEVEL`` (default 1)
-- the before / after of the fix per layer and per op (expected: bit-identical to arm A in every op of every layer at
T = 256) -- plus ``SOLAR_OPEN_BISECT_VERIFY_FIX=1``: bit-identity checks of the fix's levers on identical inputs (the
pinned qkv / o_proj configs at 4096 / 512 rows, the lm_head at 1024 rows, the piece-wise shared expert, the tile-wise
head) and the E0 question (which final-norm kernel is closer to fp32). ``SOLAR_OPEN_BISECT_ISO=0`` / ``_TRUTH=0`` skip
the D1 isolation / truth sections.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.solar_open.config import Mode
from models.demos.solar_open.tests.accuracy.gen_prefill_reference import default_prefill_reference_path
from models.demos.solar_open.tests.test_factory import TestFactory, parametrize_mesh_with_fabric
from models.demos.solar_open.tt import model_config as mc
from models.demos.solar_open.tt import packed_numerics, packed_prefill
from models.demos.solar_open.tt.attention import prefill as attn_prefill
from models.demos.solar_open.tt.experts import prefill as experts_prefill
from models.demos.solar_open.tt.layer import DecoderLayer
from models.demos.solar_open.tt.model import batched_prefill_flag, prefill_forward_text_batched
from models.demos.solar_open.tt.rms_norm import RMSNorm
from models.demos.solar_open.tt.shared_expert import SharedExpert, shared_expert_program_configs
from models.demos.solar_open.tt.topk import TopKRouter, router_linear_program_config
from models.tt_transformers.tt.common import get_padded_prefill_len, preprocess_inputs_prefill

PROMPTS_128 = "models/demos/solar_open/demo/sample_prompts/input_data_questions_ko_en_prefill_128.json"
OUT_DIR_ENV = "SOLAR_OPEN_BISECT_OUT"
DEFAULT_OUT = (
    "/tmp/claude-1000/-home-eslim-experiments-solar/ee21423e-da8a-4596-9953-951b67590906/scratchpad/phase3g/D1/out"
)
THINK, CONTENT = 22, 23
SEQ_LEN = 128

# Order of the hooked ops inside one layer (the "first divergent op" scan walks this list).
OP_ORDER = [
    "embed",
    "norm1",
    "qkv",
    "rope_q",
    "rope_k",
    "sdpa",
    "sdpa_concat",
    "oproj_partial",
    "attn_out",
    "resid_attn",
    "norm2",
    "router_dense",
    "shared_partial",
    "routed_partial",
    "moe_partial",
    "moe_out",
    "resid_moe",
]


# ---------------------------------------------------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------------------------------------------------


class Recorder:
    """Host copies of device-0 tensors keyed by ``(arm, layer, tag)`` while ``enabled``."""

    def __init__(self):
        self.enabled = False
        self.arm = None
        self.layer = -1
        self.store = {}
        self.norm_tags = {}
        self.resid_count = 0
        self.rope_count = 0
        self.final_norm_id = None
        self.moe_paths = {}
        self.replication = {}

    def put(self, tag, tensor, all_devices=False):
        if not self.enabled:
            return
        devs = ttnn.get_device_tensors(tensor)
        if all_devices:
            host = torch.stack([ttnn.to_torch(d) for d in devs])
        else:
            host = ttnn.to_torch(devs[0])
        if host.dtype == torch.float32 and tensor.dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
            host = host.to(torch.bfloat16)  # exact for bf16 / bfp8 values
        self.store[(self.arm, self.layer, tag)] = host.clone()

    def get(self, arm, layer, tag):
        return self.store.get((arm, layer, tag))


def install_hooks(monkeypatch, rec, model):
    """Patch the model's module-level functions / classes so every block output of every layer is captured."""
    orig_layer_call = DecoderLayer.__call__

    def layer_call(self, hidden_states, *args, **kwargs):
        rec.layer = self.layer_idx
        rec.norm_tags = {id(self.input_layernorm): "norm1", id(self.post_attention_layernorm): "norm2"}
        rec.resid_count = 0
        rec.rope_count = 0
        if self.layer_idx == 0:
            rec.put("embed", hidden_states)
        out = orig_layer_call(self, hidden_states, *args, **kwargs)
        if rec.enabled and self.layer_idx == 0:
            rec.moe_paths[rec.arm] = dict(experts_prefill.LAST_PREFILL_MOE_PATH)
        return out

    monkeypatch.setattr(DecoderLayer, "__call__", layer_call)

    orig_residual_add = DecoderLayer._residual_add

    def residual_add(residual, branch):
        out = orig_residual_add(residual, branch)
        rec.resid_count += 1
        rec.put("resid_attn" if rec.resid_count == 1 else "resid_moe", out)
        return out

    monkeypatch.setattr(DecoderLayer, "_residual_add", staticmethod(residual_add))

    orig_norm_forward = RMSNorm.forward
    rec.final_norm_id = id(model.norm)

    def norm_forward(self, x):
        out = orig_norm_forward(self, x)
        tag = rec.norm_tags.get(id(self))
        if tag is None and id(self) == rec.final_norm_id:
            tag = "final_norm"
        if tag is not None:
            rec.put(tag, out)
        return out

    monkeypatch.setattr(RMSNorm, "forward", norm_forward)

    def wrap_output(module, name, tag):
        orig = getattr(module, name)

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            rec.put(tag, out)
            return out

        monkeypatch.setattr(module, name, wrapped)

    wrap_output(attn_prefill, "apply_qkv_projection", "qkv")
    wrap_output(attn_prefill, "concat_heads", "sdpa_concat")
    wrap_output(attn_prefill, "apply_output_projection", "oproj_partial")

    orig_allreduce = attn_prefill.apply_allreduce

    def allreduce(tensor, mesh_config, ccl_manager, hidden_size):
        out = orig_allreduce(tensor, mesh_config, ccl_manager, hidden_size)
        if rec.enabled and rec.layer == 0 and rec.arm not in rec.replication:
            devs = ttnn.get_device_tensors(out)
            d0, d7 = ttnn.to_torch(devs[0]).float(), ttnn.to_torch(devs[-1]).float()
            rec.replication[rec.arm] = {"attn_out_dev0_vs_dev7_max_abs": float((d0 - d7).abs().max())}
        rec.put("attn_out", out)
        return out

    monkeypatch.setattr(attn_prefill, "apply_allreduce", allreduce)

    orig_rope = attn_prefill.apply_rope

    def rope(tensor, rope_mats, transformation_mat, is_decode_mode):
        out = orig_rope(tensor, rope_mats, transformation_mat, is_decode_mode)
        rec.rope_count += 1
        rec.put("rope_q" if rec.rope_count == 1 else "rope_k", out)
        return out

    monkeypatch.setattr(attn_prefill, "apply_rope", rope)

    orig_sdpa = ttnn.transformer.scaled_dot_product_attention

    def sdpa(*args, **kwargs):
        out = orig_sdpa(*args, **kwargs)
        rec.put("sdpa", out)
        return out

    monkeypatch.setattr(ttnn.transformer, "scaled_dot_product_attention", sdpa)

    orig_router_call = TopKRouter.__call__

    def router_call(self, hidden_states, is_decode=True):
        indices, dense = orig_router_call(self, hidden_states, is_decode)
        rec.put("router_dense", dense)
        return indices, dense

    monkeypatch.setattr(TopKRouter, "__call__", router_call)

    orig_shared_call = SharedExpert.__call__

    def shared_call(self, x, is_decode=None):
        out = orig_shared_call(self, x, is_decode)
        rec.put("shared_partial", out)
        return out

    monkeypatch.setattr(SharedExpert, "__call__", shared_call)

    wrap_output(experts_prefill, "_process_prefill_chunk", "routed_partial")

    orig_moe_allreduce = experts_prefill.apply_tensor_parallel_allreduce

    def moe_allreduce(tensor, mesh_config, mesh_device, seq_len, ccl_manager):
        rec.put("moe_partial", tensor)  # routed + shared partial of device 0 (the op frees its input)
        out = orig_moe_allreduce(tensor, mesh_config, mesh_device, seq_len, ccl_manager)
        rec.put("moe_out", out)
        return out

    monkeypatch.setattr(experts_prefill, "apply_tensor_parallel_allreduce", moe_allreduce)


# ---------------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------------


def _rows_of(tag, tensor, slot, seq_len):
    """The ``[seq_len, D]`` rows of user ``slot`` in a captured tensor (any arm)."""
    t = tensor
    if tag in ("sdpa", "rope_q", "rope_k"):  # [B, heads, S, hd] -> [S, heads * hd]
        return t[slot].permute(1, 0, 2).reshape(seq_len, -1).float()
    if tag == "sdpa_concat":  # [B, 1, S, heads * hd]
        return t[slot].reshape(seq_len, -1).float()
    rows = t.reshape(-1, t.shape[-1])
    return rows[slot * seq_len : (slot + 1) * seq_len].float()


def _pcc_rows(a, b):
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)


def compare_rows(a, b, n_real):
    """Metrics of ``b`` against the reference ``a`` (``[S, D]``) over the ``n_real`` prompt rows and the last one."""
    a, b = a[:n_real], b[:n_real]
    d = b - a
    pcc = _pcc_rows(a, b)
    scale = (a * b).sum(-1) / ((a * a).sum(-1) + 1e-20) - 1.0  # relative scale of b along a, per row
    norm_ratio = b.norm(dim=-1) / (a.norm(dim=-1) + 1e-20) - 1.0
    row_diff = d.abs().amax(dim=-1)
    last = n_real - 1
    return {
        "pcc_min": float(pcc.min()),
        "pcc_mean": float(pcc.mean()),
        "max_abs": float(d.abs().max()),
        "mean_signed": float(d.mean()),
        "rel_rms": float(d.norm() / (a.norm() + 1e-20)),
        "scale_mean": float(scale.mean()),
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "norm_ratio_mean": float(norm_ratio.mean()),
        "frac_rows_diff": float((row_diff > 0).float().mean()),
        "last_pcc": float(pcc[last]),
        "last_max_abs": float(row_diff[last]),
        "last_scale": float(scale[last]),
        "last_mean_signed": float(d[last].mean()),
        "last_rel_rms": float(d[last].norm() / (a[last].norm() + 1e-20)),
        "bit_identical": bool(row_diff.max() == 0),
    }


def router_flips(a, b, n_real):
    """Rows whose selected expert set differs between two ``[S, E]`` dense routing tensors; weight diff on same sets."""
    a, b = a[:n_real], b[:n_real]
    set_a, set_b = a != 0, b != 0
    flipped = (set_a != set_b).any(dim=-1)
    same = ~flipped
    wdiff = float((a[same] - b[same]).abs().max()) if same.any() else 0.0
    return {
        "rows_set_flipped": int(flipped.sum()),
        "rows": int(n_real),
        "last_row_flipped": bool(flipped[n_real - 1]),
        "same_set_weight_max_abs": wdiff,
    }


def _kl(ref, other):
    lp = torch.log_softmax(ref.float(), dim=-1)
    lq = torch.log_softmax(other.float(), dim=-1)
    return float((lp.exp() * (lp - lq)).sum())


def logits_summary(name, logits, hf, seq_logits):
    l = logits.float()
    top2 = l.topk(2)
    out = {
        "name": name,
        "top1": int(top2.indices[0]),
        "top2": int(top2.indices[1]),
        "margin": float(top2.values[0] - top2.values[1]),
        "gap_22_23": float(l[THINK] - l[CONTENT]),
        "logit_22": float(l[THINK]),
        "logit_23": float(l[CONTENT]),
    }
    if hf is not None:
        out["kl_hf"] = _kl(hf, l)
        out["pcc_hf"] = float(_pcc_rows(hf.float().unsqueeze(0), l.unsqueeze(0))[0])
        out["top1_eq_hf"] = bool(int(l.argmax()) == int(hf.float().argmax()))
    if seq_logits is not None:
        out["kl_seq"] = _kl(seq_logits, l)
        out["pcc_seq"] = float(_pcc_rows(seq_logits.float().unsqueeze(0), l.unsqueeze(0))[0])
        out["max_abs_vs_seq"] = float((seq_logits.float() - l).abs().max())
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------------------------------------------------


def read_logit_rows(model, tt_logits, rows):
    """``[rows]`` full-vocab fp32 logits of the given row indices of a ``[1, 1, T, V / TP]`` device logits tensor:
    the Generator's readback (32-row tile slice -> ROW_MAJOR -> host -> TP concat -> row % 32)."""
    tp = model.mesh_config.get_config(Mode.PREFILL).tp
    out = []
    for r in rows:
        start = (r // 32) * 32
        tile = ttnn.slice(tt_logits, (0, 0, start, 0), (1, 1, start + 32, tt_logits.shape[-1]))
        rm = ttnn.to_layout(tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tile.deallocate(True)
        host = rm.cpu()
        rm.deallocate(True)
        shards = ttnn.get_device_tensors(host)
        cat = torch.cat([ttnn.to_torch(shards[i]) for i in range(tp)], dim=-1)
        out.append(cat[0, 0, r % 32, : model.vocab_size].float().clone())
    return out


def upload_rows(mesh_device, rows_bf16, hidden_size):
    """``[T, H]`` bf16 host rows -> replicated ``[1, 1, T, H]`` bf16 TILE DRAM tensor."""
    return ttnn.from_torch(
        rows_bf16.reshape(1, 1, -1, hidden_size),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def head_full(model, hidden_tt, rows):
    """norm + lm_head on ALL rows of ``hidden_tt`` (the tt_transformers batched path's head), logits of ``rows``."""
    normed = model.norm(hidden_tt)
    logits = ttnn.matmul(normed, model.lm_head_weight, dtype=ttnn.bfloat16)
    normed.deallocate(True)
    out = read_logit_rows(model, logits, rows)
    logits.deallocate(True)
    return out


def head_seq32(model, hidden_tt, row):
    """The single-user head: slice the 32-row tile holding ``row``, norm + lm_head on it (Model._forward_layers_and_head
    with get_last_token = (row // 32) * 32)."""
    start = (row // 32) * 32
    tile = ttnn.slice(hidden_tt, (0, 0, start, 0), (1, 1, start + 32, hidden_tt.shape[-1]))
    normed = model.norm(tile)
    tile.deallocate(True)
    logits = ttnn.matmul(normed, model.lm_head_weight, dtype=ttnn.bfloat16)
    normed.deallocate(True)
    out = read_logit_rows(model, logits, [row % 32])[0]
    logits.deallocate(True)
    return out


def head_gather(model, hidden_tt, rows):
    """Model._gather_head (phase 3c gather head) on ``rows`` (padded with row 0 to 32)."""
    idx = list(rows) + [0] * (32 - len(rows))
    logits = model._gather_head(hidden_tt, idx)  # consumes hidden_tt
    return model._prefill_head_logits_to_host(logits)[: len(rows)]


def isolate_ops(mesh_device, model, rec, users2, lens, layers=(0, 24, 47), row_counts=(256, 4096)):
    """Identical inputs (arm A's captured block inputs of two users) through each row-wise block of a layer at 128 rows
    (user 0 alone) and at T rows (users 0 + 1 stacked, replicated to T): the per-op effect of the row count on user 0's
    rows, free of everything upstream. Returns ``{layer: {op: {T: metrics}}}``."""
    u0, u1 = users2
    H = model.hf_config.hidden_size
    n0 = lens[u0]

    def rows_of_users(tag, layer):
        return torch.cat([_rows_of(tag, rec.get(f"A{u}", layer, tag), 0, SEQ_LEN) for u in (u0, u1)]).to(torch.bfloat16)

    def run(fn, rows256, T, width, consumes=False):
        inp = rows256[:SEQ_LEN] if T == SEQ_LEN else rows256.repeat(T // (2 * SEQ_LEN), 1)
        x = upload_rows(mesh_device, inp, width)
        y = fn(x)
        host = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
        host = host.reshape(-1, host.shape[-1])[:SEQ_LEN]
        y.deallocate(True)
        if not consumes:
            x.deallocate(True)
        return host

    results = {}
    for L in layers:
        layer = model.layers[L]
        attn = layer.self_attn
        keep_bf16 = attn_prefill.attention_bf16_output(attn.program_config)
        ops = {
            "norm1(resid_in)": (
                rows_of_users("resid_moe", L - 1) if L > 0 else rows_of_users("embed", 0),
                H,
                lambda x: layer.input_layernorm(x),
                False,
            ),
            "qkv(norm1)": (
                rows_of_users("norm1", L),
                H,
                lambda x: attn_prefill.apply_qkv_projection(x, attn.weights),
                False,
            ),
            "oproj(sdpa_concat)": (
                rows_of_users("sdpa_concat", L),
                rows_of_users("sdpa_concat", L).shape[-1],
                lambda x: attn_prefill.apply_output_projection(x, attn.weights, ttnn.bfloat16, keep_bf16=keep_bf16),
                False,
            ),
            "norm2(resid_attn)": (
                rows_of_users("resid_attn", L),
                H,
                lambda x: layer.post_attention_layernorm(x),
                False,
            ),
            "router(norm2)": (rows_of_users("norm2", L), H, lambda x: layer.mlp.router(x, is_decode=False)[1], False),
            "shared(norm2)": (
                rows_of_users("norm2", L),
                H,
                lambda x: layer.mlp.shared_expert(x, is_decode=False),
                False,
            ),
            "mlp_block(norm2)": (rows_of_users("norm2", L), H, None, True),
        }
        table = {}
        for op, (rows256, width, fn, consumes) in ops.items():
            if op.startswith("shared") and layer.mlp.shared_expert is None:
                continue
            per_T = {}
            try:
                if fn is None:

                    def fn_mlp(x, _T=SEQ_LEN):
                        with experts_prefill.packed_prefill_pass(_T > SEQ_LEN):
                            return layer.mlp(x, is_decode=False)

                    base = run(lambda x: fn_mlp(x, SEQ_LEN), rows256, SEQ_LEN, width, consumes=True)
                else:
                    base = run(fn, rows256, SEQ_LEN, width, consumes)
                for T in row_counts:
                    if fn is None:
                        other = run(lambda x, _T=T: fn_mlp(x, _T), rows256, T, width, consumes=True)
                    else:
                        other = run(fn, rows256, T, width, consumes)
                    m = compare_rows(base, other, n0)
                    if op.startswith("router"):
                        m.update(router_flips(base, other, n0))
                    per_T[T] = m
                    logger.info(
                        f"[isolate L{L}] {op:20s} T={T:4d} vs 128: bit_identical {m['bit_identical']} pcc_min {m['pcc_min']:.6f} "
                        f"max|d| {m['max_abs']:.4f} rel_rms {m['rel_rms']:.2e} scale_mean {m['scale_mean']:+.2e} "
                        f"frac_rows {m['frac_rows_diff']:.3f} last_scale {m['last_scale']:+.2e}"
                        + (f" flips {m['rows_set_flipped']}/{m['rows']}" if op.startswith("router") else "")
                    )
                ttnn.synchronize_device(mesh_device)
            except Exception as exc:  # keep the rest of the diagnostic alive
                logger.warning(f"[isolate L{L}] {op} failed: {type(exc).__name__}: {exc}")
                per_T["error"] = f"{type(exc).__name__}: {exc}"[:300]
            table[op] = per_T
        results[L] = table
    return results


def _dev0(t):
    """Device-0 shard of a mesh tensor as fp32 host tensor (exact for bf16 / bfp8 values)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def truth_metrics(truth, dev, n_real):
    """``dev`` (device result, bf16 values) against the fp32 host ``truth``, both ``[rows, D]``, over ``n_real`` rows."""
    truth, dev = truth[:n_real].float(), dev[:n_real].float()
    d = dev - truth
    scale = (dev * truth).sum(-1) / ((truth * truth).sum(-1) + 1e-20) - 1.0
    pcc = _pcc_rows(truth, dev)
    last = n_real - 1
    return {
        "rel_rms": float(d.norm() / (truth.norm() + 1e-20)),
        "scale_mean": float(scale.mean()),
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "pcc_min": float(pcc.min()),
        "max_abs": float(d.abs().max()),
        "last_scale": float(scale[last]),
        "last_rel_rms": float(d[last].norm() / (truth[last].norm() + 1e-20)),
    }


def truth_and_fixes(mesh_device, model, rec, users2, lens, head_results, layers=(0, 24, 47), row_counts=(256, 4096)):
    """fp32 host truth for the row-wise blocks whose result depends on the row count (identical inputs, the device's
    own bfp8 weights), the device result at 128 rows (sequential arm) and at T rows (packed arm), and fix candidates
    (explicit 1D configs / fp32 destination accumulation) against the same truth. Returns ``{layer: {op: {...}}}``."""
    from models.demos.solar_open.tt.experts.operations import apply_glu
    from models.demos.solar_open.tt.linear_configs import mcast_1d_linear_config

    u0, u1 = users2
    n0 = lens[u0]
    H = model.hf_config.hidden_size
    grid = mesh_device.compute_with_storage_grid_size()
    arch = mesh_device.arch()
    hifi2_fp32 = ttnn.init_device_compute_kernel_config(
        arch, math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    hifi4_fp32 = ttnn.init_device_compute_kernel_config(
        arch, math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    def rows_of_users(tag, layer):
        return torch.cat([_rows_of(tag, rec.get(f"A{u}", layer, tag), 0, SEQ_LEN) for u in (u0, u1)]).to(torch.bfloat16)

    def run(fn, rows256, T, width, consumes=False):
        inp = rows256[:SEQ_LEN] if T == SEQ_LEN else rows256.repeat(T // (2 * SEQ_LEN), 1)
        x = upload_rows(mesh_device, inp, width)
        y = fn(x)
        host = _dev0(y)
        host = host.reshape(-1, host.shape[-1])[:SEQ_LEN]
        y.deallocate(True)
        if not consumes:
            x.deallocate(True)
        return host

    def attempt(entry, key, fn, truth, base=None):
        try:
            y = fn()
            entry[key] = truth_metrics(truth, y, n0)
            if base is not None:
                entry[key]["vs_128_bit_identical"] = bool(torch.equal(y[:n0], base[:n0]))
                entry[key]["vs_128_rel_rms"] = float((y[:n0] - base[:n0]).norm() / (base[:n0].norm() + 1e-20))
            m = entry[key]
            logger.info(
                f"[truth] {key:44s} rel_rms {m['rel_rms']:.3e} scale_mean {m['scale_mean']:+.3e} "
                f"[{m['scale_min']:+.2e},{m['scale_max']:+.2e}] pcc_min {m['pcc_min']:.6f} last_scale {m['last_scale']:+.3e}"
                + (
                    f" | vs128 bit {m['vs_128_bit_identical']} rel {m['vs_128_rel_rms']:.2e}"
                    if base is not None
                    else ""
                )
            )
            return y
        except Exception as exc:
            entry[key] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
            logger.warning(f"[truth] {key} failed: {type(exc).__name__}: {exc}")
            return None

    results = {}
    for L in layers:
        layer = model.layers[L]
        attn = layer.self_attn
        se = layer.mlp.shared_expert
        res = {}
        norm1 = rows_of_users("norm1", L)
        norm2 = rows_of_users("norm2", L)
        sc = rows_of_users("sdpa_concat", L)

        # ---- qkv: [T, H] bf16 x [H, 1280] bfp8, auto config (HiFi2, bf16 dest, l1 acc) ------------------------------
        wqkv = _dev0(attn.weights.wqkv).reshape(H, -1)
        qkv_truth = norm1[:SEQ_LEN].float() @ wqkv
        entry = {
            "bf16_rounding_floor_rel_rms": float((qkv_truth.bfloat16().float() - qkv_truth).norm() / qkv_truth.norm())
        }
        base = attempt(
            entry,
            f"L{L} qkv auto T=128",
            lambda: run(lambda x: attn_prefill.apply_qkv_projection(x, attn.weights), norm1, SEQ_LEN, H),
            qkv_truth,
        )
        for T in row_counts:
            attempt(
                entry,
                f"L{L} qkv auto T={T}",
                lambda T=T: run(lambda x: attn_prefill.apply_qkv_projection(x, attn.weights), norm1, T, H),
                qkv_truth,
                base,
            )
        for T in (SEQ_LEN,) + tuple(row_counts):
            attempt(
                entry,
                f"L{L} qkv auto+fp32acc T={T}",
                lambda T=T: run(
                    lambda x: ttnn.linear(x, attn.weights.wqkv, dtype=ttnn.bfloat16, compute_kernel_config=hifi2_fp32),
                    norm1,
                    T,
                    H,
                ),
                qkv_truth,
                base,
            )
        # explicit 1D in0-mcast config, whole K in 4 blocks of 32 tiles, one output tile column per core (Nt = 40 -> 8x5)
        for T in (SEQ_LEN, 256):

            def qkv_explicit(x, T=T):
                cfg = mcast_1d_linear_config((8, 5), T, wqkv.shape[1], H, 32, out_subblock_w=1)
                return ttnn.linear(
                    x,
                    attn.weights.wqkv,
                    dtype=ttnn.bfloat16,
                    program_config=cfg,
                    compute_kernel_config=attn.program_config._hifi2_compute_config(arch, False),
                )

            attempt(
                entry,
                f"L{L} qkv explicit1d(k32) T={T}",
                lambda T=T: run(lambda x: qkv_explicit(x, T), norm1, T, H),
                qkv_truth,
                base,
            )
        res["qkv"] = entry

        # ---- o_proj: typecast bfp8 then [T, 1024] bfp8 x [1024, H] bfp8, auto (LoFi) ----------------------------------
        woproj = _dev0(attn.weights.o_proj).reshape(-1, H)
        x_t = upload_rows(mesh_device, sc[:SEQ_LEN], sc.shape[-1])
        x_b = ttnn.typecast(x_t, ttnn.bfloat8_b)
        sc_bfp8 = _dev0(x_b).reshape(SEQ_LEN, -1)
        x_b.deallocate(True)
        x_t.deallocate(True)
        oproj_truth = sc_bfp8 @ woproj
        entry = {}
        base = attempt(
            entry,
            f"L{L} o_proj auto(LoFi) T=128",
            lambda: run(
                lambda x: attn_prefill.apply_output_projection(x, attn.weights, ttnn.bfloat16, keep_bf16=False),
                sc,
                SEQ_LEN,
                sc.shape[-1],
            ),
            oproj_truth,
        )
        for T in row_counts:
            attempt(
                entry,
                f"L{L} o_proj auto(LoFi) T={T}",
                lambda T=T: run(
                    lambda x: attn_prefill.apply_output_projection(x, attn.weights, ttnn.bfloat16, keep_bf16=False),
                    sc,
                    T,
                    sc.shape[-1],
                ),
                oproj_truth,
                base,
            )
        oproj_truth_bf16in = sc[:SEQ_LEN].float() @ woproj
        for T in (SEQ_LEN,) + tuple(row_counts):
            attempt(
                entry,
                f"L{L} o_proj bf16in auto(HiFi2) T={T} [truth=bf16 in]",
                lambda T=T: run(
                    lambda x: attn_prefill.apply_output_projection(x, attn.weights, ttnn.bfloat16, keep_bf16=True),
                    sc,
                    T,
                    sc.shape[-1],
                ),
                oproj_truth_bf16in,
            )
        res["o_proj"] = entry

        # ---- shared expert: gate / up [T, H] x [H, 160], GLU, down [T, 160] x [160, H]; explicit 1D <= 128 rows, auto above
        if se is not None:
            wg, wu, wd = (_dev0(w).reshape(w.shape[-2], w.shape[-1]) for w in (se.w_gate, se.w_up, se.w_down))
            x0 = norm2[:SEQ_LEN].float()
            gate = (x0 @ wg).bfloat16().float()
            up = (x0 @ wu).bfloat16().float()
            act = (up * torch.nn.functional.silu(gate)).bfloat16().float()
            shared_truth = act @ wd
            entry = {
                "bf16_rounding_floor_rel_rms": float(
                    (shared_truth.bfloat16().float() - shared_truth).norm() / shared_truth.norm()
                )
            }
            base = attempt(
                entry,
                f"L{L} shared production T=128 (explicit1d k32)",
                lambda: run(lambda x: se(x, is_decode=False), norm2, SEQ_LEN, H),
                shared_truth,
            )
            for T in row_counts:
                attempt(
                    entry,
                    f"L{L} shared production T={T} (auto)",
                    lambda T=T: run(lambda x: se(x, is_decode=False), norm2, T, H),
                    shared_truth,
                    base,
                )

            def shared_manual(x, T, explicit, ckc, in0_block_w=32):
                gu_cfg = d_cfg = None
                if explicit:
                    gu_cfg = mcast_1d_linear_config((5, 1), T, se.intermediate_size_per_device, H, in0_block_w, 1)
                    d_cfg = mcast_1d_linear_config((8, 8), T, H, se.intermediate_size_per_device, 5, 2)
                g = ttnn.linear(
                    x,
                    se.w_gate,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=ckc,
                    program_config=gu_cfg,
                )
                u = ttnn.linear(
                    x,
                    se.w_up,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=ckc,
                    program_config=gu_cfg,
                )
                a = apply_glu(g, u)
                g.deallocate(True)
                u.deallocate(True)
                p = ttnn.linear(
                    a,
                    se.w_down,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=ckc,
                    program_config=d_cfg,
                )
                a.deallocate(True)
                return p

            attempt(
                entry,
                f"L{L} shared manual explicit1d(k32) T=128 [=production?]",
                lambda: run(lambda x: shared_manual(x, SEQ_LEN, True, se.compute_kernel_config), norm2, SEQ_LEN, H),
                shared_truth,
                base,
            )
            attempt(
                entry,
                f"L{L} shared explicit1d(k32) T=256",
                lambda: run(lambda x: shared_manual(x, 256, True, se.compute_kernel_config), norm2, 256, H),
                shared_truth,
                base,
            )
            attempt(
                entry,
                f"L{L} shared explicit1d(k16) T=256",
                lambda: run(lambda x: shared_manual(x, 256, True, se.compute_kernel_config, 16), norm2, 256, H),
                shared_truth,
                base,
            )
            for T in (SEQ_LEN, 256):
                attempt(
                    entry,
                    f"L{L} shared auto+fp32acc(HiFi2) T={T}",
                    lambda T=T: run(lambda x: shared_manual(x, T, False, hifi2_fp32), norm2, T, H),
                    shared_truth,
                    base,
                )
                attempt(
                    entry,
                    f"L{L} shared explicit1d(k32)+fp32acc T={T}",
                    lambda T=T: run(lambda x: shared_manual(x, T, True, hifi2_fp32), norm2, T, H),
                    shared_truth,
                    base,
                )
            attempt(
                entry,
                f"L{L} shared auto+HiFi4+fp32acc T=256",
                lambda: run(lambda x: shared_manual(x, 256, False, hifi4_fp32), norm2, 256, H),
                shared_truth,
                base,
            )
            res["shared"] = entry

        # ---- routed experts (dense bmm path): T = 128 vs 256 must be bit-identical per row ----------------------------
        entry = {}

        def routed(x, T):
            with experts_prefill.packed_prefill_pass(T > SEQ_LEN):
                _idx, dense = layer.mlp.router(x, is_decode=False)
                return layer.mlp.experts(x, topk_expert_weights=dense, is_decode=False, shared_expert=None)

        try:
            r128 = run(lambda x: routed(x, SEQ_LEN), norm2, SEQ_LEN, H, consumes=True)
            for T in row_counts:
                rT = run(lambda x, T=T: routed(x, T), norm2, T, H, consumes=True)
                entry[f"L{L} routed T={T} vs 128"] = compare_rows(r128, rT, n0)
                m = entry[f"L{L} routed T={T} vs 128"]
                logger.info(
                    f"[truth] L{L} routed experts (no shared) T={T} vs 128: bit_identical {m['bit_identical']} rel_rms {m['rel_rms']:.2e} scale_mean {m['scale_mean']:+.2e} pcc_min {m['pcc_min']:.6f}"
                )
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"[:400]
            logger.warning(f"[truth] routed experts isolation failed: {exc}")
        res["routed"] = entry

        # ---- routed experts vs fp32 truth (device-0 partial, no all-reduce): dense bmm at 128 rows vs the sorted
        # hot / cold path at 1024 rows (per-split plan) and 4096 rows (per-chunk plan of a packed pass) --------------
        if os.getenv("SOLAR_OPEN_BISECT_ROUTED_TRUTH", "0") == "1":
            entry = {}
            try:
                ex = layer.mlp.experts
                x0 = norm2[:SEQ_LEN].float()
                x_t = upload_rows(mesh_device, norm2[:SEQ_LEN], H)
                _idx, dense_t = layer.mlp.router(x_t, is_decode=False)
                dense_host = _dev0(dense_t)[:SEQ_LEN]  # [128, E] routing weights (bf16 values)
                dense_t.deallocate(True)
                x_t.deallocate(True)
                gu = _dev0(ex.weights.gate_up_proj)[0]  # [E, H, 2 * Ip] device-0 shard
                dn = _dev0(
                    ex.weights.down_proj_padded if ex.weights.down_proj_padded is not None else ex.weights.down_proj
                )[0]
                ip = ex.weights.intermediate_padded_per_device
                truth = torch.zeros(SEQ_LEN, H)
                for r in range(n0):
                    for e in torch.nonzero(dense_host[r]).flatten().tolist():
                        g_u = x0[r] @ gu[e]  # [2 Ip]
                        a = torch.nn.functional.silu(g_u[:ip]) * g_u[ip:]
                        truth[r] += float(dense_host[r, e]) * (a @ dn[e])
                del gu, dn
                entry["truth_note"] = "fp32, no bfp8 emulation of the gate|up / down outputs; device-0 partial"

                def routed_partial(x, T):
                    with experts_prefill.packed_prefill_pass(T > 1024):
                        _i, dense = layer.mlp.router(x, is_decode=False)
                        return experts_prefill._process_prefill_chunk(
                            x,
                            dense,
                            ex.weights,
                            ex.config,
                            ex.prefill_sparsity,
                            ex.program_config,
                            1,
                            model.mesh_config.get_config(Mode.PREFILL).tp,
                            dense_core_grid=experts_prefill._dense_core_grid(
                                mesh_device, ex.program_config.dense_grid_max_width
                            ),
                        )

                base = attempt(
                    entry,
                    f"L{L} routed dense bmm T=128",
                    lambda: run(lambda x: routed_partial(x, SEQ_LEN), norm2, SEQ_LEN, H, consumes=True),
                    truth,
                )
                for T in (256, 1024, 4096):
                    attempt(
                        entry,
                        f"L{L} routed T={T} ({'dense bmm' if T <= 256 else 'sorted, per-split plan' if T == 1024 else 'sorted, per-chunk plan (packed)'})",
                        lambda T=T: run(lambda x, T=T: routed_partial(x, T), norm2, T, H, consumes=True),
                        truth,
                        base,
                    )
                    logger.info(
                        f"[truth] L{L} routed T={T}: plan {dict(experts_prefill.LAST_SORTED_MOE_PLAN)} path {dict(experts_prefill.LAST_PREFILL_MOE_PATH)}"
                    )
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"[:400]
                logger.warning(f"[truth] routed truth failed: {type(exc).__name__}: {exc}")
            res["routed_truth"] = entry
        ttnn.synchronize_device(mesh_device)
        results[L] = res

    # ---- lm_head: normed last row x [H, 32768] bfp8 (device-0 vocab shard 0..32767 holds tokens 22 / 23) ----------------
    entry = {}
    try:
        w_lm = _dev0(model.lm_head_weight).reshape(H, -1)  # [H, V8]
        V8 = w_lm.shape[1]
        final = _rows_of("resid_moe", rec.get(f"A{u0}", len(model.layers) - 1, "resid_moe"), 0, SEQ_LEN).to(
            torch.bfloat16
        )
        x = upload_rows(mesh_device, final, H)
        normed = model.norm(x)
        normed_host = _dev0(normed).reshape(SEQ_LEN, H)
        li = n0 - 1
        truth = (normed_host[li : li + 1] @ w_lm)[0]  # [V8] fp32
        entry["truth_gap_22_23"] = float(truth[THINK] - truth[CONTENT])
        entry["truth_logit_22"], entry["truth_logit_23"] = float(truth[THINK]), float(truth[CONTENT])
        for variant, lg in head_results[f"A{u0}"].items():
            dev = lg[:V8]
            m = truth_metrics(truth.unsqueeze(0), dev.unsqueeze(0), 1)
            m["gap_22_23"] = float(dev[THINK] - dev[CONTENT])
            m["top1_shard0"] = int(dev.argmax())
            entry[f"head {variant}"] = m
            logger.info(
                f"[truth] lm_head {variant:10s} vs fp32 truth: rel_rms {m['rel_rms']:.3e} scale {m['scale_mean']:+.3e} gap(22,23) {m['gap_22_23']:+.3f} (truth {entry['truth_gap_22_23']:+.3f})"
            )
        # fix candidate: fp32 destination accumulation in the head at M = 32 and M = 4096
        for T in (32, 4096):
            if T == 32:
                start = (li // 32) * 32
                tile = ttnn.slice(normed, (0, 0, start, 0), (1, 1, start + 32, H))
                lg_t = ttnn.matmul(tile, model.lm_head_weight, dtype=ttnn.bfloat16, compute_kernel_config=hifi2_fp32)
                tile.deallocate(True)
                dev = read_logit_rows(model, lg_t, [li % 32])[0][:V8]
            else:
                big = upload_rows(
                    mesh_device, _dev0(normed).reshape(SEQ_LEN, H).to(torch.bfloat16).repeat(T // SEQ_LEN, 1), H
                )
                lg_t = ttnn.matmul(big, model.lm_head_weight, dtype=ttnn.bfloat16, compute_kernel_config=hifi2_fp32)
                big.deallocate(True)
                dev = read_logit_rows(model, lg_t, [li])[0][:V8]
            lg_t.deallocate(True)
            m = truth_metrics(truth.unsqueeze(0), dev.unsqueeze(0), 1)
            m["gap_22_23"] = float(dev[THINK] - dev[CONTENT])
            entry[f"head fp32acc M={T}"] = m
            logger.info(
                f"[truth] lm_head fp32acc M={T}: rel_rms {m['rel_rms']:.3e} scale {m['scale_mean']:+.3e} gap(22,23) {m['gap_22_23']:+.3f}"
            )
        normed.deallocate(True)
        x.deallocate(True)
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"[:400]
        logger.warning(f"[truth] lm_head truth failed: {exc}")
    results["lm_head"] = entry
    return results


def verify_fix_levers(mesh_device, model, rec, users2, lens):
    """Phase 3g / D2: every lever of the "sequential numerics" knob on IDENTICAL inputs (arm A's captured block inputs)
    against the 128-row production result -- bit-identity is the pass criterion -- plus the single-launch shared-gate
    alternative (1D in0_block_w 32 with out_block_h chunking at 4096 rows) for the next stage, and E0: the two final-norm
    kernels (width-sharded decode kernel on a 32-row tile, default kernel on 128 rows) against a host fp32 RMSNorm."""
    u0, u1 = users2
    n0 = lens[u0]
    H = model.hf_config.hidden_size
    grid = mesh_device.compute_with_storage_grid_size()
    arch = mesh_device.arch()
    layer = model.layers[0]
    attn = layer.self_attn
    se = layer.mlp.shared_expert
    out = {}

    def rows_of_users(tag, layer_idx):
        return torch.cat([_rows_of(tag, rec.get(f"A{u}", layer_idx, tag), 0, SEQ_LEN) for u in (u0, u1)]).to(
            torch.bfloat16
        )

    def dev_rows(y, width):
        return _dev0(y).reshape(-1, width)

    def record(name, ref, got, nrows=None):
        nrows = n0 if nrows is None else nrows
        m = compare_rows(ref[:nrows], got[:nrows], nrows)
        out[name] = m
        logger.info(
            f"[fix verify] {name:58s} bit_identical {m['bit_identical']} rel_rms {m['rel_rms']:.2e} "
            f"scale {m['scale_mean']:+.2e} max|d| {m['max_abs']:.4f}"
        )
        return m

    # qkv: production 128 rows (auto 1D k=2) vs the knob's 2D k=2 configs at 4096 / 512 / 256 rows
    norm1 = rows_of_users("norm1", 0)
    x128 = upload_rows(mesh_device, norm1[:SEQ_LEN], H)
    ref = dev_rows(attn_prefill.apply_qkv_projection(x128, attn.weights), attn.weights.wqkv.shape[-1])
    x128.deallocate(True)
    for T in (256, 512, 4096):
        x = upload_rows(mesh_device, norm1.repeat(T // (2 * SEQ_LEN), 1), H)
        pc, ckc = packed_numerics.seq_numerics_matmul(
            T, SEQ_LEN, int(attn.weights.wqkv.shape[-1]), H, grid, arch, ttnn.MathFidelity.HiFi2
        )
        try:
            y = attn_prefill.apply_qkv_projection(x, attn.weights, program_config=pc, compute_kernel_config=ckc)
            record(f"qkv knob 2D k=2 T={T} vs production 128", ref, dev_rows(y, ref.shape[-1]))
            y.deallocate(True)
        except Exception as exc:
            out[f"qkv knob T={T}"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            logger.warning(f"[fix verify] qkv knob T={T} failed: {exc}")
        x.deallocate(True)

    # o_proj (bfp8 typecast, LoFi): production 128 vs the knob's 2D k=2 at 512 / 4096
    sc = rows_of_users("sdpa_concat", 0)
    keep_bf16 = attn_prefill.attention_bf16_output(attn.program_config)
    x128 = upload_rows(mesh_device, sc[:SEQ_LEN], sc.shape[-1])
    ref = dev_rows(attn_prefill.apply_output_projection(x128, attn.weights, ttnn.bfloat16, keep_bf16=keep_bf16), H)
    x128.deallocate(True)
    for T in (512, 4096):
        x = upload_rows(mesh_device, sc.repeat(T // (2 * SEQ_LEN), 1), sc.shape[-1])
        pc, ckc = packed_numerics.seq_numerics_matmul(
            T,
            SEQ_LEN,
            H,
            int(attn.weights.o_proj.shape[-2]),
            grid,
            arch,
            ttnn.MathFidelity.HiFi2 if keep_bf16 else ttnn.MathFidelity.LoFi,
        )
        try:
            y = attn_prefill.apply_output_projection(
                x, attn.weights, ttnn.bfloat16, keep_bf16=keep_bf16, program_config=pc, compute_kernel_config=ckc
            )
            record(f"o_proj knob 2D k=2 T={T} vs production 128", ref, dev_rows(y, H))
            y.deallocate(True)
        except Exception as exc:
            out[f"o_proj knob T={T}"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            logger.warning(f"[fix verify] o_proj knob T={T} failed: {exc}")
        x.deallocate(True)

    # shared expert: the module under the knob (T = 4096 in 128-row pieces) vs its 128-row production result, and the
    # single-launch alternative (1D k=32, per_core_M 128, out_block_h 4)
    if se is not None:
        norm2 = rows_of_users("norm2", 0)
        x128 = upload_rows(mesh_device, norm2[:SEQ_LEN], H)
        ref = dev_rows(se(x128, is_decode=False), H)
        x128.deallocate(True)
        for T in (256, 4096):
            x = upload_rows(mesh_device, norm2.repeat(T // (2 * SEQ_LEN), 1), H)
            with packed_prefill.packed_seq_numerics(1), experts_prefill.packed_prefill_pass(True, seq_len=SEQ_LEN):
                y = se(x, is_decode=False)
            record(f"shared expert knob pieces T={T} vs production 128", ref, dev_rows(y, H))
            y.deallocate(True)
            x.deallocate(True)
        x128 = upload_rows(mesh_device, norm2[:SEQ_LEN], H)
        g_ref = dev_rows(
            ttnn.linear(
                x128,
                se.w_gate,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=se.compute_kernel_config,
                program_config=se._get_program_configs(SEQ_LEN)[0],
            ),
            se.intermediate_size_per_device,
        )
        x128.deallocate(True)
        for T, out_block_h in ((4096, 4), (4096, 8), (1024, 4)):
            x = upload_rows(mesh_device, norm2.repeat(T // (2 * SEQ_LEN), 1), H)
            cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(5, 1),
                in0_block_w=32,
                out_subblock_h=1,
                out_subblock_w=1,
                out_block_h=out_block_h,
                out_block_w=1,
                per_core_M=T // 32,
                per_core_N=1,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )
            try:
                y = ttnn.linear(
                    x,
                    se.w_gate,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=se.compute_kernel_config,
                    program_config=cfg,
                )
                record(
                    f"shared gate 1D k=32 pcM={T // 32} out_block_h={out_block_h} vs production 128",
                    g_ref,
                    dev_rows(y, se.intermediate_size_per_device),
                )
                y.deallocate(True)
            except Exception as exc:
                out[f"shared gate 1D k=32 T={T} out_block_h={out_block_h}"] = {
                    "error": f"{type(exc).__name__}: {exc}"[:300]
                }
                logger.info(f"[fix verify] shared gate single-launch T={T} out_block_h={out_block_h}: {str(exc)[:200]}")
            x.deallocate(True)

    # head: lm_head at 1024 rows (auto) vs 32 rows (auto) on the same normed rows; the knob's head vs the seq32 head
    final = _rows_of("resid_moe", rec.get(f"A{u0}", len(model.layers) - 1, "resid_moe"), 0, SEQ_LEN).to(torch.bfloat16)
    li = n0 - 1
    tile0 = (li // 32) * 32
    x = upload_rows(mesh_device, final, H)
    normed = model.norm(x)
    x.deallocate(True)
    normed_host = _dev0(normed).reshape(SEQ_LEN, H).to(torch.bfloat16)
    normed.deallocate(True)
    x32 = upload_rows(mesh_device, normed_host[tile0 : tile0 + 32], H)
    lm_ref = dev_rows(ttnn.matmul(x32, model.lm_head_weight, dtype=ttnn.bfloat16), model.lm_head_weight.shape[-1])
    x32.deallocate(True)
    for T in (1024, 2048):
        x = upload_rows(mesh_device, normed_host.repeat(T // SEQ_LEN, 1), H)
        y = ttnn.matmul(x, model.lm_head_weight, dtype=ttnn.bfloat16)
        got = dev_rows(y, model.lm_head_weight.shape[-1])[tile0 : tile0 + 32]
        record(f"lm_head auto M={T} vs auto M=32 (same rows)", lm_ref, got, nrows=32)
        y.deallocate(True)
        x.deallocate(True)
    # the knob's head vs the sequential head (norm of the 32-row TILE with the sharded kernel + lm_head at M = 32)
    x32 = upload_rows(mesh_device, final[tile0 : tile0 + 32], H)
    seq_head_ref = dev_rows(model._norm_and_lm_head(x32), model.lm_head_weight.shape[-1])
    x32.deallocate(True)
    for T in (256, 4096):
        x = upload_rows(mesh_device, final.repeat(T // SEQ_LEN, 1), H)
        with packed_prefill.packed_seq_numerics(1), experts_prefill.packed_prefill_pass(True, seq_len=SEQ_LEN):
            y = model._norm_and_lm_head(x)  # consumes x
        got = dev_rows(y, model.lm_head_weight.shape[-1])[tile0 : tile0 + 32]
        record(f"knob head T={T} vs seq32 head (same rows)", seq_head_ref, got, nrows=32)
        record(f"knob head T={T} vs default-norm head M=32 (same rows)", lm_ref, got, nrows=32)
        y.deallocate(True)

    # E0: which final-norm kernel is closer to fp32 on the last tile of user 0's final residual
    w = _dev0(model.norm.tt_weight).reshape(-1)  # [1, 1, H / 32, 32] bf16 ROW_MAJOR, replicated
    xf = final[tile0 : tile0 + 32].float()
    truth = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + model.norm.eps) * w
    x32 = upload_rows(mesh_device, final[tile0 : tile0 + 32], H)
    sharded = dev_rows(model.norm(x32), H)  # [1, 1, 32, H] -> the width-sharded decode kernel (decode_norm_applies)
    x32.deallocate(True)
    x128 = upload_rows(mesh_device, final, H)
    default = dev_rows(model.norm(x128), H)[tile0 : tile0 + 32]
    x128.deallocate(True)
    for name, got in (
        ("final norm: sharded decode kernel (32-row tile)", sharded),
        ("final norm: default kernel (128 rows)", default),
    ):
        m = truth_metrics(truth, got, 32)
        m_last = truth_metrics(truth[li - tile0 : li - tile0 + 1], got[li - tile0 : li - tile0 + 1], 1)
        m["last_row_rel_rms"], m["last_row_scale"] = m_last["rel_rms"], m_last["scale_mean"]
        out[name] = m
        logger.info(
            f"[fix verify / E0] {name:52s} vs fp32 RMSNorm: rel_rms {m['rel_rms']:.3e} scale {m['scale_mean']:+.3e} "
            f"pcc_min {m['pcc_min']:.6f}; last row rel_rms {m_last['rel_rms']:.3e} scale {m_last['scale_mean']:+.3e}"
        )
    out["final norm: sharded vs default kernel"] = compare_rows(default, sharded, 32)
    logger.info(
        f"[fix verify / E0] sharded vs default kernel on the same 32 rows: rel_rms "
        f"{out['final norm: sharded vs default kernel']['rel_rms']:.3e} scale "
        f"{out['final norm: sharded vs default kernel']['scale_mean']:+.3e}"
    )
    ttnn.synchronize_device(mesh_device)
    return out


def verify_auto_configs(mesh_device, model, rec, users2, lens):
    """Identify ttnn's auto program configs by BIT-IDENTITY: the model's matmuls (auto config) at the sequential and
    packed row counts against explicit candidate configs derived from matmul_program_config.cpp (1D systolic for
    narrow shapes: in0_block_w = 2, mcast_in0; 2D mcast for the rest: in0_block_w = Kt % 11 ? 1 : Kt / 11 on the 11x10
    grid); a sweep of in0_block_w for the 1D form maps the K-block length to the magnitude bias against fp32. Every
    candidate at the packed row count is also compared with the SEQUENTIAL row count's production result (row
    independence: a config that reproduces it bit for bit is a packed == sequential lever)."""
    u0, u1 = users2
    n0 = lens[u0]
    H = model.hf_config.hidden_size
    arch = mesh_device.arch()
    hifi2 = ttnn.init_device_compute_kernel_config(
        arch, math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    lofi = ttnn.init_device_compute_kernel_config(
        arch, math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    layer = model.layers[0]
    attn = layer.self_attn
    se = layer.mlp.shared_expert
    out = {}

    def rows_of_users(tag, layer_idx):
        return torch.cat([_rows_of(tag, rec.get(f"A{u}", layer_idx, tag), 0, SEQ_LEN) for u in (u0, u1)]).to(
            torch.bfloat16
        )

    def cfg_1d(k, per_core_M, per_core_N, sub_w=1):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            in0_block_w=k,
            out_subblock_h=1,
            out_subblock_w=sub_w,
            out_block_h=per_core_M,
            out_block_w=per_core_N,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def cfg_2d(k, per_core_M, per_core_N, sub_w):
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            in0_block_w=k,
            out_subblock_h=1,
            out_subblock_w=sub_w,
            out_block_h=per_core_M,
            out_block_w=per_core_N,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
        )

    def compare_all(name, x, weight, ref, candidates, truth, ckc, ref_other, nrows):
        entry = {}
        if truth is not None:
            entry["auto_vs_truth"] = truth_metrics(truth, ref, nrows)
        if ref_other is not None:
            ot, sl = ref_other
            entry["auto_vs_ref_other"] = {
                "bit_identical": bool(torch.equal(ref[sl], ot)),
                "rel_rms": float((ref[sl] - ot).norm() / (ot.norm() + 1e-20)),
            }
            logger.info(
                f"[configs] {name} :: auto vs the other row count's production result: bit-identical "
                f"{entry['auto_vs_ref_other']['bit_identical']} rel {entry['auto_vs_ref_other']['rel_rms']:.2e}"
            )
        for cname, cfg in candidates.items():
            try:
                y = ttnn.matmul(x, weight, dtype=ttnn.bfloat16, program_config=cfg, compute_kernel_config=ckc)
                got = _dev0(y).reshape(-1, weight.shape[-1])[:nrows]
                y.deallocate(True)
                m = {
                    "bit_identical_to_auto": bool(torch.equal(got, ref)),
                    "rel_rms_vs_auto": float((got - ref).norm() / (ref.norm() + 1e-20)),
                }
                if truth is not None:
                    tm = truth_metrics(truth, got, nrows)
                    m["truth_rel_rms"], m["truth_scale_mean"] = tm["rel_rms"], tm["scale_mean"]
                if ref_other is not None:
                    ot, sl = ref_other
                    m["bit_identical_to_ref_other"] = bool(torch.equal(got[sl], ot))
                    m["rel_rms_vs_ref_other"] = float((got[sl] - ot).norm() / (ot.norm() + 1e-20))
                entry[cname] = m
                logger.info(
                    f"[configs] {name} :: {cname:46s} == auto {m['bit_identical_to_auto']} rel {m['rel_rms_vs_auto']:.2e}"
                    + (
                        f" | vs truth rel {m['truth_rel_rms']:.2e} scale {m['truth_scale_mean']:+.2e}"
                        if truth is not None
                        else ""
                    )
                    + (
                        f" | == other-rowcount {m['bit_identical_to_ref_other']} rel {m['rel_rms_vs_ref_other']:.2e}"
                        if ref_other is not None
                        else ""
                    )
                )
            except Exception as exc:
                entry[cname] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                logger.info(f"[configs] {name} :: {cname}: {type(exc).__name__}: {str(exc)[:160]}")
        return entry

    def check(name, x_rows, width, weight, auto_fn, candidates, truth=None, ckc=hifi2, ref_other=None, rows=None):
        nrows = n0 if rows is None else rows
        x = upload_rows(mesh_device, x_rows, width)
        ref = _dev0(auto_fn(x)).reshape(-1, weight.shape[-1])[:nrows]
        entry = compare_all(name, x, weight, ref, candidates, truth, ckc, ref_other, nrows)
        x.deallocate(True)
        return entry, ref

    # ---- qkv: M=128 narrow (128 vs 1280 > 8x) -> 1D systolic mcast_in0 pcM 4 pcN 1 k 2; M=256 (5x) -> 2D k 1 pcM 1 pcN 4
    norm1 = rows_of_users("norm1", 0)
    wqkv = _dev0(attn.weights.wqkv).reshape(H, -1)
    qkv_truth = norm1[:SEQ_LEN].float() @ wqkv
    out["qkv M=128"], qkv_ref128 = check(
        "qkv M=128",
        norm1[:SEQ_LEN],
        H,
        attn.weights.wqkv,
        lambda x: attn_prefill.apply_qkv_projection(x, attn.weights),
        {f"1D mcast_in0 k={k} pcM=4 pcN=1": cfg_1d(k, 4, 1) for k in (1, 2, 4, 8, 16, 32, 64, 128)}
        | {"2D k=1 pcM=1 pcN=4": cfg_2d(1, 1, 4, 4)},
        truth=qkv_truth,
    )
    out["qkv M=256"], _ = check(
        "qkv M=256",
        norm1,
        H,
        attn.weights.wqkv,
        lambda x: attn_prefill.apply_qkv_projection(x, attn.weights),
        {f"2D k={k} pcM=1 pcN=4": cfg_2d(k, 1, 4, 4) for k in (1, 2, 4)}
        | {f"1D mcast_in0 k={k} pcM=8 pcN=1": cfg_1d(k, 8, 1) for k in (1, 2, 4)},
        truth=qkv_truth,
        ref_other=(qkv_ref128, slice(0, n0)),
    )

    # ---- o_proj (bfp8 x bfp8, LoFi): M <= 256 narrow -> 1D k 2 pcN 2; M >= 1024 -> 2D k 1 -------------------------------
    sc = rows_of_users("sdpa_concat", 0)
    woproj = _dev0(attn.weights.o_proj).reshape(-1, H)

    def oproj_check(name, rows, cands, truth, ref_other=None):
        x = upload_rows(mesh_device, rows, sc.shape[-1])
        xb = ttnn.typecast(x, ttnn.bfloat8_b)
        x.deallocate(True)
        ref = _dev0(ttnn.matmul(xb, attn.weights.o_proj, dtype=ttnn.bfloat16)).reshape(-1, H)[:n0]
        entry = compare_all(name, xb, attn.weights.o_proj, ref, cands, truth, lofi, ref_other, n0)
        xb.deallocate(True)
        return entry, ref

    x_t = upload_rows(mesh_device, sc[:SEQ_LEN], sc.shape[-1])
    x_b = ttnn.typecast(x_t, ttnn.bfloat8_b)
    sc_bfp8 = _dev0(x_b).reshape(-1, sc.shape[-1])
    x_b.deallocate(True)
    x_t.deallocate(True)
    oproj_truth = sc_bfp8[:SEQ_LEN] @ woproj
    out["o_proj M=128"], oproj_ref128 = oproj_check(
        "o_proj M=128",
        sc[:SEQ_LEN],
        {f"1D mcast_in0 k={k} pcM=4 pcN=2": cfg_1d(k, 4, 2, 2) for k in (1, 2, 4, 8, 32)},
        oproj_truth,
    )
    out["o_proj M=4096"], _ = oproj_check(
        "o_proj M=4096",
        sc.repeat(16, 1),
        {f"2D k={k} pcM=13 pcN=12": cfg_2d(k, 13, 12, 4) for k in (1, 2)},
        oproj_truth,
        ref_other=(oproj_ref128, slice(0, n0)),
    )

    # ---- lm_head: M=32 (sequential / gather head) vs M=128..1024 (full head) vs M=4096 -----------------------------------
    final = _rows_of("resid_moe", rec.get(f"A{u0}", len(model.layers) - 1, "resid_moe"), 0, SEQ_LEN).to(torch.bfloat16)
    x = upload_rows(mesh_device, final, H)
    normed = model.norm(x)
    normed_host = _dev0(normed).reshape(SEQ_LEN, H).to(torch.bfloat16)
    normed.deallocate(True)
    x.deallocate(True)
    w_lm = _dev0(model.lm_head_weight).reshape(H, -1)
    lm_truth = normed_host.float() @ w_lm
    li = n0 - 1
    tile0 = (li // 32) * 32
    lm_cands_32 = {f"1D mcast_in0 k={k} pcM=1 pcN=10": cfg_1d(k, 1, 10, 2) for k in (1, 2, 4, 8, 16, 32, 64, 128)}
    lm_cands_32["1D mcast_in0 k=2 pcM=1 pcN=10 subblock_w=1"] = cfg_1d(2, 1, 10, 1)
    lm_cands_32["1D mcast_in0 k=2 pcM=1 pcN=5"] = cfg_1d(2, 1, 5, 1)
    lm_cands_32["1D mcast_in0 k=2 pcM=1 pcN=20"] = cfg_1d(2, 1, 20, 2)
    lm_cands_32["2D k=1 pcM=1 pcN=94"] = cfg_2d(1, 1, 94, 2)
    out["lm_head M=32"], lm_ref32 = check(
        "lm_head M=32",
        normed_host[tile0 : tile0 + 32],
        H,
        model.lm_head_weight,
        lambda x: ttnn.matmul(x, model.lm_head_weight, dtype=ttnn.bfloat16),
        lm_cands_32,
        truth=lm_truth[tile0 : tile0 + 32],
        rows=32,
    )
    out["lm_head M=128"], _ = check(
        "lm_head M=128",
        normed_host,
        H,
        model.lm_head_weight,
        lambda x: ttnn.matmul(x, model.lm_head_weight, dtype=ttnn.bfloat16),
        {f"1D mcast_in0 k={k} pcM=4 pcN=10": cfg_1d(k, 4, 10, 2) for k in (1, 2, 4)}
        | {"1D mcast_in0 k=2 pcM=1 pcN=10 (the M=32 form)": cfg_1d(2, 1, 10, 2)},
        truth=lm_truth,
        ref_other=(lm_ref32, slice(tile0, tile0 + 32)),
        rows=SEQ_LEN,
    )

    # ---- shared gate (auto at M=256 with HiFi2 given) vs the production explicit 1D k=32 config at 128 rows ------------
    if se is not None:
        norm2 = rows_of_users("norm2", 0)
        wg = _dev0(se.w_gate).reshape(H, -1)
        g_truth = norm2[:SEQ_LEN].float() @ wg
        x128 = upload_rows(mesh_device, norm2[:SEQ_LEN], H)
        g_prod = _dev0(
            ttnn.linear(
                x128,
                se.w_gate,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=se.compute_kernel_config,
                program_config=se._get_program_configs(SEQ_LEN)[0],
            )
        ).reshape(-1, wg.shape[1])[:n0]
        x128.deallocate(True)
        out["shared gate M=256"], _ = check(
            "shared gate M=256",
            norm2,
            H,
            se.w_gate,
            lambda x: ttnn.linear(
                x,
                se.w_gate,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=se.compute_kernel_config,
            ),
            {f"2D k={k} pcM=1 pcN=1": cfg_2d(k, 1, 1, 1) for k in (1, 2, 4)}
            | {f"1D mcast_in0 (5 cores) k={k} pcM=8 pcN=1": cfg_1d(k, 8, 1) for k in (2, 4, 8, 16, 32)},
            truth=g_truth,
            ckc=se.compute_kernel_config,
            ref_other=(g_prod, slice(0, n0)),
        )
    ttnn.synchronize_device(mesh_device)
    return out


def probe_auto_matmul_configs(mesh_device, model, out_dir):
    """Best effort: graph-capture standalone matmuls of the model's row-wise shapes at the arms' row counts and dump
    the captured nodes (the chosen program config is in the op's attributes when the tracker records them)."""
    layer0 = model.layers[0]
    wqkv = layer0.self_attn.weights.wqkv
    o_proj = layer0.self_attn.weights.o_proj
    H = model.hf_config.hidden_size
    results = {}
    for name, weight, in_dtype, k in (
        ("qkv", wqkv, ttnn.bfloat16, H),
        ("o_proj", o_proj, ttnn.bfloat8_b, None),
        (
            "lm_head",
            model.lm_head_weight,
            ttnn.bfloat16,
            H,
        ),
    ):
        k = k or int(weight.shape[-2])
        for m in (32, 64, 128, 256, 1024, 4096):
            x = ttnn.from_torch(
                torch.randn(1, 1, m, k) * 0.1,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=in_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            try:
                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                y = ttnn.matmul(x, weight, dtype=ttnn.bfloat16)
                captured = ttnn.graph.end_graph_capture()
                y.deallocate(True)
                text = json.dumps(captured, default=str)
                hits = []
                for node in captured:
                    s = json.dumps(node, default=str)
                    if "rogram" in s and ("Matmul" in s or "matmul" in s):
                        hits.append(s[:1600])
                results[f"{name}_m{m}"] = hits[:3] if hits else ["<no program_config text in captured nodes>"]
                (out_dir / f"graph_{name}_m{m}.json").write_text(text[:200000])
            except Exception as exc:  # the probe must never sink the run
                try:
                    ttnn.graph.end_graph_capture()
                except Exception:
                    pass
                results[f"{name}_m{m}"] = [f"probe failed: {type(exc).__name__}: {exc}"[:400]]
            x.deallocate(True)
    return results


# ---------------------------------------------------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------------------------------------------------


def _load_hf_reference(tokens, lens, users):
    path = default_prefill_reference_path()
    if not path.exists():
        logger.warning(f"no HF prefill reference at {path}; HF columns will be empty")
        return None
    ref = torch.load(path, weights_only=False)
    meta = ref["meta"]
    out = {}
    for u in users:
        entry = ref["prompts"][u]
        ids = tokens[u, : lens[u]].tolist()
        if ids != entry["prompt_ids"].tolist():
            raise AssertionError(f"user {u}: token ids differ from the HF reference ({meta})")
        out[u] = entry["logits"].float()
    logger.info(f"HF reference {path}: date {meta['date_string']}, effort {meta['reasoning_effort']}")
    return out


def _fmt(v):
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:.3e}" if abs(v) < 1e-3 and v != 0 else f"{v:.5f}"
    return str(v)


@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric([(1, 8)])
def test_packed_bias_bisect(mesh_device, device_params, state_dict, pinned_template_date, monkeypatch):
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape[0] != 1 or mesh_shape[1] != 8:
        pytest.skip(f"sized for the 1x8 TP=8 mesh, got {mesh_shape}")
    from models.demos.solar_open.demo.text_demo import prepare_solar_open_generator_args
    from models.tt_transformers.tt.generator import Generator

    out_dir = Path(os.getenv(OUT_DIR_ENV, DEFAULT_OUT))
    out_dir.mkdir(parents=True, exist_ok=True)
    users = [int(u) for u in os.getenv("SOLAR_OPEN_BISECT_USERS", "0,1,12,19").split(",")]
    passes = [
        tuple(int(u) for u in p.split("+")) for p in os.getenv("SOLAR_OPEN_BISECT_PASSES", "0+1,12+19,0+0").split(",")
    ]
    head_ts = [int(t) for t in os.getenv("SOLAR_OPEN_BISECT_HEAD_T", "256,1024,4096").split(",")]
    for p in passes:
        assert len(p) == 2, f"passes are pairs of users (T = 256), got {p}"

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    batch_size, max_seq_len, block_size = 32, 4096, 64
    page_params = {
        "page_block_size": block_size,
        "page_max_num_blocks_per_dp": batch_size * (max_seq_len // block_size),
    }
    model_args, models, page_table, tt_kv_cache, tokenizer, _processor, _paged_cfg = prepare_solar_open_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=batch_size,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=setup["mesh_config"],
        state_dict=state_dict,
        users_row_sharded=False,
    )
    model = models[0]
    generator = Generator(models, model_args, mesh_device, processor=None, tokenizer=tokenizer)
    vocab = model_args[0].vocab_size
    H = model.hf_config.hidden_size

    with open(PROMPTS_128) as f:
        prompts = [e["prompt"] for e in json.load(f)][:batch_size]
    input_tokens, _enc, decoding_pos, _plens = preprocess_inputs_prefill(
        prompts, tokenizer, model_args, instruct=False, max_generated_tokens=64, max_prefill_len=max_seq_len
    )
    tokens = torch.stack(input_tokens).view(batch_size, -1)
    lens = [int(n) for n in decoding_pos]
    assert all(get_padded_prefill_len(n) == SEQ_LEN for n in lens), lens
    logger.info(
        f"users {users}: prompt lengths {[lens[u] for u in users]}; passes {passes}; template date {pinned_template_date}"
    )
    hf = _load_hf_reference(tokens, lens, sorted(set(users) | {u for p in passes for u in p}))

    rec = Recorder()
    install_hooks(monkeypatch, rec, model)

    def clear_kv():
        for m in models:
            m.clear_kv_caches()
        generator.prev_page_table = None
        ttnn.synchronize_device(mesh_device)

    results = {"users": users, "lens": {u: lens[u] for u in users}, "passes": [list(p) for p in passes], "logits": {}}
    logits = {}  # name -> [vocab] fp32

    # ---- host-side configs the two row counts resolve to (deterministic functions of T) -----------------------------
    grid = mesh_device.compute_with_storage_grid_size()
    layer0 = model.layers[0]
    pc = layer0.mlp.experts.program_config
    attn_pc = layer0.self_attn.program_config
    dense_grid = experts_prefill._dense_core_grid(mesh_device, pc.dense_grid_max_width)
    cfg = {}
    for T in (128, 256, 1024, 4096):
        entry = {
            "router_linear": str(router_linear_program_config(T, 128, H, grid)),
            "shared_expert_gate_up_down": [str(c) for c in shared_expert_program_configs(T, H, 160, grid)],
            "moe_path": experts_prefill.moe_prefill_path(min(T, 1024), pc, 128, 0),
        }
        if T <= pc.dense_bmm_max_tokens:
            entry["dense_gate_up_bmm"] = str(
                experts_prefill._dense_bmm_config(dense_grid, T, layer0.mlp.experts.weights, pc.dense_bmm_max_tokens)
            )
            entry["dense_down"] = str(pc.get_dense_down_config(T, H, 160))
        cfg[T] = entry
    cfg["sdpa_S128"] = str(attn_pc.get_prefill_sdpa_config(mesh_device, SEQ_LEN))
    cfg["sdpa_compute"] = str(attn_pc.get_compute_kernel_config())
    cfg["attention_bf16_output"] = attn_prefill.attention_bf16_output(attn_pc)
    cfg["qkv_matmul"] = "ttnn.linear auto (bf16 x bfp8 -> HiFi2, bf16 dest acc, packer_l1_acc), M = T"
    cfg["o_proj_matmul"] = (
        "typecast bfp8 then ttnn.matmul auto (bfp8 x bfp8 -> LoFi, bf16 dest acc, packer_l1_acc), M = T"
        if not cfg["attention_bf16_output"]
        else "bf16 x bfp8 auto HiFi2"
    )
    cfg["lm_head_matmul"] = "ttnn.matmul auto (bf16 x bfp8 -> HiFi2), M = 32 (sequential / gather) or T (full head)"
    results["configs"] = cfg
    logger.info("configs by row count: " + json.dumps(cfg, indent=1)[:4000])

    # ---- arm A: sequential per-user prefill (production path) -------------------------------------------------------
    clear_kv()
    for u in users:
        rec.enabled, rec.arm = True, f"A{u}"
        with batched_prefill_flag(model_args, False):
            out = generator.prefill_forward_text(
                tokens[u : u + 1],
                page_table=page_table[u : u + 1],
                kv_cache=tt_kv_cache,
                prompt_lens=[lens[u]],
                enable_trace=False,
                warmup_prefill=False,
            )
        rec.enabled = False
        ttnn.synchronize_device(mesh_device)
        logits[f"A{u}"] = out.reshape(-1).float()[:vocab].clone()
        logger.info(f"[arm A] user {u} done: {logits_summary(f'A{u}', logits[f'A{u}'], hf and hf[u], None)}")

    # ---- arm B: packed passes of two users (T = 256, dense-bmm MoE) --------------------------------------------------
    options = mc.BatchedPrefillOptions(enabled=True, tokens_per_pass=256, max_seq_len=SEQ_LEN, head="full")
    for p in passes:
        name = "B" + "+".join(str(u) for u in p)
        clear_kv()
        rec.enabled, rec.arm = True, name
        out = prefill_forward_text_batched(
            generator,
            tokens[list(p)],
            page_table=page_table[list(p)],
            kv_cache=tt_kv_cache,
            prompt_lens=[lens[u] for u in p],
            enable_trace=False,
            options=options,
        )
        rec.enabled = False
        ttnn.synchronize_device(mesh_device)
        log = generator.batched_prefill_pass_log
        assert len(log) == 1 and log[0].packed and log[0].padded_batch == 2, log
        for slot, u in enumerate(p):
            logits[f"{name}/u{u}s{slot}"] = out[slot].reshape(-1).float()[:vocab].clone()
        logger.info(f"[arm B] pass {name} done; MoE path of layer 0: {rec.moe_paths.get(name)}")

    # ---- arm D (phase 3g / D2): arm B's passes with the "sequential numerics" knob -------------------------------------
    fix_level = int(os.getenv("SOLAR_OPEN_BISECT_FIX_LEVEL", "1"))
    fix_passes = [p for p in passes if p[0] != p[1]] if fix_level else []
    results["fix_level"] = fix_level
    for p in fix_passes:
        name = "D" + "+".join(str(u) for u in p)
        clear_kv()
        rec.enabled, rec.arm = True, name
        with packed_prefill.packed_seq_numerics(fix_level):
            out = prefill_forward_text_batched(
                generator,
                tokens[list(p)],
                page_table=page_table[list(p)],
                kv_cache=tt_kv_cache,
                prompt_lens=[lens[u] for u in p],
                enable_trace=False,
                options=options,
            )
        rec.enabled = False
        ttnn.synchronize_device(mesh_device)
        log = generator.batched_prefill_pass_log
        assert len(log) == 1 and log[0].packed and log[0].padded_batch == 2, log
        for slot, u in enumerate(p):
            logits[f"{name}/u{u}s{slot}"] = out[slot].reshape(-1).float()[:vocab].clone()
        logger.info(
            f"[arm D] pass {name} (knob level {fix_level}) done; MoE path of layer 0: {rec.moe_paths.get(name)}"
        )

    # ---- arm C: one user through the packed code path (B = 1, T = 128, head on all rows) -----------------------------
    num_blocks = -(-SEQ_LEN // block_size)
    for u in users[:2]:
        clear_kv()
        rec.enabled, rec.arm = True, f"C{u}"
        prefill_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        prefill_ids[0, : lens[u]] = tokens[u, : lens[u]]
        pt = page_table[u : u + 1, :num_blocks].to(torch.int32)
        x, rot_g, rot_l, pt_tt, *_ = model.prepare_inputs_prefill(prefill_ids, page_table=pt, batch_size=1, user_id=0)
        tt_logits = model.ttnn_prefill_forward(
            x,
            rot_mats_global=rot_g,
            rot_mats_local=rot_l,
            user_id=0,
            page_table=pt_tt,
            get_last_token=-1,
            kv_cache=tt_kv_cache[0],
            batch_size=1,
        )
        rec.enabled = False
        logits[f"C{u}"] = read_logit_rows(model, tt_logits, [lens[u] - 1])[0]
        tt_logits.deallocate(True)
        ttnn.synchronize_device(mesh_device)

    # ---- per-layer / per-op comparison ------------------------------------------------------------------------------
    n_layers = len(model.layers)
    comparisons = {}  # name -> {layer: {tag: metrics}}

    def compare_arms(cmp_name, ref_arm, ref_slot, other_arm, other_slot, u):
        table = {}
        for layer in range(n_layers):
            row = {}
            for tag in OP_ORDER:
                a = rec.get(ref_arm, layer, tag)
                b = rec.get(other_arm, layer, tag)
                if a is None or b is None:
                    continue
                ra, rb = _rows_of(tag, a, ref_slot, SEQ_LEN), _rows_of(tag, b, other_slot, SEQ_LEN)
                row[tag] = compare_rows(ra, rb, lens[u])
                if tag == "router_dense":
                    row[tag].update(router_flips(ra, rb, lens[u]))
            table[layer] = row
        a = rec.get(ref_arm, n_layers - 1, "final_norm")
        b = rec.get(other_arm, n_layers - 1, "final_norm")
        if a is not None and b is not None and a.shape[-2] >= SEQ_LEN and b.shape[-2] >= SEQ_LEN:
            table["final_norm"] = compare_rows(
                _rows_of("final_norm", a, ref_slot, SEQ_LEN), _rows_of("final_norm", b, other_slot, SEQ_LEN), lens[u]
            )
        comparisons[cmp_name] = table
        # console summary: first divergent op of layer 0 and the residual trajectory
        first = next((t for t in OP_ORDER if t in table[0] and not table[0][t]["bit_identical"]), None)
        logger.info(f"[{cmp_name}] layer 0 first non-bit-identical op: {first}")
        for tag in OP_ORDER:
            if tag in table[0]:
                m = table[0][tag]
                logger.info(
                    f"[{cmp_name}] L0 {tag:14s} pcc_min {m['pcc_min']:.6f} max|d| {m['max_abs']:.4f} rel_rms {m['rel_rms']:.2e} "
                    f"scale_mean {m['scale_mean']:+.2e} frac_rows {m['frac_rows_diff']:.3f} last: pcc {m['last_pcc']:.6f} "
                    f"scale {m['last_scale']:+.2e}"
                    + (f" flips {m['rows_set_flipped']}" if tag == "router_dense" else "")
                )
        for layer in (0, 1, 2, 4, 8, 16, 24, 32, 40, 47):
            if layer in table and "resid_moe" in table[layer]:
                m = table[layer]["resid_moe"]
                r = table[layer].get("router_dense", {})
                logger.info(
                    f"[{cmp_name}] L{layer:2d} resid_moe pcc_min {m['pcc_min']:.6f} last_pcc {m['last_pcc']:.6f} "
                    f"rel_rms {m['rel_rms']:.3e} scale_mean {m['scale_mean']:+.2e} last_scale {m['last_scale']:+.2e} "
                    f"router flips {r.get('rows_set_flipped')}/{r.get('rows')}"
                )

    for p in passes:
        name = "B" + "+".join(str(u) for u in p)
        for slot, u in enumerate(p):
            if u in users:
                compare_arms(f"A{u}_vs_{name}_s{slot}", f"A{u}", 0, name, slot, u)
    for u in users[:2]:
        compare_arms(f"A{u}_vs_C{u}", f"A{u}", 0, f"C{u}", 0, u)
    fix_verdict = {}
    for p in fix_passes:
        name = "D" + "+".join(str(u) for u in p)
        for slot, u in enumerate(p):
            if u in users:
                cmp_name = f"A{u}_vs_{name}_s{slot}"
                compare_arms(cmp_name, f"A{u}", 0, name, slot, u)
                table = comparisons[cmp_name]
                per_layer = {
                    L: all(m["bit_identical"] for m in table[L].values()) for L in range(n_layers) if L in table
                }
                first_bad = next(
                    (
                        (L, t)
                        for L in range(n_layers)
                        for t in OP_ORDER
                        if t in table.get(L, {}) and not table[L][t]["bit_identical"]
                    ),
                    None,
                )
                fix_verdict[cmp_name] = {
                    "layers_bit_identical": sum(per_layer.values()),
                    "layers": n_layers,
                    "first_non_bit_identical_layer_op": first_bad,
                    "final_norm_bit_identical": table.get("final_norm", {}).get("bit_identical"),
                    "logits_max_abs_vs_A": float((logits[f"{name}/u{u}s{slot}"] - logits[f"A{u}"]).abs().max()),
                }
                logger.info(f"[fix verdict] {cmp_name}: {fix_verdict[cmp_name]}")
    results["fix_verdict"] = fix_verdict
    # pass-mate independence: user 0 in B0+1 slot 0 vs user 0 in B0+0 slots 0 and 1
    b01 = next((p for p in passes if p == (0, 1)), None)
    b00 = next((p for p in passes if p == (0, 0)), None)
    if b01 and b00:
        compare_arms("B0+1_s0_vs_B0+0_s0", "B0+1", 0, "B0+0", 0, 0)
        compare_arms("B0+1_s0_vs_B0+0_s1", "B0+1", 0, "B0+0", 1, 0)
    results["comparisons"] = comparisons
    results["moe_paths"] = rec.moe_paths
    results["replication"] = rec.replication

    # ---- head: identical hidden states through the three heads ------------------------------------------------------
    head_rows = {}
    final = {}
    for u in users:
        final[f"A{u}"] = _rows_of("resid_moe", rec.get(f"A{u}", n_layers - 1, "resid_moe"), 0, SEQ_LEN).to(
            torch.bfloat16
        )
    for p in passes:
        name = "B" + "+".join(str(u) for u in p)
        for slot, u in enumerate(p):
            final[f"{name}/u{u}s{slot}"] = _rows_of(
                "resid_moe", rec.get(name, n_layers - 1, "resid_moe"), slot, SEQ_LEN
            ).to(torch.bfloat16)
    for p in fix_passes:
        name = "D" + "+".join(str(u) for u in p)
        for slot, u in enumerate(p):
            final[f"{name}/u{u}s{slot}"] = _rows_of(
                "resid_moe", rec.get(name, n_layers - 1, "resid_moe"), slot, SEQ_LEN
            ).to(torch.bfloat16)
    head_results = {}
    for src, rows in final.items():
        u = int(src.split("/u")[-1].split("s")[0]) if "/u" in src else int(src[1:])
        li = lens[u] - 1
        entry = {}
        # sequential 32-row head
        x = upload_rows(mesh_device, rows, H)
        entry["seq32"] = head_seq32(model, x, li)
        x.deallocate(True)
        # full heads at T = 256 / 1024 / 4096: this user's 128 rows in slot 0, copies of them in the other slots
        for T in head_ts:
            copies = T // SEQ_LEN
            x = upload_rows(mesh_device, rows.repeat(copies, 1), H)
            entry[f"full{T}"] = head_full(model, x, [li])[0]
            x.deallocate(True)
        # gather head (32 gathered rows; row li of this user's rows in slot 0)
        x = upload_rows(mesh_device, rows.repeat(2, 1), H)
        entry["gather"] = head_gather(model, x, [li])[0]
        head_results[src] = entry
    # the exact production heads on the exact production residuals must reproduce the arms' logits
    for u in users:
        d = float((head_results[f"A{u}"]["seq32"] - logits[f"A{u}"]).abs().max())
        logger.info(f"[head check] seq32 head on A{u}'s captured residual vs arm A logits: max |diff| {d:.4f}")
        results.setdefault("head_checks", {})[f"A{u}_seq32"] = d
    for p in passes:
        name = "B" + "+".join(str(u) for u in p)
        rows_all = _rows_of("resid_moe", rec.get(name, n_layers - 1, "resid_moe"), 0, 2 * SEQ_LEN).to(torch.bfloat16)
        x = upload_rows(mesh_device, rows_all, H)
        got = head_full(model, x, [slot * SEQ_LEN + lens[u] - 1 for slot, u in enumerate(p)])
        x.deallocate(True)
        for slot, u in enumerate(p):
            d = float((got[slot] - logits[f"{name}/u{u}s{slot}"]).abs().max())
            logger.info(
                f"[head check] full256 head on {name}'s captured residual vs arm B logits (slot {slot}): {d:.4f}"
            )
            results["head_checks"][f"{name}_s{slot}_full256"] = d
            head_results[f"{name}/u{u}s{slot}"]["full256_actual_pass"] = got[slot]
    for p in fix_passes:
        # the knob's head (norm per 32-row tile + lm_head in <= 1024-row pieces) must be the sequential head bit for bit
        name = "D" + "+".join(str(u) for u in p)
        rows_all = _rows_of("resid_moe", rec.get(name, n_layers - 1, "resid_moe"), 0, 2 * SEQ_LEN).to(torch.bfloat16)
        x = upload_rows(mesh_device, rows_all, H)
        with packed_prefill.packed_seq_numerics(1), experts_prefill.packed_prefill_pass(True, seq_len=SEQ_LEN):
            fixed_logits = model._norm_and_lm_head(x)  # consumes x
        got = read_logit_rows(model, fixed_logits, [slot * SEQ_LEN + lens[u] - 1 for slot, u in enumerate(p)])
        fixed_logits.deallocate(True)
        for slot, u in enumerate(p):
            d = float((got[slot] - logits[f"{name}/u{u}s{slot}"]).abs().max())
            d_seq = float((got[slot] - head_results[f"{name}/u{u}s{slot}"]["seq32"]).abs().max())
            logger.info(
                f"[head check] knob head on {name}'s captured residual vs arm D logits (slot {slot}): {d:.4f}; "
                f"vs the seq32 head on the same rows: {d_seq:.4f}"
            )
            results["head_checks"][f"{name}_s{slot}_knob_head"] = d
            results["head_checks"][f"{name}_s{slot}_knob_head_vs_seq32"] = d_seq

    # ---- op isolation: identical inputs through each block at 128 vs T rows --------------------------------------------
    iso_layers = tuple(int(x) for x in os.getenv("SOLAR_OPEN_BISECT_ISO_LAYERS", "0,24,47").split(","))
    iso_ts = tuple(int(x) for x in os.getenv("SOLAR_OPEN_BISECT_ISO_T", "256,4096").split(","))
    results["isolation"] = (
        isolate_ops(mesh_device, model, rec, users[:2], lens, layers=iso_layers, row_counts=iso_ts)
        if os.getenv("SOLAR_OPEN_BISECT_ISO", "1") == "1"
        else {}
    )
    # ---- phase 3g / D2: the fix's levers on identical inputs + E0 (which final-norm kernel is right) ------------------
    if os.getenv("SOLAR_OPEN_BISECT_VERIFY_FIX", "0") == "1":
        results["fix_verify"] = verify_fix_levers(mesh_device, model, rec, users[:2], lens)

    # ---- fp32 truth of the row-count-dependent blocks + fix candidates ---------------------------------------------------
    if os.getenv("SOLAR_OPEN_BISECT_TRUTH", "1") == "1":
        results["truth"] = truth_and_fixes(
            mesh_device, model, rec, users[:2], lens, head_results, layers=iso_layers, row_counts=iso_ts
        )
    # ---- identify the auto program configs by bit-identity against explicit candidates -------------------------------
    if os.getenv("SOLAR_OPEN_BISECT_VERIFY_CONFIGS", "0") == "1":
        results["auto_configs"] = verify_auto_configs(mesh_device, model, rec, users[:2], lens)

    # ---- logits table ------------------------------------------------------------------------------------------------
    rows_out = []
    for name, l in logits.items():
        u = int(name.split("/u")[-1].split("s")[0]) if "/u" in name else int(name[1:])
        rows_out.append(logits_summary(name, l, hf and hf[u], logits[f"A{u}"]))
    for src, entry in head_results.items():
        u = int(src.split("/u")[-1].split("s")[0]) if "/u" in src else int(src[1:])
        for hname, l in entry.items():
            rows_out.append(logits_summary(f"head:{src}:{hname}", l, hf and hf[u], logits[f"A{u}"]))
    results["logits"] = rows_out
    for r in rows_out:
        logger.info(
            f"[logits] {r['name']:32s} top1 {r['top1']:3d} margin {r['margin']:.3f} gap(22,23) {r['gap_22_23']:+.3f}"
            + (f" KL(HF||x) {r['kl_hf']:.4f} pcc_hf {r['pcc_hf']:.5f}" if "kl_hf" in r else "")
            + (f" KL(A||x) {r['kl_seq']:.4f} max|d| {r['max_abs_vs_seq']:.3f}" if "kl_seq" in r else "")
        )

    # ---- auto program-config probe (best effort) --------------------------------------------------------------------
    results["auto_matmul_probe"] = probe_auto_matmul_configs(mesh_device, model, out_dir)

    # ---- write ---------------------------------------------------------------------------------------------------------
    (out_dir / "bisect_results.json").write_text(json.dumps(results, indent=1, default=str))
    torch.save({k: v for k, v in logits.items()}, out_dir / "logits.pt")
    torch.save(final, out_dir / "final_residuals.pt")
    lines = ["# D1 bisect tables", ""]
    for cmp_name, table in comparisons.items():
        lines += [
            f"## {cmp_name}",
            "",
            "### layer 0, per op",
            "",
            "| op | bit_identical | pcc_min | max_abs | rel_rms | mean_signed | scale_mean | scale_min/max | frac_rows_diff | last_pcc | last_max_abs | last_scale | router flips |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for tag in OP_ORDER:
            if tag in table[0]:
                m = table[0][tag]
                lines.append(
                    f"| {tag} | {_fmt(m['bit_identical'])} | {m['pcc_min']:.6f} | {m['max_abs']:.4f} | {m['rel_rms']:.2e} | "
                    f"{m['mean_signed']:+.2e} | {m['scale_mean']:+.2e} | {m['scale_min']:+.2e}/{m['scale_max']:+.2e} | "
                    f"{m['frac_rows_diff']:.3f} | {m['last_pcc']:.6f} | {m['last_max_abs']:.4f} | {m['last_scale']:+.2e} | "
                    f"{m.get('rows_set_flipped', '')} |"
                )
        lines += [
            "",
            "### per layer (resid_attn / resid_moe, prompt rows; last = last prompt token)",
            "",
            "| layer | attn pcc_min | attn rel_rms | attn scale_mean | moe pcc_min | moe last_pcc | moe rel_rms | moe last_rel_rms | moe scale_mean | moe last_scale | router flips |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for layer in range(n_layers):
            t = table.get(layer, {})
            a, m, r = t.get("resid_attn"), t.get("resid_moe"), t.get("router_dense", {})
            if a is None or m is None:
                continue
            lines.append(
                f"| {layer} | {a['pcc_min']:.6f} | {a['rel_rms']:.2e} | {a['scale_mean']:+.2e} | {m['pcc_min']:.6f} | "
                f"{m['last_pcc']:.6f} | {m['rel_rms']:.2e} | {m['last_rel_rms']:.2e} | {m['scale_mean']:+.2e} | "
                f"{m['last_scale']:+.2e} | {r.get('rows_set_flipped', '')}/{r.get('rows', '')} |"
            )
        lines.append("")
    lines += [
        "## op isolation (arm A's captured inputs; user 0's rows at 128 rows vs the same rows inside T rows)",
        "",
        "| layer | op | T | bit_identical | pcc_min | max_abs | rel_rms | scale_mean | scale_min/max | frac_rows_diff | last_pcc | last_scale | router flips |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for L, table in results["isolation"].items():
        for op, per_T in table.items():
            for T, m in per_T.items():
                if T == "error":
                    lines.append(f"| {L} | {op} | - | error: {m} | | | | | | | | | |")
                    continue
                lines.append(
                    f"| {L} | {op} | {T} | {_fmt(m['bit_identical'])} | {m['pcc_min']:.6f} | {m['max_abs']:.4f} | {m['rel_rms']:.2e} | "
                    f"{m['scale_mean']:+.2e} | {m['scale_min']:+.2e}/{m['scale_max']:+.2e} | {m['frac_rows_diff']:.3f} | "
                    f"{m['last_pcc']:.6f} | {m['last_scale']:+.2e} | {m.get('rows_set_flipped', '')} |"
                )
    lines.append("")
    if "truth" in results:
        lines += [
            "## fp32 truth (device result vs fp32 host matmul of the device's own operands)",
            "",
            "| block | case | rel_rms | scale_mean | scale_min/max | pcc_min | last_scale | vs128 bit-identical | vs128 rel_rms | gap(22,23) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for L, table in results["truth"].items():
            blocks = table.items() if L != "lm_head" else [("lm_head", table)]
            for op, entry in blocks:
                for key, m in entry.items():
                    if not isinstance(m, dict):
                        lines.append(f"| {op} | {key} | {m} | | | | | | | |")
                        continue
                    if "error" in m:
                        lines.append(f"| {op} | {key} | error: {m['error']} | | | | | | | |")
                        continue
                    if "rel_rms" not in m:
                        continue
                    lines.append(
                        f"| {op} | {key} | {m['rel_rms']:.3e} | {m['scale_mean']:+.3e} | "
                        f"{m.get('scale_min', float('nan')):+.2e}/{m.get('scale_max', float('nan')):+.2e} | "
                        f"{m['pcc_min']:.6f} | {m.get('last_scale', float('nan')):+.3e} | {m.get('vs_128_bit_identical', '')} | "
                        f"{m.get('vs_128_rel_rms', float('nan')):.2e} | {m.get('gap_22_23', float('nan')):+.3f} |"
                    )
        lines.append("")
    if results.get("fix_verdict"):
        lines += [
            "## phase 3g / D2 fix verdict (arm D = arm B's passes with the knob)",
            "",
            "| comparison | layers bit-identical | first non-bit-identical (layer, op) | final norm bit-identical | logits max|d| vs A |",
            "|---|---|---|---|---|",
        ]
        for k, v in results["fix_verdict"].items():
            lines.append(
                f"| {k} | {v['layers_bit_identical']} / {v['layers']} | {v['first_non_bit_identical_layer_op']} | "
                f"{v['final_norm_bit_identical']} | {v['logits_max_abs_vs_A']:.4f} |"
            )
        lines.append("")
    if results.get("fix_verify"):
        lines += [
            "## phase 3g / D2 fix levers on identical inputs (+ E0)",
            "",
            "| check | bit_identical | rel_rms | scale_mean | pcc_min | max_abs |",
            "|---|---|---|---|---|---|",
        ]
        for k, m in results["fix_verify"].items():
            if "error" in m:
                lines.append(f"| {k} | error: {m['error']} | | | | |")
            else:
                lines.append(
                    f"| {k} | {_fmt(m.get('bit_identical', ''))} | {m['rel_rms']:.3e} | {m['scale_mean']:+.3e} | "
                    f"{m['pcc_min']:.6f} | {m['max_abs']:.4f} |"
                )
        lines.append("")
    lines += [
        "## logits",
        "",
        "| name | top1 | margin | gap(22,23) | KL(HF||x) | pcc_hf | top1=HF | KL(A||x) | pcc_A | max|d| vs A |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_out:
        lines.append(
            f"| {r['name']} | {r['top1']} | {r['margin']:.3f} | {r['gap_22_23']:+.3f} | {r.get('kl_hf', float('nan')):.4f} | "
            f"{r.get('pcc_hf', float('nan')):.5f} | {r.get('top1_eq_hf', '')} | {r.get('kl_seq', float('nan')):.4f} | "
            f"{r.get('pcc_seq', float('nan')):.5f} | {r.get('max_abs_vs_seq', float('nan')):.3f} |"
        )
    (out_dir / "tables.md").write_text("\n".join(lines) + "\n")
    logger.info(f"wrote {out_dir}/bisect_results.json, tables.md, logits.pt, final_residuals.pt")

    # The only assertion: the hooked, re-run heads reproduce the production logits (else the bisection is not reading
    # the tensors the model used).
    bad = {k: v for k, v in results["head_checks"].items() if v > 0.0}
    assert not bad, f"re-run heads do not reproduce the arms' logits bit for bit: {bad}"
