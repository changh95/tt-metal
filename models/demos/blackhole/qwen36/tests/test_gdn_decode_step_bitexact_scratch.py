# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: bit-exact regression of ttnn.experimental.kda.gdn_decode_step (fused-conv batched + plain variants).

Modes, selected by ``QWEN36_GDN_REF=save|check|compare`` (``QWEN36_GDN_REF_DIR`` = output dir). The default is
``check``: a run without the variable can never overwrite a saved reference. Saving is always explicit and refuses to
overwrite an existing reference file unless ``QWEN36_GDN_REF_OVERWRITE=1`` is also set.

* ``save``  -- run on the REFERENCE kernel sources, dump every output (out, state, packed history) per step.
* ``check`` -- run on the changed tree, assert every saved tensor is reproduced EXACTLY (torch.equal on the raw bits).
* ``compare`` -- like check but only logs pcc / max|d| per tensor (for numerics-changing opt-ins such as
  GDN_DECODE_FUSED_UPDATE=1 in gdn_decode_step_conv.cpp).

Cases: fused-conv batched op for B in QWEN36_GDN_REF_BS (default 1,2,8,32; users in rows 0..B-1 so B >= 2 covers both
row parities) chained over STEPS steps from a Bmax=32 state, and the plain (non-conv, B=1) op chained over STEPS steps.
Both also print pcc / max|d| against the torch reference from test_gdn_decode_step_scratch.py for information.

  QWEN36_GDN_REF=save  pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step_bitexact_scratch.py -s
  QWEN36_GDN_REF=check pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step_bitexact_scratch.py -s
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_gdn_decode_step_scratch import (
    KD,
    VD,
    Dk,
    Dv,
    Nk,
    Nv,
    _pack_rows_user,
    _pcc,
    _reference,
    _reference_conv,
)

MODE = os.environ.get("QWEN36_GDN_REF", "check")
assert MODE in ("save", "check", "compare"), f"QWEN36_GDN_REF must be save|check|compare, got {MODE!r}"
OVERWRITE = os.environ.get("QWEN36_GDN_REF_OVERWRITE", "0") == "1"
REF_DIR = os.environ.get("QWEN36_GDN_REF_DIR", "/home/eslim/experiments/qwen36/logs/itemC_ref")
BS = [int(b) for b in os.environ.get("QWEN36_GDN_REF_BS", "1,2,8,32").split(",")]
STEPS = int(os.environ.get("QWEN36_GDN_REF_STEPS", "3"))
BMAX = 32


def _save_or_check(name, tensors):
    os.makedirs(REF_DIR, exist_ok=True)
    path = os.path.join(REF_DIR, f"{name}.pt")
    if MODE == "save":
        assert OVERWRITE or not os.path.exists(
            path
        ), f"{path} exists; refusing to overwrite a saved reference (set QWEN36_GDN_REF_OVERWRITE=1 to replace it)"
        torch.save({k: v.clone().contiguous() for k, v in tensors.items()}, path)
        logger.info(f"[ref] saved {len(tensors)} tensors -> {path}")
        return
    ref = torch.load(path)
    if MODE == "compare":  # numerics-changing variants (e.g. GDN_DECODE_FUSED_UPDATE=1): report, do not assert
        for k, v in tensors.items():
            r = ref[k].float()
            d = (r - v.float()).abs()
            logger.info(
                f"[ref] {name}/{k}: pcc={_pcc(r, v.float()):.8f} max|d|={d.max().item():.3e} "
                f"mismatching elements={int((d != 0).sum())}/{d.numel()}"
            )
        return
    assert MODE == "check", MODE
    bad = []
    for k, v in tensors.items():
        r = ref[k]
        if r.dtype != v.dtype or r.shape != v.shape or not torch.equal(r, v):
            nd = (r.float() - v.float()).abs()
            bad.append(
                f"{k}: shape {tuple(r.shape)} vs {tuple(v.shape)}, mismatches={int((nd != 0).sum())} max|d|={nd.max().item():.3e}"
            )
    assert not bad, f"{name}: NOT bit-identical to {path}:\n  " + "\n  ".join(bad)
    logger.info(f"[ref] {name}: {len(tensors)} tensors bit-identical to {path}")


@pytest.mark.parametrize("B", BS)
def test_gdn_decode_step_conv_batched_bitexact(device, B):
    torch.manual_seed(100 + B)
    scale = Dk**-0.5
    C = 2 * KD + VD
    W = C + VD + 32
    az = C + VD
    h_all = (0.05 * torch.randn(BMAX, Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    taps = [(0.3 * torch.randn(C)).bfloat16() for _ in range(4)]
    dtb = (0.1 * torch.randn(Nv)).bfloat16()
    nea = (-torch.exp(0.2 * torch.randn(Nv))).bfloat16()
    cs_all = [[(0.5 * torch.randn(C)).bfloat16() for _ in range(4)] for _ in range(BMAX)]
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev(h_all.clone(), ttnn.float32)
    w_dev = to_dev(w, ttnn.bfloat16)
    taps_dev = to_dev(_pack_rows_user(taps, 0, both=True), ttnn.bfloat16)
    hist_host = torch.stack([_pack_rows_user(cs_all[b], b) for b in range(BMAX)])
    hist_dev = to_dev(hist_host.clone(), ttnn.bfloat16)
    h_ref = h_all.clone()
    cs_ref = [[t.float() for t in cs_all[b]] for b in range(BMAX)]
    for step in range(STEPS):
        rows = (0.5 * torch.randn(B, W)).bfloat16()
        rows[:, az + 2 * Nv :] = 0
        refs = []
        for b in range(B):
            gated_ref, h_b, nh = _reference_conv(
                cs_ref[b][1:],
                rows[b].float(),
                [t.float() for t in taps],
                dtb.float(),
                nea.float(),
                h_ref[b],
                w.float(),
                scale,
            )
            refs.append(gated_ref)
            h_ref[b] = h_b
            cs_ref[b] = nh
        out = ttnn.experimental.kda.gdn_decode_step(
            to_dev(rows.reshape(1, B, W), ttnn.bfloat16),
            to_dev(dtb, ttnn.bfloat16),
            to_dev(nea, ttnn.bfloat16),
            state,
            w_dev,
            Nv,
            Nk,
            Dk,
            Dv,
            scale=scale,
            output_dtype=ttnn.float32,
            conv_hist=hist_dev,
            conv_taps=taps_dev,
            qkvz_dim=az,
        )
        out_t = ttnn.to_torch(out)
        h_t = ttnn.to_torch(state)
        hist_t = ttnn.to_torch(hist_dev)
        worst = min(_pcc(out_t.reshape(B, -1)[b], refs[b]) for b in range(B))
        max_d = max((out_t.reshape(B, -1)[b] - refs[b]).abs().max().item() for b in range(B))
        logger.info(
            f"conv B={B} step {step}: out pcc(min)={worst:.6f} max|d|={max_d:.3e} state pcc(min)={min(_pcc(h_t[b], h_ref[b]) for b in range(B)):.6f}"
        )
        _save_or_check(f"conv_B{B}_step{step}", {"out": out_t, "state": h_t, "hist": hist_t})
        ttnn.deallocate(out)


def test_gdn_decode_step_plain_bitexact(device):
    torch.manual_seed(7)
    scale = Dk**-0.5
    h = (0.05 * torch.randn(Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    to_dev = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
    state = to_dev(h.reshape(1, Nv, Dk, Dv), ttnn.float32)
    w_dev = to_dev(w, ttnn.bfloat16)
    h_ref = h.clone()
    for step in range(STEPS):
        q = (0.5 * torch.randn(Nk, Dk)).bfloat16()
        k = (0.5 * torch.randn(Nk, Dk)).bfloat16()
        v = (0.5 * torch.randn(Nv, Dv)).bfloat16()
        beta = torch.sigmoid(torch.randn(Nv)).float()
        g = (-0.5 * torch.rand(Nv)).float()
        out_ref, h_ref = _reference(q.float(), k.float(), v.float(), beta, g, h_ref, w.float(), scale)
        qkv_row = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)]).reshape(1, 1, 2 * KD + VD)
        out = ttnn.experimental.kda.gdn_decode_step(
            to_dev(qkv_row, ttnn.bfloat16),
            to_dev(beta.reshape(1, 1, Nv), ttnn.float32),
            to_dev(g.reshape(1, 1, Nv), ttnn.float32),
            state,
            w_dev,
            Nv,
            Nk,
            Dk,
            Dv,
            scale=scale,
            output_dtype=ttnn.float32,
        )
        out_t = ttnn.to_torch(out)
        h_t = ttnn.to_torch(state)
        logger.info(
            f"plain step {step}: out pcc={_pcc(out_t.reshape(-1), out_ref):.6f} max|d|={(out_t.reshape(-1) - out_ref).abs().max().item():.3e} "
            f"state pcc={_pcc(h_t.reshape(Nv, Dk, Dv), h_ref):.6f}"
        )
        _save_or_check(f"plain_step{step}", {"out": out_t, "state": h_t})
        ttnn.deallocate(out)
