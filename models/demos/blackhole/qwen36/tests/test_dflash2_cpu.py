# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the DFlash2 host reference (tt/dflash2_head.py) vs the OFFICIAL z-lab ``DFlash2DraftModel`` (fp32), and its
incremental context state vs a from-scratch one. No device.

  DFLASH2_OFFICIAL_PY=/path/to/dflash/model.py (z-lab/dflash, MIT)  -- the official module; without it only the
  incremental-state and selector self-checks run.
  HF_MODEL / the HF cache must hold Qwen/Qwen3.8-27B (the target's embedding table + LM head) and
  z-lab/Qwen3.8-27B-DFlash2 (DFLASH2_MODEL overrides).
"""
import importlib.util
import os
import time

import pytest
import torch
from loguru import logger

from models.demos.blackhole.qwen36.tt.dflash2_head import (
    DFlash2Config,
    DFlash2Drafter,
    DFlash2HostReference,
    dflash2_snapshot_dir,
    grouped_dynamic_conv,
    load_dflash2_state_dict,
    load_target_embed_and_head,
)

OFFICIAL = os.environ.get("DFLASH2_OFFICIAL_PY")


def _target_dir():
    p = os.environ.get("HF_MODEL", "Qwen/Qwen3.8-27B")
    if os.path.isfile(os.path.join(p, "config.json")):
        return p
    from huggingface_hub import snapshot_download

    return snapshot_download(p, local_files_only=True)


@pytest.fixture(scope="module")
def ref():
    t0 = time.perf_counter()
    path = dflash2_snapshot_dir()
    cfg = DFlash2Config(path)
    sd = load_dflash2_state_dict(path)
    emb, head = load_target_embed_and_head(_target_dir())
    r = DFlash2HostReference(cfg, sd, emb, head)
    logger.info(f"host reference loaded in {time.perf_counter() - t0:.1f}s")
    return r


def _inputs(cfg, n_ctx, seed=0):
    g = torch.Generator().manual_seed(seed)
    aux = torch.randn(n_ctx, 5 * cfg.dim, generator=g) * 3.0  # realistic residual-stream magnitude
    anchor = int(torch.randint(0, 100000, (1,), generator=g))
    return aux, anchor


def test_selector_and_incremental_state(ref):
    cfg = ref.cfg
    aux, anchor = _inputs(cfg, 40)
    # from scratch: 40 context rows; incremental: 30 rows + 10 appended (+ a re-written position)
    s_full = ref.new_state()
    ref.append_context(s_full, aux, torch.arange(40))
    s_inc = ref.new_state()
    ref.append_context(s_inc, aux[:30], torch.arange(30))
    ref.append_context(s_inc, torch.randn(3, 5 * cfg.dim), torch.arange(30, 33))  # rejected-row garbage
    ref.append_context(s_inc, aux[30:], torch.arange(30, 40))  # re-written
    d_full = ref.draft(s_full, anchor, 40)
    d_inc = ref.draft(s_inc, anchor, 40)
    assert torch.allclose(d_full["hidden"], d_inc["hidden"], atol=1e-4, rtol=1e-4)
    assert d_full["tokens"] == d_inc["tokens"]
    # selector: the path is a valid walk through the candidates and differs from plain argmax only by the bilinear term
    cands = d_full["candidates"]
    for t, tok in enumerate(d_full["tokens"]):
        assert tok in cands[t].tolist()
    argmax = d_full["logits"].argmax(-1).tolist()
    logger.info(
        f"path {d_full['tokens']} argmax {argmax} (agree {sum(a == b for a, b in zip(argmax, d_full['tokens']))}/7)"
    )


@pytest.mark.skipif(not OFFICIAL, reason="DFLASH2_OFFICIAL_PY not set")
def test_vs_official_dflash2(ref):
    cfg = ref.cfg
    spec = importlib.util.spec_from_file_location("dflash_official", OFFICIAL)
    dm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dm)
    t0 = time.perf_counter()
    off = dm.DFlash2DraftModel.from_pretrained(cfg.path, torch_dtype=torch.float32)
    off.eval()
    logger.info(f"official DFlash2DraftModel (fp32) loaded in {time.perf_counter() - t0:.1f}s")
    head = torch.nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
    head.weight.data = ref.lm_head.float()
    n_ctx = 37
    for seed in (0, 1):
        aux, anchor = _inputs(cfg, n_ctx, seed)
        toks = [anchor] + [cfg.mask_token_id] * cfg.n_draft
        pos = torch.arange(n_ctx + cfg.block_size)[None]
        with torch.no_grad():
            hid_off = off(
                position_ids=pos,
                noise_embedding=ref.embed[torch.tensor(toks)].float()[None],
                target_hidden=aux[None],
            )[
                0
            ]  # [8, dim] post-norm
            path_off, cands_off, _ = off.propose(hid_off[None, 1:], torch.tensor([anchor]), head, 0.0)
        st = ref.new_state()
        ref.append_context(st, aux, torch.arange(n_ctx))
        hid = ref.block_forward(st, toks, torch.arange(n_ctx, n_ctx + cfg.block_size))
        d = torch.max(torch.abs(hid - hid_off)).item()
        rel = d / torch.max(torch.abs(hid_off)).item()
        mine = ref.draft(st, anchor, n_ctx)
        logger.info(
            f"seed {seed}: block hidden max|d| {d:.3e} (rel {rel:.2e}); path official {path_off[0].tolist()} "
            f"mine {mine['tokens']}; candidate sets equal: "
            f"{[set(cands_off[0, t].tolist()) == set(mine['candidates'][t].tolist()) for t in range(cfg.n_draft)]}"
        )
        assert rel < 1e-4, rel
        assert mine["tokens"] == path_off[0].tolist()


def test_device_constant_matrices():
    """The device path's 0/1 constants: the block-local shift equals the conv's x_{t-1} (zero at each block's first
    row) and the K/V spread puts grid row s*T+j of kv head h at shard row s*32 + h*T + j."""
    w, T = 3, 8
    R = w * T
    x = torch.randn(R, 64)
    shift = DFlash2Drafter.shift_matrix(w, T, R)[0, 0]
    xs = shift @ x
    for s in range(w):
        blk = x[s * T : (s + 1) * T]
        ref = grouped_dynamic_conv(blk, torch.zeros(T, 2, 4), torch.tensor([[0.0] * 64, [1.0] * 64]), 16)  # = x_{t-1}
        assert torch.equal(xs[s * T : (s + 1) * T], ref)
    NKV = 2
    spreads = DFlash2Drafter.spread_matrices(w, T, R, NKV)
    heads = [torch.randn(R, 64) for _ in range(NKV)]
    sh = sum(spreads[h][0, 0] @ heads[h] for h in range(NKV))  # [w*32, 64]
    for s in range(w):
        for h in range(NKV):
            for j in range(T):
                assert torch.equal(sh[s * 32 + h * T + j], heads[h][s * T + j])
        assert torch.equal(sh[s * 32 + NKV * T : (s + 1) * 32], torch.zeros(32 - NKV * T, 64))
