# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The paged SDPA decode's ``sliding_window_size`` in the DFlash2 drafter's configuration (tt/dflash2_head.py
``step_forward``): one virtual user per block row (R = 8 rows of one block sharing cur_pos = P + 7 and one page-table
row), the drafter's per-device shapes at TP=4 (8 q heads / 2 kv heads x 128, bfp8 paged K/V, block 64), the window of
the reference (2048). Checks, against a torch reference over the bfp8-rounded K/V:
  * the windowed op equals attention over keys [cur_pos + 1 - window, cur_pos] (the kernel's window, rt_args_common.hpp:
    ``window_start = cur_pos + 1 - sliding_window_size``, chunk-aligned reads masked to it);
  * the un-windowed op equals attention over the whole context and DIFFERS from the windowed reference (the window
    actually cuts the context past 2048 positions).
Single device (any chip):  TT_VISIBLE_DEVICES=2 python_env/bin/python -m pytest models/demos/blackhole/qwen36/tests/test_dflash2_window_op.py -s
"""
import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_mtp_spec_scratch import _pcc

NH, NKV, HD, BLK, B = 8, 2, 128, 64, 8  # the drafter's per-device heads at TP=4, the model's block size, the block rows


def _ref(q, k, v, lo, hi):
    """q [B, NH, HD], k/v [NKV, S, HD] -> [B, NH, HD] attending keys lo..hi inclusive (GQA: q head h -> kv head h // g)."""
    g = NH // NKV
    out = torch.zeros(B, NH, HD)
    for h in range(NH):
        kk, vv = k[h // g, lo : hi + 1].float(), v[h // g, lo : hi + 1].float()
        s = (q[:, h].float() @ kk.T) * HD**-0.5
        out[:, h] = torch.softmax(s, dim=-1) @ vv
    return out


@pytest.mark.parametrize("ctx,window", [(3000, 2048), (5000, 2048), (1500, 2048)])
def test_paged_sdpa_decode_sliding_window(device, ctx, window):
    torch.manual_seed(0)
    P = ctx  # the block starts at P (context = positions 0..P-1); rows at P..P+7 share cur_pos = P + 7
    cur = P + B - 1
    n_blocks = -(-(P + B) // BLK) + 1
    S = n_blocks * BLK
    k = torch.randn(NKV, S, HD, dtype=torch.bfloat16)
    v = torch.randn(NKV, S, HD, dtype=torch.bfloat16)
    paged = lambda t: t.reshape(NKV, n_blocks, BLK, HD).permute(1, 0, 2, 3).contiguous()  # [blocks, NKV, BLK, HD]
    k_tt = ttnn.from_torch(paged(k), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    v_tt = ttnn.from_torch(paged(v), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    unpaged = lambda t: t.permute(1, 0, 2, 3).reshape(NKV, S, HD)
    k_r, v_r = unpaged(ttnn.to_torch(k_tt)), unpaged(ttnn.to_torch(v_tt))  # the bfp8-rounded values the op reads
    q = torch.randn(1, B, NH, HD, dtype=torch.bfloat16)
    q_tt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pt = ttnn.from_torch(torch.arange(n_blocks, dtype=torch.int32).repeat(B, 1), dtype=ttnn.int32, device=device)
    pos = ttnn.from_torch(torch.full((B,), cur, dtype=torch.int32), dtype=ttnn.int32, device=device)
    grid = device.compute_with_storage_grid_size()
    cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
    )

    def run(**kw):
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt, k_tt, v_tt, page_table_tensor=pt, cur_pos_tensor=pos, scale=HD**-0.5, program_config=cfg, **kw
        )
        return ttnn.to_torch(out).float().reshape(B, -1, HD)[:, :NH]

    dev_win = run(sliding_window_size=window)
    dev_full = run()
    lo = max(0, cur + 1 - window)
    ref_win = _ref(q[0], k_r, v_r, lo, cur)
    ref_full = _ref(q[0], k_r, v_r, 0, cur)
    p_ww, p_ff = _pcc(dev_win, ref_win), _pcc(dev_full, ref_full)
    p_wf = _pcc(dev_win, ref_full)
    print(
        f"ctx={ctx} window={window} keys [{lo}, {cur}]: windowed op vs windowed ref PCC {p_ww:.5f}; whole-context op vs "
        f"whole-context ref {p_ff:.5f}; windowed op vs whole-context ref {p_wf:.5f}"
    )
    assert p_ww > 0.99, p_ww
    assert p_ff > 0.99, p_ff
    if lo > 0:
        assert p_wf < 0.98, f"the window did not cut the context: {p_wf}"
    else:
        assert p_wf > 0.99, p_wf  # context shorter than the window: identical
