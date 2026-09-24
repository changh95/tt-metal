# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: multi-token (speculative-verify) mode of ttnn.experimental.kda.gdn_decode_step (num_tokens = T > 1).

Row grid: tensor row r = s*T + j (user s, offset j); qkv row j = 0 = the user's last committed token, rows 1..k = the
new drafts; qkv_prev = last step's qkv; accept[s] = drafts of last step accepted. Per user the op commits prev rows
1..accept[s] then cur row 0 (state + packed history written once), continues over cur rows 1..k in L1 only and emits
one gated-norm output row per cur row.

Checks, chained over STEPS steps (qkv_prev <- qkv, fresh random accept each step; step 0: zeros / accept 0):
  (a) torch reference of the sequence semantics built from _reference_conv (pcc, information + sanity),
  (b) BIT-EXACT (torch.equal on raw bits) against the T=1 op applied token by token: per user a private
      [1, Nv, Dk, Dv] state + [1, Nv, 4, 32, 32] history on device, the T=1 op run on (prev rows 1..a, cur row 0) for
      the committed state / history and on clones for each draft row's output,
  (c) B users sharing one accept value: the T=1 op in its batched form (user s in row s, the model's layout) as a
      second, model-like reference,
  (d) padding rows of the output are exactly 0, state / history slots >= B untouched.

  pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_step_multitoken_scratch.py -s
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
    _reference_conv,
)

STEPS = int(os.environ.get("QWEN36_GDN_MT_STEPS", "3"))
BMAX = 32
C = 2 * KD + VD
W = C + VD + 32  # [q | k | v | z | a b (padded to a tile)]
AZ = C + VD
SCALE = Dk**-0.5


def _to_dev(device, t, dt, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dt, layout=layout, device=device)


def _rows_tensor(device, rows_bt, R):
    """[B*T, W] token rows (+ random garbage in the padding rows) -> [1, 1, R, W] bf16 TILE on device."""
    full = (0.5 * torch.randn(R, W)).bfloat16()
    full[: rows_bt.shape[0]] = rows_bt
    return _to_dev(device, full.reshape(1, 1, R, W), ttnn.bfloat16)


def _hist_content(packed, par):
    """The chunk rows (2c + par) of a packed [.., 4, 32, 32] history and its other-parity rows."""
    return packed[..., par::2, :], packed[..., 1 - par :: 2, :]


class _T1Ref:
    """The T=1 op applied token by token on private per-user state / history tensors (row 0, parity 0)."""

    def __init__(self, device, B, h_all, cs_all, taps_dev, dtb_dev, nea_dev, w_dev):
        self.device = device
        self.taps_dev, self.dtb_dev, self.nea_dev, self.w_dev = taps_dev, dtb_dev, nea_dev, w_dev
        self.state = [_to_dev(device, h_all[s : s + 1].clone(), ttnn.float32) for s in range(B)]
        self.hist = [_to_dev(device, _pack_rows_user(cs_all[s], 0).unsqueeze(0), ttnn.bfloat16) for s in range(B)]

    def step(self, row, state, hist):
        """One T=1 op on (state, hist) with the token `row` [W]; returns the output row [Nv*Dv] fp32 (host)."""
        out = ttnn.experimental.kda.gdn_decode_step(
            _to_dev(self.device, row.reshape(1, 1, W), ttnn.bfloat16),
            self.dtb_dev,
            self.nea_dev,
            state,
            self.w_dev,
            Nv,
            Nk,
            Dk,
            Dv,
            scale=SCALE,
            output_dtype=ttnn.float32,
            conv_hist=hist,
            conv_taps=self.taps_dev,
            qkvz_dim=AZ,
        )
        o = ttnn.to_torch(out).reshape(-1)
        ttnn.deallocate(out)
        return o

    def commit(self, s, rows):
        for row in rows:
            self.step(row, self.state[s], self.hist[s])

    def commit_then_drafts(self, s, prev_rows, cur_rows):
        """Commit prev_rows then cur_rows[0]; outputs for every cur row (drafts on clones). Returns [T, Nv*Dv]."""
        self.commit(s, prev_rows)
        outs = [self.step(cur_rows[0], self.state[s], self.hist[s])]
        if len(cur_rows) > 1:
            st, hi = ttnn.clone(self.state[s]), ttnn.clone(self.hist[s])
            for row in cur_rows[1:]:
                outs.append(self.step(row, st, hi))
            ttnn.deallocate(st)
            ttnn.deallocate(hi)
        return torch.stack(outs)


def _torch_ref_user(cs, h, prev_rows, cur_rows, taps, dtb, nea, w):
    """Torch sequence semantics from _reference_conv: returns (outputs [T, Nv*Dv], committed h, committed cs (4 rows))."""
    tf = [t.float() for t in taps]
    for row in prev_rows:
        _, h, cs = _reference_conv(cs[1:], row.float(), tf, dtb, nea, h, w, SCALE)
    outs = []
    gated, h, cs = _reference_conv(cs[1:], cur_rows[0].float(), tf, dtb, nea, h, w, SCALE)
    outs.append(gated)
    h_c, cs_c = h, cs
    for row in cur_rows[1:]:
        gated, h, cs = _reference_conv(cs[1:], row.float(), tf, dtb, nea, h, w, SCALE)
        outs.append(gated)
    return torch.stack(outs), h_c, cs_c


@pytest.mark.parametrize("B", [1, 2, 8, 32])
@pytest.mark.parametrize("T", [2, 4, 8])
def test_gdn_decode_step_multitoken_bitexact(device, B, T):
    torch.manual_seed(1000 + 10 * B + T)
    k = T - 1
    R = ((B * T + 31) // 32) * 32
    h_all = (0.05 * torch.randn(BMAX, Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    taps = [(0.3 * torch.randn(C)).bfloat16() for _ in range(4)]
    dtb = (0.1 * torch.randn(Nv)).bfloat16()
    nea = (-torch.exp(0.2 * torch.randn(Nv))).bfloat16()
    cs_all = [[(0.5 * torch.randn(C)).bfloat16() for _ in range(4)] for _ in range(BMAX)]
    state = _to_dev(device, h_all.clone(), ttnn.float32)
    w_dev = _to_dev(device, w, ttnn.bfloat16)
    taps_dev = _to_dev(device, _pack_rows_user(taps, 0, both=True), ttnn.bfloat16)
    hist_host = torch.stack([_pack_rows_user(cs_all[b], b) for b in range(BMAX)])  # user b at parity b & 1
    hist_dev = _to_dev(device, hist_host.clone(), ttnn.bfloat16)
    dtb_dev = _to_dev(device, dtb, ttnn.bfloat16)
    nea_dev = _to_dev(device, nea, ttnn.bfloat16)
    ref = _T1Ref(device, B, h_all, cs_all, taps_dev, dtb_dev, nea_dev, w_dev)
    # torch reference state
    h_ref = [h_all[s].clone() for s in range(B)]
    cs_ref = [[t.float() for t in cs_all[s]] for s in range(B)]

    prev_rows = torch.zeros(B * T, W).bfloat16()
    qkv_prev = _rows_tensor(device, prev_rows, R)
    accept = [0] * B
    for step in range(STEPS):
        rows = (0.5 * torch.randn(B * T, W)).bfloat16()
        rows[:, AZ + 2 * Nv :] = 0
        qkv = _rows_tensor(device, rows, R)
        acc_dtype = ttnn.uint32 if step % 2 == 0 else ttnn.int32
        acc_dev = ttnn.from_torch(
            torch.tensor(accept, dtype=torch.int32), dtype=acc_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        out = ttnn.experimental.kda.gdn_decode_step(
            qkv,
            dtb_dev,
            nea_dev,
            state,
            w_dev,
            Nv,
            Nk,
            Dk,
            Dv,
            scale=SCALE,
            output_dtype=ttnn.float32,
            conv_hist=hist_dev,
            conv_taps=taps_dev,
            qkvz_dim=AZ,
            num_tokens=T,
            qkv_prev=qkv_prev,
            accept=acc_dev,
        )
        assert tuple(out.shape) == (1, 1, R, Nv * Dv), out.shape
        out_t = ttnn.to_torch(out).reshape(R, Nv * Dv)
        h_t = ttnn.to_torch(state)
        hist_t = ttnn.to_torch(hist_dev)
        ttnn.deallocate(out)
        # (d) padding rows / untouched slots
        assert torch.equal(out_t[B * T :], torch.zeros_like(out_t[B * T :])), "padding rows not zero"
        if B < BMAX:
            assert torch.equal(h_t[B:], h_all[B:]) and torch.equal(hist_t[B:], hist_host[B:]), "slots >= B touched"
        # (b) T=1 op token by token, (a) torch
        bad = []
        worst_pcc = 1.0
        for s in range(B):
            a = accept[s]
            p_rows = [prev_rows[s * T + j] for j in range(1, a + 1)]
            c_rows = [rows[s * T + j] for j in range(T)]
            outs_ref = ref.commit_then_drafts(s, p_rows, c_rows)
            st_ref = ttnn.to_torch(ref.state[s])[0]
            hi_ref = ttnn.to_torch(ref.hist[s])[0]
            par = s & 1
            mt_rows, mt_other = _hist_content(hist_t[s], par)
            ref_rows, _ = _hist_content(hi_ref, 0)
            if not torch.equal(h_t[s], st_ref):
                d = (h_t[s] - st_ref).abs()
                bad.append(f"user {s} (a={a}): state mismatches={int((d != 0).sum())} max|d|={d.max().item():.3e}")
            if not torch.equal(mt_rows, ref_rows) or not torch.equal(mt_other, torch.zeros_like(mt_other)):
                bad.append(f"user {s} (a={a}): packed history differs (parity {par})")
            got = out_t[s * T : (s + 1) * T]
            if not torch.equal(got, outs_ref):
                d = (got - outs_ref).abs()
                bad.append(
                    f"user {s} (a={a}): output rows mismatches={int((d != 0).sum())} max|d|={d.max().item():.3e} "
                    f"per-row pcc={[round(_pcc(got[j], outs_ref[j]), 6) for j in range(T)]}"
                )
            outs_torch, h_ref[s], cs_ref[s] = _torch_ref_user(
                cs_ref[s], h_ref[s], p_rows, c_rows, taps, dtb.float(), nea.float(), w.float()
            )
            worst_pcc = min(worst_pcc, min(_pcc(got[j], outs_torch[j]) for j in range(T)), _pcc(h_t[s], h_ref[s]))
        logger.info(
            f"MT B={B} T={T} step {step}: accept={accept} torch pcc(min)={worst_pcc:.6f} "
            f"T=1-op bit-exact={'yes' if not bad else 'NO'}"
        )
        assert not bad, f"B={B} T={T} step {step}: NOT bit-identical to the T=1 op:\n  " + "\n  ".join(bad)
        assert worst_pcc > 0.999, worst_pcc
        # next step: this step's rows become prev, random accept
        ttnn.deallocate(qkv_prev)
        ttnn.deallocate(acc_dev)
        qkv_prev, prev_rows = qkv, rows
        accept = [int(x) for x in torch.randint(0, k + 1, (B,))]
    ttnn.deallocate(qkv_prev)


@pytest.mark.parametrize("B", [2, 8, 32])
@pytest.mark.parametrize("T", [4])
def test_gdn_decode_step_multitoken_vs_batched_t1(device, B, T):
    """All users share one accept value -> the batched T=1 op (user s in row s) applied per token is an exact reference
    in the model's own layout; also exercises the group's row block straddling two tile rows when B*T > 32."""
    torch.manual_seed(2000 + B + T)
    k = T - 1
    R = ((B * T + 31) // 32) * 32
    h_all = (0.05 * torch.randn(BMAX, Nv, Dk, Dv)).float()
    w = (1.0 + 0.1 * torch.randn(Dv)).bfloat16()
    taps = [(0.3 * torch.randn(C)).bfloat16() for _ in range(4)]
    dtb = (0.1 * torch.randn(Nv)).bfloat16()
    nea = (-torch.exp(0.2 * torch.randn(Nv))).bfloat16()
    cs_all = [[(0.5 * torch.randn(C)).bfloat16() for _ in range(4)] for _ in range(BMAX)]
    hist_host = torch.stack([_pack_rows_user(cs_all[b], b) for b in range(BMAX)])
    w_dev = _to_dev(device, w, ttnn.bfloat16)
    taps_dev = _to_dev(device, _pack_rows_user(taps, 0, both=True), ttnn.bfloat16)
    dtb_dev = _to_dev(device, dtb, ttnn.bfloat16)
    nea_dev = _to_dev(device, nea, ttnn.bfloat16)
    state = _to_dev(device, h_all.clone(), ttnn.float32)
    hist_dev = _to_dev(device, hist_host.clone(), ttnn.bfloat16)
    state_r = _to_dev(device, h_all.clone(), ttnn.float32)
    hist_r = _to_dev(device, hist_host.clone(), ttnn.bfloat16)

    def t1_batched(rows_b, st, hi):  # rows_b [B, W]: user s in row s
        out = ttnn.experimental.kda.gdn_decode_step(
            _to_dev(device, rows_b.reshape(1, B, W), ttnn.bfloat16),
            dtb_dev,
            nea_dev,
            st,
            w_dev,
            Nv,
            Nk,
            Dk,
            Dv,
            scale=SCALE,
            output_dtype=ttnn.float32,
            conv_hist=hi,
            conv_taps=taps_dev,
            qkvz_dim=AZ,
        )
        o = ttnn.to_torch(out).reshape(B, -1)
        ttnn.deallocate(out)
        return o

    prev_rows = torch.zeros(B * T, W).bfloat16()
    qkv_prev = _rows_tensor(device, prev_rows, R)
    a = 0
    for step in range(STEPS):
        rows = (0.5 * torch.randn(B * T, W)).bfloat16()
        rows[:, AZ + 2 * Nv :] = 0
        qkv = _rows_tensor(device, rows, R)
        acc_dev = ttnn.from_torch(
            torch.full((1, B), a, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        out = ttnn.experimental.kda.gdn_decode_step(
            qkv,
            dtb_dev,
            nea_dev,
            state,
            w_dev,
            Nv,
            Nk,
            Dk,
            Dv,
            scale=SCALE,
            output_dtype=ttnn.float32,
            conv_hist=hist_dev,
            conv_taps=taps_dev,
            qkvz_dim=AZ,
            num_tokens=T,
            qkv_prev=qkv_prev,
            accept=acc_dev,
        )
        out_t = ttnn.to_torch(out).reshape(R, Nv * Dv)[: B * T].reshape(B, T, -1)
        ttnn.deallocate(out)
        # reference: batched T=1 steps over prev rows 1..a, cur row 0 (committed), then drafts on clones
        grid_prev = prev_rows.reshape(B, T, W)
        grid_cur = rows.reshape(B, T, W)
        for j in range(1, a + 1):
            t1_batched(grid_prev[:, j], state_r, hist_r)
        outs = [t1_batched(grid_cur[:, 0], state_r, hist_r)]
        st_c, hi_c = ttnn.clone(state_r), ttnn.clone(hist_r)
        for j in range(1, T):
            outs.append(t1_batched(grid_cur[:, j], st_c, hi_c))
        ttnn.deallocate(st_c)
        ttnn.deallocate(hi_c)
        outs = torch.stack(outs, dim=1)  # [B, T, Nv*Dv]
        same_state = torch.equal(ttnn.to_torch(state), ttnn.to_torch(state_r))
        same_hist = torch.equal(ttnn.to_torch(hist_dev), ttnn.to_torch(hist_r))
        same_out = torch.equal(out_t, outs)
        d = (out_t - outs).abs()
        logger.info(
            f"MT-vs-batched-T1 B={B} T={T} step {step} accept={a}: state={same_state} hist={same_hist} out={same_out} "
            f"(out mismatches={int((d != 0).sum())} max|d|={d.max().item():.3e})"
        )
        assert same_state and same_hist and same_out
        ttnn.deallocate(qkv_prev)
        ttnn.deallocate(acc_dev)
        qkv_prev, prev_rows = qkv, rows
        a = int(torch.randint(0, k + 1, (1,)))
    ttnn.deallocate(qkv_prev)
