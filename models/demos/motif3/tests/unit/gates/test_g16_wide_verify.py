# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G-S1w and G16-lite — the 64-row (T64) speculative verify step at op / module level (docs/p5_t64/P5_T64_DESIGN.md
§6.2 "G-S1w", §4.1-§4.4, T1-T4; design note docs/p5_t64/t64.md §2, §5.1, §6).

T64 gives each DP row 16 physical rows, still one 32-row tile row: rows 0..7 hold the anchors of the row's 8 lanes at
``n`` (-1 = idle lane), rows 8..15 their drafts at ``n + 1`` (-1 = no draft), each through its owner lane's page-table
row. Only the KV write, FlashMLA, the MoE (M = 64) and the LM heads see more work. Standalone on B0's builders: the
T64 KV writers and the M = 64 configs (C1a: ``cfg.router_decode_pc / experts_gate_up_pc / experts_down_pc /
lm_head_pc(m_tiles=2)``) are injected into this process (module instances / a context-managed op wrapper) exactly like
the G16-lite probes (``docs/p5_t64/scripts/t64_bench*.py``); K1 / D1 / A2 repeat them with the real modules.

(a) ``test_gs1w_kv_write`` (G-S1w a): the KV write at 16 rows per DP row, latent ``[1, 1, 16, 576]`` per DP row, in
    three orders: **split** (KV-R, the production candidate: ROW_MAJOR view ``[1, 2, 8, 576]``, ``MotifCCL.all_gather``
    dim 2 over DP -> ``[1, 1, 64, 576]`` = rows 0..31 anchors / 32..63 drafts in T32 lane order; call A on the anchor
    half with ``cur_a [32]``, call B on the draft half with ``cur_b [32]``, one replicated ``pt [32, W]``), **natural**
    (KV-R: ``ccl.ag_dp_rows`` rows ``16 dp + j``, per 32-row chunk call A + call B = 4 calls) and **row_split** (no
    KV-R: per DP row ``[1, 16, 1, 576]`` on 16 cores, call A then call B); bfp8 and bf16 caches (bf16: <= 16 users per
    call, review R-E8); anchors at ``p % 64`` in {0, 30, 31, 62, 63}, drafts at ``p + 1``, idle lanes, lanes without a
    draft. Every chip's whole cache must equal B0's host model (``kv_write.apply_kv_writes_host`` on the 64 physical
    rows, ``lanes_per_row=16``; it equals an independent direct oracle). The one-call variants (row_split: one 16-user
    call per DP row; natural: one 32-user call per chunk) document the race. Trace: the split writer captured, then
    replayed with rewritten inputs, equals the host model. CB end of every writer below an L1 pin.
(b) ``test_gs1w_flashmla`` (G-S1w b): FlashMLA decode with duplicated page-table rows (the drafts carry their owner's
    row), SWA (window 129) and global, contexts n ~ 1000 / 4095 / 32000 with ``n % 64`` in {0, 31, 63} among the 8
    users: option A (one ``Q [1, 16, 10, 576]`` call, ``cur_pos [16] = (n.., n+1..)``), option A' (two
    ``Q [1, 8, 10, 576]`` calls on slices of the 16-row Q, outputs concatenated on dim 1) and the T32 reference (B = 8
    calls). Random data: A' bitwise == B = 8 on both kinds, A bitwise on SWA, A global PCC >= 0.9999 (documented
    non-bitwise), PCC vs fp64 (overall >= 0.9998, worst user >= 0.9995: G12 bars). Probes (|delta| <= 1e-3): row p never
    sees p + 1, row p + 1 sees p (G12 pair probe); window edges: the anchor attends p - 128 and not p - 129, the draft
    attends p - 127 and not p - 128, neither sees its future key. Then the traced per-call costs (G16-lite).
(c) ``test_g16lite_ops``: (i) gathers: ``ag_dp_rows`` at 8 / 16 rows and the split-order gather, W 4096 / 576: order vs
    host, replicas, cost; (ii) LM head GEMM ``[64, 4096] @ [4096, 6880]`` with ``cfg.lm_head_pc("mesh", m_tiles=2)``:
    rows 0..31 bitwise == the M = 32 GEMM on every chip; the argmax at 64 rows (64-row constants) == the host argmax of
    the device logits and == the 32-row argmax on rows 0..31; costs; (iii) the MoE of layer 2 (real weights, TT cache)
    at 16 rows per DP row with the C1a M = 64 configs (router per_core_M 2 + fused sigmoid, a 64-row top-k pad, gate_up
    10 x 8, down 8 x 8): rows 0..7 of every DP row bitwise == the M = 32 module, costs.
(d) ``test_g16lite_layers``: whole decoder layers L2 (SWA + MoE) and L4 (global + MoE) from the TT cache: T32 (the
    production ``DecodeKVWrite`` all_split at 8 rows per DP row) vs T64 (16 rows: the split-order writer, the MoE at
    M = 64, FlashMLA A or A'): T64 anchor / draft rows bitwise == the T32 rows of the same tokens / positions / cache
    (A'' = A on SWA, A' on global must be bitwise), traced layer costs at 4K context.

Every case also checks: non-finite values 0, replicas, CB end below a one-page L1 pin (track B's method), and
``ring_gather="safe"`` (every gather goes through ``MotifCCL``, F3 rule R1). Pass (G-S1w): (a) bitwise on every chip
and the one-call variants lose updates; (b) as listed. Kill (G-S1w): fail -> no T64 (packed verify stays; escalate).
G16-lite has no kill bar of its own (G16 owns it: T64 / T32 <= 1.20x per step, kill > 1.30x); its numbers feed the
step model of t64.md §5.2.

Run (device; from a B0 + C1a snapshot, GATES_RESULTS.md §13; results land in ``results/G16.jsonl``)::

    scripts/devrun.sh -t 2400 -n g16 -- bash -c "cd <snap> && MOTIF3_GATES_RESULTS_DIR=<live>/results \
        PYTHONPATH=<snap>:<tt-metal> python -m pytest \
        <snap>/models/demos/motif3/tests/unit/gates/test_g16_wide_verify.py -c <snap>/pytest.ini --rootdir <snap> \
        -s -p no:cacheprovider -k 'not host'"
    scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q \
        models/demos/motif3/tests/unit/gates/test_g16_wide_verify.py -k host
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import subprocess
import time
import types
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu

QUICK = gu.env_flag("MOTIF3_GATES_QUICK")
GATE = "G16_quick" if QUICK else "G16"
REC = gu.Recorder(GATE)
# a snapshot run (GATES_RESULTS.md §13) points MOTIF3_GATES_RESULTS_DIR at the live tree's results/
RESULTS_DIR = Path(os.environ.get("MOTIF3_GATES_RESULTS_DIR") or Path(__file__).resolve().parent / "results")
REC.path = RESULTS_DIR / f"{GATE}.jsonl"
MESH = gu.mesh_params(trace_region_size=256 << 20, l1_small_size=32768)

LANES, ROWS, LPR = 32, 4, 8
WR = 2 * LPR  # 16 physical rows per DP row: [8 anchors | 8 drafts]
WROWS = ROWS * WR  # 64
D, NH, DV, BS, D_ROPE = 576, 10, 512, 64, 64
OFFSETS = (0, 30, 31, 62, 63)  # same tile at 0 / 30 / 62 (p, p + 1), tile seam 31, block seam 63
W_KV = 8  # KV-write gate: blocks per lane (positions < 512), the G12 geometry
N_KV = LANES * W_KV + 8  # 264 blocks: the whole cache is compared on all 32 chips
N_BIG, W_FL = 4129, 512  # FlashMLA gate: the serving pool and page-table width
CONTEXTS = (1000, 4095, 32000)
TRIALS = 2 if QUICK else 3
PROBE_TOL = 1e-3
A_PROBE = 16.0  # probe query magnitude along its direction
LAYERS = (2, 4)  # L2 = SWA + MoE, L4 = global + MoE
CTX_TIMING, CTX_BITS = 4096, 1000
if QUICK:
    CONTEXTS = (4095,)
CLASH_RE = re.compile(r"region ends at (\d+)")


# ======================================================================================================================
# host layout of one T64 verify step (pure torch)
# ======================================================================================================================
def phys_row(r: int, j: int, draft: bool) -> int:
    """Physical (natural-order) row of DP row r's local lane j: anchor ``16 r + j``, draft ``16 r + 8 + j``."""
    return WR * r + (LPR if draft else 0) + j


class WideStep:
    """One T64 verify step (design §4.1). Lane ``l = 8 r + j``: anchor at ``anchor[l]`` (-1 idle), draft at
    ``draft[l] = anchor[l] + 1`` (-1 none), both through ``pt[l]``. Random layout (``seed``): per DP row local lane 7
    idle, lane 6 anchor without a draft, lanes 0..5 anchor + draft; ``p % 64`` cycles through :data:`OFFSETS`."""

    def __init__(self, anchor: torch.Tensor, draft: torch.Tensor, pt: torch.Tensor):
        self.anchor = anchor.to(torch.int32).clone()
        self.draft = draft.to(torch.int32).clone()
        self.pt = pt.to(torch.int32).clone()
        ok = (self.draft < 0) | ((self.anchor >= 0) & (self.draft == self.anchor + 1))
        if not bool(ok.all()):
            raise ValueError("a draft must sit at its active anchor + 1")

    @classmethod
    def random(cls, seed: int, W: int = W_KV) -> "WideStep":
        g = torch.Generator().manual_seed(seed)
        pt = (torch.randperm(LANES * W, generator=g) + 1).to(torch.int32).reshape(LANES, W)
        anchor = torch.full((LANES,), -1, dtype=torch.int32)
        draft = torch.full((LANES,), -1, dtype=torch.int32)
        for r in range(ROWS):
            for j in range(LPR):
                if j == LPR - 1:
                    continue  # idle lane
                l = LPR * r + j
                blk = int(torch.randint(0, W - 1, (1,), generator=g))
                p = BS * blk + OFFSETS[(3 * r + j) % len(OFFSETS)]
                anchor[l] = p
                if j != LPR - 2:
                    draft[l] = p + 1
        return cls(anchor, draft, pt)

    @property
    def W(self) -> int:
        return int(self.pt.shape[1])

    def phys_positions(self) -> torch.Tensor:
        """``[64]`` natural-order positions: anchors at ``16 r + j``, drafts at ``16 r + 8 + j``."""
        out = torch.full((WROWS,), -1, dtype=torch.int32)
        for l in range(LANES):
            r, j = divmod(l, LPR)
            out[phys_row(r, j, False)] = self.anchor[l]
            out[phys_row(r, j, True)] = self.draft[l]
        return out

    def phys_page_table(self) -> torch.Tensor:
        """``[64, W]``: both rows of lane l carry ``pt[l]`` (a draft reads its owner's history)."""
        out = torch.zeros(WROWS, self.W, dtype=torch.int32)
        for l in range(LANES):
            r, j = divmod(l, LPR)
            out[phys_row(r, j, False)] = self.pt[l]
            out[phys_row(r, j, True)] = self.pt[l]
        return out

    def is_draft_row(self) -> torch.Tensor:
        return torch.tensor([(i % WR) >= LPR for i in range(WROWS)])

    def kv_step(self):
        """B0's ``KVWriteStep.packed_verify`` on the 64 physical rows: the draft rows are the anchors' partners
        (``partner_of = {16 r + j: 16 r + 8 + j}``; what K1's ``KVWriteStep.wide_verify`` will build)."""
        from models.demos.motif3.tt.kv_write import KVWriteStep

        pos = torch.where(self.is_draft_row(), torch.full((WROWS,), -1, dtype=torch.int32), self.phys_positions())
        partner_of = {}
        for l in range(LANES):
            if int(self.draft[l]) >= 0:
                r, j = divmod(l, LPR)
                partner_of[phys_row(r, j, False)] = phys_row(r, j, True)
        return KVWriteStep.packed_verify(pos, self.phys_page_table(), partner_of)

    def writes(self, dp_rows=None):
        """``[(physical row, block id, row in block)]`` of every write (optionally only those of ``dp_rows``)."""
        out = []
        pos, pt = self.phys_positions(), self.phys_page_table()
        for i in range(WROWS):
            if dp_rows is not None and i // WR not in dp_rows:
                continue
            p = int(pos[i])
            if p >= 0:
                out.append((i, int(pt[i, p // BS]), p % BS))
        return out


def expected_direct(base: torch.Tensor, step: WideStep, rows64: torch.Tensor, replicated: bool):
    """Independent oracle: per DP row, the base cache with every anchor at n and every draft at n + 1 written (KV-R:
    all 64 rows on every chip; row_split: the row's own 16)."""
    out = []
    for r in range(ROWS):
        c = base.clone()
        for i, b, o in step.writes(None if replicated else (r,)):
            c[b, 0, o] = rows64[i]
        out.append(c)
    return out


def expected_host_model(base, step: WideStep, rows64, mode: str, lanes_per_call):
    """B0's host model: ``check_kv_write_step`` + ``apply_kv_writes_host`` with 16 rows per DP row."""
    from models.demos.motif3.tt.kv_write import apply_kv_writes_host, check_kv_write_step

    ks = step.kv_step()
    check_kv_write_step(ks, mode, block_size=BS, lanes_per_call=lanes_per_call, lanes_per_row=WR)
    return apply_kv_writes_host(base, ks, mode, rows64, block_size=BS, lanes_per_call=lanes_per_call, lanes_per_row=WR)


def split_order(x64: torch.Tensor) -> torch.Tensor:
    """Natural order (rows ``16 r + j``) -> split order: rows 0..31 = every DP row's 8 anchors (lane order ``8 r + j``),
    rows 32..63 = their drafts."""
    v = x64.reshape(ROWS, 2, LPR, *x64.shape[1:])
    return torch.cat([v[:, 0].reshape(LANES, *x64.shape[1:]), v[:, 1].reshape(LANES, *x64.shape[1:])])


# ----------------------------------------------------------------------------------------------------------------------
# FlashMLA cases and probes (pure torch)
# ----------------------------------------------------------------------------------------------------------------------
def flash_positions(base: int):
    """8 anchor positions around ``base`` covering ``n % 64`` in {0, 31, 63} (window edges at block / tile seams)."""
    b64 = (base // BS) * BS
    return [base, b64, b64 + 31, b64 + 63, b64 - 64, b64 - 33, b64 - 1, base + 1]


class FlashCase:
    """8 users (one B = 8 FlashMLA call, the T32 layout of a DP row), anchors at n, drafts at n + 1 on the owner's
    page-table row ``[8, 512]``; each user owns the blocks of ``[0, n + 3)`` in a real-size cache."""

    def __init__(self, base: int, seed: int):
        self.base = base
        self.pos = torch.tensor(flash_positions(base), dtype=torch.int32)
        g = torch.Generator().manual_seed(seed)
        perm = (torch.randperm(N_BIG - 1, generator=g) + 1).to(torch.int32)
        self.need = [(int(p) + 2) // BS + 1 for p in self.pos]  # p + 2: the draft's future probe key
        if sum(self.need) > N_BIG - 1 or max(self.need) > W_FL:
            raise ValueError("flash case does not fit the pool / page table")
        self.pt = torch.zeros(LPR, W_FL, dtype=torch.int32)
        off = 0
        for u, n in enumerate(self.need):
            self.pt[u, :n] = perm[off : off + n]
            off += n
        self.n_blocks_used = off

    def scatter(self, seqs) -> torch.Tensor:
        """Per-user sequences ``[need * 64, 576]`` -> host cache ``[4129, 1, 64, 576]`` bf16 (zeros elsewhere)."""
        host = torch.zeros(N_BIG, 1, BS, D, dtype=torch.bfloat16)
        for u, s in enumerate(seqs):
            ids = self.pt[u, : self.need[u]].long()
            host[ids, 0] = s.reshape(self.need[u], BS, D).to(torch.bfloat16)
        return host

    def seq(self, host: torch.Tensor, u: int) -> torch.Tensor:
        return host[self.pt[u, : self.need[u]].long(), 0].reshape(-1, D).float()


def _quant_bfp8(x: torch.Tensor) -> torch.Tensor:
    """bfp8-exact values (the device cache content). ``ttnn.from_torch`` to bfloat8_b needs the device runtime, so the
    host self-checks set :data:`QUANT` to a bf16 rounding instead."""
    return gu.host_roundtrip(x.reshape(-1, BS, D), ttnn.bfloat8_b).reshape(x.shape)


QUANT = {"fn": _quant_bfp8}


def flash_random(case: FlashCase, seed: int, scale: float):
    """Random cache content (N(0, 1) * 0.5, bfp8-exact) and random queries (scale folded in, op scale 1.0)."""
    g = torch.Generator().manual_seed(seed)
    seqs = [QUANT["fn"](torch.randn(n * BS, D, generator=g) * 0.5) for n in case.need]
    qa = (torch.randn(LPR, NH, D, generator=g) * scale).bfloat16().float()
    qd = (torch.randn(LPR, NH, D, generator=g) * scale).bfloat16().float()
    return case.scatter(seqs), qa, qd


def _background(n_rows: int, g) -> torch.Tensor:
    s = torch.zeros(n_rows, D)
    s[:, :DV] = 0.05 * torch.randn(n_rows, DV, generator=g)
    s[:, DV:] = 0.02 * torch.randn(n_rows, D_ROPE, generator=g)
    return s


def flash_probe_pair(case: FlashCase, seed: int):
    """G12's pair probe: key n carries V ``[0 | +1]`` (dims [:256] | [256:512]) and key n + 1 ``[+3 | 0]``, both with
    score 30 along the user's direction; anchor (at n) and draft (at n + 1) query along it. Expected: anchor
    ``[0 | 1]`` (never sees n + 1), draft ``[1.5 | 0.5]`` (sees n)."""
    g = torch.Generator().manual_seed(seed)
    a = A_PROBE
    seqs, q = [], torch.zeros(LPR, NH, D)
    for u, n in enumerate(case.need):
        s = _background(n * BS, g)
        e = torch.randn(D_ROPE, generator=g)
        e = e / e.norm()
        p = int(case.pos[u])
        s[p, :256], s[p, 256:DV], s[p, DV:] = 0.0, 1.0, (30.0 / a) * e
        s[p + 1, :256], s[p + 1, 256:DV], s[p + 1, DV:] = 3.0, 0.0, (30.0 / a) * e
        seqs.append(QUANT["fn"](s))
        q[u, :, DV:] = a * e
    q = q.bfloat16().float()
    return case.scatter(seqs), q, q.clone()


DIMS_A, DIMS_D = slice(0, 16), slice(16, 32)


def flash_probe_edges(case: FlashCase, seed: int, window):
    """Window / causal edge probes with an orthonormal pair (e_A, e_D) per user: the anchor queries along e_A, the draft
    along e_D; each key's score along a direction is its probe score (others ~0). SWA (window 129): anchor (at p) -
    key p - 128 (+1 in dims A, score 20) must win, p - 129 (-1, 28) and p + 1 (+3, 36) must be masked; draft (at p + 1)
    - key p - 127 (+1 in dims D, 20) must win, p - 128 (-1 in dims D, 28) and p + 2 (+3, 36) must be masked. Global:
    anchor - key p (+1, 20) wins, p + 1 (+3, 36) masked; draft - key p + 1 (+1, 20) wins, p + 2 (+3, 36) masked.
    Expected output: +1 in the row's own 16 value dims."""
    g = torch.Generator().manual_seed(seed)
    a = A_PROBE
    seqs = []
    qa, qd = torch.zeros(LPR, NH, D), torch.zeros(LPR, NH, D)
    for u, n in enumerate(case.need):
        s = _background(n * BS, g)
        E, _ = torch.linalg.qr(torch.randn(D_ROPE, 2, generator=g))
        eA, eD = E[:, 0], E[:, 1]
        p = int(case.pos[u])

        def key(pos, direction, score, dims, val):
            s[pos, DV:] += (score / a) * direction
            s[pos, dims] = val

        if window is not None:
            key(p - 128, eA, 20.0, DIMS_A, 1.0)  # anchor's oldest key: attended
            key(p - 128, eD, 28.0, DIMS_D, -1.0)  # ... and just outside the draft's window
            key(p - 129, eA, 28.0, DIMS_A, -1.0)  # just outside the anchor's window
            key(p - 127, eD, 20.0, DIMS_D, 1.0)  # draft's oldest key: attended
            key(p + 1, eA, 36.0, DIMS_A, 3.0)  # anchor's future (the draft's own key)
            key(p + 2, eD, 36.0, DIMS_D, 3.0)  # draft's future
        else:
            key(p, eA, 20.0, DIMS_A, 1.0)
            key(p + 1, eA, 36.0, DIMS_A, 3.0)
            key(p + 1, eD, 20.0, DIMS_D, 1.0)
            key(p + 2, eD, 36.0, DIMS_D, 3.0)
        seqs.append(QUANT["fn"](s))
        qa[u, :, DV:] = a * eA
        qd[u, :, DV:] = a * eD
    return case.scatter(seqs), qa.bfloat16().float(), qd.bfloat16().float()


@torch.no_grad()
def mla_golden(q: torch.Tensor, seq: torch.Tensor, p: int, window) -> torch.Tensor:
    """fp64 latent decode attention of one user (scale folded into q): keys ``[lo, p]``, V = the first 512 columns."""
    lo = 0 if window is None else max(0, p - window + 1)
    k = seq[lo : p + 1].double()
    a = torch.softmax(q.double() @ k.T, dim=-1)
    return a @ k[:, :DV]


def pair_verdicts(out_a: torch.Tensor, out_d: torch.Tensor):
    """Pair probe verdicts per user: anchor ``[0 | 1]``, draft ``[1.5 | 0.5]`` (means over heads)."""
    res, worst = [], 0.0
    for u in range(out_a.shape[0]):
        a0, a1 = float(out_a[u, :, :256].mean()), float(out_a[u, :, 256:DV].mean())
        d0, d1 = float(out_d[u, :, :256].mean()), float(out_d[u, :, 256:DV].mean())
        dev = max(abs(a0), abs(a1 - 1.0), abs(d0 - 1.5), abs(d1 - 0.5))
        worst = max(worst, dev)
        va = "ok" if max(abs(a0), abs(a1 - 1.0)) <= PROBE_TOL else ("SEES_P+1" if abs(a0 - 1.5) < 0.3 else "OFF")
        vd = "ok" if max(abs(d0 - 1.5), abs(d1 - 0.5)) <= PROBE_TOL else ("MISSES_P" if abs(d0 - 3.0) < 0.3 else "OFF")
        res.append((u, va, vd))
    return res, worst


def edge_verdicts(out_a: torch.Tensor, out_d: torch.Tensor):
    """Edge probe verdicts per user: +1 in the row's own dims (A for the anchor, D for the draft)."""

    def one(m: torch.Tensor) -> str:
        mean, dev = float(m.mean()), float((m - 1.0).abs().max())
        if dev <= PROBE_TOL:
            return "ok"
        if abs(mean + 1.0) < 0.5:
            return "WINDOW_LEAK"
        if abs(mean - 3.0) < 0.75:
            return "CAUSAL_LEAK"
        return f"OFF({mean:.4f})"

    res, worst = [], 0.0
    for u in range(out_a.shape[0]):
        ma, md = out_a[u, :, DIMS_A], out_d[u, :, DIMS_D]
        worst = max(worst, float((ma - 1.0).abs().max()), float((md - 1.0).abs().max()))
        res.append((u, one(ma), one(md)))
    return res, worst


# ----------------------------------------------------------------------------------------------------------------------
# host self-checks
# ----------------------------------------------------------------------------------------------------------------------
def test_g16_host_wide_step_and_host_model():
    """The T64 layout, B0's host model at 16 rows per DP row (all modes and call widths) == the direct oracle, and the
    split-order permutation."""
    st = WideStep.random(seed=5)
    pos, pt = st.phys_positions(), st.phys_page_table()
    for l in range(LANES):
        r, j = divmod(l, LPR)
        a, d = phys_row(r, j, False), phys_row(r, j, True)
        assert torch.equal(pt[a], pt[d]) and int(pos[a]) == int(st.anchor[l]) and int(pos[d]) == int(st.draft[l])
    offs = {int(p) % BS for p in st.anchor if int(p) >= 0}
    assert offs == set(OFFSETS), offs
    assert int((st.anchor < 0).sum()) == ROWS and int((st.draft < 0).sum()) == 2 * ROWS
    g = torch.Generator().manual_seed(6)
    base = torch.randn(N_KV, 1, BS, D, generator=g)
    rows = torch.randn(WROWS, D, generator=g)
    for mode, n, rep in (("all_split", 32, True), ("all_split", 16, True), ("row_split", None, False)):
        hm = expected_host_model(base, st, rows, mode, n)
        dr = expected_direct(base, st, rows, rep)
        assert all(torch.equal(a, b) for a, b in zip(hm, dr)), (mode, n)
    # a one-call variant (anchor and draft of a lane in one call) is rejected by the host model (the G12 race)
    from models.demos.motif3.tt.kv_write import KVWriteStep, check_kv_write_step

    with pytest.raises(ValueError):
        check_kv_write_step(KVWriteStep.ordinary(pos, pt), "all", block_size=BS, lanes_per_call=32, lanes_per_row=WR)
    x = torch.arange(WROWS)[:, None].float()
    s = split_order(x)[:, 0].long().tolist()
    assert s[:LANES] == [phys_row(r, j, False) for r in range(ROWS) for j in range(LPR)]
    assert s[LANES:] == [phys_row(r, j, True) for r in range(ROWS) for j in range(LPR)]


def _emulate_flash(host, case: FlashCase, qa, qd, window):
    """Host emulation of the per-user FlashMLA semantics (fp64 golden): anchors at n, drafts at n + 1."""
    oa = torch.stack([mla_golden(qa[u], case.seq(host, u), int(case.pos[u]), window) for u in range(LPR)])
    od = torch.stack([mla_golden(qd[u], case.seq(host, u), int(case.pos[u]) + 1, window) for u in range(LPR)])
    return oa.float(), od.float()


def test_g16_host_flash_probes_discriminate(monkeypatch):
    """The probes pass on the correct semantics and catch a wrong window / causal edge (host emulation)."""
    monkeypatch.setitem(QUANT, "fn", lambda x: x.bfloat16().float())
    case = FlashCase(1000, seed=7)
    host, qa, qd = flash_probe_pair(case, seed=8)
    res, worst = pair_verdicts(*_emulate_flash(host, case, qa, qd, 129))
    assert all(r[1] == "ok" and r[2] == "ok" for r in res), res
    assert worst <= PROBE_TOL, worst
    # a draft that cannot see its anchor (n), or an anchor that sees n + 1, fails the pair probe
    oa, od = _emulate_flash(host, case, qa, qd, 129)
    bad_anchor = torch.stack([mla_golden(qa[u], case.seq(host, u), int(case.pos[u]) + 1, 129) for u in range(LPR)])
    assert all(v[1] != "ok" for v in pair_verdicts(bad_anchor.float(), od)[0])
    for window in (129, None):
        host, qa, qd = flash_probe_edges(case, seed=9, window=window)
        res, worst = edge_verdicts(*_emulate_flash(host, case, qa, qd, window))
        assert all(r[1] == "ok" and r[2] == "ok" for r in res), (window, res)
        assert worst <= PROBE_TOL, (window, worst)
    host, qa, qd = flash_probe_edges(case, seed=9, window=129)
    for wrong in (128, 130):
        res, _ = edge_verdicts(*_emulate_flash(host, case, qa, qd, wrong))
        assert all(r[1] != "ok" for r in res), (wrong, res)


# ======================================================================================================================
# device helpers
# ======================================================================================================================
def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:400]}"


def _setup(mesh_device, tag: str):
    """``(cfg, ccl)`` of the mesh (B0 + C1a builders, ``ring_gather`` from the config: "safe") + a provenance record."""
    import models.demos.motif3 as m3
    from models.demos.motif3.tt import ccl as ccl_mod
    from models.demos.motif3.tt import generator_api, kv_write, model_config
    from models.demos.motif3.tt.ccl import MotifCCL, log_fabric
    from models.demos.motif3.tt.model_config import MotifTTConfig

    cfg = MotifTTConfig.from_hf_config(None, mesh_device=mesh_device)
    ccl = MotifCCL(mesh_device, cfg)
    try:
        head = subprocess.run(["git", "-C", str(model_config.PROJECT_ROOT / "tt-metal"), "rev-parse", "--short=11",
                               "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()  # fmt: skip
    except Exception:  # pragma: no cover
        head = "?"
    fab = log_fabric(mesh_device, f"G16 {tag}")
    REC.add(f"provenance/{tag}", status="info", motif3_package=str(Path(m3.__file__).parent), git_head=head,
            md5={m.__name__.rsplit(".", 1)[-1]: _md5(m.__file__) for m in (model_config, generator_api, ccl_mod,
                 kv_write)}, fabric_committed=fab.get("committed"), dp_ring=fab.get("dp_ring"),
            l1_small=fab.get("l1_small"), ring_gather=ccl.ring_gather, router_logits=cfg.router_logits,
            kv_cache=cfg.dtypes.kv_cache_name)  # fmt: skip
    if ccl.ring_gather != "safe":
        raise AssertionError(f"G16 runs with ring_gather='safe' (F3 rule R1), got {ccl.ring_gather!r}")
    return cfg, ccl


def up(t: torch.Tensor, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=None):
    return gu.to_mesh(t.contiguous(), mesh_device, dtype, layout=layout, memory_config=memory_config)


def up_i32(t: torch.Tensor, mesh_device):
    return gu.to_mesh(t.contiguous().to(torch.int32), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def host_i32(t: torch.Tensor, mesh_device):
    return ttnn.from_torch(t.contiguous().to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))  # fmt: skip


def rows_dev(x: torch.Tensor, cfg, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=True):
    """``[dp, ...]`` -> per DP row tensor (``rope.shard_lanes``); ``device=False`` = a host tensor for a copy."""
    from models.demos.motif3.tt.rope import shard_lanes

    return shard_lanes(x.contiguous(), cfg, mesh_device, dtype=dtype, layout=layout,
                       device=mesh_device if device else None)  # fmt: skip


def l1_pin(mesh_device):
    """One-page interleaved L1 tensor at the top of main L1, just below L1_SMALL (track B's probe_sp1_cbend)."""
    return ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                           memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))


def run_pinned(mesh_device, fn):
    """``fn()`` once with :func:`l1_pin` alive: a static-CB region reaching the pin raises a clash naming its end.
    Returns ``(output or None on a clash, record)``."""
    pin = l1_pin(mesh_device)
    addr = int(pin.buffer_address())
    try:
        out = fn()
        ttnn.synchronize_device(mesh_device)
        rec = {"cb_check": "fits_below_pin", "pin_addr": addr}
    except Exception as e:
        m = CLASH_RE.search(str(e))
        if m is None:
            raise
        out = None
        rec = {"cb_check": "clash_with_pin", "pin_addr": addr, "cb_end": int(m.group(1)), "error": str(e)[:300]}
    finally:
        ttnn.deallocate(pin)
    return out, rec


def cb_ok(rec: dict) -> bool:
    return rec.get("cb_check") == "fits_below_pin"


def count_programs(mesh_device) -> int:
    ttnn.synchronize_device(mesh_device)
    return int(mesh_device.num_program_cache_entries())


def all_chips(t, mesh_device) -> torch.Tensor:
    from models.demos.motif3.tt.ccl import device_tensors_to_torch

    return device_tensors_to_torch(t, mesh_device).float()


def capture(mesh_device, fn):
    """Exception-safe trace capture: ``(trace id, fn's output)``."""
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        out = fn()
    except BaseException:
        try:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        finally:
            ttnn.release_trace(mesh_device, tid)
        raise
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    return tid, out


# ======================================================================================================================
# (a) KV write at 16 rows per DP row: injected writers (what K1's DecodeKVWrite(rows=64) will issue)
# ======================================================================================================================
def _update(cache, u, cur, pt):
    ttnn.experimental.paged_update_cache(cache, u, update_idxs_tensor=cur, page_table=pt)


class SplitWriter:
    """KV-R, split order (design §4.3, the production candidate): untilize the ``[1, 1, 16, 576]`` latent, view it as
    ``[1, 2, 8, 576]``, all-gather dim 2 over DP (``MotifCCL``: safe ring gathers) -> ``[1, 2, 32, 576]`` = ``[anchors
    of lanes 0..31 | drafts]``, tilize ``[1, 1, 64, 576]``; call A on rows 0..31 (``cur_a``), call B on rows 32..63
    (``cur_b``), one replicated lane-order page table; ``n`` users per call (32 bfp8, 16 bf16: R-E8)."""

    name = "split"

    def __init__(self, mesh_device, ccl, step: WideStep, n: int):
        from models.demos.motif3.tt.kv_write import update_input_memory_config

        self.mesh, self.ccl, self.n = mesh_device, ccl, int(n)
        self.chunks = LANES // self.n
        self.mc = update_input_memory_config(self.n, mesh_device.compute_with_storage_grid_size())
        self.pt, self.cur_a, self.cur_b = [], [], []
        for c in range(self.chunks):
            sl = slice(c * self.n, (c + 1) * self.n)
            self.pt.append(up_i32(step.pt[sl], mesh_device))
            self.cur_a.append(up_i32(step.anchor[sl], mesh_device))
            self.cur_b.append(up_i32(step.draft[sl], mesh_device))
        self.calls_per_layer = 2 * self.chunks

    def set_step(self, step: WideStep):
        for c in range(self.chunks):
            sl = slice(c * self.n, (c + 1) * self.n)
            for src, dst in ((step.pt[sl], self.pt[c]), (step.anchor[sl], self.cur_a[c]), (step.draft[sl],
                                                                                         self.cur_b[c])):
                ttnn.copy_host_to_device_tensor(host_i32(src, self.mesh), dst)

    def gather(self, kv_row):
        rm = ttnn.to_layout(kv_row, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        v = ttnn.reshape(rm, (1, 2, LPR, D))
        g = self.ccl.all_gather(v, 2, "dp", memory_config=ttnn.L1_MEMORY_CONFIG)  # [1, 2, 32, 576]
        gu.free(rm)
        gt = ttnn.to_layout(ttnn.reshape(g, (1, 1, 2 * LANES, D)), ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gu.free(g)
        return gt

    def write(self, kv_row, kv_cache, *, cur_pos=None, page_table=None):
        gt = self.gather(kv_row)
        for half, curs in ((0, self.cur_a), (1, self.cur_b)):
            for c in range(self.chunks):
                r0 = LANES * half + c * self.n
                s = ttnn.slice(gt, [0, 0, r0, 0], [1, 1, r0 + self.n, D])
                u = ttnn.transpose(s, 1, 2, memory_config=self.mc)  # [1, n, 1, 576], one user per core
                gu.free(s)
                _update(kv_cache, u, curs[c], self.pt[c])
                gu.free(u)
        gu.free(gt)

    def end_step(self):
        return None

    def free(self):
        gu.free(self.pt + self.cur_a + self.cur_b)


class NaturalWriter:
    """KV-R, natural order: ``ccl.ag_dp_rows`` -> rows ``16 dp + j``; per chunk of ``n`` rows call A (anchor rows) and
    call B (draft rows) through the chunk's page table (the existing ``num_chunks > 1`` path); ``one_call``: one call
    with both (documents the race)."""

    name = "natural"

    def __init__(self, mesh_device, ccl, step: WideStep, n: int):
        from models.demos.motif3.tt.kv_write import update_input_memory_config

        self.mesh, self.ccl, self.n = mesh_device, ccl, int(n)
        self.chunks = WROWS // self.n
        self.mc = update_input_memory_config(self.n, mesh_device.compute_with_storage_grid_size())
        pos, pt, dr = step.phys_positions(), step.phys_page_table(), step.is_draft_row()
        neg = torch.full_like(pos, -1)
        cur_a, cur_b = torch.where(dr, neg, pos), torch.where(dr, pos, neg)
        self.pt, self.cur_a, self.cur_b, self.cur_one = [], [], [], []
        for c in range(self.chunks):
            sl = slice(c * self.n, (c + 1) * self.n)
            self.pt.append(up_i32(pt[sl], mesh_device))
            self.cur_a.append(up_i32(cur_a[sl], mesh_device))
            self.cur_b.append(up_i32(cur_b[sl], mesh_device))
            self.cur_one.append(up_i32(pos[sl], mesh_device))
        self.calls_per_layer = 2 * self.chunks

    def write(self, kv_row, kv_cache, *, one_call: bool = False, **_):
        g = self.ccl.ag_dp_rows(kv_row)  # [1, 1, 64, 576] TILE, rows 16 dp + j
        for c in range(self.chunks):
            s = ttnn.slice(g, [0, 0, c * self.n, 0], [1, 1, (c + 1) * self.n, D])
            u = ttnn.transpose(s, 1, 2, memory_config=self.mc)
            gu.free(s)
            if one_call:
                _update(kv_cache, u, self.cur_one[c], self.pt[c])
            else:
                _update(kv_cache, u, self.cur_a[c], self.pt[c])
                _update(kv_cache, u, self.cur_b[c], self.pt[c])
            gu.free(u)
        gu.free(g)

    def free(self):
        gu.free(self.pt + self.cur_a + self.cur_b + self.cur_one)


class RowSplitWriter:
    """No KV-R (``row_split``): per DP row the 16 rows on 16 cores, call A (anchors; drafts -1) then call B (drafts),
    the row's ``[16, W]`` page table (rows 8..15 = rows 0..7); ``one_call``: one 16-user call (documents the race)."""

    name = "row_split"

    def __init__(self, mesh_device, cfg, step: WideStep):
        from models.demos.motif3.tt.kv_write import update_input_memory_config

        self.mc = update_input_memory_config(WR, mesh_device.compute_with_storage_grid_size())
        pos, pt, dr = step.phys_positions(), step.phys_page_table(), step.is_draft_row()
        neg = torch.full_like(pos, -1)

        def dev(t):
            return rows_dev(t.contiguous(), cfg, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

        self.pt = dev(pt)  # [16, W] per DP row
        self.cur_a, self.cur_b, self.cur_one = dev(torch.where(dr, neg, pos)), dev(torch.where(dr, pos, neg)), dev(pos)
        self.calls_per_layer = 2

    def write(self, kv_row, kv_cache, *, one_call: bool = False, **_):
        u = ttnn.transpose(kv_row, 1, 2, memory_config=self.mc)  # [1, 16, 1, 576] per DP row
        if one_call:
            _update(kv_cache, u, self.cur_one, self.pt)
        else:
            _update(kv_cache, u, self.cur_a, self.pt)
            _update(kv_cache, u, self.cur_b, self.pt)
        gu.free(u)

    def free(self):
        gu.free([self.pt, self.cur_a, self.cur_b, self.cur_one])


DTYPES = {"bfp8": ttnn.bfloat8_b} if QUICK else {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}


def _quant(x: torch.Tensor, dtype) -> torch.Tensor:
    return gu.host_roundtrip(x, ttnn.bfloat8_b) if dtype == ttnn.bfloat8_b else x.bfloat16().float()


def _fresh_cache(mesh_device, dtype, seed: int):
    g = torch.Generator().manual_seed(seed)
    host = _quant(torch.randn(N_KV, 1, BS, D, generator=g), dtype)
    return host, up(host, mesh_device, dtype)


def _lat16(rows64: torch.Tensor, cfg, mesh_device, device=True):
    """Natural-order rows ``[64, 576]`` -> the per-DP-row latent ``[1, 1, 16, 576]`` bf16 TILE."""
    return rows_dev(rows64.reshape(ROWS, 1, WR, D), cfg, mesh_device, device=device)


def _lost(chip_cache: torch.Tensor, step: WideStep, rows64: torch.Tensor, dp_rows) -> dict:
    out = {"anchor": [], "draft": []}
    for i, b, o in step.writes(dp_rows):
        if not torch.equal(chip_cache[b, 0, o], rows64[i]):
            out["draft" if (i % WR) >= LPR else "anchor"].append((i, o))
    return out


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_gs1w_kv_write(mesh_device):
    from models.demos.motif3.tt.kv_write import cache_mismatches

    torch.set_num_threads(16)
    cfg, ccl = _setup(mesh_device, "gs1w_kv_write")
    failures = []
    step = WideStep.random(seed=160)
    for dname, dtype in DTYPES.items():
        n_rep = 32 if dtype == ttnn.bfloat8_b else 16  # users per call (kv_write.MAX_LANES_PER_CALL, R-E8)
        for kind in ("split", "natural", "row_split"):
            case = f"kv_write/{dname}/{kind}"
            replicated = kind != "row_split"
            try:
                host, cache = _fresh_cache(mesh_device, dtype, seed=gu.seed_of("kvw", dname, kind))
                g = torch.Generator().manual_seed(gu.seed_of("rows", dname, kind))
                rows64 = _quant(torch.randn(WROWS, D, generator=g) * 2.0, dtype)
                lat = _lat16(rows64, cfg, mesh_device)
                w = (SplitWriter(mesh_device, ccl, step, n_rep) if kind == "split" else
                     NaturalWriter(mesh_device, ccl, step, n_rep) if kind == "natural" else
                     RowSplitWriter(mesh_device, cfg, step))  # fmt: skip
                n0 = count_programs(mesh_device)
                _, cb = run_pinned(mesh_device, lambda: w.write(lat, cache))
                if not cb_ok(cb):
                    w.write(lat, cache)
                n1 = count_programs(mesh_device)
                mode = "all_split" if replicated else "row_split"
                want = expected_direct(host, step, rows64, replicated)
                hm = expected_host_model(host, step, rows64, mode, n_rep if replicated else None)
                oracle_agree = all(torch.equal(a, b) for a, b in zip(want, hm))
                bad = cache_mismatches(cache, mesh_device, cfg, want)
                ok = not bad and oracle_agree and cb_ok(cb)
                REC.add(case, status="pass" if ok else "fail", order=kind, cache_dtype=dname, users_per_call=n_rep if
                        replicated else WR, update_calls_per_layer=w.calls_per_layer, chips_checked=32,
                        mismatching_chips={f"{k[0]},{k[1]}": v for k, v in bad.items()},
                        host_model_eq_direct_oracle=oracle_agree, writes=len(step.writes(None if replicated else
                        (0,))), programs_first_call=n1 - n0,
                        same_tile_pairs=sum(1 for l in range(LANES) if int(step.draft[l]) >= 0 and
                                            (int(step.anchor[l]) % BS) in (0, 30, 62)), **cb)  # fmt: skip
                if not ok:
                    failures.append(f"{case}: mismatching chips {bad}, oracle agree {oracle_agree}, cb {cb}")
                gu.free([cache, lat])
                # ---- the one-call variants (race documentation; not gated) ----------------------------------
                if kind in ("natural", "row_split"):
                    per_trial = []
                    host2, cache2 = _fresh_cache(mesh_device, dtype, seed=gu.seed_of("kvw1", dname, kind))
                    for t in range(TRIALS):
                        r2 = _quant(torch.randn(WROWS, D, generator=g) * 2.0, dtype)
                        lat2 = _lat16(r2, cfg, mesh_device)
                        w.write(lat2, cache2, one_call=True)
                        chip0 = gu.read_dev(cache2, 0)
                        lost = _lost(chip0, step, r2, None if replicated else (0,))
                        per_trial.append({"anchors_lost": len(lost["anchor"]), "drafts_lost": len(lost["draft"])})
                        gu.free(lat2)
                    REC.add(f"{case}/single_call_race", status="documented", trials=TRIALS, per_trial=per_trial,
                            lost_total=sum(x["anchors_lost"] + x["drafts_lost"] for x in per_trial),
                            note="anchor (p) and draft (p + 1) of a lane in ONE paged_update_cache call: expected to "
                                 "lose updates on same-tile pairs (G12)")  # fmt: skip
                    gu.free(cache2)
                w.free()
            except Exception as e:
                REC.add(case, status="error", error=_err(e))
                failures.append(f"{case}: {_err(e)}")
    # ---- trace: the split writer captured, then replayed with rewritten inputs (every program compiled above) -------
    try:
        ok, why = _kv_write_trace(mesh_device, cfg, ccl)
        if not ok:
            failures.append(f"kv_write/trace: {why}")
    except Exception as e:
        REC.add("kv_write/bfp8/split/trace_rewritten_inputs", status="error", error=_err(e))
        failures.append(f"kv_write/trace: {_err(e)}")
    assert not failures, "G-S1w (a) failures:\n" + "\n".join(failures)


def _kv_write_trace(mesh_device, cfg, ccl):
    from models.demos.motif3.tt.kv_write import cache_mismatches

    s1, s2 = WideStep.random(seed=171), WideStep.random(seed=172)
    host, cache = _fresh_cache(mesh_device, ttnn.bfloat8_b, seed=173)
    g = torch.Generator().manual_seed(174)
    r1 = _quant(torch.randn(WROWS, D, generator=g) * 2.0, ttnn.bfloat8_b)
    r2 = _quant(torch.randn(WROWS, D, generator=g) * 2.0, ttnn.bfloat8_b)
    lat = _lat16(r1, cfg, mesh_device)
    w = SplitWriter(mesh_device, ccl, s1, 32)
    n0 = count_programs(mesh_device)
    tid, _ = capture(mesh_device, lambda: w.write(lat, cache))
    try:
        # a capture records without executing; restore the base anyway, then step 2's inputs, then replay
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                                                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)), cache)
        w.set_step(s2)
        ttnn.copy_host_to_device_tensor(_lat16(r2, cfg, mesh_device, device=False), lat)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        bad = cache_mismatches(cache, mesh_device, cfg, expected_direct(host, s2, r2, True))
    finally:
        ttnn.release_trace(mesh_device, tid)
    n1 = count_programs(mesh_device)
    ok = not bad and n1 == n0
    REC.add("kv_write/bfp8/split/trace_rewritten_inputs", status="pass" if ok else "fail", chips_checked=32,
            mismatching_chips={f"{k[0]},{k[1]}": v for k, v in bad.items()}, programs_during_capture=n1 - n0)
    w.free()
    gu.free([lat, cache])
    return ok, f"mismatching chips {bad}, programs {n1 - n0}"


# ======================================================================================================================
# (b) FlashMLA with duplicated page-table rows: options A / A' vs the T32 B = 8 calls
# ======================================================================================================================
class FlashRunner:
    """Device inputs of one :class:`FlashCase` (page tables / positions persistent) and the three call forms."""

    def __init__(self, mesh_device, cfg, case: FlashCase):
        self.mesh = mesh_device
        self.pc = cfg.flash_mla_decode_pc()
        self.ckc = cfg.compute_config("sdpa_decode")
        pos = case.pos
        self.pt8 = up_i32(case.pt, mesh_device)
        self.pt16 = up_i32(torch.cat([case.pt, case.pt]), mesh_device)
        self.pa, self.pd = up_i32(pos, mesh_device), up_i32(pos + 1, mesh_device)
        self.p16 = up_i32(torch.cat([pos, pos + 1]), mesh_device)

    def flash(self, q, cache, pt, pos, window):
        return ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q, cache, None, head_dim_v=DV, page_table_tensor=pt, cur_pos_tensor=pos, scale=1.0,
            sliding_window_size=window, program_config=self.pc, compute_kernel_config=self.ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

    def b8(self, qa, qd, cache, window):
        return self.flash(qa, cache, self.pt8, self.pa, window), self.flash(qd, cache, self.pt8, self.pd, window)

    def option_a(self, q16, cache, window):
        return self.flash(q16, cache, self.pt16, self.p16, window)

    def option_a_prime(self, q16, cache, window):
        """What A2 will issue on global layers: slices of the 16-row Q, two B = 8 calls, one concat (dim 1)."""
        qa = ttnn.slice(q16, [0, 0, 0, 0], [1, LPR, NH, D])
        qd = ttnn.slice(q16, [0, LPR, 0, 0], [1, WR, NH, D])
        oa, od = self.flash(qa, cache, self.pt8, self.pa, window), self.flash(qd, cache, self.pt8, self.pd, window)
        out = ttnn.concat([oa, od], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gu.free([qa, qd, oa, od])
        return out

    def free(self):
        gu.free([self.pt8, self.pt16, self.pa, self.pd, self.p16])


def _users(t) -> torch.Tensor:
    """Device 0's ``[1, B, NH(pad), 512]`` -> ``[B, NH, 512]``."""
    return gu.read_dev(t, 0)[0, :, :NH]


def _flash_all(mesh_device, fr: FlashRunner, cache, qa_h, qd_h, window, *, pinned: bool = False):
    """Run B8 (anchors, drafts), A and A'; returns host outputs + replica / CB records."""
    qa, qd = up(qa_h[None], mesh_device), up(qd_h[None], mesh_device)
    q16 = up(torch.cat([qa_h, qd_h])[None], mesh_device)
    ra, rd = fr.b8(qa, qd, cache, window)
    cbs = {}
    if pinned:
        oA, cbs["A"] = run_pinned(mesh_device, lambda: fr.option_a(q16, cache, window))
        oAp, cbs["A_prime"] = run_pinned(mesh_device, lambda: fr.option_a_prime(q16, cache, window))
        oA = oA if oA is not None else fr.option_a(q16, cache, window)
        oAp = oAp if oAp is not None else fr.option_a_prime(q16, cache, window)
    else:
        oA, oAp = fr.option_a(q16, cache, window), fr.option_a_prime(q16, cache, window)
    out = {"ref_a": _users(ra), "ref_d": _users(rd)}
    a, ap = _users(oA), _users(oAp)
    out.update(A_a=a[:LPR], A_d=a[LPR:], Ap_a=ap[:LPR], Ap_d=ap[LPR:])
    reps = {"A": gu.replicas_identical(oA)[0], "A_prime": gu.replicas_identical(oAp)[0]}
    gu.free([qa, qd, q16, ra, rd, oA, oAp])
    return out, reps, cbs


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_gs1w_flashmla(mesh_device):
    from models.demos.motif3.tests.unit.gates import goldens as gd

    torch.set_num_threads(16)
    cfg, _ = _setup(mesh_device, "gs1w_flashmla")
    failures = []
    for base in CONTEXTS:
        case = FlashCase(base, seed=gu.seed_of("flash", base))
        fr = FlashRunner(mesh_device, cfg, case)
        for kind, window, scale in (("swa", 129, gd.SCALE_SWA), ("global", None, gd.SCALE_GLOBAL)):
            tag = f"flashmla/{kind}/ctx{base}"
            try:
                # ---- random data: A' == B8 bitwise, A == B8 bitwise on SWA (PCC on global), PCC vs fp64 ----------
                host, qa_h, qd_h = flash_random(case, seed=gu.seed_of("fr", base, kind), scale=scale)
                cache = up(host, mesh_device, ttnn.bfloat8_b)
                o, reps, cbs = _flash_all(mesh_device, fr, cache, qa_h, qd_h, window, pinned=True)
                gu.free(cache)
                ap_bit = torch.equal(o["Ap_a"], o["ref_a"]) and torch.equal(o["Ap_d"], o["ref_d"])
                a_bit_a, a_bit_d = torch.equal(o["A_a"], o["ref_a"]), torch.equal(o["A_d"], o["ref_d"])
                a_pcc = gu.pcc(torch.cat([o["ref_a"], o["ref_d"]]), torch.cat([o["A_a"], o["A_d"]]))
                a_max = max(float((o["A_a"] - o["ref_a"]).abs().max()), float((o["A_d"] - o["ref_d"]).abs().max()))
                gold_a, gold_d = _emulate_flash(host, case, qa_h, qd_h, window)
                pccs = [gu.pcc(gold_a[u], o["ref_a"][u]) for u in range(LPR)] + [
                    gu.pcc(gold_d[u], o["ref_d"][u]) for u in range(LPR)]  # fmt: skip
                overall = gu.pcc(torch.cat([gold_a, gold_d]), torch.cat([o["ref_a"], o["ref_d"]]))
                nonfinite = sum(int((~torch.isfinite(v)).sum()) for v in o.values())
                a_ok = (a_bit_a and a_bit_d) if kind == "swa" else (a_pcc >= 0.9999)
                ok = (ap_bit and a_ok and overall >= 0.9998 and min(pccs) >= 0.9995 and nonfinite == 0
                      and all(reps.values()) and all(cb_ok(c) for c in cbs.values()))  # fmt: skip
                REC.add(f"{tag}/random", status="pass" if ok else "fail", positions=case.pos.tolist(),
                        pos_mod64=[int(p) % BS for p in case.pos], option_a_prime_bitwise_eq_b8=ap_bit,
                        option_a_bitwise_anchors=a_bit_a, option_a_bitwise_drafts=a_bit_d, option_a_pcc_vs_b8=a_pcc,
                        option_a_max_abs_vs_b8=a_max, pcc_vs_fp64_overall=overall, worst_user_pcc=min(pccs),
                        nonfinite=nonfinite, replicas_identical_32=reps, **{f"cb_{k}": v["cb_check"] for k, v in
                        cbs.items()})  # fmt: skip
                if not ok:
                    failures.append(f"{tag}/random: A' bitwise {ap_bit}, A {a_bit_a}/{a_bit_d} pcc {a_pcc:.6f}, "
                                    f"overall {overall:.6f} worst {min(pccs):.6f}, reps {reps}, cb {cbs}")
                # ---- probes: the pair probe (G12) and the window / causal edges ----------------------------------
                for pname, builder in (("pair", lambda: flash_probe_pair(case, seed=gu.seed_of("pp", base, kind))),
                                       ("edges", lambda: flash_probe_edges(case, seed=gu.seed_of("pe", base, kind),
                                                                           window=window))):  # fmt: skip
                    host, qa_h, qd_h = builder()
                    cache = up(host, mesh_device, ttnn.bfloat8_b)
                    o, reps, _ = _flash_all(mesh_device, fr, cache, qa_h, qd_h, window)
                    gu.free(cache)
                    verd = {}
                    worst = 0.0
                    for form, (oa, od) in (("b8", (o["ref_a"], o["ref_d"])), ("A", (o["A_a"], o["A_d"])),
                                           ("A_prime", (o["Ap_a"], o["Ap_d"]))):  # fmt: skip
                        res, w = pair_verdicts(oa, od) if pname == "pair" else edge_verdicts(oa, od)
                        verd[form] = [r for r in res if r[1] != "ok" or r[2] != "ok"]
                        worst = max(worst, w)
                    ok = all(not v for v in verd.values()) and worst <= PROBE_TOL
                    REC.add(f"{tag}/probe_{pname}", status="pass" if ok else "fail", failing=verd, worst_dev=worst,
                            positions=case.pos.tolist(), replicas_identical_32=reps)  # fmt: skip
                    if not ok:
                        failures.append(f"{tag}/probe_{pname}: {verd} worst {worst:.2e}")
            except Exception as e:
                REC.add(tag, status="error", error=_err(e))
                failures.append(f"{tag}: {_err(e)}")
        # ---- costs (G16-lite): every program was compiled by the first context's checks above ------------------------
        try:
            _flash_costs(mesh_device, cfg, fr, case, base)
        except Exception as e:
            REC.add(f"flashmla/cost/ctx{base}", status="error", error=_err(e))
            failures.append(f"flashmla/cost/ctx{base}: {_err(e)}")
        fr.free()
    assert not failures, "G-S1w (b) failures:\n" + "\n".join(failures)


def _flash_costs(mesh_device, cfg, fr: FlashRunner, case: FlashCase, base: int):
    """Traced per-call cost (slope method) of B8, A (B16) and A' (2 x B8 + slices + concat), SWA and global."""
    from models.demos.motif3.tests.unit.gates import goldens as gd

    for kind, window, scale in (("swa", 129, gd.SCALE_SWA), ("global", None, gd.SCALE_GLOBAL)):
        host, qa_h, qd_h = flash_random(case, seed=gu.seed_of("fc", base, kind), scale=scale)
        cache = up(host, mesh_device, ttnn.bfloat8_b)
        qa = up(qa_h[None], mesh_device)
        q16 = up(torch.cat([qa_h, qd_h])[None], mesh_device)
        res = {}
        for form, fn in (("b8", lambda: fr.flash(qa, cache, fr.pt8, fr.pa, window)),
                         ("A_b16", lambda: fr.option_a(q16, cache, window)),
                         ("A_prime_2xb8", lambda: fr.option_a_prime(q16, cache, window))):  # fmt: skip
            us, raw = gu.time_traced(mesh_device, fn, ops_per_trace=32, reps=7, compile_first=True)
            res[form] = round(us, 2)
        REC.add(f"flashmla/cost/{kind}/ctx{base}", status="info", traced_us=res, delta_A_us=round(res["A_b16"] -
                res["b8"], 2), delta_A_prime_us=round(res["A_prime_2xb8"] - res["b8"], 2))  # fmt: skip
        gu.free([cache, qa, q16])


# ======================================================================================================================
# (c) G16-lite ops: gathers, LM head + argmax at 64 rows, MoE at M = 64 (C1a configs injected)
# ======================================================================================================================
class MoEM64:
    """The C1a M = 64 decode configs injected into one ``MotifMoE`` instance (what D1 will make the module do,
    design §4.2 / T4): router ``cfg.router_decode_pc(m_tiles=2)`` (+ the fused-sigmoid variant), a 64-row -inf top-k pad
    (allocated here, before any capture: F3 rule R3), gate_up ``cfg.experts_gate_up_pc(m_tiles=2)`` (10 x 8) and down
    ``cfg.experts_down_pc(m_tiles=2)`` (8 x 8). Outside :meth:`active` the module is untouched (its M = 32 path)."""

    def __init__(self, moe, cfg):
        self.moe, self.r = moe, moe.router
        self.pc64 = cfg.router_decode_pc(m_tiles=2)
        self.pc64s = cfg.router_decode_pc(sigmoid=True, m_tiles=2)
        self.gu64, self.dn64 = cfg.experts_gate_up_pc(m_tiles=2), cfg.experts_down_pc(m_tiles=2)
        self.pad64 = self.r._make_pad(64)

    @contextlib.contextmanager
    def active(self):
        r, moe = self.r, self.moe
        saved = (r._pc, r._decode_pc_base, r.decode_pc_sigmoid, r._pads.get(64), moe.experts)
        orig_pc, orig_experts = r._pc, moe.experts
        pc64, gu64, dn64 = self.pc64, self.gu64, self.dn64

        def pc_by_rows(f):
            return pc64 if int(f.shape[-2]) == 64 else orig_pc(f)

        def experts(m, f, *, polynorm, decode, row_scale=None, memory_config=None):  # MotifMoE.experts at M = 64
            if not (decode and int(f.shape[-2]) == 64):
                return orig_experts(f, polynorm=polynorm, decode=decode, row_scale=row_scale,
                                    memory_config=memory_config)  # fmt: skip
            mc = memory_config or m.dram
            x12 = ttnn.repeat(f, ttnn.Shape([1, m.e_loc, 1, 1]), memory_config=mc)
            gu_dtype = m.gate_up_dtype or (ttnn.float32 if polynorm == "fp32" else ttnn.bfloat16)
            g_u = ttnn.matmul(x12, m.w_gate_up, program_config=gu64, compute_kernel_config=m.ckc_experts,
                              dtype=gu_dtype, memory_config=mc)  # fmt: skip
            if x12 is not f:
                gu.free(x12)
            h = m.polynorm(g_u, mode=polynorm, row_scale=row_scale, memory_config=mc, impl=m.polynorm_impl)
            gu.free(g_u)
            y = ttnn.matmul(h, m.w_down, program_config=dn64, compute_kernel_config=m.ckc_experts, dtype=m.down_dtype,
                            memory_config=mc)  # fmt: skip
            gu.free(h)
            return y

        r._pc, r._decode_pc_base, r.decode_pc_sigmoid = pc_by_rows, pc64, self.pc64s
        r._pads[64] = self.pad64
        moe.experts = types.MethodType(experts, moe)
        try:
            yield self
        finally:
            r._pc, r._decode_pc_base, r.decode_pc_sigmoid = saved[0], saved[1], saved[2]
            if saved[3] is None:
                r._pads.pop(64, None)
            moe.experts = saved[4]

    def free(self):
        gu.free(self.pad64)


class HeadL64:
    """The 64-row argmax constants of ``MotifLMHead.argmax_decode`` (``L`` = 64; allocated here, before any capture)
    swapped in by :meth:`active` (what D1's L-keyed constants will do)."""

    def __init__(self, head, cfg, mesh_device):
        from models.demos.motif3.tt.lm_head import TILE, argmax_offsets

        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        kw = dict(layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
        self.head = head
        self.c = {
            "argmax_lanes": 64,
            "_am_zeros": ttnn.from_torch(torch.zeros(1, 1, 64, TILE), dtype=head.logits_dtype, **kw),
            "_am_zeros_i32": ttnn.from_torch(torch.zeros(1, 1, 64, TILE, dtype=torch.int32), dtype=ttnn.int32, **kw),
            "_am_iota": ttnn.from_torch(torch.arange(head.vc, dtype=torch.int32).reshape(1, 1, 1, -1).expand(
                1, 1, 64, -1).contiguous(), dtype=ttnn.int32, **kw),  # fmt: skip
            "_am_offsets": ttnn.from_torch(argmax_offsets(cfg, "mesh", 64), dtype=ttnn.int32, **kw),
        }

    @contextlib.contextmanager
    def active(self):
        saved = {k: getattr(self.head, k) for k in self.c}
        for k, v in self.c.items():
            setattr(self.head, k, v)
        try:
            yield self
        finally:
            for k, v in saved.items():
                setattr(self.head, k, v)

    def free(self):
        gu.free([v for k, v in self.c.items() if k != "argmax_lanes"])


def split_gather(ccl, x, W: int):
    """Split-order DP gather of a per-row ``[1, 1, 16, W]`` TILE tensor -> ``[1, 1, 64, W]`` (rows 0..31 = every row's
    first 8, lane order; 32..63 = their last 8): what D1's ``ccl.ag_dp_rows(x, halves=2)`` will do."""
    rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.reshape(rm, (1, 2, LPR, W))
    g = ccl.all_gather(v, 2, "dp", memory_config=ttnn.L1_MEMORY_CONFIG)
    gu.free(rm)
    out = ttnn.to_layout(ttnn.reshape(g, (1, 1, 2 * LANES, W)), ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gu.free(g)
    return out


def _host_argmax(lg, cfg, mesh_device) -> torch.Tensor:
    """Lowest index of the row maximum over the full vocab, assembled from every chip's vocab block."""
    from models.demos.motif3.tt.lm_head import vocab_block_of_coord

    t = all_chips(lg, mesh_device)  # [R, C, 1, 1, L, Vc]
    R, C = t.shape[0], t.shape[1]
    L, Vc = t.shape[-2], t.shape[-1]
    full = torch.empty(L, R * C * Vc)
    for r in range(R):
        for c in range(C):
            b = vocab_block_of_coord(cfg, r, c, "mesh")
            full[:, b * Vc : (b + 1) * Vc] = t[r, c, 0, 0]
    return torch.argmax(full, dim=-1)


# Retired 2026-10-04 (lead decision, docs/FINAL_VALIDATION.md §6.2). The in-place G16-lite probes inject the B0 + C1a
# builders, and the shipped modules refuse them: MotifMoE's F3N rule R3 guard (no 64-row constants without a config
# that stages T64) and the global layer's flash_groups guard (option A''). The T64 path is covered by the real-module
# T64 tests (test_moe.py, test_attention_*, test_kv_write.py) and by the model-level G16 (test_t64_g16_step_cost) and
# G-S5w in test_spec_decode_device.py, which pass on TORUS_XY. Their numbers are in GATES_RESULTS.md §13 (history).
G16LITE_RETIRED = (
    "retired 2026-10-04: the shipped MotifMoE / global layer refuse the injected B0 + C1a builders; covered by the "
    "real-module T64 tests and the model-level G16 / G-S5w (docs/FINAL_VALIDATION.md §6.2)"
)


@pytest.mark.skip(reason=G16LITE_RETIRED)
@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g16lite_ops(mesh_device):
    torch.set_num_threads(16)
    cfg, ccl = _setup(mesh_device, "g16lite_ops")
    failures, timers, cleanup = [], [], []
    for part in (_lite_gathers, _lite_head, _lite_moe, _lite_kv_write):
        try:
            ok, why, t, c = part(mesh_device, cfg, ccl)
            timers += t
            cleanup += c
            if not ok:
                failures.append(f"{part.__name__}: {why}")
        except Exception as e:
            REC.add(f"lite/{part.__name__}", status="error", error=_err(e))
            failures.append(f"{part.__name__}: {_err(e)}")
    # ---- traced costs, after every program was compiled by the parts above (no compile after a capture) -------------
    for name, fn, ctx, n in timers:
        try:
            with ctx() if ctx is not None else contextlib.nullcontext():
                us, raw = gu.time_traced(mesh_device, fn, ops_per_trace=n, reps=7, compile_first=True)
            REC.add(f"lite/cost/{name}", status="info", traced_us=us, traced_raw_us=raw)
        except Exception as e:
            REC.add(f"lite/cost/{name}", status="error", error=_err(e))
            failures.append(f"cost {name}: {_err(e)}")
    for c in cleanup:
        c()
    assert not failures, "G16-lite ops failures:\n" + "\n".join(failures)


def _lite_gathers(mesh_device, cfg, ccl):
    ok_all, why, timers, keep = True, [], [], []
    for Wd in (4096, 576):
        g = torch.Generator().manual_seed(Wd)
        x8h = torch.randn(ROWS, 1, LPR, Wd, generator=g).bfloat16()
        x16h = torch.randn(ROWS, 1, WR, Wd, generator=g).bfloat16()
        x8, x16 = rows_dev(x8h, cfg, mesh_device), rows_dev(x16h, cfg, mesh_device)
        keep += [x8, x16]
        g8, g16 = ccl.ag_dp_rows(x8), ccl.ag_dp_rows(x16)
        gs, cb = run_pinned(mesh_device, lambda: split_gather(ccl, x16, Wd))
        if gs is None:
            gs = split_gather(ccl, x16, Wd)
        want8 = x8h[:, 0].reshape(LANES, Wd).float()
        want16 = x16h[:, 0].reshape(WROWS, Wd).float()
        res = {
            "ag_dp_rows_8_lane_order": torch.equal(gu.read_dev(g8, 0)[0, 0], want8),
            "ag_dp_rows_16_natural_order": torch.equal(gu.read_dev(g16, 0)[0, 0], want16),
            "split_order": torch.equal(gu.read_dev(gs, 0)[0, 0], split_order(want16)),
            "split_replicas_32": gu.replicas_identical(gs)[0],
        }
        ok = all(res.values()) and cb_ok(cb)
        ok_all &= ok
        REC.add(f"lite/gather/W{Wd}", status="pass" if ok else "fail", **res, **cb)
        if not ok:
            why.append(f"W{Wd}: {res} {cb}")
        gu.free([g8, g16, gs])
        timers += [(f"ag_dp_rows_L8_W{Wd}", lambda x=x8: ccl.ag_dp_rows(x), None, 32),
                   (f"ag_dp_rows_L16_W{Wd}", lambda x=x16: ccl.ag_dp_rows(x), None, 32),
                   (f"split_gather_L16_W{Wd}", lambda x=x16, w=Wd: split_gather(ccl, x, w), None, 32)]  # fmt: skip
    return ok_all, "; ".join(why), timers, [lambda: gu.free(keep)]


def _lite_head(mesh_device, cfg, ccl):
    from models.demos.motif3.tt.lm_head import MotifLMHead
    from models.demos.motif3.tt.model import LazySource

    t0 = time.time()
    head = MotifLMHead(mesh_device, cfg, source=LazySource(cfg.weights_dir), ccl=ccl, cache=True)
    print(f"[{GATE}] LM head loaded from the TT cache in {time.time() - t0:.1f} s")
    g = torch.Generator().manual_seed(3)
    h32 = (torch.randn(1, 1, LANES, cfg.hidden_size, generator=g) * 2.0).bfloat16()
    h64 = torch.cat([h32, (torch.randn(1, 1, LANES, cfg.hidden_size, generator=g) * 2.0).bfloat16()], dim=2)
    g32, g64 = up(h32, mesh_device), up(h64, mesh_device)
    pc64 = cfg.lm_head_pc("mesh", m_tiles=2)

    def proj64():
        return ttnn.linear(g64, head.weight, program_config=pc64, compute_kernel_config=head.ckc_lm,
                           dtype=head.logits_dtype, memory_config=head.memory_config)  # fmt: skip

    lg32 = head.project(g32)
    lg64, cb = run_pinned(mesh_device, proj64)
    if lg64 is None:
        lg64 = proj64()
    l32, l64 = all_chips(lg32, mesh_device), all_chips(lg64, mesh_device)
    rows_bit = torch.equal(l64[..., :LANES, :], l32)
    max_abs = float((l64[..., :LANES, :] - l32).abs().max())
    hl = HeadL64(head, cfg, mesh_device)
    a32 = gu.read_dev(head.argmax_decode(lg32), 0).reshape(-1).long()
    with hl.active():
        am64, cb_am = run_pinned(mesh_device, lambda: head.argmax_decode(lg64))
        if am64 is None:
            am64 = head.argmax_decode(lg64)
    a64 = gu.read_dev(am64, 0).reshape(-1).long()
    host64 = _host_argmax(lg64, cfg, mesh_device)
    res = {
        "gemm_rows_0_31_bitwise_all_chips": rows_bit,
        "argmax64_eq_host": torch.equal(a64, host64),
        "argmax64_rows_0_31_eq_argmax32": torch.equal(a64[:LANES], a32),
        "argmax32_eq_host": torch.equal(a32, _host_argmax(lg32, cfg, mesh_device)),
    }
    nonfinite = int((~torch.isfinite(l64)).sum())
    ok = all(res.values()) and nonfinite == 0 and cb_ok(cb) and cb_ok(cb_am)
    REC.add("lite/lm_head_m64", status="pass" if ok else "fail", **res, gemm_max_abs_vs_m32=max_abs,
            nonfinite=nonfinite, pc64=str(pc64)[:200], cb_gemm=cb["cb_check"], cb_argmax=cb_am["cb_check"])
    gu.free(am64)
    timers = [("lm_head_gemm_M32", lambda: head.project(g32), None, 32),
              ("lm_head_gemm_M64_pcM2", proj64, None, 32),
              ("argmax_32", lambda: head.argmax_decode(lg32), None, 16),
              ("argmax_64", lambda: head.argmax_decode(lg64), hl.active, 16)]  # fmt: skip

    def cleanup():
        gu.free([g32, g64, lg32, lg64])
        hl.free()
        head.close()

    return ok, f"{res} nonfinite {nonfinite} cb {cb} {cb_am}", timers, [cleanup]


def _lite_moe(mesh_device, cfg, ccl):
    from models.demos.motif3.tt.model import LazySource
    from models.demos.motif3.tt.moe import MotifMoE

    t0 = time.time()
    moe = MotifMoE(mesh_device, cfg, 2, source=LazySource(cfg.weights_dir), ccl=ccl, cache=True,
                   router_logits=cfg.router_logits)  # fmt: skip
    print(f"[{GATE}] MoE L2 loaded from the TT cache in {time.time() - t0:.1f} s")
    p64 = MoEM64(moe, cfg)
    g = torch.Generator().manual_seed(11)
    h8 = torch.randn(ROWS, 1, LPR, cfg.hidden_size, generator=g).bfloat16()
    h16 = torch.cat([h8, torch.randn(ROWS, 1, LPR, cfg.hidden_size, generator=g).bfloat16()], dim=2)
    x8, x16 = rows_dev(h8, cfg, mesh_device), rows_dev(h16, cfg, mesh_device)
    o8 = moe.forward_decode(x8)
    with p64.active():
        o16, cb = run_pinned(mesh_device, lambda: moe.forward_decode(x16))
        if o16 is None:
            o16 = moe.forward_decode(x16)
    t8, t16 = all_chips(o8, mesh_device), all_chips(o16, mesh_device)  # [R, C, 1, 1, rows, 4096]
    bit = torch.equal(t16[..., :LPR, :], t8)
    max_abs = float((t16[..., :LPR, :] - t8).abs().max())
    nonfinite = int((~torch.isfinite(t16)).sum())
    ok = bit and nonfinite == 0 and cb_ok(cb)
    REC.add("lite/moe_m64_L2", status="pass" if ok else "fail", rows_0_7_bitwise_eq_m32_all_chips=bit,
            max_abs_vs_m32=max_abs, nonfinite=nonfinite, router_logits=cfg.router_logits,
            configs={"router": str(p64.pc64)[:160], "gate_up": str(p64.gu64)[:160], "down": str(p64.dn64)[:160]},
            **cb)  # fmt: skip
    gu.free([o8, o16])
    timers = [("moe_L2_module_M32", lambda: moe.forward_decode(x8), None, 8),
              ("moe_L2_module_M64_c1a_configs", lambda: moe.forward_decode(x16), p64.active, 8)]  # fmt: skip

    def cleanup():
        gu.free([x8, x16])
        p64.free()
        moe.deallocate()

    return ok, f"bitwise {bit} max_abs {max_abs} nonfinite {nonfinite} cb {cb}", timers, [cleanup]


def _lite_kv_write(mesh_device, cfg, ccl):
    """Per-layer KV write cost (G16-lite, t64.md §2.4): the production ``DecodeKVWrite`` at 8 rows per DP row
    (``all_split``, ``row_split``) vs the T64 writers at 16 (split / natural / row_split), bfp8 and bf16 caches (bf16:
    16 users per call, R-E8). Ordinary T32 steps (call B all -1) and full T64 verify steps (every lane drafting)."""
    from models.demos.motif3.tt.kv_write import DecodeKVWrite, KVWriteStep
    from models.demos.motif3.tt.model_config import MotifTTConfig

    W, N = 64, 520
    pt = torch.zeros(LANES, W, dtype=torch.int32)
    for l in range(LANES):
        pt[l, :8] = torch.arange(1 + 8 * l, 9 + 8 * l, dtype=torch.int32)
    pos = torch.full((LANES,), 300, dtype=torch.int32)
    step = WideStep(pos, pos + 1, pt)
    g = torch.Generator().manual_seed(21)
    row8 = rows_dev(torch.randn(ROWS, 1, LPR, D, generator=g).bfloat16(), cfg, mesh_device)
    row16 = rows_dev(torch.randn(ROWS, 1, WR, D, generator=g).bfloat16(), cfg, mesh_device)
    timers, keep, mods = [], [row8, row16], []
    for dname, dtype in (("bfp8", ttnn.bfloat8_b), ("bf16", ttnn.bfloat16)):
        c = cfg if dname == "bfp8" else MotifTTConfig.from_hf_config(None, mesh_device=mesh_device,
                                                                      kv_cache_dtype="bf16")
        e = ttnn.empty([N, 1, BS, D], dtype, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        cache = ttnn.fill(e, 0.0)
        gu.free(e)
        keep.append(cache)
        n = 32 if dname == "bfp8" else 16
        t32 = {m: DecodeKVWrite(mesh_device, c, ccl=ccl, page_table_width=W, mode=m)
               for m in ("all_split", "row_split")}
        for w in t32.values():
            w.write_step(KVWriteStep.ordinary(pos, pt))
        t64 = {"split": SplitWriter(mesh_device, ccl, step, n), "natural": NaturalWriter(mesh_device, ccl, step, n),
               "row_split": RowSplitWriter(mesh_device, c, step)}  # fmt: skip
        mods += list(t32.values()) + list(t64.values())
        for m, w in t32.items():
            fn = (lambda w=w, cache=cache: w.write(row8, cache, cur_pos=w.cur_pos, page_table=w.page_table))
            fn()
            timers.append((f"kv_write_{dname}_T32_{m}", fn, None, 32))
        for m, w in t64.items():
            fn = (lambda w=w, cache=cache: w.write(row16, cache))
            fn()
            timers.append((f"kv_write_{dname}_T64_{m}", fn, None, 32))
    REC.add("lite/kv_write_cost_setup", status="info", note="timed below (lite/cost/kv_write_*): per layer, W 64, "
            "caches [520, 1, 64, 576]; T64 every lane drafting (anchors at 300, drafts at 301)")  # fmt: skip

    def cleanup():
        for w in mods:
            (w.deallocate if hasattr(w, "deallocate") else w.free)()
        gu.free(keep)

    return True, "", timers, [cleanup]


# ======================================================================================================================
# (d) G16-lite whole decoder layers: T32 vs T64 (A / A')
# ======================================================================================================================
@contextlib.contextmanager
def flash_two_calls(pt8_rows, cur_a_rows, cur_d_rows, *, only_global: bool):
    """Option A' inside the block: every FlashMLA decode call on a 16-row query (``[1, 16, H, D]``) runs as two B = 8
    calls (rows 0..7 at the anchors' ``cur_pos``, 8..15 at the drafts'), same per-row page table, outputs concatenated
    on dim 1. ``only_global``: SWA calls (``sliding_window_size`` set) keep the one B = 16 call (= A'')."""
    orig = ttnn.transformer.paged_flash_multi_latent_attention_decode

    def wrapped(q, cache, v=None, **kw):
        if int(q.shape[1]) != WR or (only_global and kw.get("sliding_window_size") is not None):
            return orig(q, cache, v, **kw)
        kw = dict(kw)
        kw.pop("page_table_tensor", None)
        kw.pop("cur_pos_tensor", None)
        H, Dq = int(q.shape[2]), int(q.shape[3])
        qa = ttnn.slice(q, [0, 0, 0, 0], [1, LPR, H, Dq])
        qd = ttnn.slice(q, [0, LPR, 0, 0], [1, WR, H, Dq])
        oa = orig(qa, cache, v, page_table_tensor=pt8_rows, cur_pos_tensor=cur_a_rows, **kw)
        od = orig(qd, cache, v, page_table_tensor=pt8_rows, cur_pos_tensor=cur_d_rows, **kw)
        out = ttnn.concat([oa, od], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gu.free([qa, qd, oa, od])
        return out

    ttnn.transformer.paged_flash_multi_latent_attention_decode = wrapped
    try:
        yield
    finally:
        ttnn.transformer.paged_flash_multi_latent_attention_decode = orig


class LayerInputs:
    """T32 / T64 inputs of one decoder layer at context ``ctx`` (the G16-lite probe's setup): 32 lanes, each with its
    own blocks; anchors at ``ctx - 1 - (l % 5)``; T64 drafts at + 1 on the owner's row; random streams."""

    def __init__(self, mesh_device, cfg, ccl, rope, layer_idx: int, ctx: int, random_cache: bool):
        from models.demos.motif3.tt.attention import MotifAttention
        from models.demos.motif3.tt.kv_write import DecodeKVWrite, KVWriteStep

        self.mesh = mesh_device
        Wp = W_FL
        nb = (ctx + 2 + BS) // BS + 1
        N = LANES * nb + 1
        if random_cache:
            g = torch.Generator().manual_seed(7 + layer_idx)
            self.cache = up((torch.randn(N, 1, BS, D, generator=g) * 0.5).bfloat16(), mesh_device, ttnn.bfloat8_b)
        else:
            e = ttnn.empty([N, 1, BS, D], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
            self.cache = ttnn.fill(e, 0.0)
            gu.free(e)
        pt = torch.zeros(LANES, Wp, dtype=torch.int32)
        for l in range(LANES):
            pt[l, :nb] = torch.arange(1 + nb * l, 1 + nb * (l + 1), dtype=torch.int32)
        pos = torch.tensor([ctx - 1 - (l % 5) for l in range(LANES)], dtype=torch.int32)
        g = torch.Generator().manual_seed(100 + layer_idx)
        xa = torch.randn(ROWS, 4, LPR, cfg.hidden_size, generator=g)  # anchors' streams (lane order per row)
        xd = torch.randn(ROWS, 4, LPR, cfg.hidden_size, generator=g)  # drafts' streams

        def rot_of(pos_rows):  # [4, R] -> rot tables (R used rows of the [1, 32] index row)
            idx = torch.zeros(ROWS, 32, dtype=torch.int32)
            idx[:, : pos_rows.shape[1]] = pos_rows.clamp_min(0)
            ri = rows_dev(idx, cfg, mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            return ri, MotifAttention.decode_rope_tables(rope, ri, kinds=("plain", "yarn"))

        def xr(x):
            return rows_dev(x.bfloat16(), cfg, mesh_device)

        # T32: the production DecodeKVWrite (all_split) at 8 rows per DP row; anchors at p, and the draft reference
        self.kvw = DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=Wp, mode="all_split")
        self.kvw.write_step(KVWriteStep.ordinary(pos, pt))
        self.ri8, self.rot8 = rot_of(pos.reshape(ROWS, LPR))
        self.act8 = MotifAttention.active_mask_from_cur_pos(self.kvw.cur_pos, LPR)
        self.kvw_d = DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=Wp, mode="all_split")
        self.kvw_d.write_step(KVWriteStep.ordinary(pos + 1, pt))
        self.ri8d, self.rot8d = rot_of((pos + 1).reshape(ROWS, LPR))
        self.act8d = MotifAttention.active_mask_from_cur_pos(self.kvw_d.cur_pos, LPR)
        # T64: per DP row [8 anchors at p | 8 drafts at p + 1], the drafts on the owner's page-table row
        step = WideStep(pos, pos + 1, pt)
        pos16, pt16 = step.phys_positions(), step.phys_page_table()
        self.cur16 = rows_dev(pos16, cfg, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.pt16 = rows_dev(pt16, cfg, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.ri16, self.rot16 = rot_of(pos16.reshape(ROWS, WR))
        self.act16 = MotifAttention.active_mask_from_cur_pos(self.cur16, WR)
        self.writer = SplitWriter(mesh_device, ccl, step, 32)
        # option A' halves per DP row: [8, W] page table, anchors' / drafts' cur_pos [8]
        self.pt8r = rows_dev(pt, cfg, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.ca8r = rows_dev(pos, cfg, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.cd8r = rows_dev(pos + 1, cfg, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.X8, self.X8d, self.X16 = xr(xa), xr(xd), xr(torch.cat([xa, xd], dim=2))

    def t32(self, layer, draft: bool = False):
        kv = self.kvw_d if draft else self.kvw
        return layer.forward_decode(self.X8d if draft else self.X8, rot=self.rot8d if draft else self.rot8,
                                    cur_pos=kv.cur_pos, page_table=kv.page_table, kv_cache=self.cache,
                                    active=self.act8d if draft else self.act8, kv_write=kv)  # fmt: skip

    def t64(self, layer):
        return layer.forward_decode(self.X16, rot=self.rot16, cur_pos=self.cur16, page_table=self.pt16,
                                    kv_cache=self.cache, active=self.act16, kv_write=self.writer)  # fmt: skip

    def a_prime(self, only_global: bool):
        return flash_two_calls(self.pt8r, self.ca8r, self.cd8r, only_global=only_global)

    def free(self):
        self.kvw.deallocate()
        self.kvw_d.deallocate()
        self.writer.free()
        gu.free([self.cache, self.act8, self.act8d, self.act16, self.cur16, self.pt16, self.pt8r, self.ca8r,
                 self.cd8r, self.X8, self.X8d, self.X16, self.ri8, self.ri8d, self.ri16]
                + [t for r in (self.rot8, self.rot8d, self.rot16) for cs in r.values() for t in cs])  # fmt: skip


@pytest.mark.skip(reason=G16LITE_RETIRED)
@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g16lite_layers(mesh_device):
    from models.demos.motif3.tt.decoder import MotifDecoderLayer, free_tensors
    from models.demos.motif3.tt.model import LazySource
    from models.demos.motif3.tt.rope import MotifRope

    torch.set_num_threads(16)
    cfg, ccl = _setup(mesh_device, "g16lite_layers")
    failures = []
    rope = MotifRope(mesh_device, cfg)
    layers, patches, inputs = {}, {}, {}
    for li in LAYERS:
        t0 = time.time()
        layers[li] = MotifDecoderLayer(mesh_device, cfg, li, source=LazySource(cfg.weights_dir), ccl=ccl, rope=rope,
                                       cache=True)  # fmt: skip
        patches[li] = MoEM64(layers[li].moe, cfg) if layers[li].is_moe else None
        print(f"[{GATE}] layer {li} ({cfg.layer(li).kind}) loaded in {time.time() - t0:.1f} s")
    variants = (("A_b16", None), ("A_prime_2xb8", False), ("A_dprime", True))  # (name, only_global for the wrapper)

    def t64_ctx(li, d, only_global):
        stack = contextlib.ExitStack()
        if patches[li] is not None:
            stack.enter_context(patches[li].active())
        if only_global is not None:
            stack.enter_context(d.a_prime(only_global))
        return stack

    # ---- phase 1 (eager): bitwise rows at ctx 1000 (random cache) + compile the 4K-context timing shapes -------------
    for li in LAYERS:
        layer = layers[li]
        kind = "global" if cfg.layer(li).attn_kind == "global" else "swa"
        try:
            d = LayerInputs(mesh_device, cfg, ccl, rope, li, CTX_BITS, random_cache=True)
            o_a, o_d = d.t32(layer), d.t32(layer, draft=True)  # anchors at p, then the drafts at p + 1 (sees p)
            ta, td = all_chips(o_a, mesh_device), all_chips(o_d, mesh_device)  # [R, C, 1, 4, 8, 4096]
            res, cbs = {}, {}
            for name, only_global in variants:
                with t64_ctx(li, d, only_global):
                    if name == "A_b16":
                        o, cbs[name] = run_pinned(mesh_device, lambda: d.t64(layer))
                        o = o if o is not None else d.t64(layer)
                    else:
                        o = d.t64(layer)
                t = all_chips(o, mesh_device)  # [R, C, 1, 4, 16, 4096]
                res[name] = {"anchors_bitwise": torch.equal(t[..., :LPR, :], ta),
                             "drafts_bitwise": torch.equal(t[..., LPR:, :], td),
                             "anchors_max_abs": float((t[..., :LPR, :] - ta).abs().max()),
                             "drafts_max_abs": float((t[..., LPR:, :] - td).abs().max()),
                             "finite": bool(torch.isfinite(t).all())}  # fmt: skip
                gu.free(o)
            gu.free([o_a, o_d])
            a2 = res["A_dprime"]
            ok = a2["anchors_bitwise"] and a2["drafts_bitwise"] and a2["finite"] and all(cb_ok(c) for c in cbs.values())
            REC.add(f"lite/layer/L{li}_{kind}/bitwise_ctx{CTX_BITS}", status="pass" if ok else "fail", **res,
                    ref_max_abs=float(ta.abs().max()), cb_t64=cbs.get("A_b16", {}).get("cb_check"),
                    note="A'' = A on SWA layers, A' on global layers (design T3)")  # fmt: skip
            if not ok:
                failures.append(f"L{li}: A'' rows not bitwise {a2} cb {cbs}")
            d.free()
            # compile at the timing context's shapes (another cache size): T32, T64 A, A', A''
            d = LayerInputs(mesh_device, cfg, ccl, rope, li, CTX_TIMING, random_cache=False)
            gu.free(d.t32(layer))
            for name, only_global in variants:
                with t64_ctx(li, d, only_global):
                    gu.free(d.t64(layer))
            inputs[li] = d
        except Exception as e:
            REC.add(f"lite/layer/L{li}_{kind}", status="error", error=_err(e))
            failures.append(f"L{li}: {_err(e)}")
    # ---- phase 2: traced layer costs at 4K context (no program compiled after a capture) --------------------------
    for li, d in inputs.items():
        layer = layers[li]
        kind = "global" if cfg.layer(li).attn_kind == "global" else "swa"
        try:
            costs = {}
            us, _ = gu.time_traced(mesh_device, lambda: d.t32(layer), ops_per_trace=8, reps=5, compile_first=False)
            costs["T32_all_split"] = round(us, 1)
            for name, only_global in variants:
                with t64_ctx(li, d, only_global):
                    us, _ = gu.time_traced(mesh_device, lambda: d.t64(layer), ops_per_trace=8, reps=5,
                                           compile_first=False)  # fmt: skip
                costs[f"T64_{name}"] = round(us, 1)
            REC.add(f"lite/layer/L{li}_{kind}/cost_ctx{CTX_TIMING}", status="info", traced_us=costs,
                    delta_us={k: round(v - costs["T32_all_split"], 1) for k, v in costs.items() if k.startswith("T64")},
                    ratio={k: round(v / costs["T32_all_split"], 4) for k, v in costs.items() if k.startswith("T64")})
        except Exception as e:
            REC.add(f"lite/layer/L{li}_{kind}/cost_ctx{CTX_TIMING}", status="error", error=_err(e))
            failures.append(f"L{li} cost: {_err(e)}")
    for d in inputs.values():
        d.free()
    for li, layer in layers.items():
        if patches[li] is not None:
            patches[li].free()
        layer.deallocate()
    free_tensors(rope)
    assert not failures, "G16-lite layer failures:\n" + "\n".join(failures)
