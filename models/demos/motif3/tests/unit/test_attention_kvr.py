# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Attention decode with the shared KV write of ``tt/kv_write.py`` (features design §3.5, §3.8.2, §5.2
``test_attention_kvr.py``; README §16): ``MotifAttention.forward_decode(..., kv_write=w)`` (the
``tt.attention.DecodeKVWriter`` hook) in every mode, one :class:`DecodeKVWrite` serving decoder layers 0 (global,
YaRN), 1 (SWA) and the MTP layer 53 (SWA, plain RoPE) in the same step.

Host (``-k cpu``; recording fake ``ttnn``): the decode op sequence with ``kv_write`` in ``row`` mode is draft 1's (the
same ops with the same operands; only ``kv_row`` is freed after the update instead of before it), and in every mode
only the KV-write ops between the latent ``concat`` and FlashMLA change; FlashMLA reads the kv_write's ``cur_pos`` /
``page_table``.

Device (``-k "not cpu"``; random weights, a random bfp8 cache as history, 32 lanes over 4 DP rows with owners at
``p % 64`` in {0, 30, 31, 62, 63}, plain and idle lanes):

* reference (draft 1, ``kv_write=None``): step A = every active lane at ``n``; step B = the drafted owners at
  ``n + 1`` on their own lanes, after A;
* ``row`` / ``all`` ordinary step: outputs bitwise equal to step A; caches: ``row`` bitwise draft 1 on every chip,
  ``all`` (KV-R) holds every lane's step-A row on all 32 chips, bitwise;
* ``row_split`` / ``all_split`` packed verify (owners at ``n``, drafts at ``n + 1`` on idle partner lanes: same-row
  partners, and cross-row partners under KV-R), then the overflow pass for drafts without an idle lane: owner and
  plain lanes bitwise equal to step A, overflow lanes bitwise equal to step B, partner lanes vs step B (PCC >= 0.9999;
  bitwise equality reported: the lane-relocation probe of review R5 at attention level); every chip's cache changed
  at exactly the expected slots (KV-R: all lanes on all 32 chips, identical copies), owner / plain / overflow rows
  bitwise equal to the reference's;
* ``all_split`` traced: one capture of the 3-layer step, replayed for an ordinary and a verify step: outputs and caches
  bitwise equal to the eager runs, no program compiled after the warmup.

T64 verify step, 16 rows per DP row ``[8 anchors | 8 drafts]`` with FlashMLA option A''
(``docs/p5_t64/P5_T64_DESIGN.md`` §4.1, §4.3, §7.2 "T64, attention"; work package A2):

* host (``-k cpu``; the recording fake ``ttnn``): at 16 rows every op outside the KV write and FlashMLA is the T32 op
  (same kwargs and configs, 16 rows instead of 8); the global layer runs one B = 8 FlashMLA call per entry of
  ``kv_write.flash_groups()`` on a dim-1 slice of ``q_mla`` (anchors on ``flash_cur_a``, drafts on ``flash_cur_d``,
  both on ``flash_pt``), then one dim-1 concat; the SWA layers and the MTP layer make one B = 16 call on ``cur_pos`` /
  ``page_table``; nothing leaks. The guards: ``kv_write=None`` at 16 rows, a writer whose ``lanes_per_row`` differs
  from ``x``, a global layer at 16 rows without ``flash_groups()``, and groups that do not partition the rows are
  refused before any op. Option A (one group over the 16 rows) is one B = 16 call. ``rope``'s ``rows_per_dp`` host
  helpers and ``active_mask_host(rows_per_dp=16)``.
* device (``-k wide``; random weights, layers 0 (global), 1 (SWA) and the MTP layer; anchors at ~1K / ~4K / ~32K
  with ``n % 64`` in {0, 31, 40, 63}, idle and undrafted lanes; bfp8 ``all_split`` (split gather) and ``row_split``,
  bf16 ``all_split``; an L1 pin page alive, so a new program whose static CBs reach it fails): the T64 step with the
  real ``DecodeKVWrite(rows=64)`` vs two T32 steps of the same mode (the anchors at ``n``, then the drafts at ``n +
  1``): every anchor row and every draft row bitwise equal to its T32 row on all three layers, replicas identical
  over TP, every chip's cache bitwise equal to the T32 caches. Negative control: option A (one B = 16 call) is not
  bitwise on the global layer (it is the same call on the SWA layers). bfp8 ``all_split`` traced: one capture of the
  T64 3-layer step, replayed for a verify and an ordinary T64 step: outputs and caches bitwise equal to eager, no
  program compiled after the warmup.

Run::

    S=/home/ttuser/hchang/experiments/motif-3/scripts
    $S/hostrun.sh -n attention_kvr_host -- python -m pytest -p no:cacheprovider -q \
        models/demos/motif3/tests/unit/test_attention_kvr.py -k cpu
    $S/devrun.sh -t 1500 -n attention_kvr -- python -m pytest models/demos/motif3/tests/unit/test_attention_kvr.py \
        -k "not cpu" -s -p no:cacheprovider

Every device check prints an ``[attn-kvr]`` line.
"""

from __future__ import annotations

import inspect
from collections import Counter
from types import SimpleNamespace
from typing import List

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.test_attention import (
    _lane_x,
    _per_lane_out,
    _rows,
    _setup,
    hf_source,
    pcc,
    random_attn_tensors,
    ref_args,
)
from models.demos.motif3.tests.unit.test_attention_resumed import _FakeTTNN, _FT, _fake_attention, source_for
from models.demos.motif3.tests.unit.test_kv_write import (
    BS,
    D,
    HF_META,
    LPR,
    WB,
    WLPR,
    fake_kv_write,
    flash_layout,
    host_cfg,
    l1_pin,
    lane_layout,
    quant,
    same_tile,
    t64_ccl,
    upload_cache,
)
from models.demos.motif3.tt import attention as A
from models.demos.motif3.tt import kv_write as KW
from models.demos.motif3.tt import rope as RO
from models.demos.motif3.tt.attention import MTP_ATTN_PREFIX, MotifAttention
from models.demos.motif3.tt.model_config import MotifTTConfig, device_params
from models.demos.motif3.tt.rope import shard_lanes

B = 32
MESH = [pytest.param((4, 8), device_params(), id="4x8")]
HOOK = "kv_write" in inspect.signature(MotifAttention.forward_decode).parameters
needs_hook = pytest.mark.skipif(not HOOK, reason="MotifAttention.forward_decode has no kv_write= hook (WP2b)")
# T64 rows of each DP row r: anchors 16 r + j, drafts 16 r + 8 + j (lane 8 r + j), in lane order
A_ROWS = [WLPR * r + j for r in range(B // LPR) for j in range(LPR)]
D_ROWS = [a + LPR for a in A_ROWS]


def log(msg: str) -> None:
    print(f"[attn-kvr] {msg}", flush=True)


# ======================================================================================================================
# host: op sequences with a recording fake ttnn
# ======================================================================================================================
WRITE_OPS = {
    None: ["transpose", "deallocate", "paged_update_cache", "deallocate"],  # draft 1
    "row": ["transpose", "paged_update_cache", "deallocate", "deallocate"],
    "row_split": ["transpose", "paged_update_cache", "paged_update_cache", "deallocate", "deallocate"],
    "all": ["ag_dp_rows", "transpose", "deallocate", "paged_update_cache", "deallocate", "deallocate"],
    "all_split": ["ag_dp_rows", "transpose", "deallocate", "paged_update_cache", "paged_update_cache", "deallocate",
                  "deallocate"],  # fmt: skip
}


@needs_hook
def test_cpu_kv_write_hook_op_sequences(monkeypatch):
    cfg = host_cfg()
    for layer in (0, 1):
        spec = cfg.layer(layer)
        runs = {}
        for mode in WRITE_OPS:
            kvw = None
            if mode is not None:
                _, kvw = fake_kv_write(monkeypatch, mode)  # built with test_kv_write's fake, rewired below
            fake = _FakeTTNN()
            monkeypatch.setattr(A, "ttnn", fake)
            monkeypatch.setattr(KW, "ttnn", fake)
            attn = _fake_attention(A, cfg, spec, fake)
            cur, pt = _FT((8,), "i32", {"cur"}), _FT((8, 5), "i32", {"pt"})
            if kvw is not None:
                kvw._dev = {k: cur if k == "cur_pos" else pt if k == "page_table" else _FT((32,), "i32", {k})
                            for k in kvw._dev}  # fmt: skip
                kvw._bind_inputs()
                kvw.row_mc, kvw.call_mc = "mc:update", "mc:all"  # the fake attention's update_mc sentinel
                kvw.ccl = SimpleNamespace(
                    ag_dp_rows=lambda t, fake=fake: fake._out("ag_dp_rows", (t,), {}, (1, 1, 32, t.shape[-1]))
                )
            rot = {k: (_FT((1, 1, 32, 64), "bf16", {f"cos_{k}"}), _FT((1, 1, 32, 64), "bf16", {f"sin_{k}"}))
                   for k in ("yarn", "plain")}  # fmt: skip
            attn.forward_decode(
                _FT((1, 1, 8, 4096), "bf16", {"x"}), rot=rot, cur_pos=cur, page_table=pt,
                kv_cache=_FT((20, 1, 64, 576), "bfp8", {"cache"}), active=_FT((1, 1, 8, 1024), "bf16", {"act"}),
                kv_write=kvw,
            )  # fmt: skip
            runs[mode] = list(fake.calls)
        d1 = runs[None]

        def cut(calls):
            i_fm = next(i for i, c in enumerate(calls) if c[0] == "paged_flash_multi_latent_attention_decode")
            i_kv = max(i for i, c in enumerate(calls[:i_fm]) if c[0] == "concat") + 3  # concat, free n, free k_pe
            return calls[:i_kv], calls[i_kv:i_fm], calls[i_fm:]

        pre1, w1, post1 = cut(d1)
        assert [c[0] for c in w1] == WRITE_OPS[None]
        for mode in ("row", "row_split", "all", "all_split"):
            pre, w, post = cut(runs[mode])
            assert pre == pre1 and post == post1, mode  # q path, FlashMLA (cur / pt operands), epilogue: unchanged
            assert [c[0] for c in w] == WRITE_OPS[mode], (mode, [c[0] for c in w])
            ups = [c for c in w if c[0] == "paged_update_cache"]
            srcs = [dict(c[2])["update_idxs_tensor"][3] for c in ups]
            want = {"row": [("cur",)], "row_split": [("cur_a",), ("cur_b",)], "all": [("cur0",)],
                    "all_split": [("cur_a0",), ("cur_b0",)]}[mode]  # fmt: skip
            assert srcs == want, (mode, srcs)
        # row mode == draft 1: identical ops and operands, the same tensors freed (kv_row later: after the update)
        nd = lambda calls: [c for c in calls if c[0] != "deallocate"]  # noqa: E731
        assert nd(runs["row"]) == nd(d1)
        assert Counter(c for c in runs["row"] if c[0] == "deallocate") == Counter(c for c in d1 if c[0] == "deallocate")
        log(
            f"L{layer}: kv_write='row' issues draft 1's {len(nd(d1))} ops (operands identical, same frees); every mode "
            f"changes only the KV-write ops; FlashMLA reads kv_write.cur_pos / page_table"
        )


# ---- T64: 16 rows per DP row, FlashMLA option A'' (host) -------------------------------------------------------------
FM = "paged_flash_multi_latent_attention_decode"
A2_BLOCK = ["slice", FM, "deallocate", "slice", FM, "deallocate", "concat", "deallocate", "deallocate"]


def _fake_writer(monkeypatch, fake, mode: str, *, rows=None, gather=None, W: int = 4):
    """A :class:`DecodeKVWrite` (T32, or T64 with ``rows=64``) built on test_kv_write's fake, then rewired to the
    attention's recording fake ``ttnn``: every persistent input becomes an ``_FT`` named after it, with its per-chip
    shape (``cur_pos`` ``[16]``, ``flash_pt`` ``[8, W]``, ...)."""
    kw = {} if rows is None else {"rows": rows, "gather": gather}
    _, kvw = fake_kv_write(monkeypatch, mode, W=W, **kw)
    monkeypatch.setattr(KW, "ttnn", fake)
    dp = kvw.cfg.dp
    vals = kvw._values(KW.KVWriteStep.inactive(W, lanes=kvw.lanes))
    kvw._dev = {
        k: _FT((v.shape[0] // dp, *v.shape[1:]) if place == "row" else tuple(v.shape), "i32", {k})
        for k, (place, v) in vals.items()
    }
    kvw._bind_inputs()
    kvw.row_mc, kvw.call_mc = f"mc:update{kvw.lanes_per_row}", "mc:all"

    def ag_dp_rows(t, halves=1):
        return fake._out("ag_dp_rows", (t,), {"halves": halves}, (1, 1, dp * t.shape[-2], t.shape[-1]))

    kvw.ccl = SimpleNamespace(ag_dp_rows=ag_dp_rows)
    return kvw


def _decode_calls(attn, fake, rows: int, kvw, W: int = 4, active_rows=None):
    """``attn.forward_decode`` on ``rows`` rows per DP row with ``kvw`` (``None`` = draft 1): ``(recorded calls,
    output)``. FlashMLA's ``cur_pos`` / ``page_table`` are the writer's; the ``active`` mask has ``active_rows`` rows
    (default ``rows``)."""
    fake.calls.clear()
    cur = kvw.cur_pos if kvw is not None else _FT((rows,), "i32", {"cur_pos"})
    pt = kvw.page_table if kvw is not None else _FT((rows, W), "i32", {"page_table"})
    rot = {k: (_FT((1, 1, 32, 64), "bf16", {f"cos_{k}"}), _FT((1, 1, 32, 64), "bf16", {f"sin_{k}"}))
           for k in ("yarn", "plain")}  # fmt: skip
    out = attn.forward_decode(
        _FT((1, 1, rows, 4096), "bf16", {"x"}), rot=rot, cur_pos=cur, page_table=pt,
        kv_cache=_FT((20, 1, 64, 576), "bfp8", {"cache"}),
        active=_FT((1, 1, rows if active_rows is None else active_rows, 1024), "bf16", {"act"}), kv_write=kvw,
    )  # fmt: skip
    return list(fake.calls), out


def _rows8(v):
    """A recorded call (or part of one) with every tensor descriptor's dims of 16 read as 8: a T64 op on its 16 rows
    then compares equal to the T32 op on 8 rows (W = 4 and the head dims never equal 16)."""
    if isinstance(v, tuple):
        if len(v) == 4 and v[0] == "T":
            return ("T", tuple(8 if s == 16 else s for s in v[1]), v[2], v[3])
        return tuple(_rows8(x) for x in v)
    return v


def _split_calls(calls):
    """``(pre, kv write, FlashMLA block, post)`` of a decode op sequence: ``pre`` ends with the latent ``concat`` and
    its two frees; the FlashMLA block is one call, or the A'' block (``slice``, call, free, ..., ``concat``, frees)."""
    fms = [i for i, c in enumerate(calls) if c[0] == FM]
    i_kv = max(i for i, c in enumerate(calls[: fms[0]]) if c[0] == "concat") + 3  # concat(n, k_pe), free n, free k_pe
    if len(fms) == 1:
        return calls[:i_kv], calls[i_kv : fms[0]], calls[fms[0] : fms[0] + 1], calls[fms[0] + 1 :]
    lo, hi = fms[0] - 1, fms[-1] + 5  # the first q slice .. the two output frees after the concat
    return calls[:i_kv], calls[i_kv:lo], calls[lo:hi], calls[hi:]


@needs_hook
def test_cpu_wide_flash_groups_op_sequences(monkeypatch):
    """T64 at 16 rows per DP row vs T32 at 8, per layer kind and writer: every op outside the KV write and FlashMLA is
    the T32 op on 16 rows (same kwargs / configs / operand lineage); the global layer's FlashMLA is option A'' (one
    B = 8 call per ``kv_write.flash_groups()`` entry on a dim-1 slice of ``q_mla``, then one dim-1 concat); SWA and the
    MTP layer make one B = 16 call on the writer's ``cur_pos [16]`` / ``page_table [16, W]``; nothing leaks."""
    cfg = host_cfg()
    mtp = A.canonical_attn_spec(cfg, cfg.mtp_layer_idx)
    specs = {"L0 global": cfg.layer(0), "L1 swa": cfg.layer(1), "L53 mtp": mtp}
    writers = (("all_split", "split", ["cur_a0", "cur_b0"]), ("row_split", "natural", ["cur_a", "cur_b"]))
    for name, spec in specs.items():
        glob = spec.sliding_window_size is None
        for mode, gather, upd_src in writers:
            fake32 = _FakeTTNN()
            monkeypatch.setattr(A, "ttnn", fake32)
            c32, _ = _decode_calls(_fake_attention(A, cfg, spec, fake32), fake32, 8, _fake_writer(monkeypatch, fake32,
                                                                                                    mode))  # fmt: skip
            fake = _FakeTTNN()
            monkeypatch.setattr(A, "ttnn", fake)
            kvw = _fake_writer(monkeypatch, fake, mode, rows=WB, gather=gather)
            assert kvw.lanes_per_row == WLPR and [g[0] for g in kvw.flash_groups()] == [slice(0, 8), slice(8, 16)]
            c64, out = _decode_calls(_fake_attention(A, cfg, spec, fake), fake, WLPR, kvw)
            assert out.shape == (1, 1, WLPR, 4096)
            pre32, w32, fm32, post32 = _split_calls(c32)
            pre64, w64, fm64, post64 = _split_calls(c64)
            # q path, latent, epilogue: the T32 ops on 16 rows
            assert [_rows8(c) for c in pre64] == pre32, (name, mode)
            assert [_rows8(c) for c in post64] == post32, (name, mode)
            ups = [dict(c[2])["update_idxs_tensor"][3] for c in w64 if c[0] == "paged_update_cache"]
            assert ups == [(s,) for s in upd_src], (name, mode, ups)  # call A (anchors), then call B (drafts)
            kw32 = dict(fm32[0][2])
            if glob:  # option A'': two B = 8 calls on the anchors' / drafts' slices of q_mla, one dim-1 concat
                assert [c[0] for c in fm64] == A2_BLOCK, (name, mode, [c[0] for c in fm64])
                q_mla = fm64[0][1][0]
                assert q_mla[1] == (1, WLPR, 10, 576)
                for k, (lo, hi, cur) in enumerate(((0, 8, "flash_cur_a"), (8, 16, "flash_cur_d"))):
                    sl, call = fm64[3 * k], fm64[3 * k + 1]
                    assert sl[1] == (q_mla, (0, lo, 0, 0), (1, hi, 10, 576))
                    assert dict(sl[2]) == {"memory_config": "DRAM"}
                    kw = dict(call[2])
                    assert call[1][0] == ("T", (1, 8, 10, 576), "bf16", q_mla[3])  # the slice of q_mla
                    assert kw.pop("cur_pos_tensor")[3] == (cur,) and kw.pop("page_table_tensor")[3] == ("flash_pt",)
                    assert kw == {k2: v for k2, v in kw32.items() if k2 not in ("cur_pos_tensor", "page_table_tensor")}
                cat = fm64[6]
                assert dict(cat[2]) == {"dim": 1, "memory_config": "DRAM"}
                assert [t[1] for t in cat[1]] == [(1, 8, 10, 512), (1, 8, 10, 512)]
            else:  # one B = 16 call on the writer's per-row inputs: the T32 call on 16 rows
                assert [c[0] for c in fm64] == [FM] and fm64[0][1][0][1] == (1, WLPR, 10, 576)
                kw = dict(fm64[0][2])
                assert kw["cur_pos_tensor"][1:] == ((WLPR,), "i32", ("cur_pos",))
                assert kw["page_table_tensor"][1:] == ((WLPR, 4), "i32", ("page_table",))
                assert [_rows8(c) for c in fm64] == fm32
            never, bad = fake.leaks(keep=[out])
            assert not never and not bad, (name, mode, [t.shape for t in never], [t.shape for t in bad])
        log(
            f"{name}: T64 (16 rows) == T32 ops outside the KV write and FlashMLA; FlashMLA "
            + ("A'': 2 x B=8 on q_mla[:, 0:8] / [8:16] (flash_cur_a / flash_cur_d, flash_pt) + dim-1 concat"
               if glob else "one B=16 call on cur_pos [16] / page_table [16, W]")  # fmt: skip
            + "; no leaks (all_split split, row_split)"
        )


class _Writer:
    """A T64 writer stand-in: delegates to ``kvw``; ``groups`` replaces ``flash_groups()`` (``None``: no
    ``flash_groups`` at all) and ``rows`` replaces ``lanes_per_row`` (``None``: no ``lanes_per_row``)."""

    _MISSING = object()

    def __init__(self, kvw, groups=_MISSING, rows=_MISSING):
        self.kvw, self._groups, self._rows = kvw, groups, rows
        self.write, self.cur_pos, self.page_table = kvw.write, kvw.cur_pos, kvw.page_table

    def __getattr__(self, name):
        if name == "flash_groups":
            if self._groups is None:
                raise AttributeError(name)
            if self._groups is not self._MISSING:
                return lambda: self._groups
        if name == "lanes_per_row":
            if self._rows is None:
                raise AttributeError(name)
            if self._rows is not self._MISSING:
                return self._rows
        return getattr(self.kvw, name)


@needs_hook
def test_cpu_wide_guards(monkeypatch):
    """The host guards of ``forward_decode`` raise before any device op: ``kv_write=None`` (the draft-1 8-lane update)
    at 16 rows; a writer whose ``lanes_per_row`` differs from ``x``; an ``active`` mask of another row count; more than
    one tile row; a global layer at 16 rows whose writer has no ``flash_groups()``; groups that do not partition the
    rows in order. SWA layers ignore the groups (one call). Option A = one group over the 16 rows: one B = 16 call on a
    global layer."""
    cfg = host_cfg()
    fake = _FakeTTNN()
    monkeypatch.setattr(A, "ttnn", fake)
    glob = _fake_attention(A, cfg, cfg.layer(0), fake)
    swa = _fake_attention(A, cfg, cfg.layer(1), fake)
    kv64 = _fake_writer(monkeypatch, fake, "all_split", rows=WB, gather="split")
    kv32 = _fake_writer(monkeypatch, fake, "all_split")

    def refused(attn, rows, kvw, match, **kw):
        fake.calls.clear()
        with pytest.raises(ValueError, match=match):
            _decode_calls(attn, fake, rows, kvw, **kw)
        assert fake.calls == [], f"refused after {len(fake.calls)} ops"

    def fm_calls(attn, rows, kvw):
        calls, _ = _decode_calls(attn, fake, rows, kvw)
        return [c for c in calls if c[0] == FM]

    for attn in (glob, swa):
        refused(attn, WLPR, None, "draft-1 update")
        refused(attn, LPR, kv64, "lanes_per_row")
        refused(attn, WLPR, kv32, "lanes_per_row")
        refused(attn, 40, kv64, "one tile row")
        refused(attn, WLPR, _Writer(kv64, rows=LPR), "lanes_per_row")
        refused(attn, WLPR, kv64, "active mask has 8 rows", active_rows=LPR)
        refused(attn, LPR, kv32, "active mask has 16 rows", active_rows=WLPR)
    # a writer without flash_groups: global layers at 8 rows only (one call); SWA layers at 16 too
    refused(glob, WLPR, _Writer(kv64, groups=None), "flash_groups")
    assert len(fm_calls(glob, LPR, _Writer(kv32, groups=None, rows=None))) == 1
    assert [dict(c[2])["cur_pos_tensor"][1] for c in fm_calls(swa, WLPR, _Writer(kv64, groups=None))] == [(WLPR,)]
    # groups must partition [0, 16) in order (slices, step 1, (rows, cur_pos, page_table) triples)
    d = kv64.device_inputs
    ca, cd, fp = d["flash_cur_a"], d["flash_cur_d"], d["flash_pt"]
    for bad in (
        [],
        [(slice(0, 8), ca, fp)],
        [(slice(0, 8), ca, fp), (slice(9, 16), cd, fp)],
        [(slice(8, 16), cd, fp), (slice(0, 8), ca, fp)],
        [(slice(0, 9), ca, fp), (slice(8, 16), cd, fp)],
        [(slice(0, 8), ca, fp), (slice(8, 17), cd, fp)],
        [(slice(0, 16, 2), ca, fp)],
        [(slice(0, 8), ca), (slice(8, 16), cd, fp)],
        [((0, 8), ca, fp), ((8, 16), cd, fp)],
    ):
        refused(glob, WLPR, _Writer(kv64, groups=bad), "flash_groups")
        assert len(fm_calls(swa, WLPR, _Writer(kv64, groups=bad))) == 1  # SWA: one B = 16 call, groups unused
    # option A: one group over all 16 rows = one B = 16 call (G16 measures it; not bitwise on global layers)
    opt_a = fm_calls(glob, WLPR, _Writer(kv64, groups=[(slice(0, WLPR), kv64.cur_pos, kv64.page_table)]))
    assert [(c[1][0][1], dict(c[2])["cur_pos_tensor"][3]) for c in opt_a] == [((1, WLPR, 10, 576), ("cur_pos",))]
    # three groups of any sizes are run as given
    three = [(slice(0, 4), ca, fp), (slice(4, 8), ca, fp), (slice(8, 16), cd, fp)]
    assert [c[1][0][1][1] for c in fm_calls(glob, WLPR, _Writer(kv64, groups=three))] == [4, 4, 8]
    log("T64 guards: draft-1 / writer-row / mask-row / tile-row / flash_groups refusals before any op; SWA ignores "
        "the groups; option A = one B=16 call")  # fmt: skip


def test_cpu_rope_rows_per_dp(monkeypatch):
    """``rope``'s decode host helpers at 16 rows per DP row (``rows_per_dp=16``): the T64 step's 64 row positions ->
    ``[dp, 32]`` index rows with each DP row's ``[8 anchors (n) | 8 drafts (n + 1)]`` first, idle rows and pads 0;
    ``MotifRope.rot_idxs_host`` / ``rot_idxs_device`` pass it through; the 32-lane default is unchanged;
    ``MotifAttention.active_mask_host(rows_per_dp=16)``; the refusals."""
    from models.demos.motif3.tests.unit.test_kv_write import wide_lanes
    from models.demos.motif3.tt import embedding as E

    cfg = host_cfg()
    for r in (None, 1, 8, 16, 32):
        assert RO.decode_rows_per_dp(cfg, r) == E.decode_rows_per_dp(cfg, r) == (8 if r is None else r)
    for r in (0, -1, 33):
        with pytest.raises(ValueError, match="rows_per_dp"):
            RO.decode_rows_per_dp(cfg, r)
    W = 6
    for seed in range(4):
        pos, pt, draft = wide_lanes(seed, W)
        st = KW.KVWriteStep.wide_verify(pos, pt, draft)
        idx = RO.positions_to_rot_idxs(st.positions, cfg, rows_per_dp=WLPR)
        assert idx.shape == (cfg.dp, 32) and idx.dtype == torch.int32
        for lane in range(B):
            r, j = divmod(lane, LPR)
            assert KW.wide_rows(lane) == (WLPR * r + j, WLPR * r + LPR + j)
            assert int(idx[r, j]) == max(int(pos[lane]), 0)
            assert int(idx[r, LPR + j]) == (int(pos[lane]) + 1 if bool(draft[lane]) else 0)
        assert not bool(idx[:, WLPR:].any())
        assert torch.equal(RO.lanes_to_rows(st.page_table, cfg, rows_per_dp=WLPR).reshape(WB, W), st.page_table)
        # the 32-lane default: unchanged, and equal to rows_per_dp=8
        assert torch.equal(RO.positions_to_rot_idxs(pos, cfg), RO.positions_to_rot_idxs(pos, cfg, rows_per_dp=LPR))
        assert torch.equal(RO.positions_to_rot_idxs(pos, cfg)[:, :LPR].reshape(B), pos.clamp_min(0))
    with pytest.raises(ValueError, match="expected 64"):
        RO.positions_to_rot_idxs(pos, cfg, rows_per_dp=WLPR)
    with pytest.raises(ValueError, match="expected 32"):
        RO.positions_to_rot_idxs(st.positions, cfg)
    with pytest.raises(ValueError, match="max_model_len"):
        RO.positions_to_rot_idxs(torch.full((WB,), cfg.max_model_len), cfg, rows_per_dp=WLPR)
    # MotifRope.rot_idxs_host / _device: the same index rows through shard_lanes
    sent = []
    monkeypatch.setattr(RO, "shard_lanes", lambda rows, cfg_, mesh, dtype=None, device=None: sent.append(rows) or rows)
    rope = object.__new__(RO.MotifRope)
    rope.cfg, rope.mesh_device = cfg, "mesh"
    assert torch.equal(rope.rot_idxs_host(st.positions, rows_per_dp=WLPR), idx)
    assert torch.equal(rope.rot_idxs_device(st.positions, rows_per_dp=WLPR), idx)
    assert torch.equal(rope.rot_idxs_host(pos), RO.positions_to_rot_idxs(pos, cfg)) and len(sent) == 3
    # active_mask_host: [dp, 1, rows_per_dp, width] 0/1 rows
    monkeypatch.setattr(A, "_shard_rows", lambda rows, cfg_, mesh, dtype, layout, device=None: rows)
    m64 = MotifAttention.active_mask_host(st.positions, cfg, "mesh", rows_per_dp=WLPR)
    assert m64.shape == (cfg.dp, 1, WLPR, A.MASK_WIDTH)
    assert torch.equal(m64[:, 0, :, 0].reshape(WB), (st.positions >= 0).float())
    m32 = MotifAttention.active_mask_host(pos, cfg, "mesh", width=1)
    assert m32.shape == (cfg.dp, 1, LPR, 1) and torch.equal(m32.reshape(B), (pos >= 0).float())
    with pytest.raises(ValueError, match="expected 32 positions"):
        MotifAttention.active_mask_host(st.positions, cfg, "mesh")
    log("rope rows_per_dp=16: T64 index rows [anchors n | drafts n + 1 | pads 0] per DP row; the 32-lane default "
        "unchanged; active_mask_host(rows_per_dp=16)")  # fmt: skip


# ======================================================================================================================
# device
# ======================================================================================================================
def read_caches(caches, mesh_device, cfg, all_chips=(0,)):
    """Per layer ``{(dp, tp): host cache}``: layer indices in ``all_chips`` on all 32 chips, the others on tp 0 / 7."""
    out = []
    C = int(cfg.axes.mesh_shape[1])
    for l, c in enumerate(caches):
        shards = ttnn.get_device_tensors(c)
        tps = range(cfg.tp) if l in all_chips else (0, cfg.tp - 1)
        d = {}
        for dp in range(cfg.dp):
            for tp in tps:
                r, cc = cfg.axes.coord(dp, tp)
                d[(dp, tp)] = ttnn.to_torch(shards[r * C + cc]).float()
        out.append(d)
    return out


def slot(pt: torch.Tensor, lane: int, p: int):
    return int(pt[lane, p // BS]), p % BS


def changed_slots(cache: torch.Tensor, base: torch.Tensor):
    return {(int(b), int(i)) for b, i in torch.nonzero((cache != base).any(-1)[:, 0]).tolist()}


@needs_hook
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_attention_kvr_decode(mesh_device, device_params):
    from models.demos.motif3.tt.ccl import replicas_identical

    cfg, ccl, rope = _setup(mesh_device, "kv_write")
    args = ref_args()
    layer_ids = (0, 1, cfg.mtp_layer_idx)
    attns = []
    for l in layer_ids:
        t = random_attn_tensors(args, seed=2000 + l)
        src = source_for(t, MTP_ATTN_PREFIX) if l == cfg.mtp_layer_idx else hf_source(t, l)
        attns.append(MotifAttention(mesh_device, cfg, l, source=src, ccl=ccl, rope=rope, cache=False))
    kinds = [f"L{l} {a.spec.attn_kind if l < cfg.num_layers else 'mtp'}" for l, a in zip(layer_ids, attns)]
    dt = cfg.dtypes.kv_cache
    W = 4
    N = B * W + 8
    g = torch.Generator().manual_seed(4100)
    bases = [quant(torch.randn(N, 1, BS, D, generator=g), dt) for _ in layer_ids]
    hosts = [upload_cache(mesh_device, b, dt, device=False) for b in bases]
    caches = [upload_cache(mesh_device, b, dt) for b in bases]
    pos, pt, draft = lane_layout(41, W)
    act_l = [l for l in range(B) if int(pos[l]) >= 0]
    x_n = torch.randn(B, 4096, generator=g).bfloat16().float()
    x_n1 = torch.randn(B, 4096, generator=g).bfloat16().float()
    failures: List[str] = []

    def reset():
        for c, h in zip(caches, hosts):
            ttnn.copy_host_to_device_tensor(h, c)

    def run(positions, x32, kvw=None, step=None):
        """One decode step of the three layers; returns per layer ``[32, 4096]`` (lane order) and whether every
        output is identical over the TP chips of its row."""
        if kvw is not None:
            kvw.write_step(step)
            cur, ptt = kvw.cur_pos, kvw.page_table
        else:
            cur = _rows(mesh_device, cfg, positions.to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            ptt = _rows(mesh_device, cfg, torch.where((positions >= 0)[:, None], pt, 0).to(torch.int32), ttnn.int32,
                        ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        x_tt = _lane_x(mesh_device, cfg, x32)
        rot_idx = rope.rot_idxs_device(positions)
        rot = MotifAttention.decode_rope_tables(rope, rot_idx)
        act = MotifAttention.active_mask_from_cur_pos(cur, cfg.lanes_per_row)
        outs, same = [], True
        for attn, cache in zip(attns, caches):
            o = attn.forward_decode(x_tt, rot=rot, cur_pos=cur, page_table=ptt, kv_cache=cache, active=act,
                                    kv_write=kvw)  # fmt: skip
            outs.append(_per_lane_out(cfg, mesh_device, o))
            same = same and replicas_identical(o, mesh_device, "tp", cfg.axes)
            ttnn.deallocate(o)
        if kvw is not None:
            kvw.end_step()
        else:
            ttnn.deallocate(cur)
            ttnn.deallocate(ptt)
        for t in [x_tt, rot_idx, act] + [t for cs in rot.values() for t in cs]:
            ttnn.deallocate(t)
        return outs, same

    # ---- reference (draft 1): step A at n, then step B = drafted owners at n + 1 on their own lanes ----------------
    reset()
    out_a, _ = run(pos, x_n)
    ref_a = read_caches(caches, mesh_device, cfg, all_chips=())
    pos_b = torch.where(draft, pos + 1, torch.full_like(pos, -1))
    x_b = torch.where(draft[:, None], x_n1, torch.zeros_like(x_n1))
    out_b, _ = run(pos_b, x_b)
    ref_ab = read_caches(caches, mesh_device, cfg, all_chips=())
    owners = [l for l in range(B) if bool(draft[l])]

    def ref_row(ref, l: int, lane: int, p: int):
        b, i = slot(pt, lane, p)
        return ref[l][(lane // LPR, 0)][b, 0, i]

    def check_step(tag, mode, outs, same, writes, out_want, partner_ref=None):
        """``writes`` = [(lane, position, page-table lane, value source)], source ``"a"`` | ``"ab"`` (bitwise from the
        reference caches) or ``"partner"`` (vs step B: PCC, bitwise reported). Checks outputs, slots, values."""
        got = read_caches(caches, mesh_device, cfg)
        ok = same
        bit_out = all(torch.equal(outs[l][ln], out_want[l][ln]) for l in range(len(layer_ids)) for ln in out_want[l])
        ok = ok and bit_out
        part_pcc, part_bit = 1.0, True
        if partner_ref:
            for l in range(len(layer_ids)):
                for d, o in partner_ref.items():
                    part_pcc = min(part_pcc, pcc(out_b[l][o], outs[l][d]))
                    part_bit = part_bit and torch.equal(out_b[l][o], outs[l][d])
            ok = ok and part_pcc >= 0.9999
        slot_ok, val_ok, val_bit, kvr_same = True, True, True, True
        for l in range(len(layer_ids)):
            for (dp, tp), c in got[l].items():
                lanes = [w for w in writes if KW.is_replicated(mode) or w[0] // LPR == dp]
                want = {slot(pt, w[2], w[1]) for w in lanes}
                if changed_slots(c, bases[l]) != want:
                    slot_ok = False
                for lane, p, pl, src in lanes:
                    b, i = slot(pt, pl, p)
                    if src == "partner":
                        ref = ref_row(ref_ab, l, pl, p)
                        val_bit = val_bit and torch.equal(c[b, 0, i], ref)
                        val_ok = val_ok and pcc(ref, c[b, 0, i]) >= 0.9999
                    else:
                        val_ok = val_ok and torch.equal(c[b, 0, i], ref_row(ref_a if src == "a" else ref_ab, l, pl, p))
            if KW.is_replicated(mode):
                first = next(iter(got[l].values()))
                kvr_same = kvr_same and all(torch.equal(first, c) for c in got[l].values())
        ok = ok and slot_ok and val_ok and kvr_same
        n_part = len(partner_ref or {})
        log(
            f"{mode:9s} {tag:9s}: {len([w for w in writes])} writes ({n_part} partners); outputs bitwise vs draft 1 "
            f"{bit_out}, replicas(tp) identical {same}"
            + (f"; partners vs step B pcc {part_pcc:.6f} bitwise {part_bit}" if n_part else "")
            + f"; caches: exactly the expected slots on every chip {slot_ok}, rows bitwise vs the reference {val_ok}"
            + (f" (partner rows bitwise {val_bit})" if n_part else "")
            + (f", 32 copies identical (KV-R) {kvr_same}" if KW.is_replicated(mode) else "")
            + f" [{', '.join(kinds)}]"
        )
        if not ok:
            failures.append(f"{mode}/{tag}: same {same} out {bit_out} partner {part_pcc:.6f} slots {slot_ok} values "
                            f"{val_ok} kvr {kvr_same}")  # fmt: skip
        return part_bit and val_bit

    relocation_bitwise = True
    for mode in ("row", "all", "row_split", "all_split"):
        kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode=mode)
        # ordinary step == draft 1
        reset()
        outs, same = run(pos, x_n, kvw, KW.KVWriteStep.ordinary(pos, pt))
        writes = [(l, int(pos[l]), l, "a") for l in act_l]
        check_step("ordinary", mode, outs, same, writes, [{ln: out_a[l][ln] for ln in act_l} for l in range(3)])
        if kvw.split:
            partner_of, overflow = KW.assign_partner_lanes(pos, draft, cross_row=kvw.cross_row_partners)
            st = KW.KVWriteStep.packed_verify(pos, pt, partner_of)
            x_p = x_n.clone()
            for o, d in partner_of.items():
                x_p[d] = x_n1[o]
            reset()
            outs, same = run(st.positions, x_p, kvw, st)
            writes = [(l, int(pos[l]), l, "a") for l in act_l]
            writes += [(d, int(pos[o]) + 1, o, "partner") for o, d in partner_of.items()]
            n_cross = sum(d // LPR != o // LPR for o, d in partner_of.items())
            n_same_tile = sum(same_tile(int(pos[o])) for o in partner_of)
            relocation_bitwise &= check_step(
                "verify",
                mode,
                outs,
                same,
                writes,
                [{ln: out_a[l][ln] for ln in act_l} for l in range(3)],
                partner_ref={d: o for o, d in partner_of.items()},
            )
            log(
                f"{mode:9s} verify   : partners {len(partner_of)} ({n_cross} on another DP row, {n_same_tile} sharing "
                f"the owner's tile), overflow {len(overflow)}"
            )
            if overflow:  # second replay: each overflow draft on its own lane at n + 1 (the caches keep pass 1)
                st2 = KW.KVWriteStep.overflow_pass(pos, pt, overflow)
                x_o = torch.where(st2.active[:, None], x_n1, torch.zeros_like(x_n1))
                outs, same = run(st2.positions, x_o, kvw, st2)
                writes += [(o, int(pos[o]) + 1, o, "ab") for o in overflow]
                check_step("overflow", mode, outs, same, writes, [{o: out_b[l][o] for o in overflow} for l in range(3)])
        kvw.deallocate()

    # ---- all_split traced: one capture, replayed for an ordinary and a verify step --------------------------------
    kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode="all_split")
    inactive = KW.KVWriteStep.inactive(W)
    x_tt = _lane_x(mesh_device, cfg, torch.zeros(B, 4096))
    rot_idx = rope.rot_idxs_device(inactive.positions)
    partner_of, _ = KW.assign_partner_lanes(pos, draft, cross_row=True)
    st_v = KW.KVWriteStep.packed_verify(pos, pt, partner_of)
    x_p = x_n.clone()
    for o, d in partner_of.items():
        x_p[d] = x_n1[o]
    eager = {}
    for name, st, x32 in (("ordinary", KW.KVWriteStep.ordinary(pos, pt), x_n), ("verify", st_v, x_p)):
        reset()
        eager[name] = (run(st.positions, x32, kvw, st)[0], read_caches(caches, mesh_device, cfg, all_chips=()))

    def step_fn():
        rot = MotifAttention.decode_rope_tables(rope, rot_idx)
        act = MotifAttention.active_mask_from_cur_pos(kvw.cur_pos, cfg.lanes_per_row)
        outs = []
        for attn, cache in zip(attns, caches):
            outs.append(attn.forward_decode(x_tt, rot=rot, cur_pos=kvw.cur_pos, page_table=kvw.page_table,
                                            kv_cache=cache, active=act, kv_write=kvw))  # fmt: skip
        kvw.end_step()
        for t in [act] + [t for cs in rot.values() for t in cs]:
            ttnn.deallocate(t)
        return outs

    kvw.write_step(inactive)
    for o in step_fn():  # eager warmup of the exact captured shapes
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh_device)
    n_prog = mesh_device.num_program_cache_entries()
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        touts = step_fn()
    finally:
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    try:
        for name, st, x32 in (("ordinary", KW.KVWriteStep.ordinary(pos, pt), x_n), ("verify", st_v, x_p)):
            reset()
            kvw.write_step(st)
            ttnn.copy_host_to_device_tensor(rope.rot_idxs_host(st.positions), rot_idx)
            ttnn.copy_host_to_device_tensor(
                shard_lanes(x32.reshape(cfg.dp, 1, cfg.lanes_per_row, 4096), cfg, mesh_device, dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=None), x_tt)  # fmt: skip
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            outs = [_per_lane_out(cfg, mesh_device, o) for o in touts]
            got = read_caches(caches, mesh_device, cfg, all_chips=())
            e_out, e_c = eager[name]
            act_lanes = [l for l in range(B) if int(st.positions[l]) >= 0]
            out_eq = all(torch.equal(outs[l][act_lanes], e_out[l][act_lanes]) for l in range(3))
            c_eq = all(torch.equal(got[l][k], e_c[l][k]) for l in range(3) for k in got[l])
            log(f"all_split traced {name:8s}: outputs bitwise == eager {out_eq}, caches bitwise == eager {c_eq}")
            if not (out_eq and c_eq):
                failures.append(f"traced {name}: outputs {out_eq} caches {c_eq}")
        n_prog_end = mesh_device.num_program_cache_entries()
    finally:
        ttnn.release_trace(mesh_device, tid)
    log(f"all_split traced: programs compiled by the capture and the replays {n_prog_end - n_prog}")
    if n_prog_end != n_prog:
        failures.append(f"{n_prog_end - n_prog} programs compiled after the warmup")
    for t in touts + [x_tt, rot_idx] + caches:
        ttnn.deallocate(t)
    kvw.deallocate()
    log(f"lane relocation (partner rows / outputs bitwise equal to the owner lane's next step): {relocation_bitwise}")
    assert not failures, "\n".join(failures)


# ---- T64 device: the attention at 16 rows per DP row vs two T32 steps (design §7.2 "T64, attention") -----------------
WIDE_DEVICE_RUNS = {"bfp8": (("all_split", "split"), ("row_split", "natural")), "bf16": (("all_split", "split"),)}


def _rows_x(mesh_device, cfg, x: torch.Tensor):
    """``x [dp * L, 4096]`` in DP-row order (the 32 lanes, or the T64 step's 64 rows) -> the decode input ``[1, 1, L,
    4096]`` per DP row."""
    return _rows(mesh_device, cfg, x.reshape(cfg.dp, 1, -1, x.shape[-1]), ttnn.bfloat16, ttnn.TILE_LAYOUT)


def _per_row_out(cfg, mesh_device, out) -> torch.Tensor:
    """Decode output ``[1, 1, L, 4096]`` per DP row -> ``[dp * L, 4096]`` in DP-row order, from TP index 0 of each
    row."""
    from models.demos.motif3.tt.ccl import device_tensors_to_torch

    full = device_tensors_to_torch(out, mesh_device).float()  # [R, C, 1, 1, L, 4096]
    L = int(full.shape[-2])
    res = torch.zeros(cfg.dp * L, full.shape[-1])
    for dp in range(cfg.dp):
        r, c = cfg.axes.coord(dp, 0)
        res[L * dp : L * (dp + 1)] = full[r, c, 0, 0]
    return res


@needs_hook
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("kv_name", ["bfp8", "bf16"])
def test_attention_kvr_wide_decode(mesh_device, device_params, kv_name):
    """Work package A2's module test (design §7.2 "T64, attention"): ``forward_decode`` at 16 rows per DP row with
    option A'' through the real ``DecodeKVWrite(rows=64)`` vs two T32 steps of the same KV mode on the same history:
    step A = the anchors at ``n`` (an ordinary T32 step), then step D = the drafts at ``n + 1`` (an ordinary T32 step
    of the drafted lanes, after A).

    Pass: every anchor row bitwise == its step-A row and every draft row bitwise == its step-D row (inactive rows 0 in
    both), on layers 0 (global; contexts ~1K / ~4K / ~32K), 1 (SWA) and the MTP layer; replicas identical over TP;
    every chip's cache (layer 0: all 32 chips; the others: TP 0 / 7 of every row) bitwise == the caches after steps A
    and D. Negative control: option A (one B = 16 call) is not bitwise on the global layer (PCC >= 0.9999) and is the
    same call on the SWA layers. bfp8 ``all_split``: one capture of the T64 step, replayed for the verify step and an
    ordinary T64 step (no drafts), outputs and caches bitwise == eager, no program compiled after the warmup. An L1
    pin page is alive throughout: a new program whose static CBs reach it fails (the design's CB-end check)."""
    from models.demos.motif3.tt.ccl import MotifCCL, log_fabric, replicas_identical
    from models.demos.motif3.tt.rope import MotifRope

    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device, kv_cache_dtype=kv_name)
    assert log_fabric(mesh_device, f"attention kvr wide {kv_name}")["committed"] is not None
    ccl0 = MotifCCL(mesh_device, cfg)
    ccl, how = t64_ccl(ccl0)
    rope = MotifRope(mesh_device, cfg)
    pin, pin_addr = l1_pin(mesh_device)
    args = ref_args()
    layer_ids = (0, 1, cfg.mtp_layer_idx)
    attns = []
    for l in layer_ids:
        t = random_attn_tensors(args, seed=2000 + l)
        src = source_for(t, MTP_ATTN_PREFIX) if l == cfg.mtp_layer_idx else hf_source(t, l)
        attns.append(MotifAttention(mesh_device, cfg, l, source=src, ccl=ccl0, rope=rope, cache=False))
    kinds = [f"L{l} {a.spec.attn_kind if l < cfg.num_layers else 'mtp'}" for l, a in zip(layer_ids, attns)]
    dt = cfg.dtypes.kv_cache
    W = 512
    pos, pt, draft, n_blocks = flash_layout(W)  # anchors at ~1K / ~4K / ~32K, idle and undrafted lanes
    N_ = max(n_blocks + 8, W)  # paged_update_cache needs the page-table width <= the cache's blocks
    g = torch.Generator().manual_seed(4200)
    bases = [quant(torch.randn(N_, 1, BS, D, generator=g), dt) for _ in layer_ids]
    hosts = [upload_cache(mesh_device, b, dt, device=False) for b in bases]
    caches = [upload_cache(mesh_device, b, dt) for b in bases]
    x_a = torch.randn(B, 4096, generator=g).bfloat16().float()  # the anchors' attention inputs, lane order
    x_d = torch.where(draft[:, None], torch.randn(B, 4096, generator=g).bfloat16().float(), torch.zeros(B, 4096))
    pos_d = torch.where(draft, pos + 1, torch.full_like(pos, -1))
    x64 = torch.zeros(WB, 4096)
    x64[A_ROWS], x64[D_ROWS] = x_a, x_d
    act_a = [l for l in range(B) if int(pos[l]) >= 0]
    act_d = [l for l in range(B) if bool(draft[l])]
    failures: List[str] = []
    log(f"T64 attention {kv_name}: split-order gather by {how}; L1 pin page at {pin_addr}; {len(act_a)} anchors, "
        f"{len(act_d)} drafts; anchors n in {sorted(set(int(p) for p in pos if p >= 0))}")  # fmt: skip

    def reset():
        for c, h in zip(caches, hosts):
            ttnn.copy_host_to_device_tensor(h, c)

    def run(writer, step, x_rows, rows_per_dp=None):
        """One decode step of the three layers through ``writer``; per layer ``[dp * L, 4096]`` (DP-row order) and
        whether every output is identical over the TP chips of its row."""
        writer.write_step(step)
        x_tt = _rows_x(mesh_device, cfg, x_rows)
        rot_idx = rope.rot_idxs_device(step.positions, rows_per_dp=rows_per_dp)
        rot = MotifAttention.decode_rope_tables(rope, rot_idx)
        act = MotifAttention.active_mask_from_cur_pos(writer.cur_pos, int(x_rows.shape[0]) // cfg.dp)
        outs, same = [], True
        for attn, cache in zip(attns, caches):
            o = attn.forward_decode(x_tt, rot=rot, cur_pos=writer.cur_pos, page_table=writer.page_table,
                                    kv_cache=cache, active=act, kv_write=writer)  # fmt: skip
            outs.append(_per_row_out(cfg, mesh_device, o))
            same = same and replicas_identical(o, mesh_device, "tp", cfg.axes)
            ttnn.deallocate(o)
        writer.end_step()
        for t in [x_tt, rot_idx, act] + [t for cs in rot.values() for t in cs]:
            ttnn.deallocate(t)
        return outs, same

    for mode, gather in WIDE_DEVICE_RUNS[kv_name]:
        tag = f"{kv_name} {mode}" + (f"/{gather}" if KW.is_replicated(mode) else "")
        # ---- T32 references: step A (anchors at n), then step D (the drafted lanes at n + 1) on the same caches -----
        w32 = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl0, page_table_width=W, mode=mode)
        reset()
        out_a, same_a = run(w32, KW.KVWriteStep.ordinary(pos, pt), x_a)
        out_d, same_d = run(w32, KW.KVWriteStep.ordinary(pos_d, pt), x_d)
        ref = read_caches(caches, mesh_device, cfg)
        w32.deallocate()
        # ---- T64: anchors (call A) and drafts (call B) in one step, FlashMLA option A'' ---------------------------
        w64 = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode=mode, rows=WB, gather=gather)
        assert w64.lanes_per_row == WLPR and len(w64.flash_groups()) == 2
        st = KW.KVWriteStep.wide_verify(pos, pt, draft)
        reset()
        out64, same64 = run(w64, st, x64, rows_per_dp=WLPR)
        got = read_caches(caches, mesh_device, cfg)
        ok = same_a and same_d and same64
        for li, kind in enumerate(kinds):
            o_an, o_dr = out64[li][A_ROWS], out64[li][D_ROWS]
            a_bit, d_bit = torch.equal(o_an, out_a[li]), torch.equal(o_dr, out_d[li])
            a_max, d_max = float((o_an - out_a[li]).abs().max()), float((o_dr - out_d[li]).abs().max())
            idle = [x for x in range(B) if x not in act_a]
            undrafted = [x for x in range(B) if x not in act_d]
            zero = not bool(o_an[idle].any()) and not bool(o_dr[undrafted].any())
            live = all(bool(o[act].abs().amax(-1).gt(0).all()) for o, act in ((out_a[li], act_a), (out_d[li], act_d)))
            fin = bool(torch.isfinite(out64[li]).all())
            log(f"{tag:22s} {kind:10s}: T64 anchors == T32 step A bitwise {a_bit} (max |d| {a_max:.3g}), drafts == T32 "
                f"step D bitwise {d_bit} (max |d| {d_max:.3g}); inactive rows 0 {zero}; active rows non-zero {live}; "
                f"finite {fin}")  # fmt: skip
            if not (a_bit and d_bit and zero and live and fin):
                failures.append(f"{tag} {kind}: anchors {a_bit} ({a_max:.3g}) drafts {d_bit} ({d_max:.3g}) zero {zero} "
                                f"live {live} finite {fin}")  # fmt: skip
        c_eq = all(torch.equal(got[l][k], ref[l][k]) for l in range(len(layer_ids)) for k in got[l])
        changed = all(not torch.equal(ref[l][(0, 0)], bases[l]) for l in range(len(layer_ids)))
        n_chips = sum(len(got[l]) for l in range(len(layer_ids)))
        log(f"{tag:22s} caches: T64 == T32 steps A + D bitwise on {n_chips} chip copies {c_eq} (written {changed}); "
            f"replicas(tp) identical: T32 {same_a and same_d}, T64 {same64}")  # fmt: skip
        if not (ok and c_eq and changed):
            failures.append(f"{tag}: caches {c_eq} written {changed} replicas T32 {same_a and same_d} T64 {same64}")
        # ---- negative control: option A (one B = 16 call; G16 measures it) is not bitwise on the global layer ------
        reset()
        opt_a = _Writer(w64, groups=[(slice(0, WLPR), w64.cur_pos, w64.page_table)])
        out_oa, _ = run(opt_a, st, x64, rows_per_dp=WLPR)
        for li, kind in enumerate(kinds):
            eq = torch.equal(out_oa[li], out64[li])
            p = pcc(out_oa[li], out64[li])
            d_max = float((out_oa[li] - out64[li]).abs().max())
            glob = attns[li].window is None
            want = (not eq and p >= 0.9999) if glob else eq
            log(f"{tag:22s} {kind:10s}: option A (one B=16 call) vs A'' bitwise {eq}, max |d| {d_max:.3g}, pcc {p:.6f} "
                f"(expected: {'different, pcc >= 0.9999' if glob else 'the same call'}) {want}")  # fmt: skip
            if not want:
                failures.append(f"{tag} {kind}: option A vs A'' bitwise {eq} pcc {p:.6f}")
        # ---- traced: one capture of the T64 step, replayed for the verify step and an ordinary T64 step ------------
        if kv_name == "bfp8" and mode == "all_split":
            st0 = KW.KVWriteStep.wide_verify(pos, pt)  # no drafts: idle draft rows (the `wide` mode's ordinary step)
            x0 = x64.clone()
            x0[D_ROWS] = 0.0
            eager = {"verify": (out64, got)}
            reset()
            o0, _ = run(w64, st0, x0, rows_per_dp=WLPR)
            eager["ordinary"] = (o0, read_caches(caches, mesh_device, cfg, all_chips=()))
            inactive = KW.KVWriteStep.inactive(W, lanes=WB)
            w64.write_step(inactive)
            x_tt = _rows_x(mesh_device, cfg, torch.zeros(WB, 4096))
            rot_idx = rope.rot_idxs_device(inactive.positions, rows_per_dp=WLPR)

            def step_fn():
                rot = MotifAttention.decode_rope_tables(rope, rot_idx)
                act = MotifAttention.active_mask_from_cur_pos(w64.cur_pos, WLPR)
                outs = [attn.forward_decode(x_tt, rot=rot, cur_pos=w64.cur_pos, page_table=w64.page_table,
                                            kv_cache=cache, active=act, kv_write=w64)
                        for attn, cache in zip(attns, caches)]  # fmt: skip
                w64.end_step()
                for t in [act] + [t for cs in rot.values() for t in cs]:
                    ttnn.deallocate(t)
                return outs

            for o in step_fn():  # eager warmup of the exact captured shapes (all rows inactive: writes nothing)
                ttnn.deallocate(o)
            ttnn.synchronize_device(mesh_device)
            n_prog = mesh_device.num_program_cache_entries()
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            try:
                touts = step_fn()
            finally:
                ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            try:
                for name, s_, xr in (("verify", st, x64), ("ordinary", st0, x0)):
                    reset()
                    w64.write_step(s_)
                    ttnn.copy_host_to_device_tensor(rope.rot_idxs_host(s_.positions, rows_per_dp=WLPR), rot_idx)
                    ttnn.copy_host_to_device_tensor(
                        shard_lanes(xr.reshape(cfg.dp, 1, WLPR, 4096), cfg, mesh_device, dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=None), x_tt)  # fmt: skip
                    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                    outs = [_per_row_out(cfg, mesh_device, o) for o in touts]
                    got_t = read_caches(caches, mesh_device, cfg, all_chips=())
                    e_out, e_c = eager[name]
                    out_eq = all(torch.equal(outs[l], e_out[l]) for l in range(len(layer_ids)))
                    c_eq = all(torch.equal(got_t[l][k], e_c[l][k]) for l in range(len(layer_ids)) for k in got_t[l])
                    log(f"{tag:22s} traced {name:8s}: outputs (64 rows, 3 layers) bitwise == eager {out_eq}, caches "
                        f"bitwise == eager {c_eq}")  # fmt: skip
                    if not (out_eq and c_eq):
                        failures.append(f"{tag} traced {name}: outputs {out_eq} caches {c_eq}")
                n_prog_end = mesh_device.num_program_cache_entries()
            finally:
                ttnn.release_trace(mesh_device, tid)
            log(f"{tag:22s} traced: programs compiled by the capture and the replays {n_prog_end - n_prog}")
            if n_prog_end != n_prog:
                failures.append(f"{tag}: {n_prog_end - n_prog} programs compiled after the warmup")
            for t in touts + [x_tt, rot_idx]:
                ttnn.deallocate(t)
        w64.deallocate()
    for t in caches + [pin]:
        ttnn.deallocate(t)
    assert not failures, "\n".join(failures)
