# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``tt/kv_write.py``: the decode KV write of every decode layer (features design §3.4-§3.5, D6 KV-R, D10; README §16;
gates G12 / G13a in ``tests/unit/gates/GATES_RESULTS.md`` §12.6-§12.7).

Host (devices hidden; ``-k host``): modes and lanes per call; :class:`KVWriteStep` constructors (ordinary, packed
verify, overflow pass); the reference partner packing; every refusal of :func:`check_kv_write_step`; the update-call
plan (covers every write once, no two users of a call on one block or tile); the host model; a tile-race emulation
showing why the split exists (one call loses same-tile owner / partner updates, the split loses none); the device-input
values; and the device op wiring of :meth:`DecodeKVWrite.write` / ``write_step`` with a recording fake ``ttnn``.

Device (``-k device``; mesh (4, 8), ``model_config.device_params()``):

* ``test_kv_write_device_modes``: every mode x KV dtype (bfp8, bf16) x step kind (ordinary, packed verify with
  owners at ``p % 64`` in {0, 30, 31, 62, 63}, cross-row partners in ``all_split``, overflow pass, inactive): three
  layer caches, every chip's cache bit-exact vs the host model (KV-R: all 32 chips hold all 32 lanes; row modes: each
  DP row its own 8).
* ``test_kv_write_device_trace``: one capture per mode (bfp8; plus ``all_split`` as 2 x 16-lane calls with a bf16
  cache and as 4 x 8-lane calls, the L1 fallback, with bfp8), replayed for different steps written by ``write_step``:
  bit-exact vs the host model after every replay, no program compiled by the capture or the replays, host cost of
  ``write_step``.
* ``test_kv_write_device_flash_reads``: FlashMLA (G1 config) on ``kv_write.cur_pos`` / ``page_table`` after a packed
  verify write, every active lane vs the fp64 golden on the host-model cache (global and SWA), and the negative
  control: ``row_split`` with cross-row partners (refused by the check, forced with ``validate=False``) reads a stale
  anchor.
* ``test_kv_write_device_cost``: traced us per layer of each mode (54 writes per trace, slope method of the gates).

T64, 64 rows with ``[8 anchors | 8 drafts]`` per DP row (``docs/p5_t64/P5_T64_DESIGN.md`` §4.1, §4.3, §6.2 G-S1w, §7.1):

* host (``-k host_wide``): :func:`wide_rows` / :func:`split_order`; :meth:`KVWriteStep.wide_verify` = ``packed_verify``
  with ``partner_of = {16 r + j: 16 r + 8 + j}``; :func:`check_wide_layout`; the gather orders and users per call
  (R-E8: bf16 = 2 x 16 per call kind); T64N §2.4's 2000-step random check (every T64 layout, the calls cover each write
  once, no block twice per call, call A before call B, the host model in both orders); the one-call race emulation; the
  device-input values (split order and the A'' FlashMLA groups); the op wiring and ``flash_groups()`` of
  ``DecodeKVWrite(rows=64)`` on the fake ``ttnn``; its refusals and input copies.
* device, gate G-S1w with the real writer (``-k device_wide``). The split-order gather is WP-D's
  ``ag_dp_rows(halves=2)`` when ``ccl.py`` has it, else a test shim through ``MotifCCL.all_gather``
  (:func:`t64_ccl`). The programs run with an L1 pin page alive (the CB-end check).

  * ``test_kv_write_device_wide_modes`` (G-S1w a): split / natural / ``row_split`` x bfp8 / bf16 x verify / ordinary /
    inactive T64 steps; every chip's cache bit-exact vs the split-order host model. The one-call variant loses
    updates, only at same-tile anchor / draft slots.
  * ``test_kv_write_device_wide_trace``: one capture per T64 writer, replays of different steps bit-exact vs the host
    model and vs eager, no program compiled after the warmup.
  * ``test_kv_write_device_wide_flash`` (G-S1w b): FlashMLA through the writer's inputs, option A (B = 16) and option
    A'' (``flash_groups()``: two B = 8 calls), vs the T32 B = 8 calls on T32 writers' inputs (bitwise) and fp64. The
    probes: row ``p`` never sees ``p + 1``, the draft sees the anchor, window edges ``p - 128`` / ``p - 129``.
  * ``test_kv_write_device_wide_cost``: traced us per layer of the T64 writers next to T32.

Run::

    S=/home/ttuser/hchang/experiments/motif-3/scripts
    $S/hostrun.sh -n kv_write_host -- python -m pytest -p no:cacheprovider -q \
        models/demos/motif3/tests/unit/test_kv_write.py -k host
    $S/devrun.sh -t 1800 -n kv_write -- python -m pytest models/demos/motif3/tests/unit/test_kv_write.py \
        -k device -s -p no:cacheprovider

Every device check prints a ``[kv-write]`` line.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Dict, List

import pytest
import torch

import ttnn
from models.demos.motif3.tt import kv_write as KW
from models.demos.motif3.tt.generator_api import KV_WRITE_MODES, kv_write_mode
from models.demos.motif3.tt.model_config import DEFAULT_HF_META_DIR, MotifTTConfig, device_params

HF_META = str(DEFAULT_HF_META_DIR)
B, LPR, DP, D, BS = 32, 8, 4, 576, 64
MESH = [pytest.param((4, 8), device_params(), id="4x8")]
OFFSETS = (0, 30, 31, 62, 63)  # owner p % 64: same tile as p + 1 (0, 30, 62), tile seam (31), block seam (63)


def log(msg: str) -> None:
    print(f"[kv-write] {msg}", flush=True)


def host_cfg(**kw) -> MotifTTConfig:
    return MotifTTConfig.from_hf_config(HF_META, mesh_shape=(4, 8), **kw)


# ======================================================================================================================
# host layouts
# ======================================================================================================================
# per DP row: (owner slots, plain slots, idle slots); row 0 has more drafts than idle lanes (overflow without KV-R,
# cross-row partners with it), row 3 has idle lanes to spare
ROW_ROLES = {
    0: ((0, 1, 2), (3, 4, 5, 6), (7,)),
    1: ((0, 1), (2, 3), (4, 5, 6, 7)),
    2: ((0,), (1, 2, 3, 4, 5), (6, 7)),
    3: ((), (0, 1), (2, 3, 4, 5, 6, 7)),
}


def lane_layout(seed: int, W: int):
    """``(positions [32], page_table [32, W], has_draft [32])`` in owner-lane order: every lane owns ``W`` shuffled
    blocks (block 0 = null), owners at ``p % 64`` cycling through :data:`OFFSETS` (``p + 1 < 64 W``), plain lanes at
    random positions, idle lanes ``-1``."""
    g = torch.Generator().manual_seed(seed)
    pt = (torch.randperm(B * W, generator=g) + 1).to(torch.int32).reshape(B, W)
    pos = torch.full((B,), -1, dtype=torch.int32)
    draft = torch.zeros(B, dtype=torch.bool)
    k = 0
    for r, (owners, plain, _) in ROW_ROLES.items():
        for s in owners:
            blk = int(torch.randint(0, W - 1, (1,), generator=g))
            pos[LPR * r + s] = BS * blk + OFFSETS[(k + seed) % len(OFFSETS)]
            draft[LPR * r + s] = True
            k += 1
        for s in plain:
            pos[LPR * r + s] = int(torch.randint(0, BS * W, (1,), generator=g))
    return pos, pt, draft


def steps_for(mode: str, seed: int, W: int) -> Dict[str, KW.KVWriteStep]:
    """The step kinds of ``mode``: ordinary and inactive for every mode; packed verify (partners chosen by
    :func:`assign_partner_lanes` under the mode's row rule) and the overflow pass for the split modes."""
    pos, pt, draft = lane_layout(seed, W)
    out = {"ordinary": KW.KVWriteStep.ordinary(pos, pt), "inactive": KW.KVWriteStep.inactive(W)}
    if KW.is_split(mode):
        partner_of, overflow = KW.assign_partner_lanes(pos, draft, cross_row=KW.allows_cross_row_partners(mode))
        out["verify"] = KW.KVWriteStep.packed_verify(pos, pt, partner_of)
        out["overflow"] = KW.KVWriteStep.overflow_pass(pos, pt, overflow or tuple(partner_of)[:2])
    return out


def same_tile(p: int) -> bool:
    return (p + 1) % BS != 0 and (p % BS) // 32 == ((p + 1) % BS) // 32


# ======================================================================================================================
# host tests
# ======================================================================================================================
def test_host_modes_and_lanes_per_call():
    assert KW.SPLIT_MODES == ("row_split", "all_split") and KW.REPLICATED_MODES == ("all", "all_split")
    for rep in (False, True):
        for spec in (False, True):
            m = kv_write_mode(rep, spec)
            assert KW.is_replicated(m) == rep and KW.is_split(m) == spec
            assert KW.allows_cross_row_partners(m) == (rep and spec)
    for m in KV_WRITE_MODES:
        for dt in ("bfp8", "bf16"):
            n = KW.lanes_per_call_for(m, dt)
            assert n == (8 if not KW.is_replicated(m) else {"bfp8": 32, "bf16": 16}[dt])
    assert KW.lanes_per_call_for("all_split", "bfp8", requested=8) == 8
    for bad in (3, 64, 0):
        with pytest.raises(ValueError, match="lanes_per_call"):
            KW.lanes_per_call_for("all", "bfp8", requested=bad)
    with pytest.raises(ValueError, match="lanes_per_call"):
        KW.lanes_per_call_for("all", "bf16", requested=32)  # 32 x 18 x 2048 B output CB: the G12 L1 clash
    with pytest.raises(ValueError, match="per call"):
        KW.lanes_per_call_for("row", "bfp8", requested=16)
    with pytest.raises(ValueError, match="mode"):
        KW.check_mode("deferred")
    with pytest.raises(ValueError, match="dtype"):
        KW.lanes_per_call_for("row", "fp32")
    for rep, spec in ((False, False), (True, True), (True, False), (False, True)):
        cfg = host_cfg(kv_replicated_decode=rep, spec_tokens=int(spec))
        assert cfg.kv_write_mode == kv_write_mode(rep, spec)


def test_host_step_constructors():
    W = 4
    pos, pt, draft = lane_layout(0, W)
    st = KW.KVWriteStep.ordinary(pos, pt)
    assert st.lanes == B and st.width == W and st.num_partners == 0 and int(st.owner.max()) == -1
    assert torch.equal(st.call_positions(""), pos) and torch.equal(st.call_positions("A"), pos)
    assert int(st.call_positions("B").max()) == -1
    ina = KW.KVWriteStep.inactive(W)
    assert int(ina.positions.max()) == -1 and not bool(ina.active.any())
    partner_of = {0: 7, 8: 12, 16: 30}
    v = KW.KVWriteStep.packed_verify(pos, pt, partner_of)
    for o, d in partner_of.items():
        assert int(v.positions[d]) == int(pos[o]) + 1 and torch.equal(v.page_table[d], pt[o])
        assert bool(v.call_b[d]) and int(v.owner[d]) == o and int(v.positions[o]) == int(pos[o])
    assert v.num_partners == 3 and torch.equal(
        v.call_positions("A")[list(partner_of.values())], torch.full((3,), -1, dtype=torch.int32)
    )
    assert torch.equal(pos, lane_layout(0, W)[0]), "packed_verify must not modify its inputs"
    ov = KW.KVWriteStep.overflow_pass(pos, pt, [1, 9])
    assert torch.nonzero(ov.active).reshape(-1).tolist() == [1, 9]
    assert int(ov.positions[1]) == int(pos[1]) + 1 and ov.num_partners == 0
    with pytest.raises(ValueError, match="not idle"):
        KW.KVWriteStep.packed_verify(pos, pt, {0: 3})  # lane 3 is a plain (active) lane
    with pytest.raises(ValueError, match="not idle"):
        KW.KVWriteStep.packed_verify(pos, pt, {0: 7, 1: 7})
    with pytest.raises(ValueError, match="inactive"):
        KW.KVWriteStep.packed_verify(pos, pt, {7: 15})
    with pytest.raises(ValueError, match="inactive"):
        KW.KVWriteStep.overflow_pass(pos, pt, [7])
    with pytest.raises(TypeError):
        KW.KVWriteStep(pos.long(), pt, torch.zeros(B, dtype=torch.bool), torch.full((B,), -1, dtype=torch.int32))
    with pytest.raises(ValueError, match="-1"):
        KW.KVWriteStep.ordinary(torch.full((B,), -2, dtype=torch.int32), pt)


def test_host_assign_partner_lanes():
    g = torch.Generator().manual_seed(7)
    n_cross = n_overflow = 0
    for trial in range(400):
        pos = torch.where(torch.rand(B, generator=g) < 0.4, -1, torch.randint(0, 4000, (B,), generator=g)).to(
            torch.int32
        )
        draft = (pos >= 0) & (torch.rand(B, generator=g) < 0.6)
        idle = {l for l in range(B) if int(pos[l]) < 0}
        owners = [l for l in range(B) if bool(draft[l])]
        for cross in (False, True):
            p_of, ov = KW.assign_partner_lanes(pos, draft, cross_row=cross)
            again = KW.assign_partner_lanes(pos, draft, cross_row=cross)
            assert again == (p_of, ov), "deterministic"
            assert set(p_of) | set(ov) == set(owners) and not (set(p_of) & set(ov))
            assert len(set(p_of.values())) == len(p_of) and set(p_of.values()) <= idle
            if not cross:
                assert all(d // LPR == o // LPR for o, d in p_of.items())
                for r in range(DP):  # per-row maximum matching
                    n_own = sum(1 for o in owners if o // LPR == r)
                    n_idle = sum(1 for l in idle if l // LPR == r)
                    assert sum(1 for o in p_of if o // LPR == r) == min(n_own, n_idle)
            else:
                assert len(p_of) == min(len(owners), len(idle))
                used = set(p_of.values())
                for o, d in p_of.items():  # same row first: a cross-row partner only when the owner's row is full
                    if d // LPR != o // LPR:
                        n_cross += 1
                        assert not any(l // LPR == o // LPR for l in idle - used)
            n_overflow += len(ov)
            st = KW.KVWriteStep.packed_verify(pos, torch.arange(B * 8, dtype=torch.int32).reshape(B, 8) + 1, p_of)
            assert st.num_partners == len(p_of)
    assert n_cross > 50 and n_overflow > 50, (n_cross, n_overflow)
    with pytest.raises(ValueError, match="inactive"):
        KW.assign_partner_lanes(
            torch.full((B,), -1, dtype=torch.int32), torch.ones(B, dtype=torch.bool), cross_row=True
        )


def test_host_check_accepts_every_valid_step():
    n = 0
    for seed in range(40):
        for W in (2, 4, 8):
            for mode in KV_WRITE_MODES:
                for kind, st in steps_for(mode, seed, W).items():
                    for lpc in ((None,) if not KW.is_replicated(mode) else (None, 16, 8)):
                        KW.check_kv_write_step(st, mode, block_size=BS, max_seq_len=32768, lanes_per_call=lpc)
                        n += 1
    assert n > 1000
    # owners at every offset class appear with their partners in the same call-B
    st = steps_for("all_split", 3, 4)["verify"]
    pairs = [(int(st.owner[d]), d) for d in torch.nonzero(st.call_b).reshape(-1).tolist()]
    assert any(same_tile(int(st.positions[o])) for o, _ in pairs) and any(d // LPR != o // LPR for o, d in pairs)


def _refuse(st, mode, match, **kw):
    with pytest.raises(ValueError, match=match):
        KW.check_kv_write_step(st, mode, block_size=BS, **kw)


def test_host_check_refusals():
    W = 4
    pos, pt, draft = lane_layout(1, W)
    p_same, _ = KW.assign_partner_lanes(pos, draft, cross_row=False)
    p_cross, _ = KW.assign_partner_lanes(pos, draft, cross_row=True)
    v_same = KW.KVWriteStep.packed_verify(pos, pt, p_same)
    v_cross = KW.KVWriteStep.packed_verify(pos, pt, p_cross)
    for mode in ("row", "all"):  # no call B: one call would race owner and partner (G12)
        _refuse(v_same, mode, "no call B")
    _refuse(v_cross, "row_split", "own DP row|its owner's row")
    KW.check_kv_write_step(v_cross, "all_split", block_size=BS)
    KW.check_kv_write_step(v_same, "row_split", block_size=BS)

    def edit(st, **ch):
        f = {k: getattr(st, k).clone() for k in ("positions", "page_table", "call_b", "owner")}
        for k, fn in ch.items():
            fn(f[k])
        return KW.KVWriteStep(**f)

    d, o = next((int(x), int(v_same.owner[x])) for x in torch.nonzero(v_same.call_b).reshape(-1))
    _refuse(edit(v_same, positions=lambda p: p.__setitem__(d, p[o] + 2)), "row_split", r"\+ 1")
    _refuse(
        edit(v_same, page_table=lambda t: t.__setitem__((d, W - 1), t[d, W - 1] + 1000)), "row_split", "page-table row"
    )
    _refuse(edit(v_same, owner=lambda t: t.__setitem__(d, d)), "row_split", "has owner")
    assert int(v_same.positions[31]) == -1  # row 3 has no owners: its idle lanes stay idle
    _refuse(edit(v_same, owner=lambda t: t.__setitem__(d, 31)), "row_split", "active call-A")  # an idle lane as owner
    d_other = next(int(x) for x in torch.nonzero(v_same.call_b).reshape(-1) if int(x) != d)
    _refuse(edit(v_same, owner=lambda t: t.__setitem__(d, d_other)), "row_split", "active call-A")  # a partner as owner
    _refuse(edit(v_same, owner=lambda t: t.__setitem__(3, 0)), "row_split", "not a call-B")
    d2 = next(int(x) for x in torch.nonzero(v_same.call_b).reshape(-1) if int(x) != d)
    two = edit(v_same, owner=lambda t: t.__setitem__(d2, o), positions=lambda p: p.__setitem__(d2, p[o] + 1),
               page_table=lambda t: t.__setitem__(d2, t[o].clone()))  # fmt: skip
    _refuse(two, "all_split", "two partners|two rows|both write")
    st = KW.KVWriteStep.ordinary(pos, pt)
    _refuse(edit(st, page_table=lambda t: t.__setitem__((0, int(pos[0]) // BS), 0)), "row", "null block")
    _refuse(edit(st, positions=lambda p: p.__setitem__(3, BS * W)), "row", "width")
    _refuse(st, "row", "max_seq_len", max_seq_len=int(pos.max()))
    # two lanes writing one block (two rows of one request in one call) / the same slot. Lanes 3, 4: plain lanes of DP
    # row 0; lane 12: an idle lane of row 1. q = another row of lane 3's block.
    p3 = int(pos[3])
    q = p3 + 1 if p3 % BS != BS - 1 else p3 - 1

    def twin(lane, at):
        return edit(
            st, page_table=lambda t: t.__setitem__(lane, t[3].clone()), positions=lambda p: p.__setitem__(lane, at)
        )

    _refuse(twin(4, q), "row", "must not carry two rows")
    _refuse(twin(4, q), "all", "must not carry two rows")
    _refuse(twin(4, p3), "row", "both write")
    # across DP rows, one block in one KV-R call is refused too (every chip writes both lanes)
    _refuse(twin(12, q), "all", "must not carry two rows")
    KW.check_kv_write_step(twin(12, q), "row", block_size=BS)  # row mode: different chips, different cache copies
    _refuse(twin(12, p3), "row", "both write")  # the same slot is refused in every mode
    # a partner lane must be active / owner must be active
    _refuse(edit(v_same, positions=lambda p: p.__setitem__(d, -1)), "row_split", "inactive")


def test_host_calls_cover_writes_and_are_race_free():
    for seed in range(30):
        for mode in KV_WRITE_MODES:
            for kind, st in steps_for(mode, seed, 4).items():
                for lpc in ((None,) if not KW.is_replicated(mode) else (None, 16, 8)):
                    calls = KW.kv_write_calls(st, mode, lanes_per_call=lpc)
                    per_row = {r: [] for r in range(DP)}
                    for c in calls:
                        users = [(l, p) for l, p in zip(c.lanes, c.positions.tolist()) if p >= 0]
                        tiles = [(int(st.page_table[l, p // BS]), (p % BS) // 32) for l, p in users]
                        assert len(set(tiles)) == len(tiles), (mode, kind, c)
                        assert c.kind in (KW.CALL_KINDS if KW.is_split(mode) else ("",))
                        if KW.is_replicated(mode):
                            assert c.rows == tuple(range(DP))
                        else:
                            assert len(c.rows) == 1 and all(l // LPR == c.rows[0] for l in c.lanes)
                        for r in c.rows:
                            per_row[r] += users
                    want = [(l, int(st.positions[l])) for l in range(B) if int(st.positions[l]) >= 0]
                    for r in range(DP):
                        exp = want if KW.is_replicated(mode) else [w for w in want if w[0] // LPR == r]
                        assert sorted(per_row[r]) == sorted(exp), (mode, kind, r)
                    if KW.is_split(mode):  # call A precedes call B within each row / chunk
                        kinds = [c.kind for c in calls]
                        assert kinds == ["A", "B"] * (len(calls) // 2)
                        for c in calls:
                            for l, p in zip(c.lanes, c.positions.tolist()):
                                if p >= 0:
                                    assert bool(st.call_b[l]) == (c.kind == "B")


def test_host_model():
    W = 4
    g = torch.Generator().manual_seed(5)
    base = torch.randn(B * W + 4, 1, BS, D, generator=g)
    rows = torch.randn(B, D, generator=g)
    for mode in KV_WRITE_MODES:
        for kind, st in steps_for(mode, 11, W).items():
            outs = KW.apply_kv_writes_host(base, st, mode, rows, block_size=BS)
            base_copy = base.clone()
            assert len(outs) == DP and torch.equal(base, base_copy)
            for r in range(DP):
                diff = {(b, i) for b, i in torch.nonzero((outs[r] != base).any(-1)[:, 0]).tolist()}
                lanes = [l for l in range(B) if int(st.positions[l]) >= 0 and (KW.is_replicated(mode) or l // LPR == r)]
                want = {(int(st.page_table[l, int(st.positions[l]) // BS]), int(st.positions[l]) % BS) for l in lanes}
                assert diff == want, (mode, kind, r)
                for l in lanes:
                    p = int(st.positions[l])
                    assert torch.equal(outs[r][int(st.page_table[l, p // BS]), 0, p % BS], rows[l])
            if KW.is_replicated(mode):
                assert all(torch.equal(outs[0], o) for o in outs)
                for lpc in (16, 8):  # chunking never changes what lands where
                    alt = KW.apply_kv_writes_host(base, st, mode, rows, block_size=BS, lanes_per_call=lpc)
                    assert all(torch.equal(a, o) for a, o in zip(alt, outs))
            elif kind != "inactive":
                assert not torch.equal(outs[0], outs[1])
    with pytest.raises(ValueError, match="one cache copy per DP row"):
        KW.apply_kv_writes_host([base], steps_for("row", 0, W)["ordinary"], "row", rows, block_size=BS)


def emulate_racy_call(
    cache: torch.Tensor, st: KW.KVWriteStep, users: List[int], rows: torch.Tensor, order
) -> torch.Tensor:
    """Emulation of one ``paged_update_cache`` call (writer_update_cache_interleaved_start_id.cpp): every user reads its
    32-row tile at the start of the call (all cores run in parallel), replaces its row and writes the whole tile back;
    tiles written later (``order``) win. Two users of one tile in one call lose one update."""
    snap = {}
    for l in users:
        p = int(st.positions[l])
        b, t = int(st.page_table[l, p // BS]), (p % BS) // 32
        snap[l] = (b, t, cache[b, 0, 32 * t : 32 * t + 32].clone())
    for l in order(users):
        b, t, tile = snap[l]
        tile[int(st.positions[l]) % 32] = rows[l]
        cache[b, 0, 32 * t : 32 * t + 32] = tile
    return cache


def test_host_race_emulation_needs_the_split():
    """G12's device finding on the host model: owners and partners in ONE call lose updates exactly on same-tile pairs;
    call A then call B (the split modes) never lose one, whatever the core order."""
    W = 4
    g = torch.Generator().manual_seed(9)
    base = torch.randn(B * W + 4, 1, BS, D, generator=g)
    rows = torch.randn(B, D, generator=g)
    lost_total = 0
    for seed in range(20):
        pos, pt, draft = lane_layout(seed, W)
        p_of, _ = KW.assign_partner_lanes(pos, draft, cross_row=True)
        st = KW.KVWriteStep.packed_verify(pos, pt, p_of)
        want = KW.apply_kv_writes_host(base, st, "all_split", rows, block_size=BS)[0]
        act = [l for l in range(B) if int(st.positions[l]) >= 0]
        pairs = [(int(st.owner[d]), d) for d in torch.nonzero(st.call_b).reshape(-1).tolist()]
        for forward in (True, False):
            order = (lambda u: list(u)) if forward else (lambda u: list(u)[::-1])
            one = emulate_racy_call(base.clone(), st, act, rows, order)
            lost = {l for l in act if not torch.equal(one[int(st.page_table[l, int(st.positions[l]) // BS]), 0,
                                                          int(st.positions[l]) % BS], rows[l])}  # fmt: skip
            # the earlier writer of each same-tile owner / partner pair is overwritten by the later one's stale copy
            exp_lost = {min(o, d) if forward else max(o, d) for o, d in pairs if same_tile(int(st.positions[o]))}
            assert lost == exp_lost, (seed, forward, lost, exp_lost)
            lost_total += len(lost)
            split = base.clone()
            for kind in ("A", "B"):
                users = [l for l in act if bool(st.call_b[l]) == (kind == "B")]
                split = emulate_racy_call(split, st, users, rows, order)
            assert torch.equal(split, want)
    assert lost_total > 20


def test_host_input_values():
    W = 4
    for mode in KV_WRITE_MODES:
        for kind, st in steps_for(mode, 2, W).items():
            for lpc in ((None,) if not KW.is_replicated(mode) else (None, 16)):
                v = KW.kv_write_inputs_host(st, mode, lanes_per_call=lpc)
                assert torch.equal(v["cur_pos"][1], st.positions) and v["cur_pos"][0] == "row"
                pt = v["page_table"][1]
                assert torch.equal(pt[st.active], st.page_table[st.active]) and int(pt[~st.active].abs().sum()) == 0
                names = {"row": {"cur_pos", "page_table"}, "row_split": {"cur_pos", "page_table", "cur_a", "cur_b"}}
                if not KW.is_replicated(mode):
                    assert set(v) == names[mode]
                    if mode == "row_split":
                        a, b = v["cur_a"][1], v["cur_b"][1]
                        assert torch.equal(torch.maximum(a, b), st.positions) and int(torch.minimum(a, b).max()) == -1
                    continue
                n = B if lpc is None else lpc
                for c in range(B // n):
                    sl = slice(c * n, (c + 1) * n)
                    assert v[f"pt{c}"][0] == "rep" and torch.equal(v[f"pt{c}"][1], pt[sl])
                    if mode == "all":
                        assert torch.equal(v[f"cur{c}"][1], st.positions[sl])
                    else:
                        assert torch.equal(v[f"cur_a{c}"][1], st.call_positions("A")[sl])
                        assert torch.equal(v[f"cur_b{c}"][1], st.call_positions("B")[sl])
                assert len(v) == 2 + (B // n) * (2 if mode == "all" else 3)


# ---- device op wiring on the host (recording fake ttnn) --------------------------------------------------------------
class _T:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class FakeTTNN:
    """Records the ops :class:`DecodeKVWrite` issues (names of the tensors / memory configs involved)."""

    DRAM_MEMORY_CONFIG, ROW_MAJOR_LAYOUT, TILE_LAYOUT = "DRAM", "RM", "TILE"
    int32, bfloat16 = "i32", "bf16"
    ShardStrategy = SimpleNamespace(HEIGHT="HEIGHT")
    ShardOrientation = SimpleNamespace(ROW_MAJOR="ROW_MAJOR")

    def __init__(self):
        self.calls, self.copies, self.n = [], [], 0
        self.experimental = SimpleNamespace(paged_update_cache=self._update)

    def _new(self, op):
        self.n += 1
        return _T(f"{op}{self.n}")

    def num_cores_to_corerangeset(self, n, grid, row_wise=True):
        assert row_wise and grid == "grid"
        return n

    def create_sharded_memory_config(
        self, shape, core_grid, strategy, orientation, use_height_and_width_as_shard_shape
    ):
        assert tuple(shape) == (32, D) and strategy == "HEIGHT" and orientation == "ROW_MAJOR"
        return f"mc{core_grid}"

    def from_torch(self, t, dtype=None, layout=None, mesh_mapper=None):
        assert dtype == "i32" and layout == "RM" and mesh_mapper == "replicate"
        return ("rep", t.clone())

    def ReplicateTensorToMesh(self, mesh):
        return "replicate"

    def to_device(self, host, mesh, memory_config=None):
        t = self._new("in")
        t.place, t.value = host
        return t

    def copy_host_to_device_tensor(self, host, dev):
        assert host[0] == dev.place
        self.copies.append(dev.name)
        dev.value = host[1]

    def transpose(self, t, a, b, memory_config=None):
        self.calls.append(("transpose", t, a, b, memory_config))
        return self._new("tr")

    def slice(self, t, s, e):
        self.calls.append(("slice", t, tuple(s), tuple(e)))
        return self._new("sl")

    def to_memory_config(self, t, mc):
        self.calls.append(("to_memory_config", t, mc))
        return self._new("sh")

    def deallocate(self, t):
        self.calls.append(("deallocate", t))

    def _update(self, cache, u, update_idxs_tensor=None, page_table=None):
        self.calls.append(("update", cache, u, update_idxs_tensor, page_table))


def describe_calls(calls, inputs: Dict[str, "_T"]) -> List[str]:
    """Recorded calls as strings: persistent inputs by role (``cur_pos``, ``cur_a0``, ``pt1``, ...), op outputs by the
    op that made them (``tr``, ``ag``, ``sl``, ``sh``), everything else as is."""
    role = {id(t): n for n, t in inputs.items()}
    out = []
    for c in calls:
        parts = []
        for x in c:
            if isinstance(x, _T):
                parts.append(role.get(id(x)) or x.name.rstrip("0123456789"))
            else:
                parts.append(str(x))
        out.append(" ".join(parts))
    return out


def fake_kv_write(monkeypatch, mode: str, kv_name: str = "bfp8", W: int = 4, lanes_per_call=None, **kw):
    """A :class:`DecodeKVWrite` on the recording fake ``ttnn`` (``kw``: ``rows`` / ``gather`` of a T64 writer)."""
    fake = FakeTTNN()
    monkeypatch.setattr(KW, "ttnn", fake)
    monkeypatch.setattr(KW, "shard_lanes", lambda v, cfg, mesh, dtype=None, device=None: ("row", v.clone()))

    def ag_dp_rows(x, halves=1):  # ccl.ag_dp_rows(x, halves=1 or 2): WP-D's split-order gather (D1)
        fake.calls.append(("ag_dp_rows", x) if halves == 1 else ("ag_dp_rows", x, f"halves={halves}"))
        return fake._new("ag")

    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: "grid")
    ccl = SimpleNamespace(ag_dp_rows=ag_dp_rows)
    cfg = host_cfg(kv_cache_dtype=kv_name)
    kvw = KW.DecodeKVWrite(mesh, cfg, ccl=ccl, page_table_width=W, mode=mode, lanes_per_call=lanes_per_call, **kw)
    return fake, kvw


def test_host_write_op_wiring(monkeypatch):
    def updates(u, *pairs):
        return [f"update cache {u} {cur} {pt}" for cur, pt in pairs]

    def chunk(c, n, curs):  # one lane chunk of a chunked KV-R write: slice, shard, its call(s), free
        lo, hi = c * n, (c + 1) * n
        return (
            [f"slice tr (0, {lo}, 0, 0) (1, {hi}, 1, 576)", f"to_memory_config sl mc{n}", "deallocate sl"]
            + updates("sh", *[(f"{cur}{c}", f"pt{c}") for cur in curs])
            + ["deallocate sh"]
        )

    g32 = ["ag_dp_rows kv", "transpose ag 1 2 mc32", "deallocate ag"]
    g_dram = ["ag_dp_rows kv", "transpose ag 1 2 DRAM", "deallocate ag"]
    ab = ["cur_a", "cur_b"]
    expected = {
        ("row", "bfp8", None): ["transpose kv 1 2 mc8"] + updates("tr", ("cur_pos", "page_table")) + ["deallocate tr"],
        ("row_split", "bfp8", None): ["transpose kv 1 2 mc8"]
        + updates("tr", ("cur_a", "page_table"), ("cur_b", "page_table"))
        + ["deallocate tr"],
        ("all", "bfp8", None): g32 + updates("tr", ("cur0", "pt0")) + ["deallocate tr"],
        ("all_split", "bfp8", None): g32 + updates("tr", ("cur_a0", "pt0"), ("cur_b0", "pt0")) + ["deallocate tr"],
        ("all", "bf16", None): g_dram + chunk(0, 16, ["cur"]) + chunk(1, 16, ["cur"]) + ["deallocate tr"],
        ("all_split", "bf16", None): g_dram + chunk(0, 16, ab) + chunk(1, 16, ab) + ["deallocate tr"],
        ("all_split", "bfp8", 8): g_dram + sum((chunk(c, 8, ab) for c in range(4)), []) + ["deallocate tr"],
    }
    for (mode, kv_name, lpc), want in expected.items():
        fake, kvw = fake_kv_write(monkeypatch, mode, kv_name, lanes_per_call=lpc)
        assert kvw.cur_pos is kvw.device_inputs["cur_pos"] and kvw.page_table is kvw.device_inputs["page_table"]
        assert kvw.calls_per_layer == len([w for w in want if w.startswith("update")])
        kv, cache = _T("kv"), _T("cache")
        fake.calls.clear()
        kvw.write(kv, cache, cur_pos=kvw.cur_pos, page_table=kvw.page_table)
        got = describe_calls(fake.calls, kvw.device_inputs)
        assert got == want, (mode, kv_name, lpc, got)
        assert "deallocate kv" not in got, "kv_row belongs to the caller (DecodeKVWriter protocol)"
        # every op output is freed inside the call (nothing but the in-place cache update survives a layer)
        n_out = sum(c[0] in ("transpose", "slice", "to_memory_config", "ag_dp_rows") for c in fake.calls)
        assert sum(c[0] == "deallocate" for c in fake.calls) == n_out
        with pytest.raises(ValueError, match="kv_write.cur_pos"):
            kvw.write(kv, cache, cur_pos=_T("other"), page_table=kvw.page_table)
        kvw.end_step()  # no-op hook: issues nothing
        assert describe_calls(fake.calls, kvw.device_inputs) == want


def test_host_write_step_copies(monkeypatch):
    W = 4
    fake, kvw = fake_kv_write(monkeypatch, "all_split", "bfp8", W)
    steps = steps_for("all_split", 4, W)
    assert kvw.write_step(steps["inactive"]) == 0  # the construction wrote the inactive step
    n = kvw.write_step(steps["ordinary"])
    assert n == 4 and set(fake.copies) == {
        kvw.device_inputs[k].name for k in ("cur_pos", "page_table", "cur_a0", "pt0")
    }
    fake.copies.clear()
    assert kvw.write_step(steps["ordinary"]) == 0 and kvw.write_step(steps["ordinary"], force=True) == 5
    fake.copies.clear()
    n = kvw.write_step(
        steps["verify"]
    )  # partners: cur_pos, page_table, cur_b, pt; cur_a unchanged (partners were idle)
    assert {k for k, t in kvw.device_inputs.items() if t.name in fake.copies} == {
        "cur_pos",
        "page_table",
        "cur_b0",
        "pt0",
    }
    for name, (place, val) in KW.kv_write_inputs_host(steps["verify"], "all_split").items():
        assert torch.equal(kvw.device_inputs[name].value, val)
    with pytest.raises(ValueError, match="width"):
        kvw.write_step(KW.KVWriteStep.inactive(W + 1))
    pos, pt, draft = lane_layout(4, W)
    p_of, _ = KW.assign_partner_lanes(pos, draft, cross_row=True)
    bad = KW.KVWriteStep.packed_verify(pos, pt, p_of)
    fake2, kvw2 = fake_kv_write(monkeypatch, "row_split", "bfp8", W)
    with pytest.raises(ValueError, match="owner's row"):
        kvw2.write_step(bad)
    assert kvw2.write_step(bad, validate=False) > 0  # the negative control of the device test forces it


# ======================================================================================================================
# T64 host tests: 64 rows, per DP row [8 anchors | 8 drafts] (docs/p5_t64/P5_T64_DESIGN.md §4.1, §4.3, §7.1)
# ======================================================================================================================
WB, WLPR = 2 * B, 2 * LPR  # T64 rows, rows per DP row
# (mode, gather, lanes per KV-R call): every T64 update-call layout (split order: 2 x 32 bfp8, 4 x 16 bf16 (R-E8),
# 8 x 8; natural order: 4 x 32 bfp8, 8 x 16 bf16; row_split: per DP row 2 x 16)
WIDE_LAYOUTS = (
    ("row_split", "natural", None),
    ("all_split", "split", 32),
    ("all_split", "split", 16),
    ("all_split", "split", 8),
    ("all_split", "natural", 32),
    ("all_split", "natural", 16),
)


def wide_lanes(seed: int, W: int, *, p_idle: float = 0.15, p_draft: float = 0.85):
    """An owner-lane T64 batch ``(positions [32], page_table [32, W], has_draft [32])``: every lane owns ``W`` shuffled
    blocks (block 0 = null), idle lanes ``-1``; the anchors of about half the lanes sit at ``p % 64`` in
    :data:`OFFSETS` (the draft at ``p + 1`` shares the anchor's tile at 0 / 30 / 62, crosses a tile at 31 and a block
    at 63), the rest at random positions; ``p + 1 < 64 W``."""
    g = torch.Generator().manual_seed(seed)
    pt = (torch.randperm(B * W, generator=g) + 1).to(torch.int32).reshape(B, W)
    pos = torch.full((B,), -1, dtype=torch.int32)
    draft = torch.zeros(B, dtype=torch.bool)
    for lane in range(B):
        if float(torch.rand(1, generator=g)) < p_idle:
            continue
        if lane % 2:
            blk = int(torch.randint(0, W - 1, (1,), generator=g))
            pos[lane] = BS * blk + OFFSETS[(lane // 2 + seed) % len(OFFSETS)]
        else:
            pos[lane] = int(torch.randint(0, BS * W - 1, (1,), generator=g))
        draft[lane] = float(torch.rand(1, generator=g)) < p_draft
    return pos, pt, draft


def wide_physical(pos: torch.Tensor, pt: torch.Tensor, draft: torch.Tensor):
    """The design's definition of a T64 step, by hand: rows ``16 r + j`` = lane ``8 r + j`` at ``n`` with its page-table
    row; ``partner_of = {16 r + j: 16 r + 8 + j}`` for the drafted lanes (``packed_verify`` puts each draft at ``n +
    1`` with the owner's row)."""
    pos64 = torch.full((WB,), -1, dtype=torch.int32)
    pt64 = torch.zeros(WB, pt.shape[1], dtype=torch.int32)
    partner_of = {}
    for lane in range(B):
        r, j = divmod(lane, LPR)
        pos64[WLPR * r + j] = pos[lane]
        pt64[WLPR * r + j] = pt[lane]
        if bool(draft[lane]):
            partner_of[WLPR * r + j] = WLPR * r + LPR + j
    return pos64, pt64, partner_of


def test_host_wide_rows_and_split_order():
    for lane in range(B):
        r, j = divmod(lane, LPR)
        assert KW.wide_rows(lane) == (16 * r + j, 16 * r + 8 + j)
    order = KW.split_order(WB, WLPR)
    assert sorted(order) == list(range(WB)), "a permutation of the 64 rows"
    # users 0..31 = every DP row's anchors in T32 lane order 8 dp + l; users 32..63 = the drafts in the same order
    assert all(order[lane] == KW.wide_rows(lane)[0] and order[B + lane] == KW.wide_rows(lane)[1] for lane in range(B))
    # the device's split-order gather: view [1, 2, 8, W] per DP row, gather dim 2 over DP -> [1, 2, 32, W] -> 64 rows
    rows = torch.arange(WB).reshape(DP, 2, LPR)  # rows[r, h, j] = physical row 16 r + 8 h + j
    gathered = torch.cat([rows[r] for r in range(DP)], dim=1)  # [2, 32]: dim 1 is the gathered (DP-major) axis
    assert gathered.reshape(-1).tolist() == list(order)
    assert KW.split_order(8, 2) == (0, 2, 4, 6, 1, 3, 5, 7)
    for bad in ((64, 7), (64, 0), (60, 16)):
        with pytest.raises(ValueError, match="even number"):
            KW.split_order(*bad)


def test_host_wide_verify_step():
    W = 8
    n_same_tile = 0
    for seed in range(60):
        pos, pt, draft = wide_lanes(seed, W)
        args = [t.clone() for t in (pos, pt, draft)]
        st = KW.KVWriteStep.wide_verify(pos, pt, draft)
        assert all(torch.equal(a, b) for a, b in zip(args, (pos, pt, draft))), "wide_verify must not modify its inputs"
        # exactly the design's definition: packed_verify with partner_of = {16 r + j: 16 r + 8 + j} on the 64 rows
        ref = KW.KVWriteStep.packed_verify(*wide_physical(pos, pt, draft))
        for name in ("positions", "page_table", "call_b", "owner"):
            assert torch.equal(getattr(st, name), getattr(ref, name)), (seed, name)
        assert st.lanes == WB and st.width == W and st.num_partners == int(draft.sum())
        for lane in range(B):
            ra, rd = KW.wide_rows(lane)
            assert int(st.positions[ra]) == int(pos[lane]) and torch.equal(st.page_table[ra], pt[lane])
            if bool(draft[lane]):
                assert int(st.positions[rd]) == int(pos[lane]) + 1 and torch.equal(st.page_table[rd], pt[lane])
                assert bool(st.call_b[rd]) and int(st.owner[rd]) == ra
                n_same_tile += same_tile(int(pos[lane]))
            else:
                assert int(st.positions[rd]) == -1 and int(st.page_table[rd].abs().sum()) == 0
        KW.check_wide_layout(st)
    assert n_same_tile > 50
    # no drafts: the wide mode's ordinary step (anchors only, draft rows idle)
    pos, pt, _ = wide_lanes(1, W)
    for st in (KW.KVWriteStep.wide_verify(pos, pt), KW.KVWriteStep.wide_verify(pos, pt, torch.zeros(B, dtype=bool))):
        assert st.num_partners == 0 and int(st.positions[[KW.wide_rows(x)[1] for x in range(B)]].max()) == -1
    ina = KW.KVWriteStep.inactive(W, lanes=WB)
    KW.check_wide_layout(ina)
    with pytest.raises(ValueError, match="inactive lane"):
        KW.KVWriteStep.wide_verify(torch.full((B,), -1, dtype=torch.int32), pt, torch.ones(B, dtype=torch.bool))
    with pytest.raises(ValueError, match="has_draft"):
        KW.KVWriteStep.wide_verify(pos, pt, torch.ones(B - 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="page_table"):
        KW.KVWriteStep.wide_verify(pos, pt[:-1])
    with pytest.raises(ValueError, match="DP rows"):
        KW.KVWriteStep.wide_verify(pos[:-1], pt[:-1])


def test_host_wide_layout_refusals():
    W = 4
    pos, pt, draft = wide_lanes(3, W, p_idle=0.0, p_draft=1.0)
    st = KW.KVWriteStep.wide_verify(pos, pt, draft)
    KW.check_wide_layout(st)
    p64, pt64, partner_of = wide_physical(pos, pt, draft)
    # a draft in another draft slot of its DP row: valid for packed verify (row_split), not for the T64 layout
    swapped = dict(partner_of)
    swapped[0], swapped[1] = partner_of[1], partner_of[0]
    st_sw = KW.KVWriteStep.packed_verify(p64, pt64, swapped)
    KW.check_kv_write_step(st_sw, "row_split", block_size=BS, lanes_per_row=WLPR)
    with pytest.raises(ValueError, match="draft of anchor row"):
        KW.check_wide_layout(st_sw)
    with pytest.raises(ValueError, match="draft of anchor row"):  # the split gather would mismatch the page tables
        KW.check_kv_write_step(st_sw, "all_split", block_size=BS, lanes_per_row=WLPR, gather="split")
    # a draft in an anchor slot: lane 0 idle, its anchor row 0 takes the draft of anchor row 1
    p2 = p64.clone()
    p2[[0, 8]] = -1
    st_a = KW.KVWriteStep.packed_verify(p2, pt64, {1: 0})
    with pytest.raises(ValueError, match="anchor slot"):
        KW.check_wide_layout(st_a)
    # an active non-draft row in a draft slot (a plain lane in the drafts' half)
    p3 = p64.clone()
    p3[8] = 5  # row 8 = DP row 0's first draft slot
    st_p = KW.KVWriteStep.ordinary(p3, pt64)
    with pytest.raises(ValueError, match="draft slot"):
        KW.check_wide_layout(st_p)
    for per in (15, 0, 6):
        with pytest.raises(ValueError, match="even number"):
            KW.check_wide_layout(st, lanes_per_row=per)


def test_host_wide_gather_modes_and_lanes_per_call():
    assert KW.GATHER_ORDERS == ("natural", "split")
    assert KW.gathers_split("all_split", "split") and not KW.gathers_split("all_split", "natural")
    for m in ("row", "row_split"):  # the row modes have no gather
        assert not KW.gathers_split(m, "split")
    with pytest.raises(ValueError, match="no call B"):
        KW.gathers_split("all", "split")
    with pytest.raises(ValueError, match="gather order"):
        KW.check_gather("interleaved")
    kw = dict(lanes=WB, lanes_per_row=WLPR)
    assert KW.lanes_per_call_for("all_split", "bfp8", gather="split", **kw) == 32
    assert KW.lanes_per_call_for("all_split", "bf16", gather="split", **kw) == 16  # R-E8: 2 x 16 per call kind
    assert KW.lanes_per_call_for("all_split", "bfp8", gather="natural", **kw) == 32
    assert KW.lanes_per_call_for("all_split", "bf16", gather="natural", **kw) == 16
    assert KW.lanes_per_call_for("row_split", "bfp8", gather="split", **kw) == 16
    assert KW.lanes_per_call_for("all_split", "bfp8", gather="split", requested=8, **kw) == 8
    for bad, dt in ((64, "bfp8"), (12, "bfp8"), (32, "bf16")):
        with pytest.raises(ValueError, match="lanes_per_call"):
            KW.lanes_per_call_for("all_split", dt, gather="split", requested=bad, **kw)
    with pytest.raises(ValueError, match="lanes_per_call"):  # natural: at most 32 users per bfp8 call
        KW.lanes_per_call_for("all_split", "bfp8", gather="natural", requested=64, **kw)
    with pytest.raises(ValueError, match="per call"):
        KW.lanes_per_call_for("row_split", "bfp8", requested=8, **kw)
    st = KW.KVWriteStep.inactive(4, lanes=WB)
    with pytest.raises(ValueError, match="lanes_per_call"):  # the split host plan: a divisor of the 32 anchors
        KW.kv_write_calls(st, "all_split", lanes_per_call=64, lanes_per_row=WLPR, gather="split")
    with pytest.raises(ValueError, match="no call B"):
        KW.kv_write_calls(st, "all", lanes_per_row=WLPR, gather="split")


def _call_checks(st, mode, gather, lpc, calls):
    """Every write of ``st`` lands exactly once on each DP row's copy that must hold it, every call writes a block at
    most once (G12), the users per call and the call kinds match the layout, and each lane's anchor call precedes its
    draft call."""
    split_g = gather == "split" and KW.is_replicated(mode)
    n = (lpc or (WB // 2 if split_g else WB)) if KW.is_replicated(mode) else WLPR
    pt_l, cb_l, pos_l = st.page_table.tolist(), st.call_b.tolist(), st.positions.tolist()
    per_row = {r: [] for r in range(DP)}
    first = {}
    for i, c in enumerate(calls):
        assert len(c.lanes) == n, (mode, gather, lpc, c)
        users = [(l, p) for l, p in zip(c.lanes, c.positions.tolist()) if p >= 0]
        blocks = [pt_l[l][p // BS] for l, p in users]
        assert len(set(blocks)) == len(blocks), ("G12: a block twice in one call", mode, gather, c)
        for l, p in users:
            assert cb_l[l] == (c.kind == "B") and p == pos_l[l]
            first.setdefault(l, i)
        if split_g:  # anchor halves in calls A, draft halves in calls B, in DP-major lane order
            half = [(l % WLPR) >= LPR for l in c.lanes]
            assert all(h == (c.kind == "B") for h in half)
        for r in c.rows:
            per_row[r] += users
    want = [(l, p) for l, p in enumerate(pos_l) if p >= 0]
    for r in range(DP):
        exp = want if KW.is_replicated(mode) else [w for w in want if w[0] // WLPR == r]
        assert sorted(per_row[r]) == sorted(exp), (mode, gather, lpc, r)
    for d, o in enumerate(st.owner.tolist()):
        if o >= 0:
            assert first[o] < first[d], "the anchor's call A precedes its draft's call B"


def test_host_wide_random_steps():
    """T64N §2.4's host check, as a test: 2000 random T64 steps (idle lanes, undrafted lanes, anchors at the tile and
    block seams, so same-tile ``p`` / ``p + 1`` pairs) under every T64 layout pass :func:`check_kv_write_step` at 16
    rows per DP row; the update calls cover every write once, never write a block twice in one call, and order each
    lane's call A before its call B. On every 10th step the host model holds every anchor at ``n`` and every draft at
    ``n + 1`` on the chips that must hold them, identically for the split and the natural order."""
    W, D_ = 16, 8
    base = torch.randn(B * W + 1, 1, BS, D_, generator=torch.Generator().manual_seed(77))
    n_pairs = n_same = 0
    for trial in range(2000):
        pos, pt, draft = wide_lanes(1000 + trial, W, p_idle=0.11, p_draft=0.9)
        st = KW.KVWriteStep.wide_verify(pos, pt, draft)
        n_pairs += st.num_partners
        n_same += sum(same_tile(int(pos[x])) for x in range(B) if bool(draft[x]))
        for mode, gather, lpc in WIDE_LAYOUTS:
            KW.check_kv_write_step(
                st, mode, block_size=BS, max_seq_len=W * BS, lanes_per_call=lpc, lanes_per_row=WLPR, gather=gather
            )
            calls = KW.kv_write_calls(st, mode, lanes_per_call=lpc, lanes_per_row=WLPR, gather=gather)
            _call_checks(st, mode, gather, lpc, calls)
        if trial % 10:
            continue
        rows = torch.randn(WB, D_, generator=torch.Generator().manual_seed(trial))
        outs = {}
        for mode, gather, lpc in WIDE_LAYOUTS:
            outs[(mode, gather, lpc)] = KW.apply_kv_writes_host(
                base, st, mode, rows, block_size=BS, lanes_per_call=lpc, lanes_per_row=WLPR, gather=gather
            )
        ref = outs[("all_split", "natural", 32)]
        for key, o in outs.items():
            if key[0] == "all_split":
                assert all(torch.equal(a, b) for a, b in zip(o, ref)), key
        for lane in range(B):
            p = int(pos[lane])
            if p < 0:
                continue
            ra, rd = KW.wide_rows(lane)
            for mode, o in (("row_split", outs[WIDE_LAYOUTS[0]]), ("all_split", ref)):
                for r in (range(DP) if mode == "all_split" else [lane // LPR]):
                    assert torch.equal(o[r][int(pt[lane, p // BS]), 0, p % BS], rows[ra])
                    if bool(draft[lane]):
                        assert torch.equal(o[r][int(pt[lane, (p + 1) // BS]), 0, (p + 1) % BS], rows[rd])
    assert n_pairs > 40000 and n_same > 5000, (n_pairs, n_same)


def test_host_wide_race_emulation_needs_the_split():
    """At 64 rows the anchor and the draft of a lane still race in one update call (the G12 tile read-modify-write):
    one call per DP row (16 users) or per 32 users loses exactly the earlier writer of every same-tile pair. Every T64
    layout (call A, then call B, in either gather order) loses none, whatever the core order."""
    W = 4
    g = torch.Generator().manual_seed(19)
    base = torch.randn(B * W + 4, 1, BS, D, generator=g)
    rows = torch.randn(WB, D, generator=g)
    lost_total = 0
    for seed in range(20):
        pos, pt, draft = wide_lanes(500 + seed, W, p_idle=0.1, p_draft=0.9)
        st = KW.KVWriteStep.wide_verify(pos, pt, draft)
        want = KW.apply_kv_writes_host(base, st, "all_split", rows, block_size=BS, lanes_per_row=WLPR)[0]
        act = [l for l in range(WB) if int(st.positions[l]) >= 0]
        pairs = [(int(st.owner[d]), d) for d in torch.nonzero(st.call_b).reshape(-1).tolist()]
        for forward in (True, False):
            order = (lambda u: list(u)) if forward else (lambda u: list(u)[::-1])
            for width in (WLPR, 32):  # the one-call variants: a DP row's 16 rows, or 32 rows (2 DP rows)
                one = base.clone()
                for c in range(WB // width):
                    users = [l for l in act if c * width <= l < (c + 1) * width]
                    one = emulate_racy_call(one, st, users, rows, order)
                slot = lambda l: (int(st.page_table[l, int(st.positions[l]) // BS]), 0, int(st.positions[l]) % BS)
                lost = {l for l in act if not torch.equal(one[slot(l)], rows[l])}
                exp = {min(o, d) if forward else max(o, d) for o, d in pairs if same_tile(int(st.positions[o]))}
                assert lost == exp, (seed, forward, width, lost, exp)
                lost_total += len(lost)
            for mode, gather, lpc in WIDE_LAYOUTS[1:]:
                split = base.clone()
                for c in KW.kv_write_calls(st, mode, lanes_per_call=lpc, lanes_per_row=WLPR, gather=gather):
                    users = [l for l, p in zip(c.lanes, c.positions.tolist()) if p >= 0]
                    split = emulate_racy_call(split, st, users, rows, order)
                assert torch.equal(split, want), (seed, mode, gather, lpc)
    assert lost_total > 50


def test_host_wide_input_values():
    W = 4
    pos, pt, draft = wide_lanes(8, W)
    st = KW.KVWriteStep.wide_verify(pos, pt, draft)
    act = pos >= 0
    pt_act = torch.where(act[:, None], pt, torch.zeros_like(pt))  # the anchors' rows (idle lanes zeroed)
    pos_d = torch.where(draft, pos + 1, torch.full_like(pos, -1))
    for mode, gather, lpc in WIDE_LAYOUTS:
        v = KW.kv_write_inputs_host(st, mode, lanes_per_call=lpc, lanes_per_row=WLPR, gather=gather, wide=True)
        assert v["cur_pos"] == ("row", v["cur_pos"][1]) and torch.equal(v["cur_pos"][1], st.positions)
        assert tuple(v["page_table"][1].shape) == (WB, W)
        # the FlashMLA groups of option A'' (per DP row, owner lane order): anchors at n, drafts at n + 1, one table
        assert v["flash_cur_a"][0] == v["flash_cur_d"][0] == v["flash_pt"][0] == "row"
        assert torch.equal(v["flash_cur_a"][1], pos) and torch.equal(v["flash_cur_d"][1], pos_d)
        assert torch.equal(v["flash_pt"][1], pt_act)
        if mode == "row_split":
            assert set(v) == {"cur_pos", "page_table", "cur_a", "cur_b", "flash_cur_a", "flash_cur_d", "flash_pt"}
            continue
        if gather == "split":  # per chunk of the 32 lanes: anchors' call A, drafts' call B, ONE shared pt{c}
            for c in range(32 // lpc):
                sl = slice(c * lpc, (c + 1) * lpc)
                assert v[f"pt{c}"][0] == "rep" and torch.equal(v[f"pt{c}"][1], pt_act[sl])
                assert torch.equal(v[f"cur_a{c}"][1], pos[sl]) and torch.equal(v[f"cur_b{c}"][1], pos_d[sl])
            assert len(v) == 2 + 3 * (32 // lpc) + 3
        else:  # natural: per chunk of lpc physical rows
            pt64 = torch.where(st.active[:, None], st.page_table, torch.zeros_like(st.page_table))
            for c in range(WB // lpc):
                sl = slice(c * lpc, (c + 1) * lpc)
                assert torch.equal(v[f"pt{c}"][1], pt64[sl])
                assert torch.equal(v[f"cur_a{c}"][1], st.call_positions("A")[sl])
                assert torch.equal(v[f"cur_b{c}"][1], st.call_positions("B")[sl])
    # T32 inputs are unchanged: no FlashMLA group inputs without ``wide``
    st32 = steps_for("all_split", 2, W)["verify"]
    assert not any(k.startswith("flash_") for k in KW.kv_write_inputs_host(st32, "all_split"))


def test_host_wide_write_op_wiring(monkeypatch):
    def slice_tile(lo, hi):  # one tile row of the TILE gather [1, 1, 64, 576]
        return [f"slice ag (0, 0, {lo}, 0) (1, 1, {hi}, 576)", "transpose sl 1 2 mc32", "deallocate sl"]

    def chunk(lo, hi, pairs, n):  # one chunk of the DRAM-transposed gather [1, 64, 1, 576]
        return (
            [f"slice tr (0, {lo}, 0, 0) (1, {hi}, 1, 576)", f"to_memory_config sl mc{n}", "deallocate sl"]
            + [f"update cache sh {cur} {pt}" for cur, pt in pairs]
            + ["deallocate sh"]
        )

    def upd(*pairs, u="tr"):
        return [f"update cache {u} {cur} {pt}" for cur, pt in pairs]

    def ab_pairs(c):  # a natural-order chunk's call A and call B
        return [(f"cur_a{c}", f"pt{c}"), (f"cur_b{c}", f"pt{c}")]

    ag2 = "ag_dp_rows kv halves=2"
    expected = {
        ("row_split", "bfp8", "split"): ["transpose kv 1 2 mc16"]
        + upd(("cur_a", "page_table"), ("cur_b", "page_table"))
        + ["deallocate tr"],
        ("all_split", "bfp8", "split"): [ag2]
        + slice_tile(0, 32)
        + upd(("cur_a0", "pt0"))
        + ["deallocate tr"]
        + slice_tile(32, 64)
        + upd(("cur_b0", "pt0"))
        + ["deallocate tr", "deallocate ag"],
        ("all_split", "bf16", "split"): [ag2, "transpose ag 1 2 DRAM", "deallocate ag"]
        + chunk(0, 16, [("cur_a0", "pt0")], 16)
        + chunk(16, 32, [("cur_a1", "pt1")], 16)
        + chunk(32, 48, [("cur_b0", "pt0")], 16)
        + chunk(48, 64, [("cur_b1", "pt1")], 16)
        + ["deallocate tr"],
        ("all_split", "bfp8", "natural"): ["ag_dp_rows kv"]
        + slice_tile(0, 32)
        + upd(("cur_a0", "pt0"), ("cur_b0", "pt0"))
        + ["deallocate tr"]
        + slice_tile(32, 64)
        + upd(("cur_a1", "pt1"), ("cur_b1", "pt1"))
        + ["deallocate tr", "deallocate ag"],
        ("all_split", "bf16", "natural"): ["ag_dp_rows kv", "transpose ag 1 2 DRAM", "deallocate ag"]
        + [x for c in range(4) for x in chunk(16 * c, 16 * c + 16, ab_pairs(c), 16)]
        + ["deallocate tr"],
        ("all", "bfp8", "natural"): ["ag_dp_rows kv"]  # the one-call variant of the device negative control
        + slice_tile(0, 32)
        + upd(("cur0", "pt0"))
        + ["deallocate tr"]
        + slice_tile(32, 64)
        + upd(("cur1", "pt1"))
        + ["deallocate tr", "deallocate ag"],
    }
    for (mode, kv_name, gather), want in expected.items():
        fake, kvw = fake_kv_write(monkeypatch, mode, kv_name, rows=WB, gather=gather)
        assert kvw.wide and kvw.lanes == WB and kvw.lanes_per_row == WLPR
        assert kvw.gather == (gather if KW.is_replicated(mode) else "natural")
        assert kvw.calls_per_layer == len([w for w in want if w.startswith("update")]), (mode, kv_name, gather)
        kv, cache = _T("kv"), _T("cache")
        fake.calls.clear()
        kvw.write(kv, cache, cur_pos=kvw.cur_pos, page_table=kvw.page_table)
        got = describe_calls(fake.calls, kvw.device_inputs)
        assert got == want, (mode, kv_name, gather, got)
        n_out = sum(c[0] in ("transpose", "slice", "to_memory_config", "ag_dp_rows") for c in fake.calls)
        assert sum(c[0] == "deallocate" for c in fake.calls) == n_out, "every op output is freed inside the call"
        # FlashMLA groups of option A'': two B = 8 groups on the anchors' table; the SWA call keeps cur_pos / pt [16]
        d = kvw.device_inputs
        groups = kvw.flash_groups()
        assert [g[0] for g in groups] == [slice(0, 8), slice(8, 16)]
        assert groups[0][1] is d["flash_cur_a"] and groups[1][1] is d["flash_cur_d"]
        assert groups[0][2] is d["flash_pt"] and groups[1][2] is d["flash_pt"]
        assert tuple(d["cur_pos"].value.shape) == (WB,) and tuple(d["flash_pt"].value.shape) == (B, 4)
    # T32: one group, the unchanged FlashMLA inputs
    for mode in KV_WRITE_MODES:
        _, kvw = fake_kv_write(monkeypatch, mode)
        assert not kvw.wide and kvw.flash_groups() == [(slice(0, 8), kvw.cur_pos, kvw.page_table)]
        assert not any(k.startswith("flash_") for k in kvw.device_inputs)


def test_host_wide_writer_refusals(monkeypatch):
    for kw, match in (
        (dict(rows=48), "rows"),
        (dict(rows=96), "rows"),
        (dict(gather="split"), "rows=64"),  # T32 has no draft halves
        (dict(rows=WB, gather="interleaved"), "gather order"),
        (dict(rows=WB, gather="split", mode="all"), "no call B"),
        (dict(rows=WB, lanes_per_call=64), "lanes_per_call"),
        (dict(rows=WB, lanes_per_call=12), "lanes_per_call"),
    ):
        mode = kw.pop("mode", "all_split")
        with pytest.raises(ValueError, match=match):
            fake_kv_write(monkeypatch, mode, **kw)
    _, kvw = fake_kv_write(monkeypatch, "all", rows=WB)  # default gather: natural without call B
    assert kvw.gather == "natural" and kvw.calls_per_layer == 2
    _, kvw = fake_kv_write(monkeypatch, "all_split", "bf16", rows=WB)
    assert kvw.gather == "split" and kvw.lanes_per_call == 16 and kvw.num_chunks == 2 and kvw.calls_per_layer == 4


def test_host_wide_write_step_copies(monkeypatch):
    W = 4
    pos, pt, draft = wide_lanes(21, W, p_idle=0.1, p_draft=0.9)
    verify = KW.KVWriteStep.wide_verify(pos, pt, draft)
    ordinary = KW.KVWriteStep.wide_verify(pos, pt)
    for gather in ("split", "natural"):
        fake, kvw = fake_kv_write(monkeypatch, "all_split", "bfp8", W, rows=WB, gather=gather)
        assert kvw.write_step(KW.KVWriteStep.inactive(W, lanes=WB)) == 0
        n_inputs = len(kvw.device_inputs)
        assert kvw.write_step(verify) == n_inputs  # every input changes from the inactive step
        for name, (place, val) in KW.kv_write_inputs_host(
            verify, "all_split", lanes_per_call=32, lanes_per_row=WLPR, gather=gather, wide=True
        ).items():
            assert torch.equal(kvw.device_inputs[name].value, val), name
        fake.copies.clear()
        n = kvw.write_step(ordinary)  # same anchors, no drafts: only the drafts' inputs change
        changed = {k for k, t in kvw.device_inputs.items() if t.name in fake.copies}
        assert n == len(changed)
        assert "flash_cur_a" not in changed and "flash_pt" not in changed and "flash_cur_d" in changed
        if gather == "split":
            assert changed == {"cur_pos", "page_table", "cur_b0", "flash_cur_d"}
    # the T64 layout is enforced by the writer's check in every gather order
    p64, pt64, partner_of = wide_physical(pos, pt, draft)
    swapped = dict(partner_of)
    a, b2 = sorted(swapped)[:2]
    swapped[a], swapped[b2] = partner_of[b2], partner_of[a]
    bad = KW.KVWriteStep.packed_verify(p64, pt64, swapped)
    for mode, gather in (("all_split", "split"), ("all_split", "natural"), ("row_split", "natural")):
        _, kvw = fake_kv_write(monkeypatch, mode, "bfp8", W, rows=WB, gather=gather)
        with pytest.raises(ValueError, match="draft of anchor row"):
            kvw.write_step(bad)
        with pytest.raises(ValueError, match="lanes"):
            kvw.write_step(KW.KVWriteStep.inactive(W))  # a T32 step on a T64 writer


# ======================================================================================================================
# device helpers
# ======================================================================================================================
def quant(t: torch.Tensor, dtype) -> torch.Tensor:
    return ttnn.to_torch(ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()


def upload_cache(mesh_device, host: torch.Tensor, dtype, device=True):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device if device else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def kv_rows_device(mesh_device, cfg, rows: torch.Tensor):
    """rows [32, 576] lane order (or [64, 576] T64 physical rows) -> the attention's ``kv_row`` ``[1, 1, 8 (16), 576]``
    bf16 TILE per DP row."""
    from models.demos.motif3.tt.rope import shard_lanes

    per = int(rows.shape[0]) // cfg.dp
    return shard_lanes(rows.reshape(cfg.dp, 1, per, D).contiguous(), cfg, mesh_device,
                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)  # fmt: skip


def check_chips(caches, expected, cfg, mesh_device, *, all_chips_layers=(0,)) -> Dict:
    """{(layer, dp, tp): n_bad}: layer ``l`` in ``all_chips_layers`` on all 32 chips, the others on tp 0 / 7 of every
    row."""
    bad = {}
    for l, (c, e) in enumerate(zip(caches, expected)):
        copies = KW.cache_copies(c, mesh_device, cfg)
        for (dp, tp), got in copies.items():
            if l not in all_chips_layers and tp not in (0, cfg.tp - 1):
                continue
            if not torch.equal(got, e[dp].float()):
                bad[(l, dp, tp)] = int((got != e[dp].float()).sum())
    return bad


def capture(mesh_device, fn):
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        fn()
    except BaseException:
        try:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        finally:
            ttnn.release_trace(mesh_device, tid)
        raise
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    return tid


def _setup(mesh_device, kv_name="bfp8"):
    from models.demos.motif3.tt.ccl import MotifCCL, log_fabric

    cfg = MotifTTConfig.from_hf_config(HF_META, mesh_device=mesh_device, kv_cache_dtype=kv_name)
    rep = log_fabric(mesh_device, f"kv_write {kv_name}")
    assert rep["committed"] is not None
    return cfg, MotifCCL(mesh_device, cfg)


# ======================================================================================================================
# device tests
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("kv_name", ["bfp8", "bf16"])
def test_kv_write_device_modes(mesh_device, device_params, kv_name):
    cfg, ccl = _setup(mesh_device, kv_name)
    dt = cfg.dtypes.kv_cache
    W, NL = 4, 3
    N = B * W + 8
    g = torch.Generator().manual_seed(100)
    bases = [quant(torch.randn(N, 1, BS, D, generator=g), dt) for _ in range(NL)]
    hosts = [upload_cache(mesh_device, b, dt, device=False) for b in bases]
    caches = [upload_cache(mesh_device, b, dt) for b in bases]
    failures = []
    for mode in KV_WRITE_MODES:
        kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode=mode)
        for kind, st in steps_for(mode, 17, W).items():
            rows = [quant(torch.randn(B, D, generator=g) * 2.0, dt) for _ in range(NL)]
            kvs = [kv_rows_device(mesh_device, cfg, r) for r in rows]
            for c, h in zip(caches, hosts):
                ttnn.copy_host_to_device_tensor(h, c)
            n_copied = kvw.write_step(st)
            for c, kv in zip(caches, kvs):
                kvw.write(kv, c, cur_pos=kvw.cur_pos, page_table=kvw.page_table)
            kvw.end_step()
            alive = all(kv.is_allocated() for kv in kvs)
            lpc = kvw.lanes_per_call if kvw.replicated else None
            expected = [
                KW.apply_kv_writes_host(b, st, mode, r, block_size=BS, lanes_per_call=lpc) for b, r in zip(bases, rows)
            ]
            nonvacuous = kind == "inactive" or all(not torch.equal(e[0], b) for e, b in zip(expected, bases))
            bad = check_chips(caches, expected, cfg, mesh_device)
            n_writes = int(st.active.sum())
            partners = torch.nonzero(st.call_b).reshape(-1).tolist()
            n_tile = sum(same_tile(int(st.positions[int(st.owner[d])])) for d in partners)
            log(
                f"{kv_name} {mode:9s} {kind:8s}: {n_writes} writes ({len(partners)} partners, {n_tile} same-tile "
                f"pairs), {kvw.calls_per_layer} calls/layer of {kvw.lanes_per_call} lanes, {n_copied} inputs copied: "
                f"caches bit-exact vs the host model {not bad} (layer 0 on 32 chips, layers 1-2 on 8), kv_row kept "
                f"{alive}"
            )
            if bad or not alive or not nonvacuous:
                failures.append(
                    f"{kv_name}/{mode}/{kind}: mismatch {dict(list(bad.items())[:6])} alive {alive} "
                    f"nonvacuous {nonvacuous}"
                )
            for kv in kvs:
                ttnn.deallocate(kv)
        kvw.deallocate()
    for t in caches:
        ttnn.deallocate(t)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_kv_write_device_trace(mesh_device, device_params):
    failures = []
    # (KV dtype, modes, lanes per KV-R call): bfp8 32 lanes, bf16 2 x 16 (G12), bfp8 4 x 8 (the L1 fallback)
    for kv_name, modes, lpc in (
        ("bfp8", KV_WRITE_MODES, None),
        ("bf16", ("all_split",), None),
        ("bfp8", ("all_split",), 8),
    ):
        cfg, ccl = _setup(mesh_device, kv_name)
        dt = cfg.dtypes.kv_cache
        W, NL = 4, 2
        N = B * W + 8
        g = torch.Generator().manual_seed(200)
        bases = [quant(torch.randn(N, 1, BS, D, generator=g), dt) for _ in range(NL)]
        hosts = [upload_cache(mesh_device, b, dt, device=False) for b in bases]
        caches = [upload_cache(mesh_device, b, dt) for b in bases]
        rows = [quant(torch.randn(B, D, generator=g) * 2.0, dt) for _ in range(NL)]
        kvs = [kv_rows_device(mesh_device, cfg, r) for r in rows]
        for mode in modes:
            kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode=mode, lanes_per_call=lpc)
            n_call = kvw.lanes_per_call if kvw.replicated else None
            tag = f"{kv_name}/{mode}/{kvw.lanes_per_call}"

            def step_fn():
                for c, kv in zip(caches, kvs):
                    kvw.write(kv, c, cur_pos=kvw.cur_pos, page_table=kvw.page_table)
                kvw.end_step()

            step_fn()  # eager warmup (inactive step: writes nothing), compiles every program
            ttnn.synchronize_device(mesh_device)
            n_prog = mesh_device.num_program_cache_entries()
            tid = capture(mesh_device, step_fn)
            try:
                seq = [
                    (f"seed{s}/{k}", st) for s in (31, 32) for k, st in steps_for(mode, s, W).items() if k != "inactive"
                ]
                host_us = []
                for name, st in seq:
                    for c, h in zip(caches, hosts):
                        ttnn.copy_host_to_device_tensor(h, c)
                    t0 = time.perf_counter()
                    kvw.write_step(st)
                    host_us.append((time.perf_counter() - t0) * 1e6)
                    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                    exp = [KW.apply_kv_writes_host(b, st, mode, r, block_size=BS, lanes_per_call=n_call)
                           for b, r in zip(bases, rows)]  # fmt: skip
                    bad = check_chips(caches, exp, cfg, mesh_device, all_chips_layers=(0, 1))
                    if bad:
                        failures.append(f"{tag}/{name} replay: {dict(list(bad.items())[:6])}")
                    # trace == eager: the same step run eagerly gives the same caches
                    for c, h in zip(caches, hosts):
                        ttnn.copy_host_to_device_tensor(h, c)
                    step_fn()
                    bad_e = check_chips(caches, exp, cfg, mesh_device, all_chips_layers=())
                    if bad_e:
                        failures.append(f"{tag}/{name} eager: {dict(list(bad_e.items())[:6])}")
                n_prog_end = mesh_device.num_program_cache_entries()
            finally:
                ttnn.release_trace(mesh_device, tid)
            log(
                f"{kv_name} {mode:9s} ({kvw.calls_per_layer} calls of {kvw.lanes_per_call} lanes): one capture, "
                f"{len(seq)} replays with different steps bit-exact vs the host model and vs eager: "
                f"{not any(f.startswith(tag + '/') for f in failures)}; programs compiled by the capture and replays "
                f"{n_prog_end - n_prog}; write_step host cost median {sorted(host_us)[len(host_us) // 2]:.0f} us (max "
                f"{max(host_us):.0f})"
            )
            if n_prog_end != n_prog:
                failures.append(f"{tag}: {n_prog_end - n_prog} programs compiled after the warmup")
            kvw.deallocate()
        for t in caches + kvs:
            ttnn.deallocate(t)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_kv_write_device_flash_reads(mesh_device, device_params):
    from models.demos.motif3.tests.unit.gates import goldens as gd
    from models.demos.motif3.tt.rope import shard_lanes

    cfg, ccl = _setup(mesh_device, "bfp8")
    dt = cfg.dtypes.kv_cache
    W = 4
    N = B * W + 8
    NH, DV = cfg.q_heads_per_chip, cfg.kv_lora_rank
    g = torch.Generator().manual_seed(300)
    base = quant(torch.randn(N, 1, BS, D, generator=g), dt)
    host = upload_cache(mesh_device, base, dt, device=False)
    cache = upload_cache(mesh_device, base, dt)
    rows = quant(torch.randn(B, D, generator=g) * 2.0, dt)
    kv = kv_rows_device(mesh_device, cfg, rows)
    q = torch.randn(B, NH, D, generator=g).bfloat16().float()
    pc, ckc = cfg.flash_mla_decode_pc(), cfg.compute_config("sdpa_decode")
    failures = []
    pos, pt, draft = lane_layout(301, W)
    p_cross, _ = KW.assign_partner_lanes(pos, draft, cross_row=True)
    cases = [
        ("row_split", KW.assign_partner_lanes(pos, draft, cross_row=False)[0], True),
        ("all_split", p_cross, True),
        ("row_split", p_cross, False),  # negative control: cross-row partners without KV-R
    ]
    for mode, partner_of, valid in cases:
        st = KW.KVWriteStep.packed_verify(pos, pt, partner_of)
        kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode=mode)
        ttnn.copy_host_to_device_tensor(host, cache)
        kvw.write_step(st, validate=valid)
        kvw.write(kv, cache, cur_pos=kvw.cur_pos, page_table=kvw.page_table)
        exp = KW.apply_kv_writes_host(base, st, mode, rows, block_size=BS)
        truth = KW.apply_kv_writes_host(base, st, "all_split", rows, block_size=BS)[0]  # every lane's write everywhere
        res = {}
        for window, scale in ((None, gd.SCALE_GLOBAL), (gd.WINDOW, gd.SCALE_SWA)):
            qs = (q * scale).bfloat16().float()
            q_tt = shard_lanes(
                qs.reshape(cfg.dp, cfg.lanes_per_row, NH, D).contiguous(),
                cfg,
                mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
            o = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                q_tt, cache, None, head_dim_v=DV, page_table_tensor=kvw.page_table, cur_pos_tensor=kvw.cur_pos,
                scale=1.0, sliding_window_size=window, program_config=pc, compute_kernel_config=ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            outs = KW.cache_copies(o, mesh_device, cfg)
            pccs = {}
            for l in range(B):
                p = int(st.positions[l])
                if p < 0:
                    continue
                got = outs[(l // LPR, 0)][0, l % LPR, :NH].double()
                want_cache = exp[l // LPR] if valid else truth
                kvl = want_cache[st.page_table[l].long(), 0].reshape(1, -1, D)
                want = gd.mla_decode_golden(qs[l : l + 1], kvl, [p], 1.0, window)[0]
                a, b = want.flatten(), got.flatten()
                pccs[l] = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
            partners = [int(d) for d in torch.nonzero(st.call_b).reshape(-1).tolist()]
            others = [l for l in pccs if l not in partners]
            res[window] = (min(pccs[l] for l in others), min(pccs[l] for l in partners))
            ttnn.deallocate(q_tt)
            ttnn.deallocate(o)
        tag = f"{mode} {'cross-row' if partner_of is p_cross else 'same-row'} partners"
        if valid:
            ok = all(a >= 0.9995 and b >= 0.9995 for a, b in res.values())
            log(
                f"FlashMLA after the {tag} write (global / SWA): worst owner+plain lane pcc "
                f"{res[None][0]:.6f} / {res[gd.WINDOW][0]:.6f}, worst partner {res[None][1]:.6f} / "
                f"{res[gd.WINDOW][1]:.6f} vs fp64 on the host-model cache (bar 0.9995): {ok}"
            )
            if not ok:
                failures.append(f"{tag}: {res}")
        else:
            detected = min(b for _, b in res.values()) < 0.999
            log(
                f"negative control {tag} (refused by the check, forced): partners vs the KV-R truth worst pcc "
                f"{res[None][1]:.4f} / {res[gd.WINDOW][1]:.4f}: stale anchor detected {detected}"
            )
            if not detected:
                failures.append(f"negative control not detected: {res}")
        kvw.deallocate()
    for t in (cache, kv):
        ttnn.deallocate(t)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_kv_write_device_cost(mesh_device, device_params):
    from models.demos.motif3.tests.unit.gates import gate_utils as gu

    res = {}
    for kv_name, modes, lpc in (("bfp8", KV_WRITE_MODES, None), ("bf16", ("all", "all_split"), None),
                                ("bfp8", ("all_split",), 8)):  # fmt: skip
        cfg, ccl = _setup(mesh_device, kv_name)
        dt = cfg.dtypes.kv_cache
        W, N = 512, 4129
        e = ttnn.empty([N, 1, BS, D], dt, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        cache = ttnn.fill(e, 0.0)
        ttnn.deallocate(e)
        g = torch.Generator().manual_seed(400)
        kv = kv_rows_device(mesh_device, cfg, torch.randn(B, D, generator=g))
        pos, pt8, draft = lane_layout(401, 8)
        pt = torch.zeros(B, W, dtype=torch.int32)
        pt[:, :8] = pt8
        for mode in modes:
            kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode=mode, lanes_per_call=lpc)
            st = KW.KVWriteStep.ordinary(pos, pt)
            if kvw.split:
                st = KW.KVWriteStep.packed_verify(
                    pos, pt, KW.assign_partner_lanes(pos, draft, cross_row=kvw.cross_row_partners)[0]
                )
            kvw.write_step(st)
            fn = lambda: kvw.write(kv, cache, cur_pos=kvw.cur_pos, page_table=kvw.page_table)  # noqa: E731
            per, raw = gu.time_traced(mesh_device, fn, ops_per_trace=54, reps=9)
            res[(kv_name, mode, kvw.lanes_per_call)] = per
            kvw.deallocate()
        ttnn.deallocate(kv)
        ttnn.deallocate(cache)
    base = res[("bfp8", "row", 8)]
    for (kv_name, mode, lpc), per in res.items():
        log(f"cost {kv_name} {mode:9s} ({lpc:2d} lanes per call): {per:5.1f} us per layer traced (x 54 layers = "
            f"{per * 54 / 1000:.2f} ms per step; +{(per - base) * 54 / 1000:.2f} ms vs bfp8 row)")  # fmt: skip
    # loose regression bounds (G13a: all_split = AG 24 + transpose ~8 + 2 x 32-lane update ~6.5 us)
    assert res[("bfp8", "row", 8)] < 15 and res[("bfp8", "all_split", 32)] < 60, res


# ======================================================================================================================
# T64 device tests: gate G-S1w with the real DecodeKVWrite(rows=64) (docs/p5_t64/P5_T64_DESIGN.md §6.2, §7.2)
# ======================================================================================================================
def t64_ccl(ccl):
    """``(ccl, how)``: the model's ``MotifCCL`` when its ``ag_dp_rows`` takes ``halves`` (WP-D's D1), else a wrapper
    adding the split-order gather the way T64N's probes did (``docs/p5_t64/scripts/t64_bench.py`` ``ag_split``:
    ROW_MAJOR view ``[1, 2, L / 2, W]``, all-gather dim 2 over DP through ``MotifCCL.all_gather`` (F3N rule R1), view
    ``[1, 1, 4 L, W]``, tilize). The writer's result does not depend on which one runs: the caches are checked
    bit-exactly against the split-order host model."""
    import inspect

    if "halves" in inspect.signature(ccl.ag_dp_rows).parameters:
        return ccl, "ccl.ag_dp_rows(halves=2) (WP-D D1)"

    class SplitGather:
        def __getattr__(self, name):
            return getattr(ccl, name)

        def ag_dp_rows(self, x, *, halves=1, **kw):
            if halves == 1:
                return ccl.ag_dp_rows(x, **kw)
            L, Wd = int(x.shape[-2]), int(x.shape[-1])
            dp = int(ccl.axis_size("dp"))
            rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            g = ccl.all_gather(ttnn.reshape(rm, (1, halves, L // halves, Wd)), 2, "dp",
                               memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
            ttnn.deallocate(rm)
            out = ttnn.to_layout(ttnn.reshape(g, (1, 1, dp * L, Wd)), ttnn.TILE_LAYOUT,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            ttnn.deallocate(g)
            return out

    return SplitGather(), "test shim: ag_dp_rows(halves=2) via MotifCCL.all_gather (D1 not in this tree)"


def l1_pin(mesh_device):
    """A one-page L1 tensor (allocated top-down, just below L1_SMALL) kept alive while new programs run: a program
    whose static circular buffers reach it fails with tt-metal's "static circular buffer region ends at ..." clash
    instead of silently overlapping the CCL semaphores (the design's CB-end check, track B's method)."""
    pin = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    try:
        addr = int(pin.buffer_address())
    except Exception as e:  # report only
        addr = f"n/a ({type(e).__name__})"
    return pin, addr


# (mode, gather) of the T64 writers G-S1w (a) checks; the KV dtype decides the users per KV-R call
WIDE_WRITERS = (("all_split", "split"), ("all_split", "natural"), ("row_split", "natural"))


def wide_steps(seed: int, W: int) -> Dict[str, KW.KVWriteStep]:
    pos, pt, draft = wide_lanes(seed, W, p_idle=0.12, p_draft=0.8)
    return {
        "verify": KW.KVWriteStep.wide_verify(pos, pt, draft),
        "ordinary": KW.KVWriteStep.wide_verify(pos, pt),
        "inactive": KW.KVWriteStep.inactive(W, lanes=WB),
    }


def race_slots(st: KW.KVWriteStep) -> set:
    """(block, row) slots of the anchors and drafts of same-tile pairs (where a one-call write may lose one)."""
    out = set()
    for d in torch.nonzero(st.call_b).reshape(-1).tolist():
        o = int(st.owner[d])
        p = int(st.positions[o])
        if same_tile(p):
            out |= {(int(st.page_table[o, p // BS]), p % BS), (int(st.page_table[d, (p + 1) // BS]), (p + 1) % BS)}
    return out


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("kv_name", ["bfp8", "bf16"])
def test_kv_write_device_wide_modes(mesh_device, device_params, kv_name):
    """G-S1w (a) with the real writer: ``DecodeKVWrite(rows=64)`` in the split and the natural gather order and in
    ``row_split``; T64 verify steps (anchors at ``p % 64`` in {0, 30, 31, 62, 63}, drafts at ``p + 1``, idle and
    undrafted lanes), ordinary T64 steps and inactive steps; three layer caches, every chip's cache bit-exact vs the
    split-order host model (layer 0 on all 32 chips). Negative control, the one-call variant (``all``: anchors and
    drafts in one update call, forced past the check): updates are lost, and only at same-tile anchor / draft slots."""
    cfg, ccl0 = _setup(mesh_device, kv_name)
    ccl, how = t64_ccl(ccl0)
    pin, pin_addr = l1_pin(mesh_device)
    dt = cfg.dtypes.kv_cache
    W, NL = 4, 3
    N_ = B * W + 8
    g = torch.Generator().manual_seed(600)
    bases = [quant(torch.randn(N_, 1, BS, D, generator=g), dt) for _ in range(NL)]
    hosts = [upload_cache(mesh_device, b, dt, device=False) for b in bases]
    caches = [upload_cache(mesh_device, b, dt) for b in bases]
    failures = []
    log(f"T64 {kv_name}: split-order gather by {how}; L1 pin page at {pin_addr}")
    for mode, gather in WIDE_WRITERS:
        kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode=mode, rows=WB, gather=gather)
        lpc = kvw.lanes_per_call if kvw.replicated else None
        for kind, st in wide_steps(611, W).items():
            rows = [quant(torch.randn(WB, D, generator=g) * 2.0, dt) for _ in range(NL)]
            kvs = [kv_rows_device(mesh_device, cfg, r) for r in rows]
            for c, h in zip(caches, hosts):
                ttnn.copy_host_to_device_tensor(h, c)
            n_copied = kvw.write_step(st)
            for c, kv in zip(caches, kvs):
                kvw.write(kv, c, cur_pos=kvw.cur_pos, page_table=kvw.page_table)
            kvw.end_step()
            expected = [
                KW.apply_kv_writes_host(b, st, mode, r, block_size=BS, lanes_per_call=lpc, lanes_per_row=WLPR,
                                        gather=kvw.gather)  # fmt: skip
                for b, r in zip(bases, rows)
            ]
            nonvacuous = kind == "inactive" or all(not torch.equal(e[0], b) for e, b in zip(expected, bases))
            bad = check_chips(caches, expected, cfg, mesh_device)
            n_tile = len(race_slots(st)) // 2
            log(
                f"T64 {kv_name} {mode:9s} {kvw.gather:7s} {kind:8s}: {int(st.active.sum())} writes "
                f"({st.num_partners} drafts, {n_tile} same-tile pairs), {kvw.calls_per_layer} calls/layer of "
                f"{kvw.lanes_per_call} users, {n_copied} inputs copied: caches bit-exact vs the split-order host model "
                f"{not bad} (layer 0 on 32 chips, layers 1-2 on 8)"
            )
            if bad or not nonvacuous:
                failures.append(f"{kv_name}/{mode}/{gather}/{kind}: mismatch {dict(list(bad.items())[:6])} "
                                f"nonvacuous {nonvacuous}")  # fmt: skip
            for kv in kvs:
                ttnn.deallocate(kv)
        kvw.deallocate()
    # negative control: the one-call variant (mode "all": a lane's anchor and draft in one update call)
    one = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode="all", rows=WB, gather="natural")
    lost_total, stray = 0, {}
    for seed in (621, 622, 623):
        st = wide_steps(seed, W)["verify"]
        with pytest.raises(ValueError, match="no call B"):
            one.check(st)
        rows = quant(torch.randn(WB, D, generator=g) * 2.0, dt)
        kv = kv_rows_device(mesh_device, cfg, rows)
        ttnn.copy_host_to_device_tensor(hosts[0], caches[0])
        one.write_step(st, validate=False)
        one.write(kv, caches[0], cur_pos=one.cur_pos, page_table=one.page_table)
        truth = KW.apply_kv_writes_host(bases[0], st, "all_split", rows, block_size=BS, lanes_per_row=WLPR)[0]
        slots = race_slots(st)
        for (dp, tp), got in KW.cache_copies(caches[0], mesh_device, cfg).items():
            diff = {(b, i) for b, i in torch.nonzero((got != truth.float()).any(-1)[:, 0]).tolist()}
            lost_total += len(diff)
            if diff - slots:
                stray[(seed, dp, tp)] = sorted(diff - slots)[:4]
        ttnn.deallocate(kv)
    one.deallocate()
    log(
        f"T64 {kv_name} negative control (one update call per {one.lanes_per_call} rows, anchors and drafts together): "
        f"{lost_total} lost updates over 3 steps x 32 chips (expected > 0, G12 race), all at same-tile anchor / draft "
        f"slots: {not stray}"
    )
    if stray:
        failures.append(f"{kv_name} one-call variant: mismatches outside the same-tile pairs {stray}")
    if lost_total == 0:
        failures.append(f"{kv_name} one-call variant lost no update: the race did not fire (G12 expects losses)")
    for t in caches + [pin]:
        ttnn.deallocate(t)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_kv_write_device_wide_trace(mesh_device, device_params):
    """The T64 writers inside a trace: one capture each (after one eager step compiled every program), replayed for
    different T64 steps written by ``write_step``; bit-exact vs the host model and vs eager after every replay, no
    program compiled by the capture or the replays."""
    failures = []
    for kv_name, mode, gather in (
        ("bfp8", "all_split", "split"),
        ("bfp8", "all_split", "natural"),
        ("bfp8", "row_split", "natural"),
        ("bf16", "all_split", "split"),
    ):
        cfg, ccl0 = _setup(mesh_device, kv_name)
        ccl, how = t64_ccl(ccl0)
        dt = cfg.dtypes.kv_cache
        W, NL = 4, 2
        N_ = B * W + 8
        g = torch.Generator().manual_seed(700)
        bases = [quant(torch.randn(N_, 1, BS, D, generator=g), dt) for _ in range(NL)]
        hosts = [upload_cache(mesh_device, b, dt, device=False) for b in bases]
        caches = [upload_cache(mesh_device, b, dt) for b in bases]
        rows = [quant(torch.randn(WB, D, generator=g) * 2.0, dt) for _ in range(NL)]
        kvs = [kv_rows_device(mesh_device, cfg, r) for r in rows]
        kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode=mode, rows=WB, gather=gather)
        lpc = kvw.lanes_per_call if kvw.replicated else None
        tag = f"{kv_name}/{mode}/{kvw.gather}"

        def step_fn():
            for c, kv in zip(caches, kvs):
                kvw.write(kv, c, cur_pos=kvw.cur_pos, page_table=kvw.page_table)
            kvw.end_step()

        step_fn()  # eager warmup (inactive step: writes nothing), compiles every program
        ttnn.synchronize_device(mesh_device)
        n_prog = mesh_device.num_program_cache_entries()
        tid = capture(mesh_device, step_fn)
        try:
            seq = [(f"seed{s}/{k}", st) for s in (731, 732) for k, st in wide_steps(s, W).items() if k != "inactive"]
            host_us = []
            for name, st in seq:
                for c, h in zip(caches, hosts):
                    ttnn.copy_host_to_device_tensor(h, c)
                t0 = time.perf_counter()
                kvw.write_step(st)
                host_us.append((time.perf_counter() - t0) * 1e6)
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                exp = [KW.apply_kv_writes_host(b, st, mode, r, block_size=BS, lanes_per_call=lpc, lanes_per_row=WLPR,
                                               gather=kvw.gather) for b, r in zip(bases, rows)]  # fmt: skip
                bad = check_chips(caches, exp, cfg, mesh_device, all_chips_layers=(0, 1))
                if bad:
                    failures.append(f"{tag}/{name} replay: {dict(list(bad.items())[:6])}")
                for c, h in zip(caches, hosts):
                    ttnn.copy_host_to_device_tensor(h, c)
                step_fn()  # trace == eager
                bad_e = check_chips(caches, exp, cfg, mesh_device, all_chips_layers=())
                if bad_e:
                    failures.append(f"{tag}/{name} eager: {dict(list(bad_e.items())[:6])}")
            n_prog_end = mesh_device.num_program_cache_entries()
        finally:
            ttnn.release_trace(mesh_device, tid)
        log(
            f"T64 trace {tag} ({kvw.calls_per_layer} calls of {kvw.lanes_per_call} users; {how}): one capture, "
            f"{len(seq)} replays with different steps bit-exact vs the host model and vs eager: "
            f"{not any(f.startswith(tag + '/') for f in failures)}; programs compiled by the capture and replays "
            f"{n_prog_end - n_prog}; write_step host cost median {sorted(host_us)[len(host_us) // 2]:.0f} us"
        )
        if n_prog_end != n_prog:
            failures.append(f"{tag}: {n_prog_end - n_prog} programs compiled after the warmup")
        kvw.deallocate()
        for t in caches + kvs:
            ttnn.deallocate(t)
    assert not failures, "\n".join(failures)


# ---- G-S1w (b): FlashMLA through the real writer's T64 inputs --------------------------------------------------------
# anchor positions n at ~1K / ~4K / ~32K with n % 64 in {0, 31, 63} (+ 1000), so the draft at n + 1 crosses a block,
# a tile or nothing, and the SWA window edges n - 128 / n - 127 sit on every block offset class
FLASH_NS = (960, 991, 1023, 1000, 4032, 4063, 4095, 32000, 32031, 32063)
FLASH_IDLE = (5, 14, 22, 27)
FLASH_UNDRAFTED = (3, 9, 18, 30)
PROBE_C = 4.0  # rope magnitude of a probe key (exact in bfp8): score s needs q_rope = s / PROBE_C
# per probe and row type: {key offset from the anchor position n: pre-softmax score}; every other key scores ~0
PROBES = {
    "causal": {"anchor": {0: 20.0, 1: 36.0}, "draft": {0: 36.0, 1: 20.0}},
    "window": {
        "anchor": {-129: 40.0, -128: 30.0, 0: 10.0, 1: 45.0},
        "draft": {-128: 40.0, -127: 30.0, 0: 20.0, 1: 10.0},
    },
}
PROBE_V = {-129: -1.0, -128: 2.0, -127: -2.0, 0: 1.0, 1: 3.0}  # value (all 512 V dims) of each probe key
PROBE_DIM = {-129: 0, -128: 1, -127: 2, 0: 3, 1: 4}  # its rope dim (512 + k)
# expected output per probe and row type: the value of the highest-scoring key the row may attend
PROBE_WANT = {("causal", "anchor"): 1.0, ("causal", "draft"): 1.0, ("window", "anchor"): 2.0, ("window", "draft"): -2.0}


def flash_layout(W: int):
    """Owner-lane ``(positions [32], page_table [32, W], has_draft [32], n_blocks)``: lanes on :data:`FLASH_NS`,
    idle / undrafted lanes; each lane owns the blocks from ``n - 129`` to ``n + 1``; below that its entries cycle
    through 8 shared blocks (read-only, as a shared prefix), so 32K contexts fit a small pool."""
    pos = torch.full((B,), -1, dtype=torch.int32)
    pt = torch.zeros(B, W, dtype=torch.int32)
    draft = torch.zeros(B, dtype=torch.bool)
    shared = list(range(1, 9))
    nxt = 9
    for lane in range(B):
        if lane in FLASH_IDLE:
            continue
        n = FLASH_NS[lane % len(FLASH_NS)]
        pos[lane] = n
        draft[lane] = lane not in FLASH_UNDRAFTED
        lo, hi = (n - 129) // BS, (n + 1) // BS
        for e in range(hi + 1):
            if e < lo:
                pt[lane, e] = shared[e % len(shared)]
            else:
                pt[lane, e] = nxt
                nxt += 1
    return pos, pt, draft, nxt


def probe_cache(base: torch.Tensor, pos, pt) -> torch.Tensor:
    """``base`` with the window probe keys of every active lane: V (512 dims) = :data:`PROBE_V`, rope = ``PROBE_C`` on
    its :data:`PROBE_DIM` dim (zeros elsewhere), at ``n - 129``, ``n - 128``, ``n - 127``."""
    out = base.clone()
    for lane in range(B):
        n = int(pos[lane])
        if n < 0:
            continue
        for off in (-129, -128, -127):
            p = n + off
            row = torch.zeros(D)
            row[:512] = PROBE_V[off]
            row[512 + PROBE_DIM[off]] = PROBE_C
            out[int(pt[lane, p // BS]), 0, p % BS] = row
    return out


def probe_rows(st: KW.KVWriteStep) -> torch.Tensor:
    """The latent rows the writer writes in the probe run: each anchor row = the key at ``n`` (V +1), each draft row =
    the key at ``n + 1`` (V +3)."""
    rows = torch.zeros(WB, D)
    for r in range(WB):
        if int(st.positions[r]) < 0:
            continue
        off = 1 if bool(st.call_b[r]) else 0
        rows[r, :512] = PROBE_V[off]
        rows[r, 512 + PROBE_DIM[off]] = PROBE_C
    return rows


def probe_queries(st: KW.KVWriteStep, probe: str, NH: int) -> torch.Tensor:
    """``q [64, NH, 576]`` (pre-scaled: FlashMLA runs with scale 1.0): V dims 0, rope dims = score / ``PROBE_C`` per
    probe key, the same on every head."""
    q = torch.zeros(WB, NH, D)
    for r in range(WB):
        kind = "draft" if (r % WLPR) >= LPR else "anchor"
        for off, s in PROBES[probe][kind].items():
            q[r, :, 512 + PROBE_DIM[off]] = s / PROBE_C
    return q


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
@pytest.mark.parametrize("mode", ["all_split", "row_split"])
def test_kv_write_device_wide_flash(mesh_device, device_params, mode):
    """G-S1w (b) with the real writer: FlashMLA on the caches the T64 writer just wrote, through the writer's own
    inputs.

    * option A: one B = 16 call per DP row on ``kv_write.cur_pos [16]`` / ``kv_write.page_table [16, W]`` (the draft
      rows carry their owners' rows);
    * option A'' groups: one B = 8 call per entry of ``kv_write.flash_groups()`` (anchors at ``n``, drafts at ``n +
      1``) on ``q[:, rows]``, outputs concatenated on dim 1;
    * the T32 references: B = 8 calls on T32 writers' inputs (the anchors as an ordinary step at ``n``, the drafts as
      an ordinary step at ``n + 1``).

    The checks: A'' rows bitwise equal to the T32 B = 8 calls on both kinds; A bitwise equal to A'' on SWA; A on
    global at PCC >= 0.9999 (documented non-bitwise: 7 vs 15 cores per user); every active row vs the fp64 golden
    on the host-model cache. Then the probes: a row at ``p`` never sees ``p + 1`` (the draft's key, written by call
    B of the same step), the draft at ``n + 1`` sees the anchor's key ``n`` (call A), on SWA key ``p - 128`` attended
    and ``p - 129`` not (anchor and draft), |output - the attended key's value| <= 1e-3. Positions: n in {960, 991,
    1023, 1000, 4032, 4063, 4095, 32000, 32031, 32063}."""
    from models.demos.motif3.tests.unit.gates import goldens as gd
    from models.demos.motif3.tt.rope import shard_lanes

    cfg, ccl0 = _setup(mesh_device, "bfp8")
    ccl, how = t64_ccl(ccl0)
    pin, pin_addr = l1_pin(mesh_device)
    dt = cfg.dtypes.kv_cache
    NH, DV = cfg.q_heads_per_chip, cfg.kv_lora_rank
    W = 512
    pos, pt, draft, n_blocks = flash_layout(W)
    N_ = max(n_blocks + 8, W)  # paged_update_cache requires page-table width <= the cache's blocks
    st = KW.KVWriteStep.wide_verify(pos, pt, draft)
    pc, ckc = cfg.flash_mla_decode_pc(), cfg.compute_config("sdpa_decode")
    kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode=mode, rows=WB)
    kvw.write_step(st)
    # T32 references: the same lanes as an ordinary 32-lane step at n (anchors) and at n + 1 (drafts)
    pos_d = torch.where(draft, pos + 1, torch.full_like(pos, -1))
    ref_a = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl0, page_table_width=W, mode=mode)
    ref_a.write_step(KW.KVWriteStep.ordinary(pos, pt))
    ref_d = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl0, page_table_width=W, mode=mode)
    ref_d.write_step(KW.KVWriteStep.ordinary(pos_d, pt))
    groups = kvw.flash_groups()
    assert [gr[0] for gr in groups] == [slice(0, 8), slice(8, 16)]
    g = torch.Generator().manual_seed(800)
    rows_ab = [r for r in range(WB) if int(st.positions[r]) >= 0]
    failures = []
    log(f"T64 FlashMLA {mode}: split-order gather by {how}; L1 pin page at {pin_addr}; {len(rows_ab)} active rows "
        f"({st.num_partners} drafts), n in {sorted(set(FLASH_NS))}")  # fmt: skip

    def up_q(q: torch.Tensor):  # [R, NH, D] in physical row order -> [1, R / dp, NH, D] per DP row
        return shard_lanes(q.reshape(cfg.dp, -1, NH, D).bfloat16().contiguous(), cfg, mesh_device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=mesh_device)  # fmt: skip

    def flash(q_tt, cur, ptab, window):
        return ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_tt, cache, None, head_dim_v=DV, page_table_tensor=ptab, cur_pos_tensor=cur, scale=1.0,
            sliding_window_size=window, program_config=pc, compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

    def option_a2(q_tt, window):  # A'': one B = 8 call per flash group, concatenated on dim 1
        outs = []
        for rows, cur, ptab in groups:
            qg = ttnn.slice(q_tt, [0, rows.start, 0, 0], [1, rows.stop, NH, D])
            outs.append(flash(qg, cur, ptab, window))
            ttnn.deallocate(qg)
        o = ttnn.concat(outs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in outs:
            ttnn.deallocate(t)
        return o

    def read(o) -> Dict:  # {(dp, tp): [rows_per_dp, NH, DV] float}
        return {k: v[0, :, :NH].double() for k, v in KW.cache_copies(o, mesh_device, cfg).items()}

    for run in ("random", "causal", "window"):
        if run == "random":
            base = quant(torch.randn(N_, 1, BS, D, generator=g), dt)
            rows = quant(torch.randn(WB, D, generator=g) * 2.0, dt)
        else:
            bg = torch.cat([0.1 * torch.randn(N_, 1, BS, 512, generator=g),
                            0.02 * torch.randn(N_, 1, BS, D - 512, generator=g)], dim=-1)  # fmt: skip
            base = quant(probe_cache(bg, pos, pt), dt)
            rows = quant(probe_rows(st), dt)
        cache = upload_cache(mesh_device, base, dt)
        kv = kv_rows_device(mesh_device, cfg, rows)
        kvw.write(kv, cache, cur_pos=kvw.cur_pos, page_table=kvw.page_table)  # call A (anchors), call B (drafts)
        ttnn.deallocate(kv)
        host = KW.apply_kv_writes_host(base, st, mode, rows, block_size=BS,
                                       lanes_per_call=kvw.lanes_per_call if kvw.replicated else None,
                                       lanes_per_row=WLPR, gather=kvw.gather)  # fmt: skip
        bad_cache = check_chips([cache], [host], cfg, mesh_device, all_chips_layers=())
        if bad_cache:
            failures.append(f"{mode}/{run}: cache after the write differs from the host model {bad_cache}")
        if run == "random":
            q64 = torch.randn(WB, NH, D, generator=g)
        else:
            q64 = probe_queries(st, run, NH)
        kinds = (("swa", gd.WINDOW),) if run == "window" else (("global", None), ("swa", gd.WINDOW))
        for kind, window in kinds:
            scale = 1.0 if run != "random" else (gd.SCALE_SWA if window else gd.SCALE_GLOBAL)
            qs = (q64 * scale).bfloat16().float()
            q_tt = up_q(qs)
            o_a, o_a2 = flash(q_tt, kvw.cur_pos, kvw.page_table, window), option_a2(q_tt, window)
            got_a, got_a2 = read(o_a), read(o_a2)
            # T32 references: B = 8 calls on the anchors' / drafts' queries with the T32 writers' inputs
            qa = up_q(qs.reshape(cfg.dp, 2, LPR, NH, D)[:, 0].reshape(-1, NH, D))
            qd = up_q(qs.reshape(cfg.dp, 2, LPR, NH, D)[:, 1].reshape(-1, NH, D))
            o_ra, o_rd = flash(qa, ref_a.cur_pos, ref_a.page_table, window), flash(qd, ref_d.cur_pos, ref_d.page_table,
                                                                                 window)  # fmt: skip
            got_ra, got_rd = read(o_ra), read(o_rd)
            res = dict(a2_vs_t32=True, replicas=True, a_vs_a2_bitwise=True, a_vs_a2_pcc=1.0, a_vs_a2_max=0.0)
            golden_pcc, probe_err, verdicts = [], [], {}
            for (dp_, tp_), a2 in got_a2.items():
                for r in range(WLPR):
                    phys = WLPR * dp_ + r
                    p = int(st.positions[phys])
                    if p < 0:
                        continue
                    ref = (got_ra if r < LPR else got_rd)[(dp_, tp_)][r % LPR]
                    res["a2_vs_t32"] &= bool(torch.equal(a2[r], ref))
                    a = got_a[(dp_, tp_)][r]
                    # the 8 TP chips of a DP row hold bitwise identical rows (A'' and A)
                    same = torch.equal(a2[r], got_a2[(dp_, 0)][r]) and torch.equal(a, got_a[(dp_, 0)][r])
                    res["replicas"] &= bool(same)
                    res["a_vs_a2_bitwise"] &= bool(torch.equal(a, a2[r]))
                    res["a_vs_a2_max"] = max(res["a_vs_a2_max"], float((a - a2[r]).abs().max()))
                    if tp_ != 0:
                        continue
                    pcc_ab = float(torch.corrcoef(torch.stack([a.flatten(), a2[r].flatten()]))[0, 1])
                    res["a_vs_a2_pcc"] = min(res["a_vs_a2_pcc"], pcc_ab)
                    lane = LPR * dp_ + (r % LPR)
                    kvl = host[dp_][pt[lane, : p // BS + 1].long(), 0].reshape(1, -1, D)  # the blocks it reads
                    want = gd.mla_decode_golden(qs[phys : phys + 1], kvl, [p], 1.0, window)[0]
                    if run == "random":
                        golden_pcc.append(float(torch.corrcoef(torch.stack([want.flatten(), a2[r].flatten()]))[0, 1]))
                    else:
                        rk = "draft" if r >= LPR else "anchor"
                        exp_v = PROBE_WANT[(run, rk)]
                        if run == "causal" and rk == "anchor" and not bool(draft[lane]):
                            exp_v = 1.0  # no draft key at n + 1: the anchor's own key still dominates
                        for opt, o in (("A''", a2[r]), ("A", a)):
                            err = float((o - exp_v).abs().max())
                            probe_err.append(err)
                            if err > 1e-3:
                                verdicts[(opt, kind, lane, rk, p % BS)] = round(float(o.mean()), 4)
                        golden_pcc.append(float((want - a2[r]).abs().max()))
            tag = f"{mode} {run:6s} {kind:6s}"
            if run == "random":
                gp = min(golden_pcc)
                ok = res["a2_vs_t32"] and res["replicas"] and gp >= 0.9995 and (
                    res["a_vs_a2_bitwise"] if kind == "swa" else res["a_vs_a2_pcc"] >= 0.9999
                )
                log(f"T64 FlashMLA {tag}: A'' rows == T32 B=8 rows bitwise {res['a2_vs_t32']} (32 chips, replicas "
                    f"identical {res['replicas']}); A vs A'' "
                    f"bitwise {res['a_vs_a2_bitwise']}, max |d| {res['a_vs_a2_max']:.4g}, "
                    f"pcc {res['a_vs_a2_pcc']:.6f}; worst row pcc vs fp64 {gp:.6f} (bar 0.9995): {ok}")  # fmt: skip
            else:
                ok = res["a2_vs_t32"] and res["replicas"] and not verdicts
                ok = ok and (res["a_vs_a2_bitwise"] or kind == "global")
                log(f"T64 FlashMLA {tag} probe: A'' == T32 bitwise {res['a2_vs_t32']}; every row's output within "
                    f"1e-3 of the attended key's value (max |d| {max(probe_err):.3g}, vs fp64 {max(golden_pcc):.3g}) "
                    f"for A'' and A: {not verdicts}; A vs A'' bitwise {res['a_vs_a2_bitwise']}: {ok}")  # fmt: skip
            if not ok:
                failures.append(f"{tag}: {res} golden {min(golden_pcc) if golden_pcc else None} "
                                f"probe misses {dict(list(verdicts.items())[:8])}")  # fmt: skip
            for t in (q_tt, o_a, o_a2, qa, qd, o_ra, o_rd):
                ttnn.deallocate(t)
        ttnn.deallocate(cache)
    for w in (kvw, ref_a, ref_d):
        w.deallocate()
    ttnn.deallocate(pin)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_kv_write_device_wide_cost(mesh_device, device_params):
    """Traced us per layer of the T64 writers (54 writes per trace, the gates' slope method) next to T32 ``all_split``
    (G16-lite measured 59.0 us split / 66.3 us natural / 16.9 us row_split with an injected writer, T64N §5.1)."""
    from models.demos.motif3.tests.unit.gates import gate_utils as gu

    res = {}
    for kv_name, configs in (
        ("bfp8", ((32, "all_split", "natural"), (64, "all_split", "split"), (64, "all_split", "natural"),
                  (32, "row_split", "natural"), (64, "row_split", "natural"))),
        ("bf16", ((32, "all_split", "natural"), (64, "all_split", "split"))),
    ):  # fmt: skip
        cfg, ccl0 = _setup(mesh_device, kv_name)
        ccl, how = t64_ccl(ccl0)
        dt = cfg.dtypes.kv_cache
        W, N_ = 512, 4129
        e = ttnn.empty([N_, 1, BS, D], dt, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        cache = ttnn.fill(e, 0.0)
        ttnn.deallocate(e)
        g = torch.Generator().manual_seed(900)
        pos, pt8, draft = wide_lanes(901, 8, p_idle=0.0, p_draft=1.0)
        pt = torch.zeros(B, W, dtype=torch.int32)
        pt[:, :8] = pt8
        for rows_n, mode, gather in configs:
            kvw = KW.DecodeKVWrite(mesh_device, cfg, ccl=ccl, page_table_width=W, mode=mode, rows=rows_n,
                                   gather=gather if rows_n == WB else None)  # fmt: skip
            if rows_n == WB:
                st = KW.KVWriteStep.wide_verify(pos, pt, draft)
            else:
                st = KW.KVWriteStep.packed_verify(pos, pt, KW.assign_partner_lanes(
                    pos, draft, cross_row=kvw.cross_row_partners)[0])  # fmt: skip
            kvw.write_step(st)
            kv = kv_rows_device(mesh_device, cfg, torch.randn(rows_n, D, generator=g))
            fn = lambda: kvw.write(kv, cache, cur_pos=kvw.cur_pos, page_table=kvw.page_table)  # noqa: E731
            per, raw = gu.time_traced(mesh_device, fn, ops_per_trace=54, reps=9)
            res[(kv_name, rows_n, mode, kvw.gather, kvw.lanes_per_call)] = per
            ttnn.deallocate(kv)
            kvw.deallocate()
        ttnn.deallocate(cache)
    for (kv_name, rows_n, mode, gather, lpc), per in res.items():
        base = res[(kv_name, 32, "all_split", "natural", 32 if kv_name == "bfp8" else 16)]
        log(f"T64 cost {kv_name} rows {rows_n} {mode:9s} {gather:7s} ({kvw_calls(rows_n, mode, gather, lpc)}): "
            f"{per:5.1f} us per layer traced (x 54 = {per * 54 / 1000:.2f} ms per step; "
            f"{(per - base) * 54 / 1000:+.2f} ms vs T32 {kv_name} all_split)")  # fmt: skip
    # loose regression bounds (G16-lite: split 59.0, natural 66.3, row_split 16.9 us per layer)
    assert res[("bfp8", 64, "all_split", "split", 32)] < 90 and res[("bfp8", 64, "row_split", "natural", 16)] < 30, res


def kvw_calls(rows_n: int, mode: str, gather: str, lpc: int) -> str:
    if not KW.is_replicated(mode):
        return f"2 calls of {rows_n // DP} users per DP row"
    users = rows_n // 2 if gather == "split" else rows_n
    return f"{(users // lpc) * (2 if gather == 'split' or KW.is_split(mode) else 1)} calls of {lpc} users"
