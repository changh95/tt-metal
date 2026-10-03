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


def fake_kv_write(monkeypatch, mode: str, kv_name: str = "bfp8", W: int = 4, lanes_per_call=None):
    fake = FakeTTNN()
    monkeypatch.setattr(KW, "ttnn", fake)
    monkeypatch.setattr(KW, "shard_lanes", lambda v, cfg, mesh, dtype=None, device=None: ("row", v.clone()))

    def ag_dp_rows(x):
        fake.calls.append(("ag_dp_rows", x))
        return fake._new("ag")

    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: "grid")
    ccl = SimpleNamespace(ag_dp_rows=ag_dp_rows)
    cfg = host_cfg(kv_cache_dtype=kv_name)
    kvw = KW.DecodeKVWrite(mesh, cfg, ccl=ccl, page_table_width=W, mode=mode, lanes_per_call=lanes_per_call)
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
    """rows [32, 576] lane order -> the attention's ``kv_row`` ``[1, 1, 8, 576]`` bf16 TILE per DP row."""
    from models.demos.motif3.tt.rope import shard_lanes

    return shard_lanes(rows.reshape(cfg.dp, 1, cfg.lanes_per_row, D).contiguous(), cfg, mesh_device,
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
