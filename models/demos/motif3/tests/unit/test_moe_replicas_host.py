# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DESIGN-2 replica slots, host side (no device): replica choice, the greedy assignment mirror, the per-chip table,
the plan file and the byte-level replica tensorbin transform (on a synthetic file with the cache's layout).

    scripts/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -q \
        models/demos/motif3/tests/unit/test_moe_replicas_host.py
"""
import json
import random
import struct

import pytest

from models.demos.motif3.tt import replicas as RP


def _freq(seed=0):
    rng = random.Random(seed)
    return [rng.randrange(0, 1000) for _ in range(RP.N_EXPERTS)]


def test_choose_replicas_bijection_and_hot():
    f = _freq()
    slots = RP.choose_replicas(f, 4)
    assert len(slots) == 32 and all(len(r) == 4 for r in slots)
    flat = [e for r in slots for e in r]
    assert len(set(flat)) == 128  # every replicated expert exactly once
    for c, row in enumerate(slots):
        for j, e in enumerate(row):
            h = e // 12
            assert h != c
            assert RP.spread_targets(h, 4)[j] == c
            local = sorted(range(h * 12, h * 12 + 12), key=lambda x: (-f[x], x))
            assert local[j] == e  # rank j of its home chip
    code = RP.rep_codes(slots)
    assert sum(1 for x in code if x != RP.NONE) == 128


def test_assign_basic_and_invariants():
    f = _freq(1)
    slots = RP.choose_replicas(f, 4)
    code = RP.rep_codes(slots)
    rng = random.Random(3)
    for trial in range(200):
        M = rng.choice([1, 8, 32, 64])
        rows = [rng.sample(range(384), 8) for _ in range(M)]
        live = [rng.random() < 0.8 for _ in range(M)]
        act = RP.active_from_rows(rows, live)
        where, load = RP.assign(act, code)
        # every active expert exactly once, on one of its holders
        assert set(where) == {e for e, a in enumerate(act) if a}
        for e, c in where.items():
            assert c == e // 12 or (code[e] != RP.NONE and c == code[e] >> 8)
        assert load == [sum(1 for c in where.values() if c == k) for k in range(32)]
        # keep masks over all chips cover each active expert exactly once
        cover = {}
        for c in range(32):
            for e, k in zip(RP.chip_slots(slots, c), RP.keep_slots(where, slots, c)):
                if k:
                    cover[e] = cover.get(e, 0) + 1
        assert cover == {e: 1 for e in where}
        base, rep = RP.max_load(rows, live, slots)
        assert rep <= base and rep == max(load)


def test_assign_no_replicas_is_home():
    code = [RP.NONE] * 384
    act = [e % 5 == 0 for e in range(384)]
    where, load = RP.assign(act, code)
    assert all(c == e // 12 for e, c in where.items())


def test_assign_tie_goes_home_and_moves():
    slots = RP.choose_replicas([0] * 384, 4)  # ranks by id: chip h donates experts 12h..12h+3
    code = RP.rep_codes(slots)
    act = [False] * 384
    act[0] = True  # home 0, replica chip 1: both empty -> home
    where, _ = RP.assign(act, code)
    assert where[0] == 0
    for e in range(4, 12):  # chip 0 natives without a replica
        act[e] = True
    where, load = RP.assign(act, code)
    assert where[0] == code[0] >> 8 and load[0] == 8


def test_chip_table_layout():
    slots = RP.choose_replicas(_freq(2), 4)
    t = RP.chip_table(slots, 5)
    assert len(t) == RP.TABLE_WORDS == 192
    assert t[:12] == list(range(60, 72)) and t[12:16] == slots[5] and t[16] == 128 and t[17] == 5
    fl = t[32:160]
    code = RP.rep_codes(slots)
    assert fl == sorted(fl) and {v >> 16 for v in fl} == {e for r in slots for e in r}
    for v in fl:
        e = v >> 16
        assert (v >> 12) & 15 == e % 12 and (v >> 6) & 63 == e // 12 and v & 63 == code[e] >> 8
    for c in range(32):
        assert t[160 + c] == sum(1 << (e % 12) for r in slots for e in r if e // 12 == c)
    t0 = RP.chip_table(slots, 5, flex=False)
    assert t0[16] == 0 and t0[32:] == [0] * 160


def test_plan_roundtrip(tmp_path):
    slots = RP.choose_replicas(_freq(4), 4)
    plan = {"format": RP.PLAN_FORMAT, "R": 4, "layers": {"2": slots}, "source": "test"}
    plan["hash"] = RP.plan_hash(plan)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    assert RP.load_plan(p)["layers"]["2"] == slots
    bad = dict(plan)
    bad["layers"] = {"2": [[0, 1, 2, 3]] + slots[1:]}  # chip 0 holding its own experts
    bad["hash"] = RP.plan_hash(bad)
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        RP.load_plan(p)


def _fake_tensorbin(path, n_chips, per, K, N, header_len=3000):
    eb = RP.expert_bytes(K, N)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len) + bytes(header_len))
        for c in range(n_chips):
            for j in range(per):
                e = c * per + j
                f.write(bytes([e % 251]) * (eb - 4) + struct.pack("<I", e))
    return eb


def test_build_replica_tensorbin(tmp_path):
    K, N, per, n = 64, 64, 12, 32
    src = tmp_path / "src.tensorbin"
    eb = _fake_tensorbin(src, n, per, K, N)
    data, shard, size = RP.tensorbin_layout(src, n)
    assert data == 3008 and shard == per * eb
    slots = RP.choose_replicas(_freq(5), 4)
    header = struct.pack("<Q", 100) + bytes(range(100))
    dst = tmp_path / "rep.tensorbin"
    total = RP.build_replica_tensorbin(src, dst, slots, header, K, N, check=False)
    raw = dst.read_bytes()
    assert total == len(raw) == 108 + 32 * 4 * eb
    off = 108
    for c in range(n):
        for j in range(4):
            e = slots[c][j]
            blob = raw[off: off + eb]
            assert struct.unpack("<I", blob[-4:])[0] == e and blob[0] == e % 251
            off += eb
    meta = {"layer": 2, "kind": "x"}
    d2, built = RP.ensure_replica_file(src, tmp_path / "w" / "a.tensorbin", slots, header, K, N, meta=meta, check=False)
    assert built
    d3, built = RP.ensure_replica_file(src, d2, slots, header, K, N, meta=meta, check=False)
    assert not built and d3 == d2
    d4, built = RP.ensure_replica_file(src, d2, slots, header, K, N, meta={"layer": 3}, check=False)
    assert built
