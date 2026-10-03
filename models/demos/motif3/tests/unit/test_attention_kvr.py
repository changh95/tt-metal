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
    LPR,
    fake_kv_write,
    host_cfg,
    lane_layout,
    quant,
    same_tile,
    upload_cache,
)
from models.demos.motif3.tt import attention as A
from models.demos.motif3.tt import kv_write as KW
from models.demos.motif3.tt.attention import MTP_ATTN_PREFIX, MotifAttention
from models.demos.motif3.tt.model_config import device_params
from models.demos.motif3.tt.rope import shard_lanes

B = 32
MESH = [pytest.param((4, 8), device_params(), id="4x8")]
HOOK = "kv_write" in inspect.signature(MotifAttention.forward_decode).parameters
needs_hook = pytest.mark.skipif(not HOOK, reason="MotifAttention.forward_decode has no kv_write= hook (WP2b)")


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
