# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G13a — KV-R: replicated decode KV writes at op level (FEATURES_DESIGN §4 G13 / review R9, D6, §3.4-3.5, §3.12.3).

With prefix caching on, every decode KV write (53 layers + MTP) must land on all 32 chips (the KV-R invariant), not
only on the lane's DP row: one vLLM block pool serves all 32 lanes, a hit can land on any DP row, and the replicated
prefill + MoE RS(dp) spreads stale rows to every row. Per layer the design gathers the row's ``[1, 1, 8, 576]``
latent over the DP axis and writes all 32 lanes, split in call A (owners + plain lanes) and call B (packed-verify
partners, all -1 in ordinary steps). G13a measures that at op level (no model code), before WP5 integrates it; the
full-model G13b belongs to WP5.

KV-write variants (per layer, then FlashMLA reads the per-row ``cur_pos [8]`` / ``page_table [8, W]``):

=================  ==================================================================================================
``row``            draft 1: transpose -> 8-core height-sharded ``[1, 8, 1, 576]`` -> one 8-lane update (own row only)
``row_split``      spec without APC: the same input, two 8-lane updates (A, B); partners must stay on the owner's row
``all``            APC without spec: ``ccl.ag_dp_rows`` -> ``[1, 1, 32, 576]`` -> transpose -> 32-core sharded -> one
                   32-lane update (``cur_all [32]``, ``pt_all [32, W]`` replicated)
``all_split``      APC + spec (production, design §3.5): the gathered latent, two 32-lane updates (A, B)
``all_split_sag``  alternative gather: transpose to the 8-core sharded update layout, then ``all_gather(dim=1)`` straight
                   into the 32-core sharded layout (no untilize / tilize)
``all_split_dag``  alternative gather: transpose to DRAM ``[1, 8, 1, 576]``, ``all_gather(dim=1)`` in DRAM,
                   ``to_memory_config`` -> 32-core sharded
``deferred``       §3.12.3: per layer ``row_split``; after the last layer one all-gather of the 54 staged latents
                   ``[1, 54, 8, 576]`` and 54 x (slice, transpose, 2 x 32-lane update) for the remote rows (partners
                   must stay on the owner's row: a remote row reads ``n`` only after the deferred write)
=================  ==================================================================================================

Tests:
(1) ``test_g13a_correctness``: 3 layer caches ``[520, 1, 64, 576]`` bfp8; an ordinary step and a verify step (owners at
    ``n``, partners at ``n + 1``; cross-row partners for the KV-R variants, same-row for row_split / deferred): every
    chip's cache bit-exact vs the expected writes (KV-R: all 32 chips hold all 32 lanes; row modes: each row its own),
    FlashMLA outputs of all KV-R variants bitwise equal and vs fp64 golden, trace replay == eager bitwise, and the
    negative control (row_split with cross-row partners must read a stale anchor).
(2) ``test_g13a_cost``: a 54-layer decode-sized trace (per layer: KV write + FlashMLA (14 global + 40 SWA, the Motif
    pattern + MTP) + the ``wo`` AR(tp) ``[1, 1, 8, 4096]``) at contexts 1K and 8K, caches ``[4129, 1, 64, 576]``,
    ``W = 512``: replay min per variant and the step delta vs ``row`` (gate: <= 2.0 ms for ``all_split``; > 2 ms ->
    deferred KV-R), plus standalone traced op costs.
(3) ``test_g13a_serving_order_l1``: prefill 8K (sp0 global + SWA SDPA + fill; sp1 chunked latent SDPA in both compute
    roles (bf16 dest and G9's fp32 dest) + square SWA)
    -> 54-layer ``all_split`` decode (eager compile, capture, replay) -> the same prefills (bitwise equal to before, no
    static-CB clash) -> decode replay -> prefills; main-L1 allocator use unchanged by the KV-R programs (semaphores in
    L1_SMALL), no program compiled after the capture, trace-region use reported.

Run::

    scripts/devrun.sh -t 2400 -n g13a -- python -m pytest models/demos/motif3/tests/unit/gates/test_g13_kv_replicated_decode.py \
        -s -p no:cacheprovider
    scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q models/demos/motif3/tests/unit/gates/test_g13_kv_replicated_decode.py -k host
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

QUICK = gu.env_flag("MOTIF3_GATES_QUICK")
REC = gu.Recorder("G13_quick" if QUICK else "G13")
MESH = gu.mesh_params(trace_region_size=256 << 20, l1_small_size=32768)


def capture_trace(mesh_device, fn):
    """Exception-safe trace capture of ``fn()``: returns ``(trace_id, fn's result)``. If ``fn`` raises inside the
    capture, the capture is ended *and* the trace released before re-raising (a dangling capture hung
    close_mesh_device once, GATES_RESULTS §11.6)."""
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

LANES, ROWS, LPR = 32, 4, 8
D, NH, DV, BS = 576, 10, 512, 64
N_LAYERS = 54  # 53 + MTP (layer 53, SWA)
WINDOWS = [None if (l % 4 == 0 and l < 53) else gd.WINDOW for l in range(N_LAYERS)]  # 14 global, 40 SWA
KV_R_VARIANTS = ("all", "all_split", "all_split_sag", "all_split_dag", "deferred")
ALL_VARIANTS = ("row", "row_split") + KV_R_VARIANTS
if QUICK:
    ALL_VARIANTS = ("row", "row_split", "all_split", "all_split_sag", "deferred")
DELTA_GATE_US = 2000.0
PARTNER_SLOTS, OWNER_SLOTS, PLAIN_SLOT = (4, 5, 6), (0, 1, 2), 3


# ----------------------------------------------------------------------------------------------------------------
# host layout
# ----------------------------------------------------------------------------------------------------------------
class Layout:
    """Lane roles of one step. ``verify=False``: ordinary step (every lane writes its own position; call B all -1).
    ``verify=True``: owners at n (local slots 0..2), partners at n + 1 (slots 4..6) with the owner's page-table row,
    on the owner's row (``cross_row=False``) or on the next DP row (``cross_row=True``, needs KV-R); slot 3 plain,
    slot 7 idle (-1)."""

    def __init__(self, W: int, seed: int, *, verify: bool, cross_row: bool = True, ctx=None, block_offset: int = 0):
        g = torch.Generator().manual_seed(seed)
        # every lane owns `need` real blocks (ids block_offset + 1 ..), the rest of its row is the null block 0
        need = W if ctx is None else min(W, -(-(ctx + 1) // BS))
        base = torch.zeros(LANES, W, dtype=torch.int32)
        base[:, :need] = torch.arange(1, LANES * need + 1, dtype=torch.int32).reshape(LANES, need) + block_offset
        base = base[torch.randperm(LANES, generator=g)]
        self.n_blocks_used = block_offset + LANES * need + 1
        self.pt = base.clone()
        self.pos = torch.full((LANES,), -1, dtype=torch.int32)
        self.owner_of = {}
        hi = BS * need - 2
        for l in range(LANES):
            self.pos[l] = (ctx - 1) if ctx is not None else int(torch.randint(BS, hi, (1,), generator=g))
        is_partner = torch.zeros(LANES, dtype=torch.bool)
        if verify:
            for r in range(ROWS):
                for k in range(3):
                    o = LPR * r + OWNER_SLOTS[k]
                    d = LPR * ((r + 1) % ROWS if cross_row else r) + PARTNER_SLOTS[k]
                    self.pt[d] = self.pt[o]
                    self.pos[d] = self.pos[o] + 1
                    self.owner_of[d] = o
                    is_partner[d] = True
                self.pos[LPR * r + 7] = -1
        self.is_partner = is_partner
        self.call_a = torch.where(is_partner, torch.full_like(self.pos, -1), self.pos)
        self.call_b = torch.where(is_partner, self.pos, torch.full_like(self.pos, -1))

    def writes(self, lanes=None):
        out = []
        for l in range(LANES) if lanes is None else lanes:
            p = int(self.pos[l])
            if p >= 0:
                out.append((l, int(self.pt[l, p // BS]), p % BS))
        return out


def apply_writes(cache: torch.Tensor, rows: torch.Tensor, writes) -> torch.Tensor:
    out = cache.clone()
    for l, b, i in writes:
        out[b, 0, i] = rows[l]
    return out


def expected_caches(variant: str, base, rows, L: Layout):
    """{dp_row: expected cache} after one step of ``variant``."""
    if variant in ("row", "row_split"):
        return {r: apply_writes(base, rows, L.writes(range(LPR * r, LPR * r + LPR))) for r in range(ROWS)}
    full = apply_writes(base, rows, L.writes())
    return {r: full for r in range(ROWS)}


def test_g13_host_layout_selfcheck():
    for cross in (True, False):
        L = Layout(16, seed=1, verify=True, cross_row=cross)
        for d, o in L.owner_of.items():
            assert torch.equal(L.pt[d], L.pt[o]) and int(L.pos[d]) == int(L.pos[o]) + 1
            assert ((d // LPR) != (o // LPR)) == cross
        assert int(L.call_b[~L.is_partner].max()) == -1 and int(L.call_a[L.is_partner].max()) == -1
    L = Layout(16, seed=2, verify=False)
    assert int(L.call_b.max()) == -1 and int(L.pos.min()) >= 0
    e_row = expected_caches("row", torch.zeros(LANES * 16 + 8, 1, BS, D), torch.ones(LANES, D), L)
    e_all = expected_caches("all_split", torch.zeros(LANES * 16 + 8, 1, BS, D), torch.ones(LANES, D), L)
    assert float(e_row[0].sum()) * 4 == float(e_all[0].sum()) and torch.equal(e_all[0], e_all[3])
    assert sum(w is None for w in WINDOWS) == 14 and len(WINDOWS) == 54


# ----------------------------------------------------------------------------------------------------------------
# device helpers
# ----------------------------------------------------------------------------------------------------------------
def make_ccl(mesh_device):
    from models.demos.motif3.tt.ccl import MotifCCL
    from models.demos.motif3.tt.model_config import MeshAxes

    return MotifCCL(mesh_device, axes=MeshAxes.detect(tuple(mesh_device.shape)))


def row_mapper(mesh_device):
    return ttnn.create_mesh_mapper(
        mesh_device, ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], ttnn.MeshShape(ROWS, LPR))
    )


def sharded_mc(mesh_device, users: int):
    grid = ttnn.num_cores_to_corerangeset(users, gu.grid_size(mesh_device), row_wise=True)
    return ttnn.create_sharded_memory_config(shape=(32, D), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT,
                                             orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                             use_height_and_width_as_shard_shape=True)


def lanes_tensor(mesh_device, t: torch.Tensor, per_row: bool):
    return gu.to_mesh(t.contiguous(), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                      mapper=row_mapper(mesh_device) if per_row else None)


class Step:
    """Per-step device inputs (persistent, as the generator will hold them)."""

    def __init__(self, mesh_device, L: Layout, ccl):
        self.ccl = ccl
        self.cur8 = lanes_tensor(mesh_device, L.pos, True)
        self.cur8_a = lanes_tensor(mesh_device, L.call_a, True)
        self.cur8_b = lanes_tensor(mesh_device, L.call_b, True)
        self.pt8 = lanes_tensor(mesh_device, L.pt, True)
        self.cur_all = lanes_tensor(mesh_device, L.pos, False)
        self.cur_a = lanes_tensor(mesh_device, L.call_a, False)
        self.cur_b = lanes_tensor(mesh_device, L.call_b, False)
        self.pt_all = lanes_tensor(mesh_device, L.pt, False)
        self.mc8 = sharded_mc(mesh_device, LPR)
        self.mc32 = sharded_mc(mesh_device, LANES)
        self.staged = []

    def free(self):
        gu.free([self.cur8, self.cur8_a, self.cur8_b, self.pt8, self.cur_all, self.cur_a, self.cur_b, self.pt_all])


def _upd(cache, u, cur, pt):
    ttnn.experimental.paged_update_cache(cache, u, update_idxs_tensor=cur, page_table=pt)


def gather_ag_rows(kv, st):
    g = st.ccl.ag_dp_rows(kv)  # [1, 1, 32, 576] TILE DRAM, lane order 8 dp + l
    u = ttnn.transpose(g, 1, 2, memory_config=st.mc32)  # [1, 32, 1, 576], one lane per core
    ttnn.deallocate(g)
    return u


def gather_sharded(kv, st):
    u8 = ttnn.transpose(kv, 1, 2, memory_config=st.mc8)
    u = st.ccl.all_gather(u8, 1, "dp", memory_config=st.mc32)
    ttnn.deallocate(u8)
    return u


def gather_dram(kv, st):
    u8 = ttnn.transpose(kv, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g = st.ccl.all_gather(u8, 1, "dp", memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(u8)
    u = ttnn.to_memory_config(g, st.mc32)
    ttnn.deallocate(g)
    return u


def kv_write(variant: str, kv, cache, st: Step):
    if variant in ("row", "row_split", "deferred"):
        u = ttnn.transpose(kv, 1, 2, memory_config=st.mc8)
        if variant == "row":
            _upd(cache, u, st.cur8, st.pt8)
        else:
            _upd(cache, u, st.cur8_a, st.pt8)
            _upd(cache, u, st.cur8_b, st.pt8)
        ttnn.deallocate(u)
        if variant == "deferred":
            st.staged.append((kv, cache))
        return
    gather = {"all": gather_ag_rows, "all_split": gather_ag_rows, "all_split_sag": gather_sharded,
              "all_split_dag": gather_dram}[variant]
    u = gather(kv, st)
    if variant == "all":
        _upd(cache, u, st.cur_all, st.pt_all)
    else:
        _upd(cache, u, st.cur_a, st.pt_all)
        _upd(cache, u, st.cur_b, st.pt_all)
    ttnn.deallocate(u)


def deferred_tail(st: Step):
    """§3.12.3: one AG of every staged layer latent, then the remote rows' updates (own rows rewritten identically)."""
    kvs = [kv for kv, _ in st.staged]
    staged = ttnn.concat(kvs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, n, 8, 576] TILE
    rm = ttnn.to_layout(staged, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(staged)
    g = st.ccl.all_gather(rm, 2, "dp", memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, n, 32, 576] RM
    ttnn.deallocate(rm)
    gt = ttnn.to_layout(g, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(g)
    for l, (_, cache) in enumerate(st.staged):
        s = ttnn.slice(gt, [0, l, 0, 0], [1, l + 1, LANES, D], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        u = ttnn.transpose(s, 1, 2, memory_config=st.mc32)
        ttnn.deallocate(s)
        _upd(cache, u, st.cur_a, st.pt_all)
        _upd(cache, u, st.cur_b, st.pt_all)
        ttnn.deallocate(u)
    ttnn.deallocate(gt)
    st.staged.clear()


def mla_pc(mesh_device, window=None):
    """The serving FlashMLA decode program config of the layer kind (review I-7): SWA (``window`` set) takes the shipped
    SWA ``max_cores_per_head_batch`` (A2: ``MOTIF3_FLASH_MLA_SWA_MCPH``, default 4; 16 = the release), global 16.
    Imported lazily (gate import rule)."""
    from models.demos.motif3.tt.model_config import (
        FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH_SWA,
        _env_int,
        flash_mla_decode_pc,
    )

    swa_mcph = _env_int("MOTIF3_FLASH_MLA_SWA_MCPH", FLASH_MLA_DECODE_MAX_CORES_PER_HEAD_BATCH_SWA)
    return flash_mla_decode_pc(gu.grid_size(mesh_device), "global" if window is None else "swa", swa_mcph=swa_mcph)


MLA_CKC = None


def flash(mesh_device, q, cache, st: Step, window):
    global MLA_CKC
    if MLA_CKC is None:
        MLA_CKC = gu.compute_cfg("HiFi4", fp32_acc=True, approx=False)  # sdpa_decode role
    return ttnn.transformer.paged_flash_multi_latent_attention_decode(
        q, cache, None, head_dim_v=DV, page_table_tensor=st.pt8, cur_pos_tensor=st.cur8, scale=1.0,
        sliding_window_size=window, program_config=mla_pc(mesh_device, window), compute_kernel_config=MLA_CKC,
        memory_config=ttnn.DRAM_MEMORY_CONFIG)


def decode_step(mesh_device, variant, st: Step, caches, kv_rows, qs, *, windows=None, x_ar=None, keep=False):
    """One decode-proxy step over len(caches) layers. qs = (q_global, q_swa) per-row [1, 8, 10, 576]; windows[l] =
    129 (SWA) or None (global), default the Motif pattern ``WINDOWS``."""
    windows = WINDOWS if windows is None else windows
    outs = []
    for l, cache in enumerate(caches):
        kv_write(variant, kv_rows[l], cache, st)
        o = flash(mesh_device, qs[0] if windows[l] is None else qs[1], cache, st, windows[l])
        if keep:
            outs.append(o)
        else:
            ttnn.deallocate(o)
        if x_ar is not None:
            ttnn.deallocate(st.ccl.ar_tp(x_ar))
    if variant == "deferred":
        deferred_tail(st)
    return outs


def read_lanes(out_tt) -> torch.Tensor:
    shards = ttnn.get_device_tensors(out_tt)
    return torch.cat([ttnn.to_torch(shards[r * LPR]).float()[0, :, :NH] for r in range(ROWS)], dim=0)


def kv_rows_tensor(mesh_device, rows: torch.Tensor):
    """rows [32, 576] lane order -> per DP row [1, 1, 8, 576] bf16 TILE DRAM (the attention's ``kv_row``)."""
    return gu.to_mesh(rows.reshape(ROWS, 1, LPR, D), mesh_device, ttnn.bfloat16, mapper=row_mapper(mesh_device))


def q_tensor(mesh_device, q: torch.Tensor):
    return gu.to_mesh(q.reshape(ROWS, LPR, NH, D), mesh_device, ttnn.bfloat16, mapper=row_mapper(mesh_device))


def host_cache_tensor(mesh_device, host: torch.Tensor):
    return ttnn.from_torch(host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))


def check_caches(caches, expected, layers, chips):
    """{(layer, chip): n_bad} for chips whose cache differs from expected[layer][dp_row]."""
    bad = {}
    for l in layers:
        shards = ttnn.get_device_tensors(caches[l])
        for i in chips:
            got = ttnn.to_torch(shards[i]).float()
            want = expected[l][i // LPR]
            if not torch.equal(got, want):
                bad[(l, i)] = int((got != want).sum())
    return bad


# ----------------------------------------------------------------------------------------------------------------
# (1) correctness
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g13a_correctness(mesh_device):
    try:
        from models.demos.motif3.tt.ccl import log_fabric

        log_fabric(mesh_device, "G13a")
    except Exception as e:  # pragma: no cover
        print(f"[G13] fabric report unavailable: {e}")
    ccl = make_ccl(mesh_device)
    W, NL = 16, 3
    N = LANES * W + 8
    windows = [None, gd.WINDOW, gd.WINDOW]  # layer kinds of the 3-layer step (global, SWA, SWA)
    failures = []
    g = torch.Generator().manual_seed(13)
    bases = [gu.host_roundtrip(torch.randn(N, 1, BS, D, generator=g), ttnn.bfloat8_b) for _ in range(NL)]
    caches = [gu.to_mesh(b, mesh_device, ttnn.bfloat8_b) for b in bases]
    host_bases = [host_cache_tensor(mesh_device, b) for b in bases]
    q = torch.randn(LANES, NH, D, generator=g).bfloat16().float()
    qs = (q_tensor(mesh_device, (q * gd.SCALE_GLOBAL).bfloat16().float()), q_tensor(mesh_device, (q * gd.SCALE_SWA).bfloat16().float()))
    q_host = {None: (q * gd.SCALE_GLOBAL).bfloat16().float(), gd.WINDOW: (q * gd.SCALE_SWA).bfloat16().float()}

    def reset():
        for c, h in zip(caches, host_bases):
            ttnn.copy_host_to_device_tensor(h, c)

    try:
        for step_kind in ("ordinary", "verify"):
            ref_outs = None
            for variant in ALL_VARIANTS:
                if step_kind == "verify" and variant == "all":
                    continue  # the single 32-lane call would race on owner / partner tiles (G12)
                cross = variant not in ("row_split", "deferred")
                if step_kind == "verify" and variant == "row":
                    continue
                L = Layout(W, seed=131, verify=step_kind == "verify", cross_row=cross)
                tag = f"{step_kind}/{variant}" + ("/cross_row_partners" if step_kind == "verify" and cross else "")
                rows = [gu.host_roundtrip(torch.randn(LANES, D, generator=torch.Generator().manual_seed(1300 + l)) * 2.0,
                                          ttnn.bfloat8_b) for l in range(NL)]
                kvs = [kv_rows_tensor(mesh_device, r) for r in rows]
                st = Step(mesh_device, L, ccl)
                reset()
                try:
                    outs = decode_step(mesh_device, variant, st, caches, kvs, qs, windows=windows, keep=True)
                    eager_o = [read_lanes(o) for o in outs]
                    gu.free(outs)
                    expected = [expected_caches(variant, bases[l], rows[l], L) for l in range(NL)]
                    chips_l0 = range(32)
                    bad = check_caches(caches, expected, [0], chips_l0)
                    bad.update(check_caches(caches, expected, range(1, NL), (0, 9, 18, 27)))
                    snap = {l: ttnn.to_torch(ttnn.get_device_tensors(caches[l])[9]).float() for l in range(NL)}
                    # trace == eager: reset, capture the same step, replay once
                    reset()
                    tid, touts = capture_trace(mesh_device, lambda: decode_step(mesh_device, variant, st, caches, kvs, qs, windows=windows, keep=True))
                    act = [l for l in range(LANES) if int(L.pos[l]) >= 0]  # FlashMLA never writes inactive lanes' rows
                    try:
                        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                        tr_o = [read_lanes(o) for o in touts]
                        tr_eq = all(torch.equal(a[act], b[act]) for a, b in zip(tr_o, eager_o)) and all(
                            torch.equal(ttnn.to_torch(ttnn.get_device_tensors(caches[l])[9]).float(), snap[l]) for l in range(NL))
                    finally:
                        ttnn.release_trace(mesh_device, tid)
                    gu.free(touts)
                except Exception as e:
                    REC.add(f"correctness/{tag}", status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
                    failures.append(f"{tag}: {type(e).__name__}: {str(e)[:200]}")
                    st.free()
                    gu.free(kvs)
                    continue
                rec = dict(cache_mismatch=bad, trace_bitwise_eq_eager=tr_eq, chips_checked_layer0=32)
                active = [l for l in range(LANES) if int(L.pos[l]) >= 0]
                if variant in KV_R_VARIANTS or step_kind == "ordinary" or not cross:
                    # FlashMLA vs fp64 golden on the expected cache (what the lane's row holds)
                    worst = 1.0
                    for li in range(NL):
                        for l in active:
                            exp = expected[li][l // LPR]
                            kv = exp[L.pt[l].long(), 0].reshape(1, -1, D)
                            want = gd.mla_decode_golden(q_host[windows[li]][l : l + 1], kv, [int(L.pos[l])], 1.0, windows[li])[0]
                            worst = min(worst, gu.pcc(want, eager_o[li][l]))
                    rec["flashmla_worst_user_pcc"] = worst
                    if worst < 0.999:
                        failures.append(f"{tag}: FlashMLA worst user pcc {worst:.6f}")
                if variant in KV_R_VARIANTS and step_kind == "verify":
                    if ref_outs is None:
                        ref_outs = (variant, eager_o)
                    elif variant != "deferred":  # active lanes only: inactive rows are never written (DRAM garbage)
                        rec["flashmla_bitwise_eq_" + ref_outs[0]] = all(torch.equal(a[active], b[active]) for a, b in zip(eager_o, ref_outs[1]))
                ok = not bad and tr_eq
                REC.add(f"correctness/{tag}", status="pass" if ok else "fail", **rec)
                if not ok:
                    failures.append(f"{tag}: cache mismatch {dict(list(bad.items())[:6])} trace_eq={tr_eq}")
                st.free()
                gu.free(kvs)

        # ---- negative control: row_split with cross-row partners (no KV-R) reads a stale anchor ----
        L = Layout(W, seed=131, verify=True, cross_row=True)
        rows = [gu.host_roundtrip(torch.randn(LANES, D, generator=torch.Generator().manual_seed(1300 + l)) * 2.0,
                                  ttnn.bfloat8_b) for l in range(NL)]
        kvs = [kv_rows_tensor(mesh_device, r) for r in rows]
        st = Step(mesh_device, L, ccl)
        reset()
        outs = decode_step(mesh_device, "row_split", st, caches, kvs, qs, windows=windows, keep=True)
        got = [read_lanes(o) for o in outs]
        gu.free(outs)
        stale = []
        for li in range(NL):
            full = apply_writes(bases[li], rows[li], L.writes())  # the KV-R truth
            for d, o in L.owner_of.items():
                kv = full[L.pt[d].long(), 0].reshape(1, -1, D)
                want = gd.mla_decode_golden(q_host[windows[li]][d : d + 1], kv, [int(L.pos[d])], 1.0, windows[li])[0]
                stale.append(gu.pcc(want, got[li][d]))
        detected = min(stale) < 0.999
        REC.add("correctness/negative_control/row_split_cross_row_partners", status="detected" if detected else "NOT_detected",
                partner_pcc_min=min(stale), partner_pcc_median=sorted(stale)[len(stale) // 2],
                note="without KV-R a partner on another DP row reads the owner's anchor KV at n from its own row's stale "
                     "copy (design §3.4 / §3.8.2: partners must then stay on the owner's row)")
        if not detected:
            failures.append("negative control (cross-row partners without KV-R) not detected")
        st.free()
        gu.free(kvs)
    finally:
        gu.free(host_bases)
    assert not failures, "G13a correctness failures:\n" + "\n".join(failures)


# ----------------------------------------------------------------------------------------------------------------
# (2) cost: 54-layer decode-sized trace
# ----------------------------------------------------------------------------------------------------------------
LAST_TRACE_BYTES = {}


def replay_us(mesh_device, fn, reps: int = 9):
    """Capture fn() once, replay ``reps`` times; returns sorted µs per replay (replay + synchronize). The trace-region
    bytes the capture took land in ``LAST_TRACE_BYTES["bytes"]``."""
    before = trace_region(mesh_device)
    tid, _ = capture_trace(mesh_device, lambda: fn())
    after = trace_region(mesh_device)
    LAST_TRACE_BYTES["bytes"] = (after[0] - before[0]) if (before and after) else None
    try:
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            ts.append((time.perf_counter() - t0) * 1e6)
    finally:
        ttnn.release_trace(mesh_device, tid)
    return sorted(ts)


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g13a_cost(mesh_device):
    ccl = make_ccl(mesh_device)
    W, N = 512, 4129
    failures = []
    caches = {}
    for kind in ("global", "swa"):
        e = ttnn.empty([N, 1, BS, D], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        caches[kind] = ttnn.fill(e, 0.0)
        ttnn.deallocate(e)
    layer_caches = [caches["global"] if w is None else caches["swa"] for w in WINDOWS]
    g = torch.Generator().manual_seed(1313)
    kvs = [kv_rows_tensor(mesh_device, torch.randn(LANES, D, generator=g)) for _ in range(N_LAYERS)]
    q = torch.randn(LANES, NH, D, generator=g)
    qs = (q_tensor(mesh_device, q * gd.SCALE_GLOBAL), q_tensor(mesh_device, q * gd.SCALE_SWA))
    x_ar = gu.to_mesh(torch.randn(ROWS, 1, LPR, 4096, generator=g), mesh_device, ttnn.bfloat16, mapper=row_mapper(mesh_device))
    floor = gu.time_trace_replay_floor(mesh_device)
    REC.add("cost/trace_replay_floor", status="info", us=floor)
    results = {}
    for ctx in ((8192,) if QUICK else (1024, 8192)):
        for step_kind in ("ordinary", "verify"):
            if step_kind == "verify" and ctx != 8192:
                continue
            for variant in ALL_VARIANTS:
                if step_kind == "verify" and variant in ("row", "all"):
                    continue
                cross = variant not in ("row_split", "deferred")
                L = Layout(W, seed=7, verify=step_kind == "verify", cross_row=cross, ctx=ctx)
                st = Step(mesh_device, L, ccl)
                case = f"cost/ctx{ctx}/{step_kind}/{variant}"
                try:
                    decode_step(mesh_device, variant, st, layer_caches, kvs, qs, x_ar=x_ar)  # eager: compile
                    ttnn.synchronize_device(mesh_device)
                    t0 = time.perf_counter()
                    decode_step(mesh_device, variant, st, layer_caches, kvs, qs, x_ar=x_ar)
                    ttnn.synchronize_device(mesh_device)
                    eager_ms = (time.perf_counter() - t0) * 1e3
                    ts = replay_us(mesh_device, lambda: decode_step(mesh_device, variant, st, layer_caches, kvs, qs, x_ar=x_ar))
                except Exception as e:
                    REC.add(case, status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
                    if variant in ("row", "all_split"):
                        failures.append(f"{case}: {type(e).__name__}: {str(e)[:200]}")
                    st.free()
                    continue
                results[(ctx, step_kind, variant)] = ts[0]
                REC.add(case, status="measured", step_min_us=ts[0], step_median_us=ts[len(ts) // 2], replays_us=[round(t) for t in ts],
                        eager_ms=eager_ms, layers=N_LAYERS, trace_bytes=LAST_TRACE_BYTES.get("bytes"))
                st.free()
        base = results.get((ctx, "ordinary", "row"))
        base_split = results.get((ctx, "ordinary", "row_split"))
        for variant in ALL_VARIANTS:
            t = results.get((ctx, "ordinary", variant))
            if t is None or base is None:
                continue
            d = t - base
            rec = dict(step_us=t, delta_vs_row_us=d, delta_per_layer_us=d / N_LAYERS,
                       delta_vs_row_split_us=(t - base_split) if base_split is not None else None)
            if variant == "all_split":
                rec["gate_delta_le_2ms"] = d <= DELTA_GATE_US
                if d > DELTA_GATE_US:
                    rec["decision"] = "> 2 ms: deferred KV-R (design §3.12.3)"
            REC.add(f"cost/ctx{ctx}/delta/{variant}", status="measured", **rec)

    # ---- standalone traced op costs (per call) ----
    L = Layout(W, seed=7, verify=False, ctx=8192)
    st = Step(mesh_device, L, ccl)
    kv = kvs[0]
    u8 = ttnn.transpose(kv, 1, 2, memory_config=st.mc8)
    u32 = gather_ag_rows(kv, st)
    none32 = lanes_tensor(mesh_device, torch.full((LANES,), -1, dtype=torch.int32), False)
    ops = {
        "transpose_to_8core_shard": lambda: ttnn.transpose(kv, 1, 2, memory_config=st.mc8),
        "update_8lane": lambda: _upd(caches["swa"], u8, st.cur8, st.pt8),
        "ag_dp_rows_8x576": lambda: st.ccl.ag_dp_rows(kv),
        "gather_ag_rows_to_32core_shard": lambda: gather_ag_rows(kv, st),
        "gather_sharded_ag": lambda: gather_sharded(kv, st),
        "gather_dram_ag": lambda: gather_dram(kv, st),
        "update_32lane_active": lambda: _upd(caches["swa"], u32, st.cur_all, st.pt_all),
        "update_32lane_all_minus1": lambda: _upd(caches["swa"], u32, none32, st.pt_all),
        "flashmla_swa_8k": lambda: flash(mesh_device, qs[1], caches["swa"], st, gd.WINDOW),
        "flashmla_global_8k": lambda: flash(mesh_device, qs[0], caches["global"], st, None),
        "ar_tp_8x4096": lambda: st.ccl.ar_tp(x_ar),
    }
    for name, fn in ops.items():
        try:
            tr, raw = gu.time_traced(mesh_device, fn, ops_per_trace=32, reps=9)
            REC.add(f"cost/op/{name}", status="measured", traced_us=tr, traced_raw_us=raw)
        except Exception as e:
            REC.add(f"cost/op/{name}", status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
    gu.free([u8, u32, none32])
    st.free()
    assert not failures, "G13a cost failures:\n" + "\n".join(failures)


# ----------------------------------------------------------------------------------------------------------------
# (3) serving order / L1
# ----------------------------------------------------------------------------------------------------------------
def l1_main_bytes(mesh_device) -> int:
    try:
        return int(ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_allocated_per_bank)
    except Exception:
        return -1


def trace_region(mesh_device):
    try:
        v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.TRACE)
        nb = int(v.num_banks)
        return int(v.total_bytes_allocated_per_bank) * nb, int(v.total_bytes_per_bank) * nb
    except Exception:
        return None


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g13a_serving_order_l1(mesh_device):
    ccl = make_ccl(mesh_device)
    W, N, C = 512, 4129, 8192
    failures = []
    g = torch.Generator().manual_seed(1330)
    caches = {}
    for kind in ("global", "swa"):
        e = ttnn.empty([N, 1, BS, D], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
        caches[kind] = ttnn.fill(e, 0.0)
        ttnn.deallocate(e)
    layer_caches = [caches["global"] if w is None else caches["swa"] for w in WINDOWS]
    ckc_pf = gu.compute_cfg("HiFi4", fp32_acc=False, approx=False)
    ckc_pf32 = gu.compute_cfg("HiFi4", fp32_acc=True, approx=False)  # G9's role for the sp1 global op (legacy kernel)

    def pcfg(q, k):
        return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=gu.grid_size(mesh_device), q_chunk_size=q,
                                      k_chunk_size=k, exp_approx_mode=False)

    # prefill inputs (8K bucket, replicated as in serving)
    qx = gu.to_mesh(torch.randn(1, NH, C, 192, generator=g) * 0.1, mesh_device, ttnn.bfloat16)
    kx = gu.to_mesh(torch.randn(1, 2, C, 192, generator=g), mesh_device, ttnn.bfloat16)
    vx = gu.to_mesh(torch.randn(1, 2, C, 192, generator=g), mesh_device, ttnn.bfloat16)
    sq = gu.to_mesh(torch.randn(1, NH, 128 + C, 192, generator=g) * 0.1, mesh_device, ttnn.bfloat16)
    sk = gu.to_mesh(torch.randn(1, 2, 128 + C, 192, generator=g), mesh_device, ttnn.bfloat16)
    sv = gu.to_mesh(torch.randn(1, 2, 128 + C, 192, generator=g), mesh_device, ttnn.bfloat16)
    ql = gu.to_mesh(torch.randn(1, NH, C, D, generator=g) * 0.14, mesh_device, ttnn.bfloat16)
    xf = gu.to_mesh(gu.host_roundtrip(torch.randn(1, 1, C, D, generator=g), ttnn.bfloat8_b), mesh_device, ttnn.bfloat8_b)
    fill_pt = gu.to_mesh((torch.arange(C // BS, dtype=torch.int32) + 1)[None], mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    sdpa_pt_h = torch.zeros(1, 640, dtype=torch.int32)
    sdpa_pt_h[0, : 2 * C // BS] = torch.arange(1, 2 * C // BS + 1, dtype=torch.int32)
    sdpa_pt = gu.to_mesh(sdpa_pt_h, mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    start = gu.to_mesh(torch.tensor([C], dtype=torch.int32), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    prefills = {
        "sp0_global_sdpa_q256": lambda: ttnn.transformer.scaled_dot_product_attention(
            qx, kx, vx, is_causal=True, scale=1.0, program_config=pcfg(256, 256), compute_kernel_config=ckc_pf),
        "sp0_swa_sdpa_q128": lambda: ttnn.transformer.scaled_dot_product_attention(
            qx, kx, vx, is_causal=True, scale=1.0, sliding_window_size=gd.WINDOW, program_config=pcfg(128, 128),
            compute_kernel_config=ckc_pf),
        "sp0_fill": lambda: (ttnn.experimental.paged_fill_cache(caches["global"], xf, fill_pt, batch_idx=0), None)[1],
        "sp1_global_latent_q128": lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
            ql, caches["global"], caches["global"], sdpa_pt, chunk_start_idx_tensor=start, scale=1.0,
            program_config=pcfg(128, 128), compute_kernel_config=ckc_pf),
        "sp1_global_latent_q64": lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
            ql, caches["global"], caches["global"], sdpa_pt, chunk_start_idx_tensor=start, scale=1.0,
            program_config=pcfg(64, 64), compute_kernel_config=ckc_pf),
        "sp1_global_latent_q128_fp32acc": lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
            ql, caches["global"], caches["global"], sdpa_pt, chunk_start_idx_tensor=start, scale=1.0,
            program_config=pcfg(128, 128), compute_kernel_config=ckc_pf32),
        "sp1_global_latent_q64_fp32acc": lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
            ql, caches["global"], caches["global"], sdpa_pt, chunk_start_idx_tensor=start, scale=1.0,
            program_config=pcfg(64, 64), compute_kernel_config=ckc_pf32),
        "sp1_swa_square_q128": lambda: ttnn.transformer.scaled_dot_product_attention(
            sq, sk, sv, is_causal=True, scale=1.0, sliding_window_size=gd.WINDOW, program_config=pcfg(128, 128),
            compute_kernel_config=ckc_pf),
    }

    def run_prefills(tag):
        res = {}
        for name, fn in prefills.items():
            try:
                o = fn()
                ttnn.synchronize_device(mesh_device)
                res[name] = None if o is None else gu.read_dev(o, 0)
                if o is not None:
                    ttnn.deallocate(o)
            except Exception as e:
                res[name] = f"{type(e).__name__}: {str(e)[:300]}"
        return res

    u_start = l1_main_bytes(mesh_device)
    first = run_prefills("before")
    for name, v in first.items():
        if isinstance(v, str):
            failures.append(f"prefill {name} before any decode: {v}")
    u_after_prefill = l1_main_bytes(mesh_device)

    # decode lanes on blocks 301.. (disjoint from the prefill's blocks 1..256, so the prefill outputs must not move)
    L = Layout(W, seed=7, verify=True, cross_row=True, ctx=4096, block_offset=300)
    assert L.n_blocks_used <= N
    st = Step(mesh_device, L, ccl)
    kvs = [kv_rows_tensor(mesh_device, torch.randn(LANES, D, generator=g)) for _ in range(N_LAYERS)]
    q = torch.randn(LANES, NH, D, generator=g)
    qs = (q_tensor(mesh_device, q * gd.SCALE_GLOBAL), q_tensor(mesh_device, q * gd.SCALE_SWA))
    x_ar = gu.to_mesh(torch.randn(ROWS, 1, LPR, 4096, generator=g), mesh_device, ttnn.bfloat16, mapper=row_mapper(mesh_device))
    decode_step(mesh_device, "all_split", st, layer_caches, kvs, qs, x_ar=x_ar)  # eager: compile every decode program
    ttnn.synchronize_device(mesh_device)
    u_after_decode_compile = l1_main_bytes(mesh_device)
    tr0 = trace_region(mesh_device)
    n_prog_before_capture = mesh_device.num_program_cache_entries()
    tid, _ = capture_trace(mesh_device, lambda: decode_step(mesh_device, "all_split", st, layer_caches, kvs, qs, x_ar=x_ar))
    tr1 = trace_region(mesh_device)
    detail = []
    try:
        for rnd in range(2):
            for _ in range(3):
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            again = run_prefills(f"after_decode_{rnd}")
            for name, v in again.items():
                ref = first[name]
                if isinstance(v, str):
                    detail.append((rnd, name, v))
                    failures.append(f"prefill {name} after decode round {rnd}: {v}")
                elif isinstance(ref, torch.Tensor) and not torch.equal(v, ref):
                    detail.append((rnd, name, "differs from the pre-decode output"))
                    failures.append(f"prefill {name} after decode round {rnd} differs from before")
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
    finally:
        ttnn.release_trace(mesh_device, tid)
    n_prog_end = mesh_device.num_program_cache_entries()
    u_end = l1_main_bytes(mesh_device)
    ok = not failures and n_prog_end == n_prog_before_capture and u_after_decode_compile == u_after_prefill
    REC.add("serving_order_l1", status="pass" if ok else "fail",
            main_l1_bytes={"start": u_start, "after_prefill": u_after_prefill, "after_kvr_decode_compile": u_after_decode_compile,
                           "end": u_end},
            programs_after_capture=n_prog_end - n_prog_before_capture,
            trace_region_bytes={"before": tr0, "after_capture": tr1},
            decode_trace_mib=((tr1[0] - tr0[0]) / 2**20) if (tr0 and tr1) else None,
            prefill_problems=detail, order="prefill 8K sp0+sp1 -> decode(all_split, 54 layers) x3 -> prefill -> x3 -> prefill")
    if n_prog_end != n_prog_before_capture:
        failures.append(f"{n_prog_end - n_prog_before_capture} programs compiled after the decode capture")
    if u_after_decode_compile != u_after_prefill:
        failures.append(f"main L1 grew by {u_after_decode_compile - u_after_prefill} B over the KV-R decode compile")
    st.free()
    assert not failures, "G13a serving-order failures:\n" + "\n".join(failures)
