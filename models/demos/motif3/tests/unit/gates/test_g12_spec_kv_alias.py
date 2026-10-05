# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G12 — speculative-verify KV aliasing: owner at ``p`` and partner at ``p + 1`` in the same blocks
(FEATURES_DESIGN §4 G12, D10, §3.5, §3.8.2; spec_mtp.md §2.9).

Packed verify puts a draft on an idle partner lane whose page-table row is the owner's, so in one decode step the owner
writes its anchor latent at ``p`` and the partner the draft latent at ``p + 1`` of the same block.
``paged_update_cache`` runs one user per core and read-modify-writes the whole 32-row tile holding ``update_idx``
(``writer_update_cache_interleaved_start_id.cpp``): two users writing ``p`` / ``p + 1`` of one tile in one call can lose
an update. The design therefore splits every decode KV write into call A (owners + plain lanes) and call B (partners).

(1) ``test_g12_update_aliasing``: ``p % 64`` in {0, 30, 31, 62, 63} (same tile for 0 / 30 / 62; tile seam 31; block
    seam 63), plain lanes, ``-1`` lanes, bfloat8_b and bf16 caches, in two layouts:
      * per-row 8-lane mode (``row_split``: partners on the owner's DP row; per-row cur_pos / page table / input), and
      * 32-lane gathered mode (``all_split`` under KV-R: replicated 32-lane input, partners on *other* DP rows).
    The A/B split must be bit-exact on all 32 chips (G7 method: pre-quantized rows, the whole cache compared). The
    single-call version is run ``TRIALS`` times to document the race (lost updates counted, not gated).
(2) ``test_g12_flashmla_partners``: FlashMLA decode (G1 config: k_chunk 128, sdpa_decode role) with the partner rows,
    window 129 and global, in both layouts: probes (row ``p`` never sees ``p + 1``, row ``p + 1`` sees ``p``; a lost
    anchor / draft write is visible too), per-user PCC vs the fp64 golden on the expected cache (design target 0.9999
    per user is recorded; G1's kernel floor with fp32 acc was 0.99978 for the worst SWA user, so the hard bar is
    overall >= 0.9998 and worst user >= 0.9995), and the lane-relocation probe of review R5: every user moved to another
    lane on another DP row (same query, position and page-table row) must give a bitwise-identical output.

A bf16 cache takes the 32 gathered lanes as two 16-lane calls: ``paged_update_cache`` sizes its output CB as
``B x Wt`` tiles per core, 1.18 MB at 32 bf16 lanes (L1 clash, recorded as ``bf16/all_split_32lane/single_call_limit``).

Fail -> spec blocked (no packing); escalate.

Run::

    scripts/devrun.sh -t 1800 -n g12 -- python -m pytest models/demos/motif3/tests/unit/gates/test_g12_spec_kv_alias.py \
        -s -p no:cacheprovider
    scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q models/demos/motif3/tests/unit/gates/test_g12_spec_kv_alias.py -k host
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

QUICK = gu.env_flag("MOTIF3_GATES_QUICK")
REC = gu.Recorder("G12_quick" if QUICK else "G12")
MESH = gu.mesh_params(trace_region_size=256 << 20, l1_small_size=32768)

LANES, ROWS, LPR = 32, 4, 8
BS, D, NH, DV = 64, 576, 10, 512
W = 8  # blocks per lane: positions < 512
N_BLOCKS = LANES * W + 8  # 264 (block 0 = null)
OFFSETS = (0, 30, 31, 62, 63)
TRIALS = 2 if QUICK else 6
PROBE_TOL = 2e-2
OWNERS, PLAIN, PARTNERS, IDLE = (0, 1, 2), 3, (4, 5, 6), 7  # local lane slots of every DP row


# ----------------------------------------------------------------------------------------------------------------
# host layout (pure torch)
# ----------------------------------------------------------------------------------------------------------------
def same_tile(p: int) -> bool:
    """Owner p and partner p + 1 share one 32-row tile of one block (the race condition)."""
    return (p + 1) % BS != 0 and (p % BS) // 32 == ((p + 1) % BS) // 32


class Layout:
    """Lane roles for one step. ``mode="row"``: partners sit on the owner's DP row (local slots 4..6 of the same row);
    ``mode="all"``: the partner of owner (r, k) is lane 8 ((r + 1) % 4) + 4 + k (another DP row; needs KV-R).

    pos [32] (-1 inactive), pt [32, W] (partner rows = the owner's row), call_a / call_b [32] (the A / B split
    positions), one_call [32] (everything in one call), owner_of {partner lane: owner lane}."""

    def __init__(self, mode: str, seed: int):
        g = torch.Generator().manual_seed(seed)
        base = (torch.randperm(LANES * W, generator=g) + 1).to(torch.int32).reshape(LANES, W)
        self.mode = mode
        self.pos = torch.full((LANES,), -1, dtype=torch.int32)
        self.pt = base.clone()
        self.owner_of = {}
        role = ["idle"] * LANES
        for r in range(ROWS):
            for k, o_slot in enumerate(OWNERS):
                o = LPR * r + o_slot
                blk = int(torch.randint(1, W - 1, (1,), generator=g))
                p = BS * blk + OFFSETS[(3 * r + k) % len(OFFSETS)]
                prow = r if mode == "row" else (r + 1) % ROWS
                d = LPR * prow + PARTNERS[k]
                self.pos[o], self.pos[d] = p, p + 1
                self.pt[d] = base[o]
                self.owner_of[d] = o
                role[o], role[d] = "owner", "partner"
            pl = LPR * r + PLAIN
            self.pos[pl] = int(torch.randint(0, BS * W, (1,), generator=g))
            role[pl] = "plain"
        self.role = role
        is_partner = torch.tensor([x == "partner" for x in role])
        self.call_a = torch.where(is_partner, torch.full_like(self.pos, -1), self.pos)
        self.call_b = torch.where(is_partner, self.pos, torch.full_like(self.pos, -1))
        self.one_call = self.pos.clone()

    def writes(self, lanes=None):
        """[(lane, block, row)] for every active lane (optionally restricted)."""
        out = []
        for l in range(LANES) if lanes is None else lanes:
            p = int(self.pos[l])
            if p >= 0:
                out.append((l, int(self.pt[l, p // BS]), p % BS))
        return out

    def row_lanes(self, r: int):
        return list(range(LPR * r, LPR * r + LPR))


def apply_writes(cache: torch.Tensor, rows: torch.Tensor, writes) -> torch.Tensor:
    out = cache.clone()
    for l, b, i in writes:
        out[b, 0, i] = rows[l]
    return out


def relocate(l: int) -> int:
    """Lane-relocation permutation for the R5 probe: next DP row, mirrored local slot."""
    return LPR * ((l // LPR + 1) % ROWS) + (LPR - 1 - l % LPR)


def virt_seq(cache: torch.Tensor, pt_row: torch.Tensor) -> torch.Tensor:
    """A lane's virtual sequence [W * 64, 576] through its page-table row."""
    return cache[pt_row.long(), 0].reshape(-1, D)


def test_g12_host_layout_selfcheck():
    assert [same_tile(o) for o in OFFSETS] == [True, True, False, True, False]
    for mode in ("row", "all"):
        L = Layout(mode, seed=3)
        for d, o in L.owner_of.items():
            assert torch.equal(L.pt[d], L.pt[o]) and int(L.pos[d]) == int(L.pos[o]) + 1
            assert (d // LPR == o // LPR) == (mode == "row")
            assert int(L.call_a[d]) == -1 and int(L.call_b[o]) == -1 and int(L.call_b[d]) == int(L.pos[d])
        offs = {int(L.pos[o]) % BS for o in L.owner_of.values()}
        assert offs == set(OFFSETS), offs
        assert int(L.pos[IDLE]) == -1 and all(int(L.pos[LPR * r + IDLE]) == -1 for r in range(ROWS))
        # writes of call A and B are disjoint rows and together equal the single call
        a = {(b, i) for _, b, i in L.writes([l for l in range(LANES) if int(L.call_a[l]) >= 0])}
        b = {(b, i) for _, b, i in L.writes([l for l in range(LANES) if int(L.call_b[l]) >= 0])}
        assert not (a & b) and len(a | b) == len(L.writes())
    assert sorted(relocate(l) for l in range(LANES)) == list(range(LANES))
    assert all(relocate(l) // LPR != l // LPR for l in range(LANES))


# ----------------------------------------------------------------------------------------------------------------
# device helpers
# ----------------------------------------------------------------------------------------------------------------
def row_mapper(mesh_device):
    """Tensor dim 0 split over the 4 DP rows (mesh dim 0), replicated over TP (rope.shard_lanes placement)."""
    return ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], ttnn.MeshShape(ROWS, LPR)),
    )


def update_mc(mesh_device, users: int):
    grid = ttnn.num_cores_to_corerangeset(users, gu.grid_size(mesh_device), row_wise=True)
    spec = ttnn.ShardSpec(grid, [32, D], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


def upload_rows(mesh_device, rows: torch.Tensor, mode: str):
    """rows [32, 576] (lane order) -> the update input: "row" = per DP row [1, 8, 32, 576] on 8 cores; "all" =
    replicated [1, 32, 32, 576] on 32 cores (row 0 of each user's tile holds the latent, G7 layout)."""
    if mode == "row":
        x = torch.zeros(ROWS, LPR, 32, D)
        x[:, :, 0] = rows.reshape(ROWS, LPR, D)
        return gu.to_mesh(x, mesh_device, ttnn.bfloat16, memory_config=update_mc(mesh_device, LPR), mapper=row_mapper(mesh_device))
    x = torch.zeros(1, LANES, 32, D)
    x[0, :, 0] = rows
    return gu.to_mesh(x, mesh_device, ttnn.bfloat16, memory_config=update_mc(mesh_device, LANES))


def upload_lanes(mesh_device, t: torch.Tensor, mode: str, dtype=ttnn.int32):
    """[32, ...] lane-order int tensor -> per-row [8, ...] ("row") or replicated [32, ...] ("all")."""
    mapper = row_mapper(mesh_device) if mode == "row" else None
    return gu.to_mesh(t.contiguous(), mesh_device, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mapper=mapper)


def lane_partitions(mode: str, dtype):
    """Lanes per paged_update_cache call. ``None`` = the per-row 8-lane tensors. The op sizes its output CB as
    ``B x Wt`` tiles per core (paged_update_cache_program_factory.cpp ``num_output_tiles = B * Wt``): 32 users x 18
    tiles x 2048 B = 1.18 MB for a bf16 cache, which overflows into the sharded input (L1 clash, measured
    2026-10-02 17:00), so a bf16 cache takes the 32 gathered lanes in two 16-lane calls; bfloat8_b (627 KB) fits."""
    if mode == "row":
        return [None]
    if dtype == ttnn.bfloat16:
        return [list(range(0, 16)), list(range(16, 32))]
    return [list(range(LANES))]


def upload_rows_subset(mesh_device, rows: torch.Tensor):
    """rows [n, 576] -> replicated [1, n, 32, 576] bf16, height-sharded one user per core on n cores."""
    n = rows.shape[0]
    x = torch.zeros(1, n, 32, D)
    x[0, :, 0] = rows
    return gu.to_mesh(x, mesh_device, ttnn.bfloat16, memory_config=update_mc(mesh_device, n))


class UpdateCalls:
    """Device inputs of one step's KV writes, partitioned per :func:`lane_partitions`; ``run(cache, "a"|"b"|"one")``
    issues the call(s) of that kind in order."""

    def __init__(self, mesh_device, L: "Layout", mode: str, dtype, rows: torch.Tensor):
        self.parts = []
        for lanes in lane_partitions(mode, dtype):
            if lanes is None:
                part = dict(x=upload_rows(mesh_device, rows, "row"), pt=upload_lanes(mesh_device, L.pt, "row"),
                            a=upload_lanes(mesh_device, L.call_a, "row"), b=upload_lanes(mesh_device, L.call_b, "row"),
                            one=upload_lanes(mesh_device, L.one_call, "row"))
            else:
                idx = torch.tensor(lanes)
                part = dict(x=upload_rows_subset(mesh_device, rows[idx]), pt=upload_lanes(mesh_device, L.pt[idx], "all"),
                            a=upload_lanes(mesh_device, L.call_a[idx], "all"), b=upload_lanes(mesh_device, L.call_b[idx], "all"),
                            one=upload_lanes(mesh_device, L.one_call[idx], "all"))
            self.parts.append(part)

    @property
    def calls_per_kind(self) -> int:
        return len(self.parts)

    def run(self, cache, kind: str):
        for p in self.parts:
            ttnn.experimental.paged_update_cache(cache, p["x"], update_idxs_tensor=p[kind], page_table=p["pt"])

    def free(self):
        for p in self.parts:
            gu.free(list(p.values()))


def chip_row(idx: int) -> int:
    return idx // LPR  # get_device_tensors order is row-major over the (4, 8) mesh


def check_chips(cache_tt, expected_for_row) -> tuple:
    """Compare every chip with the expected cache of its DP row. Returns (ok, {chip: n_bad})."""
    bad = {}
    for i, t in enumerate(ttnn.get_device_tensors(cache_tt)):
        got = ttnn.to_torch(t).float()
        want = expected_for_row(chip_row(i))
        if not torch.equal(got, want):
            bad[i] = int((got != want).sum())
    return not bad, bad


def lost_updates(cache_chip: torch.Tensor, rows: torch.Tensor, L: Layout) -> dict:
    """Which written rows did not land (single-call race), per role, from one chip's readback."""
    lost = {"owner": [], "partner": [], "plain": []}
    for l, b, i in L.writes():
        if not torch.equal(cache_chip[b, 0, i], rows[l]):
            lost[L.role[l]].append((l, int(L.pos[l])))
    return lost


def fresh_cache(mesh_device, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    host = torch.randn(N_BLOCKS, 1, BS, D, generator=g)
    host = gu.host_roundtrip(host, dtype) if dtype == ttnn.bfloat8_b else host.bfloat16().float()
    return host, gu.to_mesh(host, mesh_device, dtype)


def quant_rows(rows: torch.Tensor, dtype) -> torch.Tensor:
    return gu.host_roundtrip(rows, ttnn.bfloat8_b) if dtype == ttnn.bfloat8_b else rows.bfloat16().float()


DTYPES = {"bfp8": ttnn.bfloat8_b} if QUICK else {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}


# ----------------------------------------------------------------------------------------------------------------
# (1) update aliasing
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g12_update_aliasing(mesh_device):
    try:
        from models.demos.motif3.tt.ccl import log_fabric

        log_fabric(mesh_device, "G12")
    except Exception as e:  # pragma: no cover
        print(f"[G12] fabric report unavailable: {e}")
    failures = []
    for dname, dtype in DTYPES.items():
        for mode in ("row", "all"):
            L = Layout(mode, seed=12 + (mode == "all"))
            pt_tt = upload_lanes(mesh_device, L.pt, mode)
            lanes_of_chip_row = (lambda r: L.row_lanes(r)) if mode == "row" else (lambda r: list(range(LANES)))
            nparts = len(lane_partitions(mode, dtype))
            tag = f"{dname}/{'row_split_8lane' if mode == 'row' else 'all_split_32lane'}" + (f"_as_{nparts}x16" if nparts > 1 else "")

            # ---- the A / B split (the design) ----
            host, cache = fresh_cache(mesh_device, dtype, seed=120)
            g = torch.Generator().manual_seed(121)
            rows = quant_rows(torch.randn(LANES, D, generator=g) * 2.0, dtype)
            calls = UpdateCalls(mesh_device, L, mode, dtype, rows)
            try:
                calls.run(cache, "a")
                calls.run(cache, "b")
                expected = {r: apply_writes(host, rows, L.writes(lanes_of_chip_row(r))) for r in range(ROWS)}
                ok, bad = check_chips(cache, lambda r: expected[r])
            except Exception as e:
                REC.add(f"{tag}/split", status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
                failures.append(f"{tag}/split: {type(e).__name__}: {str(e)[:200]}")
                calls.free()
                gu.free([cache, pt_tt])
                continue
            REC.add(f"{tag}/split", status="pass" if ok else "fail", chips_checked=32, mismatching_chips=bad,
                    calls_per_kind=calls.calls_per_kind,
                    owner_offsets=sorted(int(L.pos[o]) % BS for o in L.owner_of.values()),
                    same_tile_pairs=sum(same_tile(int(L.pos[o])) for o in L.owner_of.values()))
            if not ok:
                failures.append(f"{tag}/split: mismatching chips {bad}")
            calls.free()
            gu.free([cache])
            if nparts > 1:
                # the single 32-lane bf16 call cannot run at all (output CB B x Wt overflows L1): record, no race test
                REC.add(f"{dname}/all_split_32lane/single_call_limit", status="limitation",
                        note="bf16 cache: one 32-lane paged_update_cache needs a 32 x 18 x 2048 B = 1.18 MB output CB "
                             "(L1 clash with the sharded input, TT_THROW 2026-10-02 17:00); KV-R with a bf16 cache "
                             "must issue <= 16-lane calls (2 calls per A / B kind); bfloat8_b (627 KB) is fine")
                gu.free([pt_tt])
                continue

            # ---- the single call (documents the race; not gated) ----
            one_tt = upload_lanes(mesh_device, L.one_call, mode)
            host, cache = fresh_cache(mesh_device, dtype, seed=122)
            per_trial = []
            for t in range(TRIALS):
                rows = quant_rows(torch.randn(LANES, D, generator=g) * 2.0, dtype)
                x_tt = upload_rows(mesh_device, rows, mode)
                ttnn.experimental.paged_update_cache(cache, x_tt, update_idxs_tensor=one_tt, page_table=pt_tt)
                ttnn.deallocate(x_tt)
                chip = 0 if mode == "all" else 0  # chip 0 = DP row 0 (its 8 lanes in the per-row mode)
                got = ttnn.to_torch(ttnn.get_device_tensors(cache)[chip]).float()
                lanes = lanes_of_chip_row(0)
                Lr = L if mode == "all" else _restrict(L, lanes)
                lost = lost_updates(got, rows, Lr)
                n_lost = sum(len(v) for v in lost.values())
                same_tile_lost = all(same_tile(int(L.pos[L.owner_of.get(l, l)]) if L.role[l] == "partner" else int(L.pos[l]))
                                     for v in lost.values() for l, _ in v if L.role[l] != "plain")
                per_trial.append({"lost": n_lost, "owners": lost["owner"], "partners": lost["partner"],
                                  "plain": lost["plain"], "only_same_tile_pairs": same_tile_lost})
                # resync the expectation: whatever the race left is the new baseline
                host = got
            total = sum(x["lost"] for x in per_trial)
            REC.add(f"{tag}/single_call_race", status="documented", trials=TRIALS, lost_updates_total=total,
                    per_trial=per_trial, note="one paged_update_cache with owner p and partner p+1: expected to lose "
                    "updates on same-tile pairs (p % 64 in {0, 30, 62}); never on 31 / 63 or plain lanes")
            if any(x["plain"] for x in per_trial):
                failures.append(f"{tag}/single_call: a plain lane lost its update {per_trial}")
            gu.free([cache, one_tt, pt_tt])
    assert not failures, "G12 update failures:\n" + "\n".join(failures)


def _restrict(L: Layout, lanes) -> Layout:
    """A view of the layout keeping only ``lanes`` active (per-row readback of the single-call race)."""
    import copy

    R = copy.copy(L)
    keep = torch.zeros(LANES, dtype=torch.bool)
    keep[list(lanes)] = True
    R.pos = torch.where(keep, L.pos, torch.full_like(L.pos, -1))
    return R


# ----------------------------------------------------------------------------------------------------------------
# (2) FlashMLA with partner rows
# ----------------------------------------------------------------------------------------------------------------
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


def flash_mla(mesh_device, q_tt, cache, pt_tt, pos_tt, window, scale):
    return ttnn.transformer.paged_flash_multi_latent_attention_decode(
        q_tt, cache, None, head_dim_v=DV, page_table_tensor=pt_tt, cur_pos_tensor=pos_tt, scale=scale,
        sliding_window_size=window, program_config=mla_pc(mesh_device, window),
        compute_kernel_config=gu.compute_cfg("HiFi4", fp32_acc=True, approx=False),  # sdpa_decode role (G1)
        memory_config=ttnn.DRAM_MEMORY_CONFIG)


def read_lanes(out_tt) -> torch.Tensor:
    """[1, 8, 10, 512] per DP row -> [32, 10, 512] in lane order (first chip of every row)."""
    shards = ttnn.get_device_tensors(out_tt)
    return torch.cat([ttnn.to_torch(shards[r * LPR]).float()[0, :, :NH] for r in range(ROWS)], dim=0)


def probe_rows_for(L: Layout, a: float, seed: int):
    """Probe latent rows: owner writes k_p (score 30, V [0 | +1]), partner writes k_{p+1} (score 30, V [+3 | 0]);
    queries a * e on both. Expected: owner [0 | 1], partner [1.5 | 0.5] (dims [:256] | [256:])."""
    g = torch.Generator().manual_seed(seed)
    e = torch.randn(64, generator=g)
    e = e / e.norm()
    rows = torch.zeros(LANES, D)
    rows[:, :DV] = 0.05 * torch.randn(LANES, DV, generator=g)
    rows[:, DV:] = 0.02 * torch.randn(LANES, 64, generator=g)
    for d, o in L.owner_of.items():
        rows[o, :256], rows[o, 256:DV], rows[o, DV:] = 0.0, 1.0, (30.0 / a) * e
        rows[d, :256], rows[d, 256:DV], rows[d, DV:] = 3.0, 0.0, (30.0 / a) * e
    q = torch.zeros(LANES, NH, D)
    q[:, :, DV:] = a * e
    return rows, q


def probe_check(out: torch.Tensor, L: Layout):
    res, ok = [], True
    for d, o in L.owner_of.items():
        own = (float(out[o, :, :256].mean()), float(out[o, :, 256:].mean()))
        par = (float(out[d, :, :256].mean()), float(out[d, :, 256:].mean()))
        good_o = abs(own[0]) <= PROBE_TOL and abs(own[1] - 1.0) <= PROBE_TOL
        good_p = abs(par[0] - 1.5) <= PROBE_TOL and abs(par[1] - 0.5) <= PROBE_TOL
        verdict_o = "ok" if good_o else ("SEES_P+1" if abs(own[0] - 1.5) < 0.2 else ("ANCHOR_LOST" if abs(own[1]) < 0.2 else "OFF"))
        verdict_p = "ok" if good_p else ("MISSES_P" if abs(par[0] - 3.0) < 0.3 else ("DRAFT_LOST" if abs(par[0]) < 0.2 else "OFF"))
        res.append({"owner": o, "partner": d, "p": int(L.pos[o]), "owner_out": [round(x, 4) for x in own],
                    "partner_out": [round(x, 4) for x in par], "verdict": (verdict_o, verdict_p)})
        ok &= good_o and good_p
    return ok, res


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g12_flashmla_partners(mesh_device):
    failures = []
    a = 16.0
    for mode in ("row", "all"):
        L = Layout(mode, seed=40 + (mode == "all"))
        pt_up = upload_lanes(mesh_device, L.pt, mode)
        a_tt, b_tt = upload_lanes(mesh_device, L.call_a, mode), upload_lanes(mesh_device, L.call_b, mode)
        # FlashMLA always runs per DP row on the 32-lane tables (8 rows each); with "all" the partners of row r's
        # owners sit on row r + 1 and read the owner's blocks there (valid because of the KV-R gathered write)
        pt_fl = upload_lanes(mesh_device, L.pt, "row")
        pos_fl = upload_lanes(mesh_device, L.pos, "row")
        lanes_of_chip_row = (lambda r: L.row_lanes(r)) if mode == "row" else (lambda r: list(range(LANES)))
        for wname, window, scale in (("swa", gd.WINDOW, gd.SCALE_SWA), ("global", None, gd.SCALE_GLOBAL)):
            tag = f"{'row_split' if mode == 'row' else 'all_split'}/{wname}"
            # ---- probes: base cache of small noise, the probe latents written through the A / B split ----
            g = torch.Generator().manual_seed(400)
            base = torch.zeros(N_BLOCKS, 1, BS, D)
            base[..., :DV] = 0.05 * torch.randn(N_BLOCKS, 1, BS, DV, generator=g)
            base[..., DV:] = 0.02 * torch.randn(N_BLOCKS, 1, BS, 64, generator=g)
            base = gu.host_roundtrip(base, ttnn.bfloat8_b)
            cache = gu.to_mesh(base, mesh_device, ttnn.bfloat8_b)
            rows, q = probe_rows_for(L, a, seed=401)
            rows = quant_rows(rows, ttnn.bfloat8_b)
            x_tt = upload_rows(mesh_device, rows, mode)
            ttnn.experimental.paged_update_cache(cache, x_tt, update_idxs_tensor=a_tt, page_table=pt_up)
            ttnn.experimental.paged_update_cache(cache, x_tt, update_idxs_tensor=b_tt, page_table=pt_up)
            ttnn.deallocate(x_tt)
            q_tt = gu.to_mesh(q.reshape(ROWS, LPR, NH, D), mesh_device, ttnn.bfloat16, mapper=row_mapper(mesh_device))
            try:
                o = flash_mla(mesh_device, q_tt, cache, pt_fl, pos_fl, window, 1.0)
                out = read_lanes(o)
                ttnn.deallocate(o)
            except Exception as e:
                REC.add(f"{tag}/probe", status="error", error=f"{type(e).__name__}: {str(e)[:400]}")
                failures.append(f"{tag}/probe: {type(e).__name__}: {str(e)[:200]}")
                gu.free([cache, q_tt])
                continue
            ok, res = probe_check(out, L)
            REC.add(f"{tag}/probe", status="pass" if ok else "fail", pairs=res)
            if not ok:
                failures.append(f"{tag}/probe: {[r['verdict'] for r in res]}")
            gu.free([cache, q_tt])

            # ---- random data: per-user PCC vs fp64 on the expected cache, then the lane relocation ----
            host, cache = fresh_cache(mesh_device, ttnn.bfloat8_b, seed=402)
            g = torch.Generator().manual_seed(403)
            rows = quant_rows(torch.randn(LANES, D, generator=g), ttnn.bfloat8_b)
            x_tt = upload_rows(mesh_device, rows, mode)
            ttnn.experimental.paged_update_cache(cache, x_tt, update_idxs_tensor=a_tt, page_table=pt_up)
            ttnn.experimental.paged_update_cache(cache, x_tt, update_idxs_tensor=b_tt, page_table=pt_up)
            ttnn.deallocate(x_tt)
            q = torch.randn(LANES, NH, D, generator=g).bfloat16().float()
            q_tt = gu.to_mesh(q.reshape(ROWS, LPR, NH, D), mesh_device, ttnn.bfloat16, mapper=row_mapper(mesh_device))
            o = flash_mla(mesh_device, q_tt, cache, pt_fl, pos_fl, window, scale)
            out = read_lanes(o)
            ttnn.deallocate(o)
            active = [l for l in range(LANES) if int(L.pos[l]) >= 0]
            per_user, want_by_lane = {}, {}
            for r in range(ROWS):
                exp_r = apply_writes(host, rows, L.writes(lanes_of_chip_row(r)))  # what row r's chips hold
                for l in L.row_lanes(r):
                    if int(L.pos[l]) < 0:
                        continue
                    kv = virt_seq(exp_r, L.pt[l])[None]
                    want_by_lane[l] = gd.mla_decode_golden(q[l : l + 1], kv, [int(L.pos[l])], scale, window)[0]
                    per_user[l] = gu.pcc(want_by_lane[l], out[l])
            overall = gu.pcc(torch.stack([want_by_lane[l] for l in active]), out[active])
            worst = min(per_user.values())
            # bar = the FlashMLA kernel floor measured by G1 (fp32 acc: worst SWA user 0.99978, overall 0.99993-0.99994
            # on 8 users) with margin; the design's per-user 0.9999 is recorded, the floor is below it with or without
            # partners (the lane-relocation probe shows partner rows compute exactly what an ordinary row computes)
            ok = overall >= 0.9998 and worst >= 0.9995
            REC.add(f"{tag}/random_pcc", status="pass" if ok else "fail", pcc_overall=overall, worst_user_pcc=worst,
                    per_user_target_0p9999_met=worst >= 0.9999,
                    per_user_pcc={str(k): round(v, 6) for k, v in per_user.items()})
            if not ok:
                failures.append(f"{tag}/random: overall {overall:.6f} worst user {worst:.6f}")

            # lane relocation (R5): needs identical caches on every DP row -> only in the gathered (KV-R) layout
            if mode == "all":
                perm = [relocate(l) for l in range(LANES)]
                q2, pos2, pt2 = torch.zeros_like(q), torch.full_like(L.pos, -1), torch.zeros_like(L.pt)
                for l in range(LANES):
                    q2[perm[l]], pos2[perm[l]], pt2[perm[l]] = q[l], L.pos[l], L.pt[l]
                q2_tt = gu.to_mesh(q2.reshape(ROWS, LPR, NH, D), mesh_device, ttnn.bfloat16, mapper=row_mapper(mesh_device))
                pt2_tt, pos2_tt = upload_lanes(mesh_device, pt2, "row"), upload_lanes(mesh_device, pos2, "row")
                o = flash_mla(mesh_device, q2_tt, cache, pt2_tt, pos2_tt, window, scale)
                out2 = read_lanes(o)
                ttnn.deallocate(o)
                diff = [l for l in active if not torch.equal(out2[perm[l]], out[l])]
                REC.add(f"{tag}/lane_relocation_bitwise", status="pass" if not diff else "fail", users=len(active),
                        differing_users=diff, note="same query / position / page-table row on another lane of another "
                        "DP row (review R5 probe at the FlashMLA level)")
                if diff:
                    failures.append(f"{tag}: lane relocation changed {len(diff)} users' outputs: {diff}")
                gu.free([q2_tt, pt2_tt, pos2_tt])
            gu.free([cache, q_tt])
        gu.free([pt_up, a_tt, b_tt, pt_fl, pos_fl])
    assert not failures, "G12 FlashMLA failures:\n" + "\n".join(failures)
