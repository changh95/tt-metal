# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G15a-rest — packed (batched) prefill attention at op level (docs/p5_t64/P5_T64_DESIGN.md §6.2 "G15a-rest", §3.2-§3.5;
design note docs/p5_t64/p5.md §5, §14).

A packed pass runs B segments of S rows (``T = B * S``) through every row-local program at bucket T; only the SDPA,
the RoPE positions and the KV fill are per segment. The SDPA becomes one batch-B call through a metadata view
``[1, H, T, d] -> [H, B, S, d]`` and a CN transpose ``-> [B, H, S, d]`` (batch row k = segment k), and back. G15a-lite
(p5.md §14) measured pk0 at S 64-512 / B 4-32 and pk1 global at B 8 / 32 bitwise per segment. This gate covers the
rest. Standalone: no model code, random data at Motif per-chip shapes, B0's program-config builders and compute roles
(``tt/model_config.py``: ``sdpa_prefill_pc``, ``resumed_prefill_pc``, ``compute_config``), torch fp32 goldens.

(a) ``test_g15a_pk0_batched_sdpa``: pk0 causal SDPA ``q [B,10,S,192]``, ``k / v [B,2,S,192]`` bf16 (view + CN transpose
    of ``[1,H,T,d]``), ``scale 1.0``, ``cfg.sdpa_prefill_pc(kind, seq_len=S)``, role ``sdpa_prefill``, window 129 on
    SWA at S >= 129 (``MotifAttention.prefill_sdpa_window_and_config``): (S, B) in {(1024, 2), (1024, 4), (1024, 8),
    (512, 2), (512, 16), (256, 32), (128, 2), (64, 2)} x {global, swa}, against the single-row SDPA ``[1,H,S,d]`` on
    the same slices with the same config. (d) eager time of the packed call (3 input transposes + SDPA + 1 output
    transpose) vs the B single calls, and of the 4 transposes alone (T = 8192 at (1024, 8), (512, 16), (256, 32)).
(b) ``test_g15b_pk1_global``: pk1 global ``chunked_scaled_dot_product_attention(Q [B,10,S,576], K = V = the paged
    latent cache [4129, 1, 64, 576], page_table [B, 640], chunk_start_idx_tensor [1] = a)``,
    ``cfg.resumed_prefill_pc("global", S, kv_dtype)`` (per-bucket q / k), role ``sdpa_prefill_fp32``: (B, S, a) in
    {(2, 1024, 2048), (8, 512, 8192), (16, 256, 128), (32, 128, 24576)}; the B = 32 case shares its 384 prefix blocks
    (the system-prompt case: 4129 blocks cannot hold 32 distinct 24K prefixes), the others use distinct prefixes;
    dummy segments copy segment 0's SDPA row (review edit R-E2): one in the B = 8 case, two in the B = 32 case. Plus a
    bf16-cache case (8, 512, 2048) (64/64 table) and a traced replay with a rewritten start (captured at 8192,
    replayed at 2048 / 4096 / 8192) against eager. (d) eager packed (2 transposes + SDPA) vs B single calls.
(c) ``test_g15c_pk1_swa``: the pk1 SWA square, the full dataflow of design §3.4: tails by 2B tensor-args block slices
    (``slice_dim=0, num_devices=4129``) + one concat of the 2B blocks + view ``[B,1,128,576]`` + typecast
    (``distinct``), or one 2-block gather + typecast + ``repeat`` (``shared``); ``lat = concat([tail_b, view(kv_row,
    [B,1,S,576])], dim=2)``; the draft-1 expansion (``nlp_create_q_heads_split``, ``linear`` with a non-batched
    ``[512, 640]`` in1, ``repeat``, ``concat``) on ``[B, ...]``; ``Q_cat = concat([qb[:, :, :128], qb], dim=2)``;
    causal + window 129, ``cfg.resumed_prefill_pc("swa", S)``, role ``sdpa_prefill``; rows ``[128, 128 + S)``, CN back:
    (B, S) in {(32, 128), (8, 512), (2, 1024)} x {distinct, shared}, two dummies (copying segment 0's tails) in the
    B = 32 distinct case. Against the solo sp1 SWA dataflow per segment (``MotifAttention._prefill_sp1_swa`` ops at
    batch 1) and, separately, the batched SDPA alone against single SDPA calls on its own batch slices; the program
    count of each variant (review D9: the B = 32 distinct concat takes 64 inputs) and that it stays constant when the
    tail ids change; a traced replay with rewritten tail bounds against eager; window probes on the batched square SDPA
    (G10's probes per segment). (d) eager packed dataflow vs B solo dataflows.

Every case also checks: non-finite outputs 0; the packed output's replicas bitwise identical on all 32 chips; the
static-CB end of the new programs below a one-page L1 pin at the top of main L1 (track B's ``probe_sp1_cbend`` method:
tt-metal checks static CBs against the lowest L1 buffer only, not against the L1_SMALL CCL semaphores above main L1);
views metadata-only (same buffer address). These ops run no CCL, so ``ring_gather`` does not apply (recorded).

Pass (§6.2): every real segment bitwise equal to the single-row op on the same slices; PCC vs fp32 >= 0.9997 (pk1
global >= 0.9998 on cols [:512]); (c) rows [128, 128 + S) PCC >= 0.999 vs window attention at absolute positions,
probes exact; views share the buffer; trace == eager; CB end below the pin. Kill / fallback (§6.2): a kind not bitwise
but PCC >= 0.9999 -> accept and widen the CP-P bars; a kind failing -> per-segment loop; (c) failing -> pk1 off.

Run (device; from a B0 + C1a snapshot with copies of the root conftest.py / pytest.ini and of this file, so concurrent
edits of the shared tree cannot change the imported package, GATES_RESULTS.md §13; results land in the live tree's
``results/G15.jsonl`` through ``MOTIF3_GATES_RESULTS_DIR``)::

    scripts/devrun.sh -t 2400 -n g15a_rest -- bash -c "cd <snap> && MOTIF3_GATES_RESULTS_DIR=<live>/results \
        PYTHONPATH=<snap>:<tt-metal> python -m pytest \
        <snap>/models/demos/motif3/tests/unit/gates/test_g15_packed_prefill.py -c <snap>/pytest.ini --rootdir <snap> \
        -s -p no:cacheprovider -k 'not host'"
    scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q \
        models/demos/motif3/tests/unit/gates/test_g15_packed_prefill.py -k host
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.motif3.tests.unit.gates import gate_utils as gu
from models.demos.motif3.tests.unit.gates import goldens as gd

QUICK = gu.env_flag("MOTIF3_GATES_QUICK")
GATE = "G15_quick" if QUICK else "G15"
REC = gu.Recorder(GATE)
# a snapshot run (GATES_RESULTS.md §13) points MOTIF3_GATES_RESULTS_DIR at the live tree's results/
RESULTS_DIR = Path(os.environ.get("MOTIF3_GATES_RESULTS_DIR") or Path(__file__).resolve().parent / "results")
REC.path = RESULTS_DIR / f"{GATE}.jsonl"
MESH = gu.mesh_params(trace_region_size=256 << 20, l1_small_size=32768)

NQ, NKV, DQK, DV = 10, 2, 192, 128  # expanded GQA prefill heads per chip (G2): V zero-padded 128 -> 192
D_LAT, D_NOPE, D_ROPE = 576, 512, 64  # absorbed latent [n 512 | k_pe 64] (sp1 global, G9)
BS, N_BIG = 64, 4129  # KV block; serving pool blocks per layer (MOTIF3_KV_POOL_TOKENS 262144 / 64 + 1)
TAIL = 128  # SWA tail rows = window - 1
WINDOW = gd.WINDOW  # 129
EXP_W = 2 * (128 + 192)  # prefill KV expansion [512, 640]: per group [k_nope 128 | v 128 | 0 64]
PCC_PK0 = 0.9997  # design §6.2 bar; recorded per case
# The packed SDPA is gated on bitwise equality with the single-row op (today's solo prefill); its PCC vs fp32 is
# then the single-row op's own. G2 validated that op down to 0.99948 (global q/k 256/256), so a bitwise case passes
# with PCC >= this floor and records whether it also meets the design bar.
PCC_PK0_FLOOR = 0.9994
PCC_PK1_GLOBAL = 0.9998
PCC_PK1_SWA = 0.999
TRACE_START_CAPTURE, TRACE_STARTS = 8192, (2048, 4096, 8192)

PK0_CASES = [(1024, 2), (1024, 4), (1024, 8), (512, 2), (512, 16), (256, 32), (128, 2), (64, 2)]  # (S, B)
# (B, S, a, prefix, dummies): the last ``dummies`` segments copy segment 0's SDPA row (R-E2)
PK1_GLOBAL_CASES = [(2, 1024, 2048, "distinct", 0), (8, 512, 8192, "distinct", 1), (16, 256, 128, "distinct", 0),
                    (32, 128, 24576, "shared", 2)]  # fmt: skip
PK1_GLOBAL_BF16 = (8, 512, 2048, "distinct", 0)
PK1_GLOBAL_TRACE_CASE = 1  # index into PK1_GLOBAL_CASES: (8, 512, 8192), starts rewritten in place
PK1_SWA_CASES = [(32, 128), (8, 512), (2, 1024)]  # (B, S) x tails in {distinct, shared}
PK1_SWA_START = 2048  # common start a of every pk1 SWA case (the tail is positions [a - 128, a))
if QUICK:
    PK0_CASES = [(1024, 2), (64, 2)]
    PK1_GLOBAL_CASES = [(2, 1024, 2048, "distinct", 0), (8, 512, 8192, "distinct", 1)]
    PK1_SWA_CASES = [(8, 512)]

CLASH_RE = re.compile(r"region ends at (\d+)")
L1_SMALL_BYTES = 32768


# ======================================================================================================================
# host builders (pure torch)
# ======================================================================================================================
def to_batch_host(t: torch.Tensor, B: int, S: int) -> torch.Tensor:
    """Host model of the device view + CN transpose: ``[1, H, B*S, d] -> [B, H, S, d]``."""
    H, d = t.shape[1], t.shape[3]
    return t.reshape(H, B, S, d).transpose(0, 1).contiguous()


def from_batch_host(t: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_batch_host`: ``[B, H, S, d] -> [1, H, B*S, d]``."""
    B, H, S, d = t.shape
    return t.transpose(0, 1).reshape(1, H, B * S, d).contiguous()


class BlockPool:
    """Distinct random block ids of the real-size pool (block 0 = the zero null block, never handed out)."""

    def __init__(self, seed: int, n_blocks: int = N_BIG):
        g = torch.Generator().manual_seed(seed)
        self.free = (torch.randperm(n_blocks - 1, generator=g) + 1).tolist()

    def take(self, n: int):
        if n > len(self.free):
            raise ValueError(f"block pool exhausted: {n} wanted, {len(self.free)} left")
        out, self.free = self.free[:n], self.free[n:]
        return out


def pk1_page_tables(pool: BlockPool, B: int, S: int, a: int, *, prefix: str, dummies: int, width: int) -> torch.Tensor:
    """SDPA page tables ``[B, width]`` int32 of a pk1 pass at start ``a`` (sp1: the blocks of positions ``[0, a + S)``,
    0-padded). ``prefix="shared"``: blocks ``[0, a / 64)`` are one set shared by every real segment (a system prompt),
    else each segment has its own; every real segment owns its chunk's ``S / 64`` blocks. The last ``dummies`` segments
    copy segment 0's row (review edit R-E2: a pk1 dummy reads segment 0's read-only prefix and writes nothing)."""
    if a % BS or S % BS or not 0 <= dummies < B:
        raise ValueError(f"bad pk1 table request a={a} S={S} dummies={dummies} B={B}")
    nb_pre, nb_own = a // BS, S // BS
    if nb_pre + nb_own > width:
        raise ValueError(f"{nb_pre + nb_own} blocks do not fit a {width}-wide table")
    shared = pool.take(nb_pre) if prefix == "shared" else None
    rows = []
    for _ in range(B - dummies):
        row = torch.zeros(width, dtype=torch.int32)
        pre = shared if shared is not None else pool.take(nb_pre)
        row[:nb_pre] = torch.tensor(pre, dtype=torch.int32)
        row[nb_pre : nb_pre + nb_own] = torch.tensor(pool.take(nb_own), dtype=torch.int32)
        rows.append(row)
    rows += [rows[0].clone() for _ in range(dummies)]
    return torch.stack(rows)


def pk1_tail_ids(pool: BlockPool, B: int, *, tails: str, dummies: int = 0):
    """Per segment the two tail block ids (positions ``[a - 128, a)``): ``"shared"`` = one pair for every segment,
    ``"distinct"`` = each real segment its own pair; the last ``dummies`` segments copy segment 0's (R-E2)."""
    if tails == "shared":
        pair = tuple(pool.take(2))
        return [pair] * B
    out = [tuple(pool.take(2)) for _ in range(B - dummies)]
    return out + [out[0]] * dummies


def tail_variant(tail_ids) -> str:
    """Review edit R-E2: ``shared`` iff all B tail sets are identical, else ``distinct`` (no partial dedupe)."""
    return "shared" if all(t == tail_ids[0] for t in tail_ids) else "distinct"


def kv_expansion_host(seed: int) -> torch.Tensor:
    """A random prefill KV expansion ``E [512, 640]`` (bf16 values) with the layout of
    ``weights.prefill_kv_expansion_for_chip``: per group ``g`` columns ``[320 g, 320 g + 128)`` = W_UK (k_nope),
    ``[+128, +256)`` = W_UV (v), ``[+256, +320)`` = 0 (V padding to 192)."""
    g = torch.Generator().manual_seed(seed)
    E = torch.zeros(D_NOPE, EXP_W)
    for gi in range(NKV):
        c = 320 * gi
        E[:, c : c + 256] = torch.randn(D_NOPE, 256, generator=g) / D_NOPE**0.5
    return E.bfloat16().float()


def expand_host(lat: torch.Tensor, E: torch.Tensor):
    """fp32 model of the draft-1 expansion of a latent ``lat [R, 576]``: ``K [2, R, 192] = [n @ E_k(g) | k_pe]``,
    ``V [2, R, 128] = n @ E_v(g)``."""
    n, kpe = lat[:, :D_NOPE].float(), lat[:, D_NOPE:].float()
    kvx = n @ E
    K = torch.stack([torch.cat([kvx[:, 320 * g : 320 * g + 128], kpe], dim=-1) for g in range(NKV)])
    V = torch.stack([kvx[:, 320 * g + 128 : 320 * g + 256] for g in range(NKV)])
    return K, V


@torch.no_grad()
def window_golden(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q0: int, window: int = WINDOW) -> torch.Tensor:
    """fp32 sliding-window GQA attention (scale folded in q): rows ``q [NQ, R, d]`` at sequence indices ``q0 ..`` over
    ``k / v [NKV, L, d]``; row p sees keys ``[p - window + 1, p]``. Returns ``[NQ, R, dv]``."""
    nq, R, _ = q.shape
    rep = nq // k.shape[0]
    qp = torch.arange(q0, q0 + R)[:, None]
    kp = torch.arange(k.shape[1])[None, :]
    allow = (kp <= qp) & (kp >= qp - (window - 1))
    out = torch.empty(nq, R, v.shape[-1])
    for g in range(k.shape[0]):
        s = q[g * rep : (g + 1) * rep].float() @ k[g].float().T
        s = s.masked_fill(~allow[None], float("-inf"))
        out[g * rep : (g + 1) * rep] = torch.softmax(s, -1) @ v[g].float()
    return out


@torch.no_grad()
def latent_golden(q: torch.Tensor, kv: torch.Tensor, start: int) -> torch.Tensor:
    """fp32 resumed global attention (sp1, absorbed): ``q [NQ, S, 576]`` at positions ``start ..`` over the latent
    sequence ``kv [>= start + S, 576]`` (K = V = the latent, scale folded): row i sees keys ``[0, start + i]``."""
    S = q.shape[1]
    keys = kv[: start + S].float()
    s = q.float() @ keys.T
    mask = torch.arange(start + S)[None, :] > (start + torch.arange(S))[:, None]
    s = s.masked_fill(mask[None], float("-inf"))
    return torch.softmax(s, -1) @ keys


def pk0_inputs(kind: str, T: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    scale = gd.SCALE_SWA if kind == "swa" else gd.SCALE_GLOBAL
    q = (torch.randn(1, NQ, T, DQK, generator=g) * scale).bfloat16()
    k = torch.randn(1, NKV, T, DQK, generator=g).bfloat16()
    v = torch.zeros(1, NKV, T, DQK).bfloat16()
    v[..., :DV] = torch.randn(1, NKV, T, DV, generator=g).bfloat16()
    return q, k, v


def golden_segments(B: int):
    return sorted({0, B // 2, B - 1})


# ----------------------------------------------------------------------------------------------------------------------
# host self-checks
# ----------------------------------------------------------------------------------------------------------------------
def test_g15_host_view_transpose_model():
    """Segment k of the batched layout is rows [k S, (k + 1) S) of every head (the device view + CN transpose)."""
    B, S, H, d = 4, 64, 3, 32
    t = torch.randn(1, H, B * S, d)
    b = to_batch_host(t, B, S)
    assert b.shape == (B, H, S, d)
    for k in range(B):
        assert torch.equal(b[k], t[0, :, k * S : (k + 1) * S])
    assert torch.equal(from_batch_host(b), t)


def test_g15_host_pk1_tables_and_tails():
    pool = BlockPool(seed=1)
    pt = pk1_page_tables(pool, 32, 128, 24576, prefix="shared", dummies=2, width=640)
    assert pt.shape == (32, 640) and pt.dtype == torch.int32
    assert all(torch.equal(pt[b, :384], pt[0, :384]) for b in range(32))  # one system prompt
    own = [tuple(pt[b, 384:386].tolist()) for b in range(30)]
    assert len(set(own)) == 30 and all(min(o) >= 1 for o in own)  # every real segment its own chunk blocks
    assert torch.equal(pt[30], pt[0]) and torch.equal(pt[31], pt[0])  # dummies copy segment 0 (R-E2)
    assert int((pt[:, 386:] != 0).sum()) == 0 and int((pt[:, :386] == 0).sum()) == 0
    pt2 = pk1_page_tables(pool, 8, 512, 8192, prefix="distinct", dummies=1, width=640)
    pre = [tuple(pt2[b, :128].tolist()) for b in range(7)]
    assert len(set(pre)) == 7  # distinct prefixes
    used = set(pt.reshape(-1).tolist()) | set(pt2.reshape(-1).tolist())
    assert len(used - {0}) == 384 + 30 * 2 + 7 * 136  # every block handed out once
    d = pk1_tail_ids(pool, 32, tails="distinct", dummies=2)
    s = pk1_tail_ids(pool, 8, tails="shared")
    assert tail_variant(d) == "distinct" and tail_variant(s) == "shared"
    assert d[30] == d[0] and len(set(d[:30])) == 30
    # a pass whose real segments share one prefix stays "shared" with dummies copying segment 0 (R-E2)
    assert tail_variant([s[0]] * 6 + [s[0]] * 2) == "shared"


def test_g15_host_import_provenance():
    """Prints which motif3 package this process imports (a gate run imports it from a B0 + C1a snapshot; §13)."""
    import models.demos.motif3 as m3

    print(f"[{GATE}] motif3 package: {Path(m3.__file__).parent}; results: {REC.path}")
    assert (Path(m3.__file__).parent / "tt" / "model_config.py").is_file()


def test_g15_host_goldens_selfcheck():
    """The square [tail | chunk] layout with the window reproduces window attention at absolute positions; the
    latent golden is causal over [0, start + i]; the expansion model splits per group."""
    g = torch.Generator().manual_seed(3)
    S, a = 64, 256
    lat = torch.randn(a + S, D_LAT, generator=g)
    E = kv_expansion_host(seed=4)
    K, V = expand_host(lat, E)
    assert K.shape == (NKV, a + S, DQK) and V.shape == (NKV, a + S, DV)
    assert torch.equal(K[1, :, 128:], lat[:, D_NOPE:]) and torch.allclose(K[0, :, :128], lat[:, :D_NOPE] @ E[:, :128])
    q = torch.randn(NQ, S, DQK, generator=g) * 0.1
    full = window_golden(q, K, V, a)  # rows at absolute a .. a + S - 1 over the whole sequence
    Ks, Vs = K[:, a - TAIL : a + S], V[:, a - TAIL : a + S]  # the square: [tail 128 | chunk S]
    sq = window_golden(q, Ks, Vs, TAIL)
    assert torch.allclose(full, sq, atol=1e-5), (full - sq).abs().max()
    # latent golden: row 0 attends [0, a] only
    ql = torch.randn(NQ, S, D_LAT, generator=g) * 0.05
    out = latent_golden(ql, lat, a)
    want0 = torch.softmax(ql[:, 0] @ lat[: a + 1].T, -1) @ lat[: a + 1]
    assert torch.allclose(out[:, 0], want0, atol=1e-5)


# ======================================================================================================================
# device helpers
# ======================================================================================================================
def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _setup(mesh_device, tag: str):
    """The MotifTTConfig of the mesh (B0's builders) + a provenance record: which motif3 package is imported (the
    snapshot of B0 + C1a during the gate runs), the md5 of its C1a files, the fabric."""
    import models.demos.motif3 as m3
    from models.demos.motif3.tt import generator_api, model_config
    from models.demos.motif3.tt.ccl import log_fabric
    from models.demos.motif3.tt.model_config import MotifTTConfig

    cfg = MotifTTConfig.from_hf_config(None, mesh_device=mesh_device)
    try:
        head = subprocess.run(["git", "-C", str(model_config.PROJECT_ROOT / "tt-metal"), "rev-parse", "--short=11",
                               "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()  # fmt: skip
    except Exception:  # pragma: no cover
        head = "?"
    fab = log_fabric(mesh_device, f"G15 {tag}")
    REC.add(f"provenance/{tag}", status="info", motif3_package=str(Path(m3.__file__).parent), git_head=head,
            model_config_md5=_md5(model_config.__file__), generator_api_md5=_md5(generator_api.__file__),
            fabric_committed=fab.get("committed"), l1_small=fab.get("l1_small"), ring_gather=cfg.ring_gather,
            ring_gather_note="no CCL in the G15 ops", compute_grid=list(cfg.compute_grid))
    return cfg


def up(t: torch.Tensor, mesh_device, dtype=ttnn.bfloat16):
    return gu.to_mesh(t.contiguous(), mesh_device, dtype)


def up_i32(t: torch.Tensor, mesh_device):
    return gu.to_mesh(t.contiguous(), mesh_device, ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)


def host_i32(t: torch.Tensor, mesh_device):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))  # fmt: skip


def buf_addr(t) -> int:
    return int(ttnn.get_device_tensors(t)[0].buffer_address())


def bview(t, B: int, S: int):
    """``[1, H, B*S, d]`` -> view ``[H, B, S, d]`` (metadata only) -> CN transpose -> ``[B, H, S, d]`` (new buffer)."""
    H, d = int(t.shape[1]), int(t.shape[3])
    return ttnn.transpose(ttnn.reshape(t, (H, B, S, d)), 0, 1)


def unbview(t, *, consume: bool = True):
    """``[B, H, S, d]`` -> CN transpose -> ``[H, B, S, d]`` -> view ``[1, H, B*S, d]`` (shares the transposed buffer:
    free the returned view, never the transposed tensor separately)."""
    B, H, S, d = (int(x) for x in t.shape)
    tt = ttnn.transpose(t, 0, 1)
    if consume:
        ttnn.deallocate(t)
    return ttnn.reshape(tt, (1, H, B * S, d))


def views_metadata_only(mesh_device, t, B: int, S: int) -> bool:
    """Both views of the packed layout keep the buffer address (no data movement)."""
    H, d = int(t.shape[1]), int(t.shape[3])
    v = ttnn.reshape(t, (H, B, S, d))
    ok = buf_addr(v) == buf_addr(t)
    tb = ttnn.transpose(v, 0, 1)
    tt = ttnn.transpose(tb, 0, 1)
    back = ttnn.reshape(tt, (1, H, B * S, d))
    ok = ok and buf_addr(back) == buf_addr(tt)
    gu.free([tb, back])
    return bool(ok)


def l1_pin(mesh_device):
    """A one-page interleaved L1 tensor: allocated top-down, it sits just below the L1_SMALL region (track B,
    ``logs/dev/trackB_probes/probe_sp1_cbend.py``), so a static-CB region reaching it raises a clash naming its end."""
    return ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                           memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))


def run_pinned(mesh_device, fn):
    """``fn()`` once with :func:`l1_pin` alive (tt-metal re-validates static CBs on every enqueue against the lowest L1
    buffer). Returns ``(fn's output or None on a clash, record)``. A clash leaves the case to re-run ``fn`` unpinned."""
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
        end = int(m.group(1))
        rec = {"cb_check": "clash_with_pin", "pin_addr": addr, "cb_end": end, "error": str(e)[:300]}
    finally:
        ttnn.deallocate(pin)
    return out, rec


def cb_ok(rec: dict) -> bool:
    return rec.get("cb_check") == "fits_below_pin"


def readback(t, idx: int = 0) -> torch.Tensor:
    return gu.read_dev(t, idx)


def seg_rows(x: torch.Tensor, k: int, S: int) -> torch.Tensor:
    """Segment k of a packed ``[H, T, d]`` host tensor."""
    return x[:, k * S : (k + 1) * S]


def count_programs(mesh_device) -> int:
    ttnn.synchronize_device(mesh_device)
    return int(mesh_device.num_program_cache_entries())


def build_cache(mesh_device, ids, dtype, seed: int):
    """Real-size latent cache ``[4129, 1, 64, 576]``: block 0 and every unused block zero, blocks ``ids`` N(0, 1)
    pre-quantized to ``dtype`` (bfp8 values are exact in bf16, so the host copy is the device content). Returns
    ``(host bf16 copy, device tensor)``."""
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    ids = sorted(set(int(i) for i in ids))
    host = torch.zeros(N_BIG, 1, BS, D_LAT, dtype=torch.bfloat16)
    vals = torch.randn(len(ids), 1, BS, D_LAT, generator=g)
    vals = gu.host_roundtrip(vals, ttnn.bfloat8_b) if dtype == ttnn.bfloat8_b else vals.bfloat16().float()
    host[torch.tensor(ids, dtype=torch.long)] = vals.bfloat16()
    dev = gu.to_mesh(host, mesh_device, dtype)
    print(f"[{GATE}] cache [{N_BIG},1,64,576] {dtype} with {len(ids)} random blocks built in {time.time() - t0:.1f} s")
    return host, dev


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {str(e)[:400]}"


# ======================================================================================================================
# (a) pk0
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g15a_pk0_batched_sdpa(mesh_device):
    torch.set_num_threads(16)
    cfg = _setup(mesh_device, "pk0")
    ckc = cfg.compute_config("sdpa_prefill")
    failures = []
    for S, B in PK0_CASES:
        for kind in ("global", "swa"):
            case = f"pk0/{kind}/S{S}_B{B}"
            try:
                ok, why = _pk0_case(mesh_device, cfg, ckc, kind, S, B, case)
            except Exception as e:
                REC.add(case, status="error", error=_err(e))
                failures.append(f"{case}: {_err(e)}")
                continue
            if not ok:
                failures.append(f"{case}: {why}")
    try:
        _pk0_transposes_traced(mesh_device)
    except Exception as e:
        REC.add("pk0/transposes_traced", status="error", error=_err(e))
        failures.append(f"pk0/transposes_traced: {_err(e)}")
    assert not failures, "G15a pk0 failures:\n" + "\n".join(failures)


def _pk0_transposes_traced(mesh_device):
    """(d), review D6: traced device time of a layer's 4 CN transposes (q, k, v in, o out) at T = 2048 and 8192 (the
    eager numbers above are host-dispatch bound). Every program is compiled before the first capture."""
    shapes = [(256, 8), (1024, 8)]  # (S, B): T = 2048, 8192
    keep, fns = [], {}
    for S, B in shapes:
        q, k, v = pk0_inputs("global", B * S, seed=gu.seed_of("tr", S, B))
        q_t, k_t, v_t = (up(x, mesh_device) for x in (q, k, v))
        o_like = bview(q_t, B, S)
        keep += [q_t, k_t, v_t, o_like]

        def fn(q_t=q_t, k_t=k_t, v_t=v_t, o_like=o_like, B=B, S=S):
            return [bview(q_t, B, S), bview(k_t, B, S), bview(v_t, B, S), unbview(o_like, consume=False)]

        gu.free(fn())  # compile
        fns[(S, B)] = fn
    ttnn.synchronize_device(mesh_device)
    for (S, B), fn in fns.items():
        per_call, raw = gu.time_traced(mesh_device, fn, ops_per_trace=16, reps=7, compile_first=False)
        REC.add(f"pk0/transposes_traced/T{B * S}", status="info", S=S, B=B, T=B * S, traced_us=per_call,
                traced_raw_us=raw, per_pass_ms_53_layers=round(per_call * 53 / 1e3, 2),
                note="q [1,10,T,192], k / v [1,2,T,192] in, o [B,10,S,192] out: one attention layer")  # fmt: skip
    gu.free(keep)


def _pk0_case(mesh_device, cfg, ckc, kind: str, S: int, B: int, case: str):
    T = B * S
    window = WINDOW if (kind == "swa" and S >= WINDOW) else None  # MotifAttention.prefill_sdpa_window_and_config
    prog = cfg.sdpa_prefill_pc(kind, seq_len=S)
    q, k, v = pk0_inputs(kind, T, seed=gu.seed_of("pk0", kind, S, B))
    q_t, k_t, v_t = (up(x, mesh_device) for x in (q, k, v))

    def sdpa(qq, kk, vv):
        return ttnn.transformer.scaled_dot_product_attention(
            qq, kk, vv, is_causal=True, scale=1.0, sliding_window_size=window, program_config=prog,
            compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

    def packed():
        qb, kb, vb = bview(q_t, B, S), bview(k_t, B, S), bview(v_t, B, S)
        o = sdpa(qb, kb, vb)  # [B, 10, S, 192]
        gu.free([qb, kb, vb])
        return unbview(o)  # [1, 10, T, 192]

    n0 = count_programs(mesh_device)
    out, cb = run_pinned(mesh_device, packed)
    if out is None:
        out = packed()
    n1 = count_programs(mesh_device)
    got = readback(out)[0]  # [10, T, 192]
    rep_ok, rep_bad = gu.replicas_identical(out)
    gu.free(out)
    gu.free(packed())  # a second packed call: no new program
    n2 = count_programs(mesh_device)
    views_ok = views_metadata_only(mesh_device, q_t, B, S) and views_metadata_only(mesh_device, k_t, B, S)
    nonfinite = int((~torch.isfinite(got)).sum())

    # ---- per segment: the single-row SDPA [1, H, S, d] with the same program / compute config --------------------
    singles = [tuple(up(x[:, :, b * S : (b + 1) * S], mesh_device) for x in (q, k, v)) for b in range(B)]
    bit, max_abs = 0, 0.0
    for b in range(B):
        o = sdpa(*singles[b])
        ref = readback(o)[0]
        gu.free(o)
        seg = seg_rows(got, b, S)
        bit += int(torch.equal(seg, ref))
        max_abs = max(max_abs, float((seg - ref).abs().max()))
    pccs = {}
    for b in golden_segments(B):
        sl = slice(b * S, (b + 1) * S)
        want = gd.gqa_prefill_golden(q[0, :, sl].float(), k[0, :, sl].float(), v[0, :, sl, :DV].float(), 1.0, window)
        pccs[b] = gu.pcc(want, seg_rows(got, b, S)[..., :DV])
    worst = min(pccs.values())

    # ---- (d) eager cost: packed (4 transposes + SDPA) vs B single calls; the 4 transposes alone -------------------
    o_like = bview(q_t, B, S)
    t_packed = gu.time_eager(mesh_device, packed, iters=5, warmup=1) / 1e3
    t_single = gu.time_eager(mesh_device, lambda: [sdpa(*singles[b]) for b in range(B)], iters=3, warmup=1) / 1e3
    t_tr = gu.time_eager(
        mesh_device, lambda: [bview(q_t, B, S), bview(k_t, B, S), bview(v_t, B, S), unbview(o_like, consume=False)],
        iters=5, warmup=1) / 1e3  # fmt: skip
    gu.free([o_like, q_t, k_t, v_t] + [t for s in singles for t in s])

    pcc_ok = worst >= PCC_PK0 or (bit == B and worst >= PCC_PK0_FLOOR)
    ok = bit == B and pcc_ok and nonfinite == 0 and rep_ok and views_ok and cb_ok(cb) and n2 == n1
    REC.add(case, status="pass" if ok else "fail", kind=kind, S=S, B=B, T=T, q_k_chunk=[prog.q_chunk_size,
            prog.k_chunk_size], window=window, role="sdpa_prefill", segments_bitwise=f"{bit}/{B}",
            max_abs_vs_single=max_abs, pcc_vs_fp32={str(b): round(p, 6) for b, p in pccs.items()},
            worst_pcc=worst, design_bar_0p9997_met=worst >= PCC_PK0, pcc_note=None if worst >= PCC_PK0 else
            "the single-row op's own PCC (bitwise equal); G2 floor 0.99948", nonfinite=nonfinite,
            replicas_identical_32=rep_ok, replicas_bad=rep_bad, views_metadata_only=views_ok,
            programs_first_call=n1 - n0, programs_second_call=n2 - n1,
            eager_ms_packed=round(t_packed, 3), eager_ms_B_singles=round(t_single, 3),
            eager_ms_4_transposes=round(t_tr, 3), **cb)  # fmt: skip
    why = (f"bitwise {bit}/{B}, worst pcc {worst:.6f}, nonfinite {nonfinite}, replicas {rep_ok}, views {views_ok}, "
           f"cb {cb.get('cb_check')}, programs on the 2nd call {n2 - n1}")  # fmt: skip
    return ok, why


# ======================================================================================================================
# (b) pk1 global
# ======================================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g15b_pk1_global(mesh_device):
    torch.set_num_threads(16)
    cfg = _setup(mesh_device, "pk1_global")
    ckc = cfg.compute_config("sdpa_prefill_fp32")
    W = int(cfg.sp1_page_table_width)
    failures = []
    pool = BlockPool(seed=1500)
    tables = [pk1_page_tables(pool, B, S, a, prefix=p, dummies=d, width=W) for (B, S, a, p, d) in PK1_GLOBAL_CASES]
    used = set()
    for pt in tables:
        used |= set(pt.reshape(-1).tolist())
    used.discard(0)
    host, cache = build_cache(mesh_device, used, ttnn.bfloat8_b, seed=1501)
    keep = {}
    for i, ((B, S, a, prefix, dummies), pt) in enumerate(zip(PK1_GLOBAL_CASES, tables)):
        case = f"pk1_global/bfp8/B{B}_S{S}_a{a}_{prefix}"
        try:
            ok, why, kept = _pk1_global_case(mesh_device, cfg, ckc, cache, host, B, S, a, pt, dummies, case,
                                             keep_inputs=(i == PK1_GLOBAL_TRACE_CASE))  # fmt: skip
        except Exception as e:
            REC.add(case, status="error", error=_err(e))
            failures.append(f"{case}: {_err(e)}")
            continue
        if kept is not None:
            keep = kept
        if not ok:
            failures.append(f"{case}: {why}")

    # ---- traced replay with a rewritten start (G9 method): every compile happened above ---------------------------
    if keep:
        try:
            ok, why = _pk1_global_trace(mesh_device, keep)
            if not ok:
                failures.append(f"pk1_global/trace: {why}")
        except Exception as e:
            REC.add("pk1_global/trace_rewritten_start", status="error", error=_err(e))
            failures.append(f"pk1_global/trace: {_err(e)}")
        gu.free([keep["q_t"], keep["pt_t"], keep["st_t"]])
    gu.free(cache)
    del host

    # ---- bf16 latent cache (MOTIF3_KV_CACHE_DTYPE=bf16): the 64/64 table at every bucket --------------------------
    if not QUICK:
        B, S, a, prefix, dummies = PK1_GLOBAL_BF16
        pool = BlockPool(seed=1502)
        pt = pk1_page_tables(pool, B, S, a, prefix=prefix, dummies=dummies, width=W)
        used = set(pt.reshape(-1).tolist()) - {0}
        host, cache = build_cache(mesh_device, used, ttnn.bfloat16, seed=1503)
        case = f"pk1_global/bf16/B{B}_S{S}_a{a}_{prefix}"
        try:
            ok, why, _ = _pk1_global_case(mesh_device, cfg, ckc, cache, host, B, S, a, pt, dummies, case)
            if not ok:
                failures.append(f"{case}: {why}")
        except Exception as e:
            REC.add(case, status="error", error=_err(e))
            failures.append(f"{case}: {_err(e)}")
        gu.free(cache)
    assert not failures, "G15a pk1 global failures:\n" + "\n".join(failures)


def _pk1_global_case(mesh_device, cfg, ckc, cache, host, B, S, a, pt, dummies, case, keep_inputs=False):
    T = B * S
    prog = cfg.resumed_prefill_pc("global", S, kv_dtype=cache.dtype)
    qc, kc = int(prog.q_chunk_size), int(prog.k_chunk_size)
    if a % qc or a % kc:  # attention._check_sp1_global (G9: the kernels floor a misaligned start silently)
        raise ValueError(f"start {a} is not a multiple of the chunked SDPA's q / k chunks ({qc}, {kc})")
    g = torch.Generator().manual_seed(gu.seed_of("pk1g", B, S, a))
    q = (torch.randn(1, NQ, T, D_LAT, generator=g) * gd.SCALE_GLOBAL).bfloat16()
    q_t, pt_t = up(q, mesh_device), up_i32(pt, mesh_device)
    st_t = up_i32(torch.tensor([a], dtype=torch.int32), mesh_device)

    def chunked(qq, ptt):
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            qq, cache, cache, ptt, chunk_start_idx_tensor=st_t, scale=1.0, program_config=prog,
            compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

    def packed():
        qb = bview(q_t, B, S)  # [B, 10, S, 576]
        o = chunked(qb, pt_t)
        gu.free(qb)
        return unbview(o)  # [1, 10, T, 576]

    n0 = count_programs(mesh_device)
    out, cb = run_pinned(mesh_device, packed)
    if out is None:
        out = packed()
    n1 = count_programs(mesh_device)
    got = readback(out)[0]  # [10, T, 576]
    rep_ok, rep_bad = gu.replicas_identical(out)
    gu.free(out)
    gu.free(packed())
    n2 = count_programs(mesh_device)
    views_ok = views_metadata_only(mesh_device, q_t, B, S)
    real = B - dummies
    nonfinite = int((~torch.isfinite(got[:, : real * S])).sum())
    nonfinite_dummies = int((~torch.isfinite(got[:, real * S :])).sum())

    singles = [(up(q[:, :, b * S : (b + 1) * S], mesh_device), up_i32(pt[b : b + 1], mesh_device)) for b in range(real)]
    bit, max_abs = 0, 0.0
    for b in range(real):
        o = chunked(*singles[b])
        ref = readback(o)[0]
        gu.free(o)
        seg = seg_rows(got, b, S)
        bit += int(torch.equal(seg, ref))
        max_abs = max(max_abs, float((seg - ref).abs().max()))
    need = (a + S) // BS
    pccs = {}
    for b in sorted({0, real - 1}):
        kv = host[pt[b, :need].long(), 0].reshape(need * BS, D_LAT).float()
        want = latent_golden(q[0, :, b * S : (b + 1) * S].float(), kv, a)
        pccs[b] = gu.pcc(want[..., :D_NOPE], seg_rows(got, b, S)[..., :D_NOPE])
    worst = min(pccs.values())
    t_packed = gu.time_eager(mesh_device, packed, iters=3, warmup=1) / 1e3
    t_single = gu.time_eager(mesh_device, lambda: [chunked(*singles[b]) for b in range(real)], iters=2, warmup=1) / 1e3
    gu.free([t for s in singles for t in s])

    ok = (bit == real and worst >= PCC_PK1_GLOBAL and nonfinite == 0 and nonfinite_dummies == 0 and rep_ok
          and views_ok and cb_ok(cb) and n2 == n1)  # fmt: skip
    REC.add(case, status="pass" if ok else "fail", B=B, S=S, T=T, start=a, real_segments=real, dummies=dummies,
            cache_dtype=str(cache.dtype), q_k_chunk=[qc, kc], role="sdpa_prefill_fp32",
            page_table_width=int(pt.shape[1]),
            segments_bitwise=f"{bit}/{real}", max_abs_vs_single=max_abs,
            pcc_vs_fp32_cols512={str(b): round(p, 7) for b, p in pccs.items()}, worst_pcc=worst, nonfinite=nonfinite,
            nonfinite_dummies=nonfinite_dummies, replicas_identical_32=rep_ok, replicas_bad=rep_bad,
            views_metadata_only=views_ok, programs_first_call=n1 - n0, programs_second_call=n2 - n1,
            eager_ms_packed=round(t_packed, 3), eager_ms_singles=round(t_single, 3), **cb)  # fmt: skip
    why = (f"bitwise {bit}/{real}, worst pcc {worst:.7f}, nonfinite {nonfinite}/{nonfinite_dummies}, replicas "
           f"{rep_ok}, views {views_ok}, cb {cb.get('cb_check')}, programs on the 2nd call {n2 - n1}")  # fmt: skip
    kept = None
    if keep_inputs:
        kept = dict(q_t=q_t, pt_t=pt_t, st_t=st_t, cache=cache, prog=prog, ckc=ckc, B=B, S=S, real=real)
    else:
        gu.free([q_t, pt_t, st_t])
    return ok, why, kept


def _pk1_global_trace(mesh_device, k):
    """Capture the packed pk1 global call at start 8192, replay with the start tensor rewritten in place (2048, 4096,
    8192): bitwise equal to eager at each start, no program compiled after the capture (G9's method)."""
    B, S, real = k["B"], k["S"], k["real"]

    def packed():
        qb = bview(k["q_t"], B, S)
        o = ttnn.transformer.chunked_scaled_dot_product_attention(
            qb, k["cache"], k["cache"], k["pt_t"], chunk_start_idx_tensor=k["st_t"], scale=1.0,
            program_config=k["prog"], compute_kernel_config=k["ckc"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        gu.free(qb)
        return unbview(o)

    def set_start(a):
        ttnn.copy_host_to_device_tensor(host_i32(torch.tensor([a], dtype=torch.int32), mesh_device), k["st_t"])

    eager = {}
    for a in TRACE_STARTS:  # eager references first (same program at every start: nothing new to compile)
        set_start(a)
        o = packed()
        eager[a] = readback(o)[0]
        gu.free(o)
    n0 = count_programs(mesh_device)
    set_start(TRACE_START_CAPTURE)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        tout = packed()
    except BaseException:
        try:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        finally:
            ttnn.release_trace(mesh_device, tid)
        raise
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    res = {}
    try:
        for a in TRACE_STARTS:
            set_start(a)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            got = readback(tout)[0]
            res[str(a)] = bool(torch.equal(got[:, : real * S], eager[a][:, : real * S]))
    finally:
        ttnn.release_trace(mesh_device, tid)
    gu.free(tout)
    n1 = count_programs(mesh_device)
    ok = all(res.values()) and n1 == n0
    REC.add("pk1_global/trace_rewritten_start", status="pass" if ok else "fail", captured_at=TRACE_START_CAPTURE,
            replay_bitwise_eq_eager=res, programs_during_capture_and_replay=n1 - n0, B=B, S=S)  # fmt: skip
    return ok, f"replay vs eager {res}, programs {n1 - n0}"


# ======================================================================================================================
# (c) pk1 SWA
# ======================================================================================================================
class TailBounds:
    """Persistent tensor-args slice bounds (``[blk, 0, 0, 0]`` / ``[blk + 1, 1, 64, 576]`` int32) of ``n`` tail blocks:
    one program serves every block id (G10a); :meth:`set` rewrites them in place (trace-safe)."""

    def __init__(self, mesh_device, n: int):
        self.mesh = mesh_device
        z = torch.zeros(4, dtype=torch.int32)
        self.pairs = [(up_i32(z, mesh_device), up_i32(z, mesh_device)) for _ in range(n)]

    def set(self, blocks):
        if len(blocks) != len(self.pairs):
            raise ValueError(f"{len(blocks)} blocks for {len(self.pairs)} bound pairs")
        for (s, e), blk in zip(self.pairs, blocks):
            ttnn.copy_host_to_device_tensor(host_i32(torch.tensor([blk, 0, 0, 0], dtype=torch.int32), self.mesh), s)
            ttnn.copy_host_to_device_tensor(host_i32(torch.tensor([blk + 1, 1, BS, D_LAT], dtype=torch.int32),
                                                     self.mesh), e)  # fmt: skip

    def free(self):
        gu.free([t for p in self.pairs for t in p])


def _slice_block(cache, pair):
    return ttnn.slice(cache, pair[0], pair[1], slice_dim=0, num_devices=int(cache.shape[0]),
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip


def _gather_tail_solo(cache, pairs):
    """``MotifAttention._gather_tail``: 2 block slices -> concat(dim 2) in the cache dtype -> bf16
    ``[1, 1, 128, 576]``."""
    parts = [_slice_block(cache, p) for p in pairs]
    cat = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gu.free(parts)
    if cat.dtype == ttnn.bfloat16:
        return cat
    out = ttnn.typecast(cat, ttnn.bfloat16)
    gu.free(cat)
    return out


def _expand_and_square(lat, q_rows, w_exp, *, ckc_mm, prog, ckc_sdpa, B: int, S: int):
    """The sp1 SWA tail of ``MotifAttention._prefill_sp1_swa`` after the latent concat, at batch ``B``:
    ``lat [B, 1, 128 + S, 576]`` -> draft-1 expansion -> ``K, V_pad [B, 2, 128 + S, 192]``;
    ``Q_cat = [q_rows[:, :, :128] | q_rows]`` -> causal + window-129 SDPA -> rows ``[128, 128 + S)``
    ``[B, 10, S, 128]``. Consumes ``lat``. Returns ``(o_v, (q_cat, k_full, v_pad))``: the SDPA inputs too (the caller
    frees them)."""
    n_cat, kpe_cat = ttnn.experimental.nlp_create_q_heads_split(lat, num_heads=1, split_head_dim=D_NOPE)
    gu.free(lat)
    kvx = ttnn.linear(n_cat, w_exp, dtype=ttnn.bfloat16, compute_kernel_config=ckc_mm,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, 1, 128 + S, 640]; in1 not batched  # fmt: skip
    gu.free(n_cat)
    k_nope, v_pad = ttnn.experimental.nlp_create_q_heads_split(kvx, num_heads=NKV, split_head_dim=128)
    gu.free(kvx)
    kpe_g = ttnn.repeat(kpe_cat, ttnn.Shape([1, NKV, 1, 1]))
    gu.free(kpe_cat)
    k_full = ttnn.concat([k_nope, kpe_g], dim=-1)  # [B, 2, 128 + S, 192]
    gu.free([k_nope, kpe_g])
    q_pad = ttnn.slice(q_rows, [0, 0, 0, 0], [B, NQ, TAIL, DQK])
    q_cat = ttnn.concat([q_pad, q_rows], dim=2)  # [B, 10, 128 + S, 192]
    if buf_addr(q_pad) != buf_addr(q_rows):  # S = 128: a full-extent slice is a no-op returning q_rows (slice.cpp)
        gu.free(q_pad)
    o = ttnn.transformer.scaled_dot_product_attention(
        q_cat, k_full, v_pad, is_causal=True, scale=1.0, sliding_window_size=WINDOW, program_config=prog,
        compute_kernel_config=ckc_sdpa, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    o_v = ttnn.slice(o, [0, 0, TAIL, 0], [B, NQ, TAIL + S, DV])
    gu.free(o)
    return o_v, (q_cat, k_full, v_pad)


class Pk1SwaCase:
    """Device inputs and the two dataflows of one pk1 SWA case (B segments of S rows at one start)."""

    def __init__(self, mesh_device, cfg, cache, host_cache, w_exp_t, B: int, S: int, tails: str, ids1, ids2, seed):
        self.mesh, self.cache, self.B, self.S, self.tails = mesh_device, cache, B, S, tails
        self.T = B * S
        self.ids = {1: ids1, 2: ids2}
        if tail_variant(ids1) != tails or tail_variant(ids2) != tails:
            raise ValueError(f"tail ids do not match the {tails} variant")
        self.prog = cfg.resumed_prefill_pc("swa", S)
        self.ckc_sdpa = cfg.compute_config("sdpa_prefill")  # never fp32 acc with a window
        self.ckc_mm = cfg.compute_config("attn_heads")
        self.w_exp = w_exp_t
        g = torch.Generator().manual_seed(seed)
        self.q = (torch.randn(1, NQ, self.T, DQK, generator=g) * gd.SCALE_SWA).bfloat16()
        self.kv = torch.randn(1, 1, self.T, D_LAT, generator=g).bfloat16()  # the chunk rows' latent [n | k_pe]
        self.q_t, self.kv_t = up(self.q, mesh_device), up(self.kv, mesh_device)
        # bounds: distinct = one pair per (segment, block) = 2B; shared = one pair of 2 (the solo gather, then repeat)
        self.bounds = TailBounds(mesh_device, 2 * B if tails == "distinct" else 2)
        self.host_cache = host_cache
        self.set_tails(1)
        self.seg_q = [up(self.q[:, :, b * S : (b + 1) * S], mesh_device) for b in range(B)]
        self.seg_kv = [up(self.kv[:, :, b * S : (b + 1) * S], mesh_device) for b in range(B)]

    def set_tails(self, which: int):
        ids = self.ids[which]
        self.cur = which
        flat = [blk for pair in ids for blk in pair] if self.tails == "distinct" else list(ids[0])
        self.bounds.set(flat)

    def seg_pairs(self, b: int):
        if self.tails == "distinct":
            return self.bounds.pairs[2 * b : 2 * b + 2]
        return self.bounds.pairs

    def packed(self, *, keep_sdpa_inputs: bool = False):
        """The batched dataflow -> ``[1, 10, T, 128]`` (+ the SDPA inputs when asked)."""
        B, S = self.B, self.S
        if self.tails == "shared":
            tail1 = _gather_tail_solo(self.cache, self.bounds.pairs)  # [1, 1, 128, 576] bf16
            tail_b = ttnn.repeat(tail1, ttnn.Shape([B, 1, 1, 1]))  # [B, 1, 128, 576]
            gu.free(tail1)
        else:
            parts = [_slice_block(self.cache, p) for p in self.bounds.pairs]  # 2B x [1, 1, 64, 576]
            cat = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, 128 B, 576]
            gu.free(parts)
            view = ttnn.reshape(cat, (B, 1, TAIL, D_LAT))  # metadata view
            tail_b = ttnn.typecast(view, ttnn.bfloat16) if cat.dtype != ttnn.bfloat16 else ttnn.clone(view)
            gu.free(cat)
        kv_b = ttnn.reshape(self.kv_t, (B, 1, S, D_LAT))  # metadata view of the chunk latents
        lat = ttnn.concat([tail_b, kv_b], dim=2)  # [B, 1, 128 + S, 576]
        gu.free(tail_b)
        qb = bview(self.q_t, B, S)  # [B, 10, S, 192]
        o_v, sd = _expand_and_square(lat, qb, self.w_exp, ckc_mm=self.ckc_mm, prog=self.prog,
                                     ckc_sdpa=self.ckc_sdpa, B=B, S=S)  # fmt: skip
        gu.free(qb)
        out = unbview(o_v)  # [1, 10, T, 128]
        if keep_sdpa_inputs:
            return out, sd
        gu.free(list(sd))
        return out

    def solo(self, b: int):
        """Segment b through the solo sp1 SWA dataflow (``MotifAttention._prefill_sp1_swa`` ops at batch 1)."""
        tail = _gather_tail_solo(self.cache, self.seg_pairs(b))
        lat = ttnn.concat([tail, self.seg_kv[b]], dim=2)  # [1, 1, 128 + S, 576]
        gu.free(tail)
        o_v, sd = _expand_and_square(lat, self.seg_q[b], self.w_exp, ckc_mm=self.ckc_mm, prog=self.prog,
                                     ckc_sdpa=self.ckc_sdpa, B=1, S=self.S)  # fmt: skip
        gu.free(list(sd))
        return o_v  # [1, 10, S, 128]

    def golden(self, b: int, E: torch.Tensor) -> torch.Tensor:
        """fp32 window attention at absolute positions for segment b: keys ``[tail (cache) | chunk]`` expanded."""
        ids = self.ids[self.cur][b]
        tail = torch.cat([self.host_cache[blk, 0] for blk in ids], dim=0).float()  # [128, 576]
        lat = torch.cat([tail, self.kv[0, 0, b * self.S : (b + 1) * self.S].float()], dim=0)
        K, V = expand_host(lat, E)
        return window_golden(self.q[0, :, b * self.S : (b + 1) * self.S].float(), K, V, TAIL)

    def free(self):
        gu.free([self.q_t, self.kv_t] + self.seg_q + self.seg_kv)
        self.bounds.free()


@pytest.mark.parametrize("mesh_device, device_params", [MESH], indirect=True, ids=["4x8_serving"])
def test_g15c_pk1_swa(mesh_device):
    torch.set_num_threads(16)
    cfg = _setup(mesh_device, "pk1_swa")
    failures = []
    pool = BlockPool(seed=1600)
    plan = []
    for B, S in PK1_SWA_CASES:
        for tails in ("distinct", "shared"):
            dummies = 2 if (tails == "distinct" and B == 32) else 0
            ids1 = pk1_tail_ids(pool, B, tails=tails, dummies=dummies)
            ids2 = pk1_tail_ids(pool, B, tails=tails, dummies=dummies)
            plan.append((B, S, tails, dummies, ids1, ids2))
    used = {blk for (*_, i1, i2) in plan for ids in (i1, i2) for pair in ids for blk in pair}
    host, cache = build_cache(mesh_device, used, ttnn.bfloat8_b, seed=1601)
    E = kv_expansion_host(seed=1602)
    w_exp_t = up(E, mesh_device, ttnn.bfloat16)  # cfg.dtypes.attention

    cases = {}
    for B, S, tails, dummies, ids1, ids2 in plan:
        case = f"pk1_swa/{tails}/B{B}_S{S}"
        try:
            c = Pk1SwaCase(mesh_device, cfg, cache, host, w_exp_t, B, S, tails, ids1, ids2,
                           seed=gu.seed_of("pk1swa", B, S, tails))  # fmt: skip
            ok, why = _pk1_swa_case(mesh_device, c, E, dummies, case)
            cases[case] = c
        except Exception as e:
            REC.add(case, status="error", error=_err(e))
            failures.append(f"{case}: {_err(e)}")
            continue
        if not ok:
            failures.append(f"{case}: {why}")

    # ---- window probes on the batched square SDPA (G10's probes, one probe set per segment) ------------------------
    for B, S in PK1_SWA_CASES:
        case = f"pk1_swa/probes/B{B}_S{S}"
        try:
            ok, why = _pk1_swa_probes(mesh_device, cfg, B, S, case)
        except Exception as e:
            REC.add(case, status="error", error=_err(e))
            failures.append(f"{case}: {_err(e)}")
            continue
        if not ok:
            failures.append(f"{case}: {why}")

    # ---- traced replay with rewritten tail bounds vs eager (every program compiled above) --------------------------
    for case, c in cases.items():
        try:
            ok, why = _pk1_swa_trace(mesh_device, c, case + "/trace_rewritten_tails")
            if not ok:
                failures.append(f"{case}/trace: {why}")
        except Exception as e:
            REC.add(case + "/trace_rewritten_tails", status="error", error=_err(e))
            failures.append(f"{case}/trace: {_err(e)}")
    for c in cases.values():
        c.free()
    gu.free([w_exp_t, cache])
    assert not failures, "G15a pk1 SWA failures:\n" + "\n".join(failures)


def _pk1_swa_case(mesh_device, c: Pk1SwaCase, E, dummies: int, case: str):
    B, S, T = c.B, c.S, c.T
    real = B - dummies
    n0 = count_programs(mesh_device)
    out, cb = run_pinned(mesh_device, c.packed)
    if out is None:
        out = c.packed()
    n1 = count_programs(mesh_device)
    got = readback(out)[0]  # [10, T, 128]
    rep_ok, rep_bad = gu.replicas_identical(out)
    gu.free(out)
    c.set_tails(2)  # other tail ids: the program cache must stay constant (tensor-args slices, shape-keyed concat)
    out2 = c.packed()
    n2 = count_programs(mesh_device)
    got2 = readback(out2)[0]
    gu.free(out2)
    c.set_tails(1)
    nonfinite = int((~torch.isfinite(got[:, : real * S])).sum()) + int((~torch.isfinite(got2[:, : real * S])).sum())
    nonfinite_dummies = int((~torch.isfinite(got[:, real * S :])).sum())

    # views: kv_row [1,1,T,576] -> [B,1,S,576]; the 2B-block concat [1,1,128B,576] -> [B,1,128,576]
    kv_v = ttnn.reshape(c.kv_t, (B, 1, S, D_LAT))
    views_ok = buf_addr(kv_v) == buf_addr(c.kv_t)
    if c.tails == "distinct":
        parts = [_slice_block(c.cache, p) for p in c.bounds.pairs]
        cat = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        views_ok = views_ok and buf_addr(ttnn.reshape(cat, (B, 1, TAIL, D_LAT))) == buf_addr(cat)
        gu.free(parts + [cat])
    views_ok = views_ok and views_metadata_only(mesh_device, c.q_t, B, S)

    # ---- per segment: the solo sp1 SWA dataflow (tails set 1) ---------------------------------------------------
    bit, max_abs = 0, 0.0
    for b in range(real):
        o = c.solo(b)
        ref = readback(o)[0]
        gu.free(o)
        seg = seg_rows(got, b, S)
        bit += int(torch.equal(seg, ref))
        max_abs = max(max_abs, float((seg - ref).abs().max()))
    # ---- the batched SDPA alone vs single SDPA calls on its own batch slices (isolates the SDPA batching) ----------
    out, (q_cat, k_full, v_pad) = c.packed(keep_sdpa_inputs=True)
    gu.free(out)
    sd_bit = 0
    o_b = ttnn.transformer.scaled_dot_product_attention(
        q_cat, k_full, v_pad, is_causal=True, scale=1.0, sliding_window_size=WINDOW, program_config=c.prog,
        compute_kernel_config=c.ckc_sdpa, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    batched = readback(o_b)  # [B, 10, 128 + S, 192]
    gu.free(o_b)
    for b in range(real):
        sl = [ttnn.slice(x, [b, 0, 0, 0], [b + 1, int(x.shape[1]), int(x.shape[2]), int(x.shape[3])])
              for x in (q_cat, k_full, v_pad)]
        o1 = ttnn.transformer.scaled_dot_product_attention(
            *sl, is_causal=True, scale=1.0, sliding_window_size=WINDOW, program_config=c.prog,
            compute_kernel_config=c.ckc_sdpa, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        sd_bit += int(torch.equal(readback(o1)[0], batched[b]))
        gu.free(sl + [o1])
    gu.free([q_cat, k_full, v_pad])

    pccs = {}
    for b in sorted({0, real - 1}):
        pccs[b] = gu.pcc(c.golden(b, E), seg_rows(got, b, S))
    worst = min(pccs.values())
    t_packed = gu.time_eager(mesh_device, c.packed, iters=3, warmup=1) / 1e3
    t_solo = gu.time_eager(mesh_device, lambda: [c.solo(b) for b in range(B)], iters=2, warmup=1) / 1e3

    ok = (bit == real and sd_bit == real and worst >= PCC_PK1_SWA and nonfinite == 0 and nonfinite_dummies == 0
          and rep_ok and views_ok and cb_ok(cb) and n2 == n1)  # fmt: skip
    REC.add(case, status="pass" if ok else "fail", B=B, S=S, T=T, start=PK1_SWA_START, tails=c.tails,
            real_segments=real, dummies=dummies, tail_blocks_gathered=len(c.bounds.pairs),
            q_k_chunk=[c.prog.q_chunk_size, c.prog.k_chunk_size], role="sdpa_prefill", window=WINDOW,
            segments_bitwise_vs_solo_dataflow=f"{bit}/{real}", max_abs_vs_solo=max_abs,
            sdpa_only_segments_bitwise=f"{sd_bit}/{real}", pcc_vs_fp32_window={str(b): round(p, 6) for b, p in
            pccs.items()}, worst_pcc=worst, nonfinite=nonfinite, nonfinite_dummies=nonfinite_dummies,
            replicas_identical_32=rep_ok, replicas_bad=rep_bad, views_metadata_only=views_ok,
            programs_first_call=n1 - n0, programs_after_tail_rewrite=n2 - n1, eager_ms_packed=round(t_packed, 3),
            eager_ms_B_solo=round(t_solo, 3), **cb)  # fmt: skip
    why = (f"bitwise vs solo {bit}/{real}, sdpa-only {sd_bit}/{real}, worst pcc {worst:.6f}, nonfinite "
           f"{nonfinite}/{nonfinite_dummies}, replicas {rep_ok}, views {views_ok}, cb {cb.get('cb_check')}, programs "
           f"after the tail rewrite {n2 - n1}")  # fmt: skip
    return ok, why


def _pk1_swa_probes(mesh_device, cfg, B: int, S: int, case: str):
    """G10's square-layout probes, one independent set per segment (rows {0, 1, 63, 64, 127, S - 1}): key p - 128
    attended (+1), p - 129 not (-1 would leak), p + 1 not (+3 would leak), |delta| <= 1e-3, at batch B."""
    from models.demos.motif3.tests.unit.gates import test_g10_swa_tail as g10

    rows = g10.probe_rows(S)
    qs, ks, vs, dims = [], [], [], None
    for b in range(B):
        q, k, v, dims = g10.probe_square(S, rows, seed=gu.seed_of("probe", S, b))
        qs.append(q)
        ks.append(k)
        vs.append(v)
    q_t, k_t, v_t = (up(torch.stack(x), mesh_device) for x in (qs, ks, vs))  # [B, H, 128 + S, 192]
    o = ttnn.transformer.scaled_dot_product_attention(
        q_t, k_t, v_t, is_causal=True, scale=1.0, sliding_window_size=WINDOW,
        program_config=cfg.resumed_prefill_pc("swa", S), compute_kernel_config=cfg.compute_config("sdpa_prefill"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
    got = readback(o)  # [B, 10, 128 + S, 192]
    gu.free([o, q_t, k_t, v_t])
    bad = []
    for b in range(B):
        verdicts = g10.probe_verdict(got[b, :, TAIL:, :DV], rows, dims)
        bad += [(b,) + tuple(v) for v in verdicts if v[3] != "ok"]
    ok = not bad
    REC.add(case, status="pass" if ok else "fail", B=B, S=S, probe_rows=rows, segments=B, failing=bad[:20])
    return ok, f"failing probes {bad[:8]}"


def _pk1_swa_trace(mesh_device, c: Pk1SwaCase, case: str):
    """Capture the batched dataflow at tail set 1, replay with the 2B (or 2) bounds rewritten to set 2 and back:
    bitwise equal to eager at each set; no program compiled after the capture."""
    eager = {}
    for which in (2, 1):  # eager references first (no new programs: the shapes ran above)
        c.set_tails(which)
        o = c.packed()
        eager[which] = readback(o)[0]
        gu.free(o)
    n0 = count_programs(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)  # bounds at set 1
    try:
        tout = c.packed()
    except BaseException:
        try:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        finally:
            ttnn.release_trace(mesh_device, tid)
        raise
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    res = {}
    try:
        for which in (2, 1):
            c.set_tails(which)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            res[f"set{which}"] = bool(torch.equal(readback(tout)[0], eager[which]))
    finally:
        ttnn.release_trace(mesh_device, tid)
    gu.free(tout)
    n1 = count_programs(mesh_device)
    ok = all(res.values()) and n1 == n0
    REC.add(case, status="pass" if ok else "fail", replay_bitwise_eq_eager=res,
            programs_during_capture_and_replay=n1 - n0, tails=c.tails, B=c.B, S=c.S)  # fmt: skip
    return ok, f"replay vs eager {res}, programs {n1 - n0}"
