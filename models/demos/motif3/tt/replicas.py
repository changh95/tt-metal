# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DESIGN-2 (phase E): EP32 + R replica slots per chip with a per-step greedy expert assignment on device
(``MOTIF3_MOE_REPLICAS``; design ``logs/opt/phaseD/DESIGN-2``, build ``logs/opt/phaseE/DESIGN2``).

Decode only. Every chip keeps its 12 native experts (EP32, ``weights.ep_layout``) and holds R more experts of other
chips in "replica slots". Each step and MoE layer, every chip runs the same deterministic greedy assignment on the
same inputs (the gathered rows' top-8 ids and live-lane mask, identical on all chips): each active expert is computed
by exactly one of its holders, the less loaded one, so the busiest chip's expert count drops (c = 32: 8.79 -> 6.76
experts on real routes [E, sim]). The combine is unchanged (the routed partials are summed over all 32 chips), so an
expert's contribution simply lands in another chip's partial: **not bitwise equal** to the plain EP32 path (another
summation order), and a row's result depends on the other rows of the step (no T32 = T64 row identity: keep it off for
MTP).

Replica choice (:func:`choose_replicas`): every home chip h donates its R most-routed experts (real routes, ties: the
lower id) to the chips ``h + 1 + j * (31 // R)`` (mod 32), j = 0..R-1, so every chip receives exactly R replicas, one
per slot j (the "spread" pattern of the DESIGN-2 sim: popularity barely generalizes to held-out prompts, the dynamic
assignment does the balancing).

Replica weights (:func:`build_replica_tensorbin`): a pure host transform of the TT cache. A cached EP32 expert tensor
(``[1, 12, K, N]`` bfp8 TILE per chip) is a flatbuffer header and the 32 shards in mesh row-major order; each expert is
a contiguous run of ``KT * NT`` bfp8 tiles inside its home shard. A replica tensorbin is the header of a ``[1, R, K, N]``
per-chip tensor (built once per process with ttnn from zeros) followed by byte copies of those runs: no dequantize /
requantize, the replica tiles are the cached bytes. The file goes to a work directory (default tmpfs, deleted after the
upload unless kept; resumable per layer when kept: a complete file with a matching sidecar is reused).

Assignment (host mirror :func:`assign`, device ``kernels/router_topk.py`` replica mode):
  1. active[e]: some live row routes to e;
  2. load[c] = active experts without a replica whose home is c;
  3. the active replicated experts in ascending id order go to the less loaded of (home, replica chip), ties: home;
  4. up to ``passes`` sweeps (ascending id) move an expert to its other holder when that lowers the pair's maximum
     (load[other] + 1 < load[current]).
A chip computes native slot j (expert 12 c + j) iff it is active and assigned to c (always, without a replica), and
replica slot j (expert ``slots[c][j]``) iff active and assigned to c.

Per-chip table (``[1, 1, 1, TABLE_WORDS]`` uint32 ROW_MAJOR): words ``[0, 384)`` the replica code of every expert
(``chip << 8 | slot``, ``NONE`` without a replica), ``[384, 400)`` the chip's 16 slot experts (12 natives, R replicas).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

N_EXPERTS = 384
N_CHIPS = 32
PER_CHIP = 12
DEFAULT_R = 4
NONE = 0xFFFF  # (fits the uint32 tables as a small positive value; codes are < 32 << 8)
TABLE_WORDS = 400  # 384 replica codes + 16 slot experts
SLOT_WORDS = 16
ASSIGN_WORDS = 416  # device debug output: where[384] (chip or NONE), load[32]
PASSES = 2
REPLICA_MODES = ("off", "r4")
BFP8_TILE_BYTES = 1088
DEFAULT_WORK_DIR = "/dev/shm/motif3_replicas"
PLAN_FORMAT = "motif3-moe-replica-plan/1"


# =====================================================================================================================
# replica choice and the assignment (pure python)
# =====================================================================================================================
def spread_targets(home: int, R: int, n_chips: int = N_CHIPS) -> List[int]:
    """The chips that hold replica slot j = 0..R-1 of home chip ``home``'s donated experts."""
    step = max(1, (int(n_chips) - 1) // int(R))
    return [(int(home) + 1 + j * step) % int(n_chips) for j in range(int(R))]


def choose_replicas(freq: Sequence[float], R: int = DEFAULT_R, *, per: int = PER_CHIP,
                    n_chips: int = N_CHIPS) -> List[List[int]]:
    """``freq [n_chips * per]`` (routed-token counts) -> ``slots[c][j]``: the global expert held in replica slot j of
    chip c. Home chip h donates its R most-routed experts (ties: lower id), expert of rank j to ``spread_targets(h)[j]``
    (a bijection over h for every j, so every chip receives exactly R)."""
    if len(freq) != per * n_chips:
        raise ValueError(f"freq must have {per * n_chips} entries, got {len(freq)}")
    if not 1 <= R <= per or R > n_chips - 1:
        raise ValueError(f"R = {R} outside [1, {min(per, n_chips - 1)}]")
    slots: List[List[Optional[int]]] = [[None] * R for _ in range(n_chips)]
    for h in range(n_chips):
        local = sorted(range(h * per, (h + 1) * per), key=lambda e: (-float(freq[e]), e))
        for j, t in enumerate(spread_targets(h, R, n_chips)):
            if slots[t][j] is not None:
                raise AssertionError("spread_targets is not a bijection")
            slots[t][j] = local[j]
    return [[int(e) for e in row] for row in slots]


def rep_codes(slots: Sequence[Sequence[int]], *, n_experts: int = N_EXPERTS) -> List[int]:
    """``slots[c][j]`` -> ``code[e]`` = ``c << 8 | j`` for a replicated expert, else :data:`NONE`."""
    code = [NONE] * n_experts
    for c, row in enumerate(slots):
        for j, e in enumerate(row):
            if code[e] != NONE:
                raise ValueError(f"expert {e} has more than one replica")
            code[e] = (c << 8) | j
    return code


def assign(active: Sequence[bool], code: Sequence[int], *, per: int = PER_CHIP, n_chips: int = N_CHIPS,
           passes: int = PASSES) -> Tuple[Dict[int, int], List[int]]:
    """Host mirror of the device assignment (module docstring): ``active[e]`` and the replica codes -> ``(where, load)``
    with ``where[e]`` = the chip that computes active expert e and ``load[c]`` = its active expert count."""
    load = [0] * n_chips
    where: Dict[int, int] = {}
    flex = []
    for e, a in enumerate(active):
        if not a:
            continue
        if code[e] == NONE:
            load[e // per] += 1
            where[e] = e // per
        else:
            flex.append(e)
    for e in flex:
        h, r = e // per, code[e] >> 8
        c = h if load[h] <= load[r] else r
        load[c] += 1
        where[e] = c
    for _ in range(int(passes)):
        moved = False
        for e in flex:
            h, r = e // per, code[e] >> 8
            c = where[e]
            o = r if c == h else h
            if load[o] + 1 < load[c]:
                load[c] -= 1
                load[o] += 1
                where[e] = o
                moved = True
        if not moved:
            break
    return where, load


def active_from_rows(idx_rows: Sequence[Sequence[int]], live: Optional[Sequence[bool]] = None,
                     n_experts: int = N_EXPERTS) -> List[bool]:
    """``idx_rows [M][K]`` (top-K ids per gathered row), ``live [M]`` (None: all) -> ``active[e]``."""
    act = [False] * n_experts
    for t, row in enumerate(idx_rows):
        if live is not None and not live[t]:
            continue
        for e in row:
            act[int(e)] = True
    return act


def chip_slots(slots: Sequence[Sequence[int]], c: int, *, per: int = PER_CHIP) -> List[int]:
    """The 12 + R global experts of chip c's local slots (natives first)."""
    return list(range(c * per, (c + 1) * per)) + [int(e) for e in slots[c]]


def keep_slots(where: Dict[int, int], slots: Sequence[Sequence[int]], c: int, *, per: int = PER_CHIP) -> List[bool]:
    """Which of chip c's 12 + R slots compute this step."""
    return [where.get(e) == c for e in chip_slots(slots, c, per=per)]


def chip_table(slots: Sequence[Sequence[int]], c: int, *, per: int = PER_CHIP) -> List[int]:
    """The per-chip uint32 table (module docstring): replica codes, then the 16 slot experts (padded with NONE)."""
    code = rep_codes(slots, n_experts=len(slots) * per)
    sl = chip_slots(slots, c, per=per)
    return code + sl + [NONE] * (SLOT_WORDS - len(sl))


def max_load(idx_rows, live, slots, *, per: int = PER_CHIP) -> Tuple[int, int]:
    """(busiest chip's expert count without replicas, with the plan's replicas) for one step."""
    act = active_from_rows(idx_rows, live, n_experts=len(slots) * per)
    base = [0] * len(slots)
    for e, a in enumerate(act):
        if a:
            base[e // per] += 1
    _, load = assign(act, rep_codes(slots, n_experts=len(slots) * per), per=per, n_chips=len(slots))
    return max(base), max(load)


# =====================================================================================================================
# the replica plan (one JSON per model; data, not code)
# =====================================================================================================================
def plan_hash(plan: Dict) -> str:
    body = json.dumps({"R": plan["R"], "layers": plan["layers"]}, sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()[:16]


def load_plan(path) -> Dict:
    """``{"format", "R", "layers": {"<l>": slots [32][R]}, "source", "hash"}``; validated."""
    p = json.loads(Path(path).read_text())
    if p.get("format") != PLAN_FORMAT:
        raise ValueError(f"replica plan {path}: format {p.get('format')!r} != {PLAN_FORMAT!r}")
    R = int(p["R"])
    for l, slots in p["layers"].items():
        if len(slots) != N_CHIPS or any(len(r) != R for r in slots):
            raise ValueError(f"replica plan {path}: layer {l} must have {N_CHIPS} x {R} slots")
        code = rep_codes(slots)  # raises on a duplicate
        for c, row in enumerate(slots):
            for e in row:
                if not 0 <= e < N_EXPERTS or e // PER_CHIP == c:
                    raise ValueError(f"replica plan {path}: layer {l} chip {c} holds a replica of its own expert {e}")
        del code
    if p.get("hash") and p["hash"] != plan_hash(p):
        raise ValueError(f"replica plan {path}: hash mismatch")
    return p


def default_plan_path() -> Path:
    return Path(__file__).resolve().parent / "replica_plan_r4.json"


# =====================================================================================================================
# replica weight files: byte-level host transform of the TT cache
# =====================================================================================================================
def tensorbin_layout(path, n_chips: int = N_CHIPS) -> Tuple[int, int, int]:
    """``(data_offset, shard_bytes, file_size)`` of a mesh tensorbin with ``n_chips`` equal shards after the header
    (``u64 header_size``, header, then the shards; the shard table of the ttnn writer lists them in mesh row-major
    order at ``i * shard_bytes``)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        hs = struct.unpack("<Q", f.read(8))[0]
    data = 8 + hs
    if (size - data) % n_chips:
        raise ValueError(f"{path}: {size - data} data bytes do not split into {n_chips} shards")
    return data, (size - data) // n_chips, size


def expert_bytes(K: int, N: int) -> int:
    return (int(K) // 32) * (int(N) // 32) * BFP8_TILE_BYTES


_COORD_TAG = struct.pack("<II", 4, 2)  # (uoffset 4 to the coordinate vector, its length 2) right after each shard struct


def shard_table(header: bytes, mesh_shape=(4, 8)) -> List[Tuple[int, int, int, int]]:
    """The shard table of a tensorbin header (``u64`` size prefix included): ``[(struct position, offset, size,
    linear mesh index)]``. Each ``TensorShard`` stores its ``InlineFileStorage {u64 offset, u64 size}`` struct followed
    (within 24 bytes) by its ``MeshCoordinate`` vector ``(len 2, row, col)``; checked by :func:`check_shard_table`."""
    rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
    out = []
    pos = header.find(_COORD_TAG)
    while pos >= 0:
        r, c = struct.unpack_from("<II", header, pos + 8)
        # the 16-byte struct, 8-byte aligned in the flatbuffer (which starts at file byte 8), ends right before the
        # table's 4-byte soffset (pos - 4) or before 4 bytes of alignment padding
        st = pos - 20 if (pos - 20) % 8 == 0 else pos - 24
        off, size = struct.unpack_from("<QQ", header, st)
        # the TensorTopology's coordinate vectors match the tag too: a shard struct is an aligned, sane extent
        if 0 < size < (1 << 40) and off % 64 == 0 and off < (1 << 42) and size % 32 == 0 and r < rows and c < cols:
            out.append((st, off, size, r * cols + c))
        pos = header.find(_COORD_TAG, pos + 4)
    return out


def check_shard_table(path, mesh_shape=(4, 8)) -> int:
    """A cached EP32 tensorbin's table must place mesh index i at ``i * shard`` (the layout
    :func:`build_replica_tensorbin` reads); returns the shard bytes."""
    n = int(mesh_shape[0]) * int(mesh_shape[1])
    data, shard, _ = tensorbin_layout(path, n)
    with open(path, "rb") as f:
        hdr = f.read(data)
    tab = shard_table(hdr, mesh_shape)
    if sorted(i for *_, i in tab) != list(range(n)) or any(o != i * shard or s != shard for _, o, s, i in tab):
        raise ValueError(f"{path}: unexpected shard table {[(o, s, i) for _, o, s, i in tab]}")
    return shard


def template_header(K: int, N: int, R: int, *, mesh_shape=(4, 8), work_dir: Optional[str] = None) -> bytes:
    """The tensorbin header (``u64`` size prefix included) of a host mesh tensor with ``[1, R, K, N]`` bfp8 TILE per
    chip, from ttnn itself (zeros, ``ttnn.from_host_shards`` in mesh row-major order, ``ttnn.dump_tensor``). ttnn stores
    one buffer shared by several shards once; the shard offsets are then patched to mesh index i at ``i * shard``."""
    import tempfile

    import torch

    import ttnn

    s = ttnn.from_torch(torch.zeros(1, int(R), int(K), int(N)), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    n = int(mesh_shape[0]) * int(mesh_shape[1])
    m = ttnn.from_host_shards([s] * n, ttnn.MeshShape(int(mesh_shape[0]), int(mesh_shape[1])))
    d = work_dir or DEFAULT_WORK_DIR
    Path(d).mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tensorbin", dir=d)
    os.close(fd)
    try:
        ttnn.dump_tensor(tmp, m)
        with open(tmp, "rb") as f:
            hs = struct.unpack("<Q", f.read(8))[0]
            f.seek(0)
            hdr = bytearray(f.read(8 + hs))
    finally:
        os.unlink(tmp)
    shard = int(R) * expert_bytes(K, N)
    tab = shard_table(bytes(hdr), mesh_shape)
    if sorted(i for *_, i in tab) != list(range(n)) or any(s != shard for _, _, s, _ in tab):
        raise ValueError(f"template header: unexpected shard table {[(o, s, i) for _, o, s, i in tab]}")
    for st, _, _, i in tab:
        struct.pack_into("<Q", hdr, st, i * shard)
    return bytes(hdr)


def build_replica_tensorbin(src, dst, chip_experts: Sequence[Sequence[int]], header: bytes, K: int, N: int, *,
                            per: int = PER_CHIP, check: bool = True) -> int:
    """Writes ``dst``: ``header`` + for every chip (mesh row-major) the bytes of its replica experts
    ``chip_experts[c][j]`` copied from their home shards in the cached ``src``. Returns the bytes written. Written to a
    temporary name and renamed (an interrupted build never leaves a complete-looking file)."""
    n = len(chip_experts)
    data, shard, size = tensorbin_layout(src, n)
    if check:
        check_shard_table(src, (4, n // 4) if n == 32 else (1, n))
    eb = expert_bytes(K, N)
    if shard != per * eb:
        raise ValueError(f"{src}: shard {shard} B != {per} x {eb} B (K {K}, N {N})")
    tmp = Path(str(dst) + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(src, "rb") as fs, open(tmp, "wb") as fo:
        fo.write(header)
        total += len(header)
        for row in chip_experts:
            for e in row:
                h, i = divmod(int(e), per)
                fs.seek(data + h * shard + i * eb)
                buf = fs.read(eb)
                if len(buf) != eb:
                    raise IOError(f"{src}: short read for expert {e}")
                fo.write(buf)
                total += eb
    os.replace(tmp, dst)
    return total


def sidecar(dst) -> Path:
    return Path(str(dst) + ".json")


def replica_file(work_dir, tag: str, layer: int, kind: str) -> Path:
    return Path(work_dir) / tag / f"L{int(layer):02d}" / f"{kind}.tensorbin"


def ensure_replica_file(src, dst, chip_experts, header: bytes, K: int, N: int, *, meta: Dict,
                        check: bool = True) -> Tuple[Path, bool]:
    """Resumable per file: reuse ``dst`` when it is complete (size) and its sidecar matches ``meta`` (source size and
    mtime, the plan hash, layer, kind); else build it. Returns ``(dst, built)``."""
    dst = Path(dst)
    want = len(header) + sum(len(r) for r in chip_experts) * expert_bytes(K, N)
    sc = sidecar(dst)
    if dst.is_file() and sc.is_file() and dst.stat().st_size == want:
        try:
            if json.loads(sc.read_text()) == meta:
                return dst, False
        except ValueError:
            pass
    build_replica_tensorbin(src, dst, chip_experts, header, K, N, check=check)
    sc.write_text(json.dumps(meta, sort_keys=True))
    return dst, True


__all__ = [
    "ASSIGN_WORDS", "DEFAULT_R", "NONE", "N_CHIPS", "N_EXPERTS", "PASSES", "PER_CHIP", "REPLICA_MODES", "SLOT_WORDS",
    "TABLE_WORDS", "active_from_rows", "assign", "build_replica_tensorbin", "chip_slots", "chip_table",
    "check_shard_table", "choose_replicas", "default_plan_path", "shard_table", "ensure_replica_file", "expert_bytes", "keep_slots", "load_plan",
    "max_load", "plan_hash", "rep_codes", "replica_file", "spread_targets", "template_header", "tensorbin_layout",
]
