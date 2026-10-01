# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact-fp32 Motif-3 router logits on device: ``logits = x @ W_router^T`` with true fp32 accumulation.

WAVE_A_REVIEW D1(b) (MOE-2 / MOE-7 / GATE-3; design §2.3.7). The draft-1 router (``ttnn.linear`` HiFi4, fp32 dest
acc) forms the dot products on the FPU, whose partial sums are TF32-class (G5: logit RMS error 3.3e-4, 99.63-99.71 %
top-8 set agreement; neither the fidelity escalation, fp32 weights nor a K split help, G5 diagnose). This kernel
never sums on the FPU: every product ``x[t,k] * W[k,e]`` (bf16 x bf16: <= 16 significant bits, exact in fp32) is
accumulated with the SFPU's fp32 multiply-add (SFPMAD, whose product is exact here, so it rounds once: the add), and
the cross-core partials are added with SFPU fp32 adds after an exact fp32 unpack. Every chip runs the same program in a
fixed order, so the result is the deterministic fp32 sum that :func:`emulate_device_fp32` reproduces **bit for bit**
on the host (verified on all 32 chips, eager and traced, decode and prefill shapes).

Contract (``tt/moe.py`` ``MotifRouter.route_logits``)::

    x       [1, 1, M, 4096]  bf16, TILE, interleaved (DRAM or L1), M = 32 n (n >= 1); leading dims must be 1
            (``[M, 4096]`` / ``[1, M, 4096]`` are accepted too). Decode: M = 32, the gathered tokens in lane order
            8 dp + l. Prefill: one call per chunk of M tokens.
    logits  [1, 1, M, 384]   fp32, TILE, interleaved (DRAM default, or L1): a new tensor, bitwise identical on every
            chip for identical x (the routing-consistency invariant of design §2.3.7)

:meth:`RouterLogitsFP32.supports` checks this contract (any M = 32 n), :meth:`RouterLogitsFP32.supports_decode` the
decode shape (M = 32). Choosing the kernel for prefill is the call site's policy (cost below).

Decomposition (one ``ttnn.generic_op``, SPMD on every chip, 96 of the 120 Tensix cores, rows 0..7 of the grid):

* Worker ``q = 32 g + h`` (core ``(q % 12, q // 12)``, derived in the kernels from the logical core coordinates: no
  per-core runtime args) owns expert group ``g`` (experts ``[128 g, 128 g + 128)``, 4 SFPU vectors of 32 lanes) x k
  group ``h`` (x k-tiles ``h, 32 + h, 64 + h, 96 + h``: strided, so the 96 workers' DRAM reads of a chunk hit all
  banks). Phase 1 of a tile row of 32 tokens: per chunk (one k-tile = 32 k) the 4 weight tiles of the chunk
  (host-permuted, :func:`prepare_router_weight`, chunk-major tile ids; read once per launch and kept in L1) are copied
  into DEST (bf16 -> fp32, exact); per token the 4 accumulators live in LREGs; per k the math RISC-V reads ``x[t, k]``
  (bf16 bits) from the x tile in L1 and pushes an SFPLOADI with that 16-bit immediate (broadcast to all lanes), then a
  REPLAY of 4 x (SFPLOAD weight vector, SFPMAD): 9 SFPU instructions per 4 x 32 MACs; each chunk is summed from zero
  and added to the running total (``((c0 + c1) + c2) + c3``). An SFPTRANSP pass then puts the partials into natural
  tile layout (fp32, packed).
* Phase 2 (reduce-scatter, no dedicated reducer cores): each worker sends 32 pieces of 512 B (8 tokens x 16 experts)
  of its partials to the 32 workers of its expert group and receives the 32 k-group partials of its own piece
  (UnpackToDestFp32, exact), sums them with a balanced pairwise fp32 tree in k-group order and writes its 512 B of the
  logits.
* Prefill (n > 1 tile rows): the rows are pipelined, phase 1 of row r + 1 runs while the reduce-scatter of row r is in
  flight (4 receive buffers and 4 semaphores, x double-buffered; ``worker_compute.cpp`` has the ordering argument).
  The decode program (n = 1) keeps one receive buffer (76 KB of CBs per worker; prefill program 152 KB). No
  persistent L1 either way.

Numerics (tests, device): bitwise equal to :func:`emulate_device_fp32` (the same arithmetic for every tile row); vs an
fp64 golden on real router inputs (layers 2-35) max error 1.2e-7 relative to max |logit|; top-8 set agreement with the
reference's own fp32 CPU routes 100 % on all 32,681 real token-layers of the MoE capture (FPU composite: 99.81 %), vs
fp64 99.997 % (1 near-tie, the same token the reference's fp32 router flips).

Cost (traced, decode, 32 tokens): ~38 us per call (phase 1 ~29 us: 9 SFPU instructions per 4 x 32 MACs; the rest is
DRAM start-up, reduce-scatter and the final sums). The FPU ``ttnn.linear`` logits are 52.5 us with the auto config and
13.7 us with the MoE owner's 12-core multicast config (sigmoid fused); inside ``tt/moe.py`` ``MotifRouter`` the exact
path (``logits_fn=``) costs ~25 us more per MoE layer than the composite (149 vs 124 us traced, L1 intermediates).
Prefill (traced): 30.4 us per tile row of 32 tokens at M = 4096 (3.89 ms per 4096-token chunk per MoE layer, vs 0.38 ms
for the FPU ``ttnn.linear``; 128 / 1024 tokens: 128 / 982 us); splitting the rows over the 32 chips (then all-gathering
the logits) would divide that by ~32 (call-site change, not done here). Eager (host) cost ~140 us per decode call (FPU
linear ~66 us): ``generic_op`` builds and patches one program per mesh coordinate (32 per call, ~73 us), the output
alloc + free ~34 us, the Python checks / patch ~30 us. The kernels take no per-core runtime args (the worker index
comes from the logical core coordinates, the receivers' NoC coordinates from a compile-time table), so a cache hit
re-applies only the common args; with the former 69 per-core args per writer core an eager call cost ~1 ms.

Memory (per chip, replicated): 3 MiB bf16 per MoE layer (the bytes of ``W_router^T``, permuted), TT-cache name
:data:`WEIGHT_CACHE_NAME` (:meth:`RouterLogitsFP32.from_source`). While ``MotifRouter`` also keeps its composite
``W^T`` (prefill, or when the kernel is not used) that is a second copy: +3 MiB per layer, ~150 MiB per chip for the 51
MoE layers (and the same on disk in the TT cache). Using the kernel for every shape would make the composite copy
unnecessary (requested of the MoE owner).

The weight must be bf16-valued (:func:`prepare_router_weight` raises otherwise): the kernel's exactness argument and
the host model assume bf16 x bf16 products.

Host helpers (pure torch, device free): :func:`prepare_router_weight`, :func:`golden_logits_fp64`,
:func:`emulate_device_fp32`, :func:`partials_fp32`, :func:`emulate_layout`. Device: :class:`RouterLogitsFP32`.
Kernel sources: ``router_fp32/worker_{reader,compute,writer}.cpp`` (+ ``timing.h`` for the tests' stamps).
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Optional, Tuple

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "router_fp32"
WORKER_READER = KERNEL_DIR / "worker_reader.cpp"
WORKER_COMPUTE = KERNEL_DIR / "worker_compute.cpp"
WORKER_WRITER = KERNEL_DIR / "worker_writer.cpp"

# ---- problem and decomposition (fixed: Motif-3 router) --------------------------------------------------------------
TILE = 32
HIDDEN = 4096
N_EXPERTS = 384
TOKENS = 32  # one tile row of tokens (the 32 decode lanes); a launch handles n_rows = M / 32 of them
E_GROUPS = 3  # expert groups of 128 (4 SFPU vectors of 32 lanes)
E_PER_GROUP = N_EXPERTS // E_GROUPS  # 128
K_GROUPS = 32  # k groups of 128
K_PER_GROUP = HIDDEN // K_GROUPS  # 128
CHUNK_K = 32  # one x tile per chunk
N_CHUNKS = K_PER_GROUP // CHUNK_K  # 4
N_VEC = E_PER_GROUP // 32  # 4 accumulator vectors per token
N_WORKERS = E_GROUPS * K_GROUPS  # 96
N_OUT_TILES = N_EXPERTS // TILE  # 12 output tiles per tile row
X_TILES_PER_ROW = HIDDEN // TILE  # 128 x tiles per tile row
N_PIECES = K_GROUPS  # reduce-scatter: each worker owns 1/32 of its group's 4 output tiles (512 B = 8 rows x 16 cols)
PIECE_BYTES = N_VEC * TILE * TILE * 4 // N_PIECES  # 512
RECV_TILES = N_PIECES * PIECE_BYTES // (TILE * TILE * 4)  # 4 fp32 tiles of received pieces per row
W_TILES_PER_CHUNK = 4  # 32 k x 4 vectors = 128 SFPU vectors = 4 DEST tiles
W_TILES_PER_WORKER = N_CHUNKS * W_TILES_PER_CHUNK  # 16
W_TILES = N_WORKERS * W_TILES_PER_WORKER  # 1536 (= 4096 x 384 / 1024, same bytes as W^T)
WEIGHT_BYTES = W_TILES * TILE * TILE * 2  # 3 MiB per layer per chip (bf16)
# DRAM bank spreading: in chunk cc the 96 workers read weight tiles cc * 384 + 4 q + i (chunk-major, consecutive ids
# over all banks) and x k-tiles 32 cc + h (strided k split: worker h owns k-tiles h, 32 + h, 64 + h, 96 + h).
X_TILE_STRIDE = HIDDEN // TILE // N_CHUNKS  # 32
W_CHUNK_STRIDE = N_WORKERS * W_TILES_PER_CHUNK  # 384
GRID_X = 12  # worker core (q % 12, q // 12): rows 0..7 of the 12 x 10 grid
GRID_Y = N_WORKERS // GRID_X  # 8

# Buffering of the two program variants: decode (one tile row) / prefill (n_rows > 1, rows pipelined). The prefill
# variant needs 4 receive buffers + 4 semaphores for the pipelined order (worker_compute.cpp) and double-buffers x,
# the partials and the final pieces.
RECV_BUFS = {False: 1, True: 4}
X_BUFS = {False: 1, True: 2}
OUT_BUFS = {False: 1, True: 2}
FIN_BUFS = {False: 1, True: 2}

# TT-cache name of the prepared weight (bf16 [1, 1, 49152, 32], replicated). The suffix versions the layout of
# prepare_router_weight: change it whenever the permutation changes, so a stale cache file is never reloaded.
WEIGHT_CACHE_NAME = "moe.router.weight_fp32k_v1"
WEIGHT_SHAPE = (1, 1, W_TILES * TILE, TILE)

# CB indices (shared with the kernels' compile-time args)
CB_X, CB_W, CB_OUT, CB_RECV, CB_FIN, CB_TIME = 0, 1, 2, 3, 4, 5
SEM_ID0 = 0  # semaphores SEM_ID0 .. SEM_ID0 + RECV_BUFS - 1 (row r uses SEM_ID0 + r % RECV_BUFS)
TIME_WORDS = 16  # timing.h: uint32 stamp slots per core (64 B rows of the timing tensor)

# kernel modes (worker compute CT arg): 0 = logits (replay-buffer inner loop); 1 = debug: worker partial = chunk-0
# weight DEST tiles (raw copy); 2 = debug: accumulators = bf16(x[t, k0]) broadcast (x addressing check);
# 3 = logits with the SFPI compiler-scheduled inner loop (reference; same arithmetic, ~1.7x slower)
MODE_LOGITS, MODE_DEBUG_W, MODE_DEBUG_X, MODE_LOGITS_SFPI = 0, 1, 2, 3


# =====================================================================================================================
# Host layout (pure torch)
# =====================================================================================================================
def k_of(h: int, cc: int, kc: int) -> int:
    """Hidden index of column ``kc`` of chunk ``cc`` of k group ``h``: x k-tile ``32 cc + h``."""
    return TILE * (X_TILE_STRIDE * cc + h) + kc


def k_order() -> torch.Tensor:
    """``[32 k groups, 128]``: the hidden indices of k group ``h`` in the kernel's accumulation order."""
    return torch.tensor([[k_of(h, cc, kc) for cc in range(N_CHUNKS) for kc in range(CHUNK_K)] for h in range(K_GROUPS)])


def weight_tile_id(q: int, cc: int, i: int) -> int:
    """DRAM tile id of weight tile ``i`` (DEST tile 4 + i) of worker ``q`` in chunk ``cc`` (chunk major)."""
    return W_CHUNK_STRIDE * cc + W_TILES_PER_CHUNK * q + i


def lane_expert(j: int, r: int, c: int) -> int:
    """Local expert (0..127 inside the worker's group) held by SFPU lane ``8 r + c`` of accumulator vector ``j``.

    Chosen so that the SFPTRANSP of 4 tokens' accumulators lands every lane on its natural tile position: after the
    transpose, register ``i`` holds (4 tokens on the row groups) x (experts ``32 j + 16 (i >> 1) + 2 c + (i & 1)``),
    i.e. face column ``i >> 1``, column parity ``i & 1`` of output tile ``j``."""
    return 32 * j + 16 * (r >> 1) + 2 * c + (r & 1)


def slot_lane_tile_pos(ii: int, r: int, c: int) -> Tuple[int, int]:
    """(tile row, tile col) of SFPU lane ``8 r + c`` of the vector at DEST slot ``ii`` (0..31) of a 32 x 32 tile.

    SFPLOAD/SFPSTORE at dst_reg index ``ii`` (DEST rows ``4 (ii // 2) .. + 3``, columns of parity ``ii % 2``): lane
    ``8 r + c`` <-> DEST row ``4 (ii // 2) + r``, column ``2 c + ii % 2`` (tt-isa SFPLOAD); DEST rows 16 f .. 16 f + 15
    hold face ``f`` (0: rows 0-15 x cols 0-15, 1: rows 0-15 x cols 16-31, 2, 3)."""
    rq, p = ii // 2, ii % 2
    f = rq // 4
    fr = 4 * (rq % 4) + r
    fc = 2 * c + p
    return 16 * (f // 2) + fr, 16 * (f % 2) + fc


@functools.lru_cache(maxsize=1)
def _vector_tile_index() -> Tuple[torch.Tensor, torch.Tensor]:
    """``(tr, tc)`` LongTensors ``[32 slots, 32 lanes]``: vector ``ii`` lane ``L`` of a tile = ``tile[tr, tc]``."""
    tr = torch.empty(32, 32, dtype=torch.long)
    tc = torch.empty(32, 32, dtype=torch.long)
    for ii in range(32):
        for lane in range(32):
            tr[ii, lane], tc[ii, lane] = slot_lane_tile_pos(ii, lane // 8, lane % 8)
    return tr, tc


@functools.lru_cache(maxsize=1)
def _weight_gather_index() -> Tuple[torch.Tensor, torch.Tensor]:
    """``(k_idx, e_idx)`` LongTensors ``[W_TILES, 32, 32]``: element ``(tile, row, col)`` of the prepared weight is
    ``W^T[k_idx, e_idx]``. Tile :func:`weight_tile_id` ``(q, cc, i)`` = DEST tile ``4 + i`` of worker ``q`` in chunk
    ``cc``; its slot ``ii`` holds weight vector ``w = 32 i + ii`` = ``(kc, j) = (w // 4, w % 4)``."""
    k_idx = torch.empty(W_TILES, 32, 32, dtype=torch.long)
    e_idx = torch.empty(W_TILES, 32, 32, dtype=torch.long)
    tr, tc = _vector_tile_index()
    lanes = torch.arange(32)
    r, c = lanes // 8, lanes % 8
    e_lane = [torch.tensor([lane_expert(j, int(rr), int(cv)) for rr, cv in zip(r, c)]) for j in range(N_VEC)]
    for q in range(N_WORKERS):
        g, h = divmod(q, K_GROUPS)
        for cc in range(N_CHUNKS):
            for i in range(W_TILES_PER_CHUNK):
                tile = weight_tile_id(q, cc, i)
                for ii in range(32):
                    w = 32 * i + ii
                    kc, j = divmod(w, N_VEC)
                    k_idx[tile, tr[ii], tc[ii]] = k_of(h, cc, kc)
                    e_idx[tile, tr[ii], tc[ii]] = E_PER_GROUP * g + e_lane[j]
    return k_idx, e_idx


def check_bf16_valued(t: torch.Tensor, what: str = "router weight") -> None:
    """Raise unless every value of ``t`` is exactly representable in bf16 (the kernel's exactness argument and
    :func:`emulate_device_fp32` assume bf16 x bf16 products; an fp32 weight would be rounded silently at upload)."""
    if t.dtype == torch.bfloat16:
        return
    if not t.is_floating_point():
        raise ValueError(f"{what} must be a floating-point tensor, got {t.dtype}")
    rounded = t.to(torch.bfloat16).to(t.dtype)
    if not torch.equal(rounded, t):
        bad = int((rounded != t).sum())
        raise ValueError(
            f"{what} must be bf16-valued (the checkpoint stores it as bf16): {bad} of {t.numel()} values change "
            f"when rounded to bf16 (max |delta| {float((rounded - t).abs().max()):.3e})"
        )


def prepare_router_weight(w_t: torch.Tensor) -> torch.Tensor:
    """``W_router^T [4096, 384]`` (``weights.router_weights(...)[0]``, ``y = x @ w_t``), bf16-valued -> the kernel's
    weight layout ``[1, 1, 1536 * 32, 32]`` (same values, permuted; upload bf16 TILE, replicated). fp32 / fp64 inputs
    keep their dtype; values that are not exactly bf16 raise (:func:`check_bf16_valued`)."""
    if tuple(w_t.shape) != (HIDDEN, N_EXPERTS):
        raise ValueError(f"router weight must be W^T [{HIDDEN}, {N_EXPERTS}], got {tuple(w_t.shape)}")
    check_bf16_valued(w_t)
    k_idx, e_idx = _weight_gather_index()
    src = w_t if w_t.dtype in (torch.float32, torch.float64) else w_t.float()
    return src[k_idx, e_idx].reshape(WEIGHT_SHAPE).contiguous()


def golden_logits_fp64(x: torch.Tensor, w_t: torch.Tensor) -> torch.Tensor:
    """fp64 logits of the bf16-valued inputs: ``[..., 4096] @ [4096, 384]``."""
    return x.double() @ w_t.double()


def _tree_sum(parts, lo: int, n: int) -> torch.Tensor:
    """Balanced pairwise fp32 tree ``sum(lo, n) = sum(lo, n / 2) + sum(lo + n / 2, n / 2)`` (the kernel's phase 2)."""
    if n == 1:
        return parts[lo]
    return _tree_sum(parts, lo, n // 2) + _tree_sum(parts, lo + n // 2, n // 2)


def emulate_device_fp32(x: torch.Tensor, w_t: torch.Tensor, block: int = 256) -> torch.Tensor:
    """Bit-exact host model of the kernel's arithmetic: ``x [M, 4096]`` (M = 32 n; any leading dims of size 1),
    ``w_t [4096, 384]`` (bf16 values) -> fp32 ``[M, 384]``. Per (token, expert, k group h): chunk sums
    ``c_cc = sum_kc x * w`` sequential from zero over the 32 k of x k-tile ``32 cc + h`` (exact products, one fp32
    rounding per add), group partial ``((c_0 + c_1) + c_2) + c_3``; logits = balanced pairwise fp32 tree over the 32
    group partials (h = 0..31). Every token is computed the same way, whatever its tile row (``block`` only bounds the
    host memory)."""
    xs = x.reshape(-1, HIDDEN)
    if xs.shape[0] % TOKENS:
        raise ValueError(f"x must have a multiple of {TOKENS} rows, got {xs.shape[0]}")
    outs = []
    for s0 in range(0, xs.shape[0], block):
        acc = partials_fp32(xs[s0 : s0 + block], w_t)
        outs.append(_tree_sum([acc[h] for h in range(K_GROUPS)], 0, K_GROUPS))
    return torch.cat(outs)


def partials_fp32(x: torch.Tensor, w_t: torch.Tensor) -> torch.Tensor:
    """``[32 h, M t, 384 e]`` fp32: every k group's partial sum in the kernel's order (the worker partials before
    the reduce-scatter): per chunk a sequential sum from zero, then ``((c_0 + c_1) + c_2) + c_3``."""
    xs = x.reshape(-1, HIDDEN).float()
    order = k_order()  # [h, 128] = chunk-major
    xg = xs[:, order].permute(1, 0, 2)  # [h, t, kk]
    wg = w_t.float()[order]  # [h, kk, e]
    total = None
    for cc in range(N_CHUNKS):
        c = torch.zeros(K_GROUPS, xs.shape[0], N_EXPERTS, dtype=torch.float32)
        for kc in range(CHUNK_K):
            kk = CHUNK_K * cc + kc
            c = c + xg[:, :, kk : kk + 1] * wg[:, kk : kk + 1, :]  # exact products for bf16-valued operands
        total = c if total is None else total + c
    return total


def _transp4(m):
    """SFPTRANSP on 4 vectors of 32 lanes: ``new[i][8 r + c] = old[r][8 i + c]`` (tt-isa SFPTRANSP)."""
    old = torch.stack(m).reshape(4, 4, 8)  # [reg, row group, col]
    return list(old.permute(1, 0, 2).reshape(4, 32))


def emulate_layout(x: torch.Tensor, w_prep: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
    """Host model of the kernel's *data movement* (DEST slots, SFPU lanes, SFPTRANSP, packing, reduction) on the
    prepared weight; arithmetic in ``dtype``. ``x [32, 4096]``, ``w_prep [1, 1, 49152, 32]`` -> ``[32, 384]``.
    Equal to ``x @ w_t`` (to ``dtype`` rounding) iff :func:`prepare_router_weight` and the kernel layout agree."""
    tr, tc = _vector_tile_index()
    xs = x.reshape(TOKENS, HIDDEN).to(dtype)
    tiles = w_prep.reshape(W_TILES, TILE, TILE).to(dtype)
    vec = tiles[:, tr, tc]  # [W_TILES, 32 slots, 32 lanes]
    partial = torch.zeros(N_WORKERS, N_VEC, TILE, TILE, dtype=dtype)
    for q in range(N_WORKERS):
        g, h = divmod(q, K_GROUPS)
        acc = torch.zeros(TOKENS, N_VEC, 32, dtype=dtype)  # acc slot 4 t + j, lanes
        for cc in range(N_CHUNKS):
            t0 = weight_tile_id(q, cc, 0)
            wv = vec[t0 : t0 + W_TILES_PER_CHUNK].reshape(CHUNK_K, N_VEC, 32)  # slot w = 4 kc + j
            k0 = k_of(h, cc, 0)
            xt = xs[:, k0 : k0 + CHUNK_K]  # [t, kc]
            acc = acc + torch.einsum("tk,kjl->tjl", xt, wv)
        for tq in range(TOKENS // 4):
            for j in range(N_VEC):
                m = _transp4([acc[4 * tq + i, j] for i in range(4)])
                for i in range(4):
                    f = 2 * (tq // 4) + (i >> 1)
                    idx = 8 * f + 2 * (tq % 4) + (i & 1)
                    partial[q, j][tr[idx], tc[idx]] = m[i]
    out = torch.zeros(TOKENS, N_EXPERTS, dtype=dtype)
    for r in range(N_OUT_TILES):
        g, j = divmod(r, N_VEC)
        out[:, TILE * r : TILE * (r + 1)] = partial[K_GROUPS * g : K_GROUPS * (g + 1), j].sum(0)
    return out


def worker_core(q: int) -> Tuple[int, int]:
    return q % GRID_X, q // GRID_X


def piece_of(h: int) -> Tuple[int, int]:
    """Worker (g, h)'s piece of the reduce-scatter: (output tile 4 g + j with j = h // 8, byte offset 512 (h % 8))
    = face (h % 8) // 2, token rows 8 (h % 2) .. + 7 of that fp32 tile."""
    return h // 8, PIECE_BYTES * (h % 8)


def cb_bytes(multi: bool) -> int:
    """L1 bytes of circular buffers per worker core (decode program: ``multi=False``; prefill: True)."""
    return (
        X_BUFS[multi] * N_CHUNKS * 2048
        + W_TILES_PER_WORKER * 2048
        + OUT_BUFS[multi] * N_VEC * 4096
        + RECV_BUFS[multi] * RECV_TILES * 4096
        + FIN_BUFS[multi] * 4096
    )


def input_rows(shape) -> int:
    """Token count M of an input shape under the kernel contract (``[..., M, 4096]``, leading dims all 1,
    M = 32 n >= 32); raises ValueError otherwise."""
    shape = tuple(int(d) for d in shape)
    if len(shape) < 2 or shape[-1] != HIDDEN or math.prod(shape[:-2]) != 1:
        raise ValueError(f"router_fp32 input must be [1, 1, M, {HIDDEN}] (leading dims of size 1), got {shape}")
    m = shape[-2]
    if m < TOKENS or m % TOKENS:
        raise ValueError(f"router_fp32 input rows must be a positive multiple of {TOKENS}, got {shape}")
    return m


def check_interleaved(mc, what: str) -> None:
    """Raise unless ``mc`` is an interleaved DRAM or L1 memory config."""
    if (
        mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
        or mc.is_sharded()
        or mc.buffer_type not in (ttnn.BufferType.DRAM, ttnn.BufferType.L1)
    ):
        raise ValueError(f"router_fp32 {what} must be interleaved DRAM or L1, got {mc}")


# =====================================================================================================================
# Device
# =====================================================================================================================
def _core_range_set(x0: int, y0: int, x1: int, y1: int):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _cb(index: int, n_tiles: int, dtype, cores):
    page = {ttnn.bfloat16: 2048, ttnn.float32: 4096}[dtype]
    return ttnn.CBDescriptor(
        total_size=n_tiles * page,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _accessor(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


class RouterLogitsFP32:
    """Exact-fp32 router logits of one MoE layer, replicated on every chip (see the module docstring).

    Args:
        mesh_device: the opened mesh (any shape; every chip runs the same program).
        w_t: ``W_router^T [4096, 384]`` torch, bf16-valued (``weights.router_weights(...)[0]``), or ``None`` with
            ``weight_tensor`` given.
        weight_tensor: an already uploaded prepared weight (``[1, 1, 49152, 32]`` bf16 TILE, replicated; e.g. another
            instance's ``.weight``).
        owns_weight: whether :meth:`deallocate` frees the weight. Default: True iff this object created it (``w_t``,
            :meth:`from_source`); a caller's ``weight_tensor`` stays the caller's.
        cfg / layer_idx / cache_name: when ``cfg`` is given the prepared weight is uploaded with ``weights.as_tensor``,
            through its TT cache when ``cache_name`` is not None (normally :data:`WEIGHT_CACHE_NAME`, see
            :meth:`from_source`).
        debug: allocate a per-call DRAM dump of every worker's partial tiles of tile row 0 (tests only).
        timing: allocate a per-call DRAM dump of per-core wall-clock stamps (``kernels/router_fp32/timing.h``; tests).
        output_memory_config: default memory config of the logits (interleaved DRAM or L1; DRAM default).

    Call: ``logits = router(x)`` (alias ``route_logits``) with ``x [1, 1, M, 4096]`` bf16 TILE interleaved, M = 32 n
    -> fp32 ``[1, 1, M, 384]`` TILE interleaved (DRAM unless ``memory_config``; a new tensor). Trace safe: one fixed
    program per (input / output buffer type, decode or prefill variant), no host round trip, no persistent L1
    (receive buffers are per-program CBs, semaphores re-armed by the kernel).
    """

    def __init__(
        self,
        mesh_device,
        w_t: Optional[torch.Tensor] = None,
        *,
        weight_tensor=None,
        owns_weight: Optional[bool] = None,
        cfg=None,
        layer_idx: Optional[int] = None,
        cache_name: Optional[str] = None,
        mode: int = MODE_LOGITS,
        debug: bool = False,
        timing: bool = False,
        output_memory_config=None,
    ):
        self.mesh_device = mesh_device
        self.mode = int(mode)
        if self.mode not in (MODE_LOGITS, MODE_DEBUG_W, MODE_DEBUG_X, MODE_LOGITS_SFPI):
            raise ValueError(f"unknown router_fp32 mode {mode}")
        self.debug = bool(debug) or self.mode in (MODE_DEBUG_W, MODE_DEBUG_X)
        self.timing = bool(timing)
        self.output_memory_config = output_memory_config or ttnn.DRAM_MEMORY_CONFIG
        check_interleaved(self.output_memory_config, "output_memory_config")
        grid = mesh_device.compute_with_storage_grid_size()
        if int(grid.x) < GRID_X or int(grid.y) < GRID_Y:
            raise ValueError(f"router_fp32 needs a >= {GRID_X} x {GRID_Y} compute grid, got {grid.x} x {grid.y}")
        if weight_tensor is not None:
            if w_t is not None:
                raise ValueError("give w_t or weight_tensor, not both")
            self._check_weight_tensor(weight_tensor)
            self.weight = weight_tensor
            self._owns_weight = bool(owns_weight) if owns_weight is not None else False
        else:
            if w_t is None:
                raise ValueError("give w_t (W_router^T [4096, 384]) or weight_tensor")
            check_bf16_valued(w_t)  # before any device allocation (and before a TT-cache lookup)
            src = lambda: prepare_router_weight(w_t)  # noqa: E731
            if cfg is not None:  # README §10: weights through weights.as_tensor (TT cache when cache_name is given)
                from .. import weights as W

                self.weight = W.as_tensor(
                    src, mesh_device=mesh_device, cfg=cfg, dtype=ttnn.bfloat16, cache_name=cache_name, layer=layer_idx
                )
            else:
                self.weight = ttnn.from_torch(
                    src(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            self._owns_weight = True if owns_weight is None else bool(owns_weight)
        self.workers = _core_range_set(0, 0, GRID_X - 1, GRID_Y - 1)
        self._noc_x, self._noc_y = self._noc_grid(mesh_device)
        self.last_debug = None
        self.last_timing = None
        # (accessor CT args of x, w, out, debug, timing; prefill variant) -> ProgramDescriptor; the buffer addresses
        # and the row count are common runtime args, patched per call
        self._desc = {}

    @staticmethod
    def _check_weight_tensor(t) -> None:
        shape = tuple(int(d) for d in t.shape)
        if shape != WEIGHT_SHAPE or t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError(
                f"weight_tensor must be the prepared weight {list(WEIGHT_SHAPE)} bf16 TILE, got {shape} {t.dtype} "
                f"{t.layout}"
            )
        check_interleaved(t.memory_config(), "weight_tensor")

    @staticmethod
    def _noc_grid(mesh_device):
        """NoC x of the 12 worker columns and NoC y of the 8 worker rows (the writer's compile-time table). Checks that
        ``worker_core_from_logical_core`` is a separable grid on the worker block."""
        xs, ys = [None] * GRID_X, [None] * GRID_Y
        for q in range(N_WORKERS):
            lx, ly = worker_core(q)
            c = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(lx, ly))
            cx, cy = int(c.x), int(c.y)
            xs[lx] = cx if xs[lx] is None else xs[lx]
            ys[ly] = cy if ys[ly] is None else ys[ly]
            if (xs[lx], ys[ly]) != (cx, cy) or cx > 255 or cy > 255:
                raise RuntimeError(f"router_fp32: worker NoC coordinates are not a separable grid at {(lx, ly)}")
        return xs, ys

    @classmethod
    def from_source(cls, mesh_device, cfg, layer_idx: int, *, source, cache: bool = True, **kw):
        """Router weight of layer ``layer_idx`` from a ``weights`` source (HF names: ``moe.router.gate.weight``,
        ``moe.expert_bias``), TT-cached as :data:`WEIGHT_CACHE_NAME` when ``cache`` (the source is not read on a hit).
        ``kw`` goes to the constructor (``output_memory_config``, test options). The instance owns the weight."""
        from .. import weights as W

        l = int(layer_idx)

        def w_t():
            return W.router_weights(
                source.get(W.hf_name(l, "moe.router.gate.weight")), source.get(W.hf_name(l, "moe.expert_bias"))
            )[0]

        weight = W.as_tensor(
            lambda: prepare_router_weight(w_t()),  # raises before any upload if the weight is not bf16-valued
            mesh_device=mesh_device,
            cfg=cfg,
            dtype=ttnn.bfloat16,
            cache_name=WEIGHT_CACHE_NAME if cache else None,
            layer=l,
        )
        try:
            return cls(mesh_device, weight_tensor=weight, owns_weight=True, **kw)
        except BaseException:
            ttnn.deallocate(weight)  # e.g. a bad output_memory_config: do not leak the upload
            raise

    # ---- contract ---------------------------------------------------------------------------------------------------
    @staticmethod
    def supports(x) -> bool:
        """True iff ``x`` has the kernel's input contract (``[1, 1, M, 4096]`` bf16 TILE interleaved, M = 32 n >= 32;
        decode and prefill shapes). Whether to use the kernel for prefill (~30 us per 32 tokens) is the call site's
        choice; :meth:`supports_decode` is the decode-shape test."""
        try:
            RouterLogitsFP32._check_input(x)
            return True
        except ValueError:
            return False

    @staticmethod
    def supports_decode(x) -> bool:
        """True iff ``x`` has the contract and exactly one tile row of tokens (M = 32: the decode call)."""
        return RouterLogitsFP32.supports(x) and int(x.shape[-2]) == TOKENS

    @staticmethod
    def _check_input(x) -> int:
        m = input_rows(x.shape)
        if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"router_fp32 input must be bf16 TILE, got {x.dtype} {x.layout}")
        check_interleaved(x.memory_config(), "input")
        return m

    # ---- program ---------------------------------------------------------------------------------------------------
    def _program(self, x, out, dbg, tim, n_rows: int):
        """The cached program for these tensors' buffer types (and debug / timing / prefill variant), patched with
        this call's buffer addresses and row count (common runtime args only: there are no per-core runtime args, so
        a cache hit rebuilds nothing per core)."""
        multi = n_rows > 1
        # The key must cover everything the program bakes in at build time: the tensor-accessor compile-time args
        # (buffer type DRAM / L1, layout) of every tensor, the debug / timing variant and the CB / semaphore layout.
        key = tuple(tuple(_accessor(t)) if t is not None else None for t in (x, self.weight, out, dbg, tim)) + (multi,)
        if key not in self._desc:
            self._desc[key] = self._build(x, out, dbg, tim, multi)
        desc = self._desc[key]
        desc.kernels[0].common_runtime_args = [x.buffer_address(), self.weight.buffer_address(), n_rows]
        desc.kernels[1].common_runtime_args = [n_rows]
        desc.kernels[2].common_runtime_args = [
            dbg.buffer_address() if dbg is not None else 0,
            tim.buffer_address() if tim is not None else 0,
            out.buffer_address(),
            n_rows,
        ]
        return desc

    def _build(self, x, out, dbg, tim, multi: bool):
        timing = int(tim is not None)
        recv_bufs = RECV_BUFS[multi]
        cbs = [
            _cb(CB_X, X_BUFS[multi] * N_CHUNKS, ttnn.bfloat16, self.workers),
            _cb(CB_W, W_TILES_PER_WORKER, ttnn.bfloat16, self.workers),
            _cb(CB_OUT, OUT_BUFS[multi] * N_VEC, ttnn.float32, self.workers),
            # same address on every worker (senders target it); receive buffer b = row % recv_bufs at b * 16 KB
            _cb(CB_RECV, recv_bufs * RECV_TILES, ttnn.float32, self.workers),
            _cb(CB_FIN, FIN_BUFS[multi], ttnn.float32, self.workers),
        ]
        if timing:
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=4 * TIME_WORDS,
                    core_ranges=self.workers,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=CB_TIME, data_format=ttnn.uint32, page_size=4 * TIME_WORDS)
                    ],
                )
            )
        sems = [
            ttnn.SemaphoreDescriptor(id=SEM_ID0 + b, core_ranges=self.workers, initial_value=0)
            for b in range(recv_bufs)
        ]

        cc = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[CB_RECV] = ttnn.UnpackToDestMode.UnpackToDestFp32  # fp32 partials: the SrcA path would truncate to TF32
        cc.unpack_to_dest_mode = modes

        filler = _accessor(out)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(WORKER_READER),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.workers,
                compile_time_args=[CB_X, CB_W, N_CHUNKS, W_TILES_PER_CHUNK, timing, CB_TIME, CB_RECV, SEM_ID0, N_PIECES,
                                   RECV_TILES, X_TILE_STRIDE, W_CHUNK_STRIDE, X_TILES_PER_ROW, recv_bufs, GRID_X,
                                   K_GROUPS]
                + _accessor(x)
                + _accessor(self.weight),
                runtime_args=[],
                common_runtime_args=[x.buffer_address(), self.weight.buffer_address(), 1],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(WORKER_COMPUTE),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.workers,
                compile_time_args=[CB_X, CB_W, CB_OUT, self.mode, timing, CB_TIME, CB_RECV, CB_FIN, RECV_TILES],
                runtime_args=[],
                common_runtime_args=[1],
                config=cc,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(WORKER_WRITER),
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=self.workers,
                compile_time_args=[CB_OUT, CB_RECV, SEM_ID0, N_VEC, int(dbg is not None), timing, CB_TIME, CB_FIN,
                                   N_PIECES, recv_bufs, GRID_X, K_GROUPS, N_OUT_TILES, RECV_TILES]
                + list(self._noc_x)
                + list(self._noc_y)
                + (_accessor(dbg) if dbg is not None else filler)
                + (_accessor(tim) if tim is not None else filler)
                + _accessor(out),
                runtime_args=[],
                common_runtime_args=[0, 0, out.buffer_address(), 1],
                config=ttnn.WriterConfigDescriptor(),
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    def __call__(self, x, *, memory_config=None):
        """``x [1, 1, M, 4096]`` bf16 TILE (M = 32 n) -> fp32 logits ``[1, 1, M, 384]`` TILE (DRAM unless
        ``memory_config``)."""
        if self.weight is None:
            raise RuntimeError("router_fp32: called after deallocate()")
        m = self._check_input(x)
        n_rows = m // TOKENS
        if n_rows > 1 and self.mode in (MODE_DEBUG_W, MODE_DEBUG_X):
            raise ValueError("router_fp32 debug modes 1 / 2 take one tile row of tokens (M = 32)")
        mc = memory_config or self.output_memory_config
        check_interleaved(mc, "output memory_config")
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, m, N_EXPERTS]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh_device, mc
        )
        dbg = None
        if self.debug:
            dbg = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, N_WORKERS * N_VEC * TILE, TILE]),
                ttnn.float32,
                ttnn.TILE_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        tim = None
        if self.timing:
            tim = ttnn.allocate_tensor_on_device(
                ttnn.Shape([N_WORKERS, TIME_WORDS]),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        desc = self._program(x, out, dbg, tim, n_rows)
        io = [x, self.weight] + [t for t in (dbg, tim) if t is not None] + [out]
        ttnn.generic_op(io, desc)
        self.last_debug = dbg
        self.last_timing = tim
        return out

    route_logits = __call__

    @property
    def owns_weight(self) -> bool:
        return self._owns_weight

    def deallocate(self) -> None:
        """Free the prepared weight if this object owns it (see ``owns_weight``); the instance is unusable after."""
        if self.weight is not None and self._owns_weight:
            ttnn.deallocate(self.weight)
        self.weight = None
        self._desc.clear()
