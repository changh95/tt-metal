# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 3g / D2: a packed multi-user prefill pass with the SEQUENTIAL pass's numerics (backlog item 8).

``SOLAR_OPEN_PACKED_PREFILL_SEQ_NUMERICS`` (levels in ``packed_prefill.py``, the pure half: the switch, ttnn's auto
``in0_block_w`` rule, the row-piece arithmetic). Everything here runs only INSIDE a packed pass (``Model.
ttnn_prefill_forward`` with ``batch_size > 1`` marks it through ``experts.prefill.packed_prefill_pass(True, seq_len=S)``);
a single-user prefill, chunked or not, and decode never reach these branches, so the sequential path stays bit-identical
by construction.

What a packed pass of ``T = B x S`` rows does differently from ``B`` sequential ``S``-row passes, and what each level pins
(D1, ``tests/test_packed_bias_bisect.py``; every lever verified bit-identical on device before it was wired):

* ``qkv`` ``ttnn.linear`` ``[T, 4096] x [4096, 1280]``: auto 1D ``in0_block_w 2`` at M = 128, 2D ``in0_block_w 1`` from
  M = 256 -> level 1 pins a 2D config with the S-row ``in0_block_w`` and the HiFi2 compute config the auto path uses
  (``attention_seq_numerics_configs``; r7: 2D k=2 at 256 rows == the 128-row production result bit for bit).
* ``o_proj`` ``[T, 1024] x [1024, 4096]`` (bfp8 x bfp8, LoFi): auto 1D k=2 up to 256 rows, 2D k=1 from 1024 -> the
  same pin with a LoFi compute config (r7: 2D k=2 at 4096 == 128 bit for bit).
* shared expert (``shared_expert.py``): explicit 1D ``in0_block_w 32`` configs up to 128 rows, auto above (4.6-6.5 %
  RMS apart, the packed output 1.8-4.8 % smaller) -> level 1 runs it in S-row pieces (``ttnn.split`` + the S-row
  configs + ``ttnn.concat``; r3 / r8: the explicit config at 256 rows == 128 bit for bit, so the piece form is exact).
* head: the sequential head normalizes ONE 32-row tile with the width-sharded decode kernel (``rms_norm.py::
  decode_norm_applies``) and runs the lm_head at M = 32 (1D k=2); the packed "full" head normalizes all T rows with the
  default kernel (2.8 % apart in magnitude) and takes the 2D k=1 lm_head config at M = 4096 -> level 1 splits the head
  input into 32-row tiles for the norm and the lm_head into <= ``LM_HEAD_SEQ_ROWS``-row pieces (``head_seq_numerics``;
  r8: M = 32 == 128 == 1024 rows bit for bit). The gather head (``Model._gather_head``) is that head already.
* routed experts: dense bmm for splits <= ``dense_bmm_max_tokens`` (bit-identical per row at 128 / 256, r3), the
  expert-sorted hot / cold path for 1024-row splits (0.7-2.6 % RMS apart, an algorithm rather than a config) -> level 2
  cuts the packed pass into dense-bmm splits (``experts/prefill.py::seq_numerics_dense_split_size``) when the per-user
  S runs the dense bmm itself: the whole pass then equals the sequential one bit for bit, at the dense path's cost.
Unchanged by any level (bit-identical on identical inputs, D1 section 3): the norms of the 48 layers, RoPE, the per-user
SDPA and KV fill, the router (fp32 accumulation), the all-reduces, the residual adds.
"""

import ttnn

from . import packed_prefill
from .experts import prefill as experts_prefill

TILE = ttnn.TILE_SIZE


def seq_numerics_level():
    """Knob level in effect (0 outside a packed pass, see ``experts.prefill.packed_seq_numerics_level``)."""
    return experts_prefill.packed_seq_numerics_level()


def seq_numerics_active():
    """True inside a packed pass with the knob at level >= 1."""
    return seq_numerics_level() >= 1


def seq_numerics_rows():
    """Per-user row count S of the active packed pass when the knob is active, else None."""
    return experts_prefill.packed_prefill_seq_len() if seq_numerics_active() else None


def compute_config(arch, fidelity, fp32_dest_acc=False):
    """The compute config ttnn's auto matmul uses, restated for an explicit program config (no approx, L1 packer
    accumulation, bf16 destination): ``fidelity`` HiFi2 for bf16 x bfp8, LoFi for bfp8 x bfp8 (``matmul_device_
    operation.cpp``). A program config WITHOUT a compute config would drop the bf16 x bfp8 pair to LoFi."""
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=True,
    )


def matmul_config_2d(m, n, k, in0_block_w, grid):
    """2D in0 / in1-multicast config for ``[.., m, k] x [k, n]`` with the auto path's blocking (``per_core_M =
    div_up(Mt, grid.y)``, ``per_core_N = div_up(Nt, grid.x)``, one output block per core) and an explicit
    ``in0_block_w`` (snapped to a divisor of Kt). The result depends on ``in0_block_w`` only (D1 r7: 1D k=2 at 128 rows
    == 2D k=2 at 256 / 4096 rows for qkv / o_proj), so this reproduces a narrow-shape 1D result at any M."""
    mt, kt, nt = -(-m // TILE), -(-k // TILE), -(-n // TILE)
    if kt % in0_block_w != 0:
        in0_block_w = max(d for d in range(1, in0_block_w + 1) if kt % d == 0)
    per_core_m = -(-mt // grid.y)
    per_core_n = -(-nt // grid.x)
    out_subblock_w = next(d for d in (8, 4, 2, 1) if per_core_n % d == 0)  # 1 x w tiles <= 8 dst registers (bf16)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


def seq_numerics_matmul(m, rows, n, k, grid, arch, fidelity):
    """``(program_config, compute_kernel_config)`` pinning a ``[m, k] x [k, n]`` matmul of a packed pass to the
    ``rows``-row auto ``in0_block_w``, or ``(None, None)`` when the auto choice at ``m`` rows is the same already."""
    in0_block_w = packed_prefill.seq_numerics_in0_block_w(m, rows, n, k, grid.x)
    if in0_block_w is None:
        return None, None
    return matmul_config_2d(m, n, k, in0_block_w, grid), compute_config(arch, fidelity)


def attention_seq_numerics_configs(total_rows, seq_len, batch_size, weights, mesh_device, program_config, keep_bf16):
    """``((qkv_program_config, qkv_compute_config), (out_program_config, out_compute_config))`` for one attention
    prefill call: ``(None, None)`` pairs (ttnn auto) unless this is a packed pass (``batch_size > 1``) under the knob,
    in which case the projections whose auto ``in0_block_w`` at ``total_rows`` differs from the one at the per-user
    ``seq_len`` get the S-row value (qkv: HiFi2 bf16 x bfp8; o_proj: LoFi bfp8 x bfp8, HiFi2 with ``keep_bf16``)."""
    none = (None, None)
    if batch_size <= 1 or not seq_numerics_active():
        return none, none
    grid = mesh_device.compute_with_storage_grid_size()
    arch = mesh_device.arch()
    hidden = int(weights.wqkv.shape[-2])
    qkv = seq_numerics_matmul(
        total_rows, seq_len, int(weights.wqkv.shape[-1]), hidden, grid, arch, ttnn.MathFidelity.HiFi2
    )
    out = seq_numerics_matmul(
        total_rows,
        seq_len,
        int(weights.o_proj.shape[-1]),
        int(weights.o_proj.shape[-2]),
        grid,
        arch,
        ttnn.MathFidelity.HiFi2 if keep_bf16 else ttnn.MathFidelity.LoFi,
    )
    return qkv, out


def shared_expert_piece_rows(rows):
    """Rows per piece of the shared expert under the knob: the per-user S when a packed pass of ``rows`` > S rows is
    active and S divides ``rows`` (each piece then runs the S-row program configs -- explicit up to
    ``SHARED_EXPERT_CONFIG_MAX_ROWS``, auto above -- exactly as the sequential pass does); None = run ``rows`` as is."""
    seq_len = seq_numerics_rows()
    if seq_len is None or rows <= seq_len or rows % seq_len != 0:
        return None
    return seq_len


def split_rows(x, piece):
    """``[1, 1, R, W]`` -> list of ``[1, 1, piece, W]`` device tensors (``ttnn.split`` along the row axis; a single
    piece returns ``[x]`` itself)."""
    rows = int(x.shape[-2])
    if piece >= rows:
        return [x]
    return ttnn.split(x, piece, dim=2)


def head_seq_numerics(hidden, norm, lm_head_weight):
    """Final norm + lm_head of ``hidden`` ``[1, 1, R, H]`` bf16 (consumed; R a multiple of 32) with the single-user
    head's programs: the norm per 32-row tile (each tile is a ``[1, 1, 32, H]`` interleaved bf16 tensor, so
    ``RMSNorm.forward`` takes its width-sharded decode kernel exactly as for the sequential head's tile) and the lm_head
    in ``packed_prefill.lm_head_piece_rows`` pieces (auto config = the 32-row one). Returns ``[1, 1, R, V / TP]`` bf16 --
    the same shape the R-row head returns, so the Generator's per-user readback is unchanged."""
    rows, hidden_size = int(hidden.shape[-2]), int(hidden.shape[-1])
    if rows % TILE != 0:
        raise ValueError(f"the seq-numerics head needs a tile multiple of rows, got {rows}")
    tiles = split_rows(hidden, TILE)
    if len(tiles) > 1:
        hidden.deallocate(True)  # the tiles are device copies
    normed_tiles = [norm(t) for t in tiles]
    for t in tiles:
        t.deallocate(True)  # frees ``hidden`` itself when it was the single tile
    normed = normed_tiles[0] if len(normed_tiles) == 1 else ttnn.concat(normed_tiles, dim=2)
    if len(normed_tiles) > 1:
        for t in normed_tiles:
            t.deallocate(True)
    vocab_per_device = int(lm_head_weight.shape[-1])
    piece = packed_prefill.lm_head_piece_rows(rows, vocab_per_device, hidden_size)
    pieces = split_rows(normed, piece)
    if len(pieces) > 1:
        normed.deallocate(True)  # the pieces are device copies
    logits_pieces = [ttnn.matmul(p, lm_head_weight, dtype=ttnn.bfloat16) for p in pieces]
    for p in pieces:
        p.deallocate(True)
    if len(logits_pieces) == 1:
        return logits_pieces[0]
    logits = ttnn.concat(logits_pieces, dim=2)
    for p in logits_pieces:
        p.deallocate(True)
    return logits
