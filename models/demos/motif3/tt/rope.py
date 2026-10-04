# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RoPE for Motif-3 GDLA on TT (design §2.3.4 steps 5 / prefill 1; study 01 §3.10, 04 §3.6).

Semantics (identical to HF ``modeling_motif.py:270-434`` and the CPU reference ``reference/rope.py``):

* The 64-dim rope slice of every q head and the shared 64-dim ``k_pe`` are rotated with the HF / NeoX
  **half-split** convention: pairs ``(i, i + 32)``, ``rotate_half(x) = cat(-x[32:], x[:32])``,
  ``cos/sin = cat(freqs, freqs)``. The checkpoint stores the rope rows de-interleaved, so **no weight permutation**
  is needed (study 01 §3.10, Appendix A.3).
* Global layers (``l % 4 == 0``): YaRN ``inv_freq`` (factor 64, original 4096, beta 32/1 -> correction range
  [10, 23]); always on because ``max_position_embeddings (262144) > 4096``. **No mscale on cos/sin**; the YaRN
  mscale lives in the softmax scale (``MotifTTConfig.softmax_scale``).
* SWA layers: plain RoPE, ``swa_rope_theta = 1e4``.
* Precision: angles ``pos * inv_freq`` in fp32 (HF's K=1 matmul is the same single multiply), cos/sin rounded
  to bf16 (the activation dtype) -- the device tables below are exactly HF's bf16 cos/sin.

Device layouts (all tables replicated on every chip):

====================  ===============================  ===================================================
use                   tensor                           how it is produced
====================  ===============================  ===================================================
decode, lanes on rows ``cos/sin [1, 1, 32, 64]`` TILE  ``ttnn.embedding(rot_idxs [1,32] uint32, table)``:
                      (rows 0..7 = this row's lanes;   one gather per step, trace-safe (DeepSeek pattern)
                      T64: rows 0..15)
decode, lanes on dim1 ``cos/sin [1, L, 1, 64]`` TILE   transpose + slice of the above; optionally
                                                       HEIGHT_SHARDED (one lane per core) for
                                                       ``rotary_embedding_hf(..., is_decode_mode=True)``
prefill               ``cos/sin [1, 1, S, 64]`` TILE   host-built table slice for positions ``0..S-1``
prefill chunk at any  ``cos/sin [1, 1, C, 64]`` TILE   ``ttnn.embedding(rot_idxs [1,C] uint32, table)``:
start (offset RoPE)                                    one gather per chunk for every layer
                                                       (``chunk_rope_tables``; features design §3.2.4)
====================  ===============================  ===================================================

Two ways to apply it:

* ``ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=...)`` -- HF half-split kernel. Prefill
  mode takes ``x [1, H, S, 64]`` with ``cos [1, 1, S', 64]`` (S' >= S); decode mode needs ``x [1, L, H, 64]``
  HEIGHT_SHARDED and ``cos [1, L, 1, 64]`` sharded. Prefill mode also works for decode with lanes on rows
  (``x [1, H, 32, 64]``, ``cos [1, 1, 32, 64]``: row t is rotated by row t's position).
* composite ``x * cos + (x @ R) * sin`` with the signed permutation ``R`` (``x @ R == rotate_half(x)``),
  :meth:`MotifRope.apply_composite`; broadcasting handles the head dim.

Gate G8 decided (``tests/unit/gates/GATES_RESULTS.md`` §10): attention uses the **fused**
``rotary_embedding_hf`` with the ``rope`` compute role (HiFi4 + fp32 acc: PCC >= 0.999996, <= 3 us traced in decode,
halves the max error vs the op default). Wave B1 (``tt/attention.py``) runs it in **prefill mode in decode too**: the
heads-on-dim-1 / lanes-on-rows ``q_pe [1, 10, 8, 64]`` and ``k_pe`` with ``decode_cos_sin(kind, rot, layout="rows")``
(row t rotated by lane t's position; the decode-mode variant would need a transpose + reshard of q_pe and k_pe, 4 extra
ops). The tables are built ONCE per decode step for all 53 layers (``MotifAttention.decode_rope_tables(rope,
rot_idxs)``). The decode-mode layout (q_pe ``[1, 8, 10, 64]`` / k_pe ``[1, 8, 1, 64]`` HEIGHT_SHARDED ``[32, 64]`` on 8
cores, :meth:`MotifRope.batch_sharded_memory_config`, ``layout="batch_sharded"``) stays available and tested; prefill:
``[1, H, S, 64]`` with ``prefill_cos_sin(kind, S)``. The composite (19 us traced) stays here as the fallback.

The tables come from the config's YaRN fields, which ``MotifTTConfig.from_hf_config`` reads from ``rope_scaling`` or
the transformers-5 ``rope_parameters`` (INFRA-1), so a config built from vLLM's ``hf_config`` object gives the same
``inv_freq_for_kind(cfg, "yarn")`` as one built from ``config.json``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch

import ttnn

from .model_config import MotifTTConfig

KINDS = ("yarn", "plain")


# ------------------------------------------------------------------------------------------------------------
# inv_freq (host, fp32) -- op-for-op copies of the HF formulas
# ------------------------------------------------------------------------------------------------------------
def plain_inv_freq(dim: int = 64, theta: float = 1e4) -> torch.Tensor:
    """``1 / theta^(2i/dim)`` fp32 ``[dim // 2]`` (HF ``MotifRotaryEmbedding`` default branch, :331-334)."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))


def yarn_correction_range(
    dim: int = 64, theta: float = 1e4, original_max_pos: int = 4096, beta_fast: float = 32.0, beta_slow: float = 1.0
) -> Tuple[int, int]:
    """``(low, high)`` of the YaRN ramp (HF ``find_correction_range``): (10, 23) for Motif-3."""

    def corr_dim(num_rotations: float) -> float:
        return dim * math.log(original_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = math.floor(corr_dim(beta_fast))
    high = math.ceil(corr_dim(beta_slow))
    return max(low, 0), min(high, dim - 1)


def yarn_inv_freq(
    dim: int = 64,
    theta: float = 1e4,
    max_pos: int = 262144,
    original_max_pos: int = 4096,
    factor: float = 64.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """YaRN ``inv_freq`` fp32 ``[dim // 2]`` == HF ``_compute_yarn_inv_freq`` (:270-306): dims < low unchanged,
    dims > high divided by ``factor``, linear blend in between."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if max_pos > original_max_pos:
        low, high = yarn_correction_range(dim, theta, original_max_pos, beta_fast, beta_slow)
        lo, hi = float(low), float(high)
        if lo == hi:
            hi += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - lo) / (hi - lo), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def inv_freq_for_kind(cfg: MotifTTConfig, kind: str) -> torch.Tensor:
    if kind == "yarn":
        return yarn_inv_freq(
            cfg.rope_dim,
            cfg.yarn_theta,
            cfg.max_position_embeddings,
            cfg.yarn_original_max_pos,
            cfg.yarn_factor,
            cfg.yarn_beta_fast,
            cfg.yarn_beta_slow,
        )
    if kind == "plain":
        theta = cfg.swa_rope_theta if cfg.swa_rope_theta is not None else cfg.rope_theta
        return plain_inv_freq(cfg.rope_dim, theta)
    raise ValueError(f"unknown rope kind {kind!r}")


def inv_freq_for_layer(cfg: MotifTTConfig, layer_idx: int) -> torch.Tensor:
    return inv_freq_for_kind(cfg, cfg.layer(layer_idx).rope_kind)


# ------------------------------------------------------------------------------------------------------------
# tables and torch helpers
# ------------------------------------------------------------------------------------------------------------
def cos_sin_table(
    inv_freq: torch.Tensor, positions: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``cos/sin [..., dim]`` for integer ``positions [...]``: fp32 angles, ``cat(freqs, freqs)``, rounded to
    ``dtype`` (bit-equal to HF ``MotifRotaryEmbedding.forward`` in bf16)."""
    freqs = positions.to(torch.float32)[..., None] * inv_freq.to(torch.float32)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF math: upcast to fp32, ``x cos + rotate_half(x) sin``, cast back. ``cos/sin`` broadcast against ``x``."""
    xf = x.to(torch.float32)
    return (xf * cos.to(torch.float32) + rotate_half(xf) * sin.to(torch.float32)).to(x.dtype)


def rotate_half_matrix(dim: int = 64, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Signed permutation ``R [dim, dim]`` with ``x @ R == rotate_half(x)``: ``R[j+h, j] = -1``, ``R[j, j+h] = 1``."""
    h = dim // 2
    R = torch.zeros(dim, dim, dtype=dtype)
    idx = torch.arange(h)
    R[idx + h, idx] = -1
    R[idx, idx + h] = 1
    return R


# ------------------------------------------------------------------------------------------------------------
# per-lane decode inputs (host side)
# ------------------------------------------------------------------------------------------------------------
def decode_rows_per_dp(cfg: MotifTTConfig, rows_per_dp: Optional[int] = None) -> int:
    """Decode rows per DP row: ``cfg.lanes_per_row`` (8 lanes) by default, or ``rows_per_dp`` (the T64 verify step's
    16 = ``[8 anchors | 8 drafts]``, ``docs/p5_t64/P5_T64_DESIGN.md`` §4.1), at most one 32-row tile row (the ``[1,
    32]`` index row of :func:`positions_to_rot_idxs`). ``ValueError`` outside ``[1, 32]``."""
    r = int(cfg.lanes_per_row) if rows_per_dp is None else int(rows_per_dp)
    if not 1 <= r <= 32:
        raise ValueError(f"rows_per_dp must be in [1, 32] (one tile row per DP row), got {rows_per_dp}")
    return r


def lanes_to_rows(
    values: torch.Tensor,
    cfg: MotifTTConfig,
    *,
    pad_to: Optional[int] = None,
    fill=0,
    rows_per_dp: Optional[int] = None,
) -> torch.Tensor:
    """Row-ordered ``values [dp * n, ...]`` -> ``[dp, n', ...]``: DP row ``r`` holds entries ``n r .. n r + n - 1``
    (design §2.3.10), padded with ``fill`` to ``n' = pad_to`` entries (``pad_to=None`` keeps ``n``). ``n`` =
    :func:`decode_rows_per_dp` (``rows_per_dp``): by default the 32 lane-ordered values, row ``r`` = lanes ``8r ..
    8r+7``; ``rows_per_dp=16`` takes the T64 step's 64 rows in physical order (row ``r`` = rows ``16r .. 16r+15``,
    ``[8 anchors | 8 drafts]``)."""
    L = decode_rows_per_dp(cfg, rows_per_dp)
    B = cfg.dp * L
    if values.shape[0] != B:
        raise ValueError(f"expected {B} lanes ({cfg.dp} DP rows x {L}), got {values.shape[0]}")
    rows = values.reshape(cfg.dp, L, *values.shape[1:])
    if pad_to is not None and pad_to > L:
        pad = torch.full((cfg.dp, pad_to - L, *values.shape[1:]), fill, dtype=values.dtype)
        rows = torch.cat([rows, pad], dim=1)
    return rows


def positions_to_rot_idxs(
    positions: torch.Tensor, cfg: MotifTTConfig, *, rows_per_dp: Optional[int] = None
) -> torch.Tensor:
    """Decode positions ``[dp * n]`` (``-1`` = inactive) -> rot table indices ``[dp, 32]`` int32: each DP row's ``n``
    rows first, then ``32 - n`` pad entries; inactive rows and pads read row 0 (their output is masked later). ``n`` =
    :func:`decode_rows_per_dp`: by default the 32 lane positions (lane order, 8 per DP row); ``rows_per_dp=16`` takes
    the T64 step's 64 rows in physical order (anchors at ``n``, drafts at ``n + 1``), so row ``t`` of the gathered
    ``[1, 1, 32, 64]`` tables rotates row ``t`` of the step's ``[1, 1, 16, 4096]`` input."""
    pos = torch.as_tensor(positions, dtype=torch.int64).clamp_min(0)
    if int(pos.max()) >= cfg.max_model_len:
        raise ValueError(f"position {int(pos.max())} >= max_model_len {cfg.max_model_len}")
    return lanes_to_rows(pos.to(torch.int32), cfg, pad_to=32, fill=0, rows_per_dp=rows_per_dp)


def chunk_rot_rows(positions, max_positions: int) -> torch.Tensor:
    """Table rows of a prefill chunk's RoPE gather: ``int32 [1, C]`` from the rows' absolute ``positions [C]`` (for
    an sp1 chunk ``prefill_plan.rope_positions(chunk, max_positions)``, which clamps the padded rows; any positions
    work, e.g. several requests' rows concatenated). Every position must lie in ``[0, max_positions)`` and ``C`` must
    be a positive multiple of 32 (the gathered tables are TILE)."""
    pos = torch.as_tensor(positions)
    C = int(pos.numel())
    if C == 0 or C % 32:
        raise ValueError(f"a chunk's RoPE rows must be a positive multiple of 32, got {C}")
    if pos.dtype.is_floating_point or pos.dtype == torch.bool:
        raise TypeError(f"chunk positions must be integers, got {pos.dtype}")
    pos = pos.reshape(-1).to(torch.int64)
    lo, hi = int(pos.min()), int(pos.max())
    if lo < 0 or hi >= int(max_positions):
        raise ValueError(f"chunk positions [{lo}, {hi}] outside the {int(max_positions)}-row RoPE tables")
    return pos.to(torch.int32)[None].contiguous()


def shard_lanes(
    rows: torch.Tensor,
    cfg: MotifTTConfig,
    mesh_device,
    *,
    dtype=ttnn.int32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=None,
    memory_config=None,
):
    """``[dp, ...]`` host tensor -> mesh tensor whose DP row ``r`` holds ``rows[r]`` (shape ``[1, ...]``), replicated
    over TP. ``device=None`` returns a host mesh tensor for ``ttnn.copy_host_to_device_tensor`` into a persistent
    device input (trace replay); pass ``device=mesh_device`` to upload directly."""
    dims = cfg.axes.mesh_dims(dp_dim=0, tp_dim=None)
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate() if d is None else ttnn.PlacementShard(d) for d in dims],
            ttnn.MeshShape(*cfg.axes.mesh_shape),
        ),
    )
    return ttnn.from_torch(
        rows,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=(memory_config or ttnn.DRAM_MEMORY_CONFIG) if device is not None else None,
        mesh_mapper=mapper,
    )


# ------------------------------------------------------------------------------------------------------------
# device tables
# ------------------------------------------------------------------------------------------------------------
class MotifRope:
    """Device-resident RoPE tables and helpers (one instance per model; ~16 MB per chip for 32K positions).

    * ``tables[kind] = (cos, sin)``: ``[1, 1, P, 64]`` bf16 ROW_MAJOR DRAM, replicated -- the ``ttnn.embedding``
      weight (must be ROW_MAJOR, otherwise the op untilizes the whole table every call).
    * ``rot_mat``: ``R [1, 1, 64, 64]`` bf16 TILE (composite path).
    * prefill tables ``[1, 1, S, 64]`` TILE are built on demand per (kind, S) and cached.
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        *,
        kinds: Sequence[str] = KINDS,
        max_positions: Optional[int] = None,
        compute_kernel_config=None,
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.dim = cfg.rope_dim
        self.max_positions = int(max_positions or cfg.max_model_len)
        self.ckc = compute_kernel_config or cfg.compute_config("rope")
        self._replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        self.inv_freq: Dict[str, torch.Tensor] = {k: inv_freq_for_kind(cfg, k) for k in kinds}
        pos = torch.arange(self.max_positions)
        self.host_tables: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
            k: cos_sin_table(self.inv_freq[k], pos, torch.bfloat16) for k in kinds
        }
        self.tables = {
            k: tuple(self._upload(t[None, None], ttnn.ROW_MAJOR_LAYOUT) for t in self.host_tables[k]) for k in kinds
        }
        self.rot_mat = self._upload(rotate_half_matrix(self.dim, torch.float32)[None, None], ttnn.TILE_LAYOUT)
        self._prefill: Dict[Tuple[str, int], Tuple[ttnn.Tensor, ttnn.Tensor]] = {}

    def _upload(self, t: torch.Tensor, layout):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=layout,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate,
        )

    # ---- decode ----------------------------------------------------------------------------------------
    def rot_idxs_host(self, positions: torch.Tensor, *, rows_per_dp: Optional[int] = None):
        """Host mesh tensor ``[1, 32]`` uint32 per chip (DP row r: its rows, then pad; :func:`positions_to_rot_idxs`)
        for ``ttnn.copy_host_to_device_tensor`` into the persistent decode input. ``positions``: the 32 lane positions,
        or with ``rows_per_dp=16`` the T64 step's 64 row positions (physical order ``16 r + j``)."""
        return shard_lanes(
            positions_to_rot_idxs(positions, self.cfg, rows_per_dp=rows_per_dp),
            self.cfg,
            self.mesh_device,
            dtype=ttnn.uint32,
            device=None,
        )

    def rot_idxs_device(self, positions: torch.Tensor, *, rows_per_dp: Optional[int] = None):
        """:meth:`rot_idxs_host` uploaded to the mesh (a device tensor ``[1, 32]`` uint32 per chip)."""
        return shard_lanes(
            positions_to_rot_idxs(positions, self.cfg, rows_per_dp=rows_per_dp),
            self.cfg,
            self.mesh_device,
            dtype=ttnn.uint32,
            device=self.mesh_device,
        )

    def decode_cos_sin(self, kind: str, rot_idxs, *, layout: str = "rows", memory_config=None):
        """Trace-safe per-lane gather (device ops only).

        ``layout="rows"``: ``[1, 1, 32, 64]`` TILE, row t = row t of this DP row (lane t; in the T64 step row t of
        ``[8 anchors | 8 drafts]``); the pad rows past the used ones hold position 0.
        ``layout="batch"``: ``[1, L, 1, 64]`` TILE (L = ``cfg.lanes_per_row``, 8 lanes; the T64 step uses the rows
        layout). ``layout="batch_sharded"``: the same, HEIGHT_SHARDED one lane per core (``rotary_embedding_hf`` decode
        mode).
        """
        cos_t, sin_t = self.tables[kind]
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        out = []
        for table in (cos_t, sin_t):
            v = ttnn.embedding(rot_idxs, table, layout=ttnn.TILE_LAYOUT, memory_config=mc)  # [1, 32, 64]
            v = ttnn.reshape(v, (1, 1, 32, self.dim))
            if layout != "rows":
                v = ttnn.transpose(v, 1, 2)  # [1, 32, 1, 64]
                L = self.cfg.lanes_per_row
                if L != 32:
                    v = ttnn.slice(v, (0, 0, 0, 0), (1, L, 1, self.dim))
                if layout == "batch_sharded":
                    v = ttnn.to_memory_config(v, self.batch_sharded_memory_config())
                elif layout != "batch":
                    raise ValueError(f"unknown layout {layout!r}")
            out.append(v)
        return out[0], out[1]

    def batch_sharded_memory_config(self, n_heads_padded: int = 32):
        """HEIGHT_SHARDED, one lane per core: shard ``[32, 64]`` on L cores (decode-mode ``rotary_embedding_hf``)."""
        grid = self.mesh_device.compute_with_storage_grid_size()
        cores = ttnn.num_cores_to_corerangeset(self.cfg.lanes_per_row, grid, row_wise=True)
        return ttnn.create_sharded_memory_config(
            shape=(n_heads_padded, self.dim),
            core_grid=cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # ---- prefill chunks at any position (offset RoPE; features design §3.2.4) -----------------------------------
    def chunk_rot_idxs_host(self, positions):
        """Host mesh tensor ``[1, C]`` uint32 (replicated; :func:`chunk_rot_rows`) for
        ``ttnn.copy_host_to_device_tensor`` into a persistent chunk input."""
        return ttnn.from_torch(
            chunk_rot_rows(positions, self.max_positions),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._replicate,
        )

    def chunk_rot_idxs_device(self, positions):
        """``[1, C]`` uint32 ROW_MAJOR DRAM device tensor (replicated): the rows' table indices
        (:func:`chunk_rot_rows`)."""
        return ttnn.from_torch(
            chunk_rot_rows(positions, self.max_positions),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate,
        )

    def chunk_cos_sin(self, kind: str, rot_idxs, *, memory_config=None):
        """``cos/sin [1, 1, C, 64]`` TILE whose row ``i`` is table row ``rot_idxs[0, i]`` (device ops only, so one
        program per ``C`` whatever the positions, trace-safe): ``ttnn.embedding`` on the ROW_MAJOR tables, the decode
        gather with ``C`` rows (gate G11b). The values are exact copies of the host bf16 table rows; for rows ``0 ..
        C-1`` they equal :meth:`prefill_cos_sin` (``kind``, ``C``) bitwise. ``rot_idxs``: ``[1, C]`` uint32
        (:meth:`chunk_rot_idxs_device`). The caller frees the two tensors."""
        cos_t, sin_t = self.tables[kind]
        C = int(rot_idxs.shape[-1])
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        out = []
        for table in (cos_t, sin_t):
            v = ttnn.embedding(rot_idxs, table, layout=ttnn.TILE_LAYOUT, memory_config=mc)  # [1, C, 64]
            out.append(ttnn.reshape(v, (1, 1, C, self.dim)))
        return out[0], out[1]

    def chunk_rope_tables(self, rot_idxs, kinds: Optional[Sequence[str]] = None) -> Dict[str, Tuple]:
        """``{kind: (cos, sin)}`` of :meth:`chunk_cos_sin` for every kind (default: all tables): build it once per
        chunk and pass it as ``rot=`` to every layer (``MotifAttention.fill_kv``; the sp1 prefill paths)."""
        return {k: self.chunk_cos_sin(k, rot_idxs) for k in (kinds if kinds is not None else tuple(self.tables))}

    # ---- prefill -----------------------------------------------------------------------------------------
    def prefill_cos_sin(self, kind: str, seq_len: int):
        """``cos/sin [1, 1, S, 64]`` TILE for positions ``0..S-1`` (cached per bucket; prefill starts at 0)."""
        key = (kind, int(seq_len))
        if key not in self._prefill:
            if seq_len > self.max_positions:
                raise ValueError(f"seq_len {seq_len} > table length {self.max_positions}")
            cos, sin = self.host_tables[kind]
            self._prefill[key] = tuple(self._upload(t[:seq_len][None, None], ttnn.TILE_LAYOUT) for t in (cos, sin))
        return self._prefill[key]

    def release_prefill_tables(self) -> None:
        for c, s in self._prefill.values():
            ttnn.deallocate(c)
            ttnn.deallocate(s)
        self._prefill.clear()

    # ---- application -----------------------------------------------------------------------------------
    def apply_composite(self, x, cos, sin, *, memory_config=None, fp32_math: bool = True):
        """``x cos + (x @ R) sin``; ``cos/sin`` broadcast over ``x``'s head dim (``[1,1,T,64]`` vs ``[1,H,T,64]``, or
        ``[1,L,1,64]`` vs ``[1,L,H,64]``). With ``fp32_math`` the two products and the sum are fp32 and the result is
        cast once to ``x.dtype`` (HF does the rotation in fp32)."""
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        rot = ttnn.matmul(x, self.rot_mat, memory_config=mc, compute_kernel_config=self.ckc)  # exact: signed perm
        if fp32_math:
            a = ttnn.multiply(x, cos, dtype=ttnn.float32, memory_config=mc)
            b = ttnn.multiply(rot, sin, dtype=ttnn.float32, memory_config=mc)
            ttnn.deallocate(rot)
            out = ttnn.add(a, b, dtype=x.dtype, memory_config=mc)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            return out
        a = ttnn.multiply(x, cos, memory_config=mc)
        b = ttnn.multiply(rot, sin, memory_config=mc)
        ttnn.deallocate(rot)
        out = ttnn.add(a, b, memory_config=mc)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        return out

    def apply_hf(self, x, cos, sin, *, is_decode_mode: bool = False, memory_config=None):
        """``ttnn.experimental.rotary_embedding_hf`` (HF half-split kernel); see the module docstring for shapes."""
        return ttnn.experimental.rotary_embedding_hf(
            x, cos, sin, is_decode_mode=is_decode_mode, memory_config=memory_config, compute_kernel_config=self.ckc
        )


__all__ = [
    "KINDS",
    "MotifRope",
    "apply_rope_torch",
    "chunk_rot_rows",
    "cos_sin_table",
    "decode_rows_per_dp",
    "inv_freq_for_kind",
    "inv_freq_for_layer",
    "lanes_to_rows",
    "plain_inv_freq",
    "positions_to_rot_idxs",
    "rotate_half",
    "rotate_half_matrix",
    "shard_lanes",
    "yarn_correction_range",
    "yarn_inv_freq",
]
