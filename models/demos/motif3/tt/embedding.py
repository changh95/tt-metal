# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 token embedding and 4-stream expand on TT (design §2.3.2; WAVE_A_REVIEW EMB-1).

Math (HF ``modeling_motif.py:1379, 1398-1400``; reference ``MotifModel.forward`` / ``expand_streams``)::

    h = embed_tokens(ids)                       # [B, S, 4096] bf16, a pure row gather (bit-exact)
    X = h.unsqueeze(2).expand(B, S, 4, 4096)    # 4 identical residual streams

TT layout (README CONVENTIONS §3): streams are **stream-major** ``[1, 4, T, 4096]`` (reference ``[B, S, 4, D]`` ->
``x.permute(0, 2, 1, 3)``), bf16, TILE, DRAM interleaved.

* **Decode**: chip (dp, tp) produces its DP row's 8 lanes, ``X [1, 4, 8, 4096]`` (rows = local lanes ``l``, i.e. global
  lanes ``8 dp + l``), bitwise identical on the 8 TP chips of the row. The 4 streams come out of the gather itself:
  the per-chip token input is ``[4, 8]`` uint32 (the row's 8 lane tokens once per stream, ROW_MAJOR), so one
  ``ttnn.embedding`` call (row gather + tilize) yields ``[4, 8, 4096]``, and the reshape to ``[1, 4, 8, 4096]`` is a
  view. No repeat / concat op on the decode path.
* **Prefill**: one user, ``S`` = bucket (a multiple of 32), replicated on all 32 chips: tokens ``[1, S]`` uint32 ->
  fused tilized ``ttnn.embedding`` -> ``[1, 1, S, 4096]`` -> ``ttnn.repeat`` -> ``[1, 4, S, 4096]`` ("repeat"), or
  tokens ``[4, S]`` -> one gather -> ``[1, 4, S, 4096]`` ("gather4"). ``prefill_mode="auto"`` (default) uses
  "repeat" below 1024 tokens and "gather4" from 1024 (the measured crossover); ``forward_prefill`` follows the shape
  of the token tensor it is given.

Weights (README §7, design §1.1 memory table): ``model.embed_tokens.weight`` ``[220160, 4096]`` bf16, uploaded
**ROW_MAJOR** (``ttnn.embedding`` requires it; a TILE table would be untilized on every call), DRAM interleaved:

* ``shard_hidden=False`` (default, EMB-1): replicated, 1.80 GB per chip (cache ``global/embed.weight__rep``).
* ``shard_hidden=True`` (memory lever, -1.58 GB per chip): chip ``tp`` holds hidden columns ``[512 tp, +512)``
  (``[220160, 512]``, 225 MB); the gathered ``[.., 512]`` slices are concatenated with ``all_gather(dim=-1, "tp")``
  (one TP all-gather per call; the result is still bitwise identical on the row's chips).

Measured (``tests/unit/test_embed_head.py``, device profiler / traced): decode 25 us device (RM gather + tilize; traced
24.9 us/call, eager 285 us); hidden-sharded decode traced 44.9 us (+ TP all-gather); prefill "repeat" 48.6 us at
S = 128 and 794 us at S = 4096, "gather4" 55.4 / 682 us (so "gather4" is the cheaper one from ~1K tokens up). Table
upload (replicated, 1.8 GB / chip) 0.8-1.5 s; outputs bit-exact against the checkpoint rows on all 32 chips.

The token inputs are built on the host by :meth:`MotifEmbedding.decode_tokens_host` (lane-ordered ``tokens[32]`` ->
per-row ``[4, 8]``; for ``ttnn.copy_host_to_device_tensor`` into a persistent decode-trace input) and
:meth:`MotifEmbedding.prefill_tokens_host` (``[S]`` padded to the bucket). Out-of-range ids are rejected on the
host (``ttnn.embedding`` would read past the table). ``forward_decode`` is trace-safe (README §10 rule 2).

Import rule: ttnn, torch and the shared motif3 infra only; no other ``models/demos/**`` package.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn

from . import weights as W
from .ccl import MotifCCL
from .model_config import MotifTTConfig

EMBED_WEIGHT = "model.embed_tokens.weight"
PREFILL_MODES = ("auto", "repeat", "gather4")
GATHER4_MIN_TOKENS = 1024  # measured crossover (device profiler): repeat 48.6 vs gather4 55.4 us at 128, 794 vs 682 at 4K


def decode_token_rows(tokens: torch.Tensor, cfg: MotifTTConfig, *, n_streams: Optional[int] = None) -> torch.Tensor:
    """Lane-ordered decode tokens ``[max_batch]`` (32) -> ``[dp * n_streams, lanes_per_row]`` int32 host tensor whose
    DP shard ``r`` (rows ``[n_streams r, n_streams (r + 1))``) is row ``r``'s 8 lane tokens repeated once per stream.

    Inactive lanes may hold any valid id (their rows are ignored downstream; the bridge sends 0); a negative id (an
    "inactive" marker) is replaced by ``pad_token_id``. Ids ``>= vocab_size`` raise (``ttnn.embedding`` would read past
    the table)."""
    n_streams = cfg.n_streams if n_streams is None else int(n_streams)
    t = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
    if t.numel() != cfg.max_batch:
        raise ValueError(f"expected {cfg.max_batch} lane tokens, got {t.numel()}")
    t = torch.where(t < 0, torch.full_like(t, int(cfg.pad_token_id)), t)
    check_token_ids(t, cfg)
    rows = t.to(torch.int32).reshape(cfg.dp, 1, cfg.lanes_per_row)  # lanes_to_rows
    return rows.expand(cfg.dp, n_streams, cfg.lanes_per_row).reshape(cfg.dp * n_streams, cfg.lanes_per_row).contiguous()


def prefill_token_row(tokens: torch.Tensor, bucket: int, cfg: MotifTTConfig, *, n_copies: int = 1) -> torch.Tensor:
    """Prompt ``tokens [S]`` -> ``[n_copies, bucket]`` int32, padded with ``pad_token_id`` (the padded positions'
    outputs are don't-care: causal attention never lets them reach positions < S)."""
    t = torch.as_tensor(tokens).reshape(-1).to(torch.int64)
    S = int(t.numel())
    if S < 1 or S > int(bucket):
        raise ValueError(f"prompt of {S} tokens does not fit bucket {bucket}")
    if int(bucket) % 32:
        raise ValueError(f"prefill bucket {bucket} must be a multiple of 32")
    check_token_ids(t, cfg)
    row = torch.full((int(bucket),), int(cfg.pad_token_id), dtype=torch.int32)
    row[:S] = t.to(torch.int32)
    return row.reshape(1, -1).expand(int(n_copies), -1).contiguous()


def check_token_ids(t: torch.Tensor, cfg: MotifTTConfig) -> None:
    if t.numel() and (int(t.min()) < 0 or int(t.max()) >= cfg.vocab_size):
        raise ValueError(f"token ids must be in [0, {cfg.vocab_size}), got [{int(t.min())}, {int(t.max())}]")


class MotifEmbedding:
    """``ttnn.embedding`` over the (replicated or hidden-sharded) bf16 table, producing the 4-stream residual ``X``.

    Args:
        mesh_device: the opened (4, 8) (or (8, 4)) mesh.
        cfg: :class:`MotifTTConfig`.
        source: ``HFWeightLoader`` / ``DictWeightSource`` with ``model.embed_tokens.weight`` ``[V, 4096]``.
        ccl: shared :class:`MotifCCL` (only used with ``shard_hidden``).
        cache: write / reuse the TT weight cache (``global/embed.weight*``); False for random weights.
        shard_hidden: hidden-shard the table over TP (memory lever) instead of replicating it.
        prefill_mode: ``"auto"`` (default), ``"repeat"`` (gather S rows, then ``ttnn.repeat`` x4) or ``"gather4"``
            (gather 4 S rows); decides the token tensor :meth:`prefill_tokens_host` builds.
        memory_config: output memory config (DRAM interleaved, the module-boundary convention).
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        *,
        source,
        ccl: Optional[MotifCCL] = None,
        cache: bool = True,
        shard_hidden: bool = False,
        prefill_mode: str = "auto",
        memory_config=None,
    ):
        if prefill_mode not in PREFILL_MODES:
            raise ValueError(f"prefill_mode must be one of {PREFILL_MODES}, got {prefill_mode!r}")
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.ccl = ccl if ccl is not None else MotifCCL(mesh_device, cfg)
        self.n_streams = int(cfg.n_streams)
        self.hidden = int(cfg.hidden_size)
        self.lanes = int(cfg.lanes_per_row)
        self.shard_hidden = bool(shard_hidden)
        self.prefill_mode = prefill_mode
        self.memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        if self.shard_hidden and self.hidden % (cfg.tp * 32):
            raise ValueError(f"hidden {self.hidden} does not split into tile-aligned TP slices")

        def build() -> torch.Tensor:
            w = source.get(W.hf_name(None, EMBED_WEIGHT))
            if tuple(w.shape) != (cfg.vocab_size, self.hidden):
                raise ValueError(f"{EMBED_WEIGHT} has shape {tuple(w.shape)}, expected {(cfg.vocab_size, self.hidden)}")
            # A row gather: keep the checkpoint's bf16 values as they are (no fp32 round trip needed).
            return w if w.dtype == torch.bfloat16 else w.to(torch.float32)

        self.weight = W.as_tensor(
            build,
            mesh_device=mesh_device,
            cfg=cfg,
            dtype=cfg.dtypes.embedding,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            tp_dim=1 if self.shard_hidden else None,
            cache_name=("embed.weight" if cache else None),
            layer=None,
        )

    # ---- host-side inputs ----------------------------------------------------------------------------------
    def _row_mapper(self):
        """Mapper giving DP row ``r`` the ``r``-th dim-0 chunk, replicated over TP (built once; used every step)."""
        if getattr(self, "_row_mapper_cached", None) is None:
            dims = self.cfg.axes.mesh_dims(dp_dim=0, tp_dim=None)
            self._row_mapper_cached = ttnn.create_mesh_mapper(
                self.mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate() if d is None else ttnn.PlacementShard(d) for d in dims],
                    ttnn.MeshShape(*self.cfg.axes.mesh_shape),
                ),
            )
        return self._row_mapper_cached

    def decode_tokens_host(self, tokens: torch.Tensor):
        """Lane-ordered ``tokens [32]`` -> **host** mesh tensor (chip of DP row ``r``: ``[4, 8]`` uint32 ROW_MAJOR =
        lanes ``8r .. 8r+7`` once per stream) for ``ttnn.copy_host_to_device_tensor(host, persistent_input)``."""
        return ttnn.from_torch(
            decode_token_rows(tokens, self.cfg, n_streams=self.n_streams),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._row_mapper(),
        )

    def decode_tokens_device(self, tokens: torch.Tensor, *, memory_config=None):
        """As :meth:`decode_tokens_host`, uploaded (allocate the persistent decode input once with this)."""
        return ttnn.from_torch(
            decode_token_rows(tokens, self.cfg, n_streams=self.n_streams),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._row_mapper(),
        )

    def prefill_copies(self, bucket: int) -> int:
        """Rows of the prefill token tensor: 4 (one per stream, "gather4") or 1 ("repeat")."""
        if self.prefill_mode == "gather4" or (self.prefill_mode == "auto" and int(bucket) >= GATHER4_MIN_TOKENS):
            return self.n_streams
        return 1

    def prefill_tokens_host(self, tokens: torch.Tensor, bucket: int):
        """Prompt ``tokens [S]`` padded to ``bucket`` -> host tensor replicated on every chip: ``[1, bucket]`` or
        ``[4, bucket]`` uint32 (:meth:`prefill_copies`)."""
        return ttnn.from_torch(
            prefill_token_row(tokens, bucket, self.cfg, n_copies=self.prefill_copies(bucket)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def prefill_tokens_device(self, tokens: torch.Tensor, bucket: int, *, memory_config=None):
        return ttnn.from_torch(
            prefill_token_row(tokens, bucket, self.cfg, n_copies=self.prefill_copies(bucket)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    # ---- device ops ------------------------------------------------------------------------------------
    def _gather(self, tok):
        e = ttnn.embedding(tok, self.weight, layout=ttnn.TILE_LAYOUT, memory_config=self.memory_config)
        if self.shard_hidden:
            full = self.ccl.all_gather(e, len(e.shape) - 1, "tp", memory_config=self.memory_config)
            ttnn.deallocate(e)
            e = full
        return e

    def forward_decode(self, tokens) -> ttnn.Tensor:
        """``tokens``: device ``[4, 8]`` uint32 ROW_MAJOR per chip (:meth:`decode_tokens_device`, persistent).
        Returns ``X [1, 4, 8, 4096]`` bf16 TILE DRAM: this DP row's 8 lanes, 4 identical streams, replicated over TP.
        Trace-safe: one ``ttnn.embedding`` (+ one TP all-gather with ``shard_hidden``) and a view reshape."""
        e = self._gather(tokens)  # [4, 8, D] TILE
        return ttnn.reshape(e, (1, self.n_streams, self.lanes, self.hidden))

    def forward_prefill(self, tokens) -> ttnn.Tensor:
        """``tokens``: device ``[1, S]`` or ``[4, S]`` uint32 replicated (:meth:`prefill_tokens_device`; S = bucket, a
        multiple of 32). Returns ``X [1, 4, S, 4096]`` bf16 TILE DRAM, identical on all 32 chips."""
        S = int(tokens.shape[-1])
        n = int(tokens.shape[0]) if len(tokens.shape) == 2 else 1
        if n not in (1, self.n_streams):
            raise ValueError(f"prefill tokens must be [1, S] or [{self.n_streams}, S], got {list(tokens.shape)}")
        e = self._gather(tokens)  # [n, S, D] TILE (fused tilized gather since S % 32 == 0)
        if n == self.n_streams:
            return ttnn.reshape(e, (1, self.n_streams, S, self.hidden))
        e4 = ttnn.reshape(e, (1, 1, S, self.hidden))
        x = ttnn.repeat(e4, (1, self.n_streams, 1, 1), memory_config=self.memory_config)
        ttnn.deallocate(e4)
        return x

    # ---- device-side greedy feedback (optional, v1 device sampling) ---------------------------------------------
    def decode_tokens_from_device(self, token_ids, *, output_tensor=None):
        """Lane-ordered device token ids ``[1, 1, 1, 32]`` uint32 ROW_MAJOR replicated on every chip (the output of
        ``MotifLMHead.argmax_decode`` in the default vocab split) -> this row's decode input ``[4, 8]`` uint32: the
        DP row keeps its 8 lanes (``partition`` over DP, no fabric traffic) once per stream. With ``output_tensor``
        (the persistent decode input) the result is copied into it in place. Trace-safe; traced 3.0 us per call (slope
        of 64 vs 256 calls per trace, ``test_embedding``)."""
        part = self.ccl.partition(token_ids, 3, "dp")  # [1, 1, 1, 8]
        rows = ttnn.repeat(ttnn.reshape(part, (1, 1, 1, self.lanes)), (1, 1, self.n_streams, 1))  # [1, 1, 4, 8]
        out = ttnn.reshape(rows, (self.n_streams, self.lanes))
        if part is not token_ids:
            ttnn.deallocate(part)
        if output_tensor is not None:
            ttnn.copy(out, output_tensor)
            ttnn.deallocate(out)
            return output_tensor
        return out


__all__ = [
    "EMBED_WEIGHT",
    "GATHER4_MIN_TOKENS",
    "PREFILL_MODES",
    "MotifEmbedding",
    "check_token_ids",
    "decode_token_rows",
    "prefill_token_row",
]
