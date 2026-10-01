# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MLA latent KV cache used by every GDLA layer (global and SWA alike).

Per token and layer the cache holds 576 values: ``c = kv_norm(c_raw)`` (512, gamma included, exactly what the
fork caches, fork motif.py:694 + mla_attention.py:546-553) and the RoPE-rotated shared ``k_pe`` (64). Both the
expanded attention form (K/V re-expanded through ``wkv_b``) and the absorbed form read the same cache, so the
two are interchangeable mid-sequence.

Slots are indexed by ABSOLUTE position (slot t <-> position t). Masks are built from positions, which makes
causal + sliding-window masking identical for prefill, chunked prefill and decode, and lets a batch of users sit
at different positions during decode (each user only sees slots ``<= its own position``). SWA layers keep every
slot (no ring buffer); the window is enforced by the mask, which is the simplest exact behaviour.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch


class LatentKVCache:
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        kv_lora_rank: int,
        rope_dim: int,
        dtype: torch.dtype,
        window: Optional[int] = None,
    ):
        self.c = torch.zeros(batch_size, max_seq_len, kv_lora_rank, dtype=dtype)
        self.k_pe = torch.zeros(batch_size, max_seq_len, rope_dim, dtype=dtype)
        self.seq_lens = torch.zeros(batch_size, dtype=torch.long)  # max written position + 1, per user
        self.window = window  # informational (keys incl. current); masking happens in attention
        self.max_seq_len = max_seq_len

    @property
    def batch_size(self) -> int:
        return self.c.shape[0]

    def reset(self) -> None:
        self.c.zero_()
        self.k_pe.zero_()
        self.seq_lens.zero_()

    def update(self, c: torch.Tensor, k_pe: torch.Tensor, positions: torch.Tensor) -> None:
        """Write ``c [B, S, r]`` and roped ``k_pe [B, S, rope]`` at slots ``positions [B, S]``."""
        B, S = positions.shape
        if B != self.batch_size:
            raise ValueError(f"cache batch {self.batch_size} != input batch {B}")
        if int(positions.max()) >= self.max_seq_len:
            raise ValueError(f"position {int(positions.max())} exceeds cache max_seq_len {self.max_seq_len}")
        if bool((positions[:, 0] > self.seq_lens).any()):
            raise ValueError("cache writes must be contiguous: a chunk starts after the last written position")
        b_idx = torch.arange(B)[:, None].expand(B, S)
        self.c[b_idx, positions] = c.to(self.c.dtype)
        self.k_pe[b_idx, positions] = k_pe.to(self.k_pe.dtype)
        self.seq_lens.copy_(torch.maximum(self.seq_lens, positions.max(dim=1).values + 1))  # in place (views)

    def user_view(self, b: int) -> "LatentKVCache":
        """A batch-1 cache sharing user ``b``'s storage (per-user prefill into a batch cache)."""
        view = LatentKVCache.__new__(LatentKVCache)
        view.c, view.k_pe, view.seq_lens = self.c[b : b + 1], self.k_pe[b : b + 1], self.seq_lens[b : b + 1]
        view.window, view.max_seq_len = self.window, self.max_seq_len
        return view

    def keys(self, upto: int):
        """Cached ``(c [B, T, r], k_pe [B, T, rope], key_positions [T])`` for slots ``[0, upto)``."""
        return self.c[:, :upto], self.k_pe[:, :upto], torch.arange(upto)


class MotifKVCache:
    """One :class:`LatentKVCache` per (built) decoder layer, keyed by absolute layer index."""

    def __init__(self, layers: Dict[int, LatentKVCache]):
        self.layers = dict(layers)

    @classmethod
    def create(
        cls, args, layer_ids: Iterable[int], batch_size: int, max_seq_len: int, dtype: torch.dtype
    ) -> "MotifKVCache":
        return cls(
            {
                int(i): LatentKVCache(
                    batch_size, max_seq_len, args.kv_lora_rank, args.qk_rope_head_dim, dtype, args.attention_window(i)
                )
                for i in layer_ids
            }
        )

    def __getitem__(self, layer_idx: int) -> LatentKVCache:
        return self.layers[int(layer_idx)]

    def get(self, layer_idx: int) -> Optional[LatentKVCache]:
        return self.layers.get(int(layer_idx))

    def reset(self) -> None:
        for layer in self.layers.values():
            layer.reset()

    def user_view(self, b: int) -> "MotifKVCache":
        return MotifKVCache({i: layer.user_view(b) for i, layer in self.layers.items()})
