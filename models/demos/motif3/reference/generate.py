# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KV-cache prefill/decode for the reference model (a correct replacement for HF ``generate()``).

HF ``generate()`` is NOT a valid Motif-3 decode reference past 128 tokens: it builds ``DynamicCache(config=...)``,
which turns all 53 layers (including the 14 global ones) into 127-token sliding layers (study §3.12). This module
keeps one latent cache per layer, indexed by absolute position, and masks with each layer's own window (global:
full causal; SWA: 129 keys incl. the current one), so decode equals the full-sequence forward at every position.

    gen = MotifGenerator(model, batch_size=1, max_seq_len=4096)
    logits = gen.prefill(prompt_ids)          # [B, S, V] fp32, positions 0..S-1
    next_logits = gen.decode(next_ids)        # [B, V] fp32, one token per user
    tokens = gen.generate(prompt_ids, max_new_tokens=32)  # greedy unless sample=True

Prefill of a batch needs equal-length prompts (no padding support); prefill can be chunked by calling
``prefill`` repeatedly. Users of one batch may be advanced to different positions with ``decode(positions=...)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .modules import MotifForCausalLM


class MotifGenerator:
    def __init__(
        self,
        model: MotifForCausalLM,
        batch_size: int = 1,
        max_seq_len: int = 4096,
        attn_mode: str = "absorbed",
        cache_dtype: Optional[torch.dtype] = None,
    ):
        self.model = model
        self.attn_mode = attn_mode
        self.cache = model.new_cache(batch_size, max_seq_len, cache_dtype)
        self.positions = torch.zeros(batch_size, dtype=torch.long)  # next position per user

    @property
    def batch_size(self) -> int:
        return self.positions.shape[0]

    def reset(self) -> None:
        self.cache.reset()
        self.positions.zero_()

    @torch.no_grad()
    def prefill(
        self, tokens: torch.Tensor, last_token_only: bool = False, tap=None, user: Optional[int] = None
    ) -> torch.Tensor:
        """Append ``tokens [B, S]`` (same length for every user) at the users' current positions.

        With ``user=b`` only that user's prompt ``tokens [S]`` is prefilled (into its rows of the batch cache), which
        is how users of one batch get different prompt lengths before a batched ``decode``.
        """
        if tokens.dim() == 1:
            tokens = tokens[None]
        B, S = tokens.shape
        if user is not None:
            if B != 1:
                raise ValueError("per-user prefill takes a single prompt")
            positions = self.positions[user] + torch.arange(S)[None, :]
            logits = self.model(
                tokens, positions, self.cache.user_view(user), self.attn_mode, tap=tap, last_token_only=last_token_only
            )
            self.positions[user] += S
            return logits
        if B != self.batch_size:
            raise ValueError(f"batch {B} != generator batch {self.batch_size}")
        positions = self.positions[:, None] + torch.arange(S)[None, :]
        logits = self.model(tokens, positions, self.cache, self.attn_mode, tap=tap, last_token_only=last_token_only)
        self.positions += S
        return logits

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor, positions: Optional[torch.Tensor] = None, tap=None) -> torch.Tensor:
        """One token per user (``tokens [B]``) -> logits ``[B, V]``. ``positions`` overrides the tracked positions."""
        tokens = tokens.reshape(self.batch_size, 1)
        pos = self.positions if positions is None else positions.reshape(self.batch_size).long()
        logits = self.model(tokens, pos[:, None], self.cache, self.attn_mode, tap=tap)[:, 0]
        self.positions = pos + 1
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        eos_token_ids: Optional[Sequence[int]] = None,
        sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> List[List[int]]:
        """Prefill ``prompt [B, S]`` and decode up to ``max_new_tokens`` per user (greedy by default)."""
        eos = set(self.model.args.eos_token_ids if eos_token_ids is None else eos_token_ids)
        logits = self.prefill(prompt, last_token_only=True)[:, -1]
        out: List[List[int]] = [[] for _ in range(self.batch_size)]
        done = torch.zeros(self.batch_size, dtype=torch.bool)
        for _ in range(max_new_tokens):
            nxt = _pick(logits, sample, temperature, top_p, generator)
            for b in range(self.batch_size):
                if not done[b]:
                    out[b].append(int(nxt[b]))
                    done[b] = int(nxt[b]) in eos
            if bool(done.all()):
                break
            logits = self.decode(nxt)
        return out


def _pick(logits: torch.Tensor, sample: bool, temperature: float, top_p: float, generator) -> torch.Tensor:
    if not sample:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
    if top_p < 1.0:
        sorted_p, sorted_i = probs.sort(dim=-1, descending=True)
        keep = sorted_p.cumsum(-1) - sorted_p < top_p
        sorted_p = sorted_p * keep
        probs = torch.zeros_like(probs).scatter(-1, sorted_i, sorted_p)
        probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
