# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host side of the speculative-decoding VERIFY step: the row grid, the accept rule and the per-step commit.

Pure torch / python (no ttnn import) so the value logic is unit-testable without a device. The device half
(``verify_step.VerifyStep``) uploads what the builders here produce and hands back one argmax per row.

Row grid (docs of the Flash-Next ``mtp_slots.py`` oracle, reused): ``w`` users x ``T = k+1`` rows, tile row
``r = s*T + j`` is user ``s``'s row ``j``. Row 0 carries the user's last committed token ``t'_s`` (the bonus
token, not yet fed to the model), rows ``1..k`` the drafts ``d_1..d_k``; row ``j`` sits at absolute position
``P_s + j`` where ``P_s`` is the user's committed position (its KV cache and GDN state reflect every token
at a position ``< P_s``, i.e. every committed token EXCEPT ``t'_s``, which the step feeds as row 0).

Accept rule (``accept_drafts``): ``a_s`` = the longest prefix with ``argmax[j] == d_{j+1}``; the user commits
``argmax[0..a_s]`` (``= [d_1 .. d_a, argmax_a]``, ``a_s + 1`` tokens), its position becomes ``P_s + a_s + 1``
and ``argmax_a`` is its next row-0 token.

Lazy GDN prefix: the GDN kernel commits the PREVIOUS step's rows ``1..a_s`` (from that step's saved
projections) plus the current row 0 to the per-user recurrent state, so the host uploads the previous step's
``a_s`` (``accept_prev``) with every step; zeros on the first step.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

TILE = 32


def grid_rows(users: int, T: int) -> int:
    """R = tile-padded users*T: exactly users*T when it fits one tile (<= 32), else rounded up to a tile multiple."""
    n = users * T
    return n if n <= TILE else -(-n // TILE) * TILE


def row(s: int, j: int, T: int) -> int:
    return s * T + j


def accept_drafts(argmaxes: Sequence[int], drafts: Sequence[int]) -> int:
    """Longest prefix with argmax[j] == draft[j] (the batch-1 MTP verify rule of mtp_slots.accept_drafts)."""
    k = len(drafts)
    if len(argmaxes) < k + 1:
        raise ValueError(f"{k} drafts need {k + 1} verify argmaxes, got {len(argmaxes)}")
    a = 0
    while a < k and int(argmaxes[a]) == int(drafts[a]):
        a += 1
    return a


def row_tokens(tokens: Sequence[Sequence[int]], T: int, R: int, pad_token: int = 0) -> torch.Tensor:
    """[R] int32 token column: user s's [x_0, d_1..d_k] in rows s*T..s*T+k; pad_token on tile-padding rows."""
    col = torch.full((R,), int(pad_token), dtype=torch.int32)
    for s, toks in enumerate(tokens):
        assert len(toks) == T, f"user {s}: {len(toks)} tokens, grid rows per user is {T}"
        for j, t in enumerate(toks):
            col[row(s, j, T)] = int(t)
    return col


def row_positions(positions: Sequence[int], T: int, R: int, pad_pos: int = 0) -> torch.Tensor:
    """[R] int32 absolute position column: P_s + j for row (s, j); pad_pos on tile-padding rows."""
    col = torch.full((R,), int(pad_pos), dtype=torch.int32)
    for s, p in enumerate(positions):
        for j in range(T):
            col[row(s, j, T)] = int(p) + j
    return col


def offset_positions(positions: Sequence[int], j: int) -> torch.Tensor:
    """[w] int32: every user's position for token offset j (the per-j cur_pos of the paged attention ops)."""
    return torch.tensor([int(p) + j for p in positions], dtype=torch.int32)


def rope_cos_sin(positions: torch.Tensor, rope_head_dim: int, theta: float):
    """Per-row cos/sin [1,1,n,rope_head_dim] bf16 for absolute positions [n] -- the same arithmetic as
    model.prepare_decode_inputs_host / rope_tp.rot_mats_decode (row b of the tile block is position b's rotation)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_head_dim, 2).float() / rope_head_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    n = positions.shape[0]
    return (
        emb.cos().reshape(1, 1, n, rope_head_dim).to(torch.bfloat16),
        emb.sin().reshape(1, 1, n, rope_head_dim).to(torch.bfloat16),
    )


def select_matrices(users: int, T: int, R: int):
    """The 0/1 gather / scatter constants of the row grid, as float32 (uploaded bf16; 0.0/1.0 are exact).

    sel[j]  [1,1,users,R]: sel[j][s, s*T+j] = 1 -> ``sel[j] @ X`` gathers every user's row j (rows in user order).
    selT[j] [1,1,R,users]: its transpose -> ``selT[j] @ Y_j`` scatters per-user rows back to rows s*T+j.
    A 0/1 matmul with fp32 accumulation reproduces the selected bf16 row bit-exactly (one 1.0*x plus zeros).
    """
    sel = torch.zeros(T, 1, 1, users, R, dtype=torch.float32)
    for s in range(users):
        for j in range(T):
            sel[j, 0, 0, s, row(s, j, T)] = 1.0
    selT = sel.transpose(-1, -2).contiguous()
    return [sel[j] for j in range(T)], [selT[j] for j in range(T)]


def accept_onehot_masks(accept_prev: Sequence[int], users: int, T: int, bmax: int) -> torch.Tensor:
    """[bmax, T, 1, 1] float32 one-hot of the previous step's accept count per GDN state slot.

    mask[s, j] = 1.0 iff a_s == j for the grid's users s < users (row j of the commit chain is the state the
    slot keeps); slots outside the grid (s >= users) keep their state untouched: mask[s, 0] = 1. Consumed by the
    GDN stub's masked select of the lazy commit chain (gdn/tp.py TPGatedDeltaNet._verify_stub).
    """
    m = torch.zeros(bmax, T, 1, 1, dtype=torch.float32)
    for s in range(bmax):
        if s < users:
            a = int(accept_prev[s])
            assert 0 <= a < T, f"user {s}: accept {a} outside [0, {T - 1}]"
            m[s, a, 0, 0] = 1.0
        else:
            m[s, 0, 0, 0] = 1.0
    return m


def commit(argmax_rows: torch.Tensor, tokens: Sequence[Sequence[int]], T: int):
    """The accept/commit of one step. argmax_rows: [R] ints (row order). tokens[s] = [x_0, d_1..d_k].
    Returns (accepts [w], committed [w] lists) with committed[s] = argmax[s, 0..a_s]."""
    accepts, committed = [], []
    for s, toks in enumerate(tokens):
        am = [int(argmax_rows[row(s, j, T)]) for j in range(T)]
        a = accept_drafts(am, list(toks[1:]))
        accepts.append(a)
        committed.append(am[: a + 1])
    return accepts, committed


@dataclass
class VerifyController:
    """Host bookkeeping of a batch of users driven through verify steps.

    ``run(tokens [w][T], positions [w], accept_prev [w]) -> argmax_rows [R]`` is the device step (or a CPU oracle).
    ``positions[s]`` is P_s, ``last[s]`` the row-0 token t'_s, ``committed[s]`` the user's token stream so far
    (seeded with the first token t_0 the prefill produced at position P_s).
    """

    T: int
    run: Callable[[Sequence[Sequence[int]], Sequence[int], Sequence[int]], torch.Tensor]
    positions: list
    last: list
    committed: list = field(default_factory=list)
    accept_prev: list = field(default_factory=list)
    accept_history: list = field(default_factory=list)

    def __post_init__(self):
        w = len(self.positions)
        assert len(self.last) == w
        self.positions = [int(p) for p in self.positions]
        self.last = [int(t) for t in self.last]
        if not self.committed:
            self.committed = [[t] for t in self.last]
        if not self.accept_prev:
            self.accept_prev = [0] * w

    @property
    def users(self) -> int:
        return len(self.positions)

    @property
    def k(self) -> int:
        return self.T - 1

    def step(self, drafts: Sequence[Sequence[int]]):
        """One verify step with drafts[s] = [d_1..d_k]; returns the per-user accept counts."""
        w, T = self.users, self.T
        assert len(drafts) == w and all(len(d) == T - 1 for d in drafts), "one k-draft list per user"
        tokens = [[self.last[s]] + [int(d) for d in drafts[s]] for s in range(w)]
        argmax_rows = self.run(tokens, list(self.positions), list(self.accept_prev))
        accepts, committed = commit(argmax_rows, tokens, T)
        for s in range(w):
            self.committed[s].extend(committed[s])
            self.positions[s] += accepts[s] + 1
            self.last[s] = committed[s][-1]
        self.accept_prev = list(accepts)
        self.accept_history.append(list(accepts))
        return accepts


class CpuGreedyOracle:
    """A CPU stand-in for the device step: ``next_fn(prefix_tokens) -> next token id`` is a deterministic
    'model'; row j's argmax is next_fn(prefix_s + [x_0, d_1..d_j]).

    It keeps the SAME lazy-prefix state the GDN kernel keeps: at the start of a call its prefix for user s holds
    every token at a position < P_s, brought up to date from the PREVIOUS call's rows 0..accept_prev[s] (row 0 =
    that step's x_0, rows 1..a its accepted drafts), exactly as the kernel commits prev rows 1..a_s + cur row 0.
    A VerifyController driven by it must reproduce the plain greedy stream for ANY drafts (pinned by
    tests/test_verify_grid.py)."""

    def __init__(self, next_fn, prompts: Sequence[Sequence[int]], T: int):
        self.next_fn = next_fn
        self.prefixes = [list(p) for p in prompts]  # tokens at positions < P_s (excludes the row-0 token)
        self.T = T
        self.prev_tokens = None
        self.calls = 0

    def __call__(self, tokens, positions, accept_prev):
        w, T = len(tokens), self.T
        R = grid_rows(w, T)
        if self.prev_tokens is not None:
            for s in range(w):
                self.prefixes[s].extend(int(t) for t in self.prev_tokens[s][: int(accept_prev[s]) + 1])
        out = torch.zeros(R, dtype=torch.int64)
        self.calls += 1
        for s in range(w):
            assert len(self.prefixes[s]) == positions[s], (s, len(self.prefixes[s]), positions[s])
            for j in range(T):
                out[row(s, j, T)] = int(self.next_fn(self.prefixes[s] + [int(t) for t in tokens[s][: j + 1]]))
        self.prev_tokens = [list(t) for t in tokens]
        return out
