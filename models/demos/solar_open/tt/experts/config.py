# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Generic expert configuration interfaces.
No model-specific code - models provide their own ProgramConfig implementations.
"""

import math
from dataclasses import dataclass

import ttnn


@dataclass
class ExpertConfig:
    """Core expert configuration - model agnostic.

    The experts compute ``down(act(gate(x)) * up(x))`` per expert without biases. ``activation`` names
    the gating non-linearity applied to the gate projection (Solar-Open: ``"silu"``, see
    operations.apply_glu); ``intermediate_size`` is the routed experts' intermediate width
    (``moe_intermediate_size`` in the HF config, NOT the dense ``intermediate_size``).
    """

    intermediate_size: int
    num_experts: int
    hidden_size: int
    num_experts_per_tok: int
    activation: str = "silu"


@dataclass(frozen=True)
class IndexedRouting:
    """Top-k routing of ONE token in the layout the sparse_matmul indexed/gather mode consumes (decode.py
    ``_decode_forward_indexed``; produced by the model's router, e.g. ``TopKRouter.route_indexed``).

    Attributes:
        indices: ``[1, 1, 1, k]`` **uint16 ROW_MAJOR** device tensor of the selected expert ids (unsorted, distinct,
            all < num_experts; a single row-major stick as ``ttnn.sparse_matmul(indices=...)`` requires).
        weights: ``[1, 1, 1, k]`` **bf16 TILE** device tensor of the routing weights in the same order as ``indices``
            (normalised by the router; the experts multiply them into the k compact GLU rows before the down projection
            and sum the k down outputs).
        top_k: k (static per model; the compact shapes ``[1, k, 1, *]`` make the path trace-safe).
    """

    indices: ttnn.Tensor
    weights: ttnn.Tensor
    top_k: int

    def __post_init__(self):
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        shape = (1, 1, 1, self.top_k)
        if tuple(self.indices.shape) != shape or tuple(self.weights.shape) != shape:
            raise ValueError(
                f"IndexedRouting tensors must be {shape}: indices {tuple(self.indices.shape)}, "
                f"weights {tuple(self.weights.shape)}"
            )
        if self.indices.dtype != ttnn.uint16 or self.indices.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(
                f"IndexedRouting.indices must be uint16 ROW_MAJOR, got {self.indices.dtype} {self.indices.layout}"
            )
        if self.weights.dtype != ttnn.bfloat16 or self.weights.layout != ttnn.TILE_LAYOUT:
            raise ValueError(
                f"IndexedRouting.weights must be bfloat16 TILE, got {self.weights.dtype} {self.weights.layout}"
            )

    def deallocate(self):
        self.indices.deallocate(True)
        self.weights.deallocate(True)


@dataclass(frozen=True)
class MinimalMatmulBlocking:
    """Blocking of one ``ttnn.experimental.minimal_matmul`` ``[M, K] x [K, N]`` (the newer 2D matmul kernel: in0 reuse
    across N blocks, deferred writes, one weight with no batch dim): the core grid, the K block and the output subblock,
    all in tiles. ``config(m, k, n)`` derives the per-core M / N blocks the way the kernel partitions its output -- M
    over the grid's x axis and N over y when M > N (the kernel transposes its grid for tall outputs), else M over y and
    N over x; each axis is padded up to a multiple of its core count and every core owns one ``[M_block x N_block]``
    block -- snaps the K block to a divisor of Kt (a non-dividing K block would make the kernel read padded K tiles)
    and the subblock to divisors of the blocks within the destination-register budget (``dst_tiles``: 8 with a bf16
    destination, 4 with fp32 accumulation; the kernel requires ``subblock_h | M_block``, ``subblock_w | N_block`` and
    ``subblock_h * subblock_w <= dst``). The kernel needs a grid of at least 2x2 that fits the device: ``fits``.

    Solar-Open TP=8 (phase 3, tests/perf/test_prefill_matmul_candidates.py on P150): the per-expert fused gate|up
    ``[1024, 4096] x [4096, 320]`` with ``cores (11, 5), k_block 16, subblock (3, 2)`` -> M_block 3 (33 padded rows
    over 11 columns), N_block 2, 46 us vs 81 us for ``ttnn.linear``'s auto config; the hot group's K-concatenated down
    ``[1024, n_hot * 160] x [n_hot * 160, 4096]`` with ``cores (11, 10), k_block 5, subblock (4, 2)`` -> M_block 4,
    N_block 12, 50 / 74 / 117 us at n_hot 4 / 8 / 15.
    """

    cores: tuple[int, int]
    k_block: int
    subblock: tuple[int, int] = (1, 1)
    dst_tiles: int = 8

    def __post_init__(self):
        if not (isinstance(self.cores, tuple) and len(self.cores) == 2 and min(self.cores) >= 2):
            raise ValueError(f"MinimalMatmulBlocking.cores must be an (x, y) grid of at least 2x2, got {self.cores}")
        if self.k_block < 1:
            raise ValueError(f"MinimalMatmulBlocking.k_block must be positive, got {self.k_block}")
        if not (isinstance(self.subblock, tuple) and len(self.subblock) == 2 and min(self.subblock) >= 1):
            raise ValueError(f"MinimalMatmulBlocking.subblock must be a positive (h, w) pair, got {self.subblock}")
        if self.dst_tiles < 1:
            raise ValueError(f"MinimalMatmulBlocking.dst_tiles must be positive, got {self.dst_tiles}")

    def fits(self, grid) -> bool:
        """True when ``cores`` fits the compute grid ``grid`` (an object with ``x`` / ``y`` or an (x, y) pair)."""
        if grid is None:
            return False
        gx, gy = (grid.x, grid.y) if hasattr(grid, "x") else tuple(grid)
        return self.cores[0] <= gx and self.cores[1] <= gy

    def blocks(self, m: int, k: int, n: int) -> tuple[int, int, int, int, int]:
        """``(M_block, K_block, N_block, subblock_h, subblock_w)`` in tiles for an ``[m, k] x [k, n]`` product."""
        Mt, Kt, Nt = -(-m // 32), -(-k // 32), -(-n // 32)
        core_x, core_y = self.cores
        m_cores, n_cores = (core_x, core_y) if m > n else (core_y, core_x)
        m_block = -(-Mt // m_cores)
        n_block = -(-Nt // n_cores)
        k_block = max(d for d in range(1, min(self.k_block, Kt) + 1) if Kt % d == 0)
        sub_h = max(d for d in range(1, min(self.subblock[0], m_block) + 1) if m_block % d == 0)
        sub_w = max(d for d in range(1, min(self.subblock[1], n_block) + 1) if n_block % d == 0)
        while sub_h * sub_w > self.dst_tiles:  # over the dst budget: narrow the subblock first, then shorten it
            if sub_w > 1:
                sub_w = max(d for d in range(1, sub_w) if n_block % d == 0)
            else:
                sub_h = max(d for d in range(1, sub_h) if m_block % d == 0)
        return m_block, k_block, n_block, sub_h, sub_w

    def config(self, m: int, k: int, n: int) -> ttnn.MinimalMatmulConfig:
        m_block, k_block, n_block, sub_h, sub_w = self.blocks(m, k, n)
        return ttnn.MinimalMatmulConfig(
            M_block_size=m_block,
            K_block_size=k_block,
            N_block_size=n_block,
            subblock_h=sub_h,
            subblock_w=sub_w,
            compute_with_storage_grid_size=ttnn.CoreCoord(*self.cores),
        )


@dataclass
class ProgramConfig:
    """
    Base configuration for expert program configs.

    Models just need to specify grid sizes and chunking parameters.
    The boilerplate MatmulProgramConfig generation is handled automatically.

    Example:
        # Solar-Open-100B at TP=8 (fused gate|up N = 10 tiles, down N = 128 tiles)
        config = ProgramConfig(
            decode_gate_up_cores=(5, 2),
            decode_down_cores=(8, 4),
        )

        # A model with different shapes / chunking
        config = ProgramConfig(
            decode_gate_up_cores=(4, 6),
            decode_down_cores=(6, 8),
            sequence_chunk_size=2048,
        )
    """

    # Core grid sizes for decode
    decode_gate_up_cores: tuple[int, int] = (3, 4)
    decode_down_cores: tuple[int, int] = (5, 6)
    # Optional grid for the down projection when a decode step carries many users. With many active
    # experts the per-expert compute dominates and a wider grid pays off, while with few users the
    # 128-slot sparsity scan dominates and fewer multicast receivers are cheaper (measured on P150x8:
    # 1 user 120 vs 174 us, 32 users 385 vs 289 us per layer for 5x6 vs 9x10). The crossover has not
    # been measured; the threshold below is conservative.
    decode_down_cores_batched: tuple[int, int] | None = None
    decode_down_batched_min_tokens: int = 16

    # Core grid sizes for prefill
    prefill_gate_up_cores: tuple[int, int] = (3, 4)
    prefill_down_cores: tuple[int, int] = (5, 6)

    # Sparse matmul subblock widths (out_subblock_w == out_block_w; 1 = one output tile per compute pass). The
    # batched down grid has its own value: with per_core_N 2 (8x8) the widest legal subblock is 2, with per_core_N 4
    # (8x4) it is 4.
    decode_gate_up_subblock_w: int = 1
    decode_down_subblock_w: int = 1
    decode_down_batched_subblock_w: int = 1
    prefill_gate_up_subblock_w: int = 1
    prefill_down_subblock_w: int = 1

    # Input block widths (in0_block_w)
    decode_gate_up_in0_block_w: int = 1
    decode_down_in0_block_w: int = 1
    prefill_gate_up_in0_block_w: int = 1
    prefill_down_in0_block_w: int = 1

    # Chunking parameters
    sequence_chunk_size: int = 4 * 1024
    base_down_split_size: int = 1024

    # Dense (EP=1) prefill path: the batched per-expert matmuls run on the full compute grid capped at this
    # width (one expert per core per round), and the one-launch batched gate/up matmul is used for splits
    # up to dense_bmm_max_tokens tokens (per_core_M <= max_tokens/32 keeps the per-core output block in L1);
    # longer splits take the expert-sorted hot/cold path or the per-expert loop.
    dense_grid_max_width: int = 12
    dense_bmm_max_tokens: int = 256
    # Dense prefill down projection [1, E, S, Ip] x [1, E, Ip, H] as a 1D-multicast matmul on this grid (one K block
    # of Kt tiles, per_core_N = Nt / cores) instead of ttnn's ``core_grid=`` auto choice (which picks in0_block_w 1
    # for Kt = 5 and runs 2.7x slower at S = 128). None keeps the auto config. See get_dense_down_config.
    dense_down_cores: tuple[int, int] | None = None
    # get_dense_down_config: the tallest out_subblock_h <= this value that divides per_core_M with out_subblock_h *
    # out_subblock_w <= 8 destination tiles (bf16 destination, the dense compute config). The subblock only sets how
    # many output tiles one compute pass accumulates (bit-identical output); 1 = one tile row per pass (the phase-2
    # config), 4 = the whole [Mt x 2] block at <= 128 rows (measured on P150: 650 -> 596 us at 128 rows, 524 -> 484 at
    # 96, -8 %).
    dense_down_max_subblock_h: int = 1
    # Dense-path per-expert fused gate|up linear [1, 1, split, H] x [1, 1, H, 2Ip] (the sorted path's hot group and
    # the per-expert loop of long splits) as ttnn.experimental.minimal_matmul with this blocking; None keeps
    # ttnn.linear with ttnn's ``core_grid=`` auto config (a 2D multicast config with in0_block_w 4). See
    # get_dense_expert_gate_up_config.
    dense_expert_gate_up_minimal: MinimalMatmulBlocking | None = None
    # Hot group's down projection + sum over the hot experts as ONE K-concatenated minimal_matmul
    # [1, 1, split, n_hot * Ip] x [1, 1, n_hot * Ip, H] over bf16 GLU pieces (the sum moves into the matmul's
    # accumulation); None keeps the batched per-expert matmul (auto config) + fast_reduce_nc. See
    # get_hot_down_kconcat_config.
    hot_down_kconcat_minimal: MinimalMatmulBlocking | None = None
    # Dense bmm path (splits <= dense_bmm_max_tokens, i.e. the traced 128-token prefill): round the activation to
    # bfloat8_b BEFORE its per-expert replication (ttnn.repeat), so the broadcast writes half the bytes and the gate|up
    # bmm reads bfp8 in0 (measured on P150 at 128 tokens: 633 -> 366 us + bmm 808 -> 771 us per layer). A numerics
    # change of every routed expert's input on that path (the sorted / per-expert paths keep bf16 activations): off
    # unless the teacher-forced accuracy test admits it.
    dense_activation_bfp8: bool = False
    # Split lengths (tokens) whose MoE must be TRACE-SAFE (no device->host read, static shapes, no persistent
    # allocation). A split longer than dense_bmm_max_tokens normally takes the host-planned expert-sorted hot/cold path
    # (prefill._sorted_moe_plan reads the per-expert token counts back to the host per split): a trace capture would
    # record ONE prompt's plan and replay it for every other prompt. Splits listed here take the static per-expert
    # loop instead (device-only; ~2.5x the sorted path's cost at 1024 tokens -- the placeholder until a trace-safe
    # sorted plan exists, design_traced_prefill.md (C)). Empty by default: the only shipped traced prefill bucket
    # (128 tokens) runs the one-launch dense bmm, trace-safe by construction. Consulted by prefill.moe_prefill_path;
    # ModelArgs.get_trace_prefill_supported_seq_lens refuses a traced length whose splits are not trace-safe.
    trace_safe_split_lens: tuple[int, ...] = ()

    def __post_init__(self):
        """Validate configuration on creation"""
        self._validate_cores("decode_gate_up_cores", self.decode_gate_up_cores)
        self._validate_cores("decode_down_cores", self.decode_down_cores)
        if self.decode_down_cores_batched is not None:
            self._validate_cores("decode_down_cores_batched", self.decode_down_cores_batched)
        if self.dense_down_cores is not None:
            self._validate_cores("dense_down_cores", self.dense_down_cores)
        self._validate_cores("prefill_gate_up_cores", self.prefill_gate_up_cores)
        self._validate_cores("prefill_down_cores", self.prefill_down_cores)

        if self.sequence_chunk_size <= 0:
            raise ValueError(f"sequence_chunk_size must be positive, got {self.sequence_chunk_size}")
        if self.sequence_chunk_size % 32 != 0:
            raise ValueError(f"sequence_chunk_size must be multiple of 32, got {self.sequence_chunk_size}")

        if self.base_down_split_size <= 0:
            raise ValueError(f"down_split_size must be positive, got {self.base_down_split_size}")
        if self.base_down_split_size % 32 != 0:
            raise ValueError(f"down_split_size must be multiple of 32, got {self.base_down_split_size}")

        if self.dense_grid_max_width <= 0:
            raise ValueError(f"dense_grid_max_width must be positive, got {self.dense_grid_max_width}")
        if self.dense_bmm_max_tokens <= 0 or self.dense_bmm_max_tokens % 32 != 0:
            raise ValueError(f"dense_bmm_max_tokens must be a positive multiple of 32, got {self.dense_bmm_max_tokens}")
        if not 1 <= self.dense_down_max_subblock_h <= 8:
            raise ValueError(f"dense_down_max_subblock_h must be in 1..8, got {self.dense_down_max_subblock_h}")
        for name in ("dense_expert_gate_up_minimal", "hot_down_kconcat_minimal"):
            blocking = getattr(self, name)
            if blocking is not None and not isinstance(blocking, MinimalMatmulBlocking):
                raise ValueError(f"{name} must be a MinimalMatmulBlocking or None, got {blocking!r}")
        if not isinstance(self.trace_safe_split_lens, (tuple, list)) or any(
            not isinstance(n, int) or n <= 0 or n % 32 != 0 for n in self.trace_safe_split_lens
        ):
            raise ValueError(
                f"trace_safe_split_lens must be a tuple of positive multiples of 32, got {self.trace_safe_split_lens!r}"
            )
        self.trace_safe_split_lens = tuple(sorted(set(self.trace_safe_split_lens)))

    def _validate_cores(self, name: str, cores: tuple[int, int]):
        """Validate core grid dimensions"""
        if not isinstance(cores, tuple) or len(cores) != 2:
            raise ValueError(f"{name} must be a tuple of (x, y), got {cores}")

        core_x, core_y = cores
        if core_x <= 0 or core_y <= 0:
            raise ValueError(f"{name} must have positive dimensions, got {cores}")

    def get_down_split_size(self, seqlen: int) -> int:
        if seqlen <= 32 * 1024:
            return self.base_down_split_size
        else:
            # For very long sequences, decrease split size to avoid OOM
            return self.base_down_split_size // 2

    def _build_matmul_config(
        self,
        cores: tuple[int, int],
        m: int,
        n: int,
        in0_block_w: int = 1,
        out_subblock_w: int = 1,
        k: int = None,
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """
        Build MatmulProgramConfig with standard settings.

        This is the single source of truth for matmul config generation.
        All get_*_config methods use this builder.

        Args:
            cores: (core_x, core_y) grid dimensions
            m: Input height dimension
            n: Output width dimension
            in0_block_w: Block width for input tensor
            out_subblock_w: Output subblock width (for sparse matmuls)
            k: Input contraction dimension (used to snap in0_block_w to a
                divisor of Kt; passing None preserves the configured value).

        Returns:
            MatmulMultiCoreReuseMultiCast1DProgramConfig
        """
        core_x, core_y = cores
        Nt = int(math.ceil(n / 32))
        # The mcast_in0 sparse matmul hands out ceil(Nt / per_core_N) output blocks to the first
        # cores of the grid in row-major order and multicasts in0 to the bounding box of those
        # cores; the factory requires the two sets to be identical (a partially filled last row
        # would leave receivers without work and hang). Pick the largest sub-rectangle (w <= core_x,
        # h <= core_y) whose block count fills it exactly; ties prefer the wider shape (closest to
        # the requested grid). For the Solar-Open TP=8 shapes (fused gate|up Nt=10 on 5x2, down
        # Nt=128 on 8x4 / 8x8) this is the identity; for other TP factors it shrinks the grid
        # instead of tripping the factory's rectangularity check.
        best = None
        for w in range(core_x, 0, -1):
            for h in range(core_y, 0, -1):
                num_cores = w * h
                if num_cores == 1:
                    # A single-core mcast_in0 sparse_matmul has no multicast receivers and deadlocks the device
                    # (measured 2026-09-07 on P150: 1x1 grid, per_core_N 10 -> hang, board reset needed).
                    continue
                pcn = (Nt + num_cores - 1) // num_cores
                if (Nt + pcn - 1) // pcn != num_cores:
                    continue
                if best is None or num_cores > best[0] or (num_cores == best[0] and w > best[1]):
                    best = (num_cores, w, h, pcn)
        if best is None:
            raise ValueError(
                f"sparse matmul with N = {n} ({Nt} tiles) on a {core_x}x{core_y} grid: no multi-core rectangle is an "
                "exact fill and a single-core mcast_in0 grid hangs the device; use a grid with >= 2 cores that "
                f"divides ceil({Nt} / per_core_N)"
            )
        _, core_x, core_y, per_core_N = best
        # The sparse matmul kernel asserts `Kt % in0_block_w == 0`. Different
        # tp factors produce different Kt (e.g. down's K = intermediate/tp:
        # tp=8 → Kt=5, tp=1 → Kt=40), and the configured in0_block_w may not
        # divide them all. Snap to the largest divisor of Kt that does not
        # exceed the configured ceiling; when Kt is already divisible by the
        # configured value this is a no-op (so existing tunings are
        # preserved). When Kt is prime (or coprime with every value ≤
        # configured) the only divisor under the ceiling is 1, which collapses
        # the matmul to a tile-by-tile inner loop and roughly halves prefill
        # throughput. In that case fall back to Kt itself — always a divisor,
        # and small enough to fit in L1 for the Kt range produced by realistic
        # TP shardings of these models (Kt <= ~128).
        if k is not None:
            Kt = int(math.ceil(k / 32))
            if Kt % in0_block_w != 0:
                divisors = [d for d in range(2, in0_block_w + 1) if Kt % d == 0]
                in0_block_w = max(divisors) if divisors else Kt

        # Derive out_block_w so the subblock axis is expressible instead of
        # hardcoding it to 1. The mcast_in0 compute kernel bakes
        # in1_num_subblocks = out_block_w // out_subblock_w in as a compile-time
        # arg; if out_subblock_w does not divide out_block_w that integer
        # division yields 0, the in1_subblock loop runs zero times, cb_out is
        # never pushed, and the in1 writer parks forever on cb_out.wait_front()
        # -- a silent device hang. (The same expression also underflows uint32
        # in the last-block padded skip count.) The host guard added in
        # sparse_matmul_device_operation.cpp now TT_FATALs on that illegal combo
        # instead of hanging; here we keep the model from generating one.
        #
        # dst-register file limit: out_subblock_h * out_subblock_w <= 8.
        # out_subblock_h is 1 for decode (M=1 tile), so cap out_subblock_w at 8.
        out_subblock_w = min(out_subblock_w, 8)
        # out_subblock_w must divide per_core_N so that out_block_w (a multiple
        # of it) can also divide per_core_N. If it does not, snap it DOWN to the
        # largest divisor of per_core_N that does not exceed the requested value
        # -- mirroring the in0_block_w snap-to-divisor idiom above (1 always
        # divides, so a legal value always exists).
        if per_core_N % out_subblock_w != 0:
            divisors = [d for d in range(1, out_subblock_w) if per_core_N % d == 0]
            out_subblock_w = max(divisors)
        # out_block_w is the smallest legal block that holds one full subblock,
        # i.e. out_block_w == out_subblock_w. This guarantees both
        # out_block_w % out_subblock_w == 0 and per_core_N % out_block_w == 0, so
        # in1_num_subblocks is never 0. With the shipped default out_subblock_w=1
        # this yields out_block_w=1 -- byte-identical to the previous hardcode.
        out_block_w = out_subblock_w

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            out_block_h=1,
            out_block_w=out_block_w,
            per_core_M=max(32, m) // 32,
            per_core_N=per_core_N,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def get_decode_gate_up_config(
        self, m: int, n: int, k: int = None
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for decode gate/up projections"""
        return self._build_matmul_config(
            self.decode_gate_up_cores,
            m,
            n,
            in0_block_w=self.decode_gate_up_in0_block_w,
            out_subblock_w=self.decode_gate_up_subblock_w,
            k=k,
        )

    def get_decode_down_config(
        self, m: int, n: int, k: int = None
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for decode down projection (m = tokens in the step)"""
        cores, subblock_w = self.decode_down_cores, self.decode_down_subblock_w
        if self.decode_down_cores_batched is not None and m >= self.decode_down_batched_min_tokens:
            cores, subblock_w = self.decode_down_cores_batched, self.decode_down_batched_subblock_w
        return self._build_matmul_config(
            cores,
            m,
            n,
            in0_block_w=self.decode_down_in0_block_w,
            out_subblock_w=subblock_w,
            k=k,
        )

    def get_dense_down_config(self, m: int, n: int, k: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
        """Program config for the DENSE prefill down projection ``[1, E, m, k] x [1, E, k, n]`` (all experts, one
        batched matmul), or None when ``dense_down_cores`` is unset (caller uses ``core_grid=`` auto).

        1D multicast of in0 over ``dense_down_cores``, the whole K (``Kt`` tiles) as one block, ``per_core_N = Nt /
        cores`` and one ``[per_core_M x per_core_N]`` output block per core (``out_subblock_w`` = the widest divisor
        of per_core_N <= 4; ``out_subblock_h`` = the tallest divisor of per_core_M <= ``dense_down_max_subblock_h``
        with out_subblock_h * out_subblock_w <= 8 destination tiles). Measured on P150 for Solar-Open (Kt = 5, Nt =
        128, 8x8 cores): 311 / 642 / 1096 us at m = 32 / 128 / 256 vs 396 / 1148 / 1716 us for the auto config
        (in0_block_w 1) with out_subblock_h 1; the [Mt x 2] subblock (dense_down_max_subblock_h 4) takes 128 rows to
        596 us and 96 rows 524 -> 484 us, bit-identical (phase 3). Requires ``Nt % cores == 0`` and ``m % 32 == 0``;
        raises otherwise (no silent fallback)."""
        if self.dense_down_cores is None:
            return None
        core_x, core_y = self.dense_down_cores
        num_cores = core_x * core_y
        Mt, Kt, Nt = m // 32, int(math.ceil(k / 32)), int(math.ceil(n / 32))
        if m % 32 != 0 or Nt % num_cores != 0:
            raise ValueError(
                f"dense down config: m={m} must be a tile multiple and Nt={Nt} must be divisible by the "
                f"{core_x}x{core_y} = {num_cores} cores of dense_down_cores"
            )
        per_core_N = Nt // num_cores
        out_subblock_w = max(d for d in (4, 2, 1) if per_core_N % d == 0)
        # dst-register budget of the bf16-destination dense compute config: 8 tiles per compute pass
        out_subblock_h = max(
            h for h in range(1, min(Mt, self.dense_down_max_subblock_h) + 1) if Mt % h == 0 and h * out_subblock_w <= 8
        )
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=Kt,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            out_block_h=Mt,
            out_block_w=per_core_N,
            per_core_M=Mt,
            per_core_N=per_core_N,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def get_dense_expert_gate_up_config(self, m: int, k: int, n: int, grid=None) -> ttnn.MinimalMatmulConfig | None:
        """``ttnn.MinimalMatmulConfig`` of the dense-path per-expert fused gate|up linear ``[1, 1, m, k] x [1, 1, k,
        n]`` (sorted hot group, per-expert loop) from ``dense_expert_gate_up_minimal``, or None (caller runs
        ``ttnn.linear`` with the auto config) when the blocking is unset or does not fit ``grid`` (the compute grid the
        caller may use; None = no grid check)."""
        blocking = self.dense_expert_gate_up_minimal
        if blocking is None or (grid is not None and not blocking.fits(grid)):
            return None
        return blocking.config(m, k, n)

    def get_hot_down_kconcat_config(self, m: int, k: int, n: int, grid=None) -> ttnn.MinimalMatmulConfig | None:
        """``ttnn.MinimalMatmulConfig`` of the hot group's K-concatenated down ``[1, 1, m, k = n_hot * Ip] x [1, 1, k,
        n = H]`` from ``hot_down_kconcat_minimal``, or None (caller keeps the batched matmul + fast_reduce_nc) when the
        blocking is unset or does not fit ``grid``."""
        blocking = self.hot_down_kconcat_minimal
        if blocking is None or (grid is not None and not blocking.fits(grid)):
            return None
        return blocking.config(m, k, n)

    def get_prefill_gate_up_config(
        self, m: int, n: int, k: int = None
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for prefill gate/up projections"""
        return self._build_matmul_config(
            self.prefill_gate_up_cores,
            m,
            n,
            in0_block_w=self.prefill_gate_up_in0_block_w,
            out_subblock_w=self.prefill_gate_up_subblock_w,
            k=k,
        )

    def get_prefill_down_config(
        self, m: int, n: int, k: int = None
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Get program config for prefill down projection"""
        return self._build_matmul_config(
            self.prefill_down_cores,
            m,
            n,
            in0_block_w=self.prefill_down_in0_block_w,
            out_subblock_w=self.prefill_down_subblock_w,
            k=k,
        )
