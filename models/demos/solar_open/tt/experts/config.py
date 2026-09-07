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

    # Sparse matmul subblock widths
    decode_gate_up_subblock_w: int = 1
    decode_down_subblock_w: int = 1
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

    def __post_init__(self):
        """Validate configuration on creation"""
        self._validate_cores("decode_gate_up_cores", self.decode_gate_up_cores)
        self._validate_cores("decode_down_cores", self.decode_down_cores)
        if self.decode_down_cores_batched is not None:
            self._validate_cores("decode_down_cores_batched", self.decode_down_cores_batched)
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
                pcn = (Nt + num_cores - 1) // num_cores
                if (Nt + pcn - 1) // pcn != num_cores:
                    continue
                if best is None or num_cores > best[0] or (num_cores == best[0] and w > best[1]):
                    best = (num_cores, w, h, pcn)
        _, core_x, core_y, per_core_N = best  # num_cores == 1 always qualifies, so best is never None
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
        cores = self.decode_down_cores
        if self.decode_down_cores_batched is not None and m >= self.decode_down_batched_min_tokens:
            cores = self.decode_down_cores_batched
        return self._build_matmul_config(
            cores,
            m,
            n,
            in0_block_w=self.decode_down_in0_block_w,
            out_subblock_w=self.decode_down_subblock_w,
            k=k,
        )

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
