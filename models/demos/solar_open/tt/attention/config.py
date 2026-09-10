# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import ttnn


def _cores_row_major(core_range_set) -> list[tuple[int, int]]:
    """Row-major (x, y) cores of a ttnn.CoreRangeSet in the order its ranges are stored (the order
    ``corerange_to_cores(..., row_wise=True)`` and the sharded-buffer page mapping use)."""
    cores = []
    for core_range in core_range_set.ranges():
        start, end = core_range.start, core_range.end
        cores.extend((x, y) for y in range(start.y, end.y + 1) for x in range(start.x, end.x + 1))
    return cores


@dataclass
class AttentionConfig:
    """Core attention configuration - model agnostic"""

    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    max_local_batch_size: int

    users_row_sharded: bool = False
    sliding_window: int | None = None
    scaling: float | None = None  # Computed if None

    def __post_init__(self):
        """Compute scaling factor if not provided"""
        if self.scaling is None:
            self.scaling = self.head_dim**-0.5


@dataclass
class ProgramConfig:
    """
    Base configuration for SDPA program configs.

    Models just need to specify chunk sizes and compute settings.
    The boilerplate SDPA config generation is handled automatically.
    """

    # Decode SDPA config
    decode_q_chunk_size: int = 0
    decode_k_chunk_size: int = 128

    # Prefill SDPA config
    prefill_q_chunk_size_small: int = 32
    prefill_k_chunk_size_small: int = 32
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Compute config
    math_fidelity: str = "HiFi4"
    math_approx_mode: bool = False
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = False

    # Matmul program config parameters (optional - None means no program config)
    # Decode QKV projection
    decode_qkv_cores: tuple[int, int] | None = None
    decode_qkv_in0_block_w: int = 1
    decode_qkv_out_subblock_h: int = 1
    decode_qkv_out_subblock_w: int = 1
    decode_qkv_fp32_dest_acc: bool = False  # fp32 destination accumulation for the explicit qkv config

    # Decode output projection
    decode_out_cores: tuple[int, int] | None = None
    decode_out_in0_block_w: int = 1
    decode_out_out_subblock_h: int = 1
    decode_out_out_subblock_w: int = 1
    # With ``decode_out_cores`` set: run the o_proj from an L1-INTERLEAVED in0 (``sharded_to_interleaved`` of the
    # width-sharded ``nlp_concat_heads_decode`` output BEFORE the matmul, whose output is then already interleaved
    # for the reshape / all-reduce) instead of on the sharded in0 (phase-2 arm: ``in0_block_w`` pinned to the
    # 4-tile shard, no gain). Phase 3e lever B1 slice 2; ignored while ``decode_out_cores`` is None.
    decode_out_interleave_in0: bool = False
    decode_out_fp32_dest_acc: bool = False  # fp32 destination accumulation for the explicit o_proj config

    # Phase 3e lever B1 slice 1: fused decode attention chain. ``nlp_create_qkv_heads_decode(overlap_qk_coregrid=
    # False)`` puts user b's Q (and V) on core b and its K on core B + b of a 2B-core grid, ``rotary_embedding_llama_
    # fused_qk`` rotates Q and K in ONE launch (each core reads the cos/sin/trans_mat shard resident in its L1, so the
    # RotarySetup must be built with ``use_qk_fused=True``: 2B cos/sin rows, positions repeated for the K half) and
    # ``paged_fused_update_cache`` writes K and V in ONE launch (core i of the K grid pairs with core i of the V grid).
    # Default False = the phase-1 chain (create_heads on B cores, rope q, rope k, update k, update v).
    fused_qk: bool = False

    # Prefill QKV projection
    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_qkv_in0_block_w: int = 1
    prefill_qkv_out_subblock_h: int = 1
    prefill_qkv_out_subblock_w: int = 1

    # Prefill output projection
    prefill_out_cores: tuple[int, int] | None = None
    prefill_out_in0_block_w: int = 1
    prefill_out_out_subblock_h: int = 1
    prefill_out_out_subblock_w: int = 1

    def __post_init__(self):
        """Validate configuration on creation"""
        if self.decode_q_chunk_size < 0 or self.decode_k_chunk_size <= 0:
            raise ValueError("Decode chunk sizes must be non-negative (q) and positive (k)")

        if self.prefill_q_chunk_size_small <= 0 or self.prefill_k_chunk_size_small <= 0:
            raise ValueError("Prefill small chunk sizes must be positive")

        if self.prefill_q_chunk_size_large <= 0 or self.prefill_k_chunk_size_large <= 0:
            raise ValueError("Prefill large chunk sizes must be positive")

        if self.prefill_threshold <= 0:
            raise ValueError("Prefill threshold must be positive")

        # Validate math_fidelity
        valid_fidelities = ["LoFi", "HiFi2", "HiFi3", "HiFi4"]
        if self.math_fidelity not in valid_fidelities:
            raise ValueError(f"math_fidelity must be one of {valid_fidelities}, got {self.math_fidelity}")

    @staticmethod
    def get_decode_user_grid(mesh_device, batch_size: int, fused_qk: bool = False):
        """Per-user core placement for decode: (cores holding user b's Q/K/V shard, SDPA program grid).

        Three consumers read a user's data from a core they *compute* rather than from the shard spec,
        so they must all agree: rotary_embedding_llama borrows cos/sin/trans_mat from the shard resident
        in the local core's L1 (RotarySetup places them, see models/tt_transformers/tt/rope.py
        get_batch_grid); paged SDPA decode's reducer/output core for batch b is (b % grid.x, b // grid.x)
        of its program grid and reads Q from that core's L1; and nlp_create_qkv_heads_decode places user
        b on the b-th core (row-major) of the grid it is given. RotarySetup's rule is: an 8x8 grid when
        the batch is a multiple of 32, otherwise the device compute grid (8x8 on Wormhole, 13x10 / 11x10 on
        Blackhole), row-major. Mirror it here and pick the SDPA grid with the matching width. Up to 8
        users both layouts coincide (first row), so the 8x8 SDPA grid is kept for them.

        ``fused_qk`` (phase 3e, ``ProgramConfig.fused_qk``): the RotarySetup is built with ``use_qk_fused=True`` and
        its batch is 2B (Q rows then K rows), so RotarySetup's rule is evaluated on 2B: B = 16 moves from the device
        grid to 8x8 (2B = 32), B = 32 stays 8x8, B <= 8 stays in the first row. The returned ``user_cores`` are the Q
        cores (= the SDPA reducer cores); the K cores come from ``get_decode_qk_fused_grids``. NOTE the B = 16 SDPA
        grid therefore changes (device grid -> 8x8: fewer cores per user), which changes the SDPA reduction split at
        contexts beyond one k chunk; B = 1 / 32 (the production batches) keep their grids.
        """
        device_grid = mesh_device.compute_with_storage_grid_size()
        grid_8x8 = ttnn.CoreCoord(8, 8)
        # RotarySetup's rule (see docstring): 8x8 for multiples of 32; the first row of any grid coincides with the
        # 8x8 layout for <= 8 users; on an 8-wide device grid both layouts are the same anyway.
        rope_batch = 2 * batch_size if fused_qk else batch_size
        rope_uses_8x8 = rope_batch % 32 == 0
        fits_first_row = batch_size <= 8
        device_is_8_wide = device_grid.x == grid_8x8.x
        uses_8x8_grid = rope_uses_8x8 or fits_first_row or device_is_8_wide
        sdpa_grid = grid_8x8 if uses_8x8_grid else ttnn.CoreCoord(device_grid.x, device_grid.y)
        user_cores = ttnn.num_cores_to_corerangeset(batch_size, sdpa_grid, row_wise=True)
        return user_cores, sdpa_grid

    @staticmethod
    def get_decode_qk_fused_grids(mesh_device, batch_size: int):
        """Core grids of the fused decode chain (``fused_qk``): ``(heads_grid, q_cores, k_cores)``.

        ``heads_grid`` is the 2B-core grid handed to ``nlp_create_qkv_heads_decode(overlap_qk_coregrid=False)`` as its
        output shard grid; the op derives Q (and V) = the first B cores of that grid in row-major order and K = the B
        cores after them (its ``compute_output_specs``: K starts at the last core of the (B + 1)-core prefix). The
        grid is the first 2B cores, row-major, of the 8x8 grid when 2B is a multiple of 32 and of the device compute
        grid otherwise -- exactly ``RotarySetup(use_qk_fused=True).batch_grid`` (rope.py ``get_batch_grid`` on the
        doubled batch), so core i holds cos/sin/trans_mat row i: Q of user b on core b next to cos row b, K of user b
        on core B + b next to cos row B + b (= the same position, repeated by ``get_rot_idxs`` / ``get_tt_pos_idx``).
        ``paged_fused_update_cache`` pairs K core i with V core i in row-major order of each grid, so both write user
        i's row. Raises if the derived Q / K placement does not tile ``heads_grid`` or if the Q cores differ from
        ``get_decode_user_grid(..., fused_qk=True)`` (the SDPA reducer cores).
        """
        if not 1 <= batch_size <= 32:
            raise ValueError(
                f"fused_qk decode: batch size {batch_size} is not in 1..32 (rotary_embedding_llama_fused_qk "
                "parallelizes Q and K over at most 64 cores)"
            )
        device_grid = mesh_device.compute_with_storage_grid_size()
        doubled = 2 * batch_size
        base_grid = ttnn.CoreCoord(8, 8) if doubled % 32 == 0 else ttnn.CoreCoord(device_grid.x, device_grid.y)
        if doubled > base_grid.x * base_grid.y:
            raise ValueError(f"fused_qk decode: {doubled} cores needed, the {base_grid.x}x{base_grid.y} grid has fewer")
        heads_grid = ttnn.num_cores_to_corerangeset(doubled, base_grid, row_wise=True)
        q_cores = ttnn.num_cores_to_corerangeset(batch_size, base_grid, row_wise=True)
        k_start = ttnn.CoreCoord(batch_size % base_grid.x, batch_size // base_grid.x)
        k_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(k_start, batch_size, heads_grid, row_wise=True)
        heads_list = _cores_row_major(heads_grid)
        q_list, k_list = _cores_row_major(q_cores), _cores_row_major(k_cores)
        if q_list + k_list != heads_list:
            raise RuntimeError(
                f"fused_qk decode: Q cores {q_list[:3]}... + K cores {k_list[:3]}... do not tile the {doubled}-core "
                f"heads grid {heads_list[:3]}... for batch {batch_size}"
            )
        user_cores, _ = ProgramConfig.get_decode_user_grid(mesh_device, batch_size, fused_qk=True)
        if _cores_row_major(user_cores) != q_list:
            raise RuntimeError(
                f"fused_qk decode: the Q cores {q_list[:3]}... differ from the SDPA reducer cores "
                f"{_cores_row_major(user_cores)[:3]}... for batch {batch_size}"
            )
        return heads_grid, q_cores, k_cores

    @staticmethod
    def get_decode_concat_grid(batch_size: int):
        """Single-rectangle core grid with one core per user for nlp_concat_heads_decode's input.

        The op derives its output grid from the bounding box of the input grid, so the input must be
        exactly one CoreRange (16 users laid out row-major on a 13-wide grid, 13 + 3, is not). Pick the
        widest w <= 8 that divides the batch with at most 8 rows; every practical batch (powers of two,
        multiples of 8) gets a full rectangle, and a batch that fits in one row stays a row.
        """
        if batch_size <= 8:
            return ttnn.num_cores_to_corerangeset(batch_size, ttnn.CoreCoord(8, 8), row_wise=True)
        for width in range(8, 0, -1):
            if batch_size % width == 0 and batch_size // width <= 8:
                return ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(width - 1, batch_size // width - 1))}
                )
        raise ValueError(
            f"batch size {batch_size}: nlp_concat_heads_decode needs one core per user on a single rectangle of "
            "at most 8x8 cores and no w <= 8 with batch/w <= 8 divides this batch; pad the batch to a multiple "
            "of 8 or a power of two (<= 32)."
        )

    def get_decode_sdpa_config(self, mesh_device, batch_size: int = 1) -> ttnn.SDPAProgramConfig:
        """Get SDPA config for decode mode"""
        _, sdpa_grid = self.get_decode_user_grid(mesh_device, batch_size, fused_qk=self.fused_qk)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_grid,
            q_chunk_size=self.decode_q_chunk_size,
            k_chunk_size=self.decode_k_chunk_size,
            exp_approx_mode=False,
        )

    def get_prefill_sdpa_config(self, mesh_device, seq_len: int) -> ttnn.SDPAProgramConfig:
        """Get SDPA config for prefill mode based on sequence length"""
        if seq_len >= self.prefill_threshold:
            q_chunk = self.prefill_q_chunk_size_large
            k_chunk = self.prefill_k_chunk_size_large
        else:
            q_chunk = self.prefill_q_chunk_size_small
            k_chunk = self.prefill_k_chunk_size_small

        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
        )

    def get_compute_kernel_config(self) -> ttnn.WormholeComputeKernelConfig:
        """Get compute kernel config"""
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.math_fidelity),
            math_approx_mode=self.math_approx_mode,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=self.packer_l1_acc,
        )

    @staticmethod
    def _hifi2_compute_config(arch, fp32_dest_acc: bool):
        """The auto bf16 x bfp8 matmul fidelity restated for an explicit program config (HiFi2, no approx, L1 packer
        accumulation), optionally with fp32 destination accumulation."""
        return ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc,
            packer_l1_acc=True,
        )

    def get_decode_qkv_compute_config(self, arch):
        """Compute kernel config to pass WITH ``get_decode_qkv_config``'s program config.

        ttnn raises a bf16 x bfp8 matmul to HiFi2 (no approx, L1 packer accumulation) only for the AUTO config
        (``matmul_device_operation.cpp::create_matmul_attributes``: a program config or a core grid drops it to
        LoFi), so an explicit program config must restate those defaults to keep the auto numerics;
        ``decode_qkv_fp32_dest_acc`` adds fp32 destination accumulation on top."""
        return self._hifi2_compute_config(arch, self.decode_qkv_fp32_dest_acc)

    def get_decode_out_compute_config(self, arch):
        """Compute kernel config to pass WITH ``get_decode_out_config``'s program config (the same LoFi-fallback trap
        as the qkv config: an explicit o_proj program config without a compute config runs LoFi);
        ``decode_out_fp32_dest_acc`` adds fp32 destination accumulation."""
        return self._hifi2_compute_config(arch, self.decode_out_fp32_dest_acc)

    def _build_matmul_config(
        self,
        cores: tuple[int, int],
        m: int,
        n: int,
        k: int,
        in0_block_w: int,
        out_subblock_h: int,
        out_subblock_w: int,
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        """Build a 1D in0-multicast matmul program config for an attention projection ``[1, 1, m, k] x [k, n]``.

        The N tiles are split evenly over the ``cores`` grid (``n // 32`` must be divisible by the core count) and
        every core computes one ``[per_core_M x per_core_N]`` output block, so ``out_block_h/w`` equal the per-core
        block (the matmul validation requires ``per_core_M % out_block_h == 0``, ``per_core_N % out_block_w == 0``
        and the subblocks to divide the blocks; ``out_subblock_h * out_subblock_w <= 8`` dst registers). Requested
        subblock widths that do not divide the per-core block snap down to the largest divisor; ``in0_block_w``
        snaps to a divisor of ``Kt`` (a WIDTH-sharded in0 such as ``nlp_concat_heads_decode``'s output additionally
        needs ``in0_block_w`` to divide its shard width in tiles -- 4 for head_dim 128). ``fuse_batch=True`` so a
        sharded in0 is accepted (the inputs are ``[1, 1, m, k]``, so M is ``m // 32`` either way).
        """
        core_x, core_y = cores
        num_cores = core_x * core_y
        Mt, Kt, Nt = max(32, m) // 32, max(32, k) // 32, n // 32
        if n % 32 != 0 or Nt % num_cores != 0:
            raise ValueError(
                f"attention matmul config: N = {n} ({Nt} tiles) must be a tile multiple divisible by the "
                f"{core_x}x{core_y} = {num_cores} cores"
            )
        per_core_N = Nt // num_cores
        if Kt % in0_block_w != 0:
            in0_block_w = max(d for d in range(1, in0_block_w + 1) if Kt % d == 0)
        if per_core_N % out_subblock_w != 0:
            out_subblock_w = max(d for d in range(1, out_subblock_w + 1) if per_core_N % d == 0)
        if Mt % out_subblock_h != 0:
            out_subblock_h = max(d for d in range(1, out_subblock_h + 1) if Mt % d == 0)
        while out_subblock_h * out_subblock_w > 8:
            out_subblock_h = max(d for d in range(1, out_subblock_h) if Mt % d == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            out_block_h=Mt,
            out_block_w=per_core_N,
            per_core_M=Mt,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def get_decode_qkv_config(self, m: int, n: int, k: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
        """Get program config for decode QKV projection"""
        if self.decode_qkv_cores is None:
            return None
        return self._build_matmul_config(
            self.decode_qkv_cores,
            m,
            n,
            k,
            self.decode_qkv_in0_block_w,
            self.decode_qkv_out_subblock_h,
            self.decode_qkv_out_subblock_w,
        )

    def get_decode_out_config(self, m: int, n: int, k: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
        """Get program config for decode output projection"""
        if self.decode_out_cores is None:
            return None
        return self._build_matmul_config(
            self.decode_out_cores,
            m,
            n,
            k,
            self.decode_out_in0_block_w,
            self.decode_out_out_subblock_h,
            self.decode_out_out_subblock_w,
        )

    def get_prefill_qkv_config(
        self, m: int, n: int, k: int
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
        """Get program config for prefill QKV projection"""
        if self.prefill_qkv_cores is None:
            return None
        return self._build_matmul_config(
            self.prefill_qkv_cores,
            m,
            n,
            k,
            self.prefill_qkv_in0_block_w,
            self.prefill_qkv_out_subblock_h,
            self.prefill_qkv_out_subblock_w,
        )

    def get_prefill_out_config(
        self, m: int, n: int, k: int
    ) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
        """Get program config for prefill output projection"""
        if self.prefill_out_cores is None:
            return None
        return self._build_matmul_config(
            self.prefill_out_cores,
            m,
            n,
            k,
            self.prefill_out_in0_block_w,
            self.prefill_out_out_subblock_h,
            self.prefill_out_out_subblock_w,
        )
