# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Solar-Open attention program configurations."""

from dataclasses import dataclass

from models.demos.solar_open.tt.attention.config import ProgramConfig


@dataclass
class SolarOpenAttentionProgramConfig(ProgramConfig):
    """
    Solar-Open-100B attention configuration.

    Shapes: hidden=4096, heads=64, kv_heads=8, head_dim=128; at TP=8 every device holds 8 q heads + 1 kv head, so the
    fused wqkv is N=1280 (40 tiles) and o_proj is K=1024 (32 tiles) per device.

    Phase 1 (PCC bring-up): ttnn auto matmuls (cores=None, the configuration validated on P150x8 for the 8+1+1 head
    layout) and the head_dim-128 SDPA chunk sizes used by Llama-3.x-70B.
    """

    # SDPA chunk sizes
    decode_k_chunk_size: int = 128
    prefill_q_chunk_size_small: int = 64  # head_dim-128 values (the hd64 fork used 32/32)
    prefill_k_chunk_size_small: int = 64
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Matmul configs - None = ttnn auto-optimize (recommended for phase 1)
    decode_qkv_cores: tuple[int, int] | None = None
    decode_out_cores: tuple[int, int] | None = None
    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_out_cores: tuple[int, int] | None = None

    # Phase 2 candidates (only after PCC passes; each must satisfy ProgramConfig._build_matmul_config's
    # n // 32 % (cx * cy) == 0 and in0_block_w | K_tiles):
    #   decode_qkv_cores=(8, 5), decode_qkv_in0_block_w=4       (N=40 tiles -> 1 tile per core, K=128 tiles)
    #   decode_out_cores=(8, 8), decode_out_in0_block_w=4       (N=128 tiles -> per_core_N=2, K=32 tiles)
    #   prefill_qkv_cores=(8, 5), prefill_out_cores=(8, 8)
    # or Llama-70B-style DRAM-sharded configs (QKV: 32 cores, in0_block_w=4, per_core_N=2; WO: 8 cores, per_core_N=16).
