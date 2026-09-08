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
    layout) and the head_dim-128 SDPA chunk sizes used by Llama-3.x-70B. Phase 2 (2026-09-07): SDPA chunks unchanged
    (decode SDPA is 8-10 us per layer at short contexts; re-measure at 8K-32K before touching decode_k_chunk_size);
    the decode qkv projection runs the explicit 1D config below (attention/decode.py), o_proj and the prefill
    projections stay auto.
    """

    # SDPA chunk sizes
    decode_k_chunk_size: int = 128
    prefill_q_chunk_size_small: int = 64  # head_dim-128 values (the hd64 fork used 32/32)
    prefill_k_chunk_size_small: int = 64
    prefill_q_chunk_size_large: int = 256
    prefill_k_chunk_size_large: int = 256
    prefill_threshold: int = 2048

    # Matmul configs - None = ttnn auto-optimize. Measured on one device at the real shapes
    # (tests/perf/test_config_candidates.py, torch fp32 reference of the device-rounded operands; tracy profile of the
    # real-weight layer 0):
    #   decode_qkv_cores=(8, 5), decode_qkv_in0_block_w=16      qkv [32,4096]x[4096,1280] -> L1 width sharded (1 tile per
    #                                                            core; nlp_create_qkv_heads_decode accepts the layout):
    #                                                            60.3 us auto (in0_block_w 2) -> 19.8 us kernel (bw8
    #                                                            20.9), ~-2 ms per decode step. WIRED (phase 2, perf-p0)
    #                                                            together with get_decode_qkv_compute_config: ttnn runs
    #                                                            an auto bf16 x bfp8 matmul at HiFi2 but drops to LoFi
    #                                                            as soon as a program config is given -- the candidate
    #                                                            test's PCC loss (0.99983 vs 0.99994) was that LoFi
    #                                                            fallback, not the K-block size; with HiFi2 + fp32 dst
    #                                                            restated the explicit config beats the auto numerics.
    #   decode_out_cores=(8, 8), decode_out_in0_block_w=4, decode_out_out_subblock_w=2
    #                                                            o_proj on the WIDTH-sharded nlp_concat_heads_decode
    #                                                            output (fuse_batch=True, shard 4 tiles wide -> in0_block_w
    #                                                            <= 4): 18.6 -> 11.9 us in the micro-benchmark with an
    #                                                            interleaved in0, but NO gain in the production layout
    #                                                            (wall 56 -> 64 us, PCC vs fp32 0.99988 vs 0.99996): not
    #                                                            recommended.
    #   prefill: leave auto (qkv 83 us / o_proj 28 us at 128 tokens; the attention all_reduce is the larger item).
    # decode_qkv_cores None restores the auto qkv matmul (A/B switch).
    decode_qkv_cores: tuple[int, int] | None = (8, 5)
    decode_qkv_in0_block_w: int = 16
    # fp32 destination accumulation for the wired qkv config (get_decode_qkv_compute_config). One-device numerics
    # vs the fp32 reference (tests/perf/test_config_candidates.py, 2026-09-07): auto 0.999936; (8,5) bw16 HiFi2
    # 0.999862 (bf16 dst, 8 K blocks of 16 tiles spill through the 16-bit dst), bw8 HiFi2 0.999927, bw16 HiFi2 +
    # fp32 dst 0.999994 at the same wall time (0.084 vs 0.083 ms; auto 0.125). Whole-model A/B (gate-p0 stage, same
    # day, teacher-forced vs the bf16 HF model): bf16 dst b1 top-1 0.9258 / top-5 0.9234 / full PCC 0.99162 / KL 0.0300,
    # b32 0.9297 / 0.9211 / 0.99064 / 0.0312 vs fp32 dst 0.9219 / 0.9086 / 0.99056 / 0.0333 and 0.9258 / 0.9164 /
    # 0.99123 / 0.0311; traced layer time identical (0.378 vs 0.377 ms); the random-init 1-layer test_model decode_b1_s1
    # logits PCC 0.99935 (bf16 dst) vs 0.99769 (fp32 dst, an interaction with the sharded decode norm) -> bf16 dst is
    # the default, True is the A/B switch.
    decode_qkv_fp32_dest_acc: bool = False
    decode_out_cores: tuple[int, int] | None = None
    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_out_cores: tuple[int, int] | None = None

    # Precision option read by attention/operations.py::attention_bf16_output (design_misc (a)): True keeps the
    # attention branch bf16 through o_proj and its TP all-reduce (skips the two bfp8 typecasts); the env
    # SOLAR_OPEN_ATTENTION_BF16_OUTPUT=1 turns it on without this field. Default off until the teacher-forced test and
    # the demo step times have been measured with it (device lane).
    bf16_output: bool = False
