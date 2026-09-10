# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Solar-Open attention program configurations."""

import os
from dataclasses import dataclass

from loguru import logger

from models.demos.solar_open.tt.attention.config import ProgramConfig

# Phase 3e (design_decode_levers.md 3.5, lane B1 / P1) decode attention levers. Read when a
# ``SolarOpenAttentionProgramConfig`` is constructed with the field left at its "resolve" value (the layers construct
# one each with the mesh's TP, ``tt/layer.py``; ``tt/model.py::create_rope_setup`` resolves ``fused_qk`` the same way
# for the RotarySetup), so one process sees one consistent answer; an explicit field value wins over the environment.
# P1 (2026-09-10) measured both levers BIT-IDENTICAL to the phase-3d chain (op gate, component cells, real-weight
# layer 0, teacher-forced b1 / b32 to the digit) and faster (traced real layer 0: fused_qk -4 us at b1, o_proj -8 us
# at b32; demo b32 36.64 -> 36.0 ms/step), so both DEFAULT ON for TP > 1 (README "Phase 3e rows", stage P1). TP = 1
# keeps the phase-3d chain: its qkv is DRAM interleaved (no 2B-core create-heads placement) and the (8, 8) o_proj was
# measured at the TP = 8 shapes only.
#   SOLAR_OPEN_ATTENTION_FUSED_QK=1|0|auto  slice 1: create_heads on a 2B-core grid + rotary_embedding_llama_fused_qk +
#                                     paged_fused_update_cache (3 launches instead of 5 per layer; bit-identical);
#                                     auto / unset = the default rule (on for TP > 1); 1 forces it (TP = 1 raises)
#   SOLAR_OPEN_ATTENTION_OUT_GRID=8x8|auto  slice 2: o_proj as the explicit (8, 8) 1D config from an L1-interleaved in0
#                                     with the HiFi2 compute config restated (bit-identical to the auto linear: the
#                                     same 4-tile K blocks); auto / 0 / none = the auto linear; unset = the default
#                                     rule ((8, 8) for TP > 1); any "WxH" grid that divides the 128 N tiles
ATTENTION_FUSED_QK_ENV = "SOLAR_OPEN_ATTENTION_FUSED_QK"
ATTENTION_OUT_GRID_ENV = "SOLAR_OPEN_ATTENTION_OUT_GRID"
ATTENTION_FUSED_QK_DEFAULT = True  # phase 3e / P1 (False = the phase-3d chain; was the default through the P1 gates)
ATTENTION_OUT_GRID_DEFAULT: tuple[int, int] | None = (
    8,
    8,
)  # phase 3e / P1 (None = the auto o_proj, the P1 gates' reference)
AUTO = "auto"  # attention_out_grid_from_env: the auto linear asked for explicitly
FROM_ENV = "env"  # SolarOpenAttentionProgramConfig.decode_out_cores: resolve from the environment / the default rule


def attention_fused_qk_from_env() -> bool | None:
    """``SOLAR_OPEN_ATTENTION_FUSED_QK``: ``1`` -> True, ``0`` -> False, unset / empty / ``auto`` -> None (= the default
    rule of ``resolve_fused_qk``); anything else raises."""
    raw = (os.getenv(ATTENTION_FUSED_QK_ENV) or "").strip().lower()
    if raw in ("", AUTO):
        return None
    if raw == "1":
        return True
    if raw == "0":
        return False
    raise ValueError(f"{ATTENTION_FUSED_QK_ENV}={raw!r}: expected 1, 0 or auto")


def resolve_fused_qk(tp: int | None) -> bool:
    """The effective ``fused_qk`` for a mesh with ``tp`` devices on the TP axis: the environment when set (``1`` even at
    TP = 1, where the decode chain then raises), else ``ATTENTION_FUSED_QK_DEFAULT`` restricted to TP > 1 -- TP = 1 has a
    DRAM-interleaved qkv, which forces the overlapped create-heads grid, so the phase-3d chain runs there. ``tp`` None =
    unknown (a bare config) = the TP > 1 production layout."""
    env = attention_fused_qk_from_env()
    if env is not None:
        return env
    return ATTENTION_FUSED_QK_DEFAULT and (tp is None or tp > 1)


def attention_out_grid_from_env() -> tuple[int, int] | str | None:
    """``SOLAR_OPEN_ATTENTION_OUT_GRID=WxH`` -> ``(W, H)``; ``auto`` / ``0`` / ``none`` -> ``AUTO`` (the auto o_proj,
    explicitly); unset / empty -> None (= the default rule of ``resolve_out_grid``)."""
    raw = (os.getenv(ATTENTION_OUT_GRID_ENV) or "").strip().lower()
    if raw == "":
        return None
    if raw in (AUTO, "0", "none"):
        return AUTO
    try:
        w, h = (int(part) for part in raw.split("x"))
    except ValueError as exc:
        raise ValueError(f"{ATTENTION_OUT_GRID_ENV}={raw!r} is not a WxH core grid (e.g. 8x8)") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"{ATTENTION_OUT_GRID_ENV}={raw!r}: the grid must be positive")
    return (w, h)


def resolve_out_grid(tp: int | None) -> tuple[int, int] | None:
    """The effective ``decode_out_cores``: the environment when set (``AUTO`` -> None = the auto linear), else
    ``ATTENTION_OUT_GRID_DEFAULT`` for TP > 1 and None at TP = 1 (the (8, 8) config from the interleaved in0 was measured
    at the TP = 8 shapes only). ``tp`` None = unknown (a bare config) = the TP > 1 production layout."""
    env = attention_out_grid_from_env()
    if env is not None:
        return None if env == AUTO else env
    return ATTENTION_OUT_GRID_DEFAULT if (tp is None or tp > 1) else None


_LEVERS_LOGGED = False


def log_attention_levers_once(program_config) -> None:
    """One INFO line per process naming the decode attention arms in force (the log confirmation of every device run)."""
    global _LEVERS_LOGGED
    if _LEVERS_LOGGED:
        return
    _LEVERS_LOGGED = True
    logger.info(
        f"decode attention levers: fused_qk={program_config.fused_qk} ({ATTENTION_FUSED_QK_ENV}), "
        f"o_proj grid={program_config.decode_out_cores} ({ATTENTION_OUT_GRID_ENV}; None = the auto linear)"
    )


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
    projections stay auto. Phase 3e (lane B1, design_decode_levers.md 3.5) adds two OFF-by-default decode levers:
    ``fused_qk`` (SOLAR_OPEN_ATTENTION_FUSED_QK=1: fused Q/K RoPE + fused K/V cache update, bit-identical expected)
    and the explicit o_proj from an interleaved in0 (SOLAR_OPEN_ATTENTION_OUT_GRID=8x8, PCC-gated).
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
    # o_proj (phase 3e, lever B1 slice 2; default ON since P1): FROM_ENV (the default) = SOLAR_OPEN_ATTENTION_OUT_GRID
    # when set, else ATTENTION_OUT_GRID_DEFAULT = (8, 8) for TP > 1 and the auto linear at TP = 1 (resolve_out_grid in
    # __post_init__); None = the auto linear on the width-sharded concat_heads output (phases 1-3d), explicitly; a
    # tuple = that grid. (8, 8) = the explicit 1D config (per_core_N 2, in0_block_w 4, out_subblock_w 2, HiFi2 restated
    # by get_decode_out_compute_config) that measured 11.9 us vs 18.6-23 us in the micro-benchmark FROM AN INTERLEAVED
    # in0 -- so decode_out_interleave_in0 moves the sharded_to_interleaved reshard (1 us, needed anyway before the
    # reshape / all-reduce) in front of the matmul. P1 measured it BIT-IDENTICAL to the auto linear (the auto blocking
    # is the same 4-tile K blocks: test_config_candidates ``o_proj_8x8_interleaved`` identical, real-weight layer 0 and
    # teacher-forced b1 / b32 to the digit) and -8 us per traced b32 layer (-2.5 us at b1).
    decode_out_cores: tuple[int, int] | str | None = FROM_ENV
    decode_out_in0_block_w: int = 4
    decode_out_out_subblock_w: int = 2
    decode_out_interleave_in0: bool = True
    decode_out_fp32_dest_acc: bool = False
    # Fused decode attention chain (phase 3e, lever B1 slice 1; ProgramConfig.fused_qk docstring; default ON since P1):
    # None = resolve_fused_qk(tp) (SOLAR_OPEN_ATTENTION_FUSED_QK when set, else on for TP > 1), an explicit bool wins.
    # Needs the width-sharded qkv of TP > 1 (a non-sharded create_heads input forces overlap_qk_coregrid=True) and a
    # RotarySetup built with use_qk_fused=True (tt/model.py::create_rope_setup resolves the same rule). P1: bit-identical
    # to the phase-3d chain on every gate; -4 us per traced b1 layer (2 launches fewer).
    fused_qk: bool | None = None
    # The mesh's TP (tt/layer.py passes mesh_config.tp, create_rope_setup its own): decides the TP = 1 fallback of the
    # two default rules above. None = unknown = the TP > 1 production layout.
    tp: int | None = None
    prefill_qkv_cores: tuple[int, int] | None = None
    prefill_out_cores: tuple[int, int] | None = None

    # Precision option read by attention/operations.py::attention_bf16_output (design_misc (a)): True keeps the
    # attention branch bf16 through o_proj and its TP all-reduce (skips the two bfp8 typecasts); the env
    # SOLAR_OPEN_ATTENTION_BF16_OUTPUT=1 turns it on without this field. Default off until the teacher-forced test and
    # the demo step times have been measured with it (device lane).
    bf16_output: bool = False

    def __post_init__(self):
        if self.fused_qk is None:
            self.fused_qk = resolve_fused_qk(self.tp)
        if isinstance(self.decode_out_cores, str):
            if self.decode_out_cores != FROM_ENV:
                raise ValueError(
                    f"decode_out_cores: expected a (W, H) grid, None (auto) or FROM_ENV, got {self.decode_out_cores!r}"
                )
            self.decode_out_cores = resolve_out_grid(self.tp)
        super().__post_init__()
