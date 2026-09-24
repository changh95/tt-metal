# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-9B config for Blackhole P150.

Subclasses tt_transformers.ModelArgs. HF_MODEL env var is canonical (hub id or local dir);
hub ids are snapshot_download'd first (AutoConfig on bare hub id is unreliable here).
Qwen3.5-specific params (GDN, partial RoPE, layer types) come from HF text config.
load_state_dict/weight_cache_path override the base meta-key (wq/wk/wv) scheme.
"""
import os
from pathlib import Path

from models.tt_transformers.tt.model_config import ModelArgs

# Qwen3.6-27B Blackhole (P300X2/P150x4 TP) serving optimizations are ON BY DEFAULT. These are the
# validated "optimized" config from BENCHMARKS.md (TTFT 1.4-4.2x, TPOT 1.3-2.2x vs the reference
# build); each is set via setdefault so any flag can still be overridden from the environment (set a
# flag to its old value to disable it). Runs at import of this module, before any layer __init__
# reads a flag. QWEN_SDPA_BF8 is the one precision change (bf16->bf8 KV; decode PCC 0.9999,
# 64k retrieval matches bf16) and is included by request; export QWEN_SDPA_BF8=0 for bf16 KV.
_QWEN36_SERVING_OPT_DEFAULTS = {
    "QWEN36_GDN_OUT_MODE": "agmm",
    "QWEN36_GDN_CONV": "kda",
    "QWEN36_AGMM_LAYOUT": "nt11x8",
    "QWEN36_SDPA_K_CHUNK": "256",
    "QWEN36_GDN_DECODE_FUSED": "2",
    "QWEN36_GDN_SLOT_DEVICE_COPY": "2",
    "TT_SDPA_GQA_MCAST": "1",
    "TT_GDN_SCAN_MCAST": "1",
    "TT_SDPA_GQA_MCAST_QPAIR": "1",
    "QWEN36_GDN_PROJ_CHUNKS": "1",
    "QWEN36_GDN_GB_BF16": "1",
    "QWEN36_KDA_TILE_IN": "1",
    "QWEN36_AGMM_BARRIER": "1",
    "QWEN36_PREFILL_LOGITS_FAST": "1",
    "QWEN36_PREFILL_BUCKET_TRACE": "1",
    "QWEN_SDPA_BF8": "1",
    # Small-M prefill matmuls (tp_common "Small-M prefill matmuls"): buckets <= 128 rows run all-gather + 1D mcast
    # matmuls instead of the M-padding AGMM / 4-row 2D configs. "0" = old path. 256 is measured faster too (-5 ms)
    # but its greedy-token check failed (tests/test_prefill_smallm_ref_scratch.py, logs/itemJ_ref_check.log), so the
    # 256 bucket stays on the AGMM/2D path until a passing check exists; the 128 bucket passed (PCC >= 0.9995, greedy
    # identical over 8 steps).
    "QWEN36_PREFILL_SMALLM_MAX": "128",
}
for _k, _v in _QWEN36_SERVING_OPT_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

# l1_small_size the GDN prefill depthwise ttnn.conv1d requires.
GDN_CONV1D_L1_SMALL_SIZE = 24576

# DRAM-sharded decode matmul tunables (27B TP=4 per-device shapes; see _init_tp_config):
#   workers    = num_workers_per_dram_bank (1-3; 2-3 need the tt-metal fix in
#                tt_metal/impl/device/experimental/device.cpp that measures the reader NOC hop distance on the
#                mesh's first device instead of asserting a unit mesh -- an older build TT_FATALs at program creation)
#   in0_cores  = L1 width-shard grid of the activation (row-major cores of the compute grid)
#   per_core_n = OUTPUT storage shard width in tiles (output grid = ceil(N_tiles / per_core_n) cores; must give
#                >= 8 * workers cores, every reader core has to own an output shard)
#   in0_block_w (optional) = K block; default largest divisor <= 8 of the in0 shard width
# Microbench (tests/test_decode_proj_dram_sharded_bench_scratch.py, P150x4, traced us/op, 2026-09-24
# logs/itemA_bench2_w123.log; 1D = the tuned default):
#   gate  1D 50.7 | w1 71.1 | w2 50.3 | w3 43.5 (in0 32x5, per_core_n 4)      -> gateup on w3
#   up    1D 45.7 | w1 54.0 | w2 39.5 | w3 35.7 (in0 32x5, per_core_n 4)      -> gateup on w3
#   down  1D 66.1 | w1 54.2 | w2 82.4 | w3 60.1                               -> stays w1
#   qkvzab 1D 62.6 | w1 86.6 | w2 65.2 | w3 60.1;  attn_qkv 1D 53.4 | w1 73.4 | w2 51.9 | w3 50.2;
#   out/wo 1D 27.2 | w1 34.7 | w2 32.6 | w3 27.5  -> not worth it: both in-projections feed ttnn.slice, which
#   needs an interleaved re-layout of the width-sharded output (~3 us) that eats the 2.5-3 us standalone gain.
DS_DECODE_CFG = {
    # down: K 4352 (136 tiles -> 17 cores x 8) N 5120 bfp8 LoFi; 32 storage cores x 5 tiles. Measured 54.6 us
    # (1D 33-core: 66.2); 34 cores x 4 tiles: 54.7; per_core_n 10: 54.7.
    "mlp_w2": {"workers": 1, "in0_cores": 17, "per_core_n": 5},
    # gate/up: K 5120 (the ff-norm shard: 32 cores x 5 tiles, fed as-is) N 4352 bfp4 LoFi, 3 readers/bank
    # (24 reader cores, per-bank storage 18 tiles -> 4608 padded columns, +6%); output 34 cores x 4 tiles.
    "mlp_w13": {"workers": 3, "per_core_n": 4},
}


def decode_dram_sharded_matrices():
    """QWEN36_DECODE_DRAM_SHARDED -> the set of decode projections that run on the DRAM-sharded kernel.

    Unset: "all" (the default since 2026-09-24). "0": none (the 1D mcast path). "1": down only (the original opt-in).
    Otherwise a comma list of down | gateup (aliases gate, up) | all, e.g. "gateup" or "down,gateup"."""
    # Default "all" since the 2026-09-24 P/D-stack gate: gateup GSM8K-200 0.835, all 0.82 (baseline 0.825 +- 0.027), zero
    # degenerate answers, greedy deterministic; b1 TPOT 29.4 -> 27.2 ms, 32 users 128/128 591 -> 648 t/s. "0" restores the
    # 1D mcast path byte-for-byte.
    v = os.environ.get("QWEN36_DECODE_DRAM_SHARDED", "all").strip().lower()
    if v in ("", "0"):
        return set()
    if v == "1":
        return {"down"}
    names = set()
    for tok in v.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "all":
            names |= {"down", "gateup"}
        elif tok in ("gate", "up", "gateup", "gate_up", "w1", "w3"):
            names.add("gateup")
        elif tok in ("down", "w2"):
            names.add("down")
        else:
            raise ValueError(f"QWEN36_DECODE_DRAM_SHARDED: unknown entry {tok!r} (down | gateup | all | 0 | 1)")
    return names


class Qwen36ModelArgs(ModelArgs):
    """Qwen3.5-9B ModelArgs for Blackhole P150."""

    # Opt into base ModelArgs TP > n_kv_heads path; attention/tp.py replicates via replicate_kv_weight.
    SUPPORTS_KV_REPLICATION = True

    def __init__(
        self,
        mesh_device=None,
        max_batch_size=1,
        max_seq_len=2048,
        **kwargs,
    ):
        # HF_MODEL is canonical (defaults to Qwen/Qwen3.6-27B). Snapshot hub ids unless
        # config.json exists locally (avoids cache-dir false positives).
        hf_model = os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.6-27B")
        if not os.path.isfile(os.path.join(hf_model, "config.json")):
            from huggingface_hub import snapshot_download

            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
            os.environ["HF_MODEL"] = snapshot_download(hf_model, local_files_only=offline)
        super().__init__(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)
        if mesh_device is not None:
            self.model_config["SAMPLING_AG_CONFIG"]["allow_force_argmax"] = True

        # Mirror CKPT_DIR -> checkpoint_dir for weight_cache_path / load_state_dict.
        self.checkpoint_dir = self.CKPT_DIR

        # Qwen3.5-specific params from HF text config (base sets dim, heads, layers, etc.).
        text_config = self.hf_config.get_text_config()

        # RoPE: read partial_rotary_factor from rope_parameters first (some configs nest only there).
        # Top-level-only read silently used 1.0 and broke long-context RoPE on 3.5-27B.
        rope_params = getattr(text_config, "rope_parameters", None) or {}
        self.rope_theta = rope_params.get("rope_theta", 10_000_000)
        self.partial_rotary_factor = rope_params.get(
            "partial_rotary_factor", getattr(text_config, "partial_rotary_factor", 1.0)
        )
        self.rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

        # M-RoPE (multimodal rotary). The 3 sections (T, H, W) sum to rope_head_dim // 2 and drive
        # the interleaved-mrope cos/sin (modeling_qwen3_5.Qwen3_5RotaryEmbedding). For the "default"
        # rope type Qwen3.5 uses, attention_scaling is 1.0 (so text cos/sin are unchanged). The
        # spatial_merge_size + image/video token ids let the model derive the 3D position ids on
        # host from input_ids + image_grid_thw (no dependency on mm_token_type_ids from the caller).
        self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        self.rope_attention_scaling = 1.0
        vision_config = getattr(self.hf_config, "vision_config", None)
        self.spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
        self.image_token_id = getattr(self.hf_config, "image_token_id", None)
        self.video_token_id = getattr(self.hf_config, "video_token_id", None)

        # DeltaNet-specific parameters (base does not know about these)
        self.linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
        self.linear_num_value_heads = getattr(text_config, "linear_num_value_heads", 32)
        self.linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
        self.linear_value_head_dim = getattr(text_config, "linear_value_head_dim", 128)
        self.linear_conv_kernel_dim = getattr(text_config, "linear_conv_kernel_dim", 4)

        # Full layer_types list for DeltaNet vs full-attn dispatch.
        self.attention_type_list = getattr(text_config, "layer_types", None) or (
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
        )

        # Derived
        self.linear_q_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_k_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_v_dim = self.linear_num_value_heads * self.linear_value_head_dim

        # Lazy import for CPU-only testing.
        if mesh_device is not None:
            import ttnn

            self.weight_dtype = ttnn.bfloat8_b
            self.act_dtype = ttnn.bfloat16
        else:
            self.weight_dtype = None
            self.act_dtype = None

        # TP config (num_devices>1 only). 27B (1,4) sharded dims + DRAM matmul cfgs; see tp_common.py.
        self.num_devices = mesh_device.get_num_devices() if mesh_device is not None else 1
        if mesh_device is not None and self.num_devices > 1:
            self._init_tp_config(mesh_device)

    def _init_tp_config(self, mesh_device):
        """Per-device sharded dims + DRAM matmul/mem configs for TP (num_devices>1)."""
        import ttnn
        from models.common.utility_functions import is_blackhole
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        tp = self.num_devices
        self.cluster_shape = list(mesh_device.shape)

        # GDN dims (match qwen35_27b reference names).
        self.gdn_nk = self.linear_num_key_heads
        self.gdn_dk = self.linear_key_head_dim
        self.gdn_nv = self.linear_num_value_heads
        self.gdn_dv = self.linear_value_head_dim
        self.gdn_conv_kernel_size = self.linear_conv_kernel_dim
        self.gdn_key_dim = self.linear_q_dim  # q and k equal
        self.gdn_value_dim = self.linear_v_dim
        self.gdn_qkv_dim = self.linear_q_dim + self.linear_k_dim + self.linear_v_dim
        self.gdn_z_dim = self.linear_v_dim
        self.gdn_chunk_size = 128  # GDN seq kernel requires 128

        # Per-device (sharded) dims
        assert self.n_heads % tp == 0, f"n_heads {self.n_heads} not divisible by TP={tp}"
        assert self.gdn_nk % tp == 0 and self.gdn_nv % tp == 0, "GDN head counts must divide by TP"
        self.n_local_heads = self.n_heads // tp
        self.n_local_kv_heads = max(1, self.n_kv_heads // tp)
        self.kv_replication = tp > self.n_kv_heads  # False at TP=4 (4 KV heads)
        self.gdn_nk_tp = self.gdn_nk // tp
        self.gdn_nv_tp = self.gdn_nv // tp
        self.gdn_qkv_dim_tp = self.gdn_qkv_dim // tp
        self.gdn_z_dim_tp = self.gdn_z_dim // tp
        self.gdn_qkvz_dim_tp = (self.gdn_qkv_dim + self.gdn_z_dim) // tp
        # Per-device width of the [qkv|z|a|b] fused in-projection: folding the tiny a/b (decay/beta)
        # projection into qkvz removes a whole decode matmul while keeping the (good) K=dim. Default
        # (was QWEN36_GDN_FUSE_AB); gdn/tp.py fuses whenever the qkvz weight is DRAM-sharded.
        self.gdn_qkvzab_dim_tp = self.gdn_qkvz_dim_tp + 2 * self.gdn_nv_tp
        self.gdn_value_dim_tp = self.gdn_value_dim // tp
        self.gdn_key_dim_tp = self.gdn_key_dim // tp
        self.attn_out_dim_tp = (self.n_heads * self.head_dim) // tp
        kv_dim_per_device = self.n_local_kv_heads * self.head_dim

        # DRAM-sharded weights: column-parallel [hidden, out_tp]
        self.gdn_qkvz_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.gdn_qkvz_dim_tp)
        self.gdn_qkvzab_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.gdn_qkvzab_dim_tp)
        self.attn_qg_weight_memcfg = tpc.create_dram_sharded_mem_config(
            self.dim, self.n_local_heads * self.head_dim * 2
        )
        self.attn_k_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, kv_dim_per_device)
        self.attn_v_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, kv_dim_per_device)
        # Fused [q+gate | k | v] in-projection (P4: QWEN36_FUSED_QKV) — one column-parallel matmul.
        self.attn_qkv_fused_dim_tp = self.n_local_heads * self.head_dim * 2 + 2 * kv_dim_per_device
        self.attn_qkv_fused_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.attn_qkv_fused_dim_tp)
        self.mlp_w1_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.hidden_dim // tp)
        self.mlp_w3_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.hidden_dim // tp)
        # row-parallel out-projections: DRAM-INTERLEAVED (None -> plain ttnn.linear); DRAM-sharding narrow-K here loses to the interleaved 1D kernel and adds 2 reshards/layer.
        self.gdn_out_weight_memcfg = None
        self.attn_wo_weight_memcfg = None
        self.mlp_w2_weight_memcfg = tpc.create_dram_sharded_mem_config(self.hidden_dim // tp, self.dim)

        # DRAM-sharded matmul progcfgs (decode, M=1)
        M = 1
        self.gdn_qkvz_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.gdn_qkvz_dim_tp)
        self.gdn_qkvzab_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.gdn_qkvzab_dim_tp)
        self.gdn_out_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.gdn_value_dim_tp, self.dim)
        self.attn_qg_progcfg = tpc.create_dram_sharded_matmul_program_config(
            M, self.dim, self.n_local_heads * self.head_dim * 2
        )
        self.attn_k_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, kv_dim_per_device)
        self.attn_v_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, kv_dim_per_device)
        self.attn_qkv_fused_progcfg = tpc.create_dram_sharded_matmul_program_config(
            M, self.dim, self.attn_qkv_fused_dim_tp
        )
        self.attn_wo_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.attn_out_dim_tp, self.dim)
        self.mlp_w1_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.hidden_dim // tp)
        self.mlp_w3_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.hidden_dim // tp)
        self.mlp_w2_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.hidden_dim // tp, self.dim)

        # 1D decode MLP matmuls (DEFAULT): small grids beat the ~80-core DRAM-sharded grid on the
        # bandwidth-bound skinny (M<=1) decode matmuls. Interleaved weights.
        # decode_grid_w = the device worker-grid width (11 on BH P150, 8 on WH). Shaping the 1D-mcast
        # grid WIDE-first (up to this many cols) beats the old cols<=8 shaping by ~2% on this matmul —
        # a wide-short grid shortens the in0 multicast column (test_mlp_matmul_sweep wide1d_* vs
        # forced1d_*). Applied to gate/up ONLY (the swept, verified projections); the others below keep
        # the legacy cols<=8 shaping (grid_w default) until their shapes are swept too.
        self.decode_grid_w = mesh_device.compute_with_storage_grid_size().x
        self.mlp_1d_decode = True
        # gate/up: num_cores=44 -> 11x4 on BH, the fastest measured config (wide1d_11x4c, 42.8us vs
        # 43.9us for the old 8x4=forced1d_32c). On WH (decode_grid_w=8) this falls back to 8x6.
        self.mlp_w1_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.dim,
            self.hidden_dim // tp,
            num_cores=44,
            fused_activation=ttnn.UnaryOpType.SILU,
            grid_w=self.decode_grid_w,
        )
        self.mlp_w3_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.hidden_dim // tp, num_cores=44, grid_w=self.decode_grid_w
        )
        # down: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~63us, +28% vs
        # the old 8x2). On WH (decode_grid_w=8) this falls back to 8x5.
        self.mlp_w2_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.hidden_dim // tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )

        # Input-projection 1D decode (DEFAULT): same idea for attn QKV+gate and GDN QKVZAB in-projections.
        # Weights load interleaved (prefill AGMM verified bit-identical); tuned grids per test_mlp_matmul_sweep.
        self.proj_1d_decode = True
        self.attn_qkv_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.attn_qkv_fused_dim_tp, num_cores=64
        )
        # gdn_qkvz: num_cores=44 -> 11x4 on BH, the fastest measured config (wide1d_11x4c, ~59us, +22%
        # vs the old 8x5). On WH (decode_grid_w=8) this falls back to 8x6.
        self.gdn_qkvz_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.gdn_qkvzab_dim_tp, num_cores=44, grid_w=self.decode_grid_w
        )
        # Output projections (attn wo, GDN o_proj): already interleaved+auto (no weight relayout, not in
        # the prefill AGMM fusion), so this just swaps ttnn-auto for a tuned ~32-core 1D decode grid.
        # attn_wo: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~24us, +25%
        # vs the old 8x4). On WH (decode_grid_w=8) this falls back to 8x5.
        self.attn_wo_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.attn_out_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )
        # gdn_out: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~24us, +25%
        # vs the old 8x4; same 1536x5120 shape as attn_wo). On WH (decode_grid_w=8) this falls back to 8x5.
        self.gdn_out_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.gdn_value_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )

        # DRAM-sharded decode down-projection (OPT-IN: QWEN36_DECODE_DRAM_SHARDED=1 or ...,down). Item A findings
        # (P150x4, TP=4, 2026-09-21):
        #  * The multi-reader kernel (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig.num_workers_per_dram_bank
        #    2-3, tt-metal #54242) WAS unreachable on a multi-device mesh: its reader placement calls
        #    experimental::Device::get_worker_noc_hop_distance, which TT_FATALed "only supported on unit MeshDevice"
        #    for the (1,4) mesh. Fixed 2026-09-24 in tt_metal/impl/device/experimental/device.cpp (measures on the
        #    mesh's first local device, best-effort under heterogeneous harvesting); DS_DECODE_CFG has the numbers.
        #  * Measured with tests/test_decode_proj_dram_sharded_bench_scratch.py (traced us/op, 1 worker vs the
        #    tuned 1D path): gate 71 vs 51, up 54 vs 46, gdn qkvzab 87 vs 63, out/wo 35 vs 28, attn qkv 74 vs 53
        #    -> those stay on the 1D kernels; down 54.6 vs 66.2 -> the 8 bank readers stream the 23.7 MB bfp8
        #    weight at ~430 GB/s (the 1D 33-core grid: ~360). Model harness (profile_prefill_decode, TP=4):
        #    decode w1 25.63 -> 25.12 ms, w32 33.15 -> 32.56 ms with down on this path.
        #  * Numerics: the K accumulation differs at the bf16-lsb level per layer (max|d| 0.0078 vs the 1D result
        #    on random data, pcc_vs_1d 0.9999998), which through 64 layers + the GDN state gives decode-logits
        #    PCC 0.9991-0.99997 vs the 1D reference (tests/test_decode_proj_dram_sharded_ref.py; greedy tokens
        #    over 8 steps x 4 prompts identical, prefill argmax identical, but top-1 flips on 1-4 of 32 rows of
        #    the synthetic fixed-input steps). That misses the item's PCC >= 0.9999 bar, so the path is OFF by
        #    default (the flag reproduces the 1D reference bit-exactly when off) and kept as an opt-in.
        # When on: decode-only WIDTH_SHARDED copy of w2 (~24 MB/layer; prefill keeps the interleaved one for the
        # 2D kernel); in0 = silu(gate)*up resharded to 17 cores x 8 tiles (in0_block_w 8; 8 cores x 1-tile blocks
        # measured 144 us); the L1 width-sharded output is re-laid-out to L1 interleaved for the reduce-scatter.
        self.decode_grid_size = mesh_device.compute_with_storage_grid_size()
        _ds = decode_dram_sharded_matrices()
        self.mlp_w2_ds_decode = is_blackhole() and "down" in _ds
        c2 = DS_DECODE_CFG["mlp_w2"]
        self.mlp_w2_ds_workers = c2["workers"]
        self.mlp_w2_ds_memcfg = tpc.create_dram_sharded_mem_config(self.hidden_dim // tp, self.dim, c2["workers"])
        _hid_tiles = self.hidden_dim // tp // tpc.TILE_SIZE
        assert _hid_tiles % c2["in0_cores"] == 0, (_hid_tiles, c2["in0_cores"])
        self.act_shard_mlp_hidden = tpc.create_width_shard_config(
            self.hidden_dim // tp, c2["in0_cores"], self.decode_grid_size
        )
        self.mlp_w2_ds_progcfg = tpc.create_dram_sharded_decode_progcfg(
            _hid_tiles // c2["in0_cores"], c2["per_core_n"], c2["workers"], in0_block_w=c2.get("in0_block_w")
        )
        # DRAM-sharded gate/up with 3 readers per bank (OPT-IN: QWEN36_DECODE_DRAM_SHARDED=gateup). in0 is the
        # ff-norm width shard itself (act_shard_hidden: 32 cores x 5 tiles -> in0_block_w 5), so mlp.py feeds it
        # without the interleave the 1D path needs; the L1 width-sharded outputs are multiplied shard-locally and
        # re-laid out once for the down-proj input.
        self.mlp_w13_ds_decode = is_blackhole() and "gateup" in _ds
        c13 = DS_DECODE_CFG["mlp_w13"]
        self.mlp_w13_ds_workers = c13["workers"]
        self.mlp_w1_ds_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.hidden_dim // tp, c13["workers"])
        self.mlp_w3_ds_memcfg = self.mlp_w1_ds_memcfg
        _dim_tiles = self.dim // tpc.TILE_SIZE
        _nr, _nc = tpc._find_grid(_dim_tiles)  # the act_shard_hidden grid (create_activation_shard_config)
        assert _dim_tiles % (_nr * _nc) == 0, (_dim_tiles, _nr, _nc)
        self.mlp_w1_ds_progcfg = tpc.create_dram_sharded_decode_progcfg(
            _dim_tiles // (_nr * _nc),
            c13["per_core_n"],
            c13["workers"],
            fused_activation=ttnn.UnaryOpType.SILU,
            in0_block_w=c13.get("in0_block_w"),
        )
        self.mlp_w3_ds_progcfg = tpc.create_dram_sharded_decode_progcfg(
            _dim_tiles // (_nr * _nc), c13["per_core_n"], c13["workers"], in0_block_w=c13.get("in0_block_w")
        )

        # Prefill matmul factory (M = seq_len)
        self._prefill_grid = tpc.prefill_grid_default()
        self.prefill_tuning = tpc.prefill_tuning(tp)
        self.prefill_progcfg = lambda seq_len, k, n: tpc.create_prefill_matmul_program_config(
            seq_len, k, n, grid_size=self._prefill_grid, tuning=self.prefill_tuning
        )

        # Activation shard configs
        self.act_shard_hidden = tpc.create_activation_shard_config(self.dim)
        self.act_shard_gdn_value = tpc.create_activation_shard_config(self.gdn_value_dim_tp)
        self.act_shard_attn_out = tpc.create_activation_shard_config(self.attn_out_dim_tp)

        # KV-cache height shard for paged_update_cache (one user per core).
        _B = max(1, self.max_batch_size)
        _cols = next(c for c in range(min(8, _B), 0, -1) if _B % c == 0)
        _rows = _B // _cols
        self.kv_update_shard_cfg = ttnn.create_sharded_memory_config(
            shape=(tpc.TILE_SIZE, self.head_dim),
            core_grid=ttnn.CoreGrid(x=_cols, y=_rows),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _set_hf_params(self, checkpoint_dir):
        # trust_remote_code before base AutoConfig load.
        self.trust_remote_code_hf = True
        super()._set_hf_params(checkpoint_dir)

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return self.attention_type_list[layer_idx] == "full_attention"

    def is_deltanet_layer(self, layer_idx: int) -> bool:
        return self.attention_type_list[layer_idx] == "linear_attention"

    def weight_cache_path(self, dtype=None):
        """Weight tensor cache dir, rooted at model_cache_path (TT_CACHE_PATH + device), NOT the HF
        snapshot (often read-only in CI -> caching there silently never persists); falls back to the
        checkpoint dir. TP caches qualified by mesh shape: per-device layouts differ by mesh and
        as_tensor reloads a cache file as-is, IGNORING mesh_mapper (single device keeps the
        unqualified path so validated 9B behavior is unchanged)."""
        if dtype is None:
            dtype = self.weight_dtype
        import ttnn

        if dtype == ttnn.bfloat8_b:
            suffix = "tensor_cache_bfp8"
        else:
            suffix = "tensor_cache_bf16"
        if self.num_devices > 1:
            suffix += "_mesh" + "x".join(str(d) for d in self.cluster_shape)
        root = getattr(self, "model_cache_path", None) or Path(self.checkpoint_dir)
        return Path(root) / suffix

    def load_state_dict(self):
        """Load + remap weights via the text-only HF Qwen3_5ForCausalLM.
        Overrides base meta-key loader."""
        from models.demos.blackhole.qwen36.tt.weight_mapping import (
            is_fp8_checkpoint,
            load_qwen36_state_dict_fp8,
            remap_qwen36_state_dict,
        )

        # Block FP8 checkpoints: dequant + remap for TP loaders (skip the HF model).
        if is_fp8_checkpoint(self.CKPT_DIR):
            return load_qwen36_state_dict_fp8(self.CKPT_DIR)

        # Import the HF classes directly rather than going through AutoModelForCausalLM.
        # Serving out-of-tree, vllm.transformers_utils.config registers vLLM's OWN
        # Qwen3_5Config for model_type "qwen3_5" into transformers' AutoConfig
        # (AutoConfig.register(..., exist_ok=True)), so AutoConfig hands back vLLM's class.
        # transformers only unwraps a composite config to its text sub-config when
        # `model_class.config_class == config.sub_configs["text_config"]` — an identity
        # check that cannot hold across libraries — so the composite config would reach
        # Qwen3_5ForCausalLM and fail on `config.vocab_size` (which lives one level down,
        # in text_config). Naming the classes here keeps config and model from the same
        # library, matching vision/vision_model_config.py::reference_vision_model.
        #
        # Qwen3_5TextConfig.from_pretrained picks the `text_config` sub-dict on composite
        # (3.6 VLM) checkpoints via base_config_key, and reads a text-only (3.5) config.json
        # as-is, so both checkpoint layouts land on the config Qwen3_5ForCausalLM expects.
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        text_config = Qwen3_5TextConfig.from_pretrained(self.CKPT_DIR)
        assert text_config.vocab_size == self.vocab_size and text_config.hidden_size == self.dim, (
            f"HF text config disagrees with model args: vocab_size {text_config.vocab_size} vs "
            f"{self.vocab_size}, hidden_size {text_config.hidden_size} vs {self.dim}"
        )
        model = Qwen3_5ForCausalLM.from_pretrained(self.CKPT_DIR, config=text_config, dtype="auto")
        state_dict = remap_qwen36_state_dict(model.state_dict())
        del model
        return state_dict
