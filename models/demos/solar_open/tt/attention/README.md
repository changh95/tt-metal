# Attention Module

Generic attention implementation with clean decode/prefill separation and configurable matmul program configs.
Solar-Open-100B uses it as bias-free, sink-free grouped-query attention with uniform full (causal) attention on all
48 layers and YaRN RoPE (theta 1e6, factor 2.0 beyond 65536 positions, head_dim 128).

## Structure

```
attention/
├── __init__.py      # Main Attention class with decode/prefill dispatch
├── config.py        # AttentionConfig + ProgramConfig base (SDPA chunking, Blackhole per-user decode grid)
├── weights.py       # Weight loading (fused per-device [Q|K|V] + row-parallel o_proj)
├── kv_cache.py      # KV cache initialization (paged or contiguous, bfloat8_b)
├── operations.py    # Common ops (RoPE incl. the fused Q/K variant, head split/concat, projections, TP all-reduce)
├── decode.py        # Decode forward (seq_len=1): decode_qkv_heads (legacy / fused chain), decode_output_projection
└── prefill.py       # Prefill forward (seq_len>1)
```

## Shapes (Solar-Open-100B, TP=8 on P150x8)

| tensor | host | per device |
|---|---|---|
| `wqkv` (column-parallel) | `[1, 1, 4096, 10240]` | `[1, 1, 4096, 1280]` = 8 q heads + 1 k head + 1 v head of head_dim 128 |
| `o_proj` (row-parallel) | `[8192, 4096]` | `[1024, 4096]` (partial sum, combined by the TP all-reduce) |
| paged KV cache per layer | | `[max_num_blocks, 1, block_size, 128]` bfloat8_b |

The `self_attn` state dict must be in Meta (interleaved) RoPE format (`convert_hf_qkv_to_meta_format`) and contains
only `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight`; biases and attention-sink logits are rejected.

## Usage

```python
from models.demos.solar_open.tt.attention import Attention, AttentionConfig
from models.demos.solar_open.tt.attention_configs import SolarOpenAttentionProgramConfig

# Create config (Solar-Open-100B values; sliding_window=None -> full attention)
config = AttentionConfig(
    hidden_size=4096,
    num_heads=64,
    num_kv_heads=8,
    head_dim=128,
    max_seq_len=131072,
    max_local_batch_size=32,
    sliding_window=None,
)

# Create attention with model-specific program config
attention = Attention(
    mesh_device=mesh_device,
    config=config,
    state_dict=state_dict,
    ccl_manager=ccl_manager,
    mesh_config=mesh_config,
    program_config=SolarOpenAttentionProgramConfig(),
    layer_idx=0,
    paged_attention_config=paged_attention_config,
    transformation_mats=transformation_mats,
    weight_dtype=ttnn.bfloat8_b,
)

# Forward: decode ([1, 1, batch, hidden]) or prefill ([1, 1, batch * seq_len, hidden])
output = attention(hidden_states, rope_mats, position_idx, page_table, is_decode=True)
```

## Decode levers (phase 3e, lane B1; both OFF by default)

| knob | env | effect | numerics |
|---|---|---|---|
| `SolarOpenAttentionProgramConfig.fused_qk` | `SOLAR_OPEN_ATTENTION_FUSED_QK=1` | `decode.py::decode_qkv_heads`: `nlp_create_qkv_heads_decode(overlap_qk_coregrid=False)` onto the 2B-core grid of `ProgramConfig.get_decode_qk_fused_grids` (Q / V of user b on core b, K on core B + b), then ONE `rotary_embedding_llama_fused_qk` and ONE `paged_fused_update_cache` instead of rope q + rope k + update k + update v; the two `to_memory_config(kv_mem_cfg)` no-ops are dropped. `tt/model.py::create_rope_setup` reads the same knob and builds `RotarySetup(use_qk_fused=True)` (2B cos/sin rows and trans_mat tiles on that grid; `get_tt_pos_idx` repeats the positions for the K half). TP > 1 only. | bit-identical expected (same per-core rotary kernel body and default compute config, same KV repack); gate `tests/unit/test_attention_fused_qk.py` (`torch.equal` on Q, K / V pages and the output, all replicas, B = 1 / 8 / 16 / 32). B = 16 moves the SDPA grid from the device grid to 8x8 (2B = 32 is a tile multiple), B = 1 / 32 keep theirs |
| `SolarOpenAttentionProgramConfig.decode_out_cores` (+ `decode_out_interleave_in0`, default True) | `SOLAR_OPEN_ATTENTION_OUT_GRID=8x8` | `decode.py::decode_output_projection`: `sharded_to_interleaved` of the concat_heads output FIRST, then o_proj as the explicit (8, 8) 1D mcast config (per_core_N 2, in0_block_w 4, out_subblock_w 2) with the HiFi2 compute config restated (`get_decode_out_compute_config`; an explicit program config alone drops the bf16 x bfp8 matmul to LoFi) straight into the L1-interleaved partial the reshape / all-reduce consume. `decode_out_interleave_in0=False` = the phase-2 sharded-in0 arm (no gain) | NOT bit-identical (K-block order): fp32-reference PCC in `tests/perf/test_config_candidates.py` (`o_proj_8x8_interleaved`), teacher-forced floors on the model |

Placement contract of the fused chain (every consumer computes its core instead of reading the shard spec):
RotarySetup's `get_batch_grid` on the doubled batch (8x8 when 2B % 32 == 0, the device compute grid otherwise,
row-major) == the create-heads output grid; the op derives Q = the first B cores and K = the B cores after them
(`compute_output_specs`: K starts at the last core of the (B + 1)-core prefix); the paged SDPA reducer of user b is
`(b % grid.x, b // grid.x)` of `get_decode_user_grid(..., fused_qk=True)`'s grid; `paged_fused_update_cache` pairs
K core i with V core i in row-major order of each grid. `get_decode_qk_fused_grids` asserts all of it at construction
(`Model.__init__`) and `tests/unit/test_attention_fused_qk_config.py` checks every B in 1..32 on the 11x10 / 13x10 /
8x8 grids without a device.

## Customization

Subclass `ProgramConfig` for different models or matmul configs (`None` cores = ttnn auto program config):

```python
# models/demos/your_model/tt/attention_configs.py
from dataclasses import dataclass

from models.demos.solar_open.tt.attention.config import ProgramConfig


@dataclass
class YourModelAttentionProgramConfig(ProgramConfig):
    # SDPA configs
    decode_k_chunk_size: int = 256
    prefill_q_chunk_size_large: int = 512

    # Matmul program configs (optional); n // 32 must be divisible by the core count
    decode_qkv_cores: tuple[int, int] | None = (8, 5)
    decode_qkv_in0_block_w: int = 4

    decode_out_cores: tuple[int, int] | None = (8, 8)
    decode_out_in0_block_w: int = 4
```
