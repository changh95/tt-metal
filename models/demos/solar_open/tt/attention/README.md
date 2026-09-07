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
├── operations.py    # Common ops (RoPE, head split/concat, projections, TP all-reduce)
├── decode.py        # Decode forward (seq_len=1)
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
