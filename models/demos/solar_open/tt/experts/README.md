# MoE Experts Module

Bias-free GLU MoE experts (the routed experts of Solar-Open-100B) with clean decode/prefill separation.

## Structure

```
experts/
├── __init__.py      # Main Experts class with auto-dispatch (+ shared_expert hook)
├── config.py        # ExpertConfig + ProgramConfig base (sparse-matmul grid builder)
├── weights.py       # HF fused layout -> per-device [gate_d | up_d] / down shards, tensor cache
├── operations.py    # Core operations (apply_glu, routing weights, reductions, CCLs)
├── decode.py        # Decode forward (1..32 users; union-of-experts sparse_matmul)
└── prefill.py       # Prefill forward (dense bmm / expert-sorted hot-cold / per-expert paths)
```

## Usage

```python
from models.demos.solar_open.tt.experts import Experts, ExpertConfig
from models.demos.solar_open.tt.expert_configs import solar_open_program_config

# Solar-Open-100B: 128 routed experts, top-8, moe_intermediate_size 1280 (NOT intermediate_size), silu GLU
config = ExpertConfig(
    intermediate_size=1280,
    num_experts=128,
    hidden_size=4096,
    num_experts_per_tok=8,
    activation="silu",
)

# Create experts with model-specific program config
experts = Experts(
    mesh_device=mesh_device,
    config=config,
    state_dict=state_dict,        # {"gate_up_proj": [E, 2I, H], "down_proj": [E, H, I]} or {} (cache only)
    ccl_manager=ccl_manager,
    mesh_config=mesh_config,
    program_config=solar_open_program_config(mesh_device),
    weight_dtype=ttnn.bfloat8_b,  # bfloat4_b halves the footprint (SOLAR_OPEN_EXPERT_DTYPE=bfp4 upstream)
)

# Forward: dense [tokens, E] routing weights (0 for unselected experts), optional shared-expert callable
output = experts(hidden_states, routing_weights, is_decode=True, shared_expert=shared_expert)
```

## Contract

* `hidden_states` is `[1, 1, tokens, hidden]`, consumed. Decode: `tokens` = users (1..32, padded to a 32-row
  tile internally). Prefill: `tokens` = seq_len, a multiple of 32, processed in `sequence_chunk_size` chunks.
* `routing_weights` is the DENSE `[tokens, num_experts]` bf16 TILE tensor from the router: the normalised routing
  weight for the selected experts, 0 elsewhere (the union-of-experts decode mask relies on weights `>= 0`).
* `shared_expert(x)` (optional) is called once per decode call on the padded input and once per prefill chunk,
  before the input is deallocated. It must return this device's PARTIAL `[1, 1, rows, hidden]` bf16 tensor; the
  experts add it in place to their routed partial so the single TP `all_reduce` completes routed + shared.
* Output: `[1, 1, tokens, hidden]` bfloat8_b, all-reduced over TP.

## Weight layout

`weights.prepare_expert_weights_torch(state_dict, config, tp)` (pure torch, unit-tested on host) turns the
transformers >= 5 fused layout (`gate_up_proj [E, 2I, H]`, gate rows first; `down_proj [E, H, I]`) into
`gate_up_proj [1, E, H, tp * 2 * Ip]` whose column-parallel shard `d` is `[gate_d | up_d]` and
`down_proj [1, E, I, H]` whose row-parallel shard `d` holds intermediate rows `d*I/tp:(d+1)*I/tp`. Any other
input shape raises `ValueError`. Cache stems: `gate_up_proj_fused_tp{tp}`, `down_proj`. Per-expert weight
slices needed by the dense prefill paths are created on demand and freed after use (no persistent copies).

## Customization

Override `ProgramConfig` for different shapes (grids resolve to exact-fill rectangles automatically):

```python
# models/demos/your_model/tt/expert_configs.py
@dataclass
class YourModelProgramConfig(ProgramConfig):
    # Core grid sizes
    decode_gate_up_cores: tuple[int, int] = (5, 2)
    decode_down_cores: tuple[int, int] = (8, 4)
    decode_down_cores_batched: tuple[int, int] | None = (8, 8)

    # Sparse matmul parameters
    decode_gate_up_in0_block_w: int = 32
    decode_down_in0_block_w: int = 5

    # Chunking / dense prefill
    sequence_chunk_size: int = 4096
    base_down_split_size: int = 1024
    dense_grid_max_width: int = 12
    dense_bmm_max_tokens: int = 256
```
