# Motif-3 on a Blackhole Galaxy (`models/demos/motif3`)

Port of [Motif-Technologies/Motif-3](https://huggingface.co/Motif-Technologies/Motif-3) (314B MoE, revision
`2ed2ed5c`) to one Blackhole Galaxy (32 chips, 8x4 torus) with tt-metal / ttnn, served by vLLM 0.26.0 +
vllm-tt-plugin under tt-inference-server at batch 32.

Authoritative design: `/home/ttuser/hchang/experiments/motif-3/docs/study/00_feasibility_and_design.md` (cited
below as "design §x"); details in study reports `01_motif_reference.md` ... `09_memory_perf_plan.md` ("study NN").

## Model in one paragraph

53 decoder layers. Every layer: two mHC sites over a 4-stream residual (4 x 4096, 20-iteration Sinkhorn), GDLA
attention (MLA-style latent 512 + rope 64; 80 q heads = 16 groups x (4 signal + 1 noise), one KV head per group,
elementwise sigmoid gate, per-signal-head sigmoid lambda), then an FFN. Layers `l % 4 == 0` (14) are global (full
causal, YaRN RoPE, softmax scale 0.14467963); the other 39 are sliding-window (129 keys incl. the current one,
plain RoPE theta 1e4, scale 0.07216878). Layers 0-1 have a dense PolyNorm MLP (12288); layers 2-52 a MoE (384
experts, sigmoid top-8 with selection-only expert bias, renorm x2.0, one shared expert, per-expert PolyNorm,
intermediate 1280). Final: mean over the 4 streams, RMSNorm, LM head (vocab 220160).

## Directory

```
motif3/
  README.md            this file (CONVENTIONS below are binding for every module)
  reference/           CPU golden (pure torch; never imports ttnn). MotifArgs, modules (GDLAttention, MHCLayer,
                       MoE, ...), weights (MotifCheckpoint, random_state_dict), rope, cache.
  tt/
    model_config.py    MotifTTConfig: axes, per-chip heads/experts, dtypes, layer schedule, KV pool, buckets,
                       compute configs, trace size, cache paths; device_params() for the pytest fixture
    ccl.py             MotifCCL: role-named all_gather / reduce_scatter / all_reduce / partition; readback helpers
    weights.py         HFWeightLoader / DictWeightSource, torch-side transforms, as_tensor + TT weight cache
    rope.py            YaRN / plain tables, MotifRope (decode per-lane gather, prefill tables, composite / HF apply)
    generator_api.py   bridge <-> runtime contract (owned by the vLLM-bridge work)
    generator_vllm.py  MotifForCausalLM (vLLM bridge; device-free import)
    <modules>          embedding, mhc, attention, polynorm, mlp, moe, decoder, model, generator (next wave)
  tests/
    unit/test_infra_*.py   shared-infra tests (CPU: config, rope, weights, import; device: test_infra_device.py)
    unit/gates/            device op gates G0-G8 (results in unit/gates/results, summary GATES_RESULTS.md)
```

## Running tests

**CPU tests must not touch the chips.** tt-metal's root `conftest.py` opens the UMD cluster even for
`pytest --collect-only`, so run CPU tests with `--noconftest` inside a namespace where `/dev/tenstorrent` is an
empty tmpfs (works unprivileged on this host):

```bash
cd /home/ttuser/hchang/experiments/motif-3/tt-metal
unshare -Urm --propagation private bash -c 'mount -t tmpfs none /dev/tenstorrent && \
  source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD && \
  python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib -q \
    models/demos/motif3/tests/unit/test_infra_config.py models/demos/motif3/tests/unit/test_infra_rope.py \
    models/demos/motif3/tests/unit/test_infra_weights.py models/demos/motif3/tests/unit/test_infra_import.py'
```

CPU tests may import `ttnn` (the import itself is device-free) but must **not create ttnn tensors**: even a
host-only `ttnn.from_torch(..., layout=TILE)` initializes the Metal context, which opens the UMD cluster (inside
the namespace above it fails with "No chips detected" instead of touching the chips). Keep CPU tests pure torch.

**Device tests run only through the lock wrapper** (exclusive flock on the 32 chips, env, logs in `logs/dev/`).
Keep runs short; never SIGKILL; after a hang/timeout run `scripts/devreset.sh` once, then retry:

```bash
/home/ttuser/hchang/experiments/motif-3/scripts/devrun.sh -t 1200 -n infra_device -- \
  pytest models/demos/motif3/tests/unit/test_infra_device.py
```

Device tests use tt-metal's fixtures with `device_params()` from `tt/model_config.py`:

```python
from models.demos.motif3.tt.model_config import device_params
@pytest.mark.parametrize("mesh_device, device_params",
                         [((4, 8), device_params())],      # FABRIC_2D_TORUS_XY, trace_region_size 256 MiB
                         indirect=True)
def test_x(mesh_device): ...
```

`device_params(fabric="FABRIC_1D_RING", trace_region_size=..., l1_small_size=...)` gives the fallback fabric or
other keys the root conftest understands (`fabric_config`, `trace_region_size`, `l1_small_size`,
`worker_l1_size`, `num_command_queues`, `dispatch_core_axis`, `reliability_mode`, `fabric_tensix_config`).

---

# CONVENTIONS

Every `tt/` module and test follows these rules. When a module needs something not covered here, extend the
shared infra (or ask its owner) instead of re-deriving it locally.

## 1. Mesh, axes, chips

* Logical mesh **(4, 8)** (`MESH_DEVICE="(4, 8)"`, rotated onto the (8, 4) torus by the runtime). An **(8, 4)**
  mesh (the plugin's `BH-Galaxy` preset) is also supported: `MeshAxes.detect` makes the **size-8 dim the TP
  axis** and the other the DP axis.
* Roles (design §3.1):
  * **TP axis** ("cols" in the (4, 8) view, 8 chips, index `tp` = 0..7): attention heads, dense/shared
    intermediates and LM-head vocab are split over it; every TP matmul is closed by `all_reduce` over it.
  * **DP axis** ("rows", 4 chips, index `dp` = 0..3): 4 DP groups of 8 decode lanes; the MoE gather / combine
    runs over it.
* **Never hard-code `cluster_axis` 0/1.** Use `cfg.axes.tp_axis` / `cfg.axes.dp_axis`, or pass role names to
  `MotifCCL` (`"tp"`/`"cols"`, `"dp"`/`"rows"`). Mesh coordinate of a chip: `cfg.axes.coord(dp, tp)`; inverse
  `cfg.axes.roles(row, col)`.
* **Chip linear order** (expert placement): `k = dp * 8 + tp` (`cfg.axes.chip_index`), independent of mesh
  orientation.
* Fabric `FABRIC_2D_TORUS_XY` (default; `MOTIF3_FABRIC=FABRIC_1D_RING` is the named fallback, kill criterion G4).
  Compute grid 12 x 10 = 120 cores per chip (1x harvested), read from the device into `cfg.compute_grid`.

## 2. Lanes (decode) and users (prefill)

* 32 physical decode lanes. **Lane `l` lives on DP row `l // 8`** (local lane `l % 8`); every TP chip of that
  row processes the same 8 lanes. A request keeps its lane for its whole life (its KV is only written on its
  row in decode); the bridge maps vLLM rows/slots to lanes (`generator_vllm.LaneMap`, design §2.3.10).
* Per-lane host inputs are laid out row-wise with `rope.lanes_to_rows(values[32, ...], cfg, pad_to=...)` ->
  `[4, n, ...]` and uploaded with `rope.shard_lanes(rows, cfg, mesh, dtype=..., device=None|mesh)` (row `r` of the
  DP axis receives `rows[r]`, replicated over TP). `device=None` gives the host tensor for
  `ttnn.copy_host_to_device_tensor` into a persistent trace input.
* Inactive lanes: position `-1` (paged_update_cache / FlashMLA skip it), rot index 0, outputs masked with
  `ttnn.where(active, o, 0)` (design §2.3.4 step 12).
* Prefill: one user per call, padded to a bucket `S` in `cfg.prefill_buckets` (128 ... 32768, powers of 2; all
  warmed before decode capture), **replicated on all 32 chips** (design §3.3).

## 3. Tensor layouts at module boundaries

Logical shapes per chip; TILE layout pads the row dim to 32. "Replicated in the row" means bitwise identical on
the 8 TP chips of a DP row (the routing-consistency invariant, design §2.3.7) -- tests check it with
`ccl.replicas_identical(t, mesh, "tp")`.

**Decode** (L = 8 lanes of this chip's DP row, in local lane order):

| Tensor | Shape | dtype | Layout / memory | Placement |
|---|---|---|---|---|
| residual streams X | `[1, 4, 8, 4096]` (dim 1 = stream `i`) | bf16 | TILE, DRAM interleaved | replicated in the row; rows differ |
| sublayer input (mHC pre output `x_red`) and normed input | `[1, 1, 8, 4096]` | bf16 | TILE, DRAM | replicated in the row |
| sublayer output (attention `wo`, dense MLP, MoE) after `all_reduce(tp)` | `[1, 1, 8, 4096]` | bf16 | TILE, DRAM | replicated in the row |
| MoE gathered tokens (inside MoE) | `[1, 1, 32, 4096]` (lane order `8 dp + l`) | bf16 | TILE | replicated on all chips |
| `cur_pos` / `update_idxs` | `[8]` | int32 | ROW_MAJOR | per row |
| RoPE indices | `[1, 32]` (8 lanes + 24 pad) | uint32 | ROW_MAJOR | per row (`MotifRope.rot_idxs_host`) |
| page table | `[8, 512]` (`W = 32768 / 64`) | int32 | ROW_MAJOR | per row |
| KV cache, per layer | `[4128, 1, 64, 576]` | bfp8 | TILE, DRAM | one copy per chip (all chips) |
| logits | `[1, 1, 8, 27520]` | bf16 | TILE | chip (dp, tp) = lanes of row dp x vocab block tp |

**Prefill** (one user, S = bucket; identical on all 32 chips unless noted):

| Tensor | Shape | dtype | Layout |
|---|---|---|---|
| residual streams X | `[1, 4, S, 4096]` | bf16 | TILE, DRAM |
| sublayer input / output | `[1, 1, S, 4096]` | bf16 | TILE, DRAM |
| RoPE cos / sin | `[1, 1, S, 64]` | bf16 | TILE (`MotifRope.prefill_cos_sin(kind, S)`) |
| page table | `[1, 512]` | int32 | ROW_MAJOR |
| MoE partial after RS(dp) (inside MoE) | `[1, 1, S/4, 4096]` | bf16 | TILE (row slice `dp`) |
| logits | last real token's tile row `[1, 1, 32, 27520]` per chip | bf16 | TILE |

Stream layout vs the reference: TT keeps streams **stream-major** `[1, 4, T, 4096]`; the reference uses
`[B, S, 4, 4096]` -> convert with `x.permute(0, 2, 1, 3)`.

## 4. Dtype policy (`cfg.dtypes`, design §1.5)

| Class | Device dtype | Math |
|---|---|---|
| routed experts (gate_up, down) | bfp8 | HiFi2, fp32 dest acc (`cfg.compute_config("experts")`); PolyNorm output stays **bf16** (never block-float, study 01 N3) |
| shared expert, dense MLP | bfp8 | HiFi2, fp32 acc |
| attention (W_lat, wq_b, wq_b_gate, W_UK', W_UV', wo) | bf16 | HiFi4, fp32 acc |
| router weights | bf16 (bias fp32) | HiFi4, fp32 acc, **fp32 logits / sigmoid / bias / top-k** |
| mHC fused projection | bf16 (gamma folded in fp32, rounded once) | HiFi4, fp32 acc; mixes, Sinkhorn, coefficients fp32 |
| PolyNorm constants | fp32 | moments in fp32 |
| norms (gamma) | bf16 | HiFi4, fp32 acc |
| LM head / embedding | bf16 / bf16 | HiFi2, fp32 acc |
| KV cache (latent: unit-RMS `n` 512 + roped `k_pe` 64) | bfp8 | gamma_kv folded into W_UK'/W_UV' (cache holds the unit-RMS latent) |
| activations, residual streams, RoPE tables | bf16 | - |

Compute-kernel configs: `cfg.compute_config(role)` with roles `attn_latent, attn_heads, sdpa, rope, norm, mhc,
router, polynorm, experts, shared, dense_mlp, lm_head, eltwise` (math approx mode off everywhere). Gates may
retune a role in `model_config.COMPUTE_ROLES`; do not build ad-hoc configs in modules.

## 5. Per-chip partitioning (design §2.3.4-2.3.8)

| What | Chip `tp` (TP) / chip `(dp, tp)` (EP) holds |
|---|---|
| q heads | `[10tp, 10tp+10)` = KV groups `{2tp, 2tp+1}` (`cfg.chip_heads(tp)`) |
| signal heads (`s = 4g + j`: lambda, gate, `wo` inputs) | `[8tp, 8tp+8)`; local index `s_loc = 4 g_loc + j` |
| local q head order | `h_loc = 5 g_loc + j` (j = 4 is the group's noise head) |
| dense MLP / shared expert intermediate | `[1536tp, +1536)` / `[160tp, +160)` |
| routed experts | `[12k, 12k+12)`, `k = 8dp + tp` (`cfg.experts_of_chip(dp, tp)`) |
| LM-head vocab | `[27520tp, +27520)` (replicated over DP) |
| everything else (W_lat, router, mHC, norms, embedding) | replicated |

## 6. Weights (`tt/weights.py`)

* **Sources**: `HFWeightLoader()` (lazy safetensors over `model.safetensors.index.json` in `cfg.weights_dir`;
  `get(name)`, `get_rows(name, a, b)` for expert slices, `layer_available(l)`; partially downloaded shards are
  never opened) or `DictWeightSource({hf_name: tensor})` for random weights. Both expose the same API; modules
  take a `source` and never call safetensors directly. Names are HF checkpoint names (`weights.hf_name(l, ...)`).
  Reference-named random weights: `reference_to_hf_state_dict(random_state_dict(args, layers))`.
* **Transforms** (host, fp32 -- fp64 inputs stay fp64 -- rounded once at upload). They return the device
  orientation `[in, out]` (`y = x @ W`). Per-chip transforms take `tp` and are concatenated with
  `stack_tp(fn, cfg, dim)` so that a TP shard on `dim` gives chip `tp` exactly `fn(tp)`:
  * attention: `latent_projection_for_chip` (`[4096, 1664]` = `[cq 1024 | c_raw 512 | k_pe 64 | lambda 64]`, the
    chip's 8 lambda columns first at `[1600, 1608)`), `wq_b_for_chip(layout="split")` (`[1024, 1920]`:
    `[q_nope 10x128 | q_pe 10x64]`), `wq_b_gate_for_chip` (`[1024, 1024]`), `absorb_weights_for_chip`
    (`[2, 128, 512]`, `q_lat = q_nope @ W_UK'`), `unabsorb_weights_for_chip` (`[2, 512, 128]`),
    `prefill_kv_expansion_for_chip` (`[512, 640]`: per group `[k_nope | v | 0_64]`), `wo_for_chip`
    (`[1024, 4096]`); optional `absorb_weights_per_head_for_chip`, `absorbed_q_weights_per_head_for_chip`.
  * mHC: `mhc_fused_projection` (`fn [24, 16384]`, gamma folded) -> `mhc_projection_blocks` (`[1, 4, 4096, 32]`),
    `mhc_scalars` (alphas, biases fp32; `bias_res` row-major `4i + j`).
  * MLP / shared: `mlp_gate_up_for_chip` (`[4096, 2 I/8]`, gate first), `mlp_down_for_chip` (x0.5 folded),
    `polynorm_coefficients` (sigmoid(w); bias clamp only for routed).
  * MoE: `experts_gate_up` / `experts_down` (x0.5 folded), `ep_layout` (`[384, ...] -> [4, 96, ...]`),
    `local_expert_ids`, `expert_polynorm_tensors` (`c0, c1, c2, b` as `[4, 96, 1, 1]`), `router_weights`
    (optional pad to 512 with bias -1e9).
  * `lm_head_for_chip` (`[4096, 27520]`), `norm_weight` (`[1, 1, dim/32, 32]`, upload ROW_MAJOR).
* **Upload**: `weights.as_tensor(src, mesh_device=, cfg=, dtype=, layout=TILE, dp_dim=None, tp_dim=None,
  cache_name="attn.wq_b", layer=l)`. `src` may be a zero-arg callable (not called on a cache hit). Mapping by
  role: both `None` = replicate (cached as one unsharded copy); `tp_dim` = shard over TP, replicate over DP;
  `dp_dim` = shard over DP; both = 2D; EP = `ep_layout(t)` + `dp_dim=0, tp_dim=1` -> each chip `[1, 12, ...]`.
  `weights.shard_for_device(t, cfg.axes, row, col, dp_dim=, tp_dim=)` emulates the mapper on the host.
* **Cache**: `<TT_CACHE_PATH>/<version-tag>/mesh<R>x<C>/{L<nn>|global}/<name>__<mapping>_dtype_<D>_layout_<L>.tensorbin`
  with version tag `motif3-<rev8>-c<CACHE_FORMAT_VERSION>-<dtype tag>` (default root
  `/home/ttuser/hchang/experiments/motif-3/tt_cache`). Name tensors `"<module>.<tensor>"` (`attn.wq_b`,
  `mhc_attn.blocks`, `moe.experts.gate_up`, ...); `layer=None` for embedding / final norm / LM head. **Bump
  `model_config.CACHE_FORMAT_VERSION` whenever a transform or layout changes.** Random weights: `cache_name=None`.
  Cached files reload in the memory config they were built with (DRAM interleaved); re-shard after loading if
  needed. The converter marks a finished layer with `weights.mark_layer_cached(cfg, l, names)`.

## 7. Collectives (`tt/ccl.py`)

`ccl = MotifCCL(mesh_device, cfg)`; `ccl.all_gather(x, dim, "dp")`, `ccl.reduce_scatter(x, dim, "dp")`,
`ccl.all_reduce(x, "tp")`, `ccl.partition(x, dim, "dp")` (per-device slice, no fabric) and shortcuts `ar_tp`,
`ar_dp`, `ag_dp`, `rs_dp`, `ag_tp`. Generic semaphore-free ops only (GPT-OSS BH precedent): no persistent
buffers / semaphores needed, trace-safe once compiled; `num_links` / topology come from the fabric unless set.
Draft-1 CCL schedule (design §3.2): attention `wo` AR(tp); dense MLP AR(tp) (+ PolyNorm moments AR(tp)); MoE
decode AG(dp) -> local experts -> AR(dp) -> `partition(dim=2, "dp")` -> + shared (moments AR(tp)) -> AR(tp); MoE
prefill RS(dp) -> AR(tp) -> AG(dp). All-reduces are RS+AG (or AG+local sum), so replicas stay bitwise identical.

Measured constraint: `ttnn.mesh_partition` is a `slice` underneath and rejects TILE slices of the last two dims
that are not tile-aligned (8 of 32 rows -> TT_FATAL "Can only slice tilized tensor with height begin index aligned
to tiles"). `MotifCCL.partition` therefore round-trips such slices through ROW_MAJOR (2 extra layout ops). On the
decode hot path prefer keeping each DP group's lanes tile-aligned (pad the group's 8 lanes to a full 32-row tile
before the gather, so the gather gives `[1, 1, 128, 4096]` and the partition is whole tiles) -- the MoE module
chooses; both work.

Measured precision / cost (see `tt/ccl.py` docstring): bf16 all-reduces round like bf16; fp32 all-reduces on the
RS+AG path are TF32-class (~1e-3 relative; enough when the result is cast to bf16 afterwards), small fp32
payloads (PolyNorm moments) are exact; an 8-row all_gather is ~4x cheaper eager in ROW_MAJOR than in TILE.

## 8. RoPE (`tt/rope.py`)

`rope = MotifRope(mesh_device, cfg)` builds both tables once (`"yarn"` for global layers, `"plain"` for SWA;
`cfg.layer(l).rope_kind`). Half-split (HF/NeoX) convention on the checkpoint as stored -- **no weight permutation**.
* Decode: per step `rot_idxs` (`[1, 32]` uint32 per row) -> `rope.decode_cos_sin(kind, rot_idxs, layout=...)`
  (`ttnn.embedding` gather, trace-safe): `"rows"` -> `[1, 1, 32, 64]` (lanes on tile rows: for `x [1, H, 32, 64]`
  / `k_pe [1, 1, 32, 64]`), `"batch"` -> `[1, 8, 1, 64]` (for `x [1, 8, H, 64]`), `"batch_sharded"` -> the same
  HEIGHT_SHARDED one lane per core (for `rotary_embedding_hf(is_decode_mode=True)` with `x` sharded by
  `rope.batch_sharded_memory_config()`).
* Prefill: `rope.prefill_cos_sin(kind, S)` -> `[1, 1, S, 64]` TILE (positions `0..S-1`, cached per bucket).
* Apply: `rope.apply_hf(x, cos, sin, is_decode_mode=...)` (HF kernel) or `rope.apply_composite(x, cos, sin)`
  (`x cos + (x @ R) sin`, fp32 math). Gate G8 (`tests/unit/gates/test_g8_rope.py`, summary in
  `tests/unit/gates/GATES_RESULTS.md`) decides which path attention uses; both stay available here.

## 9. Module interface pattern

```python
class MotifAttention:                                   # likewise MHCSite, PolyNormMLP, MotifMoE, ...
    def __init__(self, mesh_device, cfg: MotifTTConfig, layer_idx: int, *,
                 source,                                # HFWeightLoader | DictWeightSource
                 ccl: MotifCCL, rope: MotifRope | None = None,
                 cache: bool = True):                   # False for random weights (no files written)
        self.spec = cfg.layer(layer_idx)                # window, softmax_scale, rope_kind, is_moe, ...
        self.wq_b = weights.as_tensor(lambda: weights.stack_tp(lambda tp: weights.wq_b_for_chip(
                        source.get(weights.hf_name(layer_idx, "self_attn.wq_b.weight")), cfg, tp), cfg, dim=1),
                    mesh_device=mesh_device, cfg=cfg, dtype=cfg.dtypes.attention, tp_dim=1,
                    cache_name="attn.wq_b" if cache else None, layer=layer_idx)
    def forward_decode(self, x, *, rot, cur_pos, page_table, kv_cache, ...) -> ttnn.Tensor: ...
    def forward_prefill(self, x, *, rot, page_table, kv_cache, seq_len, ...) -> ttnn.Tensor: ...
```

Rules:
1. Constructor: all weights via `weights.as_tensor` (lazy sources), dtypes from `cfg.dtypes`, compute configs
   from `cfg.compute_config(role)`, per-layer constants from `cfg.layer(l)`. No device work in forward except ops.
2. `forward_decode` is **trace-safe**: no host round trips, no `from_torch`, no data-dependent Python control
   flow, no new program shapes after warmup; per-step inputs (positions, rot indices, page table) arrive as
   persistent device tensors. Integer slice starts must be per-layer constants.
3. Inputs/outputs follow section 3; modules do not deallocate their inputs unless the docstring says
   "consumes"; outputs are new tensors in DRAM interleaved unless documented.
4. Collectives only through `MotifCCL` with role names; mesh-dependent indices only via `cfg.axes`.
5. Each module documents which design section it implements and keeps a fallback (composite) path where the
   design names one.

## 10. Tests and comparison against the reference

* Golden = `models/demos/motif3/reference` (import it **lazily inside tests**; its API may still change; write a
  local torch golden if something is missing and say so). CPU-side equivalences of the transforms are in
  `tests/unit/test_infra_weights.py` (absorbed == expanded GDLA at real dims, folded mHC projection, TP PolyNorm).
* Random-weight module tests: build HF-named random weights (or the reference's `random_state_dict` +
  `reference_to_hf_state_dict`), wrap in `DictWeightSource`, construct the TT module with `cache=False`, run the
  matching reference module on CPU, read back with `ccl.device_tensors_to_torch(t, mesh)` (`[R, C, ...]`, entry
  `[r, c]` = mesh coordinate `(r, c)`), compare per chip. Decode tests must give each DP row different lanes /
  positions and compare row `dp` against reference lanes `8dp .. 8dp+7`.
* Thresholds (design §4.3): RMSNorm >= 0.9999, PolyNorm >= 0.9995, GDLA >= 0.999, MoE >= 0.995 (bfp8), decoder
  layer >= 0.995, H_res max-abs <= 5e-3, router expert-set agreement >= 99.9 %. Always also assert
  `ccl.replicas_identical(out, mesh, "tp")` for row-replicated outputs. PCC helper:
  `models.common.utility_functions.comp_pcc`.
* Real-weight tests: `HFWeightLoader().layer_available(l)` (layers arrive over time; `.download_state.json` lists
  complete layers); skip, never download, if missing. Large artifacts only under
  `/home/ttuser/hchang/experiments/motif-3/tt_cache`.

## 11. Import rule

Nothing under `models/demos/motif3` imports another `models/demos/**` package at module import time (vLLM imports
the bridge before the mesh opens; demo imports can open the cluster). Copy small helpers, or import lazily inside
functions. `tests/unit/test_infra_import.py` checks the shared infra in a fresh interpreter; new modules should
add themselves to its list.

## 12. Environment

| Variable | Meaning (default) |
|---|---|
| `MOTIF3_WEIGHTS_DIR` / `HF_MODEL` | checkpoint dir (`/home/ttuser/hchang/experiments/motif-3/weights/Motif-3`) |
| `TT_CACHE_PATH` | TT weight-cache root (`/home/ttuser/hchang/experiments/motif-3/tt_cache`) |
| `TT_MODEL_WEIGHTS_REVISION` | checkpoint revision in the cache tag (`2ed2ed5c...`) |
| `MOTIF3_NUM_LAYERS` | truncated bring-up runs (53) |
| `MOTIF3_KV_POOL_TOKENS` | KV pool tokens before per-lane headroom (262144 -> 4128 blocks of 64) |
| `MOTIF3_MAX_MODEL_LEN` | max context (32768) |
| `MOTIF3_TRACE_REGION_SIZE` | trace region bytes (268435456) |
| `MOTIF3_FABRIC` | `FABRIC_2D_TORUS_XY` (fallback `FABRIC_1D_RING`) |
| `MESH_DEVICE` | `"(4, 8)"` for serving (`(8, 4)` also works) |

## Status

* Shared infra (this README, `tt/model_config.py`, `tt/ccl.py`, `tt/weights.py`, `tt/rope.py`, `test_infra_*`):
  36 CPU tests pass (absorbed == expanded GDLA to 1e-10 in fp64 at real dims, also on real layer-0/1 weights;
  folded mHC projection; TP PolyNorm; shards; EP placement; loader). Device smoke test `test_infra_device.py`
  (5 tests, ~20 s, 2026-10-01, `logs/dev/20261001_174321_infra_device4.log`) passes on 4x8 with
  FABRIC_2D_TORUS_XY and FABRIC_1D_RING and on 8x4: every draft-1 CCL payload (eager, replicas bitwise
  identical), the decode chain AR(tp) -> AG(dp) -> AR(dp) -> partition(dp) captured in a trace and replayed with new
  inputs (~150-170 us per replay), role mappers + cache reload, RoPE gathers and every apply path (composite
  rows/batch, HF prefill mode, HF decode mode height-sharded; PCC >= 0.999996).
  Measured while the fabric reported only a TORUS_Y grouping (X wrap links down after resets), so latencies may
  change on a healthy torus. Measured facts that matter downstream are in section 7 and the `tt/ccl.py` docstring
  (fp32 CCL reduction is TF32-class; TILE sub-tile partition needs ROW_MAJOR; 8-row TILE gathers are composite).
* Gates G0-G8: `tests/unit/gates/` (see `GATES_RESULTS.md` when present).
* Modules, decoder, model, generator, vLLM bridge, TIS integration: in progress (design §8 task list).
