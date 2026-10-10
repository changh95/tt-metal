# Motif-3 on a Blackhole Galaxy (`models/demos/motif3`)

Port of [Motif-Technologies/Motif-3](https://huggingface.co/Motif-Technologies/Motif-3) (314B MoE, revision
`2ed2ed5c`) to one Blackhole Galaxy (32 chips, 8x4 torus) with tt-metal / ttnn, served by vLLM 0.26.0 +
vllm-tt-plugin under tt-inference-server at batch 32.

Authoritative design: `/home/ttuser/hchang/experiments/motif-3/docs/study/00_feasibility_and_design.md` (cited
below as "design §x"); details in study reports `01_motif_reference.md` ... `09_memory_perf_plan.md` ("study NN").
Wave-A review and the wave-B action list (IDs such as INFRA-3, ATTN-2): `docs/WAVE_A_REVIEW.md`. Wave-B1 module
results and requested shared changes: `docs/WAVE_B1_SUMMARY.json`. Device op gates: `tests/unit/gates/GATES_RESULTS.md`.

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
                       MoE, ...), weights (MotifCheckpoint, random_state_dict), rope, cache, golden_stream (C2 goldens)
  tt/
    model_config.py    MotifTTConfig: axes, per-chip heads/experts, dtypes, layer schedule, KV pool, buckets,
                       compute roles + program-config builders (gate- and module-validated), fabric, L1_SMALL, trace
                       size, cache paths; device_params() for the pytest fixture, open_motif_mesh() for standalone runs
    ccl.py             MotifCCL: role-named all_gather / reduce_scatter / all_reduce / ar_exact / partition /
                       ag_dp_rows with every semaphore in L1_SMALL; readback helpers; fabric_report / log_fabric
    weights.py         HFWeightLoader / DictWeightSource, torch-side transforms, as_tensor + TT weight cache
    rope.py            YaRN / plain tables, MotifRope (decode per-lane gather, prefill tables, HF / composite apply)
    generator_api.py   bridge <-> runtime contract (MotifGenerator ABC, GeneratorSettings, KV-pool sizing, weights
                       location, the serving "tt" config incl. L1_SMALL_SIZE; resumed prefill rows, the speculative
                       decode types, KV-R modes: §15-§17); torch-only, shared by bridge and config
    prefill_plan.py    resumed / chunked prefill planning (alignment, chunks, fill / SDPA / tail / RoPE tables,
                       writer-first row order, vLLM scheduler checks; §15) and the packed passes (§18); torch-only
    verify_plan.py     T64 verify planning: the 64-row step layout, trace choice per step, c* (§18); torch-only
    kv_write.py        DecodeKVWrite: decode KV writes (KV-R, the speculative split, the 64-row T64 writer; §16, §18)
    mtp.py             MotifMTP: the MTP layer (KV-only prefill fill, decode drafts; §17)
    sampling.py        MotifDeviceSampler: exact on-device sampling inside the decode trace
                       (docs/sampling/DEVICE_SAMPLER.md)
    generator_vllm.py  MotifForCausalLM (vLLM bridge; device-free import)
    embedding.py       MotifEmbedding (token gather -> 4 streams)        lm_head.py   MotifLMHead (mean, norm, head)
    mhc.py             MHCSite (pre / post of one mHC site)              attention.py MotifAttention (GDLA)
    polynorm.py        PolyNorm (scalar TP / grouped)                    mlp.py       PolyNormMLP (dense / shared)
    moe.py             MotifRouter, MotifMoE (EP32)
    kernels/           out-of-tree ttnn.generic_op kernels: sinkhorn_motif (exact mHC coefficients), router_fp32,
                       moe_polynorm (B3), router_topk (B4 fused router tail), shared_polynorm (B5),
                       moe_compact (B2b: moe_dispatch/ prefill row dispatch, moe_combine/ gather combine)
    decoder.py, model.py, generator.py   integration wave (WAVE_A_REVIEW §5.8 GEN-1..7)
  tests/
    unit/test_infra_*.py   shared-infra tests (CPU: config, rope, weights, import; device: test_infra_device.py,
                           test_infra_l1_small.py)
    unit/test_<module>.py  module tests (each owner's)
    unit/gates/            device op gates G0-G13a, G15 / G16 (results in unit/gates/results, summary GATES_RESULTS.md,
                           which also records the model-level P5 / T64 gates in §13.7)
    test_generator_vllm_host.py   vLLM bridge host suite (real vLLM + plugin, fake generator)
    test_resumed_prefill.py, test_spec_decode_device.py, test_pt_soak.py   model-level device tests (CP-*, G16 /
                           G-S5w / G-S6w-lite, CP9-P / G-X)
```

## Running tests

**Host-only tests must not touch the chips** (other agents share them through the device lock). tt-metal's root
`conftest.py` opens the UMD cluster even for `pytest --collect-only`, and `import vllm` activates the TT platform, so
run every host-only command through the wrapper, which hides `/dev/tenstorrent` behind an empty tmpfs in a private
user + mount namespace (`unshare -Urm --propagation private`, unprivileged on this host; a stray device open then
fails with "No chips detected"). It sets up the venv / `TT_METAL_HOME` / `PYTHONPATH` like `devrun.sh`, runs from the
tt-metal root and takes **no** device lock. Never run plain `pytest` / `python` for host work on this machine.

```bash
S=/home/ttuser/hchang/experiments/motif-3/scripts
# skeleton CPU suites (--noconftest: no root conftest)
$S/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib -q \
    models/demos/motif3/tests/unit/test_infra_config.py models/demos/motif3/tests/unit/test_infra_rope.py \
    models/demos/motif3/tests/unit/test_infra_weights.py models/demos/motif3/tests/unit/test_infra_import.py
# vLLM bridge host suite (root conftest active, devices hidden; -n NAME also logs to logs/host/)
$S/hostrun.sh -n bridge -- python -m pytest -p no:cacheprovider -q models/demos/motif3/tests/test_generator_vllm_host.py
# reference package
$S/hostrun.sh -- python -m pytest --noconftest -o addopts="" --import-mode=importlib models/demos/motif3/reference/tests
# CPU tests of a file that also has device tests: keep the root conftest (the indirect mesh_device / device_params
# parametrization cannot be collected with --noconftest), devices stay hidden
$S/hostrun.sh -- python -m pytest -p no:cacheprovider -q models/demos/motif3/tests/unit/test_moe.py -k "cpu or host"
```

CPU tests may import `ttnn` (the import itself is device-free) and build ttnn *config objects* (compute-kernel and
program configs, `ttnn.CoreCoord`, sharded memory configs, `ComputeConfigDescriptor`), but must **not create ttnn
tensors**: even a host-only `ttnn.from_torch(..., layout=TILE)` initializes the Metal context, which opens the UMD
cluster (inside the hidden namespace it fails with "No chips detected" instead of touching the chips); the same holds
for fabric / memory queries such as `ttnn.get_fabric_config()` or `ttnn.get_memory_view()`
(`model_config.active_fabric_name` / `mesh_l1_small_bytes` only ask with a real, opened `ttnn.MeshDevice`). Keep CPU
tests pure torch; bfp8 round trips belong in device tests.

**Device tests run only through the lock wrapper** (exclusive flock on the 32 chips, env, logs in `logs/dev/`).
Keep runs short and batch cases; never SIGKILL; after a hang/timeout run `scripts/devreset.sh` once, then retry:

```bash
/home/ttuser/hchang/experiments/motif-3/scripts/devrun.sh -t 1200 -n infra_device -- \
  python -m pytest models/demos/motif3/tests/unit/test_infra_device.py models/demos/motif3/tests/unit/test_infra_l1_small.py \
  -s -p no:cacheprovider
```

Device tests use tt-metal's fixtures with `device_params()` from `tt/model_config.py`, and **log the committed
fabric topology** first (WAVE_A_REVIEW §5; `grep "committed:"` in the log; the line also shows `l1_small=`):

```python
from models.demos.motif3.tt.ccl import log_fabric
from models.demos.motif3.tt.model_config import MotifTTConfig, device_params

@pytest.mark.parametrize("mesh_device, device_params",
                         [((4, 8), device_params())],   # FABRIC_2D_TORUS_XY, trace 256 MiB, l1_small_size 32768
                         indirect=True)
def test_x(mesh_device):
    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)   # fabric / grid / mesh / L1_SMALL from the device
    log_fabric(mesh_device, "test_x")  # [motif3.fabric] test_x committed: TORUS_Y requested: ... l1_small=32768 ...
```

**Every mesh is opened with an L1_SMALL region of 32768 B per core** (`device_params()` default,
`generator_api.L1_SMALL_SIZE`; attention wave-B1 P0). The CCL global semaphores live there (§8); without it they land
in main L1 next to whatever is live when their program is first built and later static circular buffers clash with
them ("Statically allocated circular buffers ... clash with L1 buffers": a global-layer prefill at S >= 1024 after any
decode step, the bf16-KV FlashMLA decode). `device_params(l1_small_size=0)` exists only to reproduce that hazard. vLLM:
`--additional-config '{"tt": {..., "l1_small_size": 32768}}'` (the bridge refuses a server without it, §14). Standalone
(no pytest, no vLLM): `mesh = model_config.open_motif_mesh()` / `close_motif_mesh(mesh)` (fabric, COL dispatch, trace
region and L1_SMALL as served). `device_params(fabric="FABRIC_1D_RING", trace_region_size=..., l1_small_size=...)`
gives the fallback fabric or other keys the root conftest understands (`fabric_config`, `trace_region_size`,
`l1_small_size`, `worker_l1_size`, `num_command_queues`, `dispatch_core_axis`, `reliability_mode`,
`fabric_tensix_config`). Trace captures must be exception-safe (always end and release the capture on error; a
dangling capture hung `close_mesh_device` once, GATES_RESULTS §11.6): see `_Capture` in `tests/unit/test_infra_device.py`.

---

# CONVENTIONS

Every `tt/` module and test follows these rules. When a module needs something not covered here, extend the
shared infra (or ask its owner) instead of re-deriving it locally.

## 1. Mesh, axes, chips, fabric, device memory

* Logical mesh **(4, 8)** (`MESH_DEVICE="(4, 8)"`, rotated onto the (8, 4) torus by the runtime). An **(8, 4)**
  mesh (the plugin's `BH-Galaxy` preset) is also supported: `MeshAxes.detect` makes the **size-8 dim the TP
  axis** and the other the DP axis.
* Roles (design §3.1):
  * **TP axis** ("cols" in the (4, 8) view, 8 chips, index `tp` = 0..7): attention heads, dense/shared
    intermediates are split over it; every TP matmul is closed by `all_reduce` over it.
  * **DP axis** ("rows", 4 chips, index `dp` = 0..3): 4 DP groups of 8 decode lanes; the MoE gather / combine
    runs over it.
* **Never hard-code `cluster_axis` 0/1.** Use `cfg.axes.tp_axis` / `cfg.axes.dp_axis`, or pass role names to
  `MotifCCL` (`"tp"`/`"cols"`, `"dp"`/`"rows"`). Mesh coordinate of a chip: `cfg.axes.coord(dp, tp)`; inverse
  `cfg.axes.roles(row, col)`.
* **Chip linear order** (expert placement): `k = dp * 8 + tp` (`cfg.axes.chip_index`), independent of mesh
  orientation. The LM-head vocab blocks use the *mesh* linear index `b = r * C + c` instead (§6).
* Fabric `FABRIC_2D_TORUS_XY` (default; `MOTIF3_FABRIC=FABRIC_1D_RING` is the named fallback, kill criterion G4).
  Built with a mesh, `cfg.fabric` is what the mesh was opened with (`ttnn.get_fabric_config()`, INFRA-6). The
  topology mapper may **commit less**: since 2026-10-01 17:31 every open commits `TORUS_Y` = the 8-chip TP axis a
  ring, the 4-chip **DP axis a line** (X wrap links down; full reset needs `sudo ipmitool`, decision D3;
  `ttnn.get_usable_topology`: TP `Torus`, DP `Mesh`). Results are correct; DP-axis CCL latencies (MoE gather /
  combine, prefill RS/AG) are line numbers. `ccl.fabric_report(mesh)` returns `{requested, committed,
  physical_shape, tp_ring, dp_ring, degraded, l1_small}`; `ccl.log_fabric(mesh, tag)` prints it as one `committed:`
  line.
* Device memory: **L1_SMALL 32768 B per core** (above; `cfg.l1_small_size` = what the mesh is opened with,
  `cfg.mesh_l1_small_size` = what the opened mesh has; `model_config.require_l1_small(mesh)` raises below 32 KiB).
  The CCL semaphores use 4.7 KiB of it for every draft-1 payload (`test_infra_l1_small.py`). Trace region 256 MiB.
* Compute grid 12 x 10 = 120 cores per chip (1x harvested), read from the device into `cfg.compute_grid`.

## 2. Lanes (decode) and users (prefill)

* 32 physical decode lanes. **Lane `l` lives on DP row `l // 8`** (local lane `l % 8`); every TP chip of that
  row processes the same 8 lanes. A request keeps its lane for its whole life (its KV is only written on its
  row in decode); the bridge maps vLLM rows/slots to lanes (`generator_vllm.LaneMap`, design §2.3.10).
* **`cfg.max_batch` is always 32** (the decode trace runs all 32 lanes, `lanes_per_row = max_batch // dp`), whatever
  vLLM's `--max-num-seqs` is; that value goes to `cfg.max_num_seqs` and only sizes the plugin's per-sequence block
  reservation (WAVE_A_REVIEW M3). `MotifTTConfig.from_settings` does this mapping.
* Per-lane host inputs are laid out row-wise with `rope.lanes_to_rows(values[32, ...], cfg, pad_to=...)` ->
  `[4, n, ...]` and uploaded with `rope.shard_lanes(rows, cfg, mesh, dtype=..., device=None|mesh)` (row `r` of the
  DP axis receives `rows[r]`, replicated over TP). `device=None` gives the host tensor for
  `ttnn.copy_host_to_device_tensor` into a persistent trace input.
* Inactive lanes: position `-1` (paged_update_cache / FlashMLA skip it), rot index 0, outputs masked with
  `ttnn.where(active, o, 0)` (design §2.3.4 step 12). Attention **requires** the per-step `active` mask (below).
* Prefill: one user per call, padded to a bucket `S` in `cfg.prefill_buckets` (128 ... 32768, powers of 2; all
  warmed before decode capture), **replicated on all 32 chips** (design §3.3). That is draft 1; serving now makes one
  `prefill_forward_batch` call per vLLM step with all its rows, in chunks of buckets up to the 8192 span cap (§15), and
  with packed prefill several short chunks share one pass of `T = B * S` rows (§18). `paged_fill_cache` gets exactly the
  bucket's first `cfg.prefill_page_table_entries(S)` page-table entries, so every bucket has one fixed program shape
  whatever the page-table width is (M10).
* **Per decode step, once, shared by all 53 layers** (attention wave B1): `rot = MotifAttention.decode_rope_tables(
  rope, rot_idxs)` (`{"yarn" | "plain": (cos, sin)}` `[1, 1, 32, 64]`, §9) and `active =
  MotifAttention.active_mask_from_cur_pos(cur_pos)` (`[1, 1, 8, 1024]` bf16 0/1). Both are trace-safe device ops on
  the persistent per-step inputs.

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
| mHC coefficients (`MHCCoeffs`, between `site.pre` and `site.post`) | `w_pre [1, 4, T, 1]`, `w_post [4, 5, T, 1]` (T = the 8 lanes, one tile) | fp32 | TILE, **L1** (`mix_l1`) | replicated in the row |
| MoE gathered tokens (inside MoE, `ccl.ag_dp_rows`) | `[1, 1, 32, 4096]` (lane order `8 dp + l`) | bf16 | TILE | replicated on all chips |
| shared-expert TP partial (`add_partial` of the MoE) | `[1, 1, 8, 4096]` | bf16 | TILE, DRAM | this chip's TP partial |
| `cur_pos` / `update_idxs` | `[8]` | int32 | ROW_MAJOR | per row |
| RoPE indices | `[1, 32]` (8 lanes + 24 pad) | uint32 | ROW_MAJOR | per row (`MotifRope.rot_idxs_host`) |
| RoPE tables (per step) | `{kind: (cos, sin)}` `[1, 1, 32, 64]` | bf16 | TILE | per row (`decode_rope_tables`) |
| active mask (per step) | `[1, 1, 8, 1024]` | bf16 0/1 | TILE | per row (`active_mask_from_cur_pos`) |
| page table | `[8, W]` (`W = cdiv(32768, 64)` = 512) | int32 | ROW_MAJOR | per row |
| KV cache, per layer | `[N, 1, 64, 576]`, **N from `allocate_kv_cache`** (4129 for the default flags) | bfp8 | TILE, DRAM | one copy per chip (all chips) |
| logits (`vocab_split="mesh"`, default, decision EMB-D1) | `[1, 1, 32, 6880]` (all 32 lanes, lane order `8 dp + l`) | bf16 | TILE (ROW_MAJOR with `forward_decode(row_major=True)`) | chip (r, c) = vocab block `b = r C + c` |
| logits (`vocab_split="tp"`) | `[1, 1, 8, 27520]` | bf16 | TILE | chip (dp, tp) = lanes of row dp x vocab block tp |

**Prefill** (one user, S = bucket; identical on all 32 chips unless noted):

| Tensor | Shape | dtype | Layout |
|---|---|---|---|
| residual streams X | `[1, 4, S, 4096]` | bf16 | TILE, DRAM |
| sublayer input / output | `[1, 1, S, 4096]` | bf16 | TILE, DRAM |
| RoPE cos / sin | `[1, 1, S, 64]` | bf16 | TILE (`MotifRope.prefill_cos_sin(kind, S)`) |
| page table | `[1, W]` (feed the fill `[1, S / 64]`) | int32 | ROW_MAJOR |
| MoE partial after RS(dp) (inside MoE); shared-expert partial for it | `[1, 1, S/4, 4096]` (row slice `dp`, `MotifMoE.dp_slice`) | bf16 | TILE |
| logits | the last real token's **tile row** `[1, 1, 32, Vc]` per chip (Vc = 6880 "mesh" / 27520 "tp") | bf16 | ROW_MAJOR |

Logits consumer rule (EMB-D1): **never index device logits directly**; use only `MotifLMHead.logits_to_host`
(fresh `[32, 220160]` in lane order), `prefill_logits_to_host(tile, last_index)` (fresh `[220160]`: the host picks row
`last_index % 32`), `argmax_decode` / `tokens_to_host`; they handle both splits. Prefill: `set_prefill_position(
last_index)` writes the tile-row start into persistent device tensors (tensor-args `ttnn.slice`: programs depend on the
bucket only).

Stream layout vs the reference: TT keeps streams **stream-major** `[1, 4, T, 4096]`; the reference uses
`[B, S, 4, 4096]` -> convert with `x.permute(0, 2, 1, 3)`.

**KV pool geometry (INFRA-2).** vllm-tt-plugin allocates `plugin_num_blocks(pool + 32, block, max_num_seqs)` blocks:
the bridge returns the usable pool (`MOTIF3_KV_POOL_TOKENS`, 262,144) plus a 32-token null-block reserve, the plugin
adds `block * max_num_seqs` -> **4129** blocks of 64 for `--max-num-seqs 32` (4105 for 8), 8.57 GB per chip in bfp8
for 53 layers; block 0 is vLLM's null block. The generator takes `num_blocks` / `block_size` from
`allocate_kv_cache` and records them with `cfg.set_kv_geometry(num_blocks, block_size)`; before that
`cfg.kv_num_blocks` is the expected value (`generator_api.expected_num_blocks`). Block sizes: **32 or 64** (what
G1/G7 validated; BRIDGE-1). Allocate each layer with `ttnn.empty` + on-device `ttnn.fill(0)` (G7: `ttnn.zeros` of
the pool takes ~20 s), and typecast the prefill latent to the cache dtype before `paged_fill_cache` (raw tile copy).

## 4. Dtype policy and compute roles (`cfg.dtypes`, `cfg.compute_config(role)`; design §1.5, INFRA-3)

| Class | Device dtype | Math (compute role) |
|---|---|---|
| routed experts (gate_up, down) | bfp8 | HiFi4, fp32 dest acc (`experts`; G6: HiFi4 costs nothing, 2.6x lower max error); PolyNorm output stays **bf16** (never block-float, study 01 N3) |
| shared expert, dense MLP | bfp8 | HiFi4, fp32 acc (`shared`, `dense_mlp`) |
| attention (W_lat, wq_b, wq_b_gate, W_UK', W_UV', wo) | bf16 | HiFi4, fp32 acc (`attn_latent`, `attn_heads`) |
| FlashMLA decode | KV bfp8, Q bf16 | HiFi4, fp32 acc, approx off (`sdpa_decode`, G1); scores / statistics bf16 inside the op regardless (upstream, §12 gotchas) |
| SDPA prefill | bf16 | HiFi4, **fp32 acc off**, approx off (`sdpa_prefill`, G2, default on every layer); `sdpa_prefill_fp32` (fp32 acc on) is an opt-in for **window-free** calls only (§12 gotchas) |
| router | bf16 weights (bias fp32) | HiFi4, fp32 acc, **fp32 logits / sigmoid / bias / top-k** (`router`); decode logits `cfg.router_logits`: `"composite"` (FPU, 99.81 % top-8 agreement on real tokens) or `"exact_fp32"` (`tt/kernels/router_fp32`, 99.997 %, +24 us per MoE layer; decision D1 on model-level metrics, `MOTIF3_ROUTER_LOGITS`) |
| mHC fused projection | bf16 (gamma folded in fp32, rounded once) | HiFi4, fp32 acc (`mhc`); mixes, Sinkhorn, coefficients fp32 |
| mHC coefficients (Sinkhorn) | fp32 | `cfg.mhc_sinkhorn`: `"motif"` = `tt/kernels/sinkhorn_motif` (Option B, Motif's exact semantics in fp32 SFPU math, max dH 3e-7, ~3 us/site; **the default**); `"stock"` = the pre-clamped `mhc_split_sinkhorn` fallback (MHC-3; misses the 5e-3 bound on 1 of 56 real sites) |
| PolyNorm constants | fp32 | moments in fp32 (`polynorm`), all-reduced exactly (`ccl.ar_exact`) |
| norms (gamma) | bf16 | HiFi4, fp32 acc (`norm`) |
| RoPE | bf16 tables | fused `rotary_embedding_hf`, HiFi4, fp32 acc (`rope`, G8) |
| LM head / embedding | bf16 / bf16 | HiFi4, fp32 acc (`lm_head`) |
| KV cache (latent: unit-RMS `n` 512 + roped `k_pe` 64) | bfp8 (bf16 via `MOTIF3_KV_CACHE_DTYPE`, `cfg.dtypes.kv_cache`) | gamma_kv folded into W_UK'/W_UV' (cache holds the unit-RMS latent) |
| activations, residual streams, RoPE tables | bf16 | - |

Roles (`model_config.COMPUTE_ROLES`, each a `ComputeRole(fidelity, fp32_acc, approx, packer_l1_acc, source)`):
`attn_latent, attn_heads, sdpa_decode, sdpa_prefill, sdpa_prefill_fp32, rope, norm, mhc, router, polynorm, experts,
shared, dense_mlp, lm_head, eltwise, ccl_reduce`. All HiFi4, math approx mode off, **`packer_l1_acc` off** (G6 / G8
ran it on; the infra device test measured the G6 matmuls both ways: identical PCC and latency). fp32 dest acc
everywhere except `sdpa_prefill` (`FP32_ACC_OFF_ROLES`). The old single `sdpa` role is gone (a lookup raises naming
the two replacements). Do not build ad-hoc compute configs in modules; retune a role here, with a module test.

**generic_op kernels** (`tt/kernels/`): a `ttnn.generic_op` compute kernel takes a `ttnn.ComputeConfigDescriptor`
(with per-CB `unpack_to_dest_mode`), not the `BlackholeComputeKernelConfig` a role gives. Build it from the role:
`model_config.compute_config_descriptor(role, fp32_unpack_cbs=(cb, ...), dst_full_sync_en=...)` (fidelity / fp32 acc /
approx from the role, `UnpackToDestFp32` on the listed fp32 input CBs: the default unpack truncates fp32 to TF32).

## 5. Program configs (`model_config`, INFRA-4) -- mandatory where the gates / module sweeps showed it

Verbatim copies of the gate configs (`tests/unit/test_infra_config.py` compares them with the gate files;
`test_infra_device.py::test_program_configs_reproduce_gates` reruns the gate ops through them):

| Builder | Config | Gate result through the builder (this Galaxy) |
|---|---|---|
| `cfg.flash_mla_decode_pc(kind)` / `flash_mla_decode_pc(grid, kind)` | `SDPAProgramConfig(12x10, q_chunk 0, k_chunk 128, exp_approx False, max_cores_per_head_batch m)` with `sdpa_decode`; `m` = 16 on global layers (`kind="global"`, the default), `cfg.flash_mla_swa_mcph` = 4 on SWA layers (A2, `MOTIF3_FLASH_MLA_SWA_MCPH`; 16 = the release config) | PCC 0.99994 SWA / global, SWA window + causal-edge probe exact, 29.8 us SWA / 102 us global 4K traced at m = 16. A2 (`logs/opt/phaseA/A2`, 53-layer T32 trace on TORUS_XY): SWA m = 4 is 23.3 us per call and bitwise equal to m = 16 (the 129-key window spans 2 k-chunks); -0.18 to -0.26 ms per step. A lower global m is not bitwise neutral. **Never** `k_chunk` 0/None (dynamic chunking drops the causal mask under a window: upstream bug), 256 clashes with a sharded Q, 512 overflows L1 |
| `cfg.sdpa_prefill_pc(layer or "swa"/"global", seq_len=S)` | `SDPAProgramConfig(12x10, q/k 128/128 SWA, 256/256 global, exp_approx False)`, clamped to 128 for S = 128; with `sdpa_prefill` | PCC 0.99975 SWA / 0.99961 global at S = 1024; V zero-padded to 192 (dv = 128 is rejected); q512/k512 overflows L1 |
| `cfg.resumed_prefill_pc(layer or "global"/"swa", C)` | sp1 (resumed) chunk of bucket C. **global**: `chunked_scaled_dot_product_attention` config with gate G9's per-bucket q/k (`SP1_GLOBAL_CHUNKS` = `prefill_plan.DEFAULT_SP1_GLOBAL_CHUNKS`: 128/128 at C = 128 and C >= 2048, 64/64 at 256-1024; 64/64 everywhere for a bf16 cache, `kv_dtype=`), with **`sdpa_prefill_fp32`**; **swa**: the G2 square config over 128 + C rows, with `sdpa_prefill` | G9: fp32 dest acc PCC 0.99983-0.99999 in all 68 cases; the bf16-dest `sdpa_prefill` role fails all 68 (worst row 0.988). A start that is not a multiple of q and k is silently floored (no device check): `attention.py` asserts it before every call |
| `cfg.experts_gate_up_pc()` | `MatmulMultiCoreReuseMultiCast1DProgramConfig` grid 10x8, `in0_block_w` 8, `per_core_M` 1, `per_core_N` 1, `out_subblock_w` 1, `fuse_batch` False, `mcast_in0` True; with `experts` | `[1,12,32,4096] @ [1,12,4096,2560]` bfp8: PCC 0.9999986, 438 us traced (auto config: 983 us) |
| `cfg.experts_down_pc()` | same, grid 8x4, `in0_block_w` 4, `per_core_N` 4, `out_subblock_w` 4 | `[1,12,32,1280] @ [1,12,1280,4096]` bfp8: PCC 0.9999986, 230 us traced (auto: 289 us) |

Module builders (wave B1; each returns exactly the module's measured local config, `test_infra_config.py::
test_module_program_configs_match_the_modules`; a grid that does not fit `cfg.compute_grid` gives `None` = auto):

| Builder | Config (measured traced on this Galaxy) |
|---|---|
| `cfg.attn_decode_matmul_pc(name)` / `attn_decode_matmul_pcs()` | 1D mcast `q_lat` 8x4 bw 8 (62 -> 28 us), `kv_lat` 5x4 bw 16 (59 -> 21), `wq_b` 12x5 bw 4 (20 -> 12), `gate` 8x4 bw 8 (17 -> 8); `wo` None (auto 25 us is fastest); per-head bmm `w_uk` / `w_uv` (`reuse_matmul_pc`, one head per core, bw 4, subblock 2: 20 -> 8 / 65 -> 7 us) |
| `cfg.mlp_decode_pc(kind, matmul)` / `mlp_decode_pcs(kind)` | `decode_matmul_pc` (1D mcast, uneven N split, `fuse_batch` True) on `MLP_DECODE_MATMUL_GRIDS`: dense `gate_up` 12x8 bw 8 (40 vs 94 us auto), `down` 8x4 bw 8; shared `gate_up` 10x1 bw 32 (13 vs 51), `down` 8x4 bw 5; `(pcs, fallbacks)` logs grids that do not fit |
| `cfg.mhc_decode_proj_pc(decode_split=32)` | `MatmulMultiCoreReuseProgramConfig`, one output tile per core, the whole 16-tile K chunk in one block (8 us vs 148 us auto; K-blocked configs are TF32-biased, §12) |
| `cfg.router_decode_pc(sigmoid=False)` | 1D mcast 12x1, bw 32 (13.6 vs 52.5 us); `sigmoid=True` puts the SFPU sigmoid in `fused_activation` |
| `cfg.experts_prefill_gate_up_pc(m_rows)` / `_down_pc(m_rows)` / `experts_prefill_pc(m_rows, n_out, out_block_w)` | 1D **in1** mcast on 8x8, bw 8, `per_core_M = M / 64` tiles, `out_block_w` 20 / 16 (17.8 / 10.1 ms vs 35.6 / 18.3 ms auto at M = 4096, bitwise identical); None for small M |
| `cfg.decode_norm_configs(grid=(8, 4))` | width-sharded decode RMSNorm `(memory_config, LayerNormShardedMultiCoreProgramConfig)`: ~6 us (+3 us reshards) vs 65 us interleaved (one core) -- every decoder norm |
| `cfg.lm_head_pc(vocab_split)` | 1D mcast 12x9, `per_core_N` 2 ("mesh", 215 tiles) / 8 ("tp", 860 tiles), bw 16, last block partial (`LM_HEAD_PC`): at the DRAM stream ceiling |
| `mcast1d_matmul_pc(grid, n_tiles, bw, k_tiles, per_core_n=, fuse_batch=, fused_activation=)`, `reuse_matmul_pc(grid, bw, per_core_n, ...)` | the generic builders behind the above (`per_core_n` allows a partial last block; out subblock = widest of 1..4 dividing `per_core_N` with fp32 acc) |

## 6. Per-chip partitioning (design §2.3.4-2.3.8)

| What | Chip `tp` (TP) / chip `(dp, tp)` (EP) holds |
|---|---|
| q heads | `[10tp, 10tp+10)` = KV groups `{2tp, 2tp+1}` (`cfg.chip_heads(tp)`) |
| signal heads (`s = 4g + j`: lambda, gate, `wo` inputs) | `[8tp, 8tp+8)`; local index `s_loc = 4 g_loc + j` |
| local q head order | `h_loc = 5 g_loc + j` (j = 4 is the group's noise head); attention decode keeps a *virtual* order (signal heads first, `attention.virtual_head_order`) |
| dense MLP / shared expert intermediate | `[1536tp, +1536)` / `[160tp, +160)` |
| routed experts | `[12k, 12k+12)`, `k = 8dp + tp` (`cfg.experts_of_chip(dp, tp)`) |
| LM-head vocab (`vocab_split="mesh"`, default, EMB-D1) | `[6880 b, +6880)`, `b = r * C + c` = the **mesh linear index** of chip (r, c), over all 32 chips (`8 dp + tp` on 4x8); the normed hidden is gathered first (`ag_dp_rows`, ~27-45 us). Measured: 218 vs 605 us device per decode step, 169 MB less DRAM per chip |
| LM-head vocab (`vocab_split="tp"`) | `[27520tp, +27520)` (replicated over DP) |
| everything else (W_lat, router, mHC, norms, embedding) | replicated |

## 7. Weights (`tt/weights.py`)

* **Location** (one order for the bridge and the TT config, `generator_api.resolve_weights_location`):
  `MOTIF3_WEIGHTS_DIR` > `HF_MODEL` (a directory) > the HF-cache snapshot of a repo-id `HF_MODEL` at
  `TT_MODEL_WEIGHTS_REVISION` > vLLM's `hf_config._name_or_path` > the local snapshot. A set but missing
  `MOTIF3_WEIGHTS_DIR`, or a path-like `HF_MODEL` that does not exist, raises. An uncached repo id is handed to the
  generator as-is (`GeneratorSettings.weights_are_local` False): the generator downloads it at that revision.
* **Sources**: `HFWeightLoader()` (lazy safetensors over `model.safetensors.index.json` in `cfg.weights_dir`;
  `get(name)`, `get_rows(name, a, b)` for expert slices, `layer_available(l)`; partially downloaded shards are
  never opened) or `DictWeightSource({hf_name: tensor})` for random weights. Both expose the same API; modules
  take a `source` and never call safetensors directly. Names are HF checkpoint names (`weights.hf_name(l, ...)`).
  Reference-named random weights: `reference_to_hf_state_dict(random_state_dict(args, layers))`.
* **Transforms** (host, fp32 -- fp64 inputs stay fp64 -- rounded once at upload). They return the device
  orientation `[in, out]` (`y = x @ W`). Per-chip transforms take `tp` and are concatenated with
  `stack_tp(fn, cfg, dim)` so that a TP shard on `dim` gives chip `tp` exactly `fn(tp)`:
  * attention: `latent_projection_for_chip`, `wq_b_for_chip(layout="split"|"interleaved")`, `wq_b_gate_for_chip`,
    `absorb_weights_for_chip`, `unabsorb_weights_for_chip`, `prefill_kv_expansion_for_chip`, `wo_for_chip`; the
    module's folded / reordered forms (virtual head order, softmax scale and q_norm gamma folded into `wq_b`, per-head
    W_UK / W_UV) stay in `tt/attention.py`, versioned by `ATTN_CACHE_VERSION` in the cache names.
  * mHC: `mhc_fused_projection` (`fn [24, 16384]`, gamma folded) -> `mhc_projection_blocks` (`[1, 4, 4096, 32]`),
    `mhc_scalars` (alphas, biases fp32; `bias_res` row-major `4i + j`); the module's x128 "stream rows" form stays in
    `tt/mhc.py` (`projection_tensor`).
  * MLP / shared: `mlp_gate_up_for_chip` (`[4096, 2 I/8]`, gate first), `mlp_down_for_chip` (x0.5 folded),
    `polynorm_coefficients` (sigmoid(w); bias clamp only for routed).
  * MoE: `experts_gate_up` / `experts_down` (x0.5 folded; the MoE also folds route_scale x2.0), `ep_layout`
    (`[384, ...] -> [4, 96, ...]`), `local_expert_ids`, `expert_polynorm_tensors`, `router_weights` (width 384).
  * `lm_head_for_chip` (`[4096, 27520]`, "tp"; the "mesh" layout is `lm_head.lm_head_device_layout`),
    `norm_weight` (`[1, 1, dim/32, 32]`, upload ROW_MAJOR).
* **Upload**: `weights.as_tensor(src, mesh_device=, cfg=, dtype=, layout=TILE, dp_dim=None, tp_dim=None,
  cache_name="attn.wq_b", layer=l)`. `src` may be a zero-arg callable (not called on a cache hit). Mapping by
  role: both `None` = replicate (cached as one unsharded copy); `tp_dim` = shard over TP, replicate over DP;
  `dp_dim` = shard over DP; both = 2D; EP = `ep_layout(t)` + `dp_dim=0, tp_dim=1` -> each chip `[1, 12, ...]`.
  `weights.shard_for_device(t, cfg.axes, row, col, dp_dim=, tp_dim=)` emulates the mapper on the host.
* **Cache**: `<TT_CACHE_PATH>/<version-tag>/mesh<R>x<C>/{L<nn>|global}/<name>__<mapping>_dtype_<D>_layout_<L>.tensorbin`
  with version tag `motif3-<rev8>-c<CACHE_FORMAT_VERSION>-<dtype tag>` (root: `MOTIF3_TT_CACHE_PATH` > `TT_CACHE_PATH` >
  `/home/ttuser/hchang/experiments/motif-3/tt_cache`, the same order for the bridge and the config). Name tensors `"<module>.<tensor>"`; `layer=None` for embedding /
  final norm / LM head. **Bump `model_config.CACHE_FORMAT_VERSION` whenever a shared transform or layout changes**
  (a module-local transform carries its own version in the name, e.g. `attn.v2.*`). Random weights:
  `cache_name=None`. Cached files reload in the memory config they were built with (DRAM interleaved); re-shard after
  loading if needed. Test caches go under `tt_cache/test/<module>/` and stay small.
* **Cache policy** (`MotifModel(cache=...)`, `tt/model.py` module docstring). Serving takes it from
  `MOTIF3_TT_CACHE_POLICY` (`GeneratorSettings.tt_cache_policy`): `MotifGenerator.create` passes it to `MotifModel`
  unless its model kwargs carry `cache`, and its `create:` log line shows `TT cache policy '<policy>'`. The policy
  acts per part (`global`, each `L<nn>`, the MTP layer `L53`); a part is complete when its `.complete` marker exists.
  * `auto` (default): a complete part loads from the cache. A tensor missing from it (an option variant) is uploaded
    from the checkpoint, listed in `model.cache_misses` and not written. A part that is not complete loads from the
    BF16 checkpoint and **nothing is written**, so every start converts it again in memory: about 30 s per MoE layer,
    about 26 min for the 51 MoE layers (`docs/WEIGHTS_RUNBOOK.md` §7).
  * `write`: a part that is not complete is converted, written under `cfg.cache_dir` and marked complete (the marker
    lists its files), after the disk guard: the cache filesystem must keep `model.MIN_FREE_GB` = 60 GB free after
    `GUARD_FACTOR` 1.1 × the part's estimated size (`estimate_part_bytes`: 3.6 GB globals, 0.36 GB per dense layer,
    6.65 GB per MoE layer, 0.42 GB for `L53`). Otherwise `DiskGuardError` stops the start before that part; the parts
    already written stay marked, and the next start resumes after them. So the first start writes the cache and later
    starts load it. A complete part loads as under `auto`, except that a missing variant file is written into it
    (without the guard; the marker is kept). Disk: about 344 GB for the full bfp8 cache (344.2 GB on this host:
    globals, 53 layers and `L53`) on top of the 630 GB BF16 checkpoint, and the 60 GB floor on top of both.
    Measured on 2026-10-04 (TORUS_XY; `logs/dev/20261004_071616_final_c1_cache_write.log`; BF16 shards in the page
    cache): writing took 8.1 s for the globals, 2.8 s per dense layer and 34.0 s for MoE layer 2, and the next start
    loaded the same four parts in 1.4 s with no cache miss. At these rates a first start of the full model spends
    about 29 min converting (51 MoE layers), plus the BF16 reads when the shards are not in the page cache.
  * `off`: the cache is never read or written; every tensor comes from the checkpoint (random-weight tests).
  * The production converter `scripts/convert_weights.py` (`docs/WEIGHTS_RUNBOOK.md`: staging, sha256, verification,
    option variants, the mock target that needs no device) builds the same files with the same constructors: the 84
    files `write` produced for the globals and layers 0-2 are byte-identical (`cmp`) to the production cache's. The
    production cache also holds option-variant files a default build does not write (stock mHC constants, the
    exact-fp32 router, 8 per dense and 9 per MoE layer). `write` covers a server started on a host where nobody ran
    the converter.
* **Cache names per module** (wave B1; the module docstrings are authoritative; mapping tags: `rep`, `tp<d>`,
  `dp<d>tp<d>`). Every module reads its source lazily, so it builds from the TT cache alone once these exist:

  | Module | Names (per layer `L<nn>` unless global) |
  |---|---|
  | attention | `attn.v2.{wq_lat, wkv_lat, wq_b, wq_b_gate, w_uk, w_uv, kv_expand, wo, lam_expand, noise_expand}` |
  | mHC (`site` = `mhc_attn`, `mhc_ffn`) | `{site}.proj_rows_x128`; `{site}.motif_consts` (Motif kernel) or `{site}.{alpha,bias,lo,hi}_row` (stock) |
  | dense MLP (layers 0-1) | `mlp.gate_up`, `mlp.down`, `mlp.polynorm.{D,E,b}` |
  | shared expert | `moe.shared.gate_up`, `moe.shared.down` (`stats="replicated_gate"`: `moe.shared.gate_full`, `moe.shared.up`), `moe.shared.polynorm.{D,E,b}` |
  | MoE router | `moe.router.weight`, `moe.router.expert_bias`; exact router (`router_logits="exact_fp32"`): `moe.router.weight_fp32k_v1` (+3 MiB per layer per chip; the FPU weight stays, prefill uses it) |
  | MoE experts | `moe.experts.gate_up`, `moe.experts.down_x1` (default `fold_route_scale`: x0.5 PolyNorm x 2.0 route scale; `moe.experts.down` without it), `moe.experts.polynorm.{c0,c1,c2,b,D,E}`, `moe.local_expert_ids` |
  | global | `embed.weight` (`__rep`, 1.8 GB; `tp1` with `shard_hidden`), `final_norm.weight`, `lm_head.weight_mesh` (`__dp0tp3` on 4x8, 1.8 GB over 32 shards) or `lm_head.weight_tp` |

  The converter (CONV-1) builds each module with `cache=True` (that writes exactly these files), then marks the layer
  with `weights.mark_layer_cached(cfg, l, names)`; it never re-derives a module transform.

## 8. Collectives (`tt/ccl.py`)

`ccl = MotifCCL(mesh_device, cfg)`; `ccl.all_gather(x, dim, "dp")`, `ccl.reduce_scatter(x, dim, "dp")`,
`ccl.all_reduce(x, "tp")`, `ccl.ar_exact(x, "tp")`, `ccl.partition(x, dim, "dp")` (per-device slice, no fabric) and
shortcuts `ar_tp`, `ar_dp`, `ag_dp`, `rs_dp`, `ag_tp`, plus **`ag_dp_rows(x)`** for the decode MoE token gather
(INFRA-5). The generic ttnn collectives with op-internal semaphores (GPT-OSS BH precedent): no persistent buffers /
semaphores to manage, trace-safe once compiled; `num_links` / topology come from the fabric unless set.

* **Semaphores in L1_SMALL, never in main L1** (attention P0). With an L1_SMALL region (§1) every MotifCCL path puts
  its global semaphores there (`ccl.l1_small_semaphores`, on by default when the region exists): native all-gathers
  and the direct reduce-scatter do it themselves; composite all-gathers run `all_broadcast(use_l1_small_for_semaphores
  =True)` + `concat`; every other reduce-scatter (DP line, prefill sizes, ROW_MAJOR) gets
  `use_l1_small_for_semaphores=True`; `all_reduce` (which has no such argument in ttnn) is a Python mirror of
  `ttnn.all_reduce` built from those. Measured (`test_infra_l1_small.py`, every draft-1 payload): main L1 +0 B on each
  program-cache miss, 4.7 KiB of L1_SMALL in all, results **bitwise equal** to the plain ttnn ops. The plain ops left
  256 B (4 semaphores) in main L1 per DP-axis / prefill all-reduce and 128 B per composite gather. Cost: traced
  identical (AR(tp) `[8,4096]` 33.5 vs 33.5 us, AR(dp) `[32,4096]` 33.1 vs 33.3, AR(tp) `[1024,4096]` 251.8 vs 250.5);
  eager +23-67 us per all-reduce (Python dispatch of the two ops). Upstream ask: a `use_l1_small_for_semaphores`
  argument on `ttnn.all_reduce` (or L1_SMALL by default in the ring reduce-scatter and `all_broadcast`).
* **Exact small all-reduce** `ccl.ar_exact(x, axis)` (MLP wave B1): `[..., T, 1]` -> zero-pad to one tile -> native
  all-gather on the last dim -> fp32-accumulated `ttnn.sum` (role `ccl_reduce`); other widths gather on dim -3. Exact
  fp32 sums for any row count (the RS+AG all-reduce adds fp32 at TF32 class above 32 rows), 2-3 ops instead of the
  composite all-reduce's ~7 (9-26 us faster for the PolyNorm moments). Equal to `polynorm._ar_ag_sum` bitwise.
* **Size-1 axes** (small test meshes such as (1, 8)): every collective returns a **new** tensor (`ttnn.clone`),
  never its input, so a caller's free of the CCL input is always safe (MoE wave B1; §10 rule 3).
* **Ring all-gather race** (FEATURES_REVIEW P1 and F3; `docs/determinism/INVESTIGATION.md`, `docs/p5_t64/f3.md`): on an
  even ring, ttnn's multicast all-gather factory can signal completion before its alternate-route pages land, so the
  next op reads stale tiles. `MotifCCL(ring_gather=cfg.ring_gather)` (`MOTIF3_RING_GATHER`, §14) runs the gathers its
  predicate (`_ag_race_prone`) marks as `ttnn.all_broadcast` + `ttnn.concat`, bitwise the same data: `safe` (the default
  since B0 `277df0e9f2d`) all of them, decode included (+0.26-0.45 ms per decode step); `lean` only the multi-page ones
  and those of `race_free=True` calls (the MoE prefill combine); `native` none. `MOTIF3_SPEC_VERIFY=wide` / `auto`
  refuse anything but `safe` (F3N rule R1, §18), and `test_infra_config.py::test_r1_no_raw_ttnn_collectives_outside_ccl`
  fails on a raw ttnn collective under `tt/` outside `ccl.py`.

Draft-1 CCL schedule (design §3.2): attention `wo` AR(tp); dense MLP AR(tp) (+ PolyNorm moments `ar_exact(tp)`); MoE
decode `ag_dp_rows` -> local experts -> AR(dp) -> `partition(dim=2, "dp")` -> + shared partial (`add_partial`) ->
AR(tp); MoE prefill RS(dp) -> + shared partial (on `dp_slice` rows) -> AR(tp) -> AG(dp). All-reduces are RS+AG (or
AG+local sum), so replicas stay bitwise identical.

**MoE contracts** (wave B1): decode `moe.forward_decode(f, add_partial=shared.forward_decode(f, all_reduce=False))`, so
one AR(tp) closes routed + shared; prefill `moe.forward_prefill(f, add_partial=shared.forward_prefill(moe.dp_slice(f),
all_reduce=False))`. Decode intermediates live in L1 (~12 MB per chip at peak, all freed inside the call); outputs and
CCL payloads stay in DRAM. The decode router pads the biased scores to width 1024 for the multi-core top-k and uses the
12-core decode config with the fused sigmoid. Inputs (`x`, `add_partial`) are never freed, on any mesh.

* **Token gather in ROW_MAJOR** (G4): an 8-row TILE all_gather takes the composite path (61.7 us traced);
  `ag_dp_rows` untilizes into L1, gathers in ROW_MAJOR (11-12 us) and tilizes back to DRAM: 26.5-28 us traced in all
  (30-31 us with DRAM intermediates), output `[1, 1, 32, 4096]` TILE in lane order `8 dp + l`, replicas identical.
* `ttnn.mesh_partition` is a `slice` underneath and rejects TILE slices of the last two dims that are not tile-aligned
  (8 of 32 rows -> TT_FATAL "Can only slice tilized tensor with height begin index aligned to tiles");
  `MotifCCL.partition` round-trips such slices through ROW_MAJOR (2 extra layout ops).
* Measured precision / cost (`tt/ccl.py` docstring, G4): bf16 all-reduces round like bf16; fp32 all-reduces on the
  RS+AG path are TF32-class (~1e-3 relative; enough when the result is cast to bf16 afterwards); small fp32 payloads
  on the AG + local-sum path and `ar_exact` are exact. Budget **30-40 us per decode CCL** traced (AR(cols) `[8,4096]`
  33 us, AR(rows) `[32,4096]` 33 us; the 8-row RS is 109 us: use AR + partition for the combine). All measured on the
  degraded TORUS_Y fabric: re-measure DP-axis CCLs on a healthy torus (GATE-1).

## 9. RoPE (`tt/rope.py`)

`rope = MotifRope(mesh_device, cfg)` builds both tables once (`"yarn"` for global layers, `"plain"` for SWA;
`cfg.layer(l).rope_kind`). Half-split (HF/NeoX) convention on the checkpoint as stored -- **no weight permutation**.
The tables come from the config's YaRN fields, which `MotifTTConfig.from_hf_config` reads from `rope_scaling` or
transformers-5 `rope_parameters` (INFRA-1: a config built from vLLM's `hf_config` object is identical to one built
from `config.json`). Never use the generic TT RoPE helpers (they ignore or rescale Motif's YaRN).
* **G8 decision: the fused `ttnn.experimental.rotary_embedding_hf`** with the `rope` role (HiFi4 + fp32 acc;
  PCC >= 0.999996, <= 3 us traced in decode). The composite `x cos + (x @ R) sin` (19 us) is the fallback.
* Decode: per step `rot_idxs` (`[1, 32]` uint32 per row) -> `rope.decode_cos_sin(kind, rot_idxs, layout=...)`
  (`ttnn.embedding` gather, trace-safe). **Attention uses `layout="rows"`** (`[1, 1, 32, 64]`, row t = lane t) with
  `rotary_embedding_hf` in *prefill* mode on the heads-on-dim-1 / lanes-on-rows `q_pe [1, 10, 8, 64]` and `k_pe` (same
  kernel, row t rotated by lane t's position; the decode-mode variant would need a transpose + reshard, 4 extra ops):
  build the tables once per step with `MotifAttention.decode_rope_tables(rope, rot_idxs)` and pass them to every layer
  (§2). `"batch_sharded"` (`[1, 8, 1, 64]` HEIGHT_SHARDED, one lane per core, for `apply_hf(..., is_decode_mode=True)`
  with `rope.batch_sharded_memory_config()`) and `"batch"` (`[1, 8, 1, 64]`) remain for the composite / decode-mode
  paths (G8).
* Prefill: `rope.prefill_cos_sin(kind, S)` -> `[1, 1, S, 64]` TILE (positions `0..S-1`, cached per bucket), applied
  with `rope.apply_hf(x [1, H, S, 64], cos, sin)`.

## 10. Module interface pattern

```python
class MotifAttention:                                   # likewise MHCSite, PolyNormMLP, MotifMoE, ...
    def __init__(self, mesh_device, cfg: MotifTTConfig, layer_idx: int, *,
                 source,                                # HFWeightLoader | DictWeightSource
                 ccl: MotifCCL, rope: MotifRope | None = None,
                 cache: bool = True):                   # False for random weights (no files written)
        self.spec = cfg.layer(layer_idx)                # window, softmax_scale, rope_kind, attn_kind, is_moe, ...
        self.wq_b = weights.as_tensor(lambda: ..., mesh_device=mesh_device, cfg=cfg, dtype=cfg.dtypes.attention,
                                      tp_dim=1, cache_name="attn.v2.wq_b" if cache else None, layer=layer_idx)
        self.decode_pc = cfg.flash_mla_decode_pc()      # program configs from model_config, never ad hoc
        self.decode_ckc = cfg.compute_config("sdpa_decode")
    def forward_decode(self, x, *, rot, cur_pos, page_table, kv_cache, active) -> ttnn.Tensor: ...
    def forward_prefill(self, x, *, page_table=None, kv_cache=None) -> ttnn.Tensor: ...
```

Rules:
1. Constructor: all weights via `weights.as_tensor` (lazy sources), dtypes from `cfg.dtypes`, compute configs
   from `cfg.compute_config(role)` (generic_op kernels: `compute_config_descriptor(role, ...)`), program configs from
   the `cfg.*_pc()` builders, per-layer constants from `cfg.layer(l)`, module defaults (`cfg.mhc_sinkhorn`,
   `cfg.router_logits`) from the config. No device work in forward except ops.
2. `forward_decode` is **trace-safe**: no host round trips, no `from_torch`, no data-dependent Python control
   flow, no new program shapes after warmup; per-step inputs (positions, rot indices, page table) arrive as
   persistent device tensors. Integer slice starts must be per-layer constants (or tensor-args slices).
3. Inputs/outputs follow section 3; modules do not deallocate their inputs unless the docstring says
   "consumes"; outputs are new tensors in DRAM interleaved unless documented (MotifCCL included, on every mesh).
4. Collectives only through `MotifCCL` with role names; mesh-dependent indices only via `cfg.axes`.
5. Each module documents which design section it implements and keeps a fallback (composite) path where the
   design names one.
6. **Warm-up before trace capture**: every program a decode trace replays must be compiled by an eager call first;
   for `ttnn.generic_op` kernels (`tt/kernels/*`, `mhc` finalize / layout / post kernels) one *eager* call per shape is
   mandatory ("Cannot load new binaries during trace capture" otherwise: `prepare_generic_op` alone does not load the
   binaries). Prefill: one call per bucket before decode capture (the LM head's prefill programs depend on the bucket
   only); free a prefill's outputs before the next decode replay.
7. **L1 across calls**: a tensor kept in L1 between two module calls narrows L1 for everything in between. Draft 1 has
   one such case: mHC decode keeps `w_pre` / `w_post` (~100 KB per chip) in L1 between `site.pre` and `site.post`
   (across the sublayer), plus a 1.3 MB transient inside `post` (`mix_l1=False` moves them to DRAM, +20 us per site).
   Everything else frees its L1 intermediates inside the call (MoE decode peaks at ~12 MB per chip).

Module APIs (the module docstrings are authoritative; wave-B1 summary in `docs/WAVE_B1_SUMMARY.json`):

| Module | Decode | Prefill |
|---|---|---|
| `MotifEmbedding(mesh, cfg, source=, ccl=)` | `forward_decode(tokens [4, 8] uint32)` -> `X [1, 4, 8, 4096]` (`decode_tokens_host/device`) | `forward_prefill(tokens)` -> `[1, 4, S, 4096]` (`prefill_tokens_host/device(tokens, bucket)`) |
| `MHCSite(mesh, cfg, l, "mhc_attn"\|"mhc_ffn", source=, sinkhorn=cfg.mhc_sinkhorn)` | `x_red, coeffs = site.pre(X)`; `X' = site.post(X, out, coeffs)` (frees `coeffs`) | same (any T; buckets warmed per shape) |
| `MotifAttention(mesh, cfg, l, source=, ccl=, rope=)` | `forward_decode(x, rot=, cur_pos=, page_table=, kv_cache=, active=)` | `forward_prefill(x, page_table=, kv_cache=)` |
| `PolyNormMLP` / `MotifDenseMLP` / `MotifSharedExpert(mesh, cfg, l, source=, ccl=)` | `forward_decode(x, all_reduce=True\|False)` | `forward_prefill(x, all_reduce=...)` (row chunks of 8192) |
| `MotifMoE(mesh, cfg, l, source=, ccl=, router_logits=cfg.router_logits)` | `forward_decode(x, add_partial=)` | `forward_prefill(x, add_partial=)` (`dp_slice` rows) |
| `MotifLMHead(mesh, cfg, source=, ccl=, vocab_split="mesh")` | `forward_decode(X, row_major=True)` -> `logits_to_host` / `argmax_decode` | `set_prefill_position(last)`; `forward_prefill(X)` -> `prefill_logits_to_host(tile, last)` |

## 11. Building the config (`tt/model_config.py`)

* `MotifTTConfig.from_hf_config(src=None, *, mesh_device=None, mesh_shape=None, **overrides)`: `src` = a
  `config.json` path or dir (None = weights dir / hf_meta), a dict, or a transformers config object (vLLM's
  `hf_config`). All give identical fields since INFRA-1 (`rope_scaling` or `rope_parameters`; EOS (0, 3, 6) from the
  `generation_config.json` next to the file or the object's `_name_or_path`). With a real mesh: mesh shape, compute
  grid, fabric and the L1_SMALL size come from the device. Overrides win; `kv_cache_dtype="bf16"` maps onto
  `dtypes.kv_cache`. PolyNorm semantics are parsed too: `polynorm_sigmoid_weight` (default True; False raises:
  unsupported), `polynorm_output_scale_per_layer` (`{layer: scale}`, `cfg.polynorm_output_scale_for_layer(l)`).
* `MotifTTConfig.from_settings(settings, *, mesh_device, hf_config=None)`: what `MotifGenerator.create` builds
  (GEN-1): `<settings.weights_path>/config.json`, `num_layers`, `max_model_len = max_seq_len`, `max_batch = 32`,
  `max_num_seqs = max_batch_size`, KV dtype, block size when the bridge saw it (BRIDGE-4), cache root, revision.
  `create` should also call `model_config.require_l1_small(mesh_device)` (the bridge already checked it).
* `cfg.set_kv_geometry(num_blocks, block_size)` in `allocate_kv_cache` (GEN-2); `cfg.kv_blocks_per_seq` =
  `min(cdiv(max_model_len, block), N)` (the bridge's W; the width `warmup_decode` receives is authoritative).
* `max_model_len` must be a multiple of 32 (the last bucket is `max_model_len`); serving requires a multiple of
  **256** (SDPA prefill chunk; the bridge refuses others before loading weights, BRIDGE-3).

## 12. Tests and comparison against the reference

* Golden = `models/demos/motif3/reference` (import it **lazily inside tests**; its API may still change; write a
  local torch golden if something is missing and say so). CPU-side equivalences of the transforms are in
  `tests/unit/test_infra_weights.py` (absorbed == expanded GDLA at real dims, folded mHC projection, TP PolyNorm).
  Real-prompt goldens of the 4-stream state after layers {0-4, 7, 8, 15, 16, 23, 24, 31, 32, 35} and an early-exit
  head: `goldens/c2` (`reference/golden_stream.py`). Name collisions: `generator_api.MotifGenerator` vs
  `reference.generate.MotifGenerator`, `generator_vllm.MotifKVCache` vs `reference.cache.MotifKVCache` -- import with
  explicit aliases (M11).
* Random-weight module tests: build HF-named random weights (or the reference's `random_state_dict` +
  `reference_to_hf_state_dict`), wrap in `DictWeightSource`, construct the TT module with `cache=False`, run the
  matching reference module on CPU, read back with `ccl.device_tensors_to_torch(t, mesh)` (`[R, C, ...]`, entry
  `[r, c]` = mesh coordinate `(r, c)`), compare per chip. Decode tests must give each DP row different lanes /
  positions and compare row `dp` against reference lanes `8dp .. 8dp+7`; caches hold stale (non-zero) data outside the
  history (a zero slot cannot reveal a mask leak).
* KV contents: the TT cache holds the gamma-free unit-RMS latent; compare `TT_cache[..., :512] * gamma_kv` with the
  reference `self_attn.c_kv` tap (which includes gamma) (M18).
* Thresholds (design §4.3, WAVE_A_REVIEW D1): RMSNorm >= 0.9999, PolyNorm >= 0.9995, GDLA >= 0.999 (per lane too),
  MoE output >= 0.995 (bfp8), decoder layer >= 0.995 (prefill and teacher-forced decode vs `goldens/c2`), truncated
  model state >= 0.99 after several layers, H_res max-abs <= 5e-3. Router: report top-8 set agreement on **real**
  inputs; the acceptance is model-level (layer PCC, teacher-forced top-1 >= 95 %). Report max-abs error and PCC per
  test and decode latency per module call (eager and traced). Always also assert `ccl.replicas_identical(out, mesh,
  "tp")` for row-replicated outputs. PCC helper: `models.common.utility_functions.comp_pcc` (import inside the test;
  it zeroes NaN, so also count non-finite values). Serving-order runs (prefill -> decode -> global prefill S >= 1024
  -> decode) must work in ONE session; trace replay with new inputs must equal eager bitwise.
* Real-weight tests: `HFWeightLoader().layer_available(l)` (layers 0-35 are local; `.download_state.json` lists
  complete layers); skip, never download, if missing. Large artifacts only under
  `/home/ttuser/hchang/experiments/motif-3/tt_cache`.

**Gotchas (measured in wave B1; each cost someone a debugging session)**

* `ttnn.pad` that stays inside the tile padding (e.g. 8 -> 32 rows, 1 -> 32 columns) is an in-place
  `fill_implicit_tile_padding` of its input + a **view**: the result shares the input's buffer (deallocating it frees
  the caller's tensor) and the input's padding is zero-filled (`pad.cpp:478-480`).
* Decode tensors with 8 logical rows (of a 32-row tile) trigger FillPad device ops inside reductions and composite
  CCLs (8-14 us per PolyNorm); pad to 32 logical rows inside a module (`mlp.pad_decode_rows`) and slice back.
* An interleaved `ttnn.rms_norm` on a one-tile-row decode input runs on ONE core (65 us): use
  `cfg.decode_norm_configs()` (width-sharded, ~6 us).
* `ttnn.experimental.fast_reduce_nc` on fp32 truncates its inputs to TF32 (-3.5e-4 relative bias); `ttnn.sum(dim=1)`
  on fp32 is exact. The FPU unpacks fp32 operands to TF32 by truncation (mHC weighted reduce: round the weights to TF32
  first); generic_op kernels need `UnpackToDestFp32` for exact fp32 inputs (§4).
* `MatmulMultiCoreReuseProgramConfig` with a batched in1 reloads K-blocked partials through UnpackToSrc (TF32, biased
  up to -1.2e-3) and mis-computes with more than one output block per core: keep the whole K in one block
  (`cfg.mhc_decode_proj_pc()`).
* SDPA prefill with `sliding_window_size` and fp32 dest acc returns wrong output for S >= 256 (the legacy kernel's
  window mask uses the Q tile's first row): **never** `sdpa_prefill_fp32` on a windowed call (upstream issue; pinned by
  `test_sdpa_prefill_window_fp32_acc_upstream_bug`).
* `sdpa_decode_program_factory.cpp` hard-codes the FlashMLA scores / statistics to bf16 even with fp32 acc: the decode
  accuracy floor (op PCC ~0.9999) is the kernel's (upstream issue).
* `ttnn.slice` with integer starts hashes them into the program: one program per position. Use the tensor-args
  slice (LM-head prefill) so a new prompt length never compiles at serving time.
* `ttnn.generic_op`: one eager call per shape before a trace capture (§10 rule 6).
* Host side of serving: torch's intra-op OpenMP workers spinning after a large host op (e.g. vLLM's sampler upcast of
  14 MB of logits, `torch.cat`) slowed the next 32-chip device read from 0.5 to 2.4-3.2 ms (lm_head); avoid torch
  parallel ops right before device reads (`OMP_WAIT_POLICY=PASSIVE` is set in both TIS specs, TIS runbook pitfall 29).

## 13. Import rule

Nothing under `models/demos/motif3` imports another `models/demos/**` package at module import time (vLLM imports
the bridge before the mesh opens; demo imports can open the cluster). Copy small helpers, or import lazily inside
functions. vLLM / transformers / huggingface_hub / safetensors are imported lazily too.
`tests/unit/test_infra_import.py` checks, in a fresh interpreter, the shared infra, every wave-B1 module
(`mhc, attention, polynorm, mlp, moe, embedding, lm_head`) and the kernels (`kernels`, `kernels.sinkhorn_motif`,
`kernels.router_fp32`), plus `decoder`, `model` and `generator` as soon as their files exist; the bridge host suite
checks that the default generator class imports device-free (`test_real_generator_class_imports_device_free`).

## 14. Environment

Every environment variable `tt/*.py` reads (`os.environ`; grep of B1 `d3597ae4977`, plus `MOTIF3_TT_CACHE_POLICY`
and `MOTIF3_WIDE_STEP_RATIO`, added after it, and the Phase A knobs of branch `motif3-opt`: `MOTIF3_CHUNK_BUDGET`, `MOTIF3_FLASH_MLA_SWA_MCPH`, `MOTIF3_ROUTER_MASK`; Phase B: `MOTIF3_DECODE_EXPERTS`, `MOTIF3_MOE_POLYNORM`, `MOTIF3_SHARED_POLYNORM`, `MOTIF3_HOST_STAGING`, `MOTIF3_HOST_WAIT`, `MOTIF3_ASYNC_DECODE`, `MOTIF3_PREFILL_MOE`, `MOTIF3_PREFILL_MOE_BLOCK`, `MOTIF3_PREFILL_MOE_MIN_ROWS`, `MOTIF3_PREFILL_MOE_DISPATCH`, `MOTIF3_PREFILL_MOE_COMBINE`; Phase C: `MOTIF3_PREFILL_MOE_UPLOAD`, `MOTIF3_SHM_TRACKING`, `MOTIF3_ATTN_EPILOGUE`, `MOTIF3_ATTN_MM_PCS`, `MOTIF3_MHC_DECODE`, `MOTIF3_MOE_DECODE_CCL`, `MOTIF3_AG_ROWS_LAYOUT`, `MOTIF3_PREFILL_SP`, `MOTIF3_PREFILL_SP_MIN_ROWS`, `MOTIF3_DEBUG_SYNC`; Phase D: `MOTIF3_DECODE_EXPERT_MM`, `MOTIF3_PGD`; Phase E: `MOTIF3_MOE_REPLICAS`, `MOTIF3_MOE_REPLICA_PLAN`, `MOTIF3_MOE_REPLICA_DIR`, `MOTIF3_MOE_REPLICA_KEEP`; Phase F: `MOTIF3_ATTN_IN`, `MOTIF3_ATTN_OUT`, `MOTIF3_MOE_LOCAL`). "Validated" names the function that refuses a bad value. Under TIS,
the specs set `MESH_DEVICE`, `MOTIF3_KV_POOL_TOKENS`, `OMP_WAIT_POLICY`, `MOTIF3_PACKED_PREFILL`,
`MOTIF3_TT_CACHE_POLICY` (both specs) and `MOTIF3_SPEC_VERIFY` (MTP spec), and a spec value beats a shell export; TIS
itself sets `HF_MODEL` and `TT_CACHE_PATH`; every other variable reaches the server from the shell
(`docs/TIS_RUNBOOK.md` §1.2, §2.1, pitfall 34).

| Variable | Default | Effect | Validated |
|---|---|---|---|
| `MOTIF3_WEIGHTS_DIR` | unset | checkpoint dir; wins over everything | `generator_api.resolve_weights_location`: set but not a directory raises |
| `HF_MODEL` | unset | snapshot dir (TIS), or a repo id resolved through the HF cache (`HF_HUB_CACHE`, else `$HF_HOME/hub`) at `TT_MODEL_WEIGHTS_REVISION` | `resolve_weights_location`: neither a directory nor an `org/name` repo id raises |
| `MOTIF3_TT_CACHE_PATH` | unset | TT weight-cache root; wins over `TT_CACHE_PATH` (shares a converted cache with a TIS server without the runbook's symlink, CONV-2) | `generator_api.resolve_tt_cache_path`: the first non-empty value wins, no check |
| `TT_CACHE_PATH` | `/home/ttuser/hchang/experiments/motif-3/tt_cache` | TT weight-cache root; TIS sets it under its host volume | as above |
| `MOTIF3_TT_CACHE_POLICY` | `auto` in the code; **`write` in both TIS specs** | how the model build uses the TT cache (§7, "Cache policy"): `auto` loads the parts a converter marked complete and never writes, so a start without a converted cache converts the BF16 checkpoint in memory every time (~26 min for the MoE layers); `write` converts, writes and marks every part that is not complete, after the disk guard (60 GB must stay free after 1.1 × the part), so the first start writes the cache (~344 GB) and later starts load it; `off` never uses the cache. `MotifGenerator.create` passes it to `MotifModel(cache=...)` unless its model kwargs carry `cache`, and logs it at the end of its `create:` line | `generator_api.tt_cache_policy_from_env` (case and blanks ignored) and `check_tt_cache_policy` (`GeneratorSettings`, again in `create` before anything loads): `auto` / `write` / `off`, a typo raises |
| `TT_MODEL_WEIGHTS_REVISION` | `2ed2ed5c...` | checkpoint revision in the cache tag and for repo-id resolution | none |
| `MOTIF3_NUM_LAYERS` | 53 | truncated bring-up runs | `GeneratorSettings.from_env` and `MotifTTConfig.validate`: outside [1, 53] raises |
| `MOTIF3_KV_POOL_TOKENS` | 262144 | usable KV pool tokens (262144 -> 4129 blocks of 64 with the reserve); TIS `max_tokens_all_users_override` must equal it | `generator_api.kv_pool_tokens_from_env`: a multiple of 128 in [128, 4194304] |
| `MOTIF3_KV_CACHE_DTYPE` | `bfp8` | or `bf16` (needs a smaller pool; A = 64, §15) | `kv_cache_dtype_from_env`, `GeneratorSettings` |
| `MOTIF3_KV_MAX_GB_PER_CHIP` | 16 | KV budget of the bridge's fail-fast memory check | `generator_vllm.kv_max_bytes_per_chip`: must be positive |
| `MOTIF3_MAX_MODEL_LEN` | 32768 | max context for host-built configs (serving takes vLLM's `--max-model-len`) | `MotifTTConfig.validate`: a positive multiple of 32 |
| `MOTIF3_TRACE_REGION_SIZE` | 268435456 | trace region bytes of `device_params()` / `open_motif_mesh()` (serving: the `"tt"` config's `trace_region_size`) | integer parse only |
| `MOTIF3_L1_SMALL_SIZE` | 32768 | L1_SMALL bytes per core for `device_params()` / `open_motif_mesh()` (experiments only) | `device_params`, `MotifTTConfig.validate`: >= 0; `create` refuses a mesh with less than 32768 (`require_l1_small`) |
| `MOTIF3_FABRIC` | `FABRIC_2D_TORUS_XY` | fallback `FABRIC_1D_RING`; with a mesh the device's fabric wins | `model_config.fabric_config_from_name`: an unknown name raises |
| `MESH_DEVICE` | `(4, 8)` | `"(4, 8)"` for serving (`(8, 4)` also works; `open_motif_mesh()` reads it; the plugin's preset names map to (8, 4)) | `model_config.mesh_shape_from_env`: not a 2D shape raises |
| `MOTIF3_ROUTER_LOGITS` | `composite` | `cfg.router_logits`: `composite` or `exact_fp32` (decision D1 A/B; `tt/kernels/router_fp32`, +24 µs per MoE layer). The exact kernel runs at 32 gathered decode rows and, since B1, at the 64 rows of a T64 step (review I-2) | `MotifTTConfig.validate`: the two values; with `MOTIF3_SPEC_VERIFY=auto`, the T64 row count must be in `model_config.ROUTER_EXACT_FP32_DECODE_ROWS` (= `moe.EXACT_ROUTER_DECODE_ROWS` = (32, 64)), re-checked in `MotifGenerator._check_wide_launch` |
| `MOTIF3_ROUTER_MASK` | `fused` | A5 / B4 (`docs/OPTIMIZATION_PLAN.md` §3.3; branch `motif3-opt`): `cfg.router_mask`, the decode routing-weight path of every MoE layer (`MotifMoE(router_mask=)`). `gather` = the release (`ttnn.gather` of the top-8 scores + the idx-based local mask); `scatter` = `MotifRouter.route_local`: topk's indices scattered into a 0/1 mask (tie-exact: the plain `>= 8th value` threshold picks 9 experts on an exact fp32 tie, 1 in 32,681 real token-layers), normalized over the 384 columns, this chip's 12 weights extracted with a constant one-hot (`weights.local_expert_mask`, +30 MB DRAM per chip). Probe (`logs/opt/phaseA/A5`, TORUS_XY): same top-8 sets on every real token, weights within 1-4 fp32 ulp, MoE output <= 1 bf16 ulp (PCC 0.9999999998), T64 rows == T32 rows and reruns bitwise; router + local mask 140 -> 104 us per layer, -1.7 to -2.0 ms per decode step (T64 verify -1.6). `fused` = B4 (`MotifRouter.route_fused`; results `logs/opt/phaseB/B4`): the router matmul (sigmoid fused, unchanged) + one `generic_op` (`tt/kernels/router_topk.py`, one core per gathered row) that adds the bias on the fp32 SFPU (the op `ttnn.add` runs), selects the top-8 by exact fp32 compares (an exact tie at the 8th value goes to the lower expert id), normalizes over the 8 unbiased scores in fp32 and writes this chip's 12 weights, replacing add / concat / topk and every mask op. Same top-8 sets as `gather` on all 35,652 real token-layers (11 layers + L2 exact), weights within 4 fp32 ulp, T64 rows == T32 rows; router + mask 140 -> 27 us per layer (A5: 104), MoE decode 1036 -> 926 us at M = 32, 1210 -> 1083 us at M = 64. **The default** since the Phase B accuracy eval passed (CI_NIGHTLY on TIS, `docs/OPT_PHASE_B_EVAL.md`); decode outputs are not bitwise equal to the release, and `gather` restores it. `scatter` (A5) is deprecated: B4 selects the same top-8 sets and is faster. Prefill always takes the gather path | `MotifTTConfig.validate`: `gather` / `scatter` / `fused` (case and blanks ignored) |
| `MOTIF3_DECODE_EXPERTS` | `sparse` | B1 (`docs/OPTIMIZATION_PLAN.md` §3.3; branch `motif3-opt`; probe `logs/opt/phaseA/M6`, results `logs/opt/phaseB/B1`, default flip `logs/opt/phaseB2/B1-FLIP`): `cfg.decode_experts`, the decode routed experts of every MoE layer (`MotifMoE(decode_experts=)`). `sparse` is the default since 2026-10-07 (bitwise equal to `dense` on live rows, so no eval was needed; held back until then only by the G16 ratio bar, which was restated, `test_spec_decode_device.py::G16_STEP_BAR`). `dense` = the release (each chip runs its 12 local experts on all 32 / 64 gathered rows); `sparse` = `ttnn.sparse_matmul` (`nnz=None`: a static count deadlocks on BH) skips the local experts no live row routes to. The routing weights of inactive lanes are zeroed first with the step's gathered live-row mask (`MotifMoE.decode_lane_mask`, built once per step by `MotifModel.decode*`), so idle lanes activate no expert. Live rows are bitwise equal to `dense`; inactive rows become 0 + the shared expert (nothing reads them). Needs `combine_mode="fold"` (the default). Prefill is not affected | `MotifTTConfig.validate`: `dense` / `sparse` (case and blanks ignored); `moe.resolve_decode_experts` |
| `MOTIF3_MOE_POLYNORM` | `fused` | B3 (`docs/OPTIMIZATION_PLAN.md` §3.3; branch `motif3-opt`; prototype `logs/opt/phaseA/M10`, results `logs/opt/phaseB/B3`): `cfg.moe_polynorm`, the decode routed-expert PolyNorm of every MoE layer (`MotifMoE(moe_polynorm=)`). `composite` = the release (`tt/polynorm.py` grouped Horner fp32, ~17 ops); `fused` = one `generic_op` (`tt/kernels/moe_polynorm.py`: fp32 moments on 4 cores per expert + moment exchange + Horner with the routing weight folded in, `h` rounded to bf16 once) at the decode row counts 32 / 64, dense and B1 sparse. Not bitwise equal to `composite` (1-ulp bf16 differences in ~2e-5 of the values; deterministic, T64 rows == T32 rows), and it is **the default** since the Phase B accuracy eval passed (CI_NIGHTLY on TIS, `docs/OPT_PHASE_B_EVAL.md`); `composite` restores the release. Needs the fp32 decode PolyNorm and `combine_mode="fold"` (the defaults). Prefill is not affected | `MotifTTConfig.validate`: `composite` / `fused` (case and blanks ignored); `moe.resolve_moe_polynorm` |
| `MOTIF3_SHARED_POLYNORM` | `fused` | B5 (`docs/OPTIMIZATION_PLAN.md` §3.3; branch `motif3-opt`; results `logs/opt/phaseB/B5`): `cfg.shared_polynorm`, the decode PolyNorm of every MoE layer's shared expert (`PolyNormMLP(shared_polynorm=)`, `mlp.resolve_shared_polynorm`). `composite` = the release (`tt/polynorm.py` `polynorm_tp`: 19 small programs between the gate_up and down matmuls, incl. the moments all-gather); `fused` = `tt/kernels/shared_polynorm.py`: a one-core moments kernel on the gate_up output, the same TP all-gather, a one-core apply kernel (partials summed in gather order, `rsqrt(s D + E)`, Horner, `* up`, bf16 once). Both kernels issue the release's LLK operations in the release's order, so the output is bitwise the composite's; **on by default** since the B5 gates passed (all 51 shared experts bitwise, 53-layer decode logits bitwise, FMV identical); `composite` restores the release ops. Needs the fp32 decode PolyNorm, `stats="tp"` and the default PolyNorm options; the dense MLPs (layers 0-1, MTP) and prefill are not affected | `MotifTTConfig.validate`: `composite` / `fused` (case and blanks ignored); `mlp.resolve_shared_polynorm` |
| `MOTIF3_HOST_STAGING` | `fast` | B6a (`docs/OPT_PHASE_A_REVIEW.md` §7.1-7.2; branch `motif3-opt`; results `logs/opt/phaseB/B6a`): `cfg.host_staging` (`MotifGenerator.host_staging`) and the bridge's `MotifForCausalLM.host_staging`, the per-decode-step host input staging. `release` = the release code; `fast` = the same device inputs with fewer host ops: the generator builds its path inputs (`_path_host_rows`) with one cached DP-row mesh mapper and does not copy an input whose values equal the ones it copied last (`DecodePath.host_last`; on the plain `row` path the page table is copied only when a lane crosses a block or joins / leaves, as `DecodeKVWrite.write_step` already does for its own inputs); the bridge fits the page table on the used columns only (`_fit_page_table_fast`) and caches the lane index tensor. Host only: the device sees bit-identical inputs (device gate `tests/test_host_staging_device.py`), so `fast` is the default since the B6a gates; `release` restores the release code | `MotifTTConfig.validate` / `generator_api.check_host_staging`: `release` / `fast` (case and blanks ignored) |
| `MOTIF3_HOST_WAIT` | `spin` | B6a: `cfg.host_wait`, how a decode step waits for its replayed trace. `block` = the release (the step's blocking read sleeps for the whole replay, ~85 ms; under the host's `schedutil` governor the host code after it then runs ~2.5-3x slower than on a busy core); `spin` = `generator.ReplayWaiter`: the calling thread polls `time.sleep(0)` (GIL released every iteration) until 3 ms before the predicted end of the replay (the shortest of the last 8 timed replays of that path / pass), then makes the same blocking read. Costs one busy host core during decode. Host only: the device sees the same commands; default since the B6a gates (TPOT c = 1 / 32: -3.8 / -6.3 ms with `fast`, outputs bitwise equal, `logs/opt/phaseB/B6a/report.md`); `block` restores the release | `MotifTTConfig.validate` / `generator_api.check_host_wait`: `block` / `spin` (case and blanks ignored) |
| `MOTIF3_ASYNC_DECODE` | `off` | B6b (`docs/OPTIMIZATION_PLAN.md` §3.3 B6; branch `motif3-opt`; results `logs/opt/phaseB/B6b`): the bridge's `model_capabilities["supports_async_decode"]` (read at import, like the feature switches). `on` = vLLM asynchronous scheduling for the launches WITHOUT `--speculative-config` (launch without `--no-async-scheduling`): a device-sampled decode step returns before its 1 KB read (`MotifPendingDecode`; `generator.submit_decode_sampled` / `read_decode_sampled`), and a steady step (`reload_inputs=False`) continues the previous one (positions + 1, page table from the call, tokens = the previous step's sampled tokens read on the host after every other input was written), so vLLM's scheduling and the plugin's input build overlap the replay. The device runs the same trace on the same inputs (token-exact vs `off`). Structured output, logprobs, penalties and other host-sampled steps reload and read synchronously (the plugin drains first). Refused with speculation (the plugin refuses async scheduling with it). `off` = the release (every decode reloads and reads before returning). Not the default yet: TPOT c = 1 / 8 / 32 -2.5 / -3.0 / -3.4 ms with every serving gate token-exact, but the launch flag changes too (`logs/opt/phaseB/B6b/report.md`) | `generator_api.check_async_decode`: `off` / `on` (case and blanks ignored) |
| `MOTIF3_PREFILL_TRACE` | `128` | B7 (`docs/OPTIMIZATION_PLAN.md` §3.3 B7; prototype `logs/opt/phaseA/m8`; results `logs/opt/phaseB/B7/report.md`): `cfg.prefill_trace`, traced prefill of small solo chunks. `off` = the release (every prefill chunk eager: ~0.72 s of host dispatch for a 128-row chunk whose device time is ~0.21 s). `128` (the default since its gates passed: traced == eager bitwise; served TTFT c = 1 (`dsamp_pk`, 2 boots per arm, medians [M]) at 128 tokens 0.754 / 0.748 -> 0.214 / 0.213 s (0.214 s on the MTP launch too), at the cost of 1K 1.043 / 1.044 -> 1.066 / 1.078 s and 4K 1.888 / 1.887 -> 1.892 / 1.900 s; greedy outputs equal; the MTP launch holds its two decode traces and both prefill shapes in 187 of 256 MiB), `on` (= `128`) or a comma list of buckets from 128 / 256 / 512 (256 / 512: correct at 4 layers, trace size and gain at 53 layers unmeasured): `warmup_decode(enable_trace=False)` also stages the persistent inputs of every `(sp0, b)` / `(sp1, b)` shape (F3N R3), and `warmup_decode(enable_trace=True)` captures one trace per shape after the decode traces (embedding -> layers -> the LM-head tile row at the head's persistent position; with the MTP layer a second trace for its KV-only fill, whose last next token is the host argmax of the chunk's logits), once (R5), compiling nothing (R2, checked). Every solo chunk of a captured shape then replays it (`MotifGenerator._run_chunk_traced`: tables, tokens and head position written, replay, logits read before any other replay, R4); packed passes, other shapes and calls with a `chunk_observer` stay eager. Bitwise equal to eager (logits, KV, MTP cache, the decode after it). 128: sp0 0.719 -> 0.208 s, the sp1 tail 0.729 -> 0.240 s per chunk; 42.6 / 43.3 MiB of the trace region per shape (53 layers), +4-5 s capture each. A shape the trace region cannot hold (estimate: 1 MiB per layer + 4 MiB) is not captured (logged, `prefill_traces_skipped`) and runs eagerly. A bucket whose MoE chunk would run compacted (B2a) is refused | `MotifTTConfig.validate` / `generator_api.check_prefill_trace`: `off` / `on` / a list of 128, 256, 512 (case and blanks ignored) |
| `MOTIF3_CAPTURE_THREAD` | `main` | E3 / B7 (`logs/opt/phaseB/B7/report.md`): `cfg.capture_thread`, the host thread every trace capture (decode; and B7 prefill) runs on. A capture keeps thousands of small host objects alive for the trace's lifetime; made on the serving thread they fragment glibc's main malloc arena (3.2 k -> 30 k free chunks) and every later eager prefill pass of a dispatch-bound shape runs ~20-30 ms slower per live trace (sp0 128: 716 -> 742 ms after one capture, 750 ms with the decode trace too). `worker` = each capture on a short-lived worker thread (its own arena; joined before warmup returns): in process 716 -> 720 ms with both traces alive, bitwise neutral (B7's device gates, determinism, serving order, GS-6 / CP9 / T64 all pass with it). Not the default: served with `MOTIF3_PREFILL_TRACE=128` (`dsamp_pk`, c = 1, 2 boots per arm, medians) it is slower than `main` at 1K (1.111 / 1.120 vs 1.066 / 1.078 s; release 1.043 / 1.044 s) and 4K (1.918 / 1.915 vs 1.892 / 1.900 s; release 1.888 / 1.887 s) [M]. (An earlier claim of 1.156 -> 1.077 s at 1K rested on a contaminated release arm.) `dedicated` (B7-FIX `d9ff03e219f`) = one long-lived capture thread: in process it removes the post-capture eager slowdown (+42 / +57 / +17 -> +2 / +5 / +2 ms at 128 / 1K / 4K, `logs/opt/phaseB2/B7-FIX/e3f`), but served (honest `dsamp_pk`, prefix caching off, c = 1, 2 boots x n = 10 per ISL, `logs/opt/phaseB3/B7-SERVED`) it is slower than `main`: TTFT 128 / 1K / 4K 0.213 / 1.122 / 1.912 s vs 0.213 / 1.078 / 1.889 s; TPOT 128/1024 unchanged (48.1 / 61.7 ms at c = 1 / 32) [M]. Its gates pass (serving order 3/3, prefill determinism 4 buckets x 2 processes x 5 cold repeats identical), but it is not the default. The served E3 slowdown is removed instead by the launch env `GLIBC_TUNABLES=glibc.malloc.tcache_count=65535` (row `GLIBC_TUNABLES` below), which works with `main`. `main` (the default) = the release | `MotifTTConfig.validate` / `generator_api.check_capture_thread`: `main` / `worker` / `dedicated` (case and blanks ignored) |
| `MOTIF3_PREFILL_MOE` | `compact` | B2a (`docs/OPTIMIZATION_PLAN.md` §3.3 B2; branch `motif3-opt`; prototype `logs/opt/phaseA/m7`, results `logs/opt/phaseB/B2a/report.md`): `cfg.prefill_moe`, the prefill routed experts of every MoE layer (`MotifMoE(prefill_moe=)`, `moe.resolve_prefill_moe`). `dense` = the release (each chip runs its 12 local experts on every row of the chunk). `compact` (the default since its gates passed: bitwise equal to `dense` in situ on all 51 MoE layers and 32 chips, FMV identical, CP-P / CP-L / CP-H / CP9 / determinism pass; 53-layer prefill passes of 2K / 4K rows x0.69 / x0.53, 8K x0.55, packed 4K passes x0.54, 1K solo flat) = token-compacted: per MoE chunk of at least `MOTIF3_PREFILL_MOE_MIN_ROWS` rows the host reads the routes from chip 0 (one blocking read), builds every chip's expert-sorted row lists (`moe.compact_upload_fast`) and uploads them once (uint32); each chip gathers only its routed rows (`ttnn.embedding`), runs them in blocks of `mb` rows through `ttnn.sparse_matmul` (one expert per block, `nnz` exact) and the grouped PolyNorm, and combines them with a one-hot matmul. Bitwise equal to `dense` on all 32 chips (`test_moe_device_prefill_compact`); MoE block x0.41-0.63 of dense eagerly at 1K-8K rows (real L2 / L20 / L35). Block counts come from a ladder per chunk size (`moe.compact_ladder`: x1.25 steps up to 2 x the rows); the generator compiles every ladder entry and the dense fallback in `warmup_prefill` (`MotifModel.warm_prefill_moe`), and after the decode capture an unwarmed shape or a chunk beyond the cap runs dense (`CompactPrefillState`, F3N rule R2). Needs `combine_mode="fold"`, the bf16 prefill PolyNorm and its `rms` impl (else the config default stays dense). Decode is not affected | `MotifTTConfig.validate`: `dense` / `compact` (case and blanks ignored) |
| `MOTIF3_PREFILL_MOE_BLOCK` | `auto` | B2a: rows per expert block of the compacted prefill (`cfg.prefill_moe_block`, `moe.compact_block`): `auto` = 32 for chunks up to 2048 rows, 64 above; or a fixed `32` / `64` / `128` | `MotifTTConfig.validate`: the four values |
| `MOTIF3_PREFILL_MOE_MIN_ROWS` | 1024 | B2a: MoE chunks with fewer rows keep the dense path under `compact` (`cfg.prefill_moe_min_rows`) | `MotifTTConfig.validate`: a multiple of 32, >= 32 |
| `MOTIF3_PREFILL_MOE_DISPATCH` | `device` | B2b (`docs/OPTIMIZATION_PLAN.md` §3.3 B2; branch `motif3-opt`; results `logs/opt/phaseB/B2b/report.md`; default since the Phase C flip on `motif3-opt-c`, after the NB-disagreement fix `f3fcb0db873` passed unit35, CP-L, CP-L / CP-H / CP9 ×3, determinism and CP-P bitwise equal to B2a, `logs/opt/phaseC2/B2b-FIX-VALIDATE`; served 4K/128 TTFT c = 1 1.29 → 0.77 s; set `host` / `matmul` for the B2a path): `cfg.prefill_moe_dispatch`, who builds the compacted rows of a `compact` chunk (`MotifMoE(prefill_moe_dispatch=)`, `moe.resolve_prefill_moe_kernels`). `host` = B2a (a blocking read of the routes from chip 0, numpy row lists, one upload, ~20 small unpack ops; the read drains the device: 9-10 ms of idle device per MoE layer at 4K rows). `device` = one `generic_op` (`tt/kernels/moe_compact.py` `CompactDispatch`, `moe_dispatch/dispatch.cpp`; 13 cores: one per local expert + one that counts every expert) writes the same rows on device from the router's `idx` / `w` (tokens, keys, block one-hots and PolyNorm words, the rows' routing weights) into capacity buffers of the ladder's last entry; the host reads only its 32-byte block count NB (the same ladder entry as B2a) and slices the first NB blocks. Bitwise equal to `host` and to `dense` (`test_moe_device_prefill_compact_kernels`). Only chunks of 1024 / 2048 / 4096 rows (`moe.PREFILL_MOE_DEVICE_ROWS`, the sizes validated on device) use `device`; any other chunk (a remainder such as 3968 rows, or above 4096) keeps `host`; a chunk whose chips disagree on NB runs the dense MoE (`nb_split` fallback, `f3fcb0db873`) | `MotifTTConfig.validate`: `host` / `device` (case and blanks ignored) |
| `MOTIF3_PREFILL_MOE_COMBINE` | `gather` | B2b (default since the Phase C flip, with `MOTIF3_PREFILL_MOE_DISPATCH=device`): `cfg.prefill_moe_combine`, how a `compact` chunk's rows are summed back per token (`MotifMoE(prefill_moe_combine=)`). `matmul` = B2a (one-hot `P^T [M, R] @ y`: 2.5-3.8 ms per layer at 4K rows, mostly multiplying zeros). `gather` = `tt/kernels/moe_compact.py` `GatherCombine` (`moe_combine/`): `y` untilized, each token's rows read in row order (= local expert order) as row segments, tilized and added into an fp32 dest with `fast_reduce_nc`'s ops (the dense path's expert sum), packed once: 0.05-0.24 ms. Bitwise equal to `matmul` and to `dense`. Like `device`, only for chunks of `moe.PREFILL_MOE_DEVICE_ROWS`; other chunks keep `matmul` | `MotifTTConfig.validate`: `matmul` / `gather` (case and blanks ignored) |
| `MOTIF3_PREFILL_MOE_UPLOAD` | `staged` | P2 (`logs/opt/phaseC/P2/report.md`): `cfg.prefill_moe_upload`, how a B2a (`host` dispatch) chunk uploads its per-chip row words (`MotifMoE._upload_words`). `from_torch` = B2a (one sharded `ttnn.from_torch` per MoE layer, ~0.67 ms of host time). `staged` (default; bitwise neutral) = the first call per row count is the `from_torch` upload and allocates a host mesh tensor of that spec once (zero-copy numpy views of its 32 shards, checked); later calls write the words into the views, allocate the device tensor and run one `copy_host_to_device_tensor` (the write is copied into the command queue before it returns). The same bytes reach the same ops. With P2's faster `compact_upload_fast` (16-bit radix argsort; word for word the B2a output) and `MOTIF3_SHM_TRACKING=off`: 53-layer solo prefill 1K / 2K / 4K / 8K 0.967 / 1.066 / 1.797 / 3.590 -> 0.720 / 0.949 / 1.688 / 3.405 s in process, logits bitwise equal [M] | `MotifTTConfig.validate`: `staged` / `from_torch` (case and blanks ignored) |
| `MOTIF3_PREFILL_SP` | `dp` | P4 / plan C1 (`logs/opt/phaseC/P4`, `tt/prefill_sp.py`): `cfg.prefill_sp`, the DP-row sequence split of the non-MoE prefill. `off` = the release: every DP row runs every row of the pass outside the MoE. `dp` (default; bitwise neutral) = an sp0, non-packed pass of at least `MOTIF3_PREFILL_SP_MIN_ROWS` and at most `cfg.max_prefill_span` (8192) rows keeps its residual streams split over the 4 DP rows (DP row `d` owns rows `[d S/4, (d+1) S/4)`): mHC, norms, attention projections / epilogue / AR(tp), dense MLP and shared expert run on S/4 rows; the attention all-gathers the 576-wide latent over DP (the release's cache-fill tensor, filled as before) and runs SDPA with explicit per-DP-row masks (global: causal over the gathered S rows; SWA: the three fixed 128-row tails + the row's own rows, window 129) under the release's program and compute configs, so fully masked k-chunks add exact zeros; the MoE all-gathers its input rows and skips its final AG(dp); the streams are gathered once after the last layer. sp1, packed (pk0 / pk1) and traced passes keep the release path. 53-layer solo prefill 2K / 4K / 8K 0.958 / 1.696 / 3.409 -> 0.796 / 1.261 / 2.545 s in process; logits and the KV pages of layers 0 / 26 / 52 (chips 0 and 31) bitwise equal to `off`, deterministic [M]. Masks cost 2.75 + 8 MB (4K) / 7 + 32 MB (8K) of DRAM per chip once built | `MotifTTConfig.validate`: `off` / `dp` (case and blanks ignored) |
| `MOTIF3_PREFILL_SP_MIN_ROWS` | `2048` | P4: `cfg.prefill_sp_min_rows`, the smallest pass `MOTIF3_PREFILL_SP=dp` splits (the rows must also split into four whole 256-row SDPA chunks: 2048, 3072, 4096, ...). 1K passes are host-bound: split, a 1K pass gains nothing (5-layer wrapper 56.1 -> 58.1 ms) [M] | `MotifTTConfig.validate`: a multiple of 512 |
| `MOTIF3_DEBUG_SYNC` | `off` | Phase C P1diag (`tt/debug_sync.py`, `logs/opt/phaseC/P1diag`): debug only, never for serving. `layer` = in an eager prefill pass of more than 512 rows (traced buckets never sync), `ttnn.synchronize_device` + one `[motif3.dbgsync]` log line before the first layer, after every decoder layer (`MotifModel.prefill_chunk`, release and P4 split paths), and before B2b's dispatch and gather combine of a compacted MoE chunk: a device hang leaves the last completed site in the log. Serialises host and device (slower prefill); changes no program or value. `off` = no call | `debug_sync.debug_sync_mode`: `off` / `layer` (case and blanks ignored) |
| `MOTIF3_SHM_TRACKING` | `off` | P2 (`tt/host_env.py`): `off` sets `TT_METAL_SHM_TRACKING_DISABLED=1` when the package is first imported (`tt/__init__.py`, before any ttnn import of the package) and in `open_motif_mesh`, unless the caller set that variable. tt-metal's tracker only feeds tt-smi's per-process memory view and calls `getpid()` twice per chip per buffer allocation and free: 20 % of the host time of an eager 1K prefill (py-spy), 1K 0.98 -> 0.77 s in process [M]. Host bookkeeping only (device work unchanged, logits bitwise equal). `on` = tt-metal's default (tt-smi shows this process's memory). tt-metal reads the variable once, at its first device open: set it in the launch env if something opens a device before the package is imported | `host_env.apply_host_env`: `off` / `on` (case and blanks ignored) |
| `MOTIF3_PGD` | `auto` | Phase D (`tt/host_env.py`, `tt/pgd/`): tt-metal's fabric init chooses the 4x8 placement from a physical grouping descriptor (PGD) it looks up under `$TT_METAL_HOME/tests/tt_metal/tt_fabric/physical_groupings/` or at `TT_METAL_PHYSICAL_GROUPING_DESCRIPTOR_PATH`. The ttnn wheel ships none (`TT_METAL_HOME` = `site-packages/ttnn`), so a wheel install logged `MGD placement fallback M0 (MESH)` instead of `committed: 4x8_Mesh_flat_torus_xy (TORUSXY)`. `auto` (default): when that variable is unset and the file tt-metal would pick for this board is missing under `TT_METAL_HOME`, point the variable at the byte-identical copy in `tt/pgd/` (rev C if the Blackhole Galaxy board id (kmd `tt_serial`) bits [35:32] >= 3, else rev A/B, as tt-metal's `find_and_load`). Not a BH Galaxy, unreadable sysfs or boards that disagree: unset (tt-metal's own lookup). A dev tree (files present) is unchanged. `off`: never set it | `host_env.apply_pgd_env`: `auto` / `off` (case and blanks ignored) |
| `MOTIF3_ATTN_EPILOGUE` | `fused` | D1 (`logs/opt/phaseC/D1`): `cfg.attn_epilogue`, the decode attention epilogue between the `W_UV` bmm and `wo` (`MotifAttention._absorbed_epilogue`, every decode layer and the MTP layer). `ops` = the release: `nlp_concat_heads`, the channel split, the exact 0/1 noise expansion `noise @ X`, `addcmul`, `multiply`, `where` (6 programs). `fused` = `tt/kernels/attn_combine.py`: one program on 32 cores that selects the `u` tiles directly (the concat / split / expansion are tile selections) and runs the LLK sequence of the ternary `addcmul` and the binary_ng SFPU `multiply` / `where` with their dest and unpack modes, so the `wo` input and the attention output are bitwise the op chain's (53-layer decode logits bitwise at c = 1 / 8 / 16 / 32). Traced `forward_decode` 278.6 -> 246.2 us (SWA) and 298.0 -> 269.7 us (global) per layer; 53-layer decode replay -1.3 to -1.4 ms per step [M]. `sigmoid(lam @ E)` stays its own matmul. Prefill and the sp1 absorbed path keep the ops | `MotifTTConfig.validate`: `fused` / `ops` (case and blanks ignored) |
| `MOTIF3_ATTN_IN` | `fused` | Phase F F1 (`logs/opt/phaseF/F1`): `cfg.attn_in` -> `MotifAttention.attn_in`, the decode attention input chain (every decode layer and the MTP layer, T32 and T64). `ops` = the release: the q_a / kv_a latent linears, then 15 programs (q norm, q_b, gate + sigmoid, kv split + norm, q heads split, W_UK bmm, 2 RoPEs, concats, transposes). `post` = those 15 as ONE `generic_op` (`tt/kernels/attn_in.py`) on the 12 x 10 grid. `fused` = `post` plus the two latent projections in the same program. Each stage issues the replaced ops' LLK sequences with their configurations, so q_mla, g, lam and the KV-cache input are bitwise the op chain's: unit test on all 32 chips (8 / 16 rows, 20 repeated calls), 53-layer E1 decode logits bitwise ops = post = fused at c = 1 / 8 / 16 / 32 (row and KV-R write), spec T32 / T64 gates pass. Traced `forward_decode` 241.6 -> 174.0 us (SWA), 265.3 -> 199.8 us (global); 53-layer decode replay -3.1 to -3.7 ms per step at 1 / 8 / 16 / 32 lanes (T32 all / row / spec, T64 verify) [M]. Falls back to `ops` outside the kernel contract (rope mode, grid, rows). Prefill keeps the ops | `MotifTTConfig.validate`: `fused` / `post` / `ops` (case and blanks ignored) |
| `MOTIF3_ATTN_OUT` | `fused` | Phase F F2 (`logs/opt/phaseF/F2`): `cfg.attn_out` -> `MotifAttention.attn_out`, the decode attention output chain before the AR(tp) (every decode layer and the MTP layer, T32 and T64). `ops` = the o_lat transpose, the per-head W_UV bmm, sigmoid(lam @ E) (2 programs), the attention epilogue (`MOTIF3_ATTN_EPILOGUE`) and the wo linear. `uv` = transpose + W_UV + lam expansion + sigmoid + combine as ONE `generic_op` (`tt/kernels/attn_out.py`, 12 x 10 grid), wo stays the stock linear. `fused` = `uv` plus wo in the same program (its weights streamed alongside, dg multicast once). Each stage issues the replaced ops' LLK sequences with their configurations (incl. the reuse bmm's TF32 partial reload), so the AR(tp) input is bitwise the op chain's: unit test on all 32 chips (8 / 16 rows, 20 repeated calls), `forward_decode` bitwise. Traced chain 45.9 -> 28.5 us (8 rows), 48.0 -> 28.3 us (16 rows); `forward_decode` -12.5 us (SWA), -14.7 us (global) [M]. Falls back to `ops` outside the kernel contract (grid, shapes, dtypes). Prefill keeps the ops. Experiments only: `MOTIF3_AOUT_EXP`, `MOTIF3_AOUT_WO_DELAY_NS`, `MOTIF3_AOUT_WO_LATE_NS`, `MOTIF3_AOUT_WO_N1` | `MotifTTConfig.validate`: `fused` / `uv` / `ops` (case and blanks ignored) |
| `MOTIF3_MOE_LOCAL` | `shared` | Phase F F3 (`logs/opt/phaseF/F3`; the default since every gate passed bitwise: E1 53 layers row + KV-R equal to the 0d91a513420 hashes, spec_t32 / t64, KV-R + MTP, host suite): `cfg.moe_local` -> `PolyNormMLP.moe_local`, the decode shared expert of every MoE layer (layers 2-52; production, async and the spec launch alike; the MTP layer has a dense MLP). `ops` = the Phase E path (row pad, gate_up, B5 moments, the moments all-gather, B5 apply, the down linear, a row slice). `shared` = no row pad (every op is row-local, so the input's tile-padding rows reach padding rows only), and the B5 apply, the down linear and the row slice as ONE `generic_op` (`tt/kernels/shared_tail.py`): the apply split over 5 cores (3 moment coefficient tiles + b in parallel, one h tile each, the B5 LLK sequences), h gathered and multicast once, the stock down config (8 x 4 cores, one K block) issued as the bmm kernel issues it, the T output rows written directly (padding +0). Bitwise the `ops` rows: unit test on all 51 shared experts (8 and 16 rows, trace replays). Shared expert partial 61.9 -> 44.8 us traced per layer; decode step (53 layers, trace replay) -0.73 to -1.01 ms at c = 1..32 for T32 row / all / spec and T64 [M]. Falls back to `ops` outside the contract (taps, all-reduce, T not 8 / 16 / 32, B5 off). Experiments only: `MOTIF3_STAIL_EXP` | `MotifTTConfig.validate`: `ops` / `shared` (case and blanks ignored) |
| `MOTIF3_ATTN_MM_PCS` | `tuned` | D1 (`logs/opt/phaseC/D1`): `cfg.attn_mm_pcs`, the decode attention matmul program configs (`attention.decode_matmul_program_configs`). `tuned` = `Wkv_lat` 1D multicast on 10 x 2 cores (was 5 x 4) and `wq_b` with `in0_block_w` 8 (was 4); `release` = the previous configs. Bitwise equal (the configs only move output columns between cores; module output and cache bitwise on SWA and global layers), traced `forward_decode` -4.6 us per layer [M] | `MotifTTConfig.validate`: `tuned` / `release` |
| `MOTIF3_MHC_DECODE` | `fused` | D3 (`logs/opt/phaseC/D3`): `cfg.mhc_decode`, the decode mHC site (`MHCSite`, both sites of every decode layer). `ops` = the release: projection, statistics, `finalize_mixes` (1 core), `motif_sinkhorn` (1 core), `coefficient_layout`, `attn_res_weighted_reduce_nc` (pre) and `post_mix` (7 programs; every one of ~120 mix workers read the whole weight set over the NOC: 80 KB per worker for the post mix). `fused` = `tt/kernels/mhc_decode.py`: finalize + Sinkhorn (the `sinkhorn_motif` SFPU routine, shared header `motif_mhc_sfpu.h`) + TF32 layout in ONE single-core program writing a packed 32x32 coefficient tile (16 copies), and pre / post mixes whose readers expand that tile into their weight CBs locally; the finalize sums run on the faces that reach the single-half Sinkhorn only. Same LLK / SFPU operations in the same order: x_red and X' bitwise equal to `ops`, padding rows included (`test_mhc.py::test_mhc_decode_fused`). Traced site 75.2 -> 47.9 us at 8 lanes per row; 53-layer decode replay -3.3 to -3.5 ms per step at 1 / 32 lanes (T64 verify -3.1 ms), logits bitwise at c = 1 / 8 / 16 / 32 [M]. Prefill keeps the ops | `MotifTTConfig.validate`: `ops` / `fused` (case and blanks ignored) |
| `MOTIF3_MOE_DECODE_CCL` | `ar` | D4 (`logs/opt/phaseC/D4`): `cfg.moe_decode_ccl`, the decode MoE combine collectives (`MotifMoE.forward_decode`, every MoE layer, T32 and T64). `ar` = the release: AR(dp) of the `[1, 1, 4 L, 4096]` routed partial (RS + safe AG), `partition(2, "dp")` (untilize + `mesh_partition` + tilize), `+ add_partial`, AR(tp) of the tile-padded `[1, 1, L, 4096]`. `rs` = `tt/kernels/row_fold.py`: fold each DP row's `L x 4096` block into `4 L x 1024` whole tiles (exactly the logical reshape), ONE direct reduce-scatter over DP on dim 1 (no DP all-gather), `+ fold(add_partial)` (one program, bitwise `ttnn.add`), AR(tp) on the folded rows (4x fewer tiles at L = 8), unfold. Chain 80.4 -> 53.1 us traced at L = 8 (98.7 -> 69.1 at L = 16); 53-layer decode replay -1.35 to -1.69 ms per step (T32 all / row, 1 and 32 lanes, P128 / P4096, T32-spec, T64 verify) [M]. NOT bitwise equal to `ar` (the same terms summed on other chips: 24-30 % of MoE output words differ by <= 1 bf16 ulp, identical PCC vs the fp32 reference; teacher-forced 53-layer logits: argmax flips only at reference top-1/top-2 margins <= 0.25, max KL 0.10), and **it breaks lane-position invariance**: a row's DP sum is reduced on its own DP row's chip, in an order relative to that chip, so the same request gives different bits in a different DP row (spec gates fail: `test_r5_lane_relocation` "lane relocation 5 -> 29 is not bitwise", G-S6 serving order, spec == plain, `test_t64_gs5w_lossless`; `logs/dev/20261008_082154_*`, `20261008_083025_*`). Experimental only: it must not become the default (an eval window does not fix it). Deterministic per lane position (bitwise across re-capture, 800 traced 53-layer steps; ring-race repro 0 / 4000), T64 rows bitwise == T32 rows of the same lane | `MotifTTConfig.validate`: `ar` / `rs` (case and blanks ignored) |
| `MOTIF3_AG_ROWS_LAYOUT` | `kernel` | D4 (`logs/opt/phaseC/D4`): `cfg.ag_rows_layout` -> `MotifCCL.rows_layout`, the layout changes of `MotifCCL.ag_dp_rows` (the MoE token gather of every MoE layer, the LM head's, the KV-R row gather). `ops` = `ttnn.to_layout` (untilize 4.6 us, tilize of `[32, 4096]` 11 us: one core per 32-row block). `kernel` = `tt/kernels/rm_tile.py`: the same byte moves as data-movement kernels on 64 cores (bf16 TILE <-> L1 ROW_MAJOR; other dtypes keep the ops), and `MotifCCL.partition`'s 8 / 16-row TILE slice (the decode MoE combine after AR(dp): untilize + `mesh_partition` + tilize, 12.8 us) as `rm_tile/pick_rows.cpp` (two NoC reads per tile; the chip's DP index from a per-chip uint32 tensor allocated when the MoE is built). Bitwise equal (53-layer logits bitwise at c = 1 / 8 / 16 / 32, row and KV-R write); MoE token gather 28.5 -> 19.2 us, decode replay -0.45 to -0.52 ms per step (row write), -1.13 to -1.17 ms (KV-R write) [M] | `MotifTTConfig.validate`: `kernel` / `ops` (case and blanks ignored) |
| `MOTIF3_DECODE_EXPERT_MM` | `fused` | Phase D DESIGN-3 stage 1 (`logs/opt/phaseD/D3BUILD`): `cfg.decode_expert_mm` -> `MotifMoE.decode_expert_mm`, the two expert matmuls of the sparse decode path (`decode_experts=sparse`, M = 32 / 64). `stock` = `ttnn.sparse_matmul` (1D in0 multicast, weights read on one RISC / NoC per core, plus a `zeros_like` of the output). `dualnoc` = `tt/kernels/moe_sparse_mm.py`: one `generic_op` per matmul, (active expert, column) units over all 120 cores, each RISC streams one K half on its own NoC, gate_up `x` multicast by core 0, inactive slices zero-filled in-kernel. Bitwise equal to `stock` (kernel unit test incl. k = 0 and M = 64, 53-layer E1 logits at c = 1 / 8 / 16 / 32 row and KV-R write, spec T32 / T64 gates). Kernel: gate_up 34.9 -> 25.6 us / active expert, down 17.7 -> 13.6 [M]; 53-layer decode replay -4.3 to -4.5 ms at 32 lanes, -2.6 at 8, -1.2 to -1.3 at 1 (P4096, real contexts), T32-spec -4.4, T64 verify -4.8 [M]. `fused` (Phase E, D3 stage 2, `logs/opt/phaseE/D3S2`): `dualnoc` plus the sparsity built inside the gate_up kernel from the lane-masked routing weights (no `decode_sparsity` ops), the expert sum inside the down kernel (column owners add the bf16 expert tiles in expert order into the fp32 DEST like `fast_reduce_nc`; no `y` tensor, zero-fill or reduce op) and the down `h` read once per expert and multicast to its workers. Bitwise equal to `dualnoc` / `stock` (kernel unit test 300 cases, traced soak with every replay checked); per MoE layer (gate_up + down + the removed ops, M = 32, traced) intercept 53.8 -> 18.1 us, slope unchanged [M]; 53-layer E1 bitwise vs `dualnoc` at c = 1 / 8 / 16 / 32 (row and KV-R write), spec T32 / T64 gates pass; decode replay -1.8 to -1.9 ms at 1 / 8 / 16 / 32 lanes, T64 verify -3.3 [M]. Default since the D3S2 gates | `MotifTTConfig.validate`: `dualnoc` / `stock` / `fused` (case and blanks ignored); `resolve_decode_expert_mm` falls back to `stock` without the sparse path or bfp8 experts |
| `MOTIF3_MOE_REPLICAS` | `r4` | Phase E DESIGN-2 (`logs/opt/phaseE/DESIGN2`, `tt/replicas.py`): `cfg.moe_replicas` -> `MotifMoE.moe_replicas`, decode only. `r4` = EP32 + 4 replica slots per chip: every home chip donates its 4 most-routed experts (real routes, plan `tt/replica_plan_r4.json`, `MOTIF3_MOE_REPLICA_PLAN` overrides) to chips h + 1 + 7 j; each step and MoE layer the fused router tail (`kernels/router_topk.py` replica mode) runs the same deterministic greedy assignment on every chip (each active expert to the less loaded of its 2 holders) and writes the lane-masked routing weights of the 12 + 4 slots this chip computes; gate_up / fused PolyNorm / down_sum run over 16 slots (`DualNocSparseMM(w_rep=...)` reads the replica slots from a second weight tensor). Replica weights: byte copies of the cached experts (no requantization) built per layer into `MOTIF3_MOE_REPLICA_DIR` (default `/dev/shm/motif3_replicas`) and deleted after the upload unless `MOTIF3_MOE_REPLICA_KEEP=1` (then reused, resumable per layer); +3.4 GB DRAM per chip. **Not bitwise equal** to `off` (a reassigned expert lands in another chip's partial; module PCC vs `off` >= 0.999994, max |d| <= 1.1e-2 of the row max); with every replica code removed it is bitwise `off`. A row's result depends on the step's other rows, so T64 verify rows differ from T32 rows: an MTP launch (spec_tokens > 0) resolves it to `off`. Needs `router_mask=fused`, `decode_experts=sparse`, `decode_expert_mm=fused`, `moe_polynorm=fused`; a config that does not meet this gets `off` (logged), as does a module built without the TT cache or on a mesh other than 32 chips x 12 experts. Default `r4` since the eval passed (`logs/opt/phaseE/D2EVAL`, production launch with `r4`: MATH-500 99.0, AIME24 86.7, GPQA-D 85.0 / 85.0 / 87.5, pooled 85.83 vs production 84.58 and bar 79.23, every task inside the production 95 % Wilson interval [M]); production and async launches use it, MTP launches stay `off` (T32 != T64 rows). Boot loads about 2.1 GB of tmpfs per layer transiently; `off` restores the bitwise D3S2 path | `MotifTTConfig.validate`: `off` / `r4`; `moe.resolve_moe_replicas` |
| `MOTIF3_RING_GATHER` | `safe` | `cfg.ring_gather` (`MotifCCL`, §8): `safe` routes every race-prone TP-ring all-gather through `all_broadcast` + `concat` (lead decision 2026-10-03, B0; +0.26-0.45 ms per decode step); `lean` keeps the single-page decode gathers native; `native` is the plain `ttnn.all_gather`, which races (`docs/determinism/INVESTIGATION.md`; FEATURES_REVIEW P1 and F3) | `MotifTTConfig.validate`: the three values, and `MOTIF3_SPEC_VERIFY=wide` / `auto` with speculation refuse anything but `safe` (F3N rule R1); `_check_wide_launch` re-checks the config and the model's `MotifCCL`; every other launch logs a warning once |
| `MOTIF3_FLASH_MLA_SWA_MCPH` | 4 | A2 (`docs/OPTIMIZATION_PLAN.md` §3.3; branch `motif3-opt`): FlashMLA decode `max_cores_per_head_batch` of the 39 SWA layers (`cfg.flash_mla_swa_mcph`, `cfg.flash_mla_decode_pc("swa")`, `MotifAttention.decode_pc`; MTP layer included). The 14 global layers keep 16 at every setting. 4 is bitwise equal to the release's 16 on SWA (probe `logs/opt/phaseA/A2`: T32 plain launch, row and sampled paths, B32 / B1, P = 128-16384; G-T64 and the spec gates still to run) and saves 0.18-0.26 ms per step; `16` restores the release config | `model_config.check_flash_mla_mcph` (`MotifTTConfig.validate`): an integer in [1, 16] |
| `MOTIF3_GENERATOR_CLASS` | `models.demos.motif3.tt.generator:MotifGenerator` | runtime class for the bridge | `generator_vllm._resolve_generator_class`: `module:Class`, a `MotifGenerator` subclass |
| `MOTIF3_PREFIX_CACHING` / `MOTIF3_CHUNKED_PREFILL` / `MOTIF3_SPEC_DECODE` | on | bridge class capabilities (§15-§17); they only allow a feature, vLLM's flags enable it. `MOTIF3_*=0` alone is **not** draft 1 for prompts over 8192 tokens: the span cap still splits them (sp0 + sp1). Draft-1 behaviour is the pair `MOTIF3_*=0` + `MOTIF3_PREFILL_MAX_BUCKET=32768` (below) | `generator_api.feature_switch_from_env`: `1/true/yes/on` or `0/false/no/off`, a typo raises |
| `MOTIF3_DEVICE_SAMPLING` | on | bridge class capability `supports_sample_on_device` (never with `max_device_top_k`). It only allows device sampling; the `"tt"` config's `"sample_on_device_mode": "decode_only"` enables it (the production launch below). `0` and no `sample_on_device_mode` is draft 1's host sampling (rollback) | `generator_vllm.device_sampling_switch`: a typo raises; with the switch at `0` the plugin refuses `sample_on_device_mode` at boot |
| `MOTIF3_SAMPLING_LOG_EVERY` | 2000 | decode steps between the bridge's `Motif-3 device sampling: {...}` JSON lines; `0` = only at shutdown | none: a value that is not a decimal integer falls back to 2000 (`MotifForCausalLM.__init__`) |
| `MOTIF3_KV_REPLICATED_DECODE` | `auto` | KV-R (§16): `auto` (on iff prefix caching), `1` (forced on), `0` | `kv_replicated_decode_from_env`; `GeneratorSettings` refuses `0` with prefix caching |
| `MOTIF3_PREFILL_MAX_BUCKET` | 8192 | span cap (§15): largest prefill bucket of a resumed-prefill generator; longer spans are split into chunks. `32768` restores the draft-1 single-shot buckets (rollback, with `MOTIF3_*=0`) | `check_prefill_span_cap`: a power of two in [128, 32768] |
| `MOTIF3_CHUNK_BUDGET` | unset (= `auto`: 8064) | A1a (`docs/OPTIMIZATION_PLAN.md` §3.3; branch `motif3-opt`): the chunk budget (= long-prefill threshold) the launch means to run. `generator_vllm.launch_chunk_budget` = `prefill_plan.recommended_budget(span cap, A, target)`: the target rounded down to a multiple of A = 128 and capped at span cap - A (`4096` stays 4096: every chunk of a lone prompt is one 4096 bucket, already in the warm-up compile set; `4000` -> 3968). It does **not** change vLLM's budget: the launcher passes `--max-num-batched-tokens` / `--long-prefill-token-threshold` (`generator_vllm.feature_vllm_args`, `P5T64_BUDGET` of `p5t64_hold.sh`, the TIS spec `vllm_args`), and `check_serving_config` warns at boot when those differ from it. 8064 stays the code default until the TIS A/B (plan §6 M2) decides | `generator_api.chunk_budget_from_env` / `check_chunk_budget`: `auto` or an integer in [128, 32768]; `recommended_budget` refuses a target below A |
| `MOTIF3_PACKED_PREFILL` | `0` in the code; **`1` in both TIS specs** (`motif3_galaxy`, `motif3_galaxy_mtp`; TIS `f0484e96`) | packed multi-row prefill (P5, §18): the short chunks of one prefill call run as packed passes. With it on, a request's greedy and seeded tokens depend on which requests share its pass (near-tie flips at the bucket floor; the lead signed this contract off on 2026-10-04, `docs/P5_T64_REVIEW.md` §8, I-1). `0` restores per-row prefill (under TIS only through a runtime spec JSON, `docs/TIS_RUNBOOK.md` §4.2) | `generator_api.packed_prefill_from_env`: `1/true/yes/on` or `0/false/no/off`, a typo raises |
| `MOTIF3_PACKED_PREFILL_MAX_SEG` | 1024 | P5: the largest packed segment S (one of 64 / 128 / 256 / 512 / 1024; caps the pk0 and pk1 segment sizes); a chunk of more rows runs solo | `check_packed_prefill_max_seg` (`GeneratorSettings`, `MotifTTConfig`) |
| `MOTIF3_PACKED_PREFILL_MAX_TOKENS` | 8192 | P5: the largest packed pass T = B × S (also bounded by the span cap) | `check_packed_prefill_max_tokens`: a power of two in [128, 32768] |
| `MOTIF3_PACKED_PREFILL_PK1` | on | P5: also pack resumed (sp1) chunks that share a start (pk1, both SWA tail variants); off, they run solo (gate G15a's fallback) | `packed_prefill_pk1_from_env`: a typo raises |
| `MOTIF3_PACKED_WARMUP` | `attention` | P5 warm-up per packed shape: `attention` = the attention of one global and one SWA layer on zeros (56 shapes in 2.2 s per boot, `docs/P5_T64_RESULTS.md` §3.1); `full` = one full packed pass per shape (~60-70 s; for program-cache growth) | `packed_warmup_from_env`: `attention` / `full` |
| `MOTIF3_SPEC_VERIFY` | `packed` in the code; **`auto` in the MTP TIS spec** (`motif3_galaxy_mtp`) | verify mode of a speculating launch (no effect without `--speculative-config`; §18): **`packed`** = one T32-spec decode trace, drafts on idle lanes, drafts without one in a second replay (overflow pass); **`wide`** = the 64-row T64 trace alone serves every step (ordinary steps with idle draft rows, the device sampler on its anchor rows; the one-trace fallback); **`auto`** = both traces, each captured once at warmup (T32 first): T32 for ordinary and sampled steps and for verify steps whose drafts fit idle lanes, one T64 replay for every other verify step | `spec_verify_from_env`: `packed` / `wide` / `auto`, a typo raises. `wide` / `auto` are refused unless `ring_gather="safe"` on the config and on the model's `MotifCCL` (`MotifTTConfig.validate`, `MotifGenerator._check_wide_launch`; F3N rule R1, review edit R-E5), with a split KV-write mode and the LM head's `mesh` vocab split (`_check_wide_launch`). `auto` + `MOTIF3_ROUTER_LOGITS=exact_fp32` is accepted at rows (32, 64) since B1 (see that row) |
| `MOTIF3_WIDE_MIN_LANES` | unset (the generator's c*) | `auto` only: the live-lane count from which every live lane drafts (33 = never). Unset: c* = `verify_plan.crossover_lanes(alpha, cfg.wide_step_ratio)` from the running acceptance pulled toward the prior 0.85 and r = 1.21 (`MOTIF3_WIDE_STEP_RATIO`), clamped to [17, 33]: 20 at the prior (19 with the release's r 1.13). A4' (`docs/OPTIMIZATION_PLAN.md` §3.3, M14) sweeps it at 8 / 16 users on the MTP launch (re-run with B1 sparse, r 1.21, c* 20 on 2026-10-07, `logs/opt/phaseB2/B1-FLIP/tis`: unset still lost drafts on 16.9 % of tokens at 1024/1024 x 16, t/s 246 vs 297 with 8 or 16; every other point of {128, 1024}/1024 x 8 / 16 / 20 / 32 equal within 1-2 %; recommended 8 on the honest MTP launch): from n live lanes every live lane drafts, and a verify step whose drafts do not fit idle lanes runs on T64 (on the honest launch, `row_split`, a DP row full of users has no idle lane: below c* those users do not draft at all); not a TIS spec key, so a shell export reaches the server (`test_verify_plan.py::test_wide_min_lanes_sweep_a4prime`) | `check_wide_min_lanes`: [1, 33] |
| `MOTIF3_WIDE_STEP_RATIO` | unset (`model_config.DEFAULT_WIDE_STEP_RATIO` = 1.13) | `auto` only: the T64 / T32-spec step-time ratio r of c* (`GeneratorSettings.wide_step_ratio` → `MotifTTConfig.from_settings` → `cfg.wide_step_ratio`; scripts that build the config with `from_hf_config` do not read it). At the prior 0.85, c* is 17 / 19 / 20 / 22 for r = 1.0 / 1.13 / 1.2 / 1.3, and 33 (never) once r - 1 reaches the acceptance. The default 1.21 is G16 on `motif3-opt` with B1 sparse + B3 / B4 fused (TORUS_XY, 2026-10-07: 1.208 / 1.218 / 1.246 at 1K / 8K / 32K `all_split`; `logs/opt/phaseB2/B1-FLIP`); the release used 1.13 (G16 1.119 / 1.134 / 1.174, `tests/unit/gates/GATES_RESULTS.md` §13.7; dense experts on this tree: 1.111 / 1.129 / 1.176). Set this after re-measuring r with other decode kernels or another fabric (`MOTIF3_DECODE_EXPERTS=dense`: about 1.12). Unset gives exactly the config without it; the `create:` line shows r on a T64 launch | `generator_api.check_wide_step_ratio` (`GeneratorSettings`), the rule of `MotifTTConfig.validate`: finite and >= 1 |
| `OMP_WAIT_POLICY` | unset (both TIS specs: `PASSIVE`) | not used by the model: `MotifGenerator.create` warns on a speculating launch without `PASSIVE` (torch's spinning OpenMP workers added 4.7 ms per verify step) | none (a warning) |
| `GLIBC_TUNABLES` | unset; **recommended launch env `glibc.malloc.tcache_count=65535`** (B7-SERVED, `logs/opt/phaseB3/B7-SERVED`) | not read by the model: glibc's per-thread malloc cache. With the default 7 chunks per size class, the host-dispatch-bound eager prefill (thousands of small ttnn host objects per pass) keeps going to the main arena, which the decode / B7 trace captures fragment (E3, `MOTIF3_CAPTURE_THREAD` row). With 65535 the serving thread recycles its own chunks. Served on the honest `dsamp_pk` launch with the motif3-opt defaults (c = 1, 2 boots x n = 10 per ISL) [M]: TTFT 1K 1.078 -> 0.947 s (-12 %), 4K 1.889 -> 1.859 s, 128 0.213 s unchanged; a c = 8 burst of 60-128-token prompts 1.59 -> 1.45 s (first round); TPOT 128/1024 unchanged (c = 1 48.1 -> 47.9 ms, c = 32 61.8 -> 61.9 ms); greedy texts identical (6 prompt lengths); serving order and prefill determinism gates pass. Cost: EngineCore RSS +0.8 GB (6.07 -> 6.6-7.0 GB, levels off after ~2 load rounds). Combined with `MOTIF3_CAPTURE_THREAD=dedicated` it is no faster (1K 0.941 s). Set it in the launch environment (TIS: the spec's env, a runtime spec JSON until a TIS commit is approved); it must reach the EngineCore process (vLLM's spawned workers inherit it) | none |

Serving flags: `--block-size 64` (32 also allowed), `--max-model-len 32768` (a multiple of 256), `--max-num-seqs 32`,
`--additional-config '{"tt": {"trace_mode": "decode_only", "trace_region_size": 268435456, "fabric_config":
"FABRIC_2D_TORUS_XY", "dispatch_core_axis": "col", "l1_small_size": 32768}}'` (`generator_api.SERVING_TT_CONFIG`; TIS:
`override_tt_config`). Without `l1_small_size` the bridge refuses to start (`get_max_tokens_all_users`, in
`init_device`, before the weights load) and refuses a mesh with less L1_SMALL (`initialize_vllm_model`).

Launches (lead decisions; TIS dev specs `motif3_galaxy` and `motif3_galaxy_mtp` at TIS `f0484e96`,
`docs/TIS_RUNBOOK.md` §1). Every number below was measured on this host with TORUS_Y committed instead of TORUS_XY
(§1; `docs/P5_T64_REVIEW.md` I-5), through direct `vllm serve` (`docs/P5_T64_RESULTS.md`, "RESULTS" below;
throughput: greedy, thinking on, 256 tokens):

| Launch | vLLM flags and env on top of the above | Use |
|---|---|---|
| **production default** (`motif3_galaxy`) | `--enable-chunked-prefill --max-num-batched-tokens 8064 --long-prefill-token-threshold 8064 --enable-prefix-caching --no-async-scheduling`; `"tt"` adds `"sample_on_device_mode": "decode_only"` (device sampling; the bridge must declare `supports_sample_on_device`, no `max_device_top_k`) and `"decode_interleave_prefill_steps": 1, "decode_interleave_decode_steps": 1`; env `OMP_WAIT_POLICY=PASSIVE` **`MOTIF3_PACKED_PREFILL=1`** (§18) | all traffic. A burst of 32 short prompts gets its first tokens after 2.42 s instead of 22.66 s per row, 32 prompts behind a shared 2K system prompt after 5.29 s instead of 25.75 s, and a decoding request stalls at most 1.70 s instead of 22.0 s (RESULTS §3.2). A request's tokens then depend on which requests share its prefill pass (§18, I-1) |
| **MTP opt-in** (`motif3_galaxy_mtp`) | the production default + `--speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 1}'` + env **`MOTIF3_SPEC_VERIFY=auto`** (T32-spec and T64 decode traces, §18) | greedy / agentic / low-concurrency serving and the TIS greedy benchmarks. Greedy rows decode 1.89-1.91x faster than without MTP at c <= 16 and 1.70-1.71x at c = 20-32 (c = 32: 559.2 vs 327.9 tok/s), with greedy and seeded tokens identical to the launch without speculation (RESULTS §3.4-§3.5). Sampled traffic gets no speedup, and while a sampled request is live nothing speculates (PS-1); sampled TPOT 93.4-95.0 ms against 95.1-96.4 ms without MTP (RESULTS §3.6). vLLM refuses sampled `min_p` and `logit_bias`, the plugin `logprobs`, structured output, `bad_words`, `allowed_token_ids` and `min_tokens` on it |
| honest benchmark | either of the above + `--no-enable-prefix-caching` (KV-R then off: decode skips its ~1.9 ms; with MTP, `auto` runs T64 with `row_split` writes, gates G16 and G-S5w (iv) in `tests/unit/gates/GATES_RESULTS.md` §13.7) | `vllm bench` repeats identical prompts: prefix hits would flatter TTFT |
| P5 off (per-row prefill) | either of the above with `MOTIF3_PACKED_PREFILL=0` (under TIS a runtime spec JSON, `docs/TIS_RUNBOOK.md` §4.2: a shell export does not override a spec value) | outputs that do not depend on the concurrent traffic (A/B against earlier results, sample-level reproducibility); bursts prefill row by row again |
| T64 off (`packed` verify) | the MTP launch with `MOTIF3_SPEC_VERIFY=packed` (TIS: a runtime spec JSON) | one T32-spec decode trace, drafts only on idle lanes: identical to `auto` at c <= 16, 1.53x / 1.30x / 1.02x at c = 20 / 24 / 32 (RESULTS §3.4) |
| rollback (draft-1 behaviour) | `MOTIF3_PREFIX_CACHING=0 MOTIF3_CHUNKED_PREFILL=0 MOTIF3_SPEC_DECODE=0 MOTIF3_DEVICE_SAMPLING=0 MOTIF3_PREFILL_MAX_BUCKET=32768`, `MOTIF3_PACKED_PREFILL` unset or `0`, the draft-1 flags (`--max-num-batched-tokens 32768 --no-enable-prefix-caching`, no speculative or interleave keys) and no `sample_on_device_mode` (host sampling); keep `OMP_WAIT_POLICY=PASSIVE` | A/B against draft 1: the `docs/SERVING_RESULTS.md` server also ran with PASSIVE (its §4), and the TIS rollback cannot drop it (a spec env value overrides a shell export). Against a draft-1 launch without it, the rollback is 2.5-4 ms per step faster (`docs/SERVING_SMOKE.md` §7.2) |

The budget is `prefill_plan.recommended_budget(8192, A)` = 8064 for A = 128 (§15); the bridge's `FEATURE_VLLM_ARGS`
derives from `generator_api.DEFAULT_PREFILL_ALIGNMENT` (128) and re-checks against the generator's real A at init.
`MOTIF3_CHUNK_BUDGET=4096` (A1a) gives 4096 / 4096 through `generator_vllm.feature_vllm_args` (TIS: a runtime spec
JSON with the two `vllm_args` overridden, `logs/opt/phaseA/tis/`).

## 15. Resumed and chunked prefill (`tt/prefill_plan.py`; `docs/features/FEATURES_DESIGN.md` §2, §3.1-§3.3, §3.7)

* **Contract** (`generator_api.PrefillRequest`): a prefill row is `PrefillRequest(lane, tokens, page_table, start)` with
  `tokens` = ALL tokens at positions `0 .. end-1` and positions `[0, start)` already in the cache (vLLM
  `num_computed_tokens`: a prefix-cache hit, a block multiple, or the end of the previous chunk, any integer). Full
  blocks below `floor(start / bs)` are **read-only** (they may be shared); the generator writes `[floor(start / bs) *
  bs, end)` and bucket padding only into the request's own last block. The bridge makes **one**
  `generator.prefill_forward_batch(requests)` call per plugin step (logits `[B, vocab]` of each row's `end - 1`, in
  input order). `start = 0` everywhere is draft 1 (the ABC default runs the rows through `prefill_forward`).
* **Every alignment and table rule lives in `prefill_plan.py`** (pure torch, host-tested exhaustively by
  `tests/unit/test_prefill_plan.py`). Modules never re-derive them; `cfg.plan_prefill_row(start, end)` is
  `prefill_plan.plan_prefill_row` with the config's geometry.

  | Quantity | Rule | Default |
  |---|---|---|
  | write floor `w0` | `floor(s / bs) * bs`: full blocks below it are never written | |
  | resume alignment `A` | `cfg.prefill_resume_alignment = lcm(bs, q, k)` over the span buckets of the sp1 global op (`prefill_plan.resume_alignment`; per-bucket q/k from gate G9: `model_config.SP1_GLOBAL_CHUNKS` = `prefill_plan.DEFAULT_SP1_GLOBAL_CHUNKS`, 128/128 at C = 128 and C >= 2048, 64/64 at 256-1024). A **bf16** latent cache uses 64/64 everywhere (`SP1_GLOBAL_CHUNKS_BF16_KV`, A = 64; `cfg.sp1_global_chunk_table`, and attention picks by the cache tensor's dtype): at 128/128 the op's static CBs end at 1,572,480 B, through the L1_SMALL region of the CCL semaphores, which tt-metal does not check | 128 (bfp8) |
  | compute floor `c0` | `floor(s / A) * A`, or 0 when that is below the SWA tail `cfg.prefill_swa_tail` (128) | |
  | span cap | `cfg.max_prefill_span` (`MOTIF3_PREFILL_MAX_BUCKET`), buckets `cfg.prefill_span_buckets` (128 ... 8192) | 8192 |
  | chunks | consecutive, `A`-aligned starts; full-cap chunks while more than the cap remains; then one padded chunk, or a head chunk + the rest when the cost model says it is cheaper (`cfg.prefill_cost_table` per bucket + gate G9's per-bucket price of an sp1 chunk's global attention over its prefix, `prefill_plan.DEFAULT_SP1_GLOBAL_COST`) | 16,736 -> 8192 + 8192 + 512 |
  | path | **sp0** (start 0: the draft-1 square causal SDPA over the chunk's own rows, no cache reads) / **sp1** (start > 0: global layers attend over the paged latent from the chunk start; SWA layers read the 128-row tail from the cache) | |

* **Tables per chunk** (`prefill_plan.chunk_tables`): fill table `[C / bs]` with **-1** (skipped by
  `paged_fill_cache`) for shared blocks (entirely below `w0`) and pure-padding blocks; sp1 SDPA table `[W']`
  (`cfg.sp1_page_table_width` = 640) with the real block ids, then **0** -- **never -1 in an SDPA table** (the SDPA
  reader maps every entry as a block id); SWA tail block ids `[128 / bs]` of positions `[a - 128, a)`; RoPE rows
  `min(a + i, max_model_len - 1)` gathered from the ROW_MAJOR tables. The start, block ids, RoPE rows and the LM-head
  row are device tensors, so programs depend on `(path, bucket)` only. Every sp1 chunk start is a multiple of the
  sp1 op's q/k chunk (the kernels divide it by `q_chunk` without a device check): never start an sp1 call anywhere
  else (assert `start % A == 0`).
* **Order inside a chunk**: global sp1 fills the chunk's latent **before** the chunked SDPA (its own keys come from the
  cache); global sp0 and every SWA call fill after their SDPA, as draft 1. **Order of rows**: writer-first
  (`prefill_plan.order_prefill_requests`): vLLM caches a row's full blocks when it allocates them, so a request
  admitted later in the same step can hit blocks another row of the same call computes.
* **No per-lane or per-slot state across chunks**: a request may change lane between chunks; the paged cache is the
  only cross-chunk state.
* **Warmup** (§10 rule 6): every `(path, bucket)` with `bucket <= cfg.max_prefill_span` (sp0, sp1, and the MTP KV-only
  fill with speculation) compiles before the decode capture; the generator refuses any other shape afterwards. The
  16K / 32K buckets are no longer compiled (the planner splits those spans). With packed prefill on, the 56 packed
  shapes follow (§18); after the capture a pass of an unwarmed packed shape runs solo instead of being refused.
* **vLLM flags**: `--max-num-batched-tokens` = `--long-prefill-token-threshold` = `span cap - A`
  (`prefill_plan.recommended_budget`: **8064** for A = 128; 8128 only for an all-64/64 table): a lone prompt's chunk
  ends are then multiples of `A` (no recompute) and every span fits one bucket. A prefix hit at an odd multiple of 64
  recomputes 64 rows per continuation (the price of the uniform A = 128). The bridge runs
  `prefill_plan.check_scheduler_config` at `init_device`: it raises on prefix caching without KV-R or with a
  `--prefix-match-unit` other than the block size, and warns on unaligned budgets / thresholds, spans over the cap and
  budgets below `min(4096, span cap) - A` (vLLM's unpinned `vllm serve` default on TT is 2048); each warning names
  the change that clears it. Every span cap `MOTIF3_PREFILL_MAX_BUCKET` allows is accepted: at 128 with A = 128 no
  budget fits a resumed chunk plus its recompute in one bucket, so the span-cap warning says to raise the cap.

## 16. Decode KV writes: KV-R and the speculative split (`tt/kv_write.py`; features design §3.4-§3.5)

* **KV-R** (`cfg.kv_replicated_decode` = `GeneratorSettings.kv_replicated`): with prefix caching on, every decode KV
  write (53 layers + the MTP layer) lands on **all 32 chips**. Without it, decode writes a lane's latent only on its DP
  row, so a prefix hit on such a block from another row (multi-turn chat, preemption-resume on another lane) reads
  stale KV, and the replicated prefill plus the MoE reduce-scatter spread the error to every row.
* **Modes** (`cfg.kv_write_mode`, `generator_api.kv_write_mode` / `KV_WRITE_MODES`): one per server, shared by every
  layer, fixed when the decode trace is captured.

  | Mode | When | Write per layer |
  |---|---|---|
  | `row` | draft 1 (no prefix caching, no speculation) | one 8-lane `paged_update_cache` per DP row |
  | `row_split` | speculation without prefix caching | two 8-lane calls: (A) anchors and plain lanes, (B) draft lanes |
  | `all` | prefix caching (KV-R) | `ccl.ag_dp_rows` of the `[1, 1, 8, 576]` latent, one 32-lane update on every chip |
  | `all_split` | prefix caching + speculation (production) | the gathered latent, two 32-lane calls (A, B) |

* **Why two calls**: `paged_update_cache` runs one user per core and read-modify-writes the whole 32-row tile, so two
  users writing `p` and `p + 1` of one block in one call can lose an update (gate G12). On ordinary steps call B is all
  -1. FlashMLA runs after both calls.
* **Guarantees** every path keeps (features design §3.4): (G1) a block's KV depends only on its token prefix and
  absolute positions; (G2) full blocks below `w0` are never written; (G3) writes precede reads inside a call; (G4)
  KV-R; (G5) padding only into the own last block; (G7) nothing compiles after the decode capture; (G8) the MTP cache
  holds an entry for every prefilled and decoded position.

## 17. MTP layer and speculative decoding (`tt/mtp.py`; features design §3.6, §3.8)

* **Layer spec**: `cfg.mtp_layer_spec()` = `LayerSpec(53, is_global=False, is_moe=False, window=129,
  softmax_scale=0.07216878, rope_kind="plain")` (reference `MotifMTP`: SWA in "all" mode, dense MLP with output scale
  `cfg.polynorm_output_scale_for_layer(53)` = 0.5, no mHC). `cfg.layer(53)` does not exist and `cfg.is_moe_layer(53)`
  would say MoE: pass the spec explicitly. `cfg.mtp_layer_idx` = 53 = `num_hidden_layers` (also in truncated runs);
  weights `model.mtp_layers.0.*` (shard 104, BF16 on disk); TT-cache part `L53`.
* **Cache**: one more `[N, 1, bs, 576]` latent cache in the KV dtype, indexed by the same vLLM block ids, so it travels
  with prefix hits and is freed with the request. `cfg.kv_pool_layers` = 53 + `cfg.mtp_kv_layers`; vLLM keeps
  accounting 53; +612 B per token per chip in bfp8 (`cfg.kv_pool_bytes_per_chip()`).
* **Prefill is KV-only**: the MTP KV at `p` depends only on `(hn_p, t_{p+1})`, so prefill runs no MTP attention,
  MLP or head. A row's last position uses the host argmax of its logits as `t_{p+1}`.
* **Decode** (`settings.spec_tokens = 1`): every decode step calls `generator.decode_forward_spec(SpecDecodeBatch,
  want_logits=)`. The batch is in owner-lane order; `draft_tokens[l] >= 0` asks for the draft to be verified at `n +
  1`; lanes with `positions == -1` are idle and may host drafts. The result is `SpecDecodeResult(logits or None,
  argmax [32, 2] = (a0, a1), mtp_argmax [32, 2] = (m0, m1))`, checked with `generator_api.check_spec_result`. With
  `MOTIF3_SPEC_VERIFY=packed` one decode trace (the T32-spec trace) serves ordinary, verify and overflow steps; `auto`
  (the MTP spec's mode) adds the 64-row T64 trace for the verify steps whose drafts do not fit idle lanes, and `wide`
  serves every step from T64 (§18). The bridge contract above is the same in every mode.

## 18. Packed prefill (P5) and the 64-row speculative verify (T64) (`docs/p5_t64/P5_T64_DESIGN.md` §2-§4)

Shipped in B1 `d3597ae4977` (2026-10-04). Gates: `tests/unit/gates/GATES_RESULTS.md` §13 (op level) and §13.7 (model
level); serving: `docs/P5_T64_RESULTS.md`; review and the lead's decisions: `docs/P5_T64_REVIEW.md` §8.

* **P5 switch.** `MOTIF3_PACKED_PREFILL` (§14): off in the code, on in both TIS specs. The bridge is unchanged: one
  `prefill_forward_batch` call per plugin step and the same per-row logits; only the generator's passes change.
* **Passes** (`prefill_plan.plan_prefill_passes`, host only, exhaustively tested in `tests/unit/test_prefill_plan.py`).
  A chunk of `r` rows becomes a segment of `S` rows, the smallest of `cfg.pack_seg_buckets` (64 ... 1024) or, for a
  resumed chunk, `cfg.pack_sp1_seg_buckets` (128 ... 1024) that holds it. Segments whose dependencies have run (the
  row's previous chunk; for an sp1 segment, every writer of its row's read-only prefix `[0, w0)`, review edit R-E1)
  group as **pk0** (start 0, by `S`) or **pk1** (one common start `a`, by `(S, a)`). Groups are cut into passes of `B`
  segments (`B` in 2, 4, 8, 16, 32; dummy segments fill it), `T = B * S` rows, a compiled bucket <= 8192. Everything
  else runs **solo**: bitwise the per-row path of §15, in writer-first order.
* **Device dataflow of a pass.** Every op runs once at bucket `T`; only the attention treats the segments apart: pk0
  one batched causal SDPA (a view + transpose of `[1, H, T, d]`), pk1 global one batched chunked SDPA with one
  page-table row per segment, pk1 SWA `[tail || segment]` squares with the tails gathered once (`shared`) or per
  segment (`distinct`, review edit R-E2). The RoPE rows, the fill table and the LM-head row are per segment; the MTP
  KV-only fill runs once per pass. Gate G15 found every segment bitwise equal to the per-row attention at the
  planner's bucket.
* **Warm-up and refusals.** The 56 packed shapes of the production geometry (`cfg.packed_prefill_shapes()`: pk0
  `(T, S)`, pk1 `(T, S)` x both tail variants) compile after the 14 solo shapes and before the decode capture
  (`MOTIF3_PACKED_WARMUP=attention`: 2.2 s per boot). After the capture a pass of an unwarmed shape runs solo
  (`packed_solo_fallbacks`), and a packed plan that fails its own checks makes the call run unpacked
  (`packed_plan_errors`). Packing never refuses a call; both counters were 0 on every launch of
  `docs/P5_T64_RESULTS.md` ("RESULTS", §3.8).
* **The batch-dependence contract (P5_T64_REVIEW I-1, signed off by the lead on 2026-10-04).** A packed row runs the
  row-local programs (the MoE above all) at `M = T` instead of its own bucket, so its numerics are those of the
  pass's bucket: the "floor". Gate CP-P: KV rows of layers 0-2 bitwise per request, last-token PCC against the
  per-row path 0.9965-0.9985 (median per case), the same as one row run per-row at bucket `T`; the same packed call
  twice bitwise (GATES_RESULTS §13.7). Through vLLM every greedy divergence starts at a near tie (both tokens in the
  top 3, <= 1.125 nats) and no judged answer changed (RESULTS §3.3). So a request's greedy and seeded tokens depend on
  which requests share its prefill pass; the same batch reproduces exactly. The device sampler and decode stay
  batch-invariant. `MOTIF3_PACKED_PREFILL=0` restores per-row prefill. Making the MoE prefill row-invariant across `M`
  is post-release work.
* **T64 rows** (`tt/verify_plan.py`, `MotifModel.decode_wide`, `tt/kv_write.py`). Per DP row 16 rows: `[8 anchors at
  n | the same lanes' 8 drafts at n + 1]`, each draft with its owner's page-table row. The split-order gather puts the
  anchors in T32 lane order in rows 0..31 and the drafts in 32..63. `DecodeKVWrite(rows=64)` writes the anchors (call
  A) before the drafts (call B), both before FlashMLA. FlashMLA option A'': two B = 8 calls on global layers, one B =
  16 call on SWA layers and the MTP layer, so **T64 rows are bitwise the T32 rows**. The MoE, LM head and argmax run
  their M = 64 configs (a module refuses an M it was not built for). `WideStepPlan.result` maps rows back to
  `argmax[l] = (a[l], a[32 + l])`: the bridge contract of §17 is unchanged.
* **Verify modes** (`MOTIF3_SPEC_VERIFY`, §14). `auto` routes each step on the host (`verify_plan.choose_verify_kind`):
  ordinary steps, steps that want logits or sampling, and verify steps whose drafts all fit idle lanes run on T32;
  every other verify step is ONE T64 replay (bridge verify steps never want logits, so `auto` never takes the
  overflow pass, review edit R-E6). The bridge drafts every live lane from c* live lanes (19 at the acceptance prior
  0.85; `MOTIF3_WIDE_MIN_LANES`) and keeps the idle-lane budget below it. The device sampler runs inside the T32
  trace; the T64 trace of `auto` is argmax only (PS-1 keeps sampled rows out of verify steps).
* **Two decode traces** are safe under the rules R1-R6 of P5_T64_DESIGN §2.3 (F3N, `docs/p5_t64/f3.md` §6: F3 was the
  TP-ring all-gather race that `ring_gather="safe"` closes, not a trace effect), each enforced in code: R1 `safe` ring
  gathers (`MotifTTConfig.validate`, `_check_wide_launch`); R2 every prefill shape and every decode path compiled before
  the first capture; R3 persistent inputs allocated before it (`_stage_path` refuses after a capture); R4 a trace's
  outputs read before another trace replays (`_replay`); R5 one capture per path, no re-capture while serving; R6 the
  second trace's outputs acknowledged for the allocation tracker. Gates G-S6w-lite and G-X (33 min, plus the tracker
  variant) held them (GATES_RESULTS §13.7).
* **Cost and speed.** T64 / T32-spec device step 1.119 at 1K and 1.134 at 8K context (G16 at the release;
  `motif3-opt` with B1 sparse + B3 / B4 fused on TORUS_XY: T64 73.3 / 76.1 ms, ratio 1.208 / 1.218, so
  `cfg.wide_step_ratio` 1.21 feeds c*; the G16 bar is now the absolute T64 step and the c = 32 gain, not the ratio); T64 trace region 6.26 MiB per bank; boot +2.2 s packed warm-up, +2.3 s eager decode warm-up and
  +6.0 s T64 capture (RESULTS §3.1). Throughput and burst numbers: the launches table of §14. All of them come from
  this host's TORUS_Y fabric (§1; `docs/P5_T64_REVIEW.md` I-5). On a healthy TORUS_XY the DP axis is a ring and the
  T64 gathers on it take the safe path, so G16's ratio would move: re-measure it after a Galaxy reset.

## Status

* **Shipped state (2026-10-04): tt-metal B1 `d3597ae4977`, TIS `f0484e96`, vllm-tt-plugin `3f70daa`.** Two launches of
  the same code (§14):
  * **production default** (TIS `motif3_galaxy`): chunked prefill (budget = threshold = 8064), prefix caching with
    KV-R decode writes, exact on-device sampling in the decode trace (`tt/sampling.py`,
    `docs/sampling/DEVICE_SAMPLER.md`), packed multi-row prefill (`MOTIF3_PACKED_PREFILL=1`, §18),
    `OMP_WAIT_POLICY=PASSIVE`, `ring_gather="safe"`, the composite router, 32 lanes, `max_model_len` 32768, a KV pool
    of 262144 tokens;
  * **MTP opt-in** (TIS `motif3_galaxy_mtp`): the same plus `--speculative-config` (MTP, K = 1) and
    `MOTIF3_SPEC_VERIFY=auto` (the T32-spec and T64 decode traces, switching at c* ~ 19 live lanes, §18).

  Headline numbers (`docs/P5_T64_RESULTS.md`): a burst of 32 short prompts gets its first tokens after 2.42 s instead
  of 22.66 s, and a decoding request stalls at most 1.70 s instead of 22.0 s while it arrives; greedy MTP decode is
  1.89-1.91x the launch without MTP at c <= 16 and 1.70-1.71x at c = 20-32 (559.2 vs 327.9 tok/s at c = 32), with
  greedy and seeded tokens identical to it. Every number comes from this host's degraded fabric: TORUS_Y committed
  instead of TORUS_XY (§1; `docs/P5_T64_REVIEW.md` I-5; restoring it needs a Galaxy reset with sudo). Contract (I-1,
  signed off by the lead on 2026-10-04, `docs/P5_T64_REVIEW.md` §8): with packed prefill a request's greedy and
  seeded tokens may depend on which requests share its prefill pass (near-tie flips at the bucket floor; quality
  unchanged); `MOTIF3_PACKED_PREFILL=0` restores per-row prefill (§14). Gates: `tests/unit/gates/GATES_RESULTS.md`
  §13 and §13.7; host suites on this tree: 440 passed (`logs/host/20261004_045052_lead_p5t64_host_all.log`). The
  coverage items of review I-3 (preemption with prefix hits under packing, `MOTIF3_SPEC_VERIFY=wide` and the
  honest-benchmark MTP launch through vLLM, the TIS before / after points of P5_T64_DESIGN §7.6) belong to the final
  validation workflow of 2026-10-04; their results go to `docs/P5_T64_RESULTS.md` and the TIS reports. Open
  follow-ups: `docs/P5_T64_RESULTS.md` finding 2 (the MTP launch keeps 32 host threads busy while it decodes), the
  G16 / G13b re-measurement on a healthy torus (I-5), and a row-invariant MoE prefill (§18).
* Shared infra (this README, `tt/model_config.py`, `tt/ccl.py`, `tt/weights.py`, `tt/rope.py`, `tt/generator_api.py`,
  `tt/generator_vllm.py`, `test_infra_*`, the bridge host suite): wave-B shared fixes INFRA-1..8 and BRIDGE-1..4 of
  `docs/WAVE_A_REVIEW.md` §5 (2026-10-01), then the wave-B1 shared changes (2026-10-02): L1_SMALL (device params,
  MotifCCL semaphore routing, bridge / TIS checks), module program-config builders, `sdpa_prefill_fp32` / `ccl_reduce`
  roles, `compute_config_descriptor`, `ar_exact`, PolyNorm config fields, module defaults, `MOTIF3_TT_CACHE_PATH`. Host: 55 skeleton CPU tests
  and 38 bridge tests (+1 skipped until `tt/generator.py` exists) pass under `scripts/hostrun.sh`. Device:
  `test_infra_device.py` (every draft-1 CCL payload eager, the decode chain AR(tp) -> ag_dp_rows -> AR(dp) ->
  partition(dp) traced and replayed with new inputs, role mappers + cache reload, RoPE, G1 / G2 / G6 through the shared
  builders) and `test_infra_l1_small.py` (main L1 untouched by every CCL path, results bitwise equal to plain ttnn;
  attention serving order prefill 1024/4096 -> decode -> prefill again (bitwise equal) -> decode for bfp8 and bf16 KV in
  one session). Measured on a fabric that commits only TORUS_Y (DP axis a line).
* Gates G0-G8, the feature gates G9-G13a (§12 of `tests/unit/gates/GATES_RESULTS.md`), the P5 / T64 op gates
  G15a-rest, G-S1w and G16-lite (§13) and the model-level P5 / T64 gates G15b, CP-P, CP9-P, G16, G-S5w, G-S6w-lite and
  G-X (§13.7).
* Modules (wave B1): embedding, mHC, attention, PolyNorm / MLP, MoE, LM head, the Sinkhorn and router kernels
  (`docs/WAVE_B1_SUMMARY.json`). Decoder, model, generator, TIS integration: the integration wave (WAVE_A_REVIEW
  §5.8-5.11).
* Features (chunked prefill, prefix caching, MTP speculation; `docs/features/FEATURES_DESIGN.md`): **in the runtime and
  validated end to end** (2026-10-03; `docs/FEATURES_RESULTS.md`, review `docs/FEATURES_REVIEW.md`). The contract
  (`generator_api`: `PrefillRequest.start`, `prefill_forward_batch`, `SpecDecodeBatch` / `SpecDecodeResult` /
  `decode_forward_spec`, the capability properties, the feature `GeneratorSettings`, `kv_write_mode`), the planner
  (`tt/prefill_plan.py`), the sp1 attention paths (`tt/attention.py`), KV-R and the split KV writes (`tt/kv_write.py`),
  the MTP layer (`tt/mtp.py`), the generator (`prefill_forward_batch`, the T32-spec decode trace) and the bridge
  (`FEATURE_SWITCH_DEFAULT = True`) all serve through a real `vllm serve`; the plugin has PS-1 (a verify step never
  holds a sampled or penalized row). Host suites: `test_prefill_plan.py` (exhaustive planning to 1100 tokens for
  blocks 32 / 64 and A 64 / 128, every lone-request vLLM step to 32768 at the production geometry, samples to 32768,
  an emulated paged cache under a vLLM-like scheduler), the `test_infra_config.py` feature tests, the bridge suite,
  the WP4 / WP5 host emulation; device: gates G9-G13, `test_attention_resumed.py`, `test_resumed_prefill.py` (CP-H /
  CP-C / CP-L / CP-X / CP9), `test_spec_decode_device.py`, `test_vllm_features_e2e.py` against live servers.
  Lead decisions: the production default launch is chunked prefill + prefix caching + device sampling (MTP is an
  opt-in launch, §14); the floor-relative accuracy bars of `test_resumed_prefill.py` are signed off; gate G9's
  per-bucket sp1 q/k (F5: A = 128, budget = threshold = 8064), with CP-H (ii) and CP-L (ii) re-signed under it on
  2026-10-03 (option (a) of the F5 bullet below; the asserted bars are in the `test_resumed_prefill.py` module
  docstring). The items open at the time are closed: prefill run-to-run nondeterminism (FEATURES_REVIEW P1) and the
  "two-trace hazard" (F3) were both the TP-ring all-gather race, fixed by `ring_gather="safe"` in B0 `277df0e9f2d`
  (`docs/determinism/FIX.md`, `docs/p5_t64/f3.md`); burst TTFT (P2) by packed prefill and MTP at c = 32
  (FEATURES_REVIEW F1) by the 64-row verify, both in B1 (§18). The MTP launch now captures two decode traces.
* F5 (2026-10-03): gate G9's per-bucket sp1 global q/k (`prefill_plan.DEFAULT_SP1_GLOBAL_CHUNKS`), A = 128, vLLM budget
  = threshold = 8064 (`recommended_budget`, `generator_api.DEFAULT_PREFILL_ALIGNMENT`, `FEATURE_VLLM_ARGS`, the TIS
  spec) and G9's per-bucket sp1 cost model in the planner (`DEFAULT_SP1_GLOBAL_COST`). Measured on HEAD + F5 alone:
  * Module level (`test_attention_resumed.py::test_wp2b_sp1_cost`, eager, bfp8 cache, prefix 128 / 8192): global sp1
    per call C = 128 2.85 / 5.36 ms (64/64: 2.81 / 6.03), 2048 4.82 / 12.08 (5.26 / 15.54), 8192 24.44 / 47.05
    (30.85 / 65.69). `test_attention_resumed.py` 7/7.
  * A bf16 latent cache keeps 64/64 (A = 64): at 128/128 the op's static CBs end at 1,572,480 B, through L1_SMALL,
    and the bf16-KV schedules returned garbage and hung a later CCL (tt-metal does not check CBs against L1_SMALL);
    with 64/64 they pass again.
  * Prefill TTFT (`test_prefill_ttft_report` alone, HEAD vs HEAD + F5 back to back): cold 32,000 tokens 27.04 ->
    25.78 s, 16,736 13.11 -> 12.85 s; single-chunk rows (128: 0.66-0.67 s, 8192: 5.85 s) and prefix hits (0.67-0.70 s)
    unchanged. (At the end of the full-suite run, after CP9, the hits measured 0.77-0.81 s.)
  * Serving (direct `vllm serve`, host sampling). Production flags: the 30,832-token needle prompt in 4 vLLM steps of
    8064 has TTFT 25.5 s (26.4 s at 8128 with 64/64), a running decode stalls at most 6.7 s (7.1 s), the same prompt
    again 0.93 s (30,784 tokens hit; that odd-64 hit recomputes 64 rows), answers equal to the pre-F5 servers'. The
    opt-in MTP launch (`FEATURE_VLLM_ARGS`, 8064) passes the live e2e tests that touch the budget, prefix hits or
    speculation: greedy speculation 8/8 identical to draft 1 (acceptance 0.93 / 0.88), a 6,196-token prompt 5.15 s
    cold -> 0.79 s hit, a decode-written hit 0.83 s (cold 2.23 s), the 30K prompt 25.5 s (26.5 s) with a 6.7 s stall
    and 0.84 s again, the PS-1 mix 16/16, the logprob refusal; its prefix, 30K and PS-1 greedy outputs equal the
    pre-F5 server's.
  * `test_resumed_prefill.py` (full depth): CP-L (i) passes on the 31,972-token needle prompt (budget path 4 calls of
    8064, 27.4 s vs 28.1-28.8 s at 8128; top-1 vs the single shot 0.912 on the 420 confident rows, 0.876-0.905 at
    8128, asserted bar 0.857, the chunked path's own repeat 0.911; NLL 1.893 vs 1.971; the needle found on every
    path); CP-C 4/4 (top-1 vs the cold rows -0.13 .. +0.13 pt), CP-H (iii) pooled -0.06 pt / -0.03 % NLL, CP-X, CP9
    (program cache constant) and the MTP fill pass.
  * Two single-row bars of the signed-off set failed (the lead's call, decided on 2026-10-03: option (a) below;
    reproduced bit-identically by the review, `logs/dev/20261003_084206_rev_defaults_f5_snap.log`). **CP-H (ii)**:
    multi_turn_chat's hit at 192 now resumes at 128, and its last row picks the fp32 golden's token (6, golden
    margin 0.24) where the cold row picks 171 at margin 0.531, just over the 0.5 near-tie bar (KL vs fp32: hit 0.024,
    cold 0.142); the rule reads the cold row's margin only, while the greedy-stream rule excuses the same token as a
    near tie (hit margin 0.25).
    **CP-L (ii)** (layers 0-3 vs the CPU reference): at 31,972 tokens both chunked paths' worst sampled
    position is 31961 at stream PCC 0.99678, 0.0015 under the single shot's worst (0.99829; bar: within 0.001),
    while their median rises to 0.99988 (64/64: 0.99972; single shot 0.99995) and the mean error (1 - PCC over the
    172 positions) falls from 0.000426 to 0.000306. Same-session diagnostics: the k chunk decides (q 64 / k 128 at
    C >= 2048 gives F5's numbers; q 128 / k 64 gives 64/64's: worst 0.99803 at 16386, median 0.99971), a repeat is
    bitwise equal, and against the single shot k = 128 is the closer table at 132 / 127 / 131 / 123 of the 172
    positions after layers 0 / 1 / 2 / 3 (mean error 0.000176 vs 0.000379 after layer 0) while single positions swing
    both ways from layer to layer (28672: closer through layer 2, 0.9971 vs 0.9996 after layer 3, an MoE layer): the
    chaotic single-row behaviour of CP-H, not a systematic loss.
  * The options for these two (lead decision 2 signed off exactly these bars; the lead took (a) on 2026-10-03). Both
    fail on a noisy reference (the cold row's margin; one worst position), not on an F5 defect:
    (a) re-sign them with references that are not noisy: CP-H (ii) against the fp32 golden's argmax and margin (the
    hit picks the golden's token), CP-L (ii) on the median or mean error over the 172 positions (both better under
    F5). Only the bar code in `test_resumed_prefill.py` changes; F5's TTFT stays.
    (b) keep the bars and go back to the all-64/64 table (A = 64, budget = threshold = 8128), under which both
    pass (CP-H (ii) on the pre-F5 tree, `logs/dev/20261003_015009_wp45_rev_cp53_b.log`; CP-L (ii) worst 0.99812 in
    the same-session 64/64 run): about +1.3 s cold TTFT at 32K (27.04 vs 25.78 s) and +0.9 s on the 30K serving
    prompt (26.4 vs 25.5 s).
    k = 64 at C >= 2048 alone (q 128 / k 64, A stays 128) clears CP-L (ii) only (worst 0.99803 vs the single shot's
    0.99829) at an unmeasured cost (G9 timed 64/64 and 128/128 there). CP-H (ii) follows A, not the large buckets:
    the hit at 192 resumes at 128 whatever the C >= 2048 entry. Restoring its pre-F5 resume point needs A = 64 for
    the C = 128 chunk: (b), or 32/64 or 64/64 at C = 128 with G9's per-bucket A (a planner change, GATES_RESULTS
    §12.2; not run against the bar).
