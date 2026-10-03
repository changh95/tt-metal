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
                       writer-first row order, vLLM scheduler checks; §15); torch-only
    generator_vllm.py  MotifForCausalLM (vLLM bridge; device-free import)
    embedding.py       MotifEmbedding (token gather -> 4 streams)        lm_head.py   MotifLMHead (mean, norm, head)
    mhc.py             MHCSite (pre / post of one mHC site)              attention.py MotifAttention (GDLA)
    polynorm.py        PolyNorm (scalar TP / grouped)                    mlp.py       PolyNormMLP (dense / shared)
    moe.py             MotifRouter, MotifMoE (EP32)
    kernels/           out-of-tree ttnn.generic_op kernels: sinkhorn_motif (exact mHC coefficients), router_fp32
    decoder.py, model.py, generator.py   integration wave (WAVE_A_REVIEW §5.8 GEN-1..7)
  tests/
    unit/test_infra_*.py   shared-infra tests (CPU: config, rope, weights, import; device: test_infra_device.py,
                           test_infra_l1_small.py)
    unit/test_<module>.py  module tests (each owner's)
    unit/gates/            device op gates G0-G8 (results in unit/gates/results, summary GATES_RESULTS.md)
    test_generator_vllm_host.py   vLLM bridge host suite (real vLLM + plugin, fake generator)
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
  warmed before decode capture), **replicated on all 32 chips** (design §3.3). `paged_fill_cache` gets exactly the
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
| `cfg.flash_mla_decode_pc()` / `flash_mla_decode_pc(grid)` | `SDPAProgramConfig(12x10, q_chunk 0, k_chunk 128, exp_approx False, max_cores_per_head_batch 16)` with `sdpa_decode` | PCC 0.99994 SWA / global, SWA window + causal-edge probe exact, 29.8 us SWA / 102 us global 4K traced. **Never** `k_chunk` 0/None (dynamic chunking drops the causal mask under a window: upstream bug), 256 clashes with a sharded Q, 512 overflows L1 |
| `cfg.sdpa_prefill_pc(layer or "swa"/"global", seq_len=S)` | `SDPAProgramConfig(12x10, q/k 128/128 SWA, 256/256 global, exp_approx False)`, clamped to 128 for S = 128; with `sdpa_prefill` | PCC 0.99975 SWA / 0.99961 global at S = 1024; V zero-padded to 192 (dv = 128 is rejected); q512/k512 overflows L1 |
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
  parallel ops right before device reads (`OMP_WAIT_POLICY=PASSIVE` is the serving-side candidate, TIS runbook).

## 13. Import rule

Nothing under `models/demos/motif3` imports another `models/demos/**` package at module import time (vLLM imports
the bridge before the mesh opens; demo imports can open the cluster). Copy small helpers, or import lazily inside
functions. vLLM / transformers / huggingface_hub / safetensors are imported lazily too.
`tests/unit/test_infra_import.py` checks, in a fresh interpreter, the shared infra, every wave-B1 module
(`mhc, attention, polynorm, mlp, moe, embedding, lm_head`) and the kernels (`kernels`, `kernels.sinkhorn_motif`,
`kernels.router_fp32`), plus `decoder`, `model` and `generator` as soon as their files exist; the bridge host suite
checks that the default generator class imports device-free (`test_real_generator_class_imports_device_free`).

## 14. Environment

| Variable | Meaning (default) |
|---|---|
| `MOTIF3_WEIGHTS_DIR` | checkpoint dir; wins over everything (set but missing raises) |
| `HF_MODEL` | snapshot dir (TIS), or a repo id resolved through the HF cache at `TT_MODEL_WEIGHTS_REVISION` |
| `MOTIF3_TT_CACHE_PATH` | TT weight-cache root; wins over `TT_CACHE_PATH` (shares a converted cache with a TIS server without the runbook's symlink, CONV-2) |
| `TT_CACHE_PATH` | TT weight-cache root (`/home/ttuser/hchang/experiments/motif-3/tt_cache`); TIS sets it under its host volume |
| `TT_MODEL_WEIGHTS_REVISION` | checkpoint revision in the cache tag and for repo-id resolution (`2ed2ed5c...`) |
| `MOTIF3_NUM_LAYERS` | truncated bring-up runs (53) |
| `MOTIF3_KV_POOL_TOKENS` | usable KV pool tokens, a multiple of 128 (262144 -> 4129 blocks of 64 with the reserve) |
| `MOTIF3_KV_CACHE_DTYPE` | `bfp8` (default) or `bf16` (needs a smaller pool) |
| `MOTIF3_KV_MAX_GB_PER_CHIP` | KV budget for the bridge's fail-fast check (16) |
| `MOTIF3_MAX_MODEL_LEN` | max context for host-built configs (32768; serving takes vLLM's `--max-model-len`) |
| `MOTIF3_TRACE_REGION_SIZE` | trace region bytes (268435456) |
| `MOTIF3_L1_SMALL_SIZE` | L1_SMALL bytes per core for `device_params()` / `open_motif_mesh()` (32768; experiments only) |
| `MOTIF3_ROUTER_LOGITS` | `cfg.router_logits`: `composite` (default) or `exact_fp32` (decision D1 A/B) |
| `MOTIF3_FABRIC` | `FABRIC_2D_TORUS_XY` (fallback `FABRIC_1D_RING`); with a mesh the device's fabric wins |
| `MOTIF3_GENERATOR_CLASS` | runtime class for the bridge (`models.demos.motif3.tt.generator:MotifGenerator`) |
| `MESH_DEVICE` | `"(4, 8)"` for serving (`(8, 4)` also works; `open_motif_mesh()` reads it) |
| `MOTIF3_PREFIX_CACHING` / `MOTIF3_CHUNKED_PREFILL` / `MOTIF3_SPEC_DECODE` | bridge class capabilities (§15-§17; `generator_api.feature_switch_from_env`: `1/true/yes/on` or `0/false/no/off`, a typo raises); they only allow a feature, vLLM's flags enable it |
| `MOTIF3_KV_REPLICATED_DECODE` | KV-R (§16): `auto` (default: on iff prefix caching), `1` (forced on), `0` (refused with prefix caching) |
| `MOTIF3_PREFILL_MAX_BUCKET` | span cap (§15): largest prefill bucket of a resumed-prefill generator, a power of two in [128, 32768] (8192); longer spans are split into chunks |
| `MOTIF3_PACKED_PREFILL` | optional packed multi-row prefill (0; only after gate G15) |
| `MOTIF3_SPEC_VERIFY` | `packed` (default: drafts in idle lanes of the 32-lane trace) or `wide` (64-row verify trace, only after gate G16) |

Serving flags: `--block-size 64` (32 also allowed), `--max-model-len 32768` (a multiple of 256), `--max-num-seqs 32`,
`--additional-config '{"tt": {"trace_mode": "decode_only", "trace_region_size": 268435456, "fabric_config":
"FABRIC_2D_TORUS_XY", "dispatch_core_axis": "col", "l1_small_size": 32768}}'` (`generator_api.SERVING_TT_CONFIG`; TIS:
`override_tt_config`). Without `l1_small_size` the bridge refuses to start (`get_max_tokens_all_users`, in
`init_device`, before the weights load) and refuses a mesh with less L1_SMALL (`initialize_vllm_model`).

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
  | resume alignment `A` | `cfg.prefill_resume_alignment = lcm(bs, q, k)` of the sp1 global op (`model_config.SP1_GLOBAL_CHUNKS`, gate G9 decides) | 64 |
  | compute floor `c0` | `floor(s / A) * A`, or 0 when that is below the SWA tail `cfg.prefill_swa_tail` (128) | |
  | span cap | `cfg.max_prefill_span` (`MOTIF3_PREFILL_MAX_BUCKET`), buckets `cfg.prefill_span_buckets` (128 ... 8192) | 8192 |
  | chunks | consecutive, `A`-aligned starts; full-cap chunks while more than the cap remains; then one padded chunk, or a head chunk + the rest when the cost table (`cfg.prefill_cost_table`) says it is cheaper | 16,736 -> 8192 + 8192 + 512 |
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
  16K / 32K buckets are no longer compiled (the planner splits those spans).
* **vLLM flags**: `--max-num-batched-tokens` = `--long-prefill-token-threshold` = `span cap - A`
  (`prefill_plan.recommended_budget`: 8128 for A = 64, 8064 if G9 moves the large buckets to 128/128): a lone prompt's
  chunk ends are then multiples of `A` (no recompute) and every span fits one bucket. The bridge runs
  `prefill_plan.check_scheduler_config` at `init_device`: it raises on prefix caching without KV-R or with a
  `--prefix-match-unit` other than the block size, and warns on unaligned budgets / thresholds, spans over the cap and
  budgets below `4096 - A` (vLLM's unpinned `vllm serve` default on TT is 2048).

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
  argmax [32, 2] = (a0, a1), mtp_argmax [32, 2] = (m0, m1))`, checked with `generator_api.check_spec_result`. One
  decode trace (the spec trace) serves ordinary, verify and overflow steps.

## Status

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
* Gates G0-G8: `tests/unit/gates/` (see `GATES_RESULTS.md`).
* Modules (wave B1): embedding, mHC, attention, PolyNorm / MLP, MoE, LM head, the Sinkhorn and router kernels
  (`docs/WAVE_B1_SUMMARY.json`). Decoder, model, generator, TIS integration: the integration wave (WAVE_A_REVIEW
  §5.8-5.11).
* Features (chunked prefill, prefix caching, MTP speculation; `docs/features/FEATURES_DESIGN.md`), WP1 contract +
  config (2026-10-02): `generator_api` (`PrefillRequest.start`, `prefill_forward_batch`, `SpecDecodeBatch` /
  `SpecDecodeResult` / `decode_forward_spec`, the capability properties, the feature `GeneratorSettings` and their
  environment, `kv_write_mode`, `check_generator_features`), `tt/prefill_plan.py`, the `model_config` feature fields
  (§15-§17). Host only: `test_prefill_plan.py` (exhaustive planning to 1100 tokens for blocks 32 / 64 and A 64 / 128,
  samples to 32768, an emulated paged cache under a vLLM-like scheduler) and the `test_infra_config.py` feature tests
  pass; nothing in the runtime uses them yet (WP2-WP6 build on this contract).
