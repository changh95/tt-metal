# Motif-3 device op gates G1-G8: results

**Run:** 2026-10-01 (17:17-17:56 UTC), host `bh-glx-exp-a03u07`.
**Software:** tt-metal branch `motif3-bh-galaxy` at `d611b1ec78d` (Release build).
**Mesh:** logical (4, 8) on the 8×4 Blackhole Galaxy torus. Rows are `cluster_axis 0` (4 chips, the DP groups); columns are `cluster_axis 1` (8 chips, TP8).
**Device params:** `FABRIC_2D_TORUS_XY` (G4 also runs `FABRIC_1D_RING`), `dispatch_core_axis=COL`, `trace_region_size` 96 MiB. The compute grid is 12×10 = 120 cores per chip.
**Design references:** design doc §1.2, §1.6, §4.2, and the per-chip shapes of §3.2/§3.3. Study reports are cited as [R:NN].

| Item | Location |
|---|---|
| Tests | `models/demos/motif3/tests/unit/gates/test_g{0..8}_*.py` |
| Shared helpers | `gate_utils.py` (mesh params, PCC, timing, results), `goldens.py` (torch goldens) |
| Raw results | `results/G*.jsonl`. These are append-only; the last record per `case` wins. Some cases from earlier runs are kept: G1 `k_chunk512/qshard` errors and the G6 `reuse_*` errors |
| Tables | `python models/demos/motif3/tests/unit/gates/make_report.py G1 G3 …` (host-only) |
| Run | `scripts/devrun.sh -t 1500 -n gates -- pytest models/demos/motif3/tests/unit/gates -q -p no:cacheprovider`. About 6-8 min for all gates; run G4 separately if you want to isolate CCL risk |

## 1. Verdicts

Final pytest lines come from the logs in `logs/dev/`:

| File | Content |
|---|---|
| `20261001_175220_final_g0g1g3g5g7g8.log` | G0, G1, G3, G5, G7, G8 final run |
| `20261001_174928_g4_ccl.log` | G4 |
| `20261001_175118_g6_v2.log` | G6 |
| `20261001_173932_g1g2g3_v2.log` | G2 |
| `20261001_174319_g5g7g8_g2g3fix.log` | G2 at S=4096 |
| `20261001_175505_g1_perf_fp32acc.log` | G1 fp32-accumulation latency |

### G1 — paged MLA decode with a sliding window: **PASS**

Requires an explicit `k_chunk_size=128`.

- **Accuracy:** PCC 0.99988 (SWA) and 0.99956 (global) against a fp64 golden on the same bfp8 cache. With fp32 accumulation these rise to 0.99994 and 0.99993.
- **Window edges:** the boundary probes are exact.
- **Inactive lane:** position −1 is skipped.
- **Replicas:** all 32 chips agree bitwise.
- **Latency (traced):** 29.7 µs (SWA), 101.8 µs (global, 4K context), 470 µs (global, 32K context, 342 GB/s).
- **Decision:** keep the latent MLA decode on all 53 layers. The SWA-ring fallback is not needed. Always pass `SDPAProgramConfig(k_chunk_size=128)`.

**Bug found (marked xfail).** With `k_chunk_size=0` the kernel uses dynamic chunking. That is also what the op does when no program config is passed. In that mode a sliding window drops the causal mask whenever the window fits in one chunk, and key p+1 leaks into the output.

### G2 — GQA SDPA prefill, 10/2 heads, d=192, window 129: **PASS**

- **Accuracy:** PCC 0.99948–0.99982 for S = 128/1024/4096, both layer types.
- **V padding is required.** The plain op rejects `dv=128`. With V zero-padded to 192, output columns 128–191 come back exactly 0.
- **Latency (eager):** SWA 204 µs (S=1024) and 272–525 µs (S=4096); global 245 µs and 803 µs.
- **Decision:** use the plain SDPA with V padded to 192, chunks 128/128 for SWA layers and 256/256 for global layers.
  - Global layers may use `flash_mla_prefill(q, k, v128)` instead: PCC 0.99962, no padding.

### G3 — mHC Sinkhorn: **PASS** (stock op, `eps=0`, realistic logits)

- **Accuracy (realistic logits):** max|ΔH| 1.1e-3–1.9e-3, max|Δh_pre| ≤ 8.9e-4, max|Δh_post| ≤ 1.1e-3.
- **Out-of-range logits** (|L| > 20 or > 10): the stock op deviates by up to 0.9. Pre-clamping the logits brings this back to ≤ 5.6e-3, except one extreme regime.
- **Deriving Motif's maps:** h_post = post/2 exactly, h_pre = pre, H = comb.
- **Latency:** 52 µs traced per call (T = 8 or 32).
- **Decision:** use Option A (the stock op). Switch to the pre-clamped call mode, or to the Option-B `generic_op` kernel, only if CPU check C3 shows out-of-range logits.

### G3 — stream mix (`attn_res_weighted_reduce_nc`): **PASS** (with caveat)

- **Accuracy:** PCC ≥ 0.999998 and 88 % of outputs bit-exact against an fp32 einsum followed by one bf16 cast.
- **Caveat:** the worst error is 2.3–4.5 ulps of the largest summand. That is at most one extra rounding, but it is **not** the single rounding the design assumed.
- **Latency (traced):** 15 µs (pre), 28 µs (H·X), 36 µs (fused post, C=5).
- **Decision:** use the op with fp32 weights. The composite fallback (multiply + sum) is bit-exact but 7–16× slower.

### G4 — CCLs with Motif payloads on the 2D torus and the 1D ring: **PASS**

- **Correctness:** all payloads are correct on both fabrics, in eager and traced mode, and every reduce group's replicas are bitwise identical.
- **Traced latency per decode CCL:**

  | Collective | Traced µs |
  |---|---|
  | AR(cols) `[8,4096]` | 33 |
  | AG(rows) 8-row, TILE | 62 |
  | AG(rows) 8-row, ROW_MAJOR | 12 |
  | AR(rows) `[32,4096]` | 33 |
  | stats AR `[8,3]` fp32 | 39 |
  | RS(rows) 8-row shards | 109 |

  The 1D ring is 5–15 % faster.
- **Decision:** keep `FABRIC_2D_TORUS_XY`. Do the MoE token gather in ROW_MAJOR. Keep AR + slice for the 8-row combine (not RS). Budget about 30–40 µs per decode CCL; the design assumed 7–15 µs.

### G5 — fp32 router: **FAIL** (agreement 99.63–99.71 % against the 99.9 % target)

- **Cause:** every flip is a near-tie (the 8th/9th score gap is ≤ 1.8e-4). The flips come from the matmul logits (RMS error 3.3e-4), consistent with TF32-class FPU partial sums. Sigmoid and topk are exact.
- **fp32 weights do not help.** They give bit-identical logits, because the operands are bf16-valued either way.
- **Routing is consistent:** all 32 chips produce bitwise-identical routes.
- **Latency:** 189 µs per call, traced.
- **Decision:** the named escalation (HiFi4 + fp32 weights) is ineffective. Choosing the next step is an open issue (§11).

### G6 — batched bfp8 expert matmuls and PolyNorm: **PASS**

- **Accuracy:** PCC 0.999999 (HiFi4).
- **Default program config is slow:** gate_up 983 µs (136 GB/s), down 289 µs (231 GB/s).
- **1D-multicast config is fast:** gate_up 426 µs (314 GB/s), down 222 µs (301 GB/s), so 648 µs per MoE layer.
- **PolyNorm:** PCC 0.999999 with fp32 intermediates (294 µs) and 0.999991 with bf16 intermediates (164 µs).
- **Decision:** use `MatmulMultiCoreReuseMultiCast1DProgramConfig` (gate_up on 80 cores, down on 32 cores) and HiFi4. PolyNorm uses bf16 intermediates in prefill and fp32 in decode, or bf16 in both (both pass).

### G7 — paged cache update and fill: **PASS (bit-exact)**

- The bfp8 latent cache with block 32 or 64 reads back bit-identical everywhere: written rows, untouched rows, the −1 lane (left untouched), and on every chip checked.
- Fill works with both `batch_idx` and `batch_idx_tensor`.
- **Latency:** decode update 4.8 µs traced.
- **Decision:** use the ops as designed. Prefill input must be cast to bfloat8_b before the fill.

### G8 — `rotary_embedding_hf`, head dim 64: **PASS**

- **Accuracy:** PCC ≥ 0.999996 for decode (height-sharded, per-user positions up to 32767) and prefill (S up to 4096), with both the YaRN and the plain tables.
- **Latency:** decode is ≤ 3 µs traced. The composite fallback (x·cos + (x@R)·sin) also passes but costs 19 µs.
- **Decision:** use the fused op.

### Overall

**Kill criteria (§1.6):**

| Criterion | Result |
|---|---|
| G1 (MLA decode + window) | Not triggered (pass) |
| G3 (Sinkhorn op range or semantics) | Not triggered for realistic ranges |
| G4 (CCLs on the 2D torus) | Not triggered |
| Router agreement below target | **Triggered.** The named escalation does not fix it; see §11 |

## 2. Measurement method

These caveats apply to every latency number in this report.

- **Golden.** Every gate computes an in-test torch fp32/fp64 golden from `goldens.py`, which transcribes `hf_meta/modeling_motif.py`.
  - The CPU reference package was still in flux. It is cross-checked lazily: `goldens.motif_sinkhorn` equals `reference.modules.sinkhorn` with max diff 0. The `reference.rope.yarn_inv_freq` signature differs, so it was not compared. The local YaRN table matches the values in [R:01 A.1].
- **Inputs.** Single-chip ops receive *replicated* inputs, so all 32 chips run the same program. Device 0 is compared with the golden, and all 32 replicas are checked for bitwise identity. G4 and G5 shard distinct data per chip.
- **Eager latency** is the mean over N back-to-back dispatches followed by `ttnn.synchronize_device`. On 32 chips it is dominated by host dispatch: about 80–100 µs per small op, and 260–350 µs for an isolated op.
- **Traced latency** is measured with a slope method.
  - Two traces hold n1 = n/2 and n = 64 back-to-back calls (n = 24 or 32 for heavy ops).
  - Each trace is replayed 9× (5× for the heavy G6 matmuls) with `synchronize_device`.
  - per_op = (min t(n) − min t(n/2)) / (n/2).
- **G0 calibration** (`test_g0_timing_calibration.py`) explains the choices above:
  - `synchronize_device` alone costs 133–149 µs on the 32-chip mesh.
  - Device work shorter than that hides behind it, so t(n) stays flat at about 150–185 µs until n·t_op exceeds about 150 µs. Ops under about 2–3 µs read as "0" even at n = 64; for those, the raw t(64)/64 is reported as an upper bound.
  - A traced 1-tile `ttnn.add` costs 5.7–6.7 µs. An 8 MB eltwise add costs 44 µs (about 390 GB/s).
  - Replays are **bimodal**. Besides the fast path, some take a slow path quantized at 1/2/3 ms (the host sleeps while polling). The estimator therefore uses the **minimum** over repeats, not the median. Medians from the first runs were corrupted by this (G8 decode), and all numbers here come from the final runs.
- **Results files** keep every attempt. The tables below quote the final runs.

## 3. G1 — `paged_flash_multi_latent_attention_decode` (`test_g1_mla_decode.py`)

### Setup

**Per-chip shape:**

| Item | Value |
|---|---|
| Q | `[1, 8 users, 10 heads, 576]` bf16 |
| Cache | `[num_blocks, 1, block, 576]` bfloat8_b, DRAM |
| V | `v=None`, `head_dim_v=512`, nkv=1 |
| `page_table` | int32 `[8, W]`, ROW_MAJOR, random permutation |
| `cur_pos_tensor` | int32 `[8]`, ROW_MAJOR |
| Positions (one per user) | 0, 1, 127, 128, 129, 130, 1000, 5000 |
| Context | 5120 |
| Data | random N(0,1) |

**10 heads are supported unpadded.** The op tile-pads 10 to 32 internally; the output is `[1,8,10,512]`.

### Accuracy

The golden is fp64 masked attention over the *quantized* cache, with keys in [max(0, p−128), p] for SWA and [0, p] for global.

| Config | SWA (W=129, s=0.07216878) PCC / min-user | Global (s=0.14467963) PCC / min-user | vs unquantized cache (SWA / global) |
|---|---|---|---|
| block 64, k_chunk 128, Q DRAM, HiFi4, fp32_acc off | 0.999883 / 0.99917 | 0.999557 / 0.99853 | 0.999842 / 0.999502 |
| block 32 (same otherwise) | bit-identical to block 64 | bit-identical | |
| Q height-sharded `[32,576]` on 8 cores, out `[32,512]` | identical | identical | |
| k_chunk 256 | 0.999882 | 0.999568 | |
| **fp32_dest_acc_en=True** | **0.999944 / 0.99978** | **0.999933 / 0.99987** | 0.999891 / 0.999848 |
| SWA window with global scale / global with SWA scale | 0.999684 / 0.999911 | | |

### Probes

**Window and causal-edge probes.** Each user's cache is built so that three keys would dominate the softmax if attended:
- the first in-window key (V=+1), which should be attended;
- the key just outside the window (larger score, V=−1);
- key p+1 (largest score, V=+3).

The positions are 129, 159, 160, 161, 255, 256, 1000 and 5000, covering the tile and chunk edges of the window start. With k_chunk 128, all 8 users output +1.000 (max |Δ| 3e-7). The global probe (key 0 dominant, p up to 5000) is also exact.

**Inactive lane.** Position −1 is skipped; the active lanes stay at PCC 0.99990. The 32 replicas are bitwise identical in every configuration.

### Bug: dynamic chunking drops the causal mask under a sliding window

This is case `b64_kdyn_qdram`, marked xfail with the reason recorded in the test.

**Trigger.** `k_chunk_size=0` turns on DYNAMIC_CHUNK_SIZE. That is the default for `paged_flash_multi_latent_attention_decode` when no `program_config` is passed (`sdpa_decode.cpp`, the `k_chunk_size = 0` path).

**Effect.**
- With a window, the probe verdict is CAUSAL_LEAK at p ∈ {129, 159, 160, 161, 1000, 5000}. Key p+1 is attended.
- SWA PCC drops to 0.976, and to 0.927 with the global scale.
- Users at p=255 and p=256 are fine: for them the window spans two chunks, or key p+1 falls in the next chunk.
- Global layers (no window) are unaffected.

**Root cause.**
- In `sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:339-353` the DYNAMIC_CHUNK_SIZE branch fuses a single mask into the QK matmul.
- When `k_chunk == window_start_chunk == k_chunk_end-1`, it picks only `cb_sliding_window_mask_in`.
- `generate_sliding_window_mask` (`dataflow_common.hpp:308`) masks only the positions before the window start.
- So the causal mask is never applied.
- The non-dynamic path applies both masks separately and is correct.

**Consequences.**
- The same kernel serves `paged_scaled_dot_product_attention_decode`, so the G1 fallback must pin `k_chunk_size` too.
- This is worth an upstream issue.

### Latency

Traced, per op. The 8 users all sit at pos = ctx−1, with a block-64 bfp8 cache allocated on device with `ttnn.empty` + `ttnn.fill`.

| Case | Q DRAM | Q height-sharded | Q DRAM + fp32_acc | Eager |
|---|---|---|---|---|
| SWA (any context), k_chunk 128 | 29.7 µs | 27.2 µs | 30.1 µs | 145–220 µs |
| Global, 4K context, k_chunk 128 | 101.8 µs (197 GB/s KV) | 106.6 µs | 102.8 µs | 230–255 µs |
| Global, 32K context, k_chunk 128 | 469.7 µs (342 GB/s) | 475.8 µs | 470.9 µs | 490–502 µs |
| k_chunk 256 / 0 (Q DRAM) | SWA 39.1 / 38.4; global 4K 116 / 116; 32K 475 / 477 | CB clash with the sharded Q's L1 (error) | | |
| k_chunk 512 | error: static CBs 1.70–1.75 MB > 1.5 MB L1 | | | |

### Working configuration (recommended)

```python
pc  = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),  # 12x10
                             q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False, max_cores_per_head_batch=16)
ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                              fp32_dest_acc_en=True, packer_l1_acc=False)   # fp32 acc: free, +accuracy
out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
    q,            # [1, 8, 10, 576] bf16, DRAM interleaved (or HEIGHT_SHARDED [32,576] x 8 cores + same-grid [32,512] output)
    kvpe_cache,   # [N, 1, 64, 576] bfloat8_b DRAM
    None, head_dim_v=512, page_table_tensor=pt, cur_pos_tensor=pos, scale=s_l,
    sliding_window_size=129 if swa else None, program_config=pc, compute_kernel_config=ckc,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

**Never omit `program_config`, and never use `k_chunk_size=0`, on SWA layers.**

## 4. G2 — `scaled_dot_product_attention` prefill (`test_g2_sdpa_prefill.py`)

### Setup

**Per-chip shape:**

| Tensor | Shape |
|---|---|
| q | `[1,10,S,192]` |
| k | `[1,2,S,192]` |
| v | `[1,2,S,192]`, zero-padded from 128 |

- All three are bf16, DRAM interleaved, `is_causal=True`.
- The golden is fp32 attention with q head h reading kv group h // 5 and keys in [p−128, p] for SWA.

### Confirmations

- **V must be padded.** V `[1,2,S,128]` into the plain op is **rejected**: TT_FATAL `sdpa_device_operation.cpp:101 k_shape[3] == DH && v_shape[3] == DH`.
- **Padding is clean.** With zero-padded V, output columns 128..191 are exactly 0.
- **No-padding alternative for global layers.** `flash_mla_prefill(q, k, v_128)` with a separate, narrower V works: S=1024, PCC 0.99962, 233 µs eager. The MLA op has no `sliding_window_size`, so SWA layers still need the plain op.

### Results

PCC is the minimum over the HiFi2-approx (op default) and HiFi4 compute configs and the chunk pairs 128/128, 256/256 and 128/256. All configs pass.

| S | SWA (W=129) PCC | Global PCC | Eager, SWA (q128/k128) | Eager, global (q256/k256) |
|---|---|---|---|---|
| 128 | 0.99981 | 0.99978 | 230–400 µs (dispatch-bound) | 225 µs |
| 1024 | 0.99975 | 0.99961 | 204 µs | 245 µs |
| 4096 | 0.99972 | 0.99948 | 272–525 µs (run-to-run eager spread, HiFi2 or HiFi4) | 803 µs (q128/k128: 1436 µs) |
| 4096, K/V bfloat8_b | 0.99972 | 0.99951 | 403 µs | 558 µs |

### Notes

- At S=4096, `q512/k512` overflows L1 ("Statically allocated circular buffers … beyond max L1 size"). The test records it as `unsupported_L1`, not as a failure.
- One 32-replica bitwise check per S: identical.
- HiFi2 (the op default) is as accurate as HiFi4 here.

## 5. G3 — mHC: `mhc_split_sinkhorn` and `attn_res_weighted_reduce_nc` (`test_g3_mhc.py`)

### Sinkhorn: what the kernel computes and how Motif's maps relate

**Constants.** The consts are built by a copy of the d_p builder (`goldens.build_sinkhorn_consts`), with α = (−0.22, 0.33, 0.18) and checkpoint-like biases [R:01 §6.4].

**Kernel semantics** (from `mhc_split_sinkhorn_compute.cpp`):
- pre = σ(·) + eps
- post = 2σ(·)
- comb = Sinkhorn(exp(min(x, 80))), with eps added to every column divisor and no 1e-8 floor.

**Deriving Motif's maps:**
- **h_post = post / 2, exactly.** The measured ratio post / (2σ) is 1.000000.
- **h_pre = pre when eps = 0.** The SFPU sigmoid has a small bias: mean(pre − σ) is −1.7e-4 to −2.0e-4, and the max is 9e-4. With eps = 1e-6, pre is shifted by +eps and the column divisors change too, so use eps = 0.
- **H = comb.reshape(T, 4, 4)**, row-major. This is exactly Motif's M[i][j] = p_res[4i+j]. No transpose is needed; DeepSeek's wrapper applies combᵀ, Motif applies H.
- **Clamps.** Motif's ±10 (pre/post) and ±20 (res) clamps can be reproduced exactly by *pre-clamping the logits*: compute L = clamp(α·p + b) on device and call the kernel with identity SEL and zero base. Then exp(min(L, 80)) = exp(clamp(L, ±20)). That costs about 3 tiny ops per site. Only Motif's 1e-8 row/column floor remains different.

**Error against Motif's exact fp32 parametrization.** Each cell gives max|ΔH| for the direct mode / the pre-clamped mode.

| Logit regime | T=8 | T=32 | T=4096 |
|---|---|---|---|
| realistic (std 0.5), **gated** | 1.07e-3 / 0.76e-3 | 1.15e-3 / 0.91e-3 | 1.94e-3 / 1.44e-3 |
| moderate (std 3) | 2.0e-3 / 1.5e-3 | 2.8e-3 / 3.1e-3 | 5.0e-3 / 5.4e-3 |
| wide (std 10, \|L\| up to 33) | 2.6e-2 / 2.6e-3 | 0.46 / 4.9e-3 | 0.89 / 5.6e-3 |
| peaked (std 30) | 1.0 / 1.6e-3 | 1.0 / 4.2e-3 | 1.0 / **0.10** |
| degenerate rows (4 logits ≤ −20) | 3.5e-2 / 6.7e-4 | 3.7e-2 / 9.2e-4 | 4.1e-2 / 1.6e-3 |
| overflow (one logit at +90) | 1.8e-3 / 2.5e-3 | 2.5e-3 / 3.5e-3 | 4.3e-3 / 4.4e-3 |

**Pre and post errors** in all regimes: max|Δh_pre| ≤ 9.6e-4 and max|Δh_post| ≤ 1.1e-3. The thresholds are 2e-3.

**Arithmetic floor.** Against its own (no-clamp) semantics in fp32, the kernel is off by 1–2e-3 in realistic regimes. This is the TF32-class FPU in the RB/CB row and column sums. Column sums of H deviate from 1 by ≤ 2.3e-3, which is consistent with [R:04 §7.5].

**The one pre-clamped outlier.** The 0.10 for peaked logits at T=4096 comes from rare tokens where Motif's 1e-8 floor binds after clamping, which breaks Sinkhorn's scale invariance. For real weights, |α| ≤ 0.33 makes this regime implausible; C3 should confirm.

**Latency:**

| Shape | Traced per call | Eager |
|---|---|---|
| T=8 or T=32 | 52.2 µs (one core per 32-token tile) | 160 µs |
| T=4096 | | 368 µs |

At 106 calls per step this is **about 5.5 ms per decode step**; the design estimated 3–4 ms.

### Weighted reduce on `[1,4,T,4096]` bf16

**Variants:**
- **x_red:** weights `[1,4,T,1]` (R=1, C=4).
- **H·X:** weights `[4,4,T,1]` (R=4, C=4).
- **Fused post `[X ‖ out]`:** weights `[4,5,T,1]` (R=4, C=5).

**Golden:** fp32 einsum followed by one bf16 cast.

**Pass criterion (gated):**
- PCC ≥ 0.9999; and
- error ≤ C ulps of the largest summand, i.e. at most one extra rounding beyond the ≤ 0.5·C bound of a correct single rounding.

| Variant (fp32 weights) | PCC | Bit-exact vs einsum+cast | Worst error (summand ulps) | Traced T=8 / T=32 | Eager T=4096 |
|---|---|---|---|---|---|
| x_red R1 C4 | 0.999998+ | 88 % | 2.28–2.58 | 14.9 / 14.4 µs | 442 µs |
| H·X R4 C4 | 0.999998+ | 89 % | 2.37–2.56 | 27.7 / 27.5 µs | 815 µs |
| fused post R4 C5 | 0.999998+ | 88 % | 2.40–4.54 | 36.4 / 36.7 µs | 930 µs |
| bf16 weights (any variant) | 0.999997 | 58–62 % | 3.3–6.2 | 10.3–22.7 µs | 434–879 µs |
| composite fallback (fp32 `multiply` + `sum(dim=1)` + typecast) | 0.999999 | **100 %** | 2.00 | 102 / 232 µs | 27.7 ms |

**Decision:** use the op with fp32 weights. Its error amounts to at most one extra bf16 rounding per site; [R:01 A.5] puts the bf16 residual drift over 106 sublayers at PCC 0.999988. The composite is the bit-exact debugging path.

**Per-step cost:** 106 sites × (15 + 36) µs ≈ 5.4 ms.

## 6. G4 — CCLs with Motif payloads (`test_g4_ccl.py`)

### Setup

- Distinct data on every chip.
- The golden is a torch fp64 sum or concat per reduce group.
- Calls follow `gpt_oss/tt/experts_throughput/gather_decode.py`: `ttnn.all_gather(x, dim=2, cluster_axis=…)`, `ttnn.reduce_scatter`, `ttnn.all_reduce`, with default topology and links.
- Both fabrics pass every case. PCC is 1.0; the max-abs values are pure bf16 rounding of the sums.
- **Every chip in every reduce or gather group holds bitwise-identical results**, so the routing-consistency invariant of §2.3.7 holds. Eager and trace both pass.

### Results

| Payload (per chip) | Out | 2D torus traced | 1D ring traced | Eager (2D) |
|---|---|---|---|---|
| AR(cols) `[1,1,8,4096]` bf16 (wo, MoE out), DRAM or L1 | same | **33.4 µs** | 28.4 µs | 410–675 µs |
| AG(rows) `[1,1,8,4096]` → `[1,1,32,4096]`, TILE (composite path: padded tiles) | `[1,1,32,4096]` | 62.0 µs (L1: 58.3) | 59.5 µs | 592 µs |
| AG(rows), same, **ROW_MAJOR** | `[1,1,32,4096]` | **12.2 µs** | 11.1 µs | 132 µs |
| AG(rows) on dim 1, TILE | `[1,4,8,4096]` | 26.7 µs | 24.6 µs | 128 µs |
| AR(rows) `[1,1,32,4096]` bf16 (MoE combine) | same | **33.4 µs** | 30.3 µs | 291 µs |
| AR(rows) `[1,1,32,4096]` fp32 | same | 40.9 µs | 36.0 µs | 294 µs |
| AR(cols) stats `[1,1,8,3]` fp32 (PolyNorm TP stats), DRAM or L1 | same | **38.8 µs** | 40.0 µs | 790–1750 µs |
| RS(rows) `[1,1,32,4096]` → 8-row shards | `[1,1,8,4096]` | 109 µs | 105 µs | 1199 µs |
| Prefill RS(rows), S = 128 / 1024 | S/4 rows | 37.7 / 201 µs | 33.7 / 172 µs | 201 / 215 µs |
| Prefill AG(rows), S/4 = 32 / 256 | S rows | 26.7 / 183 µs | 24.6 / 159 µs | 144 / 193 µs |
| Prefill AR(cols), S = 128 / 1024 | same | 55.1 / 251 µs | 50.2 / 235 µs | 312 / 447 µs |

### Decisions

- **Fabric.** Keep `FABRIC_2D_TORUS_XY`, the plugin default; it is fully functional. The 1D ring is 5–15 % faster and remains a fallback if needed.
- **8-row token gather (decode step 2).** Use ROW_MAJOR for the all-gather, then tilize. It is 5× faster than TILE, whose padded tiles force the composite path.
- **Combine (decode step 9).** Keep AR(rows) + slice. It is 3.3× faster than an 8-row reduce-scatter (33 vs 109 µs), which confirms [R:04 risk 6].
- **Budget.** Each decode CCL costs about 30–40 µs traced, wall-clock slope, against 7–15 µs in design §1.3/§3.6. Per MoE layer that is about 33 + 12 + 33 + 39 + 33 ≈ 150 µs, or **about 7.9 ms per decode step** against the design's 2–4 ms.

## 7. G5 — fp32 router (`test_g5_router.py`)

### Pipeline

The router runs per chip and exactly as designed:
1. `ttnn.linear(x [.,.,M,4096] bf16, W [4096,384], dtype=float32, HiFi4, fp32_dest_acc_en=True)`
2. `ttnn.sigmoid` (fp32)
3. `+ bias` (fp32 `[1,1,1,384]`; −inf for pad columns)
4. `ttnn.topk(k=8)` on FLOAT32, which returns uint32 indices
5. `ttnn.gather` of the unbiased scores
6. sum, + 1e-20, reciprocal, × 2.0

### Statistics setup

- 32 chips × 128 distinct tokens give 4096 tokens per case.
- Inputs are f = γ_post ⊙ rmsnorm(h), with h either Gaussian or Student-t(4) plus outlier channels.
- Router weights come from the **real checkpoint** (layers 2 and 20: router weight, `expert_bias`, `post_attention_layernorm` γ), and from a synthetic set.
- The golden is fp64 torch on the same bf16-valued x and W.

### Top-8 set agreement

Target ≥ 99.9 %:

| Weights / inputs | bf16 W, fp32 out (draft 1) | fp32 W (named escalation) | Padded to 512 | bf16 router output |
|---|---|---|---|---|
| real L2, Gaussian | 99.707 % | 99.707 % (identical) | identical | 96.61 % |
| real L2, heavy-tailed | 99.658 % | identical | identical | 96.48 % |
| real L20, Gaussian | 99.683 % | identical | identical | 96.90 % |
| real L20, heavy-tailed | 99.634 % | identical | identical | 96.85 % |
| synthetic (σ_W 0.02, bias N(1.1, 0.03)), Gaussian / heavy | 99.658 % / 99.658 % | identical | identical | 96.68 % / 97.05 % |

**Expert bias.** Layer 2: mean 1.103, std 0.025. Layer 20: mean 1.831, std 0.068.

**Flips are near-ties.** Every mismatching token has a golden gap between the 8th and 9th biased score of ≤ 1.8e-4. The 1st percentile of that gap over all tokens is 3–7e-5. On tokens whose sets match, the routing weights (which sum to 2.0) differ by at most 3.5e-5–9.4e-5. With a bf16 router output the difference is 2.6e-4–3.2e-4.

### Diagnosis (`test_g5_router_diagnose`, real L2)

| Variant | Agreement | Logit RMS error (logit RMS 1.06) |
|---|---|---|
| HiFi4, fp32 acc (default) | 99.707 % | 3.29e-4 (max 1.75e-3) |
| + `packer_l1_acc` | 99.707 % | 3.29e-4 |
| K split into 8 matmuls, summed with fp32 SFPU adds | 99.707 % | 3.29e-4 |
| W mantissa split into 2 / 3 bf16 parts, fp32 adds | 99.756 % / 99.780 % | 3.14e-4 / 2.91e-4 |
| HiFi2, fp32 acc | 97.46 % | 4.28e-3 |

**The sigmoid and topk are not the cause:**
- The sigmoid is exact to 1.4e-7. An fp32 input enables fp32 dest, which selects the accurate exp and 2 Newton steps (`ckernel_sfpu_sigmoid.h`).
- On-device topk equals a host topk of the device scores.
- "Agreement if the sigmoid were exact" is identical to the measured agreement.

**The logit error is the only source.** It does not depend on how K is accumulated, and barely on product precision. That is consistent with TF32-class rounding of the FPU's partial dot products ([R:04 §7.5]; `docs/…/fp32_accuracy.rst` says the FPU uses bf16/TF32).

**fp32 weights cannot help.** x and W are bf16-valued, so both operand formats carry identical values and produce bit-identical logits.

### Consistency and latency

**Consistency.** With replicated input, the indices and weights are **bitwise identical on all 32 chips**.

**Latency**, traced, decode `[32,4096]` → top-8:

| Variant | Full router | Linear + sigmoid | topk | Eager |
|---|---|---|---|---|
| bf16 W, width 384 | 189 µs | 55 µs | 63 µs | 0.95–2.1 ms |
| fp32 W | 199 µs | 64 µs | | |
| Padded to 512 | 218–228 µs | | 83 µs | |

At 51 layers this is about 9.6 ms per step.

### Decision

**The named escalation (HiFi4 + fp32 weights) is ineffective.** Keep bf16 weights: fp32 weights only double the DRAM footprint. Keep width 384: padding to 512 is slower and unnecessary. The choice of what to do next is in §11.

## 8. G6 — dense-local expert matmuls and composite PolyNorm (`test_g6_experts.py`)

### Data

- Chip 0's real experts 0–11 of layer 2, sliced per expert from the local BF16 shards (`gate_up_proj`, `down_proj`, `act_fn`), converted to bfloat8_b in DRAM.
- σ(w) ∈ [0.32, 0.66] and the clamped bias b ∈ [−0.35, −0.04].
- Inputs are f = γ_post ⊙ rmsnorm(h), repeated to `[1,12,32,4096]`.
- The bfp8 weight bytes are 133.7 MB (gate_up) and 66.8 MB (down).

### Matmuls

Traced latency and effective weight bandwidth:

| Matmul | Program config | HiFi2 | HiFi4 | PCC (HiFi4) vs fp64 on bfp8 weights |
|---|---|---|---|---|
| gate_up `[1,12,32,4096]@[1,12,4096,2560]` | default (auto) | 983 µs, 136 GB/s | 982 µs | 0.999999 (0.99997 vs the bf16 weights) |
| | `MatmulMultiCoreReuseProgramConfig` | rejected: TT_FATAL "N (80) must equal per_core_N", only batch×M is split (12 cores) | | |
| | **1D mcast**, grid 10×8 = 80 cores, in0_block_w 8, per_core_N 1 | **426 µs, 314 GB/s** | 433 µs, 309 GB/s | 0.999999 |
| | 1D mcast, 40 cores, in0_block_w 8 / 16 | 476 / 484 µs | 472 / 496 µs | |
| | 1D mcast, 20 cores | 487 µs | 501 µs | |
| | flattened `[32,4096]@[4096,30720]`, DRAM-sharded | L1 overflow (1.6–4.0 MB CBs) on 16–64 cores | | |
| down `[1,12,32,1280]@[1,12,1280,4096]` | default | 289 µs, 231 GB/s | 291 µs | 0.999999 |
| | **1D mcast**, 8×4 = 32 cores, in0_block_w 4, per_core_N 4 | **222 µs, 301 GB/s** | 226 µs, 295 GB/s | |
| | 1D mcast, 64 cores / 16 cores | 241 / 258 µs | 238 / 275 µs | |
| | flattened K-concat `[32,15360]@[15360,4096]`, DRAM-sharded, in0 on 16 cores | 225 µs, 297 GB/s | | 0.99999 |

**Why the default is slow.** It splits only batch×M, so it runs on about 12 cores.

**With the 1D-multicast configs:**
- One MoE layer reads its 200 MB of bfp8 expert weights in **648 µs**.
- At 51 layers that is **about 33 ms per decode step**, which is the bandwidth floor of the dense-local draft 1. The design §3.6 range assumed 150–300 GB/s; we get about 300.
- HiFi4 costs nothing (the op is bandwidth-bound) and cuts the max error 2.6× (0.0085 vs 0.022), so use HiFi4.
- The flattened gate_up formulation is possible (all 12 experts see the same 32 tokens) but needs a narrower per-core N. The K-concat down (routing weights folded into h) matches the batched 1D-mcast speed and removes the separate weighted sum.

### Composite PolyNorm on `[1,12,32,1280]`

The composite is 13 ops: slice ×2, typecast, mul ×2, `rms_norm(eps 1e-6, no weight)` ×3, mul ×3 with `[1,12,1,1]` coefficients, add ×3, and mul by up. The golden is fp32 HF `GroupedPolyNorm` without the ×0.5, which is folded into W_down.

| Intermediates | PCC | max\|Δ\| (\|h\| ≤ 24) | Relative Frobenius error | Traced | Eager |
|---|---|---|---|---|---|
| fp32 | 0.9999990 | 0.060 | 1.7e-3 | 294 µs | 3.2 ms |
| bf16 (fp32 dest acc in `rms_norm`) | 0.999991 | 0.155 | 5.0e-3 | 164 µs | 2.7–2.9 ms |

- Both pass PCC ≥ 0.9995.
- The `[1,12,1,1]` per-expert broadcast works (binary_ng) for fp32 and bf16.
- At 51 layers this is 8.4–15 ms per step. The fused PolyNorm kernel (v1) is the lever.

## 9. G7 — `paged_update_cache` / `paged_fill_cache` (`test_g7_paged_cache.py`)

### Setup

| Item | Value |
|---|---|
| Cache | `[8·1024/block, 1, block, 576]` bfloat8_b, block 32 and 64 |
| Contents | random, pre-quantized through a host bfp8 round trip |
| `page_table` | random permutation, int32 `[8, 1024/block]` |
| Decode input | `[1,8,32,576]` bf16, HEIGHT_SHARDED one user per core; the op requires `num_cores == num_users` |
| Positions | two steps, including block edges, position 1023, and one lane at −1 |
| Prefill input | `[1,1,S,576]` **bfloat8_b**, S ∈ {256, 1024}; `batch_idx=u` and `batch_idx_tensor=[u]` |

### Results

**Every check is bit-identical** (`torch.equal`) on devices 0, 7, 13 and 31:
- the written rows;
- all untouched rows;
- the −1 lane, which stays untouched.

### Latency

| Operation | Time |
|---|---|
| Decode update (traced) | 4.8 µs |
| Decode update (eager) | 46–50 µs |
| Prefill fill, S=1024 (eager) | 39–56 µs |
| Prefill fill, S=4096 (eager) | 109–294 µs |

### Notes

- **Fill dtype.** `paged_fill_cache` is a raw tile copy, so its input dtype must equal the cache dtype. Prefill must typecast the latent to bfloat8_b before the fill. This case was deliberately not exercised, because a mismatched dtype could corrupt DRAM if the check were absent.
- **KV-pool allocation** (for `allocate_kv_cache`, design §5.1), measured on 1/8 of one layer's 264K-token pool (20 MB per chip):

  | Method | Time for 1/8 layer | Extrapolated to the full pool (53 layers) |
  |---|---|---|
  | `ttnn.zeros(bfloat8_b)` (host-built, then copied to 32 chips) | 41–47 ms | **17.5–20 s** |
  | `ttnn.empty` + on-device `ttnn.fill(0)` | 0.4–2.8 ms | **0.2–1.2 s** |

  Use empty + fill.

## 10. G8 — `ttnn.experimental.rotary_embedding_hf` (`test_g8_rope.py`)

### Setup

- **Tables:** HF half-split (NeoX), from host fp32 angles rounded to bf16. Global layers use the YaRN inv_freq (ramp over [10, 23]; matches the [R:01 A.1] values); SWA layers use plain θ = 1e4.
- **Golden:** x·cos + rotate_half(x)·sin in fp32.

### Results

**Decode** (per-user positions 0, 1, 127, 128, 129, 1000, 5000, 32767):
- q_pe `[1,8,10,64]` and k_pe `[1,8,1,64]` are HEIGHT_SHARDED `[32,64]` on 8 cores; cos and sin `[1,8,1,64]` are sharded on the same grid.

| Path | PCC | Traced | Eager |
|---|---|---|---|
| Fused op (default compute) | 0.999996–0.999997 | ≤ 3 µs (hidden under sync, raw bound) | 94–150 µs |
| Fused op, HiFi4 + fp32 acc | 0.999998 | ≤ 3 µs | |
| Composite x·cos + (x@R)·sin (R the exact ±1 rotate-half matrix) | 0.999998 | 19.4 µs | 360–690 µs |

**Prefill** (S = 128 / 1024 / 4096; q_pe `[1,10,S,64]`, k_pe `[1,1,S,64]`):

| Path | PCC | Eager |
|---|---|---|
| Fused op | ≥ 0.999996 | 200–480 µs |
| Composite | ≥ 0.999997 | |

**Decision:** use the fused op in both modes, with HiFi4 + fp32 acc (it halves the max error, from 0.029 to 0.013). The composite is the fallback.

## 11. Open issues

**1. G5 router agreement (decision needed).**
- Every on-device fp32 variant plateaus at 99.63–99.78 % top-8 set agreement on random and real-weight inputs, against the 99.9 % target. All flips are near-ties (gap ≤ 1.8e-4).
- The cause is the precision of the FPU's partial dot products; the named escalation does not help.
- Options, in suggested order:
  - **(a)** Measure agreement on real C1 router inputs and the teacher-forced top-1 / logits impact. If the model-level target holds, accept for draft 1.
  - **(b)** Implement an exact-fp32 router as an SFPU dot-product kernel through `ttnn.generic_op` (32×4096×384 per chip; the Option-B Sinkhorn route already plans `generic_op`).
  - **(c)** Re-score only near-tie candidates in exact arithmetic (not trace-friendly).

**2. Upstream tt-metal bug: SDPA decode with dynamic chunking and a sliding window.**
- See §3 for the cause. It affects any caller that omits `program_config` or uses `k_chunk_size=0`, including `paged_scaled_dot_product_attention_decode`.
- Our modules must always pass `k_chunk_size=128`. File an upstream issue.

**3. The latency model needs updating with measured traced costs.** The design's 7–15 µs per CCL and 1.5–2.5 µs per small op are optimistic on this mesh.

| Component | Measured |
|---|---|
| Each decode CCL | 30–40 µs |
| A tiny eltwise op in trace | 5.7–6.7 µs |
| Sinkhorn, per call | 52 µs |
| Router, per layer | 189 µs |
| PolyNorm | 164–294 µs |
| Experts (bfp8, best configs) | 648 µs |
| Stream mixes, per site | 51 µs |

Rough MoE-layer decode sum from gate costs alone is about 1.5–1.7 ms, i.e. 75–90 ms per step for 51 layers, which is the upper half of design §3.6. The main v1 levers are:
- the fused PolyNorm kernel;
- a cheaper Sinkhorn (Option C/D);
- sparse experts;
- fewer, fused CCLs.

**4. mHC logit ranges (CPU check C3).** These decide between the direct, pre-clamped and Option-B call modes for Sinkhorn. The stock op meets the thresholds at realistic spreads (std ≈ 0.5). At std 3 it reaches the 5e-3 bound (5.0–5.4e-3 at T=4096) from TF32-class arithmetic alone. Once res logits pass ±20 or pre/post logits pass ±10, Motif's clamps matter and pre-clamping (or Option B) is required.

**5. Expert matmul program configs** must be set explicitly (§8), because the auto config reaches only 12 cores. The flattened DRAM-sharded gate_up needs a different core split to fit L1. This was not pursued.

**6. Shared-host incident (for whoever runs next).**
1. An exception inside `begin_trace_capture` / `end_trace_capture` left the mesh in capture mode, and teardown hung in `close_mesh_device` (`20261001_172023_g1g2g3.log`).
2. After SIGTERM and `tt-smi -r`, one Ethernet edge (ASIC 2628543044524051161 ↔ 8797283523083441937) trained 1 of 2 channels, and FABRIC_2D_TORUS_XY mapping failed ("Graph specified in MGD could not fit").
3. `tt-smi -glx_reset_auto` needs `sudo ipmitool` and fails on this host, leaving the PCIe devices needing a re-init ("No such device"). `scripts/devreset.sh` has since been changed to refuse the `glx` mode.
4. Two more `tt-smi -r` resets restored the full torus (`20261001_173715_g0_probe3.log`).

The harness now always ends and releases a capture on exceptions, frees outputs during capture, and never accumulates L1 outputs.

**7. Reference cross-check.** The local goldens were checked against `models/demos/motif3/reference` only for Sinkhorn (exact); that package's API was still changing. Re-point the goldens once it is stable.

---

## 12. Feature gates G9-G13a: chunked prefill, prefix caching, MTP speculative decoding

**Run:** 2026-10-02 16:56-17:21 UTC, host `bh-glx-exp-a03u07`, tt-metal `motif3-bh-galaxy` at `cfd102b90fc` (Release
build; the `models/` tree carried other work packages' uncommitted edits, none of which these op-level gates import).
**Mesh:** logical (4, 8); fabric requested `FABRIC_2D_TORUS_XY`, **committed TORUS_Y** (the 4-chip DP axis is a line, as
in FMV), `l1_small_size` 32768, `trace_region_size` 256 MiB, dispatch axis COL (the serving set).
**Design:** `docs/features/FEATURES_DESIGN.md` §4 (G9-G13, review R6/R9/R10), D1-D6, D10, §3.1-§3.5, Appendix C.3
(lead decisions: KV-R adopted and measured here; `tt/kv_write.py` owned by KVW; vLLM budget per G9). Op level only: no
model code, synthetic data, torch fp32 / fp64 goldens on host-dequantized bfp8 data (the G2/G7 method). G13b (the
full-model KV-R run) belongs to WP5.

| Item | Location |
|---|---|
| Tests | `test_g9_resumed_global_attn.py`, `test_g10_swa_tail.py`, `test_g11_chunk_io.py`, `test_g12_spec_kv_alias.py`, `test_g13_kv_replicated_decode.py` |
| Raw results | `results/G9.jsonl` ... `results/G13.jsonl` (append-only, last record per case wins); `results/G*_quick.jsonl` = the `MOTIF3_GATES_QUICK=1` smoke run |
| Host self-checks | every file's `test_*_host_*` (goldens, page tables, lane layouts, probe discrimination on a host emulation): `scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q <file> -k host` |
| Run | `scripts/devrun.sh -t 1800 -n gates_g9_g13 -- python -m pytest models/demos/motif3/tests/unit/gates/test_g{9,10,11,12,13}_*.py -s -p no:cacheprovider -k "not host"` (about 6 min for all five) |
| Knobs | G9: `MOTIF3_G9_ROLES` (`fp32acc,bf16acc`), `MOTIF3_G9_CS` (bucket list) |
| Logs (`logs/dev/`) | final clean run of all five gates (11 passed, 3 min): `20261002_171722_gates_g9_g13_final.log`; development runs: `20261002_152147_wp0_gates_F.log` (smoke; G10, G11, first G12), `20261002_160208_wp0_gates_G.log` (G9 bf16acc + reference ops, G13 full), `20261002_170543_g9fp32_g12.log` (G9 fp32acc, G12 final), `20261002_170749_g9_buckets_256_1024_4096.log`, `20261002_171142_g13a_correctness_rerun.log`, `20261002_172201_g13a_serving_fp32acc.log` (serving order with the fp32-acc sp1 op), `20261002_171023_g13_dag_diag.log` (diagnostic) |

### 12.1 Verdicts

| Gate | Verdict | Decision / key numbers |
|---|---|---|
| **G9** resumed global attention | **PASS with fp32 dest accumulation; FAIL with the `sdpa_prefill` role** | `chunked_scaled_dot_product_attention(Q [1,10,C,576], K = V = latent cache, page_table [1,640], chunk_start_idx_tensor)` is correct and trace-safe: one program per (C, q, k) over all starts, trace captured at s = 128 replays bitwise equal to eager at every start, latent columns bitwise equal to `chunked_flash_mla_prefill` at 1.03-1.07x its time (G9b C++ patch **not needed**). With the `sdpa_prefill` role (bf16 dest) PCC is 0.9978-0.9991 and the worst row down to 0.988 (all 68 cases fail); with HiFi4 + **fp32 dest acc** (the window-free `sdpa_prefill_fp32` role) PCC 0.99983-0.99999, worst row >= 0.99965 (all 68 cases pass). Per-bucket (q, k): 128/128 for C = 128, 64/64 for C = 256-1024, 128/128 for C >= 2048 (§12.2) -> vLLM budget **8064** |
| **G10** SWA tail | **PASS** (a) and (b) | Tensor-args dim-0 slice of the `[4129,1,64,576]` bfp8 cache: bitwise for every block id, one program, trace-safe. Square `[tail 128 ‖ chunk]` causal + window-129 SDPA: **bitwise equal to the draft-1 single-shot rows `[s, s+C)`** in all 9 (C, s) cases (PCC 0.99971-0.99973 vs fp32), probes exact, window 128 / 130 negatives detected. S-C (C++ window patch) not needed |
| **G11** chunk I/O | **PASS** | `paged_fill_cache` through fill tables with -1 (leading / trailing / mixed / 4-row packed): bit-exact on 32 chips, skipped blocks untouched. Offset RoPE via `ttnn.embedding` gather: tables bitwise, `rotary_embedding_hf` PCC >= 0.999997, bitwise equal to the draft-1 tables at c0 = 0, no program per offset, trace replay == eager |
| **G12** spec KV aliasing | **PASS** (one documented limitation) | A/B split bit-exact on all 32 chips (per-row 8-lane and 32-lane gathered, bfp8 and bf16); a single call loses 2-3 (8-lane) / 10-13 (32-lane) updates in **every** trial of three runs, only on same-tile pairs (`p % 64` in {0, 30, 62}). FlashMLA with partner rows: probes exact (row p never sees p+1, row p+1 sees p) in both layouts, SWA and global; **lane relocation bitwise** (review R5: 28 users moved to another lane on another DP row give identical outputs). Limitation: a bf16 cache cannot take one 32-lane `paged_update_cache` (output CB `B x Wt` = 1.18 MB, L1 clash); 2 x 16-lane calls are bit-exact |
| **G13a** KV-R decode (op level) | **PASS** | KV-R invariant bit-exact on all 32 chips for every write variant, ordinary and verify steps, trace == eager; cross-row partners without KV-R read a stale anchor (negative control: partner PCC down to 0.40). Cost in a 54-layer decode-sized trace: **`all_split` +1.87 ms (1K ctx) / +1.89 ms (8K ctx) per step vs draft-1 `row`** (34.6-34.9 us per layer; +1.68-1.69 ms vs `row_split`) <= 2.0 ms gate (margin 0.11 ms); deferred KV-R +1.35-1.36 ms. Prefill 8K sp0/sp1 after the KV-R decode trace: no static-CB clash, bitwise-identical outputs, no program after capture, main L1 unchanged; trace +0.46 MB per 54 layers |

No named fallback was triggered: G9b (MLA tensor-start C++ patch), S-C (window through the chunked wrappers), per-block
fills, "spec blocked", and deferred KV-R (Δ > 2 ms) all stay unused. The one design change is G9's compute role.

### 12.2 Decisions published to the work packages

**WP1 (`model_config` / `prefill_plan.py`) and WP2 (attention sp1 global):**

| Bucket C | sp1 global (q, k) | A = lcm(64, q, k) | ms / global layer at s = 128 / 2048 / 8192 / 24448 (fp32 acc, eager) | Runner-up |
|---|---|---|---|---|
| 128 | 128 / 128 | 128 | 0.30 / 1.00 / 3.68 / 10.77 | 64/64: 0.39 / 1.18 / 4.38 / 12.82 (A = 64); 32/64: 0.60 / 1.07 / 3.94 / 11.55 (A = 64) |
| 256 | 64 / 64 | 64 | 0.60 / 1.21 / 4.40 / 12.87 | 128/128: 0.38 / 2.02 / 7.38 / 21.54 |
| 512 | 64 / 64 | 64 | 0.38 / 1.28 / 4.48 / 12.94 | 128/128: 0.60 / 2.13 / 7.48 / 21.64 |
| 1024 | 64 / 64 | 64 | 0.50 / 1.48 / 4.80 / 13.74 | 128/128: 0.71 / 2.38 / 7.73 / 21.89 |
| 2048 | 128 / 128 | 128 | 1.22 / 2.95 / 8.54 / 23.59 | 64/64: 1.67 / 4.12 / 11.97 / 32.64 |
| 4096 | 128 / 128 | 128 | 4.21 / 7.70 / 19.12 / 49.87 | 64/64: 4.97 / 9.14 / 22.56 / 58.86 |
| 8192 | 128 / 128 | 128 | 12.28 / 17.86 / 36.47 / 87.06 | 64/64: 18.85 / 27.30 / 54.49 / 127.24 |

* **Compute config of the sp1 global op: HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False`** (the existing
  `sdpa_prefill_fp32` role; the op is window-free, so the legacy kernel's window-mask bug does not apply). The
  `sdpa_prefill` role (bf16 dest) accumulates the 576-wide QK^T in bf16 and misses the bar everywhere (§12.3).
* **Alignment.** A start must be a multiple of q and k (§12.3, negative control). With the table, `A(C) = 64` for
  C <= 1024 and 128 for C >= 2048 (C = 128 may use 32/64 instead, A = 64, at +7 % attention time for s >= 2048, to
  avoid recomputing 64 rows behind 64-granular prefix hits; 64/64 costs +18-19 %). The planner resolves `c0 = floor(s / A(C)) * A(C)` with C the bucket of `e - c0`
  (A is monotone in C, so at most one re-plan). A uniform A = 128 with the same (q, k) per bucket is also valid (64/64
  serves 128-aligned starts) at up to 127 recomputed rows.
* **vLLM:** `--max-num-batched-tokens 8064 --long-prefill-token-threshold 8064` (lead decision 4: threshold = budget).
  8064 = 8192 - 128 keeps every span `e - c0 <= 8064 + 127 = 8191` inside the 8192 bucket, and lone-request chunk ends
  are multiples of 128, so they need no recompute. (8128 would be correct only if every bucket had A = 64.)
* **SDPA page table** `[1, 640]` (A = 64 starts like 8256 verified), padded with block 0; fill tables `[1, C/64]` with
  -1 (G11). The flexible start has **no device-side check**: a start of 96 with q = 64 silently answered as start 64
  (PCC 0.99924 vs the floored golden, 0.869 vs the intended one), so the generator must assert `start % q == 0` and
  `start % k == 0` before every sp1 call (design R6).
* **SWA sp1 (S-A):** tail = two tensor-args slices (one `[4]` start / end pair per tail block, shared by all SWA layers)
  -> `concat(dim=2)` in bfp8 -> typecast bf16 (16.3-16.5 us traced per layer; bf16-then-concat 18.4 us); square SDPA at
  q/k 128/128 with the `sdpa_prefill` role (no fp32 acc with a window), rows `[128, 128+C)`.
* **Offset RoPE:** `ttnn.embedding(idx [1, C] uint32, table [1,1,32768,64] RM, layout=TILE)` -> reshape `[1,1,C,64]`;
  traced 5 / 51 / 198 us per kind (cos + sin) at C = 128 / 2048 / 8192, once per chunk.
* **sp1 cost model (for `prefill_cost_table` / `prefill_plan.prefill_cost_model`).** Per global layer the measured time is
  close to linear in the prefix: `t(C, s) = t0(C) + slope(C) * s` with, for the recommended (q, k), slope = 0.43 / 0.50 /
  0.52 / 0.54 / 0.92 / 1.88 / 3.07 us per prefix key and t0 = 0.30 / 0.60 / 0.38 / 0.50 / 1.22 / 4.21 / 12.28 ms at
  C = 128 / 256 / 512 / 1024 / 2048 / 4096 / 8192 (x 14 global layers per chunk). The current single constant
  (`DEFAULT_SP1_ATTN_S_PER_ROW_KEY` = 6.5e-9 s per row-key for 14 layers) matches C >= 1024 (6.1-7.7e-9 measured) but
  underestimates short chunks behind long prefixes (C = 512: 1.5e-8, C = 256: 2.9e-8, C = 128: 4.8e-8), where the op is
  bound by K / V streaming, not FLOPs. sp0 chunks keep the draft-1 table.

**KVW (`tt/kv_write.py`), WP5:** G13a passed, so the `all_split` integration may start. Recipe (per layer, before
FlashMLA): `g = ccl.ag_dp_rows(kv_row [1,1,8,576])` -> `transpose(g, 1, 2, memory_config=32-core HEIGHT_SHARDED [32,576])`
-> `paged_update_cache(cache, u, update_idxs_tensor=cur_a [32], page_table=pt_all [32,W])` -> the same with `cur_b`
(all -1 in ordinary steps, 0.2 us). Measured parts (traced): AG(dp) of `[8,576]` 24.0 us (latency-bound; the DP axis is
a line under TORUS_Y), gather + transpose to the sharded layout 31.9 us, 32-lane update 6.5 us. The sharded alternative
(`all_gather(dim=1)` of the 8-core sharded `[1,8,1,576]` straight into the 32-core layout) costs the same (31.8 us,
bitwise-identical results); the DRAM-AG variant is slower (35.7 us). With a **bf16 KV cache** issue the 32 lanes as two
16-lane calls per kind (the op's output CB is `B x Wt` tiles per core, §12.6). FlashMLA keeps the per-row
`cur_pos [8]` / `page_table [8, W]`.

**WP4/WP5, plugin:** packed partners on other DP rows are valid only with KV-R (negative control §12.7); with
`row_split` (no APC) they must stay on the owner's row. Lane relocation is bitwise at the FlashMLA level (R5 evidence;
G-S5 still owns the full-model probe).

### 12.3 G9 — resumed global attention (`test_g9_resumed_global_attn.py`)

**Setup.** One virtual 32,640-token latent sequence (random N(0,1), bfp8-quantized on the host) in a shuffled
`[520, 1, 64, 576]` bfp8 cache (block 0 = null). Q `[1,10,C,576]` = N(0,1) x 0.1447 in bf16 (the global softmax scale
folded into q, op `scale=1.0`). Persistent page table `[1,640]` (real ids for `[0, cdiv(s+C, 64))`, then 0) and start
`[1]`, rewritten with `copy_host_to_device_tensor` like the generator will. Golden: fp32 causal latent attention with
V = K over keys `[0, s+i]` (all rows for C <= 2048, ~2.3K sampled rows incl. every 256-row seam for C >= 4096). Cases:
C in {128, 256, 512, 1024, 2048, 4096, 8192} x s in {128, 2048, 8192, 24448} (+ s = 8256 for the A = 64 configs) x
(q, k) in {64/64, 128/128} (+ 32/64 at C = 128) x role in {fp32acc, bf16acc}: 2 x 68 cases.

**Accuracy** (PCC on columns `[:512]`; worst = worst query row over 10 heads x 512):

| Role | Min PCC | Max PCC | Worst row | Bar (PCC >= 0.999, target 0.9995, worst row >= 0.998) |
|---|---|---|---|---|
| fp32acc (HiFi4, fp32 dest; legacy kernel) | 0.999827 (C 8192, s 24448, 64/64) | 0.999994 | 0.99965 | **pass, target met in all 68 cases** |
| bf16acc (`sdpa_prefill`: HiFi4, bf16 dest; streaming kernel) | 0.997834 | 0.999126 | 0.98784 | **fail in all 68 cases** (PCC reaches 0.999 only at s = 128, where the worst row is 0.997) |

PCC falls slowly with the prefix (fp32acc: 0.99999 at s = 128, 0.99983-0.99989 at s = 24448). The k_pe columns track
the latent ones. The bf16-dest loss comes from accumulating QK^T over 18 tiles (576 dims) in a bf16 destination, three
times the 192-dim expanded case G2 validated; a bf16 latent cache does not help (0.99794 at C 2048 / s 8192, both
configs). Non-finite values: 0. Replicas: bitwise identical (checked on 3-5 chips per configuration).

**Mechanism** (both roles, every (C, q, k)): program cache **+1 over all starts**, **+0 for the trace**; a trace
captured at s = 128 replayed at every other start (start and page table rewritten in place) is **bitwise equal to
eager**. No configuration of either role hit an L1 / static-CB limit.

**Cost** (fp32acc, eager ms per call; §12.2 has the full table): at short prefixes the op is FLOP-bound (C 8192 at
s 128: 12.3 ms, 65 TFLOP/s per chip); at long prefixes it is bound by K / V streaming (every (head, q-chunk) work unit
reads its whole prefix, no sharing in causal mode, design R10: at s = 24448 the time hardly depends on C between 256 and
1024), and the better q chunk is an empirical function of C: 64/64 for C = 256-1024 (1.6-1.7x faster than 128/128 at
s >= 2048), 128/128 for C = 128 (1.2x) and C >= 2048 (1.2-1.46x). bf16acc is faster at 128/128 (e.g. C 8192 / s 24448:
69.5 vs 87.1 ms) but fails accuracy; at 64/64 the two roles cost about the same.

**Comparison ops:**

| Item | Result |
|---|---|
| K = V op vs `chunked_flash_mla_prefill` (scalar start, one program per start), same role (bf16acc) | latent columns **bitwise equal**; time ratio 1.026-1.070 over (C, s) in {(128, 24448), (512, 8192), (2048, 8192), (2048, 24448), (8192, 8192), (8192, 24448)} x both (q, k). Kill criterion (> 1.5x at C 2048 / s 8192) not met: 1.03 (128/128), 1.06 (64/64) -> **no G9b** |
| draft-1 expanded global SDPA (sp0, q/k 256/256) | 0.22 / 0.22 / 0.47 / 3.85 ms eager at C = 128 / 512 / 2048 / 8192 (traced 0.027 / 0.16 ms at 128 / 512); the absorbed sp1 op costs ~3.2x at C 8192 even at s = 128 (FLOPs 576 + 576 vs 192 + 192) |
| negative control: start 96 with q = 64 | detected: PCC 0.869 vs the intended start, 0.99924 vs the floored start 64 (the kernels use `start // q_chunk`) |
| negative control: page table shifted by one block | detected: PCC 0.968 |

**Impact on the design's estimates [I, from the table]:** a 32K prompt in 8064-token chunks spends about 200 ms per
global layer in sp1 attention (chunks at s = 8064, 16128, 24192 plus a 512-row tail at 32256) instead of 88.8 ms
single-shot, i.e. **+1.6 s** over 14 layers on 23.4 s (+7 %); a 128-row suffix behind a 24K hit costs ~10.8 ms per
global layer (~0.15 s per chunk), consistent with design §8's 0.8-0.9 s TTFT for a 30K document + short question.

### 12.4 G10 — SWA tail (`test_g10_swa_tail.py`)

**(a) Tail gather** on a random `[4129, 1, 64, 576]` bfp8 cache (the serving pool size): `ttnn.slice(cache, start [blk,0,0,0],
end [blk+1,1,64,576], slice_dim=0, num_devices=4129)` with persistent `[4]` int32 bounds.

| Check | Result |
|---|---|
| blk in {1, 2, 2063, 4127, 4128, 2937, 2297, 317}, eager | bitwise equal to the host block on chips 0 / 13 / 31 (never equal to the neighbour) |
| programs over all block ids | +1 |
| one slice captured, replayed with rewritten bounds | bitwise |
| tail = 2 slices -> concat(dim 2) -> bf16, 5 block pairs incl. (4127, 4128), (4128, 1), eager and traced | bitwise; **16.3-16.5 us traced** (concat in bfp8, then typecast), 18.4 us (typecast first); 0.61 / 0.72 ms eager (4 dispatches) |

**(b) Square windowed SDPA** (`Q_cat [1,10,128+C,192]` rows < 128 zero, `K_cat / V_cat [1,2,128+C,192]` = keys
`[s-128, s+C)`, causal, window 129, q/k 128/128, `sdpa_prefill`), random N(0,1) data (q x 0.0722), V zero-padded
128 -> 192:

| C | s | PCC square vs fp32 window attention | square vs draft-1 single shot rows [s, s+C) |
|---|---|---|---|
| 128 | 128 / 4096 / 30720 | 0.99970 / 0.99973 / 0.99971 | **bitwise equal** (all three) |
| 1024 | 128 / 4096 / 30720 | 0.99971 / 0.99972 / 0.99972 | **bitwise equal** |
| 8192 | 128 / 4096 / 24576 | 0.99972 / 0.99972 / 0.99972 | **bitwise equal** |

Bitwise equality holds because `s - 128` is a multiple of the 128-key chunk, so both layouts visit the same K chunks in
the same order. **Probes** (chunk rows {0, 1, 63, 64, 127, C-1}; orthonormal query directions; key p-128 = +1,
p-129 = -1 with a larger score, p+1 = +3 with the largest score, each in its own 16 value dims): every row outputs
+1 within 1e-3 at C = 128 / 1024 / 8192; window 128 (p-128 lost) and window 130 (p-129 leaks) are both detected.
**Cost** (eager): square 0.28 / 0.23 / 0.59 ms vs the draft-1 C-row SDPA 0.10 / 0.10 / 0.49 ms at C = 128 / 1024 /
8192; the small-C ratios are dispatch-bound, at C = 8192 the square call costs 1.2x.

### 12.5 G11 — chunk I/O (`test_g11_chunk_io.py`)

**(a) Fills** into a random `[520, 1, 64, 576]` bfp8 cache, x pre-quantized (exact expectation), 14 cases:

| C | Tables (skipped blocks) | Result |
|---|---|---|
| 128 | all real (0), leading -1 (1), trailing -1 (1), mixed (2) | bit-exact |
| 2048 | all real, leading (8), trailing (8), mixed (5), 4-row packed (10) | bit-exact |
| 8192 | all real, leading (32), trailing (32), mixed (5), 4-row packed (10) | bit-exact |

Checked on chips 0 / 13 / 31 after every fill (written blocks exact, -1 blocks and all other blocks unchanged) and on all
32 chips at the end. Eager 0.29 ms per fill at every C (dispatch-bound). The fill page-table width is C/64 (2 entries at
C = 128 work; the stick is padded by the buffer alignment).

**(b) Offset RoPE**, C in {128, 2048, 8192} x c0 in {0, 128, 24576, 32704} (the last clamps to row 32767) x {yarn,
plain}: gathered tables bitwise equal to the host rows on chips 0 and 31; `rotary_embedding_hf` (rope role) PCC
0.999997-0.999998 for q_pe `[1,10,C,64]` and k_pe `[1,1,C,64]`; at c0 = 0 bitwise equal to the draft-1 path (TILE
tables uploaded from the host); 3 programs for the first case per C, **0** for the other 7 offsets / kinds and for the
trace; gather + rotary captured at c0 = 128 and replayed at every c0: bitwise equal to eager.

### 12.6 G12 — spec KV aliasing (`test_g12_spec_kv_alias.py`)

**Layout.** 32 lanes, 8 blocks each; per DP row: owners in slots 0-2 at `p` with `p % 64` cycling through
{0, 30, 31, 62, 63}, partners in slots 4-6 at `p + 1` with the owner's page-table row, a plain lane (slot 3), an idle lane
(slot 7, -1). "row": partners on the owner's DP row, per-row tensors (`row_split`). "all": replicated 32-lane tensors,
the partner of owner (r, k) on row r + 1 (`all_split`, needs KV-R). Rows pre-quantized; the whole cache compared on all
32 chips (each chip against its row's expectation).

| Case | bfp8 | bf16 |
|---|---|---|
| A/B split, 8-lane per row | **bit-exact, 32 chips** | **bit-exact, 32 chips** |
| A/B split, 32-lane gathered | **bit-exact, 32 chips** | one 32-lane call: **TT_THROW** (static CB region ends at 1,516,576 B, the sharded input sits at 1,466,368 B); as **2 x 16-lane calls: bit-exact** |
| single call, 6 trials, 8-lane (row 0) | lost 2-3 updates in every trial (3 runs) | lost 2-3 in every trial |
| single call, 6 trials, 32-lane | lost 10-13 in every trial (3 runs) | (not run: see above) |

Every lost update is on a same-tile pair (`p % 64` in {0, 30, 62}); never on 31 (tile seam), 63 (block seam) or a plain
lane. A pair can lose both rows (the 18 tile writes of the two cores interleave), which is why the 32-lane count can exceed
the 8 same-tile pairs. The split is mandatory: the race fired in every trial.

**bf16 limitation.** `paged_update_cache_program_factory.cpp` sizes the output CB as `num_output_tiles = B * Wt` per
core (B = users of the call): 32 x 18 tiles x 2048 B = 1.18 MB for a bf16 cache (627 KB for bfloat8_b). A bf16 KV cache
(`MOTIF3_KV_CACHE_DTYPE=bf16`) under KV-R must therefore split each 32-lane call into <= 16-lane calls.

**FlashMLA with partner rows** (G1 config: k_chunk 128, `max_cores_per_head_batch` 16, HiFi4 + fp32 acc), after the
split write:

| Check | row_split SWA / global | all_split SWA / global |
|---|---|---|
| probes (12 pairs): owner outputs 0 in V dims [:256] and 1 in [256:] (sees k_p, not k_{p+1}), partner 1.5 and 0.5 (sees both) | all ok / all ok | all ok / all ok |
| random data, PCC vs fp64 on the expected cache: overall / worst user | 0.99986 / 0.99979 ; 0.99990 / 0.99975 | 0.99984 / 0.99977 ; 0.99992 / 0.99985 |
| lane relocation (every user to another lane on another DP row, same query / position / page-table row) | - | **bitwise, 28 / 28 users** (SWA and global) |

The design's per-user bar (0.9999) is below the kernel's own floor (G1: worst SWA user 0.99978 with fp32 acc, without
partners); the gate uses overall >= 0.9998 and worst user >= 0.9995, and the relocation result shows partner rows compute
exactly what an ordinary row computes.

### 12.7 G13a — KV-R at op level (`test_g13_kv_replicated_decode.py`)

**Variants** (per layer, then FlashMLA on the per-row `cur_pos [8]` / `page_table [8, W]`): `row` (draft 1), `row_split`
(spec, no APC), `all` (APC, one 32-lane call), `all_split` (production: two 32-lane calls), `all_split_sag` (sharded
all-gather), `all_split_dag` (DRAM all-gather), `deferred` (§3.12.3: `row_split` per layer, then one all-gather of the
54 staged latents and 54 x 2 remote updates).

**Correctness** (3 layer caches `[520,1,64,576]` bfp8; ordinary step and verify step with owners at n and partners at
n + 1, cross-row partners for the KV-R variants):

| Check | Result |
|---|---|
| caches vs the expected writes (layer 0 on all 32 chips, the others on one chip per row; a diagnostic re-run checked every chip of every layer for `all_split` and `all_split_dag`) | **bit-exact** for every variant and step (KV-R: all 32 lanes on every chip; row modes: each row its own) |
| trace replay vs eager (caches and FlashMLA outputs) | **bitwise** for every variant |
| FlashMLA vs fp64 golden, worst user | 0.99977-0.99980 (kernel floor) |
| FlashMLA of `all_split_sag` / `all_split_dag` vs `all_split` (active lanes; idle lanes' rows are never written) | bitwise |
| negative control: `row_split` with cross-row partners | detected: partner PCC min 0.404, median 0.923 (stale anchor on the partner's row) |

**Cost** (54-layer trace: per layer the KV write, FlashMLA (14 global + 40 SWA), the `wo` AR(tp) `[1,1,8,4096]`;
caches `[4129,1,64,576]`, W = 512; replay + sync min of 9; the 235 us sync floor cancels in the deltas):

| Variant | step ctx 1K (us) | step ctx 8K (us) | delta vs `row` 1K / 8K (ms) | per layer (us) | trace bytes |
|---|---|---|---|---|---|
| `row` (draft 1) | 4308 | 5771 | 0 / 0 | 0 | 3.87 MB |
| `row_split` | 4497 | 5964 | 0.19 / 0.19 | 3.5 | 3.93 MB |
| `all` | 5865 | 7345 | 1.56 / 1.57 | 28.8-29.1 | 4.19 MB |
| **`all_split`** | 6176 | 7656 | **1.87 / 1.89** | **34.6-34.9** | 4.33 MB |
| `all_split_sag` | 6212 | 7674 | 1.90 / 1.90 | 35.2 | 4.19 MB |
| `all_split_dag` | 6378 | 7851 | 2.07 / 2.08 | 38.3-38.5 | 4.33 MB |
| `deferred` | 5663 | 7128 | 1.35 / 1.36 | 25.1 | 4.72 MB |

Verify steps (call B active for 12 partners) cost the same as ordinary ones (8K: `all_split` 7475 us, `row_split` 5854 us).
Gate: `all_split` +1.89 ms <= 2.0 ms (**pass**, 0.11 ms margin); the §3.12.3 target (<= 1.0 ms) is not reached by
the deferred variant either (1.36 ms: its 54 x (slice, transpose, 2 updates) tail costs ~25 us per layer). The design's
estimate (30-35 us per layer, 1.6-1.9 ms per step) holds at its upper end; that is ~2.2 % of the 83.8 ms decode step.

**Serving order / L1** (caches `[4129,...]`): prefill 8K (sp0 global SDPA 256/256, sp0 SWA SDPA, fill, sp1 latent SDPA at
128/128 and 64/64 in both compute roles, incl. G9's fp32-acc legacy kernel, sp1 square SWA) -> `all_split` decode (54 layers) eager compile, capture, 3 replays -> the prefills
again -> 3 replays -> prefills: **no static-CB clash, prefill outputs bitwise equal to the first run, 0 programs after
the capture**, main-L1 allocator bytes unchanged by the KV-R programs (the all-gather semaphores sit in L1_SMALL), decode
trace 4.1 MiB for the proxy.

### 12.8 Open issues and requests

1. **G9 role and configs (WP1 / WP2):** the sp1 global op needs `sdpa_prefill_fp32` (fp32 dest acc), not
   `sdpa_prefill`. Requests to WP1 (`tt/model_config.py`, `tt/prefill_plan.py`, `tt/generator_api.py`):
   `SP1_GLOBAL_CHUNKS = ((128, (128, 128)), (1024, (64, 64)), (32768, (128, 128)))` (§12.2), the docstring of
   `resumed_prefill_pc("global", C)` pairing it with the `sdpa_prefill_fp32` role, `prefill_resume_alignment` then = 128
   (its lcm over all buckets; a per-bucket A is an optional refinement), `generator_api.DEFAULT_PREFILL_ALIGNMENT` = 128,
   and the per-bucket sp1 cost model above instead of the single row-key constant. Request to WP2 (`tt/attention.py`):
   use that role (and `resumed_prefill_pc`) in the sp1 global path and assert `start % q == 0 and start % k == 0`.
2. **vLLM budget (WP6 / WP8):** 8064 / threshold 8064 (not 8128). TIS `LLM_YAML` and `check_scheduler_config` should use
   these values.
3. **KV-R margin (KVW / WP5):** 1.89 ms of the 2.0 ms budget at op level. The full-model G13b will add whatever L1 / CCL
   interplay the proxy lacks; if it lands above 2 ms, the measured cheaper path is deferred KV-R (1.36 ms; partners on
   the owner's row). Not measured [I]: riding the MoE token gather (`ag_dp_rows` of `[8, 4096 + 576]`) would save most
   of the 24 us KV all-gather in the 51 MoE layers, also with partners on the owner's row. The 32-lane update's static
   CBs are ~0.9-1.1 MB per core with a bfp8 cache [I: from the program factory's CB formulas; the output CB alone is
   627 KB]: an L1 buffer kept live above that on the update cores (e.g. mHC's decode L1 intermediates) would clash, and
   only G13b in the real decode can show it.
4. **bf16 KV cache + KV-R (KVW):** split the 32-lane write into two 16-lane calls (G12); upstream improvement: size
   `paged_update_cache`'s output CB per core (`Wt`), not `B x Wt`.
5. **Upstream ttnn notes:** the flexible chunked SDPA start has no device-side alignment check (silent wrong answer);
   the streaming SDPA kernel's bf16 destination is inadequate for 576-wide heads.

---

## 13. P5 / T64 gates: G15a-rest (packed prefill attention), G-S1w (64-row KV write + FlashMLA), G16-lite

**Run:** 2026-10-03 15:27-15:41 UTC, host `bh-glx-exp-a03u07`, tt-metal `motif3-bh-galaxy` at **B0 `277df0e9f2d`** plus
C1a's uncommitted `tt/generator_api.py` / `tt/model_config.py` (md5 `55ad7175...` / `258aa5f2...`).
**Isolation:** other work packages were editing `ccl.py`, `kv_write.py`, `moe.py`, `lm_head.py`, `embedding.py` and
`generator_vllm.py` in the shared tree, so these gates import `models.demos.motif3` from a snapshot:
- `git archive 277df0e9f2d models/demos/motif3` plus the two C1a files;
- copies of the root `conftest.py` / `pytest.ini` as the snapshot's root, and the test file itself inside the snapshot
  (pytest imports a test file's parent packages from the file's own directory);
- cwd = the snapshot, `PYTHONPATH=<snap>:<tt-metal>`, `MOTIF3_GATES_RESULTS_DIR` = this directory's `results/`.

Every test records the imported package path and the md5 of its modules (`provenance/*` records). The first G15 run
(`20261003_152055_g15a_rest.log`) imported the live tree by mistake. Its numbers are identical, because the G15 ops run
no CCL (only `ccl.log_fabric` of a modified file was used), but it is superseded.
**Mesh:** (4, 8), `FABRIC_2D_TORUS_XY` requested, **TORUS_Y committed** (the DP axis is a line), `l1_small_size`
32768, trace region 256 MiB, dispatch COL. `ring_gather="safe"` (`MotifCCL` default; every gather of these gates goes
through `MotifCCL`, F3 rule R1).
**Design:** `docs/p5_t64/P5_T64_DESIGN.md` §6.2 (G15a-rest, G-S1w; the G16-lite pieces of §6.1), §3.4, §4.1-§4.4;
notes `docs/p5_t64/p5.md` §14 and `t64.md` §5.1. Op / module level: random data and torch fp32 / fp64 goldens; G16-lite
uses the real layer-2 / layer-4 / LM-head weights from the TT cache. The T64 KV writers, the two-call FlashMLA and the
M = 64 configs (C1a's builders) are injected into the test process the way the probes did it. K1, D1, A2 and I2 repeat
them with the real modules.

| Item | Location |
|---|---|
| Tests | `test_g15_packed_prefill.py` (3 device tests, 4 host self-checks); `test_g16_wide_verify.py` (4 device tests, 2 host self-checks) |
| Raw results | `results/G15.jsonl`, `results/G16.jsonl` (append-only; the last record per case wins) |
| Host self-checks | `scripts/hostrun.sh -- python -m pytest -p no:cacheprovider -q <file> -k host`: view / transpose model, pk1 tables and tail variants (R-E2), goldens, the T64 layout, B0's KV host model at 16 rows per DP row vs a direct oracle, FlashMLA probe discrimination (`logs/host/20261003_154123_g15_g16_host_final.log`: 6 passed) |
| Run | `scripts/devrun.sh -t 2400 -n <name> -- bash -c "cd <snap> && MOTIF3_GATES_RESULTS_DIR=<this dir>/results PYTHONPATH=<snap>:<tt-metal> python -m pytest <snap>/models/demos/motif3/tests/unit/gates/<file> -c <snap>/pytest.ini --rootdir <snap> -s -p no:cacheprovider -k 'not host'"`. Test time: G15 ~40 s, G-S1w ~60 s, G16-lite ~35 s, plus ~5 s per mesh open |
| Logs (`logs/dev/`) | `20261003_152716_g15a_rest_snap.log` (pk0, pk1 global; its pk1 SWA B = 32 cases hit a harness bug, fixed), `20261003_152958_g15c_pk1_swa.log` (pk1 SWA), `20261003_153653_gs1w.log` (G-S1w a + b), `20261003_153806_g16lite.log` (G16-lite ops + layers), `20261003_154007_g16lite_ops_kvw.log` (G16-lite ops again, with the KV-write costs: these op records are the final ones) |

### 13.1 Verdicts

| Gate | Verdict | Key numbers |
|---|---|---|
| **G15a-rest (a)** pk0 batched SDPA | **PASS** | All 16 cases ((S, B) x {global, swa}): every segment **bitwise** equal to the single-row SDPA (max \|Δ\| 0). Views metadata-only, replicas identical on 32 chips, CB end below the pin, 0 programs on a repeat. PCC vs fp32 0.99970-0.99983 (SWA; global S ≤ 256). Global S ≥ 512 (q/k 256/256): 0.99960-0.99966, the single-row op's own value (bitwise equal; G2 floor 0.99948). There the design's 0.9997 bar measures the op, not packing (§13.6 item 1) |
| **G15a-rest (b)** pk1 global | **PASS** | (2, 1024, 2048), (8, 512, 8192) + 1 dummy, (16, 256, 128), (32, 128, 24576) with a shared prefix + 2 dummies, and a bf16 cache (8, 512, 2048): every real segment bitwise (2/2, 7/7, 16/16, 30/30, 8/8). PCC on cols `[:512]` 0.99989-0.99999. Dummies finite. A trace captured at start 8192 and replayed at 2048 / 4096 / 8192 is bitwise equal to eager, 0 programs |
| **G15a-rest (c)** pk1 SWA square | **PASS** | (32, 128), (8, 512), (2, 1024) x {distinct, shared}. Every real segment is **bitwise** equal to the solo sp1 SWA dataflow (tail gather, latent concat, the batched expansion matmul, Q_cat, square SDPA), and the batched SDPA alone is bitwise equal to single calls. PCC vs fp32 window attention 0.99972-0.99973. Window probes exact at B = 32 / 8 / 2. The program cache stays constant when the tail ids change. A traced replay with rewritten tail bounds is bitwise equal to eager |
| **G15a-rest (d)** cost | info | §13.3. Traced 4 CN transposes per layer: 115 µs at T 2048, 397 µs at T 8192, so 6 / 21 ms per pass (review D6) |
| **G-S1w (a)** KV write at 16 rows / DP row | **PASS** | split (KV-R), natural (KV-R) and row_split orders, bfp8 and bf16 caches: every chip's whole cache bitwise == B0's host model (`KVWriteStep.packed_verify` + `apply_kv_writes_host`, 16 rows per DP row) == a direct oracle. The one-call variants lose updates in every trial. The split writer captured and replayed with rewritten inputs is bitwise |
| **G-S1w (b)** FlashMLA, duplicated page-table rows | **PASS** | A' (2 x B = 8) is bitwise == B = 8 on both kinds at n ≈ 1000 / 4095 / 32000. A (B = 16) is bitwise on SWA; on global, PCC vs B = 8 is 0.99995-0.99997, max \|Δ\| ≤ 0.0078 (documented non-bitwise). PCC vs fp64: overall 0.99980-0.99996, worst user ≥ 0.99977. Pair and edge probes exact (\|Δ\| ≤ 1.4e-13) for B8, A and A' |
| **G16-lite** | **PASS** (every bitwise check) | Split-order gather correct, 42.7 µs at 16 rows x 4096. LM head M = 64: rows bitwise, argmax 64 == host. MoE L2 at M = 64 (C1a configs): rows bitwise, +174 µs. Whole layers: A'' rows bitwise on L2 and L4; T64 / T32 = 1.108x (L2), 1.172x (L4). Step model **r = 1.119 / 1.126 / 1.174** at 1K / 4K / 32K (design: 1.118 / 1.126 / 1.173) |

No named fallback was triggered:
- no per-segment SDPA loop;
- pk1 stays on;
- S_min stays 64;
- T64 is not blocked;
- MoE intermediates stay in L1 (no DRAM fallback).

Review disagreements D6 (the transpose term) and D9 (the B = 32 distinct-tail program count) are measured: §13.3.

### 13.2 Decisions published to the work packages

**WP-A (A1: packed attention; A2: A'').**
- **G15a passes: build the packed paths of design §3.4 exactly as the gate ran them.** No per-segment fallback is
  needed for any kind.
- **pk0:** view `[1, H, T, d] -> [H, B, S, d]` + `ttnn.transpose(0, 1)`, `cfg.sdpa_prefill_pc(kind, seq_len=S)`, role
  `sdpa_prefill`, window 129 only when S ≥ 129. This is the solo `prefill_sdpa_window_and_config(S)` rule, so S 64 / 128
  run unwindowed. Then CN back and the view.
- **pk1 global:** `chunked_scaled_dot_product_attention(qb [B, 10, S, 576], cache, cache, sdpa_pt [B, 640],
  chunk_start_idx_tensor [a])`, `cfg.resumed_prefill_pc("global", S, kv_dtype=cache.dtype)`, role
  `sdpa_prefill_fp32`. A dummy copying segment 0's row (R-E2) is harmless.
- **pk1 SWA**, both tail variants as tested:
  - `distinct`: 2B tensor-args slices + **one** `concat(dim=2)` of the 2B blocks + view `[B, 1, 128, 576]` + typecast.
  - `shared`: the solo 2-block gather + typecast + `repeat([B, 1, 1, 1])`.
  - Then `lat = concat([tail_b, view(kv_row, [B, 1, S, 576])], 2)`, the draft-1 expansion on `[B, ...]` (the
    `[512, 640]` weight unbatched), `Q_cat = concat([qb[:, :, :128], qb], 2)` and the square SDPA with
    `cfg.resumed_prefill_pc("swa", S)`, role `sdpa_prefill`. Keep rows `[128, 128 + S)`.
  - In the tested (B, S), the batched expansion matmul gave per-segment rows bitwise equal to the batch-1 call.
- **Programs per pk1 shape (attention ops only; first call):**
  - `distinct` compiles 14 at B ≤ 8, and 18 at B = 32, where ttnn splits the 64-input concat (review D9).
  - `shared` adds 1 (`repeat`).
  - Neither variant compiles anything when the tail ids or the start change.
  - So warming both variants per pk1 shape (R-E2) covers the attention part.
- **Gotcha (S = 128):** `ttnn.slice(qb, [0, 0, 0, 0], [B, 10, 128, 192])` covers the full extent, so ttnn returns `qb`
  itself (`slice.cpp` no-op branch). Never deallocate `q_pad` while `qb` is still needed (the gate checks buffer
  identity). The solo `_prefill_sp1_swa` frees the same buffer twice at C = 128, which is harmless only because both
  frees follow the concat.
- **CB end:** every new program fits below the L1 pin (`1533824`), at all tested shapes, bfp8 and bf16.
- **A2:** A'' is confirmed at op level (G-S1w b) and on whole layers (G16-lite). On SWA layers use one B = 16 call
  (+0.6-0.9 µs, bitwise). On global layers use two B = 8 calls on `ttnn.slice`s of the 16-row Q, one `[8, W]` page
  table for both halves (review D3 holds) and `concat(dim=1)`.

**WP-K (K1: `DecodeKVWrite(rows=64)`, `flash_groups`, `verify_plan`).**
- **Build the split writer exactly as `SplitWriter` in `test_g16_wide_verify.py`:**
  - untilize (L1);
  - view `[1, 2, 8, 576]`;
  - `MotifCCL.all_gather(dim=2, "dp")` (L1);
  - view `[1, 1, 64, 576]` and tilize (DRAM);
  - slice rows `[0, 32)` and `[32, 64)`, each transposed to the 32-core sharded update layout;
  - call A (`cur_a [32]`) and call B (`cur_b [32]`), one replicated `pt [32, W]`.
- **Correctness:** bitwise on all 32 chips. bf16 caches: 2 x 16 users per kind (R-E8), also bitwise.
- **Cost per layer (traced):**

  | Writer | T32 | T64 | Δ |
  |---|---:|---:|---:|
  | bfp8 `all_split` (T64: split order) | 41.2 µs | 58.8 µs | +17.6 µs, ≈ 0.95 ms per step |
  | bfp8 natural order | — | 66.6 µs | — |
  | bfp8 `row_split` | 11.7 µs | 16.9 µs | +5.2 µs |
  | bf16 `all_split` (T64: split order) | 62.8 µs | 124.2 µs | +61.4 µs |

  For bf16, consider the production chunk pattern (one transpose to DRAM, then per 16-user chunk `slice` +
  `to_memory_config`) in place of a transpose per call (§13.6 item 3).
- **Host model:** `KVWriteStep.wide_verify` = `packed_verify` with `partner_of = {16 r + j: 16 r + 8 + j}` on the 64
  natural-order rows. B0's `check_kv_write_step` / `apply_kv_writes_host` with `lanes_per_row=16` (`lanes_per_call` 32
  for bfp8, 16 for bf16) reproduce the device caches of all three orders bit-exactly. The split order changes only the
  call grouping, not the final cache.
- **The A / B split is mandatory.** The one-call variants lost updates in every trial:
  - natural: 21-22 of 52 writes per trial with 32-user calls (bfp8), 19-23 with 16-user calls (bf16);
  - row_split, 16 users per call: 5-7 of the DP row's 13.
- **`flash_groups()`:** one group at 8 rows per DP row. At 16 rows: one B = 16 group on SWA layers, two B = 8 groups
  (anchors at n, drafts at n + 1) on global layers. The edge probes are exact at `n % 64` ∈ {0, 31, 63} up to
  n = 32063.

**WP-I (I1: P5 generator; I2: T64 generator).**
- **I1:** packed passes are validated at the attention level.
  - Warm both pk1 tail variants per shape (§13.2 WP-A).
  - The program cache stays constant across tail ids and starts. The dataflows are trace-safe (trace == eager), though
    prefill stays eager by design.
  - Cost inputs for the planner, per layer, eager: §13.3.
- **I2:** G16-lite on B0 with safe gathers reproduces the probe within ~1-2 %.
  - Layer costs: L2 +179.6 µs (A'', 1.108x); L4 +297.9 µs (A'', 1.172x).
  - Op deltas:

    | Op | Δ |
    |---|---:|
    | MoE M = 64 | +174.3 µs |
    | LM head GEMM | +6.3 µs |
    | Argmax 64 | +30.0 µs |
    | Split gather | +15.1 µs |
    | KV-R write | +17.6 µs |
    | Global FlashMLA A' at 1K / 4K / 32K | +57.6 / +111.2 / +471.5 µs |

  - The design's step model (`t64_model2.py` method) on these numbers gives T64 / T32 = 1.119 / 1.126 / 1.174 at
    1K / 4K / 32K with A'' (A: 1.112 / 1.115 / 1.165). G16's bar (≤ 1.20) is expected to hold; the kill bar (1.30) is
    far.
  - The M = 64 programs (MoE with L1 intermediates, LM head, argmax, FlashMLA B16, 16-user and 32-user updates) all fit
    below the L1 pin.
  - Only `router_logits="composite"` was run. The exact-fp32 router at M = 64 is D1's (R-E7).

**WP-D (D1) and WP-C (C1b), for information.**
- **D1:** the injected C1a M = 64 configs give bitwise rows and fit L1:
  - router: `router_decode_pc(m_tiles=2)` + the fused-sigmoid variant;
  - experts: `experts_gate_up_pc(m_tiles=2)` (10 x 8) and `experts_down_pc(m_tiles=2)` (8 x 8);
  - LM head: `lm_head_pc("mesh", m_tiles=2)`.

  D1 can adopt them as they are. The split-order gather costs the same as `ag_dp_rows` at 16 rows (42.7 / 27.2 µs at
  W 4096 / 576) and is correct for both widths.
- **C1b cost model:**
  - The CN-transpose term scales with T: 6 ms per pass at T 2048, 21 ms at 8192 (not the 85 ms of D6).
  - The pk1 SWA attention dataflow costs 5.5 ms per layer eager at B = 32 with distinct tails and 1.55 ms with shared
    ones, ≈ 0.21 s vs 0.06 s per pass over 39 SWA layers. The 2B-slice tail gather accounts for the 3.9 ms difference
    (≈ 0.15 s per pass; the design assumed 0.3-0.4 s).
  - pk1 global costs 36.7 ms per layer at B = 32, a = 24576 (≈ 0.51 s per pass) and 19.0 ms at B = 8, a = 8192.

### 13.3 G15a-rest details (`test_g15_packed_prefill.py`)

**(a) pk0.** Random bf16 q / k / v (scales folded, V zero-padded 128 -> 192). The single-row reference is the
`[1, H, S, d]` SDPA with the same program / compute config. Eager wall times include the 3 input + 1 output transposes
(host-dispatch bound at this size).

| kind | S | B | T | q/k | window | segments bitwise | PCC vs fp32 (worst) | packed ms | B singles ms | 4 transposes ms | programs (1st call) |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|
| global | 1024 | 2 | 2048 | 256/256 | - | 2/2 | 0.99960 | 0.60 | 0.56 | 0.34 | 6 |
| swa | 1024 | 2 | 2048 | 128/128 | 129 | 2/2 | 0.99974 | 0.44 | 0.22 | 0.32 | 1 |
| global | 1024 | 4 | 4096 | 256/256 | - | 4/4 | 0.99961 | 0.56 | 1.08 | 0.34 | 4 |
| swa | 1024 | 4 | 4096 | 128/128 | 129 | 4/4 | 0.99975 | 0.48 | 0.40 | 0.32 | 1 |
| global | 1024 | 8 | 8192 | 256/256 | - | 8/8 | 0.99960 | 1.10 | 2.13 | 0.42 | 4 |
| swa | 1024 | 8 | 8192 | 128/128 | 129 | 8/8 | 0.99975 | 0.86 | 0.73 | 0.42 | 1 |
| global | 512 | 2 | 1024 | 256/256 | - | 2/2 | 0.99966 | 0.43 | 0.37 | 0.33 | 4 |
| swa | 512 | 2 | 1024 | 128/128 | 129 | 2/2 | 0.99977 | 0.43 | 0.22 | 0.32 | 1 |
| global | 512 | 16 | 8192 | 256/256 | - | 16/16 | 0.99966 | 0.87 | 2.71 | 0.42 | 4 |
| swa | 512 | 16 | 8192 | 128/128 | 129 | 16/16 | 0.99977 | 0.84 | 1.36 | 0.42 | 1 |
| global | 256 | 32 | 8192 | 256/256 | - | 32/32 | 0.99970 | 0.73 | 2.57 | 0.42 | 4 |
| swa | 256 | 32 | 8192 | 128/128 | 129 | 32/32 | 0.99979 | 0.79 | 2.58 | 0.42 | 1 |
| global | 128 | 2 | 256 | 128/128 | - | 2/2 | 0.99978 | 0.66 | 0.32 | 0.52 | 4 |
| swa | 128 | 2 | 256 | 128/128 | - | 2/2 | 0.99980 | 0.92 | 0.43 | 0.73 | 0 |
| global | 64 | 2 | 128 | 64/64 | - | 2/2 | 0.99982 | 0.91 | 0.43 | 0.72 | 4 |
| swa | 64 | 2 | 128 | 64/64 | - | 2/2 | 0.99983 | 0.90 | 0.43 | 0.72 | 0 |

**(d) the transpose term, traced** (q, k, v in and o out of one attention layer; every program compiled before the
first capture):

| T | traced per layer | per pass (53 layers) |
|---:|---:|---:|
| 2048 (S 256, B 8) | 115.4 µs | 6.1 ms |
| 8192 (S 1024, B 8) | 397.3 µs | 21.1 ms |

**(b) pk1 global.** Real-size bfp8 cache `[4129, 1, 64, 576]` (block 0 zero, 1588 random blocks) and a bf16 cache
variant. Distinct prefixes unless marked. Dummies copy segment 0's SDPA row (R-E2). The reference is the single-row
chunked SDPA (`[1, 10, S, 576]`, its own `[1, 640]` row, the same start tensor). Eager ms per call.

| cache | B | S | a | prefix | dummies | q/k | real segments bitwise | PCC cols :512 (worst) | packed ms | singles ms | programs |
|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|---:|
| bfp8 | 2 | 1024 | 2048 | distinct | 0 | 64/64 | 2/2 | 0.999973 | 3.73 | 3.03 | 5 |
| bfp8 | 8 | 512 | 8192 | distinct | 1 | 64/64 | 7/7 | 0.999936 | 18.98 | 31.34 | 3 |
| bfp8 | 16 | 256 | 128 | distinct | 0 | 64/64 | 16/16 | 0.999992 | 1.34 | 3.04 | 3 |
| bfp8 | 32 | 128 | 24576 | shared | 2 | 128/128 | 30/30 | 0.999894 | 36.71 | 324.16 | 3 |
| bf16 | 8 | 512 | 2048 | distinct | 0 | 64/64 | 8/8 | 0.999975 | 9.32 | 10.83 | 1 |

Trace check: the (8, 512) case captured at 8192 and replayed with the start tensor rewritten to 2048 / 4096 / 8192 is
bitwise equal to eager at each start, with 0 programs during the capture and the replays.

**(c) pk1 SWA.** Start 2048, tails = the cache blocks of positions `[a - 128, a)`, random latent rows and a random
`[512, 640]` expansion (per group `[k_nope | v | 0]`, bf16). The golden is fp32 window attention at absolute positions
over the expanded `[tail | chunk]` keys. Eager ms per SWA layer. The program column gives the first call / the call
after a tail-id rewrite (in run order: each distinct case first, then its shared case).

| tails | B | S | dummies | tail blocks | bitwise vs solo dataflow | SDPA alone bitwise | PCC vs window fp32 (worst) | programs | packed ms | B solo ms |
|---|---:|---:|---:|---:|---|---|---:|---|---:|---:|
| distinct | 32 | 128 | 2 | 64 | 30/30 | 30/30 | 0.999727 | 18 / 0 | 5.47 | 32.88 |
| shared | 32 | 128 | 0 | 2 | 32/32 | 32/32 | 0.999716 | 1 / 0 | 1.55 | 32.45 |
| distinct | 8 | 512 | 0 | 16 | 8/8 | 8/8 | 0.999722 | 14 / 0 | 2.29 | 9.03 |
| shared | 8 | 512 | 0 | 2 | 8/8 | 8/8 | 0.999727 | 1 / 0 | 1.55 | 9.11 |
| distinct | 2 | 1024 | 0 | 4 | 2/2 | 2/2 | 0.999725 | 14 / 0 | 1.55 | 2.38 |
| shared | 2 | 1024 | 0 | 2 | 2/2 | 2/2 | 0.999726 | 1 / 0 | 1.51 | 2.38 |

**Probes** (G10's square-layout probes, one independent set per segment, rows {0, 1, 63, 64, 127, S - 1}): every
segment exact at B = 32 / 8 / 2. **Traces:** all six cases, captured at tail set 1 and replayed at sets 2 and 1, are
bitwise equal to eager, with 0 programs.

### 13.4 G-S1w details (`test_g16_wide_verify.py::test_gs1w_*`)

**(a) KV write.**
- Cache: `[264, 1, 64, 576]` (G12 geometry), compared whole on all 32 chips. Rows are pre-quantized to the cache dtype.
- Layout per DP row: local lane 7 idle, lane 6 an anchor without a draft, lanes 0-5 anchor + draft. Anchors cycle
  through `p % 64` ∈ {0, 30, 31, 62, 63} (15 same-tile anchor / draft pairs).
- 52 writes in all: 28 anchors + 24 drafts.

| order | cache | users / call | calls / layer | mismatching chips | host model == direct oracle | CB |
|---|---|---:|---:|---|---|---|
| split (KV-R) | bfp8 | 32 | 2 | none | yes | fits |
| natural (KV-R) | bfp8 | 32 | 4 | none | yes | fits |
| row_split | bfp8 | 16 | 2 | none | yes | fits |
| split (KV-R) | bf16 | 16 | 4 | none | yes | fits |
| natural (KV-R) | bf16 | 16 | 8 | none | yes | fits |
| row_split | bf16 | 16 | 2 | none | yes | fits |

One-call variants (3 trials each; lost anchor / draft updates per trial, chip 0):

| variant | bfp8 | bf16 |
|---|---|---|
| natural, one call per chunk | 10/12, 10/12, 9/12 | 8/15, 9/13, 5/14 |
| row_split, one call per DP row | 1/4, 3/4, 3/4 | 3/4, 3/4, 2/4 |

Trace check: the split writer was captured, the base cache restored, the step-2 inputs (positions, page tables, latent)
written, and the trace replayed. All 32 chips are bitwise equal to the host model, with 0 programs.

**(b) FlashMLA.**
- 8 users per call on a real-size bfp8 cache, `W = 512`. Positions: `[base, b64, b64 + 31, b64 + 63, b64 - 64, b64 - 33,
  b64 - 1, base + 1]`, so `n % 64` covers {0, 31, 63}.
- Drafts at n + 1 carry their owner's page-table row. Scale folded into q (op scale 1.0).
- G1 config: `cfg.flash_mla_decode_pc()`, role `sdpa_decode`.

| kind | n ≈ | A' == B8 | A == B8 | A vs B8 PCC / max \|Δ\| | PCC vs fp64 overall / worst user | probes (pair, edges) |
|---|---:|---|---|---|---|---|
| swa | 1000 | bitwise | bitwise | 1.0 / 0 | 0.99981 / 0.99979 | exact / exact |
| global | 1000 | bitwise | no | 0.99997 / 0.0078 | 0.99986 / 0.99983 | exact / exact |
| swa | 4095 | bitwise | bitwise | 1.0 / 0 | 0.99981 / 0.99979 | exact / exact |
| global | 4095 | bitwise | no | 0.99995 / 0.0039 | 0.99989 / 0.99987 | exact / exact |
| swa | 32000 | bitwise | bitwise | 1.0 / 0 | 0.99980 / 0.99977 | exact / exact |
| global | 32000 | bitwise | no | 0.99996 / 0.0029 | 0.99996 / 0.99992 | exact / exact |

What each probe checks:
- **Pair probe (G12):** the anchor outputs V(n) only; the draft averages V(n) and V(n + 1).
- **Edge probe, SWA:** the anchor attends key n - 128 and masks n - 129 and n + 1. The draft attends n - 127 and masks
  n - 128 and n + 2.
- **Edge probe, global:** the anchor masks n + 1; the draft masks n + 2.

Worst deviation over all probes: 1.4e-13.

Traced per-call cost (µs):

| kind | n ≈ | B8 | A (B16) | A' (2 x B8 + slices + concat) |
|---|---:|---:|---:|---:|
| swa | 1000 | 29.9 | 30.8 (+0.9) | 68.7 |
| swa | 4095 | 30.6 | 30.6 (0) | 68.9 |
| swa | 32000 | 30.0 | 30.6 (+0.6) | 68.9 |
| global | 1000 | 50.3 | 71.7 (+21.4) | 107.9 (+57.6) |
| global | 4095 | 103.1 | 148.3 (+45.2) | 214.3 (+111.2) |
| global | 32000 | 463.8 | 884.0 (+420.2) | 935.3 (+471.5) |

### 13.5 G16-lite details (`test_g16_wide_verify.py::test_g16lite_*`)

Traced per call (slope method). M = 64 runs with the C1a configs injected into the module instances.

| Op | T32 | T64 | Δ | Bitwise |
|---|---:|---:|---:|---|
| `ag_dp_rows` W 4096: 8 -> 16 rows / split order | 27.6 | 42.7 / 42.7 | +15.1 | order checked vs host, replicas 32 |
| `ag_dp_rows` W 576: 8 -> 16 rows / split order | 24.3 | 27.3 / 27.2 | +2.9 | order checked |
| LM head GEMM `[M, 4096] @ [4096, 6880]` | 143.8 | 150.1 | +6.3 | rows 0..31 on every chip |
| argmax 32 -> 64 rows | 219.0 | 249.0 | +30.0 | == host argmax; rows 0..31 == argmax 32 |
| MoE L2 module `forward_decode`, 8 -> 16 rows per DP row | 1040.3 | 1214.6 | +174.3 | rows 0..7 of every DP row, every chip |
| KV write bfp8: `all_split` -> split / natural; `row_split` | 41.2; 11.7 | 58.8 / 66.6; 16.9 | +17.6; +5.2 | (G-S1w a) |
| KV write bf16: `all_split` -> split / natural; `row_split` | 62.8; 12.1 | 124.2 / 138.9; 18.0 | +61.4; +5.9 | (G-S1w a) |

**Whole decoder layers at 4K context** (zero cache; T32 = the production `DecodeKVWrite` `all_split` at 8 rows per DP
row; T64 = 16 rows, the split writer, MoE at M = 64). Bitwise is checked at 1K context on a random cache: the T64 anchor
rows vs the T32 rows, and the draft rows vs a T32 step at n + 1 that sees the anchors' latents.

| layer | T32 | T64 A (B16) | T64 A' (2 x B8) | T64 A'' | A'' Δ / ratio | rows bitwise |
|---|---:|---:|---:|---:|---|---|
| L2 (SWA + MoE) | 1655.8 | 1839.7 | 1869.0 | 1835.4 (= A) | +179.6 / 1.108 | A, A', A'': yes / yes |
| L4 (global + MoE) | 1729.0 | 1955.7 | 2026.0 | 2026.9 (= A') | +297.9 / 1.172 | A: no (max \|Δ\| 0.0156); A', A'': yes / yes |

**Step model.** The design's `t64_model2.py` method, re-run on these records:
- 38 L2-like and 13 L4-like layer deltas, the global FlashMLA part moved to each context;
- layers 0 / 1 and the MTP layer: their attention part only;
- the head delta twice.

| Context | T32-spec | T64 A'' | r | T64 A | r |
|---|---:|---:|---:|---:|---:|
| 1K | 87.09 ms | 97.43 ms | 1.119 | 96.87 ms | 1.112 |
| 4K | 87.83 ms | 98.92 ms | 1.126 | 97.94 ms | 1.115 |
| 32K | 92.88 ms | 109.02 ms | 1.174 | 108.24 ms | 1.165 |

The T32 base, 87.09 ms at 1K, is FRV's, measured with HEAD's gathers. B0's `safe` gathers add 0.26-0.45 ms to every
step, T32 and T64 alike [DET §6]. The real number is G16's (the full traced step).

### 13.6 Open issues and requests

1. **pk0 global PCC bar (lead).** At S ≥ 512 the global-layer SDPA (q/k 256/256, `sdpa_prefill` role) gives PCC
   0.99960-0.99966 vs fp32, for the single-row op as well. The packed call is bitwise equal to it, so the gate passes
   on bitwise equality with a 0.9994 sanity floor and records `design_bar_0p9997_met` per case. Raising the op's own
   accuracy is the existing `sdpa_prefill_fp32_acc="auto"` opt-in (window-free calls). It is independent of P5 and
   would change the solo numerics too.
2. **WP-A:** the `ttnn.slice` full-extent no-op at S = 128 (§13.2).
3. **WP-K:** with a bf16 KV cache, the T64 split writer as injected (one transpose per 16-user call) costs 124 µs per
   layer (+61 µs, ≈ +3.3 ms per step). Measure the production chunk pattern for bf16. bfp8, the production dtype, is
   +17.6 µs.
4. **Re-runs after the WPs land.** These gates import B0 + C1a from a snapshot, and the T64 writers / M = 64 configs are
   injected. K1 (`DecodeKVWrite(rows=64)` + G-S1w b with the real writer), D1 (module tests) and A2 / I2 (G16) repeat
   them with the real modules. Point the snapshot at the new tree (or run in place once the tree is quiet).
5. **CB-end method.** A one-page L1 tensor sits at `1533824`, the top of main L1, just below the L1_SMALL CCL
   semaphores. tt-metal re-validates static CBs against the lowest L1 buffer on every enqueue, so a program whose CBs
   reach the pin raises with its CB end. Nothing reached it.
6. **Fabric:** committed TORUS_Y, so the DP axis is a line and its gathers run on a line, as in every gate since
   2026-10-02.

### 13.7 Model-level gates of P5 / T64

**Runs:** 2026-10-03 20:37 to 2026-10-04 01:36 UTC, host `bh-glx-exp-a03u07`, mesh (4, 8). Every log shows
`FABRIC_2D_TORUS_XY` requested and **TORUS_Y committed** ("DEGRADED (2D torus requested, not committed ...)"): the DP
axis is a line, so G16's ratios and every wall time below are this host's numbers. Restoring TORUS_XY needs a Galaxy
reset with sudo (`docs/P5_T64_REVIEW.md` I-5: re-measure G16 after it). 53 layers + the MTP layer from the serving TT
cache, `ring_gather="safe"`, router `composite` unless marked. Bars: `docs/p5_t64/P5_T64_DESIGN.md` §6.2. Every number
below is read from the log in its row; this section was written from the logs, host-only, after the commit.

**Provenance.** The commit is tt-metal B1 `d3597ae4977`. The runs below differ from it in at most these five
`tt/*.py` files, whose committed md5s (first 8 hex) are attention `2237f1da`, generator `79378147`, model `bc9302fd`,
rope `5c570c37` and model_config `26ff510c`. Three trees ran the gates:
- **P5 snapshot** (`<scratchpad>/p5fix/dev2`, its `MD5SUMS` printed at the top of each log): attention `d525aea1`,
  rope `e75e0874`, model `e65ce617`, generator `78cd489d`, model_config `258aa5f2`; the 9 other listed `tt/` files
  match the commit. An AST comparison of the snapshot with the commit (docstrings ignored; the review's §0.2, repeated
  for this record) finds only decode-side changes: `forward_decode`, `active_mask_host` and the new decode helpers in
  attention, `decode_wide` in model, the decode index helpers in rope, the generator's decode / staging / replay paths
  and T64 refusals, and in model_config only `validate` and `ROUTER_EXACT_FP32_DECODE_ROWS` (the I-2 fix). No prefill
  function changed (`prefill_plan.py` is `759187a0` in both).
- **T64 tree** of 01:12: every `tt/*.py` matches the commit except model_config (`258aa5f2`: the tree before the I-2
  fix, which changes only the exact-router row list `(32,)` -> `(32, 64)`, its refusal message and comments).
- **The I-2 overlay** of 01:25 (`logs/dev/20261004_011507_t64fix_exact_overlay.log`): all 22 `tt/*.py` match the
  commit.

| Gate (test) | Verdict | Numbers that decide it | Log (`logs/dev/`) | md5s against the commit |
|---|---|---|---|---|
| **G15b** packed attention, L0 global + L1 SWA, real weights (`test_attention_resumed.py::test_g15b_packed_attention`) | **PASS** | 16 cases, 8 per layer: pk0 (S, B) = (64, 32), (128, 16), (512, 4), (1024, 2); pk1 S 128 B 32 a 2048 `shared`, S 128 B 8 a 256 `distinct`, S 128 B 4 a 256 `distinct` with a bf16 cache, S 512 B 4 a 2048 `distinct`. Every real segment **bitwise** equal to the per-row path at the planner's bucket (32/32, 13/13, 4/4, 2/2, 32/32, 7/7, 3/3, 3/3). Against a solo run at bucket S also bitwise, except pk0 S 64 (PCC ≥ 0.999993: the planner runs a short row at bucket 128). Written cache blocks bitwise equal to the per-row fills, every other block untouched, a repeat bitwise, replicas identical, finite. 0 programs after the warm-up, and the warm-up writes nothing. R-E2: programs per call `shared`, `distinct`, `shared`, `distinct` = 14, 2, 0, 0. PCC vs the fp32 reference 0.99960-0.99999. 1 passed (38 s) | `20261003_212439_p5fix_g15b.log` | P5 snapshot. Test file `bfcc4f5e` = the commit's |
| **CP-P (i)-(vii)** packed vs per-row, 53 layers + MTP (`test_resumed_prefill.py::test_cp_p_packed_vs_per_row`, `::test_cp_p_burst_ttft_report`) | **PASS** on the floor-relative bars (the lead signed them off on 2026-10-04, `docs/P5_T64_REVIEW.md` §8, I-1). The design's bars (last-token PCC ≥ 0.999, cache rows ≥ 0.99999) are **not met** | Per case in the table below. In all 8 cases: the KV rows of L0, L1 and L2 bitwise per request; the packed call repeated bitwise (logits and cache rows); blocks outside the fill sets untouched; program cache 1430 -> 1430; argmax flips at margin ≥ 0.5: 0 except one in (iii) (the floor has the same flip) and one in (v) (taken by the count slack); last-token PCC medians 0.996471-0.998513 against the per-row path, floor medians 0.996085-0.998022; MTP acceptance within 0.005 of per-row. Wall time 1.62 s (bar 2.0), 3.01 s (3.3), 4.63 s (5.0). Burst report: 32 × ~34 tokens 22.95 -> 1.62 s, 32 × ~100 22.84 -> 3.01 s, 32 × ~300 22.86 -> 11.72 s, 32 × ~600 32.50 -> 23.37 s, 32 behind a shared 2K 24.69 -> 4.63 s; 16 decoding lanes with a packed 16 × ~34 burst between two steps: decode gap 1.16 s (bar 2.5; per row the call takes 11.43 s). The whole full-depth run (CP-L, CP-H, CP-C, CP-X, CP9 and the TTFT report included): 19 passed, 1 skipped (`test_cp_l_truncated_vs_reference`, needs `MOTIF3_CP_LAYERS=4`), 33 min | `20261003_203707_p5fix_resumed_full_l53.log` | P5 snapshot. Test file `18c50125` vs the commit's `81be6aa2`: the later edits are docstrings, comments and the `MULTI_TRACE_NOTE` text (the I-1 sign-off, the F3 history); no assertion changed |
| **CP9-P** 10-min packed soak with one T32-spec trace (`test_pt_soak.py::test_cp9p_packed_soak`) | **PASS** | Sweep: all 56 packed shapes, one call each (165.5 s). Then 104 rounds in 10 min (as logged): 178 prefill calls (56 of them the sweep's), 1,963 rows; 210 solo, 139 pk0, 72 pk1 `shared` and 34 pk1 `distinct` passes; 206 odd-block hit rows inside packed passes; decode 77 ordinary, 96 device-sampled, 16 verify and 136 overflow steps. Program cache 1452 from the capture to the end; solo fallbacks 0; the probe call bitwise identical at every repeat (6 of 6). The end check (decode replay == eager, bitwise) is asserted without a log line; the test passed. 1 passed (14 min) | `20261003_211026_p5fix_cp9p_sweep_soak.log` | P5 snapshot. Test file `c1282848` vs `5d7e226b` (G-X added later; `Soak.check` and `boot` refactored for it) |
| **G16** T64 / T32-spec device step (`test_spec_decode_device.py::test_t64_g16_step_cost`) | **PASS** (bar ≤ 1.20 at 1K-8K, kill > 1.30) | Option A″, traced. `all_split`: T32 87.43 -> T64 97.81 ms at 1K (**1.119**), 88.85 -> 100.72 ms at 8K (**1.134**), 93.10 -> 109.29 ms at 32K (1.174). `row_split`: 1.115 / 1.130 / 1.171. T64 trace region 6.26 MiB per bank (`all_split`), 6.17 (`row_split`) (bar 8.0). Prefills after a T64 replay (sp0 8192, sp1 8192, pk0 (8192, 512)) sane, program cache 1473 -> 1473. A″ costs 0.50 / 0.72 / 0.67 ms per step over option A at 1K / 8K / 32K | `20261004_011204_t64fix_spec_t64.log` | T64 tree (model_config `258aa5f2`). Test file `a147c9c9`, edited at 01:36 to `63ba00aa` (the commit's) |
| **G-S5w (i)-(v)** lossless T64 (`::test_t64_gs5w_lossless`, `::test_t64_release_readback`) | **PASS** | (i) `auto` vs non-speculative decode, 32 lanes, 256 tokens: **token-exact**. Thinking on: 150 verify steps, 116 on T64; acceptance 0.851; 415.7 vs 245.2 tok/s (1.70×). Thinking off: 148 verify steps, 52 on T64; 0.862; 1.75×. (ii) `wide` (T64 alone) drafting vs not: token-exact on both workloads. (iii) A token as a T64 draft row vs as the T32 and T64 anchor row: final hidden state (8 chips) and logits rows (32 chips) bitwise. (iv) Rollback: 33 T64 steps with 469 wrong drafts, all rejected, tokens exact, under `all_split` and `row_split`; afterwards the L0, L1, L4, L52 and MTP cache rows of 32 request pairs bitwise equal to the non-speculative run's, both modes. (v) Accept / reject sequences of 8 requests at c = 8 (T32) and c = 32 (T64): 8/8 identical. **Under `MOTIF3_ROUTER_LOGITS=exact_fp32`** (the I-2 overlay): (i) token-exact (127 of 144 verify steps on T64, 0.860, 1.73×; thinking off 54 of 137, 0.867, 1.88×); (ii)-(v) the same results, with 498 wrong drafts rejected in (iv) | `20261004_011204_t64fix_spec_t64.log`; `exact_fp32`: `20261004_011507_t64fix_exact_overlay.log` | 011204: T64 tree. 011507: every `tt/*.py` equals the commit's; test file `27e9b670` |
| **G-S6w-lite** two traces, 100 solo prefills between replays (`::test_t64_gs6w_lite_two_traces`) | **PASS** | 100 solo prefills (57 cold, 23 hits, 20 chunks) between 128 T32 and 80 T64 decode steps; the probe call bitwise identical; program cache constant at 1473. End: T64 and T32-spec replay == eager bitwise (24 lanes, 24 drafts); the same verify step on T64 == on T32 (packed + overflow pass) bitwise. The I-2 overlay repeated it with the same end checks | `20261004_011204_t64fix_spec_t64.log`; `20261004_011507_t64fix_exact_overlay.log` | as G16 / G-S5w |
| **G-X** combined soak: P5 + `auto` + the device sampler (`test_pt_soak.py::test_gx_combined_soak`) | **PASS** (bar ≥ 300 prefill calls over ≥ 30 min) | Boot to two captures 77.6 s (14 solo + 56 packed shapes warmed). 336 rounds in 33.0 min (as logged; the soak after the sweep ran 30.2 min): 410 prefill calls (56 of them the sweep's), 4,697 rows; 652 solo, 339 pk0, 196 pk1 `shared` and 74 pk1 `distinct` passes; 580 odd-block hit rows inside packed passes; all 56 packed shapes run; solo fallbacks 0. Decode: 311 T64 verify steps, 113 T64 steps with wrong drafts, 47 T32 verify, 96 T32 overflow, 121 ordinary and 195 device-sampled steps (13 lanes re-sampled on the host). The probe call bitwise identical at every repeat (17 of 17); program cache 1565 throughout. End: ordinary trace == eager, T64 trace == eager and T64 == T32 (packed + overflow), all bitwise. **Tracker variant** (`TT_METAL_TRACE_ALLOC_TRACKING=1`, F3N R6; `MOTIF3_GX_MINUTES=3`): the sweep + 34 rounds (5.9 min as logged; 51 T64 verify, 18 T64 reject, 15 T32 overflow steps), the same end checks, passed. **Re-run** (`MOTIF3_GX_MINUTES=5`): 54 rounds (7.7 min as logged), 68 T64 verify and 25 T64 reject steps, the same end checks, passed | `20261003_231404_i2_gx.log`; tracker `20261003_234853_i2_gx_tracker.log`; re-run `20261004_004554_t64rev_gx5.log` | 231404 and 234853 print **no md5s**, so their tree cannot be checked from the logs (the review's "generator `c51f4019`" is the re-run's). 004554: 20 of 22 match; generator `c51f4019` (differs from the commit's only in the R-E7 refusal message and the X3 warning of non-T64 launches, which a `safe` + `composite` run does not reach) and model_config `258aa5f2` (I-2). Test file `5d7e226b` = the commit's |

CP-P per case (the same log; "floor" = each row run per-row at the packed pass's bucket T; teacher-forced top-1 and NLL
are pooled over the case's rows):

| Case | Passes | Last-token PCC vs per-row, median: packed / floor | 32 greedy tokens identical to per-row: packed / floor | Top-1, NLL: packed vs per-row | Wall: packed vs per-row |
|---|---|---|---|---|---|
| (i) 32 short prompts, S 64 | pk0 T 2048, S 64, B 32 | 0.996623 / 0.997189 | 11/32 / 14/32 | 0.4297 vs 0.4246; 3.2497 vs 3.2540 | 1.62 vs 23.06 s |
| (i) 32 prompts of 65-128 tokens, S 128 | pk0 T 4096, S 128, B 32 | 0.996484 / 0.996407 | 20/32 / 21/32 | 0.5437 vs 0.5454; 2.4118 vs 2.4119 | 3.01 vs 23.62 s |
| (ii) 32 rows behind a shared 2K prefix | solo sp0 2048 + pk1 T 4096, S 128, B 32, `shared` | 0.996834 / 0.996085 | 27/32 / 29/32 | 0.7570 vs 0.7570; 1.2668 vs 1.2675 | 4.63 vs 25.78 s |
| (iii) the mixed step | 3 pk0 passes + 2 solo sp0 | 0.996664 / 0.996526 | 12/32 / 14/32 | 0.5728 vs 0.5732; 2.2250 vs 2.2264 | 7.24 vs 26.10 s |
| (iv) 64-token template hits | pk0 T 4096, S 128, B 32 | 0.996471 / 0.996471 | 11/32 / 10/32; one divergence beyond a near tie (margins 1.625 / 0.500), taken by the count slack | 0.3194 vs 0.3130; 4.6353 vs 4.6541 | 3.01 vs 23.29 s |
| (v) hit behind a writer with an internal split | sp0 + sp1 solo, then pk1 T 1024, S 128, B 8, `shared` | 0.997657 / 0.996510 | 8/9 / 7/9 | 0.6956 vs 0.6956; 1.4429 vs 1.4448 | 3.14 vs 8.79 s |
| (vi) odd-block same-step hit (R-E1) | 2 sp0 + 2 sp1 solo, then pk1 T 256, S 128, B 2, `shared` | 0.998513 / 0.996707 | 4/4 / 4/4 | 0.6816 vs 0.6816; 1.4288 vs 1.4289 | 5.09 vs 6.99 s |
| (vii) 8 sessions resumed at one start (R-E2) | pk1 T 1024, S 128, B 8, `distinct` | 0.997900 / 0.998022 | 6/8 / 7/8 | 0.6525 vs 0.6599; 1.8432 vs 1.8567 | 1.07 vs 5.95 s |

**Agreement with the other records.** The gate numbers that `docs/P5_T64_REVIEW.md` (§1-§4, I-1) and
`docs/P5_T64_RESULTS.md` (§3.3) quote match these logs. Two provenance statements need the qualifications above:
1. The review's §0.2 says the T64 gates "ran on the final tree" and that the E2E server logs' 22 md5s "equal the final
   tree's". Both were true when it was written. The I-2 fix then changed `model_config.py`, so the committed tree
   differs from the G16 / G-S5w / G-S6w-lite run (011204) and from all six E2E server logs of `P5_T64_RESULTS.md`
   (`logs/serve/p5_t64/vllm_*.log`, checked for this record) in that file only: `258aa5f2` served, `26ff510c`
   committed. The G-X re-run differs in it and in `generator.py` (item 2). The `exact_fp32` overlay (011507) ran the
   committed file.
2. "G-X ran on generator `c51f4019`" holds for the 5-minute re-run, the only G-X log with md5s.

§13.6 item 4 is closed: K1, D1 and A2 repeated the injected op gates with the real modules (`DecodeKVWrite(rows=64)`
6 passed, MoE / LM head / MTP at 64 rows 4 passed (the MoE under both routers), attention A″ 3 passed;
`logs/dev/20261004_002010_t64rev_modules.log` and `20261003_220232_a2_attn_kvr_final.log`; their `tt/*.py` match the
commit except generator `c51f4019`).
