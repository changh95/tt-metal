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
