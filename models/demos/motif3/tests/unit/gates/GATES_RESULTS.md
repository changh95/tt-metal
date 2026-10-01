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
