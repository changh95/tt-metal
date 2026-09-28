# DFlash2 block-diffusion drafter (M4) -- results, 2026-09-28, half B (P150x4, TP=4), tt-metal `qwen38-pd-disagg`

Code: `tt/dflash2_head.py` (config/loader, fp32 host reference, `DFlash2Drafter` device module, `DFlash2ContextProjector`,
`VerifyStepAux`, `PrefillAuxCapture`), tests `tests/test_dflash2_cpu.py` (CPU, vs the official implementation),
`tests/test_dflash2_spec_scratch.py` (device), runner `scripts/dflash2_spec_run.sh`, gate `scripts/chips_free_gate.sh`.
Drafter: `z-lab/Qwen3.8-27B-DFlash2` (Apache-2.0, HF snapshot `50307d4c`), 1.92 B params bf16 (3.85 GB).

## 1. The DFlash2 math (inferred from the references, verified against the official module)

Sources: the z-lab reference implementation `github.com/z-lab/dflash` `dflash/model.py` (`DFlash2DraftModel`,
`GroupedDynamicCausalConv`, `CandidateSelector`, `dflash_generate`), its MLX port `model_mlx.py`, the vLLM PR #52816
(`qwen3_dflash2.py`, `dflash2/speculator.py`, `eagle3_utils.py`), the model card and the DFlash 2 blog. The installed
vLLM 0.13 `qwen3_dflash.py` / `spec_decode/dflash.py` cover DFlash v1 only (no conv, no selector) and resolve the
block causality differently from every DFlash2 reference (see 1.4).

Shapes (config.json): 5 Qwen3 layers, hidden 5120, 32 q heads / 8 kv heads x **head_dim 128** (explicit `head_dim`,
not hidden/heads = 160), intermediate 17408, vocab 248320 (= the target's; the drafter has NO lm_head / embed_tokens
and uses the target's tables), rms eps 1e-6, RoPE theta 1e7 default (full 128-dim rotation, HF rotate-half),
`dflash_config`: block_size 8 (1 anchor + 7 drafts), conv_group_size 16 (320 groups), conv_kernel_size 2,
mask_token_id 248070 (its embedding = the target table's row 248070), selector_rank 256, selector_top_k 16,
target_layer_ids [5, 19, 33, 47, 61]. Tensor names / shapes read from the safetensors header: per layer
`self_attn.{q,k,v,o}_proj` ([4096|1024|1024|5120] x ...), `self_attn.{q,k}_norm` [128], `mlp.{gate,up,down}_proj`,
`input_layernorm`, `post_attention_layernorm`, `attention_conv.base_kernel` [2, 2, 5120],
`attention_conv.kernel_projection.weight` [1280, 5120], `mlp_conv.*` (same shapes); top level `fc.weight` [5120, 25600],
`hidden_norm.weight`, `norm.weight`, `candidate_selector.hidden_projection.weight` [256, 5120],
`candidate_selector.{predecessor,successor}_codebook` [248320, 256] (no `.weight` suffix).

### 1.1 Aux hidden states (target side)

`aux = cat_l hidden_states[l + 1]` for `l in target_layer_ids` where HF `hidden_states[0]` is the embedding output and
`hidden_states[l+1]` the residual stream AFTER decoder layer `l` (z-lab `extract_context_feature`, `offset = 1`).
vLLM: `get_eagle3_aux_layers_from_config` adds 1 to `target_layer_ids` and `Qwen3_5Model` appends
`hidden_states + residual` after layer `layer_idx` when `(layer_idx + 1) in aux_hidden_state_layers` -- the same rows.
On the TT model that is the output of `layer.forward` / `layer.forward_verify` for layers 5, 19, 33, 47, 61 (the
residual stream before the next layer's norm). One 25600-wide bf16 row per token.

### 1.2 Context projection (once per committed token; no drafter layer runs over the context)

    t   = RMSNorm_plain(fc(aux); hidden_norm)                      fc: [25600 -> 5120], no bias
    K_i = RoPE(RMSNorm_plain_per_head(k_proj_i(t); k_norm_i), pos)  V_i = v_proj_i(t)        for drafter layer i

RMSNorm_plain(x; w) = x * rsqrt(mean(x^2) + eps) * w (Qwen3RMSNorm: fp32 statistics, PLAIN weight -- not the target's
zero-centered 1+w). These K/V are what the drafter's attention sees for context tokens (z-lab `k_ctx = k_proj(
target_hidden)` inside every layer; vLLM's fused `precompute_and_store_context_kv` GEMM). A committed token's row is
written at its absolute position; a rejected row's position is overwritten by the next commit (the reference crops
its cache to `start`).

### 1.3 Block forward (one pass per draft step)

Query block per request: `x_0 = embed_target([anchor, mask x 7])` at positions `P .. P+7` where `anchor` = the last
committed token (the verify grid's row 0, `block_output_ids[:, 0] = output_ids[start]`) and P = its position. For each
layer i (`Qwen3DFlashDecoderLayer` with `attention_conv` / `mlp_conv` set):

    h        = RMSNorm_plain(x; input_layernorm_i)
    dyn      = kernel_projection_a(h).view(L, 2 [pre/post], 2 [tap], 320 [group])
    h'       = Conv(h, dyn[:,0], base_a[0])
    q        = RoPE(RMSNorm_per_head(q_proj(h'); q_norm))          k_blk = RoPE(RMSNorm_per_head(k_proj(h'); k_norm))
    v_blk    = v_proj(h')
    a        = o_proj(SDPA(q, [K_i ; k_blk], [V_i ; v_blk], scale 128^-0.5, mask = 1.4))   GQA 4 q heads per kv head
    x        = x + Conv(a, dyn[:,1], base_a[1])
    h2       = RMSNorm_plain(x; post_attention_layernorm_i);  dyn2 = kernel_projection_m(h2).view(L,2,2,320)
    h2'      = Conv(h2, dyn2[:,0], base_m[0])
    m        = down(silu(gate(h2')) * up(h2'))
    x        = x + Conv(m, dyn2[:,1], base_m[1])
    hidden   = RMSNorm_plain(x; norm)                              rows 1..7 = the draft positions

Conv (z-lab `_grouped_dynamic_convolve`, vLLM `_grouped_conv`), two taps, block-local and stateless:

    Conv(x, d, base)_t = (base[0] + expand16(d[t, 0])) * x_t + (base[1] + expand16(d[t, 1])) * x_{t-1},   x_{-1} = 0

`expand16` repeats each of the 320 group coefficients over its 16 channels (`d.view(groups, 1)` broadcast); `base`
[2 taps, 5120] is per channel. The first row of the block (the anchor) has only the tap-0 term (vLLM masks
`position >= tap` inside the block of `1 + num_speculative_tokens` rows). The POST coefficients `dyn[:, 1]` are computed
from the sublayer INPUT `h` (inside `prepare`) and applied to the sublayer OUTPUT (`finish`). Inferred detail: the
1280 = 2 x 2 x 320 projection columns are laid out `[side, tap, group]` (the `.view(..., 2, kernel_size, groups)`).

### 1.4 Attention mask

`Qwen3DFlashAttention`: `is_causal = layer_type == "sliding_attention" if config.is_causal is None else config.is_causal`
and config.json has top-level `"is_causal": false` -> the block attention is **non-causal** (every block row sees the
whole block) in the z-lab module, the MLX port (`is_causal = ... if config.is_causal is None else config.is_causal`) and
the vLLM PR (`_dflash_layer_causal` returns the explicit `is_causal` first). The layer type `sliding_attention` keeps a
`sliding_window` of 2048 over key positions: `query_pos - key_pos < 2048` (and, non-causal, `key_pos - query_pos <
2048`, vacuous for the block). The installed vLLM 0.13 file ignores `is_causal` and would make these layers causal --
not followed. Device: the paged SDPA decode has no window (every context position <= cur_pos is attended); identical
to the reference for contexts shorter than 2048 tokens, documented deviation above.

### 1.5 Head and candidate selector

    logits_t   = lm_head_target(hidden_t)                           t = 1..7 (draft rows), x output_multiplier 1.0, no softcap
    cands_t    = top16(logits_t)      unary_t(b) = logits_t(b)      (raw logits, greedy path; sampling variants ignored)
    H_t        = hidden_projection(hidden_t)                        [256]
    pred_0     = anchor
    score_t(b) = unary_t(b) + < A(pred_{t-1}) * H_t , B(b) >          A = predecessor_codebook, B = successor_codebook
    d_t        = argmax_b score_t(b);  pred_t = d_t

(z-lab `CandidateSelector.select`; vLLM `_score_edges` scores every (predecessor candidate, successor candidate) pair
with the anchor as the position-1 predecessor and `_selector_walk_kernel` follows the argmax greedily -- the same walk.)
The drafts d_1..d_7 are the walked path; with k < 7 drafts the block still runs 8 rows and the first k path tokens are
used (the reference shortens the block only at the end of generation).

### 1.6 Verification of the host reference (tests/test_dflash2_cpu.py, logs/dflash2_cpu1.log)

`DFlash2HostReference` (fp32, straight from the safetensors + the target's embed/lm_head) vs the official
`DFlash2DraftModel.from_pretrained(..., torch_dtype=float32)` + `propose()` on random 37-row contexts (two seeds):
block hidden **rel max|d| 8.5e-6 / 3.9e-6**, selected paths identical, candidate sets identical. The incremental
context state (30 rows + 3 garbage rows at re-written positions + 10 rows) reproduces the from-scratch state
(atol 1e-4, same path). The selector changes the draft vs the plain per-position argmax on 5 of 7 positions of a
random-context probe (expected: the walk conditions on the chosen predecessor).

## 2. Device module (`DFlash2Drafter`, TP=4) -- M4a

Logs: `logs/df2_smoke4.log` (first working E2E), `logs/df2_full1.log` + `.json` (the sweep below), `logs/df2_fill1.log`
(context-fill warm-up check: with the fill compiled for every block-multiple prompt length before the captures, the
(8,7) context write drops from 4841 ms (a post-capture JIT) to 45 ms for 8 users, results unchanged: accept 5.43, x2.18). Half B (`TT_VISIBLE_DEVICES=2,3,4,5`), AICLK pinned 1200 MHz, runner
`scripts/dflash2_spec_run.sh` (3-minute-silence kill), gate `scripts/chips_free_gate.sh` (fuser on the half-B devices).

* Weights: `load_dflash2_state_dict` (bf16 safetensors) -> bfp8 device tensors cached under `<weight cache>/dflash2/`:
  per layer fused `[q heads | k heads | v heads]` column shard [5120, 1536] per device (8 q + 2 kv heads of 128),
  `o_proj` row shard [1024, 5120], MLP gate/up column shards [5120, 4352] + down row shard [4352, 5120], the two
  conv kernel projections replicated [5120, 1280], norms / conv base rows bf16; `fc` column-parallel [25600, 1280] +
  all-gather; `hidden_projection` bf16 [5120, 256]; the group-expansion constant `E` [320, 5120] (0/1). Embedding =
  `model.embd` (the target's table holds row 248070 = the mask embedding), LM head = `model.lm_head_weight`
  (vocab-sharded). Selector codebooks stay on host (2 x 121 MiB bf16). **566 MB (540 MiB) per chip.**
* KV: 5 x [k, v] paged caches `[num_blocks + 1, 2, 64, 128]` bfp8 per device (`allocate_kv`; the main cache's block ids
  + one pad block, registered as pd_transfer KV group "dflash2" through `model.dflash2_drafter`). **2720 B/token/chip
  -> 2.86 GB at 1,052,672 tokens** (vs the target's 16 layers: 5/16 of its KV pool); 43 MiB in the 256-block harness.
  Total drafter footprint next to the target at the served pool: **~3.43 GB/chip of the ~9 GB free.**
* Draft step (`step_forward`, traced per width, R = 8w rows): embedding gather + all-gather -> per layer { fp32-stat
  RMSNorm (7 ops: the plain `ttnn.rms_norm` on a 5120-wide interleaved row clashes with the persistent L1 buffers),
  kernel projection, pre-conv (2 slices, 2 `E` matmuls + base adds, 1 block-shift 0/1 matmul, 2 muls, 1 add),
  fused qkv, q heads (RM round trip) -> q_norm -> RoPE (per-row cos/sin), per-kv-head k slices -> k_norm -> RoPE, v
  slices, 0/1 spread matmuls to the `[1, w, 32, 128]` update shard (row `s*32 + h*8 + j`) -> ONE
  `paged_update_cache(num_tokens=8)` per cache at `P_s`, paged SDPA decode over R virtual users with
  `cur_pos = P_s + 7` (non-causal block; row chunks of <= 110 users), concat heads, o_proj partial ->
  reduce-scatter + all-gather, post-conv, residual add; the same around the MLP (silu fused in gate) }; final norm;
  `hidden_projection` [R, 256]; lm_head [R, 62080] per device -> pad to 65536, two power-of-two halves -> `ttnn.topk`
  k=32 each; readback [TP, R, 64] values + ids and [R, 256] -> host: global top-16 per draft row, codebook walk from the
  anchor (`selector_walk`, shared with the host reference; < 1 ms per step at w=32: 8.3 s over 609 steps incl. readback).
* Context commit (`commit`, traced per verify plan): `project_kv` over `plan.out_aux` (fc + all-gather + hidden_norm +
  ONE fused [5120, 5 x 2 x 256] K/V GEMM per device + per-layer k_norm + RoPE at rows `P_s + j`) -> the same spread +
  `paged_update_cache(num_tokens=T)` per cache at `plan.cur_pos[0]` through `plan.page_table`. All T rows are written;
  rows past `a_s` sit at positions the next block overwrites before its SDPA reads them (cur_pos-bounded), exactly the
  reference's crop-and-rewrite. `write_context(slot, positions, aux)` (prompt / imported aux rows) fills whole blocks
  with `paged_fill_cache` (typecast to bfp8); `import_context_kv` does the same from a payload of projected K/V.
* Deviations from the reference, documented: no sliding window on device (the paged SDPA has none; identical below
  2048 context tokens); bfp8 weights / KV and bf16 activations; k < 7 uses the first k tokens of the 8-row block.

### PCC probe (128-token GSM8K prompt, w=1, k=7, 9 verify steps = 63 draft rows; `DF_PCC`)

Device draft logits (full [7, 248320] readback) vs the fp32 host reference fed the SAME aux rows (the prompt's from the
prefill hook, then the committed rows `0..a_s` of every `plan.out_aux`) and the same anchors/positions:

| | value |
|---|---|
| PCC of the draft logits, mean / min over 63 rows | **0.99930 / 0.99615** |
| per-position argmax agreement | 0.937 (59/63) |
| selected-path (drafted token) agreement | **0.984 (62/63)** |
| committed stream == plain decode | yes |

Per step the 7 rows sit at PCC 0.9981-0.9997 (one row 0.9961 at position 140); the first step's rows (context = the
prompt fill only) are 0.9995+. `logs/df2_smoke4.json` documents the harness bug the probe first showed (PCC 0.66-0.97
from step 2): the reference had been given all T grid rows as context, i.e. the rejected rows the device overwrites.

## 3. Aux hiddens (M4b)

The verify step's `VerifyStep(keep_aux_hidden=True, aux_layers=target_layer_ids)` -> `plan.out_aux` [1,1,R,25600] bf16
replicated DRAM (rows r = s*T + j; the other agent's commit `4dba9dbad0e`: eager == traced bitwise) feeds the traced
commit directly (the commit trace is captured right after its verify trace and reads the output buffer's fixed
address). The prompt's aux rows come from `model.prefill_aux_layers` + `model.prefill_aux_hook(user_ctx, aux_frac,
token_buf, actual_len, bucket, chunk_start)` (commit `fa9107b6909`; `aux_frac` = the fractured residual copies after
layers 5/19/33/47/61 of every prefill path) -- the harness hook gathers them to host per slot (`AuxRowsHook`) and
`write_context` projects + fills them; the served P side runs `aux_hidden.DFlash2ContextPrefillHook`, which calls
`DFlash2ContextProjector.project_device(aux_rep, positions)` (the replicated device rows, no host round trip) and
ships per layer K/V `[n, 8, 128]` bf16 as KV group "dflash2". Semantics check: layer OUTPUTS at 0-based indices
5/19/33/47/61 = HF `hidden_states[l+1]` = vLLM `aux_hidden_state_layers = target_layer_ids + 1` (section 1.1); the PCC
probe above is the end-to-end confirmation (a wrong layer set or offset would not give 0.999 PCC on real prompts).

## 4. End-to-end speculative decode (M4c) -- `logs/df2_full1.json`

14 real prompts (3 base, 8 GSM8K chat-templated, 3 code), >= 64 committed tokens per user, plain decode = the traced
served decode at the same width in the same process (DecodeRef). w=1 configs run 10 prompts each (0-6, 11-13), w=4 the
first 4, w=8 the first 8, w=16/32 all 14 (mod). Drafts = the first k tokens of the 8-row block's selected path.

| config (w,k) | T | R | exact vs plain decode | accept len tok/user/step (base / gsm8k / code) | accept rate | spec tok/s (agg) | plain decode tok/s | speedup | ms/step: verify + commit + draft = total (decode ms/step) |
|---|---|---|---|---|---|---|---|---|---|
| (1,7) | 8 | 8 | **bitwise** (10/10) | **5.14** (4.38 / 5.00 / 6.53) | 0.59 | **110.1** | 39.1 | **x2.82** | 38.9 + 1.0 + 8.4 = 49.0 (25.6) |
| (1,3) | 4 | 4 | **bitwise** (10/10) | 3.37 (3.03 / 3.41 / 3.71) | 0.79 | 81.5 | 39.3 | x2.07 | 32.2 + 0.9 + 8.2 = 41.8 (25.5) |
| (4,7) | 8 | 32 | **bitwise** | 4.77 (4.48 / 5.62 / -) | 0.54 | 360.4 | 153.3 | x2.35 | 42.1 + 1.1 + 9.1 = 52.9 (26.1) |
| (8,7) | 8 | 64 | near-ties only (4, gap <= 0.25) | **5.43** (4.74 / 5.84 / -) | 0.63 | 625.6 | 294.4 | x2.13 | 55.9 + 1.3 + 11.6 = 69.4 (27.2) |
| (8,3) | 4 | 32 | **bitwise** | 3.34 (3.06 / 3.50 / -) | 0.78 | 542.9 | 291.6 | x1.86 | 36.0 + 1.1 + 11.5 = 49.2 (27.4) |
| (16,3) | 4 | 64 | near-ties only (7) | 3.31 (3.01 / 3.32 / 3.78) | 0.77 | 725.3 | 527.1 | x1.38 | 54.5 + 1.3 + 16.4 = 73.0 (30.4) |
| (32,3) | 4 | 128 | near-ties only (15) | 3.35 (3.02 / 3.37 / 3.78) | 0.78 | 957.6 | 908.8 | x1.05 | 83.2 + 1.7 + 25.7 = 111.8 (35.2) |

(1,7) per prompt: base 4.85 / 4.19 / 4.19, gsm8k 5.31 / 6.27 / 3.61 / 5.58, code 5.25 / **8.00** / 6.90 tok/step (the
binary-search completion accepts all 7 drafts every step: 156.6 tok/s). Accept histograms in the json (`accept_hist`;
(8,7): 63 of 144 steps accept all 7).

Every R > 32 divergence is the documented fractured-path near-tie of the plain decode (MTP_SPEC_RESULTS.md (32,1)):
the near-tie probe finds plain-logit gaps of 0.125 (1 bf16 ulp at |logit| 16..32), 0.25 or 0.000 at every flip, the
same (prompt, index) pairs for every twin user and for (8,7) / (16,3) / (32,3) alike (u3@25, u4@7, u7@44 ...): the
stream is draft-independent; the test asserts bitwise equality for R <= 32 and gap <= 0.25 above.

### vs the MTP drafter (MTP_SPEC_RESULTS.md, same harness, same prompts, same process design)

| | MTP (k chained steps of one layer) | DFlash2 (one 8-row block) |
|---|---|---|
| accept len, w=1 k=3 | 3.33 (3.15 / 3.23 / 3.69) | 3.37 (3.03 / 3.41 / 3.71) |
| accept len, w=1 best config | 3.33 at k=3 (k=7 not measured; ~7.3 ms per 3 drafts) | **5.14 at k=7** (6.53 on code) |
| tok/s, w=1 best | 83.7 (x2.14, k=3) | **110.1 (x2.82, k=7)** |
| tok/s, w=8 best | 496 (x1.69, k=2) | **625.6 (x2.13, k=7)**; 542.9 (x1.86, k=3, bitwise) |
| tok/s, w=32 | 1077.9 (x1.18, k=1, T=2) | 957.6 (x1.05, k=3, T=4: verify 83 ms + draft 26 ms) |
| draft cost per verify step | k x (2.4-3.4 ms) + 0.3-0.5 select = 7.3 ms at k=3 (w=1) | 8.2-8.4 ms at ANY k (w=1); 9.0 (w=4), 11.6 (w=8), 16.2 (w=16), 26.2 ms (w=32) + commit 0.8-1.7 |
| drafter state per request | 17th KV layer + hidden row (prefill 5-28 ms eager) | 5 KV layers (2.7 KB/token bfp8), prompt context fill 6-10 ms per 64-token block group (eager) |
| DRAM per chip | 1 layer bfp8/bfp4 + 1/16 KV pool | 540 MiB + 5/16 KV pool (2.86 GB at 1 M tokens) |

The block drafter wins where the verify step is cheap relative to the draft (w <= 8): +23-32 % tokens/s over the best
MTP config at w=1 and w=8 with 7 drafts, at the SAME verify-machinery guarantees. At w=32 the fractured (R = 128)
verify body (83 ms) and the 256-row draft (26 ms) eat the gain; the MTP's T=2 plan (R=64, 53 ms) is faster there.
Next steps for w >= 16: the k=1/2 bands (T=2/3, R <= 96) with the same block drafter, and the draft step's row count
(the 8-row block is computed for all 32 users -- a 16-row MLP/conv path per user would halve the 26 ms).

### Draft-step / commit timings (traced, median of 30, upload + replay + readback + host selector)

| width w | rows R | draft step ms | commit ms (T=8 / T=4) |
|---|---|---|---|
| 1 | 8 | 8.21 (min 8.16) | 0.84 / 0.82 |
| 4 | 32 | 9.00 | 0.98 / - |
| 8 | 64 | 11.56 | 1.20 / 1.01 |
| 16 | 128 | 16.24 | - / 1.22 |
| 32 | 256 | 26.16 | - / 1.67 |

At w=1 the step is weight-streaming bound (~540 MB of bfp8 weights + ~100 ops per layer x 5); the row count adds
~70 us per 8 rows. In-loop averages (`draft_ms_per_step`, includes the host selector and Python) match: 8.2-8.4 (w=1),
9.1 (w=4), 11.5-11.6 (w=8), 16.4 (w=16), 25.7 (w=32).

### DRAM footprint per chip (`dram_report`, measured tensor bytes)

| item | bytes |
|---|---|
| drafter weights (bfp8 shards + replicated conv projections, bf16 norms/E/hp) | 566,149,120 (540 MiB) |
| KV pool at the harness's 256 blocks (5 layers x 2 x [257, 2, 64, 128] bfp8) | 44,738,560 |
| KV bytes per token | 2,720 |
| KV pool at 1,052,672 tokens | **2,863,267,840 (2.67 GiB)** |
| total at the served pool | **~3.43 GB** of the ~9 GB free next to the target |

## 5. What the served integration needs (D side = tt/spec_decoder.py + tt/spec_serving.py, P side = pd_transfer)

* **Drafter knob**: `QWEN36_SPEC_DRAFTER=mtp|dflash2` (today `QWEN36_SPEC_MTP=1` selects the MTP head). `dflash2`
  builds `DFlash2Drafter(model, page_tables, widths=<decode bucket widths>)` after `allocate_kv_caches` and BEFORE
  `pd_transfer.import_warmup` / any capture (its 5 x [k, v] caches, `head.kv_layers`, must be baked into the traced
  importer's cache list like the MTP head's 17th layer), then per verify plan `bind_plan` + `compile_commit` in the
  compile-first phase and `capture_commit` right after each verify capture. The drafter needs `VerifyStep(...,
  keep_aux_hidden=True)` -> `plan.out_aux` (the interface `VerifyStepAux` provides today) instead of `keep_hidden`.
* **Ladder / bands**: the block is 8 rows, so the natural bands are T = 8 (k = 7) at w <= 4 (R = 32, the fused-AR
  exact path), T = 8 at w = 8 (R = 64, fractured path, near-tie-bounded), and T = 4 (k = 3, the first 3 path tokens)
  at w = 16 / 32 (R = 64 / 128) -- the (w, k) grid measured in section 3; the draft cost is ONE traced block step per
  verify step at every k (no k x chained steps), so the ladder should prefer the largest T the verify budget allows.
  `spec_serving`'s per-(bucket, T) plans keep working unchanged: the drafter only needs `draft(w, anchors, positions,
  pad=...)` (padding users = position -1) and `commit(plan, positions)` after every verify step.
* **Per-step loop** (replaces select_hidden + k chained MTP steps): `verify.run` -> host accept -> `head.commit(plan,
  positions_before)` (traced: projector over `plan.out_aux`, one 8-row KV write per cache) -> `head.draft(w, last,
  positions)` (traced block forward + top-k/hp readback + host selector, first k path tokens) -> next verify.
* **Payload (P -> D)**: the drafter's context KV for the prompt. Two options: (a) ship the prompt's aux rows
  (25600 bf16 = 51 KB/token) and let D run `write_context` (the P side then only needs the aux capture inside its
  prefill traces: layers 5/19/33/47/61 outputs -> replicated concat, the other agent's traced hook); (b) ship the
  projected K/V (5 layers x 2 x 8 heads x 128 = 20 KB/token bf16) via `import_context_kv`, or, cheapest, export the
  drafter's KV BLOCKS as 5 extra attention layers of the pd_transfer payload (bfp8, 10.6 KB/token, same block ids as
  the target's caches: `head.kv_layers`) -- the P side then runs the projector after its prefill (fc + hidden_norm +
  the fused K/V GEMM + k_norm + RoPE, a few ms per 2048-token chunk) into the SAME paged layout D reads. (b)-blocks is
  the MTP hand-off pattern (`_attention_layers` = main + drafter layers, layer-count-tolerant import) and needs no
  per-request hidden row: the first draft step on D is `draft(w, [x_N], [N])` directly.
* **Not needed**: the drafter has no per-request host state (no `hidden_in` row, no chained KV of its own past the
  committed positions), so slot admission/eviction only has to keep the block ids; a request whose drafter context
  is missing simply decodes plainly (drafts of zeros are rejected, `pad` users are skipped).
