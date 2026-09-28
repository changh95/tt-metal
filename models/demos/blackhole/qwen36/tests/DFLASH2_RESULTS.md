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

## 6. Served integration (M4d, 2026-09-28) -- see the plugin's docs/SPECULATIVE.md for the design and the A/B tables

`QWEN36_SPEC_DRAFTER=dflash2` on both halves of the 4+4 P/D stack: `qwen36_vllm._build_drafter_if_enabled` builds the
`DFlash2Drafter` with the KV caches (540 MiB weights + 2736 MiB context KV per chip at the 1,052,672-token D pool;
6.5 GB DRAM free after warm-up), `tt/spec_decoder.py` runs it behind the drafter adapter (`keep_aux_hidden` verify
plans, traced `commit(plan, positions)` per plan captured right after each verify trace, one block draft step per
bucket width, first k of the 7 path tokens; a request whose payload lacked the KV group decodes with no drafts), the
runner reports the imported context per slot (`spec_note_admission` -> `SpecDecoder.note_context`), no catch-up step.
Ladders (`tt/spec_serving.py`, knob `QWEN36_SPEC_ALLOW_FRACTURED`): default `1:8,2:8,4:8,8:8,16:2` (R <= 64), `0` ->
`1:8,2:8,4:8,8:4,16:2` (bitwise), `1` -> + `32:2`. The P-side prefill hook projects the aux rows in <= 256-row chunks
(`spec_decoder.RowChunkedProjector`: the small-M matmul configs of `project_kv` exceed L1 at the 2048-row chunk).

* Device scenario (`tests/test_spec_serving_scratch.py SPEC_DRAFTER=dflash2`, half B, 3 -> 6 -> 9 -> 17 users, flush /
  migration / no-context user / plain-forced steps): bitwise ladder -- every stream bitwise the plain decode
  (`logs/spec_serving_df2d.log`); default ladder -- 2 divergences, both greedy near-ties of gap 0.125 (one at the
  (8,T=8) step, one at a later plain step of a user that had run it: the fractured path's ulp-level state carries
  forward), nothing else (`logs/spec_serving_df2e.log`); the MTP scenario unchanged (`logs/spec_serving_mtp_regress.log`).
* Served gate (all 8 chips, `scripts/spec_ab_gate.sh` / `spec_gate_variant.sh`, `logs/served_spec_*/summary.txt`):
  GSM8K lm-eval 200 = 0.835 (R <= 64) / 0.82 (bitwise) / 0.82 (mtp); det_probe deterministic; self-consistency over
  1..32 users x 64 tokens ALL MATCH for the bitwise ladder and mtp, 10/63 streams differ with the R = 64 plan (near-tie
  flips) -> the bundle ships `QWEN36_SPEC_ALLOW_FRACTURED=0`. GSM8K text TPOT (bitwise dflash2 / mtp): 9.4 / 12.1 ms at
  1 user, 10.3 / 13.0 at 4, 15.3 / 14.4 at 8, 31.1 / 22.9 at 16, 39 / 38 at 32 (plain); vLLM acceptance length 5.9-6.3
  at k = 7 on lm-eval GSM8K answers (8 users, R <= 64 run), 3.5-3.6 at k = 3.

## 7. Hybrid drafter policy (M4e, 2026-09-28) -- `QWEN36_SPEC_DRAFTER=hybrid`

Section 6's served A/B decides it: the block drafter wins only at <= 4 users (GSM8K TPOT 9.4 / 10.3 ms at 1 / 4 vs MTP
12.1 / 13.0) and loses at 8 (15.3 vs 14.4) and 16 (31 vs 23), and the R <= 64 ladder that would rescue the 8-user band
is not self-consistent. `hybrid` keeps BOTH drafters resident on D and the ladder picks one per plan by width
(`tt/spec_serving.py` `Ladder.drafter_for`; default `1:8,2:8,4:8,8:4,10:3,16:2`, every R <= 32): DFlash2 at T = 8 for
buckets 1 / 2 / 4, the MTP bands (8,4) / (10,3) / (16,2) above, plain past 16. A drafter change is a plan change and
takes the existing flush / migration protocol (4 -> 5 is a band-up change held for a flush, 5 -> 4 migrates, 16 -> 17
flushes into plain). Both drafters' per-slot state is kept current on every spec step (`tt/spec_decoder.py`
`_HybridDrafter`), so a switch needs no re-prefill:

* P runs both prefill hooks and ships both states (payload v3: `mtp.kv.0` + `mtp.hidden` and the `dflash2` group,
  +3 MiB per 179-token request over either single-drafter payload); D imports both (`import_mtp_hidden`, the group import).
* Verify plans keep `out_hidden` AND `out_aux`. Per step: the DFlash2 `commit(plan, P)` always (1.1-1.3 ms); the MTP
  head's K/V for the committed rows by `MTPHead.keep_current(plan, argmax)` -- one traced program per plan running the
  head's LAYER (no lm_head) over the R grid rows with `fc(norm(embed(x_{P_s+j+1})), norm(h_{P_s+j}))` (the verify's
  argmax and post-norm row of row (s, j)) and `forward_verify`'s K/V write at `P_s + j` -- on every DFlash2-drafted step
  and on the first MTP step after one (or after a plain step): **0.82 ms** per step traced (R <= 32; half B); inside the
  MTP band the chain writes the head's K/V exactly as in mtp mode. The MTP select + chain run only when MTP drafts, the
  block step only when DFlash2 drafts; the MTP catch-up of a fresh user runs whichever drafter is active.
* The MTP head's K/V then holds, for every committed position i, the entry of (x_{i+1}, h_i) with the TARGET's hidden
  (what the head's prefill writes, and what vLLM's proposer feeds it for accepted tokens). Observation on the mtp mode
  (unchanged here, byte-identical when selected): the k-step chain writes P_s-1 .. P_s+k-2, so a step that accepts all
  k drafts leaves position P_s+k-1 unwritten (stale block content) before the next chain attends over it -- a draft
  quality matter only (the committed stream never depends on the drafter's state); the hybrid's keep-current fills it
  on the steps where it runs.
* DRAM (D, after warm-up, 1,052,672-token pool): both drafters + the full DFlash2 context pool fit with **5698 MiB**
  free per chip (mtp alone 9078, dflash2 alone 6497), so the context pool is not shrunk.

Exactness: CPU (`tests/test_spec_serving_cpu.py`: the hybrid ladder, the 4 <-> 5 and 16 <-> 17 switches, the
3 -> 5 -> 9 -> 17 ramp with oracle / random / mixed drafts; 46 pass). Device (`tests/test_spec_serving_scratch.py
SPEC_DRAFTER=hybrid`, half B, `logs/spec_serving_hybrid1.log`): 3 -> 5 -> 9 -> 17 users and back, one user without
DFlash2 context, a sampled user forcing plain steps -- 50 steps (19 spec: 8 DFlash2-drafted, 9 MTP-drafted; 30 plain),
3 drafter switches ((4,8) -> flush -> (8,4); (8,4) -> flush -> (10,3) -> flush -> plain; (16,2) -> (2,8) by migration),
3 flushes, 1 migration, **every committed stream bitwise the plain traced decode**, the unheld crossing raises
`SpecProtocolError`. Step walls: (4,T=8) DFlash2 50-55 ms, (8,T=4) MTP 47-51, (10,T=3) 46-48, (16,T=2) 40-42,
plain 35.

Served gate (all 8 chips, `scripts/spec_gate_variant.sh spec_hybrid hybrid`, `logs/served_spec_hybrid/summary.txt`;
the full table is in the plugin's docs/SPECULATIVE.md): det_probe deterministic; self-consistency 1..32 users x 64 tokens
ALL MATCH; GSM8K lm-eval 200 = 0.82 with all 200 generations token-identical to the mtp and dflash2 runs; 6100 spec steps
(1503 DFlash2-drafted, 4597 MTP-drafted), 2361 plain, 29 flushes, 31 migrations, **19 drafter switches**, 0 protocol
errors. TPOT ms / tok/s (hybrid vs the better single drafter at that width): GSM8K text 9.5 / 90 (dflash2 9.4 / 93) at 1
user, 10.5 / 307 (10.3 / 316) at 4, 15.1 / 442 (mtp 14.4 / 471) at 8, 24.0 / 573 (mtp 22.9 / 599) at 16, 38.9 / 690 plain
at 32; 128/128 random 11.9 / 74, 15.2 / 226, 16.2 / 405, 24.6 / 552, 39.6 / 684. Acceptance length 5.15 at 1-4 users
(DFlash2, k = 7), 3.5 at 8 (MTP, k = 3). Per step on D: (1,8) DFlash2 verify 34.2 + commit 0.85 + keep-current 0.71 +
draft 7.6 ms; (8,4) MTP 33.5 + commit 1.1 + select 0.4 + draft 7.9; (16,2) MTP 33.9 + 1.7 + 3.1. P side: both hooks add
~11 ms to a short prefill (118.6 vs 107.2 mtp / 112.5 dflash2 ms total at <= 256 tokens) and the payload is 164.3 MiB per
179-token request (mtp 160.5, dflash2 163.5, plain ~158); D 5698 / P 13071 MiB DRAM free after warm-up.

## 8. Long-context TPOT regression of the hybrid bundle (2026-09-28) -- the missing 2048 sliding window

The v11 sweep (`QWEN36_SPEC_DRAFTER=hybrid`, K=7, decode bucketing on, D pool 1,052,672, random prompts;
`logs/points_pd_v11_merged`) had long-prompt TPOT ~2x the v10 (MTP drafter) sweep at EVERY concurrency: 16k/128
14-15 -> 27-35 ms, 32k 15 -> 34-40, 64k 13 -> 55-59, 128k 18 -> 70; 128/128 unchanged (11.9 / 39.9 at 1 / 32 users).
D's `SpecDecoding metrics` fell to a mean acceptance length of 1.04-1.25 late in the sweep.

### 8.1 A/B matrix (all 8 chips, `scripts/spec_ab_longctx.sh`, `logs/ab_<tag>/summary.txt`; AICLK under firmware control)

Four stacks on this tree, the v11 env otherwise (`vllm bench serve`, random prompts, n = users; the real-text probe
`scripts/longctx_accept_probe.py` = one user, GSM8K-train text of exactly `ctx` tokens, 128 greedy tokens, one streamed
chunk per decode step, so `tok/step` = the acceptance length + 1):

| stack | D DRAM free after warm-up | 128/128 1u / 32u | 16k/128 1u / 32u | 32k/128 1u / 15u | 16k/1024 32u | real text tok/step at 1k / 4k / 8k / 16k / 32k ctx | ms/token at 1k / 32k |
|---|---|---|---|---|---|---|---|
| (3) hybrid default (the v11 config) | 5698 MiB | 13.3 / 37.2 | 27.3 / 34.2 | 35.0 / 40.2 | 20.8 | 4.74 / 4.27 / 2.42 / 2.03 / 1.75 | 9.4 / 29.7 |
| (1) `QWEN36_SPEC_DRAFTER=mtp` K=3 (the v10 config on this tree) | 9078 | 13.0 / 34.2 | 13.6 / 14.8 | 15.0 / 15.8 | 17.0 | 3.28 / 3.12 / 3.05 / 3.05 / 3.37 | 11.3 / 11.9 |
| (2) hybrid resident, never drafting (`QWEN36_SPEC_VLLM_CONFIG=0`: both drafters built, P ships both states, D imports them, no speculative_config -> plain steps) | 5835 | 24.2 / 36.9 | 24.8 / 27.8 | 25.2 / 25.4 | 36.7 | 1.00 everywhere (plain) | 24.2 / 25.2 |
| (4) `QWEN36_SPEC_MTP=0` (no speculation, the floor) | 9956 | 24.2 / 36.7 | 24.7 / 26.4 | 25.3 / 25.4 | 35.9 | 1.00 | 24.2 / 25.2 |

TPOT in ms (mean over the point's requests). The "32-user" long-prompt points never reach 32 live users on D: P
prefills a 16k prompt every ~1.5-3 s while a 128-token decode lasts 2-5 s, so D runs 1-2 live users the whole time
(the D log's `[spec] step` lines of the 16k c32 point: `w_grid=1` x14 / `w_grid=2` x12, vLLM `Running: 0-2 reqs,
Deferred: 22-31`), i.e. inside the hybrid's DFlash2 band, not the plain band. The true 32-live-user long-context
points of the sweeps (8192/1024 and 10000/1024 at 32 users: v10 48.8 / 29.2, v11 46.7 / 29.2) never regressed.

Reading: (2) == (4) at every context length (24.2 / 24.8 / 25.2 vs 24.2 / 24.7 / 25.3 ms at 128 / 16k / 32k for one
user; 36.9 vs 36.7 at 32 users x 128 tokens) -> the resident drafters (3.4 GB of weights + context pool, the KV group
import per admission, the MTP hidden import, the P-side hooks) cost the decode NOTHING: H2 (DRAM placement / page
tables / imports / per-step keep-current) is refuted; keep-current and commit run on spec steps only (checked:
`_HybridDrafter.after_verify` is reached from `SpecDecoder.step` after the verify, never on a plain step). (1) is at
the v10 level (13.6-15.8 ms at 16k-32k; the MTP head's acceptance is context-independent, 3.05-3.37 tok/step on real
text at 1k-32k) -> the tree is fine. (3) is the regression: the block drafter's acceptance falls with the context
length on real text (4.74 tok/step at 1k-2k, 4.27 at 4k, 2.42 at 8k, 2.03 at 16k, 1.75 at 32k) and on random prompts
(3.3 at 128, 2.8 at 4k, 1.6 at 16k, 1.45-1.9 at 32k, 1.0-1.2 at 64k-128k; the pre-fix D log, `[spec] step` lines) while
its step cost grows (verify 35 -> 57 ms, draft 8 -> 15, commit + keep 1.7 -> 3.1 at 128k: three SDPAs over the whole
context), so a T=8 step of ~48-76 ms buys 1.0-2 tokens: 24-70 ms/token. H1 confirmed: the device drafter attended the
WHOLE context while z-lab/Qwen3.8-27B-DFlash2 is trained with `sliding_window: 2048` on every layer (config.json
`layer_types: sliding_attention` x5, `use_sliding_window: true`), and the reference (`dflash/model.py`, the vLLM PR's
`qwen3_dflash2.py`, `DFlash2HostReference._attn_mask`: `query_pos - key_pos < 2048`) never lets a block row see a
context key older than 2048 positions; past 2k tokens the device drafts were out of distribution.

### 8.2 The fix: the window on device (`QWEN36_DFLASH2_DEVICE_WINDOW`, default = the checkpoint's 2048)

`DFlash2Drafter.step_forward` passes `sliding_window_size=2048` to the paged SDPA decode of every layer: per virtual
user (one block row) the kernel attends keys `[cur_pos + 1 - 2048, cur_pos]` (`rt_args_common.hpp`: chunk-aligned
reads, the partial chunk masked to the exact start). The block's 8 rows share `cur_pos = P + 7`, so row j's window is
the reference's shifted by `7 - j` positions (it lacks the `7 - j` oldest keys of the reference's window, sees nothing
the reference does not); the K/V write and the commit are unchanged. `0` restores the whole-context attention (A/B
only). The transport window `QWEN36_DFLASH2_CONTEXT_WINDOW` (P ships only the tail) stays 0 by default; the D-side
`note_context` warns when a shipped tail is shorter than the device window.

Tests: `tests/test_dflash2_window_op.py` (one chip): the drafter's SDPA configuration (8 rows sharing cur_pos, 2 kv
heads x 128, bfp8 paged K/V, block 64) with the window equals attention over keys `[cur_pos + 1 - 2048, cur_pos]`
(PCC 0.9997 at 3000 / 5000 context) and differs from the whole-context result (PCC 0.81 / 0.62); below 2048 both are
identical (PCC 0.99965). `tests/test_dflash2_spec_scratch.py DF_PCC_LEN=3000 DF_BPU=64` (half B): the device draft
logits vs the fp32 host reference (which applies the window) on a 3000-token GSM8K prompt -- section 8.3.

### 8.3 Re-measure with the window (served, `logs/ab_hybrid_window/`; standalone, `logs/df2_window2.log` / `df2_window8k.log`)

Served hybrid stack, the v11 env + the device window (default), same driver / points as 8.1 (TPOT ms, mean; the
"32u" long-prompt points run 1-2 live users on D, see 8.1):

| point | v10 sweep (mtp) | v11 sweep (hybrid) | today, hybrid no window | today, mtp | **today, hybrid + window** | D tok/step (`[spec]` lines of the point) |
|---|---|---|---|---|---|---|
| 128/128 1u | 12.3 | 11.9 | 13.3 | 13.0 | 14.5 | -- |
| 128/128 8u | 15.3 | 16.2 | -- | -- | 17.0 | 3.34 |
| 128/128 32u | 38.1 | 39.9 | 37.2 | 34.2 | 37.8 | plain band |
| 16k/128 1u | 13.6 | 27.1 | 27.3 | 13.6 | **17.4** | -- |
| 16k/128 32u (31u in the sweeps) | 14.9 | 31.4 | 34.2 | 14.8 | **19.2** | 2.79 (was 1.64) |
| 32k/128 1u | 15.0 | 34.9 | 35.0 | 15.0 | **17.9** | 2.74 (was 1.88) |
| 32k/128 15u | 15.6 | 40.1 | 40.2 | 15.8 | **24.0** | 2.70 (was 1.45) |
| 64k/128 1u (n = 1) | 13.3 | 59.3 | -- | -- | 47.9 (one random request; vLLM acceptance 1.2-1.6 on it) | 2.0 |
| 64k/128 8u | 14.9 | 55.0 | -- | -- | **32.2** (median 30.4) | 2.25 (was 1.0-1.2) |
| 128k/128 1u / 4u | 18.0 / 19.5 | 69.9 / 71.5 | -- | -- | **blocked: P (half A) wedged** on the 128k prefill, below | 1.94 during the pull |
| 16k/1024 32u | -- | -- | 20.8 | 17.0 | not reached (wedge) | -- |

Per step at 16k-32k with the window (D log): verify 41.8-43.4 + commit/keep 2.6 + **draft 8.9 ms** (the block SDPA
is context-independent now: 8.8 ms at 64k-128k too, was 15.2 at 128k); the verify's own SDPA still grows with the
context (35 ms at 128 tokens, 43 at 32k, 47 at 64k, 51 at 128k: the target's T=8 rows over the whole context). Random
prompts: 2.7-2.8 tokens/step at 16k-32k (was 1.45-1.9), 2.0-2.25 at 64k-128k (was 1.0-1.2); the random-prompt
acceptance still declines past 32k, so at 64k+ the (1..4, T=8) DFlash2 band (~56 ms/step / 2.1 = 27 ms/token) stays
behind the MTP band (~50 ms / 3.2 = 15). Real text (the standalone probes, half B, GSM8K questions vs the fp32 host
reference WITH the window): 3000-token prompt -- PCC 0.9991 (min 0.998), argmax agreement 0.921, path agreement
0.841, **4.12 tok/step**; 8192-token prompt -- PCC 0.9991 (min 0.994), argmax 0.949, path 0.949, **3.38 tok/step**
(the pre-fix served probe on GSM8K text: 2.42 at 8k). The served real-text probe of the fixed stack (1k..64k) and the
128k / 16k-1024 points were not reached: the P half wedged (below).

**Wedge (2026-09-28 20:51, P = chips 0,1,6,7):** the first 128k request of the fixed stack hung P inside the traced
chunked prefill at chunk 59/64 (`_prefill_traced_chunked_tp` -> `ttnn.synchronize_device`, py-spy: the engine core
spinning in `FDMeshCommandQueue::finish`), after 118,784 positions of the DFlash2 hook had staged normally; D idle
(waiting for the payload). Killed by PID after 10 min; afterwards chip 1 reports `Read 0xffffffff over PCIe ID 1: the
board should be reset` and chips 0 / 6 / 7 hang on open, chips 2-5 (D) open and run. Not caused by the fix (P runs
none of the changed drafter code; the same P code prefilled three 128k prompts at 19:23-19:25 and every 128k point of
the v10 / v11 sweeps) -- an intermittent half-A hang at 128k prefill; no `tt-smi -r` was issued (hard rule).

Open observation (pre-existing, not the drafter's): the standalone PCC probes' committed stream at 3000 / 8192
context was NOT identical to the plain traced decode (`exact vs decode=False`) while every short-prompt config of the
same runs was (`exact_vs_plain_decode=True`, as in sections 4-7). Drafts cannot change a greedy stream, so this is
the R=8 verify's numerics vs the one-row decode at long context (the T=8-row SDPA's chunking / reduction order) --
the bitwise claim of `docs/SPECULATIVE.md` was established on prompts <= 300 tokens and 128/128; a near-tie probe at
>= 3k context should decide whether these are ulp-level near-tie flips like the R=64 ones.

Next: (a) `QWEN36_DFLASH2_CONTEXT_WINDOW=2048` on P (ships the 2048-token tail only: the hook's 73 ms per 2048-token
chunk x 64 chunks and ~2.6 GB of payload at 128k go away; +800 ms TTFT at 16k today vs mtp) once it has a served
gate; (b) the random-prompt acceptance decline past 32k with the window in place (real text at 16k-64k served, the
128k points) once half A is back; (c) the long-context bitwise question above.

## 9. Context-aware drafter selection, the transport window and the long-context near-tie question (2026-09-28)

### 9.1 The context rule (`QWEN36_SPEC_DFLASH2_MAX_CTX`, tt/spec_serving.py)

Section 8.3 left the DFlash2 band behind the MTP head at long context even with the device window (random prompts:
TPOT 16k 17.4 vs 13.6 ms, 32k 17.9 vs 15.0, 64k 32 vs 15; real text 3.38 tok/step at 8k vs MTP's context-independent
3.05-3.37 at a ~10 ms cheaper step). The hybrid ladder now chooses the drafter by width AND by context length:

* The rule: DFlash2 drafts only while the LONGEST live context on the grid -- the rows' decode positions, i.e. prompt
  + generated tokens, which `SpecDecoder.step` already has -- is `<= QWEN36_SPEC_DFLASH2_MAX_CTX` (default
  `HYBRID_DFLASH2_MAX_CTX`, set from the crossover measurement of 9.2; `0` = no rule). Above it the grid runs the LONG
  ladder (`QWEN36_SPEC_LADDER_LONG`, default the mtp ladder `1:4,2:4,4:4,8:4,10:3,16:2`): the MTP head at every
  width. The long ladder adds three verify plans, (1,4) / (2,4) / (4,4), to the decoder (9 plans, 72 migration
  pairs; `Ladder.all_plans`), the wider plans are shared. `Ladder.drafter_for(plan, long)` names the drafter: a
  plan of the short ladder with w <= 4 is DFlash2's, every other plan the MTP head's; with `QWEN36_SPEC_K=3` both
  ladders clamp onto the same (w,4) plans and the mode alone picks the drafter (no plan change at a switch).
* No flapping: the mode (`SpecServingState.long`) is evaluated every step, but a fixed set of users can only GROW, so
  for a given set it flips at most once (short -> long: a user grows past the limit, or a long-prompt user is
  admitted). It flips back only when the long users have left AND the longest remaining context is below the limit
  minus `QWEN36_SPEC_DFLASH2_CTX_HYST` (default 1024). Every mode change therefore coincides with a growth crossing
  or an ownership change; two consecutive steps with the same owners can never flip twice
  (`test_hybrid_context_rule_never_flaps_between_ownership_changes`).
* The switch is a drafter change and (except in the K=3 corner) a plan change, e.g. (4,8) -> (4,4), taken through
  the flush / migration protocol: pending rows a_s <= 3 migrate; longer ones (a DFlash2 step accepts up to 7) do not
  fit T-1 = 3, and unlike a width crossing the scheduler cannot hold anything (no admission is involved, or the
  admission lands inside the current band), so the state machine runs ONE flush step of its own at the current
  mode's plan for the grid width -- always a same-band plan (a join inside the band, a leave, or no change), so the
  pending rows fit it -- and switches on the next step (`StepPlan.ctx_flush`, stats `ctx_switches` /
  `ctx_flushes`). A width crossing that was not held still raises `SpecProtocolError`. On the way back
  ((4,4) -> (2,8), larger T) the pending MTP rows (<= 3) always migrate. `hold_info` reads the band structure of the
  mode the installed plan was chosen in (`plan_long`).
* Both drafters' state stays current on every spec step as before (`_HybridDrafter`), so neither switch direction
  re-prefills anything.

Tests. CPU (`tests/test_spec_serving_cpu.py`, 52 pass): the ladder / knob parsing (incl. the K=3 corner and
`QWEN36_SPEC_DFLASH2_MAX_CTX=0`); one user growing past the limit with 7 rows pending -> exactly one own flush at
(1,8), then (1,4) MTP for good (random drafts: a plain migration when the rows fit); a long-prompt admission INTO the
DFlash2 band (2 users at (2,8) with 7 pending -> not a width crossing, the scheduler does not hold it -> the machine
flushes at (4,8) with the new user on the grid, then (4,4) MTP; back to (2,8) by migration when it leaves; the
hysteresis keeps MTP while a user inside (limit - hyst, limit] remains); the churn scenario above; every stream the
greedy one. Device (half A, `scripts/spec_scenario_run.sh A spec_serving_ctx3 SPEC_DRAFTER=hybrid
SPEC_DFLASH2_MAX_CTX=100 SPEC_DFLASH2_CTX_HYST=20`, `logs/spec_serving_ctx3.log`; also `ctx2` with limit 40 / 10):
the 3 -> 5 -> 9 -> 17 scenario with the gsm8k users crossing the limit mid-generation -- 51 steps (20 spec, 30 plain),
3 mode changes, 3 drafter switches, 7 plan changes, plans visited (2,8) / (4,8) DFlash2 and (2,4) / (8,4) / (10,3) /
(16,2) in long mode (+ the (8,4) short-mode band), 3 flushes, 1 migration, **every committed stream bitwise the plain
traced decode** (ctx2: 51 steps, 1 mode change, 6 plan changes, bitwise). The machine's own flush did not fire on
device (the crossings there coincided with scheduler flushes or had <= 3 rows pending) -- it is the CPU tests' case;
on device it reuses the flush path the scheduler's hold takes. Step walls: (2,8) DFlash2 48.6 ms, (2,4) MTP/long
41.9, (4,8) 51.2, (8,4) 47.6, (10,3) 45.8, (16,2) 40.3, plain 35.

### 9.2 Crossover measurement (served, 1 user, real text; `scripts/spec_xover_chain.sh`, `logs/ab_xover_*/summary.txt`)

Two probe-only stacks (hybrid + transport window with the context rule OFF, i.e. DFlash2 at every context length for one user; the MTP drafter K=3), AICLK under firmware control, `scripts/longctx_accept_probe.py` with GSM8K text and a long technical document (tt-metal tech reports), 128 greedy tokens, 2 repeats (means):

| text | context | DFlash2 tok/step | DFlash2 ms/step | DFlash2 ms/token | MTP tok/step | MTP ms/step | MTP ms/token | winner | TTFT hybrid / mtp ms |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 2,048 | 4.74 | 46.4 | **9.5** | 3.20 | 38.1 | **11.7** | DFlash2 | 476 / 374 |
| gsm8k | 4,096 | 4.57 | 47.1 | **10.0** | 3.12 | 38.6 | **12.1** | DFlash2 | 831 / 642 |
| gsm8k | 6,144 | 5.57 | 48.0 | **8.3** | 3.51 | 39.0 | **10.9** | DFlash2 | 1150 / 962 |
| gsm8k | 8,192 | 3.71 | 47.5 | **12.6** | 3.01 | 38.7 | **12.7** | tie | 1352 / 1172 |
| gsm8k | 12,288 | 4.13 | 48.3 | **11.4** | 2.96 | 39.5 | **13.1** | DFlash2 | 1957 / 1766 |
| gsm8k | 16,384 | 3.88 | 49.0 | **12.4** | 3.09 | 40.2 | **12.8** | DFlash2 | 2571 / 2444 |
| doc | 2,048 | 3.37 | 45.6 | **13.4** | 2.58 | 37.8 | **14.4** | DFlash2 | 476 / 380 |
| doc | 4,096 | 4.27 | 46.8 | **10.7** | 2.98 | 38.5 | **12.8** | DFlash2 | 804 / 640 |
| doc | 6,144 | 6.74 | 48.4 | **6.9** | 3.76 | 38.8 | **10.1** | DFlash2 | 1176 / 922 |
| doc | 8,192 | 3.88 | 47.2 | **11.9** | 2.88 | 38.7 | **13.2** | DFlash2 | 1343 / 1248 |
| doc | 12,288 | 4.41 | 48.3 | **10.6** | 3.08 | 39.8 | **12.6** | DFlash2 | 1904 / 1747 |
| doc | 16,384 | 2.98 | 48.0 | **15.9** | 3.13 | 40.0 | **12.7** | MTP | 2559 / 2458 |

Reading: DFlash2 wins on both texts up to 12k (ms/token 8.3-12.5 vs 11.0-13.1 on GSM8K text, 6.9-11.9 vs 10.0-13.2 on the document; its step is ~9 ms dearer but it commits 3.7-6.7 tokens per step vs 2.6-3.8) and loses at 16k on the document (15.9 vs 12.7; 2.98 tok/step) while roughly tying on GSM8K text (12.4 vs 12.8); random prompts lose at 16k already (8.3: 17.4 vs 13.6 ms TPOT). The acceptance is not monotone in the context (the 6k prompts of both corpora happen to be easy text), so the limit is set at the last length where DFlash2 wins on both corpora: `HYBRID_DFLASH2_MAX_CTX = 12288` (contexts above run the MTP plans). TTFT of the hybrid stack with the transport window vs mtp: +150-200 ms at 8k-16k (+5-7%: both prefill hooks, the tail's projection), +130-290 ms at 32k (+2.5-5%); the straddling chunk's projection is sliced to the window since (the gate's numbers are the ones to quote).

### 9.3 Transport window on by default (`QWEN36_DFLASH2_CONTEXT_WINDOW` follows the device window)

`aux_hidden.dflash2_context_window()`: an explicit value wins; otherwise the transport window equals the device
window (`QWEN36_DFLASH2_DEVICE_WINDOW`: unset -> 2048, 0 -> 0 = ship everything, n -> n). Both engines read one
environment (scripts/serve_pd.sh), so the shipped tail always covers what D attends. P's hook skips the all-gather +
projection of every segment that ends before `total_len - 2048` (block aligned) and ships the blocks covering
`[first_pos, total_len)` (`export_kv_groups`: `block_index` / `first_pos`); D imports them at the request's blocks
(`import_kv_groups`), `note_context` checks `first_pos <= total - window` (no warning in the gate's D log), and the
device SDPA reads keys `[cur_pos + 1 - 2048, cur_pos]` only -- the blocks below `first_pos` hold stale finite bfp8
rows the kernel never attends (chunk-aligned reads are masked to the window start). Plugin host tests cover the
windowed payload (`test_pd_payload.py`: block_index = the tail, first_pos = BLOCK).

Payload per request (P log `[pd] staged`, the whole payload incl. 16 main layers + GDN + MTP state) and single-user
TTFT (`scripts/latency_probe.py`, osl 32, x2, and the real-text probe's first chunk), the A/B stacks of today
(`logs/ab_hybrid_base` = hybrid without any window, `logs/ab_xover_df2` = hybrid + device + transport window,
`logs/ab_xover_mtp` = mtp; the v12 gate's rows are in 9.5):

| prompt | hybrid, no window: payload / export | hybrid + transport window: payload / export | mtp: payload / export | TTFT hybrid no window | TTFT hybrid + window | TTFT mtp | hybrid+window vs mtp |
|---|---|---|---|---|---|---|---|
| 8,191 | 851.8 MiB / 206 ms (128 blocks of context K/V) | **733.0 MiB** / 62-76 ms (33 blocks: 2111 positions from 6080) | 691.8 MiB / 169 ms | -- | 1418-1618 (probe 1352) | 1165-1424 (probe 1172) | +9-14% |
| 16,383 | 1555.8 MiB / 336 ms (256 blocks) | **1277.0 MiB** / 126-306 ms (33 blocks from 14272) | 1235.8 MiB / 143-222 ms | 3651-3927 | 2529-3020 (probe 2565) | 2349-2849 (probe 2451) | **+5-7%** (was +28-37%) |
| 32,767 | 2963.8 MiB / 576 ms (512 blocks) | 2364 MiB (33 blocks) | 2323.8 MiB / 305-377 ms | 7544-7759 | 5243-5806 | 5116-5516 | +2.5-5% (was +22-26%) |

The DFlash2 group shrinks from 20 KB x prompt (160 / 320 / 640 MiB) to 41 MiB; the remaining +41 MiB over mtp is that
tail plus the MTP hidden row. The remaining TTFT gap over mtp (~150-200 ms at 8k-16k) is P's work for the two
drafters: the MTP prefill hook (the head's layer over every position) and the DFlash2 hook's all-gather + projection
of the tail's chunks (67 ms per 2048-row chunk; the chunk straddling the window start was projected whole in these
runs and is sliced to the window since -- the gate's TTFT rows below carry that).

### 9.4 The long-context bitwise question: near-tie classification (`tests/test_dflash2_spec_scratch.py`, `DF_PCC_LEN`)

The PCC probe's committed stream at 3000 / 8192 real-text tokens was not identical to the one-row plain decode
(8.3). The probe now classifies such a divergence: the plain decode's logits at the divergence index (eager one-row
decode, `_neartie_probe`) and the same spec loop re-run with two other draft policies -- random drafts (every step
accepts 0: the token is row 0 of its own step) and oracle drafts (the plain stream itself: every row accepted, the
token is row (i-1) mod 8) -- on half B (`logs/df2_nt3000.log`, `logs/df2_nt8192.log`, `DF_BPU=64` / `160`):

| prompt | divergence | plain decode's logits at the index | rank of the committed token | random drafts | oracle drafts |
|---|---|---|---|---|---|
| 3000 (GSM8K) | token 2: got 12702, exp 248068 | exp 17.625 vs got 17.125: **gap 0.500** (= 4 bf16 ulps at 17.x; top-2 gap 0.5) | 1 (second best) | same token 2, same pair, same gap | same |
| 8192 (GSM8K) | token 8: got 9764, exp 760 | 24.375 vs 24.375: **gap 0.000 -- an exact bf16 tie** | 0 (shares the max) | same token 8, same pair, gap 0 | same |

Reading: both divergences are greedy (near-)ties of the plain decode's bf16 logits and both are IDENTICAL across the
three draft policies (same index, same token pair, same gap), so they come from the R = 8 verify's numerics at that
position -- the T = 8 rows of one user over a >= 3000-key context (the SDPA's chunk reduction order / the argmax's
tie-break; at 8192 the two tokens have the same bf16 logit and the one-row decode's argmax takes the lower index
while the verify's takes the other) -- not from the drafts, and every other token of the streams (63 + 175 draft
rows, 4.12 / 3.38 tok/step) is the plain one. So the bitwise claim of docs/SPECULATIVE.md holds as established: on
prompts up to a few hundred tokens and 128/128 (every scenario test, the served self-consistency ALL MATCH at 1..32
users x 64 tokens); at >= 3k context the R <= 32 verify is near-tie-bounded like the fractured path, with a wider
observed gap (0.5 = 4 ulps vs the <= 0.25 of the R = 64 flips; n = 2 prompts). Not a bug (both tokens are equally
likely under the model at bf16 resolution); a possible follow-up is a lowest-index tie-break in the verify's argmax
so the exact-tie case (8192) matches the decode.

### 9.5 Served gate v12 (all 8 chips, AICLK under firmware control, `scripts/spec_gate_v12.sh`, `logs/gate_v12/summary.txt`)

Config: `QWEN36_SPEC_DRAFTER=hybrid`, K=7, the device window (2048), the transport window (default = 2048), the
context rule (default 12288 / hysteresis 1024), decode bucketing on, D pool 1,052,672, `scripts/force_aiclk.py 0`
before the start (firmware control: 1350 MHz under load). D after warm-up: 9 verify plans, 5643 MiB DRAM free per
chip (5698 with 6 plans). Stack up in 195 s.

* det_probe: deterministic (3 x 3 prompts identical). Self-consistency `pd_concurrency_check --conc 1 2 4 8 16 32
  --max-tokens 64`: **ALL MATCH** (63/63 streams). GSM8K lm-eval 200 at 8 users: **0.82 +- 0.027** flexible-extract,
  **0 degenerate** generations. D log: 0 tracebacks / protocol errors, 0 transport-window warnings.
* D stats over the gate: 5650 spec steps (1104 DFlash2-drafted, 4546 MTP-drafted), 1285 plain, 17 flushes, 61 plan
  changes, 22 migrations, **21 drafter switches, 9 context-rule mode changes** (max live context 32784 at the end),
  0 own flushes (every crossing had <= 3 rows pending or coincided with an admission). Per step at 1 user: (1,8)
  DFlash2 verify 35.6-36.3 + post 1.6 + draft 7.8 ms; (1,4) MTP in long mode verify 31.5-32.7 + post 1.1-1.3 +
  draft 6.8-7.0 (the MTP keep-current does not run inside the MTP band: 0 keep steps).
* `vllm bench serve` (random prompts; v10 = `logs/points_pd_v10_merged`, mtp (tree) = `logs/ab_mtp_tree` +
  `logs/served_spec_mtp`; v11 hybrid for reference: 128/128 11.9 / 15.2 / 16.2 / 24.6 / 39.6 ms, 16k 27-35, 32k 34-40,
  64k 55-59):

| point | v12 TPOT ms | v10 TPOT | mtp (tree) TPOT | v12 t/s | v10 t/s | mtp t/s | v12 TTFT ms | v10 TTFT | mtp TTFT | TPOT vs v10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128/128 x1 | 11.8 | 12.3 | 12.1 | 75 | 73 | 74 | 213 | 194 | 190 | -4% |
| 128/128 x4 | 15.0 | 13.8 | 13.8 | 229 | 247 | 245 | 266 | 238 | 244 | +9% |
| 128/128 x8 | 15.9 | 15.3 | 15.3 | 412 | 435 | 429 | 333 | 287 | 303 | +4% |
| 128/128 x16 | 24.3 | 23.0 | 23.1 | 559 | 569 | 574 | 431 | 521 | 501 | +5% |
| 128/128 x32 | 39.5 | 38.1 | 38.0 | 694 | 709 | 723 | 674 | 723 | 607 | +4% |
| 16,384/128 x1 | 14.0 | 13.5 | 13.6 | 27 | 28 | 28 | 2,933 | 2,843 | 2,859 | +3% |
| 16,384/128 x4 | 15.4 | 15.2 | — | 42 | 45 | — | 6,423 | 7,444 | — | +1% |
| 32,768/128 x1 | 15.5 | 15.0 | 15.0 | 16 | 16 | 16 | 6,276 | 5,887 | 6,177 | +3% |
| 32,768/128 x4 | 14.2 | 13.9 | — | 22 | 22 | — | 13,208 | 13,429 | — | +2% |
| 65,536/128 x1 | 13.5 | 13.3 | — | 9 | 9 | — | 13,036 | 13,083 | — | +2% |
| 65,536/128 x4 | 14.2 | 14.1 | — | 10 | 8 | — | 30,347 | 33,318 | — | +1% |

  Long-context TPOT is within **+1-3 %** of v10 at 16k / 32k / 64k x 1 and 4 users (the DFlash2 band is off above
  12288: these points run the MTP plans; v11 was +100 %). 128/128: 1 user -4 % (DFlash2), 8-32 users +4-5 % (the same
  as the v11 hybrid gate: 16.2 / 24.6 / 39.6 -- the MTP bands and the plain band are the v10 loop op for op, so this
  is run-to-run + the hybrid's keep-current / commit on the MTP steps, ~1 ms), 4 users +9 % (15.0 vs 13.8: the (4,8)
  DFlash2 band on RANDOM 128-token prompts accepts less than the MTP head's 3 drafts; on GSM8K text the same band is
  10.5 vs 13.0 ms, section 7). Aggregate t/s: 75 vs 73 at 1 user, -7 / -5 / -2 / -2 % at 4 / 8 / 16 / 32.
  TTFT at 16k x 1: 2933 vs v10 2843 (**+3 %**; mtp on this tree 2859), 32k 6276 vs 5887 (+7 %; mtp 6177: +1.6 %),
  64k 13036 vs 13083; x4 points 6423 / 13208 / 30347 vs 7444 / 13429 / 33318 (n = 4, arrival-order dependent).
* TTFT probe (osl 32, x2): 8k 1460-1520 (mtp stack 1165-1424), 16k 2795-2889 (mtp 2349-2849; v10 bench 2843),
  32k 5933-5940 (mtp 5116-5516; v10 5887). The hybrid+window stack is within ~5 % of v10 / the mtp bench points and
  ~+5-10 % of the mtp stack's best probe runs (the two drafters' prefill hooks on P).
* Real-text acceptance probe (1 user, 128 tokens; the rule switches drafters at 12288):

| text | ctx | ms/step | tok/step | ms/token | drafter |
|---|---|---|---|---|---|
| gsm8k | 1,024 / 2,048 / 4,096 / 8,192 | 46.0 / 46.3 / 47.0 / 47.2 | 4.74 / 4.74 / 4.57 / 3.71 | 9.4 / 9.5 / 10.0 / 12.4 | DFlash2 (1,8) |
| gsm8k | 12,288 / 16,384 / 32,768 | 39.8 / 40.3 / 41.8 | 2.84 / 2.98 / 3.46 | 13.8 / 13.3 / 11.9 | MTP (1,4), long mode |
| doc | 1,024 / 2,048 / 4,096 / 8,192 | 46.0 / 45.6 / 46.8 / 47.2 | 4.57 / 3.37 / 4.27 / 3.88 | 9.8 / 13.3 / 10.7 / 11.9 | DFlash2 (1,8) |
| doc | 12,288 / 16,384 / 32,768 | 40.0 / 40.2 / 41.8 | 3.05 / 3.20 / 3.88 | 12.9 / 12.4 / 10.5 | MTP (1,4), long mode |

  vs the mtp stack (9.2): the DFlash2 band wins 1k-8k by 1-2.5 ms/token, the MTP plans above 12k are the mtp stack's
  numbers (12.4-13.4 at 16k; 11.9 / 10.5 at 32k vs mtp's 11.9); the 12288 point is the first one past the limit
  (prompt 12288 + generated > 12288 -> long after the first step), 13.8 vs DFlash2's 11.4 in 9.2 -- the limit could
  sit one step higher (12,288 + 128) for 128-token outputs; kept at the measured crossover.
