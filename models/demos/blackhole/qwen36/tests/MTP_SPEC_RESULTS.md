# MTP head as the first drafter (M3) -- results, 2026-09-25, half B (P150x4, TP=4), tt-metal `qwen38-pd-disagg`

Code: `tt/mtp_head.py` (device head + host torch reference), `tt/weight_mapping.py::load_qwen36_mtp_state_dict`,
`tt/verify_step.py` (`keep_hidden` -> `plan.out_hidden`), `tt/model.py` (`prefill_hidden_hook`, additions only),
`tt/verify_grid.py::chain_drafts`, tests `tests/test_mtp_spec_scratch.py` (device), `tests/test_mtp_cpu.py` (CPU),
runner `scripts/mtp_spec_run.sh`. Logs: `logs/mtp_smoke2.log`, `logs/mtp_full1.log` (+ `.json`), `logs/mtp_diag32.log`.

## Semantics (vLLM qwen3_5_mtp.py / llm_base_proposer.py, HF weights `mtp.*`)

One full-attention Qwen3.5 decoder layer (`mtp.layers.0`, 24 q / 4 kv heads x 256, q/k norm, partial RoPE 64, sigmoid
output gate, SwiGLU 17408) with its own KV. Input at position i: `fc(cat(pre_fc_norm_embedding(embed(x_{i+1})),
pre_fc_norm_hidden(h_i)))`, `h_i` = the main model's POST-final-norm hidden state (vLLM feeds the target's returned
`hidden_states`, i.e. the LM-head input); output through `mtp.norm` and the SHARED lm_head -> draft for `x_{i+2}`.
Chain: the drafted token + the head's own post-`mtp.norm` output at position i+1, same layer/KV.

## M3a -- module + host reference

* Host reference (`MTPHostReference`, fp32 torch from the safetensors) == an MTP assembled from HF transformers' own
  `Qwen3_5DecoderLayer` / `Qwen3_5RMSNorm` / `Qwen3_5TextRotaryEmbedding` with the `mtp.*` weights: max|d| 0.0 on a
  15-token sequence, 1e-5 incremental (prefill + 3 chained steps), argmax agreement 1.0 (`test_mtp_cpu.py`).
* Device head vs host reference (`MTP_PCC=1`: 128-token GSM8K prompt, 8 verify steps at w=1, k=3 = 27 draft steps;
  the reference is fed the SAME device inputs -- main hidden rows from the prefill, hidden_in before each step -- and
  keeps its own KV): **PCC mean 0.9977, min 0.9951, draft-argmax agreement 0.926** (chain step 0: 0.9975 / 0.889,
  step 1: 0.9980 / 1.000, step 2: 0.9977 / 0.889). Weights bfp8 (attention, fc, o) / bfp4 (MLP gate/up) as the main layers.

## M3b -- prefill + per-step draft

* Prefill (`MTPHead.prefill_hook`, installed as `model.prefill_hidden_hook`; the hook fires inside the masked-bucket
  prefill with the bucket's pre-final-norm residual, traced-bucket output or eager tensor): ONE eager forward over the
  bucket with the tokens shifted by one (x_1..x_{n-1} at 0..n-2) -> the head's KV holds 0..n-2; row n-1 (x_n, h_{n-1})
  is the first draft step. Everything fractured (distributed pre-fc norms, row-parallel fc partials + reduce-scatter);
  the naive replicated full-width `ttnn.rms_norm` sized its static CBs into the persistent L1 buffers (smoke1).
  Cost: **15.6 ms eager per request** (159 prefills), not traced (M3d option: trace per bucket if it matters).
* Draft step (traced per width): embedding gather -> pre-fc norms (decode sharded configs) -> fc (1D decode matmuls on
  replicated weights) -> the layer's decode forward (paged KV write at the position, SDPA) -> mtp.norm -> sharded
  lm_head -> per-device (argmax, max) [TP, w] readback; the post-norm output is copied into `hidden_in` in-trace.
  **Traced step: w=1 2.39 ms, w=8 2.64 ms, w=32 3.37 ms** (median of 50, incl. upload + readback).
  `select_hidden` (verify `out_hidden` -> hidden_in, exact 0/1 matmul + copy, eager): 0.3-0.5 ms.
* VerifyStep(keep_hidden=True) keeps the LM-head input rows of every grid row as a third trace output (`plan.out_hidden`,
  [1,1,R,dim] replicated DRAM); the draft's first step selects row (s, a_s).

## M3c -- end-to-end speculative decode (real prompts: 3 base + 8 GSM8K chat-templated + 3 code; >= 64 tokens/user)

`logs/mtp_full1.json`. Plain decode = the traced served decode at the same width in the same process. w=1 configs run
10 prompts each (0-6, 11-13), w=8 prompts 0-7, w=32 all 14 (mod).

| config (w,k) | T | R | runs x users | exact vs plain decode | accept len tok/user/step (base/gsm8k/code) | accept rate | spec tok/s (agg) | plain decode tok/s (agg) | speedup | ms/step: verify + select + draft = total (decode ms/step) |
|---|---|---|---|---|---|---|---|---|---|---|
| (1,1) | 2 | 2 | 10 x 1 | **bitwise** | 1.92 (1.91/1.92/1.95) | 0.92 | 61.5 | 39.2 | x1.57 | 28.4 + 0.3 + 2.5 = 31.3 (25.5) |
| (1,2) | 3 | 3 | 10 x 1 | **bitwise** | 2.72 (2.60/2.68/2.89) | 0.86 | 76.0 | 39.0 | x1.95 | 30.3 + 0.3 + 4.9 = 35.8 (25.6) |
| (1,3) | 4 | 4 | 10 x 1 | **bitwise** | 3.33 (3.15/3.23/3.69) | 0.78 | 83.7 | 39.1 | x2.14 | 32.1 + 0.3 + 7.3 = 40.1 (25.6) |
| (8,1) | 2 | 16 | 1 x 8 | **bitwise** | 1.92 (1.91/1.92/-) | 0.92 | 450.6 | 294.1 | x1.53 | 30.9 + 0.4 + 2.8 = 34.1 (27.2) |
| (8,2) | 3 | 24 | 1 x 8 | **bitwise** | 2.65 (2.62/2.67/-) | 0.82 | 496.0 | 293.8 | x1.69 | 36.6 + 0.4 + 5.5 = 42.7 (27.2) |
| (32,1) | 2 | 64 | 1 x 32 | near-ties only (see below) | 1.93 (1.92/1.92/2.00) | 0.93 | 1077.9 | 910.0 | x1.18 | 53.3 + 0.5 + 3.5 = 57.4 (35.2) |

(1,3) per prompt: base 3.00 / 3.30 / 3.15, gsm8k 3.42 / 3.14 / 2.95 / 3.47, code 3.32 / 3.94 / 3.88 tok/step
(97 tok/s on the fibonacci/binary-search completions). Accept-length histograms in the json (`accept_hist`).

### (32,1): every divergence is a bf16 near-tie of the plain decode, and the stream is draft-independent

`logs/mtp_diag32.log`. 15 of 32 users diverge from the plain decode, always per PROMPT (users 3/17/31, 4/18, 7/21,
8/22, 9/23, 10/24, 11/25 = 7 prompts, identical token index and identical alternative for every twin -> no grid-row
dependence). The near-tie probe (an eager decode forward on the decode programs at the divergence position) gives,
for every one of the 15: plain token logit minus committed token logit = **0.125 (one bf16 ulp at |logit| 16..32)**
in 12 cases, 0.25 in 2, **0.000 (exact tie)** in 1; the committed token has rank 1 (13) or 2 (2). The oracle-draft
and random-draft runs of the same config commit streams **bitwise identical to the MTP-draft run** (mechanism =
draft-independent; accept len 1.45 / 1.00 because the oracle follows the plain stream after the flip). So (32,1)
is the documented numerics of the R>32 fractured verify path (fp32 reduce-scatter vs the decode's fused all-reduce,
verify_step.py) flipping greedy ties, not a plumbing bug; every R<=32 plan (the fused-AR path = the decode step's own
ops) is bitwise exact over 140 prompt-runs. The test asserts bitwise exactness for R<=32 and near-tie-bounded
(gap <= 0.25) divergences plus policy identity for R>32.

## M3d -- warm-up order

Compile-first (VERIFY_W32_AUDIT.md): decode references (all widths), MTP draft steps (all widths), verify bodies +
MTP selects (all plans), MTP prefills (all buckets) are compiled eagerly, then the prefill warm-up (chunk trace +
masked-bucket traces + slot writes), the warm 1-user prefill (hook installed), then the captures (decode, MTP steps,
verify). Draft step is traced per width (2.4-3.4 ms; eager it would be ~70 host-dispatched ops). No wedge in 4 device
processes (smoke1's TT_THROW left a teardown hang the runner killed after 180 s; chips stayed healthy).

## Draft cost per verify step

k x traced step + select: k=1 2.5-3.5 ms, k=2 4.9-5.5 ms, k=3 7.3 ms (+0.3-0.5 ms select) vs a verify step of
28-32 ms (w=1), 31-37 ms (w=8), 53 ms (w=32, R=64). MTP prefill 15.6 ms eager per request.

## What remains for the served integration

* P side: run the MTP prefill in the P engine (the hook), export the head's KV blocks as a 17th attention layer
  (`pd_transfer.export_kv_blocks` over `model._paged_kv_caches + [head.kv_cache]`, same block ids) and hand off the
  last main post-norm hidden row (5120 bf16 per request) with the GDN snapshot; D imports the 17th layer's blocks.
* D side: allocate the head's KV with the engine's cache (`allocate_kv_caches` returning 17 layers or a separate
  allocation with the same block count + 1 pad block), warm compile-first (MTP step per decode bucket, selects per
  verify plan) before `warmup_model_decode` captures; per-request `hidden_in` rows on prefill/import; the verify plans
  per (bucket width, k).
* Plugin: speculative scheduling (vLLM `num_speculative_tokens` + the TT executor calling draft -> verify -> commit,
  the rejection bookkeeping = VerifyController), the R<=32 restriction if bitwise equality with plain decode is
  required (w*T <= 32), the long-prompt MTP prefill (chunked; today prompts must fit one bucket) and, optionally,
  refilling the head's KV at accepted positions with the main hidden states (vLLM does; we only write the last
  committed position) and tracing the MTP prefill per bucket.
