# Motif-3 golden CPU reference

Pure-PyTorch, device-free reference for `Motif-Technologies/Motif-3` (314B MoE: 53 layers, GDLA attention,
4-stream mHC, 384-expert top-8 MoE). Every TT module is PCC-tested against it. Nothing here imports `ttnn`.

| File | Contents |
|---|---|
| `config.py` | `MotifArgs` (all consumed `config.json` fields + hard-coded constants), `from_hf_config`, layer schedule (`is_swa_layer`, `attention_window`, `softmax_scale`, `uses_yarn`, `is_moe_layer`), `tiny_random_args()` |
| `rope.py` | YaRN / plain `inv_freq`, cos/sin tables, half-split `rotate_half`, `apply_rope` |
| `modules.py` | `RMSNorm`, `PolyNorm`, `GroupedPolyNorm`, `MHCLayer` + `sinkhorn`, `GDLAttention` (expanded + absorbed), `attention_mask`, `MLP`, `Router`, `RoutedExperts`, `MoE`, `DecoderLayer`, `MotifModel`, `MotifForCausalLM`, `MotifMTP` |
| `cache.py` | `LatentKVCache` (576 values/token/layer: `kv_norm(c)` 512 + roped `k_pe` 64), `MotifKVCache` |
| `generate.py` | `MotifGenerator`: `prefill` / `decode` / `generate` with a correct KV cache |
| `weights.py` | `MotifCheckpoint` (lazy, index-driven safetensors access, `MissingWeightsError`), HF<->reference name mapping, `load_reference_model`, `load_mtp`, `random_state_dict` |
| `golden.py` | `TensorRecorder`, `capture_layer_goldens`, `capture_model_goldens`, `export_real_goldens` (+ CLI), `save_golden` / `load_golden`, `random_streams`, `pcc` |
| `tokenizer.py` | tokenizer + chat template without `trust_remote_code` |
| `tests/` | HF parity, attention forms, decode, ops, weights/golden, real weights, no-device guard, harness hygiene (`test_harness.py`: no hub kernels in the HF oracle, README commands write no shared report) |

## API

```python
import torch
from models.demos.motif3.reference import load_reference_model, tiny_random_args, build_random_model
from models.demos.motif3.reference.generate import MotifGenerator
from models.demos.motif3.reference.golden import capture_layer_goldens, capture_model_goldens, random_streams, save_golden

# tiny random model (all layer kinds: global/dense, swa/dense, swa/moe, global/moe)
model = build_random_model(tiny_random_args(), seed=0, dtype=torch.float32)
logits = model(ids)                                    # [B, S, V] fp32; attn_mode="expanded" | "absorbed"

# real weights: exact prefix model (embed, layers 0..2, final norm, lm_head); bf16 tensors are memory-mapped
model = load_reference_model(layer_ids=(0, 1, 2), dtype=torch.bfloat16, lazy_experts=True)

# KV-cache decode that is correct past the 129-key window (do NOT use HF generate(), see below)
gen = MotifGenerator(model, batch_size=1, max_seq_len=4096, attn_mode="absorbed")
logits = gen.prefill(prompt_ids)                       # [1, S, V]
step = gen.decode(next_token)                          # [1, V]; prefill(..., user=b) for per-user prompts

# goldens: every module tap (mHC h_pre/h_post/H_res, q/k_pe/c_kv/lambda/gate, router indices/weights, ...)
x = random_streams(model.args, 1, 140, embed_weight=model.model.embed_tokens.weight)  # or real layer inputs
g = capture_layer_goldens(model.model.layers["2"], x, decode_steps=4)
g = capture_model_goldens(model, prompt_ids[:, :-4], decode_ids=prompt_ids[:, -4:])
save_golden(g, "/path/golden.pt")                      # torch.load(path, weights_only=True)
```

Real goldens for layers 0-2 (default chat prompt of 145 tokens, i.e. past the 129-key window; the last 4 tokens are
teacher-forced decode steps; one file per layer plus a model-level file):

```bash
cd tt-metal && python_env/bin/python -m models.demos.motif3.reference.golden --out /tmp/motif3_goldens --layers 3 --dtype bf16
```

Tap names are listed in the `golden.py` docstring (`x_in`, `mhc_attn.h_res`, `self_attn.c_kv`, `moe.router.indices`, ...).

## Semantics decisions

Source hierarchy: training code > Motif vLLM fork > HF `modeling_motif.py`, with the study report
(`docs/study/01_motif_reference.md`) as the op-by-op spec. The shipped HF file passes the fork's training-parity
audit (`motif_docs.md`: h_post coefficient 1.0, YaRN inv_freq, routed-only bias clamp, window + 1, fp32 `poly * up`),
so the reference reproduces HF op for op: **bit-exact in fp32 and bf16** on tiny random configs and on real layers 0-2.

* **Numerics are dtype-driven.** A bf16 model reproduces HF's bf16 cast points exactly; an fp32 model is the ideal
  golden. Two knobs each switch one part of the computation to the fork's precision (they only change rounding):
  `q_path_fp32=False` (fork/training run wq_a/q_norm/wq_b in bf16; HF uses fp32, `modeling_motif.py:647-650`) and
  `mhc_mix_fp32=True` (fork tilelang folds the mHC gamma into the projection and produces the 24 mixes in fp32, fork
  `motif.py:302-360`; HF rounds to bf16, `:238-243`). They do not reproduce the fork's numerics as a whole; the
  remaining rounding differences are listed under Known deviations #4.
* **Layer schedule**: SWA iff `layer_idx % 4 != 0` (HF `:575`, fork `motif.py:503`, training `model.py:201`); the prose in
  `motif_docs.md` ("(layer_idx + 1) % period") is stale. 14 global layers (0, 4, ..., 52), 39 SWA; dense FFN on layers 0-1.
* **Window = 129 keys including the current token**: training `window_size=(128, 0)` (`model.py:185-203`); HF passes
  `sliding_window + 1` (`:571`), which transformers 5.12.1 maps to flash-attn `(sw-1, sw-1)` when `key_len > sw`
  (`modeling_flash_attention_utils.py:642-647`, causal so the right window is moot); fork `(sw-1, 0)`. Masks are built
  from absolute positions (`attention_mask`: `k <= q` and `k > q - 129`), identical for prefill, chunked prefill and decode.
* **Softmax scale**: `192^-0.5 = 0.07216878` (SWA, MTP), `x (0.1 ln 64 + 1)^2` -> `0.14467963` (global) (HF `:579-585`,
  fork `:516-522`, training `:298-308`).
* **RoPE**: global = YaRN inv_freq (correction range [10, 23], dims >= 23 divided by 64; HF `:270-306` == training
  `precompute_freqs_cis_yarn`), SWA = plain theta 1e4; no cos/sin magnitude scaling; angles fp32, cos/sin rounded to the
  activation dtype, rotation in fp32 (`:406-416`, `:669-672`); half-split (NeoX) on the checkpoint rows as stored (training
  interleaved + de-interleave == half-split, tested).
* **GDLA**: q head h -> group/KV head `h // 5`, heads `5g..5g+3` signal, `5g+4` noise; signal `s = 4g + j` indexes
  `lambda_proj`, `wq_b_gate`, `wo` columns; `out_s = sigmoid(gate_s) * (O_5g+j - bf16(sigmoid(lambda_s)) O_5g+4)`
  (`:753-773`). Expanded form = HF (K 192 / V 128, GQA x5). Absorbed form = MQA over the latent cache with
  `q_lat = W_UK,g^T q_nope` and the differential combine done in latent space, then `W_UV,g` once per signal head. Both
  read the same cache: `kv_norm(c_raw)` *with* gamma (fork `motif.py:694`) + roped `k_pe`. Attention core: fp32 scores,
  softmax and PV, one cast of the output (the real FA2 kernel also rounds P to bf16; not reproduced on purpose).
* **mHC**: merged projection rows `[pre 4 | post 4 | res 16 (i*4+j)]` (fork/training `proj_merged` layout) applied to
  `RMSNorm_{1e-6}` of the 16384-wide stream; `sigmoid(clamp(alpha*p + b, +-10))`; `h_post` coefficient 1.0 (HF `:1090`,
  fork `:1282`); Sinkhorn 20 x (rows then columns), `exp(clamp(+-20))`, sums clamped at 1e-8, fp32 (`:226-233`, training
  `mhc.py:191-216`); `pre` and `post` (`H_res @ X + h_post x out`) in fp32 with one cast to the residual dtype
  (`:1224-1226`, `:1242-1244`).
* **PolyNorm**: row reduction over the intermediate dim, fp32, one downcast; dense and shared experts are *not*
  bias-clamped, routed experts clamp the bias to +-0.5; `x 0.5` output scale; `hidden_clamp` 1e6 kept (a no-op).
* **Router / MoE**: fp32 GEMM on upcast input and weight, sigmoid, top-8 on `scores + expert_bias`, weights = unbiased
  scores renormalized (+1e-20) x 2.0 (`:843-870`); routed outputs accumulated in fp32 in ascending expert order like HF's
  loop (`:929-944`), plus the shared expert in fp32, one cast (`:1003-1008`). Experts can be fetched lazily per expert.
* **Final head**: mean over the 4 streams, RMSNorm, untied `lm_head`, fp32 logits (`:1435-1439`, `:1667-1668`).
* **MTP** (not in HF): `input_proj(cat[h_main_postnorm, embed_norm(embed(t+1))])`, one pre-norm block with SWA "all"
  attention (window 129, plain RoPE, scale 192^-0.5) and a dense PolyNorm MLP, `final_layernorm` (fork `motif_mtp.py:106-170`).

## Known deviations from HF and why

1. **HF bug, `seq_len == num_attention_heads`**: `MotifGDLAttention` guesses the attention output layout with
   `attn_out.shape[1] == self.num_heads` (`modeling_motif.py:746`); flash-attn returns (B, S, H, D), so a prefill of
   exactly 80 tokens is silently mis-transposed. The reference has no such ambiguity
   (`test_known_hf_bug_seq_len_equal_to_num_heads`; parity tests avoid that length).
2. **HF `generate()` is not a decode reference past 128 tokens**: it builds `DynamicCache(config=...)`, which makes every
   layer (including the 14 global ones) a 127-token sliding layer. Use `MotifGenerator` (or HF with `DynamicCache()`);
   `test_hf_decode_cache_pitfall_and_reference_decode` shows both behaviours.
3. **Shared-expert width**: HF uses `moe_intermediate_size` (`:974-976`); the reference uses
   `moe_intermediate_size * num_shared_experts` like training/fork (`motif.py:1049`). Identical for Motif-3 (1 shared expert).
4. **Fork/training precision differences** (all rounding-level). Only the first two are available as knobs; none is
   reproduced by default:
   * fork bf16 q path and fp32 mHC mixes (`q_path_fp32`, `mhc_mix_fp32` above);
   * fork router GEMM in TF32: fp32 storage and accumulation, 10-bit mantissa multiply (`motif.py:912-922`, `:1196`);
   * fork mHC projection and sum-of-squares GEMM in TF32 (DeepGEMM `tf32_hc_prenorm_gemm`, `layers/mhc.py:319-326`);
   * fork folds the routing weights into GEMM2: bf16 per (token, k), then an fp32 top-k sum cast to bf16
     (`fused_moe/motif_experts.py:345-346`). The bf16 shared-expert output is then added in bf16
     (`fused_moe/runner/moe_runner.py:647`); HF and the reference add it in fp32 with one cast;
   * the attention kernels round P to bf16 before the PV MMA (see GDLA above). The fork uses the flash diff-KV backend on
     SWA layers and an MLA backend on global layers (`motif.py:617-660`); training uses FA;
   * training multiplies routing scores before W2 and scatter-adds in bf16 (`moe.py:801`, `:829`), rounds
     `sigmoid(PolyNorm weight)` to the param dtype (`moe.py:143`, `:254`) and takes `sigmoid(lambda)` in bf16
     (`attention.py:407`);
   * the fork's dense MLP has no `hidden_clamp` (a no-op at 1e6).
5. `polynorm_output_scale_per_layer` (fork) is supported; it is empty for Motif-3, so HF (which ignores it) agrees.

## Running the tests

tt-metal's root `conftest.py` imports `ttnn`, and its autouse fixtures touch the device, so always disable conftests.
`test_no_device.py` fails if `ttnn` was imported. Also pass `-o addopts=""`: the root `pytest.ini` addopts include
`--junitxml=generated/test_reports/most_recent_tests.xml`, a report that every pytest run started from the tt-metal
root writes, so concurrent runs would overwrite each other's report. `-o addopts=""` drops it, and
`--import-mode=importlib` is then passed explicitly. `tests/test_harness.py` checks that these commands write no
report:

```bash
cd /home/ttuser/hchang/experiments/motif-3/tt-metal
python_env/bin/python -m pytest -p no:cacheprovider --noconftest -o addopts="" --import-mode=importlib \
    models/demos/motif3/reference/tests
# fast subset (~25 s, no checkpoint needed):
python_env/bin/python -m pytest -p no:cacheprovider --noconftest -o addopts="" --import-mode=importlib \
    models/demos/motif3/reference/tests --ignore=models/demos/motif3/reference/tests/test_real_weights.py
```

Add `-rA` (or `-s`) to see the printed parity numbers (`exact=...`, max |diff|). If you need a report file, pass
`--junitxml=<your own path>`.

The real-weight tests (~25 s with a warm page cache; ~64 GB peak RSS, mostly the fp32 copy of layer 2's experts plus
memory-mapped bf16 shards) need embed, layers 0-2, final norm, lm_head and MTP shards under `MOTIF3_WEIGHTS_DIR`
(default `/home/ttuser/hchang/experiments/motif-3/weights/Motif-3`). They skip otherwise. The HF sources come from
`MOTIF3_HF_META_DIR` (default `.../motif-3/hf_meta`). `tests/hf_reference.py` runs them on CPU: it routes
`flash_attention_2` to an exact CPU shim and blocks `import kernels` while `modeling_motif.py` loads, so the file's
import-time `kernels.get_kernel("Motif-Technologies/activation")` can never download hub kernels or replace the torch
`RMSNorm`/`PolyNorm` (`test_harness.py` checks this as well). The root `pytest.ini` still supplies the 300 s per-test
`timeout`, which the real-weight module raises; it is an ini key, not part of addopts.
