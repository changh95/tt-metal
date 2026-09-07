# Solar-Open-100B on Tenstorrent Blackhole (P150x8, TP=8)

TTNN implementation of [upstage/Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B) (102.6B-parameter
MoE: 48 layers, hidden 4096, 64 q / 8 kv heads x 128, 128 routed experts top-8 + 1 shared expert of width 1280,
YaRN RoPE to 131072 tokens, vocab 196608) for a box of 8 Blackhole P150 cards opened as a logical 1x8 mesh with
tensor parallelism 8 (`FABRIC_1D_RING`). Derived from the MoE demo of tt-metal PR #55589: the batch-32
union-of-experts decode, the fused per-device `[gate|up]` expert weights, the dense-bmm / expert-sorted prefill MoE,
traced prefill at 128 tokens and the Blackhole user-grid placement are kept; the source model's specific code
(attention-sink logits and biases, clamped SwiGLU, sliding windows, channel-token shortcuts) is gone and Solar's
sigmoid router with selection bias plus the shared expert are added.

Status: phase-1 bring-up (correctness first). See "Recorded baselines" at the end for the measured PCC values.

Numerics (relative to the bf16 HF model): weights are bfp8 (experts, attention, lm_head, KV cache) / bf16 (embeddings,
norms, router gate, fp32 router bias); the residual stream is bf16 (`DecoderLayer._residual_add` writes each residual
sum into a new bf16 tensor instead of the branch's bfp8 output, so the stream is never block-quantised), the norm
outputs and hence the attention / router / expert inputs are bf16, the attention and MoE branch outputs are bfp8
(o_proj input and the pre-all_reduce partials are bfp8, inherited from the source demo), the router linear runs in
fp32 (HiFi4, fp32 accumulation; a bfp8 input would be widened to bf16 first), the selection math in fp32, and the
lm_head emits bf16 logits (TTSampling consumes bf16 natively). Every TP all-reduce (attention prefill and decode, MoE)
is `ttnn.all_reduce`: `MeshConfig.allreduce` (reduce_scatter_minimal_async + all_gather_async on the CCLManager's
ping-pong semaphores) is NOT used on the 1x8 path because its all-gather leaves a stale block on the last ring devices
on every other call on P150x8 (see `tt/attention/operations.py::apply_allreduce` and "Recorded baselines").

## Layout

| path | content |
|---|---|
| `tt/model.py`, `tt/layer.py` | `Model` (embedding, layers, norm, lm_head, on-device sampling) and `DecoderLayer` |
| `tt/attention/` | bias-free, sink-free GQA (`AttentionWeights(wqkv, o_proj)`), paged/unpaged KV, `SolarOpenAttentionProgramConfig` |
| `tt/topk.py` | `TopKRouter`: fp32 logits -> `moe_grouped_topk` (sigmoid, +e_score_correction_bias for selection only, top-8, normalised unbiased scores) -> dense `[T,128]` routing tensor |
| `tt/experts/` | routed experts: fused `[gate_d|up_d]` per device, SiLU GLU, union-of-experts decode, dense/sorted prefill; `SolarOpenProgramConfig` |
| `tt/shared_expert.py`, `tt/mlp.py` | shared expert (TP-sharded partial, summed into the routed partial before the single all-reduce) and the MoE block |
| `tt/model_config.py`, `tt/common.py`, `config.py` | `ModelArgs` (HF config/tokenizer, stop set, weight cache), `create_tt_model`, `MeshConfig`, `MoEOptions` |
| `demo/text_demo.py`, `demo/sample_prompts/` | generation demo; `input_data_questions_ko_en_prefill_128.json` = 16 Korean + 16 English prompts |
| `tests/` | unit tests (`tests/unit/`), real-weight layer-0 test, multi-user consistency test, `tests/accuracy/` (host-side HF bf16 reference generator + teacher-forced whole-model accuracy test) |
| `unit_test_thresholds.json` | PCC thresholds per component (key `Solar-Open-100B`) |

## Prerequisites

- tt-metal built for Blackhole with the Python environment of this checkout (`python_env`), transformers 5.12.1
  (native `model_type="solar_open"`; no `trust_remote_code` model code needed), torch CPU.
- The checkpoint (42 bf16 safetensors shards, 205 GB) downloaded to the HF hub cache. `ModelArgs` derives the
  model name from the basename of `HF_MODEL`, which must be `Solar-Open-100B`; hub snapshot directories are hashes,
  so point `HF_MODEL` at the repo id (`upstage/Solar-Open-100B`, resolved from the hub cache) or at a symlink:

  ```bash
  ln -s ~/.cache/huggingface/hub/models--upstage--Solar-Open-100B/snapshots/<hash> /path/to/models/Solar-Open-100B
  ```

- A fast disk for the ttnn weight cache (`TT_CACHE_PATH`): 103 GB per (dtype, expert dtype) variant with bfp8 experts
  (measured 2026-09-07: 531 tensorbins, 48 x 2.1 GB layers - the expert tensorbins hold all 8 TP shards, i.e. the
  whole bfp8 model - plus 1.6 GB bf16 embedding and 1.1 GB lm_head; replicated tensors are stored once, ~14 GiB per
  device once loaded). The zero-initialised KV caches are NOT part of the cache:
  `attention/kv_cache.py` allocates them directly on the device with `ttnn.from_torch(torch.zeros(...))` (DRAM, TILE,
  replicated). The `k_cache_<shape>` / `v_cache_<shape>` cache stems of contract C7 were dropped on 2026-09-07 (tech-lead
  decision): nothing ever read those tensorbins and every distinct KV configuration (paged block count or unpaged
  batch x max_seq_len) would have added 2 x 48 files holding the 8 replicas, e.g. 27 GB for the batch-32 x 8K pool
  (`[4096, 1, 64, 128]` bfp8 = 35.6 MB x 8 x 96), re-read on every warm start.
- Host RAM for the phase-1 whole-model load: `AutoModelForCausalLM.from_pretrained(dtype=bf16)` peaks at 393 GB RSS
  (measured 2026-09-07 on the cold demo run; the state dict stays alive while the ttnn cache is written) once per cache
  variant - never run two whole-model loads at once on a 503 GB box; the cache marker makes later runs skip the load
  (warm start: 78 s from disk, 8-20 s with the tensorbins in the page cache).

On the bring-up box everything is exported by one script:

```bash
source /home/eslim/experiments/solar/env.sh   # venv, TT_METAL_HOME, PYTHONPATH, HF_MODEL, TT_CACHE_PATH, MESH_DEVICE=P150x8, HF_HUB_OFFLINE=1
cd $TT_METAL_HOME
```

Random-weight unit tests only need a config: `HF_MODEL=models/demos/solar_open/configs/Solar-Open-100B` works
without any checkpoint.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `HF_MODEL` | `upstage/Solar-Open-100B` | checkpoint directory / symlink / repo id (basename must be `Solar-Open-100B`) |
| `TT_CACHE_PATH` | `<HF_MODEL dir>` if it is a directory, else `~/.cache/tenstorrent/Solar-Open-100B` | root of `tensor_cache_{bf16\|bfp8}_exp{bfp8\|bfp4}_(1, 8)/` |
| `MESH_DEVICE` | - | `P150x8` on the 8-card box |
| `SOLAR_OPEN_EXPERT_DTYPE` | `bfp8` | routed expert weight dtype: `bfp8` (12 GiB/device) or `bfp4` (6.3 GiB/device); part of the cache directory name |
| `SOLAR_OPEN_SHARED_EXPERT_DTYPE` | `bfp8` | shared expert weight dtype (`bfp8` or `bf16`); the shared output is always bf16 |
| `SOLAR_OPEN_ROUTER_IMPL` | `fused` | `fused` = `moe_grouped_topk` (3 launches); `ops` = exact pure-ttnn chain (~10 launches, fidelity fallback) |
| `SOLAR_OPEN_ROUTER_FP32_LOGITS` | `1` | `0` = bf16 router logits (fp32-accumulated; the variant Blackhole CI covers); both router impls still select in fp32 |
| `SOLAR_OPEN_REASONING_EFFORT` | `high` (demo sets `low`) | chat-template `reasoning_effort`: `low`/`minimal` prepend an empty `<\|think\|>` block so the answer starts immediately |
| `SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT` | `1` | `0` disables the template's dated provider system prompt |
| `SOLAR_OPEN_KV_BUDGET_GIB` | `16` (bfp8 experts) / `22` (bfp4) | `create_tt_model` raises when the paged KV pool would exceed this per device |
| `SOLAR_OPEN_TRACE_REGION_SIZE` | from `models/model_trace_region_sizes.yaml` (`solar-open-100b`: 100 MB) | overrides the trace region in bytes; `0` = dynamic allocation |
| `SOLAR_OPEN_FORCE_MODEL_LOAD` | unset | `1` forces the HF weight load even when the ttnn cache marker says the cache is complete |
| `SOLAR_OPEN_SORTED_MOE_DEBUG` | `0` | `1` logs the expert-sorted prefill MoE plan |
| `SOLAR_OPEN_STREAMING_LOAD` | unset | phase 2: `1` selects the per-layer streaming loader (not part of phase 1) |
| `SOLAR_OPEN_TF_REFERENCE` | `$TT_CACHE_PATH/teacher_forced_reference.pt` | reference file written by `tests/accuracy/gen_reference.py` and read by `tests/accuracy/test_teacher_forced.py` |
| `SOLAR_OPEN_TF_REPORT_DIR` | unset | `test_teacher_forced.py` writes a markdown report (`teacher_forced_<case>.md`: per-prompt metrics, HF vs TT greedy continuations) into this directory |
| `SOLAR_OPEN_NUM_DEVICES` | unset | test collection only: replaces `ttnn.get_num_devices()` in the mesh parametrizations so `pytest --collect-only` / `python -c "import ..."` never touch the devices (e.g. `SOLAR_OPEN_NUM_DEVICES=8`) |

The MoE flags are recorded in the cache marker (`.weights_complete`); a run with different flags rebuilds its own
cache directory (expert dtype) or invalidates the marker (router flags). Only weights are cached (embeddings, norms,
router, experts, shared experts, attention, lm_head; the stems of contract C9 minus the KV caches): the KV caches
are zero-allocated on the device at every start, so the cache directory is independent of the batch size, the
context length and the paged block count, and a warm start reads only the weights (103 GB with bfp8 experts). WARNING: flipping `SOLAR_OPEN_ROUTER_IMPL`,
`SOLAR_OPEN_ROUTER_FP32_LOGITS` or `SOLAR_OPEN_SHARED_EXPERT_DTYPE` therefore costs a full 205 GB host load (the
dtype-suffixed tensorbins of both variants coexist on disk, but the marker records one set of flags and is rewritten
by the cold run, so flipping back reloads again); only `SOLAR_OPEN_EXPERT_DTYPE` has its own directory. A complete
48-layer cache does satisfy a `num_layers`-limited debug build (the marker's `n_layers` only has to be >= the
requested layer count).

## Test ladder (bring-up order)

Host only (no devices):

```bash
python -c "import models.demos.solar_open.tt.model"
pytest models/demos/solar_open/tests/unit/test_model_config.py \
       models/demos/solar_open/tests/unit/test_expert_weights.py \
       models/demos/solar_open/tests/unit/test_expert_parallel_config.py
grep -rIin "gpt[_-]oss\|gptoss" models/demos/solar_open   # source-demo identifiers: expect only tt-metal issue references
```

Random weights on the 1x8 mesh (`HF_MODEL` may be the config directory). Component selection is
`--test-modules=a,b`; each case id is `<mesh>-<batch/seq>-layer_0-<paged>-<pos>`:

```bash
T=models/demos/solar_open/tests
pytest $T/unit/test_router.py -k 1x8                                     # router: T in {1,8,32,128,4096}, 4 impl/dtype combos, trace smoke
pytest $T/unit/test_shared_expert.py -k 1x8                              # shared expert partial sums
pytest $T/unit/test_modules.py -k "test_decoder and 1x8" --test-modules=rms_norm,attention   # decode cases attend over a 64-token context per user
pytest $T/unit/test_rope.py -k 1x8                                       # YaRN tables vs HF (positions > 65536), 1.0693 kernel case
pytest $T/unit/test_modules.py -k "test_decoder and 1x8 and unpaged and pos0" --test-modules=experts   # decode b1/16/32, prefill 128/1024/4096
pytest $T/unit/test_modules.py -k "test_experts_shared_expert_hook and 1x8"   # shared partial added before the single all_reduce (shift == 8c)
pytest $T/test_experts_skewed_routing.py -k 1x8                          # hot/cold expert-sorted prefill path
pytest $T/unit/test_modules.py -k "test_decoder and 1x8 and unpaged and pos0" --test-modules=router,shared_expert,mlp
pytest $T/unit/test_modules.py -k "test_decoder and 1x8" --test-modules=decoder                # unpaged + paged, pos0 + pos70000 (decode: context at 70000..70064)
pytest $T/unit/test_modules.py -k "test_model and 1x8"                   # 1-layer SolarOpenForCausalLM: prefill_b1_s128, decode_b32_s1
```

Real weights, layer 0 only (needs shards 1 and 2, or whatever `model.safetensors.index.json` lists for layer 0;
input = real embeddings of the KO/EN prompts):

```bash
pytest models/demos/solar_open/tests/test_layer0_real_weights.py -k 1x8
```

Full model (all 42 shards; the first cold run per cache variant loads 205 GB on host and writes the ttnn cache):

```bash
D=models/demos/solar_open/demo/text_demo.py
pytest $D -k "prefill_128 and 1x8 and not prefill_128_en and not prefill_128k"   # batch 1, first (Korean) prompt, greedy, reasoning_effort=low
   # -k is a substring match over the whole node id: plain "prefill_128" also selects prefill_128_en and prefill_128k,
   # and "not en" deselects EVERYTHING because the function name test_solar_op*en*_demo contains "en".
pytest $D -k "prefill_128_en and 1x8"     # batch 1, first English prompt
pytest $D -k "batch32 and 1x8"            # 32 users x 8K context: 16 Korean + 16 English prompts, traced decode + prefill@128
pytest $D -k "sampling_b1 and 1x8"        # temperature 0.8 / top_p 0.95 / top_k 32 on-device sampling
pytest $D -k "reasoning_high and 1x8"     # reasoning_effort=high with a 2048-token budget (think block + answer)
pytest $D -k "prefill_1k and 1x8"         # ... prefill_4k, prefill_8k, prefill_16k, prefill_32k, prefill_64k, seqlen-sweep
pytest models/demos/solar_open/tests/test_multi_user_consistency.py -k 1x8   # slot-independence of 32 users (teacher forced; ~1.5-4 min warm)
```

Teacher-forced accuracy against the bf16 HF model (design 5.7). The reference is generated on the host in its own
process (the whole 100B model on the CPU: ~360-370 GB RSS with transformers 5.12, so never while another whole-model
load or a device process runs), then the device test replays the reference token ids and compares per-step logits:

```bash
timeout 10800 python models/demos/solar_open/tests/accuracy/gen_reference.py       # host only; 4 KO/EN prompts x 64 greedy tokens, ~3 min, writes $TT_CACHE_PATH/teacher_forced_reference.pt
pytest models/demos/solar_open/tests/accuracy/test_teacher_forced.py -k "b1 and 1x8"    # each prompt alone (batch 1)
pytest models/demos/solar_open/tests/accuracy/test_teacher_forced.py -k "b32 and 1x8"   # the prompts tiled over 32 slots (union-of-experts decode)
```

`gen_reference.py --prompt-indices ... --num-new-tokens ... --date ...` picks other prompts of the KO/EN file / a
longer continuation; the date is pinned (the chat template stamps `strftime_now` into the system prompt) so the token
ids are reproducible, and the test asserts that `ModelArgs.encode_prompt` with the same template kwargs yields the
reference's ids before it runs.

Every demo output is logged as `ANSWER` (text after `<|content|>`, cut at `<|end|>`), `REASONING` (the `<|think|>`
block) and `TOKENS` (prompt length, generated count, stop token vs budget); per-device DRAM / trace-region usage is
logged after the model load, the prefill and the decode loop. Generation stops on `{2, 24, 25}`. `reasoning_effort=low`
is a soft hint: the template ends the prompt with an empty `<|begin|>assistant<|think|><|end|>` block, yet with greedy
decoding the model re-opened a `<|think|>` block for 25 of the 32 KO/EN prompts (all 16 English ones), so the batch32
case has a 512-token budget (the 200-token budget of the batch-1 cases truncated 23/32 users before `<|content|>`). The
bf16 HF model does the same (host check 2026-09-07: greedy first token `<|think|>` for 3/3 probed prompts, margins
0.5-4 logits over `<|content|>`), so this is the model's behaviour, not a port defect. Do not run
`pytest --collect-only` on `demo/text_demo.py` while a device test is running: its import chain (tt_transformers demo
utilities / device SKU detection) opens the devices even at collection time.

Dtype A/B: prefix any command with `SOLAR_OPEN_EXPERT_DTYPE=bfp4` (separate cache directory, 58 GB, ~9.5 min cold
build) or `SOLAR_OPEN_SHARED_EXPERT_DTYPE=bf16`; router fallbacks with `SOLAR_OPEN_ROUTER_FP32_LOGITS=0` and
`SOLAR_OPEN_ROUTER_IMPL=ops`. Measured bfp4 delta (2026-09-07, see "Recorded baselines"): 5.6 GiB less DRAM per device
but NO decode speed-up (54.1 vs 54.4 ms/step at batch 1, 90.6 vs 92.1 at batch 32 - decode is not expert-bandwidth
bound on this box), teacher-forced top-1 agreement with the bf16 model 0.906 / 0.895 (bfp8 0.930 / 0.918), KL 2x, and a
behaviour change: under `reasoning_effort=low` the bfp4 model answers directly (3 of 32 think blocks in the batch32 demo
vs 25 of 32 with bfp8; it puts `<|content|>` above `<|think|>` at the first token of all 4 reference prompts, against
2-3.75-logit bf16 margins). `test_teacher_forced.py` therefore carries a separate, looser floor set for bfp4.

## Memory per device (TP=8)

| item | bfp8 experts | bfp4 experts |
|---|---|---|
| routed experts (48 x 255 MiB) | 11.95 GiB | 6.33 GiB |
| shared experts, attention, router, lm_head, RoPE tables | 0.09 + 0.45 + 0.05 + 0.13 + 0.125 GiB | same |
| embedding (bf16, replicated) | 1.5 GiB | 1.5 GiB |
| fixed total | 14.3 GiB | 8.7 GiB |
| paged KV (bfp8, 13,056 B/token) | 32 x 8K = 3.19 GiB, 32 x 16K = 6.38 GiB, 32 x 32K = 12.75 GiB, 1 x 128K = 1.59 GiB | |

Phase-1 defaults: batch 32 x 8K context; the KV budget guard (`SOLAR_OPEN_KV_BUDGET_GIB`) refuses larger pools
until the DRAM headroom has been measured.

## Known limitations (phase 1)

- Batch <= 32 users (single mesh row); row-sharded multi-row meshes are not part of this tree.
- Single-user prefill is capped at 64K tokens on the 1x8 mesh (the 128K cases skip); `max_position_embeddings`
  131072 is supported by the RoPE tables.
- On-device sampling `top_k <= 32` (`TTSampling.max_top_k`); the demo clamps larger requests with a log line.
- The demo defaults to `reasoning_effort=low`; `high` needs `max_generated_tokens >= 2048` (case `reasoning_high`).
- The trace region size (100 MB, `models/model_trace_region_sizes.yaml`) is inherited from the source demo's
  bh_loudbox entry; measured usage is 23.6 MiB (prefill@128 trace + batch-1 decode trace) / 25.6 MiB (batch-32 decode)
  of the 93.1 MiB region, so it suffices with ~3.6x headroom.
- Whole-model host load on every cold cache build (streaming loader is phase 2).
- The router prebuilds its per-token-count helper tensors (the `[T, 128]` bias copy the fused top-k op needs and
  the bf16 zeros the scatter starts from) for `T` in {decode batch, 32, 128} and keeps them for every `T <= 128`
  it meets; longer prefill lengths rebuild them per call (two device ops, freed after the scatter) and must not be
  traced (only 128 is a traced prefill length on P150x8).
- `create_tt_model` refuses KV pools above `SOLAR_OPEN_KV_BUDGET_GIB` (paged and unpaged alike). Measured on the
  full model (2026-09-07): 14.44 GiB of weights per device, 17.57 GiB with the batch-32 x 8K pool (3.19 GiB), 14.15 GiB
  still free after a 512-step batch-32 decode, i.e. room for 32 x 32K (12.75 GiB) with ~1.4 GiB spare; the default
  16 GiB budget stays until the activation peak of a 64K prefill has been measured.
- The expert-sorted prefill MoE path passes TILE tables to `ttnn.embedding` (two untilize copies per 1024-token
  split) and caches a `[split, split]` bf16 identity per layer (2 MiB at 1024; ~120 MiB per device over 48 layers
  after a long prefill); a module-level ROW_MAJOR identity is a phase-2 cleanup.
- Decode batch sizes must map onto one core per user on a single rectangle of at most 8x8 cores for
  `nlp_concat_heads_decode` (any batch <= 8, multiples of 8, powers of two up to 32); `Model.__init__` raises for
  others (11, 13, 17, 19, 22, 23, 26, 29, 31) before any weight is loaded.

## Recorded baselines (fill in during bring-up, design D13)

2026-09-07, review fixes (all rows below were re-measured afterwards): (1) the residual stream, the embeddings and the
lm_head logits are bf16 (they were bfp8 -- the source demo's in-place residual add into the bfp8 branch output and
`ttnn.embedding(..., dtype=bfloat8_b)`), so every component test now feeds bf16 activations (the production dtype;
before, attention / rms_norm / mlp / decoder / the real-weight test fed bfp8); (2) the decode cases of the attention
and decoder components (random and real weights) attend over a 64-token context per user written into the KV cache
through the TT prefill path (before, position 0 on an empty cache made the decode softmax span one key, so decode RoPE,
SDPA, the scale and the cache update were untested -- the decode attention PCC dropped from 0.99966 to 0.9991 once they
were measured), and the `pos70000` decode cases run (RoPE 70000..70064, cache slots 0..64); (3) every all-reduced
output is asserted to be bit-identical on all 8 devices. That assertion found a pre-existing device bug: attention
prefill's `MeshConfig.allreduce` (`reduce_scatter_minimal_async` + `all_gather_async` on the CCLManager's ping-pong
semaphores) left a stale 1/8-of-the-rows block on the last ring devices on every other call (128 tokens: devices 5-7,
1024 tokens: device 7, |err| up to 30 on a sum of 8 replicas; `reduce_scatter_minimal_async` alone exact in 12/12 runs,
`all_gather_async` alone stale in 1-3 of 6, `ttnn.all_reduce` exact and replica-consistent in 12/12). Device 0 was
always correct, so the device-0 PCC of every earlier stage never saw it; the corrupted replica fed that device's
router / experts and showed up as run-to-run spread of the decoder prefill-128 PCC (0.9914-0.9950 on identical
inputs, MoE output differing in exactly one 32-row block). Fix: attention prefill uses `ttnn.all_reduce` like decode
and the MoE (`tt/attention/operations.py::apply_allreduce`).

Thresholds in `unit_test_thresholds.json` start from the design table (router 0.99, experts 0.92, shared expert
0.98, mlp 0.88, decoder 0.90 decode / 0.86 prefill, attention 0.95, rms_norm 0.99) and are re-baselined ~0.02
below the first measured values. Re-baselined so far: experts 0.92 -> 0.97 (2026-09-07; measured 0.9989 decode /
0.9994 prefill with the default bfp8 experts, the skewed-routing test already asserted 0.97, and ~0.983 was the
value this op chain had demonstrated before; `SOLAR_OPEN_EXPERT_DTYPE=bfp4` is not covered by the table); mlp 0.88 ->
0.97 (2026-09-07; measured 0.9981 decode / 0.9985-0.9988 prefill with bfp8 inputs, i.e. the experts' own floor: with
fp32 selection the router's near-tie flips cost ~0.015 per flipped token at most, so 0.97 keeps one flip of headroom
even at T=1); decoder 0.90 decode / 0.86 prefill -> 0.97 / 0.97 (2026-09-07; measured 0.9990-0.9993 decode,
0.9949-0.9960 prefill, paged and unpaged, pos 0 and 70000). Router note: the fused `moe_grouped_topk` kernel's internal sigmoid is off by up
to ~8e-4, so tokens whose 8th-vs-9th biased score margin is below 1e-3 may pick a different expert set (with the
real, near-zero layer-0 bias ~27 % of tokens are that close; ~15 % of those flip). `test_router` therefore asserts
the dense PCC on the agreeing tokens (0.99999 measured) at every T and the all-token dense PCC (>= 0.99) only from
T=128 on; a whole-tensor router PCC at T <= 32 can drop below 0.99 with 3+ near-tie flips (each costs ~0.004 at
T=32). The `test_decoder --test-modules=router` component uses the same split, plus: 100 % agreement on the
decisive tokens (margin >= 1e-3) and at most `max(3, 5 %)` mismatching tokens overall (the floor of 3 covers the
decode batch, where the expected 0.6 near-tie flips arrive as 2 in about one seed out of eight; measured 2/32). Its
reference routes exactly the operands the device gets -- bf16 activations and bf16-exact random weights (the
checkpoint is bf16): with an fp32 reference input / fp32 random gate weight the logits differ by up to 9e-3 / 1e-2
(mean 1.7e-3 each), more than the decisive margin, and a legitimate near-tie flip showed up as a "decisive" one
(measured 2026-09-07; with matching operands the device fp32 linear is within 2.1e-3 max / 3.2e-4 mean of torch and
0 of 20 probe trials at T = 16..4096 flipped a decisive token). Real-weight thresholds live in
`tests/test_layer0_real_weights.py` (`REAL_WEIGHT_THRESHOLDS`). Its router metric is the same split (100 % agreement
on the decisive tokens, dense PCC on the agreeing tokens at every T and on all tokens from T=128, reference routed on
the bf16 copy of the input the device gets); the raw set-agreement cap was re-baselined 0.98 -> 0.95 (with the floor of
3 mismatching tokens) on 2026-09-07: real layer 0 has a gate rms of 0.041 and an MoE input rms of 0.16, so its router
logits have std ~0.5 and ~16 % of the tokens are near ties (margin < 1e-3); measured 3/128 and 9/1024 flips, every one
with margin <= 1.7e-4 (0 decisive), of which the HF rule applied to the device's own fp32 logits (linear error max
1.4e-3 / mean 1e-4 vs torch on identical bf16 operands) already produces 1/128 and 8/1024 -- the rest is the fused
kernel's sigmoid. Routing the reference on the fp32 input instead of the bf16 copy changes 2/128 and 6/1024 reference
decisions by itself (input rounding moves the logits by up to 4.5e-3). The real-weight mlp / decoder thresholds were
re-baselined 0.95 -> 0.97 the same day (measured mlp 0.9990-0.9997, decoder 0.9940-0.9997; see the table).

`tests/test_multi_user_consistency.py` (32 users x 24 teacher-forced steps; per-(prompt, step) logit PCC and top-1
agreement of every prompt across slot layouts, judged against a same-slot repeat) was re-baselined on 2026-09-07 from
Solar's own noise floor. The same-slot repeat (A1 vs A2) measured logit PCC min 0.99997 / mean 1.00000 and top-1
agreement 768/768 in two runs, and every layout change was indistinguishable from it: rotation by 5 slots min 0.99997 /
mean 1.00000, 768/768; prompt 0 alone in slot 7 among 31 fillers min 0.99999, 24/24; the 31 fillers (same prompt,
different slots) against each other min 1.00000, 720/720; the filler same-slot repeat min 1.00000, 744/744 -- 0 of
3024 (prompt, step) pairs disagree, i.e. the user grid, the page table and the union-of-experts mask isolate the users
and the `ttnn.all_reduce` CCL chain is run-to-run reproducible to ~1e-5 in PCC (the greedy continuations are also
identical across processes: the demo's batch32 answers start with the same 24 tokens). The source model's floors
(mean PCC 0.98, top-1 0.85, per-step 0.7, relative drops 0.01 / 0.08) were written against its reduce_scatter +
all_gather chain, which measured 0.993-1.000 / 0.95-1.00 here. Floors now (D13, ~0.02 below measured): mean logit PCC
0.98 (kept), per-step PCC 0.97 (from 0.7; one near-tie expert flip of headroom), top-1 agreement 0.98 (from 0.85),
checked as a disagreement count with a floor of one pair so the 24-pair lone-prompt comparison tolerates a single
near tie, relative drops 0.01 (kept) / 0.02 (from 0.08). A real cross-user leak collapses the per-step PCC below 0.3
(the source model's 13-wide-grid placement bug), so the margin to the failure mode the test guards stays >= 0.6.

`tests/accuracy/test_teacher_forced.py` (design 5.7) measures the whole model against the bf16 HF `SolarOpenForCausalLM`
on the CPU without ever holding both in one process: `tests/accuracy/gen_reference.py` (host only, ~370 GB RSS with
transformers 5.12, 35 s load + 30-35 s per prompt) writes the chat-templated token ids (fixed date), the greedy 64-token
continuation and the per-step logits (full bf16 + top-64) of 4 KO/EN prompts (file indices 0, 15, 22, 16); the device
test replays those ids (prefill, then decode teacher-forced with the reference tokens) and compares every step's logits.
Measured 2026-09-07 with bfp8 experts, b1 (each prompt alone) / b32 (prompts tiled over the 32 slots): top-1 agreement
0.9297 / 0.9180 over 256 steps (0.978 / 0.960 on the 226 "decisive" steps whose reference margin is >= 0.5 logit; every
flipped step takes the reference's 2nd choice and 13 of 18 / 12 of 21 flips are bf16 near ties < 0.5 logit, the
largest flipped margin is 1.25), top-5 overlap 0.928 / 0.923, top-64 logit PCC 0.982 / 0.982, full-vocab PCC 0.992 /
0.992, KL(ref || TT) 0.035 / 0.032; the per-step PCC does not decay with the position (last-8-step PCC >= first-8 for 3
of 4 prompts) and the prefill step's top-1 is right for 3 of 4 prompts (the miss is the KO capital prompt's 0.5-logit
`<|think|>` / `<|content|>` tie); two b1 runs gave bit-identical metrics; the 8 slot copies of each prompt at b32 agree
in 1792/1792 top-1 tokens (logit PCC >= 0.99995). Floors (D13, ~0.02 below the lower case): aggregate top-1 0.90,
per-prompt top-1 0.85, decisive top-1 0.94, top-5 0.90, top-64 PCC 0.96, full-vocab PCC 0.97, mean KL <= 0.06,
slot-copy PCC 0.97. With `SOLAR_OPEN_EXPERT_DTYPE=bfp4` the same runs measured top-1 0.9062 / 0.8945, decisive 0.9425 /
0.9425, top-5 0.898 / 0.891, top-64 PCC 0.971 / 0.972, full-vocab PCC 0.987 / 0.988, KL 0.069 / 0.068 (max 2.87: the
first token flips to `<|content|>` on every prompt) and fail the bfp8 floors, so the test keys its floors by the expert
dtype (`THRESHOLDS["bfp4"]`: 0.87 / 0.84 / 0.92 / 0.87 / 0.95 / 0.96 / KL <= 0.10).

| test | case | measured | date |
|---|---|---|---|
| test_router (1x8, random weights std 0.04, real layer-0 bias) | fused-fp32 T=1/8/32/128/4096 | expert-set agreement 1/1, 7/8, 32/32, 125/128, 3920/4096 (all flips on near-tie tokens, margin < 1e-3; 27.5 % of tokens are near ties with the real bias); dense PCC on agreeing tokens 0.99999 at every T, all tokens 0.9836 (T=8), 0.9969 (T=128), 0.9943 (T=4096); weights max abs err 8.2e-4; row-sum max err 4.4e-3 | 2026-09-07 |
| test_router | fused-bf16 (fp32-accumulated bf16 logits) | 1/1, 7/8, 32/32, 125/128 agreement; dense PCC on agreeing tokens 0.99999; all tokens 0.9836 (T=8), 0.9969 (T=128) | 2026-09-07 |
| test_router | ops-fp32 T=1/8/32/128/4096 | 0 flips at T <= 128, 11/4096 flips (margin < 1e-3, fp32 logits max abs err 3e-3); all-token dense PCC 0.99999 (T <= 128), 0.9996 (T=4096) | 2026-09-07 |
| test_router | ops-bf16 T=32/128/4096 | 2/32, 1/128, 116/4096 near-tie flips (margin < 5e-3); all-token dense PCC 0.9918, 0.9990, 0.9963 | 2026-09-07 |
| test_router_bias_is_selection_only / test_router_trace_replay | strong bias T=128; trace T=32/128 | 126/128 agreement, weights match the UNBIASED scores to 6.2e-4 (3.7e-2 from the biased alternative); trace replays bit-identical to eager, all-token dense PCC 0.9959 / 0.9949 | 2026-09-07 |
| test_rope (1x8) | 32 cases: tables vs HF at 233 positions to 131071, device lookup, 1.0693 kernel case, batch 32 | YaRN cos/sin tables bit-exact vs HF (max diff 0 at every checked position, attention factor 1.0693147 == expected); on-device embedding lookup cos/sin PCC 0.999998-1.0 (max err 3.9e-3, bf16) at positions 0..131071 incl. 65534-65538; rotary_embedding_llama with the 1.0693-scaled tables: 32/32 batch positions, min PCC 0.999993; table PCC vs HF 0.999998 (seq 128..131072, batch 1 and 32); 7 cases skip (4x8 mesh only) | 2026-09-07 |
| test_decoder --test-modules=attention (1x8, bf16 input; decode over a 64-token context per user, prefill causal) | decode b1/b32/b16, prefill 128/1024/4096; unpaged + paged; pos0 + pos70000 | PCC decode 0.99919 / 0.99912 / 0.99912 (pos0), 0.99923 / 0.99910 / 0.99912 (pos70000); prefill 0.99921 / 0.99871 / 0.99846 (pos0), 0.99920 / 0.99870 / 0.99846 (pos70000); paged bit-identical to unpaged in every case; all 8 replicas identical (assert_replicated) after the ttnn.all_reduce fix; threshold 0.95 (stage-3 values with bfp8 input and an empty decode cache were 0.99965 decode / 0.99864-0.99741 prefill) | 2026-09-07 |
| test_decoder --test-modules=rms_norm (1x8, bf16 input) | decode b1/b32/b16, prefill 128/1024/4096 | PCC 0.99997-1.00000 (decode), 0.99996-0.99997 (prefill); threshold 0.99 | 2026-09-07 |
| test_shared_expert (1x8, sum of 8 device partials vs SolarOpenMLP) | bfp8 T=1/32/128/1024 | PCC 0.9979 / 0.9980 / 0.9997 / 0.9997; in-place bf16 -> bfp8 add accepted (PCC >= 0.99 vs the bf16 partial) | 2026-09-07 |
| test_shared_expert | bf16 T=1/32/128/1024 (HiFi4, fp32 accumulation since the review fix) | PCC 0.99999 / 0.99998 / 0.99999 / 0.99999 (was 0.9979 / 0.9977 / 0.9997 / 0.9997 with HiFi2 + bf16 accumulation, i.e. the bfp8 floor: HiFi2 truncated the second bf16 operand); threshold 0.995 | 2026-09-07 |
| test_decoder --test-modules=experts (1x8, uniform random top-8 routing, bfp8 experts, bf16 input) | decode b1/b32/b16, prefill 128/1024/4096 | PCC 0.99887 / 0.99892 / 0.99890 (re-run after the review fixes; stage-4 values 0.99888 / 0.99889 / 0.99890) (decode: single-user and union-of-experts sparse_matmul paths, gate_up Nt=10 on 5x2, down Nt=128 on 8x4 / 8x8), 0.99938 / 0.99937 / 0.99937 (prefill); 128 tokens on the dense bmm, 1024 and every 1024-split of 4096 on the expert-sorted path (plan cap 96, hot 0); per-device DRAM after the call: first call +0.34-0.39 MiB (decode), +0.62 MiB (128), +4.7 MiB (1024), +10.9 MiB (4096) = kernel binaries of the newly cached programs (17 -> 64 entries at 4096, 8.9 MiB, all returned by clearing the program cache) + the 2 MiB `[1024, 1024]` bf16 identity of the sorted path; a second identical call grows by exactly 0 bytes (no per-expert slice persists, design X5) | 2026-09-07 |
| test_experts_shared_expert_hook (1x8) | decode b1/b32, prefill 128/1024 | zeros stub: output bit-identical (PCC 1.0, max diff 0); constant stub c = 1: all-reduced output shifts by 8.103 / 8.104 / 8.146 / 8.145 (expected tp * c = 8.0, max abs err 0.31 within the 10 % bfp8 tolerance) -> the shared partial is added before the single all_reduce on every path | 2026-09-07 |
| test_experts_skewed_routing (1x8, experts 3/7/11 hot at 60/25/12 % of the tokens) | prefill 1024/4096 | PCC 0.99932 / 0.99945; plan cap 96, hot 3 on every 1024-split (~640 / 320 / 180 routed tokens per split for the hot experts) | 2026-09-07 |
| test_decoder --test-modules=router (1x8, bf16 activations, bf16-exact random gate std 0.02, bias randint(-5,5)*2^-9; fused fp32 router) | decode b1/b32/b16, prefill 128/1024/4096 (+ pos70000, identical inputs) | expert-set mismatches 0/1, 1/32, 0/16, 1/128, 16/1024, 63/4096 (all near ties, margin < 5e-4; 0 decisive flips; decisive tokens 1/1, 23/32, 15/16, 110/128, 927/1024, 3684/4096); dense PCC all tokens 0.99999 / 0.99613 / 0.99999 / 0.99900 / 0.99810 / 0.99811, on agreeing tokens 0.99999 everywhere; row sums within 1e-2; dense tensor identical on all 8 devices (stage-5 run with a different RNG order: 2/32 mismatches, all-token PCC 0.99249 at T=32) | 2026-09-07 |
| test_decoder --test-modules=shared_expert (1x8, bf16 input, 8 device partials summed vs SolarOpenMLP 1280) | decode b1/b32/b16, prefill 128/1024/4096 | PCC 0.99799 / 0.99813 / 0.99818 (decode), 0.99965 / 0.99965 / 0.99974 (prefill); threshold 0.98 | 2026-09-07 |
| test_decoder --test-modules=mlp (1x8, bf16 input (the production dtype; stage 5 fed bfp8), router + routed experts + shared expert vs SolarOpenMoE) | decode b1/b32/b16, prefill 128/1024/4096 (+ pos70000) | PCC 0.99857 / 0.99779 / 0.99806 (decode), 0.99920 / 0.99925 / 0.99926 (prefill); exactly one CCL per call on every path (`{'all_reduce': 1}` counted over every ttnn / ttnn.experimental collective entry point: single-user and batched decode, dense-bmm 128, sorted 1024, 4096 = one 4096-token chunk); output identical on all 8 devices; threshold 0.88 -> 0.97 (stage 5, bfp8 input: 0.99811 / 0.99808 / 0.99816 decode, 0.99876 / 0.99848 / 0.99866 prefill) | 2026-09-07 |
| test_decoder --test-modules=decoder (1x8, bf16 input, full layer vs SolarOpenDecoderLayer; decode over a 64-token context per user) | decode b1/b32/b16, prefill 128/1024/4096; unpaged + paged; pos0 + pos70000 (decode and prefill) | PCC decode 0.99325 / 0.99637 / 0.99644 (pos0), 0.99309 / 0.99613 / 0.99641 (pos70000); prefill 0.99618 / 0.99561 / 0.99658 (pos0), 0.99625 / 0.99568 / 0.99657 (pos70000, YaRN beyond 65536); paged bit-identical to unpaged in every case, output bf16 and identical on all 8 devices; CCLs per layer call `{'all_reduce': 2}` (attention + MoE) in decode AND prefill since the all-reduce fix. The decode PCC is lower than stage 5's 0.9990-0.9993 because the attention branch (std ~2.3 with these random weights, PCC 0.9991 over the 65-key context) now dominates the residual sum instead of the vacuous Wo*V at position 0; before the CCL fix the prefill-128 value spread over 0.9914-0.9950 between runs. Thresholds 0.97 / 0.97 kept (>= 0.023 headroom at the decode-b1 floor 0.9931) | 2026-09-07 |
| test_decoder 1x1 (device 0 only, TP=1: DRAM-interleaved QKV, no CCL) | decode_b1 pos0, unpaged + paged, --test-modules=attention,decoder | attention PCC 0.99917 (64-token context), decoder PCC 0.99355; decoder CCL ops `{}` | 2026-09-07 |
| test_model (1x8, 1 random-init layer of `SolarOpenForCausalLM` at vocab 196608, bfp8 experts, bf16 residual / logits, unpaged KV sized by max_seq_len=128) | prefill_b1_s128, decode_b32_s1 | logits PCC 0.99189 (prefill, all 128 positions), 0.99906 (decode, 32 users); stage 6 with bfp8 embeddings / residual / logits: 0.98961 / 0.99844; embedding -> layer -> norm -> pow2-padded lm_head (per-device 32768 of 262144 columns, devices 6-7 all padding) -> host concat and truncation to 196608; RoPE placement + concat-grid checks and on-device sampling init (vocab 196608, 8 splits) at construction; threshold 0.95 (hard-coded in `test_model`) | 2026-09-07 |
| test_layer0_real_weights (1x8, real layer-0 weights from shards 1-2 + real embeddings of the chat-templated KO/EN prompts (tokenizer present); bfp8 experts, fused fp32 router; bf16 inputs; decode over a 64-token context of real embeddings per user; reference routed on the bf16 copy of the input) | decode b1/b32, prefill 128/1024; unpaged + paged | router: expert-set agreement 1/1, 31/32 (1 flip, cap 3), 126/128 (2 flips, cap 6), 1018/1024 (6 flips, cap 51), 0 decisive flips anywhere (max margin of a flipped token 3.5e-4; near-tie fraction 0 / 15.6 / 11.7 / 8.4 %), dense PCC on agreeing tokens 0.999998-0.999999, on all tokens 0.999999 / 0.99700 / 0.99892 / 0.99947, union of experts 8 / 78 / 108 / 121 of 128; mlp PCC 0.99979 / 0.99965 / 0.99987 / 0.99987; decoder PCC 0.99848 / 0.99885 / 0.99997 / 0.99982; paged bit-identical to unpaged, every output identical on all 8 devices; real layer-0 bias min -0.00198 / max 0.00201 with 10 distinct values, gate rms 0.0407; thresholds: router set agreement 0.95 (floor 3 tokens), dense PCC 0.99, mlp / decoder 0.97 (stage 7 with bfp8 inputs, random token ids and the stale-replica CCL: mlp 0.99897-0.99972, decoder 0.99404-0.99972) | 2026-09-07 |
| test_multi_user_consistency (1x8; 32 users = 16 KO + 16 EN prompts, chat template, 8K paged context (4096 x 64-token blocks), on-device greedy 24-token continuations teacher-forced; runs A1/A2 same slots, B rotated by 5 slots, C1/C2 prompt 0 alone in slot 7 among 31 `Write one sentence about the weather.` fillers) | b32, per-(prompt, step) logit PCC / top-1 agreement vs the same-slot repeat | PASSED twice (230 s / 74 s pytest wall, warm cache; identical numbers): same-slot A1 vs A2 PCC min 0.99997 / mean 1.00000, top-1 768/768; rotated min 0.99997 / mean 1.00000, 768/768; lone prompt min 0.99999 / mean 1.00000, 24/24; filler repeat C1 vs C2 min 1.00000, 744/744; fillers vs each other min 1.00000, 720/720 -> no cross-user contamination; greedy first tokens 7 x `<\|content\|>` / 25 x `<\|think\|>` (same 25/32 think rate as the demo), all 32 continuations coherent, KO in Korean, identical run to run and to the demo's batch32 outputs; greedy decode 83.0 / 82.8 ms/step over steps 2-23 (386 tok/s; the demo's iterations 2-22 average 83.0-83.2 ms before reaching its 92-93 ms plateau by iteration ~25); ASIC temps 39-46 C -> 56-64 C (board 1 hottest); thresholds re-baselined mean PCC 0.98 / per-step 0.97 / top-1 0.98 (count-based, floor 1 pair) / drops 0.01 / 0.02 (see text) | 2026-09-07 |
| text_demo cold start (`prefill_128`, full checkpoint, empty `TT_CACHE_PATH`) | HF load + ttnn cache build | PASSED; HF `from_pretrained(bf16)` + Meta permute 67 s (shards in the page cache; 215 s from a cold page cache), ttnn cache build 578 s (48 layers, ~12 s each), model + KV ready 662 s, pytest wall 697 s; peak host RSS 393 GB; cache 103 GB / 531 tensorbins (no KV files); v4 marker written | 2026-09-07 |
| text_demo warm start (`prefill_128_en`, `batch32`, ...) | cache-only startup | "Warm ttnn weight cache detected"; model + KV ready 78.5 s reading the 103 GB from disk, 7.5-20 s with the tensorbins in the page cache; prefill trace re-capture 2-7 s per process | 2026-09-07 |
| text_demo `prefill_128` / `prefill_128_en` / `sampling_b1` / `reasoning_high` (batch 1, 4K context, 78-79-token prompts, traced prefill@128) | TTFT, decode ms/step, tok/s/user, DRAM, trace | TTFT 186 / 188 / 192 / 187 ms; decode 54.4 / 54.8 / 54.4 / 54.1 ms/step (per-step 54-58 ms) = 18.2-18.5 tok/s/user (design projection 47-62 ms with bfp8 experts; gpt-oss-120b 36 ms); per-device DRAM 14.436 GiB after load, 14.450 GiB after decode (b1 4K KV pool 0.05 GiB); trace region 23.6 of 93.1 MiB. Answers: KO capital -> "대한민국의 수도는 서울입니다. ..." (31 tokens, stop token, no think block); EN moons -> correct, but a 100-token `<\|think\|>` block first despite `low` (151 tokens, stop token); sampling (T 0.8 / top_p 0.95 / top_k 32) -> correct KO answer, text differs from greedy; `high` -> 5-sentence think block + "대한민국의 수도는 **서울**입니다." (56 tokens, stop token) | 2026-09-07 |
| text_demo `batch32` (32 users = 16 KO + 16 EN, 8K context = 4096 page blocks, greedy, `low`, 512-token budget) | TTFT, decode ms/step, tok/s, DRAM headroom, answer quality | PASSED (78 s wall); TTFT 184 ms per user (32 sequential traced prefills); decode 92.1 ms/step (per-step 63-97 ms) = 10.85 tok/s/user, 347 tok/s aggregate (design projection 64-91 ms / 350-500 tok/s with bfp8 experts; gpt-oss-120b ~48 ms); per-device DRAM 17.572 GiB after load (14.436 weights + 3.19 KV), 17.589 GiB after 512 decode steps, 14.149 GiB free; trace 25.6 MiB; ASIC temps 43-51 C -> 57-65 C (board 1 hottest). Outputs: 32/32 coherent and on-topic, Korean answers in Korean, no garbage; 25/32 opened a think block under `low`, 28/32 reached `<\|content\|>`, 19/32 stopped on a stop token within 512 tokens, 9 long-form answers cut by the budget, 4 still thinking at 512 (one quasi-loop: user 22 "largest desert" cycles "We can also mention ... / Thus answer."); the 200-token run's outputs are exact prefixes of the 512-token run's (deterministic). Same user-0 prompt decodes differently at b1 and b32 after the first clause (batched numerics; both correct) | 2026-09-07 |
| text_demo `prefill_1k` (928-token Frankenstein excerpt, eager prefill at 1024) | TTFT | 520 ms (design 0.5-0.8 s); first-compile pass 29.4 s; decode 54.9 ms/step; DRAM +108 MiB after the 1K prefill (sorted-MoE identities, program binaries) | 2026-09-07 |
| text_demo `prefill_8k` (input_data_long_8k.json = 7541 tokens, padded to 8192, eager prefill in 1024-token MoE splits, 8K context) | TTFT | 4047 ms (design 2.8-5.3 s); first-compile pass 48.4 s; decode 56.1 ms/step at a 7.5K context (56-61); DRAM 14.485 GiB after load (1-user 8K pool 0.10 GiB), 14.604 GiB after the 8K prefill (+119 MiB), largest free block 16.98 GiB; ASIC temps 47-56 C -> 58-67 C | 2026-09-07 |
| test_teacher_forced (1x8; 4 KO/EN prompts (0 KO capital, 15 KO Olympics, 22 EN desert, 16 EN Australia) x 64 reference-greedy tokens, `reasoning_effort=low`, fixed date 2026-09-07, 78-81-token prompts; eager prefill + traced decode; vs `SolarOpenForCausalLM` bf16 CPU eager) | b1 (4K context) | PASSED twice, bit-identical metrics (185 s / 53 s pytest warm): top-1 0.9297 (238/256; per prompt 0.875 / 0.938 / 0.969 / 0.938), decisive-step top-1 0.9779 (221/226), top-5 overlap 0.9281, top-64 PCC mean 0.98177 (per-step min 0.808 at an exact-tie step), full-vocab PCC mean 0.99215 (min 0.889), KL mean 0.0347 / max 0.444 (the 0.5-logit first token of the KO capital prompt); all 18 flips = reference rank 1-2, margins 0-1.25; TT greedy diverges from HF at steps 0 / 4 / 43 / 23 = the first teacher-forced flip; teacher-forced decode 59.8-63.1 ms/step with full-logit readback; ASIC temps 38-43 C -> 59-69 C | 2026-09-07 |
| test_teacher_forced | b32 (8K context, prompt p in slots p, p+4, ...) | PASSED (52 s pytest warm): top-1 0.9180 (235/256; per prompt 0.906 / 0.922 / 0.938 / 0.906), decisive-step top-1 0.9602 (217/226), top-5 0.9227, top-64 PCC 0.98222, full-vocab PCC 0.99219, KL 0.0324 / max 0.435; 21 flips = reference rank 1-2, margins 0-0.875; slot copies vs first slot: logit PCC min 0.99995, 1792/1792 identical top-1 and greedy tokens; teacher-forced decode 74.0 ms/step; temps 47-55 -> 56-65 C | 2026-09-07 |
| test_teacher_forced with `SOLAR_OPEN_EXPERT_DTYPE=bfp4` (same reference) | b1 / b32 | FAIL against the bfp8 floors, PASS against `THRESHOLDS["bfp4"]`: top-1 0.9062 (232/256; per prompt 0.859 / 0.906 / 0.922 / 0.938) / 0.8945 (229/256), decisive-step top-1 0.9425 / 0.9425, top-5 0.8977 / 0.8906, top-64 PCC 0.97061 / 0.97154 (per-step min 0.60 / 0.64), full-vocab PCC 0.98747 / 0.98755, KL 0.0694 / 0.0681 (max 2.87 at step 0 of the desert prompt: bfp4 `<\|content\|>` 37.25 vs `<\|think\|>` 34.25 where bf16 has 34.5 vs 38.25); all 4 first tokens flip to `<\|content\|>`, the later steps agree 90-94 %; bfp8-vs-bfp4 device logits PCC 0.990, top-1 0.934, KL 0.04-0.06; slot copies 0.99994 / 1792/1792; decode 60.3 / 74.3 ms/step (no gain) | 2026-09-07 |
| text_demo with `SOLAR_OPEN_EXPERT_DTYPE=bfp4`: cold `prefill_128`, warm `batch32` | cache build, perf, behaviour | cold: HF load 28 s + ttnn cache build ~515 s (model + KV ready 549 s, 559 s pytest), peak process RSS 362 GB, cache 58 GB / 531 tensorbins; DRAM 8.811 GiB after load (bfp8 14.436; -5.63 GiB = the routed experts), 11.947 GiB with the 32 x 8K KV pool (19.79 GiB free); b1: TTFT 175 ms, decode 54.11 ms/step (bfp8 186 / 54.4), 33-token direct answer "대한민국의 수도는 **서울특별시**입니다. ..."; b32: TTFT 172 ms, decode 90.64 ms/step = 353 tok/s (bfp8 92.1 / 347), 3 of 32 think blocks (bfp8 25), 32/32 reach `<\|content\|>`, 30/32 stop within 512 tokens (bfp8 19), mean 155 generated tokens, all answers coherent and correct incl. the desert prompt ("Antarctic Desert ... 14.2 million square kilometers"); temps up to 75.8 C (board 1) after the cold build | 2026-09-07 |
