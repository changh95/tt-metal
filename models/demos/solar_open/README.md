# Solar-Open-100B on Tenstorrent Blackhole (P150x8, TP=8)

TTNN implementation of [upstage/Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B) (102.6B-parameter
MoE: 48 layers, hidden 4096, 64 q / 8 kv heads x 128, 128 routed experts top-8 + 1 shared expert of width 1280,
YaRN RoPE to 131072 tokens, vocab 196608) for a box of 8 Blackhole P150 cards opened as a logical 1x8 mesh with
tensor parallelism 8 (`FABRIC_1D_RING`). Derived from the MoE demo of tt-metal PR #55589: the batch-32
union-of-experts decode, the fused per-device `[gate|up]` expert weights, the dense-bmm / expert-sorted prefill MoE,
traced prefill at 128 tokens and the Blackhole user-grid placement are kept; the source model's specific code
(attention-sink logits and biases, clamped SwiGLU, sliding windows, channel-token shortcuts) is gone and Solar's
sigmoid router with selection bias plus the shared expert are added.

Status: phase 2 complete (2026-09-07): perf levers, streaming loader, KV budgets / long-context cases, vLLM wrapper
(untested), all default-on changes re-validated on the final tree. See "Recorded baselines" at the end: the phase-1 vs
phase-2 summary table first, then the per-test rows with the measured PCC / step-time values.

Numerics (relative to the bf16 HF model): weights are bfp8 (experts, attention, lm_head, KV cache) / bf16 (embeddings,
norms, router gate, fp32 router bias); the residual stream is bf16 (`DecoderLayer._residual_add` writes each residual
sum into a new bf16 tensor instead of the branch's bfp8 output, so the stream is never block-quantised), the norm
outputs and hence the attention / router / expert inputs are bf16, the attention and MoE branch outputs are bfp8
(o_proj input and the pre-all_reduce partials are bfp8, inherited from the source demo; the phase-2 option
`SOLAR_OPEN_ATTENTION_BF16_OUTPUT=1` keeps the attention branch bf16 through o_proj and its all-reduce -- measured
WORSE on the teacher-forced test (top-1 0.9336 -> 0.8828, KL 0.0355 -> 0.0654, root cause open), so it stays off, see
"Environment variables" and "Recorded baselines"), the router linear runs in
fp32 (HiFi4, fp32 accumulation; a bfp8 input would be widened to bf16 first), the selection math in fp32, and the
lm_head emits bf16 logits (TTSampling consumes bf16 natively). The shared expert runs bf16 activations with HiFi4
in its own module; with `SOLAR_OPEN_FUSE_SHARED_EXPERT=1` it runs as slot 128 of the routed sparse_matmuls, i.e. with
bfp8 activations and the routed compute config (one bfp8 re-quantisation of its partial disappears, its GLU
intermediate becomes bfp8), so the fused output is PCC- but not bit-equivalent to the unfused one. With the default
`SOLAR_OPEN_INDEXED_DECODE=1` a single-user decode step runs the routed experts in the sparse_matmul indexed/gather mode
(only the 8 selected experts, compact outputs; the routing weights multiply the bfp8 GLU rows before the down projection
instead of the bfp8 down outputs after it, `tt/experts/decode.py::INDEXED_WEIGHTS_ON_DOWN_INPUT`), which is PCC- but not
bit-equivalent to the phase-1 scan path (one bfp8 ulp on the expert output; even with the scan path's mul order the 8-slot
`fast_reduce_nc` differs from the 128-slot `ttnn.sum` by half an ulp on ~18 % of the elements, so no bit-neutral indexed
variant exists). Component PCCs are unchanged; the teacher-forced b1 metrics move inside the bfp8 rounding-noise band
(top-1 0.9453 -> 0.9336, decisive 0.9779 -> 0.9558, KL 0.0316 -> 0.0355; the output-side mul lands at 0.9258 / 0.9646 /
0.0346 -- see the perf-p2 row). Every TP all-reduce (attention prefill and decode, MoE)
is `ttnn.all_reduce`: `MeshConfig.allreduce` (reduce_scatter_minimal_async + all_gather_async on the CCLManager's
ping-pong semaphores) is NOT used on the 1x8 path because its all-gather leaves a stale block on the last ring devices
on every other call on P150x8 (see `tt/attention/operations.py::apply_allreduce` and "Recorded baselines"). The
phase-2 program configs (perf-p0) keep or improve the operand precision: `ttnn.matmul` / `ttnn.linear` run a bf16 x
bfp8 product at HiFi2 when they pick the config themselves but fall back to LoFi as soon as a `program_config` or
`core_grid` is passed without a `compute_kernel_config`, so every wired config in this tree passes an explicit compute
config (HiFi2, no approximations, packer L1 accumulation, bf16 destination; HiFi4 / fp32 accumulation for the
router). Measured against a torch fp32 reference of the device-rounded operands (`tests/perf/test_config_candidates.py`,
auto -> wired): router bit-identical, decode qkv 0.999936 -> 0.999862 (the fp32-destination variant reaches 0.999994 on
the op but measured slightly worse on the whole model, see the gate-p0 row; `decode_qkv_fp32_dest_acc=True` is its A/B
switch), shared
expert gate 0.99927 -> 0.99973, width-sharded decode norm 0.999963 -> 0.999994 (T = 32), dense prefill down 0.999959 ->
0.999953 (the bfp8 floor); the component and teacher-forced values of the merged tree are in "Recorded baselines".

## Layout

| path | content |
|---|---|
| `tt/model.py`, `tt/layer.py` | `Model` (embedding, layers, norm, lm_head, on-device sampling) and `DecoderLayer` |
| `tt/attention/` | bias-free, sink-free GQA (`AttentionWeights(wqkv, o_proj)`), paged/unpaged KV, `SolarOpenAttentionProgramConfig` |
| `tt/topk.py` | `TopKRouter`: fp32 logits -> `moe_grouped_topk` (sigmoid, +e_score_correction_bias for selection only, top-8, normalised unbiased scores) -> dense `[T,128]` routing tensor |
| `tt/experts/` | routed experts: fused `[gate_d|up_d]` per device, SiLU GLU, union-of-experts decode, dense/sorted prefill; `SolarOpenProgramConfig` |
| `tt/shared_expert.py`, `tt/mlp.py` | shared expert (TP-sharded partial, summed into the routed partial before the single all-reduce) and the MoE block |
| `tt/model_config.py`, `tt/common.py`, `config.py` | `ModelArgs` (HF config/tokenizer, stop set, weight cache), `create_tt_model`, `MeshConfig`, `MoEOptions` |
| `demo/text_demo.py`, `demo/sample_prompts/` | generation demo; `input_data_questions_ko_en_prefill_128.json` = 16 Korean + 16 English prompts; `input_data_ko_en_long_ctx_{16k,32k}.json` = 32 distinct clips (2-15K / 4-31K tokens) of the cached Frankenstein text, each followed by a KO/EN question about it (`"context_question": true`) |
| `tt/vllm_support.py`, `vllm_plugins/solar_open_parsers.py` | vllm-free helpers of the vLLM wrapper `SolarOpenForCausalLM` (`models/tt_transformers/tt/generator_vllm.py`): request validation, 48-layer KV spec, token capacity from the KV budget, `from_torch` pool allocator, stop ids, template kwargs, parser registration; the plugin file registers Upstage's reasoning / tool parsers (see "Serving with vLLM") |
| `tests/` | unit tests (`tests/unit/`), real-weight layer-0 test, multi-user consistency test, `tests/accuracy/` (host-side HF bf16 reference generator + teacher-forced whole-model accuracy test) |
| `tests/test_multi_user_regression.py`, `tests/sweep/` | ISL/OSL x batch multi-user regression sweep (tt-inference-server benchmark pairs x batch 1-32, section "ISL/OSL x batch sweep"); `sweep/run_sweep.sh` = one pytest process per batch with timeouts, thermal gates, board reset and a ledger; `sweep/report.py` = jsonl -> Markdown tables and batch x ISL/OSL matrices |
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
| `SOLAR_OPEN_FUSE_SHARED_EXPERT` | `0` | `1` runs the shared expert as always-on slot 128 of the routed expert tensors (built on device at load time from the cached routed + shared shards; cache-neutral, not in the marker) instead of the separate `SharedExpert` module: 5 launches per layer fewer (3 linears, the GLU mul and the partial add). See "Recorded baselines" (phase-2 fusion row) for the measured equivalence / speed before flipping it |
| `SOLAR_OPEN_INDEXED_DECODE` | `1` | `0` disables the phase-2 indexed/gather single-user expert path (profile lever 1): with `1` a decode step with ONE user routes through `TopKRouter.route_indexed` (top-8 ids + weights, no dense `[1, 128]` scatter) and `experts/decode.py::_decode_forward_indexed` (`ttnn.sparse_matmul(indices=...)` for gate\|up and down: only the 8 selected experts are visited, compact `[1, 8, 1, *]` outputs, no 16 MB zero-filled down output, no bfp8 transpose glue; routing weights on the compact GLU rows). Batched steps (2..32 users) and prefill are unaffected. Needs the fused router, EP=1 and the unfused shared expert (otherwise the scan path runs, logged at debug level). Cache-neutral (not in the marker). Measured: b1 decode 52.8 -> 35.8 ms/step (see "Recorded baselines", perf-p2 row) |
| `SOLAR_OPEN_REASONING_EFFORT` | `high` (demo sets `low`) | chat-template `reasoning_effort`: `low`/`minimal` prepend an empty `<\|think\|>` block so the answer starts immediately |
| `SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT` | `1` | `0` disables the template's dated provider system prompt |
| `SOLAR_OPEN_KV_BUDGET_GIB` | `8` (bfp8 experts) / `14` (bfp4) | per-device KV budget: `create_tt_model` raises (the demo skips the case) when the KV pool would exceed it. The defaults admit 32 x 16K with bfp8 experts and 32 x 32K with bfp4; `13` admits the bfp8 32 x 32K pool (12.75 GiB, 4.5 GiB left for activations). Values above the hard cap `14.5` / `20` (DRAM - weights - 2.8 GiB activation reserve) are clamped with a warning |
| `SOLAR_OPEN_ATTENTION_BF16_OUTPUT` | `0` | `1` keeps the attention branch output bf16 through o_proj and the TP all-reduce (skips the two bfp8 typecasts: the o_proj input in prefill, the pre-all_reduce partial in decode; the o_proj matmul then runs HiFi2 instead of LoFi). Weights and cache unchanged. MEASURED WORSE (2026-09-07, see "Recorded baselines"): teacher-forced b1 top-1 0.9336 -> 0.8828, KL 0.0355 -> 0.0654, b32 0.9375 -> 0.8906 / 0.0630 -- both FAIL the floors -- and slower (47.4 / 74.1 vs 45.3 / 69.8 ms/step). A/B and diagnosis switch only; keep 0 |
| `SOLAR_OPEN_TRACE_REGION_SIZE` | from `models/model_trace_region_sizes.yaml` (`solar-open-100b`: 100 MB) | overrides the trace region in bytes; `0` = dynamic allocation |
| `SOLAR_OPEN_FORCE_MODEL_LOAD` | unset | `1` forces the HF weight load even when the ttnn cache marker says the cache is complete |
| `SOLAR_OPEN_SORTED_MOE_DEBUG` | `0` | `1` logs the expert-sorted prefill MoE plan |
| `SOLAR_OPEN_STREAMING_LOAD` | unset | `1` selects the phase-2 streaming loader (`utils/streaming_loader.py::LazyStateDict`) for cold cache builds: tensors are read per access from the safetensors shards (peak host RSS ~ one layer's transients instead of 393 GB), same contract-C1 keys, bit-identical tensors (see "Streaming loader" below); unset = phase-1 whole-model `from_pretrained` |
| `SOLAR_OPEN_STREAMING_THREADS` | `4` | streaming loader: `preadv` threads assembling the fused expert tensors (7.5 GB/s from the page cache with 4, 2.1 GB/s with 1) |
| `SOLAR_OPEN_STREAMING_DONTNEED` | `0` | streaming loader: `1` = `posix_fadvise(DONTNEED)` on every finished layer's byte ranges so the page cache never holds the whole 205 GB checkpoint (off: a following whole-model load / test run reuses the cached pages) |
| `SOLAR_OPEN_TF_REFERENCE` | `$TT_CACHE_PATH/teacher_forced_reference.pt` | reference file written by `tests/accuracy/gen_reference.py` and read by `tests/accuracy/test_teacher_forced.py` |
| `SOLAR_OPEN_TF_REPORT_DIR` | unset | `test_teacher_forced.py` writes a markdown report (`teacher_forced_<case>.md`: per-prompt metrics, HF vs TT greedy continuations) into this directory |
| `SOLAR_OPEN_NUM_DEVICES` | unset | test collection only: replaces `ttnn.get_num_devices()` in the mesh parametrizations so `pytest --collect-only` / `python -c "import ..."` never touch the devices (e.g. `SOLAR_OPEN_NUM_DEVICES=8`) |

The phase-2 program-config levers (perf-p0, all default ON and cache-neutral) have no environment variable; each one is
an A/B switch in its module: `tt/rms_norm.py::DECODE_NORM_GRID = None` (or `RMSNorm(sharded_decode=False)`) restores the
single-core decode norm, `tt/shared_expert.py::SHARED_EXPERT_CONFIG_MAX_ROWS = 0` (or `SharedExpert(program_configs=False)`)
the auto shared-expert linears, `tt/topk.py::_LINEAR_CONFIG_MAX_TOKENS = 0` the auto router linear,
`SolarOpenAttentionProgramConfig(decode_qkv_cores=None)` the auto decode qkv matmul (`decode_qkv_fp32_dest_acc=True`
adds fp32 destination accumulation to the (8,5) config: closer to fp32 on the op, measured slightly worse on the whole
model), `SolarOpenProgramConfig(dense_down_cores=None)` the auto dense prefill
down; the expert-path levers of perf-p1 are module constants in `tt/experts/decode.py` / `prefill.py`
(`ROUTING_WEIGHTS_ON_DOWN_INPUT`, `HOT_EXPERTS_PER_EXPERT_LINEAR`, `ELIDE_ROUTING_COPIES`) and the indexed path has
`SOLAR_OPEN_INDEXED_DECODE`. Measured effect of every switch: "Recorded baselines" (perf rows).

The MoE flags are recorded in the cache marker (`.weights_complete`); a run with different flags rebuilds its own
cache directory (expert dtype) or invalidates the marker (router flags). Only weights are cached (embeddings, norms,
router, experts, shared experts, attention, lm_head; the stems of contract C9 minus the KV caches): the KV caches
are zero-allocated on the device at every start, so the cache directory is independent of the batch size, the
context length and the paged block count, and a warm start reads only the weights (103 GB with bfp8 experts). WARNING: flipping `SOLAR_OPEN_ROUTER_IMPL`,
`SOLAR_OPEN_ROUTER_FP32_LOGITS` or `SOLAR_OPEN_SHARED_EXPERT_DTYPE` therefore costs a full 205 GB host load (the
dtype-suffixed tensorbins of both variants coexist on disk, but the marker records one set of flags and is rewritten
by the cold run, so flipping back reloads again); only `SOLAR_OPEN_EXPERT_DTYPE` has its own directory. A complete
48-layer cache does satisfy a `num_layers`-limited debug build (the marker's `n_layers` only has to be >= the
requested layer count). `SOLAR_OPEN_FUSE_SHARED_EXPERT` is the exception: it changes no cache file and is not recorded
in the marker (the fused tensors are assembled on device from the cached routed and shared shards at every start).

## Test ladder (bring-up order)

Host only (no devices):

```bash
python -c "import models.demos.solar_open.tt.model"
pytest models/demos/solar_open/tests/unit/test_model_config.py \
       models/demos/solar_open/tests/unit/test_expert_weights.py \
       models/demos/solar_open/tests/unit/test_expert_parallel_config.py
pytest models/demos/solar_open/tests/unit/test_streaming_loader.py     # phase-2 loader; Part B needs the real shards of layer 0 (~30 s, < 4 GB RSS)
grep -rIin "gpt[_-]oss\|gptoss" models/demos/solar_open   # source-demo identifiers: expect only tt-metal issue references
```

Random weights on the 1x8 mesh (`HF_MODEL` may be the config directory). Component selection is
`--test-modules=a,b`; each case id is `<mesh>-<batch/seq>-layer_0-<paged>-<pos>`:

```bash
T=models/demos/solar_open/tests
pytest $T/unit/test_router.py -k 1x8                                     # router: T in {1,8,32,128,4096}, 4 impl/dtype combos, trace smoke, route_indexed (top-k form)
pytest $T/unit/test_shared_expert.py -k 1x8                              # shared expert partial sums
pytest $T/unit/test_modules.py -k "test_decoder and 1x8" --test-modules=rms_norm,attention   # decode cases attend over a 64-token context per user
pytest $T/unit/test_rope.py -k 1x8                                       # YaRN tables vs HF (positions > 65536), 1.0693 kernel case
pytest $T/unit/test_modules.py -k "test_decoder and 1x8 and unpaged and pos0" --test-modules=experts   # decode b1/16/32, prefill 128/1024/4096
pytest $T/unit/test_modules.py -k "test_experts_shared_expert_hook and 1x8"   # shared partial added before the single all_reduce (shift == 8c)
pytest $T/test_experts_skewed_routing.py -k 1x8                          # hot/cold expert-sorted prefill path (per-expert hot linears)
SOLAR_OPEN_NUM_DEVICES=8 pytest $T/unit/test_p1_layout.py                 # host: sorted-MoE cost model, hot-group configs, layout switch defaults
pytest $T/perf/test_layout_candidates.py -k 1x1                          # one device: layout levers (routing mul on the down input, hot group forms, copy elision) vs torch fp32
SOLAR_OPEN_NUM_DEVICES=8 pytest $T/unit/test_p2_indexed.py                # host: indexed single-user path (MoEOptions.indexed_decode rule, IndexedRouting contract, decode_forward guards, source contracts)
pytest $T/perf/test_indexed_candidates.py -k 1x1                         # one device: production decode_forward scan vs indexed vs torch fp32 (numerics, traced timing, weight-layout variants, routing-weight placement)
pytest $T/unit/test_modules.py -k "test_decoder and 1x8 and unpaged and pos0" --test-modules=router,shared_expert,mlp
SOLAR_OPEN_INDEXED_DECODE=0 pytest $T/unit/test_modules.py -k "test_decoder and 1x8 and pos0 and decode_b1" --test-modules=experts,mlp,decoder   # phase-1 scan path (A/B reference)
pytest $T/unit/test_modules.py -k "test_decoder and 1x8" --test-modules=decoder                # unpaged + paged, pos0 + pos70000 (decode: context at 70000..70064)
pytest $T/unit/test_modules.py -k "test_decoder and 1x8 and pos70000" --test-modules=attention,rms_norm,experts,router,shared_expert,mlp,decoder   # every component at RoPE 70000..70064 (YaRN beyond 65536)
pytest $T/unit/test_modules.py -k "test_model and 1x8"                   # 1-layer SolarOpenForCausalLM: prefill_b1_s128, decode_b32_s1, decode_b1_s1 (indexed expert path)
SOLAR_OPEN_NUM_DEVICES=8 pytest $T/unit/test_fused_shared_expert.py -k host   # host: fusion concat sequence, guards, router seed, torch emulation
SOLAR_OPEN_NUM_DEVICES=8 pytest $T/unit/test_p0_program_configs.py             # host: perf-p0 program-config builders + their ON defaults (shared expert, router, sharded decode norm gate, qkv HiFi2/fp32-dst, dense down)
pytest $T/unit/test_fused_shared_expert.py -k 1x8                        # fused (129-slot) vs unfused MLP: weights bit-identical, PCC >= 0.998, linears 4 -> 1 (decode / prefill_128) and unfused - 3 + one always-on linear per sorted split (1K / 4K)
SOLAR_OPEN_FUSE_SHARED_EXPERT=1 pytest $T/unit/test_modules.py -k "test_decoder and 1x8 and unpaged and pos0" --test-modules=router,mlp,decoder
```

Perf tests (phase 2; not correctness tests, `tests/perf/`). The first two are the fast A/B tools of every perf change
(`scratchpad/phase2/perf_log.md` holds the lever-by-lever history), the last two are profiling helpers that SKIP unless
`SOLAR_OPEN_PERF_PROFILE=1` and are meant to run under the tracy device profiler:

```bash
pytest $T/perf/test_layer0_device_perf.py -k 1x8         # real layer 0 (needs the layer-0 shards): traced replay ms per layer for decode b1 / b32, eager prefill 128 / 1024; x48 = the demo step within ~2 %; SOLAR_OPEN_PERF_OUT=<json> records it. Final phase-2 tree: b1 0.376-0.379 / b32 1.30-1.33 ms traced (phase 1 1.152 / 1.911), prefill_128 eager 3.94-3.97 (4.5)
pytest $T/perf/test_config_candidates.py -k 1x1          # one device, random weights at the TP=8 shapes: candidate matmul configs vs the auto config and a torch fp32 reference (PCC, max abs diff, wall)
pytest $T/perf/test_layout_candidates.py -k 1x1          # one device: perf-p1 layout levers (routing mul placement, hot-group forms, copy elision) vs torch fp32
pytest $T/perf/test_indexed_candidates.py -k 1x1         # one device: perf-p2 scan vs indexed expert block vs torch fp32, traced timing
SOLAR_OPEN_PERF_PROFILE=1 SOLAR_OPEN_PERF_OUT=/tmp/mb.json python -m tracy -r -p -v -m pytest $T/perf/test_expert_microbench.py -k 1x1   # kernel micro-benchmarks of the expert / attention / norm / router shapes per program config (signposted; never emits a 1-core sparse grid)
SOLAR_OPEN_PERF_PROFILE=1 SOLAR_OPEN_PERF_OUT=/tmp/fm.json python -m tracy -r -p -v --op-support-count 40000 -m pytest $T/perf/test_full_model_device_perf.py -k b1   # whole-model decode steps with a signpost per step: per-op device time vs dispatch gaps (also -k b32)
```

Real weights, layer 0 only (needs shards 1 and 2, or whatever `model.safetensors.index.json` lists for layer 0;
input = real embeddings of the KO/EN prompts):

```bash
pytest models/demos/solar_open/tests/test_layer0_real_weights.py -k 1x8
SOLAR_OPEN_STREAMING_LOAD=1 pytest models/demos/solar_open/tests/test_layer0_real_weights.py -k 1x8   # same, layer 0 read through the streaming loader
SOLAR_OPEN_FUSE_SHARED_EXPERT=1 pytest models/demos/solar_open/tests/test_layer0_real_weights.py -k 1x8   # same, shared expert fused as slot 128
timeout 1800 pytest models/demos/solar_open/tests/test_streaming_loader_device.py -k 1x8   # 2-layer streamed build into a temp cache: tensorbins byte-identical to the phase-1 cache, RSS < 40 GB
```

Full model (all 42 shards; the first cold run per cache variant loads 205 GB on host and writes the ttnn cache):

```bash
D=models/demos/solar_open/demo/text_demo.py
pytest $D -k "prefill_128 and 1x8 and not prefill_128_en and not prefill_128k"   # batch 1, first (Korean) prompt, greedy, reasoning_effort=low
   # -k is a substring match over the whole node id: plain "prefill_128" also selects prefill_128_en and prefill_128k,
   # and "not en" deselects EVERYTHING because the function name test_solar_op*en*_demo contains "en".
pytest $D -k "prefill_128_en and 1x8"     # batch 1, first English prompt
pytest $D -k "batch32 and 1x8 and not 16k and not 32k"   # 32 users x 8K context: 16 Korean + 16 English prompts, traced decode + prefill@128
pytest $D -k "batch32_16k and 1x8"        # 32 users x 2-15K-token Frankenstein clips + KO/EN questions, 16K pool (6.38 GiB, default budget), 256 fixed decode steps
SOLAR_OPEN_KV_BUDGET_GIB=13 pytest $D -k "batch32_32k and 1x8"   # 32 x 4-31K tokens, 32K pool (12.75 GiB): SKIPS without the env or SOLAR_OPEN_EXPERT_DTYPE=bfp4
pytest $D -k "prefill_64k and 1x8"        # single user, 64K prefill (1024 blocks); records the 64K TTFT and the transient DRAM peak
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

Thermal gate for the long cases (`batch32_16k`, `batch32_32k`, `prefill_64k`, `seqlen-sweep`): the devices run at full
compute for 3-8 minutes (32 sequential 2-31K-token prefills, or one 64K prefill). Read `tt-smi -s` before starting and
wait until every ASIC is below 60 C for a perf measurement (80 C is the bare functional gate; board 1 is the hottest:
65-76 C after earlier long runs), read it again between the compile pass and the timed prefill, and never start one
right after a cold cache build (75.8 C measured). Measured 2026-09-07 on the final tree: `batch32_16k` started at
66-75 C plateaued at 123.8 ms/step with the step oscillating 94-131 ms once board 1 passed ~80 C (throttling); the same
binary started below 58 C ran 79.7 ms/step for ~200 iterations (TTFT 4562 vs 6263 ms/user) and only its last 50 steps
rose to 98-110 ms as the board reached 84 C. The 8K `batch32` case is not thermally limited (cool / warm runs identical). The demo
logs a reminder for these cases. Record per case: the KV guard line, DRAM after load / prefill / decode (largest free
block), TTFT per user, decode ms/step at iterations 2-22 and at the plateau, tok/s, temperatures before/after and the
32 answers (they must be about the excerpt the user read, in the question's language).

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

## ISL/OSL x batch sweep (multi-user regression)

`tests/test_multi_user_regression.py` runs the tt-inference-server benchmark grid (`BENCHMARK_ISL_OSL_PAIRS`: ISL/OSL
128/128, 128/1024, 1024/128, 2048/128, 4096/128, 8192/128, 8192/1024, 16384/128, 32768/128) at batch 1, 2, 4, 8, 16 and
32 on the 1x8 mesh, one pytest case per batch: the model is built once for that batch with a paged pool of
`min(64K, SOLAR_OPEN_REGRESSION_KV_TOKENS // B)` tokens per user (default 512K tokens in total = 6.4 GiB of bfp8 KV per
device at 13,056 B/token, inside the 8 GiB default budget: 64K per user up to batch 8, 32K at batch 16, 16K at batch
32), and every pair with ISL + OSL inside that context runs the way the benchmark client does it -- all users prefilled
(sequentially, as the single-row mesh does), then exactly OSL decode steps without EOS stopping: 51 cells
(9 + 9 + 9 + 9 + 8 + 7; the cells above a batch's context are reported as `-`). Prompts: ISL 128 = the 32 KO/EN QA
prompts (cycled for a batch above 32), the other lengths the tt_transformers Gutenberg files
`input_data_long_{1k,2k,4k,8k,16k,32k}.json` (one prompt per file, the same excerpt for every user; 953 / 1714 / 3810 /
7541 / 16131 / 30263 tokens with Solar's tokenizer, i.e. padded prefill lengths 1024 .. 32768, the QA prompts 78-84
tokens -> 128). Every prompt goes through the chat template (`reasoning_effort=low` as in the demo; the long files
contribute their context alone as the user message, like the demo's `prefill_1k`..`prefill_32k` cases). Every batch
of the sweep maps onto the Blackhole decode user grids of `tt/attention/config.py` (batch <= 8 and multiples of 32 on
the 8x8 grid, 16 on the 13x10 device grid with an 8x2 concat grid); the decode batch equals `max_local_batch_size`.

Per cell the jsonl row records the encoded / padded ISL, the eager compile time of the prefill length, the prefill
total and per user, the TTFT of the first / mean / last user (sequential per-user prefill: user k waits k x the
per-user prefill), the first traced decode step, the decode step mean / p50 / p99, tok/s per user and aggregate, the
e2e time, the board temperatures around the prefill and the min AI clock (with the thermal gate on), the git revision,
the tag, the page-table seed, and the checks. Gates (any failure marks the cell `FAIL`; the batch case fails at the
end, after every pair has run and been recorded): every token id inside the vocabulary; the first generated token of
every user in {`<|think|>` 22, `<|content|>` 23} (the template ends the prompt with `<|begin|>assistant`, so garbage
prefill logits fail here even when the decoded garbage is diverse); at most max(1, B/4) users flagged by the
degeneracy heuristic (a run of > 48 identical tokens or < 15 % distinct tokens in the stop-truncated generation);
ISL 128: the answer keyword (`QA_KEYWORDS`, Korean prompts also accept the English answer) anywhere in the
stop-truncated generation -- think block plus answer, the reasoning usually states the answer -- for >= 50 % of the
users, recorded as `qa_accuracy`, with `qa_answer_accuracy` (keyword inside the `<|content|>` answer) and
`content_reached` (users that got past their think block within OSL; think-only outputs at OSL 128 are the expected
miss, not garbage) as information. The long prompts additionally record how many users agree with user 0 over the
first 16 tokens (isolation signal, reported not asserted: greedy decode is not bit-reproducible on device).

Mechanics inherited from the GPT-OSS sweep on this box: `warmup_prefill=False` everywhere with an explicit eager
pre-compile of every prefill length BEFORE the first trace is captured (programs compiled while a trace is live were
placed in a trace's freed range and overwritten by a later replay: a garbage prefill or a hang some pairs later), one
untimed warm pass per padded length (the first pair's warm pass also captures the prefill@128 and decode traces), a
seeded page table (`SOLAR_OPEN_REGRESSION_PAGE_TABLE_SEED`, default 1234, the demo's random block permutation from that
seed), and a thermal gate (`SOLAR_OPEN_REGRESSION_COOLDOWN_C`, wait up to `_COOLDOWN_TIMEOUT_S`) before each timed
prefill and before the first decode step of a pair: a traced decode step launched while board 1 is throttled
(> 84-87 C, AI clock 1350 -> 800 MHz) has deadlocked paged SDPA decode on this box. The gate returns as soon as every
board is below the limit and none is clock-throttled; with the mesh open the fabric routers and dispatch cores spin at
full clock (64-87 W per board at idle), so board 1 settles at ~85 C and a limit below that is only met while the box
is still warming up -- the wait therefore also returns once the boards have stopped cooling (no 0.5 C drop over ~6
tt-smi samples, ~35 s) provided the min AI clock is back at `SOLAR_OPEN_REGRESSION_FULL_AICLK_MHZ` (1350), and runs to
the timeout otherwise (measured 2026-09-08 with the 78 C / 120 s setting: 33-75 s per gate; board 1 at 1025-1250 MHz
right after a 16K / 32K prefill and back at 1350 MHz before the decode started). The harness builds the model
through `tt/common.py::create_tt_model` (the demo's path) rather than importing `demo/text_demo.py`, whose import
chain opens the devices at collection time, so it can be collected on the host (`SOLAR_OPEN_NUM_DEVICES=8`) while a
device run is active. `SOLAR_OPEN_REGRESSION_DECODE_TRACE=0` runs decode eagerly (debug).

```bash
source env.sh                                                  # python_env, TT_METAL_HOME, HF_MODEL, TT_CACHE_PATH, MESH_DEVICE=P150x8
T=models/demos/solar_open/tests/test_multi_user_regression.py
SOLAR_OPEN_NUM_DEVICES=8 pytest $T --collect-only -q           # host: the 6 node ids test_multi_user_regression[blackhole-1x8-batch{1,2,4,8,16,32}]
SOLAR_OPEN_REGRESSION_COOLDOWN_C=78 SOLAR_OPEN_REGRESSION_TIMEOUT_S=3600 timeout 4000 pytest "$T::test_multi_user_regression[blackhole-1x8-batch32]" --timeout-method thread   # the marker overrides --timeout, hence the env
SOLAR_OPEN_REGRESSION_PAIRS="128:128,8192:1024" pytest "$T::test_multi_user_regression[blackhole-1x8-batch1]"   # subset of pairs
models/demos/solar_open/tests/sweep/run_sweep.sh               # all six batches, one pytest process each
BATCHES="16 32" TAG=_rerun LOG_DIR=/tmp/solar_sweep models/demos/solar_open/tests/sweep/run_sweep.sh
python models/demos/solar_open/tests/sweep/report.py --matrix  # jsonl -> per-batch tables + batch x ISL/OSL matrices (step ms, TTFT, tok/s, status)
```

Select cases by their exact node id (`-k batch1` also matches `batch16`). `tests/sweep/run_sweep.sh` runs one pytest
process per batch under `timeout` (`TMO`, default 3600 s per batch; the batch-32 8192/1024 cell alone is 32 sequential
8K prefills, ~100 s, plus 1024 steps at ~65 ms), with a cool-start gate (`START_BELOW_C`, 60 C), the in-run thermal gate
(`COOLDOWN_C` 78 / `COOLDOWN_TIMEOUT_S` 120), a per-batch log and a ledger line per attempt (`$LOG_DIR/runs.txt`), and
after any failure kills leftover python processes holding `/dev/tenstorrent/*`, runs `tt-smi -r`, waits for < 60 C and
retries a timed-out batch once (an assertion failure is recorded and the sweep moves on to the next batch; the failing
cell stays in the jsonl with its sample output). Results: `generated/solar_open_multi_user_regression/Solar-Open-100B_1x8<tag>.jsonl`
(`SOLAR_OPEN_REGRESSION_TAG` / `TAG`), one row per cell, the schema of the GPT-OSS sweep (`report.py` renders both);
`report.py` keeps the latest row per (batch, ISL, OSL), so re-running one batch replaces its cells, and `--out` writes
the Markdown (the driver writes `$LOG_DIR/REPORT<tag>.md` at the end). Measured wall for the 51 cells on a warm cache
(2026-09-08): 96 min of pytest time (batch 1 / 2 / 4 / 8 / 16 / 32 = 11.2 / 10.4 / 15.4 / 21.1 / 18.8 / 18.9 min, model
build from the warm cache included) plus the cool-start gates, 1 h 49 min end to end. Device rules of this box: one device process at a time; never `pytest --collect-only demo/text_demo.py`
while a run is active; if a run hangs past its timeout, kill the leftover python holding `/dev/tenstorrent/*`,
`tt-smi -r`, wait for < 60 C and rerun that batch once.

### Recorded sweep (2026-09-08, 156304d63b5, tag `_p2`)

`generated/solar_open_multi_user_regression/Solar-Open-100B_1x8_p2.jsonl` (51 rows, `logs/REPORT_p2.md`, ledger
`logs/runs.txt`, per-batch logs `logs/sweep_b<N>_a1.log`): **51 / 51 cells ok** on the first attempt of every batch (no
timeout, no reset); QA keyword accuracy 1.00 in all twelve ISL-128 cells (the answer is stated inside the think block;
`qa_answer_accuracy` 0.88-1.0 at OSL 1024, where 15/16 and 31/32 users reached `<|content|>`, 0-0.06 at OSL 128 where
almost every user is still thinking), 0 first-token failures, 0 degenerate users. Skipped by the context budget: B16
32768/128, B32 16384/128 and 32768/128. Page-table seed 1234, bfp8 experts, `reasoning_effort=low`, traced decode.

The batch x ISL/OSL matrices (decode step ms, TTFT mean / last user, aggregate tok/s) are recorded once, under "Recorded
baselines" -> "ISL/OSL x batch sweep (2026-09-08, tag `_p2`)" at the end of this file; the full report (`SWEEP_REPORT.md`:
setup, per-batch tables with every column, matrices, skipped cells, harness changes, per-cell output checks and per-cell
temperatures / AI clocks) is generated from the jsonl, on this box under `/home/eslim/experiments/solar/results/sweep/`
next to a copy of the jsonl.

Reading the numbers: (1) the per-user prefill is flat over the batch for ISL <= 4K (128 tokens: 150-164 ms/user at
every batch; 8K: 3.9 s/user at B1-4) and thermal beyond that -- board 1 throttles during a sustained prefill
(min AI clock after the 16K / 32K prefills 1025-1250 MHz, 88-90 C), so the 32K prefill costs 14.1 s/user at B1 but
16.0 / 23.8 / 25.6 s/user at B2 / 4 / 8, and the decode that follows a long prefill is slower than the same context at
a smaller batch (8192/1024 at B32: 73 ms/step mean, p99 102 ms); those cells are limited by this box's cooling, not by
the model. (2) The ISL-128 cells decode slower than the long-prompt cells of the same batch (B8: 42.6 vs 31.4 ms/step,
B32: 60.6 vs 34.9): the 32 QA prompts are distinct, so a step touches more distinct experts than 32 users generating
the same Frankenstein continuation -- the QA cells are the realistic multi-user decode numbers and the long-prompt cells
(one prompt for every user) understate the MoE cost; B32 128/1024 at 60.9 ms/step matches the demo's b32 plateau (63
ms). (3) The long prompts' cross-user agreement over the first 16 tokens is complete up to 2K at every batch and drops
with batch and ISL (8K: 5/8, 9/16, 4/32; 16K: 1/4, 1/8, 2/16; 32K: 1/4, 1/8): the recorded `user_heads` show every
variant is coherent Frankenstein text (a `<|think|>` / `<|content|>` opening or wording near-tie), and the variants come
in runs of consecutive users (e.g. B16 8192/128: users 0-4 one wording, 5-15 another), i.e. a time-correlated factor
during the sequential prefill rather than independent noise; the GPT-OSS sweep on the same box showed the same (3/8 at
32K B8, 9/16 at 16K B16). Reported, not asserted.

## Memory per device (TP=8)

| item | bfp8 experts | bfp4 experts |
|---|---|---|
| routed experts (48 x 255 MiB) | 11.95 GiB | 6.33 GiB |
| shared experts, attention, router, lm_head, RoPE tables | 0.09 + 0.45 + 0.05 + 0.13 + 0.125 GiB (with `SOLAR_OPEN_FUSE_SHARED_EXPERT=1` the 0.09 GiB shared row folds into the routed row as slot 128: 12.04 GiB, total unchanged; init transient <= 180 MiB per layer for the fused copy) | same |
| embedding (bf16, replicated) | 1.5 GiB | 1.5 GiB |
| fixed total | 14.3 GiB | 8.7 GiB |
| paged KV (bfp8, 13,056 B/token) | 32 x 8K = 3.19 GiB, 32 x 16K = 6.38 GiB, 32 x 32K = 12.75 GiB, 1 x 64K = 0.80 GiB, 1 x 128K = 1.59 GiB | same |
| measured 2026-09-07: DRAM after load / free after the 32 x 8K pool + 512 decode steps | 14.44 GiB / 14.15 GiB free, i.e. 17.34 GiB for KV + activations | 8.81 GiB / 19.77 GiB free (22.96 GiB) |
| KV budget default / hard cap (`SOLAR_OPEN_KV_BUDGET_GIB`, `tt/common.py`) | 8 GiB (657,920 tokens = 32 x 20.5K) / 14.5 GiB (1,192,448 tokens) | 14 GiB (1,151,360 = 32 x 36K) / 20 GiB (1,644,800) |
| DRAM left for activations with a 32 x 16K / 32 x 32K pool | 10.9 GiB / 4.55 GiB (32 x 32K needs `SOLAR_OPEN_KV_BUDGET_GIB=13`) | 16.6 GiB / 10.2 GiB |

Phase-2 defaults (design_misc.md (b)): the KV budget guard admits 32 x 16K with bfp8 experts and 32 x 32K with bfp4
by default; the bfp8 32 x 32K pool needs `SOLAR_OPEN_KV_BUDGET_GIB=13`. The hard cap = DRAM (31.74 GiB) - weights -
2.8 GiB reserve for long-prefill activations, program binaries and the trace region; the env override is clamped to
it. The reserve is an estimate (a 64K single-user prefill holds ~1.5-2.5 GiB of transients: a bf16 `[64K, 4096]`
activation is 512 MiB, the `ttnn.all_reduce` composite ~0.6 GiB, RoPE slices and binaries ~0.25 GiB) until the
`prefill_64k` case has recorded the peak. `tt/common.py::kv_budget_tokens` turns the budget into whole 64-token blocks
(the max-tokens-all-users figure a serving layer needs). Batch-32 decode at a full 32K context reads the whole 12.75 GiB
pool every step: expect ~140-160 ms/step (~115-125 at 16K) against the 92 ms plateau at 8K.

## Serving with vLLM (untested here)

vLLM is NOT installed in the bring-up venv (`python -c "import vllm"` -> `ModuleNotFoundError`), so everything in
this section is code plus a host-only smoke test; nothing has run against a live vLLM server or the TT plugin.
Treat every statement about the vLLM side as a hypothesis to verify on a machine with the
[TT vLLM plugin](https://github.com/tenstorrent/vllm/tree/dev/plugins/vllm-tt-plugin).

- Wrapper: `SolarOpenForCausalLM` in `models/tt_transformers/tt/generator_vllm.py` (additive, next to the other
  text-model wrappers, subclass of `HybridAttentionForCausalLM`). Registry key = `hf_config.architectures[0]` =
  `SolarOpenForCausalLM` (`configs/Solar-Open-100B/config.json`); the plugin's `model_registry.py` line (out of
  tree) is `"SolarOpenForCausalLM": ("models.tt_transformers.tt.generator_vllm", "SolarOpenForCausalLM")`. The
  Solar-specific logic is in `tt/vllm_support.py`, importable and unit-tested without vllm.
- `initialize_vllm_model(hf_config, mesh_device, max_batch_size, max_seq_len, n_layers, tt_data_parallel,
  optimizations)` refuses anything but the 1x8 mesh, `tt_data_parallel=1`, `max_batch_size <= 32`
  (`--max-num-seqs`), `max_seq_len <= 131072` (`--max-model-len`), `optimizations=None` and a model name other
  than `Solar-Open-100B` BEFORE the host load, then calls `create_tt_model(create_kv_cache=False, dtype=bfp8,
  moe_options=MoEOptions.from_env())`: `HF_MODEL`, `TT_CACHE_PATH`, `MESH_DEVICE=P150x8` and the `SOLAR_OPEN_*`
  flags must be exported for the server process (`source env.sh`); a cold cache does the 393 GB host load.
- KV cache: `get_kv_cache_spec` emits 48 `FullAttentionSpec` entries (`model.layers.<i>.self_attn`; SolarOpenConfig
  has no `layer_types`, so the base class' lookup cannot be used), i.e. one KV group. `allocate_kv_cache((max_num_blocks,
  kv_heads, block_size, 128), dtype, num_layers)` re-runs `check_kv_budget` (`create_tt_model` skipped it) and
  zero-fills replicated bfp8 pools with `ttnn.from_torch` like `attention/kv_cache.py` - deliberately not
  `allocate_vllm_kv_cache`, whose `as_tensor(cache_file_name=...)` would write ~27 GB of zero tensorbins per pool
  shape. `kv_heads` is the per-device 1 (the TT worker divides the 8 KV heads by TP); an undivided 8 is accepted.
  Use `--block-size 64` (the demo's page block; any multiple of 32 passes the guard).
- `get_max_tokens_all_users()` = `kv_budget_tokens(MoEOptions.from_env())` from `tt/common.py`: the KV budget
  (`SOLAR_OPEN_KV_BUDGET_GIB` or its per-expert-dtype default) divided by 13,056 B per token per device in whole
  64-token blocks - 8 GiB -> 657,920 tokens (32 users x 20.5K), 13 GiB -> 1,069,120, 14 GiB -> 1,151,360. The pool
  vLLM allocates from this figure is what `allocate_kv_cache` then checks against the same budget.
- `model_capabilities`: `supports_prefix_caching False` (a nonzero `start_pos` is unvalidated), `supports_async_decode
  True`, `supports_sample_on_device True`, `max_device_top_k 32` (`TTSampling.max_top_k`). `prefill_forward` /
  `decode_forward` take `Generator`'s legacy single-`page_table` path (with one KV group the plugin's per-layer
  tables are copies of it; `page_tables_per_layer` is accepted and ignored).
- Stop ids: vLLM stops on `generation_config.json`'s `eos_token_id` `[2, 24, 25]` itself (`--generation-config auto`,
  the default); the wrapper exposes `stop_token_ids` (= `ModelArgs.stop_token_ids`) for parity checks only and never
  truncates generations. `<|end|>` (21) closes a message without ending the turn and is not a stop id.
- Chat template (`chat_template.jinja`): kwargs `reasoning_effort` (default `high`; `low` / `minimal` prepend an
  empty `<|think|><|end|>` block), `default_system_prompt` (default true, dated through the `strftime_now` Jinja
  global) and `think_render_option` (`lastthink`). Per request: `"chat_template_kwargs": {"reasoning_effort": "low"}`
  in the OpenAI request body (or the server-wide default-chat-template-kwargs option where the installed vLLM has
  it); `vllm_support.chat_template_kwargs()` returns the demo's effective values (env `SOLAR_OPEN_REASONING_EFFORT`
  / `SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT` applied). The demo's `low` is a demo choice, not a wrapper default.
- Reasoning / tool parsers: the HF repo ships `solar_open_reasoning_parser.py` (`SolarOpenReasoningParser`:
  reasoning between `<|think|>` and `<|end|>`, content after `<|content|>` / `<|tool_calls|>`, `is_reasoning_end`
  recognises the empty think block of `low`) and `solar_open_tool_parser.py` (`SolarOpenToolParser`, needs
  `pyjson5`) WITHOUT register decorators - Upstage's fork `UpstageAI/vllm@v0.12.0-solar-open` registers them
  internally (`--reasoning-parser solar_open --tool-call-parser solar_open`). On a stock vLLM + TT plugin build use
  the in-tree plugin file, which registers both under `solar_open`
  (`vllm_support.register_vllm_parsers()`; `HF_MODEL` must be the snapshot directory):

  ```bash
  P=models/demos/solar_open/vllm_plugins/solar_open_parsers.py
  vllm serve upstage/Solar-Open-100B --max-num-seqs 32 --max-model-len 8192 --block-size 64 \
      --reasoning-parser-plugin $P --reasoning-parser solar_open \
      --tool-parser-plugin $P --tool-call-parser solar_open --enable-auto-tool-choice   # UNTESTED
  ```

  The two request logits processors of the HF README (`SolarOpenTemplateLogitsProcessor`,
  `ParallelToolCallLogitsProcessor`, token ids 20-25 / 30-34) run on host logits and are incompatible with on-device
  sampling; requests that need them must take the host-sampling path (plugin behaviour, unverified).
- Open items to verify on the vLLM side, in this order: (1) decode batches arrive padded to `max_num_seqs` with
  position -1 in the unused slots (`Model.ttnn_decode_forward` raises unless the batch equals
  `max_local_batch_size`); (2) `prefill_forward` returns pre-sampled argmax tokens when `sampling_params` is given
  (`tt/model.py::process_output_prefill`) - correct for greedy, sampled first tokens must come from host sampling;
  (3) the `kv_cache_shape` head count and `block_size` the plugin passes; (4) warmup (`warmup_model_prefill`, traced
  prefill at 128 only on P150x8) and trace-region use (100 MB, `models/model_trace_region_sizes.yaml`); (5) the
  effective `--max-model-len` x `--max-num-seqs` against the 8 GiB default budget (32 x 16K fits;
  `SOLAR_OPEN_KV_BUDGET_GIB=13` or `SOLAR_OPEN_EXPERT_DTYPE=bfp4` for 32 x 32K).
- Host smoke test (no device, vllm optional): `pytest models/demos/solar_open/tests/unit/test_vllm_wrapper_import.py`
  checks `vllm_support` with mocks (token capacity vs the budget, the 48-entry KV spec, request validation, the
  `from_torch` allocator and its budget refusal, stop ids, template kwargs, the plugin file) and parses
  `generator_vllm.py` to keep the wrapper's capabilities / method set / signature in sync; the cases that import the
  wrapper class run only where vllm is importable (skipped here).

## Known limitations

- Batch <= 32 users (single mesh row); row-sharded multi-row meshes are not part of this tree.
- Single-user prefill is capped at 64K tokens on the 1x8 mesh (the 128K cases skip); `max_position_embeddings`
  131072 is supported by the RoPE tables.
- On-device sampling `top_k <= 32` (`TTSampling.max_top_k`); the demo clamps larger requests with a log line.
- The demo defaults to `reasoning_effort=low`; `high` needs `max_generated_tokens >= 2048` (case `reasoning_high`).
- The trace region size (100 MB, `models/model_trace_region_sizes.yaml`) is inherited from the source demo's
  bh_loudbox entry; measured usage is 23.6 MiB (prefill@128 trace + batch-1 decode trace) / 25.6 MiB (batch-32 decode)
  of the 93.1 MiB region, so it suffices with ~3.6x headroom.
- Cold cache builds default to the whole-model host load (393 GB peak RSS). The phase-2 streaming loader
  (`SOLAR_OPEN_STREAMING_LOAD=1`, section "Streaming loader") is host-validated bit-exact on layer 0 / embed / norm /
  lm_head and its 2-layer device smoke build is byte-identical to the phase-1 cache (8.8 GB peak RSS), but the full
  48-layer cold cache build has not been run through it yet, so it is opt-in until that run is byte-identical.
- The router prebuilds its per-token-count helper tensors (the `[T, 128]` bias copy the fused top-k op needs and
  the bf16 zeros the scatter starts from) for `T` in {decode batch, 32, 128} and keeps them for every `T <= 128`
  it meets; longer prefill lengths rebuild them per call (two device ops, freed after the scatter) and must not be
  traced (only 128 is a traced prefill length on P150x8).
- `create_tt_model` refuses KV pools above `SOLAR_OPEN_KV_BUDGET_GIB` (paged and unpaged alike; the demo skips such
  cases before loading anything). Measured on the full model (2026-09-07): 14.44 GiB of weights per device, 17.57 GiB
  with the batch-32 x 8K pool (3.19 GiB), 14.15 GiB still free after a 512-step batch-32 decode, i.e. 17.34 GiB for
  KV + activations: 32 x 16K leaves 10.9 GiB, 32 x 32K leaves 4.55 GiB (an earlier note said ~1.4 GiB; it
  double-counted the 8K pool that was still allocated when the 14.15 GiB were measured). Defaults 8 / 14 GiB, hard
  caps 14.5 / 20 GiB (bfp8 / bfp4 experts); the 2.8 GiB activation reserve behind the caps is unmeasured until the
  `prefill_64k` case has run.
- lm_head padding: `compute_per_device_vocab(196608, 8)` rounds the 24,576-column per-device shard up to the next
  power of two (32,768; padded vocab 262,144) because `ttnn.topk`'s multi-core path needs a power-of-two width, so
  devices 6 and 7 hold only zero columns (6 x 32,768 = 196,608) and every device runs the same `[32, 4096] x
  [4096, 32768]` decode matmul (~0.1 ms/step; 0.133 GiB of bfp8 weight instead of 0.100). A 6-way split is not
  possible (one shape per mesh tensor, the 8-device sampling gather and TTSampling's `padded // 8` offset stride); an
  exact 8 x 24,576 split is representable and this tree's `ttnn.topk` would route the non-power-of-two width to the
  Blackhole-only `topk_large_indices` kernel, but it needs a new cache stem, changes the sampler's tie order and buys
  ~33 MiB per device and < 0.1 ms per step, so it stays documented only (design_misc.md (c)); revisit if a profile
  shows lm_head + sampling above ~1 ms/step.
- The expert-sorted prefill MoE path passes TILE tables to `ttnn.embedding` (two untilize copies per 1024-token
  split) and caches a `[split, split]` bf16 identity per layer (2 MiB at 1024; ~120 MiB per device over 48 layers
  after a long prefill); a module-level ROW_MAJOR identity is a phase-2 cleanup.
- Decode batch sizes must map onto one core per user on a single rectangle of at most 8x8 cores for
  `nlp_concat_heads_decode` (any batch <= 8, multiples of 8, powers of two up to 32); `Model.__init__` raises for
  others (11, 13, 17, 19, 22, 23, 26, 29, 31) before any weight is loaded.
- The shared-expert fusion (`SOLAR_OPEN_FUSE_SHARED_EXPERT=1`) needs EP=1 (129 slots do not split over EP groups),
  `n_shared_experts == 1` and a tile-aligned per-device shared intermediate (160 at TP=8); a single-user decode step
  then runs the batched union-of-experts path, NOT the indexed path of `SOLAR_OPEN_INDEXED_DECODE` (the indexed and
  single-user scan paths have no contract for always-on slots), so fused batch-1 decode is slower than the unfused
  default (see the fusion row of "Recorded baselines"). With `SOLAR_OPEN_SHARED_EXPERT_DTYPE=bf16` the fused slot is
  quantised to the routed expert dtype on device (the bf16 option is ineffective in fused mode; logged once).

## Streaming loader (phase 2, `SOLAR_OPEN_STREAMING_LOAD=1`)

`utils/streaming_loader.py::LazyStateDict` (DESIGN 4.16) replaces the whole-model `from_pretrained` on cold cache
builds. It is a `Mapping` over the 42 safetensors shards that presents exactly the 627 contract-C1 keys (3 + 48 x 13):
the 384 per-expert tensors of a layer are collapsed into the virtual `mlp.experts.gate_up_proj [128, 2560, 4096]`
(gate rows first) and `mlp.experts.down_proj [128, 4096, 1280]`, assembled straight into one preallocated buffer with
`os.preadv` reads sorted by file offset on 4 threads (no mmap: `safe_open.get_tensor` leaves file-backed pages mapped);
`q_proj` / `k_proj` are Meta-permuted on access (`load_checkpoints.reverse_permute`, the rule of
`convert_hf_qkv_to_meta_format`); the router bias stays fp32; fp32 stragglers get the phase-1 safety-net cast.
`substate()` returns lazy prefix views (two `hasattr` hooks in `utils/substate.py`), so `Model`, `DecoderLayer` and every
weight consumer walk it unchanged, reading each tensor exactly once and freeing it when their constructor returns.
The loader keeps no tensor: its "one-layer LRU" is a window of <= 2 open shard fds plus an asynchronous
`posix_fadvise(WILLNEED)` prefetch of the next layer's byte ranges (embed -> layers -> norm/lm_head);
`close()` releases the fds (idempotent, a later access reopens). `ModelArgs.load_state_dict` validates the C1 layout
from the shard headers only (`meta()`) and returns the loader without the dict rebuild. `==` is identity, `pickle` /
`deepcopy` raise, and `items()` / `values()` / `nn.Module.load_state_dict(lazy)` would materialise the whole checkpoint
(the HF reference generators keep the phase-1 path).

```bash
# opt-in cold build (the warm-cache decision is unchanged: with a complete marker no loader is constructed)
SOLAR_OPEN_STREAMING_LOAD=1 pytest models/demos/solar_open/demo/text_demo.py -k "prefill_128 and 1x8 and not prefill_128_en and not prefill_128k"
pytest models/demos/solar_open/tests/unit/test_streaming_loader.py   # host only: synthetic checkpoint + real layer 0 vs phase 1
```

Host validation (2026-09-07, `tests/unit/test_streaming_loader.py`): the synthetic 3-shard checkpoint (Part A, 11
tests) is bit-exact against `torch.stack` / `torch.cat` fusion + `convert_hf_qkv_to_meta_format` + the safety-net cast,
including the consumer walk through `prepare_expert_weights_torch` and `load_attention_weights`; on the real checkpoint
(Part B) the 16 tensors of layer 0 + embedding + final norm + lm_head are sha256-identical to
`AutoModelForCausalLM.from_pretrained(dtype=bf16, num_hidden_layers=1)` + Meta permute run in a subprocess (8.3 s,
10.5 GB peak; hashes cached in `$TT_CACHE_PATH/streaming_loader_reference_layer0.json`,
`SOLAR_OPEN_REGEN_LOADER_REFERENCE=1` regenerates) and to the raw per-expert shards (`gate_up_proj[e, :1280]` == disk
gate of expert e, bias sample / min / max / 10 distinct values, `permute(reverse_permute(q)) == q`); the lazy path read
7.43 GB in 398 preads (13.2 s from a cold page cache, 0.36 s / 0.19 s for the two fused tensors from a warm one) with a
peak RSS of 3.1 GB. Device smoke (2026-09-07, `tests/test_streaming_loader_device.py -k 1x8`, PASSED): a streamed
2-layer build into a temporary cache took 31.6 s (11.6 GB in 793 preads, 4 fused builds, 0 repeat reads), `ru_maxrss`
0.68 -> 8.80 GB, its 25 tensorbins (6.73 GiB) byte-identical to the phase-1 cache (0 differ, 0 missing) and a decode step
through the 2-layer model ran; `tests/test_layer0_real_weights.py` with `SOLAR_OPEN_STREAMING_LOAD=1` passes with the same
PCCs as the phase-1 load. Still pending: the full 48-layer cold build (expected ~610-630 s vs 662 s: the ~12 s/layer bfp8
pack dominates, the disk read hides under the prefetch). The default stays the whole-model load until that cache is
byte-identical; `tt/common.py` does not yet call `state_dict.close()` after the build (the fds close when the loader is
garbage-collected).

## Recorded baselines (fill in during bring-up, design D13)

Phase 1 (commit df800791d50) vs the final phase-2 tree (perf levers A-C, perf-p1 layout levers, perf-p2 indexed
decode, perf-p0 program configs a-e; same box, warm cache, bfp8 experts, `reasoning_effort=low`, greedy). Phase-2
numbers are the gate-p0 stage's runs of 2026-09-07 16:15-17:33 (`scratchpad/gate_p0/`), the phase-1 numbers the rows
below. Demo `b1` = `prefill_128` (78-token KO prompt, 4K paged context, traced prefill@128 + traced decode), `b32` =
`batch32` (32 KO/EN prompts, 8K paged context, 512-token budget; plateau = iterations 25-60).

| metric | phase 1 | phase 2 final | note |
|---|---:|---:|---|
| b1 decode ms/step (tok/s/user) | 54.4 (18.4) | **17.97-18.15 (55.1-55.4)** | two runs (fp32-dst / final bf16-dst qkv), plateau 18.0 both, iterations 2-22 17.6-18.0; = 48 x the traced layer (0.377-0.379 ms) within 1 % |
| b1 TTFT@128 ms | 186 | **153.6-155.0** | traced prefill@128 (dense-down, shared, router and qkv configs all inside the trace) |
| b32 decode ms/step avg / plateau (tok/s aggregate) | 92.1 / 93.8 (347) | **62.1-62.7 / 63.1 (508)** | iterations 2-22 74.0 -> 53.8; last-50 80.1 -> 61.5-61.8; three runs 62.14 / 62.45 / 62.67 (cool and warm box identical) |
| b32 TTFT ms/user | 184 | **149.6** | 32 sequential traced prefills |
| prefill_1k TTFT ms | 520 | **413** (runs 609 / 413 / 619) | eager, host-load sensitive; pre-p0 same-day runs 488 / 614 |
| prefill_8k TTFT ms | 4047 | **3190** (runs 3190 / 3903 / 3199) | eager 1024-token sorted-MoE splits; decode at 7.5K context 56.1 -> 19.3-19.4 ms/step |
| batch32_16k (32 x 2-15K tokens, 16K pool) | not run | TTFT **4562** ms/user, decode **81.1** ms/step avg, plateau 79.7 (402 tok/s), last-50 86.4 | from a cool box (< 58 C); started at 66-75 C the same binary measured 6263 / 123.6 (plateau 123.8) and the pre-p0 tree 7561 / 110.95 -- board 1 reaches 84 C during the run and throttles, see the long-context row |
| batch32_32k / prefill_64k | not run | 163.84 ms/step, TTFT 17662 ms/user / TTFT 51950 ms, decode 43.98 ms/step | measured on the pre-p0 phase-2 tree (2026-09-07 15:37-15:48, `SOLAR_OPEN_KV_BUDGET_GIB=13` for 32K); not re-run after the p0 merge |
| per-layer traced replay ms (real layer 0) decode b1 / b32 | 1.152 / 1.911 | **0.376-0.379 / 1.30-1.33** | `tests/perf/test_layer0_device_perf.py`, 5 runs; b1 on the indexed path; -67 % / -31 % |
| per-layer eager ms prefill 128 / 1024 | ~4.5 / - | 3.94-3.97 / 8.8-9.6 | host-load sensitive |
| DRAM per device after load / with the b32 8K pool / 16K pool | 14.436 / 17.572 / - GiB | 14.436 / 17.572 / 20.760 GiB | unchanged weights; 14.147 GiB free after 512 b32 steps, 10.838 GiB free after 256 16K steps |
| teacher-forced b1 top-1 / decisive / top-5 / top-64 PCC / full PCC / KL | 0.9297 / 0.9779 / 0.9281 / 0.98177 / 0.99215 / 0.0347 | 0.9258 / 0.9558 / 0.9234 / 0.98089 / 0.99162 / 0.0300 | floors 0.90 (0.85 per prompt) / 0.94 / 0.90 / 0.96 / 0.97 / 0.06; per prompt 0.9375 / 0.9219 / 0.9375 / 0.9062; TF decode 59.8-63.1 -> 26.5 ms/step (final default: bf16-dst qkv; with the fp32-dst variant 0.9219 / 0.9558 / 0.9086 / 0.97922 / 0.99056 / 0.0333) |
| teacher-forced b32 (same metrics) | 0.9180 / 0.9602 / 0.9227 / 0.98222 / 0.99219 / 0.0324 | 0.9297 / 0.9690 / 0.9211 / 0.97969 / 0.99064 / 0.0312 | per prompt 0.9062 / 0.8906 / 0.9844 / 0.9375; TF decode 74.0 -> 49.0 ms/step (fp32-dst variant 0.9258 / 0.9646 / 0.9164 / 0.98038 / 0.99123 / 0.0311) |
| decoder component PCC decode b1 / b32 / b16, prefill 128 / 1024 / 4096 (random weights, pos0) | 0.99325 / 0.99637 / 0.99644, 0.99618 / 0.99561 / 0.99658 | 0.99349 / 0.99729 / 0.99713, 0.99603 / 0.99561 / 0.99658 | paged == unpaged; with the fp32-dst qkv variant 0.99341 / 0.99680 / 0.99704 (pos70000 0.99332 / 0.99701 / 0.99684, 0.99611 / 0.99568 / 0.99657) |
| real-weight layer 0 decoder PCC b1 / b32 / 128 / 1024 | 0.99844 / 0.99876 / 0.99997 / 0.99982 (after lever A) | 0.99871 / 0.99886 / 0.99997 / 0.99996 | mlp 0.99980 / 0.99966 / 0.99985 / 0.99988; router flips 0/1, 1/32, 2/128, 6/1024 (0 decisive) in both; fp32-dst qkv variant 0.99855 / 0.99891 |
| ISL/OSL x batch sweep, 51 cells (`tests/test_multi_user_regression.py`, 2026-09-08, tag `_p2`) | not run | **51/51 ok**; decode 17.9 (B1) .. 60.6 (B32) ms/step at ISL 128, 918 tok/s aggregate at B32 1024/128; TTFT 150-164 ms/user at ISL 128 for every batch | matrices in "ISL/OSL x batch sweep (2026-09-08, tag `_p2`)" at the end of this section, caveats in the sweep section; full report `SWEEP_REPORT.md` (per-batch tables, per-cell checks, thermal data); long-context cells at B >= 4 throttle board 1 |

Numerics of the phase-2 tree are PCC- but not bit-equivalent to phase 1 (single-K-block expert gate|up, indexed
single-user path, 1D configs); every component, real-weight and teacher-forced floor holds, see the rows below.

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
| phase-2 expert program configs (profile-driven, `tt/expert_configs.py`: gate\|up sparse_matmul `decode_gate_up_in0_block_w` 32 -> 128 (one K block), down sparse_matmul `decode_down_subblock_w` 1 -> 4 on the 8x4 single-user grid and new `decode_down_batched_subblock_w` = 2 on the 8x8 batched grid; `tt/experts/prefill.py` sorted-MoE cost-model constants re-measured for H=4096 / Ip=160: `_SORTED_FIXED_MS, _SORTED_PER_KROW_MS, _HOT_FIXED_MS, _HOT_PER_EXPERT_MS` 2.5, 0.27, 1.0, 0.25 -> 0.5, 0.23, 0.3, 0.25 and `_DENSE_PER_EXPERT_MS` 0.125 -> 0.15). Measured lever by lever in an isolated worktree at HEAD (the other phase-2 implementers were editing the shared checkout); details, per-iteration series and logs in `scratchpad/phase2/perf_log.md` | b1 / b32 demos, prefill_1k / prefill_8k, component tests, real-weight layer 0, teacher-forced accuracy | **before -> after** (same box, same day): b1 decode 54.27 -> 52.82 ms/step (18.4 -> 18.9 tok/s; gate\|up K-block neutral -0.03, down subblock -1.4), TTFT@128 unchanged (186-192 ms, traced dense path); b32 decode avg 92.66 -> 83.17 ms/step, plateau (iterations 25-60) 93.8 -> 83.6 ms, iterations 2-22 83.3 -> 76.5, 345 -> 385 tok/s aggregate (gate\|up K-block -5.8 ms plateau, down subblock -4.4), TTFT 183 ms/user unchanged; prefill_1k TTFT 524.8 -> 506.8 ms (constants: more layers plan cap 160/192 with fewer hot experts); prefill_8k TTFT old-constant runs 4360 / 4326 / 4339 ms vs new-constant runs 4305 / 3801 / 4267 / 4086 ms (faster in every alternating pair, -55 to -253 ms; the eager 8K path varies by up to 500 ms run to run under the concurrent host load, the morning's quiet-box value was 4047). Per-layer traced replay (tests/perf/test_layer0_device_perf.py, real layer 0): decode b1 1.152 -> 1.110 ms, b32 (union 71) 1.911 -> 1.721 ms. Numerics: the single K block accumulates the whole K = 4096 in the destination registers instead of spilling three partials, so it is NOT bit-identical to phase 1 -- experts component PCC decode b1/b32/b16 0.99887/0.99892/0.99890 -> 0.99800/0.99821/0.99812 (prefill unchanged 0.99938/0.99937/0.99937; the subblock width is bit-identical), MLP 0.99828/0.99811/0.99808, decoder 0.99313/0.99628/0.99636 (from 0.99325/0.99637/0.99644; prefill 0.99618/0.99561/0.99658 identical; paged bit-identical to unpaged; 12 passed), real layer 0 mlp 0.99975/0.99970/0.99987/0.99987 and decoder 0.99844/0.99876/0.99997/0.99982 (8 passed); teacher-forced b1 top-1 0.9453 (was 0.9297), decisive 0.9779 (same), top-5 0.9148 (0.9281), top-64 PCC 0.98180, full-vocab PCC 0.99174 (0.99215), KL 0.0316 (0.0347); b32 top-1 0.9180 (same), decisive 0.9513 (0.9602; floor 0.94), top-5 0.9227, top-64 PCC 0.98119, full-vocab PCC 0.99150 (0.99219), KL 0.0338 (0.0324); slot copies 1792/1792; teacher-forced decode 57.4 / 69.2 ms/step (was 59.8-63.1 / 74.0) -- every floor holds, nothing re-baselined. Not adopted: `dense_grid_max_width` (10 vs 11 wide identical), `dense_bmm_max_tokens` 512 (dense costs 2x the sorted per-token cost). Validated but NOT wired (their call sites are outside the perf-tuning files): `ProgramConfig.get_dense_down_config` (8x8, in0_block_w = Kt) for the dense prefill down bmm, -0.5 ms per layer per 128-token split (auto 0.43/1.18/1.75 -> 0.34/0.70/1.15 ms wall at S 32/128/256, PCC vs fp32 at the bfp8 floor) -> ~-24 ms TTFT@128 once `experts/prefill.py::_dense_tail` and the sorted cold down pass it; attention decode qkv (8,5) in0_block_w 16 (60 -> 20 us kernel; PCC vs fp32 0.99994 -> 0.99983, prefer in0_block_w 8 or fp32 dst accumulation); the o_proj candidate (8,8) pcn2 shows no gain on the width-sharded concat_heads input (56 -> 64 us wall) and is dropped. `attention/config.py::_build_matmul_config` fixed to emit legal 1D configs (out_block = per-core block, fuse_batch=True, divisibility checks); `experts/config.py::_build_matmul_config` refuses single-core sparse grids (a 1x1 mcast_in0 grid hangs the device) | 2026-09-07 |
| perf-p1 layout levers of the expert paths (PCC-equivalent; `tt/experts/decode.py` `ROUTING_WEIGHTS_ON_DOWN_INPUT`: the batched decode multiplies the routing weights into the down INPUT `[1, E, 32, 160]` instead of the down OUTPUT `[1, E, 32, 4096]`; `tt/experts/prefill.py` `HOT_EXPERTS_PER_EXPERT_LINEAR`: the sorted-MoE hot group runs one `ttnn.linear` per hot expert over the whole split instead of `ttnn.repeat` x n_hot + a batched matmul (the K-concatenated down `HOT_DOWN_KCONCAT` was implemented, measured slower than bmm + fast_reduce_nc -- 2.75 vs 2.42 ms per 15-hot 1K split -- and stays off); `ELIDE_ROUTING_COPIES`: the expert-major `[E, S, 1]` routing copy is built per split on demand for the dense-bmm / per-expert-loop splits only (sorted splits never read it), the cold index reaches `ttnn.embedding` as a `[E, cap]` view instead of a `[1, E*cap]` row-major copy and the hot routing rows are tile-transposed instead of reshape-copied; hot cost-model constants re-derived `_HOT_FIXED_MS, _HOT_PER_EXPERT_MS` 0.3, 0.25 -> 0.4, 0.135 from the measured 0.39 + 0.136 x n_hot ms fit). Each lever is a module constant (A/B by sed); one-device numerics + timing vs torch fp32 in `tests/perf/test_layout_candidates.py`, host checks in `tests/unit/test_p1_layout.py`; details and logs in `scratchpad/phase2/perf_log.md` (section perf-p1) | component tests (experts / mlp / decoder decode b1/b32/b16 and prefill 128/1024/4096), skewed routing, real-weight layer 0, teacher-forced b1 + b32, demos b32 / prefill_1k / prefill_8k, per-layer traced perf | **before -> after** (main tree, same box, 2026-09-07 13:00-13:47): per-layer traced replay (real layer 0) decode b32 (union 71) 1.737 -> 1.671 ms blocking / 1.684 -> 1.637 non-blocking, b1 unchanged 1.107; prefill_1024 eager 10.7-11.3 -> 10.1-10.6 ms per layer; b32 demo avg 83.17 -> **81.23 ms/step**, plateau (iterations 25-60) 83.6 -> **81.7**, last-50 83.2 -> 80.1, TTFT 182.6 ms/user unchanged; prefill_1k TTFT (eager, host-load sensitive) same-session OFF 596.3 -> ON 521.9 / 440.1 ms (perf-lane quiet-box OFF value 506.8 -> best-of 440.1, -13 %); prefill_8k TTFT OFF 4344.8 / 4374.3 -> ON 3320.1 / 3997.7 ms (alternating pairs, -1025 / -377 ms; best-of -24 %). Numerics: experts / MLP / decoder component PCCs IDENTICAL to the row above (decode 0.99800 / 0.99818 / 0.99809 experts, 0.99828 / 0.99811 / 0.99808 MLP, 0.99313 / 0.99628 / 0.99636 decoder; prefill 0.99938 / 0.99937 / 0.99937, 0.99899 / 0.99903 / 0.99916, 0.99618 / 0.99561 / 0.99658), skewed routing 0.99932 / 0.99945 -> 0.99937 / 0.99946, real layer 0 mlp 0.99975 / 0.99969 / 0.99987 / 0.99988 and decoder identical (8 passed); one-device candidates vs fp32: routing-on-input PCC 0.999807 -> 0.999815, hot group (4 / 8 / 15 hot) 0.99944 -> 0.99954, index / routing-row layouts bit-identical; teacher-forced b1 BIT-IDENTICAL to the row above (top-1 0.9453, decisive 0.9779, full PCC 0.99174, KL 0.0316), b32 top-1 0.9180 -> **0.9375** (240/256), decisive 0.9513 -> **0.9602** (floor 0.94), top-5 0.9227 -> 0.9148, top-64 PCC 0.98119 -> 0.98082, full-vocab PCC 0.99150 -> 0.99165, KL 0.0338 -> 0.0355 (floor 0.06), per-prompt 0.922 / 0.953 / 0.953 / 0.922, slot copies 1792/1792 -- every floor holds. Also fixed on the way: `demo/text_demo.py` read `paged_attention` before its assignment (the phase-2 KV-budget pre-check), which made EVERY demo case fail at collection-time setup in the shared tree (the assignment now precedes the check) | 2026-09-07 |
| perf-p2 indexed/gather single-user expert path (profile lever 1; `MoEOptions.indexed_decode`, env `SOLAR_OPEN_INDEXED_DECODE` default 1, cache-neutral): `tt/topk.py::TopKRouter.route_indexed` returns the fused op's top-8 uint16 ids (one untilize) and bf16 weights (a view) as an `experts.IndexedRouting` without the dense scatter; `tt/experts/decode.py::_decode_forward_indexed` runs gate\|up and down as `ttnn.sparse_matmul(indices=...)` (compact `[1, 8, 1, *]` outputs, `is_input_a_sparse` down on the compact A, `fast_reduce_nc` over the 8 slots, shared partial + single all_reduce unchanged; `nnz` never passed; the cached `[1, 1, 1, E]` prefill ones serve as the required-but-unread `sparsity`); `tt/mlp.py::MLP.route` is the single dispatch point (indexed only for ONE token with the fused router, EP=1 and the unfused shared expert). Routing weights multiply the compact bfp8 GLU rows (`INDEXED_WEIGHTS_ON_DOWN_INPUT = True`; False = the scan path's order on the compact down output, retained as the A/B variant). New tests: `tests/unit/test_p2_indexed.py` (host, 24), `tests/perf/test_indexed_candidates.py` (1x1), `test_router.py::test_router_route_indexed`, `test_model[decode_b1_s1]`; details and logs in `scratchpad/phase2/perf_log.md` (section perf-p2) | router, experts / mlp / decoder decode_b1 unpaged + paged, shared-expert hook, real-weight layer 0, test_model, teacher-forced b1, b1 demo (traced), per-layer traced perf, one-device candidates vs torch fp32 | **before -> after** (main tree, same box, 2026-09-07 14:15-14:38): b1 demo **52.83 -> 35.77 ms/step** (plateau 53.0 -> 36.0, TTFT@128 187-192 ms unchanged; 18.9 -> 28.0 tok/s), per-layer traced replay (real layer 0) decode b1 **1.109 -> 0.760 ms** blocking / 1.080 -> 0.731 non-blocking (-31 %), eager 2.96 -> 2.58 ms; traced expert block alone (1x1, random bfp8 weights) 0.477 -> 0.133 ms. Numerics: PCC vs torch fp32 of the expert block 0.998220 (scan) vs 0.998177 (indexed; output-side mul 0.998213); indexed vs scan PCC 0.99980, max diff one bfp8 ulp (no bit-equivalent indexed form exists: the 8-slot reduction alone moves ~18 % of the elements by half an ulp); component PCCs experts / MLP / decoder decode_b1 0.99800 / 0.99828 / 0.99314 (scan 0.99800 / 0.99828 / 0.99313), paged == unpaged; real layer 0 mlp 0.99974 / decoder 0.99843 (scan 0.99975 / 0.99844); test_model decode_b1_s1 0.99921; hook b1 zeros stub exact, shift 8.1028 / 8.0; teacher-forced b1 top-1 0.9453 -> **0.9336** (239/256), decisive 0.9779 -> **0.9558** (floor 0.94), top-5 0.9148 -> 0.9180, top-64 PCC 0.98180 -> 0.97945, full-vocab PCC 0.99174 -> 0.99071, KL 0.0316 -> 0.0355 (floor 0.06), per prompt 0.9375 / 0.9375 / 0.9531 / 0.9062 -- every floor holds; the output-side mul variant lands at 0.9258 / 0.9646 / 0.99142 / 0.0346 with the same component PCCs, +3-5 us per layer and 36.7 ms/step. Batched (b32) decode and prefill are untouched (dense routing). router test 26 passed | 2026-09-07 |
| perf-p0 program-config levers (profile section 6, levers 2-5 + the dense prefill down config), implemented and gated in an isolated git worktree of the same HEAD (removed after the merge) while perf-p1 / perf-p2 changed the main tree's expert paths: (a) shared expert `ttnn.linear` configs ((5,1) grid, in0_block_w 32 for gate/up; (8,8) pcn2 for the down; auto above 128 rows), (b) width-sharded 8x4 decode `rms_norm`, (c) router linear (4,1) config with fp32 accumulation, (d) decode qkv (8,5) in0_block_w 16 with HiFi2 restated (ttnn drops a bf16 x bfp8 matmul to LoFi as soon as a program config is passed without a compute config -- `matmul_device_operation.cpp::create_matmul_attributes`), (e) dense prefill down 1D 8x8 in0_block_w 5 (`ProgramConfig.dense_down_cores`; the main tree's `experts/prefill.py` already carries the inert call-site helpers) | per-lever unit tests + `tests/perf/test_layer0_device_perf.py` on the worktree (scan path, no perf-p1/p2 levers), one-device numerics vs torch fp32 (`tests/perf/test_config_candidates.py`) | Worktree measurements (MERGED into the main tree by the merge-p0 stage and re-gated there by the gate-p0 stage, see the next rows; the worktree was removed afterwards). Worktree ladder, traced replay ms per layer (real layer 0): all off 1.111 (b1) / 1.721 (b32 union 71) -> a 0.946 / 1.638 -> a+c 0.862 / 1.480 -> a+c+d 0.833 / 1.457 -> a+c+d+e 0.820 / 1.438; prefill_128 eager 4.43 -> 3.88 ms per layer with e (~-26 ms TTFT@128); lever b FAILED its unit test (`test_decoder --test-modules=rms_norm decode_b32`: `TT_FATAL tensor_spec.cpp !shard_grid_fit_error`, the [32, 128] width shards do not fit the 8x4 grid spec) and was switched off. Numerics (PCC vs torch fp32, auto -> candidate): dense down 0.999959 -> 0.999953 (wall 0.44 / 1.19 / 1.89 -> 0.35 / 0.69 / 1.13 ms at S 32 / 128 / 256), qkv bw16 without a compute config (LoFi) 0.999936 -> 0.999828, bw16 HiFi2 0.999862, bw8 HiFi2 0.999927, bw16 HiFi2 + fp32 dst 0.999994 (all 0.081-0.088 vs 0.125 ms), shared gate at M=1 0.999272 -> 0.999734 (0.127 -> 0.049 ms), o_proj identical and slower (dropped). Expected on the main tree once merged: b1 ~36 -> ~28-30 ms/step, b32 ~82 -> ~69-71, TTFT@128 -26 ms. Details: `scratchpad/phase2/perf_log.md` (sections perf-p0 / final) | 2026-09-07 |
| merge-p0: perf-p0 levers a, b, c, d, e ported hunk-by-hunk into the main tree on top of perf-p1 / perf-p2 (`tt/linear_configs.py` new; `tt/shared_expert.py` `shared_expert_program_configs` + `program_configs=` switch; `tt/topk.py` `router_linear_program_config` wired into `_select` (both `__call__` and `route_indexed` take it); `tt/attention/config.py` `decode_qkv_fp32_dest_acc` + `get_decode_qkv_compute_config`; `tt/attention_configs.py` `decode_qkv_cores=(8, 5)`, `in0_block_w=16`, `decode_qkv_fp32_dest_acc=True` (switched back to False by the gate-p0 stage after the whole-model A/B); `tt/attention/decode.py` qkv matmul with the explicit program AND compute config; `tt/expert_configs.py` `dense_down_cores=(8, 8)` (the call sites already carried the helpers); `tt/rms_norm.py` width-sharded 8x4 decode norm, `DECODE_NORM_GRID=(8, 4)`, `sharded_decode=` switch). Every wired matmul config passes an explicit compute config (HiFi2, no approx, packer_l1_acc; fp32 dst for qkv; the router's HiFi4 / fp32-acc config unchanged) because ttnn drops a program-config matmul to LoFi otherwise. Lever b root cause: the worktree gated the sharded path on `x.shape[-2] <= 32`; the rms_norm component test feeds the HF-shaped `[32, 1, 4096]` decode input whose 32 one-row batches pad to 32 tile rows (physical height 1024), so the `[32, 128]` width shard could not cover it (`TT_FATAL tensor_spec.cpp !shard_grid_fit_error`). Fixed with `decode_norm_applies`: the sharded kernel runs only for interleaved bf16 tensors whose TILE-PADDED shape is exactly one tile row of `hidden` columns (the model's decode inputs `[1, 1, T <= 32, 4096]`); everything else keeps the default kernel. Host tests `tests/unit/test_p0_program_configs.py` (39), `tests/perf/test_config_candidates.py` extended with the `RMSNorm` module gate check | py_compile + import gate; host: test_p0_program_configs, test_model_config, test_expert_parallel_config, test_p1_layout, test_p2_indexed, test_attention_precision_option (151 passed), test_fused_shared_expert -k host (7 passed); one device (1x1): `tests/perf/test_config_candidates.py -k 1x1` PASSED (PCC vs torch fp32, auto -> wired) | dense down 0.999959 -> 0.999953 (S 32 / 128 / 256 wall 0.43 / 1.18 / 1.76 -> 0.34 / 0.68 / 1.16 ms); qkv (8,5) bw16 HiFi2 + fp32 dst 0.999936 -> **0.999994** (wall 0.105 -> 0.070 ms; bw16 LoFi 0.999828 / 0.061, bw16 HiFi2 0.999862 / 0.068, bw8 HiFi2 0.999927 / 0.087); shared gate M = 1 / 32 / 128 0.99927 / 0.99919 / 0.99917 -> 0.99973 / 0.99969 / 0.99968 (0.119 / 0.117 / 0.115 -> 0.038 / 0.037 / 0.061 ms), shared down 0.999984 -> 0.999979 (0.042 -> 0.034 ms); router M = 1 / 32 / 128 BIT-IDENTICAL to auto (0.117 / 0.117 / 0.121 -> 0.038 / 0.040 / 0.064 ms); rms_norm config M = 1 0.9999963 -> 0.9999948, M = 32 0.9999626 -> 0.9999935 (eager wall 0.078 -> 0.078 / 0.086 -> 0.123 ms: three launches instead of one, the 57 -> 7 us kernel gain shows in the traced decode only); `RMSNorm` module on `[1,1,1,H]` / `[1,1,32,H]` / `[1,1,16,H]` sharded (PCC vs fp32 0.9999953 / 0.9999936 / 0.9999933 vs default 0.9999945 / 0.9999578 / 0.9999822, max abs err 0.023 / 0.053 / 0.034 vs 0.105 / 0.197 / 0.141), `[32,1,H]` / `[1,32,1,H]` / `[1,1,128,H]` on the default kernel (bit-identical to `sharded_decode=False`), no TT_FATAL. Gated on the main tree by the gate-p0 stage (next row) | 2026-09-07 |
| gate-p0: full 1x8 gate of the merged tree (levers a-e ON; `scratchpad/gate_p0/`, `perf_log.md` section gate-p0) | per-layer traced perf + A/B of the two perf-decided levers; test_router, test_shared_expert, test_decoder all 7 components pos0 + pos70000 paged + unpaged, hook, skewed routing, real-weight layer 0, test_model, teacher-forced b1 + b32, demos b1 / b32 / prefill_1k x3 / prefill_8k x3 / batch32_16k (hot and cool box), fused test | **All levers KEPT.** Per-layer traced replay (real layer 0, blocking mean): decode b1 0.758 -> **0.376-0.379** ms (lever b OFF 0.471; qkv fp32 dst OFF 0.378), b32 1.676 -> **1.30-1.33** (b OFF 1.389; fp32 dst OFF 1.318), prefill_128 eager 4.4-4.5 -> 3.94-3.97, prefill_1024 9.3-10.1 -> 8.8-9.6 -> lever b kept (-0.09 ms/layer traced despite 2 reshard launches per norm); the qkv fp32 destination (equal time) was measured on the whole model afterwards and switched OFF (below). Components (pre-merge in brackets): shared decode 0.99931 / 0.99920 / 0.99921 [0.99843 / 0.99820 / 0.99819], prefill_128 0.99917 [0.99965] (the 1D config at exactly 128 rows lands at the 1-32-row level; vs torch fp32 it is the closer one, gate M=128 0.99917 -> 0.99968), attention decode 0.99930 / 0.99919 / 0.99920 [0.99923 / 0.99907 / 0.99909], mlp decode 0.99894 / 0.99812 / 0.99896 [0.99814 / 0.99732 / 0.99812], decoder decode 0.99341 / 0.99680 / 0.99704 [0.99314 / 0.99628 / 0.99636], prefill 0.99603 / 0.99561 / 0.99658 [0.99618 / = / =]; router, experts, rms_norm unchanged; test_shared_expert bfp8 T=1/32/128 0.99915 / 0.99913 / 0.99915 [0.99794 / 0.99802 / 0.99966]; real layer 0 mlp 0.99980 / 0.99966 / 0.99985 / 0.99988, decoder 0.99855 / 0.99891 / 0.99997 / 0.99996 (all >= before); test_model prefill_b1_s128 0.99052 [0.99189] (lever a at 128 rows), decode_b32_s1 0.99920 [0.99903], decode_b1_s1 0.99769 [0.99921] -- the latter is an interaction of the sharded decode norm with the fp32-DESTINATION qkv accumulation in the 0.02-std random 1-layer model (lever b OFF 0.99950, lever d OFF 0.99939, fp32 dst OFF with b and d ON 0.99935). Teacher-forced with fp32 dst ON (floors hold): b1 top-1 0.9336 -> 0.9219, decisive 0.9558 (=), top-5 0.9180 -> 0.9086, top-64 PCC 0.97945 -> 0.97922, full 0.99071 -> 0.99056, KL 0.0355 -> 0.0333, TF decode 45.3 -> **26.2** ms/step; b32 0.9375 -> 0.9258, decisive 0.9602 -> 0.9646, top-5 0.9148 -> 0.9164, 0.98082 -> 0.98038, 0.99165 -> 0.99123, KL 0.0355 -> 0.0311, 69.8 -> **48.7** ms/step. Teacher-forced A/B with the bf16 destination (`scratchpad/gate_p0/tf/b1_fp32off`, `b32_fp32off`): b1 **0.9258 / 0.9558 / 0.9234 / 0.98089 / 0.99162 / KL 0.0300** (26.5 ms/step), b32 **0.9297 / 0.9690 / 0.9211** / 0.97969 / 0.99064 / 0.0312 (49.0) -- better-or-equal on 10 of 12 metrics at identical traced time and without the test_model interaction -> `decode_qkv_fp32_dest_acc = False` is the final default (True = A/B switch); re-gated with the flipped default (all rc=0, `scratchpad/gate_p0/h_*.log`): attention component decode 0.99893 / 0.99890 / 0.99887 (the one gate that prefers the fp32 destination: 0.99930 / 0.99919 / 0.99920; auto 0.99923 / 0.99907 / 0.99909), prefill =; decoder component decode **0.99349 / 0.99729 / 0.99713**, prefill =; test_model 0.99052 / 0.99902 / **0.99935**; per-layer traced 0.379 / 1.319 ms; demos b1 18.15 ms/step (TTFT 155), b32 62.67 (plateau 63.1, TTFT 150); real layer 0 decoder **0.99871** / 0.99886 / 0.99997 / 0.99996, mlp 0.99980 / 0.99966 / 0.99985 / 0.99988 -- everything downstream of the attention op is better-or-equal with the bf16 destination. Demos: b1 36.45 -> **17.97 ms/step** (TTFT 193 -> 154), b32 81.21 -> **62.14** (plateau 81.9 -> 63.1, 391 -> 508 tok/s; TTFT 184 -> 150), prefill_1k best-of-3 488 -> 413 ms, prefill_8k 3273 -> 3190 ms, batch32_16k from a cool box TTFT 7561 -> 4562 ms/user and 110.95 -> 81.06 ms/step (hot-box run 123.6: throttling). `test_fused_shared_expert` re-baselined: `FUSED_VS_UNFUSED_PCC` 0.999 -> 0.998 (the unfused MLP moved closer to HF, 0.99822 -> 0.99896 at decode b1; the fused slot is unchanged at 0.99827) and the launch count is checked against the unfused block per sorted split (a 4096-token prefill's 4 splits have different hot counts) | 2026-09-07 |
| text_demo `batch32_16k` / `batch32_32k` / `prefill_64k` (phase 2, design_misc.md (b): 32 distinct 2-15K / 4-31K-token clips + KO/EN questions; 8192 / 16384 / 1024 page blocks) | KV pool allocation under the new budgets, TTFT per user, decode ms/step (iterations 2-22 and plateau) at long contexts, DRAM after load / prefill / decode, 64K transient peak, answer sanity | MEASURED (2026-09-07, warm cache, `scratchpad/phase2/final/d_b32_16k.log`, `d_p64k.log`, `d_b32_32k.log` on the pre-p0 phase-2 tree; `scratchpad/gate_p0/d_b32_16k*.log` on the final tree). `batch32_16k` (pre-p0 tree, started at 67-77 C): TTFT 7561 ms/user, decode 110.95 ms/step avg (iterations 2-22 98.2, plateau 101.0, last-50 121.0), DRAM 20.760 GiB after load (16K pool 6.32 GiB), 20.879 after the prefills, 20.882 after 256 steps, largest free block 10.838 GiB; final tree from a cool box (every ASIC < 58 C): TTFT **4562** ms/user, **81.06** ms/step avg (iterations 2-22 76.9, plateau 79.7 = 402 tok/s, last-50 86.4 as board 1 climbs to 84 C), started hot (66-75 C) the same tree measured 6263 / 123.6 (plateau 123.8, steps oscillating 94-131 ms = throttling of the hottest board) -- always cool the box before this case. `prefill_64k` (pre-p0 tree): TTFT 51950 ms, decode 43.98 ms/step at a 64K context, DRAM 15.183 GiB after load (1024-block pool), 15.317 after the prefill, 15.321 after decode, largest free block 16.009 GiB (transient peak inside the 2.8 GiB reserve). `batch32_32k` with `SOLAR_OPEN_KV_BUDGET_GIB=13` (pre-p0 tree): TTFT 17662 ms/user, decode 163.84 ms/step (256 steps), DRAM 27.135 GiB after load (32K pool 12.7 GiB), 27.268 after the prefills, 27.271 after decode, largest free block 4.426 GiB. The bfp4 variants and the 64K / 32K cases on the final tree are not measured | 2026-09-07 |
| `SOLAR_OPEN_ATTENTION_BF16_OUTPUT=1` (phase 2, design_misc.md (a); skips the o_proj-input and pre-all_reduce bfp8 typecasts) | test_teacher_forced b1/b32 vs the bit-identical phase-1 baselines (b1 top-1 0.9297 / full-vocab PCC 0.99215 / KL 0.0347; b32 0.9180 / 0.99219 / 0.0324), demo `prefill_128` + `batch32` step times (54.4 / 92.1 ms) | **MEASURED WORSE, root cause open, default stays OFF** (final chain1 2026-09-07 15:19-15:22, `scratchpad/phase2/final/tf/b1_bf16out`, `b32_bf16out`; same reference, same final-tree defaults as the passing rows): b1 top-1 0.9336 -> **0.8828** (per prompt 0.859 / 0.859 / 0.938 / 0.875 vs 0.938 / 0.938 / 0.953 / 0.906), decisive 0.9558 -> 0.9248 (floor 0.94), top-5 0.9180 -> 0.9008, top-64 PCC 0.97945 -> 0.97526, full PCC 0.99071 -> 0.98985, KL 0.0355 -> **0.0654** (floor 0.06; max 0.83), teacher-forced decode 45.3 -> 47.4 ms/step; b32 top-1 0.9375 -> 0.8906, decisive 0.9602 -> 0.9248, KL 0.0355 -> 0.0630, 69.8 -> 74.1 ms/step; demos b1 36.45 -> 37.22, b32 81.21 -> 80.34 ms/step, TTFT unchanged. Both teacher-forced cases FAIL the floors. Host review of the option (merge-p0 stage): in decode the ONLY change is the dtype on the all_reduce wire (o_proj already ran bf16 x bfp8 at HiFi2 with a bf16 output; the bfp8 typecast before the reshape / `ttnn.all_reduce` is skipped), in prefill the o_proj input stays bf16 (HiFi2 instead of the LoFi of two bfp8 operands) and the all_reduce moves to bf16 too; the residual add (`ttnn.add(residual, branch, dtype=bf16)`), the norms, the router and the experts never see the option, and `ttnn.all_reduce` -> `all_reduce_async` picks the reduce_scatter_minimal_async + all_gather_async path independently of the dtype -- so on paper every changed operand is at least as precise and the accuracy loss cannot be explained from the host side. Device component check (merge-p0 stage, `scratchpad/merge_p0/attn_bf16out.log`: `test_decoder --test-modules=attention -k "1x8 and pos0 and unpaged"` with the option ON, random layer-0 weights, same seeds as the default run `final/u_decoder_pos0.log`): 6 passed, every all-reduced output replica-consistent on the 8 devices; attention PCC vs HF ON / default: decode b1 0.99915 / 0.99923, b32 0.99926 / 0.99907, b16 0.99927 / 0.99909, prefill 128 0.99930 / 0.99920, 1024 0.99880 / 0.99871, 4096 0.99856 / 0.99845 -- neutral-or-better in isolation, so the whole-model loss is NOT reproducible at the component level with random weights and needs a real-weight A/B (`tests/test_layer0_real_weights.py` with the env, then a per-layer teacher-forced probe) in the device lane. Candidates left for the device lane: the reduce_scatter / all_gather kernels with bf16 pages of 2 KiB (validated exact on P150x8 only with bfp8 pages), the bf16 dst accumulation order of the 8 partials, and the auto program config the bf16 prefill o_proj gets | 2026-09-07 |
| `SOLAR_OPEN_FUSE_SHARED_EXPERT=1` (phase-2 fusion row: shared expert as always-on slot 128; `tests/unit/test_fused_shared_expert.py -k 1x8`, random layer-0 weights, final tree) | fused vs unfused MLP, both vs `SolarOpenMoE`, weights / router columns bit-identical, one all_reduce, `ttnn.linear` launch count | PASSED 8/8 on the final tree (2026-09-07 16:52-17:14, `scratchpad/gate_p0/u_fused2.log` + `u_fused3.log`): fused vs unfused PCC decode b1 / b8 / b32 0.99890 / 0.99887 / 0.99889, prefill 128 / 1024 / 4096 0.99903 / 0.99928 / 0.99940 (floor `FUSED_VS_UNFUSED_PCC` 0.998: the phase-1 shared expert measured 0.9993-0.9994 against the fused slot, the perf-p0 1D configs moved the UNFUSED output closer to HF -- decode b1 vs HF 0.99822 -> 0.99896 -- while the fused slot (bfp8 activations, routed compute config) stays at 0.99827, so the two forms now agree at the bfp8-activation floor); vs HF unfused 0.99896 / 0.99898 / 0.99857 / 0.99863 / 0.99915 / 0.99931, fused 0.99827 / 0.99825 / 0.99780 / 0.99894 / 0.99919 / 0.99924 (the fused block is the less accurate one on decode, equal on prefill); fused weights and routed router columns bit-identical, `{'all_reduce': 1}` on both; linears 4 -> 1 (decode, prefill 128), 4 -> 2 (1K: one sorted split, always-on hot linear), 5 -> 6 (4K: 4 sorted splits, one of which also has a routed hot expert -- the expectation is `unfused - 3 x chunks + one always-on linear per sorted split`). Perf / teacher-forced ladder in fused mode NOT measured (single-user decode falls back to the batched union path, so fused b1 is slower than the unfused default): the default stays 0 | 2026-09-07 |
| test_vllm_wrapper_import (host only, no device, vllm absent) | vllm_support plumbing with mocked ttnn / mesh + AST checks of `SolarOpenForCausalLM` | PASSED: 35 cases, the 4 cases that import the wrapper class skipped (vllm not installed); token capacities 657,920 (8 GiB bfp8 default) / 1,151,360 (14 GiB bfp4) / 1,069,120 (13 GiB); 48 FullAttentionSpec keys; 96 `from_torch` calls for 48 layers; 25.5 GiB pool refused before any allocation. UNTESTED against a live vLLM | 2026-09-07 |
| test_streaming_loader (host only, no device; `SOLAR_OPEN_STREAMING_LOAD=1` loader) | Part A synthetic 3-shard checkpoint (11 tests) + Part B real layer 0 / embed / norm / lm_head vs `from_pretrained(num_hidden_layers=1)` | PASSED 13/13 (27 s with a cold page cache): 16 real tensors sha256-identical to the phase-1 path; lazy loader 7.43 GB in 398 preads, 2 fused builds, 0 repeat reads, peak RSS 3.1 GB (reference subprocess 10.5 GB peak, 8.3 s); layout validation on the lazy dict reads 0 tensor bytes; before: phase-1 cold build 393 GB peak RSS (device smoke / full cold build pending) | 2026-09-07 |

### ISL/OSL x batch sweep (2026-09-08, 156304d63b5, tag `_p2`)

`tests/test_multi_user_regression.py` via `tests/sweep/run_sweep.sh` (TMO 4000 s, cooldown gate 78 C / 120 s, cool-start < 60 C): 1x8 mesh,
TP=8, bfp8 experts / attention / KV, traced decode, prefill traced at 128 only (eager 1K-32K), sequential per-user prefill, KV pool
`min(64K, 512K // B)` tokens per user (64K at B 1-8, 32K at B 16, 16K at B 32), page-table seed 1234, `reasoning_effort=low`, greedy, exactly
OSL steps. **51 / 51 cells ok** on the first attempt of every batch (96 min of pytest, 1 h 49 min end to end); 0 first-token failures, 0
degenerate users, QA keyword accuracy 1.00 in the 12 ISL-128 cells. Skipped by the context rule (`-` below): B16 32768/128, B32 16384/128 and
32768/128. Source: `generated/solar_open_multi_user_regression/Solar-Open-100B_1x8_p2.jsonl` (51 rows; `logs/REPORT_p2.md` = `report.py
--matrix`); the full report `SWEEP_REPORT.md` (per-batch tables with every column, per-cell checks, per-cell temperatures and AI clocks,
harness changes) is generated from that jsonl and kept on this box under `/home/eslim/experiments/solar/results/sweep/` with a copy of the
jsonl. Caveats (board-1 throttling on the long-context cells at B >= 4, shared-prompt decode understating the MoE cost, cross-user
agreement pattern) are in the sweep section's "Reading the numbers".

Decode step ms (mean, steady state):

| B \ ISL/OSL | 128/128 | 128/1024 | 1024/128 | 2048/128 | 4096/128 | 8192/128 | 8192/1024 | 16384/128 | 32768/128 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 17.9 | 17.6 | 17.8 | 18.1 | 18.6 | 19.2 | 19.1 | 20.4 | 22.1 |
| 2 | 32.9 | 33.1 | 30.8 | 30.9 | 31.2 | 31.8 | 32.1 | 33.4 | 34.7 |
| 4 | 37.4 | 37.3 | 31.7 | 32.0 | 32.4 | 32.8 | 32.9 | 39.1 | 42.9 |
| 8 | 42.6 | 42.6 | 31.4 | 31.7 | 32.9 | 36.1 | 40.5 | 43.2 | 54.9 |
| 16 | 51.4 | 51.0 | 34.7 | 35.3 | 36.1 | 43.4 | 44.9 | 55.1 | - |
| 32 | 60.6 | 60.9 | 34.9 | 35.6 | 48.1 | 57.1 | 73.3 | - | - |

TTFT mean over users, ms (= per-user prefill x (B+1)/2):

| B \ ISL/OSL | 128/128 | 128/1024 | 1024/128 | 2048/128 | 4096/128 | 8192/128 | 8192/1024 | 16384/128 | 32768/128 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 163 | 168 | 426 | 798 | 1944 | 3915 | 3285 | 7075 | 14147 |
| 2 | 229 | 239 | 629 | 1192 | 2312 | 5868 | 6039 | 12178 | 24071 |
| 4 | 377 | 392 | 1174 | 2110 | 3973 | 9831 | 9945 | 27484 | 59456 |
| 8 | 676 | 680 | 2259 | 3807 | 7212 | 21144 | 20976 | 53561 | 115169 |
| 16 | 1276 | 1275 | 3680 | 6852 | 14793 | 42676 | 43632 | 103898 | - |
| 32 | 2551 | 2471 | 6921 | 13723 | 39646 | 93760 | 95009 | - | - |

TTFT last user, ms (whole batch admitted = B x per-user prefill):

| B \ ISL/OSL | 128/128 | 128/1024 | 1024/128 | 2048/128 | 4096/128 | 8192/128 | 8192/1024 | 16384/128 | 32768/128 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 163 | 168 | 426 | 798 | 1944 | 3915 | 3285 | 7075 | 14147 |
| 2 | 306 | 319 | 839 | 1589 | 3083 | 7824 | 8052 | 16237 | 32094 |
| 4 | 603 | 627 | 1878 | 3377 | 6356 | 15729 | 15912 | 43974 | 95129 |
| 8 | 1201 | 1210 | 4016 | 6768 | 12821 | 37590 | 37290 | 95219 | 204746 |
| 16 | 2402 | 2400 | 6926 | 12897 | 27845 | 80332 | 82132 | 195572 | - |
| 32 | 4947 | 4792 | 13423 | 26614 | 76890 | 181838 | 184260 | - | - |

Aggregate decode tok/s:

| B \ ISL/OSL | 128/128 | 128/1024 | 1024/128 | 2048/128 | 4096/128 | 8192/128 | 8192/1024 | 16384/128 | 32768/128 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 56 | 57 | 56 | 55 | 54 | 52 | 52 | 49 | 45 |
| 2 | 61 | 60 | 65 | 65 | 64 | 63 | 62 | 60 | 58 |
| 4 | 107 | 107 | 126 | 125 | 124 | 122 | 122 | 102 | 93 |
| 8 | 188 | 188 | 255 | 252 | 243 | 221 | 198 | 185 | 146 |
| 16 | 311 | 314 | 461 | 454 | 443 | 369 | 357 | 290 | - |
| 32 | 528 | 526 | 918 | 899 | 665 | 561 | 436 | - | - |
