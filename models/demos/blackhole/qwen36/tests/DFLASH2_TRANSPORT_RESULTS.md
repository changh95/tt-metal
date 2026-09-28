# DFlash2 drafter -- target-side aux capture + P/D transport of the context K/V (2026-09-28, half A P150x4, TP=4)

Code: `tt/aux_hidden.py` (aux semantics, `QWEN36_SPEC_DRAFTER` knob, gather / readback helpers, the P-side
`DFlash2ContextPrefillHook` + `KvGroupStage`), `tt/verify_step.py` (`keep_aux_hidden` -> `plan.out_aux`),
`tt/model.py` (`prefill_aux_layers` / `prefill_aux_hook`, per-path aux copies), `tt/pd_transfer.py` (KV groups:
`register_kv_group`, `export_kv_groups`, `import_kv_groups`, `kv_group_import_warmup`, `read_kv_group_blocks`),
`tt/qwen36_vllm.py` (`_install_dflash2_prefill_hook` behind the knob), plugin `tt_mooncake_connector.py` (payload
version 3). Tests: `tests/test_verify_aux_scratch.py` (T1), `tests/pd_dflash2_transfer_repro.py` + `tests/dflash2_stub.py`
(T2/T3 two-process round trip with a random fixed-weight projector stub), plugin `tests/test_pd_payload.py` /
`test_pd_handoff.py` (v3 host tests). Runners: `scripts/verify_step_run.sh`, `scripts/pd_mtp_run.sh` (now gated by
the half-aware `scripts/chips_gate.sh`).

## Aux hidden state semantics (confirmed in the plugin venv's vLLM)

`Qwen3NextModel.forward` (qwen3_next.py; `Qwen3_5Model` inherits it) collects
`aux_hidden_states` through `_maybe_add_hidden_state(aux, layer_idx + 1, hidden_states, residual)` after every
layer, value = `hidden_states + residual` (the FULL post-MLP residual stream), index 0 = the embeddings; so index
`j` = the residual after `j` layers = the OUTPUT of 0-based `layers[j-1]`. `gpu_model_runner._get_eagle3_aux_layers_from_config`
sets `aux_hidden_state_layers = [i + 1 for i in dflash_config.target_layer_ids]` ("Add 1 to convert DFlash's aux
layer id semantics"). Hence DFlash2's `target_layer_ids = [5, 19, 33, 47, 61]` are 0-based layer indices whose
OUTPUT residual is taken: `model.layers[5]`, `[19]`, `[33]`, `[47]`, `[61]` (all GDN layers), i.e. the `x` returned
by `Qwen36DecoderLayer.forward` / `forward_verify` for those `layer_num`s; rows concatenated in that order to
`[N, 5 * 5120]` bf16, no norm applied (DFlash2's `fc` takes them raw; `qwen3_dflash.py::precompute_and_store_context_kv`
= `fc -> hidden_norm -> per-layer kv_proj -> k_norm -> RoPE(position)`).

## T1 -- verify-step aux capture (`VerifyStep(keep_aux_hidden=True)`)

`plan.out_aux` `[1,1,R,5*5120]` bf16 replicated DRAM, allocated in `VerifyPlan.__init__` (before any capture); in
the body, after every aux layer the residual is copied (fused-AR path R <= 32: `to_memory_config` of the L1
width-sharded replicated residual -> DRAM interleaved; fractured path R > 32: one `all_gather_async` of the
`[1,1,R,dim/TP]` residual), the 5 copies concatenated and `ttnn.copy`'d into `out_aux` before the final norm --
copies / data movement only, the main path's ops are unchanged (argmax rows identical with and without it).

Exactness (`logs/vaux_exact3.log`, flow: compile every plan -> prefill warm-up -> capture all plans -> prefill the
32 users -> per plan): the eager forward's `row_check` reference rows == the eager `out_aux` == the traced replay's
`out_aux` (GDN state restored to the pre-step values between the runs through persistent device snapshots +
`reset_qkv_prev`) bitwise, all 5 layers, all rows, and eager argmax == traced argmax:

| plan | R | path | eager out_aux == ref | traced out_aux == ref | argmax eager == traced |
|---|---|---|---|---|---|
| (1,8) | 8 | fused AR | yes | yes | yes |
| (8,8) | 64 | fractured (all-gather) | yes | yes | yes |
| (32,4) | 128 | fractured (all-gather) | yes | yes | yes |

Cost (`logs/vaux_timing1.log`, traced step = upload + replay + readback, median of 50, one live trace at a time):

| plan | R | aux bytes/step | step without | step with | delta |
|---|---|---|---|---|---|
| (1,8) | 8 | 400 KB | 37.63 ms | 37.75 ms | +0.11 ms |
| (8,8) | 64 | 3.2 MB | 55.19 ms | 55.25 ms | +0.06 ms |
| (32,4) | 128 | 6.4 MB | 79.50 ms | 79.46 ms | -0.04 ms (noise) |

The copies are hidden in the trace (R * 51,200 B of bf16 rows; the eager per-section profile shows 5.6-8.4 ms only
because it synchronizes after each op).

## T2 -- prefill-side aux capture on P (`model.prefill_aux_layers`, `model.prefill_aux_hook`)

Every TP prefill path copies the residual after the aux layers and calls
`prefill_aux_hook(user_ctx, aux_list, token_buf, actual_len, bucket, chunk_start)` right after the hidden hook:
the traced 2048 chunk body (`_forward_prefill_chunk_tp`) and the traced masked-bucket bodies
(`_forward_prefill_bucket_body_tp`) `ttnn.copy` into persistent per-layer `[1,1,S,dim/TP]` buffers allocated at
capture time (so `prefill_aux_layers` must be set before the prefill warm-up -- `_install_dflash2_prefill_hook` runs
in phase 1 of `warmup_model_prefill`), the eager masked path clones. `user_ctx` = `(u, slot, page_table_row, total_len)`
(the 4th field is new; `MTPHead.prefill_hook` ignores it). Empty `prefill_aux_layers` = no extra op anywhere.

The DFlash2 hook (`aux_hidden.DFlash2ContextPrefillHook`) all-gathers the 5 fractured copies to the replicated
`[1,1,S,5*5120]` rows, runs the projector (`project_device(aux_rep, positions)` fast path, else the host-facing
`project(rows, positions)` of `dflash2_head.DFlash2ContextProjector` after a host readback) and stages the per-layer
`(K, V)` rows `[n, 8, 128]` bf16 per decode slot (`KvGroupStage`); its device programs are compiled per bucket
(`hook.compile`) before any capture (without it the first 4196-token request paid 1.4 s of all-gather JIT + 1.4 s
of projector JIT: `logs/pddf_P1.log`).

Cost per segment on P with the STUB projector (`logs/pddf_P1.log`, eager, incl. the 42 MB host readback of the
projected K/V per 2048 chunk; the real projector's cost is the drafter's): all-gather + concat of a 2048 chunk
2.5 ms (case c: 10.1 ms / 4 chunks), stub projection + readback ~330 ms per 2048 chunk, 32 ms per 128 bucket.
The aux copies themselves are inside the prefill traces (no measurable change of `prefill_paged_slots`: 201 ms for a
128-token request incl. the hook vs 173-200 ms before).

## T3 -- transport v3 (payload KV groups)

Format (plugin `tt_mooncake_connector.py`, `PAYLOAD_VERSION = 3`): after `kv.<li>.k/.v` (16 main layers),
`mtp.kv.<j>` (when the MTP drafter is selected), `gdn.rec`, `gdn.taps`, `mtp.hidden`, the tensors
`<group>.kv.<j>.k/.v` per group layer, each `[n_shipped_blocks, kv_heads, block_size, head_dim]` bf16 (dim 1 in
global kv-head order = device-major over the D shards) and `header["kv_groups"][<group>] = {n_layers, kv_heads,
head_dim, block_size, block_index, first_pos, n_tokens}` (`block_index` = the shipped blocks as indices into the
request's block list). Group `"dflash2"`: 5 layers x 8 kv heads x 128 = 20 KB/token bf16 (the main KV payload is 16 x
4 x 256 = 64 KB/token bf16; on device the main KV is bfp8 ~34 KB/token). A v2 consumer reads a v3 payload exactly as
before (it reads `kv.<li>` / `mtp.*` / `gdn.*` by name, unknown names ignored: plugin test
`test_v2_consumer_view_of_a_v3_payload_ignores_the_group_tensors`); a v3 consumer reads a v2 payload (no groups).
The mtp fields are unchanged (`pd_mtp_transfer_repro.py` asserts no `kv_groups` when the mtp drafter stages).

Knob `QWEN36_SPEC_DRAFTER=mtp|dflash2` (default mtp; master switch still `QWEN36_SPEC_MTP=1`): `mtp` = the MTP
head + hidden row as before; `dflash2` = no MTP head is built, `_install_dflash2_prefill_hook` installs the aux
hook, `_stage_one` ships the staged group through `pd_transfer.export_kv_groups`. Unset / mtp: byte-identical path.
`QWEN36_DFLASH2_CONTEXT_WINDOW` (default 0 = ship every position): with 2048 the producer computes / ships only the
drafter's sliding-window tail (an 8k prompt: 32 of 128 blocks, 40 MiB instead of 160 MiB) -- to be turned on once
the device drafter applies the window (today `DFlash2Drafter` attends the whole context).

D side: `pd_transfer.register_kv_group(model, "dflash2", drafter.kv_layers, pad_block=drafter.pad_block)` (or attach
the drafter as `model.dflash2_drafter`: `kv_groups()` registers it on first use) + `kv_group_import_warmup(model)`
BEFORE `import_warmup` / any capture; the connector drain then imports the group's blocks right after the main KV
(`import_kv_groups`: eager upload + `paged_fill_cache` per cache, power-of-two bucket padding to the group's pad
block, typecast to the cache dtype) and parks `entry[4]["kv_groups"]["dflash2"] = meta` for the runner. A consumer
without the group logs once and skips it.

### Round trip (`tests/pd_dflash2_transfer_repro.py`; P: eager masked buckets + traced chunks, D: traced masked
buckets + traced chunks; stub projector with identical seeded weights in both processes)

P: `logs/pddf_P1.log`; D: `logs/pddf_D1.log`.

| case | prompt | users | P hook calls/req | P hook ms/req (gather + stub project) | payload bytes/req (vs 16-layer baseline) | stage ms/req (export + pack) | pull | D KV + group import ms/req | imported drafter K/V == D-local rows == P rows |
|---|---|---|---|---|---|---|---|---|---|
| a | 128 | 8 | 1 (bucket 128) | 32.8 (0.6 + 32.0) | 158.2 MiB (+2.5 MiB, +1.6 %) | 28.7 (5.3 + 18.0) | 65 ms, 2.4 GB/s (file) | 8.2 (main KV traced 2 blocks + group bucket 2) | 8/8 users, 128/128 positions, all 5 layers |
| b | 4196 | 1 | 3 (2 chunks + tail 128) | 2741 in the first run (JIT, fixed by `hook.compile`) | 494.2 MiB (+82.4 MiB, +20 %) | 109.9 (68.4 + 41.5) | 189 ms, 2.55 GB/s | 515.8 (of which ~450 = the group's 128-block bucket compiled at request time; warm-up now covers <= 256) | 4196/4196 positions, eager chunked == traced |
| c | 8192 | 1 | 4 chunks | 1333 (10.1 + 1322) | 819.8 MiB (+160 MiB, +24 %) | 167.7 (96.0 + 71.7) | 306 ms, 2.62 GB/s | 68.4 (main KV 2 traced chunks of 64 ~55 + group bucket 128 ~13) | 8192/8192 positions, eager chunked == traced |

GDN slot import 12.0-12.3 ms/req as before. D's own prefill reproduces P's first token in every case. D ran the
masked buckets TRACED (`QWEN36_PREFILL_BUCKET_TRACE=1`), P eager: the traced copies (`ttnn.copy` into the persistent
aux buffers) and the eager clones stage identical rows; the eager chunked path (`_prefill_chunked_eager_tp`, no chunk
trace) stages the same rows as the traced chunk replays (case b, c). The group bytes: 20 KB/token bf16 (a bfp8 packing
would halve them but torch has no bfp8 and the D cache may be bf16 or bfp8 -- the import typecasts to the cache's dtype);
with `QWEN36_DFLASH2_CONTEXT_WINDOW=2048` case c would ship 32 blocks / 40 MiB (+6 %).

(baseline = the same blocks without a drafter, 2026-09-25 `MTP_SPEC_RESULTS.md`: 155.8 / 411.8 / 659.8 MiB, stage
27.0 / 80.3 / 97.3 ms.)

## T4 -- byte identity with the knob unset / mtp (`scripts/dflash2_regress.sh dfregress1`, half A, 2026-09-28 15:00)

Every new path is guarded (`keep_aux_hidden=False`, `prefill_aux_layers == ()`, `prefill_aux_hook is None`, no KV
group staged / registered): with `QWEN36_SPEC_DRAFTER` unset the op sequences of prefill, verify and transport are the
ones before, and the decode path is not touched at all (`git diff 7b46fe66684 HEAD` touches no decode function).

* `pd_transfer_repro.py` (`logs/dfregress1_pdrepro.log`): dst slot 3 **24/24 tokens match**, remaps slot 3 -> rows 1 /
  0 / 6 **12/12, 12/12, 12/12**; export KV 32 MiB (8 blocks x 16 layers) warm 11.6 ms, import KV 7.5 ms, import GDN
  12.0 ms (2026-09-25 `pd_regress_short.log`: the same 24/24 + 3 x 12/12).
* `profile_prefill_decode.py` verify_regress harness (`PROFILE_ISLS=128 PROFILE_DECODE_WIDTHS=1,32`,
  `logs/dfregress1_verify.log`): `PROFILE_RESULT prefill_128=141.61ms decode_w1=24.86ms decode_w32=33.69ms`. The
  2026-09-24 reference (22.58 / 30.67 ms) ran with the AICLK under firmware control (1350 MHz); today every chip
  reads `aiclk 1200` (the verify-step tests' persistent `FORCE_AICLK 1200` pin, not released while the other agent's
  test runs on half B -- a clock step mid-kernel is the documented wedge). 24.86 / 22.58 = 1.10 and 33.69 / 30.67 =
  1.10 = the clock ratio 1350 / 1200 (1.125) minus the host share of the step; the pinned-clock references of the
  verify-step timing table (DecodeRef w1 24.9 / w32 34.4 ms incl. the argmax tail, VERIFY_W32_AUDIT.md) match. Prefill
  128 at 141.6 ms reflects the round-4 P-side kernel items (7b46fe66684), independent of this work.
* Plugin host suite: 476 passed (`PYTHONPATH=ci/host-stubs .venv/bin/python -m pytest tests/ --ignore=tests/tt`);
  pre-commit clean on every changed file in both repos.
