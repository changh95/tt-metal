# Verify-step w=32 wedge audit (2026-09-24)

Six half-A wedges (PCIe reads 0xffffffff, board reset needed) in `tests/test_verify_step_scratch.py` processes; four
clean processes. `profile_prefill_decode.py` (decode w=32 traced, `verify_regress`: decode_w1 22.58 / w32 30.67 ms,
`logs/tracy_verify_regress.log`) runs the same decode body at w=32 without incident, so this is a hazard of the test
process, not of B=32 on the hardware.

## Factor table

| process (log) | DecodeRef widths | verify plans | first hang site | outcome |
|---|---|---|---|---|
| verify_smoke2 | 1 | (1,8) | - | clean |
| verify_88_stub2 / verify_88_kernel | 8 | (8,8) | - | clean |
| verify_rowdiag88c | none | (8,8) eager only | - | clean |
| verify_smoke3 | 8, **32** | (8,8),(32,8) | decode w8 replay after 8-user prefill | wedge |
| verify_full_stub | 1, 8, **32** | 4 plans | decode w32 replay after 32-user prefill | wedge |
| verify_324_stub / _kernel | **32** | (32,4) | after (32,4) capture (prefill or first decode replay) | wedge |
| verify_timing_kernel | 1, 8, **32** | 4 plans | (32,4) verify replays (no prefill) | wedge |
| verify_timing18_88_kernel | 1, 8, **32** | (1,8),(8,8) | decode **w8** replays, after decode w1 replays (no prefill, no w=32 verify body) | wedge |

The only 6/6 vs 0/4 factor is a **`DecodeRef(32)` in the process** (a width-32 decode trace captured with the test's
own two-stage argmax tail). A w=32 verify body is 5/6 (absent in the last wedge); 32-user prefills 3/6; the
`sec_x_rep` persistent L1 tensor 2/6.

## DRAM budget per chip at w=32 (TP=4, BMAX=32, BPU=8 blocks, P150 32 GB)

| item | size |
|---|---|
| weights (bfp8/bfp4 shards) + DRAM-sharded decode copies (w2_ds 23.7 MB + w1/w3_ds 2x13 MB per layer, 64 layers) + prefill swiglu gate_up | ~11.6 GB |
| paged KV 16 layers x 256 blocks x [1,64,256] bf8 x2 | 142 MB |
| GDN rec_state fp32 48 x [32,12,128,128] + packed history 48 x [32,12,4,32,32] bf16 | 1.2 GB + 151 MB |
| verify qkv_prev 48 x [1,R,4128] bf16: R=64 / 128 / 256 | 25 / 51 / 101 MB |
| verify stub scratch (S, H copies) | 28 MB |
| sel/selT, masks, per-j cos/sin/cur_pos, tokens, page table, accept | < 2 MB |
| trace region (fixed reservation) : prefill 5 masked buckets ~141 MB + chunk trace; decode traces ~few MB each; verify traces | 1 GiB |
| DecodeRef inputs (tokens, cur_pos, rope, page table) per width | < 1 MB |

Total ~13.5 GB of 32 GB: no DRAM pressure. L1 persistent: DecodeAllReduce buffers (2 x 41 KB/core), CCL semaphores,
and (timing commits only) `VerifyPlan.sec_x_rep` [32, 5120] width-sharded (10 KB/core) for R <= 32 plans.

## Allocation order vs the trace rules (masked_bucket_trace.py / model.py: persistent buffers before any capture; a
buffer allocated after a capture lands in that trace's freed-intermediate range and is clobbered by its replays)

test_verify_step_scratch order: model -> allocate_kv_caches -> **VerifyPlan(s)** (all persistent verify buffers, GDN
lazy constants forced) -> prefill warm-up (chunk + masked-bucket captures) -> warm 1-user prefill -> **DecodeRef per
width**: `prepare_inputs_decode` (4 device inputs, allocated AFTER the prefill captures), eager compile, trace capture
(outputs idx/val allocated during capture) -> verify compile + capture per plan -> replays.

profile_prefill_decode order: model -> KV -> prefill warm-up -> warm prefill -> per width: inputs, compile, capture,
50 replays, **release_trace** before the next width; output is the full replicated logits, no argmax ops; one trace alive
at a time.

Differences that matter:
1. My decode traces stay alive together (w=1, 8, 32) and their inputs/outputs were allocated after other captures; values
   are refreshed before every replay, so a clobber is harmless for INPUTS, but each trace's OUTPUT tensors may alias
   another trace's intermediates. Harmless by construction (read right after the owning replay) unless a replay is
   issued while a previous trace's readback is still in flight (`ttnn.to_torch` after `synchronize_device`: not the case).
2. The decode trace tail differs: `argmax_sharded_rows` = `to_layout(logits [1,1,B,62080] -> ROW_MAJOR)` (124 KB sticks)
   + `argmax(dim=-1)` + `max(dim=-1)` inside the trace, vs the harness's replicated all-gather of the logits.
3. VerifyPlan buffers exist before the prefill captures (rule-compliant) but change where the prefill traces' intermediates
   land relative to the harness.

## Ranked hypotheses

1. **The traced argmax tail at B=32** (`to_layout` untilize of 32 x 124 KB sticks and/or multi-core `argmax` over 32 rows,
   or `max` over 1940 tiles per row) corrupts L1 beyond its CBs (the codebase documents this class: unbounded static CBs
   overrunning the persistent CCL semaphore buffers at the top of L1, attention/tp.py non-paged SDPA comment) -> a later
   CCL op (all-gather / reduce-scatter / fused all-reduce) inside any trace waits forever -> device timeout / PCIe wedge.
   Consistent with: wedge with DecodeRef(32) present 6/6, hang site "the next trace replay" (also a decode w8 replay
   whose own trace was healthy before), harness (no argmax) clean at w=32, B <= 8 argmax clean.
2. Cross-trace aliasing of a live buffer: several decode traces + verify traces alive; an output or input of one trace
   sitting in another's freed-intermediate range while still being read (would need a readback overlapping a replay).
3. The w=32 verify body's dense back-to-back paged SDPA-decode at 3 cores/head (documented decode wedge site) -- ruled
   out as the sole cause by the last wedge (no w=32 verify body), still a candidate for the (32,*) verify replays.
4. `sec_x_rep` persistent L1 allocation before the prefill captures shifting the prefill traces' L1 layout (only in the
   last two wedges; cannot explain the first four).

## Minimal isolation experiments (each a fresh process, harness-like flow, 3-minute wedge kill)

- E1: harness flow + `DecodeRef(32)` (test argmax tail) capture + 50 replays, no VerifyPlan. Wedge => hypothesis 1.
- E1b (if E1 wedges): same with the harness output path (no argmax ops in the trace) to confirm the tail is the culprit;
  fix candidate: text_demo's tile-parallel `_maxval_dev_b` (pad + reshape + max) and an argmax on a sliced row block, or
  read the [B, V/TP] shard back and argmax on host for the reference.
- E2 (if E1 clean): harness flow + `VerifyPlan(32,4)` allocation + compile + capture (no verify replay) + `DecodeRef(32)`
  replays. Wedge => hypothesis 3/4 (w=32 verify body or its allocations).
- E3: E2 + 50 verify replays with `QWEN36_SDPA_DEC_MAX_CORES_PER_HEAD=1`.

Runner: `scripts/verify_step_chain.sh "<tag>|TEST=models/demos/blackhole/qwen36/tests/test_verify_w32_isolation_scratch.py VERIFY_ISO=E1"`.

## Isolation outcomes (2026-09-24 15:48-15:52, kernel-mode GDN, per-offset attention)

- E1 (`logs/verify_iso_E1.log`): harness flow + DecodeRef(32) with the traced argmax tail, 50 replays: **clean**,
  decode w32 31.5 ms (harness 30.7 + the argmax/readback tail). Hypothesis 1 (argmax tail at B=32) is refuted as a
  poison-by-presence.
- E2 (`logs/verify_iso_E2.log`): + VerifyPlan(32,4) allocated before the prefill captures, compiled and captured (no
  verify replay), then DecodeRef(32) 50 replays: **clean**. A w=32 verify body's allocation/compile/capture does not
  poison the process either (hypotheses 3/4 as presence effects refuted). Side finding: with the (32,4) plan resident the
  decode w32 replay costs 34.6 ms vs 31.5 ms without it (E1) -- the plan's ~80 MB of DRAM (qkv_prev, scratch) shifts the
  decode buffers' placement; worth a look when the verify step is integrated.

Revised ranking: (i) interaction of several live traces / interleaved replays and re-prefills in one process
(hypothesis 2; every wedge process had >= 2 decode traces or a verify trace replayed alongside; E1/E2 each had one decode
trace + at most one never-replayed verify trace); (ii) the w=32 verify trace REPLAY itself (E3 never ran); (iii) the
32-user prefill paths (3/6). Next experiments (fresh processes): E3 = E2 + 50 verify (32,4) replays; E4 = E1 + a second
DecodeRef(8) captured before it and replayed after it (two live decode traces, no verify code at all); E5 = E4 +
`release_trace` between widths (the harness discipline).

## E3 / E4 / E5 (2026-09-24 15:56-16:01) -- all clean

- E3 (`logs/verify_iso_E3.log`): E2 + 50 verify (32,4) replays (per-offset attention, kernel GDN, SDPA cores unrestricted):
  clean, verify (32,4) 85.2 ms, decode w32 34.7 ms.
- E4 (`logs/verify_iso_E4.log`): decode w8 and w32 traces both alive, replays interleaved 5 x (10 + 10): clean (28.9 / 34.6 ms).
- E5 (`logs/verify_iso_E5.log`): w8 trace released before the w32 capture: clean (34.7 ms).

None of the isolated factors reproduces the wedge: not the argmax tail at B=32 (E1), not the w=32 verify body's
allocations/capture (E2) nor its replays (E3), not two live decode traces with interleaved replays (E4). What every wedged
process had and no clean one did: the EXACTNESS phase's interleaving of trace replays with re-prefills of the same
slots (`prefill_paged_slots` replays of the masked-bucket prefill traces + slot writes between decode / verify replays),
or, in the two no-prefill wedges, >= 3 decode traces + >= 2 verify traces alive at once. The timing test now keeps exactly
one trace alive at a time (capture -> time -> release, `b0fe5f2c7b3`) and both timing processes (`logs/verify_timing_A.log`,
`verify_timing_B.log`) ran clean, including (32,4) and (32,8) replays. Next isolation for the exactness flow: E6 = one
decode trace + one verify trace + re-prefill of 8 users between replay bursts (fresh process); E7 = E6 with the
slot-write path forced to the host repack + `QWEN36_PREFILL_BUCKET_TRACE=0` (eager masked prefill, no prefill trace
replays) to separate "prefill-trace replay while other traces are parked" from "slot write".
