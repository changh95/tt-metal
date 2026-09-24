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

## E6 / E7 (2026-09-24 16:13) -- (a) reproduces, (b) not yet run

- **E6 = (a)** (`logs/verify_iso_E6.log`): one decode w32 trace + one verify (32,4) trace parked, then round 1: 8 x
  `prefill_traced_chunked` into the B=1 scratch (masked-bucket prefill-trace replays + KV fill, NO GDN slot write),
  followed by 10 decode + 10 verify replays: **wedged inside round 1** (no round line logged; PCIe ID 1 reads
  0xffffffff). Prefill-TRACE replays while w=32 decode/verify traces are parked reproduce the wedge on their own.
  The same pattern with w=8 traces parked (the (8,8) exactness runs: 3 x 8-user prefill_paged_slots between replays)
  was clean 3/3, so it is w=32-specific.
- **E7 = (b)** (served-D pattern: `import_gdn_slot(mode=fillcache)` + eager `import_kv_blocks` into 8 slots between
  replays, no prefill-trace replay) and **E8** (traced importers): not run -- the board was down after E6 (E7's pytest
  errored at mesh open). Both are implemented in `test_verify_w32_isolation_scratch.py` (VERIFY_ISO=E7 needs
  `QWEN36_PD_KV_IMPORT_TRACE=0`; E8 warms the traced importers before the captures) and are the next runs.

Interpretation so far: the wedge needs (i) a w=32 decode and/or verify trace parked and (ii) prefill-trace replays
(masked-bucket traces captured BEFORE those traces) executed afterwards. The one no-prefill wedge
(`verify_timing18_88_kernel`, decode w8 replays after a decode w32 capture) does not fit (ii) and stays unexplained.
Candidate mechanism for (i)+(ii): a persistent buffer of the w=32 body allocated lazily at its compile/capture (after
the prefill captures) landing in a prefill trace's freed-intermediate range -- e.g. per-chunk virtual page tables of the
batched attention path, or a kernel-side constant -- clobbered by the prefill replay; a garbage page table then sends
paged_update_cache / SDPA at out-of-range blocks. Check: dump `buffer_address()` of every device tensor the (32,4) plan
and the decode w32 inputs own after capture, and compare with the prefill traces' intermediate ranges (allocator
report), or simply re-allocate them before the prefill warm-up and re-run E6.

Test-flow rule until (b) is settled: for (32,*) exactness, do ALL prefills / slot duplication BEFORE capturing the decode
and verify traces and never replay a prefill trace afterwards (one policy per process, reference streams saved to disk).

## E7 (2026-09-24 16:23) -- (b) the SERVED-D import pattern reproduces the wedge: served-integration bug

`logs/verify_iso_E7.log`: decode w32 + verify (32,4) traces parked; round 1 = 8 x {`import_gdn_slot(mode=fillcache)`
(the eager in-place fill_cache slot write + tap row writes + packed-history repack) + eager `import_kv_blocks`} into
slots 8..15 (2.27 s), then 10 decode w32 replays (34.6 ms, fine) and 10 verify (32,4) replays (77.7 ms, fine) -- round 1
completed; the process went silent in **round 2** (no round-2 line; PCIe ID 0 reads 0xffffffff). No prefill trace was
replayed in this process. E8 (traced importers) could not start (board down).

So the wedge is reproduced by GDN slot writes + KV block imports interleaved with w=32 decode/verify trace replays --
exactly what the served D engine does between decode replays (pd_transfer import_slot / import_kv_blocks). Together
with E6 (prefill-trace replays) the common factor is "eager device ops or other traces' replays that write the GDN
slots / KV blocks / their scratch while w=32 decode + verify traces are parked". The w<=8 equivalents were clean
(3 x 8-user prefill_paged_slots between (8,8) replays). The decode w32 trace ALONE with 32-user prefills was the very
first wedge pattern (verify_full_stub), so the verify trace is not required either.

Priority: the address audit with ttnn's trace allocation tracker (`TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_TRACEBACKS=1`: `execute_trace` raises BEFORE the replay with the list of live buffers allocated
while a trace was active that the replay would corrupt, so it cannot wedge) on the E7 flow, then fix by allocating those
buffers before the captures. Run: `VERIFY_ISO=E7 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1`.

## Tracker audit (ttnn UnsafeAllocationTracker)

`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1` makes `ttnn.execute_trace` raise before a replay
that would corrupt a live buffer allocated while a trace was active. First run on the E7 flow
(`logs/verify_iso_E7_tracker.log`): it raised at the very first prefill-trace replay (the warm 1-user
`prefill_paged_slots`, i.e. the SERVED warm-up order itself) with 126 buffers, all `program_cache:` entries of
`warmup_gdn_slot_write` (ConcatDeviceOperation x32, TilizeWithValPadding x30, UntilizeWithUnpadding x31 tap-row programs,
UpdateKVCache FILL, Clone) -- program-cache buffers, benign per the tracker's own hint (skip with
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=1`); the served path runs this exact order for hours.
`VERIFY_ISO_TRACKER_AUDIT=1` (test_verify_w32_isolation_scratch.py) turns the raise into log-and-SKIP so one process
audits every replay of the flow; results in `logs/tracker_audit_<ISO>.txt`.

## Proposed served-D check (run on all 8 chips by the coordinator)

Start the P/D stack's D engine (or the standalone TP=4 vLLM server on a half) with
`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=1` in the
engine's environment (read once at import). Then drive the E7 pattern through the served path: 32 concurrent
requests (bucket-32 decode trace live) with a continuous stream of new short requests so that P/D imports
(`import_gdn_slot` + `import_kv_blocks`) land between decode-trace replays (D side), or, on the monolithic server,
prefills of new requests between decode replays. The engine raises (request error, no wedge) at the first replay
that would corrupt a live buffer and names it; no raise over ~10 minutes = the served traces allocate safely and the
verify-step body is what introduces the corruptible buffer. Cheap to run and decisive.

## Root cause and fix (2026-09-24 16:40-16:55)

**Tracker audit** (`VERIFY_ISO_TRACKER_AUDIT=1`, `logs/tracker_audit_E7.txt`, program-cache buffers skipped): the only
non-program-cache buffers a replay would corrupt are the decode w32 reference's own tensors (its 4 inputs from
`prepare_inputs_decode`, allocated after the verify capture and refreshed before every decode replay, plus its trace
outputs) -- benign by construction. With program-cache buffers INCLUDED (`logs/verify_iso_E7_tracker.log`) the first
prefill-trace replay already flags 126 program-cache buffers of programs compiled after the prefill captures
(the served warm-up order compiles the slot-write set after the prefill captures too, so that pattern is tolerated in
practice). The decisive difference of every wedged process vs every clean one: **programs compiled AFTER a trace
capture whose replays the process then interleaves with those programs' eager use** -- the (32,*) verify body compiled
after the decode capture (or vice versa), the decode w32 body compiled after a w8 capture (timing18_88), the slot/KV
import programs compiled in round 1 after both captures (E7: round-1 replays, round-2 imports hang), the verify/decode
bodies compiled after the prefill captures (E6 and every exactness-flow wedge). A program compiled after a capture owns
device buffers (its kernel binaries / config buffers via the program cache) in that trace's freed-intermediate range;
the replay overwrites them and the next launch of that program -- eager (imports, slot writes) or via another trace --
hangs the device. Whether a given placement overlaps is size dependent, which is why only the w=32 (large) bodies and
the 4-plan processes hit it.

**Fix (test discipline, the same rule the served path applies with `warmup_decode_buckets`: "compile every decode width
before capturing any bucket trace")**: compile EVERY program the process will ever run before capturing ANY trace --
decode bodies at all reference widths, verify bodies of all plans, the import / slot-write programs, the prefill
programs -- then capture (`VERIFY_ISO_COMPILE_FIRST=1`, `DecodeRef.compile()/capture()`, and the exactness test's
compile-all block before the prefill warm-up). Verified: E7 (`logs/verify_fix_E7.log`) and E6 (`logs/verify_fix_E6.log`)
both **clean over 5 rounds** (slot/KV imports resp. prefill-trace replays interleaved with decode w32 + verify (32,4)
replays), where the same flows wedged before.

**Served-D implication**: the D engine's warm-up order already compiles its decode buckets before capturing and warms
the importers (`import_warmup`, traced importer) before the decode capture, which is why v5-v7 ran for hours. Any
program that is first compiled at request time on D (a new import bucket, a new prefill length bucket without the
masked-bucket trace, a verify body added after the decode warm-up) re-creates this hazard: when the verify step is
integrated it must be compiled inside the warm-up before `warmup_model_decode` captures. The proposed tracker run on
the served D (previous section) is the cheap guard.

## (32,4) exactness after the fix (2026-09-24 17:00-18:05): no wedge, but a FUNCTIONAL failure, unresolved

With compile-first ordering the (32,4) exactness runs complete (no wedge) but the committed streams are wrong:
`logs/verify_324_exact_kernel.log` (kernel GDN, batched attention), `verify_324_exact_offsets.log`,
`verify_324_exact_stub.log` (stub GDN + per-offset attention), `verify_324_dbg*.log`. Facts:
- Row 0 of the FIRST verify step is wrong for the same 20 of 32 users in every mode (kernel/stub GDN, batched/per-offset
  attention): users {2,3,4,5,6,7,10,11,12,14,15,18,20,23,26,27,28,29,30,31}; every user with s % 8 in {0, 1} is right.
  Deterministic within a process (3 repeats after re-prefills bitwise identical; eager == traced), wrong ids are a few
  repeating values (124324, 159029, 184827, 69267, 86088: argmax of a corrupted row).
- The plain decode after the same re-prefill is right for all 32 users, so KV / GDN state at the prompt positions is fine.
- (8,4) [R=32, fused path] and (8,8) [R=64] are bitwise exact in the same flow; zeroing `qkv_prev` (kernel contract, now
  `VerifyStep.reset_sequence()`) changes nothing.
- The step-1 diagnostic `tests/test_verify_step1_scratch.py` (same model, same prompts, same page tables, same compile /
  capture order, same 16-18 reference decode steps, same re-prefill, the SAME drafts (seed 1558) and positions) is
  **correct for all 32 users**, eager and traced -- 5 runs. The two flows differ only in code the device never sees.
So a w=32 verify body (R=128) produces row-position dependent garbage in one process flow and exact results in
another with identical inputs -> device state that the verify reads but the plain decode does not, and that differs
between the flows, or an allocation-dependent read (uninitialized memory / L1 CB overlap) in one of the R=128 ops.
Next: bisect by adding the exactness test's steps to the diagnostic one at a time (the reference loop length, the
controller, the debug wrapper), then dump the verify body's per-layer row-0 activations for a failing user vs a
passing user (rowdiag hook) in the failing flow.

## (32,4) row-0 failure: cause found (2026-09-24 18:45)

Tracker run of the failing exactness flow with program-cache buffers included (`logs/tracker_audit_exactness.txt`):
the only programs first compiled after a capture are the per-slot `_write_index` Concat programs of the 32-user
prefill's slot writes (31 of them, one per slot; the served path warms them for all 32 slots in
`warmup_gdn_slot_write`; the host-repack variant is not warmed) -- and the decode reference / verify bodies, because
the compile-first block had been LOST from the exactness test (a failed edit; restored in `3a0d3ef7a03`). Neither
explained the pattern: with the device pack (`QWEN36_GDN_HIST_DEVICE_PACK=1`) the failure was bit-identical.

Bisect (`logs/verify_324_rowdiag_flow3.log`, `verify_324_head.log`): the per-layer row-0 activations of same-prompt
users are bitwise identical through all 64 layers and the LM-head logits too; the device `ttnn.argmax` matches the
host argmax on every (device, row); but **`ttnn.max(logits, dim=-1)` on the `[1,1,128,62080]` bf16 TILE logits returned
a wrong maximum for all 512 (device, row) pairs**, so `combine_sharded_argmax` picked the wrong shard for every user
whose true maximum sits on another shard -- exactly the users `s % 8 in {2..7}` here, an artifact of which shard holds
each prompt's top token. The same op was right in the step-1 diagnostic process and at R <= 64; text_demo already
avoids the plain row max with a padded two-stage `[1,B,ceil(V/32),32]` reduce (`_maxval_dev_b`). Fix: that formulation
in `verify_step.max_rows_tile_parallel` (default; `QWEN36_VERIFY_MAX=plain` restores the single reduce for a repro).
The wide single-row `ttnn.max` misbehaving only in some processes is a ttnn reduce_w bug worth filing (reproducer:
this flow with `QWEN36_VERIFY_MAX=plain`).

Status: the (32,4) rerun with the fix could not open the mesh twice ("waiting for physical cores to finish: 13-3 ...
Try resetting the board", `risc_firmware_initializer.cpp:1542`, `logs/verify_324_exact_fix2/3.log`): a Tensix core
left running by the previous watchdog-killed run (`verify_324_exact_fix.log`, hung after the verify capture);
tt-smi still lists 8 devices. Needs a half-A reset, then: (32,4) and (32,8) exactness, and the timing table's head
section re-measured with the two-stage max.
