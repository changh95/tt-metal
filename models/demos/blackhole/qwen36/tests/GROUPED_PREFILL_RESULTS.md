# Grouped traced prefill (round-4 item 18 / round-3 item K) and the conv1d block-major kernel (item 11)

Track "grouped prefill + conv1d block-major", 2026-09-29, P150x8 box, half A (chips 0,1,6,7, TP=4) for the tests, all
eight chips for the served A/B. Commits (tt-metal, branch qwen38-pd-disagg): `b86337530bb` conv1d block-major,
`f9dca7d3fa8` grouped traced prefill. Every number below has its log under `logs/` of the experiment root.

## 1. conv1d block-major work order (qkv_causal_conv1d_silu, item 11)

Patch `logs/perf_r4_rank11_conv1d_blockmajor.patch` (kernels + the factory's `{"Mt", Mt}` compile-time arg) applied and
built in ONE host rebuild window (Release + Tracy kept; `python_env` imports ttnn). Work item -> `(block = work / Mt,
mt = work % Mt)` so a core's contiguous items stay in one channel block and the 4 x block_ct tap tiles load once per
block change.

Bit-exactness (`tests/test_conv1d_blockmajor_scratch.py`, save on the unpatched tree -> check on the patched tree,
`torch.equal` on q/k/v, per-device concat): every served shape of the TP=4 GDN prefill (C = 2560, HiFi4 fp32-acc):
(T, channel_chunk) = (128, 32) (256, 64) (512, 128) (1024, 256) (2048, 512) with the TILE input and the ROW_MAJOR-input
kernel at (2048, 512): **18/18 EXACT** (`logs/r4_conv1d_op_before.log`, `logs/r4_conv1d_op_after.log`,
`logs/r4_conv1d_op_after1350.log`). Model level (`test_prefill_kernels_ref_scratch.py` save -> check with
`QWEN36_PK_BITEXACT=1`): prefill logits at ISL 128 / 512 / 4096 / 8192 max|d| = 0 (0 mismatches of 248,320 each) and
identical greedy tokens over 8 traced decode steps (`logs/r4_conv1d_model_ref_save.log`,
`logs/r4_conv1d_model_ref_check.log`).

Op timing (one trace of 48 back-to-back ops, 30 replays, per-op wall us, median):

| shape (TILE in unless noted) | before (1350 MHz) | after (1200 MHz pin) | after (1350 MHz) |
|---|---|---|---|
| 2048 x 512-channel chunk (the 2048-token chunk body) | 238.5 | 214.5 | **191.4** (-20 %) |
| 1024 x 256 | 116.0 | 108.6 | 97.3 (-16 %) |
| 512 x 128 | 57.9 | 55.3 | 50.0 |
| 256 x 64 | 32.2 | 32.6 | 29.1 |
| 128 x 32 (bucket 128) | 19.4 | 21.2 | 18.3 |
| 2048 x 512, ROW_MAJOR input | 278.0 | 238.8 | 222.5 (-20 %) |

The first "after" run sat under a 1200 MHz AICLK pin another agent's served stack had left (all chips read 1200 idle,
`scripts/force_aiclk.py 0` released it before the 1350 rerun), so the equal-clock column is the one to quote. The gain is
smaller than the plan's 145-175 us projection: the plan's own Tracy rows (TRISC 226 us of a 228 us op) show the TILE
compute, not the tap reload, bounds the op; the reload was mostly hidden behind it.

Harness (`scripts/profile_tp8.sh`, HALF=A NO_TRACY=1, min of 3): before `logs/tracy_conv1d_before.log`
`prefill_128=135.1 prefill_4096=512.0 prefill_8192=996.5 decode_w1=22.58 ms` (1350 MHz); the 1200 MHz-pinned after run
(`logs/tracy_conv1d_after.log`: 138.6 / 551.1 / 1035.4 / 24.9 ms -- decode, which does not run this op, +10 %) is a clock
artifact; the equal-clock after run `logs/tracy_conv1d_after1350.log`: `prefill_128=131.2 prefill_4096=504.8
prefill_8192=993.0 decode_w1=22.56 ms` (medians 132.5 / 513.1 / 1007.0 vs 135.6 / 517.0 / 1005.9 before): -3.9 ms at 128,
-7.2 ms at 4096, within the +-5 ms run-to-run band at 8192 (the plan projected -10..-15 ms there: 48 layers x (238.5 -
191.4 us) x 4 chunks = 9 ms of device time, partly hidden behind the chunk trace's host-side work); decode unchanged
(control: it does not run this op).

## 2. Grouped traced prefill (item K / rank 18)

### What it is
`QWEN36_PREFILL_GROUP_TRACE` (opt-in: unset/0 off; 1 = `128:2,4,8;256:2,4,8;512:2,4`; or a `bucket:B,B;...` spec).
`prefill_paged_slots` plans the step's users into same-bucket groups of B <= 8 (`masked_bucket_trace.plan_prefill_groups`)
and replays ONE captured body per group over B*bucket user-major rows: embedding / norms / MLP / GDN in- and out-proj
batched over all rows, attention per row over per-row page tables, the GDN causal conv per row (the fused KDA op is
[1, T, C]-only), the chunk recurrence per row (`QWEN36_PREFILL_GROUP_ROWSCAN=1`: the per-user programs; the batched
BH = B*Nv variant runs another scan kernel whose fp32 state differs at ~1e-5 in layer 0, `logs/grp_lay1.log`). The B-row
GDN state is assembled in-trace into per-B ROW_MAJOR buffers the pooled host snapshot reads per row; the P/D producer's
`pd_gdn_capture` / `pd_stage_hook`, the host slot write and the speculative prefill hooks (per-row views of the grouped
residual and aux copies) are served per user. Rows past the real users are 1-token dummies over the scratch KV block.

### The round-3 hang, localized and fixed
Round 3 hung on the first grouped replay twice (`logs/round3/itemK_test.log`, `logs/grp_test_all.log`). Cause (the class
of `tests/VERIFY_W32_AUDIT.md`): every grouped body was compiled -- and its output / lazily created buffers allocated --
AFTER the chunk and bucket captures, inside those traces' freed-intermediate ranges. The tracker
(`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1`: `execute_trace` raises listing the buffers a replay
would corrupt) drove the fix, run by run on half A:

| run | tracker report | change |
|---|---|---|
| `logs/grp_trk1.log` | 8 buffers: the 8 grouped traces' output tensors, allocated during capture | output buffers pre-allocated in the eager pass, written in-trace with `ttnn.copy` |
| `logs/grp_trk2.log` | 19 program-cache buffers: the test's request-time programs (host snapshot untilize, logits readout, slot write) compiled after the captures | test compiles them before the captures (one eager per-user prefill) |
| `logs/grp_trk3.log` | 14 slot-1 programs of the host slot write (slice/concat/fill_cache are keyed by the slot) | test warms the write for every slot |
| `logs/grp_trk4.log` | **clean at every replay** of the b128_B8 set (per-user bucket replays, grouped replay, decode) | -- |
| `logs/grp_smallm0.log`, `logs/grp_trk5.log` | clean, PASS | -- |

Model side: `_prepare_prefill_group_traces` (called from `_capture_prefill_trace_chunked_tp` before the chunk capture)
allocates the shared constants, every (bucket, B) input set, the per-B state outputs and the residual / aux output buffers
and runs one eager pass per body + readouts; `capture_prefill_group_traces` only captures (largest body first, skipped
when the TRACE region is short). Trace-region cost at TP=4: 128x8 56.9, 256x8 58.4, 512x4 43.0, 128x4 40.5, 256x4 41.2,
512x2 32.8, 128x2 30.9, 256x2 31.9 MiB (`logs/grp_trk4.log`).

The tracker still flags the PRE-EXISTING masked-bucket order at buckets >= 256 (each bucket's input buffers are allocated
between captures; `logs/grp_full_default.log`: 11 buffers of `_alloc_masked_bucket_bufs(256)`). Those inputs are re-DMA'd
before every replay, so the class is benign in practice (weeks of serving), but allocating all bucket inputs before the
first capture would make the whole flow tracker-clean; not changed here (production path, out of this item's scope).

### Correctness
`tests/test_prefill_grouped_trace.py` (real-text windows of the 4k sample document, per-row offsets; the chat prompt in
row 0), 5 sets: b128_B8 = lengths [29, 5, 17, 64, 100, 127, 128, 33], b256_B4 [129, 200, 256, 140], b512_B2 [300, 511],
b128_B8_full [128 x 8], mixed [29, 40, 128, 129, 250, 600] (-> groups (128, 4) + (256, 2) + one single). Per user: next-token
logits, GDN recurrent state and conv taps `torch.equal` to the per-user path, and the greedy continuation over 8 decode
steps identical:

`logs/grp_full_notrk.log`: **every set exact -- logits / rec / taps 8/8/8, 4/4/4, 2/2/2, 8/8/8, 6/6/6; tokens identical
for all 28 users** (min PCC 1.000000 everywhere). `logs/grp_trk5.log` (same, b128_B8 under the tracker): exact, PASS.

Why exactness needs `QWEN36_PREFILL_SMALLM_MAX=0`: with the serving default (128: item J's AG + 1D matmuls at <= 128 rows,
a different K accumulation order) the per-user bucket-128 body is not bit-identical to the same rows inside a 1024-row
2D body: layer-0 taps exact, layer-0 fp32 state 1.5e-5, cascading to logits PCC 0.9995 / max|d| 0.3-0.6 by layer 47
(`logs/grp_lay1.log`, `logs/grp_full_default.log`) and greedy flips on 5/8 random-token and 2/8 real-text prompts. A user's
output must not depend on who shared its prefill step, so grouping forces `QWEN36_PREFILL_SMALLM_MAX=0` for the singles
too (per-user bucket-128 replay ~+25 ms; `QWEN36_PREFILL_GROUP_KEEP_SMALLM=1` keeps item J and PCC-level equivalence).
Control: `logs/grp_smallm0.log` (SMALLM_MAX=0 forced by env before the model-side coupling): exact 8/8/8, tokens 8/8.

`tests/pd_transfer_repro.py`: 24/24 tokens match with the knob on (`logs/repro_grp_on.log`) and off
(`logs/repro_grp_off.log`) -- the refactored `prefill_paged_slots` singles path and the P->D transfer are unchanged.

### Step cost
`test_grouped_prefill_step_timing` (P/D producer configuration: host snapshots parked, no slot write; N = 8 users x 128 or
100 tokens; min of 4 per mode, alternated; AICLK 1200 pinned by the foreign pin):

| N=8 x 128 tokens | per-user path | grouped (one (128, 8) replay) |
|---|---|---|
| singles on the 2D path (the coupling above), `logs/grp_timing2.log` | 690-697 ms | **265 ms** |
| singles on the small-M path, `logs/grp_timing.log` | 560 ms | 258 ms |

i.e. 33 ms per user instead of 70-87. `[PREFILL_TIMING] N=8 ... groups=[(128, 8, 8)]` lines carry the same split per
served step.

### Served A/B (all 8 chips, P/D stack, v12 env `QWEN36_SPEC_MTP=1 QWEN36_SPEC_DRAFTER=hybrid QWEN36_SPEC_K=7 QWEN36_PD_SHM=1`)
Knob on P only (D never prefills; its trace region holds the decode / speculative traces). `vllm bench serve`, random
dataset, 32 users: see section 4 (filled from `logs/points_grp_grp_off/status`, `logs/points_grp_grp_on/status`).

## 3. Chip / process rules followed
Device gate `scripts/chips_gate.sh A` before every run; one rebuild window (`run/rebuild_window` 02:36-02:46, build waited
for the other agent's served stack to exit); no tt-smi resets; no Tracy device profiling; the foreign 1200 MHz pin was
released (`scripts/force_aiclk.py 0`) only with the box idle, before the equal-clock timings. Kernel sources were edited at
02:36 while the other agent's stack (started 02:33) was already serving -- it had compiled its conv1d kernels before that
and ran clean, but the gate must be re-checked immediately before touching kernel sources.

## 4. Served A/B numbers (P/D stack on all 8 chips, knob on P only, AICLK unpinned)

### 4a. v12 speculative env (`QWEN36_SPEC_MTP=1 QWEN36_SPEC_DRAFTER=hybrid QWEN36_SPEC_K=7 QWEN36_PD_SHM=1`)
`vllm bench serve`, random dataset, 32 users; the "128-token" prompts carry 179 tokens with the chat template (bucket
256), the 1024-token ones 1075 (bucket 2048: never grouped). `logs/points_grp_grp_off/`, `logs/points_grp_grp_on/`,
P logs `logs/pd_P_grp_off.log` / `logs/pd_P_grp_on.log`:

| point | knob off: mean / median / p99 TTFT ms, tput | knob on | reference v12 / v10 mean |
|---|---|---|---|
| 128/128 x32 (256 requests) | 842.9 / 378.4 / 4827, 678.5 tok/s | 920.4 / 879.8 / 3657, 676.4 tok/s | 1066 / 723 |
| 1024/128 x32 (128 requests) | 7804 / 8343 / 11755, 335.9 | 7539 / 8147 / 11896, 335.3 | 7335 / 5908 |

P-side `[PREFILL_TIMING]` of the same steps (the MTP + DFlash2 prefill hooks active, served per row on the grouped path):
bucket 256, N=8: per-user `prefill=` 758-790 ms (total 928-1437) -> grouped (256, 8) 435-506 ms (total 618-1019);
bucket 128 (the concurrency check's 19-30-token prompts), N=8: 590-597 ms (total 744-764) -> 320-336 ms (total
462-511); N=7: 530-532 -> 304-410. So the device time of an 8-user step drops 40-45 % -- but the client-side TTFT does
not follow: since the per-user release (round-4 rank 2) a per-user step hands user k to D after k x ~95 ms, while a
grouped step hands all eight over at once after ~600 ms, so the median TTFT doubles (378 -> 880 ms), the p99 improves
(4.8 -> 3.7 s) and the mean is a wash (843 -> 920, inside the run-to-run band of this thermally loaded point). Grouping
helps TTFT only when the grouped body is much more than ~4x faster than one per-user replay (it is 1.5-1.8x per user
here: 128 x 8 = 265 ms vs 8 x 58 ms; 256 x 8 = 435 vs 8 x 80 ms), or when P device time is the bottleneck (it is not
at 32 users x 128 tokens: D-bound, throughput unchanged at ~677 tok/s). The plan's -0.3..-0.5 s standalone projection
assumed a ~2x-per-user grouped body and predates the per-user release.

`scripts/pd_concurrency_check.py` (conc 1-32, 48 greedy tokens, PD proxy vs the D instance): 14 mismatches with the
knob OFF and 17 with it ON in this speculative env -- the baseline itself no longer self-matches on today's HEAD
(v12's gate was ALL MATCH; commits after v12 changed the drafting rules: `2acf664d24a` adaptive DFlash2 draft length),
so this run cannot attribute anything to the grouped path; section 4b isolates it.

### 4b. Plain P/D (no speculative decoding; `QWEN36_PD_SHM=1`)
`logs/points_grp_nospec_off/`, `logs/points_grp_nospec_on/` (status files carry the RESULT lines):

| point | knob off: mean / median / p99 TTFT ms, tput | knob on |
|---|---|---|
| 128/128 x32 | 953.3 / 338.5 / 5260, 688.4 tok/s | **859.1 / 653.4 / 3724, 707.3 tok/s** (mean -10 %, p99 -29 %, median +93 %, tput +2.7 %) |
| 1024/128 x32 | 4267 / 4032 / 9111, 405.2 | 4306 / 4093 / 9147, 403.3 (bucket 2048: no grouping; tie) |

`pd_concurrency_check` (PD proxy vs the D instance decoding locally, 48 greedy tokens, conc 1-32): knob off 14
mismatches -- conc 1/2/4 all match, conc 8: 6/8, 16: 12/16, 32: 24/32, always prompts 6 and 7 ("Give three tips",
"Summarize ... Romeo") from token 91 / 158 on: a pre-existing width >= 8 divergence of today's HEAD that is identical
with the knob on, and identical in the speculative env of 4a (the v12 gate had ALL MATCH; not touched by this track).
Knob on: 17 mismatches = the same 14 plus prompt 0 ("haiku", from token 38) at conc >= 8 -- the documented small-M vs
2D difference: P (grouping on) forces `QWEN36_PREFILL_SMALLM_MAX=0`, while the reference D instance still prefills
locally on the small-M path. `logs/points_grp_dsm0_on/` reran the knob-on stack with `QWEN36_PREFILL_SMALLM_MAX=0` on
D as well: `RESULT 128,128,32,256 completed 256 mean_ttft_ms 854.5 median_ttft_ms 690.2 p99_ttft_ms 3487.3 mean_tpot_ms 37.3 tput 705.2`; 1024/128 x32 4255 / 3956 / 9128 ms, 406.7 tok/s; but the concurrency check got WORSE (24 mismatches,
the haiku prompt now differing from token 38 even at conc 1, where the knob-off stack matched). So the check's D-local
reference is itself a near-tie-sensitive quantity: changing D's prefill programs (small-M -> 2D) moves the reference
continuation of the haiku prompt, and the PD output (P 2D prefill -> transfer -> D decode) lands on the other side of the
tie. The in-process tests are the exactness evidence (grouped == per-user, torch.equal, 28 users; pd_transfer_repro
24/24); `pd_concurrency_check` on today's HEAD is not a usable gate for this item (its knob-off baseline already fails at
conc >= 8) and the P-2D-vs-D-local disagreement at conc 1 deserves its own bisect (transfer/import vs prefill programs;
not started here).

### Verdict
The grouped traces are allocation-safe (tracker-clean flow, no hang in ~40 grouped replays across the runs above) and
bit-identical per row to the 2D per-user path; they cut the P device time of an 8-user short-prompt step by 40-55 %. At
32 users the client-side gain is small (plain stack: mean TTFT -10 %, p99 -29 %, throughput +2.7 %; median worse because
the per-user release already hands early users to D) and D-bound in the speculative stack (a wash). The knob therefore
stays OPT-IN (`QWEN36_PREFILL_GROUP_TRACE` unset = off, the published v12 bundle unchanged). A queue-depth-aware policy
(group only when >= 6 same-bucket users wait; B <= 4 to shorten the hold) is the natural next step if P becomes the
bottleneck (short-prompt bursts, prefill-heavy workloads).
