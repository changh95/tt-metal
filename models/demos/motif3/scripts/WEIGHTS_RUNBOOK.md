# Motif-3 TT weight cache: conversion and streaming runbook

Scope: build the bfp8/bf16 TT weight cache that `MotifModel` / the vLLM bridge load (WAVE_A_REVIEW §5.9 CONV-1..4,
design §2.3.11, README CONVENTIONS §7), and stream the BF16 checkpoint through the disk without going below the
free-space floor (60 GB by default, §4).

| File | What it does |
|---|---|
| `scripts/convert_weights.py` | converter (CONV-1): builds `global` and `L<nn>` parts with the serving constructors, mock cluster (default) or device |
| `scripts/stream_weights.py` | disk-limited pipeline: download → sha256 → C2 golden (layers ≥ 36) → convert → verify → delete consumed shards (gated, §5) |
| `tt-metal/models/demos/motif3/tests/test_weight_cache.py` | 11 host tests (rules, pipeline gate with a fake runner, mock conversion) and 2 device tests (device-built == mock-built bytes; cache-built == source-built outputs, bitwise) |

Paths below: `M=/home/ttuser/hchang/experiments/motif-3`, `S=$M/scripts`.

---

## 0. Quick reference

```bash
M=/home/ttuser/hchang/experiments/motif-3; S=$M/scripts
# status of the cache (no device, no mesh; 3 s): complete / verified / built by the current code, serving cache
$S/hostrun.sh -- python $S/convert_weights.py --status --layers all --globals            # add --json for machines
# convert whatever is local (host only, mock cluster, no device lock), resumable; stops at the 60 GB floor (§4)
$S/hostrun.sh -t 7200 -n convert -- python $S/convert_weights.py --layers 0-35 --globals
# the same on the real mesh (only if you need it; same bytes, see §2)
$S/devrun.sh -t 2400 -n convert -- python $S/convert_weights.py --target device --layers 0-2 --globals
# re-check complete parts (cache-only rebuild with a raising source + sha256 re-hash)
$S/hostrun.sh -n convert_verify -- python $S/convert_weights.py --verify-only --layers all --globals
# the streaming pipeline: plan first, then run (plain python: it imports no ttnn and opens no device)
python3 $S/stream_weights.py --dry-run --delete-converted
mkdir -p $M/logs/stream && nohup python3 $S/stream_weights.py --delete-converted > $M/logs/stream/stream_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Done 2026-10-02 03:40-04:52** (real deleting run, §5 "Real run"): the serving cache holds globals + L00-L52, all complete + verified + current code (343.75 GB); BF16 left: layers 0-3 + shard 104 (29.65 GB); both goldens (`goldens/c2`, `goldens/c2_fp32`) reach layer 52 with their final heads; 117.8 GB free.

Two safety defaults decide what the pipeline does by default (as of 2026-10-02 before the run, 60.6 GB free):

* `--keep-bf16 0-35` (default): the BF16 of the bring-up layers is never deleted, because the acceptance tests read
  it (§5 "Consequences"). With it, nothing local is deletable and the stream stops at the first MoE conversion (L03,
  exit 3; dry run below). Releasing them (`--keep-bf16 none`) is a sign-off decision, or wait for `/data` (D4).
* `--min-free-gb 60` (default, both scripts): §4.

Without `--delete-converted` the pipeline never deletes a file.

---

## 1. What a part is, and what lands on disk

Cache layout (README §7): `<root>/<tag>/mesh4x8/{global,L00..L52}/<name>__<mapping>_dtype_<D>_layout_<L>.tensorbin`,
root `MOTIF3_TT_CACHE_PATH` > `TT_CACHE_PATH` > `$M/tt_cache` (the serving order), tag
`motif3-2ed2ed5c-c1-e8s8d8a16r16m16l16v16` (checkpoint revision, `CACHE_FORMAT_VERSION`, dtype policy). The mesh shape
is part of the path. The converter always builds the serving mesh `(4, 8)` unless `--mesh` says otherwise; it ignores
`MESH_DEVICE` (the plugin preset `BH-Galaxy` maps to (8, 4), whose `mesh8x4` directory a (4, 8) server never opens).
`--status` reports whether the directory is the serving cache (mesh (4, 8), pinned revision, default dtype policy).

| Part | Built by (serving constructors, `cache=True`) | Files | Bytes (measured) |
|---|---|---:|---:|
| `global` | `MotifEmbedding` (replicated table) + `MotifLMHead(vocab_split="mesh")` | 3 | 3.607 GB (`embed.weight__rep` 1.80, `lm_head.weight_mesh__dp0tp3` 1.80, `final_norm.weight__rep`) |
| `L00`, `L01` (dense) | `MotifDecoderLayer` = 2 `MHCSite` + `MotifAttention` + 2 RMSNorms + `MotifDenseMLP`, + the stock mHC constants | 30 | 0.355 GB (attention 192.6 MB, dense MLP 160.5 MB, mHC 2.1 MB + 17 KB motif + 36 KB stock constants, norms 17 KB) |
| `L02`..`L52` (MoE) | `MotifDecoderLayer` = 2 `MHCSite` + `MotifAttention` + 2 RMSNorms + `MotifMoE` + `MotifSharedExpert`, + `RouterLogitsFP32` + the stock mHC constants | 46 | 6.656 GB (experts gate_up 4.278 + down_x1 2.139, attention 192.6 MB, shared 16.7 MB, routers 6.3 MB, local ids 1.6 MB, mHC 2.2 MB) |

Total for 53 layers + globals with the default options: 3.607 + 2 × 0.355 + 51 × 6.656 = **343.8 GB**.

**Variants** belong to one part kind each: the globals take `--lm-head-split` / `--embedding`, MoE layers `--router` /
`--mhc-sinkhorn`, dense layers `--mhc-sinkhorn`. The serving defaults (router `composite` and `cfg.router_logits`,
Sinkhorn `motif` and `cfg.mhc_sinkhorn`, LM head `mesh`, embedding `replicated`) are **always** built; an option only
adds the non-default value (`--mhc-sinkhorn stock` is the same as `both`). Defaults: `--router both` (adds
`moe.router.weight_fp32k_v1`, 3.1 MB per MoE layer: decision D1, `MOTIF3_ROUTER_LOGITS=exact_fp32`) and
`--mhc-sinkhorn both` (adds `{site}.{alpha,bias,lo,hi}_row`, 35.8 KB per layer: decision D2 / MHC-3, the stock fallback),
so either choice of those open decisions loads without the BF16 source. `--lm-head-split tp|both` adds
`lm_head.weight_tp__tp3` (1.80 GB, one copy per TP shard set like every `tp` file; the globals become 5.41 GB),
`--embedding sharded|both` adds `embed.weight__tp1` (1.80 GB); both: 7.21 GB. These sizes are in the guards'
estimates. A part is complete only if its recorded variants **of its kind** cover the requested ones; a part that
lacks only a variant is extended in place (below), not rebuilt. Variants only ever add files: a rebuild keeps every
variant the part already records (delete the part directory to drop one). A variant can only be added while the
layer's BF16 is on disk.

Per part the converter writes, besides the `.tensorbin` files:

* `.complete`: `weights.mark_layer_cached(cfg, part, files)` = `{"layer", "tensors": [file names], "version": tag}`.
  `MotifModel(cache="auto")` keys on its existence: a marked part is built with `cache=True` (each tensor from its
  file); an unmarked part loads from BF16 without writing anything. A marked part with a **missing or unreadable**
  file is not refused: that module falls back to the BF16 source for the tensor (opening the checkpoint) and writes the
  file again (no disk guard). `--status` and `--verify-only` find such parts; the converter rebuilds them.
* `.convert.json` (format `motif3-convert/2`): seconds per module bucket, bytes, per-file bytes / cache name /
  mapping / dtype / sha256 / reused, the variants of the part's kind, the target (mock / device), the **code
  fingerprint** (sha256 of the motif3 sources that produce cached tensors — `tt/{weights,model_config,decoder,mhc,
  attention,polynorm,mlp,moe,embedding,lm_head}.py`, `tt/kernels/{__init__,router_fp32,sinkhorn_motif}.py` — and of the
  ttnn build: `ttnn/ttnn/operations/core.py`, `ttnn/ttnn/distributed/distributed.py`, `_ttnn.so`, `_ttnncpp.so`,
  `libtt_metal.so`), removed stale files, and the verification record (`verify.ok`, `files_hashed`, load and re-hash
  seconds). Format /1 records (before 2026-10-02 02:23) are still read.

How a part is built (crash-safe and resumable):

1. skip when complete (marker with this tag, `.convert.json`, every listed file present with its recorded size,
   variants of its kind covered); `--force` rebuilds from BF16; a marker without `.convert.json` (e.g. written by
   `tt/model.py:convert_weights`) is rebuilt; a part that lacks only variants reuses its files (step 3);
2. preconditions, before anything in the part directory changes: every BF16 tensor of the part on disk (else exit 4),
   and the estimated bytes (variants included) x 1.1 must leave `--min-free-gb` (default 60) free (else exit 3);
3. build into `<cache dir>/.convert_staging/<host>-<pid>/` (next to the parts: same filesystem even when the tag
   directory is a symlink into another volume) while `weights.as_tensor` is recorded (every cache file the
   constructors request, hit or miss, and its seconds; an upload with `cache_name=None` is an error: serving would need
   the BF16 source for it), then free the device tensors. For a variant addition the part's files are hard-linked into
   staging first, so the constructors find them (cache hits): only the missing variant files are built and only their
   BF16 tensors are read (12 tensors for the stock constants of a layer), and the existing files are not rewritten
   (same inodes, tested);
4. commit: remove the part's `.complete` and `.convert.json` first (until step 5 the part is incomplete:
   `MotifModel(cache="auto")` loads it from BF16, a crash leaves it to be rebuilt), `os.replace` every file into place,
   delete stale `.tensorbin` files no constructor requested (renamed cache names), sha256 + fsync every file and fsync
   the directory;
5. write `.convert.json` and `.complete`, each atomically (temp file + rename) and fsync'd. A part records the code
   fingerprint of the code that built **all** its files; a variant addition onto files of other code records none;
6. verify (default): rebuild the part from the cache alone with `RaisingSource` (any `get` / `has` / `in` raises),
   require every load to be a cache hit and every listed file to be loaded, then re-hash every file against the
   recorded sha256. On the mock cluster the load parses the files but the (no-op) device write never reads their data
   pages, which is why the re-hash is part of the verification. It proves completeness and integrity relative to the
   conversion; that the mock bytes are the device bytes is shown by `test_device_convert_roundtrip` (§2), and that they
   are the current code's bytes by the fingerprint. A complete part whose verification fails is rebuilt once.

One converter per cache directory (`<cache dir>/.convert.lock`; the deleting pipeline takes it too). The converter
refuses `--target mock` while `/dev/tenstorrent` is visible (run it under `hostrun.sh`), and `--target device` unless one
of its ancestor processes is `scripts/devrun.sh` and the device lock is held (another job's lock does not count).
`MOTIF3_NUM_LAYERS` (truncated bring-up) does not truncate the converter.

---

## 2. Where conversion runs: the mock cluster (decision, with evidence)

`ttnn.as_tensor(cache_file_name=...)` converts on the host and dumps the file **before** moving the tensor to the
device (`ttnn/ttnn/operations/core.py:871-895`), and on a mock target (`TT_METAL_MOCK_CLUSTER_DESC_PATH`) every mesh
command-queue write, read and program enqueue returns early (`tt_metal/distributed/fd_mesh_command_queue.cpp`,
`write_shard_to_device`, `enqueue_read_shard_from_core`, `finish_nolock`). The serving constructors therefore run
unchanged on a mock 32-chip Blackhole Galaxy (`single_bh_galaxy_clus_desc.yaml`: one harvested Tensix column, i.e.
the same 12 x 10 grid; `open_motif_mesh((4, 8))` opens it in 0.7 s, fabric init skipped) and write the cache.

Evidence that the mock-built cache is the device cache:

| Check | Result |
|---|---|
| Probe, layer 0 (20 files, 355 MB) built on the mock mesh and on the real mesh (`logs/dev/20261002_002141_conv_probe_dev_l0.log`) | 20 / 20 files byte-identical (sha256) |
| `test_device_convert_roundtrip`, 01:16 (`logs/dev/20261002_005715_weight_cache_device_final.log`): layers 0-2 + globals converted on the real mesh vs the mock-built real cache | **85 / 85 files byte-identical** (sha256): every mapping (`rep`, `tp*`, the EP `dp0tp1` experts, the `dp0tp3` LM head), bf16 / bfp8 / fp32, TILE / ROW_MAJOR |
| the same run: modules built from the cache alone vs from BF16 (`cache=False`), on the device | **97 / 97 device weight tensors bitwise equal** (all 32 chips; the 2 multi-GB ones on 4 sampled chips) and **13 / 13 forward outputs bitwise equal**: prefill S = 128 (real tokens) embedding -> L00 -> L01 -> L02 (with KV fill) -> LM-head tile, one 32-lane decode step (heterogeneous positions, 1 inactive lane), the 3 KV caches. Building the 5 objects from the cache alone took 1.4 s |
| `test_device_model_auto_loads_mock_cache`, same run: `MotifModel(cache="auto")` on the mock-built cache with a raising source | globals + L00-L02 load in **5.0 s** (globals 2.4 s, L02 2.3 s) without touching the checkpoint; a 3-layer prefill gives finite logits |
| The rebuild of the real cache with today's code and the new defaults (`--force --layers 0-2 --globals`, 02:23, `logs/host/20261002_022320_convert_real_force_L0-2_globals_v2.log`) | 85 / 85 files bit-identical to the 00:34 build; + 24 stock-constant files; all 4 parts verified, fingerprint `0d405672da3c59ed` |
| `test_host_mock_conversion`: a fresh mock conversion of layer 0 vs the real cache | 30 / 30 files byte-identical |
| `test_device_convert_roundtrip`, 02:28, new defaults (`logs/dev/20261002_022806_weight_cache_device_v2.log`) | **109 / 109 files byte-identical** device-built vs mock-built (the 24 stock-constant files included); **134 / 134 device weight tensors** bitwise equal cached vs BF16-built (incl. the exact-fp32 router of L02 and the 6 stock mHC sites); **26 / 26 forward outputs** bitwise equal and all finite: the chain above + the exact-fp32 router logits + stock mHC pre / post of every site. Device conversion 69.7 s, cache-only build 1.5 s |
| `test_device_model_auto_loads_mock_cache`, same run | `MotifModel(cache="auto")` with a raising source: globals + L00-L02 in 1.5 s for the defaults and in 1.5 s with `sinkhorn="stock"` + `router_logits="exact_fp32"` (D2 / D1 from the cache alone); both prefills give finite logits |
| Earlier verifier (`analysis/verify_memory_disk_perf.md` (f)) | a mock-built sharded bfp8 file loads on the real mesh with PCC 1.000000 |

So the default is `--target mock`: conversion takes no device lock and never blocks device work. `--target device`
remains (same files) for runs that want the device load as part of verification.

---

## 3. Commands

```bash
M=/home/ttuser/hchang/experiments/motif-3; S=$M/scripts
# convert (host only); --layers: 0-52 | all | 3,4,10-12 | 36- | none; --globals adds the globals
$S/hostrun.sh -t 7200 -n convert -- python $S/convert_weights.py --layers 3-35
# into another root (e.g. TIS's volume, §7) -- or export MOTIF3_TT_CACHE_PATH / TT_CACHE_PATH
$S/hostrun.sh -t 7200 -n convert -- python $S/convert_weights.py --layers all --globals --cache-root "$TIS_TT_CACHE"
# a JSON report of the run (per part: stats + verification)
$S/hostrun.sh -n convert -- python $S/convert_weights.py --layers 3 --report $M/logs/convert/L03.json
# status / verify / rebuild
$S/hostrun.sh -- python $S/convert_weights.py --status --layers all --globals
$S/hostrun.sh -n convert_verify -- python $S/convert_weights.py --verify-only --layers 0-2 --globals
$S/hostrun.sh -n convert -- python $S/convert_weights.py --force --layers 2
# a global variant: extends the globals in place (+1.8 GB), the layers are untouched (per-kind variants)
$S/hostrun.sh -n convert -- python $S/convert_weights.py --globals --layers none --lm-head-split both
```

Exit codes: 0 ok, 1 error, 2 usage, 3 disk guard, 4 BF16 tensors missing, 5 verification failed.

Logs: `hostrun.sh -n NAME` tees to `$M/logs/host/<ts>_NAME.log`; one line per part, e.g.
`L02 (swa-attn/moe): 46 files, 6.656 GB in 40.2 s (58 BF16 tensors, 12.26 GB; ... experts 37.7, attention 1.4, router
0.7 ...); move 3.01 s, sha256 + fsync 5.0 s; 60.4 GB free` and `L02: verify from the cache alone (mock): OK (48 tensors
loaded with a raising source in 0.2 s, 46/46 files re-hashed in 5.0 s)`; a variant addition logs
`L00: adding variants {'mhc_sinkhorn': ['stock']} (reusing its 22 files)` and `8 new + 22 reused files`.

---

## 4. Disk math

**Floor: 60 GB free (default of both scripts; decision recorded 2026-10-02).** Design §6 option B and CONV-3 keep
≥ 60 GB, and `download_weights.py` defaults to a 60 GB margin. The JIT kernel cache (`~/.cache/tt-metal-cache`, 38 GB
on 2026-10-02) and other agents' files live on the same filesystem and grow on their own, so the floor is headroom for
them, not for these scripts. The runs of 2026-10-02 before 02:20 used a 40 GB floor (the previous default), which was
never decided by the lead. Lowering the floor is an explicit operator decision per run (`--min-free-gb`); this
runbook's measured runs that did so say so. The device test's temporary 12 GB scratch (removed in a `finally`) may take
the disk to 40 GB for ~3 minutes.

Measured 2026-10-02 (`df` Avail of `/` in GB = 1e9 bytes, 877 GB volume; other users' files change it by a few GB at a
time):

| Item | Size |
|---|---:|
| BF16 checkpoint, 155 shards | 629.7 GB |
| BF16 on disk now (globals + layers 0-35, 105 shards) | 421.6 GB |
| BF16 of one MoE layer (its 3-4 shards; layer 2: 12.585 GB) | 12.1-12.6 GB |
| BF16 still to download (layers 36-52) | 207 GB (17 × 12.2) |
| TT cache, MoE layer / dense layer / globals (default options) | 6.656 / 0.355 / 3.607 GB |
| TT cache, full model | 343.8 GB |
| BF16 kept for good (shards 1, 4, 104: embedding + L0-1, the smallest shard L2-3, LM head + norm + MTP + L52 part) | 5.0 GB |
| Free now, with globals + L0-2 converted | 60.6 GB |

The guards (both scripts share the converter's functions, `guarded()` = estimate x 1.1 and `room_ok()`): before a
conversion, free − 1.1 × the part's estimated bytes (variants included; for a variant addition just the missing
variant files) ≥ floor (pipeline and converter, the same formula); before a download, free − (missing shards + 1.1 ×
the layer's cache estimate + 0.5 GB golden reserve) ≥ floor, and `download_weights.py` gets `--margin-gb` = floor +
1.1 × cache estimate + 0.5 (67.8 GB for a MoE layer) so it refuses on its own if the disk moved meanwhile; before a
golden, 0.5 GB. With `--delete-converted` the pipeline deletes eligible shards first when a guard would trip, and a
converter disk-guard exit (3) triggers one deletion pass and one retry. These are estimates of the scripts' own writes:
another process can still take the disk below the floor meanwhile.

`python3 $S/stream_weights.py --dry-run [--delete-converted] [--keep-bf16 none]` prints the projected free space after
every step from the real index and the current state, applying the same guards and stopping where the pipeline would.
From today's state (60.9 GB free at 02:26):

| Run | Projection |
|---|---|
| `--delete-converted` (default `--keep-bf16 0-35`) | stops before converting L03 (needs 7.3 GB above the 60 GB floor; nothing local is deletable): exit 3 |
| `--delete-converted --keep-bf16 none` (after sign-off) | completes: minimum 60.9 GB (the start), final 143.1 GB. Per local MoE layer −6.7 GB convert, +12.2 GB delete; per layer 36-52 −12.2 GB download, −0.1 GB golden, −6.7 GB convert, +12.2 GB delete |
| without `--delete-converted` | stops before converting L03 |
| `--convert-arg=--lm-head-split=both --keep-bf16 none --delete-converted` | the globals get +1.8 GB (after L2's shards free 12.4 GB first); the layers are unaffected |

---

## 5. The streaming pipeline (`scripts/stream_weights.py`)

Per unit, in order (globals, then `--layers` ascending), each step re-reading the ground truth (restartable at any
point; a re-run skips what is done):

1. **shards**: every shard of the layer present with the size from `hf_meta/tree.json`, else
   `download_weights.py --layers L --margin-gb <floor + 1.1 x cache est + 0.5> --layers-per-batch 1`; still missing →
   exit 4 (the downloader exits 0 when its own margin stops it).
2. **sha256**: `verify_shards.py` (hashes every present shard whose record does not match its size + mtime;
   `.verified.json`); every shard of the layer must be `ok`, and a record counts only while the file still has the size
   and mtime that were hashed (verify_shards.py keeps a deleted shard's old record, so a shard downloaded again is
   hashed again); else exit 5 (delete the BAD shard and re-run: step 1 re-downloads it).
3. **golden** (every layer not in the C2 golden's `layers_done`, i.e. 36-52 today): `hostrun.sh -- python -m
   models.demos.motif3.reference.golden_stream --resume --layers L --out $M/goldens/c2 --ckpt-dir $M/weights/Motif-3`; it
   must be the next layer of the golden (`last_layer + 1`), else exit 6. `--skip-golden` skips the step, and those
   layers' shards are then kept. Then the same for every `--extra-golden-out DIR` (repeatable; the fp32 golden
   `$M/goldens/c2_fp32`, lead decision 2026-10-02), in order, each only when it does not list L yet (log
   `golden_<dir name>_L<nn>`, state key `golden:<dir name>`). The deletion gate uses the **intersection**: a layer is
   golden-done only when every golden lists it (a missing or unreadable manifest counts as nothing done).
4. **convert**: `hostrun.sh -- python convert_weights.py --target mock --layers L --min-free-gb <floor> --mesh 4x8
   --cache-root <root> --weights-dir $M/weights/Motif-3 [variant options]` (or `--target device` through `devrun.sh`).
   The mesh, the root (`--cache-root`, default the serving resolution `MOTIF3_TT_CACHE_PATH` > `TT_CACHE_PATH` >
   `$M/tt_cache`) and the checkpoint are pinned, so `MESH_DEVICE`, `MOTIF3_WEIGHTS_DIR` / `HF_MODEL` cannot redirect
   it; `--convert-arg` passes only `--router`, `--mhc-sinkhorn`, `--lm-head-split`, `--embedding`, `--mock-desc`
   (anything else, e.g. `--mesh`, `--cache-root`, `--no-hash`, `--no-verify`, `--force`, is refused: exit 2). The
   pipeline then asks the converter for the status (`--status --json`, 3 s, same pins) and requires complete +
   `verify.ok` + every file hashed; converter exit 3 → one deletion pass + one retry → exit 3; exit 4 → 4; else → 7.
5. **verify**: done by the converter (cache-only rebuild with a raising source + sha256 re-hash, §1).
6. **delete** (`--delete-converted` only): under the converter lock (so no conversion changes a part meanwhile), the
   pipeline re-reads the status and deletes every present shard whose decoder layers are all
   * converted, verified and fully hashed **in the serving cache directory** (`<root>/<serving tag>/mesh4x8`; the
     status must say mesh (4, 8), the pinned revision and the default dtype policy — a `TT_MODEL_WEIGHTS_REVISION`
     makes the pipeline refuse to start, exit 2),
   * built by the current code (the part's fingerprint equals the current one; a converted layer whose fingerprint
     differs, and whose shards could be released, is rebuilt with `--force` first, while its BF16 is still there),
   * in the `layers_done` of the C2 golden and of every `--extra-golden-out` golden,
   * not in `--keep-bf16` (default `0-35`);

   never a shard holding a global tensor (shards 1 and 104: embedding, final norm, LM head, MTP;
   `download_weights.py` re-fetches those on every call, and the golden's final head after layer 52 reads shard 104),
   never the smallest shard (`model-00004`, 0.17 GB: TIS's `--host-weights-dir` check needs one `model*.safetensors`),
   never a non-shard file (config, tokenizer, index, `*.json`, `*.py`). The deletion function re-derives all of these
   from the status document itself (it does not trust a caller's list), syncs the filesystem, then unlinks. Without
   the flag the shards that would go are listed.

State: `$M/weights/Motif-3/.stream_state.json` (audit log: per layer the time of each step, every deleted shard with
its bytes and layers, every run's argv). `.download_state.json` is the downloader's and is not edited (it still lists
deleted layers; `HFWeightLoader.layer_available` and the golden check the files themselves). One pipeline at a time
(`.stream.lock`). The checkpoint directory is fixed (`$M/weights/Motif-3`): `download_weights.py` and
`verify_shards.py` hard-code it, so the pipeline has no `--weights-dir` option.

Exit codes: 0 done, 1 error, 2 usage (also: not the serving cache), 3 disk guard, 4 download incomplete / BF16
missing, 5 sha256, 6 golden, 7 conversion / verification.

Integration run (2026-10-02 02:26, non-deleting, layer 3 into a scratch root under `tt_cache/test`,
`logs/stream/20261002_pipeline_integration_v2_L03_testroot.log`):

```bash
python3 $S/stream_weights.py --layers 3 --no-globals --keep-bf16 none --min-free-gb 40 \
    --cache-root $M/tt_cache/test/stream_integration --state $M/tt_cache/test/stream_integration/stream_state.json
# $ hostrun.sh ... convert_weights.py --target mock --layers 3 --min-free-gb 40 --mesh 4x8 --cache-root .../stream_integration --weights-dir .../weights/Motif-3
# L03 (swa-attn/moe): 46 files, 6.656 GB in 40.3 s ...; verify ... OK (48 tensors ..., 46/46 files re-hashed in 4.8 s)
# would delete 2 consumed shards (12.1 GB): ['model-00005-of-00155.safetensors', 'model-00106-of-00155.safetensors']
```

The shards stayed (no `--delete-converted`); the scratch root was removed afterwards. `--state` and `--cache-root`
keep such a test away from the real audit log and cache.

**Real run** (2026-10-02, lead decisions: `--keep-bf16 0-3`, `--min-free-gb 40`, the fp32 golden in lockstep; logs
`logs/stream/20261002_034047_stream_real_L04-52.log`, `..._044640_stream_real_L03.log`, `..._stream_real_operator.log`,
`..._stream_real_timings.txt`, `20261002_final_status.json`, `20261002_final_verify_only.log`):

```bash
export HF_TOKEN_PATH=/run/user/1000/hf_token   # the downloader's huggingface_hub reads it; never print the token
F=(--delete-converted --keep-bf16 0-3 --min-free-gb 40 --extra-golden-out $M/goldens/c2_fp32 --cache-root $M/tt_cache)
python3 $S/stream_weights.py --layers 4-52 "${F[@]}"   # 65.8 min
python3 $S/stream_weights.py --layers 3 "${F[@]}"      # 1 min: L03 last (its BF16 stays), so the minimum is 53 not 47 GB
```

| Phase | Wall | Per layer |
|---|---:|---|
| L04-L35 (local): convert + verify, release 2-3 shards | 31.9 min | ~60 s (converter process 48-63 s, status + delete ~7 s) |
| L36-L52 (streamed): download, sha256, C2 golden, fp32 golden, convert + verify, release | 33.9 min | ~2.0 min: download 22-39 s (330-440 MB/s, token), sha256 9 s, bf16 golden 6 s, fp32 golden 10 s, convert 46-60 s |
| L03 (kept BF16) | 1.0 min | |
| `--verify-only --layers all --globals` afterwards (cache-only rebuild + re-hash of 2409 files) | 4.3 min | 54 / 54 OK |

Disk (`df` Avail, sampled every 5 s): start 60.0 GB, minimum 53.3 GB (L04's conversion, before its shards went;
projected 53.3), maximum 238.8 GB (after L35), end 117.8 GB. 146 shards (600.0 GB) deleted, every one recorded in
`.stream_state.json`; no retry, guard stop or non-zero step. The deleting pipeline's first real use.

**Consequences of deleting** (decide before passing `--delete-converted --keep-bf16 <less than 0-35>`): these tests
read BF16 of the listed layers and **skip** when it is gone (they build with `cache=False` or need the CPU reference):

| Test | BF16 it needs |
|---|---|
| `tests/test_decoder_layer.py` (acceptance: decoder-layer PCC ≥ 0.995) | the tested layers (default 0, 1, 2, 3, 4, 8, 16, 24, 32; `MOTIF3_TEST_LAYERS`), built with `cache=False` |
| `tests/test_model_truncated.py` (truncated-model state PCC; generator path) | layers 0-7 (`real_source_or_skip`), even though `MotifModel(cache="auto")` would load converted layers from the cache; the reference head reads shard 104 (kept) |
| `tests/test_serving_order.py` (serving order in one session) | layers 0-3 (also for the CPU reference of the long prompt) |
| `tests/test_weight_cache.py` device roundtrip / mock conversion | layers 0-2 / layer 0 |
| module tests with real weights (`tests/unit/test_{attention,mhc,mlp,moe,router_fp32,embed_head}.py`), GATE-3 (router inputs of layers 2-35), the C3 recipe (WAVE_A_REVIEW Appendix A: layers 0-35) | their layers |

`MotifModel(cache="auto")` and the generator load converted layers from the cache and need no BF16 for them; it is the
tests and the CPU reference that do. Gating those tests on the cache marker (a raising source with `cache=True` where
the test only needs the TT weights) is for their owners. Re-quantizing (another dtype policy = another tag directory)
needs the BF16 again: ~0.24-0.43 GB/s sustained, i.e. 25-45 min for the full checkpoint (verifier), and the disk for a
second cache (§4: 143 GB left at the end). A `CACHE_FORMAT_VERSION` bump or a module cache-name change makes the old
tag directory unusable; delete it (`rm -r $M/tt_cache/motif3-<old tag>`) before reconverting.

---

## 6. Expected timings

Measured on this host (mock cluster, page-cached BF16, 2026-10-02; `logs/host/20261002_022320_convert_real_force_L0-2_globals_v2.log`,
with another agent's 32-thread CPU golden running meanwhile; the 00:34 run without it:
`logs/host/20261002_003452_convert_real_L0-2_globals.log`):

| Step | Seconds |
|---|---:|
| process start + mock mesh open (`open_motif_mesh`) | ~3 + 0.7 |
| `global` build (embedding 2.4, LM head 5.9) / sha256 + fsync / verify | 8.4 / 2.6 / 2.7 |
| dense layer build / sha256 + fsync / verify | 3.5 / 0.2 / 0.5 |
| MoE layer build (experts 37.7: 12.3 GB BF16 read + transposes + bfp8 packing of 6.4 GB; attention 1.4; routers 0.7) / move (frees the replaced files) / sha256 + fsync / verify | 40.2 / 3.0 / 5.0 / 5.2 |
| layers 0-2 + globals, one process (`--force`) | 80 wall (59 at 00:34 without the concurrent golden) |
| a variant addition to a complete layer (stock constants, 8 files) | ~1 build, ~4 per process |
| one MoE layer as its own process (`/usr/bin/time`, 00:52): wall / peak host RSS | 49.9 / 21.7 GB |
| converter status (`--status --json`, no mesh; hashes the fingerprint files) | ~3 |
| device target (`test_device_convert_roundtrip`, 01:16) | build global 8.2 / dense 3.1 / MoE 36.2 s, verify (device load + re-hash) 3.1 / 0.4 / 5.3 s; 67.7 s for layers 0-2 + globals: the same as mock (host bound) |
| download, one MoE layer (12.2 GB at 0.24-0.43 GB/s sustained, verifier) | 30-50 |
| `verify_shards.py`, one layer's new shards | ~10 |
| golden resume, one MoE layer (compute 2.7-3.5 s, + start and resume-state load) | ~15 |

Projected full runs:

* convert layers 3-35 (all local), one process: 33 × ~50 s ≈ 28 min; through the pipeline (one converter process per
  layer + two status queries): ~1 min per layer ≈ 35 min.
* layers 36-52 through the pipeline: ~2-2.5 min per layer (download dominates) ≈ 35-45 min.
* whole pipeline after sign-off (`--keep-bf16 none`): **~70-80 min**, no device time.

---

## 7. Serving with the converted cache

* `MotifModel(..., cache="auto")` (default) builds every part with a `.complete` marker from the cache: with every
  listed file present it never reads the checkpoint (`tt/model.py:LazySource` opens it only on a cache miss);
  `test_device_model_auto_loads_mock_cache` checks this with a raising source, for the defaults and for the stock
  Sinkhorn + exact-fp32 router choices. A marked part with a missing file falls back to BF16 for that tensor (§1).
  Load time per MoE layer: 0.3-2.3 s from the (page-cached) TT cache vs ~30.5 s from BF16 (the generator's own log
  for an unconverted layer, `logs/dev/20261002_005415_integ_tf36.log`), i.e. ~26 min of BF16 conversion per model
  start for 51 MoE layers without the cache; a cold start reads the 344 GB cache from disk (~3-4 min at 1.4-2 GB/s,
  verifier).
* The server must resolve the same root: by default `$M/tt_cache`. TIS points `TT_CACHE_PATH` at its volume
  (`$TIS_TT_CACHE`, TIS_RUNBOOK §2.4): either export `MOTIF3_TT_CACHE_PATH=$M/tt_cache` in the shell that starts the
  server (it wins over `TT_CACHE_PATH`), link the tag directory into `$TIS_TT_CACHE` (TIS_RUNBOOK §2.4 loop), or convert
  straight into `$TIS_TT_CACHE` with `--cache-root`. Same `MESH_DEVICE=(4, 8)` and the default dtype policy, or the path
  changes and the server reconverts (or fails without BF16).
* The server needs the BF16 directory only for `config.json`, the tokenizer and TIS's weights-dir check; with every
  part converted and every listed file present, the deleted shards are never read.
* `tt/model.py:convert_weights` (integration agent) is a minimal in-process variant (no staging, hashes, verification,
  fingerprint or variants; 10 GB guard). This CLI rebuilds parts it marked, so every part ends up with `.convert.json`.

---

## 8. Tests

```bash
M=/home/ttuser/hchang/experiments/motif-3
# host (devices hidden; ~45 s): rules, pipeline gate (fake runner), dry-run timeline, an end-to-end mock conversion
$M/scripts/hostrun.sh -n weight_cache_host -- python -m pytest -p no:cacheprovider -q \
    models/demos/motif3/tests/test_weight_cache.py -k host
# device (~4-5 min; needs ~12 GB of scratch with 40 GB staying free, removed afterwards)
$M/scripts/devrun.sh -t 2400 -n weight_cache -- python -m pytest -p no:cacheprovider -s \
    models/demos/motif3/tests/test_weight_cache.py -k device
```

Results: §2 table. Last runs (2026-10-02): host 11 passed in 35 s (`logs/host/20261002_023322_weight_cache_host_final_v4.log`); device 2 passed in 204 s, fabric committed TORUS_Y (requested
FABRIC_2D_TORUS_XY), L1_SMALL 32768 (`logs/dev/20261002_022806_weight_cache_device_v2.log`).

---

## 9. Open issues

1. Decoder norms are owned by `tt/decoder.py` (`input_layernorm.weight`, `post_attention_layernorm.weight`, bf16
   ROW_MAJOR `[1, 1, 128, 32]`). The converter builds `MotifDecoderLayer`, so a renamed or added cached tensor in any
   module is converted from then on; parts converted before the change do not have it. `--verify-only --layers all
   --globals` finds them (the cache-only rebuild misses the new file), `--force` rebuilds them (and removes the files of
   the old names). A changed transform under an unchanged name needs a `CACHE_FORMAT_VERSION` bump (README §7, CONV-4),
   i.e. a new tag directory; the code fingerprint only keeps the pipeline from deleting BF16 under such parts, it does
   not make the server reject them.
2. The fingerprint is coarse: any edit of a fingerprinted file (comments included) or a tt-metal rebuild marks every
   existing part "other code". Consequence: the pipeline rebuilds such a layer (~50 s) before deleting its shards;
   nothing else uses it.
3. Conversion is single-process (~50 s per MoE layer, CPU bound in ttnn's bfp8 packing). With all BF16 local, two or
   three converter processes on disjoint layer ranges (same cache dir) would need per-part locks instead of the
   per-directory one; not done.
4. Deleting BF16 shards of layers 0-35 removes the real-weight inputs of the acceptance tests (§5 table). The default
   `--keep-bf16 0-35` prevents it; releasing them is a sign-off decision, or stream only 36-52 once `/data` (design §7.3
   decision 1, D4) exists. Done 2026-10-02 with `--keep-bf16 0-3` (lead decision): only layers 0-3 + shard 104 have
   BF16 now, so the §5-table tests skip wherever they need a layer >= 4 (`test_model_truncated` (0-7), the
   decoder-layer acceptance for 4 / 8 / 16 / 24 / 32, GATE-3 router inputs, the C3 recipe).
5. Shards 1 and 104 (4.8 GB) stay for good because `download_weights.py` re-fetches the global shards on every call
   and the golden's final head reads shard 104; a downloader option to skip the globals would let shard 1 go after the
   globals are converted. `.download_state.json` keeps listing deleted layers (owned by the downloader; every consumer
   also checks the files).
6. `tt/model.py:convert_weights` (integration) is a second, minimal converter (no staging / hashes / verification /
   variants / fingerprint, 10 GB guard). Either keep it for tests only or make it call `scripts/convert_weights.py`'s
   `Converter`; this CLI rebuilds the parts it marks (marker without `.convert.json`).
7. README §7 (shared infra) does not yet describe the mock-cluster conversion, `.convert.json`, the staging directory,
   the per-kind variants, the fingerprint or the decoder-norm cache names; this runbook does.
8. `MotifModel(cache="auto")` keys on the marker file only (`tt/model.py`); it does not check `.convert.json`, the
   fingerprint or file sizes, so a marked part with a missing file silently reads BF16 and rewrites the file without a
   disk guard (§1). Owned by the integration agent.
