# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Motif-3 GDLA attention on the BH Galaxy (design §2.3.4, §3.2-3.3; WAVE_A_REVIEW §5.4 ATTN-1..7).

Grouped Differential Latent Attention: 80 q heads = 16 groups x (4 signal + 1 noise), one shared 576-wide latent
KV head ``[n (512, unit RMS) | roped k_pe (64)]`` per token and layer, per-signal-head ``sigmoid(lambda)``
differential, elementwise sigmoid output gate, ``wo``. TP8 over the mesh columns: chip ``tp`` owns q heads
``[10 tp, 10 tp + 10)`` = KV groups ``{2 tp, 2 tp + 1}`` = signal heads ``[8 tp, 8 tp + 8)``; ``wo`` partial sums are
closed with ``all_reduce(tp)``. The latent projection is replicated (no CCL before attention).

Module boundary (README CONVENTIONS §3): input = the *normalized* attention input (``input_layernorm`` output; the
decoder owns that norm), output = the attention output after ``wo`` and ``all_reduce(tp)``:

* decode: ``x [1, 1, L, 4096]`` bf16 TILE DRAM (L = 8 lanes of this chip's DP row, replicated in the row; L = 16
  rows ``[8 anchors | 8 drafts]`` in the T64 verify step, below) -> ``[1, 1, L, 4096]`` bf16 TILE DRAM, bitwise
  identical on the 8 TP chips of a row;
* prefill (one user, S = bucket, replicated on all 32 chips): ``x [1, 1, S, 4096]`` -> ``[1, 1, S, 4096]``.

Decode (all 53 layers; absorbed MLA over the paged latent cache; per chip, lanes on tile rows)::

    cq   = x @ Wq_lat [4096,1024] (fp32 out)    kvl = x @ Wkv_lat [4096,640] = [c_raw 512 | kpe 64 | lam 64]
    cq_n = rms_norm(cq) (gamma_q folded into wq_b / wq_b_gate)    n = rms_norm(c_raw) (gamma_kv folded into W_UK/W_UV)
    q    = cq_n @ wq_b'  [1024, 10 x 192]      g = sigmoid(cq_n @ wq_b_gate') [1024, 8 x 128]
    q_nope [1,10,L,128], q_pe [1,10,L,64] = nlp_create_q_heads_split(q)        (heads on dim 1, lanes on rows)
    q_lat = q_nope @ W_UK'_h [1,10,128,512]    q_pe, k_pe = rotary_embedding_hf (prefill mode, cos/sin rows = lanes)
    Q = transpose(concat(q_lat, q_pe), 1, 2) -> [1, L, 10, 576]          (FlashMLA layout: lanes on dim 1)
    paged_update_cache(cache, transpose(concat(n, k_pe)) [1,L,1,576] height-sharded one lane per core, cur_pos, pt)
    O = paged_flash_multi_latent_attention_decode(Q, cache, head_dim_v=512, scale=1.0, window 129 | None,
                                                  G1 program config (k_chunk 128), sdpa_decode role) [1, L, 10, 512]
    U = transpose(O, 1, 2) @ W_UV'_h [1,10,512,128] -> nlp_concat_heads -> [U_sig 1024 | U_noise 2 x 128]
    D = U_sig - sigmoid(lam @ E) * (U_noise @ X);  out = all_reduce_tp(where(active, D * g, 0) @ wo [1024, 4096])

Prefill (expanded GQA, one user)::

    same projections; K/V from the latent: n @ E_pref [512, 2 x 320] -> per group [k_nope | v | 0_64]
    (nlp_create_q_heads_split -> k_nope [1,2,S,128], V padded [1,2,S,192]); K = [k_nope | k_pe x 2 groups]
    O = scaled_dot_product_attention(Q [1,10,S,192] (HF head order), K, V_pad, causal, scale=1.0, window 129 | None,
                                     G2 program config)[..., :128] -> nlp_concat_heads
    same differential / gate / wo / all_reduce; the cache gets typecast(concat(n, k_pe)) via paged_fill_cache with the
    bucket's first S / block_size page-table entries (M10).

Numerics / design decisions:

* **Softmax scale folded into q** (FEASIBILITY_REPORT §4.3 M1, verify_attention d1): the decode / prefill kernels
  truncate the scale of the running-max correction to bf16 (-0.54 % for 192^-0.5, -0.10 % for the global scale).
  ``wq_b`` is multiplied by the layer's scale on the host (fp32, rounded once) and both kernels get ``scale=1.0``
  (exact in bf16). RoPE and the absorb are linear, so the scaled q_pe / q_lat are exactly s * q. ``q_norm``'s gamma is
  folded into the input rows of ``wq_b`` and ``wq_b_gate`` the same way (a weighted ``rms_norm`` costs 18 us traced at
  decode shapes, a weightless one ~6 us).
* **Head order.** Decode keeps the 10 local q heads in the *virtual* order ``[s_0..s_7 | noise_g0, noise_g1]``
  (local HF heads ``0,1,2,3,5,6,7,8,4,9``; all heads share the one latent KV head, so the order is free). After the
  un-absorb + ``nlp_concat_heads`` the 8 signal heads are one contiguous ``[L, 1024]`` channel region in ``wo`` / gate
  order and the noise heads the last 256 columns (one channel split, no slices). Prefill reorders Q to the HF order (5
  consecutive heads per KV group, the GQA mapping of the SDPA op) with one dim-1 concat.
* **Differential after the un-absorb** (HF expanded-form rounding: per-head 128-dim outputs, then
  ``signal - bf16(sigmoid(lambda)) * noise``); lambda and the noise heads are expanded to the 8 x 128 signal
  columns with exact 0/1 matmuls (``E``, ``X``; ``sigmoid(lam @ E) == sigmoid(lam) @ E`` because every column of ``E``
  selects one logit).
* q path: ``x @ Wq_lat`` in fp32 (HF runs wq_a / q_norm / wq_b in fp32), q_norm in fp32, TF32-class operands into
  ``wq_b`` / ``wq_b_gate``; kv path and lambda in bf16 like HF. KV cache = gamma-free unit-RMS latent in
  ``cfg.dtypes.kv_cache`` (bfp8; bf16 via ``MOTIF3_KV_CACHE_DTYPE``); gamma_kv lives in ``W_UK'`` / ``W_UV'`` /
  ``E_pref`` (``weights.fold_kv_norm``).
* Prefill SDPA: the shared ``sdpa_prefill`` role on every layer by default (G2: HiFi4, fp32 acc off -> the streaming
  SDPA kernel; README §4, §10 rule 1). The QK scores / softmax statistics are then bf16: module PCC 0.99965-0.99978
  (random) / 0.99996-0.99997 (real) on global layers and 0.99956-0.99973 on SWA layers, against HF-bf16's 0.99998 (HF
  keeps the scores in fp32). ``sdpa_prefill_fp32_acc="auto"`` is the measured opt-in for window-free calls: fp32 dest
  acc keeps the scores in fp32 (``qk_im_df``; module PCC 0.99996-0.99999) but makes ttnn fall back to the legacy,
  non-streaming compute kernel (``sdpa_program_factory.cpp`` ``can_use_streaming_compute``), which G2 never validated,
  costs +30-55 % per global-layer call at 8K-32K (8K 19.6 vs 15.1 ms, 16K 45.3 vs 32.5, 32K 137.8 vs 88.8: ~0.7 s TTFT
  over the 14 global layers of a 32K prompt) and needs 1.38 MB of static circular buffers (role: 1.17 MB). It is a
  layer-level decision (decoder-layer PCC / teacher-forced top-1), so it stays off here until a shared role exists
  (requested: ``sdpa_prefill_fp32``). SWA prefill cannot use fp32 acc at all (upstream bug, pinned by
  ``test_sdpa_prefill_window_fp32_acc_upstream_bug``): the legacy kernel's ``generate_causal_sliding_window_mask``
  (``sdpa/device/kernels/dataflow/dataflow_common.hpp``) counts a K tile as fully allowed when it lies in the window of
  the Q tile's *first* row (``k_tile_start >= min_window_start``; should be ``max_window_start``), so rows 1..31 of
  every Q tile also attend to up to 31 keys older than their window -- confirmed on device (output == that emulated mask
  at PCC 0.999995; the same kernel with the window as an explicit bf16 ``attn_mask`` is exact). Once fixed upstream,
  fp32 acc would lift SWA prefill from ~0.9996 to ~0.99999. Decode FlashMLA keeps its scores / statistics in bf16
  regardless (``im_df`` is hard-coded in ``sdpa_decode_program_factory.cpp``): that kernel floor (op PCC ~0.9999 at
  Motif shapes) dominates the decode error and is roughly doubled by the differential's cancellation.
* Inactive lanes (``cur_pos = -1``): skipped by ``paged_update_cache`` and FlashMLA, which leaves their output rows
  unwritten; ``forward_decode`` therefore requires the per-step ``active`` mask and replaces those rows of the ``wo``
  input by 0 with ``ttnn.where(active, D * g, 0)`` (design §2.3.4 step 12), so nothing non-finite can reach ``wo`` /
  the all-reduce / the residual streams.
* **L1_SMALL is required** (device params ``l1_small_size``; ``RECOMMENDED_L1_SMALL_SIZE``). The decode
  ``all_reduce(tp)`` of ``[1, 1, 8, 4096]`` takes ttnn's direct reduce-scatter, whose program creates its global
  semaphores while its ~512 KB L1-sharded staging buffer is live; without an L1_SMALL region they land in main L1 at
  ~1,047,424 B and stay there for the life of the cached program (``reduce_scatter_minimal_direct_factory.cpp``
  ``prefer_l1_small_buffer_type``; the all-gather's barrier semaphore likewise). Every later program whose static
  circular buffers end above that address then throws ``TT_THROW ... Statically allocated circular buffers ... clash
  with L1 buffers``: global-layer prefill at S >= 1024 (CB region up to 1.17 MB with the role, 1.38 MB with fp32
  acc), and the bf16-KV FlashMLA decode (1.09 MB). In serving (prefills between decode steps) that is the first
  prefill after any decode step. With ``l1_small_size=32768`` all of these pass and the prefill output is bitwise
  independent of the session history (``test_attention_serving_session``). The constructor warns when the mesh has no
  L1_SMALL region (``require_l1_small=True`` raises); ``test_attention_l1_small_hazard`` pins the failure.

Decode matmul program configs (measured on this Galaxy, traced, ``[1,1,8,K] @ [K,N]``): 1D multicast
(``model_config.mcast1d_matmul_pc``) for ``Wq_lat`` (8x4 cores, 62 -> 28 us), ``Wkv_lat`` (5x4, 59 -> 21 us), ``wq_b``
(12x5, 20 -> 12 us), ``wq_b_gate`` (8x4, 17 -> 8 us); ``wo`` keeps the auto config (25 us, the 1D configs are slower);
the per-head bmms use ``MatmulMultiCoreReuseProgramConfig`` (one head per core; ``W_UV`` 65 -> 7 us, ``W_UK`` 20 ->
8 us). They are built by :func:`decode_matmul_program_configs`, a module-local helper until ``model_config`` has
shared builders for them (requested: ``cfg.attn_decode_matmul_pc(name)`` with exactly these parameters). Prefill uses
ttnn's auto configs (M = S rows).

Fallback paths kept: ``rope_mode="composite"`` (``x cos + (x @ R) sin``, G8 fallback); ``matmul_program_configs``
overrides (``None`` entries = auto config).

Hooks for resumed / chunked prefill and the MTP layer (``docs/features/FEATURES_DESIGN.md`` §3.2, §3.6; README
CONVENTIONS §15, §17; work package 2a):

* ``MotifAttention(..., spec=, weight_prefix=)``: the layer constants (window, softmax scale, RoPE kind) and the HF
  module path of the 9 tensors. Defaults (:func:`resolve_attn_layer`): ``cfg.layer(l)`` and
  ``model.layers.{l}.self_attn`` for a decoder layer; ``cfg.mtp_layer_spec()`` (SWA window 129, scale 192^-0.5, plain
  RoPE) and ``model.mtp_layers.0.self_attn`` for the MTP layer ``cfg.mtp_layer_idx`` = 53 (TT-cache part ``L53``).
  The weight transforms are the same for every layer (scale folded into ``wq_b``, gammas folded, head order).
  ``cache=True`` requires the canonical spec and prefix of the index: the cache names (``attn.v2.*`` under ``L<l>``)
  carry neither the folded scale nor the source, so anything else would be written into, or read from, that layer's
  files.
* :meth:`MotifAttention.fill_kv`: KV-only latent fill. ``x @ Wkv_lat`` -> ``rms_norm(c_raw)`` -> RoPE of ``k_pe`` at the
  rows' positions -> typecast -> ``paged_fill_cache`` through a fill table. No q path, SDPA, ``wo`` or CCL. The MTP
  prefill uses it (the MTP cache row of position ``p`` depends only on the layer input at ``p``), and so can chunked
  fills. The ops are the ones :meth:`forward_prefill` runs (``_kv_latent``, ``_split_kv``, ``_fill_latent``), so the
  written rows are bitwise the rows a prefill of the same input rows at the same positions writes.
* Fill tables (``prefill_plan.fill_table``): ``paged_fill_cache`` skips ``-1`` entries (the shared full blocks below
  ``w0`` and pure bucket-padding blocks). The sp0 :meth:`forward_prefill` takes one as ``page_table=`` unchanged;
  its output and every written row are bitwise draft 1.
* Offset RoPE for chunks at any start: ``rope.chunk_rope_tables(rope.chunk_rot_idxs_device(positions))`` gathers
  ``{kind: (cos, sin)}`` ``[1, 1, C, 64]`` from the ROW_MAJOR tables once per chunk for every layer (``tt/rope.py``);
  pass it as ``rot=``.

Resumed (sp1) prefill and the decode KV-write hook (features design §3.2, §3.5, D2-D5, D12; work package 2b). One
chunk = bucket ``C`` rows at absolute positions ``[a, a + C)`` (``prefill_plan.ChunkPlan``); its device inputs are a
:class:`PrefillChunkInputs` built once per chunk and shared by every layer: ``forward_prefill(x, chunk=inp,
kv_cache=cache)``. A chunk at ``a = 0`` (sp0) is draft 1 with the chunk's fill table; a chunk at ``a > 0`` (sp1)
reads the cached prefix:

* **global layers** (absorbed MLA over the paged latent; D2)::

      Q = [q_nope @ W_UK' | rope(q_pe, yarn rows a..a+C-1)]          [1, 10, C, 576] (virtual head order, as decode)
      paged_fill_cache(cache, typecast([n | rope(k_pe)]), fill_pt)    FIRST: the chunk's own keys come from the cache
      O = chunked_scaled_dot_product_attention(Q, K = cache, V = cache, sdpa_pt [1, W'], chunk_start_idx_tensor = [a],
                                               scale 1, cfg.resumed_prefill_pc, "sdpa_prefill_fp32" role)
      O[..., :512] @ W_UV' -> nlp_concat_heads -> [sig 1024 | noise 256] -> differential, gate, wo, AR(tp)  (decode's)

  Row ``i`` attends keys ``[0, a + i]``; every key, the chunk's own included, is a bfp8 cache row (decode numerics).
  Gate G9: the op needs fp32 dest accumulation over the 576-wide heads (the bf16-dest ``sdpa_prefill`` role misses
  PCC 0.999 everywhere), and ``a`` must be a multiple of the op's q and k chunks (no device check: a misaligned start
  silently answers from the floored start) -- checked here before every call. The q / k chunks are per bucket
  (``model_config.SP1_GLOBAL_CHUNKS`` = ``prefill_plan.DEFAULT_SP1_GLOBAL_CHUNKS``, G9's fastest: 128/128 at C = 128
  and C >= 2048, 64/64 at C = 256-1024), so every start is a multiple of ``cfg.prefill_resume_alignment`` = 128. A
  bf16 cache (picked by the cache tensor's dtype) uses 64/64 everywhere (``SP1_GLOBAL_CHUNKS_BF16_KV``): at 128/128
  its static CBs end at 1,572,480 B, through the L1_SMALL region of the CCL semaphores, which tt-metal does not check
  (garbage, lost replica consistency, then a hang in a later CCL).
* **SWA layers** (square ``[tail ‖ chunk]`` window; D3, gate G10)::

      tail = typecast(concat(slice(cache, [blk_j, 0, 0, 0], [blk_j + 1, 1, bs, 576], slice_dim=0, num_devices=N)))
                                                          [1, 1, 128, 576] = cached rows a-128 .. a-1 (roped k_pe)
      latent [tail | n, rope(k_pe, plain rows a..)] -> n @ E_pref -> K [1, 2, 128 + C, 192], V_pad (draft-1 expansion)
      Q_cat = [128 filler rows | Q (HF order)]            [1, 10, 128 + C, 192] (filler outputs are dropped)
      O = scaled_dot_product_attention(Q_cat, K, V_pad, causal, sliding_window_size=129, "sdpa_prefill" role)
      rows [128, 128 + C) -> the draft-1 epilogue; then the fill (after the SDPA, as sp0)

  Square row ``128 + i`` is position ``a + i`` and sees exactly keys ``[a + i - 128, a + i]``; the tail block bounds
  are device tensors (tensor-args slice), so one program serves every block id. The tail keys are bfp8 cache rows (as
  decode reads them), the chunk's own keys the bf16 latent (as draft 1). Exact: when ``a`` is a multiple of the SDPA's
  q / k chunk (128), the square's chunk grid is the single shot's, so every row whose window lies inside the chunk
  (positions ``>= a + 128``) is bitwise the draft-1 single-shot row, and with a bf16 cache every row is (the tail
  then holds the single shot's own latents); ``tests/unit/test_attention_resumed.py`` asserts both. A ``(path,
  bucket)`` whose square would exceed ``max_model_len`` rows is refused (:func:`max_sp1_bucket`).
* Programs depend on ``(path, bucket)`` only (the start, the block ids, the RoPE rows are device tensors), so one
  warm-up call per bucket (:meth:`PrefillChunkInputs.warmup`: writes nothing, reads the null block) compiles them all
  before the decode capture (D12). :class:`ChunkHostTables` checks the tables of a chunk against each other (fill ids
  == SDPA ids, tail ids == the SDPA ids before the start, real RoPE rows at their positions, one written run ending at
  the last real row), so an inconsistent builder fails on the host. Persistent inputs are rewritten in place with
  :meth:`PrefillChunkInputs.write`; ``regather=False`` (captured prefill) drops the eager RoPE rows, which a trace
  must gather itself.
* ``forward_decode(..., kv_write=w)``: :class:`DecodeKVWriter` hook for the KV-R / speculative KV-write modes of
  ``tt/kv_write.py`` (design §3.5). ``None`` keeps the draft-1 8-lane update (bitwise unchanged; 8 rows only, see
  "T64 verify step" below).

Packed (multi-row) prefill passes (P5; ``docs/p5_t64/P5_T64_DESIGN.md`` §3.2-§3.5, gate G15a, GATES_RESULTS §13). The
short chunks of several rows run as ONE pass of ``T = B * S`` rows: ``B`` segments (``generator_api.PACK_BATCHES``,
dummy segments included) of ``S`` rows each. Segment ``k`` occupies packed rows ``[k S, k S + S)``: its real rows, then
padding. The inputs are a :class:`PrefillChunkInputs` uploaded from :class:`PackedHostTables`
(:func:`packed_host_tables` of a ``prefill_plan.PrefillPass``; ``path`` ``"pk0"`` = sp0 segments, ``"pk1"`` = sp1
segments at one common start ``a``). They hold the concatenated fill table ``[1, T / bs]`` (``-1`` for shared, padding
and dummy blocks) and, always, the gathered RoPE rows ``[1, 1, T, 64]`` of every segment's positions (review edit
R-E11). Every row-local op (projections, norms, the epilogue, ``wo``) runs unchanged at bucket ``T``; only the SDPA,
the RoPE positions and the KV fill are per segment. The SDPA runs on a metadata view ``[1, H, T, d] -> [H, B, S, d]``
plus a CN ``transpose(0, 1)`` -> ``[B, H, S, d]`` (batch row ``k`` = segment ``k``), and back:

* **pk0**: ``q_full / k_full / v_pad`` -> ONE batched causal SDPA with the per-row bucket-``S`` config
  (``cfg.sdpa_prefill_pc(kind, seq_len=S)`` and :meth:`MotifAttention.prefill_sdpa_window_and_config` ``(S)``: window
  129 only at ``S >= 129``) -> CN back -> the draft-1 epilogue; the fill after the SDPA. G15a (a): every segment
  bitwise equal to the single-row SDPA at bucket ``S``.
* **pk1 global**: ``Q_abs [1, 10, T, 576]``; the fill FIRST (no segment of a pass reads a block the pass writes:
  the planner's writer-first rule); ``chunked_scaled_dot_product_attention(Q [B, 10, S, 576], cache, cache,
  page_table [B, W'], chunk_start_idx_tensor [a])`` with ``cfg.resumed_prefill_pc("global", S, kv_dtype)`` and the G9
  role; CN back; ``[..., :512]``; the absorbed epilogue. G15a (b): bitwise per segment vs the single-row sp1 call.
* **pk1 SWA**: per segment the square ``[tail ‖ segment]``. The tails ``[B, 1, 128, 576]`` bf16
  (:meth:`MotifAttention._gather_tails_packed`) come in one of two variants, fixed per pass (review edit R-E2):
  ``shared`` (every segment has the same tail blocks: the solo 2-block gather + typecast + ``repeat``) or ``distinct``
  (2B tensor-args block slices + ONE concat + view + typecast). Then ``lat = concat([tail_b, view(kv_row, [B, 1, S,
  576])], 2)``, the draft-1 expansion at batch ``B`` (the ``[512, 640]`` weight unbatched), ``Q_cat = concat([qb[:, :,
  :128], qb], 2)``, the causal window-129 SDPA with ``cfg.resumed_prefill_pc("swa", S)``, rows ``[128, 128 + S)``, CN
  back, the epilogue; the fill after the SDPA. G15a (c): bitwise per segment vs the solo sp1 SWA dataflow. The two
  variants are different programs, so the warm-up runs both per pk1 shape (:func:`warmup_packed_host_tables`).

Programs depend on ``(path, T, S)`` and, for pk1, the tail variant only (the starts, block ids and RoPE rows are device
data). A batched SDPA's static CBs are the per-row bucket-``S`` program's (they do not depend on ``B``); G15a found
every new program's CB end below the L1_SMALL region. Dummy segments write nothing and copy segment 0's tables (pk1:
its SDPA row and tails), so they read only what segment 0 reads; their outputs are dropped.

T64 verify step (``docs/p5_t64/P5_T64_DESIGN.md`` §4.1-§4.4, T1-T3; gate G-S1w; ``tt/kv_write.py`` "T64"). Each DP row
decodes 16 rows ``[8 anchors | 8 drafts]``: lane ``8 r + j`` has its anchor at ``n`` in row ``j`` and its draft at
``n + 1`` in row ``8 + j``, with the lane's own page-table row. The 16 rows are still one 32-row tile row, so every op
of :meth:`MotifAttention.forward_decode` before and after FlashMLA runs unchanged on ``L = 16`` rows (RoPE rows from
``rope.rot_idxs_host(positions, rows_per_dp=16)``, the mask from ``active_mask_from_cur_pos(cur_pos, 16)``). The KV
write is ``kv_write.DecodeKVWrite(rows=64)``: call A writes the anchors at ``n``, then call B the drafts at ``n + 1``,
both before FlashMLA. Causality is per row through ``cur_pos``: a draft sees its anchor's key ``n``, an anchor never
sees ``n + 1``. FlashMLA runs as **option A''**:

* SWA layers (39, and the MTP layer): ONE call at B = 16 on ``kv_write.cur_pos [16]`` / ``kv_write.page_table [16,
  W]`` (a draft row carries its owner's table row). A user reads at most 2 K chunks of 128 inside the 129-key window,
  so the per-user core split of B = 16 cannot change the flash-decode reduction: bitwise the B = 8 rows, +0.7 us
  (T64N §5.1).
* global layers (14): one B = 8 call per entry of ``kv_write.flash_groups()``: ``q_mla[:, 0:8]`` (the anchors, at
  ``flash_cur_a``) and ``q_mla[:, 8:16]`` (the drafts, at ``flash_cur_d``), both on the anchors' ``[8, W]`` table
  ``flash_pt``, outputs concatenated on dim 1. That is the T32 call's program and core split (15 cores per user), so
  every row equals its T32 row bit for bit. One B = 16 call (option A) splits each user over 7 cores and changes the
  reduction tree: FlashMLA alone PCC 0.99991 (G-S1w b), a whole layer max |d| 0.0156 (T64N §5.1), this module's
  output max |d| 0.003-0.004 at PCC 0.999926 (random weights, below).

So every T64 row is bitwise the T32 row of the same token, position and cache, anchors and drafts, which keeps
``spec_verify="auto"`` lossless when steps alternate between the two traces. At 8 rows per DP row
``flash_groups()`` is the one T32 call: the T32 step issues exactly today's ops. Host guards (before any op): the
draft-1 update (``kv_write=None``) writes 8 lanes on 8 cores, so it refuses any other row count; a writer's
``lanes_per_row`` must match ``x``; a global layer at more than 8 rows needs ``flash_groups()``. Option A stays
available for measurements (gate G16): a writer whose ``flash_groups()`` returns the one group ``(0:16, cur_pos,
page_table)``. Measured (``tests/unit/test_attention_kvr.py``, 2026-10-03; layers 0 (global), 1 (SWA) and the MTP
layer, anchors at ~1K / ~4K / ~32K, the real ``DecodeKVWrite(rows=64)``): every anchor row and every draft row is
bitwise its row of two T32 steps (the anchors at ``n``, then the drafts at ``n + 1``), and every chip's cache bitwise
the T32 caches, for bfp8 ``all_split`` / ``row_split`` and bf16 ``all_split``; a traced T64 step replays bitwise equal
to eager with no program compiled after the warmup; no static-CB clash with an L1 pin page alive.

Measured on this Galaxy (``tests/unit/test_attention.py``, fabric committed TORUS_Y, mesh opened with
``l1_small_size=32768``; PCC vs the fp32 reference): prefill (default role) global 0.99965-0.99978 random / 0.99996-
0.99997 real (S = 128 ... 32768), SWA 0.99956-0.99973 (fp32-acc opt-in on window-free calls: 0.99996-0.99999);
decode global 0.99975 (random) / 0.99997 (real), SWA 0.99961-0.99976, worst lane 0.99945; FlashMLA alone vs fp64 on
its own Q / cache 0.99987-0.99999 (worst lane 0.99976), the exact mask the best match on every lane, negative controls
(window 128 / 130 / 129-on-global, key p+1 visible) flagged on every distinguishable lane; latent cache vs reference
``c_kv / gamma`` 0.99997 (bf16 KV 0.999997); bf16 KV and block 32 in the serving order (prefill, decode, prefill after
decode, decode) per user >= 0.9995.
Traced decode per call at 4K context: SWA 282 us, global 354 us (FlashMLA 30 / 102 us, AR(tp) 34 us); eager ~2.8 ms.
Eager prefill per call (incl. cache fill): S = 32768 SWA 41.7 ms, global 88.8 ms (fp32-acc opt-in 137.8 ms).
Resumed (sp1) chunks (``tests/unit/test_attention_resumed.py``, 4096-token prompt in the design §5.2 schedules; rows vs
the fp32 reference): global sp1 rows 0.99990 random (worst row 0.99983; the draft-1 sp0 single shot: 0.99968, worst
0.99934) / 0.99996-0.99997 real (worst 0.99945; sp0 0.99997, worst 0.99964): every key is a bfp8 cache row with fp32
accumulation (decode numerics); with a bf16 cache 0.99994 (worst row 0.99992). SWA sp1 rows 0.99954-0.99956 random /
0.99974-0.99975 real, the sp0 numbers (0.99956 / 0.99974), and exact: every row past the first 128 of a 128-aligned
chunk is bitwise the draft-1 single-shot row (bfp8 cache, random and real weights), and with a bf16 cache every row of
the chunks whose tail TT wrote (e.g. all 3968 sp1 rows of the 31 chunks of 128). Bitwise: sp0 chunks == draft 1,
every cache row a chunk writes == the single-shot fill, shared / other blocks untouched, sp1 trace replay at another
start == eager; all with junk bucket-padding rows, under which a continuation that skips the own partial block is
caught (global sp1 rows PCC 0.9886, worst row 0.952; own blocks differ on both kinds). A warm-up chunk per ``(path,
bucket)`` compiles every program: a real chunk after it (``forward_prefill`` + ``fill_kv``) compiles 0. Eager per call
(random weights, start 128 / 8192; sp0 at the same bucket; two runs, small buckets are dispatch-bound and noisy):
global C = 512 2.9-4.5 / 6.5-7.3 ms (sp0 3.7-3.8), 2048 5.3-5.5 / 15.5-15.6 (4.2), 8192 30.9 / 65.6-65.7 (14.9-15.0)
with 64/64 chunks at every bucket; with G9's per-bucket table (``test_wp2b_sp1_cost``, 2026-10-03, bfp8 cache) C =
128 2.85 / 5.36 ms (64/64: 2.81 / 6.03), 2048 4.82 / 12.08 (5.26 / 15.54), 8192 24.44 / 47.05 (30.85 / 65.69), C = 512
unchanged (64/64 in both); SWA C = 512 4.4 (3.7-3.8), 2048 4.9 (4.2), 8192 11.9 (11.6-11.7) at either start.
Deviations from the wave-B action list, with reasons: ATTN-4 uses ``rotary_embedding_hf`` in *prefill* mode on the
heads-on-dim-1 / lanes-on-rows q_pe ``[1, 10, L, 64]`` with ``decode_cos_sin(layout="rows")`` (same kernel, row t
rotated by lane t's position; the decode-mode variant would need a transpose + reshard of q_pe and k_pe, 4 extra ops);
ATTN-1 uses ``wq_b_for_chip(layout="interleaved")`` (one ``nlp_create_q_heads_split`` yields q_nope and q_pe) and
per-head copies of the per-group ``W_UK'`` / ``W_UV'`` (one head per core in the bmm); the ``active`` mask is applied to
the ``wo`` input (1024 columns) instead of the output (identical result, half the cost).

Import rule: only ``torch``, ``ttnn`` and the motif3 shared infra (``model_config``, ``weights``, ``ccl``, ``rope``,
``prefill_plan``; the P5 constants of ``generator_api``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import torch

import ttnn

from . import prefill_plan as PP
from . import weights as W
from .ccl import MotifCCL
from .generator_api import PACK_BATCHES, PACK_SEG_BUCKETS, PACK_SP1_SEG_BUCKETS, PACKED_PASS_KINDS, PK1_TAIL_VARIANTS
from .model_config import TILE, LayerSpec, MotifTTConfig, make_compute_kernel_config, mcast1d_matmul_pc
from .rope import MotifRope, decode_rows_per_dp
from .prefill_sp import SP_TAIL

# Packed prefill passes (P5, docs/p5_t64/P5_T64_DESIGN.md §3): "pk0" = sp0 segments, "pk1" = sp1 segments at one start.
PK0, PK1 = PACKED_PASS_KINDS
# pk1 SWA tails gathered once and repeated ("shared") or per segment ("distinct"): two program sets (review edit R-E2).
SHARED_TAILS, DISTINCT_TAILS = PK1_TAIL_VARIANTS

# Cache-name prefix of every tensor this module writes to the TT weight cache. Bump the version whenever a transform
# below changes (head order, scale / gamma folding, layouts), so stale .tensorbin files are never loaded silently.
ATTN_CACHE_VERSION = 2
_CACHE = f"attn.v{ATTN_CACHE_VERSION}"
MASK_WIDTH = 1024  # width of the decode ``active`` mask that matches the ``wo`` input (8 signal heads x 128)
# L1_SMALL region (bytes per core) the mesh must be opened with (module docstring, "L1_SMALL is required"). 32 KB =
# 512 CCL semaphores of 64 B. Measured for attention alone: 16 after the decode all_reduce (direct reduce-scatter 12 +
# all-gather) and the 128 / 1024 prefill buckets, 18 after the 4096 bucket (large prefill reduce-scatters take the ring
# path, whose 4 semaphores stay in main L1: all_reduce has no use_l1_small_for_semaphores). Verified on this Galaxy.
RECOMMENDED_L1_SMALL_SIZE = 32768


def l1_small_bytes(mesh_device) -> Optional[int]:
    """Per-core size of the mesh's L1_SMALL allocator region (0 = the mesh was opened without ``l1_small_size``), or
    ``None`` if it cannot be queried (no real device)."""
    try:
        return int(ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_per_bank)
    except Exception:
        return None


L1_SMALL_WARNING = "Motif-3 attention: the mesh has no L1_SMALL region"


def check_l1_small(mesh_device, *, require: bool = False) -> Optional[int]:
    """Warn (``require=True``: raise) when the mesh has no L1_SMALL region: the CCL semaphores then fragment main L1
    and the first global-layer prefill (or bf16-KV decode) after a decode step throws a static-CB clash (module
    docstring). Returns :func:`l1_small_bytes`. The warning is a ``RuntimeWarning`` issued at the caller of the
    constructor, so Python's default filter shows it once per call site (not once per layer)."""
    size = l1_small_bytes(mesh_device)
    if size == 0:
        msg = (
            f"{L1_SMALL_WARNING}. CCL global semaphores (decode all_reduce / direct reduce-scatter, all-gather) are "
            "then allocated in main L1 around 1.0 MB and stay there, so the first global-layer prefill at S >= 1024 "
            "(or bf16-KV FlashMLA decode) after any decode step throws 'Statically allocated circular buffers ... "
            f"clash with L1 buffers'. Open the mesh with l1_small_size={RECOMMENDED_L1_SMALL_SIZE} (pytest: "
            'device_params(l1_small_size=...); vLLM: --additional-config \'{"tt": {"l1_small_size": ...}}\').'
        )
        if require:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    return size


# ======================================================================================================================
# head bookkeeping (pure python / torch; also used by the tests)
# ======================================================================================================================
def virtual_head_order(cfg: MotifTTConfig) -> List[int]:
    """Local q-head order used on device: the chip's signal heads first (signal order ``s_loc = 4 g_loc + j``, i.e.
    local HF heads ``5 g + j``), then one noise head per local group (``5 g + 4``): ``[0,1,2,3,5,6,7,8,4,9]``."""
    G, r, hpg = cfg.kv_groups_per_chip, cfg.grouped_ratio, cfg.heads_per_group
    return [hpg * g + j for g in range(G) for j in range(r)] + [hpg * g + r for g in range(G)]


def virtual_head_groups(cfg: MotifTTConfig) -> List[int]:
    """Local KV group of each virtual head."""
    return [h // cfg.heads_per_group for h in virtual_head_order(cfg)]


def hf_order_from_virtual(cfg: MotifTTConfig) -> List[Tuple[int, int]]:
    """Dim-1 slices ``(start, stop)`` of a virtual-order head tensor whose concatenation is the HF local order
    (``[s(g0) x4, n(g0), s(g1) x4, n(g1)]``)."""
    G, r = cfg.kv_groups_per_chip, cfg.grouped_ratio
    out = []
    for g in range(G):
        out.append((r * g, r * (g + 1)))
        out.append((G * r + g, G * r + g + 1))
    return out


# ======================================================================================================================
# host-side weight transforms (fp32; built on the skeleton transforms of tt/weights.py, ATTN-1)
# ======================================================================================================================
def _attn_name(layer_idx: int, suffix: str) -> str:
    return W.hf_name(layer_idx, f"self_attn.{suffix}")


# ---- layer identity: spec + weight prefix (decoder layers and the MTP layer; WP2a) ----------------------------------
MTP_ATTN_PREFIX = "model.mtp_layers.0.self_attn"  # the MTP layer's attention (checkpoint shard 104)


def attn_weight_prefix(cfg: MotifTTConfig, layer_idx: int) -> str:
    """HF module path of a layer's 9 attention tensors (``{prefix}.{name}.weight``): ``model.layers.{l}.self_attn``
    for a decoder layer ``0 <= l < num_hidden_layers`` (whatever ``cfg.num_layers`` a truncated run builds), and
    :data:`MTP_ATTN_PREFIX` for the MTP layer ``cfg.mtp_layer_idx`` (53) when the checkpoint has one."""
    L = int(layer_idx)
    if 0 <= L < int(cfg.num_hidden_layers):
        return W.hf_name(L, "self_attn")
    if L == int(cfg.mtp_layer_idx) and int(cfg.num_nextn_predict_layers) >= 1:
        return MTP_ATTN_PREFIX
    raise ValueError(
        f"layer {L} is neither a decoder layer (0..{int(cfg.num_hidden_layers) - 1}) nor the MTP layer "
        f"({int(cfg.mtp_layer_idx)}, num_nextn_predict_layers={int(cfg.num_nextn_predict_layers)}): pass weight_prefix="
    )


def canonical_attn_spec(cfg: MotifTTConfig, layer_idx: int) -> LayerSpec:
    """The layer's own ``LayerSpec``: ``cfg.layer(l)`` for a decoder layer the config builds (``l < cfg.num_layers``),
    ``cfg.mtp_layer_spec()`` for the MTP layer (``cfg.layer(53)`` does not exist and ``cfg.is_moe_layer(53)`` would say
    MoE; README §17)."""
    L = int(layer_idx)
    if 0 <= L < len(cfg.layers):
        return cfg.layer(L)
    if L == int(cfg.mtp_layer_idx) and int(cfg.num_nextn_predict_layers) >= 1:
        return cfg.mtp_layer_spec()
    raise ValueError(
        f"layer {L} is neither a decoder layer of this config (0..{len(cfg.layers) - 1}) nor the MTP layer "
        f"({int(cfg.mtp_layer_idx)}): pass spec="
    )


def resolve_attn_layer(
    cfg: MotifTTConfig,
    layer_idx: int,
    *,
    spec: Optional[LayerSpec] = None,
    weight_prefix: Optional[str] = None,
    cache: bool = True,
) -> Tuple[LayerSpec, str]:
    """``(spec, weight_prefix)`` of :class:`MotifAttention` (pure; raises before anything touches the device).

    * ``spec=None`` -> :func:`canonical_attn_spec`; a given spec must be a ``LayerSpec`` whose ``idx`` is ``layer_idx``
      (the index names the TT-cache part ``L<l>`` and the default weights).
    * ``weight_prefix=None`` -> :func:`attn_weight_prefix`; a given prefix is the module path (one trailing ``.`` is
      dropped), e.g. :data:`MTP_ATTN_PREFIX`.
    * ``cache=True`` additionally requires both to be the canonical ones of ``layer_idx``: the cache names
      (``attn.v2.wq_b`` ... under ``L<l>``) do not encode the spec (its softmax scale is folded into ``wq_b``) or the
      weight source, so a non-canonical pair would write wrong tensors into that layer's cache files (or load the
      layer's own tensors for it). Experiments with other specs / prefixes use ``cache=False``.
    """
    L = int(layer_idx)
    try:
        canon_spec: Optional[LayerSpec] = canonical_attn_spec(cfg, L)
    except ValueError:
        canon_spec = None
    try:
        canon_prefix: Optional[str] = attn_weight_prefix(cfg, L)
    except ValueError:
        canon_prefix = None
    if spec is None:
        if canon_spec is None:
            raise ValueError(f"MotifAttention layer {L}: no LayerSpec for this index in the config; pass spec=")
        spec = canon_spec
    elif not isinstance(spec, LayerSpec):
        raise TypeError(f"spec must be a model_config.LayerSpec, got {type(spec).__name__}")
    if int(spec.idx) != L:
        raise ValueError(f"spec.idx = {spec.idx} but layer_idx = {L}: the index names the TT-cache part and weights")
    if weight_prefix is None:
        if canon_prefix is None:
            raise ValueError(f"MotifAttention layer {L}: no default weight prefix for this index; pass weight_prefix=")
        prefix = canon_prefix
    else:
        prefix = str(weight_prefix).strip()
        prefix = prefix[:-1] if prefix.endswith(".") else prefix
        if not prefix or prefix.startswith(".") or ".." in prefix:
            raise ValueError(f"weight_prefix must be a module path like {MTP_ATTN_PREFIX!r}, got {weight_prefix!r}")
    if cache and (spec != canon_spec or prefix != canon_prefix):
        raise ValueError(
            f"MotifAttention layer {L} with cache=True needs the canonical spec {canon_spec} and weight prefix "
            f"{canon_prefix!r}, got {spec} / {prefix!r}: the TT-cache names ({_CACHE}.* under L{L:02d}) encode "
            "neither, so other weights or another softmax scale (folded into wq_b) would share that layer's files; "
            "pass cache=False"
        )
    return spec, prefix


class _AttnSource:
    """The 9 GDLA tensors of one layer from a weight source (lazy, loaded once): ``{prefix}.{name}.weight`` with
    ``prefix`` = ``model.layers.{layer_idx}.self_attn`` by default (the MTP layer: :data:`MTP_ATTN_PREFIX`)."""

    NAMES = ("wq_a", "q_norm", "wq_b", "wq_b_gate", "wkv_a", "kv_norm", "wkv_b", "lambda_proj", "wo")

    def __init__(self, source, layer_idx: int, prefix: Optional[str] = None):
        self.source, self.layer_idx = source, layer_idx
        self.prefix = prefix if prefix is not None else W.hf_name(layer_idx, "self_attn")
        self._t: Dict[str, torch.Tensor] = {}

    def name(self, key: str) -> str:
        """HF name of tensor ``key`` (one of :data:`NAMES`)."""
        return f"{self.prefix}.{key}.weight"

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self._t:
            self._t[key] = self.source.get(self.name(key))
        return self._t[key]


def latent_q_weight(src: _AttnSource, cfg: MotifTTConfig) -> torch.Tensor:
    """``Wq_lat = wq_a^T`` ``[4096, 1024]`` (columns ``[0, 1024)`` of ``weights.latent_projection_for_chip``;
    identical on every chip -> replicated)."""
    return W.latent_projection_for_chip(src["wq_a"], src["wkv_a"], src["lambda_proj"], cfg, 0)[:, : cfg.q_lora_rank]


def latent_kv_weight_for_chip(src: _AttnSource, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """``Wkv_lat`` ``[4096, 640]`` = columns ``[1024, 1664)`` of ``weights.latent_projection_for_chip``: ``[c_raw 512 |
    k_pe 64 | lambda 64]`` with chip ``tp``'s 8 lambda columns first (``[576, 584)``)."""
    full = W.latent_projection_for_chip(src["wq_a"], src["wkv_a"], src["lambda_proj"], cfg, tp)
    return full[:, cfg.q_lora_rank :].contiguous()


def _q_norm_gamma(src: _AttnSource, w: torch.Tensor) -> torch.Tensor:
    g = src["q_norm"]
    g = g.to(w.dtype) if w.dtype == torch.float64 else g.to(torch.float32)
    return g[:, None]


def wq_b_virtual_for_chip(src: _AttnSource, cfg: MotifTTConfig, tp: int, scale: float) -> torch.Tensor:
    """``weights.wq_b_for_chip(layout="interleaved")`` ``[1024, 10 x 192]`` with the head blocks in
    :func:`virtual_head_order`, the softmax scale and the ``q_norm`` gamma (input rows) folded in (exact algebra; one
    rounding at upload)."""
    H, hd = cfg.q_heads_per_chip, cfg.head_dim
    w = W.wq_b_for_chip(src["wq_b"], cfg, tp, layout="interleaved")
    w = w.reshape(w.shape[0], H, hd)[:, virtual_head_order(cfg), :].reshape(w.shape[0], H * hd)
    return (w * _q_norm_gamma(src, w) * float(scale)).contiguous()


def wq_b_gate_for_chip(src: _AttnSource, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """``weights.wq_b_gate_for_chip`` ``[1024, 8 x 128]`` with the ``q_norm`` gamma folded into the input rows."""
    w = W.wq_b_gate_for_chip(src["wq_b_gate"], cfg, tp)
    return (w * _q_norm_gamma(src, w)).contiguous()


def w_uk_virtual_for_chip(src: _AttnSource, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """Absorb matrices per virtual head ``[1, 10, 128, 512]`` (``weights.absorb_weights_for_chip``, gamma_kv folded):
    ``q_lat_h = q_nope_h @ W_UK'_{g(h)}``."""
    w = W.absorb_weights_for_chip(src["wkv_b"], src["kv_norm"], cfg, tp)  # [2, 128, 512]
    return w[virtual_head_groups(cfg)].unsqueeze(0).contiguous()


def w_uv_virtual_for_chip(src: _AttnSource, cfg: MotifTTConfig, tp: int) -> torch.Tensor:
    """Un-absorb matrices per virtual head ``[1, 10, 512, 128]`` (``weights.unabsorb_weights_for_chip``):
    ``U_h = o_lat_h @ W_UV'_{g(h)}^T``."""
    w = W.unabsorb_weights_for_chip(src["wkv_b"], src["kv_norm"], cfg, tp)  # [2, 512, 128]
    return w[virtual_head_groups(cfg)].unsqueeze(0).contiguous()


def lambda_expansion(cfg: MotifTTConfig) -> torch.Tensor:
    """``E [64, 1024]``: ``lam[:, :64] @ E`` repeats signal head ``s``'s lambda logit over its 128 output columns (rows
    >= 8 are zero, so the other chips' 56 lambda columns of ``Wkv_lat`` drop out). Every column holds exactly one 1, so
    ``sigmoid(lam @ E) == sigmoid(lam) @ E`` exactly (fp32 accumulation of one term)."""
    Sg, v = cfg.signal_heads_per_chip, cfg.v_head_dim
    E = torch.zeros(cfg.n_signal_heads, Sg * v)
    for s in range(Sg):
        E[s, s * v : (s + 1) * v] = 1.0
    return E


def noise_expansion(cfg: MotifTTConfig) -> torch.Tensor:
    """``X [G x 128, 8 x 128]``: ``[noise_g0 | noise_g1] @ X`` puts group ``g``'s noise head under each of its 4 signal
    heads (signal order). Exact (0/1, one term per column)."""
    G, r, v = cfg.kv_groups_per_chip, cfg.grouped_ratio, cfg.v_head_dim
    X = torch.zeros(G * v, G * r * v)
    eye = torch.eye(v)
    for s in range(G * r):
        g = s // r
        X[g * v : (g + 1) * v, s * v : (s + 1) * v] = eye
    return X


def decode_matmul_program_configs(cfg: MotifTTConfig) -> Dict[str, Any]:
    """Decode (M = one tile of lanes) program configs measured on this Galaxy (module docstring). Entries whose grid
    does not fit ``cfg.compute_grid`` fall back to ``None`` (ttnn's auto config).

    Module-local helper (README §10 rule 1 wants ``cfg.*_pc()`` builders; ``model_config`` has none for these yet):
    the four 1D-multicast configs already go through the shared ``mcast1d_matmul_pc``; the two per-head bmm configs
    are ``MatmulMultiCoreReuseProgramConfig`` literals. Requested shared builder: ``cfg.attn_decode_matmul_pc(name)``
    returning exactly these (grid, ``in0_block_w``, subblock, ``per_core_N``) per name."""
    gx, gy = cfg.compute_grid
    K, Kq, H = cfg.hidden_size // TILE, cfg.q_lora_rank // TILE, cfg.q_heads_per_chip

    def mc(grid, n_tiles, ibw, k_tiles):
        if grid[0] > gx or grid[1] > gy:
            return None
        return mcast1d_matmul_pc(grid, n_tiles, ibw, k_tiles)

    def reuse(n_tiles, ibw, sub_w):
        if gx * gy < H:  # one head (batch entry) per core
            return None
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=sub_w,
            per_core_M=1,
            per_core_N=n_tiles,
        )

    q_cols = cfg.q_lora_rank // TILE  # 32
    kv_cols = (cfg.kv_lora_rank + cfg.rope_dim + cfg.n_signal_heads) // TILE  # 20
    qb_cols = H * cfg.head_dim // TILE  # 60
    gate_cols = cfg.signal_heads_per_chip * cfg.v_head_dim // TILE  # 32
    tuned = getattr(cfg, "attn_mm_pcs", "release") == "tuned"  # Phase C D1: bitwise equal, -4.6 us per layer
    return {
        "q_lat": mc((8, 4), q_cols, 8, K),
        "kv_lat": mc((10, 2) if tuned else (5, 4), kv_cols, 16, K),
        "wq_b": mc((12, 5), qb_cols, 8 if tuned else 4, Kq),
        "gate": mc((8, 4), gate_cols, 8, Kq),
        "wo": None,  # auto config: 25 us, faster than every 1D config tried
        "w_uk": reuse(cfg.kv_lora_rank // TILE, 4, 2),
        "w_uv": reuse(cfg.v_head_dim // TILE, 4, 2),
    }


# ======================================================================================================================
# resumed / chunked prefill: the inputs of one chunk (features design §3.1 item 4, §3.2; README §15; work package 2b)
# ======================================================================================================================
def _check_host_i32(name: str, t: Any, shape: Optional[Tuple[int, ...]] = None, ndim: Optional[int] = None):
    if not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or t.device.type != "cpu":
        raise TypeError(f"{name} must be a CPU torch.int32 tensor, got {type(t).__name__} {getattr(t, 'dtype', None)}")
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(t.shape)}")
    if ndim is not None and t.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dims, got {tuple(t.shape)}")


def _cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _block_bounds(blocks: Sequence[int], block_size: int, latent_dim: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Tensor-args dim-0 slice bounds ``([blk, 0, 0, 0], [blk + 1, 1, bs, latent_dim])`` (int32 ``[4]`` pairs) of the
    cache blocks ``blocks``, in order. The bounds are device data, so one slice program serves every block id (G10a)."""
    bs, d = int(block_size), int(latent_dim)
    return [
        (torch.tensor([b, 0, 0, 0], dtype=torch.int32), torch.tensor([b + 1, 1, bs, d], dtype=torch.int32))
        for b in blocks
    ]


@dataclass(frozen=True)
class ChunkHostTables:
    """The host tables of one prefill chunk (pure torch; :func:`chunk_host_tables`, :func:`warmup_chunk_host_tables`).

    Attributes:
        path: ``"sp0"`` (start 0: draft-1 prefill) or ``"sp1"`` (start > 0: reads the cached prefix).
        start: ``a``, absolute position of the chunk's row 0 (a multiple of the block size; ``>=`` the SWA tail).
        bucket: ``C``, the chunk's rows (a prefill bucket).
        end: one past the last real row (``a < end <= a + C``).
        block_size: KV block size ``bs``.
        fill: ``[1, C / bs]`` ``paged_fill_cache`` table: block id, or ``-1`` = skip (shared blocks below ``w0``, pure
            padding blocks; ``prefill_plan.fill_table``).
        sdpa: sp1: ``[1, W']`` page table of the global chunked SDPA: the real ids of blocks ``[0, cdiv(end, bs))``,
            then 0 (never ``-1``; ``prefill_plan.sdpa_table``).
        start_idx: sp1: ``[1]`` = ``[start]`` (the chunked SDPA's ``chunk_start_idx_tensor``).
        rope: sp1: ``[C]`` RoPE table rows ``min(a + i, max_positions - 1)`` (``prefill_plan.rope_positions``).
        tail: sp1: ``[tail / bs]`` block ids of positions ``[a - tail, a)`` (``prefill_plan.tail_blocks``), shared by
            every SWA layer. ``None`` when the config has no sliding window.

    ``__post_init__`` checks every table alone and the tables against each other, so a builder that disagrees with
    itself fails here instead of silently reading stale keys on the device (global layers fill through ``fill`` and
    attend through ``sdpa``; SWA layers read their tail through ``tail``):

    * ``fill`` entries are ``-1`` or real block ids ``>= 1`` (block 0 is the null block that padded SDPA / tail reads
      land on). The written entries form one contiguous run that ends at the block of the last real row (``end - 1``):
      what ``prefill_plan.fill_table`` writes (blocks overlapping ``[w0, end)``; the last real row is always ``>= w0``).
      A table that writes nothing (all ``-1``) is a warm-up chunk (:func:`warmup_chunk_host_tables`), which reads only
      the null block, so the real-position checks below do not apply to it.
    * sp1: every written fill id is the SDPA table's id of the same logical block; the tail ids are the SDPA table's
      ids of positions ``[a - tail, a)``; the RoPE rows of the real rows are their positions ``a .. end - 1`` (only
      padded rows may clamp); for a chunk that writes, the SDPA table maps every real position (blocks ``[0,
      cdiv(end, bs))``) and every tail block to a real block id (never the null block).
    """

    path: str
    start: int
    bucket: int
    end: int
    block_size: int
    fill: torch.Tensor
    sdpa: Optional[torch.Tensor] = None
    start_idx: Optional[torch.Tensor] = None
    rope: Optional[torch.Tensor] = None
    tail: Optional[torch.Tensor] = None

    def __post_init__(self):
        a, C, bs, e = int(self.start), int(self.bucket), int(self.block_size), int(self.end)
        if self.path not in PP.PATHS:
            raise ValueError(f"path must be one of {PP.PATHS}, got {self.path!r}")
        if bs < TILE or bs % TILE or C < bs or C % bs or C % TILE:
            raise ValueError(f"bucket {C} must be a positive multiple of the block size {bs} (a multiple of {TILE})")
        if not a < e <= a + C:
            raise ValueError(f"chunk end {e} outside ({a}, {a + C}]")
        _check_host_i32("fill", self.fill, (1, C // bs))
        fill = self.fill[0]
        if bool(((fill < -1) | (fill == 0)).any()):
            raise ValueError(
                f"fill table entries must be block ids >= 1 or -1 (skip), got {fill.tolist()}: block 0 is the null "
                "block that padded SDPA and tail reads land on"
            )
        written = torch.nonzero(fill >= 0).flatten()
        last_real = _cdiv(e - a, bs) - 1  # the fill entry of the last real row (end - 1)
        if written.numel() and (
            int(written[-1]) - int(written[0]) + 1 != written.numel() or int(written[-1]) != last_real
        ):
            raise ValueError(
                f"fill table {fill.tolist()} must write one contiguous run of blocks ending at entry {last_real} (the "
                f"block of the last real row {e - 1}): only blocks below w0 (shared) and pure padding blocks are "
                "skipped (prefill_plan.fill_table)"
            )
        sp1_fields = (self.sdpa, self.start_idx, self.rope, self.tail)
        if self.path == PP.SP0:
            if a != 0 or any(t is not None for t in sp1_fields):
                raise ValueError("an sp0 chunk starts at 0 and has no SDPA table, start index, RoPE rows or tail")
            return
        if a <= 0 or a % bs:
            raise ValueError(f"an sp1 chunk starts at a positive multiple of the block size {bs}, got {a}")
        if self.sdpa is None or self.start_idx is None or self.rope is None:
            raise ValueError("an sp1 chunk needs its SDPA table, start index and RoPE rows")
        _check_host_i32("sdpa", self.sdpa, ndim=2)
        W_ = int(self.sdpa.shape[1])
        if int(self.sdpa.shape[0]) != 1 or W_ % PP.SDPA_TABLE_WIDTH_MULTIPLE or W_ * bs < a + C:
            raise ValueError(
                f"SDPA table {tuple(self.sdpa.shape)} must be [1, W'] with W' a multiple of "
                f"{PP.SDPA_TABLE_WIDTH_MULTIPLE} covering the chunk's {a + C} positions"
            )
        if int(self.sdpa.min()) < 0:
            raise ValueError("an SDPA page table never holds -1: the SDPA reader maps every entry as a block id (D5)")
        _check_host_i32("start_idx", self.start_idx, (1,))
        if int(self.start_idx[0]) != a:
            raise ValueError(f"start_idx {self.start_idx.tolist()} != start {a}")
        _check_host_i32("rope", self.rope, (C,))
        sdpa, first = self.sdpa[0], a // bs
        if written.numel() and not torch.equal(fill[written], sdpa[first + written]):
            raise ValueError(
                f"the fill table writes blocks {fill[written].tolist()} but the SDPA table reads "
                f"{sdpa[first + written].tolist()} at the same logical blocks: global layers fill first and attend "
                "through the SDPA table, so they would read stale keys"
            )
        if written.numel() and int(sdpa[: _cdiv(e, bs)].min()) < 1:
            raise ValueError(
                f"the SDPA table maps a real position (blocks [0, {_cdiv(e, bs)})) to the null block 0: "
                f"{sdpa[: _cdiv(e, bs)].tolist()}"
            )
        if not torch.equal(self.rope[: e - a], torch.arange(a, e, dtype=torch.int32)):
            raise ValueError(
                f"the RoPE rows of the real rows must be their positions {a} .. {e - 1} (only padded rows clamp), got "
                f"{self.rope[: e - a].tolist()[:8]} ..."
            )
        if self.tail is not None:
            _check_host_i32("tail", self.tail, ndim=1)
            nt = int(self.tail.numel())
            if nt * bs > a or int(self.tail.min()) < 0:
                raise ValueError(f"SWA tail blocks {self.tail.tolist()} must be block ids of positions before {a}")
            if not torch.equal(self.tail, sdpa[first - nt : first]):
                raise ValueError(
                    f"SWA tail blocks {self.tail.tolist()} are not the SDPA table's blocks of positions "
                    f"[{a - nt * bs}, {a}): {sdpa[first - nt : first].tolist()}"
                )
            if written.numel() and int(self.tail.min()) < 1:
                raise ValueError(f"SWA tail blocks {self.tail.tolist()} of a real chunk hold the null block 0")

    @property
    def is_sp1(self) -> bool:
        return self.path == PP.SP1

    @property
    def is_packed(self) -> bool:
        """A packed pass (:class:`PackedHostTables`): never for a chunk."""
        return False

    @property
    def reads_cache(self) -> bool:
        """The chunk reads the paged cache (sp1; review edit R-E11: the guard every cache reader keys on)."""
        return self.is_sp1

    def tail_bounds(self, latent_dim: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Tensor-args slice bounds of every tail block, in position order: ``([blk, 0, 0, 0], [blk + 1, 1, bs,
        latent_dim])`` int32 ``[4]`` pairs (empty for sp0 / no tail)."""
        if self.tail is None:
            return []
        return _block_bounds(self.tail.tolist(), self.block_size, latent_dim)


def chunk_host_tables(cfg: MotifTTConfig, plan: "PP.RowPlan", chunk: "PP.ChunkPlan", page_table_row) -> ChunkHostTables:
    """Host tables of ``chunk`` of ``plan`` (``cfg.plan_prefill_row(start, end)``) for a request whose vLLM block ids
    are ``page_table_row`` (``[W]``, position order; ``PrefillRequest.page_table``). All the rules are
    ``prefill_plan``'s: fill table, sp1 SDPA table of width ``cfg.sp1_page_table_width``, RoPE rows clamped to
    ``cfg.max_model_len``, the ``cfg.prefill_swa_tail`` tail blocks."""
    bs = int(cfg.kv_block_size)
    if int(plan.block_size) != bs:
        raise ValueError(f"plan block size {plan.block_size} != the config's {bs}")
    if chunk not in plan.chunks:
        raise ValueError("chunk does not belong to plan")
    fill = PP.fill_table(page_table_row, chunk, plan.w0, bs)[None].contiguous()
    if not chunk.is_sp1:
        return ChunkHostTables(PP.SP0, 0, int(chunk.bucket), int(chunk.end), bs, fill)
    tail_rows = int(cfg.prefill_swa_tail)
    return ChunkHostTables(
        PP.SP1,
        int(chunk.start),
        int(chunk.bucket),
        int(chunk.end),
        bs,
        fill,
        sdpa=PP.sdpa_table(page_table_row, chunk.end, bs, cfg.sp1_page_table_width)[None].contiguous(),
        start_idx=torch.tensor([int(chunk.start)], dtype=torch.int32),
        rope=PP.rope_positions(chunk, cfg.max_model_len),
        tail=PP.tail_blocks(page_table_row, chunk, bs, tail_rows) if tail_rows > 0 else None,
    )


def warmup_start(cfg: MotifTTConfig) -> int:
    """Chunk start of the sp1 warm-up inputs: the smallest multiple of ``cfg.prefill_resume_alignment`` that is
    ``>=`` the SWA tail (128 for A = 64 or 128)."""
    A = int(cfg.prefill_resume_alignment)
    return max(1, -(-max(int(cfg.prefill_swa_tail), 1) // A)) * A


def max_sp1_bucket(cfg: MotifTTConfig) -> int:
    """Largest bucket of ``cfg.prefill_span_buckets`` an sp1 chunk may use: on a model with a sliding window the
    square ``[tail ‖ chunk]`` SDPA of the SWA layers has ``prefill_swa_tail + C`` rows, which must not exceed
    ``max_model_len`` (the longest single-shot SDPA validated on this model; ``MotifAttention`` refuses longer ones).
    8192 for the default config (max_model_len 32768, span cap 8192); 16384 with ``MOTIF3_PREFILL_MAX_BUCKET=32768``;
    4096 when ``max_model_len`` is 8192 (the span cap is then 8192 too). Warm sp1 only up to this bucket; the planner
    must not emit an sp1 chunk with a larger one. Raises when no bucket fits."""
    tail = int(cfg.prefill_swa_tail)
    ok = [int(b) for b in cfg.prefill_span_buckets if tail == 0 or int(b) + tail <= int(cfg.max_model_len)]
    if not ok:
        raise ValueError(f"no prefill bucket of {cfg.prefill_span_buckets} fits an sp1 chunk (+{tail}-row SWA tail)")
    return max(ok)


def warmup_chunk_host_tables(cfg: MotifTTConfig, path: str, bucket: int) -> ChunkHostTables:
    """Tables of a warm-up chunk of ``(path, bucket)`` that writes nothing and reads only the null block 0 (features
    design §3.11): fill table all ``-1``; sp1 at :func:`warmup_start` with an all-zero SDPA table, tail blocks 0 and
    the RoPE rows of that start (``end`` = ``min(start + C, max_model_len)``: every row up to the table end counts as
    real). One call per ``(path, bucket)`` compiles every program the bucket needs. sp1 buckets above
    :func:`max_sp1_bucket` raise (no sp1 chunk of that size can run)."""
    bs, C = int(cfg.kv_block_size), int(bucket)
    fill = torch.full((1, C // bs), -1, dtype=torch.int32)
    if path == PP.SP0:
        return ChunkHostTables(PP.SP0, 0, C, C, bs, fill)
    if path != PP.SP1:
        raise ValueError(f"path must be one of {PP.PATHS}, got {path!r}")
    if C > max_sp1_bucket(cfg):
        raise ValueError(
            f"no sp1 chunk of bucket {C} can run: the SWA layers' square [tail | chunk] SDPA would have "
            f"{int(cfg.prefill_swa_tail) + C} rows > max_model_len {cfg.max_model_len}; sp1 chunks use buckets <= "
            f"max_sp1_bucket(cfg) = {max_sp1_bucket(cfg)} (span cap {cfg.max_prefill_span})"
        )
    a = warmup_start(cfg)
    P = int(cfg.max_model_len)
    tail_rows = int(cfg.prefill_swa_tail)
    return ChunkHostTables(
        PP.SP1,
        a,
        C,
        min(a + C, P),
        bs,
        fill,
        sdpa=torch.zeros(1, cfg.sp1_page_table_width, dtype=torch.int32),
        start_idx=torch.tensor([a], dtype=torch.int32),
        rope=torch.arange(a, a + C, dtype=torch.int32).clamp_(max=P - 1),
        tail=torch.zeros(tail_rows // bs, dtype=torch.int32) if tail_rows > 0 else None,
    )


# ======================================================================================================================
# packed prefill passes (P5): the host tables of one pass (docs/p5_t64/P5_T64_DESIGN.md §3.2-§3.5)
# ======================================================================================================================
@dataclass(frozen=True)
class PackedHostTables:
    """The host tables of one packed prefill pass (pure torch; :func:`packed_host_tables`,
    :func:`warmup_packed_host_tables`): ``B`` segments of ``S`` rows, ``T = B * S`` rows in all, segment ``k`` at
    packed rows ``[k S, k S + S)`` (module docstring, "Packed (multi-row) prefill passes").

    Attributes:
        path: ``"pk0"`` (sp0 segments, start 0) or ``"pk1"`` (sp1 segments at one common start).
        segments: ``B``, one of ``generator_api.PACK_BATCHES`` (dummy segments included).
        seg_rows: ``S``, one of ``generator_api.PACK_SEG_BUCKETS`` (pk0) / ``PACK_SP1_SEG_BUCKETS`` (pk1), a multiple
            of the block size: every segment starts on a block, and the fill kernel maps packed row ``i`` to entry
            ``i // bs``.
        bucket: ``T = B * S``, the bucket every row-local program of the pass runs at.
        start: the common start ``a`` (pk1: a positive multiple of the block size); 0 (pk0).
        ends: ``int32 [B]``: one past each segment's last real row (absolute position).
        block_size: KV block size ``bs``.
        fill: ``int32 [1, T / bs]``: entries ``[k S / bs, (k + 1) S / bs)`` = segment ``k``'s ``fill_table``.
        rope: ``int32 [T]``: RoPE table rows; ``[k S, k S + S)`` = segment ``k``'s ``rope_positions``.
        sdpa: pk1: ``int32 [B, W']``, row ``k`` = segment ``k``'s SDPA table (real ids, then 0; never -1).
        start_idx: pk1: ``int32 [1]`` = ``[start]``.
        tail: pk1: ``int32 [B, tail / bs]``, row ``k`` = segment ``k``'s SWA tail blocks (positions ``[a - tail,
            a)``); None without a sliding window.
        tails: pk1: the SWA tail variant (review edit R-E2): ``"shared"`` (every row of ``tail`` is the same: one
            gather, then ``repeat``) or ``"distinct"`` (a gather per segment; also valid for equal rows, which is how
            the warm-up compiles it). None for pk0.
        dummies: the last ``dummies`` segments are dummies. They write nothing (fill entries all -1) and copy
            segment 0's end and RoPE rows, and in pk1 its SDPA row and tail blocks (R-E2). So they read only what
            segment 0 reads, and a pass whose real segments share one prefix stays ``shared``.

    ``__post_init__`` checks the pass structure, then runs the :class:`ChunkHostTables` checks on every segment's slice
    (:meth:`segment`: the per-chunk tables of the segment's chunk re-bucketed to ``S``, review edit R-E12): fill ids vs
    SDPA ids, tail ids vs SDPA ids, the RoPE rows of the real rows at their positions, one contiguous written run that
    ends at the block of the last real row, never -1 in an SDPA table, never the null block in a fill table. A segment
    that writes nothing is a dummy or a warm-up segment.
    """

    path: str
    segments: int
    seg_rows: int
    bucket: int
    start: int
    ends: torch.Tensor
    block_size: int
    fill: torch.Tensor
    rope: torch.Tensor
    sdpa: Optional[torch.Tensor] = None
    start_idx: Optional[torch.Tensor] = None
    tail: Optional[torch.Tensor] = None
    tails: Optional[str] = None
    dummies: int = 0

    def __post_init__(self):
        B, S, T, a, bs = int(self.segments), int(self.seg_rows), int(self.bucket), int(self.start), int(self.block_size)
        if self.path not in PACKED_PASS_KINDS:
            raise ValueError(f"packed path must be one of {PACKED_PASS_KINDS}, got {self.path!r}")
        sizes = PACK_SEG_BUCKETS if self.path == PK0 else PACK_SP1_SEG_BUCKETS
        if B not in PACK_BATCHES or S not in sizes or T != B * S:
            raise ValueError(
                f"a {self.path} pass is B in {PACK_BATCHES} segments of S in {sizes} rows, T = B * S rows; got "
                f"B = {B}, S = {S}, T = {T}"
            )
        if bs < TILE or bs % TILE or S % bs:
            raise ValueError(f"segment rows {S} must be a multiple of the block size {bs} (a multiple of {TILE})")
        if not 0 <= int(self.dummies) < B:
            raise ValueError(f"dummies must be in [0, {B}) (segment 0 is real), got {self.dummies}")
        _check_host_i32("fill", self.fill, (1, T // bs))
        _check_host_i32("rope", self.rope, (T,))
        _check_host_i32("ends", self.ends, (B,))
        if int(self.rope.min()) < 0:
            raise ValueError("RoPE rows are table positions >= 0")
        if self.path == PK0:
            if a != 0 or any(t is not None for t in (self.sdpa, self.start_idx, self.tail, self.tails)):
                raise ValueError("a pk0 pass starts at 0 and has no SDPA tables, start index, SWA tails or variant")
        else:
            if self.sdpa is None or self.start_idx is None:
                raise ValueError("a pk1 pass needs its SDPA tables and start index")
            _check_host_i32("sdpa", self.sdpa, ndim=2)
            if int(self.sdpa.shape[0]) != B:
                raise ValueError(f"pk1 SDPA tables {tuple(self.sdpa.shape)}: the batched op reads one row per segment")
            if self.tail is None:
                if self.tails not in (None,) + tuple(PK1_TAIL_VARIANTS):
                    raise ValueError(f"tail variant must be one of {PK1_TAIL_VARIANTS} or None, got {self.tails!r}")
            else:
                _check_host_i32("tail", self.tail, ndim=2)
                if int(self.tail.shape[0]) != B:
                    raise ValueError(f"pk1 SWA tails {tuple(self.tail.shape)}: one row per segment ({B})")
                if self.tails not in PK1_TAIL_VARIANTS:
                    raise ValueError(
                        f"a pk1 pass needs its SWA tail variant, one of {PK1_TAIL_VARIANTS}, got {self.tails!r}"
                    )
                if self.tails == SHARED_TAILS and not bool((self.tail == self.tail[:1]).all()):
                    raise ValueError(
                        f"shared tails gather segment 0's tail once for the whole pass, but the segments' tail blocks "
                        f"differ: {self.tail.tolist()} (use 'distinct'; review edit R-E2)"
                    )
        for k in range(B):
            try:
                self.segment(k)  # the ChunkHostTables checks of the segment's slice (R-E12)
            except ValueError as e:
                raise ValueError(f"{self.path} segment {k} of {B}: {e}") from e
            if self.path == PK0:  # an sp0 chunk carries no RoPE rows: the pass's rows must be the positions 0 .. e - 1
                e = int(self.ends[k])
                if not torch.equal(self.rope[k * S : k * S + e], torch.arange(e, dtype=torch.int32)):
                    raise ValueError(
                        f"pk0 segment {k}: the RoPE rows of its real rows must be the positions 0 .. {e - 1}, got "
                        f"{self.rope[k * S : k * S + min(e, 8)].tolist()} ..."
                    )
        n = S // bs
        for k in range(B - int(self.dummies), B):
            same = (
                int(self.ends[k]) == int(self.ends[0])
                and torch.equal(self.rope[k * S : (k + 1) * S], self.rope[:S])
                and (self.sdpa is None or torch.equal(self.sdpa[k], self.sdpa[0]))
                and (self.tail is None or torch.equal(self.tail[k], self.tail[0]))
            )
            if bool((self.fill[0, k * n : (k + 1) * n] >= 0).any()) or not same:
                raise ValueError(
                    f"dummy segment {k} must write nothing (fill entries -1) and copy segment 0's end, RoPE rows, SDPA "
                    "row and tail blocks (review edit R-E2)"
                )

    def segment(self, k: int) -> ChunkHostTables:
        """Segment ``k``'s tables as one chunk of ``seg_rows`` rows (the per-chunk tables re-bucketed to ``S``, review
        edit R-E12): ``fill [1, S / bs]``; pk1 also its SDPA row ``[1, W']``, ``start_idx``, RoPE rows ``[S]`` and tail
        blocks."""
        B, S, bs, k = int(self.segments), int(self.seg_rows), int(self.block_size), int(k)
        if not 0 <= k < B:
            raise IndexError(f"segment {k} outside the pass's {B} segments")
        n = S // bs
        fill = self.fill[:, k * n : (k + 1) * n].contiguous()
        if self.path == PK0:
            return ChunkHostTables(PP.SP0, 0, S, int(self.ends[k]), bs, fill)
        return ChunkHostTables(
            PP.SP1,
            int(self.start),
            S,
            int(self.ends[k]),
            bs,
            fill,
            sdpa=self.sdpa[k : k + 1].contiguous(),
            start_idx=self.start_idx,
            rope=self.rope[k * S : (k + 1) * S].contiguous(),
            tail=None if self.tail is None else self.tail[k].contiguous(),
        )

    @property
    def is_sp1(self) -> bool:
        """A single sp1 chunk: never for a pass (:attr:`reads_cache` covers pk1)."""
        return False

    @property
    def is_packed(self) -> bool:
        return True

    @property
    def reads_cache(self) -> bool:
        """The pass reads the paged cache (pk1; review edit R-E11)."""
        return self.path == PK1

    @property
    def real_segments(self) -> int:
        """Segments ``[0, B - dummies)`` hold real chunks."""
        return int(self.segments) - int(self.dummies)

    @property
    def end(self) -> int:
        """The largest segment end (:attr:`ends` holds each segment's)."""
        return int(self.ends.max())

    @property
    def shape(self) -> Tuple[Any, ...]:
        """The pass shape key, as ``prefill_plan.PrefillPass.shape`` and ``MotifTTConfig.packed_prefill_shapes()``
        name it: ``("pk0", T, S)``, ``("pk1", T, S, tails)``."""
        key = (self.path, int(self.bucket), int(self.seg_rows))
        return key if self.path == PK0 else key + (self.tails,)

    def head_row(self, k: int) -> int:
        """Packed row of segment ``k``'s last real row: ``k S + ends[k] - 1 - start`` (``PrefillPass.head_rows``)."""
        return int(k) * int(self.seg_rows) + int(self.ends[int(k)]) - 1 - int(self.start)

    def tail_bounds(self, latent_dim: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Tensor-args slice bounds of the tail blocks the pass gathers (:func:`_block_bounds`): ``shared``: segment
        0's blocks (the solo gather); ``distinct``: every segment's blocks, segment-major (``2 B`` pairs for bs 64).
        Empty for pk0 / no tail."""
        if self.tail is None:
            return []
        rows = self.tail[:1] if self.tails == SHARED_TAILS else self.tail
        return _block_bounds(rows.reshape(-1).tolist(), self.block_size, latent_dim)


def packed_host_tables(cfg: MotifTTConfig, pass_, requests: Sequence[Any], plans: Sequence["PP.RowPlan"]):
    """Host tables of the packed pass ``pass_`` (a ``prefill_plan.PrefillPass`` of kind ``pk0`` / ``pk1`` from
    ``prefill_plan.plan_prefill_passes``) of one call's ``requests`` (``PrefillRequest``-like: ``page_table``) and
    their row ``plans`` (``cfg.plan_prefill_row``): ``prefill_plan.pass_tables`` with this config's geometry (block
    size, SDPA width ``cfg.sp1_page_table_width``, RoPE rows clamped to ``cfg.max_model_len``, the
    ``cfg.prefill_swa_tail`` tail), checked as :class:`PackedHostTables`. Each segment's slice is the per-chunk tables
    of its chunk re-bucketed to ``S`` (R-E12); dummies write nothing and copy segment 0 (R-E2). Pure host: raises
    ``ValueError`` before any device work. Solo passes keep :func:`chunk_host_tables` of their chunk."""
    kind = getattr(pass_, "kind", None)
    if kind not in PACKED_PASS_KINDS:
        raise ValueError(
            f"packed_host_tables takes a pass of kind {PACKED_PASS_KINDS}, got {kind!r} (a solo pass runs its chunk: "
            "chunk_host_tables)"
        )
    T = int(pass_.tokens)
    if T not in cfg.prefill_span_buckets:
        raise ValueError(
            f"a packed pass of T = {T} rows runs the bucket-T programs: T must be a prefill span bucket "
            f"{cfg.prefill_span_buckets}"
        )
    bs = int(cfg.kv_block_size)
    t = PP.pass_tables(
        pass_,
        requests,
        plans,
        block_size=bs,
        sdpa_width=cfg.sp1_page_table_width,
        max_positions=cfg.max_model_len,
        swa_tail=cfg.prefill_swa_tail,
    )
    return PackedHostTables(
        path=kind,
        segments=int(pass_.batch),
        seg_rows=int(pass_.seg_rows),
        bucket=T,
        start=int(pass_.start),
        ends=t.ends,
        block_size=bs,
        fill=t.fill,
        rope=t.rope,
        sdpa=t.sdpa,
        start_idx=t.start_idx,
        tail=t.tail,
        tails=t.tails if kind == PK1 else None,
        dummies=int(pass_.dummies),
    )


def warmup_packed_host_tables(
    cfg: MotifTTConfig, path: str, bucket: int, seg_rows: int, tails: Optional[str] = None
) -> PackedHostTables:
    """Tables of a warm-up pass of the packed shape ``(path, T, S[, tails])`` (an entry of
    ``cfg.packed_prefill_shapes()``: ``warmup_packed_host_tables(cfg, *shape)``) that writes nothing and reads only the
    null block 0 (design §3.5): every segment is the warm-up chunk of :func:`warmup_chunk_host_tables` at ``S`` rows
    (fill all -1; pk1 at :func:`warmup_start` with all-zero SDPA rows and tail blocks 0). pk1 needs ``tails``: the
    ``distinct`` variant gathers ``2 B`` null-block slices, so warming both variants per pk1 shape compiles both program
    sets (review edit R-E2). One attention call per shape and layer kind compiles every program of the packed SDPA
    path; the row-local programs at ``T`` are the solo bucket-``T`` ones."""
    T, S, bs = int(bucket), int(seg_rows), int(cfg.kv_block_size)
    if path not in PACKED_PASS_KINDS:
        raise ValueError(f"packed path must be one of {PACKED_PASS_KINDS}, got {path!r}")
    if S < 1 or T % S:
        raise ValueError(f"a packed pass of T = {T} rows holds whole segments of S = {S} rows")
    if T not in cfg.prefill_span_buckets:
        raise ValueError(f"packed pass T = {T} must be a prefill span bucket {cfg.prefill_span_buckets}")
    B = T // S
    fill = torch.full((1, T // bs), -1, dtype=torch.int32)
    if path == PK0:
        if tails is not None:
            raise ValueError("a pk0 pass has no SWA tail variant")
        w = warmup_chunk_host_tables(cfg, PP.SP0, S)
        rope = torch.arange(S, dtype=torch.int32).repeat(B)
        return PackedHostTables(PK0, B, S, T, 0, torch.full((B,), w.end, dtype=torch.int32), bs, fill, rope)
    if tails not in PK1_TAIL_VARIANTS:
        raise ValueError(f"a pk1 warm-up names its SWA tail variant, one of {PK1_TAIL_VARIANTS}, got {tails!r}")
    w = warmup_chunk_host_tables(cfg, PP.SP1, S)
    return PackedHostTables(
        PK1,
        B,
        S,
        T,
        int(w.start),
        torch.full((B,), w.end, dtype=torch.int32),
        bs,
        fill,
        w.rope.repeat(B),
        sdpa=w.sdpa.repeat(B, 1),
        start_idx=w.start_idx.clone(),
        tail=None if w.tail is None else w.tail[None].repeat(B, 1),
        tails=tails,
    )


def _replicate_i32(mesh_device, t: torch.Tensor, *, device: bool = True):
    """``int32`` ROW_MAJOR replicated mesh tensor (``device=False``: a host mesh tensor for
    ``ttnn.copy_host_to_device_tensor``)."""
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device if device else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@dataclass
class PrefillChunkInputs:
    """Device inputs of one prefill chunk, shared by every layer of the chunk (features design §3.2): build them once
    per chunk (:meth:`build`, or :meth:`write` into persistent ones of the same ``(path, bucket)``) and pass
    ``chunk=`` to every layer's :meth:`MotifAttention.forward_prefill` (and ``MotifAttention.fill_kv``, the MTP KV-only
    fill). Every tensor is replicated on all chips; a layer never frees them (:meth:`free` does).

    Attributes:
        path, start, bucket, end: host copies of the chunk (``prefill_plan.ChunkPlan``); the attention checks the sp1
            start alignment against them before every call. A packed pass (:class:`PackedHostTables`): ``"pk0"`` /
            ``"pk1"``, the common start (0 for pk0), ``T``, the largest segment end.
        fill_pt: ``[1, C / bs]`` int32 ROW_MAJOR fill table (``-1`` = skip).
        rot: sp1 and packed: ``{kind: (cos, sin)}`` ``[1, 1, C, 64]`` TILE, the RoPE rows of the rows' positions
            (``a .. a + C - 1``; packed: every segment's, review edit R-E11) (``rope.chunk_rope_tables``); ``None``
            after :meth:`write` with ``regather=False`` (the caller passes tables of the new positions). sp0: ``None``
            (each layer uses ``rope.prefill_cos_sin(kind, C)``, draft 1).
        sdpa_pt: sp1: ``[1, W']`` int32 ROW_MAJOR, 0-padded (global layers); pk1: ``[B, W']``, one row per segment.
        start_idx: sp1 / pk1: ``[1]`` int32 ROW_MAJOR = ``[start]`` (global layers' ``chunk_start_idx_tensor``).
        tail_bounds: sp1: per SWA tail block a ``(start [4], end [4])`` int32 ROW_MAJOR pair (SWA layers); pk1: the
            pairs of :meth:`PackedHostTables.tail_bounds` (``shared``: segment 0's; ``distinct``: every segment's,
            segment-major).
        rot_idx: sp1 and packed: the ``[1, C]`` uint32 gather indices behind ``rot``.
        segments, seg_rows: packed: ``B`` segments (dummies included) of ``S`` rows, ``bucket = B * S``; 1 / None for a
            chunk.
        tails: pk1: the SWA tail variant (``"shared"`` / ``"distinct"``, review edit R-E2); None otherwise.
        ends: packed: one past each segment's last real row (host ints, :meth:`segment_head_row`); ``()`` otherwise.
    """

    path: str
    start: int
    bucket: int
    end: int
    fill_pt: Any
    rot: Optional[Dict[str, Tuple[Any, Any]]] = None
    sdpa_pt: Any = None
    start_idx: Any = None
    tail_bounds: Tuple[Tuple[Any, Any], ...] = ()
    rot_idx: Any = None
    segments: int = 1
    seg_rows: Optional[int] = None
    tails: Optional[str] = None
    ends: Tuple[int, ...] = ()

    @property
    def is_sp1(self) -> bool:
        return self.path == PP.SP1

    @property
    def is_packed(self) -> bool:
        """A packed pass (``"pk0"`` / ``"pk1"``): the attention runs the batched per-segment SDPA paths."""
        return self.path in PACKED_PASS_KINDS

    @property
    def reads_cache(self) -> bool:
        """The rows read the paged cache: sp1 and pk1 (review edit R-E11; ``is_sp1`` is False for a pk1 pass)."""
        return self.path in (PP.SP1, PK1)

    @property
    def head_row(self) -> int:
        """Chunk-local row of the last real token (the LM head's row on the last chunk). A packed pass has one per
        segment (:meth:`segment_head_row`)."""
        if self.is_packed:
            raise ValueError(f"a {self.path} pass has one head row per segment: segment_head_row(k)")
        return int(self.end) - 1 - int(self.start)

    def segment_head_row(self, k: int) -> int:
        """Packed row of segment ``k``'s last real token: ``k S + ends[k] - 1 - start`` (``PrefillPass.head_rows``
        names the segments that need logits). For a chunk, ``k = 0`` gives :attr:`head_row`."""
        if not self.is_packed:
            if int(k) != 0:
                raise IndexError(f"a chunk is one segment, got segment {k}")
            return self.head_row
        if not 0 <= int(k) < len(self.ends):
            raise IndexError(f"segment {k} outside the pass's {len(self.ends)} segments")
        return int(k) * int(self.seg_rows) + int(self.ends[int(k)]) - 1 - int(self.start)

    @classmethod
    def upload(
        cls, mesh_device, cfg: MotifTTConfig, rope: MotifRope, host: Union[ChunkHostTables, PackedHostTables]
    ) -> "PrefillChunkInputs":
        """Upload ``host`` (eager: a few small host-to-device copies; sp1 chunks and packed passes add 2 x 2 RoPE
        gathers): a chunk's :class:`ChunkHostTables` or a packed pass's :class:`PackedHostTables` (which always
        gathers its RoPE rows, pk0 included: every segment starts at its own position 0 / ``a``, review edit
        R-E11)."""
        if host.block_size != cfg.kv_block_size:
            raise ValueError(f"chunk tables for block size {host.block_size}, the config has {cfg.kv_block_size}")
        packed = bool(getattr(host, "is_packed", False))
        reads = bool(getattr(host, "reads_cache", host.is_sp1))
        inp = cls(host.path, int(host.start), int(host.bucket), int(host.end), _replicate_i32(mesh_device, host.fill))
        if packed:
            inp.segments, inp.seg_rows, inp.tails = int(host.segments), int(host.seg_rows), host.tails
            inp.ends = tuple(int(e) for e in host.ends.tolist())
        if reads:
            inp.sdpa_pt = _replicate_i32(mesh_device, host.sdpa)
            inp.start_idx = _replicate_i32(mesh_device, host.start_idx)
            inp.tail_bounds = tuple(
                (_replicate_i32(mesh_device, s), _replicate_i32(mesh_device, e))
                for s, e in host.tail_bounds(cfg.kv_latent_dim)
            )
        if reads or packed:
            inp.rot_idx = rope.chunk_rot_idxs_device(host.rope)
            inp.rot = rope.chunk_rope_tables(inp.rot_idx)
        return inp

    @classmethod
    def build(
        cls, mesh_device, cfg: MotifTTConfig, rope: MotifRope, plan: "PP.RowPlan", chunk: "PP.ChunkPlan", page_table_row
    ) -> "PrefillChunkInputs":
        """:meth:`upload` of :func:`chunk_host_tables` (``plan`` = ``cfg.plan_prefill_row(start, end)`` of the
        request, ``page_table_row`` = its ``PrefillRequest.page_table``)."""
        return cls.upload(mesh_device, cfg, rope, chunk_host_tables(cfg, plan, chunk, page_table_row))

    @classmethod
    def warmup(
        cls,
        mesh_device,
        cfg: MotifTTConfig,
        rope: MotifRope,
        path: str,
        bucket: int,
        seg_rows: Optional[int] = None,
        tails: Optional[str] = None,
    ) -> "PrefillChunkInputs":
        """Inputs of a warm-up chunk (:func:`warmup_chunk_host_tables`) or, for ``path`` ``"pk0"`` / ``"pk1"``, of a
        warm-up packed pass of ``bucket = T`` rows in segments of ``seg_rows`` (:func:`warmup_packed_host_tables`;
        pk1 with its ``tails`` variant): nothing is written, only the null block is read. Run every ``(path, bucket)``
        and every packed shape (both pk1 variants) once before the decode capture (D12, review edit R-E2)."""
        if path in PACKED_PASS_KINDS:
            host = warmup_packed_host_tables(cfg, path, bucket, seg_rows, tails)
        elif seg_rows is not None or tails is not None:
            raise ValueError(f"seg_rows / tails describe a packed pass ({PACKED_PASS_KINDS}), not a {path!r} chunk")
        else:
            host = warmup_chunk_host_tables(cfg, path, bucket)
        return cls.upload(mesh_device, cfg, rope, host)

    def write(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        rope: MotifRope,
        host: Union[ChunkHostTables, PackedHostTables],
        *,
        regather: bool = True,
    ) -> None:
        """Rewrite these (persistent) inputs in place for another chunk of the same ``(path, bucket)`` (a packed
        pass: of the same shape ``(path, T, S[, tails])``) (``ttnn.copy_host_to_device_tensor``; the buffers keep
        their addresses). The previous chunk's gathered RoPE rows (``rot``) are freed in either case: they hold the
        old positions.

        ``regather=True``: gather the new rows from ``rot_idx`` here (eager, new tensors), for eager calls.
        ``regather=False``: no device allocation; ``rot`` becomes ``None``, so an eager ``forward_prefill(chunk=)`` /
        ``fill_kv(chunk=)`` refuses the inputs until tables of the new positions are passed in. That is the form of a
        captured prefill, which gathers ``rope.chunk_rope_tables(inp.rot_idx)`` inside the trace and passes
        ``dataclasses.replace(inp, rot=tables)`` (a trace must never read the eager ``rot``: its buffers are freed or
        replaced on every write)."""
        if (host.path, int(host.bucket)) != (self.path, int(self.bucket)):
            raise ValueError(f"inputs of ({self.path}, {self.bucket}) cannot hold a ({host.path}, {host.bucket}) chunk")
        packed = bool(getattr(host, "is_packed", False))
        reads = bool(getattr(host, "reads_cache", host.is_sp1))
        if packed:
            want = (int(self.segments), self.seg_rows, self.tails)
            if (int(host.segments), int(host.seg_rows), host.tails) != want:
                raise ValueError(
                    f"inputs of {self.path} B x S = {self.segments} x {self.seg_rows} (tails {self.tails}) cannot hold "
                    f"a pass of {host.segments} x {host.seg_rows} (tails {host.tails}): another program set"
                )
        if host.block_size != cfg.kv_block_size:
            raise ValueError(f"chunk tables for block size {host.block_size}, the config has {cfg.kv_block_size}")
        bounds = host.tail_bounds(cfg.kv_latent_dim)
        if reads and len(bounds) != len(self.tail_bounds):  # every check before the first copy
            raise ValueError(f"{len(bounds)} tail blocks, the inputs hold {len(self.tail_bounds)}")
        ttnn.copy_host_to_device_tensor(_replicate_i32(mesh_device, host.fill, device=False), self.fill_pt)
        if reads:
            ttnn.copy_host_to_device_tensor(_replicate_i32(mesh_device, host.sdpa, device=False), self.sdpa_pt)
            ttnn.copy_host_to_device_tensor(_replicate_i32(mesh_device, host.start_idx, device=False), self.start_idx)
            for (s, e), (s_dev, e_dev) in zip(bounds, self.tail_bounds):
                ttnn.copy_host_to_device_tensor(_replicate_i32(mesh_device, s, device=False), s_dev)
                ttnn.copy_host_to_device_tensor(_replicate_i32(mesh_device, e, device=False), e_dev)
        if reads or packed:
            ttnn.copy_host_to_device_tensor(rope.chunk_rot_idxs_host(host.rope), self.rot_idx)
            for cs in (self.rot or {}).values():  # the old chunk's rows: stale from here on
                for t in cs:
                    ttnn.deallocate(t)
            self.rot = rope.chunk_rope_tables(self.rot_idx) if regather else None
        self.start, self.end = int(host.start), int(host.end)
        if packed:
            self.ends = tuple(int(e) for e in host.ends.tolist())

    def tensors(self) -> List[Any]:
        """Every device tensor these inputs hold."""
        out = [self.fill_pt, self.sdpa_pt, self.start_idx, self.rot_idx]
        out += [t for pair in self.tail_bounds for t in pair]
        out += [t for cs in (self.rot or {}).values() for t in cs]
        return [t for t in out if t is not None]

    def free(self) -> None:
        """Deallocate every device tensor (after the chunk's last layer)."""
        for t in self.tensors():
            ttnn.deallocate(t)
        self.rot, self.tail_bounds = None, ()
        self.fill_pt = self.sdpa_pt = self.start_idx = self.rot_idx = None


class DecodeKVWriter(Protocol):
    """Decode KV write of one step, shared by every layer (features design §3.5; implemented by ``tt/kv_write.py``,
    owner KVW). ``MotifAttention.forward_decode(..., kv_write=w)`` calls ``w.write(kv_row, kv_cache, cur_pos=cur_pos,
    page_table=page_table)`` exactly once per layer, after the q path and before FlashMLA, in place of the draft-1
    8-lane ``paged_update_cache``.

    * ``kv_row``: this step's latent rows ``[1, 1, L, 576]`` bf16 TILE DRAM interleaved (L = the 8 lanes of this chip's
      DP row, or the T64 step's 16 rows ``[8 anchors | 8 drafts]``; ``[n | rope(k_pe)]``, exactly what draft 1 writes).
      Not consumed: the attention frees it afterwards.
    * ``kv_cache``: the layer's paged latent cache (``cfg.dtypes.kv_cache``), updated in place.
    * ``cur_pos`` / ``page_table``: the per-row ``[L]`` / ``[L, W]`` tensors the layer's FlashMLA reads (draft-lane rows
      included). The ``row`` mode writes through them; the split and KV-R modes through their own per-step tensors.

    Optional (``tt/kv_write.DecodeKVWrite`` has both; module docstring "T64 verify step"):

    * ``lanes_per_row``: the rows per DP row the writer writes (8, or 16 at T64); ``forward_decode`` refuses an ``x``
      with another row count.
    * ``flash_groups() -> [(row slice, cur_pos, page_table)]``: the global layers' FlashMLA calls (option A''), slices
      partitioning ``[0, L)`` in order: one group over all rows = one call (the T32 step), two B = 8 groups at 16 rows.
      A writer without it serves global layers at 8 rows only (one call on ``cur_pos`` / ``page_table``).

    Device ops only (it runs inside the decode trace); one object per step serves all 53 layers and the MTP layer."""

    def write(self, kv_row: Any, kv_cache: Any, *, cur_pos: Any, page_table: Any) -> None: ...


# ======================================================================================================================
# the module
# ======================================================================================================================
RotArg = Union[None, "ttnn.Tensor", Tuple[Any, Any], Dict[str, Tuple[Any, Any]]]


class MotifAttention:
    """GDLA attention of decoder layer ``layer_idx`` (design §2.3.4), or of the MTP layer. See the module docstring for
    the dataflow.

    Args:
        mesh_device: the opened (4, 8) (or (8, 4)) mesh.
        cfg: ``MotifTTConfig`` built with this mesh.
        layer_idx: decoder layer, or ``cfg.mtp_layer_idx`` (53) for the MTP layer. It names the TT-cache part
            (``L<l>``) and, by default, the spec and the weights.
        source: ``weights.HFWeightLoader`` or ``weights.DictWeightSource`` (HF names ``{weight_prefix}.{name}.weight``).
        ccl: ``MotifCCL`` of the mesh (``all_reduce(tp)`` after ``wo``).
        rope: ``MotifRope`` (needed for prefill tables and when ``forward_decode`` gets raw rot indices).
        cache: write / read the TT weight cache (``False`` for random weights: nothing is written).
        rope_mode: ``"hf"`` (fused ``rotary_embedding_hf``, G8 decision) or ``"composite"`` (fallback).
        matmul_program_configs: overrides of :func:`decode_matmul_program_configs` (keys ``q_lat``, ``kv_lat``,
            ``wq_b``, ``gate``, ``wo``, ``w_uk``, ``w_uv``; ``None`` = ttnn's auto config). Decode only.
        sdpa_prefill_fp32_acc: ``False`` (default): the shared ``sdpa_prefill`` role (G2: HiFi4, fp32 acc off,
            streaming SDPA kernel) for every prefill call. ``"auto"``: opt-in A/B for the layer-level decision -- fp32
            dest accumulation for every prefill SDPA call that has no window mask in effect (global layers; SWA layers
            at S <= window - 1 = 128, where the 129-key window masks nothing and is dropped), the role otherwise; fp32
            acc keeps the QK scores in fp32 (``qk_im_df``) but selects the legacy non-streaming kernel (module
            docstring: +40-55 % per global call at 16K-32K, 1.38 MB static CBs, not gate-validated). ``True``: always
            fp32 acc (diagnosis only: with ``sliding_window_size`` the legacy kernel returns wrong values for every
            S >= 256 and chunk config, PCC 0.958-0.989; ``test_sdpa_prefill_window_fp32_acc_upstream_bug``). The
            fp32 variant uses a module-local compute config until ``model_config`` has a shared role for it.
        require_l1_small: raise instead of warning when the mesh has no L1_SMALL region (module docstring,
            :func:`check_l1_small`).
        spec: the layer's ``LayerSpec`` (window, softmax scale, RoPE kind). Default: :func:`canonical_attn_spec`,
            i.e. ``cfg.layer(layer_idx)``, or ``cfg.mtp_layer_spec()`` for the MTP layer. ``spec.idx`` must equal
            ``layer_idx``.
        weight_prefix: HF module path of the 9 tensors. Default: :func:`attn_weight_prefix`, i.e.
            ``model.layers.{l}.self_attn``, or ``model.mtp_layers.0.self_attn`` for the MTP layer.
            With ``cache=True``, ``spec`` and ``weight_prefix`` must be the canonical ones (:func:`resolve_attn_layer`).
    """

    def __init__(
        self,
        mesh_device,
        cfg: MotifTTConfig,
        layer_idx: int,
        *,
        source,
        ccl: MotifCCL,
        rope: Optional[MotifRope] = None,
        cache: bool = True,
        rope_mode: str = "hf",
        matmul_program_configs: Optional[Dict[str, Any]] = None,
        sdpa_prefill_fp32_acc: Union[str, bool] = False,
        require_l1_small: bool = False,
        spec: Optional[LayerSpec] = None,
        weight_prefix: Optional[str] = None,
    ):
        if rope_mode not in ("hf", "composite"):
            raise ValueError(f"rope_mode must be 'hf' or 'composite', got {rope_mode!r}")
        if sdpa_prefill_fp32_acc not in ("auto", True, False):
            raise ValueError(f"sdpa_prefill_fp32_acc must be 'auto', True or False, got {sdpa_prefill_fp32_acc!r}")
        self.layer_idx = int(layer_idx)
        self.spec, self.weight_prefix = resolve_attn_layer(
            cfg, self.layer_idx, spec=spec, weight_prefix=weight_prefix, cache=cache
        )
        self.l1_small_bytes = check_l1_small(mesh_device, require=require_l1_small)
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.ccl = ccl
        self.rope = rope
        self.rope_mode = rope_mode
        self.sdpa_prefill_fp32_acc = sdpa_prefill_fp32_acc
        self.decode_pcs = decode_matmul_program_configs(cfg)
        self.decode_pcs.update(matmul_program_configs or {})

        # ---- per-chip geometry ------------------------------------------------------------------------------------
        self.H = cfg.q_heads_per_chip  # 10
        self.G = cfg.kv_groups_per_chip  # 2
        self.r = cfg.grouped_ratio  # 4
        self.Sg = cfg.signal_heads_per_chip  # 8
        self.nope, self.rope_dim, self.vdim = cfg.qk_nope_head_dim, cfg.rope_dim, cfg.v_head_dim  # 128, 64, 128
        self.rank = cfg.kv_lora_rank  # 512
        self.latent_dim = cfg.kv_latent_dim  # 576
        self.window = self.spec.sliding_window_size  # 129 | None
        self.scale = float(self.spec.softmax_scale)
        self.kind = self.spec.rope_kind  # "yarn" | "plain"
        self.lanes = cfg.lanes_per_row

        # ---- compute / program configs (README §4-5) --------------------------------------------------------------
        self.ckc_latent = cfg.compute_config("attn_latent")
        self.ckc_heads = cfg.compute_config("attn_heads")
        self.ckc_norm = cfg.compute_config("norm")
        self.ckc_sdpa_decode = cfg.compute_config("sdpa_decode")
        self.ckc_sdpa_prefill = cfg.compute_config("sdpa_prefill")  # G2-validated: HiFi4, fp32 acc off (default)
        role = cfg.compute_role("sdpa_prefill")
        # opt-in only (sdpa_prefill_fp32_acc="auto"/True): module-local until model_config has the requested
        # "sdpa_prefill_fp32" role (the sdpa_prefill role with fp32 dest acc on)
        self.ckc_sdpa_prefill_fp32 = make_compute_kernel_config(
            role.fidelity, True, approx=role.approx, packer_l1_acc=role.packer_l1_acc
        )
        # sp1 global (absorbed chunked SDPA over the latent cache): gate G9 -- fp32 dest acc is mandatory there (the
        # bf16-dest role accumulates the 576-wide QK^T in bf16: PCC 0.998, fails everywhere); the op has no window.
        self.ckc_sdpa_sp1_global = cfg.compute_config("sdpa_prefill_fp32")
        self.ckc_rope = rope.ckc if rope is not None else cfg.compute_config("rope")
        # G1: k_chunk 128, mandatory (ATTN-2); SWA layers use cfg.flash_mla_swa_mcph cores per head batch (A2: 4,
        # bitwise equal to the global layers' 16 on a 129-key window), global layers 16
        self.decode_pc = cfg.flash_mla_decode_pc("swa" if self.window is not None else "global")
        self.dtype = cfg.dtypes.activations
        # decode epilogue (Phase C D1, MOTIF3_ATTN_EPILOGUE): "fused" (default; tt/kernels/attn_combine.py, built lazily)
        # | "ops" (the release op chain)
        self.epilogue = getattr(cfg, "attn_epilogue", "ops")
        self._fused_combine = None
        # decode input chain (Phase F F1, MOTIF3_ATTN_IN): "ops" (default, the op chain) | "fused"
        # (tt/kernels/attn_in.py, one program after the latent projections, built lazily)
        self.attn_in = getattr(cfg, "attn_in", "ops")
        self._fused_input = None

        # ---- weights (ATTN-1) -----------------------------------------------------------------------------------
        src = _AttnSource(source, self.layer_idx, self.weight_prefix)
        L = self.layer_idx
        dt = cfg.dtypes.attention

        def up(fn, name, **kw):
            return W.as_tensor(
                fn,
                mesh_device=mesh_device,
                cfg=cfg,
                dtype=kw.pop("dtype", dt),
                layout=kw.pop("layout", ttnn.TILE_LAYOUT),
                cache_name=f"{_CACHE}.{name}" if cache else None,
                layer=L,
                **kw,
            )

        def stack(fn, dim):
            return lambda: W.stack_tp(fn, cfg, dim=dim)

        s = self.scale
        self.w_q_lat = up(lambda: latent_q_weight(src, cfg), "wq_lat")  # [4096, 1024] replicated
        self.w_kv_lat = up(stack(lambda tp: latent_kv_weight_for_chip(src, cfg, tp), 1), "wkv_lat", tp_dim=1)
        self.w_q_b = up(stack(lambda tp: wq_b_virtual_for_chip(src, cfg, tp, s), 1), "wq_b", tp_dim=1)
        self.w_gate = up(stack(lambda tp: wq_b_gate_for_chip(src, cfg, tp), 1), "wq_b_gate", tp_dim=1)
        self.w_uk = up(stack(lambda tp: w_uk_virtual_for_chip(src, cfg, tp), 1), "w_uk", tp_dim=1)
        self.w_uv = up(stack(lambda tp: w_uv_virtual_for_chip(src, cfg, tp), 1), "w_uv", tp_dim=1)
        self.w_kv_expand = up(
            stack(lambda tp: W.prefill_kv_expansion_for_chip(src["wkv_b"], src["kv_norm"], cfg, tp), 1),
            "kv_expand",
            tp_dim=1,
        )
        self.w_o = up(stack(lambda tp: W.wo_for_chip(src["wo"], cfg, tp), 0), "wo", tp_dim=0)
        self.lam_expand = up(lambda: lambda_expansion(cfg), "lam_expand")  # constants, replicated
        self.noise_expand = up(lambda: noise_expansion(cfg), "noise_expand")

        # decode KV-update input: [1, L, 1(32), 576] HEIGHT_SHARDED, one lane per core (G7 / ATTN-3)
        grid = mesh_device.compute_with_storage_grid_size()
        self._update_cores = ttnn.num_cores_to_corerangeset(self.lanes, grid, row_wise=True)
        self.update_mc = ttnn.create_sharded_memory_config(
            shape=(TILE, self.latent_dim),
            core_grid=self._update_cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # ==================================================================================================================
    # per-step helpers (call ONCE per decode step, shared by all layers)
    # ==================================================================================================================
    @staticmethod
    def decode_rope_tables(rope: MotifRope, rot_idxs, kinds: Sequence[str] = ("yarn", "plain")) -> Dict[str, Tuple]:
        """``{kind: (cos, sin)}`` ``[1, 1, 32, 64]`` TILE (row t = row t of this DP row: lane t, or in the T64 step row
        t of ``[8 anchors | 8 drafts]``; ``rope.decode_cos_sin(..., layout="rows")``) from the per-step ``rot_idxs [1,
        32]`` uint32 device tensor (``rope.rot_idxs_host(positions, rows_per_dp=...)``). Trace-safe (2 embedding
        gathers per kind). Pass the dict as ``rot=`` to every layer's :meth:`forward_decode`."""
        return {k: rope.decode_cos_sin(k, rot_idxs, layout="rows") for k in kinds}

    @staticmethod
    def active_mask_from_cur_pos(cur_pos, lanes: int = 8, width: int = MASK_WIDTH):
        """Trace-safe ``[1, 1, L, width]`` bf16 TILE 0/1 mask (1 = lane active, ``cur_pos >= 0``) from the ``[L]`` int32
        ROW_MAJOR ``cur_pos`` device tensor (``lanes`` = L: 8, or 16 for the T64 step's ``kv_write.cur_pos``). Build it
        once per step and pass it as ``active=`` to every layer (``width`` 1024 = the ``wo`` input width: a full-width
        predicate is 2x cheaper than a broadcast one; ``width=1`` gives the broadcastable ``[1, 1, L, 1]``)."""
        p = ttnn.reshape(cur_pos, (1, 1, lanes, 1))
        p = ttnn.typecast(p, ttnn.float32)
        t = ttnn.to_layout(p, ttnn.TILE_LAYOUT)
        ttnn.deallocate(p)
        m = ttnn.ge(t, 0.0)
        ttnn.deallocate(t)
        if m.dtype != ttnn.bfloat16:
            m2 = ttnn.typecast(m, ttnn.bfloat16)
            ttnn.deallocate(m)
            m = m2
        if width != 1:
            m2 = ttnn.repeat(m, ttnn.Shape([1, 1, 1, int(width)]))
            ttnn.deallocate(m)
            m = m2
        return m

    @staticmethod
    def active_mask_host(
        positions: torch.Tensor,
        cfg: MotifTTConfig,
        mesh_device,
        *,
        device=None,
        width: int = MASK_WIDTH,
        rows_per_dp: Optional[int] = None,
    ):
        """Host (``device=None``, for ``ttnn.copy_host_to_device_tensor``) or device mesh tensor ``[1, 1, L, width]``
        bf16 per DP row from the positions (``-1`` = inactive) of ``dp * L`` rows in DP-row order: by default the 32
        lane positions (L = 8, lane order); ``rows_per_dp=16`` takes the T64 step's 64 rows (row ``16 r + j``)."""
        L = decode_rows_per_dp(cfg, rows_per_dp)
        pos = torch.as_tensor(positions).reshape(-1)
        if int(pos.numel()) != cfg.dp * L:
            raise ValueError(f"expected {cfg.dp * L} positions ({cfg.dp} DP rows x {L}), got {int(pos.numel())}")
        rows = (pos >= 0).to(torch.float32).reshape(cfg.dp, 1, L, 1).expand(-1, -1, -1, int(width))
        return _shard_rows(rows.contiguous(), cfg, mesh_device, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    # ==================================================================================================================
    # shared building blocks
    # ==================================================================================================================
    def _linear(self, x, w, *, ckc, pc=None, dtype=None, activation=None):
        kw = {}
        if pc is not None:
            kw["program_config"] = pc
        if activation is not None:
            kw["activation"] = activation
        return ttnn.linear(
            x, w, dtype=dtype or self.dtype, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw
        )

    def _pc(self, name: str, decode: bool):
        return self.decode_pcs.get(name) if decode else None

    def _project(self, x, decode: bool):
        """Shared projections. ``x [1,1,T,4096]`` -> ``q`` ``[1,1,T,1920]`` (virtual head order, scaled), ``g`` =
        sigmoid(gate) ``[1,1,T,1024]``, ``n`` ``[1,1,T,512]``, ``kpe`` ``[1,1,T,64]`` (raw), ``lam`` ``[1,1,T,64]``
        (lambda logits, the chip's 8 first)."""
        cq = self._linear(x, self.w_q_lat, ckc=self.ckc_latent, pc=self._pc("q_lat", decode), dtype=ttnn.float32)
        kvl = self._kv_latent(x, decode)
        cq_n = ttnn.rms_norm(cq, epsilon=self.cfg.rms_norm_eps, compute_kernel_config=self.ckc_norm)
        ttnn.deallocate(cq)
        q = self._linear(cq_n, self.w_q_b, ckc=self.ckc_heads, pc=self._pc("wq_b", decode))
        g = self._linear(cq_n, self.w_gate, ckc=self.ckc_heads, pc=self._pc("gate", decode), activation="sigmoid")
        ttnn.deallocate(cq_n)
        n, kpe, lam = self._split_kv(kvl)
        return q, g, n, kpe, lam

    def _kv_latent(self, x, decode: bool):
        """``x [1,1,T,4096] @ Wkv_lat`` -> ``[1,1,T,640]`` = ``[c_raw 512 | kpe 64 | lam 64]`` (the kv half of
        :meth:`_project`; prefill = ttnn's auto config, decode = the measured 1D-multicast config)."""
        return self._linear(x, self.w_kv_lat, ckc=self.ckc_latent, pc=self._pc("kv_lat", decode))

    def _split_kv(self, kvl):
        """``[c_raw | kpe | lam]`` -> ``n = rms_norm(c_raw)`` ``[1,1,T,512]``, ``kpe`` ``[1,1,T,64]`` (raw), ``lam``
        ``[1,1,T,64]``. Consumes ``kvl``."""
        # channel splits (one op each; tile-aligned regions): [c_raw | kpe lam] -> [kpe | lam]
        c_raw, rest = ttnn.experimental.nlp_create_q_heads_split(kvl, num_heads=1, split_head_dim=self.rank)
        ttnn.deallocate(kvl)
        kpe, lam = ttnn.experimental.nlp_create_q_heads_split(rest, num_heads=1, split_head_dim=self.rope_dim)
        ttnn.deallocate(rest)
        n = ttnn.rms_norm(c_raw, epsilon=self.cfg.rms_norm_eps, compute_kernel_config=self.ckc_norm)
        ttnn.deallocate(c_raw)
        return n, kpe, lam

    def _fill_latent(self, n, k_pe, page_table, kv_cache, *, taps: Optional[Dict[str, Any]] = None):
        """Write the latent ``typecast(concat(n, k_pe))`` ``[1, 1, T, 576]`` (cache dtype; a raw tile copy) into
        ``kv_cache`` with ``paged_fill_cache(..., batch_idx=0)`` through the first ``cfg.prefill_page_table_entries(T)``
        entries of ``page_table [1, >= cdiv(T, block)]`` (a wider table is sliced; pass the exact width to keep one
        program shape per bucket, M10). Row ``i`` goes through entry ``i // block``; ``-1`` entries are skipped. Does
        not consume ``n`` / ``k_pe``. ``taps["kv_row"]`` (when given) receives the bf16 latent before the typecast."""
        self._fill_table_entries(page_table, int(n.shape[-2]))
        kv_row = ttnn.concat([n, k_pe], dim=-1)  # [1, 1, T, 576]
        if taps is not None:
            taps["kv_row"] = kv_row
        self._fill_rows(kv_row, page_table, kv_cache, keep=taps is not None)

    def _fill_table_entries(self, page_table, T: int) -> int:
        """``cfg.prefill_page_table_entries(T)`` after checking that ``page_table`` has that many entries."""
        n_pt = self.cfg.prefill_page_table_entries(T)
        if int(page_table.shape[-1]) < n_pt:
            raise ValueError(
                f"prefill page table has {int(page_table.shape[-1])} entries, {T} rows need {n_pt} (pad with -1 = "
                f"skip, or the null block 0, beyond the user's blocks)"
            )
        return n_pt

    def _fill_rows(self, kv_row, page_table, kv_cache, *, keep: bool):
        """The fill half of :meth:`_fill_latent`: ``typecast(kv_row)`` (cache dtype) -> ``paged_fill_cache`` through the
        first ``cfg.prefill_page_table_entries(T)`` entries of ``page_table``. ``kv_row`` ``[1, 1, T, 576]`` bf16 is
        freed unless ``keep``."""
        n_pt = self._fill_table_entries(page_table, int(kv_row.shape[-2]))
        src = kv_row
        if kv_row.dtype != kv_cache.dtype:
            src = ttnn.typecast(kv_row, kv_cache.dtype)
            if not keep:
                ttnn.deallocate(kv_row)
        pt = page_table
        if int(page_table.shape[-1]) != n_pt:
            pt = ttnn.slice(page_table, [0, 0], [1, n_pt])
        ttnn.experimental.paged_fill_cache(kv_cache, src, pt, batch_idx=0)
        if pt is not page_table:
            ttnn.deallocate(pt)
        if src is not kv_row or not keep:
            ttnn.deallocate(src)

    def _rope(self, x, cos, sin):
        if self.rope_mode == "hf":
            return ttnn.experimental.rotary_embedding_hf(
                x, cos, sin, is_decode_mode=False, compute_kernel_config=self.ckc_rope
            )
        if self.rope is None:
            raise ValueError("rope_mode='composite' needs a MotifRope")
        T = int(x.shape[-2])
        if int(cos.shape[-2]) != T:  # decode "rows" tables carry 32 rows for L = 8 lanes: broadcast needs equal rows
            cos = ttnn.slice(cos, [0, 0, 0, 0], [1, 1, T, int(cos.shape[-1])])
            sin = ttnn.slice(sin, [0, 0, 0, 0], [1, 1, T, int(sin.shape[-1])])
            out = self.rope.apply_composite(x, cos, sin)
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
            return out
        return self.rope.apply_composite(x, cos, sin)

    def _combine(self, u_sig, u_noise, g, lam, active=None):
        """``(u_sig - sigmoid(lam @ E) * u_noise) * g`` ``[1,1,T,1024]`` (signal order = ``wo`` input order); inactive
        rows -> 0 when ``active`` is given (always in decode; prefill has no inactive rows). Consumes ``u_sig``,
        ``u_noise``, ``g``, ``lam``."""
        v_exp = self._linear(lam, self.lam_expand, ckc=self.ckc_heads, activation="sigmoid")
        ttnn.deallocate(lam)
        d = ttnn.addcmul(u_sig, v_exp, u_noise, value=-1.0)
        for t in (u_sig, u_noise, v_exp):
            ttnn.deallocate(t)
        dg = ttnn.multiply(d, g)
        ttnn.deallocate(d)
        ttnn.deallocate(g)
        if active is not None:
            masked = ttnn.where(active, dg, 0.0)
            ttnn.deallocate(dg)
            dg = masked
        return dg

    def _rot_tables(self, rot: RotArg):
        if isinstance(rot, dict):
            return rot[self.kind]
        if isinstance(rot, (tuple, list)):
            return rot[0], rot[1]
        if rot is None:
            raise ValueError("forward_decode needs rot= (cos, sin) | {kind: (cos, sin)} | rot_idxs tensor")
        if self.rope is None:
            raise ValueError("raw rot indices need a MotifRope (pass rope= at construction)")
        return self.rope.decode_cos_sin(self.kind, rot, layout="rows")

    # ==================================================================================================================
    # decode
    # ==================================================================================================================
    def forward_decode(
        self,
        x,
        *,
        rot: RotArg,
        cur_pos,
        page_table,
        kv_cache,
        active,
        taps: Optional[Dict[str, Any]] = None,
        kv_write: Optional[DecodeKVWriter] = None,
    ):
        """One decode step for the L = 8 lanes of this chip's DP row, or the T64 verify step's L = 16 rows ``[8 anchors
        | 8 drafts]`` (trace-safe; design §2.3.4 decode steps 1-12; module docstring "T64 verify step").

        Args:
            x: normalized input ``[1, 1, L, 4096]`` bf16 TILE DRAM (replicated in the row). L = 16 needs a ``kv_write``
                built for 16 rows per DP row (``DecodeKVWrite(rows=64)``).
            rot: this step's RoPE tables: ``{kind: (cos, sin)}`` from :meth:`decode_rope_tables` (preferred, built once
                per step), a ``(cos, sin)`` pair ``[1,1,32,64]`` for this layer's kind, or the ``rot_idxs [1, 32]``
                uint32 tensor (gathered here). Row t of the tables rotates row t of ``x``.
            cur_pos: ``[L]`` int32 ROW_MAJOR (per row; ``-1`` = inactive lane: skipped by the cache update and
                FlashMLA).
            page_table: ``[L, W]`` int32 ROW_MAJOR (per row). ``W x block`` must be a multiple of the FlashMLA k chunk
                (128, G1): the kernel does not check it, and with e.g. W = 5 (320 keys) the lanes in the partial last
                chunk (positions 256 .. 319) read past the table and return garbage (seen on this Galaxy). The serving
                width (512) and any even W at block 64 are fine.
            kv_cache: this layer's paged latent cache ``[N, 1, block, 576]`` (``cfg.dtypes.kv_cache``, TILE, DRAM);
                updated in place at ``cur_pos``.
            active: **required** ``[1, 1, L, 1024]`` (or ``[1, 1, L, 1]``) bf16 0/1 mask
                (:meth:`active_mask_from_cur_pos` or :meth:`active_mask_host`, built once per step for all layers);
                inactive rows of the output are exactly 0. FlashMLA leaves the output rows of skipped lanes unwritten,
                so there is no unmasked variant.
            taps: optional dict that receives intermediate tensors (debug / tests; never inside a trace).
            kv_write: the step's :class:`DecodeKVWriter` (``tt/kv_write.py``: ``row`` / ``row_split`` / ``all`` /
                ``all_split``, design §3.5), called once in place of the cache update. ``None`` (default) = draft 1:
                one 8-lane ``paged_update_cache`` of this row's lanes through ``cur_pos`` / ``page_table`` (8 rows
                only). On a global layer FlashMLA runs once per entry of ``kv_write.flash_groups()`` (option A''): one
                call at 8 rows per DP row, two B = 8 calls at 16.

        Returns ``[1, 1, L, 4096]`` bf16 TILE DRAM, identical on the TP chips of the row.
        """
        if active is None:
            raise ValueError(
                "forward_decode needs the per-step active mask (MotifAttention.active_mask_from_cur_pos(cur_pos) or "
                "active_mask_host): FlashMLA leaves the rows of skipped lanes (cur_pos = -1) unwritten"
            )
        if kv_write is not None and not callable(getattr(kv_write, "write", None)):
            raise TypeError(
                f"kv_write must have a write(kv_row, kv_cache, *, cur_pos, page_table) method, got {kv_write!r}"
            )
        # host checks before any device op: the row count and, on global layers, the FlashMLA groups (option A'')
        groups = self._flash_groups(kv_write, self._decode_rows(x, kv_write, active), cur_pos, page_table)
        cos, sin = self._rot_tables(rot)
        fused = self._fused_input_chain(x, cos, sin, kv_write) if taps is None else None
        if fused is not None:  # Phase F F1: q_mla, g, lam and the kv update input from one program
            q_mla, g, lam, kv = fused
            if kv_write is None:
                ttnn.experimental.paged_update_cache(kv_cache, kv, update_idxs_tensor=cur_pos, page_table=page_table)
            else:
                kv_write.write(kv, kv_cache, cur_pos=cur_pos, page_table=page_table)
            ttnn.deallocate(kv)
        else:
            q_mla, g, lam = self._input_chain_ops(x, cos, sin, kv_write, kv_cache, cur_pos, page_table, taps)

        # ---- FlashMLA decode (G1 config, scale folded into q, ATTN-2); option A'' on global layers at 16 rows ------
        if len(groups) == 1:  # draft 1, T32, every SWA layer (B = L)
            _, cur, pt = groups[0]
            o_lat = self._flash_mla(q_mla, kv_cache, cur, pt)  # [1, L, 10, 512]
        else:  # one B = 8 call per group (anchors at n, drafts at n + 1), concatenated: [1, L, 10, 512]
            o_lat = self._flash_mla_groups(q_mla, kv_cache, groups)
        if taps is not None:
            taps["q_mla"] = q_mla
            taps["o_lat"] = o_lat
        else:
            ttnn.deallocate(q_mla)

        # ---- un-absorb per head, differential, gate, mask, wo, AR(tp) ------------------------------------------
        o_heads = ttnn.transpose(o_lat, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 10, L, 512]
        if taps is None:
            ttnn.deallocate(o_lat)
        return self._absorbed_epilogue(o_heads, g, lam, active=active, taps=taps, decode=True)

    def fused_input(self):
        """The layer's :class:`~models.demos.motif3.tt.kernels.attn_in.FusedAttnIn` (built on first use)."""
        if getattr(self, "_fused_input", None) is None:
            from .kernels.attn_in import FusedAttnIn

            self._fused_input = FusedAttnIn(self.mesh_device, eps=self.cfg.rms_norm_eps)
        return self._fused_input

    def _fused_input_chain(self, x, cos, sin, kv_write):
        """``attn_in="post"`` (Phase F F1): the two latent projections (the release's ops), then ONE program for the
        rest of the input chain; ``attn_in="fused"``: the projections in that program too -> ``(q_mla [1, L, 10, 576], g, lam, kv)`` with ``kv`` the draft-1 update input
        (``kv_write=None``: :attr:`update_mc`) or ``kv_row [1, 1, L, 576]`` for the writer. ``None`` (nothing run) when
        the mode is off or the operands are outside the kernel's contract (the op chain then runs)."""
        mode = getattr(self, "attn_in", "ops")  # (host tests build the module without __init__)
        if mode not in ("post", "fused") or getattr(self, "rope_mode", "hf") != "hf":
            return None
        L = int(x.shape[-2])
        grid = self.mesh_device.compute_with_storage_grid_size()
        if (int(grid.x), int(grid.y)) != (12, 10) or not 1 <= L <= TILE or (kv_write is None and L != 8):
            return None
        if int(cos.padded_shape[-2]) != TILE or int(cos.shape[-1]) != self.rope_dim:
            return None
        umc = self.update_mc if kv_write is None else None
        if mode == "fused":  # the latent projections in the same program
            if x.dtype != ttnn.bfloat16 or x.layout != ttnn.TILE_LAYOUT \
                    or x.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
                return None
            return self.fused_input().full(x, self.w_q_lat, self.w_kv_lat, cos, sin, self.w_q_b, self.w_gate,
                                           self.w_uk, update_mc=umc)
        cq = self._linear(x, self.w_q_lat, ckc=self.ckc_latent, pc=self._pc("q_lat", True), dtype=ttnn.float32)
        kvl = self._kv_latent(x, True)
        out = self.fused_input()(cq, kvl, cos, sin, self.w_q_b, self.w_gate, self.w_uk, update_mc=umc)
        ttnn.deallocate(cq)
        ttnn.deallocate(kvl)
        return out

    def _input_chain_ops(self, x, cos, sin, kv_write, kv_cache, cur_pos, page_table, taps):
        """The release's decode input chain (``attn_in="ops"``): projections, norms, heads, RoPE, KV write ->
        ``(q_mla [1, L, 10, 576], g, lam)``."""
        q, g, n, kpe, lam = self._project(x, decode=True)

        # ---- Q [1, L, 10, 576] = [q_nope @ W_UK' | rope(q_pe)] (lanes on dim 1, heads on rows) -------------------
        q_nope, q_pe = ttnn.experimental.nlp_create_q_heads_split(q, num_heads=self.H, split_head_dim=self.nope)
        ttnn.deallocate(q)
        q_lat = ttnn.matmul(
            q_nope,
            self.w_uk,
            program_config=self._pc("w_uk", True),
            compute_kernel_config=self.ckc_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_nope)
        q_pe_r = self._rope(q_pe, cos, sin)
        ttnn.deallocate(q_pe)
        q_heads = ttnn.concat([q_lat, q_pe_r], dim=-1)  # [1, 10, L, 576]
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_pe_r)
        q_mla = ttnn.transpose(q_heads, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, L, 10, 576]
        ttnn.deallocate(q_heads)

        # ---- KV write: [n | rope(kpe)] -> [1, L, 1, 576] height-sharded one lane per core (G7, ATTN-3) -------------
        k_pe = self._rope(kpe, cos, sin)
        ttnn.deallocate(kpe)
        kv_row = ttnn.concat([n, k_pe], dim=-1)  # [1, 1, L, 576]
        ttnn.deallocate(n)
        ttnn.deallocate(k_pe)
        if kv_write is None:  # draft 1 ("row"): this DP row's 8 lanes, one update
            kv_upd = ttnn.transpose(kv_row, 1, 2, memory_config=self.update_mc)  # [1, L, 1, 576]
            if taps is not None:
                taps["kv_row"] = kv_row
            else:
                ttnn.deallocate(kv_row)
            ttnn.experimental.paged_update_cache(kv_cache, kv_upd, update_idxs_tensor=cur_pos, page_table=page_table)
            ttnn.deallocate(kv_upd)
        else:  # tt/kv_write.py: split / replicated (KV-R) writes, all before FlashMLA
            kv_write.write(kv_row, kv_cache, cur_pos=cur_pos, page_table=page_table)
            if taps is not None:
                taps["kv_row"] = kv_row
            else:
                ttnn.deallocate(kv_row)
        return q_mla, g, lam

    def _decode_rows(self, x, kv_write, active) -> int:
        """Rows ``L`` per DP row of the decode input ``x [1, 1, L, 4096]`` (host checks, no device op). ``L`` is at
        most one tile row (the decode program configs, the ``[1, 32]`` RoPE index row), and the ``active`` mask has
        ``L`` rows too. The draft-1 update (``kv_write=None``) transposes into :attr:`update_mc` (one lane per core on
        ``self.lanes`` = 8 cores) and writes every row through ``cur_pos``: it serves exactly 8 rows. The T64 step's 16
        rows ``[8 anchors | 8 drafts]`` need a writer that writes the anchors and the drafts in two calls
        (``DecodeKVWrite(rows=64)``). A writer that names its rows (``lanes_per_row``) must match ``x``."""
        L = int(x.shape[-2])
        if not 1 <= L <= TILE:
            raise ValueError(f"forward_decode: x has {L} rows per DP row; decode takes 1..{TILE} (one tile row)")
        a_rows = int(active.shape[-2])
        if a_rows != L:
            raise ValueError(
                f"forward_decode: the active mask has {a_rows} rows, x has {L} rows per DP row (build it with "
                f"active_mask_from_cur_pos(cur_pos, {L}))"
            )
        if kv_write is None:
            if L != self.lanes:
                raise ValueError(
                    f"forward_decode: x has {L} rows per DP row, but the draft-1 update (kv_write=None) writes "
                    f"exactly {self.lanes} lanes (one {self.lanes}-core paged_update_cache through cur_pos). The T64 "
                    f"verify step (16 rows = [8 anchors | 8 drafts]) needs kv_write=DecodeKVWrite(rows=64): call A "
                    f"writes the anchors at n, call B the drafts at n + 1"
                )
            return L
        per = getattr(kv_write, "lanes_per_row", None)
        if per is not None and int(per) != L:
            raise ValueError(
                f"forward_decode: x has {L} rows per DP row, kv_write writes {int(per)} (kv_write.lanes_per_row): "
                f"build the writer for the step's rows (DecodeKVWrite(rows=64) for 16 rows per DP row)"
            )
        return L

    def _flash_groups(self, kv_write, L: int, cur_pos, page_table) -> List[Tuple[slice, Any, Any]]:
        """FlashMLA's calls of this layer, ``[(row slice, cur_pos, page_table)]`` (host only, trace-safe; module
        docstring "T64 verify step", design §4.3 option A'').

        * SWA layers (and the MTP layer), draft 1 (``kv_write=None``), or a writer without ``flash_groups`` at 8 rows:
          the single call ``[(0:L, cur_pos, page_table)]``. At B = 16 the SWA rows are bitwise the B = 8 rows.
        * global layers: ``kv_write.flash_groups()``, whose row slices must partition ``[0, L)`` in order. One group =
          one call on the group's tensors (the T32 step: ``(0:8, kv_write.cur_pos, kv_write.page_table)``, today's
          operands). Two groups at 16 rows = two B = 8 calls with the T32 call's core split, so every row equals its
          T32 row bit for bit. A writer without ``flash_groups`` is refused at more than 8 rows (one B = 16 call is
          not bitwise; ask for it explicitly with the one group ``(0:16, cur_pos, page_table)``)."""
        single = [(slice(0, L), cur_pos, page_table)]
        if self.window is not None or kv_write is None:
            return single
        fg = getattr(kv_write, "flash_groups", None)
        if fg is None:
            if L != self.lanes:
                raise ValueError(
                    f"forward_decode: a global layer at {L} rows per DP row needs kv_write.flash_groups() (option A'': "
                    f"one B = {self.lanes} FlashMLA call per group, bitwise the T32 rows); {type(kv_write).__name__} "
                    f"has none. For option A (one B = {L} call, not bitwise) return [(slice(0, {L}), cur_pos, "
                    f"page_table)] from it"
                )
            return single
        groups = [tuple(gr) for gr in fg()]
        lo = 0
        for gr in groups:
            rows = gr[0] if len(gr) == 3 else None
            if not (
                isinstance(rows, slice)
                and rows.step in (None, 1)
                and rows.start == lo
                and isinstance(rows.stop, int)
                and lo < rows.stop <= L
            ):
                raise ValueError(
                    f"kv_write.flash_groups() must give (row slice, cur_pos, page_table) entries whose slices "
                    f"partition [0, {L}) in order; got {[gr[0] if gr else None for gr in groups]}"
                )
            lo = rows.stop
        if lo != L:
            raise ValueError(f"kv_write.flash_groups() covers rows [0, {lo}) of the step's {L} rows per DP row")
        return groups

    def _flash_mla(self, q_mla, kv_cache, cur_pos, page_table):
        """``paged_flash_multi_latent_attention_decode`` of ``q_mla [1, B, 10, 576]`` (G1 config, scale folded into q,
        ATTN-2) -> ``[1, B, 10, 512]``. Does not consume ``q_mla``."""
        return ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_mla,
            kv_cache,
            None,
            head_dim_v=self.rank,
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos,
            scale=1.0,
            sliding_window_size=self.window,
            program_config=self.decode_pc,
            compute_kernel_config=self.ckc_sdpa_decode,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _flash_mla_groups(self, q_mla, kv_cache, groups):
        """Option A'' on a global layer: per group ``(rows, cur_pos, page_table)`` one FlashMLA call on ``q_mla[:,
        rows]`` (a dim-1 slice of ``[1, L, 10, 576]``: the group's users), then ONE ``concat`` of the outputs on dim 1
        -> ``[1, L, 10, 512]``. Does not consume ``q_mla``; frees the slices and the per-group outputs."""
        outs = []
        for rows, cur, pt in groups:
            qg = ttnn.slice(
                q_mla,
                [0, rows.start, 0, 0],
                [1, rows.stop, self.H, self.latent_dim],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, n, 10, 576]
            outs.append(self._flash_mla(qg, kv_cache, cur, pt))
            ttnn.deallocate(qg)
        o_lat = ttnn.concat(outs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for o in outs:
            ttnn.deallocate(o)
        return o_lat

    def _absorbed_epilogue(self, o_heads, g, lam, *, active, taps, decode: bool):
        """Absorbed-form epilogue (decode, sp1 global): ``o_heads [1, 10, T, 512]`` (latent outputs per virtual head,
        heads on dim 1) ``@ W_UV'`` -> ``nlp_concat_heads`` ``[1, 1, T, 1280]`` = ``[U_sig 1024 | U_noise 2 x 128]``
        -> noise expansion -> differential, gate (``active`` mask) -> ``wo`` -> ``all_reduce(tp)``. Consumes
        ``o_heads``, ``g``, ``lam``."""
        u = ttnn.matmul(
            o_heads,
            self.w_uv,
            program_config=self._pc("w_uv", decode),
            compute_kernel_config=self.ckc_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 10, T, 128]
        ttnn.deallocate(o_heads)
        if decode and getattr(self, "epilogue", "ops") == "fused" and active is not None:
            dg = self._fused_epilogue(u, g, lam, active, taps)
            if dg is not None:
                part = self._linear(dg, self.w_o, ckc=self.ckc_heads, pc=self._pc("wo", decode))
                ttnn.deallocate(dg)
                out = self.ccl.ar_tp(part)
                ttnn.deallocate(part)
                return out
        u_flat = ttnn.experimental.nlp_concat_heads(u, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, T, 1280]
        ttnn.deallocate(u)
        u_sig, noise = ttnn.experimental.nlp_create_q_heads_split(
            u_flat, num_heads=1, split_head_dim=self.Sg * self.vdim
        )  # [1,1,T,1024], [1,1,T,256]
        if taps is not None:
            taps["u_flat"] = u_flat
        else:
            ttnn.deallocate(u_flat)
        u_noise = self._linear(noise, self.noise_expand, ckc=self.ckc_heads)  # [1, 1, T, 1024]
        ttnn.deallocate(noise)
        dg = self._combine(u_sig, u_noise, g, lam, active)
        part = self._linear(dg, self.w_o, ckc=self.ckc_heads, pc=self._pc("wo", decode))
        ttnn.deallocate(dg)
        out = self.ccl.ar_tp(part)
        ttnn.deallocate(part)
        return out

    def fused_combine(self):
        """The layer's :class:`~models.demos.motif3.tt.kernels.attn_combine.FusedAttnCombine` (built on first use)."""
        if getattr(self, "_fused_combine", None) is None:
            from .kernels.attn_combine import FusedAttnCombine

            self._fused_combine = FusedAttnCombine(
                self.mesh_device, signal_heads=self.Sg, noise_heads=self.G, vdim=self.vdim
            )
        return self._fused_combine

    def _fused_epilogue(self, u, g, lam, active, taps):
        """``epilogue="fused"`` (Phase C D1): ``v = sigmoid(lam @ E)`` (the release's op), then ONE program for the
        concat / split / noise expansion / addcmul / multiply / where of :meth:`_combine` -> the ``wo`` input
        ``[1, 1, T, 1024]``. Returns ``None`` (nothing consumed) when the operands are outside the kernel's contract
        (the op chain then runs); otherwise consumes ``u``, ``g``, ``lam``."""
        fc = self.fused_combine()
        T = int(u.shape[-2])
        if int(active.shape[-1]) != fc.width or int(active.shape[-2]) != T or int(g.shape[-2]) != T:
            return None
        v_exp = self._linear(lam, self.lam_expand, ckc=self.ckc_heads, activation="sigmoid")
        if not fc.supports(u, v_exp, g, active):
            ttnn.deallocate(v_exp)
            return None
        ttnn.deallocate(lam)
        if taps is not None:
            taps["u_flat"] = ttnn.experimental.nlp_concat_heads(u, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dg = fc(u, v_exp, g, active)
        for t in (u, v_exp, g):
            ttnn.deallocate(t)
        return dg

    # ==================================================================================================================
    # prefill
    # ==================================================================================================================
    def prefill_sdpa_window_and_config(self, S: int):
        """``(sliding_window_size, compute_kernel_config)`` of the prefill SDPA for bucket ``S``: the window is dropped
        when it cannot mask anything (``S < window``, i.e. the 128 bucket on SWA layers: keys ``[p - 128, p]`` cover
        ``[0, p]``). The config is the ``sdpa_prefill`` role by default; with ``sdpa_prefill_fp32_acc="auto"`` fp32
        accumulation is used exactly when no window is in effect (``True``: always)."""
        window = self.window if (self.window is not None and S >= self.window) else None
        mode = self.sdpa_prefill_fp32_acc
        fp32 = mode is True or (mode == "auto" and window is None)
        return window, (self.ckc_sdpa_prefill_fp32 if fp32 else self.ckc_sdpa_prefill)

    def forward_prefill(
        self,
        x,
        *,
        page_table=None,
        kv_cache=None,
        rot: RotArg = None,
        taps: Optional[Dict[str, Any]] = None,
        chunk: Optional[PrefillChunkInputs] = None,
    ):
        """Prefill of one user (eager; design §2.3.4 prefill, §3.3), or one chunk of a resumed / chunked prefill
        (``chunk=``; features design §3.2). ``x [1, 1, S, 4096]`` bf16 (S = bucket rows, replicated on all chips) ->
        ``[1, 1, S, 4096]`` bf16 (identical on all chips).

        Args:
            page_table: ``[1, n]`` int32 ROW_MAJOR, the user's blocks; the fill uses exactly the first
                ``cfg.prefill_page_table_entries(S)`` entries (a wider table is sliced here; pass the exact width to
                keep one program shape per bucket, M10). ``-1`` entries are skipped by ``paged_fill_cache``: pass the
                sp0 fill table of ``prefill_plan.fill_table`` (shared blocks below ``w0`` and pure padding blocks
                ``-1``; features design §3.2.1). ``None`` (with ``kv_cache=None``) skips the cache fill.
            kv_cache: this layer's paged cache; gets ``[n | rope(k_pe)]`` for positions ``0..S-1`` (ATTN-6).
            rot: optional ``(cos, sin)`` ``[1, 1, >=S, 64]``; default ``rope.prefill_cos_sin(kind, S)``.
            taps: optional dict receiving intermediate tensors (``u_flat``; with an sp1 ``chunk=`` also ``kv_row``, the
                bf16 latent of the chunk's rows).
            chunk: :class:`PrefillChunkInputs` of a chunk at positions ``[chunk.start, chunk.start + S)``, in place of
                ``page_table`` / ``rot``. sp0 (start 0) is the draft-1 path with ``chunk.fill_pt``: bitwise
                ``forward_prefill(x, page_table=chunk.fill_pt, kv_cache=kv_cache)``. sp1 (start > 0) reads the
                cached prefix and needs ``kv_cache``: global layers fill the chunk's latent, then run the absorbed
                chunked SDPA over the paged cache from the start; SWA layers run the square ``[tail ‖ chunk]``
                window-129 SDPA over the 128 cached rows before the start (module docstring).
        """
        if chunk is not None:
            return self._forward_prefill_chunk(x, chunk, kv_cache=kv_cache, page_table=page_table, rot=rot, taps=taps)
        S = int(x.shape[-2])
        if rot is None:
            if self.rope is None:
                raise ValueError("forward_prefill needs a MotifRope or rot=(cos, sin)")
            cos, sin = self.rope.prefill_cos_sin(self.kind, S)
        else:
            cos, sin = self._rot_tables(rot)
        q, g, n, kpe, lam = self._project(x, decode=False)
        q_full = self._q_expanded(q, cos, sin)  # [1, 10, S, 192], HF head order
        k_pe = self._rope(kpe, cos, sin)  # [1, 1, S, 64]
        ttnn.deallocate(kpe)
        k_full, v_pad = self._expanded_kv(n, k_pe)  # [1, 2, S, 192] each

        # ---- SDPA (G2 program config, scale folded into q, ATTN-5) -------------------------------------------------
        window, ckc = self.prefill_sdpa_window_and_config(S)
        o = ttnn.transformer.scaled_dot_product_attention(
            q_full,
            k_full,
            v_pad,
            is_causal=True,
            scale=1.0,
            sliding_window_size=window,
            program_config=self.cfg.sdpa_prefill_pc(self.spec, seq_len=S),
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 10, S, 192] (columns 128..191 = 0)
        ttnn.deallocate(q_full)
        ttnn.deallocate(k_full)
        ttnn.deallocate(v_pad)
        o_v = ttnn.slice(o, [0, 0, 0, 0], [1, self.H, S, self.vdim])
        ttnn.deallocate(o)
        out = self._expanded_epilogue(o_v, g, lam, taps)

        # ---- cache fill (ATTN-6): typecast to the cache dtype (raw tile copy), bucket's first S/bs entries -----
        if kv_cache is not None:
            if page_table is None:
                raise ValueError("forward_prefill: kv_cache given without page_table")
            self._fill_latent(n, k_pe, page_table, kv_cache)
        ttnn.deallocate(n)
        ttnn.deallocate(k_pe)
        return out

    def forward_prefill_sp(self, x, sp, S: int, *, page_table=None, kv_cache=None, rot: RotArg = None):
        """Phase C P4 (``tt/prefill_sp.py``, ``MOTIF3_PREFILL_SP=dp``): the sp0 prefill of a pass of ``S`` rows split
        over the DP rows. ``x [1, 1, R, 4096]`` = this DP row's rows ``[d R, d R + R)`` (``R = S / 4``) ->
        ``[1, 1, R, 4096]``, bitwise the rows of :meth:`forward_prefill`'s output. ``sp``: the model's
        :class:`~models.demos.motif3.tt.prefill_sp.PrefillSP`. The latent of all S rows is all-gathered over DP and
        filled through ``page_table`` exactly as :meth:`forward_prefill` fills it. ``rot`` as there (sp0: ``None``)."""
        R = int(x.shape[-2])
        if R * int(self.cfg.dp) != int(S):
            raise ValueError(f"forward_prefill_sp: {R} rows per DP row for a pass of {S}")
        cs = sp.cos_sin(self.kind, S, None if rot is None else self._rot_tables(rot))
        cos, sin = cs
        q, g, n, kpe, lam = self._project(x, decode=False)
        q_full = self._q_expanded(q, cos, sin)  # [1, 10, R, 192], HF head order
        k_pe = self._rope(kpe, cos, sin)  # [1, 1, R, 64]
        ttnn.deallocate(kpe)
        if not sp.is_cached_rot(cs):
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        lat = ttnn.concat([n, k_pe], dim=-1)  # [1, 1, R, 576]: this row's rows of the cache-fill latent
        ttnn.deallocate(n)
        ttnn.deallocate(k_pe)
        lat_all = self.ccl.ag_dp(lat, 2, race_free=True)  # [1, 1, S, 576] on every chip
        masks = sp.masks(S)
        rank, w = self.rank, self.rank + self.rope_dim
        if self.window is None:
            src, mask = lat_all, masks["global"]
        else:  # [tail_1 | tail_2 | tail_3 | own R rows] (tt/prefill_sp.py)
            T = SP_TAIL
            parts = [ttnn.slice(lat_all, [0, 0, k * R - T, 0], [1, 1, k * R, w]) for k in range(1, int(self.cfg.dp))]
            src = ttnn.concat(parts + [lat], dim=2)
            for t in parts:
                ttnn.deallocate(t)
            mask = masks["swa"]
        ttnn.deallocate(lat)
        Kn = int(src.shape[-2])
        n_k = ttnn.slice(src, [0, 0, 0, 0], [1, 1, Kn, rank])
        kpe_k = ttnn.slice(src, [0, 0, 0, rank], [1, 1, Kn, w])
        if src is not lat_all:
            ttnn.deallocate(src)
        k_full, v_pad = self._expanded_kv(n_k, kpe_k)  # [1, 2, Kn, 192] each
        ttnn.deallocate(n_k)
        ttnn.deallocate(kpe_k)
        _, ckc = self.prefill_sdpa_window_and_config(S)  # the release call's compute config (bitwise)
        o = ttnn.transformer.scaled_dot_product_attention(
            q_full,
            k_full,
            v_pad,
            is_causal=False,
            attn_mask=mask,
            scale=1.0,
            program_config=self.cfg.sdpa_prefill_pc(self.spec, seq_len=S),
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 10, R, 192]
        ttnn.deallocate(q_full)
        ttnn.deallocate(k_full)
        ttnn.deallocate(v_pad)
        o_v = ttnn.slice(o, [0, 0, 0, 0], [1, self.H, R, self.vdim])
        ttnn.deallocate(o)
        out = self._expanded_epilogue(o_v, g, lam, None)
        if kv_cache is not None:
            if page_table is None:
                raise ValueError("forward_prefill_sp: kv_cache given without page_table")
            self._fill_rows(lat_all, page_table, kv_cache, keep=False)
        else:
            ttnn.deallocate(lat_all)
        return out

    # ---- shared prefill building blocks (the draft-1 op sequence, split into helpers) --------------------------------
    def _q_expanded(self, q, cos, sin):
        """``q [1, 1, T, 1920]`` (virtual order) -> ``Q [1, 10, T, 192]`` in HF head order (GQA: 5 consecutive q heads
        per KV group) with ``q_pe`` roped. Consumes ``q``."""
        T = int(q.shape[-2])
        q_nope, q_pe = ttnn.experimental.nlp_create_q_heads_split(q, num_heads=self.H, split_head_dim=self.nope)
        ttnn.deallocate(q)
        q_pe_r = self._rope(q_pe, cos, sin)
        ttnn.deallocate(q_pe)
        q_virt = ttnn.concat([q_nope, q_pe_r], dim=-1)  # [1, 10, T, 192], virtual order
        ttnn.deallocate(q_nope)
        ttnn.deallocate(q_pe_r)
        hd = self.cfg.head_dim
        parts = [ttnn.slice(q_virt, [0, a, 0, 0], [1, b, T, hd]) for a, b in hf_order_from_virtual(self.cfg)]
        ttnn.deallocate(q_virt)
        q_full = ttnn.concat(parts, dim=1)
        for t in parts:
            ttnn.deallocate(t)
        return q_full

    def _expanded_kv(self, n, k_pe):
        """K / V_pad from the latent (expanded form, V zero-padded 128 -> 192, G2): ``n [1, 1, T, 512] @ E_pref`` ->
        per group ``[k_nope | v | 0_64]``; ``K = [k_nope | k_pe x 2 groups]``. Returns ``(K, V_pad)``, both
        ``[1, 2, T, 192]``; ``n`` and ``k_pe`` (roped) are not consumed."""
        kvx = self._linear(n, self.w_kv_expand, ckc=self.ckc_heads)  # [1, 1, T, 2 x 320]
        k_nope, v_pad = ttnn.experimental.nlp_create_q_heads_split(kvx, num_heads=self.G, split_head_dim=self.nope)
        ttnn.deallocate(kvx)
        k_pe_g = ttnn.repeat(k_pe, ttnn.Shape([1, self.G, 1, 1]))
        k_full = ttnn.concat([k_nope, k_pe_g], dim=-1)  # [1, 2, T, 192]
        ttnn.deallocate(k_nope)
        ttnn.deallocate(k_pe_g)
        return k_full, v_pad

    def _expanded_epilogue(self, o_v, g, lam, taps):
        """Expanded-form epilogue (sp0, sp1 SWA): ``o_v [1, 10, T, 128]`` (HF head order) -> ``nlp_concat_heads`` ->
        signal columns + noise x4 -> differential, gate -> ``wo`` -> ``all_reduce(tp)``. Consumes ``o_v``, ``g``,
        ``lam``."""
        T = int(o_v.shape[-2])
        u_flat = ttnn.experimental.nlp_concat_heads(o_v, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1,1,T,1280] HF order
        ttnn.deallocate(o_v)
        # HF order: [s(g) x4 | n(g)] per group -> signal columns (2 slices + concat) and noise x4 (concat of 8)
        v, r, hpg = self.vdim, self.r, self.cfg.heads_per_group
        sig = [ttnn.slice(u_flat, [0, 0, 0, hpg * gi * v], [1, 1, T, (hpg * gi + r) * v]) for gi in range(self.G)]
        noise = [
            ttnn.slice(u_flat, [0, 0, 0, (hpg * gi + r) * v], [1, 1, T, (hpg * gi + r + 1) * v]) for gi in range(self.G)
        ]
        if taps is not None:
            taps["u_flat"] = u_flat
        else:
            ttnn.deallocate(u_flat)
        u_sig = ttnn.concat(sig, dim=-1)
        u_noise = ttnn.concat([nz for nz in noise for _ in range(r)], dim=-1)
        for t in sig + noise:
            ttnn.deallocate(t)
        dg = self._combine(u_sig, u_noise, g, lam)
        part = self._linear(dg, self.w_o, ckc=self.ckc_heads)
        ttnn.deallocate(dg)
        out = self.ccl.ar_tp(part)
        ttnn.deallocate(part)
        return out

    # ---- resumed (sp1) chunks: features design §3.2.2 (global) / §3.2.3 (SWA) ----------------------------------------
    def _forward_prefill_chunk(self, x, chunk: PrefillChunkInputs, *, kv_cache, page_table, rot, taps):
        if not isinstance(chunk, PrefillChunkInputs):
            raise TypeError(f"chunk must be a PrefillChunkInputs, got {type(chunk).__name__}")
        if page_table is not None or rot is not None:
            raise ValueError(
                "forward_prefill: pass chunk= or page_table= / rot=, not both (the chunk carries its tables)"
            )
        self._check_chunk_rows(x, chunk, "forward_prefill")
        if chunk.is_packed:  # P5: B segments of S rows in one pass (module docstring)
            B, S = self._check_packed(chunk, "forward_prefill")
            if chunk.path == PK0:
                return self._prefill_packed_sp0(x, chunk, kv_cache, taps, B, S)
            if kv_cache is None:
                raise ValueError("a pk1 pass reads its segments' cached prefixes from the paged cache: pass kv_cache=")
            if self.window is None:
                return self._prefill_packed_sp1_global(x, chunk, kv_cache, taps, B, S)
            return self._prefill_packed_sp1_swa(x, chunk, kv_cache, taps, B, S)
        if not chunk.is_sp1:  # sp0: draft 1 with the chunk's fill table
            return self.forward_prefill(x, page_table=chunk.fill_pt, kv_cache=kv_cache, rot=chunk.rot, taps=taps)
        if kv_cache is None:
            raise ValueError("an sp1 chunk reads the cached prefix from the paged cache: pass kv_cache=")
        if self.window is None:
            return self._prefill_sp1_global(x, chunk, kv_cache, taps)
        return self._prefill_sp1_swa(x, chunk, kv_cache, taps)

    def _check_chunk_rows(self, x, chunk: PrefillChunkInputs, where: str) -> int:
        """``x``'s rows against the chunk's bucket, and the RoPE rows the chunk must carry (review edit R-E11): an sp1
        chunk and EVERY packed pass (pk0 included) need ``chunk.rot``. Without it ``forward_prefill`` / ``fill_kv``
        (the MTP fill) would rope with ``rope.prefill_cos_sin(kind, C)``, positions ``0 .. C-1``: wrong for an sp1
        chunk and for every segment but the first of a packed pass."""
        if len(x.shape) != 4 or int(x.shape[-1]) != self.cfg.hidden_size:
            raise ValueError(f"{where} expects x [1, 1, C, {self.cfg.hidden_size}], got {tuple(x.shape)}")
        C = int(x.shape[-2])
        if C != int(chunk.bucket):
            raise ValueError(f"{where}: x has {C} rows, the chunk's bucket is {chunk.bucket}")
        if (chunk.is_sp1 or chunk.is_packed) and (chunk.rot is None or self.kind not in chunk.rot):
            what = "an sp1 chunk" if chunk.is_sp1 else f"a {chunk.path} pass (every segment at its own positions)"
            raise ValueError(
                f"{where}: {what} needs the {self.kind!r} RoPE rows of its positions (chunk.rot; None after "
                "PrefillChunkInputs.write(regather=False): pass dataclasses.replace(chunk, rot=tables))"
            )
        return C

    def _check_packed(self, chunk: PrefillChunkInputs, where: str) -> Tuple[int, int]:
        """``(B, S)`` of a packed pass after its structural checks (before any device op): ``bucket = B x S`` with
        ``S`` a multiple of the block size and of a tile (the views ``[1, H, T, d] <-> [H, B, S, d]`` are then
        metadata-only and every segment starts on a block); pk1 also carries one SDPA page-table row per segment and
        the start index."""
        B, S, T, bs = int(chunk.segments), chunk.seg_rows, int(chunk.bucket), int(self.cfg.kv_block_size)
        if S is None or B < 1 or int(S) * B != T or int(S) % bs or int(S) % TILE:
            raise ValueError(
                f"{where}: a packed pass is segments x seg_rows = bucket rows, seg_rows a multiple of the block size "
                f"{bs}; got {B} x {S} for bucket {T}"
            )
        if chunk.path == PK1:
            if chunk.sdpa_pt is None or chunk.start_idx is None:
                raise ValueError(f"{where}: a pk1 pass needs its SDPA page tables and start index (sdpa_pt, start_idx)")
            if len(chunk.sdpa_pt.shape) != 2 or int(chunk.sdpa_pt.shape[0]) != B:
                raise ValueError(
                    f"{where}: pk1 SDPA page tables {tuple(chunk.sdpa_pt.shape)}: the batched chunked SDPA reads one "
                    f"row per segment ({B})"
                )
        return B, int(S)

    def sp1_global_program_config(self, bucket: int, kv_dtype=None):
        """Program config of the sp1 global chunked SDPA at ``bucket`` rows (``cfg.resumed_prefill_pc``, gate G9).
        ``kv_dtype`` = the dtype of the cache it reads (default ``cfg.dtypes.kv_cache``): a bf16 cache takes the 64/64
        table (``model_config.SP1_GLOBAL_CHUNKS_BF16_KV``: at 128/128 its static CBs overrun the CCL semaphores)."""
        return self.cfg.resumed_prefill_pc(self.spec, int(bucket), kv_dtype=kv_dtype)

    def _check_sp1_global(self, chunk: PrefillChunkInputs, pc, C: int) -> None:
        """Design R6 / G9: the chunked kernels divide the start by ``q_chunk`` with no device check, so a start that is
        not a multiple of both chunk sizes silently answers from the floored start."""
        a, q, k = int(chunk.start), int(pc.q_chunk_size), int(pc.k_chunk_size)
        if a <= 0 or a % q or a % k:
            raise ValueError(
                f"sp1 global chunk start {a} must be a positive multiple of the chunked SDPA's q / k chunks ({q}, {k}) "
                "(the kernels floor it silently; prefill_plan aligns starts to cfg.prefill_resume_alignment)"
            )
        if chunk.sdpa_pt is None or chunk.start_idx is None:
            raise ValueError("an sp1 global chunk needs its SDPA page table and start index (chunk.sdpa_pt, start_idx)")
        if int(chunk.sdpa_pt.shape[-1]) * int(self.cfg.kv_block_size) < a + C:
            raise ValueError(f"SDPA page table {tuple(chunk.sdpa_pt.shape)} does not cover positions [0, {a + C})")

    def _q_absorbed(self, q, cos, sin):
        """``q [1, 1, T, 1920]`` (virtual order) -> ``Q_abs [1, 10, T, 576] = [q_nope @ W_UK' | rope(q_pe)]`` (heads on
        dim 1, virtual order, as in decode): the query of the absorbed chunked SDPA (sp1 / pk1 global layers). Consumes
        ``q``."""
        q_nope, q_pe = ttnn.experimental.nlp_create_q_heads_split(q, num_heads=self.H, split_head_dim=self.nope)
        ttnn.deallocate(q)
        q_lat = ttnn.matmul(
            q_nope, self.w_uk, compute_kernel_config=self.ckc_heads, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, 10, T, 512]
        ttnn.deallocate(q_nope)
        q_pe_r = self._rope(q_pe, cos, sin)
        ttnn.deallocate(q_pe)
        q_abs = ttnn.concat([q_lat, q_pe_r], dim=-1)  # [1, 10, T, 576]
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_pe_r)
        return q_abs

    def _prefill_sp1_global(self, x, chunk: PrefillChunkInputs, kv_cache, taps):
        """sp1 global layer: absorbed MLA over the paged latent cache (D2, G9). Row ``i`` (position ``a + i``) attends
        keys ``[0, a + i]``, all of them read from the cache, so the chunk's latent is filled first."""
        C = int(x.shape[-2])
        pc = self.sp1_global_program_config(C, kv_dtype=kv_cache.dtype)  # the (q, k) table of the cache it reads
        self._check_sp1_global(chunk, pc, C)
        cos, sin = self._rot_tables(chunk.rot)
        q, g, n, kpe, lam = self._project(x, decode=False)
        q_abs = self._q_absorbed(q, cos, sin)  # [1, 10, C, 576]

        # ---- fill FIRST (the chunk's own keys come from the cache), through the -1-skip fill table ------------------
        k_pe = self._rope(kpe, cos, sin)
        ttnn.deallocate(kpe)
        self._fill_latent(n, k_pe, chunk.fill_pt, kv_cache, taps=taps)
        ttnn.deallocate(n)
        ttnn.deallocate(k_pe)

        # ---- O = attention over keys [0, a + i] of the paged latent (K = V = cache; V = its first 512 columns) -------
        o = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_abs,
            kv_cache,
            kv_cache,
            chunk.sdpa_pt,
            chunk_start_idx_tensor=chunk.start_idx,
            scale=1.0,
            program_config=pc,
            compute_kernel_config=self.ckc_sdpa_sp1_global,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 10, C, 576]
        ttnn.deallocate(q_abs)
        o_lat = ttnn.slice(o, [0, 0, 0, 0], [1, self.H, C, self.rank])  # the attention-weighted k_pe is discarded
        ttnn.deallocate(o)
        return self._absorbed_epilogue(o_lat, g, lam, active=None, taps=taps, decode=False)

    def _check_sp1_tail(self, chunk: PrefillChunkInputs) -> int:
        """Rows of the SWA tail the chunk carries (``len(tail_bounds) x block``): must be the window minus the current
        key (128) and lie before the start."""
        bs = int(self.cfg.kv_block_size)
        T = len(chunk.tail_bounds) * bs
        want = int(self.window) - 1
        if T != want or T != int(self.cfg.prefill_swa_tail):
            raise ValueError(
                f"an sp1 SWA chunk needs the {want} cached rows before its start ({want // bs} tail blocks of {bs}), "
                f"the chunk carries {len(chunk.tail_bounds)}"
            )
        if int(chunk.start) < T or int(chunk.start) % bs:
            raise ValueError(f"sp1 SWA chunk start {chunk.start} must be a multiple of {bs} and >= the {T}-row tail")
        if T + int(chunk.bucket) > int(self.cfg.max_model_len):
            raise ValueError(
                f"the square [tail | chunk] SDPA of {T + int(chunk.bucket)} rows exceeds max_model_len "
                f"{self.cfg.max_model_len}, the longest single-shot SDPA validated on this model: sp1 chunks use "
                f"buckets <= max_sp1_bucket(cfg) = {max_sp1_bucket(self.cfg)} (the default config's span cap 8192 "
                "does; a span cap of max_model_len -- MOTIF3_PREFILL_MAX_BUCKET=32768, or max_model_len <= 8192 -- "
                "lets the planner emit a larger one)"
            )
        return T

    def _gather_tail(self, kv_cache, chunk: PrefillChunkInputs):
        """The SWA tail ``[1, 1, T, 576]`` bf16: the cached rows of positions ``[a - T, a)`` (roped ``k_pe``), gathered
        with one tensor-args dim-0 slice per tail block (bounds on the device: one program for every block id, G10a)
        -> ``concat(dim=2)`` in the cache dtype -> typecast."""
        nb = int(kv_cache.shape[0])
        parts = [
            ttnn.slice(kv_cache, s, e, slice_dim=0, num_devices=nb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for s, e in chunk.tail_bounds
        ]  # [1, 1, bs, 576] each
        cat = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if cat is not parts[0]:
            for t in parts:
                ttnn.deallocate(t)
        if cat.dtype == ttnn.bfloat16:
            return cat
        tail = ttnn.typecast(cat, ttnn.bfloat16)
        ttnn.deallocate(cat)
        return tail

    def _prefill_sp1_swa(self, x, chunk: PrefillChunkInputs, kv_cache, taps):
        """sp1 SWA layer: square ``[tail ‖ chunk]`` causal SDPA with the 129-key window (D3, G10). Square row
        ``T + i`` is position ``a + i`` and sees exactly keys ``[a + i - 128, a + i]``; the ``T`` filler rows only
        produce dropped outputs (query rows are independent)."""
        C = int(x.shape[-2])
        T = self._check_sp1_tail(chunk)
        cos, sin = self._rot_tables(chunk.rot)
        q, g, n, kpe, lam = self._project(x, decode=False)

        # ---- Q_cat [1, 10, T + C, 192]: T filler rows (the chunk's first rows; outputs dropped) | Q (HF order) -------
        q_full = self._q_expanded(q, cos, sin)
        q_pad = ttnn.slice(q_full, [0, 0, 0, 0], [1, self.H, T, self.cfg.head_dim])
        q_cat = ttnn.concat([q_pad, q_full], dim=2)
        ttnn.deallocate(q_pad)
        ttnn.deallocate(q_full)

        # ---- latent [tail (cache) | chunk] -> expanded K / V_pad over T + C rows -------------------------------------
        k_pe = self._rope(kpe, cos, sin)
        ttnn.deallocate(kpe)
        kv_row = ttnn.concat([n, k_pe], dim=-1)  # [1, 1, C, 576] bf16: the chunk's latent (what the fill writes)
        ttnn.deallocate(n)
        ttnn.deallocate(k_pe)
        tail = self._gather_tail(kv_cache, chunk)  # [1, 1, T, 576] bf16
        lat = ttnn.concat([tail, kv_row], dim=2)  # [1, 1, T + C, 576]
        ttnn.deallocate(tail)
        n_cat, kpe_cat = ttnn.experimental.nlp_create_q_heads_split(lat, num_heads=1, split_head_dim=self.rank)
        ttnn.deallocate(lat)
        k_full, v_pad = self._expanded_kv(n_cat, kpe_cat)  # [1, 2, T + C, 192]
        ttnn.deallocate(n_cat)
        ttnn.deallocate(kpe_cat)

        # ---- square SDPA: causal + window 129, the sdpa_prefill role (never fp32 acc with a window), q/k 128/128 -----
        o = ttnn.transformer.scaled_dot_product_attention(
            q_cat,
            k_full,
            v_pad,
            is_causal=True,
            scale=1.0,
            sliding_window_size=self.window,
            program_config=self.cfg.resumed_prefill_pc(self.spec, C),
            compute_kernel_config=self.ckc_sdpa_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 10, T + C, 192]
        ttnn.deallocate(q_cat)
        ttnn.deallocate(k_full)
        ttnn.deallocate(v_pad)
        o_v = ttnn.slice(o, [0, 0, T, 0], [1, self.H, T + C, self.vdim])  # the chunk's rows
        ttnn.deallocate(o)
        out = self._expanded_epilogue(o_v, g, lam, taps)

        # ---- fill (after the SDPA, as sp0): the chunk's rows through the -1-skip fill table --------------------------
        if taps is not None:
            taps["kv_row"] = kv_row
        self._fill_rows(kv_row, chunk.fill_pt, kv_cache, keep=taps is not None)
        return out

    # ---- packed passes (P5): docs/p5_t64/P5_T64_DESIGN.md §3.4, gate G15a (GATES_RESULTS §13.2) ----------------------
    def _prefill_packed_sp0(self, x, chunk: PrefillChunkInputs, kv_cache, taps, B: int, S: int):
        """pk0 pass: ``B`` sp0 segments of ``S`` rows. :meth:`forward_prefill`'s bucket-``T`` op sequence with the
        gathered RoPE rows (``chunk.rot``: every segment at positions ``0 .. S-1``), except the SDPA: ``q_full`` /
        ``k_full`` / ``v_pad`` -> :func:`_to_segments` -> ONE batched causal SDPA with the per-row bucket-``S`` config
        and window rule (:meth:`prefill_sdpa_window_and_config` ``(S)``) -> :func:`_from_segments`. Per segment that is
        bitwise the single-row SDPA of a bucket-``S`` chunk (G15a (a)). The fill follows the SDPA, through the packed
        ``[1, T / bs]`` table (``-1``: shared blocks, padding blocks, dummy segments); ``kv_cache=None`` skips it, as
        for sp0. ``taps``: ``u_flat`` and, with a cache, ``kv_row``."""
        T = B * S
        cos, sin = self._rot_tables(chunk.rot)
        q, g, n, kpe, lam = self._project(x, decode=False)
        q_full = self._q_expanded(q, cos, sin)  # [1, 10, T, 192], HF head order
        k_pe = self._rope(kpe, cos, sin)  # [1, 1, T, 64]
        ttnn.deallocate(kpe)
        k_full, v_pad = self._expanded_kv(n, k_pe)  # [1, 2, T, 192] each

        # ---- batched SDPA: segment k = batch row k, the per-row bucket-S program config and window -----------------
        qb, kb, vb = (_to_segments(t, B, S) for t in (q_full, k_full, v_pad))  # [B, 10 | 2, S, 192]
        for t in (q_full, k_full, v_pad):
            ttnn.deallocate(t)
        window, ckc = self.prefill_sdpa_window_and_config(S)
        o_b = ttnn.transformer.scaled_dot_product_attention(
            qb,
            kb,
            vb,
            is_causal=True,
            scale=1.0,
            sliding_window_size=window,
            program_config=self.cfg.sdpa_prefill_pc(self.spec, seq_len=S),
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [B, 10, S, 192] (columns 128..191 = 0)
        for t in (qb, kb, vb):
            ttnn.deallocate(t)
        o = _from_segments(o_b)  # [1, 10, T, 192]
        o_v = ttnn.slice(o, [0, 0, 0, 0], [1, self.H, T, self.vdim])
        ttnn.deallocate(o)
        out = self._expanded_epilogue(o_v, g, lam, taps)

        # ---- cache fill (after the SDPA, as sp0): segment k's rows through entries [k S / bs, (k + 1) S / bs) --------
        if kv_cache is not None:
            self._fill_latent(n, k_pe, chunk.fill_pt, kv_cache, taps=taps)
        ttnn.deallocate(n)
        ttnn.deallocate(k_pe)
        return out

    def _prefill_packed_sp1_global(self, x, chunk: PrefillChunkInputs, kv_cache, taps, B: int, S: int):
        """pk1 pass, global layer: ``B`` sp1 segments at the common start ``a``. ``Q_abs`` at bucket ``T``; the fill
        FIRST through the packed table (a segment's own keys come from the cache; no segment of a pass reads a block the
        pass writes, the planner's writer-first rule); ``Q_abs`` -> :func:`_to_segments` ``[B, 10, S, 576]`` -> ONE
        ``chunked_scaled_dot_product_attention`` over the paged cache with one SDPA page-table row per segment and the
        one start ``[a]`` (``cfg.resumed_prefill_pc("global", S, kv_dtype)``, the G9 role): row ``i`` of segment ``k``
        attends keys ``[0, a + i]`` of its own table -> :func:`_from_segments` -> ``[..., :512]`` -> the absorbed
        epilogue. Per segment bitwise the single-row sp1 call at bucket ``S`` (G15a (b))."""
        T = B * S
        pc = self.sp1_global_program_config(S, kv_dtype=kv_cache.dtype)  # the per-bucket (q, k) of S
        self._check_sp1_global(chunk, pc, S)
        cos, sin = self._rot_tables(chunk.rot)
        q, g, n, kpe, lam = self._project(x, decode=False)
        q_abs = self._q_absorbed(q, cos, sin)  # [1, 10, T, 576]

        # ---- fill FIRST, every segment through its entries of the packed -1-skip table ------------------------------
        k_pe = self._rope(kpe, cos, sin)
        ttnn.deallocate(kpe)
        self._fill_latent(n, k_pe, chunk.fill_pt, kv_cache, taps=taps)
        ttnn.deallocate(n)
        ttnn.deallocate(k_pe)

        # ---- batched chunked SDPA: segment k = batch row k, its own page-table row, one start -----------------------
        qb = _to_segments(q_abs, B, S)  # [B, 10, S, 576]
        ttnn.deallocate(q_abs)
        o_b = ttnn.transformer.chunked_scaled_dot_product_attention(
            qb,
            kv_cache,
            kv_cache,
            chunk.sdpa_pt,
            chunk_start_idx_tensor=chunk.start_idx,
            scale=1.0,
            program_config=pc,
            compute_kernel_config=self.ckc_sdpa_sp1_global,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [B, 10, S, 576]
        ttnn.deallocate(qb)
        o = _from_segments(o_b)  # [1, 10, T, 576]
        o_lat = ttnn.slice(o, [0, 0, 0, 0], [1, self.H, T, self.rank])  # the attention-weighted k_pe is discarded
        ttnn.deallocate(o)
        return self._absorbed_epilogue(o_lat, g, lam, active=None, taps=taps, decode=False)

    def _check_pk1_tail(self, chunk: PrefillChunkInputs, B: int, S: int) -> int:
        """Rows of every segment's SWA tail (the window minus the current key, 128) after the checks of the pass's tail
        variant (review edit R-E2): ``shared`` carries segment 0's ``tail / bs`` bound pairs, ``distinct`` every
        segment's (``B x tail / bs``, segment-major); the start lies past the tail; the square ``[tail ‖ segment]`` fits
        ``max_model_len``."""
        bs, n = int(self.cfg.kv_block_size), len(chunk.tail_bounds)
        if chunk.tails not in PK1_TAIL_VARIANTS:
            raise ValueError(f"a pk1 SWA pass needs its tail variant, one of {PK1_TAIL_VARIANTS}, got {chunk.tails!r}")
        per = n if chunk.tails == SHARED_TAILS else (n // B if n % B == 0 else 0)
        Tt, want = per * bs, int(self.window) - 1
        if Tt != want or Tt != int(self.cfg.prefill_swa_tail):
            pairs = f"{want // bs}" if chunk.tails == SHARED_TAILS else f"{want // bs} x {B}"
            raise ValueError(
                f"a pk1 SWA pass needs every segment's {want} cached rows before the start ({want // bs} tail blocks "
                f"of {bs}): {chunk.tails} tails carry {pairs} bound pairs, the inputs hold {n}"
            )
        if int(chunk.start) < Tt or int(chunk.start) % bs:
            raise ValueError(f"pk1 SWA pass start {chunk.start} must be a multiple of {bs} and >= the {Tt}-row tail")
        if Tt + S > int(self.cfg.max_model_len):
            raise ValueError(f"the square [tail | segment] SDPA of {Tt + S} rows exceeds max_model_len")
        return Tt

    def _gather_tails_packed(self, kv_cache, chunk: PrefillChunkInputs, B: int, Tt: int):
        """The SWA tails of a pk1 pass, ``[B, 1, Tt, 576]`` bf16: row ``k`` = segment ``k``'s cached rows of positions
        ``[a - Tt, a)`` (roped ``k_pe``; ``Tt`` = 128). The variant is fixed per pass, never a partial dedupe (the
        program set must not depend on the data; review edit R-E2):

        * ``shared`` (every segment has segment 0's tail blocks): the solo 2-block gather (:meth:`_gather_tail` on
          segment 0's bounds) + ``repeat`` over the batch;
        * ``distinct``: ``2 B`` tensor-args dim-0 block slices (segment-major; one program for every block id) -> ONE
          ``concat(dim=2)`` ``[1, 1, B Tt, 576]`` in the cache dtype (ttnn splits a concat of more than 47 inputs into
          batches) -> metadata view ``[B, 1, Tt, 576]`` -> typecast to bf16 (a bf16 cache: the view itself, which
          shares the concat's buffer; the caller frees it)."""
        if chunk.tails == SHARED_TAILS:
            tail1 = self._gather_tail(kv_cache, chunk)  # [1, 1, Tt, 576] bf16
            tail_b = ttnn.repeat(tail1, ttnn.Shape([B, 1, 1, 1]))
            ttnn.deallocate(tail1)
            return tail_b
        nb = int(kv_cache.shape[0])
        parts = [
            ttnn.slice(kv_cache, s, e, slice_dim=0, num_devices=nb, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for s, e in chunk.tail_bounds
        ]  # 2 B x [1, 1, bs, 576]
        cat = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, B Tt, 576]
        for t in parts:
            ttnn.deallocate(t)
        view = ttnn.reshape(cat, (B, 1, Tt, self.latent_dim))  # metadata only
        if cat.dtype == ttnn.bfloat16:
            return view
        tail_b = ttnn.typecast(view, ttnn.bfloat16)
        ttnn.deallocate(cat)
        return tail_b

    def _prefill_packed_sp1_swa(self, x, chunk: PrefillChunkInputs, kv_cache, taps, B: int, S: int):
        """pk1 pass, SWA layer: per segment the square ``[tail ‖ segment]`` causal window-129 SDPA of
        :meth:`_prefill_sp1_swa`, batched. ``Q_cat [B, 10, Tt + S, 192]`` = every segment's first ``Tt = 128`` rows
        (filler, dropped) then its rows; the latent ``[tails (cache, :meth:`_gather_tails_packed`) ‖ view(kv_row,
        [B, 1, S, 576])]`` -> the draft-1 expansion at batch ``B``; one SDPA with ``cfg.resumed_prefill_pc("swa", S)``
        and the ``sdpa_prefill`` role; rows ``[Tt, Tt + S)`` -> :func:`_from_segments` -> the draft-1 epilogue at
        ``B S``; the fill after the SDPA. Per segment bitwise the solo sp1 SWA dataflow at bucket ``S`` (G15a (c))."""
        Tt = self._check_pk1_tail(chunk, B, S)
        cos, sin = self._rot_tables(chunk.rot)
        q, g, n, kpe, lam = self._project(x, decode=False)

        # ---- Q_cat [B, 10, Tt + S, 192]: per segment its first Tt rows (filler; outputs dropped) | its Q (HF order) --
        q_full = self._q_expanded(q, cos, sin)  # [1, 10, B S, 192]
        qb = _to_segments(q_full, B, S)  # [B, 10, S, 192]
        ttnn.deallocate(q_full)
        q_pad = ttnn.slice(qb, [0, 0, 0, 0], [B, self.H, Tt, self.cfg.head_dim])
        q_cat = ttnn.concat([q_pad, qb], dim=2)
        if S != Tt:  # S == Tt: the full-extent slice is a ttnn no-op returning qb itself (GATES_RESULTS §13.2)
            ttnn.deallocate(q_pad)
        ttnn.deallocate(qb)

        # ---- latent [tails (cache) | segment] -> expanded K / V_pad over Tt + S rows per segment ---------------------
        k_pe = self._rope(kpe, cos, sin)
        ttnn.deallocate(kpe)
        kv_row = ttnn.concat([n, k_pe], dim=-1)  # [1, 1, B S, 576] bf16: the pass's latent (what the fill writes)
        ttnn.deallocate(n)
        ttnn.deallocate(k_pe)
        tail_b = self._gather_tails_packed(kv_cache, chunk, B, Tt)  # [B, 1, Tt, 576] bf16
        lat = ttnn.concat([tail_b, ttnn.reshape(kv_row, (B, 1, S, self.latent_dim))], dim=2)  # [B, 1, Tt + S, 576]
        ttnn.deallocate(tail_b)
        n_cat, kpe_cat = ttnn.experimental.nlp_create_q_heads_split(lat, num_heads=1, split_head_dim=self.rank)
        ttnn.deallocate(lat)
        k_full, v_pad = self._expanded_kv(n_cat, kpe_cat)  # [B, 2, Tt + S, 192]
        ttnn.deallocate(n_cat)
        ttnn.deallocate(kpe_cat)

        # ---- batched square SDPA: causal + window 129, the sdpa_prefill role, the per-row bucket-S config ------------
        o = ttnn.transformer.scaled_dot_product_attention(
            q_cat,
            k_full,
            v_pad,
            is_causal=True,
            scale=1.0,
            sliding_window_size=self.window,
            program_config=self.cfg.resumed_prefill_pc(self.spec, S),
            compute_kernel_config=self.ckc_sdpa_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [B, 10, Tt + S, 192]
        for t in (q_cat, k_full, v_pad):
            ttnn.deallocate(t)
        o_v = ttnn.slice(o, [0, 0, Tt, 0], [B, self.H, Tt + S, self.vdim])  # every segment's rows [B, 10, S, 128]
        ttnn.deallocate(o)
        out = self._expanded_epilogue(_from_segments(o_v), g, lam, taps)

        # ---- fill (after the SDPA): segment k's rows through entries [k S / bs, (k + 1) S / bs) ----------------------
        if taps is not None:
            taps["kv_row"] = kv_row
        self._fill_rows(kv_row, chunk.fill_pt, kv_cache, keep=taps is not None)
        return out

    # ==================================================================================================================
    # KV-only latent fill (MTP prefill, chunked fills; features design §3.6.3, §3.2.2 step 3)
    # ==================================================================================================================
    def fill_kv(
        self,
        x,
        *,
        fill_pt=None,
        kv_cache=None,
        rot: RotArg = None,
        taps: Optional[Dict[str, Any]] = None,
        chunk: Optional[PrefillChunkInputs] = None,
    ) -> None:
        """Write the latent of ``x``'s rows into ``kv_cache`` without running the attention (eager; ~8 ops).

        ``x @ Wkv_lat`` -> ``n = rms_norm(c_raw)`` -> ``k_pe = rope(kpe)`` at the rows' positions ->
        ``typecast(concat(n, k_pe))`` -> ``paged_fill_cache`` through ``fill_pt``. No q path, SDPA, ``wo`` or CCL; the
        cache is the only output. The ops are those of :meth:`forward_prefill` (:meth:`_kv_latent`, :meth:`_split_kv`,
        :meth:`_fill_latent`), so the written rows are bitwise the rows a prefill of the same ``x`` with the same RoPE
        rows and table writes. A row's latent depends only on its own input row and position (prefix-independent):
        that is why the MTP prefill needs only this (features design D9) and why a chunk's rows can be filled alone.

        Args:
            x: the normalized layer input ``[1, 1, C, 4096]`` bf16 TILE DRAM, replicated on all chips (``C`` = bucket
                rows; rows past the request's end are padding and go wherever ``fill_pt`` sends them).
            fill_pt: ``[1, >= cdiv(C, block)]`` int32 ROW_MAJOR: entry ``j`` = cache block of rows ``[j * block,
                (j + 1) * block)``, ``-1`` = skip (``prefill_plan.fill_table``: shared full blocks below ``w0``, pure
                padding blocks). Several requests' rows may be concatenated with their tables (row ``i`` goes through
                entry ``i // block``).
            kv_cache: the layer's paged latent cache ``[N, 1, block, 576]`` (``cfg.dtypes.kv_cache``), updated in place.
            rot: RoPE tables of the rows' positions: ``(cos, sin)`` ``[1, 1, >= C, 64]`` TILE for this layer's kind, or
                ``{kind: (cos, sin)}`` (``rope.chunk_rope_tables(rope.chunk_rot_idxs_device(positions))`` for rows at
                any positions, built once per chunk for all layers). ``None`` = positions ``0 .. C-1``
                (``rope.prefill_cos_sin(kind, C)``, the sp0 tables).
            taps: optional dict that receives ``kv_row``, the bf16 latent ``[1, 1, C, 576]`` before the typecast
                (eager debugging / tests).
            chunk: a :class:`PrefillChunkInputs` in place of ``fill_pt`` / ``rot`` (its fill table and RoPE rows; the
                MTP KV-only fill of a chunk: ``fill_kv(a, chunk=inp, kv_cache=mtp_cache)``).
        """
        if chunk is not None:
            if not isinstance(chunk, PrefillChunkInputs):
                raise TypeError(f"chunk must be a PrefillChunkInputs, got {type(chunk).__name__}")
            if fill_pt is not None or rot is not None:
                raise ValueError("fill_kv: pass chunk= or fill_pt= / rot=, not both (the chunk carries its tables)")
            self._check_chunk_rows(x, chunk, "fill_kv")
            fill_pt, rot = chunk.fill_pt, chunk.rot
        if kv_cache is None or fill_pt is None:
            raise ValueError("fill_kv needs kv_cache= and fill_pt= (or chunk=)")
        if len(x.shape) != 4 or int(x.shape[-1]) != self.cfg.hidden_size:
            raise ValueError(f"fill_kv expects x [1, 1, C, {self.cfg.hidden_size}], got {tuple(x.shape)}")
        C = int(x.shape[-2])
        cos, sin = self._prefill_rot_tables(rot, C, "fill_kv")
        n, kpe, lam = self._split_kv(self._kv_latent(x, decode=False))
        ttnn.deallocate(lam)
        k_pe = self._rope(kpe, cos, sin)  # [1, 1, C, 64]
        ttnn.deallocate(kpe)
        self._fill_latent(n, k_pe, fill_pt, kv_cache, taps=taps)
        ttnn.deallocate(n)
        ttnn.deallocate(k_pe)

    def _prefill_rot_tables(self, rot: RotArg, rows: int, where: str):
        """``(cos, sin)`` with ``>= rows`` rows for a prefill-layout call: ``None`` -> ``rope.prefill_cos_sin(kind,
        rows)`` (positions ``0 .. rows-1``), a pair or a ``{kind: pair}`` dict as given. Raw index tensors are refused
        (gather them once per chunk with ``rope.chunk_rope_tables``; ``_rot_tables`` would gather decode tables)."""
        if rot is None:
            if self.rope is None:
                raise ValueError(f"{where} needs a MotifRope or rot=(cos, sin)")
            return self.rope.prefill_cos_sin(self.kind, rows)
        if not isinstance(rot, (dict, tuple, list)):
            raise TypeError(
                f"{where}: rot must be (cos, sin) or {{kind: (cos, sin)}} with >= {rows} rows (rows at any positions: "
                "rope.chunk_rope_tables(rope.chunk_rot_idxs_device(positions))), got "
                f"{type(rot).__name__}"
            )
        cos, sin = self._rot_tables(rot)
        if int(cos.shape[-2]) < rows or int(sin.shape[-2]) < rows:
            raise ValueError(f"{where}: RoPE tables have {int(cos.shape[-2])} rows, the input has {rows}")
        return cos, sin


# ======================================================================================================================
# small helpers
# ======================================================================================================================
def _to_segments(t, B: int, S: int):
    """Packed rows -> batch of segments: ``[1, H, B S, d]`` -> metadata view ``[H, B, S, d]`` (the last dim unchanged,
    ``S`` a tile multiple: same buffer) -> CN ``transpose(0, 1)`` -> ``[B, H, S, d]`` (a new tensor; batch row ``k`` =
    packed rows ``[k S, k S + S)``). Does not consume ``t``. The SDPA ops batch over dim 0 only: the absorbed per-head
    matmuls keep ``[1, H, T, d]`` (an in1 batch of 1), so the packed layout stays ``[1, H, T, d]`` outside the SDPA."""
    H, d = int(t.shape[1]), int(t.shape[3])
    return ttnn.transpose(ttnn.reshape(t, (H, int(B), int(S), d)), 0, 1)


def _from_segments(t):
    """Inverse of :func:`_to_segments`: ``[B, H, S, d]`` -> CN ``transpose(0, 1)`` -> ``[H, B, S, d]`` -> metadata view
    ``[1, H, B S, d]``. Consumes ``t``. The result is a view of the transposed tensor: deallocating it frees that buffer
    (never deallocate the transposed tensor separately)."""
    B, H, S, d = (int(s) for s in t.shape)
    tt = ttnn.transpose(t, 0, 1)
    ttnn.deallocate(t)
    return ttnn.reshape(tt, (1, H, B * S, d))


def _shard_rows(rows: torch.Tensor, cfg: MotifTTConfig, mesh_device, dtype, layout, device=None):
    """``[dp, ...]`` host tensor -> mesh tensor with DP row ``r`` holding ``rows[r]`` (``[1, ...]``), replicated over TP
    (same placement as ``rope.shard_lanes``, any layout / dtype)."""
    dims = cfg.axes.mesh_dims(dp_dim=0, tp_dim=None)
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate() if d is None else ttnn.PlacementShard(d) for d in dims],
            ttnn.MeshShape(*cfg.axes.mesh_shape),
        ),
    )
    return ttnn.from_torch(
        rows,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if device is not None else None,
        mesh_mapper=mapper,
    )


__all__ = [
    "ATTN_CACHE_VERSION",
    "ChunkHostTables",
    "DecodeKVWriter",
    "L1_SMALL_WARNING",
    "MASK_WIDTH",
    "MTP_ATTN_PREFIX",
    "MotifAttention",
    "PackedHostTables",
    "PrefillChunkInputs",
    "RECOMMENDED_L1_SMALL_SIZE",
    "attn_weight_prefix",
    "canonical_attn_spec",
    "check_l1_small",
    "chunk_host_tables",
    "decode_matmul_program_configs",
    "hf_order_from_virtual",
    "l1_small_bytes",
    "lambda_expansion",
    "latent_kv_weight_for_chip",
    "latent_q_weight",
    "max_sp1_bucket",
    "noise_expansion",
    "packed_host_tables",
    "resolve_attn_layer",
    "virtual_head_groups",
    "virtual_head_order",
    "w_uk_virtual_for_chip",
    "w_uv_virtual_for_chip",
    "warmup_chunk_host_tables",
    "warmup_packed_host_tables",
    "warmup_start",
    "wq_b_gate_for_chip",
    "wq_b_virtual_for_chip",
]
