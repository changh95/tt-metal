# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel full-attention for Qwen3.5 (validated 64k+ on 27B).

Q/K-norm: HF-correct (1+weight) uniformly at prefill and decode.
Keep Q bf16 into SDPA unless bf8 mode (QWEN_SDPA_BF8=1).
Weights interleaved per device; x replicated in, output reduce-scattered on dim=3.
"""
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, apply_partial_rope_prefill
from models.tt_transformers.tt.ccl import tt_all_reduce

_SDPA_DEC_K_CHUNK = int(os.environ.get("QWEN36_SDPA_DEC_K_CHUNK", "0"))  # 0 = ttnn DYNAMIC_CHUNK_SIZE (default)
_SDPA_DEC_MAX_CORES_PER_HEAD = int(os.environ.get("QWEN36_SDPA_DEC_MAX_CORES_PER_HEAD", "0"))  # 0 = factory default
# Verify step (batched attention middle) only: cap on SDPA cores per virtual user; 0 = the decode knob / factory default
_VERIFY_SDPA_MAX_CORES = int(os.environ.get("QWEN36_VERIFY_SDPA_MAX_CORES", "0"))


def load_attention_weights_tp(mesh, state_dict, args, cache_dir=None):
    """Shard one full-attention layer's weights across the mesh."""
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    tw = {}
    # Column-parallel q/k/v: fused [q+gate|k|v] per device, or separate DRAM-sharded weights.
    # Distinct cache names — as_tensor reload ignores requested memcfg.
    fused_qkv = getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None
    # De-interleave [q,gate] per head → contiguous q/gate slices (avoids ~5.3ms relayout).
    qg_deint = fused_qkv

    # TP > n_kv_heads (e.g. 27B's 4 KV heads on TP=8): there is no whole KV head per device, so
    # pre-expand K/V to tp*head_dim rows where device d holds the head its GQA query group maps
    # to (devices 2d, 2d+1 share head d at TP=8). The per-device slicing below is then uniform.
    # No-op when tp <= n_kv_heads, so TP=4 weights stay bit-identical.
    kv_rep = lambda w: tpc.replicate_kv_weight(w, args.n_kv_heads, args.num_devices, args.head_dim)
    k_proj, v_proj = kv_rep(state_dict["k_proj.weight"]), kv_rep(state_dict["v_proj.weight"])

    if fused_qkv:
        if qg_deint:
            fused = tpc.prepare_attn_qkv_deint(
                state_dict["q_proj.weight"],
                k_proj,
                v_proj,
                args.n_local_heads,
                args.head_dim,
                args.n_local_kv_heads * args.head_dim,
                args.num_devices,
            )
        else:
            fused = tpc.prepare_attn_qkv(
                state_dict["q_proj.weight"],
                k_proj,
                v_proj,
                args.n_local_heads * args.head_dim * 2,
                args.n_local_kv_heads * args.head_dim,
                args.num_devices,
            )
        # proj_1d_decode: interleaved weight (fast small-grid 1D decode matmul; prefill AGMM verified
        # bit-identical on interleaved — test_agmm_accepts_interleaved_weight). Distinct cache suffix.
        _proj1d = getattr(args, "proj_1d_decode", False)
        _base = "wqkv_fused_qkvg" if qg_deint else "wqkv_fused"
        tw["wqkv_fused"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if _proj1d else args.attn_qkv_fused_weight_memcfg,
            cache_path=c(_base + (".il" if _proj1d else ".dramshard")),
            dtype=ttnn.bfloat8_b,
        )
    else:
        qkv_sharded = getattr(args, "attn_qg_weight_memcfg", None) is not None
        qg_mc = args.attn_qg_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        k_mc = args.attn_k_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        v_mc = args.attn_v_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        tag = ".dramshard" if qkv_sharded else ""
        tw["wqkv"] = tpc.shard_w(
            state_dict["q_proj.weight"],
            mesh,
            dim=-1,
            memory_config=qg_mc,
            cache_path=c("wqkv" + tag),
            dtype=ttnn.bfloat8_b,
        )
        # k_proj/v_proj are the KV-replicated weights: shard_w splits tp*head_dim rows evenly, so
        # each device lands on its GQA-assigned head instead of a fraction of one.
        tw["wk"] = tpc.shard_w(
            k_proj,
            mesh,
            dim=-1,
            memory_config=k_mc,
            cache_path=c("wk" + tag),
            dtype=ttnn.bfloat8_b,
        )
        tw["wv"] = tpc.shard_w(
            v_proj,
            mesh,
            dim=-1,
            memory_config=v_mc,
            cache_path=c("wv" + tag),
            dtype=ttnn.bfloat8_b,
        )
    # Row-parallel wo (reduce-scatter after): DRAM-width-sharded like the in-proj — decode tput win.
    wo_sharded = getattr(args, "attn_wo_weight_memcfg", None) is not None
    tw["wo"] = tpc.shard_w(
        state_dict["o_proj.weight"],
        mesh,
        dim=0,
        memory_config=args.attn_wo_weight_memcfg if wo_sharded else ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("wo.dramshard" if wo_sharded else "wo"),
        dtype=ttnn.bfloat8_b,
    )
    # QK norms: HF-correct zero-centered (1+weight), used uniformly at prefill AND decode
    tw["q_norm"] = tpc.replicate(state_dict["q_norm.weight"].to(torch.float32) + 1.0, mesh, None)
    tw["k_norm"] = tpc.replicate(state_dict["k_norm.weight"].to(torch.float32) + 1.0, mesh, None)
    return tw


class TPAttention:
    """Standalone TP full-attention with internal per-head KV caches (decode)."""

    def __init__(self, mesh, args, tw, tt_ccl):
        self.mesh = mesh
        self.args = args
        self.tw = tw
        self.tt_ccl = tt_ccl
        self.B = args.max_batch_size
        self._kv_shard_cfg_cache = {}  # active-width B -> KV-update height shard cfg (bucketed decode)
        self.NH = args.n_local_heads
        self.NKV = args.n_local_kv_heads
        self.HD = args.head_dim
        self.scale = self.HD**-0.5
        self.rope_dim = args.rope_head_dim
        self.compute_cfg = tpc.COMPUTE_HIFI2
        # bf8 SDPA (QWEN_SDPA_BF8=1): bf8 Q + bf8 KV; keeps HiFi2 (HiFi4 was slower)
        self._sdpa_bf8 = os.environ.get("QWEN_SDPA_BF8", "0") == "1"
        # Chunked-prefill SDPA knobs (env-gated, defaults unchanged). QWEN36_SDPA_K_CHUNK: K chunk (128; 256 amortises
        # the per-k-chunk handshakes of the GQA K/V multicast schedule, TT_SDPA_GQA_MCAST=1, at 2x the K/V CB L1).
        # QWEN36_SDPA_FULLSYNC=1: dst_full_sync_en for the SDPA compute config (8 fp32 dest tiles -> 2x4 subblocks).
        self._sdpa_k_chunk = int(os.environ.get("QWEN36_SDPA_K_CHUNK", "128"))
        self._sdpa_compute_cfg = self.compute_cfg
        # QWEN36_SDPA_BF16_DEST=1: bf16 DEST accumulation for the chunked SDPA (8 dest tiles -> 2x4 subblocks; numerics change,
        # gate on long-context PCC). QWEN36_SDPA_FULLSYNC=1: dst_full_sync_en (8 fp32 dest tiles).
        _sdpa_bf16_dest = os.environ.get("QWEN36_SDPA_BF16_DEST", "0") == "1"
        if os.environ.get("QWEN36_SDPA_FULLSYNC", "0") == "1" or _sdpa_bf16_dest:
            self._sdpa_compute_cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=self.compute_cfg.math_fidelity,
                math_approx_mode=self.compute_cfg.math_approx_mode,
                fp32_dest_acc_en=(not _sdpa_bf16_dest) and self.compute_cfg.fp32_dest_acc_en,
                packer_l1_acc=self.compute_cfg.packer_l1_acc,
                dst_full_sync_en=os.environ.get("QWEN36_SDPA_FULLSYNC", "0") == "1",
            )
        # Must match load_attention_weights_tp gates
        self._dram_sharded = getattr(args, "attn_qg_weight_memcfg", None) is not None
        self._wo_sharded = getattr(args, "attn_wo_weight_memcfg", None) is not None
        self._fused_qkv = getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None
        self._qg_deint = self._fused_qkv
        # Fuse prefill norm-allgather + fused-QKV in-proj (all_gather_minimal_matmul_async).
        # Norm's prefill post-AG disabled in layer.py; decode path unchanged.
        self._fuse_agmm = self._fused_qkv
        # Decode head split/merge via nlp_create/concat_heads_decode (the batched-decode idiom).
        self._use_nlp_decode_heads = True
        self.k_caches = None
        self.v_caches = None
        # External paged KV cache (vLLM/contract path); internal caches kept for demo fallback
        self.paged_k = None
        self.paged_v = None
        self.use_paged = False

    def set_paged_kv_cache(self, k_cache, v_cache):
        """Attach an externally-allocated paged KV cache (one call after allocate_kv_caches)."""
        self.paged_k = k_cache
        self.paged_v = v_cache
        self.use_paged = True

    def _qkv(self, x):
        """Q+gate/K/V projections → (qg, kp, vp). Fused path: one matmul, then slice."""
        tw = self.tw
        if not self._fused_qkv:
            return (
                self._col_proj(x, tw["wqkv"], self.args.attn_qg_progcfg),
                self._col_proj(x, tw["wk"], self.args.attn_k_progcfg),
                self._col_proj(x, tw["wv"], self.args.attn_v_progcfg),
            )
        # Fused weight is [q|k|v|gate] (prepare_attn_qkv_deint): the q|k|v block is contiguous, so
        # return it whole (no gate wedged between q and k → no re-concat in _make_heads*). Gate is
        # the trailing block. Sentinel: vp=None flags the fused/contiguous layout to _make_heads*.
        qkv3_dim = self.NH * self.HD + 2 * self.NKV * self.HD
        gate_dim = self.NH * self.HD
        # Prefill: x is K-sharded (norm skipped its AG) -> fused all-gather + QKV matmul. Output stays
        # DRAM: L1 clashes with a downstream matmul's CBs (verified; full-attn has more L1 pressure here).
        if self._fuse_agmm and tpc.small_m_in_proj(x.shape[-2]):
            # Small-M prefill (bucket 128): plain all-gather + 1D mcast matmul on the interleaved fused weight
            # (tp_common "Small-M prefill matmuls": AGMM 128 -> 93 us at 128 rows). DRAM output, sliced below.
            qkv = tpc.all_gather_linear_small_m(
                x,
                tw["wqkv_fused"],
                self.tt_ccl,
                self.compute_cfg,
                self.args.ccl_topology(),
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                grid_w=getattr(self.args, "decode_grid_w", 8),
            )
        elif self._fuse_agmm and x.shape[-2] > tpc.TILE_SIZE:
            # QWEN36_GDN_PROJ_CHUNKS: both split widths are multiples of HD (=128), so the AGMM can
            # write qkv3 and gate directly and the two ttnn.slice ops below disappear. No weight
            # padding is needed here (qkv3_dim + gate_dim == attn_qkv_fused_dim_tp exactly), but the
            # sum is asserted against the weight so a config change falls back instead of TT_FATALing.
            _chunks = tpc.proj_chunks_mode()
            if (
                _chunks
                and qkv3_dim % tpc.TILE_SIZE == 0
                and gate_dim % tpc.TILE_SIZE == 0
                and qkv3_dim + gate_dim == tw["wqkv_fused"].shape[-1]
            ):
                # One memory config for both chunks: DRAM, matching the un-chunked op output (mode 2
                # forces L1 for A/B — that puts the FULL qkv width in L1, the clash noted above).
                qkv3, gate = tpc.all_gather_matmul_prefill(
                    x,
                    tw["wqkv_fused"],
                    self.tt_ccl,
                    self.compute_cfg,
                    self.args.ccl_topology(),
                    out_memory_config=tpc.proj_chunks_memcfg(_chunks),
                    chunk_sizes=[qkv3_dim, gate_dim],
                )
                return qkv3, gate, None
            qkv = tpc.all_gather_matmul_prefill(
                x, tw["wqkv_fused"], self.tt_ccl, self.compute_cfg, self.args.ccl_topology()
            )
        elif getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: small-grid 1D matmul (interleaved weight). Output DRAM (the gate slice below lives across
            # SDPA in DRAM; the short-lived qkv3 slice lands in L1 for the head-split).
            qkv = tpc.matmul_1d_decode(
                x,
                tw["wqkv_fused"],
                self.args.attn_qkv_decode_1d_progcfg,
                self.compute_cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            qkv = self._col_proj(x, tw["wqkv_fused"], self.args.attn_qkv_fused_progcfg)
        sh = list(qkv.shape)
        # qkv3 is short-lived (split by _make_heads* then freed) -> L1 in prefill AND decode. Decode:
        # nlp_create_qkv_heads_decode needs an L1 input (tt-metal #16667: DRAM input zeros odd Q rows), so
        # slicing straight into L1 replaces the old DRAM slice + to_memory_config Copy in _make_heads_decode.
        # (The former "L1 qkv3 breaks the decode trace" note was that Copy turning into a no-op while the
        # follow-up deallocate freed the live tensor; _make_heads_decode no longer copies or frees its input.)
        # gate lives across SDPA (applied post-concat) -> always DRAM.
        qkv3 = ttnn.slice(qkv, (0, 0, 0, 0), (sh[0], sh[1], sh[2], qkv3_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.slice(qkv, (0, 0, 0, qkv3_dim), (sh[0], sh[1], sh[2], qkv3_dim + gate_dim))
        ttnn.deallocate(qkv)
        return qkv3, gate, None

    def _col_proj(self, x, weight, decode_progcfg):
        """Column-parallel projection; DRAM-sharded decode matmul when enabled."""
        if not self._dram_sharded:
            return ttnn.linear(x, weight, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.compute_cfg,
            decode_progcfg,
            self.args.act_shard_hidden,
            self.args.prefill_progcfg,
            self.args.dim,
        )

    def _wo_proj(self, x, weight):
        """Row-parallel output projection: DRAM-sharded decode/prefill matmul (K=attn_out_dim_tp),
        matching the in-proj. Falls back to plain interleaved when no sharded memcfg."""
        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter, or L1 for the
            # fused all-reduce (L1->L1 reshard into the 32-core width shard; see gdn/tp.py _row_proj).
            _out_mc = (
                ttnn.L1_MEMORY_CONFIG
                if getattr(self.tt_ccl, "decode_all_reduce", None) is not None
                else ttnn.DRAM_MEMORY_CONFIG
            )
            return tpc.matmul_1d_decode(
                x,
                weight,
                self.args.attn_wo_decode_1d_progcfg,
                self.compute_cfg,
                out_memory_config=_out_mc,
            )
        if not self._wo_sharded:
            if tpc.small_m_rows(x.shape[-2]):
                # Small-M prefill: 1D mcast config (2D 52 -> 1D 32 us at 128 rows, 53 -> 43 at 256); L1 output
                # feeds the separate RS as below.
                return tpc.small_m_linear(
                    x,
                    weight,
                    self.compute_cfg,
                    out_memory_config=ttnn.L1_MEMORY_CONFIG,
                    grid_w=getattr(self.args, "decode_grid_w", 8),
                )
            if x.shape[-2] > tpc.TILE_SIZE:
                # Prefill: FPU-tuned 2D config beats ttnn-auto's 1x1 stall; L1 output (gated stays DRAM)
                # feeds the separate RS. max_cols = device width (11 on BH): wide grid (~10-wide) + the
                # existing L1-out. See test_mlp_matmul_sweep_prefill.
                pc = tpc.create_prefill_mlp_matmul_program_config(
                    x.shape[-2],
                    weight.shape[-2],
                    weight.shape[-1],
                    max_cols=getattr(self.args, "decode_grid_w", 8),
                    tuning=getattr(self.args, "prefill_tuning", None),
                )
                return ttnn.linear(
                    x,
                    weight,
                    compute_kernel_config=self.compute_cfg,
                    program_config=pc,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            return ttnn.linear(x, weight, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.compute_cfg,
            self.args.attn_wo_progcfg,
            self.args.act_shard_attn_out,
            self.args.prefill_progcfg,
            self.args.attn_out_dim_tp,
        )

    def _make_heads(self, qg, kp, vp, S):
        """Split qg into heads; returns (q, gate_flat, k, v) via fused nlp_create_qkv_heads.

        gate_flat stays flat [1,1,S,NH*HD] (col h*HD+d = head h, dim d), matching nlp_concat_heads'
        column order. Gate is applied AFTER concat_heads (see forward_prefill*), so no head-major
        reshape/transpose is needed; bit-identical to per-head gating, saves ~1 ms/attn-layer at S=2048.
        """
        NH, NKV, HD = self.NH, self.NKV, self.HD
        if vp is None:
            # Fused [q|k|v|gate] weight (_qkv sentinel vp=None): qg is the contiguous [q|k|v] block,
            # kp is the gate. Slice q and (already-contiguous) kv directly — no concat needed.
            gate_flat = kp
            # q_flat, kv feed nlp_create_qkv_heads then free immediately -> L1 (short-lived, no clash).
            q_flat = ttnn.slice(qg, (0, 0, 0, 0), (1, 1, S, NH * HD), memory_config=ttnn.L1_MEMORY_CONFIG)
            kv = ttnn.slice(
                qg, (0, 0, 0, NH * HD), (1, 1, S, NH * HD + 2 * NKV * HD), memory_config=ttnn.L1_MEMORY_CONFIG
            )
            ttnn.deallocate(qg)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                q_flat,
                kv,
                num_heads=NH,
                num_kv_heads=NKV,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(q_flat)
            ttnn.deallocate(kv)
            return q, gate_flat, k, v
        # Interleaved qg: split [q;gate] per head; gate flattened to [1,1,S,NH*HD] (applied post-concat).
        qg = ttnn.reshape(qg, (1, S, NH, 2 * HD))
        q_part, gate_part = ttnn.chunk(qg, 2, dim=-1)
        ttnn.deallocate(qg)
        gate_flat = ttnn.reshape(gate_part, (1, 1, S, NH * HD))
        ttnn.deallocate(gate_part)
        q_flat = ttnn.reshape(q_part, (1, 1, S, NH * HD))
        ttnn.deallocate(q_part)
        kv = ttnn.concat([kp, vp], dim=-1)
        ttnn.deallocate(kp)
        ttnn.deallocate(vp)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            q_flat,
            kv,
            num_heads=NH,
            num_kv_heads=NKV,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_flat)
        ttnn.deallocate(kv)
        return q, gate_flat, k, v

    def _concat_heads(self, gated):
        """Prefill concat-heads via nlp_concat_heads (post-gate). L1 output: short-lived post-SDPA temp,
        no kernel-CB clash."""
        return ttnn.experimental.nlp_concat_heads(gated, memory_config=ttnn.L1_MEMORY_CONFIG)

    def _make_heads_decode(self, qg, kp, vp, B):
        """Decode head-split via nlp_create_qkv_heads_decode (the batched-decode idiom).

        Returns (q, gate_flat, k, v): q [1,B,NH,HD] and k [1,B,NKV,HD] L1-interleaved (rms_norm rejects
        HEIGHT_SHARDED input, and the partial rope runs interleaved); v [1,B,NKV,HD] left in the head-split's
        L1 HEIGHT_SHARDED layout (B cores, {32,HD} ROW_MAJOR shards) because its only consumer,
        paged_update_cache, validates exactly that layout; gate_flat [1,1,B,NH*HD] (col h*HD+d = head h,
        dim d), applied AFTER nlp_concat_heads_decode like the prefill path — no head-major reshape.
        The kernel only shuffles a fused Q|K|V, so the gate block is kept aside. The fused tensor must be
        in L1 to dodge the Blackhole interleaved-reader bug (tt-metal #16667: DRAM input zeros odd-indexed
        Q rows); _qkv slices it into L1 directly.
        """
        NH, NKV, HD = self.NH, self.NKV, self.HD
        _L1 = ttnn.L1_MEMORY_CONFIG
        if vp is None:
            # Fused [q|k|v|gate] weight (_qkv sentinel vp=None): qg is already the contiguous [q|k|v]
            # the decode head-split wants — feed it directly, no concat. kp is the gate.
            if qg.memory_config().buffer_type == ttnn.BufferType.L1:
                qkv = qg
            else:
                # Defensive: a DRAM producer -> real copy into L1 (#16667), then free the source.
                qkv = ttnn.to_memory_config(qg, _L1)
                ttnn.deallocate(qg)
            gate_flat = kp
        else:
            # Interleaved qg: [q;gate] per head -> split then re-flatten to [1,1,B,NH*HD].
            qg_r = ttnn.reshape(qg, (1, B, NH, 2 * HD), memory_config=_L1)
            ttnn.deallocate(qg)
            q_part = ttnn.slice(qg_r, (0, 0, 0, 0), (1, B, NH, HD), memory_config=_L1)
            gate_part = ttnn.slice(qg_r, (0, 0, 0, HD), (1, B, NH, 2 * HD), memory_config=_L1)
            ttnn.deallocate(qg_r)
            q_flat = ttnn.reshape(q_part, (1, 1, B, NH * HD), memory_config=_L1)
            ttnn.deallocate(q_part)
            gate_flat = ttnn.reshape(gate_part, (1, 1, B, NH * HD), memory_config=_L1)
            ttnn.deallocate(gate_part)
            qkv = ttnn.concat([q_flat, kp, vp], dim=-1, memory_config=_L1)
            ttnn.deallocate(q_flat)
            ttnn.deallocate(kp)
            ttnn.deallocate(vp)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=NH, num_kv_heads=NKV, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        q = ttnn.sharded_to_interleaved(q, _L1)
        k = ttnn.sharded_to_interleaved(k, _L1)
        return q, gate_flat, k, v

    def _concat_heads_decode(self, attn, B):
        """Decode concat-heads via nlp_concat_heads_decode. attn [1,B,NH,HD] L1 -> [1,1,B,NH*HD] L1.

        The op wants a height-sharded input ([1,B,heads-padded-to-32,HD], one core per user), so the
        SDPA output is resharded across `B` cores first (a grid-width-aligned rectangle — a ragged
        core set is rejected by the height-sharded mem config). Output is width-sharded, then
        returned to L1-interleaved. Consumes (deallocates) `attn`. The caller applies the sigmoid gate
        on the flat result (col h*HD+d == head h, dim d).
        """
        from models.tt_transformers.tt.model_config import num_to_corerange

        NH, HD = self.NH, self.HD
        _L1 = ttnn.L1_MEMORY_CONFIG
        grid = self.mesh.compute_with_storage_grid_size()
        gx = min(B, grid.x)
        if B >= gx and B % gx != 0:
            gx = max(x for x in range(gx, 0, -1) if B % x == 0 and B // x <= grid.y)
        core_grid = ttnn.CoreRangeSet({num_to_corerange(B, grid_x=gx, grid_y=grid.y)})
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, HD),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn_sh = ttnn.to_memory_config(attn, shard_cfg)
        ttnn.deallocate(attn)
        out_sh = ttnn.experimental.nlp_concat_heads_decode(attn_sh, num_heads=NH)
        ttnn.deallocate(attn_sh)
        out = ttnn.sharded_to_interleaved(out_sh, _L1)  # [1, 1, 32, NH*HD] (batch padded to 32)
        ttnn.deallocate(out_sh)
        if out.shape[-2] != B:
            # nlp_concat_heads_decode always emits batch padded to 32. Shrink the LOGICAL batch to B with a
            # padded-shape view (reshape.cpp tile_tensor_view_reshape_possible: TILE, padded[-2] % 32 == 0,
            # last dim unchanged) — zero-cost, replacing the former Slice kernel; rows B..31 stay padding.
            out = ttnn.reshape(out, ttnn.Shape((1, 1, B, NH * HD)), ttnn.Shape((1, 1, 32, NH * HD)))
        return out

    def reset_state(self):
        def z():
            return ttnn.from_torch(
                torch.zeros(self.B, 1, self.args.max_seq_len, self.HD, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.k_caches = [z() for _ in range(self.NKV)]
        self.v_caches = [z() for _ in range(self.NKV)]

    def forward_prefill(self, x, cos_tt, sin_tt):
        """Causal prefill. x [1,1,S,dim]: K-sharded (dim/tp per device) when the fused in-proj
        AG-matmul path is active (``_fuse_agmm`` and S>TILE — the norm skips its post-AG); replicated
        otherwise. Output reduce-scattered on dim=3."""
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        S = x.shape[-2]

        qg, kp, vp = self._qkv(x)

        q, gate_flat, k, v = self._make_heads(qg, kp, vp, S)

        q = ttnn.multiply(
            ttnn.rms_norm(q, epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG),
            tw["q_norm"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        k = ttnn.multiply(
            ttnn.rms_norm(k, epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG),
            tw["k_norm"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH, self.rope_dim)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV, self.rope_dim)

        # Fill per-head KV cache for decode (stateful path only)
        if self.k_caches is not None:
            # Don't deallocate slices — for NKV==1 they alias k/v used by SDPA
            for h in range(NKV):
                ttnn.fill_cache(self.k_caches[h], ttnn.slice(k, (0, h, 0, 0), (1, h + 1, S, HD)), 0)
                ttnn.fill_cache(self.v_caches[h], ttnn.slice(v, (0, h, 0, 0), (1, h + 1, S, HD)), 0)

        q8, k8, v8 = q, k, v
        padded = max(32, ((S + 31) // 32) * 32)
        # SDPA flash chunk: 128 for S>=2048, 64 below. (256 wins in ISOLATION at S=3072/4096
        # -- test_sdpa_prefill_opt -- but in the full model its larger CBs clash with the resident
        # attn-input L1 buffer during a single-pass prefill of S>2048 (prefill_tp/generate_tp;
        # program.cpp "circular buffers ... clash with L1 buffers"). Production serving chunks
        # prefill at <=2048, so this path never sees S>2048 and 256 has no reachable win.)
        ch = min(128 if S >= 2048 else 64, padded)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=ch, k_chunk_size=ch
        )
        attn = ttnn.transformer.scaled_dot_product_attention(
            q8, k8, v8, is_causal=True, scale=self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=sdpa_cfg
        )
        ttnn.deallocate(q8)
        ttnn.deallocate(k8)
        ttnn.deallocate(v8)

        # Concat heads first, then gate: concat col h*HD+d == gate_flat col h*HD+d, so this is
        # bit-identical to per-head gating but skips the gate reshape+transpose to head-major.
        attn = self._concat_heads(attn)
        # concat(attn)+sigmoid(gate) in L1; gated stays DRAM (feeds the wo matmul_reduce_scatter — an L1
        # CCL activation risks clashing with its CBs).
        gated = ttnn.multiply(
            attn, ttnn.sigmoid(gate_flat, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(gate_flat)
        partial = self._wo_proj(gated, tw["wo"])
        ttnn.deallocate(gated)
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _kv_shard_cfg(self, B):
        """Height shard for paged_update_cache (one user per core), sized to the ACTIVE width B.
        Returns the precomputed max-batch config unchanged when B==self.B (byte-identical prod path);
        builds a width-B config (B cores) for bucketed decode. Mirrors model_config.kv_update_shard_cfg."""
        if B == self.B:
            return self.args.kv_update_shard_cfg
        cfg = self._kv_shard_cfg_cache.get(B)
        if cfg is None:
            cols = next(c for c in range(min(8, B), 0, -1) if B % c == 0)
            cfg = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.HD),
                core_grid=ttnn.CoreGrid(x=cols, y=B // cols),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self._kv_shard_cfg_cache[B] = cfg
        return cfg

    def forward_decode(self, x, cur_pos_tt, cos_tt, sin_tt, page_table=None):
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        # Active decode width, taken from the input (x is [1,1,B,dim_frac]). Normally == self.B.
        # BUCKETED decode: a request feeds B<self.B users; every shape/reshape/rope/head-split and
        # the KV-update shard config below run at this width, and the paged SDPA reads only these B
        # users' pages via the width-B page_table. The B==self.B path is byte-identical to before.
        B = x.shape[-2]
        _L1 = ttnn.L1_MEMORY_CONFIG  # keep decode head-prep + attn output L1-resident
        use_paged = self.use_paged and page_table is not None
        if not use_paged and self.k_caches is None:
            self.reset_state()

        qg, kp, vp = self._qkv(x)

        gate_flat = None  # nlp head-split path: flat [1,1,B,NH*HD] gate, applied after concat-heads
        if self._use_nlp_decode_heads:
            q, gate_flat, k, v = self._make_heads_decode(qg, kp, vp, B)  # v stays HEIGHT_SHARDED (KV update)
            gate = None
        elif vp is None:
            # Fused [q|k|v|gate] weight (_qkv sentinel vp=None): qg is contiguous [q|k|v], kp is gate.
            # Slice q/k/v heads directly from qg; gate is the separate block.
            q = ttnn.reshape(
                ttnn.slice(qg, (0, 0, 0, 0), (1, 1, B, NH * HD), memory_config=_L1), (1, B, NH, HD), memory_config=_L1
            )
            k = ttnn.reshape(
                ttnn.slice(qg, (0, 0, 0, NH * HD), (1, 1, B, NH * HD + NKV * HD), memory_config=_L1),
                (1, B, NKV, HD),
                memory_config=_L1,
            )
            v = ttnn.reshape(
                ttnn.slice(qg, (0, 0, 0, NH * HD + NKV * HD), (1, 1, B, NH * HD + 2 * NKV * HD), memory_config=_L1),
                (1, B, NKV, HD),
                memory_config=_L1,
            )
            ttnn.deallocate(qg)
            gate = ttnn.reshape(kp, (1, B, NH, HD), memory_config=_L1)
            ttnn.deallocate(kp)
        else:
            qg_r = ttnn.reshape(qg, (1, B, NH, HD * 2), memory_config=_L1)
            ttnn.deallocate(qg)
            q = ttnn.slice(qg_r, (0, 0, 0, 0), (1, B, NH, HD), memory_config=_L1)
            gate = ttnn.slice(qg_r, (0, 0, 0, HD), (1, B, NH, HD * 2), memory_config=_L1)
            ttnn.deallocate(qg_r)
            k = ttnn.reshape(kp, (1, B, NKV, HD), memory_config=_L1)
            ttnn.deallocate(kp)
            v = ttnn.reshape(vp, (1, B, NKV, HD), memory_config=_L1)
            ttnn.deallocate(vp)

        # QK norm — (1+w), matching prefill/HF (the prior "flat" no-+1 decode band-aided the reshape scramble).
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6, memory_config=_L1), tw["q_norm"], memory_config=_L1)
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6, memory_config=_L1), tw["k_norm"], memory_config=_L1)

        q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B, self.rope_dim)
        k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B, self.rope_dim)

        # SDPA-decode grid: use the real device grid (11x10=110 cores on P150x4), not a
        # hardcoded 64. cores_per_head = grid_total/B (sdpa_decode_program_factory.cpp), so a
        # bigger grid gives each batch row more parallel cores for its KV-reduction. At SHORT
        # context (~4k) the reduction is shallow enough that fixed per-core overhead dominates
        # and this makes ~no difference (B=1: flat; B=8: ~3% worse, both within noise). At LONG
        # context (~64k) the reduction is deep enough that the extra cores are a real win:
        # SdpaDecodeDeviceOperation duration B=8: 1569.9us -> 1396.2us (-11%); B=1: 220.8us ->
        # 215.5us (-2.4%, no regression). Using the full grid unconditionally since it never hurts
        # and helps significantly at long context, where batched decode is otherwise slowest.
        _sdpa_grid = self.mesh.compute_with_storage_grid_size()
        # Hang-isolation knobs (2026-09-21, TP=8 first-decode wedge inside SdpaDecode on one device: readers blocked in
        # read_k, writers polling the tree-reduction child semaphore): QWEN36_SDPA_DEC_K_CHUNK=<tokens> replaces the
        # data-dependent DYNAMIC_CHUNK_SIZE (k_chunk_size=0) with a fixed chunk; QWEN36_SDPA_DEC_MAX_CORES_PER_HEAD=1
        # gives every (batch row, kv head) one core, i.e. no cross-core tree reduction at all. Both default to the
        # original config.
        _sdpa_dec_kwargs = {}
        if _SDPA_DEC_MAX_CORES_PER_HEAD:
            _sdpa_dec_kwargs["max_cores_per_head_batch"] = _SDPA_DEC_MAX_CORES_PER_HEAD
        sdpa_dec_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(_sdpa_grid.x, _sdpa_grid.y),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=_SDPA_DEC_K_CHUNK,
            **_sdpa_dec_kwargs,
        )
        if use_paged:
            # External paged KV: update at cur_pos, then paged SDPA-decode
            keys, values = self.paged_k, self.paged_v
            # k [1,B,NKV,HD] TILE already has padded shape [1,B,32,HD]; paged_update_cache validates on
            # padded shapes (batch = padded[1], shard {32,HD} == padded[-2:], B-core grid) and its writer
            # copies only num_heads (= cache dim 1 = NKV) rows of each user's shard
            # (writer_update_cache_interleaved_start_id.cpp), so the old ttnn.pad to [1,B,32,HD] — a FillPad
            # kernel zeroing rows the op never reads — is gone; a single i2s to the B-core update grid remains.
            _kv_cfg = self._kv_shard_cfg(B)
            k_sh = ttnn.to_memory_config(k, _kv_cfg)
            ttnn.deallocate(k)
            if v.memory_config().is_sharded():
                # nlp head-split path: v is still nlp_create_qkv_heads_decode's height-sharded output (B
                # cores row-major from (0,0), {32,HD} ROW_MAJOR shards, shard width == HD) — the exact layout
                # paged_update_cache validates — so it goes in directly (no s2i / pad / i2s round trip).
                v_sh = v
            else:
                v_sh = ttnn.to_memory_config(v, _kv_cfg)
                ttnn.deallocate(v)
            # paged_update_cache takes bf16/fp32 and casts to bf8 cache; decode K/V stay bf16 (prefill fill needs bf8)
            ttnn.experimental.paged_update_cache(keys, k_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
            ttnn.experimental.paged_update_cache(values, v_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
            ttnn.deallocate(k_sh)
            ttnn.deallocate(v_sh)
            attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                keys,
                values,
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                program_config=sdpa_dec_cfg,
                # Emit to L1: consumed by the L1 sigmoid-gate multiply next (output-only, doesn't
                # change the SDPA reduction), before the wo matmul + all-reduce re-materialize to DRAM.
                memory_config=_L1,
            )
            ttnn.deallocate(q)
        else:
            # Internal per-head KV caches (test/generate_tp oracle only). One head per update: the sliced
            # [1,B,1,HD] TILE tensor already has padded shape [1,B,32,HD], which is all paged_update_cache
            # needs (see the paged branch), so no ttnn.pad. NOTE: for NKV == 1 the slice is a no-op that
            # returns k / v themselves, and ttnn.pad on an already-tile-padded tensor is an in-place fill_pad
            # plus a view -- the old loop's deallocate(k_h/v_h) therefore freed the buffer the following
            # to_memory_config was still reading (a latent use-after-free that only showed once the L1
            # allocation pattern changed). Per-head temporaries are left to refcounting; k/v freed after the loop.
            if v.memory_config().is_sharded():
                v = ttnn.sharded_to_interleaved(v, _L1)  # oracle path slices v per head below
            _kv_cfg = self._kv_shard_cfg(B)
            for h in range(NKV):
                k_h = ttnn.slice(k, (0, 0, h, 0), (1, B, h + 1, HD))
                v_h = ttnn.slice(v, (0, 0, h, 0), (1, B, h + 1, HD))
                k_sh = ttnn.to_memory_config(k_h, _kv_cfg)
                v_sh = ttnn.to_memory_config(v_h, _kv_cfg)
                ttnn.experimental.paged_update_cache(self.k_caches[h], k_sh, update_idxs_tensor=cur_pos_tt)
                ttnn.experimental.paged_update_cache(self.v_caches[h], v_sh, update_idxs_tensor=cur_pos_tt)
                ttnn.deallocate(k_sh)
                ttnn.deallocate(v_sh)
            ttnn.deallocate(k)
            ttnn.deallocate(v)

            if NKV == 1:
                k_full, v_full = self.k_caches[0], self.v_caches[0]
            else:
                k_full = ttnn.concat(self.k_caches, dim=1)
                v_full = ttnn.concat(self.v_caches, dim=1)

            # Non-paged oracle path (test/generate_tp only): the full-cache SDPA-decode's static CBs
            # grow with max_seq_len and, unbounded (k_chunk_size=0), overrun into the persistent CCL
            # semaphore buffers at the top of L1. Bound the K-chunk to cap the CB footprint (the paged
            # production path reads bounded blocks, so it keeps the auto config).
            nonpaged_sdpa_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=128
            )
            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                k_full,
                v_full,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                program_config=nonpaged_sdpa_cfg,
                # Emit to L1: consumed by the L1 sigmoid-gate multiply next (output-only, doesn't
                # change the SDPA reduction), before the wo matmul + all-reduce re-materialize to DRAM.
                memory_config=_L1,
            )
            ttnn.deallocate(q)

        if self._use_nlp_decode_heads:
            # Concat heads first, then gate on the flat [1,1,B,NH*HD] (concat col h*HD+d == gate_flat col
            # h*HD+d): bit-identical to per-head gating and exactly what forward_prefill does; drops the
            # gate's head-major ReshapeView kernel. sigmoid + multiply run on B rows x NH*HD instead of
            # B x 32-padded heads x HD.
            attn_flat = self._concat_heads_decode(attn_out, B)  # consumes + deallocates attn_out
            gated = ttnn.multiply(attn_flat, ttnn.sigmoid(gate_flat, memory_config=_L1), memory_config=_L1)
            ttnn.deallocate(attn_flat)
            ttnn.deallocate(gate_flat)
            gated_flat = ttnn.reshape(gated, (1, B, NH * HD), memory_config=_L1)  # view (rank change only)
        else:
            gated = ttnn.multiply(attn_out, ttnn.sigmoid(gate, memory_config=_L1), memory_config=_L1)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(gate)
            gated_flat = ttnn.reshape(gated, (1, B, NH * HD))
            ttnn.deallocate(gated)
        wo_partial = self._wo_proj(gated_flat, tw["wo"])
        ttnn.deallocate(gated_flat)
        wo_partial = ttnn.reshape(wo_partial, (1, 1, B, wo_partial.shape[-1]))  # view (interleaved)
        # Decode: fused all_reduce_async -> replicated [1,1,B,dim] in the decode norm layout (tp_common.DecodeAllReduce).
        _ar = getattr(self.tt_ccl, "decode_all_reduce", None)
        if _ar is not None:
            return _ar(wo_partial)
        return tt_all_reduce(
            wo_partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------------------------------------------ #
    # Speculative-decoding VERIFY step (tt/verify_step.py). R = w*T rows, row s*T+j = user s, token offset j.
    # ------------------------------------------------------------------------------------------------ #
    def _attend_paged_decode(self, qkv3, gate, cur_pos_tt, cos_tt, sin_tt, page_table, B):
        """Head split -> QK norm -> partial RoPE -> paged_update_cache -> paged SDPA decode -> concat heads -> gate,
        for B users (one row each) -- the middle section of forward_decode (nlp head-split + paged branch), op for op.
        qkv3 [1,1,B,NH*HD+2*NKV*HD] L1, gate [1,1,B,NH*HD]; cur_pos_tt [B] int32; cos/sin [1,1,B,rope_dim];
        page_table [B, blocks]. Returns gated_flat [1,B,NH*HD] (L1). Consumes qkv3 and gate."""
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        _L1 = ttnn.L1_MEMORY_CONFIG
        q, gate_flat, k, v = self._make_heads_decode(qkv3, gate, None, B)
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6, memory_config=_L1), tw["q_norm"], memory_config=_L1)
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6, memory_config=_L1), tw["k_norm"], memory_config=_L1)
        q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B, self.rope_dim)
        k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B, self.rope_dim)
        _sdpa_grid = self.mesh.compute_with_storage_grid_size()
        _sdpa_dec_kwargs = {}
        if _SDPA_DEC_MAX_CORES_PER_HEAD:
            _sdpa_dec_kwargs["max_cores_per_head_batch"] = _SDPA_DEC_MAX_CORES_PER_HEAD
        sdpa_dec_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(_sdpa_grid.x, _sdpa_grid.y),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=_SDPA_DEC_K_CHUNK,
            **_sdpa_dec_kwargs,
        )
        keys, values = self.paged_k, self.paged_v
        _kv_cfg = self._kv_shard_cfg(B)
        k_sh = ttnn.to_memory_config(k, _kv_cfg)
        ttnn.deallocate(k)
        if v.memory_config().is_sharded():
            v_sh = v
        else:
            v_sh = ttnn.to_memory_config(v, _kv_cfg)
            ttnn.deallocate(v)
        # One row per REAL user per call: the T token offsets of a user are separate calls (verify_step loops j),
        # because paged_update_cache read-modify-writes the whole 32-row KV tile on one core per user -- rows of the
        # same tile written by different cores in ONE call race and lose updates (measured: 6-7 of 8 rows lost,
        # tests/test_verify_probe_scratch.py).
        ttnn.experimental.paged_update_cache(keys, k_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
        ttnn.experimental.paged_update_cache(values, v_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
        ttnn.deallocate(k_sh)
        ttnn.deallocate(v_sh)
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            keys,
            values,
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos_tt,
            scale=self.scale,
            program_config=sdpa_dec_cfg,
            memory_config=_L1,
        )
        ttnn.deallocate(q)
        attn_flat = self._concat_heads_decode(attn_out, B)
        gated = ttnn.multiply(attn_flat, ttnn.sigmoid(gate_flat, memory_config=_L1), memory_config=_L1)
        ttnn.deallocate(attn_flat)
        ttnn.deallocate(gate_flat)
        return ttnn.reshape(gated, (1, B, NH * HD), memory_config=_L1)

    def _verify_sdpa_cfg(self):
        _sdpa_grid = self.mesh.compute_with_storage_grid_size()
        kw = {}
        mc = _VERIFY_SDPA_MAX_CORES or _SDPA_DEC_MAX_CORES_PER_HEAD
        if mc:
            kw["max_cores_per_head_batch"] = mc
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(_sdpa_grid.x, _sdpa_grid.y),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=_SDPA_DEC_K_CHUNK,
            **kw,
        )

    def _attend_paged_verify_batched(self, qkv, plan, page_table):
        """BATCHED verify attention middle: ONE KV write and ONE SDPA per layer for all R = w*T rows (vs T per-offset
        passes of _attend_paged_decode). qkv [1,1,R,qkv3_dim+gate_dim] DRAM (row s*T+j = user s, token offset j).
        Returns gated [1,R,NH*HD] (L1). Consumes qkv.

        * Q: the R rows' [q|...] columns -> [1,R,NH,HD] (row-major round trip, exact) -> QK-norm -> partial RoPE with
          the per-ROW cos/sin (plan.cos_rows/sin_rows, row (s,j) = position P_s+j). The per-row ops are the decode
          ops on a different row count, bit-identical per row (tests/test_verify_attn_batched_scratch.py).
        * K/V: [1,1,R,HD] -> K-norm + RoPE (prefill-shaped rope: one head, R rows) -> 0/1 spread matmul (exact) to
          [1,w,32,HD] with row j of user s's shard = token j -> ONE paged_update_cache(num_tokens=T) per cache: core s
          read-modify-writes the tile row(s) holding positions P_s..P_s+T-1 (bit-exact with T single-row calls; the
          T-row RMW is what removes the same-tile race of T 'virtual users' in one call).
        * SDPA: paged decode SDPA with B' = R VIRTUAL USERS (row (s,j) is a user at cur_pos P_s+j reading user s's page
          table row) -> [1,R,NH,HD]; one call when R <= the core count, else ceil(R/110) calls over row chunks. Row
          (s,j) sees exactly the K/V rows <= P_s+j (rows j' > j of the same user are masked by its own cur_pos). At a
          single K chunk (context < 256 with the dynamic chunk) the result is bit-identical to the per-offset call at
          B=w; with more chunks the core count per user differs (110/R vs 110/w) and so does the flash accumulation
          order across chunks: PCC >= 0.9998 vs the per-offset path (both ~0.9998 vs an fp32 host reference).
        * concat heads (row-major round trip) -> sigmoid gate."""
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        R, T, w = plan.R, plan.T, plan.w
        assert NKV == 1, "batched verify attention assumes one local KV head (rows j of a user's shard = token j)"
        _L1 = ttnn.L1_MEMORY_CONFIG
        qkv3_dim = NH * HD + 2 * NKV * HD
        gate_dim = NH * HD
        q_flat = ttnn.slice(qkv, (0, 0, 0, 0), (1, 1, R, NH * HD), memory_config=_L1)
        k_flat = ttnn.slice(qkv, (0, 0, 0, NH * HD), (1, 1, R, NH * HD + HD), memory_config=_L1)
        v_flat = ttnn.slice(qkv, (0, 0, 0, NH * HD + HD), (1, 1, R, NH * HD + 2 * HD), memory_config=_L1)
        gate = ttnn.slice(qkv, (0, 0, 0, qkv3_dim), (1, 1, R, qkv3_dim + gate_dim), memory_config=_L1)
        ttnn.deallocate(qkv)
        # --- Q heads: [1,1,R,NH*HD] -> [1,R,NH,HD] (exact row-major round trip) -> norm -> rope (per-row cos/sin)
        q_rm = ttnn.to_layout(q_flat, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(q_flat)
        q_rm4 = ttnn.reshape(q_rm, (1, R, NH, HD))
        q = ttnn.to_layout(q_rm4, ttnn.TILE_LAYOUT)
        ttnn.deallocate(q_rm4)
        if q_rm4 is not q_rm:
            ttnn.deallocate(q_rm)
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6, memory_config=_L1), tw["q_norm"], memory_config=_L1)
        q = apply_partial_rope_decode(q, plan.cos_rows, plan.sin_rows, NH, R, self.rope_dim)  # DRAM [1,R,NH,HD]
        # --- K: norm -> rope (one head, R rows) -> spread to (user, token) rows -> one T-row KV update
        k = ttnn.multiply(ttnn.rms_norm(k_flat, epsilon=1e-6, memory_config=_L1), tw["k_norm"], memory_config=_L1)
        ttnn.deallocate(k_flat)
        k = apply_partial_rope_prefill(k, plan.cos_rows, plan.sin_rows, 1, self.rope_dim)  # L1 [1,1,R,HD]
        keys, values = self.paged_k, self.paged_v
        _kv_cfg = self._kv_shard_cfg(w)
        pos0 = plan.cur_pos[0]  # [w] = P_s (token offset 0)
        for cache, src in ((keys, k), (values, v_flat)):
            sp = ttnn.matmul(plan.spread, src, compute_kernel_config=plan.exact_mm, memory_config=_L1)  # [1,1,w*32,HD]
            ttnn.deallocate(src)
            sp4 = ttnn.reshape(sp, (1, w, ttnn.TILE_SIZE, HD))
            sh = ttnn.to_memory_config(sp4, _kv_cfg)
            ttnn.deallocate(sp4)
            if sp4 is not sp:
                ttnn.deallocate(sp)
            ttnn.experimental.paged_update_cache(
                cache, sh, update_idxs_tensor=pos0, page_table=page_table, num_tokens=T
            )
            ttnn.deallocate(sh)
        # --- SDPA over B' = R virtual users (row chunks of <= 110 users)
        cfg = self._verify_sdpa_cfg()
        outs = []
        for a, b, pt_c, pos_c in plan.sdpa_chunks:
            q_c = q if (a == 0 and b == R) else ttnn.slice(q, (0, a, 0, 0), (1, b, NH, HD))
            outs.append(
                ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q_c,
                    keys,
                    values,
                    page_table_tensor=pt_c,
                    cur_pos_tensor=pos_c,
                    scale=self.scale,
                    program_config=cfg,
                    memory_config=_L1,
                )
            )
            if q_c is not q:
                ttnn.deallocate(q_c)
        ttnn.deallocate(q)
        if len(outs) == 1:
            attn = outs[0]
        else:
            attn = ttnn.concat(outs, dim=1, memory_config=_L1)
            for o in outs:
                ttnn.deallocate(o)
        # --- concat heads: [1,R,NH(pad 32),HD] -> [1,1,R,NH*HD] (exact row-major round trip) -> gate
        a_rm = ttnn.to_layout(attn, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(attn)
        a_rm2 = ttnn.reshape(a_rm, (1, 1, R, NH * HD))
        attn_flat = ttnn.to_layout(a_rm2, ttnn.TILE_LAYOUT, memory_config=_L1)
        ttnn.deallocate(a_rm2)
        if a_rm2 is not a_rm:
            ttnn.deallocate(a_rm)
        gated = ttnn.multiply(attn_flat, ttnn.sigmoid(gate, memory_config=_L1), memory_config=_L1)
        ttnn.deallocate(attn_flat)
        ttnn.deallocate(gate)
        return ttnn.reshape(gated, (1, R, NH * HD), memory_config=_L1)

    def forward_verify(self, x, plan, cur_pos_list, cos_list, sin_list, page_table):
        """Verify step of one attention layer over the R = w*T row grid.

        x: replicated norm output [1,1,R,dim] (L1 width-sharded). plan: verify_step.VerifyPlan (row grid + the
        R-row matmul configs + the 0/1 gather/scatter constants). cur_pos_list[j] [w] int32 = P_s + j; cos/sin_list[j]
        [1,1,w,rope_dim]; page_table [w, blocks] (the REAL per-user rows, never duplicated).

        In-projection once at R rows; then the attention middle in plan.attn_mode:
          * "batched" (default when the multi-token paged_update_cache is built): ONE T-row KV write per cache and ONE
            virtual-user SDPA for all R rows (_attend_paged_verify_batched).
          * "offsets" (reference / fallback): per token offset j, gather the w rows s*T+j (0/1 matmul, exact), run the
            decode attention middle for w users at their own cur_pos (row j's K/V lands at P_s+j before its SDPA; rows
            j' < j of the same user were written by the earlier iterations, so causality within the block holds and
            rejected rows' K/V are simply overwritten next step); scatter the gated rows back.
        Out-projection once at R rows, then the all-reduce: the fused decode all_reduce_async when x is the fused-AR
        residual (R <= 32), else the reduce-scatter (fractured [1,1,R,dim/TP])."""
        tw, NH, HD = self.tw, self.NH, self.HD
        R, T, w = plan.R, plan.T, plan.w
        assert x.shape[-2] == R, (x.shape, R)
        assert (
            self.use_paged and self._fused_qkv and self._use_nlp_decode_heads
        ), "verify needs the paged fused-QKV path"
        _L1 = ttnn.L1_MEMORY_CONFIG
        qkv3_dim = NH * HD + 2 * self.NKV * HD
        gate_dim = NH * HD
        # in-projection at R rows (the decode 1D config at <= 32 rows, the item-J small-M 1D config above)
        qkv = tpc.matmul_1d_decode(
            x, tw["wqkv_fused"], plan.attn_qkv_progcfg, self.compute_cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        mode = plan.attn_mode if plan.debug_attn_mode is None else plan.debug_attn_mode
        if mode == "batched":
            gated = self._attend_paged_verify_batched(qkv, plan, page_table)  # consumes qkv; [1,R,NH*HD] L1
        else:
            acc = None
            # plan.debug_attn_offsets (timing only): run the per-offset middle for the first n offsets, so the traced
            # per-offset cost is (t[T] - t[1]) / (T - 1); the un-run rows of the output stay zero.
            n_off = T if plan.debug_attn_offsets is None else plan.debug_attn_offsets
            for j in range(n_off):
                qkv_j = plan.gather(j, qkv)  # [1,1,w,W] L1
                sh = list(qkv_j.shape)
                qkv3_j = ttnn.slice(qkv_j, (0, 0, 0, 0), (sh[0], sh[1], sh[2], qkv3_dim), memory_config=_L1)
                gate_j = ttnn.slice(qkv_j, (0, 0, 0, qkv3_dim), (sh[0], sh[1], sh[2], qkv3_dim + gate_dim))
                ttnn.deallocate(qkv_j)
                gated_j = self._attend_paged_decode(
                    qkv3_j, gate_j, cur_pos_list[j], cos_list[j], sin_list[j], page_table, w
                )
                acc = plan.scatter_accumulate(j, gated_j, acc)  # consumes gated_j
            ttnn.deallocate(qkv)
            gated = ttnn.reshape(acc, (1, R, NH * HD))
        gated = plan.keep_rows(gated, "attn")  # pad_safe plans: padding users' rows -> exact zeros (no-op otherwise)
        if plan.fused_ar:
            wo_partial = self._wo_proj(gated, tw["wo"])  # the decode 1D path (L1 out for the fused all-reduce)
        else:
            wo_partial = tpc.matmul_1d_decode(
                gated, tw["wo"], plan.attn_wo_progcfg, self.compute_cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        ttnn.deallocate(gated)
        wo_partial = ttnn.reshape(wo_partial, (1, 1, R, wo_partial.shape[-1]))
        return plan.all_reduce(wo_partial, self.tt_ccl, self.mesh, self.args)

    def forward_prefill_paged(
        self,
        x,
        cos_tt,
        sin_tt,
        page_table,
        chunk_page_table=None,
        chunk_start_idx=0,
        chunk_start_idx_tensor=None,
        user_id=0,
    ):
        """Paged-KV prefill for one chunk: fill cache + chunked SDPA over prior chunks.

        x is K-sharded when the fused in-proj path is active (same contract as ``forward_prefill``).
        chunk_start_idx_tensor: optional device offset for FLEXIBLE chunked SDPA (one program
        per trace/bucket). chunk_start_idx (int) still sizes the page table host-side.
        """
        assert self.use_paged and self.paged_k is not None, "forward_prefill_paged requires a bound paged KV cache"
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        if chunk_start_idx is None:
            chunk_start_idx = 0
        S = x.shape[-2]

        qg, kp, vp = self._qkv(x)

        q, gate_flat, k, v = self._make_heads(qg, kp, vp, S)

        q = ttnn.multiply(
            ttnn.rms_norm(q, epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG),
            tw["q_norm"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        k = ttnn.multiply(
            ttnn.rms_norm(k, epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG),
            tw["k_norm"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH, self.rope_dim)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV, self.rope_dim)

        # bf8 SDPA: paged_fill_cache doesn't cast — cast K/V to cache dtype before fill
        if self._sdpa_bf8:
            _k8 = ttnn.typecast(k, ttnn.bfloat8_b)
            ttnn.deallocate(k)
            k = _k8
            _v8 = ttnn.typecast(v, ttnn.bfloat8_b)
            ttnn.deallocate(v)
            v = _v8

        # Fill this chunk into the paged cache
        k_paged, v_paged = self.paged_k, self.paged_v
        block_size = k_paged.shape[2]
        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        page_len = fill_page_table.shape[1] * block_size
        if page_len < S:
            k_fill = ttnn.slice(k, (0, 0, 0, 0), (1, NKV, page_len, HD))
            v_fill = ttnn.slice(v, (0, 0, 0, 0), (1, NKV, page_len, HD))
        else:
            k_fill, v_fill = k, v
        ttnn.experimental.paged_fill_cache(k_paged, k_fill, fill_page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(v_paged, v_fill, fill_page_table, batch_idx=user_id)
        if page_len < S:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Chunked SDPA over paged cache; keep Q bf16 unless bf8 mode (QWEN_SDPA_BF8=1), which also
        # makes the KV cache bf8 -> full bf8 matmul
        if self._sdpa_bf8:
            q8 = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
            ttnn.deallocate(q)
        else:
            q8 = q

        # chunk_start_idx % q_chunk_size == 0; FLEXIBLE path uses one program per trace.
        # q/k_chunk=128 is valid (chunk_start always divisible by 2048) and faster than 64/256.
        if chunk_start_idx_tensor is not None:
            qk_chunk = 128
        else:
            cap = 128 if S >= 2048 else 64  # 128 beats 256
            qk_chunk = cap if not chunk_start_idx else min(cap, chunk_start_idx & -chunk_start_idx)
        # K chunk may be larger than the Q chunk (QWEN36_SDPA_K_CHUNK, default = q chunk) for full 2048-token chunks.
        k_chunk = max(qk_chunk, self._sdpa_k_chunk) if S >= 2048 else qk_chunk
        # Full BH grid for SDPA perf (bit-identical to 8×8; see test_tp_chunked_prefill_pcc_sweep)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            exp_approx_mode=False,
            q_chunk_size=qk_chunk,
            k_chunk_size=k_chunk,
        )

        # Pad page table to cover Q+offset and satisfy stick-size % 32 (extra blocks masked by causality)
        sdpa_page_table = page_table
        needed_blocks = (S + chunk_start_idx + block_size - 1) // block_size
        target_blocks = max(needed_blocks, page_table.shape[-1])
        target_blocks = ((target_blocks + 31) // 32) * 32
        if page_table.shape[-1] < target_blocks:
            zeros_pad = ttnn.zeros(
                (page_table.shape[0], target_blocks - page_table.shape[-1]),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            sdpa_page_table = ttnn.concat([page_table, zeros_pad], dim=-1)
            ttnn.deallocate(zeros_pad)

        if chunk_start_idx_tensor is not None:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q8,
                input_tensor_k=k_paged,
                input_tensor_v=v_paged,
                page_table_tensor=sdpa_page_table,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                compute_kernel_config=self._sdpa_compute_cfg,
                program_config=sdpa_cfg,
            )
        else:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q8,
                input_tensor_k=k_paged,
                input_tensor_v=v_paged,
                page_table_tensor=sdpa_page_table,
                chunk_start_idx=chunk_start_idx,
                compute_kernel_config=self._sdpa_compute_cfg,
                program_config=sdpa_cfg,
            )
        if sdpa_page_table is not page_table:
            ttnn.deallocate(sdpa_page_table)
        ttnn.deallocate(q8)

        # Concat heads first, then gate (flat gate matches concat column order); see forward_prefill.
        attn = self._concat_heads(attn)
        # concat(attn)+sigmoid(gate) in L1; gated stays DRAM (feeds the wo matmul_reduce_scatter — an L1
        # CCL activation risks clashing with its CBs).
        gated = ttnn.multiply(
            attn, ttnn.sigmoid(gate_flat, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(gate_flat)
        partial = self._wo_proj(gated, tw["wo"])
        ttnn.deallocate(gated)
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
