# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device side of the speculative-decoding VERIFY step (milestone M2): a traced forward over the w x T row grid.

Row grid, accept rule and commit: tt/verify_grid.py (host). This module owns the per-(w, T) ``VerifyPlan`` (R-row
matmul / norm configs, the 0/1 gather-scatter constants, the persistent per-step input buffers, the per-GDN-layer
``qkv_prev`` buffers and the stub's state scratch) and ``VerifyStep`` (compile, trace capture, per-step upload ->
replay -> per-row argmax readback).

Body (per row r = s*T + j, token x_j of user s at position P_s + j):
  embedding at R rows -> for every layer ``layer.forward_verify`` -> final norm -> vocab-sharded LM head at R rows ->
  per-device (max, argmax) over the local vocab shard -> tiny [TP, R] readback, host select (never a full logits gather).
  * R <= 32 on the fused decode all-reduce path: the residual is the REPLICATED L1 width-sharded [1,1,R,dim] of the
    decode step and every sub-layer runs the decode-step ops (the 1D decode matmul configs, the fused all_reduce_async),
    so a row's numerics are those of the width-R decode step.
  * R > 32: the residual is FRACTURED [1,1,R,dim/TP] (reduce-scatter path); norms go through DistributedNorm with
    block_h = R/32 configs; the projections run the item-J small-M 1D mcast configs at M = R (same in0_block_w as the
    decode 1D configs); attention and GDN split the grid by token offset j (see the modules).
Trace safety (masked_bucket_trace.py rules): every buffer the trace reads/writes is allocated in VerifyPlan.__init__
BEFORE any capture, per-step values are DMA'd in with copy_host_to_device_tensor, shapes never depend on data (the
accept counts are tensors), ttnn.reshape views are never deallocated.
"""
import functools
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt import verify_grid as vg
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode

_EXACT_MM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
)


def gdn_multi_token_kernel_available():
    """True when ttnn.experimental.kda.gdn_decode_step exposes the multi-token contract (num_tokens / qkv_prev /
    accept). QWEN36_VERIFY_GDN_STUB=1 forces the per-token stub even when it does."""
    if os.environ.get("QWEN36_VERIFY_GDN_STUB", "0") == "1":
        return False
    doc = getattr(ttnn.experimental.kda.gdn_decode_step, "__doc__", None) or ""
    return all(k in doc for k in ("num_tokens", "qkv_prev", "accept"))


def attn_multi_token_update_available():
    """True when ttnn.experimental.paged_update_cache exposes num_tokens (the T-row KV write of the batched verify
    attention middle, attention/tp.py _attend_paged_verify_batched)."""
    doc = getattr(ttnn.experimental.paged_update_cache, "__doc__", None) or ""
    return "num_tokens" in doc


def gdn_multi_token_kernel(T, layer, qkv_cur, qkv_prev, accept_tt):
    """The multi-token GDN verify kernel call (interface contract with the kernel being built concurrently):
    qkv_cur [1,R,W] bf16 TILE (this step's [q|k|v|z|a|b] rows), qkv_prev [1,R,W] (the previous step's, zeros at the
    first step), accept [w] int32 (previous step's a_s). Commits prev rows 1..a_s + cur row 0 to the per-user state and
    packed history in place, evaluates cur rows 1..k read-only; returns gated [1,R,Nv*Dv] (padding rows 0)."""
    return ttnn.experimental.kda.gdn_decode_step(
        qkv_cur,
        layer.tw["dt_bias"],
        layer.tw["neg_exp_A"],
        layer.rec_state,
        layer._norm_weight_1d(),
        layer.Nv,
        layer.Nk,
        layer.Dk,
        layer.Dv,
        conv_hist=layer.conv_hist_packed,
        num_tokens=T,
        qkv_prev=qkv_prev,
        accept=accept_tt,
        **layer._verify_kernel_args(),
    )


def argmax_sharded_rows(logits):
    """Per-device (argmax, max) of vocab-sharded logits [1,1,R,V/TP] over the local shard -> ([1,1,R] uint32, [1,1,R]
    bf16) device tensors. Stage 1 of the two-stage argmax; stage 2 is combine_sharded_argmax on the host."""
    rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(rm, dim=-1, keepdim=False)
    ttnn.deallocate(rm)
    val = ttnn.max(logits, dim=-1)
    return idx, val


def combine_sharded_argmax(mesh, idx_t, val_t, R, per_shard):
    """Read the [TP, R] (idx, max) pairs and pick each row's winning shard (first max on ties, like torch.argmax);
    global id = shard * per_shard + local idx. Returns [R] int64."""
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    nd = mesh.get_num_devices()
    idxs = ttnn.to_torch(idx_t, mesh_composer=comp).reshape(nd, -1)[:, :R].to(torch.int64)
    vals = ttnn.to_torch(val_t, mesh_composer=comp).float().reshape(nd, -1)[:, :R]
    d = torch.argmax(vals, dim=0)
    return d * per_shard + idxs[d, torch.arange(R)]


class VerifyPlan:
    """Everything a (w, T) verify step needs on device, allocated before any trace capture."""

    def __init__(self, model, w, T, page_table, use_kernel=None):
        self.model = model
        mesh = model.mesh_device
        args = model.args
        self.w, self.T = int(w), int(T)
        self.R = vg.grid_rows(self.w, self.T)
        assert (
            self.w * self.T == self.R
        ), f"w*T = {self.w * self.T} must fill the tile-padded grid R = {self.R} (no padding rows yet)"
        self.bmax = args.max_batch_size
        assert self.w <= self.bmax, f"w = {self.w} users exceed the model's max_batch_size {self.bmax}"
        self.dim = args.dim
        self.fused_ar = self.R <= tpc.TILE_SIZE and model._decode_fused_all_reduce()
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self._rep = rep
        gw = getattr(args, "decode_grid_w", 8)
        R = self.R

        # --- R-row matmul configs: the decode 1D configs at <= 32 rows, the item-J small-M 1D configs above ---
        if R <= tpc.TILE_SIZE:
            self.attn_qkv_progcfg = args.attn_qkv_decode_1d_progcfg
            self.attn_wo_progcfg = args.attn_wo_decode_1d_progcfg
            self.gdn_qkvz_progcfg = args.gdn_qkvz_decode_1d_progcfg
            self.gdn_out_progcfg = args.gdn_out_decode_1d_progcfg
            self.mlp_w1_progcfg = args.mlp_w1_decode_1d_progcfg
            self.mlp_w3_progcfg = args.mlp_w3_decode_1d_progcfg
            self.mlp_w2_progcfg = args.mlp_w2_decode_1d_progcfg
            self.norm_attn = None
            self.norm_lm = None
        else:
            hid = args.hidden_dim // args.num_devices
            gdn0 = next(l.attention for l in model.layers if not l.is_full_attention)
            attn0 = next(l.attention for l in model.layers if l.is_full_attention)
            self.attn_qkv_progcfg = tpc.small_m_progcfg(R, self.dim, attn0.tw["wqkv_fused"].shape[-1], grid_w=gw)
            self.attn_wo_progcfg = tpc.small_m_progcfg(R, args.attn_out_dim_tp, self.dim, grid_w=gw)
            self.gdn_qkvz_progcfg = tpc.small_m_progcfg(R, self.dim, gdn0.tw["qkvz"].shape[-1], grid_w=gw)
            self.gdn_out_progcfg = tpc.small_m_progcfg(R, args.gdn_value_dim_tp, self.dim, grid_w=gw)
            self.mlp_w1_progcfg = tpc.small_m_progcfg(
                R, self.dim, hid, fused_activation=ttnn.UnaryOpType.SILU, grid_w=gw
            )
            self.mlp_w3_progcfg = tpc.small_m_progcfg(R, self.dim, hid, grid_w=gw)
            self.mlp_w2_progcfg = tpc.small_m_progcfg(R, hid, self.dim, grid_w=gw)
            # norms run the decode (block_h 1) configs per 32-row block (layer._verify_norm_blocks): the sharded
            # LayerNorm kernel at block_h > 1 is row-position dependent on the attn grid (test_verify_rowpos_probe_scratch).
            self.norm_attn = None
            self.norm_lm = None

        # --- 0/1 gather / scatter constants ---
        sel_h, selT_h = vg.select_matrices(self.w, self.T, R)
        self.sel = [self._up(t, ttnn.bfloat16, ttnn.TILE_LAYOUT) for t in sel_h]
        self.selT = [self._up(t, ttnn.bfloat16, ttnn.TILE_LAYOUT) for t in selT_h]

        # --- per-step inputs (values refreshed by upload()) ---
        rd = args.rope_head_dim
        self.tokens = self._up(torch.zeros(1, R, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.cur_pos = [
            self._up(torch.zeros(self.w, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for _ in range(T)
        ]
        cos0, sin0 = vg.rope_cos_sin(torch.zeros(self.w, dtype=torch.int32), rd, args.rope_theta)
        self.cos = [self._up(cos0, ttnn.bfloat16, ttnn.TILE_LAYOUT) for _ in range(T)]
        self.sin = [self._up(sin0, ttnn.bfloat16, ttnn.TILE_LAYOUT) for _ in range(T)]
        pt = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        assert pt.shape[0] == self.w, f"page_table must have one row per user ({self.w}), got {tuple(pt.shape)}"
        assert pt.shape[1] % 8 == 0, "page-table stick must be a multiple of 32 bytes = 8 blocks (paged SDPA decode)"
        self.page_table = self._up(pt.to(torch.int32).contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.accept = self._up(torch.zeros(self.w, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        # --- GDN: multi-token kernel or the per-token stub, per-layer qkv_prev, shared state scratch ---
        if use_kernel is None:
            use_kernel = gdn_multi_token_kernel_available()
        self.gdn_kernel = functools.partial(gdn_multi_token_kernel, self.T) if use_kernel else None
        self.gdn_qkv_prev = {}
        for layer in model.layers:
            if not layer.is_full_attention:
                # materialize the lazily-built decode constants NOW (before any trace capture, trace-hazard rule)
                layer.attention._conv_taps_packed_t()
                layer.attention._norm_weight_1d()
                W = layer.attention.tw["qkvz"].shape[-1]
                self.gdn_qkv_prev[layer.layer_num] = self._up(
                    torch.zeros(1, R, W, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
                )
        self.mask_f32 = self.mask_bf16 = None
        if self.gdn_kernel is None:
            # one-hot accept masks per token offset j: [bmax,1,1,1] each, broadcast over the [bmax,Nv,Dk,Dv] state
            m0 = vg.accept_onehot_masks([0] * self.w, self.w, self.T, self.bmax)
            self.mask_f32 = [self._up(m0[:, j : j + 1].contiguous(), ttnn.float32, ttnn.TILE_LAYOUT) for j in range(T)]
            # the packed history is rank 5 ([Bmax, Nv, 4, 32, 32]) -> its mask is [Bmax,1,1,1,1]
            self.mask_bf16 = [
                self._up(m0[:, j : j + 1].reshape(self.bmax, 1, 1, 1, 1).contiguous(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                for j in range(T)
            ]
            self._ensure_gdn_scratch()
        # --- attention middle: "batched" (one T-row KV write + one virtual-user SDPA per layer) or "offsets" (T
        # per-offset decode passes, the reference). QWEN36_VERIFY_ATTN overrides; batched needs the multi-token
        # paged_update_cache build and one local KV head.
        attn0 = next((l.attention for l in model.layers if l.is_full_attention), None)
        batched_ok = attn_multi_token_update_available() and attn0 is not None and attn0.NKV == 1
        self.attn_mode = os.environ.get("QWEN36_VERIFY_ATTN", "batched" if batched_ok else "offsets")
        assert self.attn_mode in ("batched", "offsets"), self.attn_mode
        if self.attn_mode == "batched":
            assert batched_ok, "QWEN36_VERIFY_ATTN=batched needs paged_update_cache(num_tokens=) and NKV == 1"
        self.debug_attn_mode = None  # timing only: force a mode for one sub-trace
        self.exact_mm = _EXACT_MM
        # 0/1 spread [w*32, R]: row s*32+j <- row s*T+j (the KV update input: user s's shard row j = token j)
        spread = torch.zeros(1, 1, self.w * tpc.TILE_SIZE, R, dtype=torch.float32)
        for s_ in range(self.w):
            for j in range(T):
                spread[0, 0, s_ * tpc.TILE_SIZE + j, vg.row(s_, j, T)] = 1.0
        self.spread = self._up(spread, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # per-ROW rope cos/sin [1,1,R,rd] (row s*T+j = position P_s+j) and the virtual-user SDPA inputs: cur_pos
        # [R] (P_s+j) and page table [R, blocks] (row s*T+j = pt[s]), in row chunks of <= the core count
        self.cos_rows = self._up(torch.zeros(1, 1, R, rd, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.sin_rows = self._up(torch.zeros(1, 1, R, rd, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        grid = mesh.compute_with_storage_grid_size()
        n_cores = grid.x * grid.y
        n_split = int(os.environ.get("QWEN36_VERIFY_SDPA_SPLIT", "0")) or -(-R // n_cores)
        per = -(-R // n_split)
        pt_rows = pt.to(torch.int32).repeat_interleave(self.T, dim=0).contiguous()  # [R, blocks]
        self.sdpa_chunks = []  # (row a, row b, page_table [b-a, blocks], cur_pos [b-a])
        for c in range(n_split):
            a, b = c * per, min(R, (c + 1) * per)
            if a >= b:
                break
            pt_c = self._up(pt_rows[a:b].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            pos_c = self._up(torch.zeros(b - a, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self.sdpa_chunks.append((a, b, pt_c, pos_c))
        self.trace_id = None
        self.out_idx = self.out_val = None
        self._host_refs = []
        self.debug_attn_offsets = None  # timing only, see attention.forward_verify
        # persistent residual-shaped inputs of the per-section sub-traces (time_sections_traced); allocated NOW
        self.sec_x_frac = self._up(
            torch.zeros(1, 1, R, self.dim // args.num_devices, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
        )
        self.sec_x_rep = None
        if self.fused_ar:
            act = args.get_norm_config("attn", Mode.DECODE)["sharded_output_config"]
            self.sec_x_rep = ttnn.to_memory_config(
                self._up(torch.zeros(1, 1, R, self.dim, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT), act
            )
        logger.info(
            f"[verify] plan w={self.w} T={self.T} R={R} fused_ar={self.fused_ar} "
            f"gdn={'multi-token kernel' if self.gdn_kernel else 'per-token STUB'} attn={self.attn_mode} "
            f"(sdpa chunks {[(a, b) for a, b, _, _ in self.sdpa_chunks]})"
        )

    # ------------------------------------------------------------------------------------------ helpers
    def _up(self, t, dtype, layout):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.model.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._rep,
        )

    def _norm_config(self, grid, R):
        """Decode-style sharded RMSNorm config for R rows on `grid` (block_h = R/32; the decode configs have 1)."""
        block_w = self.dim // grid.num_cores // ttnn.TILE_SIZE
        subblock_w = next(s for s in (4, 3, 2, 1) if block_w % s == 0)
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=subblock_w,
            block_h=R // ttnn.TILE_SIZE,
            block_w=block_w,
            inplace=False,
        )
        mc = ttnn.create_sharded_memory_config(
            (R, self.dim // grid.num_cores),
            grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return {"sharded_program_config": pc, "sharded_output_config": mc, "output_mem_config": None}

    def _ensure_gdn_scratch(self):
        model = self.model
        if getattr(model, "_verify_gdn_scratch", None) is None:
            gdn0 = next(l.attention for l in model.layers if not l.is_full_attention)
            S, H = gdn0.rec_state, gdn0.conv_hist_packed
            assert S is not None and H is not None, "GDN state must exist (allocate_kv_caches) before a VerifyPlan"
            n_dev = model.mesh_device.get_num_devices()
            scr_S = ttnn.from_torch(
                torch.zeros(*S.shape, dtype=torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=model.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._rep,
            )
            hs = list(H.shape)
            scr_H = ttnn.from_torch(
                torch.zeros(n_dev * hs[0], *hs[1:], dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=model.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(model.mesh_device, dim=0),
            )
            model._verify_gdn_scratch = (scr_S, scr_H)

    def gdn_scratch(self, S, H):
        scr_S, scr_H = self.model._verify_gdn_scratch
        assert tuple(scr_S.shape) == tuple(S.shape) and tuple(scr_H.shape) == tuple(H.shape), (
            scr_S.shape,
            S.shape,
            scr_H.shape,
            H.shape,
        )
        return scr_S, scr_H

    def mask(self, j, dtype):
        return self.mask_f32[j] if dtype == ttnn.float32 else self.mask_bf16[j]

    def gather(self, j, X):
        """Rows s*T+j of X ([1,R,W] or [1,1,R,W]) for every user s -> [1,1,w,W] (L1); exact 0/1 matmul."""
        X4 = X if len(X.shape) == 4 else ttnn.reshape(X, (1, 1, X.shape[-2], X.shape[-1]))
        return ttnn.matmul(self.sel[j], X4, compute_kernel_config=_EXACT_MM, memory_config=ttnn.L1_MEMORY_CONFIG)

    def scatter_accumulate(self, j, Y, acc):
        """Scatter per-user rows Y ([1,w,N] or [1,1,w,N]) to rows s*T+j of a [1,1,R,N] accumulator (exact: each row
        receives one 1.0*y, the others 0.0). Consumes Y (and the previous acc)."""
        Y4 = Y if len(Y.shape) == 4 else ttnn.reshape(Y, (1, 1, Y.shape[-2], Y.shape[-1]))
        part = ttnn.matmul(self.selT[j], Y4, compute_kernel_config=_EXACT_MM, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(Y)
        if acc is None:
            return part
        out = ttnn.add(acc, part, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(acc)
        ttnn.deallocate(part)
        return out

    def all_reduce(self, partial, tt_ccl, mesh, args):
        """The sub-layer all-reduce: fused all_reduce_async (replicated, decode norm layout) on the fused-AR path,
        else the reduce-scatter (fractured [1,1,R,dim/TP] DRAM).

        Above one tile row the reduce-scatter runs in FP32: the bf16 reduce_scatter_minimal_async sums the TP partials
        in a different order per TILE ROW (deterministic; rows 32..63 of identical data differ from rows 0..31,
        tests/test_verify_rs_probe_scratch.py), so a user's logits would depend on its grid row. Summing bf16 partials
        in fp32 is order independent (tile rows equal in the probe); the result is cast back to bf16 for the residual.
        (The fused decode all_reduce_async is a pairwise fp32-accumulated reduction and matches neither bitwise.)"""
        if self.fused_ar:
            return tt_ccl.decode_all_reduce(partial)
        if self.R <= tpc.TILE_SIZE:
            return tt_all_reduce(
                partial,
                mesh,
                tt_ccl,
                cluster_axis=0,
                dim=3,
                topology=args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        p32 = ttnn.typecast(partial, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        out32 = tt_all_reduce(
            p32,
            mesh,
            tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.float32,
        )
        out = ttnn.typecast(out32, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out32)
        return out

    # ------------------------------------------------------------------------------------------ per-step upload
    def upload(self, tokens, positions, accept_prev):
        """DMA one step's values into the persistent inputs. tokens[s] = [x_0, d_1..d_k]; positions[s] = P_s;
        accept_prev[s] = the previous step's a_s. Host tensors are kept alive until the caller synchronizes."""
        w, T, R = self.w, self.T, self.R
        args = self.model.args
        refs = []

        def dma(host_t, dst, dtype, layout):
            h = ttnn.from_torch(host_t, dtype=dtype, layout=layout, device=None, mesh_mapper=self._rep)
            ttnn.copy_host_to_device_tensor(h, dst)
            refs.append(h)

        dma(vg.row_tokens(tokens, T, R).reshape(1, R), self.tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        rope_delta = int(getattr(self.model.rope, "rope_delta", 0) or 0)
        for j in range(T):
            pj = vg.offset_positions(positions, j)
            dma(pj, self.cur_pos[j], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            cos, sin = vg.rope_cos_sin(pj + rope_delta, args.rope_head_dim, args.rope_theta)
            dma(cos, self.cos[j], ttnn.bfloat16, ttnn.TILE_LAYOUT)
            dma(sin, self.sin[j], ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # batched attention middle: per-row positions / rope
        rows_pos = vg.row_positions(positions, T, R)
        cos_r, sin_r = vg.rope_cos_sin(rows_pos + rope_delta, args.rope_head_dim, args.rope_theta)
        dma(cos_r, self.cos_rows, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        dma(sin_r, self.sin_rows, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        for a, b, _, pos_c in self.sdpa_chunks:
            dma(rows_pos[a:b].contiguous(), pos_c, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        acc = torch.tensor([int(a) for a in accept_prev], dtype=torch.int32)
        assert acc.shape[0] == w and int(acc.min()) >= 0 and int(acc.max()) < T, acc
        dma(acc, self.accept, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        if self.gdn_kernel is None:
            m = vg.accept_onehot_masks(accept_prev, w, T, self.bmax)
            for j in range(T):
                dma(m[:, j : j + 1].contiguous(), self.mask_f32[j], ttnn.float32, ttnn.TILE_LAYOUT)
                dma(
                    m[:, j : j + 1].reshape(self.bmax, 1, 1, 1, 1).contiguous(),
                    self.mask_bf16[j],
                    ttnn.bfloat16,
                    ttnn.TILE_LAYOUT,
                )
        self._host_refs = refs

    def release(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.model.mesh_device, self.trace_id)
            self.trace_id = None


class VerifyStep:
    """One (w, T) verify step: forward body, trace capture/replay, argmax readback.

    Usage (host bookkeeping in verify_grid.VerifyController):
        vs = VerifyStep(model, w, T, page_table[w, blocks]); vs.compile(); vs.capture()   # before prefill of real users
        argmax_rows = vs.run(tokens [w][T], positions [w], accept_prev [w])            # -> [R] int64 per step
    """

    def __init__(self, model, w, T, page_table, use_kernel=None):
        self.model = model
        self.mesh = model.mesh_device
        self.plan = VerifyPlan(model, w, T, page_table, use_kernel=use_kernel)
        self.per_shard = model.args.vocab_size // model.num_devices
        self.section_times = None

    # ------------------------------------------------------------------------------------------ forward body
    def forward(self, profile=False, row_check=None):
        """The traced body: reads only the plan's persistent buffers; returns (idx, val) device tensors.
        row_check(name, tensor): eager-only diagnostic hook called after the embedding and every layer."""
        model, plan = self.model, self.plan
        R = plan.R
        t = {} if profile else None

        def tick(name, t0):
            if profile:
                ttnn.synchronize_device(self.mesh)
                t[name] = t.get(name, 0.0) + (time.perf_counter() - t0)
                return time.perf_counter()
            return t0

        t0 = time.perf_counter()
        x = model.embd(plan.tokens)
        x = ttnn.reshape(x, (1, 1, R, x.shape[-1]))
        if plan.fused_ar:
            x = model._decode_residual_in(x)
        else:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        t0 = tick("embed", t0)
        if row_check is not None:
            row_check("embed", x)
        for li, layer in enumerate(model.layers):
            if layer.is_full_attention:
                x_new = layer.forward_verify(
                    x, plan, cur_pos_list=plan.cur_pos, cos_list=plan.cos, sin_list=plan.sin, page_table=plan.page_table
                )
                t0 = tick("attn_layers", t0)
            else:
                x_new = layer.forward_verify(x, plan, accept_tt=plan.accept)
                t0 = tick("gdn_layers", t0)
            ttnn.deallocate(x)
            x = x_new
            if row_check is not None:
                row_check(f"layer{li}_{'attn' if layer.is_full_attention else 'gdn'}", x)
        if plan.fused_ar:
            x = model._final_norm_decode(x)
        elif R <= ttnn.TILE_SIZE:
            x = model._final_norm_decode(x)  # the non-fused decode path (DistributedNorm, lm_head config)
        else:
            nc = model.args.get_norm_config("lm_head", Mode.DECODE)
            x_n = model.layers[0]._verify_norm_blocks(model.norm, x, nc, plan)  # L1 interleaved [1,1,R,dim]
            ttnn.deallocate(x)
            x = ttnn.to_memory_config(x_n, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x_n)
        t0 = tick("final_norm", t0)
        logits = ttnn.linear(x, model.lm_head_weight)  # vocab-sharded [1,1,R,V/TP]
        ttnn.deallocate(x)
        t0 = tick("lm_head", t0)
        idx, val = argmax_sharded_rows(logits)
        ttnn.deallocate(logits)
        tick("argmax", t0)
        if profile:
            self.section_times = t
        return idx, val

    # ------------------------------------------------------------------------------------------ compile / trace
    def _dummy_inputs(self):
        w, T = self.plan.w, self.plan.T
        return [[1] * T for _ in range(w)], [T] * w, [0] * w

    def compile(self, profile=False):
        """Eager run (compiles every program; MUTATES the KV cache / GDN state of users 0..w-1 -> call before the
        real prefill). With profile=True records per-section eager times in self.section_times."""
        self.plan.upload(*self._dummy_inputs())
        idx, val = self.forward(profile=profile)
        ttnn.synchronize_device(self.mesh)
        self.plan._host_refs = []
        ttnn.deallocate(idx)
        ttnn.deallocate(val)

    def capture(self):
        """Capture the trace (the programs must be compiled: compile() first). Mutates state like compile()."""
        assert self.plan.trace_id is None, "already captured"
        self.plan.upload(*self._dummy_inputs())
        ttnn.synchronize_device(self.mesh)
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.plan.out_idx, self.plan.out_val = self.forward()
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh)
        self.plan._host_refs = []
        self.plan.trace_id = tid

    def release(self):
        self.plan.release()

    # ------------------------------------------------------------------------------------------ one step
    def run(self, tokens, positions, accept_prev, eager=False):
        """Upload -> replay (or eager forward) -> per-row argmax [R] int64."""
        plan = self.plan
        plan.upload(tokens, positions, accept_prev)
        if eager or plan.trace_id is None:
            idx, val = self.forward()
            ttnn.synchronize_device(self.mesh)
            out = combine_sharded_argmax(self.mesh, idx, val, plan.R, self.per_shard)
            ttnn.deallocate(idx)
            ttnn.deallocate(val)
        else:
            ttnn.execute_trace(self.mesh, plan.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
            out = combine_sharded_argmax(self.mesh, plan.out_idx, plan.out_val, plan.R, self.per_shard)
        plan._host_refs = []
        return out

    def _section_body(self, which):
        """One section of the body on the persistent residual-shaped input (values irrelevant for timing)."""
        model, plan = self.model, self.plan
        x = plan.sec_x_rep if plan.fused_ar else plan.sec_x_frac
        if plan.fused_ar:
            model.tt_ccl.decode_all_reduce.begin_step()
        if which == "head":
            R = plan.R
            if plan.fused_ar:
                y = model._final_norm_decode(x)
            elif R <= ttnn.TILE_SIZE:
                y = model._final_norm_decode(x)
            else:
                nc = model.args.get_norm_config("lm_head", Mode.DECODE)
                y_n = model.layers[0]._verify_norm_blocks(model.norm, x, nc, plan)
                y = ttnn.to_memory_config(y_n, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(y_n)
            logits = ttnn.linear(y, model.lm_head_weight)
            ttnn.deallocate(y)
            idx, val = argmax_sharded_rows(logits)
            ttnn.deallocate(logits)
            return [idx, val]
        if which == "embed":
            e = model.embd(plan.tokens)
            e = ttnn.reshape(e, (1, 1, plan.R, e.shape[-1]))
            e = model._decode_residual_in(e) if plan.fused_ar else ttnn.to_memory_config(e, ttnn.DRAM_MEMORY_CONFIG)
            return [e]
        layer = next(l for l in model.layers if l.is_full_attention == (which.startswith("attn")))
        if which.startswith("attn"):
            out = layer.forward_verify(
                x, plan, cur_pos_list=plan.cur_pos, cos_list=plan.cos, sin_list=plan.sin, page_table=plan.page_table
            )
        else:
            out = layer.forward_verify(x, plan, accept_tt=plan.accept)
        return [out]

    def time_sections_traced(self, n=50):
        """TRACED per-section times (ms, median of n replays) from sub-traces of the body run on persistent inputs:
        one attention layer in the plan's mode ("attn_T"), the same layer on the per-offset path with all T offsets
        ("attn_offsets_T") and with a single offset ("attn_1", ~ one decode attention pass at width w), the batched
        layer ("attn_batched", when built), one GDN layer, the embedding and the head (final norm + LM head + argmax).
        Each sub-trace is compiled eagerly first, captured, replayed n times, and released. Derived: per-offset
        attention cost = (attn_offsets_T - attn_1) / (T - 1); attn_layers_total = n_attn * attn_T."""
        plan = self.plan
        res = {}
        n_attn = sum(1 for l in self.model.layers if l.is_full_attention)
        n_gdn = len(self.model.layers) - n_attn
        sections = [("attn_offsets_T", "offsets", None), ("attn_1", "offsets", 1)]
        if plan.attn_mode == "batched" or attn_multi_token_update_available():
            sections.append(("attn_batched", "batched", None))
        sections += [("gdn", None, None), ("embed", None, None), ("head", None, None)]
        for which, mode, n_off in sections:
            plan.debug_attn_mode = mode
            plan.debug_attn_offsets = n_off
            self.plan.upload(*self._dummy_inputs())
            outs = self._section_body(which)  # compile
            ttnn.synchronize_device(self.mesh)
            for o in outs:
                ttnn.deallocate(o)
            tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            outs = self._section_body(which)
            ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
            ttnn.synchronize_device(self.mesh)
            ms = []
            for _ in range(n):
                t0 = time.perf_counter()
                ttnn.execute_trace(self.mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(self.mesh)
                ms.append(1e3 * (time.perf_counter() - t0))
            ttnn.release_trace(self.mesh, tid)
            self.plan._host_refs = []
            ms.sort()
            res[which] = ms[len(ms) // 2]
        plan.debug_attn_offsets = None
        plan.debug_attn_mode = None
        T = plan.T
        res["attn_mode"] = plan.attn_mode
        res["attn_T"] = res["attn_batched"] if plan.attn_mode == "batched" else res["attn_offsets_T"]
        per_off = (res["attn_offsets_T"] - res["attn_1"]) / max(1, T - 1)
        res["attn_layer_per_offset"] = per_off
        res["attn_layers_total"] = n_attn * res["attn_T"]
        res["attn_offset_loop_total"] = n_attn * T * per_off
        res["gdn_layers_total"] = n_gdn * res["gdn"]
        res["n_attn"], res["n_gdn"] = n_attn, n_gdn
        return res

    def time_replays(self, n=50):
        """Traced step time (upload + replay + sync) over n replays with refreshed inputs; returns (median, min) ms."""
        assert self.plan.trace_id is not None
        w, T = self.plan.w, self.plan.T
        ms = []
        for i in range(n):
            toks = [[(100 + i + s + j) % 1000 for j in range(T)] for s in range(w)]
            pos = [T * (i + 1) + s for s in range(w)]
            acc = [(i + s) % T for s in range(w)]
            t0 = time.perf_counter()
            self.plan.upload(toks, pos, acc)
            ttnn.execute_trace(self.mesh, self.plan.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.mesh)
            ms.append(1e3 * (time.perf_counter() - t0))
            self.plan._host_refs = []
        ms.sort()
        return ms[len(ms) // 2], ms[0]
