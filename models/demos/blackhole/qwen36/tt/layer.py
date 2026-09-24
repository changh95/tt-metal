# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
from models.demos.blackhole.qwen36.tt.gdn import GDNConfig, Qwen36GatedDeltaNet
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.utils.substate import substate
from models.tt_transformers.tt.common import Mode


class Qwen36DecoderLayer:
    """Single transformer layer with hybrid attention dispatch.

    Pattern: x → attention_norm → attention → residual → ff_norm → MLP → residual
    Attention is either GatedAttention (full, with RoPE) or GatedDeltaNet (linear).
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, tensor_cache_path=None, tt_ccl=None):
        self.layer_num = layer_num
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1)
        self.is_full_attention = args.is_full_attention_layer(layer_num)

        prefix = f"layers.{layer_num}"

        # Zero-centered RMSNorm (Qwen3.5): output = x_normed * (1 + weight). The
        # framework RMSNorm applies the +1 internally via add_unit_offset=True and
        # is mesh-aware (replicates the weight across a MeshDevice).
        #
        # Single device: plain RMSNorm on the full hidden state (validated path).
        # TP (27B on a (1,4) mesh): the residual stream is fractured along the
        # hidden dim, so each norm is wrapped in the framework DistributedNorm,
        # which all-gathers (PREFILL: distributed rmsnorm + gather; DECODE:
        # gather-then-norm) to hand the modules a replicated full-dim input —
        # exactly as models/demos/qwen35_27b does via the framework decoder.
        # Prefill fuses the norm all-gather into the in-proj matmul (all_gather_minimal_matmul_async):
        # GDN qkvzab and full-attn QKV. attention_norm then skips its post-norm AG (prefill only;
        # decode gathers pre-norm). Gates must match the module-side _fuse_agmm gates.
        self._fuse_norm_agmm = self.num_devices > 1 and (
            (not self.is_full_attention and getattr(args, "gdn_qkvz_weight_memcfg", None) is not None)
            or (self.is_full_attention and getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None)
        )
        self.attention_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "input_layernorm",
            tensor_cache_path,
            tt_ccl,
            "attention_norm",
            enable_all_gather=not self._fuse_norm_agmm,
        )
        # Prefill: ff_norm skips AG (fused into gate/up AGMM); decode gathers pre-norm so this is a no-op there.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
        self.ffn_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "post_attention_layernorm",
            tensor_cache_path,
            tt_ccl,
            "ff_norm",
            enable_all_gather=not self._fuse_ff_agmm,
        )

        if self.num_devices > 1:
            # Tensor-parallel modules (sharded weights from the raw substate).
            # Cache the sharded mesh weights to disk so re-runs skip the (slow,
            # single-threaded) reorder+shard of the full 27B.
            tp_cache = (tensor_cache_path / f"layers.{layer_num}" / "tp") if tensor_cache_path else None
            if self.is_full_attention:
                from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp

                tw = load_attention_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.self_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPAttention(mesh_device, args, tw, tt_ccl)
            else:
                from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp

                tw = load_gdn_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.linear_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
        elif self.is_full_attention:
            attn_state = substate(state_dict, f"layers.{layer_num}.self_attn")
            attn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedAttention(mesh_device, AttentionConfig.from_args(args), attn_state, attn_cache)
        else:
            gdn_state = substate(state_dict, f"layers.{layer_num}.linear_attn")
            gdn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedDeltaNet(mesh_device, GDNConfig.from_args(args), gdn_state, gdn_cache)

        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        self.feed_forward = Qwen36MLP(mesh_device, mlp_state, mlp_cache, args=args, tt_ccl=tt_ccl)

    def _make_norm(
        self,
        mesh_device,
        args,
        state_dict,
        layer_num,
        weight_key,
        tensor_cache_path,
        tt_ccl,
        ag_key,
        enable_all_gather=True,
    ):
        """Build the per-layer RMSNorm; wrap in DistributedNorm when TP>1.

        On a single device this returns the same plain RMSNorm the validated 9B
        path used. The DistributedNorm wrapper (TP>1) mirrors tt_transformers
        decoder.py and handles the fractured->replicated transition.
        """
        norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key=weight_key,
            state_dict_prefix=f"layers.{layer_num}.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
            **(
                dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=tt_ccl)
                if self.num_devices > 1
                else {}
            ),
        )
        if self.num_devices > 1:
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            return DistributedNorm(
                norm, args, tt_ccl=tt_ccl, TG=args.is_galaxy, ag_config_key=ag_key, enable_all_gather=enable_all_gather
            )
        return norm

    def _verify_norm_blocks(self, norm, x, norm_cfg, plan):
        """Decode RMSNorm of a FRACTURED [1,1,R,dim/TP] residual at R > 32 rows: one all-gather (DRAM interleaved), then
        the decode sharded norm (block_h 1) on every 32-row block, re-assembled along the rows (L1 interleaved).

        Why per block: the sharded LayerNorm kernel with block_h = R/32 > 1 is ROW-POSITION dependent on the decode attn
        grid (rows of the 2nd+ block differ from the same rows in block 0, which equals the block_h 1 result;
        tests/test_verify_rowpos_probe_scratch.py), so a user's logits would depend on its grid row. Per block it is the
        exact decode norm. Costs R/32 x (slice, i2s, norm, s2i) + 1 concat per norm."""
        R = plan.R
        args = self.args
        ag_key = norm.ag_config_key
        mc = args.model_config.get(ag_key) if ag_key else None  # the decode all-gather tuning DistributedNorm uses
        g = ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=mc["num_links"] if mc else self.tt_ccl.get_num_links(1),
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=mc["chunks_per_sync"] if mc else 10,
            num_workers_per_link=mc["num_workers_per_link"] if mc else 2,
            num_buffers_per_channel=2,
        )
        shard_cfg = norm_cfg["sharded_output_config"]
        blocks = []
        for b in range(R // ttnn.TILE_SIZE):
            blk = ttnn.slice(g, (0, 0, b * ttnn.TILE_SIZE, 0), (1, 1, (b + 1) * ttnn.TILE_SIZE, g.shape[-1]))
            blk_sh = ttnn.to_memory_config(blk, shard_cfg)
            ttnn.deallocate(blk)
            y = norm.norm(blk_sh, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=norm_cfg)
            ttnn.deallocate(blk_sh)
            y_il = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(y)
            blocks.append(y_il)
        ttnn.deallocate(g)
        out = ttnn.concat(blocks, dim=-2, memory_config=ttnn.L1_MEMORY_CONFIG)
        for y_il in blocks:
            ttnn.deallocate(y_il)
        return out

    def forward_verify(self, x, plan, cur_pos_list=None, cos_list=None, sin_list=None, page_table=None, accept_tt=None):
        """Speculative-decoding verify step of this layer over the R = w*T row grid (tt/verify_step.py). TP only.

        x is the residual stream at R rows: on the fused decode all-reduce path (R <= 32) the REPLICATED L1
        width-sharded [1,1,R,dim] the decode step uses, otherwise the FRACTURED DRAM [1,1,R,dim/TP] of the
        reduce-scatter path. Norms always run the decode config (block_h 1): directly on the replicated residual, or
        (fractured) after one all-gather per 32-row block via _verify_norm_blocks. The sub-layer forwards are the
        modules' forward_verify."""
        assert self.num_devices > 1, "verify step is TP only"
        replicated = x.shape[-1] == self.args.dim
        norm_cfg = self.args.get_norm_config("attn", Mode.DECODE)
        if replicated:
            attn_input = self.attention_norm.norm(
                x, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=norm_cfg
            )
        elif plan.R <= ttnn.TILE_SIZE:
            attn_input = self.attention_norm(x, mode=Mode.DECODE, norm_config=norm_cfg)  # the non-fused decode path
        else:
            attn_input = self._verify_norm_blocks(self.attention_norm, x, norm_cfg, plan)
        if self.is_full_attention:
            attn_output = self.attention.forward_verify(attn_input, plan, cur_pos_list, cos_list, sin_list, page_table)
        else:
            attn_output = self.attention.forward_verify(attn_input, plan, accept_tt, plan.gdn_qkv_prev[self.layer_num])
        ttnn.deallocate(attn_input)
        h = ttnn.add(x, attn_output)
        ttnn.deallocate(attn_output)
        if replicated:
            ff_input = self.ffn_norm.norm(h, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=norm_cfg)
        elif plan.R <= ttnn.TILE_SIZE:
            ff_input = self.ffn_norm(h, mode=Mode.DECODE, norm_config=norm_cfg)
        else:
            ff_input = self._verify_norm_blocks(self.ffn_norm, h, norm_cfg, plan)
        ff_output = self.feed_forward.forward_verify(ff_input, plan)
        ttnn.deallocate(ff_input)
        output = ttnn.add(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)
        return output

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,  # = GDN long_prefill_chunk_size; the only size the chunk-seq prefill kernel supports
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
        gdn_collect=False,
        gdn_masks=None,
    ):
        # gdn_masks: persistent device (mask_f32, mask_q, conv_sel) for the traced masked-bucket
        # prefill; only the TP GDN prefill branch consumes it (None => unchanged everywhere).
        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        if self.num_devices > 1:
            # TP: DistributedNorm uses the framework's per-norm memory configs.
            _attn_norm_config = self.args.get_norm_config("attn", _norm_mode)
            # PREFILL: distributed rmsnorm outputs in L1 so the fused in-proj AGMM gathers from L1, not DRAM.
            if _norm_mode == Mode.PREFILL:
                _attn_norm_config = {**_attn_norm_config, "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG}
            # DECODE ff_norm uses the attn_norm layout (act_shard_hidden, 32-core) so Qwen36MLP's input reshard is a no-op and the norm runs on 32 cores not 8; PREFILL keeps the framework ff config.
            if _norm_mode == Mode.DECODE:
                _ff_norm_config = self.args.get_norm_config("attn", _norm_mode)
            else:
                # ff_norm output stays DRAM: L1 keeps the full-width norm resident across the whole MLP,
                # clashing with each matmul's CBs (w1/w3/w2) for no gain. Verified dead end; keep DRAM.
                _ff_norm_config = self.args.get_norm_config("ff", _norm_mode)
        else:
            # In decode the norm output stays in L1 (as the old rms_norm_ttnn(memory_config=L1) did);
            # in prefill the framework RMSNorm returns interleaved DRAM (matches the old None default).
            _attn_norm_config = _ff_norm_config = (
                {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None
            )
        # DECODE fused all-reduce path (model._decode_residual_in): x is the REPLICATED residual [1,1,B,dim], L1
        # width-sharded in the decode norm layout -> the norms run directly on it (the wrapped sharded RMSNorm, no
        # DistributedNorm all-gather) and the residual adds stay in that layout. Fractured x keeps the old path.
        _replicated = self.num_devices > 1 and _norm_mode == Mode.DECODE and x.shape[-1] == self.args.dim
        if _replicated:
            attn_input = self.attention_norm.norm(
                x, mode=_norm_mode, in_sharded=True, out_sharded=True, norm_config=_attn_norm_config
            )
        else:
            attn_input = self.attention_norm(x, mode=_norm_mode, norm_config=_attn_norm_config)

        if self.num_devices > 1:
            # TP modules: input is the gathered (full-dim) norm output [1,1,B/S,dim];
            # output is fractured along dim=3. cos/sin are in rope_tp format.
            if self.is_full_attention:
                if mode == "prefill":
                    # Contract/vLLM path supplies a page_table → paged KV prefill; the
                    # demo path (no page_table) uses the internal concat caches.
                    if page_table is not None:
                        attn_output = self.attention.forward_prefill_paged(
                            attn_input,
                            cos,
                            sin,
                            page_table,
                            chunk_page_table=chunk_page_table,
                            chunk_start_idx=chunk_start_idx if chunk_start_idx is not None else 0,
                            chunk_start_idx_tensor=chunk_start_idx_tensor,
                        )
                    else:
                        attn_output = self.attention.forward_prefill(attn_input, cos, sin)
                else:
                    attn_output = self.attention.forward_decode(
                        attn_input, position_tensor, cos, sin, page_table=page_table
                    )
            else:
                # GDN carries its recurrent/conv state internally (capture_state on
                # prefill, read on decode); it has no paged KV, so page_table is N/A.
                if mode == "prefill":
                    if gdn_collect:
                        # Batched per-user prefill: stash this user's from-scratch state for
                        # assembly into row u of the batched buffers (finalize_pending later).
                        attn_output = self.attention.forward_prefill_collect(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len
                        )
                    else:
                        attn_output = self.attention.forward_prefill(
                            attn_input,
                            chunk_size=chunk_size,
                            valid_len=valid_len,
                            capture_state=True,
                            gdn_masks=gdn_masks,
                        )
                else:
                    attn_output = self.attention.forward_decode(attn_input)
        elif self.is_full_attention:
            attn_output = self.attention.forward(
                attn_input,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(
                attn_input, mode=deltanet_mode, chunk_size=chunk_size, valid_len=valid_len
            )
        ttnn.deallocate(attn_input)

        h = ttnn.add(x, attn_output)
        ttnn.deallocate(attn_output)

        if _replicated:
            ff_input = self.ffn_norm.norm(
                h, mode=_norm_mode, in_sharded=True, out_sharded=True, norm_config=_ff_norm_config
            )
        else:
            ff_input = self.ffn_norm(h, mode=_norm_mode, norm_config=_ff_norm_config)

        ff_output = self.feed_forward.forward(ff_input)
        ttnn.deallocate(ff_input)

        output = ttnn.add(h, ff_output)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        return output
