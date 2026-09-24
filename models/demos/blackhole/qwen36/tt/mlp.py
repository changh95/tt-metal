# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP: down(silu(gate(x)) * up(x)).

9B (single device): dense matmuls, full weights.
27B TP (1,4 mesh): w1/w3 column-parallel, w2 row-parallel; tt_all_reduce
reduce-scatters on meshes with a dim-1 shape (e.g. P150x4), fracturing hidden.
"""
import os
from dataclasses import dataclass

import ttnn


@dataclass(frozen=True)
class MLPWeights:
    w1: ttnn.Tensor  # gate_proj [in, out], bfloat4_b
    w2: ttnn.Tensor  # down_proj [in, out], bfloat8_b
    w3: ttnn.Tensor  # up_proj [in, out], bfloat4_b
    w_gate_up: ttnn.Tensor = None  # TP prefill: tile-pair-interleaved packed [gate|up] for fused-swiglu AGMM
    w2_ds: ttnn.Tensor = None  # TP decode: DRAM WIDTH_SHARDED copy of down_proj for the multi-reader decode matmul
    w1_ds: ttnn.Tensor = None  # TP decode: DRAM WIDTH_SHARDED copy of gate_proj (3 readers/bank), bfloat4_b
    w3_ds: ttnn.Tensor = None  # TP decode: DRAM WIDTH_SHARDED copy of up_proj (3 readers/bank), bfloat4_b


def _build_gate_up(gate_w, up_w, mesh, tp, cache_path):
    """Packed [gate|up] weight for all_gather_swiglu_prefill: prepare_for_fused_swiglu tile-pair
    interleave, then column-parallel shard on the 2N dim so each device holds its interleaved slice."""
    import torch

    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    gk = gate_w.to(torch.bfloat16).T.contiguous()  # [K=dim, N=hidden]
    uk = up_w.to(torch.bfloat16).T.contiguous()
    packed = torch.cat([gk, uk], dim=-1)  # [dim, 2*hidden], gate first
    il = prepare_for_fused_swiglu(packed, ndev=tp, gate_is_first=True)  # [dim, 2*hidden]
    return ttnn.as_tensor(
        il,
        dtype=ttnn.bfloat4_b,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def load_mlp_weights(mesh_device, state_dict, tensor_cache_path=None, args=None) -> MLPWeights:
    """Per-layer MLP state: gate_proj, down_proj, up_proj weights."""
    tp = getattr(args, "num_devices", 1) if args is not None else 1

    if tp > 1:
        # TP: w1/w3 column-parallel (shard out dim), w2 row-parallel (shard in dim).
        # DRAM-sharded memcfgs from args.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        # w1/w3 DRAM-WIDTH_SHARDED for decode (M=1 tile, ~+10% tok/s); w2 interleaved.
        # Cache uses `.dramshard` suffix — layout incompatible with interleaved cache
        # (as_tensor ignores requested memcfg on reload). Fallback if memcfgs absent.
        # 1D-decode (default) uses interleaved weights (its mcast decode matmul needs them).
        dram_sharded = (
            args is not None
            and getattr(args, "mlp_w1_weight_memcfg", None) is not None
            and not getattr(args, "mlp_1d_decode", False)
        )

        def cache(name, tag=""):
            return str(tensor_cache_path / f"mlp.{name}.weight{tag}.tp") if tensor_cache_path else None

        # Prefill-only packed [gate|up] AGMM weight (decode keeps w1/w3; extra DRAM ~w1+w3/layer).
        wgu = (
            _build_gate_up(
                state_dict["gate_proj.weight"],
                state_dict["up_proj.weight"],
                mesh_device,
                tp,
                cache("gate_up", ".swiglu"),
            )
            if tpc.mlp_gateup_agmm_enabled(tp)
            else None
        )

        # Decode-only DRAM WIDTH_SHARDED copy of down_proj for the DRAM-sharded decode matmul (opt-in,
        # QWEN36_DECODE_DRAM_SHARDED=1 -> model_config.mlp_w2_ds_decode). Prefill keeps the interleaved w2 (2D kernel). The cache suffix carries
        # the reader count because the per-bank storage pad is baked into the cached tensor.
        w2_ds = (
            tpc.shard_w(
                state_dict["down_proj.weight"],
                mesh_device,
                dim=0,
                memory_config=args.mlp_w2_ds_memcfg,
                cache_path=cache("down_proj", f".ds{args.mlp_w2_ds_workers}"),
                dtype=ttnn.bfloat8_b,
            )
            if args is not None and getattr(args, "mlp_w2_ds_decode", False)
            else None
        )
        # Decode-only DRAM WIDTH_SHARDED gate/up copies for the 3-readers-per-bank kernel (opt-in,
        # QWEN36_DECODE_DRAM_SHARDED=gateup -> model_config.mlp_w13_ds_decode). The interleaved w1/w3 stay for the
        # small-M prefill 1D matmuls (+2 x 12.5 MB/layer/device). Cache suffix = reader count (per-bank pad baked in).
        _w13 = args is not None and getattr(args, "mlp_w13_ds_decode", False)
        w1_ds = (
            tpc.shard_w(
                state_dict["gate_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=args.mlp_w1_ds_memcfg,
                cache_path=cache("gate_proj", f".ds{args.mlp_w13_ds_workers}"),
                dtype=ttnn.bfloat4_b,
            )
            if _w13
            else None
        )
        w3_ds = (
            tpc.shard_w(
                state_dict["up_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=args.mlp_w3_ds_memcfg,
                cache_path=cache("up_proj", f".ds{args.mlp_w13_ds_workers}"),
                dtype=ttnn.bfloat4_b,
            )
            if _w13
            else None
        )

        if dram_sharded:
            return MLPWeights(
                w1=tpc.shard_w(
                    state_dict["gate_proj.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=args.mlp_w1_weight_memcfg,
                    cache_path=cache("gate_proj", ".dramshard"),
                    dtype=ttnn.bfloat4_b,
                ),
                w3=tpc.shard_w(
                    state_dict["up_proj.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=args.mlp_w3_weight_memcfg,
                    cache_path=cache("up_proj", ".dramshard"),
                    dtype=ttnn.bfloat4_b,
                ),
                w2=tpc.shard_w(
                    state_dict["down_proj.weight"],
                    mesh_device,
                    dim=0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_path=cache("down_proj"),
                    dtype=ttnn.bfloat8_b,
                ),
                w_gate_up=wgu,
            )

        # Default: INTERLEAVED DRAM shards; ttnn.linear works for decode and prefill.
        return MLPWeights(
            w1=tpc.shard_w(
                state_dict["gate_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("gate_proj"),
                dtype=ttnn.bfloat4_b,
            ),
            w3=tpc.shard_w(
                state_dict["up_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("up_proj"),
                dtype=ttnn.bfloat4_b,
            ),
            w2=tpc.shard_w(
                state_dict["down_proj.weight"],
                mesh_device,
                dim=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("down_proj"),
                dtype=ttnn.bfloat8_b,
            ),
            w_gate_up=wgu,
            w2_ds=w2_ds,
            w1_ds=w1_ds,
            w3_ds=w3_ds,
        )

    def load(name, dtype):
        t = state_dict[f"{name}.weight"].T.contiguous()  # [in, out] for ttnn.linear
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"mlp.{name}.weight") if tensor_cache_path else None,
        )

    # gate/up: bfloat4_b (bandwidth); down: bfloat8_b (accuracy).
    return MLPWeights(
        w1=load("gate_proj", ttnn.bfloat4_b),
        w2=load("down_proj", ttnn.bfloat8_b),
        w3=load("up_proj", ttnn.bfloat4_b),
    )


class Qwen36MLP:
    """SwiGLU feed-forward network for Qwen3.5."""

    def __init__(self, mesh_device, state_dict, tensor_cache_path=None, args=None, tt_ccl=None):
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1) if args is not None else 1
        # 1D-decode (default): small-grid 1D matmuls beat the ~80-core DRAM-sharded grid on the
        # bandwidth-bound skinny decode MLP matmuls (see test_mlp_matmul_sweep). Forces interleaved weights.
        self._mlp_1d_decode = args is not None and getattr(args, "mlp_1d_decode", False)
        # DRAM-sharded decode DOWN matmul (opt-in; gate/up stay 1D). See model_config.mlp_w2_ds_decode.
        self._mlp_w2_ds_decode = args is not None and getattr(args, "mlp_w2_ds_decode", False)
        # DRAM-sharded 3-readers-per-bank decode GATE/UP matmuls (opt-in). See model_config.mlp_w13_ds_decode.
        self._mlp_w13_ds_decode = args is not None and getattr(args, "mlp_w13_ds_decode", False)
        # Match load_mlp_weights dram_sharded condition for layout consistency.
        self._dram_sharded = (
            self.num_devices > 1
            and args is not None
            and getattr(args, "mlp_w1_weight_memcfg", None) is not None
            and not self._mlp_1d_decode
        )
        # Prefill fused-swiglu AGMM (ff_norm skips its AG; layer.py sets _fuse_ff_agmm to match).
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._fuse_gateup_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
        self.weights = load_mlp_weights(mesh_device, state_dict, tensor_cache_path, args=args)
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        # fuse_swiglu AGMM: fp32 acc (subblock_w=4) to match GDN/attn in-proj.
        self.compute_kernel_config_agmm = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def forward(self, x):
        if self.num_devices > 1:
            return self._forward_tp(x)
        w = self.weights
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config
        mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
        w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
        w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
        hidden = ttnn.mul(w1_out, w3_out, memory_config=mc)
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        down_pc = None
        if (
            T > 1
            and getattr(self.args, "prefill_progcfg", None) is not None
            and os.environ.get("QWEN9B_MLP_DOWN_AUTO") != "1"
        ):
            down_pc = self.args.prefill_progcfg(T, hidden.shape[-1], w.w2.shape[-1])
        output = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc, program_config=down_pc)
        ttnn.deallocate(hidden)
        return output

    def _forward_tp(self, x):
        """TP forward: replicated input; reduce-scatter output fractured on hidden dim."""
        from models.demos.blackhole.qwen36.tt import tp_common as tpc
        from models.tt_transformers.tt.ccl import tt_all_reduce

        w = self.weights
        args = self.args
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        mc = ttnn.DRAM_MEMORY_CONFIG
        _silu_fused = False
        _ds_gateup = False
        # Prefill: x is K-sharded (ff_norm skipped AG); fused AG + [gate|up] + SwiGLU
        _fused_gu = self._fuse_gateup_agmm and x.shape[-2] > ttnn.TILE_SIZE and w.w_gate_up is not None
        # Small-M prefill (rows <= QWEN36_PREFILL_SMALLM_MAX): plain all-gather + 1D mcast w1 (silu fused) and w3 on
        # the interleaved decode weights, then the mul (tp_common "Small-M prefill matmuls": 304 -> 168 us at 128 rows,
        # 304 -> 270 at 256). Replaces the fused-swiglu AGMM, whose M padding streams the 25 MB bfp4 weight per M row.
        _small_m = self._fuse_gateup_agmm and tpc.small_m_rows(x.shape[-2])
        if _small_m:
            _gw = getattr(args, "decode_grid_w", 8)
            xg = tpc.all_gather_prefill_small_m(x, self.tt_ccl, args.ccl_topology())
            w1_out = tpc.small_m_linear(
                xg,
                w.w1,
                ckc,
                fused_activation=ttnn.UnaryOpType.SILU,
                out_memory_config=ttnn.L1_MEMORY_CONFIG,
                grid_w=_gw,
            )
            w3_out = tpc.small_m_linear(xg, w.w3, ckc, out_memory_config=ttnn.L1_MEMORY_CONFIG, grid_w=_gw)
            ttnn.deallocate(xg)
            _fused_gu = False
            _silu_fused = True
        elif _fused_gu:
            hidden = tpc.all_gather_swiglu_prefill(
                x, w.w_gate_up, self.tt_ccl, self.compute_kernel_config_agmm, args.ccl_topology()
            )
            _silu_fused = True
        elif getattr(self, "_dram_sharded", False) and x.shape[-2] <= ttnn.TILE_SIZE:
            # DRAM-WIDTH_SHARDED w1/w3 decode (M=1 tile). Prefill uses fused AGMM above.
            x_sh = ttnn.to_memory_config(x, args.act_shard_hidden)
            w1_out = ttnn.linear(
                x_sh,
                w.w1,
                compute_kernel_config=ckc,
                program_config=args.mlp_w1_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_sh,
                w.w3,
                compute_kernel_config=ckc,
                program_config=args.mlp_w3_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_sh)
            # Keep gate/up in L1 for mul → w2 (avoid L1→DRAM→L1).
            w1_out = ttnn.to_memory_config(w1_out, ttnn.L1_MEMORY_CONFIG)
            w3_out = ttnn.to_memory_config(w3_out, ttnn.L1_MEMORY_CONFIG)
        elif self._mlp_w13_ds_decode and x.shape[-2] <= ttnn.TILE_SIZE and w.w1_ds is not None:
            # DRAM-sharded gate/up with 3 readers per DRAM bank (QWEN36_DECODE_DRAM_SHARDED=gateup). The ff-norm
            # width shard (32 cores x 5 tiles) is in0 as-is (in0_block_w 5), so x is not interleaved as on the 1D
            # path. Outputs: L1 width shards on ceil(136 / 4) = 34 cores (the factory needs every one of the 24
            # reader cores to own an output shard, hence per_core_N 4). silu is fused in the w1 progcfg.
            x_sh = x if x.is_sharded() else ttnn.to_memory_config(x, args.act_shard_hidden)
            w1_out = ttnn.linear(
                x_sh,
                w.w1_ds,
                compute_kernel_config=ckc,
                program_config=args.mlp_w1_ds_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_sh,
                w.w3_ds,
                compute_kernel_config=ckc,
                program_config=args.mlp_w3_ds_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            if x_sh is not x:
                ttnn.deallocate(x_sh)
            _silu_fused = True
            _ds_gateup = True
        elif self._mlp_1d_decode and x.shape[-2] <= ttnn.TILE_SIZE:
            # 1D mcast decode matmuls on a small explicit grid, silu fused in the w1 progcfg.
            # mcast_in0 needs interleaved in0, but ff-norm hands us a width-shard -> interleave first.
            x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            w1_out = ttnn.linear(
                x_il,
                w.w1,
                compute_kernel_config=ckc,
                program_config=args.mlp_w1_decode_1d_progcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_il,
                w.w3,
                compute_kernel_config=ckc,
                program_config=args.mlp_w3_decode_1d_progcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_il)
            _silu_fused = True
        elif x.shape[-2] > ttnn.TILE_SIZE:
            # Prefill (M>1 tile, compute-bound): FPU-tuned 2D config (grid width -> 1x4 subblock,
            # in0_block_w=4) beats ttnn-auto's 1x1 stall ~2.7x (test_mlp_matmul_sweep_prefill). SILU fused.
            seq = x.shape[-2]
            # max_cols = device worker-grid width (11 on BH): wide grid (gate/up -> 9x10) vs old 8-wide.
            _gw = getattr(args, "decode_grid_w", 8)
            # TP-selected prefill tuning; absent (single-device 9B) => frozen TP=4 behavior.
            _pt = getattr(args, "prefill_tuning", None)
            pc_gate = tpc.create_prefill_mlp_matmul_program_config(
                seq, args.dim, w.w1.shape[-1], fused_activation=ttnn.UnaryOpType.SILU, max_cols=_gw, tuning=_pt
            )
            pc_up = tpc.create_prefill_mlp_matmul_program_config(
                seq, args.dim, w.w3.shape[-1], max_cols=_gw, tuning=_pt
            )
            # L1 output (gate/up outputs; down output via mc_out below): +FPU, avoids the DRAM round-trip
            # (test_mlp_matmul_sweep_prefill *_outL1). The [seq,N] tensors fit L1 at the prefill chunk.
            w1_out = ttnn.linear(
                x, w.w1, compute_kernel_config=ckc, program_config=pc_gate, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            w3_out = ttnn.linear(
                x, w.w3, compute_kernel_config=ckc, program_config=pc_up, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            _silu_fused = True
        else:
            # Interleaved weights: auto matmul program for decode and prefill.
            w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
            w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
            _silu_fused = True

        # gated activation (down-proj INPUT): L1 in decode, DRAM in prefill. The L1 win is OUTPUT-only;
        # keeping both down input (hidden) and output (partial) in L1 at seq 2048 overflows L1.
        _prefill_tuned = x.shape[-2] > ttnn.TILE_SIZE and _silu_fused
        # gate * up (skipped when _fused_gu already produced `hidden` with SwiGLU in-kernel).
        if not _fused_gu:
            mc_out = ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= ttnn.TILE_SIZE else mc
            # Standalone silu only on DRAM-sharded decode path (SILU not fused there).
            if _silu_fused:
                # DS gate/up: two identical 34-core width shards -> shard-local mul, re-laid out below for down.
                hidden = ttnn.mul(w1_out, w3_out, memory_config=w1_out.memory_config() if _ds_gateup else mc_out)
                ttnn.deallocate(w1_out)
            else:
                w1_act = ttnn.silu(w1_out, memory_config=mc_out)
                ttnn.deallocate(w1_out)
                hidden = ttnn.mul(w1_act, w3_out, memory_config=mc_out)
                ttnn.deallocate(w1_act)
            ttnn.deallocate(w3_out)
            if _ds_gateup:
                # Down-proj input layout: the 17 x 8-tile shard the DS down kernel reads (one L1->L1 reshard) or L1
                # interleaved for the 1D kernel (sharded_to_interleaved). The 1D gate/up path interleaved x twice.
                _tgt = (
                    args.act_shard_mlp_hidden
                    if (self._mlp_w2_ds_decode and w.w2_ds is not None)
                    else ttnn.L1_MEMORY_CONFIG
                )
                _h = ttnn.to_memory_config(hidden, _tgt)
                ttnn.deallocate(hidden)
                hidden = _h
        # Prefill w2: 2D progcfg on (8,10); decode (M<=32) keeps ttnn-auto.
        w2_pc = None
        if self._mlp_1d_decode and hidden.shape[-2] <= ttnn.TILE_SIZE:
            # 1D mcast decode down-proj on a small explicit grid (~16 cores).
            w2_pc = args.mlp_w2_decode_1d_progcfg
        elif self.num_devices > 1 and tpc.small_m_rows(hidden.shape[-2]):
            # Small-M prefill down-proj: 1D mcast config (the 2D config lights only M_tiles grid rows at these row
            # counts: 123 -> 75 us at 128 rows, 130 -> 97 at 256).
            w2_pc = tpc.small_m_progcfg(
                hidden.shape[-2], hidden.shape[-1], w.w2.shape[-1], grid_w=getattr(args, "decode_grid_w", 8)
            )
        elif hidden.shape[-2] > ttnn.TILE_SIZE:
            # Prefill down-proj: subblock-tuned 2D config with the wide grid (max_cols=device width),
            # off the generic 8-wide prefill_progcfg. Output L1 via mc_w2_out below.
            w2_pc = tpc.create_prefill_mlp_matmul_program_config(
                hidden.shape[-2],
                hidden.shape[-1],
                w.w2.shape[-1],
                max_cols=getattr(args, "decode_grid_w", 8),
                tuning=getattr(args, "prefill_tuning", None),
            )
        # down-proj OUTPUT in L1 for the tuned prefill path (DRAM input `hidden` + L1 output = the
        # validated sweep outL1 config; tt_all_reduce already consumes an L1 partial).
        mc_w2_out = ttnn.L1_MEMORY_CONFIG if (x.shape[-2] <= ttnn.TILE_SIZE or _prefill_tuned) else mc
        if self._mlp_w2_ds_decode and hidden.shape[-2] <= ttnn.TILE_SIZE and w.w2_ds is not None:
            # Decode down-proj on the DRAM-sharded kernel (8 bank readers, ~430 GB/s vs ~360 on the 1D grid):
            # reshard the L1 gate*up product to 17 cores x 8 tiles, matmul against the width-sharded w2 copy,
            # re-lay the width-sharded result out to L1 interleaved for the reduce-scatter.
            partial = tpc.matmul_dram_sharded_decode(
                hidden, w.w2_ds, args.mlp_w2_ds_progcfg, ckc, args.act_shard_mlp_hidden, out_memory_config=mc_w2_out
            )
        else:
            partial = ttnn.linear(
                hidden, w.w2, compute_kernel_config=ckc, memory_config=mc_w2_out, program_config=w2_pc
            )
        ttnn.deallocate(hidden)

        # Decode: fused all_reduce_async -> replicated [1,1,B,dim] in the decode norm layout (tp_common.DecodeAllReduce).
        # Only the decode path feeds the MLP a width-sharded norm output (layer.py: ffn_norm(..., out_sharded=True) on
        # the replicated residual); an eager TP prefill of <= 32 tokens has the same row count but a FRACTURED,
        # interleaved residual, and must keep the reduce-scatter path (the fused op would return the replicated width).
        _ar = getattr(self.tt_ccl, "decode_all_reduce", None)
        if _ar is not None and x.shape[-2] <= ttnn.TILE_SIZE and x.is_sharded():
            return _ar(partial)

        # tt_all_reduce on (1,4) mesh reduce-scatters to hidden dim (dim=3).
        out = tt_all_reduce(
            partial,
            self.device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out
