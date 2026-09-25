# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The checkpoint's native MTP head as the FIRST DRAFTER of speculative decoding (milestone M3), TP on the 1x4 mesh.

Reference semantics (vLLM qwen3_5_mtp.py + llm_base_proposer.py, HF weights mtp.*): the head is ONE full-attention
Qwen3.5 decoder layer (q/k norm, partial RoPE, GQA 24/4 heads x 256, sigmoid output gate, SwiGLU MLP) with its own KV
cache. Its input at position i is ``fc(concat(pre_fc_norm_embedding(embed(x_{i+1})), pre_fc_norm_hidden(h_i)))`` where
``h_i`` is the main model's POST-final-norm hidden state at position i (vLLM feeds the target model's returned
``hidden_states`` = ``Qwen3_5Model.norm(...)``, i.e. the LM-head input) and ``x_{i+1}`` the token the main model produced
there; the layer output goes through ``mtp.norm`` and the SHARED lm_head -> the draft distribution for ``x_{i+2}``.
Chaining (num_speculative_tokens > 1): the drafted token and the head's own post-``mtp.norm`` output are fed back at
position i+1 (positions += 1), same layer, same KV.

Device side (``MTPHead``):
  * weights: ``weight_mapping.load_qwen36_mtp_state_dict`` (mtp.layers.0.* -> layers.{L}.*, L = n_layers = a virtual
    17th attention layer index), the layer built by the SAME ``Qwen36DecoderLayer`` class as the main layers (TP=4
    sharding, tensor cache under layers.{L}/), fc split into its embedding / hidden halves (column shards for the
    prefill's fractured residual, replicated copies for the decode step's fused-all-reduce residual), the three norms
    through the framework RMSNorm (+1 offset), the paged KV as two more cache tensors of the main cache's shape (+1 pad
    block) bound with set_paged_kv_cache; RoPE from the model's tables.
  * prefill (``prefill_hook``, installed as ``model.prefill_hidden_hook``): after the main masked-bucket prefill of a
    request, the head runs ONE eager forward over the bucket with the tokens shifted by one (x_1..x_{n-1} at positions
    0..n-2) and the main post-norm hidden (``model.norm`` of the bucket's residual) -> its KV holds positions 0..n-2;
    the row (x_n, h_{n-1}) at n-1 is left to the first draft step (one uniform draft path). The main hidden row n-1 is
    read back once per request (host [dim]) and uploaded as the batch's first-step ``hidden_in``.
  * draft step (traced per width w): ``hidden_in`` [1,1,w,dim] + token [1,w] + position [w] -> embedding gather ->
    pre-fc norms (decode sharded configs) -> fc (1D decode matmuls, replicated weights) -> the layer's decode forward
    (paged KV write at the position, SDPA over <= position) -> mtp.norm -> sharded lm_head -> per-device (argmax,max)
    readback [TP,w]; the post-norm output is copied back into ``hidden_in`` in-trace for the chain. ``select_hidden``
    (eager, 0/1 matmul + copy) loads each user's accepted row of the verify step's ``out_hidden`` into ``hidden_in``.
  Trace safety: every buffer allocated here exists before any capture; per-step values are DMA'd in; the draft-step
  programs are compiled eagerly (compile_step) before ANY trace is captured (tests/VERIFY_W32_AUDIT.md rule).

Host side (``MTPHostReference``): the same head in fp32 torch straight from the safetensors (bf16 weights), with a
per-user position-indexed KV, for the PCC / argmax-agreement check of the device draft logits.
"""
import os
import time

import torch
from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt import masked_bucket_trace as mbt
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt import verify_grid as vg
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.verify_step import _EXACT_MM, argmax_sharded_rows, combine_sharded_argmax
from models.demos.blackhole.qwen36.tt.weight_mapping import load_qwen36_mtp_state_dict
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.tt_transformers.tt.common import Mode, get_block_size
from models.tt_transformers.tt.distributed_norm import DistributedNorm

MTP_PREFIX = "mtp."
_SDPA_PT_BLOCKS = 32  # the SDPA page-table stick: 32 blocks (attention/tp.py forward_prefill_paged pads to this)


class _StepBufs:
    """Per-width persistent inputs / outputs of the draft step."""

    def __init__(self):
        self.tok = self.pos = self.cos = self.sin = self.pt = self.hidden_in = None
        self.trace_id = None
        self.out_idx = self.out_val = self.out_logits = self.out_hidden = None


class MTPHead:
    def __init__(self, model, page_tables, widths, buckets, keep_logits=False, layer_index=None):
        """page_tables: torch [BMAX, blocks] int32, row = decode slot (the verify plans' rows are its first w rows).
        widths: the draft-step batch widths to support (one trace each); buckets: the prefill buckets to compile."""
        self.model = model
        args = model.args
        mesh = model.mesh_device
        self.mesh = mesh
        self.args = args
        self.tt_ccl = model.tt_ccl
        assert model.num_devices > 1, "the MTP head is TP only"
        assert model._decode_fused_all_reduce(), "the draft step runs the fused decode all-reduce residual path"
        assert model._paged_kv_caches, "allocate_kv_caches first (the MTP KV mirrors the main cache)"
        self.keep_logits = bool(keep_logits)
        self.dim = args.dim
        self.per_shard = args.vocab_size // model.num_devices
        self.rep = ttnn.ReplicateTensorToMesh(mesh)
        pt = page_tables if isinstance(page_tables, torch.Tensor) else torch.as_tensor(page_tables)
        self.page_tables = pt.to(torch.int32)
        assert self.page_tables.shape[1] % 8 == 0, "page-table stick must be a multiple of 8 blocks"

        # --- the virtual layer index and the weights ---
        idx = int(layer_index) if layer_index is not None else len(args.attention_type_list)
        if len(args.attention_type_list) <= idx:
            # is_full_attention_layer(idx) reads this list (only from_pretrained's layer selection reads its length,
            # and that has already run): extend it so the MTP layer builds as a full-attention Qwen36DecoderLayer.
            args.attention_type_list = list(args.attention_type_list) + ["full_attention"] * (
                idx + 1 - len(args.attention_type_list)
            )
        assert args.is_full_attention_layer(idx)
        self.layer_index = idx
        t0 = time.perf_counter()
        sd = load_qwen36_mtp_state_dict(args.CKPT_DIR, idx)
        cache = args.weight_cache_path()
        os.makedirs(cache, exist_ok=True)
        self.layer = Qwen36DecoderLayer(mesh, args, sd, idx, cache, tt_ccl=self.tt_ccl)
        assert self.layer.is_full_attention

        def _norm(key, distributed):
            n = RMSNorm(
                device=mesh,
                dim=args.dim,
                state_dict=sd,
                weight_key=key,
                state_dict_prefix=MTP_PREFIX,
                weight_cache_path=cache,
                weight_dtype=ttnn.bfloat16,
                add_unit_offset=True,
                eps=args.norm_eps,
                **(
                    dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=self.tt_ccl)
                    if distributed
                    else {}
                ),
            )
            return DistributedNorm(n, args, tt_ccl=self.tt_ccl, TG=args.is_galaxy) if distributed else n

        self.final_norm = _norm("norm", True)  # like model.norm: prefill distributed+gather, decode sharded
        # pre-fc norms: bare RMSNorm with the distributed flag -- PREFILL runs the framework's distributed rmsnorm
        # (pre/post all-gather stats) on FRACTURED [S, dim/TP] rows (a plain ttnn.rms_norm over a full-width
        # interleaved [S, 5120] row sizes its static CBs into the persistent L1 buffers: "dataflow buffers clash with
        # L1 buffers", logs/mtp_smoke1.log); DECODE takes the sharded decode-norm config on the replicated residual.
        self.pre_fc_norm_emb = _norm("pre_fc_norm_embedding", True).norm
        self.pre_fc_norm_hid = _norm("pre_fc_norm_hidden", True).norm
        fc = sd["mtp.fc.weight"]  # [dim, 2*dim] = [out, in]; in = concat(embedding, hidden) (vLLM cat order)
        assert tuple(fc.shape) == (args.dim, 2 * args.dim), fc.shape
        fc_e, fc_h = fc[:, : args.dim].contiguous(), fc[:, args.dim :].contiguous()
        # prefill: ROW-parallel (K-sharded [dim/TP, dim]) on the fractured normed inputs -> per-device partial [S, dim]
        # -> reduce-scatter = the fractured residual; decode: replicated [dim, dim] -> replicated residual
        self.fc_e_row = tpc.shard_w(fc_e, mesh, 0, ttnn.DRAM_MEMORY_CONFIG, str(cache / "mtp.fc_e.row"), ttnn.bfloat8_b)
        self.fc_h_row = tpc.shard_w(fc_h, mesh, 0, ttnn.DRAM_MEMORY_CONFIG, str(cache / "mtp.fc_h.row"), ttnn.bfloat8_b)

        def _rep_w(w, name):
            return ttnn.as_tensor(
                w.to(torch.bfloat16).T.contiguous(),
                dtype=ttnn.bfloat8_b,
                device=mesh,
                mesh_mapper=self.rep,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=str(cache / name),
            )

        self.fc_e_rep = _rep_w(fc_e, "mtp.fc_e.rep")
        self.fc_h_rep = _rep_w(fc_h, "mtp.fc_h.rep")
        self.fc_progcfg = tpc.create_matmul_1d_decode_progcfg(
            tpc.TILE_SIZE, args.dim, args.dim, num_cores=64, grid_w=getattr(args, "decode_grid_w", 8)
        )
        self.compute_cfg = tpc.COMPUTE_HIFI2
        del sd
        logger.info(f"[mtp] head weights loaded (layer index {idx}) in {time.perf_counter() - t0:.1f}s")

        # --- paged KV: the main cache's shape (+1 pad block for the fixed-width prefill fill), bound to the layer ---
        k0 = model._paged_kv_caches[0][0]
        kv_shape = list(k0.shape)
        self.n_main_blocks = kv_shape[0]
        self.pad_block = kv_shape[0]  # the extra block: never in any page table
        kv_shape[0] += 1
        self.block_size = get_block_size(model._paged_kv_caches)
        self.kv_dtype = k0.dtype

        def _mk():
            return ttnn.as_tensor(
                torch.zeros(kv_shape, dtype=torch.bfloat16),
                device=mesh,
                dtype=self.kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.rep,
            )

        self.kv_cache = [_mk(), _mk()]
        self.layer.attention.set_paged_kv_cache(*self.kv_cache)

        # --- decode-side layouts ---
        self.nc_dec = args.get_norm_config("attn", Mode.DECODE)
        self.act_memcfg = self.nc_dec["sharded_output_config"]
        rd = args.rope_head_dim

        # --- per-width draft-step buffers ---
        self.sb = {}
        for w in sorted(set(int(v) for v in widths)):
            assert 1 <= w <= self.page_tables.shape[0]
            b = _StepBufs()
            b.tok = self._up(torch.zeros(1, w, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            b.pos = self._up(torch.zeros(w, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            cos0, sin0 = vg.rope_cos_sin(torch.zeros(w, dtype=torch.int32), rd, args.rope_theta)
            b.cos = self._up(cos0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            b.sin = self._up(sin0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            b.pt = self._up(self.page_tables[:w].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            b.hidden_in = self._up(
                torch.zeros(1, 1, w, args.dim, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
            )
            self.sb[w] = b

        # --- per-bucket prefill buffers (values per request: tokens, fill page table; cos/sin constant) ---
        self.pf = {}
        for bucket in sorted(set(int(v) for v in buckets)):
            assert bucket % self.block_size == 0
            cos_t, sin_t = model._rope_tp_cos_sin_torch(0, bucket)
            self.pf[bucket] = dict(
                tok=self._up(torch.zeros(1, bucket, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                cos=self._up(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                sin=self._up(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                fill_pt=self._up(
                    torch.full((1, bucket // self.block_size), self.pad_block, dtype=torch.int32),
                    ttnn.int32,
                    ttnn.ROW_MAJOR_LAYOUT,
                ),
            )
        self.pf_full_pt = self._up(
            torch.zeros(1, _SDPA_PT_BLOCKS, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.pf_csi = self._up(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        assert self.page_tables.shape[1] <= _SDPA_PT_BLOCKS

        # --- per-verify-plan 0/1 row selectors (bind_plan) ---
        self._plan_sel = {}
        self._host_refs = []
        # per-request host bookkeeping filled by prefill_hook: slot -> main post-norm hidden row n-1 (bf16 [dim])
        self.pending_rows = {}
        self.probe = False  # keep the prefill's (tokens_shift, hidden rows) per slot for the host reference
        self.probe_prefill = {}
        self.stats = {
            "draft_steps": 0,
            "draft_wall": 0.0,
            "select_calls": 0,
            "select_wall": 0.0,
            "prefill_calls": 0,
            "prefill_wall": 0.0,
        }
        logger.info(
            f"[mtp] head ready: widths {sorted(self.sb)} buckets {sorted(self.pf)} kv {kv_shape} {self.kv_dtype} "
            f"pad block {self.pad_block}"
        )

    # ------------------------------------------------------------------------------------------ helpers
    def _up(self, t, dtype, layout):
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=self.rep
        )

    def _dma(self, host_t, dst, dtype, layout):
        h = ttnn.from_torch(host_t, dtype=dtype, layout=layout, device=None, mesh_mapper=self.rep)
        ttnn.copy_host_to_device_tensor(h, dst)
        self._host_refs.append(h)

    def _sync(self):
        ttnn.synchronize_device(self.mesh)
        self._host_refs = []

    # ------------------------------------------------------------------------------------------ prefill
    def prefill_forward(self, bucket, hidden_frac):
        """Eager MTP forward over one bucket (inputs: the bucket's persistent buffers + hidden_frac, the main model's
        pre-final-norm residual [1,1,bucket,dim/TP]). Side effect: the head's KV at the fill page table's blocks.
        Everything stays FRACTURED [S, dim/TP] (no replicated full-width prefill rows): distributed pre-fc norms,
        row-parallel fc partials summed by the reduce-scatter. Returns (h_frac_n, y): the main POST-norm hidden
        (fractured; gather on the host for a row) and the layer output [1,1,bucket,dim/TP]; the caller deallocates both.
        """
        m, args = self.model, self.args
        b = self.pf[bucket]
        x_e = m.embd(b["tok"])  # [1,bucket,dim/TP]
        x_e = ttnn.reshape(x_e, (1, 1, bucket, x_e.shape[-1]))
        x_e = ttnn.to_memory_config(x_e, ttnn.DRAM_MEMORY_CONFIG)
        e_n = self.pre_fc_norm_emb(x_e, mode=Mode.PREFILL)  # distributed rmsnorm -> fractured normed rows
        ttnn.deallocate(x_e)
        h_frac_n = m.norm.norm(hidden_frac, mode=Mode.PREFILL)  # the LM-head input rows, fractured (no gather)
        h_n = self.pre_fc_norm_hid(h_frac_n, mode=Mode.PREFILL)
        pe = ttnn.linear(
            e_n, self.fc_e_row, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ph = ttnn.linear(
            h_n, self.fc_h_row, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(e_n)
        ttnn.deallocate(h_n)
        part = ttnn.add(pe, ph, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # per-device partial [1,1,bucket,dim]
        ttnn.deallocate(pe)
        ttnn.deallocate(ph)
        x = tt_all_reduce(
            part,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # fractured residual [1,1,bucket,dim/TP]; tt_all_reduce frees `part` itself
        y = self.layer.forward(
            x,
            cos=b["cos"],
            sin=b["sin"],
            mode="prefill",
            page_table=self.pf_full_pt,
            chunk_page_table=b["fill_pt"],
            chunk_start_idx=0,
            chunk_start_idx_tensor=self.pf_csi,
            valid_len=bucket,
        )
        ttnn.deallocate(x)
        return h_frac_n, y

    def compile_prefill(self, bucket):
        """Compile the bucket's prefill programs eagerly (before any trace capture) on a zero residual; writes garbage
        into the pad block only."""
        t0 = time.perf_counter()
        self._dma(
            torch.full((1, bucket // self.block_size), self.pad_block, dtype=torch.int32),
            self.pf[bucket]["fill_pt"],
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        full = torch.zeros(1, _SDPA_PT_BLOCKS, dtype=torch.int32)
        full[0, : self.page_tables.shape[1]] = self.page_tables[0]
        self._dma(full, self.pf_full_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        dummy = self._up(
            torch.zeros(1, 1, bucket, self.dim // self.model.num_devices, dtype=torch.bfloat16),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
        )
        h_full, y = self.prefill_forward(bucket, dummy)
        self._sync()
        for t in (h_full, y, dummy):
            ttnn.deallocate(t)
        logger.info(f"[mtp] prefill programs compiled for bucket {bucket} in {time.perf_counter() - t0:.1f}s")

    def prefill_hook(self, user_ctx, hidden, token_buf, actual_len, bucket, chunk_start):
        """model.prefill_hidden_hook: fill the head's KV over the prompt of the request being prefilled.
        token_buf [1,bucket] (real tokens x_0..x_{n-1} then padding), hidden = the bucket's pre-final-norm residual."""
        if user_ctx is None:
            return  # warm-up / dummy prefills outside prefill_paged_slots
        if chunk_start != 0:
            raise NotImplementedError("MTP prefill supports prompts within one bucket (no chunked long prompts yet)")
        if bucket not in self.pf:
            raise KeyError(f"MTP prefill not compiled for bucket {bucket} (have {sorted(self.pf)})")
        n = int(actual_len)
        assert n >= 2, "MTP prefill needs at least two prompt tokens"
        _, slot = user_ctx
        t0 = time.perf_counter()
        # tokens shifted by one: x_1..x_{n-1} at positions 0..n-2 (row n-1 = the first draft step)
        tok_shift = torch.zeros(1, bucket, dtype=torch.int32)
        tok_shift[0, : n - 1] = token_buf[0, 1:n].to(torch.int32)
        pt_row = self.page_tables[slot].reshape(1, -1)
        fill = mbt.fill_pt_row(pt_row, 0, n - 1, bucket, self.pad_block, self.block_size)
        full = torch.zeros(1, _SDPA_PT_BLOCKS, dtype=torch.int32)
        full[0, : pt_row.shape[1]] = pt_row[0]
        b = self.pf[bucket]
        self._dma(tok_shift, b["tok"], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._dma(fill, b["fill_pt"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._dma(full, self.pf_full_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        h_full, y = self.prefill_forward(bucket, hidden)
        self._sync()
        rows = ttnn.to_torch(h_full, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3))  # gather the shards
        rows = rows.reshape(-1, self.dim)[:n]  # bf16 [n, dim] post-final-norm (the LM-head input rows)
        self.pending_rows[slot] = rows[n - 1].clone()
        if self.probe:
            self.probe_prefill[slot] = (tok_shift[0, : n - 1].clone(), rows[: n - 1].clone())
        ttnn.deallocate(h_full)
        ttnn.deallocate(y)
        self.stats["prefill_calls"] += 1
        self.stats["prefill_wall"] += time.perf_counter() - t0

    # ------------------------------------------------------------------------------------------ draft step
    def step_forward(self, w):
        """The traced draft-step body for width w: reads the persistent inputs, writes (idx, val[, logits, hidden])
        and copies the post-norm output into hidden_in (chain input)."""
        m = self.model
        b = self.sb[w]
        x_e = m.embd(b.tok)  # [1,w,dim/TP]
        x_e = ttnn.reshape(x_e, (1, 1, w, x_e.shape[-1]))
        x_e = m._decode_residual_in(x_e)  # begin_step + all-gather -> replicated L1 width-sharded [1,1,w,dim]
        e_n = self.pre_fc_norm_emb(x_e, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=self.nc_dec)
        ttnn.deallocate(x_e)
        h_sh = ttnn.to_memory_config(b.hidden_in, self.act_memcfg)
        h_n = self.pre_fc_norm_hid(h_sh, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=self.nc_dec)
        ttnn.deallocate(h_sh)
        pe = tpc.matmul_1d_decode(e_n, self.fc_e_rep, self.fc_progcfg, self.compute_cfg, ttnn.L1_MEMORY_CONFIG)
        ph = tpc.matmul_1d_decode(h_n, self.fc_h_rep, self.fc_progcfg, self.compute_cfg, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(e_n)
        ttnn.deallocate(h_n)
        x = ttnn.add(pe, ph, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pe)
        ttnn.deallocate(ph)
        x_sh = ttnn.to_memory_config(x, self.act_memcfg)  # the decode residual layout the layer's norms expect
        ttnn.deallocate(x)
        y = self.layer.forward(x_sh, cos=b.cos, sin=b.sin, mode="decode", position_tensor=b.pos, page_table=b.pt)
        ttnn.deallocate(x_sh)
        hn = self.final_norm.norm(y, mode=Mode.DECODE, in_sharded=True, out_sharded=True, norm_config=self.nc_dec)
        ttnn.deallocate(y)
        h_out = ttnn.to_memory_config(hn, ttnn.DRAM_MEMORY_CONFIG)  # [1,1,w,dim] replicated: chain + LM-head input
        ttnn.deallocate(hn)
        logits = ttnn.linear(h_out, m.lm_head_weight)  # vocab-sharded [1,1,w,V/TP]
        idx, val = argmax_sharded_rows(logits)
        if not self.keep_logits:
            ttnn.deallocate(logits)
            logits = None
        ttnn.copy(h_out, b.hidden_in)  # chain: the next step's hidden input
        return idx, val, logits, h_out

    def _upload_step(self, w, tokens, positions):
        b = self.sb[w]
        tok = torch.tensor([int(t) for t in tokens], dtype=torch.int32).reshape(1, w)
        pos = torch.tensor([int(p) for p in positions], dtype=torch.int32)
        assert int(pos.min()) >= 0
        self._dma(tok, b.tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._dma(pos, b.pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        rope_delta = int(getattr(self.model.rope, "rope_delta", 0) or 0)
        cos, sin = vg.rope_cos_sin(pos + rope_delta, self.args.rope_head_dim, self.args.rope_theta)
        self._dma(cos, b.cos, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._dma(sin, b.sin, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    def compile_step(self, w):
        """Eager draft step (compiles every program; writes garbage KV at position 8 of users 0..w-1)."""
        t0 = time.perf_counter()
        self._upload_step(w, [1] * w, [8] * w)
        outs = self.step_forward(w)
        self._sync()
        for t in outs:
            if t is not None:
                ttnn.deallocate(t)
        logger.info(f"[mtp] draft step w={w} compiled in {time.perf_counter() - t0:.1f}s")

    def capture_step(self, w):
        b = self.sb[w]
        assert b.trace_id is None
        self._upload_step(w, [1] * w, [8] * w)
        self._sync()
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        b.out_idx, b.out_val, b.out_logits, b.out_hidden = self.step_forward(w)
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self._sync()
        b.trace_id = tid

    def release(self):
        for b in self.sb.values():
            if b.trace_id is not None:
                ttnn.release_trace(self.mesh, b.trace_id)
                b.trace_id = None

    def run_step(self, w, tokens, positions):
        """One draft step for w users: upload -> replay (or eager) -> argmax [w] (list of ints)."""
        b = self.sb[w]
        self._upload_step(w, tokens, positions)
        if b.trace_id is None:
            idx, val, logits, h_out = self.step_forward(w)
            self._sync()
            out = combine_sharded_argmax(self.mesh, idx, val, w, self.per_shard).tolist()
            for t in (idx, val, logits, h_out):
                if t is not None:
                    ttnn.deallocate(t)
            return out
        ttnn.execute_trace(self.mesh, b.trace_id, cq_id=0, blocking=False)
        self._sync()
        return combine_sharded_argmax(self.mesh, b.out_idx, b.out_val, w, self.per_shard).tolist()

    def read_hidden_in(self, w):
        """Host copy of hidden_in [w, dim] (bf16) -- the draft step's hidden input (probe)."""
        return ttnn.to_torch(ttnn.get_device_tensors(self.sb[w].hidden_in)[0]).reshape(-1, self.dim)[:w].clone()

    def read_logits(self, w):
        """Host copy of the last traced step's full draft logits [w, V] (float32); needs keep_logits and a trace."""
        b = self.sb[w]
        assert self.keep_logits and b.out_logits is not None
        lg = ttnn.to_torch(b.out_logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3)).float()
        return lg.reshape(-1, lg.shape[-1])[:w].clone()

    # ------------------------------------------------------------------------------------------ batch driving
    def begin_batch(self, w, slots):
        """Upload the first-step hidden input of a fresh batch: the main hidden row n-1 of every user's prefill."""
        rows = torch.stack([self.pending_rows[int(s)] for s in slots]).to(torch.bfloat16)  # [w, dim]
        self._dma(rows.reshape(1, 1, w, self.dim).contiguous(), self.sb[w].hidden_in, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._sync()

    def bind_plan(self, plan):
        """Persistent 0/1 selector [1,1,w,R] for a verify plan (allocate before any capture)."""
        assert plan.keep_hidden, "the verify plan must keep its hidden rows (VerifyStep(keep_hidden=True))"
        self._plan_sel[id(plan)] = self._up(
            torch.zeros(1, 1, plan.w, plan.R, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
        )

    def compile_select(self, plan):
        """Compile the select programs on a temporary [1,1,R,dim] tensor (the plan's out_hidden exists only after its
        capture); call before any trace capture."""
        tmp = self._up(torch.zeros(1, 1, plan.R, self.dim, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._select_from(plan, [0] * plan.w, tmp)
        ttnn.deallocate(tmp)

    def _select_from(self, plan, accepts, hidden_rows):
        w, T, R = plan.w, plan.T, plan.R
        sel = torch.zeros(1, 1, w, R, dtype=torch.float32)
        for s in range(w):
            sel[0, 0, s, vg.row(s, int(accepts[s]), T)] = 1.0
        sel_tt = self._plan_sel[id(plan)]
        self._dma(sel, sel_tt, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        picked = ttnn.matmul(
            sel_tt, hidden_rows, compute_kernel_config=_EXACT_MM, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.copy(picked, self.sb[w].hidden_in)
        ttnn.deallocate(picked)
        self._sync()

    def select_hidden(self, plan, accepts):
        """hidden_in[s] <- verify row (s, a_s) of plan.out_hidden (exact 0/1 matmul), eager."""
        t0 = time.perf_counter()
        assert plan.out_hidden is not None, "run the verify step first"
        self._select_from(plan, accepts, plan.out_hidden)
        self.stats["select_calls"] += 1
        self.stats["select_wall"] += time.perf_counter() - t0

    def draft(self, w, k, last, positions, observer=None):
        """k chained drafts per user (schedule: verify_grid.chain_drafts). last[s] = the user's row-0 token (t'_s),
        positions[s] = P_s (the verify grid's committed position); hidden_in holds h at position P_s - 1
        (select_hidden / begin_batch). Step j writes the head's KV at P_s - 1 + j. observer(phase, j, tokens,
        positions, drafts_j) is called with phase "pre" before each step (hidden_in still holds the step's input) and
        "post" after it (probe)."""
        t0 = time.perf_counter()
        drafts = vg.chain_drafts(lambda tok, pos: self.run_step(w, tok, pos), w, k, last, positions, observer)
        self.stats["draft_steps"] += k
        self.stats["draft_wall"] += time.perf_counter() - t0
        return drafts

    def time_step_replays(self, w, n=50):
        """Traced draft-step time (upload + replay + argmax readback), median / min ms over n replays."""
        b = self.sb[w]
        assert b.trace_id is not None
        ms = []
        for i in range(n):
            t0 = time.perf_counter()
            self.run_step(w, [(100 + i + s) % 1000 for s in range(w)], [16 + i + s for s in range(w)])
            ms.append(1e3 * (time.perf_counter() - t0))
        ms.sort()
        return ms[len(ms) // 2], ms[0]


# ============================================================================================== host reference
class MTPHostReference:
    """fp32 torch MTP head straight from the safetensors (weights bf16 -> fp32), per-user position-indexed KV.

    forward(state, tokens, hidden, positions): tokens[i] = x_{p_i+1}, hidden[i] = the main model's post-norm hidden at
    position p_i (bf16 rows read from the device), positions[i] = p_i (int). Writes k/v at those positions (overwriting
    a previous write at the same position, like the paged cache), attends over every stored position <= p_i and returns
    (logits [S, V] float32, out [S, dim] float32 = the post-mtp.norm output the chain feeds back)."""

    def __init__(self, ckpt_dir, args):
        import json

        from safetensors import safe_open

        self.dim = args.dim
        self.NH, self.NKV, self.HD = args.n_heads, args.n_kv_heads, args.head_dim
        self.rd = args.rope_head_dim
        self.theta = float(args.rope_theta)
        self.eps = float(args.norm_eps)
        self.V = args.vocab_size
        sd = load_qwen36_mtp_state_dict(ckpt_dir, 0)
        f32 = lambda k: sd[k].float()
        self.w_pre_e = f32("mtp.pre_fc_norm_embedding.weight")
        self.w_pre_h = f32("mtp.pre_fc_norm_hidden.weight")
        self.w_norm = f32("mtp.norm.weight")
        self.fc = f32("mtp.fc.weight")  # [dim, 2*dim]
        p = "layers.0."
        self.in_ln = f32(p + "input_layernorm.weight")
        self.post_ln = f32(p + "post_attention_layernorm.weight")
        self.q_proj = f32(p + "self_attn.q_proj.weight")  # [NH*2*HD, dim]
        self.k_proj = f32(p + "self_attn.k_proj.weight")
        self.v_proj = f32(p + "self_attn.v_proj.weight")
        self.o_proj = f32(p + "self_attn.o_proj.weight")
        self.q_norm = f32(p + "self_attn.q_norm.weight")
        self.k_norm = f32(p + "self_attn.k_norm.weight")
        self.gate_proj = f32(p + "mlp.gate_proj.weight")
        self.up_proj = f32(p + "mlp.up_proj.weight")
        self.down_proj = f32(p + "mlp.down_proj.weight")
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
            wm = json.load(f)["weight_map"]
        emb_key = next(k for k in wm if k.endswith("embed_tokens.weight"))
        lm_key = next(k for k in wm if k.endswith("lm_head.weight"))
        with safe_open(os.path.join(ckpt_dir, wm[emb_key]), framework="pt") as sf:
            self.embed = sf.get_tensor(emb_key)  # bf16 [V, dim]
        with safe_open(os.path.join(ckpt_dir, wm[lm_key]), framework="pt") as sf:
            self.lm_head = sf.get_tensor(lm_key)  # bf16 [V, dim]
        assert tuple(self.embed.shape) == (self.V, self.dim) and tuple(self.lm_head.shape) == (self.V, self.dim)

    def rms(self, x, w):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + w)

    def _rope(self, x, positions):
        """x [S, H, HD]; rotate the first rd dims, HF rotate-half convention (cos/sin of cat(freqs, freqs))."""
        inv = 1.0 / (self.theta ** (torch.arange(0, self.rd, 2).float() / self.rd))
        freqs = torch.outer(positions.float(), inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
        xr, xp = x[..., : self.rd], x[..., self.rd :]
        r1, r2 = xr[..., : self.rd // 2], xr[..., self.rd // 2 :]
        return torch.cat([xr * cos + torch.cat([-r2, r1], dim=-1) * sin, xp], dim=-1)

    @staticmethod
    def new_state():
        return {}

    @torch.no_grad()
    def forward(self, state, tokens, hidden, positions):
        S = len(tokens)
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        positions = torch.as_tensor(positions, dtype=torch.long)
        H = torch.as_tensor(hidden).float().reshape(S, self.dim)
        e = self.embed[tokens].float()
        x = torch.cat([self.rms(e, self.w_pre_e), self.rms(H, self.w_pre_h)], dim=-1) @ self.fc.T  # [S, dim]
        # --- attention ---
        h1 = self.rms(x, self.in_ln)
        qg = (h1 @ self.q_proj.T).view(S, self.NH, 2 * self.HD)
        q, gate = qg[..., : self.HD], qg[..., self.HD :].reshape(S, self.NH * self.HD)
        k = (h1 @ self.k_proj.T).view(S, self.NKV, self.HD)
        v = (h1 @ self.v_proj.T).view(S, self.NKV, self.HD)
        q = self._rope(self.rms(q, self.q_norm), positions)
        k = self._rope(self.rms(k, self.k_norm), positions)
        for i in range(S):
            state[int(positions[i])] = (k[i].clone(), v[i].clone())
        keys_pos = torch.tensor(sorted(state), dtype=torch.long)
        K = torch.stack([state[int(p)][0] for p in keys_pos])  # [n, NKV, HD]
        Vv = torch.stack([state[int(p)][1] for p in keys_pos])
        g = self.NH // self.NKV
        Kh = K.repeat_interleave(g, dim=1).permute(1, 0, 2)  # [NH, n, HD]
        Vh = Vv.repeat_interleave(g, dim=1).permute(1, 0, 2)
        qh = q.permute(1, 0, 2)  # [NH, S, HD]
        scores = torch.matmul(qh, Kh.transpose(1, 2)) * (self.HD**-0.5)  # [NH, S, n]
        mask = keys_pos[None, :] > positions[:, None]  # [S, n] future keys
        scores = scores.masked_fill(mask[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, Vh).permute(1, 0, 2).reshape(S, self.NH * self.HD)
        attn = (out * torch.sigmoid(gate)) @ self.o_proj.T
        x = x + attn
        h2 = self.rms(x, self.post_ln)
        mlp = (torch.nn.functional.silu(h2 @ self.gate_proj.T) * (h2 @ self.up_proj.T)) @ self.down_proj.T
        x = x + mlp
        out_h = self.rms(x, self.w_norm)
        logits = torch.empty(S, self.V, dtype=torch.float32)
        step = 32768
        for a in range(0, self.V, step):
            logits[:, a : a + step] = out_h @ self.lm_head[a : a + step].float().T
        return logits, out_h
