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
  * prefill (``prefill_hook``, installed as ``model.prefill_hidden_hook``): after the main prefill of a request's
    segment (a masked bucket, or one 2048-token chunk of a long prompt: model.py fires the hook per chunk with the
    chunk trace's output residual and the chunk's tokens plus the FIRST token of the next chunk), the head runs ONE
    eager forward over the segment with the tokens shifted by one and the main post-norm hidden (``model.norm`` of
    the segment's residual) at ``chunk_start_idx`` = the segment's position, attending over its own KV of the earlier
    chunks. A segment followed by more prompt (the next token is known) fills all its positions; the FINAL segment
    of n tokens fills positions ..n-2 and leaves the row (x_n, h_{n-1}) at n-1 to the first draft step (one uniform
    draft path). So after a prefill of N tokens the head's KV holds positions 0..N-2 and the main hidden row N-1
    (the LM-head input, bf16 [dim]) is read back once per request (an exact one-hot select, no full-bucket read) into
    ``pending_rows[slot]`` -- the batch's first-step ``hidden_in`` (``begin_batch``).
  * P/D hand-off (tt/pd_transfer.py, the plugin's TTMooncakeConnector): ``model.mtp_head`` (set by the constructor)
    makes the head's KV the 17th attention layer of ``pd_transfer._attention_layers`` -- exported/imported with the
    request's blocks in the same payload, same block ids -- and ``pending_rows[slot]`` travels as the payload's
    ``mtp.hidden`` row (``pd_transfer.export_mtp_hidden`` on P, ``import_mtp_hidden`` / ``set_hidden_in`` on D). A
    D instance therefore holds, after the import, exactly the state this process holds after its own prefill of the
    same N tokens: KV 0..N-2 + hidden row N-1; its first draft step is (x_N, h_{N-1}) at N-1 with x_N = the token
    the P side's logits (or, under the connector's N-1 truncation, the real last prompt token) supply.
    Allocation contract for D (the serving loop): ``MTPHead.allocate_kv(model)`` returns the [k, v] pair (main
    cache's shape + 1 pad block); pass it as ``paged_kv=`` or let the constructor call it. Build the head BEFORE
    ``pd_transfer.import_warmup`` / ``export_warmup`` (their pools bake the cache list) and before any trace capture.
    ``QWEN36_SPEC_MTP=1`` (``spec_mtp_enabled``) is the process-wide switch; unset, nothing here is constructed and
    the plain path is byte-identical.
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
_SDPA_PT_BLOCKS = 32  # the SDPA page-table stick granularity (attention/tp.py forward_prefill_paged pads to x32)


def spec_mtp_enabled() -> bool:
    """QWEN36_SPEC_MTP=1: speculative decoding with the MTP head is configured for this process (P: run the MTP
    prefill per request and ship the head's KV + hidden row; D: allocate the head and import them). Default off:
    nothing MTP-related is built and the served path is byte-identical to the plain one."""
    return os.environ.get("QWEN36_SPEC_MTP", "0") == "1"


class _StepBufs:
    """Per-width persistent inputs / outputs of the draft step."""

    def __init__(self):
        self.tok = self.pos = self.cos = self.sin = self.pt = self.hidden_in = None
        self.pt_host = None  # host copy of the uploaded page table (set_page_table skips a no-change)
        self.trace_id = None
        self.out_idx = self.out_val = self.out_logits = self.out_hidden = None


class MTPHead:
    def __init__(
        self,
        model,
        page_tables=None,
        widths=(),
        buckets=(),
        keep_logits=False,
        layer_index=None,
        paged_kv=None,
        sdpa_pt_blocks=None,
    ):
        """page_tables: torch [BMAX, blocks] int32, row = decode slot (the verify plans' rows are its first w rows);
        the draft step's page tables and, for a prefill hook called without a page-table row (user_ctx of two
        fields), the prefill's. None for a prefill-only head (a P instance: widths must then be empty; the served
        prefill hands the hook the request's own row). widths: the draft-step batch widths to support (one trace
        each); buckets: the prefill buckets to compile (the masked buckets + the 2048 chunk for long prompts).
        paged_kv: the head's [k, v] cache pair from ``allocate_kv`` (default: allocated here). sdpa_pt_blocks: width
        of the prefill's SDPA page table (default: the model's chunk-trace page table width when captured, else the
        page_tables' width rounded up to 32); it must cover the longest prompt's blocks. Registers itself as
        ``model.mtp_head`` (pd_transfer's 17th attention layer)."""
        self.model = model
        args = model.args
        mesh = model.mesh_device
        self.mesh = mesh
        self.args = args
        self.tt_ccl = model.tt_ccl
        assert model.num_devices > 1, "the MTP head is TP only"
        assert model._decode_fused_all_reduce(), "the draft step runs the fused decode all-reduce residual path"
        assert model._paged_kv_caches, "allocate_kv_caches first (the MTP KV mirrors the main cache)"
        assert getattr(model, "mtp_head", None) is None, "the model already has an MTP head (model.mtp_head)"
        self.keep_logits = bool(keep_logits)
        self.dim = args.dim
        self.per_shard = args.vocab_size // model.num_devices
        self.rep = ttnn.ReplicateTensorToMesh(mesh)
        widths = sorted(set(int(v) for v in widths))
        if page_tables is None:
            assert not widths, "draft-step widths need the decode page tables"
            self.page_tables = None
        else:
            pt = page_tables if isinstance(page_tables, torch.Tensor) else torch.as_tensor(page_tables)
            self.page_tables = pt.to(torch.int32)
            assert self.page_tables.shape[1] % 8 == 0, "page-table stick must be a multiple of 8 blocks"
        buf = getattr(model, "_chunk_full_page_table_buf", None)
        if sdpa_pt_blocks is None:
            if buf is not None:
                sdpa_pt_blocks = int(buf.shape[-1])
            elif self.page_tables is not None:
                sdpa_pt_blocks = int(self.page_tables.shape[1])
            else:
                sdpa_pt_blocks = _SDPA_PT_BLOCKS
        self.sdpa_pt_blocks = -(-int(sdpa_pt_blocks) // _SDPA_PT_BLOCKS) * _SDPA_PT_BLOCKS

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
        self.n_main_blocks = int(k0.shape[0])
        self.pad_block = self.n_main_blocks  # the extra block: never in any page table
        self.block_size = get_block_size(model._paged_kv_caches)
        self.kv_dtype = k0.dtype
        self.kv_cache = list(paged_kv) if paged_kv is not None else self.allocate_kv(model)
        assert len(self.kv_cache) == 2 and tuple(self.kv_cache[0].shape) == (
            self.n_main_blocks + 1,
            *list(k0.shape)[1:],
        ), (
            f"MTP KV pair must be the main cache's shape + 1 pad block: {tuple(self.kv_cache[0].shape)} vs "
            f"{tuple(k0.shape)}"
        )
        kv_shape = list(self.kv_cache[0].shape)
        self.layer.attention.set_paged_kv_cache(*self.kv_cache)
        model.mtp_head = self

        # --- decode-side layouts ---
        self.nc_dec = args.get_norm_config("attn", Mode.DECODE)
        self.act_memcfg = self.nc_dec["sharded_output_config"]
        rd = args.rope_head_dim

        # --- per-width draft-step buffers (more widths / the served page-table width: build_step_buffers) ---
        self.sb = {}
        if widths:
            self.build_step_buffers(widths, self.page_tables)

        # --- per-bucket prefill buffers (values per request/segment: tokens, fill page table, cos/sin of the
        # segment's positions, the one-hot row selector of the hidden-row readback) ---
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
                sel=self._up(mbt.host_logit_sel(1, bucket), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            )
        self.pf_full_pt = self._up(
            torch.zeros(1, self.sdpa_pt_blocks, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        )
        self.pf_csi = self._up(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        assert self.page_tables is None or self.page_tables.shape[1] <= self.sdpa_pt_blocks

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
            f"pad block {self.pad_block} sdpa page table {self.sdpa_pt_blocks} blocks"
        )

    # ------------------------------------------------------------------------------------------ KV / hidden API
    @staticmethod
    def allocate_kv(model, num_blocks=None):
        """The head's paged KV pair ``[k, v]``: the main cache's per-block shape and dtype over ``num_blocks`` (default
        = the main cache's block count, which the request block ids index) PLUS one pad block at index ``num_blocks``
        (the fixed-width prefill fill's scratch; never in a page table). D side: call it with the engine's block count
        once the main caches exist (``allocate_kv_caches``) and hand the pair to ``MTPHead(..., paged_kv=pair)``, or
        let the constructor call it; either way it must exist before ``pd_transfer.import_warmup`` (the traced
        importer bakes the cache list) and before any trace capture."""
        assert model._paged_kv_caches, "allocate_kv_caches first"
        k0 = model._paged_kv_caches[0][0]
        shape = list(k0.shape)
        if num_blocks is not None:
            assert int(num_blocks) == int(
                shape[0]
            ), f"MTP KV block count {num_blocks} must equal the main cache's {shape[0]} (same block ids index both)"
        shape[0] = int(shape[0]) + 1
        rep = ttnn.ReplicateTensorToMesh(model.mesh_device)

        def _mk():
            return ttnn.as_tensor(
                torch.zeros(shape, dtype=torch.bfloat16),
                device=model.mesh_device,
                dtype=k0.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        return [_mk(), _mk()]

    @property
    def paged_kv(self):
        """The head's [k, v] cache pair (``allocate_kv`` layout), bound to ``self.attention``."""
        return self.kv_cache

    @property
    def attention(self):
        """The head's attention module: ``paged_k`` / ``paged_v`` are the pair above -- pd_transfer's 17th layer."""
        return self.layer.attention

    def set_hidden_in(self, slot, row):
        """D side (and any importer): the request's main POST-final-norm hidden row of its last prefilled position
        (bf16 [dim], the payload's ``mtp.hidden``) for decode slot ``slot`` -- what this process's own prefill hook
        would have stored (``pending_rows[slot]``); ``begin_batch`` uploads it as the slot's first-step hidden_in."""
        row = torch.as_tensor(row)
        assert row.numel() == self.dim, f"hidden row of {row.numel()} values, expected {self.dim}"
        self.pending_rows[int(slot)] = row.reshape(self.dim).to(torch.bfloat16).clone()

    def get_hidden_in(self, slot):
        """The stored hidden row of ``slot`` (bf16 [dim]) or None."""
        return self.pending_rows.get(int(slot))

    def pop_hidden_in(self, slot):
        """P side: take the row of a request whose state is being exported (None when no prefill stored one)."""
        return self.pending_rows.pop(int(slot), None)

    def build_step_buffers(self, widths, page_tables):
        """Allocate the draft-step inputs / chain buffer of every width in ``widths`` (BEFORE any trace capture).
        page_tables: torch [BMAX, blocks] int32 -- the decode page tables' WIDTH is what matters (the served block
        table width, refreshed per step with ``set_page_table``); its rows are the initial values. A D instance
        whose head was built prefill-only (widths=()) calls this from the decode warm-up."""
        args = self.args
        rd = args.rope_head_dim
        pt = page_tables if isinstance(page_tables, torch.Tensor) else torch.as_tensor(page_tables)
        pt = pt.to(torch.int32)
        assert pt.shape[1] % 8 == 0, "page-table stick must be a multiple of 8 blocks"
        if self.page_tables is None:
            self.page_tables = pt
        for w in sorted(set(int(v) for v in widths)):
            if w in self.sb:
                continue
            assert 1 <= w <= pt.shape[0]
            b = _StepBufs()
            b.tok = self._up(torch.zeros(1, w, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            b.pos = self._up(torch.zeros(w, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            cos0, sin0 = vg.rope_cos_sin(torch.zeros(w, dtype=torch.int32), rd, args.rope_theta)
            b.cos = self._up(cos0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            b.sin = self._up(sin0, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            b.pt = self._up(pt[:w].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            b.pt_host = pt[:w].clone()
            b.hidden_in = self._up(
                torch.zeros(1, 1, w, args.dim, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT
            )
            self.sb[w] = b

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
    def prefill_forward(self, bucket, hidden_frac, chunk_start=0):
        """Eager MTP forward over one bucket-sized segment at positions [chunk_start, chunk_start + bucket) (inputs:
        the bucket's persistent buffers -- tokens, cos/sin of those positions, fill page table -- + hidden_frac, the
        main model's pre-final-norm residual [1,1,bucket,dim/TP]; pf_full_pt / pf_csi hold the request's SDPA page
        table and the segment offset). Side effect: the head's KV at the fill page table's blocks; the SDPA attends
        over the head's KV of the earlier segments through the full page table. Everything stays FRACTURED
        [S, dim/TP] (no replicated full-width prefill rows): distributed pre-fc norms, row-parallel fc partials
        summed by the reduce-scatter. Returns (h_frac_n, y): the main POST-norm hidden (fractured; select a row with
        ``_select_row`` or gather on the host) and the layer output [1,1,bucket,dim/TP]; the caller deallocates both.
        Programs depend on the bucket only (the page table is fixed-width, the offset a tensor), so ``compile_prefill``
        per bucket compiles everything a request can run."""
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
            chunk_start_idx=int(chunk_start),
            chunk_start_idx_tensor=self.pf_csi,
            valid_len=bucket,
        )
        ttnn.deallocate(x)
        return h_frac_n, y

    def _select_row(self, bucket, h_frac_n, row):
        """Host bf16 [dim]: row ``row`` of the fractured post-norm hidden ``h_frac_n`` [1,1,bucket,dim/TP] -- an exact
        one-hot matmul (HiFi4, fp32 accumulate: one 1.0 x value term, the rest exact zeros) per device, then the
        shards gathered on the host. One fixed program per bucket (a slice would compile per row)."""
        sel_tt = self.pf[bucket]["sel"]
        self._dma(mbt.host_logit_sel(row + 1, bucket), sel_tt, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        picked = ttnn.matmul(sel_tt, h_frac_n, compute_kernel_config=_EXACT_MM, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._sync()
        out = ttnn.to_torch(picked, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3))
        ttnn.deallocate(picked)
        return out.reshape(-1)[: self.dim].to(torch.bfloat16).clone()

    def _stage_prefill_inputs(self, bucket, tok_shift, fill, pt_row, chunk_start):
        """DMA a segment's per-request values into the bucket's persistent buffers: shifted tokens, the KV-fill page
        table, the request's SDPA page table (padded/clipped to sdpa_pt_blocks), the segment offset and its RoPE."""
        b = self.pf[bucket]
        full = torch.zeros(1, self.sdpa_pt_blocks, dtype=torch.int32)
        w = min(int(pt_row.shape[1]), self.sdpa_pt_blocks)
        full[0, :w] = pt_row[0, :w]
        need = (int(chunk_start) + bucket) // self.block_size
        assert need <= self.sdpa_pt_blocks, (
            f"MTP SDPA page table of {self.sdpa_pt_blocks} blocks does not cover positions up to "
            f"{chunk_start + bucket} (build the head with sdpa_pt_blocks >= {need})"
        )
        self._dma(tok_shift, b["tok"], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._dma(fill, b["fill_pt"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._dma(full, self.pf_full_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self._dma(torch.tensor([int(chunk_start)], dtype=torch.int32), self.pf_csi, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cos_t, sin_t = self.model._rope_tp_cos_sin_torch(int(chunk_start), bucket)
        self._dma(cos_t, b["cos"], ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._dma(sin_t, b["sin"], ttnn.bfloat16, ttnn.TILE_LAYOUT)

    def compile_prefill(self, bucket):
        """Compile the bucket's prefill programs eagerly (before any trace capture) on a zero residual, including the
        hidden-row select; writes garbage into the pad block only."""
        t0 = time.perf_counter()
        pt_row = (
            self.page_tables[0].reshape(1, -1)
            if self.page_tables is not None
            else torch.zeros(1, self.sdpa_pt_blocks, dtype=torch.int32)
        )
        self._stage_prefill_inputs(
            bucket,
            torch.zeros(1, bucket, dtype=torch.int32),
            torch.full((1, bucket // self.block_size), self.pad_block, dtype=torch.int32),
            pt_row,
            0,
        )
        dummy = self._up(
            torch.zeros(1, 1, bucket, self.dim // self.model.num_devices, dtype=torch.bfloat16),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
        )
        h_full, y = self.prefill_forward(bucket, dummy)
        self._sync()
        self._select_row(bucket, h_full, bucket - 1)
        for t in (h_full, y, dummy):
            ttnn.deallocate(t)
        logger.info(f"[mtp] prefill programs compiled for bucket {bucket} in {time.perf_counter() - t0:.1f}s")

    def prefill_hook(self, user_ctx, hidden, token_buf, actual_len, bucket, chunk_start):
        """model.prefill_hidden_hook: fill the head's KV over one prefilled segment of the request being prefilled.

        user_ctx: ``(u, slot)`` or ``(u, slot, page_table_row)`` (model.prefill_paged_slots passes the request's own
        [1, blocks] row; without one the head's ``page_tables[slot]`` is used). hidden: the segment's pre-final-norm
        residual [1,1,bucket,dim/TP] (traced-bucket / chunk-trace output or eager tensor). token_buf [1, >= n]: the
        segment's n = actual_len real tokens (then bucket padding) -- or, for a FULL chunk followed by more prompt,
        bucket + 1 tokens whose extra one is the next chunk's first token (model.py's chunked paths pass it): the head then
        fills ALL n positions of the segment; a final segment fills n - 1 and stores the main hidden row n - 1 for
        the first draft step (``pending_rows[slot]``). chunk_start: the segment's absolute position."""
        if user_ctx is None:
            return  # warm-up / dummy prefills outside prefill_paged_slots
        if bucket not in self.pf:
            raise KeyError(f"MTP prefill not compiled for bucket {bucket} (have {sorted(self.pf)})")
        n = int(actual_len)
        slot = int(user_ctx[1])
        pt_row = user_ctx[2] if len(user_ctx) > 2 and user_ctx[2] is not None else self.page_tables[slot]
        pt_row = torch.as_tensor(pt_row).reshape(1, -1).to(torch.int32)
        # a chunk followed by more prompt carries bucket + 1 tokens; a final segment's token_buf is at most the
        # bucket wide (the masked bucket pads it to the bucket, so `> n` would misread a padded tail as non-final)
        has_next = int(token_buf.shape[1]) > bucket
        t0 = time.perf_counter()
        # tokens shifted by one: x_{cs+1}.. at positions cs.. (a final segment's row n-1 = the first draft step)
        n_fill = n if has_next else n - 1
        tok_shift = torch.zeros(1, bucket, dtype=torch.int32)
        tok_shift[0, :n_fill] = token_buf[0, 1 : 1 + n_fill].to(torch.int32)
        if n_fill > 0:
            fill = mbt.fill_pt_row(pt_row, int(chunk_start), n_fill, bucket, self.pad_block, self.block_size)
        else:  # a one-token final segment: nothing to fill, only the hidden row
            fill = torch.full((1, bucket // self.block_size), self.pad_block, dtype=torch.int32)
        self._stage_prefill_inputs(bucket, tok_shift, fill, pt_row, chunk_start)
        h_full, y = self.prefill_forward(bucket, hidden, chunk_start)
        self._sync()
        if not has_next:
            self.pending_rows[slot] = self._select_row(bucket, h_full, n - 1)
        if self.probe:
            rows = ttnn.to_torch(h_full, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3))
            rows = rows.reshape(-1, self.dim)[:n].to(torch.bfloat16)  # [n, dim] post-final-norm rows
            if not has_next:
                assert torch.equal(rows[n - 1], self.pending_rows[slot]), "hidden-row select != full readback"
            prev = self.probe_prefill.get(slot) if int(chunk_start) else None
            toks, hs = tok_shift[0, :n_fill].clone(), rows[:n_fill].clone()
            if prev is not None:
                toks, hs = torch.cat([prev[0], toks]), torch.cat([prev[1], hs])
            self.probe_prefill[slot] = (toks, hs)
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
        """positions[s] = -1 marks a PADDING row (no live request at slot s): the layer's paged KV update and SDPA
        skip a user at -1 (its outputs are garbage rows nothing mixes across users; the next select overwrites its
        hidden_in row)."""
        b = self.sb[w]
        tok = torch.tensor([int(t) for t in tokens], dtype=torch.int32).reshape(1, w)
        pos = torch.tensor([int(p) for p in positions], dtype=torch.int32)
        assert int(pos.min()) >= -1
        self._dma(tok, b.tok, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self._dma(pos, b.pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        rope_delta = int(getattr(self.model.rope, "rope_delta", 0) or 0)
        cos, sin = vg.rope_cos_sin(pos.clamp(min=0) + rope_delta, self.args.rope_head_dim, self.args.rope_theta)
        self._dma(cos, b.cos, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self._dma(sin, b.sin, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    def set_page_table(self, w, page_table):
        """Refresh the draft step's page table [w, blocks] (the users' current block rows); skipped when unchanged.
        Row s of a padding user should be zeros (the null block)."""
        b = self.sb[w]
        pt = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        pt = pt.to(torch.int32)
        assert tuple(pt.shape) == tuple(b.pt_host.shape), (tuple(pt.shape), tuple(b.pt_host.shape))
        if not torch.equal(pt, b.pt_host):
            b.pt_host = pt.clone()
            self._dma(pt.contiguous(), b.pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

    def upload_hidden_rows(self, w, rows_by_slot):
        """hidden_in[s] <- rows_by_slot[s] (bf16 [dim]) for the given slots, zeros elsewhere: the FIRST draft step of
        freshly admitted users (their imported / prefilled hidden row, ``set_hidden_in``) while the other rows are
        padding for that step."""
        rows = torch.zeros(w, self.dim, dtype=torch.bfloat16)
        for s, r in rows_by_slot.items():
            rows[int(s)] = torch.as_tensor(r).reshape(self.dim).to(torch.bfloat16)
        self._dma(rows.reshape(1, 1, w, self.dim).contiguous(), self.sb[w].hidden_in, ttnn.bfloat16, ttnn.TILE_LAYOUT)

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

    def draft(self, w, k, last, positions, observer=None, pad=None):
        """k chained drafts per user (schedule: verify_grid.chain_drafts). last[s] = the user's row-0 token (t'_s),
        positions[s] = P_s (the verify grid's committed position); hidden_in holds h at position P_s - 1
        (select_hidden / begin_batch). Step j writes the head's KV at P_s - 1 + j. observer(phase, j, tokens,
        positions, drafts_j) is called with phase "pre" before each step (hidden_in still holds the step's input) and
        "post" after it (probe). pad[s]: padding row (token 0 at position -1 every step, drafts zero)."""
        t0 = time.perf_counter()
        drafts = vg.chain_drafts(lambda tok, pos: self.run_step(w, tok, pos), w, k, last, positions, observer, pad=pad)
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
