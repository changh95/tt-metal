# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``MotifGenerator``: the Motif-3 TT runtime behind the vLLM bridge (``generator_api.MotifGenerator``; WAVE_A_REVIEW
GEN-1..7, design §2.3.10, §5.1). Default ``MOTIF3_GENERATOR_CLASS`` = ``models.demos.motif3.tt.generator:MotifGenerator``.

Lifecycle (vllm-tt-plugin call order, ``generator_api.MotifGenerator`` docstring):

1. ``create(hf_config=, mesh_device=, settings=)`` (GEN-1): ``model_config.require_l1_small(mesh)``, then
   ``MotifTTConfig.from_settings`` (``<weights>/config.json``, ``num_layers``, ``max_model_len``, ``max_batch = 32``, KV
   dtype, fabric from the device), then the weights (:class:`~models.demos.motif3.tt.model.MotifModel`: TT cache where
   converted, else the HF checkpoint, lazily). No KV pool yet.
2. ``allocate_kv_cache(num_blocks=, block_size=, num_layers=)`` (GEN-2): ``cfg.set_kv_geometry`` with the plugin's
   values (4129 x 64 for the serving defaults), one ``ttnn.empty`` + ``ttnn.fill(0)`` cache per layer.
3. ``warmup_prefill`` (every bucket of ``cfg.prefill_buckets``, eager, all writes into null block 0) ->
   ``warmup_decode(enable_trace=False)`` (stages the persistent decode inputs for width W, one eager all-inactive step)
   -> ``warmup_decode(enable_trace=True)`` (captures the decode trace: embed -> layers -> LM head, exception-safe).
   Capture refuses to run before every prefill bucket was compiled (a program compiled after capture can corrupt the
   trace: plugin ``model_runner.py:3735-3745``; LM head P1).
4. ``prefill_forward`` (GEN-4: one user, bucket = smallest power of 2 >= S, the request's own ``cdiv(S, block)``
   page-table entries then zeros up to ``cfg.prefill_page_table_entries(bucket)`` -- the bucket padding lands in the null
   block whatever stale ids the caller left there (:func:`prefill_page_table_host`) --, cache fill on every chip, host
   logits of S-1) and
   ``decode_forward`` (GEN-3: 32 lanes = 4 DP rows x 8; the host inputs are copied into the persistent device tensors,
   then the trace is replayed -- or the step runs eager when no trace was captured -- and the logits ``[32, 220160]``
   are assembled on the host from the 32 vocab shards per the LM head's "mesh" contract (``logits_to_host``, a fresh
   tensor every step).
5. ``release_traces`` / ``close`` (GEN-6). ``release_lane`` is a no-op in draft 1 (no lane-owned device state).

Lanes and inactive lanes (README §2): the bridge maps vLLM rows / state slots onto lanes (``LaneMap``, ``slot_remap``);
this class only sees lane-ordered tensors. Position ``-1`` marks an inactive lane: ``cur_pos = -1`` (no KV write,
FlashMLA skips it), rot index 0, an all-zero page-table row, token 0; its logits row is garbage (ignored).

Persistent decode inputs (one set, allocated before the capture; per DP row): ``tokens [4, 8]`` uint32,
``cur_pos [8]`` int32, ``rot_idxs [1, 32]`` uint32, ``page_table [8, W]`` int32 (ROW_MAJOR DRAM). Every step writes all
four with ``ttnn.copy_host_to_device_tensor`` (the bridge sends ``reload_inputs=True`` on every step).

Import rule (design §2.1): stdlib, torch, ttnn and the motif3 ``tt/`` modules only (the bridge's host suite imports this
module device-free: ``test_real_generator_class_imports_device_free``).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import torch

import ttnn

from . import generator_api as api
from .model import LazySource, MotifKVPool, MotifModel
from .model_config import MotifTTConfig, require_l1_small
from .rope import positions_to_rot_idxs, shard_lanes


def _log_default(msg: str) -> None:
    print(f"[motif3.generator {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prefill_page_table_host(page_table: torch.Tensor, entries: int, seq_len: int, block_size: int) -> torch.Tensor:
    """The prefill fill's page table ``[1, entries]`` int32 (``entries = cfg.prefill_page_table_entries(bucket)``): the
    request's first ``cdiv(seq_len, block_size)`` block ids, **zero (the null block) after them** whatever the caller
    passed there. The bucket-padding positions ``seq_len .. bucket-1`` are written through the entries past the
    request's own blocks; vLLM's persistent block-table rows keep stale ids there, often of blocks other live requests
    own (the bridge zeroes them too, ``generator_vllm._fit_page_table``; this keeps the runtime safe on its own)."""
    own = min(-(-int(seq_len) // int(block_size)), int(entries), int(page_table.shape[0]))
    pt = torch.zeros(1, int(entries), dtype=torch.int32)
    pt[0, :own] = page_table[:own].to(torch.int32)
    return pt


def _free(*ts) -> None:
    for t in ts:
        if t is not None:
            try:
                if t.is_allocated():
                    ttnn.deallocate(t)
            except Exception:
                pass


class MotifGenerator(api.MotifGenerator):
    """The Motif-3 runtime (see the module docstring). Build with :meth:`create` (vLLM) or directly from an existing
    :class:`MotifModel` (tests / demo): ``MotifGenerator(mesh_device, cfg, model)``."""

    def __init__(self, mesh_device, cfg: MotifTTConfig, model: MotifModel, *,
                 settings: Optional[api.GeneratorSettings] = None, log: Optional[Callable[[str], None]] = _log_default):
        if tuple(model.layer_ids) != tuple(range(len(model.layer_ids))):
            raise ValueError(f"the generator runs a prefix model (layers 0..N-1), got layers {model.layer_ids}")
        if cfg.max_batch != api.NUM_LANES:
            raise ValueError(f"cfg.max_batch must be {api.NUM_LANES} (the decode trace runs all lanes)")
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.model = model
        self.settings = settings
        self.log = log or (lambda m: None)
        self._pool: Optional[MotifKVPool] = None
        self._inputs: Optional[Dict[str, Any]] = None  # persistent decode inputs
        self._width: Optional[int] = None  # page-table width W of the persistent inputs
        self._trace_id = None
        self._trace_out = None
        self._trace_pool = None
        self._warmed_buckets = set()
        self._decode_warmed = False
        self.timings: Dict[str, float] = {}

    # ==============================================================================================================
    # construction (GEN-1)
    # ==============================================================================================================
    @classmethod
    def create(cls, *, hf_config: Any, mesh_device: Any, settings: api.GeneratorSettings, **model_kwargs) -> "MotifGenerator":
        """Build the runtime on the plugin's open mesh (``generator_api.MotifGenerator.create``). ``model_kwargs`` go
        to :class:`MotifModel` (``cache``, ``vocab_split``, ``layer_kwargs``, ``source``)."""
        log = model_kwargs.pop("log", _log_default)
        l1s = require_l1_small(mesh_device)
        cfg = MotifTTConfig.from_settings(settings, mesh_device=mesh_device, hf_config=hf_config)
        log(f"create: {cfg.describe()} (mesh L1_SMALL {l1s} B per core; weights {settings.weights_path} "
            f"[{settings.weights_source}])")
        if "source" not in model_kwargs:
            if settings.weights_are_local:
                model_kwargs["source"] = LazySource(settings.weights_path, log=log)
            elif settings.weights_path:  # an uncached repo id: only the shards of TT-cache misses, each disk-guarded
                model_kwargs["source"] = LazySource(repo_id=settings.weights_path, revision=settings.weights_revision,
                                                    log=log)
            else:
                model_kwargs["source"] = LazySource(cfg.weights_dir, log=log)
        t0 = time.time()
        model = MotifModel(mesh_device, cfg, layers=range(int(settings.num_layers)), log=log, **model_kwargs)
        log(f"create: {model.num_layers} layers loaded in {time.time() - t0:.1f} s")
        return cls(mesh_device, cfg, model, settings=settings, log=log)

    # ==============================================================================================================
    # static facts
    # ==============================================================================================================
    @property
    def num_layers(self) -> int:
        return self.model.num_layers

    @property
    def vocab_size(self) -> int:
        return int(self.cfg.vocab_size)

    @property
    def max_prefill_len(self) -> int:
        return int(self.cfg.prefill_buckets[-1])

    @property
    def trace_captured(self) -> bool:
        return self._trace_id is not None

    # ==============================================================================================================
    # KV pool (GEN-2)
    # ==============================================================================================================
    def allocate_kv_cache(self, *, num_blocks: int, block_size: int, num_layers: int) -> MotifKVPool:
        if self._pool is not None:
            raise RuntimeError("allocate_kv_cache called twice")
        if int(num_layers) != self.num_layers:
            raise ValueError(f"allocate_kv_cache for {num_layers} layers, the generator runs {self.num_layers}")
        self.cfg.set_kv_geometry(int(num_blocks), int(block_size))  # validates the block size (32 / 64)
        t0 = time.time()
        self._pool = self.model.allocate_kv_caches(int(num_blocks), int(block_size), self.cfg.dtypes.kv_cache)
        self.timings["allocate_kv_cache_s"] = time.time() - t0
        self.log(f"KV pool: {num_layers} x [{num_blocks}, 1, {block_size}, {self.cfg.kv_latent_dim}] "
                 f"{self.cfg.dtypes.kv_cache_name} ({self.cfg.kv_cache_bytes_per_chip() / 1e9:.2f} GB per chip) in "
                 f"{self.timings['allocate_kv_cache_s']:.1f} s")
        return self._pool

    def _check_pool(self, kv_cache) -> MotifKVPool:
        if self._pool is None:
            raise RuntimeError("the KV pool is not allocated (allocate_kv_cache has not run)")
        if kv_cache is not self._pool:
            raise ValueError("kv_cache must be the handle allocate_kv_cache returned")
        return self._pool

    # ==============================================================================================================
    # prefill (GEN-4)
    # ==============================================================================================================
    def _prefill_page_table(self, page_table: torch.Tensor, bucket: int, seq_len: int):
        """:func:`prefill_page_table_host` (the request's own ``cdiv(S, block)`` block ids, then zeros up to
        ``cfg.prefill_page_table_entries(bucket)``) as a replicated ``[1, n]`` int32 device tensor -- one fixed program
        shape per bucket (M10)."""
        n = self.cfg.prefill_page_table_entries(bucket)
        pt = prefill_page_table_host(page_table, n, seq_len, self.cfg.kv_block_size)
        return ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device))

    def prefill_forward(self, request: api.PrefillRequest, *, kv_cache: Any, enable_trace: bool = False) -> torch.Tensor:
        pool = self._check_pool(kv_cache)
        S = request.seq_len
        if S > self.max_prefill_len:
            raise ValueError(f"prompt of {S} tokens exceeds the largest prefill bucket {self.max_prefill_len}")
        bucket = self.cfg.prefill_bucket(S)
        if self._trace_id is not None and bucket not in self._warmed_buckets:
            # compiling a new prefill program after the decode capture can corrupt the trace (plugin contract)
            raise RuntimeError(f"prefill bucket {bucket} was not compiled before the decode trace capture")
        model = self.model
        tok = model.embed.prefill_tokens_device(request.tokens, bucket)
        pt = self._prefill_page_table(request.page_table, bucket, S)
        tile = None
        try:
            tile = model.prefill(tok, page_table=pt, kv_caches=pool, last_index=S - 1)
            logits = model.head.prefill_logits_to_host(tile, S - 1)
        finally:
            _free(tok, pt, tile)
        return logits

    # ==============================================================================================================
    # decode (GEN-3)
    # ==============================================================================================================
    def _host_inputs(self, batch: api.DecodeBatch) -> Dict[str, Any]:
        cfg, mesh = self.cfg, self.mesh_device
        pos = batch.positions.to(torch.int32)
        active = pos >= 0
        tokens = torch.where(active, batch.tokens.to(torch.int32), torch.zeros_like(pos))
        pt = torch.where(active[:, None], batch.page_table.to(torch.int32), torch.zeros_like(batch.page_table))
        return {
            "tokens": self.model.embed.decode_tokens_host(tokens),
            "cur": shard_lanes(pos.contiguous(), cfg, mesh, dtype=ttnn.int32, device=None),  # [8] per DP row
            "rot": shard_lanes(positions_to_rot_idxs(pos, cfg), cfg, mesh, dtype=ttnn.uint32, device=None),
            "pt": shard_lanes(pt.contiguous(), cfg, mesh, dtype=ttnn.int32, device=None),
        }

    def _stage_inputs(self, width: int) -> None:
        """(Re)allocate the persistent decode inputs for page-table width ``width`` (never once a trace exists)."""
        if self._inputs is not None and self._width == int(width):
            return
        if self._trace_id is not None:
            raise RuntimeError(f"page-table width {width} differs from the captured trace's {self._width}")
        self._free_inputs()
        batch = self._inactive_batch(width)
        h = self._host_inputs(batch)
        self._inputs = {k: ttnn.to_device(v, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        for k, v in h.items()}
        self._width = int(width)

    def _free_inputs(self) -> None:
        if self._inputs is not None:
            _free(*self._inputs.values())
        self._inputs, self._width = None, None

    def _inactive_batch(self, width: int) -> api.DecodeBatch:
        n = api.NUM_LANES
        return api.DecodeBatch(
            tokens=torch.zeros(n, dtype=torch.int32),
            positions=torch.full((n,), -1, dtype=torch.int32),
            page_table=torch.zeros(n, int(width), dtype=torch.int32),
        )

    def _write_inputs(self, batch: api.DecodeBatch) -> None:
        h = self._host_inputs(batch)
        for k in ("tokens", "cur", "rot", "pt"):
            ttnn.copy_host_to_device_tensor(h[k], self._inputs[k])

    def _decode_step(self, pool: MotifKVPool):
        d = self._inputs
        return self.model.decode(d["tokens"], rot_idxs=d["rot"], cur_pos=d["cur"], page_table=d["pt"],
                                 kv_caches=pool)

    def decode_forward(self, batch: api.DecodeBatch, *, kv_cache: Any, enable_trace: bool) -> torch.Tensor:
        pool = self._check_pool(kv_cache)
        if int(batch.positions.max()) >= self.cfg.max_model_len:
            raise ValueError(f"decode position {int(batch.positions.max())} >= max_model_len {self.cfg.max_model_len}")
        use_trace = bool(enable_trace) and self._trace_id is not None
        if use_trace and batch.page_table_width != self._width:
            raise ValueError(f"page-table width {batch.page_table_width} != the traced width {self._width}")
        self._stage_inputs(batch.page_table_width)
        self._write_inputs(batch)
        head = self.model.head
        if use_trace:
            if self._trace_pool is not pool:
                raise ValueError("the decode trace was captured with another KV pool")
            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=False)
            return head.logits_to_host(self._trace_out)  # blocking read of the trace output (fresh host tensor)
        out = self._decode_step(pool)
        try:
            return head.logits_to_host(out)
        finally:
            _free(out)

    # ==============================================================================================================
    # warmup (GEN-3 / GEN-4)
    # ==============================================================================================================
    def warmup_prefill(self, *, kv_cache: Any, enable_trace: bool) -> None:
        pool = self._check_pool(kv_cache)
        if enable_trace:  # plugin trace_mode="all": draft-1 prefill is eager
            return
        if self._trace_id is not None:
            raise RuntimeError("warmup_prefill after the decode trace capture (buckets must compile before it)")
        t_all = time.time()
        for b in self.cfg.prefill_buckets:
            t0 = time.time()
            req = api.PrefillRequest(
                lane=0,
                tokens=torch.full((b,), int(self.cfg.pad_token_id), dtype=torch.int32),
                page_table=torch.zeros(self.cfg.prefill_page_table_entries(b), dtype=torch.int32),  # null block only
            )
            self.prefill_forward(req, kv_cache=pool)
            self._warmed_buckets.add(b)
            self.timings[f"warmup_prefill_{b}_s"] = time.time() - t0
            self.log(f"warmup prefill bucket {b}: {self.timings[f'warmup_prefill_{b}_s']:.1f} s")
        self.timings["warmup_prefill_s"] = time.time() - t_all

    def warmup_decode(self, *, kv_cache: Any, enable_trace: bool, page_table_width: int) -> None:
        pool = self._check_pool(kv_cache)
        W = int(page_table_width)
        if W < 1:
            raise ValueError(f"page_table_width must be >= 1, got {W}")
        if not enable_trace:
            t0 = time.time()
            self._stage_inputs(W)
            self.decode_forward(self._inactive_batch(W), kv_cache=pool, enable_trace=False)
            self._decode_warmed = True
            self.timings["warmup_decode_eager_s"] = time.time() - t0
            self.log(f"warmup decode (eager, W={W}): {self.timings['warmup_decode_eager_s']:.1f} s")
            return
        missing = [b for b in self.cfg.prefill_buckets if b not in self._warmed_buckets]
        if missing:
            raise RuntimeError(f"decode trace capture before the prefill warmup: buckets {missing} not compiled")
        if self._trace_id is not None:
            if self._width != W:
                raise RuntimeError(f"a decode trace for width {self._width} exists; release_traces() first")
            return
        if not self._decode_warmed or self._width != W:
            self.warmup_decode(kv_cache=pool, enable_trace=False, page_table_width=W)
        t0 = time.time()
        self._write_inputs(self._inactive_batch(W))  # the capture's own replay below must write nothing
        ttnn.synchronize_device(self.mesh_device)
        self._trace_id, self._trace_out = self._capture(pool)
        self._trace_pool = pool
        ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=True)  # one replay: all lanes inactive
        self.timings["capture_decode_s"] = time.time() - t0
        self.log(f"decode trace captured (W={W}, {self.num_layers} layers) in {self.timings['capture_decode_s']:.1f} s")

    def _capture(self, pool: MotifKVPool):
        """Exception-safe capture of one decode step (a dangling capture hung close_mesh_device once, GATES_RESULTS
        §11.6): on any error the capture is ended and released before the error propagates."""
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        try:
            out = self._decode_step(pool)
        except BaseException:
            try:
                ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            except Exception:
                pass
            try:
                ttnn.release_trace(self.mesh_device, tid)
            except Exception:
                pass
            raise
        try:
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        except BaseException:
            try:
                ttnn.release_trace(self.mesh_device, tid)
            except Exception:
                pass
            _free(out)
            raise
        return tid, out

    # ==============================================================================================================
    # lifecycle (GEN-6)
    # ==============================================================================================================
    def release_traces(self) -> None:
        if self._trace_id is not None:
            try:
                ttnn.release_trace(self.mesh_device, self._trace_id)
            finally:
                self._trace_id = None
                _free(self._trace_out)
                self._trace_out = None
                self._trace_pool = None

    def close(self) -> None:
        """Release the trace, the persistent inputs, the KV pool and the model weights (standalone runs)."""
        self.release_traces()
        self._free_inputs()
        if self._pool is not None:
            self._pool.deallocate()
            self._pool = None
        self.model.deallocate()


__all__ = ["MotifGenerator", "prefill_page_table_host"]
