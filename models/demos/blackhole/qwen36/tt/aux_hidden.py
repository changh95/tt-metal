# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Auxiliary hidden states of the target model for the DFlash2 drafter (z-lab/Qwen3.8-27B-DFlash2), and the
drafter-selection knob of the speculative-decoding stack.

DFlash2's draft model does not run its own layers over the prompt: the CONTEXT K/V of each of its 5 layers are
projections of the target model's auxiliary hidden states (``fc`` over the concatenation of ``target_layer_ids``
hidden states -> ``hidden_norm`` -> per-layer k/v projection -> k_norm -> RoPE(position); vLLM
``qwen3_dflash.py::precompute_and_store_context_kv``). The drafter therefore needs, for every committed position,
the 5 aux hidden rows of the target -- from the prefill on P (tt/model.py ``prefill_aux_hook``, per 2048-token
chunk / masked bucket) and from every verify step on D (tt/verify_step.py ``VerifyStep(keep_aux_hidden=True)`` ->
``plan.out_aux``).

Semantics, pinned against the vLLM in the plugin venv (models/qwen3_next.py ``Qwen3NextModel.forward`` +
interfaces.py ``_maybe_add_hidden_state``, worker/gpu_model_runner.py ``_get_eagle3_aux_layers_from_config``):
  * ``aux_hidden_state_layers`` index ``j`` = the residual stream after ``j`` decoder layers have run, i.e.
    ``hidden_states + residual`` right after ``layers[j - 1]`` (0-based), the FULL post-MLP residual (index 0 =
    the embeddings). vLLM sets ``aux_hidden_state_layers = tuple(i + 1 for i in dflash_config.target_layer_ids)``
    ("Add 1 to convert DFlash's aux layer id semantics"), so DFlash2's ``target_layer_ids`` are 0-BASED layer indices
    whose OUTPUT residual is taken: ``[5, 19, 33, 47, 61]`` = the outputs of ``model.layers[5]``, ``[19]``, ``[33]``,
    ``[47]``, ``[61]`` (all Gated DeltaNet layers here: the full-attention layers are 3, 7, 11, ...). In this code
    base that is the ``x`` returned by ``Qwen36DecoderLayer.forward`` / ``forward_verify`` for those ``layer_num``s
    (the residual add of the MLP output is the layer's last op), before the next layer's input norm.
  * Concatenation order = ``target_layer_ids`` order: row = ``[h_5 | h_19 | h_33 | h_47 | h_61]``, ``5 * 5120``.
  * The rows are the target's bf16 activations as computed (no norm applied; DFlash2's ``fc`` takes them raw).

Layouts this module hands around (TP mesh, ``n_dev`` devices):
  * REPLICATED ``[1, 1, N, n_aux * dim]`` bf16 DRAM: every device holds the full rows (the verify step's
    ``plan.out_aux``; ``gather_aux_replicated`` builds it from the prefill's fractured copies).
  * FRACTURED list of ``n_aux`` tensors ``[1, 1, S, dim / n_dev]`` bf16 DRAM: device ``d`` holds columns
    ``[d * dim / n_dev, (d + 1) * dim / n_dev)`` of aux ``j`` (the prefill's residual layout; what the model's
    ``prefill_aux_hook`` receives).

Knob: ``QWEN36_SPEC_DRAFTER=mtp|dflash2`` (``spec_drafter``, default ``mtp``) selects which drafter state the
prefill side computes and ships (tt/qwen36_vllm.py ``_install_spec_prefill_hook``, tt/pd_transfer.py kv groups):
``mtp`` = the MTP head's 17th KV layer + hidden row (unchanged path), ``dflash2`` = the DFlash2 context K/V of the
prompt (``KV group "dflash2"`` of the version-3 payload). Unset / ``mtp`` leaves every byte of the served path as
it was.
"""
import os
import time

import torch
from loguru import logger

import ttnn

# DFlash2 (z-lab/Qwen3.8-27B-DFlash2 config.json dflash_config.target_layer_ids): 0-based target layer indices
# whose output residual feeds the drafter (see the module docstring for the vLLM index convention).
DFLASH2_TARGET_LAYERS = (5, 19, 33, 47, 61)
DFLASH2_KV_HEADS = 8  # num_key_value_heads of the draft model
DFLASH2_HEAD_DIM = 128
DFLASH2_N_LAYERS = 5
DFLASH2_SLIDING_WINDOW = 2048  # config.json sliding_window (every draft layer is "sliding_attention")


def spec_drafter() -> str:
    """``QWEN36_SPEC_DRAFTER``: ``mtp`` (default) or ``dflash2``."""
    v = os.environ.get("QWEN36_SPEC_DRAFTER", "mtp").strip().lower() or "mtp"
    if v not in ("mtp", "dflash2"):
        raise ValueError(f"QWEN36_SPEC_DRAFTER={v!r}: expected mtp or dflash2")
    return v


def dflash2_selected() -> bool:
    return spec_drafter() == "dflash2"


def dflash2_context_window() -> int:
    """How many trailing prompt positions of DFlash2 context K/V the prefill side computes and ships
    (``QWEN36_DFLASH2_CONTEXT_WINDOW``; 0 = every position, the default). Every draft layer of Qwen3.8-27B-DFlash2 is a
    sliding-attention layer of window 2048 in the reference, so a value of 2048 ships only what the reference drafter
    can read (an 8k prompt: 32 of 128 blocks, 40 MiB instead of 160 MiB of bf16 context K/V) -- turn it on once the
    device drafter (tt/dflash2_head.py) applies the window; today it attends the whole context, so the default ships
    every position."""
    v = os.environ.get("QWEN36_DFLASH2_CONTEXT_WINDOW")
    return 0 if v in (None, "") else int(v)


def aux_layers_for(model, layers=DFLASH2_TARGET_LAYERS):
    """The aux layer tuple validated against the model (0-based ``layer_num`` of ``model.layers``)."""
    n = len(model.layers)
    out = tuple(int(i) for i in (DFLASH2_TARGET_LAYERS if layers is None else layers))
    bad = [i for i in out if not 0 <= i < n]
    assert not bad, f"aux layers {bad} out of range for {n} layers"
    assert len(set(out)) == len(out), f"duplicate aux layers {out}"
    return out


def all_gather_copy(model, x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """All-gather a FRACTURED [1,1,S,dim/n_dev] tensor along the hidden dim into a new REPLICATED [1,1,S,dim] tensor
    (the call tt/layer.py _verify_norm_blocks makes; pure data movement). The input is left alone -- unlike
    tt_transformers' tt_all_gather, which deallocates it -- so it can wrap the live residual or a persistent buffer.
    """
    tt_ccl, args = model.tt_ccl, model.args
    return ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=tt_ccl.get_num_links(1),
        topology=args.ccl_topology(),
        memory_config=memory_config,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )


def gather_aux_replicated(model, aux_frac, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """FRACTURED aux copies (list of ``[1,1,S,dim/n_dev]``) -> one REPLICATED ``[1,1,S,n_aux*dim]`` bf16 tensor
    (row = ``[h_a | h_b | ...]`` in list order): one all-gather per aux (exact data movement, no arithmetic) and a
    concat. Eager (a P-side prefill hook runs it between trace replays); the caller deallocates the result."""
    parts = []
    for x in aux_frac:
        x4 = x if len(x.shape) == 4 else ttnn.reshape(x, (1, 1, x.shape[-2], x.shape[-1]))
        parts.append(all_gather_copy(model, x4, memory_config))
    if len(parts) == 1:
        return parts[0]
    out = ttnn.concat(parts, dim=-1, memory_config=memory_config)
    for p in parts:
        ttnn.deallocate(p)
    return out


def aux_rows_to_host(model, aux, n_valid=None, fractured=False):
    """Host bf16 ``[n_valid, n_aux * dim]`` rows of an aux tensor: a REPLICATED ``[1,1,N,n_aux*dim]`` (device 0's
    copy) or, with ``fractured=True``, a list of fractured ``[1,1,S,dim/n_dev]`` aux copies (each gathered along the
    hidden dim on the host and concatenated in list order)."""
    if fractured:
        comp = ttnn.ConcatMeshToTensor(model.mesh_device, dim=3)
        cols = [ttnn.to_torch(x, mesh_composer=comp).to(torch.bfloat16) for x in aux]
        cols = [c.reshape(-1, c.shape[-1]) for c in cols]
        rows = torch.cat(cols, dim=-1)
    else:
        rows = ttnn.to_torch(ttnn.get_device_tensors(aux)[0]).to(torch.bfloat16)
        rows = rows.reshape(-1, rows.shape[-1])
    if n_valid is not None:
        rows = rows[: int(n_valid)]
    return rows.clone()


# ================================================================================================ P-side prefill hook
class KvGroupStage:
    """P-side host staging of one request's drafter context K/V (tt/pd_transfer.py ``export_kv_groups`` ships it):
    per drafter layer a ``(K, V)`` pair of host bf16 ``[n_tokens, kv_heads, head_dim]`` rows in position order
    (row i = prompt position ``first_pos + i``), grown chunk by chunk as the prefill hook fires."""

    __slots__ = ("n_layers", "kv_heads", "head_dim", "first_pos", "parts", "n_tokens")

    def __init__(self, n_layers, kv_heads, head_dim, first_pos=0):
        self.n_layers, self.kv_heads, self.head_dim = int(n_layers), int(kv_heads), int(head_dim)
        self.first_pos = int(first_pos)
        self.parts = [[] for _ in range(self.n_layers)]  # per layer: list of (K, V) chunk pieces
        self.n_tokens = 0

    def append(self, kv_pairs, n_valid, position):
        """kv_pairs[j] = (K, V) host ``[S, kv_heads, head_dim]`` of segment rows starting at prompt ``position``;
        rows ``>= n_valid`` are padding and dropped. Segments must arrive in position order without gaps."""
        n_valid = int(n_valid)
        expected = self.first_pos + self.n_tokens
        assert int(position) == expected, f"KV group segment at position {position}, expected {expected}"
        assert len(kv_pairs) == self.n_layers, (len(kv_pairs), self.n_layers)
        for j, (k, v) in enumerate(kv_pairs):
            k = torch.as_tensor(k)[:n_valid].to(torch.bfloat16)
            v = torch.as_tensor(v)[:n_valid].to(torch.bfloat16)
            assert tuple(k.shape) == (n_valid, self.kv_heads, self.head_dim), (tuple(k.shape), n_valid)
            assert tuple(v.shape) == (n_valid, self.kv_heads, self.head_dim), (tuple(v.shape), n_valid)
            self.parts[j].append((k.contiguous(), v.contiguous()))
        self.n_tokens += n_valid

    def rows(self):
        """Per layer ``(K, V)`` host bf16 ``[n_tokens, kv_heads, head_dim]`` (concatenated once)."""
        out = []
        for j in range(self.n_layers):
            ks = [k for k, _ in self.parts[j]]
            vs = [v for _, v in self.parts[j]]
            out.append((torch.cat(ks) if len(ks) != 1 else ks[0], torch.cat(vs) if len(vs) != 1 else vs[0]))
        return out


def kv_group_stage_store(model, name="dflash2"):
    """The model's P-side staging dict of KV group ``name``: decode slot -> ``KvGroupStage`` (created on demand)."""
    store = getattr(model, "pd_kv_group_stage", None)
    if store is None:
        store = model.pd_kv_group_stage = {}
    return store.setdefault(name, {})


class DFlash2ContextPrefillHook:
    """``model.prefill_aux_hook`` on a P (or standalone) instance with ``QWEN36_SPEC_DRAFTER=dflash2``: turn every
    prefilled segment's aux hidden rows into the DFlash2 drafter's CONTEXT K/V and stage them per decode slot for the
    P/D transport (``KvGroupStage``; tt/pd_transfer.py ``export_kv_groups`` -> payload KV group ``"dflash2"``).

    Called by tt/model.py (see ``Qwen36Model.prefill_aux_hook``) with ``(user_ctx, aux_frac, token_buf, actual_len,
    bucket, chunk_start)``: ``aux_frac`` = the segment's FRACTURED aux copies (``len(model.prefill_aux_layers)``
    tensors ``[1,1,bucket,dim/n_dev]``, rows ``>= actual_len`` padding), ``chunk_start`` = the segment's absolute
    position; ``user_ctx`` = ``(u, slot, page_table_row[, total_len])`` of the request being prefilled
    (``prefill_paged_slots``), None outside it (warm-ups: ignored).

    Projector contract (tt/dflash2_head.py ``DFlash2ContextProjector``, built by the drafter's owner; the stub in
    tests/dflash2_stub.py has the same interface): ``[(K_0, V_0), ..., (K_4, V_4)]`` with each ``K_j`` / ``V_j`` a
    HOST torch bf16 ``[S, kv_heads, head_dim]`` in GLOBAL kv-head order (heads ``[2d, 2d+1]`` of a TP=4 mesh come
    from device d), ``positions`` a host int ``[S]`` (``chunk_start + i``; the RoPE positions of the context keys),
    rows ``>= n_valid`` padding the projector may compute on and the hook drops. The hook picks the first of:
      * ``project_device(aux_rep, positions)`` -- ``aux_rep`` the REPLICATED device tensor ``[1,1,S,n_aux*dim]`` bf16
        (row = ``[h_5|h_19|h_33|h_47|h_61]``; the hook all-gathers the fractured copies); no host round trip of the
        105 MB/chunk aux rows -- the fast path the drafter's owner can expose (``DFlash2Drafter.project_kv`` is fed
        exactly this tensor);
      * ``project_fractured(aux_frac, positions)`` -- the fractured list itself (device d holds columns
        ``[d*dim/n_dev, (d+1)*dim/n_dev)`` of every aux; skips the all-gather too);
      * ``project(aux, positions)`` -- ``aux`` a HOST torch bf16 ``[S, n_aux*dim]`` (``DFlash2ContextProjector.project``
        as built: it uploads the rows itself; the hook gathers + reads them back, ~2 x 105 MB per 2048 chunk).

    Window: with ``total_len`` in ``user_ctx`` and a finite ``dflash2_context_window()``, segments that end before
    ``total_len - window`` (rounded down to a KV block) are skipped -- the drafter never reads them -- and the stage's
    ``first_pos`` is the first shipped position; ``export_kv_groups`` ships exactly the blocks covering
    ``[first_pos, total_len)``.
    """

    def __init__(
        self, model, projector, name="dflash2", block_size=64, kv_heads=DFLASH2_KV_HEADS, head_dim=DFLASH2_HEAD_DIM
    ):
        self.model = model
        self.projector = projector
        self.name = name
        self.block_size = int(block_size)
        self.kv_heads, self.head_dim = int(kv_heads), int(head_dim)
        self.n_layers = DFLASH2_N_LAYERS
        self.window = dflash2_context_window()
        self.stats = {"calls": 0, "skipped": 0, "wall": 0.0, "gather": 0.0, "project": 0.0, "tokens": 0}
        self.store = kv_group_stage_store(model, name)

    def window_start(self, total_len):
        """First prompt position whose context K/V is shipped for a prompt of ``total_len`` tokens (block aligned)."""
        if self.window <= 0 or total_len is None or int(total_len) <= self.window:
            return 0
        return ((int(total_len) - self.window) // self.block_size) * self.block_size

    def __call__(self, user_ctx, aux_frac, token_buf, actual_len, bucket, chunk_start):
        if user_ctx is None:
            return
        t0 = time.perf_counter()
        slot = int(user_ctx[1])
        total_len = int(user_ctx[3]) if len(user_ctx) > 3 and user_ctx[3] is not None else None
        n = int(actual_len)
        cs = int(chunk_start)
        first = self.window_start(total_len)
        if cs == 0 or slot not in self.store:
            # a new request in this slot (or a request whose earlier segments were all outside the window)
            self.store[slot] = KvGroupStage(self.n_layers, self.kv_heads, self.head_dim, first_pos=max(first, cs))
        stage = self.store[slot]
        if cs + n <= first:
            self.stats["skipped"] += 1
            stage.first_pos = max(stage.first_pos, cs + n)  # the next segment continues from here
            return
        positions = torch.arange(cs, cs + int(bucket), dtype=torch.int32)
        t1 = time.perf_counter()
        on_device = getattr(self.projector, "project_device", None)
        fractured = getattr(self.projector, "project_fractured", None)
        if fractured is not None and on_device is None:
            t2 = time.perf_counter()
            kv = fractured(aux_frac, positions)
        else:
            aux_rep = gather_aux_replicated(self.model, aux_frac)
            ttnn.synchronize_device(self.model.mesh_device)
            if on_device is None:
                rows = aux_rows_to_host(self.model, aux_rep)  # [S, n_aux*dim] bf16 host
                ttnn.deallocate(aux_rep)
                t2 = time.perf_counter()
                kv = self.projector.project(rows, positions)
            else:
                t2 = time.perf_counter()
                kv = on_device(aux_rep, positions)
                ttnn.deallocate(aux_rep)
        t3 = time.perf_counter()
        skip = max(0, stage.first_pos - cs)  # rows of this segment before the window start
        if skip:
            kv = [(k[skip:], v[skip:]) for k, v in kv]
        stage.append(kv, n - skip, cs + skip)
        self.stats["calls"] += 1
        self.stats["tokens"] += n - skip
        self.stats["gather"] += t2 - t1  # all-gather (+ host readback on the host-facing projector path)
        self.stats["project"] += t3 - t2
        self.stats["wall"] += time.perf_counter() - t0
        logger.debug(
            f"[dflash2] slot {slot}: context K/V for positions [{cs + skip}, {cs + n}) staged "
            f"(gather {1e3 * (t2 - t1):.1f} ms, project {1e3 * (t3 - t2):.1f} ms)"
        )

    def pop(self, slot):
        """The staged context K/V of ``slot`` (a ``KvGroupStage``) or None; taken (the export consumes it)."""
        return self.store.pop(int(slot), None)

    def compile(self, buckets):
        """Compile-first rule (tests/VERIFY_W32_AUDIT.md): run the hook's device programs once per segment width
        (every masked bucket + the 2048 chunk) on zero aux copies BEFORE the prefill warm-up captures any trace --
        the all-gathers / concat of ``gather_aux_replicated`` and the projector's ops otherwise compile at request
        time (a 4196-token request paid 1.4 s of gather + 1.4 s of projector JIT in the first run). Stages nothing."""
        t0 = time.perf_counter()
        n_aux = len(self.model.prefill_aux_layers)
        dim_tp = self.model.args.dim // self.model.num_devices
        rep = ttnn.ReplicateTensorToMesh(self.model.mesh_device)
        for bucket in sorted(set(int(b) for b in buckets)):
            aux = [
                ttnn.from_torch(
                    torch.zeros(1, 1, bucket, dim_tp, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.model.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=rep,
                )
                for _ in range(n_aux)
            ]
            positions = torch.arange(bucket, dtype=torch.int32)
            on_device = getattr(self.projector, "project_device", None)
            fractured = getattr(self.projector, "project_fractured", None)
            if fractured is not None and on_device is None:
                fractured(aux, positions)
            else:
                aux_rep = gather_aux_replicated(self.model, aux)
                if on_device is None:
                    self.projector.project(aux_rows_to_host(self.model, aux_rep), positions)
                else:
                    on_device(aux_rep, positions)
                ttnn.deallocate(aux_rep)
            ttnn.synchronize_device(self.model.mesh_device)
            for t in aux:
                ttnn.deallocate(t)
        logger.info(
            f"[dflash2] prefill hook programs compiled for buckets {sorted(set(buckets))} in {time.perf_counter() - t0:.1f}s"
        )
