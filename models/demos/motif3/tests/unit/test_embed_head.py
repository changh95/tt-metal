# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Embedding + final head (``tt/embedding.py``, ``tt/lm_head.py``; design §2.3.2, §2.3.8-2.3.9; WAVE_A_REVIEW EMB-1..3)
vs the CPU reference (``models/demos/motif3/reference``).

CPU tests (no device; run through the host wrapper with --noconftest)::

    scripts/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib \
        -q models/demos/motif3/tests/unit/test_embed_head.py -k cpu

* ``test_cpu_lm_head_layouts``: the "mesh" / "tp" LM-head host layouts give every mesh coordinate exactly its vocab
  block under the emulated mappers, on (4, 8) and (8, 4); "tp" equals ``weights.lm_head_for_chip``.
* ``test_cpu_assemble_offsets_tokens`` / ``test_cpu_concat_views`` / ``test_cpu_prefill_row_pick``: host logits
  assembly (reference and the numpy path, inline and threaded), the prefill tile-row pick (row ``last_index % 32``,
  fresh tensor), the argmax offsets table, decode / prefill token packing and id range checks.
* ``test_cpu_logits_dtype``: logits come out in the activation dtype whatever the LM-head weight dtype (a bfp8 head
  weight keeps bf16 logits); block-float logits are rejected.
* ``test_cpu_stream_sum_eps_identity``: ``rms_norm(sum_4 x; 16 eps) == rms_norm(mean_4 x; eps)`` **bit-exactly** in the
  reference's bf16 numerics (reference RMSNorm module in bf16), and to fp32 rounding in fp32.
* ``test_cpu_import_rule``: no other models/demos (or vllm / transformers / safetensors) import at module import time.
* ``test_cpu_t64_tokens_offsets_configs`` (T64, docs/p5_t64/P5_T64_DESIGN.md §4.1-§4.2): ``decode_token_rows(
  rows_per_dp=16)``, the 64-column argmax offsets table, the 64-row LM-head GEMM config (== ``model_config.lm_head_pc(
  m_tiles=2)``, the G16-lite config).
* ``test_cpu_prepare_real_streams``: writes the layer-1 real-stream golden (reference prefix model, layers 0-1,
  default chat prompt, 145 tokens) to ``tt_cache/test/embed_head/`` (5 MB). The device tests also build it on the fly
  when it is missing.

Device tests (only through the lock wrapper; ~3 min). Run them with the trace-allocation tracker on: every trace replay
then also proves that no device buffer allocated after a capture (including a program compiled after it) is alive::

    scripts/devrun.sh -t 1500 -n embed_head -- env TT_METAL_TRACE_ALLOC_TRACKING=1 python -m pytest \
        models/demos/motif3/tests/unit/test_embed_head.py -s -p no:cacheprovider -k "not cpu and not probe and not profile"

* ``test_embedding``: real table; decode X [1,4,8,4096] bit-exact per DP row on all 32 chips, replicas identical,
  trace replay with new tokens, device greedy feedback (argmax ids -> next decode input; traced cost by a 64 vs 256
  call slope); prefill (S = 128 / 256 / 1024, every prefill mode) bit-exact and identical on 32 chips; hidden-sharded
  table; eager / traced latency.
* ``test_lm_head[mesh|tp]``: real final norm + lm_head on the real layer-1 streams: final-norm output vs the reference
  RMSNorm (README §12 >= 0.9999), decode logits [32, 220160] and prefill last-token logits at several positions per
  bucket (tile edges, new tile rows / in-tile rows, no new program per position) vs the reference (bf16 model numerics
  and the fp32 ideal; PCC, max-abs, per-lane PCC, top-1), host assembly vs the per-chip tensors, device argmax ==
  torch.argmax (+ crafted ties), head / argmax / transfer latency, a traced decode tail replayed with new streams.
* ``test_lm_head_late_layers``: real late-layer streams from the reference golden-stream run (after layers 8 / 16 / 35;
  stream RMS up to ~44 and |x| up to ~4200 against 0.05 / 1.3 after layer 1): stream-sum rounding, final norm vs the
  reference and vs the run's own ``final_hidden``, decode logits vs the reference and the run's top-32 logits /
  argmax (tie-aware), and prefill for all 6 prompts at 8 positions each vs the run's per-position golden.
* ``test_prefill_head_trace_safety``: the review-P1 regression test: every bucket warmed once, then a prefill-head
  trace captured at one position and replayed at others (position on the device), and a decode trace captured,
  prefill run at ~30 unwarmed positions in 3 buckets, and the decode trace replayed bitwise-identically with no new
  program-cache entry (and, with the tracker on, no corruptible buffer); a literal-start slice is the negative control.
* ``test_lm_head_random_weights``: random weights at real dims, both stream reductions and fp32 logits; decode, device
  argmax, prefill.
* ``test_embed_head_8x4``: the (8, 4) orientation (plugin BH-Galaxy preset) for embedding, both vocab splits.
* ``test_embed_head_t64`` (WP-D D1; T64 verify step, 16 rows per DP row; globals from the serving TT cache): the
  split-order / natural-order ``ag_dp_rows`` at 16 rows, the 16-row embedding and split-order id embedding, the
  16-row final norm, ``decode_logits(halves=2)``, the 64-row argmax, ``logits_rm(rows=32)``, all bitwise == the
  32-lane calls on all 32 chips; the 32-lane head == B0's committed module; a traced 16-row head == eager; the
  static-CB end below an L1 pin; traced cost of the 64-row pieces (``-k t64``).

Real-input policy: the real-weight tests skip only when the checkpoint is not local. The layer-1 stream golden is built
on the fly if missing (never replaced by random streams); the late-layer test skips, naming the missing files, when the
golden-stream run (another workflow's read-only artifact, ``MOTIF3_GOLDEN_STREAM_DIR``) is absent.

Opt-in device probes: ``MOTIF3_EH_PROBE=1`` (``test_probe_transfer``: per-chip PCIe read bandwidth (x1 vs the 4 x8
chips), concurrency, every logits readback variant, host-copy thread contention, prefill new-position cost) and
``MOTIF3_EH_PROFILE=1`` together with the device-profiler env (``test_profile_head``: device kernel durations of every
embedding / head op, the LM-head program-config sweep, norm grids, both argmax local stages and the prefill head;
immune to host load)::

    scripts/devrun.sh -t 1500 -n eh_profile -- env MOTIF3_EH_PROFILE=1 TT_METAL_DEVICE_PROFILER=1 \
        TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 \
        python -m pytest models/demos/motif3/tests/unit/test_embed_head.py -s -p no:cacheprovider -k profile

Every device test logs the committed fabric topology first (``[motif3.fabric] ... committed:``) and the host load
average (host-timed numbers are sensitive to other agents' CPU jobs on this shared host).
"""

import os
import time
from pathlib import Path

import pytest
import torch

TEST_CACHE = Path("/home/ttuser/hchang/experiments/motif-3/tt_cache/test/embed_head")
STREAMS_FILE = TEST_CACHE / "real_streams_L01_default_prompt.pt"
# Reference golden-stream run (reference/golden_stream.py; read only, owned by the reference workflow): 4-stream residuals
# after layers 0-4, 7, 8, 15, 16, 23, 24, 31, 32, 35 for 6 prompts (2971 tokens) and the reference's own final head after
# layer 35 (final_hidden = RMSNorm output, top-32 logits / argmax per position).
GOLDEN_STREAM_DIR = Path(
    os.environ.get("MOTIF3_GOLDEN_STREAM_DIR", "/home/ttuser/hchang/experiments/motif-3/goldens/c2")
)
LATE_LAYERS = (8, 16, 35)  # first big residual jump (RMS 3), mid (RMS 11), last local layer (RMS 44)
HEAD_GOLDEN_LAYER = 35
TRACE = 64 * 1024 * 1024


# ============================================================================================================
# helpers (CPU)
# ============================================================================================================
def _pcc(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def _cpu_cfg(**kw):
    from models.demos.motif3.tt.model_config import MotifTTConfig

    return MotifTTConfig(**kw)


class _RefHead:
    """Reference final head on streams ``[N, 4, 4096]`` (rows = tokens) from the reference modules:
    ``MotifModel.reduce_streams`` (mean over the streams) -> ``RMSNorm`` -> ``lm_head(h).float()``. ``fp32=False``: the
    bf16 model numerics (HF's cast points; bit-identical to the golden-stream run's ``final_hidden``); ``fp32=True``:
    the fp32 ideal (bf16 weights / streams upcast). The fp32 copy of the LM head (3.6 GB) is made once, lazily."""

    def __init__(self, gamma: torch.Tensor, lm_head: torch.Tensor, eps: float):
        self.gamma = gamma
        self.lm16 = lm_head.to(torch.bfloat16)
        self.eps = float(eps)
        self._lm32 = None

    def norm(self, x_streams: torch.Tensor, fp32: bool) -> torch.Tensor:
        from models.demos.motif3.reference.modules import RMSNorm

        dt = torch.float32 if fp32 else torch.bfloat16
        norm = RMSNorm(x_streams.shape[-1], self.eps).to(dt)
        with torch.no_grad():
            norm.weight.copy_(self.gamma.to(dt))
            return norm(x_streams.to(dt).mean(dim=1))  # reduce_streams on [N, 4, D] (stream dim 1 here)

    def logits(self, x_streams: torch.Tensor, fp32: bool) -> torch.Tensor:
        h = self.norm(x_streams, fp32)
        if fp32 and self._lm32 is None:
            self._lm32 = self.lm16.float()
        with torch.no_grad():
            return torch.nn.functional.linear(h, self._lm32 if fp32 else self.lm16).float()


def _real_source_or_skip():
    from models.demos.motif3.tt.weights import HFWeightLoader

    try:
        src = HFWeightLoader()
    except FileNotFoundError as e:
        pytest.skip(f"no local checkpoint: {e}")
    for n in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        if not src.available(n):
            pytest.skip(f"{n} is not local (shards 1 / 104)")
    return src


def _real_streams(create: bool = False):
    """``{"ids": [S], "x": [S, 4, 4096] bf16}``: residual streams after layer 1 of the reference prefix model (real
    weights, layers 0-1, default chat prompt; bit-identical to the golden-stream run's ``after_layer_01`` of
    ``chat_default``). Cached in ``STREAMS_FILE``; ``create`` computes it if missing."""
    if STREAMS_FILE.is_file():
        return torch.load(STREAMS_FILE, weights_only=True)
    if not create:
        return None
    from models.demos.motif3.reference.golden import DEFAULT_PROMPT_MESSAGES
    from models.demos.motif3.reference.tokenizer import encode_chat, load_tokenizer
    from models.demos.motif3.reference.weights import load_reference_model

    ids = torch.tensor([encode_chat(DEFAULT_PROMPT_MESSAGES, load_tokenizer())])
    model = load_reference_model(layer_ids=(0, 1), dtype=torch.bfloat16, lazy_experts=True)
    with torch.no_grad():
        _, x = model.model(ids, return_streams=True)  # x [1, S, 4, 4096] after layer 1
    out = {"ids": ids[0].to(torch.int32).clone(), "x": x[0].contiguous().clone()}
    TEST_CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(out, STREAMS_FILE)
    return out


def _require_real_streams():
    """The layer-1 real-stream golden. Built on the fly (reference layers 0-1 on CPU, ~2 s) when the file is missing;
    skips only when the checkpoint layers are not local. Never falls back to random streams (review P3-4)."""
    g = _real_streams()
    if g is not None:
        return g
    from models.demos.motif3.tt.weights import HFWeightLoader

    try:
        src = HFWeightLoader()
    except FileNotFoundError as e:
        pytest.skip(f"no local checkpoint for the real-stream golden: {e}")
    if not (src.layer_available(0) and src.layer_available(1)):
        pytest.skip("checkpoint layers 0-1 are not local: no real-stream golden")
    t0 = time.time()
    g = _real_streams(create=True)
    print(f"[embed_head] the real-stream golden was missing: built {STREAMS_FILE} ({time.time() - t0:.1f} s)")
    return g


def _golden_stream_or_skip(layers):
    """``({layer: {prompt: [S, 4, 4096] bf16}}, {prompt: StreamPrompt})`` from the golden-stream run in
    ``GOLDEN_STREAM_DIR``. Skips, naming the missing files, when the run is absent (another workflow's artifact); fails
    when its files disagree with its prompt set or numerics."""
    from models.demos.motif3.reference import golden_stream as gs

    pfile = GOLDEN_STREAM_DIR / "prompts.json"
    missing = [str(p) for p in [pfile] + [gs.state_path(GOLDEN_STREAM_DIR, l) for l in layers] if not p.is_file()]
    if missing:
        pytest.skip(f"golden-stream run incomplete, missing {missing} (reference/golden_stream.py produces it)")
    prompts = {p.name: p for p in gs.load_prompt_set(pfile)}
    sha = gs.prompt_set_sha256(list(prompts.values()))
    out = {}
    for l in layers:
        states, meta = gs.load_tensors(gs.state_path(GOLDEN_STREAM_DIR, l))
        assert meta.get("prompt_sha256") == sha and meta.get("mode", {}).get("dtype") == "bf16", (l, meta)
        assert set(states) == set(prompts), (l, sorted(states))
        out[l] = {k: v[0] for k, v in states.items()}  # [S, 4, D]
    return out, prompts


def _head_golden_or_skip(layer):
    """The golden-stream run's final head after ``layer``: (logit tensors, hidden tensors) keyed ``<prompt>.<name>``."""
    from models.demos.motif3.reference import golden_stream as gs

    lp, hp = gs.head_paths(GOLDEN_STREAM_DIR, layer)
    missing = [str(p) for p in (lp, hp) if not p.is_file()]
    if missing:
        pytest.skip(f"golden-stream final head after layer {layer} missing: {missing}")
    return gs.load_tensors(lp)[0], gs.load_tensors(hp)[0]


def _topk_compare(dev: torch.Tensor, ids: torch.Tensor, vals: torch.Tensor, argmax: torch.Tensor) -> dict:
    """Device logits ``[N, V]`` vs a golden's top-32 (``ids`` / ``vals`` ``[N, 32]``, descending) and argmax ``[N]``.

    ``err``: max |device - golden| on the golden's top-32 entries. ``exact``: fraction of rows whose device argmax is the
    golden argmax. ``consistent``: fraction of rows whose device argmax is one of the golden's top-32 with a golden logit
    >= golden max - 2 err - 1e-6 (if the two argmaxes differ, the device logits rank the two tokens the other way round,
    which an error of at most ``err`` on each can only do within 2 err; so anything else is a real ranking error. bf16
    logits tie often: the golden run counts several exact ties)."""
    g = dev.gather(1, ids)
    err = float((g - vals).abs().max())
    am = dev.argmax(-1)
    in_top = ids == am[:, None]
    at_dev = torch.where(in_top, vals, torch.full_like(vals, float("-inf"))).max(-1).values
    return dict(
        err=err,
        pcc=_pcc(g, vals),
        exact=float((am == argmax).float().mean()),
        consistent=float((at_dev >= vals[:, 0] - 2 * err - 1e-6).float().mean()),
    )


# ============================================================================================================
# CPU tests
# ============================================================================================================
def test_cpu_lm_head_layouts():
    """The mesh / tp host layouts give each mesh coordinate exactly its vocab block (emulated mapper), and the tp
    layout equals the shared ``weights.lm_head_for_chip``."""
    from models.demos.motif3.tt import lm_head as LH
    from models.demos.motif3.tt import weights as W

    for mesh_shape in ((4, 8), (8, 4)):
        cfg = _cpu_cfg(vocab_size=32 * 32 * 3, mesh_shape=mesh_shape)  # Vc = 96 ("mesh"), 384 ("tp")
        lm = torch.randn(cfg.vocab_size, cfg.hidden_size).to(torch.bfloat16)
        R, C = mesh_shape
        for split in LH.VOCAB_SPLITS:
            host = LH.lm_head_device_layout(lm, cfg, split)
            assert host.dtype == torch.bfloat16
            dp_dim, tp_dim = LH.lm_head_mesh_dims(cfg, split)
            vc = LH.vocab_per_shard(cfg, split)
            blocks = set()
            for r in range(R):
                for c in range(C):
                    chip = W.shard_for_device(host, cfg.axes, r, c, dp_dim=dp_dim, tp_dim=tp_dim)
                    b = LH.vocab_block_of_coord(cfg, r, c, split)
                    blocks.add(b)
                    assert tuple(chip.shape) == (1, 1, cfg.hidden_size, vc)
                    assert torch.equal(chip[0, 0], lm[b * vc : (b + 1) * vc].t()), (mesh_shape, split, r, c)
                    if split == "tp":
                        _, tp = cfg.axes.roles(r, c)
                        assert torch.equal(chip[0, 0].float(), W.lm_head_for_chip(lm, cfg, tp))
            assert blocks == set(range(cfg.vocab_size // vc))


def test_cpu_assemble_offsets_tokens():
    """Host assembly of per-chip logits, the argmax offsets table, and the decode / prefill token packing."""
    from models.demos.motif3.tt import embedding as E
    from models.demos.motif3.tt import lm_head as LH

    cfg = _cpu_cfg(vocab_size=32 * 32 * 3)
    full = torch.randn(32, cfg.vocab_size)
    R, C = cfg.axes.mesh_shape
    for split in LH.VOCAB_SPLITS:
        vc = LH.vocab_per_shard(cfg, split)
        shards = torch.zeros(R, C, 32, vc)
        for r in range(R):
            for c in range(C):
                b = LH.vocab_block_of_coord(cfg, r, c, split)
                if split == "mesh":
                    shards[r, c] = full[:, b * vc : (b + 1) * vc]
                else:
                    dp, _ = cfg.axes.roles(r, c)
                    shards[r, c, :8] = full[8 * dp : 8 * dp + 8, b * vc : (b + 1) * vc]
        assert torch.equal(LH.assemble_decode_logits(shards, cfg, split), full), split
        off = LH.argmax_offsets(cfg, split)
        n = cfg.num_chips if split == "mesh" else cfg.tp
        assert tuple(off.shape) == (1, 1, 32 * n, 32 if split == "mesh" else 8)
        real = off[0, 0, ::32, 0]
        assert sorted(real.tolist()) == [b * vc for b in range(n)]
        assert (off[0, 0].reshape(n, 32, -1)[:, 1:] == LH.TIEBREAK_SENTINEL).all()
    # decode tokens: row r gets lanes 8r..8r+7 once per stream
    tok = torch.arange(32, dtype=torch.int32) * 7
    rows = E.decode_token_rows(tok, cfg)
    assert tuple(rows.shape) == (16, 8) and rows.dtype == torch.int32
    for r in range(4):
        for s in range(4):
            assert torch.equal(rows[4 * r + s], tok[8 * r : 8 * r + 8])
    with pytest.raises(ValueError):
        E.decode_token_rows(torch.full((32,), cfg.vocab_size), cfg)
    neg = tok.clone()
    neg[5] = -1  # an "inactive" marker -> pad token
    assert int(E.decode_token_rows(neg, cfg)[0, 5]) == cfg.pad_token_id
    p = E.prefill_token_row(torch.tensor([5, 6, 7]), 128, cfg, n_copies=4)
    assert tuple(p.shape) == (4, 128) and p[:, :3].tolist() == [[5, 6, 7]] * 4 and int(p[:, 3:].abs().sum()) == 0
    with pytest.raises(ValueError):
        E.prefill_token_row(torch.arange(129), 128, cfg)


def test_cpu_t64_tokens_offsets_configs():
    """T64 host helpers (docs/p5_t64/P5_T64_DESIGN.md §4.1-§4.2): ``decode_token_rows(rows_per_dp=16)`` packs the 64
    row-ordered tokens (``16 r + j``) once per stream per DP row (the default 8 lanes unchanged); the 64-column argmax
    offsets table equals the 32-lane one on every column (row ``32 j`` = shard ``j``'s vocab offset, the rest the
    sentinel); the LM-head GEMM config of the 64 gathered rows (``m_tiles=2``) equals ``model_config.lm_head_pc`` and
    the G16-lite config (``per_core_M`` 2, subblock 1 x w), and ``m_tiles=1`` is the unchanged 32-lane config."""
    from models.demos.motif3.tt import embedding as E
    from models.demos.motif3.tt import lm_head as LH
    from models.demos.motif3.tt.model_config import MotifTTConfig

    cfg = _cpu_cfg(vocab_size=32 * 32 * 3)
    g = torch.Generator().manual_seed(4)
    ta = torch.randint(0, cfg.vocab_size, (32,), generator=g, dtype=torch.int32)  # anchors, lane order
    td = torch.randint(0, cfg.vocab_size, (32,), generator=g, dtype=torch.int32)  # drafts, lane order
    t64 = torch.cat([ta.reshape(4, 8), td.reshape(4, 8)], dim=1).reshape(64)  # row order 16 r + j
    rows = E.decode_token_rows(t64, cfg, rows_per_dp=16)
    assert tuple(rows.shape) == (16, 16) and rows.dtype == torch.int32
    for r in range(4):
        for s in range(4):
            assert torch.equal(rows[4 * r + s], torch.cat([ta[8 * r : 8 * r + 8], td[8 * r : 8 * r + 8]]))
    assert torch.equal(E.decode_token_rows(ta, cfg), E.decode_token_rows(ta, cfg, rows_per_dp=8))
    assert E.decode_rows_per_dp(cfg) == 8 and E.decode_rows_per_dp(cfg, 16) == 16
    for bad in (dict(tokens=t64), dict(tokens=ta, rows_per_dp=16), dict(tokens=t64, rows_per_dp=33)):
        with pytest.raises(ValueError):
            E.decode_token_rows(bad.pop("tokens"), cfg, **bad)
    neg = t64.clone()
    neg[9] = -1  # an idle draft row -> pad token
    assert int(E.decode_token_rows(neg, cfg, rows_per_dp=16)[0, 9]) == cfg.pad_token_id
    # argmax offsets for the 64 split-order rows
    off32, off64 = LH.argmax_offsets(cfg, "mesh"), LH.argmax_offsets(cfg, "mesh", 64)
    assert tuple(off64.shape) == (1, 1, 32 * cfg.num_chips, 64)
    for c in range(64):
        assert torch.equal(off64[..., c], off32[..., c % 32]), c
    # the 64-row GEMM config
    real = MotifTTConfig.from_hf_config(mesh_shape=(4, 8))
    for split in LH.VOCAB_SPLITS:
        for m in (1, 2):
            pc = LH.lm_head_program_config(real, split, m_tiles=m)
            assert repr(pc) == repr(real.lm_head_pc(split, m_tiles=m)), (split, m)
            assert (pc.per_core_M, pc.out_block_h, pc.out_subblock_h) == (m, m, 1), (split, m)
        assert repr(LH.lm_head_program_config(real, split)) == repr(LH.lm_head_program_config(real, split, m_tiles=1))
        assert LH.lm_head_program_config(real, split, "auto", m_tiles=2) is None
    pc64 = LH.lm_head_program_config(real, "mesh", m_tiles=2)  # G16-lite: mm1d((12, 9), 16, 2, 2, 1, 2, fuse_batch)
    assert (pc64.compute_with_storage_grid_size.x, pc64.compute_with_storage_grid_size.y) == (12, 9)
    assert (pc64.in0_block_w, pc64.per_core_N, pc64.out_subblock_w, pc64.fuse_batch) == (16, 2, 2, True)
    with pytest.raises(ValueError, match="per_core_m"):
        LH.mcast1d_pc((12, 9), 215, 2, 16, per_core_m=0)


def test_cpu_concat_views():
    """The host logits assembly from per-chip views (numpy copies, inline and on a thread pool) equals the torch
    reference assembly for both vocab splits and both mesh orientations."""
    from concurrent.futures import ThreadPoolExecutor

    from models.demos.motif3.tt import lm_head as LH

    pool = ThreadPoolExecutor(4)
    try:
        for mesh_shape in ((4, 8), (8, 4)):
            cfg = _cpu_cfg(vocab_size=32 * 32 * 3, mesh_shape=mesh_shape)
            R, C = mesh_shape
            for split in LH.VOCAB_SPLITS:
                vc = LH.vocab_per_shard(cfg, split)
                lanes = 32 if split == "mesh" else 8
                shards = torch.randn(R, C, lanes, vc).to(torch.bfloat16)
                views = [shards[r, c].reshape(1, 1, lanes, vc) for r in range(R) for c in range(C)]
                want = LH.assemble_decode_logits(shards, cfg, split)
                for pl in (None, pool):
                    got = LH._concat_views(views, lanes, vc, split, cfg.axes, C, pool=pl)
                    assert got.dtype == torch.bfloat16 and torch.equal(got, want), (mesh_shape, split, pl)
    finally:
        pool.shutdown()


def test_cpu_prefill_row_pick():
    """The host half of the prefill head (``_prefill_row_from_views``): per-chip tile-row views (the device layout of
    either split) -> row ``last_index % 32`` of the full vocab, in vocab order, as a fresh tensor (the staging views are
    overwritten by the next read), for both mesh orientations; bf16 and fp32."""
    from models.demos.motif3.tt import lm_head as LH

    for mesh_shape in ((4, 8), (8, 4)):
        cfg = _cpu_cfg(vocab_size=32 * 32 * 3, mesh_shape=mesh_shape)
        R, C = mesh_shape
        for dt in (torch.bfloat16, torch.float32):
            tile = torch.randn(32, cfg.vocab_size).to(dt)  # the last token's tile row, full vocab
            for split in LH.VOCAB_SPLITS:
                vc = LH.vocab_per_shard(cfg, split)
                views = []
                for r in range(R):
                    for c in range(C):
                        b = LH.vocab_block_of_coord(cfg, r, c, split)
                        views.append(tile[:, b * vc : (b + 1) * vc].clone().reshape(1, 1, 32, vc))
                for li in (0, 5, 31, 32, 37, 4095):
                    got = LH._prefill_row_from_views(views, li % 32, vc, split, cfg.axes)
                    assert got.dtype == dt and torch.equal(got, tile[li % 32]), (mesh_shape, split, li)
                got = LH._prefill_row_from_views(views, 7, vc, split, cfg.axes)
                for v in views:
                    v.zero_()
                assert torch.equal(got, tile[7]), "the prefill row must not alias the staging buffers"


def test_cpu_logits_dtype():
    """Logits dtype = the activation dtype (bf16) by default, independent of the LM-head weight dtype (review P3-1: a
    bfp8 head weight must not give block-float logits); fp32 on request; block-float / integer dtypes rejected."""
    import dataclasses

    import ttnn

    from models.demos.motif3.tt import lm_head as LH

    cfg = _cpu_cfg()
    assert LH.resolve_logits_dtype(cfg) == cfg.dtypes.activations == ttnn.bfloat16
    cfg8 = _cpu_cfg(dtypes=dataclasses.replace(cfg.dtypes, lm_head=ttnn.bfloat8_b))
    assert cfg8.dtypes.lm_head == ttnn.bfloat8_b and LH.resolve_logits_dtype(cfg8) == ttnn.bfloat16
    assert LH.resolve_logits_dtype(cfg8, ttnn.float32) == ttnn.float32
    for bad in (ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.uint32):
        with pytest.raises(ValueError):
            LH.resolve_logits_dtype(cfg, bad)


def test_cpu_import_rule():
    """tt/embedding.py and tt/lm_head.py import no other models/demos package, no other models.* package and none of
    vllm / transformers / huggingface_hub / safetensors at import time, and open no device (fresh interpreter; same
    probe as tests/unit/test_infra_import.py)."""
    import json
    import subprocess
    import sys

    tt_metal = Path(__file__).resolve().parents[5]
    probe = (
        "import importlib, json, sys\n"
        "for m in ('models.demos.motif3.tt.embedding', 'models.demos.motif3.tt.lm_head'):\n"
        "    importlib.import_module(m)\n"
        "demos = sorted(m for m in sys.modules if m.startswith('models.demos.') and not m.startswith('models.demos.motif3'))\n"
        "other = sorted(m for m in sys.modules if m.startswith('models.') and not m.startswith('models.demos'))\n"
        "heavy = sorted(m for m in ('vllm', 'transformers', 'huggingface_hub', 'safetensors') if m in sys.modules)\n"
        "print(json.dumps({'demos': demos, 'other': other, 'heavy': heavy}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tt_metal) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    res = subprocess.run([sys.executable, "-c", probe], cwd=str(tt_metal), env=env, capture_output=True, text=True,
                         timeout=300)
    assert res.returncode == 0, res.stderr[-4000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out == {"demos": [], "other": [], "heavy": []}, out
    for marker in ("Opening user mode device driver", "Starting devices in cluster"):
        assert marker not in res.stderr and marker not in res.stdout


def test_cpu_prepare_real_streams():
    """Compute (once) the real-weight stream golden for the device tests: reference prefix model (layers 0-1, bf16,
    real weights) on the default chat prompt; streams after layer 1."""
    from models.demos.motif3.tt.weights import HFWeightLoader

    try:
        src = HFWeightLoader()
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if not (src.layer_available(0) and src.layer_available(1) and src.available("lm_head.weight")):
        pytest.skip("layers 0-1 / head not local")
    t0 = time.time()
    g = _real_streams(create=True)
    S = int(g["ids"].numel())
    x = g["x"]
    assert tuple(x.shape) == (S, 4, 4096) and x.dtype == torch.bfloat16
    rms = x.float().pow(2).mean(-1).sqrt()
    print(f"\n[embed_head] real streams after layer 1: S={S}, rms per stream min/median/max "
          f"{rms.min():.3f}/{rms.median():.3f}/{rms.max():.3f}, absmax {x.float().abs().max():.1f} ({time.time()-t0:.1f} s)")


def test_cpu_stream_sum_eps_identity():
    """``rms_norm(sum_4 x; 16 eps) == rms_norm(mean_4 x; eps)`` -- the identity the head uses to fold the 1/4 of the
    stream mean into the epsilon -- holds **bit-exactly** in the reference's bf16 numerics (the reference RMSNorm module
    in bf16, on the torch bf16 stream mean vs on bf16(fp32 sum)), and to fp32 rounding in fp32 (1/4 is a power of two,
    so it commutes with every rounding). Rows span 1e-3 .. 1e3 in scale, one row is all zero (epsilon only)."""
    from models.demos.motif3.reference.modules import RMSNorm

    eps = float(_cpu_cfg().rms_norm_eps)
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(64, 4, 4096, generator=g) * torch.logspace(-3, 3, 64)[:, None, None]).to(torch.bfloat16)
    x[0] = 0
    gamma = (1 + 0.1 * torch.randn(4096, generator=g)).to(torch.bfloat16)
    m = x.mean(1)  # the reference's reduce_streams (torch bf16 mean)
    s = x.float().sum(1).to(torch.bfloat16)  # the stream sum rounded once (what the device sum approximates)
    assert torch.equal(s.float() * 0.25, m.float())
    for dt in (torch.bfloat16, torch.float32):
        n_m, n_s = RMSNorm(4096, eps).to(dt), RMSNorm(4096, eps * 4**2).to(dt)
        with torch.no_grad():
            n_m.weight.copy_(gamma.to(dt))
            n_s.weight.copy_(gamma.to(dt))
            a, b = n_m(m.to(dt)), n_s(s.to(dt))
        if dt == torch.bfloat16:
            assert torch.equal(a, b), f"bf16 identity broken on {(a != b).float().mean().item():.2e} of the elements"
        else:
            assert (a - b).abs().max() <= 2e-7 * a.abs().max(), (a - b).abs().max()


# ============================================================================================================
# device helpers
# ============================================================================================================
def _free(o):
    import ttnn

    if isinstance(o, (list, tuple)):
        for x in o:
            _free(x)
    elif isinstance(o, ttnn.Tensor):
        try:
            ttnn.deallocate(o)
        except Exception:
            pass


class _Capture:
    """Exception-safe trace capture (a raise inside still ends and releases the capture; GATES_RESULTS §11.6)."""

    def __init__(self, mesh_device):
        self.mesh, self.tid = mesh_device, None

    def __enter__(self):
        import ttnn

        self.tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        return self

    def __exit__(self, exc_type, exc, tb):
        import ttnn

        try:
            ttnn.end_trace_capture(self.mesh, self.tid, cq_id=0)
        except Exception:
            if exc_type is None:
                raise
        if exc_type is not None:
            try:
                ttnn.release_trace(self.mesh, self.tid)
            except Exception:
                pass
        return False


def _replay_times(mesh_device, tid, reps):
    import ttnn

    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ts.append((time.perf_counter() - t0) * 1e6)
    return sorted(ts)


def _trace_n(mesh_device, fn, n, reps):
    import ttnn

    with _Capture(mesh_device) as cap:
        for _ in range(n):
            _free(fn())
    try:
        return _replay_times(mesh_device, cap.tid, reps)
    finally:
        ttnn.release_trace(mesh_device, cap.tid)


def _traced_us(mesh_device, fn, n=32, reps=11, n_lo=None):
    """Traced per-call us by the gates' slope method: (min replay of a trace of ``n`` calls - min replay of ``n_lo``
    calls) / (n - n_lo); ``n_lo`` defaults to n / 2. Returns (slope, t(n_lo), t(n)). The replay jitter is a few us, so
    a few-us op needs a few hundred calls between the two traces (review P3-3: 16 vs 32 calls read as 0). Host load
    makes replays bimodal (GATES_RESULTS §2); mins are used. The raw slope is returned (it can be slightly negative
    when the op is below the resolution)."""
    import ttnn

    n_lo = n // 2 if n_lo is None else int(n_lo)
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    t1 = _trace_n(mesh_device, fn, n_lo, reps)[0]
    t2 = _trace_n(mesh_device, fn, n, reps)[0]
    return (t2 - t1) / (n - n_lo), t1, t2


def _eager_us(mesh_device, fn, iters=10, warmup=2):
    import ttnn

    for _ in range(warmup):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        _free(fn())
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) / iters * 1e6


def _wall_ms(fn, reps=9):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return out, ts[0], ts[len(ts) // 2]


def _load():
    try:
        return os.getloadavg()[0]
    except OSError:
        return float("nan")


def _row_sharded(mesh_device, cfg, rows: torch.Tensor, dtype, layout):
    """Host ``[dp, ...]`` -> device tensor whose DP row ``r`` holds ``rows[r]`` (shape ``[1, ...]``), replicated over TP."""
    import ttnn

    dims = cfg.axes.mesh_dims(dp_dim=0, tp_dim=None)
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate() if d is None else ttnn.PlacementShard(d) for d in dims],
            ttnn.MeshShape(*cfg.axes.mesh_shape),
        ),
    )
    return ttnn.from_torch(
        rows, dtype=dtype, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
    )


def _decode_X(mesh_device, cfg, lanes_x: torch.Tensor, device=True):
    """Lane-ordered streams ``[32, 4, 4096]`` -> decode input ``X``: DP row r gets ``[1, 4, 8, 4096]`` of lanes
    8r..8r+7 (stream-major)."""
    import ttnn

    rows = lanes_x.reshape(cfg.dp, cfg.lanes_per_row, 4, -1).permute(0, 2, 1, 3).contiguous()  # [dp, 4, 8, D]
    if device:
        return _row_sharded(mesh_device, cfg, rows, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    dims = cfg.axes.mesh_dims(dp_dim=0, tp_dim=None)
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate() if d is None else ttnn.PlacementShard(d) for d in dims],
            ttnn.MeshShape(*cfg.axes.mesh_shape),
        ),
    )
    return ttnn.from_torch(rows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)


def _prefill_X(mesh_device, x_tokens: torch.Tensor, bucket: int):
    """Real streams ``[S, 4, 4096]`` -> prefill input ``X [1, 4, bucket, 4096]`` (zero-padded), replicated."""
    import ttnn

    S = int(x_tokens.shape[0])
    xp = torch.zeros(1, 4, bucket, x_tokens.shape[-1], dtype=torch.bfloat16)
    xp[0, :, :S] = x_tokens.permute(1, 0, 2)
    return ttnn.from_torch(xp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))


def _bucket(S: int) -> int:
    return max(128, 1 << (int(S) - 1).bit_length())


def _per_chip(t, mesh_device):
    from models.demos.motif3.tt.ccl import device_tensors_to_torch

    return device_tensors_to_torch(t, mesh_device)


def _lane_rows(t, mesh_device, cfg) -> torch.Tensor:
    """Row-replicated decode tensor ``[1, 1, 8, D]`` per chip -> host ``[32, D]`` in lane order (the tp = 0 chip of each
    DP row; replica identity is checked separately)."""
    per = _per_chip(t, mesh_device)  # [R, C, 1, 1, 8, D]
    L = cfg.lanes_per_row
    out = []
    for dp in range(cfg.dp):
        r, c = cfg.axes.coord(dp, 0)
        out.append(per[r, c].reshape(-1, per.shape[-1])[:L])
    return torch.cat(out)


def _mesh_params(shape=(4, 8)):
    from models.demos.motif3.tt.model_config import device_params

    return [pytest.param(tuple(shape), device_params("FABRIC_2D_TORUS_XY", TRACE), id=f"{shape[0]}x{shape[1]}-torus2d")]


def _setup(mesh_device, tag):
    from models.demos.motif3.tt.ccl import MotifCCL, log_fabric
    from models.demos.motif3.tt.model_config import MotifTTConfig

    cfg = MotifTTConfig.from_hf_config(mesh_device=mesh_device)
    rep = log_fabric(mesh_device, tag)
    print(f"[embed_head] {tag}: host load average {_load():.1f}; {cfg.describe()}")
    return cfg, MotifCCL(mesh_device, cfg), rep


def _tracker_on() -> bool:
    try:
        from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING
    except Exception:
        return False
    return bool(TRACE_ALLOC_TRACKING)


RESULTS = []


def _report(line):
    RESULTS.append(line)
    print(f"[embed_head] {line}", flush=True)


def _check_norm(head, X, lanes, ref: _RefHead, mesh_device, cfg, tag, golden_hidden=None):
    """The head's stream sum and final norm on decode streams vs the reference: stream sum == bf16(fp32 sum) fraction
    and max ulp distance; ``stream_mean_norm`` vs the reference RMSNorm in bf16 numerics (and the golden-stream run's
    ``final_hidden`` when given) and the fp32 ideal: PCC (README §12 RMSNorm >= 0.9999), max-abs, bit-exact fraction;
    TP replicas identical."""
    import ttnn
    from models.demos.motif3.tt.ccl import replicas_identical

    s = ttnn.sum(X, dim=1, keepdim=True, compute_kernel_config=head.ckc_norm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    s_dev = _lane_rows(s, mesh_device, cfg)
    _free(s)
    s_ref = lanes.float().sum(1).to(torch.bfloat16)
    same = s_dev.float() == s_ref.float()
    ulps = (s_dev.view(torch.int16).int() - s_ref.view(torch.int16).int()).abs()
    s_ulp = int(torch.where(same, torch.zeros_like(ulps), ulps).max())
    hn = head.stream_mean_norm(X)
    assert replicas_identical(hn, mesh_device, "tp", cfg.axes), tag
    got = _lane_rows(hn, mesh_device, cfg).float()
    _free(hn)
    n16, n32 = ref.norm(lanes, False).float(), ref.norm(lanes, True).float()
    p16, p32 = _pcc(got, n16), _pcc(got, n32)
    e16, e32 = float((got - n16).abs().max()), float((got - n32).abs().max())
    exact = float((got == n16).float().mean())
    line = (f"{tag} final norm [32, 4096]: PCC vs bf16 ref {p16:.7f} (max-abs {e16:.4f}, bit-exact {exact * 100:.2f}%), "
            f"vs fp32 ref {p32:.7f} (max-abs {e32:.4f}; |ref| max {n32.abs().max():.2f}); stream sum == bf16(fp32 sum) "
            f"on {float(same.float().mean()) * 100:.3f}% (max {s_ulp} ulp; |sum| max {s_ref.float().abs().max():.0f})")
    if golden_hidden is not None:
        assert torch.equal(n16, golden_hidden.float()), "the CPU reference no longer reproduces the golden-stream run"
        line += f"; CPU ref == golden-stream final_hidden bit for bit (PCC vs golden {_pcc(got, golden_hidden):.7f})"
    _report(line)
    assert p16 >= 0.9999 and p32 >= 0.9999, (tag, p16, p32)
    return dict(p16=p16, p32=p32, e16=e16, exact=exact)


# ============================================================================================================
# device tests
# ============================================================================================================
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_embedding(mesh_device, device_params):
    """EMB-1: decode ([4, 8] tokens per row -> X [1, 4, 8, 4096]) and prefill (S = 128 / 256 / 1024) with the real
    table: bit-exact rows, 4 identical streams, replicas identical over TP; replicated and hidden-sharded tables; device
    greedy feedback (argmax ids -> next decode input); latency eager / traced."""
    import ttnn
    from models.demos.motif3.tt.ccl import replicas_identical
    from models.demos.motif3.tt.embedding import MotifEmbedding

    cfg, ccl, _ = _setup(mesh_device, "embedding")
    src = _real_source_or_skip()
    streams = _require_real_streams()
    t0 = time.perf_counter()
    table = src.get("model.embed_tokens.weight")
    t1 = time.perf_counter()
    emb = MotifEmbedding(mesh_device, cfg, source=src, ccl=ccl, cache=False)
    ttnn.synchronize_device(mesh_device)
    t2 = time.perf_counter()
    _report(f"embedding: read table {t1-t0:.1f} s, upload replicated [220160, 4096] bf16 RM {t2-t1:.1f} s "
            f"(1.80 GB/chip)")

    g = torch.Generator().manual_seed(1)
    toks = torch.randint(0, cfg.vocab_size, (32,), generator=g, dtype=torch.int32)
    toks[0], toks[31] = 0, cfg.vocab_size - 1  # table edges
    expect = table[toks.long()].reshape(4, 8, 4096)  # [dp, lane, D]

    def check_decode(x, name):
        assert list(x.shape) == [1, 4, 8, 4096] and x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16, name
        per = _per_chip(x, mesh_device)  # [R, C, 1, 4, 8, D]
        R, C = cfg.axes.mesh_shape
        for r in range(R):
            for c in range(C):
                dp, _ = cfg.axes.roles(r, c)
                for s in range(4):
                    assert torch.equal(per[r, c, 0, s], expect[dp]), (name, r, c, s)
        assert replicas_identical(x, mesh_device, "tp", cfg.axes)

    tok_dev = emb.decode_tokens_device(toks)
    x = emb.forward_decode(tok_dev)
    check_decode(x, "decode replicated")
    _free(x)
    eager = _eager_us(mesh_device, lambda: emb.forward_decode(tok_dev))
    tr, a, b = _traced_us(mesh_device, lambda: emb.forward_decode(tok_dev), n=64)
    _report(f"embedding decode [4,8] -> X [1,4,8,4096]: bit-exact on 32 chips, replicas identical; eager {eager:.0f} us, "
            f"traced {tr:.1f} us/call (t32 {a:.0f} t64 {b:.0f} us)")

    # trace replay with a new persistent input (copy_host_to_device_tensor)
    with _Capture(mesh_device) as cap:
        xt = emb.forward_decode(tok_dev)
    try:
        toks2 = torch.randint(0, cfg.vocab_size, (32,), generator=g, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(emb.decode_tokens_host(toks2), tok_dev)
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
        expect = table[toks2.long()].reshape(4, 8, 4096)
        check_decode(xt, "decode trace replay")
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    _report("embedding decode trace replay with new tokens: bit-exact")

    # device greedy feedback: lane-ordered ids [1,1,1,32] uint32 RM on every chip -> per-row [4, 8] input
    ids = ttnn.from_torch(toks.reshape(1, 1, 1, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    fb = emb.decode_tokens_from_device(ids, output_tensor=tok_dev)
    per = _per_chip(fb, mesh_device)
    want = toks.reshape(4, 1, 8).expand(4, 4, 8)
    R, C = cfg.axes.mesh_shape
    for r in range(R):
        for c in range(C):
            dp, _ = cfg.axes.roles(r, c)
            assert torch.equal(per[r, c].reshape(4, 8).to(torch.int32), want[dp]), (r, c)
    tr_fb, a_fb, b_fb = _traced_us(mesh_device, lambda: emb.decode_tokens_from_device(ids), n=256, n_lo=64, reps=9)
    _report(f"embedding greedy feedback ids[32] -> [4,8] decode input: exact; traced {tr_fb:.2f} us/call "
            f"(slope of t64 {a_fb:.0f} / t256 {b_fb:.0f} us)")
    _free(ids)

    # prefill: the real prompt (S = 145 -> bucket 256) and a full 128 bucket in both explicit modes, plus "auto" on a
    # 1024 bucket (-> gather4) and a 256 bucket (-> repeat)
    prompt = streams["ids"]
    for mode in ("repeat", "gather4", "auto"):
        emb.prefill_mode = mode
        cases = ((int(prompt.numel()), 256), (128, 128)) if mode != "auto" else ((int(prompt.numel()), 1024), (int(prompt.numel()), 256))
        for S_real, bucket in cases:
            p = prompt[:S_real]
            tp_dev = emb.prefill_tokens_device(p, bucket)
            xp = emb.forward_prefill(tp_dev)
            assert list(xp.shape) == [1, 4, bucket, 4096], (mode, list(xp.shape))
            got = ttnn.to_torch(ttnn.get_device_tensors(xp)[17])  # any chip
            want = table[torch.cat([p.long(), torch.zeros(bucket - S_real, dtype=torch.long)])]
            for s in range(4):
                assert torch.equal(got[0, s], want), (mode, bucket, s)
            assert replicas_identical(xp, mesh_device, "tp", cfg.axes) and replicas_identical(xp, mesh_device, "dp", cfg.axes)
            _free(xp)
            e_us = _eager_us(mesh_device, lambda: emb.forward_prefill(tp_dev), iters=5)
            t_us, _, _ = _traced_us(mesh_device, lambda: emb.forward_prefill(tp_dev), n=16, reps=7)
            _report(f"embedding prefill {mode} S={bucket} (prompt {S_real}; token rows {int(tp_dev.shape[0])}): bit-exact, "
                    f"identical on 32 chips; eager {e_us:.0f} us, traced {t_us:.1f} us")
            if mode == "auto":
                assert int(tp_dev.shape[0]) == (4 if bucket >= 1024 else 1)
            _free(tp_dev)
    emb.prefill_mode = "auto"
    _free([tok_dev, emb.weight])

    # hidden-sharded table (memory lever): [220160, 512] per TP chip + TP all-gather
    t1 = time.perf_counter()
    emb_h = MotifEmbedding(mesh_device, cfg, source=src, ccl=ccl, cache=False, shard_hidden=True)
    ttnn.synchronize_device(mesh_device)
    up = time.perf_counter() - t1
    tok_dev = emb_h.decode_tokens_device(toks)
    expect = table[toks.long()].reshape(4, 8, 4096)
    x = emb_h.forward_decode(tok_dev)
    check_decode(x, "decode hidden-sharded")
    _free(x)
    tr_h, _, _ = _traced_us(mesh_device, lambda: emb_h.forward_decode(tok_dev), n=32)
    tp_dev = emb_h.prefill_tokens_device(prompt, 256)
    xp = emb_h.forward_prefill(tp_dev)
    got = ttnn.to_torch(ttnn.get_device_tensors(xp)[5])
    want = table[torch.cat([prompt.long(), torch.zeros(256 - prompt.numel(), dtype=torch.long)])]
    assert all(torch.equal(got[0, s], want) for s in range(4))
    _free([xp, tp_dev, tok_dev])
    _report(f"embedding hidden-sharded ([220160,512]/chip, upload {up:.1f} s): decode + prefill bit-exact; decode "
            f"traced {tr_h:.1f} us (incl. TP all-gather)")
    _free(emb_h.weight)


def _head_inputs(streams):
    """32 decode lanes (rows differ per DP row) and the prefill inputs from the real layer-1 streams."""
    x = streams["x"]  # [S, 4, D]
    S = x.shape[0]
    return x, x[S - 32 :].clone()  # 32 consecutive real tokens -> lanes 0..31


def _prefill_positions_check(mesh_device, head, ref: _RefHead, xs, cases, tag):
    """Prefill head at several last-token positions per bucket on the real streams ``xs [S, 4, D]``: every position vs
    the reference (fp32 and bf16 numerics; PCC >= 0.999), output layout, and **no new program** for any position after a
    bucket's first call (review P1). ``cases``: ``(S_real, bucket, positions)``. Returns per-case report lines."""
    import ttnn

    expected = None  # program-cache entries after the first call of the latest new bucket
    seen = set()
    lines = []
    for S_real, bucket, positions in cases:
        Xp = _prefill_X(mesh_device, xs[:S_real], bucket)
        r32 = ref.logits(xs[list(positions)], True)
        r16 = ref.logits(xs[list(positions)], False)
        worst, worst16, err, am_ok, grew = 1.0, 1.0, 0.0, 0, []
        for k, li in enumerate(positions):
            n_before = mesh_device.num_program_cache_entries()
            tile = head.forward_prefill(Xp, li)
            assert list(tile.shape) == [1, 1, 32, head.vc] and tile.layout == ttnn.ROW_MAJOR_LAYOUT, list(tile.shape)
            got = head.prefill_logits_to_host(tile, li)
            assert got.shape == (head.cfg.vocab_size,) and got.dtype == torch.bfloat16
            got = got.float()
            n_after = mesh_device.num_program_cache_entries()
            if bucket not in seen:
                seen.add(bucket)
                grew.append(n_after - n_before)
                expected = n_after
            else:
                assert n_after == expected, (f"{tag}: prefill at a new position (S={bucket}, last={li}) compiled "
                                             f"{n_after - expected} new program(s)")
            if k == 0:
                e_us = _eager_us(mesh_device, lambda: head.forward_prefill(Xp, li), iters=5)
                _, tmin, _ = _wall_ms(lambda: head.prefill_logits_to_host(tile, li), reps=9)
            _free(tile)
            p32, p16 = _pcc(got, r32[k]), _pcc(got, r16[k])
            worst, worst16 = min(worst, p32), min(worst16, p16)
            err = max(err, float((got - r32[k]).abs().max()))
            am_ok += int(int(got.argmax()) == int(r32[k].argmax()))
            assert p32 >= 0.999 and p16 >= 0.999, (tag, bucket, li, p32, p16)
        lines.append(f"{tag} prefill S={bucket} (prompt {S_real}) last-token positions {list(positions)}: min PCC vs fp32 "
                     f"ref {worst:.6f} / bf16 ref {worst16:.6f}, max-abs {err:.4f}, argmax == fp32 ref {am_ok}/"
                     f"{len(positions)}; first call +{grew[0] if grew else 0} program(s), later positions +0; eager head "
                     f"(incl. position upload) {e_us:.0f} us, host read {tmin:.2f} ms")
        _free(Xp)
    for line in lines:
        _report(line)
    return lines


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
@pytest.mark.parametrize("vocab_split", ["mesh", "tp"])
def test_lm_head(mesh_device, device_params, vocab_split):
    """EMB-2/3 with the real final norm + lm_head and the real layer-1 streams: final norm output vs the reference
    RMSNorm (README §12), decode logits [32, 220160] (lane order) and prefill last-token logits at several positions per
    bucket vs the reference (bf16 model numerics and the fp32 ideal), no new program per prefill position, the
    on-device greedy argmax (== torch.argmax of the device logits; ties -> lowest index), trace replay with new inputs,
    and the head + host-transfer latency."""
    import ttnn
    from models.demos.motif3.tt.ccl import replicas_identical
    from models.demos.motif3.tt.lm_head import MotifLMHead, assemble_decode_logits

    cfg, ccl, _ = _setup(mesh_device, f"lm_head[{vocab_split}]")
    src = _real_source_or_skip()
    streams = _require_real_streams()
    gamma = src.get("model.norm.weight")
    lm = src.get("lm_head.weight")
    ref = _RefHead(gamma, lm, cfg.rms_norm_eps)
    t0 = time.perf_counter()
    head = MotifLMHead(mesh_device, cfg, source=src, ccl=ccl, cache=False, vocab_split=vocab_split)
    ttnn.synchronize_device(mesh_device)
    _report(f"lm_head[{vocab_split}]: weights uploaded in {time.perf_counter()-t0:.1f} s; per-chip W {list(head.weight.shape)}; "
            f"logits dtype {head.logits_dtype}")

    xs, lanes = _head_inputs(streams)
    ref16, ref32 = ref.logits(lanes, False), ref.logits(lanes, True)

    # ---- decode ----
    X = _decode_X(mesh_device, cfg, lanes)
    _check_norm(head, X, lanes, ref, mesh_device, cfg, f"lm_head[{vocab_split}] L01")
    logits = head.forward_decode(X)
    want_shape = [1, 1, 32, 6880] if vocab_split == "mesh" else [1, 1, 8, 27520]
    assert list(logits.shape) == want_shape, list(logits.shape)
    host = head.logits_to_host(logits)
    assert host.shape == (32, cfg.vocab_size) and host.dtype == torch.bfloat16
    hf = host.float()
    p16, p32 = _pcc(hf, ref16), _pcc(hf, ref32)
    e16, e32 = (hf - ref16).abs().max().item(), (hf - ref32).abs().max().item()
    per_lane = min(_pcc(hf[i], ref32[i]) for i in range(32))
    top1_ref = (hf.argmax(-1) == ref32.argmax(-1)).float().mean().item()
    _report(f"lm_head[{vocab_split}] decode (real streams after layer 1): PCC vs bf16 ref {p16:.6f} (max-abs {e16:.4f}), "
            f"vs fp32 ref {p32:.6f} (max-abs {e32:.4f}, |ref| max {ref32.abs().max():.1f}); min per-lane PCC "
            f"{per_lane:.6f}; top-1 == fp32 ref on {top1_ref*100:.1f}% lanes")
    assert p32 >= 0.999 and p16 >= 0.999 and per_lane >= 0.999
    # host assembly equals the per-chip device tensors (independent composition)
    per = _per_chip(logits, mesh_device).reshape(*cfg.axes.mesh_shape, int(logits.shape[2]), -1)
    assert torch.equal(assemble_decode_logits(per, cfg, vocab_split), host)

    # ---- greedy argmax on device ----
    tok = head.argmax_decode(logits)
    ids = head.tokens_to_host(tok)
    want_ids = hf.argmax(-1)
    assert torch.equal(ids, want_ids), (ids[:8], want_ids[:8])
    if vocab_split == "mesh":
        assert list(tok.shape) == [1, 1, 1, 32] and replicas_identical(tok, mesh_device, "tp", cfg.axes)
        assert replicas_identical(tok, mesh_device, "dp", cfg.axes)
    agree = (ids == ref32.argmax(-1)).float().mean().item()
    _report(f"lm_head[{vocab_split}] device argmax == torch.argmax(device logits) on 32/32 lanes; == fp32 ref argmax on "
            f"{agree*100:.1f}%")
    # ties across shards and inside a shard: crafted logits through the same argmax (lowest index wins)
    _free(tok)
    crafted = hf.clone()
    vc = head.vc
    crafted[0, :] = -5.0
    crafted[0, [3 * vc + 7, 6 * vc + 1, 3 * vc + 100]] = 50.0  # tie across 2 shards + inside shard 3 -> 3 vc + 7
    crafted[1, :] = -5.0
    crafted[1, [cfg.vocab_size - 1, 5]] = 50.0  # first and last shards -> 5
    crafted[2, :] = 1.0  # all equal -> 0
    crafted[9, :] = -5.0
    crafted[9, cfg.vocab_size - 1] = 60.0  # last entry of the vocab
    rows_dev = _logits_like(mesh_device, cfg, head, crafted.to(torch.bfloat16))
    tok_c = head.argmax_decode(rows_dev)
    ids_c = head.tokens_to_host(tok_c)
    assert int(ids_c[0]) == 3 * vc + 7 and int(ids_c[1]) == 5 and int(ids_c[2]) == 0 and int(ids_c[9]) == cfg.vocab_size - 1, ids_c[:10]
    assert torch.equal(ids_c, crafted.to(torch.bfloat16).float().argmax(-1))
    _free([rows_dev, tok_c])
    _report(f"lm_head[{vocab_split}] argmax tie-breaking (cross-shard, in-shard, all-equal, vocab end): torch.argmax semantics")

    # ---- latency: head (decode), argmax, transfer ----
    eager_head = _eager_us(mesh_device, lambda: head.forward_decode(X))
    tr_head, a, b = _traced_us(mesh_device, lambda: head.forward_decode(X), n=32, reps=11)
    tr_rm, _, _ = _traced_us(mesh_device, lambda: head.forward_decode(X, row_major=True), n=32, reps=11)
    tr_am, _, _ = _traced_us(mesh_device, lambda: head.argmax_decode(logits), n=32, reps=11)
    _report(f"lm_head[{vocab_split}] decode head (sum+norm{'+ag_dp_rows' if vocab_split == 'mesh' else ''}+GEMM): eager "
            f"{eager_head:.0f} us, traced {tr_head:.1f} us (t16 {a:.0f} / t32 {b:.0f} us); + device untilize {tr_rm - tr_head:+.1f} us; "
            f"argmax traced {tr_am:.1f} us")
    rm = head.forward_decode(X, row_major=True)
    out_rm, tmin, tmed = _wall_ms(lambda: head.logits_to_host(rm), reps=21)
    assert torch.equal(out_rm, host)
    _, tmin_t, tmed_t = _wall_ms(lambda: head.logits_to_host(logits), reps=11)
    _, tmin_f, tmed_f = _wall_ms(lambda: head.logits_to_host(rm, dtype=torch.float32), reps=11)
    tok = head.argmax_decode(logits)
    _, amin, amed = _wall_ms(lambda: head.tokens_to_host(tok), reps=21)
    _report(f"lm_head[{vocab_split}] host logits [32, 220160] bf16 (14.1 MB) from the traced RM logits: min {tmin:.2f} / "
            f"median {tmed:.2f} ms; from TILE logits (+ eager device untilize) min {tmin_t:.2f} / median {tmed_t:.2f} ms; "
            f"as fp32 min {tmin_f:.2f} / median {tmed_f:.2f} ms; device-argmax ids read min {amin:.2f} / median "
            f"{amed:.2f} ms (load {_load():.0f})")
    _free([rm, tok])

    # ---- traced decode step: head + untilize + argmax, replayed with new streams ----
    with _Capture(mesh_device) as cap:
        lg = head.forward_decode(X)
        lrm = head.logits_rm(lg)
        tk = head.argmax_decode(lg)
    try:
        lanes2 = xs[: 32].clone()  # 32 other real tokens
        ttnn.copy_host_to_device_tensor(_decode_X(mesh_device, cfg, lanes2, device=False), X)
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
        h2 = head.logits_to_host(lrm).float()
        r2 = ref.logits(lanes2, True)
        p2 = _pcc(h2, r2)
        ids2 = head.tokens_to_host(tk)
        assert p2 >= 0.999 and torch.equal(ids2, h2.argmax(-1))
        times = _replay_times(mesh_device, cap.tid, 15)
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    _report(f"lm_head[{vocab_split}] traced decode tail (head+untilize+argmax) replay with new streams: PCC {p2:.6f} vs fp32 "
            f"ref, argmax exact; replay+sync min {times[0]:.0f} / median {times[len(times)//2]:.0f} us")
    _free([X, logits, lg, lrm, tk])

    # ---- prefill: last-token positions in two buckets (tile edges, new tile rows, new in-tile rows) ----
    S = int(xs.shape[0])  # 145 -> bucket 256
    _prefill_positions_check(mesh_device, head, ref, xs, (
        (S, 256, (S - 1, 0, 31, 32, 63, 100, 128, S - 2)),
        (128, 128, (127, 0, 64, 96, 33)),
        (31, 128, (30,)),
    ), f"lm_head[{vocab_split}]")
    _free([head.weight])


def _logits_like(mesh_device, cfg, head, full: torch.Tensor):
    """Host lane-ordered logits ``[32, V]`` -> device tensor in ``head``'s decode-logits layout (inverse assembly)."""
    import ttnn
    from models.demos.motif3.tt.lm_head import vocab_block_of_coord

    R, C = cfg.axes.mesh_shape
    vc = head.vc
    if head.vocab_split == "mesh":
        host = torch.empty(R, 1, 32, C * vc, dtype=full.dtype)
        for r in range(R):
            for c in range(C):
                b = vocab_block_of_coord(cfg, r, c, "mesh")
                host[r, 0, :, c * vc : (c + 1) * vc] = full[:, b * vc : (b + 1) * vc]
        dims = [ttnn.PlacementShard(0), ttnn.PlacementShard(3)]
    else:
        a = cfg.axes
        host = torch.empty(a.dp_size, 1, 8, cfg.vocab_size, dtype=full.dtype)
        for dp in range(a.dp_size):
            host[dp, 0] = full[8 * dp : 8 * dp + 8]
        d = [None, None]
        d[a.dp_axis], d[a.tp_axis] = 0, 3
        dims = [ttnn.PlacementShard(x) for x in d]
    mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(dims, ttnn.MeshShape(R, C)))
    return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_lm_head_late_layers(mesh_device, device_params):
    """The default ("mesh") head on REAL late-layer residual streams from the reference golden-stream run (review
    P3-4): after layers 8 / 16 / 35 the streams have RMS ~3 / 11 / 44 and |x| up to ~4200 (layer 1: 0.05 / 1.3), the
    realistic input of the final norm. Decode: 32 lanes from the 6 prompts (each prompt's last position + random
    positions; every DP row mixes prompts): stream-sum rounding, final norm vs the reference (and, after layer 35, vs
    the run's own ``final_hidden``), logits vs the reference (fp32 / bf16) and the run's top-32 logits / argmax
    (tie-aware), device argmax. Prefill (layer 35): all 6 prompts (buckets 256 / 1024) at their last position and 7
    random positions each vs the run's per-position golden, no new program for any position."""
    from models.demos.motif3.tt.lm_head import MotifLMHead

    cfg, ccl, _ = _setup(mesh_device, "lm_head_late")
    src = _real_source_or_skip()
    states, prompts = _golden_stream_or_skip(LATE_LAYERS)
    g_logits, g_hidden = _head_golden_or_skip(HEAD_GOLDEN_LAYER)
    ref = _RefHead(src.get("model.norm.weight"), src.get("lm_head.weight"), cfg.rms_norm_eps)
    head = MotifLMHead(mesh_device, cfg, source=src, ccl=ccl, cache=False)
    names = sorted(prompts)
    gen = torch.Generator().manual_seed(35)
    picks = []
    for l in range(cfg.max_batch):
        n = names[l % len(names)]
        S = len(prompts[n].ids)
        picks.append((n, S - 1 if l < len(names) else int(torch.randint(0, S, (1,), generator=gen))))
    _report(f"lm_head_late: golden-stream run {GOLDEN_STREAM_DIR} ({len(names)} prompts: "
            f"{', '.join(f'{n} {len(prompts[n].ids)}' for n in names)}); decode lanes (prompt, position): {picks}")

    for layer in LATE_LAYERS:
        st = states[layer]
        lanes = torch.stack([st[n][p] for n, p in picks])  # [32, 4, 4096] bf16
        X = _decode_X(mesh_device, cfg, lanes)
        gh = None
        if layer == HEAD_GOLDEN_LAYER:
            gh = torch.stack([g_hidden[f"{n}.final_hidden"][0, p] for n, p in picks])
        _check_norm(head, X, lanes, ref, mesh_device, cfg, f"late L{layer}", golden_hidden=gh)
        logits = head.forward_decode(X)
        host = head.logits_to_host(logits).float()
        r32, r16 = ref.logits(lanes, True), ref.logits(lanes, False)
        p32, p16 = _pcc(host, r32), _pcc(host, r16)
        e32 = float((host - r32).abs().max())
        per_lane = min(_pcc(host[i], r32[i]) for i in range(32))
        am = host.argmax(-1)
        top1 = float((am == r32.argmax(-1)).float().mean())
        top1_ref16 = float((r16.argmax(-1) == r32.argmax(-1)).float().mean())
        # a device top-1 that differs from the fp32 reference must be a near-tie there: within 2x the lane's logit error
        e_lane = (host - r32).abs().max(-1).values
        near = float((r32.gather(1, am[:, None])[:, 0] >= r32.max(-1).values - 2 * e_lane - 1e-6).float().mean())
        tok = head.argmax_decode(logits)
        ids = head.tokens_to_host(tok)
        _free([tok, logits, X])
        assert torch.equal(ids, am)
        line = (f"late L{layer} decode logits: PCC vs fp32 ref {p32:.6f} (max-abs {e32:.4f}, |ref| max "
                f"{r32.abs().max():.1f}), vs bf16 ref {p16:.6f}; min per-lane PCC {per_lane:.6f}; top-1 == fp32 ref on "
                f"{top1 * 100:.1f}% lanes (CPU bf16 ref: {top1_ref16 * 100:.1f}%; every disagreement a near-tie within 2x "
                f"the logit error: {near * 100:.0f}%); device argmax exact")
        assert near == 1.0, (layer, near)
        if layer == HEAD_GOLDEN_LAYER:
            gi = torch.stack([g_logits[f"{n}.topk_ids"][p] for n, p in picks])
            gv = torch.stack([g_logits[f"{n}.topk_logits"][p] for n, p in picks])
            ga = torch.stack([g_logits[f"{n}.argmax"][p] for n, p in picks])
            c = _topk_compare(host, gi, gv, ga)
            ref_vs_golden = float((r16.gather(1, gi) - gv).abs().max())
            line += (f"; vs the run's top-32: max-abs {c['err']:.4f}, PCC {c['pcc']:.6f}, argmax == run {c['exact'] * 100:.1f}% "
                     f"(consistent within the logit error {c['consistent'] * 100:.1f}%); CPU bf16 ref vs run top-32 "
                     f"max-abs {ref_vs_golden:.4f}")
            assert c["consistent"] == 1.0, c
        _report(line)
        assert p32 >= 0.999 and p16 >= 0.999 and per_lane >= 0.999, (layer, p32, p16, per_lane)

    # ---- prefill on the layer-35 streams: every prompt, its last position + 7 random ones, vs the run's golden ----
    st = states[HEAD_GOLDEN_LAYER]
    expected, seen, first_growth = None, set(), {}
    for n in names:
        x = st[n]
        S = int(x.shape[0])
        bucket = _bucket(S)
        Xp = _prefill_X(mesh_device, x, bucket)
        pos = [S - 1] + sorted(set(int(v) for v in torch.randint(0, S - 1, (7,), generator=gen)))
        got = []
        for li in pos:
            nb = mesh_device.num_program_cache_entries()
            tile = head.forward_prefill(Xp, li)
            got.append(head.prefill_logits_to_host(tile, li).float())
            _free(tile)
            na = mesh_device.num_program_cache_entries()
            if bucket not in seen:
                seen.add(bucket)
                first_growth[bucket] = na - nb
                expected = na
            else:
                assert na == expected, f"prefill at a new position (S={bucket}, last={li}) compiled {na - expected} program(s)"
        got = torch.stack(got)
        c = _topk_compare(got, g_logits[f"{n}.topk_ids"][pos], g_logits[f"{n}.topk_logits"][pos], g_logits[f"{n}.argmax"][pos])
        r_last = ref.logits(x[S - 1 : S], True)[0]
        p_last = _pcc(got[0], r_last)
        _report(f"late L35 prefill {n} (S={S}, bucket {bucket}) positions {pos}: last-position PCC vs fp32 ref "
                f"{p_last:.6f}; vs the run's top-32 max-abs {c['err']:.4f} (PCC {c['pcc']:.6f}); argmax == run "
                f"{c['exact'] * 100:.0f}% (consistent {c['consistent'] * 100:.0f}%)")
        assert p_last >= 0.999 and c["consistent"] == 1.0, (n, p_last, c)
        _free(Xp)
    _report(f"late L35 prefill: first call per bucket added {first_growth} program(s); every later position added none")
    _free(head.weight)


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_prefill_head_trace_safety(mesh_device, device_params):
    """Review P1 regression: the prefill head compiles nothing per position, so prefill after decode capture cannot
    corrupt the decode trace, and the prefill head itself is traceable.

    1. Warm the decode tail and every tested bucket once (one position each); record the program-cache size.
    2. Prefill-head trace: capture ``forward_prefill(X)`` at one position, replay it at 4 others after
       ``set_prefill_position`` (position on the device): bitwise equal to the eager results.
    3. Decode trace (head + untilize + argmax), replayed once; then prefill at ~30 unwarmed positions in 3 buckets (new
       tile rows and new in-tile rows; each vs the reference); then the decode trace replayed again: no new
       program-cache entry, bitwise-identical logits and tokens and -- with ``TT_METAL_TRACE_ALLOC_TRACKING=1`` -- no
       corruptible buffer alive at the replay (``execute_trace`` raises otherwise; also queried directly).
    4. Negative control (what the old head did): a literal-start ``ttnn.slice`` at a new tile row with the decode trace
       live adds a program-cache entry and, with the tracker on, a corruptible kernel-binary buffer (``SliceDeviceOperation``)
       that the tracker reports for the decode trace (the trace is released without a replay)."""
    import ttnn
    from models.demos.motif3.tt.lm_head import MotifLMHead

    cfg, ccl, _ = _setup(mesh_device, "prefill_trace_safety")
    tracker = _tracker_on()
    src = _real_source_or_skip()
    ref = _RefHead(src.get("model.norm.weight"), src.get("lm_head.weight"), cfg.rms_norm_eps)
    head = MotifLMHead(mesh_device, cfg, source=src, ccl=ccl, cache=False)
    g = torch.Generator().manual_seed(7)
    lanes = (torch.randn(32, 4, 4096, generator=g) * torch.logspace(-1, 1, 32)[:, None, None]).to(torch.bfloat16)
    X = _decode_X(mesh_device, cfg, lanes)
    buckets = (128, 256, 1024)
    xh, Xp = {}, {}
    for b in buckets:  # every row distinct (random, row-dependent scale): a wrong tile or row cannot pass
        xh[b] = (torch.randn(b, 4, 4096, generator=g) * torch.logspace(-0.5, 1, b)[:, None, None]).to(torch.bfloat16)
        Xp[b] = _prefill_X(mesh_device, xh[b], b)

    # 1. warmup: decode tail (eager) and one prefill position per bucket
    lg = head.forward_decode(X)
    _free([head.logits_rm(lg), head.argmax_decode(lg), lg, head.forward_decode(X, row_major=True)])
    for b in buckets:
        tile = head.forward_prefill(Xp[b], b - 1)
        head.prefill_logits_to_host(tile, b - 1)
        _free(tile)
    ttnn.synchronize_device(mesh_device)
    n0 = mesh_device.num_program_cache_entries()

    # 2. the prefill head inside a trace, replayed at other positions
    pos2 = (37, 200, 255, 5)
    eager = {}
    for li in pos2:
        tile = head.forward_prefill(Xp[256], li)
        eager[li] = head.prefill_logits_to_host(tile, li)
        _free(tile)
    head.set_prefill_position(5, 256)
    with _Capture(mesh_device) as cap:
        out = head.forward_prefill(Xp[256])
    try:
        for li in pos2:
            head.set_prefill_position(li, 256)
            ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
            assert torch.equal(head.prefill_logits_to_host(out, li), eager[li]), f"traced prefill head at {li}"
        times = _replay_times(mesh_device, cap.tid, 11)
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    _free(out)
    _report(f"prefill_trace_safety: prefill head captured at last=5 and replayed at {list(pos2)} (position written on the "
            f"device): bitwise == eager; traced replay+sync (S=256) min {times[0]:.0f} / median {times[len(times)//2]:.0f} us")
    assert mesh_device.num_program_cache_entries() == n0

    # 3. decode trace + prefill at unwarmed positions + replay
    with _Capture(mesh_device) as cap:
        lg = head.forward_decode(X)
        lrm = head.logits_rm(lg)
        tk = head.argmax_decode(lg)
    try:
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
        h1, t1 = head.logits_to_host(lrm), head.tokens_to_host(tk)
        new_pos = {128: (0, 1, 30, 32, 33, 64, 95, 96, 126), 256: (6, 31, 63, 64, 129, 160, 191, 192, 224, 254),
                   1024: (2, 100, 257, 511, 512, 700, 767, 900, 1000, 1022)}
        worst, n_calls = 1.0, 0
        for b, positions in new_pos.items():
            r32 = ref.logits(xh[b][list(positions)], True)
            for k, li in enumerate(positions):
                tile = head.forward_prefill(Xp[b], li)
                got = head.prefill_logits_to_host(tile, li).float()
                _free(tile)
                worst = min(worst, _pcc(got, r32[k]))
                n_calls += 1
        n1 = mesh_device.num_program_cache_entries()
        unsafe = None
        if tracker:
            from ttnn._ttnn.operations.trace import get_unsafe_tracked_ids

            ttnn.synchronize_device(mesh_device)
            unsafe = dict(get_unsafe_tracked_ids(mesh_device, cap.tid))
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)  # raises with the tracker on if unsafe
        h2, t2 = head.logits_to_host(lrm), head.tokens_to_host(tk)
        _report(f"prefill_trace_safety: decode trace captured, then {n_calls} prefill calls at unwarmed positions in "
                f"buckets {list(new_pos)} (min PCC vs fp32 ref {worst:.6f}); program cache {n0} -> {n1} entries; tracker "
                f"{'ON' if tracker else 'OFF (set TT_METAL_TRACE_ALLOC_TRACKING=1)'}: "
                f"{'no' if not unsafe else len(unsafe)} corruptible buffer(s) at the replay"
                f"{'' if tracker else ' (not checked)'}; decode replay logits / tokens bitwise identical: "
                f"{torch.equal(h1, h2) and torch.equal(t1, t2)}")
        assert n1 == n0, f"{n1 - n0} program(s) compiled by prefill after decode capture"
        assert not unsafe, unsafe
        assert torch.equal(h1, h2) and torch.equal(t1, t2)
        assert worst >= 0.999, worst

        # 4. negative control (what the old head did): a literal-start slice at a new tile row compiles a program while
        # the decode trace is live; the tracker reports its kernel-binary buffer as corruptible (no replay after it)
        xs = ttnn.slice(Xp[1024], (0, 0, 416, 0), (1, 4, 448, 4096), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _free(xs)
        n2 = mesh_device.num_program_cache_entries()
        flagged = None
        if tracker:
            flagged = dict(get_unsafe_tracked_ids(mesh_device, cap.tid))
        _report(f"prefill_trace_safety: negative control, literal-start slice at a new tile row with the decode trace "
                f"live: program cache {n1} -> {n2}; tracker flags "
                f"{'(off)' if flagged is None else [v[:90] for v in flagged.values()]}")
        assert n2 == n1 + 1
        if tracker:
            assert flagged and any("SliceDeviceOperation" in v for v in flagged.values()), flagged
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
    _free([lg, lrm, tk, X, head.weight] + list(Xp.values()))


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_lm_head_random_weights(mesh_device, device_params):
    """Random weights at real dims (lm_head [220160, 4096] N(0, 0.02), gamma 1 + N(0, 0.1)), random streams with a
    wide per-token scale; both stream reductions and fp32 logits; decode + device argmax + prefill vs the fp32 / bf16
    reference."""
    import ttnn
    from models.demos.motif3.tt.lm_head import MotifLMHead
    from models.demos.motif3.tt.weights import DictWeightSource

    cfg, ccl, _ = _setup(mesh_device, "lm_head_random")
    g = torch.Generator().manual_seed(11)
    lm = (torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) * 0.02).to(torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(cfg.hidden_size, generator=g)).to(torch.bfloat16)
    src = DictWeightSource({"lm_head.weight": lm, "model.norm.weight": gamma})
    ref = _RefHead(gamma, lm, cfg.rms_norm_eps)
    lanes = (torch.randn(32, 4, 4096, generator=g) * torch.logspace(-1, 1.5, 32)[:, None, None]).to(torch.bfloat16)
    ref32, ref16 = ref.logits(lanes, True), ref.logits(lanes, False)
    X = _decode_X(mesh_device, cfg, lanes)
    bucket = 256
    xp = torch.zeros(bucket, 4, 4096, dtype=torch.bfloat16)
    xp[199], xp[37] = lanes[5], lanes[9]
    Xp = _prefill_X(mesh_device, xp, bucket)
    for reduce, ldt in (("sum", None), ("wreduce", None), ("sum", ttnn.float32)):
        head = MotifLMHead(mesh_device, cfg, source=src, ccl=ccl, cache=False, stream_reduce=reduce, logits_dtype=ldt)
        tag = f"{reduce}, logits {'fp32' if ldt is not None else 'bf16'}"
        lg = head.forward_decode(X)
        out = head.logits_to_host(lg)
        want_dt = torch.float32 if ldt is not None else torch.bfloat16
        assert out.dtype == want_dt, out.dtype
        out = out.float()
        tok = head.argmax_decode(lg)
        ids = head.tokens_to_host(tok)
        assert torch.equal(ids, out.argmax(-1)), tag
        _free([lg, tok])
        p32, p16 = _pcc(out, ref32), _pcc(out, ref16)
        e32 = (out - ref32).abs().max().item()
        tr, _, _ = _traced_us(mesh_device, lambda: head.stream_mean_norm(X), n=64)
        pp = []
        for li, lane in ((199, 5), (37, 9)):
            tile = head.forward_prefill(Xp, li)
            got = head.prefill_logits_to_host(tile, li)
            _free(tile)
            assert got.dtype == want_dt, (tag, got.dtype)
            pp.append(_pcc(got.float(), ref32[lane]))
        _report(f"lm_head random weights ({tag}): decode PCC vs fp32 ref {p32:.6f} (max-abs {e32:.4f}, |ref| max "
                f"{ref32.abs().max():.2f}), vs bf16 ref {p16:.6f}; device argmax exact; stream mean+norm traced {tr:.1f} us; "
                f"prefill S={bucket} last=199 / 37 PCC {pp[0]:.6f} / {pp[1]:.6f}")
        assert p32 >= 0.999 and p16 >= 0.999 and min(pp) >= 0.999
        _free(head.weight)
    _free([X, Xp])


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params((8, 4)), indirect=True)
def test_embed_head_8x4(mesh_device, device_params):
    """The (8, 4) orientation (the plugin's BH-Galaxy preset; TP = mesh dim 0): decode embedding rows per DP group,
    head logits / host assembly / argmax / prefill row pick for both vocab splits (random head weights, real table)."""
    from models.demos.motif3.tt.ccl import replicas_identical
    from models.demos.motif3.tt.embedding import MotifEmbedding
    from models.demos.motif3.tt.lm_head import MotifLMHead
    from models.demos.motif3.tt.weights import DictWeightSource

    cfg, ccl, _ = _setup(mesh_device, "8x4")
    assert cfg.axes.tp_axis == 0 and cfg.axes.dp_axis == 1
    src = _real_source_or_skip()
    table = src.get("model.embed_tokens.weight")
    emb = MotifEmbedding(mesh_device, cfg, source=src, ccl=ccl, cache=False)
    toks = torch.randint(0, cfg.vocab_size, (32,), generator=torch.Generator().manual_seed(9), dtype=torch.int32)
    x = emb.forward_decode(emb.decode_tokens_device(toks))
    per = _per_chip(x, mesh_device)
    R, C = cfg.axes.mesh_shape
    expect = table[toks.long()].reshape(4, 8, 4096)
    for r in range(R):
        for c in range(C):
            dp, _ = cfg.axes.roles(r, c)
            assert all(torch.equal(per[r, c, 0, s], expect[dp]) for s in range(4)), (r, c)
    assert replicas_identical(x, mesh_device, "tp", cfg.axes)
    _free([x, emb.weight])
    g = torch.Generator().manual_seed(12)
    lm = (torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) * 0.02).to(torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(cfg.hidden_size, generator=g)).to(torch.bfloat16)
    hsrc = DictWeightSource({"lm_head.weight": lm, "model.norm.weight": gamma})
    lanes = (torch.randn(32, 4, 4096, generator=g) * 2).to(torch.bfloat16)
    ref = _RefHead(gamma, lm, cfg.rms_norm_eps).logits(lanes, True)
    X = _decode_X(mesh_device, cfg, lanes)
    bucket = 128
    xp = torch.zeros(bucket, 4, 4096, dtype=torch.bfloat16)
    xp[76], xp[33] = lanes[3], lanes[20]
    Xp = _prefill_X(mesh_device, xp, bucket)
    for split in ("mesh", "tp"):
        head = MotifLMHead(mesh_device, cfg, source=hsrc, ccl=ccl, cache=False, vocab_split=split)
        lg = head.forward_decode(X)
        host = head.logits_to_host(lg).float()
        p = _pcc(host, ref)
        tok = head.argmax_decode(lg)
        ids = head.tokens_to_host(tok)
        _free(tok)
        assert p >= 0.999 and torch.equal(ids, host.argmax(-1)), (split, p)
        pp = []
        for li, lane in ((76, 3), (33, 20)):
            tile = head.forward_prefill(Xp, li)
            pp.append(_pcc(head.prefill_logits_to_host(tile, li).float(), ref[lane]))
            _free(tile)
        assert min(pp) >= 0.999, (split, pp)
        _report(f"8x4 [{split}]: decode PCC vs fp32 ref {p:.6f}, argmax exact, prefill (last=76 / 33) PCC "
                f"{pp[0]:.6f} / {pp[1]:.6f}")
        _free([lg, head.weight])
    _free([X, Xp])


# ============================================================================================================
# T64: the 64-row verify step (docs/p5_t64/P5_T64_DESIGN.md §4.1-§4.4)
# ============================================================================================================
def _decode_X16(mesh_device, cfg, lanes_a: torch.Tensor, lanes_d: torch.Tensor, device=True):
    """Lane-ordered streams of the anchors / drafts ``[32, 4, 4096]`` -> the T64 decode input ``X [1, 4, 16, 4096]``
    per DP row r = ``[the 8 anchors of lanes 8r.. | their 8 drafts]`` (stream-major), replicated over TP."""
    import ttnn

    def rows(x):
        return x.reshape(cfg.dp, cfg.lanes_per_row, 4, -1).permute(0, 2, 1, 3)  # [dp, 4, 8, D]

    r16 = torch.cat([rows(lanes_a), rows(lanes_d)], dim=2).contiguous()  # [dp, 4, 16, D]
    if device:
        return _row_sharded(mesh_device, cfg, r16, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    dims = cfg.axes.mesh_dims(dp_dim=0, tp_dim=None)
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate() if d is None else ttnn.PlacementShard(d) for d in dims],
            ttnn.MeshShape(*cfg.axes.mesh_shape),
        ),
    )
    return ttnn.from_torch(r16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)


def _ids_tt(mesh_device, ids: torch.Tensor):
    """Ids ``[n]`` -> ``[1, 1, 1, n]`` uint32 ROW_MAJOR on every chip (``argmax_decode``'s output layout)."""
    import ttnn

    return ttnn.from_torch(ids.reshape(1, 1, 1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                           device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))  # fmt: skip


@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_embed_head_t64(mesh_device, device_params):
    """WP-D (D1) module test of the T64 verify step's gathers, embedding and LM head ("mesh"; P5_T64_DESIGN.md §4.1,
    §4.2, §4.4), real globals from the serving TT cache: 16 rows per DP row (``[8 anchors | 8 drafts]``) vs the same
    rows as two 32-lane calls (anchors, drafts), every comparison bitwise on all 32 chips:

    * ``ccl.ag_dp_rows(halves=2)`` at W = 4096 / 576: the split order (rows 0..31 every DP row's first 8 rows in lane
      order, 32..63 its last 8); ``halves=1`` at 16 rows: the natural order ``16 dp + j``;
    * embedding: ``decode_tokens_device(tokens [64], rows_per_dp=16)`` -> ``forward_decode`` ``X [1, 4, 16, 4096]`` ==
      the two 8-lane calls; ``embed_rows_from_device`` of the split-order ids ``[1, 1, 1, 64]`` -> ``[1, 1, 16, 4096]``
      == the two 32-id calls (the T64 MTP input);
    * head on real layer-1 streams: ``stream_mean_norm`` at 16 rows == the 8-row calls; ``decode_logits(hn, halves=2)``
      ``[1, 1, 64, 6880]``: rows 0..31 == the anchors' 32-lane logits, 32..63 == the drafts'; ``argmax_decode`` of
      them ``[1, 1, 1, 64]`` == the two 32-lane argmaxes == ``torch.argmax`` of the host logits, identical on every
      chip; ``logits_rm(lg, rows=32)`` == ``logits_rm`` of the anchors' logits (its host read == theirs);
      ``logits_to_host`` refuses the 64-row logits; the 32-lane head == B0's committed module (git ``HEAD``);
    * a traced 16-row head (norm, split-order logits, argmax) replayed with new streams == eager;
    * every new program's static CBs end below a one-page L1 pin; traced cost of the 64-row head pieces vs the 32-lane
      ones (informational; G16-lite: gather 42.6 vs 27.6 us, GEMM 149.2 vs 143.8 us, argmax 243.5 vs 216.5 us)."""
    import ttnn
    from models.demos.motif3.tests.unit.test_moe import _RaisingSource, b0_module, l1_pin, t64_cfg
    from models.demos.motif3.tt.ccl import MotifCCL, log_fabric
    from models.demos.motif3.tt.embedding import MotifEmbedding
    from models.demos.motif3.tt.lm_head import MotifLMHead
    from models.demos.motif3.tt.model import layer_cache_complete, read_only_cache

    cfg = t64_cfg(mesh_device)
    log_fabric(mesh_device, "embed_head_t64")
    print(f"[embed_head] t64: host load average {_load():.1f}; {cfg.describe()}")
    if not layer_cache_complete(cfg, None):
        pytest.skip(f"TT-cache part 'global' is not converted ({cfg.cache_dir})")
    ccl = MotifCCL(mesh_device, cfg)
    pin = l1_pin(mesh_device)
    L, D = cfg.lanes_per_row, cfg.hidden_size
    R, C = cfg.axes.mesh_shape
    failures = []

    def check(name, ok, detail=""):
        _report(f"t64 {name}: {'ok' if ok else 'FAIL'}{('; ' + detail) if detail else ''}")
        if not ok:
            failures.append(f"{name}: {detail}")

    # ---- the split-order gather -----------------------------------------------------------------------------
    g = torch.Generator().manual_seed(64)
    for W in (4096, 576):
        rows = torch.randn(cfg.dp, 1, 2 * L, W, generator=g).bfloat16()  # DP row r: [1, 1, 16, W]
        x = _row_sharded(mesh_device, cfg, rows, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        split = ccl.ag_dp_rows(x, halves=2)
        nat = ccl.ag_dp_rows(x)
        ps, pn = _per_chip(split, mesh_device), _per_chip(nat, mesh_device)  # [R, C, 1, 1, 64, W]
        want_s = torch.cat([rows[:, 0, :L].reshape(-1, W), rows[:, 0, L:].reshape(-1, W)])
        want_n = rows[:, 0].reshape(-1, W)
        ok = all(torch.equal(ps[r, c, 0, 0], want_s) and torch.equal(pn[r, c, 0, 0], want_n)
                 for r in range(R) for c in range(C))  # fmt: skip
        check(f"ag_dp_rows W={W}: halves=2 split order / halves=1 natural order, all 32 chips", ok)
        _free([x, split, nat])

    # ---- modules from the TT cache ----------------------------------------------------------------------------------
    t0 = time.time()
    with read_only_cache() as misses:
        emb = MotifEmbedding(mesh_device, cfg, source=_RaisingSource(), ccl=ccl, cache=True)
        head = MotifLMHead(mesh_device, cfg, source=_RaisingSource(), ccl=ccl, cache=True)
    assert misses == [], f"tensors missing from the TT cache: {misses}"
    assert sorted(head._am_wide) == [64] and sorted(head.pc_wide) == [64], (head._am_wide, head.pc_wide)
    _report(f"t64: embedding + LM head ('mesh', 64-row argmax constants) loaded from the TT cache in "
            f"{time.time() - t0:.1f} s")

    # ---- embedding ---------------------------------------------------------------------------------------------------
    ta = torch.randint(0, cfg.vocab_size, (32,), generator=g, dtype=torch.int32)
    td = torch.randint(0, cfg.vocab_size, (32,), generator=g, dtype=torch.int32)
    ta[0], td[31] = 0, cfg.vocab_size - 1  # table edges
    t64 = torch.cat([ta.reshape(cfg.dp, L), td.reshape(cfg.dp, L)], dim=1).reshape(-1)  # row order 16 r + j
    tok_a, tok_d = emb.decode_tokens_device(ta), emb.decode_tokens_device(td)
    tok16 = emb.decode_tokens_device(t64, rows_per_dp=2 * L)
    xa, xd, x16 = emb.forward_decode(tok_a), emb.forward_decode(tok_d), emb.forward_decode(tok16)
    pa, pd, p16 = _per_chip(xa, mesh_device), _per_chip(xd, mesh_device), _per_chip(x16, mesh_device)
    check("embedding forward_decode [4, 16] tokens -> X [1, 4, 16, 4096] == two 8-lane calls",
          list(x16.shape) == [1, 4, 2 * L, D] and torch.equal(p16[..., :L, :], pa) and torch.equal(p16[..., L:, :], pd))
    _free([xa, xd, x16, tok_a, tok_d, tok16])
    ids_a, ids_d, ids64 = _ids_tt(mesh_device, ta), _ids_tt(mesh_device, td), _ids_tt(mesh_device, torch.cat([ta, td]))
    ea, ed, e16 = (emb.embed_rows_from_device(i) for i in (ids_a, ids_d, ids64))
    pa, pd, p16 = _per_chip(ea, mesh_device), _per_chip(ed, mesh_device), _per_chip(e16, mesh_device)
    check("embed_rows_from_device split-order ids [1, 1, 1, 64] -> [1, 1, 16, 4096] == two 32-id calls",
          list(e16.shape) == [1, 1, 2 * L, D] and torch.equal(p16[..., :L, :], pa) and torch.equal(p16[..., L:, :], pd)
          and ids64.is_allocated())  # fmt: skip
    _free([ea, ed, e16, ids_a, ids_d, ids64])

    # ---- head on real layer-1 streams --------------------------------------------------------------------------------
    streams = _require_real_streams()["x"]  # [S, 4, 4096] bf16
    S = streams.shape[0]
    sel = torch.linspace(0, S - 1, 64).round().long()
    la, ld = streams[sel[:32]], streams[sel[32:]]
    Xa, Xd = _decode_X(mesh_device, cfg, la), _decode_X(mesh_device, cfg, ld)
    X16 = _decode_X16(mesh_device, cfg, la, ld)
    hna, hnd, hn16 = head.stream_mean_norm(Xa), head.stream_mean_norm(Xd), head.stream_mean_norm(X16)
    pa, pd, p16 = _per_chip(hna, mesh_device), _per_chip(hnd, mesh_device), _per_chip(hn16, mesh_device)
    check("stream_mean_norm at 16 rows == two 8-row calls",
          torch.equal(p16[..., :L, :], pa) and torch.equal(p16[..., L:, :], pd))
    lga, lgd = head.decode_logits(hna), head.decode_logits(hnd)
    lg64 = head.decode_logits(hn16, halves=2)
    qa, qd, q64 = _per_chip(lga, mesh_device), _per_chip(lgd, mesh_device), _per_chip(lg64, mesh_device)
    da = float((q64[..., :32, :].float() - qa.float()).abs().max())
    dd = float((q64[..., 32:, :].float() - qd.float()).abs().max())
    check("decode_logits(halves=2) [1, 1, 64, 6880]: rows 0..31 == anchors' 32-lane logits, 32..63 == drafts'",
          list(lg64.shape) == [1, 1, 64, head.vc] and torch.equal(q64[..., :32, :], qa)
          and torch.equal(q64[..., 32:, :], qd), f"max |diff| {da:.3e} / {dd:.3e}")  # fmt: skip
    host_a, host_d = head.logits_to_host(lga), head.logits_to_host(lgd)  # [32, V] each
    am_a, am_d = head.argmax_decode(lga), head.argmax_decode(lgd)
    am64 = head.argmax_decode(lg64)
    ta_, td_, t64_ = head.tokens_to_host(am_a), head.tokens_to_host(am_d), head.tokens_to_host(am64)
    same_chips = all(torch.equal(ttnn.to_torch(t), ttnn.to_torch(ttnn.get_device_tensors(am64)[0]))
                     for t in ttnn.get_device_tensors(am64))  # fmt: skip
    ref_am = torch.cat([host_a.float().argmax(-1), host_d.float().argmax(-1)])
    check("argmax_decode of the 64 rows [1, 1, 1, 64] == the two 32-lane argmaxes == torch.argmax of the host logits",
          list(am64.shape) == [1, 1, 1, 64] and torch.equal(t64_, torch.cat([ta_, td_])) and torch.equal(t64_, ref_am)
          and same_chips, f"identical on 32 chips {same_chips}")  # fmt: skip
    rm32, rma = head.logits_rm(lg64, rows=32), head.logits_rm(lga)
    check("logits_rm(lg, rows=32) == logits_rm of the anchors' logits; its host read == theirs",
          torch.equal(_per_chip(rm32, mesh_device), _per_chip(rma, mesh_device))
          and torch.equal(head.logits_to_host(rm32), host_a))  # fmt: skip
    # the 32-lane path is bitwise B0's: the committed head (git HEAD) on the anchors' hidden states
    with read_only_cache() as misses:
        head_b0 = b0_module("lm_head").MotifLMHead(mesh_device, cfg, source=_RaisingSource(), ccl=ccl, cache=True)
    assert misses == [], misses
    lg_b0 = head_b0.decode_logits(hna)
    am_b0 = head_b0.argmax_decode(lg_b0)
    check("the 32-lane head (decode_logits, argmax_decode) == B0's committed module",
          torch.equal(_per_chip(lg_b0, mesh_device), qa) and torch.equal(head_b0.tokens_to_host(am_b0), ta_))
    _free([lg_b0, am_b0, head_b0.weight, head_b0.gamma])
    head_b0.close()
    try:
        head.logits_to_host(lg64)
        check("logits_to_host refuses the 64-row logits", False, "no error raised")
    except ValueError:
        check("logits_to_host refuses the 64-row logits", True)
    _free([rm32, rma, am_a, am_d, am64, lga, lgd])

    # ---- traced 16-row head, replayed with new streams ---------------------------------------------------------------
    with _Capture(mesh_device) as cap:
        hn_t = head.stream_mean_norm(X16)
        lg_t = head.decode_logits(hn_t, halves=2, consume=True)
        am_t = head.argmax_decode(lg_t)
    try:
        sel2 = torch.linspace(1, S - 2, 64).round().long()
        ttnn.copy_host_to_device_tensor(_decode_X16(mesh_device, cfg, streams[sel2[:32]], streams[sel2[32:]],
                                                    device=False), X16)  # fmt: skip
        ttnn.execute_trace(mesh_device, cap.tid, cq_id=0, blocking=True)
        got_lg, got_am = _per_chip(lg_t, mesh_device), head.tokens_to_host(am_t)
        hn_e = head.stream_mean_norm(X16)
        lg_e = head.decode_logits(hn_e, halves=2, consume=True)
        am_e = head.argmax_decode(lg_e)
        check("traced 16-row head (norm, split-order logits, argmax) replayed with new streams == eager",
              torch.equal(got_lg, _per_chip(lg_e, mesh_device)) and torch.equal(got_am, head.tokens_to_host(am_e)))
        _free([lg_e, am_e])
    finally:
        ttnn.release_trace(mesh_device, cap.tid)
        _free([lg_t, am_t])

    # ---- traced cost of the 64-row pieces (informational) -----------------------------------------------------------
    lg64 = head.decode_logits(hn16, halves=2)
    lga = head.decode_logits(hna)
    for name, f64, f32 in (
        ("ag_dp_rows [16 | 8 rows, 4096]", lambda: ccl.ag_dp_rows(hn16, halves=2), lambda: ccl.ag_dp_rows(hna)),
        ("decode_logits (gather + GEMM)", lambda: head.decode_logits(hn16, halves=2), lambda: head.decode_logits(hna)),
        ("argmax_decode", lambda: head.argmax_decode(lg64), lambda: head.argmax_decode(lga)),
    ):
        s64, _, _ = _traced_us(mesh_device, f64, n=64)
        s32, _, _ = _traced_us(mesh_device, f32, n=64)
        _report(f"t64 traced {name}: 64 rows {s64:.1f} us vs 32 lanes {s32:.1f} us per call")
    _free([lg64, lga, hna, hnd, hn16, Xa, Xd, X16, pin, emb.weight, head.weight, head.gamma])
    head.close()
    assert not failures, "\n".join(failures)


# ============================================================================================================
# opt-in probes
# ============================================================================================================
@pytest.mark.skipif(os.environ.get("MOTIF3_EH_PROBE") != "1", reason="opt-in probe (MOTIF3_EH_PROBE=1)")
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_probe_transfer(mesh_device, device_params):
    """Host-transfer probe: per-chip read bandwidth (PCIe x1 vs the 4 x8 chips), concurrency, and the decode-logits
    readback variants (14.1 MB bf16); the prefill head's cost at new last-token positions."""
    import ttnn

    cfg, ccl, _ = _setup(mesh_device, "probe_transfer")
    R, C = cfg.axes.mesh_shape
    ids = list(mesh_device.get_device_ids())
    _report(f"probe: device ids (row-major mesh) {ids}; load {_load():.0f}")
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    for mb in (16, 0.44):
        rows = max(1, int(mb * 1024 * 1024 / 8192))
        big = ttnn.from_torch(torch.zeros(1, 1, rows, 4096, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                              mesh_mapper=rep)
        nbytes = rows * 8192
        shards = ttnn.get_device_tensors(big)
        ttnn.from_device(shards[0])
        bw = {}
        for i, s in enumerate(shards):
            _, tmin, _ = _wall_ms(lambda: ttnn.from_device(s), reps=5)
            bw[divmod(i, C)] = nbytes / (tmin * 1e-3) / 1e9
        _report(f"probe: single-chip read of {nbytes/1e6:.2f} MB, GB/s by mesh coord (rows):")
        for r in range(R):
            _report("    " + " ".join(f"{bw[(r, c)]:5.2f}{'*' if ids[r * C + c] in (5, 13, 21, 29) else ' '}" for c in range(C)))
        _, tmin, tmed = _wall_ms(lambda: ttnn.from_device(big), reps=7)
        _report(f"probe: all 32 chips at once: {32*nbytes/1e6:.1f} MB in min {tmin:.2f} / median {tmed:.2f} ms "
                f"({32*nbytes/(tmin*1e-3)/1e9:.1f} GB/s aggregate)  (* = PCIe x8 chip ids 5/13/21/29)")
        _free(big)
    # decode logits layouts
    vc = 6880
    full = torch.randn(R, 32, C, vc).to(torch.bfloat16)
    host = full.reshape(R, 1, 32, C * vc)
    mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(3)], ttnn.MeshShape(R, C)))
    lt = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper)
    lr = ttnn.untilize(lt, memory_config=ttnn.DRAM_MEMORY_CONFIG, use_multicore=True)
    expect = full.permute(1, 0, 2, 3).reshape(32, R * C * vc)
    c13 = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([1, 3]))
    c23 = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([2, 3]))

    def rm_view():
        return ttnn.to_torch(ttnn.reshape(lr, (32, 1, 1, vc)), mesh_composer=c13).reshape(32, -1)

    def tile_permute():
        x = ttnn.to_torch(lt, mesh_composer=c23)
        return x.reshape(R, 32, C * vc).permute(1, 0, 2).reshape(32, -1)

    def manual_rm():
        h = ttnn.from_device(lr)
        parts = [ttnn.to_torch(s).reshape(32, vc) for s in ttnn.get_device_tensors(h)]
        return torch.cat(parts, dim=1)

    from models.demos.motif3.tt.lm_head import HostShardReader

    reader = HostShardReader(mesh_device, lr)

    def staged():
        return torch.cat([v.reshape(32, vc) for v in reader.read(lr)], dim=1)

    def fresh_views():
        h = ttnn.from_device(lr)
        return torch.cat([s.to_torch_with_padded_shape().reshape(32, vc) for s in ttnn.get_device_tensors(h)], dim=1)

    for name, fn in (("RM view + composer[1,3]", rm_view), ("TILE composer[2,3] + permute", tile_permute),
                     ("RM from_device + per-shard to_torch + cat", manual_rm),
                     ("RM from_device + zero-copy views + cat", fresh_views),
                     ("RM persistent staging (HostShardReader) + cat", staged)):
        out, tmin, tmed = _wall_ms(fn, reps=11)
        _report(f"probe: decode logits readback {name}: correct={torch.equal(out, expect)} min {tmin:.2f} / median {tmed:.2f} ms")
    _, tmin, tmed = _wall_ms(lambda: ttnn.from_device(lr), reps=11)
    _report(f"probe: from_device only (RM, 32 x 440 KB): min {tmin:.2f} / median {tmed:.2f} ms")
    _, tmin, tmed = _wall_ms(lambda: reader.read(lr), reps=21)
    _report(f"probe: copy_device_to_host_tensor into persistent staging only: min {tmin:.2f} / median {tmed:.2f} ms")
    views = reader.read(lr)
    _, tmin, tmed = _wall_ms(lambda: torch.cat([v.reshape(32, vc) for v in views], dim=1), reps=21)
    _report(f"probe: torch.cat of the 32 zero-copy views -> fresh [32, 220160]: min {tmin:.2f} / median {tmed:.2f} ms")
    # read + copy back to back: split the time, vary torch's intra-op threads (contention with tt-metal's threads)
    import numpy as np

    nt0 = torch.get_num_threads()
    for nt in (nt0, 8, 4, 1):
        torch.set_num_threads(nt)
        ra, ca = [], []
        for _ in range(15):
            t0 = time.perf_counter()
            vv = reader.read(lr)
            t1 = time.perf_counter()
            out = torch.cat([v.reshape(32, vc) for v in vv], dim=1)
            t2 = time.perf_counter()
            ra.append((t1 - t0) * 1e3)
            ca.append((t2 - t1) * 1e3)
        ra.sort()
        ca.sort()
        _report(f"probe: staged read + cat, torch threads {nt}: read min {ra[0]:.2f} / med {ra[7]:.2f} ms, cat min "
                f"{ca[0]:.2f} / med {ca[7]:.2f} ms")
    torch.set_num_threads(nt0)
    ra, ca = [], []
    for _ in range(15):
        t0 = time.perf_counter()
        vv = reader.read(lr)
        t1 = time.perf_counter()
        out = torch.from_numpy(np.concatenate([v.view(torch.int16).numpy().reshape(32, vc) for v in vv], axis=1)).view(torch.bfloat16)
        t2 = time.perf_counter()
        ra.append((t1 - t0) * 1e3)
        ca.append((t2 - t1) * 1e3)
    ra.sort()
    ca.sort()
    assert torch.equal(out, expect)
    _report(f"probe: staged read + numpy.concatenate: read min {ra[0]:.2f} / med {ra[7]:.2f} ms, concat min {ca[0]:.2f} / "
            f"med {ca[7]:.2f} ms")
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(4)

    def threaded_copy(vv):
        out = np.empty((32, 32 * vc), dtype=np.int16)

        def part(k):
            for i in range(8 * k, 8 * k + 8):
                out[:, i * vc : (i + 1) * vc] = vv[i].view(torch.int16).numpy().reshape(32, vc)

        list(pool.map(part, range(4)))
        return torch.from_numpy(out).view(torch.bfloat16)

    ra, ca = [], []
    for _ in range(15):
        t0 = time.perf_counter()
        vv = reader.read(lr)
        t1 = time.perf_counter()
        out = threaded_copy(vv)
        t2 = time.perf_counter()
        ra.append((t1 - t0) * 1e3)
        ca.append((t2 - t1) * 1e3)
    ra.sort()
    ca.sort()
    assert torch.equal(out, expect)
    pool.shutdown()
    _report(f"probe: staged read + 4-thread numpy copy: read min {ra[0]:.2f} / med {ra[7]:.2f} ms, copy min {ca[0]:.2f} / "
            f"med {ca[7]:.2f} ms")
    h = ttnn.from_device(lr)
    _, tmin, _ = _wall_ms(lambda: ttnn.to_torch(ttnn.reshape(h, (32, 1, 1, vc)), mesh_composer=c13), reps=11)
    _report(f"probe: host-side compose only (RM host tensor -> torch [32, 220160]): min {tmin:.2f} ms")
    # prefill head at new last-token positions: no new program, so no first-call compile cost (review P1)
    from models.demos.motif3.tt.lm_head import MotifLMHead
    from models.demos.motif3.tt.weights import DictWeightSource

    g = torch.Generator().manual_seed(21)
    hsrc = DictWeightSource({"lm_head.weight": (torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) * 0.02).to(torch.bfloat16),
                             "model.norm.weight": torch.ones(cfg.hidden_size, dtype=torch.bfloat16)})
    for split in ("mesh", "tp"):
        head = MotifLMHead(mesh_device, cfg, source=hsrc, ccl=ccl, cache=False, vocab_split=split)
        S = 2048
        Xp = ttnn.from_torch(torch.randn(1, 4, S, 4096, generator=g).to(torch.bfloat16), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
        tile = head.forward_prefill(Xp, 5)
        head.prefill_logits_to_host(tile, 5)
        _free(tile)
        ttnn.synchronize_device(mesh_device)
        n0 = mesh_device.num_program_cache_entries()
        rows = []
        for li in (100, 300, 701, 1500, 1999, 2047):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                r = head.forward_prefill(Xp, li)
                ttnn.synchronize_device(mesh_device)
                ts.append((time.perf_counter() - t0) * 1e3)
                _free(r)
            rows.append(f"{li}: {ts[0]:.1f}/{ts[1]:.1f}/{ts[2]:.1f}")
        tile = head.forward_prefill(Xp, 777)
        _, rmin, rmed = _wall_ms(lambda: head.prefill_logits_to_host(tile, 777), reps=11)
        _free(tile)
        _report(f"probe: [{split}] prefill head (S=2048) eager ms at new last-token positions, 1st/2nd/3rd call: "
                f"{'; '.join(rows)}; program cache {n0} -> {mesh_device.num_program_cache_entries()}; host read of the "
                f"tile row min {rmin:.2f} / median {rmed:.2f} ms")
        _free([Xp, head.weight])
    one = ttnn.get_device_tensors(lr)
    _, tmin, _ = _wall_ms(lambda: ttnn.from_device(one[0]), reps=11)
    _report(f"probe: one chip's 440 KB alone: min {tmin:.2f} ms")
    _free([lt, lr])


@pytest.mark.skipif(os.environ.get("MOTIF3_EH_PROFILE") != "1", reason="opt-in device-profiler run (MOTIF3_EH_PROFILE=1)")
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_profile_head(mesh_device, device_params):
    """Device kernel durations (device profiler, max over chips; immune to host load) of the decode embedding, the
    head ops of both vocab splits, the LM-head GEMM program-config sweep, the argmax and the prefill head."""
    import ttnn
    from models.demos.motif3.tt.embedding import MotifEmbedding
    from models.demos.motif3.tt.lm_head import MotifLMHead, lm_head_program_config, sharded_norm_configs
    from models.demos.motif3.tt.weights import DictWeightSource

    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        pytest.skip("needs TT_METAL_DEVICE_PROFILER=1 + MID_RUN_DUMP + CPP_POST_PROCESS")
    cfg, ccl, _ = _setup(mesh_device, "profile_head")

    def device_us(fn, reps=3):
        """Device kernel time per call: per chip, the sum of its programs' DEVICE KERNEL DURATION (ops in execution
        order), averaged over ``reps`` calls; returns (max over chips, (programs per call, per-op us of the slowest
        chip's last call, min over chips)). Excludes op-to-op gaps."""
        _free(fn())
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        for _ in range(reps):
            _free(fn())
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        data = ttnn.get_latest_programs_perf_data()
        per_chip = {}
        for chip, plist in data.items():
            durs = []
            for p in sorted(plist, key=lambda q: (q.program_execution_uid.runtime_id, q.program_execution_uid.trace_id,
                                                  q.program_execution_uid.trace_id_counter)):
                d = [int(r.duration) for nm, r in p.program_analyses_results.items()
                     if nm.startswith("DEVICE KERNEL DURATION [ns]")]
                durs.append(max(d) if d else 0)
            per_chip[chip] = durs
        if not per_chip:
            return float("nan"), (0, [], float("nan"))
        totals = {c: sum(v) / reps for c, v in per_chip.items()}
        worst = max(totals, key=totals.get)
        n = len(per_chip[worst]) // reps if reps else 0
        ops = [round(x / 1000.0, 1) for x in per_chip[worst][-n:]] if n else []
        return totals[worst] / 1000.0, (n, ops, round(min(totals.values()) / 1000.0, 1))

    g = torch.Generator().manual_seed(5)
    lm = (torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g) * 0.02).to(torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(cfg.hidden_size, generator=g)).to(torch.bfloat16)
    table = torch.randn(cfg.vocab_size, cfg.hidden_size, generator=g).to(torch.bfloat16)
    src = DictWeightSource({"lm_head.weight": lm, "model.norm.weight": gamma, "model.embed_tokens.weight": table})
    emb = MotifEmbedding(mesh_device, cfg, source=src, ccl=ccl, cache=False)
    tok = emb.decode_tokens_device(torch.randint(0, cfg.vocab_size, (32,), generator=g))
    us, n = device_us(lambda: emb.forward_decode(tok))
    _report(f"profile: embedding decode: {us:.1f} us device (programs, per-op us: {n})")
    for S in (128, 4096):
        for mode in ("repeat", "gather4"):
            emb.prefill_mode = mode
            tp_dev = emb.prefill_tokens_device(torch.randint(0, cfg.vocab_size, (S,), generator=g), S)
            us, n = device_us(lambda: emb.forward_prefill(tp_dev), reps=2)
            _report(f"profile: embedding prefill {mode} S={S}: {us:.1f} us device ({n})")
            _free(tp_dev)
    emb.prefill_mode = "repeat"
    X = emb.forward_decode(tok)
    for split in ("mesh", "tp"):
        head = MotifLMHead(mesh_device, cfg, source=src, ccl=ccl, cache=False, vocab_split=split)
        grids = [(8, 4), None, (4, 4), (8, 8), (8, 2)] if split == "mesh" else [(8, 4)]
        for grid in grids:
            head.norm_grid = grid
            head._norm_sharded = sharded_norm_configs(cfg, grid) if grid is not None else None
            for reduce in (("sum", "wreduce") if grid == (8, 4) else ("sum",)):
                head.stream_reduce = reduce
                head.norm_eps = head.eps * 16 if reduce == "sum" else head.eps
                if reduce == "wreduce" and not head._w_mean:
                    head._w_mean[8] = ttnn.from_torch(torch.full((1, 4, 8, 1), 0.25), dtype=ttnn.float32,
                                                      layout=ttnn.TILE_LAYOUT, device=mesh_device,
                                                      memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                      mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
                us, n = device_us(lambda: head.stream_mean_norm(X))
                _report(f"profile: [{split}] stream mean ({reduce}) + rms_norm (grid {grid}): {us:.1f} us ({n})")
        head.norm_grid = (8, 4)
        head._norm_sharded = sharded_norm_configs(cfg, (8, 4))
        head.stream_reduce, head.norm_eps = "sum", head.eps * 16
        hn = head.stream_mean_norm(X)
        if split == "mesh":
            us, n = device_us(lambda: ccl.ag_dp_rows(hn))
            _report(f"profile: [mesh] ag_dp_rows [1,1,8,4096]: {us:.1f} us ({n})")
            gin = ccl.ag_dp_rows(hn)
        else:
            gin = hn
        specs = ["auto", None]
        if split == "mesh":
            specs += [((12, 9), 2, 8), ((12, 9), 2, 32), ((12, 6), 3, 16), ((9, 6), 4, 16), ((11, 4), 5, 16), ((12, 10), 2, 16)]
        else:
            specs += [((12, 9), 8, 8), ((12, 9), 8, 32), ((12, 8), 9, 16), ((11, 8), 10, 16), ((12, 10), 8, 16)]
        for spec in specs:
            try:
                pc = lm_head_program_config(cfg, split, spec)

                def mm():
                    return ttnn.linear(gin, head.weight, program_config=pc, compute_kernel_config=head.ckc_lm,
                                       dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                us, n = device_us(mm)
                gbs = cfg.hidden_size * head.vc * 2 / (us * 1e-6) / 1e9
                _report(f"profile: [{split}] LM-head GEMM {spec or 'default'}: {us:.1f} us ({gbs:.0f} GB/s weights; {n})")
            except Exception as e:
                _report(f"profile: [{split}] LM-head GEMM {spec}: ERROR {type(e).__name__}: {str(e)[:200]}")
        logits = head.forward_decode(X)
        us, n = device_us(lambda: head.forward_decode(X))
        _report(f"profile: [{split}] forward_decode total: {us:.1f} us ({n})")
        us, n = device_us(lambda: head.logits_rm(logits))
        _report(f"profile: [{split}] untilize logits: {us:.1f} us")
        for local in ("vector", "rm"):
            head.argmax_local = local
            if local == "vector" and head._am_iota is None:
                head._am_iota = ttnn.from_torch(
                    torch.arange(head.vc, dtype=torch.int32).reshape(1, 1, 1, -1).expand(1, 1, head.argmax_lanes, -1).contiguous(),
                    dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
            us, n = device_us(lambda: head.argmax_decode(logits))
            _report(f"profile: [{split}] argmax_decode (local {local}): {us:.1f} us ({n})")
        for S in (128, 4096):
            Xp = ttnn.from_torch(torch.randn(1, 4, S, 4096, generator=g).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
            us, n = device_us(lambda: head.forward_prefill(Xp, S - 1))
            _report(f"profile: [{split}] prefill head S={S} (tensor-args slice + sum + norm + GEMM + untilize): {us:.1f} us ({n})")
            _free(Xp)
        _free([hn, logits, head.weight] + ([gin] if split == "mesh" else []))
    _free([X, tok, emb.weight])
