# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only contract tests for the Motif-3 vLLM bridge (design 00 §4.7, §5.1; study 05 §3, §7).

Nothing here opens a device. Run them in a device-hidden namespace (the module skips itself otherwise)::

    cd $TT_METAL_HOME && unshare -Urm --propagation private bash -c \\
      'mount -t tmpfs none /dev/tenstorrent && source python_env/bin/activate && \\
       export TT_METAL_HOME=$PWD PYTHONPATH=$PWD && \\
       pytest models/demos/motif3/tests/test_generator_vllm_host.py'

Coverage:

* import hygiene: the bridge imports no ttnn, no vLLM, no other ``models/demos`` package and not the TT runtime;
* registration: ``TT_MODEL_CLASS_OVERRIDES`` registers the bare and the ``TT`` names; ``vllm_metadata.json`` /
  ``EXTRA_MODELS_DIR`` registers only ``TTMotifForCausalLM`` (the bare-name trap); vLLM's registry inspection
  classifies the class as a plain text-generation model;
* the real chain on the Motif checkpoint config: ``ModelConfig`` (trust_remote_code) resolves ``MotifForCausalLM`` to
  the bridge, ``VllmConfig`` runs ``TTPlatform.check_and_update_config`` with our capabilities, the plugin sizes the
  pool, calls ``get_kv_cache_spec``, vLLM builds the KV config, the runner derives the ``(N, 1, 64, 576)`` hint and
  the bridge allocates it; vLLM's ``BlockPool`` then holds 32 users x (8192 tokens + 1 block) after its null block;
  the tokenizer resolves with the reasoning / tool tokens as single ids;
* pool math, spec and allocation contracts, warmup order, release hooks, decode-reload contract v1 rejections;
* prefill/decode plumbing through the bridge with a fake ``MotifGenerator`` that emulates the device semantics the
  bridge relies on (KV replicated per DP group, decode writes only its lane's group, bucket-padded prefill writes),
  driven by the plugin's own state-slot bookkeeping (``TTModelRunner._alloc_prefill_state_slots`` /
  ``_decode_state_slot_remap`` / ...) and by block-table rows carrying stale ids, as vLLM's reused rows do;
* the real engine end to end: ``vllm.LLM`` -> TTPlatform -> TTWorker -> TTScheduler / TTModelRunner -> bridge -> fake
  generator, with only the mesh open/close patched (this is what found the stale block-table ids);
* the Motif reasoning and tool parser plugin files, loaded the way ``--*-parser-plugin`` loads them, and the
  documented ``vllm serve`` flags parsed by vLLM's own CLI parser.

The CPU golden package (``models/demos/motif3/reference``) is not needed: the fake generator's "model" is a
deterministic function of the token sequence, which is all the plumbing tests need.
"""

from __future__ import annotations

import functools
import json
import os
import random
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _visible_tt_devices():
    try:
        return sorted(os.listdir("/dev/tenstorrent"))
    except (FileNotFoundError, NotADirectoryError):
        return []


if _visible_tt_devices() and os.environ.get("MOTIF3_HOST_TEST_ALLOW_DEVICES") != "1":
    pytest.skip(
        "host-only vLLM bridge tests must run with the Tenstorrent devices hidden: "
        "unshare -Urm --propagation private bash -c 'mount -t tmpfs none /dev/tenstorrent && pytest ...' "
        "(or set MOTIF3_HOST_TEST_ALLOW_DEVICES=1 on a machine where touching the devices is harmless)",
        allow_module_level=True,
    )

from models.demos.motif3.tt import generator_api as api  # noqa: E402
from models.demos.motif3.tt import generator_vllm as gv  # noqa: E402

METAL_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = Path(__file__).resolve().parents[1]
_PROJECT = METAL_ROOT.parent
_CANDIDATE_DIRS = [
    os.environ.get("MOTIF3_HF_DIR"),
    os.environ.get("HF_MODEL"),
    str(_PROJECT / "weights" / "Motif-3"),
    str(_PROJECT / "hf_meta"),
]


def _motif_dir(require_tokenizer: bool = False) -> Path:
    for cand in _CANDIDATE_DIRS:
        if not cand:
            continue
        p = Path(cand)
        if (p / "config.json").is_file() and (p / "configuration_motif.py").is_file():
            if not require_tokenizer or (p / "tokenizer.json").is_file():
                return p
    pytest.skip(f"no Motif-3 config{' + tokenizer' if require_tokenizer else ''} dir among {_CANDIDATE_DIRS}")


# ================================================================================================================
# Fake generator: the device semantics the bridge depends on, on the host
# ================================================================================================================
def next_token(seq, vocab: int) -> int:
    """The fake model: a deterministic, position-sensitive function of the whole token sequence.

    Never below 100, so it never emits a Motif special / stop token (ids 0-84 are control tokens)."""
    s = torch.as_tensor(seq, dtype=torch.int64).reshape(-1)
    w = torch.arange(1, s.numel() + 1, dtype=torch.int64)
    return 100 + int(((s + 7) * w * w).sum().item() % (vocab - 100))


def one_hot(token: int, vocab: int, dtype=torch.float32) -> torch.Tensor:
    out = torch.full((vocab,), -1.0, dtype=dtype)
    out[token] = 1.0
    return out


class FakeMotifGenerator(api.MotifGenerator):
    """Emulates what matters to the bridge:

    * the latent pool is one copy per DP group (the real pool is replicated on every chip; a group's 8 chips agree):
      prefill writes every copy, decode writes only the copy of its lane's group ``lane // 8``;
    * prefill pads to its bucket and scribbles garbage at the padded positions (the request's last block / null block),
      like the real bucketed ``paged_fill_cache``;
    * a step's "logits" are a one-hot of ``next_token`` over the sequence read back through the page table from the
      lane's group copy, so a wrong lane group, page table, position or token changes the prediction;
    * inactive-lane rows are NaN, so the bridge must never hand them to vLLM for an active row.
    """

    def __init__(self, settings: api.GeneratorSettings, vocab_size: int, hf_config=None, mesh_device=None):
        self.settings = settings
        self._vocab = int(vocab_size)
        self.hf_config = hf_config
        self.mesh_device = mesh_device
        self.kv = None
        self.handle = None
        self.block_size = None
        self.alloc_args = None
        self.prefills = []  # (lane, seq_len, enable_trace)
        self.decode_steps = []  # (active lanes, enable_trace)
        self.warmups = []  # ("prefill"|"decode", enable_trace, page_table_width)
        self.released_lanes = []
        self.traces_released = 0
        self.fail_next_decode = False

    @classmethod
    def create(cls, *, hf_config, mesh_device, settings):
        return cls(settings, int(getattr(hf_config, "vocab_size", api.VOCAB_SIZE)), hf_config, mesh_device)

    @property
    def num_layers(self) -> int:
        return self.settings.num_layers

    @property
    def vocab_size(self) -> int:
        return self._vocab

    def allocate_kv_cache(self, *, num_blocks, block_size, num_layers):
        assert num_layers == self.num_layers
        self.alloc_args = dict(num_blocks=num_blocks, block_size=block_size, num_layers=num_layers)
        self.block_size = int(block_size)
        self.kv = torch.full((api.NUM_DP_GROUPS, num_blocks, block_size), -1, dtype=torch.int64)
        self.handle = ("fake-latent-pool", id(self))
        return self.handle

    def _read(self, group: int, page_table: torch.Tensor, upto: int) -> torch.Tensor:
        pos = torch.arange(upto + 1)
        return self.kv[group, page_table[pos // self.block_size].long(), pos % self.block_size]

    def prefill_forward(self, request, *, kv_cache, enable_trace=False):
        assert kv_cache is self.handle
        assert isinstance(request, api.PrefillRequest)
        s, bs, pt = request.seq_len, self.block_size, request.page_table
        assert bool((pt[: api.cdiv(s, bs)] >= 1).all()), "prompt positions on the null block"
        # Contract: the tail is zero. vLLM's rows can carry stale ids of OTHER requests' blocks there, and the
        # bucket-padding writes below would land in them.
        assert bool((pt[api.cdiv(s, bs) :] == 0).all()), f"prefill page-table tail not zeroed: {pt.tolist()}"
        pos = torch.arange(s)
        self.kv[:, pt[pos // bs].long(), pos % bs] = request.tokens.long()  # every chip
        bucket = next(b for b in api.prefill_buckets(self.settings.max_seq_len) if b >= s)
        pad = torch.arange(s, min(bucket, pt.numel() * bs))
        if pad.numel():
            self.kv[:, pt[pad // bs].long(), pad % bs] = -99  # bucket padding: overwritten by decode before any read
        self.prefills.append((request.lane, s, enable_trace))
        return one_hot(next_token(request.tokens, self._vocab), self._vocab, torch.bfloat16)

    def decode_forward(self, batch, *, kv_cache, enable_trace):
        assert kv_cache is self.handle
        assert isinstance(batch, api.DecodeBatch)
        if self.fail_next_decode:
            self.fail_next_decode = False
            raise RuntimeError("injected decode failure")
        active = batch.active
        assert bool((batch.tokens[~active] == 0).all()) and bool((batch.page_table[~active] == 0).all())
        out = torch.full((api.NUM_LANES, self._vocab), float("nan"))
        for lane in torch.nonzero(active).reshape(-1).tolist():
            group, p, pt = lane // api.LANES_PER_GROUP, int(batch.positions[lane]), batch.page_table[lane]
            assert int(pt[p // self.block_size]) >= 1, "decode position on the null block"
            assert bool((pt[p // self.block_size + 1 :] == 0).all()), f"decode page-table tail not zeroed: {pt}"
            self.kv[group, int(pt[p // self.block_size]), p % self.block_size] = int(batch.tokens[lane])
            out[lane] = one_hot(next_token(self._read(group, pt, p), self._vocab), self._vocab)
        self.decode_steps.append((torch.nonzero(active).reshape(-1).tolist(), enable_trace))
        return out

    def warmup_prefill(self, *, kv_cache, enable_trace):
        assert kv_cache is self.handle
        self.warmups.append(("prefill", enable_trace, None))

    def warmup_decode(self, *, kv_cache, enable_trace, page_table_width):
        assert kv_cache is self.handle
        self.warmups.append(("decode", enable_trace, page_table_width))

    def release_lane(self, lane):
        self.released_lanes.append(int(lane))

    def release_traces(self):
        self.traces_released += 1


def _bridge(num_slots=32, max_seq_len=1024, num_layers=3, vocab=4096, kv_dtype="bfp8"):
    settings = api.GeneratorSettings(
        max_batch_size=num_slots, max_seq_len=max_seq_len, num_layers=num_layers, kv_cache_dtype=kv_dtype
    )
    gen = FakeMotifGenerator(settings, vocab)
    return gv.MotifForCausalLM(gen, settings), gen


# ================================================================================================================
# vLLM-side driver: the parts of TTModelRunner that talk to the model, with the plugin's own slot bookkeeping
# ================================================================================================================
class PluginDriver:
    """Feeds the bridge exactly what ``vllm_tt_plugin.model_runner`` / ``async_decode`` feed it.

    State slots come from the plugin's ``TTModelRunner`` methods, called unbound on a stand-in runner (the plugin's
    own ``tests/test_state_slots.py`` pattern). Blocks come from a free list that never hands out block 0 (vLLM's
    null block). Every sampled token is checked against the fake model's ground truth.
    """

    def __init__(self, bridge, kv, *, num_slots, block_size, num_blocks, width, vocab, stale_tails=True):
        from vllm_tt_plugin.model_runner import TTModelRunner

        self.R = TTModelRunner
        self.bridge, self.kv = bridge, kv
        self.num_slots, self.bs, self.width, self.vocab = num_slots, block_size, width, vocab
        self.runner = SimpleNamespace(
            tt_per_lane_max_num_seqs=num_slots,
            _req_state_slot={},
            _pending_state_slot_settle=None,
            _pending_state_slot_moves=None,
            requests={},
            model=bridge,
        )
        self.free = list(range(num_blocks - 1, 0, -1))
        self.seqs = {}
        self.blocks = {}
        self.order = []  # persistent-batch row order of running requests
        self.remaps = 0
        self.stale_tails = stale_tails
        self.stale_rng = random.Random(99)
        self.lane_of = {}  # request -> the lane it was prefilled on (must not change until it is re-prefilled)
        self.lane_checks = 0

    def add(self, rid, prompt):
        self.seqs[rid] = list(prompt)
        self.blocks[rid] = []

    def _grow(self, rid, n_tokens):
        while len(self.blocks[rid]) < api.cdiv(n_tokens, self.bs):
            self.blocks[rid].append(self.free.pop())

    def _row_table(self, rid):
        row = torch.zeros(self.width, dtype=torch.int32)
        n = len(self.blocks[rid])
        row[:n] = torch.tensor(self.blocks[rid], dtype=torch.int32)
        if self.stale_tails:
            # vLLM's persistent block table is not cleared when a row is reused: entries past a request's own
            # blocks can hold stale ids. Emulate the worst case, blocks that other live requests own now.
            others = [b for r, owned in self.blocks.items() if r != rid for b in owned]
            k = min(self.width - n, len(others), self.stale_rng.randrange(0, 4))
            if k:
                row[n : n + k] = torch.tensor(self.stale_rng.sample(others, k), dtype=torch.int32)
        return row

    def _check_and_append(self, rids, logits):
        for rid, row in zip(rids, logits, strict=True):
            want = next_token(self.seqs[rid], self.vocab)
            got = int(torch.nan_to_num(row.float(), nan=-1e9).argmax())
            assert got == want, f"request {rid}: sampled {got}, ground truth {want} (len {len(self.seqs[rid])})"
            self.seqs[rid].append(got)

    def prefill(self, rids):
        rids = list(rids)
        slots = self.R._alloc_prefill_state_slots(self.runner, rids)
        self.runner.requests.update(dict.fromkeys(rids))
        lens = [len(self.seqs[r]) for r in rids]
        for r, n in zip(rids, lens, strict=True):
            self._grow(r, n)
        tokens = torch.full((len(rids), max(lens)), 4321, dtype=torch.int32)  # stale past each prompt
        for i, r in enumerate(rids):
            tokens[i, : lens[i]] = torch.tensor(self.seqs[r], dtype=torch.int32)
        for r, slot in zip(rids, slots, strict=True):
            self.lane_of[r] = self.bridge._lanes.lane_of_slot(slot)
        out = self.bridge.prefill_forward(
            tokens=tokens,
            page_table=torch.stack([self._row_table(r) for r in rids]),
            kv_cache=self.kv,
            enable_trace=False,
            prompt_lens=np.array(lens, dtype=np.int64),
            start_pos=np.zeros(len(rids), dtype=np.int32),
            empty_slots=list(slots),
        )
        assert tuple(out.shape) == (len(rids), 1, self.vocab)
        self._check_and_append(rids, out[:, -1, :])
        # After a prefill step the running requests come back behind the prefilled rows (vLLM re-adds them).
        self.order = rids + [r for r in self.order if r not in rids]
        return slots

    def decode(self, rows=None):
        rows = list(self.order if rows is None else rows)
        remap = self.R._decode_state_slot_remap(self.runner, rows)
        tokens = torch.zeros(self.num_slots, 1, dtype=torch.int32)
        pos = torch.full((self.num_slots,), -1, dtype=torch.int32)
        pt = torch.zeros(self.num_slots, self.width, dtype=torch.int32)
        for i, r in enumerate(rows):
            n = len(self.seqs[r])
            self._grow(r, n)
            tokens[i, 0], pos[i], pt[i] = self.seqs[r][-1], n - 1, self._row_table(r)
        kwargs = dict(
            tokens=tokens,
            page_table=pt,
            kv_cache=self.kv,
            start_pos=pos,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
            enable_trace=True,
            read_from_device=False,
        )
        if remap is not None:
            kwargs["slot_remap"] = remap
            self.remaps += 1
        out = self.bridge.decode_forward(**kwargs)
        self.R.note_decode_state_slots_settled(self.runner)
        for r in rows:  # a request keeps its lane (= its DP group's KV copy) for as long as it lives
            assert self.bridge._lanes.lane_of_slot(self.runner._req_state_slot[r]) == self.lane_of[r], r
            self.lane_checks += 1
        assert tuple(out.shape) == (self.num_slots, 1, self.vocab)
        self._check_and_append(rows, out[: len(rows), -1, :])
        self.order = rows
        return remap

    def _release(self, rid, preempted):
        self.R._release_model_request(self.runner, rid)
        out = SimpleNamespace(
            finished_req_ids=set() if preempted else {rid}, preempted_req_ids={rid} if preempted else None
        )
        self.R._release_dead_state_slots(self.runner, out)
        self.free.extend(self.blocks[rid])
        self.blocks[rid] = []
        if rid in self.order:  # condense: the last row moves into the hole
            i = self.order.index(rid)
            last = self.order.pop()
            if last != rid:
                self.order[i] = last

    def finish(self, rid):
        self._release(rid, preempted=False)
        self.runner.requests.pop(rid, None)
        self.seqs.pop(rid)
        self.blocks.pop(rid)

    def preempt(self, rid):
        self._release(rid, preempted=True)  # keeps its tokens; resumes with a full re-prefill


# ================================================================================================================
# 1. Import hygiene and registration
# ================================================================================================================
def test_bridge_import_is_device_free_and_lazy():
    """vLLM imports the bridge in the API server, the registry subprocess and EngineCore before any mesh exists."""
    probe = (
        "import json, sys\n"
        "import models.demos.motif3.tt.generator_vllm as m\n"
        "import models.demos.motif3.vllm_plugins as p\n"
        "mods = sorted(k for k in sys.modules if k.startswith('models.'))\n"
        "print(json.dumps({'models': mods, 'ttnn': 'ttnn' in sys.modules, 'vllm': 'vllm' in sys.modules,"
        " 'tokens': m.MotifForCausalLM.get_max_tokens_all_users(max_model_len=32768, max_num_seqs=32)}))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(METAL_ROOT))
    env.pop("MOTIF3_KV_POOL_TOKENS", None)
    res = subprocess.run([sys.executable, "-c", probe], cwd=METAL_ROOT, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]
    info = json.loads(res.stdout.strip().splitlines()[-1])
    # The project rule (design 00 §2.1): no other models/** package at import time (demo imports opened the cluster in
    # the prior port). ttnn itself is allowed, though the bridge and generator_api do not need it.
    foreign = [
        m for m in info["models"] if m not in ("models", "models.demos") and not m.startswith("models.demos.motif3")
    ]
    assert not foreign, f"the bridge import pulled in other models/** packages: {foreign}"
    # The TT runtime (mesh, weights) is imported lazily by initialize_vllm_model, and vLLM is never imported here.
    heavy = {"models.demos.motif3.tt.generator", "models.demos.motif3.tt.model", "models.demos.motif3.tt.weights"}
    assert not heavy & set(info["models"]) and info["vllm"] is False
    assert info["tokens"] == 262144 + gv.NULL_BLOCK_RESERVE_TOKENS


def test_vllm_metadata_json_names_the_bridge():
    meta = json.loads((PACKAGE_DIR / "vllm_metadata.json").read_text())
    assert meta == {"arch": gv.ARCHITECTURE, "main_class": gv.MAIN_CLASS}
    module, cls_name = meta["main_class"].split(":")
    import importlib

    assert getattr(importlib.import_module(module), cls_name) is gv.MotifForCausalLM
    assert gv.TT_MODEL_CLASS_OVERRIDES == "MotifForCausalLM=models.demos.motif3.tt.generator_vllm:MotifForCausalLM"


def _patched_registry(monkeypatch):
    from vllm.model_executor.models.registry import ModelRegistry

    registered = {}
    monkeypatch.setattr(ModelRegistry, "get_supported_archs", staticmethod(lambda: list(registered)))
    monkeypatch.setattr(
        ModelRegistry, "register_model", staticmethod(lambda arch, target: registered.__setitem__(arch, target))
    )
    return registered


def test_tt_model_class_overrides_registers_bare_and_tt_names(monkeypatch):
    import vllm.config  # noqa: F401  (finish vLLM init before the plugin package)
    import vllm_tt_plugin.platform as tt_platform

    registered = _patched_registry(monkeypatch)
    monkeypatch.delenv("EXTRA_MODELS_DIR", raising=False)
    monkeypatch.setenv("TT_MODEL_CLASS_OVERRIDES", gv.TT_MODEL_CLASS_OVERRIDES)
    tt_platform.register_tt_models()
    assert registered["MotifForCausalLM"] == gv.MAIN_CLASS
    assert registered["TTMotifForCausalLM"] == gv.MAIN_CLASS


def test_extra_models_dir_alone_registers_only_the_tt_name(monkeypatch, tmp_path):
    """The bare-name trap (design 00 §1.4): a bundle alone leaves ``MotifForCausalLM`` to upstream vLLM."""
    import vllm.config  # noqa: F401
    import vllm_tt_plugin.platform as tt_platform

    bundle = tmp_path / "motif-3-bh-galaxy"
    bundle.mkdir()
    (bundle / "vllm_metadata.json").write_text((PACKAGE_DIR / "vllm_metadata.json").read_text())
    registered = _patched_registry(monkeypatch)
    monkeypatch.delenv("TT_MODEL_CLASS_OVERRIDES", raising=False)
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))  # the plugin appends the bundle folder
    tt_platform.register_tt_models()
    assert registered["TTMotifForCausalLM"] == gv.MAIN_CLASS
    assert "MotifForCausalLM" not in registered

    from vllm.model_executor.models.registry import _PREVIOUSLY_SUPPORTED_MODELS

    assert "MotifForCausalLM" in _PREVIOUSLY_SUPPORTED_MODELS  # why the override is needed


FORBIDDEN_CLASS_ATTRS = (
    "is_hybrid",
    "has_inner_state",
    "is_attention_free",
    "supports_multimodal",
    "supports_pp",
    "is_pooling_model",
    "has_noops",
    "attn_type",
    "supports_transcription",
    "requires_raw_input_tokens",
    "supports_mamba_prefix_caching",
    "_HYBRID_KV_CACHE_GROUPS_ENABLED",
    "tt_supported_decode_batch_sizes",
    "already_warmed_up_prefill",
    "note_state_slots_moved",
)


def test_vllm_inspects_a_plain_text_generation_model():
    from vllm.model_executor.models.registry import _ModelInfo

    info = _ModelInfo.from_model_cls(gv.MotifForCausalLM)
    assert info.is_text_generation_model and not info.is_pooling_model
    assert not (info.is_hybrid or info.has_inner_state or info.is_attention_free or info.has_noops)
    assert not (info.supports_multimodal or info.supports_pp or info.supports_transcription)
    assert info.attn_type == "decoder"
    present = [a for a in FORBIDDEN_CLASS_ATTRS if hasattr(gv.MotifForCausalLM, a)]
    assert not present, f"vLLM/plugin read these with getattr; do not define them: {present}"
    with pytest.raises(TypeError):
        gv.MotifForCausalLM(vllm_config=object())  # no vLLM-native constructor


def test_model_capabilities_are_explicit_class_level():
    caps = gv.MotifForCausalLM.model_capabilities
    assert caps["supports_device_penalties"] is False  # plugin default for an absent key is True
    for key in (
        "supports_prefix_caching",
        "supports_chunked_prefill",
        "supports_async_decode",
        "supports_sample_on_device",
        "supports_spec_decode",
        "supports_async_spec_decode",
    ):
        assert caps[key] is False, key
    assert caps["output_tokens_per_step"] == 1
    assert "max_device_top_k" not in caps and "fabric_config" not in caps
    assert gv.MotifForCausalLM.decode_input_update_contract == 1


# ================================================================================================================
# 2. The real vLLM chain on the Motif config
# ================================================================================================================
@pytest.fixture(scope="module")
def motif_vllm_config(tmp_path_factory):
    """``VllmConfig`` for the real Motif-3 config, built the way ``vllm serve`` builds it on the TT platform.

    ``ModelConfig`` inspects the registered class in vLLM's registry subprocess, which imports the bridge in a fresh
    interpreter (it inherits this device-hidden namespace and ``PYTHONPATH``)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TT_MODEL_CLASS_OVERRIDES", gv.TT_MODEL_CLASS_OVERRIDES)
        mp.setenv("VLLM_CACHE_ROOT", str(tmp_path_factory.mktemp("vllm_cache")))
        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.setenv("PYTHONPATH", str(METAL_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        for var in ("EXTRA_MODELS_DIR", "MOTIF3_KV_POOL_TOKENS", "MOTIF3_NUM_LAYERS", "MOTIF3_KV_CACHE_DTYPE"):
            mp.delenv(var, raising=False)
        from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig

        import vllm_tt_plugin.platform as tt_platform

        tt_platform.register_tt_models()
        model_config = ModelConfig(model=str(_motif_dir()), trust_remote_code=True, max_model_len=32768, seed=0)
        resolved_arch = model_config.architecture
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=CacheConfig(block_size=64, enable_prefix_caching=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=32,
                max_num_batched_tokens=8192,
                max_model_len=32768,
                is_encoder_decoder=False,
                enable_chunked_prefill=True,
            ),
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            additional_config={"tt": {"trace_mode": "decode_only"}},
        )
        yield SimpleNamespace(vllm_config=vllm_config, model_config=model_config, resolved_arch=resolved_arch)


def test_vllm_resolves_motif_config_to_the_bridge(motif_vllm_config):
    from vllm.model_executor.model_loader import get_model_architecture

    vc, mc = motif_vllm_config.vllm_config, motif_vllm_config.model_config
    # ModelConfig (trust_remote_code) resolved the BARE name to us, not to TransformersMoEForCausalLM.
    assert motif_vllm_config.resolved_arch == "MotifForCausalLM"
    assert mc.hf_config.model_type == "Motif" and mc.trust_remote_code
    assert mc.hf_config.architectures == ["TTMotifForCausalLM"]  # rewritten by TTPlatform
    model_cls, _ = get_model_architecture(mc)
    assert model_cls is gv.MotifForCausalLM
    assert vc.parallel_config.worker_cls == "vllm_tt_plugin.worker.TTWorker"
    # Capabilities took effect at config time.
    assert vc.cache_config.enable_prefix_caching is False
    assert vc.scheduler_config.enable_chunked_prefill is False
    assert vc.scheduler_config.max_num_batched_tokens >= 32768
    assert not vc.scheduler_config.async_scheduling
    # What vLLM derives from the Motif config (study 05 §13).
    assert mc.is_moe and not mc.use_mla and not mc.uses_mrope
    assert mc.get_vocab_size() == 220160 and mc.max_model_len == 32768
    assert mc.get_num_layers_by_block_type(vc.parallel_config, "attention") == 53
    assert mc.get_sliding_window() == 128  # why the default FullAttentionSpec would be wrong


def test_kv_spec_pool_and_allocation_chain(motif_vllm_config):
    """Plugin sizing -> spec hook -> vLLM KV config -> runner hint -> bridge allocation -> vLLM BlockPool."""
    from vllm.config import set_current_vllm_config
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    from vllm_tt_plugin.model_runner import TTModelRunner
    from vllm_tt_plugin.worker import (
        TTWorker,
        _available_kv_cache_memory_bytes_for_num_blocks,
        get_num_available_blocks_tt,
    )

    vc, mc = motif_vllm_config.vllm_config, motif_vllm_config.model_config
    with set_current_vllm_config(vc):  # what EngineCore's init_device sees
        num_blocks = get_num_available_blocks_tt(vc, 32)
    assert num_blocks == (262144 + 64 * 32) // 64 + 1 == 4129

    spec = TTWorker._try_get_spec_from_model_hook(SimpleNamespace(model_config=mc, vllm_config=vc))
    assert len(spec) == 53
    assert all(isinstance(s, MLAAttentionSpec) for s in spec.values())
    first = spec["model.layers.0.self_attn"]
    assert (first.block_size, first.num_kv_heads, first.head_size, first.dtype) == (64, 1, 576, torch.bfloat16)
    assert first.sliding_window is None and first.page_size_bytes == 64 * 576 * 2

    available = _available_kv_cache_memory_bytes_for_num_blocks(vc, spec, num_blocks)
    vc.cache_config.num_gpu_blocks_override = num_blocks
    kv_cache_config = get_kv_cache_configs(vc, [spec], [available])[0]
    assert kv_cache_config.num_blocks == num_blocks
    assert len(kv_cache_config.kv_cache_groups) == 1
    assert len(kv_cache_config.kv_cache_groups[0].layer_names) == 53

    runner = SimpleNamespace(num_devices=32, tt_data_parallel_size=1)
    runner._kv_cache_shape = functools.partial(TTModelRunner._kv_cache_shape, runner)
    per_layer = TTModelRunner._build_per_layer_specs(runner, kv_cache_config, 53)
    assert [s[0] for s in per_layer] == [(4129, 1, 64, 576)] * 53
    assert [s[2] for s in per_layer] == list(range(53))

    settings = api.GeneratorSettings(max_batch_size=32, max_seq_len=32768, num_layers=53)
    gen = FakeMotifGenerator(settings, 220160)
    bridge = gv.MotifForCausalLM(gen, settings)
    kv = bridge.allocate_kv_cache_per_layer(per_layer)
    assert kv.shape == (4129, 1, 64, 576) and kv.num_layers == 53 and kv.kv_cache_dtype == "bfp8"
    assert kv.page_table_width == min(api.cdiv(32768, 64), num_blocks) == 512  # = plugin max_num_blocks_per_req
    assert kv.bytes_per_chip == 53 * 4129 * 2 * 18 * 1088 == 8_571_407_616  # design 00 §1.1: 8.57 GB
    assert gen.alloc_args == dict(num_blocks=4129, block_size=64, num_layers=53)

    pool = BlockPool(num_gpu_blocks=num_blocks, enable_caching=False, hash_block_size=64)
    per_user = api.cdiv(262144 // 32 + 1, 64)  # 8192 tokens + the next token's slot
    assert pool.get_num_free_blocks() == num_blocks - 1 == 32 * per_user
    # Without the reserve the plugin would allocate one block too few for the 32nd user.
    assert gv.plugin_num_blocks(262144, 64, 32) - 1 < 32 * per_user


def test_tokenizer_resolves_with_trust_remote_code(motif_vllm_config):
    from vllm.tokenizers import cached_tokenizer_from_config

    _motif_dir(require_tokenizer=True)
    tok = cached_tokenizer_from_config(motif_vllm_config.model_config)
    assert len(tok) == 220160
    assert tok.convert_tokens_to_ids(["<think>", "</think>", "<tool_call>", "</tool_call>"]) == [11, 12, 13, 14]
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    on = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    off = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert on.endswith("<|startofturn|><|assistant|><think>")
    assert off.endswith("<|startofturn|><|assistant|><think></think>")


# ================================================================================================================
# 3. Pool math, spec and allocation contracts
# ================================================================================================================
@pytest.mark.parametrize("block_size", api.SUPPORTED_BLOCK_SIZES)
def test_null_block_reserve_is_exactly_one_block(block_size, monkeypatch):
    monkeypatch.delenv("MOTIF3_KV_POOL_TOKENS", raising=False)
    tokens = gv.MotifForCausalLM.get_max_tokens_all_users(num_devices=32, max_model_len=32768, max_num_seqs=32)
    assert tokens == 262144 + gv.NULL_BLOCK_RESERVE_TOKENS
    blocks = gv.plugin_num_blocks(tokens, block_size, 32)
    assert blocks == 262144 // block_size + 32 + 1  # pool + one output block per user + vLLM's null block


def test_get_max_tokens_all_users_validation(monkeypatch):
    f = gv.MotifForCausalLM.get_max_tokens_all_users
    for var in ("MOTIF3_KV_POOL_TOKENS", "MOTIF3_KV_CACHE_DTYPE", "MOTIF3_NUM_LAYERS", "MOTIF3_KV_MAX_GB_PER_CHIP"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="tt_data_parallel"):
        f(num_devices=32, tt_data_parallel=4)
    with pytest.raises(ValueError, match="32-chip"):
        f(num_devices=8)
    with pytest.raises(ValueError, match="max-num-seqs"):
        f(num_devices=32, max_num_seqs=64)
    with pytest.raises(ValueError, match="max-model-len"):
        f(num_devices=32, max_model_len=262144)  # vLLM's derived default for Motif
    for bad in ("lots", "1000", "-5"):
        monkeypatch.setenv("MOTIF3_KV_POOL_TOKENS", bad)
        with pytest.raises(ValueError, match="MOTIF3_KV_POOL_TOKENS"):
            f(num_devices=32)
    monkeypatch.setenv("MOTIF3_KV_POOL_TOKENS", "16384")
    with pytest.raises(ValueError, match="does not fit"):
        f(num_devices=32, max_model_len=32768)
    monkeypatch.setenv("MOTIF3_KV_POOL_TOKENS", "393216")
    assert f(num_devices=32, max_model_len=32768, max_num_seqs=32) == 393216 + 32
    # A bf16 latent pool of 262,144 tokens needs ~16.2 GB per chip: refused before the weights load.
    monkeypatch.delenv("MOTIF3_KV_POOL_TOKENS")
    monkeypatch.setenv("MOTIF3_KV_CACHE_DTYPE", "bf16")
    with pytest.raises(ValueError, match="GB per chip"):
        f(num_devices=32, max_model_len=32768, max_num_seqs=32)
    monkeypatch.setenv("MOTIF3_KV_POOL_TOKENS", "131072")
    assert f(num_devices=32, max_model_len=32768, max_num_seqs=32) == 131072 + 32
    # A truncated bring-up run accounts only the layers it runs.
    monkeypatch.delenv("MOTIF3_KV_POOL_TOKENS")
    monkeypatch.setenv("MOTIF3_NUM_LAYERS", "3")
    assert f(num_devices=32, max_model_len=32768) == 262144 + 32


def test_get_max_tokens_all_users_rejects_bad_block_size_early(monkeypatch):
    """Inside EngineCore's init_device the current VllmConfig is visible, so --block-size 16 fails before loading."""
    from vllm.config import set_current_vllm_config

    monkeypatch.delenv("MOTIF3_KV_POOL_TOKENS", raising=False)
    fake = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=53)),
    )
    with set_current_vllm_config(fake), pytest.raises(ValueError, match="block-size"):
        gv.MotifForCausalLM.get_max_tokens_all_users(num_devices=32, max_model_len=32768)


def test_kv_cache_bytes_per_chip():
    assert api.kv_cache_bytes_per_chip(4129, 64, 53, "bfp8") == 8_571_407_616
    assert api.kv_cache_bytes_per_chip(4129, 64, 53, "bf16") == 4129 * 64 * 576 * 2 * 53
    assert api.kv_cache_bytes_per_chip(4129, 64, 14, "bfp8") * 53 == api.kv_cache_bytes_per_chip(4129, 64, 53) * 14
    with pytest.raises(ValueError):
        api.kv_cache_bytes_per_chip(10, 16, 1)


def _fake_vllm_config(block_size=64, cache_dtype="auto", num_layers=53, hf_overrides=None):
    hf = SimpleNamespace(model_type="Motif", kv_lora_rank=512, qk_rope_head_dim=64, num_hidden_layers=num_layers)
    for k, v in (hf_overrides or {}).items():
        setattr(hf, k, v)
    model_config = SimpleNamespace(
        hf_config=hf,
        hf_text_config=hf,
        dtype=torch.bfloat16,
        get_num_layers_by_block_type=lambda parallel_config, block_type="attention": num_layers,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(block_size=block_size, cache_dtype=cache_dtype),
        parallel_config=SimpleNamespace(),
    )


def test_kv_cache_spec_contract():
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    from vllm_tt_plugin.model_runner import _parse_layer_index

    spec = gv.MotifForCausalLM.get_kv_cache_spec(_fake_vllm_config())
    assert list(spec) == [f"model.layers.{i}.self_attn" for i in range(53)]
    assert [_parse_layer_index(k) for k in spec] == list(range(53))
    assert len({v for v in spec.values()}) == 1  # uniform -> one vLLM KV cache group
    s = spec["model.layers.52.self_attn"]
    assert isinstance(s, MLAAttentionSpec) and (s.num_kv_heads, s.head_size, s.block_size) == (1, 576, 64)
    for bs in api.SUPPORTED_BLOCK_SIZES:
        assert (
            gv.MotifForCausalLM.get_kv_cache_spec(_fake_vllm_config(block_size=bs))[
                "model.layers.0.self_attn"
            ].block_size
            == bs
        )
    with pytest.raises(ValueError, match="block-size"):
        gv.MotifForCausalLM.get_kv_cache_spec(_fake_vllm_config(block_size=16))  # vLLM's default
    with pytest.raises(ValueError, match="576"):
        gv.MotifForCausalLM.get_kv_cache_spec(_fake_vllm_config(hf_overrides={"kv_lora_rank": 128}))
    fp8 = gv.MotifForCausalLM.get_kv_cache_spec(_fake_vllm_config(cache_dtype="fp8"))["model.layers.0.self_attn"]
    assert fp8.dtype != torch.bfloat16  # vLLM bookkeeping follows --kv-cache-dtype; the device dtype does not


def test_allocate_kv_cache_contract(monkeypatch):
    monkeypatch.delenv("MOTIF3_KV_MAX_GB_PER_CHIP", raising=False)
    bridge, gen = _bridge(num_layers=53, max_seq_len=32768)
    with pytest.raises(ValueError, match="FullAttentionSpec"):
        bridge.allocate_kv_cache((4129, 1, 64, 192), torch.bfloat16, 53)  # the plugin's default-spec hint
    with pytest.raises(ValueError, match="576"):
        bridge.allocate_kv_cache((4129, 16, 64, 576), torch.bfloat16, 53)
    with pytest.raises(ValueError, match="block-size"):
        bridge.allocate_kv_cache((4129, 1, 16, 576), torch.bfloat16, 53)
    with pytest.raises(ValueError, match="generator runs"):
        bridge.allocate_kv_cache((4129, 1, 64, 576), torch.bfloat16, 52)
    with pytest.raises(ValueError, match="uniform"):
        bridge.allocate_kv_cache_per_layer(
            [((4129, 1, 64, 576), torch.bfloat16, 0), ((4129, 1, 32, 576), torch.bfloat16, 1)]
        )
    with pytest.raises(ValueError, match="share"):
        bridge.allocate_kv_cache_per_layer(
            [((4129, 1, 64, 576), torch.bfloat16, 0), ((4129, 1, 64, 576), torch.bfloat16, 0)]
        )
    assert gen.alloc_args is None  # nothing reached the generator
    kv = bridge.allocate_kv_cache((4129, 1, 64, 576), torch.bfloat16, 53)
    assert isinstance(kv, gv.MotifKVCache) and kv.device_cache is gen.handle
    with pytest.raises(RuntimeError, match="twice"):
        bridge.allocate_kv_cache((4129, 1, 64, 576), torch.bfloat16, 53)
    # Truncated bring-up run: vLLM still accounts 53 layers; the generator allocates the 3 it runs.
    bridge3, gen3 = _bridge(num_layers=3, max_seq_len=32768)
    kv3 = bridge3.allocate_kv_cache_per_layer([((4129, 1, 64, 576), torch.bfloat16, i) for i in range(53)])
    assert kv3.num_layers == 3 and kv3.vllm_num_layers == 53 and gen3.alloc_args["num_layers"] == 3
    # Memory budget: a bf16 pool of this size does not fit next to the weights.
    bridge_bf16, _ = _bridge(num_layers=53, max_seq_len=32768, kv_dtype="bf16")
    with pytest.raises(ValueError, match="GB per chip"):
        bridge_bf16.allocate_kv_cache((4129, 1, 64, 576), torch.bfloat16, 53)
    other = gv.MotifKVCache(1, 64, 1, 1, "bfp8", None, 1, 0, None)
    with pytest.raises(ValueError, match="allocate_kv_cache returned"):
        bridge.decode_forward(
            tokens=torch.zeros(32, 1, dtype=torch.int32),
            start_pos=torch.full((32,), -1),
            page_table=torch.zeros(32, 512, dtype=torch.int32),
            kv_cache=other,
        )


# ================================================================================================================
# 4. initialize_vllm_model
# ================================================================================================================
@pytest.fixture
def fake_generator_class(monkeypatch):
    module = types.ModuleType("motif3_host_test_fake_generator")
    module.FakeMotifGenerator = FakeMotifGenerator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("MOTIF3_GENERATOR_CLASS", f"{module.__name__}:FakeMotifGenerator")
    for var in ("MOTIF3_NUM_LAYERS", "MOTIF3_KV_CACHE_DTYPE", "TT_CACHE_PATH", "TT_MODEL_WEIGHTS_REVISION"):
        monkeypatch.delenv(var, raising=False)
    return FakeMotifGenerator


@pytest.fixture(scope="module")
def motif_hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(_motif_dir()), trust_remote_code=True)


def test_initialize_vllm_model_builds_the_generator(fake_generator_class, motif_hf_config, monkeypatch):
    monkeypatch.setenv("HF_MODEL", "/snapshots/motif-3")
    monkeypatch.setenv("TT_CACHE_PATH", "/tt_cache/motif3")
    mesh = SimpleNamespace(shape=(4, 8))
    model = gv.MotifForCausalLM.initialize_vllm_model(
        motif_hf_config, mesh, max_batch_size=32, max_seq_len=32768, tt_data_parallel=1, optimizations=None
    )
    assert isinstance(model, gv.MotifForCausalLM)
    gen = model.generator
    assert isinstance(gen, fake_generator_class) and gen.mesh_device is mesh and gen.hf_config is motif_hf_config
    s = gen.settings
    assert (s.max_batch_size, s.max_seq_len, s.num_layers, s.kv_cache_dtype) == (32, 32768, 53, "bfp8")
    assert (s.weights_path, s.cache_path) == ("/snapshots/motif-3", "/tt_cache/motif3")
    assert model.vocab_size == 220160

    # The plugin's BH-Galaxy preset opens (8, 4); the model detects the TP axis itself.
    gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, SimpleNamespace(shape=(8, 4)), 32, 32768)
    monkeypatch.setenv("MOTIF3_NUM_LAYERS", "3")
    small = gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 8, 4096)
    assert (small.generator.num_layers, small.settings.max_batch_size, small._lanes.num_slots) == (3, 8, 8)
    monkeypatch.delenv("MOTIF3_NUM_LAYERS")

    with pytest.raises(ValueError, match="MESH_DEVICE"):
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, SimpleNamespace(shape=(1, 32)), 32, 32768)
    with pytest.raises(ValueError, match="DP=1"):
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768, tt_data_parallel=4)
    with pytest.raises(ValueError, match="max_batch_size"):
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 64, 32768)
    with pytest.raises(ValueError, match="max_seq_len"):
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 65536)
    monkeypatch.setenv("MOTIF3_GENERATOR_CLASS", "models.demos.motif3.tt.generator_api:GeneratorSettings")
    with pytest.raises(TypeError, match="MotifGenerator"):
        gv.MotifForCausalLM.initialize_vllm_model(motif_hf_config, mesh, 32, 32768)


# ================================================================================================================
# 5. Prefill / decode plumbing with the plugin's slot bookkeeping
# ================================================================================================================
def _allocated_bridge(num_slots, block_size=32, num_blocks=640, max_seq_len=1024, vocab=4096):
    bridge, gen = _bridge(num_slots=num_slots, max_seq_len=max_seq_len, vocab=vocab)
    kv = bridge.allocate_kv_cache((num_blocks, 1, block_size, 576), torch.bfloat16, 3)
    driver = PluginDriver(
        bridge,
        kv,
        num_slots=num_slots,
        block_size=block_size,
        num_blocks=num_blocks,
        width=kv.page_table_width,
        vocab=vocab,
    )
    return bridge, gen, kv, driver


@pytest.mark.parametrize("num_slots", [32, 8])
def test_prefill_decode_plumbing_follows_slots(num_slots):
    """Random serving traffic: prefills into free slots, decodes with row reorders (non-identity slot_remap),
    finishes with condense, preemption + resume. Every sampled token must match the fake model's ground truth,
    which it only can if each request keeps its DP group between decode steps."""
    bridge, gen, kv, d = _allocated_bridge(num_slots)
    rng = random.Random(1234 + num_slots)
    next_id, waiting, preempted = 0, [], []

    def new_request():
        nonlocal next_id
        rid = f"r{next_id}"
        next_id += 1
        d.add(rid, [rng.randrange(1, 4000) for _ in range(rng.randrange(1, 70))])
        return rid

    for _ in range(min(3, num_slots)):
        waiting.append(new_request())
    d.prefill(waiting)
    for _ in range(60):
        if not d.order:
            d.prefill([new_request()])
        running = len(d.order)
        if preempted and running < num_slots and rng.random() < 0.3:
            d.prefill([preempted.pop(0)])  # resume: full re-prefill of prompt + generated tokens
        elif running < num_slots and rng.random() < 0.35:
            d.prefill([new_request() for _ in range(rng.randrange(1, min(4, num_slots - running) + 1))])
        rows = list(d.order)
        if rng.random() < 0.4:
            rng.shuffle(rows)  # any row order must work: the remap tells the bridge who is where
        d.decode(rows)
        if d.order and rng.random() < 0.2:
            d.finish(rng.choice(d.order))
        if len(d.order) > 1 and rng.random() < 0.1:
            victim = rng.choice(d.order)
            d.preempt(victim)
            preempted.append(victim)
    assert d.remaps > 5, "the traffic never exercised a non-identity slot_remap"
    assert d.lane_checks > 100, "decode never re-checked lane stability"
    assert gen.released_lanes, "finish/preempt must release the request's lane"
    lanes = bridge._lanes.slot_to_lane
    assert len(set(lanes)) == num_slots and all(0 <= lane < api.NUM_LANES for lane in lanes)


def test_lanes_spread_over_dp_groups_and_stay_put():
    bridge, gen, kv, d = _allocated_bridge(32)
    for i in range(4):
        d.add(f"a{i}", [5 + i] * (10 + i))
    slots = d.prefill([f"a{i}" for i in range(4)])
    assert slots == [0, 1, 2, 3]
    assert [lane for lane, _, _ in gen.prefills] == [0, 8, 16, 24]  # one request per DP group
    d.decode()
    assert gen.decode_steps[-1][0] == [0, 8, 16, 24]
    d.decode(["a3", "a2", "a1", "a0"])  # rows reversed -> remap; lanes do not move
    assert gen.decode_steps[-1][0] == [0, 8, 16, 24]
    assert [bridge._lanes.lane_of_slot(d.runner._req_state_slot[f"a{i}"]) for i in range(4)] == [0, 8, 16, 24]


def test_decode_failure_does_not_commit_the_remap():
    bridge, gen, kv, d = _allocated_bridge(8)
    for i in range(3):
        d.add(f"b{i}", [11 * (i + 1)] * 5)
    d.prefill(["b0", "b1", "b2"])
    d.decode()
    before = bridge._lanes.slot_to_lane
    rows = ["b2", "b0", "b1"]
    remap = d.R._decode_state_slot_remap(d.runner, rows)
    assert remap is not None
    gen.fail_next_decode = True
    with pytest.raises(RuntimeError, match="injected"):
        bridge.decode_forward(
            tokens=torch.zeros(8, 1, dtype=torch.int32),
            start_pos=torch.tensor([len(d.seqs[r]) - 1 for r in rows] + [-1] * 5, dtype=torch.int32),
            page_table=torch.stack([d._row_table(r) for r in rows] + [torch.zeros(d.width, dtype=torch.int32)] * 5),
            kv_cache=kv,
            slot_remap=remap,
            reload_inputs=True,
        )
    assert bridge._lanes.slot_to_lane == before  # a refused decode never moved anything
    d.runner._pending_state_slot_settle = None  # the plugin drops the pending map when the call raised
    d.runner._pending_state_slot_moves = None
    d.decode(rows)  # and the next attempt works from the unchanged state
    d.decode()


def test_decode_contract_rejections():
    bridge, gen, kv, d = _allocated_bridge(8)
    d.add("c0", [3, 4, 5])
    d.prefill(["c0"])
    base = dict(
        tokens=torch.zeros(8, 1, dtype=torch.int32),
        start_pos=torch.tensor([3] + [-1] * 7, dtype=torch.int32),
        page_table=torch.stack([d._row_table("c0")] + [torch.zeros(d.width, dtype=torch.int32)] * 7),
        kv_cache=kv,
    )
    with pytest.raises(TypeError, match="reset_batch"):
        bridge.decode_forward(**base, reset_batch=False)
    with pytest.raises(NotImplementedError, match="reload"):
        bridge.decode_forward(**base, reload_inputs=False, reload_page_table=True)
    with pytest.raises(ValueError, match="reload_page_table"):
        bridge.decode_forward(**base, reload_inputs=True, reload_page_table=True)
    with pytest.raises(NotImplementedError, match="host"):
        bridge.decode_forward(**base, sampling_params=SimpleNamespace(temperature=[0.0] * 8))
    with pytest.raises(NotImplementedError, match="page_tables_per_layer"):
        bridge.decode_forward(**base, page_tables_per_layer=[base["page_table"]])
    with pytest.raises(NotImplementedError, match="speculative"):
        bridge.decode_forward(**base, num_valid_drafts=torch.zeros(8, dtype=torch.int32))
    with pytest.raises(NotImplementedError, match="one token per row"):
        bridge.decode_forward(**{**base, "tokens": torch.zeros(8, 2, dtype=torch.int32)})
    with pytest.raises(ValueError, match="permutation"):
        bridge.decode_forward(**base, slot_remap=torch.tensor([0, 0, 1, 2, 3, 4, 5, 6], dtype=torch.int32))
    with pytest.raises(ValueError, match="null block"):
        bridge.decode_forward(**{**base, "start_pos": torch.tensor([40] + [-1] * 7, dtype=torch.int32)})
    with pytest.raises(ValueError, match="block ids"):
        bad = base["page_table"].clone()
        bad[0, 0] = kv.num_blocks
        bridge.decode_forward(**{**base, "page_table": bad})
    assert gen.decode_steps == []  # nothing reached the generator
    out = bridge.decode_forward(**base, enable_trace=False, reload_sampling_params=False, reset_sampling_state=False)
    assert tuple(out.shape) == (8, 1, 4096) and gen.decode_steps[-1] == ([0], False)
    # Narrower page tables are padded, wider ones must only carry null-block zeros past the context.
    wide = torch.nn.functional.pad(base["page_table"], (0, 7))
    bridge.decode_forward(**{**base, "page_table": wide, "start_pos": torch.tensor([4] + [-1] * 7, dtype=torch.int32)})
    assert bridge.read_decode_output(out) is out and bridge.read_decode_output(out, async_read=True) == (out, [])
    assert bridge.process_decode_output_host(out) is out
    with pytest.raises(NotImplementedError):
        bridge.process_decode_output_host(out, is_tokens=True)


def test_prefill_contract():
    bridge, gen, kv, d = _allocated_bridge(8)
    d.add("p0", list(range(1, 41)))
    d.add("p1", list(range(100, 107)))
    d._grow("p0", 40)
    d._grow("p1", 7)
    tokens = torch.full((2, 40), 4321, dtype=torch.int32)
    tokens[0, :40] = torch.tensor(d.seqs["p0"], dtype=torch.int32)
    tokens[1, :7] = torch.tensor(d.seqs["p1"], dtype=torch.int32)
    pt = torch.stack([d._row_table("p0"), d._row_table("p1")])
    base = dict(tokens=tokens, page_table=pt, kv_cache=kv, prompt_lens=np.array([40, 7]), empty_slots=[3, 5])
    with pytest.raises(NotImplementedError, match="prefix caching"):
        bridge.prefill_forward(**base, start_pos=np.array([16, 0], dtype=np.int32))
    with pytest.raises(NotImplementedError, match="host"):
        bridge.prefill_forward(**base, sampling_params=SimpleNamespace())
    with pytest.raises(ValueError, match="distinct"):
        bridge.prefill_forward(**{**base, "empty_slots": [3, 3]})
    with pytest.raises(ValueError, match="null block"):
        bridge.prefill_forward(**{**base, "page_table": torch.zeros_like(pt)})
    with pytest.raises(ValueError, match="slot"):
        bridge.prefill_forward(**{**base, "empty_slots": [3, 8]})
    assert gen.prefills == []
    out = bridge.prefill_forward(**base, start_pos=np.zeros(2, dtype=np.int32), enable_trace=True)
    assert tuple(out.shape) == (2, 1, 4096) and out.dtype == torch.bfloat16
    lanes = bridge._lanes.slot_to_lane
    assert gen.prefills == [(lanes[3], 40, True), (lanes[5], 7, True)]  # stale tokens past prompt_lens were cut
    assert int(out[0, -1].argmax()) == next_token(d.seqs["p0"], 4096)
    assert int(out[1, -1].argmax()) == next_token(d.seqs["p1"], 4096)


def test_stale_block_table_tails_are_zeroed():
    """vLLM's persistent block-table rows keep stale ids past a reused row's length (seen in the end-to-end test
    below). The generator must only ever see zeros there, or a bucket-padded prefill overwrites another request's KV.
    """
    bridge, gen, kv, d = _allocated_bridge(8)
    d.add("s0", list(range(100, 140)))  # 40 tokens = 2 blocks of 32
    d._grow("s0", 40)
    row = torch.zeros(d.width, dtype=torch.int32)
    row[:2] = torch.tensor(d.blocks["s0"], dtype=torch.int32)
    row[2:4] = torch.tensor([77, 78], dtype=torch.int32)  # stale: blocks another request owns now
    out = bridge.prefill_forward(
        tokens=torch.tensor([d.seqs["s0"]], dtype=torch.int32),
        page_table=row[None],
        kv_cache=kv,
        prompt_lens=np.array([40]),
        start_pos=np.zeros(1, dtype=np.int32),
        empty_slots=[0],
    )
    assert int(out[0, -1].argmax()) == next_token(d.seqs["s0"], 4096)
    assert bool((gen.kv[:, [77, 78]] == -1).all())  # the bucket padding (positions 40..127) never reached them
    tokens = torch.zeros(8, 1, dtype=torch.int32)
    pos = torch.full((8,), -1, dtype=torch.int32)
    pt = torch.zeros(8, d.width, dtype=torch.int32)
    tokens[0, 0], pos[0], pt[0] = 5, 40, row
    pt[3, 0] = 78  # junk on an inactive row
    bridge.decode_forward(tokens=tokens, start_pos=pos, page_table=pt, kv_cache=kv, reload_inputs=True)
    assert bool((gen.kv[:, [77, 78]] == -1).all())  # the fake also asserts every tail it saw was zero


def test_warmup_and_lifecycle_hooks():
    bridge, gen, kv, d = _allocated_bridge(32)
    with pytest.raises(RuntimeError, match="prefill warmup"):
        bridge.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=32, num_blocks=kv.page_table_width)
    with pytest.raises(ValueError, match="device"):
        bridge.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=True)
    # The plugin's two-phase warmup with trace_mode="decode_only" (model_runner.py:3727-3781).
    bridge.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=False)
    bridge.warmup_model_decode(
        kv_cache=kv, enable_trace=False, max_batch_size=32, num_blocks=kv.page_table_width, can_sample_on_device=False
    )
    bridge.warmup_model_decode(
        kv_cache=kv, enable_trace=True, max_batch_size=32, num_blocks=kv.page_table_width, can_sample_on_device=False
    )
    assert gen.warmups == [("prefill", False, None), ("decode", False, 32), ("decode", True, 32)]
    assert kv.page_table_width == api.cdiv(1024, 32)
    d.add("w0", [1, 2, 3])
    slot = d.prefill(["w0"])[0]
    d.decode()
    d.finish("w0")
    assert gen.released_lanes == [bridge._lanes.lane_of_slot(slot)]
    bridge.release_persistent_capture()
    assert gen.traces_released == 1
    bridge.close()
    assert gen.traces_released == 2


def test_lane_map_unit():
    lm = gv.LaneMap(32)
    assert lm.slot_to_lane[:8] == (0, 8, 16, 24, 1, 9, 17, 25)
    assert sorted(lm.slot_to_lane) == list(range(32))
    assert [lm.group_of_slot(s) for s in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
    remap = list(range(32))
    remap[0], remap[5] = 5, 0
    assert lm.decode_lanes(32, remap)[:6] == [9, 8, 16, 24, 1, 0]
    assert lm.slot_to_lane[0] == 0  # not committed yet
    lm.commit(torch.tensor(remap, dtype=torch.int32))
    assert lm.slot_to_lane[0] == 9 and lm.slot_to_lane[5] == 0
    lm.commit(None)
    assert lm.slot_to_lane[0] == 9
    with pytest.raises(ValueError):
        lm.decode_lanes(33)
    with pytest.raises(ValueError):
        lm.commit(list(range(31)))
    with pytest.raises(ValueError):
        gv.LaneMap(33)
    small = gv.LaneMap(5)
    assert small.slot_to_lane == (0, 8, 16, 24, 1)


# ================================================================================================================
# 6. Motif parser plugins (loaded exactly like --reasoning-parser-plugin / --tool-parser-plugin)
# ================================================================================================================
@pytest.fixture(scope="module")
def motif_tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(str(_motif_dir(require_tokenizer=True)), trust_remote_code=True)


@pytest.fixture(scope="module")
def motif_parsers():
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    from models.demos.motif3 import vllm_plugins as vp

    ReasoningParserManager.import_reasoning_parser(vp.REASONING_PARSER_PLUGIN)
    ToolParserManager.import_tool_parser(vp.TOOL_PARSER_PLUGIN)
    reasoning = ReasoningParserManager.get_reasoning_parser(vp.REASONING_PARSER_NAME)
    tools = ToolParserManager.get_tool_parser(vp.TOOL_PARSER_NAME)
    # import_*_parser swallows exceptions, so check the classes really come from our files.
    assert reasoning.__module__ == "motif_reasoning_parser" and reasoning.__name__ == "MotifReasoningParser"
    assert tools.__module__ == "motif_tool_parser" and tools.__name__ == "MotifToolParser"
    assert ToolParserManager.get_tool_parser("motif_hermes") is tools
    return SimpleNamespace(reasoning=reasoning, tools=tools)


def _chat_request(**kwargs):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    return ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], model="Motif-3", **kwargs)


def test_parser_plugin_files_and_cli_args():
    from models.demos.motif3 import vllm_plugins as vp

    for path in (vp.REASONING_PARSER_PLUGIN, vp.TOOL_PARSER_PLUGIN):
        text = Path(path).read_text()
        assert "SPDX-License-Identifier: Apache-2.0" in text and "github.com/MotifTechnologies/vllm" in text
        assert "import ttnn" not in text and "from models" not in text
    args = vp.vllm_cli_args()
    assert args[args.index("--reasoning-parser") + 1] == "motif"
    assert args[args.index("--tool-call-parser") + 1] == "motif" and "--enable-auto-tool-choice" in args


def test_vllm_serve_cli_accepts_the_motif_flags():
    """The documented launch flags (design 00 §5.1) and the parser-plugin flags parse in vLLM 0.26's ``vllm serve``
    parser, and the plugins load and validate exactly as ``api_server.setup_server`` does it."""
    from vllm.entrypoints.openai.api_server import validate_api_server_args
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from models.demos.motif3 import vllm_plugins as vp

    tt = {"trace_mode": "decode_only", "trace_region_size": 268435456, "fabric_config": "FABRIC_2D_TORUS_XY"}
    tt["dispatch_core_axis"] = "col"
    argv = ["--model", str(_motif_dir()), "--trust-remote-code", "--max-num-seqs", "32", "--block-size", "64"]
    argv += ["--max-model-len", "32768", "--no-enable-prefix-caching", "--additional-config", json.dumps({"tt": tt})]
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(argv + vp.vllm_cli_args())
    validate_parsed_serve_args(args)
    assert (args.max_num_seqs, args.block_size, args.max_model_len) == (32, 64, 32768)
    assert args.trust_remote_code and args.enable_prefix_caching is False
    assert args.additional_config == {"tt": tt}
    assert (args.reasoning_parser, args.reasoning_parser_plugin) == ("motif", vp.REASONING_PARSER_PLUGIN)
    assert (args.tool_call_parser, args.tool_parser_plugin) == ("motif", vp.TOOL_PARSER_PLUGIN)
    assert args.enable_auto_tool_choice
    ToolParserManager.import_tool_parser(args.tool_parser_plugin)
    ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)
    validate_api_server_args(args)
    assert args.reasoning_parser in ReasoningParserManager.list_registered()


def test_reasoning_parser_splits_think_blocks(motif_parsers, motif_tokenizer):
    tok, req = motif_tokenizer, _chat_request()
    parser = motif_parsers.reasoning(tok)
    assert (parser.start_token_id, parser.end_token_id) == (11, 12)
    # Motif's generation prompt already opened <think>, so outputs usually start inside the block.
    assert parser.extract_reasoning("Add the numbers.</think>2 + 2 = 4.", req) == ("Add the numbers.", "2 + 2 = 4.")
    assert parser.extract_reasoning("<think>plan</think>answer", req) == ("plan", "answer")
    ids = tok.encode("x</think>y", add_special_tokens=False)
    assert parser.is_reasoning_end(ids) and not parser.is_reasoning_end(
        tok.encode("<think>x", add_special_tokens=False)
    )
    off = motif_parsers.reasoning(tok, chat_template_kwargs={"enable_thinking": False})
    text = "No thinking here </think> literally."
    assert off.extract_reasoning(text, req) == (None, text)
    assert off.is_reasoning_end([]) is True  # never gates tool calls when thinking is off

    # Streaming, token by token, as the OpenAI server feeds it.
    out_text = "Let me check: 2 + 2.</think>The answer is 4."
    out_ids = tok.encode(out_text, add_special_tokens=False)
    assert 12 in out_ids
    stream = motif_parsers.reasoning(tok)
    reasoning, content, prev_text, prev_ids = "", "", "", []
    for tid in out_ids:
        cur_ids = prev_ids + [tid]
        cur_text = tok.decode(cur_ids)
        delta = stream.extract_reasoning_streaming(
            prev_text, cur_text, cur_text[len(prev_text) :], prev_ids, cur_ids, [tid]
        )
        if delta is not None:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
        prev_text, prev_ids = cur_text, cur_ids
    assert (reasoning, content) == ("Let me check: 2 + 2.", "The answer is 4.")


def test_tool_parser_extracts_and_repairs_tool_calls(motif_parsers, motif_tokenizer):
    parser = motif_parsers.tools(motif_tokenizer)
    req = _chat_request()
    good = 'Checking the weather.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Seoul"}}\n</tool_call>'
    info = parser.extract_tool_calls(good, req)
    assert info.tools_called and info.content == "Checking the weather.\n"
    assert [c.function.name for c in info.tool_calls] == ["get_weather"]
    assert json.loads(info.tool_calls[0].function.arguments) == {"city": "Seoul"}
    # Motif's repair ladder: a missing "[" around a string list, and two calls in one turn.
    bad = (
        '<tool_call>{"name": "search", "arguments": {"queries": "tt-metal", "vllm"}}</tool_call>\n'
        '<tool_call>{"name": "fetch", "arguments": {"urls": ["http://x"]}}}}</tool_call>'
    )
    info = parser.extract_tool_calls(bad, req)
    assert [c.function.name for c in info.tool_calls] == ["search", "fetch"]
    assert json.loads(info.tool_calls[0].function.arguments) == {"queries": ["tt-metal", "vllm"]}
    assert json.loads(info.tool_calls[1].function.arguments) == {"urls": ["http://x"]}
    plain = parser.extract_tool_calls("Just an answer.", req)
    assert not plain.tools_called and plain.content == "Just an answer."
    broken = '<tool_call>{"name": totally broken</tool_call>'
    assert parser.extract_tool_calls(broken, req).tools_called is False

    # Streaming in 5-character deltas: the name streams early, the arguments once the block closes.
    stream = motif_parsers.tools(motif_tokenizer)
    names, args, prev = [], "", ""
    for i in range(0, len(good), 5):
        cur = good[: i + 5]
        delta = stream.extract_tool_calls_streaming(prev, cur, cur[len(prev) :], [], [], [], req)
        for call in (delta.tool_calls if delta is not None else []) or []:
            fn = call.function if isinstance(call.function, dict) else call.function.model_dump()
            names += [fn["name"]] if fn.get("name") else []
            args += fn.get("arguments") or ""
        prev = cur
    assert names == ["get_weather"] and json.loads(args) == {"city": "Seoul"}


def test_reasoning_then_tool_call(motif_parsers, motif_tokenizer):
    """The server runs the reasoning parser first, then the tool parser on the content."""
    req = _chat_request()
    call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Busan"}}\n</tool_call>'
    output = "I need the weather.</think>\n" + call
    reasoning, content = motif_parsers.reasoning(motif_tokenizer).extract_reasoning(output, req)
    assert reasoning == "I need the weather."
    info = motif_parsers.tools(motif_tokenizer).extract_tool_calls(content, req)
    assert info.tools_called and json.loads(info.tool_calls[0].function.arguments) == {"city": "Busan"}


# ================================================================================================================
# 7. The real vLLM engine, end to end on the host
# ================================================================================================================
class _FakeMesh:
    """Stands in for the ``ttnn.MeshDevice`` the plugin would open; the runner itself never touches it."""

    shape = (4, 8)

    def get_num_devices(self):
        return 32

    def get_submeshes(self):
        return []


def test_vllm_offline_engine_end_to_end(fake_generator_class, monkeypatch, tmp_path):
    """``vllm.LLM`` -> TTPlatform -> TTWorker -> TTScheduler / TTModelRunner -> MotifForCausalLM -> fake generator.

    Everything is vLLM's and the plugin's real code except opening and closing the mesh (there is no device here).
    Twelve greedy requests of different lengths on ``max_num_seqs=8`` run in several waves, so state slots are reused
    and rows are reordered; every generated token must be the fake model's next token.
    """
    import vllm_tt_plugin.worker as tt_worker

    model_dir = _motif_dir(require_tokenizer=True)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # EngineCore in this process (keeps the patches)
    monkeypatch.setenv("TT_MODEL_CLASS_OVERRIDES", gv.TT_MODEL_CLASS_OVERRIDES)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm_cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("MESH_DEVICE", "(4, 8)")
    monkeypatch.setenv("PYTHONPATH", str(METAL_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for var in ("EXTRA_MODELS_DIR", "MOTIF3_KV_POOL_TOKENS", "MOTIF3_KV_CACHE_DTYPE"):
        monkeypatch.delenv(var, raising=False)
    meshes = []

    def open_fake_mesh(*args, **kwargs):
        meshes.append(_FakeMesh())
        return meshes[-1]

    monkeypatch.setattr(tt_worker, "open_mesh_device", open_fake_mesh)
    monkeypatch.setattr(tt_worker, "close_mesh_device", lambda *args, **kwargs: None)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_dir),
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=8,
        block_size=64,
        enable_prefix_caching=False,
        seed=0,
        additional_config={"tt": {"trace_mode": "decode_only"}},
    )
    try:
        bridge = llm.llm_engine.model_executor.driver_worker.model_runner.model
        assert isinstance(bridge, gv.MotifForCausalLM)
        gen = bridge.generator
        assert isinstance(gen, fake_generator_class) and gen.mesh_device is meshes[0]
        width = api.cdiv(4096, 64)
        assert gen.warmups == [("prefill", False, None), ("decode", False, width), ("decode", True, width)]
        assert gen.alloc_args == dict(num_blocks=gv.plugin_num_blocks(262144 + 32, 64, 8), block_size=64, num_layers=53)

        rng = random.Random(7)
        prompts = [
            {"prompt_token_ids": [1, 5, 3] + [rng.randrange(100, 200000) for _ in range(rng.randrange(1, 200))]}
            for _ in range(12)
        ]
        params = [SamplingParams(temperature=0.0, max_tokens=rng.randrange(1, 12), ignore_eos=True) for _ in prompts]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        for prompt, p, out in zip(prompts, params, outputs, strict=True):
            seq, want = list(prompt["prompt_token_ids"]), []
            for _ in range(p.max_tokens):
                want.append(next_token(seq, api.VOCAB_SIZE))
                seq.append(want[-1])
            assert list(out.outputs[0].token_ids) == want
        assert len(gen.prefills) == len(prompts) and not any(trace for *_, trace in gen.prefills)
        assert gen.decode_steps and all(trace for _, trace in gen.decode_steps)  # trace_mode=decode_only
        assert max(len(lanes) for lanes, _ in gen.decode_steps) <= 8

        # The plugin delivers a finished request's release_request with the NEXT step's scheduler output, so a
        # second batch flushes the releases of the first one (and reuses its freed slots and lanes).
        extra = {"prompt_token_ids": [1, 5, 3, 4321, 8765]}
        out = llm.generate([extra], SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), use_tqdm=False)
        seq, want = list(extra["prompt_token_ids"]), []
        for _ in range(3):
            want.append(next_token(seq, api.VOCAB_SIZE))
            seq.append(want[-1])
        assert list(out[0].outputs[0].token_ids) == want
        assert len(gen.released_lanes) >= len(prompts)  # every finished request of the first batch released its lane
    finally:
        try:
            llm.llm_engine.engine_core.shutdown()
        except RuntimeError as exc:
            # vLLM's cleanup_dist_env_and_memory() ends with torch.accelerator.empty_cache(), which raises on a host
            # without a torch accelerator (a TT host included); the model-side teardown already ran before it.
            if "accelerator" not in str(exc):
                raise
    assert gen.traces_released == 1  # release_persistent_capture at shutdown
