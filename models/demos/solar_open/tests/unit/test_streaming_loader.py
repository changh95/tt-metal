# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the phase-2 streaming loader (``utils/streaming_loader.py::LazyStateDict``, DESIGN.md 4.16).

    pytest models/demos/solar_open/tests/unit/test_streaming_loader.py

Part A (always runs, < 5 s): a synthetic 3-shard safetensors checkpoint in the on-disk per-expert layout, compared with
the phase-1 semantics (``torch.stack`` / ``torch.cat`` fusion, ``convert_hf_qkv_to_meta_format``, the safety-net cast).
Part B (skips unless ``HF_MODEL`` is the real Solar-Open-100B snapshot with the shards of layer 0, the embedding, the
final norm and lm_head): the 16 tensors of layer 0 + embed / norm / lm_head are read through the loader one at a time,
sha256-hashed and compared with a reference produced WITHOUT loading the whole model - the raw per-expert tensors via
``safe_open`` plus the documented conversion, and a subprocess running the phase-1 path on
``AutoModelForCausalLM.from_pretrained(dtype=bf16, num_hidden_layers=1)`` (~11 GB peak, hashes cached under
``$TT_CACHE_PATH/streaming_loader_reference_layer0.json``; ``SOLAR_OPEN_REGEN_LOADER_REFERENCE=1`` regenerates). The
loader process must peak below 25 GB RSS. No device is opened.
"""

import copy
import hashlib
import json
import os
import pickle
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.solar_open.config import MeshConfig, ModeConfig
from models.demos.solar_open.tt import model_config as mc
from models.demos.solar_open.tt.attention.config import AttentionConfig
from models.demos.solar_open.tt.attention.weights import load_attention_weights
from models.demos.solar_open.tt.experts.config import ExpertConfig
from models.demos.solar_open.tt.experts.weights import prepare_expert_weights_torch
from models.demos.solar_open.tt.model_config import ModelArgs
from models.demos.solar_open.utils import streaming_loader as sl
from models.demos.solar_open.utils.streaming_loader import LAYER_KEY_ORDER, LazyStateDict
from models.demos.solar_open.utils.substate import has_substate, indexed_substates, substate
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format, permute, reverse_permute

BF16, F32 = torch.bfloat16, torch.float32
GB = 2**30

# ------------------------------------------------------------------------------------------------------------------
# Part A: synthetic checkpoint
# ------------------------------------------------------------------------------------------------------------------
TINY = dict(
    hidden=32,
    moe_intermediate=16,
    num_experts=4,
    n_q_heads=2,
    n_kv_heads=1,
    head_dim=16,
    shared_intermediate=24,  # != moe_intermediate so a shared/routed mix-up cannot pass by shape
    vocab=64,
    n_layers=2,
)
SHARDS = tuple(f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3))
FP32_STRAGGLER = "model.layers.1.post_attention_layernorm.weight"  # deliberately fp32 on disk -> bf16 on access


def _raw_tensors(cfg=TINY, seed=0):
    """The on-disk key set (per-expert experts, HF q/k order), deterministic."""
    gen = torch.Generator().manual_seed(seed)
    H, I, E, S = cfg["hidden"], cfg["moe_intermediate"], cfg["num_experts"], cfg["shared_intermediate"]
    Q, KV = cfg["n_q_heads"] * cfg["head_dim"], cfg["n_kv_heads"] * cfg["head_dim"]

    def rnd(*shape, dtype=BF16):
        return torch.randn(*shape, generator=gen).to(dtype)

    raw = {"model.embed_tokens.weight": rnd(cfg["vocab"], H)}
    for layer in range(cfg["n_layers"]):
        p = f"model.layers.{layer}."
        raw[p + "input_layernorm.weight"] = rnd(H)
        raw[p + "post_attention_layernorm.weight"] = rnd(
            H, dtype=F32 if p + "post_attention_layernorm.weight" == FP32_STRAGGLER else BF16
        )
        raw[p + "self_attn.q_proj.weight"] = rnd(Q, H)
        raw[p + "self_attn.k_proj.weight"] = rnd(KV, H)
        raw[p + "self_attn.v_proj.weight"] = rnd(KV, H)
        raw[p + "self_attn.o_proj.weight"] = rnd(H, Q)
        raw[p + "mlp.gate.weight"] = rnd(E, H)
        raw[p + f"mlp.gate.{sl.ROUTER_BIAS_SUFFIX}"] = torch.randn(E, generator=gen) * 1e-3  # fp32, stays fp32
        for e in range(E):
            raw[p + f"mlp.experts.{e}.gate_proj.weight"] = rnd(I, H)
            raw[p + f"mlp.experts.{e}.up_proj.weight"] = rnd(I, H)
            raw[p + f"mlp.experts.{e}.down_proj.weight"] = rnd(H, I)
        raw[p + "mlp.shared_experts.gate_proj.weight"] = rnd(S, H)
        raw[p + "mlp.shared_experts.up_proj.weight"] = rnd(S, H)
        raw[p + "mlp.shared_experts.down_proj.weight"] = rnd(H, S)
    raw["model.norm.weight"] = rnd(H)
    raw["lm_head.weight"] = rnd(cfg["vocab"], H)
    return raw


def _shard_of(key):
    """Shard 1: embedding + layer 0. Shard 2: layer 1 except experts 2-3. Shard 3: layer-1 experts 2-3 + norm + lm_head
    (layer 1 spans two shards like 40 of the 48 real layers)."""
    if key.startswith("model.embed_tokens.") or key.startswith("model.layers.0."):
        return SHARDS[0]
    if key.startswith("model.layers.1."):
        m = sl._EXPERT_KEY_RE.match(key)
        return SHARDS[2] if m and int(m["expert"]) >= 2 else SHARDS[1]
    return SHARDS[2]


def _write_checkpoint(root: Path, raw, cfg=TINY):
    root.mkdir(parents=True, exist_ok=True)
    by_shard = {}
    for key, tensor in raw.items():
        by_shard.setdefault(_shard_of(key), {})[key] = tensor.contiguous()
    for shard, tensors in by_shard.items():
        save_file(tensors, str(root / shard), metadata={"format": "pt"})
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in raw.values())},
        "weight_map": {key: _shard_of(key) for key in raw},
    }
    (root / sl.INDEX_FILE).write_text(json.dumps(index, indent=1))
    (root / "config.json").write_text(
        json.dumps({"head_dim": cfg["head_dim"], "num_local_experts": cfg["num_experts"], "hidden_size": cfg["hidden"]})
    )
    return root


def _reference_state_dict(raw, cfg=TINY):
    """Phase-1 semantics: transformers' fusion (stack_e(cat([gate_e, up_e])), stack_e(down_e)), Meta-permuted q/k, the
    safety-net cast of fp32 stragglers (never the router bias)."""
    E = cfg["num_experts"]
    sd = {k: v for k, v in raw.items() if not sl._EXPERT_KEY_RE.match(k)}
    for layer in range(cfg["n_layers"]):
        base = f"model.layers.{layer}.mlp.experts"
        sd[f"{base}.gate_up_proj"] = torch.stack(
            [torch.cat([raw[f"{base}.{e}.gate_proj.weight"], raw[f"{base}.{e}.up_proj.weight"]], 0) for e in range(E)]
        )
        sd[f"{base}.down_proj"] = torch.stack([raw[f"{base}.{e}.down_proj.weight"] for e in range(E)])
    sd = convert_hf_qkv_to_meta_format(sd, cfg["head_dim"])
    return {k: (v.to(BF16) if v.dtype == F32 and not k.endswith(sl.ROUTER_BIAS_SUFFIX) else v) for k, v in sd.items()}


EXPECTED_KEY_ORDER = (
    ["model.embed_tokens.weight"]
    + [f"model.layers.{l}.{k}" for l in range(TINY["n_layers"]) for k in LAYER_KEY_ORDER]
    + ["model.norm.weight", "lm_head.weight"]
)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    raw = _raw_tensors()
    root = _write_checkpoint(tmp_path_factory.mktemp("tiny_ckpt") / "Solar-Open-100B", raw)
    return SimpleNamespace(root=root, raw=raw, ref=_reference_state_dict(raw), cfg=TINY)


@pytest.fixture
def lazy(tiny_checkpoint):
    loader = LazyStateDict(tiny_checkpoint.root, head_dim=TINY["head_dim"], num_experts=TINY["num_experts"])
    yield loader
    loader.close()


def _walk_layer(view):
    """Read every key of a (sub)state once, the way the module constructors do."""
    return {k: view[k] for k in list(view)}


class TestSyntheticCheckpoint:
    def test_key_set_and_order(self, lazy, tiny_checkpoint, expect_error):
        assert isinstance(lazy, sl.LazyStateDict) and len(lazy) == 3 + TINY["n_layers"] * 13
        assert list(lazy) == EXPECTED_KEY_ORDER
        assert set(lazy) == set(tiny_checkpoint.ref)
        assert not any(sl._EXPERT_KEY_RE.match(k) for k in lazy), "per-expert keys must be collapsed"
        assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in lazy
        assert "model.layers.0.mlp.experts.gate_up_proj" in lazy and 42 not in lazy
        assert lazy.get("nope") is None and lazy.get("nope", 7) == 7
        with expect_error(KeyError, "nope"):
            lazy["nope"]
        assert lazy.num_shards == 3 and lazy.snapshot_dir == tiny_checkpoint.root
        assert lazy.stats["bytes"] == 0 and lazy.stats["preads"] == 0, "construction must not read tensor data"
        assert lazy.open_fds == 0
        assert lazy.layer_keys(1) == [f"model.layers.1.{k}" for k in LAYER_KEY_ORDER]
        assert "LazyStateDict(29 keys" in repr(lazy)

    def test_meta_matches_reference_without_reading(self, lazy, tiny_checkpoint, expect_error):
        for key, ref in tiny_checkpoint.ref.items():
            assert lazy.meta(key) == (tuple(ref.shape), ref.dtype), key
        assert lazy.meta(FP32_STRAGGLER)[1] == BF16  # the safety-net cast is visible in meta()
        assert lazy.meta(f"model.layers.0.mlp.gate.{sl.ROUTER_BIAS_SUFFIX}")[1] == F32
        with expect_error(KeyError, "gate_proj"):
            lazy.meta("model.layers.0.mlp.experts.0.gate_proj.weight")
        mc._validate_state_dict_layout(lazy)  # contract-C1 layout check on the lazy dict ...
        assert lazy.stats["bytes"] == 0 and lazy.stats["preads"] == 0, "... must be metadata only"

    def test_every_tensor_bit_exact_with_phase1_semantics(self, lazy, tiny_checkpoint):
        raw, ref = tiny_checkpoint.raw, tiny_checkpoint.ref
        for key, expected in ref.items():
            got = lazy[key]
            assert got.dtype == expected.dtype and tuple(got.shape) == tuple(expected.shape), key
            assert torch.equal(got, expected), key
            assert got.data_ptr() != expected.data_ptr()  # a fresh tensor owned by the caller
        # the facts behind the equality, spelled out
        I = TINY["moe_intermediate"]
        gate_up = lazy["model.layers.1.mlp.experts.gate_up_proj"]  # (repeat read: counted below)
        for e in range(TINY["num_experts"]):
            assert torch.equal(
                gate_up[e, :I], raw[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"]
            )  # gate rows first
            assert torch.equal(gate_up[e, I:], raw[f"model.layers.1.mlp.experts.{e}.up_proj.weight"])
        q = "model.layers.0.self_attn.q_proj.weight"
        assert not torch.equal(ref[q], raw[q]) and torch.equal(
            ref[q], reverse_permute(raw[q], TINY["n_q_heads"], *raw[q].shape)
        )
        assert lazy[FP32_STRAGGLER].dtype == BF16 and raw[FP32_STRAGGLER].dtype == F32
        bias = lazy[f"model.layers.1.mlp.gate.{sl.ROUTER_BIAS_SUFFIX}"]
        assert bias.dtype == F32 and torch.equal(bias, raw[f"model.layers.1.mlp.gate.{sl.ROUTER_BIAS_SUFFIX}"])
        stats = lazy.stats
        assert stats["fused_builds"] == 2 * TINY["n_layers"] + 1  # + the repeated gate_up read above
        # every physical tensor once + the repeats: gate_up (2E per-expert tensors), the straggler and the bias
        assert stats["preads"] == len(raw) + 2 * TINY["num_experts"] + 2
        assert (
            stats["bytes"]
            == sum(t.numel() * t.element_size() for t in raw.values())
            + sum(
                raw[k].numel() * raw[k].element_size()
                for k in raw
                if k.startswith("model.layers.1.mlp.experts.") and not k.endswith("down_proj.weight")
            )
            + raw[FP32_STRAGGLER].numel() * 4
            + raw[f"model.layers.1.mlp.gate.{sl.ROUTER_BIAS_SUFFIX}"].numel() * 4
        )
        assert stats["repeat_reads"] == 3 and stats["layers_touched"] == 2

    def test_views_are_lazy_prefix_views(self, lazy, tiny_checkpoint):
        ref = tiny_checkpoint.ref
        layer1 = substate(lazy, "model.layers.1")
        assert isinstance(layer1, LazyStateDict) and layer1.prefix == "model.layers.1."
        assert list(layer1) == list(LAYER_KEY_ORDER) and len(layer1) == 13
        attn = substate(layer1, "self_attn")
        assert set(attn) == {f"{p}.weight" for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
        assert has_substate(layer1, "mlp") and has_substate(attn, "q_proj") and not has_substate(layer1, "vision")
        q_view = substate(attn, "q_proj")
        assert "weight" in q_view and "bias" not in q_view and list(q_view) == ["weight"]
        assert torch.equal(q_view["weight"], ref["model.layers.1.self_attn.q_proj.weight"])
        empty = substate(lazy, "model.layers.9")
        assert isinstance(empty, LazyStateDict) and len(empty) == 0 and not empty
        assert len(indexed_substates(lazy, "model.layers")) == TINY["n_layers"]
        assert attn.stats is lazy.stats and attn.snapshot_dir == lazy.snapshot_dir
        # Mapping mixins work on small views (they materialise: documented as "never on big views")
        norm = substate(lazy, "model.norm")
        assert set(norm.keys()) == {"weight"} and torch.equal(dict(norm)["weight"], ref["model.norm.weight"])
        assert (
            lazy.stats["bytes"] == 2 * TINY["hidden"] ** 2 + 2 * TINY["hidden"]
        )  # q_proj [32, 32] bf16 + norm [32] bf16
        # utils.substate on a plain dict is unchanged
        plain = substate(ref, "model.layers.1")
        assert isinstance(plain, dict) and set(plain) == set(LAYER_KEY_ORDER)

    def test_consumer_walk_matches_dict_path(self, lazy, tiny_checkpoint):
        """The tree's own consumers, fed the lazy views, produce the same torch tensors as with the reference dict."""
        ref = tiny_checkpoint.ref
        layer0, ref_layer0 = substate(lazy, "model.layers.0"), substate(ref, "model.layers.0")
        # experts -> prepare_expert_weights_torch (load_expert_weights' host half)
        expert_config = ExpertConfig(
            intermediate_size=TINY["moe_intermediate"],
            num_experts=TINY["num_experts"],
            hidden_size=TINY["hidden"],
            num_experts_per_tok=2,
        )
        got = prepare_expert_weights_torch(substate(layer0, "mlp.experts"), expert_config, tp=2)
        want = prepare_expert_weights_torch(substate(ref_layer0, "mlp.experts"), expert_config, tp=2)
        assert all(torch.equal(g, w) for g, w in zip(got, want))
        # attention -> load_attention_weights with ttnn mocked: compares the qkv_cat / o_proj host tensors
        attention_config = AttentionConfig(
            hidden_size=TINY["hidden"],
            num_heads=TINY["n_q_heads"],
            num_kv_heads=TINY["n_kv_heads"],
            head_dim=TINY["head_dim"],
            max_seq_len=64,
            max_local_batch_size=1,
        )
        mesh_config = MeshConfig((1, 2), decode=ModeConfig(tp=2))

        def run_attention(state):
            calls = []
            with patch("models.demos.solar_open.tt.attention.weights.ttnn") as mock_ttnn, patch.object(
                mesh_config, "column_parallel", return_value=object()
            ), patch.object(mesh_config, "row_parallel", return_value=object()):
                mock_ttnn.as_tensor.side_effect = lambda tensor, **kw: calls.append(tensor) or MagicMock()
                load_attention_weights(MagicMock(), attention_config, state, mesh_config, tensor_cache_path="p")
            return calls

        got_attn, want_attn = run_attention(substate(layer0, "self_attn")), run_attention(
            substate(ref_layer0, "self_attn")
        )
        assert len(got_attn) == 2 and all(torch.equal(g, w) for g, w in zip(got_attn, want_attn))
        # router (TopKRouter's two lines), norms (RMSNorm), shared expert (SharedExpert)
        gate, ref_gate = substate(layer0, "mlp.gate"), substate(ref_layer0, "mlp.gate")
        bias = gate[sl.ROUTER_BIAS_SUFFIX].reshape(1, -1).float()
        assert bias.dtype == F32 and torch.equal(bias, ref_gate[sl.ROUTER_BIAS_SUFFIX].reshape(1, -1).float())
        assert torch.equal(gate["weight"].transpose(0, 1), ref_gate["weight"].transpose(0, 1))
        for norm in ("input_layernorm", "post_attention_layernorm"):
            assert torch.equal(
                substate(layer0, norm)["weight"].reshape(1, 1, -1, 32),
                substate(ref_layer0, norm)["weight"].reshape(1, 1, -1, 32),
            )
        shared, ref_shared = substate(layer0, "mlp.shared_experts"), substate(ref_layer0, "mlp.shared_experts")
        for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            assert torch.equal(shared[k].transpose(0, 1), ref_shared[k].transpose(0, 1))
        assert lazy.stats["repeat_reads"] == 0 and lazy.stats["fused_builds"] == 2 and lazy.stats["layers_touched"] == 1

    def test_single_threaded_reads_and_config_json_fallback(self, tiny_checkpoint):
        loader = LazyStateDict(tiny_checkpoint.root, head_dim=None, num_experts=None, io_threads=1)
        try:
            assert loader._store.head_dim == TINY["head_dim"] and loader._store.num_experts == TINY["num_experts"]
            for key in (
                "model.layers.1.mlp.experts.gate_up_proj",
                "model.layers.1.mlp.experts.down_proj",
                "model.layers.1.self_attn.k_proj.weight",
            ):
                assert torch.equal(loader[key], tiny_checkpoint.ref[key])
        finally:
            loader.close()
        # convert_to_meta=False hands out the HF q/k order
        loader = LazyStateDict(tiny_checkpoint.root, head_dim=16, num_experts=4, convert_to_meta=False)
        try:
            k = "model.layers.0.self_attn.k_proj.weight"
            assert torch.equal(loader[k], tiny_checkpoint.raw[k]) and not torch.equal(loader[k], tiny_checkpoint.ref[k])
        finally:
            loader.close()

    def test_fd_window_prefetch_and_close(self, tiny_checkpoint, monkeypatch):
        advised = []
        monkeypatch.setattr(sl, "_fadvise_ranges", lambda paths, ranges, advice: advised.append((dict(ranges), advice)))
        monkeypatch.setenv("SOLAR_OPEN_STREAMING_DONTNEED", "1")
        loader = LazyStateDict(tiny_checkpoint.root, head_dim=16, num_experts=4)
        fds = loader._store._fds
        loader["model.embed_tokens.weight"]
        assert set(fds) == {SHARDS[0]}
        _walk_layer(substate(loader, "model.layers.0"))
        assert set(fds) == {SHARDS[0]}  # layer 0 lives in shard 1 only
        _walk_layer(substate(loader, "model.layers.1"))
        assert set(fds) == {SHARDS[1], SHARDS[2]} and loader.open_fds == 2  # shard 1 closed, <= 2 fds
        loader["model.norm.weight"]
        assert set(fds) == {SHARDS[2]}
        loader["lm_head.weight"]
        loader.close()  # waits for the advise thread; idempotent
        loader.close()
        assert loader.open_fds == 0 and loader._store._pool is None
        # prefetch order: embed -> layer 0 -> layer 1 -> tail (WILLNEED), each previous group DONTNEED'ed once
        willneed = [r for r, a in advised if a == os.POSIX_FADV_WILLNEED]
        dontneed = [r for r, a in advised if a == os.POSIX_FADV_DONTNEED]
        assert willneed == [
            loader.layer_byte_ranges(0),
            loader.layer_byte_ranges(1),
            loader._store.group_byte_ranges(("tail",)),
        ]
        assert dontneed == [
            loader._store.group_byte_ranges(("embed",)),
            loader.layer_byte_ranges(0),
            loader.layer_byte_ranges(1),
        ]
        assert set(loader.layer_byte_ranges(1)) == {SHARDS[1], SHARDS[2]}
        for shard, (start, stop) in loader.layer_byte_ranges(1).items():
            assert 8 < start < stop <= (tiny_checkpoint.root / shard).stat().st_size
        # a later access reopens what it needs
        assert (
            torch.equal(loader["model.norm.weight"], tiny_checkpoint.ref["model.norm.weight"]) and loader.open_fds == 1
        )
        loader.close()

    def test_construction_and_access_errors(self, tiny_checkpoint, tmp_path, monkeypatch, expect_error):
        root = tiny_checkpoint.root
        with expect_error(ValueError, "per-expert"):
            LazyStateDict(root, head_dim=16, num_experts=5)  # 4 experts on disk
        with expect_error(ValueError, "per-expert"):
            LazyStateDict(root, head_dim=16, num_experts=3)
        empty = tmp_path / "empty"
        empty.mkdir()
        with expect_error(FileNotFoundError, sl.INDEX_FILE):
            LazyStateDict(empty, head_dim=16, num_experts=4)
        # HF repo id: resolved through transformers' hub cache lookup (monkeypatched here); a miss is a FileNotFoundError
        import transformers.utils.hub as hub

        monkeypatch.setattr(hub, "cached_file", lambda repo, filename: str(root / filename))
        loader = LazyStateDict("upstage/Tiny-Solar", head_dim=16, num_experts=4)
        assert loader.snapshot_dir == root and len(loader) == 29
        loader.close()

        def miss(repo, filename):
            raise OSError("offline")

        monkeypatch.setattr(hub, "cached_file", miss)
        with expect_error(FileNotFoundError, "offline"):
            LazyStateDict("upstage/Missing", head_dim=16, num_experts=4)
        # a missing shard fails at first access of a tensor in it, naming the shard - not at construction
        partial = tmp_path / "partial"
        shutil.copytree(root, partial)
        (partial / SHARDS[2]).unlink()
        loader = LazyStateDict(partial, head_dim=16, num_experts=4)
        assert torch.equal(loader["model.embed_tokens.weight"], tiny_checkpoint.ref["model.embed_tokens.weight"])
        with expect_error(FileNotFoundError, SHARDS[2]):
            loader["model.norm.weight"]
        assert loader.meta("model.layers.1.mlp.experts.gate_up_proj") == ((4, 32, 32), BF16)  # expert 0 is in shard 2
        with expect_error(FileNotFoundError, SHARDS[2]):
            loader["model.layers.1.mlp.experts.gate_up_proj"]  # experts 2-3 are in the missing shard 3
        loader.close()

    def test_no_accidental_materialisation(self, lazy, expect_error):
        assert not (lazy == {}) and lazy == lazy and lazy != substate(lazy, "model.norm")
        assert hash(lazy) == id(lazy)
        with expect_error(TypeError, "pickled"):
            pickle.dumps(lazy)
        with expect_error(TypeError, "pickled"):
            copy.deepcopy(lazy)
        assert lazy.stats["bytes"] == 0

    def test_load_state_dict_streaming_branch_returns_the_loader(self, tiny_checkpoint, monkeypatch, expect_error):
        monkeypatch.setenv("SOLAR_OPEN_STREAMING_LOAD", "1")
        hf_config = SimpleNamespace(head_dim=16, hidden_size=32, num_attention_heads=2, num_local_experts=4)
        monkeypatch.setattr(mc, "AutoConfig", SimpleNamespace(from_pretrained=lambda path, **kw: hf_config))

        def whole_model_load(*a, **k):
            raise AssertionError("the whole-model from_pretrained path must not run with SOLAR_OPEN_STREAMING_LOAD=1")

        monkeypatch.setattr(mc, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=whole_model_load))
        out = ModelArgs.load_state_dict(str(tiny_checkpoint.root))
        assert isinstance(out, LazyStateDict) and len(out) == 29 and bool(out)
        assert out.stats["bytes"] == 0, "load_state_dict must only validate metadata (no safety-net dict rebuild)"
        q = "model.layers.0.self_attn.q_proj.weight"
        assert torch.equal(out[q], tiny_checkpoint.ref[q])
        out.close()
        out_hf = ModelArgs.load_state_dict(str(tiny_checkpoint.root), convert_to_meta_format=False)
        assert torch.equal(out_hf[q], tiny_checkpoint.raw[q])
        out_hf.close()
        assert ModelArgs.load_state_dict(str(tiny_checkpoint.root), dummy_weights=True) == {}
        # the whole-model default is untouched when the variable is unset
        monkeypatch.delenv("SOLAR_OPEN_STREAMING_LOAD")
        with expect_error(AssertionError, "whole-model"):
            ModelArgs.load_state_dict(str(tiny_checkpoint.root))

    def test_validate_layout_rejects_bad_lazy_layouts(self, tiny_checkpoint, tmp_path, expect_error):
        """_validate_state_dict_layout uses meta() on a lazy dict: a wrong on-disk expert shape or a bf16 bias is caught
        from the headers, before any tensor is read."""
        raw = dict(tiny_checkpoint.raw)
        raw["model.layers.0.mlp.gate.e_score_correction_bias"] = raw[
            "model.layers.0.mlp.gate.e_score_correction_bias"
        ].to(BF16)
        root = _write_checkpoint(tmp_path / "bf16_bias", raw)
        loader = LazyStateDict(root, head_dim=16, num_experts=4)
        with expect_error(ValueError, "fp32"):
            mc._validate_state_dict_layout(loader)
        assert loader.stats["bytes"] == 0
        loader.close()
        raw = dict(tiny_checkpoint.raw)
        for e in range(4):  # down stored [I, H] instead of [H, I]
            raw[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = raw[
                f"model.layers.0.mlp.experts.{e}.down_proj.weight"
            ].T.contiguous()
        root = _write_checkpoint(tmp_path / "bad_down", raw)
        loader = LazyStateDict(root, head_dim=16, num_experts=4)
        with expect_error(ValueError, r"\[E, 2I, H\]"):
            mc._validate_state_dict_layout(loader)
        loader.close()


# ------------------------------------------------------------------------------------------------------------------
# Part B: the real checkpoint, bit-exact against the phase-1 semantics without a whole-model load
# ------------------------------------------------------------------------------------------------------------------
REAL_HEAD_DIM, REAL_EXPERTS, REAL_LAYERS = 128, 128, 48
REAL_LAYER = "model.layers.0."
REAL_EXPECTED = {
    "model.embed_tokens.weight": ((196608, 4096), BF16),
    REAL_LAYER + "input_layernorm.weight": ((4096,), BF16),
    REAL_LAYER + "self_attn.q_proj.weight": ((8192, 4096), BF16),
    REAL_LAYER + "self_attn.k_proj.weight": ((1024, 4096), BF16),
    REAL_LAYER + "self_attn.v_proj.weight": ((1024, 4096), BF16),
    REAL_LAYER + "self_attn.o_proj.weight": ((4096, 8192), BF16),
    REAL_LAYER + "post_attention_layernorm.weight": ((4096,), BF16),
    REAL_LAYER + "mlp.gate.weight": ((128, 4096), BF16),
    REAL_LAYER + f"mlp.gate.{sl.ROUTER_BIAS_SUFFIX}": ((128,), F32),
    REAL_LAYER + "mlp.experts.gate_up_proj": ((128, 2560, 4096), BF16),
    REAL_LAYER + "mlp.experts.down_proj": ((128, 4096, 1280), BF16),
    REAL_LAYER + "mlp.shared_experts.gate_proj.weight": ((1280, 4096), BF16),
    REAL_LAYER + "mlp.shared_experts.up_proj.weight": ((1280, 4096), BF16),
    REAL_LAYER + "mlp.shared_experts.down_proj.weight": ((4096, 1280), BF16),
    "model.norm.weight": ((4096,), BF16),
    "lm_head.weight": ((196608, 4096), BF16),
}
# host_load_check facts recorded in phase 1 (stage 8) for upstage/Solar-Open-100B
REAL_BIAS_HEAD = [1.8596649169921875e-05, 1.8596649169921875e-05, 1.8715858459472656e-05, 1.8477439880371094e-05]
REAL_BIAS_MIN, REAL_BIAS_MAX, REAL_BIAS_DISTINCT = -0.001984, 0.002014, 10
RSS_LIMIT_GB = 25

REFERENCE_SCRIPT = r"""
import hashlib, json, resource, sys, torch, transformers
transformers.logging.set_verbosity_error()
from transformers import AutoModelForCausalLM
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format
snapshot, out_path, head_dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, num_hidden_layers=1)
sd = model.state_dict()
del model
sd = convert_hf_qkv_to_meta_format(sd, head_dim)
tensors = {}
for key in sorted(sd):
    t = sd[key]
    h = hashlib.sha256()
    flat = t.contiguous().view(torch.uint8).reshape(-1).numpy()
    for i in range(0, flat.shape[0], 64 << 20):
        h.update(flat[i : i + (64 << 20)].data)
    tensors[key] = {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": h.hexdigest()}
    sd[key] = None
json.dump(
    {
        "source": "AutoModelForCausalLM.from_pretrained(dtype=bf16, num_hidden_layers=1) + convert_hf_qkv_to_meta_format",
        "transformers": transformers.__version__,
        "snapshot": snapshot,
        "tensors": tensors,
    },
    open(out_path, "w"),
    indent=1,
)
print(json.dumps({"peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, "keys": len(tensors)}))
"""


def _sha256(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    flat = t.contiguous().view(torch.uint8).reshape(-1).numpy()
    step = 64 << 20
    for i in range(0, flat.shape[0], step):
        h.update(flat[i : i + step].data)
    return h.hexdigest()


def _real_snapshot():
    """The real snapshot when HF_MODEL names it and the shards of the embedding, layer 0, norm and lm_head exist."""
    env = os.getenv("HF_MODEL")
    if not env:
        return None
    path = Path(env)
    if path.name != mc.MODEL_NAME or not path.is_dir() or not (path / sl.INDEX_FILE).is_file():
        return None
    weight_map = json.loads((path / sl.INDEX_FILE).read_text())["weight_map"]
    if len({k.split(".")[2] for k in weight_map if k.startswith("model.layers.")}) != REAL_LAYERS:
        return None
    needed = {shard for key, shard in weight_map.items() if sl._group_of(key) in {("embed",), ("layer", 0), ("tail",)}}
    return path if all((path / shard).is_file() for shard in needed) else None


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


@pytest.mark.skipif(
    _real_snapshot() is None, reason="HF_MODEL is not the Solar-Open-100B snapshot with the layer-0 shards"
)
class TestRealCheckpoint:
    def test_layout_validation_is_metadata_only(self):
        snapshot = _real_snapshot()
        loader = LazyStateDict(snapshot, head_dim=REAL_HEAD_DIM, num_experts=REAL_EXPERTS)
        try:
            assert len(loader) == 3 + REAL_LAYERS * 13 == 627 and loader.num_shards == 42
            t0 = time.perf_counter()
            for key, expected in REAL_EXPECTED.items():
                assert loader.meta(key) == expected, key
            mc._validate_state_dict_layout(loader)
            assert loader.stats["bytes"] == 0 and loader.stats["preads"] == 0
            assert time.perf_counter() - t0 < 5.0
            ranges = loader.layer_byte_ranges(0)
            assert 1 <= len(ranges) <= 2 and 4.0e9 < sum(stop - start for start, stop in ranges.values()) < 4.6e9
        finally:
            loader.close()

    def test_layer0_bit_exact_vs_phase1_reference(self, tmp_path):
        snapshot = _real_snapshot()
        loader = LazyStateDict(snapshot, head_dim=REAL_HEAD_DIM, num_experts=REAL_EXPERTS)
        weight_map = loader._store.weight_map
        hashes, timings = {}, {}
        try:
            for key, (shape, dtype) in REAL_EXPECTED.items():
                t0 = time.perf_counter()
                t = loader[key]
                timings[key] = time.perf_counter() - t0
                assert tuple(t.shape) == shape and t.dtype == dtype, key
                # direct facts from the raw shards (independent of transformers)
                if key.endswith("gate_up_proj"):
                    for e in (0, 127):
                        with safe_open(
                            str(snapshot / weight_map[f"{REAL_LAYER}mlp.experts.{e}.gate_proj.weight"]),
                            "pt",
                            device="cpu",
                        ) as f:
                            gate = f.get_tensor(f"{REAL_LAYER}mlp.experts.{e}.gate_proj.weight")
                            up = f.get_tensor(f"{REAL_LAYER}mlp.experts.{e}.up_proj.weight")
                        assert torch.equal(t[e, :1280], gate) and torch.equal(
                            t[e, 1280:], up
                        ), f"expert {e}: gate rows first"
                        assert not torch.equal(gate, up)
                        del gate, up
                elif key.endswith("down_proj") and "experts" in key:
                    with safe_open(
                        str(snapshot / weight_map[f"{REAL_LAYER}mlp.experts.64.down_proj.weight"]), "pt", device="cpu"
                    ) as f:
                        assert torch.equal(t[64], f.get_tensor(f"{REAL_LAYER}mlp.experts.64.down_proj.weight"))
                elif key.endswith(sl.ROUTER_BIAS_SUFFIX):
                    assert t[:4].tolist() == REAL_BIAS_HEAD
                    assert round(t.min().item(), 6) == REAL_BIAS_MIN and round(t.max().item(), 6) == REAL_BIAS_MAX
                    assert t.unique().numel() == REAL_BIAS_DISTINCT
                elif key.endswith(("q_proj.weight", "k_proj.weight")):
                    with safe_open(str(snapshot / weight_map[key]), "pt", device="cpu") as f:
                        raw = f.get_tensor(key)
                    assert torch.equal(permute(t, raw.shape[0] // REAL_HEAD_DIM, *raw.shape), raw) and not torch.equal(
                        t, raw
                    )
                    del raw
                elif key in ("model.norm.weight", REAL_LAYER + "self_attn.v_proj.weight"):
                    with safe_open(str(snapshot / weight_map[key]), "pt", device="cpu") as f:
                        assert torch.equal(t, f.get_tensor(key))
                hashes[key] = {"shape": list(t.shape), "dtype": str(t.dtype), "sha256": _sha256(t)}
                del t
            stats = dict(loader.stats)
        finally:
            loader.close()
        peak_gb = _rss_gb()
        print(
            f"\nlazy loader: {stats['bytes'] / 1e9:.2f} GB in {stats['preads']} preads, {stats['fused_builds']} fused builds, "
            f"{stats['seconds']:.1f} s reading (gate_up {timings[REAL_LAYER + 'mlp.experts.gate_up_proj']:.2f} s, "
            f"down {timings[REAL_LAYER + 'mlp.experts.down_proj']:.2f} s); peak RSS {peak_gb:.1f} GB"
        )
        assert stats["repeat_reads"] == 0 and stats["fused_builds"] == 2 and stats["layers_touched"] == 1
        assert (
            stats["preads"] == 14 + 3 * REAL_EXPERTS - 2 + 2
        )  # 12 physical layer/embed/norm/lm_head keys + 384 expert reads
        assert peak_gb < RSS_LIMIT_GB, f"lazy path peaked at {peak_gb:.1f} GB RSS"

        # phase-1 semantics on a 1-layer config in a subprocess (from_pretrained loads only the matching tensors)
        cache_root = Path(os.getenv("TT_CACHE_PATH") or tmp_path)
        reference_path = cache_root / "streaming_loader_reference_layer0.json"
        reference = None
        if reference_path.is_file() and os.getenv("SOLAR_OPEN_REGEN_LOADER_REFERENCE") != "1":
            reference = json.loads(reference_path.read_text())
            if reference.get("snapshot") != str(snapshot) or set(reference.get("tensors", {})) != set(REAL_EXPECTED):
                reference = None
        if reference is None:
            t0 = time.perf_counter()
            result = subprocess.run(
                [sys.executable, "-c", REFERENCE_SCRIPT, str(snapshot), str(reference_path), str(REAL_HEAD_DIM)],
                text=True,
                capture_output=True,
                env=os.environ,
                timeout=1800,
            )
            assert result.returncode == 0, f"reference subprocess failed:\n{result.stderr[-4000:]}"
            child = json.loads(result.stdout.strip().splitlines()[-1])
            print(
                f"reference subprocess: {time.perf_counter() - t0:.1f} s, peak RSS {child['peak_rss_gb']:.1f} GB -> {reference_path}"
            )
            assert child["peak_rss_gb"] < RSS_LIMIT_GB and child["keys"] == len(REAL_EXPECTED)
            reference = json.loads(reference_path.read_text())
        else:
            print(
                f"reference hashes from {reference_path} ({reference['source']}, transformers {reference['transformers']})"
            )
        assert set(reference["tensors"]) == set(hashes)
        mismatch = [k for k in hashes if hashes[k] != reference["tensors"][k]]
        assert not mismatch, f"lazy loader differs from the phase-1 path for {mismatch}"
