# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the Solar-Open config / loader / weight-cache plumbing. No device is opened.

    pytest models/demos/solar_open/tests/unit/test_model_config.py

HF_MODEL may point at the real Solar-Open-100B snapshot (a directory/symlink named Solar-Open-100B) or be unset,
in which case the in-tree configs/Solar-Open-100B directory is used; the checks that need generation_config.json
or the tokenizer files skip when they are not available. The mesh device is a mock and determine_device_name is
patched to "P150x8".
"""

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.solar_open.config import MoEOptions
from models.demos.solar_open.tt import model_config as mc
from models.demos.solar_open.tt.common import check_kv_budget, kv_budget_gib, paged_kv_cache_gib, unpaged_kv_cache_gib
from models.demos.solar_open.tt.model_config import ModelArgs
from models.demos.solar_open.utils.general_utils import get_layer_types, get_sliding_window, resolve_rope_theta
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.tt_transformers.tt.load_checkpoints import reverse_permute

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "Solar-Open-100B"
MODEL_NAME = "Solar-Open-100B"

SOLAR_ENV_VARS = (
    "SOLAR_OPEN_EXPERT_DTYPE",
    "SOLAR_OPEN_SHARED_EXPERT_DTYPE",
    "SOLAR_OPEN_ROUTER_IMPL",
    "SOLAR_OPEN_ROUTER_FP32_LOGITS",
    "SOLAR_OPEN_REASONING_EFFORT",
    "SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT",
    "SOLAR_OPEN_FORCE_MODEL_LOAD",
    "SOLAR_OPEN_KV_BUDGET_GIB",
    "SOLAR_OPEN_STREAMING_LOAD",
)
DEFAULT_MARKER_MOE = {
    "expert_dtype": "bfp8",
    "shared_expert_dtype": "bfp8",
    "router_impl": "fused",
    "router_fp32_logits": True,
}


def _snapshot_dir() -> Path:
    """The real snapshot when HF_MODEL points at a config-bearing directory named Solar-Open-100B, else the in-tree config."""
    env = os.getenv("HF_MODEL")
    if env and Path(env).name == MODEL_NAME and (Path(env) / "config.json").is_file():
        return Path(env)
    return CONFIG_DIR


class _StubTokenizer:
    """Records apply_chat_template calls; used when the tokenizer files are not downloaded yet."""

    eos_token_id = 2

    def __init__(self, returns=None):
        self.calls = []
        self.returns = returns if returns is not None else [1, 20, 23]

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.returns


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Clean SOLAR_OPEN_* env, a throw-away TT_CACHE_PATH and a fixed device name (no device is opened)."""
    for var in SOLAR_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TT_CACHE_PATH", str(tmp_path / "tt_cache"))
    monkeypatch.setattr(mc, "determine_device_name", lambda mesh_device: "P150x8")


@pytest.fixture
def snapshot_dir():
    return _snapshot_dir()


@pytest.fixture
def mesh_1x8():
    mesh = MagicMock(name="mesh_device_1x8")
    mesh.shape = (1, 8)
    return mesh


@pytest.fixture
def stub_tokenizer(monkeypatch):
    stub = _StubTokenizer()
    monkeypatch.setattr(mc, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: stub))
    return stub


@pytest.fixture
def real_model_args(monkeypatch, snapshot_dir, mesh_1x8):
    """ModelArgs against the real SolarOpenConfig; the tokenizer is stubbed while its files are not downloaded."""
    monkeypatch.setenv("HF_MODEL", str(snapshot_dir))
    has_tokenizer = any((snapshot_dir / f).is_file() for f in ("tokenizer.json", "tokenizer_config.json"))
    if not has_tokenizer:
        monkeypatch.setattr(mc, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: _StubTokenizer()))
    return ModelArgs(mesh_device=mesh_1x8, max_batch_size=32, max_seq_len=8192)


class TestMoEOptions:
    def test_defaults(self):
        opts = MoEOptions()
        assert opts.expert_dtype == ttnn.bfloat8_b
        assert opts.shared_expert_dtype == ttnn.bfloat8_b
        assert opts.router_impl == "fused"
        assert opts.router_fp32_logits is True
        assert opts.expert_dtype_str == "bfp8"
        assert opts.marker_fields() == DEFAULT_MARKER_MOE
        assert MoEOptions.from_env() == opts

    def test_env_roundtrip(self, monkeypatch):
        monkeypatch.setenv("SOLAR_OPEN_EXPERT_DTYPE", "bfp4")
        monkeypatch.setenv("SOLAR_OPEN_SHARED_EXPERT_DTYPE", "bf16")
        monkeypatch.setenv("SOLAR_OPEN_ROUTER_IMPL", "ops")
        monkeypatch.setenv("SOLAR_OPEN_ROUTER_FP32_LOGITS", "0")
        opts = MoEOptions.from_env()
        assert opts.expert_dtype == ttnn.bfloat4_b
        assert opts.shared_expert_dtype == ttnn.bfloat16
        assert opts.router_impl == "ops"
        assert opts.router_fp32_logits is False
        assert opts.expert_dtype_str == "bfp4"
        fields = opts.marker_fields()
        assert fields == {
            "expert_dtype": "bfp4",
            "shared_expert_dtype": "bf16",
            "router_impl": "ops",
            "router_fp32_logits": False,
        }
        assert json.loads(json.dumps(fields)) == fields  # survives the .weights_complete JSON round trip

    @pytest.mark.parametrize(
        "var, value",
        [
            ("SOLAR_OPEN_EXPERT_DTYPE", "bf16"),  # 24 GiB of bf16 experts per device: not a supported bring-up option
            ("SOLAR_OPEN_EXPERT_DTYPE", "fp8"),
            ("SOLAR_OPEN_SHARED_EXPERT_DTYPE", "int8"),
            ("SOLAR_OPEN_ROUTER_IMPL", "torch"),
        ],
    )
    def test_invalid_env_rejected(self, monkeypatch, expect_error, var, value):
        monkeypatch.setenv(var, value)
        with expect_error(ValueError, var):
            MoEOptions.from_env()

    def test_invalid_direct_construction_rejected(self, expect_error):
        with expect_error(ValueError, "router_impl"):
            MoEOptions(router_impl="fast")
        with expect_error(ValueError, "expert_dtype"):
            MoEOptions(expert_dtype=ttnn.bfloat16)

    def test_frozen_and_hashable(self, expect_error):
        opts = MoEOptions()
        with expect_error(dataclasses.FrozenInstanceError, "cannot assign to field"):
            opts.router_impl = "ops"
        assert {opts: 1}[MoEOptions()] == 1


class TestHFConfigHelpers:
    def test_solar_config(self, snapshot_dir):
        cfg = AutoConfig.from_pretrained(str(snapshot_dir), trust_remote_code=True)
        assert type(cfg).__name__ == "SolarOpenConfig"
        # transformers 5.x: no top-level rope_theta, it lives in rope_parameters -> the naive getattr default is a trap
        assert getattr(cfg, "rope_theta", None) is None
        theta = resolve_rope_theta(cfg)
        assert isinstance(theta, float) and theta == 1_000_000.0
        assert cfg.rope_parameters["rope_type"] == "yarn" and cfg.rope_parameters["factor"] == 2.0
        assert cfg.rope_parameters["original_max_position_embeddings"] == 65536
        assert get_layer_types(cfg) == ["full_attention"] * 48
        assert all(get_sliding_window(cfg, i) is None for i in range(cfg.num_hidden_layers))
        # Numbers the TT modules depend on (Section 0 of the design)
        assert cfg.head_dim == 128 and cfg.hidden_size // cfg.num_attention_heads == 64  # head_dim is NOT hidden/heads
        assert cfg.moe_intermediate_size == 1280 and cfg.intermediate_size == 10240  # the latter is unused by the model
        assert cfg.num_local_experts == 128 and cfg.num_experts_per_tok == 8 and cfg.n_shared_experts == 1
        assert cfg.n_group == 1 and cfg.topk_group == 1 and cfg.norm_topk_prob and cfg.routed_scaling_factor == 1.0
        assert cfg.hidden_act == "silu" and not cfg.attention_bias and cfg.rms_norm_eps == 1e-5

    def test_fallbacks(self):
        assert resolve_rope_theta(SimpleNamespace(rope_theta=150000.0)) == 150000.0
        assert resolve_rope_theta(SimpleNamespace(rope_parameters={"rope_theta": 500000})) == 500000.0
        assert resolve_rope_theta(SimpleNamespace(rope_parameters=None, default_theta=7.0)) == 7.0
        assert resolve_rope_theta(SimpleNamespace()) == 1_000_000.0
        assert resolve_rope_theta(SimpleNamespace(), default=42.0) == 42.0
        legacy = SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"], sliding_window=128, num_hidden_layers=2
        )
        assert get_layer_types(legacy) == ["sliding_attention", "full_attention"]
        assert get_sliding_window(legacy, 0) == 128
        assert get_sliding_window(legacy, 1) is None
        assert get_layer_types(SimpleNamespace(num_hidden_layers=3)) == ["full_attention"] * 3


class TestModelArgs:
    def test_dummy_weights_and_cache_dir(self, monkeypatch, mesh_1x8, tmp_path):
        monkeypatch.setenv("HF_MODEL", str(CONFIG_DIR))
        args = ModelArgs(mesh_device=mesh_1x8, dummy_weights=True)
        assert args.model_name == MODEL_NAME and args.base_model_name == MODEL_NAME
        assert args.tokenizer is None and args.processor is None and args.stop_token_ids == set()
        assert args.hf_config is None and args.n_layers is None
        assert args.moe_options == MoEOptions()
        assert args.max_local_batch_size == 1 and args.disable_batched_prefill
        cache = args.weight_cache_path(ttnn.bfloat8_b)
        assert cache == tmp_path / "tt_cache" / "tensor_cache_bfp8_expbfp8_(1, 8)"
        assert cache.is_dir()
        assert args.weight_cache_path(ttnn.bfloat16).name == "tensor_cache_bf16_expbfp8_(1, 8)"
        assert args.weight_cache_path(ttnn.bfloat4_b).name == "tensor_cache_bfp4_expbfp8_(1, 8)"

    def test_expert_dtype_folded_into_cache_dir(self, monkeypatch, mesh_1x8):
        monkeypatch.setenv("HF_MODEL", str(CONFIG_DIR))
        monkeypatch.setenv("SOLAR_OPEN_EXPERT_DTYPE", "bfp4")
        args = ModelArgs(mesh_device=mesh_1x8, dummy_weights=True)
        assert args.moe_options.expert_dtype == ttnn.bfloat4_b
        assert args.weight_cache_path(ttnn.bfloat8_b).name == "tensor_cache_bfp8_expbfp4_(1, 8)"

    def test_cache_root_fallbacks(self, monkeypatch, mesh_1x8):
        monkeypatch.delenv("TT_CACHE_PATH")
        monkeypatch.setenv("HF_MODEL", str(CONFIG_DIR))
        args = ModelArgs(mesh_device=mesh_1x8, dummy_weights=True)
        assert args.weight_cache_root() == CONFIG_DIR  # local checkpoint directory
        monkeypatch.setenv("HF_MODEL", "upstage/Solar-Open-100B")  # HF repo id: never a relative "upstage/" in CWD
        args = ModelArgs(mesh_device=mesh_1x8, dummy_weights=True)
        assert args.weight_cache_root() == Path.home() / ".cache" / "tenstorrent" / MODEL_NAME
        monkeypatch.setenv("TT_CACHE_PATH", "/somewhere/fast")
        assert args.weight_cache_root() == Path("/somewhere/fast")

    def test_default_hf_model_is_repo_id(self, monkeypatch, mesh_1x8):
        monkeypatch.delenv("HF_MODEL", raising=False)
        args = ModelArgs(mesh_device=mesh_1x8, dummy_weights=True)
        assert args.model_path == "upstage/Solar-Open-100B" and args.model_name == MODEL_NAME

    def test_rejects_other_model_names(self, monkeypatch, mesh_1x8, tmp_path, expect_error):
        other = tmp_path / "some-other-120b"
        other.mkdir()
        monkeypatch.setenv("HF_MODEL", str(other))
        with expect_error(AssertionError, "Solar-Open-100B"):
            ModelArgs(mesh_device=mesh_1x8, dummy_weights=True)

    def test_single_row_batch_cap(self, monkeypatch, mesh_1x8, expect_error):
        monkeypatch.setenv("HF_MODEL", str(CONFIG_DIR))
        with expect_error(ValueError, "32 users"):
            ModelArgs(mesh_device=mesh_1x8, dummy_weights=True, max_batch_size=64)

    def test_real_config(self, real_model_args):
        args = real_model_args
        assert type(args.hf_config).__name__ == "SolarOpenConfig"
        assert args.vocab_size == 196608 and args.n_layers == 48 and args.head_dim == 128
        assert args.rope_theta == 1_000_000.0
        assert args.rope_scaling["factor"] == 2.0
        assert args.rope_scaling["original_max_position_embeddings"] == 65536
        assert args.rope_scaling["rope_theta"] == 1_000_000
        assert (
            args.max_local_batch_size == 32 and args.max_context_len == 8192 and args.max_prefill_chunk_size == 131072
        )
        assert args.processor is None and args.tokenizer is not None

    def test_stop_token_ids(self, real_model_args, snapshot_dir):
        if not (snapshot_dir / "generation_config.json").is_file():
            assert real_model_args.stop_token_ids == {2}  # fallback: tokenizer eos only
            pytest.skip(f"{snapshot_dir} has no generation_config.json; the full stop set needs the HF snapshot")
        assert real_model_args.stop_token_ids == {
            2,
            24,
            25,
        }  # <|endoftext|>, <|flush|>, <|calls|>; <|end|>=21 is NOT a stop

    def test_trace_and_warmup_seq_lens(self, real_model_args, monkeypatch, mesh_1x8, snapshot_dir):
        args = real_model_args
        assert args.trace_prefill_supported_seq_lens == [128]
        assert args.can_enable_trace(128)
        assert not args.can_enable_trace(256)
        assert not args.can_enable_trace(128, num_cached_tokens=32)
        assert args.get_warmup_prefill_supported_seq_lens() == [128, 1024, 2048]
        # Other SKUs are unvalidated -> no traced prefill (safe default)
        monkeypatch.setattr(mc, "determine_device_name", lambda mesh_device: "T3K")
        args_t3k = ModelArgs(mesh_device=mesh_1x8, dummy_weights=True)
        assert args_t3k.trace_prefill_supported_seq_lens == []
        assert not args_t3k.can_enable_trace(128)

    def test_encode_prompt_template_kwargs(self, monkeypatch, mesh_1x8, stub_tokenizer, expect_error):
        monkeypatch.setenv("HF_MODEL", str(CONFIG_DIR))
        args = ModelArgs(mesh_device=mesh_1x8)
        ids = args.encode_prompt("안녕하세요")
        messages, kw = stub_tokenizer.calls[-1]
        assert ids == [1, 20, 23]
        assert messages == [{"role": "user", "content": "안녕하세요"}]
        assert kw["add_generation_prompt"] is True and kw["tokenize"] is True
        assert kw["reasoning_effort"] == "high" and kw["default_system_prompt"] is True  # Solar template defaults
        monkeypatch.setenv("SOLAR_OPEN_REASONING_EFFORT", "low")
        monkeypatch.setenv("SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT", "0")
        args.encode_prompt("hi", system_prompt_text="be brief")
        messages, kw = stub_tokenizer.calls[-1]
        assert messages == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
        assert kw["reasoning_effort"] == "low" and kw["default_system_prompt"] is False
        chat = [{"role": "user", "content": "x"}]
        args.encode_prompt(chat, reasoning_effort="minimal")  # explicit kwarg wins over the env
        messages, kw = stub_tokenizer.calls[-1]
        assert messages is chat and kw["reasoning_effort"] == "minimal"
        with expect_error(AssertionError, "instruct=True"):
            args.encode_prompt("x", instruct=True)

    @pytest.mark.parametrize(
        "raw",
        [
            [1, 2, 3],
            SimpleNamespace(ids=[1, 2, 3]),  # tokenizers.Encoding-like
            [SimpleNamespace(ids=[1, 2]), SimpleNamespace(ids=[3])],  # list of Encodings
            {"input_ids": [1, 2, 3]},  # BatchEncoding / dict
        ],
    )
    def test_encode_prompt_normalises_token_containers(self, monkeypatch, mesh_1x8, raw):
        stub = _StubTokenizer(returns=raw)
        monkeypatch.setattr(mc, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: stub))
        monkeypatch.setenv("HF_MODEL", str(CONFIG_DIR))
        args = ModelArgs(mesh_device=mesh_1x8)
        assert args.encode_prompt("x") == [1, 2, 3]

    def test_encode_prompt_real_tokenizer(self, monkeypatch, mesh_1x8, snapshot_dir):
        if not any((snapshot_dir / f).is_file() for f in ("tokenizer.json", "tokenizer_config.json")):
            pytest.skip("tokenizer files not downloaded yet")
        monkeypatch.setenv("HF_MODEL", str(snapshot_dir))
        args = ModelArgs(mesh_device=mesh_1x8)
        ids = args.encode_prompt("What is the capital of Korea?")
        assert isinstance(ids, list) and ids and all(isinstance(t, int) for t in ids)
        text = args.tokenizer.decode(ids, skip_special_tokens=False)
        assert text.endswith("<|begin|>assistant")  # add_generation_prompt with reasoning_effort=high
        low_ids = args.encode_prompt("What is the capital of Korea?", reasoning_effort="low")
        low_text = args.tokenizer.decode(low_ids, skip_special_tokens=False)
        assert low_text.endswith("<|begin|>assistant<|think|><|end|><|begin|>assistant")


class TestWeightCacheMarker:
    @staticmethod
    def _write_marker(cache, meta):
        (cache / ModelArgs.WEIGHT_CACHE_MARKER).write_text(json.dumps(meta))

    def test_marker_roundtrip_and_rejections(self, real_model_args, monkeypatch):
        args = real_model_args
        dtype = ttnn.bfloat8_b
        assert not args.weight_cache_is_complete(dtype)  # no marker yet
        cache = args.weight_cache_path(dtype)
        args.mark_weight_cache_complete(dtype)
        assert not args.weight_cache_is_complete(dtype)  # marker but no .tensorbin files
        (cache / "model.norm_weight_dtype_bf16_layout_TILE.tensorbin").touch()
        assert args.weight_cache_is_complete(dtype)

        meta = json.loads((cache / ModelArgs.WEIGHT_CACHE_MARKER).read_text())
        assert meta == {
            "format_version": 4,
            "model_name": MODEL_NAME,
            "n_layers": 48,
            "dtype": str(dtype),
            "moe": DEFAULT_MARKER_MOE,
        }

        monkeypatch.setenv("SOLAR_OPEN_FORCE_MODEL_LOAD", "1")
        assert not args.weight_cache_is_complete(dtype)
        monkeypatch.delenv("SOLAR_OPEN_FORCE_MODEL_LOAD")
        assert args.weight_cache_is_complete(dtype)

        # A cache built for MORE layers than the run needs (full 48-layer cache, num_layers=1 debug run) is complete
        # for that run; a cache built for FEWER layers is not.
        args.n_layers = 1
        assert args.weight_cache_is_complete(dtype)
        self._write_marker(cache, {**meta, "n_layers": 1})
        assert args.weight_cache_is_complete(dtype)
        args.n_layers = 48
        assert not args.weight_cache_is_complete(dtype)  # 1-layer marker cannot serve the full model
        self._write_marker(cache, meta)

        for tampered in (
            {**meta, "format_version": 3},  # previous cache format (bias/sink-logit files, router under mlp/router/)
            {**meta, "n_layers": 47},  # partial build
            {**meta, "model_name": "other"},
            {**meta, "moe": {**DEFAULT_MARKER_MOE, "router_impl": "ops"}},
            {**meta, "moe": {**DEFAULT_MARKER_MOE, "router_fp32_logits": False}},
            {**meta, "moe": {**DEFAULT_MARKER_MOE, "shared_expert_dtype": "bf16"}},
            {k: v for k, v in meta.items() if k != "moe"},
        ):
            self._write_marker(cache, tampered)
            assert not args.weight_cache_is_complete(dtype), tampered
        (cache / ModelArgs.WEIGHT_CACHE_MARKER).write_text("not json")
        assert not args.weight_cache_is_complete(dtype)
        self._write_marker(cache, meta)
        assert args.weight_cache_is_complete(dtype)

        # Different expert dtype -> different directory -> cold load
        args.moe_options = MoEOptions(expert_dtype=ttnn.bfloat4_b)
        assert args.weight_cache_path(dtype).name == "tensor_cache_bfp8_expbfp4_(1, 8)"
        assert not args.weight_cache_is_complete(dtype)


def _tiny_state_dict(head_dim=16, n_q_heads=4, n_kv_heads=2, hidden=64, n_experts=4, intermediate=32, vocab=96):
    """Contract-C1-shaped state dict at toy sizes, with fp32 stragglers to exercise the safety-net cast."""
    bf16 = torch.bfloat16
    layer = "model.layers.0."
    return {
        "model.embed_tokens.weight": torch.randn(vocab, hidden, dtype=bf16),
        layer + "input_layernorm.weight": torch.ones(hidden),  # fp32 straggler
        layer + "post_attention_layernorm.weight": torch.ones(hidden, dtype=bf16),
        layer + "self_attn.q_proj.weight": torch.randn(n_q_heads * head_dim, hidden, dtype=bf16),
        layer + "self_attn.k_proj.weight": torch.randn(n_kv_heads * head_dim, hidden, dtype=bf16),
        layer + "self_attn.v_proj.weight": torch.randn(n_kv_heads * head_dim, hidden, dtype=bf16),
        layer + "self_attn.o_proj.weight": torch.randn(hidden, n_q_heads * head_dim, dtype=bf16),
        layer + "mlp.gate.weight": torch.randn(n_experts, hidden, dtype=bf16),
        layer + "mlp.gate.e_score_correction_bias": torch.randint(-5, 6, (n_experts,)).float() * 2**-9,
        layer + "mlp.experts.gate_up_proj": torch.randn(n_experts, 2 * intermediate, hidden, dtype=bf16),
        layer + "mlp.experts.down_proj": torch.randn(n_experts, hidden, intermediate, dtype=bf16),
        layer + "mlp.shared_experts.gate_proj.weight": torch.randn(intermediate, hidden, dtype=bf16),
        layer + "mlp.shared_experts.up_proj.weight": torch.randn(intermediate, hidden, dtype=bf16),
        layer + "mlp.shared_experts.down_proj.weight": torch.randn(hidden, intermediate, dtype=bf16),
        "model.norm.weight": torch.ones(hidden),  # fp32 straggler -> triggers the safety net
        "lm_head.weight": torch.randn(vocab, hidden, dtype=bf16),
    }


class TestLoadStateDict:
    def test_dummy_weights(self):
        assert ModelArgs.load_state_dict("anything", dummy_weights=True) == {}

    def test_layout_dtypes_and_meta_permutation(self, monkeypatch):
        head_dim, n_q, n_kv, hidden = 16, 4, 2, 64
        sd = _tiny_state_dict(head_dim=head_dim, n_q_heads=n_q, n_kv_heads=n_kv, hidden=hidden)
        calls = {}

        def fake_from_pretrained(path, **kwargs):
            calls["path"] = path
            calls.update(kwargs)
            return SimpleNamespace(config=SimpleNamespace(head_dim=head_dim), state_dict=lambda: dict(sd))

        monkeypatch.setattr(mc, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=fake_from_pretrained))
        out = ModelArgs.load_state_dict("/ckpt/Solar-Open-100B")

        # transformers 5.x spelling: dtype=, no deprecated torch_dtype / ignored low_cpu_mem_usage
        assert calls == {"path": "/ckpt/Solar-Open-100B", "dtype": torch.bfloat16}
        assert set(out) == set(sd)
        layer = "model.layers.0."
        bias = out[layer + "mlp.gate.e_score_correction_bias"]
        assert bias.dtype == torch.float32 and torch.equal(bias, sd[layer + "mlp.gate.e_score_correction_bias"])
        assert out["model.norm.weight"].dtype == torch.bfloat16
        assert out[layer + "input_layernorm.weight"].dtype == torch.bfloat16
        assert all(v.dtype == torch.bfloat16 for k, v in out.items() if not k.endswith("e_score_correction_bias"))
        # q/k Meta-permuted with n_heads = rows // head_dim (64 and 8 for Solar), v/o and the MoE tensors untouched
        q, k = layer + "self_attn.q_proj.weight", layer + "self_attn.k_proj.weight"
        assert torch.equal(out[q], reverse_permute(sd[q], n_q, n_q * head_dim, hidden))
        assert torch.equal(out[k], reverse_permute(sd[k], n_kv, n_kv * head_dim, hidden))
        assert not torch.equal(out[q], sd[q])
        for key in (
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ):
            assert out[layer + key] is sd[layer + key]
        assert tuple(out[layer + "mlp.experts.gate_up_proj"].shape) == (4, 64, 64)  # [E, 2I, H]
        assert tuple(out[layer + "mlp.experts.down_proj"].shape) == (4, 64, 32)  # [E, H, I]

        out_hf = ModelArgs.load_state_dict("/ckpt/Solar-Open-100B", convert_to_meta_format=False)
        assert torch.equal(out_hf[q], sd[q].to(torch.bfloat16))

    def test_rejects_unfused_expert_layout(self, monkeypatch, expect_error):
        sd = _tiny_state_dict()
        gate_up = sd.pop("model.layers.0.mlp.experts.gate_up_proj")
        sd.pop("model.layers.0.mlp.experts.down_proj")
        for e in range(gate_up.shape[0]):  # transformers 4.x per-expert keys
            sd[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = gate_up[e, :32]
        monkeypatch.setattr(
            mc,
            "AutoModelForCausalLM",
            SimpleNamespace(
                from_pretrained=lambda *a, **k: SimpleNamespace(
                    config=SimpleNamespace(head_dim=16), state_dict=lambda: dict(sd)
                )
            ),
        )
        with expect_error(ValueError, "transformers >= 5.12"):
            ModelArgs.load_state_dict("/ckpt/Solar-Open-100B")

    def test_rejects_wrong_fused_shapes(self, expect_error):
        sd = _tiny_state_dict()
        sd["model.layers.0.mlp.experts.down_proj"] = sd["model.layers.0.mlp.experts.down_proj"].transpose(1, 2)
        with expect_error(ValueError, r"\[E, 2I, H\]"):
            mc._validate_state_dict_layout(sd)
        sd = _tiny_state_dict()
        sd["model.layers.0.mlp.gate.e_score_correction_bias"] = sd[
            "model.layers.0.mlp.gate.e_score_correction_bias"
        ].to(torch.bfloat16)
        with expect_error(ValueError, "fp32"):
            mc._validate_state_dict_layout(sd)

    def test_streaming_loader_is_phase_2(self, monkeypatch, expect_error):
        monkeypatch.setenv("SOLAR_OPEN_STREAMING_LOAD", "1")
        with expect_error(NotImplementedError, r"DESIGN\.md"):
            ModelArgs.load_state_dict("/ckpt/Solar-Open-100B")


class TestKVBudget:
    SOLAR = dict(num_kv_heads=8, head_dim=128, n_layers=48, tensor_parallel=8)

    @pytest.mark.parametrize(
        "blocks, expected_gib",
        [
            (4096, 3.1875),  # demo batch32: 32 users x 8K / 64
            (2048, 1.59375),  # demo single user 128K / 64
            (8192, 6.375),  # 32 x 16K
            (16384, 12.75),  # 32 x 32K
        ],
    )
    def test_solar_footprint(self, blocks, expected_gib):
        cfg = PagedAttentionConfig(block_size=64, max_num_blocks=blocks)
        assert paged_kv_cache_gib(paged_attention_config=cfg, **self.SOLAR) == pytest.approx(expected_gib)
        # 13,056 B per token per device
        assert paged_kv_cache_gib(paged_attention_config=cfg, **self.SOLAR) * 2**30 / (blocks * 64) == pytest.approx(
            13056
        )

    def test_budget_defaults_and_override(self, monkeypatch):
        assert kv_budget_gib(MoEOptions()) == 16.0
        assert kv_budget_gib(MoEOptions(expert_dtype=ttnn.bfloat4_b)) == 22.0
        monkeypatch.setenv("SOLAR_OPEN_KV_BUDGET_GIB", "9.5")
        assert kv_budget_gib(MoEOptions()) == 9.5

    def test_guard(self, monkeypatch, expect_error):
        bfp8, bfp4 = MoEOptions(), MoEOptions(expert_dtype=ttnn.bfloat4_b)
        ok = PagedAttentionConfig(block_size=64, max_num_blocks=4096)
        assert check_kv_budget(paged_attention_config=ok, moe_options=bfp8, **self.SOLAR) == pytest.approx(3.1875)
        assert check_kv_budget(
            paged_attention_config=PagedAttentionConfig(64, 16384), moe_options=bfp8, **self.SOLAR
        ) == pytest.approx(12.75)
        borderline = PagedAttentionConfig(block_size=64, max_num_blocks=20480)  # 15.94 GiB: under 22, over 16
        check_kv_budget(paged_attention_config=borderline, moe_options=bfp4, **self.SOLAR)
        with expect_error(ValueError, "SOLAR_OPEN_KV_BUDGET_GIB"):
            check_kv_budget(paged_attention_config=PagedAttentionConfig(64, 32768), moe_options=bfp8, **self.SOLAR)
        with expect_error(ValueError, "above the 22.0 GiB budget"):
            check_kv_budget(paged_attention_config=PagedAttentionConfig(64, 32768), moe_options=bfp4, **self.SOLAR)
        monkeypatch.setenv("SOLAR_OPEN_KV_BUDGET_GIB", "30")
        assert check_kv_budget(
            paged_attention_config=PagedAttentionConfig(64, 32768), moe_options=bfp8, **self.SOLAR
        ) == pytest.approx(25.5)
        # TP=1 keeps all 8 KV heads on the one device
        assert paged_kv_cache_gib(8, 128, 48, ok, tensor_parallel=1) == pytest.approx(8 * 3.1875)

    def test_unpaged_guard(self, expect_error):
        bfp8 = MoEOptions()
        # Same 13,056 B per (user, position) per device as the paged pool: 32 x 8K == 4096 blocks x 64.
        assert unpaged_kv_cache_gib(max_local_batch_size=32, max_seq_len=8192, **self.SOLAR) == pytest.approx(3.1875)
        assert check_kv_budget(
            paged_attention_config=None, moe_options=bfp8, max_local_batch_size=32, max_seq_len=8192, **self.SOLAR
        ) == pytest.approx(3.1875)
        # test_model / the unit tests: 32 users x 128 positions
        check_kv_budget(
            paged_attention_config=None, moe_options=bfp8, max_local_batch_size=32, max_seq_len=128, **self.SOLAR
        )
        # Model(max_seq_len=None) fallback: 32 users x max_position_embeddings (131072) = 51 GiB per device
        with expect_error(ValueError, r"Unpaged KV cache: 32 users x 131072 positions needs 51\.00 GiB"):
            check_kv_budget(
                paged_attention_config=None, moe_options=bfp8, max_local_batch_size=32, max_seq_len=131072, **self.SOLAR
            )
        with expect_error(AssertionError, "unpaged KV check needs"):
            check_kv_budget(paged_attention_config=None, moe_options=bfp8, **self.SOLAR)
