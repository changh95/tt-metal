# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only smoke test of the Solar-Open vLLM wrapper plumbing. No device is opened; vllm need not be installed.

    pytest models/demos/solar_open/tests/unit/test_vllm_wrapper_import.py

Two layers:
- ``models.demos.solar_open.tt.vllm_support`` (vllm-free) is imported and its plumbing checked with mocks: the token
  capacity follows the KV budget, the 48-layer full-attention KV spec, the request validation, the ``ttnn.from_torch``
  pool allocator (with a mocked ttnn and a mocked mesh) incl. the budget refusal, the stop ids and the template kwargs.
- ``models.tt_transformers.tt.generator_vllm.SolarOpenForCausalLM`` imports vllm at module level, so it is imported
  only where vllm is importable (skipped otherwise). Without vllm the class is still checked structurally by parsing
  the source (capabilities dict == vllm_support.MODEL_CAPABILITIES, method set, initialize_vllm_model signature).

Nothing here talks to a live vLLM; the wrapper is UNTESTED against one (see README "Serving with vLLM").
"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.solar_open.config import MoEOptions
from models.demos.solar_open.tt import vllm_support as vs
from models.demos.solar_open.tt.common import KV_BUDGET_ENV, kv_budget_gib, kv_budget_tokens

SOLAR_OPEN_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = SOLAR_OPEN_DIR / "configs" / "Solar-Open-100B"
GENERATOR_VLLM_PY = SOLAR_OPEN_DIR.parents[1] / "tt_transformers" / "tt" / "generator_vllm.py"
PARSER_PLUGIN_PY = SOLAR_OPEN_DIR / "vllm_plugins" / "solar_open_parsers.py"
WRAPPER_METHODS = {
    "get_max_tokens_all_users",
    "get_kv_cache_spec",
    "initialize_vllm_model",
    "cache_path",
    "stop_token_ids",
    "prefill_forward",
    "decode_forward",
    "allocate_kv_cache",
    "allocate_kv_cache_per_layer",
}
INITIALIZE_PARAMS = [
    "hf_config",
    "mesh_device",
    "max_batch_size",
    "max_seq_len",
    "n_layers",
    "tt_data_parallel",
    "optimizations",
]
SOLAR_ENV_VARS = (
    KV_BUDGET_ENV,
    "SOLAR_OPEN_EXPERT_DTYPE",
    "SOLAR_OPEN_SHARED_EXPERT_DTYPE",
    "SOLAR_OPEN_ROUTER_IMPL",
    "SOLAR_OPEN_ROUTER_FP32_LOGITS",
    "SOLAR_OPEN_SHARED_DOWN_BFP8",
    "SOLAR_OPEN_ATTENTION_FUSED_QK",
    "SOLAR_OPEN_ATTENTION_OUT_GRID",
    "SOLAR_OPEN_REASONING_EFFORT",
    "SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT",
)
needs_vllm = pytest.mark.skipif(
    not vs.vllm_available(), reason="vllm is not installed; generator_vllm.py imports vllm at module level"
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in SOLAR_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def hf_config():
    return AutoConfig.from_pretrained(CONFIG_DIR)


def _ttnn_mock():
    mock = MagicMock(name="ttnn")
    mock.from_torch.side_effect = lambda zeros, **kw: ("tt-tensor", tuple(zeros.shape), kw["dtype"], kw["device"])
    mock.ReplicateTensorToMesh.side_effect = lambda mesh: ("replicate", mesh)
    mock.bfloat8_b = "bfloat8_b"
    mock.TILE_LAYOUT = "TILE"
    mock.DRAM_MEMORY_CONFIG = "DRAM"
    return mock


def _fake_model(mesh_shape=(1, 8)):
    mesh = MagicMock(name="mesh_device")
    mesh.shape = mesh_shape
    model = MagicMock(name="model")
    model.mesh_device = mesh
    return model


def _wrapper_class_ast():
    tree = ast.parse(GENERATOR_VLLM_PY.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == vs.VLLM_WRAPPER_CLASS]
    assert len(classes) == 1, f"{vs.VLLM_WRAPPER_CLASS} not defined exactly once in {GENERATOR_VLLM_PY}"
    return classes[0]


class TestVllmSupportConstants:
    def test_bytes_per_token_and_registry_names(self):
        assert vs.KV_BYTES_PER_TOKEN_PER_DEVICE == 13056
        assert vs.VLLM_ARCHITECTURE == "SolarOpenForCausalLM"
        assert (vs.VLLM_WRAPPER_MODULE, vs.VLLM_WRAPPER_CLASS) == (
            "models.tt_transformers.tt.generator_vllm",
            "SolarOpenForCausalLM",
        )
        assert vs.MODEL_CAPABILITIES["supports_prefix_caching"] is False
        assert vs.MODEL_CAPABILITIES["supports_sample_on_device"] is True
        assert vs.MODEL_CAPABILITIES["max_device_top_k"] == 32
        assert vs.GENERATION_STOP_TOKEN_IDS == (2, 24, 25)

    def test_config_architecture_matches_registry_key(self, hf_config):
        assert hf_config.architectures == [vs.VLLM_ARCHITECTURE]
        assert hf_config.num_hidden_layers == vs.NUM_LAYERS
        assert hf_config.num_key_value_heads == vs.NUM_KV_HEADS
        assert hf_config.head_dim == vs.HEAD_DIM
        assert hf_config.max_position_embeddings == vs.MAX_SEQ_LEN


class TestTokenCapacity:
    @pytest.mark.parametrize("expert_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bfp8", "bfp4"])
    def test_follows_kv_budget(self, expert_dtype):
        opts = MoEOptions(expert_dtype=expert_dtype)
        expected = int(kv_budget_gib(opts) * 2**30 // vs.KV_BYTES_PER_TOKEN_PER_DEVICE) // 64 * 64
        assert vs.max_tokens_all_users(opts) == expected == kv_budget_tokens(opts)
        assert vs.max_tokens_all_users(opts) % vs.KV_BLOCK_SIZE == 0

    def test_env_budget_override(self, monkeypatch):
        monkeypatch.setenv(KV_BUDGET_ENV, "13")
        assert vs.max_tokens_all_users(MoEOptions()) == 1_069_120  # int(13 GiB // 13056) // 64 * 64
        monkeypatch.setenv(KV_BUDGET_ENV, "8")
        assert vs.max_tokens_all_users(MoEOptions()) == 657_920

    def test_defaults_to_env_moe_options(self, monkeypatch):
        monkeypatch.setenv("SOLAR_OPEN_EXPERT_DTYPE", "bfp4")
        assert vs.max_tokens_all_users() == kv_budget_tokens(MoEOptions(expert_dtype=ttnn.bfloat4_b))


class TestKVCacheSpec:
    def test_layer_types_all_full_attention(self, hf_config):
        assert vs.full_attention_layer_types(hf_config) == ["full_attention"] * 48

    def test_rejects_sliding_layers(self, expect_error):
        cfg = SimpleNamespace(num_hidden_layers=2, layer_types=["full_attention", "sliding_attention"])
        with expect_error(ValueError, "sliding_attention"):
            vs.full_attention_layer_types(cfg)

    def test_build_spec(self):
        spec_cls = MagicMock(side_effect=lambda **kw: ("spec", tuple(sorted(kw.items()))))
        common = dict(block_size=64, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
        spec = vs.build_full_attention_kv_cache_spec(48, spec_cls, **common)
        assert list(spec) == [f"model.layers.{i}.self_attn" for i in range(48)]
        assert spec_cls.call_count == 48
        assert set(spec.values()) == {("spec", tuple(sorted(common.items())))}


class TestRequestValidation:
    OK = dict(mesh_shape=(1, 8), max_batch_size=32, max_seq_len=8192, tt_data_parallel=1)

    @pytest.mark.parametrize("name", ["upstage/Solar-Open-100B", "/models/Solar-Open-100B", "SolarOpen100B"])
    def test_accepts_validated_configuration(self, name):
        vs.validate_vllm_model_request(hf_name_or_path=name, **self.OK)
        vs.validate_vllm_model_request(hf_name_or_path=name, **{**self.OK, "max_batch_size": 1, "max_seq_len": 131072})

    def test_accepts_empty_name_with_warning(self):
        vs.validate_vllm_model_request(hf_name_or_path="", **self.OK)

    @pytest.mark.parametrize(
        "override, message",
        [
            ({"mesh_shape": (2, 4)}, "mesh shape"),
            ({"mesh_shape": (1, 4)}, "mesh shape"),
            ({"tt_data_parallel": 2}, "data parallelism"),
            ({"max_batch_size": 64}, "max-num-seqs"),
            ({"max_batch_size": 0}, "max-num-seqs"),
            ({"max_seq_len": 131073}, "max-model-len"),
            ({"hf_name_or_path": "meta-llama/Llama-3.1-70B"}, "is not Solar-Open-100B"),
            ({"optimizations": "performance"}, "optimization presets"),
        ],
    )
    def test_rejects(self, override, message, expect_error):
        kwargs = {"hf_name_or_path": "upstage/Solar-Open-100B", **self.OK, **override}
        with expect_error(ValueError, message):
            vs.validate_vllm_model_request(**kwargs)


class TestPagedKVCacheAllocator:
    def test_per_device_kv_heads(self, expect_error):
        assert vs.per_device_kv_heads(1) == 1
        assert vs.per_device_kv_heads(8) == 1  # undivided head count accepted with a warning
        with expect_error(ValueError, "KV heads"):
            vs.per_device_kv_heads(2)

    def test_allocates_replicated_zero_pools_without_tensorbins(self):
        model = _fake_model()
        with patch.object(vs, "ttnn", new=_ttnn_mock()) as ttnn_mock:
            kv_cache = vs.allocate_paged_kv_cache(
                [model], (4096, 1, 64, 128), 48, MoEOptions(), vllm_dtype=torch.bfloat16
            )
        assert ttnn_mock.from_torch.call_count == 2 * 48
        assert ttnn_mock.as_tensor.call_count == 0  # never the cache_file_name path
        assert len(kv_cache) == 1 and len(kv_cache[0]) == 48 and all(len(layer) == 2 for layer in kv_cache[0])
        for k, v in kv_cache[0]:
            # cache_dtype defaults to the real ttnn.bfloat8_b (bound at definition time, unaffected by the ttnn patch).
            assert k == v == ("tt-tensor", (4096, 1, 64, 128), ttnn.bfloat8_b, model.mesh_device)
        for call in ttnn_mock.from_torch.call_args_list:
            assert torch.count_nonzero(call.args[0]) == 0
            assert call.kwargs["layout"] == "TILE" and call.kwargs["memory_config"] == "DRAM"
            assert call.kwargs["mesh_mapper"] == ("replicate", model.mesh_device)

    def test_undivided_kv_heads_allocate_per_device_shape(self):
        model = _fake_model()
        with patch.object(vs, "ttnn", new=_ttnn_mock()):
            kv_cache = vs.allocate_paged_kv_cache([model], (512, 8, 64, 128), 2, MoEOptions())
        assert kv_cache[0][0][0][1] == (512, 1, 64, 128)

    def test_refuses_pool_above_budget_before_allocating(self, expect_error):
        model = _fake_model()
        with patch.object(vs, "ttnn", new=_ttnn_mock()) as ttnn_mock:
            # 32768 x 64 tokens x 13,056 B = 25.5 GiB: above every budget and hard cap of tt/common.py.
            with expect_error(ValueError, KV_BUDGET_ENV):
                vs.allocate_paged_kv_cache([model], (32768, 1, 64, 128), 48, MoEOptions())
        assert ttnn_mock.from_torch.call_count == 0

    def test_budget_env_admits_larger_pool(self, monkeypatch, expect_error):
        model = _fake_model()
        blocks_32x32k = 16384  # 12.75 GiB
        with patch.object(vs, "ttnn", new=_ttnn_mock()):
            if kv_budget_gib(MoEOptions()) < 12.75:
                with expect_error(ValueError, KV_BUDGET_ENV):
                    vs.allocate_paged_kv_cache([model], (blocks_32x32k, 1, 64, 128), 48, MoEOptions())
            monkeypatch.setenv(KV_BUDGET_ENV, "13")
            kv_cache = vs.allocate_paged_kv_cache([model], (blocks_32x32k, 1, 64, 128), 48, MoEOptions())
        assert len(kv_cache[0]) == 48

    @pytest.mark.parametrize(
        "shape, message",
        [((4096, 1, 64, 64), "head_dim"), ((4096, 1, 48, 128), "block_size"), ((4096, 1, 64), "kv_cache_shape")],
    )
    def test_rejects_malformed_shapes(self, shape, message, expect_error):
        with patch.object(vs, "ttnn", new=_ttnn_mock()):
            with expect_error(ValueError, message):
                vs.allocate_paged_kv_cache([_fake_model()], shape, 48, MoEOptions())


class TestStopIdsAndTemplate:
    def test_stop_ids_from_model_args(self):
        assert vs.stop_token_ids_for_vllm(SimpleNamespace(stop_token_ids={25, 2, 24})) == [2, 24, 25]
        assert vs.stop_token_ids_for_vllm(SimpleNamespace(stop_token_ids=set())) == [2, 24, 25]
        assert vs.stop_token_ids_for_vllm(SimpleNamespace()) == list(vs.GENERATION_STOP_TOKEN_IDS)

    def test_chat_template_kwargs(self, monkeypatch):
        assert vs.chat_template_kwargs() == {
            "reasoning_effort": "high",
            "default_system_prompt": True,
            "think_render_option": "lastthink",
        }
        monkeypatch.setenv("SOLAR_OPEN_REASONING_EFFORT", "low")
        monkeypatch.setenv("SOLAR_OPEN_DEFAULT_SYSTEM_PROMPT", "0")
        assert vs.chat_template_kwargs() == {
            "reasoning_effort": "low",
            "default_system_prompt": False,
            "think_render_option": "lastthink",
        }
        assert vs.chat_template_kwargs(reasoning_effort="minimal")["reasoning_effort"] == "minimal"

    @pytest.mark.skipif(vs.vllm_available(), reason="vllm is installed here; the no-vllm error path does not apply")
    def test_parser_registration_needs_vllm(self, expect_error):
        with expect_error(RuntimeError, "vllm is not importable"):
            vs.register_vllm_parsers()

    def test_parser_plugin_file_registers_on_import(self):
        tree = ast.parse(PARSER_PLUGIN_PY.read_text())
        calls = [
            n.value.func.id
            for n in tree.body
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
        ]
        assert calls == ["register_vllm_parsers"]


class TestWrapperSourceStructure:
    """Structural checks on generator_vllm.py that do not need vllm (the module cannot be imported without it)."""

    def test_class_shape(self):
        cls = _wrapper_class_ast()
        assert "HybridAttentionForCausalLM" in {b.id for b in cls.bases if isinstance(b, ast.Name)}
        defined = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        assert WRAPPER_METHODS <= defined, f"missing {WRAPPER_METHODS - defined}"
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "initialize_vllm_model")
        assert [a.arg for a in init.args.args][1:] == INITIALIZE_PARAMS  # the plugin calls by keyword
        assert any(isinstance(d, ast.Name) and d.id == "classmethod" for d in init.decorator_list)
        source = ast.get_source_segment(GENERATOR_VLLM_PY.read_text(), cls)
        assert "models.demos.solar_open.tt.vllm_support" in source
        assert "models.demos.solar_open.tt.common" in source
        assert "MoEOptions.from_env()" in source
        assert "create_kv_cache=False" in source

    def test_capabilities_match_support_module(self):
        cls = _wrapper_class_ast()
        assigns = [
            n
            for n in cls.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "model_capabilities" for t in n.targets)
        ]
        assert len(assigns) == 1
        assert ast.literal_eval(assigns[0].value) == vs.MODEL_CAPABILITIES


@needs_vllm
class TestWrapperWithVllm:
    """Runs only where vllm is importable (not on the bring-up box). Still host-only: models and ttnn are mocked."""

    @staticmethod
    def _cls():
        from models.tt_transformers.tt.generator_vllm import SolarOpenForCausalLM

        return SolarOpenForCausalLM

    def test_capabilities_and_signature(self):
        cls = self._cls()
        assert cls.model_capabilities == vs.MODEL_CAPABILITIES
        assert list(inspect.signature(cls.initialize_vllm_model).parameters) == INITIALIZE_PARAMS
        assert cls.get_max_tokens_all_users() == vs.max_tokens_all_users(MoEOptions())

    def test_kv_cache_spec(self, hf_config):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=hf_config,
                dtype=torch.bfloat16,
                get_num_kv_heads=lambda parallel_config: 8,
                get_head_size=lambda: 128,
            ),
            cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
            parallel_config=None,
        )
        spec = self._cls().get_kv_cache_spec(vllm_config)
        assert list(spec) == [f"model.layers.{i}.self_attn" for i in range(48)]
        assert all(isinstance(s, FullAttentionSpec) for s in spec.values())
        first = spec["model.layers.0.self_attn"]
        assert (first.block_size, first.num_kv_heads, first.head_size, first.dtype) == (64, 8, 128, torch.bfloat16)

    def test_initialize_rejects_before_building_the_model(self, expect_error):
        cls = self._cls()
        mesh = MagicMock()
        mesh.shape = (2, 4)
        with patch("models.demos.solar_open.tt.common.create_tt_model") as create:
            with expect_error(ValueError, "mesh shape"):
                cls.initialize_vllm_model(
                    SimpleNamespace(_name_or_path="upstage/Solar-Open-100B"), mesh, max_batch_size=32, max_seq_len=8192
                )
        assert create.call_count == 0

    def test_allocate_kv_cache_through_the_class(self):
        from models.tt_transformers.tt.generator import Generator

        cls = self._cls()
        model = _fake_model()
        instance = cls.__new__(cls)
        Generator.__init__(
            instance,
            [model],
            [SimpleNamespace(moe_options=MoEOptions(), stop_token_ids={2, 24, 25})],
            model.mesh_device,
        )
        with patch.object(vs, "ttnn", new=_ttnn_mock()) as ttnn_mock:
            kv_cache = instance.allocate_kv_cache((4096, 1, 64, 128), torch.bfloat16, 48)
            per_layer = instance.allocate_kv_cache_per_layer(
                [((4096, 1, 64, 128), torch.bfloat16, i) for i in range(48)]
            )
        assert ttnn_mock.from_torch.call_count == 2 * 2 * 48
        assert len(kv_cache[0]) == len(per_layer[0]) == 48
        assert instance.stop_token_ids == [2, 24, 25]
