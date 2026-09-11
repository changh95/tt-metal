# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only smoke test of the Solar-Open vLLM wrapper plumbing. No device is opened; vllm need not be installed.

    pytest models/demos/solar_open/tests/unit/test_vllm_wrapper_import.py

Three layers:
- ``models.demos.solar_open.tt.vllm_support`` (vllm-free) is imported and its plumbing checked with mocks: the token
  capacity follows the KV budget minus the plugin's headroom (and the pool the plugin sizes from it passes the budget
  guard), the 48-layer full-attention KV spec, the request validation, the ``ttnn.from_torch`` pool allocator (with a
  mocked ttnn and a mocked mesh) incl. the budget refusal, the stop ids and the template kwargs.
- ``models.tt_transformers.tt.generator_vllm.SolarOpenForCausalLM`` imports vllm at module level, so it is imported
  only where vllm is importable (skipped otherwise). Without vllm the class is still checked structurally by parsing
  the source (capabilities dict == vllm_support.MODEL_CAPABILITIES, method set, initialize_vllm_model signature).
- With vllm importable AND ``HF_MODEL`` pointing at a Solar-Open-100B snapshot, the vLLM-0.12 import shims are
  checked against the installed vLLM and Upstage's reasoning / tool parsers are registered (also through the plugin
  file, the way ``--reasoning-parser-plugin`` / ``--tool-parser-plugin`` import it), instantiated on the Solar
  tokenizer and run on a think/content sample.

Nothing here talks to a live vLLM server (see README "Serving with vLLM").
"""

import ast
import importlib
import inspect
import io
import json
import logging
import os
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
    # vLLM ``VllmModelForTextGeneration`` protocol shim: the plain arch resolves to this class (model-class-overrides)
    "__init__",
    "embed_input_ids",
    "forward",
    "compute_logits",
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


def _solar_model_dir():
    """The HF snapshot with Upstage's parser files (``HF_MODEL``, else the tree default), or None."""
    path = Path(os.getenv("HF_MODEL", vs.DEFAULT_HF_MODEL))
    files = (path / vs.REASONING_PARSER_FILE, path / vs.TOOL_PARSER_FILE, path / "tokenizer.json")
    return path if all(f.is_file() for f in files) else None


needs_parser_files = pytest.mark.skipif(
    _solar_model_dir() is None,
    reason="HF_MODEL does not point at a Solar-Open-100B snapshot with Upstage's parser files and tokenizer.json",
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
    def test_follows_kv_budget_minus_plugin_headroom(self, expert_dtype):
        opts = MoEOptions(expert_dtype=expert_dtype)
        budget = int(kv_budget_gib(opts) * 2**30 // vs.KV_BYTES_PER_TOKEN_PER_DEVICE) // 64 * 64
        assert vs.kv_budget_tokens_solar(opts) == budget == kv_budget_tokens(opts)
        assert vs.max_tokens_all_users(opts, max_num_seqs=0) == budget
        # What the wrapper reports: the budget minus the plugin's one-block-per-user headroom (worker.py adds it back).
        assert vs.max_tokens_all_users(opts) == budget - 32 * 64
        assert vs.max_tokens_all_users(opts, max_num_seqs=8) == budget - 8 * 64
        assert vs.max_tokens_all_users(opts) % vs.KV_BLOCK_SIZE == 0

    def test_env_budget_override(self, monkeypatch):
        monkeypatch.setenv(KV_BUDGET_ENV, "13")
        assert vs.kv_budget_tokens_solar(MoEOptions()) == 1_069_120  # int(13 GiB // 13056) // 64 * 64
        assert vs.max_tokens_all_users(MoEOptions()) == 1_069_120 - 2_048
        monkeypatch.setenv(KV_BUDGET_ENV, "8")
        assert vs.kv_budget_tokens_solar(MoEOptions()) == 657_920
        assert vs.max_tokens_all_users(MoEOptions(), max_num_seqs=32) == 655_872

    def test_defaults_to_env_moe_options(self, monkeypatch):
        monkeypatch.setenv("SOLAR_OPEN_EXPERT_DTYPE", "bfp4")
        assert vs.max_tokens_all_users() == kv_budget_tokens(MoEOptions(expert_dtype=ttnn.bfloat4_b)) - 2_048

    def test_plugin_headroom_arithmetic(self, monkeypatch):
        assert vs.plugin_headroom_tokens() == 2_048
        assert vs.plugin_headroom_tokens(max_num_seqs=1) == 64
        # ceil((655,872 + 2,048) / 64) = 10,280 blocks = 8.00 GiB; the pre-fix full-budget figure gave 10,312 = 8.025 GiB.
        assert vs.plugin_kv_pool_blocks(655_872) == 10_280
        assert vs.plugin_kv_pool_blocks(657_920) == 10_312
        monkeypatch.setenv(KV_BUDGET_ENV, "8")
        reported = vs.max_tokens_all_users(MoEOptions(), max_num_seqs=32)
        assert vs.plugin_kv_pool_blocks(reported) * 64 == vs.kv_budget_tokens_solar(MoEOptions()) == 657_920

    def test_headroom_cannot_exceed_budget(self, monkeypatch, expect_error):
        monkeypatch.setenv(KV_BUDGET_ENV, "0.01")  # 768 tokens < the 2,048-token headroom of 32 users
        with expect_error(ValueError, "headroom"):
            vs.max_tokens_all_users(MoEOptions(), max_num_seqs=32)


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


class TestResolveModelDir:
    """``_resolve_model_dir`` accepts a checkpoint directory OR an HF repo id (what vLLM's --model and a tt-model
    container export as HF_MODEL); a repo id resolves to the HF-cache snapshot through ``snapshot_download``."""

    def test_directory_is_returned_as_is(self, tmp_path):
        assert vs._resolve_model_dir(str(tmp_path)) == tmp_path

    def test_env_directory_is_used_when_no_argument(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_MODEL", str(tmp_path))
        assert vs._resolve_model_dir() == tmp_path

    def test_repo_id_resolves_through_the_hf_cache(self, tmp_path, monkeypatch):
        import huggingface_hub

        snapshot = tmp_path / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        calls = []

        def fake_snapshot_download(repo_id, revision=None, allow_patterns=None, **kw):
            calls.append((repo_id, revision, tuple(allow_patterns)))
            return str(snapshot)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
        monkeypatch.setenv("HF_MODEL", "upstage/Solar-Open-100B")
        monkeypatch.delenv("HF_MODEL_REVISION", raising=False)
        assert vs._resolve_model_dir() == snapshot
        assert calls == [
            ("upstage/Solar-Open-100B", None, (vs.REASONING_PARSER_FILE, vs.TOOL_PARSER_FILE, "config.json"))
        ]

    def test_repo_id_revision_pin(self, tmp_path, monkeypatch):
        import huggingface_hub

        seen = {}
        monkeypatch.setattr(
            huggingface_hub,
            "snapshot_download",
            lambda repo_id, revision=None, **kw: seen.update(rev=revision) or str(tmp_path),
        )
        monkeypatch.setenv("HF_MODEL_REVISION", "1f591439b14055004d1a5d1a975608953a022fea")
        assert vs._resolve_model_dir("upstage/Solar-Open-100B") == tmp_path
        assert seen == {"rev": "1f591439b14055004d1a5d1a975608953a022fea"}

    def test_repo_id_cache_miss_is_actionable(self, monkeypatch, expect_error):
        import huggingface_hub

        def boom(*a, **kw):
            raise OSError("offline and not in cache")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
        with expect_error(FileNotFoundError, r"hf download upstage/Solar-Open-100B"):
            vs._resolve_model_dir("upstage/Solar-Open-100B")

    @pytest.mark.parametrize("spec", ["/nonexistent/Solar-Open-100B", "Solar-Open-100B", "a/b/c", "../x"])
    def test_non_directory_non_repo_id_raises(self, spec, expect_error):
        with expect_error(FileNotFoundError, "is not a directory"):
            vs._resolve_model_dir(spec)


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

    def test_plugin_sized_pool_fits_the_budget(self, monkeypatch, expect_error):
        """The pool the plugin sizes from the wrapper's figure plus its headroom passes check_kv_budget; the pool it
        sized from the pre-fix figure (the full budget) is refused - the arithmetic of worker.py's
        get_num_available_blocks_tt against allocate_paged_kv_cache."""
        monkeypatch.setenv(KV_BUDGET_ENV, "8")
        model = _fake_model()
        blocks = vs.plugin_kv_pool_blocks(vs.max_tokens_all_users(MoEOptions(), max_num_seqs=32), max_num_seqs=32)
        assert blocks == 10_280
        with patch.object(vs, "ttnn", new=_ttnn_mock()):
            kv_cache = vs.allocate_paged_kv_cache([model], (blocks, 1, 64, 128), 48, MoEOptions())
            assert len(kv_cache[0]) == 48
            too_many = vs.plugin_kv_pool_blocks(vs.kv_budget_tokens_solar(MoEOptions()), max_num_seqs=32)
            assert too_many == 10_312
            with expect_error(ValueError, KV_BUDGET_ENV):
                vs.allocate_paged_kv_cache([model], (too_many, 1, 64, 128), 48, MoEOptions())

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


class TestSyncErrorLogMirror:
    """The EngineCore-side ERROR mirror for vLLM's logger (plain stdlib logging; no vllm needed)."""

    @staticmethod
    def _cleanup():
        vllm_logger = logging.getLogger("vllm")
        for h in list(vllm_logger.handlers):
            if getattr(h, "name", None) == vs._SYNC_ERROR_LOG_HANDLER_NAME:
                vllm_logger.removeHandler(h)

    def test_installs_once_and_mirrors_errors(self, monkeypatch):
        self._cleanup()
        monkeypatch.delenv(vs.SYNC_ERROR_LOG_ENV, raising=False)
        buf = io.StringIO()
        try:
            assert vs.install_vllm_sync_error_log_handler(stream=buf) is True
            assert vs.install_vllm_sync_error_log_handler(stream=buf) is False  # idempotent
            logging.getLogger("vllm.v1.engine.core").error("EngineCore encountered a fatal error. probe-%d", 42)
            logging.getLogger("vllm.v1.engine.core").info("not mirrored")
            text = buf.getvalue()
            assert "probe-42" in text and "solar_open sync mirror" in text and "not mirrored" not in text
        finally:
            self._cleanup()

    def test_env_disables(self, monkeypatch):
        self._cleanup()
        monkeypatch.setenv(vs.SYNC_ERROR_LOG_ENV, "0")
        assert vs.install_vllm_sync_error_log_handler(stream=io.StringIO()) is False
        assert not any(
            getattr(h, "name", None) == vs._SYNC_ERROR_LOG_HANDLER_NAME for h in logging.getLogger("vllm").handlers
        )


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

    def test_is_a_vllm_text_generation_model(self):
        """``--model-class-overrides`` makes upstream's ModelConfig inspect THIS class (the plain arch has no
        upstream implementation and would otherwise resolve to the Transformers backend, whose name then sticks in
        model_config.architecture and breaks the plugin's TT lookup). ``runner generate`` needs the protocol."""
        from vllm.model_executor.models.interfaces_base import is_pooling_model, is_text_generation_model

        cls = self._cls()
        assert is_text_generation_model(cls)
        assert not is_pooling_model(cls)

    def test_max_tokens_all_users_as_the_plugin_calls_it(self, monkeypatch):
        """worker.py::get_num_available_blocks_tt's call; its headroom then restores exactly the budget."""
        monkeypatch.setenv(KV_BUDGET_ENV, "8")
        cls = self._cls()
        reported = cls.get_max_tokens_all_users(
            model_name="upstage/Solar-Open-100B",
            num_devices=8,
            tt_data_parallel=1,
            max_model_len=16384,
            max_num_seqs=32,
        )
        assert reported == vs.max_tokens_all_users(MoEOptions(), max_num_seqs=32) == 655_872
        assert vs.plugin_kv_pool_blocks(reported, max_num_seqs=32) * 64 == vs.kv_budget_tokens_solar(MoEOptions())
        assert cls.get_max_tokens_all_users(max_num_seqs=4) == 657_920 - 4 * 64
        assert cls.get_max_tokens_all_users() == 655_872  # no kwarg: the 32-user cap, the conservative default

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


@needs_vllm
class TestCompatShimsWithVllm:
    """The vLLM-0.12 module paths Upstage's parser files import, re-created from the installed vLLM's modules."""

    def test_legacy_paths_resolve_to_the_current_classes(self):
        import vllm.entrypoints.openai as openai_pkg

        result = vs.install_vllm_compat_shims()
        assert set(result) == {vs.LEGACY_PROTOCOL_MODULE, vs.LEGACY_ABSTRACT_TOOL_PARSER_MODULE}
        assert set(result.values()) <= {"native", "shim"}
        assert vs.install_vllm_compat_shims() == result  # idempotent
        protocol = importlib.import_module(vs.LEGACY_PROTOCOL_MODULE)  # the path Upstage's files import
        for name, home in vs.LEGACY_PROTOCOL_NAMES.items():
            assert hasattr(protocol, name), name
            if result[vs.LEGACY_PROTOCOL_MODULE] == "shim":
                assert getattr(protocol, name) is getattr(importlib.import_module(home), name)
        assert openai_pkg.protocol is protocol
        legacy_abstract = importlib.import_module(vs.LEGACY_ABSTRACT_TOOL_PARSER_MODULE)
        assert hasattr(legacy_abstract, "ToolParser")
        if result[vs.LEGACY_ABSTRACT_TOOL_PARSER_MODULE] == "shim":
            current = importlib.import_module(vs.CURRENT_TOOL_PARSERS_PACKAGE + ".abstract_tool_parser")
            assert legacy_abstract is current
            assert importlib.import_module(vs.LEGACY_TOOL_PARSERS_PACKAGE) is importlib.import_module(
                vs.CURRENT_TOOL_PARSERS_PACKAGE
            )


@needs_vllm
@needs_parser_files
class TestParserRegistrationWithVllm:
    """Upstage's parsers registered under "solar_open" (directly and through the plugin file, the way vLLM's
    --reasoning-parser-plugin / --tool-parser-plugin import it) and run on the Solar tokenizer. Host only."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(_solar_model_dir())

    def test_register_and_parse(self, tokenizer):
        from vllm.reasoning import ReasoningParserManager
        from vllm.tool_parsers import ToolParserManager

        registered = vs.register_vllm_parsers()
        assert {k: v.__name__ for k, v in registered.items()} == {
            "reasoning": "SolarOpenTTReasoningParser",
            "reasoning_upstream": "SolarOpenReasoningParser",
            "tool": "SolarOpenToolParser",
        }
        assert ReasoningParserManager.get_reasoning_parser(vs.REASONING_PARSER_NAME) is registered["reasoning"]
        assert (
            ReasoningParserManager.get_reasoning_parser(vs.REASONING_PARSER_UPSTREAM_NAME)
            is registered["reasoning_upstream"]
        )
        assert issubclass(registered["reasoning"], registered["reasoning_upstream"])
        assert ToolParserManager.get_tool_parser(vs.TOOL_PARSER_NAME) is registered["tool"]

        for cls in (registered["reasoning"], registered["reasoning_upstream"]):
            reasoning = cls(tokenizer)
            sample = "<|think|>Plan the answer.<|end|><|content|>Seoul."
            assert reasoning.extract_reasoning(sample, request=None) == ("Plan the answer.", "Seoul.")
            assert reasoning.extract_reasoning("<|think|><|end|><|content|>Hi", request=None) == ("", "Hi")  # `low`
            assert reasoning.is_reasoning_end(tokenizer.encode(sample, add_special_tokens=False))
            assert not reasoning.is_reasoning_end(tokenizer.encode("<|think|>Still thinking", add_special_tokens=False))
        assert registered["reasoning"](tokenizer).reasoning_start_str == "<|think|>"  # thinking-budget ids, no warning

        tool = registered["tool"](tokenizer)
        call = (
            "Checking the weather.<|flush|><|tool_calls|><|tool_call:begin|><|tool_call:name|>get_weather"
            '<|tool_call:args|>{"city": "Seoul"}<|tool_call:end|><|calls|>'
        )
        info = tool.extract_tool_calls(call, request=None)
        assert info.tools_called and len(info.tool_calls) == 1 and info.content == "Checking the weather."
        assert info.tool_calls[0].function.name == "get_weather"
        assert json.loads(info.tool_calls[0].function.arguments) == {"city": "Seoul"}
        assert tool.extract_tool_calls("Plain answer.<|flush|>", request=None).tools_called is False

    def test_plugin_file_loads_through_the_vllm_import_path(self):
        from vllm.reasoning import ReasoningParserManager
        from vllm.tool_parsers import ToolParserManager

        ReasoningParserManager.reasoning_parsers.pop(vs.REASONING_PARSER_NAME, None)
        ToolParserManager.tool_parsers.pop(vs.TOOL_PARSER_NAME, None)
        ReasoningParserManager.import_reasoning_parser(str(PARSER_PLUGIN_PY))  # what --reasoning-parser-plugin does
        ToolParserManager.import_tool_parser(str(PARSER_PLUGIN_PY))  # what --tool-parser-plugin does
        assert (
            ReasoningParserManager.get_reasoning_parser(vs.REASONING_PARSER_NAME).__name__
            == "SolarOpenTTReasoningParser"
        )
        assert (
            ReasoningParserManager.get_reasoning_parser(vs.REASONING_PARSER_UPSTREAM_NAME).__name__
            == "SolarOpenReasoningParser"
        )
        assert ToolParserManager.get_tool_parser(vs.TOOL_PARSER_NAME).__name__ == "SolarOpenToolParser"


@needs_vllm
@needs_parser_files
class TestReasoningParserStreamingWithVllm:
    """The fixed reasoning parser driven through vLLM 0.25.1's own ``Parser.parse_delta`` state machine (the one the
    chat endpoint uses), token by token, against the deltas seen live on 2026-09-10; the upstream class registered as
    ``solar_open_upstream`` documents the defects it fixes. Host only: template-rendered prompts, hand-written outputs.
    """

    THINK = "The user asks for the capital of Korea. Seoul."
    ANSWER = "Seoul is the capital of South Korea."
    TOOL_CALL = (
        '<|tool_calls|><|tool_call:begin|>abc123defg<|tool_call:name|>get_weather<|tool_call:args|>{"city": "Seoul"}'
        "<|tool_call:end|>"
    )
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather in a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    USER = [{"role": "user", "content": "What is the capital of Korea?"}]

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(_solar_model_dir())

    @pytest.fixture(scope="class")
    def parsers(self):
        """``{"fixed": Parser subclass, "upstream": Parser subclass}`` composed by vLLM's ParserManager."""
        from vllm.parser import ParserManager

        vs.register_vllm_parsers()
        return {
            "fixed": ParserManager.get_parser(
                tool_parser_name=vs.TOOL_PARSER_NAME,
                reasoning_parser_name=vs.REASONING_PARSER_NAME,
                enable_auto_tools=True,
                model_name="upstage/Solar-Open-100B",
            ),
            "upstream": ParserManager.get_parser(
                tool_parser_name=vs.TOOL_PARSER_NAME,
                reasoning_parser_name=vs.REASONING_PARSER_UPSTREAM_NAME,
                enable_auto_tools=True,
                model_name="upstage/Solar-Open-100B",
            ),
        }

    @staticmethod
    def _request(tools=None):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

        body = {"model": "upstage/Solar-Open-100B", "messages": TestReasoningParserStreamingWithVllm.USER}
        if tools:
            body["tools"] = tools  # the validator sets tool_choice "auto"; without tools it stays "none"
        return ChatCompletionRequest(**body)

    def _prompt_ids(self, tokenizer, effort=None, prefill=None):
        if prefill is not None:
            text = tokenizer.apply_chat_template(
                self.USER + [{"role": "assistant", "content": prefill}],
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        else:
            kwargs = {"reasoning_effort": effort} if effort else {}
            text = tokenizer.apply_chat_template(self.USER, tokenize=False, add_generation_prompt=True, **kwargs)
        return tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _output_ids(tokenizer, text, stop_id):
        # what the engine yields: the tokens of the text, then the stop token (special: its text is "")
        return tokenizer.encode(text, add_special_tokens=False) + [stop_id]

    def _stream(self, parser_cls, tokenizer, prompt_ids, output_ids, request, tools=None):
        """Feed one token per delta the way ``chat_completion_stream_generator`` does; join what the client gets."""
        parser = parser_cls(tokenizer, tools)
        reasoning, content, tool_calls = [], [], []
        for i, token_id in enumerate(output_ids):
            delta = parser.parse_delta(
                delta_text=tokenizer.decode([token_id], skip_special_tokens=True),
                delta_token_ids=[token_id],
                request=request,
                prompt_token_ids=prompt_ids,
                finished=i == len(output_ids) - 1,
            )
            if delta is None:
                continue
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)
            tool_calls.extend(delta.tool_calls)
        return "".join(reasoning), "".join(content), tool_calls

    def _complete(self, parser_cls, tokenizer, output_ids, request, tools=None, enable_auto_tools=False):
        """The non-streaming path: ``Parser.parse`` on the full text (vLLM strips the stop token's text)."""
        parser = parser_cls(tokenizer, tools)
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        return parser.parse(text, request, enable_auto_tools=enable_auto_tools, model_output_token_ids=output_ids)

    def test_is_reasoning_end_on_prompts(self, tokenizer, parsers):
        low, high = self._prompt_ids(tokenizer, "low"), self._prompt_ids(tokenizer, "high")
        prefill = self._prompt_ids(tokenizer, prefill="Seoul is")
        fixed, upstream = parsers["fixed"](tokenizer), parsers["upstream"](tokenizer)
        assert tokenizer.decode(low[-6:]).endswith("<|begin|>assistant<|think|><|end|><|begin|>assistant")
        assert upstream.is_reasoning_end(low)  # rule 1: the template's empty block -> "ended" -> raw stream
        assert not fixed.is_reasoning_end(low)  # the model may still open a think block
        assert not fixed.is_reasoning_end(high) and not upstream.is_reasoning_end(high)
        assert fixed.is_reasoning_end(prefill) and upstream.is_reasoning_end(prefill)  # content already open
        generated = tokenizer.encode(
            f"<|think|>{self.THINK}<|end|><|begin|>assistant<|content|>", add_special_tokens=False
        )
        assert fixed.is_reasoning_end(generated) and upstream.is_reasoning_end(generated)
        assert not fixed.is_reasoning_end(generated[:-3])

    @pytest.mark.parametrize("effort", ["low", "high"])
    def test_second_think_block_streams_as_reasoning(self, tokenizer, parsers, effort):
        """The live defect: reasoning_effort low + the model re-opens <|think|> -> markers and reasoning streamed
        as content. Fixed: reasoning / content split, no marker anywhere; effort high already worked upstream."""
        prompt = self._prompt_ids(tokenizer, effort)
        output = self._output_ids(
            tokenizer, f"<|think|>{self.THINK}<|end|><|begin|>assistant<|content|>{self.ANSWER}", 24
        )
        reasoning, content, tool_calls = self._stream(parsers["fixed"], tokenizer, prompt, output, self._request())
        assert (reasoning, content, tool_calls) == (self.THINK, self.ANSWER, [])
        up_reasoning, up_content, _ = self._stream(parsers["upstream"], tokenizer, prompt, output, self._request())
        if effort == "low":
            assert up_reasoning == "" and "<|think|>" in up_content and "<|begin|>assistant" in up_content
        else:
            assert (up_reasoning, up_content) == (self.THINK, self.ANSWER)

    def test_minimal_without_think_block(self, tokenizer, parsers):
        prompt = self._prompt_ids(tokenizer, "minimal")
        output = self._output_ids(tokenizer, f"<|content|>{self.ANSWER}", 24)
        assert self._stream(parsers["fixed"], tokenizer, prompt, output, self._request()) == ("", self.ANSWER, [])
        _, up_content, _ = self._stream(parsers["upstream"], tokenizer, prompt, output, self._request())
        assert up_content.startswith("<|content|>")  # the tag leaked as content
        for name in ("fixed", "upstream"):
            assert self._complete(parsers[name], tokenizer, output, self._request()) == ("", self.ANSWER, [])

    def test_prefilled_content_passes_through(self, tokenizer, parsers):
        """continue_final_message: the prompt already opened <|content|>; the output carries no tag at all."""
        prompt = self._prompt_ids(tokenizer, prefill="Seoul is")
        rest = " the capital of South Korea."
        output = self._output_ids(tokenizer, rest, 24)
        for name in ("fixed", "upstream"):  # streaming was already right (pass-through after the prompt check)
            assert self._stream(parsers[name], tokenizer, prompt, output, self._request()) == ("", rest, [])
        assert self._complete(parsers["fixed"], tokenizer, output, self._request()) == ("", rest, [])
        _, up_content, _ = self._complete(parsers["upstream"], tokenizer, output, self._request())
        assert not up_content  # the live defect: 56 generated tokens, empty content

    def test_tool_call_after_think_block(self, tokenizer, parsers):
        prompt = self._prompt_ids(tokenizer, "low")
        output = self._output_ids(
            tokenizer, f"<|think|>Need the weather tool.<|end|><|begin|>assistant{self.TOOL_CALL}", 25
        )
        request = self._request(self.TOOLS)
        assert request.tool_choice == "auto"
        reasoning, content, tool_calls = self._stream(
            parsers["fixed"], tokenizer, prompt, output, request, tools=request.tools
        )
        assert reasoning == "Need the weather tool." and content == ""
        assert [tc.function.name for tc in tool_calls if tc.function and tc.function.name] == ["get_weather"]
        assert [tc.id for tc in tool_calls if tc.id] == ["abc123defg"]
        assert json.loads("".join(tc.function.arguments or "" for tc in tool_calls if tc.function)) == {"city": "Seoul"}
        for name in ("fixed", "upstream"):
            r, c, calls = self._complete(
                parsers[name], tokenizer, output, request, request.tools, enable_auto_tools=True
            )
            assert r == "Need the weather tool." and not c
            assert [(tc.name, json.loads(tc.arguments)) for tc in calls] == [("get_weather", {"city": "Seoul"})]

    def test_merged_delta_keeps_first_content_token(self, tokenizer, parsers):
        """A detokenizer delta "<|end|><|begin|>assistant<|content|>Seoul" must not lose "Seoul"."""
        parser = parsers["fixed"](tokenizer, None)
        request = self._request()
        prompt = self._prompt_ids(tokenizer, "low")
        chunks = ["<|think|>", self.THINK, "<|end|><|begin|>assistant<|content|>Seoul", " is the capital."]
        ids = [tokenizer.encode(c, add_special_tokens=False) for c in chunks]
        got = []
        for i, (text, tid) in enumerate(zip(chunks, ids)):
            delta = parser.parse_delta(
                delta_text=text, delta_token_ids=tid, request=request, prompt_token_ids=prompt, finished=i == 3
            )
            if delta is not None:
                got.append((delta.reasoning or "", delta.content or ""))
        assert "".join(r for r, _ in got) == self.THINK
        assert "".join(c for _, c in got) == "Seoul is the capital."

    def test_non_streaming_matches_upstream_on_tagged_outputs(self, tokenizer, parsers):
        fixed, upstream = parsers["fixed"](tokenizer), parsers["upstream"](tokenizer)
        request = self._request()
        for text in (
            f"<|think|>{self.THINK}<|end|><|begin|>assistant<|content|>{self.ANSWER}",
            f"<|content|>{self.ANSWER}",
            "<|think|>truncated reasoning without an end",
            f"<|think|><|end|><|content|>{self.ANSWER}",
        ):
            assert fixed.parse(text, request) == upstream.parse(text, request)
